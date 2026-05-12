"""
analysis.py
============
Transformer 模型分析脚本

功能：
1. 模型参数量详细分解（总参数量、各组件参数量、逐层参数量）
2. 超参数对比实验（自动运行多组配置并记录结果）
3. 训练效果分析（Loss 曲线、收敛速度）
4. 参数量与训练时间、显存占用、模型效果之间的关系分析
5. 预测样例展示

用法：
    python analysis.py              # 运行完整分析
    python analysis.py --param-only  # 只分析参数量
    python analysis.py --train-only  # 只运行训练对比实验
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import sys
import json
import argparse
from copy import deepcopy

from config import Config, get_experiment_configs, config_from_dict
from model import Transformer, create_pad_mask, create_subseq_mask
from dataset import build_vocab, get_data_loader
from utils import (
    count_parameters, print_param_details, print_param_analysis,
    count_parameters_by_component, plot_loss, plot_param_comparison,
    estimate_model_size, evaluate_model
)


# ============================================================
# 第一部分：模型参数量详细分析
# ============================================================

def analyze_model_parameters():
    """
    对默认配置的模型进行详细的参数量分析。
    包括：总参数量、各组件参数量、逐层参数量、参数量计算公式。
    """
    print("\n" + "=" * 70)
    print("  第一部分：模型参数量详细分析")
    print("=" * 70)

    config = Config()
    device = torch.device(config.device)

    # 构建词表以获取真实词表大小
    data_dir = config.data_path
    train_src_path = os.path.join(data_dir, 'train/train.de')
    train_tgt_path = os.path.join(data_dir, 'train/train.en')
    src_vocab = build_vocab(train_src_path, max_size=config.src_vocab_size)
    tgt_vocab = build_vocab(train_tgt_path, max_size=config.tgt_vocab_size)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    print(f"\n📋 模型配置:")
    print(f"   d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
    print(f"   d_ff={config.d_ff}, dropout={config.dropout}")
    print(f"   src_vocab_size={src_vocab_size}, tgt_vocab_size={tgt_vocab_size}")

    # 创建模型
    model = Transformer(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_len=config.max_len,
        dropout=config.dropout,
        pad_idx=config.pad_idx
    ).to(device)

    # 打印详细参数分析报告
    print_param_analysis(model)

    # 理论参数量计算验证
    print("\n📐 理论参数量计算验证:")
    d_m = config.d_model
    d_f = config.d_ff
    n_l = config.n_layers
    n_h = config.n_heads
    d_k = d_m // n_h
    s_v = src_vocab_size
    t_v = tgt_vocab_size

    # Embedding
    src_emb_params = s_v * d_m
    tgt_emb_params = t_v * d_m
    emb_total = src_emb_params + tgt_emb_params
    print(f"   Embedding: {s_v}×{d_m} + {t_v}×{d_m} = {src_emb_params:,} + {tgt_emb_params:,} = {emb_total:,}")

    # 单层 Encoder
    enc_sa = 4 * d_m * d_m  # Wq + Wk + Wv + Wo (each d_model × d_model)
    enc_ff = 2 * d_m * d_f  # fc1 + fc2
    enc_ln = 2 * 2 * d_m    # norm1 + norm2 (each has gamma + beta = 2×d_model)
    enc_layer_total = enc_sa + enc_ff + enc_ln
    print(f"   单层 Encoder: SA={enc_sa:,} + FF={enc_ff:,} + LN={enc_ln:,} = {enc_layer_total:,}")
    print(f"   全部 Encoder ({n_l} 层): {enc_layer_total * n_l:,}")

    # 单层 Decoder
    dec_sa = 4 * d_m * d_m   # self-attention
    dec_ca = 4 * d_m * d_m   # cross-attention
    dec_ff = 2 * d_m * d_f
    dec_ln = 2 * 3 * d_m     # norm1 + norm2 + norm3
    dec_layer_total = dec_sa + dec_ca + dec_ff + dec_ln
    print(f"   单层 Decoder: SA={dec_sa:,} + CA={dec_ca:,} + FF={dec_ff:,} + LN={dec_ln:,} = {dec_layer_total:,}")
    print(f"   全部 Decoder ({n_l} 层): {dec_layer_total * n_l:,}")

    # 输出投影
    out_proj = d_m * t_v
    print(f"   输出投影: {d_m}×{t_v} = {out_proj:,}")

    theoretical_total = emb_total + (enc_layer_total + dec_layer_total) * n_l + out_proj
    print(f"\n   理论总参数量: {theoretical_total:,}")
    print(f"   实际总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   差异: {sum(p.numel() for p in model.parameters()) - theoretical_total:,}")

    return model, config


# ============================================================
# 第二部分：不同模型规模下的参数量变化
# ============================================================

def analyze_parameter_scaling():
    """
    分析不同模型规模下参数量的变化。
    改变 d_model、n_layers、n_heads 等，观察参数量变化趋势。
    """
    print("\n" + "=" * 70)
    print("  第二部分：不同模型规模下参数量变化分析")
    print("=" * 70)

    config = Config()
    device = torch.device(config.device)

    # 构建词表
    data_dir = config.data_path
    train_src_path = os.path.join(data_dir, 'train/train.de')
    train_tgt_path = os.path.join(data_dir, 'train/train.en')
    src_vocab = build_vocab(train_src_path, max_size=config.src_vocab_size)
    tgt_vocab = build_vocab(train_tgt_path, max_size=config.tgt_vocab_size)
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # 定义不同规模的模型配置
    model_scales = [
        {'name': 'Tiny',   'd_model': 64,   'n_layers': 2, 'n_heads': 2, 'd_ff': 256},
        {'name': 'Small',  'd_model': 128,  'n_layers': 3, 'n_heads': 4, 'd_ff': 512},
        {'name': 'Medium', 'd_model': 256,  'n_layers': 4, 'n_heads': 4, 'd_ff': 1024},
        {'name': 'Base',   'd_model': 512,  'n_layers': 6, 'n_heads': 8, 'd_ff': 2048},
        {'name': 'Large',  'd_model': 1024, 'n_layers': 6, 'n_heads': 16, 'd_ff': 4096},
    ]

    results = []
    for scale in model_scales:
        print(f"\n[图表] 模型规模: {scale['name']}")
        print(f"   d_model={scale['d_model']}, n_layers={scale['n_layers']}, "
              f"n_heads={scale['n_heads']}, d_ff={scale['d_ff']}")

        model = Transformer(
            src_vocab=src_vocab_size,
            tgt_vocab=tgt_vocab_size,
            d_model=scale['d_model'],
            n_layers=scale['n_layers'],
            n_heads=scale['n_heads'],
            d_ff=scale['d_ff'],
            max_len=config.max_len,
            dropout=config.dropout,
            pad_idx=config.pad_idx
        ).to(device)

        total, trainable = count_parameters(model)
        size_mb = estimate_model_size(total)
        stats = count_parameters_by_component(model)

        print(f"   总参数量: {total:>12,} ({total/1e6:.2f}M)")
        print(f"   模型大小: {size_mb:.2f} MB (float32)")
        print(f"   Embedding占比: {stats['embedding']['total']/total*100:.1f}%")
        print(f"   Encoder占比:   {stats['encoder']['total']/total*100:.1f}%")
        print(f"   Decoder占比:   {stats['decoder']['total']/total*100:.1f}%")
        print(f"   输出投影占比: {stats['output_projection']/total*100:.1f}%")

        results.append({
            'name': scale['name'],
            'd_model': scale['d_model'],
            'n_layers': scale['n_layers'],
            'n_heads': scale['n_heads'],
            'd_ff': scale['d_ff'],
            'total_params': total,
            'size_mb': size_mb,
            'embedding_pct': stats['embedding']['total'] / total * 100,
            'encoder_pct': stats['encoder']['total'] / total * 100,
            'decoder_pct': stats['decoder']['total'] / total * 100,
            'output_pct': stats['output_projection'] / total * 100,
        })

    # 打印汇总表
    print("\n" + "-" * 100)
    print(f"{'Scale':<10} {'d_model':<10} {'Layers':<8} {'Heads':<8} {'d_ff':<8} "
          f"{'Params':<15} {'Size(MB)':<10} {'Emb%':<8} {'Enc%':<8} {'Dec%':<8} {'Out%':<8}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<10} {r['d_model']:<10} {r['n_layers']:<8} {r['n_heads']:<8} "
              f"{r['d_ff']:<8} {r['total_params']:<15,} {r['size_mb']:<10.2f} "
              f"{r['embedding_pct']:<8.1f} {r['encoder_pct']:<8.1f} "
              f"{r['decoder_pct']:<8.1f} {r['output_pct']:<8.1f}")
    print("-" * 100)

    # 绘制参数量对比图
    config_names = [r['name'] for r in results]
    param_counts = [r['total_params'] for r in results]
    plot_param_comparison(config_names, param_counts,
                          save_path='figures/parameter_scaling.png')
    print("\n参数量对比图已保存至 'figures/parameter_scaling.png'")

    return results


# ============================================================
# 第三部分：超参数对比实验
# ============================================================

def train_and_evaluate(config_dict, src_vocab, tgt_vocab, device):
    """
    使用给定配置训练模型并返回训练结果。
    
    返回:
        dict: {
            'train_losses': [...],
            'val_losses': [...],
            'best_val_loss': float,
            'total_time': float,
            'total_params': int,
            'peak_memory_mb': float (仅 CUDA),
            'val_evaluation': dict (验证集评估结果)
        }
    """
    cfg = config_from_dict(config_dict)

    # 创建 DataLoader
    data_dir = cfg.data_path
    train_loader = get_data_loader(cfg,
        os.path.join(data_dir, 'train/train.de'),
        os.path.join(data_dir, 'train/train.en'),
        src_vocab, tgt_vocab, shuffle=True)
    val_loader = get_data_loader(cfg,
        os.path.join(data_dir, 'valid/val.de'),
        os.path.join(data_dir, 'valid/val.en'),
        src_vocab, tgt_vocab, shuffle=False)

    # 初始化模型
    model = Transformer(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
        pad_idx=cfg.pad_idx
    ).to(device)

    total_params, _ = count_parameters(model)

    criterion = nn.CrossEntropyLoss(ignore_index=cfg.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)

    # 记录显存
    peak_memory_mb = 0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        # 训练一个 epoch
        model.train()
        epoch_train_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = create_pad_mask(src, cfg.pad_idx).to(device)
            tgt_pad_mask = create_pad_mask(tgt_input, cfg.pad_idx).to(device)
            tgt_sub_mask = create_subseq_mask(tgt_input).to(device)
            tgt_mask = tgt_pad_mask & tgt_sub_mask

            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

            # 记录峰值显存
            if device.type == 'cuda':
                peak_memory_mb = max(peak_memory_mb,
                    torch.cuda.max_memory_allocated() / (1024 * 1024))

        avg_train_loss = epoch_train_loss / len(train_loader)

        # 验证
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask = create_pad_mask(src, cfg.pad_idx).to(device)
                tgt_pad_mask = create_pad_mask(tgt_input, cfg.pad_idx).to(device)
                tgt_sub_mask = create_subseq_mask(tgt_input).to(device)
                tgt_mask = tgt_pad_mask & tgt_sub_mask

                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    total_time = time.time() - start_time

    # 在验证集上做完整评估
    val_eval = evaluate_model(
        model, val_loader, criterion, device, pad_idx=cfg.pad_idx,
        save_confusion_path=None  # 不保存混淆矩阵图，避免覆盖
    )

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'total_time': total_time,
        'total_params': total_params,
        'peak_memory_mb': peak_memory_mb,
        'val_evaluation': {
            'loss': val_eval['loss'],
            'token_accuracy': val_eval['token_accuracy'],
            'sentence_accuracy': val_eval['sentence_accuracy'],
            'precision': val_eval['precision'],
            'recall': val_eval['recall'],
            'f1_score': val_eval['f1_score'],
        }
    }


def run_hyperparameter_experiments():
    """
    运行超参数对比实验。
    对每个超参数维度，保持其他参数不变，改变目标超参数。
    """
    print("\n" + "=" * 70)
    print("  第三部分：超参数对比实验")
    print("=" * 70)

    config = Config()
    device = torch.device(config.device)
    print(f"  使用设备: {device}")

    # 构建词表（所有实验共享）
    data_dir = config.data_path
    train_src_path = os.path.join(data_dir, 'train/train.de')
    train_tgt_path = os.path.join(data_dir, 'train/train.en')
    src_vocab = build_vocab(train_src_path, max_size=config.src_vocab_size)
    tgt_vocab = build_vocab(train_tgt_path, max_size=config.tgt_vocab_size)
    print(f"  源语言词表大小: {len(src_vocab)}")
    print(f"  目标语言词表大小: {len(tgt_vocab)}")

    experiments = get_experiment_configs()

    # 选择要运行的实验组
    experiment_groups = {
        'd_model': ['d_model_128', 'd_model_256', 'd_model_512'],
        'n_heads': ['n_heads_2', 'n_heads_4', 'n_heads_8'],
        'n_layers': ['n_layers_2', 'n_layers_4', 'n_layers_6'],
        'batch_size': ['batch_16', 'batch_32', 'batch_64'],
        'lr': ['lr_1e-5', 'lr_1e-4', 'lr_5e-4'],
        'dropout': ['dropout_0.0', 'dropout_0.1', 'dropout_0.3'],
        'epochs': ['epochs_5', 'epochs_10', 'epochs_20'],
    }

    all_results = {}

    for group_name, exp_keys in experiment_groups.items():
        print(f"\n{'-' * 60}")
        print(f"  [实验] 实验组: {group_name}")
        print(f"{'-' * 60}")

        group_results = []
        for key in exp_keys:
            exp_cfg = experiments[key]
            exp_name = exp_cfg['name']
            print(f"\n  运行: {exp_name} ...")

            try:
                result = train_and_evaluate(exp_cfg, src_vocab, tgt_vocab, device)
                result['name'] = exp_name
                result['config_key'] = key
                group_results.append(result)

                mem_info = f" | 峰值显存: {result['peak_memory_mb']:.0f}MB" if result['peak_memory_mb'] > 0 else ""
                print(f"    [OK] 完成 | 参数量: {result['total_params']:,} "
                      f"| 最佳验证Loss: {result['best_val_loss']:.4f} "
                      f"| 训练时间: {result['total_time']:.1f}s"
                      f"{mem_info}")

                # 绘制该实验组的 loss 曲线
                save_path = f'figures/loss_{group_name}_{key}.png'
                os.makedirs('figures', exist_ok=True)
                plot_loss(result['train_losses'], result['val_losses'], save_path=save_path)

            except Exception as e:
                print(f"    [FAIL] 失败: {e}")
                continue

        all_results[group_name] = group_results

        # 打印该组汇总
        if group_results:
            print(f"\n   [结果] {group_name} 实验结果汇总:")
            print(f"   {'配置':<25} {'参数量':<12} {'最佳Val Loss':<15} {'训练时间':<12} {'显存(MB)':<10}")
            print(f"   {'-'*74}")
            for r in group_results:
                mem_str = f"{r['peak_memory_mb']:.0f}" if r['peak_memory_mb'] > 0 else "N/A"
                print(f"   {r['name']:<25} {r['total_params']:<12,} {r['best_val_loss']:<15.4f} "
                      f"{r['total_time']:<12.1f} {mem_str:<10}")

    # 保存实验结果到 JSON
    os.makedirs('figures', exist_ok=True)
    serializable_results = {}
    for group_name, group_results in all_results.items():
        serializable_results[group_name] = [
            {k: v for k, v in r.items() if k != 'train_losses' and k != 'val_losses'}
            for r in group_results
        ]
    with open('figures/experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"\n实验结果已保存至 'figures/experiment_results.json'")

    return all_results


# ============================================================
# 第四部分：训练效果分析
# ============================================================

def analyze_training_effect():
    """
    分析模型训练效果：
    - Loss 曲线分析（收敛速度、过拟合判断）
    - 预测样例展示
    - 模型效果评估
    """
    print("\n" + "=" * 70)
    print("  第四部分：训练效果分析")
    print("=" * 70)

    config = Config()
    device = torch.device(config.device)

    # 加载词表
    data_dir = config.data_path
    train_src_path = os.path.join(data_dir, 'train/train.de')
    train_tgt_path = os.path.join(data_dir, 'train/train.en')
    src_vocab = build_vocab(train_src_path, max_size=config.src_vocab_size)
    tgt_vocab = build_vocab(train_tgt_path, max_size=config.tgt_vocab_size)

    # 反向词表
    tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}

    # 初始化模型
    model = Transformer(
        src_vocab=len(src_vocab),
        tgt_vocab=len(tgt_vocab),
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        max_len=config.max_len,
        dropout=config.dropout,
        pad_idx=config.pad_idx
    ).to(device)

    # 加载训练好的模型
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n[OK] 已加载训练好的模型 (Epoch {checkpoint['epoch']}, Val Loss: {checkpoint['loss']:.4f})")
    else:
        print(f"\n⚠️  未找到训练好的模型 ({checkpoint_path})，使用未训练模型进行演示")
        print("   请先运行 train.py 训练模型")

    # 读取测试集
    test_src_path = os.path.join(data_dir, 'test/test2016.de')
    test_tgt_path = os.path.join(data_dir, 'test/test2016.en')

    with open(test_src_path, 'r', encoding='utf-8') as f:
        test_src_lines = f.readlines()
    with open(test_tgt_path, 'r', encoding='utf-8') as f:
        test_tgt_lines = f.readlines()

    # 预测函数
    def greedy_decode(model, src_tensor, max_len=50):
        model.eval()
        src_tensor = src_tensor.to(device)
        src_mask = create_pad_mask(src_tensor, config.pad_idx).to(device)
        enc_out = model.encoder(src_tensor, src_mask)

        tgt_indexes = [tgt_vocab['<sos>']]
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_indexes], dtype=torch.long).to(device)
            tgt_pad_mask = create_pad_mask(tgt_tensor, config.pad_idx).to(device)
            tgt_sub_mask = create_subseq_mask(tgt_tensor).to(device)
            tgt_mask = tgt_pad_mask & tgt_sub_mask

            dec_out = model.decoder(tgt_tensor, enc_out, src_mask, tgt_mask)
            logits = model.fc(dec_out)
            next_token = logits[0, -1, :].argmax(dim=-1).item()

            tgt_indexes.append(next_token)
            if next_token == tgt_vocab['<eos>']:
                break

        return tgt_indexes

    # 展示预测样例
    print(f"\n[样例] 预测样例展示 (共 {min(10, len(test_src_lines))} 个样例):")
    print("=" * 70)

    num_samples = min(10, len(test_src_lines))
    for i in range(num_samples):
        src_words = test_src_lines[i].strip().split()
        tgt_words = test_tgt_lines[i].strip().split()

        # 源句转索引
        src_ids = [src_vocab.get('<sos>', 1)] + \
                  [src_vocab.get(w, src_vocab['<unk>']) for w in src_words] + \
                  [src_vocab.get('<eos>', 2)]
        src_tensor = torch.tensor([src_ids], dtype=torch.long)

        # 解码
        decoded_ids = greedy_decode(model, src_tensor, max_len=50)
        decoded_words = [tgt_idx2word.get(idx, '<unk>') for idx in decoded_ids
                         if idx not in [tgt_vocab.get('<sos>'), tgt_vocab.get('<eos>'),
                                        tgt_vocab.get('<pad>')]]

        print(f"\n  样例 {i+1}:")
        print(f"  源语言 (DE): {' '.join(src_words)}")
        print(f"  参考译文 (EN): {' '.join(tgt_words)}")
        print(f"  模型预测: {' '.join(decoded_words)}")
        print(f"  {'-' * 50}")

    # 训练效果分析总结
    print(f"\n[分析] 训练效果分析总结:")
    print(f"  {'=' * 50}")
    print(f"  1. 模型已训练 {checkpoint.get('epoch', 'N/A')} 个 epoch")
    print(f"  2. 最佳验证 Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  3. 参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  4. 模型大小: {estimate_model_size(sum(p.numel() for p in model.parameters())):.2f} MB")
    print(f"  {'=' * 50}")


# ============================================================
# 第五部分：参数量与训练时间/显存/效果关系分析
# ============================================================

def analyze_param_relationships():
    """
    分析参数量与训练时间、显存占用、模型效果之间的关系。
    使用不同规模的模型配置进行对比。
    """
    print("\n" + "=" * 70)
    print("  第五部分：参数量与训练时间/显存/效果关系分析")
    print("=" * 70)

    config = Config()
    device = torch.device(config.device)
    print(f"  使用设备: {device}")

    # 构建词表
    data_dir = config.data_path
    train_src_path = os.path.join(data_dir, 'train/train.de')
    train_tgt_path = os.path.join(data_dir, 'train/train.en')
    src_vocab = build_vocab(train_src_path, max_size=config.src_vocab_size)
    tgt_vocab = build_vocab(train_tgt_path, max_size=config.tgt_vocab_size)

    # 使用不同 d_model 的配置（快速实验）
    test_configs = [
        {'name': 'd_model=128', 'd_model': 128, 'd_ff': 512, 'n_heads': 4,
         'n_layers': 3, 'dropout': 0.1, 'batch_size': 32, 'lr': 1e-4,
         'epochs': 5, 'max_len': 5000, 'pad_idx': 0, 'data_path': config.data_path,
         'src_vocab_size': config.src_vocab_size, 'tgt_vocab_size': config.tgt_vocab_size,
         'device': str(device)},
        {'name': 'd_model=256', 'd_model': 256, 'd_ff': 1024, 'n_heads': 4,
         'n_layers': 3, 'dropout': 0.1, 'batch_size': 32, 'lr': 1e-4,
         'epochs': 5, 'max_len': 5000, 'pad_idx': 0, 'data_path': config.data_path,
         'src_vocab_size': config.src_vocab_size, 'tgt_vocab_size': config.tgt_vocab_size,
         'device': str(device)},
        {'name': 'd_model=512', 'd_model': 512, 'd_ff': 2048, 'n_heads': 8,
         'n_layers': 3, 'dropout': 0.1, 'batch_size': 32, 'lr': 1e-4,
         'epochs': 5, 'max_len': 5000, 'pad_idx': 0, 'data_path': config.data_path,
         'src_vocab_size': config.src_vocab_size, 'tgt_vocab_size': config.tgt_vocab_size,
         'device': str(device)},
    ]

    results = []
    for tc in test_configs:
        print(f"\n  运行: {tc['name']} ...")
        try:
            result = train_and_evaluate(tc, src_vocab, tgt_vocab, device)
            results.append({
                'name': tc['name'],
                'total_params': result['total_params'],
                'best_val_loss': result['best_val_loss'],
                'total_time': result['total_time'],
                'peak_memory_mb': result['peak_memory_mb'],
                'params_per_sec': result['total_params'] / result['total_time'] if result['total_time'] > 0 else 0,
            })
            print(f"    [OK] 参数量: {result['total_params']:,} | "
                  f"时间: {result['total_time']:.1f}s | "
                  f"Loss: {result['best_val_loss']:.4f}" +
                  (f" | 显存: {result['peak_memory_mb']:.0f}MB" if result['peak_memory_mb'] > 0 else ""))
        except Exception as e:
            print(f"    [FAIL] 失败: {e}")

    # 打印关系分析
    if results:
        print(f"\n{'=' * 70}")
        print(f"  参数量与训练时间/显存/效果关系")
        print(f"{'=' * 70}")
        print(f"{'配置':<20} {'参数量':<12} {'训练时间(s)':<12} {'显存(MB)':<10} {'Val Loss':<10}")
        print(f"{'-' * 64}")
        for r in results:
            mem_str = f"{r['peak_memory_mb']:.0f}" if r['peak_memory_mb'] > 0 else "N/A"
            print(f"{r['name']:<20} {r['total_params']:<12,} {r['total_time']:<12.1f} "
                  f"{mem_str:<10} {r['best_val_loss']:<10.4f}")

        # 分析结论
        print(f"\n[分析] 分析结论:")
        if len(results) >= 2:
            params_list = [r['total_params'] for r in results]
            time_list = [r['total_time'] for r in results]
            loss_list = [r['best_val_loss'] for r in results]

            # 参数量 vs 时间
            if params_list[-1] > params_list[0]:
                ratio = (time_list[-1] / time_list[0]) / (params_list[-1] / params_list[0])
                print(f"  • 参数量与训练时间关系: 参数量增加 {params_list[-1]/params_list[0]:.1f} 倍, "
                      f"训练时间增加 {time_list[-1]/time_list[0]:.1f} 倍 "
                      f"(时间/参数比: {ratio:.2f})")

            # 参数量 vs Loss
            print(f"  • 参数量与模型效果关系: 参数量越大，验证 Loss 通常越低（模型容量更大）")
            for i in range(len(results)):
                print(f"    - {results[i]['name']}: {results[i]['total_params']:,} params -> Loss {results[i]['best_val_loss']:.4f}")

            # 显存分析
            mem_values = [r['peak_memory_mb'] for r in results if r['peak_memory_mb'] > 0]
            if mem_values:
                print(f"  • 显存占用: 模型越大，显存占用越高")
                for r in results:
                    if r['peak_memory_mb'] > 0:
                        print(f"    - {r['name']}: {r['peak_memory_mb']:.0f} MB")

        # 保存关系图
        names = [r['name'] for r in results]
        params = [r['total_params'] for r in results]
        losses = [r['best_val_loss'] for r in results]
        plot_param_comparison(names, params, metric_name='Val Loss', metric_values=losses,
                              save_path='figures/param_relationship.png')
        print(f"\n  关系图已保存至 'figures/param_relationship.png'")

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Transformer 模型分析工具')
    parser.add_argument('--param-only', action='store_true', help='只分析参数量')
    parser.add_argument('--train-only', action='store_true', help='只运行训练对比实验')
    parser.add_argument('--test-only', action='store_true', help='只运行测试分析')
    parser.add_argument('--all', action='store_true', help='运行所有分析')
    args = parser.parse_args()

    os.makedirs('figures', exist_ok=True)

    # 如果没有指定参数，运行所有分析
    run_all = args.all or not (args.param_only or args.train_only or args.test_only)

    if run_all or args.param_only:
        # 第一部分：参数量详细分析
        analyze_model_parameters()

        # 第二部分：不同规模参数量变化
        analyze_parameter_scaling()

    if run_all or args.train_only:
        # 第三部分：超参数对比实验
        run_hyperparameter_experiments()

        # 第五部分：参数量与训练时间/显存/效果关系
        analyze_param_relationships()

    if run_all or args.test_only:
        # 第四部分：训练效果分析
        analyze_training_effect()

    print("\n" + "=" * 70)
    print("  分析完成！所有结果已保存至 figures/ 目录")
    print("=" * 70)


if __name__ == "__main__":
    main()