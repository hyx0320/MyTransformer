import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import time
from config import Config
from model import Transformer, create_pad_mask, create_subseq_mask
from dataset import build_vocab, get_data_loader
from utils import count_parameters, plot_loss, evaluate_model, print_evaluation_results

def train_epoch(model, dataloader, optimizer, criterion, device, pad_idx):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        # Decoder 输入：去掉最后一个 token（预测目标）
        tgt_input = tgt[:, :-1]
        # Decoder 输出（真实标签）：去掉第一个 token（<sos>）
        tgt_output = tgt[:, 1:]

        # 构造 mask
        src_mask = create_pad_mask(src, pad_idx).to(device)
        tgt_pad_mask = create_pad_mask(tgt_input, pad_idx).to(device)
        tgt_sub_mask = create_subseq_mask(tgt_input).to(device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask  # 合并 padding mask 和 subsequent mask

        optimizer.zero_grad()

        # 前向传播
        logits = model(src, tgt_input, src_mask, tgt_mask)  # (batch, tgt_len, vocab_size)
        # 计算 loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, pad_idx):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = create_pad_mask(src, pad_idx).to(device)
            tgt_pad_mask = create_pad_mask(tgt_input, pad_idx).to(device)
            tgt_sub_mask = create_subseq_mask(tgt_input).to(device)
            tgt_mask = tgt_pad_mask & tgt_sub_mask

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    config = Config()
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # 路径准备
    data_dir = config.data_path
    train_src_path = os.path.join(data_dir, 'train/train.de')
    train_tgt_path = os.path.join(data_dir, 'train/train.en')
    val_src_path = os.path.join(data_dir, 'valid/val.de')
    val_tgt_path = os.path.join(data_dir, 'valid/val.en')

    # 构建或加载词表
    # dataset.py 中的 build_vocab 函数会根据训练数据构建词表，保留最常见的 max_size 个词，并添加特殊标记
    print("Building vocabularies...")
    src_vocab = build_vocab(train_src_path, max_size=config.src_vocab_size)
    tgt_vocab = build_vocab(train_tgt_path, max_size=config.tgt_vocab_size)

    # 实际词表大小可能小于配置值
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")

    # 创建 DataLoader
    train_loader = get_data_loader(config, train_src_path, train_tgt_path, src_vocab, tgt_vocab, shuffle=True)
    val_loader = get_data_loader(config, val_src_path, val_tgt_path, src_vocab, tgt_vocab, shuffle=False)

    # 初始化模型
    # 初始化 Transformer 模型
    # 模型参数全部从配置文件读取，保证结构统一、易于调整
    model = Transformer(
        src_vocab=src_vocab_size,    # 源语言（输入）词汇表大小
        tgt_vocab=tgt_vocab_size,    # 目标语言（输出）词汇表大小
        d_model=config.d_model,     # 模型维度（词向量/隐藏层维度，如 512）
        n_layers=config.n_layers,   # Encoder / Decoder 层数（如 6 层）
        n_heads=config.n_heads,     # 多头注意力的头数（如 8 头）
        d_ff=config.d_ff,           # 前馈网络中间层维度（如 2048）
        max_len=config.max_len,     # 支持的最大序列长度（位置编码上限）
        dropout=config.dropout,     # Dropout 概率，防止过拟合
        pad_idx=config.pad_idx      # 填充符<PAD>的索引，用于生成掩码
    ).to(device)  # 将模型加载到指定设备（CPU/GPU）

    # 参数统计
    # 统计模型总参数量 & 可训练参数量
    total, trainable = count_parameters(model)
    # 打印参数量（格式化带千分位逗号，方便阅读）
    print(f"模型总参数量: {total:,} (可训练参数量: {trainable:,})")

    # 损失函数（交叉熵损失，忽略填充标记）
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)

    # 优化器，使用 Adam，论文中推荐的参数设置（beta1=0.9, beta2=0.98, eps=1e-9）
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)

    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    epoch_times = []
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # 显存记录
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    total_start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config.pad_idx)
        val_loss = validate(model, val_loader, criterion, device, config.pad_idx)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start_time
        epoch_times.append(elapsed)

        # 显存占用
        mem_info = ""
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            cached = torch.cuda.memory_reserved() / (1024 * 1024)
            mem_info = f" | GPU Mem: {allocated:.0f}MB (cached: {cached:.0f}MB)"

        print(f"Epoch {epoch:2d}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s{mem_info}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'train_loss': train_loss,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'config': {
                    'd_model': config.d_model,
                    'n_layers': config.n_layers,
                    'n_heads': config.n_heads,
                    'd_ff': config.d_ff,
                    'dropout': config.dropout,
                    'batch_size': config.batch_size,
                    'lr': config.lr,
                    'epochs': config.epochs,
                }
            }, 'checkpoints/best_model.pth')
            print("  -> Saved best model")

    total_time = time.time() - total_start_time

    # 保存最后的模型
    torch.save(model.state_dict(), 'checkpoints/last_model.pth')

    # 绘制 loss 曲线
    plot_loss(train_losses, val_losses, save_path='figures/loss_curve.png')
    print(f"Training finished. Total time: {total_time:.1f}s")
    print(f"Average epoch time: {total_time/config.epochs:.1f}s")
    print(f"Loss curve saved to 'figures/loss_curve.png'")

    # 打印训练总结
    print(f"\n{'='*50}")
    print(f"训练总结")
    print(f"{'='*50}")
    print(f"总训练时间: {total_time:.1f}s")
    print(f"平均每轮时间: {total_time/config.epochs:.1f}s")
    print(f"初始训练 Loss: {train_losses[0]:.4f}")
    print(f"最终训练 Loss: {train_losses[-1]:.4f}")
    print(f"最佳验证 Loss: {best_val_loss:.4f}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"峰值显存占用: {peak_mem:.0f} MB")
    print(f"{'='*50}")

    # 在验证集上做完整评估（准确率、召回率、F1、混淆矩阵）
    print("\n在验证集上进行完整评估...")
    eval_results = evaluate_model(
        model, val_loader, criterion, device, pad_idx=config.pad_idx,
        save_confusion_path='figures/confusion_matrix_val.png'
    )
    print_evaluation_results(eval_results, title="验证集评估结果")

    # 保存训练元数据到统一文件
    training_metadata = {
        'total_time': total_time,
        'avg_epoch_time': total_time / config.epochs,
        'epoch_times': epoch_times,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'initial_train_loss': train_losses[0],
        'final_train_loss': train_losses[-1],
        'total_params': sum(p.numel() for p in model.parameters()),
        'peak_memory_mb': torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == 'cuda' else 0,
        'val_evaluation': {
            'loss': eval_results['loss'],
            'token_accuracy': eval_results['token_accuracy'],
            'sentence_accuracy': eval_results['sentence_accuracy'],
            'precision': eval_results['precision'],
            'recall': eval_results['recall'],
            'f1_score': eval_results['f1_score'],
            'correct_tokens': eval_results['correct_tokens'],
            'total_tokens': eval_results['total_tokens'],
            'correct_sentences': eval_results['correct_sentences'],
            'total_sentences': eval_results['total_sentences'],
        }
    }
    with open('figures/training_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(training_metadata, f, ensure_ascii=False, indent=2)
    print("[OK] 训练元数据已保存到 figures/training_metadata.json")


if __name__ == "__main__":
    main()