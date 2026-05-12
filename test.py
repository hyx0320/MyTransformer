import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from config import Config
from model import Transformer, create_pad_mask, create_subseq_mask
from dataset import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN, build_vocab, get_data_loader
from utils import (
    count_parameters, print_param_details, print_param_analysis,
    estimate_model_size, evaluate_model, print_evaluation_results
)

def greedy_decode(model, src, src_vocab, tgt_vocab, device, max_len=50):
    """贪婪解码：逐词生成目标句子"""
    model.eval()
    # 源句编码
    src = src.to(device)
    src_mask = create_pad_mask(src, 0).to(device)  # pad_idx=0
    enc_out = model.encoder(src, src_mask)

    # 解码器初始输入：<sos>
    tgt_indexes = [tgt_vocab[SOS_TOKEN]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_indexes], dtype=torch.long).to(device)
        tgt_pad_mask = create_pad_mask(tgt_tensor, 0).to(device)
        tgt_sub_mask = create_subseq_mask(tgt_tensor).to(device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        # 解码器输出
        dec_out = model.decoder(tgt_tensor, enc_out, src_mask, tgt_mask)
        logits = model.fc(dec_out)  # (1, seq_len, vocab_size)
        # 取最后一个位置的预测
        next_token_logits = logits[0, -1, :]
        next_token = next_token_logits.argmax(dim=-1).item()

        tgt_indexes.append(next_token)
        if next_token == tgt_vocab[EOS_TOKEN]:
            break

    return tgt_indexes


def main():
    config = Config()
    device = torch.device(config.device)
    print(f"Using device: {device}")

    # 加载测试数据
    data_dir = config.data_path
    test_src_path = os.path.join(data_dir, 'test/test2016.de')
    test_tgt_path = os.path.join(data_dir, 'test/test2016.en')

    # 构建词表（应与训练时一致）
    train_src_path = os.path.join(data_dir, 'train/train.de')
    train_tgt_path = os.path.join(data_dir, 'train/train.en')
    src_vocab = build_vocab(train_src_path, max_size=config.src_vocab_size)
    tgt_vocab = build_vocab(train_tgt_path, max_size=config.tgt_vocab_size)

    # 反向词表（索引 -> 单词）
    src_idx2word = {idx: word for word, idx in src_vocab.items()}
    tgt_idx2word = {idx: word for word, idx in tgt_vocab.items()}

    # 初始化模型并加载训练好的权重
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

    # 加载最佳模型
    checkpoint_path = 'checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with val loss {checkpoint['loss']:.4f}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}, using untrained model.")

    # ===== 1. 打印详细的参数分析报告 =====
    print("\n" + "=" * 60)
    print("  模型参数详细分析")
    print("=" * 60)
    print_param_analysis(model)

    # ===== 2. 打印分层参数详情 =====
    print("\n" + "=" * 60)
    print("  分层参数详情")
    print("=" * 60)
    print_param_details(model)

    # ===== 3. 模型存储大小估算 =====
    total_params, _ = count_parameters(model)
    model_size_mb = estimate_model_size(total_params)
    print(f"\n[存储] 模型存储大小估算 (float32): {model_size_mb:.2f} MB")

    # ===== 4. 读取测试集句子并进行预测 =====
    print("\n" + "=" * 60)
    print("  预测样例展示")
    print("=" * 60)
    with open(test_src_path, 'r', encoding='utf-8') as f_src, open(test_tgt_path, 'r', encoding='utf-8') as f_tgt:
        src_lines = f_src.readlines()
        tgt_lines = f_tgt.readlines()

    num_samples = min(10, len(src_lines))
    for i in range(num_samples):
        src_words = src_lines[i].strip().split()
        tgt_words = tgt_lines[i].strip().split()

        # 源句转索引
        src_ids = [src_vocab.get(SOS_TOKEN, 1)] + \
                  [src_vocab.get(w, src_vocab[UNK_TOKEN]) for w in src_words] + \
                  [src_vocab.get(EOS_TOKEN, 2)]
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)

        # 解码
        decoded_ids = greedy_decode(model, src_tensor, src_vocab, tgt_vocab, device, max_len=50)
        # 转换为单词（跳过 <sos> 和 <eos>）
        decoded_words = [tgt_idx2word[idx] for idx in decoded_ids
                         if idx not in [tgt_vocab[SOS_TOKEN], tgt_vocab[EOS_TOKEN], tgt_vocab[PAD_TOKEN]]]

        print(f"\n  样例 {i+1}:")
        print(f"  源语言 (DE): {' '.join(src_words)}")
        print(f"  参考译文 (EN): {' '.join(tgt_words)}")
        print(f"  模型预测: {' '.join(decoded_words)}")
        print(f"  {'─' * 50}")

    # ===== 5. 在测试集上做完整评估（准确率、召回率、F1、混淆矩阵）=====
    print("\n" + "=" * 60)
    print("  测试集完整评估")
    print("=" * 60)

    # 创建测试集 DataLoader
    test_loader = get_data_loader(config, test_src_path, test_tgt_path, src_vocab, tgt_vocab, shuffle=False)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)

    eval_results = evaluate_model(
        model, test_loader, criterion, device, pad_idx=config.pad_idx,
        save_confusion_path='figures/confusion_matrix_test.png'
    )
    print_evaluation_results(eval_results, title="测试集评估结果")

    # ===== 6. 最终统计 =====
    total, trainable = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"  最终统计")
    print(f"{'='*50}")
    print(f"  模型总参数量: {total:,}")
    print(f"  可训练参数量: {trainable:,}")
    print(f"  模型存储大小: {model_size_mb:.2f} MB")
    print(f"  测试 Loss: {eval_results['loss']:.4f}")
    print(f"  Token 准确率: {eval_results['token_accuracy']:.4f}")
    print(f"  句子准确率: {eval_results['sentence_accuracy']:.4f}")
    print(f"  F1 分数: {eval_results['f1_score']:.4f}")
    print(f"  混淆矩阵已保存至: figures/confusion_matrix_test.png")
    print(f"{'='*50}")

    # ===== 7. 保存测试结果到 JSON =====
    test_results = {
        'test_loss': eval_results['loss'],
        'token_accuracy': eval_results['token_accuracy'],
        'sentence_accuracy': eval_results['sentence_accuracy'],
        'precision': eval_results['precision'],
        'recall': eval_results['recall'],
        'f1_score': eval_results['f1_score'],
        'correct_tokens': eval_results['correct_tokens'],
        'total_tokens': eval_results['total_tokens'],
        'correct_sentences': eval_results['correct_sentences'],
        'total_sentences': eval_results['total_sentences'],
        'total_params': total,
        'model_size_mb': model_size_mb,
    }
    with open('figures/test_results.json', 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    print("[OK] 测试结果已保存到 figures/test_results.json")


if __name__ == "__main__":
    main()