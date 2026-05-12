import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from collections import Counter

def count_parameters(model):
    """统计模型参数总量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def print_param_details(model):
    """分层打印参数数量"""
    print("=" * 60)
    print(f"{'Layer':<30} {'Parameters':<15}")
    print("=" * 60)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<30} {param.numel():<15,}")
    total, trainable = count_parameters(model)
    print("=" * 60)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print("=" * 60)

def count_parameters_by_component(model):
    """
    按组件详细统计参数量，返回结构化字典
    返回:
        {
            'embedding': {'src_embedding': N, 'tgt_embedding': N, 'total': N},
            'positional_encoding': 0,
            'encoder': {'total': N, 'per_layer': [...], 'layers_total': N},
            'decoder': {'total': N, 'per_layer': [...], 'layers_total': N},
            'output_projection': N,
            'total': N
        }
    """
    stats = {
        'embedding': {'src_embedding': 0, 'tgt_embedding': 0, 'total': 0},
        'positional_encoding': 0,
        'encoder': {'total': 0, 'per_layer': [], 'layers_total': 0},
        'decoder': {'total': 0, 'per_layer': [], 'layers_total': 0},
        'output_projection': 0,
        'total': 0
    }

    for name, param in model.named_parameters():
        numel = param.numel()
        if 'encoder.emb' in name:
            stats['embedding']['src_embedding'] += numel
        elif 'decoder.emb' in name:
            stats['embedding']['tgt_embedding'] += numel
        elif 'encoder.layers' in name:
            stats['encoder']['total'] += numel
        elif 'decoder.layers' in name:
            stats['decoder']['total'] += numel
        elif 'fc' in name and 'encoder' not in name and 'decoder' not in name:
            stats['output_projection'] += numel

    stats['embedding']['total'] = stats['embedding']['src_embedding'] + stats['embedding']['tgt_embedding']
    stats['total'] = (stats['embedding']['total'] + stats['positional_encoding'] +
                      stats['encoder']['total'] + stats['decoder']['total'] +
                      stats['output_projection'])

    # 逐层详细统计 Encoder
    for layer_idx in range(len(model.encoder.layers)):
        layer_stats = {
            'self_attention': {'wq': 0, 'wk': 0, 'wv': 0, 'fc': 0, 'total': 0},
            'feed_forward': {'fc1': 0, 'fc2': 0, 'total': 0},
            'layer_norm': {'norm1': 0, 'norm2': 0, 'total': 0},
        }
        enc_layer = model.encoder.layers[layer_idx]
        layer_stats['self_attention']['wq'] = sum(p.numel() for p in enc_layer.attn.wq.parameters())
        layer_stats['self_attention']['wk'] = sum(p.numel() for p in enc_layer.attn.wk.parameters())
        layer_stats['self_attention']['wv'] = sum(p.numel() for p in enc_layer.attn.wv.parameters())
        layer_stats['self_attention']['fc'] = sum(p.numel() for p in enc_layer.attn.fc.parameters())
        layer_stats['self_attention']['total'] = sum(layer_stats['self_attention'].values())
        layer_stats['feed_forward']['fc1'] = sum(p.numel() for p in enc_layer.ff.fc1.parameters())
        layer_stats['feed_forward']['fc2'] = sum(p.numel() for p in enc_layer.ff.fc2.parameters())
        layer_stats['feed_forward']['total'] = sum(layer_stats['feed_forward'].values())
        layer_stats['layer_norm']['norm1'] = sum(p.numel() for p in enc_layer.norm1.parameters())
        layer_stats['layer_norm']['norm2'] = sum(p.numel() for p in enc_layer.norm2.parameters())
        layer_stats['layer_norm']['total'] = sum(layer_stats['layer_norm'].values())
        stats['encoder']['per_layer'].append(layer_stats)

    stats['encoder']['layers_total'] = sum(
        sum(v['total'] for v in layer.values()) for layer in stats['encoder']['per_layer']
    )

    # 逐层详细统计 Decoder
    for layer_idx in range(len(model.decoder.layers)):
        layer_stats = {
            'self_attention': {'wq': 0, 'wk': 0, 'wv': 0, 'fc': 0, 'total': 0},
            'cross_attention': {'wq': 0, 'wk': 0, 'wv': 0, 'fc': 0, 'total': 0},
            'feed_forward': {'fc1': 0, 'fc2': 0, 'total': 0},
            'layer_norm': {'norm1': 0, 'norm2': 0, 'norm3': 0, 'total': 0},
        }
        dec_layer = model.decoder.layers[layer_idx]
        layer_stats['self_attention']['wq'] = sum(p.numel() for p in dec_layer.self_attn.wq.parameters())
        layer_stats['self_attention']['wk'] = sum(p.numel() for p in dec_layer.self_attn.wk.parameters())
        layer_stats['self_attention']['wv'] = sum(p.numel() for p in dec_layer.self_attn.wv.parameters())
        layer_stats['self_attention']['fc'] = sum(p.numel() for p in dec_layer.self_attn.fc.parameters())
        layer_stats['self_attention']['total'] = sum(layer_stats['self_attention'].values())
        layer_stats['cross_attention']['wq'] = sum(p.numel() for p in dec_layer.cross_attn.wq.parameters())
        layer_stats['cross_attention']['wk'] = sum(p.numel() for p in dec_layer.cross_attn.wk.parameters())
        layer_stats['cross_attention']['wv'] = sum(p.numel() for p in dec_layer.cross_attn.wv.parameters())
        layer_stats['cross_attention']['fc'] = sum(p.numel() for p in dec_layer.cross_attn.fc.parameters())
        layer_stats['cross_attention']['total'] = sum(layer_stats['cross_attention'].values())
        layer_stats['feed_forward']['fc1'] = sum(p.numel() for p in dec_layer.ff.fc1.parameters())
        layer_stats['feed_forward']['fc2'] = sum(p.numel() for p in dec_layer.ff.fc2.parameters())
        layer_stats['feed_forward']['total'] = sum(layer_stats['feed_forward'].values())
        layer_stats['layer_norm']['norm1'] = sum(p.numel() for p in dec_layer.norm1.parameters())
        layer_stats['layer_norm']['norm2'] = sum(p.numel() for p in dec_layer.norm2.parameters())
        layer_stats['layer_norm']['norm3'] = sum(p.numel() for p in dec_layer.norm3.parameters())
        layer_stats['layer_norm']['total'] = sum(layer_stats['layer_norm'].values())
        stats['decoder']['per_layer'].append(layer_stats)

    stats['decoder']['layers_total'] = sum(
        sum(v['total'] for v in layer.values()) for layer in stats['decoder']['per_layer']
    )

    return stats


def print_param_analysis(model):
    """打印详细的参数分析报告"""
    stats = count_parameters_by_component(model)
    total, trainable = count_parameters(model)

    print("\n" + "=" * 70)
    print("              Transformer 模型参数详细分析报告")
    print("=" * 70)

    # 1. 整体参数统计
    print(f"\n[统计] 1. 整体参数统计")
    print(f"   {'总参数量':<20} {total:>12,}")
    print(f"   {'可训练参数量':<20} {trainable:>12,}")
    print(f"   {'模型存储大小 (float32)':<20} {estimate_model_size(total):>10.2f} MB")

    # 2. 各组件参数量
    print(f"\n[组件] 2. 各组件参数量")
    print(f"   {'组件':<30} {'参数量':<15} {'占比':<10}")
    print(f"   {'-'*55}")
    components = [
        ('Embedding (源+目标)', stats['embedding']['total']),
        ('位置编码 (PositionalEncoding)', stats['positional_encoding']),
        ('Encoder 堆栈', stats['encoder']['total']),
        ('Decoder 堆栈', stats['decoder']['total']),
        ('输出投影层 (fc)', stats['output_projection']),
    ]
    for name, count in components:
        pct = count / total * 100 if total > 0 else 0
        print(f"   {name:<30} {count:>12,}  {pct:>7.2f}%")

    # 3. Embedding 层详情
    print(f"\n[嵌入] 3. Embedding 层详情")
    print(f"   源语言嵌入: {stats['embedding']['src_embedding']:>12,}")
    print(f"   目标语言嵌入: {stats['embedding']['tgt_embedding']:>12,}")

    # 4. Encoder 逐层分析
    n_enc_layers = len(stats['encoder']['per_layer'])
    print(f"\n[编码器] 4. Encoder 逐层参数分析 (共 {n_enc_layers} 层)")
    for i, layer in enumerate(stats['encoder']['per_layer']):
        print(f"   --- Layer {i+1} ---")
        sa = layer['self_attention']
        print(f"     自注意力 (Multi-Head Attention):")
        print(f"       Wq: {sa['wq']:>10,} | Wk: {sa['wk']:>10,} | Wv: {sa['wv']:>10,} | Fc: {sa['fc']:>10,}")
        print(f"       => 小计: {sa['total']:>10,}")
        ff = layer['feed_forward']
        print(f"     前馈网络 (Feed-Forward):")
        print(f"       Fc1: {ff['fc1']:>10,} | Fc2: {ff['fc2']:>10,}")
        print(f"       => 小计: {ff['total']:>10,}")
        ln = layer['layer_norm']
        print(f"     层归一化 (LayerNorm): {ln['total']:>10,}")
        layer_total = sa['total'] + ff['total'] + ln['total']
        print(f"     -- 单层合计: {layer_total:>12,}")

    # 5. Decoder 逐层分析
    n_dec_layers = len(stats['decoder']['per_layer'])
    print(f"\n[解码器] 5. Decoder 逐层参数分析 (共 {n_dec_layers} 层)")
    for i, layer in enumerate(stats['decoder']['per_layer']):
        print(f"   --- Layer {i+1} ---")
        sa = layer['self_attention']
        print(f"     自注意力 (Self-Attention):")
        print(f"       Wq: {sa['wq']:>10,} | Wk: {sa['wk']:>10,} | Wv: {sa['wv']:>10,} | Fc: {sa['fc']:>10,}")
        print(f"       => 小计: {sa['total']:>10,}")
        ca = layer['cross_attention']
        print(f"     交叉注意力 (Cross-Attention):")
        print(f"       Wq: {ca['wq']:>10,} | Wk: {ca['wk']:>10,} | Wv: {ca['wv']:>10,} | Fc: {ca['fc']:>10,}")
        print(f"       => 小计: {ca['total']:>10,}")
        ff = layer['feed_forward']
        print(f"     前馈网络 (Feed-Forward):")
        print(f"       Fc1: {ff['fc1']:>10,} | Fc2: {ff['fc2']:>10,}")
        print(f"       => 小计: {ff['total']:>10,}")
        ln = layer['layer_norm']
        print(f"     层归一化 (LayerNorm): {ln['total']:>10,}")
        layer_total = sa['total'] + ca['total'] + ff['total'] + ln['total']
        print(f"     -- 单层合计: {layer_total:>12,}")

    # 6. 参数量公式
    print(f"\n📐 6. 参数量计算公式")
    print(f"   Embedding: vocab_size × d_model")
    print(f"   Multi-Head Attention: 4 × d_model² (Wq + Wk + Wv + Wo)")
    print(f"   Feed-Forward: 2 × d_model × d_ff (fc1 + fc2)")
    print(f"   LayerNorm: 2 × d_model (gamma + beta)")
    print(f"   输出投影: d_model × tgt_vocab_size")

    print("=" * 70 + "\n")


def plot_loss(train_losses, val_losses, save_path="loss_curve.png"):
    """绘制训练和验证 loss 曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training & Validation Loss Curve', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_param_comparison(configs, param_counts, metric_name='BLEU', metric_values=None, save_path='figures/param_analysis.png'):
    """
    绘制不同配置下的参数量对比图
    configs: 配置名称列表
    param_counts: 参数量列表
    metric_values: 对应指标值列表（可选）
    """
    import os
    os.makedirs('figures', exist_ok=True)

    fig, axes = plt.subplots(1, 2 if metric_values else 1, figsize=(12, 5))

    if metric_values:
        ax1, ax2 = axes
    else:
        ax1 = axes

    # 左图：参数量柱状图
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(configs)))
    bars = ax1.bar(range(len(configs)), [p / 1e6 for p in param_counts], color=colors)
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=30, ha='right', fontsize=9)
    ax1.set_ylabel('Parameters (Millions)', fontsize=11)
    ax1.set_title('Model Parameters by Configuration', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, param_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val/1e6:.2f}M', ha='center', va='bottom', fontsize=8)

    if metric_values:
        ax2.scatter([p / 1e6 for p in param_counts], metric_values, c=colors, s=80)
        ax2.set_xlabel('Parameters (Millions)', fontsize=11)
        ax2.set_ylabel(metric_name, fontsize=11)
        ax2.set_title(f'Parameters vs {metric_name}', fontsize=12)
        ax2.grid(True, alpha=0.3)
        for i, (p, m) in enumerate(zip(param_counts, metric_values)):
            ax2.annotate(configs[i], (p / 1e6, m), fontsize=8,
                        xytext=(5, 5), textcoords='offset points')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def estimate_model_size(param_count):
    """估算模型存储大小 (float32: 4 bytes per parameter)"""
    size_bytes = param_count * 4
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


# ============================================================
# 评估指标：准确率、精确率、召回率、F1、混淆矩阵
# ============================================================

def compute_token_accuracy(logits, targets, pad_idx=0):
    """
    计算 token 级别的准确率（忽略 padding 位置）
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len)
    """
    predictions = logits.argmax(dim=-1)  # (batch, seq_len)
    mask = targets != pad_idx
    correct = (predictions == targets) & mask
    total = mask.sum().item()
    correct_count = correct.sum().item()
    accuracy = correct_count / total if total > 0 else 0.0
    return accuracy, correct_count, total


def compute_sentence_accuracy(predictions, targets, pad_idx=0):
    """
    计算句子级别的准确率（整句完全匹配）
    predictions: (batch, seq_len)
    targets: (batch, seq_len)
    """
    mask = targets != pad_idx
    # 对每个句子，检查所有非 pad 位置是否完全匹配
    match = ((predictions == targets) | ~mask).all(dim=1)
    total = targets.size(0)
    correct = match.sum().item()
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def compute_precision_recall_f1(logits, targets, pad_idx=0, num_classes=None):
    """
    计算宏观平均精确率、召回率、F1 分数
    对每个类别分别计算，然后取平均（macro-averaging）
    """
    predictions = logits.argmax(dim=-1)  # (batch, seq_len)
    mask = targets != pad_idx

    pred_flat = predictions[mask].cpu().numpy()
    tgt_flat = targets[mask].cpu().numpy()

    if num_classes is None:
        num_classes = max(pred_flat.max(), tgt_flat.max()) + 1

    # 每个类别的 TP, FP, FN
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for c in range(num_classes):
        tp[c] = np.sum((pred_flat == c) & (tgt_flat == c))
        fp[c] = np.sum((pred_flat == c) & (tgt_flat != c))
        fn[c] = np.sum((pred_flat != c) & (tgt_flat == c))

    # 宏观平均
    precision = np.mean([tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0 for c in range(num_classes)])
    recall = np.mean([tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0 for c in range(num_classes)])
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def compute_confusion_matrix(logits, targets, pad_idx=0, num_classes=None, top_k=20):
    """
    计算混淆矩阵（仅限 Top-K 高频类别，其余归为 'other'）
    返回: (top_k+1, top_k+1) 的 numpy 数组，最后一行/列是 'other'
    """
    predictions = logits.argmax(dim=-1)
    mask = targets != pad_idx

    pred_flat = predictions[mask].cpu().numpy()
    tgt_flat = targets[mask].cpu().numpy()

    # 统计目标 token 出现频率，取 top_k 个
    from collections import Counter
    tgt_counter = Counter(tgt_flat.tolist())
    top_classes = [cls for cls, _ in tgt_counter.most_common(top_k)]

    # 构建映射：原类别 -> 新索引（0..top_k-1），其余 -> top_k
    cls_to_idx = {cls: i for i, cls in enumerate(top_classes)}
    other_idx = top_k

    size = top_k + 1  # top_k 类 + 1 个 other
    conf_matrix = np.zeros((size, size), dtype=np.int64)

    for p, t in zip(pred_flat, tgt_flat):
        t_idx = cls_to_idx.get(t, other_idx)
        p_idx = cls_to_idx.get(p, other_idx)
        conf_matrix[t_idx, p_idx] += 1

    return conf_matrix, top_classes


def plot_confusion_matrix(conf_matrix, class_names=None, save_path='figures/confusion_matrix.png',
                          title='Confusion Matrix'):
    """
    绘制混淆矩阵热力图（已预先限制类别数）
    conf_matrix: (N, N) numpy array, N = top_k + 1 (最后一个是 'other')
    class_names: 长度为 N 的类别名称列表
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    plt.figure(figsize=(12, 10))
    # 归一化显示
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    norm_conf = np.divide(conf_matrix, row_sums, out=np.zeros_like(conf_matrix, dtype=float),
                          where=row_sums > 0)

    plt.imshow(norm_conf, cmap='Blues', interpolation='nearest')
    plt.colorbar(label='Normalized Frequency')
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    if class_names is not None:
        plt.xticks(range(len(class_names)), class_names, rotation=90, fontsize=8)
        plt.yticks(range(len(class_names)), class_names, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def evaluate_model(model, dataloader, criterion, device, pad_idx=0, num_classes=None,
                   save_confusion_path='figures/confusion_matrix.png'):
    """
    完整评估模型：计算 Loss、准确率、精确率、召回率、F1、混淆矩阵
    返回评估结果字典
    """
    model.eval()
    total_loss = 0
    total_correct_tokens = 0
    total_tokens = 0
    total_correct_sentences = 0
    total_sentences = 0
    all_logits = []
    all_targets = []

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

            # Loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()

            # Token 准确率
            acc, correct, total = compute_token_accuracy(logits, tgt_output, pad_idx)
            total_correct_tokens += correct
            total_tokens += total

            # 句子准确率
            preds = logits.argmax(dim=-1)
            sent_acc, sent_correct, sent_total = compute_sentence_accuracy(preds, tgt_output, pad_idx)
            total_correct_sentences += sent_correct
            total_sentences += sent_total

            all_logits.append(logits)
            all_targets.append(tgt_output)

    avg_loss = total_loss / len(dataloader)
    token_acc = total_correct_tokens / total_tokens if total_tokens > 0 else 0
    sent_acc = total_correct_sentences / total_sentences if total_sentences > 0 else 0

    # 合并所有 logits 和 targets 计算精确率/召回率/F1/混淆矩阵
    # 注意：不同 batch 的序列长度可能不同（动态 padding），
    # 因此先将每个 batch 展平为 [batch*seq_len, vocab_size] 和 [batch*seq_len] 再拼接
    all_logits_flat = [logits.reshape(-1, logits.size(-1)) for logits in all_logits]
    all_targets_flat = [targets.reshape(-1) for targets in all_targets]
    all_logits = torch.cat(all_logits_flat, dim=0)
    all_targets = torch.cat(all_targets_flat, dim=0)

    precision, recall, f1 = compute_precision_recall_f1(all_logits, all_targets, pad_idx, num_classes)
    conf_matrix, top_classes = compute_confusion_matrix(all_logits, all_targets, pad_idx, num_classes, top_k=20)

    # 保存混淆矩阵图（使用 top_classes 作为类别名称，最后加一个 'other'）
    class_names = [str(c) for c in top_classes] + ['other']
    if save_confusion_path is not None:
        plot_confusion_matrix(conf_matrix, class_names=class_names, save_path=save_confusion_path,
                              title=f'Confusion Matrix (Token Acc: {token_acc:.4f})')

    return {
        'loss': avg_loss,
        'token_accuracy': token_acc,
        'sentence_accuracy': sent_acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix.tolist(),
        'top_classes': top_classes,
        'correct_tokens': total_correct_tokens,
        'total_tokens': total_tokens,
        'correct_sentences': total_correct_sentences,
        'total_sentences': total_sentences,
    }


def print_evaluation_results(results, title="评估结果"):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Loss:              {results['loss']:.4f}")
    print(f"  Token 准确率:      {results['token_accuracy']:.4f} ({results['correct_tokens']}/{results['total_tokens']})")
    print(f"  句子准确率:        {results['sentence_accuracy']:.4f} ({results['correct_sentences']}/{results['total_sentences']})")
    print(f"  精确率 (Macro):    {results['precision']:.4f}")
    print(f"  召回率 (Macro):    {results['recall']:.4f}")
    print(f"  F1 分数 (Macro):   {results['f1_score']:.4f}")
    print(f"{'='*60}")


# 需要在 evaluate_model 中使用的 mask 函数
# 这里导入避免循环依赖
from model import create_pad_mask, create_subseq_mask


# Transformer 论文的 Noam 学习率调度器
class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self.d_model ** (-0.5) * min(self._step ** (-0.5), self._step * self.warmup_steps ** (-1.5))
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr