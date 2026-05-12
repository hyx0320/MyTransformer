# config.py
import torch

class Config:
    """默认配置（基础模型，对应论文 base model）"""
    # 模型参数
    d_model = 512          # Transformer 的嵌入维度
    n_layers = 6           # Encoder 和 Decoder 的层数
    n_heads = 8            # 多头注意力头数
    d_ff = 2048            # 前馈网络隐藏层维度
    dropout = 0.1          # Dropout 比率
    max_len = 5000         # 位置编码支持的最大序列长度

    # 数据参数
    src_vocab_size = 8000  # 源语言词表大小
    tgt_vocab_size = 8000  # 目标语言词表大小
    batch_size = 32
    pad_idx = 0            # 填充标记索引

    # 训练参数
    epochs = 20
    lr = 1e-4              # 学习率
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择计算设备
    
    # 数据集路径
    data_path = "./data/Multi30K"


# ============================================================
# 超参数对比实验配置
# 用于分析不同超参数对训练效果的影响
# ============================================================

def get_experiment_configs():
    """
    返回一组实验配置字典，用于超参数对比分析。
    每个配置只改变一个超参数，其余与 base 保持一致。
    """
    base = {
        'd_model': 512,
        'n_layers': 6,
        'n_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'batch_size': 32,
        'lr': 1e-4,
        'epochs': 10,  # 对比实验用较少 epoch 以节省时间
        'max_len': 5000,
        'src_vocab_size': 8000,
        'tgt_vocab_size': 8000,
        'pad_idx': 0,
        'data_path': "./data/Multi30K",
        'device': "cuda" if torch.cuda.is_available() else "cpu",
    }

    experiments = {
        # ===== 1. 改变 embedding dimension (d_model) =====
        'd_model_128':  {**base, 'd_model': 128,  'd_ff': 512,  'n_heads': 4,  'name': 'd_model=128'},
        'd_model_256':  {**base, 'd_model': 256,  'd_ff': 1024, 'n_heads': 4,  'name': 'd_model=256'},
        'd_model_512':  {**base, 'd_model': 512,  'd_ff': 2048, 'n_heads': 8,  'name': 'd_model=512 (base)'},

        # ===== 2. 改变注意力头数 (n_heads) =====
        'n_heads_2':    {**base, 'n_heads': 2,   'name': 'n_heads=2'},
        'n_heads_4':    {**base, 'n_heads': 4,   'name': 'n_heads=4'},
        'n_heads_8':    {**base, 'n_heads': 8,   'name': 'n_heads=8 (base)'},

        # ===== 3. 改变编码器/解码器层数 (n_layers) =====
        'n_layers_2':   {**base, 'n_layers': 2,  'name': 'n_layers=2'},
        'n_layers_4':   {**base, 'n_layers': 4,  'name': 'n_layers=4'},
        'n_layers_6':   {**base, 'n_layers': 6,  'name': 'n_layers=6 (base)'},

        # ===== 4. 改变 batch size =====
        'batch_16':     {**base, 'batch_size': 16, 'name': 'batch_size=16'},
        'batch_32':     {**base, 'batch_size': 32, 'name': 'batch_size=32 (base)'},
        'batch_64':     {**base, 'batch_size': 64, 'name': 'batch_size=64'},

        # ===== 5. 改变 learning rate =====
        'lr_1e-5':      {**base, 'lr': 1e-5,      'name': 'lr=1e-5'},
        'lr_1e-4':      {**base, 'lr': 1e-4,      'name': 'lr=1e-4 (base)'},
        'lr_5e-4':      {**base, 'lr': 5e-4,      'name': 'lr=5e-4'},

        # ===== 6. 改变 dropout =====
        'dropout_0.0':  {**base, 'dropout': 0.0,  'name': 'dropout=0.0'},
        'dropout_0.1':  {**base, 'dropout': 0.1,  'name': 'dropout=0.1 (base)'},
        'dropout_0.3':  {**base, 'dropout': 0.3,  'name': 'dropout=0.3'},

        # ===== 7. 改变训练 epoch =====
        'epochs_5':     {**base, 'epochs': 5,     'name': 'epochs=5'},
        'epochs_10':    {**base, 'epochs': 10,    'name': 'epochs=10 (base)'},
        'epochs_20':    {**base, 'epochs': 20,    'name': 'epochs=20'},
    }

    return experiments


def config_from_dict(d):
    """将字典转换为 Config 对象"""
    cfg = Config()
    for key, value in d.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg