# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

# 1. Input Embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx):
        super().__init__()
        # 创建一个嵌入层，并指定 padding_idx 以确保填充标记不会被更新
        # 向量查找器，输入句子的整数编号，输出对应的向量矩阵
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        # d_model 嵌入维度
        self.d_model = d_model
        self.scale = math.sqrt(d_model) # 预计算，后续用于缩放

    def forward(self, x):
        # 前向，查询嵌入层向量，并放缩，让词嵌入值和位置编码值处于同一数量级，便于模型训练
        return self.emb(x) * self.scale
        # x经过emb：(batch_size, seq_len) -> (batch_size, seq_len, d_model)
'''
为什么要缩放嵌入向量？

引入嵌入向量缩放因子 sqrt(d_model)，目的是对齐词嵌入与正弦位置编码的数值量级
正弦位置编码取值固定在 [-1,1] 区间，原始词嵌入随模型维度增大模长偏小
若不进行缩放，位置编码数值会过大，掩盖词向量本身的语义信息
以 sqrt(d_model) 后使两者尺度均衡，直接逐元素相加
既能保留词向量语义特征，又能有效融入位置信息，避免互相压制

为什么放缩到这个数值，d_model 的平方根？
需要极深的数学功底，留待后人说。
'''

# 2. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):   
        super().__init__()
        # 创建一个位置编码矩阵，大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 生成序列，并添加一个列维度，转为列向量 (max_len, 1)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 生成频率，奇数用cos，偶数用sin，频率随着维度增加而增加（模拟位置编码）
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        # 对于每一个词，每两维度，对应一个相位，奇数维度用sin，偶数维度用cos
        pe[:, 0::2] = torch.sin(pos * div)
        # 通过乘以频率，生成不同周期的正弦波，编码位置信息
        # 位置 0：[sin(0×ω₀), cos(0×ω₀), sin(0×ω₁), cos(0×ω₁), ...]一共512维
        pe[:, 1::2] = torch.cos(pos * div)
        # 添加一个批次维度，变为 (1, max_len, d_model)，以便后续与输入嵌入相加时广播
        pe = pe.unsqueeze(0)
        # register_buffer 会把 pe 保存为模型的一部分，但不会当成可训练参数（训练时不更新它）
        self.register_buffer('pe', pe)

    def forward(self, x):
        #  表示在第二维（长度维度）只取前 seq_len 个位置，只取有效位置的编码
        return x + self.pe[:, :x.size(1)]
'''
为什么用频率变化来表示位置编码？不直接用绝对大小？
1.绝对大小，会造成数值爆炸；
正弦波把值稳稳锁死在 [-1, 1] 之间，不管位置多大，数值永远有限，训练稳定。

2.正弦余弦自带相对位置推理能力（Transformer 最关键）
3.具备外推性：能预测没见过的超长序列 
两者同源但不等价：
位置 pos + k 的编码，可以由 位置 pos 的编码 做线性变换直接得到；
模型不需要死记每一个绝对位置，天然能算出「我和你隔几个词」；
注意力机制能自动学到：距离越近，关联越强。

推理时能处理比训练集更长的句子，能够处理没有见过的句子。
4.归一化有界、分布均匀，不干扰词嵌入语义;
正弦编码有界、尺度匹配，刚好微弱叠加位置信息，不破坏原词语义。
'''

# 3. Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    # 取最后一维的维度
    d_k = q.size(-1)
    # 计算相似度矩阵（注意力分数）
    '''
        seq_q,seq_k 是查询和键的序列长度，d_k 是每个头的维度
        q 形状：[batch, heads, seq_q, d_k]
        k 形状：[batch, heads, seq_k, d_k]
        将k的最后两维度转置，得到 [batch, heads, d_k, seq_k]
        这样做是为了计算 q 和 k 的点积，得到 [batch, heads, seq_q, seq_k] 
        的注意力分数矩阵，每个位置的分数表示查询与所有键的相似度
        matmul 是批量矩阵乘法，计算 q 和 k 的点积，得到注意力分数矩阵
        然后再缩放，除以 sqrt(d_k)，防止随着维度增加，点积结果过大导致 softmax 后梯度消失
    '''
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果存在掩码，则将需要屏蔽的位置填充为 -1e9（负无穷）
    # 使这些位置在 softmax 后概率趋近于 0，模型不会关注无效位置
    if mask is not None:
        attn = attn.masked_fill(~mask, -1e9)

    attn = F.softmax(attn, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    # 用刚刚得到的注意力权重，对 v（书的内容摘要）做加权平均。
    output = torch.matmul(attn, v)
    # output: (batch_size, heads, seq_len, d_k)
    return output, attn


# 4. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads          # 专家个数，比如 8
        self.d_k = d_model // n_heads   # 每个专家看的维度，512/8=64
        self.wq = nn.Linear(d_model, d_model)  # 生成“查询”的线性变换
        self.wk = nn.Linear(d_model, d_model)  # 生成“键”的线性变换
        self.wv = nn.Linear(d_model, d_model)  # 生成“值”的线性变换
        self.fc = nn.Linear(d_model, d_model)  # 最后的融合层，把8份报告揉在一起

    def forward(self, q, k, v, mask=None, dropout=None):
        batch_size = q.size(0) # 获取批次大小
        # q = self.wq(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k = self.wk(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # v = self.wv(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 1. 线性投影：将 q/k/v 映射到 d_model 维空间 (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 2. 切分多头：把 d_model 维均分成 n_heads 个 d_k 维的子空间
        #    形状变为 (batch_size, seq_len, n_heads, d_k)
        q = q.view(batch_size, -1, self.n_heads, self.d_k)
        k = k.view(batch_size, -1, self.n_heads, self.d_k)
        v = v.view(batch_size, -1, self.n_heads, self.d_k)

        # 3. 交换维度：把 seq_len 和 n_heads 互换
        #    新形状 (batch_size, n_heads, seq_len, d_k)
        #    这样每个注意力头可以独立地看到完整的序列，并且各自拥有独立的查询、键、值
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out, attn = scaled_dot_product_attention(q, k, v, mask, dropout)
        # 4. 多头拼接：把 n_heads 个头的输出拼接回 d_model 维
        #    (batch, n_heads, seq, d_k) -> (batch, seq, n_heads, d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        # .view(batch_size, -1, d_model)  # 最后变回 (batch_size, seq_len, d_model)
        # 这相当于把每个词的 8 个头输出首尾相接拼成一条长向量。
        # 每个词重新获得了一个 512 维的表示，它包含所有头从不同角度提取的信息。
        
        # 5. 最后通过一个线性层融合多头信息，得到最终的输出
        return self.fc(out)
    
# 1. 多头注意力采用先整体投影再切分的融合实现，而非独立构建多个小头投影：
#    - 计算并行性与硬件吞吐量：合并为批量矩阵运算，充分利用GPU并行加速，提升运算效率
#    - 数学表达等价性：切分与独立投影数学等价，不损失模型表达能力
#    - 批处理效率与内存局部性：统一张量布局，减少内存碎片与IO开销，优化计算流水线

# 2. 多头注意力划分低维子空间，而非单头直接使用完整模型维度：
#    - 计算效率：在参数约束下降低单头注意力计算开销，提升运算效率
#    - 表征多样性：迫使各头学习差异化特征，避免表征坍塌与冗余，捕捉语法、语义、指代等多视角信息
#    - 集成学习：多头独立捕捉不同模式，融合后增强模型鲁棒性与泛化能力


# 5. Position-wise Feed Forward（逐词前馈网络）
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 先升维，然后激活，dropout，最后降维回 d_model
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
# 为什么加入FFN前馈网络：
# 1. 增加模型的非线性表示能力
#    注意力机制本质是线性加权求和，虽然能捕捉全局依赖，但非线性拟合能力有限；
#    加入两层全连接层 + ReLU激活，让模型学习更复杂的特征交互与高阶语义信息。

# 2. 位置独立处理
#    每个token独立通过同一套MLP，不产生跨位置信息流动；
#    跨位置的信息交互完全由注意力机制负责，职责分工明确，让模型更易训练、更稳定。

# 3. 维度先升后降，提升模型容量
#    中间层将维度扩大到4倍（如2048），大幅增强特征表达空间；
#    最后映射回原始维度，保证与下一层网络结构兼容。

# 通俗理解：
# 注意力层负责让每个词“收集全局相关信息”，相当于收集原材料；
# FFN前馈层负责对每个词单独“深度加工、提炼语义”，相当于独立思考与特征提纯。

# 6. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力，传入三个x（查询、键、值都是同一个输入）
        attn_out = self.attn(x, x, x, mask, self.dropout)  # 传入 dropout
        # 残差连接 + 层归一化
        x = self.norm1(x + self.dropout(attn_out))
        # 前馈网络
        ff_out = self.ff(x)
        # 残差连接 + 层归一化
        x = self.norm2(x + self.dropout(ff_out))
        return x
# =============================================================================
# 编码层核心结构：残差连接 + 层归一化 原理注释
# =============================================================================
# 1. 残差连接 (Residual Connection)
# 公式：x_out = x + Sublayer(x)
#
# 核心作用：为梯度开辟“高速公路”，解决深层网络梯度消失/爆炸问题
# - 反向传播时，梯度 = 1 + 子层梯度
# - 即使子层梯度趋近于0，恒等映射的+1也能保证梯度正常传递
# - 让模型可以安全堆叠数十/上百层，不会退化、不会优化困难

# 2. 层归一化 (Layer Normalization)
# 作用：稳定训练、加速收敛、解决内部协变量偏移
# - 对每个样本独立做归一化（不受batch/序列长度影响）
# - 把每层输出拉回均值0、方差1的稳定分布
# - 允许使用更大学习率，收敛更快
#
# 与批归一化区别：
# - BatchNorm：按批次归一化，不适合NLP变长序列
# - LayerNorm：按样本/Token归一化，天然适合Transformer
#
# 计算流程：
# 1. 求当前特征均值 μ 和方差 σ²
# 2. 标准化：(x - μ) / sqrt(σ² + ε)
# 3. 可学习参数 γ（缩放）+ β（偏移）恢复表达能力
#
# 3. 残差 + 归一化 组合方式（本代码使用：Post-LN）
# 公式：x = LayerNorm(x + Dropout(Sublayer(x)))
# 流程：子层计算 → 残差相加 → Dropout → 层归一化
#
# 配合优势：
# - 残差保证梯度/信息畅通
# - 归一化清理分布偏移，让下一层输入更规范
# - 两者结合是Transformer能堆叠深层的关键


# 7. Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        # 交叉注意力，查询来自 Decoder 输入，键和值来自 Encoder 输出
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 带掩码的多头自注意力，保证解码器只能看到当前位置及之前的位置
        attn_out = self.self_attn(x, x, x, tgt_mask, self.dropout)
        x = self.norm1(x + self.dropout(attn_out))
        # 交叉注意力
        cross_out = self.cross_attn(x, enc_out, enc_out, src_mask, self.dropout)
        x = self.norm2(x + self.dropout(cross_out))
        # 前馈
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

# 8. Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout, pad_idx=0):
        super().__init__()
        self.emb = Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)  # 新增 dropout 层

    def forward(self, x, mask):
        x = self.emb(x)
        x = self.pe(x)
        x = self.dropout(x)  # 在嵌入+位置编码后应用 dropout
        for layer in self.layers:
            x = layer(x, mask)
        return x

# 9. Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout, pad_idx=0):
        super().__init__()
        self.emb = Embedding(vocab_size, d_model, pad_idx)
        self.pe = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)  # 新增 dropout 层

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.emb(x)
        x = self.pe(x)
        x = self.dropout(x)  # 在嵌入+位置编码后应用 dropout
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

# 10. Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model, n_layers, n_heads, d_ff, max_len, dropout, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, n_layers, n_heads, d_ff, max_len, dropout, pad_idx)
        self.decoder = Decoder(tgt_vocab, d_model, n_layers, n_heads, d_ff, max_len, dropout, pad_idx)
        self.fc = nn.Linear(d_model, tgt_vocab)

        # 可选：权重共享（decoder embedding 与输出层）
        # self.decoder.emb.emb.weight = self.fc.weight

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc(dec_out)

# 11. Mask 生成

# 这两个函数的seq是原始的整数索引序列，形状必须是 (batch_size, seq_len)，
# 因为它们需要根据 pad_idx 来生成掩码，或者根据 seq_len 来生成未来位置掩码。
# 填充掩码：标记输入序列中哪些位置是填充（pad），模型不应该关注这些位置
def create_pad_mask(seq, pad_idx):
    # seq: (batch_size, seq_len)，pad_idx 是填充标记的索引
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
# 维度变化说明：
# (seq != pad_idx)        : (batch_size, seq_len)          原始布尔掩码
# unsqueeze(1)            : (batch_size, 1, seq_len)       扩展多头维度
# unsqueeze(2)            : (batch_size, 1, 1, seq_len)     匹配注意力矩阵形状
# 最后两个 1 用于广播对齐注意力分数 (batch, heads, seq_q, seq_k)
# seq_q 维度自动广播，seq_k 维度指定需要屏蔽的 key 位置


# 未来位置掩码：在解码器中，防止模型看到未来的词（即当前位置之后的词）
# 构造一个 上三角全 False 的下三角矩阵 来遮掩未来的词
def create_subseq_mask(seq):
    # seq 是目标序列的整数 ID 张量，形状 (batch_size, seq_len)
    batch_size, seq_len = seq.size()

    # 创建一个 (seq_len, seq_len) 的全 1 张量
    # 然后沿最后一个维度取下三角部分（包括主对角线）
    # 结果：下三角区域为 True，上三角区域为 False
    mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device)).bool()

    # 在 0 和 1 位置各插入一个维度，变成 (1, 1, seq_len, seq_len)
    mask = mask.unsqueeze(0).unsqueeze(0)

    # 扩展到整个 batch，变成 (batch_size, 1, seq_len, seq_len)
    # -1 表示该维度不变，即保留原来的 1；seq_len 不变
    return mask.expand(batch_size, -1, -1, -1)

# 这里做的维度变换，是为了对齐注意力分数矩阵
#if mask is not None:
#        attn = attn.masked_fill(~mask, -1e9)
# 这里通过将softmax前的分数矩阵中需要屏蔽的位置填充为一个非常大的负数（-1e9），
# 使得这些位置在softmax后对应的概率趋近于0，从而实现对未来词的屏蔽，保证模型只能关注当前位置及之前的位置。