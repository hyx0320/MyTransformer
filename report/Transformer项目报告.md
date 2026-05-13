# Transformer 论文阅读与代码复现项目报告

> **项目名称**: Transformer 机器翻译模型实现  
> **原始论文**: Vaswani et al., *Attention Is All You Need*, NIPS 2017  
> **数据集**: Multi30K（德语→英语）  
> **开发框架**: PyTorch  

---

## 目录

1. [项目背景](#1-项目背景)
2. [论文介绍](#2-论文介绍)
3. [Transformer 模型原理](#3-transformer-模型原理)
4. [代码结构说明](#4-代码结构说明)
5. [核心模块分析](#5-核心模块分析)
6. [数据集介绍](#6-数据集介绍)
7. [实验设置](#7-实验设置)
8. [实验结果](#8-实验结果)
9. [参数分析](#9-参数分析)
10. [遇到的问题与解决方法](#10-遇到的问题与解决方法)
11. [小组分工](#11-小组分工)
12. [总结与收获](#12-总结与收获)
13. [参考文献与参考项目](#13-参考文献与参考项目)

---

## 1. 项目背景

### 1.1 深度学习在序列建模中的发展

在 Transformer 出现之前，自然语言处理（NLP）领域处理序列数据的主流架构是**循环神经网络（RNN）**及其变体 **LSTM**（长短期记忆网络）和 **GRU**（门控循环单元）。这些模型通过逐步迭代的方式处理序列，每个时间步的隐藏状态依赖于前一个时间步的输出，天然具有"顺序计算"的特性。

然而，这种顺序计算带来了一个根本性的瓶颈：**无法并行化**。当处理长序列时，RNN 必须逐个时间步依次计算，训练速度极慢。此外，RNN 还存在长程依赖问题——随着序列长度增加，较早位置的信息在反向传播过程中容易发生梯度消失或梯度爆炸，导致模型难以捕捉远距离的语义关联。

### 1.2 此前工作的局限性

| 模型类型 | 优点 | 缺点 |
|---------|------|------|
| RNN / LSTM / GRU | 天然适合序列建模 | 顺序计算无法并行，长程依赖困难 |
| CNN（如 ConvS2S） | 可并行计算 | 长程依赖路径随距离线性增长 |
| 注意力机制（早期） | 能捕捉全局依赖 | 仅作为 RNN 的辅助组件，未独立使用 |

卷积神经网络（CNN）虽然可以并行计算，但任意两个位置之间的路径长度随距离线性增长（ConvS2S）或对数增长（ByteNet），学习远距离依赖仍然困难。注意力机制此前仅被用作 RNN 的辅助组件，从未被当作序列建模的主要计算原语。

### 1.3 项目目标

本项目旨在：
1. **深入阅读** Transformer 原始论文，理解其核心思想和架构设计
2. **从零复现** Transformer 模型，使用 PyTorch 框架实现完整的 Encoder-Decoder 架构
3. **在真实数据集上训练**，验证模型在机器翻译任务上的效果
4. **进行系统的超参数分析**，探究各组件对模型性能的影响
5. **撰写完整的项目报告**，记录学习过程和实验发现

---

## 2. 论文介绍

### 2.1 论文基本信息

- **标题**: Attention Is All You Need
- **作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin（*共同第一作者，随机排序*）
- **机构**: Google Brain / Google Research / University of Toronto
- **会议**: NIPS 2017（Neural Information Processing Systems）
- **arXiv**: 1706.03762

### 2.2 研究背景与动机（结合个人理解）

#### 2.2.1 时代背景：RNN 的统治与困境

在 2017 年之前，序列建模领域几乎是 RNN 及其变体的天下。机器翻译、语音识别、文本生成等任务中，LSTM 和 GRU 是当之无愧的主力架构。这些模型的设计思想非常直观——人类阅读句子是一个词一个词按顺序读的，那么模型也应该按顺序处理。每个时间步的隐藏状态 `h_t` 依赖于前一个时间步 `h_{t-1}`，形成了一条"时间链"。

然而，这种看似自然的"顺序处理"方式隐藏着三个根本性问题：

**第一，无法并行化。** 这是最致命的瓶颈。在 GPU 时代，并行计算是加速的关键。但 RNN 必须逐个时间步依次计算——要计算第 10 个词，必须先算完前 9 个。这意味着即使使用最先进的 GPU，RNN 的训练速度也受限于序列长度。打个比方：RNN 像一条单车道公路，车辆必须一辆接一辆通过；而 Transformer 像一条八车道高速公路，所有车辆可以同时行驶。

**第二，长程依赖困难。** 当句子长度增加时，较早位置的信息需要通过多个时间步才能传递到较后位置。在反向传播过程中，梯度需要沿着时间链逐层回传，容易发生梯度消失（信息被"稀释"）或梯度爆炸（信息被"放大"）。虽然 LSTM 通过门控机制在一定程度上缓解了这个问题，但并没有从根本上解决。直观理解：读一篇长文章时，RNN 读到后半段时已经"忘记"了开头的内容。

**第三，注意力机制仅作为辅助。** 早在 2015 年，Bahdanau 等人就提出了注意力机制用于机器翻译，但其作用仅限于帮助 RNN 在解码时"回顾"编码器的各个位置。注意力从未被当作主要的计算原语，而是依附于 RNN 的"配角"。

#### 2.2.2 论文的核心洞察

论文作者提出了一个颠覆性的假设：

> **完全可以仅用注意力机制替代 RNN 和 CNN，构建一个更可并行化、训练更快且翻译质量不降反升的序列转录模型。**

这一假设并非凭空而来，而是基于三个关键洞察：

1. **注意力机制天然具有 O(1) 的最大路径长度**：在 Self-Attention 中，任意两个位置之间只需要一步计算就能直接交互，而 RNN 需要 O(n) 步。这意味着注意力机制在捕捉长程依赖方面具有根本性的优势。

2. **多头注意力可以并行关注不同子空间**：单头注意力只能从一个角度计算相关性，而多头注意力允许模型同时关注语法、语义、指代等不同层面的信息，类似于"多个专家从不同角度分析同一个问题"。

3. **GPU 并行计算能力被 RNN 严重浪费**：RNN 的顺序计算本质导致 GPU 的并行计算能力无法充分发挥。如果能设计一种完全并行的架构，训练效率将得到数量级的提升。

#### 2.2.3 我的理解：为什么这个想法在当时是"反直觉"的

站在今天的视角，Transformer 的成功似乎是理所当然的。但在 2017 年，这个想法其实是相当"反直觉"的。原因在于：

- **序列天然是有顺序的**，而注意力机制本身不感知顺序。让一个"不分先后"的机制去处理"有先后"的序列，听起来就不合理。这也是为什么位置编码成为 Transformer 的关键组件——它是在"补救"注意力机制的天生缺陷。
- **抛弃已被验证的 RNN 架构**需要极大的勇气。当时 RNN/LSTM 已经在无数任务上证明了有效性，而纯注意力的想法还没有被充分验证过。
- **O(n²) 的复杂度看似是倒退**：Self-Attention 的计算复杂度是 O(n²·d)，而 RNN 是 O(n·d²)。当序列长度 n 大于模型维度 d 时，Self-Attention 的计算量更大。作者在论文中也坦诚地讨论了这一点。

正是这种"敢于挑战主流范式"的精神，使得这篇论文成为深度学习史上的里程碑。

### 2.3 Transformer 解决了什么问题（结合个人理解）

#### 2.3.1 并行化问题——最直接的突破

Transformer 最直接的贡献是**彻底解决了序列模型的并行化问题**。在 Transformer 训练的过程中，整个序列可以一次性输入模型，所有位置同时计算 Self-Attention。这意味着：

- 训练速度从 O(n) 降低到 O(1)（顺序操作数）
- GPU 的并行计算能力得到充分利用
- 训练时间从几天缩短到几小时

**个人理解**：RNN 中每个工人（时间步）必须等前一个工人完成才能开始工作；而在Transformer训练过程中，一次性把所有的输入数据输入给解码器，并行的生成输出。这种效率提升是架构层面的根本性改进，不是通过优化硬件或工程技巧能实现的。

#### 2.3.2 长程依赖问题——最根本的改进

Self-Attention 的最大路径长度为 O(1)，这意味着**任意两个位置的 token 之间只需要一步计算就能建立联系**。相比之下：

- RNN：O(n) 步（需要沿着时间链逐层传递）
- CNN：O(log_k(n)) 步（需要逐层扩大感受野）

**个人理解**：这就像两个人对话。RNN 的方式是让信息通过"传话游戏"逐人传递，传到后面可能已经失真了；Transformer 的方式是让两个人直接对话，信息传递没有中间损耗。在机器翻译中，这意味着模型可以轻松捕捉"主语-谓语"之间跨越多个词的依赖关系。

#### 2.3.3 计算效率问题——条件性的优势

论文的 Table 1 给出了一个重要的分析：当序列长度 n 小于模型维度 d 时，Self-Attention 的计算复杂度 O(n²·d) 实际上低于 RNN 的 O(n·d²)。在机器翻译等典型 NLP 任务中，句子长度通常为 20-50 个词，而模型维度为 512，因此 n < d 的条件通常成立。

**个人理解**：这解释了为什么 Transformer 在机器翻译上特别有效——句子长度适中，Self-Attention 的计算优势得以发挥。但对于文档级翻译（n 很大）或图像处理（n = 像素数），O(n²) 的复杂度就成了瓶颈。这也是后续工作（如 Longformer、Linformer）致力于降低 Self-Attention 复杂度的原因。

#### 2.3.4 架构统一问题——最深远的影响

Transformer 提出了一个通用的 Encoder-Decoder 架构，这个架构后来被证明具有极强的泛化能力：

- **Encoder-only**（如 BERT）：适用于自然语言理解任务
- **Decoder-only**（如 GPT）：适用于语言生成任务
- **Encoder-Decoder**（如 T5、BART）：适用于序列到序列任务

**个人理解**：Transformer 的架构统一性是其最具影响力的贡献之一。在 Transformer 之前，不同任务需要设计不同的模型架构（文本分类用 CNN，翻译用 RNN，生成用 LSTM）。Transformer 之后，几乎所有 NLP 任务都可以用同一个架构的不同变体来解决。这种"大一统"的趋势在后续的 GPT 和 BERT 中得到了进一步强化。

### 2.4 Transformer 与 RNN、LSTM、GRU 的区别（结合个人理解）

#### 2.4.1 核心区别对比表

| 对比维度 | RNN / LSTM / GRU | Transformer |
|---------|-----------------|-------------|
| **计算方式** | 顺序计算，逐步迭代 | 并行计算，一次性处理整个序列 |
| **最大路径长度** | O(n)（随序列长度线性增长） | O(1)（任意位置直接连接） |
| **每层复杂度** | O(n·d²) | O(n²·d)（n < d 时更优） |
| **顺序操作数** | O(n) | O(1) |
| **长程依赖** | 困难（梯度消失/爆炸） | 容易（直接注意力） |
| **位置信息** | 天然包含（时间步顺序） | 需要额外注入（Positional Encoding） |
| **可解释性** | 隐藏状态难以解释 | 注意力权重可可视化 |
| **参数量** | 与序列长度无关 | 与序列长度无关（但计算量与 n² 相关） |
| **训练稳定性** | 需要梯度裁剪、门控机制 | 残差连接 + LayerNorm 保证稳定 |

#### 2.4.2 从"递推"到"加权"的范式转变

**个人理解**：RNN 和 Transformer 最本质的区别不在于具体的数学公式，而在于**建模序列关系的范式**。

RNN 的范式是**递推（Recurrence）**：通过一个不断更新的隐藏状态来隐式地编码序列信息。每个时间步，新输入与旧状态融合，产生新状态。这种方式的优点是天然包含顺序信息，缺点是信息在传递过程中会衰减和混淆。

Transformer 的范式是**加权（Attention）**：通过计算所有位置之间的相关性权重来显式地建模序列关系。每个位置直接与所有其他位置交互，没有信息传递的中间损耗。这种方式的优点是信息传递无损，缺点是需要人工注入位置信息。

用一个比喻来理解：
- **RNN** 像一个人在读一本书，必须一页一页按顺序读，读到后面时对前面的内容只有模糊的记忆
- **Transformer** 像一个人把整本书摊开在桌面上，可以随时看向任何一页，精确地找到需要的信息

#### 2.4.3 为什么 LSTM 和 GRU 没有解决根本问题

LSTM 和 GRU 通过引入门控机制（遗忘门、输入门、输出门）来缓解 RNN 的梯度消失问题，但它们并没有改变 RNN 的**顺序计算本质**。这意味着：

- 即使 LSTM 能更好地保留长程信息，训练速度仍然受限于序列长度
- 即使 GRU 比标准 RNN 更稳定，仍然无法并行化
- 门控机制增加了参数量和计算量，但并没有从根本上改变 O(n) 的顺序操作数

**个人理解**：LSTM 和 GRU 是在 RNN 框架内的"修补"，而 Transformer 是对 RNN 框架的"颠覆"。前者是改良，后者是革命。这也解释了为什么 Transformer 出现后，LSTM 和 GRU 在 NLP 领域迅速被取代——因为 Transformer 不是在 RNN 的基础上改进，而是提供了一个全新的、更优的替代方案。

#### 2.4.4 各自的适用场景

虽然 Transformer 在大多数 NLP 任务上优于 RNN，但 RNN 在某些场景下仍有其价值：

- **极短序列**：当序列长度很短（如 1-5 个 token）时，RNN 的简单性可能更有优势
- **在线/流式处理**：RNN 天然支持逐个 token 的增量处理，适合实时场景
- **资源受限设备**：RNN 的参数量通常小于 Transformer，更适合移动端部署
- **时间序列预测**：在某些时间序列任务中，RNN 的 inductive bias（时间连续性）可能更有利

### 2.5 论文的主要贡献和影响（结合个人理解）

#### 2.5.1 五大核心贡献

**贡献一：Transformer 架构——首个完全基于注意力的序列模型**

这是论文最核心的贡献。Transformer 证明了注意力机制不仅可以作为 RNN 的辅助组件，还可以独立承担序列建模的全部职责。这一发现彻底改变了深度学习的研究方向。

**个人理解**：论文标题"Attention Is All You Need"虽然带有修辞色彩，但它传达的核心信息是准确的——注意力机制确实足够强大，可以替代循环和卷积。后续的研究进一步证明，注意力机制不仅"够用"，而且"更好"。

**贡献二：Scaled Dot-Product Attention——一个简单但关键的技术创新**

缩放因子 1/√d_k 的引入看似微小，但至关重要。论文在 Section 3.2.1 中通过方差分析说明了其必要性：当 d_k 较大时，点积结果的方差随 d_k 线性增长，导致 softmax 进入梯度饱和区域。除以 √d_k 后，方差被归一化为 1，梯度保持在有效区域。

**个人理解**：这个设计体现了"魔鬼在细节中"的工程智慧。很多人在实现 Transformer 时可能会忽略这个缩放因子，但实验表明没有它模型确实难以训练。这也提醒我们，在深度学习研究中，看似微小的技术细节可能对最终效果产生决定性影响。

**贡献三：Multi-Head Attention——从"单一视角"到"多视角"的飞跃**

多头注意力的核心思想是：不只用一组注意力，而是用多组并行的注意力从不同角度关注信息。论文的消融实验（Table 3）明确显示，多头（h=8）显著优于单头（h=1），BLEU 从 24.9 提升到 25.8。

**个人理解**：多头注意力可以类比为"多个专家会诊"。每个专家（头）从自己的专业角度分析问题，有的关注语法结构，有的关注语义关系，有的关注指代消解。最后将各位专家的意见汇总，得到更全面、更准确的判断。这种"集成学习"的思想在深度学习中屡试不爽。

**贡献四：Positional Encoding——用正弦波编码位置信息**

由于 Transformer 没有循环结构，模型本身无法感知序列中 token 的顺序。位置编码的作用就是注入位置信息。论文选择了正弦/余弦函数，而非可学习的位置嵌入。

**个人理解**：正弦位置编码的设计非常巧妙。它不仅数值稳定（锁定在 [-1, 1]），而且具有相对位置推理能力——位置 pos+k 的编码可以由位置 pos 的编码通过线性变换得到。这意味着模型不需要死记每一个绝对位置，而是天然能算出"两个词隔了几个位置"。此外，正弦编码还具有外推性，可以处理比训练时更长的序列。

**贡献五：SOTA 翻译结果——用更少的成本获得更好的效果**

论文在 WMT 2014 英德翻译上取得了 28.4 BLEU（超越此前所有模型 2+ BLEU），英法翻译上取得了 41.0 BLEU（单模型 SOTA）。更重要的是，训练成本仅为此前最佳模型的几分之一——Transformer (base) 的训练成本为 3.3×10¹⁸ FLOPs，而 ConvS2S 为 9.6×10¹⁸，GNMT 为 2.3×10¹⁹。

**个人理解**：这个结果最有说服力的地方在于——Transformer 不仅在效率上胜出，在质量上也胜出。通常我们认为"更快"意味着"质量妥协"，但 Transformer 打破了这种权衡。它用更少的计算资源获得了更好的翻译质量，这在工程实践中具有巨大的价值。

#### 2.5.2 深远影响

**对 NLP 领域的重塑**：
- **BERT**（Devlin et al., 2019）使用 Transformer Encoder 在 11 项 NLP 任务上刷新了 SOTA
- **GPT 系列**（Radford et al., 2018; Brown et al., 2020）使用 Transformer Decoder 展示了语言模型的可扩展性
- **T5、BART** 等模型使用完整的 Encoder-Decoder 架构

**跨模态扩展**：
- **Vision Transformer (ViT)** 将图像分割为 patch 序列，用 Transformer 处理
- **CLIP、DALL-E** 使用 Transformer 处理图文多模态数据
- **语音识别、音乐生成**等领域也开始采用 Transformer

**大语言模型的基础**：
- ChatGPT、GPT-4、Claude、Gemini 等现代大语言模型均基于 Transformer Decoder 架构
- Transformer 的可扩展性（scaling law）使得模型可以通过增加参数量和训练数据来持续提升性能

**个人总结**：Transformer 的影响力远超一篇论文的范畴。它不仅是机器翻译的更好方案，更是一种全新的计算范式。从 NLP 到 CV，从语音到多模态，Transformer 正在重塑整个 AI 领域。而这一切的起点，就是 2017 年那篇标题看似夸张的论文——"Attention Is All You Need"。

---

## 3. Transformer 模型原理

### 3.1 整体架构（结合个人理解）

#### 3.1.1 Encoder-Decoder 架构的设计思想

Transformer 采用 **Encoder-Decoder** 架构，这是机器翻译任务的经典设计。整体结构如下：

```
输入序列 → [Embedding + Positional Encoding] → Encoder (N×6层) → 编码表示
                                                                      ↓
目标序列 → [Embedding + Positional Encoding] → Decoder (N×6层) → Linear → Softmax → 输出
```

**个人理解**：Encoder-Decoder 架构的设计非常直观——它模拟了"理解-生成"的认知过程。Encoder 负责"阅读理解"：把源语言句子（如德语）编码为一个丰富的语义表示；Decoder 负责"写作生成"：基于这个语义表示，逐词生成目标语言句子（如英语）。这种"先理解再表达"的方式符合人类翻译的直觉。

#### 3.1.2 Encoder 的组成

Encoder 由 N=6 个相同的层堆叠而成，每层包含：
1. **Multi-Head Self-Attention**（多头自注意力）：让每个词关注句子中的所有其他词
2. **Position-wise Feed-Forward Network**（逐位置前馈网络）：对每个词独立进行非线性变换
3. 每个子层周围有**残差连接**（Residual Connection）和**层归一化**（Layer Normalization）

**个人理解**：Encoder 的每一层可以看作一个"信息增强"步骤。自注意力让每个词收集全局上下文信息，FFN 对收集到的信息进行深度加工，残差连接保证信息不会在层层传递中丢失，层归一化则确保数值稳定。6 层堆叠意味着信息经过了 6 轮"收集-加工"的迭代，每次迭代都让表示更加丰富和精确。

#### 3.1.3 Decoder 的组成

Decoder 也由 N=6 个相同的层堆叠而成，每层包含：
1. **Masked Multi-Head Self-Attention**（带掩码的多头自注意力）：防止看到未来位置
2. **Multi-Head Cross-Attention**（交叉注意力）：查询来自 Decoder，键和值来自 Encoder
3. **Position-wise Feed-Forward Network**
4. 每个子层周围有残差连接和层归一化

**个人理解**：Decoder 比 Encoder 多了一个交叉注意力层，这是"理解-生成"架构的关键。Decoder 的自注意力负责"回顾已经生成的词"，交叉注意力负责"查看源语言信息"，FFN 负责"整合信息生成下一个词"。这种三层结构确保了生成过程既尊重已生成的内容，又忠实于源语言信息。

#### 3.1.4 为什么堆叠 6 层？

论文选择 N=6 是通过实验确定的。层数太少（如 2 层）模型容量不足，层数太多（如 12 层）训练困难和过拟合风险增加。6 层在当时的计算资源和任务复杂度下是一个合理的折中。后续的 BERT 使用了 12 层（Base）和 24 层（Large），GPT-3 使用了 96 层，说明随着计算资源的增长，更深的 Transformer 能带来更好的性能。

### 3.2 Scaled Dot-Product Attention（缩放点积注意力）（结合个人理解）

#### 3.2.1 核心公式与计算流程

这是 Transformer 中最核心的计算单元。其计算过程如下：

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

**计算步骤**:
1. **查询（Query）与键（Key）的点积**: 计算每个查询与所有键的相似度分数
2. **缩放（Scale）**: 除以 √d_k，防止随着维度增加点积结果过大导致 softmax 梯度消失
3. **掩码（Mask，可选）**: 将需要屏蔽的位置填充为 -1e9（负无穷），使 softmax 后概率趋近于 0
4. **Softmax**: 将分数归一化为概率分布
5. **加权求和**: 用注意力权重对值（Value）进行加权平均

#### 3.2.2 用检索系统类比理解 Q、K、V

**个人理解**：Q、K、V 的概念可以用一个检索系统来类比：

- **Query（查询）** = 你在搜索引擎中输入的关键词。在 Transformer 中，当前词会"发出查询"，询问其他词与自己的相关性。
- **Key（键）** = 网页的标题或标签。每个词都有一个"键"，用于匹配其他词发来的查询。
- **Value（值）** = 网页的实际内容。一旦确定了哪些词与当前词相关（通过 Q 和 K 的匹配），就根据相关性权重对这些词的内容（V）进行加权求和。

具体到机器翻译的例子：当翻译句子 "The cat sat on the mat" 时，对于 "sat" 这个词，它的 Query 会与 "cat" 的 Key 产生很高的匹配分数（因为猫是"坐"这个动作的主体），因此 "cat" 的 Value 会在 "sat" 的最终表示中占据较大权重。

#### 3.2.3 为什么需要缩放？——深入分析

当 d_k 较大时，点积结果 Q·K^T 的方差会随着 d_k 增大而增大（方差 = d_k）。如果不对其进行缩放，较大的方差会使 softmax 函数的梯度进入饱和区域（极端值附近梯度趋近于 0），导致训练困难。除以 √d_k 后，方差被归一化为 1，softmax 的梯度保持在有效区域。

**个人理解**：这个问题的本质是"维度灾难"的一个具体表现。假设 Q 和 K 的每个维度都是均值为 0、方差为 1 的独立随机变量，那么 d_k 个维度的点积的方差就是 d_k。当 d_k=64 时，点积结果的方差为 64，标准差为 8。这意味着大部分点积结果会落在 [-24, 24] 的范围内（3 倍标准差）。经过 softmax 后，较大的正值会"压倒"其他值，导致注意力分布几乎变成 one-hot 分布，梯度趋近于 0。除以 √64=8 后，结果被压缩到 [-3, 3] 的范围，softmax 的梯度保持在有效区域。

#### 3.2.4 为什么选择点积注意力而不是加性注意力？

论文在 Section 3.2.1 中比较了点积注意力和加性注意力（additive attention）。点积注意力可以使用高度优化的矩阵乘法实现，计算更快、空间效率更高。当 d_k 较小时，两种注意力性能相近；当 d_k 较大时，不加缩放的点积注意力性能下降，但缩放后两者性能相当。因此，缩放点积注意力在保持加性注意力性能的同时，获得了矩阵乘法的速度优势。

### 3.3 Multi-Head Attention（多头注意力）（结合个人理解）

#### 3.3.1 核心思想

多头注意力是 Scaled Dot-Product Attention 的扩展，其核心思想是：**不只用一组注意力，而是用多组并行的注意力从不同角度关注信息**。

**计算过程**:
1. 将 Q、K、V 分别通过 h 个不同的线性投影，映射到 h 个 d_k 维的子空间
2. 在每个子空间上独立执行 Scaled Dot-Product Attention
3. 将 h 个头的输出拼接（Concat）起来
4. 通过一个线性层融合多头信息

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
其中 head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
```

#### 3.3.2 为什么多头比单头好？——深入分析

**个人理解**：多头注意力的优势可以从三个层面理解：

**第一，表征多样性**。每个头通过不同的线性投影被映射到不同的子空间，因此它们会关注不同的特征模式。论文附录中的注意力可视化显示，有的头关注语法依赖关系（如动词-宾语），有的头关注语义相似性（如同义词），有的头关注位置邻近性。这种多样性使得模型能够从多个角度理解输入。

**第二，集成学习效应**。多头注意力本质上是一种隐式的集成学习（ensemble learning）。每个头是一个"弱学习器"，独立捕捉不同的模式。将这些弱学习器的输出拼接融合后，得到的表示比任何单头都更鲁棒、更全面。这与随机森林中多棵决策树集成的原理类似。

**第三，计算效率**。在参数量基本不变的前提下（d_k = d_model / h），多头显著优于单头。这是因为单头注意力需要在 d_model 维空间中计算一个全局注意力分布，而多头注意力将问题分解为 h 个 d_k 维子空间中的独立注意力计算。这种"分而治之"的策略降低了每个头的学习难度。

**一个直观的类比**：单头注意力就像只有一个医生看病，他可能只关注某些症状而忽略其他；多头注意力就像多个科室的专家会诊——眼科医生关注视觉信息，耳鼻喉科医生关注听觉信息，神经科医生关注神经系统——最后综合各位专家的意见，得到更全面的诊断。

#### 3.3.3 头数的选择

在我们的实现中，d_model=512，h=8，因此每个头的维度 d_k=64。论文的消融实验（Table 3）显示：
- h=1（单头）：BLEU 24.9
- h=4：BLEU 25.5
- h=8（最佳）：BLEU 25.8
- h=16：BLEU 25.6
- h=32：BLEU 25.4

头数过少（h=1）性能差，头数过多（h=32）性能也下降。这是因为头数过多时，每个头的维度 d_k 太小（512/32=16），不足以捕捉有意义的特征模式。h=8 是一个合理的折中。

### 3.4 Position-wise Feed-Forward Network（逐位置前馈网络）（结合个人理解）

每个 Encoder 和 Decoder 层中都包含一个前馈网络：

```
FFN(x) = max(0, x × W_1 + b_1) × W_2 + b_2
```

这是一个两层的全连接网络，中间使用 ReLU 激活函数。维度变化为：d_model → d_ff → d_model，其中 d_ff=2048（4 倍于 d_model）。

**个人理解**：FFN 在 Transformer 中的作用常被低估，但它实际上承担了至关重要的职责。

**第一，增加非线性**。注意力机制的本质是加权求和——这是一个线性操作。如果没有 FFN，多层注意力堆叠后仍然是线性变换，模型的表达能力极其有限。FFN 中的 ReLU 激活函数引入了非线性，使得模型能够学习复杂的特征交互。

**第二，职责分工**。注意力层和 FFN 层有明确的分工：注意力层负责"收集全局信息"（跨位置的交互），FFN 层负责"深度加工提炼语义"（每个位置独立处理）。这种"通信-计算"分离的设计使得模型更易训练、更稳定。

**第三，维度扩展**。先升维（512→2048）再降维（2048→512）的设计类似于 SVM 中的核技巧——在高维空间中，数据更容易被线性分离。4 倍的扩展因子（d_ff = 4 × d_model）是经验选择，更大的扩展因子（如 8 倍）可能带来更好的性能，但参数量也更大。

**一个直观的类比**：注意力层像是一个"信息收集员"，从各个位置收集相关信息；FFN 层像是一个"分析师"，对收集到的信息进行深入分析和提炼。两者配合，先收集再分析，循环迭代。

### 3.5 Residual Connection（残差连接）（结合个人理解）

每个子层（注意力或 FFN）的输出都会与输入相加：

```
output = LayerNorm(x + Sublayer(x))
```

**个人理解**：残差连接是 Transformer 能够堆叠 6 层甚至更深的关键技术。它的核心作用可以从两个角度理解：

**从梯度流动的角度**：反向传播时，梯度通过残差连接可以直接传递到前面的层，路径为：梯度 = 1 + 子层梯度。即使子层的梯度趋近于 0（梯度消失），恒等映射的 "+1" 也能保证梯度正常传递。这就像在深山中修建了一条"高速公路"，车辆（梯度）可以快速直达目的地，而不需要在崎岖的山路（子层）上缓慢行驶。

**从信息保留的角度**：每一层都在"增量更新"输入表示，而不是完全替换。子层学习的是"残差"——即需要在输入基础上增加或修改的部分。这使得模型更容易学习恒等映射（当子层不需要改变输入时，只需输出 0），从而避免了深层网络中的退化问题（degradation problem）。

**个人理解**：残差连接的巧妙之处在于，它让"什么都不做"变得容易。如果某一层发现当前表示已经足够好，不需要修改，它只需要让子层输出接近 0 即可。如果没有残差连接，每一层都必须学习一个完整的变换，即使这个变换是恒等映射也很难学习。

### 3.6 Layer Normalization（层归一化）（结合个人理解）

对每个样本独立做归一化，将输出拉回均值 0、方差 1 的稳定分布：

```
LayerNorm(x) = γ × (x - μ) / √(σ² + ε) + β
```

**个人理解**：层归一化的作用是解决"内部协变量偏移"（Internal Covariate Shift）问题——即每一层的输入分布会随着前一层参数的变化而不断变化，导致训练不稳定。LayerNorm 通过将每层的输出归一化到标准分布，使得下一层的输入分布保持稳定，从而允许使用更大的学习率，加速收敛。

**为什么用 LayerNorm 而不是 BatchNorm？**

这是 Transformer 设计中一个常被问及的问题。两者的核心区别在于归一化的维度：

- **BatchNorm**：在批次维度上归一化。对于形状为 (batch, seq_len, d_model) 的张量，BatchNorm 对每个特征维度（d_model）计算批次和序列维度上的均值和方差。这意味着 BatchNorm 依赖于批次大小，且对变长序列处理不便（不同长度的序列需要不同的统计量）。

- **LayerNorm**：在特征维度上归一化。LayerNorm 对每个样本的每个 token 独立计算均值和方差。这意味着 LayerNorm 不受批次大小和序列长度的影响，天然适合 NLP 中的变长序列。

**个人理解**：选择 LayerNorm 而非 BatchNorm 是 Transformer 设计中的一个关键决策。如果使用 BatchNorm，当批次中的序列长度不同时，填充位置的统计量会引入噪声，影响归一化效果。LayerNorm 对每个 token 独立归一化，完美避开了这个问题。此外，在推理阶段，LayerNorm 不需要维护全局统计量（如 BatchNorm 的 running mean 和 running var），实现更简洁。

### 3.7 Positional Encoding（位置编码）（结合个人理解）

#### 3.7.1 为什么需要位置编码？

由于 Transformer 没有循环结构，模型本身无法感知序列中 token 的顺序。对于 Self-Attention 来说，交换两个 token 的位置不会改变计算结果——因为注意力权重只取决于 token 之间的语义相似度，而不取决于它们的位置。位置编码的作用就是**注入位置信息**，让模型能够区分"猫追老鼠"和"老鼠追猫"。

#### 3.7.2 正弦/余弦位置编码的数学原理

论文使用正弦/余弦函数生成位置编码：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

其中 pos 是位置索引，i 是维度索引。

**个人理解**：这个公式的精妙之处在于，不同维度使用不同频率的正弦波。低维度（i 较小）使用高频波（周期短），相邻位置的编码差异大，用于区分邻近位置；高维度（i 较大）使用低频波（周期长），编码值变化平缓，用于编码全局位置信息。这样，每个位置都获得了一个独特的"位置指纹"——一个由不同频率正弦波叠加而成的 d_model 维向量。

#### 3.7.3 为什么用正弦/余弦编码？——深入分析

**1. 数值稳定**：正弦波把值锁定在 [-1, 1] 之间，不管位置多大，数值永远有限。如果使用绝对位置值（如直接用 pos 作为编码），数值会随着位置增大而无限增长，导致训练不稳定。

**2. 相对位置推理**：这是最关键的数学性质。可以证明，位置 pos+k 的编码可以由位置 pos 的编码通过一个线性变换得到。具体来说，存在一个旋转矩阵 T(k)，使得 PE(pos+k) = T(k) × PE(pos)。这意味着模型不需要死记每一个绝对位置，而是天然能算出"两个词隔了几个位置"。注意力机制可以自动学习到：距离越近，关联越强。

**3. 外推性**：由于正弦函数是周期性的，模型可以处理比训练时更长的序列。即使训练时只见过长度为 100 的序列，推理时也可以处理长度为 200 的序列——因为位置 101 的编码可以通过正弦函数直接计算得到。这与可学习位置嵌入（Learned PE）形成对比，后者无法处理训练时未见过的位置。

**4. 不干扰语义**：编码有界、尺度匹配，与词嵌入相加时不会破坏语义信息。词嵌入乘以 √d_model 后与位置编码处于同一数量级，两者相加时语义信息和位置信息都能得到保留。

#### 3.7.4 正弦编码 vs 可学习编码

论文的消融实验（Table 3）显示，正弦编码（BLEU 25.8）与可学习编码（BLEU 25.7）性能几乎相同。作者选择正弦版本的主要理由是外推能力——虽然论文没有实验验证这一点，但后续的研究（如 ALiBi、RoPE）进一步探索了位置编码的外推性，证实了正弦编码在这方面的优势。

### 3.8 Mask（掩码机制）（结合个人理解）

Transformer 中使用两种掩码：

**1. Padding Mask（填充掩码）**

标记输入序列中的填充位置（`<pad>`），使注意力机制忽略这些无效位置。在批处理中，不同长度的序列被填充到相同长度，填充位置不包含有效信息，模型不应该关注它们。

**实现方式**：将填充位置的注意力分数设置为 -1e9（负无穷），经过 softmax 后这些位置的概率趋近于 0。

**2. Subsequent Mask（未来位置掩码）**

在 Decoder 的自注意力中使用，构造一个下三角矩阵，确保每个位置只能关注当前位置及之前的位置，防止"看到未来的词"。

**个人理解**：Subsequent Mask 是保证自回归生成正确性的关键。在训练时，Decoder 一次性接收整个目标序列（教师强制，Teacher Forcing），但如果没有掩码，每个位置都可以"偷看"后面的词——这相当于考试时提前看到了答案。Subsequent Mask 强制每个位置只能利用当前位置及之前的信息，保证了训练和推理时行为一致。

**两种掩码的合并**：在 Decoder 中，Padding Mask 和 Subsequent Mask 通过 `&` 运算合并，同时屏蔽填充位置和未来位置。

**个人理解**：掩码机制体现了 Transformer 设计中的一个重要原则——**信息流控制**。模型不是无限制地访问所有信息，而是通过掩码精确控制每个位置可以看到哪些信息。Encoder 可以看到整个源序列（全连接），Decoder 的自注意力只能看到已生成的部分（下三角），交叉注意力只能看到源序列的有效部分（排除填充）。这种精细的信息流控制是 Transformer 成功的关键之一。

---

## 4. 代码结构说明

### 4.1 项目文件结构

```
MyTransformer/
├── config.py              # 配置文件（模型参数、训练参数、超参数实验配置）
├── model.py               # Transformer 模型核心实现
├── dataset.py             # 数据集加载与预处理
├── train.py               # 训练脚本
├── test.py                # 测试与评估脚本
├── analysis.py            # 参数分析与超参数对比实验
├── generate_report.py     # 自动生成实验报告
├── main.py                # 主入口（顺序执行所有脚本）
├── utils.py               # 工具函数（参数统计、绘图、评估）
├── checkpoint.py          # 模型断点保存与加载
├── requirements.txt       # 依赖包列表
├── data/
│   └── Multi30K/          # 数据集（德语→英语平行语料）
│       ├── train/         # 训练集
│       ├── valid/         # 验证集
│       └── test/          # 测试集
├── checkpoints/           # 模型权重保存
├── figures/               # 实验图表输出
├── results/               # 实验报告输出
├── papers/                # 论文原文与笔记
├── poster/                # 海报
└── PPT/                   # 演示文稿
```

### 4.2 各模块功能说明

| 文件 | 功能 | 关键类/函数 |
|------|------|------------|
| `config.py` | 集中管理所有超参数，提供对比实验配置 | `Config`, `get_experiment_configs()` |
| `model.py` | Transformer 模型完整实现 | `Embedding`, `PositionalEncoding`, `MultiHeadAttention`, `FeedForward`, `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, `Transformer` |
| `dataset.py` | 数据加载、词表构建、批处理 | `TranslationDataset`, `build_vocab()`, `collate_fn()` |
| `train.py` | 模型训练与验证 | `train_epoch()`, `validate()`, `main()` |
| `test.py` | 模型测试、参数分析、预测样例 | `greedy_decode()`, `main()` |
| `analysis.py` | 参数分析、超参数对比实验 | `analyze_model_parameters()`, `run_experiments()` |
| `utils.py` | 工具函数集合 | `count_parameters()`, `plot_loss()`, `evaluate_model()` |
| `checkpoint.py` | 模型断点管理 | `save_checkpoint()`, `load_checkpoint_for_train()` |

### 4.3 数据流

```
原始文本 → dataset.py (分词、建词表、转索引) → DataLoader (批处理、填充)
    → model.py (Embedding → PositionalEncoding → Encoder → Decoder → Linear)
    → 输出 logits → CrossEntropyLoss → 反向传播 → 参数更新
```

---

## 5. 核心模块分析

### 5.1 Embedding（嵌入层）

```python
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, pad_idx):
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        return self.emb(x) * self.scale
```

**设计要点**:
- 使用 `padding_idx` 确保填充标记的嵌入向量不会被更新
- 嵌入向量乘以 √d_model，目的是对齐词嵌入与位置编码的数值量级
- 位置编码取值固定在 [-1, 1]，若不缩放，位置编码会掩盖词向量的语义信息

#### 代码注释问答：为什么要缩放嵌入向量？

**问题**：为什么嵌入向量要乘以 √d_model 这个缩放因子？

**解答**：引入嵌入向量缩放因子 √d_model，目的是对齐词嵌入与正弦位置编码的数值量级。正弦位置编码的取值固定在 [-1, 1] 区间，而原始词嵌入的模长会随着模型维度增大而偏小。如果不进行缩放，位置编码的数值会相对过大，从而掩盖词向量本身的语义信息。以 √d_model 缩放后，两者尺度均衡，可以直接逐元素相加，既能保留词向量的语义特征，又能有效融入位置信息，避免互相压制。

**问题**：为什么缩放因子恰好是 √d_model？

**解答**：这涉及到嵌入向量初始化的数学性质。在 PyTorch 的 `nn.Embedding` 中，嵌入向量默认使用均匀分布初始化，其方差与 d_model 相关。√d_model 的缩放使得词嵌入和位置编码的数值范围大致匹配。更深层的数学分析需要涉及随机矩阵理论和信息论，这里不做展开。

---

### 5.2 PositionalEncoding（位置编码）

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

**设计要点**:
- 使用 `register_buffer` 将位置编码注册为模型的一部分但不参与训练
- 不同维度使用不同频率的正弦/余弦波，形成独特的"位置指纹"
- 偶数维度用 sin，奇数维度用 cos

#### 代码注释问答：为什么用频率变化来表示位置编码？

**问题1：为什么不用绝对大小来表示位置？**

**解答**：如果直接用绝对位置值（如 0, 1, 2, 3, ...）作为编码，会造成数值爆炸——位置 1000 的编码值远大于位置 1，导致训练不稳定。正弦波把值稳稳锁死在 [-1, 1] 之间，不管位置多大，数值永远有限，训练稳定。

**问题2：正弦/余弦编码如何实现相对位置推理？**

**解答**：这是 Transformer 最关键的设计之一。位置 pos+k 的编码，可以由位置 pos 的编码通过线性变换直接得到。具体来说，存在一个旋转矩阵 T(k)，使得 PE(pos+k) = T(k) × PE(pos)。这意味着模型不需要死记每一个绝对位置，而是天然能算出"两个词隔了几个位置"。注意力机制可以自动学习到：距离越近，关联越强。

**问题3：什么是外推性？**

**解答**：外推性指模型能处理比训练集更长的序列。由于正弦函数是周期性的，推理时即使遇到训练时未见过的位置，也可以通过正弦函数直接计算其编码。这使得 Transformer 在推理时能处理比训练集更长的句子。

**问题4：正弦编码如何不干扰词嵌入语义？**

**解答**：正弦编码有界（[-1, 1]）、尺度匹配（与缩放后的词嵌入处于同一数量级），刚好微弱叠加位置信息，不破坏原词语义。如果位置编码的数值过大，会掩盖词嵌入的语义信息；如果过小，则无法有效传递位置信息。√d_model 的缩放确保了这种微妙的平衡。

---

### 5.3 Scaled Dot-Product Attention（缩放点积注意力）

```python
def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn = attn.masked_fill(~mask, -1e9)
    attn = F.softmax(attn, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    output = torch.matmul(attn, v)
    return output, attn
```

**维度变化**:
```
q: (batch, heads, seq_q, d_k)
k: (batch, heads, seq_k, d_k) → 转置 → (batch, heads, d_k, seq_k)
attn = q × k^T: (batch, heads, seq_q, seq_k)  ← 注意力分数矩阵
attn = softmax(attn): (batch, heads, seq_q, seq_k)  ← 注意力权重
output = attn × v: (batch, heads, seq_q, d_k)  ← 加权求和结果
```

**代码逻辑详解**:
1. `q.size(-1)` 获取每个头的维度 d_k
2. `torch.matmul(q, k.transpose(-2, -1))` 计算查询与所有键的点积，得到注意力分数矩阵。其中 `k.transpose(-2, -1)` 将 k 的最后两维转置，使得矩阵乘法能够正确计算每个查询与每个键的点积
3. `math.sqrt(d_k)` 缩放因子，防止点积结果过大导致 softmax 梯度消失
4. `masked_fill(~mask, -1e9)` 将需要屏蔽的位置填充为 -1e9（负无穷），使这些位置在 softmax 后概率趋近于 0
5. `F.softmax(attn, dim=-1)` 在最后一个维度（seq_k 维度）上做 softmax，得到每个查询对所有键的注意力权重分布
6. `torch.matmul(attn, v)` 用注意力权重对值做加权平均，得到最终的注意力输出

---

### 5.4 Multi-Head Attention（多头注意力）

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads          # 专家个数，比如 8
        self.d_k = d_model // n_heads   # 每个专家看的维度，512/8=64
        self.wq = nn.Linear(d_model, d_model)  # 生成"查询"的线性变换
        self.wk = nn.Linear(d_model, d_model)  # 生成"键"的线性变换
        self.wv = nn.Linear(d_model, d_model)  # 生成"值"的线性变换
        self.fc = nn.Linear(d_model, d_model)  # 最后的融合层，把8份报告揉在一起

    def forward(self, q, k, v, mask=None, dropout=None):
        batch_size = q.size(0)
        # 1. 线性投影：将 q/k/v 映射到 d_model 维空间
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # 2. 切分多头：把 d_model 维均分成 n_heads 个 d_k 维的子空间
        q = q.view(batch_size, -1, self.n_heads, self.d_k)
        k = k.view(batch_size, -1, self.n_heads, self.d_k)
        v = v.view(batch_size, -1, self.n_heads, self.d_k)
        # 3. 交换维度：把 seq_len 和 n_heads 互换
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 4. 并行计算注意力
        out, attn = scaled_dot_product_attention(q, k, v, mask, dropout)
        # 5. 多头拼接：把 n_heads 个头的输出拼接回 d_model 维
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        # 6. 通过线性层融合多头信息
        return self.fc(out)
```

#### 代码注释问答：为什么采用先整体投影再切分的实现方式？

**问题**：为什么不独立构建多个小头的投影矩阵，而是先整体投影再切分？

**解答**：这种实现方式有三个优势：

1. **计算并行性与硬件吞吐量**：合并为批量矩阵运算，充分利用 GPU 的并行加速能力。如果独立构建 8 个小头，需要执行 8 次独立的矩阵乘法；而先整体投影再切分只需要 3 次矩阵乘法（Wq、Wk、Wv），计算效率大幅提升。

2. **数学表达等价性**：切分与独立投影在数学上是等价的。整体投影矩阵 Wq 的形状为 (d_model, d_model)，切分后每个头的投影矩阵 Wq_i 的形状为 (d_model, d_k)，且 Wq = [Wq_1, Wq_2, ..., Wq_h]（按列拼接）。因此，整体投影再切分与独立投影再拼接得到完全相同的结果。

3. **批处理效率与内存局部性**：统一张量布局，减少内存碎片与 I/O 开销，优化计算流水线。GPU 在处理连续内存块时效率最高，整体投影保证了张量在内存中的连续性。

#### 代码注释问答：为什么划分低维子空间而非单头直接使用完整维度？

**问题**：为什么不直接用单头注意力（d_k = d_model），而是划分成多个低维子空间？

**解答**：这有三个层面的考虑：

1. **计算效率**：在参数量约束下，单头注意力的计算复杂度为 O(n·d_model²)，而多头注意力的计算复杂度为 O(n·d_model²/h)。虽然总参数量相同（因为 d_k = d_model / h），但每个头的计算量降低，整体计算效率提升。

2. **表征多样性**：迫使各头学习差异化特征，避免表征坍塌与冗余。每个头被映射到不同的子空间，因此会关注不同的特征模式——有的捕捉语法结构，有的捕捉语义关系，有的捕捉指代消解。这种多视角的信息提取方式比单头更全面。

3. **集成学习**：多头独立捕捉不同模式，融合后增强模型鲁棒性与泛化能力。每个头相当于一个"弱学习器"，多个弱学习器的集成效果通常优于单个强学习器。

---

### 5.5 FeedForward（前馈网络）

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))
```

#### 代码注释问答：为什么加入 FFN 前馈网络？

**问题1：注意力机制已经能捕捉全局依赖了，为什么还需要 FFN？**

**解答**：注意力机制本质是线性加权求和——虽然能捕捉全局依赖，但非线性拟合能力有限。加入两层全连接层 + ReLU 激活，让模型学习更复杂的特征交互与高阶语义信息。可以这样理解：注意力层负责"收集原材料"（全局信息），FFN 层负责"深度加工"（特征提纯）。

**问题2：为什么 FFN 是逐位置独立处理的？**

**解答**：每个 token 独立通过同一套 MLP，不产生跨位置信息流动。跨位置的信息交互完全由注意力机制负责，职责分工明确。这种"通信-计算"分离的设计让模型更易训练、更稳定——注意力层负责"通信"（哪些词需要交换信息），FFN 层负责"计算"（对每个词的信息进行深度处理）。

**问题3：为什么采用先升维再降维的结构？**

**解答**：中间层将维度扩大到 4 倍（如 512→2048），大幅增强特征表达空间，类似于 SVM 中的核技巧——在高维空间中，数据更容易被线性分离。最后映射回原始维度，保证与下一层网络结构兼容。

---

### 5.6 Encoder Layer（编码器层）

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_out = self.attn(x, x, x, mask, self.dropout)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x
```

#### 代码注释问答：残差连接和层归一化的原理是什么？

**问题1：残差连接（Residual Connection）的核心作用是什么？**

**解答**：残差连接的公式为 `x_out = x + Sublayer(x)`。其核心作用是为梯度开辟"高速公路"，解决深层网络梯度消失/爆炸问题。反向传播时，梯度 = 1 + 子层梯度——即使子层梯度趋近于 0，恒等映射的 "+1" 也能保证梯度正常传递。这让模型可以安全堆叠数十甚至上百层，不会退化、不会优化困难。

**问题2：层归一化（Layer Normalization）的作用是什么？**

**解答**：层归一化的作用是稳定训练、加速收敛、解决内部协变量偏移（Internal Covariate Shift）。它对每个样本独立做归一化（不受 batch 大小和序列长度影响），把每层输出拉回均值 0、方差 1 的稳定分布，允许使用更大的学习率，收敛更快。

**问题3：为什么用 LayerNorm 而不是 BatchNorm？**

**解答**：BatchNorm 按批次维度归一化，不适合 NLP 中的变长序列——不同长度的序列需要不同的统计量，填充位置会引入噪声。LayerNorm 按样本/Token 维度归一化，对每个 token 独立计算均值和方差，天然适合 Transformer。

**问题4：残差 + 归一化的组合方式是什么？**

**解答**：本代码使用 Post-LN（Post-Layer Normalization）方式，公式为：
```
x = LayerNorm(x + Dropout(Sublayer(x)))
```
流程为：子层计算 → 残差相加 → Dropout → 层归一化。残差保证梯度/信息畅通，归一化清理分布偏移，两者结合是 Transformer 能堆叠深层的关键。

---

### 5.7 Decoder Layer（解码器层）

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        # 带掩码的多头自注意力
        attn_out = self.self_attn(x, x, x, tgt_mask, self.dropout)
        x = self.norm1(x + self.dropout(attn_out))
        # 交叉注意力：Q来自Decoder，K和V来自Encoder
        cross_out = self.cross_attn(x, enc_out, enc_out, src_mask, self.dropout)
        x = self.norm2(x + self.dropout(cross_out))
        # 前馈
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x
```

**Decoder 与 Encoder 的关键区别**：
1. **两个注意力层**：Decoder 包含 Masked Self-Attention 和 Cross-Attention，而 Encoder 只有 Self-Attention
2. **Masked Self-Attention**：使用 Subsequent Mask 防止看到未来位置，保证自回归生成
3. **Cross-Attention**：Q 来自 Decoder 的当前状态，K 和 V 来自 Encoder 的输出，实现"查看源语言信息"
4. **三个 LayerNorm**：Decoder 有三个子层（Self-Attention、Cross-Attention、FFN），因此有三个 LayerNorm

---

### 5.8 Mask 生成

```python
def create_pad_mask(seq, pad_idx):
    # seq: (batch_size, seq_len)
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
# 维度变化：
# (seq != pad_idx)        : (batch_size, seq_len)          原始布尔掩码
# unsqueeze(1)            : (batch_size, 1, seq_len)       扩展多头维度
# unsqueeze(2)            : (batch_size, 1, 1, seq_len)    匹配注意力矩阵形状
# 最后两个 1 用于广播对齐注意力分数 (batch, heads, seq_q, seq_k)
# seq_q 维度自动广播，seq_k 维度指定需要屏蔽的 key 位置


def create_subseq_mask(seq):
    # seq: (batch_size, seq_len)
    batch_size, seq_len = seq.size()
    # 创建一个下三角矩阵（包括主对角线）
    mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device)).bool()
    # 添加维度以匹配注意力分数矩阵形状
    mask = mask.unsqueeze(0).unsqueeze(0)
    # 扩展到整个 batch
    return mask.expand(batch_size, -1, -1, -1)
```

**维度变换说明**：
- `create_pad_mask` 的输出形状为 `(batch_size, 1, 1, seq_len)`，通过广播机制对齐注意力分数矩阵 `(batch, heads, seq_q, seq_k)`。其中 seq_q 维度自动广播，seq_k 维度指定需要屏蔽的 key 位置。
- `create_subseq_mask` 的输出形状为 `(batch_size, 1, seq_len, seq_len)`，下三角区域为 True，上三角区域为 False。

**掩码合并**：两种掩码通过 `&` 运算合并，同时屏蔽填充位置和未来位置。在 `scaled_dot_product_attention` 中，通过 `masked_fill(~mask, -1e9)` 将需要屏蔽的位置填充为 -1e9（负无穷），使得这些位置在 softmax 后对应的概率趋近于 0，从而实现屏蔽效果。

---

## 6. 数据集介绍

### 6.1 Multi30K 数据集

本项目使用 **Multi30K** 数据集，这是一个专门用于多模态机器翻译研究的基准数据集。

| 属性 | 说明 |
|------|------|
| **任务** | 德语（DE）→ 英语（EN）翻译 |
| **训练集** | 29,000 句对 |
| **验证集** | 1,014 句对 |
| **测试集** | 1,000 句对 |
| **领域** | Flickr30K 图片描述（描述日常场景的短句） |
| **句子长度** | 通常 5-20 个词 |

### 6.2 数据预处理

预处理流程：
1. **读取文件**: 按行读取 `.de` 和 `.en` 文件，每行一个句子
2. **分词**: 按空格切分为 token 列表
3. **词表构建**: 统计词频，保留最常见的 8000 个词，其余映射为 `<unk>`
4. **特殊标记**: 添加 `<pad>`(0)、`<sos>`(1)、`<eos>`(2)、`<unk>`(3)
5. **句子截断**: 过滤长度超过 max_len-2 的句子（留出 `<sos>` 和 `<eos>` 的位置）
6. **动态填充**: 同一 batch 内的句子填充到相同长度

### 6.3 词表统计

| 语言 | 词表大小 | 覆盖情况 |
|------|---------|---------|
| 德语（源语言） | ~8000 | 覆盖训练集中绝大多数词 |
| 英语（目标语言） | ~8000 | 覆盖训练集中绝大多数词 |

---

## 7. 实验设置

### 7.1 硬件环境

| 项目 | 内容 |
|------|------|
| 操作系统 | Windows 11 |
| Python 版本 | 3.x |
| PyTorch 版本 | ≥ 1.9.0 |
| 计算设备 | NVIDIA GeForce RTX 4060 Laptop GPU |
| 显存 | 8GB |

### 7.2 默认模型配置（Base Model）

| 参数 | 值 | 说明 |
|------|:---:|------|
| d_model | 512 | 嵌入维度 |
| n_layers | 6 | Encoder/Decoder 层数 |
| n_heads | 8 | 多头注意力头数 |
| d_ff | 2048 | 前馈网络隐藏层维度 |
| dropout | 0.1 | Dropout 比率 |
| batch_size | 32 | 批次大小 |
| learning_rate | 1e-4 | 学习率 |
| epochs | 20 | 训练轮数 |
| src_vocab_size | 8000 | 源语言词表大小 |
| tgt_vocab_size | 8000 | 目标语言词表大小 |
| max_len | 5000 | 最大序列长度 |
| pad_idx | 0 | 填充标记索引 |

### 7.3 优化器与损失函数

- **优化器**: Adam（β₁=0.9, β₂=0.98, ε=1e-9，与论文一致）
- **损失函数**: CrossEntropyLoss（忽略填充标记 `pad_idx=0`）
- **梯度裁剪**: max_norm=1.0

### 7.4 超参数对比实验设计

为了系统分析各超参数的影响，我们设计了 7 组对比实验，每组只改变一个超参数：

| 实验组 | 变化参数 | 取值 |
|-------|---------|------|
| 1. d_model | 嵌入维度 | 128, 256, 512 |
| 2. n_heads | 注意力头数 | 2, 4, 8 |
| 3. n_layers | 层数 | 2, 4, 6 |
| 4. batch_size | 批次大小 | 16, 32, 64 |
| 5. lr | 学习率 | 1e-5, 1e-4, 5e-4 |
| 6. dropout | Dropout 比率 | 0.0, 0.1, 0.3 |
| 7. epochs | 训练轮数 | 5, 10, 20 |

---

## 8. 实验结果

### 8.1 训练过程

使用默认配置（d_model=512, n_layers=6, n_heads=8）训练 20 个 epoch：

| 指标 | 数值 |
|------|:----:|
| 总训练时间 | 1184.1s |
| 平均每轮时间 | 59.2s |
| 峰值显存占用 | 2293 MB |
| 初始训练 Loss | 4.1171 |
| 最终训练 Loss | 0.6049 |
| 最佳验证 Loss | 2.1719 |
| 模型参数量 | 56,434,496 |

**Loss 曲线分析**: 训练 Loss 从 4.12 持续下降到 0.60，说明模型在训练集上学习效果良好。验证 Loss 在约 8 个 epoch 后趋于平稳（约 2.17），之后出现轻微上升趋势，表明模型开始过拟合。

### 8.2 验证集评估

| 指标 | 数值 |
|------|:----:|
| Token 准确率 | 0.5970 (7870/13182) |
| 句子准确率 | 0.0256 (26/1015) |
| 精确率 (Macro) | 0.0712 |
| 召回率 (Macro) | 0.0788 |
| F1 分数 (Macro) | 0.0748 |
| 验证 Loss | 2.6644 |

### 8.3 测试集评估

| 指标 | 数值 |
|------|:----:|
| Token 准确率 | 0.5948 (7659/12877) |
| 句子准确率 | 0.0220 (22/1000) |
| 精确率 (Macro) | 0.0543 |
| 召回率 (Macro) | 0.0563 |
| F1 分数 (Macro) | 0.0553 |
| 测试 Loss | 2.2101 |

### 8.4 预测样例

| 样例 | 源语言 (DE) | 参考译文 (EN) | 模型预测 |
|:----:|:-----------:|:-------------:|:--------:|
| 1 | Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt. | A man in an orange hat starring at something. | A man in a hat \<unk\> something with an orange machine. |
| 2 | Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun. | A Boston Terrier is running on lush green grass in front of a white fence. | A \<unk\> is running across grass in front of a white fence. |
| 3 | Ein Mädchen in einem Karateanzug bricht ein Brett mit einem Tritt. | A girl in karate uniform breaking a stick with a front kick. | A girl with a \<unk\> is putting a ball in a chair in a pool. |
| 4 | Fünf Leute in Winterjacken und mit Helmen stehen im Schnee mit Schneemobilen im Hintergrund. | Five people wearing winter jackets and helmets stand in the snow, with snowmobiles in the background. | Five people wearing helmets and helmets are standing in the snow with snow in the background. |
| 5 | Leute Reparieren das Dach eines Hauses. | People are fixing the roof of a house. | People are painting the roof of a house. |

**分析**: 模型能够捕捉句子的基本结构和主要语义（如样例 4 和 5 的翻译基本正确），但在处理罕见词（输出 `<unk>`）、复杂名词短语和细节描述时仍有不足。这主要是因为词表大小限制（8000）和训练数据量有限（29K 句对）。

### 8.5 超参数对比实验结果

#### 嵌入维度 (d_model) 的影响

| 配置 | 参数量 | 最佳 Val Loss | 训练时间 | 显存占用 |
|------|:------:|:-------------:|:--------:|:--------:|
| d_model=128 | 5.86M | 2.7970 | 343.4s | 415MB |
| d_model=256 | 17.21M | 2.3533 | 379.1s | 716MB |
| d_model=512 (base) | 56.43M | 2.1725 | 593.2s | 1511MB |

**结论**: d_model 越大，模型容量越大，验证 Loss 越低，但参数量呈平方级增长，存在边际递减效应。

#### 注意力头数 (n_heads) 的影响

| 配置 | 参数量 | 最佳 Val Loss | 训练时间 |
|------|:------:|:-------------:|:--------:|
| n_heads=2 | 56.43M | 2.1734 | 648.3s |
| n_heads=4 | 56.43M | 2.1795 | 582.6s |
| n_heads=8 (base) | 56.43M | 2.1788 | 590.0s |

**结论**: 在参数量相同的情况下，头数对性能影响不大。这与论文的消融实验结果一致——头数在 4-8 之间性能接近。

#### 层数 (n_layers) 的影响

| 配置 | 参数量 | 最佳 Val Loss | 训练时间 |
|------|:------:|:-------------:|:--------:|
| n_layers=2 | 27.01M | 2.3339 | 251.0s |
| n_layers=4 | 41.72M | 2.2322 | 421.0s |
| n_layers=6 (base) | 56.43M | 2.1785 | 591.1s |

**结论**: 层数越多，模型越深，性能越好，但训练时间和参数量线性增长。

#### 批次大小 (batch_size) 的影响

| 配置 | 参数量 | 最佳 Val Loss | 训练时间 |
|------|:------:|:-------------:|:--------:|
| batch_size=16 | 56.43M | 2.2441 | 986.6s |
| batch_size=32 (base) | 56.43M | 2.1754 | 588.1s |
| batch_size=64 | 56.43M | 2.1497 | 480.0s |

**结论**: 更大的 batch size 训练更快、Loss 更低，但显存占用更高（batch=64 时达 2324MB）。

#### 学习率 (lr) 的影响

| 配置 | 参数量 | 最佳 Val Loss | 训练时间 |
|------|:------:|:-------------:|:--------:|
| lr=1e-5 | 56.43M | 3.0329 | 589.4s |
| lr=1e-4 (base) | 56.43M | 2.1864 | 590.9s |
| lr=5e-4 | 56.43M | 5.2760 | 588.4s |

**结论**: 学习率对训练效果影响极大。lr=1e-4 最佳，lr=5e-4 过大导致不收敛（Loss 高达 5.28），lr=1e-5 过小导致收敛缓慢。

#### Dropout 比率的影响

| 配置 | 参数量 | 最佳 Val Loss | 训练时间 |
|------|:------:|:-------------:|:--------:|
| dropout=0.0 | 56.43M | 2.2840 | 571.9s |
| dropout=0.1 (base) | 56.43M | 2.1674 | 616.8s |
| dropout=0.3 | 56.43M | 2.3433 | 808.6s |

**结论**: 适当的 dropout（0.1）有助于防止过拟合，提升验证集性能。dropout 过大（0.3）则可能阻碍学习。

#### 训练轮数 (epochs) 的影响

| 配置 | 参数量 | 最佳 Val Loss | 训练时间 |
|------|:------:|:-------------:|:--------:|
| epochs=5 | 56.43M | 2.3132 | 359.1s |
| epochs=10 (base) | 56.43M | 2.1603 | 597.5s |
| epochs=20 | 56.43M | 2.1789 | 1179.6s |

**结论**: 10 个 epoch 已基本收敛，继续训练到 20 个 epoch 验证 Loss 不再下降，说明模型已充分训练。

---

## 9. 参数分析

### 9.1 模型参数量详细分解

| 组件 | 参数量 | 占比 |
|------|:------:|:----:|
| Embedding（源+目标） | 8,192,000 | 14.52% |
| Encoder 堆栈（6 层） | 18,914,304 | 33.52% |
| Decoder 堆栈（6 层） | 25,224,192 | 44.70% |
| 输出投影层 | 4,104,000 | 7.27% |
| **总计** | **56,434,496** | **100%** |

### 9.2 Encoder 单层参数组成

| 子组件 | 参数量 | 占比 |
|--------|:------:|:----:|
| 自注意力 (Multi-Head Attention) | 1,048,576 | ~33.3% |
| 前馈网络 (Feed-Forward) | 2,097,152 | ~66.6% |
| 层归一化 (LayerNorm) | 2,048 | ~0.1% |

### 9.3 Decoder 单层参数组成

| 子组件 | 参数量 | 占比 |
|--------|:------:|:----:|
| 自注意力 (Self-Attention) | 1,048,576 | ~25.0% |
| 交叉注意力 (Cross-Attention) | 1,048,576 | ~25.0% |
| 前馈网络 (Feed-Forward) | 2,097,152 | ~50.0% |
| 层归一化 (LayerNorm) | 3,072 | ~0.1% |

### 9.4 不同模型规模参数量变化

| 规模 | d_model | Layers | Heads | d_ff | 参数量 | 模型大小 |
|:----:|:------:|:------:|:-----:|:----:|:------:|:--------:|
| Tiny | 64 | 2 | 2 | 256 | 1.78M | 6.8 MB |
| Small | 128 | 3 | 4 | 512 | 4.47M | 17.0 MB |
| Medium | 256 | 4 | 4 | 1024 | 13.52M | 51.6 MB |
| **Base** | **512** | **6** | **8** | **2048** | **56.43M** | **215.3 MB** |
| Large | 1024 | 6 | 16 | 4096 | 200.94M | 766.5 MB |

### 9.5 参数量计算公式

| 组件 | 公式 | 说明 |
|------|------|------|
| Embedding | vocab_size × d_model | 词嵌入矩阵 |
| Multi-Head Attention | 4 × d_model² | Wq + Wk + Wv + Wo |
| Feed-Forward | 2 × d_model × d_ff | fc1 + fc2 |
| LayerNorm | 2 × d_model | gamma + beta |
| 输出投影 | d_model × tgt_vocab_size | 线性分类层 |

### 9.6 关键发现

1. **参数量主要受 d_model 控制**，呈平方级增长
2. **FFN 是参数大户**，占 Encoder 单层的 66.6%，Decoder 单层的 50%
3. **Decoder 比 Encoder 参数多**（多一个交叉注意力模块）
4. **Embedding 层在词表大时占比显著**
5. **参数量与训练时间近似线性关系**，与显存占用也近似线性

---

## 10. 遇到的问题与解决方法

### 10.1 训练 Loss 不下降

**问题**: 初始训练时 Loss 下降缓慢甚至不下降。

**原因与解决**:
- **学习率过大/过小**: 实验发现 lr=1e-4 最佳，过大（5e-4）导致不收敛，过小（1e-5）收敛缓慢
- **梯度爆炸**: 添加了梯度裁剪（max_norm=1.0），防止梯度爆炸
- **优化器参数**: 使用论文推荐的 Adam 参数（β₁=0.9, β₂=0.98, ε=1e-9）

### 10.2 显存不足（OOM）

**问题**: 训练过程中出现 CUDA Out of Memory 错误。

**原因与解决**:
- **batch_size 过大**: 将 batch_size 从 64 调整为 32
- **序列过长**: 在数据预处理中过滤了过长的句子
- **梯度累积**: 对于显存有限的场景，可以考虑使用梯度累积

### 10.3 模型过拟合

**问题**: 训练 Loss 持续下降但验证 Loss 不再下降甚至上升。

**原因与解决**:
- **Dropout 不足**: 将 dropout 从 0.0 调整为 0.1
- **训练轮数过多**: 通过早停（Early Stopping）在验证 Loss 不再下降时停止训练
- **数据量有限**: Multi30K 仅有 29K 句对，模型容量（56M 参数）相对数据量偏大

### 10.4 预测中出现大量 `<unk>`

**问题**: 模型预测结果中出现大量未知词标记 `<unk>`。

**原因与解决**:
- **词表大小限制**: 词表仅 8000 词，训练集中出现频率较低的词被映射为 `<unk>`
- **BPE 分词**: 可以考虑使用 Byte-Pair Encoding（BPE）子词分词，减少未登录词
- **增大词表**: 可以增大词表大小到 16000 或 32000

### 10.5 句子准确率低

**问题**: 句子准确率仅约 2-3%，远低于 Token 准确率（约 60%）。

**原因**: 这是机器翻译任务的常见现象。生成一个句子时，只要有一个词翻译错误，整个句子就被判定为错误。60% 的 Token 准确率意味着平均每个 20 词的句子有 8 个词正确，但完全正确的概率仅为 0.6²⁰ ≈ 0.0036%，与观察到的 2-3% 基本吻合。

---

## 11. 小组分工

| 成员 | 主要负责内容 |
|------|------------|
| 胡亚鑫 | 论文精读、Transformer 原理解析、模型架构设计 |
| 胡亚鑫 | 代码实现（model.py、config.py）、模型训练与调试 |
| 赵子仪 | 数据处理（dataset.py）、词表构建、数据预处理 |
| 赵子仪 | 实验设计与分析（analysis.py）、超参数对比实验 |
| 赵子仪 | 报告撰写、PPT 制作、结果可视化 |
| 胡亚鑫| 测试评估（test.py）、预测样例分析、海报制作 |

---

## 12. 总结与收获

### 12.1 对 Transformer 的理解

通过本次项目，我们深入理解了 Transformer 的核心设计思想：

1. **注意力机制是核心**: Transformer 证明了仅靠注意力机制就能构建强大的序列模型，无需循环或卷积
2. **并行化是关键优势**: 抛弃 RNN 的顺序计算，使训练速度大幅提升
3. **多头注意力是创新**: 从多个子空间联合关注信息，比单头注意力更强大
4. **位置编码是必要补充**: 为并行模型注入序列顺序信息
5. **残差连接和层归一化是深层网络的基石**: 保证 6 层甚至更深网络的稳定训练

### 12.2 实验收获

1. **理论到实践的完整闭环**: 从阅读论文到代码实现，再到实验验证，完成了完整的深度学习项目流程
2. **超参数调优经验**: 系统对比了 d_model、n_heads、n_layers、batch_size、lr、dropout 等超参数的影响
3. **模型分析能力**: 学会了从参数量、计算量、显存占用等多个维度分析模型
4. **问题排查能力**: 遇到了 Loss 不下降、显存不足、过拟合等问题，并找到了相应的解决方法

### 12.3 Transformer 的深远影响

Transformer 自 2017 年提出以来，已经彻底改变了深度学习领域：

- **NLP 领域**: BERT、GPT、T5 等模型均基于 Transformer
- **计算机视觉**: Vision Transformer (ViT) 将 Transformer 应用于图像分类
- **多模态**: CLIP、DALL-E 等模型使用 Transformer 处理图文数据
- **大语言模型**: ChatGPT、GPT-4、Claude 等均基于 Transformer Decoder 架构

可以说，Transformer 是深度学习史上最具影响力的架构之一，而"Attention Is All You Need"这篇论文则是这一革命的起点。

### 12.4 未来改进方向

1. **使用更大的数据集**: 在 WMT 2014 等更大规模的数据集上训练
2. **使用 BPE 分词**: 减少未登录词，提高翻译质量
3. **实现 Beam Search**: 替代贪婪解码，提升生成质量
4. **使用学习率预热**: 实现论文中的 warmup + 衰减学习率调度
5. **尝试 Label Smoothing**: 论文中使用的正则化技术
6. **评估 BLEU 分数**: 使用标准机器翻译评估指标

---

## 13. 参考文献与参考项目

### 参考文献

1. **Vaswani et al., *Attention Is All You Need*, NIPS 2017**  
   arXiv: 1706.03762 — 本项目核心论文

2. **Bahdanau et al., *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015**  
   提出注意力机制用于机器翻译的先驱工作

3. **Gehring et al., *Convolutional Sequence to Sequence Learning*, ICML 2017**  
   基于 CNN 的序列转录模型（ConvS2S），本文的主要对比基线

4. **Ba et al., *Layer Normalization*, arXiv 2016**  
   Transformer 使用的归一化方法

5. **Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019**  
   Transformer Encoder 在 NLU 任务上的里程碑

6. **Radford et al., *Improving Language Understanding by Generative Pre-Training*, 2018**  
   Transformer Decoder 在语言生成上的开创工作（GPT）

7. **Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021**  
   Transformer 在视觉领域的扩展（ViT）

8. **Brown et al., *Language Models are Few-Shot Learners*, NIPS 2020**  
   GPT-3，展示 Transformer 规模扩展的威力

### 参考项目与代码

1. **Tensor2Tensor (tensorflow/tensor2tensor)**  
   论文官方实现，Google 发布的 TensorFlow 工具库

2. **The Annotated Transformer (Harvard NLP)**  
   http://nlp.seas.harvard.edu/2018/04/03/attention.html  
   逐行注释的 Transformer PyTorch 实现，本项目的重要参考

3. **PyTorch 官方 Transformer 文档**  
   https://pytorch.org/docs/stable/nn.html#transformer-layers

4. **Multi30K 数据集**  
   https://github.com/multi30k/dataset

