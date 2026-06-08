---
type: paper
tier: deep
title: "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
arxiv_id: "2405.21060"
source: "Sources/Mamba-2.pdf"
authors: [Tri Dao, Albert Gu]
year: 2024
venue: "ICML 2024"
tags: [SSM, state-space-model, selective-SSM, sequence-modeling, architecture, linear-complexity, structured-matrix, semiseparable, attention, linear-attention, hardware-efficient, tensor-parallel, language-model]
concepts: ["[[CodecLanguageModel]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 1
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 0 个已确认实体页直接命中; 1 个间接参考: [[CodecLanguageModel]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。本 vault 以 TTS 为核心,不含 SSM/state-space 专属概念页,KB 命中有限。
> 检索命中: 无直接命中 | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无
>
> **谱系定位**: Mamba-2 是 [[论文笔记/Mamba|Mamba]] 的直接后继,由同一组作者(Gu & Dao, CMU/Princeton/Cartesia)提出。核心改进方向是将 Mamba 的 selective SSM 与 attention 的理论和工程生态统一,形成 structured state space duality (SSD) 框架。在本 vault 中,[[论文笔记/MamTra|MamTra]] 已直接使用 Mamba-2 layer 替换 CosyVoice 2 的 Transformer 层,实现 TTS backbone 的效率优化(VRAM 降 34%,RTF 降 25%);[[论文笔记/SpeechSpeculativeDecoding|SpeechSpeculativeDecoding]] 则将 Mamba-2 架构用于 draft model。
>
> **本文定位**: Mamba-2 不是 TTS 论文,而是通用序列建模架构的理论+系统工作。其 SSD 框架揭示了 SSM 与 attention 在数学结构上的深层等价性(通过半可分矩阵),并由此设计出比 Mamba-1 快 2-8x 的硬件高效算法。对 TTS 领域的意义在于:提供了更实用的 Mamba 变体(更大状态空间、tensor parallel 友好),使 Mamba 作为 Transformer 替代 backbone 进入 TTS 生产系统成为可能。

## 速查

> [!summary] 速查
> - **一句话**: 通过证明 selective SSM 与 structured masked attention 是半可分矩阵变换的对偶形式,设计出兼具 SSM 线性复杂度和 attention 硬件友好性的 SSD 算法与 Mamba-2 架构
> - **路线**: 输入 u → 并行线性投影得 (A, X, B, C) → 1D Conv(X) → SSD layer(半可分矩阵分块乘法: chunk 内用二次 attention 形式,chunk 间用线性 SSM 递推) → 门控 → GroupNorm → 线性投影输出
> - **指标**: LM: Mamba-2 2.7B Pile ppl 6.09 (vs Mamba-2.8B 6.22, Transformer++ 6.13); SSD 速度 2-8x > Mamba scan, seq_len >= 2K 时快于 FlashAttention-2; 状态维度 N=256 几乎不减速 [Table 1, Fig 10]
> - **可借鉴**: (1) 分块矩阵乘法统一 linear/quadratic 形式是通用算法设计范式; (2) 多头模式(MVA/MQA/MKA/MHA)移植到 SSM 的系统方法论; (3) 混合架构(~10% attention 层)在纯 SSM 和纯 Transformer 之间取最优; (4) Tensor Parallel 设计(并行投影 A/B/C/X + GroupNorm)
> - **局限**: A 矩阵限制为 scalar-identity(比 Mamba-1 的 diagonal 表达力弱); 仍未验证 10B+ 规模; 不含 softmax 因此不完全泛化 standard attention; 音频/语音实验缺失(仅 LM)

## 核心问题

Mamba (selective SSM) 在语言建模上匹配 Transformer,但存在两个系统性问题 [§1]:

1. **训练速度**: Mamba 的 selective scan 算法无法利用 GPU 的矩阵乘法单元(tensor cores),因为其核心计算是标量级的逐步递推。即便有 hardware-aware CUDA kernel,仍慢于利用 matmul 单元的 FlashAttention。
2. **生态隔离**: SSM 的开发与 Transformer 生态(理论理解、硬件优化、并行训练技术)完全割裂。研究者难以在 SSM 上复用 Transformer 社区积累的 tensor parallelism、sequence parallelism、multi-head 设计等工程成果。

**本文的策略**: 不是"改进 SSM 使其更快",而是从数学上证明 SSM 和 attention 是同一类结构化矩阵变换的两种计算形式,从而将 Transformer 的算法和系统优化直接迁移到 SSM。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Mamba-2 的架构是在 Mamba block 基础上做两个关键修改 [§7.1, Fig 6]:

1. **并行投影 (Parallel Projections)**: Mamba-1 中 A/B/C 是 X 的函数(先投影得 X,再从 X 投影得 A/B/C,串行)。Mamba-2 中 A/B/C/X 全部从原始输入 u 并行投影得到,类比 Transformer 中 Q/K/V 并行投影。[论文原文] 这不仅减少参数,更关键的是消除了 tensor parallelism 的额外同步点 [§7.1, §8.1]。

2. **额外归一化层 (Extra Normalization)**: 在门控乘法之后、最终输出投影之前加 GroupNorm。[论文原文] 大规模训练时(>1B)可显著改善稳定性,类似 NormFormer [§7.1]。GroupNorm 的组数设为 TP degree 的倍数,确保 tensor parallel 时无需跨 GPU 通信。

内层序列变换从 Mamba-1 的 selective scan (S6) 替换为 SSD 算法。

### 关键设计选择

#### 理论框架: State Space Duality (SSD)

**核心定理链**:

**Step 1: SSM = 半可分矩阵** (Theorem 3.5) [§3]:
任何 state size 为 N 的 SSM 变换 y = SSM(A, B, C)(x) 等价于乘以一个 N-半可分矩阵 M:
```
M_{ji} = C_j^T * A_j * ... * A_{i+1} * B_i    (j >= i)
```
半可分矩阵的定义性质: 对角线及以下的任何子矩阵秩 <= N [Definition 3.1]。[论文原文] 这个等价是精确的,SSM/SSS/SS 三个缩写恰好可互换指代 state space model 或 semiseparable matrix [§3.3]。

**Step 2: Structured Masked Attention 泛化 Linear Attention** (Definition 4.2) [§4]:
标准 masked attention 的计算 Y = (L ◦ QK^T) · V 可视为四路张量缩并。线性 attention 的"快"来自换缩并顺序,但其本质只是 L = 全 1 因果掩码的特例。[论文原文] 任何结构化矩阵 L 都可以同样加速 — 只需 L 的矩阵-向量乘法是亚二次的 [§4.3]。

**Step 3: Scalar SSM = 1-SS Structured Masked Attention** (SSD) [§5]:
当 SSM 的 A 矩阵是 scalar × identity(标量结构)时:
- SSM 的二次(朴素)形式恰好 = 结构化掩码 attention(L 为 1-半可分矩阵,L_{ji} = a_j * ... * a_{i+1})
- SSM 的线性(递推)形式恰好 = 该 attention 的线性对偶形式
- 两者是同一 4-way 张量缩并的两种缩并顺序 [§5.3]

[论文原文] 更强的逆定理: 任何有界阶的高效自回归 attention 必须是半可分矩阵结构的 SMA,即必须是 SSM [Theorem 5.2, Appendix C.2]。这意味着 SSM 不是 attention 的"替代",而是高效自回归注意力的数学本质。

#### SSD 算法: 分块半可分矩阵乘法

[论文原文] SSD 算法的核心思想: 将半可分矩阵 M 分成 Q×Q 大小的块,对角块用二次 attention 形式(可利用 matmul 单元),非对角块利用秩结构分解为三步小矩阵乘法 [§6, Fig 5]:

1. **对角块 (Intra-chunk)**: 每个 chunk 内是一个小规模 SSM,用二次 SMA 形式计算(Q×Q 的 attention-like matmul)。所有 chunk 可并行。
2. **非对角块 — 右因子 (Input→State)**: 每个 chunk 内计算 "假设初始状态为 0 的最终状态",即 B-block-factor × X,得到每个 chunk 的局部最终状态 h。BMM(T/Q, N, P, Q)。
3. **非对角块 — 中心因子 (State→State)**: 在 chunk 之间传递状态,本质是一个长度为 T/Q 的标量 SSM scan(A-block-factor)。因为 chunk 化后序列缩短 Q 倍,这步的计算量可忽略。
4. **非对角块 — 左因子 (State→Output)**: 用正确的初始状态(来自 step 3)和 C-block-factor 计算每个 chunk 的输出贡献。BMM(T/Q, Q, P, N)。

最终输出 = 对角块输出 + 非对角块输出。

**复杂度分析** [§6.3]:
设 N = P = Q(状态维度 = head 维度 = chunk 大小),所有 BMM 项统一为 BMM(T/N, N, N, N):
- 训练 FLOPs: O(TN^2),与纯 SSM 相同
- 推理 FLOPs: O(N^2),与纯 SSM 相同
- 内存: O(TN),比朴素 SSM 的 O(TN^2) 降一个数量级
- 计算全部由 matmul 主导,可利用 tensor cores

[论文原文] Listing 1 提供了约 50 行 PyTorch 的完整 SSD 实现,无需自定义 CUDA kernel [§6]。这与 Mamba-1 需要精心优化的 hardware-aware kernel 形成鲜明对比。

#### 多头模式 (Multi-head Patterns)

[论文原文] SSD 框架使得 attention 的多头概念可直接移植到 SSM [§7.2]:

| SSM 模式 | Attention 类比 | A heads | B heads | C heads | X heads | 效果 |
|---|---|---|---|---|---|---|
| Multi-input (MIS) | Multi-value (MVA) | H | 1 | 1 | H | **最优** (Mamba-1/2 默认) |
| Multi-contract (MCS) | Multi-query (MQA) | H | 1 | H | 1 | 较差 |
| Multi-expand (MES) | Multi-key (MKA) | H | H | 1 | 1 | 较差 |
| Multi-head (MHS) | Multi-head (MHA) | H | H | H | H | 中等 |

[论文原文] MVA 显著优于 MQA/MKA (ppl 8.73 vs 9.33/9.36, 360M model),尽管总状态量相同 [Table 5, §9.4.2]。这个发现是 SSM-centric 视角的产物 — 从 attention 视角出发不会自然得到 MVA。

Mamba-2 默认使用 grouped-value attention (GVA),即 MVA 的分组扩展,以支持 tensor parallelism(B/C 组数为 TP degree 的倍数)[§7.2]。

#### Scalar-Identity 约束 vs Diagonal Structure

Mamba-2 将 A 矩阵从 Mamba-1 的 diagonal 进一步简化为 scalar × identity [§2.4]:
- Mamba-1: A_t ∈ R^{N×N} 为对角矩阵(N 个独立衰减因子)
- Mamba-2: A_t = a_t · I (所有状态维度共享一个衰减因子)

[论文原文] 代价: 轻微损失表达力。收益: 使 SSD 的二次对偶形式具有 attention-like 的可解释性和可实现性(否则二次形式变成通道混合的复杂张量运算) [§10.1]。

[agent 解读] 这是一个典型的 expressivity-efficiency trade-off。实验表明(Table 1),在匹配参数量时 Mamba-2 性能略优于 Mamba-1,说明大状态空间(N=64/128/256)带来的容量提升补偿了 scalar-identity 的表达力损失。

### 系统优化

**Tensor Parallelism** [§8.1]:
- Mamba-1: 因 A/B/C 是 X 的函数,TP 需要额外 all-reduce 获取完整 X 才能计算 A/B/C,导致每 block 2 次 all-reduce(Transformer 仅 1 次)。
- Mamba-2: A/B/C/X 并行从 u 投影,每个 GPU 各算各的 A/B/C/X,仅在最终输出 all-reduce 一次。通信量与 Transformer 相同。

**Sequence Parallelism** [§8.2]:
SSM 天然适合序列并行: 每个 GPU 处理一段序列,计算本地最终状态后传递给下一 GPU。通信量与 worker 数线性(vs attention 的二次)。

**Variable Length Sequences** [§8.3]:
无需 padding/packing: 将 batch 内所有序列拼接为一条长序列,在序列边界处设 A_t = 0(截断状态传递)。

### 混合架构

[论文原文] 350M 模型 48 层实验表明,约 10% 的 attention 层与 SSD 层交错可获最优性能(ppl 8.26 vs 纯 Mamba-2 8.60, 纯 Transformer++ 8.68) [Table 2, §9.2.3]。

2.7B 规模验证: Mamba-2-Attention (58 SSD + 6 Attention) ppl 5.95 vs 纯 Mamba-2 6.09 vs Transformer++ 6.13 [Table 3]。

[论文原文] 假说: SSM 层擅长通用序列到序列映射(信息压缩),attention 层擅长精确检索(从历史中精准提取特定 token),两者互补 [§9.2.3]。

### 训练策略

- 延续 Mamba-1 设置: AdamW, cosine LR schedule, 300B tokens on the Pile, GPT-NeoX tokenizer [Appendix D]
- 改进 recipe (inspired by PaLM/LLaMA): 5x GPT-3 LR, no bias, RMSNorm, AdamW β=(.9,.95) [Appendix D.2]
- Head dim P = 64 (default), state dim N = 64 (可扩至 256) [§2.4]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| LM Perplexity (Pile, 2.7B) | 6.09 | 6.22 (Mamba-2.8B) / 6.13 (Transformer++) | Pile | [Table 1] |
| LM Avg Acc (2.7B, 7-task) | 60.2% | 59.9% (Mamba-2.8B) / 55.7% (Pythia-2.8B) | zero-shot | [Table 1] |
| LM Perplexity (Pile, 1.3B) | 6.66 | 6.80 (Mamba-1.4B) / 7.51 (Pythia-1.4B) | Pile | [Table 1] |
| LM Avg Acc (1.3B, 7-task) | 56.4% | 56.4% (Mamba-1.4B) / 51.7% (Pythia-1.4B) | zero-shot | [Table 1] |
| SSD vs Mamba Scan (N=64) | 2-8x faster | 1x (Mamba fused scan) | A100 | [Fig 10] |
| SSD vs FlashAttention-2 | faster @ seq_len >= 2K | 1x (FA-2) | A100 | [Fig 10] |
| MQAR Acc (dim=256, seq=1024) | ~1.0 (N=64) | ~0.3 (Mamba-1 N=16) / ~0.9 (Attention) | synthetic | [Fig 8] |
| Hybrid ppl (350M, 5 attn layers) | 8.28 | 8.60 (pure Mamba-2) / 8.68 (Transformer++) | Pile | [Table 2] |
| Hybrid ppl (2.7B, 58 SSD + 6 attn) | 5.95 | 6.09 (pure Mamba-2) / 6.13 (Transformer++) | Pile | [Table 3] |

**关键消融结果**:

- **Block 设计** [Table 4]: 并行投影 + 额外 Norm 是最优组合(ppl 11.49 vs 串行无 Norm 11.76, 125M)
- **Multi-head 模式** [Table 5]: MVA >> MQA ≈ MKA (ppl 8.73 vs 9.33 vs 9.36, 360M), MHA 中等(9.01)
- **Kernel 近似** [Table 6, 7]: 各种 linear attention 的 kernel feature map (cosFormer, Performer, RFA, Based, ReBased) 均不优于简单 Swish 或 identity; 1-SS mask 的存在使 softmax 近似不再必要
- **状态维度扩展** [Fig 10 right]: Mamba scan 速度随 N 线性下降; SSD 在 N=8 到 N=256 几乎无减速(归功于 matmul 单元利用)

## 局限性

1. **A 矩阵表达力受限**: Scalar-identity 约束比 Mamba-1 的 diagonal 更严格。虽然实验未显示性能下降,但在需要 per-dimension 不同衰减速率的任务(如多尺度时序建模)上可能有影响 [§10.1]
2. **不包含 softmax**: SSD 不泛化标准 softmax attention,只泛化 kernel attention。在需要精确 attention 分布(如 copying, in-context learning)的任务上仍有差距 — MQAR 实验需要 N=256 才完全解决 [Fig 8, §10.3]
3. **音频/语音实验缺失**: 与 Mamba-1(含 SC09, YouTubeMix 音频实验)不同,Mamba-2 论文无任何音频实验。[agent 解读] 这留下了 SSD 在连续信号建模上表现如何的开放问题,后续 MamTra 等工作才开始填补
4. **规模验证**: 最大仅 2.7B。论文承认 tensor parallelism 的设计是为更大规模准备的,但未实际验证 [§8]
5. **理论-实践 gap**: SSD 框架的理论贡献(半可分矩阵等价)主要适用于 scalar-identity A。一般 diagonal SSM 仍无法享受同等的硬件效率提升 [§10.1]
6. **chunk 大小 Q 的选择**: 论文默认 Q=64 但未系统消融。Q 过小则 chunk 间递推开销增加; Q 过大则 chunk 内二次计算增加。最优 Q 可能依赖硬件和序列长度

## 点评

Mamba-2 的核心贡献不在于刷点,而在于改变了 SSM 研究的思维框架。Mamba-1 的叙事是"SSM vs Transformer";Mamba-2 的叙事是"SSM 和 Transformer 本质上是同一类计算的两种形式"。这个 reframing 的价值在于:

1. **消除生态壁垒**: TP、SP、multi-head、variable length 等 Transformer 工程技术可直接迁移到 SSM,不再需要 SSM 社区从零开发
2. **算法设计新范式**: "选择最佳矩阵乘法算法"替代"设计 SSM 递推实现",将序列建模的效率问题归约为经典数值线性代数
3. **架构统一**: MVA/MQA 等 head 模式在 SSM 和 attention 间建立了对应词典,使混合架构的设计有了系统方法论

**与 Mamba-1 的对比**:

| 维度 | Mamba-1 | Mamba-2 |
|---|---|---|
| A 结构 | Diagonal (N 个独立衰减) | Scalar × Identity (1 个共享衰减) |
| 核心算法 | Hardware-aware parallel scan | SSD 分块半可分矩阵乘法 |
| 训练速度 | 受限于标量 scan | 利用 matmul 单元,2-8x 加速 |
| 状态维度 N | 默认 16,扩大则线性减速 | 默认 64-128,N=256 几乎不减速 |
| TP 支持 | 需 2 次 all-reduce/block | 1 次 all-reduce/block (= Transformer) |
| 代码复杂度 | 需 custom CUDA kernel | ~50 行 PyTorch (Listing 1) |

**对 TTS/语音领域的意义**: Mamba-2 使 Mamba 成为 Transformer 的 drop-in replacement 在工程上变得可行。MamTra 已证明可将 CosyVoice 2 的部分 Transformer 层替换为 Mamba-2 层,VRAM 降 34%,RTF 降 25%,且语音质量几乎无损。混合架构(~10% attention)的发现也与 MamTra 的结论一致: 纯 Mamba 替换不如保留少量 attention 层。

**数学贡献的独立价值**: SSM-semiseparable 等价(Theorem 3.5)和"高效自回归 attention 必须是半可分 SMA"(Theorem 5.2)是超越具体架构选择的基础性结果。它们不仅解释了现有 SSM 为什么 work,也划定了高效序列建模的理论边界 — 任何满足有界阶自回归性质的序列变换都必须是某种 SSM,这是一个非常强的不可能性/充分必要条件。

## 可复用的 idea

1. **分块矩阵乘法统一 dual forms**: 对角块用计算密集型算法(二次 matmul),非对角块用内存高效型算法(线性 scan),通过分块组合获得两者优势。这个范式可迁移到任何具有类似 local+global 结构的计算问题
2. **Multi-value Attention (MVA) 模式**: B/C 共享(类比 K/Q 共享)而 X 独立(类比 V 独立)的 head 结构显著优于 MQA/MKA。TTS 中 codec LM 的多头设计可参考
3. **Parallel projections for TP-friendliness**: 将依赖于中间结果的投影重构为与输入直接并行的投影,消除 TP 同步点。适用于任何需要 TP 的非标准层(如 VQ 层、duration predictor)
4. **混合架构的最优比例**: ~10% attention 层足以补偿 SSM 的 recall 弱点,同时保留绝大部分效率优势。TTS backbone 设计可直接采用(如 MamTra 的 20/24 SSD + 4/24 attention)
5. **Variable length via A_t=0**: 无需 padding/packing 的变长处理,只需在序列边界截断状态传递。对 TTS 训练(utterance 长度差异大)直接可用
6. **50 行 SSD 实现**: Listing 1 证明高效序列建模不需要 custom kernel。对资源有限的语音团队,用 native PyTorch 即可获得接近 FlashAttention 的训练速度

> [!review] 审阅状态
> 结论: pass | high: 0 | medium: 0 | low: 3
> 审阅报告: [[_review/Mamba-2-review.yml]]
> 审阅人: auto (inline, same-session) | 日期: 2026-06-08

---

检索命中: 无直接命中 | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无
