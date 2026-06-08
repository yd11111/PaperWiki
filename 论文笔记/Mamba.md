---
type: paper
tier: deep
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
arxiv_id: "2312.00752"
source: "Sources/Mamba.pdf"
authors: [Albert Gu, Tri Dao]
year: 2023
venue: "arXiv preprint (COLM 2024)"
tags: [SSM, state-space-model, selective-SSM, sequence-modeling, architecture, linear-complexity, audio, language-model, hardware-aware, recurrent, parallel-scan, foundation-model]
concepts: ["[[SpeechLanguageModel]]", "[[LLM-basedTTS]]", "[[CodecLanguageModel]]"]
models: [Mamba, Pythia, RWKV, SaShiMi, H3]
tasks: [language-modeling, audio-generation, DNA-modeling]
datasets: [ThePile, SC09, YouTubeMix, HG38]
kb_context_sources: 3
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 1 个待确认: [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓, [[CodecLanguageModel]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓ | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无
>
> **谱系定位**: Mamba 是 Transformer 的直接竞争架构,提出线性复杂度的选择性状态空间模型(Selective SSM)替代 self-attention。在 TTS/语音领域,Mamba 已作为 backbone 被多个后续工作采用: [[论文笔记/MamTra|MamTra]] 探索 Mamba-Transformer 混合替代 CosyVoice 2 backbone, [[论文笔记/MambaVoiceCloning|MambaVoiceCloning]] 用 Bi-Mamba 替代 Transformer encoder, [[论文笔记/RWKVTTS|RWKVTTS]] 则走了 RWKV-7 的平行路线。KB 中 [[LLM-basedTTS]] 记录的主流 TTS 架构全部基于 Transformer,Mamba 代表了效率优化方向的架构替代路线。
>
> **已有认知**: KB 中 [[SpeechLanguageModel]] 和 [[CodecLanguageModel]] 均以 Transformer/attention 为默认 backbone,尚未系统记录 SSM/Mamba 作为替代 backbone 的可行性。Small-E (Lemerle et al., 2024) 是已有笔记中最早验证 LCLM (线性复杂度循环架构含 Mamba) 在 codec LM 中可行性的工作。
>
> **本文定位**: Mamba 是上述所有 TTS-Mamba 工作的理论根基。它不是 TTS 论文,而是通用序列建模架构,但其 Section 4.4 直接在音频波形建模和语音生成上取得了 SOTA 结果,奠定了 Mamba 进入语音领域的基础。

## 速查

> [!summary] 速查
> - **一句话**: 通过让 SSM 参数随输入变化(选择性机制),突破了结构化状态空间模型的 LTI 限制,实现首个在语言建模上匹配 Transformer 的线性复杂度序列模型
> - **路线**: 输入 x → 线性投影扩展(E=2) → 1D Conv → 选择性 SSM(输入依赖的 Delta/B/C) → 门控乘法 → 线性投影输出;堆叠 Mamba block 构成完整架构
> - **指标**: LM: Mamba-2.8B avg acc 63.3% 超越 Pythia-6.9B 61.7% 和 RWKV-7.4B 62.5% [Table 3]; 音频: SC09 FID 0.94 vs SaShiMi 1.99 (同规模 ~6M), 扩展至 24.3M 后 FID 0.67 vs DiffWave+SaShiMi 1.42 [Table 4]; 推理: 5x throughput vs 同规模 Transformer [Fig 8]
> - **可借鉴**: (1) 选择性机制(输入依赖的 gating/遗忘)可迁移到任何序列建模; (2) hardware-aware 算法设计(SRAM/HBM 分层)是通用优化思路; (3) 将 H3 SSM block 和 MLP 合并为单一 block 的架构简化策略
> - **局限**: 仅验证到 3B 参数规模; 在连续信号(音频)上选择性机制可能不如 LTI SSM(需用 complex 参数); 下游生态(fine-tuning, RLHF, ICL)未充分验证; 后续 Mamba-2 已显著改进

## 核心问题

Transformer 的 self-attention 在序列长度上是 O(L^2) 计算和 O(L) KV cache,对长序列(音频波形、基因组)造成根本性瓶颈。已有 subquadratic 替代(线性 attention、全局卷积、结构化 SSM)能解决效率问题,但在离散模态(尤其语言)上性能远逊于 attention。

**根本原因诊断** [§3.1]: 先前所有结构化 SSM 都是 LTI(线性时不变)的 — 参数 (A, B, C) 在所有时间步固定。这意味着它们无法根据输入内容选择性地记住或遗忘信息。从卷积视角看,LTI = 固定卷积核,无法处理变间距的内容选择任务;从循环视角看,LTI = 固定状态转移,无法做内容感知的信息路由。

**本文的核心主张**: 序列建模的根本问题是将上下文压缩到有限状态中 [§3.1]。Attention 不压缩(KV cache 保留一切)所以 effective 但 inefficient; RNN 压缩到固定状态所以 efficient 但 effective 取决于压缩质量。关键在于选择性(selectivity) — 让模型根据输入内容决定保留什么、遗忘什么。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Mamba 架构由重复堆叠的同质 Mamba block 构成 [§3.4, Fig 3],每个 block 内部:

1. **输入投影**: 将维度 D 扩展到 ED (E=2),分两个分支
2. **分支 1 (主干)**: 1D 因果卷积 → SiLU 激活 → 选择性 SSM (S6)
3. **分支 2 (门控)**: SiLU 激活
4. **合并**: 两分支逐元素相乘(门控)
5. **输出投影**: ED → D

[论文原文] 这种设计的灵感来自 GAU (Gated Attention Unit, Hua et al. 2022),将 H3 架构中分开的 SSM block 和 MLP block 合并为单一 block [§3.4]。与 Transformer 的参数匹配: 两个 Mamba block 的参数量 ≈ 一个 MHA + MLP block (均为 12D^2) [§3.4]。

### 关键设计选择

#### 选择性机制 (Selection Mechanism)

核心改动极其简洁: 让 SSM 参数 Delta, B, C 成为输入的函数 [§3.2, Algorithm 2]:

- **S4 (LTI)**: B: (D, N) = 固定参数; C: (D, N) = 固定参数; Delta: (D) = 固定参数
- **S6 (Selective)**: B: (B, L, N) = Linear_N(x); C: (B, L, N) = Linear_N(x); Delta: (B, L, D) = softplus(Parameter + Linear_1(x)) 广播到 D

[论文原文] B 和 C 选择性让模型精细控制"让哪些输入进入状态"和"让状态的哪些部分输出",分别对应 content-based 和 context-based 的调制 [§3.5.2]。

#### Delta 的核心角色

[论文原文] Delta 控制对当前输入的"关注 vs 忽略"平衡 [§3.5.1]:
- 大 Delta → 重置状态,聚焦当前输入("选择"它)
- 小 Delta → 保持状态,忽略当前输入("跳过"它)

**与 RNN gating 的严格等价** (Theorem 1) [§3.5]: 当 N=1, A=-1, B=1 时,选择性 SSM 退化为:
```
g_t = sigma(Linear(x_t))
h_t = (1 - g_t) * h_{t-1} + g_t * x_t
```
这正是经典 GRU/LSTM 的 gating 方程。[论文原文] SSM 的离散化是启发式 gating 的有原则基础 [§3.5]。

[agent 解读] 这个等价关系解释了为什么 Mamba 能在离散数据上 work: 选择性机制本质上给 SSM 赋予了 RNN 式的内容感知 gating,但通过更大的状态维度 N 获得了远超传统 RNN 的表达能力。

#### 选择性机制的三个机械效应

[论文原文] [§3.5.2]:
1. **变间距 (Variable Spacing)**: 过滤掉不相关的噪声 token (如语言中的 "um"),通过 g_t → 0 实现
2. **上下文过滤 (Filtering Context)**: 选择性重置状态丢弃无关历史,使性能随上下文长度单调递增
3. **边界重置 (Boundary Resetting)**: 在独立序列拼接处重置状态 (Delta → ∞ 即 g_t → 1),替代 Transformer 的 attention mask

#### Hardware-Aware Parallel Scan 算法

选择性使参数时变,无法再用全局卷积 (O(BLD log L))。直接展开循环需 O(BLDN) 且串行。Mamba 用三个技术克服 [§3.3.2]:

1. **Kernel Fusion**: 不在 GPU HBM 中 materialize 完整状态 (B, L, D, N),而是从 HBM 加载参数到 SRAM,在 SRAM 中完成离散化+循环,只写回 (B, L, D) 的输出到 HBM
2. **Parallel Scan** (Blelloch 1990): 用 work-efficient 并行扫描算法解决循环的顺序依赖
3. **Recomputation**: 前向不存中间状态,反向时从 HBM 重新加载到 SRAM 重算(类似 FlashAttention 的重计算策略)

[论文原文] 结果: fused selective scan 的内存需求与 FlashAttention 优化的 Transformer 相当 [§3.3.2]。

### 训练策略

- 训练配方遵循 GPT-3/Chinchilla: AdamW, cosine LR schedule, 300B tokens on the Pile [§4.2, Appendix E.2]
- 模型规模: 130M 到 2.8B 参数,与 GPT-3 规格镜像 [§4.2]
- 上下文长度: 2048 tokens (与 Pythia 相同) [§4.2.2]
- 音频任务使用 complex-valued SSM 替代 real-valued (唯一例外) [§4.4.1]
- A 矩阵使用 S4D-Real 初始化: A_n = -(n+1),对语言建模优于 S4D-Lin complex 初始化 [§4.6.2, Table 8]
- Delta 初始化: softplus^{-1}(Uniform([0.001, 0.1])) [§3.6]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| LM Perplexity (Pile, 1.4B) | 6.80 | 7.51 (Pythia-1.4B) / 7.70 (RWKV-1.5B) | The Pile | [Table 3] |
| LM Avg Acc (1.4B, 6-task) | 59.7% | 55.2% (Pythia-1.4B) / 54.3% (RWKV-1.5B) | zero-shot | [Table 3] |
| LM Avg Acc (2.8B, 6-task) | 63.3% | 59.1% (Pythia-2.8B) / 59.6% (RWKV-3B) | zero-shot | [Table 3] |
| Audio FID (SC09, 6.1M) | 0.94 | 1.99 (SaShiMi 5.8M) | SC09 | [Table 4] |
| Audio FID (SC09, 24.3M) | 0.67 | 1.42 (DiffWave+SaShiMi 23M) | SC09 | [Table 4] |
| Audio IS (SC09, 24.3M) | 7.33 | 5.94 (DiffWave+SaShiMi) | SC09 | [Table 4] |
| Audio mIS (SC09, 24.3M) | 144.9 | 69.17 (DiffWave+SaShiMi) | SC09 | [Table 4] |
| Audio BPB (YouTubeMix, 1M ctx) | ~1.20 | ~1.32 (SaShiMi) | YouTubeMix | [Fig 7] |
| DNA Perplexity (40M params) | ~1.38 | ~1.42 (Transformer++) / ~1.41 (HyenaDNA) | HG38 | [Fig 5] |
| Inference Throughput (2.8B) | ~5x | 1x (Transformer-2.8B) | - | [Fig 8] |
| Selective Copying Acc | 99.8% | 57.0% (H3-S4) / 30.1% (H3-Hyena) | synthetic | [Table 1] |
| Induction Heads (extrapolate 1M) | perfect | fail >2x (all others) | synthetic | [Table 2] |

**关键消融结果** [§4.6]:
- 选择性 vs 非选择性: S6 vs S4 在 Mamba 架构中 perplexity 8.69 vs 10.56 [Table 6]
- Delta 是最重要的选择参数: 仅选择性 Delta 即 perplexity 9.81 (vs 基线 10.93); 三者全选 8.71 [Table 7]
- 状态维度 N 的影响: N=1→16, 选择性时 perplexity 从 9.73 降到 8.71; 非选择性时几乎不变 (9.88→9.81) [Table 10]
- 架构: Mamba block 与 H3 block 性能相似,但更简洁 [Table 6]

## 局限性

1. **规模验证不足**: 仅到 3B 参数,未在 7B+ 规模验证。后续工作(Mamba-2, Jamba)表明纯 Mamba 在大规模 LM 上仍有差距,混合架构更优 [论文原文, §5]
2. **连续-离散 trade-off**: 选择性机制改善离散数据但可能损害连续信号 — 音频任务需切回 complex 参数才最优 [§4.4.1, §5]。[agent 解读] 这暗示对 TTS 这种连续音频生成任务,Mamba 的选择性可能需要特殊处理
3. **下游生态未验证**: fine-tuning, instruction tuning, RLHF, ICL, quantization 等 Transformer 生态的关键能力未系统测试 [§5]
4. **音频实验仅限原始波形**: SC09 是 1 秒短语音片段的无条件生成,与实际 TTS (条件生成、长文本、多说话人) 差距大。后续 MamTra 等工作显示纯 Mamba 在 TTS 上仍需 attention 层辅助
5. **推理优势来自无 KV cache**: 但这也意味着没有 prefix caching / prompt reuse 的能力,对需要重复 prompt 的场景(如 few-shot ICL)可能不利

## 点评

Mamba 的核心贡献是一个优雅的 insight: LTI 是 SSM 的瓶颈而非效率保障,选择性 + hardware-aware 算法可以同时解决 expressivity 和 efficiency。从方法论角度,Theorem 1 将 SSM 离散化与 RNN gating 严格关联,不仅解释了 Mamba 为什么 work,也为 LSTM/GRU 等经典方法提供了理论新解读。

**对 TTS/语音的影响**: Mamba 在 SC09 上的结果(FID 0.67 vs 之前 SOTA 1.42)首次证明 SSM 在语音生成上可超越 Transformer/扩散模型,直接催生了后续 TTS 领域的 Mamba 探索。但需注意 SC09 是极简场景(1 秒无条件生成),后续 MamTra 等工作表明在完整 TTS pipeline 中纯 Mamba 仍不如 Transformer 或混合架构,最优方案是将中间层的 attention 替换为 Mamba 而保留底层和顶层 attention [参考 MamTra 笔记]。

**历史地位**: Mamba 是 Transformer 之后最成功的序列建模架构提案之一。虽然后续 Mamba-2 和混合架构(Jamba, Zamba)进一步发展了这一方向,但原始 Mamba 的选择性机制、hardware-aware scan、以及 block 设计已成为后 Transformer 架构的核心组件。

## 可复用的 idea

1. **选择性机制作为通用模块**: 让任何序列模型的参数成为输入函数。这不限于 SSM — 任何需要"根据内容决定记住/遗忘"的场景都适用。在 TTS 中,duration predictor、prosody encoder 等都可受益
2. **Hardware-aware 算法设计**: 从内存层级(SRAM vs HBM)出发设计算法,而非仅从 FLOP 数优化。IO-bound 操作的 kernel fusion 策略是通用的工程方法论
3. **合并 block 简化架构**: 将功能互补的两个 block (SSM + MLP) 合并为一个,减少架构复杂度。这个 pattern 后来在 RWKV-6、Griffin 等模型中被广泛采用
4. **选择性 Delta 作为学习的时间尺度**: Delta 控制模型在不同时间尺度上的行为(大 Delta = 关注瞬时,小 Delta = 关注长期)。对 TTS 中 prosody 建模(需要同时捕捉音素级和句子级韵律)有启发
5. **状态扩展 (N >> 1) + 选择性 B/C**: 增大隐状态维度几乎不增加参数 (~1% for N=16),但与选择性结合后效果显著(perplexity 降 1.0+)。对计算受限的 TTS 模型是低成本改进路径

---

> [!review] 审阅: pass-with-fixes (2026-06-08)
> 0 high / 3 medium / 3 low
> 主要问题: 速查卡片 FID 跨规模比较 (已修正), Pythia-7B 数字错误 (已修正), frontmatter 空字段 (已修正)
> 剩余 low: 核心问题节缺 [论文原文] 标签, 点评 overclaim 缺限定词, 可复用 idea 缺来源标注
> 详见: `_review/Mamba-review.yml`

检索命中: [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓ | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无
