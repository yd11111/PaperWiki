---
type: paper
tier: deep
title: "Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing"
arxiv_id: "2502.14458"
source: "Sources/Llamba.pdf"
authors: [Aviv Bick, Tobias Katsch, Nimit Sohoni, Arjun Desai, Albert Gu]
year: 2025
venue: "arXiv preprint"
tags: [SSM, Mamba, knowledge-distillation, cross-architecture-distillation, recurrent-model, efficient-inference, on-device, language-model, MOHAWK, subquadratic]
concepts: ["[[SpeechLanguageModel]]", "[[CodecLanguageModel]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 1 个待确认: [[SpeechLanguageModel]]$\checkmark$, [[CodecLanguageModel]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]$\checkmark$ | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无
>
> **谱系定位**: Llamba 是 [[论文笔记/Mamba|Mamba]](Gu & Dao, 2023)架构在大规模语言建模上的实际落地验证,通过跨架构蒸馏从 Llama-3.x 获取知识。在本 vault 中,Mamba 作为 Transformer 替代架构已有详细笔记,且后续在 TTS 领域催生了 [[论文笔记/MamTra|MamTra]]、[[论文笔记/MambaVoiceCloning|MambaVoiceCloning]]、[[论文笔记/RWKVTTS|RWKVTTS]] 等探索。KB 中 [[SpeechLanguageModel]] 和 [[CodecLanguageModel]] 均以 Transformer 为默认 backbone;[[论文笔记/Small-E|Small-E]] 是 KB 中最早验证 Mamba 在 codec LM 中可行性的工作。
>
> **已有认知**: Mamba 笔记已详细分析了选择性 SSM 机制、hardware-aware parallel scan、和 block 设计。Llamba 在此基础上引入 Discrete-Mamba-2 变体(去除 Delta 离散化)和 MOHAWK 蒸馏框架,是 Mamba → 实际部署的关键中间步骤。
>
> **本文定位**: Llamba 不是 TTS 论文,但它验证了两个对 TTS-Mamba 方向有重大意义的结论:(1) Mamba 架构可通过蒸馏而非从头训练获得强 LLM 能力;(2) Mamba 在推理效率(吞吐量、内存)上显著优于 Transformer,特别适合 on-device 部署。这为 TTS 领域用 Mamba 替代 Transformer backbone 提供了方法论参考(蒸馏路线)和部署可行性证据(Apple Silicon 实测)。

## 速查

> [!summary] 速查
> - **一句话**: 用 MOHAWK 三阶段跨架构蒸馏将 Llama-3.x Transformer 知识迁移到 Mamba-2 架构,仅用 <0.1% 训练数据即获得接近教师模型的性能,同时大幅提升推理吞吐和降低内存消耗
> - **路线**: Llama-3.x (teacher) → MOHAWK 3阶段蒸馏(Matrix Orientation → Hidden-State Alignment → Weight Transfer + KD) → Discrete Mamba-2 blocks + Llama MLP → Llamba-{1B,3B,8B}
> - **指标**: Llamba-8B AVG 68.8% vs 教师 Llama-3.1-8B 69.4% (8 benchmarks zero-shot) [Table 1]; MMLU 61.0 (relative 80.6% vs teacher) [Table 3]; 仅用 12B tokens 蒸馏 [Table 2]; 推理吞吐远超 Transformer 尤其大 batch [Fig 4]; Apple Silicon M3 Pro 上恒定内存/吞吐 [Fig 5]
> - **可借鉴**: (1) MOHAWK 三阶段渐进蒸馏策略可迁移到 TTS 中 Transformer→Mamba 的架构替换; (2) Discrete-Mamba-2 的简化(去除 Delta 离散化+非线性)有助于蒸馏对齐; (3) 交替 Mamba+MLP 的架构比纯 Mamba 更省内存且更快(MLP 无时间状态)
> - **局限**: MMLU 仍有 gap (61.0 vs 68.0); 未验证 instruction tuning/RLHF 后表现; 对比中部分 baseline 是 hybrid 架构(含 attention); 在 TTS/语音任务上未测试

## 核心问题

Transformer 的二次 attention 复杂度使其在长序列推理时计算和内存开销大,难以部署到资源受限设备。然而,SSM/Mamba 等亚二次架构虽然效率高,但从头训练需要海量数据(万亿 token 级),且在知识密集型任务(如 MMLU)上仍落后于 Transformer [§1]。

**Llamba 要解决的问题**: 能否通过蒸馏而非从头训练,用极少量数据(<0.1% 原始训练量)将强 Transformer LLM 的知识迁移到 Mamba 架构,同时保留接近的 benchmark 性能?

这个问题的难点在于跨架构蒸馏: Transformer 的 attention 机制(软权重矩阵)和 Mamba 的 SSM 机制(循环状态)有本质结构差异,直接的 logit-level KD 不够 [§4.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Llamba 保留 Llama 教师模型的整体结构,仅将 self-attention 替换为 Discrete Mamba-2 层 [§3, Fig 2b]:

每个 residual block 包含:
1. RMSNorm → Discrete Mamba-2 (替代 attention)
2. RMSNorm → Feed-Forward (Gated MLP, 直接复用 Llama 的 MLP)

三个规模: Llamba-1B (16 blocks), Llamba-3B (28 blocks), Llamba-8B (32 blocks) [§3]。共享 Llama-3.1 的 tokenizer 和词表。

### 关键设计选择

#### Discrete Mamba-2: 为蒸馏而生的架构简化

[论文原文] Llamba 使用的 Discrete-Mamba-2 (Bick et al., 2024) 与原始 Mamba-2 有四处关键修改 [§3, Fig 2a]:

1. **去除 Delta 离散化**: 原始 SSM 通过 Delta 参数将连续时间 A 矩阵离散化; Discrete-Mamba-2 直接从输入投影 A 矩阵,因为 [论文原文] attention 本质上是离散操作,去除离散化更利于与 attention 对齐 [§3]
2. **去除 post-convolution 激活**: 去除了卷积后的非线性激活
3. **去除 pre-output projection 归一化**: 去除了输出投影前的归一化层
4. **Multi-Head 结构**: 使用 32 个独立 head (d=64/96/128),与 Llama 的 32 query heads 数量匹配,但不像 GQA 那样共享 KV heads — 因为 Mamba 的循环层不依赖 KV cache,无需 GQA 式优化 [§3]

[agent 解读] 修改 2-3 的动机是: 非线性操作在 attention block 中不存在,这些不对称性会损害 MOHAWK 第一阶段的矩阵对齐。通过去除它们,student 的 mixing 操作变得更接近线性,使矩阵层面的对齐更有效。

#### 交替 MLP 的效率优势

[论文原文] 与原始 Mamba/Mamba-2 的纯 SSM block 不同,Llamba 在每个 Mamba-2 层之间交替 Llama 的 Gated MLP [§3]。这不仅是蒸馏约束(保留教师的 MLP 结构),还带来额外效率优势 [§6.2]:

1. **更大 batch size**: MLP 在时间维度上无状态,减少了临时内存占用,允许 batch size 比纯 Mamba-2 大一倍
2. **减少 kernel launch overhead**: 更少的 temporal mixing layers 意味着更少的 Mamba kernel 调用

[agent 解读] 这是一个巧妙的 side effect: 蒸馏的结构约束(必须保留教师 MLP)反而成了效率优势,因为 MLP 层是纯 stateless 计算,不贡献循环状态的内存开销。

### 训练策略

#### MOHAWK 三阶段蒸馏 [§4.1]

**阶段 1 — Matrix Orientation** (300-500M tokens):
- 目标: 对齐 student 和 teacher 的 matrix mixer
- 方法: 最小化 Llamba Mamba-2 混合矩阵与 Llama attention 矩阵的距离
- [论文原文] Llama 使用 GQA (32 query, 8 KV heads),权重共享; Llamba 的 32 heads 独立不共享,因此学到的是独立权重而非教师的 dependent matrices [§4.1]

**阶段 2 — Hidden-State Alignment** (2.7-5B tokens):
- 目标: 逐层对齐中间表征
- 方法: 每个 Mamba-2 block 独立用 L2 距离对齐,参照前一层输出

**阶段 3 — Weight Transfer + Knowledge Distillation** (5-6.5B tokens):
- 首先转移 MLP weights, normalization layers, input embedding, output head
- [论文原文] 与前人(Wang et al., 2024; Bick et al., 2024)不同,Llamba 不冻结 MLP,对 MLP 和 mixing 层用相同学习率联合优化 [§4.1]
- 用 cross-entropy logit KD 对齐教师输出
- [论文原文] Loss 饱和后,所有模型进一步从 Llama-3.1-70B-Instruct 蒸馏以获取剩余 token 的知识 [§4.1]

#### 数据选择 [§4.3]

[论文原文] 数据质量对蒸馏至关重要,因为 MOHAWK 只转移 MLP 权重(hidden dimension),不转移 sequence mixer 权重(time dimension),限制了时间信息的直接获取 [§4.3]:

- 阶段 1-2: fineweb-edu-4.0 (教育性网页,4.0 分类器阈值), packed sequences of 2048
- 阶段 3 初期: fineweb-edu-4.0
- 阶段 3 后期: Open-Hermes-2.5 (4 epochs, 200M tokens/epoch, seq_len=4096)

[论文原文] fineweb-edu 对 MMLU 提升尤为关键: C4 和 fineweb 在其他 benchmark 上表现相似,但 MMLU 在使用 fineweb-edu 时显著改善 [Fig 3, §4.3]。这仅用公开数据集即可实现,无需私有高质量数据。

#### 训练配置 [§4.2]

- 硬件: 单节点 8x H100 GPU + FSDP + activation checkpointing
- 优化器: AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1)
- 学习率: 阶段 1-2 统一 1e-4; 阶段 3 Llamba-1B/3B 用 5e-5, Llamba-8B 用 1e-5
- Batch: 阶段 1 用 64, 阶段 2-3 用 128
- Scheduler: Warm-Stable-Decay (WSD), min LR=1e-8, warmup/decay 各 10%

#### 总 token 预算 [Table 2]

| 模型 | 阶段1 | 阶段2 | 阶段3 | 总计 |
|------|-------|-------|-------|------|
| Llamba-1B | 300M | 2.7B | 5B | **8B** |
| Llamba-3B | 500M | 4B | 5.5B | **10B** |
| Llamba-8B | 500M | 5B | 6.5B | **12B** |

对比: Llama-3.1-8B 训练用 15T tokens; Llamba-8B 用 12B tokens = **0.08%**。

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| AVG (8-bench, 0-shot) | 68.8% (Llamba-8B) | 69.4% (Llama-3.1-8B teacher) | multi | [Table 1] |
| AVG (8-bench, 0-shot) | 63.9% (Llamba-3B) | 61.9% (Llama-3.2-3B teacher) | multi | [Table 1] |
| AVG (8-bench, 0-shot) | 53.2% (Llamba-1B) | 55.3% (Llama-3.2-1B teacher) | multi | [Table 1] |
| MMLU (0-shot) | 61.0 (Llamba-8B) | 68.0 (Llama-3.1-8B) | MMLU | [Table 1] |
| MMLU relative score | 80.6% | 76.2% (Mamba2-Llama3 hybrid) | MMLU | [Table 3] |
| MMLU relative score | 80.6% | 66.8% (LoLCATs hybrid) | MMLU | [Table 3] |
| MMLU relative score | 80.6% | 24.6% (SUPRA recurrent) | MMLU | [Table 3] |
| ARC-C (0-shot) | 54.6 (Llamba-8B) | 55.1 (teacher) / 53.2 (Falcon3-Mamba) | ARC-C | [Table 1] |
| HellaSwag (0-shot) | 77.6 (Llamba-8B) | 79.3 (teacher) / 79.8 (Falcon3-Mamba) | HS | [Table 1] |
| Winogrande (0-shot) | 73.3 (Llamba-8B) | 73.9 (teacher) / 76.9 (Zamba2) | WG | [Table 1] |
| Throughput (batch 2048) | ~250K tok/s (Llamba-8B) | OOM (Llama-3.1-8B gen=8192) | H100 | [Fig 4] |
| On-device decode (32K ctx) | ~35 tok/s, ~5GB (Llamba-8B 4-bit) | ~5 tok/s, ~15GB (Llama-3.1-8B 4-bit) | M3 Pro | [Fig 5] |

**关键发现**:

1. **蒸馏效率**: Llamba-8B 用 12B tokens 达到教师 8-benchmark 平均性能的 99.1%,而教师本身用了 15T tokens [Table 1, §1]

2. **MMLU 是最难蒸馏的 benchmark**: [论文原文] 其他任务在很少 token 内就达到教师水平,但 MMLU 改善缓慢 [§6.1]。尽管如此,Llamba 的 MMLU relative score (80.6%) 显著超越所有先前蒸馏工作,包括 hybrid 架构(含 attention 层)的 baseline [Table 3]

3. **Llamba-3B 超越教师**: 在 ARC-C (48.5 vs 45.6), ARC-E (79.0 vs 74.3), PIQA (78.6 vs 75.8), Winogrande (70.4 vs 67.6), HellaSwag (73.8 vs 70.4), OBQA (42.8 vs 35.8) 上均超越 Llama-3.2-3B 教师,仅 MMLU 和 Lambada 落后 [Table 1]

4. **推理效率**: Llamba-8B 在 batch=2048 时仍正常运行,而 Llama-3.1-8B 在 gen_len=8192 时 batch=128 左右就 OOM [Fig 4]。Apple Silicon M3 Pro 上,Llamba 在 32K 上下文时吞吐和内存几乎恒定,而 Llama 线性退化 [Fig 5]

## 局限性

1. **MMLU gap 未关闭**: Llamba-8B MMLU 61.0 vs teacher 68.0,差距 7 分。[论文原文] 作者承认 MMLU 对循环模型仍然困难,sliding window attention (即使很小)对 MMLU 有强影响 [§6.1]。[agent 解读] 这可能反映了 attention 在精确信息检索(MMLU 依赖的 in-context fact recall)上的固有优势
2. **未验证 instruction tuning/RLHF**: 蒸馏使用了 instruct 版教师,但 Llamba 本身未做 instruction tuning 或 RLHF。不清楚蒸馏是否保留了教师的 instruction-following 能力
3. **对比 baseline 不完全公平**: Table 3 中部分 baseline (Mamba2-Llama3 hybrid, LoLCATs) 包含 attention 层,不是纯循环架构。Llamba 是纯循环架构,但部分 benchmark 比较中与 hybrid 对比可能给人高估的印象
4. **仅 LLM benchmark**: 未在语音/音频/TTS 任务上测试,无法直接推断蒸馏方法对 speech backbone 的效果
5. **蒸馏依赖强教师**: MOHAWK 的效果高度依赖教师模型质量。从 70B 教师二次蒸馏是性能提升的重要因素 [§4.1],但这意味着方法的效果与可用教师规模绑定

## 点评

Llamba 的核心贡献不是架构创新(Mamba-2 和 MOHAWK 是已有工作),而是工程验证: 证明 MOHAWK 蒸馏在实际规模(1B-8B)上 work,且效果远超预期 — 用 0.08% 训练数据达到 99% 教师性能。这从根本上改变了 Mamba 架构的实用性: 不需要从零训练万亿 token,只需找一个好的 Transformer 教师蒸馏即可。

**对 TTS 方向的启示**: Llamba 验证的蒸馏路线对 TTS-Mamba 方向有直接参考价值。目前 [[论文笔记/MamTra|MamTra]] 等工作仍在从头训练或做简单的层替换; 如果借鉴 MOHAWK 的三阶段渐进对齐策略(matrix orientation → hidden-state alignment → KD),可能更高效地将 CosyVoice/F5-TTS 等 Transformer TTS 的知识迁移到 Mamba backbone 上。

**与 Small-E 的交叉参考**: [[论文笔记/Small-E|Small-E]] 已证明 Mamba 作为 codec LM backbone 可行(吞吐 +62% vs decoder-only Transformer)。Llamba 在更大规模上补充证明了 Mamba 可通过蒸馏获取 Transformer 知识,两者结合指向一个可行路线: 先用 Transformer 训练强 TTS LM → 蒸馏到 Mamba → 部署到边缘设备。

**方法论价值**: Discrete-Mamba-2 的简化设计(去除 Delta 离散化和非线性)表明,当目标是蒸馏而非从头训练时,架构应向蒸馏友好方向修改,而非追求独立最优。这是一个值得记住的 principle。

## 可复用的 idea

1. **MOHAWK 三阶段渐进蒸馏**: Matrix Orientation → Hidden-State Alignment → Weight Transfer + KD 的分层策略可迁移到 TTS 领域的 Transformer→Mamba 蒸馏。关键 insight: 先对齐 low-level 结构(矩阵),再对齐 mid-level 表征(hidden state),最后 end-to-end KD。比直接 logit KD 有效得多
2. **蒸馏友好的架构简化**: 为了对齐而去除非线性(post-conv activation, pre-output norm)。当做跨架构蒸馏时,student 应尽量简化到接近 teacher 的操作结构
3. **交替 MLP 减少内存占用**: MLP 在时间维度无状态,交替 MLP+Mamba 比纯 Mamba stack 节省推理内存。对 TTS streaming inference 有直接工程价值
4. **数据质量 > 数据量 (蒸馏场景)**: fineweb-edu-4.0 (高教育性阈值) 对 MMLU 的提升远大于更大但质量一般的 C4/fineweb。蒸馏 token 预算极小时,数据选择比数据量更重要
5. **二次教师蒸馏**: Loss 饱和后从更大教师(70B)继续蒸馏。对 TTS 的启示: 可先从同规模教师蒸馏,再从更大规模 TTS 模型补充蒸馏

## 审阅

> [!review] 审阅 (2026-06-08, self-review)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 三阶段蒸馏机制、架构修改动机、效率优势均有因果解释 |
> | 可信赖 | pass | 关键数字均标注 [Table/Fig] 出处;指标名称正确 |
> | 可区分 | pass | 因果解释标注了 [论文原文] vs [agent 解读],覆盖率 >80% |
> | 可定位 | pass-with-fixes | KB 背景给出了与 Mamba/MamTra/Small-E 的谱系定位;但 KB 本身 TTS-centric,与本文(通用 LLM)的关联需要读者自行桥接 |
> | 不污染 | pass | 未修改概念页;无反向更新 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> - medium [weak-kb-overlap]: KB 以 TTS 为核心,对本文(通用 LLM 蒸馏)的覆盖天然有限,KB 背景节主要靠 Mamba 笔记而非概念页提供定位。这不是笔记质量问题,而是 vault 领域边界的体现
> - low [fig-approximation]: Fig 4/Fig 5 的吞吐量和内存数字从图中近似读取,非精确值

---

检索命中: [[SpeechLanguageModel]]$\checkmark$ | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无
