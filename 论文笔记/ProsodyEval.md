---
type: paper
tier: deep
title: "Measuring Prosody Diversity in Zero-Shot TTS: A New Metric, Benchmark, and Exploration"
arxiv_id: "2509.19928"
source: "Sources/2509.19928.pdf"
authors: [Yifan Yang, Bing Han, Hui Wang, Long Zhou, Wei Wang, Mingyu Cui, Xu Tan, Xie Chen]
year: 2025
venue: "ICASSP 2026"
tags: [TTS, evaluation, prosody, prosody-diversity, semantic-tokens, zero-shot, benchmark, metrics]
concepts: ["[[Prosody Modeling]]", "[[Self-Supervised Speech Representation]]", "[[Semantic vs Acoustic Tokens]]", "[[TTS Evaluation]]", "[[Conditional Flow Matching]]", "[[Masked Generative Modeling]]", "[[Non-autoregressive TTS]]"]
models: ["[[HuBERT]]", "[[WavLM]]", "[[CosyVoice]]", "[[CosyVoice 2]]", "[[EnCodec]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 TTS 评估方法论领域,填补了韵律多样性定量评估的空白。在已有 KB 中,[[TTS Evaluation]] [待确认] 记录了评估体系从 MOS/WER/SIM 到分布级评估 (TTSDS/TTSDS2) 再到 LLM-as-Judge (GSRM, SpeechJudge, TTS-PRISM) 的演进,但对**韵律多样性**这一维度几乎没有专门的评估方法。[[Prosody Modeling]] 已详细记录了韵律建模从 GST 到 LLM-based in-context learning 的演进,以及 SSL 中间层对超音段韵律的编码能力 (de la Fuente & Jurafsky, 2024: 中间层 8-9 韵律表征最强),但缺少对 TTS 系统韵律输出**多样性**的量化手段。本文恰好连接了 [[Self-Supervised Speech Representation]] [待确认] 中 HuBERT/WavLM 的 semantic token 表征与 [[Semantic vs Acoustic Tokens]] 中 "semantic tokens 编码语义但丢失声学细节" 的知识,将 semantic tokens 重新定位为韵律多样性度量的载体。
>
> **已有认知**: [[Conditional Flow Matching]] 记录了 flow matching 在 TTS 中作为 fine-stage renderer 的应用; [[Masked Generative Modeling]] [待确认] 记录了 MaskGCT 的迭代 mask-and-prediction 范式。这些与本文对 AR/NAR/MGM 三种范式的韵律多样性对比直接相关。
>
> **创新判断**: 已有 KB 中 TTS 评估体系聚焦于 naturalness (MOS), intelligibility (WER), speaker similarity (SIM), 以及最近的分布级整体质量 (TTSDS2) 和多维诊断 (TTS-PRISM),但缺少 prosody diversity 这一独立评估维度。本文首次提供了 human-annotated prosody diversity benchmark + 客观指标 (DS-WED),是 TTS 评估体系的重要补充。
>
> 检索命中: [[Prosody Modeling]]✓, [[Conditional Flow Matching]]✓, [[Semantic vs Acoustic Tokens]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[TTS Evaluation]](pending-review), [[Self-Supervised Speech Representation]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 DS-WED (基于 semantic token 加权编辑距离的韵律多样性指标) + ProsodyEval (首个人类标注韵律多样性数据集),系统性 benchmark 了零样本 TTS 系统的韵律多样性
> - **路线**: 合成语音对 → VAD 去静音 → SSL 模型 (HuBERT/WavLM) 提取表征 → k-means 离散化 → 加权 Levenshtein 编辑距离 → DS-WED 分数
> - **指标**: DS-WED 与人类 PMOS 相关性 r=0.77 [Table 1] (vs log F0 RMSE 0.30, MCD 0.66); RTF 0.110 (vs F0 RMSE 0.549, MCD 0.203) [§4.2]
> - **可借鉴**: (1) 将 semantic tokens 重新定位为韵律度量载体而非语义载体 -- SSL 中间层天然编码韵律; (2) 加权编辑距离作为离散序列差异的可解释度量; (3) duration perturbation 作为诊断 NAR 系统韵律瓶颈的手段
> - **局限**: 仅在英语上验证跨语言适用性; 仅评估 0.5B 规模单一模型 backbone; DS-WED 不区分"好的多样性"和"坏的多样性"(如随机错误也会增大编辑距离)

## 核心问题

**现有韵律评估指标无法可靠量化零样本 TTS 系统的韵律多样性**。log F0 RMSE 仅捕获 pitch 维度 (忽略 rhythm、intensity),与人类感知弱相关 (r=0.30) [Table 1],且需要 DTW 对齐,计算量大。MCD 虽相关性稍好 (r=0.66) 但仍不足。缺少一个既与人类判断高度相关、又高效可扩展的韵律多样性指标。

更深层的问题是: **不同生成范式 (AR / flow matching NAR / masked generative NAR) 在韵律多样性上有何系统性差异?原因是什么?** 这在之前几乎未被研究。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文不是一个新的 TTS 系统,而是一个**评估框架**,包含三个组件:

1. **ProsodyEval 数据集**: 1000 个合成语音样本 (来自 7 个 TTS 系统) + 2000 条人类韵律多样性评分 (20 名评估者) [§2]
2. **DS-WED 指标**: 基于 semantic token 序列的加权编辑距离 [§3]
3. **Benchmark + Exploration**: 对 7 个开源零样本 TTS 系统的系统性韵律多样性评测 [§4.4-4.5]

### 关键设计选择

#### 为什么用 semantic tokens 而非 acoustic tokens 或 F0

这是本文最核心的设计选择。论文给出三个理由 [§3, Discussion]:

1. **SSL semantic tokens 编码韵律信息**: HuBERT/WavLM 的 k-means 离散化 token 已被证明能有效捕获韵律 (引用 Onda et al., 2025) [论文原文]。这与 KB 中 de la Fuente & Jurafsky (2024) 的发现一致: SSL 中间层 (8-9) 对超音段韵律 (stress, tone, accent) 分类最强 [agent 解读]
2. **监督式 semantic tokens (S3Tokenizer) 不适合**: CTC loss 等序列级 ASR 损失会扭曲 token 的时长信息 [论文原文]。这是因为 CTC 的路径压缩会丢弃 duration 变化,而 duration 是韵律多样性的关键维度 [agent 解读]
3. **Acoustic tokens (EnCodec) 不适合**: 保留了与韵律无关的低层声学信号细节 [论文原文]

#### 为什么用加权编辑距离

DS-WED 用 Levenshtein distance 的加权变体衡量两个 token 序列的差异 [§3, Eq. 3]:

$$DS\text{-}WED(c_1, c_2) = \min_{\pi \in A(c_1, c_2)} \sum_{(i,j,o) \in \pi} w_o \cdot c_o(c_{1,i}, c_{2,j})$$

权重设置: w_sub = 1.2 (替换权重), w_ins = w_del = 1.0 (插入/删除权重) [§3, Discussion]

**理由**: 听力测试显示人类对**语调和重音变化** (对应 token 替换) 比**停顿时长变化** (对应 token 插入/删除) 更敏感 [论文原文]。增大替换权重反映了这种感知不对称性。

#### 为什么选择 HuBERT-base 第 8 层、50 clusters

通过消融实验确定 [§4.3, Fig 1]:
- **中间层 (6-9) 相关性最强**: 与 SSL 韵律信息集中在中间层的已知结论一致 [论文原文]
- **较小 cluster 数效果更好**: 更大的 codebook 使编辑距离过于敏感,与人类韵律感知粒度不匹配 [论文原文]
- **HuBERT-base 第 8 层 + 50 clusters 达到最高相关性** [Fig 1]
- **WavLM-base 更稳定**: variance 更小,但 peak 不如 HuBERT-base [论文原文]

### ProsodyEval 数据集设计

**系统覆盖**: 7 个开源零样本 TTS 系统,覆盖三种生成范式 [§2.1]:
- AR (next-token prediction): XTTS-v2, CosyVoice, CosyVoice 2
- NAR (flow matching): E2 TTS, F5-TTS, ZipVoice
- NAR (masked generative modeling): MaskGCT

**采样策略**: 每个系统对每条输入用 5 个不同随机种子生成 5 个样本,组内两两配对 (10 对) 评估韵律差异 [§2.1]

**质量过滤**: 过滤掉存在合成错误 (非逐字对齐) 的组,保留 1000 个样本 [§2.1]

**PMOS 收集**: 20 名有 TTS 研究经验的研究生评估者,每对在 5 分 Likert 量表上打分韵律差异度,共 2000 条评分 [§2.2]

### 训练策略

本文无模型训练。DS-WED 的核心组件 (HuBERT/WavLM + k-means) 使用**已有的预训练模型**:
- SSL 模型: HuBERT-base / WavLM-base (预训练,不 finetune)
- k-means: 50 clusters,在 LibriSpeech 960h 上训练 [§4.1]
- 编辑距离: 确定性算法,无需训练

## 实验

| 指标 | DS-WED | log F0 RMSE | MCD | 出处 |
| --- | --- | --- | --- | --- |
| 与 PMOS 平均 Pearson 相关 | **0.77** [0.73, 0.81] | 0.30 [0.19, 0.40] | 0.66 [0.58, 0.73] | [Table 1] |
| RTF (计算效率) | **0.110** | 0.549 | 0.203 | [§4.2] |

### Benchmark 结果 (DS-WED Avg, LibriSpeech test-clean / Seed-TTS test-en)

| 系统 | 范式 | DS-WED (LS) | DS-WED (Seed) | 出处 |
| --- | --- | --- | --- | --- |
| MaskGCT | NAR-MGM | **139.75** | 80.36 | [Table 2] |
| CosyVoice 2 | AR | 134.34 | 88.04 | [Table 2] |
| XTTS-v2 | AR | 127.84 | **93.15** | [Table 2] |
| CosyVoice | AR | 120.59 | 75.74 | [Table 2] |
| ZipVoice | NAR-FM | 114.52 | 58.56 | [Table 2] |
| E2 TTS | NAR-FM | 84.91 | 52.35 | [Table 2] |
| F5-TTS | NAR-FM | 79.59 | 49.00 | [Table 2] |

### Duration Perturbation 实验

| 系统 | 原始 DS-WED | +DP DS-WED | 变化 | 出处 |
| --- | --- | --- | --- | --- |
| F5-TTS (LS) | 79.59 | 100.88 | +26.7% | [Table 3] |
| F5-TTS (Seed) | 49.00 | 62.95 | +28.5% | [Table 3] |
| MaskGCT (LS) | 139.75 | 159.10 | +13.8% | [Table 3] |
| MaskGCT (Seed) | 80.36 | 92.71 | +15.4% | [Table 3] |

### DPO 对韵律多样性的影响

| 系统 | 原始 DS-WED | +DPO DS-WED | 变化 | 出处 |
| --- | --- | --- | --- | --- |
| CosyVoice 2 (LS) | 134.34 | 109.09 | **-18.8%** | [Table 4] |
| CosyVoice 2 (Seed) | 88.04 | 71.64 | **-18.6%** | [Table 4] |
| MaskGCT (LS) | 139.75 | 135.75 | -2.9% | [Table 4] |
| MaskGCT (Seed) | 80.36 | 77.80 | -3.2% | [Table 4] |

### LALM 作为韵律评估者

| 模型 | 与 PMOS 相关 | 与 log F0 RMSE | 与 MCD | 与 DS-WED | 出处 |
| --- | --- | --- | --- | --- | --- |
| Gemini 2.5 Pro | 0.27* | 0.10 (n.s.) | 0.16* | 0.22* | [Table 5] |

## 局限性

1. **仅英语验证**: DS-WED 的跨语言适用性未被验证,不同语言的韵律系统差异巨大 (如声调语言) [Limitations]
2. **单一模型规模**: 仅评估 0.5B 级别模型,未探索 scaling 对韵律多样性的影响 [Limitations]
3. **多样性 vs 质量未解耦**: DS-WED 度量的是差异大小,不区分"有意义的韵律变化"和"合成错误导致的差异" [agent 解读]。一个频繁出错的系统理论上也可能获得高 DS-WED 分数
4. **PMOS 标注规模有限**: 20 名评估者 + 1000 样本,且评估者均为 TTS 研究领域研究生,可能存在专家偏差 [agent 解读]
5. **未覆盖闭源系统**: 未 benchmark GPT-4o-mini-tts、Gemini TTS 等闭源系统的韵律多样性 [agent 解读]

## 点评

**优势**:
1. **问题定义精准**: 在 TTS 评估领域,韵律多样性是一个被长期忽视的维度。本文首次将其独立出来并提供完整的评估工具链 (数据集 + 指标 + benchmark)
2. **方法简洁有效**: DS-WED 的设计体现了"用已有组件的正确组合解决新问题"的思路 -- 复用 SSL 模型 + k-means + 编辑距离,计算效率高 (RTF 0.110),可扩展
3. **洞察有深度**: 对 AR/NAR/MGM 三种范式的韵律多样性差异给出了有说服力的因果解释 (flow matching 的隐式对齐导致 mean-mode collapse → 韵律单调)
4. **DPO 与多样性的 trade-off 发现很有价值**: 这一发现 (DPO 提升可懂度但降低韵律多样性 -18.8%) 对 TTS 系统的后训练策略有实际指导意义

**不足**:
1. DS-WED 缺乏对韵律质量的建模 -- 它衡量"变化多大",但不判断"变化是否合理/自然"。一个理想的韵律评估指标应同时考虑多样性和适当性
2. 权重 w_sub=1.2 的设定依据 (听力测试) 较弱,仅基于定性观察而非量化的 perceptual weighting 实验
3. 论文未深入讨论 DS-WED 与 TTSDS/TTSDS2 中 Prosody 因子的关系 -- TTSDS2 也用 Wasserstein 距离衡量韵律维度,两者是互补还是冗余?

## 可复用的 idea

1. **SSL 中间层 semantic tokens 作为韵律 proxy**: 不需要显式 F0/duration 提取,直接利用 SSL 表征的中间层韵律信息进行韵律相关任务 (评估、控制、对齐)
2. **Duration perturbation 作为韵律诊断工具**: 通过在推理时对 NAR 系统施加时长扰动,可以快速诊断韵律单调是否源于时长控制缺失
3. **加权编辑距离的感知定制**: 根据人类感知特性调整不同编辑操作的权重,可推广到其他离散序列的感知距离度量
4. **"RL 削弱多样性" 的启发**: 在 TTS 后训练中使用 DPO/RLHF 时,需监控韵律多样性的退化,可能需要在 reward 中加入多样性保护项

> [!review] 审阅 (agent-v2, 2026-06-04)
> **结论: pass-with-fixes** | 2 issues (0 high, 1 medium, 1 low)
> 
> - **[medium] factual-error**: Benchmark 表中 MaskGCT Seed-TTS DS-WED (80.36) 原被加粗标为最高,但 XTTS-v2 (93.15) 才是 Seed-TTS 最高 → **已修正**
> - **[low] template-compliance**: datasets 字段原含 [[Emilia]],但 Emilia 是被评测系统的训练数据,非本文直接使用 → **已修正**
> 
> 详见 `_review/ProsodyEval-review.yml`
