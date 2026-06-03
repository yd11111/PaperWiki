---
type: paper
tier: deep
title: "Improving Robustness of Diffusion-Based Zero-Shot Speech Synthesis via Stable Formant Generation"
arxiv_id: "2409.09311"
source: "Sources/StableFormant.pdf"
authors: [Changjin Han, Seokgi Lee, Gyuhyeon Nam, Gyeongsu Chae]
year: 2024
venue: "arXiv (ICASSP format)"
tags: [TTS, diffusion, zero-shot, source-filter, formant, robustness, pronunciation]
concepts: ["[[Diffusion-based TTS]]", "[[Score Matching]]", "[[Speech Factorization]]", "[[Non-autoregressive TTS]]", "[[F0 Modeling]]", "[[Prosody Modeling]]", "[[Mel Spectrogram]]", "[[Duration Predictor]]"]
models: ["[[论文笔记/StableForm-TTS|StableForm-TTS]]", "[[Grad-TTS]]", "[[Grad-StyleSpeech]]", "[[FastPitchFormant]]", "[[HiFi-GAN]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[LibriTTS]]", "[[LibriTTS-R]]", "[[VCTK]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: StableForm-TTS 属于 [[Diffusion-based TTS]] 范式中 Grad-TTS / Grad-StyleSpeech 一脉,是非自回归 diffusion 声学模型的改进工作。在 [[Diffusion-based TTS]] 演进链中,Grad-TTS (2021, SDE 形式化) → Grad-StyleSpeech (2023, 零样本多说话人适配) → StableForm-TTS (2024, 解决零样本场景下的发音鲁棒性)。当前领域主流已从 diffusion 转向 [[Conditional Flow Matching]] (Voicebox, Matcha-TTS, F5-TTS),且 [[Zero-shot Speech Synthesis]] 的 SOTA 由 LLM+discrete token 方案主导 (CosyVoice 3, Seed-TTS)。

**已有认知**: [[Speech Factorization]] (confirmed) 页记录了语音属性解耦的主流方法 (对抗训练、信息瓶颈、self-distillation),但主要聚焦 content-timbre 解耦。StableForm-TTS 引入的是**源-滤波器 (source-filter)** 分解 — 一种物理声学层面的分解,将 excitation (源,载体声带激励/韵律) 与 formant (滤波器,声道共振/音素内容) 分离,与现有知识库中记录的统计学解耦方法正交。[[F0 Modeling]] [待确认] 中提到 SiFiSinger 也采用源滤波器模型,但应用于 SVS 而非 TTS diffusion。

**创新判断**: 本文是首个将 source-filter theory 引入 diffusion TTS 的工作 — 核心 insight 是仅对 excitation pathway 施加 diffusion,formant pathway 绕过扩散过程直接生成,从而保护发音关键信号不被扩散随机性破坏。这与已有 diffusion TTS 方法 (Grad-TTS, ProDiff 等) 将整个 mel spectrogram 送入 diffusion 过程截然不同。

> 检索命中: [[Speech Factorization]]✓, [[Prosody Modeling]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Diffusion-based TTS]](pending-review), [[Score Matching]](pending-review), [[F0 Modeling]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首次将 source-filter theory 引入 diffusion TTS,将 mel spectrogram 分解为 excitation 和 formant 两路,仅对 excitation 施加扩散,formant 直接生成以保护发音稳定性
> - **路线**: Phonemes → Text Encoder (SALN) → Decomposed Variance Adaptor (pitch/energy → excitation path; content → formant path) → E-F Generators → Score-based Diffusion (仅 excitation) → X'_E + X_F = 合成 Mel → HiFi-GAN vocoder
> - **指标**: CER 1.44 vs 2.73 (GSS, LT-460, VCTK) [Table I]; 放大到 69M/19K h 后 CER 0.52 (接近 Tortoise 0.44, 参数量仅 1/14) [Table III]
> - **可借鉴**: (1) 将生成模型仅施加于需要多样性的路径 (excitation/韵律),确定性内容 (formant) 用非随机路径直接生成; (2) 用 CER ratio 可视化诊断扩散步数对发音鲁棒性的影响
> - **局限**: 基于 Grad-StyleSpeech (非自回归, ~35M 参数),与当前 LLM-based 零样本 TTS SOTA 差距大; SECS 略低于 baseline; 未在 SEED-TTS-Eval 等标准 benchmark 评估; 未开源模型权重

## 核心问题

**问题**: Diffusion-based TTS 在零样本场景下存在严重的发音错误 (mispronunciation) 问题,且现有研究主要关注推理速度-音质权衡,忽视了这一鲁棒性缺陷。

**问题来源的实证分析** [§I, Fig 1a]:
1. **数据分布复杂度**: 从 single-speaker → multi-speaker → zero-shot,CER 随反向步数增加的恶化程度逐级加剧 [论文原文]
2. **扩散随机性累积**: SDE solver 比 ODE solver (PF) 发音更不稳定,因为 SDE 引入额外随机性; 且随反向步数增加,CER 近线性恶化 [论文原文]
3. **弱信号损伤**: 幅度或对比度较弱的音素信号 (formant) 在扩散过程中被噪声破坏 [Fig 3, 论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

StableForm-TTS 由三个核心模块组成 [§II, Fig 2]:
1. **Style Encoder**: 来自 Meta-StyleSpeech,将参考语音编码为 style vector s (训练时=目标语音,推理时=不同说话人参考)
2. **Linguistic Encoder**: Text Encoder + Decomposed Variance Adaptor,生成两条信息不对称的隐表示
3. **Source-Filter Decoder**: Excitation Generator + Formant Generator + Score-based Diffusion Model,仅对 excitation 施加扩散

最终合成: X_hat = X'_E + X_F [Eq. 4],其中 X'_E 是经扩散精化的 excitation,X_F 是直接生成的 formant。

### 关键设计选择

#### 1. 为什么要将 diffusion 仅施加于 excitation pathway?

核心 insight: Formant (共振频率) 携带关键的音素区分信息但信号幅度/对比度较弱,容易被扩散噪声破坏; 而 excitation (声带激励) 携带韵律/说话人特征,需要 diffusion 提供的多样性 [论文原文]。将 formant 从扩散过程中隔离,保护了发音稳定性,同时保留了 diffusion 对韵律和音质的增益 [agent 解读]。

这一思路源自 source-filter theory [Chiba & Kajiyama 1941, Fant 1960]: 人类语音由声门激励 (source) 和声道滤波 (filter) 两个独立过程产生,pitch 由 source 决定,formant 由 filter 决定 [§I, 论文原文]。

#### 2. Decomposed Variance Adaptor: 如何实现双路分离?

参考 FastPitchFormant [26],将 variance adaptor 拆分为两条路径 [§II-B2]:
- **Excitation pathway**: phoneme hidden sequence + pitch embedding + energy embedding → 携带完整韵律信息
- **Formant pathway**: phoneme hidden sequence 直接通过 → 保留纯内容 (非韵律) 信息

为什么不仅用 pitch 还加 energy? FastPitchFormant 只用了 pitch,但消融实验 (Table II) 表明 energy 的移除导致 CER 从 1.44 → 1.87 (LT-460),说明 energy 信息也有助于 excitation-formant 分离 [论文原文]。[agent 解读]: energy 帮助更完整地将韵律信息集中到 excitation pathway,使 formant pathway 更"干净"地只保留内容信息。

#### 3. Score-based Diffusion 的条件化方式

遵循 Grad-TTS 的 SDE 形式化 [Eq. 1-3],但在条件化上有关键区别 [§II-C2]:
- Prior mu = excitation representation X_E (非完整 mel)
- 条件输入: 将 style vector s 和 formant representation X_F channel-wise 拼接到 mu 作为条件
- U-Net score estimator: epsilon_phi(X_t, t, mu, s, X_F) [Eq. 5]

这意味着 diffusion model 的输入和学习目标仅是 excitation 部分,formant X_F 作为条件信息指导生成,但 X_F 自身不被修改 [论文原文]。

#### 4. SALN (Style-Adaptive Layer Normalization)

Text Encoder 和 E-F Generators 均使用 SALN 替代标准 Layer Normalization [§II-B1, §II-C1],接收 style vector 计算 gains 和 biases。[agent 解读]: 这使每一层的归一化都 speaker-aware,是零样本多说话人适配的关键机制,继承自 Meta-StyleSpeech / Grad-StyleSpeech。

### 训练策略

总损失 [Eq. 6]:
L_total = L_d + L_p + L_e + L_align + L_prior + L_diff

各项:
- L_d, L_p, L_e: Duration, Pitch, Energy predictor 的 MSE loss (与 FastSpeech 2 相同) [§II-D]
- L_align: CTC forward-sum + KL 散度 (训练 aligner) [§II-B2, ref 21]
- L_prior = ||mu - (X - X_F)||^2: 重建 loss,注意减去了 formant 部分,即 prior 仅对应 excitation [§II-D]
- L_diff: 标准 diffusion loss (score matching, Eq. 5) [§II-D]

训练细节: 1M steps, A100, batch 16, Adam + warmup 4000, temperature tau=1.5 [§III-A2]

## 实验

### 主实验: 零样本 TTS (VCTK, Table I)

| 指标 | StableForm-TTS (LT-460) | GSS (LT-460) | StableForm-TTS (LT-R) | GSS (LT-R) | Ground Truth | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| CER (↓) | **1.44** | 2.73 | **3.04** | 5.80 | 1.66 | [Table I] |
| WER (↓) | **3.64** | 6.49 | **6.74** | 11.95 | 3.75 | [Table I] |
| SECS (↑) | 78.64 | **79.45** | 78.43 | 78.45 | 80.95 | [Table I] |
| UTMOS (↑) | **4.131** | 3.958 | **3.976** | 3.694 | 4.065 | [Table I] |
| MOS (↑) | **3.76** | 3.73 | **3.76** | 3.67 | 3.81 | [Table I] |
| SMOS (↑) | 3.68 | 3.66 | 3.70 | 3.67 | 3.71 | [Table I] |
| Params | 34.86M | 33.98M | 34.86M | 33.98M | - | [Table I] |

LT-460 训练时,StableForm-TTS CER (1.44) 甚至超过 Ground Truth (1.66),说明模型发音比真人录音经 ASR 识别更准确 [§III-B1, 论文原文]。

### 消融实验 (Table II)

| 模型 | CER (↓) | WER (↓) | SECS (↑) | MOS-pred (↑) | CMOS | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| StableForm-TTS | **1.44** | **3.64** | 78.64 | **4.249** | 0.00 | [Table II] |
| w/o E-F generators | 1.63 | 4.05 | **79.12** | 4.235 | -0.38 | [Table II] |
| w/o Energy | 1.87 | 4.52 | 77.87 | 4.191 | -0.39 | [Table II] |

移除 E-F generators 或 Energy predictor 均导致显著性能下降,验证了源-滤波器分解和 energy 信息的必要性 [§III-B2]。

### 可扩展性 (Table III)

| 模型 | CER (↓) | WER (↓) | SECS (↑) | MOS-pred (↑) | Hours | Params | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| StableForm-large | **0.52** | **1.55** | 79.26 | **4.333** | 19K | 69.30M | [Table III] |
| GSS-large | 1.80 | 4.31 | 77.93 | 3.912 | 19K | 68.13M | [Table III] |
| Tortoise | 0.44 | 1.02 | 76.51 | 4.336 | 20K | 972.57M | [Table III] |
| XTTS-v2 | 0.43 | 1.51 | **81.43** | 4.219 | 14K | 466.87M | [Table III] |
| YourTTS | 2.53 | 5.83 | 76.17 | 4.031 | - | 86.82M | [Table III] |

StableForm-large (69M, 19K h) 在 CER/WER 上接近 Tortoise (973M) 和 XTTS-v2 (467M),参数量仅为它们的 1/14 和 1/7 [Table III]。scale-up 使设计优势放大:与 GSS-large 相比,CER 从 1.80 降至 0.52,相对改善 71% [agent 解读]。

## 局限性

1. **范式代差**: 基于 Grad-StyleSpeech (非自回归, mel spectrogram 空间),与当前 LLM-based 零样本 TTS 主流路线差距显著。当前 SOTA (CosyVoice 3, Seed-TTS) 使用自回归 LLM + 离散 token,在数据规模 (100K+ h) 和参数量 (1B+) 上完全不同量级 [agent 解读]
2. **说话人相似度微降**: SECS 在两个数据集上均略低于 baseline GSS (78.64 vs 79.45, LT-460),说明 formant 分离可能轻微影响了说话人特征的保真度 [Table I]
3. **评估局限**: 仅在 VCTK (11 speakers) 上评估,未使用 SEED-TTS-Eval 等标准化 benchmark;主观评估规模有限 (20 participants) [§III-A3]
4. **未开源**: 仅提供 demo 页面,未公开模型权重或训练代码
5. **Vocoder 依赖**: 依赖外部预训练 HiFi-GAN,论文指出使用更好的 vocoder 可进一步提升 [§III-B3]

## 点评

**亮点**: 这篇论文的核心贡献在于将一个物理声学原理 (source-filter theory) 转化为深度学习架构的 inductive bias。作者不是简单地"加模块提指标",而是从问题本质出发 — 发现扩散噪声破坏弱信号 (formant),因此将 formant 从扩散过程中隔离。这种**"只在需要随机性的维度施加随机性"**的设计思路具有广泛适用性。

CER ratio 随反向步数变化的可视化 (Fig 1) 是一个非常有效的诊断工具,清晰展示了问题的存在和解决。

**局限**: 从 2024/2025 的视角看,这个工作的实用价值有限 — 当前零样本 TTS 的主战场已经转向 LLM+codec 范式,Grad-StyleSpeech 这类 ~35M 非自回归模型已不是前沿。但**方法论层面的 insight** (选择性施加生成模型的随机性、基于物理先验的架构分解) 在更大模型中仍可能有价值。

## 可复用的 idea

1. **选择性扩散/流匹配**: 不对全部特征施加生成模型,仅对需要多样性/表现力的维度 (如韵律) 使用随机生成,确定性内容用确定性路径。可推广到 flow matching 或 LLM-based TTS 中
2. **CER ratio 诊断法**: 用 CER(step_n)/CER(step_0) 的比值随步数变化,可视化诊断扩散/流匹配对发音鲁棒性的影响。适用于任何 iterative refinement 系统的鲁棒性分析
3. **Decomposed variance adaptor**: 将 variance adaptor 拆为韵律路径和内容路径,各自输出不同信息密度的表示,可用于任何需要内容-韵律分离的 TTS 架构

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes (3 low issues)
> - (low) venue 标注待确认是否为 ICASSP 2025
> - (low) scalability 对比模型未全部列入 frontmatter.models
> - (low) 局限性中 SOTA 数据规模/参数量数字未注出处 (标注 [agent 解读] 已充分)
> 详见 `_review/StableForm-TTS-review.yml`
