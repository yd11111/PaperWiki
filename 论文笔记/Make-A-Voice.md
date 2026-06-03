---
type: paper
tier: deep
title: "Make-A-Voice: Unified Voice Synthesis With Discrete Representation"
arxiv_id: "2305.19269"
source: "https://arxiv.org/abs/2305.19269"
authors: [Rongjie Huang, Chunlei Zhang, Yongqi Wang, Dongchao Yang, Luping Liu, Zhenhui Ye, Ziyue Jiang, Chao Weng, Zhou Zhao, Dong Yu]
year: 2023
venue: "arXiv preprint"
tags: [TTS, voice-conversion, SVS, zero-shot, discrete-token, unified-framework, autoregressive, coarse-to-fine]
concepts: ["[[Semantic vs Acoustic Tokens]]", "[[Residual Vector Quantization]]", "[[Neural Vocoder]]", "[[LLM-based TTS]]", "[[F0 Modeling]]", "[[Singing Voice Synthesis]]"]
models: ["[[模型库/Make-A-Voice|Make-A-Voice]]", "[[模型库/HuBERT|HuBERT]]", "[[模型库/SoundStream|SoundStream]]", "[[模型库/BigVGAN|BigVGAN]]"]
tasks: [TTS, voice-conversion, singing-voice-synthesis]
datasets: [LibriTTS, LibriLight, OpenCPOP, OpenSinger, CSMSC, M4Singer]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Semantic vs Acoustic Tokens]], [[Residual Vector Quantization]], [[Neural Vocoder]], [[LLM-based TTS]], [[F0 Modeling]], [[Singing Voice Synthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Semantic vs Acoustic Tokens]]✓, [[Residual Vector Quantization]]✓, [[Neural Vocoder]]✓, [[LLM-based TTS]]✓, [[Speech Factorization]]✓, [[Speaker Embedding]]✓ | 过滤: [[F0 Modeling]](pending-review), [[Mel Spectrogram]](pending-review) | 未命中但可能相关: [[Singing Voice Synthesis]]

- **Semantic vs Acoustic Tokens**: Make-A-Voice 是 AudioLM 范式的直接继承者,采用 HuBERT semantic tokens → SoundStream acoustic tokens 的两阶段离散表示。KB 记录了此串联策略的优缺点: 概念简单但序列长。本文用 12 层 RVQ 但推理仅取前 3 层作为 vocoder 输入,以避免 codebook mismatch。
- **RVQ**: 本文使用 SoundStream 的 12 层 RVQ (codebook size 1024),但关键设计选择是推理时仅用 3 层 acoustic tokens + unit-based vocoder,而非全部 12 层 + SoundStream decoder。
- **Neural Vocoder**: 本文改进 BigVGAN 训练 unit-based vocoder,从 3 层 acoustic tokens 直接合成波形。KB 记录了 BigVGAN 的 AMP block 和 Snake activation。
- **LLM-based TTS**: Make-A-Voice 属于 LLM-based TTS 中较早的统一框架尝试。与 VALL-E 同期,但 Make-A-Voice 进一步统一了 TTS/VC/SVS 三个任务。
- **Speech Factorization**: 本文通过 semantic-acoustic 两级分离实现 content-style 解耦: semantic tokens 编码内容,acoustic tokens 通过 speaker prompt 条件化编码风格。
- **F0 Modeling** [待确认]: SVS 应用中引入 quantized F0 prompt 作为额外条件,实现精确音高控制。

> [!summary] 速查
> - **一句话**: 基于离散表示的统一语音合成框架,通过 coarse-to-fine 三阶段 (semantic→acoustic→waveform) 统一处理 TTS、VC 和 SVS 三个任务 [论文原文]
> - **路线**: Text/Speech→S1(Text-to-Semantic/HuBERT)→Semantic Tokens→S2(Semantic-to-Acoustic Transformer, conditioned on speaker+F0 prompt)→Acoustic Tokens (3 levels)→S3(Unit-based Vocoder)→Waveform [Fig 1]
> - **指标**: TTS MOS 4.04, SMOS 3.81, CER 0.068, Cos 0.77 (vs YourTTS MOS 3.89, SMOS 3.72); VC MOS 4.07, Cos 0.80 (vs PPG-VC MOS 3.97, Cos 0.78); SVS MOS 3.99, FFE 0.05 (vs Diffsinger MOS 3.98, FFE 0.08) [Table 2-4]
> - **可借鉴**: (1) 三阶段统一框架设计,同一 backbone 处理 TTS/VC/SVS; (2) Unit-based vocoder 替代 codec decoder 避免 codebook mismatch; (3) Quantized F0 prompt 条件化 acoustic stage
> - **局限**: (1) 推理需三阶段串联,无端到端; (2) HuBERT mono-lingual 限制跨语言; (3) 无代码开源; (4) 评估规模较小 (50 sentences)

## 核心问题

1. **任务分散**: TTS、VC、SVS 各自独立开发,方法论碎片化,无法共享模型组件 [§1] [论文原文]
2. **数据标注依赖**: 大多数语音合成模型需要 text-audio parallel data,限制了数据 scalability [§1] [论文原文]
3. **Zero-shot 质量不足**: 在零样本场景下,现有模型的 style similarity 和 audio quality 仍有提升空间 [§1] [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Make-A-Voice 采用 coarse-to-fine 三阶段设计 [Fig 1]:

**Stage S1 - Semantic Stage**: 将 text 或 speech 映射为 semantic tokens [§3.4]
- TTS: phoneme sequence → autoregressive encoder-decoder Transformer → semantic tokens [Eq. 1]
- VC: speech → HuBERT → semantic tokens (无需 seq2seq)

**Stage S2 - Acoustic Stage**: 将 semantic tokens 转化为 acoustic tokens,注入 acoustic conditions [§3.5]
- 输入: semantic tokens $s$ + speaker prompt acoustic tokens $a_p$ (+ F0 prompt $F$ for SVS) [Eq. 2-3]
- 自回归 Transformer 预测 acoustic tokens $a$
- Self-supervised audio-only data 可大规模训练 (无需文本标注)

**Stage S3 - Generation Stage**: 从 acoustic tokens 重建高保真波形 [§3.6]
- Unit-based vocoder (BigVGAN-inspired) 从 3 层 acoustic tokens 合成波形
- 不用 SoundStream decoder,避免 training (12 levels) vs inference (3 levels) 的 codebook mismatch

### 关键设计选择

#### 1. 离散语音表示 [§3.3]

**Semantic tokens** [§3.3.1]:
- HuBERT 12th layer → k-means ($K_1$ clusters) → 离散 semantic tokens
- 20ms per frame, 50 Hz frame rate
- 为什么选 12th layer: ablation 显示 HuBERT-12 CER 最低 (0.39 vs HuBERT-10 的 0.54) [Table 5]

**Acoustic tokens** [§3.3.2]:
- SoundStream encoder → RVQ (12 levels, $K_2=1024$, 20ms frame) → 离散 acoustic tokens
- 与 semantic tokens 同样 20ms frame rate,便于对齐
- 推理时只取前 3 层 tokens 作为 vocoder 输入

**WHY 两种 token**: semantic tokens 捕获语言内容 (what is said),acoustic tokens 捕获声学特性 (how it sounds)。分离后可独立控制内容和风格 [§3.2] [论文原文]。[agent 解读] 这直接遵循 AudioLM 提出的 semantic→acoustic 层级范式。

#### 2. Acoustic Stage 的条件化机制 [§3.5]

训练时从每个样本随机抽取两段不重叠的窗口:
- 一段作为 prompt (提供 speaker/style 信息)
- 另一段作为 target output [§3.5]

条件注入方式: 将 prompt acoustic tokens $a_p$ 和 semantic tokens $s$ 按顺序 concatenate,中间插入 separating token [Eq. 2]:
$$p(a | s, a_p; \theta_{AR}) = \prod_{t=0}^{T} p(a_t | a_{<t}, s, a_p; \theta_{AR})$$

**SVS 扩展** [§3.5]: 额外引入 quantized F0 prompt $F$:
- YAPPT 提取 F0 (320 hop, log-scale)
- 量化为 256 level integer tokens ($K_f = 256$) [§3.5]
- $F$, $a_p$, $s$ 三者 concatenate 并用 separating tokens 分隔 [Eq. 3]

**WHY quantized F0**: SVS 需要精确 pitch control,而 semantic tokens 不包含 F0 信息。离散化 F0 使其能与 token 序列统一处理 [§3.5] [论文原文]。

#### 3. Unit-based Vocoder vs SoundStream Decoder [§3.6]

**关键 ablation 发现** [Table 5]:
- SoundStream decoder: STOI 0.92, MCD 1.90
- Unit vocoder: STOI **0.93**, MCD **1.56**

**WHY**: SoundStream 训练用 12 层 RVQ,推理只用 3 层 → codebook mismatch 导致"a distinct drop in perceptual quality" [§3.6] [论文原文]。Unit vocoder 专门为 3 层 coarse tokens 训练,无 mismatch 问题。

**Vocoder 架构**: 
- Generator: LUT (lookup table) embedding + transposed conv + dilated residual blocks (BigVGAN inspired) [§3.6]
- Discriminator: MRD (multi-resolution discriminator) [§3.6]
- Training: least-square adversarial + feature matching + spectral regression on mel [§3.7.1]
- SVS 版: 额外引入 F0-driven source excitation (harmonic source) 稳定长音合成 [§3.6]

#### 4. 三任务统一推理 [§3.7.2, Fig 3]

通过 re-synthesizing 不同 stages 的 representation 实现任务统一:

| Task | S1 Input | S1 Output | S2 Condition | S2 Output | S3 |
|------|----------|-----------|-------------|-----------|-----|
| VC | Source speech → HuBERT | Semantic tokens | Target speaker prompt | Acoustic tokens | Vocoder |
| TTS | Phonemes → seq2seq | Semantic tokens | Speaker prompt | Acoustic tokens | Vocoder |
| SVS | Phonemes → seq2seq | Semantic tokens | Speaker + F0 prompt | Acoustic tokens | F0-conditioned vocoder |

**WHY 统一框架可行**: S2 和 S3 在三个任务间完全共享参数,只需为每个任务适配 S1 [论文原文]。这得益于 semantic/acoustic 分离使得 content 和 style 可独立注入 [agent 解读]。

### 训练策略

- S1 和 S2: 12-layer Transformer (1024 embed, 4096 FFN), cross-entropy with label smoothing [§4.1.2, §3.7.1]
- S1 训练: 4 V100 GPUs, 100K steps, batch 2000 tokens, fairseq framework [§4.1.3]
- S2 训练: 32 V100 GPUs, 500K steps, batch 2000 tokens, crop up to 8s [§4.1.3]
- S3 训练: 4 V100 GPUs, 500K steps, segment 8192 samples [§4.1.3]
- Adam optimizer: $\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$, lr $1\times10^{-4}$ [§4.1.3]
- Beam search: size 10 for both S1 and S2 inference [§4.1.3]

**数据策略** [Table 1]:
- English: S1 on LibriTTS train, S2/S3 on LibriLight (60K hours, 7K speakers)
- Chinese: S1 on OpenCPOP train, S2/S3 on OpenCPOP+OpenSinger+CSMSC
- **关键**: S2/S3 只需 audio-only data,因此可利用海量无标注数据 [论文原文]

## 实验

### Text-to-Speech (Table 2)

| 指标 | Make-A-Voice | GenerSpeech | YourTTS | VALL-E | SPEAR-TTS | GT | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOS | **4.04** | 3.99 | 3.89 | 3.92 | 3.98 | 4.23 | [Table 2] |
| SMOS | **3.81** | 3.77 | 3.72 | 3.81 | **3.84** | - | [Table 2] |
| CER | 0.068 | 0.059 | 0.143 | - | - | 0.030 | [Table 2] |
| Cos | **0.77** | 0.75 | 0.72 | - | - | - | [Table 2] |

- MOS 4.04 超越所有 zero-shot 基线; SMOS 3.81 接近 SPEAR-TTS 的 3.84 [论文原文]
- Small-scale 对比: MOS 4.05 > VALL-E 3.92, SPEAR-TTS 3.98 [Table 2]

### Voice Conversion (Table 3)

| 指标 | Make-A-Voice | NANSY | PPG-VC | GT | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS | **4.07** | 3.89 | 3.97 | 4.26 | [Table 3] |
| SMOS | 3.77 | 3.73 | **3.80** | - | [Table 3] |
| Cos | **0.80** | 0.68 | 0.78 | - | [Table 3] |

- MOS 4.07 和 Cos 0.80 均为最高; SMOS 略低于 PPG-VC [Table 3]
- 无需 text transcript 训练,仅用 audio data [论文原文]

### Singing Voice Synthesis (Table 4)

| 指标 | Make-A-Voice | FFT-Singer | Diffsinger | GT | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS | **3.99** | 3.83 | 3.98 | 4.08 | [Table 4] |
| SMOS | 3.96 | 3.91 | **3.98** | - | [Table 4] |
| Cos | 0.88 | **0.93** | 0.94 | - | [Table 4] |
| FFE | **0.05** | 0.17 | 0.08 | - | [Table 4] |

- MOS 和 FFE 最优; Cos 偏低因为零样本场景 (训练时未见 test singers) [§4.4] [论文原文]
- FFE 0.05 远优于 Diffsinger 0.08,证明 F0-guided acoustic stage 的精确 pitch control [Table 4]

### Ablation (Table 5)

| 模型 | CER | STOI | MCD |
| --- | --- | --- | --- |
| HuBERT-10 | 0.54 | - | - |
| HuBERT-11 | 0.44 | - | - |
| **HuBERT-12** | **0.39** | - | - |
| S3: SoundStream decoder | - | 0.92 | 1.90 |
| **S3: Unit Vocoder** | - | **0.93** | **1.56** |

- HuBERT 12th layer 提供最佳 CER → 选为 semantic tokens [§4.5.1]
- Unit vocoder 在 STOI 和 MCD 上均优于 SoundStream decoder [§4.5.2]

### Zero-shot Transfer (§4.5.3)

Demo 页展示三项扩展能力:
1. **Cross-lingual**: 从跨语言 prompt 复制 style [§4.5.3]
2. **Emotion preservation**: 在非情感数据集训练的模型可零样本保留 prompt 情感 [§4.5.3]
3. **Noise consistency**: 复制 prompt 中的背景噪声特性 [§4.5.3]

## 局限性

1. **三阶段串联**: 推理链条长 (S1→S2→S3),每阶段误差可能累积; beam search 增加计算开销 [agent 解读]
2. **HuBERT 语言限制**: HuBERT 为 mono-lingual,需为每种语言独立训练 semantic tokenizer [Table 1] [论文原文]
3. **评估规模小**: 每个任务仅 50 个测试句,统计显著性有限 [§4.1.1]
4. **无代码开源**: 仅提供 demo 页面,无法复现 [论文原文]
5. **Cos similarity 较低 (SVS)**: 0.88 低于 supervised baselines (0.93-0.94),因为零样本 singer 未在训练中见过 [§4.4]
6. **Noise reproduction**: 与 HierSpeech++ 类似,会复制 prompt 中的噪声 [§4.5.3]

## 点评

**优势**:
- **统一框架思想**: 首次证明 TTS/VC/SVS 可通过共享 semantic-to-acoustic backbone 统一处理,S2 和 S3 完全共享 [论文原文]
- **数据 scalability**: S2/S3 仅需 audio data,LibriLight 60K 小时可直接使用
- **Coarse-to-fine 的清晰层次**: semantic tokens (内容) → acoustic tokens (风格) → waveform (保真度) 的分工明确
- **SVS 中的 F0 conditioning**: quantized F0 作为 acoustic stage 条件是简洁有效的 pitch control 方案

**不足**:
- 发表时间接近 VALL-E 但影响力较小,可能因为未开源 [agent 解读]
- 与 HierSpeech++ 对比: 后者非自回归 + 并行推理更快; Make-A-Voice 的 AR 方式在鲁棒性上可能不如 [agent 解读]
- 实验对比缺少与 VALL-E 的同条件公平比较 (数据集不同)
- 指标体系不够全面: 无 WER, 无 UTMOS, 无推理速度对比

## 可复用的 idea

1. **Unit-based vocoder 替代 codec decoder**: 当 RVQ 训练/推理层数不匹配时,训练专用 vocoder 从 coarse tokens 合成波形,避免 codebook mismatch -- 通用且有效
2. **Quantized F0 as token prompt**: 将连续 F0 量化为离散 tokens 并 concatenate 到 prompt 中,简单实现 pitch control
3. **Semantic-acoustic 两阶段的 data efficiency**: S2 (core backbone) 只需 audio-only data,可利用海量无标注音频
4. **同帧率对齐**: semantic/acoustic tokens 使用相同 downsampling rate (320, 20ms frame),简化跨阶段对齐
5. **Separating tokens**: 多条件 concatenation 时用 separating tokens 分隔不同来源的 token 序列

检索命中: [[Semantic vs Acoustic Tokens]], [[Residual Vector Quantization]], [[Neural Vocoder]], [[LLM-based TTS]], [[Speech Factorization]], [[Speaker Embedding]] | 过滤: [[F0 Modeling]](pending-review), [[Mel Spectrogram]](pending-review) | 未命中但可能相关: [[Singing Voice Synthesis]]
