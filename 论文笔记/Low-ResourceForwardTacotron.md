---
type: paper
tier: deep
title: "Low-Resource Text-to-Speech Synthesis Using Noise-Augmented Training of ForwardTacotron"
arxiv_id: "2501.05976"
source: "Sources/Low-ResourceForwardTacotron.pdf"
authors: [Kishor Kayyar Lakshminarayana, Frank Zalkow, Christian Dittmar, Nicola Pia, Emanuël A.P. Habets]
year: 2025
venue: "IEEE (preprint)"
tags: [TTS, low-resource, data-augmentation, non-autoregressive, multi-speaker, speaker-adaptation, noise-augmentation]
concepts: ["[[Non-autoregressiveTTS]]", "[[SpeakerAdaptation]]", "[[SpeakerEmbedding]]", "[[NeuralVocoder]]", "[[DurationPredictor]]", "[[MelSpectrogram]]"]
models: ["[[模型库/HierSpeech++|HierSpeech++]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["LJSpeech", "TC-Star", "Hi-Fi-TTS"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文处于 [[SpeakerAdaptation]] 技术谱系的 **数据增强** 分支。与主流的参数高效适应 (AdaSpeech 系列 CLN/adapter) 和零样本方法 (VALL-E/CosyVoice in-context learning) 不同,本文回到了更朴素的数据层面策略 --- 通过噪声增强和采样策略解决低资源说话人的数据不平衡问题。

**已有认知**:
- [[SpeakerEmbedding]] (confirmed): 多说话人 TTS 中编码说话人身份的表示方法。本文使用可学习的 acoustic condition embedding (32-dim) 而非传统 speaker encoder/lookup table,且为 LR 说话人分配 clean/noisy 两个 cond-ID --- 这是一种受限的 speaker embedding 变体。评估时使用 ECAPA-TDNN (当前最常用 SECS encoder) 计算余弦相似度。
- [[Non-autoregressiveTTS]] [待确认]: ForwardTacotron 属于 FastSpeech 系列 NAR 声学模型,使用 CBHG + duration/pitch/energy predictor。NAR 模型的优势 (低推理延迟、无 skip/repeat) 和劣势 (one-to-many mapping → over-smoothing) 在本文中均有体现。
- [[NeuralVocoder]] (confirmed): 本文使用预训练 StyleMelGAN vocoder (Mustafa et al., 2021),属于 GAN-based vocoder,独立于声学模型训练。
- [[HierSpeech++]] [待确认]: 主要 baseline,108.5M 参数,基于层级 VAE 的零样本 TTS,需 2311 个 LibriTTS 说话人训练。本文用仅 4 个 HR 说话人 + 5-20 min LR 数据即在 speaker similarity 上超越它。
- [[SpeakerAdaptation]] [待确认]: Survey 定义的 adaptation 子课题中,本文对应 "Handling Speech Variability → Noisy speech" 分支 (Neekhara et al. 的噪声数据训练) 和 "全模型适应" 架构。
- [[Zero-shotSpeechSynthesis]] (confirmed): 本文不是零样本方法 (需要 LR 说话人数据重新训练),但以零样本方法 HierSpeech++ 为主要比较对象,展示了"少量数据重训"vs"零样本推理"的 trade-off。

**创新判断**: 本文的创新不在于提出新架构或新模型,而在于极简的数据增强策略 (WGN + binned sampling) 在低资源场景下的有效性 --- 这与当前领域追求更大模型、更多数据的趋势形成对比。用 4 个 HR 说话人 + 5 min LR 数据 + 43M 参数即超越需要 2311 说话人 + 108.5M 参数的 HierSpeech++ 的 speaker similarity,这一结果挑战了"数据规模决定质量"的假设。

> 检索命中: [[SpeakerEmbedding]]✓, [[Zero-shotSpeechSynthesis]]✓, [[NeuralVocoder]]✓ | 过滤: [[Non-autoregressiveTTS]](pending-review), [[HierSpeech++]](pending-review), [[SpeakerAdaptation]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 仅用 4 个高质量说话人 + 5-20 分钟目标说话人数据,通过 WGN 噪声增强 + binned/weighted sampling 训练 ForwardTacotron,在 speaker similarity 上大幅超越零样本 HierSpeech++
> - **路线**: 文本 → multi-speaker ForwardTacotron (CBHG + duration/pitch/energy predictors + acoustic condition embedding) → mel spectrogram → StyleMelGAN → 波形
> - **指标**: speaker sim. MOS +1.1 over HierSpeech++ (male); cos-sim 0.69 vs 0.45 (female, 5min); 43M params vs 108.5M (HierSpeech++); naturalness MOS ≈ HierSpeech++ (20min) [Fig 2] [Table II]
> - **可借鉴**: (1) clean/noisy dual condition-ID 让模型既利用噪声增强数据又在推理时只用 clean embedding; (2) binned sampling 让 LR 说话人独占部分 batch 获得充分梯度更新; (3) 仅 4 个 HR 说话人即可构建多说话人基座
> - **局限**: 需要针对每个新说话人重新训练 (非 zero-shot); 1 min 数据时质量下降明显; 仅在英语 + 2 个说话人上验证; vocoder 固定为 StyleMelGAN (非 SOTA)

## 核心问题

本文要解决的问题是: **如何在仅有 5-20 分钟目标说话人数据的情况下,用尽可能少的高资源说话人 (4 个) 实现高质量低资源 TTS?**

现有低资源 TTS 方法的痛点 [§I]:
1. 零样本方法 (HierSpeech++, YourTTS) 需要 40-2311 个说话人训练基座模型,高质量多说话人数据获取困难
2. 多说话人数据集 (VCTK, LibriTTS) 的录音质量参差不齐,质量上限由最差样本决定
3. 现有数据增强方法要么需要 1 小时以上目标数据,要么依赖复杂的预训练 VC/TTS 模型生成增强数据

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

基于 ForwardTacotron [3] 的多说话人扩展 [§II-A] [Fig 1]:
- **声学模型**: CBHG 作为 pre-net 和 post-net,包含 phoneme duration/pitch/energy/voicing confidence 四个 variation predictors [21]
- **Acoustic condition embedding**: 32 维可学习向量,拼接到 pre-net 输出。每个 HR 说话人一个 cond-ID,LR 说话人两个 cond-ID (clean + noisy) [§II-A]
- **Vocoder**: 预训练 StyleMelGAN [20],独立于声学模型

**为什么用 acoustic condition embedding 而非标准 speaker embedding?** [agent 解读] 作者将 LR 说话人的 clean 和 noisy 数据视为不同"声学条件",而非不同"说话人"。这让模型学到: clean 条件 → 干净语音特征,noisy 条件 → 含噪语音特征,同一说话人的身份信息在两个条件间共享。推理时仅用 clean cond-ID,从而输出干净语音。

### 关键设计选择

#### 1. 噪声增强 (Noise Augmentation) [§II-B]

对 LR 说话人的每个样本创建多个 WGN (White Gaussian Noise) 副本,SNR 固定为 20 dB [§III]:
- 5 min / 20 min 子集: 5 倍噪声增强 (每个样本 5 个噪声版本)
- 1 min 子集: 10 倍噪声增强 + 6 倍加权采样

**为什么不用更复杂的增强?** 论文原文 [§II-A] 指出: "Although this WGN augmentation is simple, it enables high-quality synthesis in our approach. Hence, we consider more elaborate augmentation techniques unnecessary." 与前序工作 [18] 使用三种不同噪声类型不同,本文发现仅 WGN 在多说话人 NAR 模型中就足够。[agent 解读] 原因可能是: NAR 模型的 variance predictor (pitch/energy) 已经提供了充分的条件信息,噪声增强主要是增加"样本数量"而非"样本多样性"。

#### 2. 短样本分割 (Sample Splitting) [§II-B]

将 LR 说话人的长句子在语音停顿处用 WhisperX [23] 自动分割为短段。

**为什么需要短样本?** [论文原文] 训练需要大量 LR 句子 (约 1000 句) 以保证稳定性 [§III]。分割增加了句子数量。只对 LR 说话人分割,HR 说话人保留完整句子。作者假设模型可以从 HR 说话人学到长上下文信息 [§II-B],且 LR 说话人本身数据有限,长上下文信息量也有限,因此分割造成的信息损失很小。[§IV-C] 的实验验证: 使用 WhisperX 分割的结果与使用天然短句相当,但使用未分割长句时质量明显下降。

#### 3. Binned Sampling [§II-B]

将训练样本分为 HR bin 和 LR bin,每个 batch 从一个 bin 中随机抽取 (按 bin 大小比例选择 bin)。

**为什么 binned sampling 有效?** [论文原文] 在不平衡训练集中,随机抽取 batch 导致 LR 说话人对权重更新的贡献被 HR 说话人稀释。Binned sampling 确保部分 batch 仅包含 LR 样本,使训练 loss 完全由 LR 样本决定 [§II-B]。这类似于分类任务中的 class imbalance 策略 [22]。

#### 4. Weighted Sampling [§II-B]

增加 LR 说话人样本被抽中的概率,仅在 1 min 数据场景使用 (加权因子 6)。5 min / 20 min 场景不需要。

### 训练策略

- **HR 说话人**: 4 个高质量说话人 (LJSpeech + 2 proprietary + TC-Star/Hi-Fi-TTS),完整使用全部数据 [Table I]
- **LR 说话人**: 从 TC-Star (male) 或 Hi-Fi-TTS-92 (female) 中取 1/5/20 min 子集
- **训练**: batch size 32,300K steps [§IV]
- **目标**: 约 1000 LR 句子 (通过噪声增强 + 分割达到) 保证训练稳定性 [§III]

## 实验

### 客观指标 [Table II]

| 指标 | 本文 (20min) | 本文 (5min) | 本文 (1min) | HierSpeech++ | AdapterMix (20min) | HR ForwardTacotron | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MCD-DTW ↓ | 50.6±0.8 | 53.7±0.8 | 59.7±1.1 | 54.9±0.6 | 53.7±0.8 | 44.6±0.8 | TC-Star | [Table II] |
| cos-sim ↑ | 0.56±0.009 | 0.47±0.012 | 0.34±0.028 | 0.30±0.008 | 0.44±0.011 | 0.65±0.009 | TC-Star | [Table II] |
| MCD-DTW ↓ | 47.2±0.6 | 49.6±0.6 | 50.6±0.6 | 52.8±0.8 | 49.5±0.6 | 39.5±0.6 | Hi-Fi-TTS | [Table II] |
| cos-sim ↑ | 0.75±0.008 | 0.69±0.009 | 0.60±0.009 | 0.45±0.009 | 0.64±0.007 | 0.83±0.007 | Hi-Fi-TTS | [Table II] |

**关键发现**:
1. 本文 (5min) 在 cos-sim 上已超越 HierSpeech++: TC-Star 0.47 vs 0.30 (+0.17), Hi-Fi-TTS 0.69 vs 0.45 (+0.24) [Table II]
2. 即使仅 1 min 数据,cos-sim 仍优于 HierSpeech++ [Table II]
3. AdapterMix 输出存在 vocoder artifacts,未纳入主观测试 [§IV-A]
4. 模型复杂度: 43M (本文) vs 108.5M (HierSpeech++) vs 52M (AdapterMix) [§IV]

### 主观指标 [Fig 2]

以下 MOS 值从 [Fig 2] boxplot 中近似读取,论文仅给出差值数据 (similarity +1.1/+0.25, naturalness -0.2)。

| 指标 | 本文 (20min) | 本文 (5min) | 本文 (1min) | HierSpeech++ | HR | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Naturalness MOS | ≈4.0 | ≈3.5 | ≈2.8 | ≈4.2 | ≈4.5 | [Fig 2] |
| Speaker Sim. MOS | ≈4.3 | ≈3.4 | ≈2.5 | ≈3.2 | ≈4.5 | [Fig 2] |

**关键发现**: 20 min 本文方法 speaker similarity MOS +1.1 over HierSpeech++,但 naturalness 略低 -0.2 [§IV-B]。5 min 方法 speaker sim. MOS +0.25 over HierSpeech++ [§IV-B]。

### 消融实验 [Table II bottom]

| 配置 | MCD-DTW (TC-Star) | cos-sim (TC-Star) | MCD-DTW (Hi-Fi-TTS) | cos-sim (Hi-Fi-TTS) | 出处 |
| --- | --- | --- | --- | --- | --- |
| 5min full | 53.7 | 0.47 | 49.6 | 0.69 | [Table II] |
| 5min w/o binning | 61.3 | 0.31 | 48.7 | 0.69 | [Table II] |
| 5min w/o noise+binning | 61.1 | 0.28 | 53.5 | 0.48 | [Table II] |

**消融结论**: 噪声增强和 binned sampling 对不同数据集影响不同 --- TC-Star 受 binned sampling 影响更大 (cos-sim 从 0.47 降至 0.31),Hi-Fi-TTS 受噪声增强影响更大 (cos-sim 从 0.69 降至 0.48 when both removed)。两者组合提供一致的好结果 [§IV-C]。

## 局限性

1. **非 zero-shot**: 每个新说话人需要重新训练 300K steps,远不如 HierSpeech++ 的推理时 few-shot [§V]
2. **数据量敏感**: 从 20 min 到 1 min,naturalness MOS 下降约 1.2 分 [Fig 2],1 min 场景质量不足以实用
3. **评估范围有限**: 仅在 2 个说话人 (1 male, 1 female) 上验证,且均为英语。主观测试仅用 male speaker [§IV-B]
4. **Vocoder 瓶颈**: 使用 StyleMelGAN 而非更先进的 HiFi-GAN/BigVGAN,vocoder 质量可能限制了整体表现
5. **与现代方法差距**: 在 2025 年的语境下,43M 参数的 ForwardTacotron 基座与 LLM-based TTS (VALL-E, CosyVoice) 的能力差距巨大,后者通过大规模预训练实现了更强的 zero-shot 能力
6. **未考虑跨语言**: 虽然声称可扩展到其他语言 [§V],但缺乏实验验证

## 点评

**优势**:
- 方法极其简单易实现: WGN + binned sampling,无需额外预训练模型或复杂管线
- 明确了一个被忽视的 insight: 低资源 TTS 中数据不平衡问题比模型架构更关键
- 仅需 4 个 HR 说话人 (vs 100+),大幅降低了多说话人 TTS 的数据门槛
- 消融实验虽简单但信息充分,揭示了不同策略对不同数据集的差异化效果

**不足**:
- 论文定位偏传统 (2024 年仍在 ForwardTacotron 上做增量),与 LLM-TTS 时代脱节
- speaker similarity 的提升部分源于比较对象的选择 --- HierSpeech++ 是 2023 年的零样本模型,且使用 LibriTTS 数据训练,并非为 speaker similarity 优化
- naturalness vs similarity 的 trade-off 未被充分讨论: 高 similarity + 低 naturalness 在实际应用中不一定优于低 similarity + 高 naturalness
- 缺少与 AdaSpeech/LoRA fine-tuning 等参数高效方法的对比

## 可复用的 idea

1. **Clean/Noisy dual condition-ID**: 为同一说话人的干净和含噪数据分配不同 condition embedding,推理时仅用 clean ID。这个策略可迁移到任何 conditioning-based 的生成模型中处理数据增强。
2. **Binned sampling for class imbalance**: 将不平衡的多说话人数据按说话人分 bin,确保少数类独占部分 batch。比简单 oversampling 更好地保证梯度更新质量。
3. **最少 HR 说话人基座**: 4 个高质量说话人 (不同性别/来源) 足以构建多说话人 TTS 基座,无需大规模多说话人数据集。这对低资源语言的 TTS 开发尤其有价值。
4. **1000 句阈值**: 经验性发现 NAR 模型 (ForwardTacotron) 在约 1000 个 LR 句子时训练稳定 [§III],具体阈值可能因架构而异,但可作为低资源场景的数据量参考起点。

> [!review] 自动审阅 (2026-06-03)
> **结论: pass-with-fixes** | 0 high, 2 medium, 1 low
> - [medium] traceability-gap: 主观 MOS 值为 boxplot 近似读取 → 已添加说明
> - [medium] template-compliance: datasets 字段为空 → 已补充 LJSpeech/TC-Star/Hi-Fi-TTS
> - [low] weak-reusability: 1000 句阈值限定范围 → 已标注架构依赖性
> 详见 `_review/Low-Resource ForwardTacotron-review.yml`
