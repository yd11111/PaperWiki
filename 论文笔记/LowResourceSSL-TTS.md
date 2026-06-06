---
type: paper
tier: deep
title: "Low-Resource Self-Supervised Learning with SSL-Enhanced TTS"
arxiv_id: "2309.17020"
source: "Sources/LowResourceSSL-TTS.pdf"
authors: [Po-chun Hsu, Ali Elkahky, Wei-Ning Hsu, Yossi Adi, Tu Anh Nguyen, Jade Copet, Emmanuel Dupoux, Hung-yi Lee, Abdelrahman Mohamed]
year: 2023
venue: "ICASSP 2024"
tags: [self-supervised-learning, low-resource, TTS, data-augmentation, HuBERT, discrete-units, speech-representation]
concepts: ["[[Self-SupervisedSpeechRepresentation]]", "[[SemanticvsAcousticTokens]]", "[[SpeakerEmbedding]]", "[[DurationPredictor]]", "[[Text-to-SpeechPipeline]]"]
models: ["[[HuBERT]]", "[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页: [[SemanticvsAcousticTokens]], [[SpeakerEmbedding]], [[Self-SupervisedSpeechRepresentation]], [[HuBERT]], [[DurationPredictor]], [[Text-to-SpeechPipeline]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SemanticvsAcousticTokens]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[HuBERT]](pending-review), [[DurationPredictor]](pending-review), [[Text-to-SpeechPipeline]](pending-review) | 未命中但可能相关: 无

**谱系定位**: 本文处于 SSL 语音表征学习与 TTS 数据增强的交叉点。在 SSL 谱系中,HuBERT (2021) 通过离线 k-means 聚类 + masked prediction 建立了迭代 refinement 范式,其离散单元 (semantic tokens) 后来成为 SpeechLM 和 unit-based TTS 的核心表征。本文反向利用这条路径: 用 HuBERT 的离散单元构建低资源 TTS,再用 TTS 合成数据反哺 HuBERT 预训练,形成 SSL↔TTS 双向循环。

**已有认知**: KB 中 SemanticvsAcousticTokens 页面记录了 HuBERT k-means tokens 作为 semantic tokens 的定位 — 与文本对齐好但缺乏高频声学细节; HuBERT 页面指出其核心洞察"标签一致性 > 正确性"; SpeakerEmbedding 页面详述了 x-vector 作为 speaker encoder 的标准方案; DurationPredictor 页面梳理了从 FastSpeech 到现代 RL-optimized duration 的演进线。

**创新判断**: 本文的核心创新不在于单个组件,而在于系统级 idea — 用受限资源建立的 SSL 模型自身产生的离散表征来构建 TTS,再用 TTS 合成数据改善 SSL。这与常见的"用大数据训练 TTS"路径相反,验证了 SSL→TTS→SSL 的自举 (bootstrapping) 可行性。与 KB 中记录的 unit-based TTS 流水线 (GSLM, Polyak et al. 2021 的 unit-to-speech resynthesis) 有密切关联。

## 速查

> [!summary] 速查
> - **一句话**: 用 100h 真实语音预训练的 HuBERT 构建 unit-based TTS,合成大量数据反哺 SSL 预训练,将语音数据需求降低 90%
> - **路线**: 100h real speech → HuBERT pre-train → k-means units → T2U + U2S TTS → synthetic corpus → HuBERT re-train
> - **指标**: WER 15.8% on LibriSpeech dev-other (100h real + 11k synth) vs 14.2% topline (960h real) [Table 5]
> - **可借鉴**: 用 SSL discrete units 降低 TTS 训练难度(10h paired data 就能训练 T2U);合成数据中 oversampling 真实数据(100x)显著提升效果
> - **局限**: 仅在 LibriSpeech (英语阅读语音) 上验证;合成语音语速偏快需后处理拉伸;未评估合成语音质量(MOS)

## 核心问题

本文试图解决: SSL 语音模型 (如 HuBERT) 依赖大量无标注语音数据 (960h+) 进行预训练,当可用语音数据极度有限 (100h) 时如何维持竞争性能?

这一问题的实际动机包括 [§1] [论文原文]:
1. 隐私保护 — 语音数据深度纠缠说话人身份和风格,难以匿名化
2. 低资源语言 — 许多语言缺乏大规模无标注语音数据
3. 数据泄漏风险 — 大规模预训练数据中的说话人信息可能泄露到下游应用

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

三阶段自举 (bootstrapping) 流水线 [Fig 1]:

**Stage I — 低资源 SSL 预训练**:
在 100h 真实语音 (S-100hr) 上预训练 HuBERT BASE (12 层 Transformer),迭代 3 轮。每轮使用前一轮最优模型的特征做 k-means 聚类生成伪标签 [§2.1]。

**Stage II — 基于 SSL 单元的 TTS 系统构建**:
从 Stage I 的 HuBERT 第 9 层提取特征,k-means (k=500) 量化为离散单元。利用这些离散单元训练 4 个模块构成完整 TTS [§2.3]:
- Session Encoder (SE): 提取 x-vector 作为说话人/风格条件
- Text-to-Unit (T2U): 文本→离散单元 (仅需 10h paired data)
- Duration Predictor (DP) + Pitch Predictor (PP): 从去重单元预测时长和 F0
- Unit-to-Speech (U2S): 单元+F0+x-vector→波形 (仅需 unpaired speech)

**Stage III — 合成数据增强 SSL 重训练**:
用 TTS 从大量文本生成合成语音 (最多 11k 小时),与真实数据混合后重新预训练 HuBERT [§2.1]。

### 关键设计选择

**为什么用离散单元而非 mel-spectrogram 作为 TTS 中间表征?** [论文原文] 作者在初步实验中发现,仅用 10h paired data 训练 multi-speaker text-to-mel 模型完全无法收敛,无法生成可理解的语音 [§2.3]。离散单元大幅降低了预测目标的复杂度: (1) 去重后单元序列长度更接近音素序列长度 (ratio 1.7 vs 4.2 without DPDP) [Table 1];(2) 离散目标比连续 mel 更容易学习。[agent 解读] 这本质上是用 SSL 的 representation learning 能力压缩声学空间,将 paired data 的需求从声学模型转移到仅需语音的 variance predictors 和 vocoder。

**为什么采用 utterance-level x-vector 而非 speaker-level 平均?** [论文原文] 为保留语音多样性,不对同一说话人的 x-vector 取平均,而是为每句话提取独立的 x-vector [§2.3.1]。[agent 解读] 这在低资源场景下尤为重要 — 只有 245 个说话人,如果做 speaker-level 平均会丢失说话风格的句内变化,导致合成数据多样性不足。

**DPDP 算法的作用**: 对离散单元序列应用 duration-penalized dynamic programming [§2.2],将平均 unit/phoneme 长度比从 4.2 降至 1.7 [Table 1]。[论文原文] 更短的输出序列有助于 T2U 模型更快收敛和更稳定的 attention 对齐。

**VAE 辅助声学建模**: U2S 中引入 VAE 建模未被 units、F0、x-vector 捕获的残余声学信息 [§2.3.4]。训练时 VAE encoder 从目标 mel 编码 latent,推理时使用标准高斯先验。[agent 解读] 这是对 unit-based resynthesis 中信息瓶颈的补偿 — 离散单元丢弃了说话人相关的声学细节,VAE 提供了一条额外的信息通道。

**数据增强策略** [§2.3.5]:
1. Duration 拉伸: 合成语音比真实语音短约 80%,对每句话将 DP 预测的 duration 乘以 [1.0, 1.5] 的均匀采样标量
2. 背景噪声: 从 MUSAN 数据集加噪 (SNR 0-15 dB),其中 MUSAN 的 speech 部分替换为 S-100hr 以避免引入额外语音

### 训练策略

- HuBERT 预训练: 32 GPU,3 轮迭代,分别 200k/400k/400k steps [§3.2]
- TTS 各模块: 8 GPU (V100),SE 在 1 GPU (2080Ti) 上训练 [§3.2]
- T2U 基于 Tacotron 2 架构,每步预测 2 个 unit (缩短序列) [§2.3.2]
- Fine-tuning: 在 ST-10hr 上微调 40k steps,使用 4-gram LM 解码 [§3.3]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (dev-other) | 25.0% (S0, 100h real) | 14.2% (S1, 960h real) | LibriSpeech | [Table 2] |
| WER (dev-other) | 23.5% (S4, 1.1k synth) | 23.2% (S3-VITS, 809h synth) | LibriSpeech | [Table 5] |
| WER (dev-other) | 20.4% (S4, 11k synth, os=100) | 14.2% (S1, 960h real) | LibriSpeech | [Table 5] |
| WER (dev-other) | 17.5% (S4 + 10k synth FT) | 14.2% (S1, 960h real) | LibriSpeech | [Table 5] |
| WER (dev-other) | 15.8% (S4, iter2) | 14.2% (S1, 960h real) | LibriSpeech | [Table 5] |
| Phone Purity (l9, k500) | 67.32% (S0-3rd) | 68.12% (pre-trained HuBERT) | LibriSpeech | [Table 3] |

### 关键实验发现

1. **100h HuBERT 的退化模式**: 与 960h 版不同,100h HuBERT 在迭代过程中 WER-last 反而变差 (25.0%→33.6%→34.1%),最优 checkpoint (WER-best) 也仅从 31.1% 改善到 25.4% [Table 2]。[论文原文] 这源于过拟合 — 数据量不足导致后续迭代无法持续改善 [§4.1]。

2. **说话人多样性至关重要**: S3 实验中,相同 TTS 模型 (VITS-LibriTTS),使用 24/245/1151/2338 个说话人的 x-vector 生成数据,WER 从 27.0%→26.0%→23.2%→22.2% 持续下降 [Table 4]。[论文原文] 说话人多样性对预训练 SSL 模型至关重要 [§4.2]。

3. **Oversampling 真实数据的显著效果**: 在混合训练集中,将真实数据的采样率从 1x 提升到 100x,WER 从 23.8% 降至 20.4% [Table 5]。[agent 解读] 这表明真实数据的分布特征 (多样化的声学条件、自然韵律) 仍是合成数据无法完全替代的。

4. **合成数据用于 fine-tuning 也有效**: 从 T-LM 生成额外 paired data 用于 fine-tuning,WER 从 20.4% 降至 17.5% (10k hours synth FT) [Table 5]。[论文原文] 注意这些文本数据已被用于构建 4-gram LM 解码器 [§4.3]。

5. **第二轮迭代进一步改善**: 使用改善后的 HuBERT 提取新的离散单元,重建 TTS 并重新生成合成数据,WER 达到 15.8% [Table 5]。[agent 解读] 这验证了自举循环的有效性 — 更好的 SSL→更好的 units→更好的 TTS→更好的合成数据→更好的 SSL。

6. **低资源 TTS 媲美高资源 TTS**: 本文的 unit-based TTS (S4, 用 100h speech + 10h paired) 达到与 VITS-LibriTTS (S3, 用 460h 高质量 TTS 数据) 相近的预训练效果 (23.5% vs 23.2%) [Table 5]。

## 局限性

1. **仅评估 ASR 下游任务**: 只用 WER 评估 SSL 模型质量,未在 speaker verification、emotion recognition 等其他 SUPERB 任务上验证 [agent 解读]
2. **仅在 LibriSpeech (英语阅读语音) 上验证**: 未验证对低资源语言的适用性,而这恰恰是论文动机之一 [§1]
3. **合成语音质量未评估**: 没有 MOS 或 PESQ 等主观/客观语音质量评估,无法判断合成语音的可理解性和自然度 [agent 解读]
4. **语速异常需手动修正**: 合成语音比真实语音短约 80%,需通过 duration scaling 后处理 [§2.3.5],表明 unit-based TTS 在韵律保真度上存在固有缺陷
5. **计算成本未讨论**: 三阶段流水线涉及多次 HuBERT 预训练 (32 GPU) + TTS 训练 + 大规模合成,总计算量可能不低 [agent 解读]
6. **迭代数有限**: 仅测试了 2 轮自举迭代,未探索收敛行为和上限 [agent 解读]

## 点评

本文提出了一个优雅的系统级 idea: SSL 和 TTS 互为手段,形成正反馈循环。核心洞察在于 — HuBERT 的离散单元虽然由质量有限的 100h 模型产生,但其"一致性 > 正确性"的特性使其足以支撑 TTS 训练;而 TTS 合成的数据虽不完美,但能有效扩展 SSL 预训练的数据多样性。

实验设计清晰且控制变量合理: S0 (纯真实 100h) → S2/S3 (高资源 TTS 对照) → S4 (低资源 TTS) 的递进对比有力支撑了结论。说话人多样性和 oversampling 的消融实验提供了有价值的工程 insight。

但论文的评估范围较窄 — 仅限于 ASR 下游任务和英语 LibriSpeech,与论文声称的"低资源语言"和"隐私保护"动机之间存在 gap。此外,与同时期的 SpeechLM、data2vec 等方案缺乏对比。

## 可复用的 idea

1. **SSL↔TTS 自举循环**: 当目标域数据稀缺时,可以用有限数据训练的 SSL 模型提取离散表征 → 构建 TTS → 合成数据 → 改善 SSL。这一范式可推广到任何需要大量无标注数据的自监督学习场景
2. **离散单元降低 paired data 需求**: 将 TTS 的预测目标从连续 mel-spectrogram 切换为离散单元,大幅降低 text-speech paired data 的需求 (从正常的 10h+ 降至可行),因为离散目标空间更小、序列更短
3. **Oversampling 真实数据**: 在混合真实+合成数据训练时,大幅 oversample 真实数据 (100x) 可显著弥补合成数据的分布偏差
4. **Utterance-level x-vector 保持多样性**: 在说话人数量有限的低资源场景,使用句级而非说话人级 speaker embedding 可最大化合成数据的多样性
