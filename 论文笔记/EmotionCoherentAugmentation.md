---
type: paper
tier: deep
title: "Emotion-Coherent Speech Data Augmentation and Self-Supervised Contrastive Style Training for Enhancing Kids's Story Speech Synthesis"
arxiv_id: "2602.10164"
source: "Sources/EmotionCoherentAugmentation.pdf"
authors: [Raymond Chung]
year: 2024
venue: "IEEE SLT 2024"
tags: [TTS, expressive-speech, data-augmentation, contrastive-learning, GST, audiobook, emotion, style-embedding, long-form-TTS]
concepts: ["[[GlobalStyleTokens]]", "[[EmotionControlinTTS]]", "[[ProsodyModeling]]", "[[Self-SupervisedSpeechRepresentation]]", "[[StyleTransferinTTS]]", "[[MelSpectrogram]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[GlobalStyleTokens]], [[EmotionControlinTTS]], [[ProsodyModeling]], [[Self-SupervisedSpeechRepresentation]], [[StyleTransferinTTS]], [[MelSpectrogram]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文处于 GST-Tacotron (2018) 之后、LLM-TTS 时代之前的"显式风格建模"范式中。GST 通过 reference encoder + style token bank 从音频中自动发现风格维度,TP-GST 进一步实现从文本预测风格 embedding。本文在这一框架基础上,提出两个训练策略改进: (1) emotion-coherent 数据增强替代随机拼接; (2) SimCLR contrastive loss 改善 reference encoder 的风格提取一致性。
>
> **已有认知**: 概念库中 [[GlobalStyleTokens]] 详细记录了 GST 的 reference encoder → style token bank → multi-head attention 机制,以及后续演进方向 (多级风格/解耦/扩散增强等)。[[EmotionControlinTTS]] 涵盖了从 emotion embedding 到 DPO/RLHF 优化的大量现代方法。[[ProsodyModeling]] 记录了从 reference encoder 到 VAE 到生成模型的韵律建模演进。[[Self-SupervisedSpeechRepresentation]] 聚焦 speech SSL (wav2vec 2.0/HuBERT/WavLM),本文涉及的是 text SSL (T5) 用于数据增强,是不同的应用场景。
>
> **创新判断**: 本文的贡献不在于提出新架构,而在于训练策略层面 -- 用 text SSL 驱动 emotion-coherent 数据增强来改善 GST 训练,以及首次将 SimCLR contrastive loss 应用于 GST reference encoder。这些都是低成本、可插拔的改进,但处于 pre-LLM 时代,与当前主流方法 (in-context learning, flow matching, codec LM) 有较大代差。
>
> 检索命中: [[GlobalStyleTokens]][待确认], [[EmotionControlinTTS]][待确认], [[ProsodyModeling]]✓, [[Self-SupervisedSpeechRepresentation]][待确认], [[StyleTransferinTTS]][待确认], [[MelSpectrogram]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 text emotion classifier 驱动 emotion-coherent 语音拼接增强 + SimCLR contrastive loss 改善 GST reference encoder,提升小数据集上有声书 TTS 的表现力
> - **路线**: 单句音频 → T5 emotion 分类 → 同情感句拼接(插入正态分布停顿) → Tacotron2 + TP-GST 训练(+ SimCLR loss on reference encoder) → mel-spectrogram → WaveGlow → 多句有声书语音
> - **指标**: TP-GST L1 loss 0.075 (M4) vs 0.155 (M2 baseline); emotion classification 75.3% vs 70.7%; MOS naturalness 3.25 vs 3.19; K-S test p=0.630 (M3) vs p=0.027 (M1) on inter-sentence pause [Table IV, V, §IV-C]
> - **可借鉴**: emotion-coherent 拼接思路可迁移到任何需要多句训练的 TTS 系统; SimCLR on reference encoder 是简单有效的正则化手段; 用正态分布建模句间停顿是低成本但有效的方法
> - **局限**: 仅 6.5h 单说话人数据,MOS 仅 3.25 且置信区间重叠; 基于 Tacotron2+WaveGlow 的过时架构; 仅 8 名评估者; 无与现代 LLM-TTS 的对比

## 核心问题

本文要解决的问题是: **在小规模表现力语音数据集 (6.5h) 上,如何训练一个能生成多句、情感一致的有声书语音的 TTS 模型?**

具体而言,面临三个子问题:
1. **数据不足**: Blizzard Challenge 2017 数据集仅 6.5h,直接切成多句 utterance 会导致训练数据量锐减 (3-sentence 只剩 2409 条 vs 单句 5150 条) [Table I]
2. **情感不一致**: 连续句子中仅 8% 有相同的非中性情感标签 [§IV-A],直接拼接连续句会让 GST 难以从混合情感的音频中学到干净的 style token
3. **句间停顿缺失**: 单句训练的模型无法学习自然的多句间停顿模式 [§IV-C]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

基础架构是 Tacotron2 + TP-GST + WaveGlow vocoder [§III]:
- **Tacotron2**: phoneme → mel-spectrogram,使用 stepwise monotonic attention + reduction factor 2 以支持更长输出 [§III]
- **TP-GST**: 从 mel-spectrogram 提取 style embedding (训练时),从 text 预测 style embedding (推理时) [§III]
- **WaveGlow**: 预训练 vocoder,mel → waveform [§III]

训练分三阶段 [§IV-A]:
1. LJSpeech 预训练 (300 epochs) — 基础语音能力
2. LibriTTS 训练 (200 epochs) — style tokens 和 TP-GST 模块
3. Blizzard 2017 微调 (100 epochs) — 目标说话人适应

### 关键设计选择

#### 1. Emotion-Coherent 数据增强 [§III-A, §III-B]

**WHY**: 连续句子的情感经常不一致 (故事中角色对话情感变化频繁),直接拼接连续句会让 GST 从混合情感音频中提取到噪声化的 style embedding [论文原文]。而且,仅 8% 的连续句对拥有相同非中性情感标签 [§IV-A],这意味着大部分多句训练数据对 GST 来说是"情感混乱"的。

**HOW**:
1. 将所有训练音频切分为单句
2. 使用 fine-tuned T5 text emotion classifier (HuggingFace: mrm8488/t5-base-finetuned-emotion, 93% accuracy) 对每句文本做 7 类情感分类 [§IV-A]
3. 阈值校准: 用 LJSpeech (非虚构数据集) 的平均情感分数 0.65 作为参考,设置 0.7 为阈值,低于此值归为 neutral [§IV-A]
4. 将相同情感标签的单句音频拼接成 2-sentence utterance
5. 句间插入基于正态分布 (mean=509ms, std=223ms, 从真实数据拟合) 采样的静音停顿 [§IV-A]

**[agent 解读]**: 这个方法本质上是一种 "semantic-aware data augmentation" -- 不是简单地随机拼接音频(会引入情感不一致的噪声),而是利用 NLP 工具确保拼接后的多句音频在情感维度上保持一致,从而让 GST 的 reference encoder 更容易学到干净的情感 style token。

#### 2. Self-Supervised Contrastive Style Training [§III-C]

**WHY**: GST 的 reference encoder 需要从 mel-spectrogram 中提取稳定的 style embedding,但不同音频片段即使风格相同,也可能因为内容差异导致提取的 embedding 不一致 [论文原文]。

**HOW**:
1. 对每个训练样本的 mel-spectrogram 随机 mask 500ms 片段,生成同一样本的两个 view [§III-C]
2. 不使用 pitch shifting 或 speech perturbation — 这些会改变说话风格本身 [§III-C]
3. 在训练 batch 内,对 reference encoder 输出应用 SimCLR loss: 同一样本的两个 view 为正样本对,batch 内其他样本为负样本 [§III-C]
4. SimCLR loss 以 0.1 的缩放因子加入总训练 loss [§IV-B]

**[agent 解读]**: 这实际上是在对 reference encoder 做 contrastive regularization -- 强制其对同一 utterance 的不同 masked view 产生一致的 style embedding,从而提取更鲁棒、更 style-focused 的表征。augmentation 选择 (仅 masking,不做 pitch/speed 变换) 体现了对 style preservation 的理解: pitch/speed 是 style 的组成部分,不能用作 contrastive augmentation。

#### 3. 推理时的多句合成 [§IV-B]

对于 M2/M3/M4 模型,拼接后的两句文本作为单一输入进入 text encoder,然后按句分割 text embedding,用 TP-GST 分别预测每句的 style embedding。训练时 50% 的概率使用 GST embedding (从音频提取),50% 使用 TP-GST 预测的 embedding,模拟推理过程 [§IV-B]。

### 训练策略

四个模型变体的消融设计 [§IV-B]:
- **M1**: 仅单句训练 (baseline)
- **M2**: 单句 + 连续双句 (传统拼接 baseline)
- **M3**: 单句 + emotion-coherent 双句 (本文方法,无 contrastive)
- **M4**: 单句 + emotion-coherent 双句 + SimCLR contrastive (本文完整方法)

## 实验

| 指标 | M1 | M2 | M3 | M4 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| TP-GST L1 loss | 0.212 | 0.155 | 0.119 | 0.075 | Blizzard 2017 test | [Table IV] |
| Emotion classification (SVM on GST embedding) | 71.5% | 70.7% | 75.1% | 75.3% | ESD (unseen speaker) | [Table IV] |
| MOS Naturalness | — | 3.19±0.56 | — | 3.25±0.56 | Blizzard 2017 test | [Table V] |
| MOS Appropriateness | — | 3.36±0.45 | — | 3.42±0.52 | Blizzard 2017 test | [Table V] |
| K-S statistic (pause distribution vs GT) | 0.490 (p=0.027) | — | 0.247 (p=0.630) | — | Blizzard 2017 test | [§IV-C] |

**关键发现**:
1. **Emotion-coherent augmentation 显著改善 style embedding quality**: M3 的 TP-GST L1 loss 从 M2 的 0.155 降至 0.119,emotion classification 从 70.7% 提升至 75.1% [Table IV]
2. **SimCLR contrastive 进一步改善**: M4 将 L1 loss 进一步降至 0.075 (比 M2 减半),但 emotion classification 仅微升至 75.3% [Table IV]
3. **句间停顿更自然**: M3 的 K-S p-value=0.630 (无法拒绝与真实分布相同的零假设),而 M1 的 p-value=0.027 (拒绝,分布显著不同) [§IV-C]
4. **主观评估改善有限**: MOS naturalness 3.25 vs 3.19,appropriateness 3.42 vs 3.36,置信区间高度重叠 [Table V]

## 局限性

1. **过时的技术栈**: Tacotron2 + WaveGlow 是 2018-2019 年的架构,与 2024 年主流的 LLM-TTS / flow matching / codec LM 方法有巨大代差。论文作者自己也指出 F0 range 问题可能与 vocoder 训练数据有关 [§IV-D]
2. **极小规模实验**: 仅 6.5h 单说话人数据 + 8 名评估者,MOS 置信区间严重重叠 (3.19±0.56 vs 3.25±0.56),统计显著性不足
3. **无消融分离 emotion-coherent vs random augmentation**: M2 用的是连续句拼接,而不是随机句拼接。如果加一个"随机句拼接"baseline,才能更严格地证明 emotion-coherent 的价值
4. **文本情感分类器的可靠性**: T5 classifier 在故事文本上的准确率未验证(仅报告原始测试集 93%),且 7 类情感中某些类别极少 (love 1.16%, surprise 0.61%) [Table II],这些稀疏类的分类质量存疑
5. **仅英语,仅童话**: 结论能否泛化到其他语言、其他领域 (新闻、对话) 未知
6. **无端到端推理评估**: 论文仅评估了 GST embedding 质量和短句 MOS,未评估整页童话的长文本合成质量

## 点评

这是一篇发表在 SLT 2024 的应用型短论文,提出了两个简单但有道理的训练改进策略。

**值得肯定的地方**:
- **问题洞察准确**: "连续句子情感不一致 → GST 学到噪声化 embedding" 这个观察是对的,8% 同情感连续句的统计数据有说服力
- **方法简单可复用**: emotion-coherent 拼接 + SimCLR on reference encoder 都是低成本改进,可以插入任何 GST-based 系统
- **评估维度全面**: 同时做了 embedding 质量 (L1 loss)、情感区分度 (SVM)、停顿分布 (K-S test) 和主观 MOS 四个维度的评估

**主要不足**:
- **时代感强烈**: 2024 年 SLT 的论文仍在用 Tacotron2 + WaveGlow,与 CosyVoice/F5-TTS 等同期工作的技术水平差距明显
- **评估统计力不足**: 8 名评估者、MOS 置信区间完全重叠,难以声称主观改善有统计意义
- **消融不完整**: 缺少"随机拼接"baseline 来隔离 emotion-coherent 的贡献

**在知识库中的位置**: 这篇论文是 GST 应用改进的一个有趣案例,展示了 data augmentation 策略对 style embedding 学习的影响。但由于架构过时、规模小、评估弱,对理解当前 TTS 技术格局的参考价值有限。核心 takeaway 是 "emotion-coherent data curation 比随机拼接更有利于 style 学习" 这一定性结论。

## 可复用的 idea

1. **Emotion-coherent data curation 原则**: 在构建多句训练数据时,按情感/风格一致性筛选拼接,而非简单取连续句。这个原则可以迁移到任何需要多句或长文本训练的现代 TTS 系统 (如 Audiobook-CC 的数据处理)
2. **Contrastive regularization for style encoder**: 用 SimCLR loss 对 style/reference encoder 做正则化,强制其对同一 utterance 的不同扰动产生一致 embedding。这可以应用于任何使用 reference encoder 的系统 (包括现代 flow matching TTS 的 style conditioning)
3. **句间停顿的正态分布建模**: 从真实数据拟合停顿分布 → 在数据增强中采样插入,是一种简单有效的长文本 TTS 训练技巧
