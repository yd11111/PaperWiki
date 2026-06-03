---
type: concept
title: "Self-Supervised Speech Representation"
aliases: [自监督语音表征, SSL Speech Pre-training, Speech Self-Supervised Learning, 语音自监督预训练, Self-Supervised Speech Pre-training, SSL for Speech]
category: "model-family"
tags: [self-supervised-learning, speech-representation, contrastive-learning, masked-prediction, pre-training, ASR, speech-tokenizer]
key_papers: ["[[论文笔记/wav2vec 2.0|wav2vec 2.0]]", "[[论文笔记/HuBERT|HuBERT]]", "[[论文笔记/WavLM|WavLM]]", "[[论文笔记/w2v-BERT|w2v-BERT]]", "[[论文笔记/RepCodec|RepCodec]]", "[[论文笔记/Seed-VC|Seed-VC]]", "[[论文笔记/USM-VC|USM-VC]]", "[[论文笔记/SSL Suprasegmental Analysis|SSL Suprasegmental Analysis]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/XEUS|XEUS]]", "[[论文笔记/BEATs|BEATs]]", "[[论文笔记/w2v-BERT 2.0|w2v-BERT 2.0]]", "[[论文笔记/Seamless|Seamless]]", "[[论文笔记/TTSDS2|TTSDS2]]", "[[论文笔记/SemaVoice|SemaVoice]]", "[[论文笔记/NAST|NAST]]", "[[论文笔记/TTSDS|TTSDS]]"]
origin_paper: "van den Oord et al., Representation Learning with Contrastive Predictive Coding (CPC), 2018"
related_concepts: ["[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Gumbel-Softmax]]", "[[Codebook Collapse]]", "[[Speech Language Model]]", "[[Masked Generative Modeling]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-03
updated: 2026-06-03
---

## 定义

Self-Supervised Speech Representation (SSL for speech) 是从大量无标注语音数据中学习通用语音表征的预训练范式。核心思想是设计代理任务 (proxy task),使模型在无标注情况下学到对下游任务有用的表征。

根据预训练目标,SSL 方法可分为三大范式:

### 1. Contrastive Learning (对比学习)

通过区分正样本 (同一位置的表征) 和负样本 (其他位置) 学习表征:
- **CPC** (van den Oord et al., 2018): 预测未来帧的 InfoNCE loss [论文原文]
- **wav2vec** (Schneider et al., 2019): 对原始波形做 contrastive prediction
- **wav2vec 2.0** (Baevski et al., NeurIPS 2020): mask + contrastive on quantized targets [§3.2]
- **特点**: 端到端可微量化 (Gumbel-Softmax PQ); 连续输入 + 量化目标 [wav2vec 2.0 Table 4]

### 2. Masked Prediction (掩码预测)

mask 输入的一部分,预测被 mask 位置的离散标签:
- **HuBERT** (Hsu et al., 2021): 离线 k-means 聚类产生伪标签 + masked cross-entropy loss [§II]
- **核心洞察**: 标签的一致性 (consistency) 比正确性 (correctness) 更重要 [HuBERT §II]
- **特点**: 迭代 refinement (聚类 → 预训练 → 再聚类); 仅在 masked region 计算 loss
- **优势**: 训练稳定,无需复杂的量化模块; 可产生高质量 semantic tokens

### 3. Combined (联合)

同时优化 contrastive 和 masked prediction 两个目标:
- **w2v-BERT** (Chung et al., ASRU 2021): contrastive module 产生 token IDs + masked prediction module 预测 IDs [§3]
- **WavLM** (Chen et al., 2022): HuBERT masked prediction + masked speech denoising [§IV]
- **优势**: contrastive loss 防止 codebook collapse [w2v-BERT §5.2]; MLM loss 学习高层语义
- **w2v-BERT 发现**: 纯 contrastive (无 MLM) ASR 性能明显更差; 纯 MLM (无 contrastive) 导致 codebook collapse [w2v-BERT §5.2, Fig 2]

## 在 TTS 中的应用

SSL 语音模型在 TTS 系统中主要作为 semantic tokenizer:
- **HuBERT k-means tokens**: 被 GSLM, AudioLM, pGSLM, TWIST 等 SpeechLM 系统广泛采用
- **w2v-BERT 2.0 tokens**: 被 AudioLM 采用,MaskGCT 用 VQ-VAE 量化其第 17 层特征
- **WavLM tokens**: 被 Mimi (Moshi) 用作 semantic 信号源; Survey benchmark 中声学建模最强

对于 TTS 系统,SSL tokenizer 的价值在于:将语音的高层语义信息压缩为离散 token,供 LLM 生成,然后由 vocoder/CFM 恢复声学细节。详见 [[Semantic vs Acoustic Tokens]]。

## 关键设计 Trade-offs

| 设计选择 | 方案 A | 方案 B | 典型工作 |
|---------|--------|--------|---------|
| 量化方式 | 在线 (Gumbel-Softmax) | 离线 (k-means) | wav2vec 2.0 vs HuBERT |
| 预训练目标 | Contrastive only | Masked prediction only | wav2vec 2.0 vs HuBERT |
| 位置编码 | Convolutional relative PE | Gated relative PE | wav2vec 2.0 vs WavLM |
| Building block | Transformer | Conformer | wav2vec 2.0 vs w2v-BERT |
| 输入扰动 | Clean speech | Noisy/overlapped | HuBERT vs WavLM |
| 预训练数据 | Audiobook only | Multi-domain | HuBERT (960h) vs WavLM (94k) |

## 关键论文

- CPC (van den Oord et al., 2018): 首个 contrastive speech representation
- wav2vec (Schneider et al., Interspeech 2019): 首个端到端 contrastive on waveform
- vq-wav2vec (Baevski et al., ICLR 2020): 引入 VQ,两阶段
- **wav2vec 2.0** (Baevski et al., NeurIPS 2020): contrastive + Gumbel-Softmax PQ,端到端,10min labeled → 4.8/8.2 WER
- **HuBERT** (Hsu et al., 2021): masked prediction + offline k-means,迭代 refinement
- **w2v-BERT** (Chung et al., ASRU 2021): 首个端到端 contrastive + masked prediction; 证明 contrastive 防 codebook collapse
- **WavLM** (Chen et al., 2022): masked speech denoising,首个 full-stack SSL (SUPERB + speaker + separation + diarization)

## 相关概念

- [[Speech Tokenizer]]: SSL 模型的表征经离散化后成为 speech tokenizer
- [[Semantic vs Acoustic Tokens]]: SSL 表征产生的是 semantic tokens
- [[Gumbel-Softmax]]: wav2vec 2.0 端到端量化使用的技术
- [[Codebook Collapse]]: contrastive loss 是端到端量化中防 collapse 的关键 (w2v-BERT)
- [[Speech Language Model]]: 消费 SSL 表征/tokens 的下游模型
- [[Masked Generative Modeling]]: 与 SSL 的 masked prediction 训练有关联但目标不同 (理解 vs 生成)

## 超音段韵律的 Layer-wise 表征 [de la Fuente & Jurafsky, 2024]

对 wav2vec 2.0, HuBERT, WavLM 三个 12 层 BASE 模型的 probing 分析揭示了 SSL 表征的内在结构:
- **中间层 (8-9) 对超音段分类最强**: stress/tone/accent 的 F1 在 layer 8-9 达到 peak [Fig 1]
- **超音段表征是抽象的**: F0 regression 的 peak 不与超音段 peak 重合,说明模型习得的是抽象语言学类别,不是简单的 F0 追踪 [Fig 1, panel 4]
- **语言特异性仅在 Transformer 层**: CNN 层 (layer 0) 对所有模型/语言表现一致,上下文网络才编码语言特定信息 [Fig 1]
- **ASR fine-tuning 增强词级韵律**: stress 和 tone (lexical features) 通过正字法间接获益; phrasal accent 增强较弱 [Fig 2]
- **三种 SSL 预训练目标表现相似**: HuBERT, WavLM, wav2vec 2.0 的 layer-wise 趋势高度一致 [Fig 3]

这些发现对 speech tokenizer 设计有指导意义: SSL 中间层已编码丰富的韵律信息; 如果需要 prosody-aware tokens,应选择中间层而非最后层。详见 [[论文笔记/SSL Suprasegmental Analysis|SSL Suprasegmental Analysis]]。

## 演进

CPC (contrastive, 2018) → wav2vec (contrastive on waveform, 2019) → vq-wav2vec (VQ + BERT, 两阶段, 2020) → **wav2vec 2.0** (contrastive + Gumbel-Softmax PQ, 端到端, 2020) → **HuBERT** (masked prediction + k-means, 迭代, 2021) → **w2v-BERT** (contrastive + MLM, 端到端, 2021) → **WavLM** (masked denoising, full-stack, 2022) → **BEATs** (iterative acoustic tokenizer + discrete label prediction, 通用音频 SSL, 2022) → Whisper encoder (弱监督替代自监督, 2022) → **w2v-BERT 2.0** (contrastive + MLM, 580M params, 4.5M hours, 143 languages, Seamless, 2023) → **XEUS** (masked prediction + denoising + dereverberation, E-Branchformer, 1M hours, 4057 languages, ML-SUPERB SOTA, 2024) → 监督式 tokenizer (CosyVoice S3, 2024)
