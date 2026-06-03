---
type: model
title: "w2v-BERT"
aliases: [w2v-BERT XL, w2v-BERT XXL, w2v-BERT 2.0, W2V-BERT]
org: "Google Brain / MIT"
year: 2021
tags: [self-supervised-learning, speech-representation, contrastive-learning, masked-prediction, conformer, ASR, voice-search]
key_concepts: ["[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Codebook Collapse]]", "[[Self-Supervised Speech Representation]]"]
tasks: []
key_papers: ["[[论文笔记/w2v-BERT|w2v-BERT]]"]
supersedes: ["[[模型库/wav2vec 2.0|wav2vec 2.0]]"]
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

w2v-BERT (Chung et al., ASRU 2021) 首次将 wav2vec 2.0 的 contrastive learning 和 BERT 的 masked language modeling 端到端联合优化用于自监督语音预训练。Contrastive module 产生 discriminative token IDs,masked prediction module 在这些 IDs 上学习高层语义表征。在 LibriSpeech 和 Google Voice Search 上达到 SOTA。

## 核心方法

- **Feature Encoder**: 2 层 2D convolution (stride 2,2) 对 log-mel spectrogram 4x 下采样 [§3.1]
- **Contrastive Module** (N Conformer blocks): 解决 wav2vec 2.0 contrastive task + 产生 token IDs [§3.1]
- **Masked Prediction Module** (M Conformer blocks): 消费 contrastive context vectors,预测 token IDs [§3.1]
- **联合训练**: $\mathcal{L}_p = \beta \cdot \mathcal{L}_c + \gamma \cdot \mathcal{L}_m$ ($\beta=\gamma=1$) [Eq 2]
- **XL 0.6B (12+12 Conformer layers) / XXL 1.0B (12+30)** [Table 1]
- **关键发现**: contrastive loss 是防止 codebook collapse 的必要条件 [§5.2, Fig 2]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| WER test-clean/other (960h, no LM/self-train, XL) | 1.5/2.9 | LibriSpeech | [Table 2] |
| WER test-clean/other (960h, +LM+self-train, XXL) | 1.4/2.5 | LibriSpeech | [Table 2] |
| WER (100h, no LM, XXL) | 2.3/4.0 | LibriSpeech | [Table 3] |
| Voice Search WER (XL) | 6.2 | Google VS | [Table 4] |

## 演进线

wav2vec 2.0 (2020; contrastive only, Transformer) → HuBERT (2021; masked prediction + k-means, Transformer) → **w2v-BERT** (2021; contrastive + masked prediction, Conformer, end-to-end) → w2v-BERT 2.0 (2022; larger scale, used by AudioLM) → WavLM (2022; masked denoising)

## 关键贡献

1. 首次端到端联合 contrastive + masked prediction,无需 HuBERT 的多轮迭代 k-means [§1]
2. 实验证明 contrastive loss 是端到端离散化中防止 codebook collapse 的必要条件 [§5.2, Fig 2]
3. Balanced capacity 原则: contrastive/MLM 模块各 12 层 ($C_{12}$) 为最优 [Table 3]
4. Voice search 实战验证: 比 conformer baseline 相对 WER 降 30% [Table 4]
5. w2v-BERT 2.0 后续成为 AudioLM 的 semantic tokenizer,影响整个 SpeechLM 生态

## w2v-BERT 2.0 (Scaling)

w2v-BERT 2.0 是 w2v-BERT 的大规模多语言扩展 (Barrault et al., 2023):
- **v1** (SeamlessM4T, 2023a): ~600M params, 1M hours, 143 languages
- **v2** (Seamless, 2023b): 580M params, **4.5M hours**, 143 languages
- 架构继承 w2v-BERT 的 contrastive + masked prediction 双模块设计
- 权重公开但训练数据/代码闭源
- 作为 SeamlessM4T v2 的核心语音编码器
- XEUS (Chen et al., 2024) 在 ML-SUPERB 上以更少数据/参数超越 w2v-BERT 2.0 v2 (SUPERB_s 956 vs 826/916),说明预训练目标设计 (dereverberation) 可弥补数据量差距
- 详见 [[论文笔记/w2v-BERT 2.0|w2v-BERT 2.0]]
