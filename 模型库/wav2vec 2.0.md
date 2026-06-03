---
type: model
title: "wav2vec 2.0"
aliases: [wav2vec2, wav2vec 2, w2v2.0]
org: "Facebook AI (Meta)"
year: 2020
tags: [self-supervised-learning, speech-representation, contrastive-learning, ASR, low-resource]
key_concepts: ["[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Gumbel-Softmax]]", "[[Codebook Collapse]]", "[[Self-Supervised Speech Representation]]"]
tasks: []
key_papers: ["[[论文笔记/wav2vec 2.0|wav2vec 2.0]]", "[[论文笔记/Traceable TTS|Traceable TTS]]"]
supersedes: ["vq-wav2vec (Baevski et al., ICLR 2020)"]
superseded_by: ["[[模型库/HuBERT|HuBERT]]", "[[模型库/w2v-BERT|w2v-BERT]]", "[[模型库/WavLM|WavLM]]"]
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

wav2vec 2.0 (Baevski et al., NeurIPS 2020) 是自监督语音表征学习的里程碑模型。通过 contrastive learning 在 latent space 的量化表征上联合学习 contextualized representations 和离散语音单元,首次证明 SSL 预训练 + 极少标注微调可超越半监督 SOTA。

## 核心方法

- **CNN Feature Encoder**: 7 层 CNN,将原始波形编码为 latent representations z (50Hz) [§2]
- **Transformer Context Network**: 将 masked latent representations 转换为 contextualized representations c [§2]
- **Product Quantization + Gumbel-Softmax**: 端到端将 feature encoder 输出离散化为量化 targets q (G=2 codebooks, V=320 entries) [§2]
- **Contrastive Loss**: 在 masked 位置从 K+1 候选中识别正确的量化表征 [§3.2]
- **Diversity Loss**: 最大化 codebook 使用分布的熵,防止 codebook collapse [§3.2]
- **BASE 95M / LARGE 317M**: 两种模型配置 [§4.2]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| WER (test-clean/other, 960h, Transf. LM) | 1.8/3.3 | LibriSpeech | [Table 2] |
| WER (test-clean/other, 10min, Transf. LM) | 4.8/8.2 | LibriSpeech | [Table 1] |
| WER (test-clean/other, 100h, Transf. LM) | 2.3/5.0 | LibriSpeech | [Table 1] |
| PER (dev/test, no LM) | 7.4/8.3 | TIMIT | [Table 3] |

## 演进线

CPC (van den Oord, 2018) → wav2vec (Schneider et al., 2019) → vq-wav2vec (Baevski et al., 2020) → **wav2vec 2.0** (2020; contrastive + Gumbel-Softmax PQ) → HuBERT (2021; masked prediction + offline k-means) → w2v-BERT (2021; contrastive + masked prediction) → WavLM (2022; masked speech denoising for full-stack)

## 关键贡献

1. 首次证明 SSL 预训练 + 极少标注 (10min) 微调可达到实用 ASR 性能 (WER 4.8/8.2) [Table 1]
2. 端到端 contrastive + Gumbel-Softmax Product Quantization,避免两阶段训练 [§2]
3. 连续输入 + 量化目标的设计被证明为最优方案 [Table 4]
4. Diversity loss 防止 codebook collapse [§3.2]
5. CNN encoder + Transformer 架构成为后续所有 SSL 语音模型 (HuBERT, WavLM, w2v-BERT) 的标准 backbone
