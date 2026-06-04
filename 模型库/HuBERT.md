---
type: model
title: "HuBERT"
aliases: [Hidden-Unit BERT, HuBERT model]
org: "Meta AI (Facebook AI Research)"
year: 2021
tags: [self-supervised-learning, speech-representation, masked-prediction, BERT, ASR, speech-tokenizer]
key_concepts: ["[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Speech Language Model]]", "[[Self-Supervised Speech Representation]]"]
tasks: []
key_papers: ["[[论文笔记/HuBERT|HuBERT]]", "[[论文笔记/NAST|NAST]]", "[[论文笔记/PROEMO|PROEMO]]", "[[论文笔记/DiVISe|DiVISe (Liu et al., 2025)]]", "[[论文笔记/LM-SPT|LM-SPT]]", "[[论文笔记/C2F-LM|C2F-LM]]", "[[论文笔记/EmoSSLSphere|EmoSSLSphere]]", "[[论文笔记/MSR-Codec|MSR-Codec]]"]
supersedes: ["[[模型库/wav2vec 2.0|wav2vec 2.0]]"]
superseded_by: ["[[模型库/WavLM|WavLM]]"]
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

HuBERT (Hsu et al., IEEE/ACM TASLP 2021) 是自监督语音表征学习的里程碑模型。通过离线 k-means 聚类产生伪标签 + BERT-style masked prediction 预训练,在 ASR 下游任务上匹配或超越 wav2vec 2.0,同时产生的 semantic tokens (k-means 离散化后) 成为后续 SpeechLM 和 TTS 系统的核心表征。

## 核心方法

- **离线聚类伪标签**: 对 MFCC 或前一轮 HuBERT 的中间层特征做 k-means,产生帧级离散标签 [§II-A]
- **Masked Prediction**: 随机 mask 8% 的 CNN encoder 输出,仅在 masked region 计算 cross-entropy loss [§II-B]
- **迭代 Refinement**: 用预训练模型的特征做更好的聚类 → 更好的标签 → 更好的模型 [§II-D]
- **核心洞察**: "consistency of targets > correctness" — 低质量但一致的标签足以学到好表征 [§II]
- **架构**: wav2vec 2.0 结构 (7层 CNN encoder + Transformer), BASE 95M / LARGE 317M / X-LARGE 964M [Table I]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| WER (test-clean/test-other, X-Large, 960h) | 1.9/3.3 | LibriSpeech | [Table III] |
| WER (test-clean/test-other, X-Large, 10min) | 4.6/6.8 | LibriSpeech | [Table II] |
| PNMI (BASE-it1 layer6, K=500) | 0.684 | LibriSpeech | [Table IV] |
| sBLIMP (semantic eval, 25Hz tokens) | 60.89 | SALMon benchmark | [Mousavi 2025] |

## 演进线

CPC (2018) → wav2vec (2019) → wav2vec 2.0 (2020; contrastive + Gumbel-Softmax) → **HuBERT** (2021; masked prediction + offline clustering) → w2v-BERT 2.0 (2022; 结合 contrastive + masked prediction) → Whisper encoder (2022; 弱监督替代自监督)

## 关键贡献

1. 证明低质量 k-means 标签 + masked prediction 可学到 SOTA 语音表征 [Table II, V]
2. 迭代 refinement 机制持续改善聚类和表征质量 [Table IV, Fig 2]
3. 产生的 semantic tokens 催生 GSLM, AudioLM, pGSLM 等语音语言模型
4. 仅 masked loss ($\alpha=1$) 比混合 loss 更鲁棒的发现 [Table V]
