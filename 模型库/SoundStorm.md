---
type: model
title: "SoundStorm"
aliases: []
org: "Google Research"
year: 2023
tags: [audio-generation, non-autoregressive, masked-generative, parallel-decoding, acoustic-model]
key_concepts: ["[[MaskedGenerativeModeling]]", "[[ResidualVectorQuantization]]", "[[SemanticvsAcousticTokens]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
key_papers: ["[[论文笔记/SoundStorm|SoundStorm]]"]
supersedes: []
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

SoundStorm 是 Google Research 提出的高效非自回归音频生成模型,作为 AudioLM pipeline 中 acoustic generation stage 的替代方案。利用 bidirectional Conformer + MaskGIT-style iterative parallel decoding,按 RVQ 层 coarse-to-fine 逐级生成 SoundStream tokens,比 AudioLM 快两个数量级。

## 核心方法

1. **Frame-level embedding sum**: 将同一帧的多层 RVQ token embeddings 求和,序列长度 = 帧数 T,与 RVQ 层数 Q 无关
2. **Bidirectional Conformer**: 350M 参数,12 层,双向 self-attention (非 causal)
3. **RVQ level-wise iterative decoding**: 第 1 层 16 iterations (confidence-based) + 第 2-12 层 greedy
4. **训练 masking scheme**: 统一支持任意 RVQ level + voice prompting + unprompted generation

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| RTF | 0.017 | TPU-v4 | Fig 3 |
| WER (%) prompted | 2.99 (short), 2.55 (mid), 3.36 (long) | LibriSpeech test-clean | Table 1 |
| Acoustic consistency | 0.91-0.96 | LibriSpeech test-clean | Table 1 |
| Audio quality (MOS-like) | 4.01-4.20 | LibriSpeech test-clean | Table 1 |

## 历史地位

SoundStorm 建立了"非自回归生成 RVQ tokens"的技术范式,后续 MaskGCT 将 masked generative modeling 扩展到完整 TTS pipeline (T2S + S2A),NaturalSpeech 3 的 factorized discrete diffusion 也借鉴了 mask-and-predict 的公式。[[论文笔记/StellarTTS|StellarTTS]] (2026) 继承该范式并做移动端优化,复用 RVQ + masked 逐层生成 (推理步数 [8,4,1,1,1,1]),但重新引入 phone-level 时序锚点 (sparse temporal embedding) 以修复 alignment-free 路线的鲁棒性问题。

## 关键论文

- Borsos et al., "SoundStorm: Efficient Parallel Audio Generation", arXiv:2305.09636, 2023
