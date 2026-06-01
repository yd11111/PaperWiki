---
type: model
title: "SoundStream"
aliases: []
org: "Google"
year: 2021
tags: [audio-codec, neural-compression, RVQ, streaming-codec]
key_concepts: ["[[Residual Vector Quantization]]", "[[Quantizer Dropout]]"]
tasks: ["[[Neural Audio Compression]]"]
key_papers: ["[[论文笔记/DAC|DAC]]", "[[论文笔记/MaskGCT|MaskGCT]]"]
supersedes: []
superseded_by: [EnCodec]
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

SoundStream 是 Google 提出的首个端到端通用 neural audio codec,开创了 "convolutional encoder-decoder + RVQ" 的范式,后续的 EnCodec 和 DAC 均沿用此框架。

## 核心方法

1. **全卷积 encoder-decoder**: 因果卷积支持 streaming 模式
2. **Residual Vector Quantization (RVQ)**: 首次将 RVQ 用于 audio codec
3. **Quantizer dropout**: 首次提出——训练时随机使用前 n 层 quantizer,实现可变比特率
4. **VQ-GAN formulation**: 对抗训练 + feature matching + spectral reconstruction loss
5. **支持多种音频**: 语音、音乐等

## 参数

- 采样率: 24 kHz
- Striding factor: 320
- Frame rate: 75 Hz
- Codebook: 8 个, 各 10-bit
- 比特率: 6 kbps (scalable)
- 压缩因子: 64x [Table 1 of DAC]

## 历史地位

SoundStream 确立了 neural audio codec 的基本范式,后续所有主流 codec (EnCodec, DAC, Vocos 等) 都是在此框架上的改进。

## 关键论文

- Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec", IEEE/ACM TASLP 2021
