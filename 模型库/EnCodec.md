---
type: model
title: "EnCodec"
aliases: [Encodec, Meta EnCodec]
org: "Meta AI (FAIR)"
year: 2022
tags: [audio-codec, neural-compression, RVQ, speech-tokenizer]
key_concepts: ["[[Residual Vector Quantization]]", "[[Codebook Collapse]]"]
tasks: ["[[Neural Audio Compression]]"]
key_papers: ["[[论文笔记/VALL-E|VALL-E]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/NaturalSpeech 3|NaturalSpeech 3]]", "[[论文笔记/SNAC|SNAC]]", "[[论文笔记/RepCodec|RepCodec]]", "[[论文笔记/TacoLM|TacoLM]]", "[[论文笔记/SESD|SESD]]", "[[论文笔记/TTS-Transducer|TTS-Transducer]]", "[[论文笔记/MAE Style-Rich TTS|MAE Style-Rich TTS]]", "[[论文笔记/DS-Codec|DS-Codec]]", "[[论文笔记/LM-SPT|LM-SPT]]", "[[论文笔记/C2F-LM|C2F-LM]]", "[[论文笔记/FuseCodec|FuseCodec]]"]
supersedes: ["[[模型库/SoundStream|SoundStream]]"]
superseded_by: []
status: confirmed
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

EnCodec 是 Meta AI (FAIR) 提出的端到端 neural audio codec,沿用 SoundStream 的 encoder-decoder + RVQ 架构,支持 24 kHz 和 48 kHz 两种采样率,带宽范围 1.5-24 kbps。

## 核心方法

1. **架构**: 全卷积 encoder-decoder + Residual Vector Quantization
2. **EMA codebook**: 使用 Exponential Moving Average 更新码本, 配合 k-means 初始化和 random restarts
3. **Multi-scale STFT Discriminator**: 频域判别器
4. **Loss balancer**: 基于 discriminator 梯度动态调整各 loss 权重
5. **Quantizer dropout**: 支持可变比特率

## DAC 对其的超越

DAC (Kumar et al., NeurIPS 2023) 系统性地改进了 EnCodec 的 recipe:
- 更高压缩率: 91x vs 16-32x [Table 1]
- 更宽带宽: 22.05 kHz vs 12 kHz
- 在同等配置下所有指标全面优于 EnCodec [Table 4]
- 解决了 EnCodec 仍存在的 codebook collapse 问题 [Fig 1]

## 在 TTS 中的应用

EnCodec 的 discrete codes 被广泛用作 speech tokenizer:
- VALL-E (Microsoft, 2023): 使用 EnCodec tokens 作为 acoustic target
- MusicLM (Google, 2023): 音乐生成
- AudioLM (Google, 2022): 通用音频生成

## 关键论文

- Defossez et al., "High Fidelity Neural Audio Compression", ICLR 2023
