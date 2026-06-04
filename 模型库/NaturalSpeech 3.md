---
type: model
title: "NaturalSpeech 3"
aliases: [NS3, NaturalSpeech3]
org: "Microsoft Research"
year: 2024
tags: [TTS, zero-shot, diffusion, factorization, codec, non-autoregressive, discrete-diffusion]
key_concepts: ["[[Speech Factorization]]", "[[Diffusion-based TTS]]", "[[Residual Vector Quantization]]", "[[Classifier-Free Guidance]]", "[[Gradient Reversal Layer]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
key_papers: ["[[论文笔记/NaturalSpeech 3|NaturalSpeech 3]]", "[[论文笔记/E2 TTS|E2 TTS]]", "[[论文笔记/DiffCSS|DiffCSS]]", "[[论文笔记/OZSpeech|OZSpeech]]", "[[论文笔记/DiFlow-TTS|DiFlow-TTS]]", "[[论文笔记/MSR-Codec|MSR-Codec]]"]
supersedes: ["[[模型库/NaturalSpeech 2|NaturalSpeech 2]]"]
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

NaturalSpeech 3 是微软 NaturalSpeech 系列第三代 TTS 系统,提出 FACodec (Factorized Audio Codec) 将语音分解为 content/prosody/timbre/acoustic detail 四个独立子空间,配合 factorized discrete diffusion model 逐属性生成,首次在多说话人 LibriSpeech 上达到人类水平自然度。

## 核心方法

1. **FACodec**: 属性分解 codec (3 组 FVQ + timbre extractor),通过信息瓶颈 + GRL + detail dropout 三重解耦
2. **Factorized Diffusion**: 4 个共享结构的 discrete diffusion 模块按 duration→prosody→content→detail 顺序生成
3. **In-context learning**: prompt 无噪声拼接 target 有 mask,自然实现零样本能力
4. **Classifier-Free Guidance**: 提升 speaker similarity 和生成质量

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| CMOS | 0.00 (= human) | LibriSpeech test-clean | Table 1 |
| SMOS | 4.01 (GT: 3.85) | LibriSpeech test-clean | Table 1 |
| Sim-O | 0.67 (GT: 0.68) | LibriSpeech test-clean | Table 1 |
| WER (%) | 1.81 (GT: 1.94) | LibriSpeech test-clean | Table 1 |
| RTF | 0.296 | V100 GPU | Table 10 |

## 演进线

NaturalSpeech 1 (2024, 单说话人, flow-based) → NaturalSpeech 2 (2023, 多说话人, latent diffusion) → NaturalSpeech 3 (2024, factorized codec + diffusion, 人类水平)

## 关键贡献

- 首次在多说话人 LibriSpeech 数据集上实现人类水平的零样本语音合成
- FACodec 的属性分解思想可迁移至其他生成模型 (如 VALL-E + FACodec 组合)
- 验证了数据 (1K→200K hours) 和模型 (500M→1B) 的 scaling 有效性

## 关键论文

- Ju et al., "NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models", ICML 2024 (arXiv:2403.03100)
