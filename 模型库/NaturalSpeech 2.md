---
type: model
title: "NaturalSpeech 2"
aliases: [NS2, NaturalSpeech2]
org: "Microsoft Research"
year: 2023
tags: [TTS, zero-shot, diffusion, latent-diffusion, speech-prompting, non-autoregressive, singing-synthesis]
key_concepts: ["[[Diffusion-based TTS]]", "[[Residual Vector Quantization]]", "[[Diffusion Model]]", "[[Duration Predictor]]", "[[Prosody Modeling]]", "[[Non-autoregressive TTS]]"]
tasks: []
key_papers: ["[[论文笔记/NaturalSpeech 2|NaturalSpeech 2]]"]
supersedes: []
superseded_by: ["[[模型库/NaturalSpeech 3|NaturalSpeech 3]]"]
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

NaturalSpeech 2 是微软 NaturalSpeech 系列第二代 TTS 系统,提出用 continuous latent vectors (RVQ 残差求和) + latent diffusion model 替代 discrete tokens + AR language model 的范式,配合 speech prompting 机制实现强零样本 TTS/歌声合成/voice conversion。

## 核心方法

1. **Continuous Latent Vectors**: 用 neural audio codec 的 RVQ 残差向量求和作为连续表示,避免离散 token 展平导致的序列过长问题 [§3.1]
2. **Latent Diffusion Model**: SDE-based diffusion 在连续 latent 空间非自回归生成,WaveNet 作为 score network [§3.2]
3. **Speech Prompting**: 双路设计 — Duration/Pitch Predictor 直接 attend prompt (Q-K-V); Diffusion model 通过 query attention + FiLM 间接 attend (避免信息泄露) [§3.3]
4. **Multi-task**: 支持 TTS + singing synthesis + voice conversion + speech enhancement [§5.6-5.7]

## 关键指标

- CMOS 0.00 (= ground truth) on LibriSpeech/VCTK [Table 3]
- SMOS 3.83 vs VALL-E 3.53 [Table 8]
- 0% error rate on 50 hard sentences [Table 7]
- 435M total parameters, 44K hours training data [Table 11, §4.1]

## NaturalSpeech 系列演进

| 版本 | 年份 | 核心范式 | 数据规模 | 目标 |
|------|------|---------|---------|------|
| NaturalSpeech 1 | 2022 | VAE + flow + duration | LJSpeech (单说话人) | 单说话人人类级质量 |
| **NaturalSpeech 2** | 2023 | Latent diffusion + RVQ codec + speech prompting | 44K hrs MLS | 多说话人零样本多样性 |
| NaturalSpeech 3 | 2024 | Factorized diffusion codec (4 子空间) | LibriLight 60K hrs | 属性分解 + 更强零样本 |
