---
type: model
title: "MELLE"
aliases: [Mel Language Model, Continuous Token TTS]
org: "CUHK / Microsoft"
year: 2024
tags: [TTS, zero-shot, autoregressive, continuous-token, mel-spectrogram, codec-free, LLM-TTS]
key_concepts: ["[[Mel Spectrogram]]", "[[Variational Autoencoder for TTS]]", "[[LLM-based TTS]]", "[[Codec Language Model]]", "[[Neural Vocoder]]"]
tasks: [TTS, zero-shot-TTS]
key_papers: ["[[论文笔记/MELLE|MELLE]]", "[[论文笔记/FELLE|FELLE]]"]
supersedes: []
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

MELLE (Meng et al., 2024) 是首个在连续 mel-spectrogram 空间做自回归语言建模的零样本 TTS 系统,绕过了向量量化 (VQ),直接预测连续值 mel frames。通过 spectrogram flux loss 和 latent sampling module 解决连续空间 AR 的两大挑战 (训练目标 + 采样机制),实现单阶段高效推理。Demo: https://aka.ms/melle

## 核心方法

- **Decoder-only Transformer**: 12 层,1024 embed,16 heads,直接在 80-dim mel frames 上做 AR [§3.2.1]
- **Spectrogram Flux Loss**: $-\sum \|\mu_t - y_{t-1}\|_1$,惩罚静态/重复帧,防止 monotonic output [§3.3]
- **Latent Sampling Module (LSM)**: 借鉴 VAE reparameterization,为连续 AR 提供概率采样 [§3.2.2]
- **Non-standard KL prior**: $\mathcal{N}(y_t, I)$ 替代标准 $\mathcal{N}(0, I)$,加速优化 [§3.3]
- **Reduction factor $r$**: 每步预测 $r$ 帧,实现推理加速 (R2=2x, R4=4x) [§3.1]
- **Post-Net**: 5 层 conv 做 mel 残差细化 [§3.2.3]
- **HiFi-GAN vocoder**: 从 refined mel 合成波形 [§4.2]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| WER_H (Continuation) | 1.98 | LibriSpeech test-clean | [Table 1] |
| SIM (Continuation) | 0.508 | LibriSpeech test-clean | [Table 1] |
| WER_H (Cross-sentence) | 2.10 | LibriSpeech test-clean | [Table 1] |
| SIM (Cross-sentence) | 0.625 | LibriSpeech test-clean | [Table 1] |
| MOS | 4.20 (GT 4.29) | LibriSpeech test-clean | [Table 3] |
| SMOS | 4.40 (GT 3.94) | LibriSpeech test-clean | [Table 3] |
| CMOS | -0.032 | LibriSpeech test-clean | [Table 3] |
| Inference (10s) | 5.49s (R4: 1.40s) | - | [Table 5] |

## 演进线

Tacotron (mel prediction, AR) → VALL-E (discrete codec LM) → **MELLE (continuous mel LM, 回归连续表示)** → 未来: 探索其他连续表示 (VAE latent states)

## 关键贡献

1. 证明零样本 TTS 不需要向量量化,连续 mel-spectrogram 足够
2. Spectrogram flux loss 解决连续 AR 的 static frame 问题
3. Latent sampling module 为连续空间提供采样机制
4. 单阶段 AR (无需 AR+NAR 两阶段),架构最简洁
5. SMOS 4.40 超越 GT 3.94,speaker similarity 出色
6. Reduction factor 提供灵活的质量-速度 trade-off
