---
type: concept
title: "Multi-scale STFT Discriminator"
aliases: [Multi-band STFT Discriminator, STFT-D, Complex STFT Discriminator]
category: "architecture-component"
tags: [discriminator, GAN, frequency-domain, audio-codec, vocoder]
key_papers: ["[[论文笔记/DAC|DAC]]"]
origin_paper: ""
related_concepts: ["[[Snake Activation]]", "[[Residual Vector Quantization]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Multi-scale STFT Discriminator 是一种频域判别器,在多个 STFT 时间尺度 (不同 window length) 上对音频进行真/假判别。相比传统时域判别器 (MSD, MPD),它能更好地建模高频细节和相位信息。

## 变体

| 变体 | 提出者 | 特点 |
|------|--------|------|
| Multi-Resolution Spectrogram Discriminator (MRSD) | UnivNet [16] | magnitude spectrogram, 多分辨率 |
| Complex STFT Discriminator | SoundStream [46] | 保留相位信息 (complex-valued) |
| Multi-band Multi-scale STFT Discriminator | DAC (2023) | 按频带分割 + 多尺度 + complex |

## DAC 的设计 [§3.4]

- **多尺度**: window lengths [2048, 1024, 512]
- **多频带 (sub-band splitting)**: 将 STFT 按 band-limits [0.0, 0.1, 0.25, 0.5, 0.75, 1.0] 切分
- **Complex-valued**: 保留相位信息,改善相位重建

**为什么 sub-band splitting 有效**:
1. Decoder 的上采样层引入 aliasing artifacts (高频伪影)
2. 全频带 discriminator 中,这些局部高频伪影被全局特征"淹没"
3. 频带级 discriminator 能专注于检测每个频率范围的伪影
4. 更精确的梯度信号 → generator 更好地修复高频问题

## 与时域判别器的配合

DAC 同时使用:
- Multi-Period Discriminator (MPD): periods [2, 3, 5, 7, 11], 捕捉不同周期结构
- Multi-band Multi-scale Complex STFT Discriminator: 频域高频+相位

两者互补: MPD 擅长时域周期结构, STFT-D 擅长频域细节和相位。

## 关键论文

- Jang et al., "UnivNet", 2021: 提出 MRSD
- Lee et al., "BigVGAN", 2023: 使用 MRSD 替换 MSD
- DAC (Kumar et al., NeurIPS 2023): 引入 multi-band splitting + complex STFT, 缓解 aliasing artifacts

## 相关概念

- Multi-Period Discriminator (MPD): 时域互补判别器
- Multi-Scale Discriminator (MSD): 早期时域多尺度判别器
- [[Snake Activation]]: 与 STFT-D 配合减少周期性伪影
- Feature Matching Loss: 从 discriminator 中间层提取的辅助 loss

## 演进

MSD (MelGAN, 2019) → MPD (HiFi-GAN, 2020) → MRSD (UnivNet, 2021) → BigVGAN (MRSD 替换 MSD) → DAC (multi-band complex STFT-D, 2023)
