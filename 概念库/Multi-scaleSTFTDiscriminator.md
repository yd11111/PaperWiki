---
type: concept
title: "Multi-scale STFT Discriminator"
aliases: [Multi-band STFT Discriminator, STFT-D, Complex STFT Discriminator]
category: "architecture-component"
tags: [discriminator, GAN, frequency-domain, audio-codec, vocoder]
key_papers: ["[[论文笔记/DAC|DAC]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Survey-DiscreteAudioTokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/SNAC|SNAC]]", "[[论文笔记/FlexiCodec|FlexiCodec]]", "[[论文笔记/Pupu-Vocoder|Pupu-Vocoder (Gu et al., 2025)]]"]
origin_paper: ""
related_concepts: ["[[SnakeActivation]]", "[[ResidualVectorQuantization]]", "[[CodecTrainingObjectives]]"]
status: confirmed
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
| Multi-Scale Sub-Band CQT Discriminator (MS-SB-CQTD) | Gu et al. (2024) [50] | Constant-Q 变换 + 多尺度 + 子频带; Pupu-Vocoder 使用 |

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
- [[SnakeActivation]]: 与 STFT-D 配合减少周期性伪影
- Feature Matching Loss: 从 discriminator 中间层提取的辅助 loss

## 历史背景

据 Xu Tan et al. (2021) Survey 整理,GAN vocoder 判别器的设计核心目标是"如何更好地捕捉波形特征以提供 generator 更精确的梯度信号"。早期工作主要在时域探索:
1. Random window discriminators (GAN-TTS, 2019): 不同随机窗口互补判别
2. Multi-scale discriminators (MelGAN, 2019): 不同下采样率捕捉不同频率特征
3. Multi-period discriminators (HiFi-GAN, 2020): reshape 为 2D 捕捉周期结构
4. Hierarchical discriminators (VocGAN, 2020): 不同分辨率的层次判别

STFT-based 频域判别器的引入代表了从纯时域到时频联合判别的范式跃迁。

## Survey 中的 GAN 训练与 Discriminator [Mousavi et al. 2025, §2.4.2]

Survey 给出了 adversarial loss 和 feature matching loss 的统一公式:

**Adversarial loss** (hinge loss): 在 K 个 discriminator 上取平均, generator 最大化 D(x_hat), discriminator 区分真/假 [§2.4.2]

**Feature matching loss**: 从 K 个 discriminator 的 L 层中间激活中提取特征, 让生成信号匹配真实信号的高层统计量: L_Feats = (1/KL) sum ||D_k^l(x) - D_k^l(x_hat)||_1 / mean(||D_k^l(x)||_1) [§2.4.2]

Survey Table 1 显示 GAN + Feature Matching 是最普遍的训练目标组合,覆盖大多数 acoustic tokenizer (SoundStream, EnCodec, DAC, SpeechTokenizer, WavTokenizer 等)。

详见 [[CodecTrainingObjectives]] 中对完整训练目标体系的描述。

## 演进

MSD (MelGAN, 2019) → MPD (HiFi-GAN, 2020) → MRSD (UnivNet, 2021) → BigVGAN (MRSD 替换 MSD) → DAC (multi-band complex STFT-D, 2023) → MS-SB-CQTD (Gu et al., 2024, Constant-Q) → Pupu-Vocoder/Codec (2025, 4 discriminator 组合: MPD+MSD+MBD+MS-SB-CQTD)
