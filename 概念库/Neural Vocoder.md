---
type: concept
title: "Neural Vocoder"
aliases: [神经声码器, Neural Waveform Generator, 波形合成器]
category: "model-family"
tags: [TTS, vocoder, waveform-generation, audio-synthesis, GAN, flow, diffusion]
key_papers: ["[[论文笔记/Survey-Audio Diffusion Models|Survey-Audio Diffusion Models]]", "[[论文笔记/VITS|VITS]]", "[[论文笔记/Fish-Speech|Fish-Speech]]", "[[论文笔记/GLM-TTS|GLM-TTS]]", "[[论文笔记/HierSpeech++|HierSpeech++]]", "[[论文笔记/Make-A-Voice|Make-A-Voice]]", "[[论文笔记/MELLE|MELLE]]", "[[论文笔记/FlowDec|FlowDec]]", "[[论文笔记/PeriodWave|PeriodWave]]", "[[论文笔记/Meta Learning TTS 7000 Languages|Meta Learning TTS 7000 Languages]]", "[[论文笔记/Very Attentive Tacotron|Very Attentive Tacotron (Battenberg et al., 2025)]]", "[[论文笔记/DiVISe|DiVISe (Liu et al., 2025)]]", "[[论文笔记/ZipVoice|ZipVoice]]", "[[论文笔记/Shallow Flow Matching|Shallow Flow Matching]]", "[[论文笔记/FNH-TTS|FNH-TTS]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Mel Spectrogram]]", "[[Text-to-Speech Pipeline]]", "[[Multi-scale STFT Discriminator]]", "[[Snake Activation]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Neural Vocoder 是将声学特征(通常为 mel spectrogram)转换为时域音频波形的神经网络模型。它是现代 TTS pipeline 的最后一级,直接决定合成语音的音质和推理速度。

**核心挑战**: 从约 100 帧/秒的 mel spectrogram 上采样到 16k-48kHz 的音频波形(200x-480x 上采样),同时保证高保真度。

## 分类体系

### 按生成模型分类

| 类别 | 代表模型 | 特点 | 推理速度 |
|------|----------|------|----------|
| **Autoregressive** | WaveNet, WaveRNN, SampleRNN | 高质量, 慢 | O(N) |
| **Flow-based** | Parallel WaveNet, WaveGlow, FloWaveNet | 可逆变换, 快 | O(T) |
| **GAN-based** | MelGAN, HiFi-GAN, Parallel WaveGAN | 最快, 高质量 | O(1) |
| **VAE-based** | WaveVAE | 隐变量建模 | O(1) |
| **Diffusion-based** | DiffWave, WaveGrad, PriorGrad | 最高质量, 慢 | O(T) |

### 按输入类型分类
- **Linguistic Feature → Waveform**: WaveNet (原始), WaveRNN, GAN-TTS
- **Mel-Spectrogram → Waveform**: WaveGlow, HiFi-GAN, DiffWave (主流)
- **Cepstrum → Waveform**: LPCNet (BFCC 输入)

## 代表性模型详解

### WaveNet (van den Oord et al., SSW 2016)
- 首个 neural vocoder, 开创性工作
- **架构**: Dilated causal convolution stack (30 layers)
- **生成方式**: 逐样本自回归 (每步预测一个采样点)
- **条件输入**: linguistic features / mel spectrogram
- **缺点**: 推理极慢 (每秒 16000 步)

### WaveRNN (Kalchbrenner et al., ICML 2018)
- **架构**: 单层 RNN + dual softmax (粗/细)
- **优化**: weight pruning, subscale prediction
- **目标**: 移动端部署的轻量 vocoder
- **速度**: 比 WaveNet 快约 4x

### WaveGlow (Prenger et al., ICASSP 2019)
- **生成模型**: Bipartite normalizing flow (Glow architecture)
- **输入**: mel spectrogram
- **优势**: 非自回归, 并行生成, 训练简单 (仅 MLE loss)
- **缺点**: 模型大 (268M 参数), 需多步流变换

### HiFi-GAN (Kong et al., NeurIPS 2020)
- **生成器**: Multi-Receptive Field Fusion (MRF) module
  - 多个不同 kernel size 的 residual blocks 并行处理后融合
- **判别器**: Multi-Period Discriminator (MPD) + Multi-Scale Discriminator (MSD)
  - MPD: 将波形 reshape 为 2D [p, T/p], 不同 period 捕获不同结构
  - MSD: 不同下采样率, 不同频率范围
- **损失**: LS-GAN loss + STFT loss + Feature matching loss
- **优势**: 速度快 (实时 13.4x @ V100), 质量高, 轻量 (14M参数)
- **地位**: 成为 2020-2023 最广泛使用的 vocoder

### DiffWave (Kong et al., ICLR 2021)
- **生成模型**: Denoising Diffusion Probabilistic Model (DDPM)
- **正向过程**: 逐步加噪 waveform → Gaussian noise
- **反向过程**: 从 noise 逐步去噪恢复 waveform
- **条件**: mel spectrogram (upsampled)
- **优势**: 生成质量极高
- **缺点**: 需要多步迭代 (典型 50-200 steps)

### Diffusion Vocoder 后续进展 (Zhang et al. 2023 Survey)

Zhang et al. (2023) 在 audio diffusion survey [§3.3] 中系统总结了 diffusion vocoder 的三条改进路线:

| 模型 | 方法 | 效果 |
|------|------|------|
| BDDM (Lam et al., 2022) | 额外 schedule prediction network | 7 步生成,MOS 4.48,143x > WaveGrad |
| InferGrad (Chen et al., 2022) | 推理过程纳入训练 loss | 3x > WaveGrad |
| PriorGrad (Lee et al., 2021) | 数据自适应先验替代标准 Gaussian | 加速推理,更好拟合 voiced/unvoiced |
| DDGM (Nachmani et al., 2021) | Gamma 噪声替代 Gaussian | PESQ 3.308, STOI 0.969 |
| SpecGrad (Koizumi et al., 2022) | 自适应频谱噪声包络 | 高频质量提升 |
| ItôWave (Wu & Shi, 2022) | 线性 Ito SDE | MOS > WaveGrad + DiffWave |

详见 [[Diffusion-based Vocoder]]。

## 生成模型特性对比

| 特性 | AR | VAE | Flow/AR | Flow/Bipartite | Diffusion | GAN |
|------|----|----|---------|---------------|-----------|-----|
| 简单 | Y | N | N | N | N | N |
| 并行 | N | Y | Y | Y | Y | N→Y* |
| 隐变量操作 | N | Y | Y | Y | Y | Y* |
| 似然估计 | Y | Y | Y | Y | Y | N |

*GAN 某些变体不取随机噪声输入

## 演进时间线

```
2016.09  WaveNet (AR CNN, 首个 neural vocoder)
2016.12  SampleRNN (AR RNN)
2017.11  Parallel WaveNet (flow distillation, 首个并行 vocoder)
2018.02  WaveRNN (轻量 AR)
2018.10  WaveGlow (flow-based)
2018.10  FloWaveNet (flow-based)
2019.04  MelGAN (首个 GAN vocoder)
2019.10  Parallel WaveGAN (GAN + multi-STFT loss)
2019.12  WaveFlow (unified flow view)
2020.08  SC-WaveRNN (高效 AR)
2020.09  DiffWave (diffusion vocoder)
2020.09  WaveGrad (diffusion vocoder)
2020.10  HiFi-GAN (成为标准)
2020.07  VocGAN (multi-scale GAN)
2021.06  PriorGrad (自适应先验 diffusion)
2021.10  DDGM (Gamma 噪声 diffusion)
2022.03  BDDM (7步高效 diffusion)
2022.04  InferGrad (推理感知训练)
2022.03  SpecGrad (频谱自适应噪声)
2022.05  ItôWave (Ito SDE vocoder)
```

## 在 SVS 中的应用

Neural vocoder 在 SVS 级联系统中承担与 TTS 相同的波形合成角色,但有以下特殊性 [Pan et al., 2026]:

**SVS 专用 vocoder**: Huang et al. (2021) 开发了面向歌声的 task-specific neural vocoder,针对歌声的表现力特征 (vibrato, 气息, 长音) 进行优化重建。

**传统 vocoder 遗产**: SVS 领域保留了对 WORLD vocoder (Morise et al., 2016) 的使用 — 一种基于参数化重建的声码器,提供 F0、频谱包络和非周期性参数的可解释控制。

**主流选择**: 现代 SVS 系统仍以 HiFi-GAN 和 BigVGAN 为主流 vocoder [§3.1],与 TTS 领域一致。

**端到端趋势**: VISinger 系列等端到端 SVS 系统已不再需要独立 vocoder [§3.2]。

## 关键论文

- WaveNet (van den Oord et al., 2016): 首个 neural vocoder
- Parallel WaveNet (van den Oord et al., 2018): 首个并行 vocoder (知识蒸馏)
- WaveRNN (Kalchbrenner et al., 2018): 移动端轻量 vocoder
- WaveGlow (Prenger et al., 2019): flow-based 并行 vocoder
- MelGAN (Kumar et al., 2019): 首个 GAN vocoder
- HiFi-GAN (Kong et al., 2020): 成为事实标准的 GAN vocoder
- DiffWave (Kong et al., 2021): diffusion-based 高质量 vocoder
- Zhang et al. (2023): A Survey on Audio Diffusion Models — diffusion vocoder 系统总结 [§3.3]

## 相关概念

- [[Mel Spectrogram]]: vocoder 的主要输入
- [[Multi-scale STFT Discriminator]]: GAN vocoder 的频域判别器
- [[Snake Activation]]: 现代 vocoder (BigVGAN) 使用的周期性激活函数
- [[Text-to-Speech Pipeline]]: vocoder 是 pipeline 最后一级
- [[Diffusion-based Vocoder]]: diffusion vocoder 子家族详述 (WaveGrad/DiffWave/BDDM/PriorGrad)
- [[Diffusion Model]]: diffusion vocoder 的底层生成模型框架
- LPCNet: 结合 DSP (线性预测) 与 RNN 的混合 vocoder

## 演进

STRAIGHT/WORLD (SPSS vocoder) → WaveNet (2016, AR) → Parallel WaveNet (2017, distilled) → WaveGlow (2018, flow) → MelGAN/HiFi-GAN (2019-20, GAN主流) → DiffWave (2020, diffusion) → BigVGAN (2023, large-scale GAN) → Vocos (2023, iSTFT-based) → PeriodWave (2024, flow matching vocoder)
