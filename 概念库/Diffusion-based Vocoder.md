---
type: concept
title: "Diffusion-based Vocoder"
aliases: [Diffusion Vocoder, 扩散声码器]
category: "model-family"
tags: [vocoder, diffusion, waveform-generation, audio-synthesis, TTS]
key_papers: ["[[论文笔记/Survey-Audio Diffusion Models|Survey-Audio Diffusion Models]]"]
origin_paper: "Chen et al., WaveGrad: Estimating Gradients for Waveform Generation, 2020"
related_concepts: ["[[Neural Vocoder]]", "[[Diffusion Model]]", "[[Mel Spectrogram]]", "[[Score Matching]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Diffusion-based Vocoder 是使用扩散模型 (diffusion model) 从声学特征 (通常为 mel spectrogram) 生成时域音频波形的声码器。相比 GAN-based vocoder (如 HiFi-GAN) 推理较慢,但在音质上可达到最高水平,是 [[Neural Vocoder]] 家族中的重要分支。

**核心优势**: 基于似然的训练目标,生成质量极高,不存在 GAN 的模式坍塌和训练不稳定问题。

**核心挑战**: 需要多步迭代采样 (典型 50-200 步),推理速度慢。大量后续工作致力于减少采样步数。

## 代表性模型

### 开创性工作

#### WaveGrad (Chen et al., 2020)
Zhang et al. (2023) 综述描述 [§3.3.1]: WaveGrad 是将 score matching 与 diffusion 结合用于波形生成的开创性工作。

- **方法**: 估计数据对数密度梯度 (score function),弥合非自回归与自回归方法间的质量差距
- **变体**: 离散 refinement step index / 连续 noise level 两种条件方式,连续变体更灵活
- **效率**: 仅 6 步 refinement 即可生成高质量样本
- **MOS**: 4.47 (LJSpeech) [Table 3]

#### DiffWave (Kong et al., 2020)
DiffWave 是首个展示 diffusion 在波形生成中高度通用性的模型 [§3.3.1]:

- **生成模型**: DDPM
- **条件**: mel spectrogram (上采样后作为条件输入)
- **能力**: 条件生成 (vocoder) + 无条件生成 + 类别条件生成
- **效果**: 语音质量与强自回归方法可比 [Table 3]

### 高效化方向

Zhang et al. (2023) 综述 [§3.3.2] 总结了三条加速路线:

#### 1. 改进噪声调度 (Noise Schedule)

| 模型 | 方法 | 加速效果 |
|------|------|----------|
| BDDM (Lam et al., 2022) | 额外 schedule prediction network | 7 步, 143x > WaveGrad, 28.6x > DiffWave |
| InferGrad (Chen et al., 2022) | 联合训练 + 推理 schedule 优化 | 3x > WaveGrad,MOS: 3.97 |
| WaveFit (Koizumi et al., 2022) | 迭代 + 非自回归混合 | - |

**BDDM 原理** [§3.3.2]: 与 DDPM 中的去噪网络配合,BDDM 用一个额外网络预测更短的采样 schedule,两个网络联合训练。在 vocoder 任务上仅 7 步即可生成与人类语音不可区分的样本。

**InferGrad 原理** [§3.3.2]: 将推理过程纳入训练,增加额外 loss 以最小化 ground-truth 与推理 schedule 下生成样本的差距。

#### 2. 统计改进 (Noise Prior)

| 模型 | 改进 | 效果 |
|------|------|------|
| PriorGrad (Lee et al., 2021) | 数据自适应先验 (非标准 Gaussian) | 加速推理,同时可用于 vocoder 和 acoustic model |
| DDGM (Nachmani et al., 2021) | Gamma 分布替代 Gaussian 噪声 | 质量优于 WaveGrad; PESQ 3.308 [Table 3] |
| ItôWave (Wu & Shi, 2022) | 基于线性 Ito SDE | MOS > WaveGrad 和 DiffWave (95% 置信) |
| SpecGrad (Koizumi et al., 2022) | 自适应噪声频谱包络 | 高频段质量提升 |

**PriorGrad 核心思想** [§3.3.3]: DDPM 使用 Gaussian 噪声先验,但标准 Gaussian 可能不足以表示所有数据模式 (如语音的 voiced/unvoiced 段)。PriorGrad 从条件数据中计算均值和方差作为先验,使噪声在实例级别接近数据分布,显著加速推理。

## 实验结果对比

Zhang et al. (2023) Table 3 总结了 LJSpeech 数据集上的对比 [§3.3]:

| 模型 | MOS | RTF | PESQ | STOI |
|------|-----|-----|------|------|
| WaveGrad | 4.47 | - | - | - |
| DiffWave | 4.44 | - | - | - |
| DDGM | - | - | 3.308 | 0.969 |
| ItôWave | 4.35 | - | - | - |
| InferGrad | 3.97 | - | 3.578 | 0.976 |
| BDDM | 4.48 | 0.438 | 3.98 | 0.987 |

**BDDM 在质量和效率上达到最佳平衡**: MOS 4.48 最高,且 RTF 0.438 (实时 2.3 倍速)。

## 与其他类型 vocoder 的对比

| 类型 | 代表模型 | 推理速度 | 音质 | 训练稳定性 |
|------|----------|----------|------|-----------|
| GAN-based | HiFi-GAN | 最快 (实时 13x) | 高 | 需精心设计判别器 |
| Flow-based | WaveGlow | 较快 | 中高 | 稳定 (MLE) |
| **Diffusion** | WaveGrad, DiffWave | 慢 (多步迭代) | **最高** | **最稳定** |
| AR | WaveNet, WaveRNN | 极慢 | 高 | 稳定 |

## 关键论文

- WaveGrad (Chen et al., 2020): 首个 score-based diffusion vocoder [§3.3.1]
- DiffWave (Kong et al., 2020): DDPM-based 通用波形生成 [§3.3.1]
- BDDM (Lam et al., 2022): 7 步高质量生成,效率突破 [§3.3.2]
- PriorGrad (Lee et al., 2021): 自适应先验加速 diffusion [§3.3.3]
- InferGrad (Chen et al., 2022): 推理感知训练 [§3.3.2]
- SpecGrad (Koizumi et al., 2022): 频谱自适应噪声 [§3.3.3]
- ItôWave (Wu & Shi, 2022): Ito SDE vocoder [§3.3.3]

## 相关概念

- [[Neural Vocoder]]: diffusion vocoder 所属的更大家族
- [[Diffusion Model]]: 底层生成模型框架
- [[Mel Spectrogram]]: diffusion vocoder 的条件输入
- [[Score Matching]]: WaveGrad 的理论基础
- [[Diffusion-based TTS]]: 声学模型端的 diffusion 应用

## 演进

WaveNet (AR vocoder, 2016) --> WaveGlow (flow vocoder, 2018) --> HiFi-GAN (GAN vocoder 主流, 2020) --> WaveGrad + DiffWave (diffusion vocoder, 2020) --> BDDM / PriorGrad (高效化, 2021-22) --> BigVGAN (大规模 GAN 回归, 2023) --> 当前趋势: GAN vocoder 速度优势明显, diffusion vocoder 更多作为质量上限参考或用于 speech enhancement
