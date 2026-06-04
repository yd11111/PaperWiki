---
type: model
title: "VITS"
aliases: [Variational Inference with adversarial learning for end-to-end Text-to-Speech, VITS TTS]
org: "Kakao Enterprise / KAIST"
year: 2021
tags: [TTS, end-to-end, VAE, normalizing-flow, GAN, parallel-synthesis]
key_concepts: ["[[Variational Autoencoder for TTS]]", "[[Non-autoregressive TTS]]", "[[Duration Predictor]]", "[[Neural Vocoder]]", "[[Speech-Text Alignment]]"]
tasks: []
key_papers: ["[[论文笔记/VITS|VITS]]", "[[论文笔记/YourTTS|YourTTS]]", "[[论文笔记/IDEA-TTS|IDEA-TTS]]", "[[论文笔记/Llama-VITS|Llama-VITS]]", "[[论文笔记/TITW|TITW]]", "[[论文笔记/UMETTS|UMETTS]]", "[[论文笔记/Muyan-TTS|Muyan-TTS]]", "[[论文笔记/MathReader|MathReader]]", "[[论文笔记/SafeSpeech|SafeSpeech]]", "[[论文笔记/FaceSpeak|FaceSpeak]]", "[[论文笔记/FMSD-TTS|FMSD-TTS]]", "[[论文笔记/FNH-TTS|FNH-TTS]]", "[[论文笔记/TMD-TTS|TMD-TTS]]", "[[论文笔记/ParaStyleTTS|ParaStyleTTS]]"]
supersedes: []
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

VITS (Kim et al., ICML 2021) 是首个将 conditional VAE + normalizing flow + adversarial training 统一为端到端并行 TTS 系统的工作。单模型直接从 phoneme 序列生成高质量波形,无需外部 vocoder,质量超越所有当时的两阶段系统并接近真实语音 [§1, §4.1]。

## 核心方法

- **Conditional VAE**: posterior encoder (WaveNet blocks, linear spectrogram input) + prior encoder (Transformer + normalizing flow) + HiFi-GAN decoder [§2.1, §2.5]
- **Normalizing Flow 增强 prior**: affine coupling layers 将 factorized Gaussian 变换为复杂分布,消融显示去掉 flow MOS 下降 1.52 [§2.1.3, Table 2]
- **Monotonic Alignment Search**: 复用 Glow-TTS 的 MAS 在 ELBO 框架下自动估计 text-speech 对齐 [§2.2.1]
- **Stochastic Duration Predictor**: flow-based 概率 duration 建模,生成多样化的韵律 [§2.2.2]
- **Adversarial Training**: HiFi-GAN MPD + feature matching loss 提升波形质量 [§2.3]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| MOS | 4.43 (+-0.06) | LJ Speech | [Table 1] |
| MOS (multi-spk) | 4.38 (+-0.06) | VCTK | [Table 3] |
| 合成速度 | 67.12x 实时 | LJ Speech | [Table 4] |
| CMOS vs GT | -0.106 | LJ Speech | [Table 5] |

## 演进线

Glow-TTS (Kim et al., 2020; flow-based NAR) → **VITS** (2021; VAE+Flow+GAN E2E) → VITS 2 (Kim et al., 2023; 多说话人扩展) → NaturalSpeech (Tan et al., 2022; memory-based VAE 扩展) → VALL-E / LLM-based TTS (2023; 范式转换)

## 关键贡献

1. 首次证明端到端单模型 TTS 可超越两阶段系统 [Table 1]
2. Normalizing flow 增强 VAE prior 的有效性 (贡献最大的单一因素) [Table 2]
3. Flow-based stochastic duration predictor 建模韵律多样性 [§2.2.2]
4. Linear spectrogram 作为 posterior 高分辨率输入的设计 [§2.1.3]
