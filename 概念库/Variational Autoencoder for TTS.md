---
type: concept
title: "Variational Autoencoder for TTS"
aliases: [VAE-TTS, TTS中的变分自编码器, VAE for Speech Synthesis, 变分推断TTS]
category: "generative-model"
tags: [TTS, VAE, latent-variable, expressive-TTS, generative-model]
key_papers: ["[[论文笔记/VITS|VITS]]", "[[论文笔记/HierSpeech++|HierSpeech++]]", "[[论文笔记/MELLE|MELLE]]", "[[论文笔记/VQ-VAE|VQ-VAE]]", "[[论文笔记/LatentLM|LatentLM]]", "[[论文笔记/CLEAR|CLEAR]]", "[[论文笔记/VibeVoice|VibeVoice]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Prosody Modeling]]", "[[Attention-based TTS]]", "[[Non-autoregressive TTS]]", "[[Neural Vocoder]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Variational Autoencoder (VAE) 在 TTS 中用于将语音的变化信息(韵律、风格、情感等)编码到结构化隐空间中,通过 Gaussian prior 正则化实现隐变量的可操控性。VAE 解决了 TTS 中的 one-to-many mapping 问题:同一文本可对应多种合理的语音变体。

**核心公式**:
- Encoder: $q_\phi(z|x)$ 将语音参考 $x$ 编码为隐变量 $z$
- Decoder: $p_\theta(x|z, \text{text})$ 从隐变量 + 文本生成语音
- 目标: 最大化 ELBO = $\mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x) \| p(z))$

**在 TTS 中的特殊性**: VAE 的隐变量 $z$ 通常编码 text 无法覆盖的变化信息(prosody, style, emotion, speaker timbre),而 text content 由 text encoder 处理。

## 应用场景

### 1. 表现力建模 (Expressive TTS)

VAE 用于隐式建模韵律/风格变化:

| 模型 | 隐变量粒度 | 应用 |
|------|-----------|------|
| VAE-TTS (Zhang et al., 2019) | utterance-level | 整体风格 |
| VAE-Loop (Kei et al., 2018) | utterance-level | 表情建模 |
| GMVAE-Tacotron (Hsu et al., 2019) | utterance-level | GMM prior, 无监督风格分离 |
| BVAE-TTS (Lee et al., 2021) | utterance-level | beta-VAE, 增强解耦 |
| Hono et al. (2020) | multi-grained | 多层级 (utterance → word) VAE |
| Sun et al. (2020) | phoneme + word | 双粒度 variance |

### 2. 声码器 (VAE Vocoder)

- WaveVAE: mel → waveform, VAE-based 并行生成
- 实际应用较少 (GAN vocoder 更实用)

### 3. 声学模型 (VAE Acoustic Model)

- VAE-TTS: AR decoder + VAE 隐变量
- BVAE-TTS: NAR decoder + VAE 隐变量
- VITS: VAE + normalizing flow, fully E2E

### 4. 风格解耦与控制

VAE 隐空间的结构化特性支持:
- **无监督风格发现**: GMM prior 的不同 component 对应不同风格
- **插值**: 在隐空间中平滑过渡不同风格
- **半监督控制**: 部分维度对应标注属性 (speaking rate, emotion)
- **对抗训练解耦**: gradient reversal 移除 speaker 信息

## VITS: VAE 在 TTS 中的集大成者

VITS (Kim et al., ICML 2021) 将 VAE 与 normalizing flow 结合:
- **Posterior encoder**: 从 linear spectrogram 提取隐变量 $z$
- **Prior encoder**: 从 text 通过 normalizing flow 建模 $p(z|\text{text})$
- **Decoder**: HiFi-GAN 从 $z$ 生成波形
- **训练**: VAE ELBO + GAN adversarial loss
- **推理**: text → prior → $z$ → waveform (单模型, 无需外部 vocoder)

## 与其他生成模型的对比

| 特性 | VAE | Flow | GAN | Diffusion |
|------|-----|------|-----|-----------|
| 隐变量操作 | 强 (结构化) | 强 (可逆) | 弱 | 强 |
| 似然估计 | 下界 | 精确 | 无 | 下界 |
| 训练稳定性 | 高 | 高 | 需技巧 | 高 |
| 生成质量 | 中 (posterior collapse) | 中-高 | 高 | 最高 |
| 可控性 | 强 | 中 | 弱 | 中 |

## VAE 在 TTS 中的挑战

1. **Posterior collapse**: decoder 过强时忽略隐变量,KL 趋零
2. **Over-smoothing**: VAE 重建的 mel 容易模糊
3. **Prior-posterior gap**: 推理时从 prior 采样质量不如 posterior

**解决方案**:
- KL annealing / free bits (防 posterior collapse)
- 结合 flow (VITS, Glow-TTS 增强 prior 表达力)
- Adversarial training (VITS 用 GAN loss 补充)

## 关键论文

- VAE-TTS (Zhang et al., ICASSP 2019): 首次将 VAE 用于 TTS 韵律建模
- GMVAE-Tacotron (Hsu et al., ICLR 2019): GMM prior 实现无监督风格聚类
- BVAE-TTS (Lee et al., ICLR 2021): beta-VAE 用于 NAR TTS
- [[论文笔记/VITS|VITS]] (Kim et al., ICML 2021): VAE + Flow + GAN, fully E2E 最佳; 消融显示去掉 normalizing flow MOS 下降 1.52 (4.50→2.98), 用 mel 替换 linear spectrogram 降 0.19 [Table 2]
- VAE-Loop (Akuzawa et al., 2018): VAE 建模表情

## 相关概念

- [[Prosody Modeling]]: VAE 隐变量通常编码韵律信息
- [[Non-autoregressive TTS]]: VAE 帮助解决 NAR 的 one-to-many 问题
- [[Speech Factorization]]: VAE 支持语音属性的解耦
- Normalizing Flow: 常与 VAE 结合增强 prior (VITS)
- [[Conditional Flow Matching]]: VAE 的现代替代方案

## 演进

Reference Encoder (GST-Tacotron, 2018; 确定性) → VAE (VAE-TTS, 2019; 随机性+正则化) → GMVAE (Hsu, 2019; 结构化 prior) → VAE+Flow (VITS, 2021; 强表达+端到端) → Diffusion/CFM 替代 (2023+; VAE 角色弱化) → **sigma-VAE + Next-Token Diffusion (LatentLM 2024; VAE 角色复兴)**: VAE 从辅助组件变为核心 tokenizer,用连续 latent 替代离散 tokens

## sigma-VAE: 为自回归建模设计的 VAE 变体

LatentLM (Sun et al., 2024) 提出的 sigma-VAE 解决了标准 VAE 在自回归生成场景下的 **variance collapse** 问题:

- **问题**: 标准 VAE 的 sigma 是可学习参数,训练时 sigma 趋向 0 → latent space 退化为确定性映射 → 下游 diffusion head 的输入方差过小,对 exposure bias 不鲁棒 [LatentLM §2.3]
- **解决**: sigma-VAE 将 variance **固定为从 N(0, C_sigma) 采样的标量**,不参与梯度优化 [LatentLM Eq. 5]
  - z = mu + sigma * epsilon, epsilon ~ N(0,1), sigma ~ N(0, C_sigma)
  - 训练目标: minimize ||x_hat - x||^2 + beta * ||mu||^2
- **关键发现**: LatentLM 偏好更大 variance 的 tokenizer (与 image-level diffusion 模型相反) [LatentLM §3.1.3, Fig 6]
- **被 CLEAR (Wu et al., 2025) 继承**: CLEAR 的 enhanced wav-VAE 同样借鉴 sigma-VAE 设计
- **被 VibeVoice (Peng et al., 2025) 直接复用**: VibeVoice 的 acoustic tokenizer 基于 sigma-VAE 构建,3200x 压缩,7.5 Hz
