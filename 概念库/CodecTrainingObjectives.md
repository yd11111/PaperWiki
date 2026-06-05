---
type: concept
title: "Codec Training Objectives"
aliases: [Codec 训练目标, Codec Loss Landscape, Audio Codec Training Losses, Neural Codec 损失函数]
category: "training-technique"
tags: [training-objective, audio-codec, GAN, reconstruction, perceptual-loss]
key_papers: ["[[论文笔记/Survey-DiscreteAudioTokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/SoundStream|SoundStream]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/VoxtralTTS|Voxtral TTS]]", "[[论文笔记/DS-Codec|DS-Codec]]", "[[论文笔记/MBCodec|MBCodec]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/Semantic-VAE|Semantic-VAE]]", "[[论文笔记/SiTok|SiTok]]"]
origin_paper: "Mousavi et al., Discrete Audio Tokens: More Than a Survey!, TMLR 2025"
related_concepts: ["[[AudioTokenizerTaxonomy]]", "[[ResidualVectorQuantization]]", "[[Multi-scaleSTFTDiscriminator]]", "[[CodebookCollapse]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Codec Training Objectives 是训练 neural audio codec (encoder-quantizer-decoder) 时使用的损失函数体系。现代 codec 通常组合使用多个目标函数,在信号保真度、感知质量和离散表征质量之间取得平衡 [§2.4.2, Fig 2]。

## 主要训练目标 [§2.4.2]

### 1. Reconstruction Loss (L_Recon)

最基本的目标: 让重建信号逼近原始信号 [§2.4.2]:

$$\mathcal{L}_\text{Recon} = \sum_{t=1}^{T} \|x_t - \hat{x}_t\|^2$$

通常为 MSE 或 MAE。直接优化波形保真度,是几乎所有 codec 的基础损失项。

### 2. Adversarial Loss (L_GAN)

判别器 D 区分真实信号 x 和重建信号 x_hat,generator (tokenizer) 试图骗过判别器 [§2.4.2]:

**Generator loss**:
$$\mathcal{L}_G = \frac{1}{K}\sum_{k=1}^{K}\max(1 - D_k(\hat{x}), 0)$$

**Discriminator loss**:
$$\mathcal{L}_D = \frac{1}{K}\sum_{k=1}^{K}[\max(1 - D_k(x), 0) + \max(1 + D_k(\hat{x}), 0)]$$

使用 hinge loss 形式,K 为判别器数量。常见判别器配置:
- Multi-Period Discriminator (MPD): periods [2, 3, 5, 7, 11]
- Multi-Scale STFT Discriminator: window lengths [2048, 1024, 512, 256, 128]

### 3. Feature Matching Loss (L_Feats)

从判别器中间层提取特征,让重建信号匹配真实信号的高层统计量 [§2.4.2]:

$$\mathcal{L}_\text{Feats} = \frac{1}{KL}\sum_{k=1}^{K}\sum_{l=1}^{L}\frac{\|D_k^l(x) - D_k^l(\hat{x})\|_1}{\text{mean}(\|D_k^l(x)\|_1)}$$

K 为判别器数量, L 为每个判别器的层数。Feature matching 稳定 GAN 训练,鼓励生成器匹配真实信号的高层统计量,改善感知质量。

### 4. VQ Loss (L_VQ)

用于 VQ/RVQ 训练的辅助损失,有两种形式 [§2.4.2]:

**Soft-to-hard scheme** (Agustsson et al., 2017):
$$\mathcal{L}_{VQ} = \|z - \hat{z}\|, \quad \hat{z} = \sum_{m=1}^{M}\alpha_m * c_m$$

**Commitment loss** (VQ-VAE style):
$$\mathcal{L}_{VQ} = \sum_{t=1}^{T}\sum_{m=1}^{M}\left\|z_t^{(m)} - \text{sg}\left[\hat{z}_t^{(m)}\right]\right\|^2$$

其中 sg 为 stop-gradient。现代方法常用 EMA 更新码本替代 commitment loss。详见 [[CodebookCollapse]] 中关于辅助机制的讨论。

### 5. Diffusion Loss (L_diff)

当 decoder 建模为条件去噪扩散过程时使用 [§2.4.2]:

$$\mathcal{L}_\text{diffusion} = \mathbb{E}_{z_0,t,z_q}\left[\|\epsilon_t - \epsilon_\theta(z_t, t, z_q)\|\right]$$

z_q 为离散 token 作为条件。代表: LaDiffCodec (Yang et al., 2024e), SemantiCodec (Liu et al., 2024), S1/S3 (Du et al., 2024)。

### 6. Masked Prediction Loss (L_MP)

encoder 预测被遮蔽部分的语音信息,encoder 和 decoder 分别训练 [§2.4.2]:

$$\mathcal{L}_\text{MP} = \sum_{t=1}^{T}M_t \cdot \ell(\mathcal{Z}_t, x_t)$$

M 为二值 mask,l 为交叉熵。用于 HuBERT、WavLM 等 SSL 模型,以及 Discrete WavLM、NAST 等 tokenizer。

## 损失组合实践 [Table 1]

| 组合 | 代表 tokenizer | 特点 |
|------|---------------|------|
| GAN + Feat + Rec + VQ | EnCodec, DAC, SoundStream | 最经典的全套组合 |
| GAN + Feat + Rec | WavTokenizer, HARP-Net | 无 VQ loss (FSQ 不需要) |
| Rec + VQ | APCodec, BigCodec | 简化版,无 GAN |
| Diff + VQ | LaDiffCodec | 扩散替代 GAN |
| GAN + Rec + Feat + VQ + SD | SpeechTokenizer, PAST | 加入语义蒸馏 |
| MP | Discrete WavLM, Best-RQ | 纯 SSL 方案 |
| Diff + CTC + VQ | SiTok | Flow matching 重建 + CTC 语义正则化 + VQ commitment; 端到端联合训练,CTC 直接预测文本而非蒸馏 SSL 特征 |

## 训练策略 [§2.4.1]

### Separate (Post-Training)
encoder 和 decoder 独立训练,常见于 semantic tokenizer:
- Frozen SSL encoder (HuBERT/WavLM) + offline k-means/VQ
- 独立训练 decoder (HiFi-GAN vocoder / diffusion model)
- 优势: 简单; 劣势: encoder-decoder 不联合优化

### Joint (End-to-End)
encoder/quantizer/decoder 同时训练,常见于 acoustic tokenizer:
- 需要解决量化不可微问题: STE / soft-to-hard / Gumbel-Softmax
- 优势: 全局优化; 劣势: 训练复杂度高
- 典型代表: SoundStream, EnCodec, DAC

## 在 TTS 中的应用

训练目标的选择直接影响 codec 作为 TTS tokenizer 的表现:
- **GAN + Feat**: 提升感知质量,使 codec 输出更自然 (SoundStream, DAC)
- **Semantic distillation**: 第一层 RVQ 蒸馏 SSL 语义,改善 content consistency (SpeechTokenizer)
- **Survey 关键发现**: "optimizing for reconstruction alone does not guarantee better performance on downstream tasks" [§3.2] — 重建最优不等于下游最优

## 关键论文

- Mousavi et al., "Discrete Audio Tokens", TMLR 2025 — 系统总结所有训练目标并给出统一公式 [§2.4]
- Zeghidour et al., "SoundStream", 2021: 建立 GAN+Feat+Rec+VQ 的经典组合
- Kumar et al., "DAC", NeurIPS 2023: 完善损失组合, 引入 multi-band STFT discriminator
- Defossez et al., "EnCodec", 2023: EMA codebook + balancer for multi-loss training

## 相关概念

- [[AudioTokenizerTaxonomy]]: 训练范式是 taxonomy 的 Axis 3
- [[ResidualVectorQuantization]]: VQ loss 的作用对象
- [[Multi-scaleSTFTDiscriminator]]: GAN loss 中的判别器
- [[CodebookCollapse]]: VQ loss / EMA / commitment loss 与 collapse 的关系

## 演进

手工 codec (Opus, 无学习) → VQ-VAE reconstruction-only (2017) → SoundStream GAN+Feat+Rec (2021) → EnCodec 加入 EMA+balancer (2022) → DAC 完善 multi-band discriminator (2023) → Diffusion-based decoder (LaDiffCodec, 2024) → SSL masked prediction 路线 (Discrete WavLM, 2024) → 多目标联合优化 (PAST/TAAE, 2025)

---

> [!info] 来源
> 定义和公式基于 Mousavi et al., "Discrete Audio Tokens: More Than a Survey!", TMLR 2025, Section 2.4。
