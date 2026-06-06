---
type: concept
title: "Codec Training Objectives"
aliases: [Codec 训练目标, Codec Loss Landscape, Audio Codec Training Losses, Neural Codec 损失函数]
category: "training-technique"
tags: [training-objective, audio-codec, GAN, reconstruction, perceptual-loss]
key_papers: ["[[论文笔记/Survey-DiscreteAudioTokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/SoundStream|SoundStream]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/VoxtralTTS|Voxtral TTS]]", "[[论文笔记/DS-Codec|DS-Codec]]", "[[论文笔记/MBCodec|MBCodec]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/Semantic-VAE|Semantic-VAE]]", "[[论文笔记/SiTok|SiTok]]", "[[论文笔记/AffectCodec|AffectCodec]]", "[[论文笔记/HoliTok|HoliTok]]", "[[论文笔记/MagiCodec|MagiCodec]]"]
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

手工 codec (Opus, 无学习) → VQ-VAE reconstruction-only (2017) → SoundStream GAN+Feat+Rec (2021) → EnCodec 加入 EMA+balancer (2022) → DAC 完善 multi-band discriminator (2023) → Diffusion-based decoder (LaDiffCodec, 2024) → SSL masked prediction 路线 (Discrete WavLM, 2024) → 多目标联合优化 (PAST/TAAE, 2025) → 渐进式多阶段训练 (HoliTok, 2026)

### 9. Progressive Multi-Stage Training (HoliTok)

[[论文笔记/HoliTok|HoliTok]] (Li et al., 2026) 提出渐进式三阶段训练策略,解决"强 KL 正则化在 decoder 尚未学好重建流形时会迫使表示丢弃声学细节"的问题。Stage I 仅训练 AE 重建 (Rec+GAN+Feat, 500K steps); Stage II 冻结 encoder/decoder,仅训练 variational bottleneck (弱 KL β=0.1, 50K steps); Stage III 解冻全部参数,联合优化 Rec+GAN+Feat + 强 KL (β=7) + WavLM 帧级蒸馏 + x-vector 话语级蒸馏 + 多任务 LM 监督 (ASR/emotion/captioning/SED) (200K steps) [HoliTok Eq. 1-6]。关键发现: 消融显示多任务 LM 监督不仅帮助理解,去掉后 TTS WER 从 27.85%→110%,说明下游监督对生成鲁棒性也至关重要 [HoliTok Table 8]。

### 7. Relation-Preserving Distillation Loss (L_rela)

[[论文笔记/AffectCodec|AffectCodec]] (Shi et al., 2026) 引入关系保持蒸馏损失,约束 RVQ 第一层量化输出 Q(1) 的帧间 pairwise 距离与 teacher 空间 (emotion + semantic) 一致: L_rela = (1/T'^2) * sum(alpha * d(r^uni, r^emo) + beta * d(r^uni, r^sem)),其中 d 为 L1 discrepancy,r 为帧对欧氏距离。保护离散化过程中的情感-语义拓扑结构,优于直接 feature matching (recall 0.48 vs 0.42) [AffectCodec Table 7]。

### 8. Emotion-Weighted Semantic Alignment Loss (L_align)

同为 AffectCodec 引入,在 Q(1) 与文本语义 teacher 之间做 soft alignment,但用帧级情感差分 d_t = ||e_t - e_{t-1}||_1 经 softmax 生成权重 gamma_t,使情感变化大的帧获得更强语义对齐监督: L_align = -(1/T') * sum(gamma_t * log(sigma(cos(Q^(1)_t, c*_t))))。核心 idea: 情感变化大的帧更易受量化失真影响,需要更强的锚定 [AffectCodec §3.2.3]。

### 10. Gaussian Noise Injection + Staged Training (MagiCodec)

[[论文笔记/MagiCodec|MagiCodec]] (Song et al., 2025) 提出三阶段训练 + Gaussian noise injection 的组合策略。Stage 1 训练 AE (encoder+decoder,无 VQ),输入帧以 Bernoulli(p) 概率被 Gaussian noise 替换,隐式正则化高频成分 + latent regularization (L_norm = ||Z_e||_2^2); Stage 2 冻结 encoder,仅训练 VQ+decoder; Stage 3 冻结 encoder+VQ,GAN 训练 vocoder (MPD + MS-STFT Discriminator)。与 HoliTok 的渐进式三阶段不同,MagiCodec 完全不使用外部监督 (无 SSL 蒸馏/无 ASR/无多任务),仅靠内在正则化提升 token 的下游可建模性。消融显示 mask ratio 30% 时 TTS WER 从 5.51% 降至 3.30% [MagiCodec Table 7]。

### 11. Self-guidance Loss (OmniCodec)

[[论文笔记/OmniCodec|OmniCodec]] (Hu et al., 2026) 引入 self-guidance loss,用 pre-quantized continuous latent 的 decoder 输出作为 teacher,引导 quantized token 的 decoder 输出逼近: L_self_guidance = |sg(h_e) - h_q|^2,其中 h_e/h_q 分别是 acoustic transformer 处理 z_e (连续) 和 z_q (量化) 后的隐层特征,sg 为 stop-gradient (Li et al., 2024 提出)。核心 idea: 迫使 decoder 学会容忍量化误差,从而改善重建质量和 codebook 利用率 (0.974→0.982)。权重设为 0.1,属于轻度正则化。与 MagiCodec 的 staged training 不同,self-guidance 是单阶段端到端训练中的辅助 loss,无需改变训练流程 [OmniCodec §2.3, Table 2]。

---

> [!info] 来源
> 定义和公式基于 Mousavi et al., "Discrete Audio Tokens: More Than a Survey!", TMLR 2025, Section 2.4。
