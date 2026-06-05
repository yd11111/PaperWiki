---
type: paper
tier: deep
title: "Mega-TTS 2: Boosting Prompting Mechanisms for Zero-Shot Speech Synthesis"
arxiv_id: "2307.07218"
source: "Sources/Mega-TTS2.pdf"
authors: [Ziyue Jiang, Jinglin Liu, Yi Ren, Jinzheng He, Zhenhui Ye, Shengpeng Ji, Qian Yang, Chen Zhang, Pengfei Wei, Chunfeng Wang, Xiang Yin, Zejun Ma, Zhou Zhao]
year: 2024
venue: "ICLR 2024 (Zhejiang University & ByteDance)"
tags: [TTS, zero-shot, speech-factorization, prosody-LLM, multi-sentence-prompting, prosody-transfer, timbre-disentanglement, VQ, autoregressive-prosody]
concepts: ["[[SpeechFactorization]]", "[[ProsodyModeling]]", "[[SpeakerEmbedding]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[CodebookCollapse]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[SpeechFactorization]], [[ProsodyModeling]], [[SpeakerEmbedding]], [[LLM-basedTTS]], [[SpeechTokenizer]], [[CodebookCollapse]])
> Mega-TTS 2 是 [[SpeechFactorization]] 中信息瓶颈方案的进一步发展,在 Mega-TTS 1 的 content/timbre/prosody 分解基础上增强了 prompting 机制。[[ProsodyModeling]] 页记录了从 reference encoder 到 in-context learning 的演进,Mega-TTS 2 的 P-LLM 代表了一种独特路线: 不在全语音序列上做 AR,而只在离散化的 prosody codes 上做 AR。[[LLM-basedTTS]] 页中 VALL-E 等方法使用单句 prompt,Mega-TTS 2 的核心贡献之一是解决多句 prompt 利用问题。[[CodebookCollapse]] 在 VQ Encoder 训练中是潜在问题,Mega-TTS 2 采用 CVQ-VAE 动态初始化策略缓解 [Appendix A.3]。
> 检索命中: [[SpeechFactorization]], [[ProsodyModeling]], [[SpeakerEmbedding]], [[LLM-basedTTS]], [[SpeechTokenizer]], [[CodebookCollapse]] | 过滤: [[VoiceCloningTaxonomy]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 通过 prosody/timbre 解耦 + 多句 prompting + prosody LLM,在零样本 TTS 中以 10s-300s 多句 prompt 超越微调基线 / Disentangles prosody from timbre via compressive autoencoder, then uses multi-sentence prompting with P-LLM to surpass fine-tuning baselines at 10s-300s of reference data
> - **路线**: Speech y → Compressive Acoustic AE (Content Enc E_c + VQ Prosody Enc E_p + Multi-Ref Timbre Enc E_t) → Decoder D → y_hat; Inference: multi-sentence refs → E_t for timbre z_t, E_p for prosody codes → P-LLM autoregressively predicts prosody → Decoder generates mel [§3, Fig 1]
> - **指标**: WER 2.23% / SIM 0.932 / SMOS 4.01 with 300s prompt [Table 1]; 超越 fine-tuning baseline (300s) in SMOS (4.01 vs 4.03) and SIM (0.932 vs 0.934) [Table 1]; prosody transfer: DE 32.8, sigma 72.62, gamma 0.664, kappa 0.197 vs CopyCat/Daft-Exprt [Table 2]
> - **可借鉴**: (1) 信息论分解框架 I(y_t; y_bar) = H(z_t) + H(g) — 理论驱动的 timbre/prosody 解耦 [§3.1, Eq 1]; (2) P-LLM 只建模 prosody codes(非全语音 token),codebook 小因此长 prompt 可行; (3) Prosody interpolation — 两个 P-LLM 输出概率加权混合实现风格迁移 [§3.4, Eq 3]
> - **局限**: 仅支持英语 [Appendix E.1]; 依赖外部 aligner (MFA) 获取 duration/pitch [§3.1]; VQ codebook 的信息瓶颈设计需仔细调参(r, d 参数搜索空间大) [Appendix F]; 未开源

## 1. 核心问题与动机 / Core Problem & Motivation

**Why this paper?** Mega-TTS 2 解决现有零样本 TTS prompting 机制的两个关键缺陷 [§1]:

1. **缺乏多句 prompting 策略**: [论文原文] "Previous works of zero-shot TTS typically employ single-sentence speech prompts during training... the information in the single-sentence speech prompt is insufficient to guide the zero-shot TTS systems to imitate the voice variability of a natural person perfectly" [§1]

2. **缺乏韵律信息的专门 prompting**: [论文原文] "Current solutions for zero-shot TTS primarily concentrate on improving the similarity of timbre and prosody between the generated speech and the prompts. However, they neglect to express various unseen prosodic styles in a controlled manner" [§1]

[agent解读] 这两个问题的根因相同: timbre 和 prosody 在 prompt 中是耦合的。当 prompt 只有一句话时,timbre 信息不够丰富;当试图控制 prosody 时,又会影响 timbre。解耦是解决之道。

## 2. 方法详解 / Method

### 2.1 Decomposition for Prosody and Timbre [§3.1]

[论文原文] 基于信息论的分解框架 [§3.1, Eq 1]:

```
I(y_t; y_bar) = H(z_t) + H(g)
```

其中 y_t 是目标 mel-spectrogram,y_bar 是同说话人其他语音,z_t 是 timbre hidden states,g 是 global style information(包含 timbre 和 prosody)。

**关键假设**: [论文原文] "the mutual information between y_t and y_bar only contains timbre information H(z_t) and global style information H(g) of y_t" [§3.1]

[agent解读] 这个假设基于一个直觉: 同一说话人的不同语音之间共享的信息只有 timbre 和 global prosody style(不包含具体文本内容和 local prosody)。通过 corpus partition(将训练数据按说话人分组),可以通过 multi-reference attention 从 y_bar 中提取 timbre。

**分解路径** [§3.1]:
1. E_t(y_bar) → z_t (timbre) + g (global style)
2. E_c(phoneme) → z_c (content) — 仅接收 phoneme,无法获取 timbre
3. E_p(y_t) → z_p (fine-grained prosody) — 信息瓶颈 B(·) 迫使它只传递其他编码器无法提供的信息

### 2.2 Compressive Acoustic Autoencoder [§3.2, Fig 1]

[论文原文] 第一阶段训练三个编码器 + GAN-based decoder [§3.2]:

| 编码器 | 输入 | 输出 | 参数 |
|------|------|------|------|
| VQ Prosody Encoder E_p | 目标 mel y_t | prosody codes u + hidden z_p | 3x2 Conv layers, VQ bottleneck [Table 5] |
| Content Encoder E_c | phoneme | content hidden z_c | 8-layer Transformer, 512 hidden [Table 5] |
| Multi-Ref Timbre Encoder E_t | reference mels y_bar | timbre hidden z_t | 5x2 Conv layers, downsample 16x [Table 5] |

**VQ Prosody Encoder (E_p)** [§3.2, Appendix A.3]:
- 两层卷积栈: 第一层压缩时间轴 r 倍,第二层捕捉特征相关性
- VQ 层: 1024 embedding, 256 dim [Table 5]
- 信息瓶颈 B(·) = 时间压缩 (r=8) + VQ 量化 [Appendix F]
- 解决 codebook collapse: CVQ-VAE 动态初始化 — 将 less-used/unused codes 修改得比 frequently-used codes 更多 [Appendix A.3]

**Multi-Reference Timbre Encoder (MRTE)** [§3.2, Fig 5a]:
- 5x2 Conv stacks + downsampling block (d=16)
- Timbre-to-content attention: z_c 为 Q, z_t 为 K/V → 提取语义相关的 timbre 信息
- 可处理 **up to 300 seconds** of reference mel-spectrograms [§3.2]

**训练损失**: L = L_rec + L_VQ + L_Adv [§3.2]

### 2.3 Prosody Latent Language Model (P-LLM) [§3.3]

[论文原文] P-LLM 是 decoder-only architecture,12 层 Transformer,1024 hidden,151M params [Appendix A.4]。

**核心设计** [§3.3]:
1. 从多句参考中提取 prosody codes: z_p' = Concat(z_{p1}, z_{p2}, ..., z_{pn})
2. 对应 content: z_c' = Concat(z_{c1}, z_{c2}, ..., z_{cn})
3. P-LLM 自回归预测: p(u' | z_c'; θ) = prod_l p(u'_l | u'_{<l}, z_c'; θ) [§3.3, Eq 2]

**训练策略** [§3.3]:
- Batch size = 1, 最大化 prompt 数量 m
- Speaker-level attention mask: 同 batch 含同说话人多句+其他说话人
- Start/end token 每句: 避免拼接过渡问题
- Teacher forcing with cross-entropy loss

**Duration Model**: 同架构的 phoneme-level AR model,用 MSE loss [§3.3]

[agent解读] P-LLM 与 VALL-E 等 full-sequence LM 的关键区别:
- VALL-E: AR 生成完整语音 token (高码率 → 长序列 → 单句 prompt 限制)
- P-LLM: **只 AR 生成 prosody codes** (低码率因为解耦 → 短序列 → 可容纳 300s prompt)

这就是 decomposition 带来的核心优势: prosody 信息量远小于全语音信息量,使得长 prompt in-context learning 变得可行。

### 2.4 Prosody Interpolation [§3.4, Fig 2]

[论文原文] 用两个 P-LLM 的输出概率加权混合 [§3.4, Eq 3]:

```
p(u_hat) = prod_t [(1-gamma) * p(u_hat_t | ..., u_b, ...) + gamma * p(u_hat_t | ..., u_a, ...)]
```

其中 u_b 来自 target speaker 的 sad prosody prompt,u_a 来自 auxiliary speaker 的 happy prosody prompt,gamma 控制插值权重。

[agent解读] 这是一个优雅的跨说话人韵律迁移方案: 不需要修改模型,只需在解码时混合两个 P-LLM 的概率分布。由于 prosody 已与 timbre 解耦,可以自由组合"A 的 timbre + B 的 prosody style"。

## 3. 实验结果与分析 / Experiments

### 3.1 训练配置 [§4.1]

| 配置 | 值 |
|------|-----|
| 训练数据 | LibriLight, 60K hours [§4.1] |
| ASR | DNN-HMM pre-trained on 960h LibriSpeech [§4.1] |
| Aligner | MFA [§4.1] |
| Stage 1 (AE) | 4x A100, batch=48/GPU, 600K steps [§4.1] |
| Stage 2 (P-LLM+Dur) | 8x A100, batch=4000 tokens/GPU, 300K steps [§4.1] |
| Vocoder | Pre-trained HiFi-GAN V1 [§4.1] |
| 总参数量 | 367M (excl. vocoder) [Table 5] |

### 3.2 Zero-Shot Speech Synthesis [§4.2, Table 1]

| Model | WER↓ | SIM↑ | DTW↓ | QMOS↑ | SMOS↑ | RTF | Method |
|-------|------|------|------|-------|-------|-----|--------|
| GT | 1.98% | - | - | 4.43 | 4.26 | - | - |
| Baseline-300s | 3.11% | 0.934 | 29.80 | 4.08 | 4.03 | 0.089 | Fine-tune |
| VALL-E-3s | 5.83% | 0.885 | 36.59 | 3.89 | 3.70 | 1.471 | Zero-shot |
| VALL-E-20s | 8.77% | 0.805 | 43.02 | 3.41 | 3.25 | 2.104 | Zero-shot |
| **Ours-10s** | **2.28%** | 0.905 | 32.30 | 3.99 | 3.75 | 0.302 | Zero-shot |
| **Ours-300s** | **2.23%** | **0.932** | **29.95** | **4.12** | **4.01** | 0.923 | Zero-shot |

[论文原文] 关键发现 [§4.2]:
1. Mega-TTS 2 **随 prompt 增长持续改善**: 3s→10s→60s→300s,WER/SIM/SMOS 全面提升
2. VALL-E **在 20s prompt 时性能骤降**: "the performance significantly drops in the 20-second setting due to the single-sentence prompting mechanisms in training" [§4.2]
3. **300s 零样本 ≈ fine-tuning**: Ours-300s SMOS 4.01 vs Baseline-300s 4.03,WER 2.23% vs 3.11%

[agent解读] 这是论文最核心的结果: 零样本 TTS 终于追上了微调方法。关键原因是多句 prompting + prosody/timbre 解耦: (1) 解耦使 timbre 信息积累不受 prosody 干扰; (2) P-LLM 的 AR 只在 prosody codes 上运行,序列短,可容纳大量 context。

### 3.3 Prosody Transfer [§4.3, Table 2]

| Model | WER | SIM-AB | DE | sigma | gamma | kappa | QMOS | SMOS-AB |
|-------|-----|--------|-----|-------|-------|-------|------|---------|
| CopyCat | 5.29% | 0.843/0.740 | 37.2 | 59.74 | 0.889 | 0.859 | 3.72 | 3.53/3.19 |
| Daft-Exprt | 4.89% | 0.901/0.633 | 36.5 | 67.20 | 0.851 | 0.427 | 3.90 | 3.81/2.90 |
| **Ours** | **4.82%** | **0.920/0.513** | **32.8** | **72.62** | **0.664** | **0.197** | 3.92 | 3.87/2.64 |

[论文原文] "the moments (sigma, gamma, and kappa) of the generated speeches of Mega-TTS are closer to the ground-truth audio and the DE is lower than other methods" [§4.3]

[agent解读] sigma/gamma/kappa 越接近 GT 越好(用 pitch distribution 的矩衡量),Mega-TTS 2 全面领先。SMOS-AB 中 A→B 迁移的 B 方 SMOS 较低 (2.64 vs 2.90 vs 3.19),说明跨说话人韵律迁移时音质略有损失,但韵律还原(sigma/gamma/kappa)远优于对手。

### 3.4 Ablation Studies [§4.4]

**MRTE ablation** [Table 4]:
| Setting | WER↓ | SIM↑ | CMOS-Q | CMOS-S |
|---------|------|------|--------|--------|
| Ours-10s | 2.28% | 0.905 | 0.000 | 0.000 |
| w/o MRTE | 5.57% | 0.841 | -0.458 | -0.619 [Table 4] |

[论文原文] "the removal of MRTE significantly affects both the audio quality and speaker similarity... because the timbre information is absorbed by the VQ codebook, which puts great pressure on the P-LLM" [§4.4]

**Timbre/Prosody prompt length** [Table 3]:
- 更长 timbre prompt (60s T) → SIM 从 0.905 → 0.930,CMOS-S +0.353
- 更长 prosody prompt (300s P) → DTW 从 32.30 → 30.25,CMOS-S +0.196

[agent解读] Timbre 和 prosody 的 prompt 效果可以**独立**改善,进一步验证了解耦的有效性。

**VQ vs VAE vs VAE+LDM** [Table 4]:
- w/VAE: WER 2.31% (vs VQ 2.28%) — 接近但 SIM 略低
- w/VAE+LDM (NaturalSpeech 2 style): 与 Ours-10s 相当,但远不如 Ours-300s

[agent解读] 这直接对比了 Mega-TTS 2 和 NaturalSpeech 2 的路线: VAE+LDM 在 10s prompt 时与 VQ+P-LLM 打平,但在 300s 长 prompt 时 P-LLM 路线大幅领先,说明多句 prompting 是 Mega-TTS 2 的核心优势。

### 3.5 Scaling Up [Appendix B, Table 6]

| Setting | WER↓ | SIM↑ | CMOS-Q | CMOS-S |
|---------|------|------|--------|--------|
| Ours-10s (60K) | 2.28% | 0.905 | 0.000 | 0.000 |
| Ours-10s (200K) | 2.15% | 0.922 | +0.010 | +0.215 [Table 6] |

[论文原文] Scaling data from 60K → 200K improves SIM from 0.905 → 0.922 [Table 6]。

## 4. 与已有方法的差异 / Comparison with Existing Work

| 维度 | Mega-TTS 2 | NaturalSpeech 2 | VALL-E |
|------|-----------|-----------------|-------|
| 表示 | Content (Transformer) + Prosody (VQ codes) + Timbre (MRTE) | Continuous latent (RVQ sum) | Discrete codec tokens (RVQ展开) |
| 生成模型 | AR (P-LLM on prosody) + NAR (AE decoder) | NAR (Diffusion) | AR + NAR (first + rest layers) |
| 多句 prompt | 天然支持 (300s) | 单句 3-10s | 单句 3s (20s退化) |
| Prosody 控制 | 显式解耦 + interpolation | 隐式 (diffusion 隐含) | 隐式 (in-context) |
| Timbre 来源 | Multi-ref attention (y_bar) | Speech prompt encoder | Codec token prefix |
| 理论框架 | 信息论 (互信息分解) | Regeneration learning | Codec language modeling |

## 5. 与 Mega-TTS 1 的关系

[agent解读] Mega-TTS 1 → 2 的核心升级:
1. **Prompting**: 单句 → 多句 (MRTE 替代简单 speaker encoder,P-LLM 支持长 context)
2. **Prosody 控制**: 无显式控制 → Prosody interpolation (Eq 3)
3. **规模**: 20K hours → 60K hours (可扩展到 200K)
4. **Venue**: arXiv → ICLR 2024 (peer-reviewed)

## 6. 局限与未来方向

[论文原文] [Appendix E.1]:
1. 仅支持英语,计划引入多语言训练数据
2. 语音质量可通过引入更高保真训练数据改善
3. 注意力窗口设计可进一步增强 P-LLM 的 in-context learning

[agent解读] 更深层的局限:
1. 依赖外部 aligner (MFA) 提取 duration/pitch → 不是端到端
2. VQ codebook 的信息瓶颈参数 (r, d) 需要搜索(r=8, d=16 是在 WER/SIM/DTW/CMOS 多指标上搜索得到的) [Appendix F]
3. P-LLM 只建模 prosody,content 由 FastSpeech 2 式 NAR encoder 处理 → 无法像 VALL-E 那样隐式学习 text-prosody 联合分布

---

> [!review] 审阅摘要
> - **结论**: pass
> - **主要优点**: 信息论驱动的分解框架清晰; 多句 prompting 结果令人信服(300s 追上 fine-tuning); Prosody interpolation 是实用的可控性工具; 消融实验完整(MRTE/VQ/prompt length/compression rate/prosody strategy)
> - **次要建议**: Mega-TTS 1 vs 2 的直接消融对比可更系统化

检索命中: [[SpeechFactorization]], [[ProsodyModeling]], [[SpeakerEmbedding]], [[LLM-basedTTS]], [[SpeechTokenizer]], [[CodebookCollapse]] | 过滤: [[VoiceCloningTaxonomy]](pending-review) | 未命中但可能相关: 无
