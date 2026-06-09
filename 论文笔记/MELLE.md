---
type: paper
tier: deep
title: "MELLE: Autoregressive Speech Synthesis without Vector Quantization"
arxiv_id: "2407.08551"
source: "Sources/MELLE.pdf"
authors: [Lingwei Meng, Long Zhou, Shujie Liu, Sanyuan Chen, Bing Han, Shujie Hu, Yanqing Liu, Jinyu Li, Sheng Zhao, Xixin Wu, Helen Meng, Furu Wei]
year: 2024
venue: "arXiv preprint"
tags: [TTS, zero-shot, autoregressive, continuous-token, mel-spectrogram, variational-inference, LLM-TTS, codec-free]
concepts: ["[[MelSpectrogram]]", "[[VariationalAutoencoderforTTS]]", "[[LLM-basedTTS]]", "[[CodecLanguageModel]]", "[[NeuralVocoder]]", "[[ResidualVectorQuantization]]"]
models: ["[[模型库/MELLE|MELLE]]"]
tasks: [TTS, zero-shot-TTS]
datasets: [Libriheavy, LibriSpeech]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[MelSpectrogram]], [[VariationalAutoencoderforTTS]], [[LLM-basedTTS]], [[CodecLanguageModel]], [[NeuralVocoder]], [[ResidualVectorQuantization]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]]✓, [[CodecLanguageModel]]✓, [[ResidualVectorQuantization]]✓, [[NeuralVocoder]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechFactorization]]✓ | 过滤: [[MelSpectrogram]](pending-review), [[VariationalAutoencoderforTTS]](pending-review) | 未命中但可能相关: 无

- **LLM-based TTS**: MELLE 属于 LLM-based TTS 范式但颠覆了其核心假设 -- 不使用离散 token,直接在连续 mel-spectrogram 上做自回归语言建模。KB 已有 VALL-E 等系统均依赖 codec discrete tokens。
- **Codec Language Model**: MELLE 的核心创新是绕过 VQ,直接预测连续值。KB 中 CodecLM 的关键挑战之一是"多层 RVQ 建模" -- MELLE 通过消除 RVQ 从根本上避免了这个问题。
- **RVQ**: KB 记录了 RVQ 的 codebook collapse、quantization error 等固有问题。MELLE 的论点正是:离散 codec codes"originally designed for audio compression, sacrifice fidelity compared to continuous mel-spectrogram" [§1]。
- **Mel Spectrogram** [待确认]: MELLE 回归 mel spectrogram 作为中间表示,但用法与传统 TTS 不同 -- 不是 predict-then-vocoder 的两阶段,而是单一 AR LM 直接预测 mel frames。
- **VAE for TTS** [待确认]: MELLE 的 Latent Sampling Module 借鉴 VAE 的 reparameterization trick,但用途不同 -- 不是为了 VAE ELBO 训练,而是为连续空间的 AR 模型提供采样机制。
- **Neural Vocoder**: MELLE 使用 HiFi-GAN 从 refined mel-spectrogram 合成波形 [§4.2]。

> [!summary] 速查
> - **一句话**: 首个在连续 mel-spectrogram 空间做自回归语言建模的零样本 TTS,通过 spectrogram flux loss + latent sampling module 替代离散 codec tokens,实现单阶段高效推理 [论文原文]
> - **路线**: Text(BPE)→Pre-net→Transformer Decoder(AR prediction of mel frames)→Latent Sampling Module→Post-Net→HiFi-GAN→Waveform [Fig 1-2]
> - **指标**: Continuation WER_H 1.98, SIM 0.508 (vs VALL-E WER_H 3.8, SIM 0.508); Cross-sentence WER_H 2.10, SIM 0.625; MOS 4.20 (vs GT 4.29, VALL-E 2 4.08); SMOS 4.40 (超越 GT 3.94); 推理 5.49s (vs VALL-E 7.32s, MELLE-R4 仅 1.40s) [Table 1, 3, 5]
> - **可借鉴**: (1) Spectrogram flux loss 防止连续预测的 static/repetitive frames; (2) Latent sampling module 为连续 token AR 提供采样机制; (3) Reduction factor r 实现推理加速
> - **局限**: (1) 依赖外部 HiFi-GAN vocoder (非端到端); (2) 仅英文评估; (3) 连续 mel 的 SIM 在 cross-sentence 上略低于 VALL-E 2

## 核心问题

MELLE 解决两个核心挑战:

1. **连续空间的训练目标**: 离散 token 的 LM 用 cross-entropy loss,但连续 mel-spectrogram 无法直接用 CE。如何为连续值设计有效的训练目标? [§1, "(i) How to set training objectives for continuous representation?"] [论文原文]
2. **连续空间的采样机制**: 离散 LM 用 top-p/top-k random sampling 增加多样性并避免 collapse。连续空间无法做离散采样,如何引入随机性和多样性? [§1, "(ii) How to enable sampling mechanism in continuous space?"] [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MELLE 是 decoder-only Transformer + 连续 mel-spectrogram prediction [Fig 1]:

**输入**: BPE text tokens $x$ + speech prompt (text transcript $\tilde{x}$ + mel-spectrogram $\tilde{y}$)
**输出**: 目标 mel-spectrogram $y = [y_0, y_1, ..., y_{T-1}]$,每帧 80-dim log-magnitude mel [§A.2]

**核心公式** [Eq. 1]:
$$p(y | x; \theta) = \prod_{t=0}^{T-1} p(y_t | y_{<t}, x; \theta)$$

**与 VALL-E 的本质区别**: VALL-E 预测 EnCodec 离散 codes (需 AR+NAR 两阶段); MELLE 预测连续 mel frames (单阶段) [论文原文]。

### 关键设计选择

#### 1. Autoregressive Language Model [§3.2.1]

- 12-layer Transformer decoder, 16 attention heads, embed dim 1024, FFN 4096 [§4.2]
- Text pre-net: BPE embedding layer (vocab 4K) + \<EOS\> token
- Acoustic pre-net: 3-layer MLP (mel→LM dim), dropout 0.5 (训练+推理时均开启,following Tacotron) [§4.2]
- 输入为 text embeddings + acoustic embeddings 的 concatenation

**WHY dropout at inference**: Tacotron 的经典技巧 -- inference 时保持 pre-net dropout 引入微小变化,防止过度确定性导致的单调输出 [agent 解读]。

#### 2. Latent Sampling Module (LSM) [§3.2.2]

为连续 AR 模型提供采样机制,受 VAE 启发 [Fig 2 left]:

1. LM 输出 $e_t$ 通过 linear layer 预测 $\mu_t$ 和 $\log \sigma_t^2$ [Eq. 3-4]
2. 通过 reparameterization 采样: $z_t = \mu_t + \sigma_t \odot \epsilon$, 其中 $\epsilon \sim \mathcal{N}(0, I)$ [Eq. 3]
3. $z_t$ 通过 3-layer MLP (with residual connection) 映射回 mel 空间: $y_t'$ [§3.2.2]

**WHY**: 离散 LM 通过 top-p sampling 引入多样性; 连续空间无法做随机选择。LSM 提供了一种概率采样机制,增强 expressive diversity 和 robustness [§3.2.2] [论文原文]。

**KL prior 的巧妙设计**: 不使用标准正态 $\mathcal{N}(0, I)$ 作为 prior,而是用 $\mathcal{N}(y_t, I)$ (以 ground-truth mel 为中心) -- 加速优化路径 [§3.3] [论文原文]。[agent 解读] 这让 KL 损失鼓励 $\mu_t$ 接近 $y_t$ 而非零向量,与 regression loss 方向一致,避免优化冲突。

#### 3. Stop Prediction Layer [§3.2.3]

Binary classifier 从 $e_t$ 预测生成终止 [Fig 2 mid]:
- Linear layer → sigmoid → 0/1
- BCE loss with 100x positive weight (因为正样本极少 -- 每句仅最后一帧为正) [§3.3]

#### 4. Post-Net [§3.2.3]

5 层 conv block (kernel 5, 256 channels) 对 coarse mel $y'$ 做残差细化: $y'' = y' + \text{PostNet}(y')$ [Fig 2 right]
- 训练时 teacher-forcing (输入 GT mel); 推理时在 AR 完成后一次性处理 [§3.2.3]

### 训练策略

四个损失函数协同 [Eq. 5]:
$$\mathcal{L} = \mathcal{L}_{reg} + \lambda \mathcal{L}_{KL} + \beta \mathcal{L}_{flux} + \gamma \mathcal{L}_{stop}$$

#### Regression Loss $\mathcal{L}_{reg}$ [Eq. 6]
- L1 + L2 loss,同时施加在 intermediate $y'$ 和 refined $y''$ 上
- $\mathcal{L}_{reg} = \|y - y'\|_1 + \|y - y'\|_2^2 + \|y - y''\|_1 + \|y - y''\|_2^2$

#### KL Divergence Loss $\mathcal{L}_{KL}$ [Eq. 7]
- $\lambda = 0$ for first 10K steps (warm-up),then $\lambda = 0.1$ [§A.3]
- 正则化 latent space,平衡 synthesis quality 和 diversity

#### Spectrogram Flux Loss $\mathcal{L}_{flux}$ (核心创新) [Eq. 8]
$$\mathcal{L}_{flux}(y, \mu) = -\sum_{t=1}^{T-1} \|\mu_t - y_{t-1}\|_1$$
- **关键**: 负号 -- 惩罚连续帧之间差异过小 (即惩罚 static frames)
- 用 L1 norm 衡量预测均值 $\mu_t$ 与前一帧 GT $y_{t-1}$ 的差异
- $\beta = 0.5$ [§A.3]

**WHY**: 连续空间 regression loss 天然倾向于预测"安全"的均值,导致 static/repetitive frames (长静音、重复片段)。Flux loss 通过惩罚平坦预测,迫使模型产出 dynamic 和 diverse spectrograms [§3.3] [论文原文]。

#### Stop Prediction Loss $\mathcal{L}_{stop}$
- BCE loss, $\gamma = 1.0$, positive weight 100 [§A.3]

#### Training Details [§A.3]
- 16 NVIDIA V100 32G GPUs, batch 480K frames, 400K steps
- AdamW optimizer, lr warm-up to 5e-4 over 32K steps, then linear decay
- MELLE-limited: LibriSpeech 960h, batch 80K frames, phoneme text tokens

#### Reduction Factor $r$ [§3.1]

可选: 将 mel 每 $r$ 帧分组,每步预测 $r$ 帧 [Eq. 2]:
- $r=1$: 标准单帧预测 (625 AR steps for 10s speech)
- $r=2$: 双帧预测 (312 steps, ~2x 加速)
- $r=4$: 四帧预测 (156 steps, ~4x 加速)
- Robustness (WER) 随 $r$ 增大保持稳定; SIM 随 $r$ 增大而下降 [Table 1]

## 实验

### Objective Evaluation (Table 1)

| 指标 | MELLE | MELLE-R2 | VALL-E | VALL-E 2 | Voicebox | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER_C (Cont.) | 1.47 | 1.45 | - | 1.6 | - | LibriSpeech test-clean | [Table 1] |
| WER_H (Cont.) | 1.98 | 2.02 | 3.8 | 2.32 | 2.0 | LibriSpeech test-clean | [Table 1] |
| SIM (Cont.) | 0.508 | 0.489 | 0.508 | 0.504 | 0.593 | LibriSpeech test-clean | [Table 1] |
| WER_C (Cross) | 1.47 | 1.50 | 5.9 | 2.44 | 1.9 | LibriSpeech test-clean | [Table 1] |
| WER_H (Cross) | 2.10 | 2.14 | - | - | - | LibriSpeech test-clean | [Table 1] |
| SIM (Cross) | 0.625 | 0.608 | 0.580 | 0.643 | 0.662 | LibriSpeech test-clean | [Table 1] |

### Subjective Evaluation (Table 3)

| 指标 | MELLE | MELLE-R2 | VALL-E 2 | VALL-E | YourTTS | GT | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOS | 4.20 | 4.14 | 4.08 | 3.18 | 2.41 | 4.29 | [Table 3] |
| SMOS | **4.40** | 4.18 | 3.88 | 3.50 | 2.62 | 3.94 | [Table 3] |
| CMOS | **-0.032** | -0.252 | -0.085 | -0.912 | -2.162 | 0.000 | [Table 3] |

**SMOS 超越 GT (4.40 vs 3.94)** -- MELLE 的 speaker similarity 被人类评估为优于真实语音,表明其 in-context learning 能力极强 [§5.2] [论文原文]。

### Ablation Study (Table 4)

| Latent Sampling | Flux Loss | WER_H (Cont.) | SIM (Cont.) | WER_H (Cross) | SIM (Cross) |
| --- | --- | --- | --- | --- | --- |
| X | X | 6.91 | 0.483 | 23.65 | 0.518 |
| V | X | 4.07 | 0.486 | 10.87 | 0.584 |
| X | V | 2.61 | 0.506 | 5.90 | 0.602 |
| V (train only) | V | 2.13 | 0.506 | 2.72 | 0.615 |
| **V** | **V** | **1.98** | **0.508** | **2.10** | **0.625** |

关键发现:
- **Flux loss 对 robustness 贡献最大**: WER_H 从 6.91→2.61 (无 LS) [§5.3] [论文原文]
- **Latent sampling 对 SIM 贡献最大**: SIM 改善主要来自 LS,especially cross-sentence [§5.3] [论文原文]
- **Both + inference LS 最优**: 训练+推理都开 LS 比只训练时开更好 [Table 4 row 4 vs 5]

### Inference Efficiency (Table 5)

| 系统 | AR Steps | 推理时间 (10s speech) |
| --- | --- | --- |
| VALL-E / VALL-E 2 | 750 | 7.32s |
| MELLE | 625 | 5.49s |
| MELLE-R2 | 312 | 2.76s |
| MELLE-R4 | 156 | 1.40s |
| VALL-E R | 375 | 3.67s |
| CLaM-TTS | - | 4.15s |

MELLE-R4 仅需 1.40s 生成 10s 语音,超越所有对比系统 [Table 5]。

## 局限性

1. **Vocoder 依赖**: 使用 585h LibriTTS 训练的 HiFi-GAN,合成质量受 vocoder 限制 [§6] [论文原文]。Voicebox 使用 60K 小时自研 vocoder,这可能是其 SIM 更高的原因之一 [§5.1]
2. **仅英文评估**: 虽有中文 demo 版,但论文仅报告英文 LibriSpeech 结果 [§6] [论文原文]
3. **仅 mel-spectrogram 作为连续表示**: 未探索其他连续表示 (如 VAE latent states) [§6] [论文原文]
4. **Cross-sentence SIM 略低于 VALL-E 2**: 0.625 vs 0.643,归因于 speaker verification model bias [§5.1] [论文原文] (用 ECAPA-TDNN 评估时 MELLE 0.680 vs VALL-E 2 0.662)

## 点评

**优势**:
- 范式创新: 证明 zero-shot TTS 不需要 VQ,连续 mel 就够了,消除了 RVQ 的信息损失 [论文原文]
- 架构简洁: 单一 decoder-only Transformer,无需 AR+NAR 两阶段,比 VALL-E 简单得多
- SMOS 超越 GT: 4.40 vs 3.94,表明 speaker similarity 出色
- Reduction factor 提供质量-速度的灵活 trade-off

**不足**:
- Mel-spectrogram 是"老技术"的回归,可能被视为技术退步 [agent 解读]
- 推理时仍需 pre-net dropout,暗示模型对确定性输入不够 robust [agent 解读]
- 未与 flow-matching/diffusion-based TTS (CosyVoice, NaturalSpeech 3) 直接对比

## 可复用的 idea

1. **Spectrogram Flux Loss**: 通用技巧 -- 对任何连续序列预测任务,惩罚连续帧间差异过小可防止 static output。公式简单: $-\sum \|\mu_t - y_{t-1}\|_1$
2. **Latent Sampling Module**: 为连续 token AR 提供概率采样,可用于任何非离散 AR 生成任务
3. **Non-standard KL prior**: 用 $\mathcal{N}(y_t, I)$ 替代 $\mathcal{N}(0, I)$,加速 VAE 优化收敛
4. **Reduction factor $r$**: 多帧预测加速推理,适用于任何帧级 AR 模型 (mel, latent 等)
5. **Ground truth mel > EnCodec**: Table 1 直接证明从 mel reconstruction 比从 EnCodec reconstruction 有更好的 WER 和 SIM,为"回归连续表示"提供实证支持

检索命中: [[LLM-basedTTS]], [[CodecLanguageModel]], [[ResidualVectorQuantization]], [[NeuralVocoder]], [[SemanticvsAcousticTokens]], [[SpeechFactorization]] | 过滤: [[MelSpectrogram]](pending-review), [[VariationalAutoencoderforTTS]](pending-review) | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
