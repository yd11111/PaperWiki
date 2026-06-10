---
type: paper
tier: deep
title: "VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech"
arxiv_id: "2106.06103"
source: "Sources/VITS.pdf"
authors: [Jaehyeon Kim, Jungil Kong, Juhee Son]
year: 2021
venue: "ICML 2021 (PMLR 139)"
tags: [TTS, end-to-end, VAE, normalizing-flow, GAN, parallel-synthesis, duration-prediction]
concepts: ["[[VariationalAutoencoderforTTS]]", "[[Non-autoregressiveTTS]]", "[[DurationPredictor]]", "[[NeuralVocoder]]", "[[MelSpectrogram]]", "[[PhonemeRepresentation]]", "[[Speech-TextAlignment]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个实体页: [[NeuralVocoder]], [[VariationalAutoencoderforTTS]], [[Non-autoregressiveTTS]], [[DurationPredictor]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[NeuralVocoder]]✓(confirmed), [[VariationalAutoencoderforTTS]](pending-review), [[Non-autoregressiveTTS]](pending-review), [[DurationPredictor]](pending-review) | 过滤: [[MelSpectrogram]](pending-review), [[PhonemeRepresentation]](pending-review) | 未命中但可能相关: Normalizing Flow (无独立页)

**已有知识要点**:
- [[NeuralVocoder]]: HiFi-GAN (Kong et al., NeurIPS 2020) 是当前事实标准的 GAN vocoder,采用 Multi-Period Discriminator + Multi-Scale Discriminator,速度快 (13.4x 实时) 且质量高 [待确认]
- [[VariationalAutoencoderforTTS]]: VAE 在 TTS 中用于建模 one-to-many mapping,VITS 是 VAE 在 TTS 中的集大成者,结合了 normalizing flow 和 GAN [待确认]
- [[Non-autoregressiveTTS]]: VITS 被归类为 VAE+Flow fully E2E NAR 系统,在 NAR TTS 演进中位于 Glow-TTS 之后 [待确认]
- [[DurationPredictor]]: Glow-TTS 提出的 Monotonic Alignment Search (MAS) 是 VITS 对齐估计的基础 [待确认]

## 速查

> [!summary] 速查
> - **一句话**: 首个将 conditional VAE + normalizing flow + GAN adversarial training 统一为端到端并行 TTS 系统的工作,单模型直接从 phoneme 生成波形,质量接近真实语音
> - **路线**: Phonemes → Text Encoder → Prior (normalizing flow) → z → HiFi-GAN Decoder → Waveform; 训练时: Linear Spectrogram → Posterior Encoder → z; 对齐由 MAS 自动估计; Duration 由 flow-based stochastic duration predictor 预测
> - **指标**: LJ Speech MOS 4.43 (GT 4.46) [Table 1]; VCTK MOS 4.38 (GT 4.38) [Table 3]; 合成速度 67.12x 实时 [Table 4]
> - **可借鉴**: (1) normalizing flow 增强 VAE prior 表达力; (2) stochastic duration predictor 建模韵律多样性; (3) linear spectrogram 作为 posterior encoder 输入提供高分辨率信息; (4) MAS 在 ELBO 框架下自动学习对齐
> - **局限**: 单说话人数据集实验为主 (LJ Speech 24h); 多说话人仅用 VCTK (44h, 109人); 文本前端仍依赖外部 G2P; 未在大规模数据上验证

## 核心问题

**VITS 要解决什么问题?** [论文原文]

传统两阶段 TTS pipeline (acoustic model + vocoder) 存在三个根本问题 [§1]:
1. **模块独立训练**: 两个阶段分别优化,无法联合优化全局目标
2. **依赖预定义中间表示**: mel spectrogram 是人为设计的瓶颈,限制了隐式表示学习的潜力
3. **级联误差**: 第一阶段的预测误差传播到第二阶段 (vocoder 在 teacher-forcing vs 生成 mel 上表现差异)

此前的端到端尝试 (FastSpeech 2s, EATS, Wave Tacotron) 虽然直接生成波形,但质量仍不如两阶段系统 [§1]。

**核心假设**: [agent 解读] 通过 VAE 框架将两个阶段统一到同一个隐空间中 — posterior encoder 提取语音的隐变量 z, prior encoder 从文本预测 z 的分布, decoder 从 z 生成波形 — 可以实现真正的端到端优化,同时通过 normalizing flow 和 GAN 分别增强 prior 表达力和生成质量。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VITS 可以表述为一个 conditional VAE,由五个组件构成 [§2, Fig 1]:

1. **Posterior Encoder** $q_\phi(z|x_{lin})$: 从 linear spectrogram 提取隐变量 z [§2.1.1]
2. **Prior Encoder** $p_\theta(z|c_{text}, A)$: 从 phoneme 序列通过 normalizing flow 建模 z 的先验分布 [§2.1.3, §2.5.2]
3. **Decoder** $p_\theta(x|z)$: HiFi-GAN V1 generator,从 z 生成波形 [§2.5.3]
4. **Discriminator** D: Multi-Period Discriminator,对抗训练提升质量 [§2.3]
5. **Stochastic Duration Predictor**: flow-based 模型,预测音素时长分布 [§2.2.2]

**训练**: Posterior encoder 从 linear spectrogram 提取 z, decoder 从 z 重建波形; prior encoder 学习从 text 预测 z 的分布 (通过 KL 散度与 posterior 对齐) [§2.1.1]

**推理**: Text → Prior Encoder → z → Decoder → Waveform (无需 posterior encoder 和 discriminator) [Fig 1b]

### 关键设计选择

#### 1. Normalizing Flow 增强 Prior [§2.1.3, §2.5.2]

[论文原文] "We found that increasing the expressiveness of the prior distribution is important for generating realistic samples." [§2.1.3]

Prior encoder 由 text encoder + normalizing flow $f_\theta$ 组成:
$$p_\theta(z|c) = N(f_\theta(z); \mu_\theta(c), \sigma_\theta(c)) \left|\det \frac{\partial f_\theta(z)}{\partial z}\right|$$

- Text encoder: Transformer encoder (relative positional encoding) → $h_{text}$ → linear projection → $\mu_\theta, \sigma_\theta$ [§2.5.2]
- Normalizing flow: affine coupling layers (4 层, 每层 4 个 WaveNet residual blocks), volume-preserving (无 scale 参数) [Appendix B.1]
- [agent 解读] normalizing flow 将简单的 factorized Gaussian prior 变换为复杂的多模态分布,弥补了 VAE 中 prior-posterior gap 的经典问题

**消融验证**: 去掉 normalizing flow 导致 MOS 下降 1.52 (4.50 → 2.98) [Table 2] — 这是所有消融中影响最大的因素

#### 2. Linear Spectrogram 作为 Posterior 输入 [§2.1.2, §2.1.3]

[论文原文] "We aim to provide more high-resolution information to the posterior encoder... use the linear-scale spectrogram of target speech $x_{lin}$ as input rather than the mel-spectrogram." [§2.1.3]

- Posterior encoder: 16 个 WaveNet residual blocks,输入 linear-scale log magnitude spectrogram [§2.5.1, Appendix B.1]
- [agent 解读] linear spectrogram 保留了 mel 映射丢失的高频细节,使 posterior 能提取更精确的隐变量,间接提升了 prior 学习的上界
- **消融验证**: 用 mel spectrogram 替换 linear spectrogram 导致 MOS 下降 0.19 (4.50 → 4.31) [Table 2]

#### 3. Monotonic Alignment Search (MAS) [§2.2.1]

VITS 复用了 Glow-TTS 的 MAS 方法来估计 text-speech 对齐 A [§2.2.1]:

$$\hat{A} = \arg\max_A \log N(f_\theta(z); \mu_\theta(c_{text}, \hat{A}), \sigma_\theta(c_{text}, \hat{A}))$$

- [论文原文] 由于 VITS 优化 ELBO (而非精确 log-likelihood),MAS 搜索最大化隐变量 z 的 log-likelihood 的对齐 [§2.2.1, Eq 6]
- 单调约束 + 非跳跃约束: 保证人类阅读文本的自然顺序 [§2.2.1]
- 动态规划 $O(|c_{text}| \times |z|)$ 复杂度 [Appendix A, Fig 4]

#### 4. Stochastic Duration Predictor [§2.2.2, §2.5.5]

[论文原文] VITS 设计了 flow-based stochastic duration predictor,学习音素时长的概率分布而非确定性预测 [§2.2.2]:

- **动机**: 确定性 duration predictor "cannot express the way a person utters at different speaking rates each time" [§2.2.2]
- **实现**: 引入两个隐变量 u (variational dequantization) 和 v (variational data augmentation),构建连续的 duration 分布 [§2.2.2, Eq 7]
- **训练目标**: 最大化 phoneme duration 的对数似然变分下界 [§2.2.2]
- **Stop gradient**: duration predictor 的训练与其他模块解耦,通过 stop gradient operator 阻断梯度 [§2.2.2]
- **架构**: dilated depth-wise separable convolution (DDSConv) residual blocks + neural spline flows [§2.5.5, Fig 5]

**消融验证**: VITS (DDP, deterministic) MOS 4.39 vs VITS (stochastic) MOS 4.43 [Table 1] — stochastic duration predictor 提升了韵律自然度

#### 5. 对抗训练 [§2.3]

VITS 在 VAE 基础上引入 GAN 训练:
- Discriminator: HiFi-GAN 的 Multi-Period Discriminator [§2.5.4]
- Generator loss: least-squares GAN loss + feature matching loss [§2.3, Eq 8-10]
- [agent 解读] GAN loss 补偿了 VAE 重建 loss 的 over-smoothing 倾向,是 VITS 音质超越纯 VAE/Flow 系统的关键

### 训练策略

**总损失函数** [§2.4, Eq 11]:
$$L_{vae} = L_{recon} + L_{kl} + L_{dur} + L_{adv}(G) + L_{fm}(G)$$

- $L_{recon}$: mel spectrogram L1 loss (仅在 mel 域比较,不在波形域) [§2.1.2, Eq 2]
- $L_{kl}$: KL divergence between posterior and prior [§2.1.3, Eq 3]
- $L_{dur}$: stochastic duration predictor loss (stop gradient 隔离) [§2.2.2, Eq 7]
- $L_{adv}(G)$: GAN adversarial loss for generator [§2.3, Eq 9]
- $L_{fm}(G)$: feature matching loss [§2.3, Eq 10]

**训练细节** [§3.3]:
- Optimizer: AdamW ($\beta_1=0.8, \beta_2=0.99, \lambda=0.01$)
- Learning rate: $2 \times 10^{-4}$, decay factor $0.999^{1/8}$ per epoch
- Windowed generator training: 随机截取 z 的长度 32 窗口送入 decoder [§3.3]
- Mixed precision training on 4x NVIDIA V100
- Batch size 64/GPU, 800k steps

## 实验

| 指标 | 本文 (VITS) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| MOS | 4.43 (+-0.06) | Glow-TTS+HiFi-GAN: 4.32 (+-0.07) | LJ Speech | [Table 1] |
| MOS | 4.43 (+-0.06) | Tacotron 2+HiFi-GAN (FT): 4.25 (+-0.07) | LJ Speech | [Table 1] |
| MOS | 4.43 (+-0.06) | Ground Truth: 4.46 (+-0.06) | LJ Speech | [Table 1] |
| MOS (multi-spk) | 4.38 (+-0.06) | Glow-TTS+HiFi-GAN (FT): 3.82 (+-0.07) | VCTK | [Table 3] |
| CMOS vs GT | -0.106 | (GT = 0) | LJ Speech | [Table 5] |
| CMOS vs GT | -0.262 | (GT = 0) | VCTK | [Table 5] |
| 合成速度 (kHz) | 1480.15 | Glow-TTS+HiFi-GAN: 606.05 | LJ Speech | [Table 4] |
| 实时倍率 | 67.12x | Glow-TTS+HiFi-GAN: 27.48x | LJ Speech | [Table 4] |
| VITS (DDP) 速度 | 2005.03 kHz (90.93x) | - | LJ Speech | [Table 4] |

**关键实验发现**:

1. **VITS 超越所有公开 TTS 系统** [Table 1]: MOS 4.43 优于最佳两阶段系统 Glow-TTS+HiFi-GAN (4.32),接近 GT (4.46) [论文原文]

2. **消融实验** [Table 2]: normalizing flow 贡献最大 (去掉 → MOS 从 4.50 降至 2.98); linear spectrogram 输入也有明显增益 (替换为 mel → 降 0.19) [论文原文]

3. **语音变化性** [§4.3, Fig 2, Fig 3]: stochastic duration predictor 使 VITS 生成的语音在时长和 F0 上展现丰富多样性,而 Glow-TTS 只能产生固定时长 [论文原文]

4. **多说话人** [Table 3]: VITS 在 VCTK 上 MOS 4.38,大幅超越 Tacotron 2+HiFi-GAN (3.19) 和 Glow-TTS+HiFi-GAN (3.82) [论文原文]

5. **合成速度** [Table 4]: VITS 67.12x 实时,是 Glow-TTS+HiFi-GAN (27.48x) 的 2.4 倍 [论文原文]

## 局限性

1. **数据规模受限**: 仅在 LJ Speech (24h, 单人) 和 VCTK (44h, 109人) 上验证,未在大规模数据 (>1000h) 上测试 [§6]
2. **文本前端依赖**: 仍需外部 G2P (phonemizer) 将文本转为 IPA 音素序列 [§3.2]
3. **无零样本能力**: 新说话人需重新训练或 fine-tune,不支持 prompt-based voice cloning [agent 解读]
4. **语言覆盖**: 仅英语实验,未验证跨语言泛化 [agent 解读]
5. **文本前端可改进**: [论文原文] "Investigating self-supervised learning of language representations could be a possible direction for removing the text preprocessing step" [§6]

## 点评

**历史地位**: [agent 解读] VITS 是端到端 TTS 的里程碑工作,首次证明单模型可以从 phoneme 直接生成高质量波形,质量超越最佳两阶段系统。其架构设计 (VAE + Flow + GAN) 成为后续众多工作的基础:
- VITS 2 (Kim et al., 2023): 多说话人扩展
- Naturalspeech (Tan et al., 2022): 在 VITS 基础上引入 memory-based VAE
- MB-iSTFT-VITS: 轻量化 decoder
- 众多 TTS 系统将 VITS 的 posterior encoder + normalizing flow 架构作为模块复用

**方法论贡献**: 
1. 将 MAS (来自 Glow-TTS) 从 flow-based 模型推广到 VAE 框架 [§2.2.1]
2. Stochastic duration predictor 的设计 (variational dequantization + data augmentation) 优雅地解决了离散 duration 的连续建模问题 [§2.2.2]
3. Linear spectrogram posterior encoder 是一个简单但有效的设计选择,为 posterior 提供更丰富的信息 [§2.1.3]

**与当前 LLM-TTS 的关系**: [agent 解读] VITS 代表了 NAR TTS 的巅峰。后续 VALL-E (2023) 将 TTS 转向 LLM-based codec language model 范式,但 CosyVoice 等系统仍在 flow matching decoder 部分借鉴了 VITS 中 VAE + flow 的思想。VITS 的"从 phoneme 到波形"的端到端理念,通过不同的技术路径 (LLM + codec + flow matching),在后续工作中得到传承。

## 可复用的 idea

1. **Normalizing flow 增强 VAE prior**: 当 VAE 的 prior 表达力不足时,用 normalizing flow 变换可以极大提升生成质量 — 这是一个通用的生成模型增强策略
2. **Stochastic duration predictor**: flow-based 概率 duration 建模可以迁移到任何需要建模离散时长分布的场景 (SVS, speech editing 等)
3. **Linear spectrogram 作为高分辨率 posterior 输入**: 在信息瓶颈设计中,给 encoder 更丰富的输入往往比增大模型更有效
4. **MAS 在 ELBO 框架下的推广**: 将对齐搜索嵌入变分推断框架,消除外部对齐工具依赖
5. **Windowed generator training**: 随机截取 latent 序列的固定窗口送入 decoder,大幅减少训练显存 [§3.3]

---

检索命中: [[NeuralVocoder]]✓, [[VariationalAutoencoderforTTS]](pending-review), [[Non-autoregressiveTTS]](pending-review), [[DurationPredictor]](pending-review) | 过滤: [[MelSpectrogram]](pending-review), [[PhonemeRepresentation]](pending-review) | 未命中但可能相关: Normalizing Flow (无独立页)


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排

---

## 代码级分析

> [!info] 代码来源
> - 仓库: https://github.com/jaywalnut310/vits
> - commit: 2e561ba
> - 分析日期: 2026-06-10
> - 说明: 作者官方实现

### 架构验证

代码与论文 Fig 1 完全对应:

- `models.py:SynthesizerTrn` = 整体训练架构,包含 5 个核心组件:
  - `enc_p` = `TextEncoder`: phoneme embedding → Transformer encoder → 投影到 (mu, logs),即 prior encoder 的文本编码部分
  - `flow` = `ResidualCouplingBlock`: 4 层 affine coupling (mean_only=True, volume-preserving),即 normalizing flow
  - `enc_q` = `PosteriorEncoder`: 1x1 conv → 16 层 WN (WaveNet residual blocks) → 投影到 (mu, logs),输入 linear spectrogram
  - `dec` = `Generator`: HiFi-GAN V1 generator (upsample rates [8,8,2,2],产生 256x 上采样)
  - `dp` = `StochasticDurationPredictor` (或 `DurationPredictor`): flow-based duration predictor

- `models.py:MultiPeriodDiscriminator` = HiFi-GAN 的 MPD,periods = [2,3,5,7,11],外加一个 `DiscriminatorS`

**与论文的差异**:
1. **Discriminator 包含 DiscriminatorS**: 论文仅提到 Multi-Period Discriminator,代码额外加了 Multi-Scale Discriminator (DiscriminatorS),组合为 `MultiPeriodDiscriminator`
2. **ResidualCouplingBlock 的 mean_only=True** (`models.py:199`): flow 层仅学习 mean shift,不学习 scale (volume-preserving),与论文 Appendix B.1 一致

### 论文未写的实现细节

1. **Stop gradient on duration predictor input** (`models.py:51`): `x = torch.detach(x)` — duration predictor 的输入 detach 了梯度,确保 duration loss 不影响 text encoder。同样 global conditioning `g = torch.detach(g)` 也 detach。这对应论文 Section 2.2.2 的 stop gradient operator。

2. **MAS 实现在 C++ extension** (`monotonic_align/`): `monotonic_align/core.pyx` 是 Cython 实现的动态规划,需要预先 `cd monotonic_align && python setup.py build_ext --inplace` 编译。

3. **Segment 随机裁切** (`models.py:495`): `z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)` — 训练时从 z 中随机裁取 segment_size=32 帧 (对应 32*256=8192 样本) 送入 decoder,这是论文 Section 3.3 的 "windowed generator training"。

4. **KL loss 的实际计算** (`train.py:184`): `loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask)` — 在 flow 变换后的空间计算 KL,其中 z_p = flow(z), m_p/logs_p 是 prior 参数 (已经过 attention 扩展)。

5. **Mel loss 系数 c_mel=45** (`configs/ljs_base.json`): mel reconstruction loss 的权重为 45,远大于 c_kl=1.0,说明重建质量的优先级远高于 KL 正则化。

6. **add_blank=true** (`configs/ljs_base.json`): 在每个 phoneme token 之间插入 blank token,扩展序列长度到约 2 倍。这是 Glow-TTS 的设计,帮助 MAS 更灵活地分配 duration。

7. **DistributedBucketSampler** (`data_utils.py`): 自定义采样器按音频长度分桶 [32,300,400,...,1000],减少 padding 浪费。

8. **Inference noise scales** (`models.py:499`): `noise_scale` 控制 z 采样噪声 (默认 1.0); `noise_scale_w` 控制 duration predictor 噪声 (默认 1.0); `length_scale` 控制语速 (默认 1.0)。

9. **Voice conversion** (`models.py:525-533`): 代码提供 `voice_conversion` 方法 — 将源说话人的音频通过 posterior encoder → flow → 换目标说话人 embedding → reverse flow → decoder,论文未详细讨论此功能。

### 训练 pipeline 拆解

数据流: (phoneme_ids, linear_spectrogram, waveform) → SynthesizerTrn.forward:
1. `enc_p(phonemes)` → h_text, m_p, logs_p (prior 参数)
2. `enc_q(linear_spec)` → z, m_q, logs_q (posterior 采样)
3. `flow(z)` → z_p (flow 变换后的 z)
4. MAS: 计算 neg_cent (z_p 在 prior 下的对数似然) → `monotonic_align.maximum_path` → attention matrix
5. `w = attn.sum(2)` → phoneme durations; `dp(h_text, w)` → duration loss
6. 用 attn 扩展 m_p, logs_p 到 frame 级
7. `rand_slice_segments(z)` → z_slice → `dec(z_slice)` → y_hat (波形)
8. Discriminator: `net_d(y_real, y_hat)` → disc_loss
9. Generator loss: `loss_gen + loss_fm + loss_mel*45 + loss_dur + loss_kl*1.0`

- Optimizer: AdamW (lr=2e-4, betas=[0.8,0.99], eps=1e-9)
- LR scheduler: ExponentialLR (gamma=0.999875)
- Mixed precision: FP16 via torch.cuda.amp
- Multi-GPU: DDP with NCCL backend

### 推理 pipeline 拆解

phoneme_ids → `enc_p` → m_p, logs_p → `dp(reverse=True, noise_scale_w)` → logw → exp → ceil → durations → `generate_path` → attn → expand m_p, logs_p → sample z_p from N(m_p, exp(logs_p)) * noise_scale → `flow(reverse=True)` → z → `dec(z)` → waveform

- 无需 posterior encoder, discriminator, linear spectrogram
- 单次前向,完全并行
- 三个可调参数: noise_scale (音质 vs 多样性), noise_scale_w (duration 多样性), length_scale (语速)

### 关键超参数表

| 参数 | 论文值 | 代码实际值 (ljs_base.json) | 备注 |
|------|--------|-----------|------|
| inter_channels | 192 | 192 | 隐变量 z 的通道数 |
| hidden_channels | 192 | 192 | text encoder hidden dim |
| filter_channels | 768 | 768 | FFN filter size |
| n_heads | 2 | 2 | attention heads |
| n_layers (encoder) | 6 | 6 | transformer layers |
| kernel_size | 3 | 3 | - |
| p_dropout | 0.1 | 0.1 | - |
| n_flows | 4 | 4 (ResidualCouplingBlock) | affine coupling layers |
| posterior n_layers | 16 | 16 (WN in PosteriorEncoder) | WaveNet residual blocks |
| upsample_rates | [8,8,2,2] | [8,8,2,2] | total 256x (= hop_length) |
| upsample_kernel_sizes | [16,16,4,4] | [16,16,4,4] | - |
| resblock_kernel_sizes | [3,7,11] | [3,7,11] | HiFi-GAN MRF |
| segment_size | 32 (frames) | 8192 (samples) / 256 (hop) = 32 | 一致 |
| learning_rate | 2e-4 | 2e-4 | - |
| betas | [0.8, 0.99] | [0.8, 0.99] | 论文: AdamW |
| lr_decay | 0.999^(1/8)/epoch | 0.999875/epoch | 近似等价 |
| batch_size | 64 | 64 | - |
| c_mel | 未明确 | 45 | mel loss 权重 |
| c_kl | 未明确 | 1.0 | KL loss 权重 |
| fp16 | yes (V100) | true | - |
| epochs | 未明确 (800k steps) | 20000 | - |
| MPD periods | [2,3,5,7,11] | [2,3,5,7,11] | 一致 |
| add_blank | 未明确 | true | 论文中未显著讨论 |
| sampling_rate | 22050 | 22050 | - |

### 复现 checklist (基于代码)

- [ ] 环境依赖: PyTorch >= 1.6, Cython, librosa, scipy, tensorboard, unidecode, phonemizer
- [ ] 编译 MAS: `cd monotonic_align && python setup.py build_ext --inplace`
- [ ] 数据准备: LJSpeech 下载 → 生成 filelists (已提供) → text cleaning (phonemizer)
- [ ] 预训练模型依赖: 无 (从头训练)
- [ ] 训练命令: `python train.py -c configs/ljs_base.json -m ljs_base`
- [ ] 推理命令: 通过 `inference.ipynb` (加载 checkpoint + `net_g.infer()`)
- [ ] 已知坑: Cython MAS 编译可能在 Mac/Windows 上失败; phonemizer 需要 espeak-ng 后端; 多 GPU 训练硬编码 MASTER_PORT=80000 可能冲突; 20000 epoch 在 4xV100 上约需 5-7 天; add_blank 对质量影响大但易被忽略

### 代码质量与可复现性评估

- **工程质量**: 3/5 - 作者官方代码,简洁但缺少注释; 一些变量命名不够直观 (如 c_mel, c_kl 需要看 config 才知含义)
- **文档完善度**: 2/5 - README 极简 (仅几行),无详细训练指南; inference.ipynb 是唯一的推理参考
- **社区活跃度**: 2/5 - 2021 年后无更新,但 fork 数量极大 (>4000),社区有大量改进版本 (如 vits2, MB-iSTFT-VITS)
- **复现难度**: 3/5 - 代码能跑但需要注意 MAS 编译、phonemizer 安装、数据预处理; 训练时间较长
