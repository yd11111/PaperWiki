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
