---
tier: deep
title: "FlowDec"
aliases: [FlowDec Codec, Flow-based Audio Codec]
authors: ["Simon Welker", "Matthew Le", "Ricky T.Q. Chen", "Wei-Ning Hsu", "Timo Gerkmann", "Alexander Richard", "Yi-Chiao Wu"]
year: 2025
arxiv_id: ""
source: "Sources/FlowDec.pdf"
venue: "ICLR 2025"
tags: [audio-codec, flow-matching, postfilter, non-adversarial, full-band, 48kHz, general-audio, neural-codec]
level: deep
status: draft
concepts: ["[[ConditionalFlowMatching]]", "[[ScoreMatching]]", "[[ResidualVectorQuantization]]", "[[NeuralVocoder]]", "[[Diffusion-basedVocoder]]"]
models: ["[[模型库/EnCodec|EnCodec]]"]
tasks: [audio-codec, speech-enhancement, audio-generation]
created: 2026-06-03
updated: 2026-06-03
kb_sources: ["[[ConditionalFlowMatching]]", "[[ResidualVectorQuantization]]", "[[NeuralVocoder]]"]
---
tier: deep

## KB 背景

本文涉及以下已有知识:

- **[[ConditionalFlowMatching]]** (confirmed): FlowDec 使用一种新的 joint flow matching 变体作为 stochastic postfilter。与 TTS 中 CFM 用于 mel spectrogram 生成不同,FlowDec 将 CFM 用于 **audio codec 后处理** — 增强 deterministic decoder 输出的感知质量。其 shifted prior q_0(x_0|x_1) = N(x; y, Sigma_y) 是对标准 CFM 的关键改进 [§3.2]。
- **[[ResidualVectorQuantization]]** (confirmed): FlowDec 的底层 codec (NDAC) 基于 DAC 架构,使用 RVQ 进行量化。NDAC-75 使用 10 个 codebook (0.75-7.50 kbit/s); NDAC-25 使用 16 个 codebook (0.25-4.00 kbit/s)。通过控制推理时启用的 codebook 数量实现可变比特率 [§3.4, Table 2]。
- **[[NeuralVocoder]]** (confirmed): FlowDec 的 postfilter 在概念上类似于 diffusion-based vocoder — 都是从中间表示恢复高质量波形。但 FlowDec 的 postfilter 操作在 STFT 域而非 mel 域,且目标是增强而非生成。
- **[[ScoreMatching]]** [待确认]: FlowDec 的前身 ScoreDec 基于 SGMSE/SGMSE+ (score-based generative models for speech enhancement)。FlowDec 用 flow matching 替代 score-based SDE,将 DNN evaluations 从 60 减至 6 [§1]。
- **[[Diffusion-basedVocoder]]** [待确认]: FlowDec 与 diffusion vocoder 的关系: 两者都是从中间表示 (codec output / mel spectrogram) 恢复高质量音频。但 FlowDec 是 postfilter (增强已有输出),diffusion vocoder 是 generator (从条件特征生成波形)。

> [!summary] 速查
> - **一句话**: 基于 conditional flow matching 的 stochastic postfilter,将非对抗训练的 neural codec (NDAC) 输出增强到与 GAN-based codec (DAC) 相当的感知质量,同时支持低至 4 kbit/s 的全频段 48kHz 音频编码
> - **路线**: Audio(48kHz) → Encoder → RVQ(NDAC, non-adversarial) → Decoder(deterministic) → initial estimate y → Postfilter(joint flow matching, NCSN++, 6 NFE) → enhanced audio hat{x}
> - **指标**: FAD 1.62 (NFE=6), 优于 DAC 3.0-10.0; MUSHRA 与 DAC on par @ 4.5-7.5 kbit/s; RTF 0.2285 (vs ScoreDec 1.707) [Table 4, Fig.5, Fig.6]
> - **可借鉴**: (1) 非对抗 codec + flow postfilter 的解耦设计 (2) Shifted prior q0 = N(y, Sigma_y) 消除 OT solver 需求 (3) 频率依赖噪声 sigma_y(f) 适配自然信号频谱特性 (4) CQT loss 改善低频建模
> - **局限**: 非因果架构(non-streaming); RTF 0.23 虽优于 ScoreDec 但仍非实时; 多次 NFE 推理增加延迟; 48kHz 全频段训练资源需求大

## 核心问题

### WHY: 为什么要做这个工作?

现有 neural audio codecs (SoundStream, EnCodec, DAC) 依赖 GAN 对抗训练获取高感知质量,但存在以下问题 [§1, §3.3]:

1. **GAN 训练不稳定**: 复杂多判别器 + 精心设计的损失权重,缺乏可解释性 [§3.3] [论文原文]
2. **非对抗训练的 codec 质量差**: 仅用频谱/波形损失训练的 codec 产生模糊、不自然的输出 [§3.3]
3. **ScoreDec 局限**: 前身 ScoreDec 虽用 score-based postfilter 解决了上述问题,但仅支持语音、24 kbit/s 高比特率、RTF 1.7(不可实时)、需 60 DNN evaluations [§1]

FlowDec 的目标: **去除对抗训练 + 推广到通用音频 + 低比特率 + 高效推理** [§1]。

### WHAT: 核心贡献

1. **Joint flow matching for signal enhancement** [§3.2]: 提出 shifted prior 的 CFM 变体,将 codec decoder 输出 y 作为 flow 起点而非纯噪声,消除 OT solver 需求 [§3.2, Eq.6]
2. **Non-adversarial codec (NDAC)** [§3.4]: 基于 DAC 去除所有对抗损失,加入 CQT loss + L1 waveform loss 改善低频建模 [§3.4]
3. **频率依赖噪声 sigma_y(f)** [§3.5]: 按 STFT 频带独立计算 sigma_y,适配自然信号的逆功率谱特性 [§3.5]
4. **全频段 48kHz 通用音频** [§4]: 从 speech 推广到 music + sound,比特率从 24 kbit/s 降至 4 kbit/s [§4.1, Table 1]

## 方法详解

### 1. 问题形式化 [§3, Eq.1-3]

给定 clean audio x* 和 codec encoder E,code c = E(x*) 属于 Z^l, l << L [§3, Eq.1]。

**核心观察** [§3]: Encoder E 是多对一映射(多个 x* 共享同一 c),因此 perfect decoder D 使得 D(E(x*)) = x* 不可能。**解决方案**: 将解码视为随机推理问题,目标是从 p_data(hat{x}|c) 采样 [§3]。

**Stochastic decoder** [§3, Eq.3]: D_s(c) = Omega(D_0(c))
- D_0: deterministic pre-trained decoder (NDAC)
- Omega: stochastic postfilter (flow matching model)
- 目标: 学习 p_Omega(hat{x}|y) 逼近 p_data(.|y),其中 y = D_0(c) [§3, Eq.3]

### 2. Flow Matching 背景 [§3.1]

标准 CFM (Lipman et al., 2023) [§3.1, Eq.4-5]:
- 学习 ODE: d/dt phi_t(x) = u_t(phi_t(x))
- 条件 FM loss: L_CFM = E[||v_theta(x,t) - u_t(x|x_1)||^2]

### 3. Joint Flow Matching for Signal Enhancement [§3.2]

**关键改进: Shifted prior** [§3.2, Eq.6]:

标准 CFM 中 x_0 ~ N(0, sigma^2 I) 独立于 x_1。FlowDec 改为:

p_t(x_t | x_1, y) = N(x_t; mu_t, sigma_t) = N(x_t; y + t(x_1 - y), (1-t)^2 Sigma_y)

即 x_0 ~ N(y, Sigma_y),以 decoder 输出 y 为中心 [§3.2, Eq.6]。

**WHY shifted prior 比 standard CFM 好** [§3.2, Fig.2]:
- 标准 CFM: x_0 ~ N(0, I),flow 需从纯噪声搬运到目标 — 路径长、方差大 [§3.2]
- Shifted prior: x_0 ~ N(y, Sigma_y),flow 从 decoder 输出出发 — 路径短、方差小 [§3.2]
- 当 sigma_y 足够大时,以 y 为中心的 Gaussian 之间无重叠,per-batch OT 自动满足,无需 OT solver [§3.2]
- 对比 SGMSE/FlowAVSE 的 constant sigma_t: FlowDec 的 linear sigma_t 使 flow field contractive,ODE 数值稳定且保证收敛 [§3.2, Fig.3] [论文原文]

**Conditional vector field** [§3.2, Eq.7]:

u_t(x | x_1, y) = (x_1 - x_t) / (1 - t)

**Joint FM loss** [§3.2, Eq.11]:

L_JFM = E[||v_theta(x_t, t, y) - (x* - (y + sigma_y * epsilon))||^2]

其中 sigma_t = 0.4(1-t),保证 t=1 时 sigma 为 0(contractive) [§3.2]。

### 4. Feature Domain [§3.2]

实际在 amplitude-compressed complex STFT 域操作 [§3.2]:
- Feature extractor Phi: 使用 Welker et al. (2022) 的 amplitude-compressed STFT
- Compression exponent alpha = 0.3 [§3.2]
- Y (STFT of decoder output) 通过 channel-wise concatenation 作为 v_theta 的条件 [§3.2]
- 推理: Midpoint solver, 3 steps → NFE = 6 [§3.2]

### 5. Non-adversarial Codec (NDAC) [§3.4, Table 2]

基于 DAC (Kumar et al., 2024) 移除所有对抗损失 [§3.4]:

| 变体 | fs | 比特率范围 | H (hop) | n_c | d_emb |
|------|------|---------|---------|-----|-------|
| DAC | 44.1 kHz | 0.86-7.75 | 512 | 9 | 1024 |
| **NDAC-75** | **48 kHz** | **0.75-7.50** | **640** | **10** | **1024** |
| **NDAC-25** | **48 kHz** | **0.25-4.00** | **1920** | **16** | **128** |

**WHY 48kHz** [§1]: 44.1kHz 的 feature rate 86.13Hz 和 bitrate 7751.95 bit/s 为非整数; 48kHz 的 75Hz 和 7500 bit/s 为整数(48000 有更简单的 divisor) [§1]。

**NDAC 的损失函数改进** [§3.4]:
- 移除: 所有 adversarial losses (multi-scale discriminator, multi-period discriminator) [§3.4]
- 新增: **Multiscale CQT loss** — 9 octaves, hop 256, min freq 27.5Hz, bins {16,32,48,64,80} per octave [§4.2]
  - WHY CQT: 纯 mel loss 导致低频(<2kHz) 建模差(SI-SDR 约 -30dB),CQT 的高低频分辨率适配音乐 [§3.4]
- 新增: **L1 waveform loss** (weight=50) — magnitude-only losses 对相位盲目 [§3.4]
- 保留: Multiscale Mel loss (amplitude + log-amplitude differences, 来自 DAC) [§3.4]

### 6. Frequency-dependent Noise Levels [§3.5]

**问题**: 自然信号功率谱遵循逆幂律,高频功率远低于低频。单一标量 sigma_y 会导致高频过度平滑 [§3.5]。

**解决方案**: 按 STFT frequency band 独立计算 sigma_y(f),使用启发式 quantile 方法 [§3.5]:
- 768-point frequency-dependent curves [§4.2]
- Gaussian kernel (bandwidth=3) 平滑 [§4.2]
- 基于 FlowDec-75s 训练 [§4.2]

### 7. Postfilter Architecture [§4.2, Appendix A.3]

基于 NCSN++ (Song et al., 2021) 的略微修改版 [§4.2]:
- 参数量: 26M [§4.2]
- 训练: Adam, lr=10^-4, 800K iterations [§4.2]
- 片段: 2 秒 [§4.2]
- 批大小: 64 [§4.2]
- EMA: decay=0.999 [§4.2]

## 数据集 [§4.1, Table 1]

**Codec 训练**: speech + music + sound 混合,type-balanced [§4.1]:

| 数据集 | 时长 | 类型 |
|-------|------|------|
| MSP-Podcast | 103h | Speech |
| CommonVoice 13.0 | 1602h | Speech |
| LibriTTS | 553h | Speech |
| EARS | 100h | Speech (48kHz) |
| VCTK 84spk | 20h | Speech (48kHz) |
| LibriVox | 55611h | Speech |
| Expresso | 20h | Speech (48kHz) |
| [InternalSpeech] | 1512h | Speech (48kHz) |
| [InternalMusic] | 18949h | Music |
| WavCaps-FreeSound* | 1582h | Sound |
| [InternalSound] | 5309h | Sound (48kHz) |

**Postfilter 训练**: 同上数据 + 100K clean files per type + 100K 10-second mixtures [§4.1]

**测试集**: 3000 samples (1000 per type): VCTK + EARS (speech), MUSDB18-HQ + MusicCaps (music), AudioSet (sound) [§4.1]

## 实验结果

### Objective Metrics [Fig.4, Table 8 (Appendix)]

FlowDec-75m (主模型) 在几乎所有比特率下:
- **FAD (感知)**: 显著优于 DAC-75 和 2xDAC-75 [Fig.4] [论文原文]
- **SI-SDR (失真)**: 低于 DAC — 这是 perception-distortion tradeoff 的体现 [§5.1, Fig.5]
- **fwSSNR**: 略低于 DAC,但差距小 [Fig.4]
- **SIGMOS (speech-only)**: 整体与 DAC 相当 [Fig.4]
- **logSpecMSE**: 低于 DAC [Fig.4]

**Perception-distortion tradeoff** [§5.1, Fig.5]: FlowDec 沿此 tradeoff 偏向感知端 (低 FAD, 较低 SI-SDR),DAC 偏向失真端。这与 score-based speech enhancement 和 JPEG artifact removal 的观察一致 [§5.1]。

### FlowDec vs ScoreDec vs FlowAVSE [Table 4]

| Method | FAD x100 | SI-SDR | fwSSNR |
|--------|----------|--------|--------|
| **FlowDec (NFE=6)** | **1.62** | 7.55 | **15.46** |
| ScoreDec (NFE=6) | 145.30 | -27.23 | 3.15 |
| sigma_t=0.05 (NFE=6) | 28.88 | 9.95 | 5.50 |
| sigma_t=0.66 (NFE=6) | 29.83 | **10.10** | 6.55 |
| **FlowDec (NFE=50)** | **1.34** | 7.41 | **15.65** |
| ScoreDec (NFE=50) | 5.73 | **7.50** | 14.45 |

**核心发现** [§5.1]:
- NFE=6 时 FlowDec FAD 1.62 vs ScoreDec 145.30 — ScoreDec 在低 NFE 下完全失败 [Table 4] [论文原文]
- NFE=50 时 ScoreDec 追上,但 FlowDec 仍在 FAD 上领先 [Table 4]
- Constant sigma_t (FlowAVSE 方案) 在两种 NFE 下均不如 FlowDec [Table 4]

### Subjective Listening Tests [Fig.6, §5.2]

**Test A** (75Hz models, MUSHRA): FlowDec-75m/75s 与 DAC-75 评分无显著差异 @ 同比特率 [Fig.6]
**Test B** (25Hz models, MUSHRA): FlowDec-25s 与 DAC-25 和 25Hz equivalents 评分相当 [Fig.6]

[agent解读] 主观评估证实 FlowDec 在不使用对抗训练的情况下,感知质量与 GAN-based DAC 持平。这是一个重要结论 — 表明 flow-based postfilter 可完全替代复杂 GAN 训练。

### Real-Time Factor [§5.3]

| Component | RTF |
|-----------|-----|
| NDAC-75 (codec only) | 0.0134 |
| NDAC-25 (codec only) | 0.0084 |
| Postfilter per NFE | ~0.0358 |
| **FlowDec-75m (total, NFE=6)** | **0.2285** |
| **FlowDec-25s (total, NFE=6)** | **0.2235** |
| ScoreDec (60 NFE) | 1.707 |

[agent解读] FlowDec RTF 0.23 (在 A100 上) 实现了亚实时推理,比 ScoreDec 的 RTF 1.707 快约 7.5x。但仍非真正实时 (RTF < 0.1),限制了流式应用。

## 关键设计选择与证据

| 设计选择 | WHY | 证据 |
|---------|-----|------|
| Shifted prior q0=N(y, Sigma_y) | Decoder 输出已是好的近似,只需 refine | NFE=6 时 FAD 1.62 vs ScoreDec 145 [Table 4] |
| Linear sigma_t = 0.4(1-t) | 保证 flow field contractive → 数值稳定 | Fig.3 对比: 轨迹更直,方差更小 |
| 非对抗 NDAC + postfilter 解耦 | 避免 GAN 训练复杂性 + 可独立优化 | MUSHRA 与 DAC 持平 [Fig.6] |
| CQT loss | 改善低频重建 | 无 CQT 时 SI-SDR ~-30dB [§3.4] |
| 频率依赖 sigma_y(f) | 适配自然信号频谱 | Appendix A.7.4 消融 |
| 48kHz 全频段 | 避免高频截断导致保真度损失 | 比 44.1kHz 更简单的整数 feature rates [§1] |
| STFT 域操作 | 复数频谱保留相位信息 | 源自 SGMSE (Welker 2022) 的成功经验 [§3.2] |

## 与已有方法的差异

| 维度 | FlowDec | DAC | EnCodec | ScoreDec |
|------|---------|-----|---------|----------|
| 训练方式 | Non-adversarial + flow postfilter | Adversarial (multi-D) | Adversarial | Non-adversarial + score postfilter |
| 采样率 | 48 kHz | 44.1 kHz | 24/48 kHz | 48 kHz (speech only) |
| 音频类型 | General (speech+music+sound) | General | General | Speech only |
| 比特率范围 | 0.75-7.50 (NDAC-75) | 0.86-7.75 | 1.5-24 | 24 |
| Postfilter | Flow matching (6 NFE) | 无 | 无 | Score-based (60 NFE) |
| RTF | 0.23 | ~0.01 | ~0.01 | 1.71 |
| FAD | 1.62 | 3.0-10.0 | - | 5.73 (NFE=50) |
| 开源 | 是 | 是 | 是 | 是 |

## 局限性分析

1. **非因果架构**: NCSN++ 和 NDAC 均非因果,不支持 streaming [§6] [论文原文]
2. **推理速度**: RTF 0.23 优于 ScoreDec 但仍非真正实时,不适合实时通信 [§5.3]
3. **感知-失真权衡**: FAD 优但 SI-SDR 不如 DAC,某些要求低失真的场景可能不适合 [§5.1]
4. **训练数据**: 使用了内部数据集 (InternalSpeech/Music/Sound),完全复现需替换 [§4.1]
5. **模型解耦但未联合训练**: Decoder 和 postfilter 分开训练,联合训练可能进一步提升但 "may lead to unstable training" [§6] [论文原文]

---

检索命中: [[ConditionalFlowMatching]], [[ResidualVectorQuantization]], [[NeuralVocoder]] | 过滤: [[ScoreMatching]](pending-review), [[Diffusion-basedVocoder]](pending-review) | 未命中但可能相关: 无
