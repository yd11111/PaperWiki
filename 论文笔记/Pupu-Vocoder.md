---
type: paper
tier: deep
title: "Aliasing-Free Neural Audio Synthesis"
arxiv_id: "2512.20211"
source: "Sources/Pupu-Vocoder.pdf"
authors: [Yicheng Gu, Junan Zhang, Chaoren Wang, Jerry Li, Zhizheng Wu, Lauri Juvela]
year: 2025
venue: "IEEE/ACM TASLP (submitted)"
tags: [vocoder, neural-codec, anti-aliasing, GAN-based, DDSP, waveform-generation, singing-voice, audio-synthesis]
concepts: ["[[NeuralVocoder]]", "[[SnakeActivation]]", "[[Multi-scaleSTFTDiscriminator]]", "[[ResidualVectorQuantization]]"]
models: ["[[BigVGAN]]", "[[EnCodec]]"]
tasks: ["[[NeuralAudioCompression]]"]
datasets: ["[[DAPS]]", "[[MUSDB]]"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[NeuralVocoder]], [[ResidualVectorQuantization]], [[Multi-scaleSTFTDiscriminator]], [[EnCodec]] + 2 个待确认页: [[SnakeActivation]], [[BigVGAN]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文处于 GAN-based neural vocoder 的最新进展分支,直接构建于 BigVGAN (Lee et al., 2023) 和 DAC (Kumar et al., NeurIPS 2023) 之上。BigVGAN 引入 Snake activation 和 anti-aliased representation 到 vocoder,但其 anti-aliasing 措施被本文作者认为不够彻底 — 仅用 oversampling 降低而非消除 aliasing。DAC 将 BigVGAN 的 decoder 架构 (含 Snake + RVQ) 扩展为 universal audio codec,使用 factorized codes 和 multi-band STFT discriminator。本文作者的新贡献是从 DSP 信号处理角度系统性地消除 activation 和 upsampling 两个环节的 aliasing。
>
> **已有认知**: Snake activation (SnakeBeta 变体) 通过周期性 inductive bias 让网络更容易生成周期波形,但其非线性特性本身会产生超过 Nyquist 频率的谐波(即 "folded-back" aliasing)[待确认]。ConvTranspose 上采样层会引入 "mirrored" aliasing 和 "tonal artifact"。现有 anti-aliasing 方案(BigVGAN 的 oversampling、linear/nearest interpolation 替代 ConvTranspose)要么计算开销大、要么引入 "filter artifact"。
>
> **创新判断**: ADAA (Anti-Derivative Anti-Aliasing) 技术来自数字音频 waveshaping 领域,此前未被系统引入 neural vocoder/codec。本文的核心创新在于: (1) 推导 SnakeBeta 的 closed-form ADAA,实现无需 oversampling 即达到同等 anti-aliasing 效果; (2) 用 resampling layer (zero-interlacing + Kaiser LPF) 替代 ConvTranspose,同时引入 deterministic noise prior 填充高频; (3) 将两者组合,在 vocoder 和 codec 两个场景验证。
>
> 检索命中: [[NeuralVocoder]], [[SnakeActivation]][待确认], [[BigVGAN]][待确认], [[ResidualVectorQuantization]], [[Multi-scaleSTFTDiscriminator]], [[EnCodec]] | 过滤: [[SingingVoiceSynthesis]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 从信号处理角度系统解决 neural vocoder/codec 中激活函数和上采样层的 aliasing 问题,提出 ADAA SnakeBeta + resampling upsampling;vocoder 在歌声/音乐上显著超越 BigVGAN,codec 以更少参数(119M vs 154M)与 DAC 持平或略优
> - **路线**: Mel/Waveform → Encoder → RVQ(codec)/直接输入(vocoder) → Decoder(AF Conv Blocks: ADAA SnakeBeta activation + resampling upsampling + deterministic noise prior) → Waveform
> - **指标**: 歌声 MUSHRA: Pupu-Vocoder_large 70.84 vs BigVGAN_large 53.08 [Table II]; 音乐 MUSHRA: 56.42 vs 50.42 [Table III]; AHR activation -45.95 dB vs LeakyReLU -25.25 dB [Table I]
> - **可借鉴**: ADAA 技术可迁移到任何使用非线性激活的上采样网络; deterministic noise prior 解决重采样训练不稳定; SnakeBeta 的 closed-form ADAA 消除 threshold fallback
> - **局限**: CPU RTF 较高(oversampling 的 DDSP 计算); 极低比特率 codec 性能收敛; 音乐合成整体质量仍有提升空间

## 核心问题

本文要解决的核心问题是: **为什么当前 neural vocoder 和 codec 在歌声、音乐等高保真场景下合成质量显著下降?**

作者将原因归结为两类 aliasing artifacts [§I, §II]:
1. **"Folded-back" aliasing**: 非线性激活函数 (如 LeakyReLU, SnakeBeta) 在离散信号上产生无穷多谐波,超过 Nyquist 频率的部分被"折回"低频,表现为频谱污染 [§II-A, Eq.1]
2. **"Mirrored" aliasing + tonal artifact**: ConvTranspose 上采样层通过 zero-interlacing 将信号扩展到高采样率,但其固有的频谱复制和周期性卷积导致镜像频谱伪影和 DC bias 引起的恒定频率响声 [§II-B]

这些 artifacts 在语音合成中不太明显(语音频率范围有限),但在歌声和音乐中严重可闻,因为它们恰好落在高频谐波区域。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Pupu-Codec 架构 [Fig 4]: Encoder → RVQ → Decoder + 4 个 discriminators (MPD, MSD, MBD, MS-SB-CQTD)。"AF Conv Blocks" 是将 BigVGAN/DAC 的卷积块中的激活函数和上采样层替换为 anti-aliased 版本得到的 [§IV-A]。

Pupu-Vocoder 则将输入从 waveform 改为 mel spectrogram,去掉 encoder 和 RVQ 模块 [§IV-A]。

### 关键设计选择

#### 1. Anti-Aliased Activation: ADAA SnakeBeta [§III-A]

**问题**: 任何非线性激活应用于离散信号都会产生超 Nyquist 谐波。BigVGAN 的方案是 oversampling (先上采样 → 激活 → 下采样),但 4x oversampling 才能有效,显存代价大 [§II-A]。

**ADAA 的核心思路** [论文原文]: 将离散信号通过线性插值重建为连续信号,对连续信号应用激活函数(连续域无采样率约束,因此无 aliasing),再通过低通滤波+离散化回到离散域 [Eq.3-7]。

**关键推导**: 对于矩形核滤波器,整个过程可化简为激活函数一阶反导数的差分 [Eq.7]:

$$y_t = \frac{F(x_t) - F(x_{t-1})}{x_t - x_{t-1}}$$

其中 $F(\cdot)$ 是 $f(\cdot)$ 的一阶反导数。

**SnakeBeta 的选择** [论文原文]: 作者特意选择 SnakeBeta ($f(x) = x + \frac{\sin^2(\alpha x)}{\beta}$) 而非原始 Snake,因为 SnakeBeta 的 ADAA closed-form [Eq.15] 消除了分母中的 $\Delta_x$,无需 threshold-based fallback mechanism [Eq.8],确保数值稳定 [§III-A]:

$$y_t = \frac{1}{2\beta} + \frac{\Sigma_x}{2} - \frac{\cos(\alpha\Sigma_x)\text{sinc}(\alpha\Delta_x)}{2\beta}$$

其中 $\Sigma_x = x_t + x_{t-1}$, $\Delta_x = x_t - x_{t-1}$。

**梯度稳定性分析** [§III-A]: 偏导数的值域为 $[\frac{\beta-\alpha}{2\beta}, \frac{\beta+\alpha}{2\beta}]$ [Eq.18]。$\beta \to 0$ 时有梯度爆炸风险,作者通过在 log-domain 参数化 $\beta$ 来强制正下界。$\alpha \gg \beta$ 时,正则化隐式阻止 — 过大 $\alpha$ 会产生大量无意义谐波,重建和对抗损失会自动惩罚 [论文原文]。

**最终方案**: oversampling (2x) + ADAA SnakeBeta 组合使用,在 anti-aliasing 效果和计算效率间取得平衡 [§IV-A]。

#### 2. Anti-Aliased Upsampling: Resampling + Deterministic Noise Prior [§III-B]

**问题**: ConvTranspose 的 zero-interlacing 本质上是频域展宽 + 频谱复制 [48],引入 "mirrored" aliasing 和 "tonal artifact" [§II-B]。Linear/nearest interpolation 替代方案虽消除 tonal artifact,但其等效滤波器频率响应差(通带衰减 + 阻带泄漏),引入 "filter artifact" [Fig 3]。

**方案** [论文原文]:
- **Resampling layer**: zero-interlacing + Kaiser window truncated sinc LPF (n=16),频率响应远优于 linear/nearest interpolation [Fig 5a]
- **Deterministic noise prior**: 对 decoder 第一层 Conv1D 输出 $x_0$ 做 zero-interlacing 后用 HPF 提取高频分量,作为确定性噪声先验填充上采样信号的空高频区域 [Fig 2, §III-B]
- **Channel expansion Conv1D**: 将 LPF 输出 + HPF 输出拼接后通过 Conv1D 生成最终全带信号

**HPF 设计** [论文原文]: 不直接设计高通滤波器,而是从原始信号减去低通滤波结果得到互补 HPF [§III-B, Eq.19],保证能量守恒。

**训练稳定性** [agent 解读]: deterministic noise prior 的引入解决了纯 LPF resampling 带来的训练不稳定 — 如果高频区域全是零,discriminator 会轻易区分真假信号,导致 GAN 训练崩塌。noise prior 给 generator 一个合理的高频起点。

### 训练策略

**Pupu-Codec 损失函数** [§IV-B, Eq.20]:
$$\mathcal{L}_{generator} = 15\mathcal{L}_{multi\text{-}mel} + 0.25\mathcal{L}_{commit} + \mathcal{L}_{code} + \sum_{m=1}^{M}[\mathcal{L}_{adv} + 2\mathcal{L}_{feat}]$$

训练方案沿用 DAC [23],使用 4 个 discriminator: MPD, MSD, MBD, MS-SB-CQTD [§IV-A]。

**配置** [§V-A]:
- Pupu-Vocoder: BigVGAN 修改版,upsampling ratios [8,8,2,2,2],kernel sizes [16,16,4,4,4],ADAA oversampling factor 2
- Pupu-Codec: DAC 修改版,encoder/decoder ratios [2,2,2,8,8],RVQ 86 Hz frame rate, 8 kbps max bitrate
- 训练: 8x H200 GPU, AdamW ($\beta_1=0.8, \beta_2=0.99$), lr=1e-4, 1M steps

**数据** [§V-A]: 4 个领域 — speech (1661h, 多语言), singing voice (885h), music (2343h), audio (1811h)。评估分 academic (公开数据集) 和 industrial (专业录制) 两套。

## 实验

### Test Signal Benchmark (AHR 指标) [Table I]

| 模块 | 方案 | AHR (dB) ↓ | 出处 |
| --- | --- | --- | --- |
| Activation | LeakyReLU | -25.25 | [Table I] |
| Activation | SnakeBeta (无 oversampling) | -39.63 | [Table I] |
| Activation | SnakeBeta (O=4) | -42.43 | [Table I] |
| Activation | ADAA SnakeBeta (无 oversampling) | -42.29 | [Table I] |
| Activation | **Ours (ADAA SnakeBeta + O=2)** | **-45.95** | [Table I] |
| Upsampling | ConvTranspose | -21.67 | [Table I] |
| Upsampling | Linear Interpolation | -48.11 | [Table I] |
| Upsampling | **Ours (Resampling + noise prior)** | **-53.93** | [Table I] |

ADAA SnakeBeta 不用 oversampling 就能达到 SnakeBeta O=2 的效果;加上 O=2 则达到最低 aliasing [论文原文, §V-C-1]。

### Speech & Singing Voice [Table II]

| 系统 | #Param | Speech MUSHRA (Ind.) | Singing MUSHRA (Ind.) | 出处 |
| --- | --- | --- | --- | --- |
| Ground Truth | / | 89.73 | 89.30 | [Table II] |
| Vocos (TF domain) | 14M | 66.41 | 42.92 | [Table II] |
| HiFi-GAN | 14M | 67.36 | 57.68 | [Table II] |
| BigVGAN_large | 122M | 68.23 | 53.08 | [Table II] |
| Pupu-Vocoder_small | 14M | 74.27 | 65.73 | [Table II] |
| **Pupu-Vocoder_large** | **122M** | **76.36** | **70.84** | [Table II] |
| EnCodec | 59M | 69.00 | 57.16 | [Table II] |
| DAC | 154M | 88.14 | 85.43 | [Table II] |
| Pupu-Codec_small | 32M | 77.68 | 78.97 | [Table II] |
| **Pupu-Codec_large** | **119M** | **86.51** | **85.65** | [Table II] |

### Music & Audio [Table III]

| 系统 | Music MUSHRA (Ind.) | Audio MUSHRA (Ind.) | 出处 |
| --- | --- | --- | --- |
| BigVGAN_large | 50.42 | 73.17 | [Table III] |
| **Pupu-Vocoder_large** | **56.42** | **73.47** | [Table III] |
| DAC | 72.65 | 70.56 | [Table III] |
| BigCodec | 73.09 | 73.75 | [Table III] |
| **Pupu-Codec_large** | **74.39** | **75.00** | [Table III] |

### Dynamic Bitrate (Singing Voice) [Table IV]

| Bitrate | Pupu-Codec_large MUSHRA | DAC MUSHRA | BigCodec MUSHRA | 出处 |
| --- | --- | --- | --- | --- |
| 8 kbps | 82.64 | 81.39 | 81.14 | [Table IV] |
| 5.33 kbps | 81.56 | 79.22 | 74.25 | [Table IV] |
| 2.67 kbps | 73.97 | 72.68 | 70.43 | [Table IV] |
| 1.78 kbps | 65.17 | 67.53 | 55.00 | [Table IV] |

高/中比特率优势明显;1.78 kbps 时 DAC 反超,作者推测是超低比特率下信息瓶颈限制了高频重建(恰好是本文方法最擅长的部分) [§V-C-3]。

### Ablation (Singing Voice) [Table V]

| 消融项 | C-MOS ↑ | 出处 |
| --- | --- | --- |
| w/o Oversampling | -0.38 | [Table V] |
| Ours → LeakyReLU | -1.45 | [Table V] |
| Ours → ELU | -0.93 | [Table V] |
| Ours → SnakeBeta | -0.52 | [Table V] |
| w/o Deterministic Prior | -1.49 | [Table V] |
| Ours → ConvTranspose | -0.15 | [Table V] |
| Ours → Linear Interp | -0.01 | [Table V] |
| Ours → Nearest Interp | -0.04 | [Table V] |

两个最关键组件: (1) deterministic noise prior 去掉后 C-MOS -1.49,是最大的单项损失; (2) 激活函数替换为 LeakyReLU 后 C-MOS -1.45。上采样层的替换影响相对较小(ConvTranspose -0.15),说明 noise prior 是上采样模块的关键而非滤波器本身 [agent 解读]。

## 局限性

1. **CPU 计算开销高**: oversampling + DDSP 操作导致 CPU RTF 约 10-18x (GPU RTF 约 0.01-0.03x 仍可实时),不适合纯 CPU 部署场景 [Table II, §V-C-2]
2. **极低比特率性能收敛**: 1.78 kbps 下 Pupu-Codec 不如 DAC,高频重建优势在信息极度受限时无法发挥 [Table IV, §V-C-3]
3. **音乐合成整体质量有限**: 即使最好的模型 MUSHRA 也仅约 56-75 分,多声部复调音乐的谐波结构仍难以完美重建 [Table III, §VII]
4. **未与 F0 条件化模型对比**: 实验中所有模型均去除了 F0 conditioning,虽保证了公平性,但未展示 F0 条件化模型在歌声上的真实上限 [§V-A-4]
5. **评估局限**: Sheet MOS predictor 仅限 16 kHz,无法评估全带 44.1 kHz 音质;客观指标和主观感知可能不一致 [§V-B-1]

## 点评

**优势**:
- 从信号处理第一性原理出发,系统性地解决 neural audio synthesis 的 aliasing 问题,理论分析扎实(ADAA 推导、梯度稳定性分析、滤波器频率响应对比)
- SnakeBeta 的 ADAA closed-form 推导消除了 threshold fallback,是一个干净的数学贡献
- Deterministic noise prior 是一个巧妙的工程洞见 — 不仅解决高频空白问题,还大幅稳定训练
- 实验覆盖 4 个领域(speech, singing, music, audio),多种评估指标,有 academic 和 industrial 两套测试集,rigor 较高
- 开源代码和预训练 checkpoint

**不足**:
- "Anti-aliasing" 并非全新概念 — BigVGAN 已有 oversampling anti-aliasing,Wavehax [34] 已提出 aliasing-free vocoder(但走 TF 域路线)。本文的增量是"更好地"做 anti-aliasing,而非首次解决
- CPU 开销问题未在正文给出解决方案(仅在 Future Works 中提及 higher-order ADAA),对实际部署有影响
- 消融实验仅在 singing voice 上做,speech/music/audio 的消融缺失

## 可复用的 idea

1. **ADAA 技术框架**: 任何使用非线性激活的生成模型(图像、视频)都可引入 ADAA,只需推导激活函数的反导数 closed-form。选择 activation 时应考虑 ADAA 友好性(分母是否可消除)
2. **Deterministic noise prior**: 在任何上采样场景中,用原始信号的高频分量作为 prior 填充新增频段,比纯零填充或随机噪声更稳定。可推广到 super-resolution 等任务
3. **Test signal benchmark**: 用合成正弦/锯齿/三角波量化 aliasing 程度(AHR 指标),比真实信号更可控、可解释。可作为 vocoder 设计的标准 ablation 工具
4. **SnakeBeta > Snake 的选择逻辑**: SnakeBeta 的分母 $\beta$ 使 ADAA closed-form 更优美,这种"为下游数学性质选择函数形式"的思路值得学习

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节 WHY 充分,速查可借鉴具体 |
> | 可信赖 | pass | 所有数字与原文交叉验证无误,标注覆盖率高 |
> | 可区分 | pass | 来源标注覆盖率 >90%,事实/推断分离清晰 |
> | 可定位 | pass | 谱系定位 BigVGAN→DAC→本文 清楚 |
> | 不污染 | pass | 待 KB 更新审阅 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1) — 均已修正
> 详见 `_review/Pupu-Vocoder-review.yml`
