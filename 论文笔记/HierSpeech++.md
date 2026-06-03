---
type: paper
tier: deep
title: "HierSpeech++: Bridging the Gap between Semantic and Acoustic Representation of Speech by Hierarchical Variational Inference for Zero-shot Speech Synthesis"
arxiv_id: "2311.12454"
source: "https://arxiv.org/abs/2311.12454"
authors: [Sang-Hoon Lee, Ha-Yeong Choi, Seung-Bin Kim, Seong-Whan Lee]
year: 2023
venue: "IEEE/ACM TASLP (under review)"
tags: [TTS, voice-conversion, zero-shot, hierarchical-VAE, speech-super-resolution, non-autoregressive, BigVGAN]
concepts: ["[[Variational Autoencoder for TTS]]", "[[Speech Factorization]]", "[[Semantic vs Acoustic Tokens]]", "[[F0 Modeling]]", "[[Neural Vocoder]]", "[[Speaker Embedding]]"]
models: ["[[模型库/VITS|VITS]]", "[[模型库/BigVGAN|BigVGAN]]", "[[模型库/HierSpeech++|HierSpeech++]]"]
tasks: [TTS, voice-conversion, speech-super-resolution]
datasets: [LibriTTS, VCTK, Libri-light, EXPRESSO, NIKL, MSSS]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Variational Autoencoder for TTS]], [[Speech Factorization]], [[Semantic vs Acoustic Tokens]], [[F0 Modeling]], [[Neural Vocoder]], [[Speaker Embedding]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Variational Autoencoder for TTS]]✓, [[Speech Factorization]]✓, [[Semantic vs Acoustic Tokens]]✓, [[F0 Modeling]]✓, [[Neural Vocoder]]✓, [[Speaker Embedding]]✓ | 过滤: [[Mel Spectrogram]](pending-review), [[Codec Language Model]](pending-review) | 未命中但可能相关: 无

- **VAE for TTS**: HierSpeech++ 基于 VITS 的 conditional VAE 范式,但引入 hierarchical VAE 桥接 semantic/acoustic gap。KB 已有 VITS 消融: 去掉 normalizing flow MOS 下降 1.52。本文扩展为层级 VAE + bidirectional Transformer flow。
- **Speech Factorization**: 本文通过 source-filter 理论将语义表示解耦为 speaker-agnostic 和 speaker-related 成分,使用 speech perturbation 去除说话人信息。与 Seed-TTS 的 self-distillation 路线类似但更早。
- **Semantic vs Acoustic Tokens**: 本文使用 MMS (Wav2Vec 2.0) 作为连续 semantic representation,不做 k-means 离散化。这与 HuBERT discrete token 路线不同,保留了更丰富的 acoustic 细节。
- **F0 Modeling**: 本文通过 YAPPT 提取 F0,quantize 为 log-scale 表示,作为 hierarchical synthesizer 的显式条件。TTV (Text-to-Vec) 显式预测 F0。
- **Neural Vocoder**: 本文改进 BigVGAN 的 AMP block 替换 HiFi-GAN 的 MRF block,引入 alias-free 上采样和周期性 Snake 激活函数。

> [!summary] 速查
> - **一句话**: 基于层级 VAE + bidirectional Transformer flow 的非自回归零样本 TTS/VC 框架,首次在零样本 TTS 和 VC 任务上达到人类水平质量 [论文原文]
> - **路线**: Text→TTV(semantic+F0)→Hierarchical Speech Synthesizer(层级 VAE+BiT-Flow+HAG)→16kHz 波形→SpeechSR→48kHz 波形 [Fig 1]
> - **指标**: TTS nMOS 4.56 (GT 4.32); VC nMOS 4.54; CER 0.90 (TTS), 二者均超 GT; SECS 0.899 (TTS, LT-960); SpeechSR PESQ 4.63 [Table 7, 5, 10]
> - **可借鉴**: (1) Style Prompt Replication (SPR) 技巧用于 1s 短 prompt 克隆; (2) DWT-based sub-band discriminator 提升高频重建; (3) Bidirectional Transformer Flow 解决 train-inference mismatch; (4) 无需文本标注的 hierarchical synthesizer 可利用海量无标注音频数据
> - **局限**: (1) 仍合成 noisy prompt 中的噪声; (2) 未做跨语言零样本; (3) SpeechSR 仅限 16→48 kHz; (4) 依赖 MMS 作为 semantic representation

## 核心问题

本文解决三个核心问题:

1. **Semantic-acoustic gap**: 自监督 semantic representation (如 HuBERT/MMS) 与丰富的 acoustic representation 之间存在信息鸿沟,导致直接从 semantic representation 生成语音时质量受限 [§1] [论文原文]
2. **Zero-shot adaptation 质量不足**: 现有 E2E 模型 (VITS 等) 在零样本场景下 speaker similarity 较低; LLM-based 模型虽 zero-shot 能力强但推理慢、鲁棒性差 [§1] [论文原文]
3. **低分辨率数据利用**: 大量 ASR 数据为 16 kHz,无法直接用于高质量 48 kHz 语音合成 [§3.4] [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

HierSpeech++ 由三个独立子系统组成 [Fig 1]:

1. **Hierarchical Speech Synthesizer**: 核心 — 从 semantic representation + style + F0 生成 16 kHz 波形 [§3.2]
2. **Text-to-Vec (TTV)**: 从 text 生成 semantic representation + F0,使 TTS 成为可能 [§3.3]
3. **SpeechSR**: 将 16 kHz 上采样至 48 kHz [§3.4]

**关键设计思想**: 三个子系统可独立训练且不需要文本标注 (除 TTV),大幅降低数据需求 [论文原文]。Hierarchical Synthesizer 和 SpeechSR 只需音频数据,TTV 只需 text-audio parallel 数据。

### 关键设计选择

#### 1. Source-Filter Multi-path Semantic Encoder [§3.2.2]

基于 source-filter 理论将 MMS semantic representation 分为两条路径 [Fig 3a]:
- **Speaker-agnostic**: 使用 speech perturbation 移除说话人信息,仅保留语言内容 [论文原文]
- **Speaker-related**: 保留原始说话人信息,并引入 F0 作为先验增强 pitch disentanglement [论文原文]

**WHY**: 直接从 text 预测 rich acoustic representation 是一个 one-to-many 问题,text 无法覆盖 speaker/style 信息。通过 source-filter 分解,语言模型只需预测 speaker-agnostic 内容,speaker 信息由 prompt 提供 [agent 解读]。

#### 2. Dual-audio Acoustic Encoder [§3.2.1]

为增强 acoustic representation 的容量 [Fig 3b]:
- **Wav encoder**: BigVGAN 的 AMP block 组成,直接从波形提取信息
- **Spec encoder**: 从 linear spectrogram 提取信息
- 两路 concatenate 后 projection → 更完整的 acoustic representation

**WHY**: HierSpeech 仅用 linear spectrogram 导致低 reconstruction quality (PESQ, periodicity, V/UV F1 等) [§3.2.1] [论文原文]。Linear spectrogram 难以覆盖所有 waveform 细节,额外的 wav encoder 补充了波相关和细粒度信息 [agent 解读]。

#### 3. Bidirectional Transformer Flow (BiT-Flow) [§3.2.5]

替换 VITS 中 WaveNet-based normalizing flow 为 Transformer-based:
- 4 个 residual coupling layers,每层含 preConv + 3 Transformer blocks + postConv [Fig 3c]
- AdaLN-Zero 注入 style 信息 (取代 WaveNet 的 condition)
- **Bidirectional**: 训练时双向运行 (posterior→prior + prior→posterior) 并施加 dropout [§3.2.5]

**WHY**: VITS 的 normalizing flow 只训练 forward (posterior→prior),推理只用 inverse (prior→posterior),导致 train-inference mismatch [§3.2.5] [论文原文]。BiT-Flow 双向训练可减少这种 mismatch,且 Transformer 比 WaveNet 有更强的 context 建模能力 [论文原文]。

#### 4. Hierarchical Adaptive Generator (HAG) [§3.2.4]

从 acoustic latent $z_a$、pitch $p_h$ 和 style $s$ 生成波形 [Fig 3d]:
- Source generator $G_s$: 从 F0 生成 pitch-refined 表示,使用 auxiliary F0 predictor 强化 ($L_{pitch}$ loss) [Eq. 2]
- Waveform generator $G_w$: 从 $z_a$, $p_h$, $s$ hierarchically 合成波形,使用 STFT mel reconstruction loss [Eq. 3]
- AMP blocks (BigVGAN) 替换 MRF blocks (HiFi-GAN),引入 anti-aliased Snake 激活函数 [§3.2.4]

**WHY**: BigVGAN 的 AMP block 包含 low-pass filter 和 periodic activation,对 out-of-distribution 生成更鲁棒 [论文原文]。Ablation 证实 AMP 改善所有指标 (UTMOS、CER、WER 等) [Table 4]。

#### 5. Unconditional Generation [§3.2.7]

训练时以 10% 概率将 style embedding 替换为 null embedding $\varnothing$,类似 classifier-free guidance [论文原文]。推理时只用 conditional generation。

**WHY**: 使 acoustic representation 采纳 speaker characteristics,改善整体 speaker adaptation [§3.2.7] [论文原文]。[agent 解读] 这类似于图像生成中的 CFG 训练策略,让模型学会在有/无 style 条件下都能生成。

#### 6. Text-to-Vec (TTV) [§3.3]

基于 VITS 框架的 text→semantic 模型 [Fig 4]:
- 替换 VITS 的 linear spectrogram 为 self-supervised speech representation
- VAE + MAS (monotonic alignment search) 对齐 text 和 semantic representation
- CTC phoneme predictor 增强 linguistic capacity [§3.3]
- T-Flow (Transformer-based normalizing flow + AdaLN-Zero) 适应 prosody
- 同时预测 4x 分辨率 F0 [§3.3]

**WHY**: 通过 TTV 将 text 映射到 semantic space,再由 Hierarchical Synthesizer 转为波形,实现完全并行的 TTS [论文原文]。Prosody style 可从 reference speech 分别迁移 [§3.1.3]。

#### 7. Style Prompt Replication (SPR) [§4.3]

针对 1s 短 prompt 场景的创新技巧:
- 将短 prompt 复制 n 次拼接为长 prompt 再输入 style encoder [Fig 7]
- Style encoder 通常需要 >3s prompt 才能稳定; SPR 通过 DNA-like replication "欺骗" encoder 使其认为 prompt 很长

**结果**: 1s prompt + SPR 的 UTMOS 从 2.67 提升到 4.07 [Table 9] [论文原文]

#### 8. SpeechSR [§3.4]

简单高效的 16 kHz → 48 kHz 超分辨率 [Fig 5]:
- 单 AMP block + NN upsampler (替代 transposed convolution 避免 tonal artifacts)
- DWT-based sub-band discriminator (DWTD): 将频谱分为 4 个 sub-band ([0-12], [12-24], [24-36], [36-48] kHz),各自独立判别,改善高频重建 [§3.4]
- 仅 0.13M 参数,742x 快于 AudioSR [§5.10]

### 训练策略

- Hierarchical Synthesizer: AdamW ($\beta_1=0.8, \beta_2=0.99$, weight decay $\lambda=0.01$); lr $1\times10^{-4}$, decay $0.999^{1/8}$; batch 80; 1,200k steps on 4 A6000 GPUs [§5.3]
- Adversarial training: MPD + MS-STFTD (least-square adversarial loss + feature matching loss + mel STFT reconstruction loss) [Eq. 4-5]
- BiT-Flow: $\lambda_{bi}=0.5$ for regularization (高 $\lambda$ 降低重建质量但提升 VC 性能) [§5.5]
- TTV: AdamW, lr $2\times10^{-4}$, batch 128; 950k steps on 4 A100 GPUs; 107M params [§5.3]
- SpeechSR: BigVGAN config, batch 128; 100k steps on 4 A6000 GPUs [§5.3]
- Total: HierSpeech++ 63M (+34M training-only), TTV 107M

## 实验

### TTS (Zero-shot, Table 7-8)

| 指标 | HierSpeech++ (LT-460) | HierSpeech++ (LT-960) | HierSpeech++ (Large) | VALL-E-X | XTTS | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nMOS | 4.56 | 4.55 | 4.50 | 3.50 | 3.32 | 4.32 | LibriTTS test-clean | [Table 7] |
| pMOS | 3.35 | 3.31 | 3.31 | 2.75 | 3.57 | 3.94 | LibriTTS test-clean | [Table 7] |
| sMOS | 3.70 | 3.74 | 3.72 | 3.27 | 3.50 | 3.88 | LibriTTS test-clean | [Table 7] |
| CER | 2.71 | 2.39 | 2.19 | 21.52 | 15.93 | 2.31 | LibriTTS test-clean | [Table 7] |
| WER | 4.59 | 4.20 | 3.87 | 29.33 | 18.97 | 4.13 | LibriTTS test-clean | [Table 7] |
| SECS | 0.899 | 0.907 | 0.911 | 0.865 | 0.788 | - | LibriTTS test-clean | [Table 7] |

### VC (Zero-shot, Table 5)

| 指标 | HierSpeech++ (LT-460) | HierSpeech++ (LT-960) | HierVST | DiffVC | YourTTS | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nMOS | 4.54 | 4.48 | 4.59 | 4.00 | 3.38 | 4.41 | VCTK | [Table 5] |
| sMOS | 3.63 | 3.62 | 3.35 | 3.08 | 3.13 | 3.89 | VCTK | [Table 5] |
| UTMOS | 4.19 | 4.13 | 4.19 | 3.49 | 3.09 | 4.04 | VCTK | [Table 5] |
| CER | 0.90 | 0.79 | 1.14 | 6.86 | 2.42 | 0.21 | VCTK | [Table 5] |
| SECS | 0.862 | 0.875 | 0.850 | 0.826 | 0.771 | - | VCTK | [Table 5] |

### Ablation (Table 2-4)

关键消融发现:
- **AMP Block**: 全面提升所有指标,UTMOS 最显著 [Table 4]
- **Source-Filter Encoder (SFE)**: 显著提升 F0 consistency [Table 4]
- **Dual-audio Encoder (DAE)**: 提升 reconstruction quality 但降低 VC 性能 (因 acoustic posterior 信息过多导致 speaker 泄露) [§5.5] [论文原文]
- **T-Flow**: 全面提升 speaker similarity (SECS) [§5.5]
- **Bi-Flow**: 轻微降低重建质量但提升 VC 性能 (regularization 减少 train-inference mismatch) [§5.5]
- **Large-scale data**: 无需标签即可 scale up,nMOS/sMOS 提升,CER/WER 降低 [Table 5]

### Additional Baselines (Table 11)

与 VALL-E, NaturalSpeech 2, StyleTTS 2 对比 (demo page samples):
- HierSpeech++ UTMOS 4.20, VALL-E 3.37, NaturalSpeech 2 3.79, StyleTTS 2 4.11 [Table 11]
- HierSpeech++ SECS (w. Prompt) 0.867/0.810, 远超其他模型 [Table 11]

## 局限性

1. **Noisy prompt 合成噪声**: 模型复制 prompt 中的背景噪声,style 解耦不完全 [§6] [论文原文]。使用 denoiser 可部分缓解 (UTMOS 从 4.12→4.25) 但 CER 恶化 [Table 12]
2. **跨语言未探索**: 未验证跨语言零样本能力 [§7] [论文原文]
3. **SpeechSR 局限**: 仅 16→48 kHz,其他 SR 模型可处理更低分辨率 (如 2→48 kHz) [§5.10]
4. **依赖外部 SSL 模型**: MMS 的质量直接影响 semantic representation [agent 解读]
5. **Style encoder 短 prompt 问题**: 需要 SPR trick 才能处理 <3s prompt [§4.3]

## 点评

**优势**:
- 非自回归 + 全并行,推理速度快且鲁棒 (无 repeat/skip/mispronunciation) [agent 解读]
- 不需要文本标注 (hierarchical synthesizer 和 SpeechSR) 的设计使数据 scalability 极强 [论文原文]
- 多个创新技巧 (SPR, DWT discriminator, BiT-Flow) 各自独立可复用
- 首次在零样本 TTS/VC 同时达到人类水平自然度 (nMOS > GT) [Table 7]

**不足**:
- 系统复杂度高: 3 个独立子系统 + 多个 encoder/decoder/discriminator
- DAE 训练时用到但推理不用,增加了训练资源消耗 [§5.5]
- 与 LLM-based TTS 相比,in-context learning 能力有限 (固定 style encoder 而非 in-context prompt) [agent 解读]

## 可复用的 idea

1. **Style Prompt Replication (SPR)**: 复制短 prompt 解决 style encoder 对长 prompt 的依赖,极其简单有效,可直接用于任何 reference encoder
2. **DWT-based sub-band discriminator**: 分频段判别提升高频合成质量,适用于任何 GAN-based vocoder/codec
3. **Bidirectional normalizing flow**: 训练时双向运行减少 train-inference mismatch,可用于任何 flow-based 生成模型
4. **Source-filter speech perturbation**: 使用 speech perturbation 创建 speaker-agnostic 表示,简单有效的解耦方法
5. **NN upsampler 替代 transposed convolution**: 减少 tonal artifacts 且更高效

检索命中: [[Variational Autoencoder for TTS]], [[Speech Factorization]], [[Semantic vs Acoustic Tokens]], [[F0 Modeling]], [[Neural Vocoder]], [[Speaker Embedding]] | 过滤: [[Mel Spectrogram]](pending-review), [[Codec Language Model]](pending-review) | 未命中但可能相关: 无
