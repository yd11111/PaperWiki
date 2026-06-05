---
type: concept
title: "Diffusion-based TTS"
aliases: [Diffusion TTS, 扩散语音合成, Diffusion Acoustic Model]
category: "model-family"
tags: [TTS, diffusion, acoustic-model, end-to-end, mel-generation]
key_papers: ["[[论文笔记/NaturalSpeech3|NaturalSpeech 3]]", "[[论文笔记/Survey-AudioDiffusionModels|Survey-Audio Diffusion Models]]", "[[论文笔记/NaturalSpeech2|NaturalSpeech 2]]", "[[论文笔记/MambaVoiceCloning|MambaVoiceCloning (2026)]]", "[[论文笔记/Chatterbox-Flash|Chatterbox-Flash]]", "[[论文笔记/SemaVoice|SemaVoice]]", "[[论文笔记/OmniVoice|OmniVoice]]", "[[论文笔记/DMOSpeech2|DMOSpeech 2]]", "[[论文笔记/DMOSpeech|DMOSpeech]]", "[[论文笔记/VoiceFlow|VoiceFlow]]", "[[论文笔记/SESD|SESD]]", "[[论文笔记/StableForm-TTS|StableForm-TTS]]", "[[论文笔记/Bridge-TTS|Bridge-TTS]]", "[[论文笔记/Daisy-TTS|Daisy-TTS]]", "[[论文笔记/DiffCSS|DiffCSS]]", "[[论文笔记/RapFlow-TTS|RapFlow-TTS]]", "[[论文笔记/ShallowFlowMatching|Shallow Flow Matching]]", "[[论文笔记/DLPO|DLPO]]", "[[论文笔记/MELA-TTS|MELA-TTS]]", "[[论文笔记/CTDiffusion|CTDiffusion]]"]
origin_paper: "Jeong et al., Diff-TTS: A Denoising Diffusion Model for Text-to-Speech, 2021"
related_concepts: ["[[DiffusionModel]]", "[[Diffusion-basedVocoder]]", "[[Non-autoregressiveTTS]]", "[[ConditionalFlowMatching]]", "[[NeuralVocoder]]", "[[Classifier-FreeGuidance]]", "[[DurationPredictor]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Diffusion-based TTS 是使用扩散模型作为声学模型 (acoustic model) 或端到端模型,将文本转换为语音的合成方法。相比 NAR 方法 (如 FastSpeech 2) 使用简单 L1/L2 loss 导致的 over-smoothing 问题,diffusion model 通过逐步去噪过程建模完整数据分布,生成更自然、更具表现力的语音。

Zhang et al. (2023) 综述将 diffusion TTS 分为两大框架 [§3, Table 1]:

1. **两阶段**: 声学模型生成 mel spectrogram + vocoder 生成波形 (主流)
2. **端到端**: 直接从文本生成波形

## 声学模型 (Acoustic Model)

### 开创性工作

#### Diff-TTS (Jeong et al., 2021)
首个将 DDPM 应用于 mel spectrogram 生成的工作 [§3.2.1]:
- 文本编码器提取上下文信息
- Length predictor + Duration predictor 对齐
- 使用 DiffWave 类架构的 decoder 以 DDPM 生成 mel spectrogram
- 使用 DDIM 加速采样

#### Grad-TTS (Popov et al., 2021)
基于 SDE 形式化的 diffusion TTS [§3.2.1]:
- 使用随机微分方程 (SDE) 而非 DDPM 的离散形式化
- 采用 WaveGrad 的 U-Net 作为 mel spectrogram 生成 decoder
- 支持端到端: 可将 mel decoder 替换为 waveform decoder
- **MOS**: 4.44 (LJSpeech, 与 Diff-TTS 的 4.337 相比) [Table 2]

### 高效加速

| 模型 | 加速方法 | 效果 | 出处 |
|------|----------|------|------|
| ProDiff (Huang et al., 2022) | Generator-based 参数化 + 知识蒸馏 | 首个可交互的实时 diffusion TTS | [§3.2.2] |
| DiffGAN-TTS (Liu et al., 2022) | 用预训练 GAN 作 generator + 浅 diffusion | **1 步生成**高质量音频 | [§3.2.2] |

**ProDiff 核心** [§3.2.2]: 传统 gradient-based 参数化需数百步迭代,ProDiff 改用直接预测 clean data 的 generator-based 参数化,并通过知识蒸馏将 N 步 teacher 压缩为 N/2 步 student。

**DiffGAN-TTS 核心** [§3.2.2]: 观察到 diffusion 步数多是因为 Gaussian 近似去噪分布需要小步长。用 GAN 学习更大步长的去噪分布,再在 GAN 粗预测基础上做浅 diffusion refinement,实现 1 步生成。

### 多说话人适配

| 模型 | 方法 | 特点 | 出处 |
|------|------|------|------|
| Grad-TTS with ILVR | 迭代隐变量采样 | 零样本说话人适配,无需训练 | [§3.2.3] |
| Grad-StyleSpeech | 参考语音编码为 style vector | 风格迁移 | [§3.2.3] |
| Guided-TTS (Kim et al., 2021) | 无条件 DDPM + phoneme classifier 引导 | 利用大规模无转录数据 | [§3.2.3] |
| Guided-TTS 2 (Kim et al., 2022) | Speaker-conditional DDPM + CFG | 零样本 + [[Classifier-FreeGuidance]] | [§3.2.3] |

### 离散隐空间

- **Diffsound** (Yang et al., 2022): 使用 VQ-VAE 离散化 mel spectrogram,diffusion 在离散 token 空间生成 [§3.2.4]
- **NoreSpeech** (Yang et al., 2022): VQ-VAE + diffusion 生成连续 style features,抗噪参考音频 [§3.2.4]

### 细粒度控制

- **EmoDiff** (Guo et al., 2022): 无条件 diffusion + 情感 classifier 引导,软标签控制情感强度 [§3.2.5]

## 端到端框架 (End-to-End)

直接从文本/音素生成波形,无需中间 mel spectrogram [§3.4]:

| 模型 | 基础 | 特点 |
|------|------|------|
| WaveGrad 2 (Chen et al., 2021) | WaveGrad | 音素 → 波形,集成 Tacotron 2 encoder + 非注意力 duration |
| CRASH (Rouard & Hadjeres, 2021) | SDE | 端到端鼓声合成,支持 class-mixing |
| FastDiff (Huang et al., 2022) | Diffusion | 高效端到端,时间感知位置编码 |
| DAG (Pascual et al., 2022) | SDE | 全频段音频 (fullband),encoder-decoder 架构 |
| Iton (Shi & Wu, 2022) | Ito SDE | 双 denoiser (mel + wave),两阶段训练 |

## 实验结果对比

Zhang et al. (2023) Table 2 总结了声学模型在不同数据集上的表现:

| 模型 | 数据集 | MOS | RTF | SMOS |
|------|--------|-----|-----|------|
| Diff-TTS | LJSpeech | 4.337 | 0.035 | - |
| Grad-TTS | LJSpeech | 4.44 | 0.012 | - |
| ProDiff | LJSpeech | 4.08 | 0.04 | - |
| NoreSpeech | LibriTTS | 4.11 | - | 4.14 |
| Guided-TTS 2 | LibriTTS | 4.25 | - | 3.51 |
| Grad-StyleSpeech | VCTK | 4.13 | - | 3.95 |

## 与其他 TTS 范式的对比

| 范式 | 代表 | One-to-many | 音质 | 速度 | 可控性 |
|------|------|-------------|------|------|--------|
| NAR (L1/L2) | FastSpeech 2 | Over-smoothing | 中 | 最快 | 显式 predictor |
| Flow-based | Glow-TTS, VITS | 可逆变换 | 中高 | 快 | 隐变量 |
| **Diffusion** | Grad-TTS, ProDiff | 逐步去噪 | **最高** | 慢 (可加速) | classifier guidance |
| GAN | GAN-TTS | 对抗训练 | 高 | 快 | 有限 |
| LLM-based | VALL-E | 自回归采样 | 高 | 慢 | in-context |
| Flow Matching | Voicebox, F5-TTS | 确定 ODE | 高 | **快** | CFG |

## 在 SVS 中的应用

Diffusion model 在 SVS 领域同样发挥重要作用 [Pan et al., 2026]:

**DiffSinger (Liu et al., 2022a)**: 将浅扩散 DDPM 引入歌声 mel spectrogram 生成,显著改善频谱细节,解决 Transformer 生成器的 over-smoothing 问题。成为级联 SVS 系统的标杆 [§2]。

**Diffusion F0 Predictor**: RMSSinger (He et al., 2023) 使用扩散模型预测 F0 轮廓,生成更自然、可控的 pitch 轨迹 [§2]。这是 diffusion 在 SVS 中的独特应用 — TTS 中无对应需求。

**Flow Matching for SVS**: TechSinger (Guo et al., 2025b) 采用 flow matching 作为声学模型的生成范式,在保持质量的同时实现更快、更稳定的生成 [§3.1]。

**ExpressiveSinger**: Dai et al. (2024) 使用级联 diffusion 控制模块增强歌声表现力。

## 关键论文

- Diff-TTS (Jeong et al., 2021): 首个 DDPM-based TTS [§3.2.1]
- Grad-TTS (Popov et al., 2021): SDE-based TTS,支持端到端 [§3.2.1]
- ProDiff (Huang et al., 2022): 知识蒸馏加速 [§3.2.2]
- DiffGAN-TTS (Liu et al., 2022): GAN 加速到 1 步 [§3.2.2]
- Guided-TTS 2 (Kim et al., 2022): CFG 零样本多说话人 [§3.2.3]
- WaveGrad 2 (Chen et al., 2021): 端到端 diffusion TTS [§3.4]

## 相关概念

- [[DiffusionModel]]: 底层生成模型框架
- [[Diffusion-basedVocoder]]: diffusion 在 vocoder 端的应用
- [[Non-autoregressiveTTS]]: diffusion TTS 属于 NAR 范式
- [[ConditionalFlowMatching]]: diffusion TTS 的演进方向,推理更快
- [[DurationPredictor]]: diffusion TTS 中的时长预测组件 (Diff-TTS, Grad-TTS)
- [[Classifier-FreeGuidance]]: Guided-TTS 2 使用的条件引导方法
- [[NeuralVocoder]]: 两阶段框架中 diffusion TTS 需配合的 vocoder

## 演进

Tacotron (AR attention, 2017) --> FastSpeech (NAR, 2019) --> Diff-TTS + Grad-TTS (diffusion 声学模型, 2021) --> ProDiff + DiffGAN-TTS (高效化, 2022) --> Guided-TTS 2 (零样本多说话人, 2022) --> **Flow Matching 取代 diffusion 成为主流** (Voicebox 2023, Matcha-TTS 2024, F5-TTS 2024) --> DMOSpeech (DMD2 蒸馏 + 端到端 metric 优化, 2024) --> Hybrid LLM + Flow (CosyVoice, 2024)
