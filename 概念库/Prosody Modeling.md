---
type: concept
title: "Prosody Modeling"
aliases: [韵律建模, 语音韵律, Prosody Control, Expressive TTS, 表现力语音合成]
category: "technique"
tags: [TTS, prosody, style, emotion, expressiveness, variation-information]
key_papers: ["[[论文笔记/Mega-TTS|Mega-TTS]]", "[[论文笔记/NaturalSpeech 2|NaturalSpeech 2]]", "[[论文笔记/Mega-TTS 2|Mega-TTS 2]]", "[[论文笔记/SC VALL-E|SC VALL-E]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Variational Autoencoder for TTS]]", "[[Speaker Embedding]]", "[[Attention-based TTS]]", "[[Speech Factorization]]", "[[Style Transfer in TTS]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Prosody Modeling 是 TTS 中对语音韵律信息(节奏、重音、语调、情感等)进行建模、控制和迁移的技术集合。韵律是决定合成语音自然度的关键因素,也是 TTS 中 one-to-many mapping 问题的核心来源。

**韵律的物理维度**:
- **Duration (时长)**: 音素/音节持续时间,决定语速和节奏
- **Pitch / F0 (基频)**: 声带振动频率,决定声调和语调
- **Energy / Loudness (能量)**: 音量变化,与重音和情感相关
- **Pause / Break (停顿)**: 短语和句子间的间隔

## Variation Information 分类 (Survey 框架)

Survey 将语音合成所需的信息分为四大类:

| 类别 | 描述 | 编码方式 |
|------|------|----------|
| Text content | 说什么 (what to say) | Text encoder |
| Speaker/Timbre | 谁来说 (who to say) | Speaker embedding |
| Prosody/Style/Emotion | 怎么说 (how to say) | Prosody modeling |
| Channel/Noise | 录制环境 | Denoising |

## 建模方法

### 1. 显式韵律信息 (Explicit)

直接提供可观测的韵律参数:

**方法**:
- Language/Speaker ID: 多语言/多说话人系统
- Pitch/Duration/Energy predictor: FastSpeech 2 的 variance adaptor
- Prosody tags: ToBI 标注、韵律边界标注

**代表工作**: FastSpeech 2, FastPitch

### 2. 隐式韵律信息 (Implicit)

从数据中自动学习不可直接标注的韵律变化:

#### Reference Encoder
- 从参考音频提取韵律向量,不需显式标注
- 训练时: 用 ground-truth 音频作参考
- 推理时: 用风格相似的参考音频

**代表工作**: GST-Tacotron (Wang et al., 2018), Ref-Tacotron (Skerry-Ryan et al., 2018)

#### VAE (变分自编码器)
- 将韵律变化编码到正则化隐空间
- 支持采样、插值、分离

**代表工作**: VAE-TTS, GMVAE-Tacotron, BVAE-TTS

#### 生成模型 (Flow/GAN/Diffusion)
- 用高级生成模型隐式建模韵律分布
- 缓解 one-to-many → over-smoothing 问题

**代表工作**: Glow-TTS, Flow-TTS, Multi-SpectroGAN, Grad-TTS

#### Text Pre-training
- 通过预训练学习隐含韵律的文本表示
- BERT/GPT style pre-training for text encoder

**代表工作**: PnG BERT, 自监督预训练

### 3. 信息粒度 (多层级)

| 粒度 | 描述 | 代表工作 |
|------|------|----------|
| Language/Speaker level | 语言/说话人整体特征 | Multi-speaker TTS |
| Paragraph level | 段落间连贯性 | Long-form reading |
| Utterance level | 整句风格/韵律 | GST-Tacotron, Ref-Tacotron |
| Word/Syllable level | 词级重音/节奏 | Sun et al. 2020 |
| Phoneme level | 音素级 duration/pitch | FastSpeech 2 |
| Frame level | 帧级精细控制 | Fine-grained VAE |

## 解耦、控制与迁移

### 解耦 (Disentangling)
将纠缠的韵律属性分离:
- **对抗训练**: gradient reversal 移除 speaker 信息 (Ma et al., Hsu et al.)
- **Bottleneck 重建**: SpeechFlow 用三个 bottleneck 分离 rhythm/pitch/content/timbre
- **帧级噪声建模**: 分离 noise 与 speaker (Zhang et al.)

### 控制 (Controlling)
- **Cycle consistency loss**: 鼓励合成语音保持目标风格
- **Style classifier feedback**: 分类器引导风格合成
- **半监督学习**: 部分标签时的 VAE 控制 (Habib et al.)

### 迁移 (Transferring)
- 从参考语音提取韵律 → 应用到新文本
- 从文本预测韵律特征
- 从隐空间采样

## 在可控 TTS 中的韵律控制 (Xie et al. 2024 Survey)

Survey 将 Prosody Control 定义为可控 TTS 的最基本任务,涵盖对 pitch, duration, energy 的操控,是实现自然度和表现力的关键。

### 控制策略 (Style Tagging for Prosody)

1. **离散标签控制**: StyleTagging-TTS 用短语/词汇直接指定韵律属性
2. **连续信号控制**: DiffStyleTTS 层级化建模 pitch/energy/duration/style 的 scale factors; Spark-TTS 通过专用 token 实现 pitch/speed 细粒度修改
3. **隐空间修改**: Cauliflow 通过 flow-based model 调整 latent 控制语速/停顿; DiTTo-TTS 通过 DiT 修改 latent length predictions
4. **韵律轮廓草图**: DrawSpeech (Chen et al., 2025) 让用户直接绘制韵律轮廓,由 diffusion model 细化为语音

### 在现代 LLM-TTS 中的韵律建模

传统 Prosody Modeling 技术在 LLM-TTS 时代的演变:
- **VALL-E / Seed-TTS**: 通过 in-context learning 从 prompt 音频隐式获取韵律风格
- **CosyVoice**: speech tokenizer 编码部分韵律信息 + CFM 还原声学细节
- **MaskGCT**: masked generative modeling 隐式学习韵律分布
- **LLM-based 局限**: 隐式建模使细粒度韵律控制困难 (Survey 指出这是关键挑战)

## 在 SVS 中的韵律建模

Singing Voice Synthesis 中的韵律建模与 TTS 有显著差异 [Pan et al., 2026]:

**Vibrato (颤音)**: 歌声特有的周期性音高波动 (频率 5-8 Hz, 幅度 0.5-2 半音)。Song et al. (2022) 提出 DL 模型控制 vibrato 的多个方面 (振幅、频率、起始延迟),这在 TTS 中不存在。数据增强策略中也包括向训练数据添加小 vibrato 以改善跨域泛化 [§A.1]。

**Musical Rhythm (音乐节奏)**: SVS 中时长由乐谱 BPM 和音符时值严格约束,而非数据驱动的自由预测。Duration predictor 需在乐谱约束下进行局部细化。

**Pitch 约束**: TTS 中 pitch 是柔性韵律维度,SVS 中 pitch 必须精确跟随乐谱 MIDI pitch,F0 Frame Error (FFE) 和 F0 RMSE 是核心评估指标。详见 [[F0 Modeling]]。

**歌唱技巧**: 除 vibrato 外,falsetto (假声)、breath (气息)、portamento (滑音) 等歌唱技巧也属于 SVS 韵律的扩展维度。SinTechSVS (Zhao et al., 2024) 和 TechSinger (Guo et al., 2025b) 专门建模这些技巧。

详见 [[Singing Voice Synthesis]]。

## 关键论文

- GST-Tacotron (Wang et al., ICML 2018): Global Style Tokens, reference encoder + style token bank
- Ref-Tacotron (Skerry-Ryan et al., ICML 2018): 定义 prosody 为 text/speaker 之外的剩余变化
- GMVAE-Tacotron (Hsu et al., ICLR 2019): GMM prior VAE 无监督风格聚类
- FastSpeech 2 (Ren et al., 2021): 显式 pitch + energy + duration predictor
- VITS (Kim et al., ICML 2021): flow-based prior 隐式捕捉韵律变化

## 相关概念

- [[Variational Autoencoder for TTS]]: 韵律的主要隐式建模方法
- [[Speaker Embedding]]: 与韵律共同决定 "how to say"
- [[Duration Predictor]]: 韵律的 duration 维度
- [[Speech Factorization]]: 将韵律与其他属性解耦

## 演进

规则韵律 (SPSS) → Prosody tags (ToBI) → Reference Encoder (GST, 2018) → VAE 隐式建模 (2019) → 显式 variance adaptor (FastSpeech 2, 2020) → 生成模型隐式建模 (VITS/Glow-TTS, 2020-21) → In-context learning (VALL-E, 2023; prompt 驱动)
