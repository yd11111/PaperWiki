---
type: concept
title: "Semantic vs Acoustic Tokens"
aliases: [语义 token 与声学 token, Semantic Tokens, Acoustic Tokens, Token Hierarchy, 语音 token 层级, Discrete Speech Features]
category: "representation"
tags: [speech-representation, tokenization, discrete-token, speech-LM, trade-off]
key_papers: ["GSLM (Lakhotia et al., 2021)", "AudioLM (Borsos et al., 2023)", "SpeechTokenizer (Zhang et al., 2024)", "pGSLM (Kharitonov et al., 2022)", "SPIRIT-LM (Nguyen et al., 2024)", "Moshi (Defossez et al., 2024)"]
origin_paper: "Cui et al., Speech Language Models, 2024"
related_concepts: ["[[Speech Tokenizer]]", "[[Residual Vector Quantization]]", "[[Speech Language Model]]", "[[Codec Language Model]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Semantic tokens 和 Acoustic tokens 是 Speech Language Model 中两类根本不同的语音离散表征,分别侧重于内容语义和声学保真度。这一二分法是 SpeechLM 设计中最核心的 trade-off 之一。

### Semantic Tokens (语义 token)
由语义理解目标训练的 tokenizer 产生,旨在捕获语音的内容和含义:
- **训练目标**: Self-supervised learning (masked prediction, contrastive learning)
- **代表**: HuBERT (k-means on hidden states), w2v-BERT, wav2vec 2.0
- **特点**: 与文本对齐良好,语义连贯性强,但缺乏高频声学细节 (pitch, timbre 等)
- **形式**: encoder 输出经 k-means 聚类离散化: s = d(MFCC(a); θ_d)

### Acoustic Tokens (声学 token)
由声学生成目标训练的 tokenizer 产生,旨在保留高保真语音重建所需的声学特征:
- **训练目标**: Speech reconstruction / resynthesis
- **代表**: EnCodec, SoundStream (均使用 RVQ)
- **特点**: 高保真音频重建,但语义对齐差,序列长度大
- **形式**: 多级 RVQ: s = (d_1(v), d_2(v - v̂_1), ..., d_R(v - Σv̂_r))

### Mixed Tokens (混合 token)
兼顾语义理解和声学生成的第三类,尝试平衡两者的优劣:
- **代表**: SpeechTokenizer (RVQ 第一层蒸馏 HuBERT, 后续层量化残差), Mimi (单 VQ 语义 + 额外 RVQ 声学)
- **特点**: 仍处于早期阶段,但在 Moshi/SpeechGPT-Gen 中展现前景

## 核心 Trade-off

| 维度 | Semantic Tokens | Acoustic Tokens |
|------|-----------------|-----------------|
| 语义连贯性 | 强 (与文本对齐好) | 弱 (纯声学重建) |
| 声学保真度 | 弱 (缺高频细节) | 强 (RVQ 多级量化) |
| 表现力/副语言 | 弱 (prosody/timbre 丢失) | 中 (保留但无语义结构) |
| 序列长度 | 短 (25-50 Hz) | 长 (多级 codebook 展开) |
| 下游 vocoder | 需 input-enhanced (CFM→HiFi-GAN) | 可 direct synthesis |
| 主要用途 | 语义理解+生成 (ASR, TTS) | 高保真重建 (codec) |

**Survey 核心发现**: "While semantic tokens align well with text and excel in producing semantically coherent speech, the generated speech often lacks acoustic details, such as high-frequency information. Recovering and enhancing these details typically requires post-processing, like a diffusion model, which significantly increases the model's latency."

## 层级建模方案

为解决单一 token 类型的局限,研究者提出两种层级策略:

### 策略一: 串联 (Concatenation)
将 semantic 和 acoustic tokens 拼入同一序列,先建模 semantic 再建模 acoustic:
- **AudioLM**: w2v-BERT semantic tokens → SoundStream acoustic tokens
- **优点**: 概念简单,分阶段建模
- **缺点**: 序列极长,建模复杂度高

### 策略二: 混合 (Mixed Tokenizer)
设计单一 tokenizer 同时编码语义和声学信息:
- **SpeechTokenizer**: RVQ 第一层蒸馏 HuBERT 的语义,后续层编码声学残差
- **Mimi**: 单 VQ 模块提取语义 + 额外 RVQ 提取声学
- **优点**: 统一框架,无需串联
- **缺点**: 设计复杂,仍在探索中

## Paralinguistic Tokens

Survey 特别指出第三类 "副语言 token",弥补 semantic tokens 的表现力缺陷:
- **pGSLM**: 在 HuBERT semantic tokens 基础上添加 F0 (基频) 和 unit duration 作为副语言 tokens,用 multi-stream transformer 分别预测
- **SPIRIT-LM**: 添加 pitch tokens 和 style tokens 补充 HuBERT semantic tokens
- 这些副语言 tokens 让 SpeechLM 在不牺牲语义的前提下捕获表现力

## 在 Speech LM 中的角色

Token 类型的选择直接决定 SpeechLM 的能力侧重:
- **多数 SpeechLM 选择 semantic tokens** (GSLM, TWIST, SpeechGPT, AudioPaLM, OmniFlatten, SLAM-Omni): 语义理解是口语交互的核心
- **Codec-focused 系统选择 acoustic tokens** (VioLA, Parrot): 侧重高保真生成
- **前沿系统转向 mixed tokens** (Moshi, SpeechGPT-Gen): 兼顾理解和保真度

## 关键论文

- GSLM (Lakhotia et al., 2021): 首次对比 3 种 semantic tokenizer (CPC, wav2vec2, HuBERT)
- AudioLM (Borsos et al., 2023): 提出 semantic → acoustic 层级生成框架
- SpeechTokenizer (Zhang et al., ICLR 2024): RVQ 第一层蒸馏 HuBERT 实现混合
- pGSLM (Kharitonov et al., 2022): 引入 paralinguistic tokens (F0, duration)
- SPIRIT-LM (Nguyen et al., 2024): pitch/style tokens 补充语义

## 相关概念

- [[Speech Tokenizer]]: 产生这两类 token 的模块
- [[Residual Vector Quantization]]: acoustic tokens 的核心量化方法
- [[Speech Language Model]]: 消费这些 token 的模型框架
- [[Codec Language Model]]: 专门建模 acoustic (codec) tokens 的 LM 范式

## 演进

Mel spectrogram (连续, 传统 TTS) → VQ-VAE acoustic tokens (2019) → HuBERT semantic tokens (2021) → semantic + acoustic 层级 (AudioLM, 2022) → paralinguistic tokens 补充 (pGSLM, 2022) → mixed tokenizer (SpeechTokenizer, 2024) → 统一框架 (Mimi/Moshi, 2024)
