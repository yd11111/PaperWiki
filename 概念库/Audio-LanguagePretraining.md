---
type: concept
title: "Audio-Language Pretraining"
aliases: [音频语言预训练, CLAP, Contrastive Language-Audio Pretraining, Audio-Language Model, ALM, 音频语言模型, Audio-Text Alignment, 音频文本对齐]
category: "technique"
tags: [audio-language, contrastive-learning, CLAP, pretraining, audio-text, retrieval, captioning, multimodal]
key_papers: ["[[论文笔记/Survey-AudioLanguageModels|Su et al. 2025 (ALM Survey)]]", "[[论文笔记/AudioMOSChallenge2025|AudioMOS Challenge 2025]]", "[[论文笔记/HD-PPT|HD-PPT]]", "[[论文笔记/SALMONN|SALMONN]]"]
origin_paper: "Elizalde et al., CLAP: Learning Audio Concepts from Natural Language Supervision, ICASSP 2023"
related_concepts: ["[[AudioUnderstanding]]", "[[SpeechLanguageModel]]", "[[Speech-TextAlignment]]", "[[SpeechTokenizer]]", "[[TTSEvaluation]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Audio-Language Pretraining 是在配对的音频-文本数据上训练模型,使其学习音频和自然语言之间跨模态对应关系的技术范式。与 Speech Language Model (端到端语音生成/理解) 不同,ALM 侧重于通过自然语言监督学习通用音频表征,覆盖语音、环境声、音乐等多种音频类型。

**与 Speech-Text Alignment 的区别**:
- Speech-Text Alignment: SpeechLM 内部对齐语音/文本 token (interleaved training, adapter)
- Audio-Language Pretraining: 独立的预训练范式,学习音频-文本联合表征空间 (CLAP 等)

**核心代表**: CLAP (Contrastive Language-Audio Pretraining) 类比视觉领域的 CLIP,用对称 infoNCE 对比损失学习音频-文本联合嵌入空间。

## ALM 架构分类 [Su et al. 2025, §III-A]

| 架构 | 结构 | 特点 | 代表模型 |
|------|------|------|----------|
| **Two Towers** | 独立 audio/text encoder + projector | Late interaction, 低推理延迟, 适合大规模检索 | MS-CLAP, LAION-CLAP |
| **Two Heads** | Audio encoder + Text encoder + LM | Intermediate fusion, LM 推理能力 | SALMONN, Audio Flamingo, GAMA |
| **One Head** | 单一 unimodal encoder | Early fusion, 理论高效但音频-语言应用有限 | — |
| **Cooperated Systems** | LLM agent 协调多模型 | 模块化组合, 能力组合最灵活 | AudioGPT, SpeechAgents |

## 预训练目标 [§III-B]

### 1. 对比目标 (Contrastive)

对称 audio-text infoNCE 损失,拉近配对 audio-text 嵌入,推远非配对样本:

$$\mathcal{L}_{con} = \frac{1}{2B}\sum_{i=1}^{B}(l_i^a + l_i^t)$$

其中 $l_i^a = -\log \frac{\exp(z_i^a \cdot z_i^t / \tau)}{\sum_j \exp(z_i^a \cdot z_j^t / \tau)}$

**关键挑战**: 有限 batch size 导致负样本池稀疏,引入 systematic gradient bias。解决方向: momentum encoders, large batch (FLAP: 4608), hard negative mining, debiased contrastive learning。

### 2. 生成目标 (Generative)

- **Masked spectrogram reconstruction**: 遮蔽音频频谱并重建,增强表征鲁棒性 (FLAP, M2D-CLAP)
- **Language modeling**: 自回归生成音频相关文本 (captioning objective), 增强 audio-language correlation
- **Audio generation**: 从文本描述生成音频 (AudioLDM, AudioGEN)

### 3. 判别目标 (Discriminative)

- **Audio-text matching**: 二分类判断 audio-text pair 是否匹配
- **Cross-entropy classification**: 音频分类 (sound event detection, emotion recognition)

## 关键预训练模型

| 模型 | Audio Encoder | Text Encoder | 目标 | 特点 |
|------|--------------|--------------|------|------|
| MS-CLAP | CNN14+BERT | BERT | Con | 首个 contrastive ALM |
| LAION-CLAP | HTSAT+RoBERTa | RoBERTa | Con | 大规模开源 (630K pairs) |
| MS-CLAP V2 | HTSAT-22+GPT-2 | GPT-2 | Con | 多任务 multi-task encoder |
| WavCaps | CNN14+RoBERTa | RoBERTa | Con | 400K generated captions |
| FLAP | HTSAT+Flan-T5 | Flan-T5 | Con | 超大 batch (4608) |
| COMPA | HTSAT+Flan-T5 | Flan-T5 | Con | Compositional reasoning + modular loss |
| MGA-CLAP | HTSAT/AST+BERT | BERT | Con | Multi-granularity alignment |
| T-CLAP | HTSAT+RoBERTa | RoBERTa | Con | Temporal-contrastive loss |
| MINT | Data2vec+Flan-T5 | Flan-T5 | Con+Gen+Dis | Multi-objective bridge-net |

## 下游任务

### 判别任务
- **Audio Classification (AC)**: 零样本或 fine-tuned 音频分类 (ESC-50, AudioSet)
- **Audio-Text Retrieval (ATR)**: 跨模态检索 (audio→text, text→audio)
- **Sound Event Detection (SED)**: 检测音频中的事件
- **Speech Emotion Recognition (SER)**: 语音情感识别

### 生成任务
- **Automated Audio Captioning (AAC)**: 描述音频内容的自然语言生成
- **Text-to-Audio (TTA)**: 文本条件音频生成 (AudioLDM, Diffsound)
- **Language-queried Audio Source Separation (LASS)**: 语言描述驱动的音源分离 (AudioSep)
- **Text-to-Spatial-Audio (TTSA)**: 文本驱动空间音频生成

### 理解/推理任务
- **Audio Question Answering (AQA)**: 基于音频回答问题
- **Audio Dialogue**: 多轮音频对话

## Large Audio-Language Models (LALMs)

在 CLAP 基础上进一步集成 LLM 的模型族 [§V-B]:

| 模型 | 架构 | 域 | 特点 |
|------|------|------|------|
| Pengi | CLAP+GPT-2 | Audio | 所有任务统一为 text generation |
| LTU | AST+LLaMA | Audio | Listen, Think, Understand curriculum |
| SALMONN | Whisper+BEATs+Vicuna | Audio | 3-stage training, activation tuning |
| Audio Flamingo | AF-CLAP+Qwen-2.5 | Audio | Few-shot ICL, multi-turn dialogue |
| GAMA | AST+Q-Former+LLaMA-2 | Audio | Multi-layer aggregator, CompA-R benchmark |
| SpeechGPT | HuBERT+LLaMA+HiFi-GAN | Speech | Cross-modal instruction following |
| Moshi | Mimi+Helium | Speech | Full-duplex real-time dialogue |
| SHANKS | — | Speech | Interleaved thinking tokens |
| STITCH | — | Speech | Concurrent thinking + speaking |

## 核心局限与挑战 [§VIII]

| 挑战 | 说明 |
|------|------|
| **Hallucination** | 生成源音频中不存在的内容; 结构化 QA 比 captioning 更严重 |
| **Adversarial vulnerability** | Jailbreak attacks 绕过 safety alignment |
| **Bias** | 语言偏见 (高资源语言主导), acoustic bias (voice timbre/gender) |
| **Privacy** | 端到端模型保留 voiceprint, 可推断 age/emotion/identity |
| **Training cost** | 大规模对比学习需巨量 GPU (FLAP: 72x64 GPUs) |
| **Data contamination** | 跨数据集音频重叠 (WavCaps/Clotho overlap) 导致评估不可靠 |

## 与 TTS 的关联

- CLAP 模型用于 TTS 评估: 计算合成音频与文本描述的对齐度
- Audio-text 表征用于 Predicted MOS 和 LLM-as-Judge 的底层特征
- Audio Flamingo 等 LALM 可作为自动化 TTS evaluator
- AudioSep (LASS) 可作为 TTS 前处理的音源分离工具

## 关键论文

- Elizalde et al. (2023): MS-CLAP — 首个 contrastive audio-language 模型
- Wu et al. (2023): LAION-CLAP — 大规模开源
- Ghosh et al. (2024): COMPA — compositional reasoning benchmark
- Deshmukh et al. (2023): Pengi — ALM 先驱
- Tang et al. (2024): SALMONN — multi-task LALM
- Ghosh et al. (2025): Audio Flamingo 3 — SOTA LALM
- Su et al. (2025): 首篇 ALM 系统综述

## 相关概念

- [[AudioUnderstanding]]: ALM 的下游能力体现 (captioning, QA, classification)
- [[SpeechLanguageModel]]: SpeechLM 是 speech-specific 端到端模型; ALM 更广 (general audio + text)
- [[Speech-TextAlignment]]: SpeechLM 内部模态对齐; 此处是独立的预训练范式
- [[SpeechTokenizer]]: ALM 中的 codec-based models 也使用 speech tokenizer
- [[TTSEvaluation]]: CLAP/LALM 越来越多地用作 TTS 评估工具

## 演进

Audio tagging (分类标签, 2010s) → AudioCaps / Clotho (人工 caption 数据集, 2019-2020) → MS-CLAP (首个 contrastive ALM, 2022) → LAION-CLAP (大规模开源, 2023) → Pengi/LTU (LLM 集成, 2023) → SALMONN/Audio Flamingo (LALM 多任务, 2024) → GAMA/Audio Flamingo 3 (推理+长音频, 2025) → 统一 evaluation ecosystem (2025-2026)
