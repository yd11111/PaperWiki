---
type: concept
title: "Audio Understanding"
aliases: [音频理解, Speech Understanding, 语音理解, Audio Comprehension, Speech Comprehension via LLM]
category: "task-family"
tags: [speech-LM, understanding, ASR, emotion, speaker, paralinguistic, downstream]
key_papers: ["GSLM (Lakhotia et al., 2021)", "SpeechGPT (Zhang et al., 2023)", "AudioPaLM (Rubenstein et al., 2023)", "SPIRIT-LM (Nguyen et al., 2024)", "Moshi (Defossez et al., 2024)", "VITA (Fu et al., 2024)", "[[论文笔记/Survey-Audio Language Models|Su et al. 2025 (ALM Survey)]]", "[[论文笔记/ALLD|ALLD]]"]
origin_paper: "Cui et al., Speech Language Models, 2024"
related_concepts: ["[[Speech Language Model]]", "[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Full-duplex Spoken Dialogue]]", "[[Audio-Language Pretraining]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Audio Understanding 指 Speech Language Model 对语音和音频输入进行理解、分析和推理的能力。与传统单任务系统 (如 ASR-only 或 SER-only) 不同,SpeechLM 通过统一的基础模型同时处理多种理解任务,可根据指令灵活切换任务类型。

**与 TextLM 理解的关键区别**: SpeechLM 不仅理解语义内容 (what is said),还理解副语言信息 (how it is said) -- pitch, timbre, emotion, speaking rate 等,这些信息在 ASR+LLM 管线中完全丢失。

## 任务分类体系

Survey (Cui et al., 2024, Section V) 将 SpeechLM 的理解能力分为三大类:

### 语义相关理解 (Semantic-related)

| 任务 | 定义 | SpeechLM 方式 |
|------|------|---------------|
| ASR | 语音→文本转写 | 输入 speech + instruction, 输出 text |
| Spoken Dialogue | 理解口语并生成口语回复 | 端到端 speech-in-speech-out |
| Speech Translation | 跨语言语音翻译 | speech (源语言) → speech/text (目标语言) |
| Intent Classification | 识别说话者意图 | speech → intent label (text) |
| Slot Filling | 从语音中提取结构化信息 | speech → structured slots (text) |
| Keyword Spotting | 检测语音中特定关键词 | speech → detected keyword |
| QbE-STD | 通过语音示例搜索语音 | query speech + corpus → matches |

### 说话人相关理解 (Speaker-related)

| 任务 | 定义 | SpeechLM 方式 |
|------|------|---------------|
| Speaker Identification | 识别说话人身份 | speech → speaker label |
| Speaker Verification | 验证两段语音是否同一人 | speech pair → yes/no |
| Speaker Diarization | 标注 "谁在什么时候说话" | audio → timestamped speaker labels |

**SpeechLM 独特优势**: 可以在多人对话中隐式识别不同说话人,区分不同人的发言并分别回应。

### 副语言相关理解 (Paralinguistic)

| 任务 | 定义 | SpeechLM 方式 |
|------|------|---------------|
| Emotion Recognition | 识别语音中的情感 | speech → emotion label/description |
| Speech Separation | 从混合音频中分离各说话人 | mixed speech → separated speeches |

**SpeechLM vs TextLM**: TextLM 只能通过文本推断情感 (lexical cues),而 SpeechLM 可利用语音中的声学特征 (pitch contour, energy, speaking rate) 直接识别情感。

## 理解能力的来源

SpeechLM 的理解能力取决于三个因素:

### 1. Speech Tokenizer 的信息保留
- **Semantic tokens** (HuBERT): 保留语义和语言信息,适合 ASR/dialogue
- **Acoustic tokens** (EnCodec): 保留声学细节,适合 speaker/emotion 任务
- **Paralinguistic tokens** (pGSLM 的 F0/duration): 显式编码副语言特征

### 2. 训练数据的覆盖
- 预训练数据 (Table III): ASR 语料 (LibriSpeech, GigaSpeech), 播客 (Spotify Podcasts), 对话 (Fisher)
- 指令微调数据: instruction-following 数据集 (SpeechInstruct, InstructS2S-200K, VoiceAssistant-400K)

### 3. Instruction-tuning 的任务覆盖
SpeechGPT 和 SpeechGPT-Gen 的两阶段 instruction-tuning:
- Stage 1: cross-modal (ASR 数据 → speech→text) + (TTS 数据 → text→speech)
- Stage 2: chain-of-modality (text instruction → speech response)

## 评估基准

Survey (Table VI) 总结了评估 SpeechLM 理解能力的主要 benchmark:

| Benchmark | 评估类型 | 任务数 | 音频类型 |
|-----------|----------|--------|----------|
| SUPERB | Downstream | - | Speech |
| AudioBench | Downstream | 8 | Speech, Sound |
| AIR-Bench | Downstream | - | Speech, Sound, Music |
| SD-Eval | Downstream | 4 | Speech |
| VoxDialogue | Downstream | 12 | Speech, Sound, Music |
| Dynamic-SUPERB | Downstream | 180 | Speech, Sound, Music |
| SALMON | Downstream | 8 | Speech |
| VoiceBench | Downstream | 8 | Speech |
| VoxEval | Downstream | 56 | Speech |
| MMAU | Downstream | 27 | Speech, Sound, Music |

**局限**: 多数 benchmark 要求模型以文本回答,形成端到端语音交互评估的瓶颈。VoxEval 尝试解决此问题,提供语音输出的评估管线。

## 表征评估

SpeechLM 理解能力的另一维度是内部表征质量:
- **ABX score** (GSLM): 衡量 embedding 中语音类别的可分离性
- **Speech resynthesis**: 编码→解码后 WER/CER 衡量信息保留度
- **Spoken StoryCloze**: 选择故事正确结尾,衡量语义理解深度
- **sWUGGY** (词汇层): 区分真假词对
- **sBLIMP** (句法层): 识别语法正确的句子

## 关键论文

- GSLM (Lakhotia et al., 2021): 首次评估 SpeechLM 在 speech resynthesis 和 ABX 上的理解能力
- AudioPaLM (Rubenstein et al., 2023): 大规模 SpeechLM 在 ASR 和 ST 上的 SOTA
- SpeechGPT (Zhang et al., 2023): instruction-following 多任务理解和生成
- SPIRIT-LM (Nguyen et al., 2024): 对齐训练增强跨模态理解
- Dynamic-SUPERB: 180 任务的大规模综合 benchmark

## ALM 视角: 通用音频理解 [Su et al. 2025]

除 SpeechLM 路线外,Audio-Language Models (ALMs) 通过 CLAP 式预训练提供另一种理解路径,覆盖更广泛的音频类型 (环境声、音乐、语音):

### 扩展任务

| 任务 | 定义 | 与 SpeechLM 任务的区别 |
|------|------|----------------------|
| Audio Captioning (AAC) | 自然语言描述音频内容 | SpeechLM 侧重 ASR; AAC 描述声音事件及其关系 |
| Audio QA (AQA) | 基于音频回答开放问题 | 需要推理能力,不限于识别 |
| Audio-Text Retrieval (ATR) | 跨模态检索 | SpeechLM 无此任务 |
| Audio Grounding | 定位音频中与文本对应的时间段 | 时间定位能力 |

### 扩展 Benchmark [Su et al. 2025, §VI-C]

| Benchmark | 类型 | 评估目标 |
|-----------|------|----------|
| ARCH | Cross-task | Speech, music, acoustic events 综合 |
| MMAU | Cross-task | 27 tasks, 多维泛化 |
| ADU-Bench | Task-specific | Audio-text retrieval + dialogue |
| CompA-R | Task-specific | Compositional reasoning (开放式 AQA) |
| LongAudioBench | Robustness | 长音频理解能力 |
| Audio Jailbreak | Security | 对抗性攻击鲁棒性 |

### 核心差异: SpeechLM vs ALM 理解路线

| 维度 | SpeechLM 路线 | ALM 路线 |
|------|-------------|---------|
| 训练范式 | Speech token LM (自回归) | Audio-text contrastive + LLM |
| 主要输入 | Speech | General audio (speech + sound + music) |
| 表征 | Discrete tokens or continuous latent | Joint embedding space (CLAP) |
| 典型输出 | Speech or text | Text (caption, answer, label) |
| 代表模型 | SpeechGPT, Moshi, SPIRIT-LM | SALMONN, Audio Flamingo, GAMA, LTU |

详见 [[Audio-Language Pretraining]]。

## 相关概念

- [[Speech Language Model]]: Audio Understanding 是 SpeechLM 能力体系的核心组成
- [[Speech Tokenizer]]: tokenizer 决定了理解能力的上限
- [[Semantic vs Acoustic Tokens]]: token 类型影响理解任务的侧重方向
- [[Full-duplex Spoken Dialogue]]: 全双工模型需要实时理解用户语音
- [[Audio-Language Pretraining]]: ALM 路线的预训练范式,提供通用音频理解的互补视角

## 演进

单任务 ASR/SER 系统 (2015-2020) → GSLM 基础理解 (ABX + resynthesis, 2021) → AudioPaLM 大规模 ASR/ST (2023) → SpeechGPT 指令驱动多任务理解 (2023) → SPIRIT-LM 跨模态对齐理解 (2024) → Dynamic-SUPERB 180 任务综合评估 (2024) → VoxEval 端到端语音理解评估 (2024) → ALM/LALM 通用音频理解 (SALMONN, Audio Flamingo, 2024-2025)
