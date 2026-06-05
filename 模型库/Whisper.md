---
type: model
title: "Whisper"
aliases: [OpenAI Whisper, Whisper ASR]
org: "OpenAI"
year: 2022
tags: [ASR, weak-supervision, multilingual, multitask, zero-shot, robustness, speech-recognition]
key_concepts: ["[[MelSpectrogram]]", "[[SpeechTokenizer]]", "[[LLM-enhancedASR]]"]
tasks: []
key_papers: ["[[论文笔记/Whisper|Whisper]]", "[[论文笔记/TITW|TITW]]", "[[论文笔记/GOAT-TTS|GOAT-TTS]]", "[[论文笔记/LM-SPT|LM-SPT]]", "[[论文笔记/MAVE|MAVE]]", "[[论文笔记/LightweightPromptBiasing|Lightweight Prompt Biasing]]", "[[论文笔记/W3AR|W3AR]]"]
supersedes: []
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

Whisper (Radford et al., ICML 2023) 是 OpenAI 开发的大规模弱监督语音识别系统。通过在 680,000 小时互联网音频-转录配对数据上训练标准 encoder-decoder Transformer,Whisper 在 zero-shot 设定下实现了接近人类转录员的鲁棒性,且无需任何数据集特定的 fine-tuning [§1]。其 encoder 后来成为 SpeechLM 中最流行的 speech feature extractor。

## 核心方法

- **大规模弱监督**: 680k 小时多语言多任务数据 (563k EN + 117k multilingual + 125k X→EN translation) [§2.1]
- **标准 Transformer**: encoder-decoder 结构,encoder stem 为 2 层 Conv1D,输入 80-channel log-mel spectrogram [§2.2]
- **Multitask Training Format**: 用 special tokens (<|language|>, <|transcribe|>/<|translate|>, <|timestamps|>) 统一 ASR/翻译/LID/VAD 多任务 [§2.3]
- **Zero-shot 评估哲学**: 不使用目标数据集的训练数据,衡量广义泛化能力 [§3.1]
- **模型家族**: Tiny (39M) → Base (74M) → Small (244M) → Medium (769M) → Large (1550M) [Table 1]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| WER (zero-shot) | 2.7 | LibriSpeech Clean | [Table 2] |
| WER (OOD avg, zero-shot) | 12.8 (vs wav2vec 2.0: 29.3) | 12 datasets | [Table 2] |
| BLEU (X→EN, zero-shot) | 29.1 (SOTA) | CoVoST2 | [Table 4] |
| Long-form avg WER | 10.0 | 7 datasets | [Table 7] |

## 演进线

Deep Speech 2 (2015; 监督扩展) → wav2vec 2.0 (2020; 自监督 + fine-tuning) → HuBERT (2021; masked prediction + fine-tuning) → **Whisper** (2022; 弱监督 + zero-shot) → USM (Google, 2023; 监督 + 自监督混合)

## 关键贡献

1. 证明弱监督大数据 > 小规模金标准数据的 scaling 路线 [Table 6]
2. 提出 effective robustness 分析框架,揭示监督模型的 OOD 脆弱性 [§3.3, Fig 2]
3. Multitask token format 被后续 SpeechLM 广泛借鉴 [§2.3]
4. Whisper encoder 成为 SpeechLM 最流行的 speech encoder (被 Kimi-Audio, Qwen2.5-Omni 等采用)
