---
type: model
title: "SenseVoice"
aliases: [SenseVoice-Small, SenseVoice-Large, SenseVoice-S, SenseVoice-L]
org: "Alibaba (Tongyi SpeechTeam)"
year: 2024
tags: [ASR, SER, AED, LID, multilingual, non-autoregressive, speech-understanding, open-source]
key_concepts: ["[[Self-Supervised Speech Representation]]", "[[Audio Understanding]]", "[[Speech Tokenizer]]"]
tasks: []
key_papers: ["[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/MELA-TTS|MELA-TTS]]"]
supersedes: []
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

SenseVoice 是阿里巴巴通义语音团队提出的语音理解基座模型,支持 ASR (自动语音识别)、SER (语音情感识别)、AED (音频事件检测)、LID (语言识别) 四大任务。分为两个版本:

- **SenseVoice-Small**: 非自回归 encoder-only (SAN-M),234M params,支持 5 语言 (ZH/EN/Yue/JP/KO),RTF 0.007,延迟 70ms/10s (5x faster than Whisper-small)
- **SenseVoice-Large**: 自回归 encoder-decoder (Transformer),1587M params,支持 50+ 语言,精度更高但推理更慢

SenseVoice 同时是 CosyVoice 系列 S^3 supervised semantic speech tokenizer 的基础模型——S^3 tokenizer 在 SenseVoice-Large encoder 第 6 层后插入 VQ 构建。

## 核心方法

1. **Multi-task via Task Embeddings (Small)**: 4 个 special token (e_LID, e_SER, e_AEC, e_ITN) prepend 到语音特征,通过 CTC/CE 联合训练,单一前向传播同时输出 ASR+SER+AED+LID [FunAudioLLM §2.2, Eq.1-2]
2. **Rich Transcription (Large)**: 与 Whisper 类似的 start prompt 指定任务,可输出带时间戳的转录+情感+音频事件标签 [FunAudioLLM §2.2]
3. **训练数据**: Small ~300K hours (5 languages); Large ~400K hours (50+ languages); 使用开源 AED/SER 模型自动标注 150M AED + 30M SER 条伪标签 [FunAudioLLM §3.1]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| CER | 2.09 | AISHELL-1 test | [FunAudioLLM Table 6] |
| CER | 7.68 | CommonVoice zh-CN | [FunAudioLLM Table 6] |
| WER | 2.57 | LibriSpeech clean | [FunAudioLLM Table 6] |
| RTF | 0.007 | SenseVoice-S | [FunAudioLLM Table 7] |
| 10s Latency | 70ms | SenseVoice-S | [FunAudioLLM Table 7] |
| SER WA | 96.0 | CREMA-D (SenseVoice-L) | [FunAudioLLM Table 8] |
| SER WA | 93.2 | ESD (SenseVoice-L) | [FunAudioLLM Table 8] |

## 演进线

SenseVoice (2024, ASR+SER+AED+LID) → 作为 S^3 tokenizer 基础集成入 CosyVoice (2024) → MinMo (2025) 替代 SenseVoice-Large 作为 CosyVoice 3 的 tokenizer backbone

## 关键贡献

- 首个同时覆盖 ASR+SER+AED+LID 的统一语音理解模型,通过 task embedding 实现单模型多任务
- 非自回归 (CTC-based) 设计使 SenseVoice-Small 推理速度极快 (5x Whisper-small, 15x Whisper-large)
- 作为 S^3 supervised semantic tokenizer 的基础,成为 CosyVoice 系列 TTS 系统的核心组件
- SER 性能在 7 个 benchmark 上几乎全面 SOTA (无需 fine-tuning)
- 已开源 (ModelScope + HuggingFace)
