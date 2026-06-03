---
type: model
title: "CosyVoice"
aliases: [CosyVoice1, CosyVoice-base, CosyVoice-instruct]
org: "Alibaba (Speech Lab)"
year: 2024
tags: [TTS, zero-shot, multilingual, LLM-based, coarse-to-fine, flow-matching]
key_concepts: ["[[Speech Tokenizer]]", "[[Conditional Flow Matching]]", "[[Classifier-Free Guidance]]", "[[Speaker Embedding]]"]
tasks: ["[[Zero-shot Speech Synthesis]]", "[[Cross-lingual Voice Cloning]]", "[[Instructed Speech Generation]]"]
key_papers: ["[[论文笔记/CosyVoice|CosyVoice]]", "[[论文笔记/CosyVoice 2|CosyVoice 2]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]"]
supersedes: []
superseded_by: ["[[模型库/CosyVoice 2|CosyVoice 2]]"]
status: confirmed
lifecycle: active
merged_into: ""
created: 2026-06-02
updated: 2026-06-02
---

## 概述

CosyVoice 是阿里巴巴语音实验室提出的可扩展多语言零样本 TTS 系统。其核心创新是引入**监督式 semantic tokens (S3)**,通过在 ASR encoder 中插入 VQ 层获得天然携带语义信息且与文本对齐的 speech tokens。系统采用 LLM + OT-CFM 的 coarse-to-fine 两阶段架构,并通过 x-vector 显式分离说话人建模。

## 核心方法

1. **S3 (Supervised Semantic Speech) Tokenizer**: 在 SenseVoice-Large ASR encoder 第 6 层后插入 VQ (单码本, 4096 entries),以 ASR loss 监督训练
2. **LLM**: 自回归生成 speech tokens,输入序列 [S, x-vec, text_enc, T, speech_tokens, E]
3. **OT-CFM**: 条件 flow matching 将 speech tokens 转为 mel spectrogram,使用 cosine scheduler + CFG (β=0.7)
4. **HiFi-GAN**: Mel → waveform
5. **CosyVoice-instruct**: 指令微调变体,支持 speaker identity / style / paralinguistics 控制

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| WER (%) | 3.17 | LibriTTS test-clean | Table 7 |
| SS | 69.49 | LibriTTS test-clean | Table 7 |
| WER (%) 英文 | 2.89±0.18 | LibriTTS test-clean (Whisper-L V3) | Table 8 |
| SS 英文 | 74.30±0.15 | LibriTTS test-clean | Table 8 |
| CER (%) 中文 | 3.82±0.24 | AISHELL-3 | Table 9 |
| SS 中文 | 81.58±0.16 | AISHELL-3 | Table 9 |

## 演进线

CosyVoice (2024, S3 tokenizer, LLM+OT-CFM) → [[模型库/CosyVoice 2|CosyVoice 2]] (2024, streaming, LLM init) → [[模型库/CosyVoice 3|CosyVoice 3]] (2025, MinMo tokenizer, DiffRO, 1M h)

## 关键贡献

- 首次将 ASR 监督式 speech tokens 引入 TTS,证明监督 token 在内容一致性和说话人相似度上全面优于无监督 token
- 提出 LLM + OT-CFM 的 coarse-to-fine 架构,被后续大量 TTS 系统采用
- 通过 x-vector 将语音建模分解为语义+韵律 (LLM) 和音色+环境 (CFM)
- 在英文和中文上均达到或超过人类水平的说话人相似度
