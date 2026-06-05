---
type: model
title: "HierSpeech++"
aliases: [HierSpeechpp, HierSpeech Plus Plus, 层级语音合成]
org: "Korea University"
year: 2023
tags: [TTS, voice-conversion, zero-shot, hierarchical-VAE, non-autoregressive, speech-super-resolution]
key_concepts: ["[[VariationalAutoencoderforTTS]]", "[[SpeechFactorization]]", "[[SemanticvsAcousticTokens]]", "[[F0Modeling]]", "[[NeuralVocoder]]"]
tasks: [TTS, voice-conversion, speech-super-resolution]
key_papers: ["[[论文笔记/HierSpeech++|HierSpeech++]]", "[[论文笔记/Low-ResourceForwardTacotron|Low-Resource ForwardTacotron (Kayyar et al., 2025)]]"]
supersedes: ["HierSpeech (Lee et al., 2022)", "HierVST (Lee et al., 2023)"]
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

HierSpeech++ (Lee et al., 2023) 是基于层级变分推断的快速、强大的零样本语音合成框架。由三个子系统组成: Hierarchical Speech Synthesizer (核心,层级 VAE + BiT-Flow + HAG)、Text-to-Vec (TTV, 文本→语义表示) 和 SpeechSR (语音超分辨率 16→48 kHz)。首次在零样本 TTS 和 VC 任务上同时达到人类水平自然度。开源: https://github.com/sh-lee-prml/HierSpeechpp

## 核心方法

- **Hierarchical VAE**: 使用 MMS (Wav2Vec 2.0) 作为连续 semantic representation,层级 VAE 桥接 semantic→acoustic gap [§3.2]
- **Source-Filter Multi-path Semantic Encoder**: speech perturbation 实现 speaker-agnostic/related 分离 [§3.2.2]
- **Dual-audio Acoustic Encoder**: wav encoder + spec encoder 增强 acoustic capacity [§3.2.1]
- **Bidirectional Transformer Flow (BiT-Flow)**: 双向训练减少 train-inference mismatch [§3.2.5]
- **Hierarchical Adaptive Generator (HAG)**: BigVGAN AMP block + F0 source generator [§3.2.4]
- **Style Prompt Replication (SPR)**: 复制短 prompt 解决 <3s 场景 [§4.3]
- **SpeechSR**: 0.13M 参数的轻量超分辨率,DWT sub-band discriminator [§3.4]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| TTS nMOS | 4.56 (GT 4.32) | LibriTTS test-clean | [Table 7] |
| TTS CER | 0.90 | VCTK (VC) | [Table 5] |
| TTS SECS | 0.911 | LibriTTS (large) | [Table 7] |
| VC nMOS | 4.54 | VCTK | [Table 5] |
| VC SECS | 0.875 | VCTK (LT-960) | [Table 5] |
| SpeechSR PESQ | 4.63 | VCTK | [Table 10] |

## 演进线

HierSpeech (2022, 自监督 VAE 桥接 text-speech gap) → HierVST (2023, 层级零样本 voice style transfer) → **HierSpeech++ (2023, 统一 TTS/VC + SpeechSR, 人类水平质量)**

## 关键贡献

1. 首次零样本 TTS/VC 同时超越 ground-truth 自然度 (nMOS)
2. 非自回归全并行推理,无 repeat/skip 问题
3. 不需文本标注的 hierarchical synthesizer 设计,数据 scalability 强
4. SPR 技巧使 1s prompt 零样本成为可能
5. SpeechSR 仅 0.13M 参数,742x 快于 AudioSR
