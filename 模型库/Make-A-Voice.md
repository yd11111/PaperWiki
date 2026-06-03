---
type: model
title: "Make-A-Voice"
aliases: [MakeAVoice, Make A Voice]
org: "Zhejiang University / Tencent AI Lab"
year: 2023
tags: [TTS, voice-conversion, SVS, zero-shot, unified-framework, discrete-token, coarse-to-fine, autoregressive]
key_concepts: ["[[Semantic vs Acoustic Tokens]]", "[[Residual Vector Quantization]]", "[[Neural Vocoder]]", "[[LLM-based TTS]]", "[[F0 Modeling]]", "[[Singing Voice Synthesis]]"]
tasks: [TTS, voice-conversion, singing-voice-synthesis]
key_papers: ["[[论文笔记/Make-A-Voice|Make-A-Voice]]"]
supersedes: []
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

Make-A-Voice (Huang et al., 2023) 是基于离散语音表示的统一语音合成框架,通过 coarse-to-fine 三阶段设计 (semantic→acoustic→waveform) 统一处理 TTS、VC 和 SVS 三个任务。核心 backbone (acoustic stage + generation stage) 不需要文本标注,可利用海量无标注音频数据训练。Demo: https://Make-A-Voice.github.io

## 核心方法

- **三阶段 coarse-to-fine**: S1 (Text-to-Semantic / HuBERT) → S2 (Semantic-to-Acoustic Transformer) → S3 (Unit-based Vocoder) [Fig 1]
- **Semantic tokens**: HuBERT 12th layer + k-means 离散化 [§3.3.1]
- **Acoustic tokens**: SoundStream 12 层 RVQ,推理取前 3 层 [§3.3.2]
- **Unit-based vocoder**: 替代 SoundStream decoder,避免 codebook mismatch [§3.6]
- **Speaker prompt conditioning**: concatenate prompt acoustic tokens + semantic tokens [§3.5]
- **Quantized F0 prompt**: 256-level discretized F0 作为 SVS 条件 [§3.5]
- **统一任务**: S2/S3 在 TTS/VC/SVS 间完全共享参数 [Fig 3]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| TTS MOS | 4.04 (GT 4.23) | LibriTTS test | [Table 2] |
| TTS SMOS | 3.81 | LibriTTS test | [Table 2] |
| TTS CER | 0.068 | LibriTTS test | [Table 2] |
| TTS Cos | 0.77 | LibriTTS test | [Table 2] |
| VC MOS | 4.07 (GT 4.26) | LibriTTS test | [Table 3] |
| VC Cos | 0.80 | LibriTTS test | [Table 3] |
| SVS MOS | 3.99 (GT 4.08) | M4Singer test | [Table 4] |
| SVS FFE | 0.05 | M4Singer test | [Table 4] |

## 演进线

AudioLM (semantic→acoustic 层级) → VALL-E (codec LM for TTS) → **Make-A-Voice (统一 TTS/VC/SVS + coarse-to-fine 三阶段)** → 后续: HierSpeech++ (非自回归层级), CosyVoice (监督式 semantic tokens)

## 关键贡献

1. 首个统一 TTS/VC/SVS 的离散表示框架,S2/S3 完全共享
2. Unit-based vocoder 解决 RVQ codebook mismatch 问题
3. S2/S3 仅需 audio-only data,数据 scalability 强 (LibriLight 60K hours)
4. Quantized F0 prompt 实现精确 SVS pitch control (FFE 0.05)
5. 证明 coarse-to-fine 离散表示可同时 serve 语音和歌唱合成
