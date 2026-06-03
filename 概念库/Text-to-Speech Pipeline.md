---
type: concept
title: "Text-to-Speech Pipeline"
aliases: [TTS Pipeline, TTS系统架构, 语音合成流水线, Neural TTS Architecture]
category: "system-architecture"
tags: [TTS, pipeline, system-design, end-to-end]
key_papers: ["[[论文笔记/MathReader|MathReader]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Mel Spectrogram]]", "[[Neural Vocoder]]", "[[Attention-based TTS]]", "[[Non-autoregressive TTS]]", "[[Duration Predictor]]", "[[Phoneme Representation]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Text-to-Speech (TTS) Pipeline 是将文本输入转换为语音波形的完整系统架构。现代 neural TTS 系统由三个基本组件构成:

```
Text → [Text Analysis] → Linguistic Features → [Acoustic Model] → Acoustic Features → [Vocoder] → Waveform
```

### 三大组件

| 组件 | 输入 | 输出 | 代表方法 |
|------|------|------|----------|
| Text Analysis (前端) | 原始文本 | 音素/语言特征 | G2P, TN, Prosody Prediction |
| Acoustic Model (声学模型) | 音素序列 | Mel spectrogram | Tacotron 2, FastSpeech 2 |
| Vocoder (声码器) | Mel spectrogram | 音频波形 | HiFi-GAN, WaveRNN |

## 架构演进 (5 阶段)

Survey 定义的 TTS 端到端化进程:

**Stage 0 - SPSS**: Text Analysis → Acoustic Model (HMM) → Vocoder (STRAIGHT/WORLD)
- 完整语言学特征(POS, 韵律标注等)
- 声学参数: MGC + BAP + F0

**Stage 1 - ARST (Wang et al. 2015)**: 将 text analysis + acoustic model 合并,直接从音素生成声学特征,仍用 SPSS vocoder

**Stage 2 - WaveNet (2016)**: 首次用神经网络直接从语言特征生成波形(可视为 acoustic model + vocoder 的合体),但仍需文本分析模块

**Stage 3 - Tacotron/FastSpeech (2017-2019)**: 简化文本分析(仅保留 G2P),直接从字符/音素预测 mel spectrogram,再用 neural vocoder 合成波形

**Stage 4 - Fully E2E (2019+)**: 直接从文本生成波形
- Char2Wav, ClariNet, FastSpeech 2s, EATS, VITS, Wave-Tacotron

## 数据流分类

根据 survey 的 taxonomy,TTS 中存在多种数据流路径:
1. Character → Linguistic → Acoustic → Waveform (传统 SPSS)
2. Character → Acoustic → Waveform (Tacotron 系)
3. Phoneme → Acoustic → Waveform (FastSpeech 系)
4. Character/Phoneme → Waveform (Fully E2E)

## 其他分类维度

| 维度 | 类别 |
|------|------|
| 序列生成 | Autoregressive vs Non-autoregressive |
| 生成模型 | Seq2Seq, Flow, GAN, VAE, Diffusion |
| 网络结构 | RNN, CNN, Self-Attention, Hybrid |

## 在 TTS 中的应用

现代产品级 TTS 系统的典型配置:
- **Azure TTS**: FastSpeech 系列 (NAR acoustic model) + HiFi-GAN vocoder
- **LLM-TTS 新范式** (2023+): Text → AR Language Model → Speech Tokens → Flow/Diffusion Decoder → Waveform (如 VALL-E, CosyVoice)

## 关键论文

- Xu Tan et al., "A Survey on Neural Speech Synthesis", 2021: 系统梳理 TTS pipeline 架构演进
- Tacotron 2 (Shen et al., 2018): 确立 "AM + Vocoder" 的 neural TTS 标准范式
- FastSpeech 2 (Ren et al., 2021): 确立 NAR TTS 的标准范式
- VITS (Kim et al., ICML 2021): fully E2E, 单模型直接文本到波形

## 相关概念

- [[Mel Spectrogram]]: pipeline 中声学模型和声码器之间的中间表示
- [[Neural Vocoder]]: pipeline 最后一级
- [[Attention-based TTS]]: AR 声学模型的主流方案
- [[Non-autoregressive TTS]]: NAR 声学模型的主流方案
- [[Duration Predictor]]: NAR pipeline 的关键组件
- [[Phoneme Representation]]: pipeline 前端输出

## 演进

Concatenative TTS → SPSS (HMM+WORLD) → Neural SPSS (DNN+WORLD) → End-to-End (Tacotron + WaveNet) → NAR (FastSpeech + HiFi-GAN) → Fully E2E (VITS) → LLM-based (VALL-E / CosyVoice, 2023+)
