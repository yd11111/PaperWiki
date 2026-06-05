---
type: model
title: "MinMo"
aliases: [MinMo LLM]
org: "Alibaba (Tongyi Lab)"
year: 2025
tags: [multimodal-LLM, speech-understanding, voice-interaction]
key_concepts: ["[[SpeechTokenizer]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
key_papers: ["[[论文笔记/CosyVoice3|CosyVoice 3]]"]
supersedes: []
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

MinMo (Multimodal Large Language Model for Seamless Voice Interaction) 是阿里巴巴通义实验室开发的多模态大语言模型,在超过 140 万小时语音数据上预训练,在对话、多语种 ASR、情感识别等多个语音任务上达到 SOTA 水平。

在 CosyVoice 3 中,MinMo 的 Voice Encoder 被用作 speech tokenizer 的 backbone — FSQ 模块插入其 Voice Encoder_1(12 层 Transformer + RoPE)中间层。

## 核心方法

- 大规模语音预训练(1.4M 小时)
- 多模态对齐(语音-文本)
- 支持 spoken dialogue、multilingual ASR、emotion recognition 等

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| ASR WER C.V. EN | 11.36 | CommonVoice | CosyVoice 3 Table 10 |
| ASR CER C.V. CN | 9.21 | CommonVoice | CosyVoice 3 Table 10 |
| ASR CER C.V. JA | 13.90 | CommonVoice | CosyVoice 3 Table 10 |
| LID Accuracy | 99.2 | AIR-Bench | CosyVoice 3 Table 11 |

## 演进线

FunAudioLLM / SenseVoice → MinMo (2025, 多模态语音 LLM)

## 关键贡献

- 为 CosyVoice 3 提供了强大的语音理解 backbone,使 speech tokenizer 的多任务监督训练成为可能
- 其丰富的预训练知识使得 FSQ 量化后的 token 天然携带副语言信息
