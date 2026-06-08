---
type: dataset
title: "LibriSpeech"
aliases: [LibriSpeech ASR Corpus, LibriSpeech Dataset]
domain: "English ASR training and evaluation"
scale: "960.9 hours, English, multi-speaker"
tags: [training-data, evaluation, ASR, english, audiobook, benchmark]
used_by: []
metrics_reported_on: [WER]
url: "https://www.openslr.org/12/"
origin_paper: "[[论文笔记/DiscreteVsContinuousLLM-ASR|Xu et al., 2024]]"
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-08
updated: 2026-06-08
---

## 概述

LibriSpeech 是 Panayotov et al. (2015) 基于 LibriVox 公共领域有声书创建的大规模英语 ASR 数据集。是 ASR 领域最广泛使用的基准之一,也是 LLM-based ASR 和 Speech-LLM 系统的标准评测集。

## 规模与特点

- **总量**: 960.9 小时训练集, 10.7 小时验证集, 5.4 小时 test-clean, 5.1 小时 test-other
- **采样率**: 16 kHz
- **语言**: 英语
- **数据来源**: LibriVox 有声书朗读
- **子集划分**: train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other, test-clean, test-other
- **主要评测指标**: WER (Word Error Rate) on test-clean / test-other

## 在 TTS/ASR 研究中的角色

- **ASR 基准**: 几乎所有 LLM-based ASR 系统在 LibriSpeech 上报告 WER
- **TTS 评估**: 合成语音的 intelligibility 评估通常使用 ASR 模型在 LibriSpeech 上训练的检查点计算 WER
- **SSL 预训练**: HuBERT, wav2vec 2.0 等自监督模型使用 LibriSpeech 960h 作为预训练数据

## 相关数据集

- [[LibriTTS]]: 基于 LibriSpeech 重新处理的 TTS 专用版本 (24 kHz, 句子级对齐)
- [[LibriQuote]]: LibriSpeech 衍生的引用式语音数据集

## 关键论文

- Panayotov et al., "LibriSpeech: an ASR corpus based on public domain audio books", ICASSP 2015
