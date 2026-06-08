---
type: dataset
title: "LibriTTS"
aliases: [LibriTTS-R, LibriTTS Dataset]
domain: "English TTS training and evaluation"
scale: "585 hours, English, multi-speaker"
tags: [training-data, evaluation, TTS, english, audiobook]
used_by: []
metrics_reported_on: []
url: "https://www.openslr.org/60/"
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-08
updated: 2026-06-08
---

## 概述

LibriTTS 是基于 LibriSpeech 语料库重新对齐和处理的英语 TTS 数据集，由 Zen et al. (2019) 发布。相比 LibriSpeech，LibriTTS 修正了句子级对齐、过滤了噪声样本、统一了采样率，更适合 TTS 训练和评估。

LibriTTS-R 是 Koizumi et al. (2023) 发布的增强版本，使用语音修复模型提升了音频质量。

## 规模与特点

- **总量**: 585 小时（原版）
- **说话人数**: 2,456 人
- **采样率**: 24 kHz（原版），比 LibriSpeech 的 16 kHz 更适合 TTS
- **子集划分**: train-clean-100, train-clean-360, train-other-500, dev-clean, dev-other, test-clean, test-other
- **数据来源**: LibriVox 有声书朗读
- **文本**: 包含原始书籍文本（含标点），比 LibriSpeech 的 ASR 转录更完整

## 与 LibriSpeech 的区别

- 句子级切分（LibriSpeech 为段落级）
- 保留原始标点和大小写
- 24 kHz 采样率
- 过滤了对齐质量差的样本

## 典型用途

- TTS 模型训练（尤其是学术基准实验）
- 零样本 TTS 评估
- 语音编解码器（codec）重建质量评测
- 说话人验证/识别辅助评估
