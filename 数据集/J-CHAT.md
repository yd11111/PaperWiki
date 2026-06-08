---
type: dataset
title: "J-CHAT"
aliases: [Japanese Corpus for Human-AI Talks, J-CHAT Corpus]
domain: "Japanese spoken dialogue for end-to-end SDS training"
scale: "76,036 hours, Japanese, multi-speaker dialogue"
tags: [training-data, spoken-dialogue, Japanese, spontaneous-speech, speech-LM, large-scale]
used_by: []
metrics_reported_on: []
url: "https://huggingface.co/datasets/sarulab-speech/J-CHAT"
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-08
updated: 2026-06-08
---

## 概述

J-CHAT (Japanese Corpus for Human-AI Talks) 是目前最大的开源日语对话语音语料库,由 Nakata et al. (2024) 发布。使用自动化、语言无关的 pipeline 从 YouTube 和播客构建,覆盖对话级分段、降噪和 ASR 转录。主要面向端到端 Spoken Dialogue System (dGSLM 等) 训练。详见 [[论文笔记/J-CHAT|J-CHAT 论文笔记]]。

## 规模与特点

- **总量**: 76,036 小时 (YouTube 11,017h + Podcast 65,019h)
- **对话数**: 5,424,514 个
- **平均对话时长**: 50.23 秒
- **平均轮次数**: 10.10 轮/对话
- **平均说话人数**: 3.14 人/对话
- **数据来源**: YouTube (~600k 文件) + Podcast (~880k 文件, 通过 PodcastIndex)
- **许可证**: CC BY-NC 4.0 (日本版权法"信息分析"条款)
- **预定义划分**: train/valid/test/other (YouTube: 10872.5/108.7/1.2h; Podcast: 57291.5/575/1.3h)
- **附加标注**: ASR 转录 (reazonspeech-nemo-v2) + subword 级对齐 + 说话人分段标签

## 质量

- **NISQA 音质**: Podcast 2.99 / YouTube 2.37 (vs STUDIES 4.01 录音棚, CallHome-JP 1.98 电话)
- **降噪**: 全量使用 Demucs 去除背景音乐
- **语言过滤**: Whisper LangID p > 0.8

## 多样性

- **话题多样性**: 平均余弦相似度 YouTube 0.2390 / Podcast 0.3457,远低于 CallHome-JP 0.6164,话题覆盖更广
- **语音多样性**: HuBERT 特征分布与自发对话一致,覆盖录音棚语音无法触及的区域
- **自发语音**: 包含自然的犹豫、重复、回应等自发语音特征

## 构建 pipeline

Internet → Download → Whisper LangID → PyAnnote 说话人分段 → 对话提取 (5s gap + 80% 单人过滤) → Demucs 降噪 → ASR 转录

## 与相关语料的对比

| 语料 | 规模 | 开源 | 对话 | 自发 | 干净 |
|------|------|------|------|------|------|
| STUDIES | 8.2h | Yes | Yes | No | Yes |
| DailyTalk | 20h | Yes | Yes | No | Yes |
| CallHome-JP | 49h | Yes | Yes | Yes | Yes |
| Fisher | 2kh | No | Yes | Yes | Yes |
| Seamless Interaction | 4kh | Yes | Yes | Yes | Yes |
| **J-CHAT** | **76kh** | **Yes** | **Yes** | **Yes** | **Yes** |
