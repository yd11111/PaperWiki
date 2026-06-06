---
type: dataset
title: "IndicVoices-R"
aliases: [IV-R, IndicVoices-R Dataset]
domain: "Large-scale multilingual Indian TTS training"
scale: "1,704 hours, 10,496 speakers, 22 Indian languages"
tags: [training-data, large-scale, multilingual, Indian-languages, TTS, speech-enhancement, low-resource]
used_by: []
metrics_reported_on: [N-MOS, SNR, C50, F0, Speaker-Similarity]
url: "https://github.com/AI4Bharat/IndicVoices-R"
origin_paper: "[[论文笔记/IndicVoices-R|IndicVoices-R]]"
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-06
updated: 2026-06-06
---

## 概述

IndicVoices-R (IV-R) 是首个覆盖全部 22 种印度官方语言的大规模多说话人 TTS 数据集,由 AI4Bharat (IIT Madras) 通过语音增强管道从 ASR 数据集 IndicVoices 转化而来。数据集包含 1,704 小时高质量语音,来自 10,496 位说话人,其中 93.25% 为 extempore (即兴) 录音。详见 [[论文笔记/IndicVoices-R|IndicVoices-R 论文笔记]]。

## 规模与特点

- **总量**: 1,704 小时, 689,568 条语音
- **说话人**: 10,496 (男 5,030 / 女 5,466), 覆盖 18-60+ 四个年龄段
- **语种**: 22 种印度官方语言,其中 9 种 (Dogri, Kashmiri, Konkani, Maithili, Nepali, Sanskrit, Santali, Sindhi, Urdu) 首次有开源 TTS 数据
- **采样率**: 44.1 kHz → 增强后 48 kHz (DeepFilterNet3 输出)
- **风格**: 93.25% extempore (即兴录音) + 6.75% read-speech
- **数据来源**: IndicVoices ASR 数据集 (手动标注, 知情同意)
- **质量**: N-MOS 3.38, SNR 60.47 dB, C50 53.45 dB (低于录音室数据集但在 in-the-wild 数据中具竞争力)

## 数据管道

从 IndicVoices (ASR) 到 IndicVoices-R (TTS) 的 6 步增强管道:

1. **预处理**: 保留 44.1kHz 音频, 过滤 >30s 样本, mono→stereo
2. **Demixing**: HTDemucs 源分离降噪
3. **去混响**: VoiceFixer 消除混响
4. **语音增强**: DeepFilterNet3 消除数字伪影
5. **过滤**: C50>=30dB, SNR>=25dB, pitch/speaking-rate/CER 多维阈值
6. **后处理**: PyDub 音量归一化

## IV-R Benchmark

配套发布的首个印度语 TTS speaker generalization benchmark:
- 352 zero-shot + 541 few-shot (<5min) + 1324 medium-shot (<10min) + 2126 many-shot (>10min) 说话人
- 覆盖双性别 x 4 年龄段, 测试集 ~39.5h

## 来源

Sankar et al., "IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS", arXiv:2409.05356, 2024.
