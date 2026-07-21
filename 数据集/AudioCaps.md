---
type: dataset
title: "AudioCaps"
aliases: [AudioCaps Test]
domain: "general-audio"
tags: [general-audio, audio-captioning, text-to-audio, sound-event]
key_papers: ["[[论文笔记/STAR-VAE|STAR-VAE]]"]
status: pending-review
lifecycle: active
created: 2026-07-21
updated: 2026-07-21
---

## 概述

AudioCaps (Kim et al., NAACL 2019) 是通用音频描述 (audio captioning) 领域的标准数据集,基于 AudioSet 的音频片段人工标注自然语言描述。广泛用作 text-to-audio (T2A) 生成与音频重建的训练/评测基准。

## 在 STAR-VAE 中的使用

- **STAR-Gen 训练**: 与 WavCaps 一起作为 T2A 训练数据 [STAR-VAE §4.1]
- **评测**: AudioCaps Test 作为 sound 域重建与 T2A 生成的主评测集;报告 STFT-D/MSD/SI-SDR/FAD/LC(重建)与 FDopenl3/KL/CLAP(生成)[STAR-VAE Table 1, 2]
