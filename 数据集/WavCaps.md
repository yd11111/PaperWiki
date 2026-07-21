---
type: dataset
title: "WavCaps"
aliases: [WavCaps]
domain: "general-audio"
tags: [general-audio, audio-captioning, weakly-labelled, text-to-audio]
key_papers: ["[[论文笔记/STAR-VAE|STAR-VAE]]"]
status: pending-review
lifecycle: active
created: 2026-07-21
updated: 2026-07-21
---

## 概述

WavCaps (Mei et al., TASLP 2024) 是一个 ChatGPT 辅助的弱标注音频描述数据集,规模大,用于 audio-language 多模态研究,常作为 text-to-audio 生成的训练数据。

## 在 STAR-VAE 中的使用

- **STAR-Gen 训练**: 与 AudioCaps 一起作为 T2A flow matching 训练数据 [STAR-VAE §4.1, Appendix B.3]
