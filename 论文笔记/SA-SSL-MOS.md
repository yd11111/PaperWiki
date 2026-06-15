---
type: paper
tier: card
title: "SA-SSL-MOS: Self-supervised Learning MOS Prediction with Spectral Augmentation for Generalized Multi-Rate Speech Assessment"
arxiv_id: "2602.14785"
source: "https://arxiv.org/abs/2602.14785"
authors: [Fengyuan Cao, Xinyu Liang, Fredrik Cumlin, Victor Ungureanu, Chandan K. A. Reddy, Christian Schuldt, Saikat Chatterjee]
year: 2026
venue: ""
tags: [MOS-prediction, SSL, spectral-augmentation, multi-rate, evaluation]
concepts: []
models: []
tasks: []
datasets: []
status: draft
created: 2026-06-15
updated: 2026-06-15
---

## 速查卡片

- **一句话**: 通过频谱增强让 SSL-based MOS 预测器支持 16-48kHz 多采样率语音评估
- **核心贡献**: 解决 SSL 模型通常只支持 16kHz 的限制,通过频谱增强策略泛化到多采样率
- **关键发现**: 多采样率语音的 MOS 标注数据极度稀缺,频谱增强可有效扩充训练分布
- **为何值得关注**: 实用性强,现实中 TTS 输出采样率多样(16k/22k/24k/44.1k/48k),单一模型全覆盖是刚需
