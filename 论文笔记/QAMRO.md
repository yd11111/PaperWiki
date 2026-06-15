---
type: paper
tier: card
title: "QAMRO: Quality-aware Adaptive Margin Ranking Optimization for Human-aligned Assessment of Audio Generation Systems"
arxiv_id: "2508.08957"
source: "https://arxiv.org/abs/2508.08957"
authors: [Chien-Chun Wang, Kuan-Tang Huang, Cheng-Yeh Yang, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen]
year: 2025
venue: ""
tags: [ranking-optimization, MOS, audio-generation, evaluation, TTS]
concepts: []
models: []
tasks: []
datasets: []
status: draft
created: 2026-06-15
updated: 2026-06-15
---

## 速查卡片

- **一句话**: 质量感知自适应边际排序优化,用排序损失替代回归损失做音频生成系统评估
- **核心贡献**: MOS 回归忽略感知判断的相对性;QAMRO 用自适应 margin 的 ranking loss 更好对齐人类偏好
- **关键发现**: 标准回归损失对相近质量样本区分度差;自适应 margin 根据质量差距动态调整,提升系统级排序一致性
- **为何值得关注**: 与 DRASP 同一组(中研院),覆盖 TTS/TTM/TTA 三种音频生成场景的统一评估,排序优化思路与 MOS-Reward 互补
