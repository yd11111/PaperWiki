---
type: paper
tier: card
title: "Knowing What to Stress: A Discourse-Conditioned Text-to-Speech Benchmark"
arxiv_id: "2604.10580"
source: "https://arxiv.org/abs/2604.10580"
authors: [Arnon Turetzky, Avihu Dekel, Hagai Aronowitz, Ron Hoory, Yossi Adi]
year: 2026
venue: ""
tags: [benchmark, stress, prosody, discourse, evaluation]
concepts: ["[[ProsodyModeling]]"]
models: []
tasks: []
datasets: []
status: draft
created: 2026-06-15
updated: 2026-06-15
---

## 速查卡片

- **一句话**: CAST benchmark,评估 TTS 系统能否根据话语上下文正确推断词级重音位置
- **核心贡献**: 构建对比性上下文对(同一句话 + 不同上下文 → 不同重音位置),测试 TTS 是否能从 discourse 推断重音
- **关键发现**: 相同句子在不同上下文中应强调不同词,但多数 TTS 系统缺乏 discourse-aware 重音能力
- **为何值得关注**: 韵律评估的细粒度新维度,从"听起来自然"进化到"语义上正确的重音",出自 IBM + Hebrew University
