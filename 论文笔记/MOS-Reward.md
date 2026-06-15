---
type: paper
tier: card
title: "From Scores to Preferences: Redefining MOS Benchmarking for Speech Quality Reward Modeling"
arxiv_id: "2510.00743"
source: "https://arxiv.org/abs/2510.00743"
authors: [Yifei Cao, Changhao Jiang, Jiabao Zhuang, Jiajun Sun, Ming Zhang, Zhiheng Xi, Hui Li, Shihan Dou, Yuran Wang, Yunke Zhang, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang]
year: 2025
venue: ""
tags: [MOS, preference-learning, reward-model, speech-quality, evaluation]
concepts: []
models: []
tasks: []
datasets: []
status: draft
created: 2026-06-15
updated: 2026-06-15
---

## 速查卡片

- **一句话**: 将 MOS 评估从绝对打分重定义为偏好排序,构建语音质量 Reward Model (MOS-Reward)
- **核心贡献**: 提出 MOS-Reward benchmark,用偏好对而非绝对分数训练和评估语音质量模型,对齐 RLHF 范式
- **关键发现**: 绝对 MOS 标注一致性差/可复现性低,偏好对更稳定;Reward Model 可直接用于 TTS 的 RLHF 训练
- **为何值得关注**: 连接 MOS 评估与 RLHF,来自复旦 NLP 组,是 TTS+LLM alignment 的关键基础设施
