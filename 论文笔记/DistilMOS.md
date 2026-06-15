---
type: paper
tier: card
title: "DistilMOS: Layer-Wise Self-Distillation For Self-Supervised Learning Model-Based MOS Prediction"
arxiv_id: "2601.13700"
source: "https://arxiv.org/abs/2601.13700"
authors: [Jianing Yang, Wataru Nakata, Yuki Saito, Hiroshi Saruwatari]
year: 2026
venue: ""
tags: [MOS-prediction, self-distillation, SSL, evaluation]
concepts: []
models: []
tasks: []
datasets: []
status: draft
created: 2026-06-15
updated: 2026-06-15
---

## 速查卡片

- **一句话**: 通过层级自蒸馏缓解 SSL 模型 fine-tune MOS 预测时的灾难性遗忘和过拟合
- **核心贡献**: 在 SSL 模型各层间引入自蒸馏,保留预训练知识的同时学习 MOS 回归
- **关键发现**: 标准 fine-tune SSL→MOS 会灾难性遗忘预训练特征;层级蒸馏显著提升泛化性
- **为何值得关注**: 解决 SSL-based MOS 预测的核心工程问题(过拟合小数据集),方法简洁通用
