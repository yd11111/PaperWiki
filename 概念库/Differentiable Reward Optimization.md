---
type: concept
title: "Differentiable Reward Optimization"
aliases: [DiffRO]
category: "training-strategy"
tags: [reinforcement-learning, post-training, TTS, reward-model]
key_papers: ["[[CosyVoice 3]]"]
related_concepts: ["[[Gumbel-Softmax]]", "[[Speech Tokenizer]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Differentiable Reward Optimization (DiffRO) 是 CosyVoice 3 提出的一种适用于 TTS 系统的 post-training 方法。其核心思想是: 在离散 speech token 层面直接计算 reward 并反向传播梯度,而非在最终音频层面做强化学习。

关键组件:
1. **Token2Text Reward Model**: 一个类 ASR 模型,输入 speech token 输出文本后验概率,作为内容一致性的 reward
2. **Gumbel-Softmax 采样**: 使 LLM 输出的离散 token 选择可微,梯度可回传
3. **Token-level KL 约束**: 在每个时间步的 token logits 上计算 KL 散度(而非 sequence-level),防止策略偏离参考模型
4. **Multi-task Reward (MTR)**: 可扩展到 SER、AED、MOS 等多个 reward 目标

## 在 TTS 中的应用

DiffRO 解决了 TTS RL 的两个核心难题:
- **计算成本**: 传统方法需通过 CFM + vocoder 生成完整音频后才能计算 reward;DiffRO 直接在 token 空间操作
- **正负样本区分度**: 生成的语音经过 downstream rendering 后高度相似,难以训练 reward model;DiffRO 在 token 层有更大区分度

实验显示 DiffRO 在 CosyVoice 2 和 CosyVoice 3 上均有效,WER 相对改进 20%~50%,低资源语言(如韩语)改进可达 68.7%。

## 关键论文

- CosyVoice 3 (2025): 首次提出 DiffRO

## 相关概念

- RLHF: NLP 中的对齐方法,DiffRO 可视为其在 TTS 的 token-level 适配
- [[Gumbel-Softmax]]: DiffRO 实现可微采样的关键技术
- KL Divergence: 约束策略不偏离参考模型
- [[Speech Tokenizer]]: DiffRO 优化的目标对象(token 选择)

## 演进

RLHF for NLP (2022) → RL for TTS on audio (Seed-TTS, 2024) → Token-level differentiable optimization / DiffRO (CosyVoice 3, 2025)
