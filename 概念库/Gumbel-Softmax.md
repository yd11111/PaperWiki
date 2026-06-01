---
type: concept
title: "Gumbel-Softmax"
aliases: [Gumbel Softmax, Gumbel-Softmax Trick]
category: "optimization-technique"
tags: [differentiable-sampling, discrete-optimization, gradient-estimation]
key_papers: ["[[CosyVoice 3]]"]
related_concepts: ["[[Differentiable Reward Optimization]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Gumbel-Softmax 是一种使离散类别采样过程可微分的技术(又称 Concrete distribution)。通过向 logits 添加 Gumbel 噪声后做 softmax(或 straight-through 变体),可以在前向传播中得到近似 one-hot 的离散样本,同时在反向传播中通过连续松弛传递梯度。

数学形式: y_i = softmax((log(pi_i) + g_i) / tau),其中 g_i ~ Gumbel(0,1),tau 为温度参数。

## 在 TTS 中的应用

在 CosyVoice 3 的 DiffRO 中,Gumbel-Softmax 用于:
- LLM 在每个时间步输出 speech token 的 logits 分布
- 通过 Gumbel-Softmax 采样得到 "soft" token 选择
- 将采样结果送入 Token2Text reward model 计算 reward
- Reward 的梯度通过 Gumbel-Softmax 回传到 LLM 参数

这使得整个 "LLM → token 选择 → reward 计算" 链路完全可微,无需 REINFORCE 等高方差梯度估计器。

## 关键论文

- Jang et al., "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017
- Maddison et al., "The Concrete Distribution", ICLR 2017
- CosyVoice 3 (2025): 在 TTS post-training 中使用

## 相关概念

- Straight-Through Estimator (STE): 另一种离散梯度近似,FSQ 训练时使用
- REINFORCE: 不需要可微路径的梯度估计,但方差大
- [[Differentiable Reward Optimization]]: Gumbel-Softmax 在其中的应用场景

## 演进

REINFORCE (高方差) → Gumbel-Softmax (2017, 低方差但有 bias) → Straight-Through Gumbel (结合两者) → 应用于 TTS token-level RL (CosyVoice 3, 2025)
