---
type: concept
title: "Gumbel-Softmax"
aliases: [Gumbel Softmax, Gumbel-Softmax Trick, Concrete Distribution]
category: "optimization-technique"
tags: [differentiable-sampling, discrete-optimization, gradient-estimation, reparameterization]
key_papers: ["[[论文笔记/DiffRO|DiffRO]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[论文笔记/wav2vec2.0|wav2vec 2.0]]", "[[论文笔记/NAST|NAST]]", "[[论文笔记/Dict-TTS|Dict-TTS]]"]
origin_paper: "Jang et al., Categorical Reparameterization with Gumbel-Softmax, ICLR 2017"
related_concepts: ["[[DifferentiableRewardOptimization]]", "[[FiniteScalarQuantization]]", "[[ResidualVectorQuantization]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Gumbel-Softmax 是一种使离散类别采样可微分的重参数化技巧,解决的核心问题:**神经网络需要连续可微操作来反向传播,但从 categorical distribution 采样是不可微的。**

### 原理(为什么 work)

三步推导:

1. **Gumbel-Max Trick**(背景）: 从 categorical distribution 采样等价于: 对每个类别 i,计算 `log(π_i) + g_i`(g_i ~ Gumbel(0,1)),取 argmax。Gumbel 噪声的特殊性质保证 argmax 结果服从原始 categorical 分布。

2. **问题**: argmax 不可微 → 梯度无法回传。

3. **Gumbel-Softmax 的核心 insight**: 用 softmax 替换 argmax 作为连续松弛:

```
y_i = exp((log(π_i) + g_i) / τ) / Σ_j exp((log(π_j) + g_j) / τ)
```

**为什么这是合理的**: categorical 样本住在 simplex 的顶点(one-hot),softmax 输出住在同一 simplex 的内部。通过在内部(可微、近似）和顶点(不可微、精确)之间平滑插值,实现可微训练。

### 温度参数 τ

控制"承诺程度":
- τ → 0: softmax 趋近 argmax,输出接近 one-hot(精确但梯度消失)
- τ 大: 输出平滑(梯度好但近似差)
- 实践: 训练过程中 anneal τ 从大到小,但不完全到 0

### Straight-Through 变体

前向用 argmax(真正离散),反向用 softmax 梯度近似。兼顾推理时的离散性和训练时的梯度流。

## 在 TTS 中的应用

在 CosyVoice 3 的 DiffRO(Differentiable Reward Optimization)中:
- LLM 每步输出 speech token 的 logits
- Gumbel-Softmax 采样得到 "soft" token 选择
- soft token 送入 Token2Text reward model 计算 reward
- Reward 梯度通过 Gumbel-Softmax 回传到 LLM

这使 "LLM → token 选择 → reward" 链路完全可微,无需 REINFORCE(高方差)。

**更广泛的 TTS 应用场景**: 任何需要在离散 token 空间做端到端优化的场景:
- 离散 codebook 选择的可微训练
- VQ/RVQ 中的 soft assignment
- 离散语音 token 的 RL/reward-based 优化

### 在 SSL 语音预训练中的应用 (wav2vec 2.0)

wav2vec 2.0 (Baevski et al., NeurIPS 2020) 使用 Gumbel-Softmax 实现端到端的 speech token 离散化 [§2]:
- Feature encoder 输出映射到 $G \times V$ 个 logits (Product Quantization: G=2, V=320)
- Gumbel-Softmax 选择每组最可能的 codebook entry,temperature $\tau$ 从 2 退火至 0.5 [§4.2]
- Straight-Through 变体: 前向 argmax (真正离散),反向 Gumbel-Softmax 梯度 [§2]
- 配合 diversity loss 最大化 codebook 使用熵,防止 codebook collapse [§3.2]

[agent 解读] wav2vec 2.0 是 Gumbel-Softmax 在语音 SSL 中的里程碑应用;后续 HuBERT 用离线 k-means 替代了它,避免了温度退火等超参数,但丧失了端到端可微性

### 在多音字消歧中的应用 (Dict-TTS)

[[论文笔记/Dict-TTS|Dict-TTS]] (Jiang et al., NeurIPS 2022) 将 Gumbel-Softmax 用于 TTS 前端的多音字(polyphone)消歧:
- 每个多音字有 m 个候选发音,S2PA 模块计算各发音的语义匹配权重 w_{i,j}
- Gumbel-Softmax 对权重采样,近似选择最可能的发音: w̃ = softmax((log(w) + g) / τ) [§3.3, Eq. 2-3]
- 温度 τ 按 Jang et al. 2017 的策略退火,训练时从大到小
- 消融实验证实 Gumbel-Softmax 优于直接 softmax 加权: PER-S 从 1.19%→1.08%, SER-S 从 7.75%→6.50% [Table 4]

[agent 解读] Dict-TTS 是 Gumbel-Softmax 在 TTS 中的早期应用(2022),在时间线上早于 CosyVoice 3 的 DiffRO(2025)。应用场景不同: Dict-TTS 用于前端离散发音选择,DiffRO 用于后训练阶段的 token-level reward 优化

## 关键论文

- Jang et al., "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017 — 原始论文
- Maddison et al., "The Concrete Distribution", ICLR 2017 — 独立同期工作,相同方法
- wav2vec 2.0 (Baevski et al., NeurIPS 2020): 在 SSL 语音预训练中使用 Gumbel-Softmax PQ 实现端到端离散化
- CosyVoice 3 (2025): 在 TTS post-training (DiffRO) 中使用,实现 token-level 可微 reward 优化

## 相关概念

- [[DifferentiableRewardOptimization]]: CosyVoice 3 中 Gumbel-Softmax 的应用场景
- [[FiniteScalarQuantization]]: 另一种离散化方案,训练时用 STE 而非 Gumbel-Softmax
- REINFORCE: 不需要可微路径的替代方案,但方差高
- Straight-Through Estimator (STE): 类似思路,但不加 Gumbel 噪声

## 演进

REINFORCE (高方差, 1992) → Gumbel-Softmax / Concrete (低方差可微, 2017) → ST-Gumbel (结合离散前向+可微反向) → 应用于 SSL 语音离散化 (wav2vec 2.0, 2020) → 应用于 TTS token-level RL (CosyVoice 3, 2025)

---

> [!info] 来源
> 定义和原理部分基于 Jang et al. 2017 原始论文 + Eric Jang 博客 (blog.evjang.com, 2016)。TTS 应用部分基于 CosyVoice 3 论文。
