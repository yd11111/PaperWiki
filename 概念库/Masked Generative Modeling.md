---
type: concept
title: "Masked Generative Modeling"
aliases: [Masked Generative Transformer, Mask-and-Predict, Non-autoregressive Masked Generation, MaskGIT-style Generation]
category: "generative-model"
tags: [generative-model, non-autoregressive, discrete-token, parallel-decoding]
key_papers: ["[[论文笔记/SoundStorm|SoundStorm]]", "[[论文笔记/NaturalSpeech 3|NaturalSpeech 3]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Chatterbox-Flash|Chatterbox-Flash]]", "[[论文笔记/DiSTAR|DiSTAR]]"]
origin_paper: "Chang et al., MaskGIT: Masked Generative Image Transformer, CVPR 2022"
related_concepts: ["[[Speech Tokenizer]]", "[[Residual Vector Quantization]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Masked Generative Modeling 是一类基于 mask-and-predict 范式的非自回归生成方法。训练时随机 mask 输入离散 token 序列的一部分,模型学习预测被 mask 的 token;推理时从全 mask 状态出发,通过迭代并行解码逐步填充所有 token。

### 核心机制

给定离散序列 X,定义 masked 版本 X_t = X * (1 - M_t),其中 M_t 是 Bernoulli mask (mask 比例由 schedule gamma(t) 控制)。模型训练目标为最小化被 mask token 的负对数似然:

$$\mathcal{L}_\text{mask} = -\sum_{i=1}^N m_{t,i} \cdot \log(p_\theta(x_i | X_t, C))$$

### 推理: 迭代并行解码

1. 从全 mask 序列 X_T 出发,共 S 步
2. 每步: 预测所有 mask 位置 → 取 confidence 最高的一部分 unmask → 将低 confidence 部分 remask
3. 每步 unmask 的数量由 schedule 递减: floor(N * gamma(T - i * T/S))
4. 可加 Gumbel noise 到 confidence 增加多样性

### 与 AR 模型的核心区别

| | Autoregressive | Masked Generative |
|---|---|---|
| Attention | Causal (单向) | Bidirectional (双向) |
| 生成顺序 | 固定 (left-to-right) | 灵活 (confidence-based) |
| 推理步数 | O(N) (序列长度) | O(S) (固定步数, 通常 10-50) |
| 长度控制 | 困难 (需 stop token) | 天然支持 (指定 mask 数量) |
| 鲁棒性 | error accumulation | 每步可全局修正 |

## 在 TTS 中的应用

MaskGCT 首次将 masked generative modeling 应用于 TTS 的 text-to-semantic 阶段 (此前 SoundStorm 仅用于 semantic-to-acoustic 阶段):
- T2S: text + prompt semantic tokens 作为 unmasked prefix, 目标 semantic tokens 被 mask
- S2A: 按 RVQ layer 逐层生成, 每层内使用 iterative parallel decoding
- 无需 text-speech alignment 或 phone-level duration prediction

## 关键论文

- Chang et al., "MaskGIT: Masked Generative Image Transformer", CVPR 2022 — 原始方法 (图像领域)
- Borsos et al., "SoundStorm: Efficient Parallel Audio Generation", 2023 — 首次用于音频 (acoustic token 生成)
- MaskGCT (Wang et al., 2024) — 首次完整应用于 TTS 两阶段 (T2S + S2A)

## 相关概念

- [[Speech Tokenizer]]: masked generative modeling 要求离散 token 作为输入/输出
- [[Residual Vector Quantization]]: S2A 阶段按 RVQ 层逐层生成
- Diffusion Model: 连续空间的类似迭代精化思路 (masked generative 是离散空间的对应物)

## 演进

BERT (2019, masked LM for understanding) → MaskGIT (2022, masked generation for images) → SoundStorm (2023, acoustic token generation) → MaskGCT (2024, full TTS pipeline) → Block Diffusion (Arriola et al., 2025) + Chatterbox-Flash (2026, block-causal masked denoising for streaming TTS with prior-calibrated scoring)
