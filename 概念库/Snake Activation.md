---
type: concept
title: "Snake Activation"
aliases: [Snake Function, Periodic Activation]
category: "architecture-component"
tags: [activation-function, periodic-inductive-bias, waveform-generation, vocoder]
key_papers: ["[[论文笔记/DAC|DAC]]"]
origin_paper: ""
related_concepts: ["[[Multi-scale STFT Discriminator]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Snake Activation 是一种周期性激活函数,为神经网络引入周期 inductive bias:

$$\text{snake}(x) = x + \frac{1}{\alpha}\sin^2(\alpha x)$$

其中 alpha 控制周期成分的频率。当 alpha 大时,函数更接近 identity + 高频振荡; alpha 小时,周期成分更显著。

## 为什么对音频有效

1. **音频信号本质是周期的**: voiced speech (基频 + 谐波)、乐器音等都有强周期性
2. **Leaky ReLU 的问题**: 非周期激活函数难以外推周期结构, 导致 pitch artifacts 和 periodicity artifacts
3. **直接编码周期性 prior**: Snake 让网络更容易学习和生成周期波形, 无需从 data 中完全学习周期性

## 在音频模型中的应用

- **BigVGAN** (Lee et al., 2023): 首次将 Snake activation 引入 neural vocoding, 替换 HiFi-GAN 中的 Leaky ReLU
- **DAC** (Kumar et al., NeurIPS 2023): 在 audio codec 的 decoder 中使用, SI-SDR 从 6.92 (relu) 提升到 9.12 (snake) [Table 2]
- 几乎零额外计算开销的显著质量提升

## 关键论文

- Liu et al., "Neural Networks Fail to Learn Periodic Functions and How to Fix It", 2020: 提出 Snake activation [47]
- Lee et al., "BigVGAN", 2023: 引入音频领域
- DAC (Kumar et al., NeurIPS 2023): 在 codec 中验证, 提供 ablation 证据

## 相关概念

- Leaky ReLU: Snake 替换的对象
- Periodic inductive bias: Snake 编码的先验
- [[Multi-scale STFT Discriminator]]: 与 Snake 配合的判别器
- BigVGAN: 首次在音频中使用 Snake 的模型

## 演进

ReLU → Leaky ReLU (GAN 标配) → Snake (2020, 周期性) → BigVGAN/DAC 验证在音频领域的有效性 (2023)
