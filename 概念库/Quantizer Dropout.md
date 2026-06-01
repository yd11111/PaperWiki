---
type: concept
title: "Quantizer Dropout"
aliases: [RVQ Dropout, Variable Bitrate Training]
category: "training-technique"
tags: [quantization, training-trick, variable-bitrate, audio-codec]
key_papers: ["[[论文笔记/SoundStream|SoundStream]]", "[[论文笔记/DAC|DAC]]"]
origin_paper: ""
related_concepts: ["[[Residual Vector Quantization]]", "[[Codebook Collapse]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Quantizer Dropout 是一种 RVQ 训练技术,通过在训练时随机使用前 n 层 (n < N_q) quantizer 来使模型支持可变比特率推理。目标是训练一个单一模型,在推理时可根据带宽条件选择使用不同数量的 quantizer 层。

## 原始方案 (SoundStream)

SoundStream [Zeghidour et al., 2021] 提出: 对每个训练样本, 随机采样 n ~ Uniform{1, 2, ..., N_q}, 只使用前 n 个 quantizer。

**问题**: 这导致模型在全带宽 (使用所有 N_q 层) 时质量下降, 因为训练时全带宽只出现 1/N_q 的概率 [Fig 2]。

## DAC 的改进方案 [§3.3]

以概率 p 对每个样本决定是否 apply quantizer dropout:
- 概率 p: 执行 dropout (随机选择 n 层)
- 概率 1-p: 使用全部 N_q 层 (no dropout)

DAC 发现 p=0.5 是最优平衡点:
- 在低比特率时保留可变比特率能力 (接近 always-dropout 的低比特率质量)
- 在全带宽时接近 no-dropout 的质量

## 交互效应

Quantizer dropout 与 factorized codes 协同: 使 quantized codes 学到 most-significant → least-significant 的层级信息结构, 前面的层编码 coarse 信息, 后面的层添加 fine detail [§3.3]。这对 hierarchical generation (如 AudioLM 的 coarse/fine token 分区) 非常有利。

## 关键论文

- Zeghidour et al., "SoundStream", 2021: 提出原始 quantizer dropout
- DAC (Kumar et al., NeurIPS 2023): 发现全比特率质量下降问题, 提出概率化 dropout 方案 (p=0.5)

## 相关概念

- [[Residual Vector Quantization]]: quantizer dropout 的作用对象
- [[Codebook Collapse]]: dropout 会影响 codebook 利用率
- Variable bitrate: quantizer dropout 实现的目标能力
