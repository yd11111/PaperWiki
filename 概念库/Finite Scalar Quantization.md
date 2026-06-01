---
type: concept
title: "Finite Scalar Quantization"
aliases: [FSQ]
category: "quantization"
tags: [quantization, discrete-representation, VQ-alternative]
key_papers: ["[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/DAC|DAC]]"]
related_concepts: ["[[Speech Tokenizer]]", "[[Gumbel-Softmax]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Finite Scalar Quantization (FSQ) 是一种向量量化替代方案,将连续表征的每个维度独立量化到有限整数集合 [-K, K] 中,通过 bounded round 操作实现。与 VQ-VAE 的 codebook lookup 不同,FSQ 不需要维护显式码本,避免了码本坍缩(codebook collapse)问题。

具体流程:先将高维表征投影到 D 维低秩空间,对每维做 ROUND 量化,再投影回原始维度。最终 token index 通过 (2K+1) 进制编码计算,codebook 大小为 (2K+1)^D。

## 在 TTS 中的应用

在 CosyVoice 系列中,FSQ 被插入到语音编码器的中间层,将连续语音表征离散化为 speech token。CosyVoice 2 将 FSQ 插入 SenseVoice-Large 编码器;CosyVoice 3 改为插入 MinMo 的 Voice Encoder,并通过多任务监督训练优化量化表征的语义信息含量。

训练时使用 straight-through estimation 近似 FSQ 模块的梯度。

## 关键论文

- Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", ICLR 2024
- CosyVoice 3 (2025): 在 MinMo 中使用 FSQ 构建监督式 speech tokenizer

## 相关概念

- Vector Quantization (VQ): FSQ 的前身,需显式码本
- [[Residual Vector Quantization]]: 多层级 VQ,用于声学 codec(如 SoundStream, EnCodec, DAC)
- [[Speech Tokenizer]]: FSQ 是其量化核心模块
- [[Gumbel-Softmax]]: DiffRO 中用于使离散采样可微

## 演进

VQ-VAE (2017) → RVQ/SoundStream (2021) → FSQ (2024, 去码本化) → FSQ + 监督多任务 (CosyVoice 3, 2025)
