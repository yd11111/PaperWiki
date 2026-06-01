---
type: concept
title: "Residual Vector Quantization"
aliases: [RVQ, Residual VQ, Multi-stage VQ]
category: "quantization"
tags: [quantization, discrete-representation, audio-codec, neural-compression]
key_papers: ["[[论文笔记/DAC|DAC]]", "[[论文笔记/MaskGCT|MaskGCT]]"]
origin_paper: ""
related_concepts: ["[[Finite Scalar Quantization]]", "[[Codebook Collapse]]", "[[Quantizer Dropout]]", "[[Speech Tokenizer]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Residual Vector Quantization (RVQ) 是一种多层级向量量化方法,通过递归量化残差来逐步逼近输入向量。第一层 quantizer 量化原始输入,后续每层量化前一层的残差 (residual),最终表示为多层 codebook index 的组合。

数学形式: 给定输入 z, 第 i 层量化残差 r_i:
- r_1 = z
- q_i = Quantize(r_i)  (从 codebook_i 查找最近邻)
- r_{i+1} = r_i - q_i

最终重建: z_hat = sum(q_1, q_2, ..., q_N)

## 在 Audio Codec 中的应用

RVQ 是现代 neural audio codec (SoundStream, EnCodec, DAC) 的核心量化模块:
- 将 encoder 输出的连续 latent 压缩为 N_q 层离散 code
- 每层 codebook 通常 10-bit (1024 entries)
- 总比特率 = frame_rate x N_q x bits_per_code

DAC 的改进 [§3.2]: 使用 factorized codes (低维 8d lookup) + L2-normalization, 将 codebook utilization 从 ~90% 提升到 ~99%, 有效解决 codebook collapse 问题。

## 关键特性

1. **层级信息结构**: 前面的层编码 coarse 信息, 后面的层编码 fine details — 天然适合 hierarchical generation (如 AudioLM 的 coarse/fine tokens)
2. **可变比特率**: 通过使用不同数量的层 (1...N_q) 实现运行时 bitrate 控制
3. **与 VQ 的区别**: 单层 VQ 的 codebook 需要指数级大小才能覆盖高维空间; RVQ 通过残差分解, 用 N 个小 codebook 组合表达能力

## 关键论文

- Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec", 2021: 首次将 RVQ 用于端到端 audio codec
- Defossez et al., "EnCodec: High Fidelity Neural Audio Compression", 2022: 改进 RVQ 训练 (EMA codebook)
- DAC (Kumar et al., NeurIPS 2023): factorized codes + L2-norm 解决 codebook collapse, bitrate efficiency 达 99%
- Mentzer et al., "Finite Scalar Quantization", ICLR 2024: 提出无需码本的替代方案 FSQ
- MaskGCT (Wang et al., 2024): 在 acoustic codec 中使用 12 层 RVQ (codebook size 1024, dim 8),配合 Vocos decoder 和 S2A masked generative model 逐层生成 acoustic tokens

## 相关概念

- [[Finite Scalar Quantization]]: VQ/RVQ 的替代方案, 无需显式码本
- [[Codebook Collapse]]: RVQ 训练的主要难点
- [[Quantizer Dropout]]: 实现 RVQ 可变比特率的训练技巧
- [[Speech Tokenizer]]: RVQ-based codec 可作为声学 tokenizer 用于 TTS

## 演进

VQ-VAE (2017) → RVQ/SoundStream (2021) → EnCodec (2022, EMA codebook) → DAC (2023, factorized codes) → FSQ (2024, 去码本化)
