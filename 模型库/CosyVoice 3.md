---
type: model
title: "CosyVoice 3"
aliases: [CosyVoice3, CosyVoice 3-0.5B, CosyVoice 3-1.5B]
org: "Alibaba (Tongyi Lab)"
year: 2025
tags: [TTS, zero-shot, multilingual, LLM-based, coarse-to-fine]
key_concepts: ["[[Speech Tokenizer]]", "[[Finite Scalar Quantization]]", "[[Conditional Flow Matching]]", "[[Differentiable Reward Optimization]]"]
tasks: ["[[Zero-shot Speech Synthesis]]", "[[Cross-lingual Voice Cloning]]", "[[Instructed Speech Generation]]"]
key_papers: ["[[论文笔记/CosyVoice 3|CosyVoice 3]]"]
supersedes: ["[[CosyVoice 2]]"]
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

CosyVoice 3 是阿里巴巴通义实验室开发的大规模零样本多语言语音合成模型,面向 in-the-wild 应用场景。采用 coarse-to-fine 两阶段架构(LLM + CFM),通过新型监督式 speech tokenizer、DiffRO post-training、以及数据/模型 scaling 显著超越前代。

支持 9 种语言(zh, en, ja, ko, de, es, fr, it, ru)+ 18 种中文方言,训练数据 100 万小时,模型参数 0.5B/1.5B。

## 核心方法

1. **监督多任务 Speech Tokenizer**: 基于 MinMo 构建,FSQ 量化,25 Hz token rate,530K 小时多任务监督训练
2. **DiffRO Post-training**: Token-level 可微 reward 优化,绕过 CFM/vocoder 直接提升内容一致性
3. **DiT-based CFM**: 300M 参数 Diffusion Transformer 作为声学渲染器
4. **数据 Scaling**: 1M 小时多语言多域数据,6 步 multilingual data pipeline
5. **Pronunciation Inpainting + Self-training TN**: 提升多音字控制和原始文本鲁棒性

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| CER (%) test-zh | 0.71 | SEED-TTS-Eval | Table 4 |
| WER (%) test-en | 1.45 | SEED-TTS-Eval | Table 4 |
| CER (%) test-hard | 5.09 (0.5B_RL) | SEED-TTS-Eval | Table 4 |
| SS (WavLM) test-zh | 0.840 | SEED-TTS-Eval | Table 4 |
| MOS 平均 | 4.45 | 主观评估 | Fig.4 |
| 多语言 9 语种覆盖 | 全部支持 | CV3-Eval | Table 5 |

## 演进线

CosyVoice (2024, FSQ-SenseVoice, 10K h) → CosyVoice 2 (2024, streaming, LLM init, bidirectional) → CosyVoice 3 (2025, MinMo tokenizer, DiffRO, 1M h, 1.5B)

## 关键贡献

- 首个在 TTS 中验证 token-level differentiable reward optimization 的工作
- 首个支持 9 种语言 + 18 种中文方言的开源零样本 TTS
- 提出 CV3-Eval 多语言 benchmark
- 验证了 TTS 领域的 data scaling (10K → 1M h) 和 model scaling (0.5B → 1.5B) 效果
