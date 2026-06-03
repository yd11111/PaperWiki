---
type: model
title: "CosyVoice 2"
aliases: [CosyVoice2]
org: "Alibaba (Tongyi Lab)"
year: 2024
tags: [TTS, zero-shot, streaming, LLM-based, coarse-to-fine]
key_concepts: ["[[Speech Tokenizer]]", "[[Finite Scalar Quantization]]", "[[Conditional Flow Matching]]"]
tasks: ["[[Zero-shot Speech Synthesis]]", "[[Instructed Speech Generation]]"]
key_papers: ["[[论文笔记/CosyVoice 2|CosyVoice 2]]", "[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/IndexTTS2|IndexTTS2]]"]
supersedes: ["[[模型库/CosyVoice|CosyVoice]]"]
superseded_by: ["[[模型库/CosyVoice 3|CosyVoice 3]]"]
status: confirmed
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

CosyVoice 2 是阿里巴巴通义实验室开发的可扩展流式语音合成模型,集成 LLM 和 chunk-aware flow matching 模型,实现低延迟双向流式合成且质量接近人类水平。主要面向中英文场景。

## 核心方法

1. **FSQ-SenseVoice tokenizer**: 将 FSQ 插入 SenseVoice-Large ASR 编码器
2. **Text-based LLM 初始化**: 利用文本 LLM 的语言知识
3. **双向流式方案**: 实现超低延迟且几乎无损的流式合成
4. **统一指令能力建模**: instruction-following 支持

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| CER (%) test-zh | 1.45 | SEED-TTS-Eval | CosyVoice 3 Table 4 |
| WER (%) test-en | 2.57 | SEED-TTS-Eval | CosyVoice 3 Table 4 |
| MOS 平均 | 4.36 | 主观评估 | CosyVoice 3 Fig.4 |

## 演进线

CosyVoice (2024) → CosyVoice 2 (2024, streaming + instruction) → [[论文笔记/CosyVoice 3|CosyVoice 3]] (2025)

## 关键贡献

- 首个实现几乎无损双向流式零样本 TTS 的系统
- 验证了 text-based LLM 初始化对 TTS LM 的有效性

## 作为 Baseline 被引用

- [[论文笔记/IndexTTS2|IndexTTS2]] (2025): 在基础性能和情感表达两方面均以 CosyVoice2 为 baseline,IndexTTS2 在 SS 和 WER 上全面超越 CosyVoice2;在自然语言情感控制对比中,IndexTTS2 的 EMOS (3.786) 显著优于 CosyVoice2 (3.339)
- [[论文笔记/NVSpeech|NVSpeech]] (2025): 以 CosyVoice2 为 TTS backbone,通过词表扩展+微调添加 18 类副语言发声的显式控制能力;微调后 CER_w/o_para 3.73% (in-domain),listener win rate 75.4% (vs pre-trained)
