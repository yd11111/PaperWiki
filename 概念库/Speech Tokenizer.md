---
type: concept
title: "Speech Tokenizer"
aliases: [语音分词器, Semantic Token, Discrete Speech Token]
category: "representation"
tags: [speech-representation, tokenization, discrete-token, TTS]
key_papers: ["[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/IndexTTS2|IndexTTS2]]"]
related_concepts: ["[[Finite Scalar Quantization]]", "[[Conditional Flow Matching]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Speech Tokenizer 将连续语音波形转换为离散 token 序列的模块,是 LLM-based TTS 系统的关键组件。根据训练方式可分为三类:

1. **自监督 tokenizer**: 如 HuBERT、W2v-BERT 2.0,通过 masked prediction 学习表征后做 k-means 聚类
2. **监督式 semantic tokenizer**: 如 CosyVoice 系列,通过 ASR 等下游任务监督训练,token 主要编码语义/语言信息
3. **声学 tokenizer**: 如 SoundStream、EnCodec,通过 RVQ 重建波形,token 编码全部声学信息

## 在 TTS 中的应用

在 coarse-to-fine TTS pipeline 中,semantic speech tokenizer 处于核心位置:
- **编码时**: 将参考语音 → token 序列(训练目标)
- **解码时**: LLM 生成 token 序列 → CFM/vocoder 合成语音

CosyVoice 3 的 speech tokenizer 基于 MinMo 构建,通过 FSQ 量化,以 25 Hz token rate 工作。其监督多任务训练(ASR + LID + SER + AED + SA,共 530K 小时)使 token 富含副语言信息,同时排除底层声学细节,让 CFM 可以独立控制音色。

关键 trade-off: token 编码越多语义信息(排除声学)→ speaker similarity 越高(CFM 从 prompt 学音色)+ content consistency 越好;但也可能丢失韵律细节。

## 关键论文

- CosyVoice 3 (2025): 监督多任务 FSQ-MinMo tokenizer
- CosyVoice (2024): FSQ-SenseVoice tokenizer
- HuBERT (2021): 自监督 speech representation
- SoundStream (2021): 声学 RVQ tokenizer
- IndexTTS2 (Zhou et al., 2025): 采用 MaskGCT 的 semantic codec 作为 speech tokenizer,在 T2S 模块中生成 semantic token 序列,并通过共享位置编码表(W_sem = W_num)实现精确 duration control

## 相关概念

- [[Finite Scalar Quantization]]: CosyVoice 系列使用的量化方法
- [[Conditional Flow Matching]]: tokenizer 的下游,从 token 恢复声学细节
- BPE Text Tokenizer: 文本侧的 tokenizer,speech tokenizer 是其语音对应物

## 演进

Mel spectrogram (传统 TTS) → VQ-VAE acoustic tokens (2019) → HuBERT semantic tokens (2021) → 监督式 semantic tokens (CosyVoice, 2024) → 多任务监督 + 大模型 backbone (CosyVoice 3, 2025)
