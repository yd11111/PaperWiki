---
type: concept
title: "Codec Language Model"
aliases: [CodecLM, Codec LM, 编解码器语言模型, Neural Codec Language Model, Audio Codec LM]
category: "model-family"
tags: [speech-LM, codec, autoregressive, neural-audio-codec, language-model, RVQ]
key_papers: ["VALL-E (Wang et al., 2023)", "AudioLM (Borsos et al., 2023)", "VioLA (Wang et al., 2024)", "NTPP (Wang et al., 2025)", "SpeechGPT-Gen (Zhang et al., 2024)"]
origin_paper: "Cui et al., Speech Language Models, 2024"
related_concepts: ["[[Speech Language Model]]", "[[LLM-based TTS]]", "[[Residual Vector Quantization]]", "[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Codec Language Model (CodecLM) 是 Speech Language Model 的一个子范式,特指直接在 neural audio codec (如 EnCodec, SoundStream) 产生的离散 acoustic tokens 上训练语言模型进行语音建模和生成。与使用 semantic tokens 的 SpeechLM 不同,CodecLM 直接操作编解码器的量化码本索引。

Survey (Cui et al., 2024) 明确指出: "Some studies directly model the codec tokens in a language model, which is often regarded as Codec Language Models (CodecLMs)."

**核心特征**:
- 输入/输出均为 neural codec 的 RVQ 码本索引
- 语言模型直接在这些 acoustic tokens 上做 next-token prediction
- 可用于 ASR、TTS、语音翻译等多任务

## 与 LLM-based TTS 的区别

| 维度 | Codec Language Model | LLM-based TTS |
|------|---------------------|---------------|
| 范围 | 通用语音建模 (ASR+TTS+ST+对话) | 专注于 text→speech 生成 |
| Token 类型 | 必须使用 acoustic (codec) tokens | 可用 semantic 或 codec tokens |
| 代表系统 | VioLA (多任务), NTPP (双通道对话) | VALL-E (TTS), SpearTTS (TTS) |
| 能力 | 理解 + 生成 | 主要是生成 |

简言之: LLM-based TTS 是 TTS 领域的范式,CodecLM 是 SpeechLM 领域基于 codec token 的建模范式。两者在 VALL-E 等系统上有交集,但各自外延不同。

## 代表系统

### VioLA (Wang et al., 2024)
- 在 EnCodec tokens 上训练的 CodecLM
- 同时支持 ASR、TTS、Machine Translation
- 验证了 codec tokens 作为通用语音表征的可行性

### NTPP (Wang et al., 2025)
- 在 VQ-VAE tokens 上训练
- "Next-token-pair prediction" 方法
- 建模双通道 (dual-channel) 口语对话数据
- decoder-only transformer 同时预测两个通道的 token

### SpeechGPT-Gen (Zhang et al., 2024)
- 使用 mixed tokens: 在推理时将 semantic tokens 转换为 acoustic tokens
- Chain-of-information generation 策略
- Scaling chain-of-information 信息生成

### AudioLM (Borsos et al., 2023)
- 两阶段: 先生成 w2v-BERT semantic tokens → 再生成 SoundStream acoustic tokens
- 建立了 semantic → acoustic 的层级 codec 建模范式
- 启发了后续 VALL-E、SoundStorm 等系统

## 关键技术挑战

1. **多层 RVQ 建模**: codec tokens 有多个量化层 (通常 4-8 层), 如何高效建模?
   - AR + NAR 两阶段 (VALL-E)
   - RQ-Transformer 并行 (Moshi)
   - Grouped code 编码 (VALL-E 2)

2. **语义-声学 gap**: 纯 acoustic tokens 语义信息稀疏,cross-modal 对齐困难
   - SpeechTokenizer 第一层蒸馏 HuBERT 语义
   - Mixed token 方案 (Mimi)

3. **序列长度**: 多层 codec token 展开后序列极长
   - Flattening vs interleaving 策略
   - 基于 delay pattern 的 codebook 排列

## 在 Speech LM 中的角色

CodecLM 是 SpeechLM 中侧重 **声学保真度** 的路线。Survey 分类中属于 "Discrete Features → Acoustic Token" 和 "Mixed Token" 路线:
- 纯 acoustic token 路线: VioLA, Li et al., Parrot
- Mixed token 路线: Moshi, SpeechGPT-Gen

多数 SpeechLM 更倾向使用 semantic tokens 而非 codec tokens,因为语义理解是口语交互的核心需求。

## 关键论文

- VALL-E (Wang et al., 2023): 首个大规模 codec LM TTS
- AudioLM (Borsos et al., 2023): semantic → acoustic 层级 codec 建模
- VioLA (Wang et al., 2024): 多任务 CodecLM (ASR+TTS+ST)
- NTPP (Wang et al., 2025): 双通道对话 codec LM
- SpeechGPT-Gen (Zhang et al., 2024): chain-of-information codec 生成

## 相关概念

- [[Speech Language Model]]: CodecLM 是 SpeechLM 的一个子范式
- [[LLM-based TTS]]: CodecLM 在 TTS 上的应用即为 LLM-based TTS 的 codec 路线
- [[Residual Vector Quantization]]: codec tokens 的量化方法
- [[Speech Tokenizer]]: 产生 codec tokens 的 acoustic tokenizer
- [[Semantic vs Acoustic Tokens]]: CodecLM 使用 acoustic 侧 tokens

## 演进

VQ-VAE speech tokens (2019) → SoundStream/EnCodec neural codecs (2021-2023) → AudioLM semantic→acoustic 两阶段 (2022) → VALL-E codec LM for TTS (2023) → VioLA 多任务 CodecLM (2024) → Moshi/SpeechGPT-Gen mixed-token CodecLM (2024)
