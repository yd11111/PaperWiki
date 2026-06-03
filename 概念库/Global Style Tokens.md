---
type: concept
title: "Global Style Tokens"
aliases: [GST, 全局风格标记, Style Token Bank, Style Tokens, GST-Tacotron]
category: "technique"
tags: [TTS, style, unsupervised, reference-encoder, attention, expressiveness]
key_papers: ["Wang et al., Style Tokens: Unsupervised style modeling, control and transfer in end-to-end speech synthesis, ICML 2018", "Skerry-Ryan et al., Towards end-to-end prosody transfer for expressive speech synthesis with Tacotron, ICML 2018", "[[论文笔记/SC VALL-E|SC VALL-E]]", "[[论文笔记/Controlling Emotion TTS NL Prompts|Bott et al. (Interspeech 2024)]]"]
origin_paper: "Wang et al., Style Tokens, ICML 2018"
related_concepts: ["[[Style Transfer in TTS]]", "[[Prosody Modeling]]", "[[Speaker Embedding]]", "[[Attention-based TTS]]", "[[Speech Factorization]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Global Style Tokens (GST) 是一种无监督的语音风格表示机制,通过一组可学习的 style token embeddings (token bank) 和基于注意力的参考编码器,从音频中自动发现和编码风格维度,无需显式风格标签。

**核心架构**:
```
Reference Audio → [Reference Encoder (CNN+GRU)] → Query vector
                                                      ↓
Style Token Bank (K learnable tokens) ← [Multi-head Attention] → Style Embedding
                                                      ↓
                                    Injected into TTS decoder
```

## 机制详解

### Reference Encoder
- 输入: mel spectrogram of reference audio
- 架构: 6-layer CNN → GRU → 固定维度 query vector
- 训练时: reference = ground-truth target audio
- 推理时: reference = 任意风格参考音频

### Style Token Bank
- K 个可学习的 embedding vectors (通常 K=10~20)
- 随 TTS 模型端到端训练, 无需风格标注
- 每个 token 自动聚类到不同风格维度

### Attention Mechanism
- Query: reference encoder 输出
- Keys/Values: style token bank
- Output: attention-weighted combination = style embedding
- Multi-head attention 允许同时捕捉多个风格维度

## 训练与推理

**训练**: 纯无监督
- 不需要任何风格标签
- Reference encoder 从 target mel 提取 query
- Style embedding 作为条件注入 decoder
- 通过重建损失自动学习风格因素

**推理**: 多种模式
1. **Reference-based**: 提供参考音频 → 迁移其风格
2. **Manual control**: 直接设置 token weights → 组合风格
3. **Interpolation**: 在 token 之间插值 → 渐变风格
4. **Random sampling**: 随机采样 weights → 生成多样风格

## 风格的自动发现

GST 自动发现的典型风格维度:
- 语速快/慢
- 能量高/低
- 音调升/降
- 情感色彩 (隐式)
- 正式/非正式

这些维度在无监督情况下自然涌现,证明了方法的有效性。

## 局限性

1. **可解释性**: token 学到的维度不一定对应人类可理解的属性
2. **粒度**: 全局 (utterance-level) 风格,缺乏局部控制
3. **纠缠**: speaker 和 style 可能混淆
4. **组合性**: 某些 token 组合可能产生不自然的语音

## 后续发展

GST 作为可控 TTS 的奠基工作,影响了大量后续方法:

| 方向 | 代表工作 | 改进 |
|------|----------|------|
| 多级风格 | MsEmoTTS (2022) | 全局+句子+局部三层 |
| 解耦 speaker/style | An et al. (2022) | 对抗训练分离 |
| Zero-shot | MetaStyleSpeech (2021) | Meta-learning 泛化 |
| 时变风格 | DEX-TTS (2024) | 分离 time-invariant/variant |
| 扩散增强 | StyleTTS-ZS (2024) | 风格扩散模型 |
| LLM 时代 | [[论文笔记/SC VALL-E|SC VALL-E]] (2023) | GST + scale factors 在 LLM TTS 中 |

## 在 TTS 中的应用

- 有声书朗读: 从示例音频迁移朗读风格
- 情感对话: 从情感参考中提取情感风格
- 角色配音: 定义角色特定说话方式
- 风格探索: 通过 token 权重交互探索风格空间

## 关键论文

- Wang et al. (ICML 2018): "Style Tokens" - 提出 GST, 开创无监督风格控制
- Skerry-Ryan et al. (ICML 2018): Prosody transfer - reference encoder 配对工作
- Hsu et al. (ICLR 2019): GMVAE-Tacotron - VAE prior 改进 GST
- SC VALL-E (Kim et al., 2023): GST 思想在 LLM-TTS 中的延续

## 相关概念

- [[Style Transfer in TTS]]: GST 是风格迁移的奠基机制
- [[Prosody Modeling]]: GST 的 style 包含韵律信息
- [[Speaker Embedding]]: 与 GST 的 style embedding 需解耦
- [[Attention-based TTS]]: GST 基于 Tacotron 架构
- [[Speech Factorization]]: GST 后续的解耦改进方向

## 演进

固定风格合成 → **GST (无监督风格发现, ICML 2018)** → VAE 风格先验 (GMVAE, 2019) → 解耦 speaker/style (2022) → 多级时变风格 (DEX-TTS, 2024) → LLM in-context 取代显式 token (2023-)
