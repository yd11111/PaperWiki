---
type: concept
title: "Speaker Embedding"
aliases: [说话人嵌入, Speaker Representation, d-vector, Speaker Encoder, 说话人编码]
category: "representation"
tags: [TTS, multi-speaker, voice-cloning, speaker-identity, adaptive-TTS]
key_papers: []
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Speech Factorization]]", "[[Prosody Modeling]]", "[[Text-to-Speech Pipeline]]", "[[Speech Tokenizer]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Speaker Embedding 是将说话人身份信息编码为固定维度向量的表示方法,用于在多说话人 TTS 系统中控制合成语音的音色特征。它回答 "谁来说 (who to say)" 的问题。

**获取方式**:
1. **Speaker lookup table**: 训练集中每个说话人一个可学习的 embedding (closed-set)
2. **Speaker encoder**: 从参考音频提取 speaker embedding (open-set / zero-shot)

## 两种范式

### Lookup Table (训练集内说话人)
- 每个说话人分配一个可学习的固定维度向量
- 训练时通过 speaker ID 索引
- 优势: 简单高效, 质量高
- 局限: 无法泛化到新说话人

### Speaker Encoder (零样本)
- 从参考音频提取说话人表示
- 训练: 通常在说话人验证任务上预训练
- 优势: 可泛化到未见说话人 (zero-shot)
- 局限: 表示可能不够精确, speaker similarity 低

**常见 speaker encoder 架构**:
- d-vector: DNN 在 speaker verification 上训练后取倒数第二层
- x-vector: TDNN-based, 统计池化层
- GE2E: Generalized end-to-end loss 训练的 encoder
- ECAPA-TDNN: 强 speaker verification 模型, 常用于 TTS

## 在 TTS 中的注入方式

Speaker embedding 注入 TTS 模型的常见方法:

| 方式 | 描述 | 代表工作 |
|------|------|----------|
| Concatenation | 拼接到 encoder/decoder 输入 | DeepVoice 2 |
| Addition | 加到 hidden states | Tacotron 2 multi-speaker |
| Conditional LayerNorm | 生成 LN 的 scale/bias | AdaSpeech |
| FiLM conditioning | $\gamma \cdot h + \beta$ | 各种现代模型 |
| Cross-attention | 作为 key/value | CosyVoice |
| Prefix/Prompt | 作为解码器前缀 | VALL-E, Seed-TTS |

## Adaptive TTS (语音适应)

Survey 定义的自适应 TTS 场景: 用少量目标说话人数据使源模型适应新声音。

### 适应策略

| 策略 | 数据需求 | 参数调整 | 代表工作 |
|------|----------|----------|----------|
| Few-data adaptation | 几分钟 ~ 几秒 | 全模型/部分模型 | Chen et al., Arik et al. |
| Few-parameter adaptation | 数十句 | 仅 speaker embedding / LN | AdaSpeech |
| Untranscribed data | 无转写语音 | 用 ASR 获取文本 | AdaSpeech 2 |
| Zero-shot adaptation | 仅参考音频 | 无微调 | DV3-Clone, SEA-TTS, SV-Tacotron |

### AdaSpeech 系列
- **AdaSpeech**: Conditional LayerNorm 从 speaker embedding 生成 scale/bias, 仅微调 LN 参数
- **AdaSpeech 2**: 利用 mel 重建 + latent alignment 适应无转写数据
- **AdaSpeech 3**: 从阅读风格适应到自发说话风格 (filled pauses, rhythm)

## Zero-shot Voice Cloning

不需要任何微调,仅通过参考音频实现声音克隆:

**传统方案** (Survey 时代):
- Speaker encoder (SV-Tacotron, SEA-TTS): 从参考音频提取 embedding
- 局限: 目标说话人与源说话人差异大时质量下降

**现代方案** (LLM-TTS 时代):
- VALL-E: 3秒 prompt → AR + NAR 生成, in-context learning
- CosyVoice: prompt 音频经 flow matching 提取 timbre
- Seed-TTS: self-distillation 增强 timbre disentanglement

## 关键论文

- DeepVoice 2 (Arik et al., NIPS 2017): 首个 multi-speaker neural TTS (lookup table)
- DeepVoice 3 (Ping et al., ICLR 2018): 可扩展到数千说话人
- SV-Tacotron (Jia et al., NeurIPS 2018): speaker encoder 实现零样本 TTS
- SEA-TTS (Chen et al., ICLR 2019): sample efficient adaptive TTS
- AdaSpeech (Chen et al., ICLR 2021): conditional LN 高效适应
- VALL-E (Wang et al., 2023): in-context learning 重新定义零样本 TTS

## 相关概念

- [[Speech Factorization]]: 将 speaker 信息与 content/prosody 解耦
- [[Prosody Modeling]]: speaker embedding 编码音色, prosody 编码韵律
- [[Speech Tokenizer]]: 在 LLM-TTS 中,prompt token 部分取代 speaker embedding 的功能
- [[Conditional Flow Matching]]: 现代 TTS 中从 speaker prompt 恢复音色

## 演进

Speaker ID one-hot (SPSS) → Learnable speaker embedding (DeepVoice 2, 2017) → Speaker encoder / d-vector (SV-Tacotron, 2018) → Conditional LayerNorm (AdaSpeech, 2021) → In-context prompt (VALL-E, 2023) → Self-distillation timbre disentanglement (Seed-TTS, 2024)
