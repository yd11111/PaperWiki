---
type: concept
title: "LLM-based TTS"
aliases: [LLM TTS, 大语言模型语音合成, Codec Language Model TTS, Neural Codec LM, LLM-driven Speech Synthesis]
category: "model-family"
tags: [TTS, LLM, autoregressive, codec, zero-shot, in-context-learning, decoder-only]
key_papers: ["VALL-E (Wang et al., 2023)", "VALL-E X (Zhang et al., 2023d)", "VALL-E 2 (Chen et al., 2024a)", "VALL-E R (Han et al., 2024)", "ELLA-V (Song et al., 2024)", "RALL-E (Xin et al., 2024)", "MELLE (Meng et al., 2024)", "HALL-E (Nishimura et al., 2024)", "SpearTTS (Kharitonov et al., 2023)", "Make-a-Voice (Huang et al., 2023b)"]
origin_paper: "Xie et al., Controllable TTS in LLM Era, 2024"
related_concepts: ["[[Speech Tokenizer]]", "[[Residual Vector Quantization]]", "[[Non-autoregressive TTS]]", "[[Speaker Embedding]]", "[[Conditional Flow Matching]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

LLM-based TTS 是受大语言模型 in-context learning 成功启发的语音合成范式。其核心思想是将 TTS 重构为条件语言建模任务: 将语音离散化为 token 序列,使用自回归 decoder-only transformer 生成语音 token,再解码为波形。

**典型架构** (如 Fig. 2 in survey):
```
Text + Instruction → [Text Encoder] → Text Tokens
Reference Speech → [Speech Encoder] → Audio Tokens (prompt)
[Decoder-only Transformer]: Text Tokens + Audio Prompt → Generated Audio Tokens
Audio Tokens → [Speech Decoder / Vocoder] → Waveform
```

**与传统 TTS 的本质区别**:
- 传统 TTS: 显式 duration/pitch/energy predictor → 确定性映射
- LLM-based TTS: 隐式建模所有 variation → in-context learning 从 prompt 推断

## 核心设计

### 1. 离散语音表示 (Discrete Speech Tokens)

LLM-based TTS 将连续语音信号离散化为 token,使其可被 LM 建模:
- **Neural codec tokens**: EnCodec (Defossez et al., 2023a), SoundStream, HiFi-Codec
- **Semantic tokens**: HuBERT/wav2vec2 离散化的自监督表示
- **Hierarchical tokens**: 粗粒度 semantic + 细粒度 acoustic (VALL-E 的 AR + NAR)

### 2. 两阶段生成 (Two-stage Pipeline)

VALL-E 开创的典型两阶段:
1. **AR 阶段**: 自回归生成第一层 coarse codec tokens (建模语义和韵律)
2. **NAR 阶段**: 非自回归生成剩余层 fine codec tokens (补充声学细节)

### 3. In-context Learning for Style

通过 prompt speech 实现零样本控制:
- 将参考语音的 codec tokens 作为 prefix
- 模型从 prefix 中隐式学习说话人、风格、情感
- 不需要显式 speaker embedding 或 style label

## VALL-E 系列演进

| 模型 | 年份 | 关键改进 |
|------|------|----------|
| VALL-E | 2023 | 开创 codec LM TTS, AR+NAR 两阶段 |
| VALL-E X | 2023 | 跨语言零样本 TTS |
| VALL-E 2 | 2024 | 重复感知采样 + 分组编码, 人类水平 |
| VALL-E R | 2024 | 单调对齐增强鲁棒性 |
| ELLA-V | 2024 | 对齐引导序列重排序 |
| RALL-E | 2024 | Chain-of-thought prompting |
| MELLE | 2024 | 连续值 mel 预测替代离散 token |
| HALL-E | 2024 | 分层 codec LM, 长语音合成 |

## 其他 LLM-based 方法

**Semantic token 路线**:
- SpearTTS (Kharitonov et al., 2023): semantic → acoustic 两阶段
- Make-a-Voice (Huang et al., 2023b): 离散表示统一语音合成

**Tokenizer 优化路线**:
- FireRedTTS (Guo et al., 2024a): 优化 tokenizer 架构改善重建质量
- CoFi-Speech (Guo et al., 2024b): 粗到细多尺度生成策略

**Hybrid 架构** (LLM + Flow/Diffusion):
- CosyVoice (Du et al., 2024): LLM 生成 semantic tokens + CFM 合成高保真语音
- NaturalSpeech 3 (Ju et al., 2024): factorized diffusion codec
- SimpleSpeech (Yang et al., 2024c): scalar latent transformer + flow-based

## 在可控性方面的特点

**优势**:
- 自然语言驱动控制: 通过 instruction prompt 直接指定风格
- 零样本声音克隆: 仅需几秒参考音频
- 上下文感知: 理解并生成语义一致的语音

**局限**:
- 离散 token 的量化损失 → 声学瑕疵
- 高计算成本: 长序列自回归推理慢
- 细粒度控制困难: 难以精确控制 pitch/energy/duration
- 稳定性问题: 可能出现 word skip/repeat

## 研究趋势

Survey 指出 TTS 模型架构演进路线:
```
Traditional CNN/RNN (Tacotron, DeepVoice) → 有限控制力
    ↓
Flow-based (Matcha-TTS, F5-TTS) → 快+少量控制
    ↓
LLM-based (VALL-E, InstructTTS) → 慢+自然语言控制
    ↓
Hybrid (CosyVoice) → 直觉控制 + 高保真
    ↓
Future: Instruction-Aware Frameworks → 精细指令控制
```

## 关键论文

- VALL-E (Wang et al., 2023a): 开创 neural codec language model TTS
- SpearTTS (Kharitonov et al., 2023): semantic + acoustic 两阶段 LM
- CosyVoice (Du et al., 2024): LLM + flow matching hybrid
- VoxInstruct (Zhou et al., 2024): instruction-to-speech 统一框架
- InstructSpeech (Huang et al., 2024a): multi-task LLM 语音编辑

## 相关概念

- [[Speech Tokenizer]]: LLM-based TTS 的前提, 将语音离散化
- [[Residual Vector Quantization]]: codec token 的主要量化方法
- [[Non-autoregressive TTS]]: LLM-based TTS 中 NAR 阶段 / 对比范式
- [[Conditional Flow Matching]]: hybrid 架构中的声学生成器
- [[Speaker Embedding]]: 被 in-context prompt 部分取代

## 演进

Tacotron/FastSpeech (显式 variance predictor, 2017-2020) → VALL-E (codec LM, AR+NAR, 2023) → VALL-E 2 / RALL-E (鲁棒性增强, 2024) → Hybrid CosyVoice (LLM + Flow, 2024) → Instruction-aware 架构 (VoxInstruct, Step-Audio, 2024-2025)
