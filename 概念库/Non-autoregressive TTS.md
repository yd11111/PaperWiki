---
type: concept
title: "Non-autoregressive TTS"
aliases: [NAR TTS, FastSpeech系列, 非自回归TTS, Parallel TTS, 并行语音合成]
category: "model-family"
tags: [TTS, non-autoregressive, parallel, fast-inference, acoustic-model]
key_papers: ["[[论文笔记/VITS|VITS]]", "[[论文笔记/E2 TTS|E2 TTS]]", "[[论文笔记/OmniVoice|OmniVoice]]", "[[论文笔记/LLaDA-TTS|LLaDA-TTS]]", "[[论文笔记/SESD|SESD]]", "[[论文笔记/VoiceFlow|VoiceFlow]]", "[[论文笔记/Meta Learning TTS 7000 Languages|Meta Learning TTS 7000 Languages]]", "[[论文笔记/Very Attentive Tacotron|Very Attentive Tacotron (Battenberg et al., 2025)]]", "[[论文笔记/PROEMO|PROEMO]]", "[[论文笔记/RADKA-CSS|RADKA-CSS]]", "[[论文笔记/Low-Resource ForwardTacotron|Low-Resource ForwardTacotron (Kayyar et al., 2025)]]", "[[论文笔记/FlexSpeech|FlexSpeech]]", "[[论文笔记/OpenOmni|OpenOmni]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Attention-based TTS]]", "[[Duration Predictor]]", "[[Mel Spectrogram]]", "[[Text-to-Speech Pipeline]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Non-autoregressive (NAR) TTS 是一类能并行生成所有语音帧的合成模型,通过显式预测每个音素的时长来桥接文本与语音序列长度的不匹配,从而避免逐帧自回归生成。

**核心思想**:
```
Phoneme → [Encoder] → Hidden → [Duration Predictor] → Expanded Hidden → [Decoder] → Mel (并行)
```

**关键创新**: 用 Duration Predictor + Length Regulator 替代 attention mechanism,实现:
- 并行生成所有帧: O(1) 推理复杂度
- 消除 word skipping/repeating: 显式 duration 保证完整对齐
- 可控生成: 通过调整 duration 直接控制语速

## FastSpeech 系列

### FastSpeech (Ren et al., NeurIPS 2019)
- **架构**: Feed-Forward Transformer (FFT blocks)
- **Length Regulator**: 根据 duration 将 phoneme hidden 扩展到 mel 长度
- **Duration 来源**: 从 AR teacher model (Tacotron 2) 的 attention alignment 中提取
- **训练**: 用 teacher model 生成的 distilled mel 作为目标(knowledge distillation)
- **推理速度**: mel 生成 270x, 波形合成 38x speedup

### FastSpeech 2 (Ren et al., ICLR 2021)
- **改进1**: 直接使用 ground-truth mel 训练,不再需要 teacher distillation
- **改进2**: 增加 variance adaptor (pitch + energy + duration predictor)
- **Duration 来源**: Montreal Forced Alignment (MFA) 工具提取
- **效果**: 音质超越 FastSpeech,训练更简单

### FastSpeech 2s (Ren et al., ICLR 2021)
- 完全端到端: 直接从 phoneme 生成 waveform
- 使用辅助 mel decoder 帮助学习上下文表示
- 对抗训练提升波形质量

## 其他 NAR 声学模型

| 模型 | 时间 | 生成模型 | 特点 |
|------|------|----------|------|
| ParaNet | 2019.05 | Seq2Seq (positional attn) | CNN-based NAR, 自注意力替代 |
| SpeedySpeech | 2020.08 | Seq2Seq | CNN 替代 Transformer, 更轻量 |
| AlignTTS | 2020.03 | Seq2Seq | 动态规划学习对齐 |
| JDI-T | 2020.05 | Seq2Seq | 联合训练 AR + NAR |
| FastPitch | 2020.06 | Seq2Seq | 加入 pitch predictor |
| Glow-TTS | 2020.05 | Flow | monotonic alignment search |
| EATS | 2020.06 | GAN | 端到端 + soft DTW duration |
| Flow-TTS | 2020.04 | Flow | generative flow |
| EfficientTTS | 2020.12 | GAN | duration interpolation |
| VITS | 2021.06 | VAE+Flow | fully E2E, 单模型最强 |

## NAR 生成的挑战与解决

**核心挑战**: one-to-many mapping problem
- 同一文本可对应多种语音变体(不同韵律、说话人等)
- 简单 L1/L2 loss → over-smoothing(预测均值而非任一合理变体)

**解决方案**:
1. **Variance information**: pitch/energy/duration 作为额外输入 (FastSpeech 2)
2. **Flow-based**: 可逆变换建模完整分布 (Glow-TTS, Flow-TTS)
3. **VAE**: 隐变量捕捉变化信息 (BVAE-TTS, VAE-TTS)
4. **GAN**: 对抗训练避免 over-smoothing (Multi-SpectroGAN)
5. **Diffusion**: 逐步去噪生成 (Grad-TTS, Diff-TTS)

## 时间复杂度对比

| 范式 | 代表模型 | 训练 | 推理 |
|------|----------|------|------|
| AR (RNN) | Tacotron 1/2 | O(N) | O(N) |
| AR (CNN/Self-Att) | TransformerTTS | O(1) | O(N) |
| NAR (CNN/Self-Att) | FastSpeech 1/2 | O(1) | O(1) |
| NAR (GAN/VAE) | HiFi-GAN, EATS | O(1) | O(1) |
| Flow (Bipartite) | WaveGlow, Glow-TTS | O(T) | O(T) |
| Diffusion | DiffWave, Grad-TTS | O(T) | O(T) |

## 关键论文

- FastSpeech (Ren et al., NeurIPS 2019): 开创 NAR TTS, duration predictor + FFT
- FastSpeech 2/2s (Ren et al., ICLR 2021): 简化训练 + variance adaptor + E2E
- Glow-TTS (Kim et al., NeurIPS 2020): flow-based NAR, monotonic alignment search
- VITS (Kim et al., ICML 2021): VAE+Flow fully E2E NAR
- EATS (Donahue et al., ICLR 2021): GAN-based fully E2E, soft DTW

## 相关概念

- [[Attention-based TTS]]: NAR TTS 的前身,FastSpeech 从其 attention 中提取 duration
- [[Duration Predictor]]: NAR TTS 的核心组件,桥接长度不匹配
- [[Mel Spectrogram]]: NAR 模型的输出目标
- [[Prosody Modeling]]: variance adaptor (pitch, energy) 提升 NAR 韵律

## 可控性对比 (Xie et al. 2024 Survey)

Survey 从可控性角度对比 NAR vs AR/LLM 架构:

| 维度 | NAR (FastSpeech, Matcha-TTS, F5-TTS) | LLM-based (VALL-E, CosyVoice) |
|------|--------------------------------------|-------------------------------|
| 控制方式 | 显式 variance predictor / flow conditioning | In-context learning / instruction |
| 控制精度 | 高 (直接操控 pitch/energy/duration) | 低 (隐式, 难精确控制) |
| 灵活度 | 低 (预定义属性集) | 高 (自然语言驱动) |
| 推理速度 | 快 (并行) | 慢 (自回归) |
| 零样本能力 | 有限 | 强 (few-second prompt) |
| 表达多样性 | 受限于显式标签 | 丰富 (上下文感知) |

**Hybrid 趋势**: CosyVoice 等将 LLM 的控制灵活度与 flow-based NAR 的生成质量结合。

## 演进

Tacotron (AR+Attention, 2017) → FastSpeech (NAR+Duration, 2019) → FastSpeech 2 (直接训练, 2020) → Glow-TTS/VITS (NAR+Flow, 2020-21) → **LLM-based TTS 回归 AR** (VALL-E, 2023) → Masked generation (NAR 新形态, MaskGCT, 2024) → Hybrid: LLM + NAR Flow (CosyVoice, 2024) → OmniVoice (2026, 单阶段 discrete NAR + LLM 初始化, 首个成功将 AR LLM 权重迁移至 NAR 架构的 TTS, 600+ 语言)
