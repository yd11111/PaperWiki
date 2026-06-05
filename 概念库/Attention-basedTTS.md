---
type: concept
title: "Attention-based TTS"
aliases: [Tacotron系列, Encoder-Attention-Decoder TTS, Seq2Seq TTS, 注意力机制TTS]
category: "model-family"
tags: [TTS, autoregressive, attention, seq2seq, acoustic-model]
key_papers: ["[[论文笔记/VeryAttentiveTacotron|Very Attentive Tacotron (Battenberg et al., 2025)]]", "[[论文笔记/MambaVoiceCloning|MambaVoiceCloning (2026)]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Non-autoregressiveTTS]]", "[[DurationPredictor]]", "[[MelSpectrogram]]", "[[Text-to-SpeechPipeline]]", "[[NeuralVocoder]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Attention-based TTS 是基于 encoder-attention-decoder 框架的自回归(AR)语音合成模型家族。其核心思想是利用注意力机制隐式学习文本和语音之间的对齐关系,逐帧自回归生成 mel spectrogram。

**基本架构**:
```
Text/Phoneme → [Encoder] → Hidden States
                                ↓ (Attention)
            [Decoder] ← Previous Frame → Mel Frame (逐帧)
```

**关键特性**:
- **自回归生成**: 每一步以前一帧为条件生成当前帧
- **隐式对齐**: 通过 attention 学习 text-speech 对齐,无需外部 duration 标注
- **Seq2Seq 建模**: 将 TTS 视为序列到序列的翻译问题

## Tacotron 系列演进

### Tacotron (Wang et al., Interspeech 2017)
- 首个端到端 attention-based TTS
- 字符输入 → linear spectrogram → Griffin-Lim 合成波形
- Encoder: CBHG (Conv Bank + Highway + GRU)
- Attention: 标准 content-based attention

### Tacotron 2 (Shen et al., ICASSP 2018)
- 改为预测 mel spectrogram + WaveNet vocoder
- Encoder: 3层 CNN + BiLSTM
- Attention: Location-sensitive attention (结合 content + 前一步位置)
- Decoder: 2层 LSTM + 线性投影
- 大幅提升音质,接近人类自然度

### 后续改进方向
1. **增强表现力**: GST-Tacotron (style tokens), Ref-Tacotron (reference encoder)
2. **移除 attention → duration**: DurIAN, Non-Attentive Tacotron
3. **非自回归化**: Parallel Tacotron 1/2
4. **端到端波形生成**: Wave-Tacotron (Tacotron + flow decoder)

## 其他 Attention-based 模型

| 模型 | 时间 | 结构 | 特点 |
|------|------|------|------|
| DeepVoice 3 | 2017.10 | CNN | 全卷积, 多说话人 |
| TransformerTTS | 2018.09 | Self-Attention | Transformer encoder-decoder |
| MultiSpeech | 2020.06 | Self-Attention | 多说话人 Transformer |
| DurIAN | 2019.09 | Hybrid | 用 duration 替代 attention, 仍 AR 解码 |

## 注意力机制的挑战

TTS 中的注意力需满足三个约束:
1. **Local**: 一个音素对应连续若干帧,一帧只对应一个音素
2. **Monotonic**: 对齐必须单调递增(不能回头)
3. **Complete**: 每个音素至少对应一帧(不能跳过)

标准 content-based attention 不能保证这些约束,导致:
- **Word skipping**: 跳过部分文本内容
- **Word repeating**: 重复生成某些内容
- **Attention collapse**: 注意力集中在单一位置

**改进方案**:
| 方案 | 满足约束 | 代表工作 |
|------|----------|----------|
| Location-based attention | Monotonic | Char2Wav, VoiceLoop |
| Location-sensitive (hybrid) | Monotonic | Tacotron 2 |
| Monotonic attention | Local + Monotonic | MMA, SMA |
| Stepwise monotonic attention | Local + Mono + Complete | He et al. 2019 |
| Duration prediction | All three | FastSpeech, DurIAN |

## 优势与局限

**优势**:
- 无需外部对齐标注,端到端可训练
- 能产出自然的韵律变化
- 架构简洁

**局限**:
- 推理速度慢: 自回归逐帧生成(1秒语音约 100 帧)
- 鲁棒性问题: attention 错误导致跳字/重复
- 无法并行解码: 训练 O(N) 推理 O(N)

## 关键论文

- Tacotron (Wang et al., 2017): 首个端到端 TTS
- Tacotron 2 (Shen et al., 2018): 确立 mel + vocoder 标准
- TransformerTTS (Li et al., 2019): Self-attention 替代 RNN
- DurIAN (Yu et al., 2019): duration 替代 attention 的过渡形态
- GST-Tacotron (Wang et al., 2018): style token 增强表现力

## 相关概念

- [[Non-autoregressiveTTS]]: 用 duration prediction 替代 attention,实现并行生成
- [[DurationPredictor]]: 从 attention-based 模型的对齐中提取 duration 标签
- [[MelSpectrogram]]: attention-based TTS 的输出目标
- [[ProsodyModeling]]: 通过 reference encoder / style token 增强韵律

## 演进

Tacotron (2017) → Tacotron 2 (2017.12) → TransformerTTS (2018) → **FastSpeech 取代 attention** (2019) → DurIAN/Non-Att Tacotron (混合形态) → 现代 LLM-TTS 回归 AR 但用 discrete tokens (VALL-E, 2023) → **Very Attentive Tacotron** (2025, 用 IRPB + latent alignment 解决 attention robustness,保留多头多层 cross-attention 的灵活性) → **MambaVoiceCloning** (2026, 用 SSM 完全替代推理时 attention conditioning,gated bi-Mamba + AdaLN 在 protocol-matched 实验下小幅超越 StyleTTS2)
