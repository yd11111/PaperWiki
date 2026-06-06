---
type: task
title: "Neural Audio Compression"
aliases: [Neural Audio Codec, Learned Audio Compression, End-to-End Audio Codec]
category: "audio-processing"
tags: [audio-codec, compression, discrete-representation, speech-tokenization]
key_papers: ["[[论文笔记/SoundStream|SoundStream]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/MBCodec|MBCodec]]", "[[论文笔记/VARSTok|VARSTok]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/SAC|SAC]]", "[[论文笔记/PURECodec|PURE Codec]]", "[[论文笔记/T-Mimi|T-Mimi]]", "[[论文笔记/EntropyGRVQ|EntropyGRVQ]]", "[[论文笔记/WavTokenizer|WavTokenizer]]", "[[论文笔记/OmniCodec|OmniCodec]]"]
related_tasks: ["[[Zero-shotSpeechSynthesis]]"]
metrics: [ViSQOL, Mel-distance, STFT-distance, SI-SDR, MUSHRA, Bitrate-efficiency]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Neural Audio Compression 是使用神经网络将高维音频信号压缩为低维离散表征 (discrete tokens),并能从这些 tokens 高保真重建原始音频的任务。与传统编解码器 (MP3, Opus, AAC) 不同,neural codec 使用数据驱动的 learned representations。

## 核心要求

1. **高保真重建**: 压缩后能恢复接近原始的音质
2. **高压缩率**: 用尽可能少的 bits 表示音频 (通常 1.5-24 kbps vs CD 1411 kbps)
3. **通用性**: 处理语音、音乐、环境声等所有类型
4. **低延迟** (optional): 支持实时 streaming 场景

## 典型方法

主流范式: Convolutional Encoder-Decoder + Residual Vector Quantization (RVQ) + GAN Training

```
Audio → Encoder → Continuous Latent → RVQ → Discrete Codes → Decoder → Reconstructed Audio
                                         ↕
                            (存储/传输/用于生成模型)
```

## 评估指标

| 指标 | 类型 | 说明 |
|------|------|------|
| ViSQOL | 客观 (intrusive) | 频谱相似度估计 MOS |
| Mel distance | 客观 | log-mel spectrogram 距离 |
| STFT distance | 客观 | 高频保真度 |
| SI-SDR | 客观 | 信号相位重建质量 |
| MUSHRA | 主观 | 多刺激隐藏参考评估 |
| Bitrate efficiency | 效率 | codebook 实际利用率 |

## 与 TTS 的关系

Neural audio codec 的 discrete codes 可直接作为 speech tokenizer 用于 LLM-based TTS:
- AudioLM, VALL-E, MusicLM 等均使用 EnCodec/SoundStream 的 tokens 作为生成目标
- DAC 的更高保真度和更优 bitrate efficiency 可提升下游 TTS 质量
- Codec tokens 天然具有 coarse-to-fine 层级结构 (RVQ 的层级性)

## 关键论文

- SoundStream (Google, 2021): 开创 encoder-decoder + RVQ 范式
- EnCodec (Meta, 2022): 改进训练策略和判别器
- DAC (Descript, NeurIPS 2023): 系统性改进 VQ-GAN recipe, 8 kbps 达到 90x 压缩

## SOTA 对比 (截至 2023)

| 模型 | 比特率 | 带宽 | 压缩因子 |
|------|--------|------|----------|
| DAC | 8 kbps | 22.05 kHz | 91x |
| EnCodec | 24 kbps | 12 kHz | 16-32x |
| SoundStream | 6 kbps | 12 kHz | 64x |
| WavTokenizer (1Q) | 0.9 kbps | 24 kHz | ~427x |
| Opus (传统) | 8-24 kbps | 4-16 kHz | - |
