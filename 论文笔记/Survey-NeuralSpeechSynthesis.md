---
type: paper-note
title: "A Survey on Neural Speech Synthesis"
authors: [Xu Tan, Tao Qin, Frank Soong, Tie-Yan Liu]
affiliation: Microsoft Research Asia
year: 2021
venue: "arXiv:2106.15561"
tier: card
tags: [survey, TTS, neural-TTS, cold-start]
created: 2026-06-01
updated: 2026-06-01
---

## 概要

这是 Microsoft Research Asia 团队撰写的神经语音合成综述,系统梳理了 2016-2021 年间 neural TTS 的发展。覆盖 TTS 的三大基本组件(文本分析、声学模型、声码器)和五大高级话题(快速 TTS、低资源 TTS、鲁棒 TTS、表现力 TTS、自适应 TTS)。

## 知识提取成果

本文用于概念库冷启动,提取了 TTS 领域的基础性领域知识。

### 新建概念页 (10 个)

| 概念 | 核心内容 |
|------|----------|
| [[MelSpectrogram]] | TTS 中间声学表示,mel 滤波器组 + log 压缩 |
| [[Text-to-SpeechPipeline]] | TTS 系统三级架构及 5 阶段端到端化进程 |
| [[Attention-basedTTS]] | Tacotron 系列 encoder-attention-decoder 范式 |
| [[Non-autoregressiveTTS]] | FastSpeech 系列并行生成范式 |
| [[DurationPredictor]] | 显式时长预测替代 attention 对齐 |
| [[NeuralVocoder]] | 波形合成器家族 (AR/Flow/GAN/VAE/Diffusion) |
| [[VariationalAutoencoderforTTS]] | VAE 解决 one-to-many mapping 问题 |
| [[ProsodyModeling]] | 韵律信息的建模、解耦与迁移 |
| [[PhonemeRepresentation]] | G2P 与文本前端 |
| [[SpeakerEmbedding]] | 说话人身份表示与自适应 TTS |

### 更新概念页 (2 个)

| 概念 | 更新内容 |
|------|----------|
| [[SpeechFactorization]] | 补充 survey 中 variation information 解耦的历史脉络 |
| [[Multi-scaleSTFTDiscriminator]] | 补充 GAN vocoder 判别器设计的历史背景 |

## 关键分类体系

1. **主分类**: Text Analysis → Acoustic Model → Vocoder (Figure 3a)
2. **AR vs NAR**: 自回归逐帧 vs 并行生成 (Table 8 时间复杂度)
3. **生成模型**: Seq2Seq / Flow / GAN / VAE / Diffusion (Figure 5)
4. **网络结构**: RNN / CNN / Self-Attention / Hybrid (Figure 5)
5. **端到端化程度**: Stage 0-4 渐进过程 (Figure 4)

## 演进时间线关键节点

- 2016.09: WaveNet (首个 neural vocoder)
- 2017.03: Tacotron (首个 E2E attention TTS)
- 2017.12: Tacotron 2 (确立 mel + vocoder 范式)
- 2018.06: TransformerTTS (self-attention 替代 RNN)
- 2019.05: FastSpeech (NAR + duration predictor)
- 2019.10: MelGAN (首个 GAN vocoder)
- 2020.06: FastSpeech 2 (variance adaptor)
- 2020.10: HiFi-GAN (GAN vocoder 标准)
- 2021.06: VITS (fully E2E, VAE+Flow+GAN)
