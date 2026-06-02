---
type: concept
title: "Audio Tokenizer Taxonomy"
aliases: [音频分词器分类体系, Tokenizer Taxonomy, Discrete Audio Token Taxonomy]
category: "taxonomy"
tags: [taxonomy, audio-codec, discrete-token, tokenization, survey]
key_papers: ["[[论文笔记/Survey-Discrete Audio Tokens|Survey-Discrete Audio Tokens]]"]
origin_paper: "Mousavi et al., Discrete Audio Tokens: More Than a Survey!, TMLR 2025"
related_concepts: ["[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Residual Vector Quantization]]", "[[Finite Scalar Quantization]]", "[[Codec Training Objectives]]", "[[Single-codebook vs Multi-codebook]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Audio Tokenizer Taxonomy 是 Mousavi et al. (2025) 提出的离散音频 tokenizer 五轴精细化分类体系,用于取代传统的 semantic vs acoustic 二分法。该分类基于 encoder-decoder 架构、量化方法、训练范式、目标领域和流式能力五个正交维度,覆盖 50+ 已有 tokenizer [Table 1, Fig 3]。

### 为什么需要新分类 [§1]

传统 semantic vs acoustic 二分法存在三个局限:
1. **边界模糊**: acoustic tokenizer 也能捕获语义信息 (如 Mimi 通过语义蒸馏), semantic tokenizer 也已被用于生成任务 [§1]
2. **忽略架构差异**: 二分法无法区分 CNN vs Transformer encoder、waveform vs T-F 表征等关键设计选择
3. **忽略实用维度**: 如流式能力、自适应比特率等对部署至关重要的特性

### 五轴分类 [Fig 3]

```
Audio Tokenizer
├── Axis 1: Encoder-Decoder [§2.3]
│   ├── Architecture: CNN / CNN+RNN / RNN / T / CNN+T
│   └── Input/Output Representation: Waveform domain / Time-Frequency domain
│
├── Axis 2: Quantization [§2.2]
│   ├── Algorithm: RVQ / GVQ / SVQ / MSRVQ / CSRVQ / PQ / FSQ / K-means
│   └── Bitrate: Fixed / Adaptive (entropy coding) / Scalable (variable #codebooks)
│
├── Axis 3: Training Paradigm [§2.4]
│   ├── Strategy: Separate (post-training) / Joint (end-to-end)
│   ├── Objectives: Reconstruction / Adversarial (GAN) / Feature Matching / Diffusion / Masked Prediction / VQ
│   └── Auxiliary: Disentanglement / Semantic Distillation / Supervised Semantic Tokenization
│
├── Axis 4: Target Domain [§2.5]
│   └── Speech / Music / General Audio / Multi-domain
│
└── Axis 5: Streamability [§2.5]
    └── Streamable (causal) / Non-streamable
```

## 各轴详细分类

### Axis 1: Encoder-Decoder 架构 [§2.3]

| 类型 | 特点 | 代表 |
|------|------|------|
| CNN | 最常见; 紧凑高效; 无法建模长距离依赖 | SoundStream, DAC, EnCodec |
| CNN+RNN | CNN 特征提取 + LSTM/GRU 序列建模 | EnCodec, BigCodec |
| Transformer (T) | 纯 attention; 性能强但计算量大 | TS3-Codec, ESC |
| CNN+T | CNN 提取 + Transformer 捕获长距离依赖; 新趋势 | Mimi, NAST, PAST, LLM-Codec, WMCodec |
| RNN (少见) | 早期方案, 现已基本被替代 | Best-RQ |

**输入/输出表征**:
- **Waveform domain (T)**: 直接处理时域波形, decoder 直接输出波形 (EnCodec, SoundStream)
- **Time-Frequency domain (T-F)**: 使用 mel 谱图等频域特征; decoder 输出频域特征后经 ISTFT 还原 (Vocos, DAC, SNAC)

### Axis 2: 量化方法 [§2.2]

| 方法 | 码本数 | 码本大小 | 典型比特率 | 代表 |
|------|--------|---------|-----------|------|
| RVQ | 2-32 | 1024 | 1.5-24 kbps | SoundStream, EnCodec, DAC |
| GVQ | G 组 | 各组独立 | 可变 | HiFi-Codec |
| SVQ | 1 | 4096-19683 | 低 | BigCodec, TS3-Codec, WavTokenizer |
| MSRVQ | 多层多尺度 | 可变 | 可变 | SNAC, LLM-Codec |
| CSRVQ | 多层多分辨率 | 可变 | 可变 | ESC, Disen-TF-Codec |
| PQ | 多子空间 | 可变 | 可变 | Best-RQ, SOCODEC |
| FSQ | 1 (隐式) | prod(L_i) | 可变 | SQ-Codec, HARP-Net, LFSC, TAAE |
| K-means | 1 | K 中心 | 低 | Discrete WavLM, MMM, SingOMD |

### Axis 3: 训练范式 [§2.4]

**训练策略**:
- **Separate (post-training)**: encoder 和 decoder 独立训练; 常见于 semantic tokenizer (frozen SSL encoder + offline k-means + 独立 vocoder) [§2.4.1]
- **Joint (end-to-end)**: encoder/quantizer/decoder 联合优化; 常见于 acoustic tokenizer; 使用 STE/soft-to-hard/Gumbel 传梯度 [§2.4.1]

**训练目标** (详见 [[Codec Training Objectives]]):
- Reconstruction (L_Recon), Adversarial (L_GAN), Feature Matching (L_Feats), Diffusion (L_diff), Masked Prediction (L_MP), VQ (L_VQ)

**辅助组件** [§2.4.3]:
- **Disentanglement**: 分离 speaker/content/prosody (FACodec, TiCodec, LSCodec, SD-Codec)
- **Semantic Distillation**: SSL 特征引导 RVQ 第一层学习语义 (SpeechTokenizer, Mimi, X-Codec)
- **Supervised Semantic**: 通过 ASR/phoneme 监督训练 (S3, PAST)

### Axis 5: Streamability [§2.5]

流式能力由两个因素决定:
- **算法延迟** (look-ahead window): CNN 通过 causal convolutions 实现零 look-ahead; Transformer 需要 causal attention
- **计算复杂度**: LPCNet (<2M params, 1.6 kbps) vs EnCodec (~14M) vs BigCodec (159M, 1.04 kbps)

SSL-based tokenizer (HuBERT, WavLM) 多使用 non-causal encoder,限制流式部署。

## 代表 Tokenizer 分类 (部分, 基于 Table 1)

| Tokenizer | 帧率 | 量化 | 架构 | 训练目标 | 辅助 | 流式 |
|-----------|------|------|------|----------|------|------|
| SoundStream | 75 | RVQ | CNN | GAN,Feat,Rec | - | 是 |
| EnCodec | 75 | RVQ | CNN+RNN | GAN,Feat,Rec,VQ | - | 是 |
| DAC | 75 | RVQ | CNN | GAN,Feat,Rec,VQ | - | 是 |
| SpeechTokenizer | 50 | RVQ | CNN+RNN | GAN,Rec,Feat,VQ | SD | 否 |
| Mimi | 12.5 | RVQ | CNN+T | GAN,Feat,Rec,VQ | SD | 是 |
| WavTokenizer | 75 | SVQ | CNN+T | GAN,Feat,Rec | - | 是 |
| SQ-Codec | 50 | FSQ | CNN | GAN,Rec | - | 否 |
| Discrete WavLM | 50 | K-means | CNN+T | GAN,Feat,Rec,MP | - | 否 |

## 在 TTS 中的应用

Tokenizer 的分类维度选择直接影响 TTS 系统设计:
- **量化方法**: SVQ (单码本) 简化 LM 建模; RVQ 需要 AR+NAR 两阶段或 delay pattern
- **训练辅助**: semantic distillation 可改善 TTS 的 content consistency; disentanglement 改善 speaker similarity
- **帧率**: 低帧率 (12.5-25 Hz) 降低 LM 序列长度,加速推理; 高帧率 (75-86 Hz) 保留更多声学细节

## 关键论文

- Mousavi et al., "Discrete Audio Tokens: More Than a Survey!", TMLR 2025 — 提出此五轴 taxonomy, 覆盖 50+ tokenizer [Table 1, Fig 3]
- Borsos et al., "AudioLM", 2023: 建立 semantic → acoustic 层级范式
- Cui et al., "Speech Language Models: A Survey", 2024: SpeechLM 视角的 tokenizer 三分法

## 相关概念

- [[Speech Tokenizer]]: 各类 tokenizer 的统称和通用定义
- [[Semantic vs Acoustic Tokens]]: 传统二分法,本 taxonomy 意在细化和替代
- [[Residual Vector Quantization]]: Axis 2 中最常见的量化方法
- [[Finite Scalar Quantization]]: Axis 2 中的无码本替代方案
- [[Codec Training Objectives]]: Axis 3 中训练目标的详细定义
- [[Single-codebook vs Multi-codebook]]: Axis 2 中 SVQ vs RVQ 的核心 trade-off

## 演进

Semantic vs Acoustic 二分法 (AudioLM, 2022) → SpeechLM 三分法 (Cui et al., 2024, semantic/acoustic/mixed) → 五轴精细化 Taxonomy (Mousavi et al., 2025, 架构/量化/训练/领域/流式)

---

> [!info] 来源
> 定义和分类体系基于 Mousavi et al., "Discrete Audio Tokens: More Than a Survey!", TMLR 2025, Section 2, Figure 3, Table 1。
