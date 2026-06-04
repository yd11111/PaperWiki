---
type: concept
title: "Single-codebook vs Multi-codebook"
aliases: [SVQ vs RVQ, 单码本 vs 多码本, Single Codebook, Single-stream Tokenizer, Flat Token]
category: "design-choice"
tags: [quantization, codec-design, SVQ, RVQ, single-codebook, audio-codec]
key_papers: ["[[论文笔记/Survey-Discrete Audio Tokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/Multi-Reward GRPO|Multi-Reward GRPO]]", "[[论文笔记/Spark-TTS|Spark-TTS]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/OmniVoice|OmniVoice]]", "[[论文笔记/XTTS|XTTS]]", "[[论文笔记/DS-Codec|DS-Codec]]", "[[论文笔记/UniTTS|UniTTS]]", "[[论文笔记/C2F-LM|C2F-LM]]", "[[论文笔记/Dragon-FM|Dragon-FM]]", "[[论文笔记/Llasa+|Llasa+]]", "[[论文笔记/SecoustiCodec|SecoustiCodec]]"]
origin_paper: "Mousavi et al., Discrete Audio Tokens: More Than a Survey!, TMLR 2025"
related_concepts: ["[[Residual Vector Quantization]]", "[[Token Rate and Bitrate Trade-offs]]", "[[Audio Tokenizer Taxonomy]]", "[[Codec Language Model]]", "[[Speech Tokenizer]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Single-codebook vs Multi-codebook 是 audio tokenizer 设计中的核心二元选择: 使用一个码本 (SVQ, Single Vector Quantization) 将每帧映射为一个 token, 还是使用多个码本 (RVQ, 多层 VQ) 将每帧映射为多个 token [§2.2.1]。

这一选择直接影响 token rate、LM 建模复杂度和信息分布结构。近年来出现了从多码本 (8-32 层 RVQ) 向更少码本 (1-4 层) 的明确趋势。

## 两种方案对比

### Multi-codebook (RVQ)

每帧产生 M 个 token (每层一个), 通过残差逐层细化 [§2.2.1]:

$$\hat{z}_t = \sum_{m=1}^{M}\hat{z}_t^{(m)}, \quad r_t^{(m+1)} = r_t^{(m)} - \hat{z}_t^{(m)}$$

**特点**:
- 信息层级结构: 第一层编码 coarse 信息, 后续层编码 fine details
- Token rate = frame_rate x M (如 75 Hz x 8 = 600 tokens/s)
- 需要多流建模策略: AR+NAR (VALL-E), delay pattern (MusicGen), interleaving (Moshi)
- 代表: SoundStream (M=12), EnCodec (M=2-32), DAC (M=9)

**优势**:
- 高重建质量 (RVQ 在 survey ablation 中重建指标最优 [Table 15-16])
- 可变比特率 (通过 quantizer dropout 选择使用前 n 层)
- 信息分层天然适合 hierarchical generation

**劣势**:
- Token rate 高 → LM 序列长 → 推理慢
- 多流建模增加 LM 架构复杂度
- 后层码本 contribution 递减 (信息冗余)

### Single-codebook (SVQ)

每帧仅产生 1 个 token, 类似 VQ-VAE [§2.2.1]:

$$q_t = \arg\min_{k \in \{1,...,K\}} \|z_t - c_k\|^2$$

**特点**:
- 无残差细化, 需要更大码本 (K >> 1024) 补偿表达能力
- Token rate = frame_rate (如 75 Hz = 75 tokens/s)
- 单流建模: 标准 AR LM 即可, 无需多流策略
- 代表: BigCodec (K=8192, 1.04 kbps), WavTokenizer (K=4096), TS3-Codec, RepCodec

**优势**:
- 极低 token rate → LM 序列短 → 推理快
- 单流 = 标准 LM, 无需特殊架构 (delay pattern 等)
- 简化 acoustic LM 训练 (无多码本同步问题)

**劣势**:
- 单一码本难以覆盖高维空间 → 需要大码本 (K >= 4096) [§2.2.1]
- 大码本 → LM 词表大 → softmax 计算量增加
- 重建质量通常低于 RVQ (survey ablation: SVQ 在大多数指标上最差 [Table 15-16])

## 量化方法概览 [§2.2.1]

| 方法 | 码本数 | 特点 | 代表 |
|------|--------|------|------|
| **RVQ** | M 层 | 残差逐层细化; 最成熟 | SoundStream, EnCodec, DAC |
| **SVQ** | 1 | 单码本 + 大词表; 简化 LM | BigCodec, WavTokenizer, TS3-Codec |
| **GVQ** | G 组 | 输入分组独立量化; 增强第一层表达力 | HiFi-Codec, FACodec, FunCodec |
| **MSRVQ** | 多层多尺度 | 不同时间分辨率的 RVQ; 减少 token 数 | SNAC, LLM-Codec |
| **CSRVQ** | 多层多分辨率 | 编解码器特征间做残差量化 | ESC, Disen-TF-Codec |
| **FSQ** | 1 (隐式) | 无码本,固定网格量化 | SQ-Codec, HARP-Net, LFSC |

## 向 fewer codebooks 的趋势

### 驱动力

1. **LLM 集成**: 多码本 token 难以直接输入 text LLM (需 flatten/interleave/delay), 单码本 token 可直接当 text token 用
2. **推理效率**: 10s speech, RVQ@8 层 = 6000 tokens vs SVQ@1 层 = 750 tokens → 8x 序列长度差异
3. **建模简化**: 单流 AR 建模成熟且稳定; 多流建模 (VALL-E AR+NAR, Moshi delay) 增加复杂度

### 代表工作

| 工作 | 码本 | 码本大小 | 帧率 | 比特率 | 策略 |
|------|------|---------|------|--------|------|
| BigCodec (2024) | 1 | 8192 | 80 | 1.04 kbps | CNN+RNN encoder, SVQ, 低帧率补偿 |
| WavTokenizer (2024) | 1 | 4096 | 75/40 | 0.98/0.52 kbps | CNN+T, SVQ, 极低比特率 |
| TS3-Codec (2024) | 1 | - | 40-50 | 可变 | Transformer, SVQ, 自适应 |
| RepCodec (2024) | 1 | - | 50 | - | CNN, SVQ, HuBERT 蒸馏 |
| Mimi (2024) | 1+7 | 2048 | 12.5 | 1.1-4.4 kbps | 第一层 VQ 语义 + 额外 7 层 RVQ 声学 |
| DS-Codec (2025) | 1 | 8192/65536(PQ) | 80 | 1.04/1.28 kbps | 双阶段训练(镜像→非镜像), VQ/PQ 两种方案 |

### 折中方案

**MSRVQ (Multi-Scale RVQ)** [§2.2.1]: 在不同时间分辨率上应用 RVQ,高频信息用高帧率低层量化,低频信息用低帧率高层量化:

$$\hat{z}_t^{(i)} = \text{Upsample}\left(Q^{(i)}\left(\text{Downsample}(r_t^{(i)}, W_i)\right)\right)$$

SNAC (Siuzdak et al., 2024) 使用此策略,减少总 token 数同时保留多层信息。

## Survey Benchmark 发现

### 重建质量 [Table 15-16]
- **RVQ >> SVQ** 在 SDR, SI-SNR, PESQ 等大多数信号指标
- SVQ 在 speech UTMOS 和 DNSMOS 上与 RVQ 差距较小 (感知质量差异小于信号差异)
- FSQ 介于 RVQ 和 SVQ 之间

### 下游任务 [Table 7-9]
- SVQ (SQ-Codec) 在下游判别式任务上竞争力有限
- 但 WavTokenizer (SVQ) 在 TTS 中表现良好: dWER 4.67, 仅次于 Discrete WavLM [Table 11]
- 单流简化 LM 建模,可能补偿重建质量的损失

### SLM 评估 [Table 10]
- SQ-Codec (SVQ) 在 spoken content 任务上表现中等
- WavTokenizer (SVQ) 在 acoustic consistency 任务 (gender 78.50, Spk 69.00) 上竞争力较强

## 在 TTS 中的应用

- **单码本路线**: CosyVoice (FSQ, 1 codebook, 25 Hz), MaskGCT semantic codec (1 codebook, 8192), BigCodec → 适合 AR 或 masked generative modeling
- **多码本路线**: VALL-E (8 codebook RVQ), SoundStorm (8 codebook) → 需要 AR+NAR 两阶段或 delay pattern
- **趋势**: 越来越多 TTS 系统选择单码本 semantic tokenizer + 独立 acoustic decoder (CFM/vocoder), 避免多码本 LM 建模的复杂性

## 关键论文

- Mousavi et al., "Discrete Audio Tokens", TMLR 2025 — SVQ/RVQ/FSQ 系统对比与消融 [§2.2.1, §4]
- Xin et al., "BigCodec", 2024: 单码本 1.04 kbps codec, CNN+RNN + SVQ
- Yang et al., "WavTokenizer", 2024: 单码本 40 Hz, CNN+T + SVQ
- Siuzdak et al., "SNAC", 2024: Multi-Scale RVQ 折中方案
- Zeghidour et al., "SoundStream", 2021: 经典多码本 RVQ

## 相关概念

- [[Residual Vector Quantization]]: 多码本方案的核心算法
- [[Token Rate and Bitrate Trade-offs]]: 码本数直接决定 token rate
- [[Codec Language Model]]: 多码本建模的需求方
- [[Speech Tokenizer]]: SVQ/RVQ 选择是 tokenizer 设计的关键决策

## 演进

VQ-VAE 单码本 (2017) → RVQ 多码本 (SoundStream, 2021, M=12) → EnCodec 可变码本数 (2022, M=2-32) → DAC 深层码本 (2023, M=9) → 单码本回归 (BigCodec/WavTokenizer, 2024) → MSRVQ 折中 (SNAC, 2024) → 极低帧率单码本 (Mimi 12.5Hz, LFSC 21.5Hz, 2024-2025)

---

> [!info] 来源
> 基于 Mousavi et al., "Discrete Audio Tokens: More Than a Survey!", TMLR 2025, Sections 2.2.1, 3, 4, Tables 1-2, 15-16。
