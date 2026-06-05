---
type: concept
title: "Residual Vector Quantization"
aliases: [RVQ, Residual VQ, Multi-stage VQ]
category: "quantization"
tags: [quantization, discrete-representation, audio-codec, neural-compression]
key_papers: ["[[论文笔记/SoundStream|SoundStream]]", "[[论文笔记/SoundStorm|SoundStorm]]", "[[论文笔记/NaturalSpeech3|NaturalSpeech 3]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Survey-DiscreteAudioTokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/AudioLM|AudioLM]]", "[[论文笔记/Fish-Speech|Fish-Speech]]", "[[论文笔记/UniAudio|UniAudio]]", "[[论文笔记/FireRedTTS2|FireRedTTS 2]]", "[[论文笔记/NaturalSpeech2|NaturalSpeech 2]]", "[[论文笔记/Make-A-Voice|Make-A-Voice]]", "[[论文笔记/MELLE|MELLE]]", "[[论文笔记/Moshi|Moshi]]", "[[论文笔记/SNAC|SNAC]]", "[[论文笔记/RepCodec|RepCodec]]", "[[论文笔记/VQ-VAE|VQ-VAE]]", "[[论文笔记/FlowDec|FlowDec]]", "[[论文笔记/SongGen|SongGen]]", "[[论文笔记/FlexiCodec|FlexiCodec]]", "[[论文笔记/SiTok|SiTok]]", "[[论文笔记/FishAudioS2|Fish Audio S2]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/DiSTAR|DiSTAR]]", "[[论文笔记/Cont-SPT|Cont-SPT]]", "[[论文笔记/LiveSpeech2|LiveSpeech 2]]", "[[论文笔记/OmniVoice|OmniVoice]]", "[[论文笔记/TraceableSpeech|TraceableSpeech]]", "[[论文笔记/Vec-TokSpeech|Vec-Tok Speech]]", "[[论文笔记/LLMVoX|LLMVoX]]", "[[论文笔记/TTS-Transducer|TTS-Transducer]]", "[[论文笔记/LM-SPT|LM-SPT]]", "[[论文笔记/MAEStyle-RichTTS|MAE Style-Rich TTS]]", "[[论文笔记/UniTTS|UniTTS]]", "[[论文笔记/Spotlight-TTS|Spotlight-TTS]]", "[[论文笔记/RevivalwithVoice|Revival with Voice]]", "[[论文笔记/SecoustiCodec|SecoustiCodec]]", "[[论文笔记/MBCodec|MBCodec]]", "[[论文笔记/FuseCodec|FuseCodec]]", "[[论文笔记/DiFlow-TTS|DiFlow-TTS]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/BridgeCode|BridgeCode]]", "[[论文笔记/Flamed-TTS|Flamed-TTS]]", "[[论文笔记/MAVE|MAVE]]", "[[论文笔记/SAC|SAC]]"]
origin_paper: "Zeghidour et al., SoundStream: An End-to-End Neural Audio Codec, 2021"
related_concepts: ["[[FiniteScalarQuantization]]", "[[CodebookCollapse]]", "[[QuantizerDropout]]", "[[SpeechTokenizer]]", "[[Single-codebookvsMulti-codebook]]", "[[TokenRateandBitrateTrade-offs]]", "[[CodecTrainingObjectives]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Residual Vector Quantization (RVQ) 是一种多层级向量量化方法,通过递归量化残差来逐步逼近输入向量。第一层 quantizer 量化原始输入,后续每层量化前一层的残差 (residual),最终表示为多层 codebook index 的组合。

数学形式: 给定输入 z, 第 i 层量化残差 r_i:
- r_1 = z
- q_i = Quantize(r_i)  (从 codebook_i 查找最近邻)
- r_{i+1} = r_i - q_i

最终重建: z_hat = sum(q_1, q_2, ..., q_N)

## 在 Audio Codec 中的应用

RVQ 是现代 neural audio codec (SoundStream, EnCodec, DAC) 的核心量化模块:
- 将 encoder 输出的连续 latent 压缩为 N_q 层离散 code
- 每层 codebook 通常 10-bit (1024 entries)
- 总比特率 = frame_rate x N_q x bits_per_code

DAC 的改进 [§3.2]: 使用 factorized codes (低维 8d lookup) + L2-normalization, 将 codebook utilization 从 ~90% 提升到 ~99%, 有效解决 codebook collapse 问题。

## 关键特性

1. **层级信息结构**: 前面的层编码 coarse 信息, 后面的层编码 fine details — 天然适合 hierarchical generation (如 AudioLM 的 coarse/fine tokens)
2. **可变比特率**: 通过使用不同数量的层 (1...N_q) 实现运行时 bitrate 控制
3. **与 VQ 的区别**: 单层 VQ 的 codebook 需要指数级大小才能覆盖高维空间; RVQ 通过残差分解, 用 N 个小 codebook 组合表达能力

## RVQ 变体 (Survey Taxonomy) [Mousavi et al. 2025, §2.2.1]

| 变体 | 核心思路 | 代表 |
|------|---------|------|
| **GVQ** (Group VQ) | 将输入特征分为 G 组,每组独立 RVQ; 增强第一层表达力 | HiFi-Codec (Yang 2023a), FACodec, FunCodec |
| **MSRVQ** (Multi-Scale RVQ) | 不同层在不同时间分辨率量化; 高层降采样→量化→上采样,减少 token 数 | SNAC (Siuzdak 2024), LLM-Codec |
| **CSRVQ** (Cross-Scale RVQ) | 在 encoder/decoder 不同层级间做残差量化; coarse-to-fine 多分辨率 | ESC (Gu & Diao 2024), Disen-TF-Codec |
| **RNDVQ** (Residual Normal Distribution VQ) | 将量化公式化为概率选择而非确定性最近邻; 改善 codebook 利用率和鲁棒性 | NDVQ (Niu et al. 2024) |
| **GRVQ** (Grouped RVQ) | GVQ + RVQ 的结合; 分组后做残差量化 | Prompt Codec, HiFi-Codec |

### GVQ 数学形式 [§2.2.1]

输入分为 G 组: $z_t = [z_t^{(1)} \| z_t^{(2)} \| \ldots \| z_t^{(G)}]$

每组独立量化: $\hat{z}_t = [\hat{z}_t^{(1)} \| \hat{z}_t^{(2)} \| \ldots \| \hat{z}_t^{(G)}]$

### MSRVQ 数学形式 [§2.2.1]

第 i 层残差降采样 W_i 倍后量化再上采样: $\hat{z}_t^{(i)} = \text{Upsample}(Q^{(i)}(\text{Downsample}(r_t^{(i)}, W_i)))$

## Survey 消融发现 [Mousavi et al. 2025, §4.2]

在 ESPnet-Codec 框架下控制变量实验 (Table 15-16):
- **RVQ 在大多数重建指标上优于 SVQ 和 FSQ**: SDR, SI-SNR, PESQ, Spk Sim 等
- **例外**: FSQ@16kHz 在 speech UTMOS 和 DNSMOS 上超过 RVQ (感知质量指标)
- **采样率交互**: RVQ 从 16kHz → 44.1kHz 一致性提升; FSQ 在 44.1kHz 反而部分退化
- **结论**: RVQ 具有最高重建保真度潜力, 但重建最优 =/= 下游最优

## 关键论文

- Zeghidour et al., "SoundStream: An End-to-End Neural Audio Codec", 2021: 首次将 RVQ 用于端到端 audio codec
- Defossez et al., "EnCodec: High Fidelity Neural Audio Compression", 2022: 改进 RVQ 训练 (EMA codebook)
- DAC (Kumar et al., NeurIPS 2023): factorized codes + L2-norm 解决 codebook collapse, bitrate efficiency 达 99%
- Mentzer et al., "Finite Scalar Quantization", ICLR 2024: 提出无需码本的替代方案 FSQ
- MaskGCT (Wang et al., 2024): 在 acoustic codec 中使用 12 层 RVQ (codebook size 1024, dim 8),配合 Vocos decoder 和 S2A masked generative model 逐层生成 acoustic tokens

## 相关概念

- [[FiniteScalarQuantization]]: VQ/RVQ 的替代方案, 无需显式码本
- [[CodebookCollapse]]: RVQ 训练的主要难点
- [[QuantizerDropout]]: 实现 RVQ 可变比特率的训练技巧
- [[SpeechTokenizer]]: RVQ-based codec 可作为声学 tokenizer 用于 TTS
- [[Single-codebookvsMulti-codebook]]: RVQ (多码本) vs SVQ (单码本) 的核心设计 trade-off
- [[TokenRateandBitrateTrade-offs]]: RVQ 的码本数直接决定 bitrate 和 token rate
- [[CodecTrainingObjectives]]: RVQ 训练中使用的 VQ loss / commitment loss

## 演进

VQ-VAE (2017) → RVQ/SoundStream (2021) → EnCodec (2022, EMA codebook) → DAC (2023, factorized codes) → GVQ/GRVQ (HiFi-Codec, 2023) → FSQ (2024, 去码本化) → MSRVQ (SNAC, 2024, 多尺度) → CSRVQ (ESC, 2024, 跨尺度) → RNDVQ (2024, 概率化) → 单码本回归 (BigCodec/WavTokenizer, 2024) → PURE Codec (2025, enhancement-guided entropy decomposition: 用 speech enhancement 模型引导第一层量化低熵表征,训练范式改进而非结构改进)
