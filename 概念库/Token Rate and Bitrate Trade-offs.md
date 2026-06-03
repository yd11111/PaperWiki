---
type: concept
title: "Token Rate and Bitrate Trade-offs"
aliases: [Token Rate, 帧率与比特率权衡, Frame Rate vs Bitrate, Codec 比特率设计]
category: "design-choice"
tags: [bitrate, frame-rate, token-rate, codec-design, compression]
key_papers: ["[[论文笔记/Survey-Discrete Audio Tokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/SoundStream|SoundStream]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/TADA|TADA]]"]
origin_paper: "Mousavi et al., Discrete Audio Tokens: More Than a Survey!, TMLR 2025"
related_concepts: ["[[Residual Vector Quantization]]", "[[Single-codebook vs Multi-codebook]]", "[[Audio Tokenizer Taxonomy]]", "[[Quantizer Dropout]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Token Rate 和 Bitrate 是 audio tokenizer 的两个核心设计参数,直接影响压缩效率、重建质量和下游任务性能。两者通过码本数和码本大小关联。

### 基本公式

$$\text{Bitrate (bps)} = \text{Frame Rate} \times N_q \times \log_2 K$$

$$\text{Token Rate (tokens/s)} = \text{Frame Rate} \times N_q$$

其中:
- **Frame Rate**: encoder 输出的时间分辨率 (Hz), 即每秒帧数
- **N_q**: 量化器层数 (RVQ 中的码本数)
- **K**: 每个码本的大小 (entries)
- **Token Rate**: 编码一秒音频需要的 token 总数 (对 LM 至关重要)

### 典型参数范围 [Table 2]

| 参数 | 典型范围 | 示例 |
|------|---------|------|
| 采样率 (kHz) | 16 - 44.1 | 16 (speech), 24 (multi), 44.1 (music) |
| Frame Rate (Hz) | 12.5 - 86 | 12.5 (Mimi), 50 (EnCodec/DAC@16k), 75 (DAC@24k), 86 (DAC@44k) |
| 码本数 N_q | 1 - 32 | 1 (WavTokenizer), 4-8 (EnCodec), 9 (DAC), 32 (EnCodec@24kbps) |
| 码本大小 K | 1000 - 19683 | 1024 (EnCodec/DAC), 2048 (Mimi), 4096 (WavTokenizer), 19683 (SQ-Codec) |
| 比特率 (kbps) | 0.52 - 24 | 0.98 (WavTokenizer), 1.5 (EnCodec@2Q), 6 (EnCodec@8Q), 24 (EnCodec@32Q) |
| Token Rate (tokens/s) | 40 - 2400 | 40 (WavTokenizer), 100 (ST@2Q), 600 (DAC@8Q), 2400 (EnCodec@32Q) |

## 关键 Trade-offs [§3.1, §3.2, §4.2]

### Trade-off 1: 比特率 vs 重建质量 [Table 4]

更高比特率 → 更好重建质量,但收益递减:

| Tokenizer | #Q | kbps | Token Rate | PESQ | UTMOS | WER |
|-----------|-----|------|-----------|------|-------|-----|
| EnCodec-24 | 2 | 1.5 | 150 | 1.56 | 1.58 | 5.44 |
| EnCodec-24 | 8 | 6 | 600 | 2.77 | 3.09 | 2.78 |
| EnCodec-24 | 32 | 24 | 2400 | 7.90 | 3.71 | 2.77 |

关键观察: UTMOS 和 WER 在 8Q → 32Q 时改善微弱,但 token rate 增加 4 倍。

### Trade-off 2: 比特率 vs 下游任务性能 [§3.2]

Survey 的核心发现之一: **重建最优 =/= 下游最优**

"Increasing the number of codebooks (e.g., 2, 8, 32) improves signal reconstruction but often reduces downstream task performance." [§3.2]

原因: 更多码本 → 更高维输出 → 建模复杂度增加 → 判别式和生成式下游任务退化。Medium bitrate 通常是最佳平衡点。

### Trade-off 3: Token Rate vs LM 建模效率

Token rate 直接决定 LM 的序列长度:
- WavTokenizer (1 codebook, 40 tokens/s): 10s 语音 = 400 tokens
- EnCodec@32Q (32 codebooks, 2400 tokens/s): 10s 语音 = 24000 tokens

低 token rate 对 LM 有巨大优势: 更短序列 → 更快推理 → 更低内存 → 更好的长距离建模。这驱动了 [[Single-codebook vs Multi-codebook]] 中向更少码本的趋势。

### Trade-off 4: 采样率 vs 量化方法 [§4.2, Table 15-16]

Survey 消融发现采样率与量化方法存在交互效应:
- **RVQ**: 从 16kHz → 44.1kHz 一致性提升重建质量
- **FSQ**: 44.1kHz 反而导致部分指标退化
- **SVQ**: 两种采样率都表现较差

**设计建议**: 采样率选择应与量化方法联合考虑,不能独立决定。

## 比特率分类 [§2.2.2]

### Fixed Bitrate
码本数和码本大小固定,比特率恒定:
- 每个 code index 占用固定 bits (如 10 bits for K=1024)
- 大多数 codec 的默认模式

### Adaptive Bitrate
基于 token 分布的熵编码,不同 token 占用不同 bits:
- Huffman / arithmetic coding
- 利用 token 频率不均匀性压缩
- 可在任何量化方法之上应用 [§2.2.2]
- 代表: S-TFNet (Jiang et al., 2022), HARP-Net (Petermann et al., 2021)

### Scalable Bitrate
通过改变活跃码本数量实现多档比特率:
- 通过 [[Quantizer Dropout]] 训练单一模型支持多种比特率
- 运行时选择使用前 n 层 codebook (n <= N_q)
- 比特率以层为粒度调节,不逐 token 自适应
- 代表: EnCodec, SoundStream, DAC

## 在 TTS 中的应用

TTS 系统对 token rate 和 bitrate 有特殊需求:
- **LM 序列长度**: 低 token rate (25-50 Hz, 1 codebook) 对 AR 生成有利,减少推理延迟
- **重建质量**: 足够的 bitrate 保证合成语音保真度
- **最优实践**: CosyVoice 系列使用 25 Hz / 1 codebook; VALL-E 使用 75 Hz / 8 codebook (AR first + NAR rest)

Survey TTS 实验 [Table 11]: Discrete WavLM (6 codebooks, 3 kbps, semantic) 达到 UTMOS 3.42, dWER 7.45, SpkSim 0.90 — 语义 tokenizer 在有限数据条件下更稳定。

## 关键论文

- Mousavi et al., "Discrete Audio Tokens", TMLR 2025 — 系统分析 bitrate-quality trade-off [§3.1, §4.2]
- Zeghidour et al., "SoundStream", 2021: 引入 quantizer dropout 实现 scalable bitrate
- Kumar et al., "DAC", NeurIPS 2023: 概率化 quantizer dropout (p=0.5)
- Defossez et al., "EnCodec", 2023: codebook dropout 实现多档位比特率

## 相关概念

- [[Residual Vector Quantization]]: 比特率 = frame_rate x N_q x log2(K) 中的 N_q 和 K
- [[Single-codebook vs Multi-codebook]]: 降低 token rate 的核心策略
- [[Quantizer Dropout]]: 实现 scalable bitrate 的训练技巧
- [[Audio Tokenizer Taxonomy]]: 比特率是 taxonomy Axis 2 的子维度

## 演进

固定比特率 codec (Opus, 2012) → RVQ 可变层数 (SoundStream, 2021) → Quantizer Dropout 训练 (SoundStream/EnCodec, 2021-2022) → 概率化 dropout (DAC, 2023) → 极低比特率单码本 (WavTokenizer/BigCodec, 2024) → 自适应熵编码 (HARP-Net, 2021; S-TFNet, 2022) → 采样率-量化方法联合设计 (Survey ablation, 2025)

---

> [!info] 来源
> 基于 Mousavi et al., "Discrete Audio Tokens: More Than a Survey!", TMLR 2025, Sections 2.2.2, 3.1, 3.2, 4.2, Tables 2-5。
