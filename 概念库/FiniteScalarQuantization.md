---
type: concept
title: "Finite Scalar Quantization"
aliases: [FSQ, Finite Scalar Quantization]
category: "quantization"
tags: [quantization, discrete-representation, VQ-alternative, codebook-free]
key_papers: ["[[论文笔记/CosyVoice2|CosyVoice 2]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/Survey-DiscreteAudioTokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/Fish-Speech|Fish-Speech]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/VoxtralTTS|Voxtral TTS]]", "[[论文笔记/Spark-TTS|Spark-TTS]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/LLaMA-Omni2|LLaMA-Omni 2]]", "[[论文笔记/Koel-TTS|Koel-TTS]]", "[[论文笔记/UDDETTS|UDDETTS]]", "[[论文笔记/Dragon-FM|Dragon-FM]]", "[[论文笔记/SecoustiCodec|SecoustiCodec]]", "[[论文笔记/HD-PPT|HD-PPT]]", "[[论文笔记/VoxCPM2|VoxCPM2]]", "[[论文笔记/FlexiSLM|FlexiSLM]]"]
origin_paper: "Mentzer et al., Finite Scalar Quantization: VQ-VAE Made Simple, ICLR 2024"
related_concepts: ["[[ResidualVectorQuantization]]", "[[SpeechTokenizer]]", "[[Gumbel-Softmax]]", "[[AudioTokenizerTaxonomy]]", "[[Single-codebookvsMulti-codebook]]", "[[CodecTrainingObjectives]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Finite Scalar Quantization (FSQ) 是一种用固定网格量化替代 VQ codebook lookup 的离散表征方法。核心操作:将连续表征投影到 d 维低维空间(通常 d < 10),对每个维度独立做 bound + round,得到隐式 codebook。

### 数学形式 [§3.1]

给定 d 维表征 z ∈ R^d:
1. **Bound**: 用 `f(z_i) = ⌊L_i/2⌋ · tanh(z_i)` 将每个分量限制到 [-⌊L/2⌋, ⌊L/2⌋]
2. **Round**: `ẑ_i = round(f(z_i))` 量化到整数
3. **隐式 codebook**: C = {所有可能的 ẑ 组合}, 大小 |C| = ∏ L_i

例: d=3, L=3 → 每个维度取 {-1, 0, 1} → codebook C 有 3³ = 27 个 codeword。

### 为什么 work (核心 insight) [§3]

**VQ 做什么**: 在高维空间(d≥512)学习一个 Voronoi partition,codebook 是可学习的 → 需要 commitment loss、EMA、reseeding 等复杂机制来保证 codebook 被充分利用。

**FSQ 做什么**: 在极低维空间(d<10)用固定网格 partition → codebook 是预定义的,不需要学习 → 没有 codebook collapse 问题。

**为什么低维固定网格够用**: VAE 的 encoder/decoder 有足够的模型容量,VQ 中本应由 codebook 承担的"非线性映射"能力可以被 encoder/decoder "吸收"。FSQ 把复杂性从量化层推给了 encoder/decoder,换来了量化层本身的极致简化。

**结果**: 用简单得多的设计达到与 VQ 相当的性能,同时天然避免 codebook collapse。

### 梯度传播 [§3.1]

使用 STE (Straight-Through Estimator),与 VQ-VAE 相同:
```
round_ste: x → x + sg(round(x) - x)
```
前向走离散 round,反向梯度直接穿过。

## 与 VQ 的核心对比 [Fig.2, §1]

| | VQ | FSQ |
|---|---|---|
| 量化操作 | arg min_{c∈C} \|\|z - c\|\| | round(f(z)) |
| 梯度 | STE | STE |
| 辅助 loss | Commitment loss, entropy loss | **无** |
| 防 collapse 技巧 | EMA, reseeding, splitting, projections... | **不需要**(天然不 collapse) |
| 额外参数 | Codebook (\|C\|·d 个参数, 如 2M) | **无** |
| 维度 | d ≥ 512 | d < 10 |
| Codebook 利用率 | 随 \|C\| 增大而下降(>2¹⁰ 后 <50%) [Fig.3c] | ≈100% (对所有 \|C\| 大小) [Fig.3c] |
| 大 codebook 扩展性 | 差(VQ 对 \|C\|>2¹⁰ 反而变差) [Fig.3a] | 好(性能随 \|C\| 增大持续改善) |

## 超参数选择 [§3.2, Table 1]

超参数: 通道数 d + 每通道 level 数 L = [L_1, ..., L_d]。

推荐配置(启发式规则: L_i ≥ 5):

| 目标 \|C\| | 2⁸ | 2¹⁰ | 2¹² | 2¹⁴ | 2¹⁶ |
|---|---|---|---|---|---|
| 推荐 L | [8,6,5] | [8,5,5,5] | [7,5,5,5,5] | [8,8,8,6,5] | [8,8,8,5,5,5] |

## 在 TTS 中的应用

在 CosyVoice 系列中,FSQ 被插入到语音编码器的中间层,将连续语音表征离散化为 speech token:
- **CosyVoice 2**: FSQ 插入 SenseVoice-Large 编码器
- **CosyVoice 3**: FSQ 插入 MinMo 的 Voice Encoder,通过多任务监督训练(ASR+LID+SER+AED+SA, 530K h)优化量化表征的语义信息含量

FSQ 在 TTS 中的优势:
- 无需 codebook 维护 → 训练更稳定
- 100% 利用率 → 不浪费 token 空间
- 可直接与下游 LLM 配合(token 空间确定且完全利用)

训练时通过 STE 近似梯度,与 CosyVoice 的端到端训练兼容。

## 关键论文

- Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", ICLR 2024 — 原始论文,在 MaskGIT + UViM 上验证
- CosyVoice 3 (2025): 在 TTS speech tokenizer 中使用 FSQ + 多任务监督
- DAC (2023): 使用 RVQ(FSQ 的对比方案),通过 factorized codes 解决 codebook collapse

## 实验结论 (原始论文) [§5, Fig.3, Fig.4, Table 2]

- MaskGIT 256×256: FSQ FID 4.534 vs VQ FID 4.509,FSQ codebook usage 100% vs VQ 81% [Fig.4 table]
- UViM: FSQ 在 depth/segmentation/colorization 三任务上性能接近 VQ(差距 0.5-3%) [Table 2]
- Scaling: FSQ 性能随 codebook size 增大持续改善;VQ 在 >2¹⁰ 后反而退化 [Fig.3]
- FSQ 表征的 compression cost 略高(分布更难建模) [Fig.3d] — 这是简单量化的代价

## 相关概念

- [[ResidualVectorQuantization]]: 多层级 VQ,用于 SoundStream/EnCodec/DAC。FSQ 是其替代方案。
- [[SpeechTokenizer]]: FSQ 是 CosyVoice 系列 tokenizer 的量化核心
- [[Gumbel-Softmax]]: 另一种使离散操作可微的方法,DiffRO 中使用
- VQ-VAE (Van Den Oord et al., 2017): FSQ 的前身/替代对象
- Product Quantization: codebook 分解为子空间乘积,与 FSQ 思路不同但目标类似

## 演进

VQ-VAE (2017, 学习 codebook) → RVQ (2021, 多层级 VQ) → FSQ (2024, 去码本化、固定网格) → FSQ + 监督多任务 (CosyVoice 3, 2025)

## FSQ 变体与扩展 [Mousavi et al. 2025, §2.2.1]

| 变体 | 代表工作 | 特点 |
|------|---------|------|
| **标准 FSQ** | SQ-Codec (Yang 2024d), Spectral Codecs (Langman 2024) | round(tanh(z) * S) / S 标准公式 |
| **BSQ** (Binary Spherical Quantization) | FocalCodec (Della Libera 2025) | FSQ 变体,只使用两个标量值 (binary) |
| **HARP-Net FSQ** | Petermann et al. 2021 | 保持原始帧率 (44.1kHz), 不做时间下采样, 中间层扩维后做标量量化 |

**Survey 中 FSQ-based tokenizer 分布 (Table 1)**:
- SQ-Codec: CNN encoder, CNN decoder, FSQ, 50 Hz, speech
- HARP-Net: CNN encoder, CNN decoder, FSQ, 44100 Hz, music (无时间下采样)
- LFSC: CNN encoder, CNN decoder, FSQ, 21.5 Hz, speech (低帧率)
- TAAE: CNN+T encoder, CNN+T decoder, FSQ, 25 Hz, speech
- Spectral Codecs: CNN encoder, CNN decoder, T-F, FSQ, 86.1 Hz, speech
- NAST: CNN+T encoder, CNN+T decoder, FSQ, 50 Hz, speech

## Survey 消融发现 [Mousavi et al. 2025, §4.2]

在 ESPnet-Codec 控制变量实验中 (Table 15-16):
- **FSQ@16kHz 在 UTMOS 和 DNSMOS 上超过 RVQ**: 感知质量指标更好 (FSQ-S: UTMOS 2.08, DNSMOS 3.06 vs RVQ-S: UTMOS 2.59, DNSMOS 3.35 at 16kHz — 但 RVQ 在 44.1kHz 时反超)
- **采样率交互效应**: RVQ 从 16kHz → 44.1kHz 一致性提升; FSQ 在 44.1kHz 反而部分退化
- **信号指标**: RVQ >> FSQ > SVQ 在 SDR, SI-SNR 上
- **设计建议**: FSQ 更适合 16kHz speech 场景; 高采样率场景应优先考虑 RVQ

## 局限性

- FSQ 表征的离散分布更难建模(compression cost 更高) → 下游 transformer 需要更强
- 低 codebook size (< 2^8) 时 VQ 略优(VQ 的表达力优势在小 codebook 时更明显) [Fig.3a]
- d 和 L 的选择仍需人工调参(虽然有启发式规则)
- Survey 消融发现 FSQ 在高采样率 (44.1kHz) 时性能反而退化,量化方法与采样率存在交互效应 [Mousavi 2025, §4.2]

---

> [!info] 来源
> 定义、机制、实验数据基于 Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", ICLR 2024 (arXiv:2309.15505)。TTS 应用部分基于 CosyVoice 3 论文。Survey 消融和变体信息基于 Mousavi et al., "Discrete Audio Tokens", TMLR 2025。
