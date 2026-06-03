---
title: "NaturalVoices"
aliases: [NaturalVoices Dataset, NaturalVoice Dataset]
authors: [Ali N. Salman, Zongyang Du, Shreeram Suresh Chandra, Ismail Rasim Ulgen, Carlos Busso, Berrak Sisman]
year: 2024
venue: arXiv
arxiv_id: "2406.04494"
source: "https://arxiv.org/abs/2406.04494"
tags: [dataset, voice-conversion, spontaneous-speech, emotional-speech, podcast-data, data-pipeline]
level: deep
status: draft
created: 2026-06-03
updated: 2026-06-03
concepts: ["[[Speaker Embedding]]", "[[Emotion Control in TTS]]", "[[Prosody Modeling]]"]
models: []
datasets: []
kb_sources: ["[[Speaker Embedding]]", "[[Emotion Control in TTS]]", "[[Prosody Modeling]]"]
---

# Towards Naturalistic Voice Conversion: NaturalVoices Dataset with an Automatic Processing Pipeline

## KB 背景

- **[[Speaker Embedding]]** [confirmed]: NaturalVoices 的 pipeline 包含说话人识别模块 (PyAnnote diarization + global speaker consolidation)，为 VC 模型提供 speaker identity 标注 [论文原文]
- **[[Emotion Control in TTS]]** [confirmed]: 数据集包含 4 类情感标注 (neutral/sad/angry/happy) + 3 维情感属性 (arousal/dominance/valence)，可用于情感语音合成和转换 [论文原文]
- **[[Prosody Modeling]]** [confirmed]: 自发语音数据集天然包含丰富韵律变化 (停顿、犹豫、笑声)，与朗读语音截然不同 [论文原文]

> [!summary] 速查
> - **一句话**: 首个大规模自发、表达性、情感语音数据集 (3,846h, 2,467+ speakers)，附带自动化数据标注 pipeline，面向 Voice Conversion
> - **路线**: MSP-Podcast → Diarization+ASR → Speaker Recognition → Emotion/SNR/Sound Event Detection → NaturalVoices 数据集
> - **指标**: SV% 95.65 (S2S) / 89.16 (U2U), CER 17.01/18.55, WER 27.20/30.43; MOS Quality 3.17, Intelligibility 3.77 [Table 2, 4]
> - **可借鉴**: (1) 从 podcast 自动构建大规模 VC 数据的完整 pipeline; (2) 多维标注 (情感+SNR+声事件+年龄性别) 的数据集设计范式
> - **局限**: 标注全自动 (弱标签); 仅英语; 以 VC 为目标验证, TTS 应用尚未探索

## 1. 核心问题 / Core Problem

Voice Conversion (VC) 研究长期依赖**朗读/表演语音数据集** (VCTK 44h/110 speakers, ESD 10 speakers) [§1]，存在三大不足：
1. **缺乏自发性**: 朗读语音无法反映真实对话中的犹豫、笑声、情感波动
2. **情感单一**: 大多仅含中性语音 (VCTK) 或表演情感 (ESD)
3. **规模有限**: 说话人数量和时长不足以训练泛化性强的 VC 模型

**解决方案**: 从 podcast 数据自动提取并标注大规模自发语音数据集 [§1]。

## 2. 自动数据标注 Pipeline / Automatic Data Sourcing Pipeline [§3]

| 模块 | 工具 | 功能 |
|------|------|------|
| Diarization + ASR | Faster Whisper + CTranslate2 | 分段 + 转写 [§3] |
| Speech Detection | Temporal CNN [ref 22] | 区分语音与音乐 [§3] |
| Speaker Recognition | PyAnnote [ref 23, 24] | 局部→全局说话人合并 [§3] |
| Gender & Age | Transformer-based [ref 25] | 说话人属性预测 [§3] |
| Emotion Category | PEFT-SER (WavLM+LORA) [ref 26] | 4 类情感分类 [§3] |
| Emotion Attribute | WavLM regression [ref 27] | Arousal/Dominance/Valence 连续值 [§3] |
| SNR | WADA-SNR [ref 28] | 信噪比估计 [§3] |
| Sound Event | AST [ref 29] | 500+ 种声事件检测 [§3] |

**对齐精化**: MFA (Montreal Forced Aligner) 生成 phone-level 对齐 [§3]。

## 3. NaturalVoices 数据集 [§4]

### 3.1 规模对比 [Table 1]

| 数据集 | 语音类型 | 时长 | 说话人数 | SNR | 声事件 | 情感 | 文本 |
|--------|----------|------|---------|-----|--------|------|------|
| VCTK | 朗读 | 44h | 110 | 无 | 无 | 仅中性 | 文本 |
| ESD | 朗读 | - | 10 | 无 | 无 | 有 (表演) | 文本 |
| **NaturalVoices** | **自发** | **3,846h** | **>2,467** | **有** | **有** | **有 (自然)** | **ASR** |

### 3.2 数据分布特征

- **情感分布** [Fig 2d]: Neutral 占主导，Angry 和 Happy 显著多于 VCTK；Arousal/Dominance/Valence 分布远比 VCTK 宽广 [Fig 2a-c]
- **SNR 分布** [Fig 3]: 0-100 dB 全范围覆盖，VCTK 集中于 20-40 dB
- **单说话人数据**: 2,600h 单说话人语音，1,300h 带全局说话人标签 [§4]

### 3.3 潜在应用 [§4.2]

| 应用方向 | 说明 |
|----------|------|
| Expressive VC | 表达性声音转换 (利用丰富情感标注) [§4.2] |
| Emotional VC | 情感转换 (利用 4 类情感 + VAD 属性) [§4.2] |
| Noisy-to-Noisy VC | 噪声环境下的 VC (利用多样 SNR 分布) [§4.2] |
| Spontaneous Speech Modeling | 自发语音建模 (停顿、犹豫、disfluency) [§4.2] |
| TTS | 表达性 TTS 训练数据 (配合 ASR 转写) [§4.2] |
| Weakly-Supervised Training | 弱标签半监督学习 [§4.2] |

## 4. 实验验证 / Experiments [§5]

### 4.1 设置

- VC 模型: TriAANVC (encoder-decoder + Triple Adaptive Attention Normalization) [ref 32]
- Vocoder: ParallelWaveGAN [ref 33]
- 评估: Speaker Verification (SV%), WER, CER, MOS [§5.1]

### 4.2 核心结果 [Table 2]

| 训练数据 | SV% (S2S/U2U) | CER% (S2S/U2U) | WER% (S2S/U2U) |
|---------|---------------|-----------------|-----------------|
| VCTK | 71.10 / 80.30 | 16.42 / 12.38 | 25.74 / 19.82 |
| NaturalVoices | 95.65 / 89.16 | 17.01 / 18.55 | 27.20 / 30.43 |
| NaturalVoices_VCTK | 80.75 / 82.76 | 19.31 / 19.26 | 30.70 / 31.02 |
| NaturalVoices_Large | 96.78 / 73.80 | 19.68 / 22.26 | 30.37 / 33.31 |

**关键发现** [§5.2]:
- NaturalVoices 在 speaker similarity (SV%) 上大幅超越 VCTK (+24.55% S2S) [Table 2]
- 可懂度 (WER/CER) 略有代价，因自发语音更难建模 [Table 2]
- 用 NaturalVoices 训练的 vocoder 提升了 out-of-distribution 表现 [§5.2]
- MOS: Quality 3.17, Intelligibility 3.77 (vs 原始语音 Quality 4.38, Intelligibility 4.79) [Table 4]

### 4.3 SNR 影响 [Table 3]

- Low SNR (0-20 dB): SV 85.11%, CER 32.69%, WER 46.91%
- High SNR (80-100 dB): SV 93.65%, CER 17.01%, WER 27.20%
- 验证了数据集对噪声鲁棒性研究的价值 [§5.2]

## 5. 设计选择分析 / Design Analysis

### WHY: 为什么选择 podcast 而非 YouTube

- Podcast 音质更高 (更少背景噪声) [§1]
- 说话人数量多且每人数据充足 (建模 speaker identity) [§1]
- 话题多样，自然覆盖多种说话风格和情感 [§1]

### WHY: 为什么不用人工标注

- 人工标注 3,800+ h 数据不可行 [§4.2]
- 自动标注虽为弱标签，但可用于弱监督/半监督学习 [§4.2]
- Pipeline 设计为模块化，随新模型发展可替换组件 [§3]

## 6. 局限与未来方向

- **仅英语**: 未覆盖其他语言 [论文原文]
- **弱标签质量**: 所有标注来自自动模型，存在噪声 [§4.2]
- **TTS 验证缺失**: 论文仅验证了 VC 应用，TTS 实验留作未来工作 [§6]
- **数据分布偏差**: Podcast 说话人以北美英语为主，多样性有限 [论文原文]

---

检索命中: [[Speaker Embedding]], [[Emotion Control in TTS]], [[Prosody Modeling]] | 过滤: 无 | 未命中但可能相关: 无
