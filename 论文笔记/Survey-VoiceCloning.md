---
type: paper-note
title: "Voice Cloning: Comprehensive Survey"
authors: [Hussam Azzuni, Abdulmotaleb El Saddik]
affiliation: MBZUAI / University of Ottawa
year: 2025
venue: "arXiv:2505.00579"
tier: card
tags: [survey, voice-cloning, TTS, speaker-adaptation, zero-shot, few-shot, multilingual, deepfake]
created: 2026-06-02
updated: 2026-06-02
---

## 概要

Azzuni & El Saddik 系统梳理了 voice cloning 领域的研究进展,覆盖 2015-2024 年间从 speaker adaptation 到 zero-shot voice cloning 的演进。Survey 的核心贡献在于建立了一套标准化的 voice cloning 术语体系,将方法分为四大类: Speaker Adaptation、Few-shot Voice Cloning、Zero-shot Voice Cloning、Multilingual Voice Cloning,并按子课题 (disentanglement、speaker representation、parameter efficiency 等) 组织文献。同时汇编了评估指标 (Table X) 和语音数据集 (Table IX)。

## 知识提取成果

### 新建概念页 (4 个)

| 概念 | 核心内容 |
|------|----------|
| [[VoiceCloningTaxonomy]] | 四分类体系: adaptation / few-shot / zero-shot / multilingual |
| [[SpeakerVerification]] | 双重角色: TTS 评估指标 + 安全/训练组件 |
| [[Anti-spoofingandDeepfakeDetection]] | Voice cloning 的伦理维度与检测技术 |
| [[SpeakerAdaptation]] | 基于微调的说话人声音复制方法 |

### 更新概念页 (2 个)

| 概念 | 更新内容 |
|------|----------|
| [[SpeakerEmbedding]] | 补充 voice cloning survey 视角: SECS/SV-EER 评估体系、speaker encoder 在三类 cloning 中的角色差异 |
| [[SpeechFactorization]] | 补充 voice cloning survey 中跨三类方法的 disentanglement 技术汇总 |

## 关键分类体系 (Fig. 3)

```
Voice Cloning
├── Speaker Adaptation (微调整个/部分模型)
│   ├── Disentanglement
│   ├── Speaker Representation & Verification
│   ├── Handling Speech Variability
│   ├── Untranscribed Speech
│   ├── Parameter Efficiency
│   └── Non-English
├── Few-shot Voice Cloning (少量数据微调, 几秒~5分钟)
│   ├── Disentanglement
│   ├── Untranscribed Speech
│   ├── Parameter Efficiency
│   ├── Speaker Representation
│   ├── Optimization-based
│   └── Non-English
├── Zero-shot Voice Cloning (无微调, speaker encoder)
│   ├── Disentanglement
│   ├── Emotion Transfer
│   ├── Parameter Efficiency
│   ├── Generalizable Speech Synthesis
│   └── Non-English
└── Multi-lingual Voice Cloning
    ├── 2 languages
    ├── 3-5 languages
    └── 5+ languages
```

## 关键定义 (Survey 提出的标准化术语)

1. **Voice Cloning**: Replicating a specific person's voice using a TTS system
2. **Speaker Adaptation**: Fine-tuning a TTS model to replicate a specific user's voice using limited data
3. **Few-shot Voice Cloning**: 遵循 speaker adaptation 核心原则,区别在于参考音频量 (几秒到最多5分钟)
4. **Zero-shot Voice Cloning**: 无需微调 TTS 模型,通过专用 speaker encoder 从短音频片段生成相似语音

## 评估指标体系 (Table X)

### 客观指标
| 指标 | 类别 | 描述 |
|------|------|------|
| MCD ↓ | Speech Quality | 频谱相似度 (MFCC 欧氏距离) |
| CER/WER ↓ | Speech Quality | 通过预训练 ASR 评估可懂度 |
| SECS ↑ | Speaker Similarity | Speaker embedding 余弦相似度 |
| SV-EER ↓ | Speaker Similarity | 说话人验证系统的等错误率 (FAR=FRR) |
| GPE ↓ | Speech Expression | 基频偏差 (≥20% 为错误) |
| VDE ↓ | Speech Expression | 清浊音判定错误率 |
| FFE ↓ | Speech Expression | GPE 或 VDE 出错的帧百分比 |
| RTF ↓ | Inference Speed | 生成1秒音频的实际耗时 |

### 主观指标
| 指标 | 量表 | 描述 |
|------|------|------|
| MOS ↑ | 5分制 | 自然度、相似度、可懂度综合评分 |
| CMOS | 7分制 (-3~3) | 两系统对比评分 |
| MUSHRA ↑ | 100分制 | 含锚点和参考的多系统评价 |
| AB preference | 百分比 | 两系统 A/B 偏好测试 |

## Speaker Encoder 常用架构

Survey 中提到的 SECS 计算所用 speaker encoder:
- X-vector [119]: TDNN + 统计池化
- GE2E [67]: 端到端 speaker verification loss
- XLSR-53 [214]: 跨语言自监督表示
- WavLM [148]: 大规模自监督预训练
- TitaNet-L [120]: 1D 深度可分离卷积
- SpeechBrain toolkit [215], [216]
- Resemblyzer package

## 数据集一览 (Table IX 精简)

### 英文
| 数据集 | 规模 | 说话人 | 采样率 |
|--------|------|--------|--------|
| LibriTTS | 586h | 2456 | 24kHz |
| VCTK | 44h | 109 | 48kHz |
| LibriSpeech | 982h | 2484 | 16kHz |
| Hi-Fi TTS | 292h | 10 | 44.1kHz |
| ESLTTS | 37h | 134 | 24kHz |

### 中文
| 数据集 | 规模 | 说话人 | 采样率 |
|--------|------|--------|--------|
| AISHELL-3 | 85h | 218 | 44.1kHz |
| DiDiSpeech-1 | 572h | 4500 | 48kHz |
| CSMSC | 12h | 1 | 48kHz |

### 多语言
| 数据集 | 规模 | 说话人 | 语言 |
|--------|------|--------|------|
| CommonVoice | 2508h | 58250 | Multilingual |
| MLS | 50.5kh | 6332 | Multilingual |
| Emilia | 101kh | N/A | Multilingual |

## 未来方向 (Section VII)

1. **缺乏统一 benchmark**: 不同论文使用不同测试集,难以横向比较
2. **Conversation-style synthesis**: 当前主要合成阅读风格,对话风格研究不足
3. **Fine-grained disentanglement**: content-timbre 解耦已有进展,prosody/emotion 细粒度解耦仍需深入
4. **Misuse mitigation**: 需要同步发展检测算法以限制滥用

## 核心 observation

- Few-shot TTS (Table IV) 中各系统在 NAT/QUAL 和 SIM 之间存在 trade-off,没有一个模型同时最优
- Zero-shot TTS (Tables V/VI) 是增长最快的方向,codec-based 架构成为主流 (post SPEAR-TTS 和 VALL-E)
- Multilingual voice cloning 从 2 语言逐步扩展到 50+ 语言 locale (Table VIII)
- Speaker verification 在 voice cloning 中的角色已从纯评估发展为训练信号 (feedback constraint, adversarial training)
