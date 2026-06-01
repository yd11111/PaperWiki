---
type: paper-note
title: "Survey: Towards Controllable Speech Synthesis in the Era of Large Language Models"
authors: [Tianxin Xie, Yan Rong, Pengfei Zhang, Wenwu Wang, Li Liu]
year: 2024
venue: "arXiv:2412.06602"
tags: [survey, controllable-TTS, LLM, style, emotion, prosody, instruction]
source: "/Users/xiangshu/PaperWiki/Sources/Survey-ControllableTTS-2024.pdf"
status: complete
created: 2026-06-01
updated: 2026-06-01
---

## 概述

首篇全面综述可控 TTS 方法的 survey,覆盖从传统控制技术到 LLM 时代自然语言提示驱动的新方法。系统分类模型架构、控制策略、特征表示,并总结数据集和评估方法。

## 核心贡献

1. **可控性任务分类** (6 维度): Prosody, Timbre, Emotion, Style, Language, Environment
2. **控制策略分类** (4 种): Style Tagging, Reference Speech Prompt, Natural Language Descriptions, Instruction-Guided Control
3. **架构演进路线**: CNN/RNN → Flow-based → LLM-based → Hybrid → Future Instruction-Aware
4. **特征表示对比**: Continuous (mel/latent) vs Discrete (codec tokens)

## 知识提取记录

### 新建概念页 (7)

| 概念页 | 来源章节 | 核心内容 |
|--------|----------|----------|
| [[LLM-based TTS]] | Sec 3.1.2 | Codec LM 范式, VALL-E 系列, decoder-only, hybrid |
| [[Style Transfer in TTS]] | Sec 2 + 3.2 | 控制策略全景, GST→MetaStyle→LLM |
| [[Emotion Control in TTS]] | Sec 2 + 3.2.1 | 层级情感, Emo-DPO, 跨说话人迁移 |
| [[Natural Language Description for TTS]] | Sec 3.2.3 | PromptTTS系列, NansyTTS, 环境扩展 |
| [[Instruction-Guided Speech Synthesis]] | Sec 3.2.4 + 3.2.5 | VoxInstruct, CosyVoice, InstructSpeech |
| [[Speech Attribute Disentanglement]] | Sec 3.3 | 对抗训练, information bottleneck, factorized codec |
| [[Global Style Tokens]] | Sec 3.2.2 + 引用 | 无监督风格发现, reference encoder, token bank |

### 更新概念页 (2)

| 概念页 | 更新内容 |
|--------|----------|
| [[Prosody Modeling]] | 添加 Style Tagging 控制策略 (离散/连续/隐空间/轮廓草图), 补充 LLM 时代局限 |
| [[Non-autoregressive TTS]] | 添加 NAR vs LLM 可控性对比表, 补充 Hybrid 趋势 |

## Survey 关键发现

### 架构演进趋势
```
Traditional CNN/RNN → 有限控制力, 显式特征工程
Flow-based (Matcha-TTS, F5-TTS) → 并行, 概率控制, 训练复杂
LLM-based (VALL-E, InstructTTS) → 自然语言控制, zero-shot, 慢
Hybrid (CosyVoice) → 直觉控制 + 高保真
```

### 控制策略演进
Style tagging (2018) → Reference prompt (2021) → NL description (2023) → Instruction-guided (2024)

### 未来方向
1. Instruction-guided fine-grained editing
2. Feature disentanglement for instruction control
3. Expressive multimodal synthesis (text + image + video → speech)
4. Zero-shot long speech with emotion consistency
5. Large-scale dataset generation (ChatGPT-assisted annotation)

## 关键引用

- VALL-E (Wang et al., 2023): LLM-TTS 开创
- CosyVoice (Du et al., 2024): Hybrid LLM + Flow
- VoxInstruct (Zhou et al., 2024): Instruction-to-speech
- PromptTTS (Guo et al., 2023): NL description 开创
- GST-Tacotron (Wang et al., 2018): 无监督风格控制
- MsEmoTTS (Lei et al., 2022): 多尺度情感
- NaturalSpeech 3 (Ju et al., 2024): Factorized codec
