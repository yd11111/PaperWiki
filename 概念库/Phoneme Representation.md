---
type: concept
title: "Phoneme Representation"
aliases: [音素表示, G2P, Grapheme-to-Phoneme, 音素, 语音学表示, IPA]
category: "representation"
tags: [TTS, text-analysis, phoneme, frontend, G2P, linguistics]
key_papers: ["[[论文笔记/Meta Learning TTS 7000 Languages|Meta Learning TTS 7000 Languages]]", "[[论文笔记/SpeechWeave|SpeechWeave]]", "[[论文笔记/DiaMoE-TTS|DiaMoE-TTS]]", "[[论文笔记/MAVE|MAVE]]", "[[论文笔记/ParsVoice|ParsVoice]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Text-to-Speech Pipeline]]", "[[Attention-based TTS]]", "[[Non-autoregressive TTS]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Phoneme Representation 是 TTS 系统中将文本转换为发音表示的前端处理。音素(phoneme)是语言中最小的区别性语音单位,是现代 neural TTS 声学模型最常用的输入形式。

**两种输入形式**:
- **Character (字符)**: 直接使用文字符号,如 "speech" → [s, p, e, e, c, h]
- **Phoneme (音素)**: 使用发音标注,如 "speech" → [s, p, iy, ch]

**音素的优势**: 消除发音歧义 (e.g., "read" 的过去式和现在式), 减少模型学习负担, 提高发音准确率。

## TTS 前端 (Text Analysis) 完整流程

```
Raw Text → [Text Normalization] → [Word Segmentation] → [POS Tagging] → [G2P] → [Prosody Prediction] → Phoneme + Prosody
```

### 各子任务

| 任务 | 描述 | 方法 |
|------|------|------|
| Text Normalization (TN) | 非标准文本 → 口语化 | 规则 / Seq2Seq |
| Word Segmentation | 分词 (中文等) | 统计/神经网络 |
| POS Tagging | 词性标注 | CRF / BiLSTM |
| G2P Conversion | 字 → 音素 | 词典 + Seq2Seq |
| Polyphone Disambiguation | 多音字消歧 (中文) | 上下文模型 |
| Prosody Prediction | 韵律边界预测 | CRF / Self-Attention |

### Grapheme-to-Phoneme (G2P)

**英文 G2P**:
- 字母语言,词典覆盖大部分常用词
- OOV 词使用神经网络 G2P 模型
- 代表: CMU Pronouncing Dictionary + Seq2Seq fallback

**中文 G2P**:
- 字形已覆盖全部"字符",但多音字需消歧
- 核心问题: Polyphone disambiguation (基于上下文)
- 代表: 条件神经网络 + 多级 embedding

## 在不同 TTS 范式中的角色

### SPSS 时代
- 完整语言学特征: phoneme + duration + POS + prosody boundary + ...
- 多级标注 (word, phrase, sentence level)

### End-to-end 时代 (Tacotron/FastSpeech)
- **简化**: 仅保留 G2P (或直接用 character)
- Tacotron: character 输入 (让模型自己学 G2P)
- FastSpeech: phoneme 输入 (确保发音准确)
- 混合方案: DeepVoice 3 同时使用 character + phoneme

### LLM-TTS 时代 (VALL-E/CosyVoice)
- 通常使用 phoneme (经 G2P 处理)
- 部分系统 (如 CosyVoice) 使用 text tokenizer 的 BPE tokens
- 趋势: 随着模型规模增大,character 输入也能工作

## Character vs Phoneme 的取舍

| 维度 | Character | Phoneme |
|------|-----------|---------|
| 预处理 | 无需 G2P | 需要 G2P 工具 |
| 发音准确性 | 依赖模型学习 | 显式保证 |
| OOV 处理 | 天然支持 | 需 G2P 推断 |
| 多语言 | 需处理不同字符集 | IPA 可统一 |
| 数据效率 | 较低 | 较高 |
| 产品部署 | 简单 | 需维护 G2P 模块 |

## 跨语言统一表示

- **IPA (International Phonetic Alphabet)**: 国际音标,可统一表示所有语言的发音
- **Byte representation**: 直接使用 UTF-8 bytes,无需任何语言学知识
- **Phoneme embedding mapping**: 将不同语言的音素嵌入映射到共享空间

## 关键论文

- Bisani & Ney (2008): Joint-sequence G2P model
- Deep Voice 1/2 (Arik et al., 2017): 完整神经前端 (包含 G2P)
- Char2Wav (Sotelo et al., 2017): 字符级端到端 (含隐式 G2P)
- FastSpeech (Ren et al., 2019): 确立 phoneme 作为标准输入
- LRSpeech (Li et al., 2020): 跨语言音素共享

## 相关概念

- [[Text-to-Speech Pipeline]]: phoneme representation 是 pipeline 前端的输出
- [[Duration Predictor]]: 预测每个 phoneme 的时长
- [[Attention-based TTS]]: phoneme/character 作为 encoder 输入
- BPE Tokenizer: LLM-TTS 中 text 的替代表示方式

## 演进

完整语言学特征 (SPSS; phoneme + POS + duration + prosody 标注) → 简化为 phoneme only (FastSpeech, 2019) → Character 直接输入 (Tacotron, 让模型学 G2P) → BPE text tokens (LLM-TTS, 2023+; 共享 LLM tokenizer)
