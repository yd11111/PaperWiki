---
type: dataset
title: "LibriQuote"
aliases: [LibriQuote Dataset]
domain: "Expressive audiobook speech for TTS"
scale: "18K+ hours (5.3K quotations + 12.7K narration), English"
tags: [training-data, expressive-speech, audiobook, TTS, evaluation, narrative]
used_by: ["[[论文笔记/LibriQuote|LibriQuote]]"]
metrics_reported_on: [WER, SIM-O, E-Sim, ContextMOS, Win-Rate, CMOS, MOS]
url: "https://huggingface.co/datasets/gasmichel/LibriQuote"
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-04
updated: 2026-06-04
---

## 概述

LibriQuote 是首个基于叙事结构 (narrative discourse) 构建的大规模有声书表现力语音数据集,由 Michel et al. (ACL 2026 Findings) 发布。核心思路: 按角色台词 (quotations) 与叙述 (narrations) 分割有声书音频,并用 LLM 从文本上下文中自动提取 speech verbs/adverbs 伪标签表征说话风格。详见 [[论文笔记/LibriQuote|LibriQuote 论文笔记]]。

## 规模与特点

- **台词 (Q)**: 3.51M 条,5,359 小时,平均时长 5.5s
- **叙述 (N)**: 3.87M 条,12,723 小时,平均时长 11.8s
- **高表现力子集 (Qf)**: 378K 条,379 小时 (有非中性 speech verb 或 adverb 伪标签)
- **测试集**: 5,598 条台词,7.4 小时,15 位未见说话人 (8 男 7 女)
- **开发集**: 2,921 条台词,5.1 小时,与训练集说话人重叠
- **说话人**: 3,314 个; 书籍: 2,991 本 (LibriVox Fiction)
- **采样率**: 16 kHz (提供原始音频链接支持更高采样率)
- **语言**: 英语

## 独特标注

每个台词配备:
1. **叙事语境**: 台词前后各一段文字 (~100 词),可用作表现力预测的条件信息
2. **Speech verb 伪标签**: 由 Phi-4 自动提取 (如 "whispered", "screamed"),精度 0.92
3. **Adverb 伪标签**: 由 Phi-4 自动提取 (如 "softly", "angrily"),精度 0.95

## 与其他有声书数据集的区别

| 数据集 | 规模 | 分割方式 | 标注 | 表现力 |
|--------|------|----------|------|--------|
| LibriSpeech | 1K h | 句边界 | 转写 | 低 (大量中性) |
| LibriTTS | 585 h | 句边界 | 转写+标点 | 低 |
| LibriHeavy | 50K h | 30s 句边界 | 转写+书文本 | 混合 (台词+叙述) |
| Emilia | 101K h | VAD | 转写 | 多样 (in-the-wild) |
| **LibriQuote** | **18K h** | **叙事结构** | **转写+语境+verb/adv** | **高 (台词分离)** |

## 来源

Michel et al., "Computational Narrative Understanding for Expressive Text-to-Speech", Findings of ACL 2026. arXiv: 2509.04072.

GitHub: https://github.com/deezer/libriquote
