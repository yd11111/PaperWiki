---
type: dataset
title: "Emilia"
aliases: [Emilia Dataset]
domain: "Large-scale speech generation training"
scale: "101K+ hours, multilingual"
tags: [training-data, large-scale, multilingual, TTS]
used_by: ["[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Seed-VC|Seed-VC]]"]
metrics_reported_on: []
url: ""
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

Emilia 是一个大规模、多语言、多样化的语音数据集,专门为大规模语音生成模型训练而设计。由 He et al. (2024) 发布,涵盖多种语言和说话风格。

## 规模与特点

- 总量超过 101K 小时
- 覆盖多语言(中文、英文等)
- 高度多样化: 多种说话人、风格、录音条件
- 专为语音生成任务设计(而非 ASR)

## 使用此数据集的模型

- [[论文笔记/IndexTTS2|IndexTTS2]]: 使用 Emilia 作为主要训练数据来源,55K 小时训练数据中大部分来自 Emilia (30K 中文 + 25K 英文)
- [[论文笔记/MaskGCT|MaskGCT]]: 使用 Emilia 100K 小时 (50K 英文 + 50K 中文) 训练全部模型组件

## 来源

He et al., "Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation", IEEE SLT 2024.
