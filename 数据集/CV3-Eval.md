---
type: dataset
title: "CV3-Eval"
aliases: [CV3-Eval benchmark, CosyVoice 3 Eval]
domain: "TTS evaluation (multilingual)"
scale: "9 languages x 500 samples + cross-lingual + emotion subsets"
tags: [benchmark, TTS, multilingual, zero-shot, evaluation, in-the-wild]
used_by: ["[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/Audiobook-CC|Audiobook-CC]]", "[[论文笔记/MOSS-TTS|MOSS-TTS]]", "[[论文笔记/IterateDifferentiate|I2D]]", "[[论文笔记/KineticOptimalTTS|GibbsTTS]]", "[[论文笔记/Raon-OpenTTS|Raon-OpenTTS]]", "[[论文笔记/FlowTTS-GRPO|FlowTTS-GRPO]]"]
metrics_reported_on: [CER, WER, Speaker Similarity, MOS, DNSMOS, Emotion Accuracy]
url: ""
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

CV3-Eval 是 CosyVoice 3 论文中提出的多语言零样本 TTS 评估 benchmark,设计用于解决现有 benchmark 的三大不足:
1. 参考音频来自 audiobook(过于干净),不反映真实场景
2. 大多仅覆盖中英文
3. 仅评估发音准确度和音质,忽略情感表达、韵律丰富度、跨语言能力

### 客观评估子集

- **Multilingual Voice Cloning**: 9 种语言(zh, en, ja, ko, de, fr, ru, it, es),每语种 500 样本,来源于 CommonVoice 和 FLUERS,包含噪声/静音不做过滤
- **Cross-lingual Voice Cloning**: zh/en/ja/ko 四语种双向组合
- **Emotion Cloning**: 来自 EmoBox 和 SeCap,含 happy/sad/angry 三种情感,分 text-related 和 text-unrelated 子集

### 主观评估子集

- Expressive Voice Cloning: 高表现力语音(极端情感/语速)
- Expressive Voice Continuation: 120 样本续写任务
- Chinese Accent Voice Cloning: 18 种中文方言

## 用途

面向 in-the-wild 场景的多维度 TTS 评估,特别适用于评估多语言覆盖、跨语言迁移和情感控制能力。

## 使用此数据集的模型

- [[论文笔记/CosyVoice3|CosyVoice 3]]: 唯一支持全部 9 语种的系统
- F5-TTS: 仅支持 zh/en
- Spark-TTS: 仅支持 zh/en
- GPT-SoVits: 仅支持 zh/en
- [[论文笔记/MOSS-TTS|MOSS-TTS]]: 支持 zh/en/ja/ko/de/es/fr/it/ru 全部 9 语种; MOSS-TTS-LT Clone zh CER 3.95%, en WER 4.35%
- [[论文笔记/KineticOptimalTTS|GibbsTTS]]: 支持 zh/en; UTMOS 3.238 (en), WER 4.110% (en), SIM 0.691 (en), UTMOS 2.438 (zh), CER 4.144% (zh), SIM 0.780 (zh)
- [[论文笔记/Raon-OpenTTS|Raon-OpenTTS]]: 仅 en; CV3-EN WER 3.92%, CV3-Hard-EN WER 6.15% / SIM 0.775 / DNSMOS 3.85 (1B); 全开放数据 (510K h) 训练,WER 和 SIM 均为 CV3-Hard-EN 评估模型中最优

## 注意事项

- 包含 hard-case 子集(罕见词、绕口令、领域术语)用于测试鲁棒性
- 参考音频刻意保留噪声和长静音,模拟真实应用场景
- 截至论文发表时尚未公开发布(需关注后续开源动态)
