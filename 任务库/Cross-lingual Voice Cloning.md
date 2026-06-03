---
type: task
title: "Cross-lingual Voice Cloning"
aliases: [跨语言语音克隆, Cross-lingual TTS, Cross-lingual Speech Synthesis]
tags: [TTS, cross-lingual, voice-cloning, multilingual]
key_approaches: ["Multilingual LLM + shared tokenizer", "Language-agnostic speaker embedding"]
key_models: ["[[论文笔记/CosyVoice|CosyVoice]]", "[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/Meta Learning TTS 7000 Languages|Meta Learning TTS 7000 Languages]]"]
benchmarks: ["[[CV3-Eval]]"]
metrics: [WER, CER, Speaker Similarity, MOS]
status: confirmed
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 问题定义

给定说话人 A 的一段语音(语言 L1)和目标文本(语言 L2),合成保持说话人 A 音色但使用语言 L2 的语音。例如:输入中文参考语音 + 英文目标文本,输出该说话人说英文。

核心挑战:
- 需要将说话人音色与语言特征解耦
- 不同语言的音素体系、韵律模式差异大
- 字符系统差异(如日文汉字 → 假名转换)可能引入额外错误

## 主流方法

1. **多语言统一 tokenizer + LLM**: 使用覆盖多语言的 speech tokenizer,LLM 在多语言数据上训练后自然具备跨语言能力
2. **Continual pretraining for polyglot**: 在多语言辅助数据上持续预训练,使单语说话人获得多语能力

## 代表模型

- [[论文笔记/CosyVoice 3|CosyVoice 3]] (2025): 支持 zh/en/ja/ko 等多方向跨语言克隆,WER 显著优于前代
- CosyVoice 2 (2024): 仅支持中英,日文方向因字符转换问题表现差

## 评估

### Benchmarks

- [[CV3-Eval]] Cross-lingual subset: 包含 zh/en/ja/ko 四种语言的双向组合

### Metrics

- WER/CER: 内容一致性
- Speaker Similarity: 跨语言后音色保持度
- MOS: 自然度

### 当前 SOTA

| 模型 | 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CosyVoice 3-1.5B + DiffRO | WER to-en (from zh) | 2.98 | CV3-Eval | CosyVoice 3 Table 7 |
| CosyVoice 3-1.5B + DiffRO | WER to-zh (from en) | 5.09 | CV3-Eval | CosyVoice 3 Table 7 |
| CosyVoice 3-1.5B | WER en2zh | 8.01 | CV3-Eval | CosyVoice 3 Table 8 |
| Qwen3-TTS-12Hz-1.7B | WER zh-to-en | 2.77 | CV3-Eval | Qwen3-TTS Table 7 |
| Qwen3-TTS-12Hz-1.7B | WER en-to-zh | 4.77 | CV3-Eval | Qwen3-TTS Table 7 |
| Qwen3-TTS-12Hz-1.7B | WER zh-to-ko | 4.82 | CV3-Eval | Qwen3-TTS Table 7 |

## 开放问题

- 低资源语言方向的跨语言克隆质量仍有提升空间
- 口音迁移 vs 口音消除的权衡
- 语调模式在跨语言时如何自然过渡
