---
type: task
title: "Instructed Speech Generation"
aliases: [指令式语音生成, Instruction-following TTS]
tags: [TTS, controllable, instruction-following, emotion, style]
key_approaches: ["Natural language instruction", "Fine-grained markers", "Style prompt"]
key_models: ["[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[CosyVoice 2]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/Seed-TTS|Seed-TTS]]"]
benchmarks: ["[[CV3-Eval]]"]
metrics: [Style Similarity, WER, MOS, Emotion Accuracy]
status: confirmed
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 问题定义

通过自然语言指令或结构化标记控制合成语音的情感、语速、音色风格、方言、角色等属性。例如:"用快乐的语气说"、"模仿机器人声音"、"用四川话说"。

与传统 TTS 的区别: 不仅克隆参考语音的特征,还能按指令生成参考中未出现的风格。

## 主流方法

1. **自然语言指令**: 将 style description 作为文本前缀输入 LLM,如 "You are Speaker A. Please talk happily."
2. **细粒度标记**: 在合成文本中插入控制标记(如 `[laughter]`、`[breath]`、`<strong>XXX</strong>`)
3. **Multi-task reward (MTR)**: 通过 DiffRO + SER/AED reward 强化情感表达

## 代表模型

- [[论文笔记/CosyVoice 3|CosyVoice 3]] (2025): 100+ 种风格,5000 小时 instruction-following 数据
- CosyVoice 2 (2024): 基础指令能力
- [[论文笔记/Seed-TTS|Seed-TTS]] (2024): 通过 Speaker Fine-tuning + Instruction Fine-tuning 支持情感/expressiveness/speaking rate/style 控制;RL-SER 变体将 SER accuracy 作为 reward,emotion control accuracy 从 ICL 的 0.44 提升至 0.80 (happy) [Table 9]
- [[论文笔记/IndexTTS2|IndexTTS2]] (2025): 通过 T2E 模块实现自然语言情感控制,将 DeepSeek-R1 的情感分布预测能力蒸馏到 Qwen-3-1.7b,支持 7 种情感的 soft 混合控制

## 评估

### Benchmarks

- Expresso dataset: 8 种 expressive speaking styles
- CV3-Eval Emotional Voice Cloning subset
- 内部数据集: 50+ 种情感/方言/角色风格

### Metrics

- Style Similarity (SIM): 风格相似度
- Emotion Accuracy: 情感分类准确率
- WER: 指令生成时的内容一致性
- MOS: 自然度

### 当前 SOTA

| 模型 | 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CosyVoice 3-1.5B | Style SIM | 81.06 | Internal Dataset | CosyVoice 3 Table 14 |
| CosyVoice 3-1.5B + DiffRO-EMO | Emotion Acc (happy, text-related) | 0.98 | CV3-Eval | CosyVoice 3 Table 9 |

## 开放问题

- 音色(timbre)尚不可通过文本指令控制,需要额外研究
- 歌唱风格生成尚未纳入
- 指令理解的精确度: 复杂组合指令(同时控制情感+语速+方言)的效果
- 评估难题: 缺乏标准化的 style controllability benchmark
