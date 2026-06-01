---
type: concept
title: "Speech Factorization"
aliases: [语音因子分解, Timbre Disentanglement, Self-distillation for TTS, Speaker-Content Disentanglement, 说话人-内容解耦]
category: "training-strategy"
tags: [TTS, disentanglement, voice-conversion, self-distillation]
key_papers: ["[[论文笔记/Seed-TTS|Seed-TTS]]"]
origin_paper: "[[论文笔记/Seed-TTS|Seed-TTS]]"
related_concepts: ["[[Speech Tokenizer]]", "[[Conditional Flow Matching]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Speech Factorization 是将语音分解为多个独立、可操控属性(如 timbre、content、prosody)的过程。通过解耦这些属性,TTS 系统可以灵活组合不同说话人的音色与不同来源的内容/韵律,支撑 zero-shot voice conversion 和 factorized zero-shot TTS。

## Seed-TTS 的 Self-distillation 方案

Seed-TTS 提出了一种简洁的 self-distillation 方法实现 timbre disentanglement [§4.1]:

1. **构造训练对**: 在 diffusion module 推理时引入 speaker perturbation,生成与原始语音 S_ori 具有相同 content+prosody 但不同 timbre 的 S_alt
2. **重训 diffusion**: 输入 S_alt 的 token + S_ori 的 timbre reference → 目标恢复 S_ori 的 vocoder embedding
3. **核心约束**: S_alt 和 S_ori 共享 content/prosody 但 timbre 不同,迫使网络忽略 token 中的 timbre 信息,完全依赖外部 timbre reference

优势:
- 不改变 AR LM 结构,仅修改 diffusion module 的训练数据
- 利用 Seed-TTS 本身的 zero-shot 生成能力构造数据对,无需外部工具
- 效果显著: SIM 从 0.491 (w/o) 提升至 0.753 (w/) [Table 6, EN]

## 其他方法对比

此前的 disentanglement 方法:
- **Feature engineering**: bottleneck features, PPG (Chen et al., 2023; Wang et al., 2024a)
- **Specialized loss**: 对抗性损失强制移除说话人信息 (Ju et al., 2024)
- **Architecture tuning**: AutoVC (Qian et al., 2019), DiffVC (Popov et al., 2021)

Seed-TTS 的方案更为简洁,且可扩展到任何具备 diffusion/flow 模块的大规模 TTS 系统。

## 关键论文

- [[论文笔记/Seed-TTS|Seed-TTS]] (ByteDance, 2024): 首次提出 self-distillation via speaker perturbation

## 相关概念

- [[Speech Tokenizer]]: factorization 的对象(token 中编码了哪些信息)
- [[Conditional Flow Matching]]: self-distillation 作用于 CFM/diffusion module
- Voice Conversion: speech factorization 的核心下游应用
