---
type: task
title: "Cross-lingual Voice Cloning"
aliases: [跨语言语音克隆, Cross-lingual TTS, Cross-lingual Speech Synthesis]
tags: [TTS, cross-lingual, voice-cloning, multilingual]
key_approaches: ["Multilingual LLM + shared tokenizer", "Language-agnostic speaker embedding"]
key_models: ["[[论文笔记/CosyVoice|CosyVoice]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/MetaLearningTTS7000Languages|Meta Learning TTS 7000 Languages]]", "[[论文笔记/XTTS|XTTS]]", "[[论文笔记/LatinX|LatinX]]", "[[论文笔记/Cross-LingualF5-TTS|Cross-Lingual F5-TTS]]", "[[论文笔记/DiaMoE-TTS|DiaMoE-TTS]]", "[[论文笔记/DialoSpeech|DialoSpeech]]", "[[论文笔记/PFluxTTS|PFluxTTS]]", "[[论文笔记/X-Voice|X-Voice]]"]
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

- [[论文笔记/CosyVoice3|CosyVoice 3]] (2025): 支持 zh/en/ja/ko 等多方向跨语言克隆,WER 显著优于前代
- CosyVoice 2 (2024): 仅支持中英,日文方向因字符转换问题表现差
- [[论文笔记/Cross-LingualF5-TTS|Cross-Lingual F5-TTS]] (Liu et al., 2025): 首个 NAR flow-matching 跨语言系统,通过 MMS forced alignment + speaking rate predictor 移除 prompt transcript 依赖,cross-lingual test-en WER 2.496%, SIM 0.543
- [[论文笔记/PFluxTTS|PFluxTTS]] (Pankov et al., 2026): 混合 DG+AF flow-matching 系统,通过推理时向量场融合兼得稳定性和自然度; FLUX 架构 sequence prompt embeddings 实现跨语言克隆 (无需 prompt transcript); 33 语言 cross-lingual 评估 WER 6.9%, SPK-SIM 0.68 [Table 2]
- [[论文笔记/X-Voice|X-Voice]] (Xu et al., 2026): 0.4B NAR flow-matching 系统,基于 F5-TTS 扩展,通过两阶段训练 (Stage 1 420K hrs multilingual CFM + Stage 2 SFT with synthetic prompts) 实现 30 语言 transcript-free 跨语言克隆。Dual-Level Language Injection (time-level concat + text-level FiLM) 有效抑制口音泄漏,Decoupled Scheduled CFG + A-Warmup 平衡发音精度与音色保持。Cross-lingual WER: en→it 4.70, zh→ru 2.85, ko→en 2.15, it→en 2.31 [Table 7]

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

## 早期探索: 纯文本跨语言 TTS

- [[论文笔记/LearningToSpeakFromText|Saeki et al. (IJCAI 2023)]]: 在多语言文本上做 MLM 预训练 + 冻结 language-aware embedding,仅用文本数据 (无语音) 实现未见语言的零样本 TTS。未见语言 (西班牙语) CER 11.69%,与 oracle 的 5.32% 仍有 gap [Table 3]。证明了跨语言迁移在极低资源场景的可行性,但效果依赖于已见语言中是否存在相似语言

## 开放问题

- 低资源语言方向的跨语言克隆质量仍有提升空间
- 口音迁移 vs 口音消除的权衡
- 语调模式在跨语言时如何自然过渡
- [[论文笔记/AccentVector|Accent Vector (2026)]] 提出无口音数据的口音控制: 通过 LoRA 微调 XTTS-v2 + task vector 算术实现细粒度口音强度控制和混合口音合成,但声调语言(普通话)效果受限
- [[论文笔记/Tibetan-TTS|Tibetan-TTS (He et al., 2026)]]: 验证了 AR LM + Flow Matching 骨干模型从中英预训练跨语言迁移到极低资源藏语(卫藏方言)的可行性,MOS 4.28-4.35,但仅覆盖单一方言且缺少客观指标
