---
type: concept
title: "Contrastive Decoding"
aliases: [对比解码, CD, Contrastive Decoding for Generation]
category: "technique"
tags: [decoding, training-free, hallucination, LLM, TTS, inference-time]
key_papers: ["[[论文笔记/ECCD|ECCD]]"]
origin_paper: ""
related_concepts: ["[[SpeechHallucination]]", "[[LLM-basedTTS]]", "[[Speech-TextAlignment]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-08-05
updated: 2026-08-05
---

## 定义

Contrastive Decoding (CD) 是一类 **training-free 的解码时干预**技术: 用一个 strong **expert** 分布对比一个 weak **amateur** 分布,在 expert 定义的可行集 (plausible set,通常是 expert 的 top-k) 内,用二者的对数似然比 `log pE/pA` 放大 expert 相对 amateur 被强化的信息,从而抑制 amateur 中不良的模式(退化、幻觉、不忠实)。

最初由 Li et al. (2023) 为开放式**文本生成**提出。核心公式(常规 CD):在可行集 `Vhead = TopK(pE)` 内 score = `log pE/pA`,集外为 `−∞`。

## 在 TTS 中的应用

**首次适配到语音**: [[论文笔记/ECCD|ECCD]] (Liu et al. 2026) 是首个把 CD 用于**自回归 acoustic-token 生成**的工作。它用**同一个 speech LM** 的两次前向构造对比:
- expert `pE = p(x|x<i, a, Tp, Tt)`(全条件)
- amateur `pA = p(x|x<i, a)`(去文本,作为经验信息代理)

**关键改造(为何常规 CD 不能直接用于语音)**: 常规 CD 把 amateur 支持一律当**负证据**减去,但语音的 pA 保留了有用的发音/韵律/时长/连续性信息,一刀切压制会破坏声学结构、导致**时长压缩**([[论文笔记/ECCD|ECCD]] 消融显示常规 CD 把 CER 反升、UAC 压到 native 的 66%)。ECCD 因此改为:
1. **Expert anchor**: 保留 `log pE` 作锚,对比只做修正而非替换;
2. **Positive-only enhancement** `[log pE/pA]_+`: 只增强 pE>pA 的候选,不惩罚 amateur 偏好的候选;
3. **Experience calibration** `(1−Ci)`: 用集合级经验兼容系数 ECC 自适应调节增强强度。

## 关键论文

- Li et al. 2023: 提出 Contrastive Decoding(文本生成)
- DoLa (Chuang et al. 2024): 对比不同 model layers 提升事实性
- Context-aware decoding (Shi et al. 2024): 对比有/无外部上下文,减少幻觉
- VCD (Leng et al. 2024): 视觉对比解码,缓解 LVLM 物体幻觉
- Audio-aware decoding (Hsu et al. 2025) / Temporal CD (Li et al. 2026): 移除或时间平滑音频条件的对比,用于 large audio-language models
- [[论文笔记/ECCD|ECCD]] (Liu et al. 2026): 首个 acoustic-token 生成的 CD 适配 + experience preservation/calibration

## 相关概念

- [[SpeechHallucination]]: CD 在语音上的主要应用目标
- [[LLM-basedTTS]]: ECCD 的作用对象(自回归 token 生成)
- [[Speech-TextAlignment]]: expert/amateur 的差异本质是文本对齐信息的增量

## 演进

文本生成 CD (Li et al. 2023) → 跨层/跨上下文变体 (DoLa, context-aware decoding, 2024) → 视觉幻觉 (VCD, 2024) → 音频理解 (audio-aware / temporal CD, 2025-2026) → 自回归 acoustic-token 生成 (ECCD, 2026: expert anchor + positive-only + experience calibration)
