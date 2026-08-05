---
type: concept
title: "Speech Hallucination"
aliases: [语音幻觉, TTS Hallucination, TTS 幻觉, Content Error in TTS, Robustness in LM-based TTS]
category: "problem"
tags: [TTS, LLM-based, robustness, hallucination, content-fidelity, decoding]
key_papers: ["[[论文笔记/ECCD|ECCD]]", "[[论文笔记/FPO|FPO]]", "[[论文笔记/TTS-Transducer|TTS-Transducer]]"]
origin_paper: ""
related_concepts: ["[[LLM-basedTTS]]", "[[Speech-TextAlignment]]", "[[ContrastiveDecoding]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-08-05
updated: 2026-08-05
---

## 定义

Speech Hallucination(语音幻觉)指语音合成系统生成的语音**偏离目标文本**的现象。在 [[LLM-basedTTS]] 中尤为突出,因为自回归逐 token 生成会让局部偏差沿历史传播成持续性内容错误。

**典型表现类型** (Han et al. 2024; Liu et al. 2025 归纳):
- 误读 (mispronunciation)
- 替换 (substitution)
- 遗漏 (omission)
- 重复 (repetition)
- 意外续说 (unintended continuation)

**关键特征**: 幻觉语音在偏离文本后**往往仍然流畅、像语音**——即声学层面自洽,只是内容不忠实。[[论文笔记/ECCD|ECCD]] 据此提出"经验信息仍活跃、对齐信息未有效反映到解码决策"的解释视角。

## 在 TTS 中的应用

**为何 LM-based TTS 易幻觉**: acoustic token 编码细粒度、局部相关的语音模式,采样时常有多个概率相近候选;常规采样(top-k/nucleus)不区分候选来源(文本引导 vs 声学历史),可能选中"局部合理但文本不一致"的 token,一旦进入 AR 历史即传播。长句、重复、发音困难文本上更严重。

**缓解路线** (按干预阶段分):

| 路线 | 阶段 | 代表方法 |
|------|------|----------|
| 单调对齐 | 架构 | VALL-E R(单调对齐)、VALL-T / [[论文笔记/TTS-Transducer|TTS-Transducer]](生成式 transducer 显式建模单调生成) |
| 序列重排 | 架构 | ELLA-V(交织 phoneme 与 acoustic token) |
| 注意力约束 | 训练/推理 | guided-attention 训练(CTC + attention prior)、attention-constrained inference(操纵对齐注意力头) |
| 后训练分布对齐 | 后训练 | GFlowNet 分布对齐(Liu et al. 2025 EMNLP)、[[论文笔记/FPO|FPO]](token-level DPO)、attention guidance |
| 解码时控制 | 解码 | [[论文笔记/ECCD|ECCD]](首个 training-free decoding-time 方法,对比全条件/去文本预测增强对齐支持) |

**度量**: 通常以 WER/CER(ASR 转写 vs 目标文本)作为幻觉严重度的操作性代理;[[论文笔记/ECCD|ECCD]] 进一步提出分布级对齐影响 `Ii = KL(pE‖pA)` 与决策级对齐增益 `Gi = log pE(x̂)−log pA(x̂)` 用于分析幻觉起始。

## 关键论文

- Han et al. 2024 (VALL-E R): 单调对齐增强鲁棒性,系统归纳幻觉类型
- Liu et al. 2025 (GFlowNet, EMNLP): 后训练分布对齐缓解幻觉;指出幻觉语音仍保持流畅
- [[论文笔记/ECCD|ECCD]] (Liu et al. 2026): 首个 decoding-time 条件信息视角分析 + training-free 缓解
- [[论文笔记/FPO|FPO]]: token-level 选择性 DPO 降低 bad case ratio

## 相关概念

- [[LLM-basedTTS]]: 幻觉是该范式的公认软肋(稳定性问题)
- [[Speech-TextAlignment]]: text-present 推理减少幻觉,是幻觉与对齐信息的直接联系
- [[ContrastiveDecoding]]: [[论文笔记/ECCD|ECCD]] 用其缓解幻觉的解码技术底座

## 演进

架构级单调对齐/序列重排 (VALL-E R, ELLA-V, VALL-T, 2024) → 训练/后训练分布对齐 (GFlowNet, FPO DPO, attention guidance, 2025) → 解码时条件信息控制 (ECCD, 2026: 首个 training-free decoding-time 路线)
