---
type: concept
title: "LLM-enhanced ASR"
aliases: [LLM增强ASR, LLM Rescoring, LLM GER, LLM Generative Error Correction, 大模型增强语音识别, Hypotheses-to-Transcription, H2T]
category: "technique"
tags: [ASR, LLM, rescoring, error-correction, text-based-integration, N-best, speech-recognition]
key_papers: ["Chen et al., 2023a (HyPoradise)", "Radhakrishnan et al., 2023 (Whispering Llama)", "Ma et al., 2023", "Chen et al., 2023c", "Udagawa et al., 2022", "Hu et al., 2024b", "Yang et al., 2023a", "Ko et al., 2024", "Mu et al., 2024 (MMGER)", "Lin et al., 2024 (MoE)", "[[论文笔记/ComparingDiscrete-Continuous|Xu et al., 2024 (Discrete vs Continuous LLM-ASR)]]"]
origin_paper: "Yang et al., When LLM Meet Speech, 2025"
related_concepts: ["[[Speech-LLMIntegrationTaxonomy]]", "[[SpeechLanguageModel]]", "[[AudioUnderstanding]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

LLM-enhanced ASR 是 text-based integration 路线中利用大语言模型提升自动语音识别准确率的两类方法 [§3.2, §3.3]:

1. **LLM Rescoring**: LLM 对 ASR 模型生成的 N-best 假设列表进行重排序
2. **LLM Generative Error Correction (GER)**: LLM 基于 N-best 假设列表生成全新的转写结果

两者均属于 text-based integration 的进阶形式: LLM 本身不直接处理语音,而是处理 ASR 产生的文本假设。

## LLM Rescoring [§3.2]

### 原理

ASR 模型通过 beam search 生成 N-best 假设列表 H = {Y_1, Y_2, ..., Y_n},LLM 对每个假设评分,选择语言学上最连贯的:

```
Y* = argmax_{Y_i ∈ H} [(1-λ) log p_AM(Y_i|X) + λ log p_LLM(Y_i)]
```

其中 p_AM 是声学模型概率,p_LLM 是 LLM 评估的语言模型概率,λ 为插值权重。

### 发展历程

- 传统 LM rescoring 已有悠久历史 (McDermott et al., 2019; Shin et al., 2019; Salazar et al., 2020; Xu et al., 2022)
- LLM rescoring 是将此方向扩展到更强大的语言模型 (Chen et al., 2023c; Udagawa et al., 2022)
- LLM 的强语言理解能力使 rescoring 效果显著提升

### 特点

- **不修改 ASR 模型**: LLM 作为外部评分器
- **受限于 N-best 质量**: 最终结果不能超出 N-best 中最优假设
- **计算开销**: N 次 LLM 前向推理

## LLM Generative Error Correction (GER) [§3.3]

### 原理

与 rescoring 仅选择不同,GER 允许 LLM 基于假设列表 **生成全新的转写** [Fig 3b]:

```
Input: N-best hypotheses + instructional prompt
Output: 新生成的转写 (可能与任何假设都不同)
```

Chen et al. (2023a) 将此任务定义为 Hypotheses-to-Transcription (H2T),在 fine-tuning 和 few-shot 两种设置下探索。

### 核心优势

LLM 可以产生 **比所有初始假设质量更高** 的结果,因为:
- LLM 可综合多个假设的信息互补
- 利用 LLM 的世界知识纠正领域特定错误
- 可处理 ASR 系统性偏差 (如同音异义词)

### 代表系统

| 系统 | 关键特点 |
|------|----------|
| **HyPoradise** (Chen et al., 2023a) | 定义 H2T 任务,fine-tuning + few-shot |
| **Whispering Llama** (Radhakrishnan et al., 2023) | 跨模态 GER 框架 |
| **GERTranslate** (Hu et al., 2024c) | 将 GER 扩展到语音翻译 (S2TT) |
| **MMGER** (Mu et al., 2024) | 多模态 + 多粒度 GER,联合口音和语音识别 |
| **LpGER** (Ghosh et al., 2024) | 视觉条件 GER |
| **Neko** (Lin et al., 2024) | MoE 架构处理不同类型的生成错误 |

### 训练方式

- **Full fine-tuning**: 性能好但昂贵
- **PEFT (LoRA 等)**: PEFT 可优于 full fine-tuning (Chen et al., 2023a) -- 可能因为防止过拟合
- **In-context learning / few-shot**: 无需训练 (Hu et al., 2024b; Yang et al., 2023a; Ma et al., 2023)
- **MoE (Mixture-of-Experts)**: 不同专家处理不同类型错误 (Lin et al., 2024; Fedus et al., 2022)

## LLM Rescoring vs GER 对比

| 维度 | LLM Rescoring | LLM GER |
|------|---------------|---------|
| 输出来源 | 从 N-best 中选择 | 新生成,可超越所有假设 |
| 上限 | 受限于 N-best 最优 | 无上限 (理论上) |
| 复杂度 | 简单,仅评分 | 复杂,需要生成 |
| 适用扩展 | 主要用于 ASR | 可扩展到 S2TT (GERTranslate) |
| 训练需求 | 可零样本 | 通常需 fine-tune |

## 在 TTS 中的应用

LLM-enhanced ASR 本身不直接用于 TTS,但在以下管线中间接相关:
- **Cascaded speech pipeline**: ASR → LLM (GER修正) → LLM (理解/生成) → TTS,GER 提升前端输入质量
- **TTS 评估**: 合成语音的 WER 评估中,更准确的 ASR 提供更公平的评估
- **语音交互系统**: 语音助手的 ASR 前端使用 GER 减少下游错误

## 关键论文

- Chen et al., 2023a (HyPoradise): 定义 H2T 任务,GER 的奠基工作
- Radhakrishnan et al., 2023 (Whispering Llama): 跨模态 GER
- Chen et al., 2023c: 大规模 LLM rescoring 对比
- Udagawa et al., 2022: LLM rescoring 效果分析
- Hu et al., 2024b: LLM 作为 noise-robust ASR 学习器
- Mu et al., 2024 (MMGER): 多模态多粒度 GER
- Ko et al., 2024: 日语 ASR 的多通道 GER benchmark

## 相关概念

- [[Speech-LLMIntegrationTaxonomy]]: LLM-enhanced ASR 属于 text-based integration 路线
- [[SpeechLanguageModel]]: SpeechLM 的端到端方法是 LLM-enhanced ASR 的替代路线
- [[AudioUnderstanding]]: ASR 是 Audio Understanding 的核心任务之一

## 演进

传统 LM rescoring (N-gram/LSTM, 2019-) → LLM rescoring (GPT-2/BERT, 2022) → LLM GER (HyPoradise, H2T 定义, 2023) → PEFT for GER (LoRA 优于全微调, 2023) → 跨模态 GER (Whispering Llama, 2023) → MoE-GER (多专家处理多类错误, 2024) → 多模态 GER (MMGER, 视觉+语音, 2024) → 扩展到 S2TT (GERTranslate, 2024)
