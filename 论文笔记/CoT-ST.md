---
title: "CoT-ST"
aliases: [Chain-of-Thought Speech Translation, CoT Speech Translation]
authors: [Yexing Du, Ziyang Ma, Yifan Yang, Keqi Deng, Xie Chen, Bo Yang, Yang Xiang, Ming Liu, Bing Qin]
year: 2024
venue: arXiv
arxiv_id: "2409.19510"
source: "https://arxiv.org/abs/2409.19510"
tags: [speech-translation, chain-of-thought, SLM, curriculum-learning, Q-Former, Whisper, Qwen2]
level: deep
status: draft
created: 2026-06-03
updated: 2026-06-03
concepts: ["[[Speech Language Model]]", "[[Speech-Text Alignment]]", "[[Audio Understanding]]"]
models: ["[[模型库/Whisper|Whisper]]"]
datasets: []
kb_sources: ["[[Speech Language Model]]", "[[Speech-Text Alignment]]", "[[Audio Understanding]]"]
---

# CoT-ST: Enhancing LLM-based Speech Translation with Multimodal Chain-of-Thought

## KB 背景

- **[[Speech Language Model]]** [confirmed]: CoT-ST 属于 SLM 范式——冻结 Whisper encoder + 冻结 LLM (Qwen2-7B) + 可训练 Q-Former 投影层，是 speech-LLM integration 中 latent-representation-based 路线的应用 [论文原文]
- **[[Speech-Text Alignment]]** [confirmed]: CoT-ST 的核心贡献在于通过 CoT 分解实现隐式的 speech-text 对齐——先 ASR (speech→text)，再翻译 (text→target)，中间转写为两步间的桥梁 [论文原文]
- **[[Audio Understanding]]** [confirmed]: CoT-ST 支持 ASR + MMT + SRT 三任务，体现了 SLM 的多任务理解能力 [论文原文]

> [!summary] 速查
> - **一句话**: 通过三阶段课程学习 (ASR→MMT→SRT) 激活 SLM 的 Chain-of-Thought 推理能力，将语音翻译分解为"先转写再翻译"，SOTA 于 CoVoST-2 和 MuST-C
> - **路线**: speech → Whisper encoder (frozen) → Q-Former (80 queries, trainable) → Qwen2-7B LLM (frozen) → transcription + translation
> - **指标**: CoVoST-2 en-zh BLEU 47.7 (prev SOTA 45.2), MuST-C en-zh zero-shot BLEU 21.2 (prev 18.3), 3x inference speedup vs Qwen2-Audio [Table 2, 5]
> - **可借鉴**: (1) 将 CoT 应用于 speech translation 的思路——隐式中间 ASR 转写为翻译提供上下文; (2) 仅训练 projection 层即可达到 SOTA; (3) 课程学习策略的任务排序设计
> - **局限**: 仅限 8B 参数以下 LLM 验证; 仅 text 输出 (非 S2ST); 开源

## 1. 核心问题 / Core Problem

现有 SLM 的语音翻译 (ST) 工作主要通过直接 instruction fine-tuning 实现，但**忽略了 SLM 内在的推理能力** [§1]。

**关键洞察**: LLM 已展现出强大的 Chain-of-Thought (CoT) 推理能力 [Wei et al., 2022]，但这一能力在语音翻译领域尚未被充分利用。如果让模型先转写语音 (ASR)，再基于转写文本翻译 (MT)，相当于"链式思考"——转写提供了翻译的上下文信息 [Fig 1]。

**例证** [Fig 1]: 对于 "long-lived" 一词，直接翻译可能出错（多义词），但先转写 "The long-lived bridge still stands today" 再翻译，CoT 提供了语境消歧。

## 2. 方法论 / Methodology

### 2.1 模型架构 [§3.1, Fig 2]

| 组件 | 选择 | 是否训练 | 说明 |
|------|------|---------|------|
| Speech Encoder | Whisper-large-v3 | 冻结 | 提取高维语音特征 [§3.1] |
| Projection | Q-Former | **训练** | 80 个可训练 query，固定长度压缩 [Eq 2] |
| LLM | Qwen2-7B | 冻结 | 生成文本输出 [§3.1] |

**为什么用 Q-Former**: 将变长语音特征压缩为 80 token 固定长度，大幅减少 LLM 输入 token 数，加速推理 [§3.1]。

**为什么冻结 encoder + LLM**: 实验发现 Qwen2-7B 全量微调和 LoRA 微调都未超过仅训练 projection 的效果 [§4.3]。这说明**翻译能力主要来自 LLM 本身的机器翻译能力，而非 speech 适配** [§4.3]。

### 2.2 三阶段课程学习 [§3.2]

| 阶段 | 任务 | 输入 | 输出 | 目的 |
|------|------|------|------|------|
| Stage 1 | ASR | speech + `<\|en\|>` | transcription | 建立多模态对齐 [§3.2] |
| Stage 2 | MMT | speech + transcription + `<\|lang\|>` | transcription + translation | 增强跨语言能力 [§3.2] |
| Stage 3 | SRT | speech + `<\|lang\|>` | transcription + translation | 完全激活 CoT [§3.2] |

**设计理念**: 从简单到复杂渐进训练——ASR 最简单 (单模态对齐)，MMT 中等 (双模态输入降低难度)，SRT 最难 (仅语音输入实现 CoT 推理) [§3.2]。

### 2.3 提示设计 [Table 1]

极简提示格式:
- ASR: `<|en|>`
- MMT: `<transcription><|lang|>`
- SRT: `<|lang|>`

CoT 输出格式: `<transcription> <|lang|> <translation>` [Table 1]

## 3. 实验与结果 / Experiments

### 3.1 主要结果 [Table 2]

**CoVoST-2 (supervised fine-tuning)**:

| 方法 | De→ | Ja→ | Zh→ | Avg |
|------|-----|-----|-----|-----|
| Cascaded: Whisper+NLLB-3.3B | - | 19.0 | 32.0 | - |
| End-to-end: Qwen2-Audio | 29.9 | 28.6 | 45.2 | 34.5 |
| **CoT-ST** | **28.7** | **30.8** | **47.7** | **35.7** |

**MuST-C (zero-shot)**:

| 方法 | De | Ja | Zh | Avg |
|------|-----|-----|-----|-----|
| Qwen2-Audio | 22.4 | 7.8 | 18.3 | 16.1 |
| **CoT-ST** | **18.3** | **11.3** | **21.2** | **16.9** |

**关键发现**: CoT-ST 在 en→zh 零样本场景下超越 SOTA SFT 模型 [§4.3]，表明**翻译能力源于 LLM 模型本身** [§4.3]。

### 3.2 SLM 组合消融 [Table 3]

| Encoder | LLM | BLEU (en→zh) |
|---------|-----|------|
| Whisper-v2 | Qwen1.5-7B | 47.67 |
| Whisper-v3 | Qwen1.5-7B | 47.72 |
| Whisper-v3 | Qwen2-1.5B | 42.98 |
| Whisper-v3 | **Qwen2-7B** | **47.76** |

**发现**: LLM 的机器翻译能力是关键因素——Qwen2-1.5B vs Qwen2-7B 差距 4.78 BLEU [Table 3]。

### 3.3 推理速度 [Table 5]

| 模型 | 策略 | Batch 4 | Batch 8 |
|------|------|---------|---------|
| Qwen2-Audio | Greedy | 59s | OOM |
| CoT-ST | Greedy | 74s→**19s** (bs64) | 39s |
| CoT-ST | Beam (5) | 93s→56s (bs12) | 64s |

**3x 加速**: Q-Former 压缩到 80 token 大幅减少计算量 [§4.3]。

### 3.4 多任务性能 [Table 4]

| 任务 | WER/BLEU | 说明 |
|------|----------|------|
| ASR | 11.12% WER | 不影响 ASR 能力 [§4.4] |
| MMT | 0% WER / 56.50 BLEU | 给定正确转写→极高翻译分数 [§4.4] |
| SRT | 10.89% WER / 47.76 BLEU | CoT 完整流程 [§4.4] |

### 3.5 消融: 课程各阶段的贡献 [Table 7]

| 配置 | WER | BLEU |
|------|-----|------|
| CoT-ST (full) | 10.89 | 47.76 |
| w/o ASR | 18.92 | 46.52 |
| w/o MMT | 11.63 | 47.16 |
| w/o SRT | - | 40.72 |

**发现**: 去掉 SRT 阶段 (即不做 CoT 推理) BLEU 下降 7 分 [Table 7]，证明 CoT 分解对翻译质量至关重要。

## 4. 设计选择分析 / Design Analysis

### WHY: 为什么 CoT 有效

- **上下文消歧**: 转写提供完整句子语境，帮助处理多义词和歧义结构 [Fig 1, Case 1]
- **利用 LLM 内在翻译能力**: CoT 将 speech translation 分解为 ASR (SLM 擅长) + MT (LLM 擅长)，两步各发挥所长 [§4.3]
- **减少幻觉**: 转写锚定了实际内容，降低翻译阶段的幻觉风险 [Case 2]

### WHY: 为什么仅训练 projection

- LLM 的翻译能力已经足够强 (Qwen2-7B 的 MT 基线就很好) [§4.3]
- 全量微调反而导致收敛问题 [§4.3]
- LoRA 也未带来显著改善 [§4.3]
- 说明 speech translation 的瓶颈在**语音→文本桥接** (即 projection 的质量)，而非翻译能力本身 [§4.3]

## 5. 局限与未来方向

- **仅验证到 8B 参数**: 更大 LLM 可能进一步提升 [Limitations]
- **仅 text 输出**: 不支持 S2ST (speech-to-speech translation)
- **CoT 增加输出长度**: 需生成转写+翻译，输出 token 数翻倍
- **幻觉未完全消除**: Case 2 显示模型仍可能生成高频词替代正确翻译 [Case Study]

---

检索命中: [[Speech Language Model]], [[Speech-Text Alignment]], [[Audio Understanding]] | 过滤: 无 | 未命中但可能相关: [[Modality Adaptation for Speech LLM]]
