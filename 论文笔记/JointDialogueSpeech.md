---
type: paper
tier: deep
title: "Towards Joint Modeling of Dialogue Response and Speech Synthesis based on Large Language Model"
arxiv_id: "2309.11000"
source: "Sources/JointDialogueSpeech.pdf"
authors: [Xinyu Zhou, Delong Chen, Yudong Chen]
year: 2023
venue: "arXiv"
tags: [TTS, LLM, prosody, front-end, dialogue, spoken-dialogue, Chinese-TTS, prosodic-structure-prediction]
concepts: ["[[ProsodyModeling]]", "[[Text-to-SpeechPipeline]]", "[[PhonemeRepresentation]]", "[[LLM-basedTTS]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: ["ChatGPT", "ChatGLM2-6B", "SpanPSP"]
tasks: []
datasets: ["DataBaker"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文处于 TTS 前端(text analysis)与 LLM 能力交叉的早期探索节点。在传统 TTS Pipeline 中,前端负责将文本转换为语言学特征(G2P、韵律边界、词性标注等),主流方法依赖 BERT-scale (0.1B) 小模型 [[[Text-to-SpeechPipeline]]]。本文提出用 LLM (6B-175B) 替代传统前端,是 LLM 进入 TTS 领域的早期信号之一。

**已有认知对比**:
- [[ProsodyModeling]] (confirmed): 韵律建模的演进线从规则标注 → GST/VAE → FastSpeech 2 显式预测 → VITS 隐式 → LLM in-context learning。本文的 PSP 实验属于"显式韵律标注预测"任务,但用 LLM 替代传统 CRF/BERT 方法,是这条演进线上传统显式 → LLM 驱动的过渡节点。
- [[PhonemeRepresentation]] [待确认]: TTS 前端的完整流程包括 TN → 分词 → POS → G2P → 韵律预测。本文将其中 PSP 以及更广泛的语言学特征(pinyin、duration、pitch)统一编码为 JSON 格式,让 LLM 端到端学习。
- [[LLM-basedTTS]] (confirmed): 后续的 VALL-E / CosyVoice 等直接用 LLM 生成语音 token,跳过了传统前端。本文的思路是保留前端-声学模型-vocoder 三级架构,仅用 LLM 替换前端+对话模块,是一条"增强前端"而非"端到端替代"的路线。
- [[Speech-LLMIntegrationTaxonomy]] [待确认]: 本文属于 text-based integration 的变体——用 LLM 统一对话生成和语言学特征预测,输出仍是文本形式的标注,不直接生成语音 token。

**创新判断**: 本文的核心假设(LLM 的语言理解能力可迁移到 speech-related 前端任务)在 2023 年 9 月是较新的观点。后续 LLM-TTS 路线直接跳过了前端,证明了更激进的方案可行;但本文的前端增强思路在中文 TTS 等需要精细韵律标注的场景中仍有参考价值。

> 检索命中: [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓ | 过滤: [[Text-to-SpeechPipeline]](pending-review), [[PhonemeRepresentation]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review), [[Full-duplexSpokenDialogue]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 LLM 同时生成对话回复和 TTS 语言学特征(韵律边界/拼音/时长/音高),模仿人类语音产出中语法编码与音韵编码的并行机制
> - **路线**: 用户输入 → LLM (ChatGPT prompting 或 ChatGLM2-6B fine-tuning) → 对话回复 + JSON 格式的 character/pinyin/prosody/duration/pitch 特征 → (后接 acoustic model + vocoder,本文未实现)
> - **指标**: PSP Average F-Score: ChatGPT 80.12%, ChatGLM2-6B fine-tuned 82.38% vs SpanPSP (BERT, 0.1B) 79.80% [Table 1]; 联合建模 89.70% parsable, 77.70% matched prosody [Table 3]
> - **可借鉴**: (1) 将多种 TTS 前端特征统一编码为 JSON 字符串,用 Seq2Seq 范式让 LLM 一次性生成 — 简单有效的多任务编码技巧; (2) 用 ChatGPT 生成对话上下文来扩充单句语音数据集
> - **局限**: 仅做到语言学特征预测,未接声学模型/vocoder 验证端到端效果; 数据集仅 8k 单人朗读句,严重过拟合; JSON 自回归解码极慢 (15-40s/句); 2023 年 9 月后 LLM-TTS 直接跳过前端,本文路线未被后续采纳

## 核心问题

本文试图回答: **LLM 能否像人类一样,在生成对话回复的同时决定怎么"说"(韵律、发音等)?** 具体拆解为两个子问题:

1. **LLM 是否具备语音相关的语言学理解能力?** — 通过韵律结构预测(PSP)任务验证 [§3]
2. **LLM 能否同时生成对话内容和语音特征?** — 通过联合预测对话回复+多种语言学特征验证 [§4]

动机来自 Levelt (1993) 的人类语音产出模型: 人在说话时,"语法编码"(想说什么)和"音韵编码"(怎么说)是并行进行的 [§1, Fig 1(b)]。而当前 chatbot+TTS 级联管线中,两者是串行且独立的 — 对话模型不知道韵律,TTS 前端看不到对话上下文 [§1, Fig 1(a)]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文提出一个统一框架 [Fig 1(c)]: 用 LLM 同时扮演对话模型和 TTS 前端,输出对话回复+语言学特征,再接现有的声学模型和 vocoder 生成语音。但本文实际只实现了 LLM 部分,未实现后续的语音合成。

### 实验一: 韵律结构预测 (PSP)

**任务定义**: 给定中文句子,预测三级韵律边界 — Prosodic Word (#1), Prosodic Phrase (#2), Intonation Phrase (#3) [§3, Fig 2]。这是中文 TTS 前端的典型任务。

**方法 A — Prompting ChatGPT** [§3.1, Fig 3]:
- 构建结构化 prompt: (1) 语言学知识(韵律层级的形式定义) + (2) Few-shot 示例(最多 16 个) + (3) 系统消息引导
- [论文原文] 语言学知识对 #2 (PPH) 和 #3 (IPH) 的预测帮助最大,因为 #1 (PW) 通常出现在词边界相对容易,而 #2/#3 需要更深的语义理解 [§3.4, Table 2]
- 消融结果: zero-shot 41.3% → +4 examples 64.0% → +linguistic knowledge 72.7% → +selected examples 75.1% → 最终 16 selected + knowledge = 80.12% [§3.4, Fig 4]

**方法 B — Fine-tuning ChatGLM2-6B** [§3.2]:
- 将 PSP 形式化为 Seq2Seq 任务: 输入原句,输出标注了 #1/#2/#3 的句子
- 使用 P-tuning-v2 进行参数高效微调,交叉熵损失仅在输出 token 上计算 [§3.2]
- [论文原文] 与 BERT-based SpanPSP 的关键区别: BERT 方法把 PSP 当 token classification 问题(每个字符后判断边界类型),LLM 方法当 Seq2Seq 问题(直接生成带标注的序列) [§3.2]

### 实验二: 联合对话回复 + 语言学特征预测

**对话上下文生成** [§4.1]:
- 原始数据集 (DataBaker) 只有孤立句子,没有对话上下文
- 用 ChatGPT 为每句话逆向生成可能的用户输入 (类似 LongForm 方法) [§4.1]
- [agent 解读] 这是一个"先有答案再造问题"的数据增强策略,质量依赖 ChatGPT 的对话生成能力

**语言学特征提取** [§4.1, Fig 5]:
- 从 DataBaker 语料自动提取四类特征: character (字符)、pinyin (拼音)、prosody hierarchy (韵律边界)、duration (时长)、highest/lowest pitch (D-Value 音高) [§4.1]
- D-Value 是基于沈炯 (1985) 理论的对数音高尺度: D = 5 × log₂(F/F₀) [§4.1]

**数据编码** [§4.1, Fig 5 右]:
- 将所有特征编码为 JSON 格式字典的字符串序列
- 与对话回复文本拼接,作为 LLM 的学习目标
- [论文原文] 这种实现方式相比传统方法(不同输出用不同模型/不同任务头)更简洁,且与 RT-2 等将连续值编码为文本 token 的趋势一致 [§4.1]

**训练** [§4.2]:
- P-tuning 无法学习 JSON 格式输出,改用全参数微调 + 4-bit 量化 [§4.2]
- [论文原文] JSON 编码导致上下文长度大幅增加(最大 1.6k tokens vs PSP 的 128 tokens) [§4.2]

### 关键设计选择

1. **为什么用 LLM 而非 BERT 做前端?** — [论文原文] LLM (6B-175B) 参数量远超 BERT (0.1B),拥有更强的语义理解和世界知识,可能更好地处理需要上下文理解的韵律决策 [§1]
2. **为什么用 JSON 编码?** — [论文原文] 统一了多种异构输出(离散分类+连续值)到同一文本序列,实现了最简洁的多任务联合学习 [§4.1]
3. **为什么不直接端到端生成语音?** — [agent 解读] 2023 年 9 月时 LLM-based 端到端 TTS (如 VALL-E) 刚兴起,本文选择保守路线:保留传统前端-声学模型-vocoder 架构,仅用 LLM 替换前端模块。从后续发展看,端到端路线成为主流

### 训练策略

- PSP Prompting: 无训练,依赖 ChatGPT 的 in-context learning
- PSP Fine-tuning: P-tuning-v2 on ChatGLM2-6B,单卡 A100,8k 训练样本 [§3.3]
- Joint Prediction: 全参数微调 ChatGLM2-6B + 4-bit 量化,8k 样本 [§4.2]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| PSP Avg F-Score (prompting) | 80.12% (ChatGPT, 175B) | 79.80% (SpanPSP, BERT 0.1B) | DataBaker 10k | [Table 1] |
| PSP Avg F-Score (fine-tuning) | 82.38% (ChatGLM2-6B, 6B) | 79.80% (SpanPSP) | DataBaker 10k | [Table 1] |
| PSP PW #1 F-Score | 93.86% (ChatGLM2 FT) | 96.35% (SpanPSP DataBaker) | DataBaker | [Table 1] |
| PSP PPH #2 F-Score | 73.28% (ChatGLM2 FT) | 69.34% (SpanPSP DataBaker) | DataBaker | [Table 1] |
| PSP IPH #3 F-Score | 80.00% (ChatGLM2 FT) | 65.64% (SpanPSP DataBaker) | DataBaker | [Table 1] |
| Joint: Parsable Samples (test) | 89.70% | — | DataBaker | [Table 3] |
| Joint: Matched Characters (test) | 69.26% | — | DataBaker | [Table 3] |
| Joint: Matched Pinyin (test) | 86.29% | — | DataBaker | [Table 3] |
| Joint: Matched Prosody (test) | 77.70% | — | DataBaker | [Table 3] |

**关键发现**:
- LLM prompting (ChatGPT 175B) 仅凭 16 个示例+语言学知识就能超过 BERT-based SpanPSP (80.12% vs 79.80%) [Table 1]
- Fine-tuning ChatGLM2-6B (60x 大于 BERT) 进一步提升至 82.38%,尤其在 PPH (+3.94%) 和 IPH (+14.36%) 上大幅超越 [Table 1]
- ChatGLM2-6B prompting 完全失败 (N/A),说明小规模 LLM 的 instruction-following 能力不足以支撑 prompting [Table 1]
- 联合预测训练集拟合良好(95.90% parsable, 98.79% matched pinyin),但测试集明显过拟合(89.70% parsable, 86.29% matched pinyin),train-test gap 显著 [Table 3, Fig 6]

## 局限性

1. **未接声学模型验证**: 只做到语言学特征预测,没有实际合成语音来验证这些特征是否真能改善合成质量 [§5]
2. **严重过拟合**: 8k 训练样本远远不够,train-test gap 明显;数据来自单一女性朗读者,缺乏对话自然度 [§5]
3. **推理速度极慢**: JSON 格式自回归解码单句需 15-40+ 秒,难以实用 [§5]
4. **路线被后续发展超越**: VALL-E (2023.01) 等直接用 LLM 生成语音 token,跳过了传统前端,使"LLM 增强前端"路线失去了独立存在的必要性 [agent 解读]
5. **对话上下文是合成的**: 用 ChatGPT 逆向生成的用户输入质量不可控,可能引入不自然的对话模式 [agent 解读]
6. **评估不够充分**: 联合预测实验仅比较了 train/test 的 matching rate,缺乏与 baseline 的直接对比;也没有人工评估 [agent 解读]

## 点评

本文是 LLM 进入 TTS 领域的早期探索之一,核心 insight — "LLM 的文本语言理解能力可以迁移到语音相关任务" — 在当时具有前瞻性。Levelt 语音产出模型作为动机框架是有说服力的:人类确实在"想说什么"和"怎么说"之间并行处理,而级联管线割裂了这一过程。

但从技术实现看,本文更像一个 feasibility study 而非完整的系统方案。两个实验的价值不对等:PSP 实验(实验一)是完整的,有 baseline 对比,有消融,结论可信;联合预测实验(实验二)缺乏 baseline,过拟合严重,且未连接到实际语音合成,只能证明"LLM 能学会输出 JSON 格式的特征"。

从后续发展看,本文代表了一条"保守路线":保留 TTS 三级架构,仅用 LLM 增强前端。而 VALL-E、SpeechGPT 等代表的"激进路线"直接让 LLM 生成语音 token,跳过前端、声学模型甚至 vocoder 的边界,最终成为主流。但本文中 LLM 对中文韵律的理解能力(尤其是 PPH/IPH 上大幅超越 BERT)是一个有价值的发现,暗示了后续 LLM-TTS 为何能隐式学会韵律。

## 可复用的 idea

1. **JSON 编码多任务输出**: 将多种异构特征(离散标签+连续值)统一编码为 JSON 字符串,让 LLM 以 Seq2Seq 方式一次性生成。简单但有效的技巧,适用于需要同时输出多种结构化信息的场景
2. **逆向对话上下文生成**: 从单句数据出发,用 LLM 逆向生成"谁会说这句话之前说了什么",将孤立句子转化为对话数据。在缺乏对话数据的 TTS 研究中有实用价值
3. **语言学知识注入 prompt**: 将形式化的语言学定义(韵律层级规则)写入 system message,配合 few-shot 示例,显著提升 LLM 在专业标注任务上的表现 (+8.7% for PSP) [Fig 4]
