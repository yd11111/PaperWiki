---
type: paper
tier: deep
title: "NVV-SuperBench: Beyond Words, Beyond Quality—Benchmarking Nonverbal Vocalizations in Speech Generation"
arxiv_id: "2604.16211"
source: "Sources/NVV-SuperBench.pdf"
authors: [Liumeng Xue, Weizhen Bian, Jiahao Pan, Wenxuan Wu, Yilin Ren, Boyi Kang, Jingbin Hu, Ziyang Ma, Shuai Wang, Xinyuan Qian, Hung-yi Lee, Yike Guo]
year: 2026
venue: "arXiv"
tags: [benchmark, NVV, nonverbal-vocalizations, evaluation, paralinguistic, TTS, speech-generation, taxonomy, LLM-as-judge]
concepts: ["[[TTSEvaluation]]", "[[ProsodyModeling]]", "[[NaturalLanguageDescriptionforTTS]]", "[[EmotionControlinTTS]]"]
models: []
tasks: ["[[InstructedSpeechGeneration]]"]
datasets: ["NVV-SuperBench"]
kb_context_sources: 6
status: draft
created: 2026-06-23
updated: 2026-06-23
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个 pending-review 实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[TTSEvaluation]], [[ProsodyModeling]], [[NaturalLanguageDescriptionforTTS]], [[EmotionControlinTTS]], [[SpokenDialogueEvaluation]], [[InstructedSpeechGeneration]] | 过滤: 0 | 未命中但可能相关: 无

**NVV 评估的先行工作**: [[TTSEvaluation]] 已收录 [[论文笔记/NV-Bench|NV-Bench]] (Ni et al., 2026),这是首个 NVV 生成标准化 benchmark,但仅覆盖 14 种 NVV 类型、1,651 条样本,使用 PCER (Paralinguistic CER) 作为核心指标。NVV-SuperBench 在三个维度大幅扩展: 类型覆盖 (45 vs 14)、数据规模 (4,500 vs 1,651)、评估维度 (multi-axis protocol vs 单一 PCER)。

**副语言发声建模**: [[ProsodyModeling]] 的"副语言发声"小节记录了 NVSpeech (Liao et al., 2025) 的 18 类 word-level PV 分类体系和 NV-Bench 的评估标准化贡献。NVV-SuperBench 的 45 类 taxonomy 是对 NVSpeech 18 类的进一步细化 (如 laughter spectrum 从单一 laugh 细分为 7 个子类型)。

**Prompt-based 控制范式**: [[NaturalLanguageDescriptionforTTS]] 覆盖了 caption-driven TTS (CapSpeech, Parler-TTS 等)。NVV-SuperBench 首次将 prompt-based 和 tag-based 两种 NVV 控制范式在同一 benchmark 中统一评估,填补了跨范式可比性的空白。

**情感控制上下文**: [[EmotionControlinTTS]] 记录了情感 TTS 的多种方法。NVV-SuperBench 的 crying spectrum 和 emotional vocalizations 类别直接关联情感表达,但评估角度从"情感是否正确"转向"NVV 是否存在且显著"。

**指令遵循评估**: [[InstructedSpeechGeneration]] (confirmed) 定义了 instructed speech generation 任务框架,包含 InstructTTSEval benchmark。NVV-SuperBench 使用 InstructTTSEval 作为 seed source 构建数据集,并专注于 NVV 维度的指令遵循评估。

## 速查

> [!summary] 速查
> - **一句话**: 首个统一评估 NVV 可控性、放置精度和感知显著性的双语 benchmark,覆盖 45 种 NVV 类型 × 15 个系统 × 三轴评估 (客观/主观/LLM-judge)
> - **路线**: 45-type taxonomy → 3-stage data pipeline (seed mining + controlled generation + validation) → 4,500 bilingual instances → 15 systems (7 prompt + 8 tag) → multi-axis evaluation (objective metrics + human listening + LLM multi-rater)
> - **指标**: ElevenLabs tag-based 最佳 (NVV PE 3.92, F1 0.720 EN) [Table 3, 4]; Gemini 2.5 Pro prompt-based EN 最佳 (NVV IF 2.74, NVV PE 2.68) [Table 4]; NVV 可控性与语音质量 decouple (CosyVoice 2: quality 4.35 但 NVV accuracy 1.65 ZH) [Table 4]
> - **可借鉴**: (1) GT-conditioned verification: 用 LLM 在已知 ground-truth NVV 类型约束下做 constrained editing 验证,比 open-ended 检测更可靠 [§2.4.1]; (2) Coverage-aware 评估: 不能只看 precision/recall, 需同时考虑系统支持的 NVV 类型覆盖率 [§4.1]; (3) 三阶段数据构建: seed mining → taxonomy-driven generation → iterative validation, 适用于任何需要细粒度类型平衡的 benchmark 数据集构建 [§2.3]
> - **局限**: (1) Gemini 2.5 Pro 既是数据构建工具又是 NVV verifier 又是被评估系统,存在潜在 bias [agent 解读]; (2) tag-based 评估依赖单一 LLM verifier (Gemini 2.5 Pro) 做 NVV 检测,无 cross-validation [§2.4.1]; (3) 仅评估单 NVV 实例,未涉及多 NVV 共现场景 [agent 解读]

## 核心问题

NVV (非语言发声,如笑声、叹气、抽泣) 是实现人类般语音的关键,但现有评估 benchmark 存在三个空白 [§1]:
1. **覆盖不足**: 现有系统和数据集的 NVV 类型高度偏斜 (大多只支持 laugh/cough 等少数类型),缺乏系统性分类学 [Table 2]
2. **评估维度单一**: 传统语音质量指标 (MOS, WER) 无法捕获 NVV-specific 能力 — 系统可能语音质量高但 NVV 可控性差 [§1]
3. **跨范式不可比**: prompt-based 和 tag-based 两种 NVV 控制范式在不同评估框架下测试,无法公平比较 [§1]

NVV-SuperBench 的目标是提供一个统一的双语 benchmark,将语音自然度/质量与 NVV 可控性/放置/感知显著性解耦评估。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

NVV-SuperBench 是一个 benchmark 框架,不是模型。由三部分组成 [Fig. 1]:
1. **Dataset**: 45-type NVV taxonomy + bilingual evaluation set (4,500 instances)
2. **Generation**: 支持 tag-based 和 prompt-based 两种控制范式
3. **Evaluation**: Multi-axis protocol (objective + subjective + LLM-based)

### 45-Type NVV Taxonomy

6 大类 45 种 NVV 类型,按产生机制和交际功能组织 [Table 1, §2.2]:

| 类别 | 类型数 | 代表 NVV | 设计动机 |
|------|--------|----------|----------|
| Respiratory | 10 | breath, sigh, gasp, yawn, snore | 呼吸模式控制情感强度 [论文原文, §2.2] |
| Throat/Physiological | 7 | cough, sneeze, hiccup, sniff | 半自主反射音,核心于真实交互 [论文原文] |
| Laughter Spectrum | 7 | chuckle → giggle → laugh → burst of laughter | 强度/时长/社交角色差异显著 [论文原文] |
| Crying Spectrum | 5 | whimper → sobbing → crying → wail | 传递不同程度的痛苦和脆弱 [论文原文] |
| Emotional Vocalizations | 7 | hum, groan, grunt, exclamation | 非词汇情感/态度信号 [论文原文] |
| Oral/Miscellaneous | 9 | lipsmack, tsk, sss, whisper, gulp | 词间衔接音,对话流畅度关键 [论文原文] |

分类学设计原则: 仅在变异具有感知显著性且已有数据集/控制接口区分时才引入细粒度子类型 (如 laughter 细分但 cough 不细分),以平衡覆盖度和诊断清晰度 [论文原文, §2.2]。

### 数据构建: 三阶段 Pipeline

**Stage I: Seed mining from human speech** [§2.3]
- 数据源: InstructTTSEval (双语表达性语料 + free-form captions)
- 工具: Gemini 2.5 Pro 做多模态标注 (NVV 识别 + span-level 定位 + caption 改写)
- 质量控制: 3 名标注员独立审核,majority vote 保留,分歧由第 4 名仲裁
- 产出: ~80 EN + ~30 ZH seed instances
- 选择 InstructTTSEval 而非直接合成的原因: 锚定于真实声学模式和语篇用法,减少纯合成样本的系统性伪影 [论文原文, §2.3]

**Stage II: Taxonomy-driven controlled generation** [§2.3]
- 每种 NVV 类型用 Gemini 2.5 Pro 生成双语候选文本
- 统一四字段 schema: text, text_with_nvv, caption_with_nvv, nvv_list
- 约束: 单类型约束 (nvv_list 只含一种 NVV,但允许同一句中多次出现)
- 多样性策略: 变化语篇设定 (对话/叙述/指令) 和风格线索 (中性/兴奋/平静)
- 去重: case-insensitive exact matching on text_with_nvv

**Stage III: Validation and replenishment** [§2.3]
- 自动一致性检查 (schema 合规 + NVV 标签匹配 + marker 一致性)
- 人工质量控制 (跨字段一致性 + 语境合理性 + 感知显著性筛查)
- 对易与文本感叹词混淆的 NVV: 额外筛除含 "ah/oh/uh/um/hmm" 等 token 的样本
- 未达 50 实例的类型触发补充生成 + 重新验证
- 最终: 45 × 50 = 2,250/语言, 共 4,500 实例

### 评估协议: Multi-Axis Protocol

#### 客观指标 [§2.4.1]

**通用指标** (两种范式共用):
- WER/CER↓: EN 用 Whisper-large-v3, ZH 用 paraformer-zh
- DNSMOS P.835↑: OVRL/SIG/BAK 三个子分

**Prompt-based 专用**:
- CLAP Score↑: caption-speech semantic alignment (CLAP embedding cosine similarity)

**Tag-based 专用** — GT-conditioned verification method:
- NVV 表示为 tuple (t, s): 类型 t + 位置 s
- **GT-conditioned verification**: 给 Gemini 2.5 Pro 提供 (合成语音, 参考 transcript, 目标 NVV 类型), verifier 输出 binary presence decision + constrained 单 marker 插入
- 选择 constrained editing 而非 open-ended 检测的原因: 预实验发现直接让 Gemini 检测 NVV 不可靠 — 幻觉事件和声学相似类别混淆严重 [论文原文, §2.4.1]
- Match 条件: tp,u = tg,u 且 |sp,u − sg,u| ≤ δ (位置容差,EN 按词、ZH 按字)
- **Coverage↑**: 系统支持的 NVV 类型数 / 45
- **Precision/Recall/F1↑**: 基于 TP/FP/FN 定义,FP 包含位置偏差过大和 spurious non-target 预测
- **NTD↓** (Normalized Tag Distance): 匹配样本的平均归一化位置误差
- 所有客观指标基于 3 次独立合成取均值 ± 标准差

#### 主观评估 [§2.4.2]

- 450 samples/语言 (10/NVV type), Prolific 平台, 97 名评分者, 5-point Likert + 0 分 (NVV absent)
- **共用指标**: Overall Naturalness↑, Overall Quality↑, NVV PE↑ (Perceptual Effect)
- **Prompt-based 专用**: Overall IF↑, NVV IF↑
- **Tag-based 专用**: NVV Accuracy↑, Overall Expression↑

#### LLM-based Multi-rater [§2.4.3]

- 使用 Gemini 2.5 Pro 作为 LLM judge
- 5 项控制: 匿名化 + rubric compliance + 低温 (0.2) + 4 rater multi-rater + comparative evaluation mode
- 与主观评估使用相同指标和评分量表
- 3 rounds × 3-fold partition 确保稳定性

### 关键设计选择

1. **为什么用 GT-conditioned 而非 open-ended NVV 检测?** 预实验表明 open-ended 检测幻觉率高且声学相似类别混淆严重;constrained editing (已知目标类型) 大幅提高检测可靠性 [论文原文, §2.4.1]

2. **为什么 taxonomy 仅细分 laughter 但不细分 cough?** Laughter 子类型在强度、时长和社交功能上的差异具有感知显著性且已被先前数据集/系统区分;physiological reflexes 保持粗粒度以减少稀疏性并保持跨数据集一致性 [论文原文, §2.2]

3. **为什么用 InstructTTSEval 作 seed source?** 它是双语表达性语料且提供 free-form captions (非固定 NVV 标签),允许灵活识别更广泛的 NVV 类型 [论文原文, §2.3]

4. **为什么客观指标 3 runs 但主观/LLM 只 1 run?** 人工和 LLM 评估成本显著更高 [论文原文, §2.4.1]

## 实验

### Benchmarked Systems (15 systems) [§3]

**Prompt-based (7)**: Parler-TTS Mini, Parler-TTS Large, CapSpeech, Qwen3-TTS, GPT-4o mini TTS, Gemini 2.5 Flash, Gemini 2.5 Pro

**Tag-based (8)**: ChatTTS, Higgs-Audio, Bark, Fish-Speech, Dia, CosyVoice 2, Orpheus TTS, ElevenLabs

### 核心结果

#### Table 3: 客观结果 (prompt-based)

| 指标 | 最佳 (EN) | 最佳 (ZH) | 出处 |
|------|-----------|-----------|------|
| WER/CER↓ | Qwen3-TTS 2.06 | Qwen3-TTS 4.08 | [Table 3] |
| DNSMOS OVRL↑ | GPT-4o mini TTS 3.56 | GPT-4o mini TTS 3.64 | [Table 3] |
| CLAP Score↑ | Qwen3-TTS 0.45 | GPT-4o mini TTS 0.43 | [Table 3] |

Gemini 系列在 WER/CER 上表现差 (Flash EN: 58.80, Pro EN: 5.40, Flash ZH: 16.45),但原因是 content-boundary violations (prompt leakage / hallucinated speech / 过长 NVV 段被 ASR 转写为重复非词汇 token),而非真正的可懂度问题 [论文原文, §4.1]。

#### Table 3: 客观结果 (tag-based)

| 指标 | 最佳 (EN) | 最佳 (ZH) | 出处 |
|------|-----------|-----------|------|
| Coverage↑ | Dia 0.29 | ElevenLabs 0.27 | [Table 3] |
| F1↑ | Orpheus TTS 0.728 | ChatTTS 0.703 | [Table 3] |
| NTD↓ | ChatTTS 0.0028 | Bark 0.0141 | [Table 3] |
| DNSMOS OVRL↑ | CosyVoice 2 3.67 (tied ChatTTS) | CosyVoice 2 3.63 | [Table 3] |

关键发现: **breadth-correctness trade-off** — ChatTTS coverage 最低 (0.02) 但 F1 极高,因为它只支持 laugh 一种 NVV,match 率自然高 [论文原文, §4.1]。Coverage-aware 评估不能仅看 correctness。

#### Table 4: 主观结果

| 指标 | 最佳 prompt-based (EN) | 最佳 tag-based (EN) | 出处 |
|------|----------------------|-------------------|------|
| NVV PE↑ | Gemini 2.5 Pro 2.68 | ElevenLabs 3.92 | [Table 4] |
| NVV IF↑ | Gemini 2.5 Pro 2.74 | — | [Table 4] |
| NVV Accuracy↑ | — | ElevenLabs 4.21 | [Table 4] |
| Overall Naturalness↑ | Gemini 2.5 Pro 4.07 | ElevenLabs 4.60 | [Table 4] |

**核心发现: NVV controllability decouples from speech quality** — CosyVoice 2 ZH: quality 4.35 (best tag-based) 但 NVV accuracy 1.65 (low); ElevenLabs quality 4.31 且 NVV accuracy 3.41 (both high) [Table 4]。

#### Table 5: LLM-based 评估

LLM judge 总体与人类判断一致 [§4.3]: ElevenLabs tag-based 两语言最佳; Gemini 2.5 Flash prompt-based NVV IF 最强。Qwen3-TTS EN quality 最高但 NVV controllability 不最强,再次印证 quality ≠ NVV controllability [Table 5]。

#### Table 6: Ablation (with vs without NVV control)

| 系统 | Lang | CMOS Naturalness | CMOS Quality | CMOS Expressiveness | 出处 |
|------|------|------------------|-------------|---------------------|------|
| ElevenLabs | EN | +0.65 | +0.59 | +0.93 | [Table 6] |
| Gemini 2.5 Pro | EN | −0.24 | −0.18 | +0.05 | [Table 6] |
| ElevenLabs | ZH | +0.33 | +0.25 | +0.52 | [Table 6] |
| Gemini 2.5 Pro | ZH | −0.14 | −0.33 | +0.05 | [Table 6] |

Tag-based (ElevenLabs): NVV 提升所有维度; Prompt-based (Gemini 2.5 Pro): NVV 反而略降 naturalness/quality,expressiveness 几乎不变 [Table 6]。论文解释: caption-only NVV prompting 增加生成负担但无可靠感知收益 [论文原文, §4.4]。

#### Figure 2: Per-type NVV PE Heatmap

**Easy types** (高 PE): laugh/laughter, breath/inhale/exhale, cough, sneeze, sigh, gasp [Fig. 2]
**Hard types** (低 PE): tsk, sss, lipsmack, gulps/swallows, mumble (low-SNR oral cues) + crying/sobbing/wail/whimper (long-duration affect) [Fig. 2]

两个瓶颈方向 [论文原文, §4.5]:
1. **Low-SNR oral cues**: 需要 masking-robust 高频细节建模
2. **Sustained affective NVVs**: 需要 duration/intensity trajectory control

Tag-based systems 展现 "frequency advantage": 多系统 inventory 中出现的 NVV 类型 (laugh, cough, sigh) PE 更高,罕见类型 (tsk, sss) PE 低 [论文原文, §4.5]。

### 系统级 Insight

- **ElevenLabs** (tag-based): 最均衡 — coverage 较高 + correctness 强 + PE 最高 [§4.1, §4.2]
- **Orpheus TTS** (tag-based EN): correctness 最强 (precision 0.687, F1 0.728) 但 coverage 中等 [Table 3]
- **ChatTTS**: coverage 极低 (0.02) → selective compliance 导致 F1 虚高 [§4.1]
- **CosyVoice 2**: speech quality 强但 NVV controllability 弱 — "质量与可控性分离"的典型 [§4.2]
- **Gemini 2.5 Pro/Flash** (prompt-based): NVV realization 强但 WER 膨胀 (content-boundary violations) [§4.1]

## 局限性

1. **Gemini 角色冲突**: Gemini 2.5 Pro 同时作为数据构建工具 (Stage I annotator)、NVV verifier (客观评估)、LLM judge (LLM 评估) 和被评估系统 — 存在系统性 bias 风险 [agent 解读]
2. **单 NVV 约束**: 每个实例仅含一种 NVV 类型,未评估多 NVV 共现/交互场景 [agent 解读,基于 §2.3 single-type constraint]
3. **tag-based verifier 依赖**: GT-conditioned verification 完全依赖 Gemini 2.5 Pro,无 cross-validation 或人工抽检 verifier 准确率 [agent 解读]
4. **无端到端生成模型**: 仅评估现有系统,未提出改进 NVV 合成的方法 — 纯 benchmark 论文 [agent 解读]
5. **Coverage 计算依赖系统自报**: 系统"支持"某 NVV 类型的定义取决于其公开文档/tag 列表,可能低估实际能力 [agent 解读]

## 点评

NVV-SuperBench 在三个方面推进了 NVV 评估的边界:

**贡献 1 — Taxonomy 标准化**: 45-type taxonomy 是目前最细粒度的 NVV 分类学,相比 NV-Bench 的 14 类显著扩展。6 大类的产生机制组织和子类型细化原则 (感知显著性 + 已有区分) 具有系统性 [§2.2]。

**贡献 2 — 评估解耦**: 将 speech quality 与 NVV controllability 解耦评估是核心方法论贡献。实证结果强有力地支持了这一解耦: CosyVoice 2 的 "高质量低可控" 和 ElevenLabs 的 "两者兼得" 形成鲜明对比 [Table 4]。这一发现对 TTS 系统开发有直接指导意义。

**贡献 3 — GT-conditioned verification**: 用 ground-truth NVV 类型约束 LLM verifier 的 constrained editing 方法,比 open-ended 检测更可靠。这一思路可迁移到其他需要在生成语音中检测特定事件的评估任务。

**主要顾虑**: Gemini 2.5 Pro 的多重角色 (数据标注 + 评估工具 + 被评估对象) 是最显著的方法论弱点。虽然论文在不同环节使用不同的 Gemini 功能 (multimodal annotator vs audio verifier vs LLM judge),但底层模型相同,可能产生有利于 Gemini 系列系统的系统性偏差。

## 可复用的 idea

1. **GT-conditioned verification for event detection**: 当需要在生成音频中检测特定事件时,不做 open-ended 检测,而是给 verifier 提供目标事件类型作为约束,仅做 binary 判断 + 位置标定。适用于任何"特定标签是否在输出中体现"的评估场景 [§2.4.1]

2. **Coverage-aware evaluation**: 不能仅看 precision/recall/F1 — 当系统只支持少数类型时,这些指标会虚高 (ChatTTS case)。需要同时报告 coverage 并做 coverage-normalized 分析 [§4.1]

3. **三阶段 benchmark data pipeline**: Seed mining (锚定真实数据) → Taxonomy-driven generation (LLM 生成平衡数据) → Iterative validation + replenishment (确保质量和覆盖)。可迁移到任何需要细粒度类型平衡的 benchmark 构建 [§2.3]

4. **NVV PE 0-5 scale with explicit 0**: 在评分量表中为"事件完全缺失"设置显式 0 分,区分于"存在但质量差"的 1 分。适用于任何评估特定事件是否出现的主观测试 [§2.4.2]

5. **CMOS ablation for NVV contribution**: 比较 with/without NVV 的 paired output,直接量化 NVV 对自然度/质量/表现力的边际贡献 [§4.4, Table 6]

## 审阅

> [!review] 审阅 (pending)
> **结论**: pending
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pending | — |
> | 可信赖 | pending | — |
> | 可区分 | pending | — |
> | 可定位 | pending | — |
> | 不污染 | pending | — |
> 
> Issues: pending
> 详见 `_review/NVV-SuperBench-review.yml`
