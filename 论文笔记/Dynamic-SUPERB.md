---
type: paper
tier: deep
title: "Dynamic-SUPERB Phase-2: A Collaboratively Expanding Benchmark for Measuring the Capabilities of Spoken Language Models with 180 Tasks"
arxiv_id: "2411.05361"
source: "https://arxiv.org/abs/2411.05361"
authors: [Chien-yu Huang, Wei-Chih Chen, Shu-wen Yang, Andy T. Liu, Chen-An Li, Yu-Xiang Lin, Wei-Cheng Tseng, Anuj Diwan, Yi-Jen Shih, Jiatong Shi, William Chen, Xuanjun Chen, Chi-Yuan Hsiao, Puyuan Peng, Shih-Heng Wang, Chun-Yi Kuan]
year: 2024
venue: "arXiv Preprint (National Taiwan Univ / UT Austin / CMU / NTU / INRS-EMT)"
tags: [benchmark, speech-evaluation, spoken-language-model, instruction-following, task-taxonomy, audio-understanding, music-understanding, LLM-as-judge, universal-model]
concepts: ["[[AudioUnderstanding]]", "[[SpeechLanguageModel]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[SpeechLanguageModel]], [[ProsodyModeling]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓(confirmed), [[ProsodyModeling]]✓(confirmed) | 过滤: [[AudioUnderstanding]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review) | 未命中但可能相关: [[TTSEvaluation]]

**已有知识要点**:
- [[SpeechLanguageModel]]: SpeechLM 系统可接受语音输入并通过指令完成多种下游任务 (ASR, SER, SID, QbE 等);Dynamic-SUPERB 是评估此类通用模型的关键 benchmark ✓
- [[AudioUnderstanding]] [待确认]: Dynamic-SUPERB Phase-2 被列为 180 任务的大规模 benchmark,覆盖 speech/music/audio 三个域;已有概念页 Audio Understanding 中已引用 Dynamic-SUPERB 作为核心评估平台

## 速查

> [!summary] 速查
> - **一句话**: Dynamic-SUPERB Phase-2 是迄今最大的 instruction-based 通用语音/音频模型评估 benchmark,包含 180 个任务 (91 个新增),覆盖 speech/music/audio 三大域,并提供详细的任务分类体系和 LLM-as-judge 评估 pipeline
> - **路线**: Task Collection (community contribution, 145 proposals → 91 accepted) → Task Formulation (instruction + audio + optional text) → Task Taxonomy (speech 8 domains + audio/music 9 domains) → Evaluation (LLM judge for classification, LLM post-processor for regression, direct metrics for sequence generation)
> - **指标**: 覆盖 180 tasks (vs SUPERB 13, HEAR 19, Phase-1 55) [Table 1]; 评估 8 个模型 (SALMONN-7B/13B, Qwen-Audio, Qwen2-Audio, WavLLM, LTU-AS, GAMA, MU-LLaMA) + Whisper-LLaMA baseline; 无单一模型在所有任务上表现优异 [§5, Fig 3]
> - **可借鉴**: (1) 社区协作式 benchmark 构建方法 (call for tasks); (2) 多粒度任务分类体系 (speech 8 域 + audio/music 9 域); (3) LLM-as-judge 评估 pipeline for classification + regression + sequence generation 三类任务的统一评估
> - **局限**: 缺少语音生成任务 [§6 Limitations]; 任务分类体系可能不完整; LLM-as-judge 在某些新任务上泛化性未验证; 核心任务结果与原始 SUPERB 不直接可比 (指令格式不同)

## 核心问题

**Dynamic-SUPERB Phase-2 要解决什么问题?** [论文原文]

评估通用语音/音频模型面临三个挑战 [§1, §2.2]:
1. **任务覆盖不足**: 现有 benchmark (SUPERB 13 tasks, HEAR 19 tasks) 任务数量有限,无法全面评估 instruction-based 通用模型的能力 [§2.2, Table 1]
2. **固定不可扩展**: 传统 benchmark 任务集固定,无法跟上快速发展的研究需求 [§1]
3. **跨领域缺失**: 语音模型通常忽略音乐和环境音频,但 "spoken language models outperformed music language models in certain music tasks" 提示跨领域能力值得评估 [§5.1]

**核心定位**: [论文原文] "Our goal is to evaluate universal models that meet the following criteria: (1) accept speech, music, or audio as input; (2) perform tasks without fine-tuning; (3) follow natural language instructions" [§3.1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 任务收集与组织

#### 社区协作 [§3.3]

[论文原文] Phase-2 通过 call for tasks 从全球研究社区收集任务 [§3.3]:
- 2024 年 3 月发起征稿,共收到 145 个提案,接受 91 个新任务 [§3.3]
- 通过 GitHub 透明管理提交流程,编辑审核+迭代改进 [§3.3]
- 任务数据上传至 HuggingFace space [§3.3]

#### 任务定义 [§3.2]

[论文原文] 每个任务包含 [§3.2, Fig 1c]:
1. **Text instruction**: 引导模型执行任务的自然语言指令 (每个任务有多条不同指令)
2. **Audio component**: 输入/输出中包含至少一个音频元素
3. **(Optional) Text component**: 非指令的文本输入/输出

[论文原文] 所有分类标签以文本形式输出 (而非固定类别集),回归任务输出自然语言数值描述 [§3.2]

### 任务分类体系 [§3.4]

[论文原文] Dynamic-SUPERB 建立了详细的两级任务分类体系 [§3.4, Fig 2]:

**Speech 域 (8 个子域)** [§3.4.1]:
| 子域 | 代表任务 | 评估维度 |
|------|---------|---------|
| Speech Recognition | 多语言 ASR, 自发语音, code-switching | 内容理解 |
| Speaker & Language | 说话人验证, 声纹识别, 语种识别 | 说话人/语言特征 |
| Spoken Language Understanding | 情感分析, 语音翻译 | 语义理解 |
| Phonetics, Phonology, Prosody | 音素识别, 重音分类, 口音 | 语音结构 |
| Paralinguistics | 情感识别, 非言语检测 | 副语言信息 |
| Speech Enhancement | 降噪, 去混响 | 信号质量 |
| Speech Disorders | 口吃检测, 语言障碍分类 | 病理语音 |
| Safety & Security | Deepfake 检测, 欺骗检测 | 安全性 |

**Audio & Music 域 (9 个子域)** [§3.4.2]:
| 子域 | 代表任务 |
|------|---------|
| Music Classification | 乐器/流派/情感分类 |
| Pitch Analysis | 音高估计, 和弦分类 |
| Rhythm Analysis | 节拍追踪 |
| Singing Analysis | 歌词识别, 声乐技巧分类 |
| Quality Assessment | MOS 预测 |
| Sound Event | 环境声分类, 动物声识别 |
| Safety | 歌声 Deepfake 检测 |
| Spatial Audio | 距离/位置估计 |
| Signal Characteristics | 音效检测, 时长预测 |

#### 核心任务 [§3.4.3]

[论文原文] 为降低全量评估门槛,选取了核心任务子集 [§3.4.3]:
- 从 SUPERB (speech), MARBLE (music), HEAR (audio) 三个经典 benchmark 中选取
- 手工编写 instruction,转为 Dynamic-SUPERB 格式 [§3.4.3]

### 评估方法 [§4.2]

[论文原文] 三类任务采用不同评估策略 [§4.2]:

1. **Classification**: LLM (GPT-4o, temperature=0) 作为 **referee**,判断模型输出是否与 ground truth 匹配 [§4.2]
   - Chain-of-thought 推理后输出最终判断 [§4.2]
   - Accuracy = LLM 判为正确的比例 [§4.2]

2. **Regression**: LLM (GPT-4o) 作为 **post-processor**,将自然语言输出转为数值 [§4.2]
   - 无法解析 → 标记 "N/A",计入 N/A rate [§4.2]
   - [论文原文] "A higher N/A rate indicates that the model struggles with following instructions" [§4.2]

3. **Sequence Generation** (如 ASR): 直接用原始指标 (WER, CER 等) 评估未处理的模型输出 [§4.2]

### 评估模型 [§4.1]

[论文原文] 评估了 8 个公开可用模型 [§4.1]:
- **Speech+Audio models**: SALMONN (7B/13B), Qwen-Audio, Qwen2-Audio, WavLLM, LTU-AS [§4.1]
- **Music models**: MU-LLaMA, GAMA [§4.1]
- **Baseline**: Whisper-LLaMA (Whisper-v3-large → LLaMA3.1-8B cascade) [§4.1]

## 实验

| 指标 | 最优模型 | 得分 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SUPERB PR (PER) | SALMONN-13B | 2.0 | SUPERB 核心 | [Table 2] |
| SUPERB ASR (WER) | SALMONN-13B | 2.8 | SUPERB 核心 | [Table 2] |
| SUPERB ER (Acc) | Qwen2-Audio | 68.1 | SUPERB 核心 | [Table 2] |
| SUPERB KS (Acc) | Qwen-Audio | 60.5 | SUPERB 核心 | [Table 2] |
| SUPERB SV (Acc) | SALMONN-13B | 93.5 | SUPERB 核心 | [Table 2] |
| Speech Recognition domain | Whisper-LLaMA | best | 全域 | [Fig 3] |
| Speaker & Language domain | Qwen2-Audio | best | 全域 | [Fig 3] |
| Paralinguistics domain | WavLLM | best | 全域 | [Fig 3] |
| Music Classification domain | MU-LLaMA | best | 全域 | [Fig 3] |

**关键实验发现**:

1. **无模型全面胜出** [§5, Fig 3]: [论文原文] "No single model excels across all tasks" — 每个模型仅在特定域表现好,泛化能力均不足

2. **ASR cascade 仍是强基线** [Fig 3]: [论文原文] 在 speech recognition 和 spoken language understanding 两个域,没有模型超越 Whisper-LLaMA baseline,因为 "text more explicitly represents semantic information than speech" [§5.1]

3. **跨域迁移能力** [§5.1]: [论文原文] 语音模型在音乐分类等任务上竟然超越专用音乐模型 (GAMA, MU-LLaMA),说明 "training on diverse data enhances performance across domains" [§5.1]

4. **Speaker Diarization 全面失败** [Table 2]: [论文原文] 所有模型在 speaker diarization 上表现极差 (100% N/A rate for most),这是当前通用语音模型的一个显著盲区

5. **Query-by-Example 极具挑战** [Table 2, §5.2]: [论文原文] 所有模型 QbE 准确率低于随机猜测 (50%),说明通过示例匹配是当前模型的能力空白

## 局限性

1. **缺少生成任务** [§6]: [论文原文] Phase-2 聚焦理解任务,缺少 speech generation 类任务的评估
2. **分类体系局限** [§6]: [论文原文] 任务分类体系可能不完整,"new domains may emerge as the benchmark grows"
3. **LLM 评估泛化性** [§6]: [论文原文] LLM-as-judge 在当前任务上与人类评估相关性好,但 "it may not generalize to all future tasks"
4. **核心任务不可直接比较** [§5.2]: [论文原文] SUPERB 核心任务经过 instruction 格式改写,结果与原始 SUPERB 不直接可比
5. **模型能力上限未知**: [agent 解读] 评估的模型规模较小 (7B-13B),未包含 GPT-4o-audio 等更强大的商业模型

## 点评

**历史地位**: [agent 解读] Dynamic-SUPERB Phase-2 是当前语音/音频领域最大最全面的 benchmark,其 180 个任务的规模远超之前所有 benchmark。更重要的是它建立了 "community-driven, dynamically expanding" 的 benchmark 模式,可持续吸收新任务保持前沿性。

**方法论贡献**:
1. **社区协作式 benchmark**: call for tasks 模式使 benchmark 能跟上研究发展速度,避免固定 benchmark 过时
2. **任务分类体系**: 首次为语音/音频/音乐建立了统一的多级分类体系,为模型能力分析提供了结构化框架
3. **LLM-as-judge 统一评估**: 用 GPT-4o 统一处理 classification/regression/generation 三类任务的评估,解决了 instruction-following 模型输出格式多样的问题

**关键警示**: [agent 解读]
- "No single model excels across all tasks" 的结论对整个领域有重要启示: 当前的通用语音模型离真正的 "universality" 还有很大距离
- ASR cascade baseline 在语义理解域仍然最强,质疑了端到端 speech LLM 在理解任务上的必要性

## 可复用的 idea

1. **Community-driven benchmark 模式**: call for tasks + GitHub 管理 + editor review 的组织流程,可迁移到其他领域的 benchmark 构建
2. **LLM-as-judge pipeline**: classification (referee) + regression (post-processor) + generation (direct metrics) 的三分法评估方案
3. **Relative-score-based 跨任务比较**: 以 cascade baseline 为 0 分基准,计算每个模型的相对改进,实现不同指标任务间的可比较
4. **核心任务子集**: 从大 benchmark 中选取 core tasks 降低评估门槛的方法论
5. **Instruction 格式化经典任务**: 将 SUPERB/MARBLE/HEAR 任务转写为 instruction-following 格式的方法论

---

检索命中: [[SpeechLanguageModel]]✓, [[ProsodyModeling]]✓ | 过滤: [[AudioUnderstanding]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review) | 未命中但可能相关: [[TTSEvaluation]]


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
