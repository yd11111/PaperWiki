---
type: paper
tier: deep
title: "InstructTTSEval: Benchmarking Complex Natural-Language Instruction Following in Text-to-Speech Systems"
arxiv_id: "2506.16381"
source: "Sources/InstructTTSEval.pdf"
authors: [Kexin Huang, Qian Tu, Liwei Fan, Chenchen Yang, Dong Zhang, Shimin Li, Zhaoye Fei, Qinyuan Cheng, Xipeng Qiu]
year: 2025
venue: "arXiv preprint"
tags: [TTS, evaluation, benchmark, instruction-following, controllability, style-control, paralinguistic, LLM-as-judge]
concepts: ["[[TTSEvaluation]]", "[[NaturalLanguageDescriptionforTTS]]", "[[Instruction-GuidedSpeechSynthesis]]", "[[ProsodyModeling]]", "[[EmotionControlinTTS]]", "[[StyleTransferinTTS]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]", "[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[InstructedSpeechGeneration]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文位于 TTS Evaluation 的最新演进节点 — 专门面向 instruction-following 能力的评估。在已有评估工作线中,TTS 评估经历了 WER/SIM 客观指标 → MOS 主观评估 → TTSDS 分布级评估 → LLM-as-Judge → Responsible Evaluation 框架 → SpeechJudge 自然度 GRM → TTS-PRISM 多维诊断的演进。InstructTTSEval 填补的是"指令遵循度"这一维度 — 之前的评估体系几乎不涉及。
>
> **已有认知**: [[Instruction-GuidedSpeechSynthesis]] [待确认] 概念页已记录 VoxInstruct、CosyVoice 等系统的指令控制方法,并指出"评估: 传统指标难以衡量指令遵循度"是关键挑战。[[NaturalLanguageDescriptionforTTS]] [待确认] 记录了从 PromptTTS 到 Parler-TTS 的描述控制演进,指出"描述与语音的匹配度难以自动量化"的技术挑战。[[TTSEvaluation]] [待确认] 详细记录了 WER/SIM/MOS 等指标的局限,以及 LLM-as-Judge 等新兴方法。[[InstructedSpeechGeneration]] (confirmed) 明确列出"缺乏标准化的 style controllability benchmark"为开放问题。
>
> **创新判断**: 相比已有 style description datasets (TextrolSpeech 5 标签, SpeechCraft 8 标签, ParaSpeechCraft 11 标签) 和 general TTS evaluation benchmarks,InstructTTSEval 首次构建了分层级 (APS/DSD/RP) 的指令遵循评估框架,从细粒度声学参数到抽象角色扮演,覆盖 12 个副语言特征,且使用 free-form 而非 fixed-tag 标注。Gemini-as-a-Judge 方案也是首次在指令遵循 TTS 评估中系统验证。
>
> 检索命中: [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓ | 过滤: [[TTSEvaluation]](pending-review), [[NaturalLanguageDescriptionforTTS]](pending-review), [[Instruction-GuidedSpeechSynthesis]](pending-review), [[EmotionControlinTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个分层级 (APS/DSD/RP) 的指令遵循 TTS benchmark,覆盖 12 个副语言特征 x 3 抽象层级 x 2 语言 = 6K 测试用例,用 Gemini-as-Judge 自动评估
> - **路线**: 影视数据 → 清洗/过滤(DNSMOS+DVA) → Gemini 生成 speech caption (APS) → GPT-4o 生成 DSD/RP 指令 → TTS 合成 → Gemini-as-Judge 评估 True/False
> - **指标**: 商用最佳 gemini-flash EN-Avg 88.7% (超过 reference_audio 84.3%,有 self-preference bias 嫌疑), 开源最佳 VoxInstruct EN-Avg 50.4%; 人机一致率整体 79.0% (APS 87%, DSD 79%, RP 71%) [Table 4-6]
> - **可借鉴**: (1) 从影视数据 bottom-up 构造 benchmark 的思路 — 先有表现力强的音频,再反向生成指令; (2) 12 特征 x 3 抽象层级的分层设计思想可迁移到其他可控生成评估; (3) Gemini-as-Judge 在 TTS 指令评估中 79% 一致率,成本约 $12/session
> - **局限**: (1) RP 任务主观性强,人机一致率仅 66-76%; (2) Gemini 做 judge 可能偏好自己的输出; (3) 仅 True/False 二分评估,无连续分数; (4) 数据源集中在影视,口语/对话场景不足; (5) 中文 subset 仅评测了 3 个商用 + 1 个开源系统

## 核心问题

现有 TTS 系统虽然越来越多地支持自然语言指令控制语音风格,但**缺乏标准化 benchmark 来衡量这种指令遵循能力**。传统评估指标 (WER、SIM、MOS) 仅关注可懂度/音质,无法评估"指令要求快乐语气,合成结果是否真的快乐"。少量已有 style description datasets 使用 fixed tags 标注 (5-11 个标签),无法捕捉 free-form 自然语言指令的多样性和层级性 [§1, §2.2]。

本文试图回答: 当前 TTS 系统到底能在多大程度上理解并执行复杂的自然语言风格指令?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

InstructTTSEval 不是一个 TTS 模型,而是一个评估 benchmark。它的"方法"是 benchmark 构建流程 + 评估协议:

```
影视原始数据 → 清洗/过滤 → 2000 段高表现力参考音频
   → Gemini 语音理解 → 12 维 speech caption (= APS 指令)
   → GPT-4o → DSD 指令 (随机 dropout 部分特征)
   → GPT-4o + CoT → RP 指令 (场景/角色推理)
   → 各 TTS 系统合成 → Gemini-as-Judge → True/False 判定
```

### 关键设计选择

**1. 三层任务分层 (APS → DSD → RP)**

论文将指令遵循的复杂度分为三个递进层级 [§3.1]:

- **APS (Acoustic-Parameter Specification)**: 直接指定 12 个声学参数的 free-form 描述。测试模型能否精确映射每个描述到声学实现。这是最"机械"的层级 [论文原文]。
- **DSD (Descriptive-Style Directive)**: 将 APS 的结构化指令用 LLM 重写为自然语言段落,并随机丢弃部分特征。测试模型对非结构化输入的泛化能力 [论文原文]。
- **RP (Role-Play)**: 仅给出角色/场景描述 (如"一个紧张的面试者"),模型需要用自身的世界知识推理出合适的声学表达。测试高阶推理能力 [论文原文]。

**为什么这样分层?** 论文认为从具体到抽象的渐进设计可以诊断 TTS 系统的控制瓶颈: APS 考查声学映射能力,DSD 考查语义理解能力,RP 考查推理能力 [§3.1] [论文原文]。[agent 解读] 这种分层思路类似于 NLP 中从 extractive QA 到 open-domain QA 到 reasoning 的难度递进,可以定位模型到底在哪个层级掉链子。

**2. 12 个副语言特征维度**

论文整合了 4 个层次的 12 个特征 [§3.1, Fig 3]:
- **生理层**: gender, pitch, texture (音色)
- **语言层**: clarity (清晰度), fluency (流畅度), speed
- **社会层**: accent (口音), age, volume
- **心理/语用层**: emotion, tone (语调), personality

**为什么选 12 个?** 论文参考了 Cutler et al. (1997)、Diwan et al. (2025)、Jin et al. (2024) 等先验工作 [§3.1] [论文原文]。[agent 解读] 相比 TextrolSpeech 的 5 个特征和 ParaSpeechCraft 的 11 个标签,12 个特征覆盖面更广,且新增了 texture、personality、tone 等高阶属性,这些正是 free-form 描述区别于 fixed-tag 的关键维度。

**3. Bottom-up 数据构造**

论文没有从随机 TTS 输出开始,而是从影视 (电影/电视剧/综艺) 中挖掘高表现力音频片段,再反向生成指令 [§3.2] [论文原文]:

- **数据源**: NCSSD 数据集 + 自行收集的影视音频
- **清洗**: speaker diarization (pyannote) + Whisper ASR + 标点恢复 → ~6000 小时
- **过滤**: DNSMOS ≥ 2.8 (音质) + WhisperD (单说话人) + DVA toolkit Dominance/Arousal ≥ 0.8 (表现力) + 时长 >3s + 词数 >10
- **最终**: 1000 EN + 1000 ZH = 2000 段参考音频 [Table 2]

**为什么 bottom-up?** [论文原文] 论文认为这种方式能确保生成的指令基于真实高表现力语音,而非凭空构造不切实际的风格描述。[agent 解读] 这样做的优势是 reference audio 确实存在且表现力强,使评估有 ground truth 对齐的锚点;劣势是数据源偏向影视表演风格,可能不反映日常对话场景。

**4. Gemini-as-Judge 评估**

论文选择 Gemini (gemini-2.5-pro-preview-05-06) 作为自动评估器,对合成语音做 True/False 二分判定 [§3.3, §4.1]:

- 评估 prompt: 给 Gemini 12 个维度的定义 + 判断标准 (True=主要风格属性一致, False=至少一个关键属性冲突) [Fig 8]
- 人机一致性验证: 50 样本 x 3 任务 x 2 语言,3 名标注员多数投票 vs Gemini,整体一致率 79% [Table 4]

**为什么用 Gemini 而不是其他方法?** [论文原文] 论文认为 Gemini 具有强大的语音理解能力,且可实现快速、可扩展的自动评估。[agent 解读] 但论文也承认 LLM 可能偏好自己的输出 (self-preference bias, Panickssery et al. 2024),这使 Gemini 系列 TTS 在 benchmark 上的高分需打折扣。

### 训练策略

不适用 — 本文是 benchmark 论文,不涉及模型训练。

## 实验

### 被评测系统

**闭源系统** [§4.2]:
- gemini-2.5-flash-preview-tts (gemini-flash)
- gemini-2.5-pro-preview-tts (gemini-pro)
- gpt-4o-mini-tts
- Hume

**开源系统**:
- VoxInstruct, Parler-TTS-mini, Parler-TTS-large, PromptTTS, PromptStyle

### 主要结果

| 指标 | 系统 | APS | DSD | RP | Avg. | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| True rate (%) | reference_audio | 96.2 | 89.4 | 67.2 | 84.3 | EN-subset | [Table 5] |
| True rate (%) | gemini-flash* | 92.3 | 93.8 | 80.1 | 88.7 | EN-subset | [Table 5] |
| True rate (%) | gemini-pro* | 87.6 | 86.0 | 67.2 | 80.3 | EN-subset | [Table 5] |
| True rate (%) | gpt-4o-mini-tts | 76.4 | 74.3 | 54.8 | 68.5 | EN-subset | [Table 5] |
| True rate (%) | hume* | 83.0 | 75.3 | 54.3 | 71.1 | EN-subset | [Table 5] |
| True rate (%) | VoxInstruct | 54.9 | 57.0 | 39.3 | 50.4 | EN-subset | [Table 5] |
| True rate (%) | Parler-TTS-mini | 63.4 | 48.7 | 28.6 | 46.9 | EN-subset | [Table 5] |
| True rate (%) | Parler-TTS-large | 60.0 | 45.9 | 31.2 | 45.7 | EN-subset | [Table 5] |
| True rate (%) | PromptTTS | 64.3 | 47.2 | 31.4 | 47.6 | EN-subset | [Table 5] |
| True rate (%) | PromptStyle | 57.4 | 46.4 | 30.9 | 38.2 | EN-subset | [Table 5] |
| True rate (%) | reference_audio | 90.9 | 86.7 | 69.8 | 82.5 | ZH-subset | [Table 6] |
| True rate (%) | gemini-flash* | 88.2 | 90.9 | 77.3 | 85.4 | ZH-subset | [Table 6] |
| True rate (%) | gemini-pro* | 89.0 | 90.1 | 75.5 | 84.8 | ZH-subset | [Table 6] |
| True rate (%) | gpt-4o-mini-tts | 54.9 | 52.3 | 46.0 | 51.1 | ZH-subset | [Table 6] |
| True rate (%) | VoxInstruct | 47.5 | 52.3 | 42.6 | 47.5 | ZH-subset | [Table 6] |
| 一致率 (%) | Human-Gemini (EN) | 86 | 78 | 66 | 76.7 | EN-subset | [Table 4] |
| 一致率 (%) | Human-Gemini (ZH) | 88 | 80 | 76 | 81.3 | ZH-subset | [Table 4] |

### 关键发现

1. **闭源远超开源**: 商用模型在所有任务上显著优于开源。gemini-flash EN-Avg 88.7% vs 开源最佳 VoxInstruct 50.4%,差距 38 个百分点 [Table 5] [论文原文]。

2. **Gemini 超过 reference audio**: gemini-flash EN-Avg 88.7% 超过 reference_audio 的 84.3% [Table 5]。论文承认 Gemini-as-Judge 可能偏好 Gemini TTS 的输出 (self-preference bias, Panickssery et al. 2024) [§4.3] [论文原文]。这一结果反而成为 self-preference bias 的有力证据 [agent 解读]。

3. **RP 任务难度最高但闭源仍有可观表现**: gemini-flash RP 达 80.1%,但开源最佳 VoxInstruct 仅 39.3%。参考音频在 RP 上也仅 67.2% [Table 5],说明 bottom-up 生成的 RP 指令与原始音频之间本身存在偏差 [agent 解读]。

4. **VoxInstruct 在开源中最佳但有短板**: VoxInstruct EN-Avg 50.4%,在 DSD (57.0%) 和 RP (39.3%) 上领先其他开源系统,但 APS (54.9%) 上较弱,可能因为处理长输入能力有限 [§4.3] [论文原文]。

5. **开源系统在特定维度有优势**: 部分开源系统 (如 Parler-TTS-Large) 在音色模拟 (如老年声音、儿童声音) 上表现出色,说明音色灵活性和情感表现力是正交能力 [§4.4] [论文原文]。

6. **中文表现模式不同**: ZH-subset 中 gpt-4o-mini-tts Avg 51.1% 远低于其 EN-subset 68.5%,可能源于商用模型主要以英语数据训练 [§4.3] [论文原文]。VoxInstruct 的中文合成虽然指令遵循弱 (47.5%),但自然度更像母语者 [论文原文]。

### Case Study 发现 [§4.4, Table 7]

- **副语言事件 (叹气、笑、尖叫)**: 仅 gpt-4o-mini-tts、gemini-flash、gemini-pro 能生成笑声,没有模型能成功合成叹气 [论文原文]。
- **极端情感 + 快速转变**: gpt-4o-mini-tts 能产生喊叫效果;gemini 和 VoxInstruct 对声音升高有初步能力,但情感转变 (如从平静到冲动) 仍然困难 [论文原文]。
- **唱歌**: 仅 gemini-flash/pro 展现初步歌唱能力。指令 TTS 产生歌唱效果需要韵律/旋律/情感/音色/节奏的多维协调,是极高要求 [论文原文]。

## 局限性

1. **RP 任务主观性**: RP 指令的主观性导致人类标注者之间一致率较低 (66-76%),使自动评估噪声较大 [§Limitations] [论文原文]。

2. **评估成本**: 使用 Gemini-as-Judge 每个语言子集约 $12/session,大规模持续评估成本高 [§Limitations] [论文原文]。

3. **数据不平衡**: bottom-up 构建导致某些情感类别/角色类型代表不足 [§Limitations] [论文原文]。

4. **二分评估粒度粗**: 仅 True/False 判定,无法区分"略微不匹配"和"完全偏离" [agent 解读]。

5. **Self-preference bias 未控制**: 论文虽然提及但未用第三方 judge 做对照实验 [agent 解读]。

6. **中文评测覆盖不足**: ZH-subset 仅评测了 3 个闭源 + 1 个开源系统,无法全面反映中文 TTS 生态 [agent 解读]。

7. **数据源偏影视**: 影视表演风格与日常对话场景差异大,benchmark 泛化性存疑 [agent 解读]。

## 点评

InstructTTSEval 的核心价值在于**首次系统化地拆解了 TTS 指令遵循的层次结构** — 从参数级映射 (APS) 到语义理解 (DSD) 到推理能力 (RP)。这种分层设计比之前一刀切的 style similarity 评估更有诊断力,能清楚地看到系统在哪个层级掉链子。

但作为 benchmark 论文,有几个设计选择值得讨论:

1. **Self-preference bias 是最大隐患**: gemini-flash EN-Avg 88.7% 超过 reference_audio 84.3%,这在合理世界里不应发生 — 一个 TTS 系统的输出不应比真人录音更"符合指令"。这强烈暗示 Gemini-as-Judge 对 Gemini TTS 的输出有系统性偏好。论文虽承认但未做任何缓解 (如引入 GPT-4o 或 Whisper-based classifier 做第二 judge 对照)。

2. **True/False 粒度**: 对于 12 维特征的指令,仅给出一个二分判定,丧失了大量诊断信息。比如一个系统在 11 维匹配、1 维不匹配,和完全不匹配的系统得到同样的 False,这降低了 benchmark 的区分度。

3. **参考音频 RP 仅 67%**: 这说明 RP 指令本身与音频的对齐就有问题,使得 RP 子任务的评估可靠性存疑。这可能是 bottom-up 构造的固有局限 — 从音频反推的角色描述未必能忠实还原原始表演。

4. **开源系统选择偏旧**: PromptTTS (2022) 和 PromptStyle (2023) 已较老旧,未纳入近期开源系统如 Llasa、F5-TTS、Spark-TTS 等,可能低估了开源阵营的当前水平。

总体而言,这是一个填补重要空白的工作 — "指令遵循度"这个维度确实被之前的评估体系忽略了。分层任务设计和 case study 的发现 (副语言事件、情感转变、唱歌) 对 controllable TTS 研究有实际指导价值。

## 可复用的 idea

1. **分层任务设计 (参数→描述→推理)**: 评估可控生成系统时,可按抽象程度分层诊断,定位控制瓶颈。可迁移到 controllable image/music generation 评估。

2. **Bottom-up benchmark 构造**: 先收集高质量目标样本,再反向生成控制指令,确保 benchmark 有 grounded reference。比 top-down 构造 (先想指令再生成) 更现实。

3. **特征 dropout 的 DSD 任务**: 对结构化描述随机丢弃部分特征,测试系统对不完整输入的鲁棒性。这个思路可用于测试任何 conditional generation 系统。

4. **DVA toolkit 过滤表现力**: 用 Dominance + Arousal 两个维度筛选高表现力音频段,阈值 0.8,是高效的数据筛选方法。

5. **12 维副语言特征体系**: 生理/语言/社会/心理四层 12 个特征的分类框架,可作为 style annotation schema 复用。

> [!review] 审阅 (2026-06-03, agent-auto)
> **结论**: pass-with-fixes
> - (high, fixed) 初始版本 PDF 提取列对齐错误导致 Table 5/6 所有数值错位,已通过 PDF 原文图像验证修正
> - (medium, fixed) 部分发现缺少具体 [Table N] 标注,已补充
> - (low, accepted) frontmatter models 列 related work 模型而非评测目标,benchmark 论文可接受
> 详见 `_review/InstructTTSEval-review.yml`
