---
type: paper
tier: deep
title: "MINT-Bench: A Comprehensive Multilingual Benchmark for Instruction-Following Text-to-Speech"
arxiv_id: "2604.17958"
source: "Sources/MINT-Bench.pdf"
authors: [Huakang Chen, Jingbin Hu, Liumeng Xue, Qirui Zhan, Wenhao Li, Guobin Ma, Hanke Xie, Dake Guo, Linhan Ma, Yuepeng Jiang, Bengu Wu, Pengyuan Xie, Chuan Xie, Qiang Zhang, Lei Xie]
year: 2026
venue: "Preprint"
tags: [TTS, benchmark, evaluation, instruction-following, multilingual, controllability, LALM-as-judge]
concepts: ["[[Instruction-GuidedSpeechSynthesis]]", "[[TTSEvaluation]]", "[[NaturalLanguageDescriptionforTTS]]"]
models: ["[[CosyVoice]]", "[[CosyVoice2]]"]
tasks: ["[[InstructedSpeechGeneration]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeechFactorization]], [[Instruction-GuidedSpeechSynthesis]], [[TTSEvaluation]], [[NaturalLanguageDescriptionforTTS]], [[StyleTransferinTTS]], [[EmotionControlinTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechFactorization]]✓, [[Instruction-GuidedSpeechSynthesis]][待确认], [[TTSEvaluation]][待确认], [[NaturalLanguageDescriptionforTTS]][待确认], [[StyleTransferinTTS]][待确认], [[EmotionControlinTTS]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: Instruction-following TTS 已从固定标签控制演进到自然语言指令控制 (VoxInstruct → CosyVoice 2 → OV-InstructTTS),但评估体系远落后于系统能力。此前唯一专门针对指令遵循的 benchmark 是 InstructTTSEval (Huang et al., 2025),采用三层任务设计 + Gemini-as-Judge 二分评估,但覆盖范围有限且诊断粒度不足。TTSEvaluation 概念页记录了评估方法从 MOS/WER 到 LLM-as-Judge、分布级评估 (TTSDS2)、多维诊断 (TTS-PRISM) 的演进,MINT-Bench 在此谱系中定位为"面向 instruction-following 的结构化多语言 benchmark"。与 InstructTTSEval 的区别: InstructTTSEval 用 True/False 二分判断指令遵循,MINT-Bench 引入三级评分 + 条件感知质量奖励的层级评估协议,并在 10 语言上系统覆盖 timbre/style/composition/extra-vocal 四个控制域。与 NV-Bench 的区别: NV-Bench 专攻非语言发声评估,MINT-Bench 将非语言作为 Special 分支纳入更广泛的 taxonomy。

**已有认知**: NL Description 范式 (PromptTTS 系列, Parler-TTS) 已将控制从离散标签扩展到自然语言描述,但评估仍主要依赖 MOS/WER/SIM 等传统指标。Style Transfer 和 Emotion Control 领域已积累丰富的分解与控制技术,但缺乏针对这些细粒度控制维度的统一评估框架。Speech Factorization 使 10 个原子属性 (timbre 4 + style 6) 的独立控制在技术上成为可能,评估端需要相应的结构化测试。

**创新判断**: MINT-Bench 的核心创新在于将 benchmark 构建形式化为结构化问题(taxonomy → label plan → instruction-text pair 的三阶段 pipeline),而非简单的 prompt 收集。层级评估协议 (content consistency → instruction following → perceptual quality) 比 InstructTTSEval 的二分判断和 NV-Bench 的 PCER 提供更丰富的诊断信息。

## 速查

> [!summary] 速查
> - **一句话**: 首个结构化多语言 instruction-following TTS benchmark,通过分层 taxonomy + 三阶段数据构建 + 层级混合评估协议,在 10 语言上系统评估 TTS 指令遵循能力
> - **路线**: 4 轴 Taxonomy (难度/控制域/控制规格/细粒度模式) → 3 阶段数据构建 (节点规范 → 结构化标签计划 → 指令-文本对) → 3 层评估 (WER 内容一致性 → LALM 指令遵循 → 条件感知质量+音色多样性)
> - **指标**: Gemini 2.5-Flash EN Overall PE 3.66 最高 [Table 3]; Qwen3-TTS ZH PE 3.12 超越所有商用系统 [Table 3]; LALM-human agreement Spearman 67-77 (接近人类间 69-79) [Table 5]; 覆盖 10 语言,大分割 ~1000 对/语言,迷你分割 ~300 对/语言
> - **可借鉴**: (1) 三阶段数据构建 pipeline 防止属性泄露和语义漂移; (2) 内容一致性系数 c 作为后续评分的缩放因子而非过滤门; (3) 条件感知质量评估仅对强指令遵循样本 (s=3) 启用,避免低质量样本干扰排名
> - **局限**: 覆盖范围有限 (不含长文本/对话/code-switching); 多语言不平衡 (中英大分割 vs 其他语言迷你分割); 依赖 Gemini 作为评估 LALM (可能存在偏差和版本依赖)

## 核心问题

1. **为什么需要新的 instruction-following TTS benchmark?** 现有 benchmark 存在三大 gap: (a) 覆盖不结构化,尤其是 timbre 控制、复合指令和 extra-vocal 行为; (b) 诊断粒度有限,无法区分系统失败是因为内容错误、指令遵循弱还是感知质量差; (c) 多语言扩展性不足 [§1]

2. **如何将 benchmark 构建从随意的 prompt 收集提升为系统化工程?** 通过分层 taxonomy 定义覆盖空间 + 多阶段 pipeline 控制数据生成质量 [§3.2, §3.3]

3. **当前系统在 instruction-following TTS 上的真实能力边界在哪里?** 实验揭示: Easy 控制已相对可靠,但 Hard (compositional/dynamic/layered/conflict) 和 Special (disfluency/dysphonia/nonverbal) 仍是主要瓶颈;商用系统在英语上领先,但开源系统在中文等局部场景已具竞争力 [§4.2, §4.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MINT-Bench 由三个紧耦合组件构成 [§3.1, Fig 1]:
1. **分层多轴 Taxonomy**: 组织 benchmark 的覆盖空间
2. **三阶段数据构建 Pipeline**: 将 taxonomy 节点实例化为自然语言 benchmark 条目
3. **层级混合评估协议**: 从内容一致性、指令遵循、感知质量三个维度逐层评估

### 关键设计选择

#### 1. 分层多轴 Taxonomy [§3.2]

Taxonomy 的核心是 10 个原子声学属性 (Table 1):
- **Timbre (4)**: Age (3 类), Gender (2 类), Pitch (5 级), Texture (人工标注)
- **Style (6)**: Speed (5 级), Volume (5 级), Emotion (7+1 类), Accent (语言相关), Personality (10 类), Tone (17 类)

这 10 个属性沿四个轴组织 [论文原文]:
- **Axis 1 - 难度**: Easy (原子/简单组合) → Hard (复杂组合/动态/分层/冲突/人物/场景) → Special (extra-vocal)
- **Axis 2 - 控制域**: Timbre / Style / Composition / Extra-vocal
- **Axis 3 - 控制规格**: Tag / Direct / Simple-Complex / Implicit-Explicit
- **Axis 4 - 细粒度模式**: Static / Dynamic / Layered / Conflict / Scenario / Character

**为什么用分层而非扁平 taxonomy?** [论文原文] 作者认为扁平 prompt 集无法系统暴露系统在不同控制维度上的失败模式,分层结构使诊断信息更有意义 [§3.2]。[agent 解读] 这也使 benchmark 天然具备可扩展性——新增控制维度只需在 taxonomy 中添加节点,不需要重新设计整个 benchmark。

**Taxonomy 节点**: 四轴空间中的一个有效坐标,定义了一个结构化的控制用例。每个节点指定了难度、域、规格和模式,但最终的语言表面形式是开放的 [§3.2]。

#### 2. 三阶段数据构建 Pipeline [§3.3]

**为什么不直接一步生成 prompt?** [论文原文] 一步到位的生成容易出现属性遗漏、语义漂移、目标线索泄露到合成文本中、以及指令与控制意图不一致等问题 [§3.3]。

三阶段分离了 benchmark 设计与语言实现:

**Stage 1 - 节点规范**: 为每个 taxonomy 节点指定内部属性库、目标语言和条目预算。Large split ~890 对/语言,Mini split ~274 对/语言 [Table 7, 8]。

**Stage 2 - 结构化标签计划**: 将节点规范转化为中间结构化计划——指定目标值、语义设置、文本长度和实现约束。[论文原文] 这个中间层对质量控制至关重要: 减少不受控的生成方差,防止目标线索泄露到合成文本中,确保最终实例忠于预期控制用例 [§3.3]。

**Stage 3 - 指令-文本对构建**: 将结构化计划实现为自然语言指令 + 合成文本对。合成文本在约束下生成,最小化控制目标的直接词汇泄露 [§3.3]。

**数据构建 LLM**: 使用 Gemini 2.5 Pro [§4.1]。[agent 解读] 这意味着 benchmark 的质量上界受限于 LLM 的指令理解能力,但三阶段 pipeline 通过中间约束层缓解了这个问题。

#### 3. 层级混合评估协议 [§3.4]

评估从三个互补视角逐层进行 [Fig 2]:

**Stage 1 - 内容一致性**: ASR 转写 + 逐样本 WER 检查。为每种语言计算裁剪均值 WER 阈值 tau_l [Table 9]。内容一致性系数 c = m_low / N_cc (低 WER 样本比例) [Eq. 1]。

**关键设计**: 内容一致性检查不丢弃样本,而是通过系数 c 缩放后续分数 [论文原文]。这样频繁出现内容偏差的系统即使原始指令遵循分数高也会被惩罚 [§3.4]。

**不适用范围**: WER 检查不应用于口音相关子集 (ASR 对口音变化不稳定) 和 Special 子集 (extra-vocal 现象会扭曲 ASR 输出) [§3.4]。

**Stage 2 - 指令遵循**: LALM (Gemini 3.1 Pro Preview) 使用任务感知评估 prompt,对每个样本赋 1/2/3 分 (poor/partial/strong) [§3.4]。使用三组 prompt family (Easy/Hard/Special) 匹配不同难度的诊断标准 [Fig 7-9]。

IF Score = c * (1/N * sum(s_i)) [Eq. 2]

**Stage 3 - 感知质量与音色多样性**: 仅对 s_i = 3 的样本评估两个二元奖励: 自然度 B_nat 和表达力 B_exp [§3.4]。

PE Score = IF Score + c * (1/N * sum(B_nat + B_exp)) [Eq. 3]

**为什么只对 s_i=3 的样本评估感知质量?** [论文原文] 这确保 Stage 3 仅细化高质量表现的上限,不改变明显失败或部分实现样本的排名结构 [§3.4]。

**音色多样性 (TDS)**: 对选定子集,同一指令条件下生成多个候选,计算通过指令遵循检查的候选间平均配对说话人相似度 (APS),TDS = (1-APS) * m/N [Eq. 4, 5]。使用 WavLM-Large 提取说话人嵌入 [§4.1]。

### 训练策略

不适用——MINT-Bench 是评估 benchmark,不涉及模型训练。

## 实验

### 中英双语详细结果 [Table 3]

| 指标 | 系统 | 英语 | 中文 | 出处 |
| --- | --- | --- | --- | --- |
| Overall PE Score | Gemini 2.5-Flash | 3.66 | 2.95 | [Table 3] |
| Overall PE Score | Gemini 2.5-Pro | 3.45 | 2.93 | [Table 3] |
| Overall PE Score | ElevenLabs-ttv-v3 | 3.12 | 2.61 | [Table 3] |
| Overall PE Score | Qwen3TTS-12Hz-1.7B-VD | 2.94 | 3.12 | [Table 3] |
| Overall PE Score | GPT-4o-Mini-TTS | 2.15 | 1.82 | [Table 3] |
| WER | Gemini 2.5-Flash | 1.4% | 1.6% | [Table 3] |
| WER | Qwen3TTS-12Hz-1.7B-VD | 1.5% | 1.1% | [Table 3] |
| WER | Parler-TTS Mini | 17.5% | - | [Table 3] |
| TDS | Gemini 2.5-Pro | 0.57 | 0.39 | [Table 3] |
| TDS | MiMo-Audio | 0.52 | 0.37 | [Table 3] |

### 多语言结果 [Table 4]

| 指标 | 系统 | Overall IF/PE | 出处 |
| --- | --- | --- | --- |
| Overall | Gemini 2.5-Flash | 2.50/3.75 | [Table 4] |
| Overall | Gemini 2.5-Pro | 2.48/3.63 | [Table 4] |
| Overall | Qwen3TTS-12Hz-1.7B-VD | 2.29/3.19 | [Table 4] |
| Overall | Ming-omni-tts-16.8B | 1.13/1.25 | [Table 4] |

### 人机一致性 [Table 5]

| 指标 | 中文 | 英语 | 出处 |
| --- | --- | --- | --- |
| Inter-Human Agreement | 69.45 +/- 0.79 | 77.32 +/- 0.72 | [Table 5] |
| Model-Consensus Human | 67.12 +/- 0.93 | 74.81 +/- 0.86 | [Table 5] |

### 关键发现

1. **商用前沿 vs 开源竞争力**: 英语上商用系统 (Gemini 系列) 明显领先,但中文上 Qwen3-TTS (PE 3.12) 超越所有商用系统 (Gemini Flash 2.95, Pro 2.93) [Table 3, §4.2]

2. **WER 不等于指令遵循能力**: 相似的 WER 不一定转化为相似的 IF 或 PE 分数,内容一致性不足以表征指令遵循 TTS 能力 [§4.2]

3. **难度梯度效应**: Easy → Hard 有明显性能下降,Dynamic/Layered/Conflict/Scenario/Character 子集仍是重大挑战;Special 分支最难,其中 Disfluency 控制一致性最低 [§4.2]

4. **TDS 独立于主排名**: MiMo-Audio 尽管 IF/PE 分数较弱,TDS 却较高 (EN 0.52),证实音色多样性是独立的条件诊断指标 [§4.2]

5. **LALM 评估可靠性**: 模型-人类共识与人类间一致度的差距在所有语言上保持很小且稳定 (gap 2-3 points) [Table 5]

## 局限性

1. **覆盖范围有界**: 不包含长文本生成、对话交互、更丰富的篇章上下文和 code-switching 等开放场景 [§6]
2. **多语言不平衡**: 中英用大分割 (~1000 对),其他 8 语言用迷你分割 (~300 对),跨语言可比性受限 [§6]
3. **评估依赖 LALM**: 依赖 Gemini 作为 judge 可能引入偏差、不稳定性和版本依赖;商用 API 也带来成本和可复现性问题 [§6]
4. **无端到端人类评估**: 人机一致性验证仅在部分样本上进行,且仅针对指令遵循维度 [§4.4]
5. [agent 解读] **自我引用偏差风险**: 使用 Gemini 2.5 Pro 构建数据、使用 Gemini 3.1 Pro Preview 评估,而被评系统中包含 Gemini TTS——尽管构建和评估使用不同版本,但潜在的 Gemini-family 偏差未被讨论
6. [agent 解读] **Taxonomy 覆盖上界**: 10 个原子属性主要面向单轮控制,不涵盖跨轮一致性 (如 MCLP 关注的 stylistic consistency) 或长程韵律规划

## 点评

MINT-Bench 的核心贡献在于**方法论层面**而非工具层面: 它将 instruction-following TTS 的评估从"收集 prompt + 打分"提升为一个结构化的 benchmark 构建问题。三阶段数据构建 pipeline (taxonomy node → structured label plan → instruction-text pair) 有效解决了直接生成 prompt 时的属性泄露和语义漂移问题。层级评估协议将内容一致性、指令遵循和感知质量解耦为独立层,比 InstructTTSEval 的 True/False 二分判断和 NV-Bench 的单维 CER 提供了更丰富的诊断信息。

实验结果具有重要的生态观察价值: 中文场景下 Qwen3-TTS 已超越所有商用系统,暗示 instruction-following TTS 的竞争格局正在按语言/区域分化。更有诊断价值的是 Easy → Hard → Special 的难度梯度分析,清楚地揭示了 compositional control 和 extra-vocal behavior 是当前系统的主要瓶颈。

但也应注意其局限: (1) Gemini 同时参与数据构建和评估,存在潜在的 family-level 偏差; (2) 评估仅覆盖单轮合成,不涉及对话场景; (3) 多语言扩展采用迷你分割,跨语言结论的稳健性有待加强。

## 可复用的 idea

1. **三阶段数据构建 pipeline**: 将 benchmark 构建分为 taxonomy → structured plan → natural language realization 三步,中间层作为质量控制的约束边界,防止属性泄露。可迁移到任何需要结构化覆盖的评估任务。

2. **内容一致性系数作为缩放因子**: 用 WER 阈值检查生成 c 值,作为后续评分的乘性惩罚而非二元过滤。保留了所有样本用于分析,同时对内容不一致的系统施加惩罚。

3. **条件感知质量评估**: 仅对强指令遵循样本 (s=3) 评估感知奖励 (自然度+表达力),避免低质量样本干扰排名上端的分辨力。这种"先通过门槛再评质量"的设计可用于任何分层评估场景。

4. **TDS (音色多样性) 指标**: 通过 APS × 通过率的组合衡量一对多生成能力,作为独立于 IF/PE 的条件诊断指标。
