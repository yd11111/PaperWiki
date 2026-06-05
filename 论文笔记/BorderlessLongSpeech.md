---
type: paper
tier: deep
title: "Any2Speech: Borderless Long Speech Synthesis"
arxiv_id: "2603.19798"
source: "Sources/BorderlessLongSpeech.pdf"
authors: [Xingchen Song, Di Wu, Dinghao Zhou, Pengyu Cheng, Hongwu Ding, Yunchao He, Jie Wang, Shengfan Shen, Sixiang Lv, Lichun Fan, Hang Su, Yifeng Wang, Shuai Wang, Meng Meng, Jian Luan]
year: 2026
venue: "arXiv"
tags: [TTS, long-form, agentic, data-strategy, hierarchical-annotation, chain-of-thought, instruction-following, multi-speaker, expressiveness, continuous-tokenizer]
concepts: ["[[LLM-basedTTS]]", "[[Instruction-GuidedSpeechSynthesis]]", "[[EmotionControlinTTS]]", "[[ProsodyModeling]]", "[[NaturalLanguageDescriptionforTTS]]"]
models: ["[[CosyVoice]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文自我定位为 "Generation 4: Native Agentic TTS",延续 LLM-based TTS (Gen 3) 的路线但提出两个根本性跃迁: (1) 从句级控制到长上下文全局连贯, (2) 从声音生成到完整声学场景建模。在 [[LLM-basedTTS]] 的演进谱系中,VALL-E 开创了 codec LM TTS (2023),CosyVoice 引入 hybrid LLM+Flow 架构 (2024),而本文不属于具体模型架构的创新,而是从数据标注-训练策略-系统架构三个维度提出一套完整的框架方法论。

**已有认知对比**:
- [[Instruction-GuidedSpeechSynthesis]] [待确认] 的演进线到 OV-InstructTTS 已出现 "理解-推理-合成" 的 reasoning chain 范式,与本文的 CoT 策略高度契合,但本文将其扩展到多维度多层级的场景。
- [[EmotionControlinTTS]] [待确认] 已有大量句级/词级/帧级情感控制方法,但几乎全部是 utterance-level 或 intra-utterance 粒度,缺乏跨句情感弧(emotional arc)的建模 -- 这正是本文声称的 Gap 1。
- [[ProsodyModeling]] (confirmed) 从 GST 到 in-context learning 的演进主要在隐式建模,本文的 CoT 将韵律决策显式化并可编辑,是一种反向设计。
- [[NaturalLanguageDescriptionforTTS]] [待确认] 从属性级文本描述发展到指令级控制,本文更进一步将自然语言描述扩展到多层级(场景-句子-音素)的结构化 caption。

**创新判断**: 本文的核心创新不在模型架构本身(backbone 描述简略,仅提 "continuous tokenizer"),而在三个层面: (1) 数据哲学 -- "label don't filter" 颠覆了主流清洗流水线, (2) 分层标注schema -- Global-Sentence-Token 将控制带宽从窄带(纯文本)扩展到宽带(结构化语义), (3) 系统观 -- 将 TTS 引擎定位为 Agent 架构中的执行层而非独立系统。

> 检索命中: [[LLM-basedTTS]]✓, [[ProsodyModeling]]✓ | 过滤: [[Instruction-GuidedSpeechSynthesis]](pending-review), [[EmotionControlinTTS]](pending-review), [[NaturalLanguageDescriptionforTTS]](pending-review), [[GlobalStyleTokens]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 Generation 4 Native Agentic TTS 框架,通过 "label don't filter" 数据策略 + Global-Sentence-Token 分层标注 + CoT 推理 + Dimension Dropout,实现跨句情感弧连贯的长语音场景合成
> - **路线**: 任意输入 → LLM Agent(结构化为 GST schema)→ Instruct+Think 分流 → CoT 推理(Global→Sentence→Token 逐级规划) → 连续 tokenizer backbone → 长音频
> - **指标**: 无定量评估结果 -- 作者明确指出 Gen 4 TTS 的评估框架是开放问题 [§5]
> - **可借鉴**: (1) "label don't filter" 数据哲学可用于任何含噪语音数据场景; (2) Global-Sentence-Token 分层控制协议可作为 TTS agent 系统的接口标准; (3) Dimension Dropout 的部分条件训练思路可迁移到其他条件生成任务
> - **局限**: 无开源; 无定量实验/对比; 仅面向离线内容创作,未覆盖实时交互; 模型架构细节缺失(backbone 仅一句话带过)

## 核心问题

本文要解决的核心问题是: **现有 TTS 系统(包括 Gen 3 LLM-based TTS)在两个维度上的根本不足** [§1]:

1. **Gap 1: 句级控制 ≠ 全局连贯** -- Instruct TTS 可以让单句听起来"紧张",但无法知道在上下文中应该多紧张。跨句的情感弧(如叙事中从压抑到爆发的渐进)无法通过逐句合成+拼接实现 [§1]。
2. **Gap 2: 声音控制 ≠ 场景完整** -- 现有系统只建模人声,无法建模咖啡馆背景嘈杂、剧场混响、体育场人群等声学场景。根因是纯文本对话表示(`<speaker1>text<speaker2>text`)剥离了场景上下文、说话人画像、副语言线索、声学环境和交互动态 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文提出的是一套框架而非单一模型,由三个支柱组成 [§6]:

```
支柱 1: 数据策略 — "Label, don't filter" + Global-Sentence-Token 分层标注
支柱 2: 模型策略 — CoT 推理 + Dimension Dropout (基于 continuous tokenizer backbone)
支柱 3: 系统架构 — Native Agentic: GST schema 作为 LLM Agent ↔ 合成引擎的结构化语义接口
```

**架构流程** [agent 解读,综合 §2-4]:
```
任意输入(文本/视频/...)
    ↓
前端 LLM Agent: 理解输入,生成 Global-Sentence-Token 结构化命令
    ↓
Instruct 流 (用户硬约束): 场景元数据 / 说话人画像 / 声学环境
Think 流 (模型推理计划): 全局氛围 / 逐句情感-语气-语速-意图 / 音素级发音
    ↓
连续 tokenizer backbone: 按 Think 推理链合成音频
    ↓
长音频输出(多说话人 / 情感弧 / 声学场景)
```

### 关键设计选择

#### 1. "Label, Don't Filter" 数据策略 [§2.1]

[论文原文] 标准数据流水线追求纯净: 基于 DNSMOS 分数、WER 阈值或单说话人检查激进丢弃片段。如 TouchTTS 所记录,这种激进过滤通常只保留原始语料的 10-30% [§2.1]。更严重的是,最具表现力的数据与"噪声"一同消失 -- 激烈争论、情感化多人对话、重叠对话恰恰是最丰富的声音表现力来源。

**具体做法**:
- **Label, don't filter**: 保留重叠、打断、感叹、语音+音乐+背景事件的复杂层叠,用细粒度分层标注表征声学场景,而非丢弃 [§2.1]
- **Label, don't clean**: 不做去噪,将背景声、环境噪声和人声作为统一整体建模 [§2.1]
- 数据利用率从传统的 10-30% 提升到 90% 以上 [§2.1]

[论文原文] 对"脏数据是否牺牲可控性"的回应: 当标注足够详细时,模型学习到背景条件与文本描述之间的显式关联,推理时可通过描述控制声学环境 -- "quiet background" 输出干净语音,"children crying in the background" 生成家庭场景 [§2.1]。

[论文原文] 额外发现: 仅在干净简单数据上训练的模型容易在推理时产生幻觉;暴露在从干净到噪声、从简单到复杂的完整频谱下,生成质量明显提升 [§2.1]。

#### 2. Global-Sentence-Token 分层标注 [§2.2]

三层自顶向下标注结构:

| 层级 | 内容 | 类比(专业配音) |
|------|------|----------------|
| **Global** | 节目格式、风格标签、说话人画像(性别/年龄/声音人格)、整体情感基调与轨迹、声学环境、声音事件 | 录制前的导演指令: 角色设定 + 场景背景 + 情感弧 |
| **Sentence** | 逐句: 语气、语调、语速、音量、说话意图、背景状态 + 句内富文本标记(打断线索、语气转折等) | 逐行: 情感/节奏/戏剧停顿 |
| **Token** | 音素级: 重音位置、多音字消歧、连读规则、感叹词时长 | 逐字: 重音/连读/呼吸 |

[论文原文] 关键设计决策: **每个层级的每个维度都用自由格式自然语言 caption 表达,而非固定枚举或数值编码** [§2.2]。原因是真实情感很少是一维的 -- 含泪的笑、兴奋的低语、讽刺的赞美、强装镇定下的恐慌 -- 这些复合状态抵制离散标签但可用自然语言轻松表达。caption 格式也与 LLM 的输入/输出空间天然对齐 [§2.2]。

#### 3. Chain-of-Thought: 先理解,后合成 [§3.1]

[论文原文] 传统 TTS 将理解和发声压缩为单次映射: 模型从文本直接跳到音频,所有韵律决策都埋在网络内部,无法观察也无法纠正 [§3.1]。

**Instruct-Think 两流分区** [§3.1]:
- **Instruct 流**(用户硬约束): 场景元数据、说话人身份、声学环境评级
- **Think 流**(模型推理计划): 全局氛围和情感弧、逐句语气/语调/节奏/音量/意图/背景状态、音素级发音细节(重音/连读/变调)

[论文原文] 推理时模型不直接跳到音频,而是先进入规划阶段: 读取全局指令,处理文本,逐句推理 Think 维度。对每个话语确定语气基线、语调轮廓、节奏、交际意图和语音实现 -- 全部作为显式、可检查的输出。然后才合成。这种"先想后说"的流水线将黑盒韵律决策变为可追踪、可解释、可编辑的推理链 [§3.1]。

[agent 解读] 这与 RALL-E 的 chain-of-thought prompting 有思路上的延续,但 RALL-E 的 CoT 主要用于改善 duration/prosody 预测的鲁棒性,而本文的 CoT 扩展到覆盖整个 Global-Sentence-Token 多维度空间,且不仅是 prompting 技巧而是训练时的结构化分区。与 OV-InstructTTS 的 `<think>` reasoning chain 相比,本文的 CoT 更侧重多层级场景规划而非情感推断。

#### 4. Dimension Dropout [§3.2]

[论文原文] 训练时随机 mask 某些 Think 维度(如声学环境描述或情感轨迹)。被 mask 的槽位保持空白 -- 模型不被要求重建它们,只需学习在信息不完整时仍产出高质量音频 [§3.2]。

**两个实际好处** [§3.2]:
1. 防止对任何单一线索的过度依赖 -- 当某些维度缺失时,模型更忠实地遵循剩余指令
2. 推理时用户可以只指定关心的维度,仍获得好结果

[agent 解读] 这本质上是条件生成中的一种正则化策略,与 Classifier-Free Guidance 中的无条件训练(随机丢弃条件)思路类似,但应用于多维度结构化条件而非单一条件。与 Quantizer Dropout (SoundStream 中随机丢弃 RVQ 层)的设计哲学相近 -- 通过训练时引入缺失迫使模型学习更鲁棒的表示。

#### 5. Native Agentic 架构 [§4]

[论文原文] 本文声称系统是 "Native Agentic" -- 但不是"它接受文本输入所以可以在前面放一个 LLM"这种意义上的(按此标准所有 TTS 系统都符合) [§4]。

**核心区分: 接口带宽** [§4]:
- 传统 TTS 文本接口是**窄带**的: 仅携带词汇内容和可能的 speaker ID 或 style tag。上游 LLM 即使理解场景/情感/声学,也被压缩到只能传递文字
- 本系统的 GST schema 将管道扩展为**全带宽、信息完整的控制通道**: 场景定位、说话人画像、情感弧、声学环境、交互动态都与文本一同传递 [§4]

**结构化语义接口** [§4.1]: GST schema 在更高抽象层上充当 LLM Agent ↔ 合成引擎之间的标准化合同,三层映射到分层控制协议栈:
- Global = 会话/应用层(场景上下文、说话人画像、情感弧)
- Sentence = 传输层(逐句语气、意图、韵律)
- Token = 物理层(音素级声学细节)

**上下文效率优势** [§4.2]:
[论文原文] 相比端到端口语对话模型需要将完整对话历史喂入网络(token 成本随每轮线性增长,且注意力的长距离衰减导致早期上下文被逐步忽略 -- "lost in the middle" 现象),Agent 架构用审慎的 schema 引导压缩替代蛮力上下文注入。LLM 将对话历史蒸馏为紧凑的情感-上下文状态,映射到 GST 维度。合成引擎接收的不是原始历史而是语义蒸馏的指令集 [§4.2]。

### 训练策略

[论文原文] 在相同训练集上进行受控对比后,选择了带有**连续 tokenizer** 的 backbone,认为这对无边界生成至关重要 [§3]。

[agent 解读] 论文对 backbone 架构细节几乎未做披露 -- 不清楚是 flow-matching、diffusion 还是其他生成范式,也未说明连续 tokenizer 的具体实现(是类似 MELLE 的 mel 连续预测,还是类似 CLEAR 的 rectified flow head)。这是本文最大的信息缺口之一。

## 实验

[论文原文] 作者明确声明无法以表格形式呈现量化能力边界 [§5]。他们探索了多种候选评估方案但均不满意:

| 评估方案 | 问题 | 出处 |
|----------|------|------|
| CLAP | 在短片段上训练,缺乏对长音频现象(持续情感弧、多说话人动态)的判别力 | [§5] |
| Audio Captioning | 描述粒度太粗,无法覆盖三层标注中编码的丰富语义 | [§5] |
| DNSMOS/PESQ/POLQA | 波形级声学质量指标,对场景建模/表现力/指令遵循完全盲目 | [§5] |

作者将 Generation 4 TTS 的评估框架视为独立的开放研究问题,鼓励读者通过直接聆听 demo 体验系统表现力 [§5]。

## 局限性

1. **无定量实验**: 全文无任何数字化评估结果,论证完全基于概念框架和定性描述,无法验证声称的改进 [§5]
2. **仅面向离线内容创作**: 优化目标是播客、有声书、影视旁白等离线制作场景,未适配实时交互(语音对话、直播等),后者需要毫秒级响应和流式输出 [§6]
3. **模型架构细节缺失**: backbone 仅以 "continuous tokenizer" 一句带过,训练数据规模/组成、模型参数量、推理速度等关键信息均未披露 [§3]
4. **训练数据偏向语音**: 音效和音乐语料尚未纳入,当前的音效/音乐生成能力完全是从播客/访谈/体育解说的背景音频中涌现的,有明确的能力天花板 [§6]
5. **不支持参考音频音色控制**: 仅支持自然语言音色描述,尚未实现通过参考音频锁定目标说话人音色 [§6]
6. **未开源**

## 点评

**贡献在哪里**: 本文的核心贡献不在模型架构创新(实际上架构几乎未披露),而在于提出了一套有说服力的系统级思考框架:

1. **数据哲学的颠覆**: "label don't filter" 是对 TTS 数据处理正统做法的根本挑战。传统流水线丢弃 70-90% 数据以追求"纯净",但本文论证这些被丢弃的"脏数据"(重叠对话、情感爆发、复杂场景)恰恰是表现力的最丰富来源,关键在于用足够详细的标注将"噪声"转化为"可控维度"。这一观点对标注成本vs数据量的 trade-off 有实际指导意义。

2. **控制带宽的概念**: 用"窄带 vs 宽带"类比来诊断当前 TTS agent 系统的瓶颈(上游 LLM 理解丰富但传递给 TTS 的信息被压缩到纯文本)是精准的。GST schema 作为"结构化语义接口"的定位清晰地解决了这个信息瓶颈。

3. **CoT 显式化韵律决策**: 将隐式韵律建模翻转为显式推理链,既提升可解释性又赋予用户编辑能力,是一个有价值的设计方向。

**主要质疑**:

1. **缺乏实证支撑是致命弱点**: 所有声称的优势(表现力提升、指令遵循改善、全局连贯性)均无定量验证。即便评估框架确实是开放问题,至少可以报告人类评估(A/B test、MOS)或 demo 级对比。没有任何数字,这更像是一篇立场论文(position paper)而非技术论文。

2. **分层标注的成本和可扩展性**: Global-Sentence-Token schema 的标注复杂度远超传统 TTS 数据标注。论文未讨论: 标注来源(人工? LLM 生成?),标注质量控制,标注成本是否可持续。如果依赖 LLM 标注,标注质量本身就是一个未验证的假设。

3. **架构不可复现**: backbone 几乎无描述,无法判断技术可行性。"continuous tokenizer" 是什么具体实现? 训练了多少数据? 模型多大? 这些基本信息的缺失使技术贡献难以评估。

## 可复用的 idea

1. **"Label, don't filter" 数据策略**: 适用于任何需要处理含噪/复杂真实数据的场景。核心洞察是"脏数据"+ 详细标注 > "干净数据"+ 简单标注,尤其当表现力是目标时。可直接应用于情感 TTS、多说话人对话合成等任务的数据准备。

2. **Dimension Dropout**: 多条件生成任务中的通用正则化技巧。训练时随机丢弃部分条件维度,推理时支持部分条件指定。可迁移到图像生成(部分描述生成)、音乐生成(部分风格指定)等任务。

3. **分层控制协议栈**: Global-Sentence-Token 的分层思路可作为 TTS agent 系统的接口设计参考 -- 将控制信号组织为场景层/话语层/音素层,每层用自然语言 caption,既人类可读又 LLM 可生成。

4. **Instruct-Think 两流分区**: 将用户约束(不可协商)与模型推理(可自动规划)显式分离,是构建可控生成系统的通用设计模式。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 三大支柱均有 WHY 解释,设计选择有因果链 |
> | 可信赖 | pass | 论文无定量数据,笔记如实反映;出处标注完整 |
> | 可区分 | pass | [论文原文]/[agent 解读] 覆盖率高,推断与事实边界清晰 |
> | 可定位 | pass | KB 背景有具体谱系定位和创新判断对比 |
> | 不污染 | pass | concepts 挂接合理,无新建概念页,反向更新为追加 |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/BorderlessLongSpeech-review.yml`
