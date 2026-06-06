---
type: paper
tier: deep
title: "Humane Speech Synthesis through Zero-Shot Emotion and Disfluency Generation"
arxiv_id: "2404.01339"
source: "Sources/HumaneSpeech.pdf"
authors: [Rohan Chaudhury, Mihir Godbole, Aakash Garg, Jinsil Hwaryoung Seo]
year: 2024
venue: "LREC-COLING 2024"
tags: [TTS, emotion, disfluency, zero-shot, rule-based, virtual-patient, conversational-AI, prompt-engineering]
concepts: ["[[EmotionControlinTTS]]", "[[ProsodyModeling]]", "[[Text-to-SpeechPipeline]]"]
models: ["[[SpeechT5]]", "[[MMS-TTS]]"]
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
>
> **谱系定位**: 本文提出的"GPT-4 prompt engineering + 规则映射 TTS"pipeline 处于 Emotion Control in TTS 演进链中的极早期位置 — 接近 HMM 时代的规则情感合成 (2003),远早于 emotion embedding (2021)、DPO 对齐 (2024)、activation steering (2025) 等现代方法。从 TTS Pipeline 演进看,本文使用的 SpeechT5/MMS-TTS 属于 Stage 3 (AM + Vocoder) 范式,未采用 LLM-based TTS 范式。
>
> **已有认知**: 概念库中 [[EmotionControlinTTS]] 已覆盖大量现代情感控制方法(EmoSphere++, EmoSteer-TTS, UDDETTS, WeSCon 等),均通过嵌入空间、激活空间或参数空间操作实现端到端情感合成。本文采用的文本层面标签注入 + 规则音频拼接路线在概念库中无直接对应 — 最接近的是 NVSpeech 的显式标签插入方法,但 NVSpeech 是在 neural TTS 内部建模而非外部拼接。[[ProsodyModeling]] 中副语言发声 (NVSpeech) 和情感感知韵律分句 (EmoPP) 与本文的 disfluency 建模有概念交集。
>
> **创新判断**: 本文的核心思路 — 让 LLM 在文本生成阶段同时输出情感/disfluency 标签,再由 TTS 系统消费 — 是一种 pragmatic 的工程方案。与当前主流的端到端情感控制方法相比,本文不修改 TTS 模型本身,而是在文本层面解决问题。这种"分而治之"思路在 2024 年的 conversational AI 应用中有实际价值,但在 TTS 研究层面缺乏可迁移的技术贡献。
>
> 检索命中: [[EmotionControlinTTS]][待确认], [[ProsodyModeling]]✓, [[Text-to-SpeechPipeline]][待确认], [[LLM-basedTTS]]✓, [[TTSEvaluation]][待确认], [[Zero-shotSpeechSynthesis]]✓ | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 GPT-4 零样本 prompt engineering 在文本生成阶段同时输出情感标签和 disfluency 标记,再通过规则映射将这些标记转换为预录音频片段与 TTS 合成语音拼接,实现"类人"语音
> - **路线**: User Input → GPT-4 (with emotion/disfluency prompt) → Annotated Text → Rule-based Parser → {Clean Text → SpeechT5/MMS-TTS → Speech} + {Emotion Cues → Predefined Audio Mapping} + {Disfluencies → TTS Synthesis + Stretching} → Concatenated Waveform
> - **指标**: 无标准客观指标(无 MOS/WER/SIM),仅有定性主观分析; SpeechT5 被评为三个 TTS 模型中综合最优 [§5.2.3]
> - **可借鉴**: 将 LLM 作为情感/disfluency 标签的零样本生成器,利用其世界知识在对话上下文中产生语境合适的非言语行为标注 — 这个 idea 可迁移到现代 NV-capable TTS 的标签生成阶段
> - **局限**: 无定量评估(无 MOS/MCD/WER); 规则映射不可泛化(新情感需手动添加音频); 分段拼接音频不自然; SpeechT5/MMS-TTS 已过时; 仅在单一虚拟患者场景验证; 代码可用但可复现性低(依赖 GPT-4 API + 特定音频素材)

## 核心问题

本文试图解决的问题: 现有对话式 AI 系统的语音输出缺乏情感深度和自然的口语特征(如犹豫、口吃、叹息),使其听起来机械化,尤其在需要共情交互的场景(如虚拟患者训练)中体验不佳。

**与已有方法的区别**: 此前的方法要么在 TTS 之后检测并插入情感 (Lee et al., 2017; Im et al., 2022),要么训练专门的情感 TTS 模型 (Diatlova and Shutov, 2023 EmoSpeech; Guo et al., 2023 EmoDiff)。本文的不同之处在于让 LLM 在文本生成阶段就同步产生情感和 disfluency 标记,声称这种"上下文感知"的同步生成比事后插入更自然 [§3.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统包含三个串联阶段 [§3, Fig 1]:

1. **LLM 文本生成**: GPT-4 接收带有角色背景的系统 prompt + 对话历史,零样本生成含情感标签 (`*sighs heavily*`, `*cries softly*`) 和 disfluency 标记 (filler words `uh, um`, 口吃 `y-yeah, m-my`, 停顿 `...`) 的回复文本 [§3.1-3.3]

2. **文本解析**: 规则解析器将 LLM 输出分割为 clean text segments 和 disfluency/emotion cue segments [§3.5.1]

3. **分段合成 + 拼接**: Clean text 用 SpeechT5 合成; 情感标签映射到预录音频 (按 cosine similarity 选择强度级别); disfluency 用 SpeechT5 合成后拉伸 + 加停顿; 口吃通过词重复或部分重复实现; 最后拼接所有波形 [§3.5.2]

### 关键设计选择

**为什么用零样本 prompt 而不训练专用模型?** [论文原文] 作者认为 emotion/disfluency 的训练数据稀缺,GPT-4 的 intrinsic knowledge 足以在无显式训练的情况下生成合适的情感和 disfluency 标记 [§3.2]。[agent 解读] 这实际上是将情感建模的难度从 TTS 模型转移到了 LLM 的文本生成能力上,规避了训练情感 TTS 模型的数据和计算成本,但代价是牺牲了语音层面的情感表达精度。

**三级 prompt 设计**: 通过 neutral / moderate / extreme 三种 prompt 控制情感和 disfluency 的强度 [§3.3.1-3.3.3]。Neutral prompt 不包含情感指示; moderate prompt 要求"some amount of emotional and action tags"; extreme prompt 要求"extreme amounts" [§3.3]。[agent 解读] 这本质上是一种粗粒度的离散控制,无法实现连续的情感强度调节。

**情感强度量化**: 当 LLM 生成带强度描述的情感标签 (如 `*sighs heavily*`) 时,系统用 SpaCy `en_core_web_md` 模型计算情感词和三个参考强度词的 word embedding cosine similarity,将强度分为 0-2 三级,映射到不同音频文件 [§3.5.2]。[agent 解读] 这种 3 级量化粒度非常粗糙,且 word embedding 的 cosine similarity 未必能准确反映情感强度的语义差异。

**口吃合成**: 两种策略随机切换以模拟口吃的随机性 [§3.5.2]:
- 短词 (< 4 字符): 整词重复 (`m-my` → `my my`)
- 长词: 策略 1 — 部分重复 n-1 次 (`r-recently` → `rec recently`); 策略 2 — 整词重复中间插入停顿和填充词 (`r-recently` → `recently... um... recently`)

**Memory 组件**: 基于 Garcia-Pi et al. (2023) 的三层记忆: Background Memory (角色设定), Initial Memory (最初几轮对话), Latest Memory (最近对话队列),确保长对话的上下文一致性 [§3.4]。

### 训练策略

本文无模型训练 — 完全依赖 GPT-4 的零样本能力和规则映射,不涉及任何 TTS 模型的微调或训练。

## 实验

本文未报告标准定量指标。评估完全基于定性主观分析 [§4-5]。

**文本生成分析** [§5.1]:
- Neutral prompt: 输出无情感/disfluency,被评为"robotic and detached" [Table 1]
- Moderate prompt: 输出含适量情感标签和 disfluency,"seem more humane" [Table 2]
- Extreme prompt: 输出含大量情感标签和 disfluency [Table 3]
- LLM 会"幻觉"出 prompt 中未指定的手势标签 (如 `*looks down*`, `*sobs*`),作者认为这些幻觉在语境中是合理的 [§5.1]

**语音合成对比** [§5.2]:

| TTS 模型 | Disfluency 合成质量 | 流畅性 | 综合 | 出处 |
| --- | --- | --- | --- | --- |
| SpeechT5 | 最自然 | 良好(轻微噪音) | **最优** | [§5.2.2-5.2.3] |
| Google Cloud TTS | 与 SpeechT5 接近 | **最优** | 第二 | [§5.2.2-5.2.3] |
| MMS-TTS | 差(快速、突兀) | 机械 | 最差 | [§5.2.2-5.2.3] |

**关键发现** [§5.2.1]: Extreme prompt 的过多 disfluency 导致文本过度分段,合成语音中断过多,反而听起来不自然 — 存在 disfluency 数量与语音自然度的 trade-off。

## 局限性

1. **无定量评估**: 全文无 MOS、WER、MCD、Speaker Similarity 等标准指标,所有结论基于作者自身的定性描述,可信度低 [agent 解读]

2. **规则映射不可泛化**: 情感标签到音频的映射是手动预定义的有限集合; LLM 生成的"幻觉"情感需通过 cosine similarity 回退到最近匹配,处理无界集合的能力本质受限 [§3.5.2, §7.2]

3. **拼接产生的不自然性**: 将 clean speech、emotion audio、disfluency speech 简单拼接,缺乏跨段的韵律连贯性 (过渡不平滑、语速/音高突变) [§5.2.1]

4. **TTS 模型过时**: SpeechT5 (2021) 和 MMS-TTS (2023) 在 2024 年已远落后于 CosyVoice/VALL-E/F5-TTS 等系统的自然度水平 [agent 解读]

5. **单一应用场景验证**: 仅在 "Pastor Zimmerman" 虚拟患者案例上测试,未验证在其他情感场景或说话人上的泛化能力 [§4.1]

6. **情感真实性问题**: 论文自身承认"synthetic representation of emotion may never fully capture the nuance and complexity of genuine human emotion" [§7.2]

7. **伦理风险**: 论文指出高度拟人的 AI 语音存在欺骗、情感操纵、偏见强化等风险 [§7.1],但未提出具体的防范措施

## 点评

本文是一篇 **应用导向的系统工程论文**,而非 TTS 技术研究论文。其核心贡献在于提出了一种可快速部署的工程方案: 利用 GPT-4 的文本能力生成情感/disfluency 标注,绕过了训练情感 TTS 模型的高成本。

**优势**: 思路直觉且实现简单 — 不修改 TTS 模型,仅在文本层面解决情感表达问题,适合资源受限的应用场景。利用 LLM 作为"情感标签零样本生成器"的 idea 有启发性。

**根本性不足**: (1) 零定量评估使所有结论无法验证; (2) 规则拼接方法在语音层面的自然度上限很低 — 分段合成的语音片段无法实现自然的情感韵律连贯性; (3) 与 2024 年 emotional TTS 前沿 (EmoCtrl-TTS, UDDETTS, EmoSteer-TTS 等端到端方法) 相比,技术差距巨大。

**在知识库中的定位**: 本文代表了一条"LLM 文本标注 + 规则 TTS 消费"的非主流路线。这条路线的后续发展可以在 NVSpeech (2025) 中看到 — NVSpeech 也使用文本中的显式标签 (`[Laughter]`, `[Breathing]`) 控制副语言行为,但关键区别是 NVSpeech 在 neural TTS 内部端到端建模这些标签,而非本文的外部规则拼接。

## 可复用的 idea

1. **LLM 作为情感/disfluency 标签零样本生成器**: 利用 LLM 的世界知识和上下文理解能力,在文本生成阶段同步产生语境合适的非言语行为标注。这个 idea 可迁移到 NV-capable TTS 系统的训练数据标注阶段 — 用 LLM 为无标注的对话文本自动添加 disfluency 和 emotion tag。

2. **三层记忆架构** (Background + Initial + Latest): 用于长对话的上下文管理,保持角色一致性。这是通用对话系统设计模式,非本文原创但实现清晰。

3. **Disfluency 分类与处理**: 将 disfluency 分为 interjections / stutters / pauses / emotion cues 四类并分别处理的框架 [§3.5.2],可作为 disfluency-aware TTS 数据标注的参考分类体系。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节 WHY/HOW 清晰,速查卡片具体可迁移 |
> | 可信赖 | pass-with-fixes | 论文无定量指标已如实反映; 1 处 agent 判断未标注来源(已修正) |
> | 可区分 | pass | [论文原文]/[agent 解读] 区分覆盖率高 |
> | 可定位 | pass | KB 背景谱系定位具体,与 NVSpeech 的对比有意义 |
> | 不污染 | pass | 技术贡献有限,反向更新仅 append,污染风险低 |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/HumaneSpeech-review.yml`
