---
type: paper
tier: deep
title: "Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs"
arxiv_id: "2604.12506"
source: "Sources/UnifiedAudioSchema.pdf"
authors: [Linhao Zhang, Yuhan Song, Aiwei Liu, Chuhan Wu, Sijun Zhang, Wei Jia, Yuan Liu, Houfeng Wang, Xiao Zhou]
year: 2026
venue: "arXiv preprint"
tags: [AudioLLM, perception, paralinguistic, supervision, structured-schema, audio-understanding, speech-factorization, JSON-supervision]
concepts: ["[[AudioUnderstanding]]", "[[SpeechFactorization]]", "[[ProsodyModeling]]", "[[ModalityAdaptationforSpeechLLM]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: ["[[SenseVoice]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

> [!review] 自动审阅 (2026-06-08)
> **结论:** pass
> **原则:** 复述 9 | 信赖 9 | 区分 8 | 定位 9 | 污染 9

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: UAS 论文处于 AudioLLM 感知能力提升的前沿。KB 中 [[AudioUnderstanding]] 页已建立 SpeechLM 理解能力的任务分类体系(语义/说话人/副语言三类),本文的核心贡献正是针对其中"副语言理解"和"非语言事件理解"的系统性薄弱环节提出结构化监督方案。与 [[SpeechFactorization]] 中 TTS 领域的属性解耦(content/timbre/prosody/emotion/environment)形成镜像关系:TTS 端需要将属性分解以实现可控生成,而 UAS 在理解端将相同属性显式解耦以实现全面感知——两者解耦的属性维度高度一致。
>
> **已有认知**: (1) [[SpeechFactorization]] (confirmed) 总结了 6 维属性解耦(content, speaker/timbre, emotion, prosody, language, environment),UAS 的三组件分解(Transcription/Paralinguistics/Non-linguistic Events)与之对应但在理解侧操作; (2) [[ProsodyModeling]] (confirmed) 指出 prosody 是 TTS 中 one-to-many mapping 的核心来源,UAS 将 prosody 作为 paralinguistics 的显式子字段建模; (3) [[ConditionalFlowMatching]] (confirmed) 是 UAS-Audio 语音解码器使用的生成模型; (4) [[AudioUnderstanding]] [待确认] 建立了 SpeechLM 三类理解任务的分类,指出"SpeechLM 不仅理解 what is said,还理解 how it is said"——这恰好是 UAS 要解决的核心问题; (5) [[SenseVoice]] [待确认] 是论文直接对比的相关工作,其 rich transcription 通过 interleaved tags 实现,UAS 论文指出这种"flat"标注不能推广到 prosody/timbre 等连续属性。
>
> **创新判断**: UAS 的核心新意不在于"解耦"本身(TTS 侧已有大量工作),而在于: (a) 将解耦思想从生成侧迁移到理解侧的监督信号设计; (b) 用结构化 JSON 格式(而非自然语言 caption)提供低熵、可编程的监督目标; (c) 证明感知与推理可以独立提升,即 ASR-centric 训练造成的是感知瓶颈而非推理瓶颈。
>
> 检索命中: [[SpeechFactorization]]✓, [[ProsodyModeling]]✓, [[ConditionalFlowMatching]]✓ | 过滤: [[AudioUnderstanding]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review), [[SenseVoice]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 Unified Audio Schema (UAS),将音频监督信号从纯 ASR 文本扩展为 Transcription + Paralinguistics + Non-linguistic Events 三组件结构化 JSON,解决 AudioLLM 感知能力系统性不足的问题
> - **路线**: 音频 → Audio Encoder (AuT) → Projection Layer → LLM (Qwen2.5-7B) → text/audio output; 训练时用 UAS JSON 作为监督目标,推理时可用标准 prompt 不生成完整 JSON
> - **指标**: MMSU perception +10.9% (55.7% vs best baseline 44.8%); MMSU reasoning 77.4% (与 Qwen2.5-Omni 77.6% 持平); MMAR overall 60.1% (SOTA); MMAU average 69.4%; Seed-TTS WER avg 1.6 (best) [Table 1, Table 2]
> - **可借鉴**: (1) 结构化 JSON 作为训练时的密集监督信号、推理时可灵活切换的设计模式; (2) 感知与推理独立的实验证据可迁移到 TTS 评估; (3) UAS-QA 辅助数据集将 schema 知识转化为 task performance 的训练策略
> - **局限**: 仅验证中英两种语言; 单说话人场景(不处理 cocktail party); 训练成本未公开; 对比 baseline 有限(仅 3 个 7B 模型)

## 核心问题

本文要解决的核心问题是: **为什么 AudioLLM 在复杂推理任务上表现优秀,却在基本的声学感知任务上表现糟糕?**

具体而言,MMSU benchmark 上现有 AudioLLM 在推理子集达 ~70% 准确率,但感知子集仅 ~40% [§1]。这种 perception-reasoning inversion 在不同模型规模和架构中持续存在,说明问题不在模型容量,而在训练监督信号本身。

论文的核心假设: ASR-centric 训练信号是罪魁祸首。ASR 的目标是恢复规范文本,训练过程中系统性地将韵律、说话人身份、情感和声学环境视为"噪声"并 normalize away,导致模型被隐式训练去**忽略**这些信息 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UAS-Audio 采用标准 AudioLLM 四组件框架 [§3.1]:

1. **Audio Encoder**: AuT (Audio Transformer, Xu et al., 2025b) 将原始波形编码为连续表征
2. **Projection Layer**: 线性投影层将音频表征对齐到 LLM embedding 空间
3. **LLM Backbone**: Qwen2.5-7B,在联合 audio-text 输入上执行推理
4. **Speech Decoder**: 基于 flow matching 架构 (Lipman et al., 2023),将离散 audio token 转换为 mel spectrogram,再通过 HiFi-GAN vocoder 生成波形

[论文原文] 作者强调"we do not introduce new modules or specialized loss functions; we simply plug in the UAS data" [§3.2]——即架构创新为零,核心贡献全在监督信号的设计。

### Unified Audio Schema (UAS) 设计

UAS 基于 Laver (1994) 的语音信号符号学框架,将音频信息分解为三个互补组件 [§2.1]:

**1. Transcription**: 对应 linguistic layer,保留 100% 语言保真度的逐字转录,与 ASR 输出等价。确保 UAS 不会在语义精度上牺牲任何东西。

**2. Paralinguistics**: 对应 paralinguistic + extralinguistic (speaker-intrinsic) layer,包含 6 个显式子字段:
- Age: 说话人年龄组 (Child/Adult/Elderly)
- Gender: 说话人性别
- Emotion: 情感状态 (7 类受控词表: Anger/Disgust/Sadness/Happiness/Neutral/Surprise/Fear)
- Accent: 地区/语言口音
- Prosody: 语速、语调、节奏等说话风格描述
- Timbre: 声音质量特征

**3. Non-linguistic Events**: 对应 extralinguistic (environmental) layer,包含 3 个子字段:
- Description: 录音环境整体描述
- Discrete Events: 有明确时间边界的声音事件 (如门响)
- Continuous Events: 贯穿音频的持续背景声 (如引擎噪声)

所有内容以 JSON 格式组织,每个 discrete/continuous event 附带 label + characteristic 标注。

[论文原文] 结构化设计的三重优势 [§2.1]:
1. **Disentangled learning**: 将"整体理解"的隐式任务转化为显式子任务,防止特征混淆
2. **Syntactic invariance**: 与非结构化 caption 的高变异性不同,JSON 提供低熵、一致的监督目标,降低学习难度
3. **Programmatic accessibility**: 严格 JSON 格式桥接 LLM 概率性输出与下游应用的确定性需求

### UAS 数据生成管线

三阶段自动化管线 [§2.2, Fig 2]:

**Stage 1: Acoustic Caption Generation** — 用 caption 模型 (Xu et al., 2025b) 对原始音频生成富描述,提取 ASR 监督天然丢弃的声学信息。

**Stage 2: Structured Schema Synthesis** — 将 acoustic caption + ground-truth transcription 通过 LLM (Qwen3-30B-A3B-Instruct) 合成结构化 UAS JSON [§2.2, Appendix H]。关键: transcription 字段保持原始 ground-truth 逐字不变,paralinguistic 和 non-linguistic 字段从 caption 中提取和归一化。

**Stage 3: Quality Validation** — 多层自动验证 [§2.2]:
- Ontology 约束: 分类字段 (emotion, gender) 验证受控词表
- Transcription 完整性: 与 ground truth 精确字符串匹配
- 逻辑一致性: 规则检查解决字段间冲突 (如空转录 → null paralinguistics)
- 时长-内容对齐: 启发式过滤描述复杂度超过音频时长的样本

人工验证 (N=400): 大部分 paralinguistic 和 environmental 字段准确率 >95%; Emotion 89.0%, Discrete Events 91.75% 相对较低但仍 >84% [Table 4]。

### UAS-QA 辅助数据集

基于 UAS 标注自动生成三类 QA 对 [§2.3]:
1. **Direct QA**: 查询特定 UAS 字段 ("What is the speaker's emotion?" → "Neutral")
2. **Multiple Choice**: 带候选选项的选择题
3. **Yes/No**: 二元验证问题

[论文原文] UAS annotation 教模型"what to perceive",UAS-QA 教模型"how to apply this knowledge" [§3.2]。

### 训练策略

四阶段标准多阶段对齐协议 [§3.2]:

**Stage 1: Discrete Token Alignment** — 扩展 LLM 词表加入 StableToken (Song et al., 2026) 离散声学码,通过 ASR+TTS 任务对齐文本-音频表征。仅 embedding layer + LLM head 可训练。

**Stage 2: Audio-LLM Adaptation** — 在 UAS annotation 数据上训练 projection layer (LLM + encoder frozen)。[论文原文] 关键设计选择: 在此阶段就引入结构化声学理解,"prevents the model from developing ASR-centric representations that would later need to be 'unlearned'" [§3.2]。

**Stage 3: Full Instruction Tuning** — 解冻所有参数 (除 audio encoder),在混合数据上训练: (1) ASR/TTS 基础数据; (2) UAS annotation; (3) UAS-QA。

**Stage 4: GRPO** — Group Relative Policy Optimization (Shao et al., 2024) 进一步优化。

### 关键设计选择

1. **训练-推理解耦** [Appendix D]: UAS JSON 仅在训练时作为密集监督信号; 推理时可通过 task-specific prompt 以标准格式输出 (如纯 ASR 文本、情感标签),不需要生成完整 JSON。[agent 解读] 这意味着 UAS 的训练成本体现在数据准备而非推理延迟,部署时零额外开销。

2. **UAS 在 Stage 2 引入而非 Stage 3** [§3.2]: [论文原文] 早期引入防止模型形成 ASR-centric 表征后再"unlearn"。[agent 解读] 这暗示 ASR-centric bias 一旦形成很难逆转,与 catastrophic forgetting 的机制可能相关。

3. **结构化 vs 非结构化** [Appendix F]: 在控制变量实验中 (相同数据源、无 GRPO),结构化 UAS 比非结构化 caption 高 6.4% perception [Table 7]。[论文原文] 原因: JSON 的正交 slot 防止语义干扰,低熵目标降低学习难度,强制完备性确保所有声学字段都被预测。

4. **离散架构变体 (UAS-Audio-D)** [Appendix B]: 基于 Qwen2.5-3B + GLM-4-Voice 音频分词器,省略 adapter alignment 和 GRPO,在 GLM-4-Voice baseline 上将整体平均分从 24.4% → 44.2%,近乎翻倍 [Table 1]。

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| MMSU Perception | **55.7%** | 44.8% (Kimi-Audio) | MMSU | [Table 1] |
| MMSU Reasoning | 77.4% | **77.6%** (Qwen2.5-Omni) | MMSU | [Table 1] |
| MMSU Overall | **66.2%** | 62.2% (Kimi-Audio) | MMSU | [Table 1] |
| MMAR Overall | **60.1%** | 56.8% (Step-Audio2) | MMAR | [Table 1] |
| MMAR Speech | **66.0%** | 61.2% (Step-Audio2) | MMAR | [Table 1] |
| MMAR Music | **45.2%** | 42.2% (Step-Audio2) | MMAR | [Table 1] |
| MMAU Overall | 69.4% | **72.7%** (Step-Audio2) | MMAU | [Table 1] |
| MMAU Speech | 67.0% | 68.2% (Step-Audio2) | MMAU | [Table 1] |
| MMAU Sound | 70.0% | **79.3%** (Step-Audio2) | MMAU | [Table 1] |
| MMAU Music | **71.3%** | 68.4% (Step-Audio2) | MMAU | [Table 1] |
| Cross-benchmark Avg | **65.2%** | 62.1% (Qwen2.5-Omni) | All 3 | [Table 1] |
| Seed-TTS WER Avg | **1.6** | 1.9% (Qwen2.5-Omni) | Seed-TTS | [Table 2] |
| ASR WER (LS-clean) | 2.2 (targeted) / 2.3 (holistic) | — | LibriSpeech | [Table 3] |
| ASR WER (AISHELL) | 2.3 (targeted) / 2.4 (holistic) | — | AISHELL | [Table 3] |

**消融实验 (MMSU)** [Fig 4, Table 6]:
- 移除 UAS annotation: perception -6.3% (55.7→50.7), reasoning 基本不变
- 移除 UAS-QA: perception -9.6% (55.7→47.0), reasoning 基本不变
- 移除两者: perception -15.0% (55.7→42.8), reasoning 基本不变
- 移除 GRPO: perception -0.9% (55.7→54.8), reasoning -1.4% (77.4→76.0) [Table 6]
- 结构化 UAS vs 非结构化 caption (无 GRPO): perception +6.4% (48.4→54.8) [Table 7]

**关键发现**: (1) UAS-QA 的贡献大于 UAS annotation (9.6% vs 6.3%),说明"如何应用知识"比"知道什么"更关键; (2) GRPO 仅贡献 0.9% perception 增益,证明性能提升主要来自 UAS 监督而非 RL 技巧; (3) 推理能力在所有配置中完全稳定,验证了感知-推理独立假说。

## 局限性

1. **语言多样性有限**: 仅在中英两种高资源语言上验证,UAS 对低资源语言和 code-switching 场景的效果未知 [§Limitations]
2. **单说话人假设**: 当前 paralinguistic 分析聚焦于 primary speaker,不能处理多说话人重叠场景 (cocktail party problem) [§Limitations]
3. **Baseline 范围窄**: 仅对比 3 个 7B 级别 baseline (Qwen2.5-Omni, Kimi-Audio, Step-Audio2-mini),缺乏与更大或更小规模模型的对比
4. **训练成本不透明**: 使用"数十万小时"音频数据训练,但未公开具体计算成本和训练时长
5. **UAS 数据质量依赖管线**: 三阶段管线依赖 caption 模型和 LLM 的质量,Emotion (89%) 和 Discrete Events (91.75%) 的准确率相对较低,错误会传播到训练数据中
6. **Schema 灵活性 vs 覆盖度 trade-off**: 固定 6+3 子字段的 schema 可能无法覆盖所有音频场景(如音乐分析中的调性、节拍等更精细维度),但论文未讨论 schema 的可扩展性

## 点评

**最有价值的贡献**: UAS 的核心价值不在于架构创新(作者自己承认架构无变化),而在于一个简洁但深刻的 insight——当前 AudioLLM 的感知问题是**监督信号的结构性缺陷**,不是模型能力不足。通过重新设计监督信号就能获得 +10.9% 的感知提升,且完全不损害推理能力,这个发现本身比具体的 UAS schema 设计更重要。

**与 KB 中 SpeechFactorization 的关系**: UAS 本质上是将 TTS 领域成熟的"属性解耦"思想反向迁移到理解侧。TTS 中 NaturalSpeech 3 等工作将语音分解为 content/prosody/timbre/acoustic 以实现可控生成;UAS 将监督信号分解为 transcription/paralinguistics/events 以实现全面感知。两者解耦的属性维度高度一致,但操作方向相反——一个是编码端(分解表示用于生成),一个是解码端(分解标签用于学习)。

**与 SenseVoice 的对比**: SenseVoice 通过 interleaved special tokens 实现 rich transcription,UAS 论文正确指出这种"flat"标注在处理 prosody、timbre 等连续属性时表达力不足。UAS 的结构化 JSON 是一种更通用、更可扩展的监督格式。

**感知-推理独立性假说**: 消融实验中推理能力在所有配置中完全稳定(77.0-77.4%),这是一个非常有力的实验证据,说明 AudioLLM 的感知和推理确实是独立因素,当前模型的瓶颈在感知而非推理。

**训练-推理解耦设计值得关注**: UAS JSON 仅作训练信号、推理时可自由选择输出格式,意味着部署零开销。这种"训练时密集监督、推理时灵活输出"的范式具有广泛借鉴价值。

**不足之处**: (1) 对比实验局限于 3 个 7B 模型,缺乏 scaling law 分析; (2) Schema 的 6+3 子字段设计是手工选择的,论文未探讨自动 schema 发现; (3) 数据生成管线中 Emotion 准确率仅 89%,这个噪声水平对训练的影响未被量化; (4) 未与 MiDashengLM (非结构化 caption 方法) 直接对比,仅在附录中做了 proxy 对比。

## 可复用的 idea

1. **结构化 JSON 作为密集监督信号**: 不改架构,仅通过设计更丰富的训练标签提升模型能力。可迁移到 TTS 评估(用结构化评估 schema 替代单一 MOS)、语音情感分析等领域。

2. **训练-推理解耦范式**: 训练时用高信息密度的 JSON 格式强制模型学习全面表征,推理时用 task-specific prompt 控制输出格式。这种设计可用于任何需要"内部表征丰富但输出灵活"的场景。

3. **三阶段自动数据生成管线**: 用 captioner + LLM 合成器 + 多层验证器的组合将非结构化信息转化为结构化标注,可迁移到其他需要大规模结构化标注的场景。

4. **感知-推理独立性作为诊断工具**: UAS 的消融实验提供了一套诊断 AudioLLM 瓶颈的方法论——如果增加感知数据不影响推理(反之亦然),说明两者是独立因素,可以分别优化。

5. **UAS-QA 的"知识→能力"桥接**: 仅有 UAS annotation 不够(+6.3%),加上 QA 训练才能充分释放价值(+15.0%)。这暗示结构化知识需要配套的 task-oriented 训练才能转化为 downstream performance。
