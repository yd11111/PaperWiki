---
type: paper
tier: deep
title: "OV-InstructTTS: Towards Open-Vocabulary Instruct Text-to-Speech"
arxiv_id: "2601.01459"
source: "Sources/OV-InstructTTS.pdf"
authors: [Yong Ren, Jiangyan Yi, Jianhua Tao, Haiyang Sun, Zhengqi Wen, Hao Gu, Le Xu, Ye Bai]
year: 2026
venue: "arXiv"
tags: [TTS, InstructTTS, open-vocabulary, reasoning, LALM, emotion, paralinguistic, dataset]
concepts: ["[[Instruction-GuidedSpeechSynthesis]]", "[[NaturalLanguageDescriptionforTTS]]", "[[EmotionControlinTTS]]", "[[LLM-basedTTS]]", "[[ConditionalFlowMatching]]"]
models: ["[[CosyVoice2]]"]
tasks: ["[[InstructedSpeechGeneration]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[InstructedSpeechGeneration]]✓, [[LLM-basedTTS]]✓, [[CosyVoice2]]✓ + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Instruction-GuidedSpeechSynthesis]], [[NaturalLanguageDescriptionforTTS]], [[InstructedSpeechGeneration]], [[EmotionControlinTTS]], [[LLM-basedTTS]], [[CosyVoice2]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文处于 InstructTTS 演进线的前沿位置。已有演进: Style tagging → Reference prompt → NL description (PromptTTS, 2023) → Instruction-guided (VoxInstruct, 2024) → Multi-step editing (InstructSpeech, 2024) → Omni-modal agent (Step-Audio, 2025)。现有 InstructTTS 方法 (PromptTTS, InstructTTS, PromptTTS2, TextrolSpeech, SpeechCraft, VoxInstruct) 本质上依赖预定义声学标签的排列组合或复述 [待确认: Instruction-GuidedSpeechSynthesis],无法处理高层级开放式指令。OV-InstructTTS 试图突破这一瓶颈,将指令从"受限属性描述"推向"开放词汇叙事指令"。

**已有认知**: InstructedSpeechGeneration 任务页✓记录了当前 SOTA (CosyVoice 3 Style SIM 81.06) 和评估基准 (InstructTTSEval)。LLM-basedTTS✓记录了从 VALL-E 到 Hybrid (CosyVoice) 的架构演进,本文使用的 Step-Audio-2-mini-Base 属于 LALM (Large Audio Language Model) 分支。CosyVoice2✓是本文 baseline 之一,以流式合成和 FSQ-SenseVoice tokenizer 见长。EmotionControlinTTS [待确认] 覆盖了从 emotion embedding 到 training-free steering 的多条路线,本文通过 reasoning chain 推断情感标签是一种新的间接控制方式。

**创新判断**: 与 VoxInstruct (统一 content+style 为指令) 的区别 — OV-InstructTTS 进一步将指令从预定义属性组合扩展为源自叙事上下文的开放词汇描述,并引入 reasoning chain 桥接高层指令与低层声学。与 EmoVoice (LLM freestyle text prompting) 的区别 — EmoVoice 仍以情感为中心,OV-InstructTTS 覆盖情感+声学+副语言的全维度推理。

## 速查

> [!summary] 速查
> - **一句话**: 提出 open-vocabulary InstructTTS 范式,通过 LALM reasoning chain 将高层叙事指令映射到情感/声学/副语言属性,再生成语音
> - **路线**: 开放词汇指令 + 文本 → LALM (Step-Audio-2-mini-Base) → <think> reasoning chain (情感标签+声学描述+副语言标签) → interleaved text+audio tokens → Flow Matching + HiFiGAN → 语音
> - **指标**: Gemini Score 70.42 (best), MOS 4.28 (超越 GroundTruth 4.10), ICMOS 3.91 (best), CER 3.61%, SIM 0.722 [Table 2]
> - **可借鉴**: (1) 从小说叙事上下文生成 open-vocabulary 指令的数据构造 pipeline; (2) reasoning chain 作为中间表示桥接高层指令与低层声学; (3) PC-PTI 两阶段方法插入副语言标签且几乎不破坏原始转录 (S-CER 0.35%)
> - **局限**: 仅基于有声书数据 (ContextSpeech, 476.8h),泛化到真实对话/播客场景未验证; 依赖 Step-Audio-2-mini-Base 预训练模型; 评估以 LLM-as-judge (Gemini) 为主,客观指标有限; 数据集和 demo 公开但模型权重未明确开源

## 核心问题

本文要解决的核心矛盾: 现有 InstructTTS 系统的指令空间被限制在预定义声学属性 (pitch, speaking rate, emotion) 的排列组合或复述中,无法处理用户用高层叙事语言 (如"像一个暴怒的王子在逼问下属") 表达的合成意图。这导致内容创作者仍缺乏直觉方式将表达意图传达给 TTS 模型 [§1, 引用 SpeakEasy]。

具体挑战分为两层:
1. **数据层**: 不存在将语音与 open-vocabulary 指令配对的数据集 [§1]
2. **方法层**: 高层开放指令与低层声学特征之间存在显著语义鸿沟 (semantic gap),直接映射无法奏效 [§3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

OV-InstructTTS-TEP (Thinking-Emotion-Paralanguage) 是一个 reasoning-driven 框架,将 open-vocabulary InstructTTS 重构为两步过程 [§3]:

1. **Thinking Token Generation**: 模型生成文本推理链 (`<think>` block),将开放指令分解为情感标签、声学描述和副语言标签
2. **Interleaved Text-Audio Token Generation**: 以推理链为条件,生成增强转录 (含情感标签+副语言标签) 与离散音频 token 的交错序列,最终通过 Flow Matching + HiFiGAN vocoder 合成波形

backbone 为 Step-Audio-2-mini-Base [§3],在 OV-Speech 数据集上做 SFT。

### 关键设计选择

**为什么用 reasoning chain 而不是直接映射?** [论文原文] 高层开放指令与低层声学之间存在显著语义鸿沟,reasoning chain 作为中间表示将映射分解为两个更简单的步骤: 先从指令推断声学属性,再基于属性生成语音 [§3]。消融实验 (Table 3) 验证了这一设计: 加入 thinking (对比 d→f) Gemini Score 从 67.70 提升到 68.71, ICMOS 从 3.74 提升到 3.90 [§4.4]。

**为什么选择 LALM (Step-Audio-2-mini-Base) 作为 backbone?** [论文原文] LALM 具备统一的理解、推理和生成能力,其推理能力使其能从开放指令中推断恰当的声音风格 [§3]。[agent 解读] 相比纯 TTS 模型 (如 CosyVoice2),LALM 的文本理解能力天然适合处理复杂叙事指令,且其 interleaved text-audio 生成能力避免了额外的 text-to-attribute 模块。

**为什么 reasoning + enriched transcription 组合 > 单独使用?** [论文原文] 单独预测带情感标签和副语言标签的增强转录 (variant e) 反而使 Gemini Score 从 67.70 下降到 66.98,作者解释这可能是因为缺乏 reasoning 导致情感和标签预测不准确 [§4.4]。而组合使用 (variant g) 达到最佳 (71.57),说明 reasoning 使模型能从开放指令中推断细粒度属性,使增强转录更具表现力且与指令一致 [§4.4]。

**为什么用 PC-PTI (两阶段) 方法标注副语言?** [论文原文] 三种策略对比中 (Table 1), PASR 直接从音频预测会严重破坏原始转录 (S-CER 18.91%), PRI 虽然定位略好但仍有 S-CER 4.04%; PC-PTI 先分类再插入,S-CER 仅 0.35%,几乎完美保留原始文本完整性 [§2]。[agent 解读] 保留 ground-truth 文本的完整性对 TTS 训练至关重要,因为任何转录错误都会直接导致合成语音内容错误。

### OV-Speech 数据集构造

基于 ContextSpeech 语料 (476.8h 多说话人有声书 + 对应小说文本),五阶段 pipeline [§2]:

1. **上下文信息提取**: 对齐每段音频与源小说,提取前后各 1000 词的上下文窗口,用 Qwen3-32B 蒸馏为 5 种结构化元素 (环境描述/当前事件/说话人性格/对话者状态/说话人意图)
2. **开放词汇指令生成**: 随机选择 2-5 个上下文元素,用 Qwen3-32B 合成类似导演给配音演员的指令 — 两步随机化 (元素选择 + 创意生成) 确保多样性
3. **一致性过滤**: LLM-as-Judge (DeepSeek-R1 预测 + Qwen3-32B 评分),情感 < 6 或声学 < 5 的样本丢弃
4. **推理过程标注**: 为每个通过过滤的样本用 Qwen3-32B 生成两阶段推理链 — Instruction Deconstruction (识别关键上下文元素) → Attribute Inference (推断情感标签+声学描述)
5. **副语言感知转录标注**: 微调 Qwen2-Audio-7B 在 NVSpeech170k 上,用 PC-PTI 方法插入 18 种副语言标签

训练集: 316,807 utterances (来自 83 部小说),每个 utterance 配 3 条不同的 open-vocabulary 指令 [§4.1]。
测试集: 1,500 utterances (来自 3 部 held-out 小说,确保说话人和叙事上下文在训练中未见) [§4.1]。

### 训练策略

在 Step-Audio-2-mini-Base 上做 SFT [§4.1]:
- 学习率: 1 × 10⁻⁵
- 全局 batch size: 32
- 硬件: 8 × NVIDIA A100 (80GB)

## 实验

| 指标 | OV-InstructTTS-TEP | CosyVoice2 (No-Instruct) | GPT4o⋄ | Higgs Audio V2⋄ | Step-Audio-2-mini | GroundTruth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini Score↑ | **70.42** | 66.99 | 68.31 | 65.10 | 67.59 | 75.43 | OV-Speech test | [Table 2] |
| Gemini Rank↓ | **3.39/6** | 3.59/6 | 3.48/6 | 3.73/6 | 3.56/6 | 2.94/6 | OV-Speech test | [Table 2] |
| CER(%)↓ | 3.61 | **3.09** | 3.89 | 8.42 | 5.49 | 3.10 | OV-Speech test | [Table 2] |
| SIM↑ | **0.722** | 0.659 | 0.701 | 0.707 | 0.701 | — | OV-Speech test | [Table 2] |
| MOS↑ | **4.28±0.14** | 3.84±0.19 | 3.23±0.24 | 3.81±0.20 | 3.53±0.24 | 4.10±0.14 | 主观评估 | [Table 2] |
| ICMOS↑ | **3.91±0.17** | 2.94±0.23 | 2.42±0.23 | 3.00±0.20 | 2.40±0.21 | 4.33±0.15 | 主观评估 | [Table 2] |

注: ⋄ 表示说话人音色已通过 CosyVoice2 的 tokenizer 和 flow matching 转换为目标说话人 [Table 2 注释]。

**消融实验** (Table 3, 8-way comparison):

| 变体 | Gemini Score↑ | CER(%)↓ | MOS↑ | ICMOS↑ | 出处 |
| --- | --- | --- | --- | --- | --- |
| (a) Step-Audio-2-mini (No-Instruct) | 61.49 | 8.06 | 3.70 | 2.57 | [Table 3] |
| (b) Step-Audio-2-mini | 63.18 | 5.49 | 3.53 | 2.40 | [Table 3] |
| (c) TTS (No-Instruct, SFT on OV-Speech) | 66.34 | 3.78 | 4.15 | 3.61 | [Table 3] |
| (d) OV-InstructTTS | 67.70 | 3.56 | 4.23 | 3.74 | [Table 3] |
| (e) OV-InstructTTS-EP | 66.98 | 3.65 | 4.27 | 3.81 | [Table 3] |
| (f) OV-InstructTTS-T | 68.71 | 3.45 | 4.27 | 3.90 | [Table 3] |
| **(g) OV-InstructTTS-TEP** | **71.57** | 3.61 | **4.28** | **3.91** | [Table 3] |

关键消融发现:
- **OV-Speech 数据集价值**: (a→c) 无指令 SFT 后 Gemini Score +4.85; (b→d) 有指令 SFT 后 +4.52; MOS/ICMOS 同步大幅提升 [§4.4]
- **Reasoning 的贡献**: (d→f) 引入 thinking 后 Gemini Score +1.01, ICMOS +0.16 [§4.4]
- **EP 单独使用反而下降**: (d→e) Gemini Score 从 67.70 降至 66.98,作者归因于无 reasoning 时情感/标签预测不准 [§4.4]
- **TEP 组合的协同效应**: (f→g) reasoning + enriched transcription 组合达到最佳 Gemini Score 71.57, 超越任一单独组件 [§4.4]

## 局限性

1. **数据域偏窄**: 仅基于有声书数据 (ContextSpeech),指令风格偏叙事/文学,对真实对话、播客、广告等场景的泛化能力未验证 [agent 解读]
2. **评估依赖 LLM-as-judge**: 主要指标 (Gemini Score/Rank) 依赖 Gemini 评估,存在 AI 评估偏差 (InstructTTSEval 已指出 Gemini 存在 self-preference bias); 人工评估仅 8 名评估者 [§4.2]
3. **Backbone 依赖**: 强烈依赖 Step-Audio-2-mini-Base 的预训练能力,是否能迁移到其他 LALM/LLM 架构未知 [agent 解读]
4. **推理延迟**: reasoning chain 引入额外的 text token 生成步骤,实时性指标未报告 [agent 解读]
5. **CER 非最优**: OV-InstructTTS-TEP 的 CER (3.61%) 虽然具有竞争力,但不如 CosyVoice2 (3.09%),说明指令遵循与内容准确性之间可能存在 trade-off [Table 2]
6. **一致性过滤的数据损失**: 数据 pipeline 中情感 < 6 或声学 < 5 的样本被丢弃,过滤损失率未报告 [agent 解读]

## 点评

**亮点**: 本文最核心的贡献是将 InstructTTS 的指令空间从"声学属性的组合/复述"推向"源自叙事上下文的开放词汇指令",这是 InstructTTS 演进中的一个合理方向推进。数据构造 pipeline 的五阶段设计 (特别是从小说上下文提取结构化元素再合成指令) 具有方法论价值,值得其他数据集构造工作借鉴。reasoning chain 的引入是自然且有效的 — 消融实验清晰地证明了 thinking 和 enriched transcription 的协同效应,且单独使用 EP 反而下降的发现是一个有价值的负面结果。

**不足**: 论文的"open-vocabulary"claim 需要审慎看待 — 所有指令都源自同一类型的数据 (有声书小说),语言风格和表达模式有内在的同质性。真正的 open-vocabulary 应能处理从"用播客主播的调侃语气"到"像新闻主播一样中立"等跨域指令。此外,与 InstructTTSEval 提出的 Role-Play Instruct 任务有交集但未明确对比,定位不够清晰。MOS 超越 GroundTruth (4.28 vs 4.10) 需要注意 — 这可能反映的是有声书录音质量参差,而非模型真的超越人类表现。

**在 KB 中的定位**: 在 Instruction-GuidedSpeechSynthesis 演进线中,本文位于 VoxInstruct (统一指令, 2024) → OV-InstructTTS (开放词汇 + reasoning, 2026) 这一步,核心推进是指令空间的开放化和 reasoning 桥接。在 EmotionControlinTTS 维度,本文通过 reasoning chain 间接推断情感,是一条不同于 emotion embedding / DPO / activation steering 的路线。

## 可复用的 idea

1. **从叙事上下文构造开放词汇指令的 pipeline**: 将音频对齐到源文本 → 提取结构化上下文元素 → 随机组合 + LLM 创意合成 → 一致性过滤。这个 pipeline 可迁移到任何有文本上下文的语音数据 (播客转录、影视字幕等)
2. **Reasoning chain 作为高层指令到低层声学的桥梁**: 不直接从复杂指令生成语音,而是先通过 `<think>` 块推断中间属性 (情感+声学+副语言),再基于明确属性生成。这个范式可应用于其他多模态生成中指令与输出之间存在语义鸿沟的场景
3. **PC-PTI 方法插入副语言标签**: 先分类是否存在副语言事件,再定位插入,S-CER 仅 0.35%。这种保护原始文本完整性的两阶段策略可用于任何需要在文本中插入标注的场景
4. **消融中的 EP-without-reasoning 负面结果**: 提醒我们在设计多组件系统时,组件间的依赖关系比单个组件的贡献更重要 — 细粒度标注无 reasoning 支撑反而有害

## 审阅

> [!review] 审阅 (2026-06-05, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 4个WHY因果解释充分,速查可借鉴具体 |
> | 可信赖 | pass | 数字标注覆盖率~90%,指标名正确 |
> | 可区分 | pass | [论文原文]/[agent 解读]标注覆盖率~85% |
> | 可定位 | pass | 完整演进线谱系定位,双基准创新判断 |
> | 不污染 | pass | 仅追加key_papers,不新建概念页 |
> 
> Issues: 2 (high: 0, medium: 0, low: 2)
> 详见 `_review/OV-InstructTTS-review.yml`
