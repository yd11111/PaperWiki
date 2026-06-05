---
type: paper
tier: deep
title: "NV-Bench: Benchmark of Nonverbal Vocalization Synthesis for Expressive Text-to-Speech Generation"
arxiv_id: "2603.15352"
source: "Sources/NV-Bench.pdf"
authors: [Qinke Ni, Huan Liao, Dekun Chen, Yuxiang Wang, Zhizheng Wu]
year: 2026
venue: "arXiv"
tags: [benchmark, evaluation, nonverbal-vocalizations, paralinguistic, TTS, ASR, controllability, expressive-speech]
concepts: ["[[TTSEvaluation]]", "[[ProsodyModeling]]", "[[EmotionControlinTTS]]", "[[Instruction-GuidedSpeechSynthesis]]", "[[AudioUnderstanding]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/CosyVoice3|CosyVoice 3]]", "[[模型库/SenseVoice|SenseVoice]]", "[[论文笔记/FlexiVoice|FlexiVoice]]", "Orpheus-TTS"]
tasks: []
datasets: ["NV-Bench", "Emilia-NV", "SMIIP-NV", "NVTTS", "DisfluencySpeech", "NVS", "SynParaSpeech"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页: [[ProsodyModeling]]✓, [[模型库/CosyVoice2|CosyVoice 2]]✓, [[TTSEvaluation]], [[Instruction-GuidedSpeechSynthesis]], [[AudioUnderstanding]], [[EmotionControlinTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: NV-Bench 位于 TTS 评估体系中"副语言发声 (paralinguistic vocalization) 可控性评估"这一空白区域。知识库中 [[TTSEvaluation]] [待确认] 已梳理了从 WER/SIM/MOS 到 TTSDS2/GSRM/InstructTTSEval 的评估演进脉络,但这些方法要么评估全局语音质量 (TTSDS2/DNSMOS),要么评估指令遵循的声学属性 (InstructTTSEval 的 paralinguistic 12 维),均未针对离散非语言事件 (笑/咳嗽/叹气等) 的生成准确性建立标准化评估。[[ProsodyModeling]]✓ 中 NVSpeech 一节已定义 18 类 word-level PV 分类体系并训练 paralinguistic-aware ASR,NV-Bench 可视为这一工作在评估端的延伸 — 从"能生成 NV"推进到"能可靠评估 NV 生成质量"。
>
> **已有认知**: CosyVoice 2/3 是 NV-Bench 的核心 baseline 模型; CosyVoice 2 已被大量后续工作作为 NV 生成 backbone (见 NonverbalTTS 用 CosyVoice 微调); SenseVoice-Small 是 NV-Bench NVASR 评估器的基座模型。[[EmotionControlinTTS]] [待确认] 记录了 NVSpeech 从具体副语言行为 (而非抽象情感状态) 切入情感表达的路线,NV-Bench 的 Affect Bursts 类别 (Laughter/Surprise 等) 正是这条路线需要评估的目标。
>
> **创新判断**: 知识库中尚无 NV 专项 benchmark。InstructTTSEval 评估指令遵循但用 True/False 二分判断; NV-Bench 引入 PCER (paralinguistic character error rate) 实现更精细的 NV 事件对齐度量,并结合分布级声学保真度 (FAD/FD) 构成双维评估,是该子领域的首个标准化框架。
>
> 检索命中: [[TTSEvaluation]], [[ProsodyModeling]], [[模型库/CosyVoice2|CosyVoice 2]], [[Instruction-GuidedSpeechSynthesis]], [[AudioUnderstanding]], [[EmotionControlinTTS]] | 过滤: 全部 active | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个基于功能分类学的 NV-capable TTS 标准化 benchmark,含 1,651 条 paired GT 多语言数据和双维评估协议 (指令对齐 + 声学保真度)
> - **路线**: 原始音频 → Emilia-Pipeline+MiMo-Audio 过滤 → NVASR 标注 → 人工验证 → 14 类 NV 平衡测试集; 评估: 文本 prompt → TTS 合成 → NVASR 转写 → CER/PCER/OCER (对齐) + FAD/FD/SIM/DNSMOS (声学)
> - **指标**: NVASR CER 1.29% (SMIIP-NV) [Table 3]; NV-CV3 PCER 27.69% (ZH single) [Table 4]; NV-FlexiVoice FAD 0.29 / FD 2.72 [Table 5]; NV-CV3 NMOS 4.08 / IMOS 3.95 [Table 5]; IMOS vs PCER Spearman rho=-0.65 (p<0.001) [§4.2.4]
> - **可借鉴**: (1) PCER 指标设计 — 从 CER 中隔离 NV symbol 的编辑距离,可迁移到任何带特殊标记的 TTS 评估; (2) "功能分类学"替代"声学分类",使 NV 评估从 event detection 升维到 pragmatic appropriateness; (3) 用 MiMo-Audio-7B 做 single-speaker verification 过滤多说话人残留,比传统 diarization 更鲁棒
> - **局限**: (1) 1,651 条规模偏小,每类 50 条 single-label 的统计功效有限; (2) 仅中英双语; (3) 未评估 NV 的 pragmatic appropriateness (上下文适当性),仅评估是否生成了正确的 NV 类型; (4) NVASR 评估器本身有误差,CER 1.29% 并非零; (5) 未开源评估器权重

## 核心问题

NV-capable TTS 系统日益增多,但评估缺乏标准化: (1) 无统一的 NV 分类法导致跨系统比较不可行; (2) 现有评估依赖内部测试集或文本重写参考,缺少真实 paired GT 音频; (3) NV 的长尾分布使不平衡测试集偏倚聚合指标; (4) 缺少能区分"未生成目标 NV"和"生成了但质量差"两种失败模式的评估框架 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

NV-Bench 由三个组件构成 [Fig 1]:

1. **Multi-lingual NVASR**: 基于 SenseVoice-Small 微调的 NV-aware ASR 模型,作为自动评估的骨干工具
2. **Benchmark 数据集**: 1,651 条多语言 (中/英) in-the-wild 语音,含 paired 人类 GT 音频,按 14 类 NV 分为 single-label (严格平衡) 和 multi-label (相对平衡) 两个子集
3. **双维评估协议**: Instruction Alignment (指令对齐) + Acoustic Fidelity (声学保真度)

### 关键设计选择

**为什么采用 Batliner et al. 的功能分类学?** [论文原文] 作者认为现有方法将 NV 视为"声学纹理"(acoustic artifacts),忽略了 NV 本质上是"交流行为"(communicative acts),传递生理状态、情感和交互意图 [§1]。功能分类学将 NV 分为三个语用层级 [§1]:
- **Vegetative sounds** (生理反射): Breathing, Cough, Sigh — 锚定物理真实感
- **Affect bursts** (情感爆发): Laughter, Surprise-ah/oh, Dissatisfaction-hnn — 传达瞬时情感
- **Conversational grunts** (对话管理): Uhm, Confirmation-en, Question-ei/ah/en/oh — 消歧交互意图

[agent 解读] 这一分类法使评估从 acoustic event detection 升维到 pragmatic function assessment。中文有 13 类 (3+4+6),英文有 7 类 (3+2+2),两语言并集共 14 个唯一 NV 类型 [Table 2]。英文缺少 Surprise-ah, Dissatisfaction-hnn, Confirmation-en, Question-ei/ah/en/oh 等中文特有的细粒度语用类别,反映了不同语言的副语言差异。

**为什么需要自训 NVASR 而非用现成 ASR?** [论文原文] 通用 ASR 无法识别 NV 标签; Qwen2.5-Omni 虽经 NV 微调但在 NVTTS 上 OCER 高达 26.95% [Table 3],不足以作为可靠评估器。作者选择 SenseVoice-Small 为基座 [§2.1.1],因其多任务预训练 (ASR + AED + SER + LID) 使其已具备丰富声学特征捕获能力 [论文原文]。

**为什么用 CTC loss 而非 attention-based?** [agent 解读] CTC 的单调对齐假设天然适合 NV 事件的检测 — NV 标签在文本中的位置与音频中严格对应,不需要复杂的交叉注意力对齐。

**标签归一化**: 将异构数据源 (Emilia-NV/NVTTS/DisfluencySpeech 等) 的非语音标签统一映射到 AudioSet Ontology Level 3 [§2.1.2]。关键操作: 对英文子集 (NVTTS, DisfluencySpeech) 进行靶向人工标注,补充"[Question-huh]"等之前英文数据集中缺失的细粒度语用类别 [论文原文]。

**数据集构建**: 从 565,316 条 (~1,560 小时) 2025 年上传的网络音频中筛选 [§2.2.1],选择最近一年数据以最小化 data leakage [论文原文]。三级过滤:
1. **Emilia-Pipeline**: 音频标准化 + source separation + diarization [§2.2.2]
2. **MiMo-Audio-7B-Instruct**: 用 SOTA Audio LLM 检测残留的多说话人片段 [§2.2.2]
3. **人工验证**: 10 名标注员校正 NVASR 转写,5% 交叉标注 Cohen's kappa > 0.85 [§2.2.2]

[agent 解读] MiMo-Audio 用于 single-speaker verification 是一个巧妙的工程选择: 传统 diarization 对轻微背景说话人残留不敏感,而 Audio LLM 可以通过自然语言 prompt 灵活检测"是否有超过一个说话人"。

**测试集设计**: [§3]
- **Single-label** (1,000 条): 每类 50 条,严格平衡; 650 ZH + 350 EN; 隔离基础生成能力
- **Multi-label** (651 条): 每条含 2+ NV 事件; ZH 41-91条/类, EN 75-112条/类; 测试密集副语言条件下的鲁棒性

### 评估协议

**Dimension 1: Instruction Alignment** [§3, §4.2.3]

用 NVASR 转写 TTS 合成语音,与 target text 对比:
- **CER**: 标准字符错误率 (仅文本字符)
- **OCER** (Overall CER): 扩展 CER 到包含 NV 标签的全序列 — `(S+D+I)/(N_text + N_nvv)` [Eq. 2]
- **PCER** (Paralinguistic CER): 仅在提取的 NV 符号上计算编辑距离 — `(S_nvv+D_nvv+I_nvv)/N_nvv` [§4.2.3]

[agent 解读] PCER 是本文最重要的指标创新: 它隔离了 NV 生成准确性,使得"文本说对了但 NV 没生成"和"NV 生成了但文本出错"可以被独立诊断。这解决了 CER 中 NV 信号被大量文本字符稀释的问题。

**Dimension 2: Acoustic Fidelity** [§4.2.3]

- **DNSMOS**: 感知质量预测
- **SIM**: WavLM-based 说话人相似度
- **FAD** (Fréchet Audio Distance): 全局声学分布距离
- **FD** (Fréchet Distance from PANNs): 基于音频事件特征的分布距离

**Human Evaluation** [§4.2.3]: 10 名标注员对每模型 100 条评 5 分制:
- **NMOS** (Naturalness): 声学保真度 + text-NV 韵律连续性 + 说话人一致性
- **IMOS** (Instruction Accuracy): NV 执行精确度 (无遗漏/幻觉/错误发音)

### 训练策略

**NVASR 训练**: 整合 6 个数据源 (Emilia-NV, NVTTS, DisfluencySpeech, NVS, SMIIP-NV, MNV-17) 微调 SenseVoice-Small,使用 CTC loss [Eq. 1] [§2.1]。

**TTS Baselines**:
- **NV-CV3**: 在 Emilia-NV + SMIIP-NV + NVTTS + Disfluency + NVS 合并语料上微调 CosyVoice 3 (0.5B), AdamW lr=1e-5, 4x A800 [§4.2.2]
- **NV-FlexiVoice**: 在同一合并语料上微调 FlexiVoice (0.5B), 原模型在 Emilia (无 NV) 上预训练 [§4.2.2]

## 实验

| 指标 | NV-CV3 | NV-FlexiVoice | CosyVoice3 | Emilia-NV-CV2 | SMIIP-NV-CV2 | Orpheus-TTS | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCER (%) ZH single | **27.69** | 31.08 | 57.69 | 40.00 | 75.64 | 88.77 | 9.38 | NV-Bench ZH single | [Table 4] |
| OCER (%) ZH single | **4.90** | 8.15 | 5.86 | 6.64 | 11.34 | 13.91 | 4.07 | NV-Bench ZH single | [Table 4] |
| PCER (%) ZH multi | **30.04** | 39.37 | 61.94 | 48.74 | 77.20 | 84.85 | 23.71 | NV-Bench ZH multi | [Table 4] |
| SIM ZH single | **0.768** | 0.748 | 0.764 | 0.740 | 0.719 | — | 0.781 | NV-Bench ZH single | [Table 4] |
| DNSMOS ZH single | 3.29 | 3.22 | 3.30 | 3.21 | 3.22 | **3.43** | 3.12 | NV-Bench ZH single | [Table 4] |
| PCER (%) EN single | **46.13** | 50.43 | 62.75 | 55.30 | 56.80 | 71.92 | 8.31 | NV-Bench EN single | [Table 4] |
| FAD | 0.86 | **0.29** | 0.90 | 1.08 | 1.32 | 5.71 | — | NV-Bench full | [Table 5] |
| FD (PANNs) | 3.94 | **2.72** | 9.46 | 5.57 | 6.71 | 24.49 | — | NV-Bench full | [Table 5] |
| NMOS (human) | **4.08** | 4.00 | 3.99 | 3.28 | 3.53 | 3.27 | 4.39 | NV-Bench full | [Table 5] |
| IMOS (human) | **3.95** | 3.94 | 3.56 | 3.89 | 3.28 | 3.27 | 4.39 | NV-Bench full | [Table 5] |
| NVASR CER (%) | — | — | — | — | — | — | **1.29** (NVASR) vs 3.59 (Qwen2.5-Omni) | SMIIP-NV | [Table 3] |
| NVASR OCER (%) | — | — | — | — | — | — | **1.36** (NVASR) vs 4.17 (Qwen2.5-Omni) | SMIIP-NV | [Table 3] |

**NVASR 作为评估器的可靠性** [Table 3]:
- SMIIP-NV: CER 1.29%, OCER 1.36% (大幅优于 Qwen2.5-Omni: CER 3.59%, OCER 4.17%)
- WenetSpeech test-net: CER 5.55% (接近原始 SenseVoice 5.77%)
- [agent 解读] NVASR 在保持标准 ASR 性能的同时显著提升 NV 检测能力,验证了其作为自动评估骨干的可靠性

**关键发现**:

1. **NV-CV3 在指令对齐维度最优** [§4.2.4]: PCER 27.69% (ZH single),归因于大规模多样化训练数据 [论文原文]。但 GT 的 PCER 也有 9.38%,说明 NVASR 评估器本身存在系统性漏检 [agent 解读]

2. **NV-FlexiVoice 在分布距离维度最优** [§4.2.4]: FAD 0.29, FD 2.72,表明生成语音和 NV 事件最接近真实分布 [论文原文]。[agent 解读] FlexiVoice 在 Emilia (无 NV) 上预训练后再 NV 微调,可能因预训练阶段学到了更好的声学分布先验

3. **Orpheus-TTS DNSMOS 最高但 PCER 最差**: DNSMOS 3.43 但 PCER 88.77% [Table 4]。[agent 解读] 这验证了双维评估的必要性 — 单一质量指标无法反映 NV 可控性

4. **主客观一致性**: IMOS 与 PCER 有显著负相关 (Spearman rho=-0.65, p<0.001), NMOS 与 FD 对齐 [§4.2.4],验证了 NV-Bench 客观指标的可靠性

5. **中英差距**: 英文 PCER 普遍高于中文 (NV-CV3: EN 46.13% vs ZH 27.69%) [Table 4]。[agent 解读] 可能与英文 NV 训练数据量较少和英文 NV 类别较少 (8 vs 14) 有关

6. **Multi-label 比 single-label 更难**: 所有模型在 multi-label 上 PCER 均上升 (NV-CV3: 27.69% → 30.04% ZH) [Table 4],验证了密集 NV 场景对可控性的额外挑战

## 局限性

1. **数据规模有限**: 1,651 条 (7.9 小时),每类仅 50 条 single-label,统计功效受限;与 InstructTTSEval (6K 条) 相比规模偏小 [agent 解读]

2. **仅中英双语**: 未覆盖日/韩/法/德等语言的 NV 差异 [agent 解读]

3. **评估停留在 event detection 层面**: 论文开篇强调 NV 是"communicative acts"而非"acoustic artifacts",但评估协议 (PCER) 实质仍在检测"是否生成了正确类型的 NV",未评估"NV 在上下文中是否语用适当"(pragmatic appropriateness) [agent 解读]

4. **NVASR 评估器的系统误差**: GT 音频的 PCER 为 9.38%/8.31% (ZH/EN single),OCER 为 4.07%/6.90% [Table 4],说明评估器本身存在不可忽略的漏检,可能系统性低估所有模型的真实 NV 生成能力 [agent 解读]

5. **未发布评估器权重**: NVASR 模型未开源,限制了复现性和社区采用 [agent 解读]

6. **未考虑 NV 的时间精度**: PCER 只度量 NV 是否出现,不度量 NV 在语音中的时间位置精度 (是否在正确的词边界出现) [agent 解读]

## 点评

NV-Bench 填补了 NV-capable TTS 评估领域的空白,其贡献在于: (1) 定义了一个具体且可操作的评估框架; (2) PCER 指标是一个简洁有效的设计; (3) 双维评估成功区分了"不生成 NV"和"NV 质量差"两种失败模式。

但论文与其宣称的"功能分类学"理念之间存在落差。作者批评现有方法将 NV 视为"acoustic artifacts",但 NV-Bench 的评估本身仍是 event detection 粒度 (有没有生成 [Laughter]),未进入 pragmatic function 粒度 (这个 [Laughter] 在此上下文中是否合理)。这使得 benchmark 在哲学层面上没有超越它所批评的范式。

工程层面,MiMo-Audio 用于 single-speaker verification 是一个值得借鉴的方案; 但未开源 NVASR 权重是一个遗憾,与 benchmark 的"标准化"定位有张力。

与 KB 中已有评估工作的关系: NV-Bench 可视为 [[TTSEvaluation]] 体系中 InstructTTSEval (指令遵循) 与 TTSDS2 (分布级) 两条路线在 NV 子领域的交叉产物。PCER 之于 NV 评估,类似于 InstructTTSEval 的 True/False 之于属性控制评估 — 但 PCER 提供了连续的错误率而非二分判断,更适合 NV 这种高频出错的场景。

## 可复用的 idea

1. **PCER 指标设计模式**: 从标准 CER 中隔离特定 token 类型的编辑距离。可迁移到任何 TTS 场景中需要独立评估特殊标记 (情感标签/停顿标记/方言标签等) 生成准确性的需求

2. **Audio LLM 作为数据质量验证器**: 用 MiMo-Audio 做 single-speaker verification,比传统 diarization pipeline 更灵活。可推广为"用 Audio LLM 做任意自定义音频质量检查 (是否有背景音乐/是否有回声/是否有情感表达)"

3. **标签归一化到统一 ontology**: 将多个异构数据集的非标准标签映射到 AudioSet Ontology,使跨数据集训练和评估具备一致性

## 审阅

> [!review] 自动审阅 (2026-06-06)
> **结论:** pass-with-fixes
> **原则:** 复述 8 | 信赖 7 | 区分 9 | 定位 9 | 污染 7
> **Claim 标注率:** 83% (15/18)
> **问题:** 1 high, 2 medium, 2 low
> - ❌ [factual-error] 方法节 agent 解读段: "中文有 14 类,英文有 8 类"应为 13 类和 7 类(14 是双语并集)
> - ⚠️ [traceability-gap] 实验表 FAD 行: NV-CV3 FAD=0.86 被标为"—"(Table 5 有值)
> - ⚠️ [template-compliance] frontmatter models 缺 FlexiVoice、Orpheus-TTS
> - 💡 [template-compliance] SIM 行 bold 标在 0.764(CosyVoice3)但 NV-CV3 为 0.768
> - 💡 [template-compliance] NVASR CER/OCER 两行全空,建议删除或填充
> **反向更新:** 已修正全部 5 项问题,可执行反向更新
