---
type: paper
tier: deep
title: "Position: Towards Responsible Evaluation for Text-to-Speech"
arxiv_id: "2510.06927"
source: "Sources/TTSEvaluationPositionPaper.pdf"
authors: [Yifan Yang, Hui Wang, Bing Han, Shujie Liu, Jinyu Li, Yong Qin, Xie Chen]
year: 2026
venue: "ICML 2026"
tags: [position-paper, TTS-evaluation, MOS, WER, SIM, responsible-AI, standardization, fairness, security, metrics, benchmark]
concepts: ["[[TTSEvaluation]]", "[[SpeakerVerification]]", "[[SpeakerEmbedding]]", "[[ProsodyModeling]]", "[[Anti-spoofingandDeepfakeDetection]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-02
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页: [[SpeakerEmbedding]]✓, [[ProsodyModeling]]✓, [[TTSEvaluation]], [[SpeakerVerification]], [[Anti-spoofingandDeepfakeDetection]], [[SpokenDialogueEvaluation]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[TTSEvaluation]], [[SpeakerVerification]], [[Anti-spoofingandDeepfakeDetection]], [[SpeakerEmbedding]], [[ProsodyModeling]], [[SpokenDialogueEvaluation]] | 过滤: 无 | 未命中但可能相关: 无
>
> **谱系定位**: 本文是 TTS 评估领域首篇系统性 position paper (ICML 2026),提出"Responsible Evaluation"三层框架。KB 中 [[TTSEvaluation]] 页面已记录了该框架及其后续大量工作 (GSRM, SpeechJudge, TTSDS2, TTS-PRISM, InstructTTSEval, UniSRM 等),本文是这些后续工作的共同参考锚点。
>
> **已有认知对照**: KB 中 [[SpeakerEmbedding]] 详述了 SECS 计算的 encoder 依赖性问题 (同一系统不同 encoder 差 0.1-0.3),与本文 SIM 指标批判一致; [[ProsodyModeling]] 记录了 F0 RMSE 仅捕获 pitch 一维的局限 (ProsodyEval DS-WED r=0.77 vs log F0 RMSE 0.30),呼应本文 F0 指标批判; [[Anti-spoofingandDeepfakeDetection]] 梳理了从被动检测到主动防护的安全演进线,与本文 Level 3 安全主张互补; [[SpokenDialogueEvaluation]] 也指出没有单一 benchmark 覆盖所有评估维度,但其 11 维框架侧重对话系统而非 TTS。
>
> **创新判断**: 本文的核心贡献不在技术创新而在框架构建 — 将分散的指标批判和改进建议整合为三层递进结构 (Fidelity → Comparability → Governance),并首次在顶会层面系统性论述 TTS 评估中的公平性和安全性缺失。KB 中已有大量后续工作填补了本文指出的空白 (如 TTSDS2 解决 distributional evaluation, SpeechJudge 解决 naturalness-specific reward, MINT-Bench 解决 instruction-following evaluation),说明本文的问题诊断具有前瞻性。

## 速查

> [!summary] 速查
> - **一句话**: 首篇 TTS 评估 position paper,提出三层 Responsible Evaluation 框架 (Fidelity → Comparability → Governance),系统诊断 MOS/WER/SIM/F0 等指标的结构性缺陷,并呼吁将公平性和安全性纳入标准评估协议
> - **路线**: 历史回顾 (三时代演进) → 指标缺陷诊断 (Level 1) → 可比性/标准化挑战 (Level 2) → 治理/公平/安全 (Level 3) → 每层可操作建议 → Alternative Views
> - **指标**: LibriSpeech test-clean 不同版本致 WER 差异 60% (2.63 vs 4.22) [Appendix A/Table 1]; SIM-o 含/不含 prompt 差异达 0.151 (0.754 vs 0.905) [Appendix B/Table 2]; MOS ceiling 在 4.5+ 饱和 [§3.2]
> - **可借鉴**: (1) 论文评估部分应明确报告 dataset variant/split/size + inference task 定义 + SIM 计算方式; (2) 客观指标改善应结合 uncertainty estimation; (3) 报告 RTF 时需注明硬件/batch size/prompt length/streaming mode
> - **局限**: 以框架和建议为主,不含新指标或新 benchmark 的技术实现; 公平性和安全性部分停留在问题识别层面,缺乏具体评估协议设计; 主要关注英文/中文 TTS,对低资源语言讨论有限

## 核心问题

本文要回答的核心问题是: **当 TTS 系统的合成质量已逼近人类语音时,当前的评估方法论是否仍然适用,如果不适用,应如何系统性地重构?**

论文的回答是: 不适用。当前评估存在三层结构性缺陷 — (1) 指标本身不忠实于真实能力 (fidelity 问题), (2) 跨系统比较缺乏科学严谨性 (comparability 问题), (3) 完全忽视伦理和社会影响 (governance 问题)。这三层需要递进式解决。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构: 三层 Responsible Evaluation 框架

本文不是一个技术方法论文,而是一个分析框架。其核心结构是三层递进的 Responsible Evaluation 概念 [§1]:

| 层次 | 名称 | 核心主张 | 前提 |
|------|------|---------|------|
| Level 1 | Fidelity & Accuracy | 指标应忠实反映模型真实能力与局限 | — |
| Level 2 | Comparability, Standardization & Transferability | 评估实践应遵循科学严谨性,支持有意义的跨系统比较 | Level 1 成立 |
| Level 3 | Governance, Fairness & Security | 评估应纳入伦理和社会影响考量 | Level 1+2 成立 |

论文作者认为这三层是递进的: 如果指标本身不准确 (Level 1 失败),那么可比性 (Level 2) 和治理 (Level 3) 的讨论都建立在错误基础上 [论文原文, §1]。[agent 解读] 这种递进结构意味着当前 TTS 评估的改革应该从 Level 1 开始,而非直接跳到公平性讨论。

### Level 1: Fidelity & Accuracy — 指标缺陷诊断

#### 客观指标的系统性问题 [§3.1]

**WER (Word Error Rate)** 的三重局限:
1. **ASR 系统噪声**: ASR 模型自身的识别错误 (Hsu et al., 2021; Radford et al., 2023) 导致 WER 不反映合成语音的真实可懂度 — 合成语音对人类而言已足够清晰,但 ASR 仍报错 [论文原文]
2. **非线性感知映射**: WER 关注逐词准确率,忽略关键信息是否被传达 (Tee et al., 2026) [论文原文]
3. **优化陷阱**: 直接以 WER 作为 RL reward 训练会导致韵律坍缩 — 模型为追求转写准确而牺牲韵律多样性和自然度 (Shin et al., 2026) [论文原文]

**SIM (Speaker Embedding Cosine Similarity)** 的局限:
1. **噪声敏感性**: ECAPA-TDNN 等模型对 channel variation、背景噪声、phonetic content 敏感 [论文原文]
2. **目标不对齐**: 训练于 discriminative speaker classification 任务,与 continuous perceptual similarity 不对齐 [论文原文]
3. **阈值饱和**: 超过一定阈值后,SIM 改善不代表感知增益 (Wester et al., 2016) [论文原文]

**Predicted MOS (DNSMOS, UTMOS 等)** 的局限:
1. **领域不匹配**: 如 DNSMOS 训练于 speech enhancement 数据,评估合成语音时存在系统性偏差 [论文原文]
2. **缺乏 uncertainty estimation**: 仅输出点估计,无置信区间,无法判断预测可靠性 [论文原文]
3. **泛化差距**: in-domain vs out-of-domain 表现差异显著 (Wang et al., 2025c; Cooper et al., 2022) [论文原文]

**F0 指标** (log F0 RMSE + DTW):
- 仅捕获 pitch 一个维度,忽略 rhythm, stress, intensity [论文原文]
- 与人类韵律感知判断弱相关 (Yang et al., 2026b) [论文原文]

#### 主观指标 (MOS) 的结构性缺陷 [§3.2]

MOS 的五重问题:
1. **Ceiling effect**: 高质量系统 MOS 趋于饱和 (4.5+),无法区分系统间差异 [论文原文]
2. **Non-transferability**: 不同实验的 MOS 分数不可直接比较,因评估者池、环境、标准不同 (Kirkland et al., 2023) [论文原文]
3. **Listener bias**: 评估者背景、心情、播放条件引入随机噪声 [论文原文]
4. **Cost barrier**: 大规模、多样化听众群难以组织和保证质量 [论文原文]
5. **Protocol inconsistency**: 评分量表定义、rater calibration、评估维度 (naturalness vs overall quality) 报告不全 (Chiang et al., 2023) [论文原文]

#### 评估不足的维度 [§3.3]

论文识别了四个当前评估方法未能覆盖的重要维度:

| 维度 | 挑战 | 现状 |
|------|------|------|
| 数学符号/公式朗读 | 符号发音、运算符 scope、嵌套结构、上下文依赖的阅读惯例 | EmergentTTS-Eval (Manku et al., 2025) 是早期尝试但远不系统 [论文原文] |
| 长篇合成 (audiobook, podcast) | 跨句一致性、篇章级韵律、说话人稳定性 | LibriTTS (Zen et al., 2019) 和 Seed-TTS-eval 聚焦短句 [论文原文] |
| 情感表现力 | 无统一情感分类法、emotion MOS 对细微差异不敏感 | 广泛使用的情感数据集 (IEMOCAP, CREMA-D) 依赖离散标签,覆盖有限 [论文原文] |
| 标点敏感性 | 标点对停顿/重读/语调的影响无定量评估方法 | 完全空白 [论文原文] |

### Level 2: Comparability, Standardization & Transferability

#### 不一致的评估实践 [§4.1]

**数据集不一致**: LibriSpeech test-clean 在不同论文中使用不同版本 [§4.1]:
- VALL-E (Wang et al., 2023): 1234 utterances
- NaturalSpeech 3 (Ju et al., 2024) / MaskGCT (Wang et al., 2024c): 40 utterances
- F5-TTS (Chen et al., 2025): 1127 utterances (含标点和大小写)
- 论文在 Appendix A 用 MaskGCT 实测: 同一模型在 40-utterance 版本 WER 2.63%, 在 1234-utterance 版本 WER 4.22% — 差异 60% [Table 1]

**推理任务不一致**: Zero-shot TTS 的 Continuation 任务定义不统一 [§4.1]:
- VALL-E: 前 3 秒作 prompt
- E2 TTS (Eskimez et al., 2024): 截断后 3 秒作 prompt
- [agent 解读] 这导致不同论文报告的"Continuation"结果实际评估的是不同任务

**SIM 协议不一致**: SIM-o 计算是否包含 prompt 段 [§4.1, Appendix B]:
- VALL-E: 排除 prompt
- VALL-E 2: 包含 prompt
- Appendix B 实测: 差异达 0.151 (0.754 vs 0.905) [Table 2]
- [agent 解读] 包含 prompt 的 SIM-o 天然更高,因为 prompt 部分是 ground truth,这使得报告含 prompt 的系统在数字上看起来更好

**MOS 报告不透明** [§4.2]: 许多论文声称使用 MOS 评估但不报告评分量表定义、rater calibration 流程、是否评估 naturalness 还是 overall quality。

**RTF 报告不透明** [§4.2]: 硬件配置、batch size、prompt 长度、streaming 模式等关键细节缺失,使 RTF 比较无意义。

#### 指标可迁移性困境 [§4.3]

- **SIM 不可迁移**: 需要 reference speech,外部评估者无法获取原始参考音频 [论文原文]
- **MOS 不可迁移**: 不同实验的 MOS 分数无法直接比较,每次新比较都需重新做听感测试 (Kirkland et al., 2023) [论文原文]

#### 建议 [§4.4]

1. 区分 comparable vs incomparable 结果 — 不同数据集/任务/配置的分数不应混用 [论文原文]
2. 遵循标准化协议 (如 ITU-T P.808 for MOS) [论文原文]
3. 透明报告评估细节 (dataset split, prompt list, metric config, listener test procedure, RTF setup) [论文原文]
4. 发展 transferable metrics — LLM-as-a-Judge 路线 (Wang et al., 2025e; 2026a; Zhang et al., 2025b) 可在 shared evaluation conditions 下生成可迁移的评分 [论文原文]

### Level 3: Governance, Fairness & Security

#### 治理: 数据合法性、知情同意、问责 [§5.1]

- 大规模语音数据集 (Emilia, GigaSpeech, WenetSpeech4TTS 等) 从互联网抓取,speaker consent 和 licensing 不清晰 [论文原文]
- 许多技术报告用"in-house data"描述训练数据,不披露来源和许可 [论文原文]
- 建议: 评估报告应要求披露训练数据来源、许可、采集流程 [§5.4]

#### 公平性: 群体差异与表征性伤害 [§5.2]

- 聚合 MOS/WER/SIM 分数可掩盖对 underrepresented 语言/口音群体的质量退化 [论文原文]
- ASR-based 指标继承 ASR 系统的种族偏见 (Koenecke et al., 2020) [论文原文]
- ASV-based 指标继承 speaker verification 系统的偏见 (Hutiri & Ding, 2022) [论文原文]
- 表征性伤害: underrepresented 语音被刻板化 (Ovacik, 2025; Puhach et al., 2025) 或矮化 (Michel et al., 2025) [论文原文]
- 建议: group-disaggregated reporting + representation-aware benchmarks + 多语言 ASV 模型 [§5.4]

#### 安全性: 滥用、欺骗、可追溯性 [§5.3]

- 高保真 voice cloning 降低了电信诈骗和 deepfake 的门槛 [论文原文]
- 合成语音威胁 ASV 生物识别系统 [论文原文]
- 标准评估协议几乎不纳入 traceability 评估 [论文原文]
- 新方向: imperceptible watermarking (Wen et al., 2025; Zhao et al., 2025) + TraceSpeech (Zhou et al., 2024) [论文原文]
- 建议: 将 traceability (是否可检测为合成语音) 纳入标准评估协议 [§5.4]

### 关键设计选择

**为什么选择三层递进而非并列?**
论文作者认为这三层之间存在逻辑依赖 [论文原文, §1]: 如果 Level 1 (指标准确性) 都不满足,Level 2 (可比性) 和 Level 3 (治理) 的讨论就建立在错误基础上。[agent 解读] 这种递进设计也暗示了改革优先级: 应先解决指标忠实性,再推标准化,最后纳入伦理。但实际操作中三层可以并行推进。

**为什么纳入 Alternative Views? [§6]**
论文讨论了两个反对意见并给出回应:
1. *"增加评估维度会增加复杂性"* — 作者回应: 短期阵痛换长期收益,类似其他技术领域从碎片化走向统一标准的过程 [论文原文]
2. *"过度强调法律/伦理会减缓技术进步"* — 作者回应: Fair Use 等制度可以提供灵活性,但不等于可以回避透明性和问责性 [论文原文]

### TTS 技术与评估方法的共演进 [§2, Fig 1]

论文回顾了三个历史阶段,论证评估方法与技术发展之间的 gap 如何逐步扩大:

| 时期 | TTS 技术 | 评估方法 | 评估特征 |
|------|---------|---------|---------|
| 2000s (SPSS) | HMM 参数合成 | MCD, F0 RMSE, informal listening | 基础 (Emergence Phase) |
| 2010s (E2E DL) | Tacotron, WaveNet, VITS, FastSpeech | MOS + WER + SIM 多维化, objective metrics 兴起 | 丰富化 (Enrichment Phase) |
| 2020s+ (Diffusion & FM) | VALL-E, CosyVoice, F5-TTS | CMOS+SMOS+predicted MOS+LLM-as-Judge | 统一化 (Unification Phase),但 gap 加大 |

[agent 解读] Fig 1 展示了 TTS 技术曲线远在评估方法曲线之上,且差距在 2020s 显著扩大。这是论文核心论点的可视化证据: 评估进化跟不上技术进化。

## 实验

本文是 position paper,无传统意义上的实验。但包含两个 case study 作为定量证据:

| 实验 | 数据 | 关键发现 | 出处 |
|------|------|---------|------|
| LibriSpeech test-clean 版本差异 | MaskGCT, HuBERT-Large ASR | 同一模型在 40-utt 版本 WER 2.63%, 1234-utt 版本 WER 4.22% | [Appendix A, Table 1] |
| SIM-o 计算协议差异 | LibriSpeech test-clean, WavLM-TDNN | Without prompt SIM-o 0.754, with prompt SIM-o 0.905, 差异 0.151 | [Appendix B, Table 2] |

[agent 解读] 这两个 case study 虽然简单,但极具说服力: 它们用最少的实验量证明了跨论文比较的系统性风险。仅仅是数据集版本和计算方式的差异,就能产生看似巨大的"性能差距"。

## 局限性

1. **框架导向,缺乏实现**: 论文提出的建议多为方向性 ("should", "we encourage"),但不提供具体的技术实现或 benchmark 设计。例如,如何标准化 SIM 计算、如何设计 representation-aware benchmark、如何实现 group-disaggregated reporting,均留给后续工作 [agent 解读]

2. **公平性和安全性讨论深度有限**: Level 3 的篇幅远少于 Level 1 和 Level 2,关于如何将公平性指标嵌入标准评估 pipeline 缺乏具体方案 [agent 解读]

3. **语言覆盖偏向**: 讨论主要围绕英文和中文 TTS 系统,低资源语言的特殊评估挑战 (如缺乏 ASR/ASV 基础设施) 仅被简略提及 [agent 解读]

4. **时效性与选择性**: 引用截止 ~2025 年中,部分同期工作 (如 SpeechJudge, TTSDS2, UniSRM) 可能未充分覆盖。同时作为 position paper,对已有工作的引用具有选择性 [agent 解读]

5. **缺乏工业视角的验证**: 论文的建议主要面向学术评估实践,但工业界 (如 Apple, Google, Microsoft, 字节跳动) 的内部评估流程是否存在类似问题,以及这些建议在工业环境中的可行性,缺乏讨论 [agent 解读]

## 点评

**作为框架论文的价值**: 本文的最大贡献在于将分散的"大家都知道但没人系统说过"的评估问题整合为一个连贯的三层框架。TTS 领域长期存在"每篇论文都知道 MOS 不好比较,但每篇论文都继续这么做"的困境,本文提供了一个可引用的锚点来推动改变。ICML 2026 position paper track 的发表进一步赋予了其影响力。

**三层递进的逻辑性**: Level 1→2→3 的递进结构在逻辑上是成立的,但在实践中可能过于理想化。例如,数据治理 (Level 3) 的改善不需要等待指标忠实性 (Level 1) 的完全解决。更合理的是将三层视为并行但有优先级的改革方向。

**问题诊断强于解决方案**: 论文在诊断问题方面做得出色 (SIM-o 差异 0.151, LibriSpeech WER 差异 60% 的 case study 极具说服力),但在"怎么办"方面较弱。相比之下,同期的 TTSDS2 (distributional evaluation)、SpeechJudge (naturalness-specific GRM)、TTS-PRISM (multi-dimensional diagnostic) 等工作提供了更具体的技术解决方案。

**后续影响**: 从 KB 中 [[TTSEvaluation]] 页面的演进线来看,本文确实成为了后续一系列评估改进工作的共同参考 — GSRM, SpeechJudge, TTSDS2, TTS-PRISM, InstructTTSEval, MINT-Bench, ProsodyEval, UniSRM 等都在不同维度回应了本文指出的问题。这验证了 position paper 的前瞻性价值。

**与 KB 中的关联**: 本文的 Level 3 (安全性) 与 [[Anti-spoofingandDeepfakeDetection]] 中记录的从"被动检测"到"主动防护"的范式转变相呼应; Level 1 (F0 指标批判) 与 [[ProsodyModeling]] 中 ProsodyEval/DS-WED 的工作高度相关; Level 2 (SIM 不一致) 与 [[SpeakerEmbedding]] 中 SECS encoder 依赖性的分析完全一致。

## 可复用的 idea

1. **评估论文写作清单**: 任何 TTS 论文在撰写评估部分时,应明确报告: (a) 数据集版本和 utterance 数量, (b) 推理任务定义 (如 Continuation 的 prompt 选取方式), (c) SIM 计算是否包含 prompt 段, (d) MOS 使用的评分量表和评估维度, (e) RTF 的硬件环境和推理配置。这可以作为自查清单 [§4]

2. **Uncertainty estimation for predicted MOS**: 当使用 DNSMOS/UTMOS 等自动 MOS 预测时,应报告 uncertainty estimates (如置信区间),特别是在 out-of-domain 条件下。minor 分数差异不应被解读为性能差距 [§3.1]

3. **Transferable evaluation via LLM-as-a-Judge**: 基于 LLM 的评估可以在 shared conditions 下产生跨系统可比的评分,避免了传统 MOS 的不可迁移性问题。这是解决 Level 2 问题的技术路线 [§4.4]

4. **Group-disaggregated reporting**: 在多语言/多口音 TTS 系统评估中,应拆分报告不同群体的指标 (按语言、口音、性别等),而非仅报告聚合指标,以避免公平性问题被平均化掩盖 [§5.2]

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 三层框架因果解释清晰,关键设计选择有 WHY 论述,可借鉴字段具体 |
> | 可信赖 | pass | 关键数字均有出处标注,指标名称准确,position paper 数字型 claim 较少但已覆盖 |
> | 可区分 | pass | 来源标注覆盖率 >90%,[论文原文]/[agent 解读] 边界清晰 |
> | 可定位 | pass | KB 背景有谱系定位 + 后续工作对比基准,frontmatter 完整 |
> | 不污染 | pass | 未执行反向更新 |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/Survey-ResponsibleTTSEvaluation-review.yml`
