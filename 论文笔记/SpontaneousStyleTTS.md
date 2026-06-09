---
type: paper
tier: deep
title: "Spontaneous Style Text-to-Speech Synthesis with Controllable Spontaneous Behaviors Based on Language Models"
arxiv_id: "2407.13509"
source: "Sources/SpontaneousStyleTTS.pdf"
authors: [Weiqin Li, Peiji Yang, Yicheng Zhong, Yixuan Zhou, Zhisheng Wang, Zhiyong Wu, Xixin Wu, Helen Meng]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, spontaneous-speech, language-model, prosody, expressive-speech, VALL-E, behavior-modeling, Mandarin]
concepts: ["[[ProsodyModeling]]", "[[LLM-basedTTS]]", "[[CodecLanguageModel]]"]
models: ["[[EnCodec]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[ProsodyModeling]], [[LLM-basedTTS]], [[EnCodec]]; 1 个待确认实体页: [[CodecLanguageModel]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓, [[EnCodec]]✓ | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: [[StyleTransferinTTS]], [[GlobalStyleTokens]]

**谱系定位**: 本文处于 spontaneous speech synthesis 与 LM-based TTS 的交叉领域。在 Prosody Modeling 演进线上,传统 spontaneous TTS 方法 (AdaSpeech 3, SponTTS) 使用 FastSpeech 2 等非自回归 backbone,在有限数据上训练,韵律自然度和自发行为多样性受限。本文将自发语音合成提升到 LM-based TTS 框架(VALL-E backbone),利用大规模预训练获取语义理解和语音表达多样性,同时显式建模 19 种自发行为和细粒度韵律。

**已有认知**:
- [[ProsodyModeling]] (confirmed) 区分了显式韵律(FastSpeech 2 variance adaptor)和隐式韵律(reference encoder、VAE、生成模型),以及副语言发声(NVSpeech 18 类 PV)。本文的 spontaneous behavior modeling 属于显式标签控制分支,但创新地将标签嵌入与句法信息结合;其 prosody extractor 则属于 reference encoder 分支的细粒度变体。
- [[LLM-basedTTS]] (confirmed) 记录了 VALL-E 的 AR+NAR 两阶段架构及其在 zero-shot voice cloning 上的成功。本文以 VALL-E 为 backbone,但将其从 voice cloning 拓展到 spontaneous style 合成,验证了 LM-based TTS 在自发语音场景下的潜力。
- [[CodecLanguageModel]] [待确认] 梳理了 codec LM 的多层 RVQ 建模挑战。本文使用 EnCodec 编码/解码,沿用 VALL-E 的 AR+NAR 策略处理多层 codec tokens。
- [[EnCodec]] (confirmed) 是本文使用的 neural audio codec,24kHz 采样率。

**创新判断**: 相比 KB 中已有的 NVSpeech(18 类副语言发声,词级控制)和 DiffCSS(diffusion 韵律预测器建模对话韵律多样性),本文的独特贡献在于:(1) 将 19 种自发行为从分散的个别研究(filled pause、breathing、laughter 各自独立)统一到一个系统中;(2) 提出句法感知的行为编码,利用自发行为在句中的句法位置信息;(3) 引入细粒度的自发韵律提取-预测机制,将行为标签与韵律建模耦合。这是已知的 LM-based TTS 框架中最全面的自发行为建模工作。

## 速查

> [!summary] 速查
> - **一句话**: 基于 VALL-E 的自发语音合成系统,通过句法感知的自发行为编码器(19 种行为)和细粒度韵律建模,显著提升自发语音的韵律自然度和行为自然度
> - **路线**: 音素序列 + 自发行为标签 + 句法结构 → 句法感知行为编码器(L-embeddings)→ 加到文本 embeddings; 参考音频 → 韵律提取器(P-embeddings)/ 推理时由 AR 韵律预测器生成 → 加到文本 embeddings → VALL-E AR/NAR decoder → EnCodec 解码 → 语音
> - **指标**: PN-MOS 4.09 / LN-MOS 4.05 (vs VALL-E 3.30/3.32, vs FastSpeech 2 2.61/2.54); MCD 4.879 (vs VALL-E 5.291) [Table 1, 内部自发语料]
> - **可借鉴**: (1) 句法位置信息辅助自发行为建模; (2) 多头注意力将行为标签与韵律表征耦合; (3) 三步渐进式微调策略(backbone → 行为+韵律提取 → 韵律预测+联合精调)
> - **局限**: 仅 5.4h 单人普通话数据; 无公开 MOS 基准可比; 未开源; 自发行为分类体系仅覆盖普通话

## 核心问题

本文要解决的核心问题是:**如何在 LM-based TTS 中有效建模和控制多样化的自发语音行为**。

自发语音(日常对话、脱口秀、播客)与朗读式语音的关键差异在于各种自发行为(filled pause、重复、笑声、感叹词等)和丰富的韵律变化 [§1]。现有研究面临三个瓶颈:(1) 大多数工作只关注个别自发行为(如仅建模 filled pause 或仅建模呼吸),忽略了自发行为的多样性 [§1];(2) 隐式韵律建模(仅从文本预测)难以捕捉自发语音的多样复杂韵律特征 [§1];(3) 数据规模有限,合成语音与真实人声差距明显 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由四个核心模块组成 [§2, Fig 1]:

1. **VALL-E Backbone**: AR+NAR transformer decoder,接受文本编码和音频 prompt 作为条件
2. **Syntactic-aware Spontaneous Behavior Encoder**: 将自发行为标签和句法位置编码为 L-embeddings,加到文本 embeddings 上
3. **NAR Label Predictor**: 从文本 embeddings 预测自发行为标签(推理时使用)
4. **Spontaneous Prosody Modeling**: 由 Prosody Extractor(训练时从音频提取 P-embeddings)和 LM-based Prosody Predictor(推理时预测 P-embeddings)组成

### 关键设计选择

#### 1. 为什么将 19 种自发行为统一建模?

[论文原文] 作者基于语言学特征将自发行为分为三大类 [§2.2]:
- **不流畅(Disfluency)**: filled pause、重复、口吃、拖音
- **感叹词(Interjections)**: 按语用功能分为 9 种(疑问、回应、惊讶、正面反馈、提醒、领悟、叹息、撒娇、嗤之以鼻)
- **非语言声(Non-speech sounds)**: 笑声进一步细分为 6 种(微笑、大笑、苦笑、尴尬笑、嘲笑、不自觉笑)

共计 19 种自发行为,每种在语料中显式标注。

[agent 解读] 相比之前单独处理个别行为的工作(AdaSpeech 3 仅 filled pause,Lameris et al. 仅 creaky phonation),统一建模的优势在于:(1) 共享 backbone 可以学习行为间的共性(如都影响韵律);(2) 组合控制成为可能;(3) 避免了为每种行为设计专用模块的工程开销。这一思路与 NVSpeech 的 18 类副语言发声建模高度一致,但 NVSpeech 通过词表扩展实现,本文通过专用编码器实现。

#### 2. 句法感知的行为编码: 为什么句法位置很重要?

[论文原文] 自发行为的句法位置对应其不同的语用功能 [§2.3]。作者提取每个自发行为标签在字符级的 6 维句法结构信息 [Fig 2]:
- S1: 字在句中的索引 (Index_c,s)
- S2: 字在句中的计数 (Cnt_c,s)
- S3: 字在子句中的索引 (Index_c,sub)
- S4: 字在子句中的计数 (Cnt_c,sub)
- S5: 子句在句中的索引 (Index_sub,s)
- S6: 子句在句中的计数 (Cnt_sub,s)

句法信息经两层线性层+ReLU处理后,与标签嵌入合并,得到句法感知的 L-embeddings [§2.3]。

[agent 解读] 这个设计的直觉是:同一个"嗯"出现在句首(表犹豫)和句中(表思考/回应)有不同的声学表现。句法位置编码为模型提供了区分这些细微差异的显式信号,而非让模型从数据中隐式学习这种位置依赖性。这是本文相对于简单标签嵌入方法的关键创新点。

#### 3. 细粒度韵律建模: 提取器+预测器的解耦设计

**Prosody Extractor** [§2.4.1, Fig 3]:
- 3 层卷积 + 多头注意力,从 mel spectrogram 提取帧级韵律表征
- 第一层卷积压缩 mel 到帧级 hidden states
- 中间层(两层卷积栈 + 平均池化)生成音素级中间表征
- 最终卷积层生成细粒度韵律表征
- 关键: 用多头注意力以 L-embeddings 为 query、韵律表征为 key/value,生成 P-embeddings

[论文原文] 之所以将行为标签与韵律耦合,是因为自发行为显著影响韵律 [§2.4.1]。通过注意力机制,模型学习每个行为标签对韵律的选择偏好。

**LM-based Prosody Predictor** [§2.4.2]:
- 基于 transformer decoder 的自回归预测器
- 输入: 文本、行为标签、句法结构信息
- 输出: 预测的 P-embeddings
- AR 结构可捕捉韵律的局部和全局依赖

[agent 解读] 提取器-预测器的解耦设计是 TTS 中常见的 teacher-student 模式(类似 FastSpeech 的 knowledge distillation)。训练时用 ground-truth 音频提取韵律目标,推理时用预测器从文本预测,避免了推理时对参考音频的依赖。

### 训练策略

三步渐进式微调 [§2.5]:

1. **Step 1 — 联合训练 backbone + 行为模块 + 韵律提取器**:
   - 损失函数: L = L_ce(C_gt, C_predict) + 0.1 * L_ce(L_gt, L_predict) [Eq. 1]
   - 其中 C 为 acoustic tokens,L 为行为标签
   - 冻结 audio codec
   - 训练 30k iterations

2. **Step 2 — 单独训练韵律预测器**:
   - 冻结 Step 1 训练的韵律提取器作为 teacher
   - MSE loss: 预测 P-embeddings 与提取 P-embeddings 之间
   - 训练 20k iterations

3. **Step 3 — 联合精调韵律预测器 + 声学模型**:
   - 冻结韵律提取器,用较低学习率联合训练
   - 目的: 弥补韵律预测误差对声学输出的影响
   - 训练 16k iterations

[agent 解读] 这种三步策略的关键是先建立可靠的韵律目标(提取器),再训练预测器逼近它,最后让声学模型适应预测器的误差。每步都有明确的冻结/解冻策略,避免了端到端训练时不同模块间的梯度干扰。

## 实验

| 指标 | Proposed | Base-L (无韵律) | VALL-E (Baseline) | FastSpeech 2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| PN-MOS ↑ | **4.090 +/- 0.073** | 3.792 +/- 0.084 | 3.302 +/- 0.099 | 2.606 +/- 0.106 | 内部自发语料 | [Table 1] |
| LN-MOS ↑ | **4.046 +/- 0.737** | 3.898 +/- 0.084 | 3.324 +/- 0.098 | 2.536 +/- 0.097 | 内部自发语料 | [Table 1] |
| MCD ↓ | **4.879** | 4.961 | 5.291 | 5.355 | 内部自发语料 | [Table 1] |

### 关键发现

1. **LM-based backbone 显著优于传统 TTS**: VALL-E 在未做任何自发行为建模的情况下,PN-MOS 就已从 2.61 提升到 3.30 (+0.70),MCD 从 5.355 降至 5.291,验证了大规模语言模型在自发语音合成中的优势 [§3.3, Table 1]

2. **显式行为建模的贡献(Base-L vs VALL-E)**: 仅添加句法感知行为编码器(不含韵律模块),PN-MOS 从 3.30 → 3.79 (+0.49),LN-MOS 从 3.32 → 3.90 (+0.58),说明显式标签控制是提升自发行为自然度的关键 [§3.3]

3. **韵律建模的增量贡献(Proposed vs Base-L)**: 在行为建模基础上添加韵律建模,PN-MOS 从 3.79 → 4.09 (+0.30),MCD 从 4.961 → 4.879,表明韵律建模进一步改善了整体自然度 [§3.3]

### 消融实验 [Table 2]

| 消融设置 | CMOS (PN) | CMOS (LN) | 出处 |
| --- | --- | --- | --- |
| Proposed (full) | 0 | 0 | [Table 2] |
| - spontaneous prosody modeling | -0.320 | — | [Table 2] |
| - spontaneous behavior modeling | — | -0.408 | [Table 2] |

### 标签预测器效果 [Fig 4]

主观偏好测试显示,使用预测标签的语音与手动标签的语音差距仅 6.8%(33.8% vs 27.0%,39.2% 无偏好)[Fig 4]。这说明标签预测器能从文本中预测合理的自发行为,实现无显式标签的推理。

### Case Study [§3.6, Fig 5]

对同一句话("嗯,风景真好看")的"嗯"添加不同标签(正面反馈、撒娇、filled pause),合成语音的 mel spectrogram、pitch 轮廓和时长显著不同 [Fig 5]:
- 正面反馈: 最短时长
- 撒娇: 更高 pitch
- Filled pause: 更低 pitch,后续语音也偏低
这验证了模型可以通过标签有效控制自发行为的声学表现。

## 局限性

1. **数据规模极小**: 微调数据仅 5.4h 单人普通话自发语料(4968 条),200 条测试集 [§3.1]。无法确认方法在大规模、多说话人场景下的泛化能力
2. **仅覆盖普通话**: 19 种自发行为分类基于普通话语言学特征 [§2.2],不同语言的自发行为差异显著(如英语的 um/uh、日语的相槌),方法的跨语言适用性未知
3. **缺乏公开基准对比**: 使用内部数据集和自行训练的 baseline,无法与其他系统在公开数据集上直接对比
4. **标注成本高**: 19 种行为的显式标注需要语言学专家,限制了方法的可扩展性
5. **LN-MOS 置信区间异常**: Proposed 的 LN-MOS 置信区间为 +/-0.737,远大于其他模型(0.084-0.098),可能暗示评估中存在评分者间一致性问题 [Table 1]
6. **未开源**: 论文提供 demo 页面但未提及代码和模型开源
7. **推理效率未报告**: 额外的行为编码器和韵律预测器增加了推理成本,但未报告具体开销
8. **未与 NVSpeech 等近期工作对比**: NVSpeech 同样建模多类副语言发声,两者的方法路线和效果未做直接对比

## 点评

本文是将 LM-based TTS 系统化地应用于自发语音合成的代表性工作。其最大贡献不在于单个模块的技术创新,而在于**系统性地整合了三个维度**: (1) LM backbone 的大规模预训练能力,(2) 19 种自发行为的统一显式建模,(3) 细粒度韵律的提取与预测。这种系统化思路使得自发语音合成从"处理个别现象"走向"统一框架"。

句法感知的行为编码器是方法论上最有新意的设计。利用自发行为在句中的位置信息(6 维句法结构)来区分同一行为标签在不同语境下的语用功能差异,这一思路直觉清晰且被实验验证有效(Base-L vs VALL-E 的提升)。

然而,5.4h 单人数据的验证规模是显著的局限。在当前 LM-based TTS 动辄使用数万小时数据的背景下,本文的实验设置偏保守。此外,19 种行为的显式标注依赖语言学专家,在工业落地时标注成本是主要障碍。NVSpeech 的解法(训练 paralinguistic-aware ASR 自动标注)在这方面更具可扩展性。

从知识库视角看,本文与 [[论文笔记/DiffCSS|DiffCSS]] 关注的都是"超越朗读式语音的自然表达",但切入点不同: DiffCSS 通过 diffusion 建模对话韵律多样性(one-to-many mapping),本文通过显式行为标签控制自发现象(one-to-one mapping with label)。两者互补:DiffCSS 解决"多种合理韵律中选哪种",本文解决"如何产出特定的自发行为"。

## 可复用的 idea

1. **句法位置编码辅助副语言建模**: 将自发行为/副语言现象在句中的位置(句首/句中/句尾、子句索引等)编码为显式特征,帮助区分同一行为的不同语用功能。可推广到任何需要建模位置依赖声学变化的任务
2. **行为标签与韵律的注意力耦合**: 用多头注意力以行为标签嵌入为 query、韵律表征为 key/value,学习"每种行为对应什么韵律模式"。这种显式耦合机制可迁移到情感-韵律、口音-韵律等场景
3. **三步渐进微调策略**: 先训 backbone + 提取器 → 再训预测器(teacher-student)→ 最后联合精调(适应预测误差)。这种分步解耦训练在多模块 TTS 系统中普遍适用
4. **19 种自发行为分类体系**: 基于语言学的 disfluency/interjections/non-speech 三大类、19 小类分类法,可作为中文自发语音标注的参考标准

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> Issues: 4 (high: 0, medium: 0, low: 4)
> 详见 `_review/SpontaneousStyleTTS-review.yml`

---

检索命中: [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓, [[EnCodec]]✓ | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: [[StyleTransferinTTS]], [[GlobalStyleTokens]]
