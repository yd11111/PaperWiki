---
type: paper
tier: deep
title: "Spontaneous Style Text-to-Speech Synthesis with Controllable Spontaneous Behaviors Based on Language Models"
arxiv_id: "2407.13509"
source: "Sources/Multi-ScaleAcousticPrompts.pdf"
authors: [Weiqin Li, Peiji Yang, Yicheng Zhong, Yixuan Zhou, Zhisheng Wang, Zhiyong Wu, Xixin Wu, Helen Meng]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, spontaneous-speech, language-model, prosody-modeling, spontaneous-behavior, VALL-E, Mandarin]
concepts: ["[[ProsodyModeling]]", "[[LLM-basedTTS]]", "[[CodecLanguageModel]]", "[[StyleTransferinTTS]]", "[[MelSpectrogram]]"]
models: ["[[模型库/EnCodec|EnCodec]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认页: [[ProsodyModeling]], [[LLM-basedTTS]], [[SpeechTokenizer]], [[模型库/EnCodec|EnCodec]], [[CodecLanguageModel]][待确认], [[StyleTransferinTTS]][待确认], [[MelSpectrogram]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文是 LLM-based TTS 范式(以 VALL-E 为代表)在自发式语音合成方向的扩展。VALL-E 开创了 codec language model TTS,采用 AR+NAR 两阶段在 EnCodec tokens 上做 next-token prediction。本文在此骨干上增加了显式自发行为建模和细粒度韵律建模两个模块。
>
> **已有认知 — 韵律建模**: ProsodyModeling 页(confirmed)记录了韵律建模从显式(FastSpeech 2 variance adaptor)到隐式(GST/VAE/生成模型)再到 LLM in-context learning 的演进。值得注意的是,ProsodyModeling 页已收录副语言发声(Paralinguistic Vocalizations)建模,NVSpeech (2025) 定义了 18 类 word-level PV 分类体系。本文 (2024) 定义了 19 类自发行为分类体系,与 NVSpeech 的 PV 体系高度互补,两者可视为同一方向(非语言声音的显式建模)的独立探索。
>
> **已有认知 — 风格迁移**: StyleTransferinTTS 页记录了风格控制从 GST(无监督)到 reference encoder 到 LLM in-context learning 的演进。本文属于 "Style Tagging + Reference Prompt" 的混合路线:用显式标签控制自发行为类型,同时用声学 prompt 提供说话人风格。
>
> **创新判断**: 本文的核心创新是将自发行为建模与 LM-based TTS 结合。在 ProsodyModeling 页的演进线中,这属于 "Reference Encoder (2018) → VAE (2019) → In-context learning (2023)" 之后的一个分支:不依赖隐式学习,而是通过显式标注+句法感知编码器显式建模。这与 LLM-TTS 主流的隐式韵律建模方向形成对比。
>
> 检索命中: [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[模型库/EnCodec|EnCodec]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[StyleTransferinTTS]](pending-review), [[MelSpectrogram]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 VALL-E 骨干上增加显式自发行为编码器(19 类分类体系)+ 细粒度韵律提取/预测模块,使 LM-based TTS 能合成可控的自然自发式语音
> - **路线**: Phoneme → Text Encoder + Syntactic-aware Behavior Encoder(L-embeddings）→ AR/NAR Acoustic Decoder → EnCodec Decoder → Waveform; 并行: Mel → Prosody Extractor → P-embeddings (训练); Text+L-emb → Prosody Predictor → P-embeddings (推理)
> - **指标**: PN-MOS 4.09 / LN-MOS 4.05 (vs VALL-E 3.30/3.32); MCD 4.879 (vs 5.291); 预测标签 vs 手动标签仅差 6.8% 偏好率 [Table 1, Fig 4]
> - **可借鉴**: (1) 句法位置感知的自发行为编码思路 — 同一标签在不同句法位置有不同韵律表现; (2) 三阶段渐进微调策略(先联合训练主模块 → 单独训练 predictor → 低 LR 联合精调); (3) 用 cross-attention 将行为标签作为 query、韵律表征作为 KV 来融合两类信息
> - **局限**: 仅 5.4 小时单人普通话语料; 仅 20 句主观评测; 无与 CosyVoice/Seed-TTS 等现代系统对比; 19 类分类体系可能语言特异; 未开源

## 核心问题

1. **LM-based TTS 能否生成真正自然的自发式语音?** 现有 LM-TTS(如 VALL-E)能从大规模数据中学到一些自发表达,但缺乏对自发行为的显式建模和控制,导致自发行为的多样性和韵律自然度不足 [§1]。

2. **如何系统化地建模多样的自发行为?** 先前工作只关注个别类型(如 filled pause、呼吸、笑声),忽略了自发行为的广泛多样性。本文提出了覆盖 19 种行为的分类体系 [§2.2]。

3. **如何捕捉自发语音中的细粒度韵律变化?** 自发语音的韵律本质上比朗读语音更复杂多变,仅靠文本信息难以准确预测 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统基于 VALL-E [§2.1],包含四个主要组件 [Fig 1]:

1. **骨干框架**: VALL-E 的 text encoder + AR/NAR acoustic transformer decoder + EnCodec。数据流和训练方式与 VALL-E 相同。
2. **句法感知自发行为编码器**: 将自发行为标签和句法位置信息编码为 L-embeddings,加到 text embeddings 上。
3. **NAR 标签预测器**: 从 text embeddings 预测自发行为标签序列,推理时无需显式标签。
4. **自发韵律建模**: 包含韵律提取器(训练时从 mel 提取 P-embeddings)和 LM-based 韵律预测器(推理时预测 P-embeddings)。

### 关键设计选择

**自发行为分类体系 [§2.2]**: 将普通话自发行为分为三大类 19 种:
- **不流畅(Disfluency)**: filled pause、重复、口吃、延长
- **感叹词(Interjections)**: 疑问、回应、惊讶、正面反馈、提醒、领悟、叹息、撒娇、冷笑(按语用功能分类)
- **非语言声音(Non-speech sounds)**: 微笑、大笑、苦笑、尴尬笑、嘲笑、不自觉笑

[论文原文] 作者声称这是"迄今最全面的"自发行为建模工作 [§1]。[agent 解读] 与 NVSpeech (2025) 的 18 类 PV 体系对比,本文的分类更细粒度地划分了笑声子类型(6 种),但在生理性声音(如呼吸、咳嗽)方面覆盖不如 NVSpeech。

**句法感知编码 [§2.3, Fig 2]**: 核心 insight 是自发行为的句法位置决定了其语用功能和韵律表现 [论文原文]。编码过程:
1. 字符级自发标签序列 → 音素级扩展(简单重复)→ label embedding
2. 提取每个自发行为标签在字符级的句法结构(含 6 种句法信息:字在子句中的索引、字在句中的索引、子句中的字数等)[Fig 2]
3. 句法信息扩展到音素级 → 经两层线性+ReLU 编码
4. 句法表征与标签 embedding 组合 → L-embeddings → 加到 text embeddings 上

[agent 解读] 将行为标签的句法位置信息显式编码是本文的核心设计选择。这与 LLM-TTS 主流的"让模型自己学"形成对比。优势是可控性强,代价是需要显式标注。

**韵律提取与预测 [§2.4, Fig 3]**:
- **韵律提取器**: 3 层卷积(mel → 帧级 → 音素级平均池化 → 最终韵律表征)+ multi-head attention(L-embeddings 作 Q,韵律表征作 K/V)→ P-embeddings。[agent 解读] 这个 cross-attention 设计使韵律提取能感知行为类型,即"不同行为关注不同的韵律维度"。
- **韵律预测器**: AR transformer decoder,输入 text + L-embeddings + 句法信息,预测 P-embeddings。采用 AR 而非 NAR 以捕捉局部和全局韵律依赖 [§2.4.2]。

### 训练策略

**预训练**: 在 WenetSpeech (10K 小时多领域普通话)上预训练骨干模型,40 epochs,8x A100 [§3.1]。

**三阶段渐进微调 [§2.5]**:
1. 联合训练骨干+行为编码器+标签预测器+韵律提取器(不含 codec)。损失 = CE(acoustic tokens) + 0.1 * CE(behavior labels) [Eq. 1]。30K iterations。
2. 用训练好的韵律提取器作 teacher,单独训练韵律预测器(MSE loss on P-embeddings)。20K iterations。
3. 冻结韵律提取器,用更低学习率联合训练韵律预测器+声学模型。16K iterations。

[agent 解读] 三阶段策略的逻辑:第一阶段建立行为表征空间;第二阶段让预测器学习模仿提取器;第三阶段弥合预测-提取的 gap。这种 teacher-student + progressive training 是处理多模块系统的标准做法。

**微调数据**: 5.4 小时自发式语音(4968 条,单女性说话人),含 5907 个自发行为标注 [§3.1]。

## 实验

| 指标 | Proposed | Base-L | VALL-E | FastSpeech 2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| PN-MOS ↑ | 4.090 ± 0.073 | 3.792 ± 0.084 | 3.302 ± 0.099 | 2.606 ± 0.106 | 内部自发语料 | [Table 1] |
| LN-MOS ↑ | 4.046 ± 0.737 | 3.898 ± 0.084 | 3.324 ± 0.098 | 2.536 ± 0.097 | 同上 | [Table 1] |
| MCD ↓ | 4.879 | 4.961 | 5.291 | 5.355 | 同上 | [Table 1] |

**消融实验 [Table 2]**:

| 消融条件 | CMOS |
| --- | --- |
| 去除韵律建模 (PN) | -0.320 |
| 去除行为建模 (LN) | -0.408 |

**标签预测 vs 手动标签 [Fig 4]**: 手动标签偏好 33.8%,预测标签 27.0%,无偏好 39.2%。差距仅 6.8%,说明标签预测器能从文本合理推断自发行为。

**Case Study [§3.6, Fig 5]**: 同一句话 "um, the scenery is really beautiful" 赋予不同标签(positive feedback / coquetry / filled pause)后,mel spectrogram 和 pitch contour 呈现明显差异:正面反馈时长最短,撒娇 pitch 更高,filled pause 音调低且后续语句也偏低。

## 局限性

1. **数据规模极小**: 仅 5.4 小时单人普通话自发语料,无法验证方法的泛化能力(跨说话人、跨语言、跨领域)[agent 解读]。
2. **评测规模有限**: 主观测试仅 20 句,25 名评测者。LN-MOS 的置信区间异常大(4.046 ± 0.737 vs 其他 ±0.08 级别),可能是排版错误或统计问题 [Table 1]。
3. **无现代系统对比**: 仅与 FastSpeech 2 和原始 VALL-E 对比,缺少与 CosyVoice、Seed-TTS、NaturalSpeech 系列等 2024 年主流系统的比较 [agent 解读]。
4. **语言特异性**: 19 种分类体系基于普通话语言学研究,跨语言适用性未验证 [agent 解读]。
5. **依赖显式标注**: 需要字符级自发行为标注,标注成本高,限制了数据扩展 [agent 解读]。
6. **骨干过时**: 使用 VALL-E(2023 初)+ EnCodec 作骨干,不是当前最优的 codec 或 TTS 架构 [agent 解读]。

## 点评

**优势**:
- 在 LM-TTS 时代首次系统化处理自发行为建模,19 种行为分类是目前最全面的之一。
- 句法位置感知的行为编码是有见地的设计 — 承认同一类型行为在不同句法位置的韵律表现不同。
- 三阶段渐进微调策略合理,避免了多模块同时训练的优化困难。
- 消融实验清晰展示了行为建模和韵律建模各自的贡献。

**不足**:
- 实验规模(数据量、测试集大小、对比系统)不足以支撑方法的泛化性声明。
- 与 NVSpeech (2025, 18 类 PV)的工作高度相关但时间线上略早,两者的分类体系异同值得深入对比。
- 缺乏客观韵律指标(如 F0 RMSE、duration accuracy)的细粒度分析。
- LN-MOS 置信区间数据可能有误(0.737 与其他行的 0.08 量级不符)。

**定位**: 这是一篇 Interspeech 2024 短文(4 页正文),在 LLM-TTS 时代为自发语音合成提供了一个系统化框架。其方法设计(行为分类+句法编码+韵律提取/预测)比结果本身更有参考价值。

## 审阅

> [!review] 审阅 pass-with-fixes (2026-06-08, auto)
> 0 high / 0 medium / 3 low issues。详见 `_review/Multi-ScaleAcousticPrompts-review.yml`。
> - [low] frontmatter datasets 为空,可补 WenetSpeech
> - [low] PDF 文件名与实际论文内容可能不匹配(Multi-ScaleAcousticPrompts.pdf 实际为 Spontaneous Style TTS 2407.13509)
> - [low] 训练配置出处标注确认通过

## 可复用的 idea

1. **句法位置感知编码**: 对任何需要建模"位置依赖语用功能"的任务(如对话中的 turn-taking 信号、朗读中的重音模式),将元素的句法位置信息显式编码到 embedding 中可能有效。

2. **Cross-attention 融合标签与韵律**: L-embeddings 作 Q,韵律表征作 K/V 的 attention 设计,使得行为类型能"选择"关注哪些韵律维度。这种设计可迁移到任何需要条件感知特征提取的场景。

3. **三阶段渐进微调**: 先建立表征空间 → 训练预测器(teacher-student)→ 低 LR 联合精调。对多模块系统是通用的训练策略。

4. **自发行为分类体系**: 19 种行为的三层分类(disfluency/interjection/non-speech)可作为中文自发语音标注的参考框架。
