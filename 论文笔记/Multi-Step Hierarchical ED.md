---
type: paper
tier: deep
title: "Multi-Step Prediction and Control of Hierarchical Emotion Distribution in Text-to-Speech Synthesis"
arxiv_id: "2507.04598"
source: "Sources/2507.04598.pdf"
authors: [Sho Inoue, Kun Zhou, Shuai Wang, Haizhou Li]
year: 2025
venue: "arXiv"
tags: [TTS, emotion, hierarchical, multi-step-prediction, emotion-distribution, variance-adaptor, FastSpeech2, controllability, prosody, emotion-editing, fine-grained-control]
concepts: ["[[Emotion Control in TTS]]", "[[Prosody Modeling]]", "[[Non-autoregressive TTS]]", "[[F0 Modeling]]", "[[Duration Predictor]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文属于 Emotion Control in TTS 的**层级建模**分支,直接继承 MsEmoTTS (Lei et al., 2022) 的多尺度思路,但从"单步并行预测"升级为"多步顺序预测"。在 KB 的情感控制演进线中,本文处于"多尺度层级建模 (MsEmoTTS, 2022)"之后、"LLM 自由文本情感 (EmoVoice, 2025)"之前的位置,代表了 NAR TTS 框架下细粒度情感控制的进一步深化。

**已有认知**:
- [[Emotion Control in TTS]] [待确认] 记录了情感控制从 one-hot label → embedding → 多尺度层级 → DPO/RLHF → 球面向量 → LLM 文本描述的完整演进线。该页已收录 MsEmoTTS 和 EmoSphere-TTS 等工作,但尚未覆盖本文的"多步预测"范式。
- [[Prosody Modeling]] 指出韵律的物理维度包括 Duration/Pitch/Energy/Pause,FastSpeech 2 的 variance adaptor 是显式韵律建模的标杆。本文正是在 FastSpeech 2 的 variance adaptor 框架上扩展情感维度。
- [[Non-autoregressive TTS]] 描述了 FastSpeech 2 架构: Encoder → Duration Predictor → Length Regulator → Pitch/Energy Predictor → Decoder,本文的两种集成方式都建立在此基础上。
- [[Duration Predictor]] 记录了 variance adaptor 中的 duration 预测机制,本文在此基础上增加了层级 ED 预测分支。
- [[Speaker Embedding]] 中 Resemblyzer (GE2E) 是本文多说话人场景使用的 speaker encoder。

**创新判断**: 相比 KB 中已有的情感控制方法(大多是 embedding/标签/参考音频驱动),本文的独特性在于: (1) 将情感量化为连续的多层级分布向量而非离散标签; (2) 用多步预测建模层级间的上下文依赖而非独立预测; (3) 提供外部模块和 VA 内嵌两种即插即用的集成方式。

> 检索命中: [[Prosody Modeling]]✓, [[Speaker Embedding]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[Non-autoregressive TTS]](pending-review), [[F0 Modeling]](pending-review), [[Duration Predictor]](pending-review) | 未命中但可能相关: [[Style Transfer in TTS]]

## 速查

> [!summary] 速查
> - **一句话**: 提出多步预测框架,将情感分布从 utterance→word→phoneme 逐级细化,使高层情感上下文引导底层韵律,在 FastSpeech 2 上显著提升情感表达力和可控性
> - **路线**: Text → Encoder → Linguistic Embedding + Hierarchical ED (utterance→word→phoneme, 多步预测) → Variance Adaptor (pitch/duration/energy) → Decoder → Mel → HiFi-GAN → Waveform
> - **指标**: MUSHRA 自然度 62.2 (GT ED, VA Multi-Step) vs 57.5 (GT ED, VA Single-Step); WER 2.45% (Pred, VA Multi-Step) vs 4.61% (Pred, VA Single-Step); BWS 测试中情感可控性全面优于 MsEmoTTS [Table 1][Table 2][Table 4]
> - **可借鉴**: 多步预测的设计模式 — 先预测粗粒度(utterance)再条件化细粒度(word→phoneme),这种 coarse-to-fine 逐级细化策略可迁移到任何需要多层级控制的场景(如韵律/风格/说话人属性的层级建模)
> - **局限**: 仅在 FastSpeech 2 上验证,未在 LLM-based TTS 或 flow-based 模型上测试; 仅支持 5 类基础情感(ESD),未涉及混合情感或连续维度(arousal-valence); 不支持零样本情感迁移

## 核心问题

本文要解决的核心问题是: **现有情感 TTS 方法将不同层级(utterance/word/phoneme)的情感变量独立预测,忽略了人类情感在语音中的层级依赖结构 — 即全局情感基调应当自上而下地影响局部韵律细节** [§1]。

具体而言,作者之前的工作 (Inoue et al., 2024a; 2024b) 已经提出了层级情感分布 (Hierarchical ED) 的概念来量化 utterance/word/phoneme 三级情感强度,但采用的是单步并行预测策略 — 三个层级的 ED 同时独立预测,无法利用高层信息约束低层 [§1]。这导致了两个问题: (1) 低层预测缺乏全局情感上下文的引导,可能产生局部不一致; (2) 合成语音的情感表现力和自然度受限 [§2.2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由两个核心模块组成 [§3]:

1. **层级 ED 提取器 (Hierarchical ED Extractor)** [§3.1]: 从音频中提取三级情感分布向量(训练时用 ground-truth 音频,推理时由文本预测或从参考音频提取)
2. **多步层级 ED 建模模块 (Multi-Step Hierarchical ED Modeling)** [§3.2]: 将层级 ED 信息集成到 TTS 框架中,以多步方式逐级预测 ED

运行时支持三种场景 [Fig 1]:
- **(a) TTS with Emotion Prediction**: 从文本直接预测层级 ED
- **(b) TTS with Emotion Control**: 从文本预测 ED 后,用户可修改
- **(c) Emotion Editing**: 从参考音频提取 ED,用户手动调整

### 关键设计选择

#### 1. 情感分布 (ED) 的定义与提取

**选择**: 用 SVM 排序函数将 OpenSMILE 88 维特征映射为连续的情感强度值 [0,1],而非用离散类别标签。

**为什么这样设计而不是用离散标签**: [论文原文] 作者采用 relative attributes (Parikh & Grauman, 2011) 的思路,将情感视为可排序的语音属性,通过二分类 SVM (e.g., Angry vs Non-angry) 对情感进行连续量化 [§3.1]。[agent 解读] 这种设计的核心优势在于: 连续量化天然支持强度控制(用户可将愤怒强度设为 0.3 或 0.8),而离散标签只能切换类别; 同时,SVM 排序函数的输出可解释性强,每个维度对应一种情感的相对强度。

**具体流程** [§3.1, Fig 2]:
1. 用 Montreal Forced Aligner (MFA) 将音频分段为 phoneme/word/utterance 三级
2. 用 OpenSMILE 提取每个片段的 88 维特征
3. 预训练的 SVM 排序函数估计每个片段的 ED 向量(每个元素 = 一种情感的强度)
4. 为保持层级一致性,将 utterance-level ED 复制到所有 phoneme,word-level ED 复制到对应 phoneme [Fig 2(a)]

#### 2. 多步预测 vs 单步预测

**选择**: ED 按 utterance → word → phoneme 顺序逐步预测(高层先于低层),而非三级同时并行预测。

**为什么多步优于单步**: [论文原文] 人类调节语音情感的方式是从整体基调出发,再逐步细化语调和发音 [§1]。单步预测将三个层级视为独立任务,无法建模"全局情感上下文→局部韵律细节"的因果链 [§2.2]。[论文原文] 多步预测使高层 ED 信息成为低层预测的条件,确保层级间的一致性 [§3.2]。

**实验验证**: [Table 1] VA Multi-Step + Predicted ED 的 WER 为 2.45%,而 VA Single-Step 为 4.61%,降幅达 47%。MUSHRA 自然度从 52.2 提升到 53.2。[agent 解读] WER 的巨大改善暗示多步预测不仅提升了情感表达,更重要的是通过层级一致的 ED 改善了语音的整体质量和可懂度。

#### 3. 两种集成策略

**External Integration ("External")** [§3.2.1, Fig 3]:
- ED 作为外部模块,通过全连接网络映射为 ED embedding 后拼接到 encoder 输出
- 文本 encoder 在 ED 预测阶段保持冻结
- **优势**: 模型无关,可插入任意 TTS 系统
- **劣势**: ED 预测与声学模型训练分离,可能导致不一致

**Variance Adaptor Integration ("VA")** [§3.2.2, Fig 4]:
- 层级 ED 建模直接嵌入 FastSpeech 2 的 variance adaptor
- 语言编码器联合训练,使用 MSE loss 最小化预测 ED 与 ground-truth ED 的差异
- **优势**: ED 预测与韵律预测紧耦合,端到端优化
- **劣势**: 与 FastSpeech 2 绑定,[论文原文] 联合训练可能使模型对 ED 变化更敏感,在 pitch 和 FD 指标上产生波动 [§5.1.2]

**为什么提供两种方案**: [论文原文] 为展示框架的灵活性 [§1]。[agent 解读] 两种方案覆盖不同使用场景: External 适合已有 TTS 系统的即插即用增强,VA 适合从头训练的高质量需求。

### 训练策略

**ED 提取器训练** [§4]:
- 使用 ESD 数据集训练 SVM 排序函数
- 每个说话人每种情感随机选 100 样本,共 5000 个样本

**TTS 训练** [§4.1]:
- Backbone: FastSpeech 2 (Transformer encoder + variance adaptor + Transformer decoder)
- 损失函数: L1 loss (mel) + MSE (prosodic predictions)
- 多说话人: Resemblyzer speaker embeddings 注入 encoder 输出
- 训练: batch 32, 200K iterations, 48h 单 GPU
- External ED 预测模块: 额外 100K iterations
- Vocoder: HiFi-GAN (在 ESD + LibriTTS-R 上训练)

**数据** [§4]:
- 情感预测实验: LibriTTS-R train-clean-100 + train-clean-360 (~580h, 2306 speakers)
- 情感编辑实验: ESD 英语部分 (5 情感 × 10 speakers, ~29h)

## 实验

### 语音质量 (Speech Quality)

| 指标 | VA (Multi-Step) + Pred | VA (Single-Step) + Pred | External (Multi-Step) + Pred | External (Single-Step) + Pred | GT Speech | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MUSHRA 自然度 ↑ | 53.2±2.4 | 52.2±2.6 | 54.0±2.3 | 50.7±2.4 | 79.4±1.9 | [Table 1] |
| WER ↓ | 2.45% | 4.61% | 3.25% | 3.80% | 2.16% | [Table 1] |

### 情感表现力 (Emotion Expressiveness)

| 指标 | VA (Multi-Step) + Pred | VA (Single-Step) + Pred | External (Multi-Step) + Pred | External (Single-Step) + Pred | 出处 |
| --- | --- | --- | --- | --- | --- |
| MUSHRA 情感相似度 ↑ | 49.1±2.2 | 48.2±2.5 | 51.9±2.2 | 47.2±2.3 | [Table 2] |
| MCD ↓ | 6.91±0.12 | 7.23±0.20 | 6.89±0.12 | 7.59±0.14 | [Table 2] |
| Pitch Distortion ↓ | 17.2±1.1 | 16.7±1.0 | 16.9±1.2 | 18.2±1.1 | [Table 2] |
| Energy Distortion ↓ | 0.416±0.025 | 0.426±0.025 | 0.409±0.024 | 0.438±0.027 | [Table 2] |
| FD ↓ | 46.3±5.1 | 41.4±4.7 | 42.6±4.7 | 46.3±6.5 | [Table 2] |

### 情感可控性 (BWS Test, Emotion Editing)

| 情感 | Hierarchical ED: Least@0.0 / Most@1.0 | MsEmoTTS: Least@0.0 / Most@1.0 | 出处 |
| --- | --- | --- | --- |
| Angry | 79% / 74% | 42% / 63% | [Table 4] |
| Happy | 63% / 75% | 32% / 72% | [Table 4] |
| Sad | 67% / 75% | 21% / 47% | [Table 4] |
| Surprise | 81% / 86% | 33% / 53% | [Table 4] |

### 关键发现

1. **多步一致优于单步**: 在 GT ED 和 Predicted ED 条件下,Multi-Step 在 MUSHRA 自然度和 WER 上均优于 Single-Step [Table 1][Table 2]
2. **VA 设置下多步在 Pitch/FD 上不优于单步**: [论文原文] 可能源于多级 ED 预测的误差累积,以及 VA 的 MSE 联合训练使模型对 ED 变化更敏感 [§5.1.2]
3. **ED 差异与音频质量不完全相关**: [Table 3] 单步和多步预测的 ED 差异相近,但合成音频质量差异显著,说明多步预测学到了"层级依赖关系"而不仅仅是更准确的 ED 值 [§5.1.3]
4. **愤怒与惊讶在 word level 正相关**: [Fig 5] 这两种情感在词级分布上显示出较强的互相关,与心理学研究中两者共享相似声学特征(高 pitch/energy)的发现一致 [§5.1.3]

## 局限性

1. **仅验证于 FastSpeech 2**: 未在现代 LLM-based TTS (VALL-E, CosyVoice) 或 flow-based 模型 (MaskGCT) 上验证,这些模型隐式建模韵律,层级 ED 的集成方式需重新设计 [agent 解读]
2. **情感类别受限**: 仅支持 5 类基础离散情感 (Neutral/Angry/Happy/Sad/Surprise),不支持连续情感维度 (arousal-valence-dominance) 或混合情感,与 EmoSphere++ 和 EmoCtrl-TTS 等方法相比覆盖面窄 [agent 解读]
3. **ED 提取依赖外部工具**: OpenSMILE + SVM 的 pipeline 需要 MFA 强制对齐和预训练排序函数,增加了部署复杂度 [§3.1]
4. **无零样本能力**: 需要在情感数据 (ESD) 上训练 ED 提取器和排序函数,无法泛化到未见情感类型或说话人 [agent 解读]
5. **主观评估样本量有限**: 20 名参与者评估 210 个样本,统计检验力偏低 [§4.3]
6. **VA 设置下的误差累积**: 论文自身承认 VA 集成的多步方案在 Pitch Distortion 和 FD 上反而不如单步 [§5.1.2, Table 2]

## 点评

本文的核心贡献是将**多步顺序预测**引入层级情感分布建模,这是一个简洁有效的思路: 用 utterance-level ED 条件化 word-level 预测,再条件化 phoneme-level 预测,使高层情感上下文自然传播到底层韵律。WER 从 4.61% 降到 2.45% 的改善尤其值得注意,说明情感层级一致性不仅影响表达力,更直接影响可懂度。

但本文的局限也比较明显: (1) 建立在 FastSpeech 2 上的设计在 LLM-TTS 时代显得较为过时,能否迁移到 CosyVoice 等现代框架存疑; (2) 5 类离散情感 + SVM 排序函数的方案与近期基于 wav2vec 2.0 / 连续 arousal-valence 的方法 (EmoCtrl-TTS) 相比,在情感覆盖和泛化能力上有明显差距; (3) External vs VA 两种集成方式各有优劣但都未展现出压倒性优势,且 VA 的误差累积问题削弱了多步预测的优势。

## 可复用的 idea

1. **多步 coarse-to-fine 预测模式**: 先预测全局属性,再用全局属性作为条件预测局部属性。这种模式不限于情感,可用于任何需要层级控制的场景(如先预测 utterance-level 语速,再条件化 word-level duration)
2. **连续强度量化 via SVM 排序**: 将分类问题转化为排序问题获取连续强度值,比 softmax 概率更直观,适用于任何需要从离散标签生成连续属性的场景
3. **模型无关的外部模块设计 (External Integration)**: 将可控属性建模为独立模块,通过 embedding 注入而非修改 TTS 内部结构,降低集成成本 — 类似 ControlNet 对 diffusion model 的增强方式
4. **层级 ED 的一致性复制策略**: utterance ED → 复制到所有 phoneme, word ED → 复制到对应 phoneme,确保训练时标签的层级一致性 [Fig 2(a)]

> [!review] 审阅待补
> 本笔记生成后待审阅。
