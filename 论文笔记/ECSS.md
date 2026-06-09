---
type: paper
tier: deep
title: "ECSS: Emotion Rendering for Conversational Speech Synthesis with Heterogeneous Graph-Based Context Modeling"
arxiv_id: "2312.11947"
source: "Sources/ECSS.pdf"
authors: [Rui Liu, Yifan Hu, Yi Ren, Xiang Yin, Haizhou Li]
year: 2023
venue: "AAAI 2024"
tags: [TTS, conversational-speech-synthesis, emotion-graph, contrastive-learning, dataset-annotation, heterogeneous-graph, emotion-rendering]
concepts: ["[[ProsodyModeling]]", "[[EmotionControlinTTS]]", "[[GlobalStyleTokens]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-09
updated: 2026-06-09
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页: [[ProsodyModeling]]; 2 个待确认实体页: [[EmotionControlinTTS]], [[GlobalStyleTokens]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]]✓ | 过滤: [[EmotionControlinTTS]](pending-review), [[GlobalStyleTokens]](pending-review) | 未命中但可能相关: 无

**谱系定位**: ECSS 处于 Conversational Speech Synthesis (CSS) 与 Emotion Control in TTS 的交叉地带。在 CSS 领域,既有方法 (Guo et al. 2020 GRU-based; Li et al. 2022a DialogueGCN; Li et al. 2022b MRGCN) 建模对话上下文的speaking style,但均未显式建模情感动态。在 Emotion Control in TTS 领域,情感建模方法 (MsEmoTTS, Daisy-TTS, Emo-DPO 等) 大多面向单句情感控制,未考虑对话交互中的情感上下文依赖。ECSS 是首个将异构图网络引入 CSS 以显式建模对话中情感流 (emotion flow) 的工作。

**已有认知**:
- [[ProsodyModeling]] (confirmed) 指出韵律建模的演进从 GST reference encoder (2018) → VAE → 显式 variance adaptor (FastSpeech 2) → 生成模型。ECSS 使用 FastSpeech 2 作为声学 backbone + GST 作为音频节点编码器,属于显式韵律预测 + reference encoder 组合。
- [[EmotionControlinTTS]] [待确认] 记录了情感建模方法从 emotion embedding 到层级建模再到 DPO/对抗解耦的演进。ECSS 的情感 renderer 通过对比学习区分 7 类情感 + 3 级强度,在技术路线上属于 "Emotion Embedding + Contrastive Learning" 分支。
- [[GlobalStyleTokens]] [待确认] 描述了 GST 的 reference encoder + token bank 机制。ECSS 将 GST 用作音频节点初始化编码器和韵律提取目标 (prosody predictor 的 MSE target 来自 GST),属于 GST 的下游应用。

**创新判断**: ECSS 的核心新颖性在于: (1) 首次在 CSS 中引入异构图网络,将 text/audio/speaker/emotion/intensity 五种信息建模为不同类型节点; (2) 首次在 CSS 中显式建模情感流和情感强度; (3) 对比学习训练准则增强情感/强度类别区分; (4) 为 DailyTalk 数据集新增 7 类情感 + 3 级强度标注 (开源)。

## 速查

> [!summary] 速查
> - **一句话**: 首个显式建模情感表达的对话语音合成系统,通过异构图 (HGT) 编码 5 种对话信息节点理解情感上下文,用对比学习情感渲染器推断当前句情感风格
> - **路线**: 多模态对话历史 (text+audio+speaker+emotion+intensity) → 异构情感对话图 ECG (5 类节点, 14 种边) → HGT 编码 (HMA+HMP+EKA) → 图增强节点特征 → Emotion Renderer (emotion predictor + intensity predictor + prosody predictor) + CL loss → FastSpeech 2 backbone → HiFi-GAN vocoder
> - **指标**: N-DMOS 3.506 / E-DMOS 3.619 (vs GRU 3.314/3.288, vs HomoGraph 3.384/3.493) [Table 1, DailyTalk]
> - **可借鉴**: (1) 异构图将对话中 multi-source knowledge 统一建模的框架设计; (2) 对比学习增强情感/强度类别区分的训练策略; (3) DailyTalk 情感标注方案 (7 category + 3 intensity) 可作为后续 emotional CSS 研究的 baseline 标注
> - **局限**: 仅单数据集 (DailyTalk, 20h, 2 speakers); FastSpeech 2 backbone 限制了音质天花板; 情感分布严重不平衡 (neutral 76.5%); 推理时需 ground-truth 情感标签作为图节点输入

## 核心问题

ECSS 要解决的核心问题是 **对话语音合成中情感理解与情感渲染的缺失** [Abstract, Introduction]。

具体而言,存在三个子问题:
1. **数据稀缺**: 现有情感多模态数据集 (IEMOCAP, MELD 等) 面向情感识别,语音保真度不够高,无法满足 CSS 训练需求 [§Introduction]
2. **情感上下文建模**: 对话中情感流 (emotion flow) 直接影响当前句的情感表达 (如 Fig. 1 所示,相同文本在不同情感上下文下表达不同情感),但既有 CSS 方法仅建模 speaking style 而非 emotion state [§Introduction]
3. **情感渲染精度**: 需要准确推断当前句的情感类别和情感强度,而非仅捕获笼统的风格信息 [§Introduction]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ECSS 由三大组件组成 [§Methodology, Fig. 2]:

1. **Multi-Source Knowledge**: 将对话历史中每个 utterance 表示为 5-tuple <text, speaker, audio, emotion, intensity>
2. **Heterogeneous Graph-based Emotional Context Encoder**: 构建异构情感对话图 (ECG),编码情感上下文依赖
3. **Emotional Conversational Speech Synthesizer**: 基于图增强特征,预测情感/强度/韵律,合成最终语音

### 关键设计选择

#### 1. 为什么用异构图而非同构图或序列模型?

[论文原文] 对话中的信息是天然异构的 --- text、audio、speaker、emotion、intensity 属于不同类型,它们之间的关系也各不相同。同构图 (如 DialogueGCN) 将所有 utterance 作为同类型节点,无法表达这种异构结构 [§Related works]。异构图允许为不同类型的节点和边定义不同的参数化,从而捕获更丰富的依赖关系。

[agent 解读] 关键的设计差异在于: 同构图中 emotion 信息只能作为节点特征的一部分被隐式编码;异构图中 emotion/intensity 作为独立节点存在,能与其他节点建立显式的图结构依赖,使得远距离 utterance 间的情感交互可通过 emotion-to-emotion 边直接传播。

#### 2. ECG 构建: 5 类节点 + 14 种边

节点类型 [§Heterogeneous Graph-based Emotional Context Encoder]:
- **Text nodes** (f_u): BERT (distiluse-base-multilingual) 编码文本语义
- **Audio nodes** (f_a): GST reference encoder 编码声学风格
- **Speaker nodes** (f_s): 可训练参数矩阵 (2 speakers)
- **Emotion nodes** (f_e): 可训练参数矩阵 (7 emotions)
- **Intensity nodes** (f_i): 可训练参数矩阵 (3 levels)

边类型: 14 种有向边,连接 text↔{audio, speaker, emotion, intensity}; audio↔speaker; emotion↔{speaker, intensity, audio}; intensity↔{speaker, audio},每种包含双向连接 [§ECG Construction]。

#### 3. HGT 编码: 三阶段信息传播

采用 Heterogeneous Graph Transformer (HGT, Hu et al. 2020) 作为 backbone [§ECG Encoding]:

1. **Heterogeneous Mutual Attention (HMA)**: target 节点作 Query,source 节点作 Key,计算 type-specific attention score
2. **Heterogeneous Message Passing (HMP)**: source 节点特征经线性投影 + 边依赖矩阵 W_{src→tgt} 生成 message
3. **Emotional Knowledge Aggregation (EKA)**: 用 attention score 加权聚合所有 neighbor message,得到情感增强的节点表征

[论文原文] 所有 ECG 节点 (包括 emotion/intensity) 都既可作为 source 也可作为 target,因此 EKA 最终将情感信息融入所有节点的表征 [§ECG Encoding]。

#### 4. Emotion Renderer: 情感渲染三预测器

情感渲染器从图增强特征预测当前句的情感表达 [§Emotion Renderer]:

- **Emotion Predictor**: CNN + BiLSTM + FC,输入为对话历史中所有 emotion 节点的图增强特征 f'_e,输出情感表征 H_C^e
- **Intensity Predictor**: CNN + BiLSTM + FC + AvgPooling,输入为 intensity 节点特征 f'_i,输出强度表征 H_C^i
- **Prosody Predictor**: Multi-head attention,用对话历史 text 节点特征 f'_u 预测韵律,MSE loss 对齐 GST 提取的韵律 target

[论文原文] 不使用 audio 节点特征是因为 text 和 audio 已在 ECG encoding 中交互,text 节点已包含 audio 信息 [§Emotion Renderer]。

#### 5. 对比学习训练准则

[论文原文] 受 Supervised Contrastive Learning (Khosla et al. 2020) 启发,设计 emotion CL loss (L_emo^cl) 和 intensity CL loss (L_int^cl): 同类情感/强度为正样本,不同类为负样本,通过拉近/推远提升 renderer 的类别区分能力 [§Contrastive Learning Training Criterion, Eq. 3-4]。

总 loss: L = L_emo^cl + L_int^cl + L_pro^mse + L_fs2

### 与 baseline 的关键差异

| 系统 | 上下文建模 | 情感建模 | 图结构 |
|------|-----------|---------|--------|
| Guo et al. 2020 | GRU (text only) | 无 | 无 |
| Li et al. 2022a | DialogueGCN (multi-modal) | 隐式 | 同构 |
| Li et al. 2022b | MRGCN (multi-scale) | 隐式 | 同构 |
| **ECSS** | HGT (5 types) | **显式** (emotion+intensity nodes + CL) | **异构** |

## DailyTalk 情感标注

这是本文的重要数据贡献 [§Dataset]:

**标注方案**:
- 7 类情感: happy, sad, angry, disgust, fear, surprise, neutral
- 3 级强度: weak, medium, strong
- 标注方式: 专业标注员同时听音频 + 理解文本语义

**标注分布**:
| 情感 | 数量 | 占比 |
|------|------|------|
| Neutral | 18,197 | 76.5% |
| Happy | 3,871 | 16.3% |
| Sad | 722 | 3.0% |
| Surprise | 497 | 2.1% |
| Angry | 226 | 1.0% |
| Disgust | 186 | 0.8% |
| Fear | 74 | 0.3% |

| 强度 | 数量 | 占比 |
|------|------|------|
| Weak | 19,973 | 84.0% |
| Medium | 3,646 | 15.3% |
| Strong | 154 | 0.6% |

[agent 解读] 分布极度不平衡 --- neutral 占 76.5%,fear 仅 74 条;strong 仅 154 条。这对对比学习的正样本构造可能造成挑战 (如 fear 类在 batch 中很难有足够正样本对)。标注已开源,可作为后续 emotional CSS 研究的 baseline。

## 实验

### 主实验结果 [Table 1]

| 系统 | N-DMOS | E-DMOS | MAE-M | MAE-P | MAE-E | MAE-D |
|------|--------|--------|-------|-------|-------|-------|
| No context | 3.232 | 3.100 | 0.681 | 0.506 | 0.346 | 0.300 |
| GRU-based | 3.314 | 3.288 | 0.675 | 0.506 | 0.352 | 0.296 |
| HomoGraph | 3.384 | 3.493 | 0.662 | 0.456 | 0.204 | 0.150 |
| **ECSS** | **3.506** | **3.619** | **0.654** | **0.455** | 0.215 | 0.152 |

ECSS 在主观指标 (N-DMOS, E-DMOS) 上全面最优;客观指标 MAE-M/MAE-P 最优,MAE-E/MAE-D 次优 [§Main Results]。

### 消融实验 [Table 1, rows 6-10]

| 移除项 | N-DMOS 下降 | E-DMOS 下降 |
|--------|------------|------------|
| w/o emotion node | -0.082 | -0.123 |
| w/o intensity node | -0.019 | -0.096 |
| w/o speaker node | -0.105 | -0.108 |
| w/o audio node | -0.115 | -0.114 |
| w/o L_cl (用 CE 替代) | -0.118 | -0.307 |

[论文原文] 对比学习 loss 影响最大 (E-DMOS 下降 0.307),说明 CL 是情感表达区分度的关键; emotion 节点对 E-DMOS 贡献第二大 (下降 0.123) [§Ablation Results]。

### 上下文长度分析 [Table 2]

最优长度为 10 (与训练设置一致); 过短 (2) 或过长 (14) 均导致性能下降 [§Context Length Analysis]。DailyTalk 平均对话长度 9.3 turns,设置 10 合理。

### 可视化分析 [Fig. 3, Fig. 4]

SER 模型对 ECSS 合成语音的情感分类混淆矩阵呈现清晰对角线,优于所有 baseline [§Visualization Study]; 5 名听众标注的强度混淆矩阵同样清晰。

## 对比定位

### vs 同期 CSS 方法 (Li et al. 2022b MRGCN)

MRGCN 使用同构图在多尺度建模 multi-modal 依赖,但不显式建模情感; ECSS 引入异构图 + emotion/intensity 节点,首次在 CSS 中显式建模情感流。技术差异: 同构 GCN vs 异构 HGT。

### vs 同组后续 DiffCSS (Wu et al. 2025)

DiffCSS 是 DailyTalk 上的后续工作,用 diffusion 建模韵律多样性,backbone 升级为 LM-based TTS (ParlerTTS)。ECSS 关注情感理解 (异构图+对比学习),DiffCSS 关注韵律多样性 (diffusion)。两者互补: ECSS 解决"理解什么情感",DiffCSS 解决"同一情感下如何多样表达"。

### vs GST-based 方法

ECSS 将 GST 降级为组件 (音频节点编码器 + prosody target 提取器),而非核心风格控制机制。GST 的全局风格粒度无法捕获对话中动态变化的情感流。

### vs EmoPP (Liu et al. 2023, 同一作者)

EmoPP 从文本前端 (prosodic phrasing) 切入情感感知; ECSS 从声学后端 (emotion rendering) 切入对话情感。两者是同一作者在情感CSS方向的不同视角。

## 局限性

1. **数据规模与多样性**: 仅在 DailyTalk (20h, 2 speakers) 验证,无法证明方法在大规模、多说话人场景下的泛化性 [agent 解读]
2. **情感分布极度不平衡**: neutral 76.5%, fear 仅 0.3%,少数类的情感渲染质量未被充分验证 [agent 解读,基于 §Dataset 统计]
3. **推理依赖**: 对话历史中的 emotion/intensity 节点需要 ground-truth 标签初始化,实际部署时需要额外的 SER 模型提供预测标签 [agent 解读,基于 §Task Definition 的 5-tuple 定义]
4. **Backbone 限制**: FastSpeech 2 + HiFi-GAN 是 2021 年水平的 TTS backbone,音质天花板低于当前 LM-based TTS [agent 解读]
5. **评估局限**: 主观评估仅 30 名第二语言为英语的学生,非 native speaker 对英语情感的感知可能有偏差 [§Evaluation Metrics]

## 反向更新计划

- [[ProsodyModeling]]: 追加 key_papers (ECSS 作为 CSS 中情感韵律建模的首个深度工作)
- [[EmotionControlinTTS]]: 追加 key_papers (ECSS 首次将情感控制引入 CSS 场景,+ 对比学习训练范式)
- [[GlobalStyleTokens]]: 追加"在 TTS 中的应用" --- ECSS 将 GST 用作异构图的音频节点编码器

> [!review] 自动审阅
> 审阅报告: [[_review/ECSS-review.yml]]
> 审阅状态: pending
