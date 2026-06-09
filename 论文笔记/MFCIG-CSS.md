---
type: paper
tier: deep
title: "MFCIG-CSS: Multimodal Fine-grained Context Interaction Graph Modeling for Conversational Speech Synthesis"
arxiv_id: "2509.06074"
source: "Sources/MFCIG-CSS.pdf"
authors: [Zhenqi Jia, Rui Liu, Berrak Sisman, Haizhou Li]
year: 2025
venue: "EMNLP 2025"
tags: [TTS, conversational-speech-synthesis, graph-neural-network, multimodal, word-level-prosody, context-modeling, GraphSAGE]
concepts: ["[[ProsodyModeling]]", "[[EmotionControlinTTS]]"]
models: ["MFCIG-CSS"]
tasks: []
datasets: ["DailyTalk"]
kb_context_sources: 2
status: draft
created: 2026-06-09
updated: 2026-06-09
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页: [[ProsodyModeling]]; 1 个待确认实体页: [[EmotionControlinTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]]✓ | 过滤: [[EmotionControlinTTS]](pending-review) | 未命中但可能相关: 无

**谱系定位**: MFCIG-CSS 属于 Conversational Speech Synthesis (CSS) 子领域,以 FastSpeech 2 为 TTS backbone,聚焦对话历史中 **词级语义和韵律的交互建模**。在 [[ProsodyModeling]] 的粒度分类中,本文对应 "Word/Syllable level" 层级,但区别于 Sun et al. 2020 的单句词级韵律建模,将粒度拓展到对话多轮历史中的词级交互。在 CSS 演进链上,前序工作 Base-CTTS (句级文本上下文) → FCTalker (粗细粒度文本编码) → M2-CTTS (多模态多尺度) → CONCSS (对比学习) → MSRGCN-CSS (多尺度关系图) → ECSS (异构图情感渲染) → I3-CSS (模态内+模态间交互) 均建模 **utterance-level** 交互; MFCIG-CSS 首次将交互建模下沉到 **word-level**,使用双图结构 (SIG + PIG) 显式编码词级语义/韵律如何影响后续话语。

**已有认知**:
- [[ProsodyModeling]] (confirmed) 记录了从显式 variance adaptor 到隐式生成模型的韵律建模全谱系,MFCIG-CSS 属于"显式多粒度建模"路线的新实例 — 在 word/utterance 两级间建立图交互而非简单拼接。
- [[EmotionControlinTTS]] (pending-review) 记录了对话情感建模方法,ECSS (Liu et al., 2024a) 已在该页中被引用为"基于多源知识异构图增强情感表达"的代表工作。MFCIG-CSS 与 ECSS 同组,是从 utterance-level heterogeneous graph 到 word-level interaction graph 的粒度细化。
- 同组前序: [[论文笔记/RADKA-CSS|RADKA-CSS]] 使用 RAG + 多粒度异构图; I3-CSS (Jia & Liu, 2024) 使用模态内+模态间 utterance-level 交互。MFCIG-CSS 在 I3-CSS 基础上将交互粒度从 utterance 下沉到 word。

**创新判断**: MFCIG-CSS 的核心新颖性在于双图结构 (SIG/PIG) 的设计 — 在对话历史中为每句话建立 word→next-utterance 的图边,使 GNN 能传播词级信息对后续话语的影响。相比 MSRGCN-CSS 的多尺度关系图 (sentence-level 节点) 和 ECSS 的异构图 (知识源节点),MFCIG-CSS 的节点粒度最细 (word-level text + word-level speech + utterance-level),且明确区分语义交互和韵律交互两条路径。

## 速查

> [!summary] 速查
> - **一句话**: 首个从词级粒度建模对话历史中语义-韵律交互的 CSS 框架,通过双图 (语义交互图 SIG + 韵律交互图 PIG) 显式编码词级文本/语音对后续话语的影响,提升合成语音的对话韵律表现力
> - **路线**: MDH (text+speech) → TOD-BERT 词级文本特征 + Wav2Vec2.0 词级语音特征 + SentenceBERT 句级文本特征 + Wav2Vec2.0-IEMOCAP 句级语音特征 → SIG (词级→句级语义交互图, GraphSAGE 编码) → PIG (词级→句级韵律交互图, GraphSAGE 编码) → Feature Aggregator 融合 Is'+Ip'+当前文本 → FastSpeech 2 Acoustic Decoder → Vocoder → 语音
> - **指标**: N-DMOS 3.980 (+0.122) / P-DMOS 3.899 (+0.104) / MAE-P 0.439 / MAE-E 0.314 / MCD 9.53 (vs best baseline I3-CSS, DailyTalk) [Table 1]
> - **可借鉴**: (1) 用 MFA 对齐 + Wav2Vec2.0 + Average Pooling 获取词级语音特征,建立 word-level prosody node 的标准流程; (2) 对交互图设计"interaction backbone branch + special aggregation node (Is/Ip)"结构,先逐句传播再全局聚合; (3) 将语义交互和韵律交互拆分为两个独立图分别编码,避免混杂
> - **局限**: (1) 仅在 DailyTalk (20h, 2 speakers, 固定交替) 上验证,未验证多说话人真实场景; (2) backbone 为 FastSpeech 2,未探索 LLM-based TTS; (3) 图构建假设词级语义/韵律对 **下一句** 有影响,但未建模跨多句的长距离依赖; (4) 未融入更细粒度声学特征 (emotion, emphasis, pause) 到交互图中; (5) 论文未报告推理延迟和图构建开销

## 核心问题

现有 CSS 方法在对话历史 (MDH) 建模中存在两个层面的不足 [§1]:

1. **粒度不足**: 现有方法 (ECSS, I3-CSS 等) 仅建模 utterance-level 交互,忽略了对话中 **关键词** 的语义和韵律对后续话语的精细影响。论文用例子说明: "I lost my wallet" vs "I lost my pen" 中,关键词 "wallet"/"pen" 的语义差异和说话人表达这两个词时的韵律差异,共同决定了回复的语义方向和情感韵律 [§1, 第3段]。

2. **交互未显式建模**: 即使有些方法使用细粒度编码器 (如 FCTalker 的 fine-grained text encoder),也只是分别编码粗/细粒度特征然后拼接,未显式建模词级特征如何 **影响** 后续话语的语义和韵律 [§1, 第2段]。

MFCIG-CSS 要回答: 如何显式建模 MDH 中词级语义、词级韵律与后续话语语义/韵律之间的交互关系,从而帮助系统更好理解对话上下文并生成自然的对话韵律?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MFCIG-CSS 包含三个模块 [§2.2]:
1. **SIG (Semantic Interaction Graph)**: 建模词级语义/韵律对后续话语 **语义** 的影响
2. **PIG (Prosody Interaction Graph)**: 建模词级语义/韵律对后续话语 **韵律** 的影响
3. **Speech Synthesizer**: 基于 FastSpeech 2 的声学模型,将交互特征注入合成

### SIG 构建与编码

**图结构** [§2.3]: SIG Gs = (N, E) 包含三种节点类型和三种关系边:

| 节点类型 | 特征来源 | 维度 |
|----------|----------|------|
| Word-level text | TOD-BERT (Wu et al., 2020) | 256 |
| Word-level speech | Wav2Vec2.0 + MFA 对齐 + Average Pooling | 256 |
| Utterance-level text | SentenceBERT (Reimers & Gurevych, 2019) + speaker embedding | 256 |

**三个交互分支** [§2.3]:
1. **Word-level semantic interaction branch**: 词级语义节点 → 下一句句级语义节点
2. **Word-level prosody interaction branch**: 词级韵律节点 → 下一句句级语义节点
3. **Semantic interaction backbone branch**: 句级语义节点 → 下一句句级语义节点

末尾添加特殊节点 Is (初始化为零向量),用于聚合整个 MDH 的语义交互特征。

**编码过程** [§2.3, Eq.1]: 使用 GraphSAGE (Hamilton et al., 2017) 作为图卷积编码器,从第一句开始依次传播:
```
F^t_{i+1} = SAGE(F^t_i, W^t_{i,1→q}, W^s_{i,1→q}),  i ∈ [1, J)
Is = SAGE(F^t_J, W^t_{J,1→q}, W^s_{J,1→q})
Is' = Average Pooling(F^t_{1→J}, Is)
```

[agent 解读] 这个设计的核心思想是: 每句话的句级语义特征在传递到下一句时,被当前句的所有词级文本和语音特征调制。这种逐句传播机制使最终的 Is' 编码了"词级特征如何逐步影响对话语义走向"的累积交互信息。GraphSAGE 的 neighborhood aggregation 天然适合这种"当前节点被邻居节点调制"的建模需求。

### PIG 构建与编码

**与 SIG 的区别** [§2.4]:
- Backbone 分支变为 **prosody interaction backbone branch** (句级 speech 节点链)
- 句级语音节点使用 Wav2Vec2.0-IEMOCAP (情感识别微调版) 提取,而非 SentenceBERT
- 末尾特殊节点为 Ip (聚合韵律交互特征)

**编码过程** [§2.4, Eq.2]:
```
F^s_{i+1} = SAGE(F^s_i, W^t_{i,1→q}, W^s_{i,1→q}),  i ∈ [1, J)
Ip = SAGE(F^s_J, W^t_{J,1→q}, W^s_{J,1→q})
Ip' = Average Pooling(F^s_{1→J}, Ip)
```

[agent 解读] SIG 和 PIG 的对称设计体现了一个关键假设: 词级语义和韵律对后续话语的 **语义走向** 和 **韵律走向** 分别产生影响,两条影响路径应独立建模后再融合。这比单一图同时建模两种影响更清晰,也为消融实验提供了自然的消融单元。

### 关键设计选择

**Q: 为什么用图结构而非 Transformer attention 建模交互?**
[agent 解读] 图结构天然适合建模"哪些节点影响哪些节点"的显式关系 — 词级节点只连接到下一句的句级节点,而非全连接。这种稀疏拓扑约束比 Transformer 的全注意力更能体现论文的核心假设: 词级特征对 **紧邻的下一句** 产生直接影响。同时,GraphSAGE 的 inductive learning 特性支持变长对话。

**Q: 为什么词级语音特征用 Wav2Vec2.0 而句级语音特征用 Wav2Vec2.0-IEMOCAP?**
[agent 解读] 词级语音需要编码通用韵律信息 (pitch, energy, duration pattern),基础 Wav2Vec2.0 足够; 句级语音需要编码整体情感/风格信息,IEMOCAP 微调版在情感维度上更有判别力。这种分层特征选择在 ECSS (同组前作) 中已被验证有效。

**Q: 为什么在 backbone branch 末尾用 Average Pooling 而非 attention 聚合?**
[论文原文] 论文未解释选择原因,直接使用 Average Pooling 聚合 {F_1→J, Is/Ip} 得到最终特征 [Eq.1, Eq.2]。
[agent 解读] Average Pooling 是最简方案,在节点数有限 (平均 9.356 轮对话) 的情况下可能足够。但这也意味着所有轮次的交互贡献被等权对待,缺乏对"哪轮历史对当前合成更重要"的建模。

### Speech Synthesizer

采用与 I3-CSS (Jia & Liu, 2024) 相同架构的 speech synthesizer [§2.5]。Feature Aggregator 将 Is' 和 Ip' 加到当前文本编码 PtC 上,约束合成语音带有对话交互韵律。训练 loss 沿用 FastSpeech 2 (duration + pitch + energy + mel reconstruction) [§2.5]。

### 训练策略

在 DailyTalk 上训练 400k steps, batch size 16, 单卡 A800 [§3.2]。所有特征维度统一为 256。

## 实验

| 指标 | MFCIG-CSS | I3-CSS (best baseline) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| N-DMOS (↑) | **3.980** | 3.858 | DailyTalk | [Table 1] |
| P-DMOS (↑) | **3.899** | 3.795 | DailyTalk | [Table 1] |
| MAE-P (↓) | **0.439** | 0.450 | DailyTalk | [Table 1] |
| MAE-E (↓) | 0.314 | **0.310** | DailyTalk | [Table 1] |
| MCD (↓) | **9.53** | 11.47 | DailyTalk | [Table 1] |

**消融结果** [Table 2]:
- w/o SIG: N-DMOS 3.833 (-0.147), P-DMOS 3.793 (-0.106), MCD 11.45 (-1.92)
- w/o PIG: N-DMOS 3.824 (-0.156), P-DMOS 3.765 (-0.134), MCD 11.36 (-1.83)
- w/o SIG+PIG: N-DMOS 3.592 (-0.388), P-DMOS 3.512 (-0.387), MCD 12.31 (-2.78)

**消融分析**:
- SIG 和 PIG 都有显著贡献,移除任一图都导致所有指标下降 [§3.6]
- w/o PIG 的韵律指标 (P-DMOS, MAE-P/E) 下降更大,说明 PIG 对韵律建模贡献更直接 [§3.6, 论文原文]
- w/o SIG+PIG 的性能远低于任一单独移除,表明两者互补而非冗余 [§3.6]
- [agent 解读] w/o SIG+PIG (即去掉所有图模块) 性能甚至低于多数 baseline,这可能是因为去掉图后剩余架构即 I3-CSS backbone 在缺少词级信息输入时退化。

**评估设置**: 20 名语音方向研究生进行主观评估,均通过 CET-6/IELTS/TOEFL [§3.4]。

## 局限性

1. **单数据集验证**: 仅在 DailyTalk (20h, 2 speakers, 固定交替) 上验证,未验证多说话人、多情感或真实场景 [§5]
2. **Backbone 局限**: 基于 FastSpeech 2,未扩展到 VITS-based 或 discrete token-based 架构 [§5, 论文原文]
3. **声学特征缺失**: 交互图中未融入 emotion, emphasis, pause 等更细粒度声学特征 [§5, 论文原文]
4. **交互假设的局限**: 图中词级节点仅连接 **下一句** 的句级节点,假设影响是 one-hop 的 [agent 解读] — 实际对话中某个关键词的影响可能跨越多轮
5. **聚合方式简单**: Average Pooling 等权聚合所有轮次的交互特征,缺乏 attention-based 的重要性加权 [agent 解读]
6. **MAE-E 未最优**: 能量预测略低于 I3-CSS (0.314 vs 0.310),说明词级交互图对 energy 的建模增益有限 [Table 1]

## 点评

MFCIG-CSS 的核心贡献在于将 CSS 中的上下文交互建模从 utterance-level 细化到 word-level,这个方向是有道理的 — 人类对话中确实是关键词 (而非整句) 驱动后续回复的方向和语气。双图 (SIG/PIG) 的设计清晰地分离了语义和韵律两条影响路径。

但值得注意的局限:
- 论文的改进幅度虽然统计显著 (N-DMOS +0.122, P-DMOS +0.104),但都在 DailyTalk 这个小而简单的数据集上验证 (2 固定说话人交替,无情感标注)
- 整体框架仍基于 FastSpeech 2,在 LLM-based TTS 时代的可扩展性存疑
- 与同组 RADKA-CSS 相比,MFCIG-CSS 走的是"更深的交互建模"路线而非"更广的知识来源"路线,两者可互补

## 可复用的 idea

1. **MFA + Wav2Vec2.0 + Average Pooling 词级语音特征提取**: 对任何需要词级声学表征的任务 (prosody transfer, emphasis detection) 都可复用
2. **双图分离语义/韵律交互**: 当需要建模多种影响路径时,用独立图分别编码后融合,而非单一混合图
3. **Backbone branch + special aggregation node**: 在 GNN 编码中,用一条"主干链"串联句级节点,词级节点作为"侧输入"调制主干传播,最后用特殊聚合节点收集全局特征 — 这种图拓扑可推广到其他序列+细粒度信息融合场景
4. **特征源的层级匹配**: 词级用通用 SSL (Wav2Vec2.0)、句级用任务特定微调版 (Wav2Vec2.0-IEMOCAP),根据粒度选择合适的预训练模型

## 审阅

> [!review] 审阅 (2026-06-09, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节包含图结构设计的因果解释,关键设计选择有 Q&A 分析 |
> | 可信赖 | pass | 数字 claim 均标注 Table/Section 出处,指标名称正确 |
> | 可区分 | pass | 因果解释标注了 [论文原文] 和 [agent 解读],边界清晰 |
> | 可定位 | pass | KB 背景含完整 CSS 演进链定位 + 同组前作对比 |
> | 不污染 | pass | 反向更新仅追加 key_papers,不改定义/related_concepts |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/MFCIG-CSS-review.yml`
