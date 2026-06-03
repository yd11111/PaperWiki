---
type: paper
tier: deep
title: "Retrieval-Augmented Dialogue Knowledge Aggregation for Expressive Conversational Speech Synthesis"
arxiv_id: "2501.06467"
source: "Sources/RAD-KECSS.pdf"
authors: [Rui Liu, Zhenqi Jia, Feilong Bao, Haizhou Li]
year: 2025
venue: "arXiv preprint (submitted to Elsevier)"
tags: [TTS, conversational-speech-synthesis, RAG, heterogeneous-graph, contrastive-learning, style-modeling, prosody, multi-granularity]
concepts: ["[[Prosody Modeling]]", "[[Style Transfer in TTS]]", "[[Emotion Control in TTS]]", "[[Global Style Tokens]]", "[[Non-autoregressive TTS]]", "[[Self-Supervised Speech Representation]]"]
models: ["[[模型库/wav2vec 2.0|wav2vec 2.0]]"]
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
>
> **谱系定位**: RADKA-CSS 属于 Conversational Speech Synthesis (CSS) 领域,以 FastSpeech 2 为 TTS backbone。它处理的核心问题 -- 对话上下文建模以驱动风格韵律 -- 与 [[Prosody Modeling]] 中的隐式韵律建模方向一致,但将建模范围从单句扩展到多轮对话。论文中使用 Wav2Vec 2.0 (IEMOCAP fine-tuned) 提取情感风格特征,x-vector 提取说话人信息,这两个组件在 [[Speaker Embedding]] 和 [[Self-Supervised Speech Representation]] 中有详细记录。
>
> **已有认知**: 知识库已覆盖风格建模的完整谱系 -- 从 GST 无监督风格发现 ([[Global Style Tokens]]) 到 reference encoder 方法 ([[Style Transfer in TTS]]),再到情感控制 ([[Emotion Control in TTS]])。CSS 作为专门子领域尚无独立概念页。RADKA-CSS 的 FastSpeech 2 backbone 属于 [[Non-autoregressive TTS]] 范式。
>
> **创新判断**: 知识库中尚无 RAG-for-CSS 或 dialogue-level heterogeneous graph 的记录。本文的核心创新 -- 将 stored dialogue 作为外部知识库并通过 RAG 检索增强当前对话的风格合成 -- 是全新方向。现有概念页覆盖了 TTS 的风格/韵律建模,但均面向单句或参考音频驱动,未涉及对话级别的多轮上下文建模。
>
> 检索命中: [[Prosody Modeling]]✓, [[Speaker Embedding]]✓ | 过滤: [[Style Transfer in TTS]][待确认], [[Emotion Control in TTS]][待确认], [[Non-autoregressive TTS]][待确认], [[Global Style Tokens]][待确认], [[Self-Supervised Speech Representation]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首次将 RAG 引入 CSS,从历史存储对话中检索场景和风格相似的对话片段,通过多粒度异构图聚合多源风格知识,使合成语音更贴合当前对话风格
> - **路线**: 当前对话 (CD) → [数据库准备: bart-large-cnn-samsum 提取摘要 + Sentence-BERT 语义向量 + Wav2Vec2.0-IEMOCAP 风格向量 + x-vector 说话人信息] → [多属性检索: 语义+风格联合 Top-K] → [多源风格知识聚合: AMgHG + TMgHG 编码 + 对比学习约束] → [Style Renderer 注入 FastSpeech 2] → HiFi-GAN → 语音
> - **指标**: N-DMOS 3.904 / S-DMOS 3.879 (vs ECSS 最佳 baseline 3.720/3.698, GT 4.448/4.498); MAE-P 0.442 / MAE-E 0.305 / MAE-D 0.130,均优于所有 baseline [Table 3]
> - **可借鉴**: (1) 多属性联合检索 (语义+风格同时考虑) 优于级联或单属性检索 [Table 5/6]; (2) 多粒度异构图 (word/sentence/dialogue 三级节点 + 四种关系) 比同构图和纯 sentence-level 更全面 [Table 4, Abl.5-6]; (3) 用 bart-large-cnn-samsum 对对话文本做摘要再向量化,实现对话级语义表示
> - **局限**: (1) 仅在 DailyTalk (2 说话人固定交替) 上验证,未验证多说话人公共场景 [§7]; (2) backbone 为 FastSpeech 2,未探索 LLM-based TTS; (3) SDSSD 的构建依赖完整的历史对话文本+音频,冷启动时无 SD 可检索; (4) 代码开源但论文仍为 preprint

## 核心问题

传统 CSS 只建模当前对话 (CD) 的历史,忽略了用户-智能体交互中积累的存储对话 (SD) 中蕴含的风格表达知识 [§1]。SD 包含与 CD 场景相似的对话片段,这些片段中的风格信息能帮助智能体更好地理解和适配当前对话风格。RADKA-CSS 要回答三个子问题: (1) 如何从 SD 中检索出场景和风格都与 CD 相似的对话? (2) 如何有效编码 CD 和检索到的 SD 以充分捕获对话风格? (3) 如何将提取的风格特征有效应用到待合成语音?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

RADKA-CSS 由四个组件组成 [§4.1, Fig 2]:

1. **Database Preparation**: 构建存储对话语义-风格数据库 (SDSSD)
2. **Multi-attribute Retrieval**: 从 SDSSD 中检索与 CD 语义和风格相似的 Top-K 对话
3. **Multi-source Style Knowledge Aggregator**: 编码并聚合 CD、SD、预测 $a_N$ 风格向量的多源知识
4. **Speech Synthesizer**: 基于 FastSpeech 2 的语音合成器,通过 Style Renderer 注入聚合后的风格知识

### 关键设计选择

#### 1. 为什么引入 Stored Dialogue (SD)?

[论文原文] 传统 CSS 仅建模当前对话历史,面对新场景时缺乏足够的风格参考。SD 包含用户-智能体早期交互中保存的对话片段,这些片段蕴含与 CD 类似场景的风格表达知识 [§1]。CSS 是一个知识密集型任务 (knowledge-intensive task),需要参考大量相似对话场景才能生成自然的对话风格语音 [§1]。

[agent 解读] 这本质上是将 RAG 的"外部知识库补偿模型知识不足"思想迁移到 CSS: 单个对话的上下文有限,通过检索外部对话库来扩展可用的风格知识。

#### 2. 数据库准备: 如何表示一组对话的语义和风格?

**语义表示** [§4.2]: 将一组对话的文本合并为长段落 → bart-large-cnn-samsum 提取对话摘要 (含说话人信息) → Sentence-BERT 向量化 → 对话级语义向量。

[agent 解读] 选择先做摘要再向量化,而非直接向量化全文,是因为对话文本通常较长且含大量冗余 (问答交替),摘要能提取核心语义特征,使检索更精准。

**风格表示** [§4.2]: 将每句音频输入 Wav2Vec2.0-IEMOCAP → 句级风格向量 → 加上 x-vector 编码的说话人信息 → 聚合为对话级风格向量。

[论文原文] 使用 IEMOCAP 微调的 Wav2Vec 2.0 提取情感维度的风格特征,同时用 x-vector 编码说话人维度的风格特征 [§4.2]。

#### 3. 多属性检索: 为什么同时用语义和风格?

[论文原文] 仅用语义检索可能选到场景相似但风格不同的对话;仅用风格检索可能选到风格相似但场景不匹配的对话。联合考虑语义和风格相似度并求和后选 Top-K,能同时保证场景和风格的匹配 [§4.3]。

**$a_N$ 向量预测器**: 推理时 $a_N$ (待合成语音) 不可用,导致 CD 的风格向量不完整。因此设计了一个 $a_N$ vector predictor (双向 GRU + 两层线性层) 来预测 $a_N$ 的句级风格向量,补全 CD 的风格表示 [§4.3]。

[agent 解读] 这是一个实用的工程设计: 训练时有 GT 音频可以提取完整风格向量,推理时缺少 $a_N$ 需要预测。虽然预测精度有限 (Abl.4 显示去除后影响最小 [Table 4]),但至少提供了风格的近似补全。

#### 4. 多粒度异构图 (MgHG): 为什么用三级节点 + 四种关系?

[论文原文] 理解一组完整对话需要从三个角度考虑: (1) 整体对话主题 (dialogue-level), (2) 每句话对整体风格的贡献 (sentence-level), (3) 每个词对句义的影响 (word-level)。这种从局部到全局的特征提取使模型能更全面地捕获对话的语义和风格特征 [§4.4.1]。

四种关系:
- 父子关系: 词属于句、句属于对话
- 兄弟关系: 同一句中的相邻词、同一对话中的相邻句

[论文原文] 这些关系使节点能充分聚合来自时序关系和粒度结构的上下文风格信息 [§4.4.1]。

**AMgHG (音频)**: word-level 用 Wav2Vec 2.0 + MFA 对齐提取,sentence-level 用 Wav2Vec2.0-IEMOCAP,dialogue-level 用句级聚合 [§4.4.1]。
**TMgHG (文本)**: word-level 用 TOD-BERT,sentence-level 用 Sentence-BERT,dialogue-level 用 bart-large-cnn-samsum 摘要 + Sentence-BERT [§4.4.1]。

**编码**: HeteroConv (四个 SAGEConv 层) 进行邻域信息聚合 → 双向 LSTM 融合句级和词级节点 → 与对话级节点拼接 → 线性层投影到 256 维 [§5.2]。

#### 5. 知识聚合: 如何融合 SD、CD、$a_N$ 风格知识?

[§4.4.3, Eq. 1]: 
1. 用 SD 文本语义特征 $H^{p-t}_{1 \to k}$ 和 CD 文本语义特征 $H^t_{cur}$ 计算 softmax 权重 $W$
2. 用权重 $W$ 加权 SD 音频风格特征 $H^{p-a}_{1 \to k}$ 得到 Retrieved Style Embedding ($R_{Semb}$)
3. 将 $R_{Semb}$、$H^t_{cur}$、$H^a_{cur}$、$V^{style}_{a_N}$ 拼接得到最终风格嵌入 $F_{Semb}$

[agent 解读] 这个设计的巧妙之处在于: 用文本语义相似度来加权音频风格特征。直觉是 -- 与当前对话语义越相近的检索对话,其风格知识越值得借鉴。这避免了直接在音频空间做加权可能引入的噪声。

#### 6. 检索对话对比学习

[§4.5] 设计了两个对比学习目标:
- $\mathcal{L}^{cl}_t$: 拉近正样本 (与 CD 相似的对话) 的语义表示,推远负样本
- $\mathcal{L}^{cl}_a$: 拉近正样本的风格表示,推远负样本

[论文原文] 正样本在场景和风格上都与 CD 相似;负样本不仅与 CD 不一致,彼此之间的场景和风格也不同 [§4.5]。

### 训练策略

- 优化器: Adam ($\beta_1=0.9, \beta_2=0.98$) [§5.2]
- G2P: Grapheme-to-Phoneme 工具包 [§5.2]
- 对齐: Montreal Forced Alignment (MFA) [§5.2]
- 重采样: 22.05 kHz; Mel: window 25ms, shift 10ms [§5.2]
- GPU: A800, batch size 16, 最优性能在 300k 步 [§5.2]
- Vocoder: 预训练 HiFi-GAN [§4.6]
- 检索数量: Z=25 (推理时的最优检索量) [§6.4, Fig 5]

## 实验

| 指标 | RADKA-CSS | ECSS (最佳 baseline) | DailyTalk | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| N-DMOS (↑) | **3.904** | 3.720 | 3.453 | 4.448 | DailyTalk | [Table 3] |
| S-DMOS (↑) | **3.879** | 3.698 | 3.434 | 4.498 | DailyTalk | [Table 3] |
| MAE-P (↓) | **0.442** | 0.505 | 0.530 | - | DailyTalk | [Table 3] |
| MAE-E (↓) | **0.305** | 0.332 | 0.467 | - | DailyTalk | [Table 3] |
| MAE-D (↓) | **0.130** | 0.134 | 0.204 | - | DailyTalk | [Table 3] |

### 消融实验关键发现 [Table 4]

| 消融 | N-DMOS 下降 | S-DMOS 下降 | 启示 |
| --- | --- | --- | --- |
| Abl.1: w/o SD 风格知识 | -0.197 | -0.185 | **SD 检索是最大贡献者**: 去掉后影响最大 |
| Abl.6: w/o 多粒度 | -0.219 | -0.196 | 多粒度节点比纯 sentence-level 更全面 |
| Abl.5: w/o 异构图 | -0.172 | -0.155 | 异构图优于同构图 |
| Abl.8: w/o 对比学习 | -0.147 | -0.180 | 对比学习显著增强风格区分 |
| Abl.4: w/o $a_N$ 风格向量 | -0.129 | -0.128 | 预测的 $a_N$ 风格向量贡献最小 |
| Abl.9: w/ GT (Retrieved) | +0.058 | +0.027 | 更好的检索能进一步提升性能 |

### 检索方案对比 [Table 5/6]

Rs.1 (语义+风格联合) 在所有 Recall 指标上最优: R@1=0.450, R@5=0.800, R@10=0.900 [Table 6]。CMOS 对比显示 Rs.1 优于所有单属性和级联检索方案,仅低于 GT Top-K (Rs.7) [Table 5]。

### 检索数量分析 [Fig 5]

Z (检索对话数) 从 1 增至 25 时风格相似度从 0.617 升至 0.797;Z=25-32 区间达到峰值;Z>32 后相似度下降 [§6.4]。选择 Z=25 平衡性能和计算复杂度。

## 局限性

1. **单一数据集验证**: 仅在 DailyTalk (2 说话人固定交替、高质量录制) 上实验,未验证多说话人或真实噪声环境下的表现 [§7]
2. **不支持公共场景**: 论文明确承认不支持多说话人交替互动的公共场景,仅适用固定用户-智能体交互 [§7]
3. **backbone 较老**: 使用 FastSpeech 2 作为 TTS backbone,未探索 LLM-based TTS 或 flow matching 等更现代的声学模型
4. **冷启动问题**: [agent 解读] SDSSD 依赖历史对话积累,新用户/新场景缺乏存储对话时无法发挥 RAG 优势
5. **预训练模型依赖多**: 系统依赖 bart-large-cnn-samsum、Sentence-BERT、Wav2Vec2.0-IEMOCAP、TOD-BERT、x-vector 等多个预训练模型,部署复杂度高
6. **评估局限**: [agent 解读] 20 名英语二语研究生的主观评估,评估者同质性较高,可能不代表母语者感知

## 点评

**贡献与定位**: RADKA-CSS 是首个将 RAG 引入 CSS 的工作,思路清晰且有说服力 -- CSS 确实是一个需要外部知识辅助的场景,单靠当前对话历史难以充分建模对话风格。实验设计也较扎实,5 个 baseline、9 个消融、7 个检索方案对比提供了充分的验证。

**方法评价**: 多粒度异构图 (MgHG) 是本文最具技术含量的组件,三级节点 + 四种关系的设计在 CSS 领域是新颖的。消融实验 (Abl.5 vs Abl.6) 证实了异构结构和多粒度都有独立贡献。但整体系统偏重工程堆叠 -- 多个预训练模型 (bart-large-cnn-samsum、Sentence-BERT、Wav2Vec2.0、TOD-BERT、x-vector) 和多个自设计模块的组合,增加了复现和部署难度。

**局限性评价**: 最大的局限是仅在 DailyTalk (2 说话人) 上验证。CSS 真实场景通常涉及多说话人、不同领域、噪声环境,当前结果的泛化性存疑。此外,backbone 使用 FastSpeech 2 使得生成质量存在天花板 (GT N-DMOS 4.448 vs 合成 3.904,差距仍明显),如果换用 LLM-based TTS backbone,聚合的风格知识可能发挥更大作用。

**在领域中的位置**: CSS 是 TTS 的一个重要但相对小众的子方向,RADKA-CSS 提出的 RAG 思路为这个方向引入了新范式。与同期工作 (DiffCSS, ICASSP 2025) 相比,RADKA-CSS 侧重外部知识检索增强,DiffCSS 侧重扩散模型的多样性生成,两者互补。

## 可复用的 idea

1. **对话级语义向量**: bart-large-cnn-samsum 摘要 + Sentence-BERT 向量化的两步法,适用于任何需要对话级语义表示的场景 (对话检索、对话聚类等)
2. **多属性联合检索**: 将语义和风格相似度直接相加后排序,简单但有效,可推广到其他需要多维度匹配的检索任务
3. **文本语义加权音频风格**: 用 softmax(文本相似度) 加权音频风格特征的融合方式 [Eq. 1],适用于多模态知识聚合
4. **MgHG 的三级节点设计**: word/sentence/dialogue 三级粒度 + 父子/兄弟关系的图结构,可迁移到其他多层级文本/音频建模任务
5. **$a_N$ 向量预测器**: 推理时补全缺失信息的简单但实用的设计模式

> [!review] 审阅待补充
> 审阅报告将在下一步生成。
