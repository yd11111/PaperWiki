---
type: paper
tier: deep
title: "Anatomy of the Modality Gap: Dissecting the Internal States of End-to-End Speech LLMs"
arxiv_id: "2603.01502"
source: "Sources/AnatomyofModalityGap.pdf"
authors: [Ming-Hao Hsu, Xueyao Zhang, Xiaohai Tian, Jun Zhang, Zhizheng Wu]
year: 2026
venue: "arXiv preprint"
tags: [speech-LM, modality-gap, interpretability, CKA, speech-representation, reasoning, multimodal]
concepts: ["[[ModalityAdaptationforSpeechLLM]]", "[[SpeechLanguageModel]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[AudioUnderstanding]]", "[[Self-SupervisedSpeechRepresentation]]", "[[SemanticvsAcousticTokens]]"]
models: ["Qwen2.5-Omni-7B", "MiniCPM-o 2.6", "Qwen2-Audio-7B", "LLaMA-Omni"]
tasks: []
datasets: ["SpeechMMLU", "VoiceBench BBH"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 Speech-LLM 可解释性/诊断研究。已有知识库中,[[SpeechLanguageModel]] 页详细记载了 LSLM 的三大组件 (tokenizer + LM + vocoder) 和分类体系,[[ModalityAdaptationforSpeechLLM]] 页记载了语音编码器输出如何映射到 LLM 输入空间的三种方法 (Conv Downsampling, CTC Compression, Q-Former)。本文关注的恰好是这些 adapter 输出后,语音表征在 LLM 内部逐层演化的动态过程 -- 这是已有页面尚未覆盖的诊断视角。
>
> **已有认知**: [[SemanticvsAcousticTokens]] 页已记录语音的信息密度低于文本 (语音帧率 25-50 Hz vs 语义词率 2-5 Hz),以及语义 token 与文本对齐良好但缺乏声学细节的 trade-off。本文从 LLM 内部机制层面解释了这种信息密度差异如何导致推理性能下降。[[Speech-LLMIntegrationTaxonomy]] [待确认] 记载了 latent-representation-based 路线中 speech-text 表征对齐的挑战,本文正是对这一挑战的机制级诊断。
>
> **创新判断**: 已有工作 (Mousavi et al. 2025, Xiang et al. 2025) 从几何对齐角度量化 modality gap,本文的独创性在于: (1) 提出三阶段框架将静态几何分析升级为动态推理轨迹分析; (2) 用因果干预 (文本注入冗余、统计校准) 验证假设; (3) 首次将 LLM 可解释性工具 (Logit Lens, probing) 应用于语音推理任务。
>
> 检索命中: [[SpeechLanguageModel]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[ModalityAdaptationforSpeechLLM]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review), [[AudioUnderstanding]](pending-review), [[Self-SupervisedSpeechRepresentation]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过逐层表征分析,发现 Speech LLM 的 modality gap 根源不是几何对齐不足,而是语音冗余导致的 Information Dilution -- 语义信息分散在过多 token 中,阻碍了 LLM 晚期层的决策锐化
> - **路线**: 语音/文本输入 → 四个开源 LSLM → 逐层提取 hidden states → DTW 对齐 speech-text tokens → Cross-layer CKA + Logit Lens + Probing + Attention 分析 → 三阶段框架 (Structural Transformation → Semantic Smearing → Decision Instability)
> - **指标**: Qwen2.5-Omni S2T vs T2T: SpeechMMLU -6.7%, BBH -3.9%; MiniCPM-o: SpeechMMLU -13.8%, BBH -10.2%; 输入层统计校准导致 BBH -15.5% 崩溃; 文本注入 8x 冗余: SpeechMMLU 73.9%→51.2%, BBH 62.4%→53.5% [Fig 1, Table 1, Table 3]
> - **可借鉴**: (1) Cross-layer CKA + DTW 对齐的分析框架可迁移到任何多模态 LLM 的模态差异诊断; (2) "注入冗余到文本"的因果验证思路 -- 通过降低文本信息密度来模拟语音特性; (3) KV token merging 作为推理时轻量级缓解方案 (cosine>0.90 阈值)
> - **局限**: 纯诊断性工作,未提出有效的工程解决方案; KV token merging 仅带来 +0.1~+0.5 pp 改善; 文本冗余注入是离散的,与语音连续冗余机制不同; 仅评估 4 个模型,缺乏对 discrete-token-based LSLM 的分析

## 核心问题

本文试图回答: **为什么端到端 Speech LLM 在语音输入下的推理和知识任务性能系统性地低于文本输入,即使它能正确转写语音?**

已有工作将这一 "modality gap" 归因于语音与文本表征的几何不对齐 (cosine distance, Euclidean distance),并尝试通过更好的 projector 或对比损失来弥合。本文挑战了这一假设,论证 geometric alignment 是必要但不充分条件,真正的瓶颈在于语音的信息冗余阻碍了 LLM 晚期层的决策形成。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文是一篇**诊断性分析论文**,不提出新模型,而是设计分析框架来解剖 modality gap 的内部机制。

**实验设置** [§3.2]:
- 四个开源端到端 LSLM: Qwen2.5-Omni-7B, MiniCPM-o 2.6, Qwen2-Audio-7B-Instruct, LLaMA-Omni
- 两个 benchmark: SpeechMMLU (知识密集型, 57 学科) 和 VoiceBench BBH (多步推理, 4 个任务)
- 对同一语义内容分别用文本 (T2T) 和语音 (S2T) 输入,比较逐层表征演化

**分析工具链** [§3.3]:
1. **Speech-Text Token Alignment**: 用 DTW 在 cosine similarity 矩阵上找最优单调对齐路径,解决语音帧数远多于文本 token 数的问题 [§3.3.1]
2. **Cross-layer CKA**: Linear CKA with double-centering,衡量任意两层 speech/text 表征的结构相似性 [§3.3.2]
3. **Standardized L2 Distance**: 标准化后的欧氏距离,衡量直接几何接近度 [§3.3.2]
4. **Layer-wise Token Norm**: 各层表征的平均 L2 范数,检测模态间幅度差异 [§3.3.2]
5. **Logit Lens**: 将中间层 hidden states 投射到词表空间,观察逐层 "决策形成" 过程
6. **Linear/MLP Probing**: 在各层 hidden states 上训练分类器,测试信息是否被编码

### 三阶段框架

论文的核心贡献是将 LSLM 对语音输入的处理分解为三个阶段 [§4]:

**Phase I: Structural Transformation (结构变换)** [§4.1]

CKA 热力图的早期层 (前 ~5 层) 显示"暗区" -- 语音表征与任何文本层都极低相似度 [Fig 2]。这说明 speech adapter 输出的表征与文本 embedding 处于完全不同的流形上,早期 Transformer 层执行的是非线性投射,而非简单对齐 [论文原文]。

**关键实验**: 在 Layer 0 (adapter 输出) 对语音表征做 mean/std 统计校准以匹配文本分布,结果 BBH 性能从 62.3% 暴跌到 46.8% (-15.5%),SpeechMMLU 从 70.2% 暴跌到 23.0% (-47.2%) [Table 1, Table 4]。这排除了"分布偏移"解释 -- 结构不匹配是几何性质的,需要非线性变换,简单仿射变换反而破坏了早期层学到的投射 [论文原文]。

**Phase II: Semantic Smearing (语义弥散)** [§4.2]

中间层出现宽阔的对角线对齐带,而非尖锐的一对一对角对应 [Fig 2, Fig 3]。每个 speech layer 大约对齐到 ~6 个 text layer [论文原文]。这是因为语音的语义信息分散在多个冗余帧中 -- 一个语义单元跨越多个 speech token,导致 CKA 对齐呈现"弥散"特征 [论文原文]。

**微观证据 -- Attention Dispersion** [Table 2, Fig 4]: 在决策 token 的 attention 中,文本仅需 8 个 token 即可覆盖 90% 的 attention mass (Cov_0.90),而语音需要 101 个 token。最大单 token attention 从 0.64 (text) 降到 0.11 (speech),normalized entropy 从 0.36 升到 0.66 [Table 2]。这直接将宏观的 CKA 弥散与微观的注意力分散联系起来 [论文原文]。

**Layer-wise 变换保持同步** [Fig 5]: 尽管存在弥散,speech 和 text 的逐层 update vector cosine similarity 很高,说明两种模态在每层执行的变换是同步的 -- 弥散是语音的结构性质,不是处理路径偏离 [论文原文]。

**因果验证 -- 文本注入冗余** [Table 3]: 将文本中每个词重复 r 次 (r=1,2,4,8) 输入 Qwen2.5-Omni,SpeechMMLU 从 73.9% 降到 51.2%,BBH 从 62.4% 降到 53.5%,同时 decision-step entropy 单调上升。这确认冗余本身就足以降低性能 [论文原文]。[agent 解读] 但需注意,文本离散重复与语音连续弥散的机制不同,这一因果验证有局限性。

**Phase III: Decision Instability (决策不稳定)** [§4.3]

尽管中间层对齐,语音表征在晚期层未能完成文本所经历的"决策锐化" (residual sharpening)。Best-match text layer 在晚期 speech layer 停止增长,SpeechMMLU 上 Qwen2.5-Omni 的最终 speech layer 仅对齐到 text layer 22 (而非 28),存在 6 层 gap [Fig 3] [论文原文]。

**排除信息丢失**: Linear/MLP probing 在各层 hidden states 上的分类准确率在中晚期保持高值,即使模型最终 S2T 输出错误 [Fig 7]。语义信息**存在**于表征中,但模型**无法利用**它 [论文原文]。

**Logit margin 分析** [Fig 6]: 对 "only_t2t" 样本 (文本正确但语音错误),文本的 logit margin (正确答案 logit - 最强竞争者 logit) 在晚期层变为强正值,而语音 margin 保持负值直到最终层。这是"决策不稳定"的直接签名 -- 语音无法将正确答案从竞争者中分离出来 [论文原文]。

**全局 entropy 并非解释**: Logit Lens 显示 speech 和 text 的 projected-logit entropy 曲线高度重叠 [Fig 8],排除"语音全局更不确定"的假设。问题不在于整体信噪比,而在于**选项级分离**的失败 [论文原文]。

### 关键设计选择

**为什么用 CKA 而非 cosine similarity?** [§3.3.2] CKA 对不可逆线性变换不变,能捕捉结构相似性而非简单的向量相似性。论文补充用 Standardized L2 Distance 衡量直接几何接近度 [论文原文]。

**为什么用 DTW 对齐而非 forced alignment?** [§3.3.1] 论文需要在 hidden states 层面 (而非音素层面) 对齐,且 DTW 最大化 cosine similarity 的累积对齐分数,适合逐层分析。DTW 敏感性分析 (Appendix A.1) 显示 base layer >= 5 时路径稳定 [Fig 9] [论文原文]。

**模型选择逻辑** [§3.2.1]: 优先选择有非trivial S2T 能力的模型 (Qwen2.5-Omni, MiniCPM-o) 做深度诊断,确保诊断的是 reasoning gap 而非 recognition failure。Qwen2-Audio 和 LLaMA-Omni 作为对照,展示更基础的 Representation Failure 模式 [论文原文]。

### 训练策略

不适用 -- 本文为诊断性分析,不涉及模型训练。

### 补充分析: Qwen2-Audio 的 Representation Failure [Appendix F]

Qwen2-Audio 表现出定性不同的失败模式: CKA 热力图全层低相似度,无中间层对齐带 [Fig 17]。论文将此命名为 Representation Failure -- 语音表征**从未**进入文本对齐的子空间,与 Qwen2.5-Omni/MiniCPM-o 的 Phase III Decision Instability 是不同层级的失败 [论文原文]。

SpeechMMLU 上 Qwen2-Audio 几乎 chance-level,而 BBH 仍有部分功能。论文解释: 知识检索 (SpeechMMLU) 依赖 FFN 层的 key-value memory (Geva et al., 2021; Meng et al., 2022),需要精确的表征对齐才能触发正确记忆;而推理 (BBH) 更依赖中间表征的组合操作,对精确对齐的要求较低 [论文原文]。

### Token Merging 实验 [Appendix C]

KV token merging: 在指定层范围内,合并 key vector cosine similarity > 0.90 的 token,最大合并比例 0.10-0.25。最佳配置 (layers 7-21) 在 SpeechMMLU 上 +0.5 pp,BBH 上 +0.5 pp [Table 5]。改善幅度微小,但方向与假设一致 [论文原文]。

[agent 解读] token merging 效果有限可能说明: 单纯减少 token 数量不够,关键是提取和浓缩语义信号,而非简单合并相似 token。语音 token 即使合并后仍携带非语义信息 (prosody, acoustic residue),这与文本 token 的高度符号化不同。

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Qwen2.5-Omni T2T Acc | 76.9% | - | SpeechMMLU | [Fig 1] |
| Qwen2.5-Omni S2T Acc | 70.2% | T2T (gap -6.7%) | SpeechMMLU | [Fig 1] |
| Qwen2.5-Omni T2T Acc | 66.2% | - | VoiceBench BBH | [Fig 1] |
| Qwen2.5-Omni S2T Acc | 62.3% | T2T (gap -3.9%) | VoiceBench BBH | [Fig 1] |
| MiniCPM-o T2T Acc | 65.1% | - | SpeechMMLU | [Fig 1] |
| MiniCPM-o S2T Acc | 51.3% | T2T (gap -13.8%) | SpeechMMLU | [Fig 1] |
| MiniCPM-o T2T Acc | 64.2% | - | VoiceBench BBH | [Fig 1] |
| MiniCPM-o S2T Acc | 54.0% | T2T (gap -10.2%) | VoiceBench BBH | [Fig 1] |
| Input calibration BBH Acc | 46.8% | 62.3% (baseline) | BBH | [Table 1] |
| Output calibration BBH Acc | 61.2% | 62.3% (baseline) | BBH | [Table 1] |
| Text redundancy r=8 Acc | 51.2% | 73.9% (r=1) | SpeechMMLU | [Table 3] |
| Text redundancy r=8 Acc | 53.5% | 62.4% (r=1) | VoiceBench BBH | [Table 3] |
| Cov_0.90 (text) | 8 tokens | - | SpeechMMLU | [Table 2] |
| Cov_0.90 (speech) | 101 tokens | 8 tokens (text) | SpeechMMLU | [Table 2] |
| KV merge best config | +0.5 pp | baseline | SpeechMMLU + BBH | [Table 5] |

## 局限性

1. **纯诊断,无有效解法**: 论文明确声明是 diagnostic 而非 prescriptive [Appendix H]。KV token merging 仅 +0.1~0.5 pp 改善,远不足以解决问题。
2. **因果验证的有限性**: 文本冗余注入是离散的词级重复,与语音的连续时域弥散机制不同 [Appendix H]。论文承认 "the discrete manipulation does not capture all aspects of how speech encodes information" [Appendix H]。
3. **模型覆盖有限**: 仅 4 个 latent-representation-based 模型 (adapter + frozen/tuned LLM),未分析 discrete-token-based LSLM (如 SpeechGPT, SPIRIT-LM),后者的 modality gap 机制可能不同。[agent 解读]
4. **Token redundancy vs token quality**: 论文自己指出,即使合并 token,语音 token 仍携带 noise, prosodic variation 等非语义信息,这是 token quality 问题而非 token quantity 问题 [Appendix H]。
5. **Benchmark 局限**: SpeechMMLU 和 BBH 都是选择题形式,modality gap 在开放式生成任务上的表现未被分析。[agent 解读]
6. **TTS 合成语音**: SpeechMMLU 和 BBH 的语音输入是 TTS 合成的,未测试真实人声,后者可能有更大的 acoustic variability。[agent 解读]

## 点评

**优势**:
- **分析框架的系统性极强**: 三阶段框架不仅描述现象 (CKA),还提供因果验证 (统计校准实验、冗余注入实验、probing),以及微观机制解释 (attention dispersion)。这种"描述-验证-机制"三步法是做诊断性研究的标杆。
- **关键洞察有实践价值**: "input-level calibration is harmful" (Table 1) 直接否定了一类流行的研究方向 (projector-level alignment),可能影响后续工作的方向选择。
- **实验设计的巧思**: 文本注入冗余实验 (Table 3) 是一个聪明的因果干预 -- 将文本"降级"到类似语音的信息密度来验证假设。

**不足**:
- **与已有认知的增量有多大?** 语音冗余 (Zuo et al. 2025) 和模态不对齐 (Mousavi et al. 2025) 都是已知问题。本文的核心增量在于 "three-phase framework" 和 "calibration is harmful",但这些发现的practical implication 仍不清晰。
- **缺少与 cascade 系统的对比**: 如果 ASR+LLM 管线在 SpeechMMLU 上的 gap 更小 (因为 ASR 输出是文本),那么 end-to-end 路线的 modality gap 是否 inherent? 论文未讨论这一对照。[agent 解读]
- **KV merging 实验不够深入**: 仅试了一种 merging 策略和一组超参数,未尝试更先进的 token compression 方法 (如 learned abstractors)。

**对领域的影响**:
本文最重要的贡献是从"对齐到什么程度"转向"对齐之后还需要什么"的视角转换。它为 Speech-LLM 的 modality adaptation 研究提供了更精细的诊断工具,并明确指出未来方向应聚焦于 token-level 或 temporal granularity 的解决方案 (adaptive token merging, hierarchical pooling),而非 feature-level matching。

## 可复用的 idea

1. **Cross-layer CKA + DTW 诊断框架**: 可直接应用于任何 multimodal LLM 中分析模态间的表征动态。DTW 解决长度不匹配问题,CKA 不受线性变换影响。关键配置: base layer >= 5,cosine similarity metric,unconstrained DTW [Table 7]。
2. **"注入冗余"的因果验证范式**: 当怀疑某种输入特性 (如冗余) 导致性能下降时,可在对照模态中人工注入该特性来验证因果关系。这一思路可推广到其他模态差异的研究。
3. **Information Dilution 视角**: 对于 speech adapter 设计,应关注 per-token information density 而非仅关注整体对齐质量。这提示 adapter 设计应包含 token compression 或 semantic condensation 模块,而非仅做 downsampling + linear projection。
4. **Probing 排除信息丢失**: 当模型表现差时,先用 probing 确认信息是否被编码在 hidden states 中 -- 如果是,问题在 readout 而非 encoding,优化方向应不同。
5. **Attention Cov_0.90 指标**: 用 "覆盖 90% attention mass 所需的最少 token 数" 量化注意力集中度,比 entropy 更直观地反映决策信号的分散程度。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 三阶段框架因果逻辑清晰,关键设计选择有 WHY 解释,速查卡片可借鉴具体 |
> | 可信赖 | pass | 数字标注覆盖率 >90%,指标使用正确,实验表格 baseline 对照明确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率高,无未标注的强断言 |
> | 可定位 | pass | KB 背景谱系定位具体,创新判断有 Mousavi/Xiang 对比基准 |
> | 不污染 | pass | 跳过 KB 反向更新,无污染风险 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/AnatomyofModalityGap-review.yml`
