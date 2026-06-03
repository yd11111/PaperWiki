---
type: paper
tier: deep
title: "AutoStyle-TTS: Retrieval-Augmented Generation based Automatic Style Matching Text-to-Speech Synthesis"
arxiv_id: "2504.10309"
source: "Sources/AutoStyle-TTS.pdf"
authors: [Dan Luo, Chengyuan Ma, Weiqin Li, Jun Wang, Wei Chen, Zhiyong Wu]
year: 2025
venue: "ICME 2025"
tags: [TTS, style-control, RAG, retrieval-augmented, CosyVoice, flow-matching, embedding, podcast]
concepts: ["[[Style Transfer in TTS]]", "[[Speech Factorization]]", "[[Conditional Flow Matching]]", "[[Speaker Embedding]]", "[[Emotion Control in TTS]]", "[[Global Style Tokens]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]", "MaskGCT"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["EXPRESSO", "Common Voice", "DiDiSpeech-2", "IEMOCAP", "M3ED"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: AutoStyle-TTS 处于 [[Style Transfer in TTS]] 演进线的最新阶段。现有风格控制方法从 GST (2018) 的无监督风格发现,经过 reference speech prompt (2021-2023) 和 NL descriptions (2023-2024),到 instruction-guided (2024-)。AutoStyle-TTS 提出了一个正交路线: 不改进风格表示本身,而是用 RAG 技术自动选择最合适的 style prompt,解决的是 "如何选对 prompt" 而非 "如何更好地建模风格"。
>
> **已有认知**: CosyVoice 系统已被充分分析 ([[模型库/CosyVoice|CosyVoice]], confirmed),其 LLM + OT-CFM 的 coarse-to-fine 两阶段架构和 x-vector 分离音色建模是本文 backbone 的基础。[[Speech Factorization]] (confirmed) 中的 style-timbre 解耦是本文的核心前提 — 只有将风格与音色解耦后,才能独立替换 style prompt 而不影响音色。[[Conditional Flow Matching]] (confirmed) 提供了从 speech tokens 到 mel spectrogram 的生成路径。
>
> **创新判断**: 本文的核心创新不在 TTS 模型本身(backbone 就是 CosyVoice),而在于引入 RAG 实现风格 prompt 的自动选择,这是现有概念页([[Style Transfer in TTS]] [待确认])尚未覆盖的方向。将 LLM embedding (Llama + fine-tuned PER-LLM-Embedder + Moka) 用于语音风格匹配是新颖尝试,但实验规模偏小。
>
> 检索命中: [[Speech Factorization]]✓, [[Conditional Flow Matching]]✓, [[Speaker Embedding]]✓, [[LLM-based TTS]]✓, [[模型库/CosyVoice|CosyVoice]]✓ | 过滤: [[Style Transfer in TTS]](pending-review), [[Emotion Control in TTS]](pending-review), [[Global Style Tokens]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 CosyVoice backbone 上引入 RAG 机制,根据文本内容自动从风格知识库中检索匹配的 speech style prompt,实现 podcast 场景下的动态风格切换
> - **路线**: 用户文本 → Llama(角色画像) + PER-LLM-Embedder(情景情感) + Moka(用户偏好) → 三部分 embedding 拼合 → Milvus MIPS 检索 Top-K style prompt → CosyVoice(style-timbre 解耦 TTS) → 生成语音
> - **指标**: SM-MOS 3.85/3.90 vs CosyVoice 3.35/3.38, SC-MOS 3.81/3.83 vs CosyVoice 3.48/3.51 (英/中) [Table II]; IS 1.325 vs 0.750 [Table I]; AB test ~50:50 vs 人工选择 [Fig 6]
> - **可借鉴**: (1) 三维 embedding 拆分 (profile + emotion + user preference) 捕捉不同层次风格语义; (2) 将 RAG 范式从 NLP 迁移到 TTS prompt selection; (3) 用 LLM fine-tune 做情感 embedding 提取 (PER-LLM-Embedder)
> - **局限**: 实验规模小(30 speakers, 2000 segments); 无客观风格匹配指标; 评估数据集非标准 benchmark; 对 CosyVoice backbone 的修改有限,更像应用层工作; 未开源

## 核心问题

**要解决什么**: 在 podcast/有声书等长文本合成场景中,LM-based TTS 的生成质量严重依赖 speech prompt 的选择 [§I]。现有系统要么手动选择 prompt(耗时且不一致),要么用固定风格对所有文本一刀切(无法适应上下文变化)。具体来说存在两个问题:

1. **可用性有限** (Limited availability): 传统 TTS 对所有句子应用同一风格,在长文本/对话场景中无法灵活调整语调、语速和情感表达 [§I]
2. **内容-风格不协调** (Content and style disharmony): 现有基于 speech prompt 的方法不考虑文本内容与风格的匹配度,导致合成语音缺乏自然变化和连贯性 [§I]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由三个模块组成 [§II-A, Fig 1]:

1. **Speech Style Knowledge Database**: 预构建的风格知识库,存储高质量语音片段及其 style embeddings
2. **RAG Style Selection Module**: 分析用户输入文本的风格需求,从知识库中检索最匹配的语音片段作为 style prompt
3. **Style and Timbre Decoupled TTS Module**: 接收 RAG 选出的 style prompt + 用户指定的 timbre prompt,分别提取风格和音色生成目标语音

### 关键设计选择

#### 1. Style-Timbre 解耦的 TTS Backbone [§II-B, Fig 2]

基于 CosyVoice 修改,将 style 和 timbre 信息注入分离:

- **Timbre**: 通过 CAM++ speaker encoder 提取 speaker embedding,注入 LLM 和 flow matching 两个阶段(全局音色控制) [论文原文]
- **Style**: 通过 speech tokenizer 从 style prompt 提取 speech tokens,仅注入 LLM 阶段(风格控制) [论文原文]
- **为什么只在 LLM 阶段注入 style**: [agent 解读] 因为 LLM 阶段建模的是 semantic/prosody 级别信息(与风格直接相关),而 flow matching 阶段建模的是声学细节(与音色更相关)。CosyVoice 原设计中 speech tokens 已携带风格信息,本文复用了这一特性

LLM 输入序列构造为 `[(S), v, {t}_{i∈[1:I]}, (T), {x}_{k∈[1:K]}, (E)]` [§II-B]:
- `v` = speaker embedding (timbre)
- `{t}` = text encoder 输出的文本 tokens
- `{x}` = speech tokenizer 从 style prompt 提取的 speech tokens (style)

#### 2. 风格知识库构建 [§II-C, Fig 3]

数据预处理 pipeline:
- 原始语音 → 降噪 → 说话人分离 → VAD 切分 (5-10s 片段) → 质量筛选 → ASR 转录 → 结构化存储 [论文原文]
- 最终数据: 30 speakers, 2000+ segments,包含 EXPRESSO 英文数据和高质量中文数据 [§II-C]

#### 3. 三维 Style Embedding [§II-C, Eq. 2, Fig 4]

风格 embedding 由三部分组成:

$$E_{style} = E_{profile} + E_{emotion} + E_{user}$$

- **E_profile** (角色画像): 由 Llama 3.2 从全部合成文本中提取,提供全局人物特征(如"一个温和的中年男性叙述者") [论文原文]
- **E_emotion** (情景情感): 由 PER-LLM-Embedder 从当前文本 + 角色画像中分析,捕捉逐句变化的情感 [论文原文]
- **E_user** (用户偏好): 由 Moka 编码用户指定的年龄、性别、地区等偏好 [论文原文]

**为什么拆成三部分**: [agent 解读] 三个维度分别对应风格的不同层次 — profile 保证全局一致性(防止角色突变),emotion 跟踪局部变化(跟随情节波动),user 尊重用户个性化需求。这种分层设计比单一 embedding 更可控,且 profile 提供的全局信息可以防止 emotion 匹配时的风格跳变

#### 4. 检索与匹配 [§II-C]

检索流程:
1. 判断用户 query 是否需要检索
2. 用 Llama + PER-LLM-Embedder + Moka 提取 query 的三维 embedding
3. 用 Max Inner Product Search (MIPS) 在 Milvus 索引中检索 Top-K 最相似的 style prompt [§II-C]
4. 将 Top-K style prompts 拼接后送入 TTS 模块 [§III-D]

### 训练策略

#### PER-LLM-Embedder 训练 [§II-D]

- Base model: LLaMA 3.2
- 训练数据: IEMOCAP (英文) + M3ED (中文) 情感对话数据集
- 任务: 输入对话文本 + 角色画像,预测情感标签
- 超参: lr=5e-4, dropout=0.2, epochs=3, context window=5 [§II-D]
- 角色画像: 英文由 LLaMA 3.2 生成,中文由 Qwen 2.5 生成 [§II-D]

## 实验

| 指标 | AutoStyle-TTS | CosyVoice | MaskGCT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| SIM↑ | 0.750 | 0.753 | - | Common Voice + DiDiSpeech-2 | [Table I] |
| WER(%)↓ | 3.600 | 2.312 | - | Common Voice + DiDiSpeech-2 | [Table I] |
| VISQOL↑ | 4.075 | 4.083 | - | Common Voice + DiDiSpeech-2 | [Table I] |
| KL↓ | 0.160 | 0.165 | - | Common Voice + DiDiSpeech-2 | [Table I] |
| IS↑ | **1.325** | 1.007 | - | Common Voice + DiDiSpeech-2 | [Table I] |
| SM-MOS↑ (EN) | **3.85±0.13** | 3.35±0.13 | 3.85±0.13 | 12 audio samples | [Table II] |
| SC-MOS↑ (EN) | **3.81±0.12** | 3.48±0.13 | 3.81±0.12 | 12 audio samples | [Table II] |
| SM-MOS↑ (ZH) | **3.90±0.12** | 3.38±0.14 | 3.85±0.13 | 12 audio samples | [Table II] |
| SC-MOS↑ (ZH) | **3.83±0.13** | 3.51±0.14 | 3.81±0.12 | 12 audio samples | [Table II] |

**关键发现**:

1. **Style-Timbre 解耦有效** [Table I]: 在风格多样化(IS 从 1.007→1.325)的同时,SIM、VISQOL、KL 基本不变,说明风格变化没有损害音色和质量。但 WER 略升(2.3%→3.6%),可能是风格变化引入了额外的韵律/发音变异 [agent 解读]

2. **RAG 风格匹配显著提升主观指标** [Table II]: 相对 CosyVoice,SM-MOS 提升 ~0.5 分,SC-MOS 提升 ~0.3 分。与 MaskGCT 持平(英文)或略优(中文),说明 RAG 风格选择的效果与另一个 SOTA 模型相当 [论文原文]

3. **AB test 表明可替代手工选择** [Fig 6]: 自动选择 vs 手工选择的用户偏好约 50:50,说明 RAG 机制在实用场景中可以替代人工 [论文原文]

4. **消融实验** [Table III]:
   - profile + emotion 联合使用效果最佳; 仅用 profile(SM-MOS 3.40,下降 0.45)丧失逐句情感匹配; 仅用 emotion(SC-MOS 下降)丧失全局一致性 [§III-D]
   - Top-K=3 效果最佳; K=1 信息不足,K=5 因来源不一致反而降低连贯性 [§III-D]

## 局限性

1. **实验规模偏小**: 风格知识库仅 30 speakers/2000 segments,评估仅 12 audio samples/15-30 评估者。难以证明方法在大规模场景的效果 [agent 解读]
2. **缺乏客观风格匹配指标**: 风格匹配度仅用主观 MOS 评估,没有使用 emotion recognition accuracy、style classifier accuracy 等客观指标 [agent 解读]
3. **对 CosyVoice 依赖强**: 本文的 TTS 模块几乎就是 CosyVoice 原版,核心创新集中在检索侧。如果 backbone 换成其他 TTS 系统,RAG 模块是否仍有效未经验证 [agent 解读]
4. **WER 上升**: 3.6% vs 2.3%,虽然绝对值不大,但 56% 的相对增长值得关注 [Table I]
5. **三维 embedding 设计缺乏对比**: 没有与其他 embedding 提取方案(如 CLAP、wav2vec 2.0 style embeddings)做对比 [agent 解读]
6. **非标准评估协议**: SM-MOS 和 SC-MOS 是本文自定义指标,与领域通用的 SMOS/CMOS 不完全等价 [agent 解读]

## 点评

AutoStyle-TTS 提出了一个有价值的问题: 在 LM-based TTS 中,style prompt 的选择同样重要,不应被忽视。将 RAG 从 NLP 引入 TTS prompt selection 是一个直觉合理的方向,三维 embedding (profile + emotion + user) 的设计也体现了对风格建模层次性的理解。

然而,本文更像一个应用层工作(engineering contribution)而非方法层突破:
- TTS backbone 直接复用 CosyVoice,没有在模型架构上做创新
- RAG 模块本质上是标准的 embedding similarity retrieval,缺少针对 TTS 场景的特殊设计(如时序风格变化建模、风格过渡平滑化)
- 实验设计偏弱,评估规模小且缺乏客观指标
- 风格知识库的构建高度依赖数据质量和多样性,扩展性未被讨论

尽管如此,这项工作为 "如何自动化 TTS prompt selection" 提供了一个可用的 baseline 方案,消融实验也给出了有用的设计 insight(如 Top-K=3 最优、profile+emotion 缺一不可)。

## 可复用的 idea

1. **RAG for TTS prompt selection**: 将 RAG 范式应用于 TTS 的 style prompt 自动选择,可推广到其他需要 prompt 的生成任务。核心思路是构建 style knowledge base + embedding-based retrieval,替代手工选择
2. **三维 embedding 分层**: 将风格拆分为全局 profile(角色一致性)、局部 emotion(情景变化)、用户 preference(个性化)三个维度,这种分层思路可迁移到其他需要多层次控制的生成任务
3. **LLM fine-tune for emotion embedding**: 用 LLaMA 在情感对话数据上微调(PER-LLM-Embedder),输入文本 + 角色画像预测情感,作为 TTS 的风格条件。这比传统的 SER 模型更能理解上下文语义
4. **Style-timbre 分离注入策略**: style 仅注入 LLM stage (via speech tokens),timbre 注入 LLM + flow matching (via speaker embedding),实现两者独立控制。这种分阶段注入策略可参考

> [!review] 审阅待补充
> 审阅将在下一步自动执行。
