---
type: paper
tier: deep
title: "DiffCSS: Diverse and Expressive Conversational Speech Synthesis with Diffusion Models"
arxiv_id: "2502.19924"
source: "Sources/DiffCSS.pdf"
authors: [Weihao Wu, Zhiwei Lin, Yixuan Zhou, Jingbei Li, Rui Niu, Qinghua Wu, Songjun Cao, Long Ma, Zhiyong Wu]
year: 2025
venue: "arXiv preprint"
tags: [TTS, conversational-speech-synthesis, diffusion, prosody, diversity, LM-based-TTS, context-modeling]
concepts: ["[[DiffusionModel]]", "[[Diffusion-basedTTS]]", "[[ProsodyModeling]]", "[[CodecLanguageModel]]", "[[LLM-basedTTS]]", "[[SpeakerEmbedding]]", "[[NaturalLanguageDescriptionforTTS]]"]
models: ["[[NaturalSpeech3]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[ProsodyModeling]], [[LLM-basedTTS]], [[SpeakerEmbedding]]; 3 个待确认实体页: [[Diffusion-basedTTS]], [[CodecLanguageModel]], [[NaturalLanguageDescriptionforTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[Diffusion-basedTTS]](pending-review), [[CodecLanguageModel]](pending-review), [[NaturalLanguageDescriptionforTTS]](pending-review) | 未命中但可能相关: 无

**谱系定位**: DiffCSS 处于 Conversational Speech Synthesis (CSS) 与 LM-based TTS 的交叉地带。在 Prosody Modeling 的演进线上,传统 CSS 系统 (Guo et al. 2021 GRU-based; Li et al. 2022 DialogueGCN) 使用确定性韵律预测,面临 one-to-many mapping 问题 — 即同一对话上下文可对应多种合理韵律。DiffCSS 引入 diffusion model 解决这一问题,属于知识库中 [[ProsodyModeling]] "生成模型隐式建模韵律分布" 分支的新实例,但将其拓展到 **对话上下文条件化** 场景。

**已有认知**:
- [[ProsodyModeling]] (confirmed) 指出韵律建模的 one-to-many 问题是核心挑战,传统方法通过 VAE/Flow/Diffusion 等生成模型隐式建模韵律分布来缓解 over-smoothing。DiffCSS 是该思路在 CSS 中的具体应用。
- [[LLM-basedTTS]] (confirmed) 记录了 LM-based TTS 的典型架构 (decoder-only transformer + codec tokens),以及 Hybrid 架构 (LLM + Flow/Diffusion) 的趋势。DiffCSS 的 TTS backbone 基于 ParlerTTS,属于 LM-based TTS 范式。
- [[SpeakerEmbedding]] (confirmed) 描述了 speaker embedding 在 TTS 中的注入方式,包括 cross-attention、concatenation 等。DiffCSS 将 speaker embedding 与 prosody embedding 组合后通过 cross-attention 注入 TTS backbone。
- [[Diffusion-basedTTS]] [待确认] 梳理了 diffusion 在 TTS 声学模型中的应用,但主要关注 text→mel 的直接生成。DiffCSS 的创新在于将 diffusion 用于 **韵律嵌入** 的生成而非直接的声学特征生成。
- [[NaturalLanguageDescriptionforTTS]] [待确认] 介绍了 Parler-TTS 作为 text-description-based TTS,DiffCSS 将其改造为 prosody-controllable backbone。

**创新判断**: DiffCSS 的核心新颖性在于将 diffusion model 从传统的"直接生成声学特征"转向"生成韵律嵌入条件",配合 LM-based TTS backbone,实现对话韵律的多样性。相比 KB 中已有的 Diffusion-based TTS (Grad-TTS, ProDiff 等直接在 mel space 做 diffusion) 和 Prosody Modeling (FastSpeech 2 显式预测、VITS flow-based 隐式),DiffCSS 是首个将 diffusion 专用于 CSS 韵律多样性的工作。

## 速查

> [!summary] 速查
> - **一句话**: 首个将 diffusion model 引入 CSS 的框架,通过扩散韵律预测器从多模态对话上下文中采样多样韵律嵌入,配合 LM-based TTS backbone 合成富有表现力的对话语音
> - **路线**: 多模态对话上下文 (text+speech) → Sentence-T5 文本编码 + FACodec 韵律提取 → Diffusion-based 韵律预测器 (Transformer encoder denoiser) → 韵律嵌入 → Prosody-enhanced ParlerTTS (decoder-only transformer + DAC codec) → 语音
> - **指标**: E-MOS 3.602 / C-MOS 3.574 (vs GRU 3.209/3.177); NDB 4 / JSD 0.036 (vs GRU 16/0.227, vs Transformer 14/0.181) [Table I, DailyTalk]
> - **可借鉴**: (1) 用 diffusion 建模韵律嵌入空间而非直接建模声学特征,解耦韵律多样性与音质; (2) 用 learnable query + cross-attention 将变长韵律特征压缩为固定长度嵌入; (3) 两阶段训练策略 (先训 TTS backbone 再冻结训韵律预测器) 保证模块独立可控
> - **局限**: 仅在 DailyTalk (20h, 2 speakers) 上验证; 无真实对话场景评估; ParlerTTS backbone 本身并非 SOTA; 未开源代码

## 核心问题

DiffCSS 要解决的核心问题是 **对话语音合成中韵律预测的多样性缺失**。

在对话场景中,同一段文本在相同对话上下文下可以有多种合理的韵律表达方式 (不同的情感、语气、强调模式),这是一个 one-to-many mapping 问题 [§I]。现有 CSS 系统 (如 GRU-based、DialogueGCN-based) 使用确定性的韵律预测,只能为每个上下文生成一种韵律,导致合成语音缺乏多样性和表现力。此外,现有 CSS 系统几乎都基于传统 TTS backbone (如 FastSpeech 2),限制了合成语音的自然度和质量 [§I]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DiffCSS 由两个核心组件组成 [§II, Fig 1]:

1. **Prosody-enhanced ParlerTTS** (TTS backbone): 基于 ParlerTTS 改造的 LM-based TTS,接受韵律嵌入作为额外条件,合成高质量语音
2. **Diffusion-based Context-aware Prosody Predictor**: 从多模态对话上下文中,通过扩散过程采样多样的韵律嵌入

两者通过韵律嵌入 (prosody embedding) 连接: 推理时,韵律预测器从高斯噪声出发,经去噪过程生成韵律嵌入,TTS backbone 据此合成语音 [§II.C]。

### 关键设计选择

#### 1. 为什么用 diffusion 建模韵律而不是直接建模声学特征?

[论文原文] 作者的动机是利用 diffusion model 在捕捉复杂数据分布和生成多样输出方面的成功 (引用 CV 领域的 DDPM, IDDPM 等) [§I],将这一能力用于解决 CSS 中的 one-to-many 韵律映射问题。通过在韵律嵌入空间 (而非 mel/waveform 空间) 做 diffusion,可以专注于韵律多样性,而将音质保证交给 LM-based TTS backbone。

[agent 解读] 这个设计选择巧妙地解耦了两个目标: diffusion 负责多样性,LM backbone 负责音质。如果直接在声学空间做 diffusion,多样性和音质会耦合在一起,难以同时优化。

#### 2. 韵律提取器: FACodec + Learnable Queries

为提取与其他信息解耦的韵律特征,模型使用预训练的 FACodec (来自 NaturalSpeech 3) 提取帧级韵律特征 {F_1, F_2, ..., F_n} [§II.A]。

[论文原文] 然而,TTS backbone 的计算成本随韵律特征长度线性增长。为减少资源消耗,引入 cross-attention 层,用 m 个可学习 query tokens {Q_1, ..., Q_m} 从变长帧级特征中提取固定长度的韵律嵌入 {P_1, ..., P_m},其中 m << n [§II.A]。

[agent 解读] 这种 learnable query 压缩方式类似于 Perceiver/Q-Former 的设计理念,既保留了韵律信息又控制了计算成本。选择 FACodec 而非其他韵律提取器的原因是 FACodec 通过因子化设计已经将 prosody 与 content/timbre 解耦 [§II.A]。

#### 3. Diffusion 韵律预测器: Transformer Encoder 作为去噪网络

韵律预测器使用一组 Transformer encoder blocks 作为去噪网络 theta [§II.B, Fig 2]。

对话上下文建模方式 [§II.B]:
- 对长度为 N+1 的对话片段,前 N 轮的文本信息 (由预训练 Sentence-T5 提取) 和韵律嵌入交替拼接为上下文向量 c = [s_1, p_1, ..., s_N, p_N]
- 当前轮的文本信息 s_{N+1} 与加噪韵律嵌入 z_t 拼接作为去噪网络输入
- 上下文信息 c 通过 cross-attention 注入

训练目标 [§II.B]:
- 标准 DDPM 噪声预测: L(theta) = E_{t,z_t,epsilon} || epsilon - epsilon_theta(z_t, s_{N+1}, t, c) ||^2 [Eq. 4]

推理流程 [§II.B]:
- 从高斯噪声 z_T ~ N(0, I) 出发,迭代 T 步去噪得到生成的韵律嵌入 z_hat_0 [Eq. 5, 6]

#### 4. 韵律嵌入如何注入 TTS backbone

韵律嵌入与预提取的 speaker embedding 组合后,作为 TTS backbone decoder-only transformer 中 cross-attention 层的 keys 和 values,引导语音合成过程 [§II.A]。

[agent 解读] 这种注入方式使得韵律控制与说话人控制处于同一级别,两者通过 cross-attention 共同影响每一步的自回归解码。

### 训练策略

两阶段训练 [§II.C]:

1. **Stage 1 — TTS backbone 训练**:
   - 先在大规模数据集 (LibriTTS-R, 585h) 上预训练 TTS backbone [§III.A]
   - 再在对话数据集 (DailyTalk, 20h) 上微调 [§III.A]
   - 训练时使用 ground-truth 参考语音提取的韵律嵌入

2. **Stage 2 — 韵律预测器训练**:
   - 冻结 TTS backbone 参数 [§II.C]
   - 用 TTS backbone 从参考语音提取 ground-truth 韵律嵌入
   - 将对话切成等长片段 (每段 5 个 utterance,前 4 个为上下文) [§III.A]
   - 训练 diffusion-based 韵律预测器

[论文原文] 这种两阶段策略的目的是增强 TTS 模块的可控性和合成语音的韵律多样性,韵律作为显式的中间表示被建模 [§II.C]。

## 实验

| 指标 | DiffCSS (Proposed) | GRU-based | DialogueGCN-based | Transformer Encoder-based | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS (Expressiveness) | **3.602 +/- 0.101** | 3.209 +/- 0.108 | 3.347 +/- 0.104 | 3.264 +/- 0.112 | DailyTalk | [Table I] |
| MOS (Coherence) | **3.574 +/- 0.096** | 3.177 +/- 0.087 | 3.362 +/- 0.093 | 3.253 +/- 0.103 | DailyTalk | [Table I] |
| MCD | **7.745** | 8.011 | 7.867 | 7.892 | DailyTalk | [Table I] |
| NDB (20 bins) | **4** | 16 | 13 | 14 | DailyTalk | [Table I] |
| JSD | **0.036** | 0.227 | 0.156 | 0.181 | DailyTalk | [Table I] |

### 关键发现

1. **韵律多样性显著提升**: NDB 从 baseline 最低的 13 降至 4, JSD 从 0.156 降至 0.036,说明 diffusion 生成的韵律分布与 ground truth 分布高度一致 [§III.D]
2. **表现力与上下文连贯性均提升**: E-MOS 和 C-MOS 均优于所有 baseline,表明 diffusion 不仅增加多样性,还改善了韵律的恰当性 [§III.C]
3. **Transformer encoder 作为确定性 baseline 表现一般**: 与 DiffCSS 共享相同结构和参数但无 diffusion 的 Transformer encoder 方法在 MOS 和多样性上均弱于 DialogueGCN [§III.C],说明 diffusion 是关键因素
4. **可视化验证**: Fig 3 显示 Transformer encoder 预测的韵律嵌入集中在少数几个 cluster,而 DiffCSS 的分布更均匀,接近 ground truth [§III.D]

### 消融实验 (多模态上下文) [Table II]

| 设置 | MCD | NDB | JSD | 出处 |
| --- | --- | --- | --- | --- |
| Full (text + acoustic) | 7.745 | 4 | 0.036 | [Table II] |
| w/o textual context | 7.791 | 5 | 0.041 | [Table II] |
| w/o acoustic context | 7.946 | 10 | 0.129 | [Table II] |
| w/o full context | 8.038 | 12 | 0.152 | [Table II] |

[论文原文] 声学上下文的缺失导致性能下降更大 (NDB 4→10, JSD 0.036→0.129),表明声学信息在对话上下文建模中比文本信息更关键 [§III.E]。

## 局限性

1. **数据规模有限**: 仅在 DailyTalk (20h, 2 speakers, 2541 conversations) 上验证,无法确认方法在大规模、多说话人、多语言场景下的泛化能力 [§III.A]
2. **评估场景受限**: 所有评估都在 DailyTalk 的阅读式对话上进行,未涉及真实自发对话场景
3. **TTS backbone 非 SOTA**: ParlerTTS 虽是 LM-based TTS,但并非当前最强 backbone (CosyVoice, Seed-TTS 等表现更好),可能低估了方法在更强 backbone 上的潜力
4. **对话片段长度固定**: 所有对话被切成固定 5 utterance 的 chunk [§III.A],未探索变长上下文或更长上下文窗口的影响
5. **MOS 绝对值偏低**: 最高 E-MOS 仅 3.602 (5 分制),距离自然语音仍有较大差距
6. **未开源**: 论文提供了 demo 页面但未提及代码开源 [§I]
7. **推理效率未报告**: 未给出 diffusion 韵律预测器的推理速度/步数,也未与确定性方法比较延迟

## 点评

DiffCSS 是一项概念清晰、设计合理的工作,其核心洞察 — 用 diffusion model 建模韵律嵌入空间而非声学空间 — 是对现有 Diffusion-based TTS 和 CSS 研究的有意义延伸。将多样性建模和音质保证解耦到两个独立模块是一个优雅的设计选择。

实验结果在多样性指标 (NDB/JSD) 上的提升非常显著,从 baseline 的 13-16 / 0.156-0.227 降至 4 / 0.036,这强有力地验证了 diffusion model 在韵律多样性建模上的优势。消融实验也揭示了声学上下文比文本上下文更重要这一有价值的发现。

然而,工作的验证范围较窄: 仅 2 个说话人、20 小时对话数据、DailyTalk 一个数据集。CSS 的真正挑战在于多说话人、多风格、自发对话场景。此外,MOS 绝对值偏低 (3.6 左右) 可能反映了 ParlerTTS backbone 本身的局限,使用更强的 backbone (如 CosyVoice 或 Seed-TTS) 可能会进一步释放 diffusion 韵律预测器的潜力。

从知识库视角看,DiffCSS 为 [[ProsodyModeling]] 中 "生成模型隐式建模韵律分布" 分支提供了一个新的 CSS 应用实例,也为 [[Diffusion-basedTTS]] 提供了一个 diffusion 用于韵律嵌入而非声学特征的新方向。

## 可复用的 idea

1. **韵律嵌入空间做 diffusion,声学空间用 LM**: 将多样性建模与音质保证解耦的思路可推广到任何需要多样性的 TTS 任务 (如情感 TTS、风格 TTS)
2. **Learnable query 压缩韵律特征**: 用 cross-attention + learnable queries 将变长帧级韵律特征压缩为固定长度嵌入,控制下游计算成本
3. **多模态上下文拼接**: 将文本 (Sentence-T5) 和声学 (prosody embedding) 信息交替拼接作为对话上下文表示,简洁有效
4. **NDB/JSD 评估韵律多样性**: 通过聚类 + 分布比较评估生成韵律的多样性,可作为 CSS/expressive TTS 的标准评估协议

---

> [!review] 审阅 (auto, 2026-06-03)
> **结论**: pass-with-fixes
> **问题**: 2 medium, 2 low
> 详见 `_review/DiffCSS-review.yml`

---

检索命中: [[ProsodyModeling]], [[LLM-basedTTS]], [[SpeakerEmbedding]] | 过滤: [[Diffusion-basedTTS]](pending-review), [[CodecLanguageModel]](pending-review), [[NaturalLanguageDescriptionforTTS]](pending-review) | 未命中但可能相关: 无
