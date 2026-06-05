---
type: paper
tier: deep
title: "CoCoEmo: Composable and Controllable Human-Like Emotional TTS via Activation Steering"
arxiv_id: "2602.03420"
source: "Sources/CoCoEmo.pdf"
authors: [Siyi Wang, Shihong Tan, Siyi Liu, Hong Jia, Gongping Huang, James Bailey, Ting Dang]
year: 2026
venue: "arXiv"
tags: [TTS, emotion-control, activation-steering, mixed-emotion, text-emotion-mismatch, linear-separability, training-free, hybrid-TTS, interpretability]
concepts: ["[[EmotionControlinTTS]]", "[[ConditionalFlowMatching]]", "[[SpeechLanguageModel]]", "[[ProsodyModeling]]", "[[StyleTransferinTTS]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: []
datasets: ["ESD", "RAVDESS", "CREMA-D", "IEMOCAP"]
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 2 个待确认实体页: [[ConditionalFlowMatching]]✓, [[SpeechLanguageModel]]✓, [[模型库/CosyVoice2|CosyVoice 2]]✓, [[EmotionControlinTTS]], [[StyleTransferinTTS]])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: CoCoEmo 属于 [[EmotionControlinTTS]] 中 "推理时激活操控" (activation steering) 路线,是 EmoSteer-TTS (Xie et al., 2025) 之后的第二篇系统性工作。EmoSteer-TTS 首次证明 training-free 激活 steering 可用于 flow-matching TTS 的 DiT 层;CoCoEmo 将 steering 场景从 flow-matching 转移到 SLM (语音语言模型) 阶段,并聚焦于混合情感 (mixed emotion) 和文本-情感错配 (text-emotion mismatch) 两个更复杂的场景。

**已有认知**:
- [[EmotionControlinTTS]] 记录了六条主要技术路线: embedding/label、层级建模、对抗解耦、DPO/RLHF、Emotion-adaptive 表示、Training-free steering。CoCoEmo 扩展了第六条路线的分析深度。
- [[ConditionalFlowMatching]] 是 CosyVoice2 第二阶段的声学渲染器。CoCoEmo 的核心发现之一是情感韵律主要编码在 SLM 而非 flow-matching 模块。
- [[模型库/CosyVoice2|CosyVoice 2]] 是 CoCoEmo 的主要实验 backbone (24 层 Qwen2-based decoder + DiT-based CFM + HiFi-GAN)。
- [[ProsodyModeling]] 中 SSL 超音段分析 (de la Fuente & Jurafsky, 2024) 发现中间层 (8-9) 对韵律表征最强;CoCoEmo 在 TTS SLM 中也发现中后层 (10-17) 情感可分性最高,形成互证。

**创新判断**: 与 EmoSteer-TTS 对比,CoCoEmo 的核心差异在于: (1) steering 目标从 flow-matching DiT 转到 SLM;(2) 引入 cross-conditioning diagnostic 证明情感韵律源于 SLM;(3) 引入 discriminability-driven 层/操作选择替代启发式选择;(4) 提出 multi-rater 混合情感评估协议。

检索命中: [[ConditionalFlowMatching]]✓, [[SpeechLanguageModel]]✓, [[CosyVoice2]]✓ | 过滤: [[EmotionControlinTTS]](pending-review), [[StyleTransferinTTS]](pending-review) | 未命中但可能相关: [[论文笔记/EmoSteer-TTS|EmoSteer-TTS]]

## 速查

> [!summary] 速查
> - **一句话**: 首次系统分析 hybrid TTS (SLM+Flow) 中情感表征的位置和线性可分性,提出 discriminability-driven activation steering 实现可组合混合情感和文本-情感错配合成
> - **路线**: 情感语音数据集 → SLM 各层各操作 linear probe → Top-K 层/操作选出 → mean-difference steering vector 提取 → 加权组合混合情感 → 推理时注入 SLM 中间层 attn_output → 重归一化 → 合成
> - **指标**: Mixed-emotion CREMA-D: E-SIM 0.795, TEP 0.315, H-Rate 0.755 (alpha=5.0); High-mismatch IEMOCAP: E-SIM 0.862, TEP 0.504 (alpha=6.0); 跨 CosyVoice2 和 IndexTTS2 两个 backbone 验证
> - **可借鉴**: (1) cross-conditioning diagnostic 解耦 SLM vs flow 的情感贡献;(2) linear separability 作为 steering site 选择的指导原则;(3) multi-rater 软标签直接作为混合情感 steering 权重
> - **局限**: 未开源代码(声称将公开);仅验证 5 类离散情感(angry/sad/happy/surprise/neutral);steering 仅在 last-token 位置操作;alpha > 4.5 时 WER 有一定退化

## 核心问题

本文回答三个基本问题 [§1]:
1. **Where to steer?** — 在 hybrid TTS (SLM + flow matching) 的哪个模块、哪一层、哪个操作注入 steering vector 最有效?
2. **How to steer?** — 如何构造和组合 steering vector 实现混合情感和文本-情感错配?
3. **How to evaluate?** — 如何评估混合情感合成质量(传统单标签分类无法捕捉)?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CoCoEmo 不修改 TTS 模型参数,仅在推理时向 SLM 的选定中间层注入 steering vector [§3.2]。框架分三步:

1. **分析 (Section 2)**: 通过 cross-conditioning diagnostic 和 layer-wise linear probing 确定最优 steering 位置
2. **构造 (Section 3.1)**: 用 mean-difference 方法从情感/中性语音对提取 steering vector,通过加权组合实现混合情感
3. **注入 (Section 3.2)**: 推理时在选定层和操作处加上 steering vector,强度由 alpha 控制,随后重归一化

### 关键设计选择

**设计选择 1: 为什么在 SLM 而不是 flow-matching 模块 steering?**

论文通过 cross-conditioning diagnostic 实验回答 [§2.1]:
- SLM-driven 条件: 情感参考仅输入 SLM,flow-matching 用中性条件
- Flow-driven 条件: SLM 用中性条件,情感参考仅输入 flow-matching

结果 [Table 1]: SLM-driven 在 F0 CCC (0.109 vs 0.305) 和 energy CCC (0.308 vs 0.737) 上均更低(即跨情感差异更大),SR STD 更高 (0.691 vs 0.518),说明 SLM 产生了更强的情感韵律差异。Flow-matching 主要做声学渲染,不额外引入情感韵律变化。[论文原文]

[agent 解读] 这与 CosyVoice2 的架构设计一致:SLM (Qwen2-based) 将文本+参考语音 token 映射为 speech tokens,情感信息通过参考语音 token 进入 SLM 并影响生成的 speech token 序列;flow-matching 接收这些 token 后只负责"渲染"成 mel spectrogram,本身不额外编码情感。

**设计选择 2: 为什么选 mid-to-late layers 的 attn_output?**

论文采用 discriminability-driven 方法 [§2.2]: 对每一层每个操作的 last-token 激活值训练 linear probe 预测情感标签,用分类准确率衡量线性可分性。

结果 [Fig 3]: CosyVoice2 中 layers 10-17 的 attn_output 一致达到最高判别准确率 (~0.8)。IndexTTS2 中 peak 在 layers 5-10 [Appendix B, Fig 8]。[论文原文]

为什么高可分性 = 好的 steering site? 因为 steering vector 要求激活空间中情感类别呈几何分离,单一方向的平移能可靠地改变情感而不影响其他属性 [§2.2]。[论文原文]

最终选择: CosyVoice2 用 top-2 层 (layers 17, 14);IndexTTS2 用 top-3 层 (layers 6, 8, 1) [Appendix I, Table 9]。选择依据是 WER 约束下 (≤ no-steer + 0.5) 的最大可用 alpha 范围 [§4.5]。[论文原文]

**设计选择 3: 为什么 steering vector 在 last-token 而非所有 token 位置操作?**

论文使用 last-token representation [§2.2, §3.1],因为它 "summarizes cumulative emotional encoding" [§2.2]。

[agent 解读] 在 causal AR decoder 中,last-token 的隐藏状态包含了所有前文的注意力聚合信息。对 last-token 操作等于在整句层面施加情感偏移。这也解释了为什么该方法是 utterance-level 控制而非 word-level。

**设计选择 4: 为什么用 mean-difference 而非其他方法构造 steering vector?**

Steering vector 的构造 [§3.1, Eq. 6]:
$$\mathbf{v}_e^{(l,o)} = \frac{1}{|D^{(e)}|} \sum_{i \in D^{(e)}} \mathbf{h}_i^{(l,o)} - \frac{1}{|D_0^{(e)}|} \sum_{j \in D_0^{(e)}} \mathbf{h}_j^{(l,o)}$$

即: 同一 speaker+transcript 下的情感语音激活均值 - 中性语音激活均值。通过 speaker-matched neutral-emotion pairing 控制 speaker 和 content 变量,确保差值主要反映情感维度 [§3.1]。[论文原文]

[agent 解读] 这与 EmoSteer-TTS 使用的 difference-in-means 方法一致,但 EmoSteer-TTS 额外引入了 top-k token 稀疏选择,而 CoCoEmo 不做 token 选择(只操作 last-token)。

### 训练策略

**CoCoEmo 不需要任何训练/微调。** 整个框架是 training-free 的 [§1, §3]:
- Steering vector 提取: 仅需前向传播提取激活值 + 均值计算
- 推理时注入: 在选定层做加法 + 重归一化

数据需求: ESD + RAVDESS + CREMA-D 共 ~20K 条情感语音,speaker-independent split 50/20/30 [§4.1]。Steering vector 从 training split 提取。

混合情感 steering [Eq. 7]:
$$\mathbf{v}_{mix}^{(l,o)} = \sum_{e=1}^{E} p_e \mathbf{v}_e^{(l,o)}, \quad \sum p_e = 1$$

其中权重 $p_e$ 可以来自: (1) 用户指定比例;(2) multi-rater 标注共识分布 [§3.3, Eq. 9]。[论文原文]

推理时注入 [Eq. 8]:
$$\tilde{\mathbf{h}}_i^{(l,o)} = \mathbf{h}_i^{(l,o)} + \alpha \cdot \mathbf{v}^{(l,o)}$$

随后重归一化: $\tilde{\mathbf{h}} \leftarrow \frac{\|\mathbf{h}\|}{\|\tilde{\mathbf{h}}\|} \cdot \tilde{\mathbf{h}}$ [§3.2],以保持激活尺度和语义连贯。[论文原文]

## 实验

| 指标 | 本文 (CoCoEmo) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| E-SIM (mixed, alpha=5.0) | 0.795 | No-steer 0.743 / Ins2 0.762 | CREMA-D ID | [Table 2] |
| TEP (mixed, alpha=5.0) | 0.315 | No-steer 0.065 / Ins2 0.169 | CREMA-D ID | [Table 2] |
| H-Rate (mixed, alpha=5.0) | 0.755 | Ins1 0.694 / Ins2 0.688 | CREMA-D ID | [Table 2] |
| Spearman rho (mixed, alpha=5.0) | 0.297 | Ins1 0.111 / Ins2 0.104 | CREMA-D ID | [Table 2] |
| S-SIM (mixed, alpha=5.0) | 0.870 | No-steer 0.871 / Ins2 0.851 | CREMA-D ID | [Table 2] |
| WER (mixed, alpha=5.0) | 0.78 | No-steer 1.07 / Ins2 0.06 | CREMA-D ID | [Table 2] |
| N-MOS (mixed, alpha=5.0) | 3.96 | No-steer 4.11 / Ins2 3.36 | CREMA-D ID | [Table 2] |
| E-SIM (high mismatch) | 0.862 | No-steer 0.802 / Ins 0.843 | IEMOCAP OOD | [Table 3] |
| TEP (high mismatch) | 0.504 | No-steer 0.197 / Ins 0.436 | IEMOCAP OOD | [Table 3] |
| N-MOS (high mismatch) | 4.40 | No-steer 4.22 / Ins 4.23 | IEMOCAP OOD | [Table 3] |
| E-SIM (single, alpha=6.0) | 0.589 | No-steer 0.444 / Ins 0.600 | ESD+RAVDESS+CREMA-D | [Table 8] |
| TEP (single, alpha=6.0) | 0.366 | No-steer 0.023 / Ins 0.278 | ESD+RAVDESS+CREMA-D | [Table 8] |

**关键实验发现**:

1. **混合情感**: CoCoEmo 在 E-SIM、TEP、H-Rate、Spearman rho 上全面优于 instruction-based 和 no-steering baseline [Table 2, Fig 4]。随 alpha 增大,steering 提供 proportional 定量控制,而 instruction-based 方法在混合比例控制上受限 [§4.2]。

2. **文本-情感错配**: 在 high-mismatch 子集上,steering 的增益最大 (E-SIM: 0.862 vs no-steer 0.802),说明 steering 能有效克服文本隐含情感偏差 [§4.3, Fig 5, Table 3]。

3. **Cross-backbone 泛化**: 在 IndexTTS2 上也观察到一致改进趋势,说明方法不依赖特定架构 [Table 2, Table 3]。

4. **插件式增强**: CoCoEmo 可叠加在已有情感控制方法之上 (Ins1/Ins2 + steering, Emo_V + steering),进一步提升 TEP 和 H-Rate [Table 2, §4.2]。

5. **Layer-wise steering 验证**: TEP 在 layers 17 和 14 达到峰值,与 linear separability 分析 [Fig 3] 一致,correlation rho=0.5078 [§4.5, Fig 7]。

6. **超参数**: alpha 在 [0, 4.5] 内稳定;alpha=6.0 时 WER 有一定上升 [§4.5]。Top-2 层 (CosyVoice2) / Top-3 层 (IndexTTS2) 是最优配置 [Table 9]。

## 局限性

1. **仅 utterance-level 控制**: steering 在 last-token 位置操作,无法实现 word-level 或 segment-level 情感变化 (与 TTS-CtrlNet、WeSCon、TED-TTS 形成对比)
2. **离散情感类别限制**: 仅验证 5 类基本情感 (angry/sad/happy/surprise/neutral),未覆盖连续 valence-arousal 空间或细粒度情感 (讽刺、紧张等)
3. **重归一化的潜在信息损失**: 虽然保持激活尺度,但可能导致语义方向的微妙偏移 [agent 解读]
4. **WER 退化**: alpha > 4.5 时 WER 有上升趋势,说明强 steering 可能干扰语义内容 [§4.5]
5. **IndexTTS2 上的 OOD 异常**: 在 IEMOCAP OOD 上 dominant-emotion steering 的 E-SIM 略超 mixed steering [Fig 12, §4.2],可能与 IndexTTS2 显式 emotion conditioning 的交互有关
6. **代码未开源**: 声称将公开但截至提交时未开源 [§4.1 footnote]

## 点评

**优点**:
- **分析深度出色**: cross-conditioning diagnostic (§2.1) 和 discriminability-driven site selection (§2.2) 为 activation steering 在 TTS 中的应用提供了系统性的理论框架,而非 EmoSteer-TTS 的启发式选择
- **评估框架创新**: multi-rater 软标签直接作为 steering 权重的设计 [§3.3] 优雅地解决了混合情感的 ground-truth 定义问题
- **跨 backbone 验证**: 同时在 CosyVoice2 (implicit conditioning) 和 IndexTTS2 (explicit conditioning) 上验证,增强了结论的通用性
- **插件式设计**: 可叠加在已有情感控制方法之上,不冲突

**不足**:
- 与 EmoSteer-TTS 的对比不够直接 — 两者 steering 目标不同 (SLM vs flow-matching DiT),但在 CosyVoice2 上的相同评估指标缺少 head-to-head 对比
- 混合情感的主观评估偏弱 — 仅有 N-MOS (自然度),缺少专门的混合情感感知 MOS (如"你是否感知到多种情感共存?")
- IndexTTS2 上的 steering 层选择 (layers 6, 8, 1) 中 layer 1 的选入需要更多解释 — 浅层通常被 input conditioning 主导 [Appendix B]

## 可复用的 idea

1. **Cross-conditioning diagnostic**: 适用于任何两阶段 TTS 系统 (LM + acoustic model),可快速判断哪个模块编码了目标属性 (情感/风格/口音等)
2. **Discriminability-driven site selection**: 用 linear probe 的分类准确率指导 steering/probing/intervention 的位置选择,比启发式搜索更 principled
3. **Multi-rater consensus as steering weights**: 将标注者分歧直接建模为混合比例,避免了强制单标签的信息损失
4. **重归一化 trick**: $\tilde{h} \leftarrow \frac{\|h\|}{\|\tilde{h}\|} \cdot \tilde{h}$ 简单但有效,保持激活尺度一致性,可迁移到任何 activation steering 场景
5. **WER-constrained layer selection**: 在可控性和语义完整性之间做 Pareto 权衡的实用方法 [Table 9, Appendix I]

## 审阅

> [!review] 审阅 (2026-06-05, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 四个设计选择的 WHY/HOW 清晰,3 个月后可复述 |
> | 可信赖 | pass-with-fixes | 核心数字均有出处;datasets frontmatter 缺失已修正 |
> | 可区分 | pass | 来源标注覆盖率高,agent 解读明确标识 |
> | 可定位 | pass | KB 背景谱系定位详细,与 EmoSteer-TTS 四点差异清晰 |
> | 不污染 | pass-with-fixes | EmotionControlinTTS key_papers 已超限,反向更新需改为正文追加 |
> 
> Issues: 4 (high: 0, medium: 1, low: 3)
> 详见 `_review/CoCoEmo-review.yml`
