---
type: paper
tier: deep
title: "ZET-Speech: Zero-shot adaptive Emotion-controllable Text-to-Speech Synthesis with Diffusion and Style-based Models"
arxiv_id: "2305.13831"
source: "Sources/ZET-Speech.pdf"
authors: [Minki Kang, Wooseok Han, Sung Ju Hwang, Eunho Yang]
year: 2023
venue: "Interspeech 2023"
tags: [TTS, emotion-control, zero-shot, diffusion, domain-adversarial-training, classifier-free-guidance, style-transfer]
concepts: ["[[EmotionControlinTTS]]", "[[Classifier-FreeGuidance]]", "[[GradientReversalLayer]]", "[[Diffusion-basedTTS]]", "[[GlobalStyleTokens]]"]
models: ["[[论文笔记/ZET-Speech|ZET-Speech]]", "Grad-StyleSpeech"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[Zero-shotSpeechSynthesis]]✓, [[EmotionControlinTTS]], [[Classifier-FreeGuidance]], [[GradientReversalLayer]], [[Diffusion-basedTTS]], [[GlobalStyleTokens]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Zero-shotSpeechSynthesis]], [[EmotionControlinTTS]], [[Classifier-FreeGuidance]], [[GradientReversalLayer]], [[Diffusion-basedTTS]], [[GlobalStyleTokens]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: ZET-Speech (2023) 处于 emotion-controllable TTS 演进的较早阶段,位于 "跨说话人情感迁移 (2022)" 和 "DPO/RLHF 对齐 (Emo-DPO, 2024)" 之间。它的核心贡献在于将零样本自适应 TTS (zero-shot adaptive TTS) 与情感可控性 (emotion controllability) 结合 — 在此之前,两个能力分别独立发展:零样本 TTS (Meta-StyleSpeech, Grad-StyleSpeech) 不考虑情感控制,情感 TTS (GST, MsEmoTTS) 不考虑零样本泛化。

**已有认知**:
- [[GradientReversalLayer]] 在 TTS 中已被广泛用于 speaker-emotion disentanglement (IndexTTS2, DisCo-Speech, SelfTTS, AgentSteerTTS)。ZET-Speech 是该技术在情感 TTS 中的早期应用之一,使用标准 GRL+CE 方案解耦 style vector 中的情感信息。
- [[Classifier-FreeGuidance]] 在 diffusion TTS 中已是标准技术 (Guided-TTS 2, 各种 flow-matching TTS)。ZET-Speech 同时探索了 classifier guidance 和 classifier-free guidance 两条路线。
- [[GlobalStyleTokens]] (GST) 在论文中作为 baseline 条件输入方式与 hard emotion label 对比。GST 的无监督性质意味着情感维度不一定被干净分离 — 这正是 ZET-Speech 通过 DAT 解决的问题。
- [[Diffusion-basedTTS]] 方面,ZET-Speech 基于 Grad-StyleSpeech,属于 SDE-based diffusion TTS 路线 (Grad-TTS → Grad-StyleSpeech → ZET-Speech)。

**创新判断**: ZET-Speech 的核心创新在于将 domain adversarial training 和 diffusion guidance 两种技术组合,解决了零样本场景下的情感控制问题。对比后续工作 (EmoSphere-TTS 使用球面向量, DiEmo-TTS 使用 DINO 自监督蒸馏, EmoSteer-TTS 使用 activation steering),ZET-Speech 的方法相对简单直接,但在 2023 年时段属于合理且有效的方案。

## 速查

> [!summary] 速查
> - **一句话**: 通过 domain adversarial training 解耦 style vector 中的情感信息 + diffusion guidance 增强情感表达,实现仅凭中性参考语音和情感标签即可合成任意说话人情感语音
> - **路线**: 中性参考语音 → Mel-Style Encoder (DAT 解耦) → emotion-free style vector + emotion label → Transformer Encoder → mu → Diffusion (+ CG/CFG guidance) → 情感 mel → HiFi-GAN → 语音
> - **指标**: ECA 51.59% (seen, CG) vs baseline 16.35% [Table 1]; ECA 39.86% (unseen, CG) vs baseline 18.29% [Table 2]; MOS 3.44 (seen, CFG) vs 2.60 (baseline) [Table 4]
> - **可借鉴**: (1) DAT 解耦 style vector 中的特定属性是一个简洁有效的 trick,可推广到其他需要解耦的场景; (2) classifier-free guidance 在 TTS diffusion 中同时增强条件控制的方法; (3) 将 mu (noise prior) 也用 null embedding 生成 mu_null 的 CFG 设计细节
> - **局限**: (1) 基于 mel spectrogram + HiFi-GAN 的两阶段架构; (2) guidance 增强情感表达以牺牲语音质量为代价 (CER/SECS 下降) [§3.3]; (3) 仅支持离散情感标签,不支持连续情感控制; (4) 未开源; (5) 实验规模较小 (9 seen + 10 unseen speakers)

## 核心问题

ZET-Speech 要解决的核心问题是: **如何在零样本场景下实现情感可控的语音合成** — 即仅凭目标说话人的一段中性语音和一个情感标签,就能合成该说话人带有指定情感的语音。这需要同时解决两个子问题: (1) 如何从 style vector 中去除情感信息,使情感控制仅由外部 emotion label 决定; (2) 如何增强 diffusion 模型对情感条件的响应程度。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ZET-Speech 基于 Grad-StyleSpeech [7] 构建,由两个核心组件组成 [§2, Fig 2]:

1. **Style-based Generator**: 包括 mel-style encoder h_psi (将参考语音编码为 style vector s) 和 transformer encoder f_theta (将 phoneme + style vector + emotion 转换为 mu)
2. **Diffusion Model**: 基于 SDE 的 score-based 生成模型,以 mu 为 noise prior,逐步去噪生成 mel spectrogram

基础工作流: 参考语音 Y → mel-style encoder → style vector s → transformer encoder f_theta(x, s, e) → mu → diffusion reverse process → mel spectrogram → HiFi-GAN → 语音

### 关键设计选择

**设计选择 1: Domain Adversarial Training (DAT) 解耦 style vector 中的情感**

WHY: [论文原文] 在 style-based generator 中,style vector s 同时编码了说话人身份和情感特征。如果情感信息纠缠在 s 中,即使提供了 "happy" 的 emotion label,模型仍会被 s 中编码的中性情感主导,生成中性语音 [§2.2]。

HOW: 在 mel-style encoder h_psi 和 emotion classifier g_lambda 之间插入 gradient reversal layer (GRL) [§2.2]:
- **前向**: emotion classifier 尝试从 style vector s 预测情感标签 p(e|s; lambda)
- **反向**: GRL 将梯度乘以 -alpha,使 mel-style encoder 被训练为输出不包含情感信息的 style vector
- 损失函数: L_e = -log p(e_hat|s; lambda),mel-style encoder 的梯度为 -alpha * dL_e/d_psi

为什么不用其他解耦方法: [agent 解读] DAT/GRL 是一种轻量且成熟的方案 (Ganin et al., 2016),不需要修改模型架构,只需添加一个小分类器和 GRL,对基线模型的侵入性最小。这与后续更复杂的方案 (DiEmo-TTS 的 DINO 蒸馏, SelfTTS 的 cosine-based GRL) 形成对比。

**设计选择 2: Diffusion Guidance 增强情感表达**

WHY: [论文原文] DAT 虽然使模型能响应情感条件,但生成的情感表达力仍有提升空间。Guidance methods 可以在推理时进一步增强条件生成的效果 [§2.3]。

HOW — Classifier Guidance (CG) [§2.3, Eq. 2]:
- 训练一个在带噪 mel spectrogram 上的情感分类器 p(e|Y_t)
- 推理时将分类器梯度加入 reverse diffusion:
  nabla_Yt log p(Yt|e) = gamma * nabla_Yt log p(e|Yt) + nabla_Yt log p(Yt)
- gamma 控制情感强度

HOW — Classifier-Free Guidance (CFG) [§2.3, Eq. 3-4]:
- 训练时随机将 emotion embedding 替换为 null embedding
- 关键细节: 因为 mu = f_theta(x, s, e) 也包含情感信息,所以无条件时也需要用 mu_null = f_theta(x, s, null) [论文原文, §2.3]
- 推理时: epsilon_hat = epsilon(Yt, t, mu, s, e) + gamma * (epsilon(Yt, t, mu, s, e) - epsilon(Yt, t, mu_null, s, null))

为什么同时探索两种 guidance: [论文原文] Classifier guidance 可能获得更高 ECA (情感准确率),但 classifier-free guidance 不需要额外分类器,且可与 GST 等其他方法兼容 [§3.2]。[agent 解读] 这也反映了 2023 年时 CFG 尚未完全取代 CG 的过渡时期。

### 训练策略

1. **预训练**: 在大规模中性多说话人数据集上预训练 Grad-StyleSpeech 模型 (Korean AIHub multi-speaker / English LibriTTS) [§3.1.1]
2. **情感微调**: 在情感语音数据集上微调 200k steps,batch size 8,单 TITAN RTX GPU,Adam lr=1e-3 + 线性衰减 [§3.1.1]
3. **DAT**: 在微调阶段同时训练 GRL + emotion classifier [§2.2]
4. **CG 需额外训练**: 训练无条件 score estimator + 噪声 mel 上的 emotion classifier [§2.3]
5. **CFG 在同一模型**: 训练时随机 dropout emotion embedding [§2.3]

## 实验

| 指标 | ZET-Speech (CG) | ZET-Speech (CFG, gamma=1.25) | Grad-StyleSpeech baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| ECA (Seen, ↑) | 51.59 | 41.75 | 16.35 | Korean Emotional | [Table 1] |
| CER (Seen, ↓) | 16.22 | 13.26 | 16.84 | Korean Emotional | [Table 1] |
| SECS (Seen, ↑) | 0.759 | 0.750 | 0.778 | Korean Emotional | [Table 1] |
| ECA (Unseen, ↑) | 39.86 | 31.57 | 18.29 | Korean Emotional | [Table 2] |
| CER (Unseen, ↓) | 12.15 | 11.26 | 10.20 | Korean Emotional | [Table 2] |
| ECA (English Unseen, ↑) | - | 59.75 (gamma=1.5) | 40.50 | ESD+LibriTTS | [Table 3] |
| MOS (Seen) | 3.02 | 3.44 | 2.60 | Korean Emotional | [Table 4] |
| MOS (Unseen) | 3.06 | 3.31 | 2.63 | Korean Emotional | [Table 4] |
| SMOS (Seen) | 3.12 | 3.22 | 3.11 | Korean Emotional | [Table 4] |

关键发现:
1. **DAT 是核心贡献**: GSS+DAT 将 ECA 从 16.35 提升到 31.27 (seen),说明解耦 style vector 中的情感信息对情感控制至关重要 [Table 1]
2. **组合效果最强**: ZET-Speech (DAT + CG) ECA 达 51.59,证明 DAT 和 guidance 互补 [Table 1]
3. **CG vs CFG trade-off**: CG 在客观 ECA 上更高 (51.59 vs 41.75),但 CFG 在主观 MOS 上更好 (3.44 vs 3.02),主观评价者认为 CFG 生成的情感语音更自然 [Table 1, 4]
4. **质量-情感 trade-off**: 增大 guidance scale gamma 提升 ECA 但降低 CER,存在明确的 trade-off [Fig 5]
5. **t-SNE 验证**: 无 DAT 时 style vector 按情感聚类,有 DAT 后 style vector 更均匀分布,证明情感信息被成功解耦 [Fig 3]

## 局限性

1. **情感表达与语音质量的 trade-off**: guidance 增强情感的同时降低了 CER 和 SECS,说明模型在增强情感时可能扭曲了语音的内容准确性和说话人相似度 [§3.2, §3.3]
2. **仅支持离散情感标签**: 不支持连续情感维度 (arousal-valence) 或混合情感,控制粒度受限 [agent 解读]
3. **架构过时**: 基于 mel spectrogram + HiFi-GAN 的两阶段管线,后续已被端到端 LLM-TTS 和 flow matching 方法取代 [agent 解读]
4. **实验规模有限**: 仅 9 seen + 10 unseen speakers,7 种情感,20 名评估者,统计可靠性受限 [§3.1.3]
5. **ECA 绝对值不高**: 最高 ECA 约 51.59% (seen, CG),意味着仍有近一半的生成语音未被分类器识别为目标情感 [Table 1]
6. **未开源**: 无公开代码和预训练模型

## 点评

ZET-Speech 提出了一个清晰且实际的问题: 如何在零样本 TTS 中实现情感控制。其解决方案 — DAT 解耦 + diffusion guidance — 虽然在今天看来较为基础,但在 2023 年属于合理且有效的组合。论文的贡献不在于提出全新技术 (DAT 和 CFG 都是已有技术),而在于将它们正确地组合并验证了其在零样本情感 TTS 中的有效性。

从演进角度看,ZET-Speech 的 DAT 解耦方案与后续的 IndexTTS2 (单向 GRL)、AgentSteerTTS (双向 GRL + 正交约束)、DiEmo-TTS (DINO 蒸馏)、SelfTTS (cosine-based GRL) 形成了一条清晰的方法演进线。其 guidance 方案也与后续 EmoSteer-TTS (activation steering)、TTS-CtrlNet (ControlNet 旁挂)、DiffRO (reward guidance) 等更先进的控制方法形成对比。

一个值得注意的实验发现是 CG 和 CFG 在客观/主观评价上的分歧 (CG 客观更高但 CFG 主观更自然),这暗示了情感表达的"准确性"和"自然度"可能是不同维度。

## 可复用的 idea

1. **null prior trick for CFG**: 当 noise prior mu 也包含条件信息时,CFG 的无条件分支需要同步使用 null-conditioned prior (mu_null = f(x, s, null)),不能只在 score estimator 层面 dropout 条件 [§2.3]。这个细节在后续使用 CFG 的 TTS 系统设计中值得注意。
2. **DAT 作为通用属性解耦工具**: 在 style-based generator 中,DAT + GRL 是一种侵入性最小的属性解耦方案,不需要修改编码器架构,仅需添加一个小分类器。可推广到解耦 style vector 中的任意目标属性 (语言/口音/语速等)。
3. **CG vs CFG 的使用场景选择**: CG 在客观指标上更强但需要额外分类器,CFG 主观感知更自然且更灵活。在实际系统中可能优先选择 CFG [§3.2]。

---

检索命中: [[Zero-shotSpeechSynthesis]]✓, [[EmotionControlinTTS]][待确认], [[Classifier-FreeGuidance]][待确认], [[GradientReversalLayer]][待确认], [[Diffusion-basedTTS]][待确认], [[GlobalStyleTokens]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | WHY+HOW 因果解释完整,速查可借鉴有具体 trick |
> | 可信赖 | pass | 所有数字经 PDF 交叉验证无误,出处覆盖率 >90% |
> | 可区分 | pass | 来源标注覆盖率约 85%,论文原文/agent 解读边界清晰 |
> | 可定位 | pass | 谱系定位精确,与 6 个 KB 页面关系清晰 |
> | 不污染 | pass | 概念引用合理,models 字段已补充 baseline |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/ZET-Speech-review.yml`
