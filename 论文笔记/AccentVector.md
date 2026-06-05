---
type: paper
tier: deep
title: "Accent Vector: Controllable Accent Manipulation for Multilingual TTS Without Accented Data"
arxiv_id: "2603.07534"
source: "Sources/AccentVector.pdf"
authors: [Thanathai Lertpetchpun, Thanapat Trachu, Jihwan Lee, Tiantian Feng, Dani Byrd, Shrikanth Narayanan]
year: 2026
venue: "arXiv"
tags: [TTS, accent, task-vector, LoRA, multilingual, controllable-TTS, XTTS, zero-shot]
concepts: ["[[SpeakerAdaptation]]", "[[ProsodyModeling]]", "[[StyleTransferinTTS]]"]
models: ["[[论文笔记/XTTS|XTTS]]"]
tasks: ["[[Cross-lingualVoiceCloning]]", "[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[SpeakerAdaptation]], [[Cross-lingualVoiceCloning]], [[SpeakerEmbedding]], [[ProsodyModeling]], [[StyleTransferinTTS]], [[Zero-shotSpeechSynthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeakerAdaptation]][待确认], [[Cross-lingualVoiceCloning]]✓, [[SpeakerEmbedding]]✓, [[ProsodyModeling]]✓, [[StyleTransferinTTS]][待确认], [[Zero-shotSpeechSynthesis]]✓ | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文位于 Speaker Adaptation 和 Style Transfer 的交叉地带。传统 accent TTS 要么需要大量口音数据训练 (Xinyuan et al.)、要么通过文本音译间接引入口音 (MacST)、要么仅建模时长维度 (Onda et al.)。本文提出的 Accent Vector 核心创新在于将 task vector (Ilharco et al., 2023) 从 NLP/CV 领域迁移到语音合成的口音控制场景,通过参数空间算术实现无口音数据的细粒度口音操控。

**已有认知**:
- **Speaker Adaptation** 的参数效率方向已从 CLN (AdaSpeech) 演进到 LoRA,但现有工作主要聚焦说话人音色复制,未将 LoRA 权重差视为可操控的"风格向量"
- **Cross-lingual Voice Cloning** 当前 SOTA (CosyVoice 3, Qwen3-TTS) 通过大规模多语言训练实现跨语言克隆,但口音控制(如生成 Spanish-accented English)仍是开放问题
- **Prosody Modeling** 覆盖了口音涉及的超音段特征(时长、节奏、语调),但将口音建模为参数空间位移而非隐式特征是新视角
- **Style Transfer** 的演进从 GST → VAE → in-context learning,本文则提出参数空间的线性风格控制这一新路线

**创新判断**: 相比 KB 中已有的 Speaker Adaptation 方法,本文的独特贡献是 (1) 将 task vector 概念应用于口音控制,将 LoRA 权重差解释为口音方向向量; (2) 不需要任何口音数据,仅用目标语言的母语数据微调; (3) 支持多口音线性组合,这是此前口音 TTS 工作未实现的。

## 速查

> [!summary] 速查
> - **一句话**: 将 LoRA 微调后的参数差作为"Accent Vector",通过参数空间的线性缩放和插值实现无口音数据的细粒度口音控制和混合口音合成
> - **路线**: 目标语言母语数据 → LoRA 微调 XTTS-v2 → 提取参数差 τ = θ_ft - θ_pre = θ_LoRA → 推理时 θ_accent = θ_pre + α·τ → 口音语音
> - **指标**: British accent prob 23.3→56.7% (+143%), Hindi 2.2→24.2% (+1021%), 人类口音识别准确率 53-80%, SSIM 维持 ~0.86-0.90, UTMOS 2.59-3.61 [Table 3][Table 6]
> - **可借鉴**: 将 LoRA 权重差视为可操控的属性向量,通过线性缩放控制属性强度 — 可迁移到情感、语速等其他可控 TTS 场景; 多向量线性组合实现多属性混合
> - **局限**: 对声调语言(普通话)效果最弱(+23.6% vs 其他语言 +90-1021%); 评估依赖有偏代理模型(VoxProfile/Whisper/UTMOS); 仅在 XTTS-v2 上验证; 训练数据 UTMOS 中等偏低

## 核心问题

1. **如何在不使用口音数据的前提下生成带口音的语音?** 传统口音 TTS 依赖大规模口音数据,但非母语英语口音数据极度稀缺
2. **如何实现口音强度的细粒度连续控制?** 现有方法(音译/时长建模)产生固定口音,无法调节强度
3. **如何生成混合口音语音?** 现实中许多说话人展现多种口音影响(如 Spanish+British),这在已有系统中无法建模

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

以 XTTS-v2 为 backbone,一个支持 17 种语言的多语言零样本 TTS 模型。XTTS-v2 由三部分组成: VQ-VAE (将 mel 谱转为离散 acoustic code)、encoder (接收文本+参考语音+语言 ID,预测 acoustic code)、decoder (从 encoder 输出合成语音) [§2.2]。

核心流程:
1. **微调**: 对 XTTS-v2 进行 LoRA 微调,使用目标口音对应的母语数据(如用西班牙语数据生成 Spanish-accented English) [§3.1]
2. **提取向量**: Accent Vector τ = θ_ft - θ_pre = θ_LoRA [Eq. 1-3] [§3.2]
3. **推理控制**: θ_accent = θ_pre + α·τ,通过 α 控制口音强度 [Eq. 4] [§3.3]
4. **混合口音**: 多个 Accent Vector 线性组合 τ_interpolated = Σ αᵢ·τᵢ [Eq. 5-6] [§3.3]

### 关键设计选择

**为什么用 LoRA 微调而非全参数微调?** [论文原文] LoRA 将可训练参数从 378M 大幅减少到约 8M,同时缓解过拟合和灾难性遗忘 [§4.1]。[agent 解读] 更重要的是,LoRA 的低秩结构天然地将 Accent Vector 限制在参数空间的一个低维子空间中,这可能有助于 task vector 算术的线性假设成立。

**为什么用母语数据微调而不是用口音数据?** [论文原文] 关键洞察是:用西班牙语数据微调时设置 language ID 为英语,模型学会将英语的语言内容与西班牙语的声学特征对齐。这迫使模型在英语文本空间中编码目标口音的音段和超音段特征 [§3.4]。[agent 解读] 这本质上利用了多语言 TTS 模型中语言 ID 与声学特征之间的可分离性 — language ID 控制"说什么语言",而模型参数编码"怎么发音",微调改变后者使其偏向目标语言的发音模式。

**为什么 task vector 算术在口音控制中可行?** [论文原文] 基于 Ilharco et al. (2023) 的发现:大型预训练模型的参数空间具有近似线性结构,不同任务的参数偏移方向近似正交 [§2.1]。[agent 解读] 口音作为一种相对独立的语音属性(不同于内容或说话人身份),其参数偏移方向可能与其他属性的偏移方向有足够的正交性,从而支持线性操作。但这一线性假设对于声调语言(如普通话)可能不成立,因为声调与语调的交互更复杂。

**微调时 language ID 与数据的"错配"设计**: [论文原文] 训练时使用目标语言的文本和参考语音(如西班牙语),但将 language ID 设为基础语言(如英语) [§3.1][Fig 1]。推理时使用基础语言的文本和 language ID,口音通过修改后的模型参数引入 [§3.4]。[agent 解读] 这种 language ID-数据错配是整个方法的核心 trick。它创造了一个"语言冲突"场景:模型被告知要说英语,但参考信号和训练目标是西班牙语 — 参数空间中的这个矛盾被编码为口音信息。

### 训练策略

- 使用 LoRA rank=16,应用于 encoder 所有线性层 [§4.1]
- 训练 60,000 步,Adam optimizer,lr=3e-5 [§4.1]
- 单张 A40 GPU,约 8 小时 [§4.1]
- 数据清洗: 仅保留 DNSMOS > 3.4 的语句,丢弃 < 3s 的语句 [§4.2]
- 每种语言限制为单一代表方言(如 Peninsular Spanish,不混入拉美西班牙语) [§4.2]

## 实验

| 指标 | 本文 | Baseline (Pretrained) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| British Accent Prob (%) | 56.7 | 23.3 | VCTK (test) | [Table 3] |
| Spanish Accent Prob (%) | 39.7 | 15.5 | CommonVoice-sp (test) | [Table 3] |
| Hindi Accent Prob (%) | 24.2 | 2.2 | IndicVoices-R (test) | [Table 3] |
| French Accent Prob (%) | 23.2 | 12.2 | CommonVoice-fr (test) | [Table 3] |
| German Accent Prob (%) | 27.4 | 14.3 | CommonVoice-de (test) | [Table 3] |
| Mandarin Accent Prob (%) | 33.8 | 27.4 | KeSpeech (test) | [Table 3] |
| Speaker SIM (British) | 0.90 | — | — | [Table 3] |
| UTMOS (British) | 3.61 | — | — | [Table 3] |
| WER (British) | 5.46 | — | LibriTTS-R test-clean | [Table 3] |
| Human Accent ID Accuracy | 53-80% | random 14% | 70 samples, 16 listeners | [Table 6] |
| Human Naturalness (British) | 3.91 | 3.06 (US) | — | [Table 6] |

**口音强度可控性 [§6.3]**: α 从 0 到 1 递增时,口音概率单调递增(Hindi: ~2% → ~50%; British: ~20% → ~55%),WER 也随之上升,表现出口音强度与可懂度的 trade-off [Fig 3]。

**混合口音 [§6.4]**: 两个 Accent Vector 以 α=0.5 合并后,两个目标口音的概率通常都有提升(如 Hindi+British: Hindi 2.16→7.11%, British 3.39→9.64%),但 Spanish+British 和 Mandarin+British 组合中 British 口音主导,可能因 VoxProfile 对英语口音有偏 [Table 5]。

**跨语言口音迁移 [§6.2]**: 可将 English(England)口音迁移到西班牙语、德语、普通话等非英语语言。English accent prob 在西班牙语上从 1.20% 提升到 44.69%,德语 8.57→41.57%,普通话 0.00→3.03% [Table 4]。

## 局限性

1. **声调语言效果弱**: 普通话口音迁移提升最小(+23.6%),论文将其归因于普通话作为声调语言,F0 编码词汇意义,与英语的重音-时序系统差异过大 [§6.1.3]。[agent 解读] 这暗示 task vector 的线性假设在跨语系距离较大时可能不成立,尤其是超音段特征(声调 vs 语调)的参数空间编码可能不是线性可分的
2. **评估代理偏差**: VoxProfile 偏向英语口音,LID 模型作为口音代理本身不精确,Whisper 对非母语英语识别率低,UTMOS 主要在英语数据上训练 [§6.6]。这些偏差使得绝对指标的可信度有限
3. **训练数据质量中等**: 非英语训练数据的 UTMOS 仅 2.57-2.93 [Table 2],低于英语的 3.92,可能限制合成质量上限
4. **仅在 XTTS-v2 上验证**: 未验证 Accent Vector 是否可迁移到其他多语言 TTS 架构(如 CosyVoice, F5-TTS)
5. **主观评估规模小**: 仅 16 名听众,70 个样本,且听众多为美国居住者,对欧洲口音区分能力有限(German/French/Spanish 混淆明显 [Fig 5])

## 点评

**优势**:
- 方法简洁优雅,核心思想(参数差作为口音向量)清晰且有数学直觉支撑
- 实验覆盖 6 种英语口音 + 3 种非英语目标语言,展示了较好的泛化性
- 混合口音合成是独特贡献,此前无方法可实现多口音组合
- 训练成本极低(8 GPU hours, 8M 参数),实用性强

**不足**:
- 论文未讨论 Accent Vector 与 Speaker Embedding 的交互 — 在零样本场景下,不同参考说话人是否影响口音迁移质量?
- 缺乏与 Emotion Vector (Murata et al., 2025; Feng et al., 2025) 的直接对比,后者同样基于 task vector 但用于情感控制
- 跨语言口音迁移的评估(§6.2)使用 VoxProfile 和 LID 作为代理,但这些模型本身未在跨语言口音上训练,评估有效性存疑
- 线性假设的理论基础薄弱 — 为什么口音属性在参数空间中是线性的?论文仅引用 Ilharco et al. 的一般性结论,未做任何分析(如 CKA/loss landscape 可视化)来验证

## 可复用的 idea

1. **LoRA 权重差作为可控属性向量**: 对任何使用 LoRA 微调的模型,θ_LoRA 本身就是一个可缩放的属性向量。可直接应用于: 情感控制(用情感数据微调 → 情感向量)、语速控制(用不同语速数据微调 → 语速向量)、噪声环境适应等
2. **Language ID 与数据的"错配"微调**: 刻意制造 language ID 和训练数据之间的冲突,迫使模型将跨语言差异编码到参数偏移中。这个 trick 可推广到任何带条件标签的生成模型
3. **多向量线性组合实现多属性控制**: τ_mixed = Σ αᵢ·τᵢ 提供了一种简单的多属性联合控制机制。如果情感向量和口音向量足够正交,可以同时控制情感和口音

## 审阅

(待独立审阅 agent 填写)
