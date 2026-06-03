---
type: paper
tier: deep
title: "Controlling Emotion in Text-to-Speech with Natural Language Prompts"
arxiv_id: "2406.06406"
source: "Sources/Controlling_Emotion_TTS_NLP.pdf"
authors: [Thomas Bott, Florian Lux, Ngoc Thang Vu]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, emotion, prompting, controllability, style-transfer, NAR-TTS, multi-speaker]
concepts: ["[[Emotion Control in TTS]]", "[[Natural Language Description for TTS]]", "[[Prosody Modeling]]", "[[Speaker Embedding]]", "[[Style Transfer in TTS]]", "[[Non-autoregressive TTS]]", "[[Global Style Tokens]]"]
models: []
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
> **谱系定位**: 本文处于 [[Emotion Control in TTS]] 演进线的"NL prompt 情感控制"阶段,介于离散 emotion embedding (Li et al., 2021) 和 LLM 自由文本情感 (EmoVoice, 2025) 之间。方法上属于 [[Natural Language Description for TTS]] 范畴 (PromptTTS/InstructTTS 同族),但聚焦于情感维度而非全属性控制。架构基于 FastSpeech 2 (属 [[Non-autoregressive TTS]]),注入方式延续 [[Speaker Embedding]] 的 conditional layernorm 策略 (AdaSpeech 系列),conditioning 信号融合借鉴 squeeze-and-excitation (SE) block。与 [[Style Transfer in TTS]] 中 GST/reference encoder 系列的区别在于: 本文用文本 (而非音频) 作为风格信号来源,且不使用 style token bank,而是直接用预训练 emotion classifier 的 hidden representation 作为 prompt embedding。
>
> **已有认知**: [[Prosody Modeling]] (confirmed) 指出 FastSpeech 2 通过 variance adaptor 显式预测 pitch/energy/duration; [[Speaker Embedding]] (confirmed) 梳理了 speaker embedding 的注入方式 (concatenation, addition, conditional layernorm, FiLM, cross-attention)。[[Emotion Control in TTS]] [待确认] 覆盖了从 emotion embedding 到 DPO 优化的完整演进。[[Natural Language Description for TTS]] [待确认] 区分了 style tagging / reference speech / NL description / instruction-guided 四种控制范式。
>
> **创新判断**: 本文的独特贡献不在架构本身 (FastSpeech 2 + conditional layernorm 是成熟方案),而在于: (1) 用情感文本本身 (而非描述性文本) 作为 prompt — 与 PromptTTS 等依赖"描述说话风格的文本"不同,本文直接用"蕴含情感的句子"作为风格信号; (2) 训练中随机采样 prompt 池增强泛化; (3) 推理时可直接用待合成文本自身作为 prompt,无需额外风格指定。
>
> 检索命中: [[Prosody Modeling]]✓, [[Speaker Embedding]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[Natural Language Description for TTS]](pending-review), [[Style Transfer in TTS]](pending-review), [[Global Style Tokens]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用蕴含情感的自然语言句子 (而非风格描述文本) 作为 prompt,通过 DistilRoBERTa emotion embedding + squeeze-and-excitation 融合驱动 FastSpeech 2 架构实现情感可控 TTS
> - **路线**: 情感文本 → DistilRoBERTa (frozen emotion classifier) → 768d CLS embedding → Linear → concat(speaker_emb) → SE block → conditional layernorm 注入 Conformer encoder/decoder/prosody predictor → Mel spectrogram → HiFi-GAN → waveform
> - **指标**: Speaker similarity cosine 0.953 (vs baseline 0.950); Cramer's V 0.80 (vs ground truth 0.85, vs baseline 0.06); MOS 3.37 (vs baseline 3.30, vs GT 3.95); 推理 RTF 0.07 (GPU) [Table 2-4]
> - **可借鉴**: (1) 用情感文本本身替代风格描述作为 prompt 源 — 训练时从 Yelp 等文本数据集提取情感句子配对语音,推理时直接用输入文本做 prompt; (2) 每次训练迭代随机采样不同 prompt 增强泛化,避免过拟合特定 prompt
> - **局限**: 仅 5 类离散情感 (anger/joy/neutral/sadness/surprise); speaker embedding 用 lookup table 非 zero-shot; 情感粒度为 utterance-level; 代码开源但数据处理pipeline复杂度较高

## 核心问题

本文要解决的核心问题是: **如何在不需要用户手动选择情感类别或提供参考音频的情况下,通过自然语言 prompt 自动控制 TTS 输出的情感韵律?**

具体挑战包括:
1. **TTS 的 one-to-many mapping**: 同一文本有无数合理的语音实现,传统 TTS 缺乏对韵律变化的控制 [§1]
2. **现有 NL description 方案的限制**: PromptTTS 等依赖手工标注的风格描述数据集 (expensive to create),且描述模板化、缺乏多样性 [§1]
3. **Speaker 与 emotion 的纠缠**: prompt conditioning 不应影响 speaker identity 的保持 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统基于 IMS Toucan toolkit [32, 33] 扩展,底层为 FastSpeech 2-like 架构,额外引入 prompt conditioning 通路 [§3.1]:

```
Input Text → Phonemizer (eSpeak-NG) → Articulatory Features → Conformer Encoder
                                                                     ↓
Prompt Text → DistilRoBERTa (frozen) → CLS embedding → Linear      ↓
                                           ↓                        ↓
Speaker ID → Embedding Matrix          concat → SE Block → Conditional LayerNorm
                                                              ↓
                                    Conformer Decoder + Prosody Predictors → Mel
                                                              ↓
                                    Post-Net Flow (PortaSpeech) → Refined Mel
                                                              ↓
                                    HiFi-GAN + Avocodo discriminator → Waveform
```

训练目标包含四个 loss [Fig 2]: L_SPEC (mel spectrogram), L_PROS (prosody predictor), L_GLOW (flow post-net), L_ADV (adversarial)。

### 关键设计选择

**1. Prompt Embedding 来源: 为什么用 emotion classifier 的 CLS 而非通用 sentence embedding?**

作者使用 DistilRoBERTa fine-tuned on emotion classification [28] 的 768 维 CLS token 表示作为 prompt embedding [§3.1]。[论文原文] "Since the emotion classification is based on the embedding of this token, it is expected to effectively capture relevant information about the emotional content of the input." [agent 解读] 这是一个关键设计 — 与 InstructTTS/PromptStyle 使用通用 sentence encoder 不同,使用 emotion-specialized encoder 保证了 embedding 空间对情感信息的选择性编码,但代价是可能丢失非情感维度的 variation information (如语速、口吻细节)。

**2. Speaker-Prompt 融合: 为什么用 Squeeze-and-Excitation block?**

Speaker embedding (lookup table) 和 prompt embedding 拼接后经过 SE block 融合 [§3.1]。[论文原文] 作者在内部 pilot study 中比较了 concatenation+projection、addition、conditional layernorm 和 SE block 四种方案,"Despite the differences being minor, we decided to go forwards with the squeeze and excitation block due to its slightly better performance in picking up fine nuances in a conditioning signal perceptually" [§3.1]。[agent 解读] SE block 的 squeeze (全局池化) + excitation (FC→ReLU→FC→Sigmoid) 机制对通道维度进行自适应加权,比简单 concat/add 能更好地建模 speaker 和 emotion 信息的通道间依赖关系,对细微 nuance 的感知有利。

**3. Conditioning 注入: 为什么在 encoder/decoder/prosody predictor 三处注入?**

SE block 输出的联合表示通过 conditional layernorm [42] 注入到 encoder、decoder 和 prosody predictor 三个位置 [§3.1]。[论文原文] "Adding the conditioning signals in multiple places is motivated by StyleTTS [43] who argue that a model quickly forgets about conditioning signals and needs to be reminded of them for more accurate conditioning." 这解释了多点注入的必要性 — 单点注入在深层网络中信号会衰减。

**4. Prompt encoder 冻结 vs Speaker embedding 可训练:**

Prompt encoder (DistilRoBERTa) 在 TTS 训练中保持冻结,而 speaker embedding matrix 随 TTS 联合训练 [§3.1]。[agent 解读] 冻结 prompt encoder 有两个好处: (1) 保留预训练的情感分类能力不被 TTS loss 破坏; (2) 减少参数更新量。但缺点是 prompt embedding 空间可能与 TTS 任务不完全对齐,仅靠一层 Linear adaptation 可能不够。

### 训练策略

**Curriculum Learning (两阶段) [§3.2]:**

| 阶段 | 数据 | 步数 | 目的 |
|------|------|------|------|
| Stage 1 | LJSpeech + LibriTTS-R + ESD + RAVDESS + TESS | 120k | 高质量 multi-speaker 基础 + robustness |
| Stage 2 | 仅 ESD + RAVDESS + TESS (情感数据集) | 80k | 聚焦 prompt-emotion 对应关系 |

**Stage 1 的 prompt 策略**: 由于 LJSpeech/LibriTTS-R 无情感标注,直接使用对应 utterance 自身作为 prompt embedding 的来源 [§3.2]。[agent 解读] 这意味着 Stage 1 的 prompt conditioning 实际上是一种弱自监督 — 从文本本身提取 embedding 再条件化回去,类似于 self-reconstruction,不产生真正的风格迁移效果,但让模型习惯 conditioning 通路的存在。

**Stage 2 的 prompt 随机采样**: 对每个训练样本,根据其 emotion label 从对应类别的 10k prompts 池中随机采样一个 prompt [§3.2]。[论文原文] "This ensures a high correspondence between prompt and speech emotion and further has the advantage that a large number of different prompts is seen which reduces the risk of overfitting and increases the generalization capabilities of the system." 这是本文的核心训练 trick — 避免模型记住特定 prompt-speech 对应关系,而是学习从 prompt embedding 中提取情感类别信息。

**硬件**: 单卡 Nvidia GeForce RTX A6000 GPU [§3.2]。

## 实验

| 指标 | 本文 (Prompt Conditioned) | Baseline (无 prompt) | Ground Truth | EmoSpeech | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Speaker Similarity (cosine) | 0.953 ± 0.0025 | 0.950 ± 0.0044 | - | 0.861 ± 0.0115 | ESD | [Table 2] |
| Cramer's V (emotion accuracy) | 0.80 (Same) / 0.80 (Other) | 0.06 | 0.85 | 0.96 | ESD | [Table 3] |
| MOS (quality) | 3.37 ± 0.31 | 3.30 ± 0.45 | 3.95 ± 0.42 | - | ESD | [Table 4] |
| Prosody Similarity (5-point) | 4.19 (F) / 4.27 (M) | - | - | - | ESD | [Table 5] |
| RTF (GPU) | 0.07 | - | - | - | - | [§3.1] |
| RTF (CPU) | 0.16 | - | - | - | - | [§3.1] |

**关键实验发现:**

1. **Prompt conditioning 不损害 speaker identity**: Speaker similarity 0.953 甚至略高于 baseline 0.950 [Table 2],说明 SE block 融合有效解耦了 speaker 和 emotion 信息。EmoSpeech 显著更低 (0.861),作者归因于本文的 multi-speaker curriculum learning [§4.2.1]。

2. **Prompt-based vs label-based emotion accuracy**: Cramer's V 0.80 vs EmoSpeech 0.96 [Table 3]。EmoSpeech 使用离散 emotion label 直接条件化,自然更精确; 但本文以 NL prompt 实现了接近 ground truth (0.85) 的情感迁移,且无需手动选择类别。[论文原文] "This offers a great advantage over the state-of-the art of specialized systems, such as EmoSpeech, at the cost of a slightly degraded emotion accuracy" [§4.2.2]。

3. **Prompt 内容 vs 输入文本 — 谁决定情感?** "Prompt Conditioned Same" 和 "Prompt Conditioned Other" 的 Cramer's V 均为 0.80 [Table 3],说明情感完全由 prompt 决定,不受输入文本影响。Baseline 的 Cramer's V 仅 0.06,表明无 prompt 时合成语音几乎无情感变化 [§4.2.2]。

4. **MOS 无显著差异**: Prompt conditioned 3.37 vs baseline 3.30,统计不显著 (t-test alpha=0.005) [Table 4],说明增加 prompt conditioning 不降低语音质量。

5. **情感迁移一致性**: 主观评估中,用同一 prompt 合成不同情感文本,prosody similarity 评分整体为 4.19/4.27 (5-point),表明情感可有效跨文本迁移 [Table 5]。其中 joy 评分相对较低 (3.30/3.63),说明快乐情感的韵律传递更困难。

## 局限性

1. **离散情感粒度**: 仅覆盖 5 类基础情感 (anger/joy/neutral/sadness/surprise) [§2.1],无法处理细粒度情感 (如讽刺、不耐烦) 或连续情感维度 (arousal-valence 空间)。

2. **非 zero-shot speaker**: Speaker embedding 使用 lookup table (closed-set) [§3.1],作者明确说明 "we choose to leave out [a pretrained speaker embedding function] for simplicity in this study",限制了在新说话人上的泛化。

3. **Prompt encoder 冻结的局限**: DistilRoBERTa embedding 空间未针对 TTS 任务微调,可能无法捕捉对韵律有影响但与情感分类无关的文本特征 (如句法结构、语气词)。仅一层 Linear adaptation 可能不够。

4. **评估数据集单一**: 所有评估均在 ESD 说话人上进行 [§4.1],未测试 LibriTTS-R 等非情感数据集的说话人,无法评估模型在自然语音上的泛化。

5. **Joy 情感传递较弱**: Table 5 和 Fig 3 显示 joy 的识别率和 similarity 评分均低于其他情感,作者未对此进行分析。

6. **缺乏与同期 NL-prompt TTS 的对比**: 未与 PromptTTS、InstructTTS 等直接可比的 NL-prompt 方法对比,仅与无 prompt 的 baseline 和 label-conditioned EmoSpeech 比较。

## 点评

**积极面:**
本文提出了一个简洁而有效的思路: 用情感文本本身 (而非描述性文本) 作为 TTS 的 prompt 信号。这比 PromptTTS 等依赖"a young woman speaks happily"类描述的方法更自然 — 在实际应用中 (如有声书、对话系统),待合成文本本身就蕴含情感,无需额外标注。训练中随机采样 prompt 池是一个简单但关键的 trick,有效防止过拟合。整体方案工程简洁,单卡可训练,代码开源。

**批判性分析:**
1. **技术创新有限**: 架构层面完全基于已有组件 (FastSpeech 2 + conditional layernorm + SE block),核心贡献更偏训练策略和数据组织。
2. **Cramer's V 解释需谨慎**: 0.80 vs EmoSpeech 的 0.96 差距不小,且依赖特定 emotion recognition model [48],该模型本身在 ESD 上训练,对 ESD 生成语音可能有 bias。
3. **"Text as prompt" 的可扩展性**: 当输入文本本身情感中性 (如技术说明) 时,用文本自身作 prompt 无法产生有意义的情感控制,仍需外部 prompt。
4. **缺乏对 prompt embedding 空间的分析**: 未可视化 embedding 空间中不同情感的分布,未分析 prompt 长度/内容对控制精度的影响。

## 可复用的 idea

1. **情感文本替代风格描述作为 prompt**: 在不需要显式风格描述的场景下,直接用蕴含目标属性的文本作为 conditioning 信号,避免了构建描述数据集的成本。可迁移到其他属性 (如从新闻文本提取正式度、从诗歌文本提取韵律风格)。

2. **Prompt 池随机采样训练策略**: 为同一 emotion label 维护大规模 prompt 池,每次训练随机采样不同 prompt,迫使模型学习情感的抽象表示而非特定文本-语音配对。可推广到任何 conditioning 训练中。

3. **SE block 融合 speaker + style**: 当需要融合两个语义不同的 embedding (如 speaker + emotion、language + style) 时,SE block 提供了一种轻量但有效的通道级自适应加权方案,比简单 concat/add 有更好的细粒度控制。

4. **Curriculum learning 两阶段分离质量与控制**: 先用大规模数据训练基础质量和 robustness,再用小规模精标数据训练细粒度控制。这种策略在数据标注成本高的可控 TTS 场景中有普遍参考价值。

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes
> - [medium] frontmatter models 字段误列 VITS (已修正为空) — 本文未使用 VITS 作为 baseline
> - [low] datasets 字段为空 — 本文使用的数据集均不在数据集库中,不强制创建
> 详见 `_review/Controlling Emotion TTS NL Prompts-review.yml`
