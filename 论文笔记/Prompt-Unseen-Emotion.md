---
type: paper
tier: deep
title: "Prompt-Unseen-Emotion: Zero-shot Expressive Speech Synthesis with Prompt-LLM Contextual Knowledge for Mixed Emotions"
arxiv_id: "2506.02742"
source: "Sources/Prompt-Unseen-Emotion.pdf"
authors: [Xiaoxue Gao, Huayun Zhang, Nancy F. Chen]
year: 2025
venue: "arXiv (Interspeech 2025 投稿)"
tags: [TTS, emotion, zero-shot, mixed-emotion, LLM-based, prompt-learning, in-context-learning, CosyVoice, Plutchik]
concepts: ["[[Emotion Control in TTS]]", "[[LLM-based TTS]]", "[[Instruction-Guided Speech Synthesis]]", "[[Speech Tokenizer]]", "[[Conditional Flow Matching]]", "[[Natural Language Description for TTS]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[LLM-based TTS]], [[Speech Tokenizer]]; 4 个待确认实体页: [[Emotion Control in TTS]], [[Instruction-Guided Speech Synthesis]], [[Natural Language Description for TTS]], [[Conditional Flow Matching]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: PUE 处于**混合情感 TTS** 这一子领域,在 [[Emotion Control in TTS]] [待确认] 的演进线上位于: 离散情感标签 (Tacotron, 2020) → Plutchik 结构模型 + 韵律嵌入分解 ([[论文笔记/Daisy-TTS|Daisy-TTS]], 2024) → VITS-based rank scheme 混合情感 (Zhou et al., 2022) → **LLM prompt-based 混合情感 (PUE, 2025)**。PUE 是混合情感 TTS 中首次使用 LLM in-context learning 能力的工作。架构上基于 [[模型库/CosyVoice|CosyVoice]] (confirmed) 的 LLM + flow-matching + HiFi-GAN pipeline,属于 [[LLM-based TTS]] (confirmed) 的 hybrid 路线。
>
> **已有认知**: KB 中 [[LLM-based TTS]] (confirmed) 已系统梳理了 LLM-based TTS 的核心设计 (离散 token、两阶段生成、in-context learning),并将 CosyVoice 归类为 "LLM + Flow/Diffusion hybrid" 路线。[[模型库/CosyVoice|CosyVoice]] (confirmed) 记录了 S3 supervised semantic tokenizer + LLM + OT-CFM 的具体架构和 CosyVoice-Instruct 变体。[[Speech Tokenizer]] (confirmed) 分析了监督式 semantic tokenizer 的设计权衡。[[Emotion Control in TTS]] [待确认] 中 Daisy-TTS 的条目已记录了基于 Plutchik 模型的韵律嵌入分解方法,Zhou et al. (2022) 的 VITS-based rank scheme 是唯一的 prior work for mixed-emotion TTS。
>
> **创新判断**: 相比 KB 中已有知识,PUE 的关键新贡献是: (1) 将混合情感建模从 VITS 架构的 rank-based 方案升级到 LLM-based 架构的 prompt-based 方案,利用 LLM 的 in-context learning 而非手工设计的混合机制; (2) 通过 emotion-guided prompt (α/β/γ/ε/λ 百分比) 实现训练时单一情感、推理时任意混合,是一种 zero-shot compositionality; (3) 比 Daisy-TTS 的 PCA 分解更简洁直接 — 不需要额外的 embedding 空间操作,直接通过 prompt 文本控制。
>
> 检索命中: [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[Instruction-Guided Speech Synthesis]](pending-review), [[Natural Language Description for TTS]](pending-review), [[Conditional Flow Matching]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 CosyVoice LLM-TTS 架构,通过 emotion-guided prompt (α%/β%/γ%/ε%/λ% 百分比模板) 训练时学习单一情感-prompt 一致性,推理时调整百分比实现零样本混合情感语音合成
> - **路线**: Emotion-guided prompt (EP: "A man/woman speaks ... α% happy, β% sad, ...") + Input text → [Text Encoder + LLM Decoder (init from CosyVoice)] → Emotional speech tokens → [Frozen Flow-Matching + Frozen HiFi-GAN] → Mixed-emotion waveform
> - **指标**: WER: PUE 25.19-26.67% vs cosy 39.26-51.85% vs mix 27.41-32.59% (各混合条件) [Table I]; AB Preference: PUE vs mix 68-87% 偏好 PUE [Fig 3]; MOS (unseen emotions): PUE outrage 3.57±0.18, disappointment 3.12±0.16, delight 3.50±0.13 (均高于 cosy 和 mix baselines) [Fig 5, Table III]
> - **可借鉴**: (1) 用百分比 prompt 模板实现 zero-shot compositionality — 训练时每个样本只有一种情感 (对应参数=100, 其余=0),推理时混合任意比例即可; (2) 利用 LLM 的 in-context learning 隐式理解情感比例语义,无需设计专门的混合机制; (3) 用 label smoothing KL loss 保证情感概率分布一致性
> - **局限**: (1) 仅在 ESD 数据集 2 说话人 (~2.4h 总计) 上验证,泛化性未知; (2) 仅 5 种基础情感,混合实验仅以 surprise 为基底; (3) WER 绝对值较高 (25-27%),语音可懂度有限; (4) 主观评测仅 18 人; (5) 代码尚未开源 (upon acceptance); (6) 未与 Daisy-TTS 等 Plutchik 模型方法直接对比

## 核心问题

本文要解决的核心问题是: **如何生成训练数据中不存在的"混合情感"语音** — 例如 disappointment (surprise + sadness)、outrage (surprise + anger)、delight (surprise + happiness) [§1]。

现有方法的局限:
1. **数据受限**: 情感 TTS 数据集最多包含 8 种离散情感类别,但人类可体验约 34,000 种情感状态,且常同时感受多种情感 [§1, 引用 Plutchik 情感轮理论]
2. **架构受限**: 传统方法 (Tacotron/FastSpeech/VITS/diffusion) 只能复制训练集中出现的情感类型,无法外推 [§1]
3. **Prior work 局限**: 唯一的 prior mixed-emotion TTS 工作 (Zhou et al., 2022) 使用 VITS-based 的 rank scheme,语音质量受限于 VITS 架构 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

PUE 建立在 CosyVoice-300M-Instruct 之上 [§3.A],包含四个组件:

1. **Speech Tokenizer**: 将语音编码为离散 emotional speech tokens (继承自 CosyVoice 的 S3 tokenizer) [§2.B]
2. **Emotion-Prompt LLM-TTS Model**: 包含 text encoder + auto-regressive LLM decoder,接受 emotion-guided prompt + input text,预测 emotional speech tokens [§2.B]
3. **Flow-Matching Model**: 预训练的 flow-matching 模型,将 speech tokens 转为 mel spectrogram [§2.A],推理时冻结
4. **HiFi-GAN Vocoder**: mel → waveform [§2.A],推理时冻结

### 关键设计选择

**1. Emotion-guided Prompt 设计 [§2.B]**

核心创新: 将情感控制编码为文本 prompt 模板:

```
EP = "A man/woman speaks an utterance with,
      α percent happy emotion,
      β percent sad emotion,
      γ percent neutral emotion,
      ε percent angry emotion,
      and λ percent surprise emotion."
```

**为什么用百分比 prompt 而不是 emotion embedding?** [论文原文] 作者认为通过将情感编码为自然语言百分比描述,可以利用 LLM 的 in-context learning 和 instruction-following 能力,使模型在训练时仅见过单一情感 (如 α=100, 其余=0) 的情况下,推理时能理解混合比例 (如 α=30, λ=100) 的语义 [§2.B]。[agent 解读] 这本质上是一种 compositional generalization: LLM 对数字和比例的理解能力使其无需见过混合样本即可推理混合语义。

**训练时的 prompt 构造 [§2.B]**: 每个训练样本的 emotion prompt 根据其单一情感标签生成 — 如果是 happy 样本,则 α=100, 其余=0。数据集中没有混合情感样本。

**2. 训练输入格式 [§2.B, Eq.1]**

$$d_i \in D_e = EP \langle EOP \rangle t_i \langle T \rangle s_i \langle E \rangle$$

其中 EP 是 emotion prompt, $\langle EOP \rangle$ 标记 prompt 结束, $t_i$ 是文本 token, $\langle T \rangle$ 是 text-to-speech 转换标记, $s_i$ 是 emotional speech tokens。

**3. Emotion-guided KL Loss [§2.C, Eq.3]**

采用 label smoothing KL 散度损失:

$$L_{PUE} = KL(P_\theta || P) = E_{d_i \sim D_e} \left[ p(s_i | EP, t_i) \log \frac{p(s_i | EP, t_i)}{p_\theta(s_i | EP, t_i)} \right]$$

**为什么用 KL loss 而不是标准 cross-entropy?** [论文原文] 受 CosyVoice 中 label smoothing KL loss 在语音生成中的成功启发 [§2.C, 引用 37]。[agent 解读] KL loss 中的 label smoothing 可以防止模型对单一 token 过度自信,有利于混合情感场景中的平滑分布。

### 训练策略

- **初始化**: LLM-TTS model、speech tokenizer 和 text encoder 全部从 CosyVoice 初始化,保持相同架构配置 [§3.A]
- **训练规模**: 仅 1 epoch,使用 dynamic batching,4 GPU [§3.A]
- **冻结组件**: 推理时 flow-matching model 和 HiFi-GAN 均冻结 [§3.A]

**为什么只训练 1 epoch?** [agent 解读] 由于 LLM-TTS 从 CosyVoice 预训练模型初始化,主要需要学习的是 emotion prompt 到 speech token 的映射;训练数据极少 (~2.4h, 3500 utterances per speaker),过多训练可能导致过拟合。

### 推理时的混合情感生成 [§2.B, §3.A]

推理时通过修改 prompt 中的百分比参数实现混合:
- **Outrage** (愤怒的惊讶): λ=100 (surprise), ε=30/60/90 (angry)
- **Disappointment** (失望): λ=100 (surprise), β=30/60/90 (sad)
- **Delight** (欣喜): λ=100 (surprise), α=30/60/90 (happy)

## 实验

| 指标 | PUE | cosy baseline | mix baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER (original test) | 26.67% | 51.85% | 32.59% | ESD | [Table I] |
| WER (surprise+30% others) | 25.19% | 47.41% | 32.59% | ESD | [Table I] |
| WER (surprise+60% others) | 25.93% | 42.22% | 31.11% | ESD | [Table I] |
| WER (surprise+90% others) | 25.93% | 39.26% | 27.41% | ESD | [Table I] |
| AB Pref: PUE vs mix (surprise+happy) | 87.04% vs 9.26% | — | — | ESD | [Fig 3] |
| AB Pref: PUE vs mix (surprise+angry) | 74.07% vs 25.93% | — | — | ESD | [Fig 3] |
| AB Pref: PUE vs mix (surprise+sad) | 68.52% vs 31.48% | — | — | ESD | [Fig 3] |
| AB Pref: PUE vs cosy (surprise+happy) | 81.48% vs 11.11% | — | — | ESD | [Fig 4] |
| BWS (90% angry = strongest outrage) | 94.44% best | — | — | ESD | [Table II] |
| MOS (surprise+90% others) | 3.57±0.18 | — | — | ESD | [Table III] |

**关键发现**:
1. **WER**: PUE 在所有条件下 WER 均最低,说明 LLM-based 架构在保持可懂度方面优于 VITS-based mix baseline [Table I]
2. **AB Preference**: PUE 在 surprise+happy 组合中获得最高偏好 (87%),作者解释为 surprise 和 happy 共享声学特征 (高 pitch、快语速、高能量) [§4.D.1] [论文原文]
3. **BWS 测试**: 94.44% 的听者认为 90% angry 的版本表达了最强的 outrage,证明百分比参数确实控制了情感强度 [Table II]
4. **MOS**: surprise+90% 混合的 MOS 高于 30% 和 60% 混合 [Table III],说明更高的次要情感比例产生更好的感知效果 [§4.C.2]

## 局限性

1. **数据规模极小**: 仅 ESD 数据集 2 说话人 (~1.2h/speaker),难以验证大规模泛化能力 [agent 解读]
2. **WER 绝对值高**: PUE 最佳 WER 也有 25.19%,意味着约 1/4 的词被误识,可懂度仍有明显问题 [Table I]
3. **混合维度受限**: 仅以 surprise 为基底与其他三种情感混合,未探索任意两种情感的混合 [§3.A]
4. **主观评测规模小**: 仅 18 名听者,统计功效有限 [§3.C]
5. **无细粒度对比**: 未与 Daisy-TTS 等使用 Plutchik 模型的方法对比,也未与 EmoCtrl-TTS (帧级 arousal-valence 控制) 对比 [agent 解读]
6. **未开源**: 代码尚未公开 (承诺 upon acceptance) [§5]

## 点评

**优势**:
- 思路清晰简洁: 利用 LLM 的 prompt 理解能力实现 zero-shot compositionality,不需要复杂的混合机制
- 实验设计合理: BWS 测试直接验证了百分比参数对情感强度的可控性
- 站在巨人肩膀上: 直接基于 CosyVoice 初始化,聚焦 emotion prompt 这一增量创新

**不足**:
- **验证过于初步**: 2 说话人、5 种基础情感、仅 surprise 为基底的混合、18 人主观评测 — 这些都不足以支撑论文中"successfully facilitates expressive speech synthesis of unseen emotions"的结论 [§5]
- **WER 问题被忽视**: 即使 PUE 的 WER 最低,25% 的 WER 在实际应用中完全不可接受,论文未讨论这个根本问题
- **与 Daisy-TTS 的关系未讨论**: 两者都基于 Plutchik 情感轮理论解决混合情感,但方法路线完全不同 (PCA 分解 vs prompt learning),缺少对比分析
- **可控性验证不充分**: BWS 只测了 angry 一种次要情感的强度梯度,sad 和 happy 的梯度未验证

## 可复用的 idea

1. **百分比 prompt 模板实现 zero-shot 组合**: 训练时仅单一属性 (target=100%, rest=0%),推理时混合任意比例 — 这种范式不限于情感,可推广到任意可枚举属性的混合控制 (如口音混合、风格混合)
2. **利用 LLM 对数字/比例的理解能力**: LLM 天然理解 "30% happy + 70% sad" 的语义,无需学习 embedding 空间的插值操作
3. **冻结下游组件 (flow-matching + vocoder) 仅训练 prompt-to-token 映射**: 降低训练成本,聚焦核心创新

> [!review] 审阅结论: pass-with-fixes
> 审阅日期: 2026-06-03 | 详见 [[_review/Prompt-Unseen-Emotion-review.yml]]
> - medium: 部分实验数字缺出处标注 (MOS from Fig 5, BWS 具体数据标注可改进)
> - medium: 与 Daisy-TTS 的 Plutchik 模型路线对比缺失,但属于论文本身缺陷而非笔记问题
> - low: "路线"字段可进一步细化 speech tokenizer 类型

---

检索命中: [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[Instruction-Guided Speech Synthesis]](pending-review), [[Natural Language Description for TTS]](pending-review), [[Conditional Flow Matching]](pending-review) | 未命中但可能相关: 无
