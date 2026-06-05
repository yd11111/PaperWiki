---
type: paper
tier: deep
title: "Multi-Reward GRPO for Stable and Prosodic Single-Codebook TTS LLMs at Scale"
arxiv_id: "2511.21270"
source: "Sources/Multi-RewardGRPO.pdf"
authors: [Yicheng Zhong, Peiji Yang, Zhisheng Wang]
year: 2025
venue: "arXiv (ACM Conference submission)"
tags: [TTS, reinforcement-learning, GRPO, single-codebook, prosody, post-training, LLM-TTS]
concepts: ["[[LLM-basedTTS]]", "[[ProsodyModeling]]", "[[Single-codebookvsMulti-codebook]]", "[[ConditionalFlowMatching]]", "[[SpeakerEmbedding]]", "[[DifferentiableRewardOptimization]]", "[[SpeakerVerification]]", "[[TTSEvaluation]]"]
models: ["[[Whisper]]", "[[WavLM]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文位于 "RL post-training for TTS" 研究线上。这条线的演进为: Seed-TTS (2024, REINFORCE audio-level RL) -> SpeechAlign (2024, DPO preference optimization) -> DiffRO/CosyVoice 3 (2025, token-level differentiable RL) -> 本文 Multi-Reward GRPO (2025, audio-level 多奖励 GRPO)。同期工作 RL-for-Audio-LLM (Tongyi, 2025) 首次公平对比 GRPO vs DiffRO,发现 DiffRO 降 WER 更强但可能伤 SIM,GRPO 训练超 1500 步后退化。本文通过多奖励设计可能缓解 GRPO 的退化问题。
>
> **已有认知**:
> - [[LLM-basedTTS]] (confirmed): 单码本 TTS LLM 是 LLM-based TTS 的子范式,直接生成联合语义-声学 token,优势是紧凑且可流式,代价是韵律不稳定和说话人漂移。LLaSA 是本文使用的 backbone,已有笔记 [[论文笔记/Llasa|Llasa]]。
> - [[ProsodyModeling]] (confirmed): 传统韵律建模从规则 -> GST/VAE -> FastSpeech 2 variance adaptor -> LLM 时代 in-context learning。LLM-TTS 隐式建模使细粒度韵律控制困难,是已知的关键挑战。本文的 LLM-annotated prosody reward 是一种显式韵律监督回到 LLM-TTS 的尝试。
> - [[ConditionalFlowMatching]] (confirmed): CFM 作为 coarse-to-fine 的 fine stage,在 CosyVoice 系列中将 speech token 转为 mel spectrogram。本文在 GRPO 优化后的 AR backbone 上附加 FM decoder,验证 RL 优化与声学精炼的互补性。
> - [[SpeakerEmbedding]] (confirmed): 本文使用 WavLM-large fine-tuned for speaker verification 提取说话人嵌入计算余弦相似度作为 R_sim 奖励,这与 SEED-TTS-Eval benchmark 的标准评估方式一致。
> - [[DifferentiableRewardOptimization]] [待确认]: DiffRO 在 token 空间操作,GRPO 在 audio 空间操作。Tongyi 的对比发现两者互补但直接合并会变差,需要 sample filter。本文的多奖励 GRPO 是另一种解决方案。
> - [[Single-codebookvsMulti-codebook]] [待确认]: 单码本路线 (SVQ) 的核心优势是低 token rate 和标准 AR LM 建模,但重建质量通常低于 RVQ。本文关注的正是单码本路线带来的韵律和稳定性代价。
>
> **创新判断**: 本文的核心创新是将 GRPO 扩展为多奖励框架,特别是引入 LLM-annotated prosody alignment reward。相比 DiffRO (token-level, 需要 Gumbel-Softmax),GRPO 在 audio-level 操作更直观;相比 Seed-TTS REINFORCE (仅 SIM+WER),多奖励设计增加了显式韵律监督、长度约束和熵正则化。
>
> 检索命中: [[LLM-basedTTS]]✓, [[ProsodyModeling]]✓, [[ConditionalFlowMatching]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[Single-codebookvsMulti-codebook]](pending-review), [[DifferentiableRewardOptimization]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在单码本 TTS LLM (LLaSA) 上用多奖励 GRPO 做 RL 后训练,通过五个奖励函数 (WER + SIM + 长度约束 + 熵正则 + LLM 标注韵律对齐) 系统性提升韵律稳定性、说话人相似度和自然度
> - **路线**: Text+Ref Speech -> Single-codebook TTS LLM (LLaSA 1B/3B/8B) -> N rollouts -> 多奖励评估 (Whisper WER + WavLM SIM + length + entropy + prosody match) -> GRPO policy update; 可选 +FM decoder
> - **指标**: LLaSA+RL: CER 1.10 (zh) / WER 2.12 (en) / SIM 0.758 (zh) / MOS 4.12; +FM: CER 1.08 / WER 2.08 / SIM 0.790 / MOS 4.21; 优于 Spark-TTS / CosyVoice 1&2,接近 CosyVoice 3 (使用 4x 数据) [Table 1]
> - **可借鉴**: 1) 用推理型 LLM (DeepSeek-R1) 离线标注韵律结构作为 RL 监督信号; 2) 多奖励拆分设计使每个奖励功能明确且可独立消融; 3) 10K samples 即可获得可测量增益,RL 数据效率高于 SFT
> - **局限**: 仅 4 页短文,细节不足 (如 prosody reward 的 pause mapping 规则未完整给出); R_pro 是 binary reward,可能信号稀疏; 未与 DiffRO/DPO 做直接对比; 未开源

## 核心问题

单码本 TTS LLM (将语义和声学信息联合编码到一个 codebook 中) 虽然结构紧凑且支持流式,但由于单一 token 空间需要同时承载语义、韵律和声学信息,其自回归解码策略容易产生次优行为: **韵律不稳定** (节奏异常)、**说话人漂移** (音色随生成过程变化) 和 **长度失控** (过早停止或过长生成) [§1]。

核心问题: 如何通过 RL 后训练直接优化单码本 TTS LLM 的解码策略,使其在不改变模型架构的前提下系统性改善上述三个问题?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

基于 GRPO (Group Relative Policy Optimization) [§2.1], 将单码本 TTS LLM 视为策略 pi_theta(a_t|s_t),通过多组 rollout 生成候选语音,用五个奖励函数评估质量,然后用 group-wise advantage normalization 更新策略 [论文原文]。

总奖励函数为加权和 [§2.1, Eq.2]:
```
R = alpha_intl * R_intl + alpha_sim * R_sim + alpha_len * R_len + alpha_ent * R_ent + alpha_pro * R_pro
```

其中 alpha_intl = alpha_sim = alpha_ent = alpha_pro = 1.0, alpha_len = 0.1 [§3.1]。

### 关键设计选择

#### 1. 为什么选 GRPO 而非 PPO/DPO/DiffRO?

GRPO 通过 group-wise advantage normalization 稳定学习,不需要 dense preference labels (vs DPO) 也不需要 critic network (vs PPO) [论文原文, §1]。[agent 解读] 相比 DiffRO 需要 Gumbel-Softmax 和 token-level reward model,GRPO 在 audio-level 操作,与已有 ASR/speaker verification 工具链天然兼容。

#### 2. 五个奖励的设计逻辑

**R_intl (可懂度, §2.2)**: 用 Whisper ASR 将生成音频转写,计算与输入文本的 Levenshtein 距离,归一化为 1 - CER/WER [论文原文]。目的是确保语义一致性。

**R_sim (说话人相似度, §2.2)**: 用 WavLM-large (fine-tuned for speaker verification) 提取生成音频和参考音频的 speaker embedding,计算余弦相似度 [论文原文]。目的是防止说话人漂移。

**R_len (长度惩罚, §2.3)**: 根据参考文本长度和参考语速估算目标时长 T_target,设定容忍范围 [a, b],如果生成时长落在范围内则 reward=1,否则 reward=0 [Eq.5]。[论文原文] 这解决了 AR TTS 常见的 premature stopping 或过长生成问题。[agent 解读] 作为 binary reward,alpha_len=0.1 的低权重可能是为了避免其对策略更新的过大影响。

**R_ent (熵正则化, §2.4)**: 计算生成轨迹上的平均 token-level 熵 H_bar,与从高质量样本估算的 H_target 比较,惩罚过高熵 [Eq.6]。[论文原文] 过高熵导致不稳定的韵律变化,该奖励鼓励更平滑的生成路径。[agent 解读] 这实质是在策略空间中约束探索幅度,与 KL 约束的思路一致但操作在 entropy 维度。

**R_pro (LLM 标注韵律对齐, §2.5)**: 这是本文最核心的创新。分为两阶段:
- *离线标注*: 用 DeepSeek-R1 (推理型 LLM) 通过 few-shot in-context learning 为输入文本标注多种 plausible 停顿结构。中文用 #1-#4 离散停顿标记,英文用 PW/PPH 韵律标签 [论文原文]。
- *在线比较*: 训练时,用 Whisper 对生成的波形做 timestamped 解码,通过规则映射将静音时长转为离散停顿符号,与离线标注的模式集合比对。匹配则 reward=1,否则 reward=0 [Eq.7]。

[论文原文] DeepSeek-R1 生成的是多种 plausible 停顿结构 (而非单一标准),这提供了 human-preference-aligned 的监督信号。[agent 解读] 使用推理型 LLM 做韵律标注是一个巧妙的设计: (1) 利用 LLM 的语言理解能力捕捉语义韵律关系; (2) 多模式匹配避免了单一标准答案的过拟合; (3) 离线标注一次可复用,不增加训练成本。但 binary reward 可能导致信号稀疏 --- 如果大多数 rollout 都不匹配,梯度信号弱。

### 训练策略

- 硬件: 8xH20 GPU, mixed precision [§3.1]
- GRPO: batch size 16, learning rate 1e-6, group size 12 [§3.1]
- 解码: vLLM, top-k=75, top-p=0.9, temperature=1.1, repetition penalty=1.1 [§3.1]
- 训练数据: 从 Emilia + LibriHeavy 采样 1M (text, ref_speech, target_text) 三元组,中英 1:1 平衡,约 5115 小时音频 [§3.1]
- Base model: LLaSA-8B (单码本 TTS LLM) [§3.1]

## 实验

| 指标 | 本文 (LLaSA+RL) | 本文 (+FM) | LLaSA baseline | CosyVoice 3 | Seed-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CER (zh) ↓ | 1.10 | 1.08 | 1.59 | 1.12 | 1.12 | SEED test-zh | [Table 1] |
| SIM (zh) ↑ | 0.758 | 0.790 | 0.684 | 0.781 | 0.796 | SEED test-zh | [Table 1] |
| WER (en) ↓ | 2.12 | 2.08 | 2.97 | 2.21 | 2.62 | SEED test-en | [Table 1] |
| SIM (en) ↑ | 0.672 | 0.733 | 0.574 | 0.720 | 0.714 | SEED test-en | [Table 1] |
| CER (hard) ↓ | 6.04 | 5.98 | 10.63 | 5.83 | 7.59 | SEED test-hard | [Table 1] |
| SIM (hard) ↑ | 0.731 | 0.775 | 0.674 | 0.758 | 0.776 | SEED test-hard | [Table 1] |
| MOS ↑ | 4.12 | 4.21 | 3.76 | - | 3.53 | SEED (100 samples) | [Table 1] |

### 关键发现

**RL vs SFT 数据效率** [§3.2]: LLaSA+SFT (CER 1.51 zh) 改进有限,而 LLaSA+RL (CER 1.10 zh) 大幅改进,说明 RL 的样本效率显著优于 SFT [论文原文]。

**RL + FM 互补性** [§3.2]: 在 RL 优化后附加 FM decoder 持续带来增益 (CER 1.10→1.08, SIM 0.758→0.790, MOS 4.12→4.21),证明 RL 优化增强的是 AR 的内在策略,与下游 acoustic refiner 不重叠 [论文原文]。

**Scaling analysis** [§3.3, Fig 2]: 1B/3B/8B 模型在 1K→10K→100K→1M 数据上呈现明确单调趋势 --- 更大模型 + 更多数据 = 更好 CER 和 SIM。即使 10K 样本的小规模 RL 也能带来可测量增益 [论文原文]。

**消融** [§3.4, Table 2]: 各奖励的增量贡献 (以 zh MOS 为例):
- Baseline (LLaSA): 3.68
- +R_intl & R_sim: 3.77 (+0.09)
- +R_len: 3.81 (+0.04)
- +R_ent: 4.01 (+0.20)
- +R_pro (Full): 4.25 (+0.24)

R_pro 贡献最大 (+0.24 MOS),说明显式韵律监督对人类感知偏好的对齐效果最显著 [论文原文]。R_ent 贡献第二 (+0.20 MOS),说明解码稳定性对自然度影响很大 [论文原文]。

## 局限性

1. **论文篇幅极短 (4 页)**: 许多关键细节缺失 --- prosody reward 中 silence-to-pause 的规则映射未给出,R_len 的容忍范围 [a,b] 值未指定,GRPO 的 KL 约束系数未讨论 [agent 解读]
2. **R_pro 为 binary reward**: 匹配为 1 否则为 0,这可能导致 reward signal 稀疏。如果 rollout group 中所有样本都不匹配,则该奖励对 advantage 计算无贡献 [agent 解读]
3. **未与 DiffRO/DPO 直接对比**: RL-for-Audio-LLM (Tongyi) 已有 GRPO vs DiffRO 的公平对比,但本文未引用也未讨论 [agent 解读]
4. **R_pro 依赖 Whisper timestamp 质量**: 如果 Whisper 的时间戳不准确 (尤其对于非标准韵律的生成结果),pause detection 会引入噪声 [agent 解读]
5. **评估局限**: MOS 仅 100 样本 × 10 评估者,统计显著性有限;未报告推理速度或 RL 训练成本 [agent 解读]
6. **未开源**: 模型、代码、prosody 标注数据均未公开 [论文原文无开源声明]

## 点评

本文的核心贡献是将 GRPO 扩展为 TTS 领域的多奖励框架,特别是 LLM-annotated prosody alignment reward 的设计颇具创意。用推理型 LLM (DeepSeek-R1) 做韵律标注是一个低成本高收益的方案: 不需要人工标注,不需要训练专门的韵律评估模型,且一次标注可反复使用。

消融实验设计清晰,每个奖励的贡献被独立量化,R_pro 和 R_ent 的高贡献验证了 "显式韵律监督 + 解码稳定性" 是单码本 TTS LLM 的关键瓶颈。Scaling analysis 的结论 (RL 与模型/数据规模正相关) 对实际部署有指导意义。

但论文的主要弱点是篇幅太短导致关键细节缺失。与同期 Tongyi 的 RL-for-Audio-LLM 工作相比,本文未讨论 GRPO 的训练稳定性问题 (Tongyi 发现 GRPO 超 1500 步后退化),也未探索 GRPO + DiffRO 的组合方案。此外,与 CosyVoice 3 的对比应注意: CosyVoice 3 使用约 1M 小时预训练数据,本文 LLaSA 预训练数据量未明确 (论文仅提及 GRPO 训练使用 5115 小时音频),但论文声称 CosyVoice 3 "benefits from substantially larger training data (1M h vs. our 250k h)" [§3.2],因此 LLaSA 预训练约 250K 小时,数据规模差异约 4 倍,本文在数据效率上有优势但绝对性能仍有差距。

## 可复用的 idea

1. **LLM-annotated prosody reward**: 用推理型 LLM (如 DeepSeek-R1) 离线标注文本的多种合理韵律结构 (停顿位置),作为 RL 训练的 binary reward。可迁移到任何需要韵律优化的 TTS 系统,成本极低。
2. **多奖励解耦设计**: 将 TTS 质量拆分为可懂度、说话人一致性、长度、解码稳定性、韵律五个独立维度,每个维度一个奖励函数。这种设计使每个维度的贡献可独立消融和调优。
3. **Entropy regularization for AR TTS**: 用平均 token 熵与目标熵的差异作为奖励,约束 AR 解码器的探索幅度。这比 KL 约束更直接地针对 "解码不稳定" 问题。
4. **RL 数据效率优于 SFT**: 10K 样本即可获得可测量增益,暗示 RL 后训练可作为低成本的性能提升手段,不需要大规模标注数据。

> [!review] 审阅状态: pass-with-fixes (2026-06-03)
> 3 个 low 级问题: (1) 训练数据量 250K h vs 5115 h 已澄清, (2) MOS 缺失值用 '-' 标注合理, (3) frontmatter models 未列 LLaSA backbone。均不阻塞反向更新。详见 `_review/Multi-Reward GRPO-review.yml`
