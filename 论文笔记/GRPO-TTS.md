---
type: paper
tier: deep
title: "Group Relative Policy Optimization for Text-to-Speech with Large Language Models"
arxiv_id: "2509.18798"
source: "Sources/GRPO-TTS.pdf"
authors: [Chang Liu, Ya-Jun Hu, Ying-Ying Gao, Shi-Lei Zhang, Zhen-Hua Ling]
year: 2025
venue: "arXiv (ICASSP submission, 4 pages)"
tags: [TTS, reinforcement-learning, GRPO, post-training, LLM-TTS, reward-design, CER, NLL]
concepts: ["[[LLM-basedTTS]]", "[[SemanticvsAcousticTokens]]", "[[ConditionalFlowMatching]]", "[[SpeakerEmbedding]]", "[[SpeakerVerification]]", "[[DifferentiableRewardOptimization]]", "[[TTSEvaluation]]", "[[Gumbel-Softmax]]"]
models: ["[[Whisper]]", "[[CosyVoice2]]", "[[WavLM]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文位于 "RL post-training for LLM-based TTS" 研究线上。演进路径: Seed-TTS (2024, PPO/REINFORCE with WER+SIM) -> SpeechAlign (2024, DPO preference optimization) -> DiffRO/CosyVoice 3 (2025, token-level differentiable RL) -> Multi-Reward GRPO (2025, audio-level 多奖励 GRPO on LLaSA) -> 本文 GRPO-TTS (2025, CER+NLL composite reward on CosyVoice2 + LLaSA)。本文与 Multi-Reward GRPO 是同期独立工作,核心差异在于 reward 设计: Multi-Reward GRPO 用 5 个 reward (WER+SIM+length+entropy+prosody),本文用 CER+NLL 加权调和平均,且同时验证了两类 LLM-TTS 架构 (semantic token 路线 CosyVoice2 + acoustic token 路线 Llasa-1B)。
>
> **已有认知**:
> - [[LLM-basedTTS]] (confirmed): LLM-based TTS 分两大类: (1) 直接建模 acoustic codec token 的系统 (VALL-E, Llasa); (2) 建模 semantic token + flow matching 补充声学细节的系统 (CosyVoice 系列)。本文首次在同一 RL 框架下同时微调这两类系统,验证 GRPO 对不同 token 路线的普适性。
> - [[SemanticvsAcousticTokens]] (confirmed): 语义 token 与文本对齐良好但缺高频声学细节,声学 token 保真度高但语义对齐差。CosyVoice2 使用 ASR 监督的 semantic tokens (S3 tokenizer),Llasa-1B 使用 neural codec acoustic tokens。本文发现 GRPO 在两类系统上均能显著降低 CER/WER,但只有 semantic token 路线的 CosyVoice2 在自然度 (MOS) 上也获得显著改进,acoustic token 路线的 Llasa-1B 自然度无显著变化。
> - [[CosyVoice2]] (confirmed): Tongyi Lab 的流式零样本 TTS,使用 FSQ-SenseVoice tokenizer + LLM + chunk-aware flow matching,支持多语言和指令控制。基线性能: CER 1.45% (zh), WER 2.57% (en) on SEED-TTS-Eval。本文以 CosyVoice2 为基线验证 GRPO 有效性。
> - [[SpeakerEmbedding]] (confirmed): 本文使用 WavLM fine-tuned for speaker verification 提取说话人嵌入计算余弦相似度 (SIM) 作为评估指标,这与 SEED-TTS-Eval benchmark 的标准评估方式一致。
> - [[DifferentiableRewardOptimization]] [待确认]: DiffRO 在 token 空间操作,需要 Gumbel-Softmax 和预训练 token-to-text reward model。本文的 GRPO 在 audio 空间操作,使用现成 ASR 模型 (Whisper) 计算 reward,避免了额外模型训练,是 DiffRO 的竞争替代方案。本文在 introduction 中明确将 DiffRO 作为对比方法批评其"额外计算和数据成本"。
> - [[Whisper]] [待确认]: 本文使用 Whisper-large-v3 作为 reward model 的核心组件: (1) 计算 CER reward (ASR 转写 → 与 ground truth 比对); (2) 计算 NLL reward (ASR decoder 对 ground truth text 的负对数似然)。Whisper 同时服务于 reward 计算和评估 (英文 WER)。
>
> **创新判断**: 本文的核心创新是利用现成 ASR 模型的 NLL 作为 CER 的互补 reward 信号,通过加权调和平均 (harmonic mean) 组合两者。相比 Multi-Reward GRPO 的 5 维 reward,本文的设计更简约,且不需要额外模型 (如 DeepSeek-R1 做韵律标注)。另一个重要发现是 GRPO 对不同 token 类型 TTS 系统的差异化效果。
>
> 检索命中: [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓, [[CosyVoice2]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[DifferentiableRewardOptimization]](pending-review), [[Whisper]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用现成 ASR 模型 (Whisper) 的 CER + NLL 构建复合 reward,通过 GRPO 微调两类 LLM-based TTS (CosyVoice2 / Llasa-1B),在不引入额外训练模型的前提下显著提升语音可懂度和自然度
> - **路线**: Text → TTS LLM (CosyVoice2 或 Llasa-1B) → G=8 rollouts → Codec Decoder/FM+Vocoder → Speech → Whisper ASR → CER reward + NLL reward → 调和平均 → GRPO policy update
> - **指标**: CosyVoice2+GRPO: CER 1.07 (zh, from 1.41) / WER 2.30 (en, from 2.46) / MOS 4.58 (zh, from 4.42); Llasa+GRPO: CER 1.30 (zh, from 7.73!) / WER 2.17 (en, from 4.95) [Table 1, Table 2]
> - **可借鉴**: 1) ASR 模型的 NLL 作为 CER 的互补 reward — 在 CER=0 时仍提供区分度; 2) 调和平均组合 reward 对低值更敏感,避免单维度掩盖缺陷; 3) GRPO 仅需 4000 句训练数据即可显著改善两类 TTS 系统
> - **局限**: 仅 4 页会议论文,细节有限; 未与 DiffRO/DPO 做实验对比; Llasa-1B 自然度无显著改善未深入分析; 代码/模型已开源但论文未详述超参数搜索过程

## 核心问题

LLM-based TTS 的自回归采样虽然能生成多样且韵律自然的语音,但也导致模型有时生成与人类偏好不一致的语音 — 内容错误 (漏字/误读)、自然度不足 [§1]。已有 RL 方法各有缺陷: PPO/REINFORCE 需要训练/维护多个模型 (value network, reward model),训练复杂不稳定 [§1]; DPO 依赖高质量偏好数据,对标注噪声敏感且对 reward 的控制有限 [§1]; DiffRO 需要预训练 token-to-text 模型,增加计算和数据成本 [§1]。

核心问题: 能否用一种**不需要额外训练模型**、仅依赖现成 ASR 模型的 RL 方法来微调 LLM-based TTS,同时适用于两类主流架构 (acoustic token 和 semantic token 路线)?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

标准 GRPO 框架应用于 TTS [§2.2, Fig 2]: 预训练 TTS LLM 作为策略模型 pi_theta,固化副本作为参考模型 pi_ref。给定输入文本 y,策略模型生成 G=8 组候选输出,通过 codec decoder (Llasa) 或 flow matching + vocoder (CosyVoice2) 还原为语音波形,再送入 Whisper ASR 计算 reward,最后用 group-relative advantage normalization 更新策略 [论文原文]。

### 关键设计选择

#### 1. 为什么选 CER + NLL 双 reward 而非仅用 CER?

[论文原文, §2.1] CER 作为 reward 有三个局限: (1) 只衡量表面转写准确率,忽略 ASR 模型的预测置信度 — 不同质量的语音可能得到相同 CER 但差别很大; (2) 对韵律、流畅性等声学细微差异不敏感; (3) CER 是离散值,导致 reward signal 稀疏或不稳定。

NLL 通过 ASR decoder 对 ground-truth token 的条件概率分布提供连续、细粒度的信号 [论文原文, §2.1]。具体实现 [§2.1, Fig 1]: 将 ground-truth text 送入 Whisper decoder,以 ASR encoder 输出为条件,计算每个 token 的对数概率之和。NLL 越低 → ASR 对该语音中包含目标文本的置信度越高。

[agent 解读] NLL 和 CER 的互补性体现在: CER 是 "硬" 指标 (对/错),NLL 是 "软" 指标 (置信度)。当多个 rollout 的 CER 都为 0 (转写完全正确) 时,NLL 仍能区分它们的质量差异 — 让 ASR 更有信心的语音通常在韵律和清晰度上更好。论文的散点图 [Fig 3] 印证了这一点: 大多数样本 R_CER=1 (完全正确),但 R_NLL 分布广泛 (r=0.3371)。

#### 2. 为什么用调和平均而非算术平均组合 reward?

[论文原文, §2.1] 调和平均对低值更敏感,有效惩罚极端情况。例如,如果某个指标表现很差 (如高 CER),整体 reward 会被大幅拉低,防止单一指标的高分掩盖另一指标的缺陷。

具体公式 [§2.1, Eq.2-4]:
```
R_CER = 1 - tanh(α_c · CER)         # α_c = 3
R_NLL = exp(-NLL / α_n)              # α_n = 3
R = (λ_c + λ_n) / (λ_c/R_CER + λ_n/R_NLL)   # λ_c=0.6, λ_n=0.4
```

[agent 解读] CER 权重 λ_c=0.6 > NLL 权重 λ_n=0.4,说明内容正确性仍是主要优化目标,NLL 起辅助/精细化作用。tanh 和 exp 的映射保证两个 reward 都在 [0,1] 范围内。

#### 3. 为什么 GRPO 而非 PPO?

[论文原文, §1-2] GRPO 移除了 PPO 中的 value model,降低资源消耗和训练复杂度。GRPO 用 group-wise normalization 估计 advantage [§2.2, Eq.6]: A_i = (R_i - mean(R)) / std(R),不需要学习 value function。

#### 4. 为什么同时验证两类 TTS 模型?

[论文原文, §1] 论文声称 GRPO 方法"applicable to both categories of LLM-based TTS models"。这是本文相对于 Multi-Reward GRPO (仅 LLaSA) 和 DiffRO (仅 CosyVoice 系列) 的独特贡献。[agent 解读] 这也暴露了一个重要发现: RL 对两类系统的效果不对称 — semantic token 路线的自然度改善显著,而 acoustic token 路线的自然度无显著改善 [§3.2.3]。

### 训练策略

- 训练数据: 从 Emilia 采样 4000 句,覆盖中英日韩四种语言,中英占约 90% [§3.1.1]
- ASR model: Whisper-large-v3 (计算 reward + 英文评估) [§3.1.2]
- α_c = α_n = 3; λ_c = 0.6, λ_n = 0.4 [§3.1.2]
- GRPO: G=8, β=0.1 (KL penalty), lr=1e-5 [§3.1.2]
- 基线模型: CosyVoice2 (开源,含训练代码) + Llasa-1B (开源,含训练代码) [§3.1.1]
- 评估: 中/英用 SEED-TTS-Eval (2020+1088 样本),日/韩用 Common Voice 1000 样本 [§3.2.1]
- 评估 ASR: 中文用 Paraformer-zh,其他用 Whisper-large-v3 [§3.2.2]
- SIM: WavLM fine-tuned for speaker verification [§3.2.2]
- MOS: CosyVoice2 各语言 30 样本 × 10 native speakers [§3.2.3]

## 实验

### 客观评估 (Table 1)

| 指标 | CosyVoice2 | +GRPO-CER | +GRPO-NLL | +GRPO-CER-NLL | Llasa-1B | +GRPO-CER-NLL (Llasa) | Human | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CER (zh) ↓ | 1.41 | 1.34 | 0.98 | **1.07** | 7.73 | **1.30** | 1.33 | SEED test-zh | [Table 1] |
| SIM (zh) ↑ | 0.753 | 0.751 | 0.753 | 0.753 | 0.636 | 0.669 | 0.755 | SEED test-zh | [Table 1] |
| WER (en) ↓ | 2.46 | 2.43 | 2.36 | **2.30** | 4.95 | **2.17** | 2.10 | SEED test-en | [Table 1] |
| SIM (en) ↑ | 0.655 | 0.655 | 0.659 | 0.659 | 0.578 | 0.580 | 0.734 | SEED test-en | [Table 1] |
| CER (ja) ↓ | 12.45 | 10.05 | 9.36 | **9.09** | - | - | 8.53 | Common Voice | [Table 1] |
| CER (ko) ↓ | 8.58 | 6.37 | 6.59 | **6.16** | - | - | 7.43 | Common Voice | [Table 1] |

### 主观评估 (Table 2, CosyVoice2 only)

| 指标 | CosyVoice2 | +GRPO-CER | +GRPO-NLL | +GRPO-CER-NLL | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS (zh) ↑ | 4.42±0.05 | 4.44±0.06 | 4.52±0.05 | **4.58±0.05** | [Table 2] |
| MOS (en) ↑ | 4.22±0.06 | 4.26±0.07 | 4.31±0.06 | **4.43±0.06** | [Table 2] |
| MOS (ja) ↑ | 4.10±0.08 | 4.15±0.08 | 4.21±0.08 | **4.29±0.08** | [Table 2] |
| MOS (ko) ↑ | 4.18±0.08 | 4.23±0.08 | 4.24±0.08 | **4.30±0.08** | [Table 2] |

### 关键发现

**CER+NLL 互补性** [§3.2.4, Fig 3]: CER 和 NLL 的 Pearson 相关系数仅 r=0.3371,说明两者提供的信息高度互补。在 R_CER=1 (完全正确) 的区域,R_NLL 仍有广泛分布,提供额外区分度 [论文原文]。

**GRPO-NLL > GRPO-CER** [§3.2.2]: 在大多数子集上,仅用 NLL reward 的效果优于仅用 CER reward (如 CosyVoice2 zh CER: 0.98 vs 1.34; MOS zh: 4.52 vs 4.44)。[agent 解读] 这暗示 NLL 作为连续信号的梯度信号质量优于离散 CER;NLL 的优化本身就隐含了内容正确性,因为正确内容会获得更高 ASR 置信度。

**Llasa-1B CER 大幅下降但 SIM 改善有限** [Table 1]: Llasa-1B 的 CER 从 7.73 降至 1.30 (↓83%),但 SIM 仅从 0.636 到 0.669 (↑5%)。[论文原文, §3.2.2] 论文解释: 基线 CER 极高时,严重的发音错误可能导致 WavLM speaker verification 模型产生不可靠的相似度分数,因此 SIM 改善不能完全反映真实进步。

**Llasa-1B 自然度无显著改善** [§3.2.3]: 论文坦承 Llasa-1B + GRPO 在 MOS 上无统计显著改善,"the underlying reasons will be investigated in future work"。[agent 解读] 可能原因: (1) acoustic token 的量化损失是自然度的主要瓶颈,RL 优化了 token 选择但无法改善 codec decoder 的重建质量; (2) CER+NLL reward 仅优化内容一致性,不直接优化韵律/自然度; (3) 与 CosyVoice2 不同,Llasa 没有 flow matching 做后续声学精炼。

**四语言泛化** [Table 1]: CosyVoice2+GRPO-CER-NLL 在训练数据占比较低的日语和韩语上也有明显改善 (ja: 12.45→9.09, ko: 8.58→6.16),说明 GRPO 对低资源语言也有效 [论文原文, §3.2.2]。

**频谱分析** [§3.2.4, Fig 4-5]: 两个具体案例表明 GRPO-CER-NLL 能纠正基线模型的发音错误 (漏字、误读) 并产生更自然的韵律断句 [论文原文]。

## 局限性

1. **篇幅过短 (4 页)**: 许多重要细节缺失 — GRPO 训练步数、训练时长、收敛行为、超参数搜索过程均未报告。Tongyi 的 RL-for-Audio-LLM 发现 GRPO 超 1500 步后退化,但本文未讨论训练稳定性 [agent 解读]
2. **未与 DiffRO/DPO 做实验对比**: 仅在 introduction 中文字对比 DiffRO/DPO 的缺点,缺少公平的消融实验。同期 RL-for-Audio-LLM 已有 GRPO vs DiffRO 的对比框架 [agent 解读]
3. **Reward 仅优化内容一致性**: CER 和 NLL 都源自 ASR,不涵盖说话人相似度、韵律自然度等维度。与 Multi-Reward GRPO (5 维 reward) 和 TTS-1 (CER+SIM+DNSMOS) 相比,reward 维度较窄 [agent 解读]
4. **Llasa-1B 自然度未改善**: 论文承认但未深入分析原因,且未尝试增加 SIM/MOS 相关 reward 来改善 [§3.2.3]
5. **评估规模有限**: MOS 评估仅 30 样本/语言 × 10 评估者,且仅针对 CosyVoice2 [§3.2.3]
6. **未报告推理速度影响**: GRPO 微调后模型推理速度是否变化未知 [agent 解读]

## 点评

本文是首篇将 GRPO 显式应用于 LLM-based TTS 并在两类主流架构上验证的工作。其核心贡献在于两个方面: (1) NLL 作为 CER 互补 reward 的设计,通过 ASR decoder 的 teacher-forcing NLL 提供连续、细粒度的优化信号,巧妙利用了现成 ASR 模型的内部表示而不需要额外训练; (2) 在同一框架下对比 semantic token 路线 (CosyVoice2) 和 acoustic token 路线 (Llasa-1B),揭示了 RL 对不同 token 类型系统的差异化效果。

NLL reward 的设计思路值得借鉴: 它本质上是将 ASR 模型当作"differentiable judge" — 不需要像 DiffRO 那样训练专门的 token-to-text reward model,也不需要 Gumbel-Softmax 做可微采样。但这也带来了局限: NLL 只衡量 ASR 的置信度,不直接捕获韵律或说话人信息,因此 Llasa-1B 在自然度上无显著改善可能正源于此。

与同期工作的对比:
- vs **Multi-Reward GRPO**: 本文更简约 (2 维 vs 5 维 reward),但范围更窄 (仅内容一致性 vs 内容+韵律+稳定性); Multi-Reward GRPO 的 LLM-annotated prosody reward 是更丰富的信号来源
- vs **DiffRO**: 本文不需要额外训练模型,pipeline 更简单; 但 DiffRO 在 token 空间操作避免了完整音频渲染的计算成本
- vs **TTS-1** (Inworld): TTS-1 同样使用 GRPO + ASR reward,但额外加入了 SIM 和 DNSMOS,且在 8.8B 规模验证

论文最有启发的发现是: 即使仅用 4000 句训练数据和单一 ASR 模型,GRPO 就能大幅改善 TTS 可懂度,尤其对 acoustic token 路线 (Llasa CER 7.73→1.30) 的效果非常显著。这暗示 acoustic token TTS 的内容一致性问题很大程度上是"对齐问题"而非"能力问题",RL 能有效桥接这个 gap。

## 可复用的 idea

1. **ASR NLL 作为 CER 的互补 reward**: 用 ASR decoder teacher-forcing NLL 提供连续信号,在 CER=0 时仍有区分度。可迁移到任何使用 ASR 做 reward 的 TTS RL 系统。成本极低 (仅多一次 decoder forward pass)。
2. **调和平均组合 reward**: 相比算术平均,对单维度极差值更敏感,防止"高分掩盖低分"。可用于任何多 reward RL 系统。
3. **跨架构 RL 验证**: 用同一 RL 方法同时验证 semantic token 和 acoustic token 两类 TTS 系统,发现差异化效果。这种实验设计对理解 RL + TTS 的交互很有价值。
4. **极少数据高效 RL**: 仅 4000 句训练数据即可显著改善两类 TTS 系统的可懂度,说明 GRPO 的样本效率很高,适合作为低成本 post-training 手段。

> [!review] 审阅状态: pass-with-fixes (2026-06-04)
> 3 个 low 级问题: (1) frontmatter concepts 含外围概念 (Gumbel-Softmax/Speaker Verification/TTS Evaluation); (2) KB 背景 CosyVoice2 CER 引用 KB 值 (1.45%) vs 论文实测值 (1.41) 有微小差异,属正常评估方差; (3) 均不阻塞反向更新。详见 `_review/GRPO-TTS-review.yml`
