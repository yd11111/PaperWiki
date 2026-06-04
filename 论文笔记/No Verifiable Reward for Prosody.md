---
type: paper
tier: deep
title: "No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS"
arxiv_id: "2509.18531"
source: "Sources/2509.18531.pdf"
authors: [Seungyoun Shin, Dongha Ahn, Jiwoo Kim, Sungwook Jeon]
year: 2026
venue: "arXiv"
tags: [TTS, prosody, preference-optimization, DPO, GRPO, reinforcement-learning, reward-design, Korean-TTS, conversational-TTS]
concepts: ["[[Prosody Modeling]]", "[[Differentiable Reward Optimization]]", "[[TTS Evaluation]]", "[[F0 Modeling]]"]
models: ["[[Llasa]]", "[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文处于 TTS post-training / RL alignment 路线的最新进展中。知识库中 [[Differentiable Reward Optimization]] 页面已系统记录了这一路线的演进: RLHF for NLP (2022) → RL for TTS on audio (Seed-TTS, 2024) → Preference optimization for codec LM (SpeechAlign, 2024) → Token-level DiffRO (CosyVoice 3, 2025) → Multi-Reward GRPO (Tencent, 2025) → Industrial-scale GRPO (TTS-1, 2025)。本文的位置是: 在 GRPO 路线暴露韵律坍缩问题后,回归 DPO 路线,以极少量人类偏好数据修复韵律。

**已有认知 (confirmed)**: 
- [[Prosody Modeling]]: 韵律建模覆盖 duration/pitch/energy/pause 四维度。在 LLM-TTS 时代,韵律被 in-context learning 隐式建模,导致细粒度控制困难 — 这正是本文所诊断的问题根源。
- [[LLM-based TTS]]: Llasa 属于典型的 codec language model TTS (LLaMA-initialized Transformer + XCodec2),通过自回归生成离散 speech tokens。本文在 Llasa-1B 上做 continual training + GRPO/DPO post-training。
- [[Speaker Embedding]]: 本文将 speaker-similarity (cosine similarity of speaker embeddings) 作为 GRPO reward 的扩展项,发现它引入训练不稳定。

**已有认知 [待确认]**: 
- [[Differentiable Reward Optimization]]: 该页面已记录 GRPO vs DiffRO 对比、Multi-Reward GRPO、FPO 等多条路线。本文提供了 GRPO 韵律坍缩的直接实验证据,是对该页面"GRPO 优化 WER 但可能伤害其他维度"判断的有力支撑。
- [[TTS Evaluation]]: 该页面记录了 WER 的局限 — "直接优化 WER 作为 reward 会导致韵律坍缩 (Shin et al., 2026)",正是引用本文。本文使用 CER + ELO (Chatbot Arena-style human preference) 双轨评估。
- [[F0 Modeling]]: 本文用 logF0 分布图直接展示 GRPO 导致的 pitch 变异性降低。

**创新判断**: 相比已有工作 (SpeechAlign 用合成偏好对、FPO 用 token-level 选择性 DPO、DiffRO 用可微 reward model),本文的独特之处在于: (1) 明确诊断 GRPO 的韵律坍缩机制 (reward 设计缺陷而非优化器问题); (2) 用极少量真人偏好标注 (~200 pairs/round) 做 iterative DPO; (3) 在韩语客服对话这一高韵律要求场景验证。

> 检索命中: [[Prosody Modeling]]✓, [[LLM-based TTS]]✓, [[Speaker Embedding]]✓ | 参考(待确认): [[Differentiable Reward Optimization]], [[TTS Evaluation]], [[F0 Modeling]] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: GRPO 用 CER/NLL reward 优化 TTS 会将韵律坍缩为近单调语音;iterative DPO 仅需 ~200 人类偏好对/轮即可恢复对话韵律且保持 CER 竞争力
> - **路线**: Llasa-1B → Korean continual training (36k h) → single-speaker fine-tune (18 h) → channel-base → {GRPO / iterative DPO (3 rounds)} → evaluation on KoCC-TTS
> - **指标**: DPO-R2 ELO 1190.1 (最高) + CER 3.60% | GRPO CER 2.20% (最低) + ELO 753.7 (最低) | channel-base CER 2.90% + ELO 1150.1 [Table 1]
> - **可借鉴**: 当目标维度缺乏可验证的自动 reward 时,用极少量人类偏好对做 iterative DPO 是 cost-effective 的修复路径;moving reference (πref = πθr-1) + 不跨轮复用数据的设计防止过优化
> - **局限**: 仅验证单说话人韩语客服场景;ELO 评估基于 596 votes / 27 人,样本量有限;未与 DiffRO / FPO 等 token-level RL 方法对比;DPO Round 3 已出现收益递减

## 核心问题

本文解决的核心问题是: **TTS post-training 中的 reward 设计缺陷导致韵律坍缩**。

具体而言:
1. GRPO 使用 CER + NLL 作为 reward 时,优化器忠实地最大化可度量指标,但韵律 (pitch variation, phrasing) 不在 reward 范围内,因此被"优化掉" [§1, §4.3]
2. 试图通过加入 speaker-similarity reward 补偿韵律维度时,非韵律信号 (speaker identity) 引入优化不稳定,CER 反而恶化,且出现 EOS 失败 (生成无法终止) [§4.4]
3. 根本原因在于: 韵律自然度目前没有可靠的自动 reward function — 这是一个 "reward gap" [§1]

作者的核心论点: **bottleneck 在 reward formulation,不在 optimizer choice** [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

基础模型: Llasa-1B (LLaMA-initialized Transformer + XCodec2 decoder) [§3.2]

训练路线:
```
Llasa-1B checkpoint
  → Continual training on 36k h Korean (AIHUB)
  → Fine-tune on 18 h proprietary single-speaker (manager channel)
  = channel-base
  → 分支 A: GRPO (CER + NLL reward)
  → 分支 B: GRPO-sim (CER + NLL + speaker-similarity reward)
  → 分支 C: Iterative DPO (3 rounds, 200 human pairs/round)
```

### 关键设计选择

**1. GRPO Reward 设计 (base reward)** [§3.3.1]

将 CER 和 NLL 映射为 (0,1] 的 utility:
- CER utility: $U_c = 1 - \tanh(\tau_c \cdot c)$
- NLL utility: $U_\ell = \exp(-\ell / \tau_\ell)$

最终 reward 用 **调和平均**: $R = \frac{\lambda_c + \lambda_\ell}{\lambda_c / U_c + \lambda_\ell / U_\ell}$

为什么用调和平均而非算术平均: 调和平均对小分量惩罚更强,创造对高 error 的强约束 [论文原文, §3.3.1]。权重 $(\lambda_c, \lambda_\ell) = (0.6, 0.4)$。

**2. Speaker-similarity 扩展 (失败尝试)** [§3.3.2]

将 cosine similarity $s \in [-1, 1]$ 映射为 $U_s = \min(\max((s+1)/2, 0), 1)$,加入调和平均 reward。权重 $(0.5, 0.3, 0.2)$。

为什么失败: speaker-similarity 不是韵律信号,它鼓励模型模仿参考说话人的整体声学特征,而非改善韵律自然度 [论文原文, §4.4]。实验显示 CER 严重退化 (42.63%),模型生成过长输出且无法产生 EOS token — 作者认为 RL objective 被 "hacked" [论文原文, §4.4]。[agent 解读] 这类似于 reward hacking: 模型找到提高 speaker-similarity 但破坏内容和终止行为的捷径。

**3. Iterative DPO (核心方案)** [§3.4]

每轮 $r \in \{1, 2, 3\}$:
1. 从上一轮 checkpoint $\pi_{\theta_{r-1}}$ 初始化
2. $\pi_{\theta_{r-1}}$ 同时作为 moving reference $\pi_{ref}$
3. 用 $\pi_{\theta_{r-1}}$ 生成候选语音
4. 收集 200 人类偏好对 $\{(x, y^+, y^-)\}$ (A/B 听比)
5. 优化 DPO loss: $\mathcal{L}_{DPO}(\theta) = -\mathbb{E}[\log \sigma(\beta[\Delta\ell_\theta - \Delta\ell_{ref}])]$ [§3.4, Eq. 7]
6. 偏好数据不跨轮复用

为什么用 moving reference 而非固定 reference: [agent 解读] 固定 reference 在多轮迭代中 policy-reference gap 会持续增大,导致 KL 正则化效果减弱;moving reference 每轮重新锚定,保持有效正则化。这与 iterative preference optimization 中 diminishing returns 的观察一致 [§4.5]。

为什么不跨轮复用数据: [agent 解读] 每轮的候选由更新后的 policy 生成,旧轮偏好对在新 policy 下 log-likelihood 比值含义不同,复用可能引入噪声。

### 训练策略

- **数据**: 36k h 公开韩语数据 (AIHUB) + 18 h 专有单说话人 (女声客服经理) [§3.1]
- **语音分割**: pyannote.audio v3.0 VAD → Whisper-large-v3 转写 [§3.1]
- **DPO 偏好收集**: 每轮 200 对,blind A/B comparison,目标维度为韵律自然度 [§3.4]
- **GRPO 训练集**: 1.6M text prompts [§4.3]

## 实验

| 指标 | 本文最优 (DPO-R2) | Baseline (channel-base) | GRPO (clean) | GRPO-sim | DPO-R1 | DPO-R3 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CER (%) ↓ | 3.60 | 2.90 | **2.20** | 42.63 | 5.80 | 3.30 | [Table 1] |
| ELO ↑ | **1190.1** | 1150.1 | 753.7 | 878.7 | 1096.5 | 1064.2 | [Table 1] |

**外部系统对比** (ELO / CER) [Table 1]:
- ElevenLabs Multilingual v2: 955.1 / 4.74%
- Supertone: 1046.9 / 2.98%
- GPT-4o-mini-tts (sage): 848.9 / 2.91%

**关键观察**:

1. **GRPO 的韵律坍缩**: GRPO 达到最低 CER (2.20%) 但最低 ELO (753.7),logF0 分布显示 pitch 变异性显著降低 [Fig 2] — 验证了 reward 设计缺陷导致的韵律坍缩假说 [§4.3]

2. **Speaker-similarity 的不稳定性**: GRPO-sim CER 暴涨至 42.63%,出现 EOS 失败 [§4.4]

3. **DPO 的 round-wise 行为** [§4.5]:
   - R1: 探索阶段,ELO 下降至 1096.5 (低于 baseline),CER 升至 5.80%
   - R2: 最优点,ELO 1190.1 (超越所有外部系统),CER 回落至 3.60%
   - R3: 收益递减,ELO 降至 1064.2,CER 继续改善至 3.30%

4. **为什么 R2 最优**: 作者假设早期轮次 chosen/rejected 样本的 reward gap 更大,提供更 informative 的梯度;随着迭代 policy-reference gap 缩小,新偏好对的信息量递减 [论文原文, §4.5]

**评估方法**: 596 votes from 27 participants (age 20-60), blind A/B pairwise comparison, Chatbot Arena-style ELO aggregation [§4.2]

## 局限性

1. **场景局限**: 仅验证韩语单说话人客服对话,未测试多说话人、多语言、非对话场景 [agent 解读]
2. **评估规模有限**: 596 votes / 27 raters 的 ELO 估计统计置信度值得关注 [agent 解读]
3. **缺少 RL baseline 对比**: 未与 DiffRO、FPO、SpeechAlign 等 token-level / preference-based 方法对比 [agent 解读]
4. **DPO 收益递减**: R3 已出现 ELO 下降,long-term iteration 策略不明 [§4.5]
5. **人工标注成本**: 每轮 200 对虽少,但需要持续人工参与,不适合快速迭代场景 [agent 解读]
6. **KoCC-TTS 仅 50 条**: 评估集规模非常小,覆盖的韵律多样性可能不足 [§4.1]
7. **未分析 DPO 与 GRPO 结合**: 是否可先 GRPO 再 DPO (取 GRPO 的 CER 优势 + DPO 的韵律优势) 未探索 [agent 解读]

## 点评

**贡献定位**: 本文的核心贡献不在方法新颖性 (iterative DPO 已有前例如 SpeechAlign),而在于**清晰诊断了 GRPO 在 TTS 中的韵律坍缩问题,并验证了极低数据量偏好优化的可行性**。"No verifiable reward for prosody" 这一观察切中了 TTS RL alignment 的核心痛点。

**与 KB 中已有工作的关系**:
- 与 [[论文笔记/Multi-Reward GRPO|Multi-Reward GRPO]] (Tencent) 形成互补: Multi-Reward GRPO 尝试用 LLM-annotated prosody alignment 作为 reward 之一来解决韵律问题,本文则直接绕开自动 reward,用人类偏好替代 — 两条路线各有 trade-off
- 与 [[论文笔记/FPO|FPO]] 的比较: FPO 也用偏好优化但在 token-level 操作 (200 utterances 即有效);本文在 utterance-level 操作但用真人标注而非 ASR-guided 标注
- 与 [[论文笔记/DiffRO|DiffRO]] 的比较: DiffRO 走 token-level 可微 reward model 路线,可自动化但受限于 reward model 质量;本文走 human-in-the-loop 路线,更直接但不可全自动

**方法论启示**: 本文隐含的一个重要观点是 **reward 可度量性与优化目标完整性之间的张力** — 只优化能度量的 (CER/NLL) 会伤害不能度量的 (韵律)。这与 Goodhart's Law ("When a measure becomes a target, it ceases to be a good measure") 在 TTS RL 中的体现一致。

## 可复用的 idea

1. **"当 reward 不可自动验证时,少量人类偏好就够了"模式**: 200 pairs/round 的 data efficiency 提示:在任何生成任务中,如果某个质量维度缺乏可靠自动指标,少量人类偏好 + iterative DPO 可能是最 cost-effective 的方案
2. **Moving reference DPO**: $\pi_{ref} = \pi_{\theta_{r-1}}$ + 不跨轮复用数据,是 iterative DPO 防过优化的简洁设计
3. **调和平均 reward**: 对多目标 GRPO,调和平均比算术平均更能防止某一维度 collapse
4. **F0 分布可视化作为韵律诊断工具**: logF0 distribution 比较是检测韵律坍缩的简单有效手段 [Fig 2]
5. **反面教训 — speaker-similarity 作为 reward 的不稳定性**: 提醒设计 multi-reward 时需注意各 reward 维度之间的兼容性

> [!review] 审阅状态
> 待审阅。审阅报告见 `_review/No Verifiable Reward for Prosody-review.yml`。
