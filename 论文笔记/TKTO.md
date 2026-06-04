---
type: paper
tier: deep
title: "TKTO: Data-efficient Targeted Token-level Preference Optimization for LLM-based TTS"
arxiv_id: "2510.05799"
source: "Sources/Data-efficient-Targeted-Token-level-PO.pdf"
authors: [Rikuto Kotoge, Yuichi Sasaki]
year: 2025
venue: "arXiv (SpiralAI)"
tags: [preference-optimization, token-level-optimization, KTO, LLM-based-TTS, data-efficiency, pronunciation, Japanese-TTS, polyphonic-disambiguation]
concepts: ["[[LLM-based TTS]]", "[[Differentiable Reward Optimization]]", "[[Speech Tokenizer]]"]
models: ["[[CosyVoice 2]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[LLM-based TTS]], [[CosyVoice 2]], [[Speech Tokenizer]], [[Conditional Flow Matching]]; 2 个待确认: [[Differentiable Reward Optimization]], [[TTS Evaluation]])
> **谱系定位**: TTS 偏好优化已形成多条路线: (1) 音频级 RL (Seed-TTS REINFORCE, 2024); (2) utterance-level DPO/KTO (SpeechAlign, 2024); (3) token-level 选择性 DPO (FPO, 2025); (4) token-level 可微优化 (DiffRO/CosyVoice 3, 2025); (5) GRPO 多奖励 (Multi-Reward GRPO / TTS-1, 2025)。TKTO 处于路线 (2) 和 (3) 之间——将 KTO 从 utterance-level 扩展到 token-level,但与 FPO 不同的是不需要显式标注 error segments,而是通过 contrastive LLMs 自动估计 token-level importance weights。
>
> **已有认知**: SpeechAlign 首次将 DPO 引入 codec LM; FPO 将 DPO loss 下沉到 error token segments (3-4x 数据效率); DiffRO 通过 Gumbel-Softmax + token-level reward model 实现端到端可微优化。KTO (Kahneman-Tversky Optimization) 作为 DPO 的无配对数据替代方案在 NLP 已有应用,但此前未被扩展到 token-level 用于 TTS。CosyVoice 2 是阿里通义实验室的 LLM-based TTS 模型,采用 FSQ-SenseVoice tokenizer + chunk-aware flow matching。
>
> **创新判断**: TKTO 的核心贡献不是 KTO 用于 TTS 本身 (Tian et al. 2025 已做过 DPO for TTS),而是 (a) 通过 contrastive LLM pair (label-flipped KTO) 自动估计 token-level importance,无需 error segment 标注; (b) 将 KTO 从 sequence-level 推广到 token-level weighted optimization; (c) 消除对配对数据的依赖,实现 6x 数据利用率。
> 检索命中: [[LLM-based TTS]], [[CosyVoice 2]], [[Speech Tokenizer]], [[Conditional Flow Matching]] | 过滤: [[Differentiable Reward Optimization]](pending-review), [[TTS Evaluation]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过构建 contrastive LLMs 自动估计 token-level importance weights,将 KTO 从 sequence-level 推广到 token-level,消除配对数据需求,在日语发音歧义任务上 accuracy 提升 39%、CER 降低 54%
> - **路线**: Unpaired data → KTO 训练 π+ 和 label-flipped π- → log-ratio 计算 token weights → Token-level KTO 优化 πθ → CosyVoice 2 生成语音
> - **指标**: 日语 Acc 0.668→0.958 (+43%), CER 0.138→0.066 (-52%), Bad 0.095→0.027 (-72%) (Male, unpaired) [Table 1]; 中文 Acc 0.645→0.898, CER 0.024→0.014 [Table 3]; NMOS 4.09→4.21 [Table 2]; ABX win rate 61.3% vs Base [Fig 6]
> - **可借鉴**: (1) Label-flipped KTO 构建 contrastive LLM pair 用于 token-level importance 估计——可推广到任何序列级 PO 需要 token-level 精细化的场景; (2) 无配对数据偏好优化 (KTO) 在 TTS 上的有效性,89.5% 的数据本来无法被 DPO 利用
> - **局限**: 仅在 0.5B 模型验证,无 scaling 实验; 仅评估日语/中文歧义发音场景,未覆盖英语或通用 TTS 质量; 需额外训练两个 contrastive LLMs (8xA100 ~10min,成本低但仍是额外步骤); 无开源代码/模型

## 核心问题

1. **配对数据稀缺**: DPO 需要成对的 desirable/undesirable 样本,但 TTS 系统往往对同一文本产生一致性的正确或错误输出——89.5% 的数据只有单侧结果,仅 10.5% 可构成配对,导致 DPO 可用数据仅 1.5K (vs TKTO 的 9K) [§4.4]
2. **序列级与 token 级的粒度错配**: 发音生成本质是 character/token 级任务,但传统偏好优化在 utterance 级别进行标签标注和损失计算。模型被迫在整个序列上优化,而非直接在发音单元级别学习 [§1]
3. **如何不依赖人工标注获取 token-level 信号**: FPO 通过 ASR forced alignment 定位 error segments,但仍需显式标注;TKTO 能否自动发现哪些 token 对偏好学习最重要? [§3.1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TKTO 是一个两步 post-training 框架 [Fig 2]:

```
Step 1: Token Importance Estimation
  Unpaired data → KTO 训练 π+ (original labels)
                → KTO 训练 π- (label-flipped: desirable↔undesirable)
  Token weight: wt = exp(µ · clamp(log(π+(yt|x,y<t) / π-(yt|x,y<t)), L, U))

Step 2: Token-level KTO Optimization
  定义 token-level reward: rθ,t = log(πθ(yt|x,y<t) / πref(yt|x,y<t))
  Token-level value: vt = λ·σ(β(rθ,t - z0,t))  (desirable)
                         λ·σ(β(z0,t - rθ,t))  (undesirable)
  Loss: L_TKTO = -E[Σ wt · vt]
```

基座模型为 CosyVoice 2 (0.5B),在 20K 小时日语语音数据上微调 [§4.1]。

### 关键设计选择

**1. 为什么用 KTO 而不是 DPO?**
KTO 是 prospect theory 驱动的 pair-free 对齐目标,使用 binary feedback (desirable/undesirable) 而非 pairwise comparison。这使得 89.5% 的 "one-sided" 数据 (只有正样本或只有负样本) 可以被利用,数据量从 1.5K (DPO paired) 提升到 9K (KTO unpaired) [§4.4] [论文原文]。

**2. Contrastive LLM construction 的 WHY**
[论文原文] 要估计每个 token 对偏好学习的重要性,需要知道模型在 desirable 和 undesirable 两个方向上对各 token 的"偏好程度差异"。通过训练 π+ (按正常标签) 和 π- (标签翻转),两者的 log-ratio 自然反映了 token 在偏好维度上的区分度:
- 如果某个 token 在 π+ 和 π- 中预测概率相近 → 该 token 对偏好无贡献 → weight ≈ 1
- 如果某个 token 在 π+ 高概率但 π- 低概率 → 该 token 是偏好决定性因素 → weight >> 1

[agent 解读] 这一设计巧妙地将 token-level importance 的估计转化为两个模型的 divergence 度量,类似于 NLP 中的 contrastive decoding 思想,但用于 reward estimation 而非生成。

**3. Clamping bounds 的作用**
Token weight 使用 clamp(·, L, U) 限制 log-ratio 范围,防止极端 weight 导致训练不稳定 [§3.1, Eq.2]。敏感性分析 [Table 4] 表明 (-2, 2) 是最优选择,过宽范围 (-3, 3) 会导致 accuracy 轻微退化 [Appendix C] [论文原文]。

**4. Token-level KTO 的扩展**
TKTO 将 KTO 的 value function 从 sequence-level 分解到每个 token 位置:
- 每个 token 有独立的 reward rθ,t 和 reference baseline z0,t [Eq.3-4]
- Value function vt 使用 logistic-shaped 函数建模 loss aversion (prospect theory) [Eq.5]
- 最终 loss 是 importance-weighted token values 的求和 [Eq.6]

[论文原文] z0,t 是 per-token 的 KL divergence baseline,在 microbatch 中计算且不回传梯度,用于调整每个位置的 reward 基准 [§3.2]。

### 训练策略

- Base model: CosyVoice 2 (0.5B), fine-tuned on 20K hours Japanese data [§4.1]
- Preference data: 5000 sentences with ambiguous "辛い" → 5 samples each × 2 speakers = desirable/undesirable selection by pronunciation correctness + CER [§4.1]
- KTO hyperparams: λD = λU = 1, β = 0.10, µ = ±1, L = -2, U = 2 [Appendix B]
- Training: 1 epoch, lr = 1e-6, 8×A100 GPUs [Appendix B]
- Contrastive LLMs training: ~10 min on 8×A100 (negligible overhead) [§4.4]
- ASR for evaluation: Whisper-v3-large (primary), Parakeet-tdt_ctc-0.6b-ja (robustness check) [§4.1, Appendix D]

## 实验

| 指标 | TKTO (unpaired) | KTO (unpaired) | DPO (paired) | Base model | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Acc ↑ (Male) | **0.958** | 0.952 | 0.693 | 0.668 | 日语 5K 歧义句 | [Table 1] |
| CER ↓ (Male) | **0.066** | 0.074 | 0.130 | 0.138 | 同上 | [Table 1] |
| Bad ↓ (Male) | **0.027** | 0.032 | 0.082 | 0.095 | 同上 | [Table 1] |
| Acc ↑ (Female) | **0.949** | 0.933 | 0.706 | 0.683 | 同上 | [Table 1] |
| CER ↓ (Female) | **0.059** | 0.066 | 0.120 | 0.128 | 同上 | [Table 1] |
| NMOS ↑ | **4.21** | 4.17 | - | 4.09 | 同上 | [Table 2] |
| ABX win% vs Base | **61.3%** | - | - | 35.7% | 同上 | [Fig 6] |
| Acc ↑ (中文) | **0.898** | 0.832 | 0.682 | 0.645 | 中文 5K 多音字 "行" | [Table 3] |
| CER ↓ (中文) | **0.014** | 0.018 | 0.021 | 0.024 | 同上 | [Table 3] |

**与工业模型对比** [Table 1]:
- TKTO 超越 gpt-4o-mini-tts (Acc 0.958 vs 0.939, Male) 和 gemini-2.5-pro-preview-tts (Acc 0.958 vs 0.885)
- F5-TTS 无论用不用 G2P,accuracy 均约 0.5 (随机水平),说明非 LLM 方法无法处理上下文依赖发音 [Table 1]

**Token weight 分析** [Fig 4-5]:
- 目标歧义字 "辛" 的 desirable tokens 获得 reward 0.22 (vs overall mean 0.12)
- Undesirable tokens 获得 reward -1.54,导致 weight 被大幅放大 (12.8× stronger)
- Case study [Fig 5] 可视化了 token weight 分布,目标字符的 tokens 获得显著更高权重

**训练动态** [Fig 3]:
- TKTO 有效地仅增加 desirable tokens 的 log-likelihood,undesirable tokens 保持稳定
- 对比 SFT: SFT 会同时增加 desirable 和 undesirable tokens 的 log-likelihood [论文原文]

## 局限性

1. **评估范围窄**: 仅在日语/中文歧义发音场景验证,未测试通用 TTS 质量 (如英语、长文本、情感表达);评估数据集由 GPT-5 生成,可能存在 domain bias [§Limitations]
2. **模型规模单一**: 仅测试 0.5B CosyVoice 2,未探索 scaling behavior (更大模型不可用) [§Limitations]
3. **Speaker similarity 未报告**: 缺少 SIM 指标,无法确认 TKTO 是否保持说话人一致性 (FPO 论文中发现 token-level PO 对 SECS 无显著影响,TKTO 可能类似)
4. **Contrastive LLMs 额外开销**: 虽然作者称 ~10min 训练可忽略 [§4.4],但需要存储两个额外模型用于 weight 估计
5. **Off-policy 限制**: TKTO 是 off-policy 方法,on-policy 扩展是未来方向 [§Conclusion]

## 点评

**核心贡献的价值**: TKTO 提出了一种优雅的 token-level importance 自动估计方案。通过 label-flipped KTO 构建 contrastive LLM pair,避免了 FPO 中需要 ASR forced alignment 标注 error segments 的步骤。这种"让两个模型的分歧告诉你哪些 token 重要"的思路在方法论上具有泛化性。

**与 FPO 的互补关系**: FPO 通过 ASR+forced alignment 显式标注 error segments,TKTO 通过 contrastive LLMs 隐式估计 token importance。两者解决同一问题 (token-level PO for TTS) 但路线不同:
- FPO 依赖 error type 分类 (temporal vs semantic-phonetic),标注更精确但成本更高
- TKTO 完全自动化,但 importance estimation 的精度取决于 contrastive LLMs 的质量
- TKTO 额外消除了配对数据需求 (KTO vs DPO),而 FPO 仍需 paired data

**实验设计的局限**: 评估场景高度特化 (仅歧义发音),这使得 accuracy 指标极其有利——歧义字的 binary correct/incorrect 判断非常适合 token-level 优化。但这不能直接推广到通用 TTS 质量改善 (如韵律自然度、长句稳定性)。KTO unpaired 的数据效率优势在通用场景下是否同样显著,有待验证。

**在演进线上的位置**: TKTO 贡献了 TTS 偏好优化的第五条路线——基于 prospect theory 的 token-level 无配对优化。它与 DiffRO (可微 reward 路线) 和 FPO (选择性 DPO 路线) 构成三种 token-level PO 方案,各有取舍。

## 可复用的 idea

1. **Contrastive LLM pair for token importance**: 用 label-flipped 训练构建两个模型,其 log-ratio 作为 token-level importance weight。可推广到任何需要 token-level credit assignment 的序列优化任务 (如 ASR error correction, machine translation)
2. **KTO for TTS (pair-free PO)**: 89.5% 的 TTS 数据只有单侧偏好标签;KTO 的 binary feedback 机制天然适配这种数据分布,值得在更多 TTS 场景验证
3. **Clamped exponential weighting**: wt = exp(µ · clamp(log-ratio, L, U)) 的 weight 设计,既保持梯度信号强度又防止极端权重,是实用的 token weighting trick

> [!review] 审阅 (auto, 2026-06-04, pass-with-fixes)
> **结论**: pass-with-fixes (2 low issues, 不阻塞反向更新)
> - (low) 速查一句话的 39%/54% 对应 female speaker,详细指标行报告 male speaker 43%/52%,两组正确但对应不同 speaker
> - (low) frontmatter tasks 为空,polyphonic disambiguation 任务无对应 KB 页 (合理)
> 详见 `_review/TKTO-review.yml`

---

检索命中: [[LLM-based TTS]], [[CosyVoice 2]], [[Speech Tokenizer]], [[Conditional Flow Matching]] | 过滤: [[Differentiable Reward Optimization]](pending-review), [[TTS Evaluation]](pending-review) | 未命中但可能相关: 无
