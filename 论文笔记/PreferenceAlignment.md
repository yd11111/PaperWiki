---
type: paper
tier: deep
title: "Preference Alignment Improves Language Model-Based TTS"
arxiv_id: "2409.12403"
source: "Sources/PreferenceAlignment.pdf"
authors: [Jinchuan Tian, Chunlei Zhang, Jiatong Shi, Hao Zhang, Jianwei Yu, Shinji Watanabe, Dong Yu]
year: 2024
venue: "arXiv"
tags: [TTS, preference-alignment, DPO, LM-based-TTS, RLHF, post-training]
concepts: ["[[DifferentiableRewardOptimization]]", "[[LLM-basedTTS]]", "[[TTSEvaluation]]", "[[SpeakerEmbedding]]", "[[CodecLanguageModel]]"]
models: ["[[SoundStream]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

> [!review] 审阅状态
> 审阅人: agent-auto | 日期: 2026-06-08 | 结论: pass | Issues: 0 high, 0 medium, 3 low
> 报告: [[_review/PreferenceAlignment-review.yml]]

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓, [[SoundStream]]✓, [[DifferentiableRewardOptimization]][待确认], [[TTSEvaluation]][待确认], [[CodecLanguageModel]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]], [[SpeakerEmbedding]], [[SoundStream]], [[DifferentiableRewardOptimization]], [[TTSEvaluation]], [[CodecLanguageModel]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文属于 LLM-based TTS 的 **post-training / preference alignment** 方向。知识库中 [[DifferentiableRewardOptimization]] 页详细记录了 TTS 领域 RL/偏好优化的完整演进线: RLHF for NLP → Seed-TTS REINFORCE → SpeechAlign DPO → DiffRO → GRPO → FPO/TKTO 等。本文 (arXiv 2409.12403, 2024年9月) 在时间线上早于 DiffRO/GRPO,与 SpeechAlign (2024年4月) 同期,是 LM-based TTS 上 DPO 的较早系统性实证研究。

**已有认知**: 知识库已记录 SpeechAlign 首次将偏好学习引入 codec LM (golden vs synthetic AR tokens 构建偏好对),以及 Seed-TTS 首次在工业 TTS 中应用 DPO/PPO。本文的独特贡献在于: 在 SpeechAlign 仅用 ground truth 作 positive 的基础上,系统地探索了多种 preference pair 构建策略、超参数、metric 选择等实践问题,提供了此前缺失的 engineering transparency。

**创新判断**: 本文的核心价值不在于算法创新,而在于**工程实践透明度** — 系统回答了"如何在 LM-based TTS 中正确实施 DPO"的8个关键问题。这填补了 SpeechAlign/Seed-TTS 等工作在实验细节上的不透明。

## 速查

> [!summary] 速查
> - **一句话**: 在 1.15B 参数的 LM-based TTS 上系统验证 DPO 的最佳实践,证明 preference alignment 可以用 1 小时数据显著提升 WER/SIM/MOS,后两项超越人类语音
> - **路线**: 预训练 LM-TTS (Multi-Scale Transformer + SoundStream) → 生成 preference pairs (metric-based ranking) → DPO 训练 (β=0.01, ~350 updates) → 评估
> - **指标**: WER 4.5→3.0, SPK SIM 0.635→0.667, Proxy MOS 3.80→4.23 (LibriSpeech test-clean) [Table I]; OOD VCTK WER 1.6→1.5, SIM 0.677→0.688, MOS 3.87→4.15 [Table III.b]
> - **可借鉴**: (1) 用 generated win-lose pairs 而非 ground truth 作 positive, (2) 小 β (0.01) 优于默认值, (3) 多 metric 联合排序构建 preference pairs 取得均衡提升, (4) 仅 1h 数据即可有效
> - **局限**: 仅用 proxy metric 代替人类偏好,无真人 MOS 评估; iterative DPO 不稳定; 未探索 PPO 等其他 PA 方法; 开源基于 ESPnet,非最优基线

## 核心问题

本文要回答: **如何在 LM-based TTS 中有效实施 DPO?** 具体拆解为 8 个子问题:

1. Preference pair 怎么构建? (ground truth vs generated samples)
2. 超参数 β 怎么选?
3. Length normalization 是否有用?
4. 用哪个 metric 做 preference pair ranking?
5. DPO 前是否需要 SFT?
6. 需要多少 preference 数据?
7. Iterative DPO 是否有效?
8. DPO 的改进能否泛化到 OOD 场景?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

**Baseline TTS 系统** [§III-A]:
- 模型: Multi-Scale Transformer (来自 UniAudio [13]) — 全局 Transformer (25层, 1600 dim, 25 heads) 预测帧级 embedding,局部 Transformer (6层, 384 dim, 6 heads) 在每帧内自回归预测 nq=8 个 codec codes
- 参数量: 1.15B
- 音频 tokenizer: 复现的 SoundStream,50 fps, 8 codes/frame
- 文本处理: g2p-en 转 phoneme 序列
- 训练数据: LibriSpeech + GigaSpeech + MLS-EN ≈ 55k 小时
- 训练: 1M steps, batch size ~80k frames, AdamW, peak lr 2e-4, 70k warmup + exponential decay, 8xA100-40G
- 推理: top-k (k=30), temperature=1.2, 每个样本 batch inference 10 次取平均

**DPO 在 LM-TTS 上的适配** [§II-B]:
- DPO 目标函数直接套用 Rafailov et al. (2023) [24] 的公式 [Eq. 6]
- 关键适配: 二维 codec 序列 (T x nq) 展平为一维 (row-first),然后按自回归方式累加 code-level log-posterior 计算 P(y|x) [论文原文]
- Reference model: 与 Pθ 初始化相同,训练时冻结 [论文原文]

### 关键设计选择

**1. Preference pair 构建** [§III-B1]:
- **方案 A (ground truth as positive)**: yw = ground truth, yl = 随机选一个生成样本 → 效果差,win rate 20 步后达 99.8%,优化变 trivial [Fig 1.d]
- **方案 B (both generated)**: 用 metric 对所有生成样本排序,取 top 20% 为 yw, bottom 20% 为 yl → 效果好,因为 yw 和 yl 都在模型分布内,优化非 trivial [论文原文]

[agent 解读] 方案 A 失败的根因: 自然语音和生成语音在离散 codec token 空间中差异过大 (分布完全不同),DPO 的 KL 约束使模型无法有效探索。方案 B 的 win-lose pairs 都在模型自身分布上,DPO 的 on-policy 假设更好满足。这与 SpeechAlign 仅用 ground truth 作 positive 的设计形成对比。

**2. β 选择** [§III-B2]:
- 测试 β ∈ {1, 0.1, 0.01}
- β=0.01 一致最优 [Fig 2]
- [agent 解读] 极小的 β 意味着几乎不约束策略偏离参考模型,这与 NLP 中 β 通常 ≥ 0.1 的经验不同,暗示 TTS 的 preference landscape 比文本更 smooth,需要更大的探索空间。

**3. Length normalization** [§III-B3]:
- Seed-TTS [14] 报告 DPO 会使生成变长,SimPO [26] 建议 length normalization
- 实验: C1-3 (有 LN) 一致不如 B3 (无 LN),但 LN 增加了对 β 的鲁棒性 [Fig 2]
- 实际长度影响小: 无 LN 生成比 GT 长 5.1%,有 LN 为 4.1% [§III-B3]

**4. Metric 选择** [§III-B4]:
- 单 metric: SPK SIM (B3), WER (D1), Proxy MOS (D2)
- 任何单 metric 做 preference pair 都能改善所有三个 metric [Table II]
- WER 单独用效果最差 — [论文原文] 推测 WER 关注局部词级错误,而 DPO 是 sequence-level 优化,两者不匹配
- 多 metric 联合排序 (D3): 对每个 metric 分别打分 (0-9),取总分排序 → 取得均衡提升 [Table II]

**5. SFT** [§III-B5]:
- 在 DPO 前用 yw 做一轮 SFT (E1) → 仅有 marginal 改善 [Table II]
- [agent 解读] 因为 baseline 已经在 55k 小时上充分预训练,额外 SFT 的增量有限。

### 训练策略

- DPO 基于 LibriSpeech train-960 (已在 baseline 预训练数据中) → 排除引入新数据的影响 [§III-B]
- 学习率: 恒定 3e-7 [§III-B]
- Batch size 大 → 仅 350 updates/epoch [§III-B]
- DPO 对 update 数量敏感,~300 updates 接近最优 [Fig 1]

## 实验

| 指标 | Baseline | DPO (E1) | Ground Truth | 出处 |
| --- | --- | --- | --- | --- |
| WER ↓ | 4.5 | **3.0** | 1.8 | [Table I] |
| SPK SIM ↑ | 0.635 | **0.667** | 0.625 | [Table I] |
| Proxy MOS ↑ | 3.80 | **4.23** | 4.08 | [Table I] |

**关键发现**:

1. **DPO 后 SIM 和 MOS 超越人类语音**: SPK SIM 0.667 > GT 0.625, Proxy MOS 4.23 > GT 4.08 [Table I]
2. **Label efficiency**: 仅 1 小时 (258 examples) 的 preference pairs 即可取得与 960 小时接近的效果 [Fig 3]
3. **OOD 泛化**: 在 VCTK 上 WER 1.6→1.5, SIM 0.677→0.688, MOS 3.87→4.15 [Table III.b]
4. **Unseen metric 泛化**: 用 OWSM v3.2 / ECAPA-TDNN / DNSMOS 替代训练时的 metric models,WER 5.0→3.2, SIM 0.655→0.679, MOS 3.90→4.00 [Table III.c]
5. **Iterative DPO 不稳定**: G1 (一轮迭代后) WER 5.1 > baseline 4.5 [Table III.a],多次尝试未能改进

## 局限性

1. **无真人主观评估**: 全部使用 proxy metric (Whisper WER, RawNet SIM, UTMOS MOS),存在 metric-model overfitting 风险,虽然 Table III.c 的 unseen metric 实验部分缓解了这一担忧
2. **Preference 来自 proxy metric 而非人类**: 论文称"proxy of real human preferences" [§II-B],但未验证 proxy metric ranking 与真人偏好的一致性
3. **仅测试 DPO**: 未与 PPO、REINFORCE、SimPO、KTO 等其他 PA 方法对比,无法判断 DPO 是否最优选择
4. **Iterative DPO 未能成功**: G1 退化 [Table III.a],论文称"fragile"但未深入分析原因
5. **SIM 超过 GT 的合理性存疑**: [agent 解读] SPK SIM 0.667 > GT 0.625 可能因为模型过度优化了 speaker encoder (RawNet) 的特征空间,而非真正在人类感知维度上超越。知识库 [[TTSEvaluation]] 和 [[SpeakerEmbedding]] 都指出 SIM 指标的局限性 — 超过阈值后 SIM 改善不代表感知增益
6. **基线非 SOTA**: 1.15B Multi-Scale Transformer 是 research 级系统,与 Seed-TTS / CosyVoice 等工业系统的差距使结论的普适性受限

## 点评

**定位**: 这是一篇**工程实证研究**,价值在于系统地回答了"如何在 LM-based TTS 上用好 DPO"的 8 个实践问题,而非提出新算法。

**贡献的独特性**: 与 SpeechAlign (首次将 DPO 引入 codec LM) 和 Seed-TTS (首次在工业 TTS 中用 RL) 相比,本文的贡献是**透明度**: 逐一消融了 preference pair 构建、β 选择、metric 选择、数据效率等关键因素。这些 practical insights 在后续工作 (DiffRO, Multi-Reward GRPO, FPO 等) 中被广泛参考。

**最关键的 insight**: 方案 B (generated win-lose pairs) 远优于方案 A (ground truth as positive) [Fig 1]。这个发现的因果链很清晰: codec token 空间中自然语音与生成语音的分布差异使 ground truth 作 positive 时 DPO 优化变 trivial (win rate 迅速饱和)。后续 SpeechAlign 的 iterative self-improvement 也验证了类似结论。

**在演进线中的位置**: 从知识库 [[DifferentiableRewardOptimization]] 的演进视角看,本文处于"Preference optimization for codec LM"阶段 (2024),之后 DiffRO (2025) 将优化从 sequence-level DPO 下沉到 token-level,FPO (2025) 进一步精细化到 error segment tokens,GRPO (2025) 则走 audio-level reward + group-relative advantage 路线。本文的 1h 数据效率发现也预示了后续 FPO 以 200 utterances 达到 3-4x 效率的方向。

## 可复用的 idea

1. **Generated win-lose pairs >> ground truth as positive**: 对任何 discrete token 空间的 DPO 应用都有参考价值 — 当 positive 与 negative 的 token 分布差异过大时,DPO 优化可能变 trivial
2. **多 metric 联合排序构建 preference pairs**: 简单的分数求和就能取得均衡提升,避免单 metric 偏向
3. **极低 β (0.01)**: TTS DPO 需要比 NLP 更小的 KL 约束,这可能适用于其他非文本模态的 DPO
4. **Label efficiency**: DPO 用 1h 数据即可有效,暗示 post-training alignment 对数据量不敏感,更依赖数据质量 (preference pair 的区分度)
5. **Length normalization 增加 β 鲁棒性**: 即使不带来性能增益,如果需要简化超参数搜索,LN 是一个可考虑的选项
