---
type: paper
tier: deep
title: "UniSRM: A Unified Speech Reward Model for Reasoning-Based Fine-grained Assessment"
arxiv_id: "2605.23261"
source: "Sources/UniSRM.pdf"
authors: [Yuanyuan Wang, Dongchao Yang, Yayue Deng, Zhiyong Wu, Yiwen Guo, Helen Meng, Xixin Wu]
year: 2026
venue: "arXiv"
tags: [speech-evaluation, reward-model, GRPO, reasoning, LLM-as-judge, multi-dimensional, preference-learning, AudioLLM]
concepts: ["[[TTSEvaluation]]", "[[DifferentiableRewardOptimization]]", "[[Audio-LanguagePretraining]]"]
models: ["[[Qwen2.5-Omni]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 2 个待确认实体页: [[SpeechLanguageModel]], [[LLM-basedTTS]], [[TTSEvaluation]], [[DifferentiableRewardOptimization]])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: UniSRM 处于 TTS 评估演进线的 "LLM-as-Judge + Reward Model" 交叉点。在 [[TTSEvaluation]] 页记录的演进中,从 Predicted MOS (UTMOS/DNSMOS) 到 LLM-as-Judge (SpeechLLM-as-Judges) 到 Generative Reward Model (GSRM, SpeechJudge),再到多维诊断 (TTS-PRISM),每条路线都在解决"单一标量评估不够用"的问题。UniSRM 的定位是: 在 task coverage 和 evaluation dimensions 上做统一 (4 种任务 x 多维度评分),并在 RL 训练中引入 reasoning-consistent supervision。

**已有认知**:
- SpeechJudge [待确认] 首创 speech naturalness 的完整 GRM 套件 (SFT+GRPO on Qwen2.5-Omni-7B),达 77.2% accuracy,但仅覆盖 utterance-level pairwise naturalness [TTSEvaluation §Naturalness-Specific]
- GSRM [待确认] 走 acoustic-feature-grounded CoT reasoning 路线,PCC 0.465 接近人类 inter-rater,但仅评估 naturalness 单维度 [TTSEvaluation §GSRM]
- TTS-PRISM [待确认] 走显式 12 维 schema + 端到端模型路线,但仅限 utterance-level 评估 [TTSEvaluation §TTS-PRISM]
- [[DifferentiableRewardOptimization]] 页详细记录了 TTS RL 后训练的 6+ 条路线 (DiffRO/GRPO/DPO/FPO/TKTO/W3AR),UniSRM 从"被优化的 TTS 模型"反转为"提供 reward 的评估模型"

**创新判断**: UniSRM 相对已有工作的增量主要在两处: (1) 任务覆盖从 utterance-level 单任务扩展到 4 种任务 (含 scenario-aware 和 multi-turn dialogue); (2) 提出 RCR-GRPO 在维度级 reasoning 过程上给予监督,而非仅优化最终答案。这填补了 SpeechJudge/GSRM 在 task coverage 上的空白,以及 rule-based GRPO 在 reasoning supervision 上的不足。

> 检索命中: [[TTSEvaluation]][待确认], [[DifferentiableRewardOptimization]][待确认], [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓ | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 Qwen2.5-Omni-7B 的统一语音 reward model,通过 SFT+RCR-GRPO 两阶段训练,在 4 种评估任务上实现多维度、可解释的语音质量判断
> - **路线**: (text prompt + speech audio) → Qwen2.5-Omni-7B-thinker → SFT → RCR-GRPO → <think> 多维度推理 </think> <answer> 偏好/评分 </answer>
> - **指标**: T1 acc 65.06% / T3-Zh acc 91.30% / T4 acc 88.89% / T2 PCC 0.551,均超越 Gemini-2.5-Pro 和 SpeechJudge [Table 1]; BVCC cross-dataset PCC 0.498 vs Gemini-2.5-Pro 0.339 [Table 7]
> - **可借鉴**: RCR-GRPO 的维度级 reasoning supervision (Eq. 10-11) 可迁移到任何需要 CoT+多维打分的 judge model 训练中; 冲突过滤 (cyclic A>B>C>A 移除) 提升标注质量
> - **局限**: 训练数据标注主要依赖 Gemini (LLM-as-annotator 而非人类);仅在 Qwen2.5-Omni-7B 上验证;GRPO 计算成本高 (480 GPU-hours);未覆盖重口音/重叠语音等困难场景

## 核心问题

本文要解决的核心问题是: **语音生成领域缺乏统一的、多维度的、推理可靠的 reward model**。现有方法存在四个缺陷 [§1]:

1. **缺乏透明度**: WER/SIM/UTMOS 等客观指标是黑盒标量,不提供中间解释
2. **评估维度不完整**: 单一指标仅捕获语音质量的一个方面 (如 WER 只反映文本正确性)
3. **任务覆盖有限**: WavReward/SageLM 仅限单轮对话,SpeechJudge 仅限 utterance-level naturalness
4. **推理过程监督不足**: rule-based GRPO (如 SageLM) 仅监督最终答案,reasoning 与 decision 可能不一致

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UniSRM 是一个两阶段训练的统一语音 reward model [§4, Fig 2]:
- **Stage 1: SFT** — 在 UniSRM-Data 上对 Qwen2.5-Omni-7B-thinker 做多任务有监督微调,学习结构化输出格式 (<think> 多维推理 + <answer> 最终判断)
- **Stage 2: RCR-GRPO** — 在人工验证的高质量子集上做 GRPO 强化学习,加入维度级 reasoning-consistent rewards

模型处理 4 种任务,统一为条件生成问题 [§4.1, Eq. 4]:
```
o = π_θ(x) = <think> r̂ </think> <answer> ŷ </answer>
```
其中 r̂ 是多维推理 trace (维度打分+解释), ŷ 是最终判断 (偏好选择/评分)。

### 数据构建: UniSRM-Data 和 UniSRM-Bench

数据覆盖 4 种任务 [§3, Fig 1]:

| 任务 | 输入 | 评估维度 | 数据源 | 标注方式 |
|------|------|----------|--------|----------|
| T1: Utterance A/B Preference | text + prompt + speech A/B | Text Fidelity, SIM, Prosody, Naturalness (4 维, 0-10) | LibriTTS-R + 多 TTS 合成 | Gemini-2.0-Flash |
| T2: Quality Assessment | speech | 7 维 MOS-like (1-5) | QualiSpeech | 已有人工标注 |
| T3: Scenario-Aware Style | scenario + text + speech A/B | Text Fidelity, Style Match, Naturalness (3 维, 0-10) | ESD + GPT-4.1 生成场景 | Gemini-2.5-Pro |
| T4: Multi-turn Dialogue | dialogue history (audio) + speech A/B | Intent, SIM, Context, Emotion, Naturalness (5 维, 0-10) | DailyTalk | Gemini-2.5-Pro |

数据总量: 46,259 样本 (SFT 33,061 / RL 9,674 / Bench 3,524) [Table 9]。

**关键数据质量控制** [§3, Appendix G]:
- **位置去偏**: 随机打乱 (sA, sB) 顺序 [论文原文]
- **冲突过滤**: 移除 cyclic contradiction (A>B, B>C, C>A) [论文原文]
- **人工验证**: RL 和 Test 子集经 3 人标注,仅保留 majority vote 与 auto label 一致的样本 [§3, Appendix G.2]

[agent 解读] T1 的数据构建策略值得注意: 通过多个开源 TTS 模型 + ground truth 组合配对,可以自然产生质量差异,避免了人工构造偏好对的成本。将 ground truth 作为 candidate 之一是一个巧妙设计,因为它提供了质量上界参照。

### 关键设计选择

#### 为什么用两阶段而非纯 SFT 或纯 RL?

[论文原文] SFT 教模型模仿 LLM judge 的推理格式和评分模式,但不能显式优化 reward-aligned correctness,且可能导致固定模式推理 (fixed pattern reasoning) [§4.2]。GRPO 进一步优化推理多样性和可靠性,但需要 SFT 提供稳定初始化 [§4.1]。

[agent 解读] 消融实验 (Table 2) 验证了这个设计: w/o GRPO (SFT-only) 在 T3-En 仅 67.16%,加 GRPO 后 80.81%,再加 RCR 后 85.61%。上下文相关任务 (T3/T4) 上改进最大,说明 SFT 学到的模式在需要深层推理的场景不够用。

#### 为什么需要 Reasoning-Consistent Rewards (RCR)?

[论文原文] 仅优化最终答案的 accuracy reward 可能鼓励 shallow reasoning — 模型可能给出正确标签但 reasoning 与 decision 不一致 [§4.2]。

RCR 对 pairwise 任务 (T1/T3/T4) 的实现 [Eq. 10]:
```
R_rc(o) = (1/D) Σ 1[sign(a_i - b_i) = sign(a*_i - b*_i)]
```
即: 对每个维度,检查模型给 Speech A 和 B 的打分差异方向是否与 ground truth 一致。

对 MOS 评估任务 (T2) [Eq. 11]:
```
R_rc(o) = 1 - (1/D) Σ |m̂_k - m*_k| / (m_max - m_min)
```
即: 对每个维度计算归一化距离 reward。

总 reward 由三部分加权 [Eq. 7]:
```
R = λ_fmt * R_fmt + λ_acc * R_acc + λ_rc * R_rc  (λ_fmt = λ_acc = λ_rc = 1)
```

[agent 解读] RCR 的核心洞察是: 在多维评估中,最终偏好可能因维度间的抵消效应而"碰巧正确" — 比如维度 1 打高了、维度 2 打低了,总分恰好对了。RCR 通过 dimension-wise supervision 消除这种虚假正确性。Table 3-6 的维度级消融证实了这一点: w/o RCR-GRPO 在某些维度上甚至劣于 SFT-only (如 T4 Emotion: 65.87% vs SFT 50.79%),说明 accuracy-only GRPO 确实导致了 reasoning drift。

### 训练策略

**SFT 阶段** [§4.1, Table 10]:
- Backbone: Qwen2.5-Omni-7B-thinker
- LR: 1e-5, batch size 64 (8 GPU x 8 gradient accumulation)
- 4 GPU-hours (1 epoch, 8 GPUs), ~40 GB peak memory/GPU

**GRPO 阶段** [§4.2, Table 10]:
- LR: 1e-6, batch size 16 (8 GPU x 2 gradient accumulation)
- G=8 rollouts per prompt
- KL coefficient β=0.04
- 60 hours (480 GPU-hours), ~30 GB peak memory/GPU

[agent 解读] GRPO 阶段的计算成本 (480 GPU-hours) 是 SFT (30.94 GPU-hours) 的约 15 倍。这与 TTS RL 后训练 (如 GRPO-TTS 等) 的成本分布类似: RL 阶段始终是主要瓶颈。β=0.04 的 KL 正则化防止策略过度偏离 SFT 初始化,这对于 judge model 尤其重要,因为过度偏离可能导致评估标准漂移。

## 实验

| 指标 | UniSRM | Gemini-2.5-Pro | Gemini-2.5-Flash | Qwen2.5-Omni-7B | SpeechJudge | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 acc (%) | **65.06** | 60.67 | 60.44 | 51.20 | 57.20 | UniSRM-Bench | [Table 1] |
| T2 acc/PCC | **39.74/0.551** | 28.93/0.517 | 34.50/0.522 | 24.03/0.289 | -/- | UniSRM-Bench | [Table 1] |
| T3-En acc (%) | **85.61** | 67.31 | 65.68 | 49.45 | - | UniSRM-Bench | [Table 1] |
| T3-Zh acc (%) | **91.30** | 63.47 | 71.74 | 52.17 | - | UniSRM-Bench | [Table 1] |
| T4 acc (%) | **88.89** | 82.40 | 71.43 | 56.35 | - | UniSRM-Bench | [Table 1] |
| BVCC PCC | **0.498** | 0.339 | 0.342 | 0.256 | - | BVCC (cross-dataset) | [Table 7] |
| SOMOS-Clean PCC | **0.261** | 0.201 | 0.250 | 0.156 | - | SOMOS (unseen) | [Table 7] |

**消融实验 (Table 2)**:
| 配置 | T1 | T3-En | T3-Zh | T4 |
|------|-----|-------|-------|-----|
| UniSRM (full) | 65.06 | 85.61 | 91.30 | 88.89 |
| w/o RCR-GRPO | 60.44 | 80.81 | 81.42 | 82.54 |
| w/o GRPO (SFT-only) | 60.24 | 67.16 | 70.95 | 74.60 |

**关键发现**:
1. 上下文相关任务 (T3/T4) 改进最大: T3-Zh 从 SFT 70.95% → +GRPO 81.42% → +RCR 91.30% [Table 2]
2. Accuracy-only GRPO 有时劣于 SFT: T3-En 某些维度 w/o RCR-GRPO 低于 SFT [Tables 3-6],说明 accuracy-only reward 导致 reasoning drift
3. Cross-dataset 泛化: 在完全未见的 SOMOS 上仍优于 Gemini-2.5-Pro [Table 7],说明模型学到了可迁移的评估能力
4. Evidence Groundedness (EG): UniSRM EG_mean 始终高于 w/o RCR-GRPO (T1: 1.57 vs 1.28, T4: 1.98 vs 1.95) [Table 11],证实 RCR 改善了推理质量

## 局限性

1. **标注依赖 LLM**: T1/T3/T4 的训练标注来自 Gemini,本质上是 LLM distillation 而非 human preference learning [agent 解读]。虽然有人工验证 (RL+Test 子集),但 SFT 数据的大部分仍依赖 LLM 标注,模型可能继承 Gemini 的系统性偏差
2. **场景覆盖不足**: 论文承认未覆盖重口音、重叠语音等困难场景 [Limitations]
3. **仅 7B 规模验证**: 只在 Qwen2.5-Omni-7B 上实验,更大/更小模型上效果未知
4. **计算成本高**: GRPO 训练 480 GPU-hours,推理 8.98s/sample [Appendix D],限制了在线部署
5. **T1 绝对性能有限**: utterance-level A/B preference 仅 65.06% [Table 1],虽然最优但仍有大量错误 [agent 解读]。这可能反映了语音偏好的内在主观性
6. **Benchmark 自建自测**: UniSRM-Bench 由同一团队构建,可能存在 benchmark overfitting [agent 解读]。Cross-dataset 实验 (BVCC/SOMOS) 部分缓解了这一担忧

## 点评

**贡献层面**: UniSRM 的主要贡献在于将语音评估从"单任务单维度"推向"多任务多维度统一框架"。在 TTS 评估演进线上,SpeechJudge 解决了 naturalness GRM,GSRM 解决了 feature-grounded reasoning,TTS-PRISM 解决了多维 diagnostic schema,而 UniSRM 补上了 task coverage + reasoning supervision 这块拼图。

**方法层面**: RCR-GRPO 是一个简洁而有效的设计 — 维度级 sign consistency check (Eq. 10) 计算成本低但提供了强信号。消融结果令人信服: accuracy-only GRPO 在某些维度上反而劣于 SFT (Table 3-6),直观地展示了 reasoning shortcut 问题。这个发现对所有使用 CoT+RL 训练 judge model 的工作都有参考价值。

**不足层面**: 论文的核心弱点在于 evaluation protocol — 所有 4 个任务的 benchmark 都是自建的 (UniSRM-Bench),且 T1/T3/T4 的 ground truth 来自 Gemini 标注。虽然有人工验证 (保留 majority vote 一致样本),但 SFT 训练集的标注质量仍取决于 Gemini 的能力天花板。Cross-dataset 实验 (Table 7) 一定程度上缓解了这一问题,但仅覆盖了 T2 (MOS 预测),其他 3 个任务的泛化性缺乏外部验证。

**与 SpeechJudge 的对比**: 两者都基于 Qwen2.5-Omni-7B + SFT + GRPO,但 SpeechJudge 专注 utterance-level naturalness pairwise preference (单任务),UniSRM 覆盖 4 种任务 + 多维度。UniSRM 在 T1 上 acc 65.06% vs SpeechJudge 57.20% [Table 1],但需注意两者 benchmark 不同 (UniSRM-Bench vs SpeechJudge-Eval),直接比较需谨慎。

## 可复用的 idea

1. **RCR 维度级 sign consistency reward**: 对任何多维度评估+CoT 的 judge model 训练均可使用。核心: 不仅优化最终 answer 的正确性,还监督每个维度的比较方向是否与 ground truth 一致。可直接应用于 TTS-PRISM 的 RL 优化
2. **Cyclic conflict filtering**: 数据清洗时移除 A>B, B>C, C>A 循环矛盾,适用于任何 pairwise preference 数据集构建
3. **Hard negative 构造策略**: T3 的 emotion mismatch + TTS negative, T4 的 text/audio/mixed 三类 negative,均是系统性的 hard negative 设计,可应用于对话/风格评估的数据构建
4. **四任务统一为条件生成**: 不同评估任务共享 <think>+<answer> 格式,仅通过 system prompt 区分任务,实现参数共享和跨任务知识迁移
