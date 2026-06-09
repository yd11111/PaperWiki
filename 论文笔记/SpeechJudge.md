---
type: paper
tier: deep
title: "SpeechJudge"
aliases: [SpeechJudge, SpeechJudge-Data, SpeechJudge-Eval, SpeechJudge-GRM]
arxiv_id: "2511.07931"
source: "Sources/SpeechJudge.pdf"
authors: [Xueyao Zhang, Chaoren Wang, Huan Liao, Ziniu Li, Yuancheng Wang, Li Wang, Dongya Jia, Yuanzhe Chen, Xiulin Li, Zhuo Chen, Zhizheng Wu]
year: 2025
venue: "arXiv 2025 (ByteDance Seed + CUHK Shenzhen)"
tags: [TTS-evaluation, naturalness, reward-model, GRPO, SFT, human-feedback, pairwise-preference, benchmark, AudioLLM, speech-quality]
concepts: ["[[TTSEvaluation]]", "[[DifferentiableRewardOptimization]]", "[[AudioUnderstanding]]"]
models: []
tasks: [speech-naturalness-judgment, reward-modeling, TTS-post-training]
datasets: [SpeechJudge-Data, SpeechJudge-Eval]
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: 无 confirmed 命中 | 过滤: [[TTSEvaluation]](pending-review), [[DifferentiableRewardOptimization]](pending-review), [[AudioUnderstanding]](pending-review) | 未命中但可能相关: 无

- **[[TTSEvaluation]]** [待确认]: SpeechJudge 直接针对 TTS 评估的核心痛点 -- naturalness 维度。现有客观指标 (WER, SIM, UTMOS) 与人类偏好弱相关 [Table 2, 最高 Gemini-2.5-Flash 仅 69.1%]; MOS 主观评估存在 ceiling effect 和不可比性。SpeechJudge 构建的 99K pairwise preference 数据集和 GRM 填补了 naturalness-specific reward model 的空白 [agent 解读]。
- **[[DifferentiableRewardOptimization]]** [待确认]: SpeechJudge-GRM 使用 GRPO (与 DiffRO 同属 RL 后训练范式) 训练 reward model。与 CosyVoice 3 的 DiffRO 不同,SpeechJudge-GRM 的 GRPO 直接在 AudioLLM 上做 RLVR,reward 为 human preference label (pairwise),不需要 token-level 反传。SpeechJudge-GRM 还可反向作为 TTS 后训练的 reward function (§5.4) [agent 解读]。
- **[[AudioUnderstanding]]** [待确认]: SpeechJudge-Eval 发现 AudioLLM 在 naturalness 判断上有潜力但表现参差 -- 最好的 Gemini-2.5-Flash 也仅 69.1%,部分模型接近随机 (GPT-4o mini Audio 50.5%) [Table 2]。这揭示了 AudioLLM 理解 fine-grained 语音质量差异的能力瓶颈 [agent 解读]。

> [!summary] 速查
> - **一句话**: 首个针对 TTS naturalness 的完整评估套件: 99K pairwise 人类偏好数据集 (SpeechJudge-Data) + 1K 高一致性 benchmark (SpeechJudge-Eval) + 基于 GRPO 训练的 generative reward model (SpeechJudge-GRM, 77.2% accuracy)
> - **路线**: 6 个零样本 TTS 模型生成语音对 → 69 名标注员 pairwise preference 标注 (99K pairs) → SFT (Gemini-2.5-Flash CoT distillation) + GRPO (human preference as verifiable reward) → SpeechJudge-GRM; 可用作 TTS 后训练 reward function
> - **指标**: SpeechJudge-GRM 77.2% accuracy (vs BTRM 72.7%); Voting@10 79.4%; 用于 TTS 后训练 N-CMOS +0.25 (online) [Table 3, Fig 6]
> - **可借鉴**: (1) Pairwise preference + GRPO 训练 GRM 比 Bradley-Terry 标量 reward 更强; (2) CoT-based SFT 冷启动; (3) 用 GRM 做 TTS 后训练 (offline DPO + online reward); (4) 大规模 naturalness preference 数据集构建方法
> - **局限**: 仅覆盖中英文 + code-switching; 标注者均为中国专业评分员; 仅针对合成语音 (非自发语音); 数据集中中文一致性显著高于英文; GRM 在 expressive + 极端风格上仍有 failure cases; CoT 质量依赖 Gemini-2.5-Flash teacher

## 核心问题

### WHY: 为什么要做这个工作?

TTS naturalness 评估面临三个根本问题 [§1]:
1. **缺乏大规模 naturalness preference 数据集**: 现有人类反馈语料要么是 MOS 点数值,要么聚焦 intelligibility/acoustic quality,缺乏专门针对 naturalness 的 pairwise preference 数据 [论文原文]
2. **客观指标与人类感知不一致**: WER, SIM, FAD, UTMOS 等客观指标在 naturalness 判断上表现差 -- WER 仅 57.9%, SIM 仅 44.5%, UTMOS 仅 53.7% [Table 2] [论文原文]
3. **AudioLLM as judge 潜力未挖掘**: 最强 Gemini-2.5-Flash 也只有 69.1% accuracy,说明 AudioLLM 需要专门训练才能做好 naturalness 评估 [Table 2] [论文原文]

### WHAT: 核心贡献

1. **SpeechJudge-Data** [§3]: 99K pairwise speech preference 数据集,69 名标注员,2+ 个月标注,涵盖 6 种 TTS 模型 x 中英文 + code-switching x regular + expressive 风格 [论文原文]
2. **SpeechJudge-Eval** [§4]: 1,000 样本 benchmark (仅 Full Agreement 子集),评估各类模型的 naturalness 判断能力 [论文原文]
3. **SpeechJudge-GRM** [§5]: 基于 Qwen2.5-Omni-7B 的 generative reward model,SFT + GRPO 两阶段训练,77.2% accuracy 超越 BTRM 72.7% [Table 3] [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SpeechJudge 是一个三组件套件:

**组件 1: SpeechJudge-Data** [§3]
- 形式: D = {(t, a_1, a_2)} -- target text + speech pair [§3] [论文原文]
- TTS 模型选择: AR-based (ARS, CosyVoice2, CosyVoice2-INTP, Ints-INTP), FM-based (F5-TTS), MGM-based (MaskGCT) [§3.1, Fig 2a] [论文原文]
- Speech reference: regular (Emilia-Large 69.9%) + expressive (ParaSpeechCaps, L2-Arctic, KeSpeech, Genshin Impact whispers, in-house) [Fig 2b] [论文原文]
- 语言设置: en2en, zh2zh, en2zh, zh2en, en2mixed, zh2mixed [Fig 2c] [论文原文]
- 标注任务: (a) pointwise intelligibility (有无错误) + (b) pairwise naturalness preference (5 档 CMOS) [Fig 1] [论文原文]
- 标注员: 69 名中国专业标注员,中文母语,英文 CET-6+; 每样本 2-3 名标注员,平均 2.49 名 [§3.2] [论文原文]
- 规模: 99K (t, a_1, a_2) 样本; 标注成本 ~50 万 RMB (~7 万 USD) [§3.2] [论文原文]

**标注一致性分析** [§3.2, Fig 3]:
- Full Agreement (FA): 51.4%
- Weak Agreement (WA): 17.2%
- Weak Disagreement (WD): 19.1%
- Full Disagreement (FD): 12.3%
- Expressive 子集一致性低于 regular (FA 44.3% vs 54.5%),说明 expressive 语音的 naturalness 评估更主观 [论文原文]

**组件 2: SpeechJudge-Eval** [§4]
- 构建: 从 SpeechJudge-Data 中筛选 preference-only (排除 Tie) + Full Agreement (FA) 子集 [§4.1] [论文原文]
- 规模: 1,000 samples,按 regular/expressive 和 zh/en/mixed 比例采样 [§4.1] [论文原文]
- 评估 protocol [Table 1]: 4 类评估者 -- 客观指标 (WER/SIM/FAD), MOS predictors (DNSMOS/UTMOS), Deepfake detectors (AASIST/ADV), AudioLLMs (open/closed-source) [论文原文]

**组件 3: SpeechJudge-GRM** [§5]

### 关键设计选择

**SFT 阶段 (冷启动)** [§5.1]:
- 基座: Qwen2.5-Omni-7B (Thinker) [论文原文]
- Teacher: Gemini-2.5-Flash (SpeechJudge-Eval 上最强 closed-source 模型, 69.1%) [论文原文]
- 数据构建: 用 CoT prompt 让 Gemini-2.5-Flash 生成 rationale-based output → 解析 preference → 与 human label 一致的样本作为 SFT 数据; 不一致的保留给 RL 阶段 [Fig 4a] [论文原文]
- SFT 训练: next token prediction 仅在 output 段 (不含 prompt), LoRA fine-tuning [§5.1] [论文原文]
- **WHY 用 Gemini-2.5-Flash 做 teacher**: Qwen2.5-Omni 的 instruction-following 和 reasoning 能力弱,直接做 RLVR 效果差,需要 SFT 冷启动 [Appendix D] [论文原文]

**GRPO (RL) 阶段** [§5.1]:
- 以 human preference y_H 作为 verifiable reward [论文原文]
- RL 数据 = SFT 阶段 Gemini 判断错误的 challenging cases [论文原文]
- 用 CoT prompt 做 multiple rollouts → 解析每个 rollout 的 preference y_M → reward = +1 if y_M = y_H, -1 otherwise [论文原文]
- 只约束最终 naturalness preference,reasoning 过程自由优化 [论文原文]
- LoRA fine-tuning [§5.1] [论文原文]

**GRM vs BTRM 的关键区别** [§5]:
- BTRM (Bradley-Terry Reward Model): 线性层输出标量 reward,无 CoT [论文原文]
- GRM: 生成 rationale + preference,支持 inference-time scaling (majority voting) [论文原文]
- 核心优势: GRM 的 CoT 提供可解释性 + voting@10 额外涨 ~2% [论文原文]

### 训练策略

两阶段递进: SFT (CoT distillation from Gemini) → GRPO (human preference as verifiable reward on hard cases)。关键 insight: SFT 用 teacher 的 easy cases 训练基础能力,RL 用 teacher 失败的 hard cases 进一步提升 [agent 解读]。

## 实验

### SpeechJudge-Eval Benchmark [Table 2]

| 类别 | 模型 | Regular | Expressive | Total | 出处 |
| --- | --- | --- | --- | --- | --- |
| 客观指标 | WER | 59.3 | 57.0 | 57.9 | [Table 2] |
| 客观指标 | SIM | 47.5 | 42.5 | 44.5 | [Table 2] |
| 客观指标 | FAD | 50.3 | 47.5 | 48.6 | [Table 2] |
| MOS Predictor | DNSMOS | 61.0 | 55.8 | 57.9 | [Table 2] |
| MOS Predictor | UTMOS | 54.0 | 53.5 | 53.7 | [Table 2] |
| AudioLLM (open) | Kimi-Audio-7B-Instruct | 65.5 | 68.0 | 67.0 | [Table 2] |
| AudioLLM (open) | Qwen2.5-Omni-7B | 59.7 | 60.6 | 60.6 | [Table 2] |
| AudioLLM (closed) | Gemini-2.5-Flash | 73.5 | 66.2 | 69.1 | [Table 2] |
| AudioLLM (closed) | GPT-4o Audio | 71.5 | 64.7 | 67.4 | [Table 2] |

### SpeechJudge-GRM 效果 [Table 3]

| 模型 | Regular | Expressive | Total | 出处 |
| --- | --- | --- | --- | --- |
| Qwen2.5-Omni-7B | 62.0 | 59.7 | 60.6 | [Table 3] |
| Gemini-2.5-Flash | 73.5 | 66.2 | 69.1 | [Table 3] |
| SpeechJudge-BTRM | 77.5 | 69.5 | 72.7 | [Table 3] |
| SpeechJudge-GRM (SFT) | 77.8 | 73.7 | 75.3 | [Table 3] |
| SpeechJudge-GRM (SFT+RL) | 79.0 | 76.0 | 77.2 | [Table 3] |
| SpeechJudge-GRM w/ Voting@10 | 80.5 | 78.7 | 79.4 | [Table 3] |

### TTS 后训练效果 [§5.4, Fig 6]

| 模型 | T-ACC | N-CMOS | 出处 |
| --- | --- | --- | --- |
| Qwen2.5-0.5B-TTS (base) | 84.0% | 0.00 | [Fig 6a] |
| w/ INTP (DPO) | 87.0% | +0.18 | [Fig 6a] |
| w/ SpeechJudge-Data (DPO) | 91.0% | +0.16 | [Fig 6a] |
| w/ SpeechJudge-GRM (offline) | 91.0% | +0.21 | [Fig 6a] |
| w/ SpeechJudge-GRM (online) | 90.0% | +0.25 | [Fig 6a] |

## 局限性

1. **语言和标注者偏差**: 仅覆盖中英文 + code-switching; 标注者均为中国人,中文一致性显著高于英文 [Appendix B.2] [论文原文]
2. **仅合成语音**: 数据集仅含零样本 TTS 合成语音,非自发对话 [§6] [论文原文]
3. **Residual failure cases**: GRM 在 "clean but robotic vs slightly noisy but lively", "extreme expressive styles (very high-F0, whispers)" 上仍有困难 [§6, Appendix F.3] [论文原文]
4. **CoT 质量依赖 teacher**: SFT 阶段的 CoT 来自 Gemini-2.5-Flash,继承其推理偏差 [§6] [论文原文]
5. **Utterance-level 粒度**: 标注和建模均在整句级别,无法定位句内质量不均匀的区域 [§6] [论文原文]

## 点评

SpeechJudge 填补了 TTS 评估中 naturalness 维度的关键空白。99K pairwise preference 数据集的规模和质量 (~50 万 RMB 标注成本) 在语音领域罕见,体现了 ByteDance 的数据优势 [agent 解读]。

SpeechJudge-Eval 的最重要发现是: **所有现有客观指标在 naturalness 判断上都接近随机** (WER 57.9%, SIM 44.5%, UTMOS 53.7%)。这意味着当前 TTS 论文中普遍使用的 UTMOS/DNSMOS 作为 naturalness 代理是不可靠的 [agent 解读]。

GRM 相比 BTRM 的优势主要来自两点: (1) CoT reasoning 提供了 inference-time scaling 能力 (voting@10: 77.2% → 79.4%); (2) 生成式输出使模型可以灵活适应评估维度 (prosody, rhythm, articulation, overall naturalness) [agent 解读]。

将 SpeechJudge-GRM 用作 TTS 后训练 reward function (§5.4) 是最有实用价值的贡献 -- online reward 模式下 N-CMOS +0.25,且不需要预先构建 preference pairs [agent 解读]。

## 可复用的 idea

1. **Pairwise preference + GRPO 训练 GRM**: 比 Bradley-Terry 标量 reward 更强,且支持 inference-time scaling; 可迁移到任何需要 reward model 的语音任务
2. **SFT + RL 两阶段**: SFT 用 teacher 的 easy cases 冷启动, RL 用 hard cases 进一步提升 -- 通用 reward model 训练范式
3. **大规模 pairwise naturalness preference 数据**: 构建方法 (多模型 x 多风格 x 多语言 x intra/inter-model pairs) 可指导类似数据集建设
4. **GRM 作为 TTS 后训练 reward**: online DPO 模式下直接用 GRM 打分替代人工偏好标注,实现自动化 TTS 改进

---

检索命中: 无 confirmed 命中 | 过滤: [[TTSEvaluation]](pending-review), [[DifferentiableRewardOptimization]](pending-review), [[AudioUnderstanding]](pending-review) | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass-with-fixes
> 
> 结构检查: 速查卡片 ✗ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
