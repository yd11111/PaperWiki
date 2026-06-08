---
type: paper
tier: deep
title: "AnyAudio-Judge: A Dynamic Rubric-Based Benchmark and Evaluator for Audio Instruction Following"
arxiv_id: "2606.03116"
source: "Sources/AnyAudio-Judge.pdf"
authors: [Haitao Li, Tian Tan, Yuguang Yang, Shan Yang, Xie Chen]
year: 2026
venue: "arXiv preprint"
tags: [evaluation, audio-judge, rubric-based, instruction-following, LALM, GRPO, reward-model, benchmark, multi-domain]
concepts: ["[[TTSEvaluation]]", "[[Audio-LanguagePretraining]]", "[[Instruction-GuidedSpeechSynthesis]]", "[[AudioUnderstanding]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 5 个待确认实体页: [[TTSEvaluation]], [[Audio-LanguagePretraining]], [[Instruction-GuidedSpeechSynthesis]], [[AudioUnderstanding]], [[DifferentiableRewardOptimization]])
> 自动生成,不保证完整覆盖所有相关知识。所有命中页均为 pending-review,仅供参考。
>
> **谱系定位**: AnyAudio-Judge 处于 TTS Evaluation 演进线的最前沿,针对 instruction-following 音频生成的**指令-音频对齐评估**这一子问题。与 KB 中已有的评估工作形成清晰的对比谱系:
> - **SpeechJudge** (Zhang et al., 2025): 专注 naturalness pairwise preference,单任务 GRM
> - **GSRM** (Shen et al., 2026): acoustic-feature-grounded CoT reasoning,专注 naturalness
> - **TTS-PRISM** (Wang et al., 2026): 12 维 schema-driven diagnostic,专注中文 TTS 质量
> - **TTSDS/TTSDS2**: distributional evaluation,关注整体合成质量分布
> - **InstructTTSEval** (Huang et al., 2025): instruction-following TTS benchmark,True/False 二分判断
> - **MINT-Bench** (Chen et al., 2026): 结构化多语言 instruction-following benchmark,三级评分
> - **AnyAudio-Judge** (本文): 统一 speech/sound/music/mixed 四域的指令-音频对齐评估,dynamic rubric 分解
>
> AnyAudio-Judge 与已有工作的根本区别: (1) 评估维度从 naturalness/quality 转向 instruction-audio alignment; (2) 覆盖域从 speech-only 扩展到 speech+sound+music+mixed 四域统一; (3) 评估粒度从 holistic/pairwise 推进到 dynamic instance-specific rubric items。
>
> **已有认知**: GRPO 在 TTS RL 中已被广泛验证 ([[DifferentiableRewardOptimization]])。CLAP ([[Audio-LanguagePretraining]]) 已是音频-文本对齐的标准 embedding metric,但仅提供 coarse global similarity。InstructTTS 系统 ([[Instruction-GuidedSpeechSynthesis]]) 日益成熟,但评估方法滞后。
>
> 检索命中: [[TTSEvaluation]][待确认], [[Audio-LanguagePretraining]][待确认], [[Instruction-GuidedSpeechSynthesis]][待确认], [[AudioUnderstanding]][待确认], [[DifferentiableRewardOptimization]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 dynamic rubric-based evaluation 范式,将复杂音频指令分解为可验证的二值 rubric items 逐项评估,配套 7,920 样本四域 benchmark + 105K 样本训练语料 + 专用 judge 模型,还可作为 InstructTTS RL 的 reward model
> - **路线**: instruction → LLM 分解为 n 个 binary rubric items → LALM judge 逐项评估 yes/no logits → softmax 归一化 → 平均概率得到对齐分数
> - **指标**: Bench 上 zh avg ACC 85.26 / en avg ACC 84.45 (超越 Gemini-2.5-Pro zh 78.31 / en 77.27); PAM 外部 benchmark LCC 0.614 / SRCC 0.601 (超越 CLAPScore 0.472/0.477); InstructTTS RL 后 Gemini Score 69.6→72.9, 人类偏好 Win 34.6%
> - **可借鉴**: (1) dynamic rubric 分解思路可迁移到任何 instruction-following 评估场景; (2) hard negative 构造 pipeline (instruction swapping + attribute perturbation) 是通用的 benchmark 建设方法论; (3) SFT→GRPO 两阶段训练 judge model 的范式; (4) rubric-level 概率聚合比 holistic scoring 更 interpretable
> - **局限**: rubric 分解质量依赖 LLM (Qwen3),分解不完美会遗漏隐含约束或过度拆分; rubric 生成引入额外推理时间; 仅在自建 benchmark + PAM 上验证,缺乏更多外部 benchmark 交叉验证; reward model 下游 RL 仅用 DiTAR 一个模型验证

## 核心问题

1. **评估粒度不足**: 现有 instruction-audio 评估方法 (holistic yes/no judgment) 无法定位到具体哪个指令属性未被满足 [§1]
2. **benchmark 缺失**: 缺乏专门为 judge model discriminative ability 设计的、包含 hard negatives 的多域 benchmark [§1]
3. **judge model 泛化性**: 通用 LLM (如 Gemini) 作为 surrogate judge 的判断与人类感知不一致 [§3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

AnyAudio-Judge 包含三个组件:

**1. AnyAudio-Judge Bench** (评估 benchmark, 7,920 样本) [§3]:
- 覆盖 4 个音频域: Speech, Sound, Music, Mixed
- 7 个子集: Speech-Real, Speech-Gen, Sound-Real, Sound-Gen, Music-Real, Music-Gen, Mix
- 严格 1:1 正负样本比例
- 完全对称的英中双语评估集 (仅翻译指令,不改音频)
- Hard negative 构造: Instruction Swapping (跨样本交换指令) + Attribute Perturbation (LLM 修改指令细节)

**2. AnyAudio-Judge Corpus** (训练语料, 105K 样本) [§4.2]:
- 数据来源与 benchmark 不相交 (OOD 评估)
- 30K Speech + 30K Sound + 30K Music + 15K Mixed,1:1 正负比例
- 两层标注: per-rubric binary labels + CoT rationales
- 负样本标注: 不是全部标 "no",而是通过 text-only LLM (Qwen3) 对比原始 caption 与修改后 caption,逐 rubric 判断 [§4.2]

**3. AnyAudio-Judge Model** (专用 judge) [§4.3]:
- Base model: Qwen3-Omni-30B-A3B-Captioner
- 两阶段训练: SFT → GRPO

### 关键设计选择

**Dynamic Rubric-based Evaluation** [§4.1]:

核心思路: 将一条复杂指令 i 分解为 n 个原子化 binary rubric items {p1, ..., pn},每个 rubric item 是一个 "是否满足?" 的二值问题。

[论文原文] 分解由 LLM (Qwen3-30B-A3B-Instruct-2507) 通过结构化 prompt 完成,要求将指令分解为"独立的、可直接验证的陈述" [§4.1]。分解后还经过 hallucination filter (Table 12) 去除 LLM 幻觉出的非指令内容。

对每个 rubric item pj,judge model 输出 "yes"/"no" 两个 token 的 logits,通过 two-way softmax 归一化得到 soft satisfaction probability:

$$p_j^{yes} = \frac{\exp(z_j^{yes})}{\exp(z_j^{yes}) + \exp(z_j^{no})}$$

最终对齐分数 s = (1/n) * sum(p_j^{yes})

[agent 解读] 这种设计有两个关键优势: (1) 将 holistic matching 分解为多个更简单的 binary verification 问题,降低了每个子问题的推理难度; (2) softmax 概率提供了连续的置信度信号而非离散判断,使其可作为 dense reward signal 用于下游 RL。n 的动态性是关键 — 简单指令可能只分解出 2-3 个 items,复杂指令可分解出 20+ 个 items [Fig 4],这使评估粒度自适应于指令复杂度。

**Hard Negative Construction** [§3]:

[论文原文] 两种策略:
- Instruction Swapping: 交换不同样本的指令,制造明确的语义不匹配
- Attribute Perturbation: 用 LLM (Qwen3) 修改指令中的具体属性 (情感、音色、语速、方言等),模拟生成模型的细粒度失败模式

[agent 解读] 关键的质量保证: 正样本用 Gemini/CLAP 过滤确认对齐; 负样本用 Gemini/CLAP 反向过滤移除 false negatives (交换后仍匹配的 pair)。Speech-Gen 子集还通过 dual-pass Gemini evaluation 挖掘真实的合成失败 (synthesis failure mining),这比人工构造更贴近实际失败模式。

**GRPO Reward Design** [§4.3]:

三项加权奖励:
- Format consistency (权重 0.1): 输出是否为有效 JSON 且包含必需字段
- Global accuracy (权重 0.2): 整体 match/mismatch 判断是否正确
- **Balanced rubric accuracy (权重 0.7)**: 在 gold "yes" 和 gold "no" rubric items 上的平均准确率

[agent 解读] Balanced rubric accuracy 占 70% 权重的设计值得注意 — 这迫使模型在细粒度 rubric 层面学习判断,而非仅靠全局判断拿分。"balanced" (yes/no 各自准确率的平均) 防止模型偏向过度预测某一类。

GRPO 的样本筛选也值得注意: 先做 4 次 rollout,去掉全部正确的 easy samples,仅保留 8,454 个 hard samples 用于优化,避免在已掌握的样本上浪费训练预算。

### 训练策略

**SFT 阶段** [§4.3]:
- 初始化: Qwen3-Omni-30B-A3B-Captioner (audio-capable LALM)
- 全参数微调 1 epoch
- 16x H20 GPU (96GB), batch size 4, lr 1e-5
- 目标: 学习 rubric-following 行为 — 输出 binary judgments + CoT rationales

**GRPO 阶段** [§4.3]:
- LoRA (rank 16, alpha 32) 1 epoch
- 8 generations per prompt, batch size 8, lr 5e-6
- 训练集: 8,454 hard samples (从 105K 中筛选)
- 输出: JSON array,每条包含 rubric ID, binary answer, supporting evidence

[agent 解读] SFT 用全参数微调,GRPO 用 LoRA — 这个选择有道理: SFT 需要大幅调整模型行为 (学习新的 rubric-following 格式),而 GRPO 只是在已有能力上微调决策边界,LoRA 足够且更稳定。

## 实验

| 指标 | 本文 (AnyAudio-Judge) | Best Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Chinese Avg ACC | **85.26** | 78.31 (Gemini-2.5-Pro, holistic) / 76.82 (Qwen3-Omni, dynamic rubric) | AnyAudio-Judge Bench (zh) | [Table 1] |
| English Avg ACC | **84.45** | 77.27 (Gemini-2.5-Pro, dynamic rubric) / 77.34 (Qwen3-Omni, dynamic rubric) | AnyAudio-Judge Bench (en) | [Table 2] |
| PAM LCC | **0.614** | 0.582 (AF3-Think) | PAM | [Table 3] |
| PAM SRCC | **0.601** | 0.589 (Qwen2.5-Omni-7B) | PAM | [Table 3] |
| PAM KTAU | **0.435** | 0.429 (Qwen2.5-Omni-7B) | PAM | [Table 3] |
| InstructTTS RL Gemini Score | 72.9 (after RL) | 69.6 (before RL) | InstructTTSEval | [Fig 6a] |
| InstructTTS RL Human Win Rate | 34.6% win, 38.5% tie, 26.9% lose | — | Human eval | [Fig 6b] |

**关键实验发现**:

1. **Dynamic rubric 范式本身就是一个巨大改进** [Table 4]: 在 Qwen3-Omni-30B-A3B-Captioner 上,仅切换 prompt (holistic→dynamic rubric) 就从 zh 65.33→76.66, en 64.24→76.77,提升约 11-12 个百分点。这说明 explicit decomposition 本身比 holistic judgment 更适合 fine-grained alignment 评估 [agent 解读]。

2. **Holistic prompting 使多数 LALM 接近随机** [Table 1, Table 2]: Audio-Flamingo3, MiDashengLM, Kimi-Audio, Qwen2.5-Omni 在 holistic 模式下多数子集 ACC ≈ 50% (随机水平)。只有 Gemini-2.5-Pro (zh 80.01) 和 Qwen3-Omni (zh 60.24) 能在 holistic 模式下显著超越随机 [agent 解读]。

3. **SFT + GRPO 的增量贡献** [Table 4]: Dynamic rubric (76.66) → +SFT (84.02) → +GRPO (85.26),SFT 贡献 ~7.4 点,GRPO 贡献 ~1.2 点。SFT 是主要提升来源,GRPO 在 hard cases 上做精细调优。

4. **Mix 子集增益最大** [Table 1]: AnyAudio-Judge 在 Mix 子集上 zh ACC 90.60 vs Gemini 75.50 (+15.1 点),说明多域混合场景是现有通用 LLM 的最大弱点,专用训练帮助最大 [agent 解读]。

5. **作为 reward model 的有效性** [§6.2, Fig 5, Fig 6]: DiTAR + GRPO + AnyAudio-Judge reward → Gemini Score 69.6→72.9, 人类偏好 Win 34.6% (vs Lose 26.9%)。Reward 曲线稳步上升 [Fig 5],说明 rubric-level probability 聚合提供了比 binary preference 更 dense 的 reward signal。

6. **InstructTTS 系统评估** [Table 5]: Gemini 2.5-Pro (87.5) > Qwen3-TTS (84.8) > MiMo-Audio (81.1) > MOSS (80.6)。Qwen3-TTS 是最强开源系统。

## 局限性

1. **Rubric 分解质量瓶颈** [Limitations]: 分解依赖 Qwen3 LLM,可能遗漏隐含约束 (如 "在安静的图书馆里" 隐含 "无背景噪声") 或过度拆分单一属性。分解质量直接限制 judge 的上限 [论文原文]。

2. **额外推理开销**: rubric 生成需要 LLM 额外推理一次,在实时/大规模评估场景中可能成为瓶颈 [论文原文]。

3. **Benchmark 的 self-evaluation 风险** [agent 解读]: Bench 和 Corpus 使用相同的 negative construction pipeline (instruction swapping + attribute perturbation),虽然数据源不相交,但负样本的"失败模式分布"可能高度相关,导致在 Bench 上的表现高估了真实世界泛化能力。外部验证仅有 PAM 一个 benchmark (text-to-audio preference,非 instruction-following 场景)。

4. **RL downstream 验证不充分** [agent 解读]: 仅在 DiTAR 一个模型上验证 reward model 效果,且 RL 后改善幅度有限 (Gemini Score +3.3, human win rate 仅 34.6%)。对比 SpeechJudge-GRM 的 N-CMOS +0.25 或 MCLP 的 MOS 3.178→3.576,此处的 RL 收益偏弱。

5. **Gemini 依赖** [agent 解读]: Bench 构造中 Gemini 用于正样本过滤、负样本质量验证、Speech-Gen failure mining,这引入了 Gemini 偏见。如果某些 "false negative" 恰好是 Gemini 与人类判断不一致的 case,这些系统性偏差会传递到 benchmark 中。

6. **音频理解能力天花板** [agent 解读]: Base model (Qwen3-Omni-30B) 的音频理解能力限制了 rubric-level 判断的准确性。在 Sound-Real 等需要精细环境声理解的子集上,AnyAudio-Judge 的优势相对较小 (zh 77.90 vs Gemini 72.00,仅 +5.9)。

## 点评

**创新点**: Dynamic rubric-based evaluation 是一个简洁且有效的想法。将 "这段音频是否匹配这条复杂指令?" 分解为 "男性在说话吗? → 说话轻柔吗? → 带悲伤语气吗? → 背景有钢琴吗? → 音量渐强吗?" [Fig 3],每个子问题的判断难度大幅降低,且失败定位变得透明。这个思路本质上是将 NLP 领域的 rubric-based evaluation (LLM-Rubric, AutoRubric) 首次引入到 audio-instruction alignment 场景,并做了关键适配 — "dynamic" (instance-specific 而非 fixed rubric set) 和 "binary" (yes/no 而非 Likert scale)。

**方法论贡献**: Bench 的 hard negative 构造方法论 (instruction swapping + domain-specific attribute perturbation + multi-pass quality filtering) 是可复用的 benchmark 建设模板,适用于任何 instruction-following 评估场景。

**与 KB 中已有路线的关系**: 如果将 TTS 评估路线概括为 (1) naturalness quality (SpeechJudge/GSRM/TTSDS), (2) multi-dimensional diagnostic (TTS-PRISM), (3) instruction-following alignment (InstructTTSEval/MINT-Bench/AnyAudio-Judge),那么 AnyAudio-Judge 是路线 (3) 的最新进展,且通过统一四个音频域 + dynamic rubric 范式,在方法论深度上超越了 InstructTTSEval (True/False) 和 MINT-Bench (三级评分)。

**局限性思考**: 论文的最弱环节是 downstream RL 验证 — 仅一个模型 (DiTAR),且改善幅度不大。作为 reward model 的价值主张需要更多验证。此外,rubric 分解的质量完全依赖外部 LLM,这意味着 AnyAudio-Judge 的评估上限被 rubric 生成器约束,而非 judge model 本身。

## 审阅

> [!review] 审阅: pass, 0 high / 0 medium / 3 low
> 审阅日期: 2026-06-08 | 审阅报告: `_review/AnyAudio-Judge-review.yml`
> 
> 所有五项原则基本满足 (可复述 9, 可信赖 9, 可区分 9, 可定位 8, 不污染 9)。
> 3 个 low issue 均为 frontmatter 字段留空 (models/datasets/concepts 可补充), 不影响笔记可用性。
> source 字段已从 SpeechQualityEval.pdf 修正为 AnyAudio-Judge.pdf。

## 可复用的 idea

1. **Dynamic rubric decomposition 范式**: 任何 instruction-following 评估 (不限于音频) 都可以用 "分解指令为 atomic binary items → 逐项验证 → 聚合概率" 的模式。关键: 确保 rubric 的 atomicity + verifiability + faithfulness (不引入幻觉)。

2. **Hard negative 构造 pipeline**: Instruction swapping (粗粒度不匹配) + Attribute perturbation (细粒度不匹配,模拟生成模型失败模式) + Multi-pass quality filtering (Gemini/CLAP 去除 false negatives)。这个三层 pipeline 可直接迁移到其他 benchmark 建设。

3. **Balanced rubric accuracy 作为 GRPO reward**: 70% 权重给 rubric-level balanced accuracy 而非 global accuracy,迫使模型学习细粒度判断。这种 reward 设计可用于任何需要多维度/多项评估的 judge model 训练。

4. **GRPO hard sample mining**: 先做多次 rollout 筛掉 easy samples,仅在 hard samples 上做 GRPO。这个策略在 RL 训练中普遍有用。

5. **Rubric-level probability 作为 dense reward**: 相比 holistic binary label 或 scalar embedding score,逐 rubric 的 softmax 概率聚合提供了更 fine-grained 且 interpretable 的 reward signal,可用于任何下游 RL。
