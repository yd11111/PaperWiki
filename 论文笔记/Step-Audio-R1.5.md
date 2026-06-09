---
type: paper
tier: deep
title: "Step-Audio-R1.5 Technical Report"
arxiv_id: "2604.25719"
source: "Sources/Step-Audio-R1.5.pdf"
authors: [StepFun-Audio Team]
year: 2026
venue: "arXiv"
tags: [speech-LM, audio-reasoning, RLHF, RLVR, chain-of-thought, multi-turn-dialogue, reinforcement-learning, audio-understanding, paralinguistic]
concepts: ["[[SpeechLanguageModel]]", "[[AudioUnderstanding]]", "[[DifferentiableRewardOptimization]]", "[[ModalityAdaptationforSpeechLLM]]", "[[SpokenDialogueEvaluation]]", "[[ProsodyModeling]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[SpeechLanguageModel]], [[ProsodyModeling]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[ProsodyModeling]]✓ | 过滤: [[AudioUnderstanding]](pending-review), [[DifferentiableRewardOptimization]](pending-review), [[SpokenDialogueEvaluation]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: Step-Audio-R1.5 定位为 audio reasoning model,而非完整的 speech-in-speech-out SpeechLM。它采用 latent-representation-based integration: Qwen2 audio encoder (frozen, 25Hz) → audio adaptor (2x down, 12.5Hz) → Qwen2.5 32B LLM → text output。与 Step-Audio 系列的演进关系值得注意: Step-Audio v1 (130B AQTA + 3B TTS decoder) → Step-Audio 2 (端到端 text+audio token 交织) → Step-Audio-R1/R1.5 (专注 audio reasoning 的文本输出模型)。R1.5 的架构更接近 Qwen2-Audio/SALMONN 等 latent-representation 理解模型,而非 Step-Audio 2 的全端到端路线。
>
> **Prosody Modeling**: Step-Audio-R1.5 的核心论点恰恰围绕韵律问题: RLVR 训练的模型虽然准确但"韵律上死气沉沉",因为 RLVR 只奖励正确的离散标签而完全忽视韵律自然度、情感连续性和对话沉浸感。论文将此命名为 "verifiable reward trap"。这与 KB 中 "No Verifiable Reward for Prosody" (Channel Corp, 2026) 的发现高度呼应——后者在 TTS 领域也证实 GRPO 导致 logF0 分布收窄,韵律坍缩; 而本文在 audio reasoning 领域揭示了类似现象。

> [!summary] 速查
> - **一句话**: 提出 "verifiable reward trap" 概念,论证 RLVR 系统性损害音频模型的对话自然度,通过整合 RLHF (rubric-guided preference reward model + PPO) 在保持推理能力的同时大幅改善多轮交互体验 [§Abstract]
> - **路线**: Input audio → Qwen2 audio encoder (frozen, 25Hz) → Audio adaptor (2x down, 12.5Hz) → Qwen2.5 32B LLM decoder (CoT: reasoning trace → final reply) → text output [§2]
> - **指标**: Avg 77.97 / 8 benchmarks (vs R1 72.50, +5.47); Audio MC 41.15 (vs R1 24.61, +67%); Step-DU 82.76 (vs R1 64.37, +18.39); 竞争 Gemini 3 Pro (79.67) [Table 1]
> - **可借鉴**: (1) "Verifiable reward trap" 诊断框架: RLVR 将连续感官信号压缩为离散标签,系统性忽视 "how to say it"; (2) Rubric-based generated reward model: 支持 rubric-guided + ordinary preference 两种评估模式的统一 RLHF; (3) 联合优化 instruction-sensitive 和 preference-sensitive 目标 (非分阶段训练,避免遗忘)
> - **局限**: 纯文本输出模型,不生成语音 (对话体验改善的评估依赖 benchmark 分数而非主观 MOS); 论文声称"改善对话自然度"但缺乏韵律/自然度维度的直接评估 (无 MOS/CMOS/prosody metrics); 仅 32B 参数; 训练数据细节未公开; RLHF 的 reward model 质量和 human annotation 规模未说明

## 核心问题

Step-Audio-R1.5 要解决的核心问题是: RLVR 训练的 audio reasoning 模型在客观 benchmark 上表现优异,但在真实交互中体验极差 [§1]。论文将此问题定义为 **"verifiable reward trap" (可验证奖励陷阱)**:

1. **结构性盲区**: RLVR 的 reward 信号是二值化的离散标签 (数字/类别/短字符串),只奖励 "what to say" 而对 "how to say it" 结构性失明 [§1]。音频输入被压缩为一个孤立标签,丢失了韵律自然度、情感连续性和对话连贯性 [论文原文]
2. **经验退化**: 长时间 RLVR 训练后,模型变得准确但不自然——回复简短、机械、情感平淡,在多轮对话中退化为 "answering machine" [§1] [论文原文]
3. **优化目标错配**: factual correctness 是必要条件但非充分条件; 用户在多轮对话中期待的不仅是正确答案,更是真实的对话流 [§1] [论文原文]

**与 Step-Audio-R1 的关系**: R1 是纯 RLVR 训练的版本,R1.5 在 R1 基础上引入 RLHF 来弥补 RLVR 的缺陷 [§1, §3]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Step-Audio-R1.5 沿用 Step-Audio-R1 的三组件架构 [§2]:

1. **Audio Encoder**: Qwen2 audio encoder, 25Hz 帧率, 全程冻结,保持鲁棒的听觉感知能力 [§2] [论文原文]
2. **Audio Adaptor**: 时间下采样率 2x, 将帧率从 25Hz 压缩至 12.5Hz, 缓解多轮交互中的序列长度爆炸 [§2] [论文原文]
3. **LLM Decoder**: 从 Qwen2.5 32B 初始化, 直接消费下采样后的音频特征, 生成纯文本输出 [§2] [论文原文]

**CoT 机制**: 生成过程结构性分为两阶段——先产出显式中间推理轨迹 (reasoning trace),再自回归生成最终回复。这种解耦是 RLHF 集成的架构基础: RLHF 可以分别优化推理过程和最终响应的质量 [§2] [论文原文]。

[agent 解读] 架构选择上,Step-Audio-R1.5 走的是 latent-representation-based 集成路线 (Qwen2 encoder + adaptor → LLM),而非 Step-Audio 2 的 audio-token-based 端到端路线。这意味着它只能做理解+推理,不能生成语音——是一个专注 audio reasoning 的 text-only 输出模型。

### 关键设计选择

#### 为什么 RLHF 而非纯 RLVR

论文论证了 RLVR 在音频域的根本局限 [§1]:

| 维度 | RLVR | RLHF (Step-Audio-R1.5) |
|------|------|------------------------|
| 奖励信号 | 二值化离散标签 (对/错) | 人类整体偏好判断 |
| 优化目标 | isolated factual correctness | correctness + fluency + emotional resonance |
| 评估粒度 | 单个标签 | 端到端交互质量 |
| 在多轮对话中 | 退化为 "answering machine" | 保持对话流和沉浸感 |

[agent 解读] 这个诊断与 KB 中 DifferentiableRewardOptimization 的演进线高度相关: TTS 领域已充分证明单维 reward (如只优化 WER) 会导致韵律坍缩 (Channel Corp, 2026)。Step-Audio-R1.5 在 audio reasoning 领域独立发现了同样的问题模式——reward 信号维度不足导致模型在被优化维度上过度收敛,在未被覆盖的维度上退化。

#### Rubric-based Generated Reward Model

Step-Audio-R1.5 的 RLHF 核心是一个支持两种模式的 generated reward model [§3.3]:

**模式 1: Rubric-guided evaluation** — 对有明确评估标准的样本,reward model 以 task-specific rubric 为条件评估响应是否满足要求 [§3.3] [论文原文]

**模式 2: Ordinary pairwise preference** — 对没有明确标准的样本 (如对话自然度、语气适当性),执行标准的 pairwise preference judgment [§3.3] [论文原文]

形式化: 给定多轮对话历史 $H_{1:T}$, policy 响应 $y$, 参考响应 $y^{ref}$, 可选 rubric $c$ [§3.3]:
$$g = R(H_{1:T}, y, y^{ref}; c), \quad c \in C \cup \{\emptyset\}$$
$c = \emptyset$ 为普通偏好比较, $c \neq \emptyset$ 为 rubric 条件评估。判断 $g$ 通过映射函数 $r = \phi(g)$ 转为标量 reward [§3.3]。

**相对比较而非绝对评分**: reward model 比较 policy 响应与参考响应的相对质量,而非给出绝对分数。论文认为这更适合口语对话对齐,因为交互质量难以用单一绝对分数校准。多序数级别 (multiple ordinal levels) 的相对偏好信号比二值区分提供更具判别力的监督 [§3.3] [论文原文]。

### 训练策略

三阶段训练 pipeline [§3]:

#### Stage 1: Audio-Centric Mid-Training [§3.1]

在 post-training 对齐之前,先强化音频理解、音频推理和一般推理能力:

$$L_{mid} = \mathbb{E}_{(x,q,r,y) \sim D_{audio}} [\log \pi_\theta(r, y | x, q)] + \mathbb{E}_{(q,r,y) \sim D_{text}} [\log \pi_\theta(r, y | q)]$$

混合两种数据: (1) audio-grounded samples $(x, q, r, y)$ 包含音频输入 + 推理轨迹 + 响应; (2) text-only samples $(q, r, y)$ 提供高质量推理轨迹和长链推理结构 [§3.1]。

[agent 解读] 文本推理数据的加入是典型的跨模态迁移策略——用纯文本的 CoT reasoning 能力 "warm up" 音频推理。这与 Step-Audio 2 的 reasoning-centric RL 中使用文本 reasoning 数据的思路一致。

#### Stage 2: Cold-start SFT [§3.2]

为交互导向行为提供监督初始化,重点强化四个方面 [§3.2]:
1. **Multi-turn dialogue continuity**: 跨轮次维持上下文和用户约束
2. **Instruction following**: 在内容/格式/风格要求下保持一致响应
3. **Response naturalness**: 产出对话式而非任务式的回复
4. **Interaction awareness**: 鲁棒处理追问、澄清、打断和修改

[agent 解读] 这个 cold-start SFT 阶段的设计意图很明确: 它不扩展领域知识,而是给 RLHF 提供一个"有对话能力"的起点。如果直接从 mid-training 后的模型做 RLHF,RLHF 需要同时教会模型"对话"和"偏好",效率很低。Cold-start SFT 把"对话"部分先搞定,让 RLHF 集中精力优化交互质量。

#### Stage 3: RLHF with PPO [§3.3]

使用 PPO-style 目标优化 policy [§3.3]:

$$L_{RLHF}(\theta) = \mathbb{E}_t \left[\min\left(\rho_t(\theta) \hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right) - \beta D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

**联合优化**: rubric-guided 和 preference-based 两种监督信号在同一 stage 联合优化,而非分阶段训练。论文指出经验上分离训练会导致非平凡遗忘——后优化的目标会退化先优化的行为 [§3.3] [论文原文]。

[agent 解读] 这是一个重要的工程经验: 在 TTS RL 领域,Multi-Reward GRPO (Tencent, 2025) 也发现组合多个 reward 不如联合训练; DiffRO + GRPO 的直接合并也会变差 (Tongyi, 2025)。Step-Audio-R1.5 在 audio reasoning RLHF 中也碰到类似问题,选择了联合优化而非串行。

## 实验

### 评估体系

Step-Audio-R1.5 在 8 个 speech-to-text benchmark 上评估 [§4.1]:

| Benchmark | 评估维度 |
|-----------|----------|
| AudioMultiChallenge (Audio MC) | 多轮对话: Inference Memory, Instruction Retention, Self Coherence, Voice Editing [§4.1] |
| Big Bench Audio | 复杂多步逻辑推理 [§4.1] |
| MMSU | 专家级音频理解推理 [§4.1] |
| MMAU | 多模态音频理解 [§4.1] |
| Spoken MQA | 口语数学推理 [§4.1] |
| Step-Caption | 细粒度音频描述 (16 维度, 905 样本) [§4.1] |
| Step-DU | 副语言特征问答 (87 样本) [§4.1] |
| Step-SPQA | 音频副语言评估 [§4.1] |

### 主要结果

| 指标 | Step-Audio-R1.5 | Step-Audio-R1 | Gemini 3 Pro | Gemini 3 Flash | qwen3.5-omni-plus | 出处 |
|------|----------------|---------------|-------------|----------------|-------------------|------|
| **Avg (8 benchmarks)** | **77.97** | 72.50 | **79.67** | 77.56 | 75.77 | [Table 1] |
| Audio MC | 41.15 | 24.61 | 66.37 | 56.42 | 39.38 | [Table 1] |
| Big Bench Audio | 98.30 | 98.29 | 99.40 | 96.80 | 73.03 | [Table 1] |
| MMSU | 79.03 | 75.68 | 83.70 | 76.64 | 82.74 | [Table 1] |
| MMAU | 77.90 | 77.00 | 79.80 | 75.90 | 79.60 | [Table 1] |
| Spoken MQA | 93.74 | 95.06 | 96.56 | 95.37 | 96.03 | [Table 1] |
| Step-Caption | 71.48 | 70.60 | 75.55 | 65.12 | 74.93 | [Table 1] |
| Step-DU | 82.76 | 64.37 | 72.41 | 80.46 | 85.63 | [Table 1] |
| Step-SPQA | 79.40 | 74.36 | 63.60 | 73.80 | 74.80 | [Table 1] |

### 关键发现

1. **平均得分排名第二** (77.97),仅次于 Gemini 3 Pro (79.67),以 32B 参数竞争远超自身规模的商业模型 [§4.2]
2. **多轮对话大幅提升**: Audio MC 从 R1 的 24.61 跃至 41.15 (+67%),这正是 RLHF 对 multi-turn interaction quality 改善的直接证据 [§4.2]
3. **副语言理解显著改善**: Step-DU +18.39, Step-SPQA +5.04 [§4.2]
4. **推理能力保持**: Big Bench Audio 98.30 (与 R1 的 98.29 基本持平), MMSU/MMAU 小幅提升 [§4.2]
5. **Spoken MQA 轻微下降**: 93.74 vs R1 的 95.06,说明 RLHF 优化对话质量时对纯数学推理有微小 trade-off [Table 1]

## 局限性

1. **"对话体验改善"缺乏直接证据**: 论文的核心论点是 RLHF 改善了对话自然度和情感连续性,但评估完全基于客观 benchmark 分数,没有主观评估 (MOS/CMOS)、韵律指标 (F0 variance)或人类偏好测试。Audio MC 分数提升可能反映的是 multi-turn 推理能力而非对话"感觉"的改善 [agent 解读]

2. **纯文本输出**: 模型只输出文本,不生成语音。论文讨论的"prosodic naturalness"和"emotional continuity"在一个不生成语音的模型中如何体现?这些特质需要在下游 TTS 合成后才能被感知 [agent 解读]

3. **消融实验缺失**: 没有分离 RLHF 各组件 (rubric-guided vs preference, 联合 vs 分阶段) 的消融,难以判断增益来源 [agent 解读]

4. **Reward model 细节不足**: human annotation 的规模、标注指南、annotator 间一致性均未说明; rubric 的具体内容未公开 [agent 解读]

5. **评估公平性**: 所有 baseline 使用官方 API 评估是好的实践,但 Step-Caption / Step-DU / Step-SPQA 是 StepFun 自建 benchmark,存在潜在的 "home advantage" [agent 解读]

6. **与 Step-Audio 2 的关系不清**: R1.5 使用 Qwen2.5 32B 而非 Step-Audio 2 的 LLM; 两条线 (reasoning model vs full-duplex dialogue model) 如何整合未说明 [agent 解读]

## 点评

Step-Audio-R1.5 的最大贡献在于**概念诊断**而非技术方案。"Verifiable reward trap" 是一个精准的概念提炼——它将 RLVR 在音频域的失败模式命名和框架化,使其可以被系统性地讨论和解决。这与 TTS 领域的平行发现 (Channel Corp, 2026 的 "no verifiable reward for prosody") 形成跨领域互证,共同指向一个更深层的 insight: **reward 信号的维度必须匹配优化目标的维度,否则低维 reward 必然导致高维特征的坍缩**。

然而技术方案本身 (RLHF + PPO) 并不新颖——SpeechLM 领域已有 SpeechAlign (2024) 引入偏好学习的先例。Rubric-based generated reward model 是一个有趣的设计 (条件化 reward 评估),但论文对其实现细节披露不足。

**最值得警惕的是论文叙事与实证之间的张力**: 论文用大量篇幅渲染 RLVR 模型的"情感扁平"和 RLHF 模型的"沉浸式对话",但所有评估都是 speech-to-text benchmark 的客观分数。Audio MC 的改善可能仅反映多轮指令遵循和上下文记忆的提升,而非论文所强调的"韵律自然度"或"情感共鸣"。一个不生成语音的模型,其"对话体验"改善需要更审慎的验证。

## 可复用的 idea

1. **"Verifiable reward trap" 诊断框架**: 适用于任何将连续信号压缩为离散标签进行 RL 优化的场景——不限于音频,可推广到视频理解、多模态推理等。核心检查: 你的 reward 信号是否覆盖了用户实际关心的所有维度?

2. **Rubric-conditioned reward model**: 对有明确标准和无明确标准的样本使用不同评估模式,比固定单一 reward 更灵活。可借鉴到 TTS RL 中: 内容准确性用 rubric (WER 阈值),自然度用偏好比较。

3. **联合优化 vs 分阶段优化**: 当多个优化目标的梯度方向可能冲突时,联合训练优于串行——这个经验在 TTS RL (DiffRO+GRPO combined, Multi-Reward GRPO) 中也有佐证。

4. **Cold-start SFT 作为 RLHF 前置**: 不要让 RLHF 从零学会对话行为,先用 SFT 建立交互基线,让 RLHF 集中精力优化质量。

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> Issues: 3 (high: 0, medium: 0, low: 3)
> 详见 `_review/Step-Audio-R1.5-review.yml`

---

检索命中: [[SpeechLanguageModel]]✓, [[ProsodyModeling]]✓ | 过滤: [[AudioUnderstanding]](pending-review), [[DifferentiableRewardOptimization]](pending-review), [[SpokenDialogueEvaluation]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review) | 未命中但可能相关: 无
