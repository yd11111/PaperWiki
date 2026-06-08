---
type: paper
tier: deep
title: "CORD: Bridging the Audio-Text Reasoning Gap via Weighted On-policy Cross-modal Distillation"
arxiv_id: "2601.16547"
source: "Sources/BridgingAudio-TextGap.pdf"
authors: [Jing Hu, Danxiang Zhu, Xianlong Luo, Dan Zhang, Shuwei He, Yishu Lei, Haitao Zheng, Shikun Feng, Jingzhou He, Hua Wu, Yu Sun, Haifeng Wang]
year: 2026
venue: "arXiv preprint"
tags: [speech-LM, cross-modal-alignment, knowledge-distillation, GRPO, on-policy, LALM, audio-text-gap, reasoning]
concepts: ["[[SpeechLanguageModel]]", "[[AudioUnderstanding]]", "[[ModalityAdaptationforSpeechLLM]]", "[[Audio-LanguagePretraining]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[DifferentiableRewardOptimization]]"]
models: ["Qwen2-Audio-7B-Instruct", "Step-Audio2-mini"]
tasks: []
datasets: ["NuminaMath", "MMSU", "OpenBookQA", "GSM8K", "MMAU", "VoiceBench"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeechLanguageModel]], [[AudioUnderstanding]], [[ModalityAdaptationforSpeechLLM]], [[Audio-LanguagePretraining]], [[Speech-LLMIntegrationTaxonomy]], [[DifferentiableRewardOptimization]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[AudioUnderstanding]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review), [[Audio-LanguagePretraining]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review), [[DifferentiableRewardOptimization]](pending-review) | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: CORD 针对的是 latent-representation-based Speech-LLM 集成路线中的核心问题 -- 音频模态的推理性能劣化。[[Speech-LLMIntegrationTaxonomy]] 指出该路线使用语音编码器+模态适配器+LLM 的架构,而 [[ModalityAdaptationforSpeechLLM]] 详述了连接桥梁的技术方案。但这些工作聚焦于表征层面的对齐(如何投影到 LLM 空间),而 CORD 发现即使表征对齐了,音频条件下的推理行为仍然劣于文本条件,提出在推理轨迹层面做对齐。

**已有认知**: [[SpeechLanguageModel]] 记录了 speech LLM 的演进从 GSLM 到 Moshi/VITA 等 omni-model。[[AudioUnderstanding]] 指出 SpeechLM 相对 TextLM 的独特优势是保留副语言信息,但实际评估中多数 benchmark 要求文本输出,理解能力的评估存在瓶颈。[[DifferentiableRewardOptimization]] 记录了 GRPO 等 RL 方法在 TTS 后训练中的应用,但此前主要用于生成质量优化而非跨模态对齐。

**创新判断**: CORD 的核心创新在于: (1) 把 text modality 作为内部 teacher 做 on-policy self-distillation,不需要外部教师模型; (2) 发现 token 级 KL 散度分布高度偏斜,据此设计 importance-aware 加权; (3) 首次将 GRPO 用于跨模态推理轨迹对齐。相比 [[Audio-LanguagePretraining]] 中的对比学习路线和 [[ModalityAdaptationforSpeechLLM]] 中的适配器方案,CORD 在训练范式层面提出了新方向 -- 不改架构,只改训练目标。

## 速查

> [!summary] 速查
> - **一句话**: 用模型自身的文本模态作为 teacher,通过 on-policy 跨模态蒸馏(token 级加权 KL + 序列级 GRPO)弥合 LALM 中音频-文本推理差距
> - **路线**: 音频输入 → LALM rollout → token 级 reverse KL (Top-K 重要性加权 + 位置衰减) + 序列级 judge-based GRPO → 对齐到文本条件推理行为
> - **指标**: 音频-文本 gap 平均减少 41.6% (Qwen2-Audio) / 44.8% (Step-Audio2-mini); MMSU 36.04→38.06 [Table 1]; 仅用 80K 训练样本 [§4.1]
> - **可借鉴**: (1) token 级 KL 偏斜分析方法可迁移到任何跨模态蒸馏场景; (2) Top-K 重要性加权避免梯度稀释; (3) on-policy distillation + GRPO 的组合解决 GRPO 单独使用时的 collapse 问题 [Table 3]
> - **局限**: 仅在 Baidu 内部 judge model 上验证(未开源); 训练数据为数学单一领域; 未在真正 text-insufficient 任务(如情感识别)上验证效果

## 核心问题

LALM (Large Audio Language Models) 构建在文本 LLM 之上,通过音频编码器 + 模态适配模块接入音频输入。核心矛盾: 尽管接收语义等价的输入,音频条件下的推理性能显著低于文本条件 [§1]。这个差距在数据受限场景下尤为严重 [§1, para 2]。

现有方法的三个根本缺陷 [§1]:
1. **有限可扩展性**: SFT 依赖大规模标注语音数据,昂贵且难以跨域扩展
2. **Off-policy 分布失配**: 传统蒸馏沿 teacher 的文本生成轨迹提供监督,与 student 实际音频推理状态不一致
3. **均匀 token 级监督**: 常规 KL 蒸馏对所有 token 等权,无法聚焦于驱动跨模态错位的关键 token

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CORD 在单个 LALM 内部执行 on-policy 跨模态 self-distillation [§3, Fig 1]:

```
给定语义等价的 (音频输入 x_a, 文本输入 x_t):

1. Policy Model: 从 p_θ(·|x_a) 采样 on-policy 轨迹 y
2. Teacher Model: 冻结参数,计算 p_θ(·|y_{<t}, x_t) -- 文本条件分布
3. Token 级对齐: 加权 reverse KL(y_{<t}, x_a ∥ y_{<t}, x_t)
4. 序列级对齐: Judge model 评估 y 与文本条件输出的语义一致性 → GRPO
```

关键设计: policy model 和 teacher model 是同一个模型的两次前向传播(不同输入模态),无需外部教师 [论文原文, §3.2]。

### 关键设计选择

**为什么用 Reverse KL 而非 Forward KL?**

Reverse KL = KL(p_θ(·|x_a) ∥ p_θ(·|x_t)) 对文本条件分布中的高概率 token 施加更强约束,鼓励音频条件 policy 恢复文本模态的关键推理决策 [论文原文, §3.2, Eq. 2]。Forward KL 则沿 teacher 轨迹对齐,是 off-policy 的。

**为什么需要 Top-K 重要性加权?**

论文对 MMSU benchmark 的 token 级 KL 分布分析发现 [§3.3, Fig 3]:
- 分布高度偏斜: 80th percentile 的 KL 值仅 0.23,绝大多数 token 已跨模态对齐
- 高 KL token 集中在语义关键推理词(如 "Therefore", "answer")和选项(A, B) [Fig 3(a)]
- 高 KL 状态集中在解码早期 (r = -0.139 相关系数) [Fig 3, bottom-left]
- 均匀平均导致高 KL token 的梯度被大量低 KL token 稀释

[agent 解读] 这个发现本质上说明:音频-文本模态差异不是均匀分布在整个序列上的,而是集中在少数"决策点"。这与 NLP 中发现 LLM 推理的 "anchor token" 现象一致。

**Top-K + 位置衰减的具体实现 [§3.4, Eq. 4-6]:**

```
最终 token 权重 w_t = w_t^KL * w_t^pos

w_t^KL = α (if t ∈ Top-K by D_t) else 1    # α=2, K=20
w_t^pos = β - (β-1)*(t-1)/(T-1)             # β=2, 线性衰减
```

[论文原文] 硬选择(Top-K)防止低散度 token 主导优化;位置衰减强调早期对齐,因为早期语义偏差会级联传播到后续解码 [§3.4]。

**为什么需要序列级 GRPO?**

[论文原文, §3.5] Token 级对齐修正局部偏差但不显式约束全局推理行为 -- 局部对齐的 token 分布仍可能导致全局不一致或错误的最终答案。

GRPO 序列级对齐 [§3.5, Eq. 8-10]:
1. 对每个 x_a 采样 N=4 条 on-policy 轨迹
2. Judge model J(y, y_hat) 给出二值 reward: 音频条件答案是否与文本条件答案语义一致
3. GRPO 计算组内相对优势 A^(i) = r_i - mean(r),优化 policy 增加高 reward 轨迹的概率

**GRPO 为什么需要 OPD 配合?**

[论文原文, Table 3] 消融实验发现: 单独 GRPO 在 500 步时有增益,但 1000 步后严重崩溃(GSM8K: 35.59 → 19.89,低于 baseline)。加入 OPD(on-policy distillation)作为正则化后,可稳定训练至 3000 步(GSM8K: 36.12)。[agent 解读] OPD 的 token 级蒸馏锚定了模型在文本模态上的行为,防止 RL 信号导致模型偏离有效推理路径。

### 训练策略

- 数据: 80K NuminaMath 数学题,用 Kokoro TTS 合成语音 [§4.1]
- 优化器: AdamW, lr = 3e-5
- CORD 和 Forward-KL: 每 prompt 1 条 rollout,温度 1.0
- GRPO: 每 prompt 4 条 rollout,温度 1.5 (鼓励多样性)
- 超参: α = β = 2 (via sensitivity analysis [Fig 4])
- Judge model: Baidu 内部蒸馏的评估模型,自评准确率 > 99% [§4.1]

## 实验

| 指标 | 本文 (CORD) | Baseline (Base Model) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| MMSU Acc | 38.06% | 36.04% (Qwen2-Audio) | MMSU (VoiceBench) | [Table 1] |
| OBQA Acc | 52.77% | 51.20% (Qwen2-Audio) | OpenBookQA (VoiceBench) | [Table 1] |
| GSM8K Acc | 36.20% | 20.73% (Qwen2-Audio) | GSM8K | [Table 1] |
| 平均 gap 减少 | 41.6% | - | Qwen2-Audio, 3 benchmarks | [Table 1] |
| MMSU Acc | 57.63% | 52.31% (Step-Audio2-mini) | MMSU (VoiceBench) | [Table 1] |
| OBQA Acc | 77.74% | 72.30% (Step-Audio2-mini) | OpenBookQA (VoiceBench) | [Table 1] |
| GSM8K Acc | 47.56% | 43.75% (Step-Audio2-mini) | GSM8K | [Table 1] |
| 平均 gap 减少 | 44.8% | - | Step-Audio2-mini, 3 benchmarks | [Table 1] |
| MMAU Music | 60.18% | 58.98% (Qwen2-Audio base) | MMAU | [Table 2] |
| MMAU Sound | 64.44% | 64.74% (Qwen2-Audio base) | MMAU | [Table 2] |
| MMAU Speech | 55.42% | 58.73% (Qwen2-Audio base) | MMAU | [Table 2] |

**关键对比**: CORD vs Forward-KL vs SFT [Table 1]

- Forward-KL: 平均 gap 减少 28.5% (Qwen2-Audio) / 10.5% (Step-Audio2-mini)
- SFT: 不稳定,在 OBQA 上 gap 反而增大 (+1.71 on Qwen2-Audio)
- CORD: 一致性最优,在所有 benchmark 上 gap 减少最大

**辅助能力保持 [Table 2]**: Forward-KL 导致 MMAU Music -2.99, Sound -3.04 的明显退化; CORD 在 Music 上反而微升 (+1.20),Sound 持平 (-0.30),表明 on-policy 对齐有效减轻了辅助音频能力的灾难性遗忘。

**消融实验 [Table 3]**:
- GRPO only (500 steps): 有提升
- GRPO only (1000 steps): 崩溃 (OBQA 36.48, GSM8K 19.89)
- GRPO + OPD: 稳定至 3000 步
- GRPO + OPD + Weight (Full CORD): 最优 (平均 +6.35 over base)

**超参敏感性 [Fig 4]**: α=β=2.0 达到最佳平衡; 1.0 退化为均匀 KL; 2.5 过度集中梯度导致性能下降。

## 局限性

1. **Judge model 不透明**: 使用 Baidu 内部蒸馏的 judge model,自评 > 99% 但未公开、未在第三方数据上验证 [§4.1]。GRPO 的 reward 质量完全依赖此 judge,可复现性存疑。
2. **单一训练领域**: 80K NuminaMath 为数学单域。虽然论文声称跨域迁移 [§4.2],但 MMSU 和 OBQA 提升幅度有限(+2.02 和 +1.57 on Qwen2-Audio),而 GSM8K 的 +15.47 可能部分归因于域内效应。
3. **Text-insufficient 任务未验证**: 论文声称方法"bridge the audio-text gap",但 MMAU 结果表明 Speech 维度下降 3.31%(58.73→55.42) [Table 2],说明 CORD 的对齐实际上是把音频推理向文本推理对齐 -- 在需要利用非文本信息的任务上可能反而有害。
4. **仅两个 backbone**: 仅在 Qwen2-Audio 和 Step-Audio2-mini 上验证,未覆盖 Gemini、Phi-4-MM 等架构差异更大的系统。
5. **TTS 合成训练数据**: 用 Kokoro 合成的语音可能与真实语音分布有差异,模型是否能迁移到自然语音场景未验证 [§4.1]。

## 点评

CORD 的核心洞察 -- "音频-文本推理差距集中在少数关键 token 而非均匀分布" -- 是一个扎实的实证发现 [Fig 3],比简单的 KL 蒸馏更有针对性。将 on-policy distillation 与 GRPO 结合的思路也有机制合理性: OPD 提供稳定的 token 级锚定,GRPO 提供序列级全局信号。消融实验 [Table 3] 清晰验证了两者的协同效应。

但论文的实际改进幅度需要谨慎解读。"gap 减少 41.6%" 的表述容易被高估 -- 在绝对值上,MMSU 提升仅 2.02%(36.04→38.06),OBQA 提升 1.57%。gap 减少的百分比大是因为 gap 本身定义为相对值。此外,最大提升发生在 GSM8K (+15.47%),这是与训练域最接近的 benchmark。

方法论上的一个担忧是 judge model 的角色: 如果 judge 本身有偏差(例如偏好特定格式的答案),GRPO 可能会放大这种偏差而非真正弥合模态差距。论文未对 judge model 做 ablation 或替换实验。

与知识库中 [[DifferentiableRewardOptimization]] 的 GRPO 应用对比: DiffRO 系列在 TTS 中用 GRPO 优化生成质量,CORD 在音频理解中用 GRPO 对齐推理轨迹 -- 两者展示了 GRPO 作为 post-training 工具的通用性。

## 可复用的 idea

1. **Token 级 KL 偏斜分析方法**: 对任意跨模态蒸馏任务,先绘制 token 级 KL 分布,识别高散度 token 的语义和位置特征,再设计加权策略 -- 这个分析流程本身是可迁移的 [§3.3, Fig 3]。

2. **OPD + GRPO 协同防崩溃**: GRPO 单独使用容易 collapse [Table 3],加入 on-policy distillation 作为正则化可稳定训练。这一经验对任何将 RL 用于 LLM post-training 的场景都有参考价值。

3. **内部 teacher 自蒸馏**: 利用同一模型不同模态的输出作为 teacher/student,无需外部教师。可迁移到视觉-语言模型或其他多模态 LLM 的跨模态对齐 [§3.2]。

4. **Top-K 硬选择 + 线性位置衰减**: 简单的加权方案(两个超参 α, β),避免了 soft attention 等复杂方法,且超参不敏感(2.0 附近均可) [Fig 4]。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节详述 WHY,速查卡片可借鉴具体 |
> | 可信赖 | pass | 主要数字有出处,训练超参出处可补全 |
> | 可区分 | pass | [论文原文]/[agent 解读] 覆盖率 > 80% |
> | 可定位 | pass | KB 背景谱系定位清晰,frontmatter 完整 |
> | 不污染 | pass | 无新建概念页,引用关联合理 |
> 
> Issues: 4 (high: 0, medium: 2, low: 2)
> 详见 `_review/BridgingAudio-TextGap-review.yml`
