---
type: paper
tier: deep
title: "Overcoming State Inertia in Full-Duplex Spoken Language Models via Activation Steering"
arxiv_id: "2606.11386"
source: "Sources/StateInertia-FD-SLM.pdf"
authors: [Cheng-Kuang Chang, Kai-Wei Chang, Alexander H. Liu, James Glass]
year: 2026
venue: "arXiv"
tags: [full-duplex, speech-LM, activation-steering, interruption-handling, interpretability, turn-taking, logit-lens]
concepts: ["[[Full-duplexSpokenDialogue]]", "[[Turn-takinginSpokenDialogue]]", "[[SpeechLanguageModel]]", "[[StreamingSpokenDialogue]]", "[[SpokenDialogueEvaluation]]"]
models: ["[[论文笔记/PersonaPlex|PersonaPlex]]", "[[论文笔记/Moshi|Moshi]]", "[[论文笔记/Raon-Speech|Raon-SpeechChat]]"]
tasks: []
datasets: ["Zero-Buffer Benchmark (ZBB)", "Full-Duplex Bench (FDB)"]
kb_context_sources: 5
status: draft
created: 2026-06-12
updated: 2026-06-12
---

## KB 背景

> [!info] KB 背景 (基于 5 个实体页: [[Full-duplexSpokenDialogue]], [[Turn-takinginSpokenDialogue]], [[SpeechLanguageModel]], [[StreamingSpokenDialogue]], [[SpokenDialogueEvaluation]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Full-duplexSpokenDialogue]]✓, [[Turn-takinginSpokenDialogue]]✓, [[SpeechLanguageModel]]✓, [[StreamingSpokenDialogue]]✓, [[SpokenDialogueEvaluation]]✓ | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: Full-duplex SLM (FD-SLM) 是 SpeechLM 的前沿交互范式,允许模型同时听和说。KB 中已覆盖 dGSLM (首个全双工)、Moshi (RQ-Transformer 全双工)、LSLM (边说边听)、PersonaPlex (voice+role control)、Raon-SpeechChat (SIL/BOW/BC 三状态建模) 等系统。Turn-taking 的已有技术路线包括: VAD-only (高误判) → 多模态检测 → 端到端 irq/n-irq markers → chunk-level state prediction → SIL/BOW/BC 三状态建模。

**已有认知**: KB 中 Turn-taking 页面已记录"打断处理"的多种实现方式 (LSLM 的 IRQ token, Mini-Omni2 的 irq/n-irq, Freeze-Omni 的 State 0/1/2, Raon 的 SIL/BOW/BC), 但所有方法关注的是"模型何时应停止说话并开始听",**没有任何工作分析过 FD-SLM 在被打断时内部隐藏表征的转换动态**。SpokenDialogueEvaluation 页面记录了 FDB 等 benchmark,但指出"Interaction Capability 的评估严重不足"。

**创新判断**: 本文是首个从 mechanistic interpretability 角度分析 FD-SLM 内部"听-说"协调机制的工作。State inertia 是一个全新概念 — KB 中无任何页面涉及。将 activation steering 从 text LLM 迁移到 FD-SLM 也是首次。ZBB benchmark 填补了 SpokenDialogueEvaluation 中"即时打断理解"的评估空白。

## 速查

> [!summary] 速查
> - **一句话**: 发现 FD-SLM 在用户打断时存在 state inertia (延迟切换到感知状态),通过 training-free 的 activation steering 注入 perception vector 将打断理解正确率从 28% 提升到 45%
> - **路线**: logit lens 分析 → 定义 generation/perception affinity → 识别 state inertia → 构造 perception vector (mean diff) → 打断时注入 steering → ZBB 评估
> - **指标**: PersonaPlex 上 Correctness 0.28→0.45 (+81% recovery), IWOR 0.40→0.72 (+94% recovery); Moshi 上 Correctness 0.22→0.34 (+57%), IWOR 0.29→0.64 (+92%) [Table 2]
> - **可借鉴**: 用 logit lens 分析多流模型的 stream-specific predictive focus; 用 generation/perception affinity 量化内部状态; perception vector 构造方法 (mean diff of contrastive timesteps) 可推广到其他多状态切换场景
> - **局限**: 仅在 3 个开源 FD-SLM 上验证; 依赖 onset detector 检测打断时刻; perception vector 从单独的 turn-by-turn 数据构建,未验证跨领域泛化; Raon 上绝对正确率仍很低 (0.17)

## 核心问题

FD-SLM 如何在内部协调同时进行的听和说?当用户突然打断时,模型是否能立即切换到"感知模式"?如果不能,如何在不微调的情况下改善?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文不提出新模型架构,而是一项分析性工作 (analysis + intervention)。研究对象是现有的 FD-SLM (PersonaPlex, Moshi, Raon-SpeechChat),工作流程分为三个阶段:

1. **内部机制分析**: 用 logit lens 分析 FD-SLM 隐藏表征的 stream-specific 预测行为 [§3.2]
2. **问题识别**: 定义 generation/perception affinity,发现 state inertia 现象 [§3.3-3.4]
3. **干预方案**: 构造 perception vector 进行 activation steering [§5]

### 关键设计选择

#### Finding 1: Stream-specific Predictive Focus

用 logit lens 将中间层隐藏表征投影到词表空间,发现 FD-SLM 在听时中间层预测的是**用户流的下一个 token**(而非模型自身的输出),说时预测的是**模型输出流的 token** [§3.2, Table 1]。

例如: 用户说 "explain their pros and cons" 时,中间层 decode 出 "why, how", "own, pro", "and", "con, cons" — 这些追踪的是用户语流的延续,而非模型的沉默 token [Table 1]。

[论文原文] 这表明 LM backbone 不只是静态地向前传递文本前缀信息,而是在中间层建立了与当前活跃语流对齐的 sparse directions [§3.2]。

#### Finding 2: Generative vs Perceptive State

定义两个 affinity score 来量化内部状态 [§3.3]:

- **Generation Affinity** $S_{gen}(t)$: 隐藏表征 $h^{(t)}$ 对模型输出 token 的投影概率均值 [Eq. 4]
- **Perception Affinity** $S_{perc}(t)$: 隐藏表征对下一个用户输入 audio token 的投影概率 [Eq. 5]

分析 100 个 turn-by-turn 对话,发现: 用户说话时 $S_{perc}$ 高 (perceptive state), 模型说话时 $S_{gen}$ 高 (generative state), 两者**动态切换**而非固定 [Fig. 2, 3]。

[agent 解读] 最后几层表现不同: $S_{perc}$ 即使在用户说话时也低,因为最终层必须输出模型侧 token (通常是 silence),这符合 autoregressive 解码的机制约束。

#### Finding 3: State Inertia

对比"有打断"和"无打断"条件:
- 无打断时: 用户开始说话后模型立即进入 perceptive state [Fig. 4]
- 有打断时: 模型被打断后仍维持 generative state **约 7-8 个 timestep (~0.6 秒)**,才过渡到 perceptive state [Fig. 5]

[论文原文] 作者将这种延迟的内部状态转换命名为 **state inertia**,类比人类的 speech-induced suppression (说话时听觉皮层活动被抑制) [§3.4, ref 28, 20]。

[agent 解读] Generation affinity 在打断后同样表现出延迟下降 (约 20 timesteps / ~2 秒才恢复, Fig. 12), 说明 inertia 是双向的: 既难以进入 perceptive state,也难以退出 generative state。

#### Perception Vector 构造

从对比数据构造 steering vector [§5, Eq. 6]:

1. 使用 turn-by-turn 数据集 (与 ZBB 评估集不重叠) 计算每个 timestep 的 $S_{gen}$ 和 $S_{perc}$
2. 在 12-24 层平均 affinity,用阈值 $\Theta_{gen}$, $\Theta_{perc}$ 将 timestep 分为 generation-dominant ($T_{gen}$) 和 perception-dominant ($T_{perc}$) [Table 3]
3. Perception vector: $\mu_{g \to p} = \text{mean}(h_{perc}) - \text{mean}(h_{gen})$ [Eq. 6]

[论文原文] PCA 分析确认 generation-dominant 和 perception-dominant timestep 在隐藏空间中形成**明确分离的 clusters**,验证了 perception vector 是有意义的方向而非噪声 [§D, Fig. 13]。

#### Steering Schedule

推理时,在打断 onset 检测到后,对后续 $\Delta T_{steer}$ 个 timestep 线性衰减地注入 perception vector [§5, Eq. 7]:

$$\tilde{h}^{(t)} = h^{(t)} + \alpha \left(1 - \frac{t - t_{int}}{\Delta T_{steer}}\right) \mu_{g \to p}$$

关键超参数 [Table 3]:
- PersonaPlex/Moshi: layer 23, $\alpha=5.5$, $\Delta T_{steer}=3$
- Raon: layer 26, $\alpha=1.2$, $\Delta T_{steer}=3$

[论文原文] $\Delta T_{steer}=3$ 是最优,更长的 steering span 反而降低性能,说明 steering 在打断 onset 的短暂窗口内最有效 [§F, Fig. 19]。

### 训练策略

本文**不涉及任何训练或微调**。整个方法是 training-free, inference-time intervention:
- Perception vector 从 100 个 turn-by-turn 对话样本的 affinity 统计中构造
- Onset 检测使用 energy-based detector
- Steering 仅在检测到打断后的 3 个 timestep 内施加

## 实验

### Zero-Buffer Benchmark (ZBB)

专门设计的 benchmark,将关键语义词放在打断话语的第一个词:
- 100 个 zero-buffer query (50 subjects × 2 correct/incorrect descriptions)
- 模板: `<Subject>, <Description>, <Confirmation Request>` (如 "Submarine flies in the clouds, right?")
- 用 Dia2-2B TTS 合成; nvidia/parakeet-tdt-0.6b-v2 做 ASR; GPT-4.1-mini 做评判 [§4, §A.4]
- 评估指标: **Correctness** (回答是否正确) + **IWOR** (Initial Word Occurrence Rate, 是否识别第一个语义词)

### 主实验结果

| 指标 | 本文 (Steer) | Baseline (Interrupt) | No Interrupt | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Correctness | 0.45±0.05 | 0.28±0.04 | 0.49±0.05 | ZBB (PersonaPlex) | [Table 2] |
| IWOR | 0.72±0.04 | 0.40±0.05 | 0.74±0.04 | ZBB (PersonaPlex) | [Table 2] |
| Correctness | 0.34±0.05 | 0.22±0.04 | 0.43±0.05 | ZBB (Moshi) | [Table 2] |
| IWOR | 0.64±0.05 | 0.29±0.05 | 0.67±0.05 | ZBB (Moshi) | [Table 2] |
| Correctness | 0.17±0.03 | 0.03±0.02 | 0.10±0.03 | ZBB (Raon) | [Table 2] |
| IWOR | 0.24±0.04 | 0.16±0.04 | 0.29±0.05 | ZBB (Raon) | [Table 2] |
| FDB Score | 3.41±0.08 | 3.34±0.08 | — | FDB (PersonaPlex) | [Table 5] |
| FDB Score | 3.36±0.08 | 3.45±0.08 | — | FDB (Moshi) | [Table 5] |

**Recovery 比例** (Interrupt+Steer 恢复了 Interrupt→No Interrupt 差距的多少):
- PersonaPlex: Correctness 81%, IWOR 94% [Table 2]
- Moshi: Correctness 57%, IWOR 92% [Table 2]
- Raon: Correctness 200% (超过 no-interrupt baseline), IWOR 62% [Table 2]

### Ablation: Steering 超参数

- **Steering layer**: Layer 23 在 PersonaPlex 上最优 [Fig. 18]
- **Steering strength α**: α=5.5 最优 (PersonaPlex); α 过大会降低性能 [Fig. 18]
- **Steering span**: $\Delta T_{steer}=3$ 最优; 更长 span 反而降低性能 [Fig. 19]
- **FDB 影响**: Steering 不降低一般全双工对话性能 (FDB score 在误差范围内不变) [Table 5]

### Attention Recovery 分析

Steering 后,模型对打断早期 timestep 的 attention weight 明显恢复,接近 no-interruption 水平 [§G, Fig. 20]。

### False Trigger 鲁棒性

在模型正常回答时随机注入 perception vector,response quality 随误触频率增加而逐渐下降 (非断崖式),说明偶尔误触的影响可控 [§I, Fig. 21]。

## 局限性

1. **单一模型规模**: 仅在 3 个开源 FD-SLM 上验证,未涉及更大规模模型 [§7]
2. **Onset 检测依赖**: 使用 energy-based detector,在嘈杂/多说话人场景下可能不够鲁棒 [§7]
3. **Logit lens 局限**: Affinity score 是诊断性近似,对单个样本可能有噪声 [§7]
4. **Raon 绝对性能低**: 即使 steering 后 Raon 的 Correctness 仅 0.17,说明 state inertia 不是 Raon 性能差的唯一原因 [agent 解读]
5. **Perception vector 泛化**: 从 turn-by-turn 数据构建,是否泛化到任意对话场景未验证 [agent 解读]
6. **不同 FD-SLM 需要不同超参数**: 三个模型的 layer, α, 阈值均不同 [Table 3], 新模型需要重新调参 [agent 解读]

## 点评

**核心贡献**: 本文的核心价值在于**发现**而非方案。State inertia 是一个有意义的新概念,揭示了 FD-SLM 在处理打断时的内部机制短板。将 logit lens + contrastive representation engineering 从 text LLM 迁移到多流 speech LM 是方法论层面的贡献。

**ZBB 设计精巧**: 将关键语义词放在打断的第一个词,直接测试 state inertia 的下游影响。这比 FDB 等关注宏观对话质量的 benchmark 更具诊断性 — FDB 查询通常有 filler 作为缓冲,掩盖了 state inertia 的影响 [§H]。

**方案简洁但局限明显**: Training-free activation steering 的简洁性是优势 (无需微调、推理开销小),但每个模型需要独立调参、依赖 onset detector、且 Raon 上效果有限,距离实际部署有差距。

**与 KB 中已有工作的关系**: KB 中 Turn-taking 页面记录的方法 (IRQ token, SIL/BOW/BC 等) 都是"训练时"解决方案 — 在模型训练阶段学习何时切换状态。本文提供了互补的"推理时"视角: 即使训练好的模型仍存在 state inertia,可以通过推理时干预缓解。这为两类方法的结合指出了方向。

## 可复用的 idea

1. **Logit lens 分析多流模型**: 将 logit lens 从 text LLM 推广到 multi-stream SLM,用于分析各流的 predictive focus 如何随层深变化。可直接应用于任何同时处理多个 token stream 的模型。

2. **Generation/Perception Affinity 量化**: 用隐藏表征对各流 token 的投影概率定义 affinity score,是一种通用的"内部状态监控"方法。可推广到: 多任务模型中监控各任务的 attention 偏好;多模态模型中监控模态切换时机。

3. **Contrastive state vector for steering**: 从对比条件 (generative vs perceptive timestep) 构造 steering vector 的方法论可推广到任何需要切换内部模式的场景: 如 RAG 模型中"检索模式 vs 生成模式"的切换、多轮对话中"遵循上下文 vs 忽略上下文"的平衡。

4. **ZBB 式诊断 benchmark 设计**: 将关键信息放在最敏感位置 (打断 onset) 来测试模型的即时处理能力。这种"设计 adversarial 但自然的评估场景"的思路可推广。

## 审阅

(待独立审阅 agent 填充)
