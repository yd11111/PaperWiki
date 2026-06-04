---
type: paper
tier: deep
title: "Fine-Tuning Text-to-Speech Diffusion Models Using Reinforcement Learning with Human Feedback"
arxiv_id: "2508.03123"
source: "Sources/2508.03123.pdf"
authors: [Jingyi Chen, Ju Seung Byun, Micha Elsner, Pichao Wang, Andrew Perrault]
year: 2025
venue: "Interspeech 2025 (arXiv preprint)"
tags: [TTS, diffusion, RLHF, reinforcement-learning, fine-tuning, waveform-generation, non-autoregressive, reward-model]
concepts: ["[[Diffusion Model]]", "[[Diffusion-based TTS]]", "[[Diffusion-based Vocoder]]", "[[Non-autoregressive TTS]]", "[[TTS Evaluation]]", "[[Differentiable Reward Optimization]]"]
models: ["[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 6 个待确认实体页: [[Diffusion-based TTS]], [[Diffusion Model]], [[Differentiable Reward Optimization]], [[TTS Evaluation]], [[Diffusion-based Vocoder]], [[Non-autoregressive TTS]])
> 自动生成,不保证完整覆盖所有相关知识。所有引用页面均为 pending-review 状态,仅供参考 [待确认]。

**谱系定位**: 本文将 RLHF 引入 diffusion-based 端到端 TTS (WaveGrad 2) 的微调,位于 **Diffusion-based TTS + RL post-training** 的交叉点。这一交叉点在知识库中已有丰富积累: [[Differentiable Reward Optimization]] 页面记录了 TTS RL 后训练的完整演进线 (Seed-TTS REINFORCE → SpeechAlign DPO → DiffRO token-level → Multi-Reward GRPO → Component-level GRPO),但这些工作的 RL 对象均为 **LLM-based 或 flow-matching TTS**,而非直接在 **DDPM 去噪过程** 上做 RL。本文是知识库中首篇将 RLHF 直接应用于 DDPM 去噪 MDP 的工作。

**已有认知**:
- [[Diffusion-based TTS]] 记录了 WaveGrad 2 作为端到端 diffusion TTS 的地位: 音素→波形,集成 Tacotron 2 encoder + 非注意力 duration [§3.4]
- [[Diffusion-based Vocoder]] 记录了 WaveGrad 系列的起源和高效化路线 (BDDM 7步, PriorGrad 自适应先验)
- [[TTS Evaluation]] 详细记录了 UTMOS/NISQA 等 predicted MOS 的局限: 领域不匹配、缺乏 uncertainty estimation、跨域泛化差
- [[Differentiable Reward Optimization]] 演进线显示: 2024-2025 年 TTS RL 后训练已从 audio-level REINFORCE 进化到 token-level DiffRO/FPO/GRPO,而本文回到 **audio-level + denoising trajectory-level** 的 RL 方案

**创新判断**: 本文的独特性在于将 text-to-image diffusion RL 方法 (RWR/DDPO/DPOK/KLinR) 系统地迁移到 TTS diffusion 并诊断其失败原因,然后提出 DLPO 这一 diffusion loss 正则化的 RLHF 方案。但对比知识库中同期更先进的 RL-for-TTS 方法 (DiffRO/Multi-Reward GRPO/FPO/F5R-TTS),本文的实验规模较小 (单说话人 LJSpeech, WaveGrad 2R 复现模型),且 UTMOS 作为 reward model 的 reward hacking 风险已被 [[TTS Evaluation]] 文献充分讨论。

> 检索命中: [[Diffusion-based TTS]], [[Diffusion Model]], [[Differentiable Reward Optimization]], [[TTS Evaluation]], [[Diffusion-based Vocoder]], [[Non-autoregressive TTS]] | 过滤: 全部 pending-review | 未命中但可能相关: [[Prosody Modeling]](confirmed)

## 速查

> [!summary] 速查
> - **一句话**: 提出 DLPO,将 diffusion 原始训练 loss 作为正则项嵌入 RLHF reward function,解决 text-to-image RL 方法迁移到 TTS diffusion 时的语音质量退化问题
> - **路线**: 文本 → WaveGrad 2R (DDPM 去噪 T 步) → 波形 x_0 → UTMOS 评分 reward → 梯度更新 (reward + diffusion loss penalty)
> - **指标**: UTMOS 3.65 / NISQA 4.02 / WER 1.2% (LJSpeech 200 test, 优于 base WaveGrad 2R UTMOS 2.90 / NISQA 3.74 / WER 1.5%); 人类评估 67% 偏好 DLPO [§2.3]
> - **可借鉴**: 在 RL 微调 diffusion 模型时,将原始训练 loss 嵌入 reward function 作为正则项 — 这是一种通用的 "task-specific regularization" 策略,可防止模型偏离预训练分布
> - **局限**: 仅在单说话人 LJSpeech + WaveGrad 2R 复现模型上验证; UTMOS 作为 reward 存在 reward hacking 风险; 未与同期 token-level RL 方法 (DiffRO/GRPO) 对比; 代码未正式开源 (anonymous link)

## 核心问题

本文要解决的核心问题: **如何用 RLHF 改善 diffusion-based TTS 模型的语音自然度,同时避免语音质量退化?**

直接将 text-to-image 领域的 diffusion RL 方法 (RWR, DDPO, DPOK, KLinR) 应用到 TTS diffusion 模型会失败或效果有限 [§1]:
1. TTS 要求更高的 **时间一致性** (temporal coherence) 和 **声学精度** — 音频波形的时域结构比静态图像复杂
2. TTS 的评价指标 (naturalness MOS, WER) 比图像质量指标更难优化
3. 现有 KL 正则化方法 (DPOK, KLinR) 虽能稳定训练但改进幅度有限 — KL 约束可能抑制了 task-specific 创新空间

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DLPO 的整体框架 [Fig 1, §2]:

```
Pre-trained WaveGrad 2R (P_pre)
        ↓ 生成波形 x_0
UTMOS Reward Model → r(x_0, c)  (自然度评分)
        ↓ 
Reward + Diffusion Loss Penalty → 更新 P_θ
        ↓ 重复
Fine-tuned WaveGrad 2R (P_θ)
```

关键: 将去噪过程建模为 T 步有限 horizon MDP [§2.1.2]:
- **State**: s_t = (c, x_{T-t}),其中 c 是文本 prompt, x_{T-t} 是当前去噪步的中间结果
- **Action**: a_t = x_{T-t-1},即下一步的去噪输出
- **Reward**: 仅在最终步 (t=T-1) 给出 reward r(x_0, c) = UTMOS(x_0),中间步 reward = 0
- **Policy**: pi_theta(a_t|s_t) = p_theta(x_{T-t-1}|x_{T-t}, c)

RL 目标为最大化生成语音的 UTMOS 分数 [Eq. 4]:
```
J_DDRL(θ) = E_{c~p(c)} E_{x_0~p_θ(x_0|c)} [r(x_0, c)]
```

### 关键设计选择

**为什么不用 KL 正则化?** [论文原文] DPOK 和 KLinR 使用 KL(p_θ || p_pre) 作为正则项防止过拟合,但 KL 正则化的改进在 TTS 上很有限 (DPOK UTMOS 3.18 vs base 2.90) [§2.3.1]。作者认为原因是: TTS 在时域操作,需要时间一致性和声学连贯性,KL 正则化可能在限制偏离的同时也限制了 task-specific 的优化空间 [论文原文]。

**为什么用 diffusion loss 做正则?** [论文原文] 受 Ouyang et al. (2022) 在 NLP 中混合预训练梯度与 RL 梯度的启发 [§2.2],作者假设将 diffusion model 原始训练 loss 加入目标函数可以 (1) 保持模型固有的生成能力; (2) 防止过拟合和偏移 [§2.2]。[agent 解读] Diffusion loss 直接作用于每个去噪步的噪声预测,相比 KL 散度这种 distribution-level 约束,它是 **function-level** 约束 — 确保模型在每一步的噪声预测仍然合理,这对维护时域连贯性更为直接。

**DLPO 的目标函数** [Eq. 5]:
```
E_{c~p(c)} E_{p_θ(x_{0:T}|c)} [-α·r(x_0, c) - β·||ε̃(x_t, t) - ε_θ(x_t, c, t)||²]
```
其中:
- α: reward 权重
- β: diffusion loss 权重
- r(x_0, c): UTMOS reward
- ||ε̃ - ε_θ||²: 标准 DDPM 噪声预测 loss

[agent 解读] 这个设计的关键洞察是: diffusion loss 不仅是正则项,它还与 TTS 的训练流程**结构对齐** (structurally aligned) — 它惩罚的是去噪过程中的每一步偏离,而非仅在序列层面约束分布差异。这使得 DLPO 比 KL 正则化能更精确地保持模型的时域生成能力。

### 训练策略

- **基础模型**: WaveGrad 2R — MINDs Lab 对 WaveGrad 2 的 PyTorch 复现 [§2.1]
- **预训练**: 在 LJSpeech 上预训练 (13,100 clips, ~24h, 单女声) [§2.3]
- **RL 微调**: 8x A100-SXM-80GB, batch size 64, 10 denoising steps, 5.5h [§2]
- **Reward model**: UTMOS (UTokyo-SaruLab MOS 预测系统) [§2.1.1] — 在 VoiceMOS Challenge 2022 数据集上训练,14h 音频
- **独立评估**: 用 NISQA (另一个 MOS 预测模型) 评估,以防 reward model 过拟合 [§2.3]

## 实验

| 指标 | Ground Truth | WaveGrad 2R (base) | RWR | DDPO | DPOK | KLinR | **DLPO** | OnlyDL | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UTMOS ↑ | 4.20 | 2.90 | 2.18 | 2.69 | 3.18 | 3.02 | **3.65** | 3.16 | [Table 1] |
| NISQA ↑ | 4.37 | 3.74 | 3.00 | 2.96 | 3.76 | 3.73 | **4.02** | 3.45 | [Table 1] |
| WER ↓ | 0.99% | 1.5% | 8.9% | 2.1% | 1.1% | 1.3% | 1.2% | 1.4% | [Table 1] |

**关键发现**:

1. **RWR 和 DDPO 严重退化** [§2.3.1]: RWR UTMOS 降至 2.18, WER 升至 8.9% — 因为 RWR 在静态数据集上微调且忽略去噪过程的序列性 [论文原文]; DDPO UTMOS 降至 2.69 — 因为单步优化不足以捕获音频波形的时域依赖 [论文原文]

2. **KL 正则化改进有限** [§2.3.1]: DPOK UTMOS 3.18 (+0.28), KLinR UTMOS 3.02 (+0.12) — KL 约束稳定了训练但也限制了任务特定的性能提升 [论文原文]

3. **DLPO 大幅提升** [§2.3.1]: UTMOS 3.65 (+0.75), NISQA 4.02 (+0.28), WER 1.2% (略优于 base 1.5%) — diffusion loss 正则化比 KL 正则化更适合 TTS diffusion 微调

4. **OnlyDL 消融** [§2.3.1]: 仅用 diffusion loss 作为 reward (不加 UTMOS),UTMOS 维持 3.16 不变,NISQA 3.45 — 说明 diffusion loss 能防止模型退化但不能提升自然度,**必须结合 reward 优化** [论文原文]

5. **人类评估** [§2.3.1]: 11 位评估者, 20 对 AB 测试, **67% 偏好 DLPO** vs 14% 偏好 base, 19% 无差异 (binomial test p < 10^-16)

## 局限性

1. **实验规模受限**: 仅在单说话人 LJSpeech + WaveGrad 2R 复现模型上验证。WaveGrad 2R 本身 UTMOS 仅 2.90 (ground truth 4.20),说明基础模型质量较低,DLPO 的改进可能部分来自"低基线效应" [agent 解读]

2. **Reward hacking 风险**: 以 UTMOS 作为 reward model,用 NISQA 独立评估虽然是好的实践,但 UTMOS 和 NISQA 都是 predicted MOS 系统,共享类似偏差。[[TTS Evaluation]] 已记录: predicted MOS 领域不匹配且缺乏 uncertainty estimation [§3.1]。67% 的人类偏好率部分缓解了这一担忧,但样本量 (11 人, 20 对) 偏小

3. **未与现代 TTS RL 方法对比**: 同期的 DiffRO (CosyVoice 3)、Multi-Reward GRPO、FPO 等方法在更大规模、更强 baseline 上取得了更显著的改进。本文的 RL 方案在 audio-level denoising trajectory 上操作,计算成本高于 token-level 方法

4. **缺乏消融分析**: α 和 β 的敏感性分析缺失;10 步 vs 更多/更少去噪步的影响未探讨;不同 reward model (如 NISQA, SpeechJudge) 作为 reward 的效果未对比

5. **可扩展性未验证**: 论文讨论了多说话人、韵律控制、多语言等未来方向 [§3],但均未实验验证

## 点评

**方法论价值**: DLPO 提出的 "将原始训练 loss 嵌入 RL reward function 作为正则项" 是一个简洁且有理论动机的想法。这个策略的通用性值得关注 — 它不依赖于特定的 RL 算法或 TTS 架构,原则上可以应用于任何 diffusion model 的 RL 微调场景。

**与知识库已有工作的关系**: 本文处于 "RL for TTS" 演进线的较早期位置。相比 Seed-TTS (2024) 和 DiffRO/GRPO (2025) 等在大规模 LLM-based TTS 上的 RL 工作,本文选择了一个更小的 testbed (WaveGrad 2R on LJSpeech),这使得结论的外推性有限。但本文的系统性对比 (RWR/DDPO/DPOK/KLinR/DLPO/OnlyDL) 提供了有价值的 empirical evidence: text-to-image diffusion RL 方法不能直接迁移到 TTS。

**实践意义**: 对于仍使用 diffusion vocoder 或端到端 diffusion TTS 的场景 (如 speech enhancement, singing voice synthesis),DLPO 的 diffusion loss 正则化策略值得考虑。但对于主流 LLM-based TTS pipeline,token-level RL 方法 (DiffRO/GRPO/FPO) 更为适用。

## 可复用的 idea

1. **Diffusion loss 作为 RL 正则项**: 在 RL 微调任何 diffusion model 时,将原始训练 loss 嵌入 reward function,而非仅依赖 KL 散度约束。这是 "function-level regularization" vs "distribution-level regularization" 的思路

2. **独立评估模型防止 reward hacking**: 训练用 UTMOS 作 reward,评估用 NISQA — 这是 RL for generative model 的好实践

3. **MDP 形式化 diffusion 去噪**: 将 T 步去噪过程建模为有限 horizon MDP,reward 仅在最终步给出。这个 formulation 来自 Black et al. (2023) 的 DDPO 工作,但在 TTS 领域的适配经验 (失败案例 + 成功方案) 有参考价值

> [!review] 审阅待补充
> 此笔记尚未经过审阅。
