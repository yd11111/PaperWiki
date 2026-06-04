---
type: paper
tier: deep
title: "ARDM-DPO: Direct Preference Optimization for Speech Autoregressive Diffusion Models"
arxiv_id: "2509.18928"
source: "Sources/2509.18928.pdf"
authors: [Zhijun Liu, Dongya Jia, Xiaoqiang Wang, Chenpeng Du, Shuai Wang, Zhuo Chen, Haizhou Li]
year: 2025
venue: "arXiv preprint"
tags: [DPO, preference-optimization, autoregressive-diffusion, zero-shot-TTS, post-training, RLHF, continuous-representation, expressiveness, robustness]
concepts: ["[[Next-Token Diffusion]]", "[[Differentiable Reward Optimization]]", "[[Diffusion Model]]", "[[Classifier-Free Guidance]]", "[[Score Matching]]", "[[Diffusion-based TTS]]"]
models: ["[[模型库/Whisper|Whisper]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓ + 4 个待确认实体页: [[Next-Token Diffusion]], [[Differentiable Reward Optimization]], [[Diffusion-based TTS]], [[Classifier-Free Guidance]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: TTS 领域的 RL/偏好优化已形成多条路线: (1) 音频级 RL (Seed-TTS 的 REINFORCE, 2024); (2) utterance-level DPO (SpeechAlign, 2024); (3) token-level 选择性 DPO (FPO, 2025); (4) token-level 可微优化 (DiffRO/CosyVoice 3, 2025); (5) GRPO (Multi-Reward GRPO, TTS-1, 2025); (6) DDPM 去噪 MDP 上的 RL (DLPO, 2025)。这些工作几乎全部针对**离散 token AR** 或 **NAR diffusion** 系统。ARDM-DPO 开辟了第七条路线: 首次将 DPO 适配到**连续 token 自回归扩散模型 (ARDM)**,填补了 next-token diffusion 范式下的偏好对齐空白。
>
> **已有认知**: [[Next-Token Diffusion]] 页 [待确认] 记录了 ARDM 的核心机制 — 在 causal Transformer 每个位置上挂 diffusion/flow head 逐 token 生成连续 latent。代表模型: LatentLM → CLEAR → VibeVoice → SemaVoice。但该页未提及 DiTAR (本文的 base model),也未记录任何 ARDM 上的 RL/DPO 工作。[[Differentiable Reward Optimization]] 页 [待确认] 记录了完整的 TTS RL 演进线,其中 Diffusion-DPO (Wallace et al., CVPR 2024) 是本文方法的直接前驱 — 将 DPO 扩展到 sequence-level diffusion model。ARDM-DPO 的技术贡献在于将 Diffusion-DPO 进一步适配到 token-level autoregressive diffusion 的马尔可夫链结构。
>
> **创新判断**: 对比 KB 中已有的 TTS 偏好优化工作,ARDM-DPO 的独特性在于: (a) 对象是连续 token ARDM 而非离散 token LM 或 NAR diffusion; (b) 推导了完整的 ARDM-specific DPO 目标函数,处理了 token 维度和 diffusion 维度的双重边际化; (c) 首次为 DiTAR 级别的 SOTA zero-shot TTS 引入偏好对齐。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓, [[Next-Token Diffusion]], [[Differentiable Reward Optimization]], [[Diffusion-based TTS]], [[Classifier-Free Guidance]] | 过滤: 无 | 未命中但可能相关: [[F0 Modeling]](pending-review, 与 Task A expressiveness 相关)

## 速查

> [!summary] 速查
> - **一句话**: 首次将 DPO 扩展到自回归扩散模型 (ARDM),在 DiTAR 上实现 F0 方差翻倍 (提升表现力) 和 CER 降低 25% (提升鲁棒性)
> - **路线**: 文本+prompt → DiTAR (causal LM + diffusion head) → 连续 token 序列 → 音频; DPO 通过比较 winning/losing 样本的 denoising loss 差异优化策略
> - **指标**: Task A: F0V 14.2→29.2 Hz (β=200, 200 steps), SIM 0.770→0.765, WER 5.17→3.73 [Table 1]; Task B: CER 8.37→6.32 (β=1600, 9K steps), SIM 0.711→0.712 [Table 2]
> - **可借鉴**: (1) ARDM 的马尔可夫链视角 + Jensen 不等式近似使 DPO loss 分解为逐 token 逐 timestep 的 denoising error 比较,工程上非常简洁; (2) 用可自动计算的 proxy reward (F0V, CTC NLL) 构建偏好对, 免除人工标注
> - **局限**: Task A 训练不稳定需 early stopping; winning/losing 双方 loss 均上升 (与 LLM DPO 类似); 偏好数据构建策略未充分探索; 未开源

## 核心问题

ARDM (自回归扩散模型) 已在 zero-shot TTS 中达到 SOTA,但预训练后生成的语音可能不符合人类偏好 — 例如给定情感化 prompt 仍输出单调语音、在复杂长文本上出现 word insertion/deletion。DPO 已在 LLM 和 sequence-level diffusion 中验证有效,但 ARDM 的**双重马尔可夫结构** (token-level AR + step-level diffusion) 使得直接套用现有 DPO 公式不可行。本文需要:

1. 推导适用于 ARDM 的 DPO 目标函数,处理 token 维度 (n) 和 diffusion 时间维度 (t) 的联合边际化
2. 在 DiTAR (当前 ARDM SOTA) 上验证 ARDM-DPO 对表现力和鲁棒性的改善

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ARDM 的采样过程被建模为马尔可夫链 [Fig 1] [§2.2]:
- 状态 $s_n^t$ 由 (token index n, diffusion time t) 索引
- 每个状态包含已去噪的历史 token $x_{<n}^0$ 和当前噪声 token $x_n^t$
- 在 token n 内: DDPM 采样从 $s_n^T$ 迭代到 $s_n^0$ (step-level diffusion) [论文原文]
- 跨 token: 完成一个 token 去噪后采样新噪声开始下一个 (token-level AR) [论文原文]

DiTAR 的核心架构优势: **将 LM encoding (历史编码) 和 diffusion denoising (当前 token 去噪) 分离**,使得 LM 只需前向一次,diffusion head 多次迭代 [§1] [论文原文]。

### 关键设计选择

**1. 从 KL-constrained policy optimization 到 implicit reward (标准 DPO 推导)**

出发点: 最大化轨迹 reward 同时约束策略不偏离参考模型 [Eq. 3] [论文原文]:

$$\max_\pi E_{\pi(x)}[r(x)] - \beta D_{KL}(\pi(x) \| \mu(x))$$

最优策略满足 $r(x) = \beta \log \frac{\pi_r(x)}{\mu(x)} + \beta \log Z_r$ [Eq. 5]。通过 Bradley-Terry 偏好模型消除 reward 和归一化常数,得到标准 DPO 框架 [论文原文]。

**为什么不直接用 LLM DPO?** [agent 解读] ARDM 的采样轨迹 x 包含所有中间扩散状态 (N tokens × T steps),不像 LLM 那样每步只有一个 discrete token。需要对中间状态做边际化才能得到 terminal reward 的表达式。

**2. 中间状态边际化 (ARDM-specific 推导)**

关键步骤是将 terminal reward $r(x_{1..N}^0)$ 表示为对所有中间状态的期望 [Eq. 6]:

$$r(x_{1..N}^0) = \beta E_{\pi_r(x|x_{1..N}^0)} \left[ \log \frac{\pi_r(x)}{\mu(x)} \right] + \beta \log Z_r$$

将 log likelihood ratio 分解为逐 token 逐 timestep 的贡献 $\ell_n^t(x)$ [Eq. 10-11] [论文原文]。

**3. Jensen 不等式近似 + denoising loss 替换**

直接计算 $\ell_n^t$ 需要知道完整策略的 transition probability,计算上不可行。论文采用 Diffusion-DPO [30] 的两步近似 [论文原文]:
- 用 $q(x_n^t|x_n^0) \cdot q(x_n^{t-1}|x_n^t, x_n^0)$ 近似联合分布 [§3]
- 用 Jensen 不等式将 log sigma 内的期望移到外面,得到下界 L [Eq. 12]

最终,每个 token-timestep 的贡献等价于 KL 散度差,进而等价于 **denoising loss 差** [Eq. 13-14]:

$$\omega_t \left( -\|v_\theta(x_n^t, x_{<n}^0) - x_n^0\|^2 + \|v_{ref}(x_n^t, x_{<n}^0) - x_n^0\|^2 \right)$$

**为什么这个简化有效?** [agent 解读] 因为 DDPM 的 transition kernel $p(x^{t-1}|x^t)$ 完全由 denoising prediction 决定,策略与参考模型的差异完全体现在 denoising error 上。这使得 ARDM-DPO loss 在工程上与标准 SFT 同构 — 只需比较两个模型在同一噪声输入上的 denoising 误差。

**4. DiTAR v-prediction 适配**

DiTAR 使用 v-prediction (连续时间 $t \in [0,1]$) 而非 DDPM epsilon-prediction。最终训练目标 [Eq. 16]:

$$L = E_{t \sim U(0,1)} \left[ \log \sigma \left( d^{-1}\beta E_n[\text{ref loss}_n - \text{policy loss}_n]_x - d^{-1}\beta E_n[\text{ref loss}_n - \text{policy loss}_n]_y \right) \right]$$

其中 $d^{-1} = 1/256$ 对 $\beta$ 做维度归一化 [§3] [论文原文]。x, y 轨迹可有不同长度 $N_x, N_y$, $E_n$ 分别计算 [论文原文]。丢弃了时间依赖权重 $\omega_t$,遵循 Diffusion-DPO 惯例 [论文原文]。

### 训练策略

- **偏好数据构建**: 对每对 (prompt, text) 用 base model 生成 K=16 或 32 个候选,用自动指标 (F0V 或 CTC NLL) 选 best/worst 构成偏好对 [§4.2, §4.3] [论文原文]
- **Task A (F0V)**: 256K 偏好对 (~1000h), K=32, 选 F0V 最高/最低 [§4.2]
- **Task B (CTC NLL)**: 430K 偏好对 (~3500h), K=16, 选 CTC loss 最低/最高 [§4.3]
- **超参**: 32×A100, local batch=1 pair, gradient accumulation 32, effective batch=1024, lr=2e-6, AdamW [§4.1]
- **推理**: 16-step DDPM, linear schedule, LM Guidance w=2 [§4.1]

## 实验

### Task A: 提升表现力 (F0 Variance)

| 指标 | Base Model | Best-of-16 | Best-of-64 | RAFT iter3 | **DPO β=200** | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0V ↑ (Hz) | 14.2 | 22.5 | 26.6 | 20.1 | **29.2** | LibriTTS test-clean | [Table 1] |
| SIM ↑ | 0.770 | 0.770 | 0.770 | 0.756 | 0.765 | LibriTTS test-clean | [Table 1] |
| WER ↓ (%) | 5.17 | 4.74 | 4.93 | 5.99 | **3.73** | LibriTTS test-clean | [Table 1] |
| KL ↓ | — | — | — | 0.237 | **0.010** | LibriTTS test-clean | [Table 1] |

关键发现:
- DPO β=200 在 200 steps 即超过 Best-of-64 和 RAFT 3 轮迭代 [Table 1] [论文原文]
- RAFT 的 KL 迅速增长 (0.237 @ iter3),SIM 持续下降;DPO KL 仅 0.010,prior preservation 更优 [Table 1] [论文原文]
- 较大 β (800) KL 约束更强但 F0V 改善更小;较小 β (200) F0V 改善大但 SIM 下降更快 → 建议 early stopping [Fig 2] [论文原文]
- **Diffusion loss 异常**: winning 和 losing 样本的 diffusion loss **均上升**,与 LLM DPO 中观察到的现象类似 [37] [Fig 3] [论文原文]

主观评价 (20 listeners, DPO β=200 vs Base) [Fig 4]:
- 自然度: 13.5% win / 81.4% tie / 5% lose → 略有下降
- Speaker Similarity: 16.9% win / 69.5% tie / 13.6% lose → 略有下降
- **表现力: 84.7% win / 15.3% tie / 0% lose → 显著提升**

### Task B: 提升长文本鲁棒性 (Text Likelihood)

| 指标 | Base Model | Best-of-8 (CER) | Best-of-8 (NLL) | **DPO β=1600** | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| NLL ↓ | 0.55 | 0.39 | 0.27 | **0.32** | Seed-TTS hard-zh | [Table 2] |
| SIM ↑ | 0.711 | 0.713 | 0.712 | **0.712** | Seed-TTS hard-zh | [Table 2] |
| CER ↓ (%) | 8.37 | 4.99 | 6.79 | **6.32** | Seed-TTS hard-zh | [Table 2] |
| KL ↓ | — | — | — | **0.009** | Seed-TTS hard-zh | [Table 2] |

关键发现:
- CER 从 8.37% 降至 6.32%,约 25% 相对降低 [Table 2] [论文原文]
- SIM 几乎无损失 (0.711→0.712) [Table 2] [论文原文]
- Task B 需要更大 β (1600) 和更长训练 (9000 steps);β ≤ 400 导致 NLL/CER 在 300 步内上升,β ≥ 6400 优化极慢 [§4.3] [论文原文]
- 主观评价: 自然度 lose/tie/win = 4.4%/88.7%/6.9%; Speaker Similarity = 2.1%/94.3%/3.6% → prior preservation 良好 [§4.3] [论文原文]

### KL Divergence 监控

论文定义了 token-average KL divergence [Eq. 17] 作为策略偏离度量。所有实验中 KL 值很低 (0.009-0.010),说明 ARDM-DPO 对参考模型的偏离可控 [Table 1, 2] [论文原文]。

## 局限性

1. **Task A 训练不稳定**: SIM 在训练中持续下降,需要 early stopping;论文承认原因未明,留作 future work [§5] [论文原文]
2. **Diffusion loss 双升现象**: winning/losing 样本 loss 均上升,与 LLM DPO 类似但未做深入分析 [Fig 3] [论文原文]
3. **偏好数据构建未充分探索**: 仅用 best/worst (F0V 或 CTC loss) 构建偏好对,未尝试更精细的标注策略 (如 FPO 的 fine-grained annotation) [§5] [论文原文]
4. **单一 reward 维度**: Task A 只优化 F0V,Task B 只优化 NLL,未尝试多维度联合优化 (对比 Multi-Reward GRPO 或 DiffRO MTR) [agent 解读]
5. **仅在 DiTAR 上验证**: 未泛化到其他 ARDM (如 LatentLM, CLEAR, VibeVoice) [agent 解读]
6. **未开源**: 代码和模型未公开,仅提供 demo 页面 [§1]

## 点评

**贡献定位**: 本文的核心价值是理论推导 — 为 ARDM 这一新兴范式提供了完整的 DPO 适配,填补了 next-token diffusion 偏好对齐的空白。推导过程清晰,从 KL-constrained optimization 到 ARDM 马尔可夫链到 Jensen 近似到 denoising loss 比较,逻辑链完整。

**与现有工作的差异化**: KB 中已有大量 TTS 偏好优化工作,但它们要么针对离散 token AR (FPO, SpeechAlign, DiffRO, GRPO),要么针对 NAR diffusion (DLPO, Emo-DPO)。ARDM-DPO 是唯一一个针对**连续 token AR diffusion** 的,这正是 DiTAR/LatentLM 等新一代 TTS 系统的核心架构。

**实验充分性**: 两个任务 (表现力 + 鲁棒性) 覆盖了 TTS post-training 的主要诉求。与 RAFT (rejection sampling SFT) 的对比展示了 DPO 的效率优势 (KL 0.010 vs 0.237)。但缺少与其他 preference alignment 方法 (如 PPO, KTO) 的对比。

**潜在影响**: 随着 ARDM (DiTAR, LatentLM 系列) 成为连续生成的主流范式,ARDM-DPO 提供的对齐工具将成为标准后训练流程的一部分。但当前验证还较初步,能否在更大规模或更多维度上保持稳定仍待验证。

## 可复用的 idea

1. **ARDM 马尔可夫链视角 + 逐 token-timestep 分解**: 将 ARDM 采样轨迹视为 (n,t) 索引的马尔可夫链,每个状态的 log-ratio 分解为 denoising error 差。这一框架可直接复用于其他连续 token AR 模型 (LatentLM, CLEAR) 的偏好对齐

2. **Proxy reward 自动构建偏好对**: 用 F0 variance (表现力) 和 CTC NLL (内容一致性) 作为自动可计算的 proxy reward,避免人工标注。这种 reward 设计思路可扩展到其他维度 (如 DNSMOS for 音质, speaker embedding distance for 音色)

3. **维度归一化 β**: 用 $d^{-1} = 1/\text{token\_dim}$ 归一化 KL 惩罚系数,使超参选择不依赖 token 维度。这是一个实用的工程 trick

4. **KL divergence 监控指标 [Eq. 17]**: 定义 token-average KL 作为 policy 偏离度的在线监控,辅助 early stopping 决策

> [!review] 审阅待补充
> 审阅将在下一步自动执行。

---

检索命中: [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓, [[Next-Token Diffusion]][待确认], [[Differentiable Reward Optimization]][待确认], [[Diffusion-based TTS]][待确认], [[Classifier-Free Guidance]][待确认] | 过滤: 无 | 未命中但可能相关: [[F0 Modeling]](pending-review)
