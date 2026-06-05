---
type: paper
tier: deep
title: "Bridge-TTS: Schrödinger Bridges Beat Diffusion Models on Text-to-Speech Synthesis"
arxiv_id: "2312.03491"
source: "Sources/Bridge-TTS.pdf"
authors: [Zehua Chen, Guande He, Kaiwen Zheng, Xu Tan, Jun Zhu]
year: 2023
venue: "arXiv preprint"
tags: [TTS, diffusion, Schrödinger-bridge, data-to-data, mel-generation, sampling-efficiency, noise-schedule]
concepts: ["[[DiffusionModel]]", "[[Diffusion-basedTTS]]", "[[ScoreMatching]]", "[[ConditionalFlowMatching]]", "[[MelSpectrogram]]", "[[DurationPredictor]]"]
models: ["[[模型库/VITS|VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[ConditionalFlowMatching]]✓, [[DiffusionModel]][待确认], [[Diffusion-basedTTS]][待确认], [[ScoreMatching]][待确认], [[MelSpectrogram]][待确认], [[DurationPredictor]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Bridge-TTS 位于 Diffusion-based TTS 演进线上,是 Grad-TTS (Popov et al., 2021) 的直接改进。Grad-TTS 是 SDE-based diffusion TTS 的开创性工作 [§3.2.1],它用 U-Net decoder 从 mean-shifted Gaussian prior 生成 mel spectrogram。Bridge-TTS 的核心创新在于将 Grad-TTS 的 data-to-noise 过程替换为 Schrödinger bridge 的 data-to-data 过程,从而使先验从含噪高斯变为干净的 text latent。
>
> **已有认知**: 知识库中已有 Diffusion Model → Score Matching → Conditional Flow Matching 的完整理论链。Diffusion Model 页记录了 SDE 统一框架和 probability flow ODE;CFM 页(confirmed)记录了 ODE 直接回归路径的范式,是 diffusion 的演进方向。Bridge-TTS 处于 diffusion 和 flow matching 之间的理论空间 — 它保留了 SDE 的随机性(bridge SDE),但也推导了确定性采样(bridge ODE),且 bridge ODE 在极限情况下恢复 DDIM。
>
> **创新判断**: 相对于 Diffusion-based TTS 页记录的加速方案(ProDiff 知识蒸馏, DiffGAN-TTS GAN 加速, CoMoSpeech consistency distillation),Bridge-TTS 的路线独特 — 不是在 data-to-noise 框架内加速,而是从根本上改变了生成过程的起点(clean prior 替代 noisy prior),这是一个理论层面的范式转换。
>
> 检索命中: [[ConditionalFlowMatching]]✓, [[DiffusionModel]][待确认], [[Diffusion-basedTTS]][待确认], [[ScoreMatching]][待确认], [[MelSpectrogram]][待确认], [[DurationPredictor]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 Schrödinger bridge 将 diffusion TTS 的 data-to-noise 过程替换为 data-to-data 过程,以 text encoder 输出(干净 latent)为先验,同时提升生成质量和采样效率
> - **路线**: Text → Phoneme Encoder + Duration Predictor → text latent z (clean prior) → Schrödinger Bridge (U-Net decoder) → Mel Spectrogram → HiFi-GAN → Waveform
> - **指标**: LJ-Speech 50-step MOS 4.09 (VP) / 4.07 (gmax) vs Grad-TTS 3.99; 2-step MOS 4.04 vs CoMoSpeech 3.87; 4-step MOS 4.10 vs FastGrad-TTS 3.87 [Table 2, Table 3]
> - **可借鉴**: (1) 当条件信号(text latent)本身就能提供目标的结构信息时,用 data-to-data bridge 替代 data-to-noise diffusion 可显著减少采样步数; (2) 非对称噪声调度(asymmetric marginal variance)优于对称调度; (3) temperature-scaled SDE 采样(tau_b=2)能有效抑制背景 artifacts
> - **局限**: 仅在 LJ-Speech 单说话人数据集上验证;未与 flow matching 方法(Matcha-TTS, VoiceFlow)直接对比;无零样本/多说话人实验;vocoder 依赖预训练 HiFi-GAN

## 核心问题

Bridge-TTS 要解决的核心问题是: **diffusion-based TTS 的先验分布被限制为含噪表示,无法充分利用 text encoder 已有的结构性信息**。

在 Grad-TTS 中,text encoder 输出 z = E(y) 被用作先验分布的均值(mean-shifted Gaussian),但前向扩散过程仍然是 data-to-noise 的,即 p_T = N(z, I),先验本身含有大量噪声 [§1]。DiffSinger 和 PriorGrad 等后续工作也尝试改善先验分布,但都受制于"必须加噪"这一前提 [§1]。

Bridge-TTS 的思路是: 既然 text latent z 在训练时被 MSE loss 监督向 ground-truth mel 对齐,它已经包含了目标的大量结构信息,那么能否直接从 z 生成 mel,完全不经过噪声? 答案是通过 Schrödinger bridge 实现 data-to-data 过程 [§1, §3.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Bridge-TTS 继承了 Grad-TTS 的整体架构(text encoder + diffusion decoder),但将 decoder 的生成过程从 diffusion 替换为 Schrödinger bridge [§3]:

1. **Text Encoder**: phoneme encoder + duration predictor + MAS (Monotonic Alignment Search),输出 text latent z = E(y),与 Grad-TTS 完全相同 [§2.2]
2. **Bridge Decoder**: 不再从 N(z, I) 去噪生成 mel,而是从 x_1 = z 出发,通过 Schrödinger bridge 的反向过程到达 x_0 = mel [§3.1]
3. **Vocoder**: 预训练 HiFi-GAN,与 Grad-TTS 使用相同 checkpoint [§4.1]

网络参数量: encoder 7.2M + U-Net decoder 7.6M,与 Grad-TTS 完全相同 [§4.1]。

### 关键设计选择

#### 1. 为什么用 Schrödinger Bridge 而不是 diffusion?

[论文原文] Diffusion 模型的前向过程预定义为 data-to-noise,先验分布被限制为高斯噪声(或均值偏移的高斯)。Schrödinger Bridge (SB) 则寻求在两个任意分布之间的最优传输路径,允许 data-to-data 过程 [§1, §2.3]。在 TTS 中,text latent z 已经通过 MSE loss 监督包含了 mel 的结构信息,用它作为先验比用噪声作为先验自然更高效 [§1]。

[agent 解读] 这实际上是在回答一个根本性问题: 当条件信号足够强时,是否还需要从纯噪声开始生成? Bridge-TTS 表明答案是否定的 — 从一个已经接近目标的结构化先验出发,可以用更少步数达到更好质量。

#### 2. 如何使 Schrödinger Bridge 在 TTS 中可解析?

一般的 SB 需要昂贵的迭代求解(IPF 等) [§2.3]。Bridge-TTS 的关键技术贡献是: **利用 paired data (x_0=mel, x_1=z) 的特殊结构,推导出在线性 drift 参考 SDE 下完全可解析 (fully tractable) 的 SB 解** [§3.1, Proposition 3.1]。

具体地,通过给边界数据加微小高斯噪声 N(0, epsilon^2 I) 来避免 Dirac delta 与连续密度的 KL 散度为无穷的问题,然后令 epsilon → 0 取极限,得到干净 paired data 之间的 SB 解 [§3.1]:

边际分布为 Gaussian,均值是 x_0 和 x_1 的插值,方差在边界为零、中间为正(类似 Brownian bridge 但更一般化) [§3.1, Eqn. 12]。

[论文原文] 当 f(t)=0, g(t)=constant 时,恢复为 Brownian bridge;当 f(t) ≠ 0(如 VP 调度),则得到推广的 Brownian bridge,这在 SB 和 TTS 文献中均为首次 [§3.1]。

#### 3. 训练目标: data prediction

Bridge-TTS 使用 **x_0 预测 (data prediction)** 作为训练目标 [§3.2, Eqn. 13]:

```
L_bridge = E_{(x0,y)} E_t [ || x_theta(x_t, t, x_1) - x_0 ||^2 ]
```

其中 x_t 按 SB 边际分布(Eqn. 12)采样 [§3.2]。

[论文原文] 虽然理论上 noise prediction、score prediction、velocity prediction 等价,但实验发现 data prediction 表现最好(Appendix D),其他参数化"在实践中表现更差或很差" [§3.2]。

[agent 解读] 这与 diffusion 文献中的发现有所不同(diffusion 中 noise prediction 通常优于 data prediction)。可能的原因是 SB 的边际分布与 diffusion 不同,data prediction 在 bridge 语境下更稳定。

#### 4. 噪声调度设计

Bridge-TTS 提出了两种噪声调度 [§3.2, Table 1]:

| 调度 | f(t) | g^2(t) | 特点 |
|------|------|--------|------|
| Bridge-gmax | 0 | beta_0 + t(beta_1 - beta_0) | 时变扩散系数,无 drift |
| Bridge-VP | -1/2(beta_0 + t(beta_1-beta_0)) | beta_0 + t(beta_1 - beta_0) | 与 SGM 的 VP 完全对齐 |

[论文原文] Bridge-gmax 和 Bridge-VP 均产生**非对称 (asymmetric) 的边际方差模式**,将更多步骤分配给去噪阶段;而常数 g(t) 产生对称方差(类似 Brownian bridge),质量明显更差 [§4.4]。这表明非对称调度是提升质量的关键设计。

#### 5. 采样方案: Bridge SDE 和 Bridge ODE

基于 SB 的可解析性,Bridge-TTS 推导了两种采样方式 [§3.3]:

- **Bridge SDE** (Eqn. 14): 随机采样,保留 Wiener 过程噪声
- **Bridge ODE** (Eqn. 16): 确定性采样,probability flow ODE 的桥版本

进一步利用指数积分器 (exponential integrator) 消除线性项,得到一阶离散化公式 [Proposition 3.2, Eqn. 19-20]:
- Bridge SDE 一阶离散化恢复了 Brownian bridge 上的后验采样 [§3.3]
- Bridge ODE 一阶离散化在极限下恢复 DDIM [§3.3]

[论文原文] Temperature-scaled SDE 采样 (tau_b = 2, 即噪声分布缩放为 N(0, tau_b^{-1} I)) 在大步数和少步数下均达到最佳质量,有效抑制背景 artifacts [§4.4]。

### 训练策略

- **两阶段训练** (fixed prior): 先训练 text encoder (warm-up),再训练 decoder [§4.4]。实验表明 fixed prior 比 joint training (mutable prior) 质量更好 (CMOS +0.13 at NFE=4, +0.17 at NFE=1000) [Table 4]。
- **Encoder loss**: 由 Grad-TTS 的 negative log-likelihood L_enc = -E[log p_enc(x|y)] 简化为 MSE loss L'_enc = E[||E(y) - x_0||^2],因为 encoder 不再参数化高斯分布 [§3.2]。
- 总训练目标: L_bridge-tts = L'_enc + L_dp + L_bridge [§3.2]。
- 训练设置: batch size 16, 1.7M iterations, 单卡 RTX 3090, 2.5 天 [§4.1]。

## 实验

| 指标 | 本文 (Bridge-TTS gmax) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| MOS (50-step) | 4.07 +/- 0.07 | Grad-TTS 3.99 +/- 0.07 | LJ-Speech | [Table 2] |
| MOS (50-step, VP) | 4.09 +/- 0.07 | VITS 3.99 +/- 0.07 | LJ-Speech | [Table 2] |
| MOS (1000-step) | 4.07 +/- 0.07 | Grad-TTS 3.98 +/- 0.07 | LJ-Speech | [Table 2] |
| MOS (4-step) | 4.10 +/- 0.06 | FastGrad-TTS 3.87 +/- 0.07 | LJ-Speech | [Table 3] |
| MOS (4-step) | 4.10 +/- 0.06 | DiffGAN-TTS 3.78 +/- 0.07 | LJ-Speech | [Table 3] |
| MOS (2-step) | 4.04 +/- 0.06 | CoMoSpeech 3.87 +/- 0.07 | LJ-Speech | [Table 3] |
| MOS (2-step) | 4.04 +/- 0.06 | FastSpeech 2 (1-step) 3.84 +/- 0.07 | LJ-Speech | [Table 3] |
| RTF (2-step) | 0.009 | CoMoSpeech 0.009 | LJ-Speech | [Table 3] |
| RTF (50-step) | 0.117 | Grad-TTS 0.116 | LJ-Speech | [Table 2] |

**关键发现**:

1. **质量提升**: 在 50-step 和 1000-step 生成中均显著优于 diffusion 对手 Grad-TTS (CMOS 显著正向) [§4.2]。
2. **少步数优势突出**: 4-step 时 MOS 4.10 甚至超过 50-step 的 Grad-TTS (3.99),说明 data-to-data 先验在少步数下优势更大 [Table 2, Table 3]。
3. **2-step 超越蒸馏方法**: 在仅 2 步推理下 MOS 4.04 超过 CoMoSpeech (3.87, 需额外 consistency distillation),且 RTF 相同 [Table 3]。
4. **无需额外训练程序**: 不需要知识蒸馏、GAN 对抗训练或辅助模型,论文声称单次训练即达到 SOTA [§5, §1]。

**消融实验** (Table 4, CMOS):

| 比较维度 | NFE=4 | NFE=1000 | 出处 |
|----------|-------|----------|------|
| Fixed prior vs mutable prior | +0.13 | +0.17 | [Table 4] |
| Bridge-gmax vs constant g(t) | +0.12 | +0.14 | [Table 4] |
| Bridge-gmax vs Bridge-VP | +0.03 | +0.08 | [Table 4] |
| SDE (tau_b=2) vs SDE (tau_b=1) | +0.07 | +0.19 | [Table 4] |
| SDE (tau_b=2) vs ODE | +0.10 | +0.00 | [Table 4] |

## 局限性

1. **仅 LJ-Speech 验证**: 所有实验限于 LJ-Speech 单说话人数据集(24 小时),未验证多说话人、零样本、跨语言等更复杂场景 [§4.1]。
2. **缺少与 flow matching 的对比**: 论文发表时 (Dec 2023) Matcha-TTS 已公开,但未与 CFM-based 方法直接对比。CFM 也是 data-to-data 路径(从 noise 到 data 的 ODE,但可视为从简单分布到复杂分布的桥),两者的对比是读者关心的问题。
3. **Vocoder 依赖**: 使用预训练 HiFi-GAN,端到端质量受 vocoder 限制。
4. **理论与实践的 gap**: 虽然推导了高阶采样器(Appendix C, predictor-corrector),但实验发现一阶即足够、高阶无显著差异 [§3.3],这限制了理论深度的实际价值。
5. **Fixed prior 的额外成本**: 两阶段训练(先 warm-up encoder)增加了训练复杂度,虽然论文未报告具体额外时间。

## 点评

Bridge-TTS 的最大价值是**观念转换**: 当条件信号(text latent)已经包含目标(mel)的结构信息时,从这个信号出发生成比从噪声出发生成更自然。这个洞察在理论上并不难理解,但 Bridge-TTS 首次在 TTS 中用完整的数学框架(tractable SB + 噪声调度 + 采样器设计)将其实现,且实验效果显著。

从 Diffusion-based TTS 的演进来看,Bridge-TTS 的位置很有趣: 它是 Grad-TTS 的"理论修正"(用 SB 替代 SGM),而非"工程加速"(如 ProDiff 的蒸馏或 DiffGAN-TTS 的 GAN 加速)。这条路线与 flow matching 的思路有交集 — 两者都试图缩短从先验到数据的"距离",但实现方式不同:CFM 选择确定性 ODE + optimal transport,SB 选择随机 SDE + 最优路径测度。

一个值得注意的信号是: Bridge-TTS 的 bridge ODE 在极限下恢复 DDIM,这暗示 SB 和 diffusion/flow matching 之间存在更深层的理论统一性。后续工作可能会在这个方向上进一步发展。

## 可复用的 idea

1. **Data-to-data 先验设计原则**: 当条件信号足够强(如 text encoder 输出与目标高度相关)时,用条件信号本身作为生成起点而非噪声,可同时提升质量和效率。这个原则可迁移到其他条件生成任务(如 voice conversion 的 source mel → target mel)。

2. **非对称噪声调度**: Bridge-gmax 和 Bridge-VP 的非对称方差模式(更多步分配给去噪阶段)优于对称模式,这个设计选择可迁移到任何 bridge/flow 采样器中。

3. **Temperature-scaled SDE 采样**: 通过缩放采样噪声的方差(tau_b > 1)来抑制 artifacts,是一个简单有效的 trick,可直接用于其他基于 SDE 的生成器。

4. **Fixed prior (两阶段训练)**: 先稳定 encoder 再训练 decoder,避免 encoder 和 decoder 的联合优化不稳定性。虽然增加训练步骤,但在 paired data 的生成任务中可能普遍有效。

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes (0 high, 1 medium, 3 low)
> - (medium) traceability-gap: "SOTA" claim 已补充来源标注 [§5, §1]
> - (low) venue 已去掉未确认的 submission 猜测
> - (low) datasets 空值: LJ-Speech 无独立页,保持空值
> - (low) 点评节 agent 评价未标注: 可接受,点评允许主观评价
> 详见 `_review/Bridge-TTS-review.yml`
