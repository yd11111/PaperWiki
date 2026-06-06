---
type: paper
tier: deep
title: "SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling"
arxiv_id: "2507.16884"
source: "Sources/SplitMeanFlow.pdf"
authors: [Yi Guo, Wei Wang, Zhihang Yuan, Rong Cao, Kuan Chen, Zhengyang Chen, Yuanyuan Huo, Yang Zhang, Yuping Wang, Shouda Liu, Yuxuan Wang]
year: 2025
venue: "arXiv"
tags: [flow-matching, few-step-generation, one-step-generation, average-velocity, algebraic-consistency, distillation, TTS, efficiency]
concepts: ["[[ConditionalFlowMatching]]", "[[DiffusionModel]]", "[[Classifier-FreeGuidance]]", "[[Diffusion-basedTTS]]", "[[ScoreMatching]]"]
models: ["[[论文笔记/Seed-TTS|Seed-TTS]]", "[[论文笔记/DSFlow|DSFlow]]"]
tasks: []
datasets: ["[[数据集/SEED-TTS-Eval|SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: SplitMeanFlow 处于 flow matching 的 **采样加速** 分支,与 MeanFlow (Geng et al., 2025) 直接竞争。[[ConditionalFlowMatching]] 页记录了 TTS 领域从 Diffusion (Grad-TTS, 2021) → Flow Matching (Voicebox, 2023) → CFM + DiT (CosyVoice 3, 2025) 的演进。同分支的加速工作包括: OZSpeech (learned prior, NFE=1), RapFlow-TTS (consistency flow matching, 2 步), Shallow Flow Matching (推理起点前移), DSFlow (dual supervision 蒸馏, 1 步)。SplitMeanFlow 与 MeanFlow/DSFlow 共享 "学习平均速度场而非瞬时速度场" 的核心思想,但从不同数学角度 (代数 vs 微分 vs 差商蒸馏) 求解同一目标。

**与 DSFlow 的关系**: [[论文笔记/DSFlow|DSFlow]] 也是 JVP-free 的 flow matching 加速方法,通过 teacher ODE 轨迹端点差商估计 mean velocity + endpoint 双重监督实现蒸馏。SplitMeanFlow 的路线更根本: 不依赖 teacher 蒸馏,而是直接从积分可加性推导训练目标,理论上 MeanFlow 的微分恒等式是其极限特例。但实验部分 SplitMeanFlow 也采用了两阶段 (teacher FM pretraining + student distillation) 策略。

**CFG 消除**: [[Classifier-FreeGuidance]] [待确认] 记录了 CFG 是 flow/diffusion 条件生成的标准技术。SplitMeanFlow 的 student 在 Stage 2 训练时直接学习 teacher 的 CFG 输出 (CFG dropout=0.0),从而在推理时无需 CFG,进一步减半计算 (不需要 unconditional forward pass)。这与 DSFlow 发现的 "蒸馏后 CFG 内化" 现象一致。

**已有认知**: [[DiffusionModel]] [待确认] 和 [[Diffusion-basedTTS]] [待确认] 记录了 diffusion 采样加速的历史: ProDiff (knowledge distillation), DiffGAN-TTS (GAN 加速到 1 步), Consistency Models (自一致性约束)。SplitMeanFlow 与 Consistency Models 有概念上的相似性 (都是 self-consistency 约束),但作用对象不同: Consistency Models 约束 ODE 轨迹上的点映射一致性,SplitMeanFlow 约束平均速度场在不同时间区间上的代数一致性。

> 检索命中: [[ConditionalFlowMatching]]✓, [[DiffusionModel]](pending-review), [[Classifier-FreeGuidance]](pending-review), [[Diffusion-basedTTS]](pending-review), [[ScoreMatching]](pending-review), [[Non-autoregressiveTTS]](pending-review) | 未命中但可能相关: MeanFlow (无实体页)

## 速查

> [!summary] 速查
> - **一句话**: 从积分可加性出发推导纯代数的 Interval Splitting Consistency 恒等式,用于训练平均速度场,证明 MeanFlow 的微分恒等式是其极限特例,消除 JVP 计算
> - **路线**: 噪声 z1 → uθ(zt, r, t) 预测平均速度 → z0 = z1 - u(z1, 0, 1) (一步生成); 训练时对区间 [r,t] 取中间点 s,强制 (t-r)u(zt,r,t) = (s-r)u(zs,r,s) + (t-s)u(zt,s,t) 的代数一致性
> - **指标**: 2-step SFT SIM 0.789 vs FM 10-step 0.787, CMOS -0.01; 1-step ICL WER 0.0286 = FM 10-step, CMOS 0; 20x 加速 [Table 1, 2]
> - **可借鉴**: (1) 从积分可加性推导 self-consistency 约束,绕过 JVP; (2) 代数恒等式 → 微分恒等式的极限关系,证明 general-to-special 的理论层次; (3) CFG 内化策略 (Stage 2 dropout=0.0,student 直接输出 guided velocity)
> - **局限**: 仅在 Seed-TTS 内部系统验证,无公开 checkpoint/代码; 未与 DSFlow/RapFlow-TTS 等同期加速方法直接对比; 实验仅覆盖 audio 域,未验证 image/video

## 核心问题

Flow matching 生成模型推理需多步 ODE 求解 (典型 10-100 步),延迟高。MeanFlow 通过学习平均速度场实现 few-step 生成,但其训练依赖微分恒等式 u = v - (t-r)du/dt,需要计算 Jacobian-Vector Product (JVP),带来三个问题 [§1]:

1. **计算代价**: JVP 需额外一次反向传播量级的计算
2. **训练不稳定**: 高精度反向传播可能引起数值不稳定 [引用 19, 24]
3. **硬件兼容性**: 某些加速器和软件后端对 JVP 支持有限或效率低

**核心问题**: 能否找到一种不依赖微分算子的方式来约束平均速度场,使训练更简单、更稳定、更通用?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SplitMeanFlow 的核心是一个纯代数的训练目标: Interval Splitting Consistency。模型学习平均速度场 uθ(zt, r, t),表示从时间 r 到时间 t 的平均速度。推理时一步生成: z0 = z1 - uθ(z1, 0, 1) [§4.1.2]。

架构本身与 MeanFlow 共享同一类神经网络 (输入额外的 r 参数),不依赖特定 backbone。实验中使用 Seed-TTS 的四模块架构 (speech tokenizer + token LM + token diffusion + vocoder),SplitMeanFlow 替换其中的 token diffusion 模块 [§5.2]。

### 关键设计选择

#### 为什么从积分可加性出发而非微分?

[论文原文] 平均速度的本质是积分: u(zt, r, t) = 1/(t-r) * ∫_r^t v(zτ, τ)dτ [Eq.7]。积分最基本的代数性质是可加性: ∫_r^t = ∫_r^s + ∫_s^t。将位移定义 D(a,b) = (b-a)u(zb, a, b) 代入可加性,直接得到 Interval Splitting Consistency [§4.2.2, Eq.10]:

```
(t-r)u(zt, r, t) = (s-r)u(zs, r, s) + (t-s)u(zt, s, t)
```

[论文原文] MeanFlow 的微分方法是 "绕道" (detour through differentiation),而 SplitMeanFlow 直接利用积分的内在结构,不引入任何微分算子 [§4.2.2]。

[agent 解读] 这个选择的深层逻辑是: MeanFlow 通过微积分基本定理把积分问题转化为微分问题 (g'(t) = v(zt,t),其中 g(t) = (t-r)u),然后用自动微分计算 g'。SplitMeanFlow 认为这个 "积分→微分→自动微分" 的链条不必要,可以直接在积分的代数层面建立自一致性,跳过微分这一步。

#### Interval Splitting Consistency 如何推广 MeanFlow?

[论文原文] 将 Eq.10 重排为 [(t-r)u(zt,r,t) - (s-r)u(zs,r,s)] / (t-s) = u(zt,s,t),取极限 s→t [§4.3.1, Eq.12-18]:

1. 右边: lim_{s→t} u(zt, s, t) = v(zt, t) (平均速度在区间趋零时退化为瞬时速度) [Eq.13]
2. 左边: 定义 g(t) = (t-r)u(zt,r,t),则左边 = [g(t)-g(s)]/(t-s) → g'(t) [Eq.14-15]
3. 展开 g'(t) = u + (t-r)du/dt,代入 g'(t) = v(zt,t),整理得 u = v - (t-r)du/dt [Eq.18]

这正是 MeanFlow 恒等式。因此 MeanFlow 是 SplitMeanFlow 在 s→t 极限下的特例 [§4.3.1]。

[agent 解读] 这个证明的力量在于: SplitMeanFlow 的代数恒等式对任意 s ∈ [r,t] 成立,而 MeanFlow 只利用了 s=t 这一个极限点。类比: 用整个函数的积分性质 vs 仅用函数在一个点的导数。理论上,对多个 s 值施加约束应该提供更丰富的训练信号。

#### 训练算法的关键细节

[论文原文] Algorithm 1 [§4.2.2]:
1. 采样 r, t (0 ≤ r ≤ t ≤ 1),采样 λ ~ U(0,1),计算 s = (1-λ)t + λr
2. 构造 zt = (1-t)x + tε (线性 schedule)
3. 前向: u2 = uθ(zt, s, t)
4. 中间点: zs = zt - (t-s)u2 (用模型自身的预测推断 zs)
5. 前向: u1 = uθ(zs, r, s)
6. 目标: target = (1-λ)u1 + λu2
7. 损失: L = ||uθ(zt, r, t) - sg(target)||^2

[agent 解读] 关键: zs 不是从 ground-truth 计算的,而是用模型自身对 u(zt,s,t) 的预测推断出来的 (step 4)。这使得整个训练完全自监督 (self-supervised),不需要知道真实 ODE 轨迹上 zs 的位置。这与 Consistency Models 的思路类似: 模型通过自己的预测来约束自己,形成 bootstrapping。stop-gradient 保证梯度只流过主预测 uθ(zt,r,t),而不通过 target 回传。

#### 边界条件: 防止退化解

[论文原文] 差分方程形式的 Eq.11 需要边界约束防止退化 (如全零解) [§4.2.2]。边界条件为: 当 r=t 时, u(zt, r, t) = v(zt, t),即平均速度退化为瞬时速度。

[论文原文] 实践中使用 flow ratio p: 每个 batch 中 p 比例的样本强制 r=t (用 teacher 的瞬时速度作为目标),1-p 比例强制 Interval Splitting Consistency loss。需 p ≥ 0.5 以保证稳定训练 [§5.1]。

[agent 解读] p ≥ 0.5 的要求说明边界条件是训练的 "锚",没有它自监督目标可能漂移。这与 MeanFlow 需要部分样本做 boundary grounding 完全一致。DSFlow 发现类似现象: endpoint supervision 是不可缺少的基础,velocity alignment 是增量信号。

### 训练策略

[论文原文] 两阶段训练 [§5.1]:

**Stage 1: Teacher 预训练**
- 标准 flow matching 训练,学习瞬时速度场 v(zt, t)
- 结果: 高质量 teacher Mteacher

**Stage 2: Student 训练 (SplitMeanFlow)**
- Student 权重从 teacher 初始化
- 混合目标: p 比例 boundary (用 teacher v 作目标) + (1-p) 比例 ISC loss
- CFG handling: student 的 CFG dropout 设为 0.0; teacher 用固定 CFG scale 生成瞬时速度,student 直接学习 guided velocity [§5.1]
- 实验用 Seed-TTS 框架,diffusion model 占推理 50%+ 开销 (10 步 diffusion) [§5.2]

#### 与 Shortcut Model 的区别

[论文原文] Shortcut model (Frans et al., 2025) 在 s = (r+t)/2 的特殊情况下与 SplitMeanFlow 部分等价,但核心设计哲学不同 [§4.3.1]: SplitMeanFlow 的恒等式对任意连续 s ∈ [r,t] 成立,而 shortcut model 的 d 参数是离散的。SplitMeanFlow 从平均速度概念和积分可加性出发,是更系统的理论推导。

## 实验

### Seed-TTS SFT (有监督微调)

| 指标 | SplitMeanFlow 2-step | DMD 2-step | FM 10-step | 出处 |
| --- | --- | --- | --- | --- |
| SIM (↑) | 0.789 | 0.787 | 0.787 | [Table 1] |
| WER (↓) | 0.0561 | 0.0561 | 0.0551 | [Table 1] |
| CMOS vs FM | -0.01 | -0.04 | 0 | [Table 1] |
| CFG | N | N | Y | [Table 1] |
| Steps | 2 | 2 | 10 | [Table 1] |

### Seed-TTS ICL (上下文学习)

| 指标 | SplitMeanFlow 2-step | SplitMeanFlow 1-step | FM 10-step | 出处 |
| --- | --- | --- | --- | --- |
| SIM (↑) | 0.681 | 0.685 | 0.686 | [Table 2] |
| WER (↓) | 0.0297 | 0.0286 | 0.0286 | [Table 2] |
| CMOS vs FM | 0 | 0 | 0 | [Table 2] |

[论文原文] 1-step SplitMeanFlow 在 ICL 任务上实现了与 10-step FM baseline 统计等价的性能,代表 20x 计算成本降低且无可辨别的质量损失 [§5.3]。

[agent 解读] 有趣的是 1-step 在 ICL 上比 2-step 表现更好 (SIM 0.685 vs 0.681, WER 0.0286 vs 0.0297)。论文未解释这一现象。一种可能的解释是: ICL 任务的条件信息 (参考音频) 提供了更强的约束,使得单步预测足以捕获目标分布,而 2-step 引入了额外的 ODE 积分误差。

### 部署

[论文原文] 1-step 和 2-step SplitMeanFlow 已部署在大规模语音合成产品 (豆包/Doubao) 中,实现 20x 加速 [Abstract]。

## 局限性

1. **仅内部验证**: 所有实验基于 Seed-TTS 框架,无公开代码/权重,可复现性受限
2. **缺少同期方法对比**: 未与 DSFlow、RapFlow-TTS、Shallow Flow Matching 等同期 flow matching 加速方法直接比较; 仅与 DMD 和标准 FM 对比
3. **Domain 限制**: 仅在 audio 域验证,论文声称的通用性 (图像/视频) 未有实验支撑
4. **Ablation 缺失**: 未报告关键超参 (flow ratio p, λ 分布, s 采样策略) 的消融实验
5. **理论 vs 实践 gap**: 理论上 SplitMeanFlow 比 MeanFlow 更 general (对任意 s 成立 vs 仅 s→t 极限),但实验中未量化这个 "更 general" 带来的具体质量提升 -- 两者未直接对比
6. **依赖 teacher**: 实验中的两阶段策略仍需先训练 teacher FM,SplitMeanFlow 的 from-scratch 训练效果未报告 (作者提到 "can be trained from scratch" 但 "two-stage approach yields significantly faster convergence and superior final performance" [§5.1])

## 点评

**理论贡献显著**: SplitMeanFlow 的核心贡献在于将 MeanFlow 的微分恒等式还原为积分可加性的极限特例。这不仅在数学上更优雅 (从更基本的公理出发),而且在工程上消除了 JVP 的计算瓶颈。证明过程清晰且具教学价值 [§4.3.1]。

**实验设计的局限性降低了论文的说服力**: 虽然理论上 SplitMeanFlow 推广了 MeanFlow,但实验中 (1) 未直接与 MeanFlow 比较质量/训练效率, (2) 未与同期的 DSFlow 比较, (3) 缺少 ablation。论文最核心的 claim -- "代数方法优于微分方法" -- 在实验中没有 apples-to-apples 的验证。

**工业部署价值明确**: JVP-free 的特性使得 SplitMeanFlow 对大规模部署极具吸引力。JVP 不仅增加计算量,还需要特定的自动微分支持,在某些推理框架 (如 TensorRT, ONNX Runtime) 中难以高效实现。消除这一依赖是非平凡的工程贡献。

**与 DSFlow 的互补性**: DSFlow 走蒸馏路线 (teacher 提供 endpoint + mean velocity 监督), SplitMeanFlow 走自监督路线 (self-consistency)。两者都实现了 JVP-free,但方法论不同: DSFlow 用差商近似 mean velocity,SplitMeanFlow 直接建立代数约束不需要任何 velocity 的显式估计。理论上 SplitMeanFlow 更 "纯粹",但 DSFlow 的 ablation 和跨架构验证更充分。

## 可复用的 idea

1. **积分可加性 → self-consistency 训练目标**: 任何定义为积分的量 (位移、累积奖励、路径积分),都可以用 interval splitting 推导代数自一致性约束,作为不需要 ground-truth 的训练信号。这个原则可推广到 RL (value function 的 temporal difference) 和 physics-informed neural networks。

2. **General-to-special 证明策略**: 证明现有方法 (MeanFlow) 是自己方法的极限特例,是非常有力的理论论证范式。在提出新训练目标时,可以检查: 已有目标是否可以作为新目标在某个极限下的特例?

3. **CFG 内化 (无额外推理成本)**: 让 student 直接学习 teacher 的 guided output (CFG dropout=0.0),推理时不再需要 unconditional pass。这与 DSFlow 的 weak CFG (w=0.05) 形成对比: SplitMeanFlow 选择完全内化 (w=0), DSFlow 保留微弱调控能力。两种策略各有场景。

4. **自监督中间点构造**: 用模型自身预测的平均速度推断中间点 zs = zt - (t-s)u2,而非依赖 teacher trajectory。这种 bootstrapping 思路在无 teacher 或 teacher 不可用的场景下特别有价值。

## 审阅

(待独立审阅 agent 填写)
