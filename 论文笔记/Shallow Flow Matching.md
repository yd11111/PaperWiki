---
type: paper
tier: deep
title: "Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis"
arxiv_id: "2505.12226"
source: "Sources/ShallowFlowMatching.pdf"
authors: [Dong Yang, Yiyi Cai, Yuki Saito, Lixu Wang, Hiroshi Saruwatari]
year: 2025
venue: "NeurIPS 2025"
tags: [TTS, flow-matching, coarse-to-fine, inference-acceleration, ODE-solver, piecewise-flow]
concepts: ["[[Conditional Flow Matching]]", "[[Classifier-Free Guidance]]", "[[Diffusion-based TTS]]", "[[Mel Spectrogram]]", "[[Neural Vocoder]]", "[[Non-autoregressive TTS]]", "[[Duration Predictor]]"]
models: ["[[CosyVoice]]", "[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文在 [[Conditional Flow Matching]] (confirmed) 的 TTS 应用谱系中,属于"coarse-to-fine FM 的推理优化"方向。既有 KB 已知 FM 在 TTS 中的标准用法是作为 second-stage renderer,将 speech token 或 coarse mel 转为高质量 mel spectrogram (如 CosyVoice 的 OT-CFM, Matcha-TTS 的 encoder-decoder FM)。本文的创新在于改变 FM 推理的起点——从纯噪声移至 coarse 表示构造的中间状态。这与 DiffSinger 的 shallow diffusion 思路一脉相承,但在 flow matching 框架下需要新的理论工具 (CondOT 路径上的正交投影 + 分段流)。
>
> **已有认知**: [[CosyVoice]] (confirmed) 采用 LLM + OT-CFM 的 coarse-to-fine 两阶段架构,FM 模块以 speech tokens、speaker embedding、masked mel 为条件生成 mel spectrogram。[[Neural Vocoder]] (confirmed) 记录了 HiFi-GAN 和 Vocos 作为最终波形合成器的地位。[[Classifier-Free Guidance]] [待确认] 记录了 CFG 在 FM-based TTS 中的标准用法 (训练时随机丢弃条件,推理时用 β 控制引导强度)。
>
> **创新判断**: KB 中尚无"改变 FM 推理起点"或"piecewise flow for TTS"的记录。现有 CFM 概念页提到 PeRFlow 做 piecewise reflow,但用途是加速 (分窗 + 重整流),而非利用 coarse 表示跳过早期阶段。本文的正交投影确定时间点 + 单段分段流是全新贡献。
>
> 检索命中: [[Conditional Flow Matching]], [[CosyVoice]], [[Neural Vocoder]] | 过滤: [[Classifier-Free Guidance]](pending-review), [[Diffusion-based TTS]](pending-review), [[Mel Spectrogram]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 coarse-to-fine TTS 中 FM 模块的推理起点从纯噪声移至 coarse 表示构造的中间状态,用正交投影自适应确定时间点,通过单段分段流实现更自然的合成和更快的自适应步长 ODE 求解
> - **路线**: 文本/speech tokens → weak generator (encoder/LLM) → coarse mel + hidden states → SFM head (预测 X_h, t_h, sigma_h) → 正交投影到 CondOT 路径构造中间状态 → FM decoder 从中间状态出发求解 ODE → mel spectrogram → vocoder → 波形
> - **指标**: Matcha-TTS/LJ Speech PMOS 4.176 (α=2.5) vs baseline 4.008, CMOS 0.00 vs baseline -0.48 [Table 4]; CosyVoice/LibriTTS PMOS 4.106 vs baseline 3.499, SMOS 3.67 vs baseline 3.47 [Table 4]; Dopri(5) solver 加速率最高 ~48% (Matcha-TTS α=5) [Table 5]
> - **可借鉴**: (1) 正交投影确定 FM 中间状态的时间位置——可迁移到任何有 coarse 初始估计的 FM 系统; (2) SFM strength α 超参数——在推理时线性放大 coarse 信号的引导效果,简单有效; (3) 轻量 SFM head (基于 VITS duration predictor 架构,仅 Conv1d+LayerNorm) 作为即插即用模块
> - **局限**: (1) SFM head 非常简单,weak generator 不稳定时可能不收敛 (StableTTS 的 reference encoder 问题); (2) 未开源训练代码 (仅 demo); (3) 加速仅对自适应步长 ODE solver 有效,固定步长 solver 无法受益; (4) WER 和 SIM 改善不一致

## 核心问题

本文要解决的核心问题是: 在 coarse-to-fine FM-based TTS 中,FM 模块从纯噪声开始生成是低效且冗余的。coarse 表示 (来自 encoder 或 LLM) 已编码了大部分语义和声学结构,但传统做法仅将其作为条件输入,生成仍从 N(0,I) 出发。这导致:

1. **建模容量浪费**: FM 的早期步骤在从噪声走向 coarse 表示的"已知区域",做了重复劳动 [§1]
2. **自适应 ODE solver 的步数浪费**: 早期阶段梯度变化大,solver 需要更多函数评估 (NFE),推理变慢 [§5.2]
3. **生成不稳定**: 从纯噪声出发的长路径容易累积误差 [§5.1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SFM 不替换现有 TTS 模型,而是在 weak generator (encoder/LLM) 和 FM decoder 之间插入一个轻量 **SFM head**。整体流程:

1. Weak generator $g_\omega$ 输出 hidden states $\hat{H}^g$ 和 coarse mel $\hat{X}^g$ [§3.2]
2. SFM head $h_\psi$ 接收 $\hat{H}^g$,输出三个量: 缩放 mel $\hat{X}^h$、预测时间 $\hat{t}_h$、预测方差 $\log\hat{\sigma}_h^2$ [§3.2, Eq.12]
3. 训练时: 通过正交投影将 $\hat{X}^h$ 映射到 CondOT 路径上,构造中间状态 $X_{\tilde{t}_h}$,FM decoder 只训练路径的后半段 [§3.2.1-3.2.2]
4. 推理时: FM 从中间状态 $X_{\tilde{t}_h}$ 出发,用 ODE solver 积分到 t=1 得到 mel spectrogram [§3.2.3]

SFM head 的架构极其简单: 两层 Conv1d + LayerNorm + ReLU + Dropout,源自 VITS 和 Matcha-TTS 的 duration predictor [Appendix B, Fig 2]。

### 关键设计选择

**设计 1: 正交投影确定时间 (Theorem 1)**

为什么不直接指定 FM 路径的中间时间点? 因为 SFM head 输出的 $\hat{X}^h$ 质量和位置事先未知,其在 CondOT 路径上的"真实时间"需要自适应确定。[论文原文]

方法: 将 $\hat{X}^h$ 正交投影到目标 $X_1$ 上,投影系数即为 $t_h = \frac{\hat{X}^h \cdot X_1}{X_1 \cdot X_1}$ [§3.2.1, Eq.13]。这使得:
- 如果 $\hat{X}^h$ 质量好 (接近 $t_h X_1$),时间 $t_h$ 大,FM 路径短,推理快
- 如果 $\hat{X}^h$ 质量差,时间 $t_h$ 小,FM 路径长,安全降级到接近标准 FM [agent 解读]

使用 stop gradient (sg) 简化梯度传播: $t_h$ 的计算对 $\hat{X}^h$ stop gradient,避免 $t_h$ 与 $\hat{X}^h$ 的循环依赖 [§3.2.1, Eq.13]。

**设计 2: CondOT 路径映射 (Theorem 1)**

$\hat{X}^h$ 不一定精确落在 CondOT 路径上。Theorem 1 定义了一个变换,将任意 $x_m \sim N(t_m x_1, \sigma_m^2 I)$ 映射回 CondOT 路径,且映射在 Wasserstein-2 度量下连续。关键是引入缩放因子 $\Delta = \max((1-\sigma_{\min})t_h + \sigma_h, 1)$:
- 当 $\Delta < 1$ (正常情况): 直接在路径上,可混入外部噪声 $X_0$ [§3.2.1, Eq.14-15]
- 当 $\Delta \geq 1$ (训练早期): $\hat{X}^h$ 偏离太远,通过 $1/\Delta$ 缩放拉回路径,此时行为确定性 [论文原文, §3.2.1]

[agent 解读] $\Delta$ 机制本质上是一个"安全网": 训练早期 SFM head 预测不准时自动降级,避免注入错误的中间状态。

**设计 3: 单段分段流 (Theorem 2)**

将 CondOT 路径在 $t_m$ 处分为两段,只使用后半段 $[t_m, 1]$:

$$X_t = (1-t_S)X_{\tilde{t}_h} + t_S(X_1 + \sigma_{\min}X_0), \quad t_S \in [0,1]$$

[论文原文, §3.2.2, Eq.18] 证明了分段流的 VF 与原始完整路径的 VF 一致 (Theorem 2 证明)。

为什么不用完整路径的后半段直接训练? 因为需要重新参数化 $t$ 使其在 $[0,1]$ 上均匀采样 ($t = (1-\tilde{t}_h)t_S + \tilde{t}_h$),确保训练分布与推理一致 [agent 解读]。

**设计 4: SFM 强度 α (推理时)**

训练时自适应确定的 $t_h$ 倾向于偏小 [§3.2.3],导致中间状态接近路径起点,先验信息不足。推理时引入 $\alpha \geq 1$ 线性放大:

$$\tilde{t}_h = \frac{\alpha}{\Delta} t_h, \quad \tilde{X}_h = \frac{\alpha}{\Delta} \hat{X}_h$$

[论文原文, §3.2.3, Eq.22]

为什么 $\alpha$ 能 work? 因为 $\hat{X}^h$ 被 $t_h X_1$ 监督 (Eq.13 的 $L_\mu$),所以线性放大近似于沿 $X_1$ 方向前进更远的距离。但 $\alpha$ 过大会放大估计误差,且减少随机性 (外部噪声 $X_0$ 的系数减小),导致生成质量下降 [论文原文, §5.1]。

最优 $\alpha$ 通过验证集 PMOS 网格搜索确定 (步长 1.0,邻域 ±0.5) [§4.5]。

**设计 5: 不使用 coarse mel 作为 FM 条件**

传统做法将 coarse mel 作为 FM 条件输入。SFM 的默认设置不这样做。实验表明 SFM-c (同时使用 coarse mel 条件 + SFM) 在 U-Net 模型上导致 $t_h \to 0$ (方法失效),在 DiT 模型上主观评测更差 [§5.1]。[论文原文] 解释: 当 coarse mel 已作为条件时,FM decoder 获取了足够信息,SFM head 的中间状态不再提供额外指导。[agent 解读] 这说明 SFM 的核心价值不是提供更多条件信息,而是改变推理的起始点。

### 训练策略

**总损失** [§3.2.2, Eq.21]:
$$L_{\text{SFM}} = L_{\text{coarse}} + L_t + L_\sigma + L_\mu + L_{\text{CFM}}$$

- $L_{\text{coarse}}$: coarse mel 重建损失 (L2)
- $L_t$: 时间预测损失 $(\hat{t}_h - \tilde{t}_h)^2$
- $L_\sigma$: 方差预测损失 $(\log\hat{\sigma}_h^2 - \log\tilde{\sigma}_h^2)^2$ (实际预测 $\log\sigma^2$ 保证数值稳定 [Appendix C])
- $L_\mu$: 均值对齐损失 $||\hat{X}^h - t_h X_1||^2$
- $L_{\text{CFM}}$: 标准条件流匹配损失,但只在路径后半段 $[t_h, 1]$

训练细节 [Table 1]:
- 所有模型在官方配置基础上微改
- Matcha-TTS: 800 epochs, LR 4e-4/2e-4
- CosyVoice: 仅训练 flow module (200 epochs), LLM 和 tokenizer 使用官方预训练权重
- Warmup: StableTTS/CosyVoice 系列使用 10% warmup
- 所有训练在 H100 GPU 上进行

## 实验

| 指标 | 本文 (SFM) | Baseline | Ablated | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| PMOS | 4.176 (α=2.5) | 4.008 | 4.026 | LJ Speech (Matcha-TTS) | [Table 3,4] |
| CMOS | 0.00 | -0.48 | -0.27 | LJ Speech (Matcha-TTS) | [Table 4] |
| PMOS | 3.679 (α=3.5) | 3.462 (α=1) | — | VCTK (Matcha-TTS) | [Table 4,6] |
| CMOS | 0.00 | -0.31* | -0.39* | VCTK (Matcha-TTS) | [Table 4] |
| PMOS | 3.486 (α=3.0) | — | 3.281 (α=1) | VCTK (StableTTS) | [Table 4,7] |
| CMOS | 0.00 | — | -0.34* | VCTK (StableTTS) | [Table 4] |
| PMOS | 4.087 (α=2.0) | 3.721 (α=1) | — | LibriTTS (CosyVoice) | [Table 4,8] |
| SMOS | 3.67 | 3.47 | 3.58 | LibriTTS (CosyVoice) | [Table 4] |
| PMOS | 3.823 (α=2.5) | 3.405 (α=1) | — | LibriTTS (CosyVoice-DiT) | [Table 4,9] |
| SMOS | 3.21 | — | 3.15 | LibriTTS (CosyVoice-DiT) | [Table 4] |
| Dopri(5) RTF | 0.076 (α=5) | 0.145 | 0.145 | LJ Speech (Matcha-TTS) | [Table 5] |
| Dopri(5) NFE | 63.74 (α=5) | 121.46 | 121.46 | LJ Speech (Matcha-TTS) | [Table 5] |
| Dopri(5) speedup | 47.6% | — | 0% | LJ Speech (Matcha-TTS) | [Table 5] |

**关键发现**:

1. **一致的自然度提升**: 所有 5 个模型配置下 SFM 的 PMOS 均优于 baseline 和 ablated,CMOS/SMOS 主观评测同样一致 [Table 4]
2. **WER/SIM 不一致**: WER 和 speaker similarity 的改善方向不统一。例如 CosyVoice SFM 的 WER 3.810 vs baseline 3.513,SIM 0.931 vs 0.932 [Table 4]。作者承认"仍有改善空间" [§5.1]
3. **自适应 solver 显著加速**: 随 α 增大,NFE 和 RTF 近乎线性下降。Matcha-TTS α=5 时 Dopri(5) 加速 47.6%,NFE 从 121 降至 64 [Table 5]
4. **固定步长 solver 不受益**: 加速仅限于自适应步长 solver,因为固定步长 solver 无法利用改善的初始状态减少步数 [§5.2]
5. **CosyVoice (SFM-t)**: 仅输入 speech token 时 SMOS 显著下降 (2.66 vs 3.67),因为 ASR 训练的 semantic token 缺乏说话人信息,早期流推理的错误难以修正 [§5.1]

## 局限性

1. **SFM head 过于简单**: 当 weak generator 不稳定时 (如 StableTTS 的 reference encoder),简单的 SFM head 难以收敛和预测准确的 $t_h$、$\sigma_h^2$,需要退而使用 ID-based speaker embedding [§4.2]
2. **WER 和 SIM 改善不一致**: 自然度 (MOS) 一致提升但智能性 (WER) 和相似度 (SIM) 的改善不稳定 [§5.1, Table 4]
3. **仅适用于自适应步长 solver**: 固定步长 ODE solver (如 Euler) 无法受益于 SFM 的推理加速 [§5.2]
4. **CosyVoice 的 naive fusion**: 直接拼接 speech tokens + speaker embedding + masked mel 作为 encoder 输入,可能影响跨模态对齐 [§7]
5. **超参数 α 需要验证集搜索**: 每个模型和数据集需要独立搜索最优 α [§4.5]
6. **代码未完全开源**: 仅提供 demo 页面,无完整训练代码 [§Abstract]

## 点评

**贡献的本质**: SFM 将 DiffSinger 的 shallow diffusion 思想严格迁移到 flow matching 框架下,核心理论贡献是 Theorem 1 (CondOT 路径映射的连续性) 和 Theorem 2 (分段流的等价性)。实践上,SFM head 作为即插即用模块可直接加到现有 coarse-to-fine TTS 系统上。

**与已有方法的关系**: 
- DiffSinger (shallow diffusion): SFM 的直接灵感来源,但 DiffSinger 在离散步骤的 diffusion 中实现,SFM 在连续时间的 flow matching 中需要不同的数学工具
- PeRFlow (piecewise reflow): 将 CondOT 路径分窗做重整流加速,SFM 只取后半段,更简单但依赖 coarse 表示的质量
- Modifying flow matching (Korostik et al., 2025): 从以 coarse 输出为中心的高斯分布采样而非标准正态,采用确定性推理。SFM 的理论更严格 (正交投影 + 连续性保证)

**方法的优雅之处**: 正交投影确定时间点是一个数学上自然的选择——它找到 CondOT 路径上离 $\hat{X}^h$ 最近的点,且有 Wasserstein-2 连续性保证。$\Delta$ 缩放因子自动处理训练早期的不稳定情况。

**值得关注的问题**: WER/SIM 未能一致改善。作者将其归因于"对齐质量仍有改善空间",但更深层的原因可能是: SFM 跳过了 FM 路径的早期阶段,而这些阶段可能对建立语义一致性 (低 WER) 和说话人特征 (高 SIM) 有重要作用。这与 CosyVoice (SFM-t) 的 SMOS 急剧下降相呼应——当 coarse 表示缺乏某些信息时,跳过早期阶段会放大这些缺失。

## 可复用的 idea

1. **正交投影定位法**: 在任何有"初始估计 + 迭代精化"的框架中 (不限于 FM),用正交投影到目标空间确定初始估计的质量/位置,从而决定需要多少精化。可迁移到 diffusion-based vocoder、图像超分辨率等
2. **SFM strength α**: 训练时让模型保守 (小 $t_h$),推理时用超参数放大——这个"训练保守、推理大胆"的策略避免了训练不稳定,同时保留推理灵活性
3. **轻量 head 即插即用**: SFM head 的设计 (Conv1d + LayerNorm,基于 duration predictor) 证明了在 hidden states 基础上做简单变换就能提供有价值的中间状态,不需要复杂模块
4. **安全降级机制 (Δ 缩放)**: 当预测不可靠时自动退化到标准行为——这种设计模式可用于任何试图"走捷径"的推理加速方法

> [!review] 审阅状态
> 待审阅。本笔记由 AI agent 自动生成,status: draft。

---
检索命中: [[Conditional Flow Matching]]✓, [[CosyVoice]]✓, [[Neural Vocoder]]✓ | 过滤: [[Classifier-Free Guidance]](pending-review), [[Diffusion-based TTS]](pending-review), [[Mel Spectrogram]](pending-review) | 未命中但可能相关: 无
