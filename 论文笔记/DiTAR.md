---
type: paper
tier: deep
title: "DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation"
arxiv_id: "2502.03930"
source: "Sources/DiTAR.pdf"
authors: [Dongya Jia, Zhuo Chen, Jiawei Chen, Chenpeng Du, Jian Wu, Jian Cong, Xiaobin Zhuang, Chumin Li, Zhen Wei, Yuping Wang, Yuxuan Wang]
year: 2025
venue: "arXiv (ByteDance Seed)"
tags: [TTS, autoregressive, diffusion, continuous-representation, patchification, zero-shot, scalability]
concepts: ["[[ConditionalFlowMatching]]", "[[Classifier-FreeGuidance]]", "[[Next-TokenDiffusion]]", "[[VariationalAutoencoderforTTS]]", "[[Diffusion-basedTTS]]"]
models: ["[[模型库/NaturalSpeech2|NaturalSpeech 2]]", "[[模型库/NaturalSpeech3|NaturalSpeech 3]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: ["[[数据集/Emilia|Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[ConditionalFlowMatching]]✓, [[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]✓, [[Next-TokenDiffusion]][待确认], [[Classifier-FreeGuidance]][待确认], [[VariationalAutoencoderforTTS]][待确认], [[Diffusion-basedTTS]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: DiTAR 处于 "AR + Diffusion" 融合范式的关键节点。在此之前,Next-Token Diffusion (LatentLM, 2024) 提出了在 causal LM 每个 token 位置挂轻量 diffusion head 的方案,但受限于 causal attention 的单向依赖,在连续 token 上表现不佳 [Li et al. 2024]。另一路线 ARDiT/Transfusion 将 LM 参数复用为 diffusion 模型,但计算开销随序列增长而剧增。DiTAR 提出了第三条路: **patch 级分治** — LM 负责 patch 间预测,bidirectional DiT (LocDiT) 负责 patch 内生成,既避免了 per-token diffusion 的单向约束,又避免了全序列 diffusion 的计算膨胀。
>
> **已有认知**: CFM/flow matching 已是 TTS 主流生成范式 (CosyVoice 系列, F5-TTS, MaskGCT 等); CFG 是 diffusion/flow 条件生成的标准增强手段; VAE 作为连续 tokenizer 在 LatentLM/CLEAR 中已验证 (sigma-VAE, wav-VAE 等变体); Zero-shot TTS 当前 SOTA 在 Seed-TTS-Eval 上 WER ~1.2-1.5%, SIM ~0.75-0.82。
>
> **创新判断**: DiTAR 的核心创新不在单个组件 (VAE/CFM/CFG 均已有),而在 **架构层面的分治策略** — 用 patchification 将连续序列切分,让 causal LM 和 bidirectional DiT 各司其职。这在 LatentLM 的 per-token 路线和 ARDiT 的全序列路线之间找到了平衡点。此外,连续 LM 的温度定义 (reverse ODE 噪声引入时间点) 是一个有实用价值的新概念。
>
> 检索命中: [[ConditionalFlowMatching]], [[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]], [[Next-TokenDiffusion]], [[Classifier-FreeGuidance]], [[VariationalAutoencoderforTTS]], [[Diffusion-basedTTS]] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: Patch-based AR 框架,用 causal LM 做 patch 间预测 + bidirectional DiT (LocDiT) 做 patch 内生成,在连续语音表示上实现 SOTA 零样本 TTS 且计算量远低于 NAR 竞品
> - **路线**: 文本 phoneme → [text embedding + speech VAE latent (40Hz, dim=64)] → causal LM (patch 聚合) → LocDiT (diffusion 解码 next patch) → VAE decoder (BigVGAN) → waveform
> - **指标**: WER 1.78% / SIM 0.64 (LibriSpeech-A, Librilight 60K h, 0.6B) [Table 1]; WER 1.685% / SIM 0.735 (Seed-EN, 280K h, 1B) [Table 3]; TFLOPs ~2.75 vs NAR 37-117 TFLOPs [Table 1]; NFE=10
> - **可借鉴**: (1) Patchification 让 causal LM 处理更短序列同时为 diffusion 提供双向建模窗口; (2) LM Guidance — 仅需 2 次 diffusion head + 1 次 LM forward 的高效 CFG 变体; (3) 连续 LM 温度 = reverse ODE 噪声引入时间点,兼容 ODE solver; (4) historical patch 作为 LocDiT context 将任务从 generation 转为 outpainting
> - **局限**: 仅评估了英文和中文; VAE tokenizer 训练细节未充分公开; 未与同期 AR+Diffusion 工作 (如 LatentLM) 直接对比; 未讨论流式推理; 24kHz 采样率在高保真场景可能不足

## 核心问题

DiTAR 要解决的核心问题是: **如何在自回归框架下高质量地生成连续语音表示,同时保持合理的计算量?**

现有方案的困境 [§1]:
1. **离散 token AR + diffusion refinement** (如 VALL-E, CosyVoice): 两阶段级联导致误差累积,限制 LM 的 scaling 潜力 [论文原文]
2. **Causal LM + per-token diffusion head** (如 LatentLM/Li et al. 2024): causal attention 的单向依赖与连续 token 的局部双向相关性冲突,导致生成质量显著低于 full-attention 系统 [论文原文,§1]
3. **全序列 diffusion 复用 LM 参数** (如 ARDiT, Transfusion): 将 LM 参数用于 diffusion 计算,随序列长度和模型规模增长计算量急剧膨胀 [论文原文,§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DiTAR 的架构由三个模块组成 [§3.1.2, Fig 1]:

1. **Aggregation Encoder** (双向 attention): 将每个 patch (P 个连续 token) 聚合为单个向量。具体做法是在 patch 序列开头加一个 learnable special token (类似 BERT [CLS]),encoder 输出中该位置的 hidden state 即为 patch embedding [§3.5.2]

2. **Causal Language Model** (因果 attention): 处理由 patch embedding 组成的缩短序列,输出 h_t 作为 diffusion 解码的条件。由于 patchification,LM 输入序列长度缩短为原始的 1/P [论文原文,§3.1.1]

3. **LocDiT (Local Diffusion Transformer)** (双向 attention): 以 LM 输出 h_t 为条件,通过 diffusion 过程生成下一个 patch 的连续 token [§3.2]

三个模块均基于 Transformer,采用 Pre-Norm (RMSNorm) + RoPE [§3.5.2]。

**为什么 patchification 有效?** 作者的核心论点是: 相邻连续 token 之间具有高度相似性,存在局部双向依赖 [论文原文,§3.1.1]。causal attention 无法利用这种局部依赖,因此在 patch 内使用 bidirectional DiT,patch 间才使用 causal LM。这一分治策略同时获得了两个优势: (1) LM 处理更短序列降低计算量; (2) DiT 在 patch 内利用双向 attention 提升生成质量 [论文原文,§2]。

### 关键设计选择

**1. LocDiT 的 Historical Context (关键创新)** [§3.2]

LocDiT 不仅接收 LM 输出 h_t 作为条件,还将 **历史 patch 的 token** 作为前缀输入。这将任务从 "从条件向量生成" 转变为 "outpainting"(从已有内容向外扩展),显著提升了生成性能 [论文原文,§3.2]。

消融验证 [Table 5]: patch size=4 时,不使用 historical context WER 飙升至 22.874% (生成无法停止),使用 1 个 historical patch 后 WER 降至 1.736%。这证明 historical context 对 LocDiT 至关重要 [agent 解读: 仅靠 LM 输出的隐式粗特征不足以锚定生成的起始状态]。

**隐式 coarse-to-fine** [§3.2]: 作者指出,LM 将每个 patch 压缩成聚合 embedding 本质上是一种隐式的粗特征,LocDiT 将其展开为细粒度 token 的过程等价于 coarse-to-fine,但在端到端框架内完成,避免了多阶段系统的累积误差 [论文原文]。

**2. LM Guidance (高效 CFG 变体)** [§3.3]

标准 CFG 需要两次完整模型 forward (条件+无条件)。DiTAR 提出了 LM Guidance: 仅需 **1 次 LM forward + 2 次 LocDiT forward**。

- 训练: 以 10% 概率将 LM 输出 h_i 替换为全零向量 h_∅ [§3.5.2]
- 推理: ε̃(z_{i,t}, h_i) = (1+w)·ε(z_{i,t}, h_i) - w·ε(z_{i,t}, h_∅) [Eq 2]

**为什么高效?** 因为 LM 输出 h_i 已编码了全部历史信息 (x_0, ..., x_i),unconditional 分支只需将 h_i 换为 h_∅ 再跑一次轻量 LocDiT,不需要重新跑整个 LM [论文原文,§3.3]。对比离散 LM 的 CFG 需要两次完整 LM 计算,这节省了约一半的 LM 计算量 [agent 解读]。

消融 [Fig 4]: 不使用 guidance (w=0) 时 WER 和 SIM 均显著退化; w=1-2 范围最优; 即使 NFE=2 配合 guidance 也能保持良好性能。

**3. 连续 LM 的温度定义** [§3.4]

离散 LM 的温度通过 softmax 缩放控制随机性,但连续 LM 缺乏直接对应物。DiTAR 将温度 τ ∈ [0,1] 定义为 **reverse ODE 求解过程中引入随机噪声的时间点** [§3.4]:

- τ=1: 标准 ODE 采样 (x_1 ~ N(0,I),从纯噪声开始)
- τ=0: 完全确定性 (x_1 ≡ 0,即标准高斯的众数)
- 0<τ<1: 先从 x_1=0 确定性求解到 τ,在 τ 处用 forward process 注入噪声,再继续求解 [Eq 3-4]

**为什么优于 SDE 温度?** Li et al. [34] 基于 DDPM reversed process 定义温度,依赖 SDE solver,需要较多步数; DiTAR 的定义兼容更快的 ODE solver (如 DDIM) [论文原文,§3.4]。

消融 [Table 6]: 低温 (τ=0) 略优 WER,高温 (τ=1) 略优 SIM。作者解释: 模拟未见说话人音色需要更大多样性 (高温有利),发音准确性需要更大确定性 (低温有利) [论文原文,§4.4.1]。

**4. Patch Size 的平衡** [§4.3, Fig 3]

Patch size 过小 (P=1) 退化为 per-token diffusion,失去 bidirectional 优势; 过大 (P=8) 使 LocDiT 成为瓶颈,需要更多参数。P=2-4 是最优区间 [Fig 3]。默认使用 P=4。

### 训练策略

- **Continuous Tokenization**: VAE (Conv encoder + BigVGAN decoder, 对抗训练) 将 24kHz waveform 压缩为 40Hz, dim=64 的 latent [§3.5.1]
- **Diffusion Loss**: Variance-preserving, cosine schedule, v-prediction + conditional flow matching loss [§3.5.3, Eq 5-7]
- **推理 Sampler**: DDIM (本质是 SNR 空间的 Euler ODE solver),NFE=10 [§3.5.3]
- **Stop Prediction**: LM 输出上加一层 FC 做二分类预测 [§3.5.4]
- **训练规模**: 16× A100, batch=15K tokens/GPU, 0.5M steps, AdamW lr=1e-4 [§A.1]; 1B 版本 32× A100

## 实验

| 指标 | 本文 (0.6B, Librilight) | VALL-E | NaturalSpeech 3 | Voicebox | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER(%)↓ | **1.78** | 6.11 | 1.81 | 2.14 | LibriSpeech-A | [Table 1] |
| SIM↑ | 0.64 | 0.47 | **0.67** | 0.48 | LibriSpeech-A | [Table 1] |
| UTMOS↑ | 4.15 | 3.68 | **4.30** | 3.73 | LibriSpeech-A | [Table 1] |
| TFLOPs↓ | **~2.75** | ~2.99 | ~8.92 | ~60.89 | LibriSpeech-A | [Table 1] |

| 指标 | 本文 (0.6B, Emilia) | MaskGCT | E2TTS | F5TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER(%)↓ | **2.39** | 2.72 | 2.95 | 2.42 | LibriSpeech-B | [Table 1] |
| SIM↑ | 0.67 | **0.69** | **0.69** | 0.66 | LibriSpeech-B | [Table 1] |
| UTMOS↑ | **4.22** | 3.90 | 3.56 | 3.88 | LibriSpeech-B | [Table 1] |
| TFLOPs↓ | **~2.75** | ~116.66 | ~56.46 | ~37.36 | LibriSpeech-B | [Table 1] |

| 指标 | 本文 (1B, 280Kh) | Seed-TTS_DiT | CosyVoice 2 | F5TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER(%)↓ | **1.685** | 1.733 | 2.57 | 1.83 | Seed-EN | [Table 3] |
| SIM↑ | **0.735** | 0.790 | 0.652 | 0.71 | Seed-EN | [Table 3] |
| WER(%)↓ | **1.023** | 1.178 | 1.45 | 1.56 | Seed-ZH | [Table 3] |
| SIM↑ | 0.753 | **0.809** | 0.748 | 0.76 | Seed-ZH | [Table 3] |

**主观评估** [Table 2, LibriSpeech-B]:
- DiTAR N-MOS 3.69 (vs E2TTS 3.27, F5TTS 3.36)
- DiTAR CMOS 0.00 (与 GT 持平,E2TTS -0.32, F5TTS -0.04)
- DiTAR Q-MOS **3.87** 超过 GT (3.61)

**Scaling 行为** [§4.2, Fig 2]:
- 模型从 0.1B→1B: WER/SIM 持续改善,无饱和迹象
- 数据从 20Kh→280Kh: 同样持续改善
- 组件 scaling [Table 4]: LM 和 LocDiT 受益于 scaling,Encoder 增大几乎无影响

**效率** [§4.4.2, Table 7, Fig 6]:
- 低 batch: NAR 吞吐更高 (无 AR 计算)
- 高 batch (>200): DiTAR 吞吐显著超过 NAR (FLOPs 低,瓶颈在 memory bandwidth)
- 延迟: DiTAR 始终低于 NAR (AR 特性,首帧即可输出) — P=4 时 latency 0.066s vs NAR 0.37s (batch=1) [Table 7]
- RTF: NAR batch=1 时 RTF 0.037 最优; DiTAR P=2 RTF 0.66 (batch=1), P=4 RTF 1.28 (batch=1) [Table 7]

## 局限性

1. **语言覆盖有限**: 仅在英文和中文上评估,未验证多语言泛化能力 [agent 解读]
2. **VAE 细节不足**: VAE tokenizer 的架构和训练细节 (如 beta, reconstruction quality) 未充分报告,仅提到 Conv encoder + BigVGAN decoder [§3.5.1] [agent 解读]
3. **缺乏同范式对比**: 未与 LatentLM (per-token diffusion) 在相同设置下直接对比,仅引用了 Li et al. [34] 的结论 [agent 解读]
4. **流式推理未讨论**: AR 框架天然支持流式,但论文未讨论 chunk-wise 流式策略和实际延迟 [agent 解读]
5. **Patch 边界不连续风险**: patch 之间的边界处理仅依赖 historical context,未分析是否存在 patch 边界 artifact [agent 解读]
6. **推理速度**: batch=1 RTF=1.28 (P=4) 仍大于 1,实时性依赖 batch 或 P 调整 [Table 7]

## 点评

DiTAR 的核心贡献是 **架构层面的分治思想**,而非单一组件的创新。将 patchification 从纯粹的计算优化手段提升为 **双向建模的使能机制**,这个 insight 比技术本身更有价值 — 它解释了为什么 per-token diffusion head (LatentLM) 表现不佳: 不是 diffusion 不行,而是 patch size=1 的 causal attention 约束了生成质量。

LM Guidance 是一个优雅的工程设计: 利用 LM 输出已编码全部历史的性质,将 unconditional branch 的计算从 LM 卸载到轻量 LocDiT,实现了几乎免费的 guidance。

温度定义虽然简洁,但实用价值需要更多验证 — Table 6 显示不同温度的性能差异不大 (WER 1.62-1.69%),说明 ODE 采样本身已足够稳定,温度更多是一个 fine-tuning knob。

**值得关注的数字**: DiTAR 的 TFLOPs (~2.75) 比 NAR 系统低 14-43 倍 [Table 1],但这主要来自 patchification 缩短序列和 LocDiT 轻量化,而非算法本质优势。当 batch 增大时 DiTAR 的吞吐优势才真正体现 [Fig 6]。

**与后续工作的关系**: SemaVoice (2026) 直接采用了 patch-wise LocDiT 架构 (L=2, 含 previous-patch conditioning),证明这个设计被验证和复用。HoliTok (2026) 的 AR+DiT 下游架构也参考了 DiTAR 范式。

## 可复用的 idea

1. **Patchification 作为双向建模窗口**: 当 causal LM 处理连续信号时,将序列切分为 patch,patch 内用 bidirectional attention — 这个思路可推广到任何需要兼顾因果性和局部双向性的序列建模任务
2. **Historical context 将 generation 转为 outpainting**: 给 decoder 前缀输入历史观测,比纯条件生成更容易,因为 decoder 能看到真实的局部上下文而非仅有隐式条件
3. **LM Guidance**: 对任何 "LM backbone + diffusion head" 架构都适用 — 只需在训练时 drop LM output,推理时用全零向量做 unconditional branch
4. **连续 LM 温度 = ODE 噪声引入点**: 提供了一种与 ODE solver 兼容的采样多样性控制方案,可迁移到其他 diffusion 生成系统
