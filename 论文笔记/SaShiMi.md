---
type: paper
tier: deep
title: "It's Raw! Audio Generation with State-Space Models"
arxiv_id: "2202.09729"
source: "Sources/SaShiMi.pdf"
authors: [Karan Goel, Albert Gu, Chris Donahue, Christopher Ré]
year: 2022
venue: "arXiv preprint (ICML 2022 workshop)"
tags: [SSM, state-space-model, S4, audio-generation, waveform-modeling, autoregressive, multi-scale, Hurwitz-stability, WaveNet-alternative, unconditional-generation, diffusion-backbone]
concepts: ["[[NeuralVocoder]]", "[[DiffusionModel]]", "[[Diffusion-basedVocoder]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 2 个待确认: [[NeuralVocoder]]✓, [[DiffusionModel]][待确认], [[Diffusion-basedVocoder]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[NeuralVocoder]]✓ | 过滤: [[DiffusionModel]](pending-review), [[Diffusion-basedVocoder]](pending-review) | 未命中但可能相关: 无
>
> **谱系定位**: SaShiMi 是 SSM (State Space Model) 家族在音频波形生成领域的首个系统性验证。它基于 S4 (Structured State Spaces for Sequence Modeling, Gu et al. 2022) 构建,是连接 S4 基础理论与后续 Mamba (2023) 音频应用的关键桥梁。在 KB 中,[[NeuralVocoder]] 记录的 vocoder 演进线覆盖了 SaShiMi 的主要对比对象: WaveNet (AR CNN vocoder, 2016)、SampleRNN (AR RNN, 2017) 和 DiffWave (diffusion-based, 2020)。SaShiMi 定位为 WaveNet 的直接替代 — 不仅在 AR 设置中超越 WaveNet,还作为 DiffWave 的 drop-in backbone 替换 WaveNet 提升非 AR 生成质量。
>
> **已有认知**: [[NeuralVocoder]] 中 WaveNet 的核心局限已被记录: 感受野受限 (最多 ~3s),实际只利用几十毫秒上下文。DiffWave 作为 diffusion vocoder 代表作在 [[Diffusion-basedVocoder]] 中有详细记录 (MOS 4.44, 条件+无条件+类别条件生成)。KB 中尚无 SSM/S4 相关概念页,SaShiMi 是填补这一空白的起点。
>
> **创新判断**: SaShiMi 的核心创新不在生成模型本身 (仍是 AR 或 diffusion),而在 backbone 架构 — 用 SSM 替代 dilated CNN/RNN,同时解决 CNN 的有限感受野和 RNN 的不可并行化问题。这一架构思路后来被 Mamba (Gu & Dao, 2023) 大幅推进,SaShiMi 的 SC09 SOTA (FID 1.99) 被 Mamba 刷新至 FID 0.67。

> [!summary] 速查
> - **一句话**: 首个基于 SSM 的音频波形生成架构,通过 Hurwitz 稳定性修复和多尺度池化,在 AR 和 non-AR 设置中均超越 WaveNet/SampleRNN,证明 SSM 是音频建模的强有力 backbone
> - **路线**: 8-bit 量化波形 → 嵌入 → 三层多尺度 S4 block (4x 下采样 x2 层,H→2H→4H) → 上采样融合 → softmax 输出下一样本分布; non-AR 模式下替换 DiffWave 中的 WaveNet backbone
> - **指标**: AR 音乐: NLL 1.294 vs WaveNet 1.449, MOS musicality 3.11 vs 2.71 (YouTubeMix) [Table 4]; AR 语音: IS 4.12 vs 2.27, MOS quality 3.29 vs 1.59 (SC09) [Table 6]; non-AR: DiffWave+SaShiMi FID 1.42 vs DiffWave 1.92 (SC09) [Table 6]
> - **可借鉴**: (1) Hurwitz 稳定性约束 (Λ-pp* 替代 Λ+pq*) 可迁移到任何需要 CNN→RNN 模式切换的 SSM; (2) reshape+linear 实现的因果池化是一种极简但有效的多尺度设计; (3) "在 non-AR 框架中 drop-in 替换 backbone" 的验证范式可复用于评估新架构的通用性
> - **局限**: 仅验证无条件生成,未测试条件生成 (如 TTS vocoding); 8-bit 量化限制音质上限; SC09 仅 1s 片段,未验证长语音生成; 训练仍需 V100 单卡 ~1M steps; 后续 Mamba 已大幅超越其 SC09 结果

## 核心问题

原始音频波形建模面临三重挑战 [§1]:
1. **全局一致性**: 波形每秒含数万样本点,生成连贯音频需要建模极长距离依赖 (音乐中跨小节的节奏/和声结构)
2. **计算效率**: 需要并行训练 + 快速推理 (AR 和 non-AR 两种模式)
3. **样本效率**: 模型需具备适合高采样率波形的归纳偏置

现有架构各有致命短板 [§1]: CNN (WaveNet) 受限于有限感受野 (~4K 样本 = ~0.25s),无法建模全局结构; RNN (SampleRNN) 无法并行训练; Transformer 对 ~1M 长度序列计算不可行。

**核心洞察** [agent 解读]: S4 的 CNN/RNN 对偶性恰好填补了这个缺口 — 训练时用卷积模式并行,推理时用循环模式逐步生成。但直接使用 S4 有一个被忽视的问题: 训练时不需要矩阵幂次运算 (卷积模式),但推理时循环模式需要反复乘以离散化状态矩阵 A,若 A 的特征值不在单位圆内,循环会发散。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SaShiMi 由两个核心组件构成 [§4, Fig 1]:

1. **S4 Block**: 基础构建单元
   - LayerNorm → S4 层 → GELU 激活 → 线性投影 → 残差连接
   - 之后接 position-wise FFN: LayerNorm → Linear (d→2d) → GELU → Linear (2d→d) → 残差连接
   - 设计参照 Transformer 的 self-attention + FFN 模式 [§4.2, Appendix A]

2. **多尺度架构**: U-Net 风格的三层结构
   - **Tier 1** (顶层): 处理原始采样率信号,shape = (H, L)
   - **Tier 2** (中层): 4x 下采样后,shape = (2H, L/4)
   - **Tier 3** (底层): 再 4x 下采样,shape = (4H, L/16)
   - 每层含 8 个 S4 block (默认配置) [Appendix C.2]
   - 下采样和上采样通过 reshape + 线性投影实现,上采样时加时间步偏移以保持因果性 [§4.2, Fig 1 Right]
   - 层间有长距离残差连接帮助信号传播 [Fig 1 Center]

**模型维度**: H=64, 池化因子 p=4, 扩展因子 q=2 [Appendix C.2]。默认 8 层/tier 配置下,总参数量 4.05M — 比 SampleRNN (35M) 小近 9 倍,比 WaveNet (4.24M) 略小 [Table 5]。

### 关键设计选择

#### 1. Hurwitz 稳定性修复 (核心理论贡献)

**问题诊断** [§4.1]: S4 将状态矩阵参数化为 DPLR 形式 A = Λ + pq*。训练时使用卷积模式 K = (CB, CAB, CA^2B, ...) 计算输出,不需要反复乘以离散化矩阵 A,因此即使 A 不是 Hurwitz 矩阵也不会出问题。但推理时切换到循环模式 h_k = A*h_{k-1} + B*x_k,需要反复乘以 A — 若 A 的特征值在单位圆外 (对应连续时间 A 不是 Hurwitz 的),循环会指数发散 [§4.1]。

**实验验证**: 作者发现训练后 S4 的 A 矩阵通常变成非 Hurwitz 的 — 即训练是稳定的,但无法用于自回归生成 [§4.1]。

**解决方案** [§4.1, Proposition 4.3]: 将参数化从 A = Λ + pq* 改为 A = Λ - pp* (绑定 p=q 并反转符号)。关键推导:
- -pp* 是负半定矩阵 (所有特征值 ≤ 0) [论文原文]
- 若 Λ 的所有条目实部为负,则 A + A* = (Λ + Λ*) - 2pp* 也是负半定的 [论文原文]
- 这保证 A 是 Hurwitz 矩阵 (所有特征值实部为负) [Proposition 4.3]
- 进一步验证: HiPPO 初始化矩阵 (LegS, LegT, LagT) 都满足 Λ-pp* 形式 [Proposition 4.1],因此新参数化不丢失 HiPPO 的长程记忆能力

**实际效果**: 新参数化 NLL 1.419 vs 原始 1.420 (几乎无性能损失),但保证了生成稳定性 [§5.2, Fig 3]。实践中不约束 Λ 也学到了稳定解,但理论保证了 worst case 安全 [§4.1]。

**为什么这个修复重要** [agent 解读]: 这不是一个 engineering trick,而是揭示了 SSM 训练-推理模式不对称的根本问题。任何 SSM 如果要在自回归推理中使用循环模式,都必须确保离散化状态矩阵的谱半径 ≤ 1。SaShiMi 的修复方案 (负半定约束) 后来被后续 SSM 工作 (S5, S4D, Mamba) 沿用。

#### 2. 多尺度池化 vs 等距 S4 堆叠

**动机** [§4.2]: 直接堆叠 S4 层 (isotropic) 虽然理论上有无限感受野,但实际效率和性能不如多尺度结构 [论文原文]。

**对比实验** [Table 5 Bottom]:
- Isotropic S4 (4 层, 2.83M): NLL 1.429, 训练速度 1900s/epoch
- Isotropic S4 (8 层, 5.53M): NLL 1.524 (反而变差), 训练速度 3700s/epoch
- SaShiMi (4 层, 2.21M): NLL 1.341, 训练速度 340s/epoch

等距 8 层 S4 性能反而下降 [agent 解读], 可能因为在全分辨率上堆叠太深导致优化困难,而多尺度结构通过降采样减少了底层序列长度,既节省计算又让底层 S4 能以更少步数覆盖更大时间跨度。

#### 3. 双向 S4 用于 non-AR 设置

对于不需要因果性的 non-AR 任务 (如 DiffWave backbone),简单地拼接正向和反向两个独立 S4 的输出,再过线性层 [§4.2]:
```
y = Linear(Concat(S4(x), rev(S4(rev(x)))))
```
双向 vs 单向 SaShiMi 在 DiffWave 中的对比: FID 1.70 vs 2.70 (小模型, 500K steps) [Table 7] — 双向版本显著更好 [论文原文]。

### 训练策略

- **输入**: 8-bit 量化波形 (线性量化用于 Beethoven,mu-law 用于 YouTubeMix/SC09) [§5, Table 1]
- **上下文长度**: SaShiMi 训练在 128K 样本 (8s @ 16kHz) — WaveNet 仅 4K (~0.25s) [Table 2]
- **输出**: 256-way softmax 预测下一个量化级别
- **优化**: S4 参数 (Λ, C) 使用 lr=0.001,其他参数冻结 (pp*, B, dt) 以简化训练 [Appendix C.2]
- **训练时长**: Beethoven 1M steps, YouTubeMix 600K steps, SC09 1.1M steps [Appendix C.2]
- **硬件**: 单 V100 GPU (AR); 8x A100 (diffusion) [Appendix C.2]
- **SC09 特殊处理**: 使用 GLU 替代 GELU 改善 NLL 和样本质量 [Appendix C.2]

## 实验

### AR 音乐生成

| 指标 | SaShiMi | WaveNet | SampleRNN | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| NLL (BPB) | **0.946** | 1.032 | 1.125 | Beethoven | [Table 2] |
| NLL (BPB) | **1.294** | 1.449 | 1.723 | YouTubeMix | [Table 4] |
| MOS (musicality) | **3.11 +/- 0.09** | 2.71 +/- 0.08 | 1.82 +/- 0.08 | YouTubeMix | [Table 4] |
| MOS (fidelity) | 2.84 +/- 0.09 | 2.91 +/- 0.08 | 2.98 +/- 0.08 | YouTubeMix | [Table 4] |

关键观察 [§5.1]: 三种方法在音质 fidelity 上相近,但 SaShiMi 在 musicality (全局结构一致性) 上显著领先 +0.40 MOS — 这正是长距离依赖建模能力的体现。

**上下文长度消融** [Table 3]: 固定计算量下,从 1s 扩展到 8s 上下文,NLL 从 1.364 降到 1.007 (200K steps),证明 SaShiMi 确实能利用更长上下文信息。

### AR 语音生成 (SC09)

| 指标 | SaShiMi | WaveNet | SampleRNN | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| NLL (BPB) | **1.891** | 1.925 | 2.042 | SC09 | [Table 6] |
| FID ↓ | **1.99** | 5.08 | 8.96 | SC09 | [Table 6] |
| IS ↑ | **4.12** | 2.27 | 1.71 | SC09 | [Table 6] |
| MOS (quality) | **3.29 +/- 0.07** | 1.59 +/- 0.06 | 1.18 +/- 0.04 | SC09 | [Table 6] |
| MOS (intelligibility) | **3.53 +/- 0.04** | 1.72 +/- 0.03 | 1.37 +/- 0.02 | SC09 | [Table 6] |
| Agreement (kappa) | **0.832** | 0.408 | 0.321 | SC09 | [Table 6] |
| Params | 4.1M | 4.2M | 35.0M | SC09 | [Table 6] |

关键观察: SaShiMi 是首个在 SC09 无条件生成上产出人类可辨识语音的 AR 模型 [§5.3] — MOS quality 3.29 vs WaveNet 1.59 (2x 提升),annotator agreement kappa 0.832 (接近 ground truth 0.921)。

### Non-AR (DiffWave + SaShiMi)

| 指标 | DiffWave+SaShiMi | DiffWave (WaveNet) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| FID ↓ | **1.42** | 1.92 | SC09 | [Table 6] |
| IS ↑ | **5.94** | 5.26 | SC09 | [Table 6] |
| mIS ↑ | **69.17** | 51.21 | SC09 | [Table 6] |
| MOS (quality) | **4.20 +/- 0.06** | 4.03 +/- 0.06 | SC09 | [Table 6] |
| MOS (intelligibility) | **4.33 +/- 0.03** | 4.15 +/- 0.03 | SC09 | [Table 6] |

关键观察 [§5.3]: 零调参 drop-in 替换 — SaShiMi 替换 DiffWave 的 WaveNet backbone 后,所有指标全面提升,且对超参更鲁棒 (小模型 WaveNet 失败,SaShiMi 仍然 work) [Table 7]。

### 效率对比

| 指标 | SaShiMi (8层) | WaveNet-1024 | SampleRNN (3层) | 出处 |
| --- | --- | --- | --- | --- |
| 上下文长度 | 128,000 | 4,092 | 1,024 | [Table 2, 5] |
| 参数量 | 4.05M | 4.24M | 35.03M | [Table 5] |
| 训练吞吐量 | 129K samples/s | 182K samples/s | 116K samples/s | [Table 5] |
| 训练时间/epoch | 875s | 1435s | 850s | [Table 5] |

SaShiMi-2 (最小配置, 1.29M) 在 NLL (1.446) 上已匹配 WaveNet-1024 (1.449),且峰值吞吐量 3x/5x 高于 WaveNet/SampleRNN [Appendix B, Fig 5]。

## 局限性

1. **仅验证无条件生成** [agent 解读]: 未测试 mel-conditioned vocoding (TTS 最常见场景) 或 text-conditioned 生成。论文自身承认 AR 模型在有条件信息时更容易生成连贯语音 [§5.3 footnote 1]。
2. **8-bit 量化上限**: 所有实验使用 8-bit 量化 (256 级) [Table 1]。[agent 解读] 这远低于现代 TTS 的 16-bit/24-bit 标准,量化级数对音质的影响论文未讨论。
3. **数据集规模有限**: Beethoven 10h, YouTubeMix 4h, SC09 5.3h [Table 1] — 远小于现代 TTS 训练数据规模 (数万小时)。
4. **SC09 仅 1s 片段**: 无条件语音生成只验证了 1s 单字生成,未验证长语音的全局一致性 (音乐任务验证了 16s)。
5. **S4 参数部分冻结**: 为简化实验,仅训练 Λ 和 C,冻结 pp*, B, dt [Appendix C.2]。作者承认使用完整 Hurwitz 参数化 (约束 Λ 实部为负) 可进一步提升性能 [Appendix C.2]。
6. **被后续工作大幅超越**: Mamba (2023) 在 SC09 上 FID 0.67 vs SaShiMi 1.99,IS 7.33 vs 4.12 — 选择性 SSM 机制的引入对音频生成同样重要。

## 点评

SaShiMi 是 SSM 进入音频生成领域的奠基工作,其价值不仅在于具体的性能数字 (已被 Mamba 超越),而在于三个关键洞察:

**1. SSM 的 CNN/RNN 对偶性天然适配音频 AR 建模** [agent 解读]: 这个洞察被后续所有 SSM 音频工作沿用 — 训练时卷积并行,推理时循环逐步生成。在 WaveNet (只有 CNN 模式,推理需特殊缓存) 和 SampleRNN (只有 RNN 模式,训练慢) 之间,SSM 提供了一个干净的统一。

**2. Hurwitz 稳定性是 SSM 实用化的隐藏前提** [agent 解读]: 这一发现揭示了 SSM 理论 (S4 的 HiPPO 初始化保证训练稳定) 与工程实现 (推理需要循环稳定) 之间的 gap。后续 S4D、S5、Mamba 都继承了这一约束 (Mamba 使用 A_n = -(n+1) 初始化确保负实部)。

**3. "Drop-in backbone replacement" 验证范式** [agent 解读]: 将 SaShiMi 直接替换 DiffWave 的 WaveNet backbone 这一实验设计很聪明 — 通过控制其他变量,干净地证明架构优势。这一范式后来被广泛采用 (如用 Mamba 替换 Transformer backbone)。

**谱系定位**: SaShiMi 处于 S4 (理论基础, 2022.01) → SaShiMi (音频应用, 2022.02) → S4D/S5 (改进, 2022) → Mamba (选择性 SSM, 2023.12) 的演进链中。它是 Stanford/Cartesia 谱系的早期关键节点,证明了 SSM 在音频领域的可行性,为后续 Mamba-based 音频系统 (如 Cartesia 的 Sonic) 奠定了基础。

## 可复用的 idea

1. **Hurwitz 约束作为 SSM 安全网**: 任何使用 SSM 做自回归推理的系统都应检查/约束离散化状态矩阵的谱半径。具体手段: 参数化为 Λ-pp* + 约束 Λ 实部为负 (通过 -exp 函数)。
2. **reshape+linear 因果池化**: 比 strided convolution 更简单的多尺度方案 — 将 (T, H) reshape 为 (T/p, pH) 再线性投影到 (T/p, qH)。上采样时加时间步偏移确保因果性。
3. **SSM 的双向扩展用于 non-AR**: 简单拼接正反两个方向的 SSM 输出,适用于任何需要全局上下文的 non-AR 任务。
4. **AR 模型的拒绝采样评估**: 生成 5120 样本后按 likelihood 排序,去掉最低 40% 和最高 5%,用中间样本评估 — 这种 rejection sampling 评估策略可减少极端样本对自动指标的干扰 [Appendix C.3.1]。
5. **长上下文 = 全局一致性**: 将 AR 训练上下文从 4K 扩展到 128K 对 musicality 提升远大于 fidelity — 这一发现对音频/音乐生成中选择上下文窗口大小有参考价值。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节包含 Hurwitz 修复的完整因果链 (问题→诊断→解决→验证),多尺度架构的 WHY 清晰 |
> | 可信赖 | pass | 数字标注覆盖率高,关键 claim 均有 Table/Figure 出处 |
> | 可区分 | pass | [论文原文] 和 [agent 解读] 标注充分 |
> | 可定位 | pass | KB 背景含 S4→SaShiMi→Mamba 谱系,与 NeuralVocoder 家族定位明确 |
> | 不污染 | pass | 未建议创建不确定的概念页 |
> 
> Issues: 2 (high: 0, medium: 2, low: 0)
> 详见 `_review/SaShiMi-review.yml`

---

检索命中: [[NeuralVocoder]]✓ | 过滤: [[DiffusionModel]](pending-review), [[Diffusion-basedVocoder]](pending-review) | 未命中但可能相关: 无
