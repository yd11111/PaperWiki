---
type: paper
tier: deep
title: "Erasing Your Voice Before It's Heard: Training-Free Speaker Unlearning for Zero-Shot Text-to-Speech"
arxiv_id: "2601.20481"
source: "Sources/TruS.pdf"
authors: [Myungjin Lee, Eunji Shin, Jiyoung Lee]
year: 2026
venue: "arXiv"
tags: [machine-unlearning, voice-privacy, zero-shot-TTS, speaker-identity, activation-steering, training-free, flow-matching, DiT]
concepts: ["[[Conditional Flow Matching]]", "[[Speaker Embedding]]", "[[Speaker Verification]]", "[[Speech Factorization]]", "[[Voice Cloning Taxonomy]]"]
models: []
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认 + 2 个待确认实体页: [[Speaker Embedding]]✓, [[Conditional Flow Matching]]✓, [[Speech Factorization]]✓, [[Zero-shot Speech Synthesis]]✓, [[Speaker Verification]][待确认], [[Voice Cloning Taxonomy]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文延续 [[论文笔记/Speaker Identity Unlearning|Kim et al. (ICML 2025)]] 开辟的 ZS-TTS speaker unlearning 方向,但提出了根本不同的范式: 从"重训练模型权重"转向"推理时激活引导"。KB 中 [[Voice Cloning Taxonomy]][待确认] 记录的 ZS-TTS 防护链条为: 被动检测 (ASVspoof) → 主动扰动 (SafeSpeech) → 水印溯源 (TraceableSpeech) → 模型级遗忘 (Kim et al.)。TruS 在模型级遗忘之上增加第五环: **推理时干预** -- 不修改模型权重,仅在生成时动态抑制身份激活。
>
> **已有认知**: [[Speaker Embedding]]✓ 记录了 speaker embedding 在 ZS-TTS 中作为身份载体的角色,TruS 的核心洞察正基于此 -- speaker identity 编码在 DiT 隐层激活的结构化方向中,可以通过 steering vector 定向抑制。[[Conditional Flow Matching]]✓ 记录了 F5-TTS 使用的 OT-CFM 框架,TruS 直接在 F5-TTS 的 DiT blocks 上操作,steering 发生在 flow step 的去噪过程中。[[Speech Factorization]]✓ 记录了 speaker-content 解耦的主流方法(GRL/信息瓶颈/self-distillation),TruS 提供了一种全新视角: 不在训练时学解耦表示,而在推理时通过 activation steering 实现"按需解耦"。[[Speaker Verification]][待确认] 记录了 ECAPA-TDNN 用于 SIM 计算的标准做法,本文使用 ECAPA-TDNN 评估 unlearning 效果(SIM 越低越好),并复用 Kim et al. 提出的 spk-ZRF 指标。
>
> **创新判断**: 相对于已有的 [[论文笔记/Speaker Identity Unlearning|Kim et al.]] 的 SGU/TGU 方法(48-430 GPU 小时重训练),TruS 实现了: (1) 零训练开销; (2) 首次支持 unseen opt-out speakers(Kim et al. 仅支持 seen speakers); (3) 从 NLP activation steering 迁移到 TTS 的首次成功应用。与 EmoSteer-TTS [19] 的区别: EmoSteer 用固定 top-k channel 选择操控 prosody,TruS 用动态 layer-step 选择操控 identity。

检索命中: [[Speaker Embedding]], [[Conditional Flow Matching]], [[Speech Factorization]], [[Zero-shot Speech Synthesis]] | 过滤: [[Speaker Verification]](pending-review), [[Voice Cloning Taxonomy]](pending-review) | 未命中但可能相关: [[Anti-spoofing and Deepfake Detection]]

> [!summary] 速查
> - **一句话**: 首个 training-free 的 ZS-TTS speaker unlearning 框架,通过推理时 activation steering 抑制 opt-out 说话人身份,零训练开销且支持 unseen speakers
> - **路线**: opt-out 语音 → DiT 隐层激活 → 与 retain speakers 的 ID-prototype 对比 → 动态选择干预层/步 → 投影减去身份方向 → 身份抑制后的语音
> - **指标**: SIM-SO 0.477 (↓0.180 vs F5-TTS 0.657), WER-SO 3.25% (优于 SGU 3.70/TGU 4.03); unseen SIM-UO 0.488 (↓0.180); 零训练小时 vs SGU 48h/TGU 430h [Table 1,2]
> - **可借鉴**: (1) ID-prototype 构建: 用 N=30 个 retain speakers 的 FFN 输出平均值作为 identity-neutral 锚点; (2) 动态 layer-step 选择: τ=µ+kσ 阈值 + 步级过滤,避免过度干预; (3) 一个投影减法公式即可抑制身份信息,设计极简
> - **局限**: (1) 仅在 F5-TTS (DiT-based) 上验证,未验证 AR-based TTS; (2) WER 有轻微上升(3.25 vs 1.95 原始); (3) 需预建 ID-prototype(N=30 utterances),实际部署需维护 retain speaker pool; (4) α=1.2 为经验值,缺乏理论指导

## 核心问题

1. **为什么不能靠删除训练数据来保护说话人?** 现代 ZS-TTS 模型通过 in-context learning 泛化到未见说话人,因此即使移除特定说话人数据重训,模型仍可能合成相似声音。重训练方法 (SGU/TGU) 还面临每新增一个 opt-out 请求就需重训的可扩展性问题 [§1]。

2. **为什么可以在推理时抑制身份?** 关键假设: speaker identity 编码在 TTS 模型隐层表示的结构化方向中 [§2.1]。如果 identity 是可识别的方向而非弥散地分布在所有维度上,则可以通过定向投影减法将其消除 [论文原文]。

3. **为什么不是所有层都需要干预?** Fig. 3 显示 cosine similarity between ID-prototype and target activation 在不同层/不同 flow step 动态变化: 浅层在后期 step 相似度降低,深层在早期 step 相似度降低 [§2.3]。全层干预 (all) 虽能略降 SIM-SO,但 WER 大幅恶化 [Table 4],证明过度干预破坏了语言内容。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TruS 是一个 plug-and-play 的推理时框架,不修改底层 TTS 模型权重。基于 F5-TTS (DiT-based flow matching model) 实现,但论文声称可推广到其他 DiT-based TTS 架构 [§2.1]。

工作流分两阶段:
1. **离线预计算** (一次性): 用 N 个 retain speakers 的语音,在每个 DiT block 的每个 flow step 提取 FFN 输出,平均得到 ID-prototype $P_{Ret}^{(\ell,t)}$ [Eq. 1]
2. **在线 steering** (每次推理): 对每个 opt-out speaker 的推理请求,计算 steering vector,动态选择干预点,投影减法抑制身份

### 关键设计选择

**设计 1: ID-prototype 的构建方式**

选择 DiT blocks 的 FFN 输出(而非 attention 输出或整体 block 输出)作为特征提取点。原因: FFN 在非线性 channel mixing 后包含强烈的 timbre/identity 信号,而 attention 更多处理序列级依赖 [论文原文,引用 [27]]。[agent 解读] 这与 LLM interpretability 中"FFN 存储知识、attention 做路由"的观点一致。

$P_{Ret}^{(\ell,t)} = \frac{1}{N} \sum_{n=1}^{N} X_{Ret(n)}^{(\ell,t)}$ [Eq. 1]

N=30 个不同说话人各一条 utterance。[agent 解读] 每个 utterance 来自不同说话人是关键 -- 平均后 speaker-specific 信息被消去,保留的是 speaker-neutral 的"平均身份锚点"。

**设计 2: Identity-specific steering vector**

给定 opt-out speaker 的激活 $X_{Opt}^{(\ell,t)}$,steering vector 定义为与 ID-prototype 差异的 L2 归一化方向:

$S^{(\ell,t)} = \frac{X_{Opt}^{(\ell,t)} - P_{Ret}^{(\ell,t)}}{\|X_{Opt}^{(\ell,t)} - P_{Ret}^{(\ell,t)}\|_2}$ [Eq. 2]

[论文原文] $S^{(\ell,t)}$ 代表 target speaker 在 latent space 中的 step-wise identity-related direction [§2.2]。[agent 解读] L2 归一化确保 steering 强度由独立参数 α 控制,而非由激活的幅度决定。

**设计 3: 动态 layer-step 选择**

核心洞察: "not all layers contribute equally to maintain speaker identity" [论文原文,§2.3]。

两阶段过滤:
1. **Layer-level**: 计算每层平均 cosine similarity $\bar{c}^{(\ell)}$ [Eq. 3],基于全局统计 µ,σ 设定阈值 τ = µ + kσ [Eq. 5],选择 $\bar{c}^{(\ell)} < \tau$ 的层(相似度低 = 偏离 ID-prototype 大 = identity 信息强)
2. **Step-level**: 在选定的层 ℓ' 内,仅干预 $c^{(\ell',t')} < \bar{c}^{(\ell')}$ 的步

[论文原文] 这种 sparse intervention 避免了过度干预对 phonetic fidelity 的破坏 [§2.3]。k 的选择: 实验表明 k=1 (即 τ=µ+σ) 是最佳平衡点 [Table 4]。

**设计 4: Unlearning via projection subtraction**

对选定的 (ℓ',t') 干预点:

$\bar{X}_{Opt}^{(\ell',t')} = X_{Opt}^{(\ell',t')} - \alpha (X_{Opt}^{(\ell',t')} \cdot S^{(\ell',t')}) S^{(\ell',t')}$ [Eq. 6]

[论文原文] 这只移除与 identity direction 对齐的分量,保留语言和韵律内容 [§2.4]。α=1.2 控制干预强度。

[agent 解读] 这本质上是一个"沿 steering vector 方向的正交投影减法"。当 α=1.0 时恰好完全移除 identity 分量;α=1.2 表示略微过补偿,可能是因为 identity 信息在非 steering 方向上也有少量残余泄漏。

### 训练策略

无训练。TruS 是 training-free 的 inference-time 方法。唯一需要预计算的是 ID-prototype(用 30 个 retain speakers 在 F5-TTS 上做一次前向传播)。

## 实验

| 指标 | 本文 (TruS) | Baseline (F5-TTS) | SGU | TGU | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| SIM-SO ↓ | 0.477 | 0.657 | 0.106 | 0.510 | Emilia (seen opt-out) | [Table 1] |
| WER-SO ↓ | 3.25 | 3.36 | 3.70 | 4.03 | Emilia (seen opt-out) | [Table 1] |
| Spk-ZRF-SO ↑ | 0.929 | 0.925 | 0.959 | 0.933 | Emilia (seen opt-out) | [Table 1] |
| SIM-R ↑ | 0.678† | 0.678 | 0.290 | 0.549 | LibriSpeech (retain) | [Table 1] |
| WER-R ↓ | 1.95† | 1.95 | 2.12 | 2.21 | LibriSpeech (retain) | [Table 1] |
| SIM-UO ↓ | 0.488 | 0.668 | — | — | LibriSpeech (unseen opt-out) | [Table 2] |
| WER-UO ↓ | 3.26 | 2.03 | — | — | LibriSpeech (unseen opt-out) | [Table 2] |
| SIM-Emo ↑ | 0.723 | 0.732 | — | — | CREMA-D (emotion) | [Table 3] |
| Training hours | 0 | — | 48 | 430 | 2x A6000 GPUs | [Table 1] |

**关键发现**:

1. **Retain speakers 不受影响**: 因为 TruS 仅在检测到 opt-out speaker 时才施加 steering,retain speakers 的生成结果与原始 F5-TTS 完全一致(†标记) [§3.2]。SGU/TGU 则因重训练导致 retain 性能退化(SIM-R 分别降至 0.290/0.549)。

2. **Unseen opt-out 泛化**: TruS 是首个能处理 unseen opt-out speakers 的方法 [Table 2]。SGU/TGU 仅能处理训练集中的说话人 [§3.2]。

3. **情感保留**: SIM-Emo 仅从 0.732 降至 0.723,说明 steering 确实只移除了 identity 方向,paralinguistic 属性得到保留 [Table 3]。

4. **Ablation -- layer selection**: τ=µ+σ 最优; τ=µ-σ (过少层) unlearning 不充分; all layers 时 WER 大幅恶化 (3.71 vs 3.25) [Table 4]。

5. **Ablation -- retain pool size**: N=30 最优; N=10 identity suppression 不足; N=50 在 seen data 上性能下降 [Table 5]。

## 局限性

1. **仅验证 DiT-based 架构**: TruS 基于 F5-TTS 的 DiT blocks 设计,是否适用于 AR-based TTS (如 VALL-E/CosyVoice 系列的 LLM 部分) 未知 [agent 解读]。论文声称"generally applicable to DiT-based architectures" [§2.1] 但未实验验证其他模型。

2. **WER 退化**: unseen opt-out 的 WER 从 2.03 升至 3.26 [Table 2],seen opt-out 的 WER 也从 3.36 降至 3.25 但原始 retain WER 是 1.95。[agent 解读] steering 投影减法虽然理论上只移除 identity 分量,实际上可能移除了与 identity 方向有少量相关的 content 信息。

3. **单一 reference utterance**: opt-out speaker 的 steering 仅基于单条参考语音。如果这条参考不能充分代表说话人的 identity 空间,steering 效果可能不稳定 [agent 解读]。

4. **对抗鲁棒性缺失**: 论文未讨论对抗场景 -- 如果攻击者对参考音频做微小扰动使 steering 失效怎么办?[agent 解读]

5. **SIM-SO (0.477) vs SGU (0.106)**: TruS 在 seen opt-out 上的身份抑制力度不如 SGU。但 SGU 的代价是 retain speakers 的 SIM-R 从 0.678 崩至 0.290,说明 SGU 实质上是"破坏所有人的声音"而非"精准遗忘" [Table 1]。

## 点评

TruS 的核心贡献是范式转换: 将 speaker unlearning 从"修改模型权重"重构为"推理时信号处理"。这带来了三个实质性优势: (1) 零训练成本; (2) 可处理 unseen speakers; (3) 不影响 retain speakers。这种"按需干预"的思路在 TTS 安全领域具有独特价值 -- 它使 opt-out 机制变成可热插拔的推理插件,而非需要服务中断的模型重训练。

方法设计上,从 NLP 领域的 activation steering (Rimsky et al., 2024; Turner et al., 2023) 迁移到 TTS 是合理的,但 TruS 增加了 TTS 特有的两个关键设计: (1) 利用 flow step 维度的动态性(NLP 中无此对应物); (2) FFN 输出作为 identity 信号的载体(NLP 中通常操作 residual stream)。

局限在于: 方法的理论基础偏弱 -- "identity 编码在结构化方向中"是一个假设而非证明,论文仅通过 cosine similarity 的统计分布间接支持 [Fig. 3]。此外,仅在一个模型 (F5-TTS) 上验证,泛化性声明尚需更多实验。

## 可复用的 idea

1. **ID-prototype 作为 identity-neutral 锚点**: 对任何需要"移除特定属性"的场景(如去口音、去情感),都可以构建类似的平均锚点然后做投影减法。关键是选择恰当的特征提取点(FFN vs attention)。

2. **动态 layer-step 选择 (τ=µ+kσ)**: 这种基于全局统计的自适应阈值机制可推广到任何需要"选择性干预模型中间层"的任务。避免了手动指定干预层的超参数搜索。

3. **推理时投影减法**: $\bar{X} = X - \alpha (X \cdot S) S$ 是一个极简的属性控制公式。如果能找到 emotion/prosody/accent 的 steering direction,同一框架可复用于细粒度语音属性编辑。与 EmoSteer-TTS 的 top-k channel selection 形成互补: TruS 做方向级操作,EmoSteer 做通道级操作。

4. **Training-free safety 插件模式**: 不修改模型权重 → 可在任意已部署模型上热加载 → 适合 production 环境的安全合规需求。这种"安全即插件"的设计思想值得 TTS 服务架构参考。
