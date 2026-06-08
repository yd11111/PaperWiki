---
type: paper
tier: deep
title: "Beyond Monologue: Interactive Talking-Listening Avatar Generation with Conversational Audio Context-Aware Kernels"
arxiv_id: "2604.10367"
source: "Sources/BeyondMonologue.pdf"
authors: [Yuzhe Weng, Haotian Wang, Xinyi Yu, Xiaoyan Wu, Haoran Xu, Shan He, Jun Du]
year: 2026
venue: "arXiv (USTC + iFLYTEK)"
tags: [full-duplex, audio-driven-video, talking-head, avatar, attention-mechanism, DiT, flow-matching, dual-stream, dataset, lip-sync]
concepts: ["[[ConditionalFlowMatching]]", "[[DiffusionModel]]", "[[Classifier-FreeGuidance]]", "[[Full-duplexSpokenDialogue]]"]
models: ["[[模型库/wav2vec2.0|wav2vec 2.0]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 4 个待确认实体页: [[ConditionalFlowMatching]]✓, [[DiffusionModel]][待确认], [[Classifier-FreeGuidance]][待确认], [[Full-duplexSpokenDialogue]][待确认], wav2vec 2.0[待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]], [[DiffusionModel]], [[Classifier-FreeGuidance]], [[Full-duplexSpokenDialogue]], wav2vec 2.0 | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: BeyondMonologue 处于 audio-driven human video generation 和 full-duplex interaction 的交叉点。在 full-duplex 演进线上, KB 中已有的系统 (Moshi, LSLM, ELLSA 等) 聚焦于 speech language model 层面的同时说+听能力; BeyondMonologue 则聚焦于视觉生成层面 — 如何让虚拟人在说话的同时通过面部表情和身体动作自然地回应对方的语音。在 audio-driven video generation 演进线上,EchoMimicV3, Hallo3, FantasyTalking 等方法仅处理单人说话(monologue)场景;StreamAvatar 尝试了双流但仅用 2D local attention 处理 listening audio,无法捕获对话的长程语义动态。BeyondMonologue 提出 MHGK (Multi-Head Gaussian Kernels) 在注意力层面统一解决了 lip-sync 精度(局部)与交互语义(全局)的 trade-off。

**已有认知**:
- [[ConditionalFlowMatching]]✓: CFM 是连续正规化流的训练方法,在 TTS 中广泛使用。BeyondMonologue 采用 Flow Matching (OT formulation) 作为视频生成的训练范式,但底座是 Wan2.2 视频生成模型而非 TTS 系统。
- [[DiffusionModel]][待确认]: Diffusion/Flow Matching 是当前视频生成的主流范式。BeyondMonologue 使用 DiT (Diffusion Transformer) 架构。
- [[Classifier-FreeGuidance]][待确认]: CFG 是条件生成的标准技术。BeyondMonologue 的 Q-Former 改进了 CFG 的 unconditional embedding — 用 Q-Former 生成的 learned embedding 替代传统 all-zero input,缩小了条件/无条件表征的分布差距。
- [[Full-duplexSpokenDialogue]][待确认]: KB 中的全双工系统 (dGSLM, Moshi, LSLM) 关注 speech token 层面的同时输入输出; BeyondMonologue 在视频生成维度实现 full-duplex — 同时处理 talking 和 listening 双流音频来驱动视频。
- wav2vec 2.0 [待确认]: wav2vec 2.0 是自监督语音表征模型,BeyondMonologue 使用其全层特征作为音频编码输入。

**创新判断**: BeyondMonologue 的核心新意是 Multi-Head Gaussian Kernels (MHGK) — 对 cross-attention 施加多尺度高斯时序约束,而非引入新模块。不同 attention head 被分配不同标准差的高斯衰减,使窄头专注帧级唇形同步、宽头捕获对话级语义上下文。该设计几乎无额外计算开销(仅在 attention score 矩阵上加 bias),但有效替代了 ALiBi 等固定衰减方案。在已有 KB 中没有直接对应的概念。

## 速查

> [!summary] 速查
> - **一句话**: 提出 Multi-Head Gaussian Kernels (MHGK) 在 3D cross-attention 中注入多尺度时序先验,结合 Wan2.2-5B 骨架和双流 Q-Former 音频注入,实现首个统一的高质量全双工 talking-listening 虚拟人视频生成
> - **路线**: Reference Image + Talking Audio + Listening Audio → Wav2Vec 2.0 → 独立 Causal Q-Former (talking/listening) → TL2V-DiT (3D Spatiotemporal CA + MHGK) → Wan2.2 3D VAE Decoder → Video [§3, Fig 2]
> - **指标**: HDTF: FID 23.96 / CSIM 0.749 / LSE-C 6.58; MEAD: FID 21.82 / CSIM 0.876 / LSE-C 6.28; ResponseNet 交互: FID 18.48 / LSE-C 6.68; User Study listening MOS: Natural 4.14 / AV Align 4.18 (vs INFP 3.86/4.05) [Table 1, 2, 5]
> - **可借鉴**: MHGK 的思路 — 用不同标准差的高斯核分配 attention head 的感受野,在任何需要同时保持局部对齐和全局上下文的跨模态 attention 中可直接复用;Q-Former 替代 all-zero CFG 的设计也可迁移到 audio-conditioned diffusion 系统
> - **局限**: 未开源代码/模型;VoxHear 数据集未公开;仅在 HDTF/MEAD/ResponseNet 上评测,缺乏 in-the-wild 验证;listening 行为评估依赖 user study (11 人) 而非自动化指标

## 核心问题

1. **Monologue → Full-duplex 视频生成为什么难?** 现有 audio-driven video generation 方法仅处理单人说话,直接扩展到 talking-listening 面临根本性 trade-off: talking 需要严格的帧级音频-视觉时序对齐以保证 lip-sync;listening 需要理解长程对话语义以生成自然反应动作。2D Spatial CA 能保 lip-sync 但完全切断全局上下文;3D Global CA 保留全局上下文但注意力矩阵过于稀疏,降低 lip-sync 精度 [§1, §3.2.2]。

2. **已有尝试为什么不够?** 状态机方法 (DIM) 将 talking 和 listening 建模为互斥状态,无法处理真实对话中的音频重叠 [§2.2]。StreamAvatar 对 listening audio 直接复用帧级 2D local attention,无法捕获对话动态,且训练/推理存在 masking gap [§2.2]。INFP 通过中间 motion codebook 生成 head-only 反应,缺乏上半身表达力 [§2.2]。

3. **如何在一个框架内同时保证 lip-sync 精度和 listening 自然度?** BeyondMonologue 观察到交互音频的 "Dual-Resolution Property" — talking 需要细粒度 hard alignment,listening 需要粗粒度 soft contextual understanding [§1]。基于此洞察,提出 MHGK 将不同 attention head 的感受野从窄到宽分配,在单个 3D cross-attention 中同时实现局部同步和全局上下文建模。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

BeyondMonologue 是一个 Double A2V (Audio-to-Video) 模型,建立在 Wan2.2-5B 视频生成模型之上 [§3.2]:

**输入**: 参考肖像图像 I_ref + 说话音频序列 A_talk + 聆听音频序列 A_listen [§3.1]

**音频编码** [§3.2.1]:
- 预训练 Wav2Vec 2.0 提取全层 (all-layer) 表征,拼接并投影到维度 D_a
- 分别通过独立的 **Talking Q-Former** 和 **Listening Q-Former** (均为 Causal Q-Former) 进行特征压缩
- Causal Q-Former: 在每个与视频 latent 对齐的时间窗口内,用 N 个 learnable queries 通过 cross-attention 聚合音频特征 [Fig 2]
- 两个 Q-Former 权重独立,允许 talking 分支关注底层语音细节 (phonetic),listening 分支关注高层语义信息 [论文原文]

**视频骨干** [§3.2]:
- Wan2.2 3D VAE: 时间压缩 4x,空间压缩 16x16
- TL2V-DiT: DiT Block x N 层,每层包含 Self-Attention + Talking-CA + Listening-CA [Fig 2]
- 冻结原有 video-text cross-attention,通过 IP-Adapter 范式注入 talking 和 listening 音频 [§3.2.1]

**输出**: 通过 Wan2.2 3D VAE Decoder 解码为视频 [Fig 2]

### 关键设计选择

**Multi-Head Gaussian Kernels (MHGK)** [§3.2.2]:

[论文原文] 在 3D spatiotemporal cross-attention 中,为每个 attention head h 引入高斯距离惩罚矩阵 B^(h):

```
Attention^(h)(Q, K, V) = Softmax(Q^(h)(K^(h))^T / sqrt(d) - B^(h)) V^(h)
```

其中:
```
B^(h)(i, j) = alpha_h * (1 - exp(-(i-j)^2 / (2*sigma_h^2)))
```

i, j 分别是视频和音频对齐后的时间索引。不同 head 被分配不同的标准差 sigma_h 和缩放系数 alpha_h:
- sigma_h → 0 的 head: 高斯分布极其陡峭,alpha_h 最大化,退化为窄局部注意力,迫使模型学习严格唇形对齐 [论文原文]
- sigma_h → ∞ 的 head: B^(h) → 0,完整保留 3D 全局注意力,捕获对话的情感和语义上下文 [论文原文]

[agent 解读] MHGK 的设计哲学类似于 NLP 中的 ALiBi (Attention with Linear Biases),但用高斯而非线性衰减,而且不同 head 有不同的 sigma。高斯衰减的优势在于:中心位置的 attention 保持完整(仅惩罚远处),而 ALiBi 的线性衰减从距离 0 开始就产生惩罚。实验也验证了 MHGK 优于 ALiBi [Table 2, 3]。计算上,MHGK 仅在 attention score 矩阵上加一个预计算的 bias 矩阵,不需要额外模块(如时序卷积或额外 local CA 层),开销可忽略。

**为什么需要 1D RoPE 对齐时间索引?** [§3.2.2]

[论文原文] 视频 latent 序列 X ∈ R^{B×(L×H·W)×D} 和音频序列 C ∈ R^{B×(L_a×S)×D} 的时间长度 L 和 L_a 不匹配。论文对两种模态都应用时间缩放的一致 1D RoPE,将它们映射到相同的时间单位索引 t,在高斯惩罚计算前提供初步的时间对齐。

**Adaptive Audio Q-Former 改善 CFG** [§3.2.1]:

[论文原文] 传统 CFG 使用 all-zero 向量作为 unconditional 音频输入,这引入了分布外 (out-of-distribution) 信号,在高 guidance scale 下导致视觉 artifacts (如颜色偏差)。Q-Former 在其学习的特征空间内生成 unconditional embedding,大幅缩小了条件/无条件表征的分布差距,允许使用更高的 guidance scale 而不产生 artifacts。

[agent 解读] 这与 KB 中记录的 LongCat-AudioDiT 发现的 "CFG oversaturation" 问题相呼应 — 那篇论文用 APG (Adaptive Projection Guidance) 解决,BeyondMonologue 则从输入端改善 unconditional embedding 的质量。两种方法针对同一问题的不同层面。

**Arbitrary-Position Guided Training + Diffusion Forcing** [§3.3]:

[论文原文] 传统方法依赖固定位置 (first-frame 或 last-frame) 锚定场景和身份。First-frame 锚定导致 Attention Sink 效应 (后续帧过度注意力集中到第一帧),降低运动多样性 [§3.3]。

BeyondMonologue 的方案:
1. 训练时从长度为 T 的 noisy video clip 中随机采样任意帧 i ∈ [1,T],将其 clean latent z_i 作为引导条件,RoPE index 设为 i [§3.3]
2. 结合 Diffusion Forcing — 对不同时间 chunk 注入不同级别随机噪声,使模型学会从任意绝对位置进行前向/后向/双向扩展 [§3.3]
3. 推理时: ablation 发现将引导帧放在未来位置 (RoPE index 22,相对当前生成窗口略远) 能最好地平衡身份保持、运动幅度和动态自然度 [Table 4]

### 训练策略

**两阶段增量训练** [§3.3, §4.1]:

[论文原文] 为稳定双流架构的收敛并避免早期信号干扰:
- **Stage 1 (Talking Priority)**: 仅引入 Talking Audio Adapter,训练精确唇形对齐和自然说话行为。数千小时公开+内部数据,DWPose 人体姿态过滤。100k steps。
- **Stage 2 (Listening Fusion)**: 添加 Listening Audio Adapter 继续训练,使模型在不破坏已建立的说话能力的前提下融入 listening 感知的交互行为。使用 VoxHear 数据集 (1,000+ hours)。30k steps。

[论文原文] 训练配置: Wan2.2-5B backbone, 720p, multi-scale bucket 动态分辨率, AdamW + bf16, EMA (decay 0.999), 新参数 lr=1e-5 / backbone 可训练参数 lr=2e-6, 16x A100, global batch 32 [§4.1]。

**VoxHear 数据集构建** [§3.4, Fig 4]:

两阶段清洗管线:
1. **Visual Track**: DWPose 关键点提取 → 过滤低质量/多人/不一致的片段 → 裁剪到上半身/肖像区域 → 10 秒时间切片
2. **Audio Track**: MossFormer2 (ClearVoice toolkit) 语音分离 → SyncNet 唇音同步验证 → 仅保留视觉+音频一致性均通过的样本

最终获得 1,206 小时的 speaking-listening 交互数据,每个样本为时间对齐的四元组 (V_a, A_a, V_b, A_b),音频完全解耦 [§3.4]。

[agent 解读] 数据质量是 BeyondMonologue 的关键贡献之一。现有数据集 (Seamless, SpeakerVid-5M) 存在音频混叠问题,VoxHear 通过 MossFormer2 分离 + SyncNet 验证解决了这个瓶颈。但论文未明确 VoxHear 是否会公开。

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| FID↓ | 23.96 / 21.82 | OmniAvatar 23.85/29.87, EchoMimicV3 25.92/25.43, Fantasy-Talking 24.03/45.24 | HDTF / MEAD | [Table 1] |
| FVD↓ | 235.73 / 206.33 | OmniAvatar 206.80/263.62, Fantasy-Talking 241.24/312.03 | HDTF / MEAD | [Table 1] |
| LPIPS↓ | 0.145 / 0.057 | OmniAvatar 0.157/0.088, Fantasy-Talking 0.149/0.108 | HDTF / MEAD | [Table 1] |
| CSIM↑ | 0.749 / 0.876 | OmniAvatar 0.703/0.782, Fantasy-Talking 0.738/0.759, EchoMimicV3 0.687/0.808 | HDTF / MEAD | [Table 1] |
| LSE-C↑ | 6.58 / 6.28 | OmniAvatar 6.50/6.26, Hallo3 6.47/5.58, Fantasy-Talking 3.65/3.86 | HDTF / MEAD | [Table 1] |
| LMD↓ | 10.25 / 3.48 | OmniAvatar 11.96/6.61, Fantasy-Talking 11.73/4.10, EchoMimicV3 13.60/5.28 | HDTF / MEAD | [Table 1] |
| Interactive FID↓ | 18.48 | DIM 35.68, 2D Spatial CA 20.61, 3D+RoPE 22.59, 3D+RoPE+ALiBi 23.41 | ResponseNet | [Table 2] |
| Interactive CSIM↑ | 0.814 | DIM 0.791, 2D Spatial CA 0.797, 3D+RoPE 0.805, 3D+RoPE+ALiBi 0.794 | ResponseNet | [Table 2] |
| Interactive LSE-C↑ | 6.68 | DIM 2.02, 2D Spatial CA 6.24, 3D+RoPE 4.54, 3D+RoPE+ALiBi 5.98 | ResponseNet | [Table 2] |
| User Study Natural.↑ | 4.14 | INFP 3.86, DIM 1.68, L2L 1.59, RLHG 1.41 | Listening | [Table 5] |
| User Study Motion↑ | 4.05 | INFP 4.00, DIM 2.05, L2L 1.45, RLHG 1.36 | Listening | [Table 5] |
| User Study AV Align.↑ | 4.18 | INFP 4.05, DIM 2.00, L2L 1.77, RLHG 1.68 | Listening | [Table 5] |
| User Study Visual↑ | 4.32 | INFP 4.55, DIM 1.86, L2L 1.73, RLHG 1.50 | Listening | [Table 5] |

**关键发现**:

1. **Monologue 场景 SOTA** [Table 1]: 在 HDTF 和 MEAD 上,BeyondMonologue 在 identity preservation (CSIM)、lip-sync (LMD, LSE-C)、perceptual similarity (FID, FVD, LPIPS) 三个维度综合最优。特别是 MEAD 上 CSIM 0.876 大幅领先第二名 EchoMimicV3 的 0.808。

2. **Interactive 场景显著优于唯一开源 baseline DIM** [Table 2]: FID 18.48 vs 35.68,LSE-C 6.68 vs 2.02,提升幅度巨大。DIM 的 LSE-C 仅 2.02 说明其在交互场景下唇形同步严重退化。

3. **MHGK >> 其他注意力机制** [Table 2, 3]: 与 2D Spatial CA、3D+RoPE、3D+RoPE+ALiBi 相比,MHGK 在 lip-sync (LSE-C) 和 identity (CSIM) 上均最优。2D Spatial CA 虽然 lip-sync 尚可,但 FVD 大幅退化 (306.72 vs 235.73),说明帧内 attention 无法捕获跨帧动态韵律 [Table 3]。3D+RoPE 的 LSE-C 仅 4.54,证实无约束的全局 attention 会稀释 lip-sync 监督信号 [Table 3, §4.3.1]。

4. **引导帧位置的影响** [Table 4]: First-frame 引导最差 (CSIM 0.614, FVD 347.65),作者将其归因于训练中未充分强化 attention sink 行为 [§4.3.2]。Index 22 (略远于生成窗口) 最优; 太近 (index 21) 限制运动动态,太远 (index 27) 引导信号不足导致振荡 [§4.3.2]。

5. **User Study (11 人)** [Table 5]: 在 listening 场景下,BeyondMonologue 在 naturalness (4.14) 和 AV alignment (4.18) 上超越 INFP,motion diversity 持平 (4.05 vs 4.00),visual quality 略低于 INFP (4.32 vs 4.55)。与更早方法 (DIM, L2L, RLHG) 差距巨大。

## 局限性

1. **未开源** [§4]: 代码、模型和 VoxHear 数据集均未公开。论文仅提及项目页面,但实际可复现性低。

2. **Listening 行为评估依赖小规模 User Study**: 仅 11 名参与者,统计显著性存疑。目前缺乏 listening 反应自然度的自动化评估指标,这是领域层面的共性问题 [§4.4]。

3. **评测数据集有限**: 仅使用 HDTF、MEAD (实验室录制)、ResponseNet (小规模 head-only) 评测。缺乏 in-the-wild 真实对话场景的评估。

4. **VoxHear 来源和多样性未充分说明**: 论文描述了清洗管线但未详细说明原始视频的来源(是否公开数据?)、身份多样性、语言分布等 [§3.4]。

5. **交互行为的上限**: Listening 反应仍受限于数据驱动的统计模式,无法理解对话内容的语义 — 模型看到的是 wav2vec 特征,不是语言理解。真正的"理解对方在说什么并做出有意义回应"需要与 LLM 结合 [§5]。

6. **推理效率未讨论**: 基于 Wan2.2-5B 的视频 DiT 推理成本通常很高,论文未报告推理时间或实时性数据。

## 点评

**核心贡献的价值**: MHGK 是一个优雅的设计 — 用参数化高斯核将 "local-global attention trade-off" 从二选一变为连续谱,让模型在同一 attention 机制内自动学习最优的时间感受野分配。该设计几乎无计算开销,却在实验中一致优于 ALiBi 和其他变体。这个思路不限于 audio-driven video generation,任何需要同时维护局部对齐和全局上下文的跨模态 attention (如 speech-text alignment in TTS, audio-motion generation) 都可以尝试。

**Q-Former 的双重作用**: Causal Q-Former 同时解决了两个问题: (1) 音频特征压缩和跨模态对齐; (2) CFG unconditional embedding 质量提升。后者的贡献容易被忽视,但它使得模型可以在更高 guidance scale 下运行而不产生 artifacts,这对视频生成质量有直接影响 [§3.2.1]。

**与 KB 中 Full-duplex 系统的关系**: BeyondMonologue 和 KB 中的全双工 speech LM (Moshi, LSLM 等) 解决的是不同维度的"全双工"问题。Speech LM 关注的是"什么时候说、说什么"; BeyondMonologue 关注的是"说的时候脸和身体怎么动、听的时候怎么反应"。两者是互补的 — 未来完整的全双工虚拟人需要两个维度都解决。论文在 §5 (Conclusion) 也指出终极目标是"感知语音指令并端到端产出任意类人行为",这将需要语言理解和视觉生成的深度融合。

**两阶段训练的必要性**: Stage 1 先稳定 talking 能力再在 Stage 2 融入 listening,与 ELLSA 的三阶段训练策略 (先训练各 expert → 整合 SA-MoE → 连接 synthesizer) 有相似的设计直觉 — 都是避免多信号源早期干扰。这种增量式训练在多条件生成中似乎是一个通用的良好实践。

**Arbitrary-position guidance 的发现**: 引导帧位置的 ablation (Table 4) 揭示了一个实用的直觉: 引导帧不能太近 (过度收敛) 也不能太远 (失去控制),optimal 位置在生成窗口前方略远处。这种 trade-off 在其他 video diffusion 系统中也可能存在。

## 可复用的 idea

1. **Multi-Head Gaussian Kernels (MHGK)**: 对 cross-attention 的 attention score 矩阵加可学习的高斯距离惩罚,不同 head 用不同 sigma,实现多尺度时序建模。计算开销近零,可直接替代 ALiBi 用于任何需要在 attention 中编码时序先验的场景 (audio-visual alignment, speech-text alignment, music-motion generation)。

2. **Q-Former 生成 CFG unconditional embedding**: 用 Q-Former 的学习空间替代 all-zero 的 unconditional 输入,缩小条件/无条件分布差距。适用于任何使用 CFG 的 audio-conditioned diffusion/flow 系统,特别是当高 guidance scale 导致 artifacts 时。

3. **Arbitrary-position guided training + Diffusion Forcing**: 训练时随机选择引导帧位置 + 不同 chunk 注入不同噪声级别,推理时可灵活选择最优引导位置。这使得长时视频生成中的身份保持和运动动态更好地平衡。

4. **两阶段数据管线构建交互数据集**: Visual Track (DWPose 过滤 + 人体裁剪) + Audio Track (MossFormer2 分离 + SyncNet 验证) 的组合,可用于从任何对话视频构建干净的双人交互数据。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | MHGK 的 WHY/HOW 因果解释完整, Q-Former 双重作用和两阶段训练逻辑清晰, 速查可借鉴具体 |
> | 可信赖 | pass | 数字标注覆盖率 ~85%, 指标使用正确, 速查有数字+数据集+来源 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注系统, 覆盖率 ~85% |
> | 可定位 | pass | audio-driven video gen + full-duplex 双线定位清晰, 正确区分 video gen 和 speech LM 全双工 |
> | 不污染 | pass | 按指令不做反向更新, frontmatter 引用合理 |
> 
> Issues: 3 (high: 0, medium: 0, low: 3)
> 详见 `_review/BeyondMonologue-review.yml`

---

检索命中: [[ConditionalFlowMatching]], [[DiffusionModel]], [[Classifier-FreeGuidance]], [[Full-duplexSpokenDialogue]], wav2vec 2.0 | 过滤: 无 | 未命中但可能相关: 无
