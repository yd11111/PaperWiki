---
type: paper
tier: deep
title: "Stochastic Pitch Prediction Improves the Diversity and Naturalness of Speech in Glow-TTS"
arxiv_id: "2305.17724"
source: "Sources/StochasticPitchPrediction.pdf"
authors: [Sewade Ogun, Vincent Colotte, Emmanuel Vincent]
year: 2023
venue: "Interspeech 2023"
tags: [TTS, pitch-prediction, normalizing-flow, multi-speaker, zero-shot, prosody, diversity, Glow-TTS]
concepts: ["[[DurationPredictor]]", "[[ProsodyModeling]]", "[[ConditionalFlowMatching]]", "[[GlobalStyleTokens]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页: [[ConditionalFlowMatching]], [[ProsodyModeling]], [[DurationPredictor]], [[SpeakerAdaptation]], [[GlobalStyleTokens]], [[VoiceCloningTaxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 flow-based TTS 中显式韵律建模的分支。在 KB 中,[[ConditionalFlowMatching]] 页面记录了从 normalizing flow 到 flow matching 的演进脉络(Glow-TTS → VITS → CFM/Voicebox),本文的 Glow-TTS + normalizing flow 是这一演进的早期阶段。[[DurationPredictor]] 页面详细记录了 VITS 的 Stochastic Duration Predictor(SDP)设计,本文直接复用了该模块,并将概率建模思路从 duration 扩展到 pitch 维度。[[ProsodyModeling]] 页面将韵律信息分为 duration/pitch/energy/pause 四个物理维度,本文同时建模了其中两个(duration + pitch)的概率分布。
>
> **已有认知**: KB 中 [[GlobalStyleTokens]] 和 [[SpeakerAdaptation]] 记录了 zero-shot multi-speaker TTS 中处理说话人风格多样性的两条技术路线: (1) 隐式风格建模(GST、reference encoder); (2) speaker embedding + 模型适应。本文走的是第三条路线 — 显式建模 pitch 分布来增加多样性,不依赖参考音频或风格标签。[[VoiceCloningTaxonomy]] 将此类方法归入 Zero-shot Voice Cloning 中 conditioning on speaker embedding 的范式。
>
> **创新判断**: KB 中 [[DurationPredictor]] 记录了 VITS 首创 SDP 用 flow 学习 duration 概率分布,但未记录将同样思路应用于 pitch 的工作。本文的核心新意在于将 SDP 的概率建模范式迁移到 pitch contour,并通过 pitch conditioning 改造 decoder 的 affine coupling layer,是一个明确的"方法迁移 + 架构适配"贡献。
>
> 检索命中: [[ConditionalFlowMatching]]✓, [[ProsodyModeling]]✓ | 过滤: [[DurationPredictor]][待确认], [[SpeakerAdaptation]][待确认], [[GlobalStyleTokens]][待确认], [[VoiceCloningTaxonomy]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 Glow-TTS 中引入 flow-based stochastic pitch predictor,显式学习 F0 分布,配合 stochastic duration predictor 显著提升 zero-shot 多说话人语音的多样性和自然度
> - **路线**: 文本 → Transformer encoder → MAS 对齐 → Stochastic Duration Predictor(采样 duration）→ 扩展 embedding → Stochastic Pitch Predictor（采样 log-F0）→ Pitch-conditioned Decoder（normalizing flow）→ Mel-spectrogram → HiFi-GAN vocoder → 音频
> - **指标**: N-MOS 3.45(vs baseline 3.13）[Table 1]; D-MOS 显著优于 baseline 和 STD-only [Fig 2]; log-F0 分布更接近真实语音 [Fig 3]; 模型仅增加 1.5M 参数（56.0M vs 54.5M）
> - **可借鉴**: 将 VITS 的 stochastic duration predictor 架构直接迁移到 pitch 维度的方法论 — 任何可以用确定性预测器建模的韵律维度,都可以尝试用 flow-based 概率预测器替代以增加多样性
> - **局限**: 仅在 Glow-TTS 上验证,未在 VITS/端到端系统上测试; 与 VCTK copy-synthesis 上界仍有较大差距（N-MOS 3.45 vs 4.21）; 未测试 energy 维度; 评估规模较小（26 评估者,6 unseen speakers）

## 核心问题

Flow-based TTS 模型（如 Glow-TTS）通过从 latent distribution 采样来生成多样化语音,但在 zero-shot multi-speaker 场景下,仅靠 latent space 的温度采样无法产生足够多样和自然的语音 — 生成的语音往往听起来单调、缺乏活力 [§1]。核心问题是: **如何在不依赖参考音频或风格标签的情况下,让 flow-based TTS 为 unseen speakers 生成更多样、更自然的语音?**

作者的假设是: Glow-TTS 的 latent noise 虽然理论上可以产生多样的 stress 和 intonation,但这种隐式建模不够 — 显式学习 pitch contour 的概率分布并用其 condition decoder,可以更直接地改善多样性 [§2.3]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

在经典 Glow-TTS（encoder + flow decoder + deterministic duration predictor）基础上,做了三处改动 [§2, Fig 1]:

1. **替换 duration predictor**: 用 VITS 的 stochastic flow-based duration predictor 替代确定性 duration predictor [§2.2]
2. **新增 stochastic pitch predictor**: 新设计一个 flow-based pitch predictor 学习 log-F0 分布 [§2.3]
3. **改造 decoder**: 在 affine coupling layer 中增加 pitch conditioning 路径 [§2.1]

### 关键设计选择

**Stochastic Duration Predictor（复用 VITS）** [§2.2]:
- 由两组 normalizing flow 组成: post flows（训练时将 MAS 对齐的离散 duration → de-discretized log-duration 分布）+ 第二组 flows（log-duration → Gaussian noise）[论文原文]
- 推理时丢弃 post flows,直接从 Gaussian noise 经第二组 flows 生成 log-duration [论文原文]
- [agent 解读] 这比确定性 predictor（MSE loss 回归均值）能更好地捕捉同一音素在不同语境下的时长变化

**Stochastic Pitch Predictor（本文核心贡献）** [§2.3]:
- 架构借鉴 SDP 的第二组 flow blocks: 4 blocks of 1D conv + dilated depth-separable conv + spline flows [§2.3]
- 使用 variational data augmentation [19]: 将标量 pitch 值串联额外 Gaussian noise 向量以增加维度,使 flow 能进行有效的高维变换 [§2.3]
- 训练时: speaker-conditioned encoder embedding 按 MAS 对齐后扩展到帧级,作为条件输入; flow 将 log-F0 反转为 Gaussian noise [论文原文]
- 推理时: 从 Gaussian noise 采样,经 flow 生成 log-F0 contour [论文原文]
- 对 encoder embedding 施加 stop-gradient,防止 pitch/duration predictor 的梯度干扰 encoder 学习 [§2.3, footnote 2]
- [agent 解读] 这一设计将 SDP 的"概率化"思路从 duration 直接迁移到 pitch,是一种方法论层面的平行推广,优点在于不需要设计新的训练目标 — 直接复用 normalizing flow 的 MLE 训练

**Decoder 的 pitch conditioning** [§2.1]:
- 从 ground-truth mel 提取 log-F0（pYIN）,经 1D conv 投影到 decoder 特征维度 [§2.1]
- 投影后的 log-F0 经 squeeze 匹配 decoder 的 squeeze ratio [§2.1]
- 通过 affine coupling layer 中的 WN-Pitch 通路 condition decoder（见 Fig 1b）[§2.1]
- [agent 解读] 将 pitch 信息注入 decoder 的 affine coupling layer 而非 encoder,使 decoder 的 flow 变换可以利用 pitch 轨迹来组织频谱特征,物理直觉是 pitch 直接决定谐波结构,在频谱域建模更高效

### 训练策略

- 联合训练所有模块（encoder + decoder + duration predictor + pitch predictor）,通过 MLE 优化 [§2.3]
- F0 提取使用 pYIN,窗口 64ms,hop 16ms（与 mel 参数对齐）[§3.1]
- Unvoiced 帧的 log-F0 设为 0 [§2.1]
- Speaker embedding: Resemblyzer（基于 VoxCeleb 训练的 SV 模型）提取 256 维 l2-normalized 向量 [§3.1]
- 输入: phoneme + character 混合,插入 blank token [§3.1]
- 训练配置: batch 192, 4x RTX GPU, RAdam + cosine annealing + 6K warmup, 200 epochs (~4 days for STDP model) [§3.2]

## 实验

| 指标 | Baseline (Glow-TTS) | GlowTTS-STD | GlowTTS-STDP | VCTK-copy (上界) | 出处 |
| --- | --- | --- | --- | --- | --- |
| N-MOS (总) | 3.13 | 3.31 | **3.45** | 4.21 | [Table 1] |
| N-MOS (男) | 2.92 | 3.11 | **3.40** | 4.00 | [Table 1] |
| N-MOS (女) | 3.35 | 3.51 | **3.51** | 4.40 | [Table 1] |
| NR-MOS (总,长句) | 2.95 | 3.26 | **3.39** | 4.79 | [Table 1] |
| S-MOS (总) | 2.26 | **2.98** | **2.99** | — | [Table 1] |
| WV-MOS | 4.11 | 4.17 | **4.18** | 4.32 | [Table 1] |
| cos-sim | 0.8287 | **0.8364** | 0.8319 | 0.8121 | [Table 1] |
| D-MOS | ~2.4 | ~2.7 | **~3.0** | — | [Fig 2] |

**关键发现**:
1. **Stochastic duration predictor 是自然度提升的主因**: STD vs baseline 的 N-MOS 提升（3.31 vs 3.13）比 STDP vs STD 的提升（3.45 vs 3.31）更大 [Table 1] [agent 解读]
2. **Stochastic pitch predictor 是多样性提升的主因**: D-MOS 从 STD 到 STDP 有明显提升 [Fig 2],且 log-F0 分布匹配真实语音分布的程度 STDP >> STD >> baseline [Fig 3]
3. **Pitch prediction 对男声改善更大**: 男声 N-MOS 从 STD 的 3.11 提升到 STDP 的 3.40（+0.29），女声持平 [Table 1]
4. **Speaker similarity 主要由 duration predictor 驱动**: S-MOS 从 baseline 2.26 到 STD 2.98 跳跃式提升,STDP 仅微增至 2.99 [Table 1]
5. **与 copy-synthesis 仍有差距**: N-MOS 3.45 vs 4.21,表明 zero-shot 多说话人 TTS 仍有较大改进空间 [Table 1]

**评估设计**: 6 个 unseen speakers（3 男 3 女）来自 VCTK; 26 位评估者; D-MOS 评估方式为同文本生成 3 条语音拼接后让评估者判断 intonation 多样性 [§3.3]

## 局限性

1. **仅验证于 Glow-TTS**: 未在 VITS 或其他端到端系统上测试 stochastic pitch predictor 的效果。VITS 本身已有 stochastic duration predictor,加入 pitch predictor 是否有类似改善未知 [agent 解读]
2. **未建模 energy**: 韵律的四个维度中仅覆盖 duration 和 pitch,energy 和 pause 未被显式建模 [agent 解读]
3. **评估规模有限**: 26 位评估者、6 个 unseen speakers,统计显著性虽有报告但样本量偏小 [§3.3]
4. **参数增量小但推理成本未量化**: 仅增加 1.5M 参数,但 flow-based pitch predictor 的推理延迟未报告 [agent 解读]
5. **Vocoder 未微调**: HiFi-GAN 在 LibriTTS 上训练,未在生成的 mel 上微调,可能成为质量瓶颈 [§3.2]
6. **F0 提取依赖 pYIN**: 在 unvoiced 帧设为 0 的处理较粗糙,可能影响 pitch predictor 的学习 [§2.1]

## 点评

这是一篇思路清晰、贡献明确的 Interspeech 短文。核心 idea — 将 VITS 的 stochastic duration predictor 的概率化方法论平行推广到 pitch 维度 — 简单但有效。实验充分证明了 stochastic pitch prediction 在增加语音多样性方面的独特贡献（D-MOS 和 F0 分布匹配），与 stochastic duration prediction 对自然度的贡献形成互补。

从 KB 背景看,这项工作处于 normalizing flow TTS 演进的"显式韵律概率化"节点: FastSpeech 2 开创了显式 pitch/energy/duration predictor 但都是确定性的 → VITS 将 duration 概率化 → 本文将 pitch 也概率化。后续的 Variance-Flow (Lee et al., 2022) 在单说话人 FastSpeech 2 上做了类似工作,但本文在 zero-shot multi-speaker 场景下的验证更有实用价值。

不足之处在于实验规模偏小,且未探索该方法在 VITS 等更先进系统上的效果。考虑到 2023 年后 LLM-based TTS（VALL-E, CosyVoice）已转向隐式建模所有韵律维度,这种显式概率建模的路线在实际部署中的价值可能有限 — 但其方法论（"任何确定性预测器都可以尝试用 flow 概率化"）仍有参考价值。

## 可复用的 idea

1. **方法迁移范式**: VITS SDP 的架构（variational data augmentation + spline flows + stop-gradient）可以几乎原封不动地迁移到其他标量/低维韵律特征（energy, speaking rate 等）的概率建模 [§2.3]
2. **Decoder conditioning 的 pitch 注入点**: 在 affine coupling layer 而非 encoder 端注入 pitch 信息,利用 flow 的可逆性保持信息流的完整性 [§2.1, Fig 1b]
3. **D-MOS 评估协议**: 同文本生成多条语音 → 拼接 → 评估 intonation 多样性,是衡量韵律多样性的一种实用方案 [§3.3]

## 审阅

(待独立审阅 agent 填写)
