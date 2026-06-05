---
type: paper
tier: deep
title: "Sample-Efficient Diffusion for Text-To-Speech Synthesis"
arxiv_id: "2409.03717"
source: "Sources/SESD.pdf"
authors: [Justin Lovelace, Soham Ray, Kwangyoun Kim, Kilian Q. Weinberger, Felix Wu]
year: 2024
venue: "arXiv (Interspeech submission)"
tags: [TTS, diffusion, latent-diffusion, sample-efficiency, data-efficiency, text-encoder, alignment]
concepts: ["[[Diffusion-basedTTS]]", "[[DiffusionModel]]", "[[Classifier-FreeGuidance]]", "[[DurationPredictor]]", "[[ResidualVectorQuantization]]", "[[Non-autoregressiveTTS]]"]
models: ["[[论文笔记/SESD|SESD]]", "[[模型库/EnCodec|EnCodec]]", "[[模型库/VITS|VITS]]", "[[模型库/NaturalSpeech2|NaturalSpeech 2]]", "[[模型库/HuBERT|HuBERT]]"]
tasks: ["Text-to-Speech"]
datasets: ["LibriSpeech"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[ResidualVectorQuantization]], [[模型库/EnCodec|EnCodec]]; 4 个待确认: [[Diffusion-basedTTS]], [[Classifier-FreeGuidance]], [[DurationPredictor]], [[Speech-TextAlignment]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: SESD 属于 [[Diffusion-basedTTS]] 中的 **latent diffusion** 分支。在 Diffusion TTS 演进线上 (Diff-TTS 2021 → Grad-TTS 2021 → ProDiff 2022 → NaturalSpeech 2 2023 → Flow Matching 取代 diffusion 2023-2024),SESD 与 NaturalSpeech 2 同属 latent diffusion 路线,但关注点不同: NS2 追求音质天花板 (44k 小时数据 + pitch 标注 + phoneme alignment),SESD 追求数据效率 (<1k 小时 + 无 phoneme alignment)。
>
> **已有认知**:
> - [[模型库/EnCodec|EnCodec]] (confirmed): SESD 使用 EnCodec 的 **连续 latent** (quantization 前的 128d embedding, 75Hz),而非 RVQ 离散 tokens。这是与 VALL-E 等 codec LM 的关键区别 — VALL-E 建模离散 tokens,SESD 建模连续 latents。
> - [[ResidualVectorQuantization]] (confirmed): EnCodec 内部使用 RVQ 压缩,但 SESD 刻意绕过 RVQ 直接使用连续 embedding,避免量化信息损失。推理时生成的连续 latent 再经 RVQ 量化 + EnCodec decoder 还原波形。
> - [[Classifier-FreeGuidance]] [待确认]: SESD 使用标准 CFG (p=0.1 随机 drop text),text-only 合成用 w=5.0,speaker-prompted 用 w=8.0。
> - [[DurationPredictor]] [待确认]: SESD 的 duration 策略与主流 TTS 完全不同 — 不预测 phoneme duration,而是用 ByT5 fine-tuned 的 seq2seq 模型预测**整体时长** (utterance-level),diffusion 过程内部隐式解决 phoneme 对齐。这与 Duration Predictor 演进线上的趋势一致 (显式 phoneme duration → 隐式端到端)。
>
> **创新判断**: SESD 的核心贡献不在于单个组件的突破,而在于多个设计选择的协同使数据效率大幅提升: (1) 连续 latent diffusion 降低建模难度,(2) character-aware LM (ByT5) 替代 phonemizer,(3) 非对称 loss weighting 强化高噪声级的 transcript alignment,(4) position-aware cross-attention 显式注入位置信息。消融实验 [Fig 4] 表明每个组件都不可或缺。
>
> 检索命中: [[ResidualVectorQuantization]]✓, [[模型库/EnCodec|EnCodec]]✓ | 过滤: [[Diffusion-basedTTS]](pending-review), [[Classifier-FreeGuidance]](pending-review), [[DurationPredictor]](pending-review), [[Speech-TextAlignment]](pending-review) | 未命中但可能相关: Latent Diffusion (无独立页)

> [!summary] 速查
> - **一句话**: 通过 latent diffusion + ByT5 文本编码 + 非对称 loss weighting,用不到 1k 小时标注数据达到接近人类水平的 TTS 可懂度 (WER 2.3% vs human 2.2%)
> - **路线**: Text → ByT5 encoder (frozen) → Position-Aware Cross-Attention → U-Audio Transformer (1D U-Net + Transformer) → EnCodec continuous latents → RVQ + EnCodec decoder → Waveform
> - **指标**: Text-only WER 2.3% [Table 1]; Speaker-prompted WER 2.3%, SIM 0.617 [Fig 3]; VALL-E 对比 WER 5.9%, SIM 0.580 (用 62.5x 数据) [§1]; 训练仅 960h LibriSpeech [§5]
> - **可借鉴**: (1) 非对称 Cauchy-Normal loss weighting 强化高噪声级学习,改善 text-speech alignment; (2) 在 EnCodec quantization 前建模连续 latent 避免信息损失 + 序列压缩 32x; (3) Position-aware cross-attention 中用 MLP 编码相对位置辅助对齐; (4) ByT5 作为 character-level text encoder 省去 phonemizer
> - **局限**: 仅在 LibriSpeech (朗读语音) 评估,未测对话/情感/多语言场景; 250 步采样推理慢; 无 MOS 主观评测; speaker similarity 0.617 低于 NS2/VoiceBox (但它们用 45-62x 数据)

## 核心问题

当前 SOTA TTS 系统 (VALL-E, NaturalSpeech 2, VoiceBox) 依赖海量标注数据 (数万到数十万小时),这对低资源语言和领域造成瓶颈。能否设计一个在 **不到 1k 小时标注数据** 下就能生成高可懂度语音的 diffusion TTS 系统,同时 **不依赖 phonemizer 和 phoneme alignment** ?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SESD 是一个 latent diffusion 框架,由三个预训练/冻结组件 + 一个可训练 diffusion 网络组成 [§4, Fig 1]:

1. **EnCodec encoder** (frozen): 24kHz waveform → 75Hz 128d continuous latents (quantization 前)
2. **ByT5-base encoder** (frozen): text transcript → character-level embeddings
3. **U-Audio Transformer (U-AT)** (trainable, 137M params): 在 latent space 做 diffusion denoising
4. **EnCodec decoder** (frozen): continuous latents → RVQ → waveform

关键洞察: 在 EnCodec latent space 做 diffusion 而非直接在波形或 mel 上,将 fine-grained 声学特征的建模卸载给预训练 autoencoder,diffusion 模型只需关注更易学习的 latent space [论文原文, §1]。10 秒音频 = 750 个 latent vectors (vs 24000 discrete tokens after RVQ),序列长度压缩 32x [§4]。

### 关键设计选择

#### 1. U-Audio Transformer (U-AT) 架构

为什么不用纯 Transformer 或纯 U-Net? [论文原文, §4]:
- 纯 Transformer 在 1504 帧的全分辨率序列上 **计算不可行** (O(n^2) attention)
- 纯 U-Net 在捕获长程依赖和融合条件信息上 **能力不足**

解决方案: 1D U-Net 先将 1504 帧 **下采样到 188 帧** (8x 压缩),再在压缩序列上使用 8 层 Transformer backbone [§4]。

U-Net 部分: 4 stage 1D 卷积 (从 2D image diffusion 的 iDDPM 设计改编),特征维度 512 [§4]。
Transformer 部分: 8 层, dim 512, 1D Dynamic Position Bias (DPB) [§4]。

**Register tokens**: 借鉴 Vision Transformer 的 register 机制 [Darcet et al., 2023],在 Transformer 输入前追加 8 个可学习 register tokens 作为 **全局记忆槽**,增强全局信息处理 [§4]。消融显示去除 register tokens WER 从 6.6% 升到 9.4% [Fig 4]。[agent 解读] register tokens 可能帮助 Transformer 在 downsampled 序列上维持全局一致性 (如韵律/节奏),避免局部注意力导致的碎片化。

#### 2. Position-Aware Cross-Attention

传统 cross-attention 依赖内容匹配来发现 text-audio 对应关系。在有限数据下,模型难以仅从数据学到正确的对齐 [论文原文, §4]。

解决方案: 引入显式位置信息到 cross-attention:

```
A_ij = q_i^T (k_j + f_θ(j/m))
```

其中 `f_θ(j/m)` 是轻量 MLP 将 transcript token 的 **归一化位置** `j/m` 映射为 position embedding,加到 key vector 上 [§4]。

[论文原文] 这使模型能 "直接搜索并关注 transcript 中的相关位置" [§4]。

消融 [Fig 4]: 用标准 cross-attention 替换 position-aware 版本,WER 从 2.3% 飙升到 **67.2%** — 这是所有消融中影响最大的组件。

#### 3. ByT5 Character-Aware Text Encoder

为什么用 ByT5 而不是 phonemizer + phoneme encoder? [论文原文, §1]:
- Phonemizer 和 aligner 本身会引入错误 [§2, ref 6]
- NaturalSpeech 2 需要 pitch 标注,VoiceBox 需要 phoneme duration 标注 — 增加数据要求
- ByT5 通过 **自监督预训练** 已学到丰富的语言知识,有助于小数据泛化

消融 [Fig 4]: 用 T5 (subword-level) 替换 ByT5 (byte-level),WER 从 2.3% 升到 47.6%。[agent 解读] byte/character-level 编码保留了对 TTS 至关重要的字母粒度信息 (如发音规则),而 subword tokenization 会丢失这种粒度。

#### 4. 非对称 Diffusion Loss Weighting

为什么不用标准 V-Weighting 或对称分布? [论文原文, §4]:
- 在 TTS 中,高噪声级 (low SNR) 是确定全局语音结构 (如词语位置) 的关键阶段
- 此时 corrupted latent 本身信号有限,conditioning information (transcript + prompt) 的贡献最大
- 应给高噪声级分配 **更多训练权重** 以改善 transcript alignment

具体实现: 分段加权函数 [§4]:
```
w(λ_t) = Cauchy(λ_t; -1, 4.8) / Z_c   if λ_t < -1 (高噪声)
        = N(λ_t; -1, 2.4) / Z_n        if λ_t ≥ -1 (低噪声)
```
Cauchy 分布的 **重尾特性** 在高噪声端提供持续的高权重 [Fig 2]。

消融 [Fig 4]: 对称 weighting WER 9.0%, VoiceBox-style weighting WER 16.9%, 本文非对称 weighting WER 2.3%。

### 训练策略

- **数据**: LibriSpeech clean + other (960h) [§5]
- **训练**: 250k steps, batch 64, 1x A6000 GPU [§4]
- **优化器**: AdamW, lr 2e-4, 1000-step warmup, cosine decay, weight decay 2e-4 [§4]
- **Dropout**: 0.1 在 feedforward/self-attention/cross-attention [§4]
- **EMA**: momentum 0.9999 [§4]
- **V-parameterization**: v = α_t ε − √(1−α_t²) x [§3]
- **Adaptive noise scheduler** [Kingma & Gao, 2023]: 减少 loss estimate 方差 [§4]

**Speaker-Prompted Generation** [§4]:
- 多任务训练: p=0.5 做 audio inpainting (concatenate clean prompt + noisy continuation)
- Prompt duration: Beta(mode=0.01, concentration=5) — 强调短 prompt 的困难情况
- 推理时 prepend reference audio + 对应 text

**Duration Prediction** [§4]:
- 训练时提供正确长度的 noisy latent,diffusion 内部隐式学习 phoneme duration
- 推理时: fine-tune ByT5-base 作为 seq2seq duration predictor (transcript → "4.51")
- Nucleus sampling (p=0.95),RMSE 1.4 秒

**Classifier-Free Guidance** [§4]:
- 训练时 p=0.1 drop text
- Text-only: DDPM sampler, w=5.0, 250 steps
- Speaker-prompted: DDIM sampler, w=8.0

## 实验

| 指标 | 本文 (SESD) | VITS-LJ | VITS-VCTK | MMS-TTS | VALL-E | NS2 | VoiceBox | Human | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER ↓ (text-only) | **2.3%** | 4.2% | 9.1% | 7.2% | - | - | - | 2.2% | LS test-clean | [Table 1] |
| WER ↓ (speaker-prompted) | **2.3%** | - | - | - | 5.9% | ~3% | ~2% | 2.2% | LS test-clean | [Fig 3] |
| Speaker SIM ↑ (prompted) | 0.617 | - | - | - | 0.580 | ~0.67 | ~0.68 | ~0.72 | LS test-clean | [Fig 3] |
| Training data | **960h** | LJSpeech | VCTK | - | 60kh | 44kh | 60kh | - | - | [§5, Fig 3] |

**消融实验** [Fig 4] (text-only WER, 越低越好):

| 配置 | WER |
| --- | --- |
| Standard Cross-Attention (替换 position-aware) | 67.2% |
| T5 Text Encoder (替换 ByT5) | 47.6% |
| Diffusion Transformer only (去除 U-Net) | 38.1% |
| VoiceBox Loss Weighting (替换非对称) | 16.9% |
| No Register Tokens | 9.4% |
| Symmetric Loss Weighting | 9.0% |
| SESD (100K steps) | 6.6% |
| **SESD (250K steps, full)** | **2.3%** |
| Human Reference | 2.2% |

**消融关键发现**:
- Position-aware cross-attention 是**最关键**组件 (去除后 WER 涨 28x) [Fig 4]
- ByT5 vs T5 差距巨大 (2.3% vs 47.6%),character-level 编码对 TTS 不可或缺 [Fig 4]
- U-AT 混合架构优于纯 Transformer (2.3% vs 38.1%) [Fig 4]
- 非对称 loss weighting 显著优于对称 (2.3% vs 9.0%) 和 VoiceBox-style (16.9%) [Fig 4]

## 局限性

1. **评估局限**: 仅在 LibriSpeech (英语朗读语音) 评估,未涵盖对话、情感、多语言等场景 [agent 解读]
2. **无 MOS 评测**: 全部使用客观指标 (WER, speaker similarity),缺乏主观音质评估 [agent 解读]
3. **推理速度慢**: 250 步采样 (DDPM/DDIM),相比 flow matching 的 10-50 步有明显差距 [agent 解读]
4. **Speaker similarity 仍有差距**: SIM 0.617 虽优于 VALL-E (0.580),但低于 NS2 (~0.67) 和 VoiceBox (~0.68),尽管后者用了 45-62x 数据 [Fig 3]
5. **依赖 EnCodec**: 音质天花板受限于 EnCodec 的重建质量,而 EnCodec 已被 DAC 等后续 codec 超越 [agent 解读, 基于 KB: [[模型库/EnCodec|EnCodec]]]
6. **ByT5 的限制**: frozen ByT5-base 的 byte-level 处理增加序列长度,可能影响长文本的效率 [agent 解读]
7. **无韵律控制**: 不支持 F0/energy 等韵律维度的显式控制 [agent 解读]

## 点评

**值得肯定的设计**:

1. **数据效率的系统性思考**: 不是单一技巧,而是从 latent space (降低建模难度)、text encoder (利用预训练知识)、loss weighting (优化训练信号)、architecture (平衡效率与表达力) 四个维度协同提升数据效率。消融实验 [Fig 4] 诚实地展示了每个组件的贡献。
2. **去除 phonemizer 依赖**: NaturalSpeech 2 和 VoiceBox 依赖 MFA phoneme alignment + duration annotation,SESD 用 frozen ByT5 + position-aware cross-attention 替代,降低了数据标注要求。这一思路与后来的 E2 TTS、F5-TTS 不谋而合。
3. **非对称 loss weighting 的 insight**: 认识到高噪声级是 TTS 中 text-speech alignment 的关键阶段,并据此设计 Cauchy-Normal 混合加权 — 这是对 TTS diffusion 训练动态的有价值的理解。

**需要注意的**:

1. **公平性问题**: 与 VALL-E/NS2/VoiceBox 的对比引用它们的论文数字,但这些系统的评估条件 (ASR 模型版本、评估集筛选) 可能不完全一致。
2. **WER 作为唯一可懂度指标的局限**: WER 2.3% 接近 human (2.2%),但不代表音质相当 — 需要 MOS 评测才能全面评估。
3. **数据效率 vs 绝对性能**: 960h 下 WER 和 SIM 不错,但 speaker similarity 仍明显低于使用大数据的系统。"数据效率" 与 "绝对质量" 是两个不同维度。

## 可复用的 idea

1. **非对称 Cauchy-Normal loss weighting**: 在 diffusion TTS 中,高噪声级对全局结构 (词位、韵律) 至关重要。用重尾 Cauchy 分布加权高噪声级,可能适用于任何需要强化条件对齐的 diffusion 系统。
2. **EnCodec continuous latent (pre-quantization) 作为 diffusion target**: 避免 RVQ 离散化的信息损失,同时享受 codec 的序列压缩优势 (32x)。这个思路可推广到其他 continuous-target 生成任务。
3. **Position-aware cross-attention**: 在 cross-attention key 上加 MLP 编码的归一化位置,用极低成本显著改善对齐。可能对其他序列到序列的条件生成有用 (如 music generation, dubbing)。
4. **Utterance-level duration prediction**: 用 seq2seq LM fine-tune 预测整体时长,让 diffusion 内部隐式解决 phoneme alignment — 简化 pipeline,避免 cascaded errors。

> [!review] 审阅结论: pass-with-fixes
> 审阅报告: [[_review/SESD-review.yml]]
> - [medium] traceability-gap: Fig 3 中 NS2/VoiceBox 的具体数值从图中读取而非精确表格数据,已标注 "~"
> - [medium] fact-inference-mixing: 局限性 §3-7 为 agent 推断,已标注 [agent 解读]
> - [low] weak-reusability: 可复用 idea §4 的 utterance-level duration prediction 描述可再具体化
> - 原则评估: 可复述✓ 可信赖✓(minor gaps) 可区分✓ 可定位✓ 不污染✓
