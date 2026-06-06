---
type: paper
tier: deep
title: "DEX-TTS: Diffusion-based EXpressive Text-to-Speech with Style Modeling on Time Variability"
arxiv_id: "2406.19135"
source: "Sources/DEX-TTS.pdf"
authors: [Hyun Joon Park, Jin Sob Kim, Wooseok Shin, Sung Won Han]
year: 2024
venue: "arXiv preprint"
tags: [TTS, diffusion, expressive, style-transfer, reference-based, DiT, zero-shot, multi-speaker]
concepts: ["[[Diffusion-basedTTS]]", "[[StyleTransferinTTS]]", "[[DiffusionModel]]", "[[GlobalStyleTokens]]", "[[DurationPredictor]]", "[[F0Modeling]]", "[[MelSpectrogram]]", "[[ScoreMatching]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 6 个待确认实体页: [[Diffusion-basedTTS]], [[StyleTransferinTTS]], [[DiffusionModel]], [[GlobalStyleTokens]], [[DurationPredictor]], [[F0Modeling]])
> 自动生成,不保证完整覆盖所有相关知识。所有参考页均为 pending-review 状态,仅供参考 [待确认]。
> 检索命中: [[Diffusion-basedTTS]], [[StyleTransferinTTS]], [[DiffusionModel]], [[GlobalStyleTokens]], [[DurationPredictor]], [[F0Modeling]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: DEX-TTS 处于 reference-based expressive diffusion TTS 的交叉点。在 [[Diffusion-basedTTS]] 谱系中,它位于 Grad-TTS (SDE formulation, U-Net decoder) → U-DiT-TTS (引入 DiT 但受限于大 patch + sinusoidal embedding) → DEX-TTS (overlapping patchify + conv-freq embedding 充分发挥 DiT 优势) 的演进线上。在 [[StyleTransferinTTS]] 谱系中,它位于 GST (全局 pooling, 单一风格向量) → MetaStyleSpeech/StyleTTS (AdaIN 但仍单向量) → GenerSpeech (多级适配但 sum/concat 注入) → DEX-TTS (时间不变/变化双路 + 多层特征图 + VQ 离散化) 的演进线上。

**已有认知**: [[GlobalStyleTokens]] 指出 GST 的核心局限是 utterance-level 全局风格缺乏局部控制。DEX-TTS 的 T-V style 通过保留时间维度的 cross-attention 直接回应了这一局限。[[DurationPredictor]] 页记录了 MAS-based 对齐的成熟性。[[F0Modeling]] 页指出 pitch 信息在韵律建模中的关键地位。

**创新判断**: DEX-TTS 的核心创新不在于单个组件,而在于将 time-invariant/time-variant 二分法系统化地应用于 extractor-adapter 全链路设计,并用 VQ 约束 T-V 风格的信息瓶颈,同时改进 DiT 在 TTS 中的 patch 处理策略。在 [[Diffusion-basedTTS]] 的演进中,它是首个充分利用 DiT 结构优势(小 patch + overlapping + conv-freq embedding)的 expressive TTS。

## 速查

> [!summary] 速查
> - **一句话**: 将参考语音风格分解为时间不变(AdaIN + 多层特征图注意力池化)和时间变化(VQ + cross-attention)两路,配合改进的 DiT backbone(overlapping patchify + conv-freq embedding),实现无需预训练的高质量 reference-based expressive TTS
> - **路线**: phoneme → text encoder(RoPE + swish gate + T-V AdaLN) → MAS aligner → diffusion decoder(conv down → T-IV AdaIN adapter + T-V cross-attention adapter → DiT blocks with overlapping patchify → conv up) → mel → HiFi-GAN → waveform
> - **指标**: VCTK seen COS 85.31 / MOS-S 3.88, unseen COS 80.45 / MOS-S 3.81; ESD seen COS 82.71 / MOS-S 3.84, unseen COS 75.58 / MOS-S 3.52; LJSpeech(GeDEX-TTS) WER 6.55 / MOS-N 4.26 [Table 1, 2, 4]
> - **可借鉴**: (1) T-IV/T-V 二分法 + 对应 adapter 设计模式(AdaIN for global, cross-attention for temporal); (2) overlapping patchify 消除 DiT patch 边界伪影; (3) VQ 作为 T-V 风格的信息瓶颈,防止过度细节导致 WER 恶化; (4) 多层特征图 + attention pooling 提取跨层公共 T-IV 统计量; (5) adapter 中加入 diffusion timestep t 条件实现自适应风格注入
> - **局限**: RTF 0.297 远高于非 diffusion 方法(MetaStyleSpeech 0.034); 50 步 Euler 推理慢; 仅在 VCTK/ESD 中等规模数据集验证; 无大规模(>10K speaker)零样本实验; 代码已开源但未见后续大规模复现报告

## 核心问题

DEX-TTS 要解决的核心问题是: **reference-based expressive TTS 中,如何同时实现丰富的风格提取(well-represented style)和零样本泛化(generalization)?** [§1]

已有方法的两类缺陷 [§1]:
1. **风格提取不足**: MetaStyleSpeech/StyleTTS 使用 pooling 将参考表示压缩为单一风格向量,丢失了时间变化的风格信息(如语调起伏、局部韵律)
2. **风格注入受限**: YourTTS/GenerSpeech 使用 summation/concatenation 注入风格,泛化能力不足,尤其在 zero-shot 场景下效果差

附加问题: DiT 架构在 TTS 中的利用不充分 — U-DiT-TTS 虽引入 DiT,但大 patch size + sinusoidal position embedding 限制了 DiT 捕获细粒度表示的能力 [§2.1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DEX-TTS 基于标准 diffusion TTS 框架(text encoder → aligner → diffusion decoder → vocoder),核心创新在于 encoder-adapter 体系的风格处理 [§3.2, Fig 1]:

1. **Text Encoder**: 8 层 Transformer,引入 RoPE 位置编码 + RetNet swish gate + AdaLN 注入 T-V 风格 h_ev [§3.2, §A.1]
2. **Aligner**: 基于卷积的 Duration Predictor + MAS 对齐,将 h_text 扩展到 mel 帧长度 [§3.2]
3. **Diffusion Decoder**: 以 EDM [34] formulation 为基础,包含 conv down/up blocks + T-IV/T-V adapters + DiT blocks [§3.2]
4. **Vocoder**: 预训练 HiFi-GAN [§4.1]

风格信息分为两路 [§3.2]:
- **T-IV (Time-Invariant)**: 全局特征(说话人身份、情感类别),通过 AdaIN 注入,不随时间变化 [论文原文]
- **T-V (Time-Variant)**: 时间变化特征(语调、韵律起伏),保留时间维度通过 cross-attention 注入 [论文原文]

### 关键设计选择

#### 1. 时间不变风格建模 (T-IV) [§3.3]

**T-IV Encoder**: 多层残差卷积块 + Instance Normalization。Instance Norm 的选择是为了消除 batch 内不同样本的时间信息差异,保持每个实例的全局特征 [论文原文]。使用多层特征图(h_inv ∈ R^{L×C×T},L 为层数)堆叠所有卷积块输出,包含跨层的 T-IV 信息 [§3.3]。

**为什么用多层特征图而非最后一层?** 消融实验 a) 将 T-IV adapter 替换为仅使用最后一层的简单 AdaIN,WER 从 8.34 恶化到 11.91 [Table 3]。[论文原文] 解释: 多层特征图中的公共特征(通过 attention pooling 提取)比单层更能代表稳定的 T-IV 风格,且不干扰文本内容。

**T-IV Adapter**: AdaIN(h_diff, μ, σ),其中 μ 和 σ 通过 attention pooling 从各层的 channel-wise 均值/标准差中提取,并以 diffusion timestep t 为条件 [Eq. 3-4]:

```
μ̃ = [t; avg(h¹_inv); ...; avg(hᴸ_inv)], μ = AP(μ̃)
σ̃ = [t; std(h¹_inv); ...; std(hᴸ_inv)], σ = AP(σ̃)
```

**为什么 AdaIN 适合 T-IV?** [论文原文] AdaIN 通过归一化消除实例特定的统计量后用参考的统计量替代,天然适合反映与时间无关的全局风格。[agent 解读] 这与图像风格迁移中 AdaIN 的经典用法一致(Huang & Belongie, 2017),将"内容"的统计量替换为"风格"的统计量。

#### 2. 时间变化风格建模 (T-V) [§3.4]

**T-V Encoder**: 残差卷积块 + Layer Normalization(而非 Instance Norm),保留时间关系 [§3.4]。输出两路:
- h_ev: 卷积输出 + pitch 信息(GRU on log-F0)→ channel-wise pooling → 注入 text encoder(通过 AdaLN)[§3.4]
- h_dv: 卷积输出 + VQ(codebook K=512, D=192)+ pitch → 不做 pooling,保留时间维度 → 注入 diffusion decoder(通过 cross-attention)[§3.4]

**为什么 T-V 分成两路(h_ev 和 h_dv)?** [agent 解读] h_ev 注入 text encoder 需要做 channel-wise pooling 因为 text 序列长度与参考语音不同,需要全局 T-V 信息;h_dv 注入 diffusion decoder 时保留时间维度因为 decoder 在 mel 帧空间操作,可以做帧对帧的 cross-attention。

**VQ 的关键作用**: VQ 将连续 T-V 表示映射到离散码本,去除连续空间中的噪声,获得精炼的风格信息作为泛化特征 [§3.4]。消融实验 f) 显示去掉 VQ 后 COS 略微提升(76.38 vs 75.58 unseen)但 WER 急剧恶化(15.70 vs 8.34 seen)[Table 3]。[论文原文] 解释: 不加 VQ 时,h_dv 包含过度细节的风格信息,虽然提高了相似度但严重破坏了语音其他维度的质量。[agent 解读] 这实质上是一个信息瓶颈(information bottleneck)设计: VQ 限制了 T-V 风格的信息容量,迫使模型只保留最重要的时变模式。

**T-V Adapter (cross-attention)**: Q = IN(h_diff)·W_q, K = h_dv·W_k, V = h_dv·W_v [Eq. 6]。对 query 应用 Instance Norm 保持实例级特征用于计算 attention scores [§3.4]。

**AdaLN 在 Text Encoder 中的作用**: AdaLN(h_text, h_ev) = g(h_ev)·LN(h_text) + b(h_ev) [Eq. 5],在每个 MHSA 和 FFN 之后注入 T-V 风格。消融 c) 显示移除 h_ev 导致最大性能下降(COS: 78.26 seen, 71.91 unseen),甚至超过移除 T-IV [Table 3]。[论文原文] 解释: text encoder 输出用于 prior loss 的初始 mel 表示计算,风格在 text encoder 中的反映对系统影响巨大。

#### 3. DiT backbone 改进 [§3.2]

**Overlapping Patchify**: 使用 kernel size = 2P-1, stride = P 的卷积层进行 patch 化,允许相邻 patch 之间有重叠 [§3.2]。

**为什么要重叠?** [论文原文] 消除 patch 边界处的伪影(boundary artifacts),实现更自然的语音合成。Table 4 (right) 显示 overlapping 将 WER 从 7.31 降至 6.55(LJSpeech)[Table 4]。

**Conv-freq Patch Embedding**: 时间轴使用卷积层取 time-wise average 获得相对位置嵌入 PE_T ∈ R^{C×1×T₂};频率轴使用固定大小可学习参数 PE_F ∈ R^{C×F₂×1} [§3.2]。

**为什么时间轴用卷积而频率轴用固定参数?** [论文原文] 语音长度是可变的,需要能处理训练时未见长度的嵌入方式;频率大小(mel bins)在语音合成中是固定的,可以用固定参数。Table 4 (right) 消融: sin-cos 编码 WER 16.37,time-freq 固定参数 WER 8.01,conv-freq WER 7.31(无 overlap)/ 6.55(有 overlap)[Table 4]。

#### 4. Diffusion timestep 条件化 adapter [§3.3-3.4]

T-IV 和 T-V adapter 都以 diffusion timestep t 为额外条件。消融 g) 移除 t 条件后整体性能下降 [Table 3]。[论文原文] 在扩散网络的迭代去噪过程中,自适应地反映风格是必要的。[agent 解读] 不同去噪阶段需要不同程度的风格信息 — 早期阶段可能更需要全局结构信息,后期需要细节修正。

### 训练策略

**Loss Function** [§3.5]:
- L_dur: duration predictor MSE loss (log domain)
- L_prior: ||h_mel - x||²₂,初始 mel 表示与 GT 的 MSE
- L_diff: λ(t)||D_θ(x_t, t, h_{mel,inv,v}) - x||²₂,EDM formulation with pre-conditioning [Eq. 7]
- L_vq: ||h - sg(e)||²₂,commitment loss [§3.5]

**训练配置** [§4.1]:
- VCTK 1000 epochs, ESD 1500 epochs, LJSpeech 2000 epochs
- Adam, lr=10⁻⁴, batch size 32
- 单张 NVIDIA 3090 GPU
- 80 mel bins, FFT 1024, hop 256, window 1024
- 推理: 50 步 Euler solver

## 实验

| 指标 | DEX-TTS (VCTK seen) | DEX-TTS (VCTK unseen) | StyleTTS (VCTK seen) | StyleTTS (VCTK unseen) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER (%) | 7.85 | 5.84 | 7.72 | 6.58 | VCTK | [Table 1] |
| COS (×100) | 85.31 | 80.45 | 82.93 | 77.90 | VCTK | [Table 1] |
| MOS-N | 3.75 | 3.76 | 3.57 | 3.53 | VCTK | [Table 1] |
| MOS-S | 3.88 | 3.81 | 3.70 | 3.65 | VCTK | [Table 1] |

| 指标 | DEX-TTS (ESD seen) | DEX-TTS (ESD unseen) | GenerSpeech (ESD seen) | GenerSpeech (ESD unseen) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER (%) | 8.34 | 8.35 | 12.75 | 11.78 | ESD | [Table 2] |
| COS (×100) | 82.71 | 75.58 | 75.09 | 70.54 | ESD | [Table 2] |
| MOS-N | 3.73 | 3.57 | 3.33 | 3.28 | ESD | [Table 2] |
| MOS-S | 3.84 | 3.52 | 3.28 | 2.78 | ESD | [Table 2] |

| 指标 | GeDEX-TTS | Grad-TTS | CoMoSpeech | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER (%) | 6.55 | 7.70 | 8.21 | 6.56 | LJSpeech | [Table 4] |
| COS (×100) | 91.75 | 91.37 | 91.58 | - | LJSpeech | [Table 4] |
| MOS-N | 4.26 | 4.16 | 4.13 | 4.60 | LJSpeech | [Table 4] |

**模型复杂度对比** [Table 8]:
- DEX-TTS: **18.36M params** (最小) vs MetaStyleSpeech 27.67M, YourTTS 94.60M, GenerSpeech 51.64M, StyleTTS 68.34M
- DEX-TTS: RTF 0.297 (最慢) vs MetaStyleSpeech 0.034, StyleTTS 0.038

**NFE 敏感性** [Table 9]: NFE 10/25/50 的 WER 分别为 6.72/7.04/6.84,COS 分别为 82.77/82.84/82.88(VCTK 平均),CMOS-N 差异仅 -0.07/-0.06/0,说明少步推理仍可保持竞争力。

**消融核心发现** [Table 3, ESD]:
- 移除 h_ev(T-V text encoder style)→ COS 下降最严重(78.26/71.91 vs 82.71/75.58)
- 移除 VQ → COS 略升但 WER 严重恶化(15.70 vs 8.34)
- 移除 timestep conditioning → 整体小幅下降

## 局限性

1. **推理速度**: diffusion 的迭代去噪导致 RTF 0.297,远慢于非 diffusion 方法(MetaStyleSpeech 0.034)。即使 NFE=10(RTF 0.087)仍慢于多数非 diffusion baseline [Table 8, 9]。论文讨论了 consistency model 作为未来加速方向 [§E]。

2. **数据规模有限**: 仅在 VCTK(109 speakers)和 ESD(10 English speakers)上验证,未在大规模数据集(LibriTTS/LibriLight 级别)上测试。零样本实验仅 10 unseen speakers(VCTK)和 2 unseen speakers(ESD),统计置信度有限 [§4.1]。

3. **mel-spectrogram 中间表示**: 仍使用 mel → HiFi-GAN 两阶段流程,受 vocoder 质量约束。未探索端到端波形生成或 codec token 空间的可能性。

4. **VQ 信息瓶颈的权衡**: VQ 改善了 WER 但略微牺牲了 COS(unseen: 76.38 without VQ vs 75.58 with VQ)[Table 3]。这个权衡是固有的还是可以通过更好的 VQ 设计缓解,论文未深入探讨。

5. **与当代方法对比不足**: 未与 StyleTTS 2(2023,SOTA expressive TTS)、MegaTTS 2 等更新的强 baseline 对比。baseline 中 StyleTTS 为 v1(2022)版本。

## 点评

**优势**:
- T-IV/T-V 二分法设计清晰,每个选择都有对应消融实验支撑,实验设计扎实
- 18.36M 参数量是所有对比方法中最小的,参数效率极高
- 无需预训练依赖(与 YourTTS 需 speaker encoder、GenerSpeech 需 style extractor、StyleTTS 需 aligner 预训练形成对比),作为独立模型部署更简洁
- DiT 改进(overlapping patchify + conv-freq embedding)在 GeDEX-TTS 实验中独立验证了通用价值,WER 达到 GT 水平

**不足**:
- 推理效率是实际部署的主要障碍,论文未实际实现 consistency distillation 或其他加速方案
- 实验规模(特别是 zero-shot speakers 数量)偏小,难以确信在大规模场景下的泛化能力
- 与 2023-2024 年更强的 baseline(如 StyleTTS 2)相比,GeDEX-TTS 的 MOS-N 4.26 可能仍有差距,但论文未直接对比
- T-SNE 可视化 [Fig 4] 定性有说服力(T-IV/T-V 分别按情感/说话人聚类),但缺少定量解耦度量

## 可复用的 idea

1. **T-IV/T-V 二分法作为 reference encoder 的通用设计模式**: 任何需要从参考信号提取风格的系统(不限于 TTS,如 VC、SVS)都可以按时间不变性拆分为两路,分别用 AdaIN 和 cross-attention 注入。关键是匹配: Instance Norm ↔ 全局统计量 ↔ AdaIN; Layer Norm ↔ 保留时序 ↔ cross-attention。

2. **VQ 作为信息瓶颈调节 similarity-quality 权衡**: 当风格注入导致内容质量下降时(典型症状: COS 上升但 WER 恶化),VQ 是一个有效的信息限流器。codebook size 可以作为控制信息量的超参数。

3. **Overlapping patchify 消除 DiT patch 边界伪影**: 对任何使用 DiT/ViT 处理连续信号(mel, waveform latent)的系统,overlapping patch 是一个低成本高收益的改进(kernel = 2P-1, stride = P)。

4. **多层特征图 + attention pooling 提取跨层统计量**: 比仅用最后一层更稳定的风格表示提取方法,可迁移到任何层级化 encoder 的风格/speaker 表示提取中。

5. **Adapter 中加入 diffusion timestep 条件**: 在 diffusion decoder 的 style adapter 中加入 timestep 作为额外条件,使风格注入在不同去噪阶段自适应调整,是 diffusion-based conditional generation 的通用改进。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节含完整因果解释,每个设计选择配消融实验 |
> | 可信赖 | pass | 核心数字全部交叉验证通过,出处标注覆盖率~90% |
> | 可区分 | pass | [论文原文]/[agent 解读]标注一致且覆盖率高 |
> | 可定位 | pass | 双维度谱系定位(Diffusion-basedTTS + StyleTransferinTTS) |
> | 不污染 | pass | 内容准确,无 overclaim |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/DEX-TTS-review.yml`
