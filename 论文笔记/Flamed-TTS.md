---
type: paper
tier: deep
title: "Flamed-TTS: Flow Matching Attention-Free Models for Efficient Generating and Dynamic Pacing Zero-shot Text-to-Speech"
arxiv_id: "2510.02848"
source: "Sources/Flamed-TTS.pdf"
authors: [Hieu-Nghia Huynh-Nguyen, Huynh Nguyen Dang, Ngoc-Son Nguyen, Van Nguyen]
year: 2025
venue: "AAAI 2026"
tags: [TTS, zero-shot, flow-matching, attention-free, FACodec, non-autoregressive, efficiency, duration-modeling, temporal-diversity]
concepts: ["[[Conditional Flow Matching]]", "[[Duration Predictor]]", "[[Speech Factorization]]", "[[Residual Vector Quantization]]", "[[Non-autoregressive TTS]]"]
models: ["[[模型库/NaturalSpeech 2|NaturalSpeech 2]]", "[[模型库/NaturalSpeech 3|NaturalSpeech 3]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[Conditional Flow Matching]], [[Zero-shot Speech Synthesis]], [[Speech Factorization]], [[Residual Vector Quantization]]; 2 个待确认实体页: [[Duration Predictor]] [待确认], [[模型库/NaturalSpeech 3|NaturalSpeech 3]] [待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Flamed-TTS 是 OZSpeech 同组 (FPT Software AI Center) 的后续工作,属于 Zero-shot TTS 的 NAR/flow-based 阵营。与 OZSpeech 共享 FACodec 编解码器和 "learned prior 替代 Gaussian noise" 的核心思路,但 Flamed-TTS 进一步 (1) 将 attention 从 Denoiser 中完全移除 (换用 ConvNeXt), (2) 引入概率化 Duration & Silence Generator 实现动态节奏。在 CFM 演进谱系中,OZSpeech 探索单步采样的极限效率,Flamed-TTS 则在多步采样 (16-128 NFE) 下追求 attention-free 效率 + 时间多样性。
>
> **已有认知 -- Conditional Flow Matching**: CFM 通过学习 ODE 向量场将先验分布映射到目标分布。典型系统使用 Gaussian noise 作为起点 (F5-TTS 32 步, CosyVoice 2 10 步)。OZSpeech 首次证明用 learned prior 替代 Gaussian noise 可实现单步采样 (NFE=1);Shallow Flow Matching 则从 coarse 表示构造中间状态跳过前半段路径。Flamed-TTS 延续 OZSpeech 的 learned prior 思路,假设语义信息已编码在先验中,故 Denoiser 不再需要 self-attention 来建模全局语义关系。
>
> **已有认知 -- Duration Predictor**: 传统 NAR TTS 使用确定性 duration predictor (MSE loss on log-duration),产生固定时长,缺乏人类语音的自然变化。VITS 的 Stochastic Duration Predictor 首次用 flow-based 模型学习概率分布;DMOSpeech 2 用 GRPO RL 优化 duration policy;FlexSpeech 用 DPO 偏好对齐。Flamed-TTS 的 Duration Generator 也采用 flow matching 训练,是对概率化 duration 建模的又一实践。
>
> **已有认知 -- Speech Factorization**: FACodec (NaturalSpeech 3) 将语音分解为 prosody (1层)、content (2层)、acoustic detail (3层) + speaker identity,共 6 层 RVQ 码。OZSpeech 直接复用 FACodec 并在其 6 层码本空间建模。Flamed-TTS 同样基于 FACodec,但引入了 Code Decoder 阶段在码本空间生成离散 token,再用这些 token 构造 semantically enriched prior 供 flow matching 使用,形成两阶段 (Code Generator + Denoiser) 架构。

检索命中: [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓, [[Speech Factorization]]✓, [[Residual Vector Quantization]]✓ | 过滤: [[Duration Predictor]](pending-review), [[模型库/NaturalSpeech 3|NaturalSpeech 3]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 在 flow matching 的 Denoiser 中用 ConvNeXt 替代 self-attention,利用离散码构造的 semantically enriched prior 保持语义,同时引入概率化 Duration & Silence Generator 增强时间多样性。
> - **路线**: Phonemes → Phoneme Encoder → Duration/Silence Generator (CFM-based) → Length Regulator → Code Decoder (6层 FFT, 含 speech prompt) → 6-code folding+CNN → Denoiser (DiT w/ ConvNeXt, no self-attn) → FACodec Decoder → Waveform
> - **指标**: WER 4% (LibriSpeech test-clean, 5s prompt), UTMOS 3.87, SIM-O 0.51; RTF 0.028 (NFE=32, A100); 仅用 500h LibriTTS 训练,143M trainable params [Table 1, Table 2]
> - **可借鉴**: (1) 当 prior 携带足够语义时可移除 Denoiser 的 self-attention 换取 L^2*d → L*k*d 复杂度; (2) flow matching 训练 Duration/Silence Generator 实现 NAR 模型的动态节奏; (3) code folding (B*6*L*D → B*L*6D → CNN → B*L*D') 作为多码本合并策略
> - **局限**: 仅在 LibriTTS 500h 英语数据训练和评估,无多语言/大数据量验证; UTMOS 不及 Spark-TTS (4.33 vs 3.87); SIM-O/SIM-R 不及 Spark-TTS; 无 MOS 人工评测; FACodec 作为 frozen codec 限制了上界

## 核心问题

这篇论文要解决零样本 TTS 中的三个问题:

1. **推理效率**: 基于 Transformer 的 flow matching 模型 (如 F5-TTS) 的 self-attention 带来 O(L^2*d) 复杂度,导致推理延迟高 [§Introduction]
2. **时间多样性**: NAR TTS 的确定性 duration predictor 产生固定节奏,缺乏人类语音中的自然停顿和时长变化 [§Introduction]
3. **离散-连续表征的桥接**: 离散 token (RVQ) 有信息损失,但直接操作连续 mel spectrogram 的 in-context learning 模型需要大数据和高计算成本 [§Introduction]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Flamed-TTS 采用两阶段架构: Code Generator + Denoiser [Fig 1]:

**阶段一 -- Code Generator**: 接收 phoneme 序列,通过 Phoneme Encoder 编码后,Duration Generator 和 Silence Generator 分别预测每个 phoneme 的时长和后续静默时长,Length Regulator 据此扩展序列。扩展后的序列送入 Code Decoder (6 层 FFT blocks),逐层生成 FACodec 的 6 层离散码 (1 prosody + 2 content + 3 acoustic detail),同时融入 speech prompt 的对应层级码作为条件 [§Overall Architecture]。

**阶段二 -- Denoiser**: 将生成的 6 层离散码通过 code encoding + folding (B*6*L*D → B*L*6D) 再经 CNN 压缩为 B*L*D',作为 semantically enriched prior。加入噪声 (x0' = x_pr + tau*epsilon) 后,通过 **无 self-attention 的 DiT** (用 ConvNeXt 替代 self-attention) 执行 flow matching,生成 FACodec 的连续 latent。最终由 frozen FACodec Decoder 合成波形 [§Flow Matching Attention-Free Models]。

### 关键设计选择

#### 1. Attention-Free Denoiser: 为什么可以移除 self-attention?

**核心假设 [论文原文]**: 传统 flow matching 从 Gaussian noise 出发,prior 不含语义信息,模型必须通过 self-attention 建模全局依赖来推断语义;但如果 prior 本身已携带语义 (Code Generator 输出的离散码包含 prosody/content/acoustic 信息),则 Denoiser 的职责简化为 "声学细节增强",不再需要全局语义建模 [§Flow Matching Attention-Free Models]。

**替代方案**: 将 DiT 中的 multi-head self-attention 替换为 ConvNeXt module,复杂度从 O(L^2*d) 降至 O(L*k*d),其中 k 为卷积核大小 [§Flow Matching Attention-Free Models]。

**[agent 解读]**: 这一设计与 OZSpeech 的 learned prior 思路一脉相承,但 OZSpeech 仍保留了 attention (虽然实现单步采样),而 Flamed-TTS 将 "语义已在 prior 中" 的假设推到极致,直接移除 attention。实验表明 WER 4% 甚至低于保留 attention 的 F5-TTS (32%),支持了该假设 [Table 1]。但这一结论可能受限于 LibriTTS 500h 的相对简单场景 (单语、朗读语音)。

#### 2. Semantically Enriched Prior

传统 flow matching (如 Matcha-TTS, F5-TTS) 使用纯 Gaussian noise x0 ~ N(0,I) 作为初始点,Flamed-TTS 改用:

x0' = x_pr + tau * epsilon,  epsilon ~ N(0,I) [Eq. 7]

其中 x_pr 是 Code Generator 输出的离散码经 encoding+folding+CNN 后的连续表示。tau=1.0 用于训练 (增强多样性),tau=0.3 用于推理 (Table 5 消融显示 tau=0.3 最优) [§Flow Matching Attention-Free Models]。

**[agent 解读]**: 加噪的作用有两层: (1) 训练时 tau=1 使 vector field 更多样化,避免过拟合; (2) 推理时 tau=0.3 在多样性和质量间取平衡。与 OZSpeech 的区别: OZSpeech 不加噪直接用 prior 实现 NFE=1,Flamed-TTS 加噪后需要多步迭代 (NFE>=16) 但换来更好的质量 (UTMOS 3.87 vs OZSpeech 3.15, 5s prompt) [Table 1]。

#### 3. Hierarchical Code Decoder

Code Decoder 建模 6 层码的条件分布 [Eq. 1]:

p(q_{1:6} | P; p; psi) = p(q1 | P; p1; F_psi^1) * prod_{j=2}^{6} p(qj | q_{j-1}; pj; F_psi^j)

每层 FFT blocks 接收前一层码和同层级的 speech prompt 作为条件,自回归地 (层间,非时间维度) 生成 6 层码。Prior Loss L_prior 最小化该联合分布的负对数似然 [§Code Decoder]。

**[论文原文]**: 与 OZSpeech 的 Code Decoder 区别在于加入了 speech prompt conditioning -- OZSpeech 的 Code Decoder 在建码阶段不使用 speech prompt,Flamed-TTS 则在每层 FFT block 中拼接 speech prompt 的对应层码 [§Code Decoder, Fig 2a]。

#### 4. Probabilistic Duration & Silence Generator

**Duration Generator** [Eq. 2]: 用 optimal transport CFM 训练,学习从噪声到 log-domain duration 的向量场。推理时从 d0 ~ N(0,I) 出发,N 步 Euler 求解得到 duration 预测值。

**Silence Generator** [Eq. 3]: 同样用 CFM 训练,预测每个 phoneme 后的静默时长。输入 phoneme 序列前添加特殊 [SIL] token,其编码作为静默的模板被按预测时长复制插入 [Algorithm 1]。

**关键约束**: phoneme 最小 duration = 1 帧,silence 最小 duration = 0 帧 (可无静默) [Algorithm 1]。

**[论文原文]**: 为什么用 CFM 而非传统 MSE regression -- 确定性 duration predictor 对每个 phoneme 产生固定时长,无法捕获人类语音的固有变异性;概率化建模允许每次推理产生不同时长/停顿模式 [§Probabilistic Duration and Silence Generation]。

**[agent 解读]**: 将 duration prediction 也用 flow matching 训练是一个统一的设计选择 (duration/silence/acoustic generation 三者共享训练范式),简化了系统复杂度。但 Duration Generator 和 Silence Generator 各自独立,没有联合建模 phoneme 间的时长依赖,这可能限制长句子的韵律连贯性。

### 训练策略

**总损失** [Eq. 9]: L_total = L_prior + L_dur + L_sil + L_CFM + L_anchor

- L_prior: Code Generator 的码本预测 loss
- L_dur: Duration Generator 的 flow matching loss [Eq. 2]
- L_sil: Silence Generator 的 flow matching loss [Eq. 3]
- L_CFM: Denoiser 的 flow matching loss [Eq. 8]
- L_anchor: 辅助 loss,x_tilde_1 = xt + (1-t)*v_theta(xt,s,t) 与 x1 的 MSE [Eq. 10],用于稳定训练

**Codec**: FACodec (frozen),102M 参数,6 层 RVQ (1 prosody + 2 content + 3 acoustic detail) + speaker identity [§Method]

**数据**: LibriTTS 500h [§Experimental Setup]

**[agent 解读]**: L_anchor 的引入值得关注 -- 它实质上是对 flow 单步预测结果的正则化,可视为 consistency regularization 的简化版本。这可能是 Flamed-TTS 在低 NFE (如 16 步) 下仍能保持良好 WER 的原因之一。

## 实验

| 指标 | Flamed-TTS | F5-TTS | OZSpeech | VALL-E | NaturalSpeech 2 | Spark-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER (5s) | **0.04** | 0.32 | 0.05 | 0.19 | 0.09 | 0.11 | LibriSpeech test-clean | [Table 1] |
| UTMOS (5s) | 3.87 | 3.71 | 3.15 | 3.72 | 2.33 | **4.33** | LibriSpeech test-clean | [Table 1] |
| SIM-O (5s) | 0.51 | 0.57 | 0.39 | 0.46 | 0.35 | **0.61** | LibriSpeech test-clean | [Table 1] |
| SIM-R (5s) | 0.59 | 0.48 | 0.59 | 0.55 | 0.44 | **0.74** | LibriSpeech test-clean | [Table 1] |
| RTF (3s, NFE=32) | **0.028** | 0.26 | 0.013 | 0.86 | 1.66 | 1.06 | A100 | [Table 2] |
| #Params (trainable) | **143M** | 336M | 145M | 594M | 378M | 507M | -- | [Table 2] |
| F0_ACC (5s) | **0.92** | 0.83 | 0.83 | 0.79 | 0.84 | 0.91 | LibriSpeech test-clean | [Table 1] |
| EN_ACC (5s) | **0.74** | 0.68 | 0.67 | 0.41 | 0.28 | 0.50 | LibriSpeech test-clean | [Table 1] |

### 时间多样性分析 [Table 3]

| 指标 | Flamed-TTS | VALL-E (AR) | F5-TTS | OZSpeech | NaturalSpeech 2 |
| --- | --- | --- | --- | --- | --- |
| Speech Rate | 4.51 +/- 0.76 | 4.02 +/- 1.85 | 4.13 +/- 0.85 | 5.61 +/- 0.55 | 5.73 +/- 0.58 |
| #Pauses | 4.47 +/- 1.65 | 4.52 +/- 1.89 | 4.18 +/- 2.21 | 1.18 +/- 0.57 | 1.20 +/- 0.63 |
| MPaD | 0.149 +/- 0.054 | 0.355 +/- 0.292 | 0.256 +/- 0.164 | 0.030 +/- 0.012 | 0.032 +/- 0.015 |

Flamed-TTS 的 #Pauses 接近 AR 模型 VALL-E (4.47 vs 4.52),远高于传统 NAR 模型 OZSpeech/NaturalSpeech 2 (~1.2),说明 Silence Generator 有效引入了人类语音中的自然停顿 [Table 3]。

### NFE 消融 [Table 4]

NFE=16 时 SIM-O/SIM-R 即饱和 (0.51/0.59),WER 在 NFE>=4 后稳定在 4%; UTMOS 随 NFE 增大持续提升 (3.13 @ NFE=2 → 3.90 @ NFE=256) [Table 4]。

### 模型尺寸消融 [Table 6]

Small (76M) vs Base (143M): WER 和 SIM 几乎不变,UTMOS 下降 4.5-6%,说明 attention-free 架构在小模型下仍保持鲁棒的内容对齐能力 [Table 6]。

## 局限性

1. **数据规模和多样性不足**: 仅在 LibriTTS 500h 英语朗读数据上训练和评估,未验证多语言、大规模数据 (如 100K h) 场景下 attention-free 设计是否仍然有效 [agent 解读]
2. **UTMOS 和 SIM 不及 Spark-TTS**: Spark-TTS 用 100K h VoxBox 训练,UTMOS 4.33 vs 3.87,SIM-O 0.61 vs 0.51; 数据量差距 200x 使得直接比较不完全公平,但提示 Flamed-TTS 在自然度和说话人相似度上仍有提升空间 [Table 1]
3. **无 MOS 人工评测**: 仅用 UTMOS 自动评估,缺乏人工 MOS 验证 [agent 解读]
4. **FACodec 的上界限制**: frozen FACodec (102M, 6 层 RVQ) 的重建质量限定了系统上界; SIM-R 的引入 (与 FACodec 重建参考比较) 部分缓解了这个问题,但 SIM-R 本身也受 FACodec 质量影响 [agent 解读]
5. **Duration/Silence Generator 独立建模**: 两者各自独立预测,没有联合建模 phoneme 间的时长-静默依赖关系,可能影响长段落韵律连贯性 [agent 解读]
6. **Attention-free 假设的边界**: 论文假设 "语义在 prior 中充分表达所以不需要 attention" 仅在 500h 朗读数据下验证;更复杂的场景 (多说话人对话、情感语音、多语言) 中该假设可能不成立 [agent 解读]

## 点评

**与 OZSpeech 的关系**: Flamed-TTS 是同组 (FPT Software AI Center) 的后续工作,共享 FACodec + learned prior 的核心思路。OZSpeech 追求单步极限效率 (NFE=1, RTF 0.013) 但牺牲了 UTMOS (3.15);Flamed-TTS 在多步采样 (NFE=32, RTF 0.028) 下追求更好的质量 (UTMOS 3.87) 和时间多样性。两者共同验证了 "semantically enriched prior" 的有效性 -- learned prior 不仅能加速采样,还能让 Denoiser 简化架构。

**WER 优势显著但需审慎解读**: WER 4% 是所有 baseline 中最优,但需注意: (1) 这得益于显式 phoneme-to-speech alignment (duration predictor + length regulator),而非模型本身的 "听说能力"; (2) Spark-TTS (WER 11%) 和 F5-TTS (WER 32%) 没有显式 alignment 机制,比较基础不同。更公平的对比对象是同样使用 duration predictor 的 OZSpeech (WER 5%) 和 NaturalSpeech 2 (WER 9%) [Table 1]。

**Attention-free 的意义和局限**: 移除 self-attention 使 RTF 从 F5-TTS 的 0.26 降至 0.028 (同 NFE=32),提速约 9x,这在部署场景中有实际意义。但要注意 F5-TTS 使用 Vocos vocoder (13.5M) 而 Flamed-TTS 使用 FACodec decoder (102M),端到端参数量差异不大。真正的效率增益来自 Denoiser 计算量的减少。

**时间多样性是亮点**: Probabilistic Duration + Silence Generator 使 NAR 模型首次在 #Pauses (4.47) 和 MPaD (0.149) 上接近 AR 模型 VALL-E (4.52, 0.355),这在 NAR TTS 领域是一个有价值的探索方向 [Table 3]。

## 可复用的 idea

1. **Semantically enriched prior + attention-free Denoiser**: 当 prior 携带语义时,flow matching 的 vector field estimator 可以简化为纯局部卷积 (ConvNeXt),O(L^2*d) → O(L*k*d)。可推广到其他 conditional generation 场景 (如 singing voice synthesis, speech editing)
2. **CFM-based Duration & Silence Generator**: 将 duration 和 silence prediction 统一用 flow matching 训练,获得概率化输出,增强时间多样性。可替换现有 NAR TTS 的确定性 duration predictor
3. **Code folding**: 多码本表示 (B*N_q*L*D) → folding (B*L*N_q*D) → CNN 压缩为 (B*L*D'),作为多码本离散-连续桥接的通用策略
4. **L_anchor (consistency regularization)**: 对 flow matching 单步预测结果施加正则化 [Eq. 10],可与 RapFlow-TTS 的 velocity consistency 和 OZSpeech 的单步采样对比参考
5. **Speech prompt 条件注入 Code Decoder**: 在层级化 code 生成的每层 FFT block 中拼接同层级 speech prompt,比 OZSpeech 无 prompt 的方案获得更好的 prosody/acoustic 一致性

> [!review] 审阅 (2026-06-04, auto)
> **结论**: pass-with-fixes (3 issues: 1 medium, 2 low)
> - (medium, fixed) SIM-R OZSpeech 数据修正: "--" → 0.59 [Table 1]
> - (low) datasets 字段空 -- vault 无 LibriTTS 页,可接受
> - (low) FACodec 参数标注 [§Method] 可改为 [Table 2] 更精确
> 详见 `_review/Flamed-TTS-review.yml`
