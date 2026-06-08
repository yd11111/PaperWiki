---
type: paper
tier: deep
title: "CoFi-Speech: Speaking from Coarse to Fine — Improving Neural Codec Language Model via Multi-Scale Speech Coding and Generation"
arxiv_id: "2409.11630"
source: "Sources/CoFi-Speech.pdf"
authors: [Haohan Guo, Fenglong Xie, Dongchao Yang, Xixin Wu, Helen Meng]
year: 2024
venue: "arXiv"
tags: [speech-codec, multi-scale, coarse-to-fine, LM-TTS, zero-shot-TTS, recency-bias, VQ-VAE, multi-resolution]
concepts: ["[[CodecLanguageModel]]", "[[ResidualVectorQuantization]]", "[[Single-codebookvsMulti-codebook]]", "[[CodebookCollapse]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[MelSpectrogram]]", "[[NeuralVocoder]]", "[[SpeakerEmbedding]]"]
models: ["[[BigVGAN]]", "[[CosyVoice]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["WenetSpeech4TTS"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页: [[ResidualVectorQuantization]]✓, [[CodebookCollapse]]✓, [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[CodecLanguageModel]][待确认], [[Single-codebookvsMulti-codebook]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ResidualVectorQuantization]]✓, [[CodebookCollapse]]✓, [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review), [[MelSpectrogram]](pending-review), [[AudioTokenizerTaxonomy]](pending-review), [[CodecTrainingObjectives]](pending-review) | 未命中但可能相关: 无

**谱系定位:** CoFi-Speech (2024.09, CUHK + 小红书) 是 SoCodec (同一团队、同月发表) 的姊妹工作。两者共享核心思路 --- 将语音压缩为多分辨率离散表示以降低 LM 建模负担,但解决问题的角度不同: SoCodec 关注流间有序性 (ordered multi-stream representation),CoFi-Speech 关注时间尺度的分层 (multi-scale temporal resolution)。在 KB 的 codec 谱系中,CoFi-Speech 属于 **Multi-Scale RVQ (MSRVQ)** 路线的早期代表 --- [[ResidualVectorQuantization]] 页记录了 MSRVQ (SNAC, LLM-Codec) 作为"不同层在不同时间分辨率量化"的变体,CoFi-Codec 正是这一思路的具体实现,但其编解码器结构更接近 VQ-VAE 而非标准 RVQ。

**已有认知:** (1) [[CodecLanguageModel]] 记录了 CLM 的序列长度挑战: 多层 codec token 展开后序列极长,以及 delay pattern 和 token 压缩等应对策略。CoFi-Speech 的 SoS (Stack-of-Scale) 方案是对这一挑战的另一种回答 --- 用多个独立 LM 分层生成而非单 LM 处理长序列。(2) [[Single-codebookvsMulti-codebook]] 记录了向 fewer codebooks 的趋势和 MSRVQ 折中方案。CoFi-Codec 的三尺度设计 (120ms/40ms/20ms) 与 SNAC 的多尺度思路一致。(3) [[CodebookCollapse]] 记录了 VQ 训练中码本利用率低的现象及解决方案。CoFi-Speech 提出的 Scale-Wise Nested Dropout (SWND) 直接针对"高尺度表示坍缩"问题,与 [[QuantizerDropout]] 的 random layer dropout 机制高度相关但目标不同 --- SWND 防止高尺度序列被忽略,而 quantizer dropout 支持可变比特率。(4) [[LLM-basedTTS]] 记录了 VALL-E 的 AR+NAR 两阶段范式和多流建模复杂性。CoFi-Speech-SoS 用多个 LM 级联替代 AR+NAR 两阶段,是一种新的层级生成思路。

**创新判断:** 相对 KB 已有知识,核心贡献: (1) Multi-scale temporal codec --- 与同期 SoCodec 的 multi-stream ordered codec 形成互补; (2) Stack-of-Scale generation --- 用 LM 级联实现 coarse-to-fine 生成,每个 LM 只需处理短序列; (3) Scale-Wise Nested Dropout --- 防止高尺度表示坍缩的训练技巧; (4) Attention 可视化直接证明多尺度建模缓解了 recency bias。

> [!summary] 速查
> - **一句话**: 提出多尺度 codec (CoFi-Codec) + 多尺度生成 (CoFi-LM) 的 coarse-to-fine CLM-TTS 框架,通过 stack-of-scale 多 LM 级联从高时间尺度到低时间尺度逐步生成,显著缓解 recency bias 并超越 CosyVoice
> - **路线**: Speech → Mel Spectrogram → Multi-scale Encoder (ResNet + strided conv, 3 级下采样) → VQ at each scale (120ms/40ms/20ms, codebook 16384, s3/s2/s1) + ECAPA-TDNN global embedding → Multi-scale Decoder (VQ add + ResNet + transposed conv 上采样) → Mel → BigVGAN Vocoder → Waveform; TTS: Text (BPE 8192) + Ref Embedding → CoFi-LM (SoS: 3 个级联 12L Transformer, 各自生成一个尺度的 token) → Speech Tokens → CoFi-Codec Decoder → Audio
> - **指标**: SoS NMOS 4.42/SMOS 3.90 vs CosyVoice NMOS 4.28/SMOS 3.70, VALL-E NMOS 3.46/SMOS 2.92, SoCodec-TTS NMOS 4.30/SMOS 3.51 [Table I]; 三尺度 SoS CER 2.60%/SIM 87.4 vs 单尺度 CER 4.35%/SIM 84.5 [Fig 3]; 数据集 WenetSpeech4TTS Basic 7k hours
> - **可借鉴**: (1) Stack-of-Scale 级联: 用多个小 LM 分别生成不同尺度 token,高尺度 LM 的 hidden states 作为 prompt + 上采样 conditioning 传给低尺度 LM,避免单 LM 处理长序列; (2) Scale-Wise Nested Dropout: 训练时随机 mask 低尺度序列,迫使高尺度序列保留有效信息,防止表示坍缩; (3) Residual 减法分离: encoder 输出与 decoder 累积输出做差取残差再量化,实现信息逐层细化
> - **局限**: 仅在中文 (WenetSpeech4TTS 7k hours) 上验证; 编码操作在 Mel 域而非波形域; BigVGAN vocoder 非联合训练; SoS 需要 3 个独立 LM,参数量和训练成本更高; 未报告推理延迟/RTF; MOS 测试仅 10 名评估者、100 条子集

## 核心问题

1. CLM-TTS 中"recency bias"导致 LM 忽视粗粒度信息 (音素、韵律、说话风格),如何通过多尺度表示和生成来解决? [§I]
2. 如何设计 multi-scale codec 使其在不同时间分辨率下学到有效的离散表示,同时保持高重建质量? [§II-A]
3. Chain-of-Scale (单 LM 串联) vs Stack-of-Scale (多 LM 级联): 哪种 coarse-to-fine 生成策略更有效? [§II-B]
4. Scale-Wise Nested Dropout 如何防止高尺度表示坍缩? [§II-A]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CoFi-Speech 由两部分组成 [§II, Fig 1]:

1. **CoFi-Codec** [§II-A]: 将语音编码为多尺度离散表示 + 解码重建
   - 多级 encoder: Mel spectrogram → 3 个 encoder block (ResNet + strided conv),逐级下采样得到 3 个不同时间分辨率的编码序列 [§II-A]
   - 多级 decoder: 从最高尺度 (最粗) 开始,逐级量化残差并上采样重建 Mel [§II-A]
   - ECAPA-TDNN reference encoder 提取全局 embedding g (speaker identity, style, environment) [§II-A]

2. **CoFi-LM** [§II-B]: 两种 LM 策略从 coarse-to-fine 生成多尺度 token
   - Chain-of-Scale (CoS): 单个 LM 串联生成所有尺度的 token [§II-B, Fig 1(b)]
   - Stack-of-Scale (SoS): 多个 LM 级联,每个 LM 生成一个尺度 [§II-B, Fig 1(c)]

### 关键设计选择

**Multi-Scale Codec 的残差分离机制** [§II-A]:

CoFi-Codec 的核心在于如何在不同尺度间分配信息 [论文原文]:
- 第 i 个 decoder block 先计算残差: r_i = e_i - d_i,即从 encoder 输出 e_i 中减去当前已解码的信息 d_i [§II-A]
- 对残差 r_i 做 VQ 得到量化序列 s_i [§II-A]
- 将 s_i 加上 d_i 和全局 embedding g,经 ResNet + 上采样得到下一级 decoder 输入 d_{i-1} [§II-A]
- 最高尺度 decoder 使用全零序列作为初始 d [§II-A]

[agent 解读] 这种"减法取残差"的设计与 RVQ 的"残差递归量化"本质一致,但发生在 feature map 层面而非量化 embedding 层面。每一级量化的是不同时间分辨率的残差,而非同一分辨率的逐层细化。这使得不同尺度的 token 天然编码不同粒度的信息 --- 高尺度 (长 frameshift) 编码全局结构,低尺度 (短 frameshift) 编码局部细节。

**Scale-Wise Nested Dropout (SWND)** [§II-A]:

训练时随机采样 b ~ [0, N_s - 1],将低尺度序列 s_{1:b} 全部 mask 为零,只保留高尺度序列进行重建 [§II-A] [论文原文]。b=0 时不做 mask。

[论文原文] 目的: 防止模型过度依赖低尺度 (高分辨率) 序列,导致高尺度序列退化为空信息 --- 即"high-scale representation collapse" [§II-A]。

[agent 解读] SWND 与 SoCodec 的 stream-wise nested dropout 设计哲学一致 (两者共享核心作者 Haohan Guo),但作用维度不同: SoCodec 的 dropout 作用于同一时间分辨率的不同 stream,CoFi-Speech 的 dropout 作用于不同时间分辨率的 scale。两者都受 ordered autoencoder (Rippel et al., 2014) 启发,用 dropout 强制信息有序分布。与 [[CodebookCollapse]] 中的 Quantizer Dropout 不同,SWND 不是为了可变比特率,而是为了防止高尺度表示坍缩。

**Stack-of-Scale (SoS) Generation** [§II-B, Fig 1(c)]:

SoS 用一组级联的 LM 逐尺度生成 [论文原文]:
1. GPT3 从 text + reference embedding 生成最高尺度 (最粗) 序列 s3 [§II-B]
2. GPT3 最后一层 hidden states h3 作为 prompt 传给 GPT2; h3 同时上采样后加到 GPT2 的 speech token embedding 上,强化尺度间对齐 [§II-B]
3. 递归重复直到生成所有尺度 [§II-B]

[论文原文] 优势: stage-wise 方法让每个 LM 只处理短序列,避免单 LM 面对长上下文 [§II-B]。

[agent 解读] SoS 的设计灵感与 AudioLM 的 semantic → coarse-acoustic → fine-acoustic 三阶段级联相似,但 AudioLM 的三阶段处理的是同一时间分辨率下的不同信息层级 (语义→声学),而 SoS 处理的是不同时间分辨率的序列。SoS 的 hidden state 上采样 conditioning 是关键 --- 它不仅传递粗粒度信息,还显式建立了尺度间的时间对齐,比简单的 prompt concatenation 更有效。

**Chain-of-Scale (CoS) Generation** [§II-B, Fig 1(b)]:

CoS 是 SoS 的简化版本: 单个 LM 依次生成 s3 → s2 → s1 [§II-B] [论文原文]。对多流序列使用 delay pattern [§II-B]。

[论文原文] 问题: 多尺度序列串联导致更长的链,计算成本和建模复杂度更高 [§II-B]。

[agent 解读] CoS 本质上是把多尺度问题转化为超长序列建模问题 --- 用显式的尺度串联替代 VALL-E 式的 AR+NAR 两阶段。从 Table I 看,CoS NMOS 4.12 已优于 VALL-E 的 3.46,但低于 SoS 的 4.42,说明"知道要先粗后细"(CoS) 就已有帮助,但"分开建模不同粒度"(SoS) 效果更好。

### 训练策略

**CoFi-Codec 训练** [§III-A]:
- 损失: L_c = lambda_vq * L_vq + lambda_reg * L_reg + lambda_adv * L_adv [Eq. 1]
  - L_vq: VQ 前后 embedding 的 L2 距离均值 [§II-A]
  - L_reg: Mel spectrogram 的 L2 回归损失 [§II-A]
  - L_adv: 对抗损失,使用 Mega-TTS 的 discriminator [§III-A]
  - 权重: lambda_vq=1, lambda_reg=1, lambda_adv=0.1 [§III-A]
- SWND 采样概率: p0=0.8, p1=0.1, p2=0.1 [§III-A]
- EMA 更新 codebook, decay rate 0.99 [§III-A]
- 三尺度配置: 120ms (单流) + 40ms (单流) + 20ms (四流, OPQ) [§III-A]
- Codebook: 每流 16384 entries [§III-A]
- 数据: WenetSpeech4TTS Basic, 7k hours, 16kHz, 80-dim Mel, 10ms frameshift [§III-A]
- Optimizer: AdamW, 100K iterations, batch size 1.6k seconds, LR 3e-4 → 1e-4 exponential decay [§III-A]

**CoFi-LM 训练** [§III-A]:
- 12-layer decoder-only Transformer, 1024 dim [§III-A]
- 多流序列使用 delay pattern [§III-A]
- Sampling: top-p 0.8, top-k 50, repetition penalty 2.0 [§III-A]
- Text: BPE 8192 tokens [§III-A]
- AdamW, 100K iterations, batch size 1.6k seconds [§III-A]

[agent 解读] 注意 SoS 需要为每个尺度训练独立的 LM (各 12L),参数量为 CoS 的 3 倍。论文未讨论这一成本增加,但 Fig 3 中 CoS-large (24L) 仍不及 SoS (3x12L),说明 SoS 的优势不仅来自参数增加,而是来自级联架构本身。

## 实验

| 指标 | CoFi-SoS | CoFi-CoS | SoCodec-TTS | VALL-E | CosyVoice | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NMOS ↑ | **4.42** | 4.12 | 4.30 | 3.46 | 4.28 | WenetSpeech4TTS | [Table I] |
| SMOS ↑ | **3.90** | 3.75 | 3.51 | 2.92 | 3.70 | WenetSpeech4TTS | [Table I] |

注: CosyVoice 使用 >100k hours 数据训练,CoFi-Speech 和其他 baseline 使用 7k hours [§IV-A]。

**Multi-scale Coding 消融** [§IV-B, Fig 2]:

| 配置 | MCD ↓ | CER ↓ | 说明 |
| --- | --- | --- | --- |
| 三尺度 CoFi-Codec (全部) | 最低 | 最低 | 多尺度 + SWND,重建最优 [Fig 2] |
| 三尺度 w/o SWND (全部) | 略高 | 略高 | 无 SWND 时高尺度学不到有效信息 [Fig 2] |
| 三尺度 CoFi-Codec (top-2) | 低 | 低 | 仅用 s3+s2 即可良好重建 [Fig 2] |
| 三尺度 w/o SWND (top-2) | 高 | 高 | 无 SWND 时高尺度几乎不含信息 [Fig 2] |
| 单尺度 CoFi-Codec (20ms) | 中 | 中 | 单尺度无多尺度优势 [Fig 2] |
| EnCodec (4-stream, 20ms) | 中 | 中 | 传统 RVQ codec [Fig 2] |

**Multi-scale Generation 消融** [§IV-C, Fig 3]:

| 配置 | CER ↓ | SIM ↑ | 出处 |
| --- | --- | --- | --- |
| 三尺度 SoS | ~2.60 | ~87.4 | [Fig 3] |
| 三尺度 CoS | ~3.50 | ~85.2 | [Fig 3] |
| 两尺度 SoS | ~2.90 | ~86.8 | [Fig 3] |
| 两尺度 CoS-large (24L) | ~3.20 | ~86.0 | [Fig 3] |
| 两尺度 CoS (12L) | ~3.50 | ~84.8 | [Fig 3] |
| 单尺度 | ~4.35 | ~84.5 | [Fig 3] |

(数值从 Fig 3 柱状图读取,为近似值)

**关键实验发现**:

1. **SoS 全面优于 CoS** [§IV-A, §IV-C]: SoS NMOS 4.42 vs CoS 4.12 [Table I]; 三尺度 SoS CER ~2.60 vs CoS ~3.50 [Fig 3] [论文原文]。CoS-large (24L, 参数翻倍) 仍不及 SoS (3x12L),说明 SoS 优势来自架构而非参数量 [论文原文]。

2. **SWND 对多尺度表示至关重要** [§IV-B]: 无 SWND 时,仅使用高尺度序列 (b=2) 重建的 MCD 和 CER 显著恶化,表明高尺度序列几乎不保留信息 [Fig 2] [论文原文]。[agent 解读] 这直接验证了"representation collapse"问题的存在 --- 模型倾向于将所有信息编码到低尺度 (高分辨率) 序列,使高尺度序列退化为空壳。

3. **多尺度生成显著改善 TTS** [§IV-C]: 随尺度数增加,CER 和 SIM 均持续改善 [Fig 3] [论文原文]。从单尺度到三尺度 SoS, CER 从 ~4.35 降至 ~2.60,SIM 从 ~84.5 升至 ~87.4 [Fig 3]。

4. **SoS 超越 CosyVoice (100k hours)** [§IV-A]: CoFi-SoS 仅用 7k hours 数据,NMOS 4.42 vs CosyVoice 4.28, SMOS 3.90 vs 3.70 [Table I] [论文原文]。[agent 解读] 这个结果需要谨慎解读 --- CosyVoice 是基于 supervised semantic token + CFM 的不同架构,且测试集为作者自建的中文测试集,不是标准公开 benchmark。但在同等 codec LM 范式内的对比 (vs VALL-E, SoCodec-TTS) 是 fair 的。

5. **Attention 可视化直接证明 recency bias 缓解** [§IV-D, Fig 4]: 单尺度 LM 的 attention map 呈现强烈的近邻偏好 (每个 speech frame 只关注最近的 token); CoS 高尺度序列生成时 attention 分布更宽广,覆盖更多 text 和 speech 上下文; SoS 的第一个 LM 展现最宽的 attention 范围 [Fig 4] [论文原文]。

## 局限性

1. **语言和数据局限**: 仅在中文 (WenetSpeech4TTS 7k hours) 上验证,跨语言泛化性未知 [agent 解读]。
2. **参数和计算成本**: SoS 需要 3 个独立 12L Transformer (36L 总参数),训练成本约为 CoS 的 3 倍。论文未报告推理延迟或 RTF,无法评估实际部署可行性 [agent 解读]。
3. **Mel 域操作**: 编解码在 Mel spectrogram 域,依赖预训练 BigVGAN vocoder 重建波形,非端到端。Vocoder 质量成为音质上限 [§II-A]。
4. **评估规模偏小**: MOS 测试仅 10 名评估者、100 条子集; 自建测试集 860 条,非公开标准 benchmark [§III-B]。
5. **与 SoCodec 缺乏直接对比**: 虽然 Table I 包含 SoCodec-TTS (NMOS 4.30),但未做 SoCodec 的消融控制实验 (如: 使用 SoCodec 的 OPQ 多流 + CoFi-LM 的 SoS 生成)。两种多尺度方法的优劣和互补性未被探索 [agent 解读]。
6. **VQ 配置非对称**: 三尺度中,120ms 和 40ms 是单流,20ms 是四流 (使用 delay pattern),这种混合配置使消融分析不够纯粹 [§III-A]。

## 点评

CoFi-Speech 的核心洞察与同期 SoCodec 一脉相承: 语音的信息结构是多层级的,CLM 的"recency bias"本质上是 LM 在长 token 序列上对粗粒度结构信息的忽视。SoCodec 通过有序多流 (multi-stream ordered by importance) 解决,CoFi-Speech 通过多尺度时间分辨率 (multi-scale by temporal resolution) 解决 --- 两者是同一问题的两个正交维度,组合空间尚未被探索。

Stack-of-Scale (SoS) 是本文最有价值的贡献。它的优势不仅仅是"多个小 LM 比一个大 LM 好"(CoS-large 24L 的实验已排除了这种解释),而是级联架构本身带来的归纳偏置: 每个 LM 专注于一个时间尺度,高尺度 LM 的 hidden states 通过上采样 conditioning 直接指导低尺度 LM,建立了显式的跨尺度依赖。这与 hierarchical generation 的直觉高度一致 --- 先确定全局结构 (what to say),再填充局部细节 (how to say it)。

Fig 4 的 attention 可视化是一个难得的直接证据。在 CLM-TTS 领域,多数工作仅通过 MOS/CER 等终端指标间接推断模型行为,而 CoFi-Speech 直接展示了多尺度建模如何改变 LM 的注意力分布 --- 高尺度序列的 attention 更宽广,说明 LM 确实在"关注全局信息"。

对比同组的 SoCodec: SoCodec 更侧重 codec 设计创新 (OPQ),CoFi-Speech 更侧重 LM 生成策略创新 (SoS)。从 Table I 看,CoFi-SoS (NMOS 4.42) 优于 SoCodec-TTS (NMOS 4.30),但这一差异是否完全来自 SoS vs 单 LM delayed pattern,无法从现有实验中分离。

值得注意的是,CoFi-Codec 操作在 Mel 域而非波形域,这使得重建质量受 BigVGAN vocoder 限制。在 2024 年末到 2025 年主流 codec (如 SoundStream, EnCodec, DAC) 都已在波形域端到端运作的背景下,Mel 域 codec 显得相对保守。但论文团队在 SoCodec 中也采用了相同策略 (HuBERT features + Mel),说明他们有意将 codec 与 vocoder 解耦,以便独立优化。

## 可复用的 idea

1. **Stack-of-Scale 级联生成**: 用多个轻量 LM 分别负责不同时间尺度的 token 生成,高尺度 LM 的 hidden states 作为 prompt + 上采样 conditioning 传给低尺度 LM。核心价值: 每个 LM 只需处理短序列 (降低建模复杂度),同时通过 conditioning 保持尺度间一致性。可迁移到任何 hierarchical token 生成场景。
2. **Scale-Wise Nested Dropout**: 训练多尺度/多层 codec 时,随机 mask 低层/低尺度表示,强制高层/高尺度保留有效信息。实现极其简单 (仅需一个随机采样),但有效防止表示坍缩。与 quantizer dropout 互补。
3. **Residual 减法 + 多分辨率量化**: 在 feature map 层面做 encoder-decoder 差值取残差,再在不同时间分辨率下量化残差。这比标准 RVQ (同一分辨率逐层残差) 更适合编码不同粒度的信息。
4. **Attention 可视化作为 CLM 诊断工具**: 聚合所有 heads 和 layers 的 attention map,用 scaling factor + max clipping 清晰可视化 CLM 的注意力分布模式,直接诊断 recency bias 问题。

## 审阅

> [!review] 审阅 pass-with-fixes (auto, 2026-06-08)
> **结论**: pass-with-fixes | 0 high, 1 medium (fixed), 2 low
> 
> **medium** (fixed): frontmatter datasets 字段已补充 WenetSpeech4TTS
> **low**: models 缺少 VALL-E/SoCodec-TTS baseline; 训练策略中 SoS 参数量对比可改为 [论文原文] 标注
> 
> 详见 `_review/CoFi-Speech-review.yml`

---

检索命中: [[ResidualVectorQuantization]], [[CodebookCollapse]], [[LLM-basedTTS]], [[SpeechTokenizer]] | 过滤: [[CodecLanguageModel]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review) | 未命中但可能相关: 无
