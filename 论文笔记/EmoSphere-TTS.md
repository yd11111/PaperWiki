---
type: paper
tier: deep
title: "EmoSphere-TTS: Emotional Style and Intensity Modeling via Spherical Emotion Vector for Controllable Emotional Text-to-Speech"
arxiv_id: "2406.07803"
source: "Sources/EmoSphere-TTS.pdf"
authors: [Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Sang-Hoon Lee, Seong-Whan Lee]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, emotion, controllable, adversarial-training, expressiveness, prosody, spherical-coordinates]
concepts: ["[[Emotion Control in TTS]]", "[[Prosody Modeling]]", "[[F0 Modeling]]", "[[Global Style Tokens]]", "[[Non-autoregressive TTS]]", "[[Mel Spectrogram]]"]
models: ["[[论文笔记/EmoSphere-TTS|EmoSphere-TTS]]"]
tasks: []
datasets: ["ESD"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[Prosody Modeling]]✓, [[Emotion Control in TTS]][待确认], [[F0 Modeling]][待确认], [[Global Style Tokens]][待确认], [[Non-autoregressive TTS]][待确认], [[Mel Spectrogram]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Prosody Modeling]], [[Emotion Control in TTS]], [[F0 Modeling]], [[Global Style Tokens]], [[Non-autoregressive TTS]], [[Mel Spectrogram]] | 过滤: 5 页 pending-review | 未命中但可能相关: [[Style Transfer in TTS]], [[Speech Factorization]]

**谱系定位**: EmoSphere-TTS 处于情感可控 TTS 演进线的中段 — 在 Emotion Embedding (2021) 和多尺度层级建模 (MsEmoTTS, 2022) 之后,在零样本情感控制 (EmoSphere++, 2024) 和 LLM 自由文本情感 (EmoVoice, 2025) 之前。概念页 [[Emotion Control in TTS]][待确认] 已记录其扩展版 EmoSphere++ 作为"球面空间建模情感分布"的代表。本文是 EmoSphere++ 的前身/会议版本。

**已有认知**: 情感控制的核心挑战在于情感与 timbre/prosody 的深度纠缠 ([[Prosody Modeling]]✓)。现有方法要么用离散 emotion label (如 FastSpeech 2 w/ emotion ID),要么用 reference encoder (如 [[Global Style Tokens]][待确认]),但前者丢失细粒度,后者受 reference mismatch 限制。AVD (arousal, valence, dominance) 维度提供连续描述,但此前缺乏直觉可控的参数化方案。

**创新判断**: 本文的核心创新是将 AVD 从笛卡尔坐标变换到球面坐标,使"情感强度"(径向距离 r)和"情感风格"(角度 θ, φ)自然解耦。这一几何变换赋予了物理可解释的控制维度,区别于 relative attribute (学习排序函数) 和 scaling factor (直接缩放 embedding) 等参数化方式。

> [!summary] 速查
> - **一句话**: 通过将 AVD 情感伪标签从笛卡尔坐标变换到球面坐标,将情感风格(角度)和情感强度(径向距离)自然解耦,实现可控情感 TTS
> - **路线**: 文本 → FastSpeech 2 Encoder → Variance Adaptor (+ Spherical Emotion Embedding + Speaker Embedding) → Decoder → Mel → BigVGAN → Waveform
> - **指标**: nMOS 3.88 vs 最佳 baseline 3.34 [Table 1]; ECA 94.02% [Table 1]; 情感强度辨别率平均 0.69 vs baseline 0.53/0.51 [Table 2]; 均在 ESD 数据集上评估
> - **可借鉴**: (1) 球面坐标变换将风格/强度解耦的思路可迁移到其他多维属性控制场景; (2) 用 SER 伪标签替代人工 AVD 标注,零标注成本获得连续情感维度
> - **局限**: 仅句子级全局情感,无 phoneme-level 控制; 仅在 ESD (5 情感, 10 说话人) 上验证; 无零样本能力; 合成质量仍低于 GT (UTMOS 3.15 vs 3.78)

## 核心问题

现有情感 TTS 方法面临两个根本矛盾:

1. **离散标签 vs 情感连续性**: emotion label 将丰富的情感表达压缩为几个离散类别 (如 sad 涵盖了 lonely、hurt 等不同细微情感),无法反映情感的连续变化和强度差异 [§1]
2. **强度控制 vs 质量稳定**: 现有的 scaling factor 方法通过直接缩放 emotion embedding 控制强度,但调整 scaling factor 往往导致音频质量不稳定 [§1]

**核心提问**: 能否找到一种情感表示空间,使"情感是什么"(风格)和"情感有多强"(强度)天然分离,从而实现直觉化的独立控制?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

EmoSphere-TTS 在 FastSpeech 2 架构上增加三个关键模块 [Fig 1]:

```
Text → Phoneme Encoder (4-layer FFT) → Variance Adaptor →
     → [+ Spherical Emotion Embedding (h_emo) + Speaker Embedding (h_spk)] →
     → Mel Decoder (4-layer FFT) → Mel Spectrogram → BigVGAN → Waveform
```

三个新增模块:
1. **AVD Encoder**: 从音频提取 arousal/valence/dominance 伪标签
2. **Cartesian-Spherical Transformation**: 将 AVD 笛卡尔坐标变换为球面坐标 (r, θ, φ)
3. **Dual Conditional Discriminator**: 以 emotion 和 speaker embedding 为条件的对抗训练

### 关键设计选择

#### 1. 为什么用 SER 伪标签而非人工 AVD 标注?

[论文原文] 人工 AVD 标注存在主观性高、采集成本大的问题,且只有少数情感语音数据集提供这类标注 [§1]。本文采用 wav2vec 2.0-based SER 模型 (Wagner et al., 2023) 提取 AVD 伪标签,输出 (d_a, d_v, d_d) 各维度范围约 0-1 [§2.1.1]。

[agent 解读] 这一选择使方法不依赖特定数据集的 AVD 标注,理论上可扩展到任意语音数据。但伪标签的准确性受 SER 模型能力限制,尤其对 dominance 维度的预测可靠性学界仍有争议。

#### 2. 为什么从笛卡尔坐标变换到球面坐标?

[论文原文] 在笛卡尔 AVD 空间中,情感风格和强度是耦合的 — 修改任一维度同时改变了风格和强度。受坐标变换文献 (Jenke & Peer, 2018) 启发,作者提出两个假设 [§2.1.2]:
- 假设 i: 情感强度随远离中性情感中心的距离增大而增强
- 假设 ii: 从中性中心出发的角度决定情感风格

变换过程:
1. 以中性情感样本的均值 M 为原点平移: e'_ki = e_ki - M [Eq.1]
2. 笛卡尔→球面变换: r = sqrt(d'_a² + d'_v² + d'_d²), θ = arccos(d'_d/r), φ = arctan(d'_v/d'_a) [Eq.2-3]
3. r 做 min-max 归一化 (用 IQR 鲁棒确定范围), 角度 (θ, φ) 按 8 个象限量化为 8 种情感风格

[agent 解读] 球面坐标的几何意义是关键: r 控制"距离中性有多远"(强度), 角度 (θ, φ) 控制"往哪个方向偏"(风格)。这比 scaling factor 方法更优雅 — scaling factor 只能沿 embedding 方向单调缩放,而球面坐标允许在任意方向独立调节风格和强度。8 象限量化是因为 AVD 三个轴各有正负方向,2³=8 个组合恰好覆盖了基本情感类型。

#### 3. 球面情感编码器 (Spherical Emotion Encoder) 的融合策略

[论文原文] 将球面空间的 style vector 和 intensity vector 与 emotion ID embedding 组合 [§2.2, Eq.4]:

```
h_emo = LN(softplus(concat(h_sty, h_cls))) + h_int
```

其中 h_sty (风格投影)、h_cls (情感类别 embedding)、h_int (强度投影) 分别通过投影层对齐维度。使用 softplus 激活 (类似 Yoon et al., 2022) 后接 LayerNorm,最终加上 intensity 向量。

[agent 解读] 加法融合 h_int 而非 concat,意味着强度信息作为残差修正,不改变风格/类别编码的主体结构。这一设计与 scaling factor 方法本质不同 — 后者是乘法缩放,前者是加法偏移。加法避免了乘法在小 factor 时信号过弱的问题。

#### 4. 为什么引入双条件判别器?

[论文原文] 使用多个 CNN-based 判别器,输入为随机窗口的 Mel clip [§2.3]。关键创新是引入双条件: 一个 Conv2D stack 仅接收 Mel clip,其余接收 Mel clip + 条件 embedding (speaker 或 emotion)。条件 embedding 拉伸到 Mel clip 长度后 concat [Eq.5-6]。

[agent 解读] 双条件设计的动机是: 普通判别器只学"自然不自然",加入 emotion 条件后学"情感一致不一致",加入 speaker 条件后学"说话人一致不一致"。这迫使生成器同时满足三方面约束 — 自然度、情感正确性、说话人保真度。这与 GANSpeech (Yang et al., 2021) 和 DiffProsody (Oh et al., 2024) 的条件判别思路一致。

### 训练策略

- 使用 ESD 数据集: 10 英语说话人, 5 情感 (neutral, happy, angry, sad, surprise), 共 17,500 样本 [§3.1]
- 声学模型: FastSpeech 2 配置 — 4 层 FFT, hidden 256, filter 1024, kernel 9 [§3.2]
- 优化器: AdamW (β₁=0.9, β₂=0.98), TTS lr=5×10⁻⁴, 判别器 lr=1×10⁻⁴ [§3.1]
- 训练时长: 单卡 NVIDIA RTX A6000 约 24 小时 [§3.1]
- Vocoder: BigVGAN 预训练模型 [§3.1]
- 判别器窗口: 3 种不同长度 [32, 64, 96] [§3.2]

## 实验

| 指标 | EmoSphere-TTS | 最佳 Baseline | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| UTMOS (↑) | 3.15 | 2.66 (FS2+SF) | 3.78 | ESD | [Table 1] |
| nMOS (↑) | 3.88±0.05 | 3.34±0.05 (FS2+SF) | 4.27±0.04 | ESD | [Table 1] |
| sMOS (↑) | 3.48±0.11 | 3.29±0.13 (FS2+SF) | 4.37±0.11 | ESD | [Table 1] |
| WER (↓) | 17.43 | 19.46 (FS2+SF) | 11.92 | ESD | [Table 1] |
| CER (↓) | 7.05 | 8.73 (FS2+SF) | 3.04 | ESD | [Table 1] |
| ECA (↑) | 94.02% | 93.75% (FS2+EL) | 85.67% | ESD | [Table 1] |
| SECS (↑) | 0.669 | 0.654 (FS2+EL/RA) | 0.753 | ESD | [Table 1] |
| EER (↓) | 4.29 | 5.89 (FS2+EL) | 2.99 | ESD | [Table 1] |

**Baselines**: FS2+EL = FastSpeech 2 w/ Emotion Label; FS2+RA = FastSpeech 2 w/ Relative Attribute; FS2+SF = FastSpeech 2 w/ Scaling Factor

**消融实验** [Table 1]:
- w/o Spherical Emotion Vector (用 emotion ID lookup table): nMOS 3.69, sMOS 3.45, ECA 93.89% — 球面向量贡献最大
- w/o Dual Conditional Discriminator: nMOS 3.39, sMOS 3.24, ECA 92.60% — 判别器对质量改善显著

**情感强度控制** [Table 2]:
- 平均强度辨别率: Weak<Medium 0.71, Medium<Strong 0.65, Weak<Strong 0.72
- 对比: Relative Attribute 0.52/0.51/0.60; Scaling Factor 0.56/0.47/0.50
- Happy 情感上优势最大: EmoSphere 0.80/0.66/0.84 vs 其他 <0.61

**风格偏移实验** [§3.5, Fig 3]: 改变球面坐标角度可观察到 pitch 轮廓的有意义变化 — A 轴正向→pitch 上升趋势, V 轴正向→平均 pitch 升高, D 轴正向→变化范围收窄。这验证了 AVD 三维度在球面空间中各自保留了物理意义。

## 局限性

1. **仅句子级情感**: 只建模全局 (sentence-level) 情感风格,无法实现 phoneme-level 细粒度控制。作者在结论中明确指出这是未来方向 [§4]
2. **数据集局限**: 仅在 ESD (5 情感, 10 说话人) 上验证,情感类别和说话人多样性有限。未在更大规模、更多情感类型的数据集上测试
3. **无零样本能力**: 需要在训练集中见过目标说话人和情感类型,不支持 unseen speaker/emotion 的泛化
4. **与 GT 差距**: 合成质量仍显著低于 GT (UTMOS 3.15 vs 3.78, nMOS 差 0.39),说明 FastSpeech 2 架构本身对表现力语音的建模能力有上限
5. **伪标签质量未验证**: AVD 伪标签来自预训练 SER 模型,未验证其准确性,尤其 dominance 维度的可靠性存疑
6. **ECA 异常高**: ECA 94.02% 甚至高于 GT 85.67%,说明模型可能过度拟合到"标准"情感模式,损失了真实人类语音中的情感模糊性

## 点评

**优点**:
- 球面坐标变换是一个优雅且有物理直觉的设计,将风格/强度控制从参数搜索问题变为几何操作。这种思路在情感 TTS 领域是新颖的。
- 用 SER 伪标签替代人工标注是务实的工程选择,大幅降低了方法的数据需求。
- 消融实验设计清晰: 球面向量 vs emotion ID 的对比直接说明了核心贡献。

**不足**:
- 8 象限量化可能过度简化: 4 种基本情感 + neutral 未必恰好对应 8 个象限方向,论文未分析哪些象限对应哪些情感。
- 与同期方法的对比不够: baseline 均为 FastSpeech 2 变体,缺少与 diffusion-based 或 LLM-based 情感 TTS 的对比。
- ESD 是一个表演性数据集 (acted emotion),结果能否迁移到自发情感语音 (spontaneous emotion) 未知。
- 论文长度限制 (4 页 Interspeech) 导致某些关键细节缺失: 如 IQR 归一化的具体阈值、emotion ID 的使用方式、训练 epoch 数等。

## 可复用的 idea

1. **笛卡尔→球面坐标解耦**: 任何多维连续属性都可尝试用坐标变换将"方向"(what)和"幅度"(how much)解耦。例如 speaker embedding 的风格维度可尝试类似变换。
2. **SER 伪标签作为无标注情感控制**: wav2vec 2.0-based SER 可为任意语音数据提供 AVD 软标签,避免人工标注。可迁移到其他需要情感条件的任务 (如情感 VC、情感对话生成)。
3. **双条件判别器**: 在 GAN 训练中加入多个条件 (speaker, emotion, ...) 的判别器分支,迫使生成器同时满足多方面约束,可推广到任何需要多属性保真的生成任务。

---

检索命中: [[Prosody Modeling]]✓, [[Emotion Control in TTS]](pending-review), [[F0 Modeling]](pending-review), [[Global Style Tokens]](pending-review), [[Non-autoregressive TTS]](pending-review), [[Mel Spectrogram]](pending-review) | 过滤: 无 | 未命中但可能相关: [[Style Transfer in TTS]], [[Speech Factorization]]
