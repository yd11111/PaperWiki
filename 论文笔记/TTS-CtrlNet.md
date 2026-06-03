---
type: paper
tier: deep
title: "TTS-CtrlNet: Time varying emotion aligned text-to-speech generation with ControlNet"
arxiv_id: "2507.04349"
source: "Sources/TTS-CtrlNet.pdf"
authors: [Jaeseok Jeong, Yuna Lee, Mingi Kwon, Youngjung Uh]
year: 2025
venue: "arXiv"
tags: [TTS, emotion-control, zero-shot, flow-matching, ControlNet, DiT, arousal-valence, time-varying-control, F5-TTS]
concepts: ["[[Conditional Flow Matching]]", "[[Emotion Control in TTS]]", "[[Mel Spectrogram]]", "[[Self-Supervised Speech Representation]]", "[[Speaker Embedding]]", "[[Diffusion-based TTS]]"]
models: ["[[模型库/wav2vec 2.0|wav2vec 2.0]]", "[[模型库/Whisper|Whisper]]"]
tasks: ["[[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: TTS-CtrlNet 位于 flow-matching zero-shot TTS 的情感可控扩展分支。在 [[Conditional Flow Matching]] 谱系中,它以 F5-TTS (Chen et al., 2024) 为 backbone,而非 Voicebox 系列。与 [[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]] (Wu et al., 2024) 解决相同问题 (时变情感控制),但路线截然不同: EmoCtrl-TTS 采用全模型微调 (87k 小时数据,修改所有参数); TTS-CtrlNet 借鉴图像领域 ControlNet (Zhang et al., 2023) 的思路,冻结原始模型,仅训练一个可学习副本,用约 400 小时公开数据即可达到甚至超越 EmoCtrl-TTS 的情感控制性能。
>
> **已有认知**: 知识库中 [[Emotion Control in TTS]] [待确认] 已记录了 EmoCtrl-TTS 的帧级 arousal-valence 条件方案及其数据策略。[[Zero-shot Speech Synthesis]] 任务页记录了 F5-TTS 作为非自回归 flow matching 方法的定位。[[Diffusion-based TTS]] [待确认] 记录了 flow matching 取代 diffusion 成为主流的演进趋势。ControlNet 概念在知识库中尚无独立页面。
>
> **创新判断**: 相比知识库已记录的方法,本文核心新贡献有三: (1) 首次将 ControlNet 范式从图像扩散模型迁移至 flow-matching TTS; (2) 发现 emotion-specific flow step 区间 -- 情感信息仅在 flow step [0, temo] 区间内决定,训练和推理都只需在此区间应用 ControlNet; (3) 通过 DiT block-level 消融找到关键 block 并将其排除在 ControlNet 连接之外,在几乎不损害 WER 的前提下实现情感控制。以上三点均未在已有概念页中覆盖。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[Diffusion-based TTS]](pending-review), [[Mel Spectrogram]](pending-review), [[Style Transfer in TTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将图像 ControlNet 范式首次迁移至 flow-matching TTS (F5-TTS),冻结原始模型仅训练可学习副本,用 ~400 小时公开情感数据实现时变情感控制,同时保留 zero-shot voice cloning 能力
> - **路线**: 参考音频 → SER (wav2vec 2.0-based, 窗口滑动插值) → arousal-valence embedding → [noisy speech + masked speech + transcript + emotion] → ControlNet (trainable DiT blocks subset) → zero-conv → 原始 F5-TTS DiT blocks (frozen) → mel spectrogram → vocoder → waveform
> - **指标**: JVNV S2ST: Emo-SIM 0.751 / Aro-Val SIM 0.742 (超越 EmoCtrl-TTS(+) 0.697/0.643); EMO-Change: Emo-SIM 0.724 / Aro-Val SIM 0.864 (超越 EmoCtrl-TTS(+) 0.679/0.811); WER: EMO-Change 0.6% (与 F5-TTS 2.9% 可比) [Table 5]
> - **可借鉴**: (1) ControlNet 冻结-副本范式可用于任何 flow-matching 生成模型的条件扩展; (2) emotion-specific flow step: 只在 flow step [0, 0.1] 区间注入情感条件,既省算力又提升质量; (3) block-level 消融定位关键 block 后将其排除在 ControlNet 连接外,保护文本保真度
> - **局限**: 仅用 F5-TTS 作为 backbone (未验证其他模型泛化); 无 NV (笑声/哭泣) 控制 (仅用 emotion encoder); WER 在 JVNV S2ST 上退化至 5.4% (F5-TTS 4.5%); 未开源代码/权重

## 核心问题

这篇论文要解决的问题: **现有情感 TTS 方法存在三个痛点 -- (1) 多数只支持 utterance-level 情感控制,无法实现单句内的时变情感转换; (2) 全模型微调方式代价高昂 (EmoCtrl-TTS 用 87k 小时数据 + 全参数微调),且会损害原始模型的语音质量和 zero-shot 能力; (3) 无法直观调节情感强度** [§1]。TTS-CtrlNet 的目标是: 用低成本、低数据量的方式为已有大规模预训练 TTS 模型"插件式"地添加细粒度情感控制能力。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TTS-CtrlNet 由两条分支构成 [§3.1, Fig 1]:

1. **原始 TTS 分支** (frozen): F5-TTS 的 22 个 DiT blocks,参数完全冻结,负责文本→语音的主生成路径
2. **ControlNet 分支** (trainable): 原始 DiT blocks 的可训练副本 (仅选择性连接部分 block),接收额外的情感条件输入

两条分支通过 **zero-convolution** 连接: zero-conv 初始化输出为零向量,确保训练初期 ControlNet 对原始模型行为无影响,随训练逐步学习注入情感信息 [§3.1.1]。

输入构成: 给定 mel spectrogram x 和对应文本 y,构造 noisy speech (1-t)x0 + tx1 和 masked speech (1-m)⊙x1,与 transcript 拼接后同时输入两条分支。ControlNet 分支额外接收情感嵌入 e (arousal + valence, 2 维 → 经 1x1 conv 映射到 F 维) [§3.1.1]。

ControlNet block k 的输出与原始 block k 的输出相加:
- 训练时: Z_new = F_k(·; θ) + m ⊙ F_k(·; ϕ) [Eq. 3]
- 推理时: Z_new = F_k(·; θ) + λ · m ⊙ F_k(·; ϕ) [Eq. 5],λ 为 control scale

### 关键设计选择

**为什么用 ControlNet 而不是全模型微调?** [论文原文] 全模型微调 (如 EmoCtrl-TTS) 存在三个问题: (1) 需要 87k 小时大规模数据和巨大训练成本; (2) 损害原始模型性能; (3) 无法直观调节情感强度 [§1]。[agent 解读] ControlNet 的核心优势在于"加法逻辑": 通过 zero-conv 从零开始渐进注入控制信号,原始模型的所有能力 (zero-shot voice cloning、自然度) 被物理性冻结保护,不存在灾难性遗忘风险。

**为什么只连接部分 DiT blocks (Selective Block)?** [论文原文] 作者对 F5-TTS 的 22 个 DiT blocks 逐一做 skip 消融 [§4.3.1, Fig 2]: 发现跳过某些 block 会导致 WER 剧增和 speaker similarity 骤降,这些 block 对文本保真度和说话人身份至关重要。将这些关键 block 排除在 ControlNet 连接之外,可在保持文本保真度的同时实现情感控制 [Table 3]。[agent 解读] 这一设计思路来源于图像领域 Stable Flow (Avrahami et al., 2024) 对 DiT layer 贡献的分析 [§2.2],但 TTS-CtrlNet 是首次在语音 DiT 上做此分析,且视角不同: Stable Flow 关注"哪些 layer 可以无损跳过以实现编辑",TTS-CtrlNet 关注"哪些 layer 不能被外部信号干扰以保护 WER"。

**Selective Block 的效果**: Full blocks 连接: WER 8.9%, SIM-o 0.630; Selective blocks: WER 0%, SIM-o 0.684, 同时 Emo-SIM 和 Aro-Val SIM 差异不大 [Table 3]。

**为什么用 emotion-specific flow step [0, temo]?** [论文原文] 作者通过实验发现,情感信息主要在 flow step 接近 0 (Gaussian noise) 的阶段被确定; 当 flow step 接近 1 (接近目标分布) 时,情感变化已锁定 [§4.3.2, Fig 3]。因此只在 [0, temo] (temo=0.1) 区间训练和推理时使用 ControlNet,其余步骤 λ=0。这带来两个好处: (1) 降低 WER — 全区间 [0,1] 训练 WER 为 0%, 但 Emo-SIM 仅 0.389; [0,0.1] 训练 WER 1.9% 且 Emo-SIM 0.565 [Table 1]; (2) 降低推理开销 — 仅在 10% 的 ODE 步骤中运行 ControlNet [Supplementary Table 8]。

[agent 解读] 这一发现与 diffusion/flow 模型的普遍直觉一致: 早期步骤确定全局结构 (包括情感、韵律),后期步骤修复局部细节 (phoneme-level 发音)。限制 ControlNet 只在早期步骤活跃,等效于让情感条件只影响"宏观情感轮廓"而不干扰"微观发音细节"。

**为什么用窗口滑动插值而非逐 token 情感?** [论文原文] wav2vec 2.0 输出的 token-level 特征直接逐 token 经 regression head 预测情感 (window size = 1) 会丢失情感上下文,WER 4.7%, Emo-SIM 0.500; 使用 window size = 30 的滑动插值后 WER 降至 1.9%, Emo-SIM 升至 0.565 [§4.3.3, Table 2]。[agent 解读] 情感是一个 supra-segmental 特征,需要跨多个 token 的上下文才能准确估计; window size = 1 退化为 frame-level 伪标签,训练时噪声过大。

**Control scale λ 的作用**: 推理时 λ 从 0 到 1 调节,实现原始模型行为和 ControlNet 增强行为之间的连续插值。λ=0.1 时 WER 0.63% / Aro-Val SIM 0.864 (最佳平衡); λ=1.0 时 WER 9.58% / Aro-Val SIM 0.892 (情感最强但发音退化) [Table 4]。

### 训练策略

1. **数据**: 合并 6 个公开情感语音数据集 (IEMOCAP, EARS, ESD, MSP-Podcast, Expresso 等),经清洗后约 400 小时,全部重采样至 24kHz [§4.1.1]
2. **SER**: wav2vec 2.0-based emotion recognition model (Wagner et al., 2023),预测 arousal-valence (去掉 dominance),chunk-wise 窗口滑动插值 [§3.2]
3. **训练配置**: LR 1e-5, 2x A5000, batch 8000 frames, 24k steps [§3.2]
4. **训练目标**: 标准 OT-CFM loss [Eq. 4],仅更新 ControlNet 参数 ϕ

## 实验

### 主要结果 (JVNV S2ST)

| 指标 | TTS-CtrlNet | EmoCtrl-TTS(+) | ELaTE | F5-TTS | SeamlessExpressive | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SIM-o ↑ | 0.464 | 0.497 | 0.441 | 0.459 | 0.268 | JVNV S2ST | [Table 5] |
| WER (%) ↓ | 5.4 | 3.2 | 3.8 | 4.5 | 1.2 | JVNV S2ST | [Table 5] |
| Emo-SIM ↑ | **0.751** | 0.697 | 0.671 | 0.684 | 0.653 | JVNV S2ST | [Table 5] |
| Aro-Val SIM ↑ | **0.742** | 0.643 | 0.548 | 0.627 | 0.494 | JVNV S2ST | [Table 5] |
| AutoPCP ↑ | 2.36 | 3.50 | 3.36 | 2.87 | 2.91 | JVNV S2ST | [Table 5] |

### EMO-Change (时变情感控制)

| 指标 | TTS-CtrlNet | EmoCtrl-TTS(+) | ELaTE | F5-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| SIM-o ↑ | 0.589 | 0.684 | 0.643 | 0.579 | EMO-Change | [Table 5] |
| WER (%) ↓ | 0.6 | 0.9 | 1.1 | 2.9 | EMO-Change | [Table 5] |
| Emo-SIM ↑ | **0.724** | 0.679 | 0.700 | 0.692 | EMO-Change | [Table 5] |
| Aro-Val SIM ↑ | **0.864** | 0.811 | 0.761 | 0.845 | EMO-Change | [Table 5] |
| AutoPCP ↑ | 3.68 | 3.44 | 3.52 | 3.62 | EMO-Change | [Table 5] |

### 主观评测

| 指标 | TTS-CtrlNet | F5-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SMOS ↑ | 2.7 / 3.6 | 2.0 / 3.7 | JVNV / EMO-Change | [Table 6] |
| NMOS ↑ | 3.2 / 3.8 | 2.3 / 3.8 | JVNV / EMO-Change | [Table 6] |
| EMOS ↑ | 3.4 / 3.3 | 2.6 / 2.0 | JVNV / EMO-Change | [Table 6] |

### 关键消融发现

1. **Selective blocks vs Full blocks**: Selective blocks WER 0% vs Full 8.9%, 同时 Emo-SIM 差异仅 0.565 vs 0.450 [Table 3]
2. **Flow step [0,0.1] vs [0,1]**: [0,0.1] WER 1.9% / Emo-SIM 0.565 vs [0,1] WER 0% / Emo-SIM 0.389 [Table 1] -- 限制区间反而提升情感相似度
3. **Emotion window size**: 30 vs 1: WER 1.9% vs 4.7%, Emo-SIM 0.565 vs 0.500 [Table 2]
4. **Control scale trade-off**: λ=0.1 是 WER-emotion 的最优平衡点 (WER 0.63%, Aro-Val SIM 0.864) [Table 4]

## 局限性

1. **WER 在 JVNV S2ST 上退化**: TTS-CtrlNet 5.4% vs F5-TTS 4.5% vs EmoCtrl-TTS 3.2% [Table 5]。但注意 F5-TTS 未训练日语,JVNV S2ST 的 WER 本身不完全反映英语生成能力;在纯英语的 EMO-Change 上 TTS-CtrlNet WER 0.6% 优于 F5-TTS 2.9% [Table 5]。
2. **无 NV 控制**: 仅使用 arousal-valence emotion encoder,不含 laughter/crying 等 NV 控制 (EmoCtrl-TTS 同时使用 NV embedding + emotion embedding) [§5 Limitations]。
3. **SER 模型限制**: 底层 SER 无法识别非语言线索 (笑声、哭泣),限制了情感表达的范围 [§5 Limitations]。
4. **仅验证 F5-TTS backbone**: 未在 Voicebox、CosyVoice 等其他 flow-matching TTS 上验证 ControlNet 范式的泛化性。
5. **未开源**: 代码和权重未公开。
6. **AutoPCP 偏低**: JVNV S2ST 上 AutoPCP 2.36 低于所有 baseline (EmoCtrl-TTS 3.50, F5-TTS 2.87) [Table 5],说明韵律对齐可能受损。

## 点评

TTS-CtrlNet 的最大价值在于"范式迁移 + 工程实用性": 将图像领域成熟的 ControlNet 范式首次成功迁移至 flow-matching TTS,证明了"冻结大模型 + 轻量可训练副本"的范式在语音场景同样有效。与 EmoCtrl-TTS 相比,TTS-CtrlNet 用 1/200 的数据量 (400h vs 87k h)、1/N 的训练成本 (2x A5000 vs 未公开但远大) 达到了更好的情感相似度指标,这是一个重要的工程启示: 当原始模型已经足够强大时,不需要修改它的参数,只需"旁挂"一个轻量控制模块。

emotion-specific flow step 的发现尤其有洞察力: 不是所有 ODE 步骤都同等重要,情感信息在早期步骤 (接近噪声端) 就已经被锁定。这个发现可能对所有 flow/diffusion 条件控制任务都有参考价值。

不足之处: (1) AutoPCP 指标偏低,说明ControlNet 可能在获得情感控制的同时牺牲了部分韵律自然度; (2) 缺少 NV 控制能力 (EmoCtrl-TTS 的重要特色); (3) block selection 的分析虽有价值,但结论依赖于 F5-TTS 的具体架构,换一个 backbone 需要重新做 block 分析。

与 [[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]] 的对比是本文最有价值的部分: 两者解决相同问题但路线相反 (全参数微调 + 大数据 vs 冻结 + 旁挂 + 小数据),TTS-CtrlNet 在情感指标上全面超越,但在 WER 和 AutoPCP 上有所退化,体现了"控制力 vs 保真度"的 trade-off。Control scale λ 提供了一个优雅的推理时调节机制来平衡这个 trade-off。

## 可复用的 idea

1. **ControlNet 范式迁移至 flow-matching 生成模型**: 冻结原始模型 + trainable copy + zero-conv,适用于任何需要为已有预训练 flow/diffusion 模型添加新条件的场景 (不限于情感,可推广到说话风格、口音、背景音等)。
2. **Emotion-specific flow step**: 通过扰动实验找到条件信号的"敏感区间",只在该区间注入条件信号 -- 既降低计算开销又避免后期步骤干扰发音精度。可推广到任何 flow/diffusion 控制任务。
3. **Block-level skip 消融定位关键 block**: 对 DiT/Transformer 逐 block 做 skip 测试,找出对 WER/speaker similarity 至关重要的 block 并保护它们。这是一种通用的"定位哪些层不能动"的方法论。
4. **Control scale 推理时调节**: 训练时固定 λ=1.0,推理时通过 λ 实现情感强度的连续调节,提供"一个模型多种强度"的灵活性。
5. **窗口滑动插值提取时变情感**: SER 模型的 token-level 输出通过 window sliding interpolation 而非逐 token 使用,保持情感上下文信息的同时获得时变特性。
