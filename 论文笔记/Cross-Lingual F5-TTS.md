---
type: paper
tier: deep
title: "Cross-Lingual F5-TTS: Towards Language-Agnostic Voice Cloning and Speech Synthesis"
arxiv_id: "2509.14579"
source: "Sources/Cross-Lingual-F5-TTS.pdf"
authors: [Qingyu Liu, Yushen Chen, Zhikang Niu, Chunhui Wang, Yunting Yang, Bowen Zhang, Jian Zhao, Pengcheng Zhu, Kai Yu, Xie Chen]
year: 2025
venue: "arXiv"
tags: [TTS, flow-matching, cross-lingual, voice-cloning, zero-shot, NAR-TTS, duration-prediction, forced-alignment]
concepts: ["[[Conditional Flow Matching]]", "[[Duration Predictor]]", "[[Non-autoregressive TTS]]"]
models: []
tasks: ["[[Cross-lingual Voice Cloning]]", "[[Zero-shot Speech Synthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 7
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文位于 "NAR flow-matching TTS" 与 "cross-lingual voice cloning" 的交叉点。在 flow-matching TTS 路线上,F5-TTS (Chen et al., 2024) 延续 E2 TTS 的极简设计(字符输入 + filler token + speech infilling),消除了 phoneme alignment 和 G2P 的需求。但 F5-TTS 的 duration 估计依赖 audio prompt transcript 的 length-ratio 方法,这在跨语言场景下失效。Cross-lingual voice cloning 路线中,VALL-E X (2023) 用 AR codec LM 实现跨语言但仍需 prompt transcript; CosyVoice 系列通过多语言 LLM + shared tokenizer 天然支持跨语言; 本文是 **首次在 NAR flow-matching TTS 上实现不依赖 prompt transcript 的跨语言零样本克隆**。
>
> **已有认知**:
> - [[Conditional Flow Matching]] (confirmed): F5-TTS 基于 OT-CFM,学习向量场从 Gaussian noise 到数据分布的直线轨迹。已在 CosyVoice 系列、Seed-TTS、MaskGCT、VoiceFlow 等系统中广泛应用,是 NAR TTS 的主流生成框架。
> - [[Cross-lingual Voice Cloning]] (confirmed): 给定语言 L1 的参考语音 + 语言 L2 的目标文本,合成保持说话人音色但使用 L2 的语音。当前 SOTA 为 CosyVoice 3 (WER to-en 2.98%) 和 Qwen3-TTS (WER zh-to-en 2.77%),均为 LLM-based 架构。NAR flow-matching 系统此前缺乏跨语言能力。
> - [[Zero-shot Speech Synthesis]] (confirmed): 本文的应用场景。F5-TTS 在 SEED-TTS-Eval 上 CER 1.56% (test-zh) / WER 1.83% (test-en),本文目标是保持这一水平的同时扩展跨语言能力。
> - [[SEED-TTS-Eval]] (confirmed): 本文使用 SEED-TTS-Eval test-en 和 test-zh 子集评估 intra-lingual 性能,使用 Whisper-large-V3 和 Paraformer-zh 计算 WER。
> - [[Duration Predictor]] [待确认]: 本文的核心创新与 duration prediction 密切相关。KB 中记录了 duration prediction 从 HMM forced alignment → FastSpeech duration predictor → MAS → E2E differentiable duration → RL-optimized duration 的演进。F5-TTS/E2 TTS 通过 filler token + speech infilling 完全绕过了显式 duration predictor,但仍需 length-ratio 方法推断目标长度。本文引入独立的 speaking rate predictor 来替代 length-ratio,是 duration 建模的又一种新路径。
> - [[Non-autoregressive TTS]] [待确认]: 本文基于 F5-TTS 这一 NAR flow-matching 系统。NAR 与 AR 的关键区别在于并行生成,但 NAR 需要显式或隐式的 duration 信息来确定输出长度。
> - [[Emilia]] [待确认]: 本文使用 Emilia 约 95K 小时中英数据训练,另取 1000 小时平衡子集训练 speaking rate predictor。
>
> **创新判断**: 对比 KB 中已有方法,本文的核心创新在于解决了 NAR flow-matching TTS(特别是 F5-TTS 系列)的跨语言瓶颈。E2 TTS/F5-TTS 通过 speech infilling 消除了显式 phoneme alignment,但推理时仍需 prompt transcript 来估算目标时长(length-ratio)。本文通过两个改造打破这一依赖: (1) 训练时用 MMS forced alignment 在 word boundary 处切分,使 prompt 部分不需要 transcript; (2) 推理时用 speaking rate predictor 从 prompt 声学特征直接估算目标时长。虽然引入了额外的 speaking rate predictor 模块,但 intra-lingual 性能保持持平或略优(WER/UTMOS 优于 baseline, SIM 仅低 0.005-0.014),同时在跨语言场景提供了原 F5-TTS 无法实现的能力。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Cross-lingual Voice Cloning]]✓, [[Zero-shot Speech Synthesis]]✓, [[SEED-TTS-Eval]]✓ | 过滤: [[Duration Predictor]](pending-review), [[Non-autoregressive TTS]](pending-review), [[Emilia]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过 MMS 强制对齐 + speaking rate predictor 移除 F5-TTS 对 audio prompt transcript 的依赖,首次在 NAR flow-matching TTS 上实现跨语言零样本语音克隆
> - **路线**: Audio Prompt → Speaking Rate Predictor (Transformer-based, mel→rate class) → predicted rate + target text phoneme/syllable/word count → target duration; Target Text + Audio Prompt (无 transcript) → Cross-Lingual F5-TTS (DiT, OT-CFM, MMS word boundary 训练) → Mel Spectrogram → Vocos vocoder → Waveform
> - **指标**: Intra-lingual: LibriSpeech-PC WER 2.079% (M1) vs F5-TTS 2.205%, UTMOS 3.884 vs 3.797 [Table 2]; SeedTTS-zh WER 1.481% (M2) vs 1.475%, SIM 0.764 vs 0.762 [Table 2]; Cross-lingual test-en: WER 2.496% (M1), SIM 0.543 [Table 3]; Cross-lingual test-zh: WER 1.801% (M2), SIM 0.565 [Table 3]
> - **可借鉴**: (1) Speaking rate 离散化为分类任务 + Gaussian Cross-Entropy loss 处理有序类别的思路,可迁移到任何需要预测连续值的场景; (2) MMS forced alignment 作为多语言 word boundary 工具,可用于其他需要 transcript-free 操作的场景; (3) 不同语言适配不同粒度的 speaking rate predictor(英语用 phoneme-level, 中文用 syllable-level)的发现
> - **局限**: (1) 跨语言 SIM 较低(0.543-0.565),说明去掉 prompt transcript 后细微说话人特征(口音/情感)的迁移能力下降 [§5]; (2) speaking rate predictor 仅在中英数据训练,泛化到其他语言的可靠性未充分验证; (3) 相比 LLM-based 跨语言系统(CosyVoice 3/Qwen3-TTS)仍有差距; (4) 需要额外训练独立的 speaking rate predictor 模块

## 核心问题

F5-TTS 等 NAR flow-matching TTS 系统通过 speech infilling 训练范式实现了极简的零样本语音克隆,但在推理时依赖一个关键假设: **audio prompt 的 transcript 可用**。具体来说,F5-TTS 用 length-ratio 方法估算目标语音时长 --- 即 prompt 时长乘以目标文本长度与 prompt 文本长度的比值 [§1]。

这一设计在跨语言场景下产生两个致命问题 [§1]:
1. **实际可用性问题**: 当参考语音来自未知语言时,无法获取其 transcript
2. **语言间文本长度不可比问题**: 不同语言的文本长度比与语音时长比不对应(如中文一个字和英文一个词的语音时长差异大),length-ratio 机制在跨语言时完全失效

本文的核心目标: **移除 F5-TTS 对 audio prompt transcript 的依赖,同时保持 intra-lingual 性能不降,并赋予其跨语言克隆能力。**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由两个独立训练的组件构成 [§2]:

1. **Cross-Lingual F5-TTS 主模型**: 基于 F5-TTS 修改,在训练时使用 MMS forced alignment 提供的 word boundary 来切分 audio prompt 和 synthesis target,使 prompt 部分完全不需要 transcript
2. **Speaking Rate Predictor**: 独立的 Transformer 分类器,从 audio prompt 的 mel spectrogram 预测说话人的语速(phoneme/syllable/word per second),用于推理时确定目标语音时长

### 关键设计选择

#### 设计 1: MMS Forced Alignment 训练策略

**做了什么**: 使用 MMS (Massively Multilingual Speech) 的强制对齐工具对 Emilia 数据集预处理,提取每个 word 的 end time。训练时,随机选择一个 word boundary 作为切分点,boundary 左边作为 audio prompt(**丢弃其 transcript**),右边被 mask 作为合成目标 [§2.2, Fig 1]。

**为什么这样设计**: [论文原文] 原始 F5-TTS 的 speech infilling 训练中,audio prompt 和 target 都保留 transcript,模型在推理时也需要 prompt transcript 来做 duration estimation。通过在训练时就丢弃 prompt 的 transcript,模型被迫学会仅从声学特征中提取说话人信息,而不依赖文本信息 [§2.2]。

**为什么用 MMS 而非其他对齐工具**: [agent 解读] MMS 是目前覆盖语言最广的强制对齐工具(支持 1000+ 语言),使用 Wav2Vec2-based 声学模型 + CTC,这与跨语言目标天然匹配。WhisperX 虽然在 Emilia-Pipe 中用于转写,但其对齐精度在特殊 token (数字/符号/混合语言)上不稳定 [§3.2]。

**实现细节**: MMS 将音频切成 15 秒段分别处理后拼接回完整对齐矩阵; 遇到数字/特殊符号/其他语言 token 时跳过,不纳入 word boundary 提取 [§2.2, §3.2]。

#### 设计 2: Speaking Rate Predictor

**做了什么**: 训练三个独立的 speaking rate predictor,分别在 phoneme/syllable/word 三个粒度上预测说话速率 [§2.3]。

**为什么建模为分类而非回归**: [论文原文] 将 speaking rate 离散化为固定间隔的类别(间隔 0.25),phoneme-level 72 类 (0.25-18.0),syllable/word-level 32 类 (0.25-8.0),用分类任务训练 [§2.3]。[agent 解读] 分类任务的训练通常比回归更稳定,且可以输出概率分布而非点估计,对于 speaking rate 这种存在说话人变异的量尤其合适。

**Gaussian Cross-Entropy (GCE) Loss**: [论文原文] 标准交叉熵将所有类视为独立,但 speaking rate 的类别是有序的。GCE 使用 Gaussian kernel 构造 soft label: 距离 ground truth 类别越近的类获得越高的 soft target,标准差 sigma=1.0 控制平滑度 [§2.3, Eq. 4-5]。这使模型对微小预测偏差有容忍度,同时保持精度。

**推理流程**: audio prompt → speaking rate predictor → predicted rate (phonemes/syllables/words per second); target text → count linguistic units → duration = count / rate [§2.3]。这一机制完全不需要 prompt transcript,因此支持任意语言的 prompt。

#### 设计 3: 多粒度 Speaking Rate 的语言适配

**实验发现**: [论文原文] M1 (phoneme-level) 在英语上最优,M2 (syllable-level) 在中文上最优,M3 (word-level) 在所有数据集上最差 [§4.1, Table 1]。

**为什么不同语言适配不同粒度**: [agent 解读] 中文是典型的单音节语言,一个字 = 一个音节,syllable 与 semantic unit 高度对齐; 英文的 phoneme 分布更均匀、粒度更细,phoneme-level 预测信息更丰富。Word-level 粒度太粗,无法捕捉 word 内部的时长变异(如英语多音节词),导致 speaking rate 预测系统性偏快,cross-lingual test-en 的 WER 从 2.496% (M1) 暴涨至 16.494% (M3) [§4.3, Table 3]。

### 训练策略

**主模型训练**: 继承 F5-TTS-Base 架构(22 层 DiT, 16 attention heads, 1024-dim),在 Emilia 95K 小时中英数据上训练 1.2M 步,8×A100,per-GPU batch 38,400 audio frames。AdamW 优化器,学习率 warmup 到 7.5e-5 后线性衰减 [§3.3]。

**Speaking Rate Predictor 训练**: 6 层 Transformer, 8 attention heads, 512-dim。从 Emilia 取 500h 中文 + 500h 英文平衡子集训练 50K 步,4×A100。学习率 warmup 到 2.5e-4 后线性衰减。GCE loss sigma=1.0 [§3.3]。

**推理设置**: Euler ODE solver, NFE=32, CFG strength=2.0, sway sampling coefficient=-1.0, Vocos vocoder [§3.3]。

## 实验

### Duration Prediction 精度 (Table 1)

| 粒度 | 数据集 | MAE(s) | MRE(%) | 出处 |
| --- | --- | --- | --- | --- |
| M1 Phoneme-level | LibriSpeech-PC test-clean | 0.759 | 11.932 | [Table 1] |
| M2 Syllable-level | LibriSpeech-PC test-clean | 0.757 | 11.945 | [Table 1] |
| M3 Word-level | LibriSpeech-PC test-clean | 1.171 | 18.406 | [Table 1] |
| M1 Phoneme-level | SeedTTS test-en | 0.637 | 15.017 | [Table 1] |
| M2 Syllable-level | SeedTTS test-zh | 0.783 | 13.771 | [Table 1] |
| M3 Word-level | SeedTTS test-zh | 0.908 | 16.156 | [Table 1] |

### Intra-lingual Performance (Table 2)

| 系统 | Duration Method | WER(%)↓ | SIM-o↑ | UTMOS↑ | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| F5-TTS Baseline | Length-ratio | 2.205 | 0.668 | 3.797 | LibriSpeech-PC test-clean | [Table 2] |
| CL-F5 | M1 | 2.079 | 0.663 | 3.884 | LibriSpeech-PC test-clean | [Table 2] |
| CL-F5 | M2 | 2.120 | 0.658 | 3.892 | LibriSpeech-PC test-clean | [Table 2] |
| CL-F5 | M3 | 2.894 | 0.652 | 3.855 | LibriSpeech-PC test-clean | [Table 2] |
| F5-TTS Baseline | Length-ratio | 1.545 | 0.676 | 3.581 | SeedTTS test-en | [Table 2] |
| CL-F5 | M1 | 1.513 | 0.662 | 3.629 | SeedTTS test-en | [Table 2] |
| F5-TTS Baseline | Length-ratio | 1.475 | 0.762 | 2.898 | SeedTTS test-zh | [Table 2] |
| CL-F5 | M2 | 1.481 | 0.764 | 2.887 | SeedTTS test-zh | [Table 2] |

### Cross-lingual Performance (Table 3)

| Duration Method | WER(%)↓ | SIM-o↑ | UTMOS↑ | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| GT length | 2.462 | 0.530 | 3.083 | Cross-lingual test-en | [Table 3] |
| M1 | 2.496 | 0.543 | 3.069 | Cross-lingual test-en | [Table 3] |
| M2 | 4.362 | 0.518 | 3.059 | Cross-lingual test-en | [Table 3] |
| M3 | 16.494 | 0.486 | 2.926 | Cross-lingual test-en | [Table 3] |
| GT length | 1.596 | 0.558 | 2.452 | Cross-lingual test-zh | [Table 3] |
| M1 | 2.446 | 0.555 | 2.494 | Cross-lingual test-zh | [Table 3] |
| M2 | 1.801 | 0.565 | 2.503 | Cross-lingual test-zh | [Table 3] |
| M3 | 1.946 | 0.563 | 2.492 | Cross-lingual test-zh | [Table 3] |

**关键观察**:
- Intra-lingual 场景中,CL-F5 的 WER 和 UTMOS 在多数设定下优于或持平 F5-TTS baseline,SIM 略低(0.005-0.014),这是去掉 prompt transcript 的可接受代价 [Table 2]
- Cross-lingual 场景中,M1 (phoneme) 在英语目标上最优 (WER 2.496%),M2 (syllable) 在中文目标上最优 (WER 1.801%),与 intra-lingual 观察一致 [Table 3]
- M3 (word-level) 在跨语言英语目标上 WER 暴涨至 16.494%,因为粗粒度预测导致语速过快,时间压缩严重损害可懂度 [§4.3]
- 有趣的是,M1 在 cross-lingual test-en 上 SIM (0.543) 甚至高于使用 GT length 的 SIM (0.530),说明 speaking rate predictor 可能产生比 GT 更自然的时长 [Table 3]
- Speaking rate predictor 仅在中英数据训练,但能泛化到德语/法语/印地语/韩语 prompt [§4.3]

## 局限性

1. **跨语言 SIM 相对较低**: 跨语言 SIM (0.543-0.565) 远低于 intra-lingual SIM (0.658-0.764),说明去掉 prompt transcript 后模型失去了部分语言相关的说话人特征提取能力,如口音和情感的细粒度迁移 [§5]
2. **与 LLM-based 系统的差距**: CosyVoice 3 在 CV3-Eval 上跨语言 WER to-en 2.98%,本文在自建 test set 上 2.496%,但评估集不同不可直接对比;LLM-based 系统天然支持跨语言,工程复杂度更低
3. **Speaking Rate Predictor 的语言泛化**: 虽然在 4 种未见语言上有效,但未定量报告不同未见语言的差异,且未测试更远距离的语言(如阿拉伯语/日语)
4. **额外模块的工程复杂度**: 需要独立训练 speaking rate predictor + MMS forced alignment 预处理,增加了 pipeline 复杂度
5. **无主观评估**: 仅报告客观指标,缺少 MOS 人工评价
6. **跨语言测试集规模有限**: 473 samples 的自建 cross-lingual test set,仅覆盖 4 种非训练语言

## 点评

本文解决了一个具体而实际的问题: NAR flow-matching TTS 系统因依赖 prompt transcript 而无法进行跨语言克隆。解决方案清晰务实,两个核心改造(MMS word boundary 训练 + speaking rate predictor)都有充分的工程动机。

**优势**:
- 方法简洁: 没有改动 F5-TTS 的核心架构 (DiT + OT-CFM),仅改变训练数据切分方式和推理时的 duration estimation 方法
- Intra-lingual 性能不降反升: 在 WER/UTMOS 上略优于 baseline,说明 transcript-free 训练并未伤害模型能力 [Table 2]
- Speaking rate predictor 的设计有工程巧妙性: GCE loss 处理有序分类、多粒度适配不同语言都是可迁移的 idea
- 未见语言泛化: 仅在中英训练但能处理德/法/印/韩 prompt,暗示 speaking rate 是相对 language-agnostic 的声学特征

**不足**:
- 跨语言 SIM 不高,作者承认去掉 transcript 后口音/情感迁移变弱 [§5]
- 与 VALL-E X、CosyVoice 3、Qwen3-TTS 等跨语言 SOTA 缺少直接对比
- Duration prediction 的 MRE 约 12-16%,这对生成质量的影响未深入分析
- 论文较短(会议论文格式),对设计选择的 ablation 不够充分(如 GCE vs CE、sigma 取值等)

**领域意义**: 本文为 NAR flow-matching TTS 打开了跨语言通道。虽然 LLM-based 系统在跨语言任务上已经很强,但 NAR 系统在推理速度和部署简便性上仍有优势。本文证明了即使不走 LLM 路线,也可以通过工程手段赋予 NAR 系统跨语言能力。

## 可复用的 idea

1. **Speaking Rate 离散分类 + GCE Loss**: 将连续的 speaking rate 离散化为有序类别,用 Gaussian soft label 做 cross-entropy,比回归任务更稳定且自然处理了有序性。适用于任何需要预测连续有序量的场景(如 F0 预测、能量预测)
2. **MMS Forced Alignment 作为多语言 word boundary 工具**: 1000+ 语言覆盖的强制对齐能力,可用于数据预处理/分段/去噪等场景
3. **多粒度语言适配**: 英语用 phoneme-level、中文用 syllable-level 的发现提供了具体的工程指导,在涉及中英双语系统设计时可直接参考
4. **Training-time transcript dropout**: 在训练时有意丢弃部分条件信息(prompt transcript),迫使模型学到更鲁棒的表征。这一思路可推广到其他条件生成任务(如去掉 speaker label 提升音色泛化)

> [!review] 审阅 (2026-06-04, auto)
> **结论**: pass-with-fixes
> - 1 medium: KB 背景创新判断中"不如"表述不准确 (已修正)
> - 1 low: models 字段为空 (vault 无 F5-TTS 模型页, 可接受)
> 详见 `_review/Cross-Lingual F5-TTS-review.yml`
