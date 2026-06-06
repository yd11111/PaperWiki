---
type: paper
tier: deep
title: "Generative Adversarial Training for Text-to-Speech Synthesis Based on Raw Phonetic Input and Explicit Prosody Modelling"
arxiv_id: "2310.09636"
source: "Sources/GATrainingRawPhonetic.pdf"
authors: [Tiberiu Boros, Stefan Daniel Dumitrescu, Ionut Mironica, Radu Chivereanu]
year: 2023
venue: "Blizzard Challenge 2023 / Interspeech SSW"
tags: [TTS, end-to-end, GAN, vocoder, prosody, pitch, duration, BERT, French, open-source]
concepts: ["[[NeuralVocoder]]", "[[ProsodyModeling]]", "[[DurationPredictor]]", "[[GlobalStyleTokens]]", "[[Text-to-SpeechPipeline]]", "[[MelSpectrogram]]"]
models: ["[[论文笔记/GATrainingRawPhonetic|TTS-Cube]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页: [[NeuralVocoder]], [[ProsodyModeling]], [[DurationPredictor]], [[GlobalStyleTokens]], [[Text-to-SpeechPipeline]], [[MelSpectrogram]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[NeuralVocoder]]✓, [[ProsodyModeling]]✓ | 过滤: [[DurationPredictor]](pending-review), [[GlobalStyleTokens]](pending-review), [[Text-to-SpeechPipeline]](pending-review), [[MelSpectrogram]](pending-review) | 未命中但可能相关: 无

**谱系定位**: 本文属于 GAN-based end-to-end TTS 路线。在 [[NeuralVocoder]] 的分类体系中,HiFi-GAN (Kong et al., NeurIPS 2020) 是 GAN-based vocoder 的事实标准,以 MPD+MSD 判别器、MRF 生成器实现实时 13.4x 推理速度和 14M 参数量。本文的关键差异在于 **不以 mel spectrogram 为中间表示**,而是将 HiFi-GAN 直接条件化于自定义网络的 learned embeddings,实现 phoneme→waveform 的端到端生成。

**已有认知**: 在 [[ProsodyModeling]] 中,显式韵律建模 (explicit prosody) 指直接预测 pitch/duration/energy 等可观测参数,代表工作为 FastSpeech 2 的 variance adaptor。本文采用类似路线但有区别: duration 用 softmax 分类 (离散分布) 而非回归,pitch 用连续预测 + voiced/unvoiced gate。在 [[DurationPredictor]] [待确认] 中,forced alignment 训练 + 推理时预测是经典范式 (FastSpeech 系列),本文使用相同策略但通过 non-uniform upsampling (PyTorch indexing) 实现帧级展开。

**创新判断**: 相比同期 Blizzard Challenge 参赛系统 (多为 Tacotron + vocoder 两阶段),本文的主要创新在于 (1) 跳过 mel spectrogram 中间表示做端到端 phoneme→audio,(2) 将预训练语言模型 (CamemBERT) 嵌入到 TTS 流水线中端到端微调,(3) 保留显式韵律建模而非纯隐式。这些选择在 2023 年已非前沿 (VITS 2021 已实现更优雅的端到端),但在 Blizzard Challenge 的实际工程场景中取得了有竞争力的结果 (6/20)。

## 速查

> [!summary] 速查
> - **一句话**: 基于 CamemBERT + 显式韵律 (pitch/duration) + HiFi-GAN 的端到端法语 TTS,跳过 mel spectrogram 直接从 phoneme 生成波形
> - **路线**: 文本→Phonemizer (G2P)→Phoneme Embeddings + CamemBERT Word Embeddings→Backbone (Conv+BiLSTM)→三路并行 BiLSTM (Duration/Pitch/HiFi-GAN Conditioning)→Non-uniform Upsampling→HiFi-GAN→Waveform
> - **指标**: Blizzard 2023 排名 6/20; NEB MOS 3.6-4.3 (按评估组), AD MOS 3.2-4.2; SUS WER 0.162 [Table 1, Fig 3-10, Fig 11]
> - **可借鉴**: (1) 将 BERT 嵌入与音素级表示对齐的两层 non-uniform upsampling 策略 (word→phoneme→frame); (2) 跳过 mel 中间表示避免两阶段 compounding error 的实践验证; (3) Duration 用 softmax 分类而非回归的设计选择
> - **局限**: 论文未给出与同期端到端系统 (如 VITS) 的直接对比; 仅在法语上验证; 未详细描述 "discreet style tokens" 的具体实现; 训练需 3 周 (RTX 3090)

## 核心问题

本文要解决的核心问题是: **如何构建一个端到端的 TTS 系统,直接从 raw phonetic input 生成高质量音频,同时保留对韵律 (pitch/duration) 的显式控制能力?**

传统 TTS pipeline 分为 text analysis → acoustic model (生成 mel spectrogram) → vocoder (mel→waveform) 三阶段。这种分离带来 compounding error: 声学模型预测的 mel 与训练时的真实 mel 之间的偏差会在 vocoder 阶段被放大,导致 artifacts [论文原文, §3.3 note 6]。已有的端到端方案 (如 WaveNet conditioned on linguistic features) 虽然避免了这个问题,但通常依赖自回归生成 (慢) 或隐式韵律建模 (不可控)。本文试图在端到端和显式韵律控制之间找到平衡。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由两个模块组成 [§3]:

1. **主 TTS 模块**: CamemBERT + 自定义韵律网络 + HiFi-GAN,端到端训练 [Fig 1]
2. **Phonemizer**: 独立训练的 hybrid grapheme-to-phoneme 模块,不参与端到端优化 [§3.2]

数据流: 文本→Phonemizer→phoneme 序列 (含标点)→phoneme embeddings + CamemBERT word embeddings→backbone (3 Conv + BiLSTM)→三路并行 BiLSTM stack→HiFi-GAN→waveform [Fig 1, Fig 2]

### 关键设计选择

**1. 跳过 mel spectrogram 的端到端路线**

系统不经过 mel spectrogram 中间表示,HiFi-GAN 直接以自定义网络输出的 learned embeddings 为条件生成波形 [§3.1]。

为什么这样做? 作者在实验中发现: "Conditioning HiFi-GAN on the mel spectrogram and training the system as two different modules (one to convert text into a mel-spectrogram and the other to generate audio based on this spectrogram) always resulted in audio artifacts, which were not removed even by fine-tuning the system" [§3.3, note 6, 论文原文]。这一观察与 [[MelSpectrogram]] 概念页中 LongCat-AudioDiT 对 mel compounding error 的量化发现一致 [agent 解读]。

**2. 两层 Non-uniform Upsampling**

系统使用两层 upsampling 解决文本到音频的长度不匹配 [§3.1]:
- **第一层**: CamemBERT word embeddings → phoneme 级,将每个 word/sub-token embedding 复制到该词对应的所有 phoneme 位置。当 BERT tokenizer 将词拆分为多个 sub-token 时,使用第一个 sub-token 的 embedding [论文原文]
- **第二层**: phoneme 级 → frame 级,按 duration (训练时用 gold-standard,推理时用预测值) 复制每个 phoneme embedding [论文原文]

实现上使用 PyTorch indexing 而非朴素的 repeat+concatenate 以加速 [§3.1, 论文原文]。

**3. 显式韵律建模: Duration 与 Pitch 的不同处理**

Duration 和 Pitch 使用不同的建模方式 [§3.1, Fig 2]:
- **Duration**: 建模为离散分布 (softmax),即对每个 phoneme 的可能帧数做分类 [论文原文]。这与 FastSpeech 2 的回归式 MSE loss 不同,也与 VITS 的 stochastic duration predictor (normalizing flow) 不同 [agent 解读]
- **Pitch**: 建模为连续变量 + voiced/unvoiced 二分类 gate [Fig 2, 论文原文]。训练时 pitch ground truth 通过 RAPT 算法 [17] 提取,duration 使用数据集提供的标注 [§3.1]

为什么 duration 用分类而非回归? 论文未给出显式理由。可能的原因是分类 (softmax) 可以自然建模 duration 的多模态分布 (同一 phoneme 在不同上下文中可能有不同的合理时长),而回归倾向于预测均值导致 over-smoothing [agent 解读]。

**4. CamemBERT 端到端微调**

使用 CamemBERT [16] (法语 BERT) 提供 contextualized word embeddings,与整个系统一起端到端优化,但使用更小的固定学习率 (10^-6 vs 主网络 2×10^-4) [§3.3, 论文原文]。

作者实验表明 [§3.3]:
- 不微调 BERT → 结果 sub-optimal [note 4, 论文原文]
- 其他法语 BERT 模型 → informal 听测不如 CamemBERT [note 1, 论文原文]
- 去上下文化的 FastText embeddings → 缺乏 emotions、character voices 和 ad-hoc pauses [note 2, 论文原文]

**5. 三路并行但非联合优化**

三路 BiLSTM stack (duration/pitch/HiFi-GAN conditioning) 共享 backbone 但独立优化 [§3.1]:
- Duration stack 和 Pitch stack 用 gold-standard 标签监督
- HiFi-GAN conditioning stack 通过 GAN loss 优化
- 由于使用 gold-standard duration 做 forced alignment,三路不会同时到达最优点 [§3.1, 论文原文]

这意味着系统虽然"端到端训练",但并非完全联合优化 — 韵律模块和波形生成模块仍然各自优化不同目标 [agent 解读]。

### Phonemizer

Grapheme-to-phoneme (G2P) 实现为序列标注网络 (非 seq2seq attention) [§3.2]:
- 利用数据集提供的 1:1 grapheme-phoneme 对齐
- 架构: 3 Conv layers + BiLSTMs + softmax [§3.2]
- 关键改进: 在输出中保留标点和空格,因为它们提供重要的韵律线索 (逗号、句号、问号等) [§3.2, 论文原文]
- 整句训练以利用上下文 (解决同形异音词和连音问题) [§3.2]
- 模型选择: SAR (sentence accuracy rate),patience=20 early stopping [§3.3]

### 训练策略

- 主网络: 衰减学习率 2×10^-4,衰减率 10^-5/step,训练 1M 步,无模型选择 (直接用最后一个 checkpoint) [§3.3]
- CamemBERT: 固定学习率 10^-6 [§3.3]
- 硬件: NVIDIA RTX 3090 (24GB),batch size 16,约 3 周 [§3.3]

## 实验

评估在 Blizzard Challenge 2023 框架下进行,使用法语数据集 [22] [§4]:
- Speaker NEB: ~50 小时有声书数据,高质量,有对齐和音素标注
- Speaker AD: 仅 ~2 小时对齐数据

| 指标 | 本文 (System E) | 排名 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Overall ranking | 6/20 | 6/20 | Blizzard 2023 | [§4] |
| MOS (NEB, SE native) | 3.8 | — | NEB | [Table 1] |
| MOS (NEB, SE non-native) | 4.3 | — | NEB | [Table 1] |
| MOS (NEB, N-SE native) | 3.6 | — | NEB | [Table 1] |
| MOS (NEB, SP) | 3.6 | — | NEB | [Table 1] |
| MOS (NEB, SR) | 4.2 | — | NEB | [Table 1] |
| MOS (AD, SE native) | 3.2 | — | AD | [Table 1] |
| MOS (AD, SE non-native) | 3.6 | — | AD | [Table 1] |
| MOS (AD, SP) | 3.5 | — | AD | [Table 1] |
| MOS (AD, SR) | 4.1 | — | AD | [Table 1] |
| SUS WER | 0.162 (std 0.23) | on-par with top | NEB | [§4, Fig 11] |

**关键发现**:
1. 语音专家 (SE) 评分高于非专家 (N-SE) — 这不寻常,因为专家通常更苛刻 [§4, 论文原文]
2. 法语母语者评分低于非母语者 — 作者推测是音素转写问题导致母语者能发现发音错误 [§4, 论文原文]
3. SUS 可懂度测试 WER 0.162,与多数 top 系统持平 [§4]

## 局限性

1. **缺乏消融实验**: 没有对端到端 vs 两阶段 (mel intermediate)、CamemBERT vs 其他 embeddings、softmax duration vs regression 等关键设计做定量消融,仅有 informal 观察 [§3.3]
2. **Style tokens 描述不足**: Abstract 提及 "discreet style tokens" 用于 character voice matching,但正文中几乎没有详细描述其实现 [agent 观察]
3. **仅法语验证**: 虽然提供了英语模型下载,但论文评估仅限于法语 Blizzard Challenge 数据 [§4, §5]
4. **训练效率**: 单 GPU 3 周训练时间,且没有模型选择 (直接用最后 checkpoint),可能不是最优 [§3.3]
5. **与现代端到端基线缺少对比**: 论文发表于 2023 年 10 月,但未与 VITS (2021)、JETS (2022) 等已有端到端系统进行直接定量比较
6. **训练非完全联合优化**: 虽然号称端到端,但 duration/pitch 使用 gold-standard 监督,与 GAN conditioning 分别优化,不是真正的联合优化 [§3.1]

## 点评

这是一篇扎实的系统描述论文 (Blizzard Challenge 参赛系统报告),工程价值大于学术贡献。

**优点**:
- 提供了"跳过 mel spectrogram 做端到端 phoneme→audio"的一个实践案例,并给出了 mel 中间表示导致 artifacts 的经验证据 [§3.3, note 6]
- Duration 用 softmax 分类而非回归是一个有趣的设计选择,虽然论文没有做消融,但后续工作 (如 DMOSpeech 2 的 300-class duration 分类) 验证了 duration 离散化的价值
- CamemBERT 端到端微调的几条负面结论 (不微调 → sub-optimal; FastText → 缺情感) 虽然 informal,但具有实际参考价值
- 完全开源 (TTS-Cube),降低了复现门槛

**不足**:
- 作为 2023 年的工作,架构 (BiLSTM + HiFi-GAN) 在方法论上已经落后于同期的 diffusion/flow/LLM-based 方法
- "discreet style tokens" 作为 abstract 中的亮点却在正文中缺乏描述,是一个明显的叙述缺陷
- 实验评估完全依赖 Blizzard Challenge 框架,没有独立的消融实验来验证各设计选择的贡献

**定位**: 在 TTS pipeline 演进中,本文处于 Stage 3 (Tacotron/FastSpeech + neural vocoder) 向 Stage 4 (fully E2E) 过渡的位置。它保留了显式韵律建模 (类似 FastSpeech 2) 但跳过了 mel 中间表示 (类似 VITS 的端到端目标),是一个工程导向的折中方案。

## 可复用的 idea

1. **两层 Non-uniform Upsampling 策略**: word embedding → phoneme → frame 的分层对齐,用 PyTorch indexing 加速,适用于任何需要将词级语言模型表示与音素级/帧级声学模型对齐的场景
2. **Duration Softmax 分类**: 将 duration 建模为分类而非回归,可自然表达 duration 的多模态分布;后续 DMOSpeech 2 的 300-class duration policy 可视为此思路的扩展
3. **Phonemizer 保留标点**: 在 G2P 输出中保留标点和空格作为韵律线索,简单但有效的工程 trick
4. **BERT 低学习率端到端微调**: 预训练语言模型用 100x 更低的学习率与 TTS 系统联合微调,平衡了保留预训练知识与适应 TTS 任务

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节含充分因果解释,速查卡片有 4 个具体可迁移 trick |
> | 可信赖 | pass-with-fixes | 2 处 MOS 数值与原文不符已修正 (NEB SE native 3.8, AD SE native 3.2) |
> | 可区分 | pass | 来源标注覆盖率 >90%,严格区分论文原文/agent 解读 |
> | 可定位 | pass | KB 背景有具体谱系 (HiFi-GAN/FastSpeech 2/VITS),创新判断有对比 |
> | 不污染 | pass | 反向更新仅追加,无新建实体页,无 overclaim |
> 
> Issues: 4 (high: 0, medium: 2, low: 2)
> 详见 `_review/GATrainingRawPhonetic-review.yml`

---
检索命中: [[NeuralVocoder]]✓, [[ProsodyModeling]]✓ | 过滤: [[DurationPredictor]](pending-review), [[GlobalStyleTokens]](pending-review), [[Text-to-SpeechPipeline]](pending-review), [[MelSpectrogram]](pending-review) | 未命中但可能相关: 无
