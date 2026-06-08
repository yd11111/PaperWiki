---
type: paper
tier: deep
title: "ZMM-TTS: Zero-shot Multilingual and Multispeaker Speech Synthesis Conditioned on Self-supervised Discrete Speech Representations"
arxiv_id: "2312.14398"
source: "Sources/ZMM-TTS.pdf"
authors: [Cheng Gong, Xin Wang, Erica Cooper, Dan Wells, Longbiao Wang, Jianwu Dang, Korin Richmond, Junichi Yamagishi]
year: 2024
venue: "IEEE/ACM Transactions on Audio, Speech, and Language Processing"
tags: [multilingual-TTS, zero-shot, self-supervised-learning, discrete-speech-representation, low-resource, multispeaker, XLSR-53]
concepts: ["[[Self-SupervisedSpeechRepresentation]]", "[[SpeakerEmbedding]]", "[[NeuralVocoder]]", "[[SemanticvsAcousticTokens]]", "[[Non-autoregressiveTTS]]", "[[PhonemeRepresentation]]", "[[MelSpectrogram]]", "[[DurationPredictor]]"]
models: ["[[VITS]]", "[[HierSpeech++]]", "[[wav2vec2.0]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: ["[[LibriTTS]]"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: ZMM-TTS 是 2023 年底提出的多语言多说话人 zero-shot TTS 系统。与当时的大规模 TTS 系统 (VALL-E, NaturalSpeech 2/3, HierSpeech++) 不同,ZMM-TTS 不追求大数据量 + 强模型的路线,而是聚焦于低资源语言和未见语言的适应性。它使用 XLSR-53 (多语言 wav2vec 2.0) 的 VQ 离散表征替代 Mel spectrogram 作为中间特征,属于 SSL discrete representation 用于 TTS 的早期探索。
>
> **已有认知**:
> - [[SpeakerEmbedding]]✓: ZMM-TTS 使用 ECAPA-TDNN speaker encoder 提取 speaker embedding,这是目前最常用的 SECS encoder 架构。KB 页记录了 ECAPA-TDNN 在 zero-shot TTS 中的标准地位,以及 SECS 评估的 encoder 依赖性问题。
> - [[Zero-shotSpeechSynthesis]]✓: 当前 SOTA 系统 (CosyVoice 3, Seed-TTS, Qwen3-TTS) 已远超 ZMM-TTS 的性能水平,但它们依赖大规模数据 (100K+ hours),而 ZMM-TTS 仅用约 130 小时。
> - [[Cross-lingualVoiceCloning]]✓: ZMM-TTS 的核心差异化在于 unseen language adaptation,这是多数大规模系统未重点验证的场景。KB 页记录的现代方法 (CosyVoice 3, PFluxTTS, X-Voice) 均面向跨语言但依赖大数据。
> - [[NeuralVocoder]]✓: ZMM-TTS 使用 HiFi-GAN 作为 vocoder,符合 2020-2023 主流选择。
> - [[SemanticvsAcousticTokens]]✓: ZMM-TTS 使用的 XLSR-53 VQ codes 属于 SSL semantic tokens 范畴,论文的核心论点即是这类 token 比 Mel spectrogram 包含更少 speaker-dependent information,有利于多语言解耦。
> - [[Self-SupervisedSpeechRepresentation]][待确认]: XLSR-53 是 wav2vec 2.0 的多语言扩展版 (53 种语言, 56K 小时),属于 contrastive learning 范式。
>
> **创新判断**: 相对于同期大规模系统,ZMM-TTS 的创新在于:(1) 将 SSL 离散表征引入多语言 TTS 以改善 speaker-language 解耦;(2) 系统性验证了零资源/极低资源场景下的语言适应能力。这一路线后来被 Seamless 等系统吸收,但 ZMM-TTS 是较早的系统性验证。
>
> 检索命中: [[SpeakerEmbedding]], [[Zero-shotSpeechSynthesis]], [[Cross-lingualVoiceCloning]], [[NeuralVocoder]], [[SemanticvsAcousticTokens]] | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 XLSR-53 的 VQ 离散表征替代 Mel spectrogram 作为中间特征,构建多语言多说话人 TTS 系统,实现对未见说话人和未见语言的零样本/少样本合成
> - **路线**: Text → (characters/IPA/XPhoneBERT) → FastSpeech-based txt2vec → XLSR-53 discrete codes → vec2wav (HiFi-GAN or multi-stage VQ + HiFi-GAN generator) → Waveform
> - **指标**: 6 语言 seen speaker MOS 最高 4.20 (ZMM-TTS2x, fr) [Table V]; unseen speaker MOS 最高 4.32 (ZMM-TTS2x, fr) [Table VI]; 低资源 Italian 15min few-shot MOS 3.33 [Table VIII]; vs LibriSpeech test-clean SECS 0.644, WER 2.37%, RTF 0.003 [Table IX]
> - **可借鉴**: (1) SSL 离散表征天然 speaker-content 解耦,可用于低资源多语言场景;(2) 去掉 language embedding 即可 zero-shot inference 未见语言;(3) 同时用 cross-entropy (分类) + MSE (回归) loss 预测离散 code,利用 codebook 的连续空间信息
> - **局限**: 英语 MOS 明显低于其他语言 (~3.0 vs ~4.2);与大规模系统 (VALL-E-X, HierSpeech++) 相比 speaker similarity 差距显著;未评估跨语言合成质量;代码已开源但数据处理流程复杂

## 核心问题

当时多语言多说话人 TTS 面临三个瓶颈 [§I]:
1. **数据瓶颈**: 高质量配对数据 (text + studio-quality audio) 仅限少数资源丰富语言
2. **Speaker-language 纠缠**: Mel spectrogram 在时频域高度相关,难以解耦说话人和语言信息 [§I, para 3]
3. **大规模系统的语言局限**: VALL-E, NaturalSpeech 2/3 等均聚焦英语或少数高资源语言,未验证低资源场景 [§II-D, Table I]

ZMM-TTS 的核心假设: SSL 模型学到的离散表征比 Mel spectrogram 包含更少 speaker-dependent information [§I, 引用 [23]],因此更适合多语言多说话人系统。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ZMM-TTS 是两阶段非自回归系统 [§III, Fig 1]:
1. **txt2vec**: 文本 → XLSR-53 离散 code index (类似 acoustic model)
2. **vec2wav**: 离散表征 → 波形 (类似 vocoder)

**中间表征**: XLSR-53 (多语言 wav2vec 2.0) 的 product quantization 输出。XLSR-53 使用 G=2 个 codebook,每个 M=320 entries,D=384 维 [§III]。给定音频,XLSR-53 编码为离散 code index 序列 V 和对应的连续表征 R (通过 codebook lookup)。

### 关键设计选择

#### 为什么用 SSL 离散表征而非 Mel spectrogram?

[论文原文] Mel spectrogram 在时频域高度相关,speaker-dependent information 难以剥离 [§I]。DSE-TTS [23] 已证明 wav2vec 2.0 的 VQ features 包含显著更少的 speaker information [§II-C]。因此 SSL 离散表征天然有利于 speaker-language 解耦。

[agent 解读] 这一设计选择的更深层逻辑是: SSL 模型在大规模多语言数据上预训练 (XLSR-53 用 56K 小时 53 种语言),其 VQ 层已经通过 contrastive learning 学会了跨语言共享的 phonetic units,这些 units 本质上更接近 [[SemanticvsAcousticTokens|semantic tokens]],即编码了内容而非音色。

#### 三种文本输入表征 [§III-A-1]

| 输入类型 | Encoder | 适用场景 |
|---------|---------|---------|
| Characters | FFT (Feed-Forward Transformer) | 简单,但跨语言 grapheme 差异大 |
| IPA (Epitran) | FFT | 统一音素空间,但缺乏上下文建模 |
| XPhoneBERT | BERT-base (330M sentences, 100+ languages) | 预训练跨语言音素表征,低资源场景最优 |

[论文原文] XPhoneBERT 在低资源场景下优势显著,因为其预训练跨语言知识可弥补目标语言数据不足 [§VI-B-1]。

#### txt2vec 的双重 Loss 设计 [§III-A-3]

txt2vec 同时优化两个目标:
1. **分类 loss** (Lcla): 预测离散 code index 的交叉熵 [Eq. 3]
2. **回归 loss** (Lmse): 预测连续 representation 的 MSE [Eq. 5]

[论文原文] 连续空间的回归 loss 帮助学习 "cross-lingual information from shared quantized latent speech representations in continuous space" [§III-A-3]。

[agent 解读] 这是一个巧妙的设计: 离散 classification loss 保证输出落在有效 codebook entry 上,连续 regression loss 利用 codebook 的几何结构传递梯度信息。推理时仅用 argmax 选 code index 然后 lookup [Eq. 7],但训练时的回归 loss 提供了更平滑的梯度信号。

#### vec2wav 的两种实现 [§III-B]

**ZMM-TTS1** (with Mel-based vocoder, Fig 3):
- SSL 离散表征 → 上采样(8x) → 下采样(5x) → FFT decoder → Mel spectrogram → HiFi-GAN
- 上下采样比例由 XLSR 帧率 (20ms) 和 Mel 帧率 (12.5ms) 的比值决定: R:Mel:Speech = 5:8:1600 [§III-B-1]

**ZMM-TTS2** (without Mel-based vocoder, Fig 4):
- SSL 离散表征 → multi-stage multi-head VQ encoder → frame decoder + HiFi-GAN generator → 波形
- 2-stage 4-head codebook 结构,不同时间分辨率建模 [§III-B-2, 引用 [71,72]]

[论文原文] ZMM-TTS2 避免了 Mel spectrogram 预测不准确带来的误差传播 [§V-A-2]。

#### Speaker/Language 控制 [§III-A-2]

- **Speaker**: ECAPA-TDNN 预训练 speaker encoder (VoxCeleb 1&2)
- **Language**: 64-dim 可学习 language embedding table
- 均注入 decoder 和 duration predictor

[论文原文] 关键: 在 zero-shot unseen language 场景下,去掉 language embedding layer 即可直接推理未见语言 [§III, Eq. 1 注]。这依赖 SSL 表征已编码的跨语言 phonetic 信息。

### 训练策略

- txt2vec, vec2mel, vec2wavVQ 各自独立训练 [§IV-C]
- txt2vec / FastSpeech / vec2mel: 1.2M steps, batch size 16, A100
- HiFi-GAN: 2.5M steps
- vec2wavVQ: 1M steps, UnivNet discriminator for adversarial training
- XPhoneBERT: 前 25% steps 冻结,后续步骤更新 [§IV-C]
- YourTTS baseline: 1M steps, batch size 32, A100

## 实验

### 数据规模

6 语言 (en/fr/ge/pt/es/sw),每种语言约 20 小时,总计约 130 小时。数据来源: MLS + GlobalPhone + CSS10 + LJSpeech + NST Swedish [Table II]。

### 高资源语言 (6 语言) 核心结果

| 指标 | 系统 | Seen Speaker 最佳 | Unseen Speaker 最佳 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS | ZMM-TTS2x | 4.20 (fr) | 4.32 (fr) | 自建 6 语言 | [Table V, VI] |
| MOS | YourTTS | 3.35 (en) | 3.48 (en) | 自建 6 语言 | [Table V, VI] |
| DMOS | ZMM-TTS2x | 4.58 (ge, seen) | 4.25 (es, unseen) | 自建 6 语言 | [Table V, VI] |
| SECS | ZMM-TTS2x | 0.955 (fr, seen) | 0.936 (fr, unseen) | 自建 6 语言 | [Table V, VI] |
| CER | ZMM-TTS1x | 2.90 (en, seen) | 2.66 (en, unseen) | Whisper ASR | [Table VII] |

**关键发现**:
1. ZMM-TTS1/2 显著优于 FSM baseline (Mann-Whitney U test, p<0.05) [§V-A-1]
2. ZMM-TTS 对训练数据音质不敏感: FSM 在 GlobalPhone (非专业录音) 上 MOS 极低,ZMM-TTS 表现稳定 [论文原文: "the proposed system is considered to be less sensitive to the sound quality of the training data", §V-A-1]
3. ZMM-TTS2 > ZMM-TTS1: 直接映射避免 Mel 预测的误差传播 [§V-A-2]
4. VQ 模块可提升自然度: FSM2 > FSM1, ZMM-TTS2 > ZMM-TTS1 [§V-A-2]
5. XPhoneBERT 对 FSM 始终有益,对 ZMM-TTS 则因语言而异 [§V-A-4]

### 低资源语言结果 (Italian, Polish)

| 场景 | 系统 | Italian MOS | Polish MOS | Italian CER | Polish CER | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| 15min few-shot | ZMM-TTS2x | 3.33 | 3.54 | 3.92% | 8.27% | [Table VIII] |
| 5min few-shot | ZMM-TTS2x | 3.09 | 3.28 | 4.02% | 10.01% | [Table VIII] |
| 2.5min few-shot | ZMM-TTS2x | 2.90 | 3.05 | 4.38% | 14.64% | [Table VIII] |
| Zero-shot (0 data) | ZMM-TTS2x | 2.88 | 2.95 | 5.11% | 15.40% | [Table VIII] |
| YourTTS 1h | YourTTS | 不可理解 | 不可理解 | 89.77% | 88.86% | [Table VIII] |
| 35/21h high-resource | ZMM-TTS2x | 4.12 | 3.99 | 6.01% | 89.68% | [Table VIII] |

**关键发现**:
1. YourTTS 无法用有限数据 fine-tune 未见语言,即使 1 小时数据仍无法合成可理解语音 [§VI-B-3]
2. ZMM-TTS2x 用 2.5 分钟数据即可达到较高 speaker similarity (SECS ~0.89-0.91) [§VI-B-4]
3. Italian (与训练集中的 Romance 语言近亲) 在 zero-shot 下可生成可理解语音;Polish (Slavic, 较远) 差距更大 [§VI-B-5]
4. XPhoneBERT 在低资源场景下优势显著: FSM2c vs FSM2x CER 7.13% vs 3.27% (Italian 15min) [§VI-B-1]

### 与大规模系统对比 (英语, LibriSpeech test-clean)

| 系统 | 训练数据 (h) | Speaker数 | WER | SECS | UTMOS | Params | RTF | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VALL-E-X | >3122 | 1739 | 26.77 | 0.512 | 3.29 | 395M | 5.917 | [Table IX] |
| HierSpeech++(a) | 2311 | 555 | 2.03 | 0.591 | 4.40 | 204M | 0.217 | [Table IX] |
| StyleTTS 2 | 546 | 130 | 3.06 | 0.455 | 4.23 | 191M | 0.070 | [Table IX] |
| ZMM-TTS2x(a) | 1151 | 245 | 4.06 | 0.432 | 3.89 | 167M | 0.003 | [Table IX] |
| ZMM-TTS2x(b) | 2311 | 555 | 2.37 | 0.644 | 4.07 | 167M | 0.003 | [Table IX] |

**关键发现**:
1. 相同数据量下 (LT-960),ZMM-TTS2x(b) SECS 最高 (0.644),证明方法有效 [§App-B-1]
2. ZMM-TTS2x RTF 仅 0.003,是所有系统中最快的 (非自回归结构) [§App-B-4]
3. UTMOS 受训练数据质量影响: ZMM-TTS2x(a) 用了 GlobalPhone 等非高质量数据 [§App-B-2]
4. 参数量最小 (167M vs 其他 191-395M) [Table IX]

## 局限性

1. **英语 MOS 低**: ZMM-TTS 在英语上的 MOS 始终低于其他语言 (~2.98 vs ~4.20),论文未给出明确解释 [Table V, VI]。[agent 解读] 可能因为英语测试使用 CMU ARCTIC 句子,与训练数据 (MLS audiobook) 风格差异大。
2. **多语言训练数据下 speaker similarity 较低**: ZMM-TTS2x(a) 用多语言混合数据 (1151h, 245 speakers) 时 SECS 仅 0.432,远低于用 LibriTTS (2311h, 555 speakers) 训练的 HierSpeech++(a) 0.591 [Table IX]。尽管同等数据量下 ZMM-TTS2x(b) 达到最高 SECS (0.644),但核心场景 (多语言低资源) 下 speaker similarity 仍是弱项。
3. **未评估跨语言合成**: 论文仅评估 intra-lingual (说话人和文本同语言),未测试 cross-lingual (中文说话人说英文) 质量 [§IV-D-2]。
4. **zero-shot 语言适应有限**: Polish (Slavic) 与训练语言差异大时,zero-shot 效果不佳 (CER 15.40%) [Table VIII]。论文指出效果 "heavily affected by the similarity between the domains" [§VI-B-5]。
5. **数据预处理复杂**: 需要 6 个不同来源的数据集组合、音频重采样 (16kHz)、sv56 幅度归一化 [§IV-A]。
6. **Speaker encoder 单一**: 仅使用 ECAPA-TDNN,未探索 SCL (Speaker Consistency Loss) 等增强 speaker similarity 的技术 [§V-B-1]。

## 点评

ZMM-TTS 的核心贡献是在低资源多语言场景中验证了 SSL 离散表征替代 Mel spectrogram 的可行性和优势。这一路线的价值不在于性能(与同期或后续大规模系统相比差距明显),而在于以下 insight:

1. **SSL 表征的鲁棒性**: ZMM-TTS 对训练数据音质不敏感这一发现具有实践意义。XLSR-53 在 56K 小时非高质量数据上预训练,其 VQ 表征已经过滤了噪声和音质差异,这让 TTS 系统可以利用更多"不完美"的数据。
2. **Language embedding 的可选性**: 去掉 language embedding 即可 zero-shot 推理未见语言,说明 SSL 预训练已经将跨语言 phonetic 信息编码到离散表征中,language embedding 是锦上添花而非必需。
3. **与后续发展的关系**: ZMM-TTS 的 SSL discrete token 路线后来被更大规模系统 (Seamless, 100+ 语言) 和监督式路线 (CosyVoice S3 tokenizer) 分别继承和超越。ZMM-TTS 可视为从"SSL 表征用于 ASR"到"SSL 表征用于 TTS"这一迁移的系统性验证。

局限方面,论文的实验设计有值得商榷之处: (1) 仅评估 intra-lingual 而非 cross-lingual,削弱了"多语言系统"的论证力度;(2) low-resource 实验中 35h Italian baseline 的 ZMM-TTS2x 高于 Polish 的 CER 竟高达 89.68% [Table VIII],但论文未讨论这一异常值。

## 可复用的 idea

1. **SSL 离散表征做中间特征**: 当训练数据音质参差不齐时,用 SSL VQ codes 替代 Mel spectrogram 可以提升鲁棒性。核心机制是 SSL 预训练已经去除了 speaker-dependent 和噪声信息。
2. **Classification + Regression 双 loss**: 预测离散 token 时,同时优化分类 loss (保证落在有效 codebook entry) 和回归 loss (利用连续空间结构),可应用于任何需要预测 VQ codes 的场景 [Eq. 3-6]。
3. **去掉 language embedding 实现 zero-shot**: 对于多语言系统,可以训练时用 language embedding 提升已见语言质量,推理时去掉 language embedding 适应未见语言。前提是中间表征本身已编码跨语言信息 (如通过多语言 SSL 预训练)。
4. **Multi-stage multi-head VQ 做波形重建**: vec2wavVQ 的多阶段多头 codebook 结构 [§III-B-2],在不同时间分辨率上量化离散表征,可用于任何 discrete-to-waveform 任务。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节因果解释充分,WHY/HOW 清晰 |
> | 可信赖 | pass | 修正了 Table IX 列交换错误; 标注覆盖率高 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注严格 |
> | 可定位 | pass | KB 背景谱系定位具体,有对比基准 |
> | 不污染 | pass | 本次不做反向更新,frontmatter 挂接合理 |
> 
> Issues: 4 (high: 0, medium: 1, low: 3)
> 详见 `_review/ZMM-TTS-review.yml`
