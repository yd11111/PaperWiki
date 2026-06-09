---
type: concept
title: "Semantic vs Acoustic Tokens"
aliases: [语义 token 与声学 token, Semantic Tokens, Acoustic Tokens, Token Hierarchy, 语音 token 层级, Discrete Speech Features]
category: "representation"
tags: [speech-representation, tokenization, discrete-token, speech-LM, trade-off]
key_papers: ["[[论文笔记/CosyVoice|CosyVoice]]", "[[论文笔记/Spark-TTS|Spark-TTS]]", "[[论文笔记/CosyVoice2|CosyVoice 2]]", "[[论文笔记/SoundStorm|SoundStorm]]", "GSLM (Lakhotia et al., 2021)", "[[论文笔记/AudioLM|AudioLM]]", "SpeechTokenizer (Zhang et al., 2024)", "pGSLM (Kharitonov et al., 2022)", "SPIRIT-LM (Nguyen et al., 2024)", "[[论文笔记/Moshi|Moshi]]", "[[论文笔记/Survey-DiscreteAudioTokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/SPEAR-TTS|SPEAR-TTS]]", "[[论文笔记/HuBERT|HuBERT]]", "[[论文笔记/UniAudio|UniAudio]]", "[[论文笔记/BASETTS|BASE TTS]]", "[[论文笔记/Make-A-Voice|Make-A-Voice]]", "[[论文笔记/HierSpeech++|HierSpeech++]]", "[[论文笔记/MELLE|MELLE]]", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/wav2vec2.0|wav2vec 2.0]]", "[[论文笔记/WavLM|WavLM]]", "[[论文笔记/w2v-BERT|w2v-BERT]]", "[[论文笔记/SNAC|SNAC]]", "[[论文笔记/RepCodec|RepCodec]]", "[[论文笔记/SpeechAlign|SpeechAlign]]", "[[论文笔记/RIO|RIO]]", "[[论文笔记/FlexiCodec|FlexiCodec]]", "[[论文笔记/StableToken|StableToken]]", "[[论文笔记/LatentLM|LatentLM]]", "[[论文笔记/CLEAR|CLEAR]]", "[[论文笔记/VibeVoice|VibeVoice]]", "[[论文笔记/SiTok|SiTok]]", "[[论文笔记/AnalysingNeuralAudioCodecs|NAC Token Language Analysis]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/VoXtream|VoXtream]]", "[[论文笔记/VoxtralTTS|Voxtral TTS]]", "[[论文笔记/FishAudioS2|Fish Audio S2]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/Cont-SPT|Cont-SPT]]", "[[论文笔记/SemaVoice|SemaVoice]]", "[[论文笔记/LiveSpeech2|LiveSpeech 2]]", "[[论文笔记/NAST|NAST]]", "[[论文笔记/Vec-TokSpeech|Vec-Tok Speech]]", "[[论文笔记/LM-SPT|LM-SPT]]", "[[论文笔记/UniTTS|UniTTS]]", "[[论文笔记/C2F-LM|C2F-LM]]", "[[论文笔记/SpeechAccentLLM|SpeechAccentLLM]]", "[[论文笔记/SMLLE|SMLLE]]", "[[论文笔记/SecoustiCodec|SecoustiCodec]]", "[[论文笔记/FuseCodec|FuseCodec]]", "[[论文笔记/MBCodec|MBCodec]]", "[[论文笔记/MELA-TTS|MELA-TTS]]", "[[论文笔记/TaDiCodec|TaDiCodec]]", "[[论文笔记/DualSpeechLM|DualSpeechLM]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/BridgeCode|BridgeCode]]", "[[论文笔记/DialoSpeech|DialoSpeech]]", "[[论文笔记/Ming-UniAudio|Ming-UniAudio]]", "[[论文笔记/GRPO-TTS|GRPO-TTS]]", "[[论文笔记/Semantic-VAE|Semantic-VAE]]", "[[论文笔记/SAC|SAC]]", "[[论文笔记/ProsodyEval|ProsodyEval]]", "[[论文笔记/DashengAudioGen|Dasheng AudioGen]]"]
origin_paper: "[[论文笔记/Survey-SpeechLanguageModels|Cui et al., Speech Language Models, 2024]]"
related_concepts: ["[[SpeechTokenizer]]", "[[ResidualVectorQuantization]]", "[[SpeechLanguageModel]]", "[[CodecLanguageModel]]", "[[AudioTokenizerTaxonomy]]", "[[Single-codebookvsMulti-codebook]]", "[[Self-SupervisedSpeechRepresentation]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Semantic tokens 和 Acoustic tokens 是 Speech Language Model 中两类根本不同的语音离散表征,分别侧重于内容语义和声学保真度。这一二分法是 SpeechLM 设计中最核心的 trade-off 之一。

### Semantic Tokens (语义 token)
由语义理解目标训练的 tokenizer 产生,旨在捕获语音的内容和含义:
- **训练目标**: Self-supervised learning (masked prediction, contrastive learning)
- **代表**: HuBERT (k-means on hidden states), w2v-BERT, wav2vec 2.0
- **特点**: 与文本对齐良好,语义连贯性强,但缺乏高频声学细节 (pitch, timbre 等)
- **形式**: encoder 输出经 k-means 聚类离散化: s = d(MFCC(a); θ_d)

### Acoustic Tokens (声学 token)
由声学生成目标训练的 tokenizer 产生,旨在保留高保真语音重建所需的声学特征:
- **训练目标**: Speech reconstruction / resynthesis
- **代表**: EnCodec, SoundStream (均使用 RVQ)
- **特点**: 高保真音频重建,但语义对齐差,序列长度大
- **形式**: 多级 RVQ: s = (d_1(v), d_2(v - v̂_1), ..., d_R(v - Σv̂_r))

### Mixed Tokens (混合 token)
兼顾语义理解和声学生成的第三类,尝试平衡两者的优劣:
- **代表**: SpeechTokenizer (RVQ 第一层蒸馏 HuBERT, 后续层量化残差), Mimi (单 VQ 语义 + 额外 RVQ 声学)
- **特点**: 仍处于早期阶段,但在 Moshi/SpeechGPT-Gen 中展现前景

## 核心 Trade-off

| 维度 | Semantic Tokens | Acoustic Tokens |
|------|-----------------|-----------------|
| 语义连贯性 | 强 (与文本对齐好) | 弱 (纯声学重建) |
| 声学保真度 | 弱 (缺高频细节) | 强 (RVQ 多级量化) |
| 表现力/副语言 | 弱 (prosody/timbre 丢失) | 中 (保留但无语义结构) |
| 序列长度 | 短 (25-50 Hz) | 长 (多级 codebook 展开) |
| 下游 vocoder | 需 input-enhanced (CFM→HiFi-GAN) | 可 direct synthesis |
| 主要用途 | 语义理解+生成 (ASR, TTS) | 高保真重建 (codec) |

**Survey 核心发现**: "While semantic tokens align well with text and excel in producing semantically coherent speech, the generated speech often lacks acoustic details, such as high-frequency information. Recovering and enhancing these details typically requires post-processing, like a diffusion model, which significantly increases the model's latency."

## 层级建模方案

为解决单一 token 类型的局限,研究者提出两种层级策略:

### 策略一: 串联 (Concatenation)
将 semantic 和 acoustic tokens 拼入同一序列,先建模 semantic 再建模 acoustic:
- **AudioLM**: w2v-BERT semantic tokens → SoundStream acoustic tokens
- **优点**: 概念简单,分阶段建模
- **缺点**: 序列极长,建模复杂度高

### 策略二: 混合 (Mixed Tokenizer)
设计单一 tokenizer 同时编码语义和声学信息:
- **SpeechTokenizer**: RVQ 第一层蒸馏 HuBERT 的语义,后续层编码声学残差
- **Mimi**: 单 VQ 模块提取语义 + 额外 RVQ 提取声学
- **FireRedTTS 2**: Whisper encoder (semantic) + acoustic encoder → concat + downsample → 16 层 RVQ,以 12.5Hz 低帧率编码混合信息,dual decoder 分别重建 semantic 和 acoustic features [§2.1]
- **LM-SPT** (Jo et al., 2025): dual encoder + Split RVQ (1 semantic VQ + 7 acoustic RVQ),核心创新是 reconstruction-driven semantic distillation — 不直接对齐 teacher-student 特征,而是用 Whisper ASR encoder 对比原始与语义重建波形的表征差异,绕过帧率对齐约束,在 TTS 下游任务上大幅超越 SpeechTokenizer/Mimi
- **Phonological Tokenizer** (Onda et al., 2026): 从 phonetic token (WavLM k-means) 出发,通过 differentiable k-means + ASR/resynthesis 多目标微调 (alpha 权重调控) 注入韵律信息,同时用 speaker embedding 条件化 vocoder 抑制 speaker identity。保持单码本 (2000 entries, 50 tok/s),仅需 44h 额外数据。在 ER (+10pp)、VC (UTMOS/SpkSim best)、speechLM 续写 (GenPPL/UTMOS best) 上全面优于 SpeechTokenizer 和 WavTokenizer [Table 2-4]。详见 [[论文笔记/PhonologicalTokenizer|Phonological Tokenizer]]。
- **优点**: 统一框架,无需串联
- **缺点**: 设计复杂,仍在探索中

## Paralinguistic Tokens

Survey 特别指出第三类 "副语言 token",弥补 semantic tokens 的表现力缺陷:
- **pGSLM**: 在 HuBERT semantic tokens 基础上添加 F0 (基频) 和 unit duration 作为副语言 tokens,用 multi-stream transformer 分别预测
- **SPIRIT-LM**: 添加 pitch tokens 和 style tokens 补充 HuBERT semantic tokens
- 这些副语言 tokens 让 SpeechLM 在不牺牲语义的前提下捕获表现力

## 在 Speech LM 中的角色

Token 类型的选择直接决定 SpeechLM 的能力侧重:
- **多数 SpeechLM 选择 semantic tokens** (GSLM, TWIST, SpeechGPT, AudioPaLM, OmniFlatten, SLAM-Omni): 语义理解是口语交互的核心
- **Codec-focused 系统选择 acoustic tokens** (VioLA, Parrot): 侧重高保真生成
- **前沿系统转向 mixed tokens** (Moshi, SpeechGPT-Gen): 兼顾理解和保真度
- **Speech editing 场景验证 semantic tokens 优势**: [[论文笔记/EditContentPreserveAcoustics|Edit Content, Preserve Acoustics]] (Ren et al., ICME 2026) 在 text-based speech editing 中系统性对比 semantic space vs acoustic space editing。在 semantic token 空间做 PSM infilling + Flow Matching 渲染,WER 全面优于 acoustic token AR baseline (VoiceCraft, 4.97% vs 12.94% on Insertion) 和 NAR baseline (FluentSpeech),且 speaker similarity 不受影响 (timbre 由 frozen FM decoder 统一重建)。实验证据支持: 当任务要求"改内容不改风格"时,semantic space 的 content-style 解耦是结构性优势。

## 关键论文

- GSLM (Lakhotia et al., 2021): 首次对比 3 种 semantic tokenizer (CPC, wav2vec2, HuBERT)
- AudioLM (Borsos et al., 2023): 提出 semantic → acoustic 层级生成框架
- SpeechTokenizer (Zhang et al., ICLR 2024): RVQ 第一层蒸馏 HuBERT 实现混合
- pGSLM (Kharitonov et al., 2022): 引入 paralinguistic tokens (F0, duration)
- SPIRIT-LM (Nguyen et al., 2024): pitch/style tokens 补充语义

## 相关概念

- [[SpeechTokenizer]]: 产生这两类 token 的模块
- [[ResidualVectorQuantization]]: acoustic tokens 的核心量化方法
- [[SpeechLanguageModel]]: 消费这些 token 的模型框架
- [[CodecLanguageModel]]: 专门建模 acoustic (codec) tokens 的 LM 范式

## Survey 核心论点: 二分法不足 [Mousavi et al. 2025, §1-2]

Mousavi et al. (2025) 明确指出传统 semantic vs acoustic 二分法的三个局限:

1. **边界模糊**: "Acoustic tokenizers can capture semantic information [and] semantic tokenizers have been effectively used in generative tasks" [§1] — 两者能力越来越重叠
2. **忽略关键架构差异**: 二分法无法区分 CNN vs Transformer encoder、waveform vs time-frequency 表征等
3. **忽略实用维度**: 流式能力、自适应比特率等部署关键特性不在二分法视野内

**Survey 提出的替代方案**: 五轴精细化 taxonomy (详见 [[AudioTokenizerTaxonomy]]):
- Encoder-Decoder 架构 x 量化方法 x 训练范式 x 目标领域 x 流式能力

**关于 "semantic" 的术语澄清** [§1, footnote 1]: Survey 特别指出 "semantic" 在语音上下文中的含义与语言学不同 — 所谓 semantic tokens 实际上更准确地应描述为 "phonetic units",通常不携带真正的语义内容。Survey 沿用 "semantic" 术语以保持与文献一致。

### SLM 评估: 无全能 tokenizer [Table 10]

Survey 在 SALMon benchmark 上对比各类 tokenizer:

| Tokenizer | 类型 | sBLIMP (语义) | sWUGGY (语义) | Gender (声学) | Spk (声学) |
|-----------|------|-------------|-------------|-------------|----------|
| HuBERT 25Hz | semantic | **60.89** | **70.51** | 69.50 | 69.00 |
| DWavL-S-16 (6Q) | semantic | 53.96 | 69.10 | **92.00** | **86.50** |
| ST-S-16* (8Q) | mixed | 52.75 | 63.46 | 67.00 | 65.50 |
| Mimi-S-24* (8Q) | mixed | 60.17 | 67.57 | 77.00 | 76.00 |
| Enc-SMA-24 (8Q) | acoustic | 51.14 | 51.29 | 70.50 | 65.00 |
| DAC-SMA-16 (8Q) | acoustic | 51.51 | 50.73 | 81.00 | 77.00 |

**关键发现**:
- HuBERT 在语义任务上保持最强; WavLM 在声学一致性上最强
- Mimi* (语义加权版) 在语义任务上接近 HuBERT 同时声学也不错 → 混合路线最有前景
- 纯 acoustic tokenizer (EnCodec, DAC) 在语义任务上几乎随机 (~50%)
- **没有任何 tokenizer 在 semantic-acoustic alignment 上取得实质性成果** → 联合建模仍是开放挑战

## 监督式 Semantic Tokens

CosyVoice (Du et al., 2024) 开创了第四类路线: **监督式 semantic tokens**。不同于 HuBERT 的自监督学习,CosyVoice 的 S3 tokenizer 直接在 ASR encoder 中插入 VQ 层,通过 ASR loss 监督训练,使 token 显式编码语义信息且与文本对齐。实验证明 S3 tokens 在 TTS 任务上的内容一致性 (WER) 远优于 HuBERT semantic tokens 和 EnCodec acoustic tokens [CosyVoice Table 7]。后续 CosyVoice 2/3 继承并扩展了这一路线。

### 理解模型 Encoder 做语义分支 [OmniCodec, 2026]

[[论文笔记/OmniCodec|OmniCodec]] (Hu et al., 2026) 扩展了监督式 semantic tokens 路线: 不使用 ASR encoder (CosyVoice S3) 或 SSL 模型 (WavLM/HuBERT),而是直接用预训练多模态理解模型的 audio encoder (Qwen3-Omni-AuT-Encoder, 2000 万小时监督数据训练) 作为 codec 语义分支的输入。论文声称这是首次展示监督式理解模型 encoder 可替代 SSL 模型做 codec 语义监督 [OmniCodec §1]。优势: 天然覆盖 speech/music/general sound 全域 (Qwen3-Omni 训练数据跨域),而 WavLM/HuBERT 主要针对 speech。劣势: speech 域 PPL 仍不如 WavLM-based Mimi (10.02 vs 8.73),论文作者归因于 WavLM BERT 架构在 phonetic details 上更优 [OmniCodec §3.3, Table 4]。

## 演进

Mel spectrogram (连续, 传统 TTS) → VQ-VAE acoustic tokens (2019) → HuBERT semantic tokens (2021) → semantic + acoustic 层级 (AudioLM, 2022) → paralinguistic tokens 补充 (pGSLM, 2022) → **监督式 semantic tokens (CosyVoice, 2024)** → mixed tokenizer (SpeechTokenizer, 2024) → 统一框架 (Mimi/Moshi, 2024) → 五轴精细化 taxonomy 取代二分法 (Mousavi et al., 2025) → 理解模型 encoder 做语义分支 (OmniCodec, 2026)

### WavLM 中层单码本路线 [WavSLM, 2026]

[[论文笔记/WavSLM|WavSLM]] (Della Libera et al., 2026) 提供了一种新的 mixed token 路线: 用 WavLM 第 6 层 (中层) 特征经 FocalCodec-Stream 量化为单码本 discrete tokens。论文实验证明这种 mid-level SSL feature 的单码本 token 在 SALMon 声学一致性 (Speaker 88.5, Gender 90.5) 和 ZeroSpeech 语义任务上均表现竞争性,支持"中层 SSL 表征天然兼顾语义和声学"的假设 [WavSLM Table 1]。
