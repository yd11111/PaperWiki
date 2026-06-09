---
type: paper
tier: deep
title: "FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs"
arxiv_id: "2407.04051"
source: "Sources/FunAudioLLM.pdf"
authors: [Tongyi SpeechTeam, Alibaba Group]
year: 2024
venue: "arXiv"
tags: [speech-LM, ASR, TTS, SER, AED, multilingual, zero-shot, voice-cloning, instruction-following, supervised-token, flow-matching, open-source]
concepts: ["[[SpeechTokenizer]]", "[[ConditionalFlowMatching]]", "[[LLM-basedTTS]]", "[[Self-SupervisedSpeechRepresentation]]", "[[AudioUnderstanding]]", "[[EmotionControlinTTS]]", "[[SpeakerEmbedding]]", "[[Instruction-GuidedSpeechSynthesis]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]", "[[模型库/SenseVoice|SenseVoice]]", "[[模型库/Whisper|Whisper]]", "[[模型库/HuBERT|HuBERT]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeechTokenizer]], [[ConditionalFlowMatching]], [[LLM-basedTTS]], [[SpeakerEmbedding]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review), [[EmotionControlinTTS]](pending-review), [[Instruction-GuidedSpeechSynthesis]](pending-review) | 未命中但可能相关: 无
>
> **Speech Tokenizer**: FunAudioLLM 提出 S^3 (Supervised Semantic Speech) tokenizer,即 CosyVoice 系列的核心 tokenizer。KB 已记录此 tokenizer 基于 SenseVoice-Large ASR encoder 第 6 层后插入 VQ (单码本, 4096 entries)。本论文是 S^3 tokenizer 的原始出处,详细描述了其设计动机和架构。
>
> **Conditional Flow Matching**: CosyVoice 使用 OT-CFM 将 speech tokens 转为 mel spectrogram。KB 记录 CosyVoice 是最早在 LLM-TTS 中采用 CFM 替代 DDPM 的系统之一。本论文描述了完整的 flow matching + HiFTNet vocoder pipeline。
>
> **LLM-based TTS**: CosyVoice 是 LLM-based TTS 的重要系统,采用 LLM + OT-CFM 的 coarse-to-fine 架构。本论文提供了完整的系统描述,包括 CosyVoice-base/instruct/sft 三个变体。
>
> **Speaker Embedding**: CosyVoice 使用 x-vector 作为 speaker embedding,在 flow matching 阶段作为条件。本论文详述了 speaker embedding 在 zero-shot in-context learning 和 cross-lingual voice cloning 中的不同使用方式。

> [!summary] 速查
> - **一句话**: 阿里巴巴通义语音的双基座框架 — SenseVoice (语音理解: ASR/SER/AED/LID,5x faster than Whisper-small) + CosyVoice (语音生成: 零样本多语言 TTS + 指令跟随),通过 S^3 supervised semantic tokenizer 连接理解与生成 [§Abstract]
> - **路线**: 理解: Speech → Feature Extractor → SAN-M Encoder (Small) / Transformer Encoder-Decoder (Large) → [concat(e_LID, e_SER, e_AEC, e_ITN, X_speech)] → CTC/CE Loss → ASR+SER+AED+LID [§2.2, Fig 2]; 生成: Text → Text Tokenizer → [S, x-vec, text, speech_tokens] → AR Transformer LM → Speech Tokens → OT-CFM (+speaker emb +ref mel) → Mel → HiFTNet → Waveform [§2.4, Fig 4]
> - **指标**: SenseVoice-S: RTF 0.007, 5x faster than Whisper-small, 15x faster than Whisper-large [Table 7]; ASR: CER 2.09 AISHELL-1 / 7.68 CommonVoice zh (best) [Table 6]; SER: WA 96.0/93.2/73.9 CREMA-D/ESD/IEMOCAP (SenseVoice-L best or near-best) [Table 8]; CosyVoice: WER 2.89±0.18 (LibriTTS) / CER 3.82±0.24 (AISHELL-3), SS 74.30/81.58 [Table 10-11]
> - **可借鉴**: (1) Supervised semantic speech tokenizer (S^3): 在 ASR encoder 第 6 层后插入 VQ,用 ASR loss 监督,tokens 天然编码语义+副语言信息 (50Hz → single codebook 4096) [§2.3, Fig 3]; (2) SenseVoice-Small 的非自回归 multi-task 设计: 4 个 task embedding prepend + SAN-M encoder + CTC,实现极低延迟 [§2.2]; (3) Cross-lingual voice cloning: 跨语言时省略 prompt text 和 speech tokens,仅用 speaker embedding + LID,防止源语言韵律泄露 [§2.4.3, Fig 5]; (4) Rich transcription: SER/AED/LID/ITN 全部作为伪标签 (150M AED + 30M SER entries) 自动标注训练数据 [§3.1]
> - **局限**: (1) SenseVoice 不支持 streaming [§6]; (2) CosyVoice 仅支持 5 语言 [§6]; (3) 情感控制依赖显式指令,不能从文本语义推断情感 [§6]; (4) 唱歌任务表现差 [§6]; (5) SenseVoice 和 CosyVoice 未端到端联合训练,pipeline 可能引入错误传播 [§6]

## 核心问题

FunAudioLLM 要解决的问题: 构建一个**完整的语音交互框架**,同时覆盖语音理解 (what the user said + how they said it) 和语音生成 (natural, expressive, controllable speech output) [§1] [论文原文]。

当时 (2024 年中) 的痛点 [§1]:
1. ASR 系统 (如 Whisper) 虽然准确但**丢失副语言信息** (情感、语气、非语言事件) [agent 解读]
2. TTS 系统虽然可以合成高质量语音但**缺乏灵活的控制性** (零样本 + 跨语言 + 指令跟随) [论文原文]
3. 理解和生成模型通常分开开发,缺乏统一框架 [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### Part I: SenseVoice — 语音理解

#### SenseVoice-Small [§2.2, Fig 2 上]

非自回归 encoder-only 架构,支持 ASR + SER + AED + LID 四任务 [论文原文]:

1. **Feature Extractor**: 80-dim log-mel filter bank → stack + 6x downsample → 映射到 encoder dim D [论文原文]
2. **Task Embedding**: 4 个 special token embeddings (e_LID, e_SER, e_AEC, e_ITN/NoITN) prepend 到语音特征 [论文原文]
   - X = concat(e_LID, e_SER, e_AEC, e_ITN, X_speech), X ∈ R^{(T+4)×D} [Eq.1]
3. **SAN-M Encoder**: memory-equipped self-attention network (Gao et al., 2020) [论文原文]
4. **Output**: P = Softmax(Linear_{D→|V'|}(Encoder(X))), 统一词表 V' 包含 ASR tokens + 其他任务标签 [Eq.2]
5. **Training**: ASR 用 CTC loss; LID/SER/AEC 用 cross-entropy loss [论文原文]

**关键设计**: LID token 在训练时以 0.8 概率替换为 ground truth language token,使模型可以既预测语言又接受指定语言 [§2.2] [论文原文]

支持语言: Chinese, English, Cantonese, Japanese, Korean [§2.2]

#### SenseVoice-Large [§2.2, Fig 2 下]

自回归 encoder-decoder 架构,支持 50+ 语言 [论文原文]:
- 与 Whisper 类似,通过 start prompts (SOS, LID, SER, AED, ASR) 指定任务 [论文原文]
- 可预测: 语言、情感、音频事件、带时间戳的转录 [论文原文]
- 优势: 转录精度更高,支持更多语言 [§2.2]

### Part II: S^3 Supervised Semantic Speech Tokenizer [§2.3, Fig 3]

**动机**: 评估了三类 speech tokenizer [§2.3] [论文原文]:
1. RVQ-based: SoundStream, EnCodec, FunCodec — 无监督,语义关联弱
2. Multi-grouped VQ: HiFi-Codec — 无监督
3. Semantic: HuBERT — 自监督,语义信息有但不稳定

**核心洞察**: 上述 tokenizer 都是无监督/自监督训练,"their association to semantic content is often tenuous, contributing to an unstable synthesis process" [§2.3] [论文原文]

**S^3 设计**:
- 基础: pre-trained SenseVoice-Large 模型 [论文原文]
- 在 Encoder_1 (前 6 层) 后插入 Vector Quantizer [Fig 3] [论文原文]
- VQ: 单码本 (single codebook), 4096 entries [§2.3] [论文原文]
- 频率: 50 Hz → 经下采样后用于 TTS [论文原文]
- 训练目标: 最小化 rich transcription 的识别错误,使 token 天然携带语义+副语言信息 [论文原文]
- 加入 positional embedding post-quantization 增强时序信息 [§2.3] [论文原文]

**关键优势** [§2.3]: "Since the speech tokenizer is trained to minimize the recognition errors of rich text in an end-to-end manner, the extracted tokens have a strong semantic relationship to textual and paralinguistic information" [论文原文]

**S^3 语义保留评估** [Table 9]: 在 Common Voice 上,S^3 tokens 的 WER/CER 接近甚至超越 Whisper-Large V3 (common_voice_zh-CN: S^3 12.24 vs Whisper 12.82 w/o lid) [论文原文]

### Part III: CosyVoice — 语音生成 [§2.4]

#### System Overview [§2.4.1, Fig 4]

三个组件 [论文原文]:
1. **AR Transformer LM**: 生成 speech tokens (S^3 tokens)
2. **OT-CFM (Flow Matching)**: speech tokens → mel spectrogram,用 convolutional Transformer U-Net (Matcha-TTS) [论文原文]
3. **HiFTNet Vocoder**: mel → waveform,已修改支持 streaming [论文原文]

#### Model Training [§2.4.2]

- **AR LM**: teacher-forcing,text + left-shifted speech tokens → predict next speech token [论文原文]
- **Flow Matching**: 学习 P(S|X, v, S_ref),X=speech tokens, v=speaker embedding, S/S_ref=target/reference mel [论文原文]
  - OT-ODE: 5-10 iterations 即可生成满意的 mel [论文原文]
  - CFG: mask out 70-100% proceeding feature conditions 增强 in-context learning [论文原文]
- **Vocoder**: HiFTNet (harmonic-plus-noise filter + iSTFT) [论文原文]

#### Zero-shot In-context Learning [§2.4.3, Fig 5a]

- 同语言: prompt speech tokens + prompt text → merge → LM 续写 → speech tokens [论文原文]
- 跨语言: **省略 prompt text 和 prompt speech tokens**,仅用 speaker embedding + LID → 防止源语言韵律泄露 [§2.4.3, Fig 5b] [论文原文]

#### Instruction Fine-tuning (CosyVoice-instruct) [§2.4.4]

支持通过自然语言指令控制 [§2.4.4, Table 3] [论文原文]:
- **Speaker Identity**: persona descriptions (e.g., "Selene 'Moonshade' is a mysterious, elegant dancer")
- **Speaking Style**: "A happy girl with high tone and quick speech"
- **Paralinguistics**: [laughter], [breath], <strong>emphasis</strong>

**训练数据** [Table 5]: Speaker Identity 101h + Speaking Style 407h + Paralinguistics 48h [论文原文]

### Part IV: 三个开源模型 [§2.4]

| 模型 | 特点 | 参数 |
| --- | --- | --- |
| CosyVoice-base-300M | speaker identity, zero-shot, cross-lingual | 300M |
| CosyVoice-instruct-300M | emotion, style, paralinguistics via instruction | 300M |
| CosyVoice-sft-300M | 7 speakers fine-tuned, ready to deploy | 300M |

**训练数据** [Table 4]: ZH 130Kh + EN 30Kh + Yue 5Kh + JP 4.6Kh + KO 2.2Kh, 总计 ~172Kh [论文原文]

数据收集 pipeline [§3.2]: speech detection → SNR estimation → speaker diarization → separation → SenseVoice-Large + Paraformer pseudo labeling → force-alignment refinement [论文原文]

## 实验

### ASR — SenseVoice [Table 6]

| Test Set | Whisper-S | Whisper-L-V3 | SenseVoice-S | SenseVoice-L | 出处 |
| --- | --- | --- | --- | --- | --- |
| AISHELL-1 | 10.04 | 5.14 | 2.96 | **2.09** | [Table 6] |
| CommonVoice zh-CN | 19.60 | 12.55 | 10.78 | **7.68** | [Table 6] |
| LibriSpeech clean | 3.13 | **1.82** | 3.15 | 2.57 | [Table 6] |
| CommonVoice 5-lang avg | 20.68 | 9.66 | 10.56 | **7.57** | [Table 6] |

### Inference Speed [Table 7]

| Model | Framework | Params | RTF | 10s Latency (ms) |
| --- | --- | --- | --- | --- |
| Whisper-S | AR | 224M | 0.042 | 518 |
| SenseVoice-S | **NAR** | 234M | **0.007** | **70** |
| SenseVoice-L | AR | 1587M | 0.110 | 1623 |

SenseVoice-S 5x faster than Whisper-small, 15x faster than Whisper-large [§4.1] [论文原文]

### SER — SenseVoice [Table 8]

SenseVoice-Large 在 7 个 SER benchmark 上几乎全部达到最佳 WA [Table 8]: CREMA-D 96.0, ESD 93.2, IEMOCAP 73.9 [论文原文]

### CosyVoice Generation Quality [Tables 10-11]

| Dataset | WER/CER (%) | SS | 出处 |
| --- | --- | --- | --- |
| LibriTTS (EN) | 2.89±0.18 | 74.30±0.15 | [Table 10] |
| AISHELL-3 (ZH) | 3.82±0.24 | 81.58±0.16 | [Table 11] |

5x re-ranking 进一步降低: LibriTTS WER 1.51, AISHELL-3 CER 1.84 [Tables 10-11] [论文原文]

### Emotion Controllability [Table 12]

CosyVoice-instruct vs CosyVoice-base 在 6 种情感上的准确率 [论文原文]:

| Emotion | CosyVoice-base | CosyVoice-instruct | 出处 |
| --- | --- | --- | --- |
| Happy | 1.00±0.00 | 1.00±0.00 | [Table 12] |
| Sad | 0.45±0.05 | **0.98±0.02** | [Table 12] |
| Angry | 0.59±0.03 | **0.83±0.04** | [Table 12] |
| Surprised | 0.26±0.02 | **0.64±0.03** | [Table 12] |

### Applications [§5]

论文展示了四个应用场景 [§5]:
1. **Speech-to-Speech Translation**: SenseVoice → LLM → CosyVoice (cross-lingual clone) [Fig 10]
2. **Emotional Voice Chat**: SenseVoice (SER+AED) → LLM (style description) → CosyVoice (emotional) [Fig 11]
3. **Interactive Podcast**: SenseVoice + Multi-Agent LLM + CosyVoice [Fig 12]
4. **Expressive Audiobook**: LLM (narrative/character/sentiment analysis) → CosyVoice [Fig 13]

## 局限性

论文自述 [§6] [论文原文]:
1. **SenseVoice**: ASR 在低资源语言上性能低; 不支持 streaming 转录
2. **CosyVoice**: 仅支持 5 语言; 不能从文本语义自动推断情感 (需显式指令); 唱歌表现差; 情感变化时保持原始音色困难
3. **框架级**: SenseVoice 和 CosyVoice 未端到端训练,pipeline 可能引入错误传播

## 点评

FunAudioLLM 的核心贡献是**系统级的**: 它将语音理解 (SenseVoice) 和生成 (CosyVoice) 整合为一个可组合的框架,并通过 S^3 tokenizer 作为桥梁连接两端 [agent 解读]。

**SenseVoice 的非自回归设计** 是一个重要的工程选择: 通过 CTC + SAN-M encoder 实现 NAR,使推理速度极快 (RTF 0.007) [Table 7],这对实时语音交互至关重要 [agent 解读]。同时,4 个 task embeddings 的 multi-task 设计使单一模型同时输出 ASR + SER + AED + LID 结果,避免了多模型串联的延迟累加 [agent 解读]。

**S^3 tokenizer 的核心创新**在于用 ASR 监督信号替代无监督/自监督训练。论文的实验 [Table 9] 证明 S^3 tokens 在 Common Voice 上的语义保留能力接近 Whisper-Large V3,但 codebook 仅有 4096 entries (vs Whisper 的连续表征) [agent 解读]。这个 tokenizer 后来成为整个 CosyVoice 系列 (1→2→3) 的基石。

**CosyVoice 的 cross-lingual 设计** [§2.4.3, Fig 5b] 特别巧妙: 跨语言时省略 prompt 的 text 和 speech tokens,仅保留 speaker embedding + LID,从根本上阻断源语言韵律信息泄露 [论文原文]。这体现了对 tokenizer 信息分布的深刻理解 [agent 解读]。

**历史定位**: 这是阿里通义语音团队的 "统一框架宣言",虽然 SenseVoice 和 CosyVoice 各自独立也有完整论文,但 FunAudioLLM 展示了两者如何组合实现更复杂的应用 (S2ST, voice chat, podcast, audiobook)。后续 CosyVoice 2 (streaming) 和 CosyVoice 3 (scaling + DiffRO) 都建立在这个基础之上 [agent 解读]。

## 可复用的 idea

1. **Supervised semantic tokenizer (S^3)**: 在 ASR encoder 中间插入 VQ,用 ASR loss 端到端训练,获得天然编码语义+副语言的 speech tokens [§2.3]
2. **非自回归 multi-task speech understanding**: 4 个 task embedding prepend + CTC,单模型极低延迟完成 ASR+SER+AED+LID [§2.2]
3. **Cross-lingual voice cloning 的韵律隔离**: 跨语言时省略 prompt text/tokens,仅用 speaker embedding + LID [§2.4.3]
4. **Pseudo label pipeline for rich transcription**: 用开源 AED/SER 模型自动标注 150M+ entries 的训练数据 [§3.1]
5. **CosyVoice-instruct 的三维度指令控制**: speaker identity + speaking style + paralinguistics,各有独立的 instruction 格式 [§2.4.4, Table 3]

---

检索命中: [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review), [[EmotionControlinTTS]](pending-review), [[Instruction-GuidedSpeechSynthesis]](pending-review) | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
