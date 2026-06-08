---
type: paper
tier: deep
title: "MiMo-Audio: Audio Language Models are Few-Shot Learners"
arxiv_id: "2512.23808"
source: "Sources/MiMo-Audio.pdf"
authors: [Xiaomi LLM-Core Team]
year: 2025
venue: "arXiv"
tags: [speech-LM, few-shot, audio-understanding, unified-model, emergent-ability, speech-continuation, patch-encoder-decoder, RVQ-tokenizer, speech-intelligence, in-context-learning]
concepts: ["[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[LLM-basedTTS]]", "[[AudioUnderstanding]]", "[[ResidualVectorQuantization]]", "[[SemanticvsAcousticTokens]]", "[[CodecTrainingObjectives]]"]
models: ["[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[LLM-basedTTS]], [[ConditionalFlowMatching]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[LLM-basedTTS]]✓, [[ConditionalFlowMatching]]✓ | 过滤: [[AudioUnderstanding]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review) | 未命中但可能相关: [[CodecLanguageModel]](confirmed but less central)
>
> **Speech Language Model**: MiMo-Audio 属于 SpeechLM 的 "continued pre-training from TextLM" 路线 (从 MiMo-7B-Base 初始化),与 Moshi (从 Helium 初始化) 和 SPIRIT-LM (从 Llama 初始化) 属同族。独特之处在于数据规模 (100M+ 小时,比 Kimi-Audio/Step-Audio 大一个量级) 和对 "lossless compression" 的执着——完全绕过 semantic-only tokenizer,坚持高保真 RVQ acoustic tokens 流入 LLM。KB 中记录的演进线 GSLM → AudioLM → TWIST → Moshi → Mini-Omni 以任务或架构创新为主,MiMo-Audio 则主要靠 **规模** 解锁 emergent ability,定位更接近 GPT-3 在文本领域的角色。
>
> **Speech Tokenizer**: MiMo-Audio-Tokenizer 是 Mixed Objective Tokenizer 的新成员,但路线与 SpeechTokenizer/Mimi/LM-SPT 不同: 不依赖任何预训练 SSL/ASR 模型做语义蒸馏,而是从零训练一个 1.2B Transformer encoder + RVQ + decoder + LLM 联合体,通过 A2T (audio-to-text) loss 和 reconstruction loss 同时获得语义和声学能力 [§2.1.2]。KB 中 SpeechTokenizer 用 HuBERT 蒸馏、Mimi 用 WavLM 蒸馏、X-Codec 用双流 encoder,均依赖外部 semantic model。MiMo-Audio-Tokenizer 的 "from scratch + scale up" 策略是对 "必须借助 SSL model" 假设的挑战。
>
> **LLM-based TTS**: MiMo-Audio 不是专门的 TTS 系统,但其 speech generation 能力 (TTS、instruct-TTS、speech continuation) 来自 LLM next-token prediction。与 StepAudio 2.5 的 "audio tokens as new language" 理念一致,但 MiMo-Audio 更进一步: 通过 text-guided interleaving 策略 (5:5 text:speech patch ratio) 让 LLM 同时生成文本和语音,而非仅生成 audio tokens [§3.2.2]。InstructTTSEval 结果超越 GPT-4o-mini-tts [Table 9],验证了 "通用 SpeechLM 的 TTS 能力可以接近 TTS 专用模型" 的假设。

> [!summary] 速查
> - **一句话**: 通过 100M+ 小时无损 speech token 预训练,首次在语音领域观察到 GPT-3 式 emergent few-shot learning,base 模型无需微调即可执行 voice conversion/style transfer/speech translation 等从未训练过的任务 [§1, §3.4]
> - **路线**: 24kHz waveform → MiMo-Audio-Tokenizer (1.2B, 25Hz, 8-layer RVQ, 200 tokens/s) → Patch Encoder (6L Transformer, 4帧→1 patch, 6.25Hz) → MiMo-7B LLM backbone → Patch Decoder (16L Transformer, 1 patch→4帧, delay pattern) → Tokenizer decoder+vocoder → waveform [§2, Fig 3]
> - **指标**: Base: SpeechMMLU S2S 69.1 / MMAU 66.0 (开源 SOTA) [Table 6]; modality gap 仅 3.4 (vs Step-Audio2 22.3) [§3.4]; Instruct: MMAU 74.90 / MMSU 61.70 / MMAU-Pro 53.35 / MMAR 63.60 (均开源 SOTA,接近 Gemini 2.5 Flash) [Table 8]; InstructTTSEval-EN 72.59 / ZH 70.52 (超越 GPT-4o-mini-tts) [Table 9]; BigBench Audio S2T 72.90 (仅次于 GPT-4o) [Table 8]
> - **可借鉴**: (1) Patch encoder-decoder 架构将 200 token/s 降至 6.25 Hz LLM 输入,解决 speech-text 长度失配 [§2.2]; (2) Tokenizer 从零联合训练 encoder + LLM 的 A2T loss 同时获得语义和声学能力,无需 SSL 蒸馏 [§2.1.2]; (3) Text-guided interleaving (5:5 ratio) 同时生成 text 和 speech tokens 提升生成质量 [§3.2.2]; (4) Loss weights 按 RVQ 层指数衰减 (100-12-8-6-4-2-2-1-1) 确保高层语义 > 低层声学 [Table 3]; (5) Layer-3 hidden states + final-layer skip connection 在 tokenizer encoder 中缓解语义-声学冲突 [§2.1.1]
> - **局限**: 训练数据 100M+ 小时,普通团队无法复现 [§3.1]; 对话语音质量不稳定 (timbre discontinuity, mispronunciation) [§6]; thinking 机制在 sound/music 任务上引入幻觉导致性能下降 [§6]; ASR WER 3.50 (LibriSpeech) 弱于 Step-Audio2 1.87 和 Kimi-Audio 2.13 [Table 9]; 模型开源但训练数据未开源

## 核心问题

MiMo-Audio 试图回答的核心问题: **语音领域是否存在类似 GPT-3 的 "规模即涌现" 现象?** [§1]

作者的假设链 [论文原文]:
1. GPT-3 证明了在文本领域,足够规模的 next-token prediction 预训练可以涌现出 few-shot learning 能力 [§1]
2. 要将此范式复制到语音领域,需要解决两个前置条件: (a) 语音信号的**无损流通** (lossless flow)——不能使用丢弃副语言信息的 semantic-only tokens; (b) 数据**规模**——需要比现有最大开源模型大一个量级的语料 [§1]
3. 满足这两个条件后,模型将自发涌现出 few-shot learning 能力,无需为每个下游任务微调 [§1]

**为什么要强调 "lossless"?** [论文原文] 作者认为,使用 semantic tokens (如 GLM-4-Voice 的 VQ-Whisper) 或 ASR-based tokens (如 Kimi-Audio 的 Whisper encoder) 会丢弃副语言信息,这限制了模型在 voice conversion / style transfer 等需要保留声学细节的任务上的泛化能力 [§1]。这与 KB 中 [[SemanticvsAcousticTokens]] 记录的核心 trade-off 直接相关: MiMo-Audio 选择站在 acoustic-preserving 一侧。

**实验验证**: 训练过程中确实观察到了 "phase transition" 式的涌现——在训练量达到约 0.7T tokens 前,SpeechMMLU S2S / voice conversion / speech translation 的性能几乎为零; 超过这个阈值后性能急剧跃升 [§3.4, Fig 1]。这是目前语音领域最清晰的 emergent ability 经验证据。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MiMo-Audio 的架构分为三个层次 [§2, Fig 3]:

**1. MiMo-Audio-Tokenizer** (1.2B 参数) [§2.1]:
- Transformer encoder (32 层, dim 1280, bidirectional) + 2x downsampling → 25Hz 连续表征
- RVQ 离散化: 20 层 RVQ (前 2 层 codebook 1024, 后 18 层 128),下游 LLM 只用前 8 层 → 200 tokens/s, ~1.55 kbps [§2.1.1, Table 1]
- Transformer decoder (causal, 支持流式生成) + Vocos-based vocoder (Transformer backbone, 16 层, sliding window attention)
- **关键设计**: encoder 的 layer-3 hidden states 通过 element-wise summation 加到 final-layer output,目的是在浅层保留更多语义信息以缓解语义-声学冲突 [§2.1.1] [论文原文]

**2. Patch Encoder-Decoder** (连接 tokenizer 和 LLM) [§2.2]:
- Patch Encoder: 将 4 个连续 RVQ 帧 (25Hz) 聚合为 1 个 patch (6.25Hz)。8 层 RVQ embeddings 逐层查表后相加 → 6 层 Transformer (bidirectional, dim 1024) → 线性投影到 LLM 维度 [§2.2.1, Eq. 11]
- Patch Decoder: 16 层 Transformer (causal, dim 1024),每个 patch 展开为 4 帧 × 8 codebooks。使用 delay pattern [0,1,2,3,4,5,6,7] 让不同 RVQ 层错位生成,避免同时预测所有层 [§2.2.3, Eq. 14-15]

**3. LLM Backbone** (MiMo-7B-Base) [§2.2.2]:
- 36 层 Transformer, dim 4096, context 8192
- 接收 text tokens 和 audio patches 的 interleaved 序列,自回归预测下一个 token 或 patch [§2.2, Eq. 10]

**为什么用 patch 而不是直接输入 RVQ tokens?** [论文原文] 200 tokens/s 的 raw token rate 对 LLM 来说序列太长,10 秒语音就需要 2000 个 LLM position。Patch 将 token rate 降至 6.25Hz (每秒 6.25 个 LLM position),使 8192 context 可覆盖约 20 分钟的纯语音 [§2.2.1]。[agent 解读] 这也是为什么 MiMo-Audio 能做长格式 speech continuation (talk show, debate 等)——context window 足够装下长语音上下文。

### 关键设计选择

**1. Tokenizer: 从零训练 vs 依赖预训练模型** [§2.1]

为什么不用 HuBERT/Whisper 做语义蒸馏? [论文原文] 作者认为现有双流方案 (X-Codec, XY-Tokenizer) 依赖预训练语义模型,导致语义和声学信息来自不同表征空间。MiMo-Audio-Tokenizer 选择从零训练,通过 A2T (audio-to-text next-token prediction) loss 让 encoder 自己学到语义表征,同时 reconstruction loss 保证声学保真 [§2.1.2]。

训练中联合一个 LLM (也从零训练),接收量化后的 audio representation Q̃ 并预测文本 token 序列。这个 LLM 在 tokenizer 训练结束后被丢弃,只是作为语义监督信号的载体 [§2.1.2, Eq. 1] [论文原文]。

**为什么这样做有效?** [agent 解读] A2T loss 相当于隐式地让 encoder 学习 ASR 级别的语义理解,而不需要 HuBERT 那样的预训练 pipeline。Stage 1 的 loss 权重 λ_A2T=10.0 >> λ_recon=1.0 说明作者非常重视语义对齐 [§2.1.2, Eq. 3]。

**2. 两阶段 tokenizer 训练** [§2.1.2]:
- Stage 1: encoder + RVQ + decoder + LLM 全参数训练 (11M+ 小时数据),联合优化 A2T + reconstruction + commitment loss
- Stage 2: 冻结 encoder + RVQ,引入 MPD + MS-STFT 判别器对抗训练 decoder + vocoder,提升波形重建细节

**为什么分两阶段?** [论文原文] Stage 2 的对抗训练会改变梯度分布,如果不冻结 encoder 和 RVQ,可能破坏已学到的语义-声学平衡 [§2.1.2]。

**3. Delay pattern 的设计** [§2.2.3]:

为什么 RVQ 各层不能同时预测? [论文原文] RVQ 不同层之间存在依赖关系 (每层量化上一层的残差),同时预测质量差。Delay pattern [0,1,2,3,4,5,6,7] 让第 r 层的 token 在第 r 步才开始预测,这样预测第 r 层时已经有了前 r-1 层的输出作为条件 [§2.2.3, Eq. 15]。这个设计源自 MusicGen (Copet et al., 2023) [论文原文]。

**4. Loss weights 的非均匀设计** [Table 3]:

Text loss weight 100 >> 第 1 层 RVQ 12 >> 第 8 层 RVQ 1。[agent 解读] 这反映了信息密度的层级: text tokens 每个承载的信息量远大于单个 audio token; RVQ 第 1 层 (粗粒度) 的错误比第 8 层 (细节) 的错误影响更大。

### 训练策略

**预训练: 两阶段渐进** [§3.2]:

**Stage 1: Understanding Only** (2.6T tokens) [§3.2.1]:
- 训练 patch encoder + LLM,不训练 patch decoder
- 四类数据: speech-text interleaved, ASR, audio captioning, text-only
- 只计算 text tokens 的 loss (不计算 audio loss)
- LR: patch encoder 2e-4, LLM 3e-5 (encoder 学习率远高于 LLM,因为 encoder 从零学习而 LLM 只需微调) [论文原文]
- **设计意图**: 先让 LLM 理解语音,再学习生成 [论文原文]

**Stage 2: Understanding + Generation Joint** (5T tokens) [§3.2.2]:
- 训练全部参数 (patch encoder + LLM + patch decoder)
- 七类数据: speech continuation, speech-text interleaved, ASR, TTS, audio captioning, instruction-following TTS, text
- 同时计算 text 和 audio loss
- **Text-guided interleaving**: TTS/continuation 等生成任务中,模型以 5:5 ratio 交替生成 text tokens 和 speech patches,text 全部生成完后再补全剩余 speech [§3.2.2] [论文原文]

**为什么用 text-guided interleaving?** [论文原文] 让 LLM 先 "想好要说什么" (text),再 "说出来" (speech),提升语音生成的语义连贯性 [§3.2.2]。[agent 解读] 这类似于 chain-of-thought,text tokens 充当隐式的规划信号。

**Post-training** (100B tokens) [§4.2]:
- SFT 全参数微调
- 六类数据: ASR, TTS, audio understanding, spoken dialogue, instruct-TTS, text dialogue
- 继续使用 text-guided interleaving 和 Stage 2 的 loss weights
- LR 降低: patch encoder/decoder 5e-5, LLM 1e-5 [Table 3]
- **Thinking (CoT) 数据**: 为 audio understanding 和 generation 任务构建 CoT 数据,但仅在 speech understanding 上有效,sound/music 上反而引入幻觉 [§6] [论文原文]

**MiMo-TTS-7B** (辅助模型) [§4.1.3]: 独立训练的 TTS 模型,7M+ 小时数据,用于将文本对话数据合成为 spoken dialogue 训练数据。不是 MiMo-Audio 的组件,而是 data pipeline 的工具。

## 实验

| 指标 | 本文 (Base) | 最强开源 Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SpeechMMLU S2S | **69.1** | Step-Audio2-mini 51.8 | SpeechMMLU (8549 entries) | [Table 6] |
| SpeechMMLU T2T | 72.5 | Step-Audio2 74.1 | SpeechMMLU | [Table 6] |
| Modality gap (T2T-S2S) | **3.4** | Step-Audio2 22.3 | SpeechMMLU | [§3.4] |
| MMAU Overall (Base) | **66.0** | Step-Audio2 60.3 | MMAU | [Table 6] |

| 指标 | 本文 (Instruct) | 最强开源 Baseline | 闭源参考 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MMAU Overall | **74.90** | Step-Audio2 72.73 | Gemini 2.5 Flash 71.80 | MMAU | [Table 8] |
| MMAU-Pro | 53.35 | Qwen2.5-Omni 52.20 | Gemini 2.5 Flash **59.20** | MMAU-Pro | [Table 8] |
| MMAR | **63.60** | Audio Flamingo 3 58.50 | Gemini 2.5 Flash **65.60** | MMAR | [Table 8] |
| MMSU Overall | **61.70** | Kimi-Audio 59.78 | - | MMSU | [Table 8] |
| BigBench Audio S2T | **72.90** | Step-Audio2 50.90 | GPT-4o 70.20 | BigBench Audio | [Table 8] |
| BigBench Audio S2S | 60.20 | Qwen2.5-Omni 53.60 | GPT-4o **67.20** | BigBench Audio | [Table 8] |
| MultiChallenge S2T | **15.15** | Step-Audio2 13.64 | - | MultiChallenge | [Table 8] |
| InstructTTSEval EN | **72.59** | - | GPT-4o-mini-tts 68.50 | InstructTTSEval | [Table 9] |
| InstructTTSEval ZH | **70.52** | - | GPT-4o-mini-tts 51.07 | InstructTTSEval | [Table 9] |
| SeedTTS ZH WER | 1.96 | Step-Audio2 **2.13** | - | SeedTTS-Eval | [Table 9] |
| ASR LibriSpeech WER | 3.50 | Step-Audio2 **1.87** | - | LibriSpeech test-clean | [Table 9] |

**关键观察**:

1. **Understanding 全面领先**: 在 MMAU/MMSU/MMAR/MMAU-Pro 上均为开源 SOTA,部分超越 Gemini 2.5 Flash [Table 8]
2. **Modality gap 极小**: S2S 与 T2T 的性能差仅 3.4 分 (Step-Audio2 为 22.3),说明 lossless tokenizer + 大规模预训练有效缩小了模态鸿沟 [§3.4]
3. **Speech intelligence 独特**: 唯一一个在 SpeechMMLU 全部四个模态组合 (T2T/S2T/T2S/S2S) 上均 >69 的模型 [Table 6]
4. **ASR 偏弱**: LibriSpeech WER 3.50,明显弱于 Step-Audio2 (1.87) 和 Kimi-Audio (2.13) [Table 9]。[agent 解读] 可能因为 MiMo-Audio 的 tokenizer 是 acoustic-preserving 路线,不如 Whisper-based encoder 在 ASR 上有天然优势
5. **Instruct-TTS 超预期**: 一个通用 SpeechLM 在 InstructTTSEval 上超越 GPT-4o-mini-tts (专用 TTS) [Table 9],说明 100M+ 小时预训练给了模型足够的 "vocal intelligence"

## 局限性

1. **数据规模不可复现**: 100M+ 小时预训练数据,普通团队无法获取或负担计算成本 [§3.1]。论文的核心结论 "scale → emergence" 是否存在更高效的路径未被探索。

2. **ASR 性能弱于专用模型**: WER 3.50 (LibriSpeech) 远弱于 Step-Audio2 (1.87) [Table 9]。[agent 解读] Acoustic-preserving tokenizer 保留了 speaker/emotion 信息但可能增加了 ASR 任务的噪声。

3. **对话语音质量不稳定**: timbre discontinuity、mispronunciation、system prompt 遵循不稳定 [§6]。这些是 SpeechLM 类模型的共性问题,MiMo-Audio 也未解决。

4. **Thinking 机制效果有限**: CoT 仅在 speech understanding 上有效 (+1.18 MMSU),在 sound/music 上引入幻觉导致性能下降 [§6, Table 8]。

5. **Few-shot 评估的局限性**: SpeechMMLU 是自建 benchmark (用 commercial TTS 合成),可能存在 TTS 质量偏差; 16-shot S2S 评估使用自动指标,与人类判断的相关性未验证。

## 点评

MiMo-Audio 的核心贡献不在于架构创新 (patch encoder-decoder 和 delay pattern 均有前例),而在于提供了语音领域 **scaling law 的经验证据**:

1. **Emergent ability 的实证价值**: Fig 1 展示的 "从零到涌现" 曲线是目前语音领域最清晰的 phase transition 证据。这不是 "模型越大越好" 的平凡结论,而是 "存在一个临界点,超过后能力质变" 的强主张 [§3.4]。但需注意,这个结论目前只有单一模型的单次训练过程作为证据,是否在不同架构/tokenizer 下可复现尚未验证。

2. **Lossless vs Semantic 的路线之争**: MiMo-Audio 坚持 acoustic-preserving tokenizer + 规模取胜,与 Kimi-Audio/GLM-4-Voice 的 semantic-first 路线形成对照。结果表明两条路线各有优势: lossless 路线在 speech intelligence (S2S reasoning) 和 generation (voice conversion, style transfer) 上占优,但在 ASR 上不如 semantic 路线 [Table 6, 9]。这提示 **tokenizer 选择决定了模型能力的 "形状" 而非 "大小"**。

3. **Text-guided interleaving 的启发**: 让模型先生成 text 再生成 speech 的策略 [§3.2.2] 与 chain-of-thought 异曲同工。[agent 解读] 这暗示即使在语音生成中,text 仍然是有价值的中间规划信号,纯 speech-to-speech 可能不是最优路径。

4. **与 StepAudio 2.5 的互补视角**: StepAudio 2.5 主张 "共享基座 + 任务特化 post-training",MiMo-Audio 主张 "足够大的基座 + 涌现式泛化"。两者的 post-training 策略差异显著: StepAudio 2.5 对 ASR/TTS/Realtime 做分支特化 (含 RLHF),MiMo-Audio 做统一 SFT (100B tokens)。MiMo-Audio 的 understanding 更强,但 TTS stability/ASR 精度不如 StepAudio 2.5 的特化分支。

## 可复用的 idea

1. **Tokenizer 从零联合训练 A2T + reconstruction**: 如果有足够数据 (10M+ 小时),可以绕过 SSL 预训练依赖,一步到位获得语义+声学 unified tokenizer。A2T loss weight 10x reconstruction loss 是关键 [§2.1.2, Eq. 3]。适用于资源充足的团队从零构建 audio foundation model。

2. **Layer-3 skip connection in tokenizer encoder**: 将浅层 (第 3 层) hidden states 直接加到 final output 缓解深层压缩导致的语义-声学冲突 [§2.1.1]。这是一个低成本的架构 trick,可直接应用于任何 deep encoder tokenizer。

3. **Patch encoder-decoder for RVQ rate reduction**: 4:1 temporal grouping + bidirectional Transformer 将 200 token/s 降至 6.25 Hz LLM input rate [§2.2.1]。对于使用高 token rate codec (>50 Hz) 的 SpeechLM,这种 patch 策略可大幅减少 LLM context 占用。

4. **Text-guided interleaving (5:5 ratio)**: 在 TTS/continuation 等生成任务中,让 LLM 交替生成 text 和 speech tokens (而非先 text 后 speech 或只 speech) [§3.2.2]。可直接集成到任何 unified speech-text LM 的训练中。

5. **RVQ loss weights 指数衰减**: 100 (text) - 12-8-6-4-2-2-1-1 (RVQ 1-8) [Table 3]。为多码本 RVQ-based SpeechLM 提供了一个经验证有效的 loss weighting baseline。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 核心问题清晰 (scale→emergence),方法节覆盖架构/tokenizer/训练三个层次的 WHY |
> | 可信赖 | pass | 关键数字均标注 [§/Table/Fig],覆盖率 >80%;指标名无混淆 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖方法节全部因果解释 |
> | 可定位 | pass | KB 背景对比 SpeechTokenizer/Mimi/GLM-4-Voice/StepAudio2.5,谱系清晰 |
> | 不污染 | pass | 无新建概念页需求;反向更新建议: SpeechLanguageModel +key_paper, SpeechTokenizer +key_paper, AudioUnderstanding +key_paper |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> 详见 `_review/MiMo-Audio-review.yml`
