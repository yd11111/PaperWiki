---
type: paper
tier: deep
title: "UniAudio 2.0: A Unified Audio Language Model with Text-Aligned Factorized Audio Tokenization"
arxiv_id: "2602.04683"
source: "Sources/UniAudio2.0.pdf"
authors: [Dongchao Yang, Yuanyuan Wang, Dading Chong, Songxiang Liu, Xixin Wu, Helen Meng]
year: 2026
venue: "arXiv"
tags: [audio-foundation-model, unified-understanding-generation, audio-tokenizer, factorized-codec, reasoning-tokens, layer-specialization, multi-task, few-shot, zero-shot, speech, sound, music, GRPO, FiLM, auditory-sentences, LLaMA, multi-stage-training]
concepts: ["[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/SpeechTokenizer|Speech Tokenizer]]", "[[概念库/ResidualVectorQuantization|RVQ]]", "[[概念库/SemanticvsAcousticTokens|Semantic vs Acoustic Tokens]]", "[[概念库/LLM-basedTTS|LLM-based TTS]]", "[[概念库/AudioUnderstanding|Audio Understanding]]", "[[概念库/AudioTokenizerTaxonomy|Audio Tokenizer Taxonomy]]", "[[概念库/SpeechFactorization|Speech Factorization]]", "[[概念库/CodecLanguageModel|Codec Language Model]]"]
models: ["[[论文笔记/UniAudio|UniAudio]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[概念库/SpeechLanguageModel|SpeechLanguageModel]], [[概念库/SpeechTokenizer|SpeechTokenizer]], [[概念库/ResidualVectorQuantization|RVQ]], [[概念库/SemanticvsAcousticTokens|SemanticvsAcousticTokens]], [[概念库/LLM-basedTTS|LLM-basedTTS]], [[概念库/SpeechFactorization|SpeechFactorization]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[概念库/SpeechLanguageModel|SpeechLanguageModel]]✓, [[概念库/SpeechTokenizer|SpeechTokenizer]]✓, [[概念库/ResidualVectorQuantization|RVQ]]✓, [[概念库/SemanticvsAcousticTokens|SemanticvsAcousticTokens]]✓, [[概念库/LLM-basedTTS|LLM-basedTTS]]✓, [[概念库/SpeechFactorization|SpeechFactorization]]✓ | 过滤: [[概念库/CodecLanguageModel|CodecLanguageModel]](pending-review), [[概念库/AudioUnderstanding|AudioUnderstanding]](pending-review), [[概念库/AudioTokenizerTaxonomy|AudioTokenizerTaxonomy]](pending-review) | 未命中但可能相关: 无

**[[概念库/SemanticvsAcousticTokens|SemanticvsAcousticTokens]]**: UniAudio 2.0 的 ReasoningCodec 提出了超越 semantic/acoustic 二分法的第三条路: reasoning tokens + reconstruction tokens。与传统 semantic tokens (HuBERT/CosyVoice) 不同,reasoning tokens 被设计为**仅编码 text-level 语义** (5 Hz,极低帧率),刻意排除副语言信息 [§3, footnote 2],再由 reconstruction tokens 编码全部声学信息。这与 SpeechTokenizer/Mimi 的 mixed token 路线(第一层蒸馏语义,后续层编码残差声学)形成鲜明对比: ReasoningCodec 不追求在单一 token 流中混合两类信息,而是用**两个独立分支 + FiLM 耦合**实现信息互补。

**[[概念库/SpeechTokenizer|SpeechTokenizer]]**: ReasoningCodec 在 tokenizer 设计中引入了几个独特选择: (1) reasoning branch 使用 query-based 压缩实现 5 Hz 极低帧率,远低于主流 tokenizer 的 12.5-50 Hz; (2) reconstruction branch 使用 group-wise VQ (1:1:6 分配) 而非标准 RVQ,将 phone/music/sound 语义分别量化; (3) 用预训练 LLM (LLaMA 3.2 3B) 作为 reasoning branch 的 decoder head 并通过 SFT+GRPO 两阶段训练。在 Survey 五轴 taxonomy 下,ReasoningCodec 属于 multi-domain + dual-branch 设计,training paradigm 包含 supervised semantic + RL 强化。

**[[概念库/ResidualVectorQuantization|RVQ]]**: ReasoningCodec 在 reasoning branch 使用 8 层 RVQ (codebook 1024) 量化 query-based 表征; reconstruction branch 使用 group-wise VQ (3 组共 8 层: 1+1+6)。值得注意的是,reasoning tokens 的 RVQ PPL 极低 (speech: 3.68 avg vs XCodec 18.40 vs DAC 90.06) [Table 2],表明 text-aligned representation 对 LM 的建模难度远低于传统 codec tokens。

**[[概念库/LLM-basedTTS|LLM-basedTTS]]**: UniAudio 2.0 在 TTS 上以 3B 参数在 SEED-TTS-Eval 上优于 MiMo-Audio 7B (EN WER 3.63 vs 5.37, ZH WER 2.30 vs 1.93) [Table 5]。与专用 TTS 系统不同,它是统一基础模型的 TTS 能力,通过 multi-task 训练自然获得而非针对 TTS 优化。

**[[概念库/SpeechLanguageModel|SpeechLanguageModel]]**: UniAudio 2.0 是 UniAudio 系列的第三代,从 generation-only (UniAudio 1.0) 和 few-shot adaptation (UniAudio 1.5) 演进到统一 understanding + generation。其 functional layer specialization (lower=understanding, middle=cross-modal, upper=generation) 是对 Moshi/Kimi-Audio 等 "全层均匀处理" 的直接挑战。3B 参数实现的多任务能力 (ASR + TTS + caption + generation across speech/sound/music) 验证了 tokenizer 设计对模型效率的杠杆作用。

**[[概念库/SpeechFactorization|SpeechFactorization]]**: ReasoningCodec 的 reasoning/reconstruction 分解是一种表征层面的因子化,与 NaturalSpeech 3 的 content/prosody/timbre/detail 分解和 Seed-TTS 的 self-distillation 属于不同层次。ReasoningCodec 的分解是 **text-aligned semantics vs everything else**,FiLM 机制让 reconstruction branch 能利用 reasoning tokens 减少冗余 (PPL 从 29.36 降至 23.77) [Table 2]。

> [!summary] 速查
> - **一句话**: 通过 ReasoningCodec (reasoning tokens 5Hz + reconstruction tokens 12.5Hz) 和 functional layer specialization (understanding/cross-modal/generation experts) 构建首个跨 speech/sound/music 的统一理解+生成音频基础模型
> - **路线**: Audio → ReasoningCodec [Reasoning Branch (Whisper+Music encoder → query-based RVQ 5Hz) + Reconstruction Branch (WavLM+Whisper+Music → group-wise VQ 12.5Hz + FiLM)] → Multi-stream 9-channel packing → Unified AR Transformer (3 understanding + 28 cross-modal/LLaMA + 2 generation layers + 4-layer local decoder) → Audio/Text output
> - **指标**: ASR LS-clean 2.71 WER; TTS SEED-EN 3.63 / SEED-ZH 2.30 WER; Codec MUSHRA speech 90.5±2.8 (best); Reasoning PPL 3.68 (vs XCodec 18.40); 1-shot emotion 67.0% EN; DSR 19.4% WER (vs Qwen2.5-Omni 80.6%); MMLU 44.1% (vs LLaMA 3.2 3B 47.6%) [Tables 1-8]
> - **可借鉴**: (1) Reasoning tokens 作为 text-aligned "planning layer" 降低 reconstruction tokens 建模难度 (信息论证明: I(S;R|X,S<t) > 0 → PPL 降低); (2) Layer specialization 避免全层处理 text+audio 导致的 text knowledge forgetting; (3) Auditory sentences 作为统一 multi-task 训练数据构造策略; (4) GRPO 训练 reasoning branch 改善 tokenizer 的理解质量
> - **局限**: Flow-based decoder 多步推理增加延迟 [Appendix C.1]; 仅探索 1B/3B 规模,未验证 scaling law [C.3]; Sound/music 数据量远少于 speech 导致非语音任务弱 [C.4]; 未做 post-training (SFT/RLHF) [C.5]; S2S 对话质量弱于 LLAMA-Omni [Table 7]

## 核心问题

UniAudio 2.0 要解决两个基础性问题 [§1]:

1. **表征问题: 理解和生成需要不同类型的 audio token,如何在统一离散表征中同时满足两者?** 连续表征 (Whisper/WavLM features) 适合理解但不便生成; 离散 codec tokens (EnCodec/DAC) 适合生成但语义抽象不足。现有 mixed tokenizers (SpeechTokenizer, Mimi) 尝试折中但效果有限 [Table 3: 纯 codec token 如 DAC 的 ASR WER 93.2%]。[论文原文]

2. **架构问题: 如何在统一 AR framework 中同时做好 audio understanding 和 generation,而不丢失预训练 text LLM 的知识?** 直接用所有 transformer 层均匀处理 text+audio tokens 是次优的,因为: (i) lossy discrete audio tokens 在所有层传播会限制理解的感知抽象; (ii) text-audio alignment 在所有层强制对齐会导致 text knowledge forgetting [§1]。[论文原文]

**[agent 解读]**: 这两个问题本质上是同一挑战的两个面: 如何让一个模型同时做好语义抽象 (理解) 和声学细节保留 (生成)。ReasoningCodec 在 tokenizer 层面分解问题,layer specialization 在模型层面分解问题,形成两层解耦策略。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UniAudio 2.0 由两个核心组件组成 [§3-4, Fig 1-2]:

**1. ReasoningCodec** (音频 tokenizer): 将音频分解为两类 token:
- **Reasoning tokens r = Tr(x)**: 5 Hz, 8 codebooks (RVQ), 编码 text-aligned high-level 语义,用于理解和"规划"生成 [§3.1]
- **Reconstruction tokens s = Ts(x|r)**: 12.5 Hz, 8 codebooks (group-wise VQ: 1+1+6), 编码 semantic-rich 声学信息,用于波形重建 [§3.2]
- 波形仅从 reconstruction tokens 重建: x̂ = D(s) [§3]
- 两个分支通过 FiLM 耦合: reasoning tokens 通过 γ(R̂)⊙Se + β(R̂) 调制 reconstruction branch 的语义特征 [§3.2, Eq.3]

**2. UniAudio 2.0 模型**: 统一 AR transformer 架构:
- **9-stream multi-modal packing**: 8 audio streams (codebooks) + 1 text stream,每个 time step 仅一种模态 active [§4.2]
- **Layer specialization**: 3 lower layers (understanding expert) + 28 middle layers (cross-modal expert, init from LLaMA 3.2 3B) + 2 upper layers (generation expert) + 4-layer local decoder [§4.3, Table 17]
- **Audio-only computation**: Understanding/generation experts 仅更新 audio positions,text positions 保持不变: H' = H + M_aud ⊙ (f(H) - H) [§4.3, Eq.7]

### 关键设计选择

**设计选择 1: 为什么要有 reasoning tokens?** [§3, §A.3.5]

[论文原文] Reasoning tokens 提供两个互补功能: (1) 作为 language-aligned bottleneck 过滤任务无关声学细节 (recording condition, noise 等),为 LLM 提供更 learnable 的目标; (2) 作为 "planning layer" 降低 reconstruction token 的生成不确定性。

[论文原文] 信息论证明 [§A.3.5]: 引入 R 后 H(St|X,R,S<t) ≤ H(St|X,S<t),减少量 = I(St;R|X,S<t) > 0。直觉: R 提供高级"计划",让模型不再需要从长距离上下文推断 intent,reconstruction token PPL 因此下降。

[论文原文] 与传统 semantic tokens 的关键区别: reasoning tokens 只编码 text-level 语义,刻意排除副语言信息 (prosody, pitch, timbre) [footnote 2]。因此 reasoning tokens 的信息量比传统 semantic tokens 更少,但与 text LLM 的对齐更好。

**实验验证**: Reasoning-only PPL avg 8.20 (speech), Reconstruction-only 29.36, Reason+Reconstruction 23.77 [Table 2]。理解任务: Reason+Reconstruction ASR WER 9.0 (vs Whisper 连续表征 8.5, vs DAC 93.2) [Table 3]。

**设计选择 2: Group-wise VQ 而非标准 RVQ** [§3.2, Table 16]

[论文原文] Reconstruction branch 将 3 种语义特征 (WavLM phone-level, music structure, Whisper environmental+acoustic) 用独立 VQ 头量化,分配 1:1:6 层。6 层分配给 environmental/acoustic 信息以编码 fine-grained 声学细节。

[论文原文] 对比实验 [Table 16]: Group-wise VQ vs RVQ vs query-based quantization。Query-based 重建最好 (PESQ WB 2.54) 但需额外 transformer 增加推理成本; Group-wise VQ 在 PPL (建模难度) 上最优 (speech 23.77, music 20.1),兼顾重建和可建模性。

**设计选择 3: GRPO 训练 reasoning branch** [§3.1, §A.2.2, §A.3.1]

[论文原文] Reasoning branch 训练分两阶段: SFT (多任务理解: ASR, captioning 等) → GRPO (reward-based RL 强化 detailed audio analysis 能力)。GRPO 使用 LLaMA 3.1-Instruct 8B 作为 judge 评估 rollout 与 ground-truth 的一致性 [§A.2.2]。

**效果**: ASR WER: 10.68 (SFT) → 7.64 (SFT+GRPO); Audio CLS: 36.4 → 40.2; Audio Reasoning GPT-score (relevance): 5.2 → 7.8 [Table 14]。

**设计选择 4: Functional layer specialization** [§4.3]

[论文原文] 将 transformer 概念性分为三类 expert,关键机制是 audio-only computation [Eq.7]: understanding 和 generation experts 只修改 audio position 的 hidden states,text position 完全透传。这保留了 LLaMA 3.2 3B 的 text processing pathway。

[论文原文] Cross-modal expert (28 层) 是全部从 LLaMA 初始化的,同时处理 text 和 audio,负责跨模态对齐。

**消融验证** [Table 8]: 移除 experts (仅用 cross-modal) 导致全面性能下降: ASR LS-clean 2.71→4.17, Audio Gen KL 3.26→3.98, Music Gen KL 1.8→4.12。

**设计选择 5: Auditory sentences** [§4.4, §B.2]

[论文原文] 受 LVM sequential training 启发,构造长上下文训练序列: 将多个语义或声学相关的 audio/text 片段拼接为 2-8 段的 "auditory sentence"。五种构造策略: (1) 长音频分段; (2) speech-text 交替; (3) audio/music-caption 交替; (4) mixture-clean triples (混合+分离); (5) semantic-consistent but acoustically-varied pairs (同文本不同风格)。

[论文原文] Auditory sentences 在 Stage 4 mid-training 中使用,context length 从 1024 扩展到 2048。移除 Stage 4 后 few-shot/zero-shot 能力无法获得 [Table 22: w/o Stage 4 → few-shot VC/Sound 为 NA]。

### 训练策略

**四阶段渐进训练** [§4.4, Table 18, Fig 5]:

| Stage | 目标 | 可训练模块 | 数据量 | Steps | Ctx |
|-------|------|-----------|--------|-------|-----|
| 1 | Understanding warm-up | Understanding experts | 3B tokens | 50K | 1024 |
| 2 | Generation warm-up | Generation experts + local decoder | 3B tokens | 50K | 1024 |
| 3 | Audio-text pre-training | All parameters | 50B tokens | 500K | 1024 |
| 4 | Mid-training | All parameters | 20B tokens | 300K | 2048 |

[§4.4, §B.3]

- Stage 1 附加 semantic distillation loss: L_stage1 = L_LM + λ_rec * L_rec,辅助 decoder 训练后丢弃 [Eq.13-14]
- Stage 2 使用 stream-weighted AR loss: 前 3 层 (reasoning 相关) 权重 2/8,后 5 层 1/8 [Eq.16-17]
- Stage 3 text loss 权重 1.6x audio loss (λ_text=1.6, λ_audio=1),保护 text 能力 [Eq.18]
- 总训练数据: 100B text tokens + 60B audio tokens,64 x H100 GPUs [§4.4, §B.3]

**ReasoningCodec 训练** (独立于 UniAudio 2.0) [§A.2, Table 11]:
- ~10,000 小时 (speech 5K + sound 3K + music 2K)
- 三阶段: SFT → GRPO → Reconstruction branch (freeze reasoning)
- 8 x A100 GPUs [§A.2.4]

## 实验

### Codec 评估

| 指标 | ReasoningCodec | MimiCodec | Higg-Audio | X-Codec | DAC | 数据集 | 出处 |
|------|----------------|-----------|------------|---------|-----|--------|------|
| MUSHRA Speech | **90.5±2.8** | 86.7±2.1 | 84.4±2.6 | 78.5±4.5 | 71.3±1.9 | VCTK | Table 4 |
| MUSHRA Sound | **80.8±2.0** | 72.6±2.1 | - | 79.2±1.8 | 70.0±1.9 | AudioCaps | Table 4 |
| MUSHRA Music | **86.6±2.3** | 69.8±1.9 | - | 81.0±1.6 | 63.0±1.8 | MusicCaps | Table 4 |
| PPL Speech avg | **8.20** (reason) | - | 38.46 | 18.40 | 90.06 | - | Table 2 |
| Understanding ASR | **9.0** (R+R) | - | 31.2 | 37.6 | 93.2 | LibriSpeech | Table 3 |
| Understanding ER | **56.4** (R+R) | - | 30.0 | 29.4 | 5.2 | ESD | Table 3 |

[§5.3] 所有指标控制在相同 token rate (100 TPS) 下比较。

### UniAudio 2.0 Seen Tasks

| 任务 | UniAudio 2.0 (3B) | MiMo-Audio 7B | Qwen2.5-Omni 7B | 数据集 | 出处 |
|------|-------------------|---------------|------------------|--------|------|
| ASR LS-clean WER | **2.71** | 3.50 | 3.92 | LibriSpeech | Table 19 |
| ASR SEED-ZH CER | **2.6** | 29.81 | 1.3 | SEED-TTS | Table 19 |
| TTS SEED-EN WER | 3.63 | 5.37 | **3.10** | SEED-TTS | Table 5 |
| TTS SEED-ZH WER | 2.30 | 1.93 | **1.21** | SEED-TTS | Table 5 |
| InstructTTS WER | **7.3** | 7.8 | - | CapSpeech | Table 5 |
| Song Gen WER | **36.5** | - | - | SongGen | Table 5 |
| Lyric Recog WER | **28.57** | - | 56.99 | SongGen | Table 5 |

### Few-shot / Zero-shot

| 任务 | UniAudio 2.0 | Baseline | Setting | 出处 |
|------|-------------|----------|---------|------|
| 1-shot Emotion EN/ZH | **67.0/59.8** | MiMo 42.5/45.0 | 1-shot | Table 6 |
| 1-shot SE WER | **14.13** | MiMo 65.29 | 1-shot | Table 6 |
| DSR (zero-shot) | **19.4** | Qwen2.5-Omni 80.6 | 0-shot | Table 7 |
| MMLU (zero-shot) | 44.1 | LLaMA 3.2 3B **47.63** | 0-shot | Table 7 |
| S2S GPT-score | 2.16/3.66 | LLAMA-Omni **3.47/3.99** | 0-shot | Table 7 |

## 局限性

1. **推理延迟**: Flow-based decoder 需 10-25 步 diffusion,增加生成延迟。作者建议未来探索 2-step 解码 [Appendix C.1]。[论文原文]

2. **规模有限**: 仅探索 1B/3B,未验证 scaling law。1B→3B 的提升显著 (few-shot/zero-shot 能力差距巨大),暗示更大模型有潜力 [Appendix C.3, Table 25: 1B MMLU 30.2 vs 3B 44.1]。[论文原文]

3. **数据不均衡**: Speech 数据远多于 sound/music,导致非语音任务相对弱。Audio generation FD 50.69 好于 Stable Audio (78.24) 但 CLAP-score 0.17 远低于 AudioLDM2 的 0.41 [Table 24]。[论文原文]

4. **S2S 对话弱**: Zero-shot S2S GPT-score 2.16,低于 LLAMA-Omni 的 3.47 [Table 7]。[agent 解读] 这可能因为训练中没有显式的 speech conversation 数据,auditory sentences 不足以替代。

5. **未做 post-training**: 没有 multi-task SFT 或 RLHF,post-training 可能进一步提升 [Appendix C.5]。[论文原文]

6. **Text 能力损耗**: MMLU 从 LLaMA 3.2 3B 的 47.63% 降到 44.1%,约 3.5pp 损耗 [Table 7]。[agent 解读] 尽管作者声称 "introduction of audio modalities does not significantly degrade text performance",7.4% 相对下降并非可忽略。

## 点评

**贡献定位**: UniAudio 2.0 的核心创新在于 ReasoningCodec,它提出了一种比 semantic vs acoustic 二分法更有原则的 token 设计哲学: 不是试图在一个 token 流中混合语义和声学 (如 SpeechTokenizer/Mimi),而是显式分解为 "对 LLM 友好的规划信号" 和 "对重建友好的声学信号"。5 Hz reasoning tokens 的极低帧率使得 LLM 可以在极短序列上做 "high-level planning",然后 reconstruction tokens 在这个 plan 的指导下做细粒度生成。

**与 KB 已有路线的对比**:
- 与 **CosyVoice 系列** (监督式 semantic tokens) 的区别: CosyVoice 在 ASR encoder 中间层插 FSQ,reasoning tokens 则 freeze audio encoder + learn lightweight projection 到 text LLM latent space,实现更直接的跨模态对齐。Table 15 显示 reasoning tokens 在 ASR (10.1 vs 32.3) 和 ER (50.2 vs 34.5) 上均大幅优于 CosyVoice 3 tokenizer [§A.3.6]。
- 与 **Moshi/Kimi-Audio** (vanilla unified AR) 的区别: 这些系统用所有 transformer 层均匀处理 text+audio,UniAudio 2.0 认为这会导致 text knowledge forgetting,提出 layer specialization 作为替代。
- 与 **UniAudio 1.0/1.5**: 从 generation-only → few-shot adaptation → unified understanding+generation,ReasoningCodec 和 layer specialization 是 2.0 的两个结构性升级。

**方法论洞察**:
- [agent 解读] GRPO 训练 tokenizer 是一个有趣的方向。传统 tokenizer 训练通过重建/对比目标优化,GRPO 引入了 "生成详细分析" 这样的高级目标,让 reasoning tokens 不仅编码 what 而且编码 why。这与 DeepSeek-R1 的思路类似但应用在表征学习而非推理。
- [agent 解读] FiLM 连接两个分支是轻量高效的选择。Table 13 显示移除 FiLM 对重建质量几乎无影响 (PESQ 2.36 vs 2.39),但对 generation PPL 有明显影响 (23.77 vs 25.9) [Table 12],说明 FiLM 的价值主要在于帮助 LM 更好地建模 reconstruction tokens,而非直接改善重建。
- [agent 解读] Auditory sentences 是一个被低估的贡献。Table 22 显示 w/o Stage 4 时 few-shot VC 和 sound classification 完全无法工作 (NA),说明 few-shot/zero-shot 能力几乎完全来自 auditory sentences 训练。这暗示 ICL 能力需要专门的数据构造策略,不会从标准 multi-task 训练中自然涌现。

**局限性分析**:
- [agent 解读] 3B 参数做到跨 speech/sound/music 的统一模型,与 MiMo-Audio 7B 和 Qwen2.5-Omni 7B 竞争,说明 tokenizer 设计对参数效率有重要影响。但这也意味着如果 MiMo-Audio/Qwen2.5-Omni 采用类似 tokenizer 设计并 scale 到更大模型,差距可能迅速拉开。
- [agent 解读] Audio generation 的 CLAP-score 偏低 (0.17 vs AudioLDM2 0.41) 揭示了统一模型的固有挑战: 在 speech-dominant 训练中,非语音生成能力可能受到压制。

## 可复用的 idea

1. **Reasoning tokens 作为 planning layer**: 在任何需要同时做理解和生成的多模态系统中,可以考虑在 tokenizer 层面引入一个 "planning" token 流,先让 LM 在低帧率上做高级决策,再在高帧率上做细粒度生成。这本质上是一种 latent chain-of-thought。

2. **GRPO 训练 tokenizer**: 用 RL 训练 tokenizer 的 reasoning branch,让 token 不仅编码信号特征,还编码对信号的分析能力。可扩展到任何需要 "理解力" 的 discrete representation 学习。

3. **Audio-only computation in specialized layers**: Eq.7 的 masked update 机制简洁优雅,可在任何多模态 transformer 中使用来保护特定模态的预训练知识。

4. **Auditory sentences 数据构造**: 五种构造策略 (分段/交替/mixture-clean/varied-style) 是构造 ICL 训练数据的通用模板,可迁移到任何需要 few-shot 能力的基础模型。

5. **Stream-weighted AR loss**: 对不同 codebook 层赋予不同权重 (reasoning-related 2x, others 1x) 是一个简单但有效的技巧,可在任何多流 codec LM 中使用 [Eq.16-17]。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 核心方法 (ReasoningCodec + layer specialization + auditory sentences) 清晰,关键公式和设计选择有 §/Eq 引用 |
> | 可信赖 | pass | [论文原文] vs [agent 解读] 标注清晰; 数值均标注出处 Table; 信息论分析正确引用自 §A.3.5 |
> | 可区分 | pass | 与 CosyVoice 3, SpeechTokenizer/Mimi, Moshi, UniAudio 1.0/1.5 的差异明确阐述; KB 背景精准定位 |
> | 可定位 | pass-with-fixes | §/Table 引用覆盖 >80%; 少数点评处缺乏精确出处 |
> | 不污染 | pass | 推测性内容标记 [agent 解读]; 未将 arXiv 系统结论当作 peer-reviewed 事实 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> - [medium] 点评中 "3.5pp text 能力损耗" 的评价 ("7.4% 相对下降并非可忽略") 是 agent 观点,已标注 [agent 解读],但可补充作者对此的解释
> - [low] 局限性第 4 点 S2S 弱的原因推测可进一步细化
> 详见 `_review/UniAudio2.0-review.yml`
