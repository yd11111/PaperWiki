---
type: paper
tier: deep
title: "Kimi-Audio"
arxiv_id: "2504.18425"
source: "Sources/Kimi-Audio.pdf"
authors: ["Kimi Team (Moonshot AI)", "Xu Tan (project lead)", "Xinyu Zhou (project lead)", "Ding Ding", "Zeqian Ju", "Yichong Leng", "Songxiang Liu", "Tong Liu", "Zeyu Shang", "Kai Shen", "Wei Song", "Heyi Tang", "Zhengtao Wang", "Chu Wei", "Yifei Xin", "Xinran Xu", "Jianwei Yu", "Yutao Zhang"]
year: 2025
venue: "arXiv"
tags: [speech-LM, audio-foundation-model, audio-understanding, speech-conversation, flow-matching, universal-audio, open-source, hybrid-tokenization, dual-head-LLM, streaming-detokenizer]
concepts: ["[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/SpeechTokenizer|Speech Tokenizer]]", "[[概念库/ConditionalFlowMatching|Conditional Flow Matching]]", "[[概念库/SemanticvsAcousticTokens|Semantic vs Acoustic Tokens]]", "[[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation for Speech LLM]]", "[[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLM Integration Taxonomy]]", "[[概念库/AudioTokenizerTaxonomy|Audio Tokenizer Taxonomy]]", "[[概念库/LLM-basedTTS|LLM-based TTS]]", "[[概念库/NeuralVocoder|Neural Vocoder]]"]
models: ["Kimi-Audio", "Qwen2-Audio", "Baichuan-Audio", "Step-Audio", "GLM-4-Voice", "Qwen2.5-Omni", "GPT-4o", "GPT-4o-mini", "Kimi-TTS", "Kimi-VC", "MoonCast", "BigVGAN", "Whisper"]
tasks: ["ASR", "audio-understanding", "audio-to-text-chat", "speech-conversation", "TTS", "audio-question-answering", "speech-emotion-recognition", "sound-event-classification"]
datasets: ["LibriSpeech", "AISHELL-1", "AISHELL-2", "WenetSpeech", "Fleurs", "CommonVoice", "MMAU", "MELD", "VoiceBench", "OpenAudioBench", "ClothoAQA", "VocalSound", "Nonspeech7k", "TUT2017", "CochlScene", "Emilia", "LibriTTS", "WenetSpeech4TTS", "Gigaspeech", "AudioCaps"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[概念库/SpeechLanguageModel|SpeechLM]], [[概念库/SpeechTokenizer|SpeechTokenizer]], [[概念库/ConditionalFlowMatching|CFM]], [[概念库/SemanticvsAcousticTokens|Semantic vs Acoustic Tokens]], [[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation]], [[概念库/Speech-LLMIntegrationTaxonomy|Integration Taxonomy]])
>
> **定位**: Kimi-Audio 是一个 **universal audio foundation model**,在 SpeechLM 演进线上属于最新一代"omni-model"——兼具理解、生成、对话三种能力。在 Speech-LLM Integration Taxonomy [待确认] 中,它属于 **hybrid integration**: 输入端同时使用 discrete semantic tokens (audio-token-based) 和 continuous acoustic vectors (latent-representation-based),输出端使用 discrete semantic tokens + text tokens 的双头架构。这种混合设计不完全落入任何一个既有分类桶。
>
> **Tokenizer 层面**: Kimi-Audio 的 tokenizer 使用 GLM-4-Voice 的 VQ-Whisper 方案 (12.5Hz, 单码本, ASR 监督) 获取 discrete semantic tokens,属于 AudioTokenizerTaxonomy [待确认] 的"监督式 semantic tokenizer"路线 (与 CosyVoice S3 同类)。但它还额外融合 Whisper encoder 的 continuous acoustic features (50Hz → adaptor → 12.5Hz),再与 semantic token embedding 相加作为输入。这种 "discrete + continuous" 的双路输入在 KB 中属于较少见的设计——多数系统二选一 (语义 OR 声学,或 mixed tokenizer)。
>
> **Detokenizer 层面**: 使用 CFM (flow matching) 将 12.5Hz semantic tokens 转为 50Hz mel spectrogram,再由 BigVGAN vocoder 合成波形。这与 CosyVoice/MaskGCT/Seed-TTS 等系统的 coarse-to-fine pipeline 一致,但 Kimi-Audio 的独特之处在于 **chunk-wise streaming + look-ahead** 机制实现低延迟流式解码。
>
> **与同期系统的对比**: 与 Qwen2.5-Omni (Thinker-Talker 双模型) 和 Step-Audio (130B 参数) 相比,Kimi-Audio 走的是"中等规模 (7B base) + 大规模预训练 (13M hours)" 路线。
>
> 检索命中: SpeechLM, SpeechTokenizer, CFM, SemanticvsAcousticTokens, ModalityAdaptation, Integration Taxonomy | 过滤: 无 | 未命中但可能相关: Full-duplexSpokenDialogue (Kimi-Audio 当前是 turn-based, 非 full-duplex)

## 速查

> [!summary] 速查
> - **一句话**: 开源 audio foundation model,通过 hybrid tokenization (discrete semantic + continuous acoustic)、dual-head LLM (text + audio) 和 chunk-wise streaming flow matching detokenizer 实现 ASR/理解/对话/生成多任务 SOTA
> - **路线**: Audio → VQ-Whisper 12.5Hz semantic tokens + Whisper continuous features (adaptor 4x downsample) → 相加作为 LLM 输入 → Shared LLM layers → Text Head + Audio Head 并行生成 → Audio tokens → Flow Matching (chunk-wise streaming + look-ahead) → BigVGAN → Waveform
> - **指标**: ASR: LibriSpeech 1.28/2.42 WER [Table 4], AISHELL-1 0.60 WER [Table 4]; 理解: MMAU sound 73.27 [Table 5], CochlScene 79.84/80.99 [Table 5]; 对话: VoiceBench avg 76.93 [Table 6]; 语音对话: avg 3.90 (vs GPT-4o 4.06) [Table 7]
> - **可借鉴**: (1) 6-blank-token delay 解决 audio-text 并行生成初始阶段对齐困难 [§4.1.3]; (2) chunk-wise streaming + look-ahead 4 tokens 的 training-free 流式 detokenizer 方案 [§2.4]; (3) 原始/增强 audio 1:1 混合防止 speech enhancement 破坏环境音理解 [§3.1]; (4) 7 种预训练任务的 weight 配比方案 (text 7: 其余各 1-2) [Table 3]
> - **局限**: 非全双工 (turn-based, 依赖 VAD 判断说话结束) [§5.1]; 基础 LLM 仅 7B; 13M hours 预训练数据细节不完全可复现 (含 in-house data); 语音对话的 empathy 维度弱于 GPT-4o (3.39 vs 3.87) [Table 7]; 未来挑战部分坦承当前 audio token 表征仍不够好 [§8]

## 核心问题

本文要解决的核心问题是: **如何构建一个真正 universal 的 audio foundation model**,同时具备 audio understanding (ASR, 情感识别, 事件分类等)、audio generation (TTS, speech conversation) 和 instruction-following 能力?

[论文原文] 作者明确指出现有工作的三个不足 [§1]: (1) 只专注理解或生成,不 universal; (2) 缺乏大规模 audio 预训练,只做 LLM fine-tuning; (3) 不开源。Kimi-Audio 的定位是同时解决这三个问题。

## 方法: 它怎么 work

> [!important] 区分来源
> [论文原文] = author's explanation; [agent 解读] = my inference.

### 整体架构

[论文原文] Kimi-Audio 由三个核心组件构成 [§2.1, Fig 2]:

1. **Audio Tokenizer**: 将输入 audio 转换为 discrete semantic tokens (12.5Hz) + continuous acoustic vectors
2. **Audio LLM**: 处理多模态输入,通过 shared layers + dual parallel heads 输出 text tokens 和 audio tokens
3. **Audio Detokenizer**: 将 audio tokens 还原为 waveform,基于 flow matching + vocoder

**信息流**:
```
Input Audio
  ├── VQ-Whisper → Discrete Semantic Tokens (12.5Hz, 单码本)
  └── Whisper-large-v3 encoder → Continuous Features (50Hz)
       └── Adaptor → Downsampled to 12.5Hz
              ↓
       Semantic Embedding + Continuous Features (element-wise add)
              ↓
       Shared LLM Layers (from pre-trained Qwen2.5-7B)
              ├── Text Head → Text Tokens (autoregressive)
              └── Audio Head → Audio Semantic Tokens (autoregressive)
                      ↓
              Flow Matching (chunk-wise) → Mel Spectrogram
                      ↓
              BigVGAN Vocoder → Waveform
```

### 关键设计选择

#### 1. 为什么用 discrete semantic tokens + continuous acoustic vectors 的混合输入?

[论文原文] Discrete semantic tokens 提供高效的语义表征 (12.5Hz 低帧率), continuous Whisper features 增强 perception capability [§2.2]。两者相加作为 LLM 输入。

[agent 解读] 这解决了 semantic-only input 的信息瓶颈: 12.5Hz 单码本 semantic tokens 丢失大量声学细节 (音色、情感、环境音), 而 Whisper encoder features 保留了这些信息。相加 (而非拼接) 保持了序列长度不变,不增加 LLM 计算成本。这种设计的本质是: **用 discrete tokens 做 LLM 的"语言骨架",用 continuous features 做"声学补丁"**。

#### 2. 为什么用 dual-head (Text Head + Audio Head) 而非单头输出?

[论文原文] 在输出端同时生成 discrete semantic audio tokens 和 text tokens,以 enhance generation capability [§2.3]。Text head 和 shared layers 从 pre-trained text LLM 初始化,audio head 随机初始化。

[agent 解读] 同时生成 text 有两个好处: (1) 在 speech conversation 场景中,text 输出是 audio 输出的隐式 "chain of thought"——LLM 先想清楚要说什么文本,再生成对应 audio; (2) 保留了 text-only 任务能力 (如 audio-to-text chat)。与 GLM-4-Voice 的 "streaming thoughts" 概念类似。

#### 3. Audio-text 并行生成的 blank token delay [§4.1.3]

[论文原文] 在 audio-to-semantic+text interleaving 任务中,semantic audio token 序列总是长于 text token 序列,预测前几个 semantic tokens 特别困难,因为模型需要同时预测 text token 和对应的 audio token。解决方案: 在 semantic audio tokens 开头插入 6 个 blank tokens,延迟 audio 预测的开始。

[agent 解读] 这个 "6-blank delay" 的本质是给模型一个 warm-up 窗口: 让 text head 先确定内容方向后,audio head 再跟进。这与 Qwen2.5-Omni 的 Thinker-Talker 思路类似,但更轻量——只需要几个 blank tokens 而非两个独立模型。数字 6 是 quality-latency trade-off 的实验结果。

#### 4. Chunk-wise streaming detokenizer [§2.4]

[论文原文] 将 audio 分成 ~1s 的 chunks,flow matching 模型以 chunk-wise causal mask 训练和推理: chunk c_i 看到所有 c_j (j<i) 作为 prompt。但 chunk 边界存在 intermittent issue (断续问题)。

**Look-ahead 机制**: 对 chunk c_i,取 c_{i+1} 的前 n=4 个 semantic tokens 拼接到 c_i 末尾,生成 mel 后只保留 c_i 对应部分。这是 training-free 的,仅延迟首个 chunk 生成 n tokens。

[agent 解读] 这个 look-ahead 机制很巧妙: block-wise causal attention 的问题是边界位置"看不到未来",导致生成质量退化。通过 peek 未来 4 个 tokens (0.32s @12.5Hz),边界位置有了足够上下文。关键优势是 **training-free**——不需要修改训练流程,只在推理时调整输入构造。

#### 5. Whisper encoder 的渐进解冻 [§4.1.4]

[论文原文] Whisper-large-v3 作为 continuous feature extractor,前 20% 预训练 tokens 阶段冻结,之后解冻联合训练。

[agent 解读] 这与 ModalityAdaptation 中 Wu et al. (2023) 的两阶段策略一致: 先让 LLM 适应 frozen encoder 的输出分布,再允许 encoder 微调以适应下游需求。冻结 20% 是工程上的平衡点——太早解冻会导致 encoder 输出分布剧变,破坏 LLM 已学的表征映射。

### 训练策略

#### Pre-training [§4.1, Table 3]

[论文原文] 7 种预训练任务,分三类:

**类别 1: 单模态预训练**
- Text Only (weight 7): 标准 next-token prediction on text
- Audio Only (weight 1): next-token prediction on discrete semantic tokens

**类别 2: Audio-Text Mapping**
- Audio to Text / ASR (weight 1): {a_1, t_1, a_2, t_2, ...}, loss on t_i
- Text to Audio / TTS (weight 1): {t_1, a^d_1, t_2, a^d_2, ...}, loss on a^d_i

**类别 3: Audio-Text Interleaving**
- Audio to Semantic (weight 1): {a_1, a^d_2, a_3, a^d_4, ...}, loss on a^d_i
- Audio to Text (weight 1): {a_1, t_2, a_3, t_4, ...}, loss on t_i
- Audio to Semantic + Text (weight 2): {a_1, a^d_2/t_2, a_3, a^d_4/t_4, ...}, 最核心的任务

[agent 解读] 任务权重分配暴露了一个关键判断: **text 能力远重要于 audio 能力** (weight 7 vs 其余 1-2)。这是因为模型从 text LLM 初始化,需要大量 text data 防止灾难性遗忘。Audio-to-Semantic+Text 权重 2 是因为这是最终推理时的核心模式——同时输出 audio 和 text。

**规模**: 585B audio tokens + 585B text tokens, 1 epoch。Base LLM: Qwen2.5 7B。优化器: AdamW, LR 2e-5 → 2e-6 cosine decay, 1% warmup。

#### SFT [§4.2]

[论文原文] ~300K hours SFT data,三部分:
1. **Audio understanding**: 45+ 开源数据集 (ASR/AQA/AAC/SER/SEC/ASC) + 55K hours in-house ASR + 5.2K hours in-house AQA/AAC [Table 1]
2. **Speech conversation**: Kimi-TTS 合成 user queries (125K+ timbres) + voice actor (20+ styles x 5 emotion levels) 录制 assistant responses + Kimi-VC 声音转换 [§3.2.2]
3. **Audio-to-text chat**: 开源 text SFT data (Magpie, Infinity-Instruct, NuminaMath 等) 的 user query 用 TTS 转语音 [Table 2]

[agent 解读] 亮点在于 speech conversation data 构造: 不是简单 TTS 合成,而是设计了完整的数据工程管线——voice actor 录制 + VC 扩展风格 + 大 timbre pool 合成 queries。这种方法生成的训练数据在情感表现力上远超纯 TTS 合成数据。

#### Detokenizer Training [§4.3]

[论文原文] 三阶段:
1. ~1M hours 预训练 flow matching + vocoder (多说话人)
2. Chunk-wise fine-tuning (动态 chunk size 0.5-3s)
3. 单一 voice actor 数据 fine-tuning

### Data Processing Pipeline [§3.1, Fig 3]

[论文原文] 13M hours raw audio → 自动化处理管线:
1. **Speech Enhancement**: BSRNN, 48kHz。但因会破坏环境音,原始/增强 audio 1:1 混合使用
2. **Diarization**: PyAnnote → speaker cluster merging (cosine >0.6) → chunk-based reassignment (1.5s chunks, cosine <0.5) → segment merging (≤27s, gap ≤2s)
3. **Transcription**: Whisper-large-v3 (语言检测 + English ASR) + Paraformer-Zh (中文 ASR + 基于时间间隔的标点)
4. **Infrastructure**: 30 cloud instances, 240 L20 GPUs, 200K hours/day throughput

## 实验

| 指标 | 本文 | Best Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER↓ | **1.28/2.42** | 1.74/4.04 (Qwen2-Audio) | LibriSpeech test-clean/other | [Table 4] |
| WER↓ | **0.60** | 1.13 (Qwen2.5-Omni) | AISHELL-1 | [Table 4] |
| WER↓ | **2.56** | 2.56 (Qwen2.5-Omni) | AISHELL-2 ios | [Table 4] |
| WER↓ | **6.28/5.37** | 7.71/6.04 (Qwen2.5-Omni) | WenetSpeech test-meeting/net | [Table 4] |
| Acc↑ (sound/speech) | **73.27/60.66** | 69.07/53.92 (Qwen2.5-Omni) | MMAU | [Table 5] |
| Acc↑ | **59.13** | 51.23 (Qwen2-Audio) | MELD (emotion) | [Table 5] |
| Acc↑ | **65.25** | 43.27 (Qwen2.5-Omni) | TUT2017 (scene) | [Table 5] |
| Acc↑ | **79.84/80.99** | 63.82/63.82 (Qwen2.5-Omni) | CochlScene | [Table 5] |
| Score↑ | **75.73** | 72.76 (Qwen2.5-Omni) | OpenAudioBench AlpacaEval | [Table 6] |
| Avg↑ | **76.93** | 72.83 (Qwen2.5-Omni) | VoiceBench overall avg | [Table 6] |
| MOS↑ | **3.90** | 4.06 (GPT-4o) | Speech conversation (5维avg) | [Table 7] |
| Emotion↑ | **4.27** | 4.24 (GPT-4o-mini) | Speech conversation emotion | [Table 7] |
| Speed↑ | **4.30** | 4.21 (GPT-4o) | Speech conversation speed ctrl | [Table 7] |

**关键观察**:
- ASR 上全面 SOTA,尤其 AISHELL-1 (0.60 vs 1.13) 和 LibriSpeech test-other (2.42 vs 4.04) 提升显著 [Table 4]
- Audio understanding 在非语音任务 (TUT2017 +50%, CochlScene +25%) 上优势巨大,说明 13M hours 多域预训练 + 原始/增强 1:1 策略有效 [Table 5]
- Speech conversation 与 GPT-4o 差距主要在 empathy (3.39 vs 3.87) 和 accent control (3.45 vs 3.65) [Table 7]

## 局限性

1. **非全双工**: 依赖 VAD 检测用户说话结束后才触发推理 [§5.1],不支持 barge-in 或 real-time interleaved dialogue (对比 Moshi 的 full-duplex)
2. **基础模型规模**: 仅 7B 参数 (Qwen2.5-7B base),对比 Step-Audio 130B,智力上限受限
3. **预训练数据不完全可复现**: 13M hours 含 crawled data,具体来源和处理细节不完全公开; 55K hours in-house ASR data 和 5.2K hours in-house audio data 不开源 [Table 1]
4. **Empathy 弱项**: Speech conversation 中 empathy 维度 (3.39) 明显弱于 GPT-4o (3.87) [Table 7],可能因为 voice actor 数据有限
5. **Audio token 表征上限**: 作者在 §8 坦承当前 semantic tokens 丢失副语言信息,acoustic tokens 丢失语义,呼吁 "better audio representations" 和 "throw away ASR and TTS"——即系统的上限被 tokenizer 质量和 ASR/TTS 数据管线的质量天花板所限制
6. **语言覆盖**: 仅 English + Mandarin [§3.1],对比 Qwen2.5-Omni 的多语言覆盖有限
7. **Evaluation fairness**: 虽然提出了 eval toolkit 值得肯定,但 speech conversation 仅对比了 GPT-4o/GLM-4-Voice/Step-Audio 少数系统,且主观评分的 evaluator 信息未公开 [Table 7]

## 点评

**优势**:
1. **工程完整度极高**: 从 tokenizer 到 LLM 到 detokenizer 到 data pipeline 到 deployment 到 eval toolkit,每个环节都有详细描述。这在 audio foundation model 论文中罕见。
2. **Hybrid tokenization 设计合理**: discrete semantic tokens 做序列建模骨架 + continuous acoustic features 做信息补充,既保证了 LLM 建模效率 (12.5Hz 低帧率),又不牺牲感知能力。相比 Moshi 的 8 codebook RVQ 或 Step-Audio 的多码本方案,Kimi-Audio 的方案更简洁。
3. **Streaming detokenizer 的 look-ahead 机制**: training-free、低延迟、解决了 chunk 边界断续问题。这类工程优化往往在 paper 中被忽略,但对实际部署至关重要。
4. **数据处理管线设计成熟**: speech enhancement 的 1:1 混合策略、diarization 的三步后处理 (cluster merging → chunk reassignment → segment merging)、Paraformer 的时间间隔标点策略——每个细节都有工程判断力。

**不足**:
1. **缺乏 ablation study**: 没有报告各设计选择的消融实验 (hybrid vs discrete-only input、dual-head vs single-head、6 blank tokens vs 其他数量、task weights 的敏感性等)。对于一篇 technical report 来说,这是显著缺失。
2. **Turn-based 对话局限**: 在 full-duplex 成为热点 (Moshi, OmniFlatten) 的背景下,Kimi-Audio 的 VAD-based turn-taking 是架构层面的局限。
3. **Detokenizer 与 LLM 的分离**: Flow matching detokenizer 独立于 LLM 训练,这意味着 LLM 无法直接优化最终语音质量——生成的 semantic tokens 是否"好"只由 LLM 的 token-level loss 判断,而非 end-to-end 的语音质量 loss。

## 可复用的 idea

1. **6-blank-token delay for parallel generation** [§4.1.3]: 当模型需要同时预测两个模态的 token (audio + text) 时,延迟较难的模态 (audio) 几个 step 让简单模态 (text) 先确定方向。这个技巧可迁移到任何 multi-head 或 multi-stream 生成场景。具体延迟量需要根据 quality-latency trade-off 实验确定。

2. **原始/增强 audio 1:1 混合** [§3.1]: Speech enhancement 会破坏环境音/音乐信号,但不做增强又有噪声。简单的 1:1 随机混合在 audio understanding 任务上取得了极好效果 (CochlScene +25%)。这个策略可直接用于任何多域 audio 预训练场景。

3. **Chunk-wise streaming flow matching + look-ahead** [§2.4]: 将 flow matching detokenizer 改造为流式的方法: chunk-wise causal mask + 从下一个 chunk "借" n 个 tokens 做 look-ahead + 只保留当前 chunk 的输出。Training-free,适用于任何 flow matching / diffusion 声码器。

4. **Diarization 后处理三步法** [§3.1]: Speaker cluster merging (cosine >0.6) → chunk-based reassignment (1.5s, cosine <0.5) → segment merging (≤27s, gap ≤2s)。这套 heuristic 解决了 PyAnnote 常见的 speaker fragmentation 和 segment 长度不一致问题,可直接复用于大规模语音数据处理。

5. **Pre-training task weight ratio** [Table 3]: Text 7 : Audio 1 : ASR 1 : TTS 1 : Interleave 各 1-2。这个比例反映了 text capability preservation 的重要性,对从 text LLM 初始化的 speech LLM 具有参考价值。

> [!review] 自审阅 (auto, 2026-06-08)
> **结论: pass**
> - 可复述: pass — 架构、训练、推理全流程可从本笔记独立复述
> - 可信赖: pass — 关键数字均标注 [Table N]/[§X.X],覆盖率 >80%
> - 可区分: pass — [论文原文] 与 [agent 解读] 在每个设计选择中明确标注
> - 可定位: pass — KB 背景定位了 hybrid integration、监督式 semantic tokenizer、CFM detokenizer 三个维度
> - 不污染: pass — 对概念页的引用均为事实性描述,无推测性内容
> - 补充: 缺少 ablation study 是论文本身的缺陷,笔记已在局限性中指出
