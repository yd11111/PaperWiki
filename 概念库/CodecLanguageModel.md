---
type: concept
title: "Codec Language Model"
aliases: [CodecLM, Codec LM, 编解码器语言模型, Neural Codec Language Model, Audio Codec LM]
category: "model-family"
tags: [speech-LM, codec, autoregressive, neural-audio-codec, language-model, RVQ]
key_papers: ["VALL-E (Wang et al., 2023)", "[[论文笔记/AudioLM|AudioLM]]", "VioLA (Wang et al., 2024)", "NTPP (Wang et al., 2025)", "SpeechGPT-Gen (Zhang et al., 2024)", "[[论文笔记/Survey-DiscreteAudioTokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/SPEAR-TTS|SPEAR-TTS]]", "[[论文笔记/UniAudio|UniAudio]]", "[[论文笔记/MELLE|MELLE]]", "[[论文笔记/Make-A-Voice|Make-A-Voice]]", "[[论文笔记/SCVALL-E|SC VALL-E]]", "[[论文笔记/SongGen|SongGen]]", "[[论文笔记/SpeechAlign|SpeechAlign]]", "[[论文笔记/RIO|RIO]]", "[[论文笔记/TextrolSpeech|TextrolSpeech]]", "[[论文笔记/Chatterbox-Flash|Chatterbox-Flash]]", "[[论文笔记/DiSTAR|DiSTAR]]", "[[论文笔记/Cont-SPT|Cont-SPT]]", "[[论文笔记/LiveSpeech2|LiveSpeech 2]]", "[[论文笔记/TacoLM|TacoLM]]", "[[论文笔记/TraceableSpeech|TraceableSpeech]]", "[[论文笔记/XTTS|XTTS]]", "[[论文笔记/DiffCSS|DiffCSS]]", "[[论文笔记/PALLE|PALLE]]", "[[论文笔记/LLMVoX|LLMVoX]]", "[[论文笔记/TTS-Transducer|TTS-Transducer]]", "[[论文笔记/CSP-FT|CSP-FT (Wang et al., 2026)]]", "[[论文笔记/Koel-TTS|Koel-TTS]]", "[[论文笔记/C2F-LM|C2F-LM]]", "[[论文笔记/CapSpeech|CapSpeech]]", "[[论文笔记/RevivalwithVoice|Revival with Voice]]", "[[论文笔记/StreamMel|StreamMel]]", "[[论文笔记/Dragon-FM|Dragon-FM]]", "[[论文笔记/Llasa+|Llasa+]]", "[[论文笔记/LatinX|LatinX]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/MAVE|MAVE]]"]
origin_paper: "[[论文笔记/Survey-SpeechLanguageModels|Cui et al., Speech Language Models, 2024]]"
related_concepts: ["[[SpeechLanguageModel]]", "[[LLM-basedTTS]]", "[[ResidualVectorQuantization]]", "[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[Single-codebookvsMulti-codebook]]", "[[TokenRateandBitrateTrade-offs]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Codec Language Model (CodecLM) 是 Speech Language Model 的一个子范式,特指直接在 neural audio codec (如 EnCodec, SoundStream) 产生的离散 acoustic tokens 上训练语言模型进行语音建模和生成。与使用 semantic tokens 的 SpeechLM 不同,CodecLM 直接操作编解码器的量化码本索引。

Survey (Cui et al., 2024) 明确指出: "Some studies directly model the codec tokens in a language model, which is often regarded as Codec Language Models (CodecLMs)."

**核心特征**:
- 输入/输出均为 neural codec 的 RVQ 码本索引
- 语言模型直接在这些 acoustic tokens 上做 next-token prediction
- 可用于 ASR、TTS、语音翻译等多任务

## 与 LLM-based TTS 的区别

| 维度 | Codec Language Model | LLM-based TTS |
|------|---------------------|---------------|
| 范围 | 通用语音建模 (ASR+TTS+ST+对话) | 专注于 text→speech 生成 |
| Token 类型 | 必须使用 acoustic (codec) tokens | 可用 semantic 或 codec tokens |
| 代表系统 | VioLA (多任务), NTPP (双通道对话) | VALL-E (TTS), SpearTTS (TTS) |
| 能力 | 理解 + 生成 | 主要是生成 |

简言之: LLM-based TTS 是 TTS 领域的范式,CodecLM 是 SpeechLM 领域基于 codec token 的建模范式。两者在 VALL-E 等系统上有交集,但各自外延不同。

## 代表系统

### VioLA (Wang et al., 2024)
- 在 EnCodec tokens 上训练的 CodecLM
- 同时支持 ASR、TTS、Machine Translation
- 验证了 codec tokens 作为通用语音表征的可行性

### NTPP (Wang et al., 2025)
- 在 VQ-VAE tokens 上训练
- "Next-token-pair prediction" 方法
- 建模双通道 (dual-channel) 口语对话数据
- decoder-only transformer 同时预测两个通道的 token

### SpeechGPT-Gen (Zhang et al., 2024)
- 使用 mixed tokens: 在推理时将 semantic tokens 转换为 acoustic tokens
- Chain-of-information generation 策略
- Scaling chain-of-information 信息生成

### AudioLM (Borsos et al., 2023)
- 两阶段: 先生成 w2v-BERT semantic tokens → 再生成 SoundStream acoustic tokens
- 建立了 semantic → acoustic 的层级 codec 建模范式
- 启发了后续 VALL-E、SoundStorm 等系统

## 关键技术挑战

1. **多层 RVQ 建模**: codec tokens 有多个量化层 (通常 4-8 层), 如何高效建模?
   - AR + NAR 两阶段 (VALL-E)
   - RQ-Transformer 并行 (Moshi)
   - Grouped code 编码 (VALL-E 2)

2. **语义-声学 gap**: 纯 acoustic tokens 语义信息稀疏,cross-modal 对齐困难
   - SpeechTokenizer 第一层蒸馏 HuBERT 语义
   - Mixed token 方案 (Mimi)

3. **序列长度**: 多层 codec token 展开后序列极长
   - Flattening vs interleaving 策略
   - 基于 delay pattern 的 codebook 排列
   - 在 LM 层面做 token 压缩: [[论文笔记/C2F-LM|C2F-LM]] 将长距离 speech token 压缩为 5Hz 紧凑表征,通过 attention mask + 压缩 token 实现信息瓶颈,兼容单/多码本 codec 和各种 LM 架构
   - 仅建模前 N 层 RVQ 以换取更长上下文: [[论文笔记/MOSS-TTSD|MOSS-TTSD]] 仅用 16 层 RVQ (2kbps/12.5Hz) + 65K token 上下文实现 3600 秒训练上下文,支持 60 分钟单次对话生成

## 在 Speech LM 中的角色

CodecLM 是 SpeechLM 中侧重 **声学保真度** 的路线。Survey 分类中属于 "Discrete Features → Acoustic Token" 和 "Mixed Token" 路线:
- 纯 acoustic token 路线: VioLA, Li et al., Parrot
- Mixed token 路线: Moshi, SpeechGPT-Gen

多数 SpeechLM 更倾向使用 semantic tokens 而非 codec tokens,因为语义理解是口语交互的核心需求。

## 关键论文

- VALL-E (Wang et al., 2023): 首个大规模 codec LM TTS
- AudioLM (Borsos et al., 2023): semantic → acoustic 层级 codec 建模
- VioLA (Wang et al., 2024): 多任务 CodecLM (ASR+TTS+ST)
- NTPP (Wang et al., 2025): 双通道对话 codec LM
- SpeechGPT-Gen (Zhang et al., 2024): chain-of-information codec 生成
- SongGen (Liu et al., 2025): 单阶段 AR Transformer 在 X-Codec (RVQ, 8 codebooks) tokens 上做 song generation,codebook-delay pattern + mixed/dual-track token patterns

## 相关概念

- [[SpeechLanguageModel]]: CodecLM 是 SpeechLM 的一个子范式
- [[LLM-basedTTS]]: CodecLM 在 TTS 上的应用即为 LLM-based TTS 的 codec 路线
- [[ResidualVectorQuantization]]: codec tokens 的量化方法
- [[SpeechTokenizer]]: 产生 codec tokens 的 acoustic tokenizer
- [[SemanticvsAcousticTokens]]: CodecLM 使用 acoustic 侧 tokens

## Survey Benchmark 发现 [Mousavi et al. 2025]

### SLM 评估 [§3.3.1, Table 10]
Survey 在统一条件下 (Qwen-2.5 357M, 50K steps) 训练 SLM 并在 SALMon + ZeroSpeech 上评估:
- Tokenizer 选择对 SLM 性能的影响巨大
- HuBERT 在语义任务 (sBLIMP, sWUGGY, sSC, tSC) 上最强
- WavLM 在声学一致性 (gender, sentiment, speaker) 上最强
- 纯 acoustic tokenizer (EnCodec, DAC) 在语义任务上接近随机 (~50%)
- 语义蒸馏加权 (Mimi*, ST*) 显著改善语义得分但伴随声学代价

### TTS 评估 [§3.3.2, Table 11]
Survey 基于 ESPnet VALL-E 实现在 LibriTTS 上评估 TTS:
- ESPnet EnCodec (speech-only 训练) 达最高 UTMOS (3.77), 证明 domain-specific 训练关键
- Discrete WavLM 第二高 UTMOS (3.42), 且训练最稳定
- WavLM (单码本) 达最低 dWER (4.32), 归因于大词表和单流简化建模
- 通用 acoustic tokenizer (EnCodec, DAC 原始版) UTMOS 仅 2.31/2.47

### 音频/音乐生成 [§3.3.3-3.3.4, Table 12-13]
- 音频生成: domain-matched EnCodec (Enc-A-16) 在 FAD/KLD/CLAP 上全面领先
- 音乐生成: EnCodec-32k (music-specific) 在 MusicCaps 上最优; DAC-44k 在多域中表现最好
- 重建质量最好的 tokenizer 不一定有最好的生成性能

## 演进

VQ-VAE speech tokens (2019) → SoundStream/EnCodec neural codecs (2021-2023) → AudioLM semantic→acoustic 两阶段 (2022) → VALL-E codec LM for TTS (2023) → VioLA 多任务 CodecLM (2024) → Moshi/SpeechGPT-Gen mixed-token CodecLM (2024) → 统一 benchmark 评估 (Mousavi et al., 2025)

> [!review] 实体页审阅 (2026-06-04)
> **结论:** restructure
> **原则:** 不混层 revise | 不过载 revise | 可溯源 pass-with-fixes | 可导航 revise | 不污染 pass-with-fixes
> **结构:** key_papers 37/12 | ## 标题 9/6 | 行数 112/200 | 长论文段落 5/3
> **问题:** 3 high, 4 medium, 2 low
> - ❌ [overload] frontmatter key_papers: 37条超出12条硬上限3倍,大量边缘论文稀释导航信号
> - ❌ [scope-drift] ## Survey Benchmark 发现: 21行survey数字搬运,属论文级细节
> - ❌ [layer-mixing] ## 演进: 研究时间线属于MOC,规则明确禁止
> - ⚠️ [layer-mixing] ## 代表系统: 4个论文专属子章节含模型级细节
> - ⚠️ [signal-dilution] key_papers格式不统一: 4条plain text + 33条wikilink
> - ⚠️ [scope-drift] ## 关键论文: 与frontmatter key_papers功能重叠(6篇 vs 37篇)
> - ⚠️ [overload] 9个##标题超过6个上限
> **重构建议:** 删除演进/Survey Benchmark/关键论文3个section; 压缩代表系统; key_papers精简到≤12条核心文献并统一wikilink格式
