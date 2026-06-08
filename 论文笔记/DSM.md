---
type: paper
tier: deep
title: "Streaming Sequence-to-Sequence Learning with Delayed Streams Modeling"
arxiv_id: "2509.08753"
source: "Sources/DSM.pdf"
authors: [Neil Zeghidour, Eugene Kharitonov, Manu Orsini, Vaclav Volhejn, Gabriel de Marmiesse, Edouard Grave, Patrick Perez, Laurent Mazare, Alexandre Defossez]
year: 2025
venue: "arXiv"
tags: [streaming, seq2seq, ASR, TTS, multi-stream, decoder-only, speech-text, latency, delay-conditioning, dialogue-TTS]
concepts: ["[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/CodecLanguageModel|Codec Language Model]]", "[[概念库/LLM-basedTTS|LLM-based TTS]]", "[[概念库/SpeechTokenizer|Speech Tokenizer]]", "[[概念库/SemanticvsAcousticTokens|Semantic vs Acoustic Tokens]]", "[[概念库/StreamingSpokenDialogue|Streaming Spoken Dialogue]]", "[[概念库/ResidualVectorQuantization|Residual Vector Quantization]]"]
models: ["[[模型库/EnCodec|EnCodec]]"]
tasks: []
datasets: ["[[数据集/LibriTTS|LibriTTS]]"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[概念库/SpeechLanguageModel|SpeechLM]]✓, [[概念库/SpeechTokenizer|SpeechTokenizer]]✓, [[概念库/SemanticvsAcousticTokens|SemanticvsAcousticTokens]]✓, [[概念库/LLM-basedTTS|LLM-basedTTS]]✓, [[概念库/CodecLanguageModel|CodecLM]](pending), [[概念库/StreamingSpokenDialogue|StreamingSpokenDialogue]](pending))
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: SpeechLM✓, SpeechTokenizer✓, SemanticvsAcousticTokens✓, LLM-basedTTS✓ | 过滤: CodecLM(pending-review), StreamingSpokenDialogue(pending-review), Full-duplexSpokenDialogue(pending-review), Turn-takinginSpokenDialogue(pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: DSM 来自 Moshi 同一团队 (Kyutai),直接继承了 Moshi 的 multi-stream decoder 架构 [Defossez et al. 2024]。但 DSM 的定位不同于 Moshi 的全双工对话——DSM 将 multi-stream 框架泛化为通用 streaming seq2seq 解法,通过 delay 方向切换即可得到 ASR 或 TTS。KB 中 SpeechLM 的演进链 GSLM→AudioLM→Moshi 可以延伸: Moshi (全双工对话) → DSM (通用 streaming seq2seq)。
>
> **Streaming Spoken Dialogue**: KB 记录了 Moshi 的 fully causal architecture (RQ-Transformer + 因果 Mimi codec) 实现 160ms 理论延迟。DSM 沿用相同的因果架构,但核心创新是 delay conditioning——单一模型在推理时可任意选择 delay 值来控制 latency-quality tradeoff,无需为每个 delay 重新训练。这是 KB 中尚未覆盖的新技术点。
>
> **Codec Language Model / Speech Tokenizer**: DSM 使用 Mimi codec (12.5Hz, 32 RVQ codebooks, cardinality 2048) 作为音频表征。不同于 Moshi 的 8 codebook,DSM-TTS 使用 32 codebook,并通过 RQ-Transformer sampler (4 层 per codebook, 共享权重 per group of 8) 建模 codebook 维度。KB 中 CodecLM 的 "多层 RVQ 建模" 挑战在此得到一个新的解法: 参数分组共享。
>
> **LLM-based TTS**: DSM-TTS 属于 codec LM 路线 (decoder-only + RVQ tokens),但与主流 LLM-based TTS 的关键区别是: (1) 真正流式——可处理任意长输入; (2) 可 batch——恒定帧率允许高效批处理; (3) 支持对话——通过 speaker conditioning + MAIN/OTHER turn tokens 实现可控多说话人对话。KB 中 FireRedTTS-2/VibeVoice 等对话 TTS 扩展也在探索类似方向,但都不是 streaming。

## 速查

> [!summary] 速查
> - **一句话**: 提出 Delayed Streams Modeling (DSM),通过将多模态序列预对齐到相同帧率并引入可控 delay,用标准 decoder-only Transformer 实现 streaming seq2seq,在 ASR 和 TTS 上同时达到 SOTA 级性能 [§1]
> - **路线**: [ASR] 24kHz audio → Mimi codec (12.5Hz, 32 RVQ) → audio stream (teacher-forced) + text stream (delayed, sampled) → Transformer backbone (2.6B) → text logits → transcription with timestamps [§3.2]; [TTS] text stream (input) + action stream (output, WORD prediction) + audio stream (output, delayed) → Transformer backbone (1B) + RQ-Transformer sampler → Mimi decode → 24kHz audio [§3.3, Fig 2-3]
> - **指标**: ASR avg WER 6.4% on OpenASR (streaming, vs best non-streaming 5.6%) [Table 1]; TTS avg WER 1.72% English long-form (vs Chatterbox 1.95%) [Table 3]; TTS speaker sim 76.46% English (vs Chatterbox 66.92%) [Table 4]; TTS throughput 137.3x real-time at batch 64 on H100 [Table 6]; ASR throughput 380x real-time at batch 256 [Table 10]; ASR latency precision ~300ms around target delay [§4.3.1]
> - **可借鉴**: (1) Delay conditioning: 将 delay 值通过 cosine embedding 注入输入,单模型覆盖全 delay 范围,且性能优于 fixed-delay 变体 [§3.2, Fig 4]; (2) Action stream: 额外输出流预测 "何时插入下一个词",解决 TTS 中输入文本缺乏时间对齐信息的问题 [§3.3]; (3) Lookahead text stream: 在主文本输入的同时前馈后续词内容,帮助模型做停顿决策 [§3.3]; (4) CFG distillation: 将 CFG 蒸馏为条件嵌入,使开源模型不需要暴露 unconditional 模式 [§4.2]
> - **局限**: 需要 word-level 时间对齐的训练数据 (靠 Whisper pseudo-labels) [§3.2]; TTS 主观质量低于 Chatterbox (54.8 vs 67.7 MUSHRA) [Table 5]; speaker encoder 未开源 (安全考量) [§4.2]; 公开版模型仅 0.75B 参数 [Appendix J]; 法语 TTS WER 偏高 (3.26%) [Table 3]

## 核心问题

DSM 要解决现有 decoder-only seq2seq 模型在流式场景下的三个根本矛盾 [§1]:

1. **前缀依赖阻碍流式推理**: 标准 decoder-only 方法将输入序列作为前缀拼接在输出序列前,必须获取完整输入才能开始生成 [§1]。这不仅阻碍实时推理,还从根本上限制了输入长度 [论文原文]。
2. **模态帧率不匹配**: 音频 token 以固定帧率采样 (如 12.5Hz),文本 token 代表语言单元,发音时长各异 [§1]。两种模态无法直接在同一时间轴上对齐,传统方法需要学习对齐策略 (如 Transducer),增加训练复杂度且不利于 batching [论文原文]。
3. **流式模型无法 batch**: 当使用推理策略 (policy) 决定何时读取输入、何时写入输出时,不同序列的步进不同步,无法批处理 [§1, §2]。这导致流式模型在实际部署中吞吐量极低 [论文原文]。

DSM 的核心洞察是: **将对齐从模型学习移到数据预处理,让所有模态共享同一帧率,然后用简单的 stream delay 控制 quality-latency tradeoff** [§1, 论文原文]。这使模型可以用标准的因果 attention 进行流式推理,天然支持 batching。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DSM 由三个组件构成 [§3, Fig 2]:

1. **Backbone**: 标准 Transformer decoder-only 模型,接收输入流和输出流的 embedding 之和,用因果 attention 处理 [§3]。可选配 cross-attention 层提供非流式上下文信息 (如 speaker embedding) [论文原文]。
2. **Input Embedder**: 将输入 X 和输出 Y 分别嵌入后求和,作为 backbone 的输入。对于离散域,使用 learnt embedding tables [§3]。
3. **Sampler**: 从 backbone 输出采样输出序列。对于离散输出,用 linear layer 计算 logits [§3]。对于多层 RVQ 的 TTS 输出,使用 RQ-Transformer (小型 Transformer,沿 codebook 维度自回归) [§3.3]。

**关键形式化** [§3, Eq 7]: 给定 delay tau, 模型估计:
```
q_tau(y | X_{<=t+tau}, Y_{<t}) ≈ P[Y_t = y | X_{<=t+tau}, Y_{<t}]
```
即在时间步 t 预测输出 Y_t 时,模型已经看到了输入 X 直到 t+tau。tau=0 为完全因果; tau=T 退化为传统前缀方法 [论文原文]。

**DSM-ASR**: backbone 2.6B (2048d, 48 层, 32 头) + text vocabulary 4000 + 固定 delay 2.5s (或可变 delay 配合 delay conditioning) [§4.1]。

**DSM-TTS**: backbone 1B (2048d, 16 层, 16 头) + RQ-Transformer sampler (1024d, 4 层/codebook) + text vocabulary 8000 (法英双语) + delay 1.28s (16 步) + speaker conditioning via cross-attention + 总参数 1.8B [§4.1]。

### 关键设计选择

**为什么 delay 能解决因果性问题?** [§3, Eq 5-7] 论文用一个 XOR 反例证明: 如果 Y_t 依赖于 X_{t+1} (未来信息),那么在流式采样时只能看到 X_{<=t},采样结果的分布会偏离真实条件分布 [§3]。引入 delay tau 使模型在采样 Y_t 时能看到 X_{<=t+tau},只要 tau 足够大让 Y_t 与 X_{>t+tau} 近似独立,流式采样就能逼近离线推理 [论文原文]。[agent 解读] 这本质上是 wait-k policy (Ma et al. 2019) 的泛化版本,但 DSM 通过预对齐到统一帧率,避免了 wait-k 需要学习 policy 的复杂性。

**为什么单一 delay conditioning 模型优于 fixed delay?** [§3.2, Fig 4] 训练时为每个序列随机采样 delay 值 (范围 [0.25s, 4s]),并通过 cosine embedding 将 delay (以 ms 为单位) 添加到输入 [论文原文]。结果显示 conditioned 模型在所有 delay 值上都优于对应的 fixed-delay 模型 [Fig 4 left]。[agent 解读] 这可能是因为多 delay 训练产生了类似数据增强的效果——模型在不同 delay 下看到相同数据的不同视角,有助于学习更鲁棒的 text-audio 依赖关系。

**为什么 TTS 需要 action stream?** [§3.3] DSM-TTS 的输入是文本流,但推理时新文本何时输入是未知的 (训练时可以 teacher-force ground-truth 时间对齐)。Action stream 是一个额外的输出流,预测下一个输入 token 是否为 WORD (新词开始) [论文原文]。当模型输出 WORD 时,系统将下一个词的 BPE tokens 依次喂入文本流 [论文原文]。[agent 解读] 这巧妙地将 duration prediction 问题转化为一个二分类任务,并将其与主生成模型联合训练,避免了传统 TTS 需要独立 duration predictor 的问题。

**为什么需要 lookahead text stream?** [§3.3] Action stream 让模型知道何时插入下一个词,但模型在做停顿/词位决策时不知道下一个词的内容。Lookahead stream 在主文本流输入第 i 个词时,同时输入第 i+l 个词的 tokens (l=2) [论文原文]。消融实验显示去除 lookahead 导致 WER 从 1.60% 大幅恶化到 3.51% [Table 15]。[agent 解读] 这说明停顿/节奏决策强烈依赖于后续文本内容——例如句末停顿 vs 句中连贯需要看后续词才能判断,这与人类阅读时的 eye-voice span 类似。

**Speaker conditioning 的设计** [§3.3]: 支持最多 5 个说话人,每人用 10s 音频提取 speaker embedding。Speaker encoder 架构复用 Mimi codec 的 encoder,卷积层冻结,Transformer 层端到端微调 [论文原文]。通过 cross-attention 注入 backbone,加 absolute positional embedding 区分不同说话人位置 [论文原文]。

**为什么对 CFG 做蒸馏?** [§4.2, Eq 8] 标准 CFG 需要模型能在无条件 (no text, no speaker) 下运行,这意味着开源模型可以不带 speaker embedding 运行,存在安全风险 [论文原文]。解决方案: 用 CFG 的输出作为 teacher 对模型做蒸馏,引入 alpha 条件嵌入 (alpha ∈ {1, 1.5, ..., 4}),微调 2400 步后模型不再需要 unconditional 模式 [§4.2]。[agent 解读] 这是一个实用的安全-质量平衡方案——保留了 CFG 的质量提升,同时阻止了无 speaker conditioning 的任意语音生成。

### 训练策略

**两阶段训练** [§4.2]:
1. **Pretraining**: 2.5M 小时公开音频 (英法) + whisper-timestamped 转写。ASR 在 48 H100 上训练 1.6M 步 (90s segments); TTS 在 32 H100 上训练 750k 步 (150s segments, batch 64) [§4.2]。
2. **Finetuning**: ASR 在 28k 小时公开数据集 (带 ground-truth 转写) 上微调 100k 步 [§4.2, Appendix A.1]; 然后长序列适应 25k 步 [§4.2, Appendix A.2]。TTS 做 CFG 蒸馏 2400 步 [§4.2]。

**预训练即 hard distillation** [§4.3.1]: 预训练阶段用 Whisper Medium 的 pseudo-transcripts,本质上是对 Whisper 的蒸馏。有趣的是 student (DSM-ASR) WER 7.1% 超越了 teacher (Whisper Medium) 8.1%。论文假设这来自 (a) 更大更多样的数据带来的 domain adaptation, (b) 低温采样消除了 teacher 的非系统性错误 [§4.3.1]。

**文本格式化** [§3.2]: Whisper 伪标签有标点大小写,但各公开数据集格式不一。解决方案: 训练一个 300M prefix-LM 做自动格式化,基于 Whisper 格式化转写训练 [§3.2]。Ground-truth 转写缺乏 word-level 时间戳的问题通过 Whisper + DTW 对齐解决 [§3.2]。

**codebook dropout** [§4.2]: 训练时随机 dropout 部分 RVQ levels (Defossez et al. 2024),使模型对不同数量的 codebook 鲁棒。消融显示 24 个以上的 codebook 对 WER 没有额外收益 [Appendix F]。

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (avg, streaming ASR) | **6.4%** | Parakeet-TDT-v2 7.0%, Whisper-Streaming 9.0% | OpenASR 8 datasets | Table 1 |
| WER (avg, non-streaming ASR) | 6.4% | Canary-Qwen 5.6%, Phi-4 6.1% | OpenASR 8 datasets | Table 1 |
| WER (avg, long-form ASR) | **7.9%** | Parakeet 7.8%, Whisper-L-v3 9.0% | TED/Meanwhile/Rev16/Earn21 | Table 2 |
| WER (English long-form TTS) | **1.72%** | Chatterbox 1.95%, Dia 2.79% | NTREX monologues + dialogs | Table 3 |
| WER (French long-form TTS) | **2.66%** | Dia 3.20% | NTREX monologues + dialogs | Table 3 |
| Speaker sim (English TTS) | **76.46%** | Chatterbox 66.92%, CSM 74.44% | VCTK test speakers | Table 4 |
| MUSHRA quality (English TTS) | 54.8 | Chatterbox **67.7**, ElevenLabs 65.0 | NTREX + dialog scripts | Table 5 |
| Speaker ELO (English TTS) | 2047.3 | Dia **2179.1**, Chatterbox 1894.3 | Pairwise comparison | Table 5 |
| TTS latency (b.s.=1) | 150ms | Dia 150ms, CSM 380ms | H100 | Table 6 |
| TTS throughput (b.s.=64) | **137.3x** | Chatterbox 1.8x, CSM 1.0x | H100 | Table 6 |
| ASR throughput (b.s.=256) | **380.1x** | - | H100 | Table 10 |
| WER (public-only TTS, LS clean) | **1.68%** | F5-TTS 2.42% | LibriSpeech test-clean | Table 7 |
| Speaker sim (public-only TTS) | **0.71** | F5-TTS 0.66 | LibriSpeech test-clean | Table 7 |

## 局限性

1. **主观质量不是最优**: MUSHRA 评分 54.8 低于 Chatterbox 67.7 和 ElevenLabs 65.0 [Table 5],说明 streaming 约束对感知质量有代价 [agent 解读]。
2. **依赖 word-level alignment 数据**: 训练需要精确的 word-level 时间戳,只能通过 Whisper pseudo-labels + DTW 获取,引入噪声 [§3.2]。这限制了可用训练数据的范围 [论文原文]。
3. **Speaker encoder 未开源**: 出于安全考虑,speaker encoder 保持闭源,开源版本仅提供预计算 speaker embeddings [§4.2, Appendix J]。
4. **公开模型规模受限**: 开源版仅 0.75B 参数 (300M backbone + 16 codebooks),speaker similarity 74.9% vs 主模型 80.9% [Appendix J]。
5. **水印脆弱性**: 论文发现 Mimi codec 一次编解码就能几乎完全去除现有水印 (AudioSeal 降至 45.2%,Perth/SilentCipher 降至 0%) [Table 13, Appendix K],TTS 安全保护仍是开放问题 [论文原文]。
6. **法语性能相对弱**: 法语 TTS WER 3.26% vs 英语 2.29% [Table 3],可能反映训练数据的语言不平衡 [agent 解读]。

## 点评

**核心贡献的意义**: DSM 的最大贡献不是某个单一技术创新,而是证明了一个简洁的范式转换——将对齐从模型内部学习移到数据预处理——可以同时解决 streaming/batching/长序列三个痛点 [§1]。这比 Transducer (需要学习 alignment policy, 不能 batch) 和传统 prefix decoder (不能 streaming) 都更优雅。

**与 Moshi 的关系**: DSM 本质上是将 Moshi 的 multi-stream 架构从"对话专用"泛化为"通用 streaming seq2seq"。Moshi 用固定 delay 做全双工对话 [Defossez et al. 2024],DSM 通过 delay 方向切换 (text delayed = ASR, audio delayed = TTS) 和 delay conditioning 扩展了这一框架的适用范围 [§1, §3.2]。可以说 DSM 是 Moshi 架构思想的系统性工程化。

**Delay conditioning 的价值**: 这是本文最有启发性的技术贡献。训练一个模型时随机 delay + cosine embedding conditioning,推理时指定任意 delay 值,且性能优于 fixed-delay 对应物 [Fig 4]。这类似于 classifier-free guidance 的思路——将一个原本需要多模型的选择变成单模型的条件推理 [agent 解读]。

**Action + Lookahead stream 的设计**: 这组设计体现了对 TTS 中 duration 控制问题的深刻理解。传统 TTS 用 duration predictor 显式预测时长,DSM 将其隐式编码为 action stream (何时插入下一个词) + lookahead (用后续内容辅助决策) [§3.3]。消融实验 [Table 15] 证明 lookahead 的影响巨大 (WER 1.60% → 3.51%),说明停顿决策对后续文本内容有强依赖。

**吞吐量优势**: DSM 的 batching 能力是其部署价值的关键差异化。TTS 在 batch 64 时达到 137.3x real-time throughput [Table 6],ASR 在 batch 256 时达到 380x [Table 10],比 Whisper-Streaming 高 100x [§4.3.1]。这来自恒定帧率设计——所有序列以相同速度推进,可以完美 batch [论文原文]。

**水印分析的诚实性**: 论文主动揭示了 Mimi codec 可以轻易移除现有水印 [Table 13],这种自我批判罕见且值得肯定。这也提醒了社区: 基于 codec 的 TTS 系统的安全保障需要根本性的新思路,而非依赖现有水印方案 [Appendix K]。

**不足**: 主观质量 (MUSHRA) 和 speaker similarity ELO 评分都不是最优 [Table 5],说明 streaming 约束确实牺牲了一定感知质量。公开模型只提供了降规格版本 (0.75B, 16 codebooks),限制了社区复现和验证。

## 可复用的 idea

1. **Delay conditioning 范式** [§3.2]: 将原本需要多个模型 (每个 delay 一个) 的选择变成单个模型的条件推理,通过 cosine embedding 注入 delay 值。这一思路可迁移到任何有 latency-quality tradeoff 的场景: 例如 streaming codec 的 lookahead 控制、simultaneous translation 的 wait-k 选择等。

2. **Action stream 解耦 duration prediction** [§3.3]: 不显式预测 phoneme/word duration,而是让模型输出"现在是否该接收下一个词"的二分类信号。这避免了 duration error 的累积,且与主生成模型端到端训练。可用于任何需要动态控制输入-输出节奏的 streaming 系统。

3. **CFG 蒸馏保安全** [§4.2]: 训练时用标准 CFG (需 unconditional mode),然后蒸馏为条件嵌入 (alpha 值作为输入),使最终模型无法在无 speaker conditioning 下运行。可应用于任何开源 voice-conditioned 生成模型的安全部署。

4. **RQ-Transformer 参数分组共享** [§4.1, Appendix J]: 前 8 个 codebook 各用独立参数,之后每 8 个 codebook 共享参数 (类似 Labiausse et al. 2025)。在 codebook 数量增多 (如 32) 时,这大幅减少参数量而不牺牲质量。可用于任何多层 RVQ 的生成模型。

5. **Student 超越 teacher 的蒸馏** [§4.3.1]: 用大量多样数据做 hard distillation (直接用 teacher 的 pseudo-labels 训练),student 可通过 (a) domain adaptation 和 (b) 隐式错误平滑超越 teacher。这为 streaming ASR 提供了一条低成本路径: 用强 offline 模型标注大量数据,训练的 streaming 模型甚至可以反超。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 因果解释充分,每个关键设计选择均回答了 WHY |
> | 可信赖 | pass | 数字 claim 均标注出处,指标名使用正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率 >90% |
> | 可定位 | pass | KB 背景定位到 Moshi 延伸线 + streaming 新技术点 |
> | 不污染 | pass | 未新建概念页,KB 信息准确 |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> 详见 `_review/DSM-review.yml`
