---
type: paper
tier: deep
title: "Moshi: a speech-text foundation model for real-time dialogue"
arxiv_id: "2410.00037"
source: "Sources/Moshi.pdf"
authors: [Alexandre Defossez, Laurent Mazare, Manu Orsini, Amelie Royer, Patrick Perez, Herve Jegou, Edouard Grave, Neil Zeghidour]
year: 2024
venue: "arXiv"
tags: [full-duplex, speech-LM, real-time, streaming, audio-codec, RVQ, inner-monologue, multi-stream, speech-to-speech]
concepts: ["[[Full-duplexSpokenDialogue]]", "[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[ResidualVectorQuantization]]", "[[SemanticvsAcousticTokens]]", "[[StreamingSpokenDialogue]]"]
models: ["[[模型库/EnCodec|EnCodec]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]], [[LLM-basedTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓, [[SemanticvsAcousticTokens]]✓, [[LLM-basedTTS]]✓ | 过滤: [[Full-duplexSpokenDialogue]](pending-review), [[StreamingSpokenDialogue]](pending-review), [[SpokenDialogueEvaluation]](pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: SpeechLM 是端到端处理和生成语音的自回归基础模型。Moshi 属于 SpeechLM 的前沿形态——首个全双工 real-time 对话系统。KB 已收录 GSLM→AudioLM→TWIST→SPIRIT-LM→SpeechGPT→Moshi 的演进链。
>
> **Speech Tokenizer**: KB 记录了三类 tokenizer 路线 (自监督/监督/声学)。Moshi 的 Mimi codec 属于 "Mixed Objective Tokenizer",通过 split RVQ 同时编码语义和声学信息,是 KB 中 SpeechTokenizer 之后的第二个混合路线代表。
>
> **Residual Vector Quantization**: RVQ 是现代 neural audio codec 的核心量化模块。Moshi 使用 8 层 RVQ (codebook 2048) 在 12.5Hz 实现 1.1kbps bitrate,并创新性地提出 split RVQ (1 VQ semantic + 7 RVQ acoustic) 缓解语义-声学冲突。
>
> **Semantic vs Acoustic Tokens**: 二分法是 SpeechLM 的核心 trade-off。Moshi 的 Mimi 通过 WavLM 蒸馏将语义信息注入第一层 VQ,后续层保留声学信息,属于 mixed tokenizer 的典型解法。Survey 评估显示 Mimi 在语义任务上接近 HuBERT,声学也不错。
>
> **LLM-based TTS**: Moshi 不是传统 TTS 系统,但其 Inner Monologue 机制允许通过调节 text-audio delay 衍生出 streaming TTS (4.7% WER on LibriSpeech test-clean),展示了统一架构的灵活性。

> [!summary] 速查
> - **一句话**: 首个全双工实时语音对话大模型,通过 Helium 7B 文本 backbone + Mimi 混合语义-声学 codec + RQ-Transformer 层级生成 + Inner Monologue 文本前缀,实现 160ms 理论延迟的 speech-to-speech 对话 [§1]
> - **路线**: 24kHz audio → Mimi encoder (causal conv + VQ/RVQ, 12.5Hz) → [text + semantic + 7 acoustic] tokens → Temporal Transformer (7B, Helium) + Depth Transformer (6L, 1024d) → Mimi decoder → 24kHz audio [§3.1, Fig 1]
> - **指标**: sWUGGY 74.8 / sBLIMP 59.9 (audio-only cold start, SOTA) [Table 7]; spoken QA: Web Q. 26.6 / LLaMA Q. 62.3 / Trivia QA 22.8 (全面超越 SpeechGPT/Spectron) [Table 8]; Mimi MUSHRA 81.0 (adversarial-only, vs RVQGAN 31.3) [Table 4]; streaming TTS WER 4.7% on LibriSpeech [§5.7]; dialogue cond. PPL 41.9 (超越 dGSLM cascaded 45.9) [Table 9]; 安全评分 83.05 (ALERT benchmark) [§6.1]
> - **可借鉴**: (1) Inner Monologue: text token 作为 audio token 的前缀,显著提升语言质量,且同一架构通过调节 text-audio delay 可派生 streaming ASR/TTS [§3.4.4]; (2) Split RVQ: 1 VQ (semantic) + 7 RVQ (acoustic) 解耦语义-声学冲突 [§3.3.2]; (3) Depthwise parametrization: Depth Transformer 的每个 codebook level 使用独立参数,减少层间竞争 [§3.4.1]; (4) Adversarial-only training: 去除重建 loss 反而大幅提升感知音质 [§3.3]
> - **局限**: 仅支持英语 [§1]; 安全评分中等 (83.05 vs GPT-4 99.98) [§6.1]; 量化到 4-bit 时 MMLU 下降 5-10 点 [§5.8]; signal-based watermarking 不可用 (codec 非幂等) [§6.4]; 训练需多阶段 (text pretrain → audio pretrain → multi-stream post-train → Fisher finetune → instruct finetune) [Table 1]; 全部开源 (github.com/kyutai-labs/moshi)

## 核心问题

Moshi 要解决传统语音对话系统的三大根本局限 [§1]:

1. **高延迟**: 传统 ASR→LLM→TTS 级联管线造成数秒延迟,而人类对话中自然响应时间仅约 230ms [论文原文]
2. **信息瓶颈**: 以文本作为中间模态,丢失情感、语气、环境声等非语言信息 [论文原文]
3. **Turn-based 假设**: 传统系统将对话分割为明确的说话人轮次,无法处理重叠语音 (10-20% 的对话时间)、打断和回传信号 [论文原文, 引用 Cetin and Shriberg 2006]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Moshi 是一个 multi-stream speech-to-speech Transformer 模型,由四个核心组件构成 [§3.1, Fig 1]:

1. **Helium** (7B text LLM): 基于 Transformer 的自回归文本语言模型,作为 Temporal Transformer backbone,提供知识和推理能力 [§3.2]
2. **Mimi** (neural audio codec): 混合语义-声学 tokenizer,将 24kHz 音频编码为 12.5Hz 的离散 token 序列 (1 semantic VQ + 7 acoustic RVQ) [§3.3]
3. **RQ-Transformer**: Temporal Transformer (大,7B) + Depth Transformer (小,6 层 1024d) 的层级架构,在每个时间步建模 text + semantic + acoustic tokens [§3.4.1]
4. **Inner Monologue**: 在 Moshi 的输出流中,先预测时间对齐的 text tokens,再预测 audio tokens,text 作为 audio 的前缀 scaffolding [§3.4.4]

**Multi-stream 设计** [§3.4.3]: 将用户和 Moshi 的音频流拼接为一个联合序列 V,包含 2Q+1 个子序列 (Q=8 codebook levels): Moshi 的 text + semantic + 7 acoustic + 用户的 semantic + 7 acoustic,共 17 个子序列 [Eq. 6, Fig 4]。这完全移除了显式 speaker turn 的概念——模型可以在任何时刻同时说话和倾听 [论文原文]。

### 关键设计选择

**为什么用 RQ-Transformer 而不是 flat sequence?** 以 Q=8 codebook 在 12.5Hz 帧率,建模 5 分钟音频需要 30,000 timesteps;如果展平为单一序列则需 100 tokens/s,计算和推理都不可行 [§3.4.1]。RQ-Transformer 将序列分解为 Temporal (时间轴,S 步) + Depth (codebook 轴,最多 K 步),使 Temporal Transformer 的步数不变 (= S),Depth Transformer 的步数很少 (= K),实现流式推理兼容 [论文原文]。

**Acoustic delay 的作用** [§3.4.2]: 在 semantic token 和 acoustic token 之间引入 delay τ=1-2 步 (80-160ms)。这让 Temporal Transformer 可以在生成 acoustic tokens 之前先看到 semantic token 的信息,从而建模 semantic-acoustic 的依赖关系。消融实验显示 delay=2 (160ms) 时 perplexity 从 135.4 降至 36.8 [Table 5] [论文原文]。

**为什么用 Split RVQ?** [§3.3.2] 标准 RVQ 中,将语义信息蒸馏到第一层会与声学重建 loss 冲突——提升 ABX (语义) 必然降低 MUSHRA (声学)。Split RVQ 将量化分为独立的 1 VQ (语义,蒸馏 WavLM) + 7 RVQ (声学),两者的输出求和用于重建,解除了语义量化器必须保留声学残差的约束 [论文原文]。结果: MUSHRA 从 57.8 提升到 64.0,ABX 仅从 6.5% 升到 8.1% [Table 3] [论文原文]。

**为什么用 adversarial-only training for Mimi?** [§3.3] 去除重建 loss (multi-scale mel-spectrogram + STFT) 后只保留 feature loss + discriminator loss,客观指标 VisQOL 急剧下降 (2.82→1.84),但 MUSHRA 从 58.8 大幅提升到 81.0。论文指出这揭示了客观指标与人类感知的严重脱节 [论文原文]。[agent 解读] 这可能是因为重建 loss 过度约束了频谱细节,而对抗训练让 codec 学会生成感知上更自然的音频,即使频谱不完全匹配。

**Depthwise parametrization** [§3.4.1]: Depth Transformer 的每个 codebook level k 使用独立的 linear layers 和 projection 参数,而非共享权重。原因是不同子序列 (text vs semantic vs acoustic) 可能需要不同的变换 [论文原文]。消融显示这对质量有益且不影响速度 [Table 6]。

**Inner Monologue 的三重价值** [§3.4.4]:
1. **语言质量**: 启用 Inner Monologue 后 transcript NLL 从 3.65 降至 2.77,transcript 长度从 602 暴增到 1920 (模型不再坍缩到沉默) [Table 6] [论文原文]
2. **Streaming ASR/TTS 派生**: 通过调节 text-audio delay,同一模型可用作 streaming ASR (text 在 audio 之后) 或 streaming TTS (text 在 audio 之前) [§3.4.4] [论文原文]
3. **推理成本极低**: 每个时间步仅多预测 1 个 text token (17 tokens vs 16 without IM),近似无额外成本 [§5.5] [论文原文]

### 训练策略

Moshi 的训练分为 4 个阶段 [§4.4, Table 1]:

1. **Helium text pre-training**: 7B 参数,2.1T tokens,500k steps,CommonCrawl + Wikipedia + StackExchange 等高质量数据 [§3.2, §4.1]
2. **Moshi audio pre-training**: 从 Helium 初始化 Temporal Transformer,随机初始化 Depth Transformer,在 7M 小时无监督音频上训练 1M steps,单流模式,30% 概率 mask text tokens,text-audio delay 随机化 ±0.6s [§4.4]
3. **Multi-stream post-training**: 用 PyAnnote 说话人分离模拟双流数据训练 100k steps,再在 Fisher 数据集 (2000h 真实双人对话) 上 finetune 10k steps [§4.4]
4. **Instruction finetuning**: 合成 20k+ 小时对话指令数据,用 Helium finetune 在 Open Hermes 上的版本生成文本脚本,用 streaming TTS 合成语音,30k steps [§4.3, §4.4]

**关键训练技巧**:
- Loss 权重: text token 权重 = 1,semantic token 权重 = 100,acoustic token 权重 = 1,解决 RVQ level 间竞争 [§4.4, Eq. 7] [论文原文]
- 音频预训练期间保持 50% 纯文本 batch 防止灾难性遗忘 [§4.4]
- 指令微调阶段对用户流施加数据增强: 随机增益 (-24 to +15dB)、30% 噪声注入、echo 模拟、混响 [§4.4]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| sWUGGY (audio LM) | 74.8 | TWIST 74.5 / Spirit-LM 69.5 | Librispeech | [Table 7] |
| sBLIMP (audio LM) | 59.9 | AudioLM 64.7 / Spirit-LM 58.0 | Librispeech | [Table 7] |
| sTopic-StoryCloze | 80.9 | TWIST 76.4 / Spirit-LM 72.9 | Librispeech | [Table 7] |
| MMLU (text understanding) | 49.8 | Spirit-LM 36.9 | - | [Table 7] |
| Web Questions (spoken QA) | 26.6 | SpeechGPT 6.5 / Spectron 6.1 | Web Questions | [Table 8] |
| LLaMA Questions (spoken QA) | 62.3 | SpeechGPT 21.6 / Spectron 22.9 | LLaMA-Questions | [Table 8] |
| Trivia QA (spoken QA) | 22.8 | SpeechGPT 14.8 | TriviaQA | [Table 8] |
| Mimi MUSHRA (audio quality) | 81.0 | RVQGAN 31.3 / SpeechTokenizer 74.3 | Librispeech | [Table 4] |
| Mimi ABX (phonetic discrim.) | 8.1% | SpeechTokenizer 3.3% | Librispeech | [Table 4] |
| Dialogue cond. PPL | 41.9 | dGSLM cascaded 45.9 | Fisher | [Table 9] |
| Streaming TTS WER | 4.7% | VALL-E 5.9% | LibriSpeech test-clean | [§5.7] |
| Streaming ASR WER | 5.7% | FastConformer 3.6% | LibriSpeech test-clean | [§5.7] |
| Speaker consistency | 98.7% | - | 100h generated | [§6.3, Table 14] |
| ALERT safety score | 83.05 | GPT-4 99.98 / Llama 2 99.18 | ALERT | [§6.1] |

## 局限性

1. **仅英语**: 当前仅支持英语对话,多语言需重新训练全部组件 [§1]
2. **知识遗忘**: 音频训练导致 MMLU 从 54.3 (Helium) 降至 49.8,Trivia QA 差距更大 (22.8 vs 56.4 for Helium text) [Table 7, Table 8] [论文原文]
3. **安全性中等**: ALERT 评分 83.05,低于 GPT-3.5 (96.95) 和 GPT-4 (99.18) [§6.1]
4. **量化敏感**: 4-bit 量化时 MMLU 下降约 5 点,3-bit 时出现 gibberish/repetitive 退化 [§5.8, Table 11-12]
5. **Watermarking 困难**: Signal-based watermarking 被 codec re-encoding 完全移除 (Mimi codec 非幂等); generative watermarking 因 codec token 的时间漂移也不可靠 [§6.4, Table 15-16]
6. **Objective-perceptual 脱节**: VisQOL/MOSNet 等客观指标与 MUSHRA 人类评估严重不相关 [§5.2]

## 点评

Moshi 是语音对话领域的里程碑式工作,其贡献在于系统性地解决了全双工对话的完整技术栈:

**最核心的贡献是 Inner Monologue**: 这个看似简单的设计——在每个时间步先预测 text token 再预测 audio tokens——带来了三重价值: (1) 大幅提升语言质量 (spoken QA 精度接近 3 倍提升); (2) 统一了 ASR/TTS/dialogue 三个任务到同一架构; (3) 几乎零额外推理成本。这种"text as scaffold for speech"的思路对后续工作有深远影响 [agent 解读]。

**Split RVQ 是务实的工程创新**: 面对语义蒸馏与声学重建的根本冲突,Moshi 选择物理解耦而非 loss 权重调优,这种解法简洁有效,且被 KB 中已有的 Mimi 条目验证为 mixed tokenizer 的代表方案 [agent 解读]。

**值得注意的局限**: Moshi 的训练极其复杂 (5 阶段),数据需求量大 (7M 小时音频 + 2.1T text tokens),且安全性和知识保留方面仍有明显差距。这些问题在后续 Step-Audio 系列中被部分解决。

## 可复用的 idea

1. **Inner Monologue**: text token 作为 audio token 的 per-timestep prefix,同一模型衍生 ASR/TTS/dialogue,适用于任何需要提升 speech generation 语言质量的系统
2. **Split RVQ**: 独立 VQ (语义) + RVQ (声学) 的物理解耦,适用于需要同时优化语义识别和声学重建的 codec 设计
3. **Adversarial-only codec training**: 去除重建 loss 大幅提升感知音质,值得在其他 codec 中验证
4. **Depthwise parametrization**: Depth Transformer 中不同 codebook level 使用独立参数,适用于任何 RQ-Transformer 架构
5. **Acoustic delay 模式 [0,2,2,2,2,2,2,2]**: semantic 和 acoustic 之间引入固定 delay,让大模型先建模语义再建模声学,适用于任何层级 codec token 生成


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排

## 代码级分析

> [!info] 代码来源
> - 仓库: https://github.com/kyutai-labs/moshi
> - commit: e6a55d2722a65870ef52a6c9f6ecfc0e90f38362
> - 分析日期: 2026-06-10

### 架构验证

**论文 vs 代码的关键对应:**

1. **LMModel (Temporal Transformer)** (`moshi/moshi/models/lm.py:49-520`): 论文中的 Helium 7B backbone。代码中 `StreamingTransformer` 使用 4096 dim, 32 heads, 32 layers, GLU (SiLU gating), RMS norm, RoPE。与论文 Table 1 的 Helium 参数完全一致。

2. **Depformer (Depth Transformer)** (`lm.py:204-217`): 论文中的 Depth Transformer。代码中使用独立的 `StreamingTransformer`,1024 dim, 16 heads, 6 layers, dim_feedforward=4224。与论文完全一致。

3. **Depthwise parametrization** (`lm.py:173-182`): 论文 3.4.1 提出的每个 codebook 使用独立参数。代码通过 `depformer_multi_linear=True` 实现:每个 codebook index 有独立的 `nn.Linear(dim, depformer_dim)` 投影层。Depformer 本身通过 `weights_per_step=True` 支持每步独立权重。

4. **Mimi Codec** (`moshi/moshi/models/compression.py:105-434`): 论文中的 Mimi neural audio codec。代码结构: SeaNet encoder → Transformer bottleneck (encoder side) → downsample → SplitRVQ (1 VQ + 7 RVQ) → upsample → Transformer bottleneck (decoder side) → SeaNet decoder。支持全因果流式处理。

5. **Split RVQ** (`moshi/quantization/`): `SplitResidualVectorQuantizer` 将量化分为 `rvq_first` (1 层 VQ, 语义) 和 `rvq_rest` (7 层 RVQ, 声学),两者输出求和用于重建。与论文 3.3.2 一致。

6. **Multi-stream 设计** (`lm.py:113-120`): `n_q=16` (8 Moshi codebooks + 8 user codebooks),`dep_q=8` (Moshi 的 codebook),文本流作为 `audio_offset=1` 之前的第 0 通道。总共 17 个子序列 (1 text + 16 audio),与论文 Eq. 6 一致。

7. **Delay pattern** (`moshi_7b config`): `delays = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1]`。第 0 位是 text (delay=0),第 1 位是 Moshi semantic (delay=0),第 2-8 位是 Moshi acoustic (delay=1),第 9 位是 user semantic (delay=0),第 10-16 位是 user acoustic (delay=1)。与论文 Eq. 4 的 tau=1 一致 (论文训练用 tau=2,但实际部署配置用 tau=1)。

### 论文未写的实现细节

1. **CUDAGraphed 推理优化** (`lm.py:527,631`): 推理时 `forward_text` 和 `depformer_step` 都被包装为 `CUDAGraphed` 对象,使用 CUDA Graph 消除 kernel launch overhead,这对流式实时推理至关重要。

2. **Streaming 状态管理** (`lm.py:522-535,556-666`): `_LMGenState` 管理一个循环缓存 (`cache`),大小为 `max_delay + 2`,通过 `offsets % CT` 实现环形读写。支持 `exec_mask` 控制 batch 中哪些样本在执行,`reset_mask` 支持动态重置单个样本的流式状态。

3. **Depformer streaming detached** (`lm.py:218`): Depformer 的 streaming 与 Temporal Transformer 解耦 (`set_streaming_detached(True)`)。每个时间步内,Depformer 独立地打开自己的 streaming context 做 K 步前向 (`lm.py:823-847`)。

4. **Special token 设计** (`lm.py:246-278`): `zero_token_id = -1` (不参与嵌入), `ungenerated_token_id = -2` (标记待生成位置,允许部分 teacher forcing), `initial_token_id = card` (序列起始)。text 和 audio 有独立的 initial/padding token。

5. **CFG 实现** (`lm.py:714-732`): CFG 通过将 batch 翻倍实现 (`input_ = input_.repeat(2, 1, 1)`)。支持两种 CFG 模式: (1) 基于 condition 的标准 CFG (使用 `condition_sum`); (2) 基于 mask 的延迟 CFG (`cfg_is_masked_until`),在前若干步将 input 置零; (3) text-only CFG (`cfg_is_no_text`)。

6. **Mimi 的 CUDA Graph 支持** (`compression.py:219-230`): Mimi 的 encoder、decoder、encoder_transformer、decoder_transformer 都在 streaming 模式下被独立包装为 CUDAGraphed,实现极低延迟的编解码。

7. **Quantizer dropout 实现**: Mimi 训练时 50% 概率跳过量化 (`compression.py:305-306`),直接将未量化的 embedding 传给 decoder。这个细节在论文中只简略提及。

8. **Rust 后端** (`rust/`): Moshi 提供完整的 Rust 推理后端 (`moshi-backend/`),包含流式 WebSocket server、音频处理、benchmark 工具。Mimi 也有 Rust+PyO3 绑定 (`mimi-pyo3/`)。

### 训练 pipeline 拆解

```
阶段 1 (Helium text pre-training):
  Text tokens (SentencePiece 32K) → Transformer (32L, 4096d) → text logits
  500K steps, batch 4.2M tokens, cosine LR 3e-4

阶段 2 (Audio pre-training):
  24kHz mono audio → Mimi encode → [text + 8 semantic/acoustic tokens] per timestep
  Temporal Transformer (from Helium) + Depth Transformer (random init)
  1M steps, batch 16h audio, 50% text-only batches
  Text delay randomized [-0.6, +0.6]s, text masked 30%

阶段 3 (Multi-stream post-training):
  PyAnnote diarization → simulated 2-stream data
  Joint sequence: V = [text, Moshi_sem, Moshi_aco x7, User_sem, User_aco x7]
  100K steps, batch 8h, fixed LR 3e-6

阶段 4 (Fisher finetuning):
  Fisher 2000h real 2-speaker conversations (8kHz → AudioSR → 24kHz)
  10K steps, batch 40min

阶段 5 (Instruction finetuning):
  20K+ hours synthetic conversations (Helium+Open Hermes → text → streaming TTS)
  30K steps, single actor voice for Moshi stream
```

### 推理 pipeline 拆解

```
每个时间步 (80ms = 1920 samples @ 24kHz):
  User audio (24kHz) → Mimi.encode → 8 user tokens
  ↓
  LMGen._step():
    1. 写入 user tokens 到 cache (带 delay)
    2. 从 cache 读取当前步的 17 通道 input
    3. forward_text: input → sum(emb[k]) → Temporal Transformer → text_logits
    4. sample text_token (top_k=25, temp=0.7)
    5. depformer_step: 依次生成 8 个 audio tokens
       for k in range(8):
         [text/prev_token] emb + depformer_in[k](transformer_out) → Depformer → linear → sample
       (top_k=250, temp=0.8)
    6. 写入 text_token + 8 audio tokens 到 cache
    7. 从 cache 读取 undelayed output (text + 8 Moshi tokens)
  ↓
  Mimi.decode(8 tokens) → 24kHz waveform chunk
```

### 关键超参数表

| 参数 | 论文值 | 代码实际值 (config) | 备注 |
|------|--------|-----------|------|
| Temporal dim | 4096 | 4096 | 一致 |
| Temporal layers | 32 | 32 | 一致 |
| Temporal heads | 32 | 32 | 一致 |
| Hidden scale | 4 (论文) | 4.125 | 略有差异 (dim_ff=16896) |
| Depth dim | 1024 | 1024 | 一致 |
| Depth layers | 6 | 6 | 一致 |
| Depth dim_ff | 未明确 | 4224 | 代码特有 |
| Depth heads | 16 | 16 | 一致 |
| Audio codebooks | 8 (Q) | n_q=16, dep_q=8 | 16=8 Moshi + 8 User |
| Codebook size | 2048 | 2048 | 一致 |
| Frame rate | 12.5Hz | 12.5Hz | 一致 |
| Text vocab | 32000 | 32000 | 一致 |
| Context (Temporal) | 3000 steps | 3000 | =4 min audio |
| Context (Depth) | 未明确 | 8 | 仅看当前步 |
| Acoustic delay | tau=2 (论文) | tau=1 (config) | 部署用 tau=1 |
| Position embedding | RoPE | RoPE (Temporal), none (Depth) | Depth 无位置编码 |
| Norm | RMS Norm | rms_norm_f32 | 一致 |
| Gating | SiLU (GLU) | SiLU | 一致 |
| Sampling temp (audio) | 未明确 | 0.8 | 代码默认 |
| Sampling top_k (audio) | 未明确 | 250 | 代码默认 |
| Sampling temp (text) | 未明确 | 0.7 | 代码默认 |
| Sampling top_k (text) | 未明确 | 25 | 代码默认 |

### 复现 checklist (基于代码)

- [ ] 环境依赖: Python 3.10+, PyTorch 2.0+, `pip install moshi` (Python) 或 `cargo build` (Rust)
- [ ] 数据准备: (1) 文本预训练: CommonCrawl + Wikipedia + StackExchange; (2) 音频: 7M hours (未公开数据集); (3) Fisher: 需购买 LDC 许可; (4) Instruct: 需自行合成
- [ ] 预训练模型依赖: Mimi codec 权重 (公开), Helium 7B (公开), Moshi 完整模型 (公开, ~7B)。也有 2B 开发版。
- [ ] 训练命令: 论文/代码未提供训练脚本 (仅推理代码开源)
- [ ] 推理命令: `python -m moshi.server` (Python server) 或 `cargo run --release` (Rust server); 客户端: `moshi.client` 或 Web UI
- [ ] 已知坑: (1) 训练代码未开源,仅推理可用; (2) 仅支持英语; (3) 4-bit 量化会降低质量; (4) Mimi codec 非幂等,watermarking 不可用

### 代码质量与可复现性评估

- **工程质量**: 5/5 -- 多语言实现 (Python + Rust + MLX), 流式架构设计精良, CUDA Graph 优化, 完善的 WebSocket server/client。代码有完整类型标注和文档字符串。
- **文档完善度**: 4/5 -- README 清晰, 有 FAQ, 安装文档完善。但训练流程完全缺失。
- **社区活跃度**: 4/5 -- 6K+ stars, Kyutai 团队维护, issue 响应较好。414 PRs (主要是 Kyutai 内部贡献)。
- **复现难度**: 4/5 (难) -- 推理部署 2/5 (容易); 但训练从零复现 5/5 (极难): 训练代码未公开, 7M hours 音频数据不可用, 多阶段训练极其复杂, Fisher 数据需商业许可。
