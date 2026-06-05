---
type: paper
tier: deep
title: "LLaMA-Omni 2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis"
arxiv_id: "2505.02625"
source: "Sources/LLaMA-Omni2.pdf"
authors: [Qingkai Fang, Yan Zhou, Shoutao Guo, Shaolei Zhang, Yang Feng]
year: 2025
venue: "arXiv"
tags: [speech-LM, streaming, real-time, autoregressive, modular-SpeechLM, speech-interaction]
concepts: ["[[SpeechLanguageModel]]", "[[StreamingSpokenDialogue]]", "[[ConditionalFlowMatching]]", "[[SpeechTokenizer]]", "[[FiniteScalarQuantization]]", "[[ModalityAdaptationforSpeechLLM]]", "[[CodecLanguageModel]]"]
models: ["[[CosyVoice2]]", "[[Whisper]]", "[[SenseVoice]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: LLaMA-Omni 2 属于 **modular SpeechLM** 路线 — 在已有文本 LLM 外围挂载语音编码器和语音解码器,以较低训练成本获得语音交互能力。这条路线与 **native SpeechLM** (如 Moshi, GLM-4-Voice) 的全量预训练路线形成对比。根据 [[SpeechLanguageModel]] 的分类体系,LLaMA-Omni 2 属于 "instruction-tuning" 阶段 + "real-time interaction" 生成范式。

**已有认知**:
- **前作 LLaMA-Omni** 是该组之前的工作,使用 NAR CTC 流式语音解码器实现同步 text+speech 生成,但 NAR 模型的建模能力限制了语音自然度 ([[StreamingSpokenDialogue]])。
- **CosyVoice 2** ([[CosyVoice2]]) 提出了 FSQ-SenseVoice tokenizer + AR TTS LM + chunk-aware causal flow matching 的流式 TTS 方案,本文的语音解码器直接采用这一架构。CosyVoice 2 已验证 text-based LLM 初始化对 TTS LM 的有效性,以及 Read-Write 策略在流式合成中的作用。
- **Conditional Flow Matching** ([[ConditionalFlowMatching]]) 用于从离散 speech token 恢复连续 mel spectrogram,在 CosyVoice 系列中已成为标准 fine-stage 渲染器。
- **FSQ** ([[FiniteScalarQuantization]] [待确认]) 是 CosyVoice 系列使用的量化方法,无需 codebook 维护,天然避免 collapse,100% 利用率。
- **Speech Tokenizer** ([[SpeechTokenizer]]) 在 CosyVoice 2 中通过将 FSQ 插入 SenseVoice-Large ASR 编码器实现,生成 25Hz 的监督式 semantic token。

**创新判断**: 本文的核心新贡献不在于各模块(均复用已有组件),而在于 (1) 将 CosyVoice 2 的流式 TTS 解码器整合进 modular SpeechLM 框架; (2) 提出 gate fusion 机制融合 LLM hidden states 和文本 embedding; (3) 系统性消融(LLM 规模/训练数据/预训练策略/Read-Write 比例)为该架构提供了工程选型依据。

> 检索命中: [[SpeechLanguageModel]]✓, [[CosyVoice2]]✓, [[ConditionalFlowMatching]]✓, [[SpeechTokenizer]]✓ | 过滤: [[StreamingSpokenDialogue]](pending-review), [[FiniteScalarQuantization]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 CosyVoice 2 的 AR 流式语音解码器嫁接到 Qwen2.5 LLM 上,用 200K 合成多轮对话数据训练出 0.5B-14B 的 modular SpeechLM 系列,以远少于 native SpeechLM 的数据量超越 GLM-4-Voice
> - **路线**: Speech → Whisper encoder → 5x downsample adapter → Qwen2.5 LLM → Gate Fusion(LLM hidden + text emb）→ AR TTS LM (Qwen2.5-0.5B init) → chunk-aware causal flow matching → HiFi-GAN → Speech
> - **指标**: SpokenQA Llama Questions S2S 62.7% (14B) vs GLM-4-Voice 49.0%; ChatGPT Score S2S 4.35 (14B) vs GLM-4-Voice 3.52; ASR-WER 3.89 (14B) vs GLM-4-Voice 5.95; Latency ~600ms [Table 1]
> - **可借鉴**: Gate fusion 融合 hidden states + text embedding 的方案可推广到任何需要从 LLM 中间表征驱动下游模块的场景; Read-Write 比例消融提供了延迟-质量 trade-off 的具体参考
> - **局限**: 不支持情感/语速等副语言控制; 仅用合成数据训练,声音多样性有限; 600ms 延迟仍高于 LLaMA-Omni (347ms) 和 Moshi (230ms); 代码开源但训练数据未开源

## 核心问题

LLaMA-Omni 的 NAR CTC 解码器虽然延迟极低(~347ms),但非自回归模型的建模能力有限,导致生成语音不够自然流畅 [§1]。Freeze-Omni 结合了 NAR+AR 模型提升了自然度,但只能做句子级流式(依赖简单分句策略),无法实现真正的低延迟流式 [§1]。

**本文要解决的问题**: 如何在保持低延迟实时交互的前提下,用 AR 模型替代 NAR 解码器以提升语音自然度?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

LLaMA-Omni 2 是一个 modular SpeechLM,以 Qwen2.5 系列为中心 LLM,外挂三个模块 [§2, Fig 1]:

1. **Speech Encoder** (Whisper-large-v3 encoder): 将输入语音编码为连续表征序列
2. **Speech Adapter**: 5x 下采样 + FFN(中间维度 2048),将语音表征映射到 LLM 的 embedding space
3. **Streaming Speech Decoder**: 三级管线 — (a) Gate Fusion 融合 LLM 输出; (b) AR TTS LM 生成 speech token; (c) Chunk-aware Causal Flow Matching + HiFi-GAN 合成波形

### 关键设计选择

#### 1. Gate Fusion 模块: 为什么不直接用 hidden states?

TTS LM 的输入来自 LLM 的两路信息: hidden states H (包含上下文信息) 和 text tokens Y^T (精确的文本内容) [论文原文, §2.2]。

作者用 element-wise gate 自适应融合两路信息:
- Hidden states 经 2 层 FFN 映射到 TTS LM 的 embedding 维度: `e_hidden = FFN(h_i)` [Eq. 1]
- Text tokens 通过 TTS LM 的 embedding 层: `e_emb = Emb(y_i^T)` [Eq. 2]
- Gate: `g = σ(W_g [e_hidden ∥ e_emb] + b_g)` [Eq. 3]
- 融合: `c = g ⊙ e_hidden + (1-g) ⊙ e_emb` [Eq. 4]

**为什么这样设计** [论文原文, §2.2]: hidden states 含有上下文(语境)信息,text embedding 提供精确的文本内容对齐。Gate 让模型学会在需要语境时侧重 hidden states,需要精确发音时侧重 text embedding。

**消融验证** [Table 2]: 去掉 gate fusion(简单相加)→ ChatGPT Score 4.15→4.02, ASR-WER 3.26→4.89; 进一步去掉 text embedding(仅用 hidden states)→ Score 3.88, WER 6.83。证明两路信息和自适应融合都不可或缺。

#### 2. Read-Write 策略: 如何实现流式?

直接复用 CosyVoice 2 的 Read-R-Write-W 策略: 每读入 R 个 fused representations,生成 W 个 speech tokens [论文原文, §2.2]。推理时 LLM 每生成 R 个 text token,就驱动 TTS LM 生成 W 个 speech token 并立即合成一个语音 chunk [§2.4]。

**R 和 W 对性能的影响** [Table 4]:
- **W 决定语音质量**: W 是 flow matching 的 chunk size,W 越大 UTMOS 越高(W=5→3.98, W=20→4.46),因为更大 chunk 给 flow matching 更多上下文 [论文原文, §5.2]
- **R 和 W 共同决定延迟**: 首个 speech chunk 的延迟 = T_LLM(R) + T_TTS(W) + T_FM(W) + T_Voc(2W) [Eq. 6]
- **R=3, W=10 是最佳 trade-off**: ASR-WER 最低(3.26), UTMOS 4.19, 延迟 583ms [论文原文, §5.2]

**为什么 R=3 W=10 的 WER 最低** [agent 解读]: R=3 对应 3 个 text token 驱动 10 个 speech token,比例约 3.3:1,接近中英文的自然 speech-text duration ratio,可能有利于对齐。过大或过小的比例都会导致对齐失配。

#### 3. TTS LM 初始化: 为什么用 Qwen2.5-0.5B?

TTS LM 用 Qwen2.5-0.5B 初始化,词表扩展为 V' = V ∪ {<i> | 0 ≤ i < 6561} 以支持 speech token 生成 [§2.2]。

**预训练策略消融** [Table 3]:
- Streaming TTS 预训练(Stage I(b))→ Score 4.15, WER 3.26
- Offline TTS 预训练 → Score 4.13, WER 3.51 (略降)
- 仅 text pretrained (不做 TTS 预训练) → Score 3.53, WER 10.34 (大幅退化)
- 从零训练 → Score 1.08, WER 80.65 (完全不收敛)

**为什么需要 TTS 预训练** [论文原文, §5.2]: TTS 预训练让模型先学会 text→speech token 的映射关系,这是后续 Stage II 联合训练的必要基础。直接用文本预训练的 LM 无法有效建模 speech token 的分布。

**为什么流式优于离线** [agent 解读]: 流式预训练让模型从一开始就适应 Read-Write 的交替模式,建立增量条件生成的能力;离线预训练则需要在 Stage II 重新适应,带来 gap。

### 训练策略

两阶段训练,仅用 200K 合成多轮对话数据 [§2.3]:

**Stage I(a) — Speech-to-Text**: 冻结 speech encoder,训练 adapter + LLM。Batch 32, 3 epochs, lr 5e-5。
**Stage I(b) — Text-to-Speech**: 单独训练 TTS LM (此时 gate fusion 不训练,仅用 text embedding 输入)。Batch 32, 5 epochs, lr 5e-4。
**Stage II — Speech-to-Speech**: 冻结 encoder + adapter + LLM,仅训练 gate fusion + TTS LM。Batch 32, 1 epoch, lr 1e-3。

**为什么分两阶段** [论文原文, §2.3]: Stage I 让各模块分别学会基本能力(语音理解和语音生成),Stage II 专注于端到端的语音对话能力。Stage II 冻结 LLM 是为了保留其文本能力。

**训练数据构造** [§3]: 基于 InstructS2S-200K 扩展为多轮对话。用 Llama-3.3-70B 生成多轮文本对话(Poisson(λ=2) 抽样轮数,clip 到 1-5),然后用 fish-speech-1.5 + CosyVoice2-0.5B 合成语音。指令侧用随机语音(diversity),响应侧用统一语音(consistency)。

**多轮 vs 单轮** [Table 5]: 相同数据量下,多轮对话训练在所有指标上优于单轮 [论文原文, §5.3]。

## 实验

| 指标 | LLaMA-Omni2-7B | LLaMA-Omni2-14B | GLM-4-Voice (9B) | LLaMA-Omni (8B) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama Questions S2T Acc | 70.3 | 73.0 | 64.7 | 67.7 | Llama Questions | [Table 1] |
| Llama Questions S2S Acc | 60.7 | 62.7 | 49.0 | - | Llama Questions | [Table 1] |
| Web Questions S2T Acc | 34.5 | 40.4 | 32.2 | 33.4 | Web Questions | [Table 1] |
| Web Questions S2S Acc | 31.3 | 37.1 | 15.9 | 23.7 | Web Questions | [Table 1] |
| ChatGPT Score S2T | 4.28 | 4.56 | 4.16 | 3.99 | Alpaca-Eval | [Table 1] |
| ChatGPT Score S2S | 4.15 | 4.35 | 3.52 | - | Alpaca-Eval | [Table 1] |
| ASR-WER (%) ↓ | 3.26 | 3.89 | 5.95 | - | Alpaca-Eval | [Table 1] |
| UTMOS ↑ | 4.19 | 4.20 | 3.67 | 3.48 | Alpaca-Eval | [Table 1] |
| Latency (ms) ↓ | 582.91 | 663.32 | 1562.81 | 346.73 | - | [Table 1] |

**关键发现**:
1. **S2T-S2S gap 显著缩小**: LLaMA-Omni2-7B 在 Web Questions 上仅下降 3.2 (34.5→31.3),vs GLM-4-Voice 下降 16.3, LLaMA-Omni 下降 9.7 [§5.1]。说明 AR TTS LM + gate fusion 有效减少了语音生成对内容的损伤。
2. **LLM 规模可扩展**: 性能随 LLM 增大持续提升,14B 相比 7B 有显著增益,说明 modular SpeechLM 能有效利用底座 LLM 能力 [§5.1]。
3. **200K 数据即够用**: 7B 模型在 200K 时性能饱和;50K 时性能骤降(Llama Questions S2T 从 70.3→50.0) [Table 5, §5.3]。

**延迟分解** [Table 6, Appendix B]:
| 组件 | 7B (R=3,W=10) |
| --- | --- |
| LLM (R=3 tokens) | 231.16 ms |
| TTS LM (W=10 tokens) | 165.83 ms |
| Flow Matching + Vocoder | 185.93 ms |
| **Total** | **582.91 ms** |

## 局限性

1. **不支持副语言控制**: 无法根据输入语音的情感/语速调整输出语音风格,因为训练数据仅包含常规对话 [Limitations, §7]
2. **合成数据的天花板**: 200K 合成数据可能引入 TTS 合成器的系统性偏差(如 fish-speech/CosyVoice 的声学特性),未讨论在真人语音数据上的表现 [agent 解读]
3. **延迟仍有提升空间**: 600ms 虽满足实时交互,但高于 LLaMA-Omni (347ms) 和 Moshi (230ms),主要瓶颈在 LLM 生成 R=3 个 token 的延迟(231ms) [Table 6]
4. **固定单一输出语音**: 所有响应使用同一语音,未探索多说话人或语音克隆能力
5. **评估局限**: SpokenQA 和 instruction following 两个 benchmark 不涵盖多轮连贯性、情感识别等维度; UTMOS 作为自动 MOS 预测器的可靠性有限

## 点评

LLaMA-Omni 2 是一个**工程集成优于方法创新**的工作。其核心贡献不在于提出全新的架构或算法,而在于验证了"modular SpeechLM + CosyVoice 2 流式解码器"的组合方案在有限数据(200K 样本 vs GLM-4-Voice 百万小时)下即可达到有竞争力的性能。

**值得注意的实验洞察**:
- Gate fusion 的消融(Table 2)清晰证明了 hidden states 和 text embedding 各自不可替代的作用,这一设计选择有广泛适用性
- Read-Write 比例消融(Table 4)提供了延迟-质量-对齐三维 trade-off 的定量参考,对所有采用类似流式方案的系统都有参考价值
- TTS LM 预训练消融(Table 3)揭示了初始化策略的决定性影响(从零训练 WER 80.65%),暗示 speech token 的序列建模对 LM 来说并非 trivial

**与 MinMo 的对比**: 同期并行工作 MinMo 也采用 CosyVoice 2 风格的 AR 流式解码器,但用 1.4M 小时数据训练。LLaMA-Omni 2 仅用数千小时(200K 样本),提供了一个更高效的训练方案,但未直接比较两者性能。

## 可复用的 idea

1. **Gate Fusion for LLM→downstream**: 当需要用 LLM 的中间表征驱动下游生成模块时,gate fusion (hidden states + text embedding, element-wise sigmoid gate) 是一个简洁有效的融合方式,可推广到 LLM→image/video/music 等生成场景
2. **TTS LM 预训练的必要性**: 即使底座是强文本 LM,直接建模 speech token 也需要专门的 TTS 预训练,不能跳过 Stage I(b)
3. **多轮合成对话数据**: 用 LLM 生成多轮文本对话 → 用 TTS 合成语音 → 训练 SpeechLM 的数据 pipeline,200K 多轮优于 200K 单轮
4. **R:W 比例的工程经验**: R=3, W=10 在延迟(~580ms)/质量(UTMOS 4.19)/对齐(WER 3.26%)三方面取得最佳平衡

---

> [!review] 审阅: pass-with-fixes (2026-06-03)
> **结论**: pass-with-fixes — 1 medium issue (已修正)
> - [x] ~~frontmatter models 包含 MinMo(论文未使用),已移除~~ (fixed)
> **原则满足**: 可复述 ✓ / 可信赖 ✓ / 可区分 ✓ / 可定位 ✓ / 不污染 ✓
> 详见 `_review/LLaMA-Omni 2-review.yml`
