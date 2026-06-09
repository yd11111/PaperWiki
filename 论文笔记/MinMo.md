---
type: paper
tier: deep
title: "MinMo: A Multimodal Large Language Model for Seamless Voice Interaction"
arxiv_id: "2501.06282"
source: "Sources/MinMo.pdf"
authors: [FunAudioLLM Team (Alibaba Tongyi Lab)]
year: 2025
venue: "arXiv"
tags: [speech-LM, aligned-multimodal, full-duplex, streaming, voice-interaction, instruction-following, speech-to-speech, CosyVoice, multi-stage-alignment, LoRA]
concepts: ["[[SpeechLanguageModel]]", "[[Full-duplexSpokenDialogue]]", "[[ModalityAdaptationforSpeechLLM]]", "[[StreamingSpokenDialogue]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[LLM-basedTTS]]"]
models: ["[[模型库/CosyVoice|CosyVoice 2]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个相关实体页: [[SpeechLanguageModel]], [[Full-duplexSpokenDialogue]], [[ModalityAdaptationforSpeechLLM]], [[StreamingSpokenDialogue]], [[Speech-LLMIntegrationTaxonomy]], [[LLM-basedTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓ (confirmed), [[LLM-basedTTS]]✓ (confirmed) | 过滤: [[Full-duplexSpokenDialogue]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review), [[StreamingSpokenDialogue]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: KB 收录了 native vs aligned 两大 SpeechLM 路线。MinMo 明确属于 **aligned multimodal model** 路线——在预训练 text LLM 基础上对齐语音能力,区别于 Moshi/GLM-4-Voice 等 native 路线。KB 已记录 Moshi (native, RQ-Transformer) 和 Freeze-Omni (aligned, frozen LLM) 的对比, MinMo 进一步扩展 aligned 路线的数据规模和任务覆盖。
>
> **Full-duplex Spoken Dialogue**: KB 记录了从 dGSLM→Moshi→LSLM→Freeze-Omni 的全双工演进。MinMo 的全双工模块采用与 Freeze-Omni 类似的 chunk-level state prediction 思路,但用 LLM 隐状态而非独立分类器驱动, 且在 Alimeeting/Fisher 实际对话数据上评估。
>
> **Modality Adaptation**: KB 记录三种适配方法 (Conv downsampling / CTC compression / Q-Former)。MinMo 的 Input Projector (2-layer Transformer + CNN downsampling) 属于 Conv downsampling 增强变体;关键差异在于其两阶段 Pre-align + Full-align 策略防止随机初始化梯度干扰预训练编码器。
>
> **Streaming Spoken Dialogue**: KB 记录了 NAR CTC decoder (LLaMA-Omni) 和 three-decoder (Freeze-Omni) 等流式方案。MinMo 的 Voice Decoder 采用独特的 5:15 ratio interleaving (5 semantic vectors + 15 speech tokens),兼顾 AR 生成质量和流式输出, 是一种新的 streaming 方案。
>
> **Speech-LLM Integration Taxonomy**: MinMo 属于 latent-representation-based integration (输入侧,SenseVoice encoder → adapter → LLM) + audio-token-based integration (输出侧,LLM hidden → CosyVoice 2 token LM → speech tokens) 的混合集成,不完全归属于 Yang et al. 2025 三分法的任何单一类别。
>
> **LLM-based TTS**: MinMo 的 Voice Decoder 本质上是 CosyVoice 2 的 LM + Flow Matching + Vocoder 流水线,但以 LLM hidden states 替代纯文本 token 作为输入条件,从而实现风格指令的端到端传递。

> [!summary] 速查
> - **一句话**: 8B aligned multimodal LLM, 通过四阶段对齐 (S2T→T2S→S2S→Duplex) 在 1.4M 小时语音上训练, 以 SenseVoice encoder + Qwen2.5-7B + CosyVoice 2 decoder 组合实现 SOTA 语音理解/生成/全双工对话, 100ms S2T 延迟 / 600ms 全双工延迟 [§1, §3.4]
> - **路线**: Audio → SenseVoice-Large encoder (636M) → Input Projector (2L Transformer + CNN, 170M, 2x downsample) → Qwen2.5-7B-instruct (LoRA) → Output Projector (6M, linear) + Voice Token LM (CosyVoice 2 LM, 370M, 5:15 interleave) → Token2wav (CFM + Vocoder) → Audio; Full Duplex Predictor (18M, 1L Transformer) 接 LLM hidden states [§3.1, Table 1, Fig 3]
> - **指标**: ASR Fleurs 10-lang avg 4.13 CER/WER (SOTA vs Whisper 4.48 / Qwen2-Audio 9.11); S2TT CoVoST2 avg 38.92 BLEU (SOTA); LID Fleurs-102 85.3% acc (SOTA); SER near-100% on acting sets (CASIA 98.1%, CREMA-D 94.8%); SQA S2S LlamaQ 48.3 / WebQ 39.9 (SOTA); Instruction-following 98.4% total (vs GLM-4-Voice 63.1%); Full-duplex assistant turn-taking F1@10 0.985 (simulated); Duplex system latency ~600ms on L20 [Tables 5-22]
> - **可借鉴**: (1) **5:15 semantic-speech interleaving**: LLM hidden states 和 text tokens 拼接作为 semantic vectors,每 5 个 semantic vectors 后接 15 个 speech tokens,单 AR decoder 即可流式生成,不需 Freeze-Omni 的三个 decoder [§3.2]; (2) **Pre-align 防梯度冲击**: 先只训 adapter 让随机初始化参数稳定,再联合训练 encoder+adapter,避免大梯度干扰预训练参数 [§3.4]; (3) **全双工基于语义理解**: 用 LLM hidden states 驱动 duplex predictor,利用 LLM 的语义能力判断是否响应, 比独立 VAD/分类器更准确 [§3.4]; (4) **Instruct speech 仅用 hidden embedding 对齐**: 不更新 LLM,仅靠 <150h 指令数据 + Voice Decoder 训练即可实现情感/方言/语速控制, 98.4% instruction accuracy [§3.4, Table 18]
> - **局限**: LLM 仅 LoRA 更新,多样指令跟随 (语言/任务切换) 能力受限 [§6]; TTS 存在长尾发音错误 (one-to-many tokens + 特殊符号) [§6]; Instruct 数据仅约 1000h,指令控制效率待提升 [§6]; 全双工仍需外部 AEC/VAD,非完全端到端 [§6]; TTS 质量略低于独立 CosyVoice 2 (zh CER 2.48 vs 2.06, NMOS 3.69 vs 3.73) [Table 17]; S2S 模式性能系统性低于 S2T [Table 19]

## 核心问题

MinMo 要解决 aligned multimodal speech-text model 的四个关键不足 [§1]:

1. **数据规模小**: 现有 aligned 模型训练数据规模有限 (LLaMA-Omni 200K samples, Freeze-Omni 120K hours),是否更大规模数据能提升模型能力且不损害 text LLM 尚不清楚 [论文原文]
2. **任务覆盖窄**: 现有模型只在有限语音任务上验证 (ASR/SQA),缺乏翻译、情感识别、说话人分析、音频事件检测等全面评估 [论文原文]
3. **缺少风格控制**: 已有 aligned 模型 (LLaMA-Omni, Freeze-Omni) 被认为只能控制内容不能控制语音风格 (情感/方言/语速),MinMo 要打破这一认知 [论文原文, 引用 Zeng et al. 2024 对 aligned 模型的批评]
4. **全双工能力缺失**: 现有 aligned 模型大多仅支持半双工 (turn-based) 交互,缺乏系统性的全双工训练和评估 [论文原文]

[agent 解读] MinMo 的核心立场是: aligned 路线的上限并未被充分探索,只要数据足够大、训练足够系统化,aligned 模型可以在保持 text LLM 能力的同时达到甚至超越 native 模型在语音任务上的表现。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MinMo (~8B) 由六个模块组成 [§3.1, Table 1, Fig 3]:

| 模块 | 初始化来源 | 参数量 | 功能 |
|------|-----------|--------|------|
| Voice Encoder | SenseVoice-Large | 636M | 多语言语音理解 (ASR/SER/AED) |
| Input Projector | 随机初始化 | 170M | 2L Transformer + CNN, 维度对齐 + 2x 降采样 |
| LLM | Qwen2.5-7B-instruct | 7B | 核心语言模型, LoRA 更新 |
| Output Projector | 随机初始化 | 6M | Linear, 维度对齐 (LLM → Voice Token LM) |
| Voice Token LM | CosyVoice 2 LM | 370M | AR 语音 token 生成 |
| Full Duplex Predictor | 随机初始化 | 18M | 1L Transformer + Linear Softmax, 双工控制 |

信号流: 用户语音 → Voice Encoder → Input Projector → LLM (生成 text tokens + hidden states) → Output Projector 将 hidden states 转换为 semantic vectors → Voice Token LM 将 semantic vectors 和 speech tokens 交替处理 → Token2wav (CFM + Vocoder, 来自 CosyVoice 2, 冻结) → 输出语音 [§3.1, §3.2]

### 关键设计选择

**1. 为什么用 SenseVoice-Large 而不是 Whisper?** SenseVoice-Large 原生支持多语言 ASR、情感识别和音频事件检测 [§3.1, 引用 An et al. 2024],这为 MinMo 的多任务能力提供了更好的初始化基础 [论文原文]。[agent 解读] Whisper 主要优化 ASR/translation,对 SER/AED 等副语言任务覆盖较弱,选 SenseVoice 是为了 MinMo 广泛任务覆盖的目标。

**2. Voice Decoder 的 5:15 交替设计** [§3.2, Fig 3]: 这是 MinMo 最核心的创新。Voice Decoder 将 LLM 的 hidden states 和 text tokens 混合为 "semantic vectors",每 5 个 semantic vectors 后 Voice Token LM 自回归生成 15 个 speech tokens (对应 CosyVoice 2 的帧率)。

为什么这种设计优于替代方案?
- vs LLaMA-Omni 的 NAR CTC decoder: AR 生成质量更好,CTC 的 NAR 方式质量受限 [论文原文]
- vs Freeze-Omni 的三个 decoder: 结构更简单,只需一个 Voice Token LM [论文原文]
- Semantic vectors 的构造: 将 "用户输入的 embeddings" + "LLM last layer hidden states" 沿特征维度拼接为 query embedding,再与 "5 个 text token embeddings + hidden states" 沿序列维度拼接,送入 Output Projector。hidden states 包含丰富上下文但语义模糊,sampled text tokens 精确但信息有限,两者互补 [论文原文]

[agent 解读] 5:15 的比率源于 CosyVoice 2 的设计——每收到 5 个 text tokens 就生成 15 个 speech tokens。MinMo 将这个比率固定化,使得 LLM 每生成 5 个 text token 就触发一次语音合成,实现了流式输出。理论延迟为 5*d_llm + 15*d_lm + 15*d_syn [Eq. 1]。

**3. 为什么 LLM 只用 LoRA 更新而不是全参数微调?** 论文虽未给出明确消融,但其策略暗示: 1.4M 小时语音数据相对 text LLM 预训练语料仍不够平衡,全参数更新可能导致灾难性遗忘 [论文原文, §1 讨论]。[agent 解读] 这也是 aligned 路线的核心哲学——语音能力应被"附加"而非"替换"到 text LLM 上。但论文在 Limitations 中承认 LoRA 限制了指令跟随的多样性 [§6]。

**4. 全双工模块设计** [§3.4]: Full Duplex Predictor 是一个轻量级分类器 (18M, 1L Transformer + Linear Softmax),以 LLM 的 hidden embeddings 作为输入,实时预测两类决策:
- 是否应该对当前用户查询生成响应 (assistant turn-taking)
- 是否应该停止当前输出以倾听用户 (user turn-taking / interruption)

[论文原文] 关键洞察: 利用 LLM 的语义理解能力 (而非简单的声学 VAD) 来驱动全双工决策,使模型能区分有效打断和背景噪声。

### 训练策略

MinMo 采用渐进式四阶段对齐训练 [§3.4]:

**Stage 1: Speech-to-Text Alignment** (~1.2M hours S2T data)
- **Pre-align**: 仅训练 Input Projector (随机初始化参数稳定化)
- **Full-align**: 训练 Input Projector + Voice Encoder, LLM 冻结
- **SFT**: 训练 Input Projector + Voice Encoder + LLM (LoRA), ~1.3M samples 覆盖多任务
- [论文原文] Pre-align 的目的: 防止随机初始化的 projector 产生大梯度,干扰预训练 Voice Encoder 的初期训练 [§3.4]

**Stage 2: Text-to-Speech Alignment** (~171K hours T2S data)
- 先训 Output Projector,再联合训练 Output Projector + Voice Token LM
- 其他所有 MinMo 参数冻结
- 包含 ~1000h instruct speech synthesis 数据 (Qwen-Max 生成指令标注)

**Stage 3: Speech-to-Speech Alignment** (~10K hours S2S data)
- 仅更新 Output Projector + Voice Token LM (与 Stage 2 一致)
- 包含 general S2S dialogue + style-controllable S2S
- [论文原文] 关键发现: 即使不更新 LLM,仅靠 <150h instruction data + hidden embedding 对齐,就能实现有效的风格控制 [§3.4]

**Stage 4: Duplex Interaction Alignment** (~4K hours)
- 仅训练 Full Duplex Predictor
- 数据来源: Alimeeting / Fisher / 模拟数据
- 启发式标注规则: assistant turn-taking 以用户轮次结束时刻为起点; user turn-taking 以 assistant 结束后 T~N(0.6, 0.4^2) 秒为起点 [§3.3]

**S2S 训练数据构造** [§3.3]: 三种策略混合
1. 文本对话 (Alpaca/ShareGPT) → CosyVoice zero-shot 合成用户语音 + CosyVoice-SFT (2h 单说话人微调) 合成助手语音
2. 真实 ASR 语音作用户查询 + Qwen-Max 生成回复文本 + CosyVoice-SFT 合成助手语音
3. Qwen-Max 生成风格可控多轮对话 + CosyVoice / CosyVoice 2 合成

[agent 解读] 三种策略的设计逻辑清晰: (1) 保证用户语音多样性, (2) 增强对真实音频的鲁棒性, (3) 覆盖风格可控场景。这也揭示了 aligned 模型的实际挑战——S2S 训练数据需要通过合成构造, 训练数据质量依赖上游 TTS 系统。

## 实验

### 语音理解

| 任务 | 测试集 | MinMo | 最佳 Baseline | 提升 |
|------|--------|-------|-------------|------|
| ASR (多语言) | Fleurs 10-lang avg | **4.13** CER/WER | Whisper 4.48 | +8% [Table 5] |
| ASR (中文) | WenetSpeech test-net | **6.78** CER | Qwen2-Audio 8.14 | +17% [Table 5] |
| S2TT | CoVoST2 avg 9-dir | **38.92** BLEU | SeamlessM4T 37.79 | +3% [Table 6] |
| LID | Fleurs 102-lang | **85.3%** acc | mSLAM-CTC 77.7% | +10% [Table 7] |
| SER | CASIA | **98.1%** UA | EmoBox best 59.6% | +65% [Table 11] |
| SER | MELD | **65.1%** WA | EmoBox best 51.9% | +25% [Table 11] |
| Speaker Age | AIR-Bench | **70.1%** acc | Qwen-Audio-Turbo 58.8% | +19% [Table 12] |

[论文原文] MinMo 在 ASR 上的一个重要特征: 有无 LID prompt 对性能影响极小 (Fleurs avg 4.13 vs 4.16),而 Whisper/Qwen2-Audio 严重依赖 LID 信息 (Whisper 4.48 vs 4.45, Qwen2-Audio 9.11 vs 13.77) [Table 5]。

### 语音生成

| 任务 | 测试集 | MinMo | Baseline | 备注 |
|------|--------|-------|----------|------|
| TTS (中文) | SEED test | 2.48 CER, 3.69 NMOS | CosyVoice 2 SFT: 2.06 CER, 3.73 NMOS | 略低于独立 TTS [Table 17] |
| TTS (英文) | SEED test | 2.90 WER, 3.56 NMOS | CosyVoice 2 SFT: 3.19 WER, 3.71 NMOS | WER 更优但 NMOS 略低 [Table 17] |
| Instruct gen. | in-house 122-turn | **98.4%** total acc | GLM-4-Voice 63.1% | 情感 97.6%, 方言 100%, 语速 100% [Table 18] |

[agent 解读] TTS 质量略低于独立 CosyVoice 2 是合理的: 论文解释为 LLM 的 input/output text 不一致导致 confused hidden states,以及 fine-tuned speaker 声学特征差异 [§4.4]。但 instruction-following 方面的巨大优势 (98.4% vs 63.1%) 说明 aligned 模型通过 hidden embedding 确实能传递风格信息。

### 语音对话

| 任务 | 测试集 | MinMo | 最佳 Baseline | 来源 |
|------|--------|-------|-------------|------|
| SQA S2T | Llama Questions | **78.9** acc | Freeze-Omni 72.0 | [Table 19] |
| SQA S2S | Llama Questions | **48.3** acc | Freeze-Omni 44.7 | [Table 19] |
| SQA S2S | Web Questions | **39.9** acc | GLM-4-Voice 15.9 | [Table 19] |
| Dialogue | Alpaca test | 6.48 / 10 | GT: 7.73, ASR+Qwen2.5: 6.59 | [Table 20] |
| Dialogue | ChitChat test | 7.20 / 10 | GT: 7.62, ASR+Qwen2.5: 7.18 | [Table 20] |

[论文原文] S2S 模式系统性低于 S2T (Llama Q: 48.3 vs 78.9),论文将此归因于测试集答案富含文本结构/专业词汇,对 TTS 要求高,且 S2S 评估依赖 ASR 转写准确度 [§4.5]。

### 全双工

| 任务 | 数据集 | F1@K=1 | F1@K=5 | F1@K=10 | 来源 |
|------|--------|--------|--------|---------|------|
| Assistant turn-taking | Simulation | 0.787 | 0.962 | **0.985** | [Table 22] |
| Assistant turn-taking | Alimeeting | 0.614 | 0.754 | 0.804 | [Table 22] |
| User turn-taking | Simulation | 0.257 | 0.815 | 0.994 | [Table 22] |
| User back-channel | Alimeeting | 70-80% acc | - | - | [Table 22] |

系统延迟分解 [Table 21, L20 GPU]:
- Full-duplex predictor (assistant turn-taking): 250ms
- Speech-to-text (5 text tokens): 150ms
- Text-to-speech token (15 speech tokens): 70ms
- Token2wav: 130ms
- **总延迟: ~600ms**

[agent 解读] 全双工在模拟数据上表现接近完美 (F1@10=0.985),但在真实 Alimeeting 数据上显著下降 (0.804)。论文将此归因于真实对话的高噪声、变化语速和停顿 [§4.5]。back-channel 准确率仅 70-80%,论文承认 user turn-taking 灵敏度与 back-channel 区分存在 trade-off。

## 局限性

论文自述的局限 [§6]:
1. **LoRA 限制指令跟随多样性**: LLM 仅 LoRA 更新,多语言/多任务的复杂指令跟随能力不足,是否需要更高比例的 text data 全量更新仍待探索
2. **长尾发音错误**: End-to-end 生成中 one-to-many tokens 和特殊符号导致发音错误
3. **指令控制效率低**: 指令数据仅 ~1000h + 仅用 hidden embedding 传递,历史信息传递受限
4. **非完全端到端全双工**: 仍需外部 AEC (回声消除) 和 VAD 模块

[agent 解读] 未自述但值得注意的局限:
- **S2S 训练数据全部为合成**: 10K hours S2S 数据通过 CosyVoice 合成,合成质量上限决定了模型能力上限;这是 aligned 路线的结构性瓶颈,native 模型可直接从真实对话学习
- **TTS 质量低于独立 TTS**: Table 17 显示 MinMo 在内容一致性和 NMOS 上均略低于 CosyVoice 2-SFT,说明集成带来了一定质量损失
- **缺乏主观评测**: 论文承认 "subjective evaluation might be more appropriate for speech-to-speech voice chat models" [§4.4],但全文缺乏 MOS 主观测试
- **仅 L20 GPU 测速**: 600ms 延迟在 L20 上测得,工业部署场景下的延迟可能不同

## 点评

> [!note] 审阅者点评

**MinMo 在 aligned multimodal 路线上的定位**: MinMo 是目前 aligned 路线最全面的系统——1.4M 小时训练数据 (比 Freeze-Omni 的 120K 多一个数量级)、覆盖最广泛的语音任务、首次在 aligned 模型上实现 instruction-following style control (打破 Zeng et al. 2024 对 aligned 模型"只能控制内容"的批评)、且系统性地实现和评估了全双工能力。

**核心贡献的独特性**: 5:15 semantic-speech interleaving 设计是本文最值得关注的创新。它巧妙地利用 CosyVoice 2 已有的 token rate 设计,将 LLM hidden states 作为 conditioning signal 注入 speech token 生成过程,避免了 Freeze-Omni 三个 decoder 的复杂性和 LLaMA-Omni NAR decoder 的质量损失。通过 hidden states 而非纯 text tokens 传递风格信息,使得仅 <150h 指令数据就能实现 98.4% 的 instruction-following accuracy。

**native vs aligned 辩论中的意义**: 在 Moshi/GLM-4-Voice 展示 native 路线优势后,MinMo 有力地回应了: aligned 路线在数据规模和系统工程足够的条件下,可以在语音任务上全面超越 native 模型,同时几乎保持 text LLM 的完整能力 (ChitChat 7.20 vs ASR+Qwen2.5 的 7.18)。

**不足**: (1) 缺乏与 GPT-4o 的直接对比,而 GPT-4o 才是 seamless voice interaction 的真正标杆; (2) 全双工评估在真实场景 (Alimeeting) 下性能下降明显,说明仅靠 4K hours 对话数据不足以覆盖真实交互的复杂性; (3) TTS 质量的 regression (vs standalone CosyVoice 2) 没有足够的分析和解决方案。

## 可复用的 idea

1. **Pre-align 防梯度冲击** [§3.4]: 当 adapter/projector 是随机初始化而连接的 encoder/LLM 是预训练的,先只训练随机初始化部分让其稳定,再联合训练。这是一个通用且低成本的训练技巧,特别适用于任何 "随机模块 + 预训练模块" 的组合。

2. **Hidden states + text tokens 互补 conditioning** [§3.2]: Hidden states 包含丰富上下文 (包括指令信息) 但语义模糊,sampled text tokens 精确但信息有限,沿特征维度拼接可以互补。这个思路可推广到任何需要从 LLM 提取 conditioning signal 的场景。

3. **Style control through hidden embedding only** [§3.4]: 不需要修改 LLM 参数,仅通过 Voice Decoder 训练 + hidden embedding 传递就能实现风格控制。这意味着 aligned 模型的输出模态可以"免费"获得 LLM 的指令理解能力,只要 decoder 能正确利用 hidden states 中的信息。

4. **三源 S2S 数据构造策略** [§3.3]: 文本对话合成 (保证多样性) + 真实语音+LLM回复 (保证鲁棒性) + 风格可控合成 (保证表达力) 的三种混合策略,可复用于任何缺乏真实 S2S 数据的场景。

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> Issues: 2 (high: 0, medium: 0, low: 2)
> 详见 `_review/MinMo-review.yml`
