---
type: paper
tier: deep
title: "Step-Audio 2 Technical Report"
arxiv_id: "2507.16632"
source: "Sources/Step-Audio2.pdf"
authors: [StepFun Audio Team]
year: 2025
venue: "arXiv"
tags: [speech-LM, end-to-end, multimodal, latent-encoder, RAG, tool-calling, RLHF, reinforcement-learning, ASR, TTS, paralinguistic, speech-interaction, open-source]
concepts: ["[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[LLM-basedTTS]]", "[[ConditionalFlowMatching]]", "[[ModalityAdaptationforSpeechLLM]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[LLM-basedTTS]], [[ConditionalFlowMatching]], [[SemanticvsAcousticTokens]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[LLM-basedTTS]]✓, [[ConditionalFlowMatching]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[ModalityAdaptationforSpeechLLM]](pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: Step-Audio 2 是 Step-Audio 系列从 AQTA 级联架构转向端到端 SpeechLM 的关键转折。Step-Audio v1 (130B) 采用 AQTA + 独立 3B TTS decoder 的分离架构 [Step-Audio §3],理解和生成不在同一 forward pass 中完成。Step-Audio 2 取消了独立 TTS decoder,改为在 LLM decoder 内直接生成 text+audio 交织 token,实现了真正的端到端。这与 KB 中 SpeechLM 的演进趋势一致: 从级联 (AudioGPT) → 半端到端 (Step-Audio v1/GLM-4-Voice) → 全端到端 (Moshi/Qwen2.5-Omni/Step-Audio 2)。
>
> **Speech Tokenizer / Semantic vs Acoustic Tokens**: Step-Audio 2 在输入端用 **latent audio encoder** (连续表征) 替代了 v1 的 dual-codebook discrete tokenizer (linguistic 16.7Hz + semantic 25Hz)。输出端仍使用 CosyVoice 2 的离散 audio tokens 通过 LLM 生成。这是一种**输入连续+输出离散**的混合策略——输入端保留最大信息量 (包括副语言信息),输出端用离散 token 便于 LLM autoregressive 建模。
>
> **LLM-based TTS**: Step-Audio 2 将 TTS 内化为 LLM 自身能力的一部分,不再需要独立 TTS 模型。text+audio token 交织输出让 LLM 同时掌控文本内容和语音表达。这与 [[论文笔记/Step-Audio2.5|StepAudio 2.5]] 的 "TTS = pure next-token prediction" 理念一脉相承,但 Step-Audio 2 是首次实现这一转变的版本。
>
> **Conditional Flow Matching**: Step-Audio 2 的 audio detokenizer 沿用 Flow Matching + HiFi-GAN 的两阶段渲染管线,并在 Transformer block 中每个 self-attention 后增加 CNN encoder layer,在 200K 小时高质量语音上训练,显著提升了发音准确性和音色相似度 [§3.1]。
>
> **Modality Adaptation** [待确认]: Step-Audio 2 使用 2x downsampling audio adaptor 将 audio encoder 的 25Hz 输出降至 12.5Hz,连接到 LLM。这是 KB 中 convolutional downsampling 适配策略的典型实例。

> [!summary] 速查
> - **一句话**: Step-Audio 系列从 AQTA 级联转向端到端的关键版本,通过 latent audio encoder + reasoning-centric RL + text-audio 交织生成 + RAG/tool-calling,在 ASR/audio-understanding/paralinguistic/speech-conversation 全面 SOTA [§Abstract]
> - **路线**: Input audio → Frozen audio encoder (25Hz) → Adaptor (2x down, 12.5Hz) → LLM decoder → interleaved text+audio tokens → Audio detokenizer (Flow Matching + HiFi-GAN) → waveform [§3.1, Fig 3]
> - **指标**: ASR CER 3.08 zh / WER 3.14 en (SOTA vs Kimi/Qwen/GPT-4o) [Table 1]; Paralinguistic 83.09 avg (vs GPT-4o 43.45) [Table 2]; MMAU 78.0 (SOTA) [Table 3]; CoVoST2 BLEU 39.26 avg (SOTA) [Table 4]; URO-Bench zh 83.32 / en 83.90 basic avg (SOTA) [Table 6]
> - **可借鉴**: (1) Latent audio encoder 替代 discrete tokenizer 做输入,保留副语言信息; (2) Reasoning-centric RL: 两阶段 PPO (binary reward → learned reward) + GRPO 提升音频推理; (3) Audio search tool: 语音检索库实现风格/音色切换; (4) Conversational speech synthesis pipeline 生成训练数据 (50K speakers); (5) StepEval-Audio-Paralinguistic: 11 维副语言理解 benchmark
> - **局限**: 模型参数量未公开 (仅说"fewer than Step-Audio v1"); TTS 质量无独立评估 (融合在 conversation 中); RL 的 reasoning trace 格式未详细说明; 端到端但仍需独立 detokenizer (非 waveform 直出)

## 核心问题

Step-Audio 2 要解决的核心问题是: 现有 LALM 在自然智能语音交互上的四个短板 [§1]:

1. **语义理解强但副语言理解弱**: 多数 LALM (如 Qwen-Audio, Spirit LM) 只关注语音中的语义信息,忽略了情感、语速、风格等副语言信息,这些对理解说话人意图至关重要 [论文原文]
2. **理解但不能表达**: 部分系统 (如 Qwen2-Audio) 能理解副语言信息但只产出文本,无法将理解转化为富表达力的语音响应 [论文原文]
3. **幻觉与知识局限**: 纯模型推理容易产生事实性错误,缺乏获取实时信息的能力 [论文原文]
4. **音色和风格受限**: 现有系统提供的音色/风格选择有限,用户无法灵活切换 [论文原文]

**与 Step-Audio v1 的关系**: v1 用 AQTA + 独立 TTS decoder 解决了理解-生成统一的问题,但理解和生成仍分属两个模块。Step-Audio 2 进一步整合,将 audio token 生成纳入 LLM 本身,实现端到端 [§3.1] [论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Step-Audio 2 由四个组件构成 [§3.1, Fig 3]:

1. **Audio Encoder** (frozen): 预训练于多种语音和音频理解任务 (ASR, 说话人年龄/性别预测, 音频事件检测等),输出帧率 25Hz,全程冻结 [论文原文]
2. **Audio Adaptor**: 2x 下采样适配器,将 encoder 输出从 25Hz 降至 12.5Hz,连接到 LLM [论文原文]
3. **LLM Decoder**: 直接接收 adaptor 输出的 latent audio features 作为输入,输出 text+audio 交织 token 序列。audio tokenizer 采用 CosyVoice 2 的方案,text 和 audio tokens 以固定比例交织,末尾 padding 对齐 [§3.1] [论文原文]
4. **Audio Detokenizer**: Flow Matching + HiFi-GAN vocoder。FM 模块从 audio tokens 生成 mel spectrogram,在 Transformer block 的每个 self-attention 后增加 CNN encoder layer,用 200K 小时高质量语音训练 [§3.1] [论文原文]

**与 Step-Audio v1 的架构差异** [§3.1]:
- v1: dual-codebook discrete tokenizer (input) → 130B LLM → text output → 独立 3B speech decoder (TTS)
- v2: latent audio encoder (input) → LLM decoder → interleaved text+audio tokens → detokenizer

[agent 解读] 这个转变的关键意义在于: (1) 输入端从离散 token 改为连续表征,保留了更多副语言信息 (情感/语速/风格); (2) 输出端将 TTS 生成权下放给 LLM 自身,消除了 v1 中 LLM 和 TTS decoder 之间的信息断层。

### 关键设计选择

**为什么用 latent audio encoder 而非 discrete tokenizer 做输入?** [§3.1]
v1 的 dual-codebook discrete tokenizer 不可避免地在量化过程中丢失副语言信息。Step-Audio 2 改用预训练于多任务 (包括情感/风格相关任务) 的 encoder,其连续输出保留了更丰富的副语言线索。encoder 全程冻结保证了表征稳定性 [论文原文]。

**RAG + Tool Calling** [§3.1]:
Step-Audio 2 设计了四种外部工具: audio search (语音检索)、date & time、weather、web search。其中 **audio search** 是专为 LALM 设计的新型工具: 维护一个包含数十万条语音及其转写和描述的语音库,模型可通过显式或隐式语音指令检索语音,实现风格模仿和音色切换 [论文原文]。

[agent 解读] Audio search tool 是 Step-Audio 2 的独特创新。传统 LALM 的音色控制依赖训练数据中已有的说话人,audio search 将音色库外挂为检索式工具,极大扩展了可用音色范围,且无需重新训练。

**Interleaved text+audio token output** [§3.1]:
text 和 audio tokens 以固定比例交织排列 (具体比例未详述),末尾 padding 对齐。推理时从交织序列中提取 audio tokens 送入 detokenizer [论文原文]。输入 audio features 和输出交织序列作为 history information 填充到下一轮对话的上下文中 [论文原文]。

### 训练策略

**Pre-training** [§3.2]: 从 textual LLM 初始化,共 1.356T tokens,21 天。分四个阶段:

| 阶段 | Token 量 | 核心内容 | 可训练模块 | 序列长度 |
|------|----------|---------|-----------|---------|
| Phase 1: Adaptor alignment | 100B ASR | 语音-文本对齐 | 仅 adaptor | 8,192 |
| Phase 2: Tokenizer extension | 128B text + 128B audio | 词表扩展 (+6.6K audio tokens) + 文本保持 | LLM + adaptor + embed/head (差异 lr) | 16,384 |
| Phase 3: Main pre-training | 800B mixed | ASR + TTS + S2TT + T2ST + continuation + conversation | 全部 (统一 lr 2e-5) | 16,384 |
| Phase 4: Cooldown | 200B high-quality | 扩展任务 + 多语言/方言 + 合成对话数据 | 全部 (lr cosine decay to 5e-6) | - |

[§3.2] Phase 2 中的差异学习率策略值得注意: LLM lr 2e-5, adaptor/embedding/output lr 5e-5/5e-5/4e-5。embedding 和 output layer 需要更高 lr 来快速学习新增的 6.6K audio token 嵌入 [论文原文]。

Phase 4 使用 conversational speech synthesis pipeline (参照约 50K 独特说话人) 合成对话训练数据,确保语音多样性 [论文原文]。

**SFT** [§3.3]: 4B tokens (text + audio),单 epoch,lr 从 1e-5 decay to 1e-6。

任务覆盖: 多语言/多方言 ASR (GigaSpeech, WenetSpeech) + 音频事件分类与字幕 (AudioSet, AudioCaps 重格式化为 QA pairs) + 11 维副语言 speech captioning (in-house) + TTS (高质量标注) + S2ST (CoVoST 2 中英子集) + text-to-text conversation (LLM 改写为口语化风格 + 随机插入情感/语速指令 + 合成为语音) + tool calling (每种工具约 1K 对话脚本) [§3.3] [论文原文]。

**Reasoning-centric RL** [§3.4]: 关键创新,分三阶段:
1. **PPO Stage 1 (60 iterations)**: binary reward 限制 thinking 序列长度 — 推理适当简洁 (非空非过长) 得 1,否则 0。batch 64, actor lr 1e-6, critic lr 2.5e-6 [论文原文]
2. **PPO Stage 2 (120 iterations)**: 从 binary 切换到 learned reward model 评估响应质量,同参数 [论文原文]
3. **GRPO (400 iterations)**: 进一步提升音频感知推理能力 [论文原文]

[agent 解读] 两阶段 PPO 的设计逻辑是: 先用简单的 binary reward 让模型学会"思考但不过度思考" (解决 CoT reasoning 的长度控制问题),再用 learned reward 提升思考质量。这与 v1 的 anti-deaf-hacking RLHF 目标不同——v1 解决 reward model 偏差,v2 解决推理效率。

**Reasoning SFT data** [§3.3]: 两类专门构建的推理数据用于 SFT cold-start RL:
1. 复杂声学场景: 将 AudioSet/AudioCaps 中的多个音频混合,创建复杂声学环境 [论文原文]
2. 副语言感知对话: 用 conversation synthesis pipeline 合成含情感描述的对话,textual LLM 生成带显式推理链的 QA pairs [论文原文]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ASR avg CER (Chinese, 6 sets) | 3.08 | Kimi-Audio 3.75 / Qwen-Omni 4.81 / Doubao 3.81 | AISHELL-1/2, FLEURS zh, KeSpeech, WenetSpeech meeting/net | [Table 1] |
| ASR avg WER (English, 4 sets) | 3.14 | Kimi-Audio 4.18 / GPT-4o-Transcribe 4.50 / Qwen-Omni 5.35 | Common Voice, FLEURS en, LibriSpeech clean/other | [Table 1] |
| ASR avg CER (dialect/accent, 6 sets) | 8.85 | Doubao 14.66 / Qwen-Omni 19.40 / Kimi-Audio 25.52 | In-house Anhui/Guangdong/Guangxi/Shanxi/Sichuan/Shanghai | [Table 1] |
| Paralinguistic avg accuracy (11 dim) | 83.09 | Kimi-Audio 49.64 / GPT-4o 43.45 / Qwen-Omni 44.18 | StepEval-Audio-Paralinguistic (550 samples) | [Table 2] |
| MMAU avg (sound+speech+music) | 78.0 | Omni-R1 77.0 / Audio Flamingo 3 73.1 / Qwen2.5-Omni 71.5 | MMAU v05.15.25 test-mini | [Table 3] |
| CoVoST 2 avg BLEU (S2TT) | 39.26 | Qwen2.5-Omni 35.40 / GPT-4o 29.61 | CoVoST 2 en-zh + zh-en | [Table 4] |
| CVSS avg BLEU (S2ST) | 30.87 | Step-Audio-AQAA 27.36 / GPT-4o 23.68 / Qwen-Omni 15.35 | CVSS en-zh + zh-en | [Table 4] |
| Tool calling trigger precision/recall (audio search) | 86.8/99.5 | Qwen3-32B (text) 67.5/98.5 | StepEval-Audio-Toolcall | [Table 5] |
| URO-Bench Basic avg (Chinese) | 83.32 | GPT-4o 78.59 / Step-Audio-AQAA 74.71 / Kimi-Audio 73.59 | URO-Bench | [Table 6] |
| URO-Bench Basic avg (English) | 83.90 | GPT-4o 84.54 / Kimi-Audio 60.04 / Qwen-Omni 70.58 | URO-Bench | [Table 6] |
| URO-Bench Pro avg (Chinese) | 68.25 | GPT-4o 67.10 / Kimi-Audio 66.07 / Step-Audio-AQAA 65.61 | URO-Bench | [Table 6] |

**Step-Audio 2 mini** (Appendix B): 开源版本,Qwen2-Audio encoder + Qwen2.5-7B 初始化,仅支持 web search 工具,性能接近 Step-Audio 2 全量版 [§B]。

## 局限性

1. **模型参数量未公开**: 论文仅说 "fewer parameters than Step-Audio" (130B),但未给出 Step-Audio 2 的具体参数量。mini 版基于 Qwen2.5-7B,全量版规模不明 [agent 解读]
2. **TTS 质量无独立评估**: v1 有独立 TTS 评估 (SEED test CER/WER),Step-Audio 2 将 TTS 融入端到端 conversation,缺乏 standalone TTS benchmark 数据 [agent 解读]
3. **端到端但仍非 waveform 直出**: 仍需独立 detokenizer (FM + HiFi-GAN),不像真正的 waveform-level 端到端 (如 LatentLM)。text-audio 交织输出增加了序列长度 [agent 解读]
4. **Paralinguistic benchmark 自建**: StepEval-Audio-Paralinguistic 是自建 benchmark,虽已开源但缺乏第三方验证。baseline 中 GPT-4o 得分极低 (43.45) 可能与 prompt/评估协议相关 [agent 解读]
5. **Audio search tool 的可复现性**: 语音库 "hundreds of thousands of speeches" 是内部资源,开源版 (mini) 仅支持 web search [§B] [agent 解读]
6. **RL reasoning trace 透明度不足**: 论文展示了一个 thinking 示例 [Fig 2] 但未系统分析 reasoning trace 的质量和对性能的贡献,也未提供 ablation [agent 解读]

## 点评

Step-Audio 2 是 Step-Audio 系列从"模块化工业系统"到"端到端基座模型"的关键转折。v1 用 130B LLM 做理解 + 3B TTS decoder 做生成的分离设计虽然实用,但信息断层和延迟叠加是固有限制。Step-Audio 2 通过将 audio token 生成纳入 LLM 自身,消除了这个断层。

**最核心的架构贡献是输入端的 latent audio encoder 替代 discrete tokenizer**。这不是微调级别的改动,而是信息流的根本重构: discrete tokenizer 在量化时必然丢失信息 (v1 用 dual-codebook 缓解但无法消除),latent encoder 保留了连续表征中完整的副语言信号。实验中 paralinguistic 理解从 v1 的 Step-Audio-AQAA 36.91 飙升至 Step-Audio 2 的 83.09 [Table 2],证实了这一设计选择的有效性 [agent 解读]。

**Reasoning-centric RL 的两阶段设计是实用创新**: binary reward (控制思考长度) → learned reward (提升思考质量) 的渐进策略,比一步到位的 reward model 训练更稳定。这与 v1 的 RLHF (对齐人类偏好) 和 v2.5 的 GRM (生成式奖励模型) 形成了系列内 RL 方法论的演进线 [agent 解读]。

**Audio search tool 是系列最具独创性的工具创新**: 将音色/风格控制从训练时的 in-context learning 扩展为推理时的 retrieval,这种"检索增强的语音表达"思路在 LALM 领域尚无先例。但工具的工业价值与学术可复现性之间存在张力——开源版不包含此工具 [agent 解读]。

**系列演进观察**: Step-Audio (2025.02, 130B AQTA+TTS) → Step-Audio 2 (2025.07, 端到端) → Step-Audio-AQAA (2025.06, 全端到端表达) → Step-Audio 2.5 (2026.05, 统一基座+三分支特化)。从分离到统一,从理解到推理,每代解决前代的核心限制。Step-Audio 2 是这条演进线上"从级联到端到端"的转折点 [agent 解读]。

## 可复用的 idea

1. **Latent audio encoder 替代 discrete tokenizer 做输入**: 输入端保留连续表征以最大化副语言信息保持,输出端仍用 discrete tokens 便于 LM 建模,实现"输入连续+输出离散"的混合策略
2. **Reasoning-centric RL 两阶段**: binary reward 先控思考长度,learned reward 再提思考质量,避免 cold-start 问题
3. **Audio search tool**: 将音色/风格检索库作为 LALM 的外部工具,检索式扩展表达能力
4. **Paralinguistic speech captioning**: 构建 11 维副语言标注数据集 (gender/age/timbre/emotion/pitch/rhythm/speed/style/scenario/event/vocal) 用于 SFT,系统性覆盖副语言理解
5. **Conversational speech synthesis pipeline**: 文本对话 → LLM 改写为口语化 → 随机插入情感/语速指令 → TTS 合成 (50K 说话人库),自动化生产多样化语音对话数据
6. **Detokenizer 增强**: FM Transformer block 中每个 self-attention 后增加 CNN encoder layer,提升发音准确性和音色相似度

---

检索命中: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[LLM-basedTTS]], [[ConditionalFlowMatching]], [[SemanticvsAcousticTokens]] | 过滤: [[ModalityAdaptationforSpeechLLM]](pending-review) | 未命中但可能相关: 无
