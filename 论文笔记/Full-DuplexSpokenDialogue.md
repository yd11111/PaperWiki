---
type: paper
tier: deep
title: "How Should LLMs Listen While Speaking? A Study of User-Stream Routing in Full-Duplex Spoken Dialogue"
arxiv_id: "2605.10199"
source: "arXiv"
authors: [Hui Lu, Xueyuan Chen, Huimeng Wang, Shuhai Peng, Shiyin Kang, Xixin Wu, Zhiyong Wu]
year: 2026
venue: "arXiv"
tags: [full-duplex, spoken-dialogue, user-stream-routing, channel-fusion, cross-attention, interruption-handling, speech-LM]
concepts: ["[[Full-duplexSpokenDialogue]]", "[[SpeechLanguageModel]]", "[[StreamingSpokenDialogue]]", "[[Turn-takinginSpokenDialogue]]", "[[SpokenDialogueEvaluation]]", "[[ModalityAdaptationforSpeechLLM]]"]
models: ["CF-Duplex", "XA-Duplex"]
tasks: ["全双工口语对话"]
datasets: ["LibriSpeech", "GigaSpeech", "PeopleSpeech", "MLS", "CommonVoice", "VoxPopuli", "Emilia-Large", "VoxBox", "OpenAudioBench", "Full-Duplex-Bench v1.0", "Full-Duplex-Bench v1.5"]
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 5 个概念页: [[Full-duplexSpokenDialogue]], [[SpeechLanguageModel]], [[StreamingSpokenDialogue]], [[Turn-takinginSpokenDialogue]], [[SpokenDialogueEvaluation]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: Full-duplexSpokenDialogue (直接匹配), SpeechLanguageModel, StreamingSpokenDialogue, Turn-takinginSpokenDialogue, SpokenDialogueEvaluation | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 全双工口语对话的核心架构挑战是如何在 LLM 生成过程中处理外部用户语音流。KB 中已记录的方案可分为两大路线: (1) **序列交错** (SyncLLM, OmniFlatten, NTPP, SALMONN-omni) -- 将双通道音频切块后交错为单流序列,简单但序列翻倍; (2) **通道融合** (Moshi, LSLM, SLAM-Duplex, FLM-Audio, Fun-Audio-Chat) -- 在每个时间步融合双流,紧凑但融合后无法在架构上区分用户输入和模型生成上下文。本文首次将 "user-stream routing" 提升为显式设计轴,并系统对比了通道融合 (CF-Duplex) 与 **cross-attention routing** (XA-Duplex, 借鉴 Flamingo/AudioFlamingo 的视觉/音频条件化机制) 在相同训练流水线下的 tradeoff。这填补了 KB 中 Full-duplexSpokenDialogue 页的一个空白: 此前对比研究仅在序列交错 vs 通道融合之间进行,cross-attention 作为第三条路线未被系统探索。

**与已有工作的关键差异**: 不同于 [[论文笔记/LSLM|LSLM]] 的 early/middle/late fusion 三种通道融合变体对比 (都属于同一路线), 本文对比的是两条结构性不同的路线 -- 融入 LLM 自回归上下文 vs 保持为外部记忆。不同于 Moshi 的 RQ-Transformer 双流建模, 本文用更轻量的 Qwen3-1.7B backbone + LoRA 即实现了竞争力。

## 速查

> [!summary] 速查
> - **一句话**: 在统一框架下对比全双工口语对话中两种 user-stream routing 策略 -- 通道融合 (CF-Duplex) 语义理解强但打断时易上下文污染, cross-attention routing (XA-Duplex) 语义弱但生成连贯性更鲁棒
> - **路线**: 用户音频 → streaming Whisper encoder (causal conv + causal attn + RoPE) → speech adapter (3 linear, 2x downsample) → [channel fusion | cross-attention adapters] → frozen Qwen3-1.7B + LoRA → text head + audio head (Qwen3-0.6B, G=4 group decoding) → CosyVoice 2 token2wav
> - **指标**: CF-Duplex: ASR WER 3.90/10.04 (clean/other), FDB v1.0 interruption TOR 1.000 + GPT-4o 3.96, FDB v1.5 interruption Respond 0.72 + Stop Latency 0.74s; XA-Duplex: FDB v1.0 Smooth Turn Taking TOR 0.983 (best) [Table 1-5]
> - **可借鉴**: (1) 显式 INT token + 动态 overlap range 训练 (概率分布 [0.6,0.3,0.06,0.03,0.01] 对应 2-6 步延迟) 大幅提升打断处理; (2) cross-attention 将用户流作为外部记忆,避免上下文污染 -- 可在高可靠性场景选用; (3) WAIT/INT 特殊 token 统一建模空闲/打断行为,损失权重 0.001 (WAIT) vs 50 (INT) 平衡稀疏信号
> - **局限**: 仅在 Qwen3-1.7B 单一尺度验证, 未探索混合路由; XA-Duplex QA 性能显著弱于 CF-Duplex, 尚不清楚是架构固有限制还是训练不足; 合成对话数据, 未用真实人机对话训练

## 核心问题

全双工口语对话要求 LLM 在生成语音响应的同时持续监听用户输入。但标准 LLM 本质上是单流 next-token prediction -- 它的 KV cache 和自回归上下文是为单一连贯序列设计的,不自然支持生成过程中的外部信号注入 [§1]。这引出本文的核心研究问题: **用户语音流应该以何种方式路由进 LLM?** [论文原文] 作者明确指出这是全双工建模的"a key architectural question"。

这个问题之所以重要,是因为路由策略决定了用户语音如何与模型正在进行的生成状态交互,进而决定了模型在语音重叠时的鲁棒性 [§1]。[agent 解读] 具体而言,如果用户流被融入 LLM 的自回归上下文 (通道融合),那么在打断场景下,重叠的用户语音会直接污染模型的生成上下文,导致语义不连贯的输出 -- 这是一个此前文献中未被系统刻画的失败模式。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由四个模块组成 [§3, Fig 1]:

1. **Streaming Speech Encoder + Adapter**: 从 Whisper 初始化,改造为因果流式版本 -- 左 padding 卷积 + causal attention mask + 将正弦位置编码替换为 RoPE [§3.1]。输入波形被切成固定大小 chunk,逐块提取 mel-spectrogram 并送入编码器,推理时通过 KV cache 维持跨 chunk 连续性。Speech adapter 由 3 个 linear layer 构成,中间层将连续两帧拼接后投影,实现 2x 时间下采样 [§3.1]。

2. **Backbone LLM**: Qwen3-1.7B, 全程冻结, 仅训练 LoRA adapter (rank=16, α=32) [§6.1]。[agent 解读] 选择冻结 backbone 的策略意味着所有全双工能力必须通过路由模块和 LoRA 来学习,这既控制了训练成本,也使两种路由策略的对比更加公平。

3. **Audio Head**: 从 Qwen3-0.6B 初始化的轻量 decoder LLM, 以 backbone 最后一层 hidden state 为条件, 自回归生成 audio token [§3.2]。[论文原文] 作者解释选择 decoder LM 而非浅层投影层的原因: "mapping backbone representations to audio-token sequences is substantially more complex than standard text-token prediction" [§3.2]。采用 **grouped decoding**: 每个 backbone hidden state 对应生成 G=4 个连续 audio token, 缓解 text-audio 速率不匹配。引入 delay factor D=2, 即 audio decoding 在生成 2 个 text token 后才开始, 保持 text-speech 的大致因果关系 [§3.2]。

4. **Speech Tokenizer & Vocoder**: 使用 CosyVoice 2 的 supervised semantic tokenizer (25 Hz) 和 token2wav vocoder [§3.2]。

### 关键设计选择

#### 设计选择 1: Channel Fusion (CF-Duplex) vs Cross-Attention Routing (XA-Duplex) [§3.3]

这是本文的核心对比轴。

**CF-Duplex (通道融合)**: 在每个对齐时间步, 将用户流 u、模型文本流 m_text、模型音频流 m_audio 拼接后通过 gated MLP 融合 [§3.3, Eq. 1]:
```
c = [u; m_text; m_audio]
y = u + m_text + m_audio + σ(W_g · c + b_g) ⊙ MLP(c)
```
sigmoid gate 控制 MLP 的贡献度。[论文原文] 这种设计让 LLM 在每一步都能直接访问时间对齐的用户流, 但同时也将用户流合并进了用于 token 生成的同一上下文 [§3.3]。

**XA-Duplex (cross-attention routing)**: 用户流作为 **外部记忆** (keys & values), LLM 中间层 hidden states 作为 queries, 通过 cross-attention adapters 访问 [§3.3]。采用 Flamingo 的 XA-Dense 变体, 插入 backbone 偶数层 (layers 2,4,6,...,28), 共 14 层 XA adapter [App F.2, Table 11]。为保持时间对应关系, 用户流和模型流共享相同的 timeline indices 并应用 RoPE [§3.3]。

[论文原文] 选择 cross-attention 的动机: "the user stream is represented as an external memory of keys and values, while the LLM's own generation remains in its native autoregressive context. This provides an explicit and gateable mechanism for attending to user speech without forcing it into the same context used for token generation" [§3.3]。

[agent 解读] 这两种路由策略的核心区别可以理解为: CF 将用户流 "混入" 了模型的生成上下文 (共享 KV cache), 而 XA 将用户流保持为 "旁路" (独立 KV cache)。当两路语音重叠时 (如打断场景), CF 的模型生成上下文会被用户内容污染, 而 XA 的模型自回归上下文不受影响。

#### 设计选择 2: 显式打断 token + 动态 overlap range [§4.1, §6.4]

引入 5 个特殊 token 建模空闲和打断行为:
- 用户侧: `<USER_WAIT>` (用户沉默)
- 模型侧: `<TEXT_WAIT>`, `<AUDIO_WAIT>` (等待/已结束), `<TEXT_INT>`, `<AUDIO_INT>` (检测到打断)

[论文原文] 这些特殊 token 被当作普通预测目标, 使等待和打断行为通过显式 token 级监督学习 [§4.1]。

训练时对 WAIT token 的损失权重降为 0.001 (因为数量多), 对 INT token 的损失权重提升为 50 (因为稀疏但行为关键) [App D]。

**动态 overlap range**: 训练时不固定打断后模型的反应延迟, 而是采用概率分布 [0.6, 0.3, 0.06, 0.03, 0.01] 对应 2-6 步 (320-960ms) 延迟 [§6.4]。[论文原文] 消融实验证明这比固定 2 步或 3 步延迟效果更好 [Table 5]。

#### 设计选择 3: 分组音频解码 (G=4, D=2) [§3.2]

每个 backbone hidden state 对应 G=4 个 audio token (25Hz / 4 = 6.25 text step/s), delay D=2 个 text token。[论文原文] 消融显示 G=4 优于 G=5 (Fun-Audio-Chat 用的 G=5) [App F.1, Table 10]: ASR WER 2.90/8.17 vs 3.32/8.47, TTS WER 2.13/2.37 vs 2.96/3.25。

### 训练策略

三阶段课程学习 [§4.2]:

- **Stage 1**: 训练 speech encoder, speech adapter, routing 模块, audio head, LoRA -- 在 ASR (217k h, 7 个英语语料库) + streaming TTS (VoxBox 104k h) 上训练。
- **Stage 2**: 冻结 speech encoder + backbone LLM, 训练其余模块 -- 在 ASR + streaming TTS + S2TD (speech-to-text dialogue) + S2TSD (speech-to-text-and-speech dialogue) 上训练。对话数据由 Qwen3-30B 将文本 QA 改写为口语风格后用 IndexTTS-2 合成, 1.9M 样本 [§5]。
- **Stage 3**: 同 Stage 2 的可训练模块 -- 在 ASR + streaming TTS + **全双工对话** 上训练。全双工数据通过在 turn-based 对话基础上模拟打断 (context-dependent + context-independent) 和 backchannel, 训练时在线动态合成 (on-the-fly), 随机化打断时机 [§5]。

[agent 解读] on-the-fly 合成全双工数据而非预合成是一个值得注意的设计: 它提供了更多样化的打断时机和语义组合, 相当于隐式的数据增强。

硬件: 16x NVIDIA H200 GPU, AdamW optimizer [App D, Table 9]。

**模型规模**: CF-Duplex 总参数 2.90B (可训练 1.16B), XA-Duplex 总参数 3.40B (可训练 1.68B)。XA-Duplex 多出 469M 参数来自 cross-attention adapters, 少了 37.8M 的 channel fusion 模块 [Table 8]。

## 实验

| 指标 | CF-Duplex | XA-Duplex | 对比基准 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| ASR WER (clean/other) | 3.90 / 10.04 | 4.56 / 12.49 | Whisper-large-v3: 2.01/3.91 | LibriSpeech | [Table 4] |
| TTS WER (EN/ZH) | 2.93 / 3.37 | 2.83 / 3.20 | CosyVoice 2: 2.57/1.45 | seed-tts-eval | [Table 4] |
| QA LLaMAQ (speech/text) | 50.7 / 57.3 | 38.3 / 40.3 | Freeze-Omni: 56.2/74.2 (7B backbone) | OpenAudioBench | [Table 1] |
| QA TriviaQ (speech/text) | 18.1 / 19.6 | 8.0 / 8.2 | Freeze-Omni: 28.5/45.1 | OpenAudioBench | [Table 1] |
| QA WebQ (speech/text) | 28.0 / 30.3 | 18.4 / 18.7 | Freeze-Omni: 27.9/40.8 | OpenAudioBench | [Table 1] |
| QA AlpacaEval | 3.94 / 4.16 | 3.87 / 4.04 | Freeze-Omni: 2.46/3.90 | AlpacaEval | [Table 1] |
| FDB v1.0 Interruption TOR | **1.000** | 0.971 | Moshi: 1.000 | Full-Duplex-Bench v1.0 | [Table 2] |
| FDB v1.0 Interruption GPT-4o | **3.96** | 2.23 | Freeze-Omni: 3.615 | Full-Duplex-Bench v1.0 | [Table 2] |
| FDB v1.0 Interruption Latency | 0.374 | **0.325** | Moshi: 0.257 | Full-Duplex-Bench v1.0 | [Table 2] |
| FDB v1.0 Turn Taking TOR | 0.924 | **0.983** | dGSLM: 0.975 | Full-Duplex-Bench v1.0 | [Table 2] |
| FDB v1.5 Interruption Respond | **0.72** | 0.32 | Freeze-Omni: 0.72 | Full-Duplex-Bench v1.5 | [Table 3] |
| FDB v1.5 Interruption Stop Latency | **0.74s** | 1.18s | Moshi: 1.16s | Full-Duplex-Bench v1.5 | [Table 3] |
| FDB v1.5 Backchannel Resume | **0.96** | 0.86 | Freeze-Omni: 0.80 | Full-Duplex-Bench v1.5 | [Table 3] |

### 关键发现

**发现 1: CF-Duplex 在语义理解上全面优于 XA-Duplex** [§6.3.1, Table 1]。在 4 个 QA 数据集上, CF-Duplex 的 speech/text 分数均显著高于 XA-Duplex, 且 gap 从 Stage 2 开始出现并在 Stage 3 持续 [§6.3.4, Table 4]。[论文原文] 这表明通道融合提供了更强的语义接地能力 (semantic grounding), CF-Duplex "effectively support spoken dialogue understanding even under a compact model scale" [§6.3.1]。

**发现 2: XA-Duplex 在打断失败时保持生成连贯性** [§6.3.3, App G]。当两种模型都未能及时停止 (missed interruption) 时, CF-Duplex 的后续生成变得语义不连贯 (混入用户打断内容), 而 XA-Duplex 虽然也未停止但继续生成连贯的原始响应 [§6.3.3, Fig 5-7]。[论文原文] 这说明 XA-Duplex "better preserves the LLM generation context and is more robust to this failure mode" [§6.3.3]。

**发现 3: 显式 INT token + 动态 overlap range 的组合是最佳配置** [§6.4, Table 5]。去掉 INT token 后 Respond 从 0.72 降至 0.56, stop latency 从 0.74s 升至 1.90s; 固定 overlap=2 时 Respond 降至 0.58 [Table 5]。

**发现 4: CF-Duplex 用 1.7B backbone 实现了接近 7B 全双工系统的 QA 性能**。AlpacaEval 上 CF-Duplex (3.94) 超越了所有对比系统 (Freeze-Omni 2.46, Moshi 1.76) [Table 1], 但在需要更多参数化知识的 TriviaQ 上仍落后于 7B Freeze-Omni [Table 1]。

## 局限性

1. **仅两种路由策略**: 未探索混合路由 (如某些层用 CF, 某些层用 XA) 或其他可能的路由设计 [§8]。
2. **单一模型尺度**: 所有实验在 Qwen3-1.7B backbone 上进行, tradeoff 是否在更大模型上仍然成立未知 [§8]。
3. **XA-Duplex 的 QA 差距过大**: speech score 差距 12.4 (LLaMAQ) 到 10.1 (TriviaQ), 是否可通过更多 XA 参数或不同训练策略缓解, 论文未深入分析。[agent 解读]
4. **合成对话数据**: 全双工数据由 LLM 改写 + TTS 合成构建, 与真实人机对话分布可能有差距。[agent 解读]
5. **未评估 backchannel 生成**: 仅评估了对用户 backchannel 的处理, 未评估模型主动生成 backchannel 的能力。[agent 解读]

## 点评

本文的核心贡献不在于提出新模型,而在于 **将 user-stream routing 提升为全双工口语对话的显式设计轴并给出受控对比**。这种"问正确的问题"的研究范式对领域推进有重要价值: 在此之前, 各全双工系统各自选择路由策略但缺乏 controlled ablation, 本文通过共享 backbone/tokenizer/training pipeline/data 的实验设计, 首次清晰揭示了 **语义接地 vs 上下文鲁棒性** 的 tradeoff。

特别值得关注的是 context corruption 这一失败模式的发现 [§6.3.3]: CF-Duplex 在打断失败时的输出 (如 "Bevercrackle the Four for Survive at Did Espec the Hoselles for Trip", Fig 5) 暴露了通道融合的结构性弱点 -- 用户语音的语义内容被不可控地注入了生成上下文。这对所有采用通道融合的全双工系统 (包括 LSLM, Moshi, Fun-Audio-Chat) 都是一个重要的 warning signal。

工程层面, 显式 INT token + 动态 overlap range 的训练策略 [§6.4] 和 WAIT/INT 的非对称损失权重 (0.001 vs 50) 是直接可复用的 trick。on-the-fly 合成全双工训练数据的做法也值得借鉴。

不足之处: XA-Duplex 的 QA 性能差距过大 (在 TriviaQ 上仅 CF 的 44%), 削弱了"tradeoff"的实用性 -- 如果 XA 在核心任务上差太多, 那它的鲁棒性优势就难以被选择。论文也承认仅在 1.7B 尺度验证, 是否 scaling 能改变 tradeoff 平衡点是开放问题。

## 可复用的 idea

1. **显式 INT token + 非对称损失权重**: 对稀疏但行为关键的特殊 token (打断/状态转换) 大幅上调损失权重 (50x), 对高频但信息量低的 token (等待/沉默) 大幅下调 (0.001x)。这种做法可推广到任何需要学习稀疏事件的序列建模任务。
2. **动态 overlap range 训练**: 用概率分布 [0.6, 0.3, 0.06, 0.03, 0.01] 控制打断后的反应延迟,比固定延迟更鲁棒。本质上是对反应时间的 curriculum/augmentation。
3. **Cross-attention 作为外部记忆路由**: 当需要保护 LLM 生成上下文不被外部输入污染时 (如高可靠性对话、安全敏感场景), 可选择 XA routing 而非 channel fusion。
4. **On-the-fly 全双工数据合成**: 训练时在线随机组合打断/backchannel 到 base 对话中, 比预合成固定数据提供更多样化的训练信号。
5. **Grouped audio decoding (G=4) + delay (D=2)**: 缓解 text-audio 速率不匹配的简洁方案, 比逐 token 解码高效且不损失太多质量。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 核心 tradeoff (语义接地 vs 上下文鲁棒性) 解释清晰, 方法节含因果解释 |
> | 可信赖 | pass | 关键数字均标注 [Table N]/[§X.X], 覆盖率 >90% |
> | 可区分 | pass | [论文原文] vs [agent 解读] 标注覆盖 >80% |
> | 可定位 | pass | KB 背景含具体谱系定位, 与 LSLM/Moshi/Flamingo 的关系明确 |
> | 不污染 | pass | 未新建概念页, 不涉及 KB 写入 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/Full-DuplexSpokenDialogue-review.yml`
