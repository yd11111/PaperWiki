---
type: paper
tier: deep
title: "Preserving Speech-to-Text LLM Capabilities in Speech-to-Speech Generation"
arxiv_id: "2606.30944"
source: "Sources/PRIME-Speech.pdf"
authors: [Yuxuan Hu, Heng Lu, Ruchao Fan, Yao Qian, Xiaofei Wang, Jian Xue, Heming Wang, Shuohang Wang, Young Jin Kim, Yelong Shen, Jinyu Li]
year: 2026
venue: "arXiv"
tags: [speech-to-speech, LLM, frozen-backbone, catastrophic-forgetting, multi-token-prediction, codec-language-model, hidden-state-synchronization, multi-turn-dialogue]
concepts: ["[[SpeechLanguageModel]]", "[[LLM-basedTTS]]", "[[CodecLanguageModel]]", "[[SpeechTokenizer]]", "[[ModalityAdaptationforSpeechLLM]]", "[[StreamingSpokenDialogue]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: ["PRIME-Speech", "Phi-4-MM-7B", "[[CosyVoice2]]", "[[Whisper]]", "GLM-4-Voice", "Kimi-Audio", "Step-Audio-2-Mini", "VocalNet", "Qwen2.5-Omni", "Qwen3-Omni", "GPT-4o"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["LibriHeavy", "CoVoST-2", "FLEURS", "UltraEval-Audio", "VocalBench", "BigBench-Audio", "TriviaQA", "VoiceAssistant-400K"]
kb_context_sources: 6
status: draft
created: 2026-07-01
updated: 2026-07-01
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: PRIME-Speech 属于 [[Speech-LLMIntegrationTaxonomy]] 中 audio-token-based integration 的变体,但走了一条独特路线: 不是将 speech token 混入 LLM 的自回归流 (如 GLM-4-Voice/Moshi 的 token interleaving),也不是传统 Thinker-Talker 分离 (如 Qwen2.5-Omni 等待文本生成完毕再驱动 talker),而是**冻结整个 S2T backbone** 并通过 hidden-state 同步驱动独立的 audio post-decoder。

**已有认知**:
- [[SpeechLanguageModel]] (confirmed): SpeechLM 从 ASR+LLM+TTS 级联向端到端演进,核心挑战包括信息丢失、高延迟、错误累积。S2S 系统演进路线为 GSLM → AudioLM → TWIST → SpeechGPT → Moshi → GLM-4-Voice。本文关注的"S2T→S2S 转换"是该演进中的新焦点。
- [[LLM-basedTTS]] (confirmed): LLM-based TTS 是 SpeechLM 在生成侧的特例。PRIME-Speech 的 audio post-decoder 本质上是一个条件 codec LM,但其条件来源不是最终文本而是 backbone 的中间 hidden states。
- [[CodecLanguageModel]] [待确认]: Codec LM 直接在 neural codec tokens 上做 next-token prediction。PRIME-Speech 使用 CosyVoice2 的 25Hz semantic codec tokens 作为目标,属于 single-codebook codec LM 路线。
- [[SpeechTokenizer]] (confirmed): PRIME-Speech 采用 CosyVoice2 tokenizer (25Hz semantic codec),属于监督式 semantic tokenizer 路线,token 主要编码语义信息,声学细节由 codec decoder 负责。
- [[ModalityAdaptationforSpeechLLM]] [待确认]: 本文的 hidden-state synchronization 可视为一种新型 modality adaptation — 不是将语音帧映射到 LLM 输入空间 (输入侧适配),而是将 LLM 中间层 hidden states 映射到语音生成空间 (输出侧适配)。
- [[StreamingSpokenDialogue]] [待确认]: PRIME-Speech 的 timestamp-synchronized loop + multi-turn cache reset 属于 streaming spoken dialogue 技术栈,但聚焦于"如何在不改变 backbone 的前提下实现流式语音生成"。

**创新判断**: 相比 GLM-4-Voice (text-audio interleaved, 需要重训 backbone) 和 Qwen2.5-Omni (Thinker-Talker 但 talker 等待文本), PRIME-Speech 的核心创新在于 frozen-backbone + hidden-state synchronization 的组合 — 这避免了 catastrophic forgetting 同时实现了 streaming audio 生成,是 S2T→S2S 转换问题的一个新 formulation。

> 检索命中: [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[StreamingSpokenDialogue]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 冻结 S2T LLM 全部参数,仅训练 2B audio post-decoder + MTP heads,通过 hidden-state synchronization 让 codec token 生成跟随 backbone 的推理轨迹而非等待完成的文本
> - **路线**: 输入语音 → frozen Phi-4-MM backbone (speech encoder + transformer) → 中间层 hidden states $h^{(\ell_{mid})}$ → audio post-decoder (混合条件: hidden state + text embedding + audio history) → CosyVoice2 codec tokens (25Hz) → waveform
> - **指标**: S2S WER 3.34% (multi-turn) / BLEU 33.24 (FLEURS X2EN) / BigBench-Audio S2S 63.4 vs backbone S2T 66.5; S2T 路径指标与 frozen backbone 几乎不变 [Table II]
> - **可借鉴**: (1) 中间层 hidden state 作为 text-audio 同步接口的设计 — 不需要 force alignment 或等待完整文本; (2) multi-turn 中 text KV cache 累积 + audio KV cache 重置的策略 — 简单有效防止跨 turn 声学漂移; (3) MTP 作为 efficiency adapter 而非语义对齐工具的定位 — 先训稳 single-token 再加 MTP
> - **局限**: 训练数据全部用 Azure TTS 合成目标语音,未验证真实语音目标; 仅支持英语; 2B post-decoder + 7B backbone 总参数 9B,推理效率 k=1 时 RTF=1.088 不满足实时; 未评估说话人一致性/韵律自然度等语音质量维度

## 核心问题

PRIME-Speech 要解决的核心问题是: **如何将一个已经训练好的 S2T LLM 转换为 S2S 模型,而不损失其原有的语音理解、文本推理和指令遵循能力?**

这个问题的难点在于两难: (1) 直接在 backbone 上做 S2S 微调会导致 catastrophic forgetting — 文本推理能力下降 [§I]; (2) 简单串联一个 TTS 模块则退化为 cascade pipeline — 必须等待完整文本响应后才能生成语音,延迟高且无法利用 backbone 的中间推理状态 [§I]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

PRIME-Speech 的核心架构如 [Fig 1] 所示: 一个完全冻结的 S2T backbone (Phi-4-MM-7B) 负责语音感知和文本推理,一个可训练的 audio post-decoder (2B params, 10 层 causal transformer) 附加在 backbone 的中间层上,通过 hidden-state synchronization 在同一个 streaming update loop 中并行生成文本和语音。

整体生成过程可形式化为:

$$P(y^\tau, y^a \mid x) = P_{bb}(y^\tau \mid x) \cdot P_{aud}(y^a \mid y^\tau, H^{mid}; \theta_a)$$

其中 $P_{bb}$ 是冻结 backbone 的分布 (不可训练), $P_{aud}$ 是 audio branch 的分布 (可训练), $H^{mid}$ 是 backbone 中间层 hidden states 序列, $\theta_a$ 是 audio branch 参数 [Eq. 1]。

**为什么选择完全冻结而非 LoRA 微调?** [论文原文] 消融实验 (Table III) 表明, LoRA + Post LM 和 LoRA + ESI 变体虽然在 S2S 指标上可行,但它们的 BigBench-Audio S2T 分数更低 (52.96/53.75 vs 66.20),证明了更新 backbone 会损害推理能力 [§IV-C.1]。

### 关键设计选择

#### 1. Hidden-State Synchronization (核心创新)

在每个 streaming update step $s$, backbone 暴露一个中间层 hidden state $h^{mid}_s$, 文本 head 和 audio post-decoder 同时消费这个状态 [§II-C]:

$$P(y^a_s \mid y^a_{<s}, y^\tau_{<s}, H^{mid}_{\le s}; \theta_a)$$

关键约束: audio branch 是因果的 — 它条件于当前 hidden state 和历史 (committed before $s$), 但**不能访问同一 update 中正在预测的文本 token $y^\tau_s$**。text token 在 step $s$ 生成后,变为 step $s+1$ 的 audio history [Eq. 4]。

**为什么用中间层而非最后一层?** [论文原文] 作者使用 layer-wise CKA (Centered Kernel Alignment) 分析 [39] 选择了约 2/3 深度处的层,该深度携带最丰富的副语言信息 (paralinguistic information) 同时保留语义基础 [§II-B]。[agent 解读] 最后几层过于特化于文本 token 预测,可能丢失语音相关的声学/韵律信息;过浅的层语义信息不足。

**为什么是 timestamp-level 而非 chunk-level 同步?** [论文原文] PRIME-Speech 不等待完成的文本或固定文本 chunk,也不需要 word-to-frame force alignment。audio branch 不在一个与 backbone 轨迹脱离的独立时钟上运行 [§II-C]。[agent 解读] 这使得系统可以在 backbone 仍在推理过程中就开始生成语音,避免了 Thinker-Talker 架构中 talker 必须等待 thinker 完成的延迟瓶颈。

#### 2. Mixed Conditioning (混合条件向量)

audio post-decoder 的输入不仅仅是 hidden state,而是三个信号的混合 [Eq. 6]:

$$h^{mix}_s = w_h \cdot h^{mid}_s + w_\tau \cdot e^\tau_{s-1} + w_a \cdot r^a_{s-1}$$

其中:
- $h^{mid}_s$: backbone 中间层 hidden state — 提供**语义状态** (semantic state)
- $e^\tau_{s-1}$: 上一步 committed 的 text token embedding — 提供**词汇锚定** (lexical anchoring)
- $r^a_{s-1}$: 上一步 audio block 的 codec token embeddings 均值 — 提供**声学连续性** (acoustic continuity)

权重在实验中固定为 $w_h = w_\tau = w_a = 1.0$,是 held-out ablations 中的最佳简单设置 [§II-C]。

[agent 解读] 这三个信号分别解决不同问题: hidden state 告诉 post-decoder "backbone 当前在想什么"; text embedding 告诉它 "上一个词是什么" (防止重复或跳字); audio history 告诉它 "上一帧声音是什么" (保持韵律连贯)。

#### 3. Multi-Turn Cache Policy (多轮缓存策略)

这是一个简单但被作者标注为 "critical" 的设计 [§II-E]:

$$C^{(n)}_m = \begin{cases} C^{(<n)}_\tau \oplus \{K^{(n)}_\tau, V^{(n)}_\tau\}, & m = \tau \text{ (text: 累积)} \\ \{K^{(n)}_a, V^{(n)}_a\}, & m = a \text{ (audio: 重置)} \end{cases}$$

即 text KV cache 跨 turn 累积以保持对话语义记忆,但 audio KV cache 在每个 assistant turn 开始时重置 [Eq. 9]。

**为什么 audio cache 必须重置?** [论文原文] Table IV 提供了因果诊断: 不重置 audio cache 时,前两轮表现尚可,但从第三轮起 WER 急剧上升 (Turn 3: 65.57%; Turn ≥5: 143.27%), S2S 准确率在 ≥5 轮时降为 0%。由于 text cache 在两种设置中都累积,退化被隔离到复用的 audio-side 状态 [§IV-C.3]。

**训练时的 packing 策略**: 由于缺乏多轮 S2S 数据,作者将不相关的单轮样本拼接为伪对话 (packed pseudo-dialogues), 让 audio branch 暴露在长 text context 和显式 turn boundaries 下,同时阻止模型将上一轮的声学实现当作有用上下文 [§II-E]。audio 位置索引在每个 packed segment 开头重置 [Eq. 11]:

$$P^{(n)}(i) = \begin{cases} i, & m_i \in \text{Text} \\ i - s_n, & m_i \in \text{Audio}_n \end{cases}$$

### 训练策略

训练分两个阶段 [§III-C]:

**Stage 1 — Audio-branch 训练**: 标准 next-token prediction,25Hz codec rate,全任务混合数据训练一个 epoch。AdamW, lr=$1 \times 10^{-4}$, linear decay。目标是建立 hidden-state-to-codec 对齐 [§III-C.1]。

**Stage 2 — MTP 训练**: 启用 Multi-Token Prediction, 继续训练 20k steps。此时 post-decoder 已学会稳定的 hidden-state conditioning, MTP 只作为效率组件训练 [§III-C.2]。

**训练数据** (~100k 加权小时) [Table I]:
- LibriHeavy (46k hrs): TTS-style alignment, 提供密集 audio-text 对应
- In-house X2EN (10k hrs) + CoVoST-2 X2EN (1k hrs): 多语言翻译,训练 post-decoder 渲染推断内容
- VoiceAssistant-400K (4k hrs) + TriviaQA (2k hrs): QA 风格响应

所有目标语音由 Microsoft Azure TTS 合成,CosyVoice2 tokenizer 编码为 25Hz codec tokens [§III-A]。

## 关键公式

**核心分解** [Eq. 1]:
$$P(y^\tau, y^a \mid x) = P_{bb}(y^\tau \mid x) \cdot P_{aud}(y^a \mid y^\tau, H^{mid}; \theta_a)$$

**Audio post-decoder 条件分布** [Eq. 4]:
$$P(y^a_s \mid y^a_{<s}, y^\tau_{<s}, H^{mid}_{\le s}; \theta_a)$$

**Mixed conditioning** [Eq. 6]:
$$h^{mix}_s = w_h \cdot h^{mid}_s + w_\tau \cdot e^\tau_{s-1} + w_a \cdot r^a_{s-1}$$

**MTP 目标** [Eq. 8]:
$$\mathcal{L}_{mtp} = -\sum_s \sum_{i=1}^{k} \lambda_i \log p_{s,i}(y^a_{s,i})$$

MTP 将有效 codec 预测率从 25Hz 降至 $25/k$ Hz,k=4 时为 6.25Hz [§II-D]。

**Multi-turn cache reset** [Eq. 9-10]:
$$P(y^a_t \mid C^{(<n)}_\tau, C^{(n)}_a, H^{mid}; \theta_a)$$

## 实验

### 主实验结果 [Table II]

| 指标 | PRIME-Speech-9B | Backbone S2T | Qwen2.5-Omni-7B | GLM-4-Voice-9B | 出处 |
| --- | --- | --- | --- | --- | --- |
| FLEURS X2EN BLEU (S2T/S2S) | 31.40/33.24 | 31.41/– | 34.59/5.94 | –/– | [Table II] |
| CoVoST X2EN BLEU (S2T/S2S) | 41.29/40.98 | 40.65/– | 39.72/10.52 | –/– | [Table II] |
| UltraEval-Audio LLaMA-QA (S2T/S2S) | 79.00/74.42 | 78.67/– | 76.33/71.00 | 64.70/50.70 | [Table II] |
| UltraEval-Audio TriviaQA (S2T/S2S) | 46.98/44.54 | 47.07/– | 47.66/45.60 | 39.10/26.50 | [Table II] |
| Multi-turn S2T | 80.45 | 79.33 | 69.83 | 74.86 | [Table II] |
| Multi-turn S2S | 79.33 | – | 67.04 | 70.95 | [Table II] |
| Multi-turn WER↓ | 3.33 | – | 21.5 | – | [Table II] |
| BigBench-Audio S2T | 66.2 | 66.5 | 54.2 | 44.8 | [Table II] |
| BigBench-Audio S2S | 63.4 | – | 53.6 | 42.7 | [Table II] |
| VocalBench Single | 4.77 | – | 4.96 | 3.64 | [Table II] |
| VocalBench Overall | 4.38 | – | 3.54 | 3.62 | [Table II] |
| VocalBench WER↓ | 3.34 | – | 4.23 | 7.83 | [Table II] |
| VocalBench Flu. (UTMOS) | 4.29 | – | 3.54 | 3.87 | [Table II] |

**关键观察**:
1. **S2T 保持**: PRIME-Speech 的 S2T 指标与 frozen backbone 几乎一致 (BigBench-Audio: 66.2 vs 66.5, FLEURS: 31.40 vs 31.41), 证明 frozen-backbone 策略有效 [§IV-B.1]
2. **S2T-S2S gap 小**: 例如 FLEURS S2T 31.40 → S2S 33.24 (甚至略升, 可能因为 ASR 转写引入的变化); UltraEval-Audio 整体 gap 约 2-5 点 [§IV-B.2]
3. **对比 Qwen2.5-Omni**: 翻译任务上 Qwen2.5-Omni S2T→S2S 巨幅下降 (FLEURS: 34.59→5.94), 而 PRIME-Speech 保持稳定,说明 unified token interleaving 架构的 S2T-S2S modality gap 问题 [Table II]

### 消融实验 [Table III]

| Variant | FLEURS S2S | Multi-turn WER↓ | BigBench S2S | 出处 |
| --- | --- | --- | --- | --- |
| LoRA + ESI | 31.11 | 3.25 | 53.25 | [Table III] |
| LoRA + Post LM | 30.96 | 5.13 | 52.36 | [Table III] |
| PRIME S1 Model (25Hz) | 33.57 | 6.12 | 59.10 | [Table III] |
| + MTP=1 (25Hz) | 33.58 | 5.66 | 63.86 | [Table III] |
| + MTP=2 (12.5Hz) | 33.56 | 3.01 | 64.16 | [Table III] |
| + MTP=4 (6.25Hz) | 33.24 | 3.33 | 63.38 | [Table III] |

**消融关键发现**:
1. **LoRA 变体 BigBench S2T 下降**: LoRA + Post LM 的 BigBench S2T=52.96, 显著低于 PRIME-Speech 的 66.20, 验证了 frozen backbone 的必要性 [§IV-C.1]
2. **MTP 的作用是效率而非语义**: S1→MTP=1 (same 25Hz rate) 已提升 BigBench S2S (59.10→63.86), 说明 Stage 2 curriculum 本身改善了 rendering; MTP=1→4 进一步将有效 codec rate 从 25Hz 降至 6.25Hz, S2S 指标基本稳定 [§IV-C.2]

### Multi-Turn Cache 消融 [Table IV]

| Cache Policy | Turn 1 Acc/WER | Turn 3 Acc/WER | Turn ≥5 Acc/WER | 出处 |
| --- | --- | --- | --- | --- |
| Text accum + audio reset | 92.86/2.44 | 71.43/1.97 | 73.13/1.48 | [Table IV] |
| w/o audio reset | 92.86/2.77 | 39.29/65.57 | 0.00/143.27 | [Table IV] |

不重置 audio cache 导致 WER 从 Turn 3 开始爆炸, Turn ≥5 时 WER 超过 143%, 准确率归零 [§IV-C.3]。

### 推理效率 [Table V]

| System | Frame Rate | TTFA↓ (s) | RTF↓ | Throughput (tok/s) | 出处 |
| --- | --- | --- | --- | --- | --- |
| PRIME-Speech (k=1) | 25Hz | 1.07 | 1.088 | 30.62 | [Table V] |
| PRIME-Speech (k=2) | 12.5Hz | 0.63 | 0.548 | 62.17 | [Table V] |
| PRIME-Speech (k=4) | 6.25Hz | 0.39 | 0.296 | 123.76 | [Table V] |
| Qwen2.5-Omni-7B | 50Hz | 1.01 | 1.093 | 45.75 | [Table V] |
| VocalNet-8B (k=1) | 12.5Hz | 0.51 | 0.250 | 216.89 | [Table V] |

MTP k=1→4: TTFA 从 1.07s 降至 0.39s, RTF 从 1.088 降至 0.296 [§IV-D]。VocalNet 的 shallow talker 吞吐更高但 S2T-S2S gap 更大 [Table II vs Table V]。

## 局限性

1. **合成目标语音**: 所有训练数据的目标语音由 Azure TTS 合成,没有使用真实人声作为目标。这意味着 (a) 学到的声学模式受限于 TTS 系统的能力范围; (b) 韵律、情感表达等维度未被真实验证 [§III-A]
2. **语音质量评估不完整**: 评估主要关注 transcript-level task correctness 和 WER,仅报告 UTMOS fluency score,未评估 speaker consistency、prosody naturalness、emotion expressiveness 等维度 [§III-A]
3. **仅支持英语**: 虽然输入侧支持多语言 (翻译任务),输出侧仅生成英语语音 [Table I]
4. **k=1 时不满足实时**: RTF=1.088 意味着生成速度略慢于实时,需要 MTP k≥2 才能达到实时 (RTF=0.548) [Table V]
5. **2B post-decoder 开销**: 总参数 9B (7B backbone + 2B post-decoder),推理时需同时运行两个大模型,对部署资源要求高 [§III-B]
6. **CKA 层选择的 generalizability**: 2/3 depth 的中间层选择基于特定 backbone 的 CKA 分析,是否适用于其他 backbone 架构未验证 [§II-B] [⚠️ 论文未详述]
7. **训练数据规模相对有限**: ~100k 小时加权数据,与 Qwen2.5-Omni 等系统的数据规模差距可能影响公平比较 [agent 解读]

## 点评

**方法论层面**:
- PRIME-Speech 将 S2S 适配清晰地 formulate 为 "frozen-backbone conversion problem",这个 problem framing 本身有价值 — 它明确了保留 S2T 能力是硬约束而非优化目标,从而自然推导出冻结 backbone + 外挂 speech branch 的设计
- Hidden-state synchronization 是本文的核心贡献,它提供了一种介于 token interleaving (如 GLM-4-Voice) 和 text-conditioned talker (如传统 Thinker-Talker) 之间的中间路线: 语音不是从文本 token 生成,而是从 backbone 的"思考过程"中实时读取
- Multi-turn cache policy 的消融 (Table IV) 是全文最有说服力的实验之一 — turn-wise 的退化轨迹清楚地展示了 audio cache 重置的因果效应

**实验层面**:
- 评估覆盖面广 (翻译/QA/理解/多轮/效率),但缺少语音质量的细粒度评估是明显短板
- 与 backbone 的 controlled comparison 是最有说服力的证据 (S2T 指标几乎不变),跨系统比较受限于 backbone/data/decoding 差异
- MTP 的效率提升显著且实用 (RTF 1.088→0.296),但论文没有分析 MTP k 值与语音质量 (如 UTMOS) 的 trade-off

**方向价值**:
- Frozen-backbone S2S conversion 是一个有实际价值的 research direction — 工业界已有大量投资在 S2T LLM 上,如何低成本将其升级为 S2S 是 deployment 关切
- 但长期来看,frozen backbone 可能限制了 speech generation 的表达力上限 — backbone 的 hidden states 是为文本预测优化的,可能不包含足够的语音生成信号

## 可复用的 idea

1. **Hidden-state synchronization 作为跨模态桥接**: 利用 LLM 中间层 hidden states 驱动另一个模态的生成,避免等待完整文本输出。可推广到任何 "text LLM + 另一模态输出" 的场景 (如视频/手势生成)
2. **Text cache 累积 + audio cache 重置**: 处理多轮交互时,区分"需要跨轮保持的信息"和"应当重新开始的信息"。对所有多轮生成系统 (不限于语音) 都适用
3. **MTP 作为 efficiency adapter 的 staged training**: 先训练 single-token 模型建立稳定对齐,再加 MTP 压缩 — 避免随机初始化的模型直接做长 horizon 预测。可用于所有高帧率序列生成任务
4. **训练时 packing unrelated samples 模拟多轮**: 用不相关单轮数据拼接为伪对话,低成本获得多轮能力。关键是配合 audio position reset 使用
5. **CKA 层选择**: 用 CKA 分析找到 backbone 中最适合驱动语音的层,可迁移到其他 multimodal output 任务

> [!review] 审阅: pass-with-fixes
> - **审阅人**: auto | **日期**: 2026-07-01
> - **原则得分**: reproducible=8 ✅ | trustworthy=8 ✅ | distinguishable=7 ✅ | positionable=9 ✅ | non-polluting=8 ✅
> - **Issues**: 0 high / 1 medium / 2 low
>   - [medium] template-compliance @ `## 关键公式`: 模板中无此 section header,但用户明确要求包含,保留
>   - [low] traceability-gap @ `## 局限性 第 7 点`: 训练数据规模判断缺少对比数字
>   - [low] weak-reusability @ `## 可复用的 idea 第 5 点`: CKA 层选择描述偏简略
> - **详见**: `_review/PRIME-Speech-review.yml`
