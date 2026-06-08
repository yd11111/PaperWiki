---
type: paper
tier: deep
title: "SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation (v1 Technical Report)"
arxiv_id: "2411.18138"
source: "Sources/SALMONN-omni-v1.pdf"
authors: [Wenyi Yu, Siyin Wang, Xiaoyu Yang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Guangzhi Sun, Lu Lu, Yuxuan Wang, Chao Zhang]
year: 2024
venue: "arXiv (Technical Report)"
tags: [full-duplex, speech-LM, codec-free, thinking-mechanism, streaming, turn-taking, barge-in, echo-cancellation, end-to-end, embedding-based]
concepts: ["[[概念库/Full-duplexSpokenDialogue|Full-duplex Spoken Dialogue]]", "[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/StreamingSpokenDialogue|Streaming Spoken Dialogue]]", "[[概念库/Turn-takinginSpokenDialogue|Turn-taking in Spoken Dialogue]]", "[[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLM Integration Taxonomy]]"]
models: ["[[论文笔记/Moshi|Moshi]]", "[[论文笔记/LSLM|LSLM]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 4 个待确认实体页: [[概念库/SpeechLanguageModel|SpeechLanguageModel]]✓, [[概念库/Full-duplexSpokenDialogue|Full-duplexSpokenDialogue]][待确认], [[概念库/StreamingSpokenDialogue|StreamingSpokenDialogue]][待确认], [[概念库/Turn-takinginSpokenDialogue|Turn-takinginSpokenDialogue]][待确认], [[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLMIntegrationTaxonomy]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[概念库/SpeechLanguageModel|SpeechLanguageModel]]✓ | 过滤: [[概念库/Full-duplexSpokenDialogue|Full-duplexSpokenDialogue]](pending-review), [[概念库/StreamingSpokenDialogue|StreamingSpokenDialogue]](pending-review), [[概念库/Turn-takinginSpokenDialogue|Turn-takinginSpokenDialogue]](pending-review), [[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLMIntegrationTaxonomy]](pending-review) | 未命中但可能相关: [[概念库/SpokenDialogueEvaluation|SpokenDialogueEvaluation]]
>
> **Speech Language Model**: KB 记录了 SpeechLM 从 GSLM→AudioLM→SpeechGPT→Moshi 的演进,其中全双工系统主要走 "audio-token-based" 路线 (Moshi 将 codec tokens 注入 LLM 词表)。本文 v1 是首次提出 codec-free 的全双工 SpeechLM 概念框架,通过 embeddings 而非 discrete tokens 连接 encoder/synthesizer 与 LLM,属于 KB 中 "latent-representation-based" 路线在全双工场景的首次探索。
>
> **[基于未确认概念页] Full-duplex Spoken Dialogue**: KB 已收录该 v1 版本 ("SALMONN-omni (Wu et al., 2024)")。在全双工系统谱系中,v1 是第一个同时满足 "codec-free" 和 "单 LLM" 两个条件的系统。此前 Moshi/SyncLLM 使用 codec injection,LSLM 仅做边说边听但不是完整全双工对话系统。
>
> **[基于未确认概念页] Turn-taking**: KB 将 turn-taking 实现分为 VAD-based、dual-tower 隐式建模 (dGSLM)、multi-stream (Moshi)、irq/n-irq markers (Mini-Omni2)、chunk-level state prediction (Freeze-Omni) 等路线。本文 v1 提出的 `<start_speak>`/`<end_speak>` + `<think>` 机制是一种新的显式状态转换路线,用特殊 token 控制说/听切换。

> [!summary] 速查
> - **一句话**: 首篇提出 codec-free 全双工 Speech LLM 概念框架的技术报告,通过 streaming encoder + LLM + streaming synthesizer 的 embedding 桥接 + "thinking" 状态转换机制实现同时听说 [§1, §2]
> - **路线**: streaming speech input → Streaming Speech Encoder (auditory embeddings, 每 block 固定时长 delta_t) → LLM (生成 n 个 word embeddings/block, 含 `<start_speak>`/`<end_speak>`/`<think>` 状态 token) → Streaming Speech Synthesizer (cross-attention, 生成 delta_t 时长语音) [§2, Fig 2]
> - **指标**: 无定量评估表格,仅提供 case study 定性展示 (streaming ASR, speech enhancement, spoken QA, context-independent/dependent barge-in, echo cancellation) [§3, Fig 3-9]
> - **可借鉴**: (1) "thinking" 机制的核心洞察: 在 listening 状态给 LLM 输入 `<think>` token 但不强制特定输出,用负系数 loss 避免 placeholder 重复导致的分布坍缩 [§2]; (2) 周期同步 time block 设计让 LLM 获得 "时间感",对齐音频和文本模态 [§2]; (3) 用 embedding 而非 codec token 连接组件,避免词表冲突和灾难性遗忘 [§1, §2]
> - **局限**: 仅为概念框架和定性 case study,无定量实验对比 [§3]; 未公开模型架构细节 (encoder/synthesizer 具体选型、LLM backbone、训练超参) [§2]; 未公开 checkpoint [§1]

## 核心问题

SALMONN-omni v1 要回答一个根本性问题: **全双工 Speech LLM 是否必须依赖 speech codec (离散化语音 tokens)?** [§1]

2024 年末的全双工系统均采用 codec injection: Moshi 将 audio codec tokens 注入 LLM 词表并行建模双流 [26]; SyncLLM 以交错方式建模输入/输出流的 discrete tokens [27]; LSLM 用 vq-wav2vec tokens 做单说话者全双工 [25]。这些系统的共同假设是: 语音必须先离散化为 tokens 才能被 LLM 处理和生成 [论文原文]。

本文提出 codec-free 替代方案,将问题分解为四个子挑战 [§2]:
1. 如何支持 streaming 输入输出 → embedding 桥接 (非 text/codec 中间表示)
2. 如何同时处理输入流和输出流 → encoder 处理输入、LLM 处理输出、cross-attention 连接
3. 如何对齐音频和文本的时间 → 周期同步 time block
4. 如何处理 turn-taking/barge-in 等对话动态 → "thinking" 状态转换机制

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

三组件端到端架构 [§2, Fig 2]:

1. **Streaming Speech Encoder**: 从输入语音中提取 auditory embeddings。v1 未公开具体架构,仅描述为 "streaming speech encoder" [§2]。

2. **Large Language Model (LLM)**: 接收 speech embeddings 作为输入,生成 word embeddings 作为输出。关键: LLM 的 output layer (vocabulary projection) 仅用于生成状态 token (`<start_speak>`, `<end_speak>`) 和 answer text; 送往 synthesizer 的是 LLM hidden embeddings,不经过 output layer [§2, Fig 2] [论文原文]。v1 未公开 LLM backbone 的具体选型。

3. **Streaming Speech Synthesizer**: 通过 cross-attention layers 接收 LLM hidden embeddings,生成语音输出 [§2]。v1 未公开 synthesizer 的具体架构。

**周期同步 (time block) 机制** [§2]: 对话被切分为一系列 time blocks。在 block i 中:
- Encoder 从输入语音中提取固定时长 delta_t 秒的 auditory embeddings
- LLM 基于 block 0 到 block i 的所有 auditory embeddings,生成 n 个 word embeddings
- 若模型处于 speaking 状态,word embeddings 被送往 synthesizer 生成 delta_t 秒语音

[agent 解读] 这个设计让模型拥有了 "时间概念"——每个 time block 对应固定的真实世界时长,使 audio 和 text 模态自然对齐。后续 v2 版本 (2505.17060) 将 delta_t 具体化为 80ms、n=1。

### 关键设计选择

**为什么 codec-free?** [§1] 论文将 codec-based 方案 (Moshi/SyncLLM) 和 codec-free 方案对比。Codec-based 系统 "tokenizing speech into discrete tokens and modelling both auditory and textual tokens in a single Transformer model" [§1],而 SALMONN-omni 使用 embeddings 代替 codecs,"relying on embeddings instead of codecs (quantized speech and audio tokens)" [Abstract]。整个模型通过 embeddings 互连,组件间无显式 text [§2] [论文原文]。

[agent 解读] 这是 KB 中 Speech-LLM Integration Taxonomy 记录的 latent-representation-based 路线首次应用于全双工场景。此前该路线的代表 (SALMONN/Qwen-Audio) 都是半双工/turn-based 系统。v1 证明了 embedding 桥接不仅能做理解,还能做生成和全双工交互。

**"Thinking" 机制** [§2, Fig 2]: 这是本文最核心的设计贡献。

模型有两个状态: **speaking** 和 **non-speaking**,通过两个特殊 token 转换:
- `<start_speak>`: LLM 生成此 token → 切换到 speaking 状态
- `<end_speak>`: LLM 生成此 token → 切换到 non-speaking 状态 (完成回答或被打断)

每个 time block 内 LLM 必须解码 n 个 tokens。但两种情况下 ground truth labels 难以确定 [§2]:
1. **Non-speaking 状态**: `<start_speak>` 之前的 tokens 内容不确定 (模型在"想"要不要开始说话)
2. **Speaking 状态,LLM 已完成文本但 synthesizer 仍在播放**: LLM 需要继续生成 tokens 以保持对输入流的感知,但生成什么?

**朴素方案的问题**: 引入特殊 placeholder token 会导致训练数据中同一 token 高频重复,使模型输出分布坍缩 (collapsing),严重降低性能 [§2] [论文原文]。

**Thinking 策略的解决方案**: 引入 `<think>` token,但仅用作 **输入** (不是输出目标)。在上述两种情况下,`<think>` 被送入 LLM 作为输入,但输出 labels 不被强制为任何特定 token——唯一的约束是输出不能包含 `<start_speak>` 或 `<end_speak>` [§2]。

[agent 解读] 这个设计的精妙之处在于: `<think>` 作为输入告知 LLM "你现在处于思考状态",但不限制 LLM 思考什么。输出 labels 设为当前状态对应的目标 token (`<start_speak>` for non-speaking, `<end_speak>` for speaking),并施加 **负系数 loss** (lambda_think < 0)。负系数的效果是让模型在这些位置 **远离** 状态转换 token,即"继续当前状态"的方向 [§2]。这避免了 placeholder 重复的问题,同时让模型自然学会何时切换状态。

**损失函数** [§2, Eqn. 1]:
```
L = lambda_text * L_text + lambda_speech * L_speech + lambda_think * L_think
```
其中 lambda_think < 0。L_text 和 L_speech 分别是 LLM 生成文本和 synthesizer 生成语音的损失。

[agent 解读] 论文将 thinking 机制类比为人类对话中的内部思考过程: "internal thoughts may differ from spoken words and are not explicitly conveyed" [§2]。这个类比虽然直觉上合理,但技术上更准确的理解是: 这是一种通过负 loss 实现的 "soft state maintenance" 策略——让模型在不需要说话时产出 low-entropy、不触发状态转换的 tokens,同时保持对输入流的编码和关注。

### 训练策略

v1 仅披露了训练数据来源,未公开训练细节 [§3]:
- ASR 数据: 60k 小时 LibriHeavy [29] + 10k 小时 GigaSpeech [30]
- 合成数据 (speech enhancement, turn-taking, barge-in): 基于上述 ASR 数据源生成

[agent 解读] 合成数据的具体构造方法未披露。从 case study (Fig 6-9) 来看,barge-in 训练可能使用了特定关键词 ("Yes") 和上下文相关的合成对话。后续 v2 版本详细披露了三阶段训练流程和合成数据的构造方法。

## 实验

本文未提供定量实验结果,仅通过 7 个 case study 定性展示能力 [§3]:

| Case Study | 任务 | 展示内容 | 出处 |
| --- | --- | --- | --- |
| Fig 3 | Streaming ASR | 流式将语音转写为文本 (不使用 synthesizer) | [§3, Fig 3] |
| Fig 4 | Speech Enhancement | 听嘈杂语音,重新以清晰语音"复述" (denoising + dereverberation) | [§3, Fig 4] |
| Fig 5 | Spoken QA + Turn-taking | 用户提问 → 模型 THINK → 用户沉默 → 模型回答 | [§3, Fig 5] |
| Fig 6 | Context-independent Barge-in | 用户说 "Yes" → 停止生成; 听到其他词或无声 → 继续 | [§3, Fig 6] |
| Fig 7 | Barge-in + Echo Cancellation | 同 Fig 6,但模型同时听到自己的语音 echo,不受 echo 干扰 | [§3, Fig 7] |
| Fig 8 | Context-dependent Barge-in (成功打断) | 用户说出列表中匹配的答案 "Dog" → 模型停止并跳到下一项 | [§3, Fig 8] |
| Fig 9 | Context-dependent Barge-in (忽略无关) | 用户说不相关的 "Bed" → 模型忽略并继续 | [§3, Fig 9] |

**Case study 的设计特点** [agent 解读]: 
- Fig 6-9 的 barge-in 实验使用了受控的 TTS 场景 (给定文本让模型合成语音),而非开放式对话,这使得评估变得可控但也限制了推广性。
- Echo cancellation (Fig 7) 展示了 codec-free 方案的一个潜在优势: 模型通过 streaming encoder 同时处理用户语音和自身 echo,cross-attention 机制让 LLM 能区分两者。
- Context-dependent barge-in (Fig 8-9) 展示了语义理解驱动的打断判断: 模型需要理解用户说的内容是否与当前上下文相关,而非简单的 VAD。

## 局限性

1. **无定量评估**: 全文仅有 case study,没有任何定量指标 (WER/accuracy/F1/MOS 等),无法与 Moshi/SyncLLM 等系统做公平对比 [§3]。[agent 解读] 作为 technical report,这是可接受的,但读者无法判断系统的可靠性和泛化性。

2. **架构细节未公开**: streaming encoder 和 streaming synthesizer 的具体架构、LLM backbone 选型、time block 参数 (delta_t, n)、训练超参、模型规模等均未披露 [§2]。[agent 解读] 后续 v2 版本 (2505.17060) 公开了所有细节: Mamba encoder + Llama-3-8B + CosyVoice2-0.5B, delta_t=80ms, n=1。

3. **Cross-attention 连接方式未详述**: 论文提到 streaming speech synthesizer 通过 cross-attention layers 与 LLM 连接 [§2],但未解释 cross-attention 的具体设计 (query/key/value 分别来自哪里,几层 cross-attention,是否预训练)。

4. **仅英语**: 训练数据 (LibriHeavy + GigaSpeech) 均为英语 [§3]。

5. **合成数据构造未公开**: turn-taking 和 barge-in 的合成数据如何从 ASR 数据源生成,未详细说明 [§3]。

6. **Echo cancellation 机制未分析**: Fig 7 展示了 echo cancellation 能力,但未分析模型如何学会区分自身 echo 和用户语音 [§3, Fig 7]。[agent 解读] v2 版本揭示了关键: assistant stream 回听 (dual-channel input) 是 echo cancellation 的核心,v1 是否已采用此设计不明。

7. **Checkpoint 未发布**: "A full technical report along with model checkpoints will be released soon" [§1],实际上 v2 完整版于 2025 年 5 月发布。

## 点评

SALMONN-omni v1 的价值在于 **概念先行**: 它是第一个明确提出 "codec-free full-duplex" 技术路线的论文,在 Moshi 发布仅两个月后即提出了根本不同的架构方案。这对全双工 Speech LLM 领域有方向性意义——证明 codec injection 不是唯一路线,embedding-based (latent-representation) 路线同样可行。

**"Thinking" 机制是最有价值的贡献**。v1 对 thinking 机制的描述虽然简短但已包含所有核心元素: (1) `<think>` 仅作为输入不作为输出目标; (2) 负系数 loss 避免分布坍缩; (3) 状态转换由 `<start_speak>`/`<end_speak>` 显式控制。这一机制后来被证明非常有效 (v2 版本中,显式 thinking 全面胜过隐式 thinking [Table 2, arXiv 2505.17060]),并与多个独立团队的设计 (Raon-SpeechChat 的 SIL/BOW/BC, Covo-Audio 的 THINK/SHIFT/BREAK, ELLSA 的 THINK/SHIFT/BREAK) 形成趋同演化。

**与已有知识库中系统的对比定位**:
- vs Moshi: Moshi 使用 RQ-Transformer 将 codec tokens 注入 LLM 词表 (audio-token-based),SALMONN-omni 使用 embeddings 桥接 (latent-representation-based)。这是全双工系统的两条根本不同的技术路线。
- vs LSLM: LSLM 做"边说边听"但是单说话者 AR TTS + streaming SSL 监听,不是完整的对话系统。SALMONN-omni 是完整的理解+生成+全双工系统。
- vs SyncLLM: SyncLLM 使用 interleaved discrete tokens,仍属 codec-based。

**作为 technical report 的局限**: 缺少定量评估是最大的不足。Case study 虽然展示了多种能力,但 (1) 无法判断成功率和稳定性; (2) 无法与同期系统对比; (3) 某些 case 设计过于简单 (如 context-independent barge-in 仅用 "Yes" 作为打断词)。v2 版本通过大量消融实验和与 Moshi/Freeze-Omni 的定量对比弥补了这一缺陷。

**与 v2 完整版 ([[论文笔记/SALMONN-omni|SALMONN-omni]]) 的关系**: v1 是概念提出,v2 是完整实现。v2 公开了所有架构细节 (Mamba encoder, Llama-3-8B, CosyVoice2)、三阶段训练、DPO 优化、assistant stream 回听等关键改进,并提供了全面的定量评估。建议将 v1 视为 "idea paper",v2 视为 "system paper"。

## 可复用的 idea

1. **Embedding 桥接替代 codec injection**: 在任何需要将 LLM 与外部模态模块 (encoder/decoder) 连接的场景中,hidden embeddings 是比 discrete tokens 更轻量的接口方案——不需要修改 LLM 词表,不需要大规模数据防止灾难性遗忘,且保留了连续表征的信息量 [§2]。

2. **Negative loss for thinking tokens**: 当模型需要在某些位置"不做特定事情"(如不触发状态转换) 时,对目标 token 施加负系数 loss 比强制输出 placeholder token 更优雅——避免了重复 token 导致的分布坍缩,同时让模型保持正常的自回归生成能力 [§2, Eqn. 1]。

3. **周期同步 time block 设计**: 将连续的音频流和离散的 text token 生成对齐到固定时长的 time blocks,是一种简洁的多模态时间对齐方案。每个 block 处理固定时长音频 + 生成固定数量 tokens,使模型自然获得 "时间感" [§2]。

4. **Context-dependent barge-in 作为评估**: Fig 8-9 的设计思路——让模型在 TTS 过程中判断用户打断是否与当前内容相关——可作为全双工系统语义理解能力的评估范式 [§3, Fig 8-9]。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节围绕四个子挑战展开因果解释; thinking 机制的 WHY (避免分布坍缩) 和 HOW (负系数 loss + 仅作输入) 均有充分阐述 |
> | 可信赖 | pass | 所有 claim 标注 [§/Fig]; 明确标注本文无定量评估; 数据来源有出处 |
> | 可区分 | pass | [论文原文] vs [agent 解读] 标注覆盖率 >80%; 推断性分析均有限定词; 与 v2 版本的关系明确标注 |
> | 可定位 | pass | KB 背景有谱系定位 (codec injection vs embedding-based 路线对比); 速查卡片 5 字段均有实质内容; 与 v2 版本的区别清晰 |
> | 不污染 | pass | 未修改概念页; 无 overclaim; 准确标注为 technical report 而非完整系统论文 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/SALMONN-omni-v1-review.yml`
