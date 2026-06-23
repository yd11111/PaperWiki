---
type: paper
tier: deep
title: "Adaptive Turn-Taking for Real-time Multi-Party Voice Agents"
arxiv_id: "2606.13544"
source: "Sources/ModeratorLM.pdf"
authors: [Soumyajit Mitra, Prabhat Pandey, Abhinav Jain, Shanmukha Sahith, K V Vijay Girish]
year: 2026
venue: "Interspeech 2026"
tags: [speech-LM, multi-party, turn-taking, role-playing, chain-of-thought, streaming, voice-agent, multi-speaker]
concepts: ["[[Turn-takinginSpokenDialogue]]", "[[Full-duplexSpokenDialogue]]", "[[SpeechLanguageModel]]", "[[StreamingSpokenDialogue]]"]
models: ["[[论文笔记/Moshi|Moshi]]"]
tasks: []
datasets: ["RolePlayConv", "NOTSOFAR-1", "AMI", "Fisher"]
kb_context_sources: 4
status: draft
created: 2026-06-23
updated: 2026-06-23
---

## KB 背景

> [!info] KB 背景 (基于 4 个实体页: [[Turn-takinginSpokenDialogue]] [待确认], [[Full-duplexSpokenDialogue]] [待确认], [[SpeechLanguageModel]]✓, [[StreamingSpokenDialogue]] [待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Turn-takinginSpokenDialogue]], [[Full-duplexSpokenDialogue]], [[SpeechLanguageModel]], [[StreamingSpokenDialogue]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: ModeratorLM 属于 multi-party voice agent 的 turn-taking 建模范畴,与 KB 中已有的全双工/打断处理系统不同,它专注于一个被忽视的维度: **multi-party 场景下的 role-conditioned turn-taking**。现有系统(Moshi, Freeze-Omni, BayLing-Duplex 等)主要针对 dyadic (两人) 对话,turn-taking 决策基于声学/语义 cue;ModeratorLM 首次将 assistant 的"角色"(如主持人、观察者)作为 turn-taking 决策的条件信号。

**已有认知**: KB 的 [[Turn-takinginSpokenDialogue]] 页面详细记录了 turn-taking 的演进: VAD → 级联系统 (Duplex Conversation) → 端到端系统 (Moshi 多流, Freeze-Omni chunk-level state prediction, Raon-SpeechChat SIL/BOW/BC, BayLing-Duplex 4 状态 token)。这些系统一律假设对话为两方(user-assistant),且 assistant 行为固定。ModeratorLM 打破了两个假设: (1) multi-party (3-6 speakers); (2) assistant 行为由角色描述动态配置。

**创新判断**: ModeratorLM 开辟了 multi-party + role-conditioning 的新方向。KB 中最接近的系统是 PaChat (EMNLP 2025, persona-based multi-party),但 PaChat 关注个性化而非 role-conditioned turn-taking。ModeratorLM 还引入了 chain-of-thought reasoning (ModeratorLM-Think) 用于 turn-taking 决策,这在 spoken dialogue 中是新颖的 — KB 中仅 BayLing-Duplex 的 DPO 间接优化了 turn-taking timing,但没有显式推理链。

> [!summary] 速查
> - **一句话**: 首个 role-conditioned multi-party voice agent,通过角色描述动态调整 turn-taking 行为,chain-of-thought 推理变体进一步提升角色忠实度
> - **路线**: Multi-speaker audio (downmix) → Speech Encoder (chunk-wise) → Linear Projection → Qwen3-4B LLM (+ chunk transcript) → Turn-taking control token + Response 或 No-turn
> - **指标**: NOTSOFAR-1: P 0.81 / R 0.74 / F1 0.76 / FP 0.01 (ModeratorLM-Think) vs Moshi P 0.14 / R 0.10; RolePlayConv: P 0.79 / R 0.82 / F1 0.79 / FP 0.03 (ModeratorLM-Think) [Table 2]
> - **可借鉴**: (1) Role-conditioning 通过 system prompt 注入角色描述,零成本扩展到不同 turn-taking 策略; (2) 动态 chunk 长度训练 (0.5-3s) 防止模型过拟合 chunk-size cue; (3) CoT 推理显著降低 reactive miss rate (从 0.14 到 0.03)
> - **局限**: 只输出文本响应(无语音合成);合成数据训练/评估(Zonos TTS);multi-channel downmix 到 single-channel 损失了空间信息;未测试真实多人实时交互场景

## 核心问题

ModeratorLM 要解决的核心问题是: **在 multi-party 对话中,voice agent 如何根据被分配的角色决定何时、是否发言?**

这个问题在两个层面上具有挑战性 [§1]:

1. **Multi-party 对话的复杂性**: 不同于两人对话中 turn-taking 主要由停顿/沉默检测决定,多人对话涉及重叠语音 (overlapping speech)、动态发言权竞争 (floor competition)、参与者之间的协商式轮替 (negotiated turn allocation)。Agent 必须持续判断"不仅是何时说,还包括是否应该介入"。

2. **角色对 turn-taking 的影响**: 用户对 agent 的期望因场景而异 — 有时需要主动引导讨论的主持人,有时只需安静记录的观察者 [§1]。Prior work 在 role-playing language agents (RLPA) 上主要关注文本对话中的语言风格 [14]、人物特质 [15] 和决策模式 [16],**角色对实时语音 turn-taking 的影响被忽视了**。

**为什么现有方法不够**: Moshi 等全双工系统设计用于 dyadic 对话,在多人场景中 recall 极低且 FP 率高 [Table 2]。MP-Baseline (同数据但无角色条件) 虽减少了误中断,但缺乏角色对齐的精确性。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ModeratorLM 由两个核心组件组成 [§2.1]:

1. **Speech Encoder**: 处理每个音频 chunk,输出 chunk-level embeddings。Multi-channel 音频先 downmix 为 single-channel 再编码。使用 in-house encoder,支持 variable lookahead 的 block-wise attention [28, 29]。
2. **Backbone LLM**: Qwen3-4B-Instruct-2507 (ModeratorLM) 或 Qwen3-4B-Thinking-2507 (ModeratorLM-Think)。

Speech embeddings 通过 **trainable linear projection** 映射到 LLM embedding space [§2.1],然后以 **chunk-wise streaming** 方式依次追加到 LLM context。

**与 Freeze-Omni 的关键区别**: 不依赖外部 VAD 模块决定 turn boundaries [§2.1],turn-taking 决策完全由 Speech LLM 自身做出 [论文原文]: "our system delegates turn-taking decisions entirely to the speech LLM itself"。

### 输入/输出设计

对每个输入 audio chunk,LLM 同时接收 [§2.1]:
- Speech embeddings (经 projection)
- 对应的文本转写 (含 speaker annotations)

LLM 产出两种可能的输出 [§2.1]:

| 输出类型 | 格式 | 含义 |
|---------|------|------|
| **Turn-taking + Response** | control token → textual response | 助手取得发言权并回应 |
| **No-turn** | 空序列 | 当前 chunk 不采取任何行动 |

**注意**: 本文关注 turn-taking 建模,助手的响应以文本形式生成,而非 speech codes [§2.1]。[agent 解读] 这意味着实际部署需外接 streaming TTS (如 Freeze-Omni [1]) 或在 LLM 内直接生成 speech tokens (如 Mini-Omni [2])。

### ModeratorLM-Think: Chain-of-Thought 推理

ModeratorLM-Think 是推理增强变体 [§2.1]: 在每个潜在 turn-taking 点,LLM 先执行 **chain-of-thought reasoning**,分析对话上下文和分配的角色,再做出 turn-taking 决策。

[Fig 1] 展示了典型的输入-输出序列:
- **Chunk 1**: 无推理,无 turn-taking
- **Chunk 2**: 生成推理 trace (分析对话状态和角色义务),但不 take turn
- **Chunk 3**: 生成推理 trace → 判断应介入 → emit turn-taking control token → 生成文本回复

推理 trace 的生成受 `<think>` control token 触发,之后切换到 sampling mode (Temperature=0.7, TopP=0.8, TopK=20) [§3.2]。

### RolePlayConv 数据集构建

由于缺乏支持 spoken multi-party + role-conditioned assistant 的数据集 [§2.2],作者构建了 RolePlayConv,一个大规模合成多人对话数据集。

**Multi-stage pipeline** [§2.2]:

**Step 1 — 角色库**: 策展 **125 个详细角色描述** (e.g., "a 42-year-old Indian CEO with a confident and assertive communication style who enjoys strategic planning and mentoring"),包含对话风格、语气等属性。

**Step 2 — 对话生成**: 使用 **Amazon Nova Pro** [25] 生成多人对话:
- 3-6 speakers per conversation
- Topic/subtopic 与角色属性一致
- 每个 turn 限制在 15 词以内,模拟口语简洁性

**Step 3 — Reasoning traces 增强**: 用 LLM 为所有 assistant turns 和部分 non-assistant turns 生成推理 traces,描述决策过程: stance selection, response planning, turn-taking considerations [§2.2]。Table 1 给出了具体示例 — 一个"21 岁文学系学生"角色在对话中的内部推理: 先观察组内动态,等群体观点稳定后再加入一句简洁的观点。

**Step 4 — 语音合成**: 使用 **Zonos-v0.1 TTS model** [26] 逐 turn 合成语音:
- 每个 speaker 分配 gender, age group, accent → 映射到 reference voice pool
- 训练/评估使用不同 speaker pool (防止 speaker leakage)
- Turn 间插入从真实对话数据估计的沉默间隔

**规模**: ~75K 训练对话,每段约 2 分钟。

### 三阶段训练

训练分三个阶段 [§3.1]:

**Stage 1 — Speech-LLM Alignment**: 在 ASR 任务上对齐 speech embeddings 与 LLM input space。使用 ~90K 小时公开数据 (VoxPopuli, MLS, Common Voice, People's Speech)。**仅更新 projection layer**,其他参数冻结。

**Stage 2 — Conversation Pretraining**: 在公开多人对话数据集 (AMI, Fisher) 上训练。因公开数据无 "assistant" 概念,通过 **speaker rotation** 模拟: N 个参与者中,除发起者外每个人轮流扮演 assistant,产生 N-1 个训练实例 [§3.1]。

**Stage 3 — Role-Conditioning Training**: 在 RolePlayConv 上 fine-tune,assistant 角色通过 **system prompt** 指定。

后两阶段使用 **LoRA** (低秩适配) fine-tune LLM 参数 [§3.1],speech encoder 全程冻结。总可训练参数: **13.4M**。优化器: Adam,两阶段学习率调度,峰值 1×10⁻⁵。

### Dynamic Chunking

训练时采用 **dynamically sized chunks** (0.5s-3s 随机采样) [§2.1, §3.1],提升对不同 chunking 行为的鲁棒性。同时确保一定比例的 chunks 在 speaker boundary 处终止 (自然 turn-taking 点) [§3.1]。

[agent 解读] 这个设计选择非常重要: ablation [§4.2] 证明,如果使用固定 chunk size,模型会过拟合 chunk-size cue 而非 conversational context。这与 BayLing-Duplex 的 fixed block size (N=10, 0.8s) 形成对比 — ModeratorLM 的方法更泛化,但代价是训练更复杂。

## 实验分析

### 评估设计

**数据集** [§3.2]:

| 数据集 | 类型 | 说话者数 | 时长 | 角色来源 |
|--------|------|---------|------|---------|
| NOTSOFAR-1 (NSF-1) | 真实会议录音 | ~4 speakers | ~6 min/session | 无标注,使用 LLM + 人工混合方法选定 assistant |
| RolePlayConv (eval) | 合成 | 3-6 speakers | ~2 min/conv | 零样本角色 + 不同 LLM (QwQ-32B) 生成 |

**推理配置**: 动态 chunk (0.5-3s), segmentation 在 speaker boundaries,10 次评估取平均 (因动态 chunking 引入不确定性)。Teacher-forcing ground-truth context + greedy decoding (default);遇 `<think>` token 切换 sampling [§3.2]。

**Metrics** [§3.2]:
- 标准二分类: Precision (P), Recall (R), F1, Macro-Accuracy (A), False Positive Rate (FP)
- **Reactive Miss Rate (RM)**: 当说话者直接 address assistant 时,错失的 turn-taking 比例
- **LLM-as-a-Judge** (Claude-Sonnet-3.5): 评估 turn-taking appropriateness (0-1 scale) 和 response role fidelity (1-10 scale)。100 instance 人工验证: Spearman ρ=0.87

### 主实验结果 [Table 2]

| Model | NOTSOFAR-1 P/R/F1/A/FP/RM | RolePlayConv P/R/F1/A/FP/RM |
|-------|---------------------------|------------------------------|
| Moshi | 0.14/0.10/0.11/0.21/**0.66**/— | 0.15/0.34/0.21/0.50/**0.47**/— |
| MP-Baseline | 0.58/0.33/0.38/0.69/0.05/— | 0.40/0.48/0.42/0.67/0.14/— |
| ModeratorLM | **0.77**/0.51/0.57/0.77/**0.01**/0.08 | **0.71**/0.57/0.61/0.76/0.05/0.14 |
| ModeratorLM-Think | **0.81**/**0.74**/**0.76**/**0.86**/**0.01**/**0.02** | **0.79**/**0.82**/**0.79**/**0.91**/**0.03**/**0.03** |

**关键发现**:

1. **Moshi 在多人场景失效** [§4.1]: 训练于 dyadic 对话的 Moshi recall 极低 (0.10 on NSF-1),FP 率极高 (0.66) — 它几乎在错误的时间回应,在正确的时间沉默。[agent 解读] 这佐证了多人对话 turn-taking 与两人对话是本质不同的问题。

2. **Role conditioning 大幅提升精度** [§4.1]: MP-Baseline vs ModeratorLM 是 controlled comparison (同数据,无 vs 有角色条件),Precision 从 0.58→0.77 (NSF-1), 0.40→0.71 (RolePlayConv)。FP 从 0.05→0.01。Role conditioning 使模型学会"何时不说",这是多人对话的关键能力。

3. **CoT 推理进一步提升 Recall 而不牺牲 Precision** [§4.1]: ModeratorLM-Think 在 NSF-1 上 Recall 从 0.51→0.74 (+45%),同时 FP 保持 0.01。Reactive Miss Rate 从 0.08→0.02。[agent 解读] 推理 trace 帮助模型区分"应该保守沉默"和"有义务响应" — 后者在角色要求 assistant 被直接询问时尤为重要。

4. **NSF-1 上的挑战** [§4.1]: 真实会议含大量 backchannel、interruption 和 overlapping speech,没有人自然遵循 assistant 式 turn-taking,使得 recall 普遍低于合成数据。ModeratorLM-Think 的 0.74 recall 表明 CoT 推理部分弥补了这一 gap。

### LLM-as-a-Judge 评估 [Table 3]

| Model | Turn-Taking (0-1) | Response (0-10) |
|-------|-------------------|-----------------|
| MP-Baseline | 0.58 | 4.6 |
| ModeratorLM | 0.68 | 6.9 |
| ModeratorLM-Think | **0.72** | **7.4** |

ModeratorLM-Think 在主观评估中同样最优,表明 CoT 推理不仅改善时机,还提升了响应内容的角色一致性 [§4.1]。[论文原文]: "the explicit reasoning stage serves as a foundational alignment layer. By first analyzing conversational context and role-specific obligations, the model ensures the decision of when to speak is cognitively grounded."

### 定性分析: 同一对话,不同角色 [Table 4]

Table 4 展示了 ModeratorLM-Think 在相同对话上下文中,面对两个不同角色时的推理 trace 差异:

- **Role A** (authoritative news anchor): 推理判断"讨论需要结构化引导",决定发言设定参数
- **Role B** (composed, non-intrusive anchor): 推理判断"对话需要先积累动量",决定保持沉默

[agent 解读] 这是本文最有说服力的定性证据 — 证明 CoT 不是空洞的推理链,而是真正将角色语义映射到了 turn-taking 行为。

### Ablation Studies [Table 5]

#### Chunking 策略

| Setup | ModeratorLM P/R/A | ModeratorLM-Think P/R/A |
|-------|-------------------|--------------------------|
| Default (dynamic) | 0.71/0.57/0.76 | 0.79/0.82/0.91 |
| Fixed (2s) | 0.88/0.78/0.88 | 0.82/0.82/0.91 |
| Turn-Fixed | 0.84/0.60/0.80 | 0.75/0.81/0.91 |

**关键洞察**:

1. **ModeratorLM 依赖 chunk-size cue** [§4.2]: Fixed chunking 下 precision 显著提高 (0.71→0.88),因为 turn-taking chunks 通常较短。这暗示无 CoT 的模型部分依赖 chunk 长度作为 turn-taking 的代理特征。

2. **ModeratorLM-Think 对 chunking 策略不敏感** [§4.2]: 三种策略下准确率稳定在 0.91。[论文原文]: "ModeratorLM-Think is notably less sensitive to chunking strategy, instead relying on explicit reasoning traces." [agent 解读] CoT 推理使模型基于语义内容决策而非统计捷径。

3. **Fixed chunking 的实际局限** [§4.2]: 真实推理时无法获得 ground-truth utterance boundary,固定大小 chunk 可能截断/延迟 mid-chunk 的 turn-taking cue,导致误时干预。[agent 解读] 这与 BayLing-Duplex 的 fixed block size 问题一致 — 固定时间窗口会引入最小延迟。

#### Transcription 依赖

| Setup | ModeratorLM P/R/A | ModeratorLM-Think P/R/A |
|-------|-------------------|--------------------------|
| Default (GT text) | 0.71/0.57/0.76 | 0.79/0.82/0.91 |
| No Transcription | 0.42/0.14/0.57 | 0.39/0.42/0.57 |
| ASR Hypotheses | 0.68/0.56/0.76 | 0.75/0.80/0.90 |
| GT Thoughts | —/—/— | 0.95/0.95/0.97 |

**关键发现**:

1. **模型严重依赖文本信息** [§4.2]: 移除转写后性能骤降 (Recall 0.82→0.42 for Think variant)。[agent 解读] 这意味着 speech embeddings 单独无法驱动有效的 role-conditioned turn-taking — 文本提供了角色语义理解的关键输入。

2. **对 ASR 错误鲁棒** [§4.2]: 使用 Kyutai-STT-2.6B streaming ASR (WER 6.7%) 时性能仅轻微下降 (Accuracy 0.91→0.90)。这对实际部署很有利。

3. **GT Thoughts 的天花板** [§4.2]: 当提供 ground-truth 推理 trace 时,ModeratorLM-Think 达到近乎完美的 0.95/0.95/0.97。[agent 解读] 这证明 turn-taking 决策本身并不困难 — 难点在于生成高质量的推理 trace。进一步优化 CoT 质量(如通过 RLHF 或更好的推理监督)可能是未来提升的方向。

## 与已有工作的对比

### 与 Moshi 的差异

| 维度 | Moshi | ModeratorLM |
|------|-------|-------------|
| 对话类型 | Dyadic (两人) | Multi-party (3-6人) |
| Turn-taking 条件 | 无 (隐式从数据学) | 角色描述 (system prompt) |
| 架构 | RQ-Transformer, 并行双流 | Speech encoder + LLM, 流式追加 |
| 输出 | 语音 (audio tokens) | 文本 (需外接 TTS) |
| 训练数据规模 | 百万小时级 | ~75K 对话 (~2500 小时) |

### 与 Freeze-Omni 的差异

ModeratorLM 与 Freeze-Omni 共享"chunk-level turn-taking"的理念,但关键不同:
- Freeze-Omni 使用独立分类头预测 State 0/1/2 [论文原文: §2.1]
- ModeratorLM 通过 LLM 自身的 generation 做决策 (control token / empty sequence)
- Freeze-Omni 无角色条件

### 与 BayLing-Duplex/Raon-SpeechChat 的差异

BayLing-Duplex 和 Raon-SpeechChat 同属全双工 turn-taking 系统,但:
- 均针对 dyadic 对话 (user-assistant)
- 通过特殊状态 token (4 token / SIL+BOW+BC) 编码 turn-taking 状态
- ModeratorLM 的创新在于角色条件化和 multi-party 支持
- ModeratorLM-Think 的 CoT 推理机制在已有系统中无先例

## 评价与局限

### 创新之处

1. **首个 role-conditioned multi-party voice agent**: 将角色描述引入 turn-taking 决策,开辟了 multi-party 场景的新维度 [§1]
2. **CoT 推理用于 turn-taking**: ModeratorLM-Think 证明显式推理可显著提升 turn-taking 的角色忠实度,且降低对 chunk-size 等统计捷径的依赖 [§4.2]
3. **RolePlayConv 数据集**: 125 角色 × 75K 对话的大规模 role-conditioned spoken multi-party 数据,填补数据空白 [§2.2]
4. **三阶段训练 + LoRA**: 仅 13.4M 可训练参数,资源友好

### 局限性

1. **只输出文本**: 未集成语音生成,实际部署需外接 TTS,引入额外延迟和系统复杂性
2. **合成数据依赖**: RolePlayConv 用 Zonos TTS 合成,与真实多人对话的声学特性 (reverberation, overlapping speech, varied recording conditions) 有差距。NSF-1 上的 gap (recall 0.74 vs 0.82 on RolePlayConv) 部分反映了这一问题
3. **Single-channel downmix**: Multi-channel 音频先混合为 single-channel 再编码 [§2.1],丢失了空间信息 (说话者方向、位置),可能影响多人场景下的 speaker attribution
4. **文本转写依赖**: 无转写时性能骤降 [Table 5],意味着系统强依赖 ASR,不适用于 ASR 不可用或 WER 过高的场景
5. **角色评估的主观性**: NSF-1 的 "assistant" 角色是后验标注(LLM ranking + 人工),而非对话中自然存在的;RolePlayConv 的角色和对话均为合成
6. **未验证实时交互**: 使用 teacher-forced context 评估,未测试真实场景中模型输出影响后续对话的闭环效应

## 可迁移经验

1. **Role-conditioning via system prompt**: 通过 system prompt 注入角色描述来调控 agent 行为,是一种低成本的行为定制方案。可迁移到其他 interactive agents (如客服、教育、医疗)
2. **Dynamic chunk 训练**: 训练时随机化 chunk 长度 (0.5-3s) 有效防止模型过拟合 chunk-size cue,对所有 chunk-based streaming 系统具有参考价值
3. **CoT 推理抗 shortcut learning**: ModeratorLM-Think 对 chunk 策略不敏感的结果表明,在模型倾向于学习统计捷径的场景中,引入显式推理链可以迫使模型关注语义内容
4. **Speaker rotation 数据增强**: 在无 assistant 标注的多人对话数据中,通过轮流指定 assistant 创建训练实例 (N-1 个/conversation),是一种有效的数据利用策略

> [!review] 审阅 callout
> 本笔记为 draft 状态,由 AI agent 生成,尚未经过人工确认或独立审阅。

## 参考链接

- arXiv: https://arxiv.org/abs/2606.13544
- 原始 PDF: [[Sources/ModeratorLM.pdf]]
