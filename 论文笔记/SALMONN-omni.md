---
type: paper
tier: deep
title: "SALMONN-omni: A Standalone Speech LLM without Codec Injection for Full-duplex Conversation"
arxiv_id: "2505.17060"
source: "Sources/SALMONN-omni.pdf"
authors: [Wenyi Yu, Siyin Wang, Xiaoyu Yang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Guangzhi Sun, Lu Lu, Yuxuan Wang, Chao Zhang]
year: 2025
venue: "arXiv"
tags: [full-duplex, speech-LM, codec-free, standalone, thinking-strategy, streaming, turn-taking, barge-in, backchannel, DPO, Mamba, echo-cancellation]
concepts: ["[[概念库/Full-duplexSpokenDialogue|Full-duplex Spoken Dialogue]]", "[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/StreamingSpokenDialogue|Streaming Spoken Dialogue]]", "[[概念库/Turn-takinginSpokenDialogue|Turn-taking in Spoken Dialogue]]", "[[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation for Speech LLM]]"]
models: ["[[论文笔记/Moshi|Moshi]]"]
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[概念库/SpeechLanguageModel|SpeechLanguageModel]], [[概念库/SpeechTokenizer|SpeechTokenizer]], [[概念库/ConditionalFlowMatching|ConditionalFlowMatching]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[概念库/SpeechLanguageModel|SpeechLanguageModel]]✓, [[概念库/SpeechTokenizer|SpeechTokenizer]]✓(间接相关), [[概念库/ConditionalFlowMatching|ConditionalFlowMatching]]✓(间接,synthesizer 侧) | 过滤: [[概念库/Full-duplexSpokenDialogue|Full-duplexSpokenDialogue]](pending-review), [[概念库/StreamingSpokenDialogue|StreamingSpokenDialogue]](pending-review), [[概念库/ModalityAdaptationforSpeechLLM|ModalityAdaptationforSpeechLLM]](pending-review), [[概念库/Turn-takinginSpokenDialogue|Turn-takinginSpokenDialogue]](pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: KB 记录了 SpeechLM 从 GSLM→AudioLM→SpeechGPT→Moshi 的演进。SALMONN-omni 与 [[论文笔记/Moshi|Moshi]] 同属全双工 SpeechLM,但两者采用截然不同的技术路线: Moshi 将 audio codec tokens 注入 LLM 词表(codec injection),SALMONN-omni 则完全不在 LLM 词表中使用 codec tokens(codec-free),转而通过 hidden embeddings 桥接 streaming encoder/synthesizer。这是对 KB 中 "audio-token-based" 与 "latent-representation-based" 两条路线的直接对比验证。
>
> **Speech Tokenizer**: KB 中 SpeechTokenizer 概念覆盖了 self-supervised / supervised / acoustic 三类路线。SALMONN-omni 的 Mamba streaming encoder 不走离散 token 路线,而是通过知识蒸馏 (Whisper-large-v3 teacher → Mamba student) 生成连续 embeddings,属于 latent-representation 而非 discrete tokenizer 路线。这使得 SALMONN-omni 不需要处理 codec token 与 text token 的词表冲突问题。
>
> **[基于未确认概念页] Full-duplex Spoken Dialogue**: pending-review 页已收录 SALMONN-omni 早期版本。该页记录了全双工系统的两大阵营: (1) codec injection 路线 (Moshi/SyncLLM/OmniFlatten) 和 (2) 非独立双进程路线 (VITA/Freeze-Omni/MinMo)。SALMONN-omni 开辟了第三条路: 单 LLM + codec-free + 独立全双工,是该概念页中唯一同时满足 "standalone" 和 "codec-free" 两个条件的系统。

> [!summary] 速查
> - **一句话**: 首个不注入 audio codec 的独立全双工 Speech LLM,通过 Mamba streaming encoder + Llama-3-8B (LoRA) + CosyVoice2 streaming synthesizer 的 codec-free 架构 + 显式 "thinking" 状态转换机制,在全双工模式下比此前开源 SOTA 平均高 35.9% [§1, §5.2]
> - **路线**: dual-stream audio (environment + assistant) → Mamba encoder (25Hz embeddings) → interleaved with text embeddings → Llama-3-8B + LoRA → text tokens + `<think>`/`<shift>` → CosyVoice2 synthesizer (从 LLM 第 24 层 embedding 输入) → speech output [§3.1-3.3, Fig 1, Appendix E]
> - **指标**: Predicted turn-taking: Llama Q. 79.3 / Web Q. 49.7 / TriviaQA 63.6 / AlpacaEval 4.01 (S2T, 全面 SOTA) [Table 3]; Turn-taking success rate 92-99.7% [Table 4]; Context-dependent barge-in F1 0.93 (DPO 后) [Table 6]; Oracle turn-taking 与 Kimi-Audio/Qwen2.5-Omni 等半双工 SOTA 竞争 [Table 3]
> - **可借鉴**: (1) 显式 thinking 策略: `<think>` + `<shift>` 两个 token 让 LLM 自然学会状态转换,验证了 "your LLM is secretly a full-duplex predictor" 的假说 [§3.3]; (2) Assistant stream 回听: 让模型在生成前先看到自己上一轮的语音 embedding,既提供 echo 信息又将 turn-taking 成功率从 ~70% 提升到 ~90% [§5.1]; (3) DPO 的 U 型曲线: DPO 训练中模型先变得极端保守 (几乎不被打断),然后恢复并超越 SFT,这一现象对 RL 训练全双工模型有参考价值 [§5.3.2, Fig 4]
> - **局限**: 仅英语 [§1]; 80ms time block 设计引入 320ms 最小输出延迟 [§4.1]; 情感表达不稳定 (偶尔不匹配) [Appendix F]; 仅 LoRA (rank 32) 微调 LLM,容量有限 [§4.1]; 训练数据 ~1.3M 样本,远少于 Kimi-Audio 13M 小时 [§1]

## 核心问题

SALMONN-omni 要解决现有全双工语音 LLM 的两个结构性缺陷 [§1]:

1. **Codec injection 的代价**: Moshi/SyncLLM/OmniFlatten 将 audio codec tokens 注入 LLM 词表,但这需要大规模 speech-text paired data 防止灾难性遗忘,且 speech 模态的性能始终落后于 text 模态 [论文原文]。从 Table 3 可见 Moshi S2T vs S2S 差距 (60.8 vs 54.5 on Llama Q.),证实了 modality gap 的存在 [agent 解读]。

2. **非独立架构的复杂性**: VITA/Freeze-Omni/MinMo 通过连接 encoder/synthesizer 避免了 codec injection,但它们的单个 LLM 实例只能要么听要么说,需要运行两个 LLM 进程才能实现全双工,带来计算/内存开销和上下文割裂 [§1, §2.1]。Freeze-Omni 额外依赖 VAD 模块,在有 echo 时 F1 从 0.68 暴跌到 0.17 [Table 5] [论文原文]。

SALMONN-omni 的核心命题是: 能否用**单个 LLM backbone + 不注入 codec + 不依赖 VAD** 实现独立全双工? 论文通过四个子问题回答这个命题 [§3]:
- 如何支持 streaming 输入输出 (→ Mamba encoder + CosyVoice2 synthesizer + embedding 桥接)
- 如何同时处理环境声和自身语音 (→ 双流交错)
- 如何对齐 audio 和 text 的时间 (→ 80ms time block 周期同步)
- 如何决定何时说何时听 (→ 显式 thinking 策略)

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SALMONN-omni 由三个核心组件组成 [§3, Fig 1]:

1. **Mamba Streaming Speech Encoder**: 32 个 Mamba LM blocks,2048 维隐状态,输出 25Hz 帧率的连续 embedding [§3.1, §4.1]。用 Whisper-large-v3 做 teacher,L1 loss 知识蒸馏 [§3.1]。输入侧先用两层卷积将 100Hz log-Mel 降采样到 50Hz,再两帧拼接成一帧得到 25Hz [§3.1]。

2. **Llama-3-8B-Instruct**: LLM backbone,以 LoRA (rank 32, scaling 1.0) 微调 [§4.1]。通过 MLP connector 接收 encoder embedding,通过 linear transformations 输出 embedding 给 synthesizer [§3.2]。

3. **CosyVoice2-0.5B Streaming Speech Synthesizer**: 基于 CosyVoice2 的 fixed-length interleaved generation 策略 [§3.2]。每 4 个 text tokens 生成 12 个 speech tokens (480ms 语音) [§4.1]。关键改造是用 LLM backbone 第 24 层的 output embeddings 替换原始文本输入 [§3.2, Appendix E]。

**双流交错机制** [§3.3, Fig 2]: 对话被分为 environment stream (所有输入声音: 用户语音、背景噪声、助手 echo) 和 assistant stream (模型生成的响应)。在每个 80ms time block 内,LLM 先处理 environment stream 的语音 embedding,再处理上一轮 assistant stream 的语音 embedding (dual-channel input),最后生成文本 token 或状态 token [论文原文]。

### 关键设计选择

**为什么 codec-free 而不是 codec injection?** [§1, §2.1] Codec injection (Moshi 路线) 将 audio tokens 直接加入 LLM 词表,概念简洁但有三个代价: (1) 需要大规模数据防止灾难性遗忘; (2) speech 模态性能始终不如 text 模态; (3) synthesizer 必须仅凭 codec stream 推断何时输出语音何时沉默,timing 不精确 [论文原文]。SALMONN-omni 通过 hidden embeddings 连接 encoder/synthesizer,保持 LLM 原始词表不变,用 LoRA 适配即可 [§3.1, §3.2]。[agent 解读] 这实质上是 KB 中 "latent-representation-based" 路线在全双工场景的首次成功应用,之前该路线仅在半双工/turn-based 系统中验证过 (如 SALMONN, Qwen-Audio)。

**为什么用显式 thinking 而不是隐式 thinking?** [§3.3, Fig 2, Table 2] 论文对比了两种策略:
- **隐式 thinking**: 每个 time block 预测 `<listen>` 或 `<speak>` token,但这些 token 不回馈到输入序列。LLM 缺乏从输出到输入的完整反馈循环 [论文原文]。
- **显式 thinking**: 使用 `<think>` 和 `<shift>` 两个 token,混入 LLM 输入序列。`<think>` 在 listening 状态下每个 time block 生成,模拟"思考要不要说话"; `<shift>` 标记 listening→speaking 和 speaking→listening 的双向状态转换 [§3.3] [论文原文]。

显式策略在 Stage 1 实验中全面胜出: AlpacaEval S2T 从 3.73 提升到 4.48 [Table 2]。论文将此归因为"训练 LLM 生成包含正常响应和状态转换 token 的完整序列,与其自回归本性天然对齐" [§3.3] [论文原文]。口号是 "your LLM is secretly a full-duplex predictor" [§3.3]。

[agent 解读] 这与 Raon-SpeechChat 的 SIL/BOW/BC 三状态建模异曲同工,但 SALMONN-omni 更简洁 (仅 2 个特殊 token vs 3 个),且不区分 backchannel 和 silence,而是让 DPO 来学习这种语义区分。

**为什么在 listening 状态用 `<think>` 而不是做 streaming ASR?** [Appendix C, Table 8] 论文探索了多种 thinking 内容: ASR 转写 (Implicit-ASR, Explicit-ASR)、inner thoughts、negative sampling (Explicit-NS)。结果是最简单的 Explicit (仅 `<think>` 一个 token) 最好: test-clean WER 2.40 vs Explicit-ASR 5.09 [Table 8]。论文解释: 增加 thinking 内容的多样性使 speaking/listening 两个状态的操作模式变得难以区分,模型需要同时学状态转换和两种模式的区分,负担过重。简单 `<think>` 让 listening 状态操作极简 (只输出一个 token),模型可以专注于学习状态转换和 speaking 状态的内容生成 [Appendix C] [论文原文]。

**为什么 assistant stream 回听如此重要?** [§5.1] Stage 1 (无 assistant stream) 的 turn-taking 成功率仅 ~70% (AlpacaEval),Stage 2 引入 assistant stream 后提升到 ~90%。论文将此归因为: (1) 模型需要知道自己在说什么才能判断何时停止; (2) 回听提供 echo 信息,使模型能处理 echo cancellation [论文原文]。[agent 解读] Freeze-Omni 不做回听,其 echo 场景 F1 从 0.68 暴跌到 0.17 (Table 5),这是不回听的直接后果。

**为什么选 LLM 第 24 层 embedding 而不是最后一层?** [Appendix E, Table 12] 消融实验显示: 前 8 层 embedding S2S 表现差 (大部分 LLM 层未参与语音生成); 最后一层 S2T 性能最差 (紧耦合语音-文本表征可能降低 LLM 核心语言理解能力); 第 24 层在 S2T 和 S2S 之间取得最佳平衡 [论文原文]。[agent 解读] 这与 "LLM 中间层保留更多通用表征,浅层/深层分别偏向输入/输出" 的一般观察一致。

**80ms time block 的设计权衡** [§4.1]: 每个 time block 处理 80ms 输入音频并生成 1 个 text token。4 个 tokens 攒够后触发 synthesizer 生成 12 个 speech tokens (480ms)。因此首个 speech token 的最小延迟是 4 × 80ms = 320ms [§4.1] [论文原文]。[agent 解读] 相比 Moshi 的 160ms 理论延迟,SALMONN-omni 的 320ms 是 codec-free 架构的代价——它需要先产出足够的 text tokens 才能驱动外部 synthesizer,而 Moshi 的 codec tokens 是 LLM 直接输出、无需二次合成。

### 训练策略

三阶段训练 [§3.4, Fig 3]:

**Stage 1: Connecting Streaming Encoder** [§3.4]
- 目标: 让模型具备 streaming 语音理解能力
- 训练: MLP connector + LoRA on LLM; encoder 和 LLM backbone 冻结
- 任务: ASR (LibriSpeech 281k + GigaSpeech 200k) + QA (~730k 样本,来自 Alpaca-52k/Web Questions/TriviaQA/SQuAD/Natural Questions/VoiceAssistant-400K/UltraChat)
- 配置: 32 A100, batch 128, lr 4e-5, 50k steps [Appendix A.2]

**Stage 2: Connecting Streaming Synthesizer** [§3.4]
- 目标: 加入语音生成能力,端到端训练
- 训练: connector (encoder+synthesizer) + LoRA + streaming synthesizer; encoder 和 LLM backbone 冻结
- 任务: 在 Stage 1 基础上新增 multi-turn conversation (~80k) + context-dependent barge-in + backchanneling 样本
- Barge-in 数据设计: context-independent (10 句直接打断语) + context-dependent (用 Llama-3-8B-Instruct 生成相关/不相关问题) [§4.2]
- Backchanneling: 7 个常用回传词 (如 "Uh-huh") [§4.2]
- 配置: 32 A100, batch 128, lr 3e-5, 30k steps [Appendix A.2]

**Stage 3: DPO for Full-duplex Modeling** [§3.4]
- 问题: SFT 后模型倾向于被打断 (precision 低 0.68),语义理解不足 [§5.3.2]
- 解决: 对 barge-in + backchanneling 任务做 DPO,同时保留部分 SFT 数据维持整体性能 [§3.4]
- 配置: batch 128/256/512 对比, lr 1e-6 [Appendix A.2]

**DPO 的 U 型曲线现象** [§5.3.2, Fig 4]: DPO 训练中出现了有趣的非单调行为——模型先变得极端保守 (几乎不被打断, 10 steps 时 F1 暴跌到 0.24),然后逐渐恢复并超越 SFT (40 steps 时 overall F1 0.90 vs SFT 0.86)。这个现象在不同 batch size 下一致重现 [Tables 9-11] [论文原文]。[agent 解读] 这很可能是 DPO 的 reference model regularization 导致: 初期 policy 快速远离 SFT distribution 变得保守,随后 KL penalty 将其拉回到更优区域。这对其他全双工系统的 RL 训练有警示价值——不能只看早期 step 就放弃。

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Llama Q. Acc (S2T, predicted) | 79.3 | Moshi 60.8 / Freeze-Omni 74.2 | Llama Questions | [Table 3] |
| Llama Q. Acc (S2S, predicted) | 73.6 | Moshi 54.5 / Freeze-Omni 56.2 | Llama Questions | [Table 3] |
| Web Q. Acc (S2T, predicted) | 49.7 | Moshi 23.4 / Freeze-Omni 40.8 | Web Questions | [Table 3] |
| Web Q. Acc (S2S, predicted) | 43.7 | Moshi 22.1 / Freeze-Omni 27.9 | Web Questions | [Table 3] |
| TriviaQA Acc (S2T, predicted) | 63.6 | Moshi 25.6 / Freeze-Omni 45.1 | TriviaQA (1000 samples) | [Table 3] |
| TriviaQA Acc (S2S, predicted) | 56.0 | Moshi 16.7 / Freeze-Omni 28.5 | TriviaQA (1000 samples) | [Table 3] |
| AlpacaEval GPTScore (S2T, predicted) | 4.01 | Moshi 1.84 / Freeze-Omni 3.90 | AlpacaEval (VoiceBench) | [Table 3] |
| AlpacaEval GPTScore (S2S, predicted) | 3.22 | Moshi 1.76 / Freeze-Omni 2.46 | AlpacaEval (VoiceBench) | [Table 3] |
| Llama Q. Acc (S2T, oracle) | 80.0 | Kimi-Audio 79.7 / Qwen2.5-Omni 78.7 | Llama Questions | [Table 3] |
| Web Q. Acc (S2T, oracle) | 50.5 | miniCPM-o 47.1 / Kimi-Audio 44.0 | Web Questions | [Table 3] |
| TriviaQA Acc (S2T, oracle) | 66.0 | miniCPM-o 65.4 / Kimi-Audio 63.6 | TriviaQA (1000 samples) | [Table 3] |
| Turn-taking success (Llama Q.) | 99.7% | Moshi 85.0 / Freeze-Omni 99.7 | Llama Questions | [Table 4] |
| Turn-taking success (TriviaQA) | 92.8% | Moshi 37.1 / Freeze-Omni 72.0 | TriviaQA | [Table 4] |
| Barge-in F1 (ctx-indep, w/ echo) | 0.88 | Moshi 0.80 / Freeze-Omni 0.17 | 自建 | [Table 5] |
| Barge-in F1 (ctx-dep, DPO 40 steps) | 0.93 | SFT 0.93 (但 overall 0.86→0.90) | 自建 | [Table 6] |
| Overall barge-in F1 (DPO) | 0.90 | SFT 0.86 | 自建 | [Table 6] |
| Emotion intensity | 3.49 | Kimi-Audio 3.39 / GLM-4-Voice 3.30 | 自建 | [Table 13] |

## 局限性

1. **仅英语**: 当前仅支持英语,多语言需要重新训练 encoder 蒸馏和 synthesizer [§1] [论文原文]。

2. **320ms 最小输出延迟**: 80ms time block x 4 = 320ms 才能触发 synthesizer,这比 Moshi 的 160ms 高出一倍 [§4.1]。[agent 解读] 这是 codec-free 架构的结构性代价: 需要先积累足够 text tokens 才能驱动外部 synthesizer,而 codec injection 方案的 audio tokens 是 LLM 直接输出。

3. **LoRA 容量有限**: rank 32, scaling 1.0 的 LoRA 微调限制了 LLM backbone 的适配能力 [§4.1]。论文未探索更高 rank 或全参微调的效果。

4. **情感表达不稳定**: 虽然 emotion intensity 最高 (3.49),但"emotional expression remains inconsistent, occasionally producing responses with inappropriate or mismatched emotions" [Appendix F] [论文原文]。

5. **训练数据规模小**: ~1.3M 训练样本,远少于 Kimi-Audio 的 13M 小时 [§1]。这既是优势 (数据效率高) 也是潜在天花板。

6. **评估局限**: barge-in 和 backchanneling 评估使用自建数据集,缺乏标准化 benchmark 对比 [§4.3]。Turn-taking 只评估了"何时开始说",未评估"何时停止说"的精确性。

7. **Mamba encoder 的流式局限**: 论文未报告 Mamba encoder 在长对话中的注意力衰减问题。Mamba 的线性复杂度虽利于 streaming,但其选择性状态空间机制在超长序列上是否保持性能未被验证 [agent 解读]。

## 点评

SALMONN-omni 的核心贡献在于证明了 **codec-free 全双工** 的可行性。在 Moshi 之后,全双工 Speech LLM 似乎锁死在 "必须注入 codec" 的范式上; SALMONN-omni 用实验证明了 latent-representation 路线 (hidden embeddings) 同样能胜任全双工,且在数据效率上大幅优于 codec injection 方案。

**最有意义的发现是显式 thinking 策略**。"your LLM is secretly a full-duplex predictor" 不仅是一个工程技巧,而是一个关于 LLM 自回归本质的洞察: 状态转换 (`<think>`/`<shift>`) 可以像普通 text token 一样被生成,不需要额外的 classifier 或 FSM。这与 Raon-SpeechChat 的 SIL/BOW/BC、ELLSA 的 THINK/SHIFT/BREAK、Covo-Audio 的 THINK/SHIFT/BREAK 形成了有趣的趋同演化——不同团队独立发现了"用特殊 token 做状态转换"这一范式。

**DPO 的 U 型曲线** 是另一个值得关注的发现。全双工场景下 RL 的训练动力学与文本 LLM 不同: barge-in 判断需要语义理解和时序感知的结合,DPO 初期的过度保守化可能是模型在探索新的 precision-recall 平衡。论文首次在全双工 Speech LLM 中应用 RL,为后续工作提供了基线。

**不足之处**: (1) 与 Moshi 的对比不够公平——Moshi 是 2024 年 10 月的模型,且论文承认"it's difficult to force moshi speak" [§4.3],即 Moshi 在 oracle turn-taking 下未被评估; (2) 论文未讨论 codec-free 方案在语音质量/韵律自然度上的表现,S2S 评估仅通过 Whisper 转写后计算 accuracy,缺少 MOS 或 MUSHRA 等感知质量指标; (3) 80ms time block 是固定值,未探索自适应 block size 的可能性。

## 可复用的 idea

1. **显式 thinking 策略** (`<think>` + `<shift>`): 任何需要在 LLM 输出中嵌入控制信号的场景都可借鉴——不需要额外 classifier,只需将控制 token 作为正常自回归序列的一部分训练。对话系统中的情感切换、主题转移、多语言切换等都可能用类似方法实现 [§3.3]。

2. **Assistant stream 回听**: 在任何需要 echo cancellation 或自我感知的系统中,将模型自身上一轮输出的 embedding 作为下一轮输入的一部分,可以大幅提升上下文一致性。不仅限于语音——文本对话中的"回读自己上一轮回答"也可能有类似效果 [§3.3, §5.1]。

3. **LLM 中间层 embedding 驱动下游模块**: 第 24 层 (而非最后一层) embedding 用于驱动 speech synthesizer,避免语音-文本表征过度耦合。这一发现可推广到任何 LLM 驱动的多模态输出场景: 如果输出模态的 decoder 需要 LLM embedding 作为输入,中间偏后层可能比最后一层更好 [Appendix E, Table 12]。

4. **DPO 用于全双工 turn-taking 优化**: 首次证明 DPO 可以有效提升 barge-in/backchanneling 的语义理解能力。关键是保留部分 SFT 数据防止整体性能退化,以及耐心等待 U 型曲线恢复而非在初期保守阶段终止训练 [§3.4, §5.3.2]。

5. **Mamba 做 streaming speech encoder**: 用 Mamba (线性复杂度 SSM) 替代 Transformer 做流式语音编码,通过 Whisper 蒸馏获得 generalization,兼顾效率和性能。适用于任何需要低延迟 streaming 语音输入的场景 [§3.1]。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节围绕四个子问题展开因果解释,每个设计选择都有 WHY; thinking 策略对比有详细机制分析 |
> | 可信赖 | pass | 数字型 claim 均标注 [§/Table/Fig]; 实验表格含对比基准和出处; 指标名使用正确 |
> | 可区分 | pass | 因果解释来源标注覆盖率 >80%; [论文原文] vs [agent 解读] 区分清晰; 推断均有限定词 |
> | 可定位 | pass | KB 背景有具体谱系定位 (codec injection vs latent-representation 路线对比); 速查卡片 5 字段均有实质内容 |
> | 不污染 | pass | 未新建概念页; 无 overclaim; frontmatter concepts/models 语义正确 |
> 
> Issues: 2 (high: 0, medium: 0, low: 2)
> 详见 `_review/SALMONN-omni-review.yml`
