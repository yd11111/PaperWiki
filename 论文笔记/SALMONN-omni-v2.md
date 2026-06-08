---
type: paper
tier: deep
title: "SALMONN-omni: A Standalone Speech LLM without Codec Injection for Full-duplex Conversation"
arxiv_id: "2505.17060"
source: "Sources/SALMONN-omni-v2.pdf"
authors: [Wenyi Yu, Siyin Wang, Xiaoyu Yang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Guangzhi Sun, Lu Lu, Yuxuan Wang, Chao Zhang]
year: 2025
venue: "arXiv"
tags: [full-duplex, speech-LM, codec-free, standalone, thinking-strategy, streaming, turn-taking, barge-in, backchannel, DPO, Mamba, echo-cancellation, reinforcement-learning, CosyVoice2]
concepts: ["[[概念库/Full-duplexSpokenDialogue|Full-duplex Spoken Dialogue]]", "[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/StreamingSpokenDialogue|Streaming Spoken Dialogue]]", "[[概念库/Turn-takinginSpokenDialogue|Turn-taking in Spoken Dialogue]]", "[[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation for Speech LLM]]", "[[概念库/SpokenDialogueEvaluation|Spoken Dialogue Evaluation]]"]
models: ["[[论文笔记/Moshi|Moshi]]", "[[模型库/CosyVoice2|CosyVoice2]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页: [[概念库/SpeechLanguageModel|SpeechLanguageModel]]✓, [[概念库/Full-duplexSpokenDialogue|Full-duplexSpokenDialogue]][待确认], [[概念库/StreamingSpokenDialogue|StreamingSpokenDialogue]][待确认], [[概念库/Turn-takinginSpokenDialogue|Turn-takinginSpokenDialogue]][待确认], [[概念库/ModalityAdaptationforSpeechLLM|ModalityAdaptationforSpeechLLM]][待确认], [[概念库/SpokenDialogueEvaluation|SpokenDialogueEvaluation]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[概念库/SpeechLanguageModel|SpeechLanguageModel]]✓ | 过滤: [[概念库/Full-duplexSpokenDialogue|Full-duplexSpokenDialogue]](pending-review), [[概念库/StreamingSpokenDialogue|StreamingSpokenDialogue]](pending-review), [[概念库/Turn-takinginSpokenDialogue|Turn-takinginSpokenDialogue]](pending-review), [[概念库/ModalityAdaptationforSpeechLLM|ModalityAdaptationforSpeechLLM]](pending-review), [[概念库/SpokenDialogueEvaluation|SpokenDialogueEvaluation]](pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: KB 记录了 SpeechLM 从 GSLM→AudioLM→SpeechGPT→Moshi 的演进,其中全双工系统存在两大阵营: (1) audio-token-based (Moshi/SyncLLM/OmniFlatten,将 codec tokens 注入 LLM 词表) 和 (2) latent-representation-based (SALMONN/Qwen-Audio,通过连续 embeddings 连接)。此前 latent-representation 路线仅在半双工/turn-based 场景验证过,本文首次将其推向全双工并取得 SOTA,是该技术路线的里程碑。
>
> **[基于未确认概念页] Full-duplex Spoken Dialogue**: KB 收录了该系统家族的 v1 版本 ("SALMONN-omni (Wu et al., 2024)")。v2 是 v1 概念框架的完整系统化实现: v1 仅有 case study 无定量评估,v2 补齐了全部架构细节、三阶段训练、DPO 优化、定量消融和全面的基线对比。KB 全双工谱系中,SALMONN-omni 是唯一同时满足 "standalone" (单一 LLM 进程) + "codec-free" (不注入 audio tokens) 两个条件的系统。
>
> **[基于未确认概念页] Turn-taking**: KB 将 turn-taking 实现分为 VAD-based、multi-stream 隐式 (Moshi)、irq/n-irq markers (Mini-Omni2)、chunk-level state prediction (Freeze-Omni)、SIL/BOW/BC 三状态 (Raon-SpeechChat) 等路线。本文的显式 thinking (`<think>`+`<shift>`) 开辟了新路线: 仅用 2 个特殊 token,通过标准自回归生成实现状态转换,不需要额外分类器或 FSM。
>
> **[基于未确认概念页] Streaming Spoken Dialogue**: KB 记录了从 Mini-Omni (delayed parallel decoding)→LLaMA-Omni (NAR CTC)→Moshi (fully causal 160ms)→Freeze-Omni (chunk-wise) 的流式技术演进。本文的 Mamba streaming encoder (25Hz) + 80ms time block 周期同步 + CosyVoice2 interleaved generation 是一种新的组合: 用 SSM (Mamba) 替代 Transformer 做流式编码,兼顾线性复杂度和长序列处理。

> [!summary] 速查
> - **一句话**: 首个 codec-free standalone 全双工 Speech LLM,用 Mamba encoder + Llama-3-8B + CosyVoice2 的 embedding 桥接架构 + 显式 thinking (`<think>`/`<shift>`) 状态转换机制,全双工模式比此前开源 SOTA 平均高 35.9%,半双工模式与 Kimi-Audio/Qwen2.5-Omni 竞争 [§1, §5.2]
> - **路线**: dual-stream audio (environment + assistant) → Mamba encoder (32 blocks, 25Hz) → MLP connector → Llama-3-8B + LoRA (text tokens + `<think>`/`<shift>`) → linear adapter → CosyVoice2-0.5B (从 LLM 第 24 层 embedding 驱动) → speech output [§3, Fig 1, Appendix E]
> - **指标**: Predicted turn-taking 全面 SOTA: Llama Q. 79.3/73.6, Web Q. 49.7/43.7, TriviaQA 63.6/56.0, AlpacaEval 4.01/3.22 (S2T/S2S) [Table 3]; Oracle turn-taking 与半双工 SOTA 竞争 (80.0/50.5/66.0/4.05, S2T) [Table 3]; Turn-taking 成功率 92-99.7% [Table 4]; Barge-in overall F1 0.90 (DPO 后 vs SFT 0.86) [Table 6]
> - **可借鉴**: (1) 显式 thinking: 2 个特殊 token 让 LLM 自然学会全双工状态转换 ("your LLM is secretly a full-duplex predictor") [§3.3]; (2) assistant stream 回听将 turn-taking 成功率从 ~70% 提升到 ~90% [§5.1]; (3) DPO U 型曲线: 先极端保守再恢复超越 SFT,对全双工 RL 训练有警示价值 [§5.3.2, Fig 4]; (4) LLM 第 24 层 embedding 而非最后层驱动 synthesizer [Appendix E]
> - **局限**: 仅英语 [§1]; 80ms time block 引入 320ms 最小输出延迟 [§4.1]; 情感表达不稳定 [Appendix F]; LoRA rank 32 容量有限 [§4.1]; 训练数据 ~1.3M 样本远少于 Kimi-Audio 13M 小时 [§1]; barge-in 评估使用自建数据集 [§4.3]

## 核心问题

SALMONN-omni v2 要解决现有全双工 Speech LLM 的两个结构性缺陷 [§1]:

1. **Codec injection 的代价**: Moshi/SyncLLM/OmniFlatten 将 audio codec tokens 注入 LLM 词表,需大规模 speech-text 数据防止灾难性遗忘,且 speech 模态性能始终落后于 text 模态 (Moshi S2T 60.8 vs S2S 54.5 on Llama Q. [Table 3]) [论文原文]。此外 synthesizer 必须仅凭 codec stream 推断何时输出语音何时沉默,timing 不精确 [§1]。

2. **非独立双进程架构的复杂性**: VITA/Freeze-Omni/MinMo 通过 encoder/synthesizer 连接避免了 codec injection,但单个 LLM 实例只能要么听要么说,需运行两个 LLM 进程才能实现全双工,引入额外计算/内存开销与上下文割裂 [§1, §2.1]。Freeze-Omni 依赖 VAD 模块,在有 echo 时 F1 从 0.68 暴跌到 0.17 [Table 5] [论文原文]。

本文的核心命题: 能否用**单个 LLM backbone + 不注入 codec + 不依赖 VAD** 实现独立全双工? 分解为四个子问题 [§3]:
- 如何支持 streaming 输入输出 → Mamba encoder + CosyVoice2 synthesizer + embedding 桥接
- 如何同时处理环境声和自身语音 → 双流 (environment + assistant) 交错
- 如何对齐 audio 和 text 的时间 → 80ms time block 周期同步
- 如何决定何时说何时听 → 显式 thinking 策略 (`<think>` + `<shift>`)

**与 v1 ([[论文笔记/SALMONN-omni-v1|SALMONN-omni v1]]) 的关系**: v1 (2411.18138) 是概念框架和定性 case study,本文是完整系统论文。v2 相比 v1 的关键变化: (1) 公开了全部架构细节 (Mamba+Llama-3-8B+CosyVoice2); (2) 将 v1 的 `<start_speak>`/`<end_speak>`/`<think>` 简化为 `<think>`/`<shift>` 二 token 方案; (3) 引入 assistant stream 回听和 DPO Stage 3; (4) 提供了全面的定量评估和消融实验。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

三组件端到端架构 [§3, Fig 1]:

1. **Mamba Streaming Speech Encoder** [§3.1, §4.1]: 32 个 Mamba LM blocks,2048 维隐状态。输入侧: 提取 100Hz log-Mel 特征 → 两层卷积降采样到 50Hz → 相邻两帧拼接为一帧 → 25Hz embedding 输出。训练: 以 Whisper-large-v3 做 teacher,L1 loss 知识蒸馏; 预训练于 LibriHeavy + GigaSpeech,300k steps, batch 512 [Appendix A.2]。

2. **Llama-3-8B-Instruct** [§4.1]: LLM backbone,LoRA 微调 (rank 32, scaling 1.0)。通过 MLP connector 接收 encoder embedding。输出: text tokens (正常文本)、`<think>` (thinking 状态)、`<shift>` (状态转换)、`<ans>` (开始回答)。

3. **CosyVoice2-0.5B Streaming Speech Synthesizer** [§3.2, §4.1]: 采用 CosyVoice2 的 fixed-length interleaved generation: 每 4 个 text tokens → 12 个 speech tokens (480ms 语音)。关键改造: 原始文本输入被替换为 LLM backbone **第 24 层** 的 output embeddings,经 linear transformations 对齐后送入 synthesizer [§3.2, Appendix E]。

**双流交错 (interleaving two streams into one)** [§3.3]: 对话分为 environment stream (用户语音 + 背景噪声 + 助手 echo) 和 assistant stream (模型上一轮生成的语音)。在每个 80ms time block 内,按固定顺序: (1) 处理 environment stream embedding → (2) 处理 assistant stream embedding (dual-channel input) → (3) 生成一个 text/state token [论文原文]。

### 关键设计选择

**为什么 codec-free?** [§1, §2.1] Codec injection 概念简洁但有三重代价: (1) 需大规模数据防灾难性遗忘; (2) speech 模态性能始终不如 text (modality gap); (3) timing 判断不精确 [论文原文]。SALMONN-omni 通过 hidden embeddings 桥接 encoder/synthesizer,保持 LLM 原始词表不变,仅需 LoRA 适配 [§3.1, §3.2]。[agent 解读] 这是 KB 中 latent-representation-based 路线首次在全双工场景成功——此前该路线仅在半双工系统 (原始 [[论文笔记/SALMONN|SALMONN]], Qwen-Audio) 中验证过,全双工被认为必须用 codec injection。

**为什么用显式 thinking 而非隐式 thinking?** [§3.3, Fig 2, Table 2] 两种策略对比:
- **隐式 thinking**: 每个 time block 预测 `<listen>` 或 `<speak>`,但这些 token 不回馈到 LLM 输入序列,缺乏从输出到输入的完整反馈循环 [论文原文]。
- **显式 thinking**: `<think>` 和 `<shift>` 两个 token 混入 LLM 输入序列。`<think>` 在 listening 状态每个 time block 生成 (模拟"思考要不要说话"),在 speaking 状态文本生成完毕后 synthesizer 仍在播放时也生成 (等待 speaking→listening 转换); `<shift>` 标记 listening↔speaking 双向状态转换 [§3.3]。

Stage 1 消融: 显式全面胜出,AlpacaEval S2T 从 3.73 提升到 4.48 [Table 2]。论文归因: "training the LLM to output sequences containing both normal responses and state-transition-related tokens better aligns with its autoregressive nature" [§3.3] [论文原文]。核心口号: "your LLM is secretly a full-duplex predictor"。

[agent 解读] 这与 v1 的设计有本质变化。v1 使用 `<start_speak>`/`<end_speak>` 两个方向性 token + `<think>` 输入 token 配合负系数 loss。v2 简化为 `<think>`/`<shift>` 二 token,`<shift>` 不区分方向 (双向通用),且取消了负系数 loss——改为标准自回归生成。这种简化验证了"越简单越好"的设计直觉: 让 listening 状态的操作极简 (只输出一个 `<think>`),模型就能专注于学习状态转换和 speaking 状态的内容生成。

**为什么简单 `<think>` 比丰富 thinking 内容更好?** [Appendix C, Table 8] 论文探索了 5 种 thinking 变体:
- Implicit (基线): 无状态 token 回馈
- Implicit-ASR: listening 状态做 streaming ASR
- Explicit (采用方案): listening 状态仅生成 `<think>` 单 token
- Explicit-ASR: listening 状态做 ASR 且回馈到输入
- Explicit-NS: `<think>`+`<shift>` 但对非转换时刻施加负 loss

结果: 最简单的 Explicit 最好 (test-clean WER 2.40 vs Explicit-ASR 5.09 vs Explicit-NS 7.65) [Table 8]。论文解释: 增加 thinking 内容的多样性使 speaking/listening 两个状态的操作模式变得难以区分,模型需同时学状态转换和模式区分,负担过重。简单 `<think>` 让 listening 状态操作极简,模型可专注于核心能力 [Appendix C] [论文原文]。此外,inference 时 "thinking" ASR 内容如果出错会通过自回归传播偏差到 speaking 状态的生成 [论文原文]。

**为什么 assistant stream 回听至关重要?** [§5.1] Stage 1 (无 assistant stream) 的 turn-taking 成功率仅 ~70% (AlpacaEval),Stage 2 引入 assistant stream 后提升到 ~90% [论文原文]。[agent 解读] 两个原因: (1) 模型需知道自己在说什么才能判断何时停止; (2) 回听提供 echo 信息用于 echo cancellation。Table 5 佐证: Freeze-Omni 不做 standalone 回听,echo 场景 F1 从 0.68 暴跌到 0.17。

**为什么选 LLM 第 24 层 embedding 驱动 synthesizer?** [Appendix E, Table 12] 系统消融: 前 8 层 S2S 差 (大部分 LLM 层未参与语音生成); 最后一层 S2T 性能最差 (紧耦合语音-文本表征可能降低语言理解能力); 第 24 层在 S2T 和 S2S 之间取得最佳平衡 [论文原文]。[agent 解读] 这与"LLM 中间偏后层保留更多通用信息,最深层过度特化于 token prediction"的一般观察一致。对任何 LLM 驱动的多模态输出场景,中间偏后层可能优于最后层。

**80ms time block 的设计权衡** [§4.1]: 每 time block 处理 80ms 输入,生成 1 个 text token。4 个 tokens 触发 synthesizer 生成 12 speech tokens (480ms)。首个 speech token 最小延迟: 4 x 80ms = 320ms [论文原文]。[agent 解读] 这比 Moshi 的 160ms 理论延迟高一倍——codec-free 架构的结构性代价: 需要先积累足够 text tokens 才能驱动外部 synthesizer,而 Moshi 的 codec tokens 是 LLM 直接输出。

### 训练策略

三阶段训练 [§3.4, Fig 3]:

**Stage 1: Connecting Streaming Encoder** [§3.4]
- 目标: streaming 语音理解能力
- 可训练: MLP connector + LoRA; encoder/LLM backbone 冻结
- 数据: ASR (LibriSpeech 281k + GigaSpeech 200k) + QA (~730k,来自 Alpaca-52k/WebQ/TriviaQA/SQuAD/NQ/VoiceAssistant-400K/UltraChat) [§4.2, Table 7]
- 配置: 32 A100, batch 128, lr 4e-5, 50k steps [Appendix A.2]

**Stage 2: Connecting Streaming Synthesizer** [§3.4]
- 目标: 端到端语音生成 + 全双工交互
- 可训练: connectors (encoder+synthesizer) + LoRA + streaming synthesizer; encoder/LLM backbone 冻结
- 新增数据: multi-turn conversation (~80k,由 Llama-3-8B-Instruct 基于 TriviaQA/NQ 主题生成,CosyVoice2 合成语音) + barge-in + backchanneling 样本 [§4.2]
- Barge-in 数据: context-independent (GPT-4o 生成 10 句直接打断语) + context-dependent (Llama-3-8B-Instruct 生成相关/不相关问题) [§4.2]
- Backchanneling: 7 个常用回传词 (如 "Uh-huh") [§4.2]
- 多说话人 barge-in: 训练时考虑第三方说话人打断场景 [§4.2]
- 配置: 32 A100, batch 128, lr 3e-5, 30k steps [Appendix A.2]

**Stage 3: DPO for Full-duplex Modeling** [§3.4]
- 问题: SFT 后模型倾向于被打断 (precision 低,仅 0.68 on ctx-independent),语义理解不足 [§5.3.2]
- 解决: DPO 对 barge-in + backchanneling 任务后训练,同时保留部分 SFT 数据维持整体性能 [§3.4]
- 配置: batch 128/256/512 对比, lr 1e-6 [Appendix A.2]

**DPO 的 U 型曲线现象** [§5.3.2, Fig 4, Tables 9-11]: DPO 训练出现非单调行为——模型先变得极端保守 (10 steps 时 recall 暴跌: ctx-independent recall 0.07, overall F1 0.24),然后逐渐恢复并超越 SFT (40 steps, batch 256 时 overall F1 0.90 vs SFT 0.86)。该现象在 batch 128/256/512 三种设置下一致重现 [Tables 9-11] [论文原文]。

[agent 解读] 这很可能是 DPO 的 reference model 正则化效应: 初期 policy 快速远离 SFT 分布变得过度保守 (对所有打断都"忍耐"),随后 KL penalty 将其拉回到更优区域。这对全双工系统的 RL 训练有重要警示: 不能看到初期退化就终止训练,需要给模型足够的步数恢复。

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
| Avg. relative improvement (predicted) | +35.9% | over previous FD SOTA | 4 datasets | [§5.2] |
| Llama Q. Acc (S2T, oracle) | 80.0 | Kimi-Audio 79.7 / Qwen2.5-Omni 78.7 | Llama Questions | [Table 3] |
| Web Q. Acc (S2T, oracle) | 50.5 | miniCPM-o 47.1 / Kimi-Audio 44.0 | Web Questions | [Table 3] |
| TriviaQA Acc (S2T, oracle) | 66.0 | miniCPM-o 65.4 / Kimi-Audio 63.6 | TriviaQA (1000 samples) | [Table 3] |
| AlpacaEval GPTScore (S2T, oracle) | 4.05 | miniCPM-o 3.99 / VITA-1.5 3.92 | AlpacaEval (VoiceBench) | [Table 3] |
| Web Q. Acc (S2S, oracle) | 45.4 | miniCPM-o 42.7 / Baichuan 40.7 | Web Questions | [Table 3] |
| TriviaQA Acc (S2S, oracle) | 58.8 | miniCPM-o 56.0 / Baichuan 53.0 | TriviaQA (1000 samples) | [Table 3] |
| Turn-taking succ. (Llama Q.) | 99.7% | Moshi 85.0 / Freeze-Omni 99.7 | Llama Questions | [Table 4] |
| Turn-taking succ. (TriviaQA) | 92.8% | Moshi 37.1 / Freeze-Omni 72.0 | TriviaQA | [Table 4] |
| Turn-taking succ. (AlpacaEval) | 92.0% | Moshi 83.4 / Freeze-Omni 87.9 | AlpacaEval | [Table 4] |
| Barge-in F1 ctx-indep. (w/ echo) | 0.88 | Moshi 0.80 / Freeze-Omni 0.17 | 自建 (100+100 samples) | [Table 5] |
| Barge-in overall F1 (SFT) | 0.86 | — | 自建 | [Table 6] |
| Barge-in overall F1 (DPO-40, bs256) | 0.90 | SFT 0.86 | 自建 | [Table 6] |
| Emotion intensity | 3.49 | Kimi-Audio 3.39 / GLM-4-Voice 3.30 | 自建 | [Table 13] |

**关键实验解读**:

1. **Predicted vs oracle turn-taking**: predicted 场景是全双工模型的真正考验——模型必须自己判断何时开始说话。SALMONN-omni 在此场景下相对 Moshi 提升 35.9%,相对 Freeze-Omni 也有显著优势。在 oracle 场景下,它与 Kimi-Audio/Qwen2.5-Omni 等使用远更多训练数据 (最多 13M 小时 [§1]) 的半双工系统竞争,证明数据效率很高 [§5.2]。

2. **Turn-taking 成功率的维度差异** [Table 4]: SALMONN-omni 在短输入场景 (TriviaQA 92.8%, AlpacaEval 92.0%) 显著优于 Moshi (37.1%, 83.4%) 和 Freeze-Omni (72.0%, 87.9%)。论文归因: 显式 thinking 机制使模型对短输入也能准确判断何时开始响应 [§5.3.1] [论文原文]。

3. **Echo cancellation 是 standalone 架构的关键优势** [Table 5]: SALMONN-omni 在有 echo (x1.0 echo factor) 时 F1 0.88,Freeze-Omni 在仅 x0.1 echo 时即跌到 0.31,x1.0 时仅 0.17。原因: Freeze-Omni 依赖外部 VAD,VAD 无法区分自身 echo 和用户语音; SALMONN-omni 通过 assistant stream 回听让 LLM 同时看到自身输出和环境输入,天然支持 echo cancellation [论文原文]。

4. **多说话人 barge-in**: 评估中一半打断来自第三方说话人 (非原始用户),SALMONN-omni 在此设置下仍保持高 F1,说明模型不是简单地对"有语音输入"做反应,而是理解了打断的语义内容 [§4.3]。

## 局限性

1. **仅英语**: 训练数据 (LibriSpeech/GigaSpeech/Alpaca) 均为英语,多语言扩展需重新训练 encoder 蒸馏和 synthesizer [§1] [论文原文]。

2. **320ms 最小输出延迟**: 4 x 80ms time block = 320ms 才能触发 synthesizer,比 Moshi 的 160ms 高一倍 [§4.1]。[agent 解读] 这是 codec-free 架构的结构性代价——需先积累足够 text tokens 驱动外部 synthesizer,而 Moshi 直接输出 codec tokens。论文未探索自适应 block size 或可变 text-to-speech ratio 的可能性。

3. **LoRA 容量受限**: rank 32, scaling 1.0 限制了 LLM backbone 的适配能力,论文未探索更高 rank 或全参微调 [§4.1]。

4. **情感表达不稳定**: 虽然 emotion intensity 最高 (3.49 [Table 13]),但"emotional expression remains inconsistent, occasionally producing responses with inappropriate or mismatched emotions" [Appendix F] [论文原文]。

5. **训练数据规模小**: ~1.3M 样本 (~1.2k 小时 ASR + 语音合成的 QA/对话),远少于 Kimi-Audio 13M 小时 [§1]。数据效率高是优势,但也可能是性能天花板。

6. **评估局限**: (a) barge-in/backchanneling 评估使用自建数据集,非标准化 benchmark [§4.3]; (b) S2S 评估仅通过 Whisper 转写后计算 accuracy/GPTScore,缺少 MOS/MUSHRA 等感知质量指标; (c) 与 Moshi 对比不完全公平——Moshi 是 2024 年 10 月模型,且论文承认 "it's difficult to force moshi speak" [§4.3]; (d) turn-taking 仅评估"何时开始说",未评估"何时停止说"的精确性。

7. **Mamba encoder 长对话未验证**: Mamba 的线性复杂度利于 streaming,但其选择性状态空间机制在超长对话中是否有注意力衰减未被测试 [agent 解读]。

## 点评

SALMONN-omni v2 的核心贡献在于 **将 codec-free 全双工从概念 (v1) 推进到 SOTA 系统**。在 Moshi 之后,全双工 Speech LLM 似乎锁死在 "必须注入 codec" 的范式上——SyncLLM、OmniFlatten 都沿此路线。SALMONN-omni 用实验证明 latent-representation 路线同样能胜任全双工,且在数据效率上大幅优于 codec injection (1.3M 样本 vs Moshi 数十万小时)。这为后续工作打开了一条新路。

**最有价值的发现是显式 thinking 策略和 assistant stream 回听的组合效应**。`<think>`/`<shift>` 二 token 方案比 v1 的三 token 方案更简洁,比各种 ASR-based thinking 变体更有效 (WER 2.40 vs 5.09-10.5 [Table 8])。"your LLM is secretly a full-duplex predictor" 不仅是工程口号,而是一个关于 LLM 自回归本质的洞察: 状态转换是正常 token 生成的自然延伸,不需要额外模块。这一发现与 Raon-SpeechChat (SIL/BOW/BC)、ELLSA (THINK/SHIFT/BREAK)、Covo-Audio (THINK/SHIFT/BREAK) 形成趋同演化——不同团队独立发现了"用特殊 token 做状态转换"这一范式。

**DPO U 型曲线** 是另一个重要贡献。这是首次在全双工 Speech LLM 中系统性地应用 RL。U 型行为 (先保守再恢复) 在三种 batch size 下一致重现 [Tables 9-11],说明这不是偶然现象,而是全双工 barge-in 判断中 precision-recall trade-off 在 DPO 优化下的固有动力学。这对后续工作有实用价值: DPO 训练全双工模型时应预期早期退化,需要给足步数等待恢复。

**定量评估的完整性相对较好但有盲区**: oracle turn-taking 对比了 8 个系统 [Table 3],覆盖了主流半双工和全双工模型。但缺少: (1) 语音质量的直接评估 (MOS); (2) 标准化的 turn-taking benchmark (如 Full Duplex Bench); (3) 长对话场景的评估。情感表达 [Appendix F, Table 13] 虽有探索但承认不稳定,更像是附加实验而非核心贡献。

**在 SALMONN 家族中的定位**: 从原始 SALMONN (2023, 纯理解, dual encoder + Q-Former) → SALMONN-omni v1 (2024, codec-free 全双工概念) → SALMONN-omni v2 (2025, 完整系统),ByteDance 的语音团队完成了从"听懂"到"实时对话"的完整技术演进。v2 的 Mamba encoder 可视为对原始 SALMONN 的 Whisper encoder 的 streaming 化替代 (通过蒸馏保留 Whisper 知识); CosyVoice2 synthesizer 则直接引入了集团内部的 TTS 能力。

## 可复用的 idea

1. **显式 thinking 策略** (`<think>` + `<shift>`): 任何需要在 LLM 输出中嵌入控制信号的场景都可借鉴。关键是: 不需要额外 classifier 或 FSM,将控制 token 作为正常自回归序列的一部分训练; listening 状态的操作越简单越好 (仅一个 token),复杂 thinking 内容反而有害 [§3.3, Appendix C]。可推广到: 对话中的情感切换、多语言切换、主题转移等控制场景。

2. **Assistant stream 回听**: 在任何需要 echo cancellation 或自我感知的系统中,将模型自身上一轮输出的 embedding 作为下一轮输入,可大幅提升上下文一致性和 echo 鲁棒性。核心效果: turn-taking 成功率从 ~70% 提升到 ~90% [§3.3, §5.1]。

3. **LLM 中间层 embedding 驱动下游模块**: 第 24 层 (而非最后一层) 用于驱动 synthesizer,避免语音-文本表征过度耦合。可推广到任何 LLM 多模态输出场景: 若下游 decoder 需要 LLM embedding 输入,中间偏后层可能优于最后层 [Appendix E, Table 12]。

4. **DPO 后训练全双工 turn-taking**: 首次证明 DPO 可提升 barge-in/backchanneling 的语义理解。实用要点: (a) 保留部分 SFT 数据防整体退化; (b) 预期 U 型曲线,不要在初期保守阶段终止训练; (c) batch size 128-512 均可,但恢复速度不同 [§3.4, §5.3.2, Fig 4]。

5. **Mamba 做 streaming speech encoder**: 用 SSM (线性复杂度) 替代 Transformer 做流式编码,通过 Whisper 蒸馏获得泛化能力。适用于任何低延迟 streaming 语音输入场景——Mamba 的线性复杂度使其比 Transformer 更适合长序列 streaming 处理 [§3.1]。

6. **简化胜于复杂的 thinking 设计**: Appendix C 的消融证明,在 thinking 阶段做 streaming ASR 或生成 inner thoughts 反而有害。这一发现对所有"在等待期间让模型做额外工作"的设计有参考价值: 额外工作可能破坏状态区分度并引入 error propagation [Appendix C, Table 8]。

---

> [!review] 审阅: pass, 0 high (2026-06-08)
> 详见 [[_review/SALMONN-omni-v2-review.yml]]

---

*检索命中: [[概念库/SpeechLanguageModel|SpeechLanguageModel]]✓ | 过滤: [[概念库/Full-duplexSpokenDialogue|Full-duplexSpokenDialogue]](pending-review), [[概念库/StreamingSpokenDialogue|StreamingSpokenDialogue]](pending-review), [[概念库/Turn-takinginSpokenDialogue|Turn-takinginSpokenDialogue]](pending-review), [[概念库/ModalityAdaptationforSpeechLLM|ModalityAdaptationforSpeechLLM]](pending-review), [[概念库/SpokenDialogueEvaluation|SpokenDialogueEvaluation]](pending-review) | 未命中但可能相关: 无*
