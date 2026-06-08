---
type: paper
tier: deep
title: "Audio Interaction Model"
arxiv_id: "2606.05121"
source: "Sources/AudioInteractionModel.pdf"
authors: [Zhifei Xie, Zihang Liu, Ze An, Xiaobin Hu, Yue Liao, Ziyang Ma, Dongchao Yang, Mingbao Lin, Deheng Ye, Shuicheng Yan, Chunyan Miao]
year: 2026
venue: "arXiv preprint"
tags: [streaming, audio-LM, interaction, proactive, full-duplex, unified-model, real-time, FIFO-inference, dataset]
concepts: ["[[Full-duplexSpokenDialogue]]", "[[StreamingSpokenDialogue]]", "[[SpeechLanguageModel]]", "[[Turn-takinginSpokenDialogue]]", "[[AudioUnderstanding]]"]
models: ["[[Qwen2.5-Omni]]"]
tasks: []
datasets: ["StreamAudio-2M", "ProactiveSound-Bench"]
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[Full-duplexSpokenDialogue]][待确认], [[StreamingSpokenDialogue]][待确认], [[Turn-takinginSpokenDialogue]][待确认], [[AudioUnderstanding]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: Audio-Interaction 试图开辟 SpeechLM 的一个新范式层级。KB 中 SpeechLM 的演进线是 Traditional → Streaming → Full-duplex,Audio-Interaction 提出在 Full-duplex 之上还有一层 **"Audio Interaction"**:不仅是对话级别的双工,而是对整个连续音频流(语音 + 环境声 + 音乐 + 噪声)做统一的感知-决策-响应。这一定位超越了现有 Full-duplex Spoken Dialogue 概念页中记录的所有系统(Moshi、LSLM、Raon-SpeechChat 等),后者仍聚焦于"人说话时模型能否打断/回传"这一子问题。

**与同作者 LSLM 的关系**: Ziyang Ma 是 LSLM 共同一作(当时在 X-LANCE/SJTU),现转至 NTU。LSLM 在 106M 参数、585h LibriTTS 上探索了"边说边听"的 middle fusion 和 IRQ token 方案;Audio-Interaction 将这一思路从单任务 turn-taking 检测扩展到 7 大类 28 子任务的统一交互框架,参数量从 106M 升至 3B,训练数据从 585h 升至 302k 小时。核心范式也从"在生成通道融入监听信号"转变为"在每个 chunk 用 <silent>/<response> token 做全局决策"。

**已有认知**:
- **Turn-taking 演进**: VAD → irq/n-irq (Mini-Omni 2) → IRQ token (LSLM) → State 0/1/2 (Freeze-Omni) → SIL/BOW/BC (Raon-SpeechChat)。Audio-Interaction 的 <silent>/<response> 二元决策是另一个分支,更接近 Freeze-Omni 的 chunk-level state prediction,但简化为二元而非三态。
- **Streaming inference**: KB 记录了多种延迟优化技术(RQ-Transformer 160ms, Chunk-wise encoder, ARIA 等)。Audio-Interaction 的 FIFO 异步调度是新方案:编码和解码完全解耦为两个独立进程,通过时序队列通信。
- **Audio Understanding**: KB 区分了 SpeechLM 路线和 ALM 路线。Audio-Interaction 是首个将 ALM 级别的通用音频理解(环境声、音乐、非语音事件)与 SpeechLM 级别的流式交互统一到单模型的工作。

**创新判断**: 核心创新不在于单一技术突破,而在于系统级整合:将离线 LALM 能力 + 流式交互 + 主动响应统一为 perceive-decide-respond 循环,并配套完整的数据-训练-推理 pipeline。ProactiveSound-Bench 是首个评估"模型能否在无指令情况下主动响应音频事件"的 benchmark。

## 速查

> [!summary] 速查
> - **一句话**: 将离线 Large Audio Language Model 转为 always-on 流式交互模型,通过 perceive-decide-respond 循环在每个 400ms chunk 决定沉默或响应,统一对话/ASR/翻译/主动响应等 7 大类 28 子任务
> - **路线**: 连续音频流 → Audio Encoder (chunk-wise 400ms) → Adapter → Qwen2.5-Omni-3B (LLM) → 每步预测 <silent>/<response> + 文本响应; FIFO 异步推理解耦编码与解码
> - **指标**: MMAU 58.15 (vs Qwen2.5-Omni-3B 57.81), CoVoST2 en-zh BLEU 55.22 (+15.72), ProactiveSound-Bench 61.2/62.8 (Sin./Mul.), 首帧延迟 392ms [Table 1-5]
> - **可借鉴**: (1) FIFO 异步推理 -- 编码器持续入队,解码器按需出队,消除 stall 并降低首帧延迟 4.5x; (2) 层次化事件策划 -- LLM 规划场景+事件细化+clip grounding 构造语义一致的流式训练数据; (3) 双目标训练 L = L_LM + λ*L_stream 给流控 token 独立优化头; (4) 综合性 silence 训练 -- history review + 验证后静音数据抑制 false triggering
> - **局限**: (1) 仅 3B 参数,对话质量不及 7B omni models (Alpaca 4.28 vs Qwen2.5-Omni-7B 4.49); (2) 无语音输出(text-only response),不是端到端 speech-to-speech; (3) StreamAudio-2M 由合成语音(CosyVoice)构造,非真实交互数据; (4) ProactiveSound-Bench 仅 644 样本,较小

## 核心问题

1. **离线范式与音频实时性的根本矛盾**: 现有 LALMs 遵循 y = f(x, A) 的离线公式,必须观察完整音频后才响应,完全不匹配音频的连续、实时本质 [§1]
2. **流式模型各自为政**: 每种流式能力(对话、ASR、翻译)需要独立训练一个模型,无法统一 [§1]
3. **两大技术挑战**:
   - (C1) 基于理解的响应触发: 交互模型必须在每个 chunk 基于语义理解(而非声学表面特征)决定是否响应,监督信号稀疏且时间模糊 [§1]
   - (C2) 分块推理下的实时上下文连续性: 固定长度 chunk 打断了时间连续性和长程上下文,模型必须跨 chunk 重建连续性 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Audio-Interaction 将音频交互形式化为 perceive-decide-respond 循环 [§3.1, Eq. 1]:
```
(d_t, r_t) = f(a_≤t, d_<t, r_<t)
```
其中 a_t 是当前 400ms 音频 chunk,d_t 是流控决策(<silent> 或 <response>),r_t 是生成的文本响应。模型基于 Qwen2.5-Omni-3B 初始化,包含:
- **Audio Encoder**: chunk-wise 编码,每个 chunk 有独立 position embedding,无跨 chunk encoder attention [§3.1]
- **Adapter**: 将 chunk-wise 声学表征映射到 LLM 空间 [§3.3]
- **LLM (Qwen2.5-Omni-3B)**: 每步接收一个 chunk 的 encoder features,预测 d_t; 若 d_t=<response> 则切换为自回归文本生成 [§3.3]

[agent 解读] 这与 LSLM 的架构逻辑完全不同: LSLM 有独立的 speaking/listening 双通道并行,用 fusion 连接;Audio-Interaction 只有一条序列,沉默和响应交替出现,没有"边说边听"的并行流。这意味着 Audio-Interaction 在响应时是半双工的(生成文本时不消费新 chunk),但通过 FIFO 调度弥补了这一限制。

### 关键设计选择

**选择 1: 400ms chunk 大小** [§3.3, Table 7]

[论文原文] 0.2s chunk 语义上下文不足(MMAU 49.7, 严重退化);0.6s/0.8s 恢复精度但延迟膨胀至 674/786ms;0.4s 在精度(MMAU 58.2)和延迟(392ms)间取得最优权衡 [§5.4, Table 7]。

[agent 解读] 400ms 与 SLAM-Omni 的每帧 100ms(5帧拼接=500ms)、Moshi 的 chunk 80ms 相比偏大。这反映了 Audio-Interaction 不追求极低延迟的全双工对话,而是在"合理延迟下的通用理解"上做 trade-off。

**选择 2: <silent>/<response> 二元决策 vs 多状态** [§3.3]

[论文原文] 模型在每个 chunk 后预测单个特殊 token d_t ∈ {<silent>, <response>}。若 d_t=<silent>,不输出文本;若 d_t=<response>,切换到自回归生成直至 <eos> [§3.3]。

[agent 解读] 对比 Freeze-Omni 的三状态(State 0=继续听, State 1=插入反馈, State 2=完整响应)和 Raon-SpeechChat 的 SIL/BOW/BC 三元,Audio-Interaction 的二元设计更简洁但丧失了 backchannel 建模能力。这一选择与论文定位一致——Audio-Interaction 不追求自然对话中的回传信号,而是追求"何时应该输出有实质内容的响应"。

**选择 3: FIFO 异步推理** [§3.4, Fig 4, Table 5]

编码器和解码器作为两个独立进程通过时序队列 Q 通信:
- **编码器** (pure producer): 持续以固定速率处理 chunk,将 features 原子地追加到 Q [§3.4]
- **解码器** (gated consumer): 检查上一步输出 r*; 若 r* ∈ {<silent>, <eos>},从 Q 原子地 drain 所有积累的 features 到 KV-cache,发出一个控制 token; 若 r* 是文本 token,继续自回归生成而不触碰 Q [§3.4, Appendix B.3]

[论文原文] Drain-on-trigger(而非 pop-one-at-a-time)使解码器的声学上下文与 wall-clock 时间对齐,避免了长响应后 stale silence decisions 的问题,这是首帧延迟降低 4.5x 的结构性来源 [§3.4]。

消除 FIFO 后,平均首帧延迟从 392ms 上升到 831ms (2.12x),stall 率从 0.0% 到 5.2% [Table 5]。

### 训练策略

**四阶段训练 pipeline** [§3.3]:

| 阶段 | 数据 | 训练模块 | 目的 |
|------|------|---------|------|
| Stage 1: Format | 离线 single-turn | LM head + new token embedding | 学习 <Spe_token> 格式 |
| Stage 2: Adapter | 离线 single-turn | Adapter only | chunk → LLM 空间映射 |
| Stage 3: Streaming SFT | StreamAudio-2M 核心 | Adapter + LLM | ASR/S2TT/对话/音频理解 |
| Stage 4: Instruction FT | 交错多轮序列 | Adapter + LLM | 连续辅助/理解干预/主动响应 |

**双目标训练** [§3.3]:
```
L = (1/N) * Σ[ -log P_θ(t_j|H_j) + λ * (-log P_θ(s_j|H_j)) ]
```
其中 t_j 是文本 token,s_j 是流控 token,λ=1.0 [Table 8]。λ 过大(2.0)伤害理解(MMAU 57.3);过小(0.5)降低触发精度(95.3% vs 96.7%) [§5.4]。

**Context Memory + Comprehension-Aware Silence** [§3.3]:
两个针对性方案解决两类失败模式:
1. **History review**: 在序列后段插入关于前段内容的问题,强迫模型保持长程记忆,解决上下文遗忘 [§3.3]
2. **Silent audio 训练**: 引入大量经 ProactiveSound-Bench agent 验证的"不需要响应"的静音/噪声片段,强化模型在不必要时保持沉默,解决 false triggering [§3.3]

### StreamAudio-2M 数据集

**规模**: 2.6M items, 302k hours, 7 大类 28 子任务 [§4.1]

| 任务类别 | 数量 | 占比 | 数据源 |
|---------|------|------|--------|
| Voice Chatting | 539k | 23.1% | MOSS, GammaCorpus |
| Str. Instr. Following | 487k | 20.8% | UltraChat, Magpie-Pro |
| Str. Audio Understanding | 382k | 16.4% | AudioSet, FMA |
| Str. Translation | 357k | 15.3% | CoVoST2, AISHELL |
| Real-time ASR | 270k | 11.6% | CommonVoice, GigaSpeech, LibriSpeech |
| Proactive Response | 171k | 7.3% | AudioSet events, AudioX |
| Env. Audio Agent | 130k | 5.5% | MOSS, AudioSet, WHAM!, DNS |

**构造 pipeline** [§3.2, §4.2, Appendix B.1/B.4]:

1. **TFJP (Time-Frequency Joint Preprocessing)**: 迭代地裁剪过长静音(silence_cut)、估计+去除背景噪声(noise_profile→denoise)、定位最密集信息段(core_locate)、半 chunk 边界对齐(boundary_norm)、短窗谱平滑(spec_smooth) [Algorithm 1]
2. **Hierarchical Event Curation**: (i) LLM 规划场景 → (ii) 细化为具体事件序列 → (iii) 检索或生成对应 clip(top-3 检索+验证,失败时用音频生成模型合成) [§3.2]
3. **CosyVoice TTS**: 文本源经 LLM 改写+ASR 验证后用多声 CosyVoice 合成语音 [Appendix B.4]
4. **双轨噪声叠加**: 前景 0dB、背景 -6dB、环境 -12dB;两层独立噪声以 crossfade 方式铺底 [Appendix B.4]

## 实验

| 指标 | 本文 (3B) | Baseline (Qwen2.5-Omni-3B) | 最佳对比 | 数据集 | 出处 |
|------|----------|---------------------------|---------|--------|------|
| MMAU (audio instr.) | 58.15 | 57.81 | Qwen2.5-Omni-7B: 49.58 | MMAU | [Table 1] |
| SpokenQA LLa.Q | 67.31 | 66.00 (3B) | Baichuan-Omni-1.5: 78.50 | SpokenQA | [Table 2] |
| Alpaca Score | 4.28 | 4.32 (3B) | Qwen2.5-Omni-7B: 4.49 | AlpacaEval | [Table 2] |
| VoiceBench SD-QA | 52.14 | 49.37 (3B) | Qwen2.5-Omni-7B: 55.71 | VoiceBench | [Table 2] |
| LibriSpeech clean WER | 3.17% | 2.87% (3B) | Canary 1B: 1.48% | LibriSpeech | [Table 3] |
| CoVoST2 en-zh BLEU | 55.22 | 39.50 (3B) | Phi-4-multi: 46.30 | CoVoST2 | [Table 3] |
| CoVoST2 zh-en BLEU | 35.21 | 18.17 (3B) | Qwen2.5-Omni-7B: 29.40 | CoVoST2 | [Table 3] |
| ProactiveSound Sin. | 61.2 | - | MiniCPM-o-4.5: 58.9 | ProactiveSound-Bench | [Table 4] |
| ProactiveSound Mul. | 62.8 | - | MiniCPM-o-4.5: 58.9 | ProactiveSound-Bench | [Table 4] |
| First-chunk latency | 392ms | - | w/o FIFO: 831ms | - | [Table 5] |
| Trigger accuracy | 96.77% | - | V2 (no TFJP): 85.35% | - | [Table 6] |
| Stability N=5 retention | >91% | Baseline: collapses 30%+ | - | - | [§5.2] |

**三个增强** [§5.2]:
- **[Enh.1] 保持离线理解能力**: 流式训练后 MMAU 从 57.81 提升至 58.15,不降反升 [Table 1]
- **[Enh.2] 核心任务竞争力**: CoVoST2 翻译 +15.72/+17.04 BLEU;对话 3/4 benchmark 持平或超越基线 [Table 2, 3]
- **[Enh.3] 解锁离线不可及能力**: 音频指令鲁棒(离线模型在 audio instruction 下急剧下降)、选择性主动响应、流式拼接下的能力稳定性 [Table 4, Fig 9]

**模型内部分析** [§5.3]:
- **[Obs.1]** 连续性在 GPT Layer 0 重建: encoder 输出的跨 chunk 边界连续性比仅 0.25,GPT Layer 0 一步提升到 0.80,通过 cross-chunk KV-cache 实现 [Fig 7]
- **[Obs.2]** 流控决策通过单一注意力头路由: 576 个 head 中 L35H14 主导所有四个任务的 <silent>/<response> 决策,消融该 head 使 S2TT token-match 下降 0.88 [Fig 8]

## 局限性

1. **无语音输出**: Audio-Interaction 只输出文本,不生成语音。在 voice chatting 场景中,需要额外 TTS 模块,不是端到端的语音交互系统 [agent 解读]
2. **合成训练数据**: StreamAudio-2M 的对话数据由 CosyVoice 合成,非真实人类交互。Appendix A.1 的 real-world validation 显示在 Travel/Commute 等嘈杂场景下 trigger accuracy 从 62.0% 降至 58.9% [Appendix A.1]
3. **半双工响应**: 一旦 d_t=<response>,模型进入自回归文本生成,不再消费新 chunk(直到 <eos>)。这意味着在长响应期间模型是"聋"的,无法处理打断 [agent 解读,基于 §3.4 的 FIFO 调度描述]
4. **3B 参数限制**: 对话质量不及 7B omni models(Alpaca 4.28 vs Qwen2.5-Omni-7B 4.49);LibriSpeech WER 轻微回归(3.17% vs 2.87%) [Table 2, 3]
5. **ProactiveSound-Bench 较小**: 仅 644 个 human-designed 样本,6 个顶级类别。作为新 benchmark 尚需社区验证 [§4.3]
6. **响应时机学习依赖合成标注**: chunk-level <silent>/<response> 标签由 pipeline 生成(Algorithm 2),不是人类标注的真实交互时机 [Appendix B.2]

## 点评

**定位准确但执行有取舍**。Audio-Interaction 精准地识别了 LALM 领域的一个重要空白——将"离线理解"和"流式交互"统一到单一模型。论文的系统性很强:数据构造(StreamAudio-2M)、训练方法(SoundFlow)、推理优化(FIFO)、评估(ProactiveSound-Bench)形成完整闭环。FIFO 异步推理的设计特别优雅,用 drain-on-trigger 机制避免了 stale context 的问题。

**但核心矛盾未解决**: 论文标题是 "Audio Interaction Model" 但模型只输出文本,无法生成语音。在 voice chatting 场景中必须外挂 TTS,这削弱了"交互模型"的完整性。此外,<response> 时的半双工本质意味着在长响应期间无法处理新事件(如用户打断),这与 Moshi/LSLM 的真正全双工相比是明显退步。

**数据贡献可能比模型更持久**: StreamAudio-2M (302k hours, 28 sub-tasks) 和 ProactiveSound-Bench 填补了流式音频交互数据的空白。特别是 hierarchical event curation pipeline 和 TFJP preprocessing 为后续工作提供了可复用的数据构造方法论。

**与 LSLM 的演进关系**: 如果说 LSLM 回答了"SLM 能否边说边听"(answer: yes, via middle fusion),Audio-Interaction 回答了"LALM 能否在持续音频流中自主决定何时响应"(answer: yes, via chunk-level decision + 大规模流式训练)。但两者走了不同的技术路线——LSLM 追求并行双通道,Audio-Interaction 追求序列化决策。

## 可复用的 idea

1. **FIFO 异步推理调度**: 编码器和解码器完全解耦,通过时序队列通信。编码器是 pure producer,永不阻塞;解码器在 interruption point (silent/eos) 时 drain 全部积累 features。这个模式可迁移到任何需要低延迟流式推理的场景。

2. **层次化事件策划 + TFJP**: 用 LLM 规划语义一致的场景→事件序列,再检索/生成 clip 并通过 TFJP 平滑拼接。这套 pipeline 可用于构造任意流式音频训练数据,不限于交互场景。

3. **双目标训练 (L_LM + λ*L_stream)**: 给流控 token 单独加 loss,让模型既学"说什么"又学"何时说"。λ 的调节提供了 comprehension vs triggering 的显式权衡旋钮。

4. **Comprehension-aware silence**: 用 LLM agent 验证候选静音片段是否真的不需要响应(而非简单随机静音),构造高质量 negative samples 抑制 false triggering。

5. **单 head 流控发现 (L35H14)**: 流式模型的 silent/respond 决策集中在单一注意力头,暗示这类决策可被稀疏化或独立优化。

## 审阅

> [!review] 审阅 pass (0 high / 0 medium / 5 low)
> 审阅日期: 2026-06-08 | 审阅报告: [[_review/AudioInteractionModel-review.yml]]
> 5 low issues: 3x traceability-gap (KB创新判断/速查路线/局限性§6 出处标注), 1x template-compliance (datasets 未 wikilink), 1x fact-inference-mixing (点评段,无需修改)
