---
type: concept
title: "Turn-taking in Spoken Dialogue"
aliases: [轮次切换, Barge-in Handling, 打断处理, Turn Management, Backchannel, 回传信号, Interruption Handling, 对话交互管理]
category: "technique"
tags: [speech-LM, dialogue, turn-taking, interaction, barge-in, backchannel, full-duplex, real-time]
key_papers: ["dGSLM (Nguyen et al., 2023)", "Moshi (Defossez et al., 2024)", "VITA (Fu et al., 2024)", "Parrot (2024)", "Mini-Omni 2 (Xie & Wu, 2024)", "SyncLLM (2024)", "OmniFlatten (Zhang et al., 2024)", "Freeze-Omni (2024)", "CleanS2S (2024)", "Duplex Conversation (2024)", "TurnGPT (Ekstedt & Skantze, 2020)", "[[论文笔记/PersonaPlex|PersonaPlex (Roy et al., 2026)]]", "[[论文笔记/DialogueAgents|DialogueAgents]]", "[[论文笔记/ZipVoice-Dialog|ZipVoice-Dialog]]", "[[论文笔记/DialoSpeech|DialoSpeech]]", "[[论文笔记/ModeratorLM|ModeratorLM (Mitra et al., 2026)]]"]
origin_paper: "Ji et al., WavChat, 2024"
related_concepts: ["[[Full-duplexSpokenDialogue]]", "[[StreamingSpokenDialogue]]", "[[SpokenDialogueEvaluation]]", "[[SpeechLanguageModel]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Turn-taking in Spoken Dialogue 是指对话中说话者有序轮替发言的过程。在 spoken dialogue systems 中,这不仅包括传统的轮次切换,还涵盖打断 (barge-in/interruption)、回传信号 (backchannel) 和重叠 (overlap) 等复杂交互行为的识别与处理。

WavChat (Ji et al., 2024, Section 5.2.2) 将 turn-taking 分解为三个基本概念:

1. **Turn-taking cues**: 语音、节奏、呼吸、凝视或手势等信号,用于判断是否轮到自己说话或让出发言权
2. **Turn-end detection vs prediction**: detection 判断当前时刻是否应接管发言,prediction 预测未来何时轮次切换
3. **Overlap**: 当用户和系统的声音重叠时,分为两类:
   - **Interruption (打断)**: 用户意图接管发言权
   - **Backchannel (回传信号)**: 用户发出 "uh-huh"、"okay" 等听者反馈,无意接管

## 三类交互事件

WavChat 将人机对话中的交互归纳为三大类:

### 1. Interruptions (打断)
用户在系统说话时主动打断,系统需要:
- 检测到用户的打断意图
- 立即停止当前输出
- 切换到监听模式,理解用户新输入
- 生成针对新输入的响应

### 2. Backchannels (回传信号)
系统在用户说话时插入简短反馈以示"正在听":
- 系统发出 "uh-huh"、"I see"、"right" 等信号
- 鼓励用户继续说话
- 不构成轮次切换
- 模拟人类对话中的 active listening

### 3. Normal Turn Exchanges (正常轮次切换)
用户说完后,系统快速识别轮次结束并响应:
- 识别用户是否已完成发言 (vs 只是暂停思考)
- 避免过早打断用户 (premature interruption)
- 最小化响应延迟

## 五项交互能力

WavChat 定义了交互系统需具备的五项关键能力:

1. **Detecting User Interactions**: 当用户试图插话或提供新信息时,系统识别意图并立即停止输出
2. **Backchanneling During User Speech**: 在用户说话时提供 "uh-huh" 等简短确认,表明正在倾听
3. **Quickly Responding After User Completion**: 用户说完后,快速识别并无延迟地响应
4. **Handling Pauses in User Speech**: 当用户短暂停顿时,将其解读为思考而非邀请回应,避免过早打断
5. **Interrupting the User When Necessary**: 系统在检测到关键信息时主动打断用户提供即时反馈

## 级联系统中的实现

### Duplex Conversation
三个核心模块协作实现全双工对话:
- **User state detection**: 多模态模型 (音频+文本输入) 判断用户意图 — 轮次切换、继续说话、或犹豫,综合语音节奏、pitch、停顿等特征
- **Response signal selection**: 多标签分类,从大量真实对话中提取并训练合适的回传信号 (如 "uh-huh"、"right"),在适当时机插入
- **Interruption detection**: 端到端多模态检测模型,区分真实用户打断与背景噪声或无意的语音信号

### Full-duplex LLM (Wang et al.)
基于感知-动作-FSM 三模块架构:
- **Perception module**: 流式 ASR 模型,每 640ms 捕获并处理用户语音
- **Action module**: 流式 TTS 模型,将 LLM 生成的文本实时转换为音频,支持暂停/恢复播放
- **Neural FSM**: 核心控制器,允许 LLM 在 "speaking" 和 "listening" 状态间动态切换
- 实验结果: 响应时间在 500ms 内的对话占 50%+,用户打断处理准确率 54.7%,打断处理率 96.7%

### CleanS2S
结构化管线实现响应式交互:
- VAD 持续监控 → ASR 即时转写 → LLM 生成 → TTS 分段输出
- **打断感知设计**: VAD 检测到新用户输入时,LLM 暂停当前任务,ASR 转写新输入后生成更新响应
- **分段输出**: TTS 将长响应拆分为小段渐进发送,打断时可立即停止并切换

### VITA
双模型双工架构:
- 一个模型生成用户查询的文本响应
- 另一个模型持续监控环境输入
- 检测到新用户查询时,生成模型暂停,监控模型处理新查询
- 使用 state tokens 区分输入类型: query audio、background noise、text input
- Enhanced listening module 防止系统响应被用户反馈误触打断

## 端到端系统中的实现

### dGSLM
- **Dual-tower Dialogue Transformer Language Model (DLM)**: cross-attention 连接双通道
- 隐式建模 turn-taking: 自回归预测过程中自然产生沉默 token 和交替发言模式
- 训练于 2000 小时 Fisher 双通道电话对话数据
- 特别擅长建模 turn-taking 和 backchanneling 能力

### Moshi
- **Multi-stream architecture**: 同时建模 user 和 moshi 两路音频流,不需显式建模 speaker turns
- **Inner Monologue**: 在 moshi 音频流内联合建模 text 和 audio tokens
- 训练数据包含 overlapping speech、noise、interruptions 等真实场景
- 理论延迟 160ms,实际约 230ms

### Parrot
- **Dual-channel audio**: 每个通道代表一个说话者,独立管理对话双方
- **Next-token-pair prediction**: 同时预测双通道 token,协调处理重叠语音和轮次切换
- 支持 streaming input: 一个通道处理用户音频,另一个通道生成响应
- 直接处理音频,无需中间文本转换

### Mini-Omni2
- **irq/n-irq 状态标记**: 训练数据中在打断命令 ("Stop Omni") 插入点标记 irq (interrupt) 和 n-irq (non-interrupt)
- 模型在推理时实时生成 irq/n-irq 标记,决定是否停止输出
- 训练数据包含各种噪声环境下的打断命令,增强鲁棒性
- 基于指令设计的打断机制,limited instruction approach

### [[论文笔记/LSLM|LSLM]]
- **IRQ token**: 在词表中添加 IRQ (interruption) 特殊 token,训练时在打断发生 0.5s 后标注 IRQ
- 使用 streaming vq-wav2vec SSL encoder 实时编码监听通道输入
- **Middle fusion**: 监听信号在每个 Transformer block 注入,优于 early/late fusion
- Command-based FDM: F1 98.00% (clean); Voice-based FDM: F1 95.50% (clean) [Table 2, 3]
- 与 Mini-Omni2 irq/n-irq 的区别: LSLM 仅用 IRQ 一个 token (打断/不打断),Mini-Omni2 用 irq+n-irq 两个标记

### SyncLLM
- **Time synchronization**: 将音频流分割为固定大小 chunks,每个 chunk 对应特定时间间隔
- 在每个时间步交替生成 user 和 system speech segments
- 预测用户下一步语音后再生成系统 chunk,实现时间同步
- 使用去重 HuBERT token + 周期性同步标记,支持 chunk-level 实时推理

### OmniFlatten
- **Progressive training** 实现全双工:
  1. Stage 1: 半双工对话训练
  2. Stage 2: 去除用户 text stream,支持实时预测
  3. Stage 3: 去除助手 text stream,实现纯语音流生成
- **Block-by-block generation**: 将输入输出语音序列分割为固定大小 blocks,逐块处理

### Freeze-Omni
- **Chunk-level state prediction**: 分类层对每个音频 chunk 预测对话状态:
  - **State 0**: 用户尚未说完,继续监听
  - **State 1**: 模型打断提供快速确认或反馈
  - **State 2**: 用户输入完成,模型准备生成完整响应
- 自然支持 turn-taking 和 backchannel 行为

### Raon-SpeechChat (KRAFTON, 2026)
- **SIL/BOW/BC 三状态建模**: 将全双工交互行为分解为三种显式特殊 token,区别于 Moshi 的单一 PAD token 和 Freeze-Omni 的 State 0/1/2:
  - **SIL** (silence): 显式编码沉默监听,让模型区分"主动沉默"和"说话中的填充"
  - **BOW** (beginning of word): 在每个 assistant text token 前发出,分离 when-to-speak 和 what-to-say
  - **BC** (backchannel): 专用于回传信号,推理时可独立控制回传频率甚至完全禁用
- FDB v1.0: interruption TOR 0.980, backchannel TOR 0.091 (均为 best); user backchannel resume 率 0.398 (弱于 MiniCPM-o 0.520)
- 详见 [[论文笔记/Raon-Speech|Raon-Speech]]

### BayLing-Duplex (Fang et al., 2026)
- 4 个对话状态 token ([SILENCE]/[ASSISTANT]/[PAD]/[EPAD]) 将 turn-taking 和打断决策完全归结为 next-token prediction,无辅助分类头或状态机
- 与 Raon-SpeechChat 的 SIL/BOW/BC 同属单序列状态 token 方案,但更极简 (无 backchannel token),且使用 DPO 优化 timing 精度
- SFT 中 token 权重调整关键: ω_sil=0.1, ω_role=10,否则 TT SR@3s 仅 60.3% (模型几乎永远沉默)
- DPO pair 仅改 timing 不改 content: turn-taking 正例 gap 0.8s / 负例 Uniform(2,5)s; interruption 正例 δ_react~Uniform(0.8,2.0)s / 负例 Uniform(3,5)s
- TT SR@3s 92.0%, ISR@2s 100% [Table 2, InstructS2S-Eval]
- 详见 [[论文笔记/BayLing-Duplex|BayLing-Duplex]]

### ELLSA (Wang et al., 2026)
- 首个将 turn-taking 概念从纯语音扩展到四模态 (speech+vision+text+action) 的系统
- **Action turn-taking**: 模型自行判断何时开始执行动作 (收到语音指令后),LIBERO 上成功率 96.4-100%
- **Action barge-in**: 在执行动作过程中收到中断命令时,模型输出 "Action Cancelled" 并停止动作,成功率 94.3%
- 同时保持 dialogue turn-taking 100% 成功率 (优于 Moshi 37-85%, Freeze-Omni 72-99.8%)
- 1 秒 time block 设计简化了 turn-taking 学习,但引入了较高的最小延迟
- 详见 [[论文笔记/ELLSA|ELLSA]]

### State Inertia 分析 (Chang et al., 2026)

[[论文笔记/StateInertia-FD-SLM|State Inertia (Chang et al., 2026)]] 提供了首个从 mechanistic interpretability 角度对 FD-SLM 打断处理的内部机制分析。与上述所有方法关注"训练时如何学习 turn-taking"不同,本文揭示了一个推理时的问题: 即使训练好的 FD-SLM (PersonaPlex, Moshi, Raon-SpeechChat) 在用户打断时仍存在 **state inertia** — 内部隐藏表征延迟约 7-8 timestep (~0.6s) 才从 generative 切换到 perceptive state [§3.4]。提出 training-free 的 activation steering 方案: 在打断 onset 注入 perception vector (从对比 timestep 的 mean diff 构造),无需微调即可恢复 81-94% 的打断造成的性能损失 [Table 2]。这表明训练时方案 (IRQ token, SIL/BOW/BC 等) 与推理时干预可以互补。

## VAD 的局限

最早期的全双工系统使用 Voice Activity Detection (VAD) 判断用户是否有打断意图。然而 VAD 存在根本性局限:
- 无法区分有意打断和无意声音 (如 backchannel "uh-huh")
- 导致频繁误判打断,引入明显延迟
- 无法处理 discontinuous expression (如用户在提及人名/地名时犹豫)

现代系统转向更复杂的方法: 多模态检测模型、语义分析 (FSM)、端到端学习 (irq/n-irq markers)、chunk-level state prediction 等。

## 关键论文

- dGSLM (Nguyen et al., 2023): dual-tower DLM, 首个隐式建模 turn-taking 的 E2E 系统
- Moshi (Defossez et al., 2024): multi-stream 无显式 turn 建模, Inner Monologue
- VITA (Fu et al., 2024): dual-model 双工 + state tokens + enhanced listening
- Parrot (2024): dual-channel + next-token-pair prediction
- Mini-Omni 2 (Xie & Wu, 2024): irq/n-irq 标记的打断机制
- SyncLLM (2024): time-synchronized chunk processing
- OmniFlatten (Zhang et al., 2024): progressive training for full-duplex
- Freeze-Omni (2024): chunk-level state prediction (State 0/1/2)
- TurnGPT (Ekstedt & Skantze, 2020): transformer-based turn-taking prediction

## 相关概念

- [[Full-duplexSpokenDialogue]]: Turn-taking 是全双工对话的核心交互机制
- [[StreamingSpokenDialogue]]: 流式处理是实时 turn-taking 的架构基础
- [[SpokenDialogueEvaluation]]: Interaction Capability 评估涵盖 turn-taking 质量
- [[SpeechLanguageModel]]: Turn-taking 是 SpeechLM 的高级交互能力

## 演进

VAD-only 打断检测 (早期, 高误判) → Duplex Conversation 三模块 (多模态检测, 2024) → Full-duplex LLM 感知-动作-FSM (2024) → dGSLM 隐式 turn-taking (dual-tower DLM, 2023) → Moshi multi-stream (无显式 turn, 单一 PAD token, 2024) → Mini-Omni2 irq/n-irq markers (2024) → SyncLLM time-sync chunks (2024) → Freeze-Omni chunk-level state prediction (State 0/1/2, 2024) → Raon-SpeechChat SIL/BOW/BC 三状态建模 (显式解耦 when-to-speak/what-to-say/backchannel, 2026) / BayLing-Duplex 4 状态 token + timing-only DPO (2026) → ELLSA action turn-taking + action barge-in (四模态扩展, 2026) → [[论文笔记/ModeratorLM|ModeratorLM]] role-conditioned multi-party turn-taking + CoT reasoning (2026) → [[论文笔记/Wan-Streamer|Wan-Streamer]] 从交错交互数据隐式学习 turn-taking (含视觉打断感知和主动发言, 无显式状态 token, 2026; 注: 无定量 turn-taking 评估)
