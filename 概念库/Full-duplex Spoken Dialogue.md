---
type: concept
title: "Full-duplex Spoken Dialogue"
aliases: [全双工口语对话, Full-duplex Speech Interaction, Real-time Speech Interaction, 实时语音交互, Duplex Conversation, 全双工语音]
category: "technique"
tags: [speech-LM, dialogue, real-time, full-duplex, turn-taking, streaming, interaction]
key_papers: ["dGSLM (Nguyen et al., 2023)", "Moshi (Defossez et al., 2024)", "VITA (Fu et al., 2024)", "NTPP (Wang et al., 2025)", "LSLM (Ma et al., 2024)", "Mini-Omni 2 (Xie & Wu, 2024)", "MiniCPM-o 2.6 (OpenBMB, 2024)", "FlexDuo (Liao et al., 2025)", "OmniFlatten (Zhang et al., 2024)", "SALMONN-omni (Wu et al., 2024)", "SyncLLM (2024)", "Parrot (2024)", "Freeze-Omni (2024)", "CleanS2S (2024)"]
origin_paper: "Cui et al., Speech Language Models, 2024"
related_concepts: ["[[Speech Language Model]]", "[[Speech-Text Alignment]]", "[[Audio Understanding]]", "[[Turn-taking in Spoken Dialogue]]", "[[Streaming Spoken Dialogue]]", "[[Spoken Dialogue Evaluation]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Full-duplex Spoken Dialogue 是 Speech Language Model 的前沿交互范式,支持同时双向语音通信,即模型可以在用户说话的同时生成响应,并处理打断和轮次切换。这模拟了人类自然对话中的动态交互模式。

Survey (Cui et al., 2024) 定义: "Full-duplex modeling allows SpeechLMs to support simultaneous bidirectional communication -- specifically, the ability to handle interruptions initiated by either the user or the model."

**两大核心能力**:
1. **User interruption (用户打断)**: 用户可在模型说话时打断,模型立即响应新指令
2. **Simultaneous response (同步响应)**: 模型可在处理用户输入的同时生成输出

## 从传统到全双工的渐进演进

Survey 将语音交互范式划分为三个渐进阶段:

### Stage 1: Traditional Generation
接收完整输入序列后生成完整响应。大多数 SpeechLM 默认采用此范式 (TWIST, SPIRIT-LM, AudioPaLM, SpeechGPT)。

### Stage 2: Streaming / Real-time Interaction
采用 **streaming tokenizers and vocoders**,无需等待完整语音编码即可开始处理:
- 消除编码等待延迟
- 支持即时低延迟响应
- 但仍是 half-duplex (半双工): 要么听要么说

### Stage 3: Full-duplex Modeling
同时处理输入和生成输出,支持打断和同步:
- 需要联合建模用户和模型的音频流
- 需要学习何时说话、何时保持沉默

## 代表系统

### dGSLM (Nguyen et al., 2023)
- **首个全双工 SpeechLM**
- 为两个说话者各分配一个 transformer,用 cross-attention 层捕获说话者间交互
- Dialogue transformer language model (DLM) 联合建模两人对话数据
- 使用 HuBERT 语义 tokens

### NTPP (Wang et al., 2025)
- "Next-token-pair prediction": decoder-only transformer 同时预测双通道 tokens
- 用 VQ-VAE tokens 建模双通道口语对话
- 相比 dGSLM 更简洁的单模型架构

### Moshi (Defossez et al., 2024)
- 将用户输入和模型响应通道数据拼接,用 **RQ-Transformer** 统一处理
- 同时生成一个 text token 序列、一个 semantic token 序列和七个 acoustic token 序列
- 使用 Mimi (mixed tokenizer) 作为 speech tokenizer
- 实现真正的实时全双工对话

### LSLM (Ma et al., 2024)
- "Language model can listen while speaking"
- 使用 decoder-only transformer 建模一个说话者的语音
- 整合 streaming self-supervised learning (SSL) encoder 实现边说边听
- 在讲话的同时编码并理解用户输入

### Mini-Omni 2 (Xie & Wu, 2024)
- 基于 Qwen2 + Whisper encoder
- 同时生成 text + 7 acoustic streams
- 支持 vision + speech + text 多模态的双工交互

### OmniFlatten (Zhang et al., 2024)
- 端到端 GPT 模型用于无缝语音对话
- 将多模态信息展平为统一序列

### SALMONN-omni (Wu et al., 2024)
- Codec-free 的全双工 LLM
- 直接在语音理解和生成任务间无缝切换

## Interactive Period Recognition (IPR)

Survey 特别提出 IPR 作为全双工的配套能力:

**定义**: 识别用户是否正在与模型交互,从而决定是否应生成响应。

**为什么重要**: 模型需要区分用户是在和自己说话还是和别人说话,以及何时应保持沉默。

**实现方法**:
- **VITA**: 训练模型区分 query speech 和 non-query audio,学习在非查询时输出 end-of-sequence token
- **MiniCPM-o 2.6**: 整合 VAD 模块,低于阈值的输入被忽略
- **FlexDuo**: 可插拔的全双工系统,分离 listening 和 speaking 能力

## 评估方法

全双工系统的评估关注轮次切换的自然度:
- **Turn-taking events**: Inter-Pausal Unit (IPU), pause, gap, overlap 的统计分布
- **dGSLM 方法**: 比较生成语音中 turn-taking 事件的统计分布与人类对话
- **Full Duplex Bench**: 专门评估全双工 SpeechLM 轮次切换能力的 benchmark
- **Talking Turns**: 训练神经网络预测全双工输出的 turn-taking 事件
- **NTPP 方法**: reflective pause (沉默能力) + interruption (被打断停止能力)

## 挑战与未来

Survey (Section VII-C) 指出:
1. **实时语音生成仍未充分探索**: 多数 vocoder 需等待完整 token 序列,造成延迟
2. **Streaming pipeline**: 需要 speech input/output 均可分块处理和生成
3. **自主波形生成**: SpeechLM 直接生成音频样本而非依赖外部 vocoder

## WavChat 补充: 更多全双工系统 (Ji et al., 2024)

WavChat survey 进一步梳理了全双工系统的更多实现:

### SyncLLM (2024)
- 自回归 transformer decoder 集成时间同步: 将语音单元与真实时钟对齐
- 预测双方交错 speech tokens,维持 timing + speaker tags
- 使用去重 HuBERT tokens 增强语义保真度,同时管理延迟
- 插值重构 token 序列以适配预期结构,实现无缝语音合成

### Parrot (2024)
- 双通道音频设置: 每个通道代表一个说话者
- "Next-token-pair prediction" 机制同时预测双通道 tokens
- 支持 streaming input: 一通道持续处理用户音频,另一通道生成响应
- 直接处理音频,无需中间文本转换,高响应性

### Freeze-Omni (2024)
- 核心: 将 ASR/TTS 功能转移到 encoder 和 decoder,而非赋予 LLM
- Chunk-level state prediction: 分类层对每个音频 chunk 预测 State 0 (继续听) / State 1 (插入反馈) / State 2 (开始响应)
- 3-stage training: 模态对齐 → 半双工对话 → 全双工
- 使用 text-speech paired data 获得 speech-to-speech 对话能力

### CleanS2S (2024)
- 结构化级联管线: VAD → ASR → LLM → TTS
- 打断感知: VAD 检测新输入时 LLM 暂停,ASR 转写后生成更新响应
- TTS 分段输出: 长响应拆分为小段,打断时可立即停止切换

## 关键论文

- dGSLM (Nguyen et al., 2023): 首个全双工对话模型, dual transformer + cross-attention
- Moshi (Defossez et al., 2024): RQ-Transformer 全双工, Mimi tokenizer, Inner Monologue
- LSLM (Ma et al., 2024): 边说边听, streaming SSL encoder
- VITA (Fu et al., 2024): IPR + 多模态全双工, dual-model 架构
- NTPP (Wang et al., 2025): next-token-pair prediction 双通道对话
- Mini-Omni 2 (Xie & Wu, 2024): vision+speech+text 全双工, irq/n-irq 打断标记
- FlexDuo (Liao et al., 2025): 可插拔全双工系统
- SyncLLM (2024): time-synchronized 全双工
- Parrot (2024): dual-channel + next-token-pair
- Freeze-Omni (2024): chunk-level state prediction
- CleanS2S (2024): 结构化级联全双工

## 相关概念

- [[Speech Language Model]]: 全双工是 SpeechLM 的前沿交互范式
- [[Speech-Text Alignment]]: 全双工中需要对齐 text-present vs text-independent 推理
- [[Audio Understanding]]: 全双工模型同时需要理解和生成能力
- [[Turn-taking in Spoken Dialogue]]: 全双工中的轮次切换、打断和回传信号机制
- [[Streaming Spoken Dialogue]]: 流式处理是全双工的架构前提
- [[Spoken Dialogue Evaluation]]: 全双工系统的交互能力评估

## 演进

Traditional (完整输入→完整输出) → Streaming (低延迟, 2023) → dGSLM (首个全双工, 双 transformer, 2023) → NTPP (单模型 token-pair, 2024) → Moshi (RQ-Transformer 全双工, 2024) → LSLM (边说边听, 2024) → VITA/MiniCPM-o (IPR, 多模态, 2024) → FlexDuo (可插拔, 2025)
