---
type: concept
title: "Streaming Spoken Dialogue"
aliases: [流式语音对话, Streaming Speech Interaction, 实时语音处理, Real-time Speech Processing, Streaming Inference for Speech, 流式推理]
category: "technique"
tags: [speech-LM, streaming, real-time, causal, latency, dialogue, inference]
key_papers: ["[[论文笔记/Moshi|Moshi]]", "Mini-Omni (Xie & Wu, 2024)", "LLaMA-Omni (Fang et al., 2024)", "IntrinsicVoice (2024)", "OmniFlatten (Zhang et al., 2024)", "Freeze-Omni (2024)", "SyncLLM (2024)", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/Step-Audio 2.5|StepAudio 2.5]]", "[[论文笔记/STITCH|STITCH (Chiang et al., ICLR 2026)]]", "[[论文笔记/LiveSpeech 2|LiveSpeech 2]]", "[[论文笔记/LLMVoX|LLMVoX]]", "[[论文笔记/OpenOmni|OpenOmni]]", "[[论文笔记/LLaMA-Omni 2|LLaMA-Omni 2]]", "[[论文笔记/OpenS2S|OpenS2S]]"]
origin_paper: "Ji et al., WavChat, 2024"
related_concepts: ["[[Full-duplex Spoken Dialogue]]", "[[Turn-taking in Spoken Dialogue]]", "[[Speech Language Model]]", "[[Neural Vocoder]]", "[[Spoken Dialogue Evaluation]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Streaming Spoken Dialogue 是 spoken dialogue systems 中支持实时、连续处理的架构范式。与传统 batch 处理 (等待完整输入后处理) 不同,streaming 系统在接收输入的同时逐步处理并生成输出,无需等待完整信号。

WavChat (Ji et al., 2024, Section 5.1) 将 streaming 能力分解为两个维度:
1. **Streaming Understanding**: 在用户说话的同时处理音频输入,不必等待用户说完
2. **Streaming Generation**: 在处理过程中逐步生成输出,不必等待所有中间状态计算完毕

这两种能力共同使系统能流畅地进行实时交互。

## 三项核心技术

WavChat 将 E2E streaming spoken dialogue 的设计归结为三项核心技术:

### 1. Causal Convolution (因果卷积)

因果卷积确保当前输出仅依赖当前和过去的输入,不访问未来信息:
- 标准卷积: 核大小 k 时,访问 (t-k/2) 到 (t+k/2) 的数据
- 因果卷积: 在左侧 padding k-1 个零,使核仅访问 t-k+1 到 t 的数据

**Dilated causal convolution** (膨胀因果卷积): 在核内插入零间隔,扩大感受野而不增加延迟,捕获更长距离的依赖关系。

**在 streaming spoken dialogue 中的作用**:
- 确保实时处理: 无需访问未来帧即可计算输出
- 降低延迟: 不需缓冲未来输入

**代表应用**: Moshi 的 Mimi codec 的 encoder 和 decoder 均采用因果卷积,整个模型 (codec + transformer + attention) 全部建立在因果结构上。

### 2. Causal Attention (因果注意力)

通过下三角 mask 确保每个位置只能 attend 到之前的位置:
- 将注意力矩阵中未来位置对应的值设为负无穷
- 每个 token 的预测仅基于当前和历史信息

**与 full attention 的区别**: Full attention 需要访问完整序列,不适用于实时处理;causal attention 可增量操作,新输入到达时立即处理。

### 3. Queue Management (队列管理)

音频流被分割为帧后,通过队列系统实现有序、实时处理:
- 音频帧入队后按序处理
- 确保处理速度匹配或快于输入速度
- 支持动态负载平衡

## E2E Streaming 系统实现

### Mini-Omni: Delayed Parallel Decoding
- 选择 SNAC codec,单秒音频产生数百 tokens (7 层)
- **Layer-by-layer delayed parallel generation**: 在每步同时生成 text token 和多层 audio tokens,各层延迟一步
- 灵感来自 MusicGen 的 delay pattern,将原本顺序解码的 multi-layer tokens 并行生成
- **Batch Parallel Decoding**: 两个样本并行推理,第一个样本的 text output 嵌入第二个样本对应位置,增强推理能力
- 实现 text + speech 同步流式输出

### LLaMA-Omni: NAR CTC Decoder
- 在 LLM 之后添加 non-autoregressive streaming speech decoder
- 使用 LLM 的 hidden states 作为输入,通过 CTC loss 训练
- 引入 upsample factor λ 处理输入输出的变长映射
- 设定预定义 chunk size 进一步实现 vocoder 流式合成
- 同时生成 text 和 speech 响应

### IntrinsicVoice: GroupFormer
- HuBERT encoder + KMeans 量化器将语音离散化
- 通过 group partition 将 tokens 组织为 grouped token sequence
- **GroupFormer**: 非自回归 transformer encoder,一步预测一组 speech tokens
- 减小 speech-text 序列长度差距,加速推理
- 但输入侧不完全支持 streaming

### Moshi: Fully Causal Architecture
- 参考 SpeechTokenizer 架构训练全因果 streaming codec (Mimi)
- Encoder、decoder、transformer、attention 全部因果
- **RQ-Transformer**: Temporal Transformer (大) + Depth Transformer (小)
  - Temporal Transformer 处理时间轴上的 K*S flattened sequence
  - Depth Transformer 处理每个时间步内的 K 层 codebook tokens
  - Depth Transformer 体积小,子序列生成近似并行
- 理论延迟 160ms,实际约 230ms

### OmniFlatten: Block-by-Block Processing
- 将输入输出语音序列分割为固定大小 blocks
- 逐块处理每个 segment,实现 streaming processing
- 配合 progressive training 实现从半双工到全双工的渐进
- 有效降低轮次切换延迟

### Freeze-Omni: Chunk-wise Streaming
- 使用 chunk-wise streaming speech encoder 将语音特征实时转换为高维表征
- Adapter 将高维表征映射到 LLM embedding space
- Downsampling layers 降低语音帧率,减少 prefill 阶段处理时间,最小化延迟
- Speech encoder 包含少量 downsampling convolutional layers + Transformer blocks

## 级联系统中的 Streaming

级联系统的 streaming 依赖外部组件:
- **Streaming ASR**: U2++ Conformer 等模型支持增量语音识别
- **Streaming TTS**: XTTS-v2 等模型支持增量语音合成

实验表明 (WavChat 引用): streaming ASR (U2++ Conformer) + streaming TTS (XTTS-v2) 的组合实现了最低延迟,且显著提升了交互打断能力。非 streaming 的 Whisper + VITS 组合延迟明显更高。

## 延迟优化技术汇总

| 技术 | 系统 | 机制 | 延迟效果 |
|------|------|------|----------|
| Delayed parallel decoding | Mini-Omni | 多层 token 延迟一步并行生成 | 显著减少解码步数 |
| Batch parallel decoding | Mini-Omni | 两样本并行,text→speech 转移 | 增强推理+降延迟 |
| NAR CTC decoder | LLaMA-Omni | 非自回归 + CTC loss | 同步 text+speech |
| GroupFormer | IntrinsicVoice | 一步预测一组 tokens | 减小序列长度差 |
| RQ-Transformer | Moshi | Temporal+Depth 分解 | 理论 160ms |
| Block-by-block | OmniFlatten | 固定大小块逐块处理 | 降低轮次延迟 |
| Chunk-wise encoder | Freeze-Omni | Downsampling + 分块处理 | 最小化 prefill 延迟 |
| Time sync chunks | SyncLLM | 固定时间间隔分块 | 实时同步 |
| Streaming ASR+TTS | 级联系统 | U2++ + XTTS-v2 | 组合最低延迟 |
| AR TTS LM + Read-Write | LLaMA-Omni 2 | CosyVoice 2 式 R:W 交替 + gate fusion | ~583ms (R=3,W=10) |

## Text-guided vs W/o Text-guided 的延迟 Trade-off

WavChat (Section 3.3.4 & 4.3) 讨论了一个关键延迟权衡:

**Text-guided generation** (如 SpeechGPT, EMOVA):
- 先生成完整文本响应,再基于文本生成语音
- **优势**: chain-of-thought 提高生成质量
- **劣势**: 必须等文本完成才能开始语音生成,延迟高,不兼容实时交互

**Parallel text+speech generation** (如 Moshi, Mini-Omni, LLaMA-Omni):
- 同时输出 text 和 speech tokens
- **优势**: 显著降低延迟
- **劣势**: 可能牺牲一定的响应质量

**W/o text generation** (如 IntrinsicVoice, SyncLLM):
- 直接 speech-to-speech,不依赖文本中间表示
- **优势**: 最低延迟
- **劣势**: 模型复杂度增加,推理难度更大

## 关键论文

- Moshi (Defossez et al., 2024): 全因果架构, RQ-Transformer, 160ms 理论延迟
- Mini-Omni (Xie & Wu, 2024): delayed parallel decoding, batch parallel decoding
- LLaMA-Omni (Fang et al., 2024): NAR CTC streaming decoder
- IntrinsicVoice (2024): GroupFormer 一步多 token 预测
- OmniFlatten (Zhang et al., 2024): block-by-block streaming + progressive training
- Freeze-Omni (2024): chunk-wise streaming encoder
- SyncLLM (2024): time-synchronized chunk processing

## 相关概念

- [[Full-duplex Spoken Dialogue]]: Streaming 是全双工的架构前提
- [[Turn-taking in Spoken Dialogue]]: 低延迟 streaming 使自然 turn-taking 成为可能
- [[Neural Vocoder]]: streaming vocoder 是输出侧 streaming 的最后一环
- [[Speech Language Model]]: Streaming 是 SpeechLM 实时交互范式的关键

## 演进

Batch processing (完整输入→完整输出) → Streaming ASR+TTS 组合 (级联系统, 2023) → Chain-of-modality text→speech (SpeechGPT, 高延迟, 2023) → Delayed parallel decoding (Mini-Omni, 2024) → NAR CTC decoder (LLaMA-Omni, 2024) → Fully causal architecture + RQ-Transformer (Moshi, 160ms, 2024) → GroupFormer (IntrinsicVoice, 2024) → Block-by-block + time-sync (OmniFlatten/SyncLLM, 2024) → Chunk-wise streaming (Freeze-Omni, 2024)
