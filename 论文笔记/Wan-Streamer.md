---
type: paper
tier: deep
title: "Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models"
arxiv_id: "2606.25041"
source: "Sources/Wan-Streamer.pdf"
authors: [Wan Team, Alibaba Group]
year: 2026
venue: "arXiv"
tags: [multimodal, full-duplex, streaming, real-time, interactive, video-generation, audio-visual, foundation-model, end-to-end, flow-matching, causal]
concepts: ["[[ConditionalFlowMatching]]", "[[Full-duplexSpokenDialogue]]", "[[StreamingSpokenDialogue]]", "[[Classifier-FreeGuidance]]", "[[Turn-takinginSpokenDialogue]]", "[[DiffusionModel]]", "[[VariationalAutoencoderforTTS]]"]
models: ["Wan-Streamer", "Moshi", "GPT-4o", "Qwen3-Omni", "Qwen3.5-Omni", "MiniCPM-o 4.5", "X-Streamer", "MIDAS", "LPM 1.0", "VASA-1", "OmniForcing", "LiveTalk", "StreamAvatar", "Hallo-Live", "TalkingMachines", "AvatarForcing"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-26
updated: 2026-06-26
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[Full-duplexSpokenDialogue]]✓, [[StreamingSpokenDialogue]]✓, [[ConditionalFlowMatching]]✓, [[Classifier-FreeGuidance]]✓, [[Turn-takinginSpokenDialogue]]✓, [[DiffusionModel]]✓)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Wan-Streamer 属于全双工交互模型的最新演进,但其关键区别在于将**视觉生成**纳入全双工循环。KB 中已有的 Full-duplex Spoken Dialogue 页面追踪了从 dGSLM → Moshi → LSLM → ELLSA 的演进,但这些系统要么仅处理语音(Moshi, LSLM, BayLing-Duplex),要么在视觉侧使用独立模块(ELLSA 的 SA-MoE 仍是双 expert 架构)。Wan-Streamer 声称用单一 Transformer 统一 text/audio/video 的输入输出,这是**首个将视频输入+视频生成完全融入因果流中的全双工基础模型**。在 Streaming Spoken Dialogue 页面中,Moshi 的 RQ-Transformer 实现了 160ms 理论延迟,Wan-Streamer 的 200ms model-side latency 在同一量级但增加了视频生成维度。CFM 页面记录了 flow matching 在 TTS 中的广泛应用,而 Wan-Streamer 将 flow matching 用于联合音视频 latent 生成(而非单独的 mel spectrogram 或 waveform),这是 CFM 的一个新应用场景。CFG 被用于 teacher 训练后通过蒸馏吸收进 student,与 KB 中记录的 DSFlow CFG 内化现象一致。
>
> **已有认知**: 全双工系统已从简单的 VAD-based 打断发展到端到端学习;流式架构的核心是因果卷积+因果注意力;flow matching 比 diffusion 推理步数更少。
>
> **创新判断**: Wan-Streamer 的主要创新不在于单项技术突破,而在于**系统级的因果流设计** -- 将音视频感知、推理、语音生成、视频生成统一到一个因果流中,并通过 thinker-performer 分离实现实时部署。这与 ELLSA 的四模态方向一致但走得更远(视频生成而非动作执行)。
>
> 检索命中: [[Full-duplexSpokenDialogue]], [[StreamingSpokenDialogue]], [[ConditionalFlowMatching]], [[Classifier-FreeGuidance]], [[Turn-takinginSpokenDialogue]], [[DiffusionModel]] | 过滤: 均为 pending-review[待确认] (CFM 为 confirmed) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个将 text/audio/video 输入输出统一在单一因果 Transformer 中,实现端到端实时全双工音视频交互的基础模型
> - **路线**: User audio/video → causal encoders → interleaved token sequence (text+audio+video latents) → single Transformer (block-causal attention) → flow matching solver (audio+video latents) → causal decoders → synchronized speech+video output
> - **指标**: ~200ms model-side latency, ~550ms total interaction latency (含 350ms 网络), 25 FPS video output, 160ms streaming unit [§2.4, Tab 1]; 192p 输出分辨率 (proof of concept)
> - **可借鉴**: (1) Thinker-performer 双 GPU 流水线设计将感知/状态更新与昂贵的 flow matching 生成解耦,通过 KV-cache 交换保持统一模型语义; (2) Rolling distillation + self-forcing 策略解决 autoregressive 长时序生成的 train-test mismatch; (3) 将 CFG 通过蒸馏吸收进 student 减少推理开销
> - **局限**: 仅 192p 输出分辨率 (proof of concept); 无定量生成质量评估 (无 MOS/FID/FVD/SIM); 训练数据混合和规模未披露; 无开源; 无 ablation 实验

## 核心问题

本文要解决的核心问题是: **如何构建一个原生流式 (native-streaming)、端到端的多模态交互基础模型,使其能够以亚秒级延迟实现全双工的音视频交互?**

现有系统的根本局限有三 [§1]:
1. **级联延迟**: 分离的 VAD → ASR → LLM → TTS → 动画/视频生成模块在每个边界引入等待时间
2. **不对称交互**: 多数系统要么只接收音视频但只输出文本/语音,要么生成音视频但依赖外部对话模块
3. **后验对齐**: 独立训练的组件无法端到端学习响应时机、轮次管理、身份保持和跨模态同步

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Wan-Streamer 将交互建模为一个连续的因果流 [§2.1]。在第 k 个 streaming unit:
- 用户观测 $u_k = (u_k^t, u_k^a, u_k^v)$ (text, audio, video)
- 模型响应 $y_k = (y_k^t, y_k^a, y_k^v)$ (text, audio, video)
- 联合概率分解为因果条件: $p_\theta(y_{1:K}|u_{1:K}) = \prod_{k=1}^K p_\theta(y_k^t, y_k^a, y_k^v | u_{\leq k}, y_{<k})$ [Eq. 1]

**模态表示分离** [§2.1]:
- **语言**: 离散 token,使用 next-token prediction + cross-entropy loss
- **音频+视频**: 连续 latent 空间,使用 conditional flow matching 联合生成

flow matching 的具体形式 [§2.1, Eq. 2-3]:
- 从 clean target latent $z_0^m$ 到 Gaussian noise $\epsilon^m$ 构造插值: $z_\tau^m = (1-\tau)z_0^m + \tau\epsilon^m$
- 训练目标: 估计 velocity field $f_\theta(z_\tau^a, z_\tau^v, c_k, \tau)$,其中 $c_k$ 为 clean streaming context (已确认的历史 latent + 当前用户观测)
- **关键**: 同一 clean context 同时条件化音频和视频的 velocity prediction,使语音、动作、外观作为耦合响应被优化 [论文原文]

### 关键设计选择

**1. 为什么选择 block-causal attention 而非标准 causal attention?**

论文将全局因果注意力组织为 block-causal 形式 [§2.1],每个 block 对应一个 160ms 的 streaming unit。[agent 解读] 标准 token-level causal attention 对视频帧序列会产生极长的序列和注意力计算;block-causal 允许在 block 内部进行局部交互(如同一时间步的音频-视频 latent 之间),同时保持跨 block 的因果性,实现流式增量处理。

**2. 为什么全栈因果化 (causal encoders + causal decoders + causal VAE)?**

论文强调"streamability is a modeling constraint rather than a serving optimization" [§1]。[论文原文] 如果编码器使用 bidirectional attention/convolution,就需要缓冲未来帧,破坏流式处理;如果解码器不因果,就无法在生成过程中增量输出。因此 Wan-Streamer 专门设计了 strictly causal audio/video VAE、causal audio-visual encoders 和 causal audio/video decoders [§2.1]。

**3. 为什么音视频 latent 要在 Transformer 中联合生成?**

[论文原文] 将音频和视频 latent 作为 coupled response 在同一 clean context 下联合生成,使得"lip motion, facial dynamics, and prosody are synchronized natively rather than repaired by post-hoc alignment" [§3]。这与级联系统先生成语音再驱动唇形动画的方式形成对比。

**4. 为什么不使用离散 token 表示音视频?**

[⚠️ 论文未详述] 论文选择连续 latent + flow matching 而非 RVQ/VQ 离散 token + language model (如 Moshi 的 RQ-Transformer 方式),但未给出明确的比较或理由。[agent 解读] 可能原因: (a) 视频 latent 的信息密度远高于语音,离散化损失更大; (b) 连续 latent 可通过 flow matching 的 ODE solver 实现更灵活的质量-速度 trade-off; (c) 音视频耦合生成在连续空间更自然。

### 训练策略

三阶段训练 [§2.3]:

**Stage 1: Independent-task Pretraining**
- 从 language model 初始化 Transformer [§2.3, 引用 Qwen2.5/Qwen3]
- 混合训练: understanding (image/audio/video understanding, ASR, TTS, dialogue) + generation (image/audio/video/AV generation)
- [论文原文] 感知、语言推理和 latent 生成在一个序列模型中对齐,而非作为独立模块优化 [§2.3]

**Stage 2: End-to-end Interaction Training**
- 在全双工交互数据上训练: user text/audio/video 与 agent text/audio/video 交错在同一因果流中 [§2.3]
- [论文原文] 响应时机、主动倾听行为、打断处理和长上下文一致性在推理时使用的相同因果格式下学习 [§2.3]

**Stage 3: Distillation for Low-latency Streaming**
- Teacher: 使用 CFG + 更多 flow matching solver steps [§2.3]
- Student: 通过蒸馏吸收 CFG 效果并减少 solver steps
- **Rolling distillation**: student 在连续 streaming units 上 rollout,使用 self-forcing 策略 [引用 Self-Forcing, Huang et al.] + distribution matching [引用 DMD, Yin et al.] 在 realistic rollout 条件下对齐 student 与 teacher
- [论文原文] 这显著减少 train-test mismatch 并改善 long-form generation quality [§2.3]

### Thinker-Performer 推理系统

部署时将单一模型分为两个角色 [§2.4, Fig. 2]:

**Thinker (GPU 0)**:
- 消费当前用户音视频观测
- 运行 causal encoder → token-causal Transformer (language prediction + state update)
- 产出当前 KV-cache slice
- 解码上一步的 audio/video latent 为实际输出

**Performer (GPU 1)**:
- 仅运行 flow matching solver 生成下一步的 clean audio/video latents
- 使用 Thinker 传来的 full-history KV cache 作为 context

**流水线重叠** [§2.4, Fig. 2]:
- 在 streaming step k:
  - Thinker: encode $u_k$ → update $KV_k$ → decode $y_{k-1}$ (同时接收 Performer 的 $y_{k-1}$ latent,发送 $KV_k$)
  - Performer: 使用 $KV_k$ 运行 flow matching 生成 $y_k$ latent
- 关键约束: Performer time + KV/latent communication overhead < 160ms (one streaming unit) [§2.4]
- [论文原文] 部署保留了统一模型的语义(通过 KV 交换),同时将大部分延迟关键工作重叠 [§2.4]

## 实验

**本文无标准定量指标** -- 论文未报告任何生成质量指标 (MOS, FID, FVD, WER, SIM 等)。实验完全聚焦于延迟比较和定性能力展示。

### 延迟比较

| 指标 | Wan-Streamer | 最相近对比 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Model-side latency | ~200 ms | Moshi 160 ms (理论)/200 ms (实际) | N/A | [§3, Tab 1] |
| Total interaction latency | ~550 ms (含 350ms 网络) | Doubao ~1s overall; Hume EVI 3 0.9-1.4s | N/A | [§3, Tab 1] |
| Video FPS | 25 FPS | X-Streamer 25 FPS; LiveTalk 24.82 FPS | N/A | [§3, Tab 2] |
| Streaming unit | 160 ms | Moshi 160 ms | N/A | [§2.4] |
| Output resolution | 192p | N/A (proof of concept) | N/A | [§5] |

### 定性能力 (无定量评估)

论文在 §3 以文字描述了以下能力,但均无定量指标:
1. **Naturalness**: idle/listening 状态下保持身份、凝视、姿态、呼吸等自然行为 [§3]
2. **Interruption handling**: 从交错交互数据中学习打断处理,而非手工规则 [§3]
3. **Proactive speaking**: 感知到视觉事件时主动发起对话 [§3]
4. **Lip-speech synchronization**: 音频和视频 latent 在解码前耦合,天然同步 [§3]

## 局限性

1. **仅 192p 输出分辨率**: 论文明确承认这是 proof of concept,"scaling to higher resolutions is straightforward and left to future work" [§5],但未提供任何高分辨率实验或 scaling 分析
2. **无生成质量评估**: 全文没有任何 MOS、FID、FVD、WER、SIM 等定量指标,也没有用户研究。延迟数据是唯一的定量结果
3. **训练数据未披露**: §2.2 仅笼统描述了数据类别 (understanding/generation/interaction),但数据规模、来源、构成比例均未给出
4. **无 ablation**: 未对任何设计选择 (block-causal vs token-causal, 三阶段训练各阶段贡献, thinker-performer 分离效果等) 进行消融实验
5. **延迟比较不完全公平**: Tab 1 中明确标注不同系统的测量边界不同 (model-side vs first-packet vs API TTFB vs product path),直接横向比较存在偏差
6. **未开源**: 无代码、无模型权重、无 demo 供复现或验证
7. **全双工能力未定量验证**: 打断处理、回传信号、轮次切换等全双工核心能力仅有文字描述,未像 BayLing-Duplex (TT SR, ISR) 或 Raon-SpeechChat (FDB TOR) 那样提供定量评估

## 点评

Wan-Streamer 提出了一个极具野心的系统设计: 将 text/audio/video 的感知和生成统一在单一因果 Transformer 中。这在概念上比现有系统 (如 ELLSA 的双 expert 架构、X-Streamer 的模块化设计) 更加"原生"。Thinker-performer 流水线的工程设计也颇为精巧 -- 通过 KV-cache 交换保持模型语义统一性,同时将昂贵的 flow matching 生成与感知/解码重叠。

然而,作为一篇系统论文,Wan-Streamer 的实验验证严重不足。192p 的输出分辨率和完全缺失的生成质量评估使得核心声称 ("real-time interactive foundation model") 难以评估其实际效果。论文更像是一份技术报告或系统设计文档,而非完整的研究论文。

与 KB 中已有系统的对比:
- vs Moshi: Wan-Streamer 增加了视频维度,但 Moshi 在语音全双工上有更充分的评估
- vs ELLSA: 两者都走向多模态全双工,但 ELLSA 有 LIBERO 上的定量结果,Wan-Streamer 没有
- vs 级联数字人系统 (X-Streamer, MIDAS 等): Wan-Streamer 的端到端路线避免了模块边界延迟,但这些系统在各自组件上有更成熟的质量评估

值得注意的是论文对 rolling distillation + self-forcing 的使用 -- 这解决了 autoregressive 流式生成中 error accumulation 的核心难题,与 CFM 蒸馏领域的最新进展 (DSFlow, SplitMeanFlow) 方向一致。

## 可复用的 idea

1. **Thinker-performer 双 GPU 流水线**: 将统一模型的感知/推理/解码 (轻) 与 latent 生成 (重) 分离到两个 GPU,通过 KV-cache 交换保持全局一致性,同时最大化计算重叠。这个模式可推广到任何 autoregressive + flow matching 的混合架构
2. **Rolling distillation with self-forcing**: 在 rollout 条件下蒸馏,student 在自身生成的 history 上训练,解决 autoregressive 生成的 exposure bias。这对所有流式生成系统 (TTS, 视频) 都有启发
3. **CFG 蒸馏吸收**: 将 teacher 的 CFG 效果蒸馏进 student 以消除推理时的双倍计算,与 DSFlow 的 CFG 内化发现一致,值得在 TTS flow matching 系统中进一步验证
4. **因果全栈设计原则**: "streamability as modeling constraint" -- 从 encoder/decoder/VAE/attention 全部因果化,而非在 bidirectional 架构上做事后 streaming 适配。这个设计哲学对流式 TTS 系统也有参考价值
5. **音视频 latent 联合 flow matching**: 在同一 ODE solver 中同时去噪音频和视频 latent,天然实现跨模态同步。可考虑在 speech+gesture、speech+facial expression 等场景中应用

> [!review] 审阅: pass-with-fixes (2026-06-26, checklist v1.2)
> 0 high / 1 medium / 2 low issues。详见 [[_review/Wan-Streamer-review.yml]]。
> Medium: models 字段含 Related Work 中仅提及的系统,可精简。
> 所有延迟数字与 PDF 交叉验证一致;来源标注覆盖率 ~95%;设计选择 WHY 充分;KB 背景命中 6 页,谱系定位清晰。
