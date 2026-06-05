---
type: paper
tier: deep
title: "DialoSpeech: Dual-Speaker Dialogue Generation with LLM and Flow Matching"
arxiv_id: "2510.08373"
source: "Sources/DialoSpeech.pdf"
authors: [Hanke Xie, Dake Guo, Chengyou Wang, Yue Li, Wenjie Tian, Xinfa Zhu, Xinsheng Wang, Xiulin Li, Guanqiong Miao, Bo Liu, Lei Xie]
year: 2025
venue: "APSIPA ASC 2025"
tags: [dialogue-TTS, dual-track, flow-matching, LLM-TTS, multi-speaker, zero-shot, cross-lingual]
concepts: ["[[ConditionalFlowMatching]]", "[[LLM-basedTTS]]", "[[SpeechLanguageModel]]", "[[SemanticvsAcousticTokens]]", "[[SpeakerEmbedding]]", "[[Turn-takinginSpokenDialogue]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/BigVGAN|BigVGAN]]", "CoVoMix", "MoonCast"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]", "[[任务库/Cross-lingualVoiceCloning|Cross-lingual Voice Cloning]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[ConditionalFlowMatching]], [[LLM-basedTTS]], [[SpeechLanguageModel]], [[SemanticvsAcousticTokens]], [[SpeakerEmbedding]], [[SpeechTokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: DialoSpeech 位于 LLM-based TTS 从单说话人向多说话人对话扩展的分支。在 KB 中,这条路线的先驱包括 dGSLM (dual-tower DLM, 隐式 turn-taking)、CoVoMix (dual-channel zero-shot 混合语音生成)、MoonCast (长对话 podcast)、FireRedTTS 2 (text-speech interleaved dual-transformer)、VibeVoice (next-token diffusion, 4 说话人 90 分钟)。DialoSpeech 延续 dGSLM/CoVoMix 的 dual-track 路线,但引入 LLM backbone + chunked flow matching,试图在更小数据/模型规模下实现可比质量。
>
> **已有认知**: (1) Conditional Flow Matching (CFM) 已在 CosyVoice 系列、F5-TTS 等中被验证为高效 mel spectrogram 生成器; DialoSpeech 的 chunked CFM 是将其扩展到长序列流式场景的新尝试。(2) Semantic vs Acoustic Tokens 的 KB 背景表明,监督式 semantic tokens (S3Tokenizer, CosyVoice 路线) 在内容一致性上优于 HuBERT/EnCodec 方案; DialoSpeech 直接采用 CosyVoice 2 的 S3Tokenizer。(3) Speaker Embedding 的 KB 背景涵盖 ECAPA-TDNN 用于 zero-shot conditioning 的标准做法,DialoSpeech 沿用。(4) Turn-taking in Spoken Dialogue [待确认] 概念页详细记录了 dGSLM dual-tower、Moshi multi-stream、Parrot dual-channel 等方案; DialoSpeech 的 causal cross-attention + [spkchange] tag + <SIL> token 机制是同类方案的新变体。
>
> **创新判断**: 对比 KB 已有系统,DialoSpeech 的关键差异在于: (a) 将 LLM (LLaMA) 与显式 dual-track token 预测结合,而非 dGSLM 的冷启动小 transformer 或 Moshi 的 multi-stream RQ-Transformer; (b) 引入 block-wise guided attention 实现 chunked CFM 解码,解决长对话的内存/延迟问题; (c) 提出完整的 dual-track 数据处理 pipeline (VAD+ASR+diarization+OSD+separation),解决对话数据稀缺问题。
>
> 检索命中: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SpeechLanguageModel]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeakerEmbedding]]✓, [[SpeechTokenizer]]✓ | 过滤: [[Turn-takinginSpokenDialogue]](pending-review), [[Full-duplexSpokenDialogue]](pending-review), [[StreamingSpokenDialogue]](pending-review), [[SpokenDialogueEvaluation]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 双轨架构 (LLM 生成 dual-track semantic tokens + chunked flow matching 合成波形),实现零样本双说话人对话语音合成,仅 10K 小时数据 + 0.5B 模型即达到与百万小时级系统可比的对话自然度
> - **路线**: 对话文本 + 双说话人 prompt → BPE text tokens + ECAPA-TDNN speaker embeddings → LLaMA-based DiaLM (causal cross-attention, dual-head) → dual-track semantic tokens → chunked CFM (DiT backbone, block-wise guided attention) → mel spectrogram → BigVGAN-v2 vocoder → 24kHz 双通道波形
> - **指标**: 中文 CER 2.27% (vs CosyVoice2 2.81%, MoonCast 3.61%) [Table II]; 中文 Spontaneity MOS 3.96 (vs CosyVoice2 3.44, MoonCast 3.87) [Table II]; 英文 WER 8.62% (vs CoVoMix 9.71%) [Table I]; SIM-O 0.67-0.69 [Table I/II]
> - **可借鉴**: (1) block-wise guided attention (causal/history/future mask) 实现固定内存的 chunked flow matching 解码; (2) 数据 pipeline: VAD+Paraformer ASR+Pyannote diarization+OSD+SpatialNet separation 构建 dual-track 对话数据; (3) <SIL> token 在 dual-stream 中隐式建模 turn-taking/overlap
> - **局限**: SIM-O 低于 CosyVoice2 (0.69 vs 0.75 中文); 英文 WER 8.62% 显著高于 CosyVoice2 的 2.40%; 仅评估 dual-speaker 场景; 长对话 (分钟级) 因内存限制未探索; 未开源 (承诺开源但尚未)

## 核心问题

DialoSpeech 要解决的核心问题是: 如何生成自然、连贯、具有交互动态性 (turn-taking、重叠语音) 的双说话人对话语音?

现有方案的三个关键瓶颈 [§I]:
1. **数据稀缺**: 高质量双通道对话数据极少,大多数对话录音是单通道混合音频
2. **交互动态建模困难**: 单流 token 表示 (如 MoonCast) 无法有效建模重叠语音和 turn-taking
3. **推理效率**: 长对话场景下 flow matching 模型的内存和延迟随序列长度线性增长

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DialoSpeech 采用两阶段解耦设计 [§III-A]:

**Stage 1: Text-to-Token (T2T)** — DiaLM 模型
- 输入: 对话文本 T + 双说话人 prompt embeddings (Sp1, Sp2)
- 输出: 双通道 semantic token 序列 (S1, S2)
- 核心: LLaMA-based LM + causal cross-attention + dual prediction head

**Stage 2: Token-to-Waveform (T2W)** — Chunked CFM
- 输入: 单通道 semantic tokens + speaker embedding
- 输出: mel spectrogram → BigVGAN-v2 → 24kHz 波形
- 每个说话人通道独立重建

[论文原文] "This design allows us to leverage specialized models for each sub-task" [§III-A] — 两阶段解耦的动机是让每个子模块专注于各自擅长的建模维度。

### 关键设计选择

#### 1. 双通道 Semantic Token 表示

DialoSpeech 用 CosyVoice 2 的 S3Tokenizer 将每个说话人的语音分别编码为 semantic token 序列 [§III-A]。

**为什么选 S3Tokenizer 而非 HuBERT/EnCodec?** [agent 解读] S3Tokenizer 是监督式 semantic tokenizer (ASR encoder + VQ),在内容一致性 (WER/CER) 上远优于自监督 semantic tokens (如 HuBERT),这对对话场景的可懂度至关重要。KB 中 CosyVoice 系列的成功验证了这一选择。

**双通道设计 vs 单流设计**: [论文原文] 论文明确对比了 MoonCast 的单流 token 方案: "its single-stream token representation prevents it from effectively modelling crucial interactional dynamics such as overlapping speech" [§I]。双通道允许两个说话人在同一时间步各自持有 token (活跃说话人持有 semantic token,非活跃说话人持有 <SIL> token),从而自然表示重叠语音。

#### 2. DiaLM: LLM + Causal Cross-Attention

DiaLM 的核心创新是在 LLaMA backbone 上增加了 **causal cross-attention 机制** 连接双通道 [§III-B]:

```
Text tokens → BPE embedding ─────────────────────────┐
                                                       │
Spk1 prompt → ECAPA-TDNN → Sp1 embedding              │
Spk2 prompt → ECAPA-TDNN → Sp2 embedding              │
                                                       │
S1 tokens → shared embedding ─→ ┐                     │
S2 tokens → shared embedding ─→ ┤ causal cross-attn   │
                                 ├→ fused repr ────────┼→ LLaMA → hidden states
                                                       │
                                              ┌────────┤
                                    Spk1 Head ─→ S1 logits → CE Loss 1
                                    Spk2 Head ─→ S2 logits → CE Loss 2
```

[论文原文] "We apply a causal cross-attention mechanism between the dual speech token streams to model inter-speaker interaction. This allows each speaker's embedding to attend to the other's content and prosodic intent contextually" [§III-B]。

**[spkchange] 控制 token**: 在输入文本的说话人切换处插入 [spkchange] 标记,帮助模型对齐语义结构与对话流 [§III-B]。

**<SIL> token 机制**: 非活跃说话人在对应时间步生成 <SIL>,活跃说话人生成 semantic token [§III-B]。[论文原文] "This design allows the model to autonomously determine when each speaker should speak, remain silent, or overlap with the other, without explicit timing or speaker control signals" [§III-B]。

[agent 解读] 这个设计与 dGSLM 的 dual-tower cross-attention 和 Parrot 的 next-token-pair prediction 思路相似,但不同之处在于 DialoSpeech 使用预训练 LLaMA 作为 backbone (而非冷启动 transformer),且通过 [spkchange] 显式标注 turn boundary (而非纯隐式学习)。

**训练目标**: dual-channel cross-entropy loss [§III-B, Eq. 1]:
$$L_{CE} = \sum_{c=1}^{2} \log P(S^c | T, Sp_c; \theta)$$

#### 3. Chunked Flow Matching: 长序列高效解码

T2W 阶段的 CFM 模型基于 F5-TTS 的 DiT 架构 [§III-C]:

**标准 CFM** [§III-C-1]: 学习向量场 $u_t = x_1 - x_0$ 沿线性路径 $\phi_t(x_0) = (1-t)x_0 + tx_1$ 从噪声 $x_0 \sim \mathcal{N}(0, I)$ 到目标 mel $x_1$,conditioned on 拼接的 semantic tokens + speaker embedding $c$。

**Block-wise Guided Attention** [§III-C-2]: 将 token 序列切分为固定大小 $b$ 的 block,通过 attention mask 控制 DiT 的感受野:

$$M_{i,j} = \begin{cases} 1, & \text{block}(i) - \text{block}(j) \leq \tau \\ 0, & \text{otherwise} \end{cases}$$

三种 mask 模式 [§III-C-2, Fig 4]:
- **Causal Mask** ($\tau=0$): 每个 block 独立,不与其他 block 交互
- **History Mask** ($\tau=1$): 每个 block 可访问前一个 block 信息,扩展感受野
- **Future Mask** ($\tau=-1$): 每个 block 可访问后一个 block 信息

[论文原文] "This chunk-wise decoding strategy ensures a fixed memory footprint and low latency" [§III-C-2]。

[agent 解读] 这一设计类似 CosyVoice 2 的 chunk-aware causal flow matching (四种 mask 统一训练),但 DialoSpeech 的 block-wise mask 更通用,参数化为偏移量 $\tau$,且明确面向长对话场景的固定内存推理。

### 训练策略

**DiaLM** [§IV-B]:
- 0.5B 参数 LLaMA (16 层, hidden 1024, 16 heads) [§IV-B]
- 8x NVIDIA A6000 48GB, batch size 64 [§IV-B]
- 预训练: lr=1e-4, cosine annealing, 150K steps [§IV-B]
- SFT: lr=2e-5, 50K steps (高质量数据微调) [§IV-B]

**Chunked CFM** [§IV-B]:
- DiT backbone, 22 Transformer 层, hidden 768, ~150M 参数 [§IV-B]
- 从 16kHz mel spectrogram 重建, BigVGAN-v2 上采样到 24kHz [§IV-B]

**数据** [§IV-A]:
- 总计 10,000 小时:
  - 3,000h 专业录制中文对话 (Biaobei Corp.)
  - 5,000h 自发中文 podcast 数据 (网络爬取)
  - 2,000h 英文电话对话 (Fisher corpus)
- 全部通过 dual-track pipeline 处理为 speaker-labeled 双通道数据

## 实验

| 指标 | DialoSpeech | CosyVoice2 | CoVoMix | MoonCast | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| CER (↓) | **2.27** | 2.81 | — | 3.61 | 中文 | [Table II] |
| SIM-O (↑) | 0.69 | **0.75** | — | 0.74 | 中文 | [Table II] |
| UTMOS (↑) | 3.410 | **3.499** | — | 2.745 | 中文 | [Table II] |
| Spontaneity MOS (↑) | **3.96** | 3.44 | — | 3.87 | 中文 | [Table II] |
| Coherence MOS (↑) | 3.79 | 3.52 | — | **3.98** | 中文 | [Table II] |
| Intelligibility MOS (↑) | 4.12 | **4.18** | — | 4.23 | 中文 | [Table II] |
| WER (↓) | 8.62 | **2.40** | 9.71 | — | 英文 | [Table I] |
| SIM-O (↑) | 0.67 | **0.72** | 0.46 | — | 英文 | [Table I] |
| UTMOS (↑) | 2.836 | **3.516** | 1.735 | — | 英文 | [Table I] |
| Spontaneity MOS (↑) | **3.71** | 3.43 | 3.64 | — | 英文 | [Table I] |
| Coherence MOS (↑) | **3.37** | 3.32 | 3.38 | — | 英文 | [Table I] |

**关键发现**:

1. **中文**: DialoSpeech 在 CER 和 Spontaneity 上最优,但 SIM-O 和 UTMOS 低于 CosyVoice2 [Table II]。[论文原文] 作者指出 CosyVoice2 和 MoonCast 使用超过 1M 小时数据,而 DialoSpeech 仅用 10K 小时 + 0.5B 模型 [§IV-D],数据效率显著。

2. **英文跨语言**: DialoSpeech 仅用 Fisher 英文数据训练,WER 8.62% 显著高于 CosyVoice2 的 2.40%,但在主观 Spontaneity/Coherence 上略胜 [Table I]。[论文原文] CosyVoice2 受益于更大英文数据集的优势 [§IV-D]。

3. **CoVoMix 对比**: DialoSpeech 在英文所有指标上全面超越 CoVoMix (Fisher 8kHz 训练),尤其 SIM-O 0.67 vs 0.46, UTMOS 2.836 vs 1.735 [Table I]。

## 局限性

1. **说话人相似度不足**: SIM-O 在中英文场景均低于 CosyVoice2 (中文 0.69 vs 0.75, 英文 0.67 vs 0.72) [Table I/II],说明 speaker identity preservation 仍有提升空间
2. **英文智能度差距大**: WER 8.62% vs CosyVoice2 2.40% [Table I],主要因为英文训练数据有限 (仅 Fisher 2000h, 且质量不高)
3. **仅双说话人**: 未探索 3+ 说话人场景,而 VibeVoice 已支持 4 说话人
4. **长对话受限**: [论文原文] "it becomes difficult on minute-scale data because of memory limitations" [§V],即使有 chunked FM,DiaLM 的 LM 部分仍面临长上下文挑战
5. **未开源**: 论文声称将公开代码和 checkpoints [§I],但截至目前尚未公开
6. **评估规模有限**: 仅 30 名评估者 [§IV-C],未报告置信区间或显著性检验

## 点评

DialoSpeech 的核心贡献在于提出了一个 **完整且可复现的对话 TTS pipeline**,从数据处理到模型训练再到评估。然而,其技术方案的各模块并非全新: DiaLM 的 dual-track cross-attention 与 dGSLM 思路相近,chunked flow matching 与 CosyVoice 2 的 chunk-aware FM 有交叉,数据 pipeline 的各步骤 (VAD+ASR+diarization+OSD+separation) 也都是成熟工具的组合。

论文最有说服力的论据是 **数据效率**: 仅用 10K 小时 + 0.5B 模型在中文 Spontaneity 上超越使用 1M+ 小时的 CosyVoice2 和 MoonCast。但这也引发疑问: 如果给 CosyVoice2 相同的 dual-track 训练数据和 dual-head 架构,差距是否仍然存在?

英文结果的 WER 差距 (8.62% vs 2.40%) 严重影响了跨语言泛化的说服力,尽管作者归因于数据量差异。

## 可复用的 idea

1. **Block-wise guided attention for chunked CFM**: 通过参数化 attention mask ($\tau$ offset) 实现 causal/history/future 三种模式的统一,为长序列 flow matching 提供固定内存推理方案。这个设计可迁移到任何基于 DiT 的 flow matching 模型。

2. **Dual-track 数据处理 pipeline**: VAD → Paraformer ASR → Pyannote diarization → word-speaker alignment → punctuation restoration → OSD (Conformer+XLSR) → SpatialNet separation → multi-stage filtering (SNR/clustering/similarity/DNSMOS)。这是构建 dual-track 对话数据集的实用参考。

3. **混合式 turn-taking 建模**: DialoSpeech 组合使用 [spkchange] (显式 turn boundary 标记) + <SIL> token (隐式 timing 学习),在 dual-stream LM 中让模型自主决定何时切换、沉默、重叠。这种显式+隐式混合策略值得参考。

---

> [!review] 审阅: pass-with-fixes (0 high, 2 medium, 1 low)
> - [medium] frontmatter models 补充 CoVoMix/MoonCast baseline — 已修正
> - [medium] 训练策略节超参数补充 [§IV-B] 出处标注 — 已修正
> - [low] 可复用 idea #3 修正 <SIL> 与 [spkchange] 的关系描述 — 已修正
> 详见 `_review/DialoSpeech-review.yml`
