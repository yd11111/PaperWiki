---
type: paper
tier: deep
title: "Voxtral TTS"
arxiv_id: "2603.25551"
source: "Sources/VoxtralTTS.pdf"
authors: [Alexander H. Liu, Alexis Tacnet, Andy Ehrenberg, Andy Lo, Chen-Yo Sun, Guillaume Lample, Henry Lagarde, Jean-Malo Delignon, Jaeyoung Kim, John Harvill, Khyathi Raghavi Chandu, Lorenzo Signoretti, Margaret Jennings, Patrick von Platen, Pavankumar Reddy Muddireddy, Rohin Arora, Sanchit Gandhi, Samuel Humeau, Soham Ghosh, Srijan Mishra, Van Phung]
year: 2026
venue: "arXiv"
tags: [TTS, zero-shot, voice-cloning, multilingual, codec, flow-matching, autoregressive, DPO, speech-tokenizer, hybrid-architecture]
concepts: ["[[Semantic vs Acoustic Tokens]]", "[[Speech Tokenizer]]", "[[LLM-based TTS]]", "[[Conditional Flow Matching]]", "[[Classifier-Free Guidance]]", "[[Finite Scalar Quantization]]", "[[Speech Factorization]]", "[[Speech Language Model]]", "[[Voice Cloning Taxonomy]]", "[[Codec Training Objectives]]"]
models: ["[[模型库/Whisper|Whisper]]", "Mimi", "Ministral 3B", "ElevenLabs v3", "ElevenLabs Flash v2.5", "Gemini 2.5 Flash TTS"]
tasks: ["zero-shot TTS", "voice cloning", "multilingual TTS"]
datasets: ["Expresso", "SEED-TTS", "MiniMax-TTS"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Semantic vs Acoustic Tokens]], [[Speech Tokenizer]], [[LLM-based TTS]], [[Conditional Flow Matching]], [[Speech Factorization]], [[Speech Language Model]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Voxtral TTS 属于 **Hybrid LLM+Flow 架构** 谱系,与 CosyVoice (LLM semantic tokens + CFM acoustic generation) 路线高度相似,但在实现层面有显著差异: (1) Voxtral 使用 VQ+FSQ 混合量化而非纯 FSQ; (2) 声学生成用 flow-matching transformer 而非 CFM 直接生成 mel; (3) 将 DPO 适配到 hybrid discrete-continuous 设定。
>
> **已有认知**:
> - [[Semantic vs Acoustic Tokens]]: Voxtral Codec 的 1 semantic + 36 acoustic 分离方案是 Mimi 路线的直接演进,但语义侧改用 ASR 蒸馏(CosyVoice 开创的监督式 semantic token 路线),声学侧改用 FSQ 替代 RVQ。
> - [[Speech Tokenizer]]: Voxtral Codec 在 tokenizer 设计上结合了 VQ (语义) + FSQ (声学) 的混合方案,这是一种新的量化组合策略,不同于 CosyVoice 的纯 FSQ 和 Mimi 的纯 RVQ。
> - [[LLM-based TTS]]: Voxtral TTS 继承了 VALL-E 开创的 AR 生成范式,但用 flow-matching transformer 替代了传统的 NAR 阶段 (depth transformer / MaskGIT),效率和质量双升。
> - [[Conditional Flow Matching]]: flow-matching 在此处的角色不同于 CosyVoice 中直接生成 mel;Voxtral 的 FM transformer 在 AR backbone 的每一步独立生成 36 维 acoustic embedding,然后 FSQ 量化回离散 token,再续入 AR 循环。
> - [[Speech Factorization]]: Voxtral Codec 通过物理分割 latent (256-dim semantic + 36-dim acoustic) 实现分解,比对抗训练/信息瓶颈更直接。
>
> **创新判断**: 与已有 KB 中的系统对比,Voxtral TTS 的主要新贡献在于 (1) VQ-FSQ 混合量化 codec, (2) FM 替代 depth/MaskGIT 用于 acoustic token 生成, (3) Flow-DPO 适配 AR 设定。
>
> 检索命中: [[Semantic vs Acoustic Tokens]]✓, [[Speech Tokenizer]]✓, [[LLM-based TTS]]✓, [[Conditional Flow Matching]]✓, [[Speech Factorization]]✓, [[Speech Language Model]]✓ | 过滤: [[Finite Scalar Quantization]](pending-review), [[Classifier-Free Guidance]](pending-review), [[Voice Cloning Taxonomy]](pending-review), [[Codec Training Objectives]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: Mistral 提出 hybrid AR+flow-matching TTS,用 VQ-FSQ 混合 codec 编码 semantic/acoustic tokens,在多语言 zero-shot voice cloning 上以 68.4% win rate 超越 ElevenLabs Flash v2.5。
> - **路线**: Text + Voice ref → Voxtral Codec (VQ semantic + FSQ acoustic, 12.5Hz, 2.14kbps) → AR decoder backbone (Ministral 3B) 生成 semantic tokens → FM transformer (8 NFE + CFG) 生成 acoustic tokens → Codec decoder → 24kHz waveform
> - **指标**: Voice cloning win rate 68.4% vs ElevenLabs Flash v2.5 [Table 5]; Speaker SIM 0.628 vs 0.392 (ElevenLabs v3) on SEED-TTS [Table 3]; WER 1.23% on SEED-TTS [Table 3]; UTMOS 4.11 on SEED-TTS [Table 3]; Codec PESQ 3.05 vs Mimi-16cb 2.67 on Expresso [Table 2]
> - **可借鉴**: (1) VQ+FSQ 混合量化: 语义用高维 VQ 保证信息容量,声学用低维 FSQ 保证利用率,避免了纯 RVQ 的 codebook collapse; (2) Flow-matching 替代 depth transformer: 3 token 序列长度(h,t,x_t) vs 38 (MaskGIT) 或 36 步 AR (depth),大幅降低 acoustic 生成开销; (3) Flow-DPO 的 AR 适配: 每个 token 独立采样 t,长度归一化导致不稳定
> - **局限**: 不支持 emotion tag/text instruction 显式控制(只能通过 voice prompt 风格迁移); 仅 CC BY-NC 开源; DPO 在 Hindi 上 WER 反而退化 (+1.61%); CFG alpha 高时 text-adherence 变差; 论文未提供 MOS 原始分数

## 核心问题

1. **如何在 zero-shot TTS 中同时实现长程一致性和丰富声学细节?** 传统 depth-wise AR 或 MaskGIT 的 acoustic 生成要么太慢要么质量不够,flow-matching 能否更优?
2. **如何设计一个统一的低码率 codec 同时编码语义和声学?** 现有 codec (Mimi) 用纯 RVQ,存在 codebook collapse 和高码率问题。
3. **如何将 DPO 对齐方法适配到 hybrid discrete+continuous 生成设定?** 标准 DPO 仅适用于离散 token,flow-matching 的连续输出需要新目标。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Voxtral TTS 是一个三组件系统 [§2, Fig 2]:

1. **Voxtral Codec**: 将 24kHz 波形压缩为 12.5Hz 帧序列,每帧 37 个离散 token (1 semantic VQ + 36 acoustic FSQ),总码率 2.14kbps [§2.1]
2. **Decoder Backbone**: 基于 Ministral 3B 的 decoder-only transformer,AR 生成 semantic token 序列 [§2.2]
3. **Flow-Matching Transformer**: 在每个 AR step 独立预测 36 维 acoustic embedding,然后 FSQ 量化回离散 token [§2.3]

生成流程: Voice ref → Codec encoder → audio tokens (semantic+acoustic) → 与 text tokens 拼接输入 backbone → AR 生成 semantic tokens (直到 `<EOA>`) → 每步 FM transformer 生成 acoustic tokens → 合并 semantic+acoustic → Codec decoder → waveform [Fig 2]

### 关键设计选择

**设计选择 1: VQ+FSQ 混合量化方案**

Voxtral Codec 将 292 维 latent 物理分割为 256-dim semantic + 36-dim acoustic [§2.1]:
- Semantic: 256-dim → VQ codebook 8192 entries (13 bits),训练时 50% 概率不量化 [论文原文]
- Acoustic: 每个 36 维度独立 → tanh + FSQ 21 levels (每维 ~4.4 bits) [论文原文]

为什么不用纯 RVQ? [agent 解读] Mimi 用 RVQ 32 层达到 4.4kbps 才获得好质量,而 Voxtral Codec 用 VQ+FSQ 在 2.1kbps 就超越了 Mimi-16cb (2.2kbps)。FSQ 的 100% codebook 利用率避免了 RVQ 的 codebook collapse;VQ 在语义侧保证了足够的信息容量(8192 entries vs FSQ 的低维网格)。

为什么 VQ 训练时 50% 不量化? [论文原文] 这是训练稳定性技巧。FSQ 也用 dither-style: 50% 量化 + 25% 加噪 + 25% 不量化 [§2.1]。[agent 解读] 这与 CosyVoice 使用的 end-to-end 训练中量化 dropout 类似,避免量化误差在训练早期主导梯度。

**设计选择 2: ASR 蒸馏的 Semantic Token**

不同于 Mimi 使用自监督 WavLM 蒸馏,Voxtral Codec 从 **Whisper ASR 模型** 蒸馏 semantic token [§2.1]。具体机制:
- 冻结 Whisper decoder 产生 hidden states h_l 和 cross-attention weights A
- 将 codec 的 post-VQ semantic embedding 线性投影后,通过 attention-weighted alignment 与 Whisper hidden states 做 cosine distance loss [Eq. 1]
- Alignment 矩阵 A 从 Whisper cross-attention heads 中自动提取(DTW 选择最佳 heads),免去外部 forced aligner [论文原文]

为什么用 ASR 蒸馏而非自监督蒸馏? 论文引用 Vashishth et al. (2024) 指出自监督表征 "更 phonetic 而非 semantic",监督 ASR 模型产生更有效的 semantic representations [§2.1]。[agent 解读] 这延续了 CosyVoice 开创的监督式 semantic token 路线,但改用 Whisper (多语言) 而非 SenseVoice (中英为主),为 9 语言支持提供基础。

**设计选择 3: Flow-Matching 替代 Depth Transformer**

为什么用 FM 而非 MaskGIT 或 Depth Transformer 预测 acoustic tokens?

论文明确对比了三种方案 [§2.3]:
- **MaskGIT**: 需 attend 36 acoustic + 2 conditioning tokens = 38 per frame,序列长度大 [论文原文]
- **Depth Transformer**: 需 36 步 AR 解码 [论文原文]
- **Flow-Matching**: 仅 3 个 input (h, t, x_t),8 NFE 即可,且人类评估中表现力最优 [论文原文]

FM transformer 是 3 层双向 transformer,与 backbone 同宽度。在每个 AR step 独立运行:接受 backbone hidden state h、time step t 的正弦编码、当前 acoustic embedding x_t,通过 Euler 积分 8 步从高斯噪声生成 acoustic embedding,再 FSQ 量化到 21 levels 后反馈给 backbone [§2.3, Eq. 4-5]。

CFG 仅在 FM transformer 中应用(10% hidden state dropout for unconditional),不在 backbone 中应用,因此开销很小(仅需额外一次 FM forward,不是整个 backbone) [§2.3]。默认 alpha=1.2 [§5.2]。

**设计选择 4: Flow-DPO 对齐**

将 DPO 适配到 hybrid discrete-continuous 设定 [§3.2]:
- Semantic: 标准 DPO objective (beta_semantic=0.1)
- Acoustic: 改编自 MR-FlowDPO (Ziv et al., 2025) [Eq. 8-9],关键适配: 每个 token 位置独立采样 t (不同于原始论文的全局 t) [Eq. 10]
- 发现长度归一化导致不稳定,因此不做归一化 [论文原文]
- beta_acoustic=0.5,学习率极低 8e-8 [论文原文]

DPO 数据构造: rejection sampling pipeline + Mistral Small Creative 生成多样 text prompts → 多次采样 → WER/SIM/UTMOS/loudness 综合打分选 winner/loser [§3.2]

### 训练策略

**Codec 训练** [§2.1, Eq. 3]:
- 端到端: alpha*L_feature + beta*L_ASR + gamma*L_L1 + gamma*L_STFT + delta*L_commit
- 关键: L1 和 STFT 重建 loss 使用指数衰减 gamma=0.9999^t,早期 bootstrap 学习,后期让 adversarial signal 主导 [论文原文]
- 多分辨率判别器: 8 个 STFT sizes,feature-matching loss 替代标准 GAN generator loss [§2.1]
- Codec ~300M params [§2.1]

**TTS 预训练** [§3.1]:
- Decoder backbone 从 Ministral 3B 初始化,FM transformer 等新模块随机初始化
- 训练数据: 伪标注 (A1, T2, A2) 三元组,A1=voice ref, T2=transcript, A2=target speech
- 冻结 text embedding layers 提高对低频文本 token 的鲁棒性 [论文原文]
- VAD 降低静音帧 loss weight,极长静音设为 0 [论文原文]
- LLM 改写 transcripts 增强文本归一化鲁棒性 [论文原文]

**DPO 后训练** [§3.2]:
- 联合 DPO loss + 预训练 loss,仅 1 epoch (更长训练导致 robotic speech) [论文原文]

## 实验

### Voxtral Codec vs Mimi [Table 2, Expresso]

| 指标 | Voxtral Codec (2.1kbps) | Mimi-16cb (2.2kbps) | Mimi-32cb (4.4kbps) | 出处 |
| --- | --- | --- | --- | --- |
| Mel distance ↓ | **0.545** | 0.618 | 0.552 | [Table 2] |
| STFT distance ↓ | **0.982** | 1.100 | 1.040 | [Table 2] |
| PESQ ↑ | **3.05** | 2.67 | 3.18 | [Table 2] |
| ESTOI ↑ | **0.882** | 0.865 | 0.910 | [Table 2] |
| ASR-WER ↓ | **10.66** | 11.01 | 10.25 | [Table 2] |
| Speaker SIM ↑ | **0.843** | 0.829 | 0.902 | [Table 2] |

在相近码率下 (2.1 vs 2.2 kbps) Voxtral Codec 全面超越 Mimi-16cb [Table 2]。

### TTS 自动评估 [Table 3, SEED-TTS + MiniMax-TTS 9 languages]

| 指标 | Voxtral TTS | ElevenLabs v3 | ElevenLabs Flash v2.5 | 出处 |
| --- | --- | --- | --- | --- |
| WER (SEED-TTS) ↓ | 1.23 | 1.26 | **0.86** | [Table 3] |
| UTMOS (SEED-TTS) ↑ | **4.11** | 3.92 | 4.09 | [Table 3] |
| Speaker SIM (SEED-TTS) ↑ | **0.628** | 0.392 | 0.413 | [Table 3] |

Voxtral TTS 在 Speaker Similarity 上大幅领先 (~60% relative improvement over ElevenLabs v3) [Table 3]。WER 略高于 ElevenLabs Flash 但 UTMOS 最优。

### 人类评估 [Table 4, Table 5]

**Flagship voices (Table 4)**:
- vs ElevenLabs v3 (explicit steering): 51.0% win rate
- vs ElevenLabs v3 (implicit steering): 55.4% win rate
- vs Gemini 2.5 Flash TTS (explicit): 35.4%, (implicit): 37.1%

**Zero-shot voice cloning (Table 5)**:
- vs ElevenLabs Flash v2.5: **68.4% overall win rate**
- 最强: Spanish 87.8%, Hindi 79.8%, Portuguese 74.4%
- 最弱: Dutch 49.4%, French 54.4%

### DPO 效果 [Table 6]

| 指标 | Pretrain → DPO | 出处 |
| --- | --- | --- |
| WER SEED-TTS | 1.58 → 1.23 (-0.35) | [Table 6] |
| WER German | 4.08 → 0.83 (-3.25) | [Table 6] |
| WER French | 5.01 → 3.22 (-1.79) | [Table 6] |
| WER Hindi | 3.39 → 4.99 (+1.61) | [Table 6] |
| UTMOS 全部语言 | 一致提升 +0.04~+0.13 | [Table 6] |
| Speaker SIM | ±0.01 (基本不变) | [§5.1] |

DPO 显著降低 WER 和提升 UTMOS,但对 speaker similarity 无明显影响,且 Hindi 上 WER 反而退化 [§5.1]。

### 推理性能 [Table 7, Table 8]

| 配置 | Latency | RTF | 出处 |
| --- | --- | --- | --- |
| Eager mode | 133ms | 0.258 | [Table 7] |
| CUDA graph | 70ms | 0.103 | [Table 7] |
| Concurrency 32 (H200) | 552ms | 0.302 | [Table 8] |

CUDA graph 加速 FM transformer 获得 47% latency 改善和 2.5x RTF 降低 [Table 7]。单 H200 支持 32 并发用户,zero wait rate [Table 8]。

## 局限性

1. **无显式情感控制**: 不支持 emotion tag 或 text instruction,只能通过 voice prompt 风格迁移。Gemini 2.5 Flash TTS 在 explicit steering 上明显优于 Voxtral (35.4% win rate) [Table 4] [论文原文]
2. **DPO 在部分语言上退化**: Hindi WER 从 3.39 → 4.99 (+1.61%),暗示 DPO 数据/目标可能存在语言偏差 [Table 6]
3. **CFG alpha 的 quality-adherence trade-off**: 高 alpha 提升 speaker similarity 但降低 text-adherence 和情感表现力 [§5.2]
4. **仅 CC BY-NC 开源**: 不允许商用 [§Abstract]
5. **缺少与更多开源系统的对比**: 未与 CosyVoice、MaskGCT、F5-TTS 等开源系统比较
6. **训练数据未公开**: 数据规模和来源未详述,仅提到 Voxtral Mini Transcribe 伪标注 [§3.1]

## 点评

**优势**:
1. **VQ+FSQ 混合量化**是一个聪明的设计: VQ 在高维语义空间提供足够信息容量 (8192 entries, 13 bits),FSQ 在低维声学空间提供 100% 利用率和训练稳定性。这比纯 RVQ (Mimi) 和纯 FSQ (CosyVoice) 都更合理地匹配了两类信息的特性。
2. **FM 替代 depth/MaskGIT**的论证非常清晰: 序列长度 3 vs 38/36,且人类评估表现力更优。这是一个在 latency/quality 都占优的 Pareto 改进。
3. **Flow-DPO**的适配思路值得关注: 将 MR-FlowDPO 从全局 t 改为 per-token t,且发现长度归一化不稳定,这些 negative results 对后续研究有价值。

**不足**:
1. 与开源 TTS 系统 (CosyVoice 3, MaskGCT, F5-TTS) 完全没有对比,只比 ElevenLabs (闭源) 和 Gemini,无法判断 architectural innovation 的真实贡献。
2. ASR 蒸馏从 Whisper → codec semantic token 的消融不够: 没有对比不用 ASR 蒸馏/改用自监督蒸馏的效果。
3. 论文标榜"自研 codec",但 Voxtral Codec 的架构 (conv-transformer autoencoder + multi-resolution discriminator) 与 Mimi 高度相似,主要差异是量化方案和 ASR 蒸馏 loss。

## 可复用的 idea

1. **VQ+FSQ 混合量化**: 对不同类型信息使用不同量化方案 (高维 VQ for 语义 + 低维 FSQ for 声学),比"一刀切"的 RVQ/FSQ 更匹配信息特性。可迁移到任何需要 semantic-acoustic 分离的 codec 设计。

2. **FM transformer 替代 depth AR**: 在每个 AR step 用 3-input FM (h, t, x_t) 预测 multi-codebook acoustic tokens,比 MaskGIT (attend 所有 position) 或 depth transformer (逐层 AR) 序列长度更短、质量更优。适用于任何需要在 AR backbone 上附加细粒度 acoustic 预测的场景。

3. **Whisper cross-attention 自动对齐**: 利用 Whisper 的 cross-attention heads (DTW 选择) 作为 soft aligner,免去外部 forced aligner 或 paired transcripts。可迁移到其他需要语音-文本对齐的任务。

4. **指数衰减重建 loss**: gamma=0.9999^t 让 L1/STFT loss 在训练早期 bootstrap 学习、后期让 adversarial signal 主导。简单但有效的 codec 训练技巧。

5. **CFG 仅在 FM transformer 中应用**: 不在 AR backbone 做 CFG,只在轻量 FM 中做,开销极小。适用于任何 hybrid AR+continuous 架构。

---

> [!review] 审阅结论: 待生成
> 审阅报告将在下方步骤中生成。

---

检索命中: [[Semantic vs Acoustic Tokens]], [[Speech Tokenizer]], [[LLM-based TTS]], [[Conditional Flow Matching]], [[Speech Factorization]], [[Speech Language Model]] | 过滤: [[Finite Scalar Quantization]](pending-review), [[Classifier-Free Guidance]](pending-review), [[Voice Cloning Taxonomy]](pending-review), [[Codec Training Objectives]](pending-review) | 未命中但可能相关: 无
