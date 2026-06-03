---
type: paper
tier: deep
title: "MambaVoiceCloning: Efficient and Expressive Text-to-Speech via State-Space Modeling and Diffusion Control"
arxiv_id: ""
source: "Under review at ICLR 2026"
authors: [Anonymous]
year: 2026
venue: "ICLR 2026 (under review)"
tags: [TTS, SSM, Mamba, diffusion, voice-cloning, streaming, efficiency, encoder-design, state-space-model]
concepts: ["[[Diffusion-based TTS]]", "[[Prosody Modeling]]", "[[F0 Modeling]]", "[[Speaker Embedding]]", "[[Voice Cloning Taxonomy]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[Speaker Embedding]], [[Prosody Modeling]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Speaker Embedding]]✓, [[Prosody Modeling]]✓ | 过滤: [[Diffusion-based TTS]](pending-review), [[F0 Modeling]](pending-review), [[Voice Cloning Taxonomy]](pending-review), [[Mel Spectrogram]](pending-review), [[Non-autoregressive TTS]](pending-review) | 未命中但可能相关: 无
>
> **Speaker Embedding**: MVC 使用全局 style embedding e (mel-derived, shallow conv/GRU) 通过 AdaLN 注入编码器,属于 KB 中 "FiLM conditioning" 和 "Conditional LayerNorm" 注入方式的 SSM 版本。与传统 speaker encoder (d-vector, x-vector) 不同,MVC 的 embedding 同时编码 timbre 和 coarse expressiveness [§3.1, Eq. 1]。[agent解读]
>
> **Prosody Modeling**: MVC 将韵律建模分解为三个 SSM 模块: (1) Bi-Mamba Text Encoder 编码语言韵律 (重音/语调); (2) Temporal Bi-Mamba 编码节奏/时长; (3) Expressive Mamba 编码说话人特有的韵律风格。这种模块化分解对应 KB 中 "显式韵律信息" 与 "隐式韵律信息" 的结合 — duration 显式预测, prosody style 隐式通过 SSM 建模。消融实验证明三个模块各贡献不可替代的信息 (CMOS-N drop -0.36 to -0.41) [Table 6]。[agent解读]
>
> [待确认] **Diffusion-based TTS**: KB 记录了 Diff-TTS→Grad-TTS→ProDiff→DiffGAN-TTS 的演进,以及 flow matching 逐步取代 diffusion 的趋势。MVC 保持 StyleTTS2 的 diffusion decoder + vocoder 不变,仅重新设计 conditioning path,因此其贡献在于编码器侧而非生成侧。diffusion decoder 仍是延迟主要来源 (54.2%) [Table 15]。[agent解读]

> [!summary] 速查
> - **一句话**: 首个推理时完全无 attention 的 diffusion-based TTS conditioning stack,用三个 selective SSM (Mamba) 模块分别编码文本、节奏和韵律,在 protocol-matched 实验下小幅超越 StyleTTS2/VITS 并将编码器参数降至 21M、吞吐提升 1.6x [§1, Table 9]
> - **路线**: phonemized text + reference audio → Bi-Mamba Text Encoder (gated bidirectional + AdaLN) + Temporal Bi-Mamba (rhythm/alignment) + Expressive Mamba (prosody/style) → [Training-only: attention-based aligner] → pitch modeling (SSM-only F0) → speech dynamics → StyleTTS2 diffusion decoder → HiFi-GAN/iSTFTNet vocoder [§3, Fig 1]
> - **指标**: LibriTTS unseen MOS-N 4.22 / MOS-S 4.07 (vs StyleTTS2 4.15/4.03, VITS 3.69/3.54) [Table 1]; LJSpeech F0 RMSE 0.653, MCD 4.91, PESQ 3.85, RTF 0.0169 [Table 4]; Encoder params 21M (vs StyleTTS2 42M), throughput 1.6x, peak memory 72% [Table 9]; OOD MOS 3.88 vs StyleTTS2 3.87 [Table 2]; Long-form MOS 4.16 vs 3.91, RTF 0.0170 vs 0.0200 [Table 3]
> - **可借鉴**: (1) Gated bi-directional Mamba fusion: sigmoid gate 调制 forward/backward 方向,改善长文本韵律一致性 [§3.2.1, Eq. 3]; (2) AdaLN + Mamba: 将 style embedding 通过 AdaLN 注入 SSM,实现 speaker conditioning [§3.2.1, Eq. 4]; (3) Training-time aligner + inference-time removal: 轻量 attention aligner 仅训练用,推理时丢弃,保持全 SSM 推理路径 [§3.3]; (4) Protocol-matched evaluation: 所有 baseline 在同一 mel-diffusion-vocoder pipeline 下重训,隔离 conditioning 效果 [§4.1]
> - **局限**: 仅英语数据集 (LJSpeech/LibriTTS) [§6 Limitations]; AdaLN 提供全局 style,不支持细粒度情感控制 [§6]; Diffusion decoder 仍占 54.2% 延迟,整体 RTF 改善有限 [Table 15]; 不与 NaturalSpeech 3/CosyVoice 3 等工业系统直接对比 (数据/规模差异) [§1, §4.1]; double-blind review 匿名

## 核心问题

MVC 要回答一个明确的架构问题 [§1]:

**一个 diffusion-based TTS 系统的 conditioning path 能否在推理时完全基于 SSM,去除所有 attention 和 recurrence,同时保持或改善文本、节奏和韵律的合成质量?**

背景动机 [§1]:
1. **Transformer attention 的问题**: 二次激活增长、全局上下文混合导致长文本合成时内存不稳定 [论文原文]
2. **RNN/GRU 的问题**: 长序列漂移 (long-range drift)、不稳定的记忆状态 [论文原文]
3. **现有 Mamba-TTS 的不彻底**: Miyazaki'24, Jiang'24, Zhang'24 等系统在推理时仍保留 attention (用于 duration prediction 或 style module),不是真正的 SSM-only [§2, Table 10]
4. **线性注意力不足**: Performers 等方法虽降低复杂度但丢失了文本-韵律的全局交互 [§1]

## 方法: MVC

### 输入处理 [§3.1]

- 24kHz 音频 → 80-bin log-mel spectrogram (FFT 1024, hop 256, Hann window) [§3.1]
- 文本通过 phonemizer (Bernard & Titeux, 2021) 转为 phoneme 序列,支持 CSS10 多语言 tag [§3.1]
- Token embeddings: x ∈ R^{T_x x d} [Eq. 1]
- Global style embedding: e = (1/T_m) Σ f_θ(M_{·,t}),f_θ 是 shallow conv/GRU 模块 [Eq. 1]

### 编码器三模块 [§3.2]

所有 Mamba block 使用 state dimension 96, depthwise conv kernel 5, gating temperature τ=1.0 [§3.2]

#### 1. Gated Bi-Mamba Text Encoder [§3.2.1]

替换 self-attention 实现线性时间文本编码:

```
h_f = Mamba_f(x),  h_b = Mamba_b(x)           [Eq. 2]
h_T = (σ(W_g[h_f; h_b]) ⊙ [h_f; h_b]) W_o    [Eq. 3, gated fusion]
h_{T,s} = AdaLN(h_T, e)                        [Eq. 4, style conditioning]
```

**关键创新**: Gated bidirectional fusion [§3.2.1]
- 先前 Mamba-TTS (Jiang'24, Zhang'24) 用简单 concatenation 合并双向 [论文原文]
- MVC 用 sigmoid gate σ(W_g[·]) 调制各方向贡献 [Eq. 3]
- Gate 基于局部句法线索动态调整前向/后向权重 [论文原文]
- 改善长文本的韵律一致性,减少漂移 [Tables 2, 3]
- 消融: 去除 gating → MOS-long 降至 4.02 (vs 4.16); 去除 AdaLN → 3.95 [Table 8]

**Gated Bi-Mamba + AdaLN 是首次出现在 Mamba-TTS 中的组合** [§3.2.1]

#### 2. Expressive Mamba Encoder [§3.2.2]

注入 speaker-specific 韵律:

```
h_E = Mamba(h_{M,s}) ∈ R^{T_m x d_h}          [Eq. 5]
```

- 输入: mel features 经 style-conditioned gated transformation (AdaLN) [§3.2.2]
- 全 SSM-based (无 attention),捕获长时韵律动态 [§3.2.2]
- 消融: 移除 → OOD CMOS-N 下降最大 (-0.41),证明韵律路径对 OOD 文本至关重要 [Table 6]

#### 3. Temporal Bi-Mamba Encoder [§3.2.3]

建模节奏结构和 phoneme-duration 对齐:

```
h_B = [h_f; h_b] W_f                           [Eq. 6, linear fusion]
```

- Style embedding e broadcast 到帧级 → shallow gated transform → Mamba + local Conv1D [§3.2.3]
- 线性融合 (无 gating),因为韵律解耦已在 text/expressive encoder 上游处理 [§3.2.3]
- 消融: 移除 → CMOS-N drop -0.36, 节奏和局部韵律不稳定 [Table 6]

### 对齐与 Pitch [§3.3]

**Training-time aligner** [§3.3]:
- 2-layer transformer, 4 heads, hidden 256, monotonic alignment loss [§3.3]
- 将 token-level h_{T,s} 映射到 frame-level h_A [Eq. 7]
- **推理时完全丢弃** — 使用 temporal predictor (Conv1D + SSM) 替代 [§3.4]
- 对 alignment noise 鲁棒: ±10% 扰动仅增加 WER <0.4%, MOS drop <0.05 [Appendix B.7]

**Pitch modeling** [§3.3]:
- 融合 expressive (h_E) 和 temporal (h_B) 编码 → gated block → F0 prediction [Eq. 8]
- 全 SSM-only pitch path,无 attention-based pitch predictor [§3.3]
- 对 bounded-memory streaming 重要 [论文原文]

### Speech Dynamics & Decoder [§3.4, §3.5]

- 从 aligned features h_A 和 pitch h_P → temporal predictor (Conv1D + SSM) → rhythm-aware representation [§3.4]
- 融合为 decoder conditioning: h_D = [F̂_0; n] ∈ R^{T_m x (1+d_h)} [Eq. 9]
- **Decoder**: StyleTTS2 diffusion model (5-step, 固定) + HiFi-GAN/iSTFTNet vocoder [§3.5]
- Loss: L_total = λ_mel L_mel + λ_adv L_adv + λ_align L_align [Eq. 10]

### Streaming [§5.3]

- 双向 text encoder 替换为 causal Uni-Mamba,SSM state 跨 chunk 保持 [§5.3]
- Look-ahead L 秒提供未来 mel context [§5.3]
- L=0.5s: WER 9.4%, MOS 3.81; L=2.0s: WER 7.3%, MOS 3.91 (接近非 streaming) [Table 5]
- Chunk boundary 处平滑: L ≥ 0.5s 感知上足够,仅 L=0.25s 偶有短停顿 [§5.3]

## 实验结果

### 主观 & 客观质量 [§5.1]

**LibriTTS unseen speakers (Table 1)**:

| Model | MOS-N ↑ | MOS-S ↑ |
|-------|---------|---------|
| Ground Truth | 4.60 | 4.35 |
| VITS | 3.69 | 3.54 |
| StyleTTS2 | 4.15 | 4.03 |
| **MVC** | **4.22** | **4.07** |

- MVC 略优于 StyleTTS2 (paired t-test, p<0.01),差异 "modest but statistically robust" [§5.1]
- 绝对改善约 +0.07 MOS [§5.1]

**LJSpeech objective (Table 4)**:

| Model | F0 RMSE ↓ | MCD ↓ | WER ↓ | PESQ ↑ | RTF ↓ |
|-------|-----------|-------|-------|--------|-------|
| VITS | 0.667 | 4.97 | 7.23% | 3.64 | 0.0211 |
| StyleTTS2 | 0.651 | 4.93 | **6.50%** | 3.79 | 0.0174 |
| **MVC** | **0.653** | **4.91** | 6.52% | **3.85** | **0.0169** |

### OOD & Long-form [§5.2, Table 2, Table 3]

- **OOD (Gutenberg 80 utterances)**: MVC MOS-OOD 3.88 vs StyleTTS2 3.87 (几乎相同),VITS 3.21, JETS 3.21 [Table 2]
- ID/OOD 差异近乎为零 (3.87 → 3.88),证明 bidirectional Mamba 泛化到未见句法结构 [§5.2]
- **Long-form (2-6 min Gutenberg)**: MVC MOS 4.16 vs StyleTTS2 3.91, RTF 0.0170 vs 0.0200 [Table 3]
- 长文本优势更明显 (+0.25 MOS),SSM conditioning stack 在多句段/多分钟文本上保持稳定 [论文原文]

### 消融: 组件移除 [§5.4, Table 6]

| Removed | CMOS-N drop (OOD) |
|---------|-------------------|
| Bi-Mamba text encoder | -0.38 |
| Expressive Mamba | **-0.41** |
| Temporal Bi-Mamba | -0.36 |

- Expressive Mamba 对 OOD 韵律最重要 [§5.4]
- 三个模块各贡献不可替代的信息,非冗余 [论文原文]
- Pitch RMSE 增加 0.12-0.18 Hz, duration error 增加 0.6-0.8 frames [§5.4]

### 消融: Fusion & Conditioning [§5.4, Table 8]

| Variant | MOS long ↑ | Pitch RMSE ↓ | RTF ↓ |
|---------|------------|--------------|-------|
| MVC (gated + AdaLN) | **4.16** | **1.92** | **0.0177** |
| Gated only (no AdaLN) | 4.02 | 2.04 | 0.0186 |
| AdaLN only (no gating) | 3.95 | 2.22 | 0.0198 |
| Concat (no gating, no AdaLN) | 3.64 | 2.89 | 0.0216 |

- Gated fusion + AdaLN 缺一不可 [§5.4]
- 纯 concat (现有 Mamba-TTS 做法) → MOS 下降 0.52, pitch RMSE 增加 50% [论文原文]
- 证明仅替换 attention 为 SSM 不够,需要 gating + style modulation [agent解读]

### 编码器深度 [§5.4, Table 7]

- BiLSTM (无 Mamba): MOS 3.61, RTF 0.0268 (最低 MOS, 最高 RTF)
- 2 Mamba layers: 3.65; 4 layers: 3.78; **6 layers (默认)**: **3.87**, RTF 0.0189
- 7 layers: MOS 3.90 但 RTF 增加; 8 layers: MOS 3.88 (过参数化)
- 6 层是 quality-efficiency 最优 [§5.4]

### 运行时分析 [§5, Table 15, Table 9]

| Module | Time (ms) | Proportion |
|--------|-----------|------------|
| Bi-Mamba encoder stack | 42.5 | 31.4% |
| Diffusion decoder | 73.4 | **54.2%** |
| Vocoder | 19.5 | 14.4% |
| Total | 135.4 | 100% |

**Encoder-only comparison (Table 9)**:

| Model | Encoder Params | Speedup ↑ | Peak Memory ↓ |
|-------|---------------|-----------|---------------|
| StyleTTS2 | 42M | 1.00 | 100% |
| Mamba-hybrid | 32M | 1.15 | 86% |
| **MVC** | **21M** | **1.60** | **72%** |

- 编码器参数减半,吞吐提升 1.6x,峰值内存降 28% [Table 9]
- 但 diffusion decoder 占主导延迟,整体 RTF 改善有限 (0.0169 vs 0.0174) [§5]

### Protocol-matched Mamba baselines [Table 12]

与同条件重训的 Mamba 变体对比:

| Model | F0 RMSE | MCD | WER | PESQ | RTF |
|-------|---------|-----|-----|------|-----|
| Hybrid-Mamba (Concat) | 0.659 | 4.95 | 6.68% | 3.79 | 0.0189 |
| Bi-Mamba (Concat-only) | 0.656 | 4.93 | 6.58% | 3.82 | 0.0181 |
| **MVC (gated + AdaLN)** | **0.653** | **4.91** | **6.52%** | **3.85** | **0.0177** |

- 去除 attention (Hybrid→Bi-Mamba) 改善 RTF,gated fusion + AdaLN 进一步改善所有指标 [Table 12]

## 核心设计选择分析

### WHY: 为什么不与 NaturalSpeech 3 / CosyVoice 3 直接对比 [§1, §4.1]

这些系统在数十万至百万小时私有多语言数据上训练,使用 LLM-scale 语义模块,与 MVC 的公开数据 + decoder-matched 设置在数据规模、任务范围和训练基础设施上有本质差异。MVC 将它们视为 contextual references 而非 numeric baselines,在 Appendix F 提供定性对比 [§1, §4.1]

### WHY: 保留 StyleTTS2 diffusion decoder 不变 [§3.5]

1. 隔离 conditioning architecture 的效果: decoder 相同则性能差异只来自编码器 [§3.5]
2. 提供公平对比基础: 所有 baseline 共享同一 mel front-end, diffusion schedule, vocoder [§4.1]
3. Decoder 已是性能瓶颈 (54.2%),改善 encoder 释放更多 GPU 资源给 decoder [Table 15]

### WHY: Gated fusion 而非 simple concatenation [§3.2.1]

- Simple concat 无法根据局部句法上下文调整前向/后向方向的贡献权重 [论文原文]
- Gate statistics 在多分钟文本上保持稳定,不会 collapse [Appendix E.1]
- 消融证明 concat-only → MOS 下降 0.52, pitch 误差增加 50% [Table 8]

### WHY: Training-time aligner 而非推理时保留 [§3.3]

- 保持推理路径全 SSM-only 是核心目标 [§3.3]
- Aligner 对 noise 鲁棒 (±10% 扰动 WER<0.4, MOS drop<0.05),说明推理路径不依赖精确对齐 [Appendix B.7]
- 推理时由 temporal predictor (Conv1D + SSM) 替代对齐功能 [§3.4]

## 与已有 Mamba-TTS 的差异 [Table 10]

| System | Inference attention? | Rhythm/duration | Prosody/style | Fusion | SSM-only? |
|--------|---------------------|-----------------|---------------|--------|-----------|
| StyleTTS2 | Yes (Transformer) | Attn/var. pred. | Ref/style enc (attn) | Attention/concat | No |
| Miyazaki'24 | Hybrid (SSM+attn) | Mixed (keeps attn) | Mixed (keeps attn) | Concat | No |
| Jiang'25 (Slytherin) | Hybrid (SSM+attn) | Attn/var. pred. (attn) | Ref/style enc. (attn) | Concat | No |
| Zhang'24 | Hybrid (SSM+attn) | Mixed (keeps attn) | Mixed (keeps attn) | Concat | No |
| **MVC** | **No (SSM-only)** | **Temporal Bi-Mamba** | **Mamba + AdaLN** | **Gated bi-dir. + AdaLN** | **Yes** |

MVC 是唯一在 text, rhythm, prosody 三个维度全部 SSM-only 的系统 [Table 10]

## 论文贡献与意义

1. **SSM-only conditioning stack 的可行性验证**: 首次证明推理时完全去除 attention 和 recurrence 的 conditioning path 可以匹配甚至超越 transformer baseline [§6]
2. **Gated Bi-Mamba + AdaLN 的编码器设计**: 提出的 gated fusion 和 AdaLN conditioning 组合是长文本韵律稳定性和 F0 tracking 的关键 [Table 8]
3. **Protocol-matched 评估范式**: 所有 baseline 在同一 pipeline 下重训,隔离 conditioning architecture 效果 [§4.1]
4. **模块化 drop-in replacement**: MVC encoder 可作为未来多语言/工业 pipeline 的 conditioning module 替换现有 transformer encoder [§6]

---

检索命中: [[Speaker Embedding]], [[Prosody Modeling]] | 过滤: [[Diffusion-based TTS]](pending-review), [[F0 Modeling]](pending-review), [[Voice Cloning Taxonomy]](pending-review) | 未命中但可能相关: 无
