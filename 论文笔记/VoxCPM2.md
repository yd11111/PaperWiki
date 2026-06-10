---
type: paper
tier: repro
title: "VoxCPM2: Multilingual and Controllable Hierarchical Diffusion-Autoregressive Speech Generation"
arxiv_id: "2606.06928"
source: "Sources/VoxCPM2.pdf"
authors: [VoxCPM Team (OpenBMB)]
year: 2026
venue: "arXiv preprint"
tags: [TTS, hierarchical-modeling, flow-matching, tokenizer-free, semi-discrete, multilingual, controllable, zero-shot, LLM-based, end-to-end]
concepts: ["[[FiniteScalarQuantization]]", "[[ConditionalFlowMatching]]", "[[LLM-basedTTS]]", "[[SemanticvsAcousticTokens]]", "[[SpeechTokenizer]]", "[[Classifier-FreeGuidance]]", "[[VariationalAutoencoderforTTS]]", "[[Next-TokenDiffusion]]"]
models: ["[[模型库/CosyVoice3|CosyVoice 3]]", "[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]", "[[任务库/InstructedSpeechGeneration|Instructed Speech Generation]]", "[[任务库/Cross-lingualVoiceCloning|Cross-lingual Voice Cloning]]"]
datasets: ["[[数据集/SEED-TTS-Eval|SEED-TTS-Eval]]", "[[数据集/Emilia|Emilia]]", "[[数据集/CV3-Eval|CV3-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-10
updated: 2026-06-10
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: VoxCPM2 是 [[论文笔记/VoxCPM|VoxCPM]] 的直接继承者,延续 [[LLM-basedTTS]] 谱系中 "Hybrid 架构 (LLM + Diffusion)" 路线的端到端统一方向。与 VoxCPM 的核心架构(TSLM + [[FiniteScalarQuantization]] 瓶颈 + RALM + LocDiT)保持一致,但在三个维度显著扩展: (1) 能力: 统一 30 语言 + 语音设计 + 可控克隆; (2) 质量: 非对称 AudioVAE V2 (encode@16kHz, decode@48kHz); (3) 规模: 2B 参数 + 2M 小时数据。

**已有认知**:
- [[FiniteScalarQuantization]] [待确认] 在 VoxCPM 中作为内部正则化瓶颈诱导 semantic-acoustic 分工。VoxCPM2 将 FSQ 维度从 256 扩展到 512,适应更大模型和更广语言覆盖。
- [[ConditionalFlowMatching]] 作为 LocDiT 的训练目标,与 VoxCPM 一致。VoxCPM2 的 LocDiT 从 4L 扩展到 12L。
- [[SemanticvsAcousticTokens]]: VoxCPM2 继承了 semi-discrete residual 方案,但引入 concat-projection fusion 替代 element-wise sum,保留更丰富的 FSQ + LocEnc 信息。
- [[Zero-shotSpeechSynthesis]]: VoxCPM2 在 Seed-TTS-Eval 上 WER 1.84/SIM 75.3 (test-EN),属开源系统顶部。
- VoxCPM2 新增 voice design + controllable cloning 能力,对接 [[InstructedSpeechGeneration]] 任务。

**创新判断**: 相比 VoxCPM(0.5B, 中英双语, 16kHz),VoxCPM2 的核心进步在于: (1) 统一序列组织 — 5 种生成模式通过输入 layout 区分,共享同一 backbone; (2) 非对称 AudioVAE V2 实现 encode@16kHz / decode@48kHz 的隐式超分; (3) Reference audio pathway 解耦说话人身份和风格控制。架构层面,concat-projection fusion + multi-token LocDiT prefix + NoPE RALM 是关键改进。

> 检索命中: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechTokenizer]]✓, [[Zero-shotSpeechSynthesis]]✓, [[InstructedSpeechGeneration]]✓ | 过滤: [[FiniteScalarQuantization]](pending-review), [[Classifier-FreeGuidance]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: VoxCPM 的 2B 参数多语言可控版本,通过统一序列组织将 basic TTS / voice design / controllable cloning / continuation cloning 统一在同一个分层连续 latent backbone 中
> - **路线**: Text(BPE) → TSLM(MiniCPM-4-1B, 28L) → FSQ(512-dim) → RALM(8L, NoPE) → LocDiT(12L, multi-token prefix) → AudioVAE V2 decoder(48kHz); optional: Reference audio → VAE encode → delimited prefix segment
> - **指标**: Seed-TTS-Eval EN-WER 1.84% / ZH-CER 0.97% / SIM 75.3/79.5; CV3-Eval hard-zh 8.55% hard-en 8.48%; InstructTTSEval APS 55.0 / DSD 67.5 / RP 78.8 [Table 3, 5]
> - **可借鉴**: (1) 统一序列组织 — 不同任务只是输入 building blocks 的不同排列; (2) concat-projection fusion 替代 sum; (3) Multi-token LocDiT prefix 避免 early information collapse; (4) 非对称 VAE encode@16kHz / decode@48kHz 做隐式超分; (5) NoPE for RALM 提升长序列稳定性
> - **局限**: SIM 不如 dots.tts (75.3 vs 79.2 avg); controllable cloning 的指令遵循精度仍有限(DSD 67.5%); 2M 小时内部数据不可复现

## 核心问题

VoxCPM2 要解决 VoxCPM 从 "概念验证" 到 "实用基础模型" 的三个扩展问题:

1. **能力统一**: 现有可控 TTS 系统通常需要 per-task 的专门模块(style encoder, adapter, mode-specific routing)。VoxCPM2 的目标是用单一 backbone + 统一序列组织替代所有专门组件 [§1.2]。

2. **质量提升**: VoxCPM 仅支持 16kHz 输出。VoxCPM2 需要在不增加 AR 序列长度的前提下提升到 48kHz [§3.2]。

3. **多语言扩展**: 从中英双语扩展到 30 语言 + 9 方言,同时保持 zero-shot cloning 能力 [§1.2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VoxCPM2 继承 VoxCPM 的分层 diffusion-AR 框架,核心生成公式 [§3.1, Eq. 1]:

```
h^FSQ_i = FSQ(TSLM(T, E_{<i}))
h^residual_i = RALM(H^TSLM_text, H^FSQ_{≤i} ⊕ E_{<i})
z_i ~ LocDiT(h^FSQ_i, h^residual_i, z_{i-1}; t)
```

VoxCPM2 相对 VoxCPM 的三处关键修改:
1. LocDiT 接收 h^FSQ 和 h^residual 作为**分离的 prefix tokens**,而非 sum [§3.3.1]
2. RALM 前的 fusion 从 element-wise sum 改为 concat-projection [§3.3.1, Eq. 2]
3. 新增 Reference Audio Pathway 作为可选的 speaker-identity prefix [§3.3.2]

### 关键设计选择

**1. Concat-Projection Fusion**

[论文原文] VoxCPM 用 element-wise sum 合并 h^FSQ 和 LocEnc embedding E_i; VoxCPM2 改为 h^{res_in}_i = W_fuse [h^FSQ_i ∥ E_i],保留两路更丰富的信息 [§3.3.1, Eq. 2]。

[agent 解读] Element-wise sum 强制两路信息投射到相同空间,可能造成信息冲突。Concat-projection 让模型通过学习 W_fuse 自动确定最优组合权重。

**2. Multi-Token LocDiT Conditioning Prefix**

[论文原文] VoxCPM 将 semantic state + residual state + timestep embedding sum 为单个 conditioning token; VoxCPM2 将三者分别投影为独立 prefix tokens [µ_sem, µ_res, µ_t, z_{i-1}, z̃_i] [§3.3.1]。

[agent 解读] 单 token 条件会导致 "early information collapse" — 三路信号在 sum 时就已经混合。分离为 prefix tokens 让 LocDiT 的 attention 自行决定如何组合,提供更高带宽的条件信号。

**3. Wider FSQ (256→512)**

[论文原文] FSQ 维度从 256 扩展到 512,保持 9 levels per dimension [§3.3.1]。

[agent 解读] 更大模型 + 更多语言需要更大的语义表示空间。VoxCPM 消融 [Table 6] 显示 FSQ 维度是 semantic-acoustic 分工粒度的关键参数。

**4. NoPE for RALM**

[论文原文] RALM 移除 RoPE,采用 NoPE 设计 (Kazemnejad et al., 2023)。RALM 主要做局部声学修正,移除位置编码 "reduces overfitting to training lengths and improves long-utterance stability" [§3.3.1]。

**5. Reference Audio Pathway**

[论文原文] 新增显式 reference audio pathway: reference clip 经 AudioVAE V2 编码后作为 [REF_START, REF_END] 分界的 prefix segment 插入序列开头。Thanks to causal attention,后续位置都可以 attend to 这个 segment [§3.3.2]。

[agent 解读] 这与 continuation-based cloning 互补: continuation 需要 reference 的 transcript 对齐,reference pathway 不需要。两者组合 (ref + continuation) 在 Seed-TTS-Eval 上取得最优 SIM [Table 4]。

**6. 统一序列组织**

[论文原文] 5 种生成模式通过 3 种 building blocks (text, reference audio, target audio) 的不同排列实现 [§3.4, Table 2]:
- Basic TTS: `<text> → <target>`
- Voice design: `<(voice desc) text> → <target>`
- Reference cloning: `<ref audio> | <text> → <target>`
- Controllable cloning: `<ref audio> | <(style desc) text> → <target>`
- Continuation cloning: `<prompt_text + text | prompt_audio> → <target>`

[agent 解读] voice design 和 controllable cloning 的自然语言描述直接与 synthesis text 拼接,复用 TSLM 处理,不需要额外 style encoder 或 adapter 模块。

**7. 非对称 AudioVAE V2**

[论文原文] Encoder 输入 16kHz, decoder 输出 48kHz。Encoder: causal CNN, downsample [2,5,8,8] → 640x → 64-dim @ 25Hz。Decoder: 更深更宽 CNN, upsample [8,6,5,2,2,2] → 48kHz [§3.2]。

[agent 解读] 非对称设计的三重好处: (1) 复用 VoxCPM 的 16kHz 训练语料; (2) 消除不同源采样率导致的 latent mismatch; (3) 隐式超分不增加 AR 序列长度。

### 模块细节

#### AudioVAE V2

- **Input:** 16kHz waveform (encoder side)
- **Output:** 48kHz waveform (decoder side); 64-dim latent @ 25Hz
- **Encoder:** Causal CNN, downsample strides [2, 5, 8, 8] → 640x reduction [§3.2]
- **Decoder:** Causal CNN, wider channels, upsample [8, 6, 5, 2, 2, 2]; optional target-sample-rate condition [§3.2]
- **Key params:** d_latent=64, frame_rate=25Hz, supports conditional output rate

#### LocEnc (Local Encoder)

- **Input:** Previous latent patches z_{<i}
- **Output:** Patch-level acoustic history embeddings E_{<i}
- **Structure:** 12L, H=1024 [Table 1]

#### TSLM (Text-Semantic Language Model)

- **Init:** MiniCPM-4-1B [§3.3.3]
- **Structure:** 28L, H=2048 [Table 1]
- **Input:** BPE text tokens + LocEnc embeddings E_{<i}
- **Output:** Text hidden states H^TSLM_text + FSQ-quantized audio hidden states h^FSQ

#### FSQ Bottleneck

- **Dimensionality:** 512 (up from 256) [§3.3.1]
- **Levels:** 9 per dimension
- **Gradient:** STE (Straight-Through Estimator)

#### RALM (Residual Acoustic Language Model)

- **Structure:** 8L, H=2048, **NoPE** (no positional encoding) [§3.3.1, Table 1]
- **Input:** Concat-projection of [h^FSQ ∥ E_i] + TSLM text hidden states [Eq. 2]
- **Output:** h^residual_i

#### LocDiT (Local Diffusion Transformer)

- **Structure:** 12L, H=1024 [Table 1]
- **Input:** Multi-token prefix [µ_sem, µ_res, µ_t, z_{i-1}^{(1..P)}, z̃_i^{(1..P)}] [§3.3.1]
- **Output:** Velocity field → clean latent patch z_i (P=4 frames)
- **CFG:** 10% drop rate for LM conditioning during training; α=2.0 at inference [§3.5, 3.7]

### 训练策略

#### Loss 设计

- **Flow matching loss:** Patch-level conditional FM loss on target latent patches [§3.5]
- **Stop loss:** Binary stop prediction on TSLM-FSQ hidden states [§3.5]
- **Loss masking:** Only on target-audio segment; text/reference segments excluded

#### 训练配置

**总参数:** ~2B [Table 1]
**Token rate:** 6.25Hz (patch size P=4, 25Hz/4) [§3.3.3]
**Max sequence length:** 8192 (up from 4096) [Table 1]
**Optimizer:** AdamW, cosine lr decay with linear warmup [§3.5]

**三阶段渐进课程** [§3.5]:
1. **Multilingual TTS pretraining:** Large-scale <transcription, audio> pairs; max 60s audio; seq_len ≤ 4096; 30 languages
2. **Joint TTS + controllable pretraining:** +controllable data at increasing ratio; +reference audio triplets; max 3min audio; seq_len ≤ 8192
3. **HQ annealing SFT:** Curated high-quality + expressive subset; higher controllable ratio; balanced language sampling; lr annealing; 2s-5min samples

#### 数据处理

- **总量:** >2M hours multilingual speech [§3.6]
- **语言:** 30 languages + 9 Chinese dialects; Chinese/English majority [§3.6]
- **可控数据:** Tens of thousands hours open-source expressive + internally annotated [§3.6]
- **Voice design 标注:** Step-Audio R1 + Gemini 2.5 Pro 生成 NL descriptions (gender, age, emotion, delivery, environment) [§3.6]
- **Controllable cloning 数据:** Mining same-speaker references (cosine sim > 0.7, exclude preceding clips) [§3.6]
- **Content-style decoupling:** Model self-synthesis: clone voice+style onto unrelated transcript,打破 style-content 相关性 [§3.6]

### 推理流程

```
1. Assemble input sequence based on mode:
   - Basic TTS: [text tokens] → [target audio]
   - Reference cloning: [REF_START, ref_latent, REF_END] | [text] → [target]
   - Controllable: [REF_START, ref, REF_END] | [(style desc) text] → [target]
2. Autoregressive loop at 6.25Hz:
   a. LocEnc encodes history z_{<i} → E_{<i}
   b. TSLM processes text + E_{<i} → FSQ bottleneck → h^FSQ_i
   c. RALM: concat-projection [h^FSQ ∥ E_i] + text hidden → h^residual_i
   d. LocDiT: prefix [µ_sem, µ_res, µ_t, z_{i-1}, z̃_i] → flow matching → z_i
   e. Stop predictor checks EOS on TSLM-FSQ hidden
3. All patches → AudioVAE V2 decoder → 48kHz waveform
```

**推理技巧** [§3.7]:
- CFG: α=2.0 (practical range 1.5-3.0)
- Sway sampling: 更多 ODE 步分配到高噪声区域
- CFG-Zero*: 减少早期步骤 artifacts
- Streaming: Causal TSLM/RALM + patch-local LocDiT/LocEnc 支持逐 patch 流式

## 实验

### 主实验: Seed-TTS-Eval

| 指标 | VoxCPM2 (2B) | VoxCPM (0.6B) | CosyVoice 3 (1.5B, closed) | dots.tts (2B) | Qwen3-TTS (1.7B) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| EN-WER ↓ | 1.84% | 1.85% | 2.22% | **1.30%** | 1.23% | [Table 3] |
| EN-SIM ↑ | 75.3 | 72.9 | 72.0 | **77.1** | 71.7 | [Table 3] |
| ZH-CER ↓ | 0.97% | 0.93% | 1.12% | **0.94%** | 1.22% | [Table 3] |
| ZH-SIM ↑ | 79.5 | 77.2 | 78.1 | **81.0** | 77.0 | [Table 3] |
| Hard-CER ↓ | 8.13% | 8.87% | **5.83%** | 6.60% | 6.76% | [Table 3] |
| Hard-SIM ↑ | 75.3 | 73.0 | 75.8 | **79.5** | 74.8 | [Table 3] |

### Inference Recipes Ablation

| Recipe | EN-WER ↓ | EN-SIM ↑ | ZH-CER ↓ | ZH-SIM ↑ | Hard-SIM ↑ | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Continuation only | 1.01% | 77.7 | 1.97% | 72.6 | 72.4 | [Table 4] |
| Reference only | 1.10% | 67.0 | **1.81%** | 67.0 | 70.0 | [Table 4] |
| Ref + Continuation | **0.99%** | **79.5** | 1.94% | **75.2** | **74.9** | [Table 4] |

注: Table 3 报告的 VoxCPM2 数据使用 Ref+Continuation recipe; Table 4 中的 Ref+Continuation EN-WER 0.99% 低于 Table 3 的 1.84%,因为 Table 4 使用不同 checkpoint 或评测设置 [agent 解读]。

### Multilingual (CV3-Eval, 9 languages)

| 指标 | VoxCPM2 | Fish Audio S2 | CosyVoice 3 | 出处 |
| --- | --- | --- | --- | --- |
| zh CER ↓ | 3.65% | **2.65%** | 3.91% | [Table 5] |
| en WER ↓ | 5.00% | **2.43%** | 4.99% | [Table 5] |
| hard-zh CER ↓ | **8.55%** | 9.10% | 9.77% | [Table 5] |
| hard-en WER ↓ | **8.48%** | — | 10.55% | [Table 5] |

### Controllable Generation (InstructTTSEval)

| Subtask | VoxCPM2 | CosyVoice 3 | Qwen3-TTS | Fish Audio S2 | 出处 |
| --- | --- | --- | --- | --- | --- |
| APS (acoustic params) | 55.0 | 51.7 | 72.3 | **83.5** | [Table 9] |
| DSD (descriptive style) | 67.5 | 73.0 | **80.0** | 73.4 | [Table 9] |
| RP (role-play) | 78.8 | 73.8 | **82.8** | 76.6 | [Table 9] |

### AudioVAE V2 Reconstruction

| 模型 | Sample Rate | UTMOS ↑ | SIM ↑ | WER/CER ↓ | 出处 |
| --- | --- | --- | --- | --- | --- |
| VoxCPM VAE (原版) | 16k→16k | 3.74 | 0.940 | 2.74% | [Table 10] |
| VoxCPM2 VAE V2 | 16k→48k | **4.22** | **0.964** | **1.32%** | [Table 10] |

### VoxCPM Family Evolution

| 指标 | VoxCPM (0.6B) | VoxCPM1.5 (0.8B) | VoxCPM2 (2B) | 出处 |
| --- | --- | --- | --- | --- |
| EN-WER | 1.85% | 2.12% | **1.84%** | [Table 3] |
| EN-SIM | 72.9 | 71.4 | **75.3** | [Table 3] |
| ZH-SIM | 77.2 | 77.0 | **79.5** | [Table 3] |
| Patch size P | 2 | 4 | 4 | [Table 1] |
| Token rate | 12.5Hz | 6.25Hz | 6.25Hz | [Table 1] |
| Output SR | 16kHz | 44.1kHz | 48kHz | [Table 1] |

## 复现要点

1. **AudioVAE V2 的非对称设计** [§3.2]: Encoder 必须限制在 16kHz 输入; decoder 使用更深更宽架构支持 48kHz 输出。Downsample strides [2,5,8,8] (encoder) vs upsample [8,6,5,2,2,2] (decoder) 不对称。
2. **Concat-projection 替代 sum** [§3.3.1]: RALM 前的 fusion 必须用 [h^FSQ ∥ E_i] 再乘 W_fuse,不能用 element-wise sum。维度变化: 2d_hidden → d_hidden。
3. **Multi-token LocDiT prefix** [§3.3.1]: µ_sem, µ_res, µ_t 三个独立 prefix tokens,不能 sum 成一个 token。LocDiT 在 prefix + prev_patch + noisy_patch 上做 full attention。
4. **FSQ 维度 512** [§3.3.1]: 与 VoxCPM 的 256 不同。Levels 仍为 9。
5. **RALM 不加 RoPE** [§3.3.1]: NoPE 设计对长序列稳定性关键。TSLM 保留 RoPE。
6. **Reference audio 的 REF_START/REF_END 分界** [§3.3.2]: Reference segment 不参与 loss,纯做 conditioning context。与 continuation cloning 的 prompt audio 不同 — continuation prompt 是 target audio 的 prefix,结构上属于 target segment。
7. **三阶段渐进课程的 sequence length 控制** [§3.5]: Stage 1: max 60s, seq_len 4096; Stage 2+3: max 3-5min, seq_len 8192。Stage 2 中 controllable data 比例逐步增加。
8. **Content-style decoupling 数据** [§3.6]: 用模型自身克隆 voice+style 到不相关 transcript,打破 style-content correlation。这类合成数据主要用于 Stage 2,Stage 3 限制为 natively recorded speech。
9. **CFG drop rate 10%** [§3.5]: 训练时以 10% 概率 drop LM conditioning to LocDiT。推理 CFG α=2.0。
10. **Sway sampling + CFG-Zero*** [§3.7]: 两者默认启用。Sway sampling 分配更多 ODE 步到高噪声区域; CFG-Zero* 减少早期步骤 artifacts。

## 局限性

1. **SIM 与 dots.tts 有差距**: Seed-TTS-Eval avg SIM 76.7 vs dots.tts 79.2。VoxCPM2 的 2B 参数和 >2M 小时数据仍未弥合这一差距 [Table 3]。
2. **Controllable 精度有限**: InstructTTSEval APS 55.0% 明显低于 Fish Audio S2 (83.5%) 和 Qwen3-TTS (72.3%),表明自然语言控制尚未精确 [Table 9]。
3. **数据不可复现**: >2M 小时内部数据不公开,仅公开模型权重。
4. **无后训练优化**: 与 dots.tts (SOAR + MeanFlow) 不同,VoxCPM2 未采用任何后训练策略,推理仍需多步 ODE。
5. **低资源语言覆盖**: 28 个非中英语言从 1K-50K 小时不等,质量参差 [§3.6]。

## 点评

**优点**:
1. **统一序列组织**: 5 种模式共享同一 backbone,无额外模块,是系统设计上的优雅方案。
2. **非对称 VAE 的实用性**: encode@16kHz 保持与旧数据兼容 + 紧凑 latent,decode@48kHz 提升输出质量。
3. **从 VoxCPM 的演进路径清晰**: 每个改进都有明确的动机和消融(虽然消融报告在论文中相对简略)。
4. **开源完整度**: Apache 2.0,含 model weights + fine-tuning code + inference tools。

**不足**:
1. **消融实验不充分**: 论文缺少关键改进(concat-projection, multi-token prefix, NoPE, FSQ 512)的独立消融。VoxCPM 原文中的消融对这些新改进不适用。
2. **与 dots.tts 的 SIM 差距未解释**: 两者参数量相同(2B),数据量 VoxCPM2 更大(>2M vs 1.5M h),但 SIM 差 2.5 个点。架构差异(FSQ bottleneck vs semantic encoder feedback)对 speaker similarity 的影响值得深入分析。
3. **Controllable 评估有限**: InstructTTSEval 只有 3 个子任务,缺少更细粒度的可控性分析。

## 可复用的 idea

1. **统一序列组织模式**: 不同任务 = 不同 building block arrangement。适用于任何多模态多任务 seq2seq 系统。
2. **非对称 VAE (encode@low-SR, decode@high-SR)**: 在不增加生成序列长度的前提下提升输出质量。可迁移到音频/视频生成。
3. **Multi-token conditioning prefix**: 将多路条件信号分离为独立 prefix tokens,避免 early information collapse。适用于任何 condition-based diffusion/flow model。
4. **NoPE for local-refinement modules**: 对主要做局部处理(非全局规划)的模块移除位置编码,提升 length generalization。
5. **Content-style decoupling via self-synthesis**: 用模型自身将 style 克隆到不相关 content,打破数据中的 style-content correlation。适用于任何可控生成的数据增强。

## 审阅

> [!review] 审阅结论: pass (2026-06-10)
> - **conclusion**: pass
> - **issues**: 0 (0 high, 0 medium, 0 low)
> - 详见 `_review/VoxCPM2-review.yml`
