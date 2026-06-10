---
type: paper
tier: repro
title: "dots.tts: A 2B-Parameter Continuous Autoregressive TTS Foundation Model"
arxiv_id: "2606.07080"
source: "Sources/dots.tts.pdf"
authors: [dots.tts Team (Rednote)]
year: 2026
venue: "arXiv preprint"
tags: [TTS, continuous-AR, flow-matching, self-corrective-alignment, MeanFlow-distillation, zero-shot, end-to-end, LLM-based, multilingual]
concepts: ["[[ConditionalFlowMatching]]", "[[LLM-basedTTS]]", "[[SemanticvsAcousticTokens]]", "[[SpeechTokenizer]]", "[[Classifier-FreeGuidance]]", "[[VariationalAutoencoderforTTS]]"]
models: ["[[模型库/CosyVoice3|CosyVoice 3]]", "[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: ["[[数据集/SEED-TTS-Eval|SEED-TTS-Eval]]", "[[数据集/Emilia|Emilia]]", "[[数据集/CV3-Eval|CV3-Eval]]"]
kb_context_sources: 5
status: draft
created: 2026-06-10
updated: 2026-06-10
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 1 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: dots.tts 位于 [[LLM-basedTTS]] 谱系中 "连续自回归" 路线的最前沿。与离散 token AR (CosyVoice, IndexTTS2) 和混合离散-连续 (VoxCPM) 不同,dots.tts 沿续 DiTAR/VibeVoice/ARDiT 的纯连续路线,但通过三项创新解决了连续 AR 的核心问题——误差累积: (1) 语义结构化的 AudioVAE; (2) 全历史条件的 AR-FM head; (3) 奖励免费的 SOAR 自修正后训练。

**已有认知**:
- [[ConditionalFlowMatching]] 在 CosyVoice 系列中作为独立 second-stage renderer; dots.tts 将 FM 用于 AR-FM head 的每步 patch 生成,但保留了 LLM 作为全局语义规划模块。
- [[SemanticvsAcousticTokens]] 的核心 trade-off: 离散稳定但损失保真度,连续保保真度但语义纠缠。dots.tts 的解决方案是语义 encoder 做 4x 下采样,只将语义摘要(非 raw latent)反馈给 LLM,实现隐式解耦。
- [[SpeechTokenizer]] 在离散路线中是必需组件; dots.tts 的 AudioVAE 不做量化,直接产生连续 latent,消除了 tokenizer 瓶颈。
- [[Zero-shotSpeechSynthesis]] 当前评测基准: Seed-TTS-Eval 上 dots.tts(SOAR) avg WER 2.95% / SIM 79.2,超越所有已报告系统。

**创新判断**: 相比 VoxCPM 用 FSQ 做内部正则化瓶颈,dots.tts 选择了不同的分离策略 — 语义 encoder 做信息瓶颈(4x 下采样 + WavLM 对齐),LLM 只看语义摘要不看 raw latent,FM head 接收 full history 做局部渲染。SOAR 自修正和 MeanFlow 蒸馏是在连续 AR TTS 领域的首次应用。

> 检索命中: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechTokenizer]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[Classifier-FreeGuidance]](pending-review), [[VariationalAutoencoderforTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 2B 参数连续 AR TTS,通过语义 AudioVAE + 全历史条件 AR-FM head + SOAR 自修正后训练三项设计,在 Seed-TTS-Eval 上取得所有已报告系统的最优平均性能
> - **路线**: Text(BPE) → LLM(Qwen2.5-1.5B, 6.25Hz) → AR-FM head(18L DiT, full-history conditioning) → 25Hz VAE latent patches → Semantic Encoder(24L, 4x downsample) → feedback to LLM; AudioVAE decoder(BigVGAN) → 48kHz waveform
> - **指标**: Seed-TTS-Eval avg WER 2.95% / SIM 79.2 (SOAR); MF NFE=4 avg WER 2.94% / SIM 78.2; MiniMax 24-lang avg SIM 83.9 [Table 2, 3]
> - **可借鉴**: (1) SOAR 奖励免费自修正后训练 — 用模型自己的 off-trajectory 状态做监督,不需要 reward model; (2) CFG-aware MeanFlow 蒸馏 — 将 CFG 融入蒸馏目标,推理时只需 1 次 forward; (3) Block-causal 并行训练让 AR-FM 的训练 forward 与逐步推理数值一致; (4) 语义 encoder 做 LLM 反馈的信息瓶颈,防止 acoustic detail 干扰全局规划
> - **局限**: 多语言 WER 被低资源语言拖累(BPE 覆盖不足); 1.5M 小时内部数据不可复现; 开源 Emilia-only 结果有差距; 无显式韵律/情感控制

## 核心问题

连续自回归 TTS 的核心瓶颈是 **long-range error accumulation** [§1]:

1. **离散 token 路线有量化缓冲**: codec 将不完美预测 snap 到有效声学配置,阻止误差传播。但量化不可逆地丢弃声学细节,限制了表达力天花板。
2. **纯连续路线无此缓冲**: 每个小的预测误差被 decoder 忠实重建并反馈到下一步 AR 条件,误差逐步累积导致长序列崩溃。

dots.tts 的核心假设: 通过三个互补设计缓解连续 AR 的不稳定性 — (1) 语义结构化的 latent space 降低预测难度; (2) full-history conditioning 保持长程一致性; (3) SOAR 后训练修正多步 ODE 推理与单步训练的 mismatch [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

dots.tts 由两个解耦网络组成: 冻结的 AudioVAE + 可训练的自回归 backbone [§2.1, Fig 1]:

```
Text (BPE) ─┐
             ├─→ LLM (Qwen2.5-1.5B, 6.25Hz) ─→ H_n (hidden state)
Semantic     │                                         │
Embedding ───┘                                    AR-FM Head (18L DiT)
   ↑                                                   │
   │  Semantic Encoder (24L, 4x downsample)      4-frame VAE latent patch P_n
   │         ↑                                         │
   └─────── P_n ──────────────────────────── AudioVAE Decoder → 48kHz waveform
```

- AudioVAE: 48kHz → 128-dim @ 25Hz (1920x downsampling), BigVGAN decoder [§2.2]
- LLM: Qwen2.5-1.5B init, BPE text + 6.25Hz audio-semantic embeddings [§2.3]
- Semantic Encoder: 24L causal Transformer, 4x downsampling (25Hz→6.25Hz) [§2.4]
- AR-FM Head: 18L DiT, full-history conditioning, block-causal attention [§2.5]
- Patch size = 4 frames @ 25Hz → LLM token rate 6.25Hz [§2.5.1]

### 关键设计选择

**1. AudioVAE 的两阶段训练**

[论文原文] Stage 1 纯重建 (multi-period + sub-band CQT adversarial + mel + feature matching + KL + flow regularization)。Stage 2 加 WavLM 对齐 + 多任务下游 (ASR + emotion + speaker classification),使 latent 既 reconstructive 又 semantically structured [§2.2]。

[agent 解读] Stage 2 的多任务下游训练是 HoliTok 式 recipe 的关键: 它让 latent 对 downstream LLM "友好" — 不再充斥声学随机变化,而是按语义组织。Stage 2 的 encoder 部分被直接复用为 backbone 的 semantic encoder,实现了 VAE 训练和 backbone 训练之间的组件共享。

**2. LLM 只看语义摘要,不看 raw latent**

[论文原文] "The LLM sees only this semantic summary, not the raw VAE latent. We found this necessary to keep continuous-AR rollouts stable" [§2.1]。

[agent 解读] 这是 dots.tts 与 DiTAR/VoxCPM 的关键区别。DiTAR 让 LM 直接条件于 raw latent,导致语义规划和声学渲染纠缠。dots.tts 通过 semantic encoder 的 4x 下采样强制信息瓶颈: 只有 "活过压缩" 的语义信息才能反馈给 LLM,声学细节被隔离在 AR-FM head 的 full-history context 中。

**3. AR-FM Head 的 Full-History Conditioning**

[论文原文] FM head 的 context 序列为 [H_0, P_0, H_1, P_1, ..., H_{n-1}, P_{n-1}, H_n, Z_n],每步接收所有历史 clean patches P_{<n} 和 LLM hidden states H_{≤n} [§2.5.1, Eq. 1]。

[agent 解读] 与 DiTAR 的 LocDiT(只看前一 patch)不同,dots.tts 的 FM head 看到完整生成历史。代价是 attention 随序列增长,但带来两个好处: (1) 长程声学一致性(不依赖 LM 传递声学信息); (2) 更好的 CFG 效果(两路 CFG: text content + speaker timbre)。

**4. Block-Causal 并行训练**

[论文原文] 训练时将所有 N patches 构建为 Cause(C) + Generation(Z) 两半序列,共享 position indices。Block-causal attention mask 使并行训练的每个 patch 看到的 context 与逐步推理数值一致 [§2.5.2, Fig 2]。

[agent 解读] 这解决了 AR-FM 训练效率问题: 无需 N 次 forward pass,一次 pass 训练所有 patches。Position indices 的 reset 技巧保证 RoPE phases 在训练和推理之间完全匹配。

**5. SOAR 自修正后训练**

[论文原文] 基于 SOAR 的奖励免费自修正: 从当前时步做一步 detached Euler rollout(使用 CFG),到达 off-trajectory state,然后 re-noise 这个 state,让模型学习将 off-trajectory 修正回 clean endpoint [§2.6.3, Eq. 7-12]。

[agent 解读] SOAR 的核心 insight: 预训练时模型只见过 on-trajectory states(从 ground truth 插值),但推理时每步 ODE 积分的误差会使状态偏离训练分布。SOAR 通过暴露模型于自己的 off-trajectory 误差来 bridge 这个 gap,且不需要 reward model。

**6. CFG-Aware MeanFlow 蒸馏**

[论文原文] 冻结 SOAR-corrected DiT 作 teacher,训练 student 预测 CFG-guided mean velocity over time interval [t_a, t_b]。因为 CFG 已融入 teacher target,student 推理时只需 1 次 conditional forward per step [§2.6.3, Eq. 13-15]。

[agent 解读] 传统 CFG 需要 2 次 forward(conditional + unconditional),MeanFlow 将 CFG 融入蒸馏让 student 用 1 次 forward 匹配 CFG-guided 行为,同时将步数压缩到 2-4 NFE。

### 模块细节

#### AudioVAE

- **Input:** 48kHz mono waveform
- **Output:** 128-dim latent @ 25Hz (1920x compression)
- **Encoder:** Causal CNN, downsampling strides [2, 2, 2, 4, 6, 10], posterior projection → mean + log-var [§2.2]
- **Decoder:** Causal BigVGAN-v2 variant
- **Training:** Stage 1: 500K steps, 9.6s crops; Stage 2: 200K steps, +WavLM (23rd layer) alignment + multitask [§3.2.1]
- **Regularization:** KL + flow prior on latent [§2.2]
- **Key params:** AdamW β=(0.8, 0.99), ε=1e-6, lr 1e-4→1e-6 exponential decay

#### LLM Backbone

- **Init:** Qwen2.5-1.5B Base [§2.3]
- **Input:** BPE text tokens + 6.25Hz audio-semantic embeddings
- **Output:** Hidden states H_n (one per audio position, 6.25Hz)
- **Sequence layouts:** Plain mode (text prefix → audio) or 1T1A interleaved (text-audio alternating) [§2.3.1]
- **Training:** End-to-end through FM gradient; text positions carry no loss

#### Semantic Encoder

- **Input:** 25Hz VAE latent patches (4 frames each)
- **Output:** 6.25Hz embeddings (4x downsample, projected to LLM dim)
- **Structure:** Strided causal-conv projector (2x) → 24L causal Transformer (H=1024, FFN=4096) → temporal grouping + linear projection [§2.4]
- **Origin:** Pretrained in AudioVAE Stage 2, transplanted with frozen weights initially

#### AR-FM Head (DiT)

- **Input:** Sequence [H_0, P_0, ..., H_{n-1}, P_{n-1}, H_n, Z_n] + speaker x-vector (CAM++ encoder, adaLN-zero)
- **Output:** Velocity field → integrated to produce clean patch P_n (4 frames @ 25Hz)
- **Structure:** 18L DiT, H=1024, FFN=4096, RoPE + RMSNorm + QK-norm + adaLN-zero [§2.5.1]
- **CFG:** p_drop=0.5 for both LM conditioning and speaker stream; CFG scale γ=1.2 [§2.5.1]
- **Inference:** 10 Euler steps with CFG (NFE=20 effective) for Pretrain/SOAR; MF student: 2-4 NFE, no CFG [§3.3.1]

### 训练策略

#### Loss 设计

- **Flow matching loss:** L_FM = MSE between predicted velocity and rectified-flow target (Eq. 3) [§2.6.2]
- **Stop loss:** Balanced binary CE on detached LLM hidden states (Eq. 4) [§2.6.2]
- **Total pretrain:** L_pre = L_FM + L_eos (equal weight) [§2.6.2]
- **SOAR:** L_soar = (Σ ℓ_on + λ_aux Σ ℓ_aux) / (B + λ_aux M_A), λ_aux=1.0 [§2.6.3, Eq. 12]
- **MeanFlow:** L_mv with adaptive per-sample weight w_mv = (sg(ℓ_mv) + ε)^{-1/2} [§2.6.3, Eq. 14-15]

#### 训练配置

**Pretraining (3 stages)** [§3.2.2]:
- **Optimizer:** AdamW, WSD schedule, peak lr=2e-4, warmup over first 1% of each stage
- **Stage 1 (Modality alignment):** 100K steps, LLM frozen, only semantic encoder + AR-FM; Emilia only; batch ~0.5h audio
- **Stage 2 (General training):** 700K steps, all modules unfrozen, full 1.5M h data; batch ~8h audio; 4 epochs
- **Stage 3 (Annealing):** 100K steps, higher-quality filtered subset; lr decay 2e-4→3e-5; ~1 epoch

**SOAR post-training** [§3.2.3]:
- Only DiT in AR-FM head trainable; 50K steps; batch 4h audio; lr 3e-5 → 2e-6 cosine; λ_aux=1.0; γ_soar=1.2; K_aux=6

**MeanFlow distillation** [§3.2.4]:
- Teacher: frozen SOAR-corrected DiT; Student: same backbone + duration embedder; 50K steps; batch 8h; lr 1e-4 → 2e-6; 16-step Euler teacher; anchor mixing prob 0.5

#### 数据处理

- **总量:** ~1.5M hours (1.2M in-house CN/EN + 300K open-source + 7K caption-paired) [§3.1]
- **预处理:** Vocal enhancement → source separation → speaker-aware diarization → language-routed ASR (Whisper-Large-v3 / Paraformer) [§3.1]
- **过滤:** Cross-ASR consistency + effective-bandwidth + UTMOS + intra-clip x-vector variance [§3.1]
- **Open-source corpora:** Emilia, LibriTTS-R, HiFi-TTS, HiFi-TTS-2, WenetSpeech4TTS, AISHELL-3, MLS, MSR-86K, IndicVoices-R, EuroSpeech, FLEURS (~300K h) [§3.1]

### 推理流程

```
1. Text → BPE tokenization → LLM prefix
2. Loop for each audio step n:
   a. LLM produces H_n from text + semantic embeddings E_{<n}
   b. AR-FM head: construct context [H_0, P_0, ..., H_n, Z_n]
   c. Run ODE solver (10 Euler steps + CFG γ=1.2, or MF 2-4 NFE)
   d. Obtain clean patch P_n (4 frames @ 25Hz)
   e. P_n → Semantic Encoder → new 6.25Hz embedding → feed back to LLM
   f. Stop predictor checks EOS
3. All patches → AudioVAE decoder → 48kHz waveform
```

**Streaming**: 1T1A interleaved mode — text token 和 audio step 交替,支持边生成文本边合成语音。First-packet latency: 85ms (plain) / 54ms (interleaved) with MF NFE=4 [§3.4]。

## 实验

### 主实验: Seed-TTS-Eval

| 指标 | dots.tts (SOAR) | dots.tts (MF4) | CosyVoice 3 | VoxCPM 2 | Seed-TTS | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| test-en WER ↓ | **1.30%** | 1.29% | 2.22% | 1.84% | 2.25% | [Table 2] |
| test-en SIM ↑ | **77.1** | 76.2 | 72.0 | 75.3 | 76.2 | [Table 2] |
| test-zh CER ↓ | **0.94%** | 0.94% | 1.12% | 0.97% | 1.12% | [Table 2] |
| test-zh SIM ↑ | **81.0** | 80.0 | 78.1 | 79.5 | 79.6 | [Table 2] |
| test-zh-hard CER ↓ | 6.60% | 6.60% | **5.83%** | 8.13% | 7.59% | [Table 2] |
| avg WER ↓ | **2.95%** | 2.94% | 3.06% | 3.65% | 3.65% | [Table 2] |
| avg SIM ↑ | **79.2** | 78.2 | 75.3 | 76.7 | 77.8 | [Table 2] |

### MiniMax Multilingual (24 Languages)

| 指标 | dots.tts (SOAR) | VoxCPM 2 | MiniMax | Fish-Audio S2 | 出处 |
| --- | --- | --- | --- | --- | --- |
| avg SIM ↑ | **83.9** | 82.3 | 76.6 | 78.0 | [Table 3] |
| avg WER (%) | 6.8 | 5.7 | **2.8** | 3.7 | [Table 3] |
| Per-lang SIM leads | 19/24 | — | — | — | [Table 3] |

注: dots.tts avg WER 被 Arabic (36%) 等低资源语言拖高,SIM 仍处顶带。BPE coverage 不足是主因 [§3.3.4]。

### CV3-Eval

| 指标 | dots.tts (SOAR) | dots.tts (MF4) | CosyVoice 3 | Fish-Audio S2 | 出处 |
| --- | --- | --- | --- | --- | --- |
| zh CER ↓ | 3.71% | 3.95% | 3.91% | **2.65%** | [Table 4] |
| en WER ↓ | 4.50% | 4.05% | 4.99% | **2.43%** | [Table 4] |
| cross-lingual en→zh ↓ | **4.49%** | 4.37% | — | — | [Table 4] |
| cross-lingual SIM ↑ | **75.0** | 73.8 | 66.9 | — | [Table 4] |

### AudioVAE Reconstruction

| 模型 | FPS | PESQ-WB ↑ | STOI ↑ | SIM ↑ | WER ↓ | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| dots.tts VAE (48kHz) | 25 | 3.95 | 0.973 | 0.969 | 4.14% | [Table 1] |
| MingTok-Audio | 50 | 4.12 | 0.981 | 0.950 | 4.27% | [Table 1] |
| SAC (discrete) | 62.5 | 2.39 | 0.90 | 0.85 | 5.77% | [Table 1] |

### 后训练效果

| Stage | avg WER ↓ | avg SIM ↑ | 出处 |
| --- | --- | --- | --- |
| Pretrain | 2.92% | 78.8 | [Table 2] |
| SOAR | 2.95% | **79.2** | [Table 2] |
| MF NFE=4 | **2.94%** | 78.2 | [Table 2] |
| MF NFE=2 | 3.43% | 77.0 | [Table 2] |

SOAR 主要提升 SIM (+0.4),MF NFE=4 保持 WER 接近 SOAR 水平,SIM 代价约 1 点 [Table 2]。

### 推理效率

| 配置 | NFE | First-packet latency | RTF | 出处 |
| --- | --- | --- | --- | --- |
| Pretrain/SOAR (10 Euler + CFG) | 20 (effective) | — | — | [§3.3.1] |
| MF NFE=4, plain | 4 | 85ms | 0.231 | [§3.4] |
| MF NFE=4, interleaved | 4 | 54ms | 0.245 | [§3.4] |

## 复现要点

1. **AudioVAE 两阶段训练是前提** [§2.2]: Stage 2 的 WavLM 对齐 + 多任务下游至关重要 — 未经 Stage 2 的 latent "reconstructive but not learnable",下游 LLM 难以生成。复现时需冻结 WavLM-Large 作 teacher,对齐目标是 23rd layer hidden representation。
2. **Pretraining Stage 1 必须用 Emilia 子集** [§3.2.2]: 全数据混合训练 Stage 1 "severely unstable"。需先冻结 LLM,只训 semantic encoder + AR-FM head 100K 步建立 modality channel。
3. **Block-causal attention 的 position reset** [§2.5.2]: Z 半序列的 RoPE positions 必须从 C 半序列对应位置重新开始,否则训练-推理不一致。这是实现 parallel training 的核心技巧。
4. **CFG 的双路 drop** [§2.5.1]: LM conditioning 和 speaker stream 各自独立以 p=0.5 drop,但推理时 CFG 是 joint (conditional vs both-dropped),不是分别外推。
5. **SOAR 的 detached rollout** [§3.2.3]: 一步 Euler rollout 必须 stop-gradient,否则会破坏 on-trajectory 训练的稳定性。K_aux=6 个 auxiliary states,通过 re-noising off-trajectory state 生成。
6. **MeanFlow 的 anchor mixing** [§3.2.4]: p=0.5 概率混入 anchor samples (interval 端点对齐 ground truth),这对优化稳定性关键。Student 需额外添加 duration embedder。
7. **Semantic encoder 复用** [§2.4]: 必须从 AudioVAE Stage 2 的下游 encoder 直接移植,保留预训练权重。这个 encoder 在 backbone 训练中参与梯度更新。
8. **1T1A interleaved mode** [§2.3.1]: 需实现 EOT marker 逻辑 — text stream 结束后 audio 继续独立生成直到 stop head fires。
9. **数据量要求** [§3.1, Table 6 (inference from multilingual results)]: 1.5M 小时是完整性能的前提; Emilia 95K 小时可复现基础能力但在 SIM 上会有明显差距。
10. **float32 + torch.compile disabled** [§3.3.1]: 评测时明确使用 float32,未启用 torch.compile。

## 局限性

1. **低资源语言 WER 高**: MiniMax 24-lang 平均 WER 6.8%,被 Arabic (36%) 等语言拖高。BPE tokenization 对低资源语言覆盖不足 [§3.3.4]。
2. **数据不可复现**: 1.2M 小时内部数据不公开。开源 Emilia-only (300K h) 的结果未单独报告,但从 multilingual coverage 推断会有显著差距。
3. **无显式可控性**: 仅支持 caption-style 自然语言描述,无细粒度韵律/情感/语速控制 [agent 解读]。
4. **SOAR 和 MeanFlow 的额外训练成本**: SOAR 50K + MF 50K 步,额外 ~100K 步训练。
5. **推理成本**: Pretrain/SOAR 模式 NFE=20 (10 Euler + CFG 双 forward),仅 MF 版本适合实时部署。

## 点评

**优点**:
1. **后训练创新**: SOAR 自修正 + MeanFlow 蒸馏的组合在连续 AR TTS 领域是首次。SOAR 解决 train-inference mismatch 的思路优雅(无 reward model,纯 self-supervised),MeanFlow 将 CFG 融入蒸馏消除推理时双 forward 开销。
2. **工程完整度极高**: AudioVAE 两阶段 + backbone 三阶段 + SOAR + MeanFlow 共 6 个训练阶段,每个都有清晰的 recipe。开源 Apache 2.0 含全部 checkpoint。
3. **Seed-TTS-Eval SOTA**: avg WER 2.95% / SIM 79.2 超越所有已报告系统(含闭源)。
4. **block-causal parallel training** 使 AR-FM 训练效率可行。

**不足**:
1. **低资源多语言**: WER 差距大(Arabic 36% vs MiniMax 1.67%),BPE 路线的固有限制。
2. **消融不够充分**: 没有专门的 full-history vs local-history AR-FM 对比; 没有 semantic encoder 信息瓶颈的消融(如去掉 4x downsample 的效果)。
3. **与 VoxCPM 的公平对比有限**: 虽然 Table 2 含 VoxCPM 2,但两者参数量相同(2B)却用了不同量级的数据(1.5M vs 2M h),数据规模混杂了模型架构的对比。

## 可复用的 idea

1. **SOAR 自修正后训练**: 适用于任何基于 ODE/SDE 的生成模型。核心 recipe: 单步 detached rollout → re-noise → supervised correction。不需要 reward model,直接 bridge train-inference gap。
2. **CFG-aware MeanFlow 蒸馏**: 将 CFG 融入蒸馏 target,student 1 次 forward 实现 teacher 2 次 forward 的效果。可推广到所有使用 CFG 的 diffusion/flow 模型。
3. **语义 encoder 做反馈瓶颈**: LLM 只看经过信息瓶颈的语义摘要,不看 raw latent。防止声学细节干扰全局语义规划。可迁移到其他多模态 AR 系统。
4. **Block-causal 并行训练**: 通过 position reset + block-causal mask 实现并行训练与逐步推理的数值一致性。适用于任何 AR-FM/AR-diffusion 模型。
5. **AudioVAE Stage 2 的多任务 learnability training**: 不只做重建,还要做下游任务对齐,让 latent space "prediction-friendly"。

## 审阅

> [!review] 审阅结论: pass (2026-06-10)
> - **conclusion**: pass
> - **issues**: 0 (0 high, 0 medium, 0 low)
> - 详见 `_review/dots.tts-review.yml`
