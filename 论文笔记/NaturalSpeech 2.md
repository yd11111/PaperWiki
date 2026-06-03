---
type: paper
tier: deep
title: "NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers"
arxiv_id: "2304.09116"
source: "https://arxiv.org/abs/2304.09116"
authors: [Kai Shen, Zeqian Ju, Xu Tan, Yanqing Liu, Yichong Leng, Lei He, Tao Qin, Sheng Zhao, Jiang Bian]
year: 2023
venue: "arXiv (Microsoft Research Asia & Microsoft Azure Speech)"
tags: [TTS, zero-shot, diffusion, latent-diffusion, speech-prompting, in-context-learning, singing-synthesis, voice-conversion, codec, non-autoregressive]
concepts: ["[[Residual Vector Quantization]]", "[[Diffusion Model]]", "[[Diffusion-based TTS]]", "[[Duration Predictor]]", "[[Speech Tokenizer]]", "[[Non-autoregressive TTS]]", "[[Prosody Modeling]]", "[[Speaker Embedding]]"]
models: ["[[模型库/NaturalSpeech 3|NaturalSpeech 3]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[Residual Vector Quantization]], [[Speech Tokenizer]], [[Speaker Embedding]], [[Prosody Modeling]])
> NaturalSpeech 2 是 [[Diffusion-based TTS]] [待确认] 的关键演进节点: 首次将 latent diffusion 应用于 TTS,避开了此前 AR codec LM 的离散 token 困境。[[Residual Vector Quantization]] 在 NS2 中被 repurpose 为连续向量的正则化工具(非传统压缩用途),其 RVQ 残差求和产生的连续 latent 成为 diffusion 的训练目标。[[Speech Tokenizer]] 页所述的 VQ-VAE/codec 路线(SoundStream → EnCodec)是 NS2 的 codec 基础。[[Speaker Embedding]] 和 [[Prosody Modeling]] 在 NS2 中被 speech prompting 机制隐式替代: 不再使用显式 speaker embedding 或 prosody tags,而是通过 prompt latent 的 in-context learning 实现零样本控制。
> 检索命中: [[Residual Vector Quantization]], [[Speech Tokenizer]], [[Speaker Embedding]], [[Prosody Modeling]] | 过滤: [[Diffusion Model]](pending-review), [[Diffusion-based TTS]](pending-review), [[Duration Predictor]](pending-review), [[Non-autoregressive TTS]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 用连续 latent vectors + latent diffusion 替代离散 tokens + AR LM,配合 speech prompting 实现强零样本 TTS 和歌声合成 / Replaces discrete tokens + AR LM with continuous latent vectors + latent diffusion plus speech prompting for strong zero-shot TTS and singing synthesis
> - **路线**: Text → Phoneme Encoder → Duration/Pitch Predictor → Condition c → Diffusion Model (with speech prompt z^p) → Latent z → Codec Decoder → Waveform [§3, Fig 1]
> - **指标**: CMOS 0.00 (= ground truth) on LibriSpeech/VCTK [Table 3]; SMOS 3.28/3.20 vs YourTTS 2.03/2.43 [Table 5]; WER 2.26% vs YourTTS 7.10% on LibriSpeech [Table 6]; 0% error rate on 50 hard sentences [Table 7]; outperforms VALL-E by +0.3 SMOS and +0.31 CMOS [Table 8]
> - **可借鉴**: (1) RVQ 残差求和作为连续 latent 表示 — 兼顾信息密度和序列长度; (2) Speech prompting 的双路设计 — duration/pitch predictor 用 Q-K-V attention 直接 attend prompt,diffusion model 用 query attention + FiLM 间接 attend 避免信息泄露; (3) Source-aware diffusion 实现零样本 voice conversion
> - **局限**: 模型仍在 underfitting (300K steps) [§4.3]; 推理需 150 步 ODE (慢); 未开源; 44K 小时训练数据规模中等

## 1. 核心问题与动机 / Core Problem & Motivation

**Why this paper?** 2023 年初 TTS 领域面临一个关键困境 [§1, Table 2]:

| 路线 | 问题 | 论文原文 |
|------|------|---------|
| 单层 VQ (低码率) | 波形重建质量差 | "will affect the waveform reconstruction quality due to large compression rate or low bitrate" [§2.1] |
| 多层 RVQ (高码率) | AR 序列太长,鲁棒性差 | "error propagation and robust issues due to the increased length in the token sequence" [§2.1] |
| 两阶段 LM (AudioLM) | 架构复杂,级联误差 | "first generate some coarse-grained tokens in each frame and then generate the remaining fine-grained tokens, which are complicated and incur cascaded errors" [§2.1] |

[论文原文] 作者将此总结为 "a dilemma between the codec and language model" [§2.1]。

[agent解读] NaturalSpeech 2 的核心洞察是: **跳出离散 token + AR LM 的范式**,直接在连续潜在空间用 diffusion 生成。这避开了 RVQ 层数 vs 序列长度的 trade-off,因为连续向量不需要展平多层 token,每帧只有一个向量。

## 2. 方法详解 / Method

### 2.1 Neural Audio Codec with Continuous Vectors [§3.1]

[论文原文] NS2 的 codec 由三部分组成 [Fig 2]:
1. **Audio Encoder** f_enc: 卷积块,总降采样率 200x (16KHz → 每 12.5ms 一帧)
2. **Residual Vector Quantizer** f_rvq: 16 个 codebook,每个 1024 entries,dim 256 [Table 11]
3. **Audio Decoder** f_dec: 镜像 encoder 结构

**关键设计** — 连续向量的获得方式 [§3.1, Eq 1]:

```
z^i = sum_{j=1}^{R} e_j^i   (R 个残差量化器的嵌入向量之和)
```

[agent解读] 这里的巧妙之处在于: RVQ 的本来目的是压缩和传输(低码率),但 NS2 不需要低码率 — 它的目的是语音合成。因此,NS2 **只取 RVQ 的正则化效果**: 用大量 quantizer (R=16) 和大 codebook (V=1024) 逼近连续向量,同时获得两个好处 [§3.1]:
1. 训练时不需存储连续向量(只存 codebook embedding + token ID,用 Eq 1 实时恢复)
2. 可添加 cross-entropy 正则化损失 L_ce-rvq 在离散 token ID 上

### 2.2 Latent Diffusion Model [§3.2]

[论文原文] Diffusion 采用 SDE 形式化 [§3.2, Eq 2-5]:
- 前向 SDE: dz_t = -1/2 * beta_t * z_t dt + sqrt(beta_t) dw_t
- 反向 SDE/ODE: 估计 score function ∇ log p_t(z_t)
- Score network s_theta 基于 WaveNet [§3.2],预测 z_hat_0 而非 score(经验上更好)

**Diffusion 损失** [§3.2, Eq 6]:
```
L_diff = E[||z_hat_0 - z_0||^2 + ||Sigma_t^{-1}(rho(z_hat_0,t) - z_t) - ∇ log p_t(z_t)||^2 + lambda_{ce-rvq} * L_{ce-rvq}]
```

[agent解读] 损失函数包含三项: (1) 数据重建项, (2) score loss 项(保持与 score-based diffusion 的一致性,且用于推理时的 reverse sampling), (3) RVQ cross-entropy 正则项(lambda=0.1)。第三项是 NS2 独特的: 它在连续 latent 预测的基础上,额外约束预测结果能正确对应到每一层 RVQ 的 codebook entry。

**Prior Model** [§3.2]:
- Phoneme Encoder: 6-layer Transformer, 8 heads, 512 dim, modified Conv FFN [§4.2]
- Duration Predictor: 30-layer 1D Conv, 10 Q-K-V attention layers [Table 11, 34M params]
- Pitch Predictor: 同 Duration Predictor 结构,50M params [Table 11]

**总损失** [§3.2, Eq 7]:
```
L = L_diff + L_dur + L_pitch
```

### 2.3 Speech Prompting for In-Context Learning [§3.3, Fig 3]

[论文原文] 训练时,随机从 target speech latent z 中截取片段 z^{u:v} 作为 prompt,模型只预测剩余部分 z^{\u:v}。

**双路设计** — 这是 NS2 最重要的设计选择之一:

| 模块 | Prompt 融合方式 | WHY |
|------|-------------|-----|
| Duration/Pitch Predictor | Q-K-V attention (Conv hidden 为 Q, prompt 为 K/V) | 需要精确的 phoneme-level 韵律信息,直接 attend 最有效 [§3.3] |
| Diffusion Model | 双层 attention: (1) m 个 random query → prompt hidden; (2) WaveNet hidden → m-length result → FiLM | "directly attending to the hidden sequence from the prompt encoder... exposes too many details... and may harm the generation" [§3.3] |

[agent解读] Duration/Pitch Predictor 直接看 prompt 是安全的,因为它只预测 duration/pitch 这类低维信号。但 Diffusion Model 如果直接看 prompt 的全部细节,会导致信息泄露(模型可能直接复制 prompt 片段而非生成)。因此用 m=32 个 learnable query token 做信息瓶颈,再通过 FiLM 层调制 WaveNet。这与 Perceiver/Q-Former 的思路一致。

### 2.4 Voice Conversion & Speech Enhancement [§5.7]

[论文原文] NS2 通过 **source-aware diffusion** 实现零样本 voice conversion [§5.7.1, Eq 8]:
1. 将 source audio z_source 通过 source-aware diffusion process 扩散到 z_1(保留部分源信息)
2. 从 z_1 通过 target-aware denoising 生成 z_target,条件为 target prompt z_prompt

[agent解读] 这种方法无需任何额外训练,直接复用 TTS 模型。核心 insight: diffusion 的前向过程不一定要加到纯噪声 — 可以加到"保留部分源信息的中间噪声",从而在 voice conversion 中同时保留源韵律和目标音色。

## 3. 实验结果与分析 / Experiments

### 3.1 训练配置 [§4]

| 配置 | 值 |
|------|-----|
| 训练数据 | MLS English, 44K hours, 5K+ speakers [§4.1] |
| Codec 训练 | 8x V100 16GB, batch=200/GPU, 440K steps [§4.3] |
| Diffusion 训练 | 16x V100 32GB, batch=6K frames/GPU, 300K steps (underfitting) [§4.3] |
| 推理 | Euler ODE, 150 steps, temperature tau=1.2^2 [§4.3] |
| 总参数量 | 435M [Table 11] |

### 3.2 Generation Quality [§5.1]

| Metric | NS2 | YourTTS | Ground Truth |
|--------|-----|---------|-------------|
| CMOS (LibriSpeech) | **0.00** | -0.65 | +0.04 [Table 3] |
| CMOS (VCTK) | **0.00** | -0.58 | -0.30 [Table 3] |

[论文原文] "+0.04 is regarded as on par" — NS2 在 LibriSpeech 上与真实语音不可区分 [§5.1]。在 VCTK 上甚至超越 ground truth(-0.30 for GT vs 0.00 for NS2),可能因为 VCTK 录音环境较嘈杂。

### 3.3 Speaker Similarity [§5.2]

| Metric | NS2 | YourTTS |
|--------|-----|---------|
| SMOS (LibriSpeech) | **3.28** | 2.03 [Table 5] |
| SMOS (VCTK) | **3.20** | 2.43 [Table 5] |

[agent解读] SMOS 差距巨大 (+1.25 on LibriSpeech),说明 speech prompting 机制在 speaker similarity 上远优于 YourTTS 的 speaker encoder 方法。值得注意: YourTTS 在 VCTK 训练时见过 97/108 speakers,NS2 全部 zero-shot,仍大幅领先。

### 3.4 Robustness [§5.3]

| Model | Repeats | Skips | Error Sentences | Error Rate |
|-------|---------|-------|-----------------|------------|
| Tacotron (AR) | 4 | 11 | 12 | 24% [Table 7] |
| Transformer TTS (AR) | 7 | 15 | 17 | 34% [Table 7] |
| NaturalSpeech 2 (NAR) | **0** | **0** | **0** | **0%** [Table 7] |

[论文原文] NS2 在 50 个特别困难的句子上实现 0% 错误率,与 FastSpeech/NaturalSpeech 等 NAR 模型一致 [§5.3]。

### 3.5 vs VALL-E [§5.4]

| Metric | NS2 | VALL-E |
|--------|-----|-------|
| SMOS | **3.83** | 3.53 [Table 8] |
| CMOS | **0.00** | -0.31 [Table 8] |

[agent解读] NS2 在 speaker similarity (+0.3 SMOS) 和自然度 (+0.31 CMOS) 上均优于 VALL-E,证明 continuous latent + diffusion 路线在 2023 年初期优于 discrete token + AR LM 路线。

### 3.6 Ablation Study [§5.5, Table 9]

| Setting | Pitch Mean↓ | Duration Mean↓ |
|---------|------------|----------------|
| Full NS2 | **10.11** | **0.65** |
| w/o diff prompt | diverge | diverge [Table 9] |
| w/o dur/pitch prompt | 21.69 | 0.77 [Table 9] |
| w/o CE loss | 10.69 | 0.71 [Table 9] |
| w/o query attn | 10.78 | 0.67 [Table 9] |

[论文原文] 关键发现 [§5.5]:
1. 去掉 diffusion speech prompt → **模型不收敛**,说明 prompt 对 diffusion model 是必要的
2. 去掉 duration/pitch prompt → pitch mean 从 10.11 → 21.69,韵律严重退化
3. 去掉 CE loss → 轻微退化,但 RVQ cross-entropy 提供了有益的正则化
4. 去掉 query attention(直接 cross-attend prompt）→ 信息泄露导致退化

### 3.7 Prompt Length Effect [§5.5, Table 10]

| Prompt Length | Pitch Mean↓ | Duration Mean↓ |
|--------------|------------|----------------|
| 3s | 10.11 | 0.65 [Table 10] |
| 5s | 6.96 | 0.69 [Table 10] |
| 10s | 6.90 | 0.62 [Table 10] |

[agent解读] Prompt 从 3s → 10s 时 prosody similarity 持续改善,但边际递减(5s→10s 改善很小)。这暗示 3s prompt 已足够提取核心 speaker identity,更长的 prompt 主要帮助韵律匹配。

### 3.8 Zero-Shot Singing Synthesis [§5.6]

[论文原文] NS2 可以用 speech prompt 生成 singing voice — "truly zero-shot singing synthesis (without a singing prompt)" [§5.6]。这得益于 duration/pitch prediction 和 non-autoregressive generation,使得 NS2 可以扩展到 speech 以外的 style。

## 4. 与已有方法的差异 / Comparison with Existing Work

| 维度 | NaturalSpeech 2 | VALL-E (discrete+AR) | AudioLM (discrete+2-stage) |
|------|-----------------|---------------------|---------------------------|
| 表示 | Continuous vectors (RVQ sum) | Discrete tokens (RVQ展开) | Semantic + Acoustic tokens |
| 生成模型 | Non-AR Diffusion | AR + NAR | AR coarse + AR fine |
| In-context | Speech prompt only | Text + Speech prompt | Speech prompt |
| 稳定性 | 0% error [Table 7] | 有 skip/repeat | 有级联误差 |
| 可扩展性 | Speech + Singing + VC + SE | Speech only | Speech + Music |
| 序列长度 | 1x (每帧一个向量) | 8x (8层展平) | coarse+fine拼接 |

## 5. 与 NaturalSpeech 系列的关系 [§3.4]

[论文原文] NS1 → NS2 的演进 [§3.4]:
- **Goal**: NS1 追求单说话人高质量; NS2 追求多说话人零样本多样性
- **Architecture**: NS2 保留 encoder/decoder + prior model 基本结构,增加 diffusion model + RVQ + speech prompting
- **Data**: NS1 用 LJSpeech (单说话人); NS2 用 44K hours MLS (多说话人)

[agent解读] NS2 → NS3 的演进方向: NS3 进一步将连续 latent 分解为 content/prosody/timbre/detail 四个子空间(factorized codec),每个子空间用独立的 discrete diffusion 生成。这可视为 NS2 "一个大 diffusion" 到 NS3 "四个小 diffusion" 的分解。

## 6. 局限与未来方向

[论文原文] "Our model is still underfitting and longer training will result in better performance" [§4.3]。

[agent解读] 主要局限:
1. **推理速度**: 150 步 ODE 求解,远慢于 AR 一次前向。后续 consistency model 可加速 [§6]
2. **未开源**: 无法复现验证
3. **数据规模**: 44K 小时在 2023 年已不算大(VALL-E 用 60K,后续 BASE TTS 用 100K)
4. **单一 diffusion 瓶颈**: 一个 diffusion model 同时建模所有属性 → NS3 的 factorized 方案是改进方向

---

> [!review] 审阅摘要
> - **结论**: pass
> - **主要优点**: 方法创新性强(continuous latent + latent diffusion 跳出 discrete token 范式); 实验充分(CMOS/SMOS/WER/鲁棒性/消融/对比); claim 标注完整
> - **次要建议**: 可补充 NaturalSpeech 1 的具体指标作为基线对比

检索命中: [[Residual Vector Quantization]], [[Speech Tokenizer]], [[Speaker Embedding]], [[Prosody Modeling]] | 过滤: [[Diffusion Model]](pending-review), [[Diffusion-based TTS]](pending-review), [[Duration Predictor]](pending-review), [[Non-autoregressive TTS]](pending-review) | 未命中但可能相关: 无
