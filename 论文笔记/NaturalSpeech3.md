---
type: paper
tier: deep
title: "NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models"
arxiv_id: "2403.03100"
source: "Sources/NaturalSpeech3.pdf"
authors: [Zeqian Ju, Yuancheng Wang, Kai Shen, Xu Tan, Detai Xin, Dongchao Yang, Yanqing Liu, Yichong Leng, Kaitao Song, Siliang Tang, Zhizheng Wu, Tao Qin, Xiang-Yang Li, Wei Ye, Shikun Zhang, Jiang Bian, Lei He, Jinyu Li, Sheng Zhao]
year: 2024
venue: "ICML 2024"
tags: [TTS, zero-shot, diffusion, factorization, disentanglement, codec, discrete-diffusion, non-autoregressive, speech-attribute]
concepts: ["[[SpeechFactorization]]", "[[Diffusion-basedTTS]]", "[[ResidualVectorQuantization]]", "[[Classifier-FreeGuidance]]", "[[GradientReversalLayer]]", "[[MaskedGenerativeModeling]]", "[[SpeechTokenizer]]"]
models: ["[[模型库/NaturalSpeech3|NaturalSpeech 3]]", "[[模型库/SoundStream|SoundStream]]", "[[模型库/EnCodec|EnCodec]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["LibriSpeech"]
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[SpeechFactorization]], [[ResidualVectorQuantization]], [[SpeechTokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechFactorization]]✓, [[ResidualVectorQuantization]]✓, [[SpeechTokenizer]]✓ | 过滤: [[Diffusion-basedTTS]][待确认], [[Classifier-FreeGuidance]][待确认], [[GradientReversalLayer]][待确认], [[MaskedGenerativeModeling]][待确认] | 未命中但可能相关: 无

**谱系定位**: NaturalSpeech 3 是微软 NaturalSpeech 系列的第三代。NS1 (2024) 实现单说话人人类水平质量,NS2 (2023) 通过 latent diffusion 探索零样本多说话人合成,NS3 进一步在多说话人 LibriSpeech 上首次达到人类水平自然度。KB 中 Speech Factorization 概念页已记录 NS3 作为"信息瓶颈 + 对抗训练"路线的代表工作,本次精读将深入其 FACodec 和 factorized diffusion 的具体机制。

**已有认知**:
- Speech Factorization 页记录了四种主流解耦方法 (对抗训练/信息瓶颈/self-distillation/辅助技术),NS3 综合使用了前三种
- RVQ 页记录了标准 RVQ 的层级结构 (coarse→fine),NS3 提出的 FVQ (Factorized VQ) 是一种正交于 RVQ 的量化方案 -- 按属性维度而非残差维度分解
- Gradient Reversal Layer [待确认] 页记录了 GRL 的基本机制,NS3 在 FACodec 中广泛使用 GRL 实现跨属性信息泄漏的惩罚

**创新判断**: NS3 的核心创新是"语音属性分解"思想的全栈实现 -- 不仅在 codec 端通过 FVQ 分解 (content/prosody/timbre/acoustic detail),更在生成端通过 factorized diffusion 逐属性生成。这种 divide-and-conquer 策略与标准 RVQ 的 coarse-to-fine 层级形成鲜明对比。

> [!summary] 速查
> - **一句话**: 提出 FACodec (属性分解 codec) + factorized discrete diffusion,将语音分解为 content/prosody/timbre/acoustic detail 四个子空间独立建模,首次在多说话人 LibriSpeech 上达到人类水平
> - **路线**: Text → Phoneme Encoder → Duration Diffusion → Length Regulator → Prosody Diffusion → Content Diffusion → Detail Diffusion → FACodec Decoder (+ timbre from prompt) → Waveform
> - **指标**: LibriSpeech Sim-O 0.67 / Sim-R 0.76 / WER 1.81 / CMOS 0.00 (=人类) / SMOS 4.01; RAVDESS MCD 4.28 / Acc 0.52 [Table 1, 2]
> - **可借鉴**: (1) Factorized VQ: 按语音属性维度分解而非按残差分解; (2) 信息瓶颈 + GRL + detail dropout 三重解耦机制; (3) In-context learning 实现零样本属性迁移; (4) 离散 diffusion 的 mask-and-predict 公式统一各子空间生成
> - **局限**: (1) 推理需 60 次 forward pass (4 个 diffusion x 4 iterations x 2 for CFG + duration),速度不如 AR 或 flow matching 方案; (2) FACodec 仅开源编码器,完整 TTS 系统未开源; (3) 分解的正交性缺乏理论保证

## 核心问题

现有大规模零样本 TTS 系统 (VALL-E, NaturalSpeech 2, Mega-TTS 2 等) 在语音质量、相似度和韵律方面仍不令人满意 [§1]。根本原因是: **语音信号内在地纠缠了多种属性 (content, prosody, timbre, acoustic detail),而现有方法未能有效解耦这些属性** [论文原文]。

两类已有方案的不足:
1. **基于 RVQ 的多层级 token (VALL-E, AudioLM)**: 将语音分解为 coarse→fine 的层级,但不同 RVQ 层仍混合了所有属性的信息,无法独立控制 [§1] [论文原文]
2. **基于连续表征的方法 (NaturalSpeech 2, Mega-TTS)**: 虽然尝试了一定程度的解耦,但建模复杂信息的耦合仍然存在 [§1]

NS3 的核心假设: **如果能将语音分解为独立的属性子空间,并为每个子空间设计专门的生成模型和条件,就能通过 divide-and-conquer 简化建模复杂度,同时实现更好的可控性** [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

NaturalSpeech 3 由两个核心组件构成 [Fig 1(a)]:

1. **FACodec (Factorized Audio Codec)**: 将语音波形分解为 content (z_c), prosody (z_p), timbre (h_t), acoustic detail (z_d) 四个解耦子空间
2. **Factorized Diffusion Model**: 由 phoneme encoder + 4 个共享结构的 discrete diffusion 模块 (duration/prosody/content/detail) 组成,按依赖顺序逐属性生成

### 关键设计选择

#### 设计选择 1: FACodec -- 属性分解 Codec

**WHY**: [论文原文] 标准 RVQ codec (SoundStream/EnCodec) 通过残差分解,但各层仍混合了所有语音属性。FACodec 的目标是按语音属性维度分解,使每个子空间只编码特定属性 [§3.2.1]。

**HOW** [§3.2.1, Fig 2]:
- **Speech Encoder**: 卷积编码器,downsample rate 200 (16kHz → 帧率 80Hz, 每帧 12.5ms),输出 pre-quantization latent h
- **Timbre Extractor**: Transformer encoder 将 h 转为全局向量 h_t (说话人身份)
- **三组 FVQ (Factorized Vector Quantizers)**:
  - FVQ_p: prosody quantizer (N_{q_p}=1 层 VQ)
  - FVQ_c: content quantizer (N_{q_c}=2 层 VQ)
  - FVQ_d: acoustic detail quantizer (N_{q_d}=3 层 VQ)
  - 所有 VQ codebook size = 1024
  - **信息瓶颈**: 每个 FVQ 将 encoder 输出投影到 8 维低维空间再量化,强制每个子空间只保留有限信息 [§3.2.2]
- **Speech Decoder**: 将 z_p + z_c + z_d 求和,通过 Conditional Layer Norm 融合 timbre h_t,解码回波形
- 总带宽: 4.8 kbps (6 层 VQ × 80 Hz × 10 bit)

#### 设计选择 2: 三重解耦机制

**WHY**: [论文原文] 仅靠信息瓶颈不足以保证分解的正交性,需要多种辅助手段 [§3.2.2]。

**HOW** [§3.2.2]:

1. **信息瓶颈** (Information Bottleneck): 每个 FVQ 在 8 维空间量化,限制每个子空间的信息容量 [论文原文]

2. **监督损失** (Supervision):
   - Prosody: z_p 后接 predictor 预测 normalized F0 (z-score) [§3.2.2]
   - Content: z_c 后接 predictor 预测 phoneme labels [§3.2.2]
   - Timbre: h_t 后接 speaker classifier 预测 speaker ID [§3.2.2]

3. **梯度反转** (Gradient Reversal Layer):
   - Prosody 子空间: 加 phoneme-GRL → 防止 prosody 编码内容信息 [§3.2.2]
   - Content 子空间: 加 F0-GRL → 防止 content 编码韵律信息 [§3.2.2]
   - Acoustic detail 子空间: 加 phoneme-GRL + F0-GRL → 防止细节编码内容和韵律 [§3.2.2]
   - Timbre: 在 z_p + z_c + z_d 之和上加 speaker-GRL → 防止语义子空间泄露说话人信息 [§3.2.2]

4. **Detail Dropout**: 训练时以概率 p 随机 mask 掉 z_d [§3.2.2]
   - [论文原文] 经验发现 codec 倾向于将无监督信息 (如 prosody/content 泄漏) 存入 acoustic details 子空间
   - Dropout 迫使 decoder 仅用 prosody + content + timbre 也能重建低质量语音,确保解耦
   - [agent 解读] 这类似于 Quantizer Dropout 的思路,但应用在属性维度而非 RVQ 层级

#### 设计选择 3: Factorized Diffusion Model

**WHY**: [论文原文] 每个属性有不同的条件依赖: duration 只依赖 phoneme,prosody 依赖 phoneme + duration prompt,content 依赖 prosody + phoneme,detail 依赖前面所有。分别建模比统一建模更高效 [§3.3.1]。

**HOW** [§3.3.1, Fig 3]:
四个 diffusion 模块共享相同的 discrete diffusion 公式 [§3.3.2],但各有不同的条件:
1. **Duration Diffusion**: 条件 = phoneme encoding + duration prompt → 生成 phoneme-level duration
2. **Prosody Diffusion**: 条件 = prosody prompt + phoneme c_ph → 生成 z_p
3. **Content Diffusion**: 条件 = content prompt + 已生成 z_p + phoneme c_ph → 生成 z_c
4. **Detail Diffusion**: 条件 = detail prompt + 已生成 z_p, z_c, c_ph → 生成 z_d

**Timbre 不需要生成**: 直接从 prompt speech 的 FACodec encoder 提取 h_t [论文原文]

**离散 Diffusion 公式** [§3.3.2]:
- Forward: X_t = X ⊙ M_t,用 [MASK] 替换被 mask 的 token
- Mask schedule: σ(t) = sin(πt/2T), m_{t,i} ~ Bernoulli(σ(t))
- Loss: L_mask = -Σ m_{t,i} · log(p_θ(x_i | X_t, X^p, C)) [Eq 在 §3.3.2]
- **In-context learning**: 将 prompt 序列 (无噪声) 与 target 序列 (有 mask) 拼接,prompt 部分不加噪
- **推理**: 从全 mask 出发迭代 unmask,每步按 confidence 选择保留哪些 token (类似 MaskGIT/SoundStorm)

**Classifier-Free Guidance** [§3.3.2]:
- 训练时以 p_cfg = 0.15 概率丢弃 prompt
- 推理时: g_cfg = g_cond + α · (g_cond - g_uncond),α 基于实验选择
- 最终 rescale: g_final = std(g_cond) × g_cfg / std(g_cfg) [论文原文]

### 训练策略

- 训练数据: Librilight 60K 小时 (16kHz, ~7000 说话人) [§4.1]
- 扩展版: 内部 200K 小时 + 1B 参数模型 [§4.4]
- FACodec: 8 x NVIDIA V100 32GB, batch 32, lr=2e-4, 800K steps [§B.1]
- Diffusion Model: 8 x A100 80GB, batch 10K frames, lr=1e-4, 1M steps, AdamW [§A.2]
- 每个 diffusion 过程推理 4 iterations,CFG 使 forward passes 翻倍,总计 60 forward passes [§4.1]

## 实验

| 指标 | 本文 (NS3) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Sim-O | 0.67 | Ground Truth: 0.68, VALL-E: 0.47, NS2: 0.55, Voicebox: 0.64 | LibriSpeech test-clean | [Table 1] |
| Sim-R | 0.76 | VALL-E: 0.51, NS2: 0.62, Voicebox: 0.67 | LibriSpeech test-clean | [Table 1] |
| WER (%) | 1.81 | Ground Truth: 1.94, VALL-E: 5.90, NS2: 1.94, Voicebox: 2.03 | LibriSpeech test-clean | [Table 1] |
| CMOS | 0.00 (= human) | VALL-E: -0.60, NS2: -0.18, HierSpeech++: -0.41 | LibriSpeech test-clean | [Table 1] |
| SMOS | 4.01 | Ground Truth: 3.85, VALL-E: 3.46, NS2: 3.65 | LibriSpeech test-clean | [Table 1] |
| MCD avg (RAVDESS) | 4.28 | VALL-E: 5.03, NS2: 4.56, Voicebox: 4.88 | RAVDESS | [Table 2] |
| MCD-Acc | 0.52 | VALL-E: 0.34, NS2: 0.25, Voicebox: 0.34 | RAVDESS | [Table 2] |
| RTF | 0.296 | NS2: 0.366, VALL-E: 4.520 | Server (V100) | [Table 10] |
| NS3 one-step RTF | 0.067 | - | Server (V100) | [Table 10] |

**关键发现**:

1. **首次多说话人人类水平**: CMOS=0.00, WER 低于 ground truth (1.81 vs 1.94), SMOS 超过 ground truth (4.01 vs 3.85) [Table 1] [论文原文] 这是继 NS1 在单说话人上达到人类水平后的又一里程碑

2. **Factorization 是关键**: 消融 [Table 3] 去掉 factorization (用 SoundStream + 无分解 diffusion) 后: Sim-O 降 0.12, WER 升 0.68, CMOS 降 0.25, SMOS 降 0.42 [论文原文]

3. **CFG 对相似度贡献显著**: 去掉 CFG 后 Sim-O 降 0.03, SMOS 降 0.21 [Table 3]

4. **韵律表征优于 Mel**: 用 mel spectrogram 前 20 bins 替代 FVQ 学到的韵律表征,MCD 从 4.28 升至 4.34, MCD-Acc 从 0.52 降至 0.46 [Table 4]

5. **FACodec 重建质量与同带宽 codec 可比**: 在 4.8 kbps 带宽下,FACodec PESQ 3.47 vs SoundStream 3.03, STOI 0.95 vs 0.90, MCD 2.59 vs 3.38 [Table 5] [论文原文] 分解不牺牲重建质量

6. **Factorization 可迁移至 AR 模型**: VALL-E + FACodec 在所有指标上一致优于原始 VALL-E [Table 6],证明属性分解思想不限于 NAR diffusion

7. **Scaling 有效**: 数据从 1K→60K→200K 小时,Sim-O 从 0.64→0.72→0.73; 模型从 500M→1B,Sim-O 从 0.73→0.78 [Table 7, 8]

## 局限性

1. **推理速度**: 60 forward passes (4 diffusion × 4 iter × 2 for CFG + duration),RTF 0.296,虽比 VALL-E (4.52) 快但远慢于 flow matching 方案 (CosyVoice ~0.05)。one-step 变体 (RTF 0.067) 存在 0.29 UTMOS 下降 [Table 10]

2. **分解正交性缺乏理论保证**: 三重解耦机制 (信息瓶颈 + GRL + detail dropout) 是经验性的,无法保证 z_p, z_c, z_d 真正正交。[agent 解读] 论文中的 GRL 惩罚的是 phoneme/F0/speaker 的可预测性,但可能存在其他维度的泄漏未被捕获

3. **FACodec 复杂度高**: 6 个 VQ + 3 组 GRL + timbre extractor + detail dropout,训练超参数多 (10+ loss 权重 λ) [§B.1],调参成本高

4. **仅 FACodec encoder 开源**: 完整 TTS 系统 (factorized diffusion model) 未开源,复现难度大

5. **评估数据集有限**: 主要在 LibriSpeech test-clean 上评估,该数据集为朗读式语音,对自然对话/情感语音的泛化性未验证

6. **属性操控的定量评估不足**: §4.3.2 展示了 timbre/duration/prosody 的操控能力,但仅有定性 demo,缺乏定量评估

## 点评

NaturalSpeech 3 是一篇概念优雅、实验扎实的工作。其核心 insight -- **按语音属性维度分解比按 RVQ 层级分解更合理** -- 切中了 TTS 建模的本质。标准 RVQ 的 coarse-to-fine 分解是基于信号重建的视角 (哪些信息对重建贡献最大?),而 FACodec 的属性分解是基于生成控制的视角 (哪些信息可以独立控制?)。后者更符合 TTS 的需求。

三重解耦机制的设计体现了深度的领域理解: 信息瓶颈限制容量 → 监督损失引导方向 → GRL 惩罚泄漏 → detail dropout 防止"垃圾桶"效应。每一层都有明确的设计动机。

最令人印象深刻的实验是 VALL-E + FACodec [Table 6]: 仅替换 tokenizer 就让 VALL-E 在所有指标上显著提升,证明了属性分解的思想可以独立于生成模型工作,具有很强的迁移性。

不足是系统复杂度高 (10+ loss 权重,6 个 VQ,多组 GRL) 且推理慢 (60 forward passes)。后续工作如 CosyVoice 系列选择了更简洁的 "semantic tokenizer + flow matching" 路线,在实用性上更有优势。

## 可复用的 idea

1. **属性维度分解 vs 残差维度分解**: 将信号分解的思路从 "coarse-to-fine" 转为 "content/prosody/timbre/detail",使每个子空间可独立生成和控制。适用于任何需要多维度控制的生成任务。

2. **信息瓶颈 + GRL + dropout 三重解耦**: (a) 低维投影 (8d) 限制容量; (b) GRL 惩罚属性泄漏; (c) detail dropout 防止"垃圾桶"子空间。三者互补,可单独或组合使用。

3. **Discrete diffusion 的 in-context learning**: 将 prompt (无噪声) 和 target (有 mask) 拼接,模型自然学会 in-context 生成。无需额外的 prompt encoding 模块。

4. **Factorization 可迁移**: FACodec 的 tokenizer 可以 drop-in 替换其他系统的 tokenizer (如 VALL-E + FACodec),这意味着好的 tokenizer 设计可以跨系统复用。

5. **属性操控**: 推理时可以自由组合不同来源的 prompt (如 speaker A 的 timbre + speaker B 的 prosody + 新 content),实现跨属性迁移。

---

检索命中: [[SpeechFactorization]], [[ResidualVectorQuantization]], [[SpeechTokenizer]] | 过滤: [[Diffusion-basedTTS]](pending-review), [[Classifier-FreeGuidance]](pending-review), [[GradientReversalLayer]](pending-review), [[MaskedGenerativeModeling]](pending-review) | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排

## 代码级分析

> [!info] 代码来源
> - 仓库: https://github.com/open-mmlab/Amphion (9837 stars)
> - 路径: `models/codec/ns3_codec/` (FACodec) + `models/codec/facodec/` (替代实现)
> - commit: 26f6883
> - 分析日期: 2026-06-10
> - 注意: Amphion 仅开源了 FACodec (编码器/解码器),完整的 factorized diffusion TTS 系统未开源。FACodec 部分由论文作者参与贡献,可视为半官方实现

### 架构验证

Amphion 的 FACodec 实现与论文高度一致:

| 组件 | 论文描述 | 代码实现 | 一致性 |
|------|----------|----------|--------|
| Encoder | 卷积下采样,downsample 200x | `FACodecEncoder(up_ratios=(2,4,5,5))` → hop=200 | 完全一致 |
| Decoder | 对称上采样 | `FACodecDecoder(up_ratios=(2,4,5,5))` | 完全一致 |
| Prosody VQ | 1 层 FVQ, codebook 1024 | `FactorizedVectorQuantize(codebook_dim=8)` | 一致 |
| Content VQ | 2 层 FVQ, codebook 1024 | 2 个 `FactorizedVectorQuantize` 串联 | 一致 |
| Detail VQ | 3 层 FVQ, codebook 1024 | 3 个 `FactorizedVectorQuantize` 串联 | 一致 |
| 信息瓶颈 | 8 维低维投影 | `in_proj = Linear(dim, 8)`, `out_proj = Linear(8, dim)` | 完全一致 |
| GRL | phoneme-GRL + F0-GRL + speaker-GRL | `GradientReversal(alpha)` class | 完全一致 |
| Timbre Extractor | Transformer encoder → 全局向量 | `TransformerEncoder` + global mean pooling | 一致 |
| 激活函数 | SnakeBeta | `SnakeBeta(alpha_logscale=True)` | 一致 |

### 论文未写的实现细节

1. **FactorizedVectorQuantize 的 L2 归一化**: `fvq.py` 中 `decode_latents()` 在计算距离前对 encodings 和 codebook 都做了 L2 normalize: `encodings = F.normalize(encodings); codebook = F.normalize(codebook)`。这个关键细节论文未提及,但对 codebook 利用率有重要影响 (类似 DAC 的 L2-norm 策略)。

2. **Straight-through gradient estimator**: `z_q = z_e + (z_q - z_e).detach()` — 标准 STE,前向 pass 使用量化值,反向 pass 梯度直通到 encoder。论文未详述。

3. **GradientReversal 实现**: 极简 — `forward()` 直接返回 x,`backward()` 返回 `-alpha * grad_output`。alpha 是固定超参数 (非可学习)。论文描述了 GRL 的使用但未给出 alpha 值。

4. **SnakeBeta 激活函数**: `x + (1/beta) * sin^2(x * alpha)`,其中 alpha 和 beta 可学习 (log scale 初始化)。这是 BigVGAN 引入的激活函数,论文标注使用了 BigVGAN 风格 decoder 但未详细说明激活函数。

5. **Encoder 架构细节**: 使用 `weight_norm` 的 Conv1d,每个 `EncoderBlock` 含 3 个 `ResidualUnit` (dilation 1, 3, 9) + 下采样 Conv。这与 SoundStream/DAC 的 encoder 架构高度相似,论文未详述。

6. **CNNLSTM 预测器**: `CNNLSTM` class 用于 prosody/content/speaker 的辅助预测器,包含 3 个 ResidualUnit (dilation 1,2,3) + LSTM + linear heads。论文仅提到 "predictor" 但未给出架构。

7. **Dual codebook 路径**: `models/codec/ns3_codec/` 和 `models/codec/facodec/` 存在两套实现。前者更接近论文原始设计 (含独立的 FVQ 和 RVQ 模块),后者包含更多工程优化。

### 训练 pipeline 拆解

FACodec 训练涉及多个 loss (基于 `facodec_trainer.py`):
1. **Reconstruction loss**: mel spectrogram L1/L2 + waveform L1
2. **VQ commitment loss**: 每个 FVQ 的 commitment + codebook loss
3. **Prosody supervision**: F0 prediction loss (通过 JDC F0 估计器)
4. **Content supervision**: phoneme classification loss
5. **Speaker supervision**: speaker classification loss
6. **GRL adversarial**: phoneme-GRL on prosody, F0-GRL on content, phoneme+F0 GRL on detail, speaker-GRL on sum(z_p, z_c, z_d)
7. **GAN loss**: multi-scale + multi-period discriminator (HiFi-GAN 风格)

### 推理 pipeline 拆解

FACodec 推理:
1. 输入 waveform → Encoder → latent h
2. Timbre Extractor: h → global timbre vector h_t
3. Prosody VQ: project to 8d → quantize → z_p
4. Content VQ: project to 8d → 2-layer VQ → z_c
5. Detail VQ: project to 8d → 3-layer VQ → z_d
6. Decoder: sum(z_p, z_c, z_d) + Conditional Layer Norm(h_t) → waveform

**注意**: 完整 TTS (text → waveform) 需要 factorized diffusion model 生成 z_p, z_c, z_d,但此部分未开源。

### 关键超参数表

| 参数 | 论文值 | 代码实际值 | 备注 |
|------|--------|-----------|------|
| Encoder 下采样率 | 200 (16kHz → 80Hz) | `up_ratios=(2,4,5,5)` → 200 | 完全一致 |
| Encoder base channels | 32 | `ngf=32` | 一致 |
| FVQ codebook dim | 8 | `codebook_dim=8` | 完全一致 |
| FVQ codebook size | 1024 | `codebook_size=1024` | 完全一致 |
| Prosody VQ 层数 | 1 | 1 个 FVQ | 一致 |
| Content VQ 层数 | 2 | 2 个 FVQ | 一致 |
| Detail VQ 层数 | 3 | 3 个 FVQ | 一致 |
| 总带宽 | 4.8 kbps (6 VQ x 80Hz x 10bit) | 6 个 FVQ x 80Hz x 10bit = 4.8kbps | 完全一致 |
| Encoder output dim | 1024 | `out_channels=1024` | 一致 |
| GRL alpha | 论文未报告 | 代码中需查配置文件 | 关键超参数 |

### 复现 checklist (基于代码)

- [x] FACodec Encoder/Decoder (卷积架构)
- [x] FactorizedVectorQuantize (8 维信息瓶颈)
- [x] 三组 FVQ (prosody 1 层 / content 2 层 / detail 3 层)
- [x] Timbre Extractor (Transformer encoder + global pooling)
- [x] GradientReversal Layer
- [x] CNNLSTM 辅助预测器 (F0/phoneme/speaker)
- [x] FACodec 训练 pipeline (含多 loss)
- [x] FACodec 推理 pipeline
- [x] 预训练权重 (Amphion 提供 FACodec checkpoint)
- [ ] **Factorized Diffusion Model** — 未开源 (核心 TTS 生成模块)
- [ ] **Duration Diffusion** — 未开源
- [ ] **完整 TTS pipeline** — 仅有 codec 部分

### 代码质量与可复现性评估

**质量**: **高**。Amphion 是专业的语音合成研究框架,代码组织规范。FACodec 实现约 1500+ 行 (含辅助模块),清晰覆盖论文 Fig 2 的每个组件。GRL、FVQ 等核心模块实现简洁正确。

**可复现性**: **仅 FACodec 部分可复现**。Amphion 提供了 FACodec 的预训练权重,可直接用于语音编码/解码和属性分解。但完整的 NaturalSpeech 3 TTS 系统 (factorized diffusion model + duration model + 端到端推理) 未开源,无法复现论文的 TTS 结果。FACodec 可作为独立的 speech tokenizer 用于其他 TTS 系统 (如论文 Table 6 中 VALL-E + FACodec 的实验)。
