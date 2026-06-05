---
type: paper
tier: deep
title: "ARCHI-TTS: A Flow-Matching-Based Text-to-Speech Model with Self-Supervised Semantic Aligner and Accelerated Inference"
arxiv_id: "2602.05207"
source: "Sources/ARCHI-TTS.pdf"
authors: [Chunyat Wu, Jiajun Deng, Zhengxi Liu, Zheqi Dai, Haolin He, Qiuqiang Kong]
year: 2026
venue: "arXiv"
tags: [TTS, flow-matching, non-autoregressive, text-speech-alignment, inference-acceleration, low-token-rate, DiT]
concepts: ["[[ConditionalFlowMatching]]", "[[Non-autoregressiveTTS]]", "[[Classifier-FreeGuidance]]", "[[SpeakerEmbedding]]", "[[VariationalAutoencoderforTTS]]"]
models: ["[[CosyVoice]]", "[[CosyVoice2]]", "[[NaturalSpeech3]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[ConditionalFlowMatching]], [[SpeakerEmbedding]]; 4 个待确认参考: [[Non-autoregressiveTTS]], [[Classifier-FreeGuidance]], [[VariationalAutoencoderforTTS]], [[Speech-TextAlignment]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]]✓, [[SpeakerEmbedding]]✓, [[Non-autoregressiveTTS]], [[Classifier-FreeGuidance]], [[VariationalAutoencoderforTTS]], [[Speech-TextAlignment]] | 过滤: Non-autoregressiveTTS, Classifier-FreeGuidance, VariationalAutoencoderforTTS, Speech-TextAlignment (pending-review) | 未命中但可能相关: [[TokenRateandBitrateTrade-offs]]

**谱系定位**: ARCHI-TTS 属于 NAR flow-matching TTS 谱系,直接对标 F5-TTS 和 E2 TTS 这类 "padding-based alignment + CFM" 的方案。与 F5-TTS 使用 padding character tokens 匹配语音长度不同,ARCHI-TTS 引入专门的 Semantic Aligner 模块显式学习文本-语音对齐,回归了 "有 aligner" 路线但避免了 NaturalSpeech3/Voicebox 的 phoneme-level hard duration。在推理加速方面,不同于 DMDSpeech/E1-TTS 需要蒸馏的路线,ARCHI-TTS 通过分离 encoder-decoder 架构实现 training-free 的 encoder 特征复用加速,这与 DDT (Decoupled Diffusion Transformer) 的思路一致。

**已有认知**:
- [[ConditionalFlowMatching]]: CFM 已是 NAR TTS 的标准训练范式 (F5-TTS, CosyVoice 系列, MaskGCT 等)。ARCHI-TTS 的 CFM 使用 OT 路径 + Euler solver,属于标准配置,但其 "condition encoder + velocity decoder" 的分离 DiT 架构在 TTS 领域尚属首次。
- [[SpeakerEmbedding]]: 使用 CAM++ (3D-Speaker) 提取 speaker embedding 作为全局条件。消融证实在低 token rate VAE latent 下 speaker embedding 对 SSIM 至关重要,暗示低帧率压缩损失了 mel spectrogram 中丰富的说话人身份信息。
- [[VariationalAutoencoderforTTS]]: 使用自训练 VAE 将 24kHz 语音压缩为 12.5Hz 连续 latent。这一帧率远低于 mel spectrogram 的 50-100Hz,与 Stable Audio 的设计一致,是降低序列长度、提升推理效率的关键。

**创新判断**: 核心创新在于 (1) Semantic Aligner 用 transformer 学习灵活长度的语义对齐表示,回避了 padding 方案的计算浪费和 hard duration 的韵律损失; (2) 分离 DiT 的 encoder 特征复用实现 training-free 加速,75% sharing ratio 下 RTF 从 0.21 降至 0.09。辅助 CTC loss 加速收敛是已有技术的合理应用。

## 速查

> [!summary] 速查
> - **一句话**: 提出 Semantic Aligner + 分离 DiT 架构的 NAR flow-matching TTS,通过显式语义对齐和 training-free 推理加速同时提升质量和效率
> - **路线**: Text→ConvNeXt V2→Semantic Aligner(+mask embeddings)→Condition Encoder(18 DiT)→Velocity Decoder(4 DiT)→VAE Decoder→Waveform
> - **指标**: LibriSpeech-PC WER 1.98% / SSIM 0.70 [Table 1]; SeedTTS-EN WER 1.47% / SSIM 0.68, SeedTTS-ZH WER 1.42% / SSIM 0.70 [Table 2]; RTF 0.21 (32 NFE) → 0.09 (75% sharing) [Fig 2]; 289M params, 100K hrs Emilia, 8×RTX5090 4天
> - **可借鉴**: (1) 分离 condition encoder 和 velocity decoder 使 encoder 输出可跨步复用,training-free 加速推理; (2) 用 learnable mask embedding 作为 temporal canvas 让 transformer 自动学习对齐,无需 hard duration 也无需 padding to speech length; (3) CTC loss 在 DiT 中间层做辅助语义监督加速收敛
> - **局限**: MOS 略低于 F5-TTS 和 CosyVoice2 [Table 3]; SSIM 在 SeedTTS-EN 上 0.68 低于 DiTAR 0.74 和 Seed-TTSDiT 0.79; 未报告长句/复杂文本的鲁棒性; 消融仅在小规模 LibriTTS/LibriHeavy 上进行

## 核心问题

1. **文本-语音对齐**: NAR TTS 中文本序列与语音帧长度不匹配。Padding 方案 (E2 TTS, F5-TTS) 浪费计算且不显式建模语义关系;hard duration (NaturalSpeech3, Voicebox) 可能损害自然度。能否设计一种"软对齐"模块,既灵活又有语义质量? [§1, §2.1]
2. **推理效率**: Flow-matching 模型需要多步迭代去噪,计算开销大。蒸馏方法 (DMDSpeech, E1-TTS) 需要额外的 teacher 模型和训练开销。能否在不增加训练成本的前提下加速推理? [§1, §2.6]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ARCHI-TTS 由三个主要模块组成 [Fig 1]:

1. **Semantic Aligner**: 接收文本 token 序列和语音长度的 mask embedding 序列,输出语义对齐表示 z
2. **Condition Encoder** (18 DiT blocks): 处理 z + speaker embedding s + audio prompt x_ref,输出 conditioned hidden states h
3. **Velocity Decoder** (4 DiT blocks): 从 noisy speech latent x_t 和 h 预测速度场 v_t

底层表示使用自训练 VAE 将 24kHz 语音压缩为 12.5Hz 连续 latent [§2.2]。

### 关键设计选择

**设计1: Semantic Aligner — 为什么用 transformer + mask embedding 而不是 padding 或 hard duration?**

[论文原文] 论文明确指出 phoneme-level duration (NaturalSpeech3, Voicebox) 的刚性对齐会损害自然度 [§1]。而 E2 TTS/F5-TTS 的 padding 方案虽然简单有效,但 [agent 解读] 实质上是让 DiT 自身通过 full attention 隐式学习对齐,没有专门的对齐模块。

ARCHI-TTS 的方案是在两者之间找到平衡 [§2.1]:
- 文本经 ConvNeXt V2 blocks 编码为语义特征 y
- 创建 N 个 learnable mask embedding m 的副本 (N = 目标语音 latent 长度),作为 "uniform temporal canvas"
- 两个序列各加一个 learnable start-of-sequence token 后送入 Transformer:
  ```
  z = Transformer(e_st, y, e_sm, m, ..., m)
  ```
- [论文原文] mask embeddings 初始只编码时间长度信息,transformer 通过自注意力聚合文本语义并与时间 canvas 对齐,将简单的 duration markers 转化为 "contextually-aware semantic features" [§2.1]
- [论文原文] 关键优势: 解除了语音长度对文本 token 长度的依赖,在低 token rate + character tokenization 场景下尤其重要(文本 token 数可能少于语音帧数) [§2.1]

**设计2: 分离 Condition Encoder / Velocity Decoder — 为什么不用统一的 DiT?**

[论文原文] 论文引用了 DDT (Decoupled Diffusion Transformer) [17] 的思路,将 DiT 分为 condition encoder 和 velocity decoder [§2.3-2.4]:
- Condition Encoder (18 layers): 处理所有条件信号 (z, s, x_ref),输出 conditioned hidden states h
- Velocity Decoder (4 layers): 从 noisy latent x_t + h 预测速度

[论文原文] 关键设计: h 不是与 x_t 拼接 (local feature),而是加到 sinusoidal timestep embedding 上作为 global condition 注入每个 DiT block [§2.4]。

[论文原文] 这种分离的直接好处: condition encoder 的输出 h 不依赖当前噪声级别 x_t 的具体值,因此可以在相邻去噪步之间共享 h,跳过 encoder 前向传播 (encoder 通常占计算主导) [§2.6]。

[agent 解读] 分配比例 18:4 暗示条件理解远比速度预测复杂,也意味着跳过 encoder 的加速收益极大——即使 75% 的步骤复用 h,decoder 仍在每步更新速度预测。

**设计3: 辅助 CTC loss — 为什么在 encoder 中间层加 CTC?**

[论文原文] 在 condition encoder 第 i 个 DiT block 的中间隐表示上应用 CTC loss [§2.3, Eq. 3]:
```
L_CTC = -log p_CTC(y | Φ_i(v_t(x_t, x_ref, z, s; θ)))
```
[论文原文] 目的是增强 hidden representation 与输入文本之间的语义对齐 [§2.3]。

[agent 解读] CTC loss 提供了一个显式的语义监督信号——要求 encoder 中间层的表示能被解码回文本,这迫使 encoder 维持语义结构。实验显示 CTC 对 WER 收敛速度有贡献: 100k updates 时 WER 已达 2.14% [§3.4]。

**设计4: 低 token rate VAE (12.5Hz)**

[论文原文] 采用 Stable Audio 架构原则训练的自定义 VAE,将 24kHz 语音编码为 12.5Hz 连续 latent [§2.2]。这比传统 mel spectrogram 的 50-100Hz 低得多。

[论文原文] 动机: 减少高时间冗余、消除独立 vocoder 需求 [§2.2]。

[agent 解读] 极低帧率意味着 10 秒语音仅 125 帧,极大降低了 DiT 的序列长度和注意力计算。但消融表明这也损失了说话人身份信息(mel 中丰富的 fine-grained 音色被压缩),因此 speaker embedding 变得必不可少 [Table 4 分析]。

### 训练策略

- **总损失**: L = L_CFM + L_DIR + η·L_CTC, η=0.1 [Eq. 4]
  - L_CFM: OT 路径上的 velocity MSE loss [Eq. 2]
  - L_DIR: velocity direction loss (cosine similarity 确保正确流方向) [§2.4]
  - L_CTC: 辅助文本对齐 CTC loss [Eq. 3]
- **Timestep sampling**: logit-normal 分布,聚焦生成轨迹的起止点 [§2.4]
- **CFG dropout**: audio prompt + speaker embedding 联合丢弃概率 0.3,全条件丢弃概率 0.2 [§3.2]
- **Audio masking**: 70-100% 的 audio latent 随机 mask 用于 infilling [§3.2]
- **训练规模**: 800k updates, batch 3750 latent frames (~0.67h audio), 8×RTX5090 32GB, 4 天 [§3.2]
- **数据**: Emilia 100K hours (多语言) [§3.1]
- **优化器**: AdamW, lr=1e-4, 1k warmup + linear decay, gradient clipping 1.0 [§3.2]
- **推理默认**: 32 NFE, CFG strength 4.0, timeshift 3.0 [§3.4]

## 实验

| 指标 | ARCHI-TTS | F5-TTS | DiTAR | CosyVoice | MaskGCT | E2 TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER(%)↓ | **1.98** | 2.42 | 2.39 | 3.59 | 2.72 | 2.95 | LibriSpeech-PC | [Table 1] |
| SSIM↑ | **0.70** | 0.66 | 0.67 | 0.66 | 0.69 | 0.69 | LibriSpeech-PC | [Table 1] |
| RTF↓ | 0.21 | - | - | 0.92 | 0.84 | 0.68 | LibriSpeech-PC | [Table 1] |
| WER(%)↓ | **1.47** | 1.83 | 1.69 | - | 2.62 | 2.19 | Seed-EN | [Table 2] |
| SSIM↑ | 0.68 | 0.67 | **0.74** | - | 0.72 | 0.71 | Seed-EN | [Table 2] |
| WER(%)↓ | **1.42** | 1.56 | 1.02 | - | 2.27 | 1.97 | Seed-ZH | [Table 2] |
| SSIM↑ | 0.70 | 0.76 | 0.75 | - | 0.77 | 0.73 | Seed-ZH | [Table 2] |
| NMOS | 3.53 | **3.62** | - | - | - | - | Seed subset | [Table 3] |
| SMOS | 3.48 | **3.54** | - | 3.32 | - | - | Seed subset | [Table 3] |
| CMOS | +0.09 | -0.03 | - | +0.10 | - | - | Seed subset | [Table 3] |

**参数与资源**: 289M params (最小), 100K hours, 8×RTX5090 4天 [§3.4]。对比: F5-TTS 336M, MaskGCT ~1.1B。

**推理加速** (encoder sharing) [Fig 2]:
- 0% sharing (baseline): RTF 0.21, WER 1.98%, SSIM 0.70 (32 NFE)
- 75% sharing: RTF **0.09**, WER 1.98%, SSIM 0.70 (32 NFE) — 2.3x 加速,无显著性能损失
- 更高 sharing ratio 有退化,但增加 NFE 可部分补偿

**消融** [Table 4]:
- 去掉 speaker embedding → SSIM 显著下降 (0.71→0.62, LibriHeavy scale),WER 略变 [Table 4]
- Semantic features + VQ: 标准 VQ WER 略升,但 codebook size×2 后 WER 反而低于无 VQ 版本 (2.15 vs 2.16) [Table 4]

## 局限性

1. **MOS 偏低**: NMOS 3.53 低于 F5-TTS 3.62 和 Ground Truth 3.72,主观质量仍有差距 [Table 3]
2. **SSIM 在 SeedTTS 上不突出**: EN 0.68 显著低于 DiTAR 0.74 和 Seed-TTSDiT 0.79;ZH 0.70 低于 F5-TTS 0.76 [Table 2]。[agent 解读] 低 token rate VAE 的信息压缩可能是 speaker similarity 不够强的原因。
3. **消融尺度有限**: 消融实验在 LibriTTS (600h) 和 LibriHeavy (50K h) 上进行,不是最终 Emilia 100K h 配置 [§3.4, Table 4]
4. **未测试鲁棒性**: 未报告长句、复杂文本、out-of-domain 场景的表现
5. **Semantic Aligner 的对齐质量未可视化**: 未提供 attention map 或对齐可视化证据来验证 aligner 确实学到了合理的对齐
6. **CTC 应用位置未消融**: 论文未报告 CTC loss 应用在哪一层 (第 i 层) 最优,也未报告去掉 CTC loss 的消融结果 [§2.3]

## 点评

ARCHI-TTS 在 NAR flow-matching TTS 的两个核心痛点上给出了简洁有效的解决方案。Semantic Aligner 的 mask embedding 设计巧妙地回避了 padding 的浪费和 hard duration 的刚性,让对齐在 transformer 内部自然发生。分离 encoder-decoder 带来的 training-free 推理加速是一个实用且低成本的工程创新——不需要蒸馏、不需要额外训练,75% sharing 就能把 RTF 从 0.21 降到 0.09。

然而,客观质量 (WER) 的优势与主观质量 (MOS) 的差距值得关注。WER 1.98% 显著优于 F5-TTS 的 2.42%,但 MOS 反而低于 F5-TTS,暗示 WER 提升可能来自更保守的生成 (更"清晰"但不够自然)。SSIM 在跨说话人场景 (SeedTTS) 上的表现说明低 token rate VAE 的信息压缩确实牺牲了 fine-grained timbre,speaker embedding 只能部分补偿。

整体而言,这是一篇工程扎实、思路清晰的 NAR TTS 工作,289M 参数在 8 张 5090 上 4 天训练就达到 SOTA WER,效率值得称赞。但在主观质量和说话人相似度方面仍有提升空间。

## 可复用的 idea

1. **分离 DiT 架构的 encoder 特征复用**: 将 diffusion/flow 模型的条件处理和去噪解码分开,使 encoder 输出可跨步共享。这个思路可迁移到任何 DiT-based 生成模型 (图像/音频/视频)。关键是确保 encoder 输出不依赖当前 timestep 的噪声级别。
2. **Learnable mask embedding 作为 temporal canvas**: 用可学习的 mask embedding 序列表示目标长度,让 transformer 在文本特征和 mask 之间学习对齐。比 padding 更高效 (mask embedding 共享参数),比 hard duration 更灵活。
3. **CTC loss 在 DiT 中间层做语义监督**: 在生成模型的中间层添加 CTC 约束,确保隐表示保持文本可解码性。这是一种低成本的语义正则化方法,可能适用于任何条件生成模型。
4. **低 token rate VAE + 显式 speaker embedding 的互补**: 当音频表示被极度压缩时,说话人身份信息丢失严重,需要外部 speaker encoder 补偿。这对设计低帧率 audio tokenizer 是重要提醒。
