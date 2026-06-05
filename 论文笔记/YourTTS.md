---
type: paper
tier: deep
title: "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for everyone"
arxiv_id: "2112.02418"
source: "Sources/YourTTS.pdf"
authors: [Edresson Casanova, Julian Weber, Christopher Shulby, Arnaldo Candido Junior, Eren Golge, Moacir Antonelli Ponti]
year: 2022
venue: "ICML 2022 (arXiv v4: Apr 2023)"
tags: [TTS, zero-shot, multi-speaker, multilingual, VITS, voice-conversion, speaker-adaptation, flow-based, cross-lingual]
concepts: ["[[SpeakerEmbedding]]", "[[SpeakerAdaptation]]", "[[NeuralVocoder]]", "[[SpeakerVerification]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeakerEmbedding]], [[NeuralVocoder]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeakerEmbedding]]✓, [[NeuralVocoder]]✓, [[ResidualVectorQuantization]]✓ (tag overlap), [[SemanticvsAcousticTokens]]✓ (tag overlap) | 过滤: [[SpeakerAdaptation]](待确认), [[VoiceCloningTaxonomy]](待确认), [[VITS]](model, 待确认) | 未命中但可能相关: Variational Autoencoder for TTS, Speech-Text Alignment

**谱系定位:** YourTTS (2021.12, Coqui/USP) 是基于 VITS 架构的零样本多说话人多语言 TTS 系统。在谱系上属于 "Flow-based end-to-end TTS + external speaker encoder" 路线,是 SC-GlowTTS (2021) 的后续工作。它早于 VALL-E (2023) 和 codec LM 时代,代表了 pre-LLM TTS 中零样本方案的重要节点。KB 中 [[VoiceCloningTaxonomy]] (pending-review) 已将 YourTTS 归入 "3-5 languages" 多语言 voice cloning 类别。

**已有认知:** KB 已知: (1) VITS (ICML 2021) 是首个 VAE+Flow+GAN 端到端 TTS,已有模型页; (2) Speaker Embedding 页记录了 H/ASP (multi-layer attention statistical pooling) encoder 为 YourTTS 所用; (3) Speaker Adaptation 页列举了 speaker encoder 在 zero-shot 场景中的核心角色; (4) Voice Cloning Taxonomy 将 zero-shot VC 定义为 "无需微调,使用 speaker encoder 从短音频推断说话人特征"。

**创新判断:** 相对 KB 已有知识,本文的核心创新: (1) 首次将 multilingual approach 引入 zero-shot multi-speaker TTS scope [§1]; (2) 在 VITS 架构上引入外部 speaker encoder + language embedding + Speaker Consistency Loss (SCL) 实现跨语言零样本 TTS; (3) 证明用单个目标语言说话人的数据集即可实现该语言的 ZS-TTS; (4) 证明 <1 min fine-tuning 可显著提升 speaker similarity,为 speaker adaptation 提供低成本方案。

> [!summary] 速查
> - **一句话**: 基于 VITS 的首个多语言零样本 TTS+VC 统一系统,在 VCTK 上达到 SOTA ZS-TTS,仅用单说话人数据即可实现目标语言的零样本多说话人合成
> - **路线**: Input Text + Lang ID → Char Embedding + Language Embedding → Transformer Encoder → MAS/Alignment Generation → Flow-Based Decoder (affine coupling) ← External Speaker Encoder (H/ASP) → HiFi-GAN Generator → Waveform; 推理: Posterior Encoder → VAE → HiFi-GAN
> - **指标**: VCTK: SECS 0.864, MOS 4.24, Sim-MOS 4.17 (Exp 2+SCL, best) [Table 1]; LibriTTS: SECS 0.856, MOS 4.18, Sim-MOS 4.07 (Exp 4+SCL) [Table 1]; MLS-PT: MOS 4.11, Sim-MOS 3.19 (Exp 3+SCL, 1 male speaker only) [Table 1]; Voice Conversion en-en: MOS 4.20, Sim-MOS 4.07 [Table 2]; Speaker Adaptation: <1 min FT → SIM 从 0.814 升至 0.896 (EN-F) [Table 3]
> - **可借鉴**: (1) Language embedding 拼接到 char embedding 实现多语言; (2) External speaker encoder conditioning 所有 affine coupling layers; (3) Speaker Consistency Loss 提升 speaker similarity; (4) <1 min speaker adaptation 可行
> - **局限**: SCL 实际未 propagate gradients (Erratum, Appendix A); Duration predictor 不稳定导致部分不自然韵律 [§7]; 葡萄牙语性别偏差大 (no female training data) [§4.3]; 不使用 phoneme 导致发音错误 [§7]; 未在 LJSpeech 等标准单说话人 benchmark 评估

## 核心问题

1. 如何构建一个同时支持零样本多说话人 TTS 和零样本 voice conversion 的统一系统? [§1]
2. 如何将多语言能力集成到零样本 TTS 框架中? [§1]
3. 仅有单个目标语言说话人的数据,能否实现该语言的零样本多说话人合成? [§1]
4. 用极少量数据(<1 min) fine-tuning 能否显著提升 speaker similarity? [§6]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

YourTTS 基于 VITS [19] 架构,增加若干 modifications for zero-shot multi-speaker and multilingual training [§2, Fig 1]:

1. **Text Encoder**: Transformer-based, 输入 raw text (非 phoneme) + 4-dim language embedding, 10 transformer blocks, hidden dim 196 [§2] [论文原文]
2. **Flow-Based Decoder**: 4 affine coupling layers [21], each layer = 4 WaveNet residual blocks [22] (same as VITS) [§2]
3. **Posterior Encoder**: 16 non-causal WaveNet residual blocks, input = linear spectrogram → latent variable z [§2] [论文原文]
4. **HiFi-GAN V1 Vocoder** [23]: Generator + discriminator modifications from [19] [§2]
5. **External Speaker Encoder**: H/ASP model [29] pre-trained with Prototypical Angular [30] + Softmax loss on VoxCeleb2 [31], EER = 1.967 [§3.1]
6. **Stochastic Duration Predictor**: Flow-based probabilistic duration modeling [§2] (from VITS [19])

### 关键设计选择

**多语言扩展** [§2]:
- 使用 raw text 而非 phoneme,避免对每种语言都需要 G2P 工具 [§2] [论文原文]
- [agent 解读] 这是一个 trade-off: 避免了 G2P 依赖但引入了发音错误风险 (作者在 §7 也承认了这个局限性)。后续 VITS 2 和 CosyVoice 都使用了 phoneme/supervised tokenizer 来缓解此问题。
- 4-dim trainable language embedding 拼接到 char embedding [§2] [论文原文]
- [agent 解读] 4 维对于区分 3 种语言已足够,但对于更多语言可能需要更高维度。

**Speaker Conditioning** [§2]:
- 所有 affine coupling layers (flow decoder) + posterior encoder 的 residual blocks 使用 global conditioning [22] on external speaker embeddings [§2] [论文原文]
- Speaker embedding 与 text encoder output 和 decoder output 通过 element-wise summation (经 linear projection 对齐维度) 后传入 duration predictor 和 vocoder [§2, Fig 1] [论文原文]
- [agent 解读] 这种"全局条件注入每一层"的设计确保了 speaker identity 在生成过程中的深度融合,比简单的 concatenation 更有效。这一设计选择已被 KB 中 Speaker Embedding 页记录为 "Global conditioning" 注入方式。

**Speaker Consistency Loss (SCL)** [§2]:
- 使用预训练 speaker encoder 从生成音频和真实音频提取 embedding,计算 cosine similarity [§2]
- L_SCL = (-alpha/n) * sum(cos_sim(phi(g_i), phi(h_i))) [Eq 1], alpha=9 [§2]
- [论文原文] 灵感来自 [26] 的 cross-lingual speaker adaptation 工作

> [!warning] **Erratum (Appendix A)**: 作者后续发现 SCL 的梯度在训练中实际未被 propagate (implementation bug)。使用 SCL 的实验等价于"训练更多 steps without SCL"。此 bug 在 Coqui TTS v0.12.0+ 中已修复。

**Zero-Shot Voice Conversion** [§5]:
- 利用 VITS 架构的 posterior encoder 是 speaker-independent 的特性: 输入 source audio → posterior encoder → z (content) + target speaker embedding → flow decoder → vocoder → target voice [§5] [论文原文]
- [agent 解读] 这是一种优雅的设计: VITS 本身训练为 TTS 系统,但因为 posterior encoder 不注入 speaker identity 信息,其 latent z 天然编码 content,配合 speaker-conditioned decoder 即可实现 VC。

**Speaker Adaptation** [§6]:
- Fine-tuning 整个 TTS 模型 with Speaker Consistency Loss, alpha=9 [§6]
- 使用 20-61s 参考音频训练 1500 steps,weighted random sampling 保证 adapted speakers 占 1/4 batch [§6]
- 结果: <1 min speech → speaker similarity 和 naturalness 显著提升,如 EN-F: Sim-MOS 从 4.12→4.17 (GT: 4.17), SECS 0.814→0.896 (GT: 0.894) [Table 3] [论文原文]

### 训练策略

**四组实验** [§3.3]:
1. **Exp 1**: VCTK only (monolingual EN)
2. **Exp 2**: VCTK + TTS-Portuguese (bilingual)
3. **Exp 3**: VCTK + TTS-Portuguese + M-AILABS French (trilingual)
4. **Exp 4**: Exp 3 + 1151 additional EN speakers from LibriTTS train-clean-100/360

每组均有 +SCL 变体。迁移学习: 从 LJSpeech 1M steps → 继续训练 200k steps (Exp 1), 后续实验从前一实验继续 ~140k steps [§3.3]

**Datasets** [§3.2]:
- VCTK: 44h, 109 speakers, 48kHz → train/dev/test (11 unseen speakers) [§3.2]
- TTS-Portuguese: 10h, single speaker, 48kHz→16kHz [§3.2]
- M-AILABS French: 2F+3M speakers, 71h, 16kHz [§3.2]
- Common Voice: 4 speakers for adaptation [§3.2]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SECS (VCTK) | 0.864 (Exp 2+SCL) | 0.824 (GT=0.824, Attentron ZS=0.731) | VCTK test (11 unseen) | [Table 1] |
| MOS (VCTK) | 4.24 (Exp 2+SCL) | 4.26 (GT), 3.86 (Attentron ZS) | VCTK | [Table 1] |
| SECS (LibriTTS) | 0.856 (Exp 4+SCL) | 0.931 (GT) | LibriTTS test-clean | [Table 1] |
| MOS (LibriTTS) | 4.18 (Exp 4+SCL) | 4.22 (GT) | LibriTTS | [Table 1] |
| MOS (MLS-PT) | 4.11 (Exp 3+SCL) | 4.61 (GT) | MLS Portuguese | [Table 1] |
| VC en-en MOS | 4.20 | AutoVC 3.54, NoiseVC 3.38 | VCTK 8 speakers | [Table 2] |
| VC en-en Sim-MOS | 4.07 | AutoVC 1.91, NoiseVC 3.05 | VCTK | [Table 2] |
| VC pt-pt MOS | 3.64 | - | MLS Portuguese | [Table 2] |
| Adapt EN-M SECS | ZS 0.851→FT 0.896 | GT 0.875 | VCTK 1 speaker, 61s | [Table 3] |
| Adapt EN-F SECS | ZS 0.814→FT 0.896 | GT 0.894 | VCTK 1 speaker, 44s | [Table 3] |
| Adapt PT-M SECS | ZS 0.817→FT 0.915 | GT 0.880 | MLS 1 speaker, 31s | [Table 3] |

## 局限性

1. **SCL 实现 bug**: Speaker Consistency Loss 的梯度未被传播,+SCL 实验等价于更多训练步数 [Appendix A]。这使得 SCL 的有效性存疑
2. **Duration predictor 不稳定**: 部分说话人和句子产生不自然的时长,导致韵律问题 [§7] [论文原文]
3. **不使用 phoneme 导致发音错误**: 特别在葡萄牙语中更严重 [§7] [论文原文]
4. **葡萄牙语性别偏差**: 训练数据仅有 1 名男性说话人,导致女性声音质量差 (Sim-MOS male 3.19 vs female 2.84) [§4.3] [论文原文]
5. **VCTK dataset 局限**: 仅 109 speakers,录音条件单一,限制了泛化到不同录音环境的能力 [§3.3] [论文原文]
6. **VC 跨性别转换困难**: 葡萄牙语男→英语女性别转换质量差,跨语言跨性别尤为困难 [§5.2]
7. **评估方法有限**: 未使用 SEED-TTS-Eval 等后续标准 benchmark (因时代原因)

## 点评

**YourTTS 在谱系中的意义** [agent 解读]: YourTTS 是 pre-LLM 时代零样本 TTS 的里程碑。它证明了基于 VITS + external speaker encoder 的方案可以同时实现零样本 TTS 和 VC,且可扩展到多语言。尽管 codec LM (VALL-E, 2023) 之后重新定义了零样本 TTS 的范式,YourTTS 的一些设计思想(全层 speaker conditioning, speaker adaptation with <1 min data)仍然影响着现代系统。

**单说话人目标语言的零样本 TTS** [agent 解读]: YourTTS 最令人惊讶的发现之一是: 在目标语言(如葡萄牙语)中仅有 1 个说话人的训练数据,模型仍能在该语言中实现有意义的零样本多说话人合成 (MLS-PT Sim-MOS 3.19)。这暗示了 speaker encoder 的 speaker representation 可跨语言迁移,尽管性能仍有显著差距。

**SCL Bug 的教训** [agent 解读]: Erratum 中报告的 SCL 梯度未传播 bug 是一个值得警惕的案例。这意味着: (1) 使用 SCL 的实验结论需要重新解读 — SCL 的正面效果可能只是因为多训练了 50k steps; (2) 工业和学术中的 loss function 实现需要验证梯度是否确实流过所有预期路径。

**与 CosyVoice/FireRedTTS 的对比** [agent 解读]: YourTTS (2021) 与 CosyVoice (2024) 在零样本 TTS 上的根本区别: (1) YourTTS 在波形域端到端 (VAE+Flow+GAN),CosyVoice 在 token 域分两阶段 (LM+CFM); (2) YourTTS 依赖 external speaker encoder,CosyVoice 通过 in-context learning 隐式学习 speaker identity; (3) YourTTS 需要多语言训练数据,CosyVoice 通过 cross-lingual transfer 更灵活。这反映了 3 年间 TTS 范式从 "end-to-end waveform" 到 "discrete token LM" 的根本转变。

## 可复用的 idea

1. **全层 speaker conditioning**: External speaker embedding 通过 global conditioning 注入所有 affine coupling layers + posterior encoder,确保 speaker identity 深度融合
2. **VITS + speaker encoder = ZS-TTS + ZS-VC**: 利用 VITS posterior encoder 的 speaker-independent 特性实现零样本 voice conversion
3. **<1 min speaker adaptation**: Fine-tuning 整个模型 1500 steps with weighted sampling 可显著提升 speaker similarity
4. **单说话人目标语言 ZS-TTS**: 仅需 1 个目标语言说话人即可启用该语言的零样本多说话人合成
5. **Language embedding 作为多语言扩展**: 4-dim trainable embedding 拼接到 char embedding,简单有效

---

检索命中: [[SpeakerEmbedding]], [[NeuralVocoder]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]] | 过滤: [[SpeakerAdaptation]](pending-review), [[VoiceCloningTaxonomy]](pending-review), [[VITS]](pending-review) | 未命中但可能相关: Variational Autoencoder for TTS, Speech-Text Alignment
