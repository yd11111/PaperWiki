---
type: paper
tier: deep
title: "SC VALL-E: Style-Controllable Zero-Shot Text to Speech Synthesizer"
arxiv_id: ""
source: "Sources/SC-VALLE.pdf"
authors: [Daegyeom Kim, Seongho Hong, Yong-Hoon Choi]
year: 2023
venue: "IEEE Access (CC BY 4.0)"
tags: [TTS, zero-shot, style-control, VALL-E, codec-LM, emotion, prosody, Korean, style-tokens, expressive-TTS]
concepts: ["[[Codec Language Model]]", "[[Global Style Tokens]]", "[[Residual Vector Quantization]]", "[[Prosody Modeling]]", "[[Emotion Control in TTS]]"]
models: ["[[模型库/EnCodec|EnCodec]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[Residual Vector Quantization]], [[Prosody Modeling]], [[Speaker Embedding]], [[Neural Vocoder]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Residual Vector Quantization]]✓, [[Prosody Modeling]]✓, [[Speaker Embedding]]✓, [[Neural Vocoder]]✓ | 过滤: [[Codec Language Model]](pending-review), [[Global Style Tokens]](pending-review), [[Emotion Control in TTS]](pending-review) | 未命中但可能相关: 无

**谱系定位**: SC VALL-E (2023) 是 VALL-E 的风格控制扩展。KB 中 [[Global Style Tokens]] [待确认] 页已将其列为"GST 思想在 LLM-TTS 中的延续"。在 [[Prosody Modeling]] 页的韵律控制演进中,SC VALL-E 处于"Reference Encoder (GST, 2018) → LLM in-context (VALL-E, 2023)"这一过渡节点 — 它保留了 VALL-E 的 codec LM 架构,同时引入了类似 GST 的 style token bank 实现显式风格控制。

**已有认知**:
- [[Codec Language Model]] [待确认] 页记录了 VALL-E 的 AR+NAR 两阶段建模: AR 生成第 1 层 RVQ tokens,NAR 生成第 2-8 层。SC VALL-E 在 NAR block 中引入 style network,从 quantized tokens 中学习风格表征。
- [[Prosody Modeling]] 页记录了 GST-Tacotron 的 unsupervised style token bank + attention 机制。SC VALL-E 将此机制从 attention-based TTS 迁移到 codec LM 框架,用 multi-head attention 从 quantized tokens 中提取 style tokens,并通过 scale factor (0.5-2.5) 实现风格强度控制。
- [[Emotion Control in TTS]] [待确认] 页记录了情感控制的多种方法。SC VALL-E 提供了一种 codec LM 框架下的 unsupervised emotion/style discovery 方案。

> [!summary] 速查
> - **一句话**: 在 VALL-E 的 NAR block 中加入 style network (N=10 style tokens + multi-head attention),从 quantized audio tokens 中无监督学习 emotion/speed/pitch 等风格维度,通过 scale factor c_i ∈ [0.5, 2.5] 实现推理时的显式风格控制
> - **路线**: Text → KoG2P → Phonemes → [Audio prompt → EnCodec → RVQ tokens (8 layers)] → AR decoder (stage 1) → NAR decoder + Style Network (N=10 tokens, 8-head attention) → Predicted RVQ tokens → DeCodec → Speech
> - **指标**: WER 0.25 / FVE 64.19 / F0GPE 56.53 (vs VALL-E 0.30/64.75/56.79, GST-Tacotron 0.21/64.24/54.53) [Table 3]; CMOS +4.10 / SMOS 4.10 (vs VALL-E -0.40/3.90) [Table 4]
> - **可借鉴**: (1) Style token bank 在 codec LM NAR block 中的应用 — 用 scale factor 实现推理时风格调节,无需标签; (2) Style token 自动发现语义维度 (c1=emotion, c2=speed, c3=volume/pitch); (3) 韩语 TTS 的 KoG2P 音素分解方案
> - **局限**: (1) 韩语专属,未验证跨语言; (2) 仅训练 1.5 epochs (21,497h 数据, 68 天 4×A100),训练不充分; (3) style token c4-c10 未学到明显风格维度; (4) 未与现代 LLM-TTS (CosyVoice, Seed-TTS) 对比; (5) 代码开源但数据为韩国 AI Hub 内部数据

## 核心问题

VALL-E 通过 in-context learning 实现了零样本 TTS,但缺乏对语音风格 (emotion, speed, pitch, volume) 的显式控制能力 [§I]:
- VALL-E 的 sampling-based inference 产生多样化输出,但这种多样性不可控 [§I] [论文原文]
- GST-Tacotron/VAE-Tacotron 可以控制风格,但在短参考音频 (3s) 场景下无法准确复制说话人风格 [§IV.B] [论文原文]
- Speaker adaptation/encoding 方法在训练集外说话人上质量下降 [§I] [论文原文]

SC VALL-E 的核心问题: **如何在 VALL-E 的零样本 codec LM 框架中加入 GST 式的可控风格建模,使得推理时可以通过 scale factors 调节 emotion、speed、pitch 等维度?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SC VALL-E 包含三个部分 [§II, Fig 1]:

1. **Text Embedding and Audio Segment Quantization Part**: 文本→音素 (KoG2P) + 音频→EnCodec tokens (8 层 RVQ, 6 kbps)
2. **Quantized Token Prediction Part**: AR decoder (stage 1) + NAR decoder + Style Network (stage 2-8)
3. **Audio Reconstruction Part**: DeCodec (EnCodec decoder) 重建波形

### 关键设计选择

#### 设计选择 1: Style Network

**WHY**: [论文原文] VALL-E 的 NAR block 已经从 quantized tokens (stage 2-8) 中学习详细的声学信息 (speaking rate, voice volume, emotions, intonation, pitch, background noise)。如果能从这些 tokens 中提取并分离出不同的风格维度,就可以实现显式控制 [§II.A]。

**HOW** [§II.B, Fig 2, Eq.1]:

1. **Prompt Embedding Layer**: 将 8 层 quantized tokens b_1, ..., b_8 (每层长度 L_T) 输入到 Prompt Embedding Layer
2. **Style Token Matrix S**: 输出 N=10 个 256 维的 style token vectors, S ∈ R^{N×256} [§II.B]
3. **Multi-Head Attention** (h_s=8 heads):
   - Input: S
   - V, K, Q: 各自通过 Linear 层
   - Scaled Dot Product Attention 提取 style 信息
   - 输出经 Linear Layer 得到 Y (style embedding matrix) [§II.B]
4. **Style Control Vector c** ∈ R^N: c_i ∈ [0.5, 2.5]
   - Training: c_i = 1 (不修改)
   - Inference: 调节 c_i 值控制风格维度 [§II.B]
5. **Broadcasting**: 将 c broadcast 到 S 的所有列,element-wise 乘积得到 K ∈ R^{N×256}
6. **K 作为 multi-head attention 的 key**: 在 NAR transformer decoder 的 attention 中使用 [§II.B]

**Style Embedding Matrix**: Y = f_θ(b_1, ..., b_8, c) [Eq.1]

[agent 解读] 设计的巧妙之处在于: style control vector c 通过 element-wise 乘法缩放 style tokens,然后作为 attention 的 key 参与解码。这意味着增大某个 c_i 会增强对应 style token 的 attention 权重,从而放大该风格维度的影响。这比 GST 的直接 weight 调节更加柔和 — 通过 attention 机制间接影响,避免了极端值产生的不自然语音。

#### 设计选择 2: Style Tokens 的 Key 角色

**WHY**: [论文原文] 实验确认,将 style tokens K 作为 multi-head attention 的 key (而非 query 或 value) 可以实现更一致的风格控制 [§II.B]。

[agent 解读] Key 决定了 attention 的"关注模式"— 哪些位置/维度被强调。将 style 信息放在 key 位置,使 decoder 在预测每个 token 时都"通过 style filter 来观察上下文",实现了全局一致的风格影响。如果放在 value 位置,则更像"给输出加 style 偏置",容易不自然。

#### 设计选择 3: AR/NAR Split

**HOW** [§II.A]:
- **AR decoder** (causal attention): 预测第 1 层 RVQ tokens,决定语音时长
  - 输入: phoneme 序列 a + first layer tokens b_1
  - 不含 style network
- **NAR decoder** (non-causal attention): 预测第 2-8 层 RVQ tokens,包含详细声学信息
  - 输入: phoneme 序列 a + style embedding Y + 前 d-1 层的 acoustic information
  - 含 style network

[agent 解读] Style network 仅在 NAR block 中,因为: (1) 第 1 层 RVQ tokens 主要编码 speaker identity 和 coarse 信息,由 AR 的 in-context learning 从 prompt 获取; (2) 第 2-8 层编码 fine-grained acoustic details (emotion, prosody, etc.),是 style 控制的目标。

### 训练策略

- 韩语 5 类数据集: 命令/对话/情感/外语发音/韩语语音,共 14,050 说话人, 21,497 小时 [Table 1]
- 音素化: KoG2Padvanced (初声/中声/终声分离) [§III.C]
- EnCodec: encoder bandwidth 6 kbps, 8 层 RVQ [§II.A]
- 训练: 4 NVIDIA A100 80GB, 仅 1.5 epochs, 68 天 [§IV]
- Optimizer: Adam, lr 0.0002, batch 20, temperature 0.2, gradient skipping 100 [Table 2]
- Style network: prompt embedding 1024, heads 8, style tokens N=10, token dim 256 [Table 2]
- Transformer decoder: text embedding 1024, 12 Token Prediction blocks, token dim 256 [Table 2]

## 实验

### Objective Evaluation [Table 3]

| Model | WER↓ | FVE↓ | F0GPE↓ | 出处 |
|---|---|---|---|---|
| GST-Tacotron | **0.21** | **64.24** | **54.53** | [Table 3] |
| VAE-Tacotron | 0.23 | 54.75 | 44.79 | [Table 3] |
| Original VALL-E | 0.30 | 64.75 | 56.79 | [Table 3] |
| SC VALL-E (ours) | 0.25 | 64.19 | 56.53 | [Table 3] |

[§IV.A] SC VALL-E 的 WER 略高于 Tacotron-based 模型,因为 VALL-E 的零样本 sampling 不如专门训练的 Tacotron 稳定 [论文原文]。FVE/F0GPE 较高归因于风格变化改变了 speaker-specific 的发音特征 [论文原文]。

### Subjective Evaluation [Table 4]

| Model | CMOS | SMOS↑ | 出处 |
|---|---|---|---|
| GST-Tacotron | -1.12 (±0.10) | 3.70 (±0.12) | [Table 4] |
| VAE-Tacotron | -0.92 (±0.08) | 3.90 (±0.09) | [Table 4] |
| Original VALL-E | -0.40 (±0.09) | 3.90 (±0.10) | [Table 4] |
| **SC VALL-E** | **-** | **4.10 (±0.10)** | [Table 4] |

[§IV.B] SC VALL-E 在 CMOS 和 SMOS 上全面超越对比模型 [论文原文]。SMOS 4.10 表明合成语音高度接近参考说话人风格 [论文原文]。GST/VAE-Tacotron 在短参考音频 (3s) 下无法复制未见说话人风格,fallback 到训练集说话人风格 [论文原文]。

### Style Control Visualization [§IV.C, Fig 3, Fig 4]

Style token 自动发现的维度 [论文原文]:
- **c1**: 情感 (0.5=happy, 1.5=anger, 2.5=sadness) — mel spectrogram 和 F0 随 c1 变化显著
- **c2**: 语速 (0.5=慢, 2.5=快) — 高值时语音时长缩短
- **c3**: 音量/音高 (0.5=低, 2.5=高) — F0 和能量同步变化
- **c4-c10**: 未学到显著风格维度 [论文原文]

[agent 解读] 仅 3/10 个 style tokens 学到有意义的维度,可能因为: (1) 1.5 epochs 训练严重不足; (2) N=10 个 tokens 过多,大部分数据中的风格变化可以用 3 个维度解释; (3) 数据集中 emotion/speed/pitch 是最显著的变化轴。

## 局限性

1. **训练不足**: 仅 1.5 epochs (21,497h, 68 天 4×A100),作者明确承认训练不充分 [§IV] [论文原文]
2. **Style token 利用率低**: 10 个 tokens 中仅 3 个学到有意义维度 [§IV.C] [论文原文]
3. **韩语限定**: 仅在韩语数据上训练和评估,未验证跨语言泛化 [agent 解读]
4. **对比基线过时**: 仅与 GST-Tacotron/VAE-Tacotron/VALL-E 对比,未与 CosyVoice, Seed-TTS, MaskGCT 等现代系统对比 [agent 解读]
5. **客观指标有限**: 未报告 speaker similarity (SECS/SSIM),无法量化零样本音色保真度 [agent 解读]
6. **WER 偏高**: 相比 Tacotron-based 方法 WER 更高,内容保真度有改进空间 [Table 3] [论文原文]

## 点评

**与已有工作的差异** [agent 解读]:
- vs GST-Tacotron: GST 在 attention-based TTS 中用 reference encoder 提取 query,从 token bank 做 weighted sum 得到 style embedding。SC VALL-E 在 codec LM 的 NAR block 中用 quantized tokens 做类似操作,但 style embedding 作为 attention key 而非简单 addition。本质上是将 GST 的 unsupervised style discovery 从 attention-based TTS 迁移到 codec LM 框架。
- vs Seed-VC / CosyVoice: 这些现代系统通过 in-context learning 隐式获取 speaker style,不支持显式风格维度控制。SC VALL-E 保留了显式控制能力 (scale factor)。
- vs PromptTTS / InstructTTS: 这些系统用自然语言描述控制风格,SC VALL-E 用数值 scale factor 控制。SC VALL-E 的优势是不需要风格标注,劣势是可解释性较差 (需要人工发现每个 c_i 的语义)。

**核心 insight**: Codec LM (VALL-E) 的 RVQ tokens 天然包含丰富的风格信息 (第 1 层=speaker/coarse, 第 2-8 层=fine-grained acoustics)。通过在 NAR block 中加入 multi-head attention style network,可以从这些 tokens 中无监督地提取出可控的风格维度,实现 zero-shot + style-controllable 的统一。

## 可复用的 idea

1. **Style Network in NAR Block**: 在 codec LM 的 NAR 阶段 (而非 AR) 加入 style network,因为 fine-grained style 信息编码在高层 RVQ tokens 中
2. **Scale Factor Control**: 训练时 c_i=1,推理时 c_i ∈ [0.5, 2.5] — 简洁的无标签风格强度控制方案
3. **Style as Key**: 将 style embedding 作为 attention 的 key (而非 query/value) 实现更一致的风格影响
4. **Unsupervised Style Discovery in Codec LM**: 证明 codec LM 的 RVQ tokens 包含可分离的 style 维度,无需标注即可发现

---

检索命中: [[Residual Vector Quantization]], [[Prosody Modeling]], [[Speaker Embedding]], [[Neural Vocoder]] | 过滤: [[Codec Language Model]](pending-review), [[Global Style Tokens]](pending-review), [[Emotion Control in TTS]](pending-review) | 未命中但可能相关: 无
