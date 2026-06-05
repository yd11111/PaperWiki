---
type: paper
tier: deep
title: "Fish-Speech: Leveraging Large Language Models for Advanced Multilingual Text-to-Speech Synthesis"
arxiv_id: "2411.01156"
source: "https://arxiv.org/abs/2411.01156"
authors: [Shijia Liao, Yuxuan Wang, Tianyu Li, Yifan Cheng, Ruoyi Zhang, Rongzhi Zhou, Yijin Xing]
year: 2024
venue: "Technical Report"
tags: [TTS, LLM, dual-AR, GFSQ, vocoder, multilingual, zero-shot, voice-cloning, open-source]
concepts: ["[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[FiniteScalarQuantization]]", "[[ResidualVectorQuantization]]", "[[NeuralVocoder]]", "[[SemanticvsAcousticTokens]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[LLM-basedTTS]], [[SpeechTokenizer]], [[NeuralVocoder]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[NeuralVocoder]]✓, [[ResidualVectorQuantization]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[FiniteScalarQuantization]](pending-review) | 未命中但可能相关: 无

**[[LLM-basedTTS]]**: Fish-Speech 属于 LLM-based TTS 范式。与 VALL-E 等使用 codec tokens 不同, Fish-Speech 使用 LLM 直接处理文本输入 (无需 G2P),并引入 Dual-AR 架构处理 GFSQ codebook tokens。这在 LLM-based TTS 的演进中代表了"用 LLM 替代传统文本前端"的路线。

**[[SpeechTokenizer]]**: Fish-Speech 使用 Grouped Finite Scalar Vector Quantization (GFSQ) 作为 speech tokenizer 的量化层,与 CosyVoice 系列的 FSQ 路线类似,但采用分组策略。GFSQ 声称实现 100% codebook 利用率,解决了 RVQ 的 codebook collapse 问题。

**[[NeuralVocoder]]**: Fish-Speech 提出了 Firefly-GAN (FF-GAN) 作为 vocoder,基于 EVA-GAN 改进,用 ParallelBlock 替代 HiFi-GAN 的 MRF 模块,并集成 GFSQ 量化层。

**[[ResidualVectorQuantization]]**: GFSQ 是 RVQ 的替代方案。Fish-Speech 在消融实验中对比了 GFSQ 与 RFSQ、RVQ、GRFSQ,GFSQ 表现最优。

**[[SemanticvsAcousticTokens]]**: Fish-Speech 的 Dual-AR 设计隐式处理了 semantic-acoustic 的分离: Slow Transformer 处理文本→semantic tokens,Fast Transformer 处理 codebook→acoustic details。

> [!summary] 速查
> - **一句话**: Dual-AR (Slow+Fast Transformer) + GFSQ + LLM 替代 G2P 的多语言 TTS 框架,用 FF-GAN 解码
> - **路线**: Text → LLM (Slow Transformer) → Semantic Tokens → Fast Transformer → GFSQ Codebook Tokens → FF-GAN (ConvNeXt Encoder + FireFly Generator) → Mel → Waveform
> - **指标**: WER 6.89% (GT 9.22%), SIM 0.914/0.762 (Resemblyzer/SpeechBrain), MOS 4.05 [Table 1-3]
> - **可借鉴**: (1) Dual-AR 分离全局语义和局部声学的 idea; (2) GFSQ 实现 100% codebook 利用率; (3) LLM 直接处理文本免去 G2P 依赖
> - **局限**: 仅在 voice cloning 场景评估,缺少 Seed-TTS-eval 等标准 benchmark; 对比基线较少 (CosyVoice/F5-TTS/reecho); 无中文实验结果

## 核心问题

Fish-Speech 要解决的核心问题:
1. **G2P 瓶颈**: 传统 TTS 依赖 G2P 转换,跨语言维护成本高,处理多音字和罕见词困难 [§1]
2. **Codebook 稳定性**: RVQ 等量化方法存在 codebook collapse、利用率低的问题 [§3.2.2]
3. **生成效率**: 基于 diffusion/flow 的方法推理慢,不适合实时交互 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Fish-Speech 采用两阶段架构 [Fig 1]:

**阶段 1 - 训练 VQ 编码器**: ConvNeXt Encoder 将 Mel Spectrogram 编码为 latent features,经 GFSQ 量化为离散 Quantized Mel Tokens [§3.2]

**阶段 2 - AR 模型训练**: 文本输入 → Dual-AR (Slow Transformer + Fast Transformer) → 预测 Quantized Mel Tokens [§3.1, Fig 1]

**推理**: Text + Prompt Tokens → AR Model → Quantized Mel Tokens → FF-GAN Decoder → Waveform [Fig 1]

### 关键设计选择

#### 1. Dual-AR Architecture [§3.1, Fig 2]

**WHY**: [论文原文] 单一 Transformer 处理 GFSQ 多 codebook 输出时序列过长,且 codebook embedding 处理效率低 [§3.1]。[agent 解读] 这本质上是将 semantic-level 和 acoustic-level 的建模职责分离到两个不同尺度的 Transformer 中。

**Slow Transformer**: 处理输入文本 embedding,生成中间 hidden states h 并预测 semantic tokens [Eq 1-2]
- h = SlowTransformer(x)
- z = W_tok * Norm(h)

**Fast Transformer**: 接收 Slow Transformer 的 hidden states 和 codebook embeddings,预测 acoustic codebook logits [Eq 3-5]
- h_tilde = [h; c], (h^fast)
- h^fast = FastTransformer(h_tilde, (h^fast))
- y = W_cbk * Norm(h^fast)

**优势** [§3.1.2]:
1. 全局-局部层级处理提升 GFSQ 序列生成稳定性 [论文原文]
2. Fast Transformer 的 codebook embedding 处理无显著计算开销 [论文原文]
3. 可直接处理多语言文本,无需 G2P [论文原文]

#### 2. Grouped Finite Scalar Vector Quantization (GFSQ) [§3.2.2]

**WHY**: [论文原文] RVQ 存在 codebook collapse 和利用率低的问题;FSQ 的固定网格方案在低维下有效但扩展性受限 [§3.2.2]

**HOW**: 将输入特征矩阵 Z 分为 G 组,每组独立进行 scalar quantization [Eq 7-12]:
1. **Feature Grouping**: Z = [Z^(1), Z^(2), ..., Z^(G)] [Eq 7]
2. **Scalar Quantization**: 每个 scalar z^(g)_{b,c,l} 独立量化 [Eq 8]
3. **Index Generation**: 映射为 codebook index [Eq 9]
4. **Decoding**: 从各组 codebook 查表重建 [Eq 9]
5. **Concatenate + Upsample**: 拼接所有组的量化向量并上采样恢复原始尺寸 [Eq 10-11]

**结果**: 实现接近 100% codebook 利用率,优于 RFSQ、RVQ、GRFSQ [§3.2.3]

#### 3. Firefly-GAN (FF-GAN) [§3.2, Fig 3]

**WHY**: [论文原文] HiFi-GAN 的 MRF 模块效率不够,需要更好地处理 GFSQ 的 typo-codebook 输入 [§3.2]

**HOW**:
- **ConvNeXt Encoder**: 将 Mel Spectrogram 下采样并用 ConvNeXt blocks 提取 latent features [Fig 3 left]
- **FireFly Generator**: 用 ParallelBlock 替代 MRF,实现可配置的卷积核大小和 dilation rates,通过 stack-and-average 融合三个 ResBlock 的输出 [§3.2.1]
- **GFSQ 量化层**: 嵌入 encoder 和 generator 之间 [Fig 3]

#### 4. LLM 替代 G2P [§1, §3]

**WHY**: [论文原文] G2P 依赖语言特定的音素规则和词典,维护成本高,对多音字和罕见词处理不佳 [§1]

**HOW**: 直接将文本 token 作为 LLM 输入,利用 LLM 的语言理解能力隐式学习发音和上下文 [§1]。训练数据中混合多语言文本增强模型的跨语言能力 [§5]

### 训练策略

三阶段训练 [§4.1]:
1. **预训练**: 大批量标准数据,720K 小时多语言语音 (英语 300K + 中文 300K + 其他语言各 20K) [§5]
2. **SFT**: 小批量高质量数据微调
3. **DPO**: 人工标注正负样本对训练 (不含 vocoder 阶段)

**训练细节** [Appendix A]:
- AR: 8x H100 80G, 一周
- Vocoder: 8x 4090, 一周
- Optimizer: AdamW (β1=0.9, β2=0.98, ε=10^-8)
- LR: 5×10^-4, cosine decay with 2K warmup
- Batch: 1M tokens, 500K steps

**推理加速** [§4.2]:
- KV-cache + torch compile
- RTF ~1:5 (RTX 4060) / ~1:15 (RTX 4090)
- First-packet latency: 150ms

## 实验

| 指标 | Fish-Speech | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (%) | **6.89** | GT: 9.22, reecho: 11.92, F5-TTS: 13.98, CosyVoice: 22.20 | Voice Cloning (10 speakers, 300 samples) | [Table 1] |
| SIM (Resemblyzer) | **0.914** | GT: 0.921, CosyVoice: 0.936, F5-TTS: 0.905, reecho: 0.887 | 同上 | [Table 2] |
| SIM (SpeechBrain) | **0.762** | GT: 0.770, CosyVoice: 0.813, F5-TTS: 0.787, reecho: 0.636 | 同上 | [Table 2] |
| MOS | **4.05** | GT: 5.00, CosyVoice: 3.80, F5-TTS: 2.90, reecho: 3.76 | 同上 | [Table 3] |

**关键发现**:
- Fish-Speech WER (6.89%) 低于 Ground Truth (9.22%) [Table 1],说明合成语音的文本一致性甚至优于原始录音 [agent 解读]
- SIM 接近 GT (差距仅 0.76%-1.04%),显著优于 F5-TTS 和 reecho [Table 2]
- MOS 4.05 在所有基线中最高 (p < 0.05) [§6.3]

## 局限性

1. **评估范围窄**: 仅评估了单语 voice cloning 场景,未涉及跨语言合成、情感控制、Seed-TTS-eval 等标准化 benchmark [§6]
2. **基线选择**: 仅对比了 CosyVoice、F5-TTS、reecho 三个基线,缺少 VALL-E、MaskGCT、Seed-TTS 等主流系统 [Table 1-3]
3. **无消融实验**: 缺乏 GFSQ vs RVQ、Dual-AR vs 单一 AR 的系统消融 (仅在 §3.2.3 有简短提及)
4. **开源但技术报告简略**: 论文仅 10 页正文,很多设计细节不够充分 (如 GFSQ 的组数 G、codebook 大小等未明确说明)
5. **DPO 训练**: 提到使用 DPO 但未报告其对指标的提升 [§4.1]

## 点评

Fish-Speech 的核心贡献在于 Dual-AR + GFSQ 的架构组合和 LLM 替代 G2P 的思路。Dual-AR 将全局语义和局部声学分离到两个 Transformer 中,是对 VALL-E AR+NAR 两阶段的一种替代设计。GFSQ 继承了 FSQ 的 100% codebook 利用率优势,通过分组策略扩展了表达能力。

但论文存在明显不足: 评估不够全面 (仅 voice cloning、仅单语、基线少)、消融实验缺失、技术细节不完整。与同期 GLM-TTS、CosyVoice 3 等工作相比,实验严谨性有差距。作为开源项目 (github.com/fishaudio/fish-speech),其工程价值可能大于论文的学术贡献。

## 可复用的 idea

1. **Dual-AR 架构**: Slow (全局语义) + Fast (局部声学) 的分离思路可推广到其他需要层级建模的生成任务
2. **GFSQ**: 分组 FSQ 实现 100% codebook 利用率,可作为 RVQ 的替代方案用于音频编码
3. **LLM 替代 G2P**: 省去语言特定的音素规则维护,对多语言 TTS 有实际价值
4. **150ms first-packet latency**: KV-cache + torch compile 的加速组合值得参考

---

检索命中: [[LLM-basedTTS]], [[SpeechTokenizer]], [[NeuralVocoder]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]] | 过滤: [[FiniteScalarQuantization]](pending-review) | 未命中但可能相关: 无
