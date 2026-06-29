---
type: paper
tier: deep
title: "ZONOS2 Technical Report"
arxiv_id: "2606.24320"
source: "Sources/ZONOS2.pdf"
authors: [Gabriel Clark, Sofian Mejjoute, Mohamed Osman, George Close, Beren Millidge]
year: 2026
venue: "arXiv"
tags: [TTS, MoE, open-source, voice-cloning, zero-shot, multilingual, decoder-only, autoregressive, byte-tokenization, benchmark, DAC, delay-pattern, speaker-embedding, LDA]
concepts: ["[[LLM-basedTTS]]", "[[ResidualVectorQuantization]]", "[[SpeakerEmbedding]]", "[[SpeechTokenizer]]", "[[CodecLanguageModel]]", "[[VoiceCloningTaxonomy]]", "[[TTSEvaluation]]", "[[CodebookCollapse]]", "[[ProsodyModeling]]"]
models: ["[[论文笔记/ZONOS2|ZONOS2]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/FishAudioS2|Fish Audio S2]]", "[[论文笔记/VoxCPM2|VoxCPM2]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/TTSDS2|TTSDS2]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: ["ZTTS1-Eval", "Seed-TTS-Eval", "CV3-Eval", "FLEURS-R", "VoxBlink2", "Common Voice"]
kb_context_sources: 6
status: draft
created: 2026-06-29
updated: 2026-06-29
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[ResidualVectorQuantization]]✓, [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓, [[SpeechTokenizer]]✓, [[CodebookCollapse]]✓, [[SpeechLanguageModel]]✓)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: ZONOS2 属于 LLM-based TTS 范式中的 codec language model 分支,直接在 RVQ-based audio codec tokens 上做自回归生成。与 CosyVoice 3 的 coarse-to-fine (LLM + CFM) 两阶段不同,ZONOS2 采用纯 decoder-only transformer 直接生成多 codebook DAC tokens,使用 delay pattern 处理多 codebook 的帧内依赖。这一路线与 Fish Audio S2、MusicGen 等系统更接近。
>
> **Speaker Embedding 方面**: KB 中 SpeakerEmbedding 概念页记载了两大范式(lookup table vs speaker encoder)。ZONOS2 使用 ECAPA-TDNN speaker encoder 提取 2048-d embedding 再做 LDA 降维至 1024-d,属于 zero-shot speaker encoder 路线。值得注意的是 ZONOS2 引入的 LDA 降维和 audio augmentation 是为解决 speaker embedding 过拟合 (causal leakage) 问题的新方案,这在已有 KB 中尚未记录。
>
> **Audio Tokenizer**: ZONOS2 使用 DAC (Descript Audio Codec, Kumar et al., 2023) 而非主流的 EnCodec/SoundStream,9 codebooks RVQ。KB 中 SpeechTokenizer 概念页覆盖了 RVQ-based 声学 tokenizer 的分类,DAC 属于声学 tokenizer 类别。
>
> **Codebook Collapse**: KB 中有专门概念页记录 codebook collapse 问题。ZONOS2 在 MoE 训练中遇到了 expert routing 不稳定/entropy 崩塌的问题,与 codebook collapse 在机制上有相似性(都是离散分配中的负反馈循环),但 ZONOS2 的问题更特异于 MoE expert balancing 在音频 token 上的表现。
>
> **创新判断**: ZONOS2 的主要创新在于 (1) 首次在开源 TTS 中引入 MoE 架构,(2) 提出基于 LDA + two-phase annealing 的 speaker embedding 防过拟合方案,(3) 提出 ZTTS1-Eval benchmark 扩展了 Seed-TTS-Eval 的语言覆盖和评估维度。这些在已有 KB 中均为新信息。
>
> 检索命中: [[ResidualVectorQuantization]]✓, [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓, [[SpeechTokenizer]]✓, [[CodebookCollapse]]✓, [[SpeechLanguageModel]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[VoiceCloningTaxonomy]](pending-review), [[TTSEvaluation]](pending-review), [[CosyVoice3]](pending-review) | 未命中但可能相关: MoE for TTS (无概念页)

## 速查

> [!summary] 速查
> - **一句话**: 首个开源 MoE TTS 模型(8B total / 900M active),用 delay pattern 自回归生成 DAC tokens,6.2M 小时数据训练,在 voice cloning 和 prosody 维度达到开源最佳
> - **路线**: Text (UTF-8 bytes) + Speaker Embedding (ECAPA-TDNN → LDA → projection) + Rate/Quality tokens → Decoder-only MoE Transformer (28 layers, top-1/top-2 routing) → Delayed DAC tokens (9 codebooks) → DAC decoder → Waveform
> - **指标**: ZTTS1-Eval Clean en: WER 2.76%, UTMOS 3.40, Spk.sim 78.6 (开源最佳 spk.sim); ITW TTSDS2 prosody 开源最佳; Quality Mode en WER 3.99% / UTMOS 3.47 [Table III, Table IV, Fig 5]
> - **可借鉴**: (1) LDA 降维 speaker embedding 以抑制 causal leakage / 延长训练 horizon; (2) Two-phase speaker conditioning annealing (先 crop+mask, 再 full embedding); (3) Byte-level text tokenization 替代 G2P 在大规模训练下的优越性; (4) Audio augmentation on clone audio 实现 quality conditioning 解耦
> - **局限**: (1) MoE expert balancing 在 audio tokens 上仍不稳定,需手动调参干预; (2) Quality Mode 提升 WER/UTMOS 但牺牲 speaker similarity; (3) 中文/日文等非英语 WER 较高; (4) ZTTS1-Eval 由 ZONOS2 团队自行提出,公正性需第三方验证

## 核心问题

ZONOS2 要解决的核心问题是: **如何同时兼顾 TTS 系统的自然度、voice cloning 保真度、多语言能力和推理效率**。现有系统往往只在某一维度突出,例如高质量但控制有限,或推理快但仅支持少数语言。

具体挑战包括:
1. 如何在扩大模型参数量(从 1.6B 到 8B)的同时保持实时推理能力?
2. 如何让 speaker embedding 既保留足够身份信息又不导致训练过拟合?
3. 如何摆脱对 G2P phonemization 的依赖,实现真正的多语言泛化?
4. 如何设计更全面的 TTS benchmark 来区分当前 SOTA 系统?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ZONOS2 采用 decoder-only MoE transformer 架构,总参数 8B,活跃参数 900M。输入端接收文本 (UTF-8 bytes) 和条件信号 (speaker embedding + speaking rate + quality tokens),输出 DAC audio tokens (9 codebooks, delay pattern) [§II, Fig 1]。

**推理流程**:
1. 文本经 UTF-8 byte tokenization 转为字节序列
2. Speaker embedding (ECAPA-TDNN → LDA → linear projection) 作为序列前缀
3. Speaking rate 和 quality conditioning 以 discrete bucket tokens 形式拼接
4. MoE transformer 自回归生成 delayed DAC token frames
5. 反向 shearing 还原对齐的 codebook tokens
6. DAC decoder 重建波形

### 关键设计选择

#### 1. Byte-level Text Tokenization 替代 G2P

**选择**: 直接使用 UTF-8 bytes (bi ∈ [0, 255]) 作为文本输入,弃用前代 Zonos-v0.1 的 phoneme tokenization。

**为什么**: [论文原文] G2P phonemization 的归纳偏置在大规模数据下价值递减。三类典型 G2P 静默失败 [Table I]:
- Code-switched 场景: 中文在英语 tag 下被错误处理
- 过度泛化规则: alpharetrovirus → retroretrovirus
- 语言库存限制: 西班牙语 G2P 无法表示日语发音 "Satoshi" → "Satosi"

[论文原文] "Early experimentation revealed that the value of the inductive bias that phonemization provides diminishes with scale" [§II-B]。Byte-level 方案在数据和模型规模增大后,"first matches and then surpasses its phoneme-based counterpart" [§II-B]。

#### 2. MoE Transformer Backbone

**选择**: 28 层 transformer,前 3 层和最后 1 层 dense,其余 MoE (16 experts, top-1 routing),最后一个 MoE 层 top-2 routing [Table V, Fig 2]。

**为什么这样分层**: [论文原文] Dense first/last layers + top-2 final MoE layer 是为了缓解 MoE 在 DAC token 预测上的 expert balancing 不稳定性。"We found that MoE balancing on audio data is substantially harder than on text data" [§IV-A]。[agent 解读] 前 3 层 dense 可能是因为低层表征需要稳定的共享计算来建立跨 modality 对齐基础,而最后 dense 层确保输出空间一致性。

**Router 设计**: 继承自 ZAYA1-8B,采用 Exponential Depth Averaging (EDA) 混合前层 router 状态 + RMSNorm + 3-layer MLP with GELU 产生 router scores [§II-F, Fig 2]。使用 GQA (4x grouping) + headwise Qwen gating [§II-F]。

**关键发现**: [论文原文] MHA 在早期消融中显著优于 GQA(更稳定、更高质量),但出于推理速度考量选择了 GQA [§VII-A]。Headwise Qwen gating 是所有 gating 变体中最有效的 [§VII-A]。

#### 3. Speaker Embedding: LDA + Two-Phase Annealing

**核心问题**: ECAPA-TDNN 2048-d embedding 从 target utterance 提取,编码了过多非 speaker 信息(时长、噪声、具体词汇、停顿),导致 causal leakage — 模型可通过 speaker embedding 走捷径而非学习泛化解 [§II-C]。

**解决方案一: LDA 降维** (2048-d → 1024-d):
- [论文原文] LDA 估计自 speaker-labeled data,最大化 between-speaker variance / within-speaker variance [§II-C]
- 保留区分不同说话人的方向,衰减同一说话人不同录音间变化的因素(时长、噪声、词汇、停顿等)
- [论文原文] "Without the LDA transformation, we found we could not take enough training steps with the speaker embedding to learn high-quality voice cloning before overfitting occurred" [§II-C]

**解决方案二: Two-Phase Annealing** [§IV-C]:
- **Annealing Phase 1**: Speaker embedding 只从 target audio 的 random crop 提取,对 crop 区域 mask loss。[论文原文] 通过移除 speaker embedding 来源片段的 loss 来减少"作弊"激励 [§IV-C]。同时对 clone audio 做 augmentation (噪声/音乐/codec压缩/混响),target DAC tokens 用 clean audio 计算,迫使模型学习对退化鲁棒
- **Annealing Phase 2**: Speaker embedding 从完整 target sequence 提取,移除 loss masking (否则整个序列都被 mask)。引入 'Quality Mode' token

[agent 解读] 这实质上是一种 curriculum learning: 先在信息受限条件下学习鲁棒表征,再逐步释放完整信息。

#### 4. Delay Pattern for Multi-Codebook Generation

**机制**: 对 9 个 DAC codebooks 施加 shearing operation,codebook j 延迟 j 帧 [Eq. 1-2]:
```
Y[t, j] = X[t-j, j]  if t >= j, else padding
```
将帧内 codebook 依赖转化为序列位置上的自回归依赖 — codebook j+1 在同一帧的 token 在 codebook j 之后生成,可利用前面 codebook 的条件信息 [§II-A]。

**代价**: 流式解码需要 N-1=8 帧的 lookahead buffer [§II-A]。

#### 5. Quality Conditioning

**动机**: [论文原文] 模型对 clone audio 的声学质量高度敏感,低质量 prompt 会降低生成质量 [§II-E]。

**实现**:
1. Clone audio augmentation (噪声/音乐/codec压缩/混响),概率 αAUG,target 仍用 clean audio tokens
2. 声学属性 (SNR, loudness, bandwidth, leading/trailing silence) 编码为 discrete bucket tokens [§II-E]
3. 'Quality Mode' token: 仅在最终 annealing 阶段,在最高质量训练数据子集上引入。推理时选择此 token 可提升 intelligibility (WER 降低) 但牺牲 speaker similarity [§II-E, §VI]

### 训练策略

**四阶段训练** [§IV]:

| 阶段 | Steps | Tokens | 关键特征 |
|------|-------|--------|----------|
| Pre-training | 77,500 | 2.9T | 纯 TTS,无 speaker embedding,Max seq 6144 frames, batch 37.7M DAC frames (121.8h) [§IV-A] |
| Mid-training | 15,000 | ~560B | 更严格的 transcript agreement 过滤 [§IV-B] |
| Annealing 1 | 10,000 | — | 引入 speaker embed (cropped + loss mask), augmentation, quality tokens, rate tokens [§IV-C] |
| Annealing 2 | 10,000 | — | Full speaker embed, 移除 loss mask, Quality Mode token, 最高质量数据子集 [§IV-C] |

**优化器**: Muon (base LR 5e-4, Muon LR 5e-3), weight decay 0.1, gradient clip 0.5, 100-step warmup, cosine decay [§IV-A]。

**训练目标**: Masked NLL over non-padding audio targets [Eq. 6] + MoE router balancing loss (bias-based, separate AdamW optimizer) [Eq. 7-8]。Logits 使用 soft-capping (τ=15, 来自 Gemma 2) [Eq. 4-5]。

**数据**: 6.2M 小时,混合公开语音语料、播客、有声书、对话数据集、多语言 web-scale 语音数据、表现力/角色语音数据集 [§III, Fig 3]。使用 multi-ASR ensemble 转录验证(pairwise WER 阈值),不同训练阶段调整阈值严格度 [§III]。

### ZTTS1-Eval Benchmark

**设计动机**: [论文原文] Seed-TTS-Eval 的三个问题: (1) 仅覆盖中英; (2) 使用过时评分模型; (3) 音频为 read speech 不代表真实应用场景 [§V]。

**核心改进** [Table II]:
| 维度 | Seed-TTS-Eval | ZTTS1-Eval |
|------|---------------|------------|
| 语言数 | 2 | up to 17 |
| 语音类型 | 仅 read | read + ITW spontaneous |
| ASR scorer | Whisper-L / Paraformer | Qwen3-ASR |
| Speaker scorer | WavLM | ReDimNet |
| Quality scorer | 无 | MSR-UTMOS |
| Prosody/diversity | 无 | TTSDS2 + DS-WED |

**数据组成**:
- Clean set: 13h, FLEURS-R, 9 languages × 500 utterances [§V]
- ITW set: 1618 utterances from VoxBlink2, ~3h, 17 languages [§V, Table VI]

## 实验

| 指标 | ZONOS2 8B | ZONOS2 QM | Qwen3-TTS 1.7B | Fish S2 Pro | VoxCPM 2 | Cartesia Sonic 3.5 | 数据集 | 出处 |
|------|-----------|-----------|-----------------|-------------|----------|---------------------|--------|------|
| WER ↓ (en) | 2.76 | 3.99 | 1.94 | 3.60 | 4.23 | 2.56 | ZTTS1 Clean | [Table III] |
| UTMOS ↑ (en) | 3.40 | 3.47 | 3.86 | 3.47 | 3.51 | 3.62 | ZTTS1 Clean | [Table III] |
| Spk.sim ↑ (en) | **78.6** | 74.4 | 68.3 | 76.9 | 65.2 | 79.9 | ZTTS1 Clean | [Table III] |
| WER ↓ (zh) | 15.62 | 6.73 | 2.91 | 4.33 | 5.01 | 4.53 | ZTTS1 Clean | [Table III] |
| WER ↓ (en, ITW) | 4.70 | 2.21 | 1.05 | 2.09 | 1.69 | 1.40 | ZTTS1 ITW | [Table IV] |
| UTMOS ↑ (en, ITW) | 2.44 | 2.99 | 3.20 | 2.92 | 2.51 | 3.05 | ZTTS1 ITW | [Table IV] |
| Spk.sim ↑ (en, ITW) | **67.0** | 56.9 | 61.5 | 65.0 | 68.1 | 70.2 | ZTTS1 ITW | [Table IV] |
| TTSDS2 prosody (ITW en) | Best (定性, 见 Fig 5b) | — | — | — | — | — | ZTTS1 ITW | [Fig 5b] |
| DS-WED (en) | Highest diversity (定性, 见 Fig 6) | — | — | — | — | — | ZTTS1 | [Fig 6] |

**Quality Mode 效果分析**:
- 英语: WER 从 2.76% 升到 3.99% (变差), 但 UTMOS 从 3.40 升到 3.47 [Table III]
- 中文: WER 从 15.62% 降到 6.73% (大幅改善), UTMOS 从 3.10 升到 3.21, Spk.sim 从 73.3 升到 81.1 [Table III]
- ITW 全语言: Quality Mode 全面提升 WER 和 UTMOS,但牺牲 speaker similarity [Table IV]

**CosyVoice 3 Eval 结果** [Table VII]:
- Zero-shot en: Spk.sim 49.66, DNSMOS 3.90, WER 4.48%
- Zero-shot zh: Spk.sim 56.93, DNSMOS 3.71, WER 12.08%
- Seed-TTS-Eval en: Spk.sim 47.60, WER 2.05%

**关键发现**:
1. [论文原文] ZONOS2 在开源模型中 speaker similarity 最高(en clean: 78.6),超过 Fish S2 Pro (76.9) 和 VoxCPM 2 (65.2) [Table III]
2. [论文原文] ITW TTSDS2 prosody 指标 ZONOS2 是所有模型中最佳 [Fig 5b]
3. [论文原文] DS-WED 显示 ZONOS2 的 prosodic variation 显著高于所有其他模型 [Fig 6]
4. [agent 解读] 然而在 WER 维度,ZONOS2 落后于 Qwen3-TTS 和 Cartesia Sonic 3.5,尤其是中文 WER (15.62% vs Qwen3 的 2.91%)

## 局限性

1. **MoE 训练不稳定**: Expert routing 在 audio data 上比 text data 更难平衡,normalized entropy 周期性崩塌至 0.6。需要手动调整 balancing-bias 和 router 学习率,"for reasons we do not fully understand" [§IV-A, §VII-B]
2. **Quality Mode 的权衡**: 提升 intelligibility/quality 但降低 speaker similarity,两者不可兼得 [§VI]
3. **中文 WER 较高**: Clean set 中 WER 15.62% (vs Qwen3 2.91%),Quality Mode 降至 6.73% 仍高于竞品 [Table III]
4. **Attention 选择的妥协**: MHA 明显优于 GQA 但出于推理速度妥协选择了 GQA [§VII-A]
5. **Speaker embedding causal leakage**: 虽然 LDA + two-phase annealing 缓解了问题,但"the extent to which the model can be safely trained using embeddings extracted from the target utterance remains limited" [§II-C]
6. **Benchmark 自评**: ZTTS1-Eval 由 ZONOS2 团队提出,且 ZONOS2 声明未在 eval 数据上训练,但 "We cannot speak to whether any of the comparable models without public training sets are trained on data used by ZTTS1-Eval" [§V]

## 点评

**优势**:
- **工程创新显著**: 首个开源 MoE TTS 系统,从 1.6B 到 8B(900M active)的扩展路径清晰,MoE 在保持推理速度的同时扩大了模型容量
- **Speaker cloning 方案有实用价值**: LDA 降维 + two-phase annealing 的组合解决了 speaker embedding overfitting 这个实际工程问题,方案简洁且有效
- **数据规模领先**: 6.2M 小时训练数据,multi-ASR ensemble 的 transcript 质量控制方案值得借鉴
- **诚实的讨论**: 论文坦承 MoE balancing 困难、MHA vs GQA 的妥协、speaker embedding 限制,这种透明度在技术报告中值得肯定

**不足**:
- **缺少消融实验**: 没有系统性消融(如 MoE vs dense、byte vs phoneme、LDA vs 无 LDA 的定量对比),大部分设计选择只有定性描述
- **WER 竞争力不足**: 尤其中文 WER 远落后于 Qwen3-TTS,byte tokenization 在 CJK 语言上的劣势未充分讨论
- **ZTTS1-Eval 客观性存疑**: 自行提出的 benchmark 自然选择了有利于自己的评估维度(prosody, diversity),且更换了所有 scoring 模型,难以与历史结果对比
- **MoE 在 TTS 中的必要性论证不充分**: 论文展示了 MoE 可以用于 TTS,但没有与同参数量 dense model 的公平对比,无法确定 MoE 的实际增益

**定位**: ZONOS2 是 Zyphra 在开源 TTS 领域的重要探索,将 LLM 领域的 MoE 技术成功迁移到 TTS,在 voice cloning 和 prosody 维度表现突出。但在 intelligibility (尤其中文) 和 overall quality (UTMOS) 上仍落后于 Qwen3-TTS 等系统。其最大贡献可能是开源了一个 8B 参数的 TTS 模型(Apache 2.0),以及 ZTTS1-Eval benchmark 对现有评估方法论的补充。

## 可复用的 idea

1. **LDA 降维 speaker embedding**: 对任何使用预训练 speaker encoder 的系统,都可以用 LDA 去除录音条件/时长/词汇内容等 nuisance factors,延长带 speaker conditioning 的训练 horizon
2. **Two-phase speaker conditioning annealing**: Crop + mask loss → Full + no mask 的 curriculum,可迁移到任何需要从 target 提取条件信息的生成模型中
3. **Multi-ASR ensemble for data quality**: 用多个 ASR 系统的 pairwise WER 作为 transcript 质量指标,不同训练阶段调整阈值,是大规模语音数据管理的实用方案
4. **Byte tokenization for multilingual TTS**: 在足够数据规模下,byte tokenization 可完全替代 G2P phonemization,避免 G2P 的静默失败,尤其适合多语言/code-switched 场景
5. **Quality Mode token**: 用 synthetic conditioning token 在推理时切换质量/保真度 trade-off,提供用户控制维度
6. **Prosody distribution 纳入 TTS 评估体系**: 将 TTSDS2 (distribution score) + DS-WED (diversity) 与传统 WER/UTMOS/Spk.sim 组合,构建更全面的 TTS 评估方法论

## 审阅

> [!review] 审阅 (2026-06-29, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 5 个关键设计选择均有 WHY 因果链 |
> | 可信赖 | pass | 数字抽查一致,prosody 指标缺具体数值(medium) |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率约 85% |
> | 可定位 | pass | KB 背景基于 6 实体页,谱系定位准确 |
> | 不污染 | pass | frontmatter 正确,反向更新目标合理 |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/ZONOS2-review.yml`
