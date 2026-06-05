---
type: paper
tier: deep
title: "Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision"
arxiv_id: "2302.03540"
source: "Sources/SPEAR-TTS.pdf"
authors: [Eugene Kharitonov, Damien Vincent, Zalán Borsos, Raphaël Marinier, Sertan Girgin, Olivier Pietquin, Matt Sharifi, Marco Tagliasacchi, Neil Zeghidour]
year: 2023
venue: "arXiv (Google Research)"
tags: [TTS, zero-shot, low-resource, semantic-token, acoustic-token, backtranslation, pretraining, in-context-learning, speaker-prompting]
concepts: ["[[SemanticvsAcousticTokens]]", "[[LLM-basedTTS]]", "[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[ResidualVectorQuantization]]"]
models: ["[[SoundStream]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[SemanticvsAcousticTokens]], [[LLM-basedTTS]], [[SpeechLanguageModel]], [[SpeechTokenizer]], [[ResidualVectorQuantization]])
> SPEAR-TTS 是 AudioLM 的直接 TTS 扩展,在 [[LLM-basedTTS]] 中被列为 semantic token 路线的代表。[[SemanticvsAcousticTokens]] 页将 SPEAR-TTS 归入串联策略 (text → semantic → acoustic)。[[VoiceCloningTaxonomy]] [待确认] 将 SPEAR-TTS 列为 codec-based zero-shot cloning 的代表。
> 检索命中: [[SemanticvsAcousticTokens]], [[LLM-basedTTS]], [[SpeechLanguageModel]], [[SpeechTokenizer]], [[ResidualVectorQuantization]] | 过滤: [[VoiceCloningTaxonomy]](pending-review), [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 TTS 分解为 reading (text→semantic) 和 speaking (semantic→acoustic) 两阶段,仅需 15 分钟平行数据即可训练接近真人的多说话人 TTS / Decomposes TTS into reading (text→semantic) + speaking (semantic→acoustic), achieving near-human quality with only 15 min of parallel data
> - **路线**: 文本 → S1 (encoder-decoder Transformer, 预训练+回译+微调) → semantic tokens → S2 (decoder-only Transformer, 音频 prompt 控制说话人) → acoustic tokens → SoundStream decoder → 波形
> - **指标**: CER 1.92% (15min parallel, LibriSpeech test-clean) [Table 1(d)]; MOS 4.96 (接近 GT 4.92) [Table 5]; Speaker accuracy 92.4% (3s prompt) [Table 3]; Speaker sim 0.56 (接近 VALL-E 0.58, 用 240000x 少数据) [Table 4]
> - **可借鉴**: (1) 回译 (backtranslation) 利用无标注音频扩展平行数据; (2) semantic tokens 作为枢轴语言解耦 reading 和 speaking; (3) example prompting 控制说话人身份而无需 speaker label
> - **局限**: 推理需三阶段串行; 仅英语实验; 16 kHz 基础采样率 (bandwidth extension 为可选附加); 未开源

## 核心问题

SPEAR-TTS 试图回答: **如何用极少量的平行文本-语音数据 (低至 15 分钟) 训练出高质量的多说话人 TTS,同时保持声音多样性和零样本说话人控制?**

传统 TTS 的瓶颈 [§1]:
- 需要数百小时文本-语音平行数据 → 大多数语言不可用
- 生成声音的多样性受限于训练数据中的说话人数量
- VALL-E 虽实现零样本但仍需 60K 小时 ASR 转录数据 [§6.5]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SPEAR-TTS 由两个阶段构成 [§3, Fig 1]:
- **S1 ("Reading")**: Text → Semantic tokens。seq2seq 翻译任务,将文本音素转为 w2v-BERT semantic tokens [§4]
- **S2 ("Speaking")**: Semantic → Acoustic tokens。序列到序列翻译,将 semantic tokens 转换为 SoundStream acoustic tokens [§5]
- 可选 **S3**: 类似 AudioLM 的 fine acoustic token 生成 (bandwidth extension) [Appendix A]

**为什么这样分?** [论文原文]
1. Semantic tokens 是 text 和 acoustic tokens 之间的天然枢轴: 语义表征接近文本, 比直接 text→acoustic 更易学习 [§3]
2. S2 仅需无标注语音 (self-supervised tokenizer 产生两类 tokens), 可利用海量音频数据 — LibriLight 60K 小时 vs LJSpeech 15 分钟平行数据 [§3]
3. 分离后, S1 可通过预训练和回译减少对平行数据的依赖 [§4]

### 关键设计选择

**S1: 如何用 15 分钟平行数据训练可用的 TTS?**

三个技巧的组合 [§4, Fig 2]:

1. **Target-domain pretraining** [§4.1]: 在 semantic token 上的去噪任务 (类似 BART/T5) 预训练 encoder-decoder Transformer
   - 使用 LibriLight 60K 小时的 semantic tokens (无需文本)
   - 删除概率 0.6 的 token → 恢复原序列
   - 架构: T5-Large (24 层 encoder-decoder) [Appendix F]
   - [agent 解读] 预训练让模型理解 semantic token 的分布规律, 减少下游需要的监督量

2. **Backtranslation** [§4.2]: 用平行数据训练 backward model (semantic→text), 然后转写大量纯音频 → 获得合成的平行数据
   - 特别适合 TTS: text→speech 是一对多映射但 speech→text 几乎一对一 [§4.2]
   - [论文原文] "The one-to-many relationship makes the text-to-speech problem highly asymmetric — unlike text translation"

3. **Progressive finetuning** [§4.1]: 预训练 P → 冻结上层 encoder + 全部 decoder → 微调下层 encoder (4-8 层)

效果 [Table 1]:
- 从零训练 (a): 15min parallel → CER 24.7% (不可用)
- 预训练 (b): 15min → CER 2.88% (可用!)
- 回译+预训练 (d): 15min → CER **2.21%** → 3h 即达到从零训练 551h 的效果

**S2: 如何控制说话人身份?** [§5, Fig 3]

- 训练时: 从同一语音中取两段, 提取 (semantic_prompt, semantic_target, acoustic_prompt, acoustic_target), 模型学会从 acoustic prompt 保持说话人特征 [§5]
- 推理时: 以 3 秒 unseen speaker 语音的 semantic+acoustic tokens 作为 prefix → 模型续写 target 的 acoustic tokens, 保持 prompt 的声音特征 [§5, Fig 3]
- [agent 解读] 这比 AudioLM 的 continuation 更进一步: AudioLM 仅靠 acoustic prefix 隐式保持说话人, SPEAR-TTS 显式设计了 prompt 协议

**S2 的训练**: decoder-only Transformer (12 层, d=768, 12 头, FFN=2048), 在 LibriLight 60K 小时上训练 [§7.3]

**噪声控制** [§5, §7.4]: 采样 n_s=3 个候选 → 用 DNSMOS 选择最高质量 → CER 和音质同时改善 [Table 7]

### 训练策略

- S1 预训练: T5-Large, Adafactor 优化器, 1M updates, batch 256 [§7.2]
- S1 微调: LJSpeech (15 min 子集), beam search (beam=10), 微调底层 4-8 层 [§7.2]
- S2: LibriLight 60K 小时, temperature sampling T=0.75, Adafactor [§7.3]
- Acoustic tokens: SoundStream, 3 层 RVQ, codebook 1024, interleaved → 150 acoustic tokens/s (1500 bps) [§7.1]
- Semantic tokens: w2v-BERT 第 7 层, k-means K=512, 25 semantic tokens/s (225 bps) [§7.1]

## 实验

| 指标 | 本文 (SPEAR-TTS) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER (15min parallel, pretrain+BT) | **2.21** | FastSpeech2-LR 24h: 1.99 | LibriSpeech test-clean | [Table 1(d)] |
| CER (3h parallel, pretrain+BT) | **2.01** | FastSpeech2-LR 3h: 2.52 | LibriSpeech test-clean | [Table 1(d)] |
| MOS (15min, prompted) | **4.96** | GT: 4.92; FastSpeech2-LR 24h: 2.11 | LJSpeech test | [Table 5] |
| MOS (prompted, vs VALL-E) | **4.75** | VALL-E: 3.35 | VALL-E demo | [Table 6] |
| Speaker accuracy (top-1) | 92.4% | - | LibriSpeech test-clean | [Table 3] |
| Speaker accuracy (top-3) | 98.1% | - | LibriSpeech test-clean | [Table 3] |
| Speaker cosine similarity | 0.56 | VALL-E: 0.58; YourTTS: 0.34 | LibriSpeech test-clean | [Table 4] |
| Voice diversity (LJSpeech 1-spk) | 6.11 entropy | GT: 2.55 | LJSpeech | [Table 2] |
| Synth detection accuracy | 82.5% | - | LibriSpeech dev-clean | [Appendix E] |

**关键发现**:
1. **数据效率惊人** [Table 1]: 预训练+回译让 SPEAR-TTS 在 15 分钟平行数据下 (CER 2.21%) 接近 FastSpeech2-LR 使用 24 小时数据的表现 (CER 1.99%) — 数据效率提升 96x
2. **超越 VALL-E 的 MOS** [Table 6]: SPEAR-TTS (MOS 4.75) 显著高于 VALL-E (MOS 3.35), 尽管后者用了 240,000x 更多平行数据
3. **声音多样性不依赖训练说话人数** [Table 2]: 即使 S1 只在单说话人 LJSpeech 上训练, S2 产出的声音多样性 (entropy 6.11) 仍远高于 FastSpeech2-LR (0.66) — 多样性来自 S2 在大规模音频上训练
4. **零样本说话人控制** [Table 3]: 3 秒 prompt 实现 92.4% top-1 speaker accuracy, 声音稳定 (variability 仅 0.41 bits)
5. **Speaker similarity 接近 VALL-E** [Table 4]: 0.56 vs 0.58, 差距极小但 SPEAR-TTS 用了 240,000x 少的数据

## 局限性

1. **三阶段串行推理**: S1 + S2 + 可选 S3 → 延迟高, 不适合实时交互 [§12]
2. **仅英语实验**: 尽管方法论适用于低资源语言, 论文未验证 [§12]
3. **16 kHz 基础**: 需额外 bandwidth extension stage (T5-small) 实现 24 kHz [Appendix A]
4. **依赖 LibriLight 质量**: 音频质量参差不齐 (volunteers recording), 影响 S2 输出 [§12]
5. **未开源**: 模型权重和代码未公开
6. **回译依赖 ASR 质量**: 在极低资源语言中, 可能没有足够好的 backward model [agent 解读]

## 点评

SPEAR-TTS 的核心洞察是: **semantic tokens 是 text 和 speech 之间的完美枢轴语言 (pivot language)**, 将 TTS 转化为两个"翻译"任务。这个视角有三个深远影响:

1. **数据解耦**: "Reading" (S1) 需要平行数据但量少, "Speaking" (S2) 完全不需要平行数据。这打破了 TTS 对大规模平行语料的依赖, 对低资源语言有重大意义。

2. **能力解耦**: 语言能力 (发什么音) 和声学能力 (怎么发声) 完全分离。S1 训练在单说话人上也能产出多说话人语音, 因为多样性来自 S2 在大规模非标注音频上的训练。

3. **预训练和回译的跨领域迁移**: 这两个 NLP 技巧在 speech 上的成功应用说明 discrete speech tokens 真正实现了 "textless NLP" 的承诺。

SPEAR-TTS 与 AudioLM 出自同一团队 (Google Research), 可视为 AudioLM 从 audio continuation 到 TTS 的自然延伸。与同期 VALL-E 相比, SPEAR-TTS 走了截然不同的路线: VALL-E 用 60K 小时 ASR 数据暴力训练单阶段 codec LM, SPEAR-TTS 用架构分解和数据增强实现极致的监督效率。两者各有优劣, 但 SPEAR-TTS 的低资源适应性更有学术意义。

## 可复用的 idea

1. **回译 (backtranslation) 用于 TTS**: 将一对多的 TTS 问题分解为先做简单的多对一 (ASR 方向) → 生成合成平行数据 → 再训练正向 TTS。适用于任何 one-to-many 的生成任务
2. **自监督预训练 + 少量微调**: BART/T5 风格的去噪预训练在 speech token 域同样有效, 预训练 1M steps 后仅需 15 分钟微调
3. **Example prompting 替代 speaker embedding**: 用 acoustic token prefix 做 in-context speaker conditioning, 无需显式 speaker encoder 或 speaker ID
4. **Quality-based resampling**: 采样 n_s=3 候选 → DNSMOS 选最优 → 同时改善质量和准确率, 成本仅增 3x
