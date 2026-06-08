---
type: paper
tier: deep
title: "Speak Foreign Languages with Your Own Voice: Cross-Lingual Neural Codec Language Modeling"
arxiv_id: "2303.03926"
source: "Sources/VALL-E-X.pdf"
authors: [Ziqiang Zhang, Long Zhou, Chengyi Wang, Sanyuan Chen, Yu Wu, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, Furu Wei]
year: 2023
venue: "arXiv preprint"
tags: [TTS, cross-lingual, zero-shot, codec-LM, in-context-learning, speech-to-speech-translation, multilingual, AR-NAR]
concepts: ["[[ResidualVectorQuantization]]", "[[CodecLanguageModel]]", "[[PhonemeRepresentation]]", "[[LLM-basedTTS]]", "[[SpeakerVerification]]"]
models: ["[[EnCodec]]"]
tasks: ["[[Cross-lingualVoiceCloning]]", "[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[ResidualVectorQuantization]], [[LLM-basedTTS]], [[SpeechLanguageModel]], [[EnCodec]], [[Cross-lingualVoiceCloning]], [[Zero-shotSpeechSynthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ResidualVectorQuantization]]✓, [[LLM-basedTTS]]✓, [[SpeechLanguageModel]]✓, [[EnCodec]]✓, [[Cross-lingualVoiceCloning]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[VoiceCloningTaxonomy]](pending-review), [[PhonemeRepresentation]](pending-review), [[SpeakerVerification]](pending-review) | 未命中但可能相关: 无

**谱系定位**: VALL-E X 是 [[论文笔记/VALL-E|VALL-E]] 的跨语言扩展,位于 LLM-basedTTS 页记录的 "codec LM" 主线上。在 Cross-lingualVoiceCloning 任务页中,VALL-E X 属于早期(2023 年初)的 "LLM + shared tokenizer" 路线,早于 CosyVoice/Qwen3-TTS 等后续系统。Zero-shotSpeechSynthesis 页记录 VALL-E 为该范式的开创者;VALL-E X 将其从单语(英文)扩展到中英双语的跨语言场景。

**已有认知**: EnCodec 页记录了其 8 层 RVQ + 75 Hz frame rate 架构,VALL-E X 沿用同一 codec。RVQ 页确认了残差量化的多层级特性(第 1 层 coarse,后续层 fine)。Cross-lingualVoiceCloning 任务页记录了当前 SOTA:CosyVoice 3 (WER to-en 2.98%, to-zh 5.09% on CV3-Eval) 和 X-Voice (30 语言, WER en→it 4.70)。VALL-E X 的评估在 EMIME 和 LibriSpeech 上进行,与当前 benchmark (SEED-TTS-Eval, CV3-Eval) 不同,直接数字对比需谨慎。

**创新判断**: VALL-E X 的核心创新是将 VALL-E 的 codec LM 范式推广到跨语言,证明了大规模多语言训练 + in-context learning 可以同时解决 speaker cloning 和 language transfer 两个问题,无需 paired bilingual data from the same speaker。这在 2023 年初是跨语言 TTS 的重要范式突破,虽然后续系统(CosyVoice 系列、XTTS)在规模和质量上已远超。

## 速查

> [!summary] 速查
> - **一句话**: 将 VALL-E 的 codec LM 范式扩展到跨语言场景,用 70K 小时中英双语数据训练多语言 AR+NAR codec LM,实现零样本跨语言 TTS 和语音到语音翻译,同时通过 Language ID 控制外语口音
> - **路线**: Source Speech → EnCodec Encoder → Source Acoustic Tokens; Source/Target Text → Multilingual G2P → Phoneme Sequences; [Source Phonemes, Target Phonemes, Source Tokens] → Multilingual AR Codec LM → 1st layer Target Tokens → Multilingual NAR Codec LM → Full Target Tokens → EnCodec Decoder → Target Waveform
> - **指标**: Cross-lingual EN TTS: ASV 0.36 / WER 4.07 vs YourTTS ASV 0.30 / WER 8.53 [Table 2]; SMOS 4.00 vs 3.42, CMOS +0.24 [Table 3]; S2ST: SMOS 4.12 vs 3.06 (baseline), BLEU 30.66 vs 27.49 [Table 4,5]
> - **可借鉴**: (1) Language ID embedding 加到 acoustic token embedding 上即可有效控制目标语言口音(简单但效果显著); (2) 跨语言推理时将 source phonemes + target phonemes + source acoustic tokens 三段拼接作为 AR prefix/prompt 的设计; (3) 无需同一说话人的双语平行数据,用独立的单语 ASR 语料即可训练跨语言能力
> - **局限**: 仅支持中英双语; 继承 VALL-E 的 AR 鲁棒性问题; Speaker similarity 与 ground truth 仍有明显 gap (ASV 0.37-0.48 vs tgt-src 0.58); 未开源; 评估数据集(EMIME, LibriSpeech)规模较小

## 核心问题

跨语言 TTS 的长期挑战是:让单语说话人用自己的声音说另一种语言,同时保持音色、情感和声学环境 [§1]。已有方法面临三个瓶颈:

1. **数据瓶颈** — 同一说话人的多语言语音数据极难获取 [§1]
2. **模型容量瓶颈** — 传统跨语言 TTS 通过额外的 speaker/language subnet 来解耦说话人和语言信息,但这些模型不够强大,无法在零样本场景下工作 [§1]
3. **口音问题** — 合成的目标语言语音常带有源语言的外语口音(L2 accent),因为模型难以将说话人特征与语言特征完全分离 [§1, Table 1]

VALL-E X 的核心洞察:如果 codec LM 的 in-context learning 能力足以在单语场景下做零样本 voice cloning (已被 VALL-E 证明),那么在多语言数据上训练相同架构,in-context learning 应当能同时迁移说话人特征和适应目标语言 [§1]。Language ID 作为额外的条件信号,帮助模型区分何时应该迁移声音特征、何时应该适应语言特征。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VALL-E X 沿用 VALL-E 的双阶段 codec LM 架构,将其扩展为多语言版本 [§3.2, Fig 2]:

1. **预处理**: 多语言文本 → Multilingual G2P (BigCiDian, 基于 IPA 的统一音素集) → 音素序列 S; 语音 → EnCodec Encoder → 8 层 acoustic token matrix A [§5.2]
2. **Stage 1 — Multilingual AR Codec LM** (φ_MAR): 自回归生成第 1 层 acoustic tokens
3. **Stage 2 — Multilingual NAR Codec LM** (φ_MNAR): 非自回归逐层生成第 2-8 层 acoustic tokens
4. **解码**: Full acoustic tokens → EnCodec Decoder → target waveform

### 关键设计选择

#### 1. 多语言条件自回归生成 [§3.2, §3.4]

AR 模型处理跨语言推理时的输入序列为 [S^s, S^t, A^s_{:,1}], 即:
- **Source phonemes** S^s: 源语言文本的音素序列
- **Target phonemes** S^t: 目标语言文本的音素序列
- **Source acoustic tokens** A^s_{:,1}: 源语音的第 1 层 codec codes, 作为 decoding prefix

AR 模型从 prefix 之后开始自回归生成目标语言的第 1 层 acoustic tokens A^t_{:,1},直到采样到 `<eos>` [Eq. 3]。

**为什么将 source phonemes 也作为 prompt**: [agent 解读] 虽然论文未显式解释,但 source phonemes 为模型提供了源语言的语言学线索,使模型能更好地将 source acoustic tokens 中的说话人特征与语言内容解耦——模型"知道"哪些 acoustic 特征对应的是什么内容,从而更准确地仅提取声音身份。

#### 2. NAR 模型的跨语言 prompt 设计 [§3.2, Eq. 4]

NAR 模型生成 2-8 层 target tokens 时,除了 target phonemes S^t 和已生成的前 l-1 层 target tokens 外,还以 source speech 的**完整 8 层** acoustic tokens A^s_{:,1:8} 作为 acoustic prompt [Eq. 4]。此外,NAR 还使用同一说话人的另一句话 Ã 的完整 acoustic tokens 作为额外参考 [§3.2]。

**为什么 NAR 用全 8 层 source tokens 而 AR 只用第 1 层**: [agent 解读] AR 只需预测 coarse 信息(第 1 层),而 NAR 需要从 source speech 提取 fine-grained 声学细节(音色纹理、背景噪声等),这些细节分布在 RVQ 的高层。论文 §3.2 描述了 NAR 以全 8 层 source tokens 作为输入的设计,但未显式解释这一选择的原因。

#### 3. Language ID Module [§3.3]

将 Language ID 嵌入为 dense vector,加到 AR 模型的 acoustic token embeddings 上 [§3.3]。

**为什么加 Language ID**: [论文原文] 多语言训练会增加模型对特定语言的建模难度,因为模型需要在多语言数据中区分不同语言的声学特征。特别是中文是声调语言而英文不是,风格差异大 [§3.3]。

**Language ID 的效果实验**: 移除 LID 或使用错误 LID 时,翻译质量下降(BLEU 30.66→29.04),但 speaker similarity 反而上升(ASV 0.37→0.41) [Table 6]。[论文原文] 这表明 target LID 会抑制 source 信息的迁移,即模型在有 LID 时更倾向于适应目标语言风格而非保留 source 特征 [§5.5]。Foreign accent 控制也依赖 LID:with LID accent score 4.10 vs without LID 2.98 (中→英) [Table 6]。

#### 4. 统一音素集 [§5.2]

使用 BigCiDian 提供的基于 IPA 的统一音素集,覆盖中英两种语言。[agent 解读] 这使 AR 模型的 phoneme embedding 空间在两种语言间共享,有助于模型学习跨语言的发音规律映射。

### 训练策略

- **数据**: LibriLight ~60K h 英文有声书(用 Kaldi ASR 自动标注) + WenetSpeech 10K+ h 中文多领域 ASR 数据,合计约 70K h [§5.1]
- **Tokenization**: EnCodec, L=8 层 RVQ, 每层 1024 entries, 75 Hz [§5.2]
- **架构**: AR 和 NAR 均为 12 层 Transformer decoder, attention dim 1024, FFN 4096 [§5.2]
- **训练**: 32 V100 GPUs, 800K steps, max lr 5e-4, warmup 8K steps [§5.2]
- **NAR 效率优化**: 每步随机选一层计算 loss,而非累加所有层 [§5.2]
- **Max sentence length**: 20 秒, LibriLight 重分段到平均 12 秒 [§5.2]
- **AR batch**: 120 秒/GPU; NAR batch: 66 秒/GPU [§5.2]
- **S2ST 额外模块**: 基于 SpeechUT 的 speech recognition & translation model, 6 层 Transformer encoder/decoder, 用 ASR+MT 数据联合预训练后在 ST 数据上微调 [§4.2, Appendix A]

## 实验

### Zero-Shot Cross-Lingual TTS

| 指标 | VALL-E X | Baseline (YourTTS) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ASV-Score (EN TTS w/ CN prompt) | 0.36±0.11 | 0.30±0.10 | LibriSpeech dev-clean + EMIME | [Table 2] |
| ASR-WER (EN TTS w/ CN prompt) | 4.07 | 8.53 | LibriSpeech dev-clean + EMIME | [Table 2] |
| Naturalness (EN TTS w/ CN prompt) | 3.54 | 3.36 | LibriSpeech dev-clean + EMIME | [Table 2] |
| ASV-Score (CN TTS w/ EN prompt) | 0.29±0.10 | - | EMIME + LibriSpeech | [Table 2] |
| ASR-WER (CN TTS w/ EN prompt) | 8.52 | - | EMIME + LibriSpeech | [Table 2] |
| SMOS | 4.00±0.20 | 3.42±0.19 | 50 samples | [Table 3] |
| CMOS vs baseline | +0.24 | 0.00 | 50 samples | [Table 3] |

### Zero-Shot S2ST

| 指标 | VALL-E X Trans | Baseline (Cascaded S2ST) | 方向 | 出处 |
| --- | --- | --- | --- | --- |
| ASV-Score (hyp vs src) | 0.37±0.10 | 0.28±0.10 | CN→EN | [Table 4] |
| ASV-Score (hyp vs tgt) | 0.38±0.10 | 0.29±0.11 | CN→EN | [Table 4] |
| ASR-BLEU | 30.66 | 27.49 | CN→EN | [Table 4] |
| Naturalness | 3.54 | 3.44 | CN→EN | [Table 4] |
| SMOS | 4.12±0.13 | 3.06±0.14 | CN→EN | [Table 5] |
| MOS | 3.87±0.21 | 3.81±0.19 | CN→EN | [Table 5] |
| ASV-Score (hyp vs src) | 0.48±0.11 | - | EN→CN | [Table 4] |
| ASR-BLEU | 34.45 | - | EN→CN | [Table 4] |
| SMOS | 3.94±0.15 | - | EN→CN | [Table 5] |
| MOS | 3.48±0.13 | - | EN→CN | [Table 5] |

**关键发现**:
- Speaker similarity 显著优于 YourTTS baseline (ASV 0.36 vs 0.30 for XTTS, 0.37 vs 0.28 for S2ST) [Table 2, 4]
- WER 大幅降低 (4.07 vs 8.53),证明 codec LM 比 mel-spectrogram baseline 在跨语言内容保真方面更优 [Table 2]
- Language ID 控制外语口音效果显著: with LID accent score 4.10 vs without 2.98 (CN→EN) [Table 6]
- S2ST 中 oracle text 的 ASR-BLEU 达 84-87, 表明 codec LM 本身的合成质量很高,翻译质量是主要瓶颈 [Table 4]
- Speaker similarity 距 upper bound (tgt vs src: 0.58) 仍有 gap,跨语言语音迁移仍有提升空间 [Table 4]
- 模型能在一定程度上保持 source 语音的情感 [§5.5],并能合成 code-switch 语音 [§5.5]

## 局限性

1. **仅双语** — 当前版本仅支持中英两种语言,未扩展到更多语言 [§6]
2. **继承 VALL-E 的 AR 鲁棒性问题** — AR 解码可能产生词漏/重复/错序 [agent 解读,基于 VALL-E 已知问题]
3. **Speaker similarity gap** — 跨语言 ASV score (0.37-0.48) 距 ground truth upper bound (0.58) 仍有明显差距,尤其是 EN→CN 方向 [Table 4]
4. **评估规模有限** — EMIME 仅 14 speakers, 350 test examples; 人工评估仅 50-56 samples [§5.1, §5.3, §5.4]
5. **S2ST 依赖外部翻译模块** — 需要额外训练的 SpeechUT-based speech recognition & translation model,非端到端 [§4.2]
6. **未开源** — 无官方代码和权重
7. **Language ID 的权衡** — LID 提高口音控制的同时会轻微降低 speaker similarity [Table 6],存在不可避免的 trade-off

## 点评

**历史地位**: VALL-E X 是首个将 codec LM 范式成功推广到跨语言零样本 TTS 和 S2ST 的工作。它验证了一个重要假设:大规模多语言训练 + in-context learning 足以同时解决 speaker cloning 和 language transfer,无需设计复杂的 speaker-language disentanglement 模块。这一范式被后续的 XTTS、CosyVoice 系列等系统继承和大幅改进。

**优势**:
- 设计简洁: 相比传统跨语言 TTS 需要 speaker encoder + language encoder + adversarial loss 等复杂组件,VALL-E X 仅需将多语言数据喂给 codec LM + 加一个 LID embedding
- Language ID 作为口音控制旋钮: 用/不用 LID 可以控制是 native accent 还是 foreign accent,提供了额外的可控性
- S2ST 应用: 证明 codec LM 可以直接插入 S2ST pipeline,替代传统 TTS 合成模块

**不足**:
- 实验评估在今天看来规模偏小: EMIME 仅 14 人, evaluation 仅 50 samples
- 未做消融实验证明"source phonemes 是否必要": 是否可以只用 source acoustic tokens + target phonemes 推理?
- Speaker similarity 绝对值偏低 (0.30-0.48 range),难以达到实用水平
- 与同期 YourTTS 对比, 但 YourTTS 不支持中文, 对比不够公平 [§5.2]

## 可复用的 idea

1. **Language ID embedding 控制口音**: 将 LID 加到 acoustic token embedding 上即可控制目标语言口音,简单有效。可推广到任何需要语言/风格切换的多语言生成系统
2. **跨语言推理的三段拼接策略**: [Source Phonemes; Target Phonemes; Source Acoustic Tokens] 作为 AR 的输入 prefix, 让模型同时获得 content cue (phonemes) 和 speaker cue (acoustic tokens)
3. **无需 paired bilingual data**: 用独立的单语 ASR 数据分别训练,在推理时通过拼接实现跨语言,完全规避了双语平行语音数据的收集难题
4. **NAR 用 full RVQ layers 做 acoustic prompt**: 与 AR 只用第 1 层不同, NAR 使用 source 完整 8 层信息, 确保 fine-grained 声学细节(如音色纹理)的迁移
5. **Language ID 的 similarity-quality trade-off 分析**: 实验清楚展示了 LID 对 speaker similarity 和 content accuracy 的相反影响, 为后续工作的 LID 设计提供了参考基准

> [!review] 自动审阅 (2026-06-08)
> **结论:** pass-with-fixes
> **评分:** 可复述 8 | 可信赖 8 | 可区分 8 | 可定位 7 | 不污染 9
> **问题:** 0 high, 1 medium, 2 low
> - [medium/template-compliance] frontmatter > datasets: 字段为空但论文使用 LibriLight/WenetSpeech/EMIME 等多个数据集
> - [low/traceability-gap] 局限性第 2 点和 NAR 设计选择的来源标注可更精确 (已修正 NAR 标注)
> **反向更新:** 用户要求跳过
