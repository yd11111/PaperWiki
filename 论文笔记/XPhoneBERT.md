---
type: paper
tier: deep
title: "XPhoneBERT: A Pre-trained Multilingual Model for Phoneme Representations for Text-to-Speech"
arxiv_id: "2305.19709"
source: "Sources/XPhoneBERT.pdf"
authors: [Linh The Nguyen, Thinh Pham, Dat Quoc Nguyen]
year: 2023
venue: "INTERSPEECH 2023"
tags: [TTS, phoneme, pre-training, multilingual, BERT, G2P, representation-learning]
concepts: ["[[PhonemeRepresentation]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 4 个待确认实体页: [[ProsodyModeling]]✓, [[PhonemeRepresentation]], [[VITS]], [[Text-to-SpeechPipeline]], [[TTSEvaluation]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: XPhoneBERT 属于 phoneme-level 预训练语言模型,处于 TTS pipeline 的**前端编码器**位置。PhonemeRepresentation 页记录了 TTS 前端从完整语言学特征 (SPSS) 到 phoneme only (FastSpeech) 再到 BPE text tokens (LLM-TTS) 的演进,并提到 PnG BERT / Mixed-Phoneme BERT / Phoneme-level BERT 三个单语预训练先驱。XPhoneBERT 是首个将 phoneme representation pre-training 从单语 (English) 推向多语言 (94 languages) 的工作。ProsodyModeling 页在 "Text Pre-training" 段明确提到 PnG BERT 等 BERT-style pre-training 方案用于隐式韵律建模,XPhoneBERT 延续同一路线但扩展了语言覆盖。VITS 页记录了 VITS 架构(VAE+Flow+GAN E2E),XPhoneBERT 直接替换 VITS 的 Transformer encoder 做评估。
>
> **已有认知**: phoneme embedding 在 end-to-end TTS 中是文本编码器的基础输入表示;预训练 BERT-style 模型提供更丰富的上下文化音素表征,已被 PnG BERT (2021) 等证明有效;但之前的工作局限于英语单语。
>
> **创新判断**: (1) 多语言 phoneme 预训练 (94 languages, 330M sentences) 是 novelty; (2) 开源且可直接用作 TTS encoder 的 phoneme BERT; (3) 低资源场景下效果显著 (Vietnamese 5% data: MOS +1.76)。
>
> 检索命中: [[ProsodyModeling]]✓, [[PhonemeRepresentation]][待确认], [[VITS]][待确认], [[Text-to-SpeechPipeline]][待确认], [[TTSEvaluation]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个多语言 phoneme BERT (330M 句, 94 语言),用 RoBERTa 方法预训练,替换 VITS encoder 后显著提升自然度,尤其在低资源场景
> - **路线**: Raw Text → CharsiuG2P (G2P) → segments (phoneme segmentation) → White-space tokenized phoneme sequence → XPhoneBERT (BERT-base, MLM) → Contextual phoneme embeddings → VITS decoder → Waveform
> - **指标**: EN MOS 4.00→4.14 (+0.14), VN MOS 3.74→3.89 (+0.15) [Table 2, 3]; 低资源 VN MOS 1.59→3.35 (+1.76) [Table 3]
> - **可借鉴**: (1) 预训练 phoneme encoder 可直接 drop-in 替换 TTS 的 text encoder; (2) 冻结预训练模型 25% 训练步后 unfreeze 的 fine-tuning 策略; (3) 低资源语言 TTS 中,phoneme 预训练的杠杆效应远大于高资源语言
> - **局限**: 仅在 VITS 上评估,未验证其他 TTS 架构; 实验仅覆盖英语和越南语两种语言; 未与 PnG BERT 等同类模型对比 (对方未开源); G2P 依赖 CharsiuG2P 的转换质量

## 核心问题

XPhoneBERT 解决的核心问题是: **如何在多语言 TTS 中获得高质量的上下文化音素表征?**

之前的 phoneme 预训练模型 (PnG BERT, Mixed-Phoneme BERT, Phoneme-level BERT) 均局限于英语 [§1]。同时,在资源匮乏的语言中训练 TTS 系统面临数据不足问题,预训练的多语言音素表征有望从跨语言知识迁移中获益。

作者的立论逻辑是: 既然 BERT 等文本预训练模型通过上下文化表征提升了 NLP 任务的性能,那么直接在 phoneme-level data 上预训练应比间接通过 word-level BERT 获取上下文信息更有效 [§1] [agent 解读: 这是从间接利用 BERT 的 word embedding 拼接到 phoneme representation,到直接预训练 phoneme-level BERT 的范式转换]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

XPhoneBERT 采用 BERT-base 架构: 12 层 Transformer encoder, hidden size 768, 12 attention heads, 总参数 87.6M [§2.1]。训练方法遵循 RoBERTa: 使用 dynamic masking strategy,不使用 next sentence prediction (NSP) objective [§2.1]。

### 关键设计选择

**1. 多语言预训练数据构建 (三阶段 pipeline)** [§2.2]:

| 阶段 | 任务 | 工具 |
|------|------|------|
| Phase 1 | 文本收集 + 分词 + 去重 + Text Normalization | wiki40b + wikipedia; spaCy + RDRSegmenter (越南语); NeMo TN (英/德/西/中) + Vinorm (越) [§2.2.1] |
| Phase 2 | Text-to-Phoneme 转换 | CharsiuG2P (90+ languages/locales) [§2.2.2] |
| Phase 3 | Phoneme segmentation | segments toolkit; 用 white space 分隔音素,用 meta symbol `_` (U+2581) 标记词边界 [§2.2.3] |

最终语料: **330M phoneme-level sentences, 94 languages/locales** [§2.2.4]。

**2. Tokenizer 设计**: 使用 white-space tokenizer (不是 BPE),词汇表仅 1960 phoneme types [§2.3]。[agent 解读: 这与 NLP 中常用的 subword tokenizer (30K+ vocab) 形成鲜明对比。phoneme 是语言学上的最小单元,天然适合 character-level tokenization,不需要 subword 分割。1960 的 vocab size 反映了覆盖 94 语言后 IPA phoneme inventory 的实际规模]。

**3. Locale 处理**: 对于有多个 locale 但无独立 Wikipedia 的语言 (如英语 eng-uk/eng-us),将该语言的 Wikipedia 数据等分后分别用对应 locale 的 G2P 转换 [§2.2.2]。[agent 解读: 这确保每种 locale 有充足且均衡的训练数据]。

**4. TTS 集成方式**: 直接替换 VITS 的 Transformer text encoder 为 XPhoneBERT [§3.2]。[论文原文: "We extend VITS with XPhoneBERT by replacing the VITS's Transformer encoder with XPhoneBERT"]。

### 训练策略

**预训练**: Adam optimizer, batch size 1024 (across 8 A100 40GB), peak learning rate 1e-4, max sequence length 512, 20 epochs, 约 18 天; 前 2 个 epoch 用于 learning rate warmup [§2.3]。

**TTS Fine-tuning**: 遵循原始 VITS 训练协议 (AdamW, beta1=0.8, beta2=0.99, weight decay 0.01, initial lr 2e-4, 0.999^{1/8} 每 epoch 衰减),300K steps, batch size 64 [§3.2]。关键: **XPhoneBERT 在前 25% 训练步冻结,之后 unfreeze 参与更新** [§3.2]。[agent 解读: 这种"先冻结后解冻"策略允许 VITS 的其他组件先适应预训练表征的分布,避免初始梯度破坏预训练权重]。

## 实验

| 指标 | VITS | VITS+XPB | VITS (5%) | VITS+XPB (5%) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS (EN) | 4.00+-0.08 | **4.14+-0.07** | 2.88+-0.11 | **3.22+-0.11** | LJSpeech | [Table 2] |
| MCD (EN, dB) | 7.04 | **6.63** | 7.40 | **7.15** | LJSpeech | [Table 2] |
| RMSE_F0 (EN, cent) | 377 | **348** | 407 | **383** | LJSpeech | [Table 2] |
| MOS (VN) | 3.74+-0.08 | **3.89+-0.08** | 1.59+-0.05 | **3.35+-0.10** | Vietnamese (自建) | [Table 3] |
| MCD (VN, dB) | 5.41 | **5.12** | 6.20 | **5.39** | Vietnamese (自建) | [Table 3] |
| RMSE_F0 (VN, cent) | 249 | **234** | 291 | **248** | Vietnamese (自建) | [Table 3] |
| MOS (GT, EN) | 4.39+-0.08 | — | — | — | LJSpeech | [Table 2] |
| MOS (GT, VN) | 4.26+-0.06 | — | — | — | Vietnamese (自建) | [Table 3] |

**评估协议**: 每种语言 50 条测试样本,10 名母语评估者,5 级 MOS 评分 (MOS 差异 p-value < 0.05) [§3.3]。客观指标: MCD (mel-cepstrum distance) 和 RMSE_F0 [§3.3]。

**关键发现**:
1. 全量数据下,XPhoneBERT 带来 +0.14 (EN) / +0.15 (VN) 的 MOS 提升,均统计显著 [Table 2, 3]
2. 低资源 (5% data, ~1.2h EN / ~0.9h VN) 下提升更大: EN +0.34, **VN +1.76** [Table 2, 3]。越南语的巨大提升说明预训练 phoneme representation 在低资源场景中价值极高 [§4]
3. 主观 MOS 与客观 MCD/RMSE_F0 **不总是正相关**: VN 全量 VITS 的 MOS 高于 5% XPB 的 MOS (3.74 vs 3.35),但 MCD/RMSE_F0 接近 (5.41/249 vs 5.39/248) [§4]

## 局限性

1. **评估范围狭窄**: 仅在 VITS 一个 TTS 模型上验证,未测试 FastSpeech 2、Tacotron 2 等其他架构。是否能推广到非端到端系统 (acoustic model + vocoder 分离的架构) 未知 [agent 解读]
2. **语言覆盖测试不足**: 训练覆盖 94 语言,但 TTS 评估仅覆盖英语和越南语。对低资源语言 (如非洲/东南亚语言) 的实际 TTS 效果未经验证 [agent 解读]
3. **缺乏同类对比**: 由于 PnG BERT、Mixed-Phoneme BERT、Phoneme-level BERT 均未开源 (截至 2023 年 3 月),论文未能进行直接对比 [§3.2 footnote 8]。无法判断多语言预训练 vs 单语预训练的具体增益
4. **G2P 质量依赖**: 预训练数据质量依赖 CharsiuG2P 的转换准确率,而 text normalization 仅对 5 种语言实施,其他语言的非标准文本可能引入噪声 [§2.2.1] [agent 解读]
5. **时代局限**: 论文发表于 2023 年,当时 LLM-based TTS (VALL-E, CosyVoice) 刚兴起。在 LLM-TTS 中 text encoder 通常使用 BPE tokenizer 而非 phoneme encoder,XPhoneBERT 在这类架构中的适用性不明 [agent 解读]

## 点评

XPhoneBERT 是一个工程贡献大于方法创新的工作。其核心方法——BERT-base + RoBERTa 预训练——在 NLP 领域早已成熟,创新点在于 (1) 构建了覆盖 94 语言的 phoneme-level 预训练语料,和 (2) 证明了多语言 phoneme 预训练对 TTS 的有效性。

最值得关注的是**低资源场景的杠杆效应**: 越南语 5% 数据下 MOS 从 1.59 跳到 3.35 (+1.76),几乎让不可用的系统变成可用的系统。这暗示预训练 phoneme representation 编码了跨语言共享的发音知识,能在目标语言数据极少时提供强先验。

然而从 2023 年的视角看,这项工作面临来自 LLM-TTS 范式转换的挑战: 当 TTS 系统越来越多地使用 BPE text tokens (如 CosyVoice) 甚至直接 character input 时,phoneme-level pre-training 的价值可能下降。XPhoneBERT 更适合传统 pipeline 中仍使用 phoneme 输入的场景。

## 可复用的 idea

1. **预训练 encoder 的冻结-解冻策略**: 先冻结 25% 训练步让下游模型适应预训练表征,再 unfreeze 做联合优化。这是一种通用的预训练模型集成策略,可应用于其他预训练-微调场景
2. **低资源语言 TTS 的预训练杠杆**: 当目标语言数据稀缺时,多语言预训练的 phoneme encoder 可提供强先验,显著提升合成质量。对任何低资源语言 TTS 项目有参考价值
3. **三阶段多语言 phoneme 语料构建 pipeline**: Text collection → G2P → Phoneme segmentation 的工程流程可复用于构建其他多语言语音相关的预训练数据
