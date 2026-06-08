---
type: paper
tier: deep
title: "Spirit-LM"
arxiv_id: "2402.05755"
source: "Sources/Spirit-LM.pdf"
authors: [Tu Anh Nguyen, Benjamin Muller, Bokai Yu, Marta R. Costa-jussa, Maha Elbayad, Sravya Popuri, Christophe Ropers, Paul-Ambroise Duquenne, Robin Algayres, Ruslan Mavlyutov, Itai Gat, Mary Williamson, Gabriel Synnaeve, Juan Pino, Benoit Sagot, Emmanuel Dupoux]
year: 2024
venue: "ICLR 2025"
tags: [speech-LM, multimodal, interleaving, expressive-speech, text-speech-alignment, few-shot, sentiment-preservation]
concepts: ["[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/SpeechTokenizer|Speech Tokenizer]]", "[[概念库/SemanticvsAcousticTokens|Semantic vs Acoustic Tokens]]", "[[概念库/SpeechFactorization|Speech Factorization]]", "[[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLM Integration Taxonomy]]", "[[概念库/EmotionControlinTTS|Emotion Control in TTS]]"]
models: [LLaMA-2-7B, HuBERT, HiFi-GAN, Whisper]
tasks: [speech-text-generation, ASR, TTS, speech-classification, sentiment-preservation]
datasets: [Expresso, EmoV, LibriSpeech, Multilingual-LibriSpeech, VoxPopuli, CommonVoice, Fisher, HolisticBias]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[概念库/SpeechLanguageModel|SpeechLanguageModel]], [[概念库/SpeechTokenizer|SpeechTokenizer]], [[概念库/SemanticvsAcousticTokens|SemanticvsAcousticTokens]], [[概念库/SpeechFactorization|SpeechFactorization]], [[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLMIntegrationTaxonomy]], [[概念库/EmotionControlinTTS|EmotionControlinTTS]])
>
> **Spirit-LM 在 SpeechLM 演进线中的位置**: 处于 TWIST (2024, 证明 TextLM 初始化优于冷启动) 和 Moshi (2024, 全双工 + Mimi mixed tokenizer) 之间。它延续了 TWIST 的 "continued pretraining from TextLM" 路线,但新增了 text-speech interleaving 训练和 paralinguistic tokens,是首个在一个 LM 中同时实现语义跨模态和表现力保持的系统。
>
> **Token 类型定位**: Spirit-LM 使用 HuBERT semantic tokens (论文称 "phonetic tokens"),属于 SemanticvsAcousticTokens 中 "semantic token + paralinguistic supplement" 路线,与 pGSLM (2022, F0 + duration) 一脉相承。Survey (Cui et al., 2024) 将此归类为 "Paralinguistic tokens" 子类。这意味着 Spirit-LM 优先保证语义能力,通过外挂 pitch/style tokens 补充表现力,而不像 Moshi/SpeechGPT-Gen 那样走 mixed tokenizer 路线。
>
> **集成方式定位**: 在 Speech-LLM Integration Taxonomy (Yang et al., 2025) 中,Spirit-LM 属于 "Audio-token-based Integration > Semantic Tokens" 子类 [§5.1.1],与 VoxtLM, SpeechGPT, TWIST 同类。它直接扩展 LLM 词表加入 speech tokens,不使用 modality adapter 或 ASR/TTS 级联。
>
> **表现力建模定位**: 在 EmotionControlinTTS 的演进中,Spirit-LM 的 pitch/style tokens 属于 "显式副语言 token 编码" 路线。与后续 EmoSteer-TTS (activation steering)、EmoCtrl-TTS (帧级 AV embedding) 不同,Spirit-LM 在 tokenization 层面解决表现力,让 LM 原生建模情感。
>
> 检索命中: SpeechLanguageModel, SpeechTokenizer, SemanticvsAcousticTokens, SpeechFactorization, Speech-LLMIntegrationTaxonomy, EmotionControlinTTS | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 LLaMA-2 7B 上通过 word-level speech-text interleaving 持续预训练,构建可自由混合文本和语音的基础模型;EXPRESSIVE 版本加入 pitch/style tokens 实现情感保持。
> - **路线**: TextLM 初始化 + semantic tokens (HuBERT) + paralinguistic tokens (pitch VQ-VAE + style k-means) + word-level interleaved training
> - **指标**: StoryCloze-S 61.0 (超 TWIST 55.4); Few-shot ASR WER 6.0 (LS-clean, +ASR+TTS); STSP sentiment preservation T->T 0.65 / S->S 0.54 (EXPRESSIVE, 6-shot)
> - **可借鉴**: (1) Word-level interleaving 比 sentence-level parallel (ASR+TTS only) 和 word-level transcription 都更好,是跨模态对齐的关键; (2) 多速率 token 混合 (25Hz HuBERT + 12.5Hz pitch + 1Hz style) 的 timestamp-sorted 拼接方法简洁有效; (3) STSP benchmark 是一个可复用的表现力评估框架
> - **局限**: (1) 语音质量依赖 HiFi-GAN vocoder 和 Expresso 4 speakers; (2) text 性能有 ~2pt 下降; (3) 仅英语; (4) 仅 7B 规模; (5) EXPRESSIVE 版本语义能力进一步下降 (~3pt on BLIMP-S)

## 核心问题

Spirit-LM 要解决的根本问题是: **如何让一个语言模型同时具备文本 LLM 的语义理解/生成能力和语音模型的表现力?** [§1]

具体来说,存在三个子问题:
1. **语义能力迁移**: 纯 SpeechLM (GSLM, TWIST) 的语义理解远弱于 TextLM,如何将 TextLM 预训练知识迁移到语音模态? [§1, §4.2]
2. **跨模态生成**: 如何让同一个模型在 text->speech, speech->text, speech->speech, text->text 之间自由切换? [§1]
3. **表现力保持**: HuBERT semantic tokens 丢失了副语言信息 (pitch, emotion, style),如何在保持语义能力的同时恢复表现力? [§1, §3.2]

[agent 解读] 这三个问题的优先级是递进的: 先保证语义 (Problem 1),再实现跨模态 (Problem 2),最后加入表现力 (Problem 3)。Spirit-LM 的两个版本 (BASE 和 EXPRESSIVE) 分别对应了前两个问题和第三个问题的解决。

## 方法: 它怎么 work

> [!important] 区分来源
> [论文原文] = author's explanation; [agent 解读] = my inference.

### 整体架构

Spirit-LM 的架构可以概括为: **TextLM (LLaMA-2 7B) + 扩展词表 (speech tokens) + 持续预训练 (interleaved data)** [§3, Fig 1a]。

**三种训练数据流** [§3.3, Table 2]:
1. **Text-only** (300B tokens, 33.3% sampling): LLaMA 训练数据子集 (排除代码等)
2. **Speech-only** (460K hours / 30B speech tokens, 33.3% sampling): 开源大规模语音数据
3. **Aligned Speech+Text** (110K hours / 7B speech + 1.5B text tokens, 33.3% sampling): 有 word-level alignment 的语音-文本平行数据

[agent 解读] 三种数据等比例采样是一个关键设计: 如果 speech-only 过多会丢失 text 能力,text-only 过多会学不到 speech,而 aligned 数据虽然量最小但对跨模态能力至关重要。实际上 text-only 只训练了 0.11 个 epoch (300B tokens 中只看到一小部分),而 speech+text 训练了 3.81 个 epoch [Table 2]。

**两个版本** [§3.1, §3.2]:
- **Spirit-LM BASE**: HuBERT phonetic tokens (501 units, 25Hz, deduplicated) + BPE text tokens
- **Spirit-LM EXPRESSIVE**: HuBERT + pitch tokens (64 units, 12.5Hz, VQ-VAE on F0) + style tokens (100 units, 1Hz, k-means on speechprop features)

### 关键设计选择

#### 1. Word-level Interleaving (核心创新) [§3.1, Fig 1b]

[论文原文] "Our hypothesis is that interleaving training will help the model learn an alignment between speech and text, unlocking better text to speech transfer." [§3.1]

具体做法: 在有 word-level alignment 的语音-文本平行数据中,**随机在词边界切换模态**。例如:
```
[TEXT] the cat [SPEECH][Hu3][Hu7]..[Hu200] [TEXT] the mat
```
其中 `[Hu3][Hu7]..[Hu200]` 是 "sat on" 的 HuBERT token 化。每个训练 step 的切换点随机采样,text span 长度 10-30 词,speech span 长度 5-15 词 [Appendix A]。

**为什么 interleaving 优于其他混合方式?** 消融实验 [Table 6] 提供了明确证据:

| 训练方式 | S-StoryCloze | T->S TopicStoryCloze | S->T TopicStoryCloze |
|---|---|---|---|
| Spirit-LM BASE (interleaving) | 61.0 | 72.7 | 88.6 |
| Word-level transcription | 60.1 | 57.5 | 71.9 |
| ASR+TTS only (sentence-level parallel) | 54.6 | 63.5 | 71.8 |
| No interleaving | 60.1 | 54.2 | 71.9 |
| Speech only | 54.8 | 52.2 | 49.4 |

[agent 解读] Interleaving 的关键优势在于它**保持了两种模态的自然因果序列结构** [§4.2]。Word-level transcription 打破了语音的因果流 (每个词都被切断);ASR+TTS only 在 sentence-level 切换,模态间对齐信号太稀疏。Interleaving 让模型在一个句子内多次体验 "同一个语义内容的两种模态表达",迫使模型在中间层建立跨模态对齐表征。Figure 6 (bottom) 的 layer-wise similarity 分析提供了直接证据: interleaving 训练的模型在 layer 2-20 之间 speech-text 特征相似度持续上升,而 no-interleaving 模型则没有这种趋势。

#### 2. HuBERT Token Deduplication [§3.1]

[论文原文] "HuBERT tokens are deduplicated for better modeling quality." [§3.1]

HuBERT 在 25Hz 产出 tokens,连续帧可能产生大量重复 (同一个 phoneme 持续多帧)。去重后序列大幅缩短,有效增加了 4K context window 所能覆盖的语音时长。推理时需要 duration predictor 恢复重复,由 HiFi-GAN 的 duration prediction module 完成 [§3.1]。

[agent 解读] 去重是 Spirit-LM 从 GSLM 系列继承的标准操作,但带来一个有趣的后果: 去重后的 HuBERT tokens 序列与 BPE tokens 序列在长度上更接近 (不再是 25:1 的比例失衡),这使得 interleaving 的跨模态对齐更加自然。

#### 3. Expressive Token Design (EXPRESSIVE 版本) [§3.2, Fig 1c]

**Pitch Tokens** [§3.2]:
- VQ-VAE 在 Expresso 数据集上训练,codebook size = 64,downsampling rate = 128 (12.5Hz)
- F0 提取: 训练用 pyaapt,推理用 FCPE (更快的 Transformer-based pitch estimator)
- 去重后与 HuBERT tokens 混合

**Style Tokens** [§3.2]:
- 基于 speechprop features (Duquenne et al., 2023),1 秒平均池化 (1Hz)
- **关键步骤**: 在 Expresso 上 fine-tune 去除 speaker 信息,保留 expressive style
- k-means 聚类到 100 units

**三种 token 的混合方式** [§3.2, Fig 1c]: 按 timestamp 排序拼成单一序列:
```
[SPEECH][St10][Pi0][Hu28][Hu22][Pi14][Hu15][Pi32][Hu78][Hu234][Hu468]
```

[agent 解读] 这个设计体现了 Spirit-LM 对 SpeechFactorization 的实践: 将语音分解为 phonetic content (HuBERT, 25Hz) + pitch (VQ-VAE, 12.5Hz) + style (speechprop, 1Hz) 三个独立因子。每种因子在不同时间分辨率上操作,反映了其信息变化频率的自然差异: 音素变化最快,基频次之,情感风格最慢。这种多速率因子分解与 pGSLM (Kharitonov et al., 2022) 的 multi-stream transformer 思路类似,但 Spirit-LM 将所有因子 flatten 到单一 token 序列中,避免了多流架构的复杂性。

#### 4. Vocoder: Expressive HiFi-GAN [§3.1, §3.2]

- BASE: HiFi-GAN conditioned on HuBERT tokens + 1-hot speaker embedding (Expresso 4 speakers)
- EXPRESSIVE: HiFi-GAN conditioned on HuBERT + pitch + style tokens + 1-hot speaker embedding
- 包含 duration predictor 恢复 deduplicated tokens 的原始长度

[agent 解读] Vocoder 限制在 Expresso 的 4 个 speakers 上,这是 Spirit-LM 的一个重要瓶颈 -- 生成语音的声音多样性很有限。这与后续工作 (如 CosyVoice, Moshi) 使用 flow-matching decoder 和大规模多说话人训练形成鲜明对比。

### 训练策略

**Continued Pretraining from LLaMA-2** [§3.3, Appendix A]:
- 扩展 LLaMA-2 词表加入 speech tokens + modality tokens ([TEXT], [SPEECH])
- 新 token embeddings 随机初始化
- 常数学习率 3.0e-5,序列长度 4K,batch size 4/GPU (64 A100),effective batch 1M tokens
- 200K steps (~2 weeks training)
- RoPE base frequency 从 10,000 提升到 100,000 (benefiting long context) [Appendix A]

**为什么 TextLM 初始化至关重要?** [§4.3, Fig 4]

消融实验 [Table 6, Fig 4] 显示:
- MMLU: Spirit-LM BASE 36.9 vs Randomly-initialized 25.8 (差 11.1pt)
- Intent Classification (30-shot): Spirit-LM 达 75% 仅需 25K steps,random init 在 100K steps 后仍 <20% [Fig 4c]

[论文原文] "Starting from a pretrained LLaMA 2 model is essential for few-shot in-context learning and our method successfully transfers the pretrained few-shot learning abilities of the model to the speech modality." [§4.3]

[agent 解读] 这验证了 TWIST (Hassid et al., 2023) 的核心发现: TextLM 预训练提供的不仅是语义知识,更关键的是 in-context learning 能力本身。Spirit-LM 将这个发现从 speech-only 场景扩展到了 text-speech 混合场景。

## 实验

### 语义理解 (Single-Modality) [Table 4]

| 指标 | Spirit-LM BASE | Spirit-LM EXPRESSIVE | TWIST (13B) | GSLM | Cascade (ASR+LLaMA2) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sWUGGY | 69.0 | 65.0 | 74.5 | 64.8 | 79.2 | - | Table 4 |
| sBLIMP | 58.3 | 54.2 | 59.2 | 54.2 | 71.6 | - | Table 4 |
| S-StoryCloze | 61.0 | 56.9 | 55.4 | 53.3 | 75.7 | - | Table 4 |
| S-TopicStoryCloze | 82.9 | 75.4 | 76.4 | 66.6 | 94.76 | - | Table 4 |
| MMLU (5-shot) | 36.9 | 33.3 | - | - | 46.2 | - | Table 4 |

[agent 解读] Spirit-LM BASE 在 StoryCloze (需要高级语义理解) 上大幅超越 TWIST (+5.6pt) 和 GSLM (+7.7pt),说明 interleaving 训练带来的跨模态知识迁移确实增强了语音语义理解。但在低级语言学任务 (sWUGGY, sBLIMP) 上反而弱于 TWIST,可能因为 TWIST 用 13B 模型而 Spirit-LM 用 7B。

### 跨模态理解 [Table 4]

| 指标 | Spirit-LM BASE | Spirit-LM EXPRESSIVE | Cascade | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| T->S TopicStoryCloze | 72.7 | 61.6 | 94.76 | - | Table 4 |
| S->T TopicStoryCloze | 88.6 | 73.2 | 94.76 | - | Table 4 |
| T->S StoryCloze | 59.5 | 54.6 | 75.7 | - | Table 4 |
| S->T StoryCloze | 64.6 | 58.8 | 75.7 | - | Table 4 |

[agent 解读] S->T 方向始终优于 T->S (~5pt 差距) [§4.3],说明模型在 speech->text 的隐式 ASR 能力比 text->speech 的隐式 TTS 能力更强。这与 ASR 在训练数据中自然出现 (interleaving 本身就提供了 speech-text 对) 而 TTS 需要更精确的语音生成能力一致。

### Few-Shot 任务 [Table 5]

| 指标 | Spirit-LM BASE | +ASR+TTS | EXPRESSIVE | Cascade | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| ASR (WER, 10-shot) | 21.9 | 6.0 | 37.9 | 3.7 | LS-clean | Table 5 |
| TTS (CER, 10-shot) | 45.5 | 6.7 | 52.0 | 4.0 | LS-clean | Table 5 |
| IC (Acc, 30-shot) | 71.9 | 75.8 | 66.2 | 89.6 | - | Table 5 |

[agent 解读] Spirit-LM 原生的 few-shot ASR/TTS 性能中等 (WER 21.9, CER 45.5),但加入 parallel ASR+TTS 训练数据后显著提升 (WER 6.0, CER 6.7)。这说明 interleaving 训练建立了跨模态表征基础,但专用 task data 仍然重要。值得注意的是 +ASR+TTS 对其他任务影响很小 [Table 6],说明这是一种低成本的任务增强。

### 表现力: STSP Benchmark [Table 3]

| 指标 (Sentiment Accuracy) | Spirit-LM BASE (0-shot) | Spirit-LM EXPRESSIVE (0-shot) | EXPRESSIVE (6-shot) | Cascade | 出处 |
| --- | --- | --- | --- | --- | --- |
| T->T | 0.63 | 0.65 | 0.67 | 0.65 | Table 3 |
| T->S | 0.38 | 0.36 | 0.39 | 0.36 | Table 3 |
| S->S | 0.33 | 0.54 | 0.51 | 0.33 | Table 3 |
| S->T | 0.34 | 0.36 | 0.37 | 0.33 | Table 3 |
| Avg | 0.42 | 0.48 | 0.48 | 0.42 | Table 3 |

[agent 解读] EXPRESSIVE 在 S->S 方向有戏剧性的提升 (0.33 -> 0.54),几乎翻倍,这证明 pitch+style tokens 确实让模型学会了在语音续写中保持情感风格。Cascade 在 S->S 上只有 0.33 (随机水平),因为 ASR 丢弃了副语言信息后 TTS 无法恢复,这正是 Spirit-LM 要解决的问题。但 T->S 提升有限 (0.36),说明从文本语义到语音情感的跨模态迁移仍然困难。

### Tokenizer 质量 (Expressive Resynthesis) [Table 9, Table 11]

| 模型 | Bitrate (BPS) | WER | EMO (Style Acc) | FFE (Pitch) | 出处 |
| --- | --- | --- | --- | --- | --- |
| Spirit-LM BASE tokenizer | 225 | 23.4 | 20.4 | 0.40 | Table 9 |
| Spirit-LM EXPRESSIVE tokenizer | 307 | 23.2 | 41.4 | 0.16 | Table 9 |
| HuBERT + HiFi-GAN | 550 | 23.0 | 22.7 | 0.30 | Table 9 |
| HuBERT + HiFi-GAN w/ GT Style | 550 | 21.4 | 61.6 | 0.27 | Table 9 |
| EnCodec (RVQ=8) | 4000 | 19.0 | 56.7 | 0.04 | Table 9 |

[agent 解读] EXPRESSIVE tokenizer 相比 BASE 在 style accuracy 上翻倍 (20.4 -> 41.4) 且 pitch error 大幅下降 (0.40 -> 0.16),同时内容保持几乎不变 (WER 23.4 vs 23.2)。这证明 pitch+style tokens 有效补充了 HuBERT 的表现力缺陷。但与 EnCodec RVQ=8 (EMO 56.7, FFE 0.04) 相比仍有差距,这是 semantic token + paralinguistic supplement 路线固有的 trade-off: 用远低于 EnCodec 的 bitrate (307 vs 4000 BPS) 实现了中等表现力。

## 局限性

1. **语音生成质量有限**: HiFi-GAN vocoder 限制在 Expresso 的 4 个 speakers,声音多样性极低。与 CosyVoice/Moshi 等使用 CFM + 大规模多说话人训练的系统差距明显 [§7]
2. **Text 性能下降**: 持续预训练导致 MMLU 下降 ~9pt (36.9 vs LLaMA-2 的 46.2),WUGGY/BLIMP/StoryCloze 也有 ~2pt 下降 [Table 4, §4.2]。EXPRESSIVE 版本进一步下降 (~3pt on BLIMP-S) [§4.2]
3. **仅英语**: 所有实验和评估限于英语,多语言能力未探索 [§7]
4. **仅 7B 规模**: 未尝试更大模型,scaling 效果未知 [§7]
5. **EXPRESSIVE 的语义代价**: 建模额外 pitch/style tokens 增加了序列长度,导致语义理解下降 [§4.2]。这是 "表现力 vs 语义能力" 的 fundamental trade-off
6. **Cascade baseline 仍然更强**: 在几乎所有语义理解任务上,ASR + LLaMA-2 cascade 大幅领先 [Table 4]。Spirit-LM 的优势仅在 expressivity preservation (STSP S->S) 上体现
7. **毒性风险**: S->S 方向的 MUTOX added toxicity (3.75%) 高于 cascade (2.70%) [Table 7],可能源于语音训练数据中的有毒内容

## 点评

**贡献与定位**: Spirit-LM 的核心贡献不在于 SOTA 性能 (cascade 仍更强),而在于**概念验证** -- 证明一个 LM 可以原生地混合文本和语音,保持跨模态语义一致性和情感表现力。Word-level interleaving 是一个简洁且有效的方法,消融实验清晰地证明了其优越性。

**与 pGSLM 的对比** [agent 解读]: pGSLM (Kharitonov et al., 2022) 也用 F0 + duration 补充 HuBERT,但用 multi-stream transformer 分别预测各因子。Spirit-LM 将所有因子 flatten 到单一 token 序列中用标准 decoder-only LM 建模,更简单但可能丢失了因子间的结构化关系。

**STSP Benchmark 的意义**: STSP 是首个系统评估 SpeechLM 跨模态情感保持能力的 benchmark。其设计 (speech/text prompt -> speech/text continuation -> sentiment classifier) 虽然简单,但揭示了一个重要现象: cascade 系统在 S->S sentiment preservation 上几乎随机 (0.33),因为 ASR 阶段丢弃了副语言信息。这为 end-to-end SpeechLM 提供了明确的价值论证。

**历史定位** [agent 解读]: Spirit-LM (2024.02) 代表了 SpeechLM 从 "speech-only" 向 "speech+text multimodal" 过渡的关键节点。之前的 GSLM/TWIST/pGSLM 主要在 speech-only 上工作,之后的 Moshi (2024.06) 将这个方向推向全双工实时交互。Spirit-LM 的 interleaving 思路被后续工作 (如 GLM-4-Voice 的 "interleaved data training") 广泛采用。

## 可复用的 idea

1. **Word-level interleaving 训练策略**: 在有 word-level alignment 的平行数据上随机切换模态,比 sentence-level parallel training 或 word-level transcription 更有效地建立跨模态对齐。适用于任何需要多模态对齐的 LM 训练场景 [§3.1, §4.2, Table 6]

2. **多速率 token 的 timestamp-sorted 混合**: 将不同帧率的 token (25Hz HuBERT + 12.5Hz pitch + 1Hz style) 按时间戳排序 flatten 到单一序列,让标准 decoder-only LM 建模。避免了 multi-stream 架构的复杂性,适用于任何需要融合多粒度信息的场景 [§3.2, Fig 1c]

3. **Style token 的 speaker normalization**: 对 speechprop 特征在 Expresso 上 fine-tune 去除 speaker 信息,再 k-means 聚类。这是一种轻量的 style-speaker disentanglement 方法,不需要 GRL 或 adversarial training [§3.2]

4. **STSP benchmark 设计**: 用 sentiment-rich prompts (控制情感变量) + modality-specific classifiers 评估跨模态情感保持,是一个可复用的表现力评估框架 [§5.2]

5. **RoPE base frequency 调整 (10K -> 100K)**: 对长序列 (speech tokens 很长) 的 positional encoding 优化,简单有效 [Appendix A]

> [!review] 审阅 (auto, 2026-06-08)
> **结论:** pass
> **原则:** 可复述 pass | 可信赖 pass | 可区分 pass | 可定位 pass | 不污染 pass
> **备注:**
> - 覆盖了论文全部核心实验 (Table 3-7, 9) 和消融 (Table 6, Fig 3-4, 6)
> - claim sourcing 标注覆盖率 >80% ([§], [Table], [Fig])
> - 区分了 [论文原文] 和 [agent 解读]
> - KB 定位清晰: 在 SpeechLM 演进线、token taxonomy、integration taxonomy 中均有锚点
> - 局限性完整覆盖 (7 项,含论文自述 + agent 补充)
