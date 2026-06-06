---
type: concept
title: "Phoneme Representation"
aliases: [音素表示, G2P, Grapheme-to-Phoneme, 音素, 语音学表示, IPA]
category: "representation"
tags: [TTS, text-analysis, phoneme, frontend, G2P, linguistics]
key_papers: ["[[论文笔记/MetaLearningTTS7000Languages|Meta Learning TTS 7000 Languages]]", "[[论文笔记/SpeechWeave|SpeechWeave]]", "[[论文笔记/DiaMoE-TTS|DiaMoE-TTS]]", "[[论文笔记/MAVE|MAVE]]", "[[论文笔记/ParsVoice|ParsVoice]]", "[[论文笔记/SonoEdit|SonoEdit]]", "[[论文笔记/CTC-TTS|CTC-TTS]]", "[[论文笔记/T5Gemma-TTS|T5Gemma-TTS]]", "[[论文笔记/UniSonate|UniSonate]]", "[[论文笔记/Tibetan-TTS|Tibetan-TTS]]", "[[论文笔记/X-Voice|X-Voice]]", "[[论文笔记/Dict-TTS|Dict-TTS]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Text-to-SpeechPipeline]]", "[[Attention-basedTTS]]", "[[Non-autoregressiveTTS]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Phoneme Representation 是 TTS 系统中将文本转换为发音表示的前端处理。音素(phoneme)是语言中最小的区别性语音单位,是现代 neural TTS 声学模型最常用的输入形式。

**两种输入形式**:
- **Character (字符)**: 直接使用文字符号,如 "speech" → [s, p, e, e, c, h]
- **Phoneme (音素)**: 使用发音标注,如 "speech" → [s, p, iy, ch]

**音素的优势**: 消除发音歧义 (e.g., "read" 的过去式和现在式), 减少模型学习负担, 提高发音准确率。

## TTS 前端 (Text Analysis) 完整流程

```
Raw Text → [Text Normalization] → [Word Segmentation] → [POS Tagging] → [G2P] → [Prosody Prediction] → Phoneme + Prosody
```

### 各子任务

| 任务 | 描述 | 方法 |
|------|------|------|
| Text Normalization (TN) | 非标准文本 → 口语化 | 规则 / Seq2Seq |
| Word Segmentation | 分词 (中文等) | 统计/神经网络 |
| POS Tagging | 词性标注 | CRF / BiLSTM |
| G2P Conversion | 字 → 音素 | 词典 + Seq2Seq |
| Polyphone Disambiguation | 多音字消歧 (中文) | 上下文模型 |
| Prosody Prediction | 韵律边界预测 | CRF / Self-Attention |

**前端子任务联合学习**: [[论文笔记/MultiTaskFrontEnd|Kang et al. (arXiv 2024)]] 提出用 shared trunk + task-specific heads 的 MTL 架构联合训练 TN/POS/HD 三个前端任务,证实了任务间正向迁移(HD macro acc +6.18%),并发现 ALBERT 浅层(句法信息)有利于 TN/POS、深层(上下文)有利于 HD [Table 1]。附带发布了 Llama 2 生成的平衡 homograph disambiguation 数据集

### Grapheme-to-Phoneme (G2P)

**英文 G2P**:
- 字母语言,词典覆盖大部分常用词
- OOV 词使用神经网络 G2P 模型
- 代表: CMU Pronouncing Dictionary + Seq2Seq fallback

**中文 G2P**:
- 字形已覆盖全部"字符",但多音字需消歧
- 核心问题: Polyphone disambiguation (基于上下文)
- 代表: 条件神经网络 + 多级 embedding
- 字典知识注入: [[论文笔记/Dict-TTS|Dict-TTS]] (Jiang et al., NeurIPS 2022) 提出 Semantics-to-Pronunciation Attention (S2PA),用在线字典作为结构化先验知识,通过注意力匹配输入语义与字典条目实现无监督多音字消歧。Biaobei PER-S 1.08% (接近 pypinyin 1.14%),预训练后 0.79% [Table 1, 2]。核心 insight: 将字符表示保持在语义空间(而非被 mel loss 拉向声学空间),由 Gumbel-Softmax 实现可微的离散发音选择。跨中/日/粤三语验证,但日语效果受限于 kanji 音读/训读的经验性规则

**吴语 (上海话) G2P**: [[论文笔记/ShanghainTTS|ShanghainTTS]] (Chen, 2023) 构建了上海话的 G2P 流水线: 词典查找 (125K 词条吴语词典, Chen 2022) → Yahwe 吴语拼音 → Qieyun 声调标注 → 宽式 IPA。由于词典仅含繁体字,需先经 OpenCC 转繁体。用记号简化减少歧义双字母组合 (⟨c⟩ for /tɕ/, ⟨ɟ⟩ for /dʑ/)。这是低资源方言 G2P 的典型案例 --- 依赖手工词典而非统计模型,覆盖率受词典规模限制 (仅 51K/125K 词条有拼音标注)

## 在不同 TTS 范式中的角色

### SPSS 时代
- 完整语言学特征: phoneme + duration + POS + prosody boundary + ...
- 多级标注 (word, phrase, sentence level)

### End-to-end 时代 (Tacotron/FastSpeech)
- **简化**: 仅保留 G2P (或直接用 character)
- Tacotron: character 输入 (让模型自己学 G2P)
- FastSpeech: phoneme 输入 (确保发音准确)
- 混合方案: DeepVoice 3 同时使用 character + phoneme

### LLM-TTS 时代 (VALL-E/CosyVoice)
- 通常使用 phoneme (经 G2P 处理)
- 部分系统 (如 CosyVoice) 使用 text tokenizer 的 BPE tokens
- 趋势: 随着模型规模增大,character 输入也能工作
- 发音修正新方向: SonoEdit (Singh et al., 2026) 证明可通过 knowledge editing (causal tracing + null-space constrained weight update) 在不重训练的情况下 one-shot 修正 LLM-TTS 中特定词的发音错误,绕过 G2P 前端直接编辑模型内部的 text-to-pronunciation 映射
- Subword 输入 + PM-RoPE duration control: [[论文笔记/T5Gemma-TTS|T5Gemma-TTS]] (Arata & Kurihara, 2026) 直接使用 T5Gemma SentencePiece 256K subword vocabulary(无 phoneme 转换）,配合 PM-RoPE 在 cross-attention 中注入生成进度信号实现 duration control。牺牲了 phoneme 的单调对齐特性,但避免了多语言 phonemizer 工程成本;phoneme vs subword 对 PM-RoPE 效果的影响是 open question

## Character vs Phoneme 的取舍

| 维度 | Character | Phoneme |
|------|-----------|---------|
| 预处理 | 无需 G2P | 需要 G2P 工具 |
| 发音准确性 | 依赖模型学习 | 显式保证 |
| OOV 处理 | 天然支持 | 需 G2P 推断 |
| 多语言 | 需处理不同字符集 | IPA 可统一 |
| 数据效率 | 较低 | 较高 |
| 产品部署 | 简单 | 需维护 G2P 模块 |

## 跨语言统一表示

- **IPA (International Phonetic Alphabet)**: 国际音标,可统一表示所有语言的发音
- **Byte representation**: 直接使用 UTF-8 bytes,无需任何语言学知识
- [[论文笔记/LearningToSpeakFromText|Saeki et al. (IJCAI 2023)]]: 在多语言文本上做 MLM 预训练后,byte-based TTS 在 7 种欧洲语言上全面超过 IPA baseline (de CER 3.79% vs 9.76%),且实现了未见语言的零样本 TTS (es CER 11.69%) [Table 2, 3]。证明通过预训练可以绕过 G2P,关键在于冻结预训练的 language-aware embedding
- [[论文笔记/MultilingualTurkicTTS|Yeshpanov et al. (Interspeech 2023)]]: 10 种突厥语的 IPA 统一表示早期实践。手动构建 42 个 IPA 符号的映射表,选择字母最多的语言(哈萨克语)作为源语言以最大化音素覆盖,仅用哈萨克语训练 Tacotron 2 即实现 9 种目标语言的零样本 TTS (全语言平均 MOS 3.25) [Table 2, 3]。与 X-Voice 的 eSpeak-NG 自动 G2P 不同,本文完全依赖手动映射
- **Phoneme embedding mapping**: 将不同语言的音素嵌入映射到共享空间
- [[论文笔记/X-Voice|X-Voice]] (Xu et al., 2026): 30 语言 IPA 统一表示的大规模实践。中文使用 Pinyin (高度标准化音节结构),其他语言使用 eSpeak-NG,泰/日/韩使用专用 G2P 工具 (PyThaiNLP/PyOpenJTalk/g2pK)。两个设计要点: (1) 显式保留 stress markers 区分语义 (如希腊语同形词仅靠重音位置区分含义); (2) 将 articulatory units 与 suprasegmental modifiers (长度/送气/声调) 分解但统一 embedding (引用 Zhang et al. 2021 的 NAR TTS 实验证明分离 embedding 无显著差异)。420K 小时 30 语言训练验证了此表示的可扩展性

## 关键论文

- Bisani & Ney (2008): Joint-sequence G2P model
- Deep Voice 1/2 (Arik et al., 2017): 完整神经前端 (包含 G2P)
- Char2Wav (Sotelo et al., 2017): 字符级端到端 (含隐式 G2P)
- FastSpeech (Ren et al., 2019): 确立 phoneme 作为标准输入
- LRSpeech (Li et al., 2020): 跨语言音素共享

## 相关概念

- [[Text-to-SpeechPipeline]]: phoneme representation 是 pipeline 前端的输出
- [[DurationPredictor]]: 预测每个 phoneme 的时长
- [[Attention-basedTTS]]: phoneme/character 作为 encoder 输入
- BPE Tokenizer: LLM-TTS 中 text 的替代表示方式

## Pseudo-Phoneme: 将非语言音频融入 Phoneme 驱动架构

[[论文笔记/UniSonate|UniSonate]] (Qiang et al., 2026) 提出 **Dynamic Token Injection**: 为缺乏语言学内容的 sound effects (SFX) 引入可学习 [SFX] special token 作为伪音素。token 数量按语音语料的平均 phoneme-to-duration 比率 lambda 确定: L_sfx = floor(lambda * T_target),使 SFX 的时间展开在 token 密度上与 phoneme 一致。这允许 phoneme 驱动的 MM-DiT 架构无需修改即可处理非语言音频,将 TTS+TTM 统一框架扩展至 TTA。TTS WER EN 1.47% (best), TTA FAD 4.21 (competitive) [Table 3, 5]。

## 多语言 Phoneme 预训练

[[论文笔记/XPhoneBERT|XPhoneBERT]] (Nguyen et al., INTERSPEECH 2023): 首个多语言 phoneme BERT。BERT-base 架构 (87.6M params),RoBERTa 预训练 (dynamic masking, 无 NSP),在 330M phoneme-level sentences (94 languages/locales, 来自 Wikipedia + CharsiuG2P 转换) 上训练。White-space tokenizer, 1960 phoneme types。替换 VITS Transformer encoder 后,EN MOS 4.00→4.14 (+0.14, LJSpeech), VN MOS 3.74→3.89 (+0.15) [Table 2, 3]; 低资源 VN (5% data, ~0.9h) MOS 1.59→3.35 (+1.76) [Table 3],证明多语言 phoneme 预训练在低资源场景中杠杆效应极大。与 PnG BERT (phoneme+grapheme) / Mixed-Phoneme BERT (phoneme+sup-phoneme) / Phoneme-level BERT (phoneme-only, ALBERT) 三个英语单语前驱相比,XPhoneBERT 的核心贡献是将 phoneme 预训练从单语推向多语言,并首个开源

## 演进

完整语言学特征 (SPSS; phoneme + POS + duration + prosody 标注) → 简化为 phoneme only (FastSpeech, 2019) → Character 直接输入 (Tacotron, 让模型学 G2P) → BPE text tokens (LLM-TTS, 2023+; 共享 LLM tokenizer) → **Pseudo-phoneme for non-linguistic audio** (UniSonate [SFX] tokens, 2026)
