---
type: paper
tier: deep
title: "Improving TTS for Shanghainese: Addressing Tone Sandhi via Word Segmentation"
arxiv_id: "2307.16199"
source: "Sources/ShanghainTTS.pdf"
authors: [Yuanhao Chen]
year: 2023
venue: ""
tags: [TTS, low-resource, tone-sandhi, prosody, word-segmentation, minority-language, VITS, Shanghainese, Wu-Chinese]
concepts: ["[[ProsodyModeling]]", "[[PhonemeRepresentation]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[ProsodyModeling]]✓, [[VITS]], [[PhonemeRepresentation]], [[Text-to-SpeechPipeline]], [[VariationalAutoencoderforTTS]], [[DurationPredictor]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]]✓, [[VITS]][待确认], [[PhonemeRepresentation]][待确认], [[Text-to-SpeechPipeline]][待确认], [[VariationalAutoencoderforTTS]][待确认], [[DurationPredictor]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**VITS 定位**: VITS (Kim et al., ICML 2021) 是首个 conditional VAE + normalizing flow + adversarial training 的端到端并行 TTS 系统,单模型直接从 phoneme 生成波形。本文使用 VITS 作为 TTS backbone,利用其 stochastic duration predictor 建模同一文本在不同声调环境下的多种发音可能 [待确认]。

**Prosody Modeling 谱系**: KB 中 Prosody Modeling 页区分了显式韵律 (prosody tags, ToBI 标注) 和隐式韵律 (VAE/Flow/Reference Encoder) 两大路线。本文的贡献位于显式韵律的"prosodic annotation"分支,但与传统 ToBI 标注不同 --- 传统韵律标注主要标记静态停顿 (pause/break),本文将韵律标注扩展到动态声调现象 (tone sandhi),用分词结果作为声调变调域 (LD domain) 的代理标注。

**Phoneme Representation 相关**: KB 中 PhonemeRepresentation 页记录了 TTS 前端从 Text Normalization → Word Segmentation → G2P → Prosody Prediction 的标准流程。本文的创新恰在 Word Segmentation 步骤 --- 利用分词结果不仅服务于 G2P,还显式编码了声调变调的韵律信息 [待确认]。

## 速查

> [!summary] 速查
> - **一句话**: 通过词分割标注上海话左主导变调 (LD) 域的边界,改善 VITS 模型的声调连读变调质量
> - **路线**: 文本 → jieba 分词(Mandarin 字典 + 上海话补丁) → 上海话词典音标化(IPA) → VITS 训练 → 语音
> - **指标**: MOS 4.14 (本文) vs 4.19 (Apple VoiceOver) vs 4.83 (人类), 无显著差异 (p=0.64); 句子 5 本文 4.53 显著优于 VoiceOver 3.08 (p<<0.001) [Table 1, 3]
> - **可借鉴**: 用分词结果作为韵律域标注的代理,避免了训练专用韵律标注模型的数据需求;对其他声调语言 (粤语/闽南语等) 的低资源 TTS 有参考价值
> - **局限**: 数据集仅 2012 条/~93 分钟/单说话人; 依赖普通话预训练的 jieba 分词器,非原生上海话分词; 音节时长问题 (glottal stop coda 处理不当导致过长停顿); 仅 10 名有效评估者

## 核心问题

本文要解决的核心问题是:**上海话 TTS 中的连读变调 (tone sandhi) 质量差,特别是左主导变调 (left-dominant sandhi, LD)。** LD 是上海话韵律的核心特征 --- 多音节词的所有音节都必须在一个变调域 (LD domain) 内执行声调展开,左边音节的声调轮廓覆盖整个域 [§1]。Apple VoiceOver 等现有系统在处理多音节 LD 域时频繁出错(如将五音节词错误拆分为两个变调域) [§4.1]。

**为什么难?** 两个层面的困难:
1. **资源匮乏**: 上海话缺乏专用的韵律标注模型和大规模训练数据 [§2.1]
2. **韵律标注的传统局限**: 常规韵律标注主要用于标记静态停顿 (pause),而 LD 是动态的声调现象,传统方法不直接覆盖 [§Abstract]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统流水线 [Fig 1]:
```
文本 → jieba 分词 → 音标化 (Yahwe Wu Chinese Romanisation → IPA) → VITS 训练/推理 → 语音
```

核心思路: 词的左边界 (left edges of lexical words) 高度相关于 LD 域的边界 [§2.1, Roberts 2020]。因此,**将词分割作为声调变调域标注的代理 (proxy)**,用特殊符号标注同一词内的音节边界,使 TTS 模型能学习到哪些音节应在同一变调域内执行 LD [论文原文]。

### 关键设计选择

**1. 用普通话 jieba 分词器 + 上海话词典补丁**

上海话没有词频统计数据,无法直接训练分词器。解决方案: 使用 jieba (基于 DFA trie + HMM Viterbi) 的普通话字典,在其上叠加上海话特有词汇的权重 [§2.3]。这可行是因为书面普通话和上海话高度相似 [论文原文]。

[agent 解读] 这是一个"借用超层语言 (superstratum) 资源"的 makeshift 方案。作者自己也明确指出这一局限性,并认为长远来看应开发上海话原生的计算语言学基础设施 [§6]。

**2. 音标化 (Phonemisation) 流水线**

分词后的词通过上海话词典 (125,000+ 词条, Chen 2022) 转换为 Yahwe 吴语拼音,再经 Qieyun 标注声调号,最终转为宽式 IPA 转写 [§2.4]。为减少歧义双字母组合,使用了 ⟨c⟩ for /tɕ/ 和 ⟨ɟ⟩ for /dʑ/ 等记号技巧 [§2.4]。

需要注意的前处理步骤: 因词典仅含繁体字,分词结果先经 OpenCC 转为繁体再查词典 [§2.4]。

**3. VITS 作为 TTS backbone**

选择 VITS 的关键原因: 其 VAE + stochastic duration predictor 天然支持 one-to-many mapping --- 同一输入文本可以根据韵律环境以不同音高和节奏发音,这恰好适合建模声调变调(同一字在不同变调域中有不同声调实现) [§2.5, 论文原文]。

训练配置: 50K steps, batch size 32, 音频重采样至 16 kHz [§2.5]。

### 训练策略

- 数据集: ASR 项目 (Cosmos-Break 2023) 的 2,012 条上海话录音 + 对应文字转写,共 5,607 秒,单说话人 [§2.2]
- 音频重采样至 16 kHz [§2.2]
- 输入表示: 经分词标注的 IPA 转写(同一词内音节用特殊符号连接) [§2.3, Fig 1]

## 实验

### 实验设计

n x 3 x 5 x 4 设计: 11 位上海话母语者(有效 10 位) x 3 位说话者 x 5 个句子 x 4 个指标 [§3]。

三位说话者:
1. 本文模型
2. Apple VoiceOver (Apple Inc., 2017)
3. 作者本人 (母语者)

四个指标 (Cardoso et al. 2015 框架): Comprehensibility / Naturalness / Accuracy / Intelligibility, 1-5 MOS 量表 [§3]。

### 结果

| 指标 | 本文 (Speaker 1) | VoiceOver (Speaker 2) | 人类 (Speaker 3) | 出处 |
| --- | --- | --- | --- | --- |
| Overall MOS | 4.14 +/- 0.12 | 4.19 +/- 0.14 | 4.83 +/- 0.06 | [Table 1] |
| Accuracy | 4.02 +/- 0.36 | 4.06 +/- 0.26 | 4.82 +/- 0.11 | [Table 2] |
| Comprehensibility | 4.58 +/- 0.18 | 4.48 +/- 0.17 | 4.82 +/- 0.11 | [Table 2] |
| Intelligibility | 4.38 +/- 0.24 | 4.36 +/- 0.22 | 4.86 +/- 0.10 | [Table 2] |
| Naturalness | 3.76 +/- 0.32 | 3.66 +/- 0.28 | 4.82 +/- 0.14 | [Table 2] |
| Sentence 5 MOS | 4.53 +/- 0.24 | 3.08 +/- 0.41 | 4.88 +/- 0.11 | [Table 3] |
| Sentence 2 MOS | 3.70 +/- 0.34 | 4.70 +/- 0.18 | 4.82 +/- 0.12 | [Table 3] |

关键统计发现:
- Speaker 1 vs 2 整体无显著差异 (p=0.64) [§4.2]
- Speaker 1 和 2 都显著低于 Speaker 3 (p<<0.001) [§4.2]
- **Sentence 5**: 本文显著优于 VoiceOver (p<<0.001) --- 五音节 LD 域正确处理 [§4.2, Table 3]
- **Sentence 2**: 本文显著低于 VoiceOver (p<<0.001) --- 音节时长问题 [§4.2, Table 3]

### 声调变调分析

23 个测试 LD 域中,本文模型和人类全部正确,VoiceOver 在句子 5 出错 [§4.1]:
- 句子 5 "弗二弗三個" /[vəʔ.ɲi.vəʔ.se.ɦəʔ]_LD/ 是五音节 LD 域
- VoiceOver 错误地拆分为两个 LD 域 + 一个 RD 域,产生错误声调轮廓 [§4.1]
- 本文模型通过正确的分词 (vəʔ-ɲi=vəʔ=se gəʔ) 保持了正确的变调域边界 [§5.2]

## 局限性

1. **极小数据集**: 仅 2,012 条/~93 分钟/单说话人,限制了模型泛化能力 [§2.2]
2. **音节时长问题**: VITS 的 BLANK token 处理导致入声韵尾 (glottal stop coda /-ʔ/) 音节过长 --- /koʔ/ 时长 0.28s (占总语音 10%) vs 母语者 0.16s (6%),音节尾部出现不自然停顿 [§5.1]
3. **依赖普通话资源**: 分词依赖普通话 jieba 模型,音标化依赖有限覆盖的词典,数据集转写中存在假借字 (phonetic loan characters) 问题 [§6]
4. **分词不等于韵律标注**: 分词只是变调域标注的近似,无法处理附着语素 (clitics) 等特殊情况 [§5.2]
5. **评估规模小**: 仅 10 名有效评估者,5 个测试句,统计功效有限 [§4.2]
6. **无消融实验**: 未对比"无分词标注"的 baseline,无法量化分词标注的独立贡献

## 点评

**贡献定位**: 这是一篇以语言学动机驱动 TTS 改进的工作,核心贡献不在模型架构创新,而在于提出了一种**低成本的韵律编码策略** --- 将分词结果作为声调变调域的代理标注。这一思路对低资源声调语言有普遍参考价值。

**方法论评价**:
- (+) 语言学动机清晰: 从 LD 的形式语言学分析出发,找到了分词与变调域的关联 (Kuang & Tian 2019, Roberts 2020),是"formal linguistic accounts → computational systems"的良好示范
- (+) 方案极度轻量: 不需训练任何新模型,只需在 TTS 前端加入分词步骤
- (-) 缺乏消融: 没有"无分词标注" vs "有分词标注"的 VITS 对比,所以严格来说无法归因 --- MOS 的差异可能来自训练数据差异而非分词标注
- (-) 评估不够严格: 测试句仅 5 个,评估者仅 10 人,且无客观指标 (如 F0 RMSE 或声调错误率)

**与 KB 已有知识的关系**:
- 本文的方法在 KB 的 [[ProsodyModeling]] 演进线中属于"显式韵律标注"分支的特殊案例 --- 用分词结果作为韵律结构的代理,介于传统 prosody tags (ToBI) 和端到端隐式建模之间
- 与 [[PhonemeRepresentation]] 页记录的 TTS 前端标准流程相比,本文在 Word Segmentation 步骤赋予了额外的韵律含义,这在以往工作中较少见

**更广泛的意义**: 作者在结论中明确提出了"少数民族语言数字化不应过度依赖超层语言资源"的观点 [§6],这对低资源语言 TTS 社区是有价值的提醒。

## 可复用的 idea

1. **分词作为韵律代理标注**: 在缺乏专用韵律标注模型的情况下,利用分词结果近似标注声调变调域。可迁移到其他声调语言 (粤语、闽南语、越南语等) 的低资源 TTS 场景
2. **跨语言资源借用策略**: 用资源丰富的相近语言 (普通话) 的分词工具,叠加目标语言词汇补丁,快速构建目标语言的 NLP 工具。代价: 可能引入超层语言偏差
3. **VITS 的 stochastic duration predictor 适配声调语言**: VITS 的概率 duration 建模天然适合声调语言中"同一字在不同韵律环境有不同声调实现"的场景,无需额外机制
