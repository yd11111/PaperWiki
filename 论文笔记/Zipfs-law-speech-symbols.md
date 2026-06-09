---
type: paper
tier: deep
title: "Do Learned Speech Symbols Follow Zipf's Law?"
aliases: [Zipf Speech Symbols, Zipf's Law for Speech Tokens]
arxiv_id: "2309.09690"
source: "Sources/2309.09690.pdf"
authors: [Shinnosuke Takamichi, Hiroki Maeda, Joonyong Park, Daisuke Saito, Hiroshi Saruwatari]
year: 2023
venue: "Interspeech 2024"
tags: [speech-tokenizer, Zipf-law, discrete-token, speech-representation, GSLM, HuBERT, k-means, statistical-analysis, speech-analysis, self-supervised-learning]
concepts: ["[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[Self-SupervisedSpeechRepresentation]]", "[[SpeechLanguageModel]]"]
models: ["[[模型库/HuBERT|HuBERT]]"]
tasks: [speech-analysis, token-analysis]
datasets: [LibriSpeech, LJSpeech, JSUT, JVS, UME-ERJ]
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 3 个待确认实体页: [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[Self-SupervisedSpeechRepresentation]][待确认], [[SpeechLanguageModel]]✓, [[HuBERT]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]](confirmed), [[SemanticvsAcousticTokens]](confirmed), [[SpeechLanguageModel]](confirmed) | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[HuBERT]](pending-review) | 未命中但可能相关: 无

- **[[SpeechTokenizer]]** (confirmed): 本文分析的核心对象是 GSLM 系统中 speech2unit 模块产生的离散 speech symbols。这些 symbols 由 HuBERT + k-means 聚类产生,属于 semantic tokenizer 的范畴。SpeechTokenizer 页面已记录 NAC Token Language Analysis (Park et al., 2025) 的分析结果 --- 而本文 (Takamichi et al., 2023) 正是该后续工作的先驱研究,首次在 SSL speech tokens 上建立 Zipf's law 分析框架。
- **[[SemanticvsAcousticTokens]]** (confirmed): 本文分析的 speech symbols (HuBERT k-means tokens) 属于 semantic tokens 的经典形态。与后续 NAC Token Language Analysis 聚焦 acoustic tokens (EnCodec, DAC 等) 不同,本文从 semantic tokens 出发,率先探索离散语音表征的自然语言统计特性,建立了两个领域的分析基础。
- **[[SpeechLanguageModel]]** (confirmed): GSLM (Lakhotia et al., 2021) 是首个 SpeechLM,由 speech2unit + unit language model + unit2speech 三模块组成。本文的 speech symbols 正是 GSLM 的 speech2unit 输出。Zipf's law 的验证对 SpeechLM 意义重大: 如果 speech tokens 遵循类自然语言的统计规律,则 NLP 的 language modeling 技术 (n-gram, Transformer LM) 可有效迁移到语音领域 [agent 解读]。

## 速查

> [!summary] 速查
> - **一句话**: 首次验证 GSLM 离散 speech symbols (HuBERT k-means tokens) 的 n-gram 分布遵循 power law (非严格 Zipf's law),且该分析可识别非标准语音 (非母语) 的偏差
> - **路线**: 语音波形 → HuBERT + k-means (200 clusters, 20ms) → speech symbols → 去重 → 计算 n-gram rank-frequency 分布 → 与 Zipf's law (f_r = a * r^{-η}) 拟合 + native/non-native 对比
> - **指标**: word-level η ≈ 0.95 (日) / 0.94 (英), 接近 Zipf's law (η=1); speech symbol n-gram η < 1 (power law but not Zipf); 语言无关 (日英分布几乎一致)
> - **可借鉴**: (1) speech token n-gram 的 Zipf 分析框架可用于 tokenizer 质量评估; (2) n 值选择: n=6 (日) / n=2 (英) 对应一个字符, n=9 对应一个词; (3) 无文本分析方法可扩展到动物叫声/音乐等非语音音频
> - **局限**: 仅 200 clusters 的 k-means; 仅朗读语音; 非母语偏差的因果解释不明; 未分析不同 SSL 模型/层级的差异

## 核心问题

### WHY: 为什么要做这个工作?

Zipf's law 是自然语言处理中的基础经验法则: 词频与排名呈幂律关系 (f_r = a * r^{-η}, η ≈ 1)。自然语言符号 (词、字符) 遵循此定律已被广泛验证 [§1]。近年来深度学习发展出数据驱动的离散语音表征 (learned speech symbols),其本质是"由机器发明的编码语音内容的符号" --- 与人类发明的自然语言符号在功能上类似 [§1] [论文原文]。

核心研究问题: **对自然语言成立的 Zipf's law,是否也适用于学习到的语音符号?** [§1] [论文原文]

如果答案是肯定的,这将:
1. 为将 NLP 统计分析技术迁移到语音领域提供理论基础 [论文原文]
2. 开辟"无文本"(textless) 语音分析的新路径: 无需转录即可分析语音 [论文原文]
3. 可能扩展到非语音音频 (动物叫声、音乐) 的统计分析 [§5] [论文原文]

### WHAT: 核心贡献

1. **首次系统验证 learned speech symbols 与 Zipf's law 的关系**: speech symbol n-gram 遵循 power law,但非严格 Zipf's law (η ≠ 1),随 n 增大趋向线性 (η → 1) [§4.2.3, Fig 4] [论文原文]
2. **发现 speech symbol 分布的语言无关性**: 日语和英语的 speech symbol n-gram 分布在相同 n 值下几乎完全一致,跨语言可迁移 [§4.2.3, Fig 4] [论文原文]
3. **非文本偏差检测**: 通过 power law 分析可识别非母语语音中的非文本偏差 (发音风格差异),且发现高水平非母语者偏差反而更大的反直觉现象 [§4.3, Fig 5] [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文是分析性工作,不提出新模型。使用 GSLM 的 speech2unit 模块作为 speech symbol 提取器 [§2, Fig 1]:

```
Input waveform → Speech2unit (HuBERT encoder + k-means clustering) → Discrete symbols
                                                                        ↓
                                                     Rank-frequency 分析 + Zipf's law 拟合
```

**Speech2unit 细节** [§2]:
- SSL 模型: HuBERT (训练于 LibriSpeech) [论文原文]
- 离散化: k-means 聚类, 200 clusters (日语和英语各自训练) [§4.1] [论文原文]
- 帧率: 每 20ms 一个 symbol [§2] [论文原文]
- **去重**: 合并连续重复的相同 symbol (e.g., [3,3,3,50,200,200] → [3,50,200]) [§2] [论文原文]
- **语言敏感性**: speech2unit 特别是 k-means 聚类对输入语言高度敏感,因此模型需与编码语音的语言匹配 [§2] [论文原文]

### 关键设计选择

**1. 分析维度: word vs character n-gram vs speech symbol n-gram** [§3]

论文系统对比三种分析单元:

| 分析单元 | 含义 | Zipf 行为 |
|---------|------|-----------|
| Word | 自然语言词汇 | 严格遵循 Zipf's law (η ≈ 1) [Fig 2] |
| Character n-gram | n 个连续字符 | 语言依赖: 日语线性, 英语凸 [Fig 3] |
| Speech symbol n-gram | n 个连续 speech symbols | 语言无关, power law (η < 1) [Fig 4] |

**2. n 值与语言粒度的对应关系** [§4.1]

论文通过字符数/词数与 speech symbol 数的比率确定 n 值:
- 日语: 平均 1.6 字符/词, 5.7 symbols/字符 → **n=6 对应一个字符, n=9 对应一个词** [§4.1] [论文原文]
- 英语: 平均 5.1 字符/词, 1.9 symbols/字符 → **n=2 对应一个字符, n=9 对应一个词** [§4.1] [论文原文]

**3. 非文本偏差检测方法** [§3, §4.3]

对比母语者和非母语者的 speech symbol n-gram rank-frequency 分布:
- 使用相同文本 (消除文本内容差异)
- 非母语者按英语流利度分三组: low (< 3.0), mid (3.0-3.5), high (≥ 3.5) [§4.1] [论文原文]
- 每组随机抽取 10,000 句以均衡数据量 [§4.1] [论文原文]

### 训练策略

本文不涉及新模型训练。所有 HuBERT 模型和 k-means 均使用预训练权重/预训练配置。

**实验语料** [§4.1]:
- Zipf 验证 (§4.2): 日语 JSUT/JVS ~7,600 句 + 英语 LJSpeech ~13,000 句 [论文原文]
- 非文本偏差 (§4.3): UME-ERJ 语料库, ~20 母语者 + ~200 非母语者, 各读 300-500 句英语 [论文原文]
- 分词/形态分析: 日语用 MeCab, 英语用 NLTK [§4.1] [论文原文]
- 英语 GSLM speech2unit 使用 fairseq 实现 [§4.1] [论文原文]

## 实验

### A. Zipf's Law 验证 [§4.2]

**Word-level** [§4.2.1, Fig 2]:

| 语言 | η | 拟合 | 出处 |
| --- | --- | --- | --- |
| 日语 | 0.951 | 线性 (Zipfian) | [Fig 2 左] |
| 英语 | 0.944 | 线性 (Zipfian) | [Fig 2 右] |

两种语言的词频分布均严格遵循 Zipf's law,确认语料库质量 [论文原文]。

**Character n-gram** [§4.2.2, Fig 3]:

| 语言 | n=2 分布形态 | n=6 分布形态 | 出处 |
| --- | --- | --- | --- |
| 日语 | 线性 | 线性 | [Fig 3 左] |
| 英语 | 凸 (convex) | 接近线性 | [Fig 3 右] |

语言间差异可追溯到书写系统: 日语为表意文字 (logographic), 英语为表音文字 (phonographic) [§4.2.2] [论文原文]。

**Speech symbol n-gram** [§4.2.3, Fig 4]:

| 指标 | 日语 n=9 | 英语 n=9 | 出处 |
| --- | --- | --- | --- |
| η (n=9, word-level) | 0.453 | 0.516 | [Fig 4] |

核心发现:
1. **语言无关性**: 相同 n 值下,日语和英语的 speech symbol n-gram 分布几乎完全一致 [§4.2.3, Fig 4] [论文原文]
2. **n 与语义粒度对应**: n=6 (日) / n=2 (英) 对应字符时,分布形状与 character 1-gram 相似; n=9 对应词时,分布趋向线性 (Zipfian) [§4.2.3, Fig 4] [论文原文]
3. **η 随 n 增大趋向 1**: speech symbol n-gram 在 n 增大时分布线性化 (更 Zipfian),但 η 始终未达到 1.0 [§4.2.3] [论文原文]
4. **speech symbol n-gram 遵循 power law 而非严格 Zipf's law**: η ≠ 1, 表明 learned speech symbols 展现类 Zipfian 但更广义的统计行为 [§4.2.3] [论文原文]

**关键洞察**: speech symbols 的统计能力---它们能够"无文本"地复现字符和词的统计特征---证明了它们确实在某种程度上编码了语言学层面的信息,尽管它们是纯声学驱动学习的 [agent 解读]。

### B. 非文本偏差识别 [§4.3, Fig 5]

| 说话人类型 | n=1 偏差 | n=3 偏差 | n=5 偏差 | n=7 偏差 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Native | baseline | baseline | baseline | baseline | [Fig 5] |
| Low fluency | 与 native 接近 | 可见偏差 | 可见偏差 | 可见偏差 | [Fig 5] |
| Mid fluency | 与 native 接近 | 中等偏差 | 中等偏差 | 中等偏差 | [Fig 5] |
| High fluency | 与 native 接近 | **最大偏差** | **最大偏差** | **最大偏差** | [Fig 5] |

核心发现:
1. **非母语者倾向更多使用高频 symbols**: 分布偏移向左上方 [§4.3] [论文原文]
2. **分布随 n 增大线性化**: 与 §4.2 一致,线性化由文本内容 (相同文本) 驱动,非语言流利度 [§4.3] [论文原文]
3. **反直觉: 高流利度者偏差最大**: 违反"低流利度 = 偏差更大"的直觉 [§4.3] [论文原文]

论文未给出高流利度者偏差更大的解释,但暗示需要进一步探索 [§4.3] [论文原文]。可能的解读: 高流利度非母语者可能发展出了独特但一致的发音策略 (如过度清晰化/hyperarticulation),导致 speech symbol 使用模式与母语者系统性偏离 [agent 解读]。

## 局限性

1. **仅 200 clusters 的 k-means**: 聚类数可能不足以捕获细粒度发音差异,也可能影响 Zipf 分析的分辨率 [agent 解读]
2. **仅朗读语音**: JSUT/JVS/LJSpeech/UME-ERJ 均为朗读风格,未验证自发语音 [§5] [论文原文]
3. **未分析不同 SSL 模型/层级**: 仅用 HuBERT,未探索 wav2vec 2.0 或 WavLM 的 speech symbols 是否有不同的 Zipf 特性 [agent 解读]
4. **未做不同 k 值的消融**: 200 clusters 是唯一配置,不清楚 k 值对统计特性的影响 [agent 解读]
5. **非母语偏差缺乏因果解释**: 高流利度者偏差更大的反直觉发现未被解释 [论文原文]
6. **语言覆盖有限**: 仅日语和英语两种语言,未覆盖声调语言 (中文) 或形态复杂语言 [agent 解读]

## 点评

本文是一项具有开创性意义的实证研究,首次在 SSL 离散语音表征 (HuBERT k-means tokens) 上系统验证 Zipf's law。核心价值不在于发现 speech symbols 严格遵循 Zipf's law (实际上它们不严格遵循),而在于 **建立了从统计语言学视角分析离散语音表征的方法论框架**。

**亮点**:
1. speech symbol n-gram 分布的语言无关性是最优雅的发现 --- 尽管日语和英语的文字系统截然不同,但 HuBERT 学到的 speech symbols 呈现高度一致的统计行为,暗示 SSL 模型捕获的是底层的语音学共性而非语言特异性模式 [agent 解读]
2. "n 对应语言粒度"的对应关系 (n=2→英文字符, n=6→日文字符, n=9→词) 为理解 speech tokens 的语言学粒度提供了定量工具
3. 非文本偏差分析展示了实际应用价值: 无需文本转录即可检测非标准语音

**不足**:
1. 分析深度有限: 仅做了 rank-frequency 拟合,未引入 Heaps' law 或 entropy 分析 (这些在后续 Park et al., 2025 的 NAC Token Language Analysis 中被补充)
2. 未建立统计特性与下游任务性能的关联 (同样在后续工作中解决)
3. 实验设置较简单: 单一 SSL 模型 (HuBERT), 单一聚类配置 (k=200), 朗读语音

本文与后续 NAC Token Language Analysis (Park et al., 2025, 共享作者 Park 和 Takamichi) 构成完整的研究程序: 本文建立 SSL tokens 的 Zipf 分析框架 → 后续扩展到 acoustic tokens (NAC) 并引入更多统计工具 (Heaps' law, entropy/redundancy) + 下游关联分析。

## 可复用的 idea

1. **speech symbol n-gram 的 rank-frequency 分析框架**: 可直接应用于评估任何 speech tokenizer 的统计特性,作为不依赖下游任务的快速质量 proxy
2. **n 值与语言粒度的对应方法**: 通过字符数/词数与 symbol 数的比率确定合适的 n,可迁移到新语言/新 tokenizer
3. **非文本偏差检测**: 无需文本转录即可检测非标准语音模式 (口音/方言/语音障碍),可扩展到 L2 语音评估
4. **"无文本语音分析"范式**: 完全基于离散 speech symbols 的统计分析,可扩展到动物叫声、音乐、环境声等非语言音频

---

检索命中: [[SpeechTokenizer]](confirmed), [[SemanticvsAcousticTokens]](confirmed), [[SpeechLanguageModel]](confirmed) | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[HuBERT]](pending-review) | 未命中但可能相关: 无

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 分析框架的 WHY/HOW 解释充分,速查卡片可操作 |
> | 可信赖 | pass | 数字标注覆盖率 ≥ 90%,指标名正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率高 |
> | 可定位 | pass | KB 背景含 NACTokenLanguageAnalysis 谱系定位 |
> | 不污染 | pass | 无新建概念页,反向更新仅追加 |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/Zipfs-law-speech-symbols-review.yml`
