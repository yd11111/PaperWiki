---
type: paper
tier: deep
title: "Analysing the Language of Neural Audio Codecs"
aliases: [NAC Language Analysis, NAC Token Statistical Analysis]
arxiv_id: "2509.01390"
source: "Sources/AnalysingNeuralAudioCodecs.pdf"
authors: [Joonyong Park, Shinnosuke Takamichi, David M. Chan, Shunsuke Kando, Yuki Saito, Hiroshi Saruwatari]
year: 2025
venue: "arXiv 2025 (UTokyo + Keio University + UC Berkeley)"
tags: [neural-audio-codec, speech-tokenizer, Zipf-law, Heaps-law, entropy, statistical-analysis, token-properties, n-gram, codec-evaluation]
concepts: ["[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[ResidualVectorQuantization]]", "[[CodebookCollapse]]", "[[AudioTokenizerTaxonomy]]", "[[Single-codebookvsMulti-codebook]]", "[[TokenRateandBitrateTrade-offs]]"]
models: ["[[模型库/EnCodec|EnCodec]]", "[[模型库/SoundStream|SoundStream]]"]
tasks: [token-analysis, codec-evaluation, speech-resynthesis]
datasets: [LJSpeech, CSS10-Mandarin]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ResidualVectorQuantization]]✓, [[CodebookCollapse]]✓, [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[AudioTokenizerTaxonomy]][待确认], [[Single-codebookvsMulti-codebook]][待确认] | 未命中但可能相关: [[TokenRateandBitrateTrade-offs]]

**谱系定位**: 本文属于 audio codec 分析类工作,研究对象是 NAC (Neural Audio Codec) token 序列的内在统计结构。在 tokenizer taxonomy 中,本文分析的 6 种 NAC 模型 (SpeechTokenizer, HiFi-Codec, AudioDec, DAC, EnCodec, FunCodec) 均为 acoustic tokenizer,使用 RVQ 量化 [AudioTokenizerTaxonomy]。此前 Takamichi et al. (2024) 对 SSL-based semantic tokens 做了 Zipf's law 分析,本文是对 acoustic (NAC) tokens 的对等分析,填补了 acoustic 侧的统计规律空白 [SemanticvsAcousticTokens]。

**已有认知**:
- **RVQ 层级信息结构** [ResidualVectorQuantization]: RVQ 前层编码 coarse 信息,后层编码 fine details。本文将多维 RVQ 输出通过 ID 偏移拼接为单流后分析 n-gram 统计特性,这种处理会模糊 RVQ 的层级结构。
- **Codebook 利用率** [CodebookCollapse]: 不同 NAC 的 codebook utilization 差异巨大 (DAC 99% vs SoundStream ~90%)。codebook collapse 会导致有效词汇量远小于理论值,可能直接影响本文的 Zipf/Heaps 统计量。
- **单码本 vs 多码本** [Single-codebookvsMulti-codebook, 待确认]: 本文涉及 n_d=2 到 n_d=32 的配置,大 n_d 的 codec 展现更接近自然语言的统计特性。这与 Single-codebookvsMulti-codebook 中 "大 n_d → 高比特率 → 更好重建" 的已有发现一致,但本文补充了统计学视角。

**创新判断**: 已有工作 (Takamichi 2024, Sicherman 2023) 分析了 SSL tokens 的语言学特性; 本文首次对 NAC tokens 做系统的 Zipf/Heaps/entropy 分析,并建立统计特性与下游表现 (WER/UTMOS) 的关联。核心发现 "3-gram 是 NAC token 与自然语言的甜蜜点" 是新贡献。

> [!summary] 速查
> - **一句话**: 首次系统分析 NAC token 序列的语言学统计特性,发现 3-gram 级别最接近自然语言分布,且 "更像语言的 token" 与更好的语音质量 (WER/UTMOS) 正相关
> - **路线**: 6 种 NAC 模型 (15 配置) → Codec-SUPERB 提取 token → 去重 + 维度偏移拼接 → n-gram (2/3/4/6) 统计分析 (Zipf alpha, Heaps beta/k, entropy/redundancy) → 与 resynthesis 质量 (WER/CER/UTMOS) 做相关分析
> - **指标**: 3-gram alpha ~2.0-2.3 (最接近 Zipf 理想值 2) [Fig 2]; Heaps' beta 与 WER Pearson r=0.37-0.61, bit reduction rate 与 UTMOS r=-0.55 to -0.78 [Fig 8-9]; 英文/中文趋势一致
> - **可借鉴**: (1) 3-gram Zipf/Heaps 统计量可作为 tokenizer 质量的快速 proxy metric,无需完整 ASR/TTS pipeline; (2) 维度偏移 + boundary token 的多码本拼接处理方法; (3) 语言学统计分析可迁移到任何离散 token 序列 (image/video tokens)
> - **局限**: 仅 LJSpeech + CSS10 (单说话人朗读); 相关性 r 较弱 (~0.2-0.3 for Zipf); 未与 SSL tokens 同框架对比; 因果方向不明确 (n_d 是混淆因素); 未开源分析代码

## 核心问题

NAC 模型越来越多地被用作 speech tokenizer (用于 TTS/SpeechLM),但其 token 序列是否具有类自然语言的统计规律仍然未知 [§I]。这个问题的实践意义在于: 如果 NAC tokens 像自然语言,则 NLP 中成熟的 language modeling 技术可以有效迁移; 如果不像,则需要根本不同的建模方法 [论文原文]。

具体来说,本文追问三个问题:
1. NAC tokens 是否遵循 Zipf's law 和 Heaps' law? [论文原文]
2. 在哪个 n-gram 粒度上最接近自然语言? [论文原文]
3. 这些统计特性是否与下游语音质量 (WER/UTMOS) 相关? [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文是**分析性工作**,不提出新模型。核心方法是对多种 NAC 模型的 token 序列做 NLP 式统计分析,并与下游质量做关联。

**流程**: 6 种 NAC 模型 (15 种配置) → 统一 token 提取 (Codec-SUPERB, codebook=1024, 20ms 帧) → 预处理 (去重 + 维度偏移) → n-gram 统计分析 (Zipf/Heaps/Entropy) → 与 resynthesis 质量 (WER/CER/UTMOS) 关联。

**NAC 模型配置** [Table I]:

| 代号 | 模型 | 采样率 | 维度 n_d | 比特率 (kbps) |
| --- | --- | --- | --- | --- |
| A | SpeechTokenizer | 16k | 8 | 4 |
| B1-B3 | HiFi-Codec (AcademiCodec) | 16k/24k | 4 | 2-3 |
| C | AudioDec | 24k | 8 | 6.4 |
| D | DAC | 24k | 32 | 24 |
| E1-E5 | EnCodec | 24k | 2/4/8/16/32 | 1.5-24 |
| F1-F4 | FunCodec | 16k | 32 | 16 |

### 关键设计选择

**1. Token 提取与预处理** [§III]:

为什么需要特殊处理: NAC 输出是多维的 (n_d 层 codebook,每层独立),但统计分析需要单一 token 序列。直接拼接会因各层共享 label space (0-1023) 而产生 label collision [论文原文]。

解决方案: **维度偏移 (index shift)** — 第 i 维的 token ID 加 i * 1024 偏移 (第 0 维: 0-1023, 第 1 维: 1024-2047, ...) [§III] [论文原文]。另外在每维序列首尾添加 "dimension start" / "dimension end" 边界标记,防止跨维度 n-gram 伪关联 [§III] [论文原文]。

去重 (deduplication): 去除连续重复的相同 token,防止统计偏差 [§III] [论文原文]。这是必要的,因为 NAC 的 20ms 帧率下同一音素会产生大量重复 token [agent 解读]。

**2. 三种统计度量** [§II-B]:

**Zipf's Law** [Eq 1]: f(r) = a * r^(-eta),理想 eta ≈ 1 (即 alpha = eta + 1 ≈ 2)。用 `powerlaw` 库 MLE 估计 alpha,KS distance 衡量拟合优度 [§II-B] [论文原文]。选择 MLE 而非线性回归是因为 tail 的 heavy/light 会使一般线性回归失效 [ref 36] [论文原文]。

**Heaps' Law** [Eq 2]: V(m) = K * m^beta (0 < beta < 1)。beta 越接近 1 表示词汇量接近线性增长 (高多样性); K 反映初始词汇增长速率 [§II-B] [论文原文]。

**Entropy & Redundancy** [Eq 3-5]: H = -sum(p_i * log2(p_i)); 通过 Huffman 编码计算 bit reduction rate R = (L-H)/L; R 越小表示编码越高效、冗余越低 [§II-B] [论文原文]。

**3. 为什么重点分析 3-gram**: 本文测试了 n=2,3,4,6 的 n-gram。结果显示 3-gram 在 Zipf alpha、Heaps beta/k、KS distance 上都最接近自然语言 baseline (word 1-gram) [§IV-A] [论文原文]。作者的解释是: 3-gram 在捕获局部语言学 pattern 和维持 token 多样性之间达到最优平衡; 2-gram 太短无法充分表达上下文依赖; 4-gram 引入过度稀疏和波动性 [§IV-B.1] [论文原文]。从语音学角度推测,3 个 20ms 帧 = 60ms 可能对应音素级或音节级的 acoustic motif [agent 解读]。

**4. 分析数据** [§III]: 英文 LJSpeech (10 hrs, 单说话人) + 中文 CSS10。Ground-truth baseline 为人工转录的词级 1-gram (经 morphological analysis 归一化) [§III] [论文原文]。

### 训练策略

不涉及新模型训练。所有 NAC 模型使用预训练权重,通过 Codec-SUPERB 统一接口提取 token [§III] [论文原文]。

## 实验

### A. 语言统计分析 [§IV-A]

**Zipf's Law 拟合** [Fig 2-3]:

| n-gram | dz (EN, 与 GT 距离) | dz (ZH, 与 GT 距离) | 出处 |
| --- | --- | --- | --- |
| 2-gram | 0.74 | 0.64 | [Fig 2] |
| **3-gram** | **0.36** | **0.35** | [Fig 2] |
| 4-gram | 0.56 | 0.63 | [Fig 2] |
| 6-gram | 0.96 | 1.10 | [Fig 2] |

3-gram 的 Z-score 归一化距离最小,即最接近自然语言 (GT word 1-gram) [§IV-A.1] [论文原文]。英文和中文趋势一致 [Fig 2a vs 2b] [论文原文]。

**Heaps' Law 拟合** [Fig 4-5]:

| n-gram | 代表 beta (EN) | 代表 k (EN) | GT word | 出处 |
| --- | --- | --- | --- | --- |
| 2-gram | 0.45 | 270.11 | beta=0.75, k=3.32 | [Fig 4-5] |
| **3-gram** | **0.81** | **8.11** | - | [Fig 4-5] |
| 4-gram | 0.94 | 1.91 | - | [Fig 4-5] |
| 6-gram | 0.99 | 1.07 | - | [Fig 4-5] |

3-gram 的 beta 和 k 值最接近自然语言 baseline [§IV-A.2] [论文原文]。高 k 的 token 序列显示亚线性增长 (高重用率), 低 k 的接近线性增长 (均匀引入新 token) [§IV-A.2] [论文原文]。

**Entropy & Bit Reduction Rate** [Fig 6]:

- 1-gram NAC tokens: bit reduction rate 显著低于自然语言 → token 重用有限 [论文原文]
- 2-3-gram: bit reduction rate 超过自然语言 → 含大量可压缩的重复 acoustic patterns [论文原文]
- 4-gram+: bit reduction rate 急剧下降并低于自然语言 → 组合高度稀疏 [论文原文]
- 3-4-gram 之间: 自然语言和 NAC token 的 bit reduction rate 交叉 → 此粒度下两者分布最相似 [§IV-A.3] [论文原文]

**解读**: NAC 模型优先保留 acoustic fidelity 而非 statistical redundancy,导致高阶 n-gram 不可压缩 [§IV-A.3] [论文原文]。

### B. 统计特性与下游表现的关联 [§IV-B]

下游评估: Codec-SUPERB 解码 → resynthesized speech → Whisper-medium ASR → WER (EN) / CER (ZH); UTMOS 评估自然度 [§IV-B] [论文原文]。

**Zipf alpha vs 下游** [Fig 7]:

| 关联 | Pearson r | 方向 | 出处 |
| --- | --- | --- | --- |
| alpha vs WER (EN) | 0.25 | alpha↓ → WER↓ | [Fig 7a] |
| alpha vs CER (ZH) | 0.21 | alpha↓ → CER↓ | [Fig 7b] |
| alpha vs UTMOS (EN) | -0.24 | alpha↓ → UTMOS↑ | [Fig 7c] |
| alpha vs UTMOS (ZH) | -0.23 | alpha↓ → UTMOS↑ | [Fig 7d] |

alpha 越接近 2 (更 Zipfian) → 下游越好。高 n_d 配置同时有更低 alpha 和更好表现 [Fig 7] [论文原文]。注: 仅 3-gram 显示一致关联,2-gram 和 4-gram 不一致 [§IV-B.1] [论文原文]。

**Heaps' beta vs 下游** [Fig 8]:

| 关联 | Pearson r | 方向 | 出处 |
| --- | --- | --- | --- |
| beta vs WER (EN) | -0.55 | beta↑ → WER↓ | [Fig 8a] |
| beta vs CER (ZH) | -0.47 | beta↑ → CER↓ | [Fig 8b] |
| beta vs UTMOS (EN) | 0.37 | beta↑ → UTMOS↑ | [Fig 8c] |
| beta vs UTMOS (ZH) | 0.61 | beta↑ → UTMOS↑ | [Fig 8d] |

beta 越接近 1 (更线性词汇增长) → 下游越好。跨 2-6-gram 一致,是三种指标中关联最强的 [§IV-B.2] [论文原文]。

**Bit reduction rate vs 下游** [Fig 9]:

| 关联 | Pearson r | 方向 | 出处 |
| --- | --- | --- | --- |
| BRR vs WER (EN) | 0.63 | BRR↓ → WER↓ | [Fig 9a] |
| BRR vs CER (ZH) | 0.41 | BRR↓ → CER↓ | [Fig 9b] |
| BRR vs UTMOS (EN) | -0.56 | BRR↓ → UTMOS↑ | [Fig 9c] |
| BRR vs UTMOS (ZH) | -0.78 | BRR↓ → UTMOS↑ | [Fig 9d] |

冗余度越低 → 下游越好。跨 2-6-gram 一致 [§IV-B.3] [论文原文]。r=-0.78 (ZH UTMOS vs BRR) 是全文最强关联 [Fig 9d]。

**维度 n_d 的角色** [Fig 7-9]: 高 n_d 的配置 (D: n_d=32, E5: n_d=32, F1-F4: n_d=32) 系统性地显示更低 alpha、更高 beta、更低 BRR,以及更好的 WER/UTMOS [论文原文]。这是预期中的,因为大 n_d → 更高比特率 → 更好重建质量 [agent 解读]。

## 局限性

1. **数据规模有限**: 仅 LJSpeech (10 hrs, 单说话人) + CSS10 中文; 无多说话人、多域、带噪环境分析 [§V] [论文原文]
2. **相关性非因果**: alpha/beta/BRR 与 WER/UTMOS 的关联不能证明因果关系。n_d (维度数) 是明显的混淆因素 — 大 n_d 既带来更接近自然语言的统计特性,又提供更高比特率和更好重建质量。需要控制 n_d 后再分析 [agent 解读]
3. **未与 SSL tokens 对比**: 同一框架下未对比 SSL semantic tokens (HuBERT k-means) 和 NAC acoustic tokens 的统计差异。Takamichi et al. (2024) [ref 15] 的 SSL 分析使用不同设置,无法直接比较 [agent 解读]
4. **仅朗读语音**: 单说话人朗读 (LJSpeech) 不代表真实多说话人对话场景 [§V] [论文原文]
5. **Pearson r 较弱**: Zipf alpha 的相关性 r 仅 ~0.2-0.3,统计显著性可能不足,仅 15 个数据点 [agent 解读]
6. **未讨论 codebook utilization**: 不同 NAC 的 codebook 利用率差异 (DAC 99% vs SoundStream ~90%) 可能直接影响 Zipf/Heaps 统计量,但本文未分析此因素 [agent 解读]

## 点评

本文揭示了一个优雅的 insight: **3-gram 是 NAC tokens 与自然语言之间的 "甜蜜点"**。1-gram NAC tokens 完全不像语言 (每帧独立编码声学信息,类似单字符而非词); 3-gram 开始展现 local pattern,可能对应 phoneme-level 或 syllable-level 的 acoustic motifs (3 x 20ms = 60ms 接近典型音素时长); 4-gram 以上因组合爆炸而失去规律性 [agent 解读]。

"更像语言的 token = 更好的语音" 这一发现有重要的 tokenizer 设计启示: 如果一个 codec 的 token 分布更 Zipfian、词汇增长更线性、冗余更低,那么 language model 可以更高效地建模它,生成质量更好。这为 tokenizer 质量评估提供了一个新的 **proxy metric** — 无需跑完整的 TTS/ASR pipeline,直接分析 token 统计即可初步判断 tokenizer 质量 [agent 解读]。

但必须审慎看待关联强度: Zipf 关联 r ~0.2-0.3 较弱,且 15 个数据点下统计功效有限。最强关联 (Heaps' beta r=0.61, BRR r=-0.78) 来自 Heaps 和 entropy 指标,但可能被 n_d 混淆 — 大 n_d 的 codec 既更像语言 (更大有效词汇) 又有更好重建质量 (更高比特率)。未来工作需要控制 n_d 后分析剩余关联是否仍然显著 [agent 解读]。

本文的另一个价值在于提供了 **分析离散表征的方法论框架**: Zipf/Heaps/entropy 分析 + 与下游指标关联,这套方法可直接迁移到 image tokens、video tokens 等任何离散表征的分析中 [agent 解读]。

## 可复用的 idea

1. **3-gram 统计量作为 tokenizer 质量 proxy**: Zipf alpha、Heaps' beta、bit reduction rate 可作为 speech tokenizer 设计的快速评估指标,无需完整 ASR/TTS pipeline。比如在训练过程中周期性计算这些指标追踪 tokenizer 质量变化
2. **维度偏移 + 边界标记**: 多码本 token 序列拼接为单流时,用 ID offset (dim_i * codebook_size) 避免 label collision,加 boundary token 防止跨维度伪关联,是处理 RVQ 多维输出的通用方法
3. **语言学统计法分析离散表征**: Zipf/Heaps/entropy 分析可迁移到任何离散 token 序列。特别是 n-gram 粒度扫描 (找统计特性最接近自然语言的粒度) 的方法论

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节清晰解释 WHY, 速查卡片可借鉴具体 |
> | 可信赖 | pass | 数字标注覆盖率 >90%, 指标名正确 |
> | 可区分 | pass-with-fixes | 来源标注 ~85%, 部分解读段可进一步明确 |
> | 可定位 | pass | KB 背景 6 源, 谱系定位清晰, frontmatter 完整 |
> | 不污染 | pass | 未触发反向更新 |
> 
> Issues: 4 (high: 0, medium: 2, low: 2)
> 详见 `_review/AnalysingNeuralAudioCodecs-review.yml`

---

检索命中: [[ResidualVectorQuantization]](confirmed), [[CodebookCollapse]](confirmed), [[SpeechTokenizer]](confirmed), [[SemanticvsAcousticTokens]](confirmed) | 过滤: [[AudioTokenizerTaxonomy]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review) | 未命中但可能相关: [[TokenRateandBitrateTrade-offs]](pending-review)
