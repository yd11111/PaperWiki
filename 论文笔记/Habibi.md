---
type: paper
tier: deep
title: "Habibi: Laying the Open-Source Foundation of Unified-Dialectal Arabic Speech Synthesis"
arxiv_id: "2601.13802"
source: "Sources/Habibi.pdf"
authors: [Yushen Chen, Junzhe Liu, Yujie Tu, Zhikang Niu, Yuzhe Liang, Chunyu Qiang, Chen Zhang, Kai Yu, Xie Chen]
year: 2026
venue: "arXiv"
tags: [TTS, Arabic, multi-dialect, low-resource, curriculum-learning, flow-matching, zero-shot, benchmark, open-source]
concepts: ["[[ConditionalFlowMatching]]", "[[Non-autoregressiveTTS]]", "[[MelSpectrogram]]", "[[TTSEvaluation]]"]
models: ["[[论文笔记/Cross-LingualF5-TTS|Cross-Lingual F5-TTS]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]]✓, [[Zero-shotSpeechSynthesis]]✓, [[Cross-lingualVoiceCloning]]✓ | 过滤: [[Non-autoregressiveTTS]][待确认], [[MelSpectrogram]][待确认], [[TTSEvaluation]][待确认] | 未命中但可能相关: 无

**谱系定位**: Habibi 基于 F5-TTS 框架 (Chen et al., 2025),后者是一个基于 Conditional Flow Matching 的非自回归 (NAR) zero-shot TTS 系统,直接在 mel spectrogram 和原始文本字符序列上操作。KB 中 [[ConditionalFlowMatching]] 页记录了 CFM 在 TTS 中的广泛应用(CosyVoice 系列、MaskGCT、Voicebox 等),F5-TTS 是其中的代表 NAR 系统。

**已有认知对比**: 当前 KB 中的 zero-shot TTS 研究主要聚焦于英语和中文等高资源语言。[[Cross-lingualVoiceCloning]] 页记录的跨语言克隆工作(CosyVoice 3, XTTS, Cross-Lingual F5-TTS)涉及多语言扩展,但均未覆盖阿拉伯语方言多样性问题。Habibi 是 KB 中首个专门针对阿拉伯语多方言统一合成的系统,填补了低资源多方言 TTS 的空白。

**创新判断**: (1) 首个开源统一多方言阿拉伯语 TTS 系统 — KB 中无先例; (2) 首个标准化多方言阿拉伯语 TTS benchmark — 与 [[TTSEvaluation]] 页记录的评估标准化趋势一致; (3) 语言学驱动的课程学习策略(MSA→方言)— 这种两阶段课程学习在 KB 已有的 TTS 系统中未见记录。Cross-Lingual F5-TTS 是最接近的先驱,但仅处理跨语言扩展,未涉及同一语言内的多方言统一建模。

## 速查

> [!summary] 速查
> - **一句话**: 首个开源统一多方言阿拉伯语 TTS 框架,通过 ASR→TTS 数据整理 + MSA→方言课程学习,在 12+ 方言上匹敌 ElevenLabs 商业系统
> - **路线**: ASR 语料 → CPS 过滤 + 源分离去噪 → mel spectrogram + 字符序列 → F5-TTS (flow matching) → MSA SFT (Stage 1) → 方言/统一 SFT (Stage 2, 含 regional identifier tokens) → zero-shot 合成
> - **指标**: Uni.D2-I DMOS 4.24 vs ElevenLabs 4.12 (MSA), SMOS 全 7 方言超越 ElevenLabs (ALG: 4.10 vs 2.48), WER-O 6/7 方言更低, SIM 全部更高 [Table 4, 5]
> - **可借鉴**: (1) 用 CPS 过滤从 ASR 语料中筛选 TTS 可用数据,简单但有效; (2) 课程学习中先学正式语体(MSA)再学方言,数据排序比计算量更重要; (3) Regional identifier tokens 训练时使用即使推理时缺失也能提升性能
> - **局限**: UTMOS 全面低于 ElevenLabs (MSA: 2.81 vs 3.35),自然度有差距; 阿拉伯语 ASR 模型本身 WER 很高(MAR GT WER-O 54.42%),评估可靠性受限; 方言 ASR 偏弱导致部分结果可能失真

## 核心问题

1. 阿拉伯语有 30+ 方言变体,但不存在统一的开源 TTS 系统。核心障碍有三:跨方言词汇/音韵差异大、缺乏合成级别数据、缺少标准化多方言评估基准 [§1]
2. 现有阿拉伯语 TTS 要么仅覆盖 MSA (ArVoice 不足 100h, 大部分为合成数据),要么支持有限方言且质量远不及英语/中文的 zero-shot TTS [§1]
3. ASR 语料大量存在但质量差(低 SNR、性别不均、缺少音标),如何有效将 ASR 数据转化为 TTS 训练数据是关键工程问题 [§2.2]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Habibi 直接使用 F5-TTS (Chen et al., 2025) 作为 backbone,这是一个基于 Conditional Flow Matching 的 NAR zero-shot TTS 系统,接收 mel spectrogram + 原始文本字符序列作为输入 [§3.1]。选择 F5-TTS 的原因是: (1) 直接操作 mel spectrogram 和字符序列,无需额外的 audio/text encoder 模块,避免了级联错误; (2) 已在中英文上验证有效,具有多语言扩展潜力 [论文原文, §3.1]。

### 关键设计选择

**1. 多方言数据整理 pipeline** [§2.2]

从 MASC、SADA、MGB-2/3/5、FLEURS、Omnilingual ASR Corpus 等 ASR 数据集中整理出 ~1857h 训练数据,覆盖 12 个区域方言标识。整理流程包括:

- **CPS (Characters Per Second) 过滤**: CPS 过低通常意味着文本标签缺失或音频有长静音,过高则意味着自动字幕错误。阈值按数据集单独设定,越嘈杂越激进 [论文原文, §2.2]。[agent 解读] 这是一个极简但有效的启发式过滤,因为大多数 ASR 语料的噪声模式可被 CPS 异常值捕获
- **源分离去噪**: 对低 SNR 数据集使用 Band-Split RNN [28] 去噪,丢弃去噪后变为静音的样本 [§2.2]
- **短段合并**: 将 MASC 中短于 6 秒的相邻片段合并为最长 30 秒的片段,使时长分布从集中于 10 秒以下向更适合 TTS 的长度偏移 [§2.2]
- **频道特定过滤**: MASC (YouTube 爬取) 中的噪声模式高度频道相关(如重复的片头/片尾音乐),因此使用基于文本模式的规则进行频道级过滤 [论文原文, §2.2]

**2. 语言学驱动的课程学习** [§2.4]

两阶段训练策略:

- **Stage 1: MSA-only SFT** — 从 F5-TTS 预训练模型(~95K h 中英数据)出发,仅在 MSA 数据上微调。理由: MSA 作为书面标准语,具有最规范的音韵和语法模式,是从中英文到阿拉伯语迁移语音合成能力的理想桥梁 [论文原文, §2.4]。过渡时机由 UTMOS (自然度) 收敛确定 — 因为继续训练虽然 WER 还在降,但生成音频会偏向 SFT 数据分布(ASR 语料, SNR 较低),导致说话人相似度和自然度下降 [论文原文, §2.4]
- **Stage 2: 方言微调** — 两条并行路线: (a) 按方言单独训练的专用模型; (b) 在所有方言数据上联合训练的统一模型

[agent 解读] 这种"先学正式→后学口语"的策略本质上利用了阿拉伯语的 diglossia 特性: MSA 是所有方言的"上层语言",先掌握 MSA 的音韵基础有助于后续方言微调。这与常见的多语言 TTS 中先高资源后低资源的策略一致,但在同一语言内部的方言谱系上应用是新颖的。

**3. 方言感知 SFT (Dialect-Aware SFT)** [§2.5]

在文本序列前添加区域标识符特殊 token: `<ID_i> <start> c1 c2 ... cn <end>`,其中 `<ID_i>` 对应 12 个区域方言之一 [§2.5]。[论文原文] 训练时加入标识符,即使推理时缺失标识符也能提升性能。作者假设两个因素: (1) 课程学习已赋予模型强 in-context learning 能力和对多样输入模式的鲁棒性; (2) 标识符帮助模型内化方言相关的分布模式 [§2.5]。

**4. 标准化 benchmark** [§2.3]

首个多方言阿拉伯语 zero-shot TTS benchmark,覆盖 7 个方言子集 (MSA, SAU, UAE, ALG, IRQ, EGY, MAR),共 11,000+ 句,筛选条件: 3-12 秒时长、纯阿拉伯文字幕、同说话人至少两条(用于 zero-shot 参考+目标对) [§2.3]。

### 训练策略

- Backbone: F5-TTS v1 base 默认配置,所有模型训练到 200K updates [§3.1]
- 硬件: 8x NVIDIA H100 SXM GPU,每次训练约 2 天 [§3.1]
- 评估使用双 ASR: WER-O (Omnilingual-ASR-LLM-7B) + WER-S (方言专用 ASR),因为多语言 ASR 有跨方言识别偏差,而专用 ASR 泛化和噪声鲁棒性差 [论文原文, §3.2]
- 去噪混合采样: 原始与去噪音频以 0.618 概率混合,仅应用于有去噪版本的数据源 [§3.7]
- 主观评估: DMOS (方言发音准确度) + SMOS (说话人相似度) + NMOS (自然度),每个方言-维度对 10-20 名母语评分者,10 轮随机采样 [§3.2]

## 实验

| 指标 | 本文 (Uni.D2-I) | Baseline (ElevenLabs 11Labs-3a) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| DMOS MSA | 4.24 | 4.12 | Habibi Benchmark | [Table 4] |
| DMOS IRQ | 4.21 | 3.97 | Habibi Benchmark | [Table 4] |
| DMOS ALG | 3.90 | 3.79 | Habibi Benchmark | [Table 4] |
| SMOS ALG | 4.10 | 2.48 | Habibi Benchmark | [Table 4] |
| SMOS MSA | 4.19 | 3.53 | Habibi Benchmark | [Table 4] |
| SMOS UAE | 3.92 | 3.23 | Habibi Benchmark | [Table 4] |
| WER-O MSA | 7.83 | 6.77 | Habibi Benchmark | [Table 5] |
| WER-O ALG | 14.96 | 17.60 | Habibi Benchmark | [Table 5] |
| WER-O IRQ | 30.13 | 35.79 | Habibi Benchmark | [Table 5] |
| WER-S UAE | 4.44 | 6.38 | Habibi Benchmark | [Table 5] |
| SIM MSA | 0.809 | 0.567 | Habibi Benchmark | [Table 5] |
| SIM ALG | 0.731 | 0.306 | Habibi Benchmark | [Table 5] |
| UTMOS MSA | 2.81 | 3.35 | Habibi Benchmark | [Table 5] |
| UTMOS EGY | 2.92 | 2.99 | Habibi Benchmark | [Table 5] |

**关键消融实验:**

1. **课程学习有效性** [Table 8]: 从零训练无法收敛; 直接微调到方言数据不如先 MSA 再方言; 即使给直接微调加倍 gradient updates 仍不如课程学习方案 — 数据排序比计算量更重要 [论文原文, §3.5]

2. **In-context learning** [Table 9]: 去掉参考音频上下文后 WER 全面恶化 (UAE WER-S: 4.88→10.68, IRQ WER-O: 17.12→20.15),确认模型在推理时主动利用参考语音-文本对来指导方言合成 [§3.6]

3. **数据质量 vs 数量** [Table 12]: EGY 上仅用干净 D1 训练优于添加 MASC 噪声子集; MAR 上引入多样时长分布数据有帮助; D1→D2 扩展在平均 WER-O (19.28→19.10) 和 SIM 上有增益 [§3.7]

4. **Regional identifiers** [Table 11]: 训练时加入标识符普遍提升 WER,对 SIM/UTMOS 影响很小; 推理时三种模板(纯文本/未知方言包裹/正确方言包裹)表现稳健,训练-推理一致性最优 [§3.8]

5. **统一 vs 专用模型** [Table 7]: 统一模型 SIM 全部超越或匹配专用模型 (MAR: 0.705 vs 0.607); WER 上专用模型在 5/7 方言领先,但统一模型在 MSA 和 MAR 更强; UTMOS 基本持平 [§3.4]

## 局限性

1. **自然度差距**: UTMOS 全面低于 ElevenLabs,且 NMOS 在多数方言上落后。ElevenLabs 可能采用了牺牲说话人保真度换取感知质量的策略 (SIM 远低于 Habibi),但其绝对自然度更高 [Table 4, 5]
2. **评估可靠性**: 阿拉伯语方言 ASR 本身 WER 极高 (MAR GT WER-O 54.42%, ALG GT WER-O 41.19%),WER 作为评估指标的可靠性存疑 [Table 6]。论文通过双 ASR + 主观评估缓解,但问题本质未解决
3. **数据局限**: 训练数据主要来自 ASR 语料,SNR 较低; 部分方言数据量很少 (LEV 仅 8.3h, SDN 仅 4.2h, LBY 仅 14.8h),这些方言未包含在 benchmark 中 [Table 1, 2]
4. **MSA 参考音频偏差**: 与 ElevenLabs 比较时,MSA/EGY/MAR 的参考音频来自 ElevenLabs PVC 语音库(ElevenLabs 在该语音上有 30+ 分钟微调数据),对 Habibi 的 zero-shot 方案不利 [§3.3]
5. **无 diacritization 的代价**: 虽然免去了文本音标化的要求,但可能在发音歧义较多的文本上存在隐含错误,论文未专门讨论这一 trade-off

## 点评

Habibi 的核心价值在于**方法论层面的洞察**而非单纯的阿拉伯语 TTS:

1. **"数据排序 > 计算量"的课程学习证据**: Table 8 的消融清楚表明,即使加倍方言训练步数也不如先经过 MSA 阶段。这对所有低资源语言/方言扩展工作都有启示 — 找到目标语言谱系中的"桥梁语体"可能比增加数据量更有效

2. **ASR→TTS 数据转化的工程范本**: 用 CPS 过滤 + 源分离 + 短段合并的简单 pipeline 将 ASR 语料改造为 TTS 训练数据,效果可观。这种思路对任何缺乏专用 TTS 数据但有 ASR 语料的语言都适用

3. **统一 vs 专用的务实结论**: 统一模型在 SIM 上全面超越专用模型,在 WER 上略逊但差距不大。对于资源有限的场景,单一统一模型是更实际的选择

4. **评估设计值得学习**: 双 ASR (多语言+专用) 互相校验的思路,以及包含 10-20 名母语评分者的主观评估,在多方言 TTS 评估中是必要的实践 — 单一 ASR 的跨方言偏差可能导致误导性结论

5. **局限**: 论文未探讨 F5-TTS 之外的 backbone (如 AR-based 或 hybrid 系统)是否更适合阿拉伯语,无法判断当前方案是否已接近最优架构。此外,UTMOS 差距较大,是否存在系统性的音频质量瓶颈(如 ASR 语料 SNR 上限)值得进一步分析

## 可复用的 idea

1. **CPS 过滤法**: 简单的字符/秒过滤可高效清洗 ASR 语料用于 TTS 训练,上下界按数据集噪声程度调整 [§2.2]
2. **课程学习中的"桥梁语体"策略**: 在同一语言的方言扩展中,先用正式/标准变体 (MSA) 建立音韵基础,再迁移到方言。可推广到其他 diglossia 语言 (如普通话→方言)
3. **训练时加 regional identifier、推理时可选**: 特殊 token 帮助训练时分离方言分布,但模型足够鲁棒以在缺失 token 时仍保持性能 [§3.8]
4. **去噪混合采样 (概率 0.618)**: 在原始和去噪音频间混合训练,兼顾数据多样性和质量 [§3.7]
5. **双 ASR 评估**: 用多语言 ASR + 方言专用 ASR 互相校验,减少单一 ASR 的系统性偏差 [§3.2]

## 审阅

> [!review] 审阅 (2026-06-05, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 4 个设计选择均有 WHY 解释,速查可借鉴具体可迁移 |
> | 可信赖 | pass | 关键数字抽查与 PDF 一致,指标名称正确 |
> | 可区分 | pass | 因果来源标注覆盖率 ~90%,agent 解读明确标注 |
> | 可定位 | pass | KB 背景有具体谱系定位,创新判断有对比基准 |
> | 不污染 | pass | 反向更新均为追加操作,无 factual error 风险 |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/Habibi-review.yml`
