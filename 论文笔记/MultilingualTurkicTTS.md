---
type: paper
tier: deep
title: "Multilingual Text-to-Speech Synthesis for Turkic Languages Using Transliteration"
arxiv_id: "2305.15749"
source: "Sources/MultilingualTurkicTTS.pdf"
authors: [Rustem Yeshpanov, Saida Mussakhojayeva, Yerbolat Khassanov]
year: 2023
venue: "Interspeech 2023 (arXiv:2305.15749)"
tags: [TTS, multilingual, zero-shot, Turkic-languages, IPA, transliteration, low-resource, cross-lingual, Tacotron2]
concepts: ["[[PhonemeRepresentation]]", "[[Attention-basedTTS]]", "[[NeuralVocoder]]", "[[MelSpectrogram]]", "[[TTSEvaluation]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: ["KazakhTTS2"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页: [[Zero-shotSpeechSynthesis]], [[Cross-lingualVoiceCloning]], [[NeuralVocoder]], [[PhonemeRepresentation]][待确认], [[Attention-basedTTS]][待确认], [[MelSpectrogram]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文是 **pre-LLM 时代的跨语言零样本 TTS 工作**,使用 Tacotron 2 + IPA 转写实现 10 种突厥语系低资源语言的语音合成。与知识库中记录的现代零样本 TTS 系统(如 CosyVoice 3、Seed-TTS 等基于 LLM + 离散 token 的方案)在技术路线上完全不同: 本文不使用说话人 embedding 或参考音频做零样本克隆,而是通过 **字母表到 IPA 的手动映射** 将多语言统一到同一输入空间,利用语系内的语音学相似性实现跨语言生成。

**已有认知对比**:
- [[PhonemeRepresentation]][待确认] 页记录了 IPA 作为跨语言统一表示的方案,但主要关注的是现代系统(X-Voice 30 语言 IPA 实践、byte representation 等)。本文是较早期的 IPA 统一表示实践,覆盖 10 种同语系语言,手动构建映射表而非使用 eSpeak-NG 等工具。
- [[Attention-basedTTS]][待确认] 页详述了 Tacotron 2 的架构和局限性(推理慢、attention robustness 问题)。本文直接使用标准 Tacotron 2 架构,未针对跨语言场景做架构改进。
- [[Cross-lingualVoiceCloning]] 页中的现代系统(CosyVoice 3、X-Voice 等)普遍使用 multilingual LLM + 大规模多语言数据训练。本文的独特之处在于 **仅使用单一源语言(哈萨克语)数据**,完全依赖语言学先验(IPA 映射)实现跨语言,是一种极端的零资源方案。

**创新判断**: 与现代 LLM-TTS 系统相比,本文的方法论较为传统,但其核心贡献在于: (1) 首次系统性地为 10 种突厥语构建 IPA 映射; (2) 验证了同语系语言间通过语音学先验做零样本迁移的可行性; (3) 大规模主观评估(548 名评估者)提供了可信的跨语言质量基线。

> 检索命中: [[Zero-shotSpeechSynthesis]]✓, [[Cross-lingualVoiceCloning]]✓, [[NeuralVocoder]]✓ | 过滤: [[PhonemeRepresentation]](pending-review), [[Attention-basedTTS]](pending-review), [[MelSpectrogram]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 仅用哈萨克语数据训练 Tacotron 2,通过 IPA 转写实现 10 种突厥语的零样本多语言 TTS
> - **路线**: 目标语言文本 → IPA 转换器 → 哈萨克字母 → Tacotron 2 (单语训练) → 80-dim log-mel → WaveGAN vocoder → 波形
> - **指标**: 全语言平均 MOS 3.25/5.0, 可理解度 92%, 可辨度 41%; 目标语言最佳 MOS: 吉尔吉斯语 3.54 [Table 3]; 548 名评估者
> - **可借鉴**: 同语系语言间 IPA 作为"万能桥梁"的思路 -- 手动构建 42 个 IPA 符号的映射表,选择字母最多的语言(哈萨克语 42 个字母)作为源语言以最大化音素覆盖
> - **局限**: 仅限突厥语系内迁移; 未使用任何目标语言数据(包括无标注数据); MOS 最低仅 2.37(土库曼语); 可辨度整体偏低(41%); Tacotron 2 架构已过时; 未与现代多语言 TTS 对比

## 核心问题

1. **低资源多语言 TTS**: 10 种突厥语中多数缺乏高质量语音合成语料,如何在无目标语言数据的条件下实现 TTS?
2. **零样本跨语言迁移**: 仅使用单一源语言(哈萨克语)训练的 TTS 模型,能否通过语音学先验(IPA 转写)为其他 9 种语言合成可用的语音?
3. **评估覆盖度**: 如何系统评估如此多种低资源语言的 TTS 质量?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由两个主要模块组成 [§3, Fig 1]:

1. **IPA 转换器**: 将目标突厥语言的字母 → IPA 符号 → 哈萨克字母
2. **TTS 模型**: 基于 Tacotron 2 的端到端声学模型 + WaveGAN vocoder

工作流程: 目标语言文本输入 → IPA 转换器将字母映射到 42 个 IPA 符号 → IPA 符号再映射到哈萨克字母 → 输入预训练的 Tacotron 2 模型 → 输出 80 维 log-mel spectrogram → WaveGAN vocoder 合成波形。

### 关键设计选择

**为什么选择哈萨克语作为源语言?**
- KazakhTTS2 语料库是唯一公开可用的高质量突厥语 TTS 数据集(270+ 小时,5 个说话人) [论文原文, §1, §2]
- 哈萨克语字母表包含 42 个字母(与巴什基尔语并列最多),覆盖了其他突厥语言的绝大多数音素 [论文原文, §3]
- [agent 解读] 这意味着从哈萨克语到其他语言的映射很少出现"目标语言有而源语言没有"的音素缺失问题,最大化了迁移覆盖率。

**为什么使用 IPA 作为中间表示?**
- 突厥语言使用四种不同的书写系统(拉丁、西里尔、阿拉伯-波斯、混合) [论文原文, Table 1]
- IPA 提供了一个与书写系统无关的统一音素空间 [论文原文, §3]
- [agent 解读] 相比直接字母映射,IPA 中间层利用了语音学知识,可以处理"不同字母但同一发音"(如多种语言中 /A/ 音对应不同字母)的情况。

**IPA 映射的构建方式**:
- 完全手动构建,基于团队的语言学专业知识 [论文原文, §3]
- 未使用自动 G2P 工具,因为找不到覆盖所有 10 种语言且无误的完整映射 [论文原文, §3]
- 映射覆盖的 IPA 符号数: 47 (全集), 各语言覆盖 29-42 个 [Table 2]

**为什么只用一个说话人 (Speaker M2) 评估?**
- 虽然训练了 5 个说话人的模型(对应 KazakhTTS2 的 5 个声音),且全部可用于合成 [论文原文, §4.1]
- 但作者发现评估者不愿参与过长的评估问卷,因此仅选一个声音评估 [论文原文, §4.1]
- Speaker M2 的数据约 58 小时

### 训练策略

- **架构**: Tacotron 2,使用 ESPnet-TTS 工具包的 LJ Speech 训练配置 [§4.2]
- **编码器**: 单层双向 LSTM (512 单元, 每方向 256) [§4.2]
- **解码器**: 2 层单向 LSTM (1024 单元) [§4.2]
- **输入**: 42 个哈萨克字母 + 5 个标点符号 ('.', ',', '-', '?', '!') [§4.2]
- **输出**: 80 维 log-mel filterbank 特征 [§4.2]
- **声码器**: WaveGAN (Parallel WaveGAN) [§4.2]
- **优化器**: Adam, 初始学习率 1e-3, 200 个 epoch, dropout 0.5 [§4.2]
- **硬件**: NVIDIA DGX A100 [§4.2]
- **无目标语言数据**: 严格遵守零样本设定,训练中不使用任何目标语言数据 [§4.1]

[agent 解读] 标点符号被显式纳入输入空间,是因为 KazakhTTS2 录制时要求说话人注意标点(逗号处停顿、问号用正确语调)[§4.2],这使得模型能通过标点学到基本的韵律控制。

## 实验

### 评估方法 [§4.3]

三部分主观评估,通过 Qualtrics XM 在线问卷进行:
1. **整体质量 (Quality, Q)**: 10 段录音,5 级 Likert 评分 (MOS)
2. **可理解度 (Comprehensibility, C)**: 5 道多选题,评估听众能否理解语音内容
3. **可辨度 (Intelligibility, I)**: 5 个语义不可预测句子 (SUS),评估听众能否准确写出听到的内容

### 主要结果

| 语言 | 分支 | 评估人数 | MOS (Q) | 可理解度 (C) | 可辨度 (I) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| 哈萨克语 (源) | Kipchak | 151 | 4.18 | 97% | 80% | [Table 3] |
| 吉尔吉斯语 | Kipchak | 14 | 3.54 | 86% | 43% | [Table 3] |
| 土耳其语 | Oghuz | 18 | 3.25 | 91% | 61% | [Table 3] |
| 维吾尔语 | Karluk | 10 | 3.01 | 45% | 26% | [Table 3] |
| 阿塞拜疆语 | Oghuz | 47 | 2.93 | 90% | 52% | [Table 3] |
| 萨哈语 | Siberian | 254 | 2.85 | 93% | 15% | [Table 3] |
| 乌兹别克语 | Karluk | 22 | 2.85 | 80% | 45% | [Table 3] |
| 鞑靼语 | Kipchak | 15 | 2.82 | 79% | 17% | [Table 3] |
| 巴什基尔语 | Kipchak | 11 | 2.67 | 92% | 47% | [Table 3] |
| 土库曼语 | Oghuz | 6 | 2.37 | 67% | 57% | [Table 3] |
| **全部** | — | **548** | **3.25** | **92%** | **41%** | [Table 3] |

### 关键发现

1. **同分支迁移并非总是最好**: 与哈萨克语同属 Kipchak 分支的吉尔吉斯语表现最佳(MOS 3.54),但巴什基尔语(同分支)仅 2.67 [Table 3]。[agent 解读] 这暗示语音学相似度比语系分类更重要。

2. **萨哈语悖论**: MOS 仅 2.85,但可理解度高达 93% [Table 3]。作者指出,这表明用哈萨克语声音合成的录音对西伯利亚突厥语使用者来说相对容易理解 [论文原文, §5]。[agent 解读] 这可能是因为萨哈语母语者对"外国口音"的容忍度较高,或者突厥语系的核心音素在各分支间保持了足够的一致性。

3. **可辨度普遍偏低**: 全语言平均仅 41%,与 SUS 方法论本身的高认知负荷一致(原始 SUS 研究中可辨度在 10-20% 范围) [论文原文, §5, 引用 [26]]。

4. **土耳其语跨维度均衡**: 在三个维度上都表现不错(MOS 3.25, C 91%, I 61%),是目标语言中最均衡的 [Table 3]。[agent 解读] 可能得益于土耳其语与哈萨克语虽属不同分支(Oghuz vs Kipchak),但在元音和谐和黏着法等核心特征上高度一致。

## 局限性

1. **架构过时**: Tacotron 2 是 2018 年的架构,存在推理慢、attention robustness 差等已知问题,未与更现代的 TTS 系统对比 [agent 解读]
2. **评估规模不均**: 各语言评估者数量差异极大 (6 人 ~ 254 人),土库曼语仅 6 人,统计显著性存疑 [Table 3]
3. **纯零样本局限**: 完全不使用目标语言数据,作者也承认 fine-tuning 可以进一步提升效果 [论文原文, §5]
4. **IPA 映射的局限**: 手动映射无法处理借词(尤其是俄语借词在前苏联突厥语中大量存在)和同形字符(homoglyphs)问题 [论文原文, §6]
5. **单说话人评估**: 仅用 1 个男性说话人评估,缺乏多说话人和女性声音的验证 [§4.1]
6. **无客观指标**: 完全依赖主观评估,未报告 WER、CER 等客观指标 [agent 解读]
7. **不支持 code-switching**: 突厥语中频繁出现的俄语混用无法处理 [论文原文, §6]

## 点评

这是一篇以语言学先验驱动的实用性工作,核心价值在于**系统性地验证了同语系内通过 IPA 转写实现零资源 TTS 的可行性**。在方法论上并无新意(Tacotron 2 + 字母映射),但在覆盖度(10 种语言)和评估规模(548 人)上做得扎实。

与现代零样本 TTS 系统对比,本文的方法论已明显落后: 现代系统(如 X-Voice)在 30 种语言上使用 eSpeak-NG 自动 G2P + 420K 小时多语言数据训练,直接跨越了"手动映射"和"单语言训练"两个瓶颈。但本文的核心 insight -- **同语系语言间的语音学相似性可以作为免费的跨语言先验** -- 对低资源场景仍有参考价值,尤其是在无法获取大规模多语言数据的情况下。

论文最有价值的贡献是 **IPA 映射表本身** (Table 2) 和对突厥语 TTS 数据现状的全面调查 (§2),为后续研究提供了实用的基础设施。

## 可复用的 idea

1. **语系内 IPA 桥接策略**: 选择字母表最大的语言作为源语言,最大化音素覆盖率。这个原则可推广到其他语系(如南亚语系、班图语系)的低资源 TTS
2. **三维评估框架**: 质量 + 可理解度 + 可辨度的三维评估比单一 MOS 更全面,尤其适合跨语言场景中区分"听着不自然但能理解"和"完全听不懂"两种失败模式
3. **分支-迁移效果矩阵**: 用语系分支结构预测跨语言迁移效果的好坏,可用于优先级排序

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | WHY 解释充分,设计选择有因果链 |
> | 可信赖 | pass | 数字标注覆盖率 ~95%,指标名正确 |
> | 可区分 | pass | 来源标注覆盖率 ~100%,无混淆 |
> | 可定位 | pass | KB 谱系定位清晰,与现代系统对比具体 |
> | 不污染 | pass | 无新建,反向更新均为追加 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> 详见 `_review/MultilingualTurkicTTS-review.yml`
