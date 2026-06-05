---
type: paper
tier: deep
title: "NonverbalTTS: A Public English Corpus of Text-Aligned Nonverbal Vocalizations with Emotion Annotations for Text-to-Speech"
arxiv_id: "2507.13155"
source: "Sources/NonverbalTTS.pdf"
authors: [Maksim Borisov, Egor Spirin, Daria Diatlova]
year: 2025
venue: "arXiv"
tags: [dataset, non-verbal-vocalization, emotion-annotation, TTS, expressive-TTS, data-pipeline, English]
concepts: ["[[EmotionControlinTTS]]", "[[ProsodyModeling]]", "[[TTSEvaluation]]", "[[Speech-TextAlignment]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]", "[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/Whisper|Whisper]]", "[[模型库/WavLM|WavLM]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: ["[[数据集/Emilia|Emilia]]"]
kb_context_sources: 3
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[模型库/CosyVoice|CosyVoice]], [[模型库/CosyVoice2|CosyVoice 2]], [[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **[[模型库/CosyVoice|CosyVoice]]** (confirmed): 阿里巴巴语音实验室提出的 LLM + OT-CFM 零样本 TTS 系统。本文以 CosyVoice-300M 为基座模型,仅微调 LM 组件来验证 NVTTS 数据集对副语言发声合成的有效性。CosyVoice 原生不支持 cough/sigh 等 NV,在 laughter/breath 上也弱于 CosyVoice2,因此是展示 NVTTS 增益的理想起点。
>
> **[[模型库/CosyVoice2|CosyVoice 2]]** (confirmed): CosyVoice 的流式升级版,原生支持 breath、laughter、cough、sigh 四类 NV,但其 NV 训练数据为闭源私有数据集。本文将 NVTTS 微调后的 CosyVoice 与 CosyVoice2 做 head-to-head 对比,证明开源数据可达到与私有数据同等水平。CosyVoice2 区分单 token `[laugh]` 和多 token span `<laughter></laughter>`,而本文实现缺乏此粒度,这解释了 laughter Jaccard 的差距。
>
> **[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]** (confirmed): 给定短参考语音和目标文本,合成保持音色的语音。本文在零样本设定下验证 NV 合成能力 — 测试集使用 VoxCeleb 中训练时未见的说话人,评估同时覆盖 speaker similarity 和 NV fidelity。
>
> **[[概念库/EmotionControlinTTS|Emotion Control in TTS]]** [待确认]: 情感控制 TTS 的演进线已出现从"抽象情感状态"到"具体副语言行为"的分支 (NVSpeech, 2025)。本文 NVTTS 提供了 8 类情感标注,但实验发现去掉情感标签后 NV 生成质量略有提升,暗示情感-NV 联合建模并非总是互利的。
>
> **[[概念库/TTSEvaluation|TTS Evaluation]]** [待确认]: 本文使用 WER、SIM-o、DNSMOS、NV Jaccard distance 等标准客观指标,加上人类偏好测试。NV Jaccard distance 是较独特的评估方式 — 用 BEATs 检测生成和参考音频中的 NV,计算集合 Jaccard 距离。
>
> 检索命中: [[模型库/CosyVoice|CosyVoice]]✓, [[模型库/CosyVoice2|CosyVoice 2]]✓, [[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]✓ | 过滤: [[概念库/EmotionControlinTTS|Emotion Control in TTS]](pending-review), [[概念库/TTSEvaluation|TTS Evaluation]](pending-review), [[数据集/Emilia|Emilia]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个包含 10 类 NV + 8 类情感标注的 17 小时公开英语 TTS 数据集,配套自动检测-人工验证-多标注者融合 pipeline,微调 CosyVoice 后 NV 生成能力与闭源 CosyVoice2 持平
> - **路线**: VoxCeleb + Expresso 原始音频 → BEATs NV 检测 + MFA 对齐 → emotion2vec 情感分类 → 人工验证+过滤 → 多标注者融合 (Merge + Majority Vote) → 17h NVTTS 语料 → CosyVoice-300M LM SFT → 零样本 NV-capable TTS
> - **指标**: (NVTTS-no-emo vs CosyVoice2-no-emo) SIM-o 0.89 vs 0.75; WER 0.19 vs 0.18; DNSMOS 3.82 vs 3.93; NV Jaccard (全类) 0.80 vs 0.78; 人类偏好 33.4% vs 35.4% (无显著差异, p>0.05) [Table 8, Fig 1]
> - **可借鉴**: (1) NV 检测 (BEATs) + 强制对齐 (MFA) 组合实现自动 NV 定位; (2) 多标注者融合算法 (Merge + Align + Majority Vote) 是通用可复用的标注融合方案; (3) 仅微调 LM 组件 (非全参) + 25 epoch + 单卡 A100 即可赋予 NV 能力,成本极低; (4) NV Jaccard distance 作为 NV 生成保真度评估指标
> - **局限**: 仅 17h 规模偏小; 仅英语; NV tokenization 无单/多 token 区分 (导致 laughter Jaccard 低于 CosyVoice2); 去掉情感标签反而更好,情感标注价值未充分体现; BEATs 检测阈值设极低 (0.1) 可能引入噪声; 仅测试 CosyVoice 系列

## 核心问题

当前表达性 TTS 系统在副语言发声 (nonverbal vocalizations, NVs) — 笑声、咳嗽、叹气、呼吸等 — 方面受限于开源数据集的匮乏 [§1]。现有带 NV 标注的数据集要么录音质量差 (Switchboard 8kHz, Fisher 电话质量),要么 NV 类别过少 (Expresso 仅标注 laughter 和 breathing),要么有丰富 NV 但未反映在文本标注中 (Emilia 101K 小时,NV 信息未进入 transcription) [§2.1]。这直接导致研究者只能依赖闭源数据训练 NV 合成模型 (如 CosyVoice2 的 NV 训练数据不公开),可复现性和可比性严重受限 [§1]。

本文要解决的核心问题: **如何构建一个高质量、开源、NV 类别丰富的 TTS 数据集,使开源模型也能达到闭源系统的 NV 合成水平?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文的贡献是一个**数据集 + 标注 pipeline**,而非新模型架构。Pipeline 四步 [§3]:

```
原始音频 (VoxCeleb + Expresso)
    ↓ Step 1: BEATs NV检测 + MFA强制对齐 → NV在transcription中定位
    ↓ Step 2: emotion2vec+ → 8类情感标签
    ↓ Step 3: 人工验证 (Argilla) + 质量过滤
    ↓ Step 4: 多标注者融合 (Merge + Align + Majority Vote) → 最终标注
    → NVTTS (17h, 10类NV, 8类情感)
```

### 关键设计选择

#### 1. NV 检测: BEATs + MFA 联合定位 [§3.1]

选择 BEATs 模型做 NV 事件检测 (10 类 NV 基于 AudioSet Ontology 选取),检测阈值设为推荐最低值 0.1 [§3.1]。[论文原文] 低阈值的合理性在于后续有人工验证环节,所以宁可多检出再人工筛除 [§3.1]。

NV 在 transcription 中的精确定位通过 MFA (Montreal Forced Aligner) 实现: MFA 提供 word-level alignment,结合 BEATs 给出的 NV 时间戳,将 NV 标签插入到 transcription 的正确位置 [§3.1]。

[agent 解读] 这是与 NVSpeech 不同的路线: NVSpeech 先做人工 word-level 标注,再训练 paralinguistic-aware ASR; 本文用自动检测 (BEATs) + 自动对齐 (MFA) 实现同样的标注目的,人力成本更低但精度依赖 BEATs 的召回质量。

#### 2. 情感标注: emotion2vec+ large [§3.2]

对所有音频使用 emotion2vec+ large 分类器划分 8 类情感 (angry, disgusted, fearful, happy, neutral, sad, surprised, other) [§3.2]。[论文原文] 即便 Expresso 已有风格标签,仍选择统一使用 emotion2vec+ 分类,以保证标注一致性 [§3.2]。

约 8.5% 的样本因标注者无共识而未分配情感标签 [§4.2]。

#### 3. 人工验证与过滤 [§3.3]

标注员在 Argilla 平台上审校自动标注,执行三类过滤 [§3.3]:
- 非英语语音 → 丢弃
- 多说话人同时说话 → 丢弃
- NV 来自非主说话人 (如背景笑声) → 丢弃

标注员可调整 transcription (加/删/替换词)、修改 NV 标签、替换情感标签 [§3.3]。

#### 4. 多标注者融合 (Annotation Fusion): Merge + Majority Vote [§3.4]

这是本文在标注方法论上的独特贡献。给定 3 个标注者的 hypothesis:

**Step 1 — Merge (Algorithm 1)**: 依次对齐合并所有标注版本,使用 Pyalign 库做动态规划对齐,插入 gap 字符 ("-") 后合并,得到一个包含所有标注者见到的全部 token 的"超级版本" m3 [§3.4]。

**Step 2 — Align**: 将 m3 与每个标注者的 hypothesis 分别对齐,得到等长的对齐序列 a1, a2, a3 [§3.4]。

**Step 3 — Majority Vote (Algorithm 2)**: 逐位置检查,保留至少出现 2 次 (多数) 的 token,丢弃仅 1 人标注的 token [§3.4]。

[论文原文] 以示例说明: 3 人对 "dog"→"cat"、"[laugh]"→"[sigh]"、"mat"→"sofa" 存在分歧,融合后取多数: "cat" (3/3)、"[laugh]" (2/3)、"mat" (2/3) [§3.4, Table 3-4]。

[agent 解读] 该融合方法本质是一种 token-level 多数投票,其优势在于可以处理标注者在 NV 标签类型和数量上的不一致 (不仅仅是内容差异)。但它假设所有标注者等权重,且多数不一定正确 — 2 vs 1 的情况下少数者的标注可能反而更准确。

### 训练策略

实验使用 CosyVoice-300M 作为基座 [§5.1]:
- 仅微调 LM 组件 (不动 flow matching 或 vocoder) [§5.1]
- Adam optimizer, lr=1e-5 (constant), 25 epochs [§5.1]
- 单卡 A100, gradient accumulation every 2 batches [§5.1]
- 按 best validation metrics 选最优 checkpoint [§5.1]

四个消融模型 [§5.1]:
1. NVTTS-full: 完整数据
2. NVTTS-no-emotion: 去掉情感标签
3. NVTTS-no-NV: 去掉 NV 标签
4. NVTTS-no-emotion-no-NV: 两者都去掉

## 实验

| 指标 | NVTTS-no-emo | CosyVoice2-no-emo | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SIM-o ↑ | 0.89 | 0.75 | NVTTS test | [Table 8] |
| WER ↓ | 0.19 | 0.18 | NVTTS test | [Table 8] |
| EMO-SIM ↑ | 0.57 | 0.50 | NVTTS test (情感子集) | [Table 8] |
| DNSMOS ↑ | 3.82 | 3.93 | NVTTS test | [Table 8] |
| Jcough ↑ | 0.20 | 0.19 | NVTTS test | [Table 8] |
| Jbreath ↑ | 0.92 | 0.89 | NVTTS test | [Table 8] |
| Jlaugh ↑ | 0.25 | 0.34 | NVTTS test | [Table 8] |
| J (全类) ↑ | 0.80 | 0.78 | NVTTS test | [Table 8] |
| 人类偏好 | 33.4% | 35.4% | NVTTS test | [Fig 1] |

### 关键实验发现

**1. 开源数据 ≈ 闭源数据 [§5.3.3]**: 人类偏好测试中 CosyVoice2 被偏好率 35.4% vs NVTTS 33.4%,Wilson score 95% CI [30.4, 40.6],无显著差异 (p>0.05) [Fig 1]。这是在 CosyVoice2 使用闭源 NV 数据的前提下。

**2. 情感标签的消极影响 [§5.3.2]**: 去掉情感标注后 NV 生成质量略有提升 (J 从 0.79 升到 0.80),因此后续对比使用 no-emotion 版本。[agent 解读] 这可能是因为情感信息在小规模数据上反而分散了模型学习 NV 模式的注意力。

**3. NV 标签的必要性 [§5.3.2]**: 去掉 NV 标签后 NV 检测准确率下降 (J 从 0.80 降到 0.76),证明显式 NV 建模是必要的。

**4. SIM-o 优势 [§5.3.1]**: NVTTS 训练的模型 SIM-o (0.89) 大幅优于 CosyVoice2 (0.75/0.85),作者归因于 NVTTS 数据集的说话人多样性更高 [§5.3.1]。

**5. DNSMOS 略低 [§5.3.1]**: CosyVoice2 DNSMOS 更高 (3.93 vs 3.82),作者认为可能是因为 VoxCeleb 来源的训练数据本身含更多噪声 [§5.3.1]。

**6. Laughter Jaccard 差距 [§5.3.1]**: CosyVoice2 在 laughter 上 Jaccard 更高 (0.34 vs 0.25),作者归因于 CosyVoice2 区分 `[laugh]` (单 token) 和 `<laughter></laughter>` (多 token span) 两种 tokenization,而本文实现缺乏此粒度 [§5.3.1]。

## 局限性

1. **规模有限**: 仅 17 小时,与现代大规模语料 (Emilia 101K h) 相比微不足道。作者自认 "modest size" [§6]。
2. **仅英语**: 数据集仅覆盖英语,无法验证跨语言 NV 的适用性 [§Abstract]。与 NVSpeech (Mandarin) 互补但尚无 bridge。
3. **NV tokenization 粗糙**: 所有 NV 均为单标签 (如 `[laugh]`),不区分 NV 的持续时长或在语句中的 span [§5.3.1]。CosyVoice2 的细粒度 laughter tokenization 在实验中表现更好。
4. **情感标注价值存疑**: 消融实验表明去掉情感标签反而略好 [§5.3.2],这质疑了投入情感标注的回报。
5. **BEATs 低阈值噪声**: 检测阈值 0.1 虽有人工验证兜底,但 grunting 仅出现 7 例 [Table 2],说明部分 NV 类别数据量极少,统计意义有限。
6. **测试集偏斜**: 测试集情感分布严重偏向 Neutral 和 Happy [Table 7],情感合成能力的评估覆盖不完整。
7. **仅验证 CosyVoice**: 未在其他 TTS 系统 (如 F5-TTS、XTTS、Seed-TTS) 上验证数据集的迁移效果。

## 点评

**贡献定位**: NonverbalTTS 是一篇**数据集论文**,核心价值在于提供了一个开源替代品,使研究者不必依赖 CosyVoice2 等系统的闭源 NV 数据。论文充分展示了"开源 17h 可以追平闭源"这一实用结论。

**与 NVSpeech 的对比**: NVSpeech (Liao et al., 2025) 与本文目标相似但路线不同 — NVSpeech 从人工 word-level 标注出发训练 paralinguistic-aware ASR,再自动扩展到 573h,范围更广 (18 类 PV) 但仅覆盖中文。NonverbalTTS 用自动检测 (BEATs) + 自动对齐 (MFA) 替代昂贵的人工标注,人力成本更低但精度受 BEATs 性能约束。两者形成英/中互补。

**方法论亮点**: 多标注者融合算法 (Merge + Align + Majority Vote) 是一个通用的标注冲突解决方案,不限于 NV 标注。该方法在 NV 标签存在/类型/位置三个维度上同时做了投票,比简单的多数表决更精细。

**关键遗憾**: 情感标注花了大量人力,但实验证明它对 NV 生成没有帮助甚至略有负面影响。这暗示在小规模数据上同时建模 NV 和 emotion 可能是不明智的 — 但作者未深入讨论原因。

**工程价值**: 仅微调 LM 组件 + 单卡 A100 + 25 epoch 的训练配置非常轻量,适合资源有限的研究组复现。数据集已在 HuggingFace 公开。

## 可复用的 idea

1. **BEATs + MFA 自动 NV 定位 pipeline**: 将音频事件检测 (BEATs) 的时间戳与强制对齐 (MFA) 的 word boundary 结合,自动将 NV 标签插入文本正确位置,无需逐词人工标注。可迁移到任何需要将非语言事件与 transcription 关联的场景。

2. **多标注者融合算法 (Merge + Align + Majority Vote)**: 处理标注者在 NV 标签类型和数量上的不一致,比逐位投票更通用。可复用于任何多标注者序列标注任务 (如 POS tagging 冲突解决)。

3. **NV Jaccard distance 作为评估指标**: 用 BEATs 分别检测参考音频和生成音频中的 NV,计算集合 Jaccard 距离。简单有效,可复用于评估任何 NV-capable TTS 系统。

4. **仅微调 LM 组件赋予 NV 能力**: 在 CosyVoice 这类 LLM-based TTS 中,仅微调 LM (不动 flow matching 和 vocoder) 即可添加新的副语言控制能力,体现了 LLM-based 架构的模块化优势。

---

> [!review] 审阅结论: pass-with-fixes (2 low issues)
> 审阅时间: 2026-06-04 | checklist v1.1
> - [low] traceability-gap: 速查卡片指标应注明对比版本 → 已修正
> - [low] template-compliance: review callout 占位符 → 已修正
> 详见 `_review/NonverbalTTS-review.yml`

检索命中: [[模型库/CosyVoice|CosyVoice]]✓, [[模型库/CosyVoice2|CosyVoice 2]]✓, [[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]✓ | 过滤: [[概念库/EmotionControlinTTS|Emotion Control in TTS]](pending-review), [[概念库/TTSEvaluation|TTS Evaluation]](pending-review), [[数据集/Emilia|Emilia]](pending-review) | 未命中但可能相关: 无
