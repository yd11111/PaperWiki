---
type: paper
tier: deep
title: "An Automated End-to-End Open-Source Software for High-Quality Text-to-Speech Dataset Generation"
arxiv_id: "2402.16380"
source: "Sources/AutomatedTTSDataset.pdf"
authors: [Ahmet Gunduz, Kamer Ali Yuksel, Kareem Darwish, Golara Javadi, Fabio Minazzi, Nicola Sobieski, Sébastien Bratières]
year: 2024
venue: "LREC-COLING 2024"
tags: [TTS, dataset-generation, data-quality, phoneme-coverage, ASR, quality-assurance, open-source, recording-pipeline]
concepts: ["[[PhonemeRepresentation]]", "[[TTSEvaluation]]", "[[Text-to-SpeechPipeline]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 5 个待确认实体页: PhonemeRepresentation, Whisper, TTSEvaluation, Text-to-SpeechPipeline, Emilia)
> 基于未确认概念页,仅供参考。
>
> **谱系定位**: 本文属于 TTS 数据工程方向,解决的是 TTS 模型训练的上游问题——如何高效构建高质量录音数据集。与 [[Emilia]] 等大规模 in-the-wild 数据集不同,本文面向的是**受控录音场景**(studio recording with voice actors),更接近 LJSpeech / LibriTTS 的传统范式,但增加了自动化工具链。
>
> **已有认知**:
> - [[PhonemeRepresentation]]: 音素覆盖是 TTS 数据集设计的经典问题,phoneme-balanced selection 在传统 TTS 数据集构建中被广泛使用。本文的 contribution 是将其自动化并集成到端到端工具中。
> - [[Whisper]]: 本文最初使用 Whisper 做 ASR-based QA,但发现其 auto-correction 功能导致转录失真,后改用 aiXplain 平台的其他 ASR。这为 Whisper 在数据管线中的使用提供了一个值得注意的失败案例。
> - [[TTSEvaluation]]: WER (Word Error Rate) 是本文 QA 管线的核心指标,用于衡量录音与参考文本的匹配度。
> - [[Text-to-SpeechPipeline]]: 本文服务于 TTS pipeline 的最上游——训练数据准备阶段。
> - [[Emilia]]: 与本文形成有趣对比——Emilia 是从 in-the-wild 音频中自动处理得到 100k+ 小时数据; 本文是从零开始的受控录音流程,目标 30h/语言,强调人工质控。两者代表 TTS 数据获取的两个极端路线。
>
> **创新判断**: 本文的核心价值不在算法创新,而在工程集成——将 phoneme-balanced 选句、自动录音管理、ASR-based QA、两轮人工审核整合为一个开源工具。在 TTS 数据集构建领域,类似的开源端到端工具确实稀缺。
>
> 检索命中: [[PhonemeRepresentation]][待确认], [[Whisper]][待确认], [[TTSEvaluation]][待确认], [[Text-to-SpeechPipeline]][待确认], [[Emilia]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出首个开源端到端 TTS 数据集生成工具,集成 phoneme-balanced 选句、自动录音管理、ASR-based QA 和预处理,覆盖 6 种语言。
> - **路线**: OPUS 语料库文本 → phoneme-balanced 迭代选句 → voice actor 录音(批量/单句) → VAD 分割 + ASR 匹配 → 两轮人工标注审核 → 格式化输出
> - **指标**: 句子匹配率 ~98% [Table 3]; 二次 QC 丢弃率 0-0.15% [Table 4]; 编辑率 1.25-11.38% [Table 4]; 6 语言各 30h 目标
> - **可借鉴**: phoneme 分布驱动的迭代选句算法可迁移到任何需要平衡语音覆盖的场景; Whisper auto-correction 导致数据管线失败的教训值得注意
> - **局限**: 仍需专业 voice actor + 母语标注员团队; 规模受限(30h/语言 vs Emilia 100k+h); 无 MOS 等合成质量评估; 依赖 ASR 模型可用性(低资源语言受限)

## 核心问题

本文要解决的核心问题是: **如何将传统 TTS 数据集构建流程(文本选择 → 录音 → 质检 → 后处理)从人工密集型转变为半自动化流程,并保证数据质量?**

具体拆解为四个子问题:
1. 如何选择文本样本以保证语言的 phoneme 覆盖完整性?
2. 如何管理 voice actor 的录音过程(尤其是批量录音的自动切分)?
3. 如何利用 ASR 自动验证录音质量,减少人工审核负担?
4. 如何标准化整个流程为一个可复用的开源工具?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由四个模块串联 [§4, Fig 1]:

```
文本预处理 → 样本选择(phoneme-balanced) → 录音管理 → ASR-based QA → 人工标注 → 后处理
```

技术栈: Streamlit (UI) + PostgreSQL (存储) + Celery + Redis (异步任务) + Docker Compose (部署) + aiXplain SDK (ASR/VAD) [§4.3.2, Fig 2]

### 关键设计选择

**1. Phoneme-balanced 迭代选句 [§4.1]**

[论文原文] 目标是让选出的子集的 phoneme 分布尽可能逼近全语料库的 phoneme 分布。

具体流程:
- 对每个句子用 Espeak Phonemizer 生成 monophone、diphone、triphone [§4.1]
- 统计全语料库的 phoneme 频率分布作为参考基准
- **迭代采样**: 每轮随机选句,但给能减小子集与全语料库 phoneme 分布 divergence 的句子赋予更高采样概率 [§4.1]
- 同时施加句子类型(陈述 80%、疑问 10-15%、感叹 5-10%)和长度(5-13 词)约束 [Table 1]
- 目标量: 每语言 ≥600,000 词,按 2.75 词/秒 估算可生成 30+ 小时音频 [§4.1]

[agent 解读] 这是经典的 greedy phoneme-coverage 策略的随机化变体。传统方法通常用确定性贪心选择,本文用概率加权可能是为了避免局部最优并保持句子多样性。论文未给出与确定性方法的对比实验。

**2. 批量录音的自动切分与匹配 [§4.2]**

voice actor 可选两种录音模式:
- 单句录音: 每句一个文件
- 批量录音: 多句连续录入一个文件(≤500 句/文件),句间留 ≥2 秒间隔

批量录音的自动处理:
1. VAD 将连续音频切分为段落
2. ASR 转录每个段落
3. 计算 ASR 输出与候选句子间的 Levenshtein 编辑距离
4. 选择编辑距离最小的句子匹配,条件: edit_distance / min(len_asr, len_ref) < 0.2 且长度差 < 20% [§4.2]
5. 匹配后用 VAD trimming 去除前后超过 100ms 的静音,保留至少 25ms 防止截断 [§4.2]

**3. Whisper 的失败与替换 [§6]**

[论文原文] 最初选用 Whisper 是因为其 language-agnostic 能力。但 Whisper 的 **auto-correction** 功能在批量录音场景中导致严重问题: 当 voice actor 录错后重读(两次朗读间隔不够),VAD 将重复内容当作一个段落,Whisper 进一步"纠正"转录文本,导致 ASR 输出与参考句子不匹配 [§6]。

[agent 解读] 这是 Whisper 作为 TTS 数据管线组件的一个重要 anti-pattern: Whisper 的鲁棒性在 ASR 评测中是优势,但在需要**忠实转录**(verbatim transcription)的数据管线场景中反而是缺陷。这与 [[TTSEvaluation]] 中讨论的 WER 局限性相关——ASR 模型本身的偏差会影响下游质量评估。

**4. 两轮人工 QA [§4.4, §5.3]**

- **第一轮**: 母语标注员逐句审核,可编辑文本、标记问题(重复、韵律不当、不一致、噪音) [§4.4]
  - 系统按 WER 从高到低排序,优先审核 ASR 认为最可疑的录音 [§4.4]
  - 并发支持多标注员,每条样本锁定到一个标注员 [§4.4]
- **第二轮**: 独立的第二组母语标注员审核 ≥70% 数据,重点验证第一轮标注质量 [§5.3]
  - [论文原文] 特别关注标点与语音的一致性(subordinate/coordinate sentences, exclamations, questions) [§5.3]

### 训练策略

本文不涉及模型训练,是纯数据工具。

## 实验

| 指标 | 本文 | 说明 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| 句子匹配率 | 94.9%-99.8% | VAD+ASR 自动匹配成功率 | 6 语言 9 批次 | [Table 3] |
| Trimming 压缩比 | ~14% 时长缩减 | 如 DE File1: 1549→1330s | DE/FR/ES/IT/EN | [Table 3] |
| 二次 QC 编辑率 | 1.25% (ES) - 11.38% (IT) | IT 显著偏高(speaker 表现问题) | 5 语言 | [Table 4] |
| 二次 QC 丢弃率 | 0.00% - 0.15% | 几乎无需丢弃 | 5 语言 | [Table 4] |
| 问题类型分布 | Bad Prosody: 2, Truncation: 3, Artifacts: 21 | 集中在 EN 和 DE | EN/DE | [Table 4] |
| 每语言目标时长 | 30 小时 | 基于 600k 词 × 2.75 词/秒 | 6 语言 | [§4.1] |

**关键观察**:
- IT (意大利语) 编辑率异常高(11.38%),论文归因于 speaker 执行质量差 [§5.4]
- EN File1 匹配率较低(94.9%),可能与 voice actor 标注错误有关 [§6]
- 整体丢弃率极低(≤0.15%),说明录音基础质量有保障

## 局限性

1. **规模受限**: 每语言 30h 在当前大数据 TTS 时代(Emilia 100k+h, LibriLight 60k+h)显得非常小,主要适用于低资源语言或精控场景 [agent 解读]
2. **依赖专业团队**: 需要专业 voice actor + 母语标注员,成本高、扩展性差 [§5.5, §8]
3. **ASR 依赖**: QA 管线依赖高质量 ASR,低资源语言可能无可用 ASR 模型 [§6]
4. **无合成评估**: 论文仅评估了数据集构建流程的效率(匹配率、编辑率),未用生成的数据集训练 TTS 模型并评估合成质量(如 MOS) [agent 解读]
5. **Whisper 问题的解决方案模糊**: 仅说"使用其他 ASR",未量化替换后的改善 [§6]
6. **单 speaker per language**: 每种语言似乎只有一位 voice actor,限制了 speaker diversity [agent 解读,基于实验设置]
7. **选句算法未对比**: phoneme-balanced 选句与随机选句、确定性贪心选句的效果未做消融对比 [agent 解读]

## 点评

这是一篇**工具/系统论文**而非方法论文。核心贡献在于将 TTS 数据集构建的各个已知环节整合为一个开源工具(github.com/aixplain/tts-qa)。

**优点**:
- 填补了开源 TTS 数据集构建工具的空白(论文 claim"首个同类工具" [§1])
- 多语言验证(6 种语言)证明了工具的通用性
- Whisper auto-correction 的失败教训是有实际价值的经验
- 两轮独立 QA 设计合理,第一轮 ASR 预排序(按 WER 降序)是聪明的做法

**不足**:
- 缺少最关键的验证: 用这个数据集训练出的 TTS 效果如何? 数据集质量的黄金标准应该是下游任务表现
- 与 Emilia-Pipe 等现代数据管线相比,在规模和自动化程度上都有明显差距
- 选句算法的理论分析薄弱(divergence 用什么度量? KL? 未说明)

**定位**: 适合需要构建小规模高质量受控录音数据集的场景(如低资源语言 TTS、定制 voice),不适合大规模数据驱动的现代 TTS 方法。

## 可复用的 idea

1. **ASR 预排序的 QA 策略**: 按 ASR WER 降序排列样本供人工审核,让标注员优先处理最可疑的样本 → 可迁移到任何 human-in-the-loop 数据标注场景
2. **Whisper verbatim 问题的意识**: 在需要忠实转录(非语义理解)的管线中,Whisper 的 auto-correction 可能是 bug 而非 feature → 选择 ASR 时需区分"理解型"与"忠实型"使用场景
3. **Phoneme divergence 驱动的选句**: 用全语料库的 phoneme 分布作为 target,迭代采样使子集分布逼近 → 可迁移到任何需要平衡覆盖的数据子集选择问题(不限于语音)
4. **批量录音的 VAD+Levenshtein 匹配**: 解决"一个音频文件包含多个句子"的自动切分与对齐问题,阈值设计(edit ratio < 0.2, length diff < 20%)有实操参考价值

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 四模块机制清晰,因果解释充分,divergence 度量论文本身未指定已标注 |
> | 可信赖 | pass | 关键数字全部与 PDF 交叉验证通过,出处标注覆盖率高 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注清晰,无断言式推断 |
> | 可定位 | pass | KB 背景受控录音 vs in-the-wild 定位精准,Emilia 对比有价值 |
> | 不污染 | pass | 不涉及新概念页创建,反向更新仅追加 key_papers |
> 
> Issues: 4 (high: 0, medium: 1, low: 3)
> 详见 `_review/AutomatedTTSDataset-review.yml`
