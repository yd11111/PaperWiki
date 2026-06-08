---
type: paper
tier: deep
title: "The NeurIPS 2023 Machine Learning for Audio Workshop: Affective Audio Benchmarks and Novel Data"
arxiv_id: "2403.14048"
source: "Sources/HUME-VOCALBURST.pdf"
authors: [Alice Baird, Rachel Manzelli, Panagiotis Tzirakis, Chris Gagne, Haoqi Li, Sadie Allen, Sander Dieleman, Brian Kulis, Shrikanth S. Narayanan, Alan Cowen]
year: 2024
venue: "NeurIPS 2023 MLA Workshop"
tags: [dataset, emotion-recognition, vocal-burst, prosody, affective-computing, speech-emotion, paralinguistic, benchmark]
concepts: ["[[EmotionControlinTTS]]", "[[ProsodyModeling]]", "[[Self-SupervisedSpeechRepresentation]]", "[[AudioUnderstanding]]", "[[TTSEvaluation]]"]
models: ["[[模型库/wav2vec2.0|wav2vec 2.0]]"]
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页: [[ProsodyModeling]]✓ + 1 个待确认: [[EmotionControlinTTS]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]], [[EmotionControlinTTS]], [[Self-SupervisedSpeechRepresentation]], [[AudioUnderstanding]], [[TTSEvaluation]] | 过滤: [[EmotionControlinTTS]](pending-review), [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review), [[TTSEvaluation]](pending-review) | 未命中但可能相关: 无

**谱系定位**: 本文不是方法论文而是**数据集/benchmark 白皮书**,发布四个情感语音数据集 (HUME-PROSODY, HUME-VOCALBURST, MODULATE-SONATA, MODULATE-STREAM),服务于语音情感识别 (SER)、vocal burst 分类、语音生成等任务。在 PaperWiki 知识体系中,它与 [[EmotionControlinTTS]] 有上游关系: 情感 TTS 的训练和评估需要高质量情感标注数据,而 HUME-PROSODY 和 MODULATE-SONATA 正是这类资源。同时,本文来自 Hume AI 团队,与本 vault 中 [[论文笔记/TADA|TADA]] 共享核心作者 (Panagiotis Tzirakis, Alice Baird, Chris Gagne, Alan Cowen),TADA 的 TTS 工作正是建立在 Hume AI 的情感数据基础设施之上。

**已有认知**: KB 中 [[EmotionControlinTTS]] 记录了 TTS 情感控制的方法演进 (从 embedding → 层级建模 → DPO/RLHF → activation steering → SAE),但缺少情感数据集的系统整理。KB 中 [[ProsodyModeling]] 已涵盖 prosody 的物理维度 (duration/pitch/energy/pause) 和建模方法。本文补充了**情感标注的数据侧基础设施**,包括 Cowen 的 48 维语义空间模型 (semantic space theory) 和 10 维 vocal burst 情感体系。

**创新判断**: 本文的核心贡献不在方法而在**数据**: (1) HUME-VOCALBURST 是目前规模最大的跨文化 vocal burst 数据集 (36h, 1702 speakers, 4 countries); (2) MODULATE-SONATA 是专业配音演员的 25 类情感表演数据,适合做 SER 和情感 TTS 的 benchmark; (3) MODULATE-STREAM 是 7000h 游戏直播音频,适合无监督/自监督 audio 任务。

## 速查

> [!summary] 速查
> - **一句话**: NeurIPS 2023 MLA Workshop 白皮书,发布四个情感/副语言音频数据集 (HUME-PROSODY, HUME-VOCALBURST, MODULATE-SONATA, MODULATE-STREAM),配套多个 SER/vocal-burst 竞赛基线
> - **路线**: 数据收集 (众包模仿/专业演员/游戏直播) → 情感标注 (多 rater intensity 评分) → 标准化 (16kHz/48kHz, speaker-independent splits) → Benchmark 任务定义 (ExVo MTL/Generation/FewShot + A-VB High/Two/Culture/Type)
> - **指标**: ExVo-MultiTask S_MTL 0.435 (NLPros best) [Table 8]; A-VB High CCC 0.736 (EIHW best) [Table 9]; MODULATE-SONATA 25-class UAR 0.90 (early fusion Wav2Vec2+HuBERT) [Table 10]
> - **可借鉴**: (1) 跨文化 vocal burst 数据设计 (4 countries, seed imitation 范式); (2) 48 维 semantic space 情感标注框架 (Cowen 2021); (3) HEEP 指标结合 FID 评估 vocal burst 生成质量; (4) Speaker-independent partition 确保公平评估
> - **局限**: 数据集为特定任务设计,非通用 SER benchmark; HUME 数据集 test set blind (需提交至 competitions@hume.ai); MODULATE-STREAM 无 baseline (7000h 纯音频); 白皮书性质,无方法创新

## 核心问题

1. **为什么情感/副语言音频数据集稀缺?** 音频是时间依赖模态,高质量数据收集成本远高于文本/图像; 精细情感标注需要多 rater 共识; 学术界缺乏大规模标注资源,限制了 SER/情感 TTS 的发展 [§1]
2. **如何获取多样化的情感语音数据?** 两种范式: (a) HUME 系列用 seed imitation — 播放情感 seed 样本让众包工人模仿,获得自然度较高的 in-the-wild 录音 [§2.1]; (b) MODULATE-SONATA 用专业演员按脚本表演 25 种情感角色 [§2.3]
3. **vocal burst 与 speech prosody 在情感标注上有何区别?** HUME-PROSODY 标注的是带文本语音的 prosodic emotion (9 类 valence-arousal balanced), HUME-VOCALBURST 标注的是非语言发声 (笑/哭/叫/叹) 的 10 类情感强度 (Amusement, Awe, Awkwardness 等) [§2.1, §2.2]
4. **如何评估 vocal burst 的生成质量?** ExVo Generate 任务结合 FID (声学分布距离) 和 HEEP (human evaluation of emotional plausibility) 为 S_GEN 指标 [Eq. 4] [§3.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 数据集概览

本文为 NeurIPS 2023 MLA Workshop 发布四个数据集 [§2]:

| 数据集 | 规模 | 说话人 | 类型 | 标注 | 来源 |
|--------|------|--------|------|------|------|
| HUME-PROSODY | 41h48m55s | 1,004 | 带文本情感语音 | 9 情感 intensity (1-100) | [§2.1, Table 1] |
| HUME-VOCALBURST | 36h47m04s | 1,702 | 非语言发声 | 10 情感 intensity (1-100) | [§2.2, Table 2] |
| MODULATE-SONATA | 6h+ | 23 actors, 15 roles | 演员情感表演 | 25 情感 class | [§2.3, Table 3] |
| MODULATE-STREAM | 7,000h (379GB) | 940 players | 游戏直播 | 无标注 (transcript + metadata) | [§2.4, Table 4] |

### HUME-PROSODY: 带文本的情感韵律数据

[论文原文] 数据来源于更大的 Hume-Prosody 数据集的子集,完整数据集包含 48 维 emotional expression dimensions (基于 Cowen 的 semantic-space model [19]) [§2.1]。本次发布选取 9 类 valence-arousal 均衡的情感: Anger, Boredom, Calmness, Concentration, Determination, Excitement, Interest, Sadness, Tiredness [§2.1]。

**数据收集范式**: 使用 MELD [15] 和 VENEC [16] 等公开数据集中 5000+ 条 seed 样本,参与者通过众包平台 (MTurk, Prolific 等) 模仿 seed 的韵律表现,用自己的麦克风 in-the-wild 录制 [§2.1]。

[agent 解读] 这种 seed imitation 范式在众包质量和自然度之间取得了折衷: 相比纯自由表达,seed 提供了情感基准; 相比实验室录制,in-the-wild 环境增加了录音条件的多样性。

**标注**: 每条样本的 intensity 评分从 1-100 归一化到 0-1。数据被 speaker-independently 划分为 train/dev/test [§2.1]。

### HUME-VOCALBURST: 非语言情感发声

[论文原文] 包含非语言发声 (vocal bursts): 笑声、哭泣、尖叫、叹息等,来自 4 个国家 (中国、南非、美国、委内瑞拉) 的 1,702 名说话人 (年龄 20-39 岁) [§2.2]。每条 burst 由平均 85.2 名 rater 评分 10 种情感的 intensity (1-100): Amusement, Awe, Awkwardness, Distress, Excitement, Fear, Horror, Sadness, Surprise, Triumph [§2.2]。

**跨文化设计**: [agent 解读] 4 国设计使得 cross-cultural emotion 分析成为可能,这也是后续 A-VB Cross-Cultural 竞赛任务的数据基础。

**数据划分**: 19,990 train / 19,396 dev / 19,815 test (speaker-independent),test set 的 speaker 信息不公开 [Table 2]。

### MODULATE-SONATA: 专业演员情感表演

[论文原文] 23 名专业配音演员表演 15 种角色 (anime voice archetypes + 知名演员 impression + fantasy characters),每种角色按脚本朗读 25 种情感类别的句子 (2-6 句/情感) [§2.3]。

[论文原文] 25 种情感类别涵盖: adoration, amusement, anger, awe, confusion, contempt, contentment, desire, disappointment, disgust, distress, elation, embarrassment, fear, hype, interest, pain, realization, relief, sadness, seduction/ecstasy, surprise (positive/negative), sympathy, triumph [§2.3]。

**评估**: 使用 HuBERT 和 Wav2Vec2 embedding + Logistic Regression 做 25-class speech emotion recognition baseline,early fusion (HuBERT + Wav2Vec2) UAR 达 0.90 [Table 10]。

### MODULATE-STREAM: 大规模游戏直播音频

[论文原文] 7000+ 小时公开可用的游戏直播音频,940 名玩家,2839 个 session,379GB opus 格式 [§2.4]。无情感标注,但提供 transcript (wav2vec-based STT)、player ID、session ID、clip duration、game metadata [§2.4]。

[agent 解读] 该数据集的定位是支持无监督/自监督音频任务,如 pre-training 或 unsupervised emotion clustering。由于规模巨大且自然,适合作为 in-the-wild 情感数据的补充。

### Benchmark 任务设计

**ExVo 竞赛 (HUME-VOCALBURST)** [§3.3]:

| 任务 | 目标 | 评估指标 | 出处 |
|------|------|----------|------|
| ExVo Multi-Task | 同时预测 10 情感 + 年龄 + 国籍 | S_MTL (harmonic mean of CCC, MAE, UAR) | [Eq. 1, §3.3] |
| ExVo Generate | 生成 10 类 vocal burst | S_GEN = (1/FID + HEEP) / 2 | [Eq. 4, §3.3] |
| ExVo FewShot | 2-shot 个性化情感识别 | CCC | [§3.3] |

**A-VB 竞赛 (HUME-VOCALBURST)** [§3.3]:

| 任务 | 目标 | 评估指标 | 出处 |
|------|------|----------|------|
| A-VB High | 10 维情感 intensity 回归 | Mean CCC | [§3.3] |
| A-VB Two | Arousal-Valence 2 维回归 | Mean CCC | [§3.3] |
| A-VB Culture | 跨 4 国的 40 维 (10x4) 情感回归 | Mean CCC | [§3.3] |
| A-VB Type | 8 类 burst 类型分类 (Laugh, Cry, Scream...) | UAR | [§3.3] |

### 关键设计选择

#### 1. 情感标注框架: 为什么用 48/10 维 intensity 而非离散类别?

[论文原文] 完整 Hume-Prosody 数据集基于 Cowen 的 semantic-space model for emotion [19],包含 48 维情感表征 [§2.1]。本次发布选取 9 类 valence-arousal 均衡的子集用于 ComParE 2023 挑战赛 [20]。

[agent 解读] 使用连续 intensity (1-100) 而非离散类别的设计哲学来自 Cowen & Keltner (2021) 的 semantic space theory: 情感不是离散类别而是高维连续空间中的区域。这允许建模混合情感 (如"既惊又喜") 和情感强度,比传统 Ekman 6 基本情感更细粒度。这一框架与 [[EmotionControlinTTS]] 中 UDDETTS 的 ADV 空间和 EmoSphere 的球面表示遥相呼应。

#### 2. Seed Imitation vs Professional Acting

[agent 解读] 两种数据收集范式各有优劣:
- **Seed imitation (HUME)**: 说话人多 (1000+)、录音条件多样 (in-the-wild)、情感自然度中等; 适合训练泛化性强的模型
- **Professional acting (MODULATE-SONATA)**: 说话人少 (23)、录音质量高 (Neumann U87 等)、情感表达夸张但清晰; 适合作为 benchmark 或 TTS 情感生成的参考

#### 3. S_GEN 评估指标: FID + HEEP 的组合

[论文原文] vocal burst 生成的评估结合了客观 FID (Fr echet Inception Distance,衡量生成分布与真实分布距离) 和主观 HEEP (Human Evaluation of Emotional Plausibility,衡量人类对情感可信度的评分与目标矩阵的相关性) [Eq. 2-4, §3.3]。

[agent 解读] 这种组合设计的原因在于: FID 衡量声学质量,HEEP 衡量情感准确性,两者缺一不可。一个声学上逼真但情感错误的 burst 应得低分,反之亦然。

### 训练策略

本文是数据集论文,不涉及模型训练策略。Baseline 方法简述:

- **HUME-PROSODY baseline** [§3.2]: Wav2Vec2 (fine-tuned on MSP-Podcast) embeddings → SVM Regressor (cost C optimized on dev) → test Pearson ρ = 0.514 (best); Late Fusion with ComParE features → 0.476 [Table 5]
- **HUME-VOCALBURST baseline** [§3.4]: Feature-driven (openSMILE/ComParE [24]) 和 end-to-end (LSTM [25]) 方法; A-VB 2022 baseline CCC: High 0.569 / Two 0.508 / Culture 0.440 / Type UAR 0.417 [Table 7]
- **MODULATE-SONATA baseline** [§3.7]: Wav2Vec2/HuBERT mean pooling → Logistic Regression; early fusion UAR 0.90 [Table 10]

## 实验

### ExVo Workshop 2022 (HUME-VOCALBURST)

| 任务 | 最佳方法 | S_MTL/S_GEN/C | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ExVo MultiTask | NLPros [31] | S_MTL = 0.435 | HUME-VOCALBURST | [Table 8] |
| ExVo MultiTask | Organisers [13] (ComParE) | S_MTL = 0.335 | HUME-VOCALBURST | [Table 8] |
| ExVo Generate | StyleMelMila [33] | S_GEN = 0.408 | HUME-VOCALBURST | [Table 8] |
| ExVo FewShot | SaruLab-UTokyo [34] | CCC = 0.739 | HUME-VOCALBURST | [Table 8] |

### A-VB Workshop 2022 (HUME-VOCALBURST)

| 任务 | 最佳团队 | CCC/UAR | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| A-VB High | EIHW [40] | CCC = 0.736 | HUME-VOCALBURST | [Table 9] |
| A-VB Two | EIHW [40] | CCC = 0.707 | HUME-VOCALBURST | [Table 9] |
| A-VB Culture | EIHW [40] | CCC = 0.620 | HUME-VOCALBURST | [Table 9] |
| A-VB Type | HCAI [37] | UAR = 0.586 | HUME-VOCALBURST | [Table 9] |

### Baseline 对比 (A-VB 2022)

| 方法 | High CCC | Two CCC | Culture CCC | Type UAR | 出处 |
| --- | --- | --- | --- | --- | --- |
| ComParE [24] | 0.521 | 0.499 | 0.380 | 0.384 | [Table 7] |
| END2YOU [25] | 0.569 | 0.508 | 0.440 | 0.417 | [Table 7] |

### MODULATE-SONATA 25-class Emotion Recognition

| Embedding | Validation UAR | Test UAR | 出处 |
| --- | --- | --- | --- |
| Wav2Vec2 | 0.80 | 0.85 | [Table 10] |
| HuBERT | 0.79 | 0.86 | [Table 10] |
| Early Fusion | 0.82 | 0.90 | [Table 10] |

## 局限性

1. **白皮书性质,无方法创新**: 本文定位为数据集发布论文,不提出新模型或新方法,主要贡献在数据资源而非算法 [§4]
2. **Test set 不公开**: HUME-PROSODY 和 HUME-VOCALBURST 的 test set 标签保持 blind,需提交预测至 competitions@hume.ai 评估 [§3],限制了独立复现和分析
3. **数据收集偏差**: HUME 数据通过众包 seed imitation 获取,存在模仿质量不均、录音设备差异大的问题; MODULATE-SONATA 仅 23 名演员,说话人多样性受限 [§2.1, §2.3]
4. **缺少跨数据集分析**: 四个数据集之间无联合实验,如 HUME-PROSODY 上训练的模型在 MODULATE-SONATA 上的泛化性未知
5. **MODULATE-STREAM 无 baseline**: 7000h 数据无标注也无 baseline,实际使用价值有待后续验证 [§3.6]
6. **情感类别覆盖不均**: VOCALBURST 中 Triumph 类的 FID/HEEP 缺失 (无生成样本) [Table 8(b)]; PROSODY 仅 9/48 类

## 点评

**优势**:
1. **填补情感语音数据空白**: 在情感 TTS 和 SER 领域,高质量大规模数据集是稀缺资源。HUME-VOCALBURST (36h, 1702 speakers, 4 countries) 和 MODULATE-SONATA (25 情感类, 专业录制) 分别满足了 vocal burst 和 acted emotion 的数据需求 [§2]
2. **跨文化设计有前瞻性**: HUME-VOCALBURST 覆盖 4 个国家和文化背景,A-VB Cross-Cultural 任务直接测试模型的跨文化泛化能力,这在当前全球化语音 AI 中很有价值 [§2.2, §3.3]
3. **评估体系完善**: 为每个数据集定义了清晰的任务和指标 (S_MTL, S_GEN, CCC, UAR),并通过竞赛建立了多方 baseline [§3]
4. **Hume AI 团队与 TADA 的连续性**: Alice Baird, Panagiotis Tzirakis, Chris Gagne, Alan Cowen 也是 [[论文笔记/TADA|TADA]] 的作者,TADA 的 TTS 系统很可能受益于这些情感数据基础设施的积累

**局限/疑问**:
1. MODULATE-SONATA 的 25-class UAR 0.90 (early fusion Wav2Vec2+HuBERT) 看似很高,但这是 speaker-independent split 在 23 名演员上的结果,说话人身份信息可能泄漏为情感线索 (每个演员有特定角色风格)
2. 缺少与其他主流 SER 数据集 (IEMOCAP, RAVDESS, ESD) 的系统对比分析,难以定位 HUME 数据集在情感识别领域的相对位置
3. 48 维 semantic space 标注框架的完整数据未在本次发布 (仅 9/10 维子集),限制了对细粒度混合情感的研究
4. Vocal burst 的 generation 评估 (S_GEN = 0.408 best) 数值偏低,说明 vocal burst 生成仍是高度困难的任务

## 可复用的 idea

1. **Seed imitation 数据收集范式**: 通过播放 seed 样本让众包工人模仿,获取大规模 in-the-wild 情感语音。相比实验室录制更经济、更自然; 相比完全自由表达更可控
2. **S_GEN = (1/FID + HEEP) / 2 评估框架**: 结合声学分布距离 (FID) 和人类情感感知 (HEEP) 评估情感音频生成,可推广到情感 TTS 的评估
3. **跨文化 speaker-independent partition 设计**: 确保 train/dev/test 在 speaker 和 culture 维度上独立,防止身份泄漏,适用于任何多说话人情感数据集
4. **MODULATE-SONATA 作为情感 TTS benchmark**: 25 类情感 + 专业录制质量,可作为 emotional TTS 系统输出的参考标准或 SER reward model 的训练数据
5. **48 维 semantic space 情感框架 (Cowen 2021)**: 比传统 Ekman 6 类或 Russell 的 valence-arousal 2 维更细粒度,适合未来的 fine-grained emotion modeling

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 数据集论文的 WHY 解释到位: 收集范式、标注框架、评估设计 |
> | 可信赖 | pass | 数字有出处 (Table 1-10, Eq. 1-4), 指标名正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率约 85% |
> | 可定位 | pass | KB 背景连接 EmotionControlinTTS + TADA 团队, 定位为数据基础设施 |
> | 不污染 | pass | 按指令不进行反向更新 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> - medium: datasets frontmatter 误列 SEED-TTS-Eval (已修正)
> - low: HUME-PROSODY 情感类数 10→9 不一致 (已修正)
> 详见 `_review/HUME-VOCALBURST-review.yml`

---

检索命中: [[ProsodyModeling]] | 过滤: [[EmotionControlinTTS]](pending-review), [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review), [[TTSEvaluation]](pending-review) | 未命中但可能相关: 无
