---
type: paper
tier: deep
title: "The AudioMOS Challenge 2025"
arxiv_id: "2509.01336"
source: "Sources/AudioMOS_Challenge_2025.pdf"
authors: [Wen-Chin Huang, Hui Wang, Cheng Liu, Yi-Chiao Wu, Andros Tjandra, Wei-Ning Hsu, Erica Cooper, Yong Qin, Tomoki Toda]
year: 2025
venue: "arXiv (challenge summary paper)"
tags: [evaluation, MOS-prediction, audio-quality, text-to-music, TTS-evaluation, speech-quality, SSL, challenge, benchmark]
concepts: ["[[TTS Evaluation]]", "[[Self-Supervised Speech Representation]]", "[[Audio-Language Pretraining]]", "[[SVS Evaluation Metrics]]"]
models: ["[[WavLM]]", "[[wav2vec 2.0]]", "[[HuBERT]]", "[[Whisper]]", "[[EnCodec]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页 [均待确认]: [[TTS Evaluation]], [[Self-Supervised Speech Representation]], [[Audio-Language Pretraining]], [[SVS Evaluation Metrics]], [[WavLM]], [[Audio Understanding]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: AudioMOS Challenge 2025 是 VoiceMOS Challenge (VMC) 系列的扩展版,将自动 MOS 预测从 speech-only 拓展到 music 和 general audio。在 [[TTS Evaluation]] 概念页中记录的评估演进线里,Predicted MOS (DNSMOS, UTMOS, SSL-MOS) 代表自动化评估的重要一环。本挑战直接推进这一方向,特别是将 MOS 预测从 TTS 扩展到 TTM/TTA 等新模态。
>
> **已有认知**:
> - [[TTS Evaluation]] 页记录了 Predicted MOS 的三大局限: 领域不匹配、缺乏 uncertainty estimation、跨域泛化差 — 本挑战的 Track 2/3 正面对这些问题 [待确认]
> - [[Self-Supervised Speech Representation]] 页记录了 WavLM/HuBERT/wav2vec 2.0 等 SSL 模型的预训练范式和层级信息分离特性 — 这些正是挑战参赛者的核心工具 [待确认]
> - [[Audio-Language Pretraining]] 页记录了 CLAP 模型的对比学习范式 — CLAP 是本挑战 Track 1 baseline 的基础 [待确认]
> - [[WavLM]] 页记录了 WavLM 的 masked speech denoising 和 full-stack 特性 — WavLM 是 Track 2 baseline 的特征提取器 [待确认]
>
> **创新判断**: 本文不是方法创新论文,而是 challenge summary paper。其核心价值在于: (1) 首次将自动 MOS 预测拓展到 text-to-music 和多模态音频; (2) 提供系统性的参赛方案对比和技术趋势分析; (3) 揭示当前 audio quality prediction 的关键瓶颈 (textual alignment 难于 musical quality、16 kHz 在混合评估中最难预测)。
>
> 检索命中: [[TTS Evaluation]], [[Self-Supervised Speech Representation]], [[Audio-Language Pretraining]], [[SVS Evaluation Metrics]], [[WavLM]], [[Audio Understanding]] | 过滤: 全部 pending-review | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个涵盖语音/音乐/通用音频的自动主观质量预测挑战赛,三个赛道分别聚焦 TTM MOS、Audiobox Aesthetics 四轴、多采样率语音 MOS,24 支队伍参赛且普遍超越 baseline
> - **路线**: 参赛者典型路线 = SSL 特征 (CLAP/WavLM/MuQ/BEATs) → 回归头 (MLP/KAN/Mamba2) → 多模型集成 (ridge/voting); 评估指标为 system-level SRCC
> - **指标**: Track 1 冠军 T09 系统级 SRCC 超 baseline 20.8% (musical quality) / 30.4% (textual alignment) [§IV-B]; Track 3 冠军 T17 SRCC=0.955 vs baseline B03=0.749 [Fig 3]
> - **可借鉴**: (1) 模型集成对 MOS 预测始终有效 — 三个赛道冠军全部使用集成 [§V-A3]; (2) 离散采样率 ID 作为额外输入特征可显著改善跨采样率预测 [§V-A3]; (3) 扩大训练数据未必有效 — Track 2 多数队伍仅用提供数据就超越了用 500h in-house 数据训练的 baseline [§IV-C]
> - **局限**: (1) 参赛队伍多集中单赛道,跨赛道通用方案缺乏; (2) Track 2 训练集含 YouTube 数据导致再分发困难; (3) 所有赛道仅考虑 sample-level/system-level 预测,未涉及 distribution-level 评估

## 核心问题

本文要回答的核心问题: **自动 MOS 预测从语音扩展到音乐和通用音频后,现有方法能否胜任?哪些技术选择最有效?** 具体分解为三个子问题:

1. **TTM 的 MOS 和 textual alignment 能否被自动预测?** (Track 1) — 此前几乎不存在 text-to-music 的 MOS 预测数据集和方法
2. **统一的多维音频质量评估是否可行?** (Track 2) — Audiobox Aesthetics 定义的 PQ/PC/CE/CU 四轴覆盖语音/音乐/音效三种模态
3. **混合采样率条件下的 MOS 预测有何新挑战?** (Track 3) — 同一评估中包含 16/24/48 kHz 样本时,听众和预测器的行为如何变化

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构: 三赛道设计

AudioMOS Challenge 2025 (AMC) 是 VoiceMOS Challenge 系列的扩展版,从 speech-only 拓展到覆盖 speech/music/general audio 三种音频类型 [§I]。

| 赛道 | 预测目标 | 数据类型 | 训练/开发/测试样本数 | 评估维度 |
|------|----------|----------|---------------------|----------|
| Track 1 | MOS 预测 | TTM | 1923/412/413 | Overall Quality + Textual Alignment |
| Track 2 | 四轴质量预测 | TTS+TTA+TTM | 2700 (natural) / N/A / 3060 (synthetic) | PQ, PC, CE, CU |
| Track 3 | MOS 预测 | 多采样率合成语音 | 400/400/400 | Single MOS |

[Table I]

### Track 1: Text-to-Music MOS 预测

**数据**: MusicEval 数据集,2748 个单声道音频片段 (16.62 小时),来自 21 个 TTM/TTA 系统的 31 个模型,384 个文本 prompt,14 名专业音乐评审 (2 名教师 + 12 名高级学生) 打分 [§II-A]。每个片段沿 overall musical quality 和 textual alignment 两个维度评分 (5-point Likert scale)。

**Baseline B01**: 基于预训练 CLAP 模型 (HTSAT audio encoder + RoBERTa text encoder),两个独立 3-layer MLP 回归头分别预测 musical quality 和 textual alignment。L1 loss,lr=0.0005,batch=64 [§III]。

**为什么选 CLAP 作为 baseline**: CLAP 的 audio-text joint embedding space 天然适配 text-to-music 场景,因为需要同时理解音频质量和文本对齐 [agent 解读]。

### Track 2: Audiobox Aesthetics 四轴预测

**数据**: 训练集为 AES-Natural 数据集 (950 speech + 1000 music + 1000 sound,自然样本),158 名经筛选的标注员 (PQ/PC 要求 Pearson r > 0.7),每样本 10 名标注 [§II-B]。测试集为合成样本 (TTS + TTA + TTM),包含 prompt-based TTS 和 reference-based TTS 两种类型,共 3060 样本 [§II-B]。

**四轴定义** [§II-B]:
- **PQ (Production Quality)**: 录音质量 — 较客观
- **PC (Production Complexity)**: 音频场景复杂度 — 较客观
- **CE (Content Enjoyment)**: 聆听体验 — 较主观
- **CU (Content Usefulness)**: 内容创作有用性 — 较主观

**Baseline B02**: 预训练 WavLM (12 层 Transformer,768 维) + learnable weighted sum + 四个独立 5-layer MLP 回归头。MAE + MSE 联合损失。所有音频降采样至 16 kHz。值得注意的是,baseline 训练于 500h in-house Meta 数据,而非仅 AES-Natural [§III]。

**关键设计**: 训练集全为自然样本,测试集全为合成样本,构成 zero-shot domain transfer 场景 [论文原文, §II-B]。这测试的是模型从自然音频质量分布泛化到合成音频质量分布的能力 [agent 解读]。

### Track 3: 多采样率语音 MOS 预测

**数据**: 四组独立听力测试 (16 kHz / 24 kHz / 48 kHz / 混合),包含 LibriTTS-R 和 HiFi-CAPTAIN 自然语音、非神经和神经 vocoded 语音、TTS 语音、AudioSR 超分辨率语音 [§II-C]。训练用 Part 1 (16/24/48 kHz 分别测试),测试用混合采样率测试的 Part 2 [§II-C]。

**Baseline B03**: 预训练 SSL-MOS 模型 fine-tune 于三组训练数据合并,50 epochs。所有音频降至 16 kHz (丢失高频信息) [§III]。

**挑战核心**: 训练时听众在同采样率条件下评分,测试时听众在混合采样率条件下评分 — 听众的内部参考系发生变化 [agent 解读]。同一段 16 kHz 语音在"全是 16 kHz"的测试中可能得分较高,但在混有 48 kHz 的测试中得分可能下降 [论文原文, §IV-D]。

### 关键设计选择

**1. 评估指标 = system-level SRCC**

选择 system-level Spearman rank correlation 而非 utterance-level 指标,理由是"评估音频生成系统时,我们主要关心系统间的排名" [论文原文, §IV-A]。这延续了 VoiceMOS Challenge 的传统。

**2. 公平性规则: 限制训练数据为公开数据集**

从 VMC 2024 起规定只能使用公开数据集训练,或在赛后开源内部数据/模型 [§II-D]。这促进了可复现性,但也限制了数据规模 [agent 解读]。

**3. Track 2 的"自然→合成"域迁移设计**

训练集全为自然样本,测试集全为合成样本 [§II-B]。这个设计揭示了一个反直觉结论: 用 500h in-house 数据训练的 baseline 被仅用 AES-Natural (~3K 样本) 训练的参赛系统超越,说明 MOS 预测中数据质量/匹配度比数据规模更重要 [§IV-C]。

### 训练策略 (参赛者共性总结)

**SSL 特征选择** [§V-A2]:
- 最常用: CLAP, WavLM, wav2vec 2.0 (baseline 已使用)
- 通用音频编码器: Qwen-Audio, BEATs, M2D, EnCodec, Dasheng
- 音乐特定: MERT, MuQ (Track 1)
- 文本: BERT, RoBERTa, Qwen-3.0 (Track 1 textual alignment)
- 其他语音 SSL: Whisper, HuBERT, MMS, EAT

**损失函数创新** [§V-A3]: 参赛者探索了 triplet loss、Huber loss、Sinkhorn optimal transport loss、smoothed cross-entropy、rank-consistent ordinal regression loss、pairwise adaptive margin ranking loss 等多种损失函数。

**模型集成的关键作用**: 24 队中 8 队使用集成,且三个赛道冠军全部使用集成策略 [§V-A3]。

**采样率 ID 输入**: Track 3 中 7 队有 4 队使用离散采样率 ID 作为额外输入 [§V-A3]。

## 实验

### Track 1 关键结果

| 队伍 | Musical Quality SRCC | Textual Alignment SRCC | 方法核心 | 出处 |
|------|---------------------|----------------------|----------|------|
| T09 (冠军) | ~0.97 | ~0.93 | MuQ + RoBERTa + 9-model stacking ensemble | [§V-B1, Fig 1] |
| T02 | ~0.96 | ~0.92 | — | [Fig 1] |
| T22 | ~0.96 | ~0.91 | — | [Fig 1] |
| B01 (baseline) | ~0.78 | ~0.70 | CLAP + 2-head MLP | [§IV-B, Fig 1] |

T09 超越 baseline 20.8% (musical quality) 和 30.4% (textual alignment) [§IV-B]。

**关键发现**: musical quality 和 textual alignment 表现高度相关但不完全线性 — 共享特征编码器贡献一致性,任务特定输出头允许部分解耦 [§IV-B]。textual alignment SRCC 一致低于 musical quality (如 T10: 0.965 vs 0.902),可能反映跨模态理解比单模态质量评估更难建模 [agent 解读]。

### Track 2 关键结果

| 指标 | B02 排名 | 冠军 (T12) | 出处 |
|------|----------|-----------|------|
| PQ system-level SRCC | 9/10 | 2nd | [§IV-C, Fig 2] |
| PC system-level SRCC | 7/10 | — | [§IV-C] |
| CE system-level SRCC | 9/10 | 1st | [§IV-C] |
| CU system-level SRCC | 10/10 | 1st | [§IV-C] |

**关键发现**: 没有一个队伍在所有四轴上排名第一 [§IV-C]。这说明 PQ/PC (客观) 和 CE/CU (主观) 可能需要不同的建模策略 [agent 解读]。

### Track 3 关键结果

| 队伍 | System-level SRCC | 出处 |
|------|-------------------|------|
| T17 (冠军) | 0.955 | [Fig 3] |
| T19 | 0.926 | [Fig 3] |
| T13 | 0.917 | [Fig 3] |
| T11 | 0.914 | [Fig 3] |
| B03 (baseline) | 0.749 | [Fig 3] |

**最难预测的条件** [§IV-D]:
1. 自然语音下采样后 AudioSR 超分辨率到 24 kHz — 8/8 预测器列为 top-5 最难
2. 16 kHz 神经 vocoder 语音 — 7/8 预测器
3. 16 kHz 自然语音 — 7/8 预测器

**方向性偏差**: 对两个最难的 16 kHz 条件,所有预测器一致低估其排名 (预测的质量比实际更差); 对 24 kHz 超分辨率自然语音,7/8 预测器高估其排名 [§IV-D]。这意味着预测器倾向于"惩罚"低采样率而"奖励"超分辨率,但人类听众的判断更为复合 [agent 解读]。

### 冠军方案详解

**T09 (Track 1 冠军)** [§V-B1]:
- 特征: MuQ (音乐 SSL) + RoBERTa (文本)
- Musical quality: MuQ → Transformer → attention pooling → FC
- Textual alignment: MuQ + RoBERTa → cross-attention Transformer → attention pooling → FC
- 集成: 5 个 smoothed CE loss 模型 + 2 个 ordinal regression 模型 + 2 个改良模型 → ridge regression 聚合
- 13/16 metrics 排名第一

**T12 (Track 2 冠军)** [§V-B2]:
- 模型 1: WavLM baseline 改进版 (MLP → GR-KAN)
- 模型 2: VERSA 工具计算 28 个非侵入式指标 → XGBoost 回归
- 最终: 4 个 KAN 模型 + 1 个 VERSA 模型集成
- 额外数据: PAM + BVCC 数据集的 noisy student 半监督学习
- 17/32 metrics 排名第一

**T17 (Track 3 冠军)** [§V-B3]:
- 四路输入: SSL 特征 + 采样率 ID + Mel spectrogram + MFCC
- 多尺度卷积块处理不同采样率的 Mel spectrogram
- 四流 → BiLSTM → FC
- 损失: MAE + rank loss + correlation loss
- 仅使用提供的训练/开发集 (无额外数据)
- 3 模型集成: 完整训练集 base / 部分训练集 base / 去掉 MFCC 完整训练集
- 5/8 metrics 排名第一

## 局限性

1. **跨赛道参与度低**: 24 队中仅 2 队参加多个赛道,多数因时间和算力限制只选一个赛道 [§III]。这限制了对跨模态统一方案的探索。

2. **Track 2 数据获取困难**: AES-Natural 的部分训练数据基于 YouTube 视频,因 YouTube 政策无法再分发,参赛者需自行下载,是最常见的负面反馈 [§V-C]。

3. **Track 2 domain gap**: 训练于自然样本,测试于合成样本,虽然揭示了泛化能力,但也意味着结果可能高估了"完全匹配"场景下的性能上限 [agent 解读]。

4. **Track 3 baseline 局限**: SSL-MOS 仅接受 16 kHz 输入,高频信息完全丢失 [§III]。这使得 baseline 性能极低 (SRCC=0.749),降低了其作为参考的意义 [agent 解读]。

5. **缺乏 distribution-level 评估**: 所有赛道仅关注 sample-level 和 system-level 预测,未涉及 TTSDS/TTSDS2 式的分布级评估方法。

6. **主观标注一致性差异**: Track 1 使用专业音乐评审 (14 人),Track 2 使用经培训的标注员 (158 人),Track 3 仅 20 名听众 — 不同标注质量和数量可能影响跨赛道比较 [agent 解读]。

## 点评

**优势**:
- 开创性地将自动 MOS 预测从 speech 拓展到 music 和 general audio,填补了重要空白。此前 TTM 评估主要依赖 FAD 和 CLAP Score,而这些指标已被证明与人类感知相关性差 [§I]
- 赛制设计巧妙: Track 2 的"自然训练→合成测试"揭示了质量预测中 domain adaptation 的关键瓶颈; Track 3 的"分离训练→混合测试"揭示了采样率上下文效应
- 从 24 队的参赛方案中提炼的技术趋势分析 (SSL 选择、损失函数、集成策略) 对后续研究有直接参考价值

**不足**:
- 作为 challenge summary paper,缺乏对方法差异的深度分析。例如 T09 为何比 T02/T22 好? 是 MuQ 的贡献还是 stacking ensemble 的贡献? 未见 ablation
- 各赛道间缺乏方法论层面的对比讨论。三个赛道本质上是同一任务 (quality prediction) 在不同 domain 的实例化,但论文未深入分析 domain-specific 和 domain-agnostic 的技术选择
- 未讨论与 LLM-as-Judge 方向的关系。TTS Evaluation 领域正在经历从 scalar prediction 到 LLM-based reasoning 的转变,但本挑战仍聚焦于传统回归范式

**定位**: 本文是音频质量自动评估领域的重要里程碑,但更多是"拓宽 scope"而非"深化方法"。其主要贡献是数据集 + benchmark + 社区推动,而非方法创新。

## 可复用的 idea

1. **采样率 ID 作为显式输入**: 在处理多采样率音频任务时,将采样率编码为离散 ID 加入特征,可显著改善跨采样率泛化 [§V-A3]。这个 trick 可直接用于任何需要处理多采样率输入的语音模型。

2. **Stacking ensemble + ridge regression**: T09 的 9 模型 stacking 方案 (不同 loss + 不同 seed + 不同架构变体 → ridge regression 聚合) 是一个通用的 MOS 预测增强方案,适用于任何语音/音频质量评估任务 [§V-B1]。

3. **VERSA 多指标 → XGBoost 的 meta-learning 路线**: T12 将 28 个非侵入式指标作为 meta-features 用 XGBoost 回归,绕过了端到端训练的数据需求,特别适合标注数据稀缺的场景 [§V-B2]。

4. **自然样本训练→合成样本测试的 benchmark 设计**: Track 2 的设计范式可用于评估任何"from natural to synthetic"的 domain transfer 能力,可推广到 TTS/voice conversion 等场景的 MOS predictor 开发 [§II-B]。

5. **Multi-scale convolution 适配多采样率 Mel spectrogram**: T17 的方案可迁移到需要处理不同采样率输入的任何频谱分析任务 [§V-B3]。

---

> [!review] 审阅 (2026-06-04, agent-v2)
> **结论**: pass-with-fixes (0 high, 1 medium, 2 low)
> - (medium) fact-inference-mixing: Track 1 "跨模态理解更难"推断已补标来源 [agent 解读] — 已修正
> - (low) traceability-gap: Track 1 SRCC 数值为从 Fig 1 估计的近似值
> - (low) template-compliance: frontmatter tasks/datasets 为空
> 详见 `_review/AudioMOS Challenge 2025-review.yml`

---

检索命中: [[TTS Evaluation]](pending-review), [[Self-Supervised Speech Representation]](pending-review), [[Audio-Language Pretraining]](pending-review), [[SVS Evaluation Metrics]](pending-review), [[WavLM]](pending-review), [[Audio Understanding]](pending-review) | 过滤: 全部 pending-review | 未命中但可能相关: 无
