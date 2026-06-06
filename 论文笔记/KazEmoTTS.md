---
type: paper
tier: deep
title: "KazEmoTTS: A Dataset for Kazakh Emotional Text-to-Speech Synthesis"
arxiv_id: "2404.01033"
source: "Sources/KazEmoTTS.pdf"
authors: [Adal Abilbekov, Saida Mussakhojayeva, Rustem Yeshpanov, Huseyin Atakan Varol]
year: 2024
venue: "arXiv"
tags: [TTS, emotion, dataset, Kazakh, low-resource, GradTTS, diffusion]
concepts: ["[[EmotionControlinTTS]]", "[[Diffusion-basedTTS]]", "[[NeuralVocoder]]", "[[MelSpectrogram]]", "[[TTSEvaluation]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[NeuralVocoder]]✓, [[EmotionControlinTTS]], [[Diffusion-basedTTS]], [[TTSEvaluation]], [[GlobalStyleTokens]], [[MelSpectrogram]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[NeuralVocoder]]✓, [[EmotionControlinTTS]], [[Diffusion-basedTTS]], [[TTSEvaluation]], [[GlobalStyleTokens]], [[MelSpectrogram]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文是一篇**数据集贡献**论文,核心贡献是 KazEmoTTS 数据集本身,而非 TTS 模型创新。从 KB 背景看:

- **情感 TTS 领域** ([[EmotionControlinTTS]]): 情感 TTS 的演进从 emotion embedding (2021) 到 DPO/RLHF 对齐 (2024) 再到 training-free steering (2025),方法日趋复杂。但这些方法几乎全部基于英语或中文高资源语料。本文填补的是**低资源语言情感 TTS 数据集**这一空白 -- Kazakh 此前没有任何公开的情感语音数据集。
- **扩散 TTS** ([[Diffusion-basedTTS]]): 本文的 baseline 模型直接采用 Grad-TTS (Popov et al., 2021) + hard emotion label 的方案,与 EmoDiff (Guo et al., 2022) 的 classifier guidance 方案在同一 GradTTS 基础上分叉。本文走的是最简单的 hard label 路线,不涉及情感强度控制。
- **声码器** ([[NeuralVocoder]]✓): 使用 HiFi-GAN 作为多说话人声码器,训练时不提供情感标签。这是 2020-2023 时期的标准选择。
- **评估** ([[TTSEvaluation]]): 使用 MCD (客观) + MOS (主观) + 情感识别准确率三维评估。MOS 范围 3.51-3.57 在当时的情感 TTS 中属于中等水平,低于 EmoDiff (4.01) 和 EmoMix (3.92) [§5],但考虑到低资源语言的特殊性,这是合理的 baseline。

**创新判断**: 本文的创新不在模型,而在数据资源 -- 首个公开的哈萨克语情感语音数据集,CC-BY-4.0 许可。模型部分是标准 GradTTS pipeline 的直接应用,无方法创新。

## 速查

> [!summary] 速查
> - **一句话**: 首个哈萨克语情感 TTS 数据集 (54,760 条, 74.85h, 6 种情感, 3 位叙述者),附带 GradTTS baseline 模型
> - **路线**: 文本 → GradTTS (hard emotion label) → 80-dim log-mel → HiFi-GAN vocoder → 波形
> - **指标**: MCD 6.02-7.67 / MOS 3.51-3.57 / 情感识别总准确率 37% [Table 3, Table 4]
> - **可借鉴**: (1) Telegram bot 用于录音收集和评估的低成本数据采集流程; (2) Whisper ASR 做录音质量筛选 (CER 检测 → 人工复核 → 淘汰),从 84,714 条筛到 54,760 条
> - **局限**: MOS 仅 3.5 级别; 情感识别准确率极低 (angry 仅 22%); 无情感强度控制; 模型无创新; 未与当时的情感 TTS SOTA 做公平对比

## 核心问题

本文要解决的核心问题是: **低资源语言 (哈萨克语) 缺乏公开的情感语音数据集,阻碍了情感 TTS 研究在非高资源语言上的推进** [§1]。

现有情感语音数据集几乎全部覆盖高资源语言 (中文、英文、法语) [§1],如 IEMOCAP (英)、EMOVO (意)、EMOVIE (中)。Kazakh 作为突厥语系低资源语言,此前没有任何可用于情感 TTS 训练的公开数据集。

本文的贡献分两部分:
1. 构建并开源 KazEmoTTS 数据集
2. 训练一个 baseline 情感 TTS 模型作为数据集可用性的验证

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 数据集构建

#### 文本收集 [§3.1]

文本来源包括哈萨克语 Wikipedia (科学、计算机、历史、国际文章)、知名哈萨克语媒体新闻、公共领域书籍、童话和短语手册。所有收集的文本按句子级别切分。

数据集包含 8,794 个唯一句子和 86,496 个唯一词,平均句子长度 10.83 个词 [§3.4]。

#### 录制流程 [§3.2]

- **叙述者**: 3 位专业叙述者 (1 女 F1 + 2 男 M1, M2)
- **录制环境**: 个人家庭录音室或机构设施,要求安静室内环境
- **情感类别**: 6 种 -- neutral, angry, happy, sad, scared, surprised
  - 情感选择基于 Ekman (1992) 的六基本情感理论,去掉 "disgust",加入 "neutral" [论文原文]
  - [agent 解读] 去掉 disgust 可能因为该情感在朗读文本中难以自然表达
- **分配方式**: 每句话配对一种情感,确保各情感均匀分布 [§3.2]
- **录制工具**: Telegram bot (见 [Fig 1a])
- **采样率**: 44.1 kHz / 16-bit 或 48 kHz / 24-bit

#### 质量筛选 [§3.3]

使用定制版 Whisper 多语言 ASR 系统进行音频-文本对齐验证:
1. ASR 对录音生成转写
2. 转写与原始文本对比计算 CER
3. CER 异常的录音由人工审核团队复核
4. 淘汰含误读或显著背景噪声的录音

结果: 从初始 84,714 条录音筛选至 54,760 条 [§3.4]。淘汰率约 35%。

#### 数据集规格 [§3.4]

| 叙述者 | 录音数 | 时长 (h) | 平均段落长 (s) |
| --- | --- | --- | --- |
| F1 (女) | 24,656 | 34.23 | 5.0 |
| M1 (男) | 19,802 | 26.51 | 4.8 |
| M2 (男) | 10,302 | 14.11 | 4.9 |
| **总计** | **54,760** | **74.85** | — |

[Table 2]

所有音频统一下采样至 22.05 kHz, 16-bit WAV 格式。经过静音移除和幅度归一化 (除以最大绝对值) 预处理。文件名格式: `narratorID_emotion_utteranceID` [§3.4]。

### 整体架构

TTS 模型基于 GradTTS (Popov et al., 2021) + hard emotion label [§4.1],与 EmoDiff (Guo et al., 2022a) 和 EmoMix (Tang et al., 2023) 采用相同的 GradTTS 基础。

```
文本 + emotion label → GradTTS encoder → diffusion decoder → 80-dim log-mel → HiFi-GAN → 波形
```

### 关键设计选择

1. **Hard emotion label**: 使用离散情感标签作为条件输入,不做情感强度控制 [§4.1]。
   - WHY: [agent 解读] 这是情感 TTS 中最简单直接的方案 (与 EmoDiff 的 soft-label guidance 或 EmoMix 的 SER embedding 方案相比),适合作为新数据集的 baseline。论文目的是验证数据集可用性,而非推进模型。

2. **去除 ground truth duration 依赖**: 调整后的 GradTTS 不依赖 GT duration 数据 [§4.1]。
   - WHY: [agent 解读] 原版 GradTTS 可使用 GT duration 进行训练,去除该依赖使模型更适合实际推理场景。

3. **HiFi-GAN 多说话人声码器**: 在 KazEmoTTS 上训练为多说话人声码器,但**不提供情感标签** [§4.1]。
   - WHY: [agent 解读] 将情感建模的职责完全交给声学模型 (GradTTS),声码器仅负责波形合成,这是标准的职责分离设计。
   - 声码器训练 1.72M 步 [§4.1]

4. **Classifier guidance γ=100**: 推理时 guidance level 设为 100 [§4.1]。
   - [agent 解读] 这是一个相当高的 guidance 值,可能导致生成语音过于依赖条件信号而牺牲多样性,但对于情感 TTS 来说高 guidance 有助于更清晰的情感表达。

### 训练策略

- **优化器**: Adam, 学习率 10^-4 [§4.1]
- **训练步数**: 3.7M 步 (GradTTS) + 1.72M 步 (HiFi-GAN) [§4.1]
- **硬件**: 1x NVIDIA DGX A100 GPU [§4.1]
- **EMA**: 训练中使用 exponential moving average 权重平滑 [§4.1]
- **输出**: 80-dim log mel-filter bank features [§4.1]
- **采样率**: 22,050 Hz [§4.1]

## 实验

### 客观评估: MCD

使用 mel-cepstral distortion (MCD) 衡量合成语音与 GT 的 MFCC 差异,DTW 对齐消除长度差异影响 [§4.2]。

| 叙述者 | neutral | angry | happy | sad | scared | surprised | 平均 MCD | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| F1 | 5.72 | 6.08 | 6.20 | 5.82 | 5.97 | 6.34 | **6.02** | [Table 3] |
| M1 | 7.63 | 7.98 | 7.88 | 7.19 | 7.68 | 7.65 | **7.67** | [Table 3] |
| M2 | 6.85 | 7.56 | 7.57 | 6.83 | 7.13 | 7.54 | **7.24** | [Table 3] |

**发现**: 女性叙述者 (F1) 合成质量最优; 低音调情感 (neutral, sad, scared) 的 MCD 低于高音调情感 (angry, happy, surprised) [§5]。[论文原文] 作者认为低音调情感与 GT 更相似,高音调情感的合成偏差更大。

### 主观评估: MOS

64 位参与者通过 Telegram bot 进行评估 [§4.3]。每位参与者评价 36 个样本 (18 GT + 18 合成)。

| 叙述者 | GT MOS | 合成 MOS | 出处 |
| --- | --- | --- | --- |
| F1 | 3.94 | 3.55 | [Table 3] |
| M1 | 3.95 | 3.51 | [Table 3] |
| M2 | 4.22 | **3.57** | [Table 3] |

**发现**: MOS 变化范围很小 (3.51-3.57),且与数据量不成正比 -- M2 数据最少 (14.11h) 但 MOS 最高 [§5]。[论文原文] 作者强调这说明 MOS 与数据量缺乏相关性。

**与已有工作对比** [§5]:
- KazakhTTS2 (Mussakhojayeva et al., 2022, 同组非情感 TTS): GT 4.18-4.73, 合成 3.95-4.53 -- 显著高于本文
- EmoDiff (Guo et al., 2022a): MOS 4.01 -- 高于本文
- EmoMix (Tang et al., 2023): MOS 3.92 -- 高于本文
- Zhou et al. (2022a, "surprised" 情感): MOS 3.45 -- 与本文接近

### 情感识别准确率

| 情感 | GT 准确率 | 合成准确率 | 总准确率 | 出处 |
| --- | --- | --- | --- | --- |
| neutral | 0.64 | 0.65 | **0.65** | [Table 4] |
| angry | 0.37 | 0.07 | **0.22** | [Table 4] |
| happy | 0.58 | 0.36 | **0.47** | [Table 4] |
| sad | 0.33 | 0.29 | **0.31** | [Table 4] |
| scared | 0.32 | 0.33 | **0.33** | [Table 4] |
| surprised | 0.31 | 0.21 | **0.26** | [Table 4] |
| **总计** | 0.43 | 0.32 | **0.37** | [Table 4] |

**发现** [§5]:
- Neutral 最容易识别 (65%)
- Angry 最难识别 (22%),常被误认为 sad 或 scared [Table 5]
- Happy 在参与者选择中频率最高 (约 45%),不论实际情感 [Table 5]
- [论文原文] 作者承认存在认知挑战 -- 文本内容本身的情感与叙述者表达的情感可能冲突 (如悲伤内容配欢快语调),可能影响参与者判断

## 局限性

1. **MOS 偏低**: 合成 MOS 3.51-3.57 显著低于同期英语/中文情感 TTS 系统 (EmoDiff 4.01, EmoMix 3.92) [§5],甚至低于同组的非情感哈萨克语 TTS (KazakhTTS2 合成 3.95-4.53) [§5]
2. **情感识别率极低**: 总体仅 37%,合成语音更低 (32%) [Table 4]。这表明模型对情感的表达能力非常有限
3. **文本-情感冲突**: 文本内容的固有情感与叙述者被要求表达的情感之间的冲突未得到解决 [§5]
4. **模型无创新**: TTS 模型直接使用 GradTTS + hard label,无情感建模的方法论贡献
5. **未与同规模数据集上的其他情感 TTS 方法对比**: 仅提供单一 baseline,未在同一数据集上对比 EmoDiff、EmoMix 等方法
6. **情感分配方式简单**: 每句随机配对一种情感,未考虑文本语义与情感的匹配性 [§3.2]
7. **评估规模有限**: 仅 64 位参与者,统计可靠性存疑 [§4.3]

## 点评

**贡献定位**: 本文的价值完全在数据资源层面 -- 为哈萨克语这一低资源语言提供了首个公开的情感语音数据集 (74.85h, 6 情感, 3 叙述者, CC-BY-4.0)。这填补了一个真实的空白,对低资源语言 TTS 研究有实际意义。

**数据质量工程值得借鉴**: 使用 Telegram bot 做数据采集 + Whisper ASR 做质量筛选 (35% 淘汰率) 的流程设计,对于其他低资源语言数据集构建有参考价值。成本低、可复制性强。

**模型部分薄弱**: GradTTS + hard label 是最直接的 baseline,没有对情感建模做任何改进。考虑到同期已有 EmoDiff (soft-label guidance) 和 EmoMix (SER embedding) 等更先进的方案,仅提供最简单的 baseline 限制了论文的学术贡献。

**情感识别结果暴露根本问题**: 37% 的总识别率 (6 类情感,随机 16.7%) 虽然高于随机,但远非令人满意。特别是 angry (22%) 的极低识别率提示: (1) 叙述者的情感表达可能不够区分度; (2) 文本内容与情感的随机配对可能造成感知混淆; (3) Kazakh 语的情感韵律模式可能与评估者预期不匹配。

**与知识库已有工作的对比**: 从 [[EmotionControlinTTS]] 的演进线看,本文处于情感 TTS 研究的早期阶段 (emotion embedding + hard label),远早于当前的 steering/DPO/reward-guided 方向。其价值不在推进前沿,而在拓展语言覆盖面。

## 可复用的 idea

1. **Telegram bot 数据采集流水线**: 用即时通讯应用 bot 同时实现录音采集 ([Fig 1a]) 和评估收集 ([Fig 1b]),避免了专用标注平台的开发成本。对低成本数据集构建场景直接可用。

2. **ASR 质量门控**: 用 Whisper ASR 生成转写 → 计算 CER → 筛选异常 → 人工复核 → 淘汰的质量管控流程 [§3.3]。35% 淘汰率表明该流程确实有效过滤了低质量录音。可迁移到任何语音数据集构建场景。

3. **不依赖 GT duration 的 GradTTS 训练**: 文中提到移除了对 ground truth duration 数据的依赖 [§4.1]。这一修改虽然技术细节未详述,但对实际部署场景有参考意义。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 数据集构建+模型架构 WHY 覆盖充分,数据集设计选择可更深入 |
> | 可信赖 | pass | 数字标注覆盖率 >90%,指标使用正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注一致 |
> | 可定位 | pass | 四维 KB 定位+创新判断明确 |
> | 不污染 | pass | 无新建页,仅 append 操作,风险低 |
> 
> Issues: 3 (high: 0, medium: 0, low: 3)
> 详见 `_review/KazEmoTTS-review.yml`
