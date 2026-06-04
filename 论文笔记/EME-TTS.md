---
type: paper
tier: deep
title: "EME-TTS: Unlocking the Emphasis and Emotion Link in Speech Synthesis"
arxiv_id: "2507.12015"
source: "Sources/EME-TTS.pdf"
authors: [Haoxun Li, Leyuan Qu, Jiaxi Hu, Taihao Li]
year: 2025
venue: "Interspeech 2025 (submitted)"
tags: [TTS, emotion, emphasis, prosody, variance-adapter, weakly-supervised, non-autoregressive]
concepts: ["[[Emotion Control in TTS]]", "[[Prosody Modeling]]", "[[Duration Predictor]]", "[[F0 Modeling]]", "[[Non-autoregressive TTS]]", "[[Mel Spectrogram]]", "[[Self-Supervised Speech Representation]]", "[[Neural Vocoder]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[Emotion Control in TTS]]✓, [[Prosody Modeling]]✓, [[Duration Predictor]] [待确认], [[F0 Modeling]] [待确认], [[Non-autoregressive TTS]] [待确认], [[Mel Spectrogram]] [待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Emotion Control in TTS]], [[Prosody Modeling]], [[Duration Predictor]], [[F0 Modeling]], [[Non-autoregressive TTS]], [[Mel Spectrogram]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: EME-TTS 处于 **情感可控 TTS** 与 **重音可控 TTS** 的交叉地带。KB 中 [[Emotion Control in TTS]] 页面记录了情感建模从 emotion embedding (2021) → 多尺度层级建模 (MsEmoTTS, 2022) → DPO 对齐 (Emo-DPO, 2024) → ADV 维度解耦 (UDDETTS, 2025) 的演进线。EME-TTS 走的是另一条路线 — 不是从情感表示本身出发,而是从 **重音 (emphasis)** 这个韵律子维度切入,通过增强重音位置来间接提升情感表达力。这条路线在 KB 中尚无先例记录。

**已有认知**: [[Prosody Modeling]] 中记录了 FastSpeech 2 的 variance adaptor (duration + pitch + energy predictor) 作为显式韵律建模的标杆。EME-TTS 的基座架构 EmoSpeech 正是 FastSpeech 2 的情感扩展版,沿用 variance adaptor 范式。EME-TTS 在此基础上新增 **variance-based emphasis features** (pitch variance + duration variance),属于 variance adaptor 的功能扩展。

**创新判断**: KB 中已有工作均将 emotion 和 emphasis 视为独立的可控维度分别建模。EME-TTS 首次系统探索两者的交互关系 — emphasis 如何增强情感表现力,以及如何在不同情感条件下维持重音的感知清晰度。EPE block 中 Emphasis Adapter 的 attention weight 调制机制是全新设计。

## 速查

> [!summary] 速查
> - **一句话**: 首次探索重音与情感在 TTS 中的交互关系,通过弱监督重音伪标签 + EPE block 实现情感增强且重音稳定的语音合成
> - **路线**: Text → Phoneme Encoder (EPE blocks) → Variance Adapter (pitch/duration/energy + variance pitch/duration predictors) → Frame Decoder (EPE blocks) → Mel-spec → iSTFTNet vocoder
> - **指标**: 主观情感准确率 Mean 0.67 (vs EmoSpeech 0.58, CosyVoice2 0.48) [Table 3]; MOS 4.22 (vs EmoSpeech 4.14, w/o EPE 3.98) [Table 4]; 重音识别准确率 Mean 0.78 (vs w/o EPE 0.73) [Table 1]
> - **可借鉴**: (1) variance-based emphasis features (pitch/duration 相对偏差) 是轻量级重音建模方案; (2) Emphasis Adapter 的 attention weight additive modulation 可推广到任何需要局部增强的注意力场景; (3) LLM 预测重音位置 + TTS 执行分离的设计
> - **局限**: 仅在 ESD 数据集 (10 说话人, 5 情感, 1.2h/人) 上验证; 仅评估 5 种基础情感; 只有 11 名评估者; 未与最新 LLM-based TTS 做公平对比; 代码/模型未开源

## 核心问题

本文试图回答两个未被系统研究过的问题:

1. **如何利用重音提升情感语音的表现力?** 情感和重音在语音中天然耦合 — 重音通过调制韵律模式影响情感感知,情感状态自然决定哪些词被强调。但已有的情感 TTS 和重音控制 TTS 各自独立发展,忽略了这种交互。

2. **如何在不同情感条件下维持重音的感知清晰度和稳定性?** 当全局情感韵律很强烈时 (如 surprise 的句尾升调),局部重音可能被淹没或错位。需要一种机制让重音在情感干扰下仍可感知。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

EME-TTS 以 EmoSpeech [9] (FastSpeech 2 的情感扩展版) 为基座,由三个模块组成 [§2.1]:
1. **Phoneme Encoder**: 处理音素序列,编码语音学特征
2. **Variance Adapter**: 建模 duration/pitch/energy 的韵律变化,同时集成 variance-based emphasis features
3. **Frame Decoder**: 将帧级隐藏表示转换为 mel-spectrogram

关键条件信号: emotion label (情感嵌入) + emphasis pseudo-label (重音伪标签位置) [§2.1]。

### 关键设计选择

#### 1. 弱监督重音伪标签 (Weakly Supervised Emphasis Pseudo-Labeling)

**为什么不用人工标注**: 重音标注存在高度主观性 — 同一句话可能有多个合理的重音位置,大规模标注成本高且一致性差 [论文原文, §2.2]。

**为什么不用 CWT-based prominence score**: 先前工作 EE-TTS [20] 使用小波变换从 pitch/energy/duration 计算 prominence score。EME-TTS 认为 EmphaClass [22] 基于 SSL fine-tuning 的帧级分类器更准确 [论文原文, §2.2]。EmphaClass 利用预训练 SSL 语音模型做帧级分类,然后聚合为词级重音判定。

**具体做法**: 用 EmphaClass 对 ESD 数据集自动标注重音伪标签,获得词级 start/end 位置 [§2.2]。

[agent 解读] 这个设计选择的本质是: 将重音检测问题交给专门的 SSL-based 分类器离线处理,TTS 模型只需学习"在给定重音位置上如何合成",降低了 TTS 模型的建模难度。

#### 2. Variance-based Emphasis Features

**核心思路**: 重音体现为韵律参数相对于句子平均值的局部偏差 [论文原文, §2.3]。基于 [19] 的假设 — pitch 和 duration 是重音的主要指标,energy 影响较小 — 仅建模 pitch variance 和 duration variance [§2.3]:

Pitch Variance = WF0 − SF0 [Eq. 1]
Duration Variance = Wdur − Sdur [Eq. 2]

其中 WF0/Wdur 为重音区域的平均 pitch/duration,SF0/Sdur 为全句平均值。

**训练过程** [§2.3]:
- Variance Pitch Predictor 和 Variance Duration Predictor 预测重音区域内的 pitch/duration 偏差
- 偏差值 **仅应用于重音区域**,非重音区域设为零
- 总 pitch loss: LP = MSE(Ppre + PpreV, Ptar) [Eq. 3]
- Pitch variance loss: LVP = MSE(PpreV, PtarV) [Eq. 4]
- Duration 同理 [Eq. 5-6]
- 偏差值归一化到 [0, 2] 范围,防止极端值 [§2.3]

[agent 解读] 这种"相对偏差"建模比直接预测重音区域的绝对 pitch/duration 更合理 — 它解耦了重音信号与全局韵律基线,使得同样的 emphasis feature 可以在不同情感条件下产生一致的"突出感"。

#### 3. Emphasis Perception Enhancement (EPE) Block

**为什么需要 EPE**: 不同情感的全局韵律特征可能干扰局部重音感知。例如 surprise 的句尾升调会让听者误认为重音在句尾,而非实际指定位置 [论文原文, §3.2.1]。直接增强重音区域的 energy/pitch 可能引入合成伪影 [论文原文, §2.4]。

**EPE Block 结构** (替换原始 FFT block) [§2.4, Fig 1c]:
1. **Multi-Head Attention (MHA)**: 捕捉隐藏序列内的全局依赖
2. **Conditional Cross Attention (CCA)**: 用情感嵌入 c 作为 K/V 重新加权自注意力,使注意力分布根据情感条件调整 [Eq. 7-9]
3. **Emphasis Adapter (EA)**: 在注意力权重上对指定重音区域做加性调制 [Eq. 10]:
   Δw = strength · mask(start, end)
   wadjusted = w + Δw
4. **Conditional Layer Normalization (CLN)**: 借鉴 AdaSpeech4 [25],将情感上下文注入归一化过程

**两重作用** [论文原文, §2.4]:
(1) 确保重音词在高表现力语音中仍保持感知可区分性
(2) 通过注意力调制而非直接操作 energy/pitch 来增强重音,减少合成伪影

[agent 解读] EA 的设计极其简洁 — 本质上就是在 attention weight 上加一个 mask 化的常数偏移。strength 在实验中固定为 0.2 [§3.1],这意味着它不是通过学习获得的,而是一个手调的超参数。这种"硬编码式"的注意力增强能否泛化到更复杂的场景值得观察。

### 训练策略

- 数据集: ESD [23] 英语子集,10 说话人 × 5 情感 × 350 句/情感,每说话人约 1.2 小时 [§3.1]
- Vocoder: iSTFTNet [26],在 ESD 英语子集上训练 [§3.1]
- EA strength: 0.2 [§3.1]
- 训练硬件: 2 × A100 + 8 × RTX 4090 [§3.1]
- Batch size 64, 100k steps, Adam (lr=0.0001, β1=0.5, β2=0.9) [§3.1]
- 推理时: 使用 GPT-4 [29] 根据 emotion label + text 预测重音位置 [§3.2.2]

## 实验

### 重音感知稳定性

| 指标 | EME-TTS | w/o EPE | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Emphasis Accuracy (Mean) | 0.78 | 0.73 | ESD test | [Table 1] |
| Emphasis Accuracy (Neutral) | 0.80 | 0.77 | ESD test | [Table 1] |
| Emphasis Accuracy (Angry) | 0.82 | 0.75 | ESD test | [Table 1] |
| Emphasis Accuracy (Surprise) | 0.64 | 0.55 | ESD test | [Table 1] |

EPE 的最大提升出现在 surprise 情感 (0.55→0.64),这正是全局韵律最容易干扰局部重音的情感类别 [§3.2.1]。

### 情感准确率

**客观评估** (Emotion2vec-plus-large 识别) [Table 2]:

| 指标 | EME-TTS | w/o EPE | EmoSpeech | CosyVoice2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Emotion Accuracy (Mean) | 0.73 | 0.74 | 0.72 | 0.68 | ESD test | [Table 2] |
| Emotion Accuracy (Sad) | 0.61 | 0.60 | 0.54 | 0.52 | ESD test | [Table 2] |

**主观评估** (11 评估者) [Table 3]:

| 指标 | EME-TTS | w/o EPE | EmoSpeech | CosyVoice2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Subjective Emotion Acc (Mean) | 0.67 | 0.58 | 0.58 | 0.48 | ESD test | [Table 3] |
| Subjective Emotion Acc (Angry) | 0.75 | 0.57 | 0.48 | 0.07 | ESD test | [Table 3] |
| Subjective Emotion Acc (Sad) | 0.82 | 0.80 | 0.50 | 0.36 | ESD test | [Table 3] |

[agent 解读] 主观评估中 EME-TTS 的优势比客观评估更显著 (Mean: 0.67 vs 0.58, 远大于客观的 0.73 vs 0.74)。这可能意味着重音增强对人类的情感感知影响比对自动分类器更大。CosyVoice2 在 angry 上仅 0.07,说明其 neutral reference + emotion prompt 模式在该测试条件下效果极差,对比价值有限。

### 情感表现力偏好排名

EEPT 排名实验 (30 组样本, 每组 4 个模型输出) [§3.2.3, Fig 2]:
- EME-TTS 在独立句子和上下文句子评估中均获得最高排名
- 上下文条件下优势更大 — 周围语境为重音提供了语义基础,进一步增强情感表现力 [§3.2.3]

### 合成质量

| 指标 | EME-TTS | w/o EPE | EmoSpeech | Original | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS | 4.22±0.28 | 3.98±0.32 | 4.14±0.20 | 4.94±0.03 | ESD test | [Table 4] |
| NISQA | 3.76±0.60 | 3.66±0.68 | 3.71±0.74 | 4.17±0.57 | ESD test | [Table 4] |

EPE 将 MOS 从 3.98 提升到 4.22 (+0.24),甚至超过基座 EmoSpeech 的 4.14 [Table 4]。[agent 解读] 这说明 EPE 不仅改善了重音清晰度,还减少了重音控制引入的合成伪影,整体提升了合成质量。

## 局限性

1. **数据集规模小**: 仅在 ESD (10 说话人, 每人 1.2h) 上验证,缺乏大规模数据集的验证 [agent 解读]
2. **情感类别有限**: 仅覆盖 5 种基础情感 (angry, happy, sad, surprise, neutral),未涉及混合情感或细粒度情感 [§4]
3. **评估规模小**: 仅 11 名评估者,主观评估统计效力有限 [§3.2]
4. **基线选择不充分**: CosyVoice2 的对比条件 (neutral reference + emotion prompt) 不是其最佳模式,对比公平性存疑 [agent 解读]
5. **EA strength 固定为 0.2**: 没有探索可学习的 strength 或自适应调节机制 [agent 解读]
6. **对 surprise 的提升有限**: surprise 的重音准确率仅从 0.55 提升到 0.64,仍远低于其他情感 [Table 1]
7. **依赖 GPT-4 推理**: 推理时需调用外部 LLM 预测重音位置,增加延迟和成本 [§3.2.2]
8. **未开源**: 无代码/模型/数据公开 [agent 解读]

## 点评

**贡献**: EME-TTS 首次系统研究了 TTS 中情感与重音的交互关系,提出了两个有价值的研究问题。在方法层面,variance-based emphasis features 和 EPE block 设计简洁有效。

**方法论局限**: 整体技术方案偏传统 (FastSpeech 2 变体 + 手工特征),在 LLM-based TTS 已成主流的 2025 年显得有些过时。EPE 中 Emphasis Adapter 的加性注意力调制 (固定 strength=0.2) 缺乏自适应性。

**实验设计**: 评估较为全面 (4 个子任务),但数据集 (ESD) 和评估者规模 (11 人) 限制了结论的泛化性。与 CosyVoice2 的对比条件不公平,削弱了说服力。

**领域价值**: 指出了一个被忽视的研究方向 (emotion-emphasis interaction),对后续工作有启发。但该方向的实际影响力取决于能否在大规模场景和 LLM-based TTS 中验证。

## 可复用的 idea

1. **Variance-based emphasis features**: 用韵律参数的局部-全局偏差 (Eq. 1-2) 建模重音,简单且可解释,可嵌入任何有 variance adaptor 的 TTS 系统
2. **Attention weight additive modulation**: EA 的 Δw = strength · mask 设计极其轻量,可推广到任何需要在特定位置增强注意力的场景 (如关键词强调、停顿控制)
3. **LLM 预测重音 + TTS 执行**: 将重音位置预测与语音合成解耦,利用 LLM 的语义理解能力决定"哪里该强调",TTS 只负责"怎么强调"
4. **弱监督伪标签策略**: 用 SSL-based 重音检测器自动标注训练数据,避免昂贵的人工标注

---

> [!review] 审阅状态
> 待审阅。审阅报告见 `_review/EME-TTS-review.yml`。

---

检索命中: [[Emotion Control in TTS]], [[Prosody Modeling]], [[Duration Predictor]], [[F0 Modeling]], [[Non-autoregressive TTS]], [[Mel Spectrogram]] | 过滤: 无 | 未命中但可能相关: 无
