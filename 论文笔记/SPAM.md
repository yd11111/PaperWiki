---
type: paper
tier: deep
title: "SPAM: Style Prompt Adherence Metric"
arxiv_id: "2601.05554"
source: "Sources/SPAM.pdf"
authors: [Chanhee Cho, Nayeon Kim, Bugeun Kim]
year: 2026
venue: "arXiv preprint"
tags: [TTS-evaluation, prompt-adherence, contrastive-learning, CLAP, style-TTS, metric, SupCon]
concepts: ["[[SpeechFactorization]]", "[[Audio-LanguagePretraining]]", "[[ProsodyModeling]]", "[[SpeakerEmbedding]]"]
models: ["[[论文笔记/WavLM|WavLM]]", "[[论文笔记/TextrolSpeech|TextrolSpeech]]"]
tasks: ["[[InstructedSpeechGeneration]]"]
datasets: [TextrolSpeech, SpeechCraft, LibriTTS-P]
kb_context_sources: 4
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeechFactorization]], [[SpeakerEmbedding]], [[ProsodyModeling]], [[InstructedSpeechGeneration]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechFactorization]]✓, [[SpeakerEmbedding]]✓, [[ProsodyModeling]]✓, [[InstructedSpeechGeneration]]✓ | 过滤: [[TTSEvaluation]](pending-review), [[Audio-LanguagePretraining]](pending-review), [[NaturalLanguageDescriptionforTTS]](pending-review), [[StyleTransferinTTS]](pending-review) | 未命中但可能相关: 无

**谱系定位**: SPAM 处于 prompt-based TTS 评估的空白地带。现有 TTS 评估体系(参考 [[TTSEvaluation]] [待确认])已发展出 MOS/WER/SIM 等标准指标以及 TTSDS 分布级评估、LLM-as-Judge 等新范式,但针对 **style prompt adherence** 的专用自动指标尚属空白。SPAM 从 [[Audio-LanguagePretraining]] [待确认] 中的 CLAP 方法论出发,将其适配到 TTS prompt adherence 评估场景,核心创新在于引入 [[SpeechFactorization]] 的思想做属性级因子化对齐,而非全局 audio-text 匹配。

**与已有知识的关系**:
- [[SpeechFactorization]] 提供了 SPAM 最核心的设计灵感——将语音分解为 pitch/speed/energy/speaker/transcript 五个维度独立评估,而非整体匹配。这与 NaturalSpeech 3 的 factorized diffusion codec、Mega-TTS 的四维分解思路一脉相承,但目的从"生成"转向"评估"。
- [[SpeakerEmbedding]] 中的 X-Vector 被 SPAM 直接使用作为 speaker 因子的编码器,frozen 后接 feed-forward adapter。
- [[ProsodyModeling]] 定义的 pitch/energy/duration 三维度与 SPAM 因子化评估的 pitch/energy/speed 高度重合,但 SPAM 的 speed 用 variance predictor 估计,而非显式 duration predictor。
- [[InstructedSpeechGeneration]] 是 SPAM 评估的目标任务。现有该任务的评估主要依赖 Style Similarity、Emotion Accuracy 和 MOS,SPAM 提出了更 fine-grained 的自动化替代方案。

## 速查

> [!summary] 速查
> - **一句话**: 基于 CLAP 框架的 prompt adherence 自动评估指标,通过声学属性因子化 + SupCon 损失实现对 style prompt 的 plausible 且 faithful 评估
> - **路线**: Speech(WavLM+X-Vector+G2P）→ cross-attention → 4 branch fusion（global+speed+energy+pitch）→ speech embedding **a** ←cosine sim→ **b** ← Llama-3.1 8B + FF → Style Prompt
> - **指标**: LCC 0.58 / SRCC 0.58 / KTAU 0.41 (plausibility, MOS correlation, TextrolSpeech) [Table 1]; AR 0.862 / H1 rejected / H2 accepted (faithfulness, TextrolSpeech) [Table 1]
> - **可借鉴**: (1) SupCon loss 替代 infoNCE 处理 style key 的 many-to-many 匹配; (2) 用 auxiliary prediction heads (variance predictor for speed, MLP for energy/pitch) 引导各分支学习对应属性; (3) 因子化评估思路可迁移到其他 audio-text alignment 场景
> - **局限**: (1) 仅在英文数据集验证; (2) 因子化仅覆盖 pitch/speed/energy/speaker/transcript 五维度,缺少 emotion/breathing/pause 等细粒度副语言; (3) 未开源 (截至论文发布); (4) MOS 相关性为中等水平 (LCC ~0.58), 尚不足以完全替代人类评估

## 核心问题

**现有 prompt adherence 评估的两大缺陷**:

1. **Plausibility 不足**: 早期方法 (Prompttts [1], PromptStyle [2], Prompttts++ [3]) 通过 style embedding 的聚类可视化评估 prompt adherence,但嵌入距离与感知距离不对应,无法用于跨模型比较 [§1]。
2. **Faithfulness 不足**: LLM-as-a-judge 方法 (InstructTTSEval [6]) 虽能给出自动分数,但 LLM 对 prompt 扰动敏感,无法确保评估真正 grounded 在 style prompt 内容上 [§1]。

**SPAM 的解决思路**: 借鉴 CLAP 的 text-audio contrastive alignment 框架,但做两个关键改造:
- 因子化:显式分解声学属性,确保模型关注具体属性而非模糊全局匹配
- SupCon loss:处理同一 style key 对应多个 positive pair 的 many-to-many 问题

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SPAM 由三个模块组成 [§2, Fig 1]:

1. **Speech Encoder**: 将语音编码为包含多维度信息的 audio embedding $\hat{a}_t$
2. **Prompt Encoder**: 将 style prompt 文本编码为 prompt embedding **b**
3. **Speech-Prompt Fusion Module**: 将 audio embedding 因子化为属性级表示,生成最终 speech embedding **a**

推理时计算 **a** 与 **b** 的 cosine similarity 作为 prompt adherence 分数。

### 关键设计选择

#### 1. 三源 speech encoder (WHY: 捕获比 waveform 更丰富的信息)

Speech encoder 融合三个信号源 [§2.1]:
- **WavLM** (waveform encoder): 16kHz 输入 → 帧级 embedding $\mathbf{w}_t \in \mathbb{R}^h$。选择 WavLM 因为在 RA-CLAP 等多个 speech processing 系统中表现稳定 [论文原文]。
- **X-Vector** (speaker encoder): frozen X-Vector module + feed-forward adapter → speaker embedding $\mathbf{s} \in \mathbb{R}^h$。选择 X-Vector 而非 ECAPA-TDNN 因为它是 widely used 的 speaker 特征 [论文原文]。
- **G2P + Embedding lookup** (transcript encoder): grapheme → phoneme → embedding $\mathbf{c}_s$。直接嵌入 phoneme 到 $\mathbb{R}^h$ 空间 [§2.1]。

三者通过 **cross-attention** 融合: $\mathbf{w}_t + \mathbf{s}$ 作为 query, $\mathbf{c}_s$ 作为 key/value,得到每帧的 audio embedding $\hat{a}_t$ [§2.1]。

**为什么不只用 waveform**: [agent 解读] 纯 waveform 表示中 speaker/transcript 信息是隐含的,通过显式注入让模型更容易学习属性间的对应关系,与因子化设计的整体思路一致。

#### 2. 大语言模型作为 prompt encoder (WHY: 区分 prompt 中的细微语义差异)

使用 Llama-3.1 8B + feed-forward adapter 编码 style prompt [§2.2]。选择 LLM 而非 BERT/RoBERTa 的原因: style prompt 中的细微文本差异 (如 tone, mood) 可能对应语音中的显著变化,需要足够大的语言模型来区分这些语义差异 [论文原文]。

#### 3. 四分支因子化 fusion module (WHY: 显式对齐具体声学属性)

Speech-Prompt Fusion Module 将 $\hat{a}_t$ 分入四个并行分支 [§2.3]:

| 分支 | 功能 | 辅助 head | 训练信号 |
|------|------|-----------|----------|
| Global waveform | 整体表示,防止过拟合单一属性 | 无 | 仅 contrastive |
| Speed | 语速属性 | Variance predictor [17] | $\mathcal{L}_\delta(\hat{v})$ |
| Energy | 能量属性 | MLP | $\mathcal{L}_\delta(\hat{e})$ |
| Pitch | 音高属性 | MLP | $\mathcal{L}_\delta(\hat{p})$ |

每个属性分支通过 feed-forward 层将 $\hat{a}_t$ 变换为属性特定的表示,再由辅助 prediction head 预测帧级属性值 ($\hat{v}_t$, $\hat{e}_t$, $\hat{p}_t$)。辅助 head 仅在训练时使用,用于引导分支学习对应属性 [论文原文]。

最终 speech embedding **a** = 四个分支 embedding 逐帧相加后跨帧平均 [§2.3]。

**为什么需要 global waveform 分支**: 正则化训练过程,防止模型只依赖特定属性分支 [论文原文]。[agent 解读] 这也为因子化之外的属性 (如 emotion, breathing 等未显式建模的维度) 提供了"兜底通道"。

#### 4. SupCon loss 替代 InfoNCE (WHY: 处理 style key 的 many-to-many 匹配)

训练使用四个损失 [§2.4]:

$$\mathcal{L} = \lambda_c \mathcal{L}_{con} + \lambda_p \mathcal{L}_\delta(\hat{p}) + \lambda_v \mathcal{L}_\delta(\hat{v}) + \lambda_e \mathcal{L}_\delta(\hat{e})$$

其中 $\mathcal{L}_{con}$ 是对称 SupCon loss (非标准 infoNCE):

$$\mathcal{L}^{sup}(X, Y) = \sum_{x \in X} \left( \mathbb{E}_{p \sim P(x)} \left[ -\log \frac{e^{x^\top p}}{\sum_{y \in Y} e^{x^\top y}} \right] \right)$$

**为什么不用 InfoNCE**: prompt-based TTS 数据集用 style key (如 "male, high pitch, normal speed") 标注声学特征。同一 style key 对应多个 prompt-audio pair,即大 batch 中存在多个 positive example (birthday paradox)。标准 infoNCE 假设每个 anchor 只有一个 positive,会将 false negative 推远。SupCon [11] 显式处理多 positive 情况,利用所有同 style key 的样本作为 positive set $P(x)$ [§2.4, 论文原文]。

### 训练策略

- **训练数据**: TextrolSpeech [18] + SpeechCraft [19] 的 train set,仅使用高质量 ground-truth 语音 [§3.3]
- **辅助损失**: speed 用 Huber loss (variance predictor 方式), energy 和 pitch 也用 Huber loss [§2.4]
- **Style key 定义**: 从数据集提供的 style key 获取,包含 pitch/speed/energy/speaker/transcription 等属性的离散标签组合 [§2.4]
- **两个 SPAM 变体**: SPAM (WavLM) 和 SPAM (CLAP),区别在于 waveform encoder 使用 WavLM 还是 CLAP encoder [§3.3]

## 实验

### Plausibility 实验 (与人类评估的相关性)

320 名 CloudResearch 标注员,5 点 Likert 量表评估 prompt adherence MOS,每对 8 人评分 [§3.1]。

| 指标 | SPAM (WavLM) | SPAM (CLAP) | RA-CLAP | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| LCC | 0.584 | 0.554 | 0.520 | TextrolSpeech | [Table 1] |
| SRCC | 0.584 | 0.560 | 0.514 | TextrolSpeech | [Table 1] |
| KTAU | 0.405 | 0.389 | 0.357 | TextrolSpeech | [Table 1] |
| LCC | 0.580 | 0.516 | 0.429 | LibriTTS-P | [Table 1] |
| SRCC | 0.568 | 0.499 | 0.435 | LibriTTS-P | [Table 1] |
| KTAU | 0.400 | 0.346 | 0.304 | LibriTTS-P | [Table 1] |

Per-model LCC (Table 2) 显示 SPAM (WavLM) 在各 TTS 系统上保持更稳定的相关性: ground truth audio 上 LCC ~0.72 (两个数据集),而 RA-CLAP 波动明显 (0.726 vs 0.545) [Table 2]。

### Faithfulness 实验 (语义区分能力)

对每个 prompt-speech pair 生成 10 个 positive prompt (语义相似的改写) 和 10 个 negative prompt (语义不同) [§3.2]。

| 指标 | SPAM (WavLM) | SPAM (CLAP) | RA-CLAP | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| AR (Adherence Rate) | **0.862** | 0.841 | 0.852 | TextrolSpeech | [Table 1] |
| H1 reject (positive = original?) | ✓ (-2.025) | ✗ (-3.699***) | ✗ (-3.479**) | TextrolSpeech | [Table 1] |
| H2 accept (negative < original?) | ✓ (-20.145***) | ✓ (-17.538***) | ✓ (-16.912***) | TextrolSpeech | [Table 1] |

**关键发现**: 只有 SPAM (WavLM) 同时满足两个假设检验条件:
- H1: positive prompt 得分与 original 无显著差异 (t=-2.025, p>0.01) → 语义等价的 prompt 得到一致评分
- H2: negative prompt 得分显著低于 original (t=-20.145, p<0.001) → 语义不同的 prompt 被正确区分

RA-CLAP 和 SPAM (CLAP) 在 H1 上失败 (positive prompt 得分与 original 显著不同),意味着即使语义等价的改写也导致分数变化,缺乏 faithfulness [Table 1]。

[agent 解读] SupCon loss 是 faithfulness 优势的关键来源——它利用 style key 的多 positive 关系,让模型学会对同 style key 的不同表述给出一致评分。RA-CLAP 使用 standard CLAP loss 无法利用这一信号。

### 评估覆盖的 TTS 系统

5 个 prompt-based TTS 模型: PromptTTS [1], PromptStyle [2], VoxInstruct [4], ParlerTTS-mini-v1, ParlerTTS-large-v1 [5] [§3.3]。

## 局限性

1. **因子化维度有限**: 仅覆盖 pitch/speed/energy/speaker/transcript,缺少 emotion、breathing、pause、speaking style 等副语言维度。[agent 解读] 这可能导致对涉及这些维度的 prompt (如"用悲伤的语气") 评估不足,尽管 global waveform 分支可能部分捕获。
2. **MOS 相关性为中等水平**: LCC ~0.58 虽优于 RA-CLAP (~0.52),但远低于理想的"替代人类评估"水平。[agent 解读] 这可能与 prompt adherence 本身的主观性有关,也可能与因子化不完整有关。
3. **英文单语**: 训练和测试均限于英文数据集,跨语言泛化未知 [§3.3]。
4. **Prompt encoder 选择未充分消融**: 为什么选 Llama-3.1 8B 而非更小的模型 (如 Llama-3.2 1B)? 推理成本 vs 性能权衡未讨论。[agent 解读]
5. **Style key 依赖数据集标注**: SupCon loss 需要 style key 定义 positive set,这要求数据集提供结构化标签,限制了对无标签数据的适用性 [§2.4]。
6. **未与最新评估范式比较**: 未对比 LLM-as-Judge (InstructTTSEval)、TTSDS 分布级评估等最新方法,仅与 RA-CLAP (情感检索模型) 比较 [§3.3]。

## 点评

SPAM 填补了 prompt-based TTS 评估中一个实际空白: 现有 MOS 和 SIM 都不直接评估"语音是否符合 style prompt 的描述"。因子化设计让评估可解释 (知道模型关注了哪些属性),SupCon loss 解决了 CLAP 框架在 TTS 评估中的实际问题 (多 positive)。

但论文的定位有些保守。作为 2026 年的工作,SPAM 仅对比了 RA-CLAP 一个 baseline (且 RA-CLAP 是为 emotional speaking style retrieval 设计,并非 prompt adherence 评估),缺少与 InstructTTSEval (LLM-as-Judge)、TTSDS2 (分布级)、SpeechJudge (preference-based) 等同期工作的比较,削弱了其位置论证。

此外,LCC 0.58 的相关性水平在 TTS 评估文献中属于"可用但不够强"的范围 (对比 TTSDS2 在标准条件下 ρ≈0.67),说明因子化覆盖的维度可能还不够完整。

## 可复用的 idea

1. **Style key 驱动的 SupCon**: 当 contrastive learning 中存在 many-to-many 匹配关系时,用数据集的结构化标签定义 positive set,选择 SupCon 替代 infoNCE。可迁移到任何有类别标签的 audio-text 对齐任务。
2. **因子化评估思路**: 将"整体相似度"分解为多个可解释属性维度单独对齐。可迁移到 voice cloning quality 评估 (分别评估 timbre/prosody/content fidelity)。
3. **辅助 prediction head 引导分支特化**: 训练时用 auxiliary loss 确保各分支学到对应属性,推理时去掉 head 只用 embedding。简单有效的多分支训练技巧。

## 审阅

_待审阅_


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
