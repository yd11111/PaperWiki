---
type: paper
tier: deep
title: "Towards Accurate Lip-to-Speech Synthesis in-the-Wild"
arxiv_id: "2403.01087"
source: "Sources/LipToSpeech.pdf"
authors: [Sindhu Hegde, Rudrabha Mukhopadhyay, C.V. Jawahar, Vinay Namboodiri]
year: 2023
venue: "ACM Multimedia 2023 (MM '23)"
tags: [lip-to-speech, visual-speech, multi-modal, TTS, lip-reading, assistive-technology, audio-visual]
concepts: ["[[SpeakerEmbedding]]", "[[MelSpectrogram]]", "[[DurationPredictor]]", "[[ProsodyModeling]]", "[[NeuralVocoder]]", "[[Non-autoregressiveTTS]]"]
models: ["[[BigVGAN]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文是一篇跨模态语音合成工作,将视觉信息(唇部视频)与文本信息联合用于语音生成。在 KB 中,它连接了 TTS pipeline 的多个关键组件:使用 [[SpeakerEmbedding]] 实现多说话人合成(open-set, speaker encoder 范式);输出 [[MelSpectrogram]] 由 [[NeuralVocoder]] (BigVGAN) 转换为波形;其 Visual-Text Attention 机制巧妙地替代了传统 [[DurationPredictor]] 的功能,利用视频帧与文本的天然时序对应来确定每个音素的时长;[[ProsodyModeling]] 是本文试图解决的核心难题之一 --- 从唇部动作中推断韵律和说话风格。
>
> **已有认知**: Speaker Embedding 在 TTS 中有成熟的注入方式(addition/concatenation/cross-attention 等),本文采用的 addition 方式是经典方案。Neural Vocoder 领域 BigVGAN 是 2022-2023 年的高质量选择。Duration Predictor 在 NAR TTS 中是核心组件,但本文通过视觉-文本注意力机制完全规避了显式 duration prediction 的需要。
>
> **创新判断**: 本文的核心新意不在 TTS 组件本身,而在于如何将 lip-to-text 的进展(尤其是 VTP 模型的视觉特征和文本预测)桥接到 lip-to-speech 任务中。这是一种任务分解策略:将困难的 lip→speech 拆解为 lip→text + visual-conditioned text→speech。
>
> 检索命中: [[SpeakerEmbedding]]✓, [[NeuralVocoder]]✓, [[ProsodyModeling]]✓ | 过滤: [[MelSpectrogram]](pending-review), [[DurationPredictor]](pending-review), [[Non-autoregressiveTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 lip-to-speech 分解为 lip-to-text + visual-conditioned TTS,利用预训练 lip reading 模型的噪声文本输出和视觉特征,首次在 in-the-wild 多说话人场景下实现高质量、唇同步的语音合成
> - **路线**: 唇部视频 → VTP (lip-to-text, 预训练冻结) → 噪声文本 + 视觉特征 → Visual TTS (Text Encoder + Visual Encoder + Visual-Text Attention + Speaker Embedding + Spectrogram Decoder) → mel spectrogram → BigVGAN → 语音波形
> - **指标**: LRS2 上 PESQ 1.47 / STOI 0.65 / ESTOI 0.47 / LSE-C 8.083 / LSE-D 6.586,全面超越 VCA-GAN/SVTS/Multi-task L2S;MOS 评估 intelligibility 3.49, content clarity 3.52, sync 3.82, overall 3.31 [Table 1, Table 2]
> - **可借鉴**: (1) 利用相邻任务(lip-to-text)的成熟模型为更难任务(lip-to-speech)提供中间监督; (2) 视觉-文本注意力天然提供时序对齐,省去 duration predictor; (3) 视频帧与 mel 帧的固定倍率关系可直接上采样,无需学习 duration
> - **局限**: 依赖预训练 lip-to-text 模型(需文本监督训练);仅测试英语;唇读文本噪声(WER ~22.6%)会传播到语音输出;未开源完整训练代码

## 核心问题

本文要解决的核心问题是:**为什么现有 lip-to-speech 模型在 in-the-wild 场景下生成的语音不可懂?**

作者的诊断 [§1, §3.1]: 现有方法试图直接从语音监督中同时学习"说了什么"(语言内容)和"怎么说"(声音/韵律),但语音信号中说话人身份、口音、韵律等变化太大,模型难以从中抽取出稳定的语言模型。已有的 lip-to-text 模型已经很好地解决了"说了什么"的问题(WER 低至 17-22%),但 lip-to-speech 领域完全忽视了这一进展。

**核心 insight**: 将 lip-to-speech 拆解为两个子任务 --- (1) lip-to-text(由预训练模型完成,提取内容信息)和 (2) visual-conditioned TTS(利用文本+视觉特征生成同步语音)。这不是简单的级联,因为还需要视觉信息来确保语音与唇部动作在时序上对齐。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

两阶段框架 [§3.2, Fig 2]:

**Stage 1 --- 预训练 Lip-to-Text (VTP)**:
- 输入: 连续 5 帧唇部视频 $S_v \in \mathbb{R}^{T \times H \times W \times 3}$
- 处理: Spatio-temporal ResNet → Visual Transformer Pooling → Transformer encoder-decoder → beam search + LM rescoring
- 输出: (1) 文本预测 $S_t$ (sub-word token 序列); (2) 逐帧视觉表示 $g \in \mathbb{R}^{T \times f_d}$ (512维)
- 状态: **冻结**,使用公开预训练权重(在 LRS2+LRS3 上训练)
- [论文原文] VTP 的视觉特征之所以有效,是因为它们用文本监督训练,因此隐含了准确的内容信息,与之前直接从语音监督学习视觉特征的方法形成对比 [§3.2.2]

**Stage 2 --- Visual Text-to-Speech**:
- 输入: 噪声文本 $S_t$ + 视觉特征 $g$ + 目标说话人语音(1秒)
- 输出: mel spectrogram $S_m$ → BigVGAN → 波形
- 五个组件:
  1. **Text Encoder**: 音素化文本 → positional encoding → FFT blocks (4层) → 文本嵌入 $E_{text} \in \mathbb{R}^{N \times d}$ [§3.2.2]
  2. **Visual Encoder**: VTP 视觉特征 $g$ → positional encoding → FFT blocks (4层) → 视觉嵌入 $E_{vis} \in \mathbb{R}^{N \times T \times d}$ [§3.2.2]
  3. **Visual-Text Attention**: scaled dot-product attention, $Q = E_{vis}$, $K = V = E_{text}$ → 对齐输出 $A \in \mathbb{R}^{T \times d}$ [Eq. 1, §3.2.2]
  4. **Speaker Embedding**: 1秒参考音频 → 预训练 speaker encoder → 256维向量 $E_{voice}$,加到上采样后的注意力输出上 [§3.2.2]
  5. **Spectrogram Decoder**: Transformer decoder layers → mel spectrogram $S_m$ [§3.2.2]

### 关键设计选择

**1. 为什么用噪声文本而不用 GT 文本?**
- [论文原文] 推理时没有 GT 文本,只有 lip-to-text 模型的噪声预测(WER ~22.6%)。训练时使用噪声预测而非 GT 文本,使模型学会在噪声文本条件下生成合理语音 [§3.2.1]
- [agent 解读] 这是一个 train-inference 一致性的设计:如果训练用 GT 文本但推理用噪声文本,模型会因分布不匹配而退化

**2. Visual-Text Attention 如何替代 Duration Predictor?**
- [论文原文] 视频帧与 mel spectrogram 之间存在固定的时序对应:mel 长度 = 视频长度 × 固定倍率 $n$(设为 4)。注意力输出 $A$ 在 $T$ 维度上直接上采样 $n$ 倍即可得到 mel 长度,无需单独的 duration predictor [§3.2.2]
- [agent 解读] 这本质上是利用了视觉模态作为"天然时钟":唇部动作的时序直接决定了语音的时序,而文本-视频注意力自动学习了哪个音素在哪个时刻发出。这比传统 NAR TTS 的 duration predictor 更优雅,因为 duration 信息直接来自视频而非预测

**3. 为什么选择 VTP 而非其他 lip-to-text 模型?**
- [论文原文] VTP 有两个独特优势:(1) 数据高效;(2) 强视觉骨干网络(visual transformer pooling 提供精确的唇部特征)[§3.2.1]
- 消融实验验证 [Table 3]: 替换为 DeepLR(WER 51.3)或 AV-HuBERT(WER 46.1)后性能显著下降,VTP(WER 22.6)的低 WER 直接转化为更好的语音生成质量

**4. 为什么不用 face crops 而用 VTP embeddings?**
- [论文原文] VTP embeddings 是用文本监督训练的,天然编码了内容信息;而 raw face crops 可能包含不必要的信息(如面部身份)[§3.2.2]
- 消融实验 [Table 4]: face crops 的 PESQ 1.17 vs VTP embeddings 1.47,差距巨大

### 训练策略

- 训练数据: LRS2 (~230h) 和 LRS3 (~430h) 的官方训练集 [§3.3.1]
- 损失函数: L1 重建损失(预测 mel vs GT mel)[§3.3.3]
- 优化器: Adam ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-9}$),学习率 schedule 同 VTP [§3.3.3]
- batch size: 16,训练约 900K 步(单卡 NVIDIA 2080 Ti)[§3.3.3]
- Mel 参数: 80 mel-bands, hop length 10ms, window 25ms, 16kHz [§3.3.2]
- Vocoder: BigVGAN,单独训练,仅在推理时使用 [§3.3.3]

## 实验

| 指标 | 本文 | Multi-task L2S | VCA-GAN | SVTS | L2T+TTS baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PESQ↑ | **1.61** | 1.56 | 1.33 | 1.49 | 0.69 | LRW | [Table 1] |
| STOI↑ | **0.71** | 0.64 | 0.56 | 0.64 | 0.10 | LRW | [Table 1] |
| ESTOI↑ | **0.56** | 0.47 | 0.36 | 0.48 | 0.01 | LRW | [Table 1] |
| PESQ↑ | **1.47** | 1.36 | 1.24 | 1.34 | 0.53 | LRS2 | [Table 1] |
| STOI↑ | **0.65** | 0.52 | 0.40 | 0.49 | 0.19 | LRS2 | [Table 1] |
| ESTOI↑ | **0.47** | 0.34 | 0.13 | 0.29 | 0.02 | LRS2 | [Table 1] |
| LSE-C↑ | **8.083** | 4.001 | 1.874 | — | 2.013 | LRS2 | [Table 1] |
| LSE-D↓ | **6.586** | 8.192 | 11.48 | — | 15.891 | LRS2 | [Table 1] |
| PESQ↑ | **1.39** | 1.31 | 1.23 | 1.25 | 0.42 | LRS3 | [Table 1] |
| STOI↑ | **0.58** | 0.48 | 0.47 | 0.50 | 0.16 | LRS3 | [Table 1] |
| LSE-C↑ | **7.886** | 3.876 | 3.905 | — | 1.771 | LRS3 | [Table 1] |
| MOS overall↑ | **3.31** | 2.64 | 2.54 | — | 2.96 | LRS3 test | [Table 2] |
| MOS sync↑ | **3.82** | 3.01 | 2.97 | — | 1.01 | LRS3 test | [Table 2] |

**关键观察**:
1. 本文在所有 in-the-wild 数据集上的所有指标均取得最佳,尤其是 lip-sync 指标(LSE-C/D)优势巨大 [Table 1]
2. L2T+TTS baseline 的 intelligibility MOS 略高于本文(3.61 vs 3.49),但 sync 极差(1.01),说明简单级联无法解决时序对齐问题 [Table 2]
3. 在约束场景(TCD-TIMIT, 3 speakers)下本文仅"comparable",优势主要体现在 in-the-wild 场景 [Table 1]
4. 使用 GT text 时 PESQ 1.51 vs 噪声文本 1.47,差距不大,说明模型对文本噪声有一定鲁棒性 [Table 3]

## 局限性

1. **依赖预训练 lip-to-text 模型**: 虽然推理不需要 GT 文本标注,但 VTP 模型本身是用大量文本标注训练的,限制了向低资源语言扩展的能力 [§6]
2. **仅测试英语**: 未在其他语言上验证,跨语言泛化性未知 [§6]
3. **文本噪声传播**: lip-to-text 的错误(WER ~22.6%)会传播到最终语音,尤其是同音异义词无法区分 [§5]
4. **指标绝对值仍然偏低**: 最佳 PESQ 1.61(LRW)[Table 1],[agent 解读] 远低于典型 TTS 系统的 PESQ 水平,lip-to-speech 任务本身仍有很大提升空间
5. **需要目标说话人参考音频**: 需要 1 秒参考语音提取 speaker embedding,限制了完全零资源场景
6. **VTP 视觉特征的瓶颈**: 使用 VTP 特征优于 face crops [Table 4],但也意味着性能上限受限于 VTP 的视觉表示质量

## 点评

**优势**:
- 问题洞察精准:准确诊断了现有 lip-to-speech 模型的核心瓶颈(从语音中学语言模型太难),并提出了合理的解决方案(借助 lip-to-text 的进展)
- 架构设计优雅:Visual-Text Attention 巧妙利用视频的天然时序来替代 duration predictor,比传统 NAR TTS 更自然
- 实验全面:覆盖约束/非约束场景,多个数据集,自动+人工评估,多组消融
- 实际应用展示:ALS 患者语音生成的 demo 有说服力,展示了技术的社会价值

**不足**:
- 论文声称"不需要文本标注",但实际依赖的 VTP 模型需要大量文本标注训练,这个声明有误导性
- 缺少与 AV-HuBERT 作为视觉特征提取器的对比(仅对比了用 AV-HuBERT 做 lip-to-text 的结果)
- MOS 绝对值偏低(最高 3.82 sync, 3.31 overall),距离实用还有距离
- 未讨论实时性/延迟,对于辅助技术应用场景这是关键指标

**在 TTS 知识体系中的位置**:
这是一篇跨模态工作,核心贡献不在 TTS 技术本身,而在于如何将 TTS 技术(文本编码、speaker embedding、mel spectrogram 生成)应用到视觉-语音跨模态场景。其中 Visual-Text Attention 替代 Duration Predictor 的思路对 TTS 领域有启发:当有外部时序信号(如视频)时,可以免去显式 duration prediction。

## 可复用的 idea

1. **任务分解策略**: 当端到端学习太难时,利用相邻任务的成熟预训练模型提供中间监督。类似思路可用于其他跨模态生成任务(如 gesture-to-speech, brain-to-speech)
2. **外部时序信号替代 duration predictor**: 当有与输出时序对齐的外部模态(视频、手势、MIDI 等)时,可通过跨模态注意力直接建立时序对应,免去 duration prediction
3. **噪声条件训练**: 训练时使用模型预测(而非 GT)作为条件输入,保持 train-inference 一致性,提升鲁棒性。这个思路在 TTS 中可用于处理噪声 phoneme、噪声 prosody 标注等场景
4. **固定倍率上采样**: 视频帧与 mel 帧之间的固定 4 倍关系被直接利用,避免了学习长度映射。类似的先验知识(如 phoneme 与 frame 的大致比例)可用于简化 TTS 架构

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节因果解释充分,设计选择 WHY 清晰 |
> | 可信赖 | pass-with-fixes | LRS2 VCA-GAN LSE 值已修正;局限性外部比较已标注 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率 >90% |
> | 可定位 | pass | KB 背景谱系定位具体,6 页 KB 参考 |
> | 不污染 | pass | 仅追加更新,无新建页风险 |
> 
> Issues: 3 (high: 1, medium: 1, low: 1) — 已当场修正 high issue
> 详见 `_review/LipToSpeech-review.yml`
