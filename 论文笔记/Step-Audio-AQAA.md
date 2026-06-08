---
type: paper
tier: deep
title: "Step-Audio-AQAA: a Fully End-to-End Expressive Large Audio Language Model"
arxiv_id: "2506.08967"
source: "Sources/Step-Audio-AQAA.pdf"
authors: [Step-Audio Team (StepFun)]
year: 2025
venue: "arXiv"
tags: [speech-LM, end-to-end, AQAA, audio-language-model, DPO, dual-codebook, flow-matching, vocoder, post-training, weight-merging]
concepts: ["[[SpeechLanguageModel]]", "[[SemanticvsAcousticTokens]]", "[[ConditionalFlowMatching]]", "[[NeuralVocoder]]", "[[Single-codebookvsMulti-codebook]]", "[[Speech-TextAlignment]]", "[[EmotionControlinTTS]]", "[[AudioUnderstanding]]"]
models: ["[[Step-Audio]]", "[[CosyVoice]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 1 个待确认实体页: [[SpeechLanguageModel]]✓, [[SemanticvsAcousticTokens]]✓, [[ConditionalFlowMatching]]✓, [[NeuralVocoder]]✓, [[CosyVoice]]✓, [[CodecLanguageModel]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]], [[SemanticvsAcousticTokens]], [[ConditionalFlowMatching]], [[NeuralVocoder]], [[CosyVoice]], [[CodecLanguageModel]] | 过滤: [[AudioUnderstanding]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review), [[Speech-TextAlignment]](pending-review) | 未命中但可能相关: 无

**谱系定位**: Step-Audio-AQAA 是 StepFun 的 Step-Audio 系列的端到端语音交互升级。前作 Step-Audio 是 AQTA (Audio Query-Text Answer) 范式,仍依赖外接 TTS 将文本转为语音。Step-Audio-AQAA 转向 AQAA (Audio Query-Audio Answer),直接从 audio input 生成 audio output,属于 [[SpeechLanguageModel]] 中"fully end-to-end"路线的新成员,与 Kimi-Audio、Qwen2.5-Omni、GLM-4-Voice 等系统同代竞争。

**已有认知**:
- **Dual-codebook**: 本文采用 linguistic + semantic 双码本,对应 [[SemanticvsAcousticTokens]] 中的混合路线。但值得注意的是,这里的"linguistic tokenizer"(Paraformer encoder)和"semantic tokenizer"(CosyVoice 1.0 式)都偏 semantic 侧 — 实际上是两种不同粒度的语义表征,而非经典的 semantic vs acoustic 二分。
- **Vocoder**: 使用 CosyVoice 1.0 启发的 OT-CFM vocoder [§2.3],属于 [[ConditionalFlowMatching]] 在 [[NeuralVocoder]] 中的标准应用。
- **DPO**: 引入 audio-token masked DPO 做偏好对齐,是 SpeechLM post-training 中 [[DifferentiableRewardOptimization]] 路线的新变体。

**创新判断**: 相对于 KB 中已有系统(CosyVoice 系列专注 TTS、Step-Audio 依赖级联 TTS),Step-Audio-AQAA 的核心新意在于 (1) 130B 规模的端到端 AQAA 系统,(2) tri-codebook interleaved output (text + linguistic + semantic tokens 10:6:9),(3) audio-token masked DPO 策略。

## 速查

> [!summary] 速查
> - **一句话**: 130B 端到端 LALM,通过双码本 tokenizer + tri-codebook interleaved output + masked DPO + 三模型权重合并,实现 audio-in audio-out 直接交互,在语音情感控制等维度领先 Kimi-Audio 和 Qwen-Omni
> - **路线**: Audio input → dual-codebook tokenizer (linguistic 16.7Hz + semantic 25Hz, 2:3 interleave) → 130B LLM (Step-Omni) → interleaved text+audio tokens (10:6:9) → OT-CFM vocoder → speech output
> - **指标**: StepEval-Audio-360 人工评估 MOS (1-5),在 Speech Emotion Control / Creativity / Language Ability / Role-playing / Gaming 维度领先 Kimi-Audio 和 Qwen-Omni [Fig 3]; ablation 中 ratio_10_15 的 Chat 4.03 / Factuality 0.67 显著优于 audio_only 的 1.72 / 0.03 [Table 1]
> - **可借鉴**: (1) DPO 时 mask audio token loss 避免文本-音频对齐退化; (2) text tokens 作为中间输出辅助 audio generation 的语义一致性; (3) 多阶段模型权重合并 (5:5:1) 融合互补能力
> - **局限**: 仅有 MOS 人工评估,缺乏 WER/SIM 等客观指标; Singing 和 Voice Instruction Following 维度弱; 模型 130B 参数未公开训练细节; 评测基准 StepEval-Audio-360 为自建,外部可比性有限

## 核心问题

Step-Audio-AQAA 要解决的核心问题是: 现有 LALMs 大多输出文本 tokens 后再级联 TTS 生成语音,这种 AQTA+TTS 范式存在信息丢失 (副语言信息在 ASR→text→TTS 管道中丢失)、高延迟 (多模块串行)、错误累积三大问题 [§1]。更关键的是,级联范式无法实现细粒度的语音控制 (如句内情感/语速切换),因为 TTS 模块作为独立后处理无法感知对话上下文 [§1, 最后一段]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由三个核心模块构成 [§2, Fig 1]:

1. **Dual-Codebook Audio Tokenizer**: 将输入音频同步编码为两种离散 token 序列
   - Linguistic tokenizer: 基于 Paraformer encoder [12],16.7 Hz,codebook 大小 1024,提取高层语言学/音素属性 [§2.1]
   - Semantic tokenizer: 参考 CosyVoice 1.0 [10],25 Hz,codebook 大小 4096,编码粗粒度声学特征 [§2.1]
   - 两种 token 以 2:3 比例交错排列形成输入序列 (因采样率近似 2:3) [§2.1]

2. **Backbone LLM (Step-Omni, 130B)**: 预训练的多模态 LLM (text+speech+image),decoder-only Transformer,使用 RMSNorm + grouped query attention [§2.2]。词表扩展 5120 个 audio token embeddings。输出为 text + audio tokens 交错序列 (text:linguistic:semantic = 10:6:9,等效 text:audio = 10:15) [§2.2, Fig 1]

3. **Neural Vocoder**: 借鉴 CosyVoice 1.0 的 OT-CFM 模型,U-Net 架构 (ResNet-1D + Transformer blocks),仅以 audio tokens 为条件合成波形 [§2.3]

### 关键设计选择

**为什么用双码本而非单码本?** 论文作者指出,linguistic tokens 和 semantic tokens 互为参照 (mutually referenced),双码本训练时两种 token 的 next-token prediction perplexity 都比单码本 [19] 有下降 [§2.1] [论文原文]。[agent 解读] 这本质上是一种互信息增益: linguistic token 提供细粒度音素线索帮助预测 semantic token,反之亦然,两种表征互补减少了不确定性。

**为什么输出 text+audio interleaved tokens 而非纯 audio?** Ablation [Table 1] 明确显示,纯 audio 输出 (audio_only) 的 Chat 得分仅 1.72,Factuality 仅 0.03,几乎无法正确回答问题; 而 ratio_10_15 (text:audio=10:15) 的 Chat 达 4.03,Factuality 达 0.67 [§5.2]。论文作者解释: "when the token information of the generated text adequately encompasses the subsequently produced speech tokens, there is a notable enhancement in quality" [§5.2] [论文原文]。[agent 解读] text tokens 起到 "chain-of-thought" 的语义锚定作用 — LLM 先在文本空间规划语义内容,再在 audio 空间生成对应的声学 token,本质上是用 text modality 的强语义建模能力辅助 audio modality。

**为什么 DPO 要 mask audio tokens?** 论文作者发现,对所有 token (含 audio) 施加 DPO 会导致"text and audio misalignment" [§3.3] [论文原文]。他们推测 DPO 部分破坏了生成 voice tokens 的能力,因此在后续 DPO 中屏蔽 audio token 的 loss,仅对 text tokens 做偏好优化 [§3.3] [论文原文]。

**为什么做三模型权重合并?** SFT-1st、SFT-2nd 和 DPO 三个模型有不同的优化目标和能力侧重: SFT-1st 增强语义一致性和端到端格式对齐,SFT-2nd 强化特定能力 (如 singing) 和输出格式稳定性,DPO 对齐人类偏好 [§3.4] [论文原文]。合并公式为 W = (5*W_SFT1 + 5*W_SFT2 + 1*W_DPO)/11 [Eq. 3] — DPO 权重仅占 1/11,论文隐含的原因是 DPO 从 SFT-1st 出发训练 (而非 SFT-2nd),能力已有偏移 [§3.3 最后一段] [agent 解读]。

### 训练策略

训练分四个阶段 [§3, Fig 2]:

1. **预训练 (Step-Omni)** [§3.1]: 三阶段多模态预训练
   - 阶段 1: audio:text:image = 2:1:1,主要更新 audio embedding + LM head
   - 阶段 2: 加入 audio-text interleaved data 强化音频能力
   - 阶段 3: 引入 ASR/TTS 数据微调
   - 每种模态约 800B tokens [§3.1]

2. **SFT 第一阶段** [§3.2]: 全参数微调,混合 AQTA + AQTAA 数据训练 1 epoch。AQTAA 数据通过 Step-Audio-TTS-3B [19] 将文本答案转为高质量语音答案生成。目标函数为标准 CE loss [Eq. 1]

3. **SFT 第二阶段** [§3.2]: 选择高质量 AQTAA 数据,稳定 text-audio interleaved 输出格式,并增强特定能力 (如 singing)

4. **DPO** [§3.3]: 从 SFT-1st 模型出发 (非 SFT-2nd,因后者强化某些能力时损伤了其他能力),使用 audio-token masked DPO [Eq. 2]。通过 indicator function I(a_t ∉ A) 屏蔽 audio token 的梯度

5. **Weight Merging** [§3.4]: 最终模型 = (5*SFT-1st + 5*SFT-2nd + 1*DPO)/11

**多标签语音数据的处理** [§5.2, Table 2]: 为实现句内情感/语速切换,论文探索了三种多标签音频拼接方式:
- concatenation with marker removal: 移除 `<audio_start>/<audio_end>` 后拼接 → 模型几乎不生成多标签语音
- pre-interleaved concatenation: 先各自与 text interleave 再拼接 → 破坏一致性
- **marker-preserving concatenation** (最优): 保留 markers 直接拼接 → Chat 4.22,最高

论文作者解释 marker-preserving 最优的原因: 单标签数据训练的模型已学会在 `<audio_start>...<audio_end>` 内保持单一语音状态,前两种方法破坏了这种一致性 [§5.2] [论文原文]。

## 实验

| 指标 | 本文 (Step-Audio-AQAA) | Kimi-Audio | Qwen-Omni | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Speech Emotion Control (MOS) | **最高** | 低于本文 | 低于本文 | StepEval-Audio-360 | [Fig 3] |
| Creativity (MOS) | **最高** | - | - | StepEval-Audio-360 | [Fig 3] |
| Language Ability (MOS) | **最高** | - | - | StepEval-Audio-360 | [Fig 3] |
| Role-playing (MOS) | **最高** | - | - | StepEval-Audio-360 | [Fig 3] |
| Gaming (MOS) | **最高** | - | - | StepEval-Audio-360 | [Fig 3] |
| Logical Reasoning (MOS) | 领先 | - | - | StepEval-Audio-360 | [Fig 3] |
| Voice Understanding (MOS) | 略领先 | - | - | StepEval-Audio-360 | [Fig 3] |
| Singing (MOS) | **弱于** | - | 较强 | StepEval-Audio-360 | [Fig 3] |
| Voice Instruction Following (MOS) | **弱于** | 较强 | - | StepEval-Audio-360 | [Fig 3] |

**注**: 论文仅提供雷达图 (Fig 3) 的视觉比较,未给出具体 MOS 数值,无法提取精确数字。

**Ablation: Text-Audio Mixing Ratio** [Table 1]:

| 配置 | Chat↑ | Relevance↑ | Factuality↑ |
| --- | --- | --- | --- |
| audio_only | 1.72 | 0.05 | 0.03 |
| ratio_6_50 | 1.20 | 0.04 | 0.05 |
| ratio_3_5 | 1.03 | 0.00 | 0.00 |
| text_cot | 4.01 | 0.59 | 0.58 |
| **ratio_10_15** | **4.03** | **0.65** | **0.67** |

**Ablation: Multi-label Interleaving** [Table 2]:

| 方法 | Chat↑ | Relevance↑ | Factuality↑ |
| --- | --- | --- | --- |
| pre-interleaved concatenation | 3.82 | 0.54 | 0.59 |
| concatenation with marker removal | 4.08 | 0.56 | 0.57 |
| **marker-preserving concatenation** | **4.22** | **0.57** | **0.57** |

## 局限性

1. **评测局限**: 仅使用自建 benchmark (StepEval-Audio-360) 进行人工 MOS 评估,缺乏 WER、SIM、PESQ 等客观自动指标,外部可比性和可复现性不足 [agent 解读]

2. **Singing 和 Voice Instruction Following 弱**: 论文坦承加入过多 singing 数据会严重损害其他能力,而 Voice Instruction Following 数据不足导致该维度表现弱 [§5.1] [论文原文]

3. **训练细节不完整**: 130B 模型的训练数据规模、SFT 数据量、DPO 数据构造方式等关键细节缺失,仅称 AQTA 数据为"proprietary",AQTAA 数据由 Step-Audio-TTS-3B 合成 [§3.2] [论文原文]

4. **Ablation 使用 GPT-4o 自动评估**: 为"save manpower and maintain objectivity"使用 LLM 做 ablation judge [§5.2],但仅 3 个维度 (Chat/Relevance/Factuality),且未验证与人工评估的一致性 [agent 解读]

5. **Vocoder 缺乏独立评估**: vocoder 对最终质量的贡献未被单独量化,也未与其他 vocoder 方案对比 [agent 解读]

6. **开放问题**: 论文自己指出三个未解决问题 — (a) 是否能无 text token 引导直接生成有意义的 audio tokens,(b) 离散 audio tokens 是否是最优表征,(c) 高质量 singing 仍是挑战 [§7]

## 点评

Step-Audio-AQAA 是 StepFun 从 AQTA 级联范式走向端到端 AQAA 的标志性工作。其核心贡献不在架构创新 (三模块管线已有先例),而在工程层面的 post-training 策略设计:

1. **Audio-token masked DPO** 是一个实用且反直觉的发现: 对 audio tokens 做 DPO 反而有害。这可能是因为当前 DPO 的偏好信号主要来自语义层面 (人类评估语义正确性),而非声学层面 (人类评估音质),直接对 audio tokens 施加语义偏好梯度会破坏声学生成能力。

2. **Text-audio interleaved output** 的 ablation 是本文最有说服力的实验: audio_only 几乎完全失败 (Chat 1.72, Factuality 0.03),证明当前规模的 LLM 仍然依赖 text modality 作为语义锚点来生成有意义的 audio 输出。这与 §7 中作者自己提出的"是否能无 text 引导生成"的未来方向形成呼应。

3. **Weight merging (5:5:1)** 作为 post-training 最后一步,是一种低成本的能力融合策略。DPO 权重仅 1/11 反映了 preference alignment 在当前 AQAA 系统中的边际地位 — 语义一致性 (SFT) 远比偏好对齐 (DPO) 重要。

4. **不足**: 论文的评测体系偏弱 — 自建 benchmark + 纯人工 MOS + 无客观指标的组合让结果难以被社区复现和公平比较。与同期 Kimi-Audio (提供了丰富的 ASR/情感/TTS 自动指标) 相比,Step-Audio-AQAA 的评测说服力明显不足。

## 可复用的 idea

1. **DPO mask audio tokens**: 在端到端 speech LLM 中做 DPO 时,仅对 text tokens 施加偏好梯度,屏蔽 audio tokens — 避免语义偏好信号破坏声学生成能力。适用场景: 任何 text-audio interleaved output 的 speech LLM post-training

2. **Text tokens 作为语义锚点**: 在 audio generation 模型中保留 text token 的中间输出,利用 LLM 在文本空间的强语义建模能力引导 audio token 生成。这一策略不仅限于对话场景,也可用于 TTS 等 conditioned generation

3. **多阶段 weight merging**: 不同训练目标的模型通过加权平均融合互补能力,简单有效,可用于任何多阶段 fine-tuning 的合并

4. **Marker-preserving multi-label concatenation**: 在构造句内多状态切换的训练数据时,保留 `<audio_start>/<audio_end>` 标记直接拼接,而非移除标记或预先 interleave — 利用模型已学会的"标记内单一状态"先验

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节包含因果解释 (WHY dual-codebook / WHY interleaved / WHY masked DPO / WHY weight merge),速查卡片"可借鉴"为具体策略 |
> | 可信赖 | pass-with-fixes | 数字标注覆盖率高 (Table/Fig/§ 引用充分),但主实验仅有雷达图视觉比较无精确数值 — 这是论文本身的局限而非笔记问题 |
> | 可区分 | pass | 因果解释标注了 [论文原文] / [agent 解读],推断性分析有限定词 |
> | 可定位 | pass | KB 背景含具体谱系定位 (Step-Audio → AQAA 的升级 + 与 Kimi-Audio/Qwen-Omni 的代际定位),创新判断有对比基准 |
> | 不污染 | pass | 无反向更新,无 KB 污染风险 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> - ⚠️ [medium/traceability-gap] 主实验 (§5.1) MOS 数值未提供,雷达图无法提取精确数字 — 已在实验表中标注"仅视觉比较"
> - ℹ️ [low/template-compliance] datasets frontmatter 为空 — StepEval-Audio-360 尚无独立实体页,暂不列入
> 详见 `_review/Step-Audio-AQAA-review.yml`

---

检索命中: [[SpeechLanguageModel]], [[SemanticvsAcousticTokens]], [[ConditionalFlowMatching]], [[NeuralVocoder]], [[CosyVoice]] | 过滤: [[CodecLanguageModel]](pending-review), [[AudioUnderstanding]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review), [[Speech-TextAlignment]](pending-review) | 未命中但可能相关: 无
