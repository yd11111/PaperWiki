---
type: paper
tier: deep
title: "UMETTS: A Unified Framework for Emotional Text-to-Speech Synthesis with Multimodal Prompts"
arxiv_id: "2404.18398"
source: "Sources/UMETTS.pdf"
authors: [Zhi-Qi Cheng, Xiang Li, Jun-Yan He, Junyao Chen, Xiaomao Fan, Xiaojiang Peng, Alexander G. Hauptmann]
year: 2024
venue: "IEEE (preprint, arXiv:2404.18398v2)"
tags: [TTS, emotion, multimodal, contrastive-learning, style-transfer, expressiveness]
concepts: ["[[EmotionControlinTTS]]", "[[GlobalStyleTokens]]", "[[StyleTransferinTTS]]", "[[ProsodyModeling]]", "[[SpeakerEmbedding]]", "[[NaturalLanguageDescriptionforTTS]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: UMETTS 处于 Emotional TTS 从"离散标签/单模态参考"向"多模态 prompt"演进的节点。在已有知识中:

- **[[EmotionControlinTTS]]** [待确认] 梳理了 E-TTS 的演进线: emotion embedding (2021) → 多尺度层级建模 (MsEmoTTS, 2022) → DPO 优化 (Emo-DPO, 2024) → 零样本情感 (EmoSphere++, 2024)。UMETTS 的创新在于**跨模态对齐**(视觉+音频+文本 → 统一 emotion embedding),这是已有演进线未覆盖的方向。
- **[[GlobalStyleTokens]]** [待确认] 描述了 GST 的 reference encoder + style token bank 架构及其局限(全局粒度、speaker/style 纠缠)。UMETTS 论文明确指出 GST 方法"may not fully disentangle speaker characteristics from emotional and prosodic elements" [§I],这与 KB 中对 GST 局限的描述一致。UMETTS 用对比学习替代了 GST 的 attention-over-token-bank。
- **[[StyleTransferinTTS]]** [待确认] 将风格控制策略分为 style tagging / reference prompt / NL description / instruction-guided 四类。UMETTS 的 multimodal prompt 方案横跨 reference prompt 和 NL description 两类: 它同时接受音频参考、视觉参考和文本 prompt。
- **[[ProsodyModeling]]** (confirmed) 区分了显式韵律建模(FastSpeech 2 的 variance adaptor)和隐式建模(VAE/flow/reference encoder)。UMETTS 的 EMI-TTS 模块在 FastSpeech2 variant 中通过 Conditional Cross-Attention 将 emotion embedding 注入 Duration Predictor 和 Mel Decoder,属于条件化隐式控制韵律的路线。
- **[[SpeakerEmbedding]]** (confirmed) 总结了 speaker embedding 的注入方式(concatenation, addition, conditional LN, cross-attention, prefix)。UMETTS 的所有 TTS variant 都采用 concatenation 方式: $h_{lg}^{emo} = \text{Concat}(h_{lg}, u_{emo}, u_{spk})$ [§II-B],将 speaker embedding 与 emotion embedding 拼接后送入 decoder。
- **[[NaturalLanguageDescriptionforTTS]]** [待确认] 梳理了从 PromptTTS 到 InstructTTS 的 NL 控制演进。UMETTS 的文本 prompt 更接近简短的 emotion label prompt(如"A person speaking with a feeling of happy"),而非自由格式的风格描述。

**已有认知与创新判断**: 现有 KB 中的 E-TTS 方法要么依赖单一模态参考(GST 的音频参考、PromptTTS 的文本描述),要么需要显式 emotion label。UMETTS 的核心创新是 EP-Align 模块将**视觉**模态引入 E-TTS(从图像/视频中提取情感线索),这在 KB 中尚无先例。但其 TTS 后端(VITS/FastSpeech2/Tacotron2 的 emotion conditioning 变体)并非新架构,而是在已有模型上的条件化扩展。

> 检索命中: [[ProsodyModeling]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[EmotionControlinTTS]](pending-review), [[GlobalStyleTokens]](pending-review), [[StyleTransferinTTS]](pending-review), [[NaturalLanguageDescriptionforTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过对比学习将视觉/音频/文本三模态情感特征对齐为统一 emotion embedding,注入多种 TTS 后端实现多模态 prompt 驱动的情感语音合成
> - **路线**: 多模态输入(图像/视频/音频/文本 prompt) → EP-Align(对比学习对齐) → emotion embedding → EMI-TTS(VITS/FastSpeech2/Tacotron2 变体) → 情感语音
> - **指标**: ESD 上 FastSpeech2 变体 WER 7.35% / CER 3.07% / SECS 0.896 / ESMOS 4.37 [Table I]; MEADTTS 上 MCD 5.927 / SECS 0.890 [Table I]
> - **可借鉴**: EP-Align 的 prompt anchoring 对齐策略 — 用固定的文本 emotion prompt 作为锚点,将其他模态向锚点对齐,避免直接跨模态对齐的分布差异问题
> - **局限**: TTS 后端均为传统模型(非 LLM-based),仅在 ESD/MEAD 小规模数据集上验证; 情感类别有限(基本情感),未覆盖细粒度/混合情感; 论文实验中 MM-TTS 基线因代码不可用仅用 demo 评估,对比公平性存疑

## 核心问题

UMETTS 试图解决 Emotional TTS 的两个核心限制:

1. **单模态情感输入的表达瓶颈**: 传统 E-TTS 依赖离散 emotion label(过于简化)或单一模态参考音频(GST 方案,speaker/emotion 纠缠) [§I]
2. **跨模态情感一致性**: 如何从不同模态(视觉、音频、文本)中提取情感信息并统一表示,使 TTS 系统能从任意模态接收情感线索 [§II-A]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UMETTS 由两个解耦的模块组成 [§II, Fig 2]:

1. **EP-Align (Emotion Prompt Alignment Module)**: 从视觉/音频/文本多模态输入中提取情感特征,通过对比学习对齐到统一的 emotion embedding 空间
2. **EMI-TTS (Emotion Embedding-Induced TTS)**: 将对齐后的 emotion embedding 注入多种预训练 TTS 模型,条件化生成情感语音

两阶段训练: 先训练 EP-Align(100 epochs, 4x A100),再用 EP-Align 输出的 emotion prompt 训练 EMI-TTS [§III-A-2]。

### 关键设计选择

#### 1. Prompt Anchoring 对齐策略

**为什么用 prompt 作为锚点而不是直接对齐模态?** [论文原文] EP-Align 使用文本 emotion prompt(如"A person speaking with a feeling of happy")作为跨模态对齐的锚点 [§II-A-2]。每个模态的 encoder 提取特征后,通过 learnable projection 映射到共享空间,然后与 prompt embedding 计算 cosine similarity 并用 symmetric cross-entropy loss 对齐:

$$\mathcal{L}_{align} = -\log \frac{e^{logits}}{\sum_K e^{logits}} - \log \frac{e^{logits^T}}{\sum_K e^{logits^T}} \quad [§II-A-2, \text{Eq. 1}]$$

其中 $logits = e^t \cdot (\sigma(u^{exp}) \cdot \sigma(u^{imp})^T)$,$t$ 是可学习温度参数。

[agent 解读] 这种 prompt anchoring 策略的合理性在于: 文本 prompt 的分布相对稳定(是预定义的情感描述句),作为锚点可以减少直接对齐视觉/音频这两个分布差异很大的模态的难度。这与 CLIP 用文本作为视觉特征的对齐目标的思路一致(论文也引用了 CLIP [24])。

#### 2. 模态特定 Encoder 选择

EP-Align 使用预训练 encoder 并 fine-tune [§III-A-2]:
- **文本**: InstructERC [36] — 一个 instruction-tuning 过的情感识别 LLM
- **音频**: wav2vec 2.0 [37] — 自监督语音表征模型
- **视觉**: ViT (CLIP) [24, 38] — 视觉 Transformer

[agent 解读] 选择 InstructERC 而非通用 text encoder 是因为它已在情感识别任务上 instruction-tuned,能更好地理解情感描述文本。wav2vec 2.0 作为音频 encoder 虽然不是专用情感模型,但其预训练表征已被证明包含丰富的韵律/情感信息。

#### 3. EMI-TTS 的三种 TTS 变体

UMETTS 不绑定单一 TTS 后端,而是展示了 emotion embedding 在三种经典架构中的注入方式 [§II-B-2]:

**VITS 变体**: emotion embedding 注入到 Normalizing Flow 的 Emotional WaveNet 中:
$$h_0, h_1 = h; \quad h'_1 = AX(EWN(h_0 + u_{emo}), h_1) \quad [\text{Eq. 2}]$$
[agent 解读] 在 flow 的 affine transform 中注入情感,使情感信息影响 prior 分布的变换,理论上能改变生成语音的整体风格特征。

**FastSpeech2 变体**: 通过 Conditional Cross-Attention 融合 speaker 和 emotion 信息:
$$h_{emo} = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d}}\right) \cdot V + h \quad [\text{Eq. 3}]$$
其中 $Q$ 来自 hidden states,$K, V$ 来自条件向量 $c = [u_{emo}; u_{spk}]$。

**Tacotron2 变体**: emotion embedding 和 speaker embedding 直接拼接到 character encoder 输出,通过 Location & Emotion Sensitive Attention 生成 context vector [§II-B-2]。

[论文原文] 论文将 vocoder 也做了差异化: VITS 用原生 decoder, FastSpeech2 用 iSTFTNet, Tacotron2 用 WaveNet,均在 ESD 上 fine-tune [§III-A-2]。

### 训练策略

**两阶段训练** [§III-A-2]:

1. **Phase 1 — EP-Align 训练**: 在 MELD + MEAD + ESD + RAF-DB 四个多模态数据集上训练对比学习模块,100 epochs, 4x A100
2. **Phase 2 — EMI-TTS 训练**: 用 Phase 1 的 aligned prompt 作为条件,在 audio-text 对上训练 TTS 模型:
   - Flow-based (VITS): 200k steps
   - Transformer-based (FastSpeech2): 40k steps
   - Recurrent-based (Tacotron2): 250k steps

[agent 解读] 两阶段解耦训练的好处是 EP-Align 可以利用多模态数据(包括纯图像/视频数据集如 RAF-DB),而 EMI-TTS 只需 audio-text 对。但这也意味着推理时 EP-Align 的质量直接决定了情感合成的上限。

## 实验

| 指标 | 本文 (最优变体) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER | 7.35% (FastSpeech) | 7.91% (EmoSpeech) | ESD | [Table I] |
| CER | 3.07% (FastSpeech) | 3.52% (EmoSpeech) | ESD | [Table I] |
| MCD | 6.258 (VITS) | 6.285 (VITS-Label) | ESD | [Table I] |
| SECS | 0.896 (FastSpeech) | 0.841 (VITS-Label/EmoSpeech) | ESD | [Table I] |
| ESMOS | 4.37 (FastSpeech) | 4.22 (EmoSpeech) | ESD | [Table I] |
| SNMOS | 4.36 (VITS) | 4.09 (VITS-Label) | ESD | [Table I] |
| SSMOS | 4.21 (VITS) | 4.09 (VITS-Label) | ESD | [Table I] |
| MCD | 5.927 (ours) | 6.69 (MM-TTS OOD) | MEADTTS | [Table I] |
| SECS | 0.890 (ours) | 0.728 (MM-TTS OOD) | MEADTTS | [Table I] |
| ESMOS | 4.30 (ours) | 4.25 (MM-TTS OOD) | MEADTTS | [Table I] |
| EP-Align F1 | 0.75 (text+audio+image+EP-Align) | 0.59 (text only) | MELD | [Table II] |

**关键发现**:

1. **Aligned Prompt > Emotion Label**: 在 VITS 变体上,使用 EP-Align 的 prompt 比离散 emotion label 在 WER (9.61 vs 10.82)、ESMOS (4.02 vs 3.80)、SNMOS (4.36 vs 4.09) 上均有提升 [Table I]。这表明多模态对齐的 emotion embedding 比离散标签携带更丰富的情感信息 [论文原文]。

2. **FastSpeech2 变体在语言指标上最优**: WER 7.35%、CER 3.07% 最低,同时 SECS 0.896 最高 [Table I]。[agent 解读] Cross-attention 机制可能比 concatenation 更好地保持了语言内容的完整性。

3. **EP-Align 对情感分类的增益**: 加入 EP-Align 后,多模态情感分类 F1 从 0.68 (text+audio+image 无对齐) 提升到 0.75 [Table II]。

4. **MM-TTS 对比的局限**: 论文承认 MM-TTS 的源代码和完整生成样本不可用,仅用 demo page 样本评估,公平性存疑 [Table I 脚注]。

## 局限性

1. **TTS 后端陈旧**: 所有三个 TTS 变体(VITS/FastSpeech2/Tacotron2)均为 2021 年及之前的模型,未使用 LLM-based TTS(如 VALL-E、CosyVoice)。在当前 LLM-TTS 时代,这些后端的表达力和泛化能力已落后 [agent 解读]。

2. **数据集规模有限**: ESD 仅 17,500 utterances,MEADTTS 是 MEAD 的子集。未在大规模 in-the-wild 数据上验证。

3. **情感粒度粗**: 仅覆盖基本情感类别(joy, surprise, sad 等),未涉及细粒度情感(讽刺、犹豫)或混合情感。

4. **视觉模态的实际意义待验证**: 虽然 EP-Align 支持从图像/视频提取情感,但实际应用场景中是否有足够的视觉情感 prompt 需求不明确。论文未对视觉 prompt 单独做详细的消融分析 — Table II 只展示了模态组合对分类 F1 的影响,未展示视觉 prompt 对最终 TTS 质量的独立贡献。

5. **推理流程复杂**: 需要先通过 EP-Align 编码 → 选择最高相似度的 emotion embedding → 再送入 EMI-TTS,整个推理 pipeline 比直接用 emotion label 复杂。

## 点评

UMETTS 的核心 idea — 用对比学习统一多模态情感表示 — 方向正确且有实际意义。特别是 prompt anchoring 策略(用文本 prompt 作为跨模态对齐锚点)是一个简洁有效的设计,避免了直接对齐异构模态的困难。EP-Align 模块本身可以脱离 TTS 独立使用于情感分类/检索,具有通用性。

但论文的主要弱点在于 TTS 后端的选择。在 2024 年仍然使用 VITS/FastSpeech2/Tacotron2 作为后端,未与 LLM-based TTS 系统(如 VALL-E、CosyVoice)集成,限制了方法的实际影响力。此外,实验仅在小规模平行情感数据集上进行,泛化到 in-the-wild 场景的能力未知。

从 KB 角度看,UMETTS 填补了 [[EmotionControlinTTS]] 中"多模态 prompt → 情感合成"这一空白路线。其 EP-Align 模块的设计理念(跨模态对比对齐 + prompt anchoring)比具体的 TTS 集成更有长期价值,可以迁移到任何条件生成任务中。

## 可复用的 idea

1. **Prompt Anchoring 对齐**: 用文本 prompt 作为多模态对齐的锚点,解决异构模态直接对齐的分布差异问题。可迁移到任何需要跨模态条件生成的场景(如 image-to-speech, video-to-speech)。

2. **解耦对齐与生成**: EP-Align 和 EMI-TTS 的两阶段训练思路 — 先在多模态数据上训练条件提取模块,再将提取的条件注入已有生成模型。这种即插即用的设计可以复用到其他 controllable TTS 场景。

3. **Conditional Cross-Attention 注入情感**: FastSpeech2 变体中用 cross-attention($Q$ from hidden, $K/V$ from condition)注入条件信息的方式,效果优于简单 concatenation(从 SECS 和 WER 指标看)。这种方式值得在其他条件化 TTS 模型中尝试。

> [!review] 审阅 (2026-06-03, auto, checklist v1.1)
> **结论: pass** | issues: 0 high, 0 medium, 1 low
> - (low) datasets frontmatter 为空(论文用 ESD/MELD/MEAD/RAF-DB,均无 vault 实体页)
> 详见 `_review/UMETTS-review.yml`
