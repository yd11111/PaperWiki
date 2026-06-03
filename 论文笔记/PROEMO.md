---
type: paper
tier: deep
title: "PROEMO: Prompt-Driven Text-to-Speech Synthesis Based on Emotion and Intensity Control"
arxiv_id: "2501.06276"
source: "Sources/PROEMO.pdf"
authors: [Shaozuo Zhang, Ambuj Mehrish, Yingting Li, Soujanya Poria]
year: 2025
venue: "arXiv (cs.SD)"
tags: [TTS, emotion, intensity-control, prompt-control, LLM, prosody, multi-speaker, FastSpeech2, HuBERT, expressive-speech]
concepts: ["[[Emotion Control in TTS]]", "[[Prosody Modeling]]", "[[Non-autoregressive TTS]]", "[[Speaker Embedding]]", "[[Mel Spectrogram]]", "[[Duration Predictor]]"]
models: ["[[HuBERT]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: PROEMO 处于 Emotion Control in TTS 演进线中 "Emotion embedding + LLM prompt" 阶段,介于显式 emotion embedding (Li et al., 2021) 与 LLM 自由文本情感控制 (EmoVoice, 2025) 之间。它的 backbone 是 FastSpeech 2 ([[Non-autoregressive TTS]] 的代表),属于显式 variance adaptor 路线,与 LLM-based TTS (VALL-E, CosyVoice) 的隐式韵律建模形成对比。

**已有认知**:
- [[Prosody Modeling]] ✓ 明确定义了 prosody 的四个物理维度 (duration, pitch, energy, pause),以及显式 vs 隐式建模的分类。PROEMO 使用 FastSpeech 2 的显式 variance adaptor (pitch + energy + duration predictor),并叠加 GPT-4 prompt 推理,属于"显式 + LLM 增强"混合路线。
- [[Speaker Embedding]] ✓ 梳理了 multi-speaker TTS 的 speaker encoder 方案。PROEMO 使用 GE2E loss 训练的 speaker encoder,是标准零样本路线。
- [[Emotion Control in TTS]] [待确认] 已收录 emotion embedding、层级建模、DPO/RLHF、EmoSphere++ 球面向量等方案。PROEMO 的 HuBERT-based emotion encoder + 独立 intensity encoder 的双编码器设计是新的组合。
- [[HuBERT]] [待确认] 作为自监督语音模型,在 PROEMO 中被用于 emotion/intensity 特征提取 (冻结 CNN + 微调 Transformer),与其在 TTS tokenizer 中的常见用法不同。
- [[Non-autoregressive TTS]] [待确认] 覆盖 FastSpeech 2 的 variance adaptor 设计,PROEMO 直接在此基础上扩展。
- [[Instruction-Guided Speech Synthesis]] [待确认] 描述的是统一指令范式 (VoxInstruct, CosyVoice),PROEMO 的 prompt control 更受限 — 仅通过 GPT-4 预测 scaling factors,不是自由指令。

**创新判断**: PROEMO 的主要组合创新在于: (1) 独立 intensity encoder 通过 relative ranking function 实现无标注的强度建模; (2) 双层 (global + local) LLM prompt control 实现推理时韵律微调。但这些组件均非首创,更多是已有方法的工程集成。

> 检索命中: [[Prosody Modeling]]✓, [[Speaker Embedding]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[HuBERT]](pending-review), [[Non-autoregressive TTS]](pending-review), [[Instruction-Guided Speech Synthesis]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 FastSpeech 2 上叠加 HuBERT-based 情感/强度双编码器 + GPT-4 推理时 prompt control,实现多说话人情感 TTS 的情感类型与强度可控
> - **路线**: Text → FS2 Encoder → Variance Adaptor (pitch/energy/duration × GPT-4 scaling) + Emotion Encoder (HuBERT) + Intensity Encoder (HuBERT + ranking function) + Speaker Encoder (GE2E) → Mel Decoder → HiFiGAN → Waveform
> - **指标**: ECA 0.797 (FS2w/E&I, local PC), MOS 3.728 (FS2w/E&I, G&L PC), PIR ~72% accuracy (ESD test set) [Table 1, Fig 3]
> - **可借鉴**: 用 relative attribute ranking 从无强度标注数据学习 per-speaker 情感强度编码器,避免主观标注; GPT-4 双层 scaling factor 设计提供推理时韵律微调入口
> - **局限**: 依赖 GPT-4 API 推理 (延迟/成本/不可复现); ESD 仅 350 句 5 类情感,规模太小; MOS 3.728 未达现代系统水平; 未与 EmoSphere/EmoCtrl-TTS 等 2024 方法对比; local-only 优于 G&L 暴露全局控制可能有害

## 核心问题

PROEMO 要解决的核心问题: **如何在多说话人 TTS 中同时控制情感类型和情感强度,并在推理时通过 prompt 灵活调整韵律?**

前序工作 Sigurgeirsson & King (2023) [16] 探索了 LLM prompt 控制 TTS prosody,但存在三个不足 [§1]:
1. 不支持多说话人情感语音生成
2. 仅依赖 LLM 输出生成情感语音,LLM 输出噪声大会影响表现力
3. 未考虑情感强度维度

PROEMO 的回应: 在 FS2 上集成 emotion encoder + intensity encoder 提供稳定的情感条件,再叠加 GPT-4 prompt control 提供韵律微调。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由 4 个模块组成 [Fig 1]:

1. **TTS Backbone (FS2)**: 音素编码器 + speaker encoder + variance adaptor (pitch/energy/duration predictor) + mel decoder + HiFiGAN vocoder [§3.1]
2. **Emotion Encoder**: HuBERT feature extractor (CNN 冻结, Transformer 微调) + classification head → emotion embedding [§3.2]
3. **Intensity Encoder**: HuBERT + regression head → 连续强度值 (0-1) [§3.3]
4. **GPT-4 Prompt Control**: 推理时预测 global + local scaling factors 修改 variance adaptor 输出 [§3.4]

三个 FS2 变体:
- FS2: 纯 backbone
- FS2w/Emo: + emotion encoder
- FS2w/Emo&Int: + emotion encoder + intensity encoder

### 关键设计选择

**为什么用 HuBERT 做 emotion/intensity encoder?**
HuBERT 的自监督预训练学到了丰富的语音表征,包含情感和韵律信息 [论文原文: §3.2]。冻结 CNN encoder 保留底层声学特征,仅微调 Transformer 层和 task head 实现高效适配 [论文原文: §3.2]。

**为什么需要独立的 intensity encoder?**
大多数情感语音数据集缺乏强度标注,直接标注对人类也困难 [论文原文: §3.3, 引用 [24]]。受 Zhou et al. (2022) [23] 启发,将情感强度视为可通过相对排序学习的属性 [论文原文: §3.3]。

**Intensity 建模细节** [§3.3]:
- 定义 learned rank function: r(xA) = WxA,其中 xA 是 384 维 openSMILE 特征
- 强度归一化到 [0, 1]
- **关键改进**: 为每个说话人 A 和每个情感类别分别学习 ranking function,建模个体差异 [论文原文: §3.3]
- 训练 W 的方法类似 SVM 问题 [§3.3, 引用 [26]]
- 用学到的排序值作为 regression 训练目标监督 intensity encoder

**为什么用 GPT-4 做 prompt control 而不是训练 prosody predictor?**
[agent 解读] 直接训练一个情感→韵律的 predictor 需要大量情感-韵律对齐数据。使用 GPT-4 利用其世界知识推理 "悲伤语音应该更慢、音调更低" 这类常识,无需额外训练数据。但代价是推理时依赖外部 API。

**Prompt Control 双层机制** [§3.4, Eq. 1-3]:
- **Global**: 整句级 scaling factor Gd, Ge, Gp,建立整体情感基调
- **Local**: 词级 scaling factor σi, ϵi, πi,细化每个词的韵律
- Duration: d'i = di · Gd · σi,范围 [0.74, 1.34] (经验上 duration 更敏感) [论文原文: §3.4]
- Energy: e'i = ei · Ge · ϵi,范围 [0.5, 2]
- Pitch: p'i = pi + Gp + πi (加性修改,非乘性)
- GPT-4 输出 [-5, 5] (pitch/energy) 或 [-2, 2] (duration) 的值,通过二次映射 f(x) = ax² + bx + c 映射到约束范围

**Prompt 设计演进** [§3.4]:
最初使用 [16] 的 prompt,但 GPT-4 输出不稳定 — pitch/energy/duration 变化一致导致表现力差 [论文原文: §3.4]。于是设计综合 prompt 模板,覆盖任务描述和输出要求,并鼓励每步推理以减少噪声 [论文原文: §3.4]。

### 训练策略

**两阶段训练** [§4.2]:
1. **预训练**: 多说话人 FS2 在 LibriTTS-100 上训练 900K steps, batch size 16, speaker embedding 用 GE2E loss 的 speaker verification model 计算 [§4.2]
2. **微调**: 在 ESD 上加入 emotion + intensity encoder,HuBERT feature extractor 冻结,仅更新 Transformer + classification/regression head, 50K steps [§4.2]

**推理**: Prompt control 仅在推理时生效,通过 GPT-4 API 获取 scaling factors [§4.2]

## 实验

| 指标 | 本文 (FS2w/E&I, L) | 本文 (FS2w/E&I, G&L) | FS2w/E (L) | Daft-Exprt (None) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| ECA ↑ | 0.797 | 0.771 | **0.820** | 0.663 | ESD test | [Table 1] |
| MCD ↓ | 6.741 | 7.058 | 6.767 | 6.278 | ESD test | [Table 1] |
| WER ↓ | 0.128 | 0.130 | 0.129 | 0.353 | ESD test | [Table 1] |
| CER ↓ | 0.040 | 0.043 | 0.039 | 0.123 | ESD test | [Table 1] |
| MOS ↑ | 3.408 (L) | **3.728** (G&L) | — | 3.450 | ESD test | [Table 1] |
| PIR accuracy | ~72% | — | — | — | ESD eval | [Fig 3] |

**关键发现**:
1. Local-only prompt control 在 ECA 上始终优于 G&L control [论文原文: §5.1] — 这是出乎意料的,因为设计初衷是 G&L 更完整
2. Emotion/intensity embedding 不显著损害语音可懂度 (WER/CER 变化很小) [论文原文: §5.1]
3. Daft-Exprt 的 WER 高达 0.353 是因为仅在单说话人 LJSpeech 上预训练,泛化差 [论文原文: §5.1]
4. MOS 测试中 FS2w/E&I with G&L 最高 (3.728),但仅 20 名评估者,且低于现代 TTS 系统通常的 4.0+ MOS [Table 1]
5. PIR ~72% 表明强度控制有效但不精确 [Fig 3]

## 局限性

1. **GPT-4 依赖**: 推理时必须调用 GPT-4 API,引入延迟、成本和不可复现性 [agent 解读]
2. **评估规模不足**: ESD 仅 350 句/10 说话人/5 情感,远小于现代数据集 (Emilia 101K+ hours); MOS 仅 20 名评估者 [§4.1, §5.2]
3. **Baseline 选择过时**: Daft-Exprt (2021) 和 vanilla FS2 作为 baseline,未与 2023-2024 的 EmoSphere-TTS、EmoCtrl-TTS、Daisy-TTS 等同期工作对比 [agent 解读]
4. **Local > G&L 矛盾**: 实验显示 local-only prompt control 在 ECA 上优于 global+local,暗示全局 scaling 可能引入噪声而非改善 — 这与双层设计的动机矛盾 [Table 1]
5. **仅英语评估**: 未验证跨语言泛化能力 [agent 解读]
6. **HiFiGAN vocoder**: 已非 SOTA 声码器,可能限制音质上限 [agent 解读]
7. **Intensity 评估间接**: PIR 测试的 ground truth 来自同一 learned rank function,存在循环论证风险 [agent 解读]

## 点评

PROEMO 尝试用模块化方式 (emotion encoder + intensity encoder + LLM prompt) 解决情感 TTS 的两个具体问题: 情感类型控制和强度控制。其中 **per-speaker, per-emotion 的 intensity ranking function** 是最有新意的设计 — 它巧妙绕过了强度标注的难题,利用相对排序学习连续强度表征。

但从整体来看,本文更像是一个 **工程组合**而非方法论创新:
- Backbone (FS2) 是 2020 年的设计
- Emotion encoder 是标准的 HuBERT + classification head
- Prompt control 直接继承 Sigurgeirsson & King (2023) 的范式
- 仅在推理时叠加 GPT-4,训练和模型本身未受 LLM 影响

**与已有方法的定位**: 在 [[Emotion Control in TTS]] 的演进线中,PROEMO 处于 "embedding + LLM" 的过渡位置。相比 Daisy-TTS 的 PCA 分解方法和 EmoSphere-TTS 的球面表征,PROEMO 对情感的建模更简单直接。相比 EmoCtrl-TTS 的帧级 arousal-valence 连续控制,PROEMO 的句级/词级 scaling factor 控制粒度更粗。

**实验说服力不足**: MOS 3.728、仅 20 名评估者、与过时 baseline 对比、ESD 小数据集 — 这些都限制了结论的可信度。Local > G&L 的结果实际上削弱了双层设计的价值主张。

## 可复用的 idea

1. **Relative attribute ranking for intensity**: 用 r(xA) = WxA 从无标注数据学习 per-speaker 情感强度,避免主观标注问题。可推广到其他难以直接标注的连续属性 (如说话人独特性、口音强度)。
2. **Constrained LLM scaling factors**: 让 LLM 输出有界整数 → 二次映射到约束范围,既利用 LLM 推理又防止输出失控。可用于任何需要 LLM 预测连续控制量的场景。
3. **Duration sensitivity insight**: Duration 比 energy 更敏感 (范围 [0.74, 1.34] vs [0.5, 2]),暗示韵律修改中 duration 需更保守。这对任何涉及 prosody editing 的系统都有参考价值。

> [!review] 审阅
> 审阅结论: **pass-with-fixes**
> 详见 [[_review/PROEMO-review.yml]]
