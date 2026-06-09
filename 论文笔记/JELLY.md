---
type: paper
tier: deep
title: "JELLY: Joint Emotion Recognition and Context Reasoning with LLMs for Conversational Speech Synthesis"
arxiv_id: "2501.04904"
source: "Sources/JELLY.pdf"
authors: [Jun-Hyeok Cha, Seung-Bin Kim, Hyung-Seok Oh, Seong-Whan Lee]
year: 2025
venue: "ICASSP 2025"
tags: [CSS, emotion, LLM, LoRA, Q-former, Whisper, conversational, context-reasoning, multi-stage]
concepts: ["[[EmotionControlinTTS]]", "[[ModalityAdaptationforSpeechLLM]]", "[[ProsodyModeling]]"]
models: []
tasks: []
datasets: ["DailyTalk", "DailyDialog", "CREMA-D", "EmoV-DB", "IEMOCAP", "MEAD", "TESS"]
kb_context_sources: 6
status: draft
created: 2026-06-09
updated: 2026-06-09
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[ProsodyModeling]], [[LLM-basedTTS]]; 4 个待确认页参考)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓, [[EmotionControlinTTS]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review), [[Full-duplexSpokenDialogue]](pending-review) | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: JELLY 处于 CSS (Conversational Speech Synthesis) + 情感推理的交叉区域。从情感控制演进看,它属于 "Emotion Embedding" 阶段的变体 — 但不同于直接从音频或标签编码情感向量 (EmoSphere-TTS, EmoCtrl-TTS 等后续工作),JELLY 通过 LLM 推理对话上下文中的情感状态,再传递给 TTS 后端。从 Speech-LLM 集成看,它属于 latent-representation-based 路线 (Q-former + PLoRA 适配 LLM)。

**已有认知**: 概念库中 [[EmotionControlinTTS]] [待确认] 已梳理了情感控制从 embedding → 层级建模 → DPO → steering 的完整演进,JELLY 定位在"LLM context reasoning"这一分支上;[[ModalityAdaptationforSpeechLLM]] [待确认] 系统对比了 Q-former > CTC > Conv 三种适配方法,JELLY 的 EQ-former 正是 Q-former 的特化应用。

**创新判断**: JELLY 的核心贡献在于将 CSS 中的情感推理从"基于图/注意力的特征融合" (ECSS, GRU-based) 升级为"LLM + 多 LoRA 的推理范式",并用三阶段训练缓解情感对话数据稀缺问题。但其 TTS 后端 (FastSpeech 2) 较为过时,且 DailyTalk 规模小 (20h),整体系统的工业应用价值有限。

## 速查

> [!summary] 速查
> - **一句话**: 用 LLM (Vicuna-7B) + 多 PLoRA + EQ-former 实现对话中的情感状态推理,再驱动 FastSpeech 2 合成情感语音
> - **路线**: 对话音频 → Whisper+TLTR+Q-former(EQ-former) → LLM with PLoRA-E/PLoRA-T → 情感/强度预测 → FastSpeech 2 + emotion/intensity encoder → 语音
> - **指标**: E-DMOS 3.987 (vs ECSS 3.914, GT 4.063); ECA 58.60% (vs ECSS 55.72%); Emotion F1 60.17 (vs ECSS 13.66) [Table I, II, DailyTalk]
> - **可借鉴**: (1) PLoRA 分流设计 — 对不同模态输入用独立 LoRA,保持 LLM 文本能力; (2) 三阶段渐进训练缓解数据稀缺; (3) EQ-former 的 TLTR 层选择机制提取情感信息
> - **局限**: TTS 后端是 FastSpeech 2 (非 SOTA); DailyTalk 仅 20h/2 说话人; 不处理重叠语音; 推理时依赖 Whisper ASR 获取 transcript

## 核心问题

1. **CSS 中为何需要情感?** 同样的对话文本在不同情感上下文下应产生不同韵律 [Fig 1] — 情感错位会导致不自然感,但现有 CSS 方法要么忽略情感,要么依赖人工标注的 ground-truth emotion label [§I]
2. **数据稀缺如何克服?** 高质量的情感对话语音数据集极少,直接端到端训练不现实 [§I, §II-C]
3. **如何在推理时不依赖文本和情感标签?** 实际场景中对话历史只有音频,没有 transcript 和 emotion label [§I]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

JELLY 是一个三阶段 pipeline [Fig 2]:

```
Stage 1: Emotion-Text Alignment
  Speech → Whisper(frozen) → TLTR → Q-former → Projection → LLM(frozen)+PLoRA → emotion label

Stage 2: Emotional Context Reasoning  
  Dialogue history (speech+text) → EQ-former(from S1) + Tokenizer → LLM+PLoRA-E+PLoRA-T → target emotion/intensity

Stage 3: Emotional Context-Aware Synthesis
  Target text + predicted emotion + intensity + speaker → FastSpeech2 + HiFi-GAN → speech
```

三个阶段训练分离,推理时 Stage 2 → Stage 3 级联 [§II-C.3]。

### 关键设计选择

**1. EQ-former (Emotion-aware Q-former Encoder)**

WHY: 需要从语音中提取情感信息,并对齐到 LLM 的文本 embedding 空间,使 LLM 能"感知"情感 [论文原文, §II-A]。

HOW:
- **Whisper Large v3 encoder** (frozen): 32 层中间表征包含丰富的副语言信息 [§II-A, 引用 Whisper-AT]
- **TLTR (Time and Layer-Wise Transformer)**: 对 32 层 Whisper 表征施加注意力机制,让模型自动聚焦情感信息更丰富的层 [§II-A]
- **Q-former**: 25 个可学习 query token (dim=768),通过 cross-attention 从 TLTR 输出中提取固定长度的情感表征 [§III-B]
- **Linear projection**: 将 query 输出映射到 LLM embedding 维度 [§II-A]

[agent 解读] TLTR 的设计动机是 Whisper 不同层编码不同信息 — 底层偏声学/音素,中层偏韵律/情感,高层偏语义。与 ModalityAdaptationforSpeechLLM 中标准 Q-former 的区别在于增加了 TLTR 层选择。

**2. PLoRA (Partial LoRA)**

WHY: 情感 embedding 和文本 embedding 是不同模态,用同一组 LoRA 参数处理会混淆 — PLoRA 让 LLM 按输入模态选择性激活不同适配器 [论文原文, §II-B]。

HOW:
- **PLoRA-E**: 仅当处理 EQ-former 输出的情感 embedding 时激活 [§II-B]
- **PLoRA-T**: 仅当处理 tokenizer 产生的文本 embedding 时激活 [§II-B]
- 实现: 注入到所有 LLaMA self-attention 层的 Q/V 投影,rank=8, scaling=4.0 [§III-B]
- LLM 原始参数不更新,保持文本生成能力 [论文原文, §II-B]

[agent 解读] 这一设计源自 InternLM-XComposer2 的多模态 PLoRA 策略,核心洞察是:不同模态的信号需要不同的适配路径,共享 LoRA 会导致模态间干扰。

**3. 三阶段训练缓解数据稀缺**

| 阶段 | 数据 | 目标 | 训练模块 | 冻结模块 |
|------|------|------|----------|----------|
| S1: Emotion-Text Alignment | 80.6h 情感语音 (6数据集) | EQ-former 学会提取情感 | EQ-former, PLoRA-E/T | Whisper, LLM |
| S2-PT: Context Reasoning 预训练 | DailyDialog 13k 对话 (纯文本) | PLoRA-T 学会情感推理 | PLoRA-T | 其余 |
| S2-FT: Context Reasoning 微调 | DailyTalk 20h | 联合推理 | PLoRA-E/T, EQ-former | Whisper, TLTR |
| S3: Speech Synthesis | DailyTalk 20h | 情感条件下合成 | FastSpeech2 全部 | — |

[论文原文, §II-C]: Stage 1 用相对丰富的情感语音数据 (80.6h, 多数据集) 训练 EQ-former; Stage 2 先用大规模文本数据预训练 PLoRA-T (解决语音对话数据不足), 再在小规模语音数据上微调。

**4. 推理时的输入构造 [Fig 2(b)]**

每个历史 utterance uk 的 embedding Uk 由三部分拼接:
- Pk (prefix): "speaker sk (says with" [text embedding]
- Ek (emotion): EQ-former 输出 [emotion embedding]; 训练时用 GT label 的文本 embedding, 推理时用 EQ-former
- Tk (transcript): Whisper decoder 生成 [text embedding]; 训练时用 GT transcript

JELLY (speech-only): 推理时 Tk 由 Whisper decoder 从语音中自动生成,不需要外部 transcript [§II-C.2]。

### 训练策略

- 优化器: AdamW, beta=(0.9, 0.999), weight_decay=0.05 [§III-B]
- 学习率: cosine decay, peak 3e-5, warmup 3k steps, min 1e-5 [§III-B]
- 训练步数: S1 180k, S2 30k, S3 275k [§III-B]
- 硬件: 4x NVIDIA RTX A6000 [§III-B]
- Vocoder: HiFi-GAN [§III-B]
- 音频: S1/S2 16kHz; S3 22.05kHz, 80-bin mel, FFT 1024, hop 256 [§III-A]

## 实验

| 指标 | JELLY | JELLY (speech-only) | ECSS | GRU-based | FastSpeech2 | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N-DMOS ↑ | 3.847 | 3.790 | 3.802 | 3.792 | 3.788 | 3.900 | DailyTalk | [Table I] |
| E-DMOS ↑ | 3.987 | 3.983 | 3.914 | 3.944 | 3.910 | 4.063 | DailyTalk | [Table I] |
| ECA ↑ | 58.60% | 57.05% | 55.72% | 56.38% | 56.38% | 56.16% | DailyTalk | [Table I] |
| MCD ↓ | 3.217 | 3.059 | 3.361 | 3.487 | 3.313 | — | DailyTalk | [Table I] |
| DDUR ↓ | 0.2157 | 0.2194 | 0.2525 | 0.2865 | 0.2386 | — | DailyTalk | [Table I] |
| Emotion WA ↑ | 78.54% | — | 43.51% | — | — | — | DailyTalk | [Table II] |
| Emotion F1 ↑ | 60.17% | — | 13.66% | — | — | — | DailyTalk | [Table II] |
| Intensity WA ↑ | 77.21% | — | 60.38% | — | — | — | DailyTalk | [Table II] |
| Intensity F1 ↑ | 51.61% | — | 25.10% | — | — | — | DailyTalk | [Table II] |

**Ablation 核心发现 [Table II]**:
- w/o Q-former: ECA 55.38% (↓3.22), Emotion F1 暴跌至 10.43 — Q-former 是跨模态对齐的关键 [§IV-C]
- w/o PT in Stage 2: ECA 56.60% (↓2.0), Emotion WA 67.15% (↓11.4) — 文本预训练显著弥补数据稀缺 [§IV-C]
- w/o Stage 1: ECA 57.38%, 各项略降 — EQ-former 预训练有益但非决定性 [§IV-C]
- w/o TLTR: ECA 57.71%, Intensity F1 50.87 (↓0.74) — TLTR 主要帮助提取情感强度细节 [§IV-C]
- w/o PLoRA: ECA 57.05%, Intensity WA 76.33 — PLoRA 对强度建模有贡献 [§IV-C]

## 局限性

1. **TTS 后端过时**: FastSpeech 2 作为 backbone,在当前 LLM-TTS/Flow-matching 时代已非竞争力方案;合成质量受限于这一 bottleneck [agent 解读]
2. **数据规模极小**: DailyTalk 仅 20h、2541 对话、2 说话人,难以泛化到更多说话人/更复杂对话 [§III-A]
3. **不处理重叠语音**: 论文承认未考虑 overlapped speech 和 larger speaker groups [§V]
4. **GT ECA ceiling 低**: Ground Truth 的 ECA 仅 56.16% (emotion2vec plus large 在 DailyTalk 上分类准确率有限),说明评估指标本身存在 noise [Table I]
5. **推理依赖**: speech-only 模式仍需 Whisper decoder 生成 transcript,增加延迟和错误传播风险 [§II-C.2]
6. **情感类别受限**: 7 类离散情感 + 3 级强度,未覆盖复合情感或连续情感维度 [§III-A]
7. **Stage 3 独立训练**: 情感推理与语音合成分开训练再级联,无法端到端优化情感表达质量 [agent 解读]

## 点评

**定位价值**: JELLY 是 CSS 领域引入 LLM 推理能力的早期探索,展示了"用 LLM 理解对话情感上下文 + 传统 TTS 合成"这一 pipeline 的可行性。三阶段训练策略和 PLoRA 分流设计具有方法论参考价值。

**与后续工作对比**:
- 相比 [[论文笔记/DiffCSS|DiffCSS]] (diffusion-based CSS): JELLY 的创新在推理端 (LLM reasoning) 而非合成端
- 相比 [[论文笔记/EmotionThinker|EmotionThinker]]: EmotionThinker 用 CoT 显式输出情感规划,JELLY 用 LLM next-token prediction 隐式推理
- 相比后续 steering 系列 (EmoSteer-TTS, CoCoEmo): 那些方法直接在 TTS 内部操控情感表征,JELLY 则是"先推理后合成"的解耦策略

**方法论贡献 > 性能贡献**: 绝对性能提升有限 (E-DMOS +0.07 vs ECSS),但情感预测能力显著提升 (Emotion F1: 60.17 vs 13.66),验证了 LLM + Q-former 在情感推理上的优势。实际应用价值受限于 FastSpeech 2 后端和小规模数据。

## 可复用的 idea

1. **PLoRA 分流**: 对不同模态输入使用独立 LoRA adapter,避免模态干扰 — 适用于任何多模态 LLM 微调场景
2. **TLTR 层选择**: 用 attention 机制在 SSL model 的多层表征中自动选择任务相关信息 — 可迁移到其他需要从 Whisper/HuBERT 提取特定属性的任务
3. **文本预训练缓解音频数据稀缺**: 先用纯文本 (含 emotion label) 训练推理能力,再在有限语音数据上微调 — 适用于所有情感/风格语音数据稀缺的场景
4. **情感推理与合成解耦**: 将 CSS 拆分为"理解"(LLM) + "生成"(TTS) 两个独立可替换模块 — 升级合成端不影响推理端

## 审阅

> [!review] 审阅 (2026-06-09, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 三阶段 pipeline + EQ-former + PLoRA 机制清晰 |
> | 可信赖 | pass | 数字均标注出处 [Table I/II],区分了论文原文与 agent 解读 |
> | 可区分 | pass | 与 ECSS/DiffCSS/EmotionThinker 的差异明确 |
> | 可定位 | pass | KB 背景定位了方法在情感控制+模态适配双线中的位置 |
> | 不污染 | pass | 未引入未验证 claim |
> 
> Issues: 1 (medium: 0, low: 1)
> - [low] 缺少与 RADKA-CSS 等同期 CSS 工作的对比定位
> 详见 `_review/JELLY-review.yml`
