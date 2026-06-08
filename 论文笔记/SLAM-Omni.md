---
type: paper
tier: deep
title: "SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training"
arxiv_id: "2412.15649"
source: "arXiv"
authors: [Wenxi Chen, Ziyang Ma, Ruiqi Yan, Yuzhe Liang, Xiquan Li, Ruiyang Xu, Zhikang Niu, Yanqiao Zhu, Yifan Yang, Zhanxun Liu, Kai Yu, Yuxuan Hu, Jinyu Li, Yan Lu, Shujie Liu, Xie Chen]
year: 2024
venue: "arXiv"
tags: [spoken-dialogue, speech-LM, semantic-token, zero-shot-timbre, single-stage-training, parallel-generation, group-modeling]
concepts: ["[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/SemanticvsAcousticTokens|Semantic vs Acoustic Tokens]]", "[[概念库/ConditionalFlowMatching|Conditional Flow Matching]]", "[[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation for Speech LLM]]", "[[概念库/StreamingSpokenDialogue|Streaming Spoken Dialogue]]", "[[概念库/SpokenDialogueEvaluation|Spoken Dialogue Evaluation]]", "[[概念库/SpeechTokenizer|Speech Tokenizer]]"]
models: ["SLAM-Omni", "Mini-Omni", "Mini-Omni2", "Freeze-Omni", "LLaMA-Omni", "GLM-4-Voice"]
tasks: ["spoken-dialogue"]
datasets: ["VoiceAssistant-400K", "UltraChat", "Belle_train_3.5M_CN"]
kb_context_sources: 7
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 7 个实体页: SpeechLanguageModel, SemanticvsAcousticTokens, ConditionalFlowMatching, ModalityAdaptationforSpeechLLM, StreamingSpokenDialogue, SpokenDialogueEvaluation, SpeechTokenizer)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: 7 | 过滤: 0 | 未命中但可能相关: Full-duplexSpokenDialogue

**SLAM-Omni 在 KB 中的定位:**

SLAM-Omni 属于 [[概念库/SpeechLanguageModel|Speech Language Model]] 中 **parallel audio-text modeling** 范式 (Survey Fig. 1c),与 Mini-Omni、PSLM 同一范式 [§2.1]。在 [[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLM Integration Taxonomy]] (Yang et al. 2025) 中,输入侧使用 Whisper encoder 连续表征属于 latent-representation-based integration,输出侧预测 semantic tokens 属于 audio-token-based integration,是**混合集成路线**。

在 [[概念库/SemanticvsAcousticTokens|Semantic vs Acoustic Tokens]] 谱系上,SLAM-Omni 选择了 CosyVoice 的监督式 semantic tokens (Du et al., 2024) 而非 HuBERT 自监督 semantic tokens 或 EnCodec/SNAC acoustic tokens。KB 已记录这一选择的核心 trade-off: semantic tokens 与文本对齐好但缺声学细节,声学细节交由 [[概念库/ConditionalFlowMatching|CFM]] vocoder 恢复。CosyVoice 已验证这一方案在 TTS 场景的有效性 ([[论文笔记/CosyVoice|CosyVoice]]),SLAM-Omni 是**首次将其扩展到 spoken dialogue** 场景。

输入侧的 Whisper encoder + linear projector 是 [[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation]] 中最简单的 convolutional downsampling 方案。KB 记录 Q-Former > CTC > Conv 的性能排序,但 SLAM-Omni 优先选择了训练效率而非理论最优适配器。

在 [[概念库/StreamingSpokenDialogue|Streaming Spoken Dialogue]] 体系中,SLAM-Omni 的 semantic group modeling (一步预测 G 个 tokens) 与 IntrinsicVoice 的 GroupFormer 思路相似,但 SLAM-Omni 用更简单的线性层而非额外 transformer 实现分组预测。

## 速查

> [!summary] 速查
> - **一句话**: 用 CosyVoice 监督 semantic tokens 将 speaker 信息解耦到 vocoder,结合 semantic group modeling 压缩音频序列、historical text prompting 压缩对话历史,实现了首个单阶段训练的音色可控端到端语音对话系统
> - **路线**: Whisper encoder → Linear projector → Qwen2-0.5B (parallel text + grouped semantic tokens) → CosyVoice CFM vocoder + speaker prompt → HiFi-GAN → waveform
> - **指标**: ChatGPT Score 39.32, UTMOS 4.45 (best among all SDMs), ASR-WER 4.54% (best), 60 GPU hours on 4x A100 [Table 3, 4]
> - **可借鉴**: (1) Semantic group modeling: 一步预测 G=3 个 semantic tokens,ASR-WER 从 18.23% 降至 4.54%,GPU 时间从 126h 降至 60h [Table 5]; (2) Historical text prompting: 用 text-only 历史替代 audio-text 交替历史,压缩上下文长度,继承 LLM 文本 in-context learning; (3) Single-stage training 反而优于 ASR/TTS 预训练,因为预训练损害指令遵循能力 [Table 6]
> - **局限**: (1) text-only 历史丢失副语言信息; (2) 仅在 0.5B LLM 验证,scale-up 未知; (3) ChatGPT Score 仍明显低于同规模 text LLM (Qwen2-0.5B-instruct: 57.70 vs 39.32) [Table 3]; (4) 训练数据为 CosyVoice 合成语音,非真实对话

## 核心问题

当前 spoken dialogue models (SDMs) 存在三个未解决问题 [§1]:

1. **音色不可控**: 现有 SDMs 无法生成多样化说话人音色的响应,因训练数据音色单一且缺少显式 speaker modeling [论文原文]
2. **audio-text 频率失配**: semantic tokens (~50Hz) 和 text tokens (~3Hz) 的频率差约 17 倍,导致训练和推理成本高、speech-text alignment 差 [论文原文]
3. **多轮对话历史膨胀**: audio-text 交替的对话历史序列过长,限制了可支持的对话轮次并降低 in-context learning 能力 [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SLAM-Omni 采用 encoder-LLM-vocoder 三段式架构,属于 parallel audio-text modeling 范式 (Fig. 2) [§3.1]:

**输入侧**: Whisper-small encoder 提取 50Hz 语音特征 → 每 k=5 帧拼接下采样为 10Hz → Linear projector 投影到 LLM embedding 空间 → 与 system prompt + historical text prompt 拼接输入 LLM [§3.2]

**输出侧**: Qwen2-0.5B LLM 在每个 step 同时输出 text logits L_t 和 audio logits L_a,L_a 经线性层扩展为 G=3 个 semantic token 的分组预测 [§3.3]。Text 和 audio 共享同一 autoregressive decoding 过程,但各自有独立的 loss 和独立的 vocabulary (V_j = V_t ∪ V_a) [§3.3]。

**合成侧**: CosyVoice 的 CFM (conditional flow matching) 将 semantic tokens + speaker prompt 转为 mel spectrogram → HiFi-GAN 合成波形。Block causal attention 用于 CFM Transformer 以支持实时生成 [§3.4]。

### 关键设计选择

**选择 1: 监督 semantic tokens (CosyVoice) 而非 acoustic tokens (EnCodec/SNAC)**

[论文原文] 监督 semantic tokens 天然将 speaker 信息与语义内容解耦 — token 只编码"说了什么",音色由 vocoder 从 speaker prompt 恢复 [§3.4]。这使 SDM 首次获得零样本音色控制能力,用户提供一段参考音频即可改变系统的输出声音。

[agent 解读] 这一选择本质上是将 CosyVoice TTS 的架构设计直接搬到 dialogue 场景: CosyVoice 已在 zero-shot TTS 中验证了"semantic token + CFM vocoder"的音色控制方案,SLAM-Omni 只需把 LLM 生成 semantic tokens 的上游从 text prompt 换成 speech input + dialogue context。

**选择 2: Semantic group modeling (G=3)**

[论文原文] 音频 semantic tokens 频率 (~50Hz) 远高于 text tokens (~3Hz),频率失配导致序列长度差异大、speech-text alignment 差、训练推理成本高 [§3.3]。Semantic group modeling 在每步预测 G 个 semantic tokens,将音频序列长度压缩 G 倍,灵感来自 VALL-E 2 的分组编码 [§3.3]。

具体实现: LLM 输出的 audio logits L_a (|V_a| 维) 通过一个 linear layer 扩展为 L_g (|V_a| × G 维),即一步预测 G 个独立的 token 分布 [§3.3, Eq. 1-4]。训练时 text 和 audio loss 加权求和 (λ_text = λ_audio = 1) [§4.2]。

[agent 解读] 分组预测的 G=3 让 audio token 的有效频率从 50Hz 降至 ~17Hz,与 text token 的 ~3Hz 差距缩小,这既减少了 LLM 的序列建模负担,也让 text 和 audio 在相似时间尺度上对齐。这解释了为什么 G=1 时 ASR-WER 高达 18.23% 而 G=3 降至 4.54% — 频率越接近,两个 stream 的信息同步越好。

**选择 3: Historical text prompting**

[论文原文] 传统多轮对话模型交替存储 audio 和 text tokens 作为历史,但 audio token 序列过长导致: (1) 训练计算成本高,(2) 限制可支持的对话轮次,(3) 长历史妨碍 in-context learning 且增加遗忘风险 [§3.5]。

解决方案: 仅用 text 模态表示对话历史。输入结构为 <System> <History (text-only)> <Input (speech embedding)> <Answer> [§3.5, Fig. 2]。推理时,用户的语音通过 Whisper decoder 得到文本转写,模型的响应从 text stream 直接获取,两者追加到 text 历史用于下一轮 [§3.5]。KV cache 跨轮复用 (Fig. 4) [§3.5]。

[论文原文] 这一设计自然继承了 LLM 基于文本的 in-context learning 能力,且消除了长音频序列的上下文负担 [§3.5]。

[agent 解读] 代价是丢失了历史中的副语言信息 (情感、韵律、语调),论文自己在 Limitations 中承认了这一局限 [§Limitations]。对于纯信息交换型对话足够,但在需要情感连续性的场景 (如情感陪伴) 可能不够。

### 训练策略

**单阶段全量微调**: Whisper encoder 冻结,Qwen2-0.5B 全量微调。AdamW, lr=1e-4, batch=24, 100K steps (warmup 1K + linear decay), 约 15h on 4x A100 [§4.2]。

**训练数据合成**: 由于公开的语音对话数据集极为稀缺,SLAM-Omni 使用 CosyVoice 合成全部训练数据 [§4.1]:
- 用户输入: CosyVoice-300M 从文本生成语音,音色从 1007 英/1010 中 speaker prompt 随机采样
- 模型回复: CosyVoice-300M-SFT 的 text-to-token LLM 直接生成 semantic tokens (非波形),作为训练 target
- 主实验仅用 VoiceAssistant-400K (460K samples, ~664h 指令 + ~3234h 回复) [Table 1]

[论文原文] 单阶段训练优于 ASR/TTS 预训练 + 对话微调的多阶段方案。原因: ASR 或 TTS 预训练虽略提升 audio-text alignment (ASR-WER 4.38% vs 4.54%),但损害指令遵循和通用知识保留,导致 ChatGPT Score 从 39.32 降至 34.02 (ASR pre) 和 27.22 (TTS pre) [§5.3.2, Table 6]。

[agent 解读] 这一结论与 Qwen2-0.5B 这样的小模型高度相关 — 小模型的容量有限,单任务预训练更容易造成灾难性遗忘。在 7B+ 模型上结论可能不同 (GLM-4-Voice 等大模型就采用多阶段训练)。

## 实验

### 主要结果 (0.5B 模型, VoiceAssistant-400K)

| 指标 | SLAM-Omni | Mini-Omni | Mini-Omni2 | Freeze-Omni (7B) | LLaMA-Omni (8B) | GLM-4-Voice (9B) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ChatGPT Score (overall) | 39.32 | 22.58 | 26.56 | 56.72 | 57.68 | 74.44 | 8 datasets | Table 3 |
| UTMOS | **4.45** | 4.42 | 4.43 | 4.37 | 4.02 | 4.15 | 8 datasets | Table 4 |
| ASR-WER | **4.54%** | 6.05% | 10.24% | 16.32% | 10.42% | 12.71% | 8 datasets | Table 4 |
| Repeat (understanding) | 12.26 | 5.07 | 8.10 | 70.89 | 45.62 | 90.95 | Repeat | Table 3 |
| AlpacaEval (conversation) | 48.98 | 30.99 | 34.81 | 52.23 | 64.36 | 80.77 | AlpacaEval | Table 3 |

### Group Size 消融 [Table 5]

| G | ChatGPT Score | UTMOS | ASR-WER | GPU Hours |
| --- | --- | --- | --- | --- |
| 1 | 34.17 | 4.44 | 18.23% | 126 |
| 2 | 35.22 | 4.46 | 8.00% | 78 |
| **3** | **39.32** | **4.45** | **4.54%** | **60** |
| 4 | 37.19 | 4.45 | 4.31% | 52 |
| 5 | 33.93 | 4.43 | 4.85% | 50 |

### 训练策略消融 [Table 6]

| Setting | ChatGPT Score | UTMOS | ASR-WER | GPU Hours |
| --- | --- | --- | --- | --- |
| SLAM-Omni (single-stage) | **39.32** | 4.45 | 4.54% | **60** |
| + ASR pre-training | 34.02 | 4.45 | 4.38% | 132 |
| + TTS pre-training | 27.22 | 4.46 | 4.53% | 160 |

### 多轮对话 (+ UltraChat 300K) [Table 9]

| 模型 | ChatGPT Score | UTMOS | ASR-WER |
| --- | --- | --- | --- |
| GLM-4-Voice (9B) | 68.35 | - | - |
| Qwen2-0.5B-Instruct (text) | 59.12 | - | - |
| SLAM-Omni (0.5B) | 32.88 | 4.45 | 7.61% |

### 中文语音对话 (+ Belle 1.4M) [Table 10, 12]

| 模型 | ChatGPT Score | UTMOS | ASR-CER |
| --- | --- | --- | --- |
| GLM-4-Voice (9B) | 67.59 | 3.09 | 4.5% |
| Freeze-Omni (7B) | 35.34 | 3.61 | 6.3% |
| SLAM-Omni (0.5B) | 25.12 | **3.67** | **4.4%** |

## 局限性

1. **Historical text prompting 丢失副语言信息**: 仅保留文本历史意味着前几轮对话中的情感、韵律、语调等信息完全丢失。作者承认"在某些场景中,保留这些历史上下文对维持对话连贯性和深度至关重要" [§Limitations]

2. **仅验证 0.5B LLM**: 作者指出 joint audio-text modeling 在大规模 LLM 上需要"显著更多的训练数据",如何平衡 audio-text 联合建模效率与保留 LLM 原有知识仍是开放问题 [§Limitations]

3. **合成训练数据**: 所有训练数据由 CosyVoice TTS 合成,非真实对话语音。合成数据可能无法覆盖真实对话中的停顿、犹豫、重叠等自然现象 [agent 解读]

4. **ChatGPT Score 显著低于 text LLM**: SLAM-Omni (39.32) vs Qwen2-0.5B-instruct (57.70),说明语音模态引入后 LLM 的文本能力有显著退化 [Table 3] [agent 解读]

5. **非流式评估**: 所有实验使用 non-streaming decoding 评估,未报告流式场景下的延迟和质量 [§4.2]

6. **评估体系完整性**: 8 个评估数据集均为作者自建或改编,且 ChatGPT Score 依赖 Whisper-large-v3 转写 + GPT-4o mini 评分,非端到端 speech-to-speech 评估 [agent 解读]

## 点评

**优势**:

SLAM-Omni 的核心贡献是将 CosyVoice 的 semantic token + CFM vocoder 方案从 TTS 扩展到 spoken dialogue,由此自然获得了零样本音色控制能力。这一思路简洁且有效 — 通过复用成熟的 TTS 组件,避免了从零设计 timbre-controllable SDM 的复杂性。Semantic group modeling 的效果令人印象深刻: G=3 将 ASR-WER 从 18.23% 降至 4.54%,同时训练成本减半,这表明 speech-text 频率对齐是 parallel generation 范式中一个被低估的关键因素。

**不足**:

"首个单阶段训练的竞争性 SDM" 这一 claim 需要审慎看待。单阶段训练优于多阶段预训练的实验仅在 0.5B 模型上验证 [Table 6],而 GLM-4-Voice (9B) 等成功的大模型都采用多阶段训练。作者虽在 Limitations 中承认 scale-up 是开放问题,但正文中的 framing 容易让读者过度推广这一结论。

UTMOS 和 ASR-WER 的优势很大程度上得益于 CosyVoice vocoder 本身的高质量,而非 SLAM-Omni 的对话建模创新。相比之下,ChatGPT Score 上仍有明显差距 (同规模 text LLM 的 57.70 vs 39.32),说明在语义内容质量上,spoken dialogue 的额外开销对小模型的知识保留影响显著。

评估框架虽然新颖 (三维度八数据集),但全部由作者自建/改编,且仅通过 Whisper 转写 + GPT 评分的间接方式评估语音输出,未采用 VoiceBench 等已有标准化 benchmark 进行端到端评估。

## 可复用的 idea

1. **Semantic group modeling 解决 audio-text 频率失配**: 当 parallel 生成 text 和 audio tokens 时,通过分组预测压缩 audio 序列长度至与 text 序列接近,是一个通用且有效的技巧。G=3 (50Hz → ~17Hz) 是实验验证的最优点,可直接迁移到其他 parallel audio-text generation 系统。注意 G 过大 (G=5) 反而损害质量,需要针对具体 token rate 调整 [Table 5]。

2. **Historical text prompting 压缩多轮历史**: 用 text-only 表示对话历史 + Whisper 解码提取用户转写 + text stream 提取模型回复,是一种低成本的多轮对话方案。代价是丢失副语言信息,但对信息交换型对话场景可接受。KV cache 跨轮复用的设计 (Fig. 4) 进一步提升推理效率。

3. **CosyVoice semantic token + CFM 方案复用到对话**: 将成熟 TTS 系统的 tokenizer + vocoder 直接嫁接到 SDM,避免重新设计语音表征和合成模块。这一"TTS 组件复用"模式可推广到其他场景: 只要 tokenizer 能解耦 content 和 speaker,vocoder 就能独立控制音色,LLM 只需学习生成 content tokens。

4. **单阶段训练在小模型场景的适用性**: 对于参数量有限 (<=1B) 的 LLM backbone,单阶段直接微调 speech-to-speech 可能优于先 ASR/TTS 预训练再微调,因为小模型更容易因预训练发生灾难性遗忘。但此结论不宜推广到大模型。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节含因果解释,三个关键设计选择均回答了 WHY |
> | 可信赖 | pass | 数字 claim 均有出处标注,指标名称正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率 >80% |
> | 可定位 | pass | KB 背景定位清晰,对比 Mini-Omni/CosyVoice/IntrinsicVoice |
> | 不污染 | pass | 未创建新概念页,frontmatter 挂接合理 |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> - [medium/traceability-gap] 局限性第 3-6 条为 agent 解读但部分未明确标注 — 已补标
> - [medium/weak-reusability] 可借鉴第 4 条 (单阶段训练适用性) 需加限定条件 — 已加 "不宜推广到大模型"
> - [low/template-compliance] 多轮和中文实验表格未含详细 per-dataset breakdown,仅用 overall — 因篇幅限制可接受
> 详见 `_review/SLAM-Omni-review.yml`
