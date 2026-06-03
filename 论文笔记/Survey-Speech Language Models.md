---
type: paper-note
title: "Recent Advances in Speech Language Models: A Survey"
arxiv_id: "2410.03751"
source: "Sources/Survey-SpeechLanguageModels-2024.pdf"
authors: [Wenqian Cui, Dianzhi Yu, Xiaoqi Jiao, Ziqiao Meng, Guangyan Zhang, Qichao Wang, Yiwen Guo, Irwin King]
year: 2024
venue: "arXiv:2410.03751v4"
tags: [survey, speech-LM, multimodal, full-duplex, training-strategy, evaluation, tokenizer, vocoder, cold-start]
concepts: ["[[Speech Language Model]]", "[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Codec Language Model]]", "[[Full-duplex Spoken Dialogue]]", "[[Speech-Text Alignment]]", "[[Audio Understanding]]", "[[Prosody Modeling]]"]
models: ["[[HuBERT]]", "[[w2v-BERT]]", "[[wav2vec 2.0]]", "[[EnCodec]]", "[[SoundStream]]", "[[Whisper]]", "[[VITS]]", "[[BigVGAN]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
tier: deep
created: 2026-06-02
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Speech Language Model]], [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[LLM-based TTS]], [[Residual Vector Quantization]], [[Neural Vocoder]])
> 本 survey 是 Speech Language Model 概念页和 Semantic vs Acoustic Tokens 概念页的 origin_paper。知识库已基于本文建立了 SpeechLM 三组件架构 (tokenizer + LM + vocoder)、semantic/acoustic/mixed token 三分法、以及 SpeechLM 分类体系 (features modeled x training stages x generation paradigm) 的框架。Speech Tokenizer 页已记录本文对 50+ 系统的 tokenizer 选择统计 (Table II)。LLM-based TTS 页厘清了 SpeechLM > CodecLM > LLM-based TTS 的包含关系。本次精读需重点关注:
> 1. 训练策略对比 (冷启动 vs 继续预训练 vs 指令微调 vs 后对齐) 的具体证据
> 2. 评估体系 (表征/语言/副语言/质量/实时交互) 的系统化分类
> 3. 四个挑战方向 (组件选择/端到端训练/实时生成/安全) 的论据
> 4. Speech-text 对齐方式 (单序列/多序列/text-present/text-independent) 的 trade-off
>
> 检索命中: [[Speech Language Model]]✓, [[Speech Tokenizer]]✓, [[Semantic vs Acoustic Tokens]]✓, [[LLM-based TTS]]✓, [[Residual Vector Quantization]]✓, [[Neural Vocoder]]✓ | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首篇系统综述 Speech Language Models (SpeechLMs) 的论文,从组件架构 (tokenizer + LM + vocoder)、训练策略 (features x stages x paradigm)、下游应用、评估方法四个维度构建 SpeechLM 分类体系,覆盖 50+ 系统
> - **路线**: Speech/Text → [Speech Tokenizer (semantic/acoustic/mixed)] → [Language Model (Transformer decoder)] → [Vocoder (GAN/Flow/Diffusion)] → Speech/Text
> - **指标**: 非实验论文,无本文实验数据;汇总的评估维度包括表征 (ABX)、语言 (sWUGGY/sBLIMP)、副语言 (pGSLM correctness/consistency/expressiveness)、质量 (MOS/MMOS/PMOS)、实时交互 (IPU/pause/overlap 统计)
> - **可借鉴**: (1) 三轴分类框架 (features modeled x training stages x generation paradigm) 可迁移到其他多模态模型的 survey 中; (2) Table II 的 50+ 系统组件选择矩阵是 tokenizer/LM/vocoder 选型的直接参考; (3) 评估体系的 auto+human 二分法和 5 子维度分类为 SpeechLM benchmark 设计提供框架
> - **局限**: (1) 对各系统间的定量横向对比较少 (缺统一 benchmark 数据); (2) 安全讨论仅限毒性和隐私两点,未深入; (3) v4 更新至 2025.08 [arXiv v4 header] 但仍缺少 2025 年部分新系统 (如 CosyVoice 3, LatentLM, CLEAR 等连续 token 路线) [agent 解读]

## 核心问题

本文要回答的核心问题是: **如何系统地理解和分类 Speech Language Models?** 具体分解为:

1. SpeechLM 相比 ASR+LLM+TTS 级联管线有哪些结构性优势? [§I]
2. SpeechLM 由哪些组件构成,各组件有哪些设计选择? [§III]
3. 训练 SpeechLM 有哪些策略,各策略的 trade-off 是什么? [§IV]
4. SpeechLM 能做哪些下游任务,与传统模型有何不同? [§V]
5. 如何系统评估 SpeechLM? [§VI]
6. SpeechLM 发展面临哪些挑战? [§VII]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SpeechLM 的形式化定义 [§II]:
```
M^out = SpeechLM(M^in; θ)
```
其中 M^in, M^out 是多模态序列 (speech 和/或 text 的混合)。这意味着 SpeechLM 是一个 **统一的多模态序列模型**,支持 speech→speech, speech→text, text→speech, speech+text→speech+text 等任何模态组合 [论文原文, §II]。

三组件架构 [§III, Fig. 1b]:
```
Speech → [Speech Tokenizer f_E] → tokens/representations s
                                       ↓
Text → [Text Tokenizer] → text tokens t → [Language Model LM] → output tokens → [Vocoder Vo] → Speech
```

**为什么是三组件?** 因为语言模型的核心是自回归预测离散 token,但语音是连续波形,需要 tokenizer 做离散化 (或连续表征提取) 和 vocoder 做波形重建 [论文原文, §III]。

### 关键设计选择

#### 设计选择 1: Speech Tokenizer 类型 [§III-A, Fig. 3]

| 类型 | 训练目标 | 代表 | 优势 | 劣势 |
|------|---------|------|------|------|
| Semantic | 自监督 (masked prediction) | HuBERT, w2v-BERT, wav2vec 2.0 | 与文本对齐好,语义连贯 | 缺高频声学细节 [§III-A1] |
| Acoustic | 波形重建 (RVQ) | EnCodec, SoundStream | 高保真音频重建 | 语义对齐差 [§III-A2] |
| Mixed | 语义蒸馏+声学重建 | SpeechTokenizer, Mimi | 兼顾语义和声学 | 仍在早期阶段 [§III-A3] |

**为什么 semantic tokenizer 最流行?** "semantic information plays the most crucial role in spoken communication" [论文原文, §IV-A1]。GSLM 对比实验证明 HuBERT 在语义理解和语音生成上优于 CPC 和 wav2vec 2.0 [§IV-A1]。但 semantic tokens 缺乏高频声学细节,恢复这些细节需要后处理 (如 diffusion model),显著增加延迟 [论文原文, §IV-A1]。

**Mixed tokenizer 为什么值得关注?** SpeechTokenizer 通过 RVQ 第一层蒸馏 HuBERT 语义,后续层量化声学残差,在单一框架内兼顾两者 [§III-A3]。Mimi 用单 VQ 提取语义 + 额外 RVQ 提取声学,被 Moshi 全双工系统采用 [§III-A3]。[agent 解读] 这种混合路线是 semantic-only 和 acoustic-only 之间的折中,避免了串联方案的序列长度爆炸问题。

#### 设计选择 2: Language Model 架构 [§III-B]

SpeechLM 的 LM 核心适配 [§III-B]:
- **Discrete tokens**: 将 text embedding matrix E_t 替换为 speech embedding matrix E_s ∈ R^{|V_s| x h},输出矩阵对应替换 [论文原文, Eq. 6]
- **Continuous tokens**: 直接将 speech encoder 的连续嵌入送入 LM,不需要修改 LM 架构 [论文原文, §III-B]
- **Multi-modal**: 扩展词表同时包含 text 和 speech tokens,E_m ∈ R^{(|V_t|+|V_s|) x h} [论文原文, Eq. 7]

Table II 统计的 LM 选择: Transformer* (自定义) 最早,后续主流转向 LLaMA, Qwen2, OPT, GLM, Mistral 等预训练 TextLM。

#### 设计选择 3: Vocoder 类型 [§III-C]

两种 pipeline [§III-C]:
1. **Direct synthesis**: 直接从 speech tokens 生成波形 (如 HiFi-GAN 以 tokens 为输入) — 适合 acoustic tokens [论文原文]
2. **Input-enhanced synthesis**: 先将 tokens 转为 mel-spectrogram 等中间表征,再用 vocoder 合成 — 适合 semantic tokens (需补充声学细节) [论文原文]

**为什么 HiFi-GAN 是最常用 vocoder?** Table II 显示 50+ 系统中 HiFi-GAN 使用频率最高。[agent 解读] 这是因为 HiFi-GAN 在速度 (实时 13.4x) 和质量之间取得了最佳平衡,且轻量 (14M 参数),易于集成。

### 训练策略

#### 三轴分类框架 [§IV, Fig. 4]

**轴 1: Features Modeled [§IV-A]**
- **Discrete features**: Semantic tokens (GSLM, TWIST, SpeechGPT) / Paralinguistic tokens (pGSLM, SPIRIT-LM) / Acoustic tokens (VioLA, Parrot) / Mixed tokens (Moshi, SpeechGPT-Gen)
- **Continuous features**: 直接使用 mel-spectrogram 或 neural network latent (Spectron, Mini-Omni, LauraGPT, SLAM-Omni)

**轴 2: Training Stages [§IV-B]**

| 阶段 | 核心思路 | 代表 | 关键发现 |
|------|---------|------|---------|
| 冷启动 (Cold Init) | 从零训练 Transformer | GSLM, SUTLM, pGSLM, dGSLM | HuBERT tokenizer 最优 [§IV-B1] |
| 继续预训练 (Continued) | 从 TextLM 继续训练 | TWIST, AudioPaLM, SPIRIT-LM, Moshi, Mini-Omni | TextLM 初始化显著加速收敛 [§IV-B1] |
| 指令微调 (Instruction-tuning) | 微调以遵循指令 | SpeechGPT, COSMIC, Llama-Omni | 关键在于指令数据构造 [§IV-B2] |
| 后对齐 (Post-alignment) | RLHF/DPO 对齐 | Align-SLM, SpeechAlign | 仍 under-explored [§IV-B3] |

**TextLM 初始化为什么有效?** TWIST (Hassid et al., 2024) 发现从 OPT/LLaMA 预训练权重继续训练 speech tokens 可以 "enhance the model's convergence rate and significantly improve its speech understanding capabilities" [论文原文, §IV-B1]。但并非所有预训练权重都有效 — "training from image-pretrained checkpoints yields poorer results compared to cold initialization" [论文原文, §IV-B1]。

**Speech-text 对齐的四种方式 [§IV-B1]:**
1. **Single-sequence interleaving**: text 和 speech tokens 交替排列 (SPIRIT-LM) — 增强模态间相似性 [论文原文]
2. **Multi-sequence parallel**: 同时生成 text 和 speech 序列 (Mini-Omni: 1 text + 7 acoustic streams; Moshi: 1 text + 1 semantic + 7 acoustic) — text-present 推理可减少幻觉 [论文原文]
3. **Text-present inference**: 推理时同时解码文本 — 增强推理能力但增加延迟 [论文原文]
4. **Text-independent**: 不依赖文本 — 效率高但可能不稳定 [论文原文]

[agent 解读] 训练一个 SpeechLM 比训练 TextLM 更难,因为 "text serves as a concentrated form of knowledge, while speech requires models to independently learn the rules of spoken language" [§IV-B1]。这解释了为什么继续预训练路线 (利用 TextLM 已有的语言知识) 优于冷启动。

**轴 3: Speech Generation Paradigm [§IV-C]**

| 范式 | 特点 | 代表 |
|------|------|------|
| Traditional | 完整输入→完整响应 | TWIST, SPIRIT-LM, AudioPaLM, SpeechGPT |
| Real-time interaction | 流式处理,低延迟 | dGSLM, NTPP, LSLM, VITA, Moshi, Mini-Omni 2, OmniFlatten |
| IPR (Interactive Period Recognition) | 识别是否应响应 | VITA, MiniCPM-o 2.6, FlexDuo |

**全双工对话的演进路线 [§IV-C]:**
1. **Streaming tokenizer + vocoder** → 消除等待完整输入的延迟 [论文原文]
2. **Full-duplex modeling** → 同时监听和生成,处理打断 [论文原文]
3. **IPR** → 区分应响应和不应响应的音频 (如用户跟旁人说话) [论文原文]

dGSLM 用 separate transformer per speaker + cross-attention 建模双人对话 [§IV-C]。Moshi 用 RQ-Transformer 同时处理用户和模型两个音频流 [§IV-C]。VITA 通过训练模型输出 EOS token 当检测到非查询音频来实现 IPR [§IV-C]。

## 下游应用分类 [§V]

SpeechLM 的三类下游应用 [§V, Table V]:

**语义相关 (Semantic-related) [§V-A]:**
- **Spoken Dialogue** (最自然的应用): SpeechLM 直接以语音对话,无需 ASR 中转,因此能在回复中保留输入语音的情感和语气上下文 [论文原文, §V-A]; 还能进行跨模态对话 (语音输入+文本输出或反之) [§V-A]
- Speech Translation, ASR, Keyword Spotting, TTS, Intent Classification, Slot Filling, QbE-STD

**说话人相关 (Speaker-related) [§V-B]:**
- Speaker Identification / Verification / Diarization
- **Voice-Conditioned Speech Generation**: SpeechLM 可通过 in-context learning 从参考语音隐式学习说话人身份,无需显式 speaker embedding [论文原文, §V-B]; 还能同时参与多人对话,区分不同说话人并分别回应 [§V-B]

**副语言相关 (Paralinguistic) [§V-C]:**
- **Emotion Recognition**: SpeechLM 不仅能直接识别语音情感,还能通过语音回复隐式反映对情感的理解 (如 "对不起听你这么难过" — 需要先理解情感再生成得体回复) [论文原文, §V-C]
- Speech Separation
- Paralinguistics-Enhanced Generation (如指定情感/语速的生成)

[agent 解读] 与 TextLM 的关键区别在于 SpeechLM 能建模副语言信息 (pitch, timbre, emotion),因此在说话人相关和副语言相关任务上具有 TextLM 不具备的能力。这也是 SpeechLM 的独特价值 — 不只是 "能说话的 LLM",而是能理解和生成语音全部维度的模型。

## 评估体系 [§VI]

### 自动评估 [§VI-A]

| 维度 | 指标 | 评估什么 |
|------|------|---------|
| 表征 (Representation) | ABX score, resynthesis WER/CER | 语音编码质量 [§VI-A] |
| 语言 (Linguistic) | sWUGGY (词级), sBLIMP (句级), StoryCloze (篇章级) | 语义理解能力 [§VI-A] |
| 副语言 (Paralinguistic) | min-MAE, Pearson correlation, std deviation | 韵律准确性/一致性/表现力 [§VI-A] |
| 质量与多样性 | AUC on perplexity/VERT (diversity), ChatGPT score | 生成质量 [§VI-A] |
| 实时交互 | IPU/pause/overlap statistics, reflective pause, interruption | 对话自然度 [§VI-A] |

### 人工评估 [§VI-B]

| 指标 | 评估什么 |
|------|---------|
| MOS | 自然度 |
| MMOS / PMOS | 韵律 / 说话人相似度 |
| SMOS | 风格相似度 |

### 主要 Benchmark [§VI, Table VI]

Survey 汇总了 15 个评估 benchmark: ABX, sWUGGY, sBLIMP, StoryCloze, STSP, MMAU (27 tasks), AudioBench (8 tasks), AIR-Bench (20 tasks), SD-Eval (4 tasks), SUPERB (10 tasks), VoxDialogue (12 tasks), Dynamic-SUPERB (180 tasks), SALMON (8 tasks), VoiceBench (8 tasks), VoxEval (56 tasks)。

[agent 解读] 评估体系的核心挑战在于: SpeechLM 横跨理解和生成两大方向,且同时处理语义和副语言信息,没有单一 benchmark 能全面评估。这也解释了为什么新 benchmark 不断涌现。

## 挑战与未来方向 [§VII]

### 挑战 A: 组件选择缺乏系统对比 [§VII-A]

现有组件比较 "primarily focusing on speech tokenizers — the comparisons tend to be limited in scope and depth" [论文原文]。缺乏在统一实验设置下对 tokenizer x LM x vocoder 的系统消融。

### 挑战 B: 端到端训练 [§VII-B]

大部分系统三个组件分别训练,限制了整体性能。探索梯度从 vocoder 回传到 tokenizer 的端到端训练可能产出 "more coherent, contextually relevant, and high-fidelity speech outputs" [论文原文]。

### 挑战 C: 实时语音生成 [§VII-C]

"a typical vocoder must wait for the entire sequence of output tokens to be generated by the language model before functioning" [论文原文]。需要开发流式 pipeline 或让 SpeechLM 自主生成波形 chunk。

### 挑战 D: 安全风险 [§VII-D]

SpeechLM 的安全问题包含 TextLM 没有的独特维度 [§VII-D]:
- **Toxicity**: 不仅是语义毒性,还包括声学维度的不当内容 (如 erotic speech) [论文原文]
- **Privacy**: 说话人身份推断风险,以及基于声学特征做种族/宗教等偏见推断的风险 [论文原文]

### 挑战 E: 低资源语言 [§VII-E]

SpeechLM 可直接建模语音,对 "low-resource" 语言友好 (语音数据通常比文本更丰富) [论文原文]。

## 实验

本文是 survey,不包含自有实验。但汇总了以下关键对比数据:

| 维度 | 最优系统/发现 | 数据来源 | 出处 |
| --- | --- | --- | --- |
| Speech tokenizer 对比 | HuBERT 在语义任务最优,但缺声学细节 | GSLM 消融 | [§IV-A1, ref 50] |
| TextLM 初始化 vs 冷启动 | TextLM 初始化显著优于冷启动和图像预训练初始化 | TWIST 实验 | [§IV-B1, ref 51] |
| Pre-trained LM 规模 | PaLM-2 基础的 AudioPaLM 受益于更大预训练 checkpoint 和更大训练集 | AudioPaLM 实验 | [§IV-B1, ref 52] |
| Speech-text 交替训练 | SPIRIT-LM 的 interleaving 提升语音理解和生成,可视化显示模态间相似性增强 | SPIRIT-LM 实验 | [§IV-B1, ref 5] |
| 全双工对话 | Moshi 实现 160ms 延迟的全双工对话 | Moshi 系统报告 | [§IV-C, ref 9] |

## 局限性

1. **缺乏统一横向对比**: Survey 汇总了 50+ 系统但缺乏在统一设置下的定量对比,各系统数据来自不同论文、不同评估协议,可比性有限 [agent 解读]
2. **连续 token 路线覆盖不足**: 对 Spectron, Mini-Omni 等连续特征方案的讨论较简略,未深入分析连续 vs 离散的系统性 trade-off [agent 解读]
3. **安全讨论较浅**: §VII-D 仅涉及 toxicity 和 privacy,未讨论 deepfake、watermarking、responsible deployment 等更广泛的安全议题 [agent 解读]
4. **评估维度缺乏整合**: 列出了 15 个 benchmark 但未提供跨 benchmark 的整合分析,缺少 "哪些 benchmark 的哪些子任务最能区分系统好坏" 的建议 [agent 解读]
5. **时效性局限**: v4 更新至 2025.08,但 2025 年的重要新方向 (连续 VAE tokenizer 如 LatentLM/CLEAR, 监督式 semantic tokenizer 如 CosyVoice 3) 未覆盖 [agent 解读]

## 点评

**作为首篇 SpeechLM survey 的定位价值**: 本文提出的三轴分类框架 (features modeled x training stages x generation paradigm) 为理解快速演进的 SpeechLM 领域提供了清晰的坐标系。特别是 Fig. 4 的完整分类图和 Table II 的 50+ 系统组件矩阵,具有很高的实用参考价值。

**核心论点的说服力**: Survey 对 ASR+LLM+TTS 管线 vs SpeechLM 的三个结构性优势 (信息保留、低延迟、无累积错误) 的论证清晰且有理论支撑。但对 "SpeechLM 真的解决了这三个问题吗?" 的反面讨论较少 — 例如, semantic tokenizer 仍然丢失副语言信息 (只是丢失的阶段从 ASR 移到了 tokenizer), 端到端延迟优势在实际部署中可能因 LM 推理长序列而被抵消。

**与知识库已有认知的关系**: 本文与 Yang et al. (2025) 的 Speech-LLM Integration Taxonomy 是互补关系 — 本文是 model-centric 视角 (从模型架构出发),Yang et al. 是 interface-centric 视角 (从集成方式出发)。两者结合能更全面地理解 speech-LLM 生态。

**覆盖度评价**: 对 tokenizer 和训练策略的覆盖最为全面,评估方法次之,下游应用的讨论偏浅 (多为任务定义,少有跨系统对比),安全和伦理讨论最薄弱。

## 可复用的 idea

1. **三轴分类框架**: features modeled x training stages x generation paradigm 的正交分类可迁移到其他多模态 (vision-language, audio-language) survey 的分类设计中
2. **组件矩阵 (Table II)**: 构建新 SpeechLM 时,可直接参考 50+ 系统的 tokenizer/LM/vocoder 选择统计,了解主流组合
3. **Speech-text 对齐四方式**: single-sequence interleaving vs multi-sequence parallel vs text-present vs text-independent 的分类,对设计新的多模态训练方案有指导价值
4. **评估五维度**: 表征/语言/副语言/质量/实时交互的系统化分类,可用于设计新 SpeechLM 的评估方案
5. **IPR 概念**: Interactive Period Recognition 作为全双工对话的关键能力 (区分 "应该响应" vs "不应该响应" 的音频),是对话系统设计的重要参考

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes | 可理解性 8 | 可溯源性 8 | 严谨性 9 | 可导航性 9 | 知识库安全性 9
> **Claim 标注覆盖率**: 86% (24/28)
> **问题**: 2 medium (速查局限行缺 [agent 解读] 标注 — 已修; 下游应用节偏列表式 — 已补充机制解释) + 2 low (可复用 idea 第4点偏泛; 实验表格表头 — 已修)
> **详见**: `_review/Survey-Speech Language Models-review.yml`
