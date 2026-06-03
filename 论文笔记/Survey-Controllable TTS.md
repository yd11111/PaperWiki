---
type: paper-note
tier: deep
title: "Survey: Towards Controllable Speech Synthesis in the Era of Large Language Models"
arxiv_id: "2412.06602"
source: "Sources/Survey-ControllableTTS-2024.pdf"
authors: [Tianxin Xie, Yan Rong, Pengfei Zhang, Wenwu Wang, Li Liu]
year: 2024
venue: "arXiv:2412.06602v3"
tags: [survey, controllable-TTS, LLM, style, emotion, prosody, instruction, disentanglement, zero-shot]
concepts: ["[[LLM-based TTS]]", "[[Prosody Modeling]]", "[[Speech Factorization]]", "[[Conditional Flow Matching]]", "[[Style Transfer in TTS]]", "[[Emotion Control in TTS]]", "[[Natural Language Description for TTS]]", "[[Global Style Tokens]]", "[[Non-autoregressive TTS]]"]
models: ["[[CosyVoice]]", "[[EnCodec]]", "[[SoundStream]]"]
tasks: ["[[Instructed Speech Generation]]", "[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-01
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[LLM-based TTS]], [[Prosody Modeling]], [[Speech Factorization]], [[Conditional Flow Matching]], [[Instructed Speech Generation]], [[Zero-shot Speech Synthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文是首篇全面综述可控 TTS 方法的 survey,覆盖从传统控制技术到 LLM 时代自然语言提示驱动的新方法。知识库中已有从各具体论文精读中沉淀的概念页(如 [[LLM-based TTS]] 追踪了 VALL-E 系列和 CosyVoice 等关键模型,[[Prosody Modeling]] 覆盖了从 GST 到 VALL-E 的韵律建模演进,[[Speech Factorization]] 详述了解耦方法谱系)。本 survey 的价值在于**提供了横向全景分类框架**(架构/控制策略/特征表示三轴分类),这是单篇论文精读无法提供的。
>
> **已有认知**: 知识库对 LLM-based TTS、CFM、Speech Factorization 已有较完整理解;但在"控制策略分类"(Style Tagging → Reference Prompt → NL Description → Instruction-Guided)方面的系统梳理尚不充分。本 survey 的 Fig.4 控制策略分类和 Table 2 优劣分析可补充此缺口。
>
> **创新判断**: 本文不是提出新方法,而是提供分类框架和研究路线图。其核心贡献在于: (1) 首次系统定义可控 TTS 的六大任务维度; (2) 提出架构/策略/表示三轴分类体系; (3) 总结数据集和评估方法; (4) 提出 Gemini-based MLLM 可控性评估。
>
> 检索命中: [[LLM-based TTS]]✓, [[Prosody Modeling]]✓, [[Speech Factorization]]✓, [[Conditional Flow Matching]]✓, [[Instructed Speech Generation]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Style Transfer in TTS]](pending-review), [[Emotion Control in TTS]](pending-review), [[Global Style Tokens]](pending-review), [[Non-autoregressive TTS]](pending-review), [[Natural Language Description for TTS]](pending-review), [[Instruction-Guided Speech Synthesis]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首篇全面综述可控 TTS 的 survey,系统分类模型架构(NAR/AR/Hybrid)、控制策略(Tag→Ref→NL→Instruction)、特征表示(连续/离散/解耦),并总结数据集、评估方法和未来方向
> - **路线**: 6 大可控任务(Prosody/Timbre/Emotion/Style/Language/Environment) × 4 大控制策略(Style Tagging/Ref Speech/NL Description/Instruction) × 2 大特征表示(Continuous/Discrete) + 解耦技术
> - **指标**: Table 6 Gemini-based 评估 — CosyVoice instruction-based: Instruction Following 4.81, Naturalness 4.92, Expressiveness 4.78; zero-shot Vevo: Naturalness 4.43, Expressiveness 4.32 [Table 6, Appendix A.5]
> - **可借鉴**: (1) 控制策略演进框架可作为新论文评估的定位坐标; (2) Gemini-based MLLM 评估 pipeline 可用于 instruction-following TTS 的自动化评测; (3) Table 1 的 100+ 方法全景对比表可作为快速 baseline 查表
> - **局限**: (1) 未讨论可控属性间交互效应; (2) 未评估计算效率; (3) 未涉及深伪造/安全伦理; (4) 覆盖截至 2025.03,缺少最新进展如 CosyVoice 3, Step-Audio 2.5 等

## 核心问题

本 survey 试图回答: **在 LLM 时代,可控 TTS 方法如何从传统的有限属性控制演进到自然语言驱动的精细化控制?** 具体拆解为三个子问题:
1. 模型架构如何演进以支撑更灵活的可控性? [§3.1]
2. 控制策略如何从预定义标签发展到自由文本指令? [§3.2]
3. 特征表示(连续 vs 离散、纠缠 vs 解耦)如何影响可控性? [§3.3]

## 方法: 它怎么 work (Survey 框架)

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 一、可控 TTS 的六大任务维度 [§2]

Survey 将可控 TTS 分解为六个可独立控制的语音属性维度:

| 任务 | 目标 | 控制对象 | 典型应用 |
|------|------|----------|----------|
| **Prosody** | 操控低层声学特征 | Pitch, Duration, Energy | 强调/节奏/语调 [§2] |
| **Timbre** | 操控音色/声质 | Gender, Age, Nasality | 个性化 TTS, Voice Conversion [§2] |
| **Emotion** | 合成带情感的语音 | Affective state | 对话系统/有声书 [§2] |
| **Style** | 控制高层属性 | Tone, Formality, Discourse mode | Newscast/Podcast [§2] |
| **Language** | 多语/方言/代码切换 | Language identity | 跨语言通信 [§2] |
| **Environment** | 模拟声学环境 | Background noise, Spatial cues | 影视/有声书 [§2] |

[agent 解读] 这六个维度从低层(Prosody)到高层(Style/Instruction)构成层级结构,控制粒度和复杂度递增。前三者(Prosody/Timbre/Emotion)是传统可控 TTS 已部分解决的问题,后三者(Style/Language/Environment)是 LLM 时代的新兴方向。

### 二、模型架构分类 [§3.1]

#### 非自回归方法 (NAR) [§3.1.1]

四大技术路线:

1. **Transformer-based**: FastSpeech/FastPitch 通过 duration/pitch/energy predictor 实现显式韵律控制,非自回归并行生成 [§3.1.1]
2. **VAE-based**: 结构化连续隐空间,支持韵律/情感/风格的采样和插值。代表: Parallel Tacotron (VAE residual encoder), CLONE (conditional VAE + NF + adversarial training) [§3.1.1]
3. **Diffusion-based**: 正/反向噪声注入过程。NaturalSpeech 2/3 (latent diffusion), DEX-TTS (DiT with overlapping patches), E3 TTS (直接建模波形) [§3.1.1]
4. **Flow-based**: 可逆流映射到高斯分布。VoiceBox/P-Flow (speech infilling), FlashSpeech (latent consistency model, 1-2步生成), E2 TTS/F5-TTS (filler-augmented flow matching) [§3.1.1]

#### 自回归方法 (AR) [§3.1.2]

两大技术路线:

1. **RNN-based**: Prosody-Tacotron (显式韵律控制), GST-Tacotron (无监督风格发现), MsEmoTTS (层级情感: global+utterance+local) [§3.1.2]
2. **LLM-based** (Fig. 2): VALL-E 开创 codec LM 范式 → 典型架构: Text + Instruction → Text Encoder → Text Tokens; Reference Speech → Speech Encoder → Audio Prompt; Decoder-only Transformer 生成 Audio Tokens → Speech Decoder → Waveform [§3.1.2]

[论文原文] "VALL-E pioneered LLM-based zero-shot TTS by framing it as a conditional language modeling task. It uses EnCodec to discretize waveforms into tokens and adopts a two-stage pipeline: an autoregressive model generates coarse audio tokens, followed by a non-autoregressive model for iterative refinement." [§3.1.2]

#### Hybrid 架构与演进趋势 [§3.1.3]

Survey 提出的架构演进路线 (Fig. 3) [§3.1.3]:

```
Traditional CNN/RNN → 有限控制力, 显式特征工程
    ↓
Flow-based (Matcha-TTS, F5-TTS) → 并行, 概率控制, 训练复杂
    ↓
LLM-based (VALL-E, InstructTTS) → 自然语言控制, zero-shot, 推理慢
    ↓
Hybrid (CosyVoice) → 直觉控制 + 高保真
    ↓
Future: Instruction-Aware Frameworks → 精细指令控制
```

[论文原文] "Hybrid architectures (e.g., CosyVoice) integrate LLM-guided semantic conditioning into flow-based generators, combining high-fidelity synthesis with intuitive, instruction-based control." [§3.1.3]

[agent 解读] 每一代架构的演进本质上是在 **控制粒度** 和 **生成质量** 之间寻找新的平衡点。CNN/RNN 有高保真但控制力差;LLM-based 有灵活控制但生成慢且有 artifact;Hybrid 是目前工程上最优的 Pareto 前沿。

### 三、控制策略分类 (Fig. 4) [§3.2]

这是本 survey 最核心的分类贡献。四大控制策略代表了可控 TTS 的演进路径:

#### 1. Style Tagging [§3.2.1]

**原理**: 用预定义标签(离散/连续)直接指定语音属性 [§3.2.1]

三种子范式:
- **离散标签**: StyleTagging-TTS 用情感词(angry/happy)学习 style-linguistic embedding; Emo-DPO 通过 DPO + LLM 实现情感控制; Spark-TTS 用专用 token 实现 pitch/speaking rate 修改 [§3.2.1]
- **连续信号**: DiffStyleTTS 层级化建模 pitch/energy/duration/style 的 scale factors; DrawSpeech 让用户直接绘制韵律轮廓,由 diffusion model 细化 [§3.2.1]
- **隐空间修改**: Cauliflow 通过 flow-based model 调整语速/停顿; DiTTo-TTS 通过 DiT 修改 latent length predictions [§3.2.1]

[论文原文] "These methods show great potential in controlling speech attributes by adjusting input signals or latent variables. However, these methods are limited in expressive diversity, as they can only model a small set of pre-defined attributes." [§3.2.1]

#### 2. Reference Speech Prompt [§3.2.2]

**原理**: 用几秒参考语音作为风格条件,模型从中提取 timbre/style/prosody 信息 [§3.2.2]

关键方法:
- MetaStyleSpeech: adaptive normalization for style conditioning → 零样本 [§3.2.2]
- MegaTTS 2: acoustic autoencoder 分离 prosody/timbre → 任意 timbre 的 style transfer [§3.2.2]
- ControlSpeech: bidirectional attention + parallel decoding → 零样本 timbre/style/content 控制 [§3.2.2]
- StyleTTS-ZS: distilled time-varying style diffusion → 捕捉多样 speaker identity + prosody [§3.2.2]

#### 3. Natural Language Descriptions [§3.2.3]

**原理**: 用自然语言文本描述期望的语音特征,作为条件输入 TTS 模型 [§3.2.3]

演进路线:
- PromptTTS (2023): 手动标注 5 种属性描述,开创 NL-driven TTS [§3.2.3]
- InstructTTS: 三阶段训练提取 NL 语义 [§3.2.3]
- NansyTTS: 跨语言描述控制,共享 timbre/style 表示 [§3.2.3]
- PromptTTS++: 额外 speaker description prompts 增强 prompt richness [§3.2.3]
- PromptTTS 2: variation network 建模 prompt 之外的残余变化 [§3.2.3]
- VoiceLDM/AST-LDM: 扩展到环境声学控制 [§3.2.3]

#### 4. Instruction-Guided Control [§3.2.4]

**原理**: 与 NL Description 不同,指令将 content 和 style 统一到单一自然语言 prompt 中,类似 chatbot 交互方式 [§3.2.4]

关键系统:
- **VoxInstruct** (2024): 重构 TTS 为通用 instruction-to-speech 任务,单一 prompt 同时传达内容+风格 [§3.2.4]
- **CosyVoice** (2024): supervised semantic tokens (ASR) + LLM token generation + flow-matching synthesis → 通过 NL instruction 控制 speaker/emotion/pitch/speed [§3.2.4]
- **Step-Audio** (2025): speech-text model + instruction-driven TTS module → 方言/情感/说唱/歌唱/说话风格 [§3.2.4]
- **AudioGPT** (2024): 多模态 LLM agent,集成多模块进行语音理解/合成/风格转换 [§3.2.4]

#### 5. Instruction-Guided Editing [§3.2.5]

- **VoiceCraft**: decoder-only transformer + causal masking + delayed stacking → 双向上下文感知编辑 (插入/删除/替换) [§3.2.5]
- **InstructSpeech**: multi-task LLM + <instruction, input, output> triplets → 自由格式语音编辑 + 多步推理 [§3.2.5]

#### 控制策略演进总结 (Table 2) [§3.2.6]

| 策略 | 优势 | 局限 |
|------|------|------|
| Style Tagging | 实现简单 | 表现力有限,无法细粒度/组合控制 [Table 2] |
| Reference Prompt | 高度个性化,灵活 | 依赖高质量参考音频,控制维度不直观 [Table 2] |
| NL Description | 可解释性好,用户友好 | 描述自由度和准确性有限,模型可能误解 [Table 2] |
| Instruction-Guided | 极高精度+自由度,理解复杂指令 | 强依赖 LLM,系统复杂度高 [Table 2] |

[论文原文] "The progression from tags to natural instructions shows a clear trajectory toward more expressive, personalized, and intuitive TTS, driven by LLM integration." [§3.2.6]

### 四、特征表示 [§3.3]

#### Speech Attribute Disentanglement [§3.3]

两种主流方法:
1. **对抗训练**: 辅助分类器 + GRL 惩罚 latent 中的 unwanted 属性(speaker/emotion/style invariance) [§3.3]
2. **信息瓶颈**: 多分支编码器各自限制容量,每个分支编码一个因子 (content/prosody) + 对抗/重建损失防止泄露。辅助技术: KL 正则化、量化 [§3.3]

#### Continuous vs Discrete [§3.3]

| 表示 | 优势 | 局限 | 典型方法 |
|------|------|------|----------|
| **Continuous** (Mel/Latent) | 细粒度细节保留,天然编码韵律/情感,平滑重建 | 计算密集,需大模型 | GAN/VAE/Flow/Diffusion [§3.3] |
| **Discrete** (Codec tokens) | 简洁高效,少样本泛化好,适配 LLM 训练 | 信息损失,缺乏连续特征的细微差异 | LLM-based methods [§3.3] |

## 实验: 数据集与评估 [§4]

### 数据集分类 (Table 3) [§4.1]

| 类型 | 特点 | 代表 | 规模 |
|------|------|------|------|
| **Tag-based** | 离散属性标签 (emotion/age/gender) | IEMOCAP, ESD, RAVDESS | 12h~29h [Table 3] |
| **Description-based** | 自然语言描述 + 语音 | PromptSpeech, TextrolSpeech, SpeechCraft, Parler-TTS | 330h~50Kh [Table 3] |
| **Dialogue** | 多轮对话,含 turn-taking/prosodic variation | Taskmaster-1, DailyTalk, MagicData-RAMC | 20h~180h [Table 3] |

### 评估方法 [§4.2]

**客观指标** [§4.2.1]:
| 指标 | 类型 | 目标 |
|------|------|------|
| MCD | 频谱距离 | <4 好, >6 差 [§4.2.1] |
| FDSD | 分布距离 | 越低越好 (类似 FID) [§4.2.1] |
| WER | 可懂度 | 越低越好 [§4.2.1] |
| Cosine Similarity | 说话人相似度 | 越高越好 (ECAPA-TDNN/x-vector) [§4.2.1] |
| PESQ | 感知质量 | [-0.5, 4.5] [§4.2.1] |

**主观指标** [§4.2.1]:
- MOS (1-5 scale), CMOS (-3 to +3 relative), AB/ABX Tests

**Model-based 评估** (本文贡献) [§4.2.2, A.5]:
- 使用 Google Gemini 2.5 Flash 评估 10 个 TTS 系统的三维可控性: Instruction Following / Naturalness / Expressiveness
- Pearson 相关系数优于 NISQA 和 UTMOS,但绝对值仍较modest [Table 7]

### Gemini 评估结果 [Table 6]

**Zero-shot 设定**:
| 模型 | Naturalness | Expressiveness |
|------|-------------|----------------|
| Vevo | 4.43 +/- 0.55 | 4.32 +/- 0.75 |
| F5-TTS | 4.27 +/- 0.87 | 4.21 +/- 0.78 |
| CosyVoice | 4.20 +/- 0.82 | 4.17 +/- 0.83 |
| CosyVoice 2 | 4.25 +/- 0.58 | 4.20 +/- 0.63 |
| SparkTTS | 3.68 +/- 0.80 | 3.83 +/- 0.79 |

**Instruction-based 设定**:
| 模型 | Instruction Following | Naturalness | Expressiveness |
|------|----------------------|-------------|----------------|
| CosyVoice | 4.81 +/- 0.28 | 4.92 +/- 0.24 | 4.78 +/- 0.29 |
| MiniMax TTS | 4.67 +/- 0.36 | 4.87 +/- 0.27 | 4.63 +/- 0.44 |
| EmoVoice | 4.67 +/- 0.44 | 4.80 +/- 0.36 | 4.67 +/- 0.44 |
| CosyVoice 2 | 4.61 +/- 0.49 | 4.85 +/- 0.31 | 4.63 +/- 0.52 |
| VoxInstruct | 4.45 +/- 0.50 | 4.83 +/- 0.32 | 4.50 +/- 0.52 |

[论文原文] "Even the lowest-scoring instruction-based method (VoxInstruct) outperforms the best zero-shot model in every aspect." [Appendix A.5.2]

## 局限性

### Survey 自身承认的局限 [§7]
1. **未探讨可控属性间交互**: 各属性被独立建模,但 pitch 变化会影响 emotion 和 naturalness [§7]
2. **未讨论效率**: description/instruction-guided 方法依赖大 LLM,计算成本高 [§7]
3. **未涉及社会影响**: deepfake/adversarial attack 风险未讨论 [§7]
4. **未覆盖相关领域**: speech enhancement, separation, pretraining, S2S translation 可能提供互补技术 [§7]

### Agent 补充的局限 [agent 解读]
5. **评估覆盖不足**: Gemini-based 评估仅覆盖 10 个模型(8 开源 + 2 商用),样本量较小(20 samples × 2 tasks)
6. **时间截止**: 方法覆盖截至 2025.03,后续重要工作(CosyVoice 3, Step-Audio 2.5, Qwen3-TTS 等)未纳入
7. **解耦评估缺失**: 虽然讨论了 disentanglement 方法,但没有对解耦质量进行系统对比

## 点评

### 贡献
1. **首篇全面可控 TTS 综述**: 填补了可控 TTS 缺乏系统综述的空白,100+ 方法的全景 Table 1 是有价值的参考工具
2. **控制策略四级分类 (Fig. 4)**: 从 Style Tagging → Reference → NL Description → Instruction-Guided 的演进框架清晰有力,是理解领域发展的有效坐标系
3. **Model-based 评估探索**: Gemini-based 可控性评估是有意义的方向,尤其对 Instruction Following 维度的评估弥补了传统指标(MCD/WER/MOS)的不足
4. **架构演进路线图 (Fig. 3)**: 虽然简化,但"效率 vs 控制力 vs 保真度"的 trade-off 分析是有洞察力的

### 不足
1. **深度不够**: 作为 survey 覆盖面广但每个方法的分析较浅,缺少对关键设计选择的深入比较
2. **缺少统一 benchmark**: 不同方法在不同数据集上评估,无法直接比较(除了 Gemini 评估覆盖的 10 个模型)
3. **对 Hybrid 架构分析不足**: CosyVoice 作为 Fig. 3 中唯一的 Hybrid 代表,缺少与其他 Hybrid 方案(如 Seed-TTS, NaturalSpeech 3)的结构对比
4. **未来方向略显泛泛**: 5 个 future direction 大多是已知的开放问题,缺少具体的技术路线建议

## 可复用的 idea

1. **控制策略作为评估坐标**: 新 TTS 论文可以按 Fig. 4 分类定位自己的控制策略层级,明确其相对于 state-of-the-art 的创新点在哪个轴上
2. **MLLM-based 可控性评估**: 用 Gemini/GPT 从 (audio, transcript, instruction) 三元组评估 Instruction Following/Naturalness/Expressiveness,虽然 Pearson correlation 尚moderate,但比 NISQA/UTMOS 在可控性维度更对齐人类
3. **六维可控性 checklist**: 设计新 TTS 系统时可用 Prosody/Timbre/Emotion/Style/Language/Environment 六维度作为功能覆盖 checklist
4. **Table 1 作为 baseline 查表**: 100+ 方法的架构/特征/可控性对比表可作为 related work 的快速索引

> [!review] 审阅标记
> 待审阅 — 已生成草稿,待自动审阅。
