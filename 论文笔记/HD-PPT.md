---
type: paper
tier: deep
title: "HD-PPT: Hierarchical Decoding of Content- and Prompt-Preference Tokens for Instruction-based TTS"
arxiv_id: "2509.19001"
source: "Sources/HD-PPT.pdf"
authors: [Sihang Nie, Xiaofen Xing, Jingyuan Xing, Baiji Liu, Xiangmin Xu]
year: 2026
venue: "arXiv"
tags: [TTS, LLM-based, instruction-following, controllability, speech-tokenizer, hierarchical-decoding, FSQ, CLAP, emotion, style]
concepts: ["[[Instruction-GuidedSpeechSynthesis]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[FiniteScalarQuantization]]", "[[Audio-LanguagePretraining]]", "[[SpeechFactorization]]", "[[SemanticvsAcousticTokens]]", "[[ProsodyModeling]]", "[[EmotionControlinTTS]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]", "[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/Whisper|Whisper]]"]
tasks: ["[[InstructedSpeechGeneration]]"]
datasets: ["[[CV3-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[LLM-basedTTS]], [[SpeechTokenizer]], [[模型库/CosyVoice2|CosyVoice 2]], [[InstructedSpeechGeneration]], [[SpeechFactorization]], [[SemanticvsAcousticTokens]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: HD-PPT 处于 LLM-based Instruct-TTS 的技术前沿。其主 baseline CosyVoice 2 (confirmed) 采用 FSQ-SenseVoice tokenizer + LLM + chunk-aware flow matching 架构,实现了低延迟双向流式合成。HD-PPT 的核心创新——将 speech tokens 分解为 content-preference 和 prompt-preference 两种中间表征——可视为 [[SpeechFactorization]] (confirmed) 在 token 级别的新实例化方式。传统 Speech Factorization 在特征层面解耦(如 NaturalSpeech 3 的 factorized diffusion codec、Seed-TTS 的 self-distillation),HD-PPT 则在 discrete token 空间做分解,通过 ASR + CLAP 双任务监督实现 content/style 解耦。
>
> **已有认知对比**: [[SemanticvsAcousticTokens]] (confirmed) 描述了传统的 semantic → acoustic 二层级 token 体系; HD-PPT 引入了正交的 content-preference vs prompt-preference 分解维度。CosyVoice 系列使用 supervised semantic token + CFM 的 coarse-to-fine 路线; HD-PPT 在 CosyVoice 2 tokenizer 输出的 speech tokens 上方再叠加一层 preference token 抽取,建立三层级生成: content → style → acoustic。[[Instruction-GuidedSpeechSynthesis]] [待确认] 从 NL Description → Instruction-Guided 的演进线上,HD-PPT 提出了结构化中间表征来弥合指令文本与语音 token 之间的模态鸿沟。[[Audio-LanguagePretraining]] [待确认] 中的 CLAP 对比学习被 HD-PPT 用于 prompt-preference token 的监督信号。
>
> **创新判断**: 相比 CosyVoice 2 (直接从 LLM hidden state 预测 speech token) 和 EmoVoice-PP (LLM + flow matching + HiFi-GAN),HD-PPT 的 hierarchical intermediate representation 是一种新颖的结构化解耦思路。关键区别在于: (1) 不是简单地增加 encoder 分支做解耦,而是在 decoder 端引入层级化的 intermediate prediction targets; (2) CLAP 监督让 prompt-preference token 直接与文本指令语义空间对齐,这比隐式学习风格映射更加 grounded。
>
> 检索命中: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[CosyVoice2]]✓, [[InstructedSpeechGeneration]]✓, [[SpeechFactorization]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[FiniteScalarQuantization]](pending-review), [[Instruction-GuidedSpeechSynthesis]](pending-review), [[Audio-LanguagePretraining]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过 ASR+CLAP 双监督的 speech codec 将 speech tokens 分解为 content-preference 和 prompt-preference 两种中间表征,再用 LLM + 轻量 decoder 层级化生成(先语义→再风格→最终声学),显著提升 Instruct-TTS 的指令遵循精度
> - **路线**: Text instruction → Qwen2.5-0.5B(AR 生成 hidden states) → 2-layer hierarchical decoder(sequentially 预测 content-pref token → prompt-pref token → speech token) → CosyVoice 2 vocoder(flow-matching + HiFi-GAN) → waveform
> - **指标**: MOS-N 4.108 / MOS-S 4.167(TextrolSpeech+EmoVoice-DB 联合测试); DNSMOS 3.84; EMO-SIM 0.753 (vs CosyVoice 2 的 0.714); WER 5.18% [Table 1]
> - **可借鉴**: (1) 用 ASR+CLAP 双监督训练 preference token codec,在 discrete token 空间实现 content/style 分离——可迁移到其他需要 fine-grained 控制的 codec 设计; (2) 层级化 decoder 的"先语义后风格"生成顺序,以及 stochastic masking + logit-embedding concat 正则化方案
> - **局限**: 仅在 TextrolSpeech + EmoVoice-DB 两个数据集上验证; RTF 0.952 比 single-step 的 0.711 慢 34%; 多组件设计增加了低资源语言适配难度; 未开源

## 核心问题

HD-PPT 要解决的核心问题是: **现有 LLM-based Instruct-TTS 的指令遵循精度不足** [§1]。

作者将根本原因归结为 **hierarchical mismatch**: 文本指令是单层级的自然语言描述,但语音 token 实际编码了三层信息——linguistic (语义内容)、paralinguistic (韵律/情感)、extralinguistic (说话人/场景) [§1, ref 13]。现有方法(如 CosyVoice 2)直接将指令映射到 monolithic speech token 序列,忽略了这种内在层级结构,导致 fine-grained 控制困难 [Fig 1(a)]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

HD-PPT 由三个组件构成 [§2, Fig 2]:

1. **Speech Token Codec**: 从 CosyVoice 2 tokenizer 产生的 speech tokens 中抽取 content-preference tokens 和 prompt-preference tokens
2. **Hierarchical LLM Decoder**: Qwen2.5-0.5B backbone + 轻量 2-layer decoder,按 content → style → acoustic 顺序逐步生成
3. **Vocoder**: CosyVoice 2 官方预训练 vocoder (flow-matching + HiFi-GAN)

### 关键设计选择

#### 设计选择 1: Dual-Preference Token Codec [§2.1, Fig 2(a)]

**做了什么**: 在 CosyVoice 2 tokenizer 的 speech token 上方构建一个 codec,将 speech tokens 分解为两种 preference tokens。

**架构**:
- **Preference Token Extractor**: 5 层 conformer,将 speech tokens 编码为连续表征 Z
- **两个独立 FSQ 模块**: 分别将 Z 量化为 content-preference tokens (codebook 1296, 25Hz) 和 prompt-preference tokens (codebook 64, 25Hz)
- **Speech Token Combiner**: 4 层 causal transformer,融合两种 preference tokens 重建原始 speech tokens

**为什么这样设计** [论文原文]: 作者认为语音 token 中纠缠了语义(content)和风格(prompt)信息,需要用不同的监督信号将它们分离到不同的表征空间 [§2.1]。选择 FSQ 是因为它避免了 codebook collapse [§2.1, ref 15]。Causal combiner 的设计是为了保证时间对齐 [§2.1]。

**监督方式**:
- **Content-preference tokens**: ASR 任务监督——用 Whisper-Small decoder 从 content-preference tokens 预测文本,迫使它们编码语义信息 [§2.1]
- **Prompt-preference tokens**: CLAP 对比学习监督——通过 cross-attention 将 tokens 映射为固定长度 embedding,最大化与对应 prompt 的 RoBERTa embedding 的 cosine similarity,最小化与不匹配 prompt 的 similarity [§2.1]

**为什么 content codebook (1296) >> prompt codebook (64)** [agent 解读]: 语义信息(内容词汇)的多样性远高于风格信息(情感/语速类别有限),因此 content-preference 需要更大的表征空间。64 的 codebook 大小暗示 prompt-preference 本质上编码的是有限类别的风格属性而非连续变化。

**总损失** [§2.1, Eq. 1]:
$$L_{total} = L_{rec} + 2.0 \cdot L_{asr} + 0.8 \cdot L_{clap}$$

其中 $L_{rec}$ 是重建交叉熵损失。ASR loss 权重 (2.0) > CLAP loss 权重 (0.8),表明语义锚定优先于风格捕获 [agent 解读]。

#### 设计选择 2: Hierarchical Decoding Strategy [§2.2, Eqs. 2-5]

**做了什么**: LLM 在每个时间步 j 产生 hidden state $T_{h,j}$,然后 2 层轻量 decoder 依次预测三种 token:

1. **Content Foundation** [Eq. 2-3]: $T_{h,j} = p(T_{h,j} | T_t, T_{s,:j})$ → $T_{c,j} = p(T_{c,j} | T_{h,j})$
2. **Style Rendering** [Eq. 4]: $T_{p,j} = p(T_{p,j} | T_{h,j}, T_{c,j})$
3. **Final Token Generation** [Eq. 5]: $T_{s,j} = p(T_{s,j} | T_{h,j}, T_{c,j}, T_{p,j})$

生成的 speech token $T_{s,j}$ 反馈给 LLM 作为下一步的输入。

**为什么先 content 再 style** [论文原文]: 作者认为这遵循了语音的内在层级——先建立语义基础,再叠加风格细节,最后渲染完整声学表示 [§1]。这种结构化生成"dramatically enhances the model's ability to execute instructions with precision" [§1]。

**为什么用条件依赖而非并行** [论文原文]: 消融实验 (Table 3) 证明并行预测三种 token 的效果更差,因为"an explicit conditional dependency is needed for effective output structuring" [§3.2.3]。单步直接预测 speech token 也更差,证明"structured intermediate representations"是必要的 [§3.2.3]。

#### 设计选择 3: 训练正则化 [§2.2]

两种正则化策略防止模型过度依赖单一信息源:

1. **Stochastic masking**: 随机遮蔽 hidden states 和 prompt tokens,并将 token logits 与 token embeddings concat 作为 decoder 输入 [§2.2]——迫使模型整合所有信息源
2. **Auxiliary projection**: 额外的线性层将 LLM hidden states 直接投影为 speech tokens [Fig 2(b)]——确保 LLM 内部表征保持声学 grounding [§2.2]

**为什么需要 auxiliary projection** [agent 解读]: 如果没有这个约束,LLM 可能会退化为仅生成适合 hierarchical decoder 处理但缺乏声学信息的 hidden states。Auxiliary projection 起到了类似 multi-task regularization 的作用,保持 hidden states 的声学丰富度。

### 训练策略

训练分两阶段 [§3.1]:

**阶段 1: Speech Token Codec**
- 架构: 5 层 conformer extractor + 4 层 causal transformer combiner
- 硬件: 4× NVIDIA 4090 GPU
- 优化器: AdamW, lr = 1e-4
- 训练: 50 epochs

**阶段 2: LLM + Hierarchical Decoder**
- LLM backbone: Qwen2.5-0.5B
- Decoder: 2 层 auto-regressive transformer, 固定长度 3 (对应 content/prompt/speech 三种 token)
- 硬件: 4× NVIDIA 4090 GPU
- 优化器: AdamW, lr = 1e-5 (比 codec 阶段低 10x)
- 训练: 16 epochs

Codec 训练完成后冻结,再训练 LLM 和 decoder [agent 解读: 论文未明确提及是否 fine-tune codec,但两阶段的 lr 差异和描述暗示 codec 是预训练后固定的]。

## 实验

| 指标 | HD-PPT | CosyVoice 2 | EmoVoice-PP | CosyVoice | PromptTTS | PromptStyle | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOS-N ↑ | **4.108±0.105** | 3.920±0.112 | 3.694±0.123 | 3.240±0.138 | 2.920±0.137 | 2.674±0.145 | TextrolSpeech+EmoVoice-DB | [Table 1] |
| MOS-S ↑ | **4.167±0.103** | 3.885±0.116 | 3.594±0.128 | 3.028±0.149 | 2.601±0.148 | 2.420±0.147 | TextrolSpeech+EmoVoice-DB | [Table 1] |
| DNSMOS ↑ | 3.84 | 3.83 | **3.87** | 3.77 | 3.65 | 3.68 | TextrolSpeech+EmoVoice-DB | [Table 1] |
| EMO-SIM ↑ | **0.753** | 0.714 | 0.613 | 0.635 | 0.588 | 0.529 | TextrolSpeech+EmoVoice-DB | [Table 1] |
| WER ↓ | 5.18% | 5.71% | 8.56% | 6.10% | **4.38%** | 17.92% | TextrolSpeech+EmoVoice-DB | [Table 1] |

**消融 1: Preference Tokens** [Table 2]

| 变体 | DNSMOS ↑ | EMO-SIM ↑ | WER ↓ |
| --- | --- | --- | --- |
| w/o Content-Pref | 3.76 | 0.742 | 8.04% |
| w/o Prompt-Pref | 3.76 | 0.728 | 5.49% |
| w/o Dual-Pref | 3.73 | 0.716 | 10.10% |
| w/o Instruct Text | 3.78 | 0.605 | 5.44% |
| **Proposed** | **3.84** | **0.753** | **5.18%** |

关键发现:
- 去掉 content-pref → WER 大幅上升 (5.18% → 8.04%),证明 content tokens 锚定语义完整性 [§3.2.2]
- 去掉 prompt-pref → EMO-SIM 下降 (0.753 → 0.728),证明 prompt tokens 捕获风格细微差别 [§3.2.2]
- 去掉 instruct text → EMO-SIM 暴跌 (0.753 → 0.605),证明模型的风格控制确实来自 prompt 而非数据集偏差 [§3.2.2]

**消融 2: Decoding Strategy** [Table 3]

| 策略 | DNSMOS ↑ | EMO-SIM ↑ | WER ↓ |
| --- | --- | --- | --- |
| Parallel | 3.76 | 0.736 | 5.99% |
| Single-step | 3.80 | 0.713 | 5.93% |
| **Hierarchical** | **3.84** | **0.753** | **5.18%** |

- 层级化 > 并行 > 单步,验证了条件依赖和结构化中间表征的必要性 [§3.2.3]
- RTF: 0.952 (hierarchical) vs 0.711 (single-step),增加 34% 计算开销 [§3.2.3]

## 局限性

1. **数据集规模有限**: 仅在 TextrolSpeech 和 EmoVoice-DB 两个公开数据集上验证,未展示在大规模工业数据(如 CosyVoice 3 使用的 350k 小时)上的效果 [agent 解读]
2. **推理速度降低**: RTF 从 0.711 增至 0.952,层级化解码的每步需要串行预测三次 [§3.2.3]
3. **多组件复杂度**: codec 训练 + LLM 训练两阶段,且 codec 本身有 extractor + 两个 FSQ + combiner + ASR decoder + CLAP module,增加了低资源语言适配难度 [§4]
4. **Vocoder 依赖**: 复用 CosyVoice 2 的预训练 vocoder,end-to-end 优化受限 [agent 解读]
5. **评估集较小**: 主观评估仅 18 名评估者 × 18 个样本,统计效力有限 [§3.1]
6. **未与最新系统对比**: 缺少与 CosyVoice 3、EmoVoice (1.5B)、VoxInstruct 等更强 baseline 的对比 [agent 解读]

## 点评

HD-PPT 提出了一个有吸引力的结构化假设: 将 speech token 空间显式分解为 content-preference 和 prompt-preference 两个子空间,可以弥合指令文本与语音 token 之间的 hierarchical mismatch。用 ASR + CLAP 双监督实现这种分解是 elegant 的——ASR 锚定语义,CLAP 对齐风格与文本描述,两者互补。

消融实验设计得好,Table 2 的"w/o Instruct Text" 变体证明了风格控制确实源自 prompt 而非数据偏差,这增加了结论的可信度。

但几个值得注意的问题:
1. **实验评估规模偏小**: 18 人 × 18 样本的 MOS 测试在当前 TTS 论文中偏薄,且缺少与 CosyVoice 3、EmoVoice 1.5B 等最新系统的对比。CosyVoice 2 已不是当前最强的 Instruct-TTS baseline。
2. **Prompt-preference codebook 只有 64**: 这意味着 HD-PPT 能表达的风格空间相当有限(最多 64 种离散风格组合)。对于 fine-grained 连续风格变化,这可能是瓶颈。
3. **推理开销 tradeoff**: 34% 的 RTF 增加换来 EMO-SIM 从 0.713 (single-step) 到 0.753 的提升,性价比是否足够取决于应用场景。

整体而言,核心思路——在 decoder 端引入有监督的 structured intermediate representation——有独立价值,且与领域内 Speech Factorization 的大趋势一致。方法的组件级验证(Table 2, Table 3)比较充分。

## 可复用的 idea

1. **ASR + CLAP 双监督 codec 训练**: 可迁移到任何需要 content/style 分离的 speech token 设计中。特别是 CLAP 监督让 style token 与文本描述直接对齐,比传统的 speaker/emotion classifier 监督更通用
2. **Hierarchical decoder 的条件依赖设计**: "先预测语义 → 基于语义预测风格 → 基于语义+风格预测声学" 的顺序,可以推广到其他多属性条件生成任务
3. **Stochastic masking + logit-embedding concat 正则化**: 防止 decoder 退化为只依赖单一信息源,适用于任何多源条件生成
4. **Auxiliary linear projection 保持 LLM hidden state 的声学 grounding**: 简单有效的多任务正则化技巧,防止 LLM 内部表征在训练过程中偏离声学信息

---

> [!review] 审阅结论: pass (2026-06-04)
> 五个原则均通过。2 个 low issues (template-compliance + traceability-gap),不阻塞。
> 详见 `_review/HD-PPT-review.yml`

---

检索命中: [[LLM-basedTTS]], [[SpeechTokenizer]], [[CosyVoice2]], [[InstructedSpeechGeneration]], [[SpeechFactorization]], [[SemanticvsAcousticTokens]] | 过滤: [[FiniteScalarQuantization]](pending-review), [[Instruction-GuidedSpeechSynthesis]](pending-review), [[Audio-LanguagePretraining]](pending-review) | 未命中但可能相关: 无
