---
type: paper
tier: deep
title: "Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages"
arxiv_id: "2303.01037"
source: "Sources/USM.pdf"
authors: [Yu Zhang, Wei Han, James Qin, Yongqiang Wang, Ankur Bapna, Zhehuai Chen, Nanxin Chen, Bo Li, Vera Axelrod, Gary Wang, Zhong Meng, Ke Hu, Andrew Rosenberg, Rohit Prabhavalkar, Daniel S. Park, Parisa Haghani, Jason Riesa, Ginger Perng, Hagen Soltau, Trevor Strohman, Bhuvana Ramabhadran, Tara Sainath, Pedro Moreno, Chung-Cheng Chiu, Johan Schalkwyk, Francoise Beaufays, Yonghui Wu]
year: 2023
venue: "arXiv"
tags: [ASR, multilingual, self-supervised-learning, speech-representation, pre-training, conformer, BEST-RQ, text-injection, long-form-ASR, speech-translation, adapter, noisy-student-training]
concepts: ["[[Self-SupervisedSpeechRepresentation]]", "[[SpeechLanguageModel]]", "[[AudioUnderstanding]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓(confirmed), [[SpeechTokenizer]]✓(confirmed) | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review), [[LLM-enhancedASR]](pending-review) | 未命中但可能相关: 无

**已有知识要点**:
- [[SpeechLanguageModel]]: SpeechLM 通过端到端 speech token 建模绕过 ASR+LLM+TTS 级联管线的三大问题 (信息丢失、高延迟、错误累积); USM 走的是不同路线 -- 不是端到端 SpeechLM,而是大规模预训练 encoder + 任务特定 transducer 的经典两阶段范式 ✓
- [[SpeechTokenizer]]: 自监督 tokenizer (HuBERT, w2v-BERT) 通过 masked prediction 学习表征后聚类离散化; USM 的 BEST-RQ 也是 masked prediction 范式,但量化方式完全不同 -- 使用随机投影量化而非学习量化 ✓
- [[Self-SupervisedSpeechRepresentation]] [待确认]: USM 的 BEST-RQ 预训练属于 Masked Prediction 范式但做了关键简化: 去掉了 Gumbel-Softmax/k-means 等复杂量化模块,用随机冻结的投影矩阵+余弦相似度代替; 这一设计在 2B 参数 + 12M 小时数据规模下显著优于 w2v-BERT
- [[CodebookCollapse]] [相关]: BEST-RQ 通过设计规避了 codebook collapse 问题 -- 随机冻结的 codebook 不参与梯度更新,因此不存在 collapse 可能

**KB 定位**: USM 是 speech foundation model 中"大规模预训练 encoder"路线的里程碑工作。与同期的 Whisper (大规模弱监督) 形成两条互补路线: USM 主打 **无监督预训练 + 少量标注微调** (12M h 无标注 + 90k h 标注), Whisper 主打 **大规模弱监督** (680k h 弱标注)。USM 用 1/7 的标注数据达到了可比甚至更好的多语言 ASR 性能。后续 XEUS (2024) 在开放性和语言覆盖方面进一步推进了 USM 开创的方向。

## 速查

> [!summary] 速查
> - **一句话**: 提出 Universal Speech Model (USM), 通过三阶段训练 (BEST-RQ 无监督预训练 → MOST 多模态监督预训练 → 任务微调) 在 2B Conformer 上实现 100+ 语言的 SOTA ASR/AST,用 1/7 标注数据匹敌 Whisper
> - **路线**: 12M h 无标注语音 → BEST-RQ (masked prediction + random-projection quantization, 16-codebook multi-softmax) → MOST (BEST-RQ + text-injection + ASR loss + modality matching) → CTC/LAS/RNN-T transducer 微调 [Fig 1]
> - **指标**: YouTube 73 语言 <30% WER; FLEURS 102 语言 WER 17.4 (6.5 CER), 相对 Whisper 66% WER 改进 [Table 3]; CoVoST 2 BLEU 30.7 (SOTA) [Table 3]; SpeechStew WER 7.0 [Table 3]; frozen encoder + 2% adapter 仅微弱性能损失 [Table 3]
> - **可借鉴**: (1) BEST-RQ 随机投影量化 -- 去掉复杂可学习量化模块,用随机冻结矩阵实现同等或更好的 SSL,在大规模时训练稳定性远优于 w2v-BERT; (2) MOST 多目标预训练 -- 同时优化 BEST-RQ + text-injection + ASR loss, 使 speech/text 表征共享同一空间; (3) chunk-wise attention -- 解决深层网络 long-form 性能退化的简单有效方案; (4) 2% adapter 冻结 encoder -- 可扩展到 100+ 语言/任务
> - **局限**: 12M h YouTube 预训练数据未公开,无法复现 [§3]; 标注数据基于 YouTube 用户上传字幕,质量参差; 仅评估 ASR/AST, 未探索 TTS/理解等任务; 架构相对保守 (Conformer+CTC/LAS/RNN-T), 非端到端 SpeechLM

## 核心问题

**USM 要解决什么问题?** [论文原文]

扩展语音识别到全世界所有口语语言面临的根本挑战 [§1]:
1. **标注数据稀缺**: 常规监督训练需要人工转写,对尾部语言 (tail languages) 极其昂贵甚至不可能 [§1]
2. **单语言模型不可扩展**: 为每种语言单独训练模型在计算和维护上不现实 [§1]
3. **大模型利用大数据更有效**: 研究表明单一大模型比多个小模型更能有效利用大数据集 [§1, ref 1,4]

**核心思路**: [论文原文] 利用三类数据 (大量无标注多语言语音 + 无标注文本 + 少量标注语音) 通过分阶段训练一个 2B 参数的通用语音模型: "large amounts of unpaired multilingual speech and text data and smaller amounts of transcribed data can contribute to training a single large universal ASR model" [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

USM 采用三阶段训练 pipeline,每阶段使用不同数据和目标 [§1.1, Fig 1]:

**Stage 1: 无监督预训练 (BEST-RQ)** — 占总计算量 ~80% [Fig 1]
- 在 12M 小时无标注多语言语音 (YT-NTL-U, 300+ 语言) 上预训练 2B Conformer encoder
- 目标: BERT-style masked prediction + random-projection quantization [§2.2]

**Stage 2: 多目标监督预训练 (MOST)** — 占总计算量 ~15% [Fig 1]
- 同时利用三类数据: 无标注语音 + 无标注文本 (28B sentences, 1140 语言) + 标注语音
- 联合优化 BEST-RQ loss + text-injection losses (ASR loss + modality matching + duration modeling + text reconstruction) [§2.5]

**Stage 3: 任务微调** — 占总计算量 ~5% [Fig 1]
- 根据下游任务选择 CTC/LAS/RNN-T transducer
- 标注数据: YT-SUP+ (90k h, 73 语言) 或 Pub-S (10k h, 102 语言) [§1.1]

[agent 解读] 三阶段设计的核心洞察是**将计算密集型预训练 (Stage 1+2) 与低成本任务适配 (Stage 3) 解耦**。一旦投入大量算力完成预训练,后续每个新任务/新语言的适配成本极低 (仅 5% 计算量或更少)。

### 关键设计选择

#### 1. BEST-RQ: 为什么选随机投影量化而非学习量化 [§2.2, §5.3]

BEST-RQ (BERT-based Speech pre-Training with Random-projection Quantizer) 的核心创新在于量化方式:
- 使用**随机初始化、冻结**的投影矩阵将语音特征投射到 codebook embedding 空间 [§2.2]
- codebook 向量也是**随机选取并冻结**,不参与梯度更新 [§2.2]
- 通过余弦相似度找最近的 codebook 向量作为 discrete label [§2.2]

[论文原文] 为什么优于 w2v-BERT: "while w2v-BERT pre-training has proven to be an effective method...it requires an additional quantization module which introduces more complexity. As we increase the model size and language coverage, the learnt codebook module proves costly to tune and can impede progress of model development. Meanwhile, the BEST-RQ algorithm does not require such a module, making it a more scalable method for pre-training." [§2.2]

实验验证 [Table 6]: 0.6B 参数时 w2v-BERT 和 BEST-RQ 性能接近 (CoVoST BLEU 20.4 vs 20.7); 但 2B 参数时 BEST-RQ 大幅领先 (22.4 vs 26.6), 相对提升 19%。[agent 解读] 这说明 w2v-BERT 的可学习量化模块在极大规模下成为瓶颈 -- 可能因为 codebook collapse 或训练不稳定性 (与 [[CodebookCollapse]] 概念页描述一致)。

#### 2. Multi-Softmax: 为什么用多 codebook [§2.2.1, §5.1]

[论文原文] 使用 N=16 个独立的 softmax 层 + N 个独立的随机 codebook, 每个 softmax 预测对应 codebook 的量化标签,等权求和 [§2.2.1]

[论文原文] 效果: "> 5% relative improvement in ASR and AST benchmarks by increasing the number of the softmax groups from 1 to 16" [§5.1]; 同时 "significantly reduces performance variation across different pre-training runs and improves convergence speed" [§5.1]

[agent 解读] multi-softmax 的效果类似于 ensemble: 每个 codebook 从不同角度量化语音特征,提供互补的监督信号,使预训练更稳定更丰富。

#### 3. MOST (Multi-Objective Supervised Pre-Training): 为什么做 speech-text 模态对齐 [§2.5]

MOST 在 BEST-RQ 预训练基础上增加 text-injection 训练,其架构 [Fig 5]:
- **Speech-only encoder**: CNN feature extractor + 1 层 Conformer (从 BEST-RQ 初始化) [§2.5]
- **Text-only encoder**: Embedding + duration upsampler + Conformer block (随机初始化) [§2.5]
- **Shared encoder**: BEST-RQ 预训练的 Conformer-2B [§2.5]
- **Decoder**: 随机初始化的 transducer [§2.5]

三类 loss 联合优化 [§2.5]:
1. **BEST-RQ loss**: 无标注语音 → shared encoder → BEST-RQ softmax [§2.5]
2. **ASR loss + consistency loss**: 标注语音 → speech encoder → shared encoder → decoder; 同时 text encoder 的输出与 speech encoder 的输出做一致性约束 (text encoder 学习对齐, speech encoder 冻结) [§2.5]
3. **Text reconstruction loss**: 无标注文本 → text encoder → 部分 masking → shared encoder → decoder → 重建原文 [§2.5]

[论文原文] MOST 的两个核心收益 [§2.5]:
- (i) "Training with paired speech and text data with alignment losses results in learning speech representations that are better aligned with text" — 改善 ASR/AST
- (ii) "Training simultaneously on unlabeled text...improves the robustness of learned representations, especially on low resource languages and domains, also generalizing to new languages with no paired data seen during training" — 提升低资源泛化

[agent 解读] MOST 的深层价值是: 通过 text-injection, 28B sentences 的文本语言知识被注入到 speech encoder 的表征空间中。这让 USM 在遇到从未见过配对数据的新语言时,仍能利用该语言的文本知识辅助识别。

#### 4. Chunk-wise Attention: 为什么解决 long-form 退化 [§2.4]

[论文原文] 核心问题: "local self attention...creates a significant receptive field mismatch between training and inference" [§2.4]。具体来说:
- 训练时用 ≤30s 语音片段 + 128 帧 local self attention [§2.4]
- 推理时喂入分钟甚至小时级音频, 由于 context leaking (每层 attention 扩展感受野), 32 层网络的实际感受野 > 327 秒 [§2.4]
- 这种训练-推理不匹配导致"high deletion errors" [§2.4]

[论文原文] 解决方案: 将 attention 限制在固定长度的 chunk (8 秒) 内,但不影响其他层 (如 convolution layers) 的上下文访问 [§2.4, Fig 4]

[论文原文] 与 Whisper 30s 分段的区别: "we only chunk the attention state, and allow the decoder to access the entire encoder output" [§2.4]。此外 CTC/RNN-T 解码器 "have not been observed to hallucinate compared to attention-based sequence-to-sequence decoders" [§2.4]

实验验证 [Table 7]: chunk-wise attention (CW-8s) 在 YouTube long-form en-US WER 从 16.2 降至 12.5 (相对 23% 改进); ko-KR 从 26.2 降至 19.5 (相对 26% 改进)。

#### 5. Residual Adapter: 为什么用 2% 参数适配 [§2.6]

[论文原文] 动机: "fine-tuning the pre-trained USM individually for various domains and tasks becomes prohibitively expensive" [§2.6]

方案: 每个 Conformer block 加两个并行 residual adapter, 总参数仅占 2%,按语言/任务动态加载 [§2.6]

[论文原文] 额外好处: "training the adapter versus fine-tuning the entire model can reduce over-fitting especially when the training data is limited" [§2.6]

### 训练策略

**数据规模** [§3]:
| 数据集 | 类型 | 规模 | 语言数 |
|--------|------|------|--------|
| YT-NTL-U | 无标注语音 | 12.1M h | 300+ |
| Web-NTL | 无标注文本 | 28B sentences | 1140 |
| YT-SUP+ | 标注语音 | 190k h (90k 真标注 + 100k pseudo) | 73 |
| Pub-S | 标注语音 | 10k h | 102 |

[§3.1, §3.2]

**模型规格** [Table 2]:
- Conformer-2B: 32 层, dimension 1536, 16 attention heads, conv kernel 5, 共 2B 参数
- Conformer-0.6B: 24 层, dimension 1024, 8 heads (用于消融实验)

**MOST 课程学习** [§2.5]: 先仅用标注数据训练 20k steps 以学习稳定的 decoder 对齐; 然后加入无标注文本联合训练 100k steps [§2.5]

**基础设施**: GShard + GSPMD 框架在 TPU 上训练 [§2.7]

## 实验

| 指标 | 本文 (USM) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| YouTube 18-lang WER | USM-LAS 14.4 / USM-CTC 13.7 | Whisper 27.8 | YT long-form 18 langs | [Table 3] |
| YouTube 73-lang WER | USM-LAS 19.0 / USM-CTC 18.7 | (无可比系统) | YT long-form 73 langs | [Table 3] |
| CORAAL WER | USM-LAS 29.8 / USM-CTC 26.7 | Whisper-longform 29.1 | CORAAL (AAVE) | [Table 3] |
| SpeechStew WER | USM-M (ft) 7.0 | BigSSL 7.5 | SpeechStew en-US | [Table 3] |
| FLEURS 62-lang WER (CER) | USM-CTC 13.2 (12.5) | Whisper 36.6 | FLEURS 62 langs | [Table 3] |
| FLEURS 102-lang WER (CER) | USM-M (ft) 17.4 (6.5) | Maestro 14.8 | FLEURS 102 langs | [Table 3] |
| FLEURS 102-lang adapter | USM-M-adapter 17.6 (6.7) | USM-M (ft) 17.4 (6.5) | FLEURS 102 langs | [Table 3] |
| CoVoST 2 BLEU | USM-M (ft) 30.7 | 之前 SOTA 28.7 | CoVoST 2 XX→En | [Table 3] |
| BEST-RQ 0.6B vs 2B CoVoST | 2B: 26.6 | 0.6B: 20.7 | CoVoST 2 | [Table 5] |
| w2v-BERT vs BEST-RQ (2B) | BEST-RQ: 26.6 | w2v-BERT: 22.4 | CoVoST 2 | [Table 6] |
| CW-8s vs LSA-128 en-US WER | CW: 12.5 | LSA: 16.2 | YT long-form en-US | [Table 7] |

**关键实验发现**:

1. **1/7 标注数据匹敌 Whisper** [Table 3]: [论文原文] USM-CTC 在 YouTube 18 语言上 WER 13.7 vs Whisper 27.8, 相对 51% 改进, 但 USM 仅用 90k h 标注 vs Whisper 680k h [§4.1]。[agent 解读] 这证明了无监督预训练的核心价值: 12M h 的无标注数据通过 BEST-RQ 学到的表征,远比简单增加弱标注数据更高效。

2. **FLEURS 62 语言 WER 减少 66%** [Table 3]: [论文原文] USM-CTC generic model (无 in-domain 微调) WER 12.5, Whisper WER 36.6, 尽管 USM 的多语言标注数据更少 (90k vs Whisper 117k, 排除 en-US) [§4.1]

3. **MOST 对 AST 贡献最大** [Table 3]: USM (无 MOST) CoVoST BLEU 28.7 → USM-M (有 MOST) 30.7, +2.0 BLEU。[论文原文] 原因是 MOST "can use both speech and text as training input. By introducing text-to-text machine translation (MT) data during fine-tuning, USM-M is able to achieve an unprecedented > 30 BLEU" [§4.5]

4. **Adapter 近乎无损** [Table 3]: USM-M-adapter (冻结 encoder + 2% adapter) FLEURS WER 17.6 vs USM-M full fine-tuning 17.4。[论文原文] "by adding only 2% to the total number of parameters...only performs slightly worse than the fine-tuning baselines" [§4.3]

5. **Noisy Student Training 对尾部语言有效** [Table 4]: [论文原文] 用 USM-LAS-Adapter (仅 FLEURS 10h 训练) 作为 teacher, 对 YT-NTL 中无标注尾部语言数据做 pseudo-labeling, student 模型在 Shona 上 WER 从 29.1 降至 22.2 (31% 相对改进) [§4.4]

6. **2B Conformer TPU serving 仅 3.9x 慢于 100M streaming 模型** [Table 8]: USM-2B 在 TPUv4i batch=32 上 1/RTF=827, 证明大模型可高效部署 [§5.5]

## 局限性

1. **数据不可复现** [§3]: [论文原文] 12M h YouTube 预训练数据和 90k h YouTube 标注数据均为 Google 内部数据,社区无法复现。这是 USM 与后续开源工作 (XEUS) 的核心差异。

2. **标注质量依赖用户上传字幕**: [agent 解读] YT-SUP 基于 YouTube 用户自行上传的字幕,质量可能参差不齐 (拼写错误、时间对齐不准、非专业标注)。论文未讨论标注质量控制。

3. **仅评估 ASR/AST 任务**: [agent 解读] 未探索 encoder 表征在 TTS、speaker verification、emotion recognition 等其他语音任务上的表现。虽然 MOST 的 speech-text 共享空间理论上应有更广泛用途,但论文未验证。

4. **long-form hallucination 未完全解决**: [论文原文] USM-LAS 仍然需要 segmented decoding 来减少 long-form 退化 [§4.1]。只有 USM-CTC 不存在此问题。[agent 解读] 这暗示 attention-based decoder 的 hallucination 是架构层面的本质问题。

5. **语言覆盖的长尾问题**: [agent 解读] YT-513-U 中 188 种语言不足 100 小时数据 [§3.1],这些语言的预训练效果存疑。论文主要报告了高/中资源语言的结果。

## 点评

**历史地位**: [agent 解读] USM 与 Whisper (OpenAI, 2022) 共同定义了 2023 年大规模多语言 ASR 的两条技术路线:
- **Whisper 路线**: 大规模弱监督 (680k h 弱标注),端到端训练,简单暴力
- **USM 路线**: 无监督预训练 (12M h 无标注) + 少量监督微调 (90k h),分阶段解耦,数据效率更高

USM 用实验证明了后者在多语言场景下的优势: 无标注数据获取成本远低于弱标注数据 (不需要字幕/转写来源),且预训练 encoder 可重用于无限多的下游任务。

**方法论贡献**:
1. **BEST-RQ 的可扩展性验证**: 证明了随机冻结量化在极大规模 (2B 参数, 12M h 数据) 下的优越性。[agent 解读] 这一发现有违直觉 -- 更"聪明"的可学习量化 (w2v-BERT) 反而不如更"简单"的随机量化。原因在于: 简单意味着更少的超参数、更稳定的训练、零 codebook collapse 风险。这是"scaling 改变最优设计"的又一案例。
2. **MOST 的 speech-text 对齐**: 首次在 1000+ 语言的文本数据和 300+ 语言的语音数据之间建立共享表征空间,使得文本语言知识可迁移到语音任务。[agent 解读] 这一思路后来被广泛采用,成为多模态预训练的标准方法之一。
3. **Chunk-wise attention**: 用极简方案 (限制 attention 到固定 chunk) 解决了深层网络的 long-form 退化问题。[agent 解读] 其优雅之处在于: convolution layers 仍可跨 chunk 共享信息,只是 attention 被限制,因此不像硬分段那样完全切断上下文。

**与 KB 已有知识的关系**:
- USM 的 BEST-RQ 是 [[Self-SupervisedSpeechRepresentation]] 演进中"简化量化"方向的代表,与 wav2vec 2.0 的 Gumbel-Softmax 和 HuBERT 的 k-means 形成三种量化范式
- USM 的 MOST 是 speech-text 模态对齐的早期大规模实践,为后续 [[SpeechLanguageModel]] 中的 text-pretrained LM 初始化 (TWIST, SPIRIT-LM) 提供了思路对比
- USM-CTC 的 long-form robustness 与当前 SpeechLM 的 streaming/full-duplex 方向形成呼应: CTC 不 hallucinate 的优势在实时场景中尤为重要

## 可复用的 idea

1. **BEST-RQ 随机投影量化**: 在任何需要 speech tokenization 的场景下,如果可学习量化不稳定或 codebook collapse,可直接改用随机冻结投影 + 随机冻结 codebook。实现极其简单 (随机矩阵 + 余弦相似度),且经 USM 验证在极大规模下性能更好。[§2.2]

2. **Multi-softmax ensemble**: 对 masked prediction 预训练,使用 N 个独立的 softmax + codebook 并等权求和,减少训练方差、加速收敛。这一 trick 与模型架构/数据规模无关,可直接嫁接到任何 BERT-style SSL 训练中。[§2.2.1, §5.1]

3. **Chunk-wise attention for long-form**: 当 encoder 层数很深 (>20 层) 且推理长度远超训练长度时,将 self-attention 限制在固定 chunk (8s) 内,但保留其他层的跨 chunk 信息流。比 Whisper 的 30s 硬分段更优雅,比重新训练更廉价。[§2.4]

4. **Residual adapter for multi-task/multi-language**: 冻结预训练 encoder, 每个 Conformer block 加两个并行 adapter (总参数 2%),按任务/语言动态加载。这使得一个 encoder 支持 100+ 语言而不需要 100 个独立模型。[§2.6]

5. **MOST 的 curriculum 设计**: text-injection 训练先用标注数据 20k steps 稳定 decoder 对齐,再加入无标注文本。[agent 解读] 这个 curriculum 很关键 -- 如果一开始就用无标注文本,duration upsampler 没有可靠的对齐信号,会导致 text encoder 学到错误的时间对齐。[§2.5]

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 三阶段 pipeline 的 WHY 解释清晰; BEST-RQ vs w2v-BERT 的可扩展性论证有因果链; chunk-wise attention 的问题-方案逻辑完整 |
> | 可信赖 | pass | 数字型 claim 出处标注覆盖率 >85%; 指标 WER/CER/BLEU 使用正确; 关键数字 (1/7 标注、66% WER 改进) 有 Table 3 支撑 |
> | 可区分 | pass | [论文原文] / [agent 解读] 标注覆盖率约 85%; 推断性分析均使用 [agent 解读] 限定 |
> | 可定位 | pass | KB 背景含具体谱系定位 (USM vs Whisper 路线对比); 与 XEUS/w2v-BERT/HuBERT 的关系明确; frontmatter 完整 |
> | 不污染 | pass | 无新建概念页; 反向更新建议仅为追加 key_papers |
> 
> Issues: 3 (high: 0, medium: 0, low: 3)
> 详见 `_review/USM-review.yml`

---

检索命中: [[SpeechLanguageModel]]✓(confirmed), [[SpeechTokenizer]]✓(confirmed) | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review), [[LLM-enhancedASR]](pending-review) | 未命中但可能相关: 无
