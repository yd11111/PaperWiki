---
type: paper
tier: deep
title: "Comparing Discrete and Continuous Space LLMs for Speech Recognition"
arxiv_id: "2409.00800"
source: "Sources/DiscreteVsContinuousLLM-ASR.pdf"
authors: [Yaoxun Xu, Shi-Xiong Zhang, Jianwei Yu, Zhiyong Wu, Dong Yu]
year: 2024
venue: "Interspeech 2024"
tags: [ASR, LLM, speech-representation, discrete-tokens, continuous-representation, HuBERT, Whisper, LLaMA2, LoRA, K-means, LibriSpeech]
concepts: ["[[Self-SupervisedSpeechRepresentation]]", "[[LLM-enhancedASR]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[ModalityAdaptationforSpeechLLM]]", "[[SpeechTokenizer]]"]
models: []
tasks: []
datasets: ["[[LibriSpeech]]"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeechTokenizer]]✓, [[LLM-enhancedASR]], [[Self-SupervisedSpeechRepresentation]], [[Speech-LLMIntegrationTaxonomy]], [[ModalityAdaptationforSpeechLLM]], [[Single-codebookvsMulti-codebook]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本论文处于 Speech-LLM Integration Taxonomy 的交叉点 -- 同时涵盖 latent-representation-based (连续表征直入 LLM) 和 audio-token-based (离散 token 作为 LLM 输入) 两条路线。在已有 KB 中,latent-representation 路线的 modality adaptation 技术 (Conv downsampling, CTC compression, Q-Former) 已有详细记录; audio-token 路线中的 speech tokenizer 设计 (SSL k-means vs 监督式 CTC) 也有覆盖。但**两条路线在同一实验框架下的直接对比**在 KB 中是空白,本论文填补了这一缺口。
>
> **已有认知**: Self-Supervised Speech Representation 页记录了 HuBERT 的 masked prediction + k-means 范式及其 layer-wise 特性 (中间层捕获韵律/语义,最后层更偏声学); LLM-enhanced ASR 页记录了 LLM 作为 error corrector 的 GER 范式 (包括 N-best rescoring, TextInput 等); Modality Adaptation 页记录了 LoRA fine-tuning 和 adapter 设计的实践; Speech Tokenizer 页记录了 discrete vs continuous tokenizer 的演进,包括连续 VAE tokenizer 的新路线。
>
> **创新判断**: 本论文的主要贡献不在于新方法,而在于**系统性实验设计** -- 在统一框架下用 2x2 矩阵 (supervised/unsupervised x discrete/continuous) 对比四类语音表征,同时用两种 LM (from-scratch vs pretrained) 验证,这种对比在之前的工作中未出现。
>
> 检索命中: [[SpeechTokenizer]]✓, [[LLM-enhancedASR]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认], [[Speech-LLMIntegrationTaxonomy]][待确认], [[ModalityAdaptationforSpeechLLM]][待确认], [[Single-codebookvsMulti-codebook]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在统一框架下系统对比 discrete/continuous x supervised/unsupervised 四类语音表征在 LLM-based ASR 中的表现,发现 token 兼容性 (matched tokens) 是 pretrained LLM 场景下的关键因素
> - **路线**: Speech → Speech Encoder (HuBERT/Whisper/HuBERT-CTC/K-means) → Discrete tokens 或 Continuous embeddings → Adapter/Embedding → LM (JTFS LM 或 LLaMA2-7b LoRA) → Text
> - **指标**: WER 1.69%/3.03% on LibriSpeech test-clean/other (HuBERT-CTC xlarge + 4-gram + LLaMA2, #29); 从零训练最优 5.28%/9.74% (Whisper continuous, #20)
> - **可借鉴**: (1) 2x2 表征分类矩阵是清晰的实验设计范式; (2) matched tokens 对 pretrained LLM 的重要性可推广到 TTS 中的 speech token 设计; (3) HuBERT layer 16 优于 layer 24 的发现可指导 SSL tokenizer 层选择
> - **局限**: 仅在 LibriSpeech 单一数据集验证; JTFS LM 仅 10 层 Transformer (较小); 开源代码但模型规模有限; continuous JTFS LM 的 MSE+CE 双损失设计缺乏消融

## 核心问题

本论文试图回答: **在 LLM-based ASR 中,语音应该以什么形式输入 LLM?** 具体拆解为三个子问题:
1. Discrete vs Continuous 表征哪个更好? [§2.1, §4]
2. Supervised vs Unsupervised 特征提取哪个更好? [§2.1, §4]
3. From-scratch LM vs Pretrained LLM 对表征选择有什么影响? [§4.1, §4.2]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由两部分组成 [Fig 1a]:
1. **Speech Encoder**: 将原始语音转为 discrete tokens 或 continuous embeddings
2. **Language Model**: 接收 tokens/embeddings 并自回归生成文本转写

作者设计了 **2x2 分类矩阵** 将语音表征分为四类 [§2.1]:

| | Unsupervised | Supervised |
|---|---|---|
| **Continuous** | HuBERT 各层 embedding [Fig 1b] | HuBERT-CTC embedding / Whisper encoder [Fig 1c] |
| **Discrete** | HuBERT + K-means 聚类 [Fig 1d] | HuBERT-CTC logits → text tokens [Fig 1e] |

### 关键设计选择

**四类 Speech Encoder 的设计细节:**

1. **Continuous Unsupervised** [Fig 1b]: 预训练 HuBERT-large 的第 0/8/16/24 层输出。无需额外训练,直接用 SSL 模型各层表征。[论文原文]

2. **Continuous Supervised** [Fig 1c]: (a) HuBERT-CTC -- HuBERT 加 prediction layer 降维到 32 维,用 CTC loss 在语音-文本对上优化; (b) Whisper encoder -- encoder-decoder 模型的 encoder 输出。两者都利用了标注数据。[论文原文]

3. **Discrete Unsupervised** [Fig 1d]: 在 HuBERT 各层输出上训练 K-means (500/1000/1500 clusters),将连续表征量化为离散 token 并去重。[论文原文]

4. **Discrete Supervised** [Fig 1e]: HuBERT-CTC 的 softmax 输出映射为文本 prompt,设计了 @1-@6 共 6 种 TextInput 格式 [Fig 1f],包括字符级分隔、概率附加、词级概率、4-gram rescoring 等。[agent 解读] 这种设计的核心洞察是: 将 CTC 输出 **转为文本域** 后直接走 LLaMA2 tokenizer,使 speech tokens 与 LLM 的预训练 token 空间完全匹配。

**两种 LM 的建模差异** [Fig 2]:

- **Discrete LM** [Fig 2a]: tokens → embedding → Transformer → autoregressive text generation。JTFS LM 全参数联合训练; LLaMA2 用 LoRA (rank=16, alpha=16) fine-tune。[论文原文]

- **Continuous LM** [Fig 2b]: embeddings → adapter (2-layer MLP) → Transformer → text generation。对 JTFS LM,输出端不做离散化,直接将连续 output 反馈到输入端 (虚线路径),避免信息损失。对 LLaMA2,仍需离散化 output 走 tokenizer 再反馈。[论文原文]

[agent 解读] JTFS LM 在连续场景下的特殊设计 (output→input 连续传递) 是本文的一个巧妙之处: 它从根本上消除了自回归过程中的量化信息损失,但代价是模型只能从零训练,无法利用 pretrained LLM 的能力。

### 训练策略

- **JTFS LM**: 10 层 Transformer, dim=1024, 8 heads,全参数从零训练 [§3]
- **LLaMA2-7b**: LoRA (rank=16, alpha=16, target=gate/down/up projection, dropout=0.05), LR=1e-5 [§3]
- **Loss**: Discrete 场景和 LLaMA2 continuous 场景用 CELoss; JTFS LM continuous 场景用 CELoss + α*MSELoss (α=100),MSE 约束 Transformer 输入输出一致性 [§2.3]
- **硬件**: 8x A100 GPUs [§3]

## 实验

### JTFS LM 结果 [Table 1]

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Continuous Supervised (Whisper, JTFS) | 5.28/9.74 WER | Discrete Unsup (HuBERT K1500 L24) 70.13/71.91 | LibriSpeech clean/other | [Table 1, #20 vs #12] |
| Continuous Supervised (HuBERT-CTC L16, JTFS) | 7.17/12.21 WER | Continuous Unsup (HuBERT L16, JTFS) 49.18/51.90 | LibriSpeech clean/other | [Table 1, #18 vs #15] |
| Continuous Unsup (HuBERT L24, JTFS) | 41.13/67.55 WER | Continuous Unsup (HuBERT L16, JTFS) 49.18/51.90 | LibriSpeech clean/other | [Table 1, #19 vs #15] |

**JTFS LM 关键发现:**
- **Continuous >> Discrete**: #15 (70.13) vs #18 (7.17) 在同一 encoder 下差异巨大 [论文原文: "Discrete tokens undergo significant information loss during clustering"] [§4.1]
- **Supervised >> Unsupervised**: #18 (7.17) vs #15 (49.18) [§4.1]
- **Whisper > HuBERT**: #20 (5.28) vs #18 (7.17) [论文原文: "Whisper's enhanced feature extraction ability, indicating that supervised training can further improve the alignment"] [§4.1]
- **K-means clusters 越多越好**: #13→#14→#15 (119.01→87.33→70.13) [§4.1]
- **HuBERT Layer 16 > Layer 24**: #18 (7.17) vs #19 (41.13/67.55),这与 SSL 文献中的发现一致 -- HuBERT 最后层可能捕获更多声学特征而非语义特征 [§4.1]

### LLaMA2 结果 [Table 2]

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Discrete Supervised (HuBERT-CTC xlarge + 4-gram + LLaMA2) | **1.69/3.03 WER** | HuBERT 原文 1.8% WER | LibriSpeech clean/other | [Table 2, #29] |
| Discrete Supervised (TextInput@5 + LLaMA2) | 1.72/3.57 WER | HuBERT-CTC + 4-gram 1.82/3.59 | LibriSpeech clean/other | [Table 2, #27 vs #22] |
| Continuous Unsupervised (HuBERT L24 emb + LLaMA2) | 13.05/16.77 WER | Discrete Unsup (K-means + LLaMA2) 65.26/75.85 | LibriSpeech clean/other | [Table 2, #31 vs #30] |
| Continuous Supervised (HuBERT-CTC L24 emb + LLaMA2) | 6.26/7.09 WER | Continuous Unsup 13.05/16.77 | LibriSpeech clean/other | [Table 2, #32 vs #31] |

**LLaMA2 关键发现:**

- **Matched tokens 的关键作用**: Discrete Supervised #27 (1.72/3.57) 大幅优于 Continuous Supervised #33 (9.99/11.93),尽管连续表征理论上信息更丰富 [论文原文: "This advantage is due to #27 generating token sequences that matched with LLaMA2's pretraining tokens, enhancing compatibility and performance"] [§4.2]。[agent 解读] 这是全文最关键的发现: pretrained LLM 对 token 格式有强偏好,能匹配预训练分布的 discrete tokens 胜过信息更多但分布不匹配的 continuous embeddings。

- **LLM as Error Corrector**: LLaMA2 在 discrete supervised 场景中实质上是二次纠错器,利用 long-context LM probabilities 改善 HuBERT-CTC 的初始识别 [论文原文] [§4.2]

- **Pretrained LLM >> From-scratch LM**: #31 (13.05) vs #19 (41.13) 在 continuous unsupervised 上,LLaMA2 凭借预训练知识大幅提升 [§4.2]

- **Discrete Unsupervised 在 LLaMA2 下仍很差**: #30 (65.26/75.85),K-means tokens 与 LLaMA2 预训练 token 空间不兼容 [§4.2]

- **SOTA 结果**: 升级到 HuBERT-CTC xlarge + TextInput@5 达到 1.69/3.03,是当时开源模型在 LibriSpeech 上的最佳结果 [§4.2]

## 局限性

1. **单一数据集**: 全部实验仅在 LibriSpeech (960h) 上进行,缺乏跨数据集/跨语言验证。LibriSpeech 是朗读体英语,结论对自发语音/嘈杂环境的泛化性未知。[agent 解读]

2. **JTFS LM 规模偏小**: 10 层 Transformer 与 LLaMA2-7b 规模差异过大,两者的对比存在 confounding factor (不仅是 pretrain 的差异,还有模型容量差异)。[agent 解读]

3. **Continuous JTFS LM 的 MSE loss 缺乏消融**: α=100 直接固定,未探索 MSE loss 权重对性能的影响; 连续 output→input 反馈机制的必要性也未消融。[agent 解读]

4. **TextInput 格式设计的搜索空间有限**: @1-@6 的 prompt 格式是人工设计的,可能存在更优的格式但未探索。[agent 解读]

5. **未涉及 streaming/延迟分析**: 实际部署中 discrete 和 continuous 路线的推理延迟差异可能是重要考量,但论文未讨论。[agent 解读]

## 点评

**优点:**
- 实验设计清晰: 2x2 矩阵 + 两种 LM 的组合提供了全面覆盖,是一份有价值的 reference
- "Matched tokens" 这一发现对 Speech-LLM 集成路线有实际指导意义
- HuBERT layer-wise 分析 (layer 16 > 24) 与 SSL 领域的发现相互印证
- 代码开源,可复现

**不足:**
- 论文标题承诺的 "comprehensive comparison" 受限于单一数据集和有限的 LM 规模
- 对 "matched tokens" 这一核心发现的分析停留在直觉层面,缺乏定量解释 (如 token 分布距离分析)
- 连续表征在 LLaMA2 下表现不佳,但论文未深入分析 adapter (2-layer MLP) 是否是瓶颈 -- 更复杂的 adapter (如 Q-Former) 可能改变结论

**定位:**
本文是一篇以实验驱动的对比研究,价值主要在实验发现而非方法创新。在 Speech-LLM Integration Taxonomy 的语境下,它为 "latent-representation vs audio-token" 的选择提供了定量证据: **当使用 pretrained LLM 时,token 兼容性比信息量更重要**。这一观点与后续 CosyVoice 系列采用监督式 semantic tokenizer (匹配 LLM token 空间) 的设计决策一致。

## 可复用的 idea

1. **Matched tokens 原则**: 当使用 pretrained LLM 时,speech tokens 应尽量匹配 LLM 预训练的 token 分布/格式。这解释了为什么 CosyVoice 用 ASR-supervised tokenizer (输出接近文本 token) 而非纯声学 codec。可推广到 TTS 中 speech token 设计的指导原则。

2. **2x2 实验设计矩阵**: supervised/unsupervised x discrete/continuous 的分类框架可用于其他模态 (如 image, music) 的表征对比研究。

3. **JTFS LM 的连续反馈机制**: output embedding 不经离散化直接反馈到 input 端,保持连续空间的完整信息。这一设计可用于任何需要避免 autoregressive quantization bottleneck 的场景。

4. **SSL 中间层 > 最后层**: HuBERT layer 16 consistently 优于 layer 24,与 SSL 领域的 probing 研究一致,可作为 tokenizer 层选择的通用指导。
