---
type: paper
tier: deep
title: "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"
arxiv_id: "2106.07447"
source: "Sources/HuBERT.pdf"
authors: [Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, Abdelrahman Mohamed]
year: 2021
venue: "IEEE/ACM Transactions on Audio, Speech, and Language Processing"
tags: [self-supervised-learning, speech-representation, masked-prediction, BERT, clustering, ASR]
concepts: ["[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[SpeechLanguageModel]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[SpeechTokenizer]], [[SemanticvsAcousticTokens]], [[SpeechLanguageModel]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓(confirmed), [[SemanticvsAcousticTokens]]✓(confirmed), [[SpeechLanguageModel]]✓(confirmed) | 过滤: 无 | 未命中但可能相关: 无

**已有知识要点**:
- [[SpeechTokenizer]]: HuBERT 是三类 speech tokenizer 中"自监督 tokenizer"的代表,通过 masked prediction 学习表征后做 k-means 聚类离散化;在 SpeechLM 中被 SynCLLM, SpeechGPT, dGSLM, SUTLM, pGSLM, GSLM, TWIST, PSLM 等系统采用 ✓
- [[SemanticvsAcousticTokens]]: HuBERT 产生的 semantic tokens 与文本对齐良好、语义连贯性强,但缺乏高频声学细节 (pitch, timbre 等); HuBERT 25Hz 在 SALMon benchmark 的语义任务上保持最强 (sBLIMP 60.89, sWUGGY 70.51) ✓
- [[SpeechLanguageModel]]: HuBERT semantic tokens 是多数 SpeechLM 系统的首选输入表征 ✓

## 速查

> [!summary] 速查
> - **一句话**: 提出 Hidden-Unit BERT (HuBERT),通过离线 k-means 聚类产生伪标签 + masked prediction 预训练的自监督语音表征学习方法,在 ASR 下游任务上匹配或超越 wav2vec 2.0
> - **路线**: Waveform → CNN Encoder (7层, 320x 下采样) → Masked Features → Transformer Encoder (12/24/48层) → 预测 k-means 聚类标签; 迭代: 第1轮聚类 MFCC, 第2轮聚类 HuBERT 中间层特征
> - **指标**: X-Large (1B) 在 LibriSpeech test-clean WER 1.9%, test-other 3.3% [Table III]; 10min labeled 仅 4.4%/6.1% WER [Table II]; 比 wav2vec 2.0 Large WER 降低 0.1-13% (相对) [Table II]
> - **可借鉴**: (1) 低质量聚类标签 + masked prediction 仍可学到好表征 (consistency > accuracy); (2) 迭代 refinement 持续改善表征质量; (3) 仅在 masked region 计算 loss 迫使模型学习 acoustic + language model
> - **局限**: 需要多轮迭代训练 (聚类 → 预训练 → 再聚类); k-means 聚类的 scalability; 仅验证了 ASR 下游任务

## 核心问题

**HuBERT 要解决什么问题?** [论文原文]

自监督语音表征学习面临三个独特挑战 [§I, Abstract]:
1. **多声音单元共存**: 一段语音中包含多个发音单元,破坏了 CV 中"一图一类"的 instance classification 假设
2. **无先验词典**: 预训练阶段不存在离散声音单元的词典,无法直接使用 NLP 的 masked prediction
3. **边界未知**: 声音单元的长度可变且边界不确定,不像 NLP 中 word/subword 有明确分割

**核心洞察**: [论文原文] "One crucial insight motivating this work is the importance of consistency of the targets, not just their correctness, which enables the model to focus on modeling the sequential structure of input data." [§II] — 聚类标签不需要精确,只需要"一致性" (同一发音单元总是映射到同一类)。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

HuBERT 的架构沿用 wav2vec 2.0 [§II-E, Fig 1]:
1. **CNN Encoder**: 7 层 512-channel CNN,strides [5,2,2,2,2,2,2],kernel widths [10,3,3,3,3,2,2],将 16kHz 波形 320x 下采样为 50Hz 特征 (20ms/帧) [§II-E, Table I]
2. **Transformer Encoder**: BASE (12层, 768d), LARGE (24层, 1024d), X-LARGE (48层, 1280d) [Table I]
3. **Projection Layer**: 将 Transformer 输出映射到聚类标签的 logit 空间 [§II-E]

**三种模型配置** [Table I]:

| 配置 | Transformer 层 | Embedding dim | FFN dim | Attention heads | 参数量 |
|------|---------------|---------------|---------|-----------------|-------|
| BASE | 12 | 768 | 3072 | 8 | 95M |
| LARGE | 24 | 1024 | 4096 | 16 | 317M |
| X-LARGE | 48 | 1280 | 5120 | 16 | 964M |

### 关键设计选择

#### 1. 离线聚类产生伪标签 [§II-A, §IV-B]

[论文原文] HuBERT 使用 acoustic unit discovery 系统 (如 k-means) 为每帧产生离散的伪标签 z_t,作为 BERT-like masked prediction 的目标 [§II-A, Fig 1]

- **第1轮**: 对 39 维 MFCC 特征做 k-means (K=100 clusters) [§IV-B]
- **第2轮**: 对 HuBERT 第6层 (BASE) 或第9层 (LARGE) 的 Transformer 输出做 k-means (K=500 clusters) [§IV-C]
- [agent 解读] k-means 聚类产生的标签是 noisy 的,但 HuBERT 对此表现出极强的鲁棒性 — 这正是 "consistency > correctness" 洞察的体现

#### 2. 仅在 Masked Region 计算 Loss [§II-B]

预测损失定义为 [§II-B, Eq 1]:
$$L = \alpha L_m + (1-\alpha) L_u$$

- $L_m$: 仅对 masked 位置计算的 cross-entropy loss
- $L_u$: 对 unmasked 位置计算的 cross-entropy loss
- [论文原文] $\alpha=1$ (仅 masked) 效果最佳: "the setup with $\alpha=1$ is more resilient to the quality of cluster targets" [§II-B]
- [agent 解读] 仅在 masked region 计算 loss 迫使模型:
  - 学习 acoustic representation (从 unmasked 上下文推断 masked 帧的底层声学特征)
  - 学习 language model (利用长距离时序结构预测 masked 内容)
  这两种能力的联合学习是 HuBERT 成功的根本原因

**消融验证** [Table V]: $\alpha=1.0$ vs $\alpha=0.5$ vs $\alpha=0.0$:
- MFCC teacher (K=500): 18.40% vs 33.42% vs 97.66% WER
- BASE-it1-layer6 teacher: 11.91% vs 13.47% vs 23.29% WER
- [论文原文] 当聚类质量差时,$\alpha=1$ (仅 masked loss) 的优势更明显 [Table V]

#### 3. 迭代 Refinement [§II-D]

[论文原文] HuBERT 支持迭代训练 — 用前一轮预训练模型的中间层特征做新一轮聚类,产生更好的伪标签 [§II-D]:

```
Round 1: MFCC → k-means(100) → labels → HuBERT-it1
Round 2: HuBERT-it1 layer6 → k-means(500) → labels → HuBERT-it2
Round 3: HuBERT-it2 layer9 → k-means(500) → labels → HuBERT-it3 (for LARGE/X-LARGE)
```

- PNMI 从 MFCC 的 ~0.25 提升到 HuBERT 特征的 ~0.68 [Table IV]
- [agent 解读] 迭代 refinement 类似于 semi-supervised learning 中的 iterative pseudo-labeling — 但 HuBERT 用的是无监督聚类而非有监督模型产生伪标签

#### 4. Cluster Ensembles [§II-C]

[论文原文] 可以同时使用多个聚类模型 (不同 K 值或不同特征) 产生多组标签,将 loss 扩展为多任务形式 [§II-C, Eq 2]:

$$L_m = \sum_{t \in M} \sum_k \log p_f^{(k)}(z_t^{(k)} | \tilde{X}, t)$$

- 每个聚类模型有独立的 projection matrix $A^{(k)}$ [§II-E, Eq 3]
- [论文原文] 类似于 product quantization (PQ),不同粒度的聚类提供互补信息 [§II-C]

**消融验证** [Table VI]: k-means {50,100,500} ensemble WER 17.56% < single k-means {50,100} 的 17.81% [Table VI]

#### 5. Masking 策略 [§IV-C]

- Mask span $l=10$ frames [§IV-C]
- Mask probability $p=8\%$ of waveform encoder output frames [§IV-C]
- [论文原文] 采用与 SpanBERT 和 wav2vec 2.0 相同的 masking 策略 [§II-B]

### 训练策略

**预训练** [§IV-C]:
- BASE: 2 iterations on LibriSpeech 960h, 32 GPUs, batch 87.5s/GPU
  - Iter 1: 250k steps (MFCC labels), Iter 2: 400k steps (HuBERT-it1 layer6 labels)
- LARGE/X-LARGE: 1 iteration on Libri-Light 60k hours, 128/256 GPUs, 400k steps
  - 使用 BASE-it2 layer9 特征的 k-means labels (第3轮迭代)
- Optimizer: Adam ($\beta=0.9, 0.98$), LR warmup (8% steps) then linear decay [§IV-C]
- Peak LR: 5e-4/1.5e-3/3e-3 for BASE/LARGE/X-LARGE [§IV-C]

**Fine-tuning** [§IV-D]:
- CTC loss,冻结 CNN encoder,替换 projection layer 为 29 类 softmax [§II-E]
- Freeze-step: 冻结 Transformer 参数前 N 步,仅训练 softmax [§IV-D]
- 解码: wav2letter++ beam search + n-gram/Transformer LM [§IV-D, Eq 4]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (test-clean/test-other) | 4.6/6.8 (X-Large, 10min) | wav2vec 2.0 Large: 4.8/8.2 | LS (LL-60k pretrain) | [Table II] |
| WER (test-clean/test-other) | 2.9/5.4 (Large, 1h) | wav2vec 2.0 Large: 2.9/5.8 | LS (LL-60k pretrain) | [Table II] |
| WER (test-clean/test-other) | 2.8/4.8 (X-Large, 10h) | wav2vec 2.0 Large: 2.9/5.8 | LS (LL-60k pretrain) | [Table II] |
| WER (test-clean/test-other) | 1.9/3.5 (X-Large, 100h) | wav2vec 2.0 Large: 2.0/4.0 | LS (LL-60k pretrain) | [Table II] |
| WER (test-clean/test-other) | 1.9/3.3 (X-Large, 960h) | wav2vec 2.0 Large+self-train: 1.4/2.6 | LS-960 (LL-60k pretrain) | [Table III] |
| PNMI (MFCC, K=100) | 0.253 | - | LS-960 | [Table IV] |
| PNMI (BASE-it1 layer6, K=500) | 0.684 | MFCC K=500: 0.287 | LS-960 | [Table IV] |

**关键实验发现**:

1. **极低资源场景优势** [Table II]: 仅用 10 分钟标注数据,HuBERT X-Large WER 4.6%/6.8%,比 wav2vec 2.0 Large 4.8%/8.2% 更优;HuBERT Large 10min 达到 6.1%/10.1%,超越 DiscreteBERT (15.7/24.1) [论文原文]

2. **一致性胜过正确性** [Table V]: 即使使用质量很差的 MFCC k-means 标签 (PNMI 仅 0.25),仍然可以学到有用表征;关键是只在 masked region 计算 loss ($\alpha=1$) [论文原文]

3. **迭代改善** [Table IV, Fig 2]: 第二轮迭代的 PNMI 从 MFCC 的 0.25 跃升至 HuBERT 特征的 0.68;中间层 (layer 6) 在第一轮最好,而第二轮随层深度持续改善 [论文原文]

4. **波形 vs 量化输入**: [论文原文] HuBERT 用原始波形而非量化 units 作为输入,避免了信息损失,这是超越 DiscreteBERT 的两个关键原因之一;另一个是迭代 refinement [§III]

5. **scaling 效果**: BASE (95M) → LARGE (317M) → X-LARGE (964M) 在所有设定下持续改善 [Table II, III] [论文原文]

## 局限性

1. **多轮迭代成本**: 需要先预训练一轮 HuBERT,再做聚类,再预训练第二轮,训练流程复杂 [agent 解读]
2. **仅验证 ASR**: 论文仅验证了 ASR fine-tuning,未探索 TTS、voice conversion 等生成任务 [§VI]
3. **聚类 scalability**: k-means 需要将整个 960h 的特征加载到内存;对于 LARGE 特征 (1024d),使用了 MiniBatchKMeans + 10% 采样 [§IV-B]
4. **英语单语**: 仅在 LibriSpeech/Libri-Light (英语) 上验证 [agent 解读]
5. **未与 self-training 结合**: [论文原文] HuBERT 落后于 pre-training + self-training 的组合方法,但作者认为 HuBERT 也可以与 self-training 互补 [§V-A]

## 点评

**历史地位**: [agent 解读] HuBERT 是语音自监督学习的里程碑之一。其产生的 semantic tokens 成为后续 SpeechLM (GSLM, AudioLM, pGSLM, TWIST 等) 和 TTS 系统的核心组件。在 PaperWiki 知识库中,HuBERT 被引用为 "自监督 tokenizer" 的代表,影响了整个 speech token 设计范式。

**方法论贡献**:
1. **"Consistency > Correctness" 洞察**: 这一发现改变了人们对 pseudo-label 质量要求的认知 — 只要标签有一致性,即使错误也能学到好表征
2. **离线聚类 + 在线 masked prediction 解耦**: 避免了 wav2vec 2.0 中 Gumbel-Softmax 在线量化的复杂性和不稳定性
3. **迭代 refinement**: 提供了一种 bootstrapping 式的持续改善机制

**与后续工作的关系**: [agent 解读]
- HuBERT 的 k-means semantic tokens 直接催生了 GSLM (2021), AudioLM (2022) 等语音语言模型
- CosyVoice 的 S3 tokenizer 通过 ASR 监督训练超越了 HuBERT 的无监督 tokens (WER 一致性更好)
- MaskGCT 使用 VQ-VAE 量化 W2v-BERT 2.0 (HuBERT 的后继者) 的隐层特征,保留更多韵律信息
- Whisper encoder 在近期 SpeechLM 中逐渐取代 HuBERT 作为首选 speech encoder

## 可复用的 idea

1. **低质量伪标签 + masked prediction**: 可迁移到任何缺乏标签的序列建模任务 — 先用简单方法产生 noisy labels,再用 masked prediction 学习表征
2. **迭代 refinement 框架**: 用预训练模型的特征做更好的聚类 → 更好的标签 → 更好的模型,适用于任何 self-training 场景
3. **仅在 masked region 计算 loss**: 比同时在 unmasked region 计算更鲁棒,尤其在标签质量差时
4. **k-means 聚类作为离散化**: 简单、可扩展的语音离散化方法,无需端到端训练量化器
5. **PNMI 指标**: 衡量聚类质量与真实 phoneme label 对齐程度的信息论指标,可用于评估任何聚类/量化方案

---

检索命中: [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechLanguageModel]]✓ | 过滤: 无 | 未命中但可能相关: 无
