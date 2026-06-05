---
type: paper
tier: deep
title: "w2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training"
arxiv_id: "2108.06209"
source: "Sources/w2v-BERT.pdf"
authors: [Yu-An Chung, Yu Zhang, Wei Han, Chung-Cheng Chiu, James Qin, Ruoming Pang, Yonghui Wu]
year: 2021
venue: "ASRU 2021"
tags: [self-supervised-learning, speech-representation, contrastive-learning, masked-prediction, conformer, ASR, voice-search]
concepts: ["[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[CodebookCollapse]]", "[[SpeechLanguageModel]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: ["[[模型库/w2v-BERT|w2v-BERT]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeechTokenizer]], [[SemanticvsAcousticTokens]], [[CodebookCollapse]], [[SpeechLanguageModel]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓(confirmed), [[SemanticvsAcousticTokens]]✓(confirmed), [[CodebookCollapse]]✓(confirmed), [[SpeechLanguageModel]]✓(confirmed) | 过滤: [[Gumbel-Softmax]](pending-review), [[MaskedGenerativeModeling]](pending-review) | 未命中但可能相关: 无

**已有知识要点**:
- [[SpeechTokenizer]]: w2v-BERT 在 SpeechLM 体系中被 AudioLM 采用作为 semantic tokenizer;其隐层特征经 k-means 后产生的 semantic tokens 是后续 TTS 系统 (如 MaskGCT) 的基础 ✓
- [[SemanticvsAcousticTokens]]: AudioLM 使用 w2v-BERT semantic tokens → SoundStream acoustic tokens 的层级生成框架 ✓
- [[CodebookCollapse]]: w2v-BERT 明确指出 contrastive loss 是防止 end-to-end 量化中 codebook collapse 的关键;移除 contrastive module 后 diversity loss 趋向 1 (最大 collapse) ✓
- [[SpeechLanguageModel]]: w2v-BERT 的表征被后续 AudioLM 等 SpeechLM 系统广泛采用 ✓

## 速查

> [!summary] 速查
> - **一句话**: 提出 w2v-BERT,首次将 wav2vec 2.0 的 contrastive learning 和 BERT 的 masked language modeling 端到端联合优化用于自监督语音预训练,在 LibriSpeech 和 voice search 上达到 SOTA [Abstract]
> - **路线**: Log-mel Spectrogram → Conv Subsampling (4x) → [Feature Encoder → Masking →] Contrastive Module (N Conformer blocks) → Quantization + Contrastive Loss; 同时 Contrastive Module context vectors → Masked Prediction Module (M Conformer blocks) → MLM Loss on discretized token IDs [Fig 1, §3]
> - **指标**: LibriSpeech 960h: WER 1.4/2.5 (XL, pre-training+self-training+LM) [Table 2]; 无 LM/self-training: WER 1.5/2.9 (XL) [Table 2]; Voice search: 6.2 WER (XL) vs conformer baseline 10.7 [Table 4]; 比 wav2vec 2.0 相对 WER 降 28-42% (XXL, 无 LM) [§5.1]
> - **可借鉴**: (1) Contrastive module 产生 discriminative token IDs → MLM module 消费 token IDs 做 masked prediction,两个 loss 端到端联合训练; (2) Conformer 替代 Transformer 作为 building block; (3) Contrastive loss 是防止 codebook collapse 的关键 (无它则 MLM trivially solved)
> - **局限**: 仅验证 ASR + voice search; 预训练 60k 小时; Conformer 增加计算量; 未探索非 ASR 下游任务

## 核心问题

**w2v-BERT 要解决什么问题?** [论文原文]

现有 SSL 语音预训练方法分两大路线 [§1, §2]:
1. **Contrastive learning** (wav2vec 2.0): 端到端,但 ASR 性能落后于加入 masked prediction 的方法 [§2]
2. **Masked prediction** (HuBERT, DiscreteBERT): 需要离线聚类 (k-means) 产生离散 targets,是两阶段过程 [§2]

**核心问题**: 能否端到端地结合 contrastive learning 和 masked prediction,避免 HuBERT 的多阶段迭代? [§1]

**为什么之前不能简单结合?** [论文原文]:
- vq-wav2vec / DiscreteBERT: 两阶段 — 先量化 (冻结),再做 masked prediction;量化错误会传播 [§2]
- HuBERT: 迭代缓解了量化错误,但仍是多阶段 [§2]
- 直接端到端训练的风险: **codebook collapse** — 没有 contrastive loss 的约束,quantizer 会退化为 trivial solution [§5.2]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

[论文原文] w2v-BERT 由三个模块组成 [§3.1, Fig 1]:

1. **Feature Encoder**: 2 层 2D convolution (stride 2,2),对 log-mel spectrogram 做 4x 下采样 [§3.1]
2. **Contrastive Module**: $N$ 层 Conformer blocks + linear projection + quantizer;解决 wav2vec 2.0 的 contrastive task,同时产生 discriminative token IDs [§3.1]
3. **Masked Prediction Module**: $M$ 层 Conformer blocks + softmax;消费 contrastive module 的 context vectors,预测 quantizer 产生的 token IDs [§3.1]

**关键区别**: [论文原文] 与 wav2vec 2.0 不同,w2v-BERT 不仅用 contrastive loss 训练 representations,还将 quantizer 产生的 token IDs 用作 masked prediction 的 target — 两个 loss 联合优化 [§3.1]

### 关键设计选择

#### 1. Contrastive Module [§3.1, §3.2]

[论文原文] 与 wav2vec 2.0 的 contrastive task 基本相同 [§3.2]:
- Feature encoder 输出 **after masking** → linear projection → Conformer blocks → context vectors [§3.1]
- Feature encoder 输出 **without masking** → quantizer → quantized vectors + token IDs [§3.1]
- Contrastive loss $\mathcal{L}_w$ + diversity loss $\mathcal{L}_d$: $\mathcal{L}_c = \mathcal{L}_w + \alpha \cdot \mathcal{L}_d$ ($\alpha=0.1$) [Eq 1, §3.2]
- 与 wav2vec 2.0 的 masking 相同: $p=0.065$, span length 10 [§4.2]
- **区别**: masked 位置替换为 random vectors (非共享可学习 vector) [§3.1]

#### 2. Masked Prediction Module [§3.1, §3.2]

[论文原文] Contrastive module 的 context vectors 直接传入 masked prediction module [§3.1]:
- 最后一层 Conformer 上加 softmax,预测 quantizer 分配的 token IDs [§3.1]
- Cross-entropy loss $\mathcal{L}_m$ 仅在 masked positions 计算 [§3.2]
- [论文原文] MLM 模块与 contrastive 模块共享 codebook,但 contrastive 模块产生的 context vectors (非 token IDs) 作为 MLM 输入 — "the module directly takes in the context vectors produced by the contrastive module" [§3.1]

总 training loss: $\mathcal{L}_p = \beta \cdot \mathcal{L}_c + \gamma \cdot \mathcal{L}_m$, 实验中 $\beta = \gamma = 1$ [Eq 2, §3.2]

#### 3. Conformer 替代 Transformer [§2]

[论文原文] w2v-BERT 使用 Conformer layers 而非 Transformer layers,结合 CNN (局部) 和 self-attention (全局) 对音频序列建模 [§2]

[agent 解读] 这是 w2v-BERT 相对 wav2vec 2.0 的另一个差异点,但作者明确指出"using a potentially more powerful building block is not the only factor" — w2v-BERT 在使用相同 Conformer 时仍优于 w2v-Conformer (仅用 contrastive loss 的 Conformer),证明 contrastive + MLM 的架构本身是性能提升的主因 [§2]

#### 4. Contrastive Loss 防止 Codebook Collapse [§5.2, Fig 2]

[论文原文] **核心分析**: 移除 contrastive module 后,w2v-BERT 退化 — MLM loss 迅速降至 ~0,prediction accuracy 达 100%,diversity loss 趋向 1 (最大 collapse) [§5.2, Fig 2]:

- [论文原文] 没有 contrastive module 时,quantizer 可以"cheat" — 让所有 masked positions 的 token 坍缩到同一个 code vector,MLM trivially solved 但没学到任何有用表征 [§5.2]
- [论文原文] contrastive loss 强制 codebook entries 具有区分性 (discriminative),从而为 MLM 提供有意义的 prediction targets [§5.2]

[agent 解读] 这一发现对 KB 中 [[CodebookCollapse]] 概念提供了重要补充:在端到端 contrastive + MLM 框架中,contrastive loss 本身就是最有效的 anti-collapse 机制,无需额外的 EMA / dead code replacement / factorized codes 等技巧

#### 5. Contrastive Module 的最优层数 [§5.2, Table 3]

[论文原文] 使用 $C_n$ 表示 contrastive module 有 $n$ 层 (总共 24 层固定) [§5.2]:
- $C_2 - C_8$: WER 持续下降 (contrastive module 越深,表征越好) [Table 3]
- $C_8 - C_{12}$: WER 停止下降 (MLM module 太小,无法充分利用 token IDs) [Table 3]
- **Sweet spot: $C_{12}$** (= w2v-BERT XL,各 12 层) [Table 3]
- $C_{24}$: 无 MLM module,退化为纯 contrastive,WER 最差 [Table 3]

[agent 解读] 这表明 contrastive 和 MLM 两个模块需要 balanced capacity — 任何一方过大/过小都会导致性能下降

### 训练策略

**模型配置** [Table 1]:

| 模型 | 参数量 | Contrastive 层 | Masked 层 | Model dim | Heads | Codebook |
|------|-------|---------------|----------|-----------|-------|----------|
| XL | 0.6B | 12 | 12 | 1024 | 8 | 1024×1024 |
| XXL | 1.0B | 12 | 30 | 1024 | 8 | 1024×1024 |

**预训练** [§4.2]:
- 数据: Libri-Light unlab-60k (60k hrs unannotated speech) [§4.1]
- XL: batch 2048, Adam, peak LR 2e-3, 25k warmup, Transformer LR schedule [§4.2]
- XXL: Adafactor, $\beta_1=0.9, \beta_2=0.98$, same LR schedule [§4.2]
- 80-dim log-mel filterbank features [§4.1]

**Fine-tuning** [§3.3, §4.3]:
- Sequence transducer: pre-trained w2v-BERT + 2-layer LSTM decoder (640 hidden) [§3.3]
- Swish activation + batch normalization between encoder and decoder [§3.3]
- Checkpoint at 400k steps → fine-tune, batch size 256 [§4.3]
- WordPiece tokenization (1024 tokens) [§4.1]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (test-clean/other, 960h, no LM, no self-train) | 1.5/2.9 (XL) | wav2vec 2.0: 2.1/4.5 | LibriSpeech | [Table 2] |
| WER (test-clean/other, 960h, no LM, no self-train) | 1.5/2.8 (XXL) | w2v-Conformer XXL: 1.6/3.3 | LibriSpeech | [Table 2] |
| WER (test-clean/other, 960h, +LM, +self-train) | 1.4/2.6 (XL) | wav2vec 2.0: 1.5/3.1 | LibriSpeech | [Table 2] |
| WER (test-clean/other, 960h, +LM, +self-train) | 1.4/2.5 (XXL) | w2v-Conformer XXL: 1.4/2.7 | LibriSpeech | [Table 2] |
| WER (100h, no LM, no self-train) | 2.3/4.0 (XXL) | wav2vec 2.0: 3.3/6.5 | LibriSpeech | [Table 3] |
| Voice Search WER | 6.2 (XL) | Conformer baseline: 10.7; w2v-Conformer-XL-tuned: 8.9 | Voice Search | [Table 4] |

**关键实验发现**:

1. **Contrastive + MLM > Contrastive alone** [Table 2]: [论文原文] w2v-BERT XL vs w2v-Conformer XL (相同 Conformer, 相同 model size,仅 pre-training method 不同): w2v-BERT 在所有评估子集上均更优,尤其 dev-other 和 test-other [§5.1] — 这是 "apple-to-apple comparison" 证明 MLM 的价值

2. **相对 wav2vec 2.0 大幅提升** [Table 2]: [论文原文] XXL 无 self-training/LM: WER 1.5/2.8 vs wav2vec 2.0 的 2.1/4.5 — 相对降低 28%/42%/32%/38% (四个评估子集) [§5.1]

3. **Voice search 实战验证** [Table 4]: [论文原文] 在 Google Voice Search 真实流量上,w2v-BERT XL 将 tuned conformer baseline WER 从 8.9 降至 6.2,相对降低 ~30% [§5.3] — 这是真实噪声、短时语音 (~5s)、含大量静音的挑战性场景

4. **Codebook collapse 分析** [Fig 2]: [论文原文] 无 contrastive module 时: MLM loss → 0, accuracy → 100%, diversity loss → 1 — 量化器完全 collapse,MLM trivially solved [§5.2]

5. **Contrastive module 最优深度** [Table 3]: $C_{12}$ (各 12 层) 为 sweet spot; $C_4$ 和 $C_{24}$ 均显著更差 [§5.2]

## 局限性

1. **仅验证 ASR**: [agent 解读] 与 wav2vec 2.0 类似,仅在 LibriSpeech ASR 和 Voice Search 上评估;未验证 speaker/separation/diarization 等任务 (WavLM 补充了这一点)
2. **预训练数据规模有限**: [agent 解读] 仅使用 60k hrs (Libri-Light);WavLM 使用 94k hrs 多样化数据效果更好
3. **Conformer 增加计算**: [agent 解读] Conformer blocks (含 depth-wise conv) 比 Transformer 更重;XXL (1.0B) 的推理成本不可忽视
4. **codebook 配置简单**: [论文原文] 单 codebook 1024 entries,code dim 1024 [Table 1];未探索 Product Quantization 或 multi-codebook 设计
5. **Voice search 数据内部**: [论文原文] 34.3k hrs pre-training + 1k hrs fine-tuning 均为内部数据,不可复现 [§5.3]

## 点评

**历史地位**: [agent 解读] w2v-BERT 是 SSL 语音预训练从 "contrastive only" 向 "contrastive + masked prediction" 融合的关键一步。它优雅地证明了两种自监督目标的互补性:contrastive loss 产生有区分性的离散 tokens (防止 collapse),MLM loss 在这些 tokens 上学习高层语义表征。这一洞察直接影响了后续 AudioLM 等系统选择 w2v-BERT 作为 semantic tokenizer。

**与知识库已有知识的关联**:
- w2v-BERT 2.0 (Chung et al., 2022,更大规模版本) 被 AudioLM 采用为 semantic tokenizer,其 semantic tokens 被 MaskGCT 进一步用 VQ-VAE 量化 [[SpeechTokenizer]]
- w2v-BERT 对 codebook collapse 的分析 (Fig 2) 直接证实了 [[CodebookCollapse]] 中 contrastive loss 作为 anti-collapse 机制的有效性
- w2v-BERT 是 [[SemanticvsAcousticTokens]] 中 "串联" 策略 (AudioLM) 的 semantic 端

**方法论贡献**:
1. **End-to-end contrastive + MLM**: 证明无需 HuBERT 的多轮迭代,可端到端联合训练
2. **Contrastive loss 防 collapse**: 首次实验证明 contrastive loss 是端到端离散化中防止 codebook collapse 的必要条件
3. **Balanced capacity 原则**: contrastive/MLM 模块需要均衡的容量分配 ($C_{12}$ optimal)

## 可复用的 idea

1. **Dual-objective pre-training**: 一个 loss 产生离散 targets (contrastive),另一个 loss 在这些 targets 上做生成式预测 (MLM) — 可推广到任何需要离散+生成式训练的场景
2. **Contrastive loss 作为 anti-collapse guard**: 在任何端到端量化系统中,contrastive loss 可作为 codebook health monitor — 当 diversity loss → 1 时表明 collapse 发生
3. **Layer capacity allocation**: 在多模块级联架构中,前后模块的容量需要均衡分配;可通过类似 $C_n$ 扫描找到最优点
4. **Real-world validation (voice search)**: SSL 模型在真实 noisy 短音频上的表现与 clean read speech 可能很不同,需要额外验证

---

检索命中: [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[CodebookCollapse]]✓, [[SpeechLanguageModel]]✓ | 过滤: [[Gumbel-Softmax]](pending-review), [[MaskedGenerativeModeling]](pending-review) | 未命中但可能相关: 无
