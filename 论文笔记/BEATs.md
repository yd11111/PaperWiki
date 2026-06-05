---
type: paper
tier: deep
title: "BEATs: Audio Pre-Training with Acoustic Tokenizers"
arxiv_id: "2212.09058"
source: "Sources/BEATs.pdf"
authors: [Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, Furu Wei]
year: 2022
venue: "ICML 2023 (Microsoft)"
tags: [self-supervised-learning, audio-representation, acoustic-tokenizer, iterative-pretraining, ViT, audio-classification, discrete-label-prediction, knowledge-distillation]
concepts: ["[[Self-SupervisedSpeechRepresentation]]", "[[SpeechTokenizer]]", "[[CodebookCollapse]]"]
models: []
tasks: []
datasets: ["[[AudioSet]]"]
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[SpeechTokenizer]], [[CodebookCollapse]], [[SemanticvsAcousticTokens]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓(confirmed), [[CodebookCollapse]]✓(confirmed), [[SemanticvsAcousticTokens]]✓(confirmed) | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review) | 未命中但可能相关: 无

**已有知识要点**:
- [[SpeechTokenizer]]: HuBERT 的离线 k-means 聚类产生 semantic tokens;BEATs 提出了完全不同的 acoustic tokenizer 路线,通过知识蒸馏迭代优化离散标签质量 ✓
- [[CodebookCollapse]]: 端到端量化训练中 codebook entries 利用率低的问题;BEATs 通过 $\ell_2$ normalization 和 EMA 优化 codebook ✓
- [[SemanticvsAcousticTokens]]: SSL 产生的 semantic tokens 与 BEATs 的 acoustic tokenizer 产生的离散标签在性质上有本质区别 — 前者编码语言信息,后者编码高层音频语义 ✓

## 速查

> [!summary] 速查
> - **一句话**: 提出 BEATs (Bidirectional Encoder representation from Audio Transformers),一个迭代式音频预训练框架,通过交替优化 acoustic tokenizer 和 audio SSL 模型,用离散标签预测替代传统重建 loss,在 AudioSet-2M 和 ESC-50 上刷新 SOTA
> - **路线**: Audio → Mel-filterbank (128d, 25ms, 10ms hop) → 16x16 patches → ViT Encoder (12层, 768d, 90M params) → 预测 Acoustic Tokenizer 产生的离散标签; Tokenizer: Random-Projection (iter1) → Self-Distilled Tokenizer (iter2+,含 12层 Transformer encoder + 3层 estimator)
> - **指标**: AS-2M mAP 48.6 (single model SOTA, vs prev 47.4) [Table 1]; AS-20K mAP 38.9 [Table 1]; ESC-50 Acc 98.1 [Table 1]; Ensemble (10) AS-2M mAP **50.6** [Table 3]; KS1 98.1, KS2 98.3 [Table 1]
> - **可借鉴**: (1) 迭代训练 tokenizer + SSL 模型的交替优化框架; (2) Random-projection tokenizer 作为冷启动; (3) 离散标签预测优于重建 loss 的实证依据; (4) 知识蒸馏式 self-distilled tokenizer 设计
> - **局限**: 仅在 AudioSet (音频分类) 上验证,未在 ASR/TTS 等语音任务上评估; 90M 参数较小,未探索 scaling; 需要多轮迭代训练,流程较复杂

## 核心问题

**BEATs 要解决什么问题?** [论文原文]

当时音频 SSL 面临一个根本矛盾 [§1]:
1. **离散标签预测优于重建**: 在 NLP (BERT) 和 CV (BEiT) 中,离散标签预测已被证明是更好的预训练目标,因为它鼓励模型学习高层语义而非低层细节 [§1]
2. **但音频缺乏好的 tokenizer**: 语音领域有 HuBERT 的 k-means tokenizer,但通用音频 (包含环境声、音乐等) 缺乏 phoneme 这样的先验结构,无法直接应用 HuBERT 的方法 [§1]
3. **重建 loss 的局限**: [论文原文] SOTA 音频 SSL (Audio-MAE, MaskSpec) 仍使用重建 loss,它 "mainly contain low-level time-frequency features and lack high-level audio semantic abstraction" [§1]

**核心洞察**: [论文原文] 解决 "没有好 tokenizer" 的问题不是一步到位,而是通过**迭代**交替优化: 用 SSL 模型的知识改进 tokenizer,再用更好的 tokenizer 产生更好的标签训练 SSL 模型 [§1, Fig 1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

BEATs 由两个交替优化的组件构成 [§3.1, Fig 1]:
1. **Acoustic Tokenizer**: 将音频 patch 序列量化为离散标签 $\hat{Z}$,作为 SSL 模型的训练目标
2. **Audio SSL Model**: ViT backbone,通过 Masked Audio Modeling (MAM) 预测离散标签

### 关键设计选择

#### 1. Random-Projection Tokenizer (冷启动) [§3.2.1]

[论文原文] 第一轮迭代无可用 teacher model,采用 random-projection tokenizer 冷启动 [§3.2.1, Fig 2a]:
- 随机初始化的线性投影层 $\mathbf{W}$ + 冻结的 codebook embeddings $\{\mathbf{v}_i\}_{i=1}^K$
- 离散标签 = 投影后特征的最近邻 codebook entry: $\hat{z}_t = \arg\min_i ||\mathbf{v}_i - \mathbf{W}\mathbf{x}_t||_2^2$ [Eq 1]
- [agent 解读] Random-projection tokenizer 的聪明之处在于: 它虽然产生的标签 **语义上无意义**,但具备 **一致性** (同一音频 patch 总映射到同一标签) — 这与 HuBERT "consistency > correctness" 的洞察一脉相承

#### 2. Self-Distilled Tokenizer (迭代优化) [§3.2.2]

[论文原文] 从第二轮开始,用上一轮 SSL 模型作为 teacher,训练 self-distilled tokenizer [§3.2.2, Fig 2b]:

- **Tokenizer Encoder**: 12 层 Transformer,将 input patches → encoded vectors $\mathbf{E}$ [§3.2.2]
- **量化**: $\ell_2$ normalized nearest-neighbor lookup from learnable codebook (K entries): $\hat{z}_t = \arg\min_i ||\ell_2(\mathbf{v}_i) - \ell_2(\mathbf{e}_t)||_2^2$ [Eq 2]
- **Tokenizer Estimator**: 3 层 Transformer,预测 teacher model 的输出 $\hat{\mathbf{O}}$ [§3.2.2]
- **训练目标**: cosine similarity (estimator 输出 vs teacher 输出) + MSE (encoded vectors vs quantized vectors) [Eq 3]
- **Straight-through gradients**: 解决 VQ 不可微问题 [§3.2.2]
- **EMA for codebook**: 指数移动平均更新 codebook embeddings,稳定训练 [§3.2.2]

[agent 解读] Self-distilled tokenizer 的设计巧妙在于: 它不仅从 teacher 学到了**什么是重要的语义**,还通过 codebook + nearest-neighbor lookup 的瓶颈结构**强制离散化**,使产生的标签既富含语义又适合做 masked prediction 目标

#### 3. Audio SSL Model: Masked Audio Modeling (MAM) [§3.3]

**Backbone** [§3.3.1]:
- Vanilla ViT: linear projection + 12 Transformer layers (768d, 8 heads) [§3.3.1]
- 90M 参数 [§4.2]
- Convolution-based relative PE + gated relative position bias [§3.3.1]
- DeepNorm for training stabilization [§3.3.1]

**Pre-training** [§3.3.2, Fig 3a]:
- 随机 mask 75% 的 input patches [§3.3.2]
- 仅将 **unmasked patches** 送入 ViT encoder (速度提升) [§3.3.2]
- Label predictor 接收所有 patch 表征 (unmasked + masked zeros) 预测离散标签 [§3.3.2]
- Loss: cross-entropy on masked positions $\mathcal{L}_{MAM} = -\sum_{t \in \mathcal{M}} \log p(\hat{z}_t | \mathbf{X}^U)$ [Eq 4]

**Fine-tuning** [§3.3.3, Fig 3b]:
- 去掉 label predictor,加 task-specific linear classifier [§3.3.3]
- 整个 patch sequence 送入 encoder (不 masking) [§3.3.3]
- SpecAugment 数据增强 [§3.3.3]
- Mean-pooling + softmax for classification [§3.3.3]

#### 4. 迭代训练流程 [§4.2]

[论文原文] BEATs 进行三轮迭代 [§4.2]:
```
Iter 1: Random-Projection Tokenizer → BEATs_iter1 (SSL only)
Iter 2: Self-Distilled Tokenizer (teacher=BEATs_iter1) → BEATs_iter2
Iter 3: Self-Distilled Tokenizer (teacher=BEATs_iter2) → BEATs_iter3
Iter 3+: Self-Distilled Tokenizer (teacher=BEATs_iter2 fine-tuned on AS) → BEATs_iter3+
```

- BEATs_iter3 性能接近 BEATs_iter2,说明 **2-3 轮即收敛** [§4.3]
- BEATs_iter3+ 使用 supervised fine-tuned 模型作为 teacher → tokenizer 学到更多任务相关语义 → 显著提升 [§4.3]

### 训练策略

[论文原文] 训练细节 [§4.2]:
- **SSL 预训练**: 400K steps, batch size 5.6K seconds, lr 5e-4 [§4.2]
- **Tokenizer (SSL teacher)**: 400K steps, batch 1.4K seconds, lr 5e-5 [§4.2]
- **Tokenizer (supervised teacher)**: 400K steps, batch 1.4K seconds, lr 5e-4 [§4.2]
- **Codebook**: 1024 entries, 256 dimensions [§4.2]
- **Acoustic feature**: 128-dim Mel-filterbank, 25ms window, 10ms hop, normalized to mean=0 std=0.5, split into 16x16 patches [§4.2]
- **数据**: AudioSet full training set (AS-2M, ~5K hours) [§4.1]

## 实验

| 指标 | 本文 (BEATs) | Baseline (prev SOTA) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| AS-2M mAP (single, SSL) | 48.6 (iter3+) | Audio-MAE: 47.3 | AudioSet-2M | [Table 1] |
| AS-20K mAP | **38.9** (iter3+) | Audio-MAE: 37.6 | AudioSet-20K | [Table 1] |
| ESC-50 Acc | **98.1** (iter3+) | Audio-MAE: 94.1 | ESC-50 | [Table 1] |
| KS1 Acc | **98.1** (iter3+) | data2vec: 96.9 | Speech Commands V1 | [Table 1] |
| KS2 Acc | **98.3** (iter3+) | Audio-MAE: 97.9 | Speech Commands V2 | [Table 1] |
| ER Acc | **65.0** (iter3+) | data2vec: 63.4* | IEMOCAP | [Table 1] |
| AS-2M mAP (ensemble 5) | 50.4 | PaSST ensemble: 49.6 | AudioSet-2M | [Table 3] |
| AS-2M mAP (ensemble 10) | **50.6** | PaSST ensemble: 49.6 | AudioSet-2M | [Table 3] |

**关键实验发现**:

1. **离散标签 > 重建** [Table 1, §4.3]: [论文原文] BEATs_iter1 (random-projection labels) 在 5/6 任务上已超越重建式 SOTA (Audio-MAE),证明即使用随机标签的离散预测也优于重建 loss

2. **迭代改善** [Table 1, §4.3]: [论文原文] BEATs_iter2 (self-distilled labels from SSL teacher) 相比 iter1 进一步提升音频分类性能; iter3 ≈ iter2,说明快速收敛 (~2 轮)

3. **Supervised teacher 大幅提升** [Table 1, §4.3]: [论文原文] BEATs_iter3+ (supervised fine-tuned teacher) 比 iter3 (SSL teacher) 在 AS-2M 上提升 0.6 mAP,证明 supervised knowledge 可通过 tokenizer 传递给 SSL 预训练

4. **Tokenizer 语义可视化** [Fig 4, §4.5]: [论文原文] T-SNE 可视化显示:
   - 重建目标对扰动敏感 (同一音频加不同噪声后特征距离远)
   - BEATs_iter3 标签对扰动鲁棒 (同类音频聚在一起)
   - BEATs_iter3+ 标签语义更清晰 (不同类别分离更好)

5. **参数效率** [Table 1]: [论文原文] BEATs 仅 90M 参数,但超越了 304M 的 Audio-MAE Large 和 93M 的 data2vec,参数效率极高

## 局限性

1. **仅验证分类任务**: [agent 解读] BEATs 仅在 AudioSet/ESC-50 等分类 benchmark 上验证,未在 ASR、TTS、speaker verification 等任务上评估;作为 "audio" SSL 模型,其在语音理解任务上的泛化性未知
2. **模型规模有限**: [论文原文] 仅用 90M ViT-Base,未探索更大规模 (ViT-Large/Huge) [§5]
3. **数据规模有限**: [agent 解读] 仅在 AudioSet (~5K h) 上预训练,远小于语音 SSL 模型 (HuBERT 60K h, WavLM 94K h)
4. **迭代训练成本**: [agent 解读] 需要多轮迭代 (SSL + tokenizer 交替训练),训练流程比单阶段方法复杂
5. **ViT 非流式**: [agent 解读] ViT 需要整段音频作为输入,不支持流式推理

## 点评

**历史地位**: [agent 解读] BEATs 是通用音频 SSL 领域的重要里程碑,首次证明离散标签预测在音频域也优于重建 loss。其提出的 iterative tokenizer-SSL 交替训练框架成为后续工作的重要参考。AudioSet-2M 上 50.6 mAP (ensemble) 的 SOTA 结果维持了相当长时间。

**方法论贡献**:
1. **迭代式 acoustic tokenizer**: 通过交替优化解决了"没有好 tokenizer → 无法做离散标签预测"的 chicken-and-egg 问题
2. **Random-projection 冷启动**: 证明了即使完全随机的离散标签也比重建目标更有效 — 进一步验证了 HuBERT "consistency > correctness" 的洞察
3. **Self-distilled tokenizer**: 知识蒸馏 + VQ 的组合,使离散标签逐轮富含更多语义信息
4. **统一预训练**: [论文原文] 离散标签预测可统一 speech、vision、language 的预训练范式 [§1]

**与已有知识的关系**: [agent 解读]
- BEATs 的 random-projection tokenizer 与 Chiu et al. (2022) 的工作一致,后者证明随机量化也足以支撑大规模 SSL 预训练
- 迭代优化框架与 HuBERT 的 iterative refinement 理念相似,但 BEATs 的 tokenizer 更复杂 (self-distilled 12层 Transformer vs k-means)
- BEATs 验证了 [[SemanticvsAcousticTokens]] 中 "discrete label prediction > reconstruction" 的论断在通用音频域也成立

## 可复用的 idea

1. **迭代 tokenizer-model 交替优化**: 适用于任何缺乏离散标签的模态 (视频、传感器信号等)
2. **Random-projection 冷启动**: 极简的 tokenizer 初始化方案,避免了 k-means 等离线聚类的计算开销
3. **Self-distilled tokenizer**: 12层 Transformer encoder + VQ + 3层 estimator 的架构设计,可迁移到其他 tokenizer 训练场景
4. **仅编码 unmasked patches**: 在预训练阶段仅将 25% 的 unmasked patches 送入 encoder,节省 ~4x 计算
5. **$\ell_2$ normalization for codebook**: 改善 codebook utilization,防止 [[CodebookCollapse]]

---

检索命中: [[SpeechTokenizer]]✓, [[CodebookCollapse]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review) | 未命中但可能相关: 无
