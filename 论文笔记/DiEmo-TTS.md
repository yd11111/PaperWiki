---
type: paper
tier: deep
title: "DiEmo-TTS: Disentangled Emotion Representations via Self-Supervised Distillation for Cross-Speaker Emotion Transfer in Text-to-Speech"
arxiv_id: "2505.19687"
source: "Sources/DiEmo-TTS.pdf"
authors: [Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Seong-Whan Lee]
year: 2025
venue: "Interspeech 2025"
tags: [TTS, emotion, cross-speaker, disentanglement, self-supervised, DINO, style-transfer, FastSpeech2]
concepts: ["[[EmotionControlinTTS]]", "[[SpeechFactorization]]", "[[StyleTransferinTTS]]", "[[SpeakerEmbedding]]", "[[GradientReversalLayer]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: ["[[论文笔记/EmoSphere-TTS|EmoSphere-TTS]]", "[[论文笔记/EmoSphere++|EmoSphere++]]"]
tasks: []
datasets: ["ESD", "MSP-Podcast"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: DiEmo-TTS 属于跨说话人情感迁移 (cross-speaker emotion transfer) 方向,是 [[EmotionControlinTTS]] 的子问题。其核心挑战是 [[SpeechFactorization]] 中的 emotion-speaker disentanglement: 从参考语音中提取不含说话人身份信息的情感嵌入。

**已有认知**:
- **解耦方法谱系**: 对抗训练 ([[GradientReversalLayer]], 2016) → 信息瓶颈 (VQ) → self-distillation (Seed-TTS, 2024)。DiEmo-TTS 提出第四条路线: 基于 DINO 的自监督蒸馏,不依赖显式标签实现解耦。
- **Speaker Embedding** (confirmed): ECAPA-TDNN 等 speaker encoder 用于提取说话人特征,DiEmo-TTS 用 ECAPA-TDNN 做情感聚类中的说话人嵌入。
- **Speech Factorization** (confirmed): 已有方法包括 GRL 对抗训练、VQ 信息瓶颈、正交损失等。DiEmo-TTS 的 cluster-driven sampling + information perturbation 是新的解耦组合方案。
- **相关工作**: EmoSphere-TTS/EmoSphere++ (Cho et al., 2024/2025) 为本文同组前作,用球面坐标建模情感; Daisy-TTS 用 PCA 分解韵律嵌入; EmoCtrl-TTS 用帧级 arousal-valence 控制。

**创新判断**: DiEmo-TTS 的核心创新在于将 DINO 自监督蒸馏从 speaker verification 领域迁移到 emotion disentanglement,并引入 cluster-driven sampling (基于情感聚类而非随机裁剪) 和 formant-based information perturbation (通过共振峰扰动破坏说话人信息同时保留情感),这两点在已有 KB 中无先例。

> 检索命中: [[SpeakerEmbedding]]✓, [[SpeechFactorization]]✓ | 参考: [[EmotionControlinTTS]][待确认], [[StyleTransferinTTS]][待确认], [[GradientReversalLayer]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 DINO 自监督蒸馏 + cluster-driven sampling + formant perturbation 实现跨说话人情感解耦迁移,无需显式情感标签
> - **路线**: 参考 mel → (cluster-driven cropping + formant perturbation) → teacher/student emotion encoder (DINO loss + cosine similarity loss) → emotion embedding → FastSpeech 2 + dual conditioning transformer → mel → BigVGAN → waveform
> - **指标**: nMOS 4.23, sMOS 3.96, eMOS 4.07 (平均); SECS 0.8505, EECS 0.4527; WER 16.16, CER 5.60 (ESD 数据集) [Table 2]
> - **可借鉴**: (1) cluster-driven sampling: 基于情感聚类构造 DINO 训练对而非随机裁剪,可迁移到任何需要属性解耦的蒸馏场景; (2) formant perturbation 作为 speaker identity 扰动手段,比 noise augmentation 更精准地保留情感同时破坏音色; (3) convex hull + 球面坐标做无监督情感聚类对齐
> - **局限**: 依赖 emotional attribute predictor (预训练在标注数据上) 生成伪标签,非完全无监督; 仅在 ESD (350 句 x 10 人) 小规模数据集上验证; 未开源代码

## 核心问题

跨说话人情感迁移需要从参考语音中提取"纯净"的情感嵌入,不携带参考说话人的音色信息。现有方法的问题:

1. **GRL 对抗训练** [§1]: 存在情感保留与说话人分离的 trade-off,超参数优化困难,合成质量受限 [论文原文]
2. **VQ 信息瓶颈** [§1]: 有效但导致信息无差别丢失 (unintended information loss),需复杂优化平衡压缩与重建 [论文原文]
3. **正交损失 (Trans-Ort)** [§1]: 依赖多个复杂损失函数,且仍以 GRL 为基础,固有限制未解决 [论文原文]
4. **条件注入方式** [§1]: 当前简单的拼接/加法条件化方式难以建模复杂的风格变化,导致风格不一致 [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DiEmo-TTS 在 FastSpeech 2 [28] 基础上增加两个核心模块: (1) 基于 DINO 自监督蒸馏的 emotion encoder (teacher-student 架构); (2) 双条件 transformer 块 (DCT) 用于融合 emotion 和 speaker 特征 [§2, Fig 1(c)]。

推理流程: 文本 → text encoder → variance adaptor → DCT 融合 (emotion embedding + speaker embedding) → mel decoder → BigVGAN vocoder [§2.3]。

### 关键设计选择

#### 1. 情感聚类与匹配 (Emotion Clustering & Matching) [§2.1.1]

**为什么需要**: DINO 原本用于 speaker verification,其随机裁剪假设同一 utterance 属于同一说话人。但在情感场景中,需要让同一情感的不同 utterance 构成正样本对。因此需要先对数据做情感聚类,再基于聚类做采样 [论文原文]。

**怎么做**:
1. 用 emotional attribute predictor [19] 预测每段语音的 valence/arousal/dominance 值
2. 对每个说话人,用 k-means 将 speaker embeddings (ECAPA-TDNN) 聚成 M 个 cluster
3. 用 convex hull 算法确定中性情感 cluster p_n: 去掉某个 cluster 后剩余 cluster 的 convex hull 面积最大的那个即为中性 [§2.1.1, Eq.1]
4. 以 p_n 为原点做球面坐标变换,提取每个 cluster 的方位角和仰角
5. 用 multiple Hungarian method [24] 对齐不同说话人的 cluster,使同一情感对应相同 cluster ID [Fig 1(b)]

**为什么用 convex hull 找中性**: 心理学中,中性情感位于 VAD 空间的中心; 去掉中心点后,剩余点的 convex hull 面积最大,因为中心点对凸包面积的贡献最小 [论文原文, 参考 Russell & Mehrabian 1977]。

**为什么用球面坐标 + Hungarian**: 不同说话人的情感 cluster 在绝对位置上不同,但相对于中性点的角度关系稳定; 球面坐标变换消除绝对位置差异,Hungarian 匹配找到跨说话人的最优情感对齐 [论文原文]。[agent 解读: 这一设计借鉴了 EmoSphere-TTS 中球面坐标表示情感的思路,但这里用于聚类匹配而非情感控制]。

#### 2. 情感解耦 DINO (Emotion Disentanglement DINO) [§2.1.2]

传统 DINO [17] 用于 speaker verification: 同一 utterance 的不同 crop 共享说话人身份。DiEmo-TTS 将其改造为情感解耦:

**Cluster-driven sampling (CDS)** [§2.1.2]: 从同一 emotion cluster 中随机选取 5 个 utterance,对每个 utterance 随机裁剪出 2 个 long crop (3s) 和 4 个 short crop (2s)。这样正样本对来自同一情感但不同内容和说话人,迫使 encoder 忽略内容和说话人信息,只保留情感 [论文原文]。

**为什么不用随机裁剪**: 随机裁剪同一 utterance 会让 encoder 学到说话人信息 (因为同一 utterance 的说话人相同),而 cluster-driven sampling 确保正样本对跨越不同说话人和内容 [agent 解读]。

**Information perturbation (IP)** [§2.1.2]: 对 crop 做 formant perturbation 而非传统 noise augmentation (MUSAN/RIR)。利用说话人音色与共振峰的相关性: 扰动共振峰可有效破坏说话人身份同时保留情感表达 [论文原文, 参考 Zhu et al. 2024]。

**为什么 formant perturbation 优于 noise augmentation**: 背景噪声或混响添加的是与说话人无关的失真,不能有效破坏说话人身份; 而共振峰直接编码声道特征 (即音色),扰动共振峰精准地破坏身份信息而保留韵律/情感 [论文原文]。

**Cosine similarity loss (CS)**: 在传统 DINO 的 cross-entropy loss 基础上,增加 embedding 级的 cosine similarity loss [§2.1.2, Eq.2],使情感嵌入在向量空间中更紧凑,更适合聚类 [论文原文]。

最终 DINO loss: L_DINO = (1/L(N-1)) * sum_l sum_m [CE(h_t^l, h_s^m) + CS(e_t^l, e_s^m)] [Eq.2]

#### 3. 双条件 Transformer (DCT) [§2.2]

**为什么需要**: 简单的拼接/加法条件化无法充分建模 emotion 和 speaker 特征的复杂交互,导致风格不一致 [论文原文]。

**怎么做**: 受 VITS2 [27] 启发,在 FFT block 中引入 weight-sharing multi-head attention: 分别对 emotion embedding 和 speaker embedding 做 attention conditioning,再通过 MLP 融合两路 attention 输出 [§2.2, Fig 1(c)]。DCT 应用于 encoder 和 decoder 的第 3 个 FFT block。

**为什么 weight sharing**: [agent 解读] 共享 attention 权重使两种 style 信息在相同的注意力空间中交互,避免独立 attention 可能产生的信息冲突,同时减少参数量。

### 训练策略

- 基础模型: FastSpeech 2 (4 层 FFT, hidden 256, filter 1024, kernel 9) [§3.2]
- 训练数据: ESD (10 说话人 x 5 情感 x ~350 句) + MSP-Podcast (情感属性标注) [§3.1]
- 优化器: AdamW, beta1=0.9, beta2=0.98, lr=5e-4 [§3.1]
- 训练时长: ~24h on single NVIDIA RTX A6000 [§3.1]
- 情感嵌入: 推理时取 student encoder 对参考 mel 的 short crop 输出 [§2.3]
- 声码器: BigVGAN (预训练,冻结) [§3.1]
- Teacher-Student encoder 架构: Mel-style encoder [32] (MetaStyleSpeech) [§3.2]
- Teacher 更新: EMA (exponential moving average) [Fig 1(c)]

## 实验

### 主实验: 跨说话人情感迁移 [Table 1]

| 指标 | DiEmo-TTS | Trans-GRL [2] | Trans-VQ [3] | Trans-Ort [4] | Expressive FS2 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| nMOS (Angry) | 3.77 | 3.77 | 3.74 | 3.67 | 3.74 | [Table 1] |
| nMOS (Happy) | **3.78** | 3.72 | 3.77 | 3.73 | 3.62 | [Table 1] |
| nMOS (Sad) | 3.70 | 3.68 | 3.71 | 3.68 | 3.74 | [Table 1] |
| nMOS (Surprise) | **3.80** | 3.70 | 3.75 | 3.79 | 3.71 | [Table 1] |
| sMOS (avg) | **~3.72** | ~3.69 | ~3.69 | ~3.69 | ~3.67 | [Table 1] |
| eMOS (Angry) | 3.95 | 3.84 | 3.86 | 3.95 | **4.10** | [Table 1] |
| eMOS (Happy) | **3.83** | 3.76 | 3.71 | 3.67 | 3.68 | [Table 1] |
| eMOS (Sad) | 3.81 | 3.76 | 3.58 | 3.61 | **3.92** | [Table 1] |
| eMOS (Surprise) | **4.03** | 3.86 | 3.63 | 3.88 | 3.98 | [Table 1] |

### 消融实验 [Table 2]

| 配置 | IP | CS | CDS | DCT | nMOS | sMOS | eMOS | WER | CER | SECS | EECS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DiEmo-TTS (full) | Y | Y | Y | Y | 4.23 | 3.96 | 4.07 | 16.16 | 5.60 | 0.8505 | 0.4527 |
| w/o DCT | Y | Y | Y | N | 4.16 | **4.02** | 3.89 | 16.63 | 6.15 | **0.8626** | 0.4056 |
| w/o CDS | Y | Y | N | N | 4.18 | 3.84 | 4.01 | 16.72 | 6.05 | 0.8261 | 0.4461 |
| w/o CS | Y | N | N | N | 4.20 | 3.83 | 4.00 | 17.47 | 6.46 | 0.8246 | 0.4478 |
| w/o IP | N | N | N | N | 4.12 | 3.81 | 3.95 | 17.20 | 6.66 | 0.8217 | 0.4357 |

消融实验关键发现 [§3.6]:
1. **DCT** 显著提升 eMOS (+0.18) 和降低 WER (-0.47),但 SECS 略降 (0.8626→0.8505),说明 DCT 在更好整合情感时轻微牺牲了说话人保持度 [Table 2] [agent 解读]
2. **CDS** 提升 sMOS (+0.16) 和 SECS (+0.0365),说明 cluster-driven sampling 有效改善说话人保持 [Table 2]
3. **CS** 略微降低 eMOS 但改善 WER/CER,说明 cosine similarity loss 在嵌入空间质量和语言一致性方面有益 [Table 2]
4. **IP (formant perturbation)** 全面提升所有指标,是最基础的解耦组件 [Table 2]

### 情感聚类可视化 [Fig 3, Fig 4]

- t-SNE 可视化显示 cluster label 与 real emotion label 高度对齐 [Fig 3]
- 球面坐标中,不同说话人的情感 cluster 相对位置一致,验证了跨说话人情感对齐的有效性 [Fig 4]

## 局限性

1. **伪标签依赖** [§4]: 情感聚类依赖预训练的 emotional attribute predictor [19],后者在标注数据 (MSP-Podcast) 上训练,非完全无监督 [论文原文]
2. **小规模数据验证**: 仅在 ESD 数据集 (~350 句/人 x 10 人) 上验证,未在大规模真实数据上测试; 推理时仅用 2 个目标说话人 [§3.1]
3. **未处理未见说话人**: 跨说话人场景中,目标说话人仍在训练集中 (仅训练时只用 neutral 数据),非真正的 zero-shot [§3.1] [agent 解读]
4. **消融实验非独立**: Table 2 的消融是递进移除 (w/o DCT → w/o DCT+CDS → ...),无法精确量化每个组件的独立贡献 [agent 解读]
5. **MOS 差异小**: 与 baseline 的 nMOS/sMOS 差异在 0.1 以内,多数在置信区间内重叠 [Table 1]
6. **Expressive FS2 在 eMOS Angry/Sad 上更强**: 无解耦的 baseline 在部分情感上情感相似度更高,说明解耦可能以轻微的情感表达为代价 [Table 1]

## 点评

DiEmo-TTS 提出了一条有价值的 emotion disentanglement 新路线: 将 DINO 自监督蒸馏从 speaker verification 迁移到 emotion disentanglement,通过三个精巧的改造 (cluster-driven sampling, formant perturbation, cosine similarity loss) 避免了 GRL 的 trade-off 和 VQ 的信息丢失问题。

**亮点**: (1) 情感聚类方案 (convex hull 定位中性 + 球面坐标 + Hungarian 匹配) 是无监督情感对齐的优雅设计; (2) formant perturbation 作为解耦手段的 insight 很有启发性 -- 利用声学先验 (共振峰≈音色) 比通用数据增强更精准。

**不足**: 实验规模偏小 (ESD 350 句),实用场景验证不足; 消融实验设计不够严格 (递进移除而非独立移除); 与 Expressive FS2 的比较显示解耦带来的提升有限且情感保留可能受损。作为 Interspeech 短文,深度有限。

**定位**: 在情感解耦谱系中,DiEmo-TTS 代表了 "自监督蒸馏解耦" 的方向,与 GRL 对抗训练、VQ 瓶颈、正交损失形成互补。但该方向能否扩展到大规模多说话人场景仍待验证。

## 可复用的 idea

1. **Cluster-driven sampling for DINO**: 将 DINO 的正样本构造从 "同一 utterance 的不同 crop" 改为 "同一属性 cluster 的不同 utterance crop",可泛化到任何需要特定属性解耦的自监督蒸馏场景 (如 content-speaker 解耦、accent-speaker 解耦)
2. **Formant perturbation as speaker identity destruction**: 利用共振峰与说话人音色的强相关性,通过共振峰扰动精准破坏 speaker identity 而保留其他属性 -- 比 noise augmentation 更有针对性的数据增强策略
3. **Convex hull + spherical coordinates for emotion cluster alignment**: 无监督情感聚类对齐方案,利用 VAD 空间中中性情感的几何性质,可用于任何需要跨说话人情感一致性的场景
4. **Dual conditioning transformer**: weight-sharing multi-head attention 融合多种 style 信息,比简单拼接/加法更好地建模属性交互

---

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes | **问题**: 2 low
> - [low/template-compliance] frontmatter models 列了同组前作而非对比 baseline
> - [low/traceability-gap] 消融分析中一处 agent 推断未标注来源 (已修正)
> 详见 `_review/DiEmo-TTS-review.yml`

> 检索命中: [[SpeakerEmbedding]]✓, [[SpeechFactorization]]✓ | 参考: [[EmotionControlinTTS]][待确认], [[StyleTransferinTTS]][待确认], [[GradientReversalLayer]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认] | 未命中但可能相关: 无
