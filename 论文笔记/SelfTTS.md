---
type: paper
tier: deep
title: "SelfTTS: Cross-Speaker Style Transfer through Explicit Embedding Disentanglement and Self-Refinement using Self-Augmentation"
arxiv_id: "2603.22252"
source: "Sources/SelfTTS.pdf"
authors: [Lucas H. Ueda, João G. T. Lima, Pedro R. Corrêa, Flávio O. Simões, Mário U. Neto, Paula D. P. Costa]
year: 2026
venue: "arXiv"
tags: [TTS, cross-speaker-style-transfer, disentanglement, contrastive-learning, VITS, emotion, self-augmentation, voice-conversion, GRL]
concepts: ["[[GradientReversalLayer]]", "[[StyleTransferinTTS]]", "[[EmotionControlinTTS]]", "[[SpeechFactorization]]", "[[SpeakerEmbedding]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页: [[SpeechFactorization]]✓, [[SpeakerEmbedding]]✓, [[GradientReversalLayer]], [[StyleTransferinTTS]], [[EmotionControlinTTS]], [[VITS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechFactorization]], [[SpeakerEmbedding]], [[GradientReversalLayer]], [[StyleTransferinTTS]], [[EmotionControlinTTS]], [[VITS]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: SelfTTS 处于 VITS-based expressive TTS + GRL 对抗解耦这条线上。[[VITS]][待确认] 提供了 CVAE + normalizing flow + adversarial training 的端到端框架,SelfTTS 在此基础上加入双 Reference Encoder (speaker + emotion) 和解耦机制。[[GradientReversalLayer]][待确认] 页记录了 GRL 在 TTS 中的典型用法(如 IndexTTS2 的 CE-based GRL emotion-speaker 解耦),SelfTTS 提出的 cosine-based GRL 是对 vanilla CE-based GRL 的改进,不依赖外部标签而是直接在 embedding 之间施加正交约束。[[SpeechFactorization]]✓ 归纳了五类解耦方法(对抗训练/信息瓶颈/self-distillation/loss weighting/辅助技术),SelfTTS 属于对抗训练分支但融入了对比学习和自增强。

**已有认知**: [[EmotionControlinTTS]][待确认] 页已详细记录了从 embedding 到 DPO/RLHF 再到 training-free steering 的情感控制演进线,其中 DiEmo-TTS 代表了"自监督蒸馏"的新范式,EmoSteer-TTS/CoCoEmo 代表"激活 steering"范式。SelfTTS 则回归"对抗训练+对比学习"的经典范式,但用 cosine-based GRL + MPCL 提供了更强的组合方案。[[StyleTransferinTTS]][待确认] 页将跨说话人风格迁移归入 "Reference Speech Prompt" 类别,SelfTTS 属此类但不依赖外部预训练 encoder。

**创新判断**: 相对于 KB 已有知识,SelfTTS 的核心新意在于: (1) 用 cosine similarity 替代 CE 作为 GRL 的损失函数,绕过标签噪声问题; (2) MPCL 对比学习同时约束 speaker 和 emotion encoder,无需复杂 batching; (3) Self-Augmentation 利用模型自身 VC 能力生成训练数据改善自然度。这三个组件构成了一个自包含(不依赖外部模型)的跨说话人情感迁移方案。

## 速查

> [!summary] 速查
> - **一句话**: 基于 VITS 的跨说话人情感迁移系统,通过 cosine-based GRL 显式解耦 speaker/emotion embedding + MPCL 对比损失聚类 + Self-Augmentation 自增强提升自然度
> - **路线**: Mel slice → 双 Reference Encoder (speaker RE + emotion RE) → MPCL 聚类 + cosine-based GRL 解耦 → VITS posterior/flow/decoder 条件化 → 波形; Self-Aug 阶段用 VC permute speaker 生成合成情感参考
> - **指标**: eMOS 2.853 (最优, vs E3-VITS 2.237, VECL 2.556); EECS 0.8423 (vs 0.5367, 0.6929); SECS 0.8163; CKA(emb) 0.0139 (最低纠缠) [Table 1, Table 2]
> - **可借鉴**: (1) cosine similarity + GRL 的 label-free 解耦方案,可迁移到任何需要双因子正交化的场景; (2) MPCL 对比损失比 CE 更适合多正例聚类,且无需特殊 batch 设计; (3) Self-Aug 的 ENC 配置(仅替换 emotion encoder 输入)比 GT 替换更安全
> - **局限**: 仅在 ESD (20 speakers, 5 emotions) 上验证,规模偏小; 跨语料库 WER 极高(>1.0); 无零样本能力; 自然度 nMOS 2.746 仍显著低于 GT 3.638; 未与 LLM-based TTS 对比

## 核心问题

SelfTTS 要解决的核心问题是: **如何在不依赖外部预训练 speaker/emotion encoder 的前提下,实现高质量的跨说话人情感迁移?**

具体而言,现有方法面临三个挑战 [§1]:
1. **Speaker leakage**: Reference encoder 提取的 emotion embedding 中混入了说话人音色信息,导致跨说话人推理时身份不匹配
2. **Label-dependent GRL 的局限**: 传统 CE-based GRL 依赖外部标签做对抗训练,但当数据集中 emotion 和 speaker 存在统计相关性(如只有一个说话人提供情感数据)时,CE-based GRL 无法有效去除 speaker 信息 [§2.1.2]
3. **合成数据质量瓶颈**: VC 生成的合成数据中的伪影会限制 TTS 的上限 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SelfTTS 基于 VITS 框架,扩展了以下核心组件 [§2.1, Fig 1]:

1. **双 Reference Encoder**: 两个独立的 6 层 CNN + GRU 架构,分别从 mel-spectrogram 的随机切片中提取 speaker embedding (g) 和 emotion embedding (e)。输入为随机窗口切片(窗口大小在全长和半长之间随机选取),目的是增强对内容泄露的鲁棒性 [§2.1, 论文原文引用 Chen & Rudnicky 2022]
2. **条件化 VITS**: Posterior Encoder、Normalizing Flows (4 个 Residual Coupling Blocks)、Stochastic Duration Predictor、Waveform Decoder 均以 (g, e) 为条件 [§2.1]
3. **三层解耦机制**: MPCL 聚类 + embedding 级 cosine-based GRL + latent 级 cosine-based GRL [§2.1.1, §2.1.2]
4. **Self-Augmentation 自增强**: 利用 normalizing flow 的可逆性做 voice conversion,生成合成情感参考数据用于自训练 [§2.1.3]

### 关键设计选择

**设计选择 1: 为什么用 MPCL 而不是 CE 做 encoder 聚类?**

MPCL (Multi Positive Contrastive Learning) 将对比学习中的目标分布定义为一个 soft 分布,所有同类样本均为正例(Eq. 3-4),通过 cross-entropy 最小化 predicted 与 target 分布的距离 [§2.1.1]。与 CE 的关键区别在于: CE 仅要求分类正确,不显式要求同类样本聚集成 cluster; MPCL 要求同类样本在嵌入空间中彼此靠近 [论文原文]。消融实验证实: 不使用 GRL 时,MPCL 相比 CE 将 SECS 从 0.8064 提升到 0.8165,EECS 从 0.7391 提升到 0.7903,CKA 从 0.3148 降低到 0.0176 [Table 2]。[agent 解读] MPCL 的另一优势是不需要复杂的 batch 采样策略(如 N-pair loss 需要特定的正负例比例),在端到端训练中更容易集成。

**设计选择 2: 为什么用 cosine similarity 替代 CE 作为 GRL 的损失?**

传统方案: GRL + CE 分类器,让 emotion encoder 的输出无法被分类为正确的 speaker ID [论文原文]。问题: 当数据中 emotion 和 speaker 存在相关性(如单一说话人提供情感数据),CE 分类器无法有效工作,因为 speaker 预测本身就是有噪声的 [§2.1.2, 论文原文]。

SelfTTS 方案: 在 emotion embedding e 和 speaker embedding g 之间直接施加 cosine similarity 约束 (Eq. 5-6)。具体做法是通过一个 3 层 Linear Processor (Phi_linear) 将一个 embedding 映射后与另一个 embedding 的 detached 版本计算 cosine similarity,然后通过 GRL 反转梯度。这样 emotion encoder 被训练为使其输出与 speaker embedding 正交,反之亦然 [§2.1.2, 论文原文]。

[agent 解读] 这个设计的巧妙之处在于它完全绕过了标签: 不需要知道"这是哪个说话人",只需要确保两个 embedding 空间不共享信息。这使得它在标签有噪声或标签-属性不对称的数据集上更鲁棒。

此外,同样的 cosine-based GRL 还施加在 latent 层面 zp 上(Eq. 7-8): 通过 3 层 1D Conv Processor (Phi_conv) + mean pooling 从 zp 预测 speaker/emotion embedding,再施加 GRL,确保 normalizing flow 输出的 prior 表征 zp 不含 speaker 和 emotion 信息 [§2.1.2, 论文原文]。

消融结果: cosine-based GRL 在所有 GRL 配置中实现了最低的 CKA (0.0139 vs CE-GRL 0.0336 vs No-GRL 0.0176-0.3148) 和最高的 EECS (0.8793 vs 0.7899/0.7276) [Table 2]。

**设计选择 3: 为什么 Self-Augmentation 只用 ENC 配置?**

Self-Augmentation 利用 VITS 的 normalizing flow 可逆性实现 voice conversion: 源音频 → posterior encoder → z → forward flow → zp (style-neutral) → inverse flow (target speaker/emotion) → decoder → 合成波形 [§2.1.3, Eq. 9]。

论文测试了三种 Self-Aug 配置 [§3.1.3, Table 5]:
- **GT**: 合成样本替代 ground-truth 波形(用于重建损失)
- **ENC**: 合成样本仅作为 emotion encoder 的新参考输入(不同 speaker 声音表达同一 emotion)
- **BOTH**: 两者同时使用

ENC 配置在 UTMOS (3.6461) 和 SECS (0.8162) 上最优,虽然 EECS (0.8027) 略低于 GT (0.8896) [Table 5]。论文解释: GT 和 BOTH 配置的自然度下降可能因为 VC 生成的合成样本包含伪影,迫使模型拟合这些伪影而非真实声学模式 [§3.1.3, 论文原文]。

[agent 解读] ENC 配置的设计思路本质上是 data augmentation for the encoder only: 让 emotion encoder 看到更多"不同说话人表达同一情感"的样本,而不污染重建目标。这与 E3-VITS 的 batch-permuted style perturbation 思路相似,但 SelfTTS 的实现更简洁(直接用自身 VC 能力)。

Self-Aug 比例的消融(Table 6)显示: proportion 0.25 在 EECS (0.8423) 和 UTMOS (3.6104) 之间取得最佳平衡;比例越高,UTMOS 提升但 EECS 下降,确认了 emotional conditioning 与 naturalness 之间的 inverse relationship [Table 6, 论文原文]。

**设计选择 4: Normalizing Flow 中信息的流动模式**

LK-CKA per-flow-step 分析(Table 4)揭示了一个有趣的模式:
- Forward flow 的第 4 步(最终输出 zp): speaker LK-CKA 降至 0.0179,emotion LK-CKA 仅 0.0025 — 验证 zp 确实是 style-neutral/speaker-agnostic [Table 4]
- Inverse flow 中: emotion 信息逐步注入(从 0.1134 增长到 0.2158),而 speaker 信息保持低位(0.0348-0.0820)

[agent 解读] 这暗示 speaker conditioning 可能主要通过 decoder 而非 flow 注入,这与 VITS 的 HiFi-GAN decoder 直接接收 speaker embedding 的设计一致。emotion 通过 inverse flow 逐步注入则说明 flow 承担了韵律/情感的细粒度建模。

### 训练策略

1. **预训练初始化**: 从 VCTK 数据集上预训练 800k steps 的 VITS checkpoint 初始化(仅迁移兼容层),加速 phonetic alignment 和 waveform 生成学习 [§2.2]
2. **主训练**: 在 ESD 英语数据(10 speakers, 5 emotions)上训练 200k steps,lr=2e-4,AdamW [§2.2]
3. **Self-Augmentation 微调**: 50k steps,lr 降至 2e-5,batch 中 25% 替换为合成样本 [§2.2]
4. **推理**: 使用训练数据的 emotion/speaker embedding centroid prototypes [§2.2, 引用 Kwon et al. 2019]
5. **硬件**: 单卡 NVIDIA L40S (48GB),每次实验约 2 天 [§2.2]

总损失(Eq. 10): VITS 原始损失(重建 + KL + duration + adversarial + feature matching) + 4 个 cosine disentanglement 损失 + 2 个 MPCL 损失。

## 实验

| 指标 | SelfTTS | SelfTTS w/o Self-Aug | E3-VITS | VECL | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nMOS↑ | 2.746±0.109 | 2.228±0.105 | **2.763±0.111** | 2.707±0.107 | 3.638±0.106 | ESD | [Table 1] |
| eMOS↑ | **2.853±0.107** | 2.849±0.108 | 2.237±0.108 | 2.556±0.108 | 3.541±0.122 | ESD | [Table 1] |
| sMOS↑ | 3.112±0.114 | **3.121±0.112** | 2.815±0.116 | 2.804±0.114 | 4.009±0.107 | ESD | [Table 1] |
| UTMOS↑ | 3.6104 | 3.4899 | **3.9614** | 3.7959 | - | ESD | [Table 1] |
| WER↓ | 0.2305 | 0.2567 | 0.2011 | **0.1944** | - | ESD | [Table 1] |
| SECS↑ | **0.8163** | 0.8103 | 0.8069 | 0.7806 | - | ESD | [Table 1] |
| EECS↑ | 0.8423 | **0.8793** | 0.5367 | 0.6929 | - | ESD | [Table 1] |
| CKA(emb)↓ | **0.0139** | - | - | - | - | ESD | [Table 2] |

**关键发现**:

1. **情感迁移 vs 自然度的 trade-off**: E3-VITS 的高 UTMOS/nMOS 伴随最差的 eMOS/EECS,论文解释为"无法生成目标情感导致生成接近 Neutral 的语音,反而更自然" [§3, 论文原文]。这揭示了一个重要洞察: UTMOS 等自然度指标可能被 emotional leakage 误导。

2. **Self-Augmentation 的效果**: w/o Self-Aug 版本在 eMOS 和 SECS 上已具竞争力,但 nMOS 最低(2.228);Self-Aug 将 nMOS 提升 0.518,同时 EECS 从 0.8793 降至 0.8423 [Table 1]。这确认了 Self-Aug 主要改善自然度,情感迁移能力主要来自解耦机制。

3. **跨语料库实验**([Table 7]): 对 LJSpeech 和 VCTK speakers (p226, p231),WER 飙升至 >1.0,EECS 降至 0.60-0.69,SECS 保持较高(0.86-0.91)。Self-Aug 在跨语料库场景中仍显著改善 UTMOS 和 WER [Table 7]。

4. **Emotion encoder 选择**([Table 3]): Reference Encoder (RE) 不使用 mel filtering 或 timbre perturbation 时 EECS 最高(0.8793);20-mel-bin filtering 和 timbre perturbation 均显著降低 EECS。StyleSpeech/StyleTTS encoder 不如 RE [Table 3]。论文推测: 显式 embedding 解耦可能受益于 speaker/emotion encoder 使用相似架构来稳定对抗训练 [§3.1.2, 论文原文]。

## 局限性

1. **数据规模小**: 仅在 ESD (10 English speakers, 5 emotions, ~14.5h) 上验证,未在大规模数据集(如 Emilia)上测试,可扩展性未知 [§4.1]
2. **无零样本能力**: 推理时使用训练集的 centroid prototypes,无法处理未见说话人或未见情感 [§4.1]
3. **跨语料库退化严重**: WER >1.0,表明模型对录音条件差异非常敏感 [Table 7]
4. **自然度仍有明显差距**: nMOS 2.746 vs GT 3.638,差距 0.892,sMOS 3.112 vs GT 4.009 [Table 1]
5. **MOS 绝对值偏低**: 即使是 GT 的 nMOS 也仅 3.638,可能反映 ESD 数据集本身质量或评估设计的限制
6. **未与 LLM-based TTS 对比**: 未比较 CosyVoice 2、IndexTTS2 等新一代系统,定位于 VITS-based 系统内的比较
7. **情感类别有限**: 仅 5 类离散情感,未探索连续情感空间或混合情感
8. **Self-Aug 的 emotion-naturalness trade-off**: 更多合成数据提高自然度但降低情感保持度,两者无法同时优化 [Table 6]

## 点评

**优势**: SelfTTS 在一个清晰定义的问题(VITS-based 跨说话人情感迁移)上提供了系统性的解决方案。cosine-based GRL 的设计直觉清晰——直接在 embedding 之间施加正交约束而不依赖可能有噪声的外部标签——并通过 CKA 指标(0.0139 vs CE-GRL 的 0.0336)验证了其有效性。消融实验覆盖全面(encoder loss、GRL 类型、encoder 架构、Self-Aug 配置和比例),每个设计选择都有 ablation 支撑。开源代码和使用公开数据集提高了可复现性。

**不足**: 在 2026 年的 TTS landscape 中,VITS-based 系统已不是主流方向。LLM-based TTS (VALL-E, CosyVoice, Seed-TTS) 和 flow-matching TTS (F5-TTS, E2-TTS) 已展示了远超 VITS 的自然度和零样本能力。SelfTTS 的 nMOS 2.746 和 WER 0.2305 在绝对值上偏低。跨语料库 WER >1.0 表明该方法在实际应用中面临严重挑战。Self-Aug 的 emotion-naturalness trade-off 也暗示该方法在根本层面上受到 VITS decoder 能力的约束。

**在研究谱系中的位置**: SelfTTS 延续了 "对抗训练 + 对比学习" 做 emotion-speaker 解耦的经典路线,cosine-based GRL 是该路线的一个有价值的改进。但该路线整体上正在被 DiEmo-TTS (自监督蒸馏)、EmoSteer-TTS (training-free 激活 steering)、DiffRO (reward-guided 优化) 等新范式超越。SelfTTS 的 Self-Augmentation 思想(利用模型自身 VC 能力生成训练数据)与 E3-VITS 类似,但 ENC 配置比全量替换更精细。

## 可复用的 idea

1. **Cosine-based GRL 替代 CE-based GRL**: 当需要解耦两个因子但标签有噪声或标签-因子存在统计相关性时,直接在 embedding 之间施加 cosine similarity + GRL 是更鲁棒的方案。可迁移到 codec disentanglement、multi-attribute TTS 等场景。
2. **MPCL 对比损失用于 embedding 聚类**: 相比 CE,MPCL 不需要特殊 batch 设计就能生成紧凑的类内聚类,适用于任何需要 label-guided embedding clustering 的场景。
3. **Self-Augmentation 的 ENC 配置**: 只替换 encoder 输入而不替换重建目标,避免合成伪影污染训练。这种"在输入端做 augmentation,在输出端保持干净"的策略可泛化。
4. **LK-CKA 逐层分析**: 用 Label Kernel CKA 追踪信息在模型各层的流动,是分析 disentanglement 效果的有用工具,不局限于 TTS。
5. **Emotion-naturalness trade-off 的量化**: proportion 参数提供了一个连续可调的旋钮来控制情感保持度和自然度的平衡,这种显式 trade-off 建模在工程实践中有价值。
