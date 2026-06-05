---
type: paper
tier: deep
title: "Targeted Speaker Poisoning Framework in Zero-Shot Text-to-Speech"
arxiv_id: "2603.07551"
source: "Sources/SpeakerPoisoning.pdf"
authors: [Thanapat Trachu, Thanathai Lertpetchpun, Sai Praneeth Karimireddy, Shrikanth Narayanan]
year: 2026
venue: "arXiv preprint"
tags: [speaker-poisoning, voice-privacy, zero-shot-TTS, machine-unlearning, knowledge-distillation, contrastive-learning, triplet-loss, speaker-identity, evaluation-framework]
concepts: ["[[SpeakerEmbedding]]", "[[SpeakerVerification]]", "[[DiffusionModel]]", "[[Anti-spoofingandDeepfakeDetection]]", "[[VoiceCloningTaxonomy]]", "[[TTSEvaluation]]"]
models: ["[[WavLM]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeakerEmbedding]], [[SpeakerVerification]], [[DiffusionModel]], [[Anti-spoofingandDeepfakeDetection]], [[VoiceCloningTaxonomy]], [[TTSEvaluation]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文是 ZS-TTS **speaker identity erasure** 路线的第二篇工作(按发布时间),与 [[论文笔记/SpeakerIdentityUnlearning|Kim et al. (ICML 2025)]] 共享问题定义但**切换了目标模型和方法设计**。Kim et al. 在 VoiceBox (CFM-based) 上实施 Teacher-Guided Unlearning (TGU),本文将类似框架 (TGP) 迁移到 **StyleTTS2 (diffusion-based)** 并新增 Encoder-Guided Poisoning (EGP)。同时期 [[论文笔记/Training-freeSpeakerUnlearning|TruS (Lee et al., 2026)]] 走了完全不同的路线 -- 无训练推理时 activation steering。三篇工作构成了 speaker unlearning/poisoning 的三条技术路线: 模型重训 (TGU on VoiceBox) → 模型微调 (TGP/EGP on StyleTTS2) → 推理时干预 (TruS on F5-TTS)。
>
> **已有认知**: [[SpeakerEmbedding]] 页 (confirmed) 记录了 speaker embedding 在 ZS-TTS 中作为 identity 载体的角色;本文操作的目标正是 StyleTTS2 diffusion module 中的 style representation,通过让模型将 forget speaker 的 style 映射到 retain speaker 的 style 来实现 identity erasure。[[SpeakerVerification]] 页 [待确认] 记录了 WavLM-TDCNN 等 encoder 用于 SIM 计算的标准做法,本文正是使用 wavlm-base-plus-sv 进行评估,并在此基础上提出了分布级新指标 AUC 和 FSSIM。[[DiffusionModel]] 页 [待确认] 记录了 diffusion 在 TTS 中的应用,StyleTTS2 正是使用 diffusion model 建模 style representation,本文仅微调 diffusion module 而冻结其他组件,确保 poisoning 仅影响 speaker identity 而不损害语音生成能力。[[Anti-spoofingandDeepfakeDetection]] 页 [待确认] 已记录 SafeSpeech (数据端扰动)、TraceableSpeech (水印溯源)、Kim et al. TGU (模型级遗忘) 三层防线,本文 EGP 属于模型级干预的改进。[[TTSEvaluation]] 页 [待确认] 记录了 SIM 指标依赖 encoder 选择的问题,本文提出的 FSSIM 和 AUC 正是为解决单一 cosine similarity 不足以评估 erasure 效果的局限。
>
> **创新判断**: 相对于 Kim et al. (TGU on VoiceBox),(1) EGP 绕过 teacher 生成,直接用 style encoder 输出作为 fine-tuning target,规避了同容量 teacher-student 蒸馏效率低的问题 [Stanton et al. NeurIPS 2021];(2) 提出 AUC + FSSIM 两个分布级评估指标,比 Kim et al. 的 spk-ZRF 更直观地衡量 retain/forget 分布的可分离性;(3) 首次系统评估了多说话人(15/100)场景下的 scalability 限制,揭示了 identity overlap 导致的 latent space crowding 问题。
>
> 检索命中: [[SpeakerEmbedding]]✓, [[SpeakerVerification]][待确认], [[DiffusionModel]][待确认], [[Anti-spoofingandDeepfakeDetection]][待确认], [[VoiceCloningTaxonomy]][待确认], [[TTSEvaluation]][待确认] | 过滤: 无 | 未命中但可能相关: [[Classifier-FreeGuidance]] (StyleTTS2 不使用 CFG,与 Kim et al. VoiceBox 的 CFG 分析无直接关联)

## 速查

> [!summary] 速查
> - **一句话**: 将 speaker identity erasure (poisoning) 从 VoiceBox 迁移到 StyleTTS2,提出 Encoder-Guided Poisoning (EGP) 绕过 teacher 生成瓶颈,引入 AUC + FSSIM 分布级评估框架,揭示多说话人 scalability 限制
> - **路线**: pre-trained StyleTTS2 → 冻结 text encoder/decoder/discriminator,仅微调 diffusion module → TGP: teacher 用 retain speaker 生成 mel target, student 训练时以 p_forget 概率替换 retain ref 为 forget ref → EGP: 改用 style encoder 输出替代 teacher 生成的 mel 作为 target → 可选加 triplet loss 显式推开 forget embedding → 60K steps, AdamW, lr=1e-4
> - **指标**: 1-speaker: EGP+Trip. AUC 0.95 (最高), SSIM-F 0.48 (最低) [Table 1]; 15-speaker: EGP+Trip. AUC 0.76, SSIM-F 0.71, Avg-FSSIM 0.69 [Table 2]; 100-speaker: EGP+Trip. AUC 0.64, Avg-FSSIM 0.71, Max-FSSIM 0.91 [Table 2]; EGP retain WER 2.90 (vs pretrained 2.75) [Table 1]
> - **可借鉴**: (1) EGP 的核心 insight -- 当 teacher 和 student 架构相同时,用 encoder 输出替代 teacher 生成输出可提供更干净的优化信号,这一思路可迁移到其他同架构蒸馏场景; (2) AUC 作为 retain/forget 分布可分离性的度量,比单一 cosine similarity 阈值更鲁棒; (3) Max-FSSIM 概念 -- 衡量生成语音与 forget set 中**任意**说话人的最大相似度,揭示 worst-case leakage
> - **局限**: (1) 仅在 StyleTTS2 (diffusion-based) 上验证,未测试 AR 或 LLM-based TTS; (2) 100 speaker 时 AUC 仅 0.64,Max-FSSIM 高达 0.91,scalability 问题未解决; (3) triplet loss 在多说话人场景效果大幅下降 -- 推开一个 negative 会推向另一个 [Sohn 2016]; (4) 训练代码和模型权重尚未公开 (upon acceptance); (5) 未讨论 forget speaker 与 retain speaker 声音高度相似时的误伤问题

## 核心问题

ZS-TTS 模型可以从短参考音频零样本克隆任意说话人的声音,引发严重的隐私风险 [§1]。现有防护思路的局限:

1. **传统 machine unlearning 不适用于 ZS-TTS** -- ZS-TTS 的零样本泛化能力意味着,即使用没有 forget speaker 的数据重训模型,模型仍可能通过 in-context learning 复制其声音 [§1]。这是 ZS-TTS 与传统分类模型 unlearning 的根本区别 [论文原文]
2. **推理时过滤可被绕过** -- 对输入 prompt 做 speaker filtering 虽然可行,但当模型权重公开时,攻击者可直接访问未过滤的模型 [§1, refs 10-11] [论文原文]
3. **需要参数级修改** -- 仅靠外部管线无法保障隐私,必须修改模型内部参数使其丧失生成特定 speaker identity 的能力 [§1] [论文原文]

本文将这个任务形式化为 **Speech Generation Speaker Poisoning (SGSP)**: 定义 forget set F (不应合成的 speaker) 和 retain set R = S \ F (需保持合成能力的 speaker),目标是修改模型使其在收到 F 中 speaker 的 prompt 时无法复制其 identity,同时保持 R 的合成质量 [§2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

**骨架模型: StyleTTS2** [§3.1]

StyleTTS2 由六个模块组成,可分为三组:
- 语音生成系统: text encoder + style encoder + speech decoder
- TTS 预测系统: duration predictor + prosody predictor
- Diffusion sampler: diffusion model (建模 style representation)

**关键设计选择: 仅微调 diffusion module** [§3.1]

作者选择冻结所有其他组件,仅用 diffusion loss (style encoder 输出与 diffusion 输出间的 L2 loss) 微调 diffusion module。理由: speaker identity 主要编码在 style representation 中,仅修改 diffusion module 确保 poisoning 精确作用于 identity 维度,不损害文本编码/解码/韵律预测等能力 [论文原文]。

### 关键设计选择

#### 1. Naive Baselines (推理时过滤) [§3.2]

两种预处理基线,用于说明外部过滤的局限:

- **Pretrained + Speaker Filtering (SF)**: 用 WavLM 提取 speaker embedding,计算输入 ref 与 forget set 的 cosine similarity,超过 0.86 阈值则判为 forget speaker,迭代替换为 retain speaker ref (similarity < 0.86)
- **Pretrained + Ground Truth Filtering (GTF)**: 假设完美知道 ref 是否属于 F,直接替换。这是外部过滤的理论上限

GTF 证明"理想过滤 + 不修改模型"可以同时实现高 utility 和 privacy [Table 1],但 SF 在现实中阈值选择困难且模型权重公开时可被绕过 [论文原文]。

#### 2. Teacher-Guided Poisoning (TGP) [§3.3]

从 Kim et al. [12] 的 VoiceBox TGU 框架改编到 StyleTTS2:

**训练流程** [Fig 1a]:
1. Teacher model (frozen pretrained StyleTTS2) 接收 retain speaker 的 transcript,生成 mel spectrogram 作为 ground truth target (y_retain)
2. Student model (初始化自同一 pretrained) 接收相同 transcript
3. 训练时,student 的 speaker reference 以概率 p_forget 被替换为 forget set 中的 speaker
4. L2 loss: student 输出 vs teacher 输出
5. 效果: student 学会在收到 forget speaker ref 时输出 retain speaker 的声音 → speaker identity 被"毒化"

**为什么 work**: 通过让模型在 forget speaker 条件下拟合 retain speaker 的输出分布,diffusion module 学到的 style mapping 被重写 -- forget speaker 的 style embedding 被映射到 retain set 中随机 speaker 的 style,实质上抹除了 forget speaker 的 identity 信息 [agent 解读]。

#### 3. Encoder-Guided Poisoning (EGP) [§3.3]

EGP 与 TGP 共享完全相同的训练流程和目标,唯一区别在于 **ground truth target 的来源**:

- TGP: target = teacher model 的生成输出 (mel spectrogram)
- EGP: target = style encoder 的输出 (直接从真实音频提取)

**为什么 EGP 优于 TGP**: Stanton et al. [17] 证明,当 student 和 teacher 的模型容量完全相同时,knowledge distillation 难以获得性能增益。TGP 中 teacher 和 student 都是 StyleTTS2,teacher 生成的 mel spectrogram 引入了不必要的生成噪声。EGP 绕过这个问题,直接用 style encoder 提取的 representation 作为优化目标,提供更干净的信号 [论文原文] [§3.3]。

[agent 解读] 本质上,TGP 的 target 经历了 "ref audio → style encoder → diffusion → decoder → mel" 的完整生成管线,每一步都引入噪声;EGP 直接用 "ref audio → style encoder" 的输出,跳过了生成管线中的信息损失。

#### 4. Contrastive Learning (Triplet Loss) [§3.3]

在 TGP/EGP 基础上叠加 triplet loss,显式地将 diffusion 输出推离 forget set:

$$L_{triplet} = \max(||x - a||^2_2 - ||x - n||^2_2 + \beta, 0)$$

- x: diffusion 输出
- a: anchor (retain speaker ground truth)
- n: negative (从 forget set 采样)
- β: margin (设为 0.3)

仅在 F-conditioned generation 时施加此损失。效果: 在 1-speaker 设定下显著提升 privacy (AUC 从 0.79→0.95 for EGP),但在多说话人设定下效果减弱 [论文原文]。

**为什么 triplet loss 在多说话人时失效**: 推开一个 negative sample 可能将 embedding 推向 forget set 中另一个 speaker 的方向 [§6.2, ref 23]。这是 metric learning 中的已知问题 (N-pair loss, Sohn 2016),forget set 越大,latent space 越拥挤,单个 negative 的 triplet loss 越难同时避开所有 forget speaker [论文原文]。

### 训练策略

- Fine-tune 60,000 steps [§5.2]
- AdamW optimizer, lr = 1e-4 [§5.2]
- Triplet loss weight = 1.0, margin β = 0.3 [§5.2]
- Forget ratio α = 0.5 (训练时 50% 概率用 forget ref 替换 retain ref) [§5.2]
- 数据: LibriTTS train-clean-100 + train-clean-360 [§5.1]
- 仅微调 diffusion module,冻结 text encoder/decoder/discriminator [§3.1]

## 实验

### 评估框架 [§4]

本文的主要贡献之一是系统化的评估框架:

**Utility 指标**:
- WER: Whisper-medium ASR,30s 以下 utterance,EnglishTextNormalizer 预处理 [§4.1]
- UTMOS: 自动 MOS 代理,1-5 分 [§4.1]
- SSIM (retain): wavlm-base-plus-sv cosine similarity,衡量 retain set 的 identity 保持 [§4.1]

**Privacy 指标**:
- **AUC (Easy Condition)** [§4.2]: 衡量 retain/forget 两个 similarity 分布的可分离性。AUC=0.5 表示完全重叠(无 poisoning 效果),AUC=1.0 表示完美分离。比单一 cosine similarity 阈值更鲁棒,因为它考虑了分布的全局形状
- **FSSIM (Strong Condition)** [§4.2]: 计算生成样本与 forget set 中**所有** speaker 的 similarity,而非仅与 prompted speaker。Avg-FSSIM 衡量平均泄露,Max-FSSIM 衡量 worst-case 泄露。Max-FSSIM 的意义: 即使生成语音不像被 prompted 的 forget speaker,如果它碰巧像 forget set 中另一个 speaker,仍算隐私泄露

### 主要结果

| 指标 | PT (baseline) | PT+GTF (理想过滤) | TGP | EGP | EGP+Trip. | 设定 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER-R | 2.75 | 2.75 | 2.74 | 2.90 | 2.89 | 1 spk | [Table 1] |
| WER-F | 2.72 | 2.57 | 2.62 | 3.00 | 7.86 | 1 spk | [Table 1] |
| MOS-R | 4.29 | 4.29 | 4.34 | 4.28 | 4.27 | 1 spk | [Table 1] |
| SSIM-R | 0.87 | 0.87 | 0.85 | 0.88 | 0.87 | 1 spk | [Table 1] |
| SSIM-F | 0.88 | 0.63 | 0.77 | 0.77 | 0.48 | 1 spk | [Table 1] |
| AUC | 0.47 | 0.90 | 0.74 | 0.79 | 0.95 | 1 spk | [Table 1] |
| SSIM-R | — | 0.87 | 0.81 | 0.87 | 0.85 | 15 spk | [Table 2] |
| SSIM-F | — | 0.63 | 0.72 | 0.74 | 0.71 | 15 spk | [Table 2] |
| AUC | — | 0.91 | 0.66 | 0.77 | 0.76 | 15 spk | [Table 2] |
| Max-FSSIM | — | 0.91 | 0.91 | 0.91 | 0.91 | 15 spk | [Table 2] |
| SSIM-R | — | 0.87 | 0.79 | 0.83 | 0.78 | 100 spk | [Table 2] |
| AUC | — | 0.91 | 0.53 | 0.57 | 0.64 | 100 spk | [Table 2] |
| Max-FSSIM | — | 0.94 | 0.95 | 0.94 | 0.91 | 100 spk | [Table 2] |

### 关键发现

1. **EGP > TGP**: EGP 在所有设定下都优于 TGP,验证了"同容量 teacher-student 蒸馏效率低"的假设 [§6.1]
2. **Triplet loss 的双面性**: 1-speaker 时大幅提升 privacy (AUC 0.79→0.95),但 WER-F 从 3.00 飙升至 7.86;100-speaker 时仅微弱改善 AUC (0.57→0.64) [Table 1, Table 2]
3. **Scalability 瓶颈**: 100 speaker 时 TGP 变体的 AUC 降至 0.51-0.53 (接近随机),EGP+Trip. 仍达 0.64 但远低于 1-speaker 的 0.95;Max-FSSIM 持续高于 0.91,说明即使 Avg-FSSIM 达标,worst-case 仍有严重泄露 [§6.2]
4. **GT Filtering 的参考价值**: PT+GTF 在所有设定下保持高 AUC (0.90-0.91),说明 identity erasure 的理论上限很高,当前 parameter-modification 方法还有很大提升空间 [agent 解读]

## 局限性

1. **Scalability 未解决**: 100 speaker 时 retain/forget 分布几乎完全重叠 (AUC ≈ 0.5),Max-FSSIM 持续高达 0.91-0.95,当前框架无法有效处理大规模 forget set [§6.2]
2. **仅验证 StyleTTS2**: 未测试 AR 架构 (VALL-E)、LLM-based TTS (CosyVoice) 或 flow-matching 架构 (F5-TTS),泛化性未知 [agent 解读]
3. **Privacy-utility trade-off 未优化**: EGP+Trip. 在 1 speaker 时 F-WER 高达 7.86 (比 pretrained 高 3x),说明 triplet loss 的显式推离有时将 embedding 推入低可懂度区域 [§6.1, ref 22]
4. **Triplet loss 在高维拥挤空间的理论限制**: N-pair loss 等改进方法 [ref 23] 可能缓解但未被探索 [agent 解读]
5. **代码未公开**: 声称 "upon acceptance" 释放训练代码和权重,但截至论文发布时不可用 [§1]
6. **forget set 从训练集采样**: 评估设定中 forget speaker 都是训练时见过的 in-domain speaker,未测试 out-of-domain speaker 的 poisoning 效果 [§2]

## 点评

**定位**: 本文是 [[论文笔记/SpeakerIdentityUnlearning|Kim et al.]] 工作的自然延伸,将 TGP 从 VoiceBox 迁移到 StyleTTS2,但真正的贡献在于 **EGP 方法** 和 **评估框架**,而非 TGP 的迁移本身。

**方法层面**: EGP 的 insight 虽然直观 (绕过 teacher 生成噪声),但验证了一个重要的实践原则 -- 在同架构 distillation 场景下,直接用 intermediate representation 作为 target 比用生成输出更有效。这一点对 TTS 领域的 knowledge distillation 有普适参考价值。

**评估层面**: AUC + FSSIM 的贡献比方法本身更有持久价值。Kim et al. 的 spk-ZRF 用 JSD 度量随机性,而 FSSIM 的 Max/Avg 分离 + AUC 的分布视角提供了更直观且更严格的评估视角。特别是 Max-FSSIM 揭示了一个被忽视的风险: 即使平均相似度合格,worst-case 仍可能暴露 identity。

**与同期工作的比较**: 与 [[论文笔记/Training-freeSpeakerUnlearning|TruS]] 相比,本文方法仍需微调 (60K steps),且仅在训练集内 speaker 上验证,而 TruS 实现了 0 训练成本和 unseen speaker 支持。但本文在 1-speaker 场景下的 AUC (0.95) 远高于 TruS 的表现,说明 parameter-modification 方法在小规模 forget set 上仍有优势。

**局限感**: 论文坦诚了 100-speaker 的 scalability 限制 (这是好的),但未深入分析 **为什么** latent space crowding 对 diffusion model 特别严重,也未提出缓解方向 (如分层 poisoning、curriculum training 等)。

## 可复用的 idea

1. **Encoder-Guided target 替代 teacher generation**: 在 teacher-student 同架构的场景下,用 encoder 的 intermediate output 替代完整生成管线的输出作为蒸馏目标,减少生成噪声。适用于任何基于 knowledge distillation 的模型修改场景
2. **AUC 作为分布级可分离性度量**: 当需要评估两组样本 (如 retain vs forget, real vs fake, in-domain vs out-of-domain) 的分离程度时,AUC 比固定阈值的准确率更鲁棒
3. **Max-FSSIM 的 worst-case 思维**: 在安全/隐私评估中,最大相似度比平均相似度更有意义。可迁移到 deepfake detection、data privacy 等场景
4. **仅微调特定模块实现定向修改**: 通过冻结与目标无关的模块 (text encoder, decoder),仅微调与 identity 直接相关的模块 (diffusion),实现精确干预且最小化 collateral damage

## 审阅

> [!review] 审阅 (2026-06-05, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法 WHY 解释清晰,EGP/triplet 理论动机充分 |
> | 可信赖 | pass-with-fixes | 数字全部正确;速查卡片 SSIM-F/FSSIM 命名曾混淆(已修正) |
> | 可区分 | pass | [论文原文]/[agent 解读] 标记一致,覆盖率 > 80% |
> | 可定位 | pass | 三条技术路线对比出色,与 KB 已有 2 篇 unlearning 笔记定位清晰 |
> | 不污染 | pass | concepts/models 挂接合理,反向更新均为 append |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> 详见 `_review/SpeakerPoisoning-review.yml`
