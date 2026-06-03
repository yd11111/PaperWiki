---
type: paper
tier: deep
title: "MPE-TTS: Customized Emotion Zero-Shot Text-To-Speech Using Multi-Modal Prompt"
arxiv_id: "2505.18453"
source: "Sources/MPE-TTS.pdf"
authors: [Zhichao Wu, Yueteng Kang, Songjun Cao, Long Ma, Qiulin Li, Qun Yang]
year: 2025
venue: "arXiv (Interspeech submission)"
tags: [TTS, zero-shot, emotion, multi-modal, disentanglement, diffusion, prosody]
concepts: ["[[Emotion Control in TTS]]", "[[Prosody Modeling]]", "[[Speech Factorization]]", "[[Natural Language Description for TTS]]", "[[Speaker Embedding]]", "[[Diffusion-based TTS]]", "[[Global Style Tokens]]"]
models: ["[[模型库/HierSpeech++|GenerSpeech]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["LibriTTS", "MEAD-TTS"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: MPE-TTS 位于 **多模态情感可控零样本 TTS** 的交叉领域。在 KB 已有认知中:
- [[Zero-shot Speech Synthesis]] 的主流方案已演进到 LLM + 离散 token 范式 (CosyVoice, Seed-TTS 等,使用 100K+ 小时数据),MPE-TTS 采用的是更传统的 conformer encoder + diffusion decoder 路线,规模较小 (LibriTTS 585h + MEAD-TTS 36h)
- [[Speech Factorization]] 记录了从 GST → 对抗训练 → Information bottleneck → Self-distillation 的解耦演进,MPE-TTS 采用的是 bottleneck + 输入设计的组合策略,属于中期方案
- [[Prosody Modeling]] 梳理了显式 (FastSpeech 2 variance adaptor) 到隐式 (VAE/Flow/LLM in-context) 的韵律建模演进,MPE-TTS 的 LLM-like AR prosody predictor 介于两者之间 — 用 AR Transformer 在 VQ prosody code 上建模
- [[Emotion Control in TTS]] [待确认] 记录了情感建模从 embedding → 层级建模 → 对抗解耦 → DPO → 球面向量的演进,MPE-TTS 的 Emotion2Vec + CLIP adapter 多模态方案是该维度的新扩展
- [[Natural Language Description for TTS]] [待确认] 梳理了从 PromptTTS → InstructTTS → Parler-TTS 的文本描述路线,MPE-TTS 进一步将描述扩展到图像和语音模态
- [[Speaker Embedding]] 记录了 ECAPA-TDNN 作为当前最常用的 speaker encoder,MPE-TTS 也采用类似架构

**创新判断**: 与 KB 已有的情感控制方案 (EmoCtrl-TTS 的帧级 arousal-valence, EmoSphere-TTS 的球面向量, Daisy-TTS 的韵律嵌入分解) 相比,MPE-TTS 的独特之处在于 **多模态情感输入** (文本/图像/语音三选一) + **Emotion2Vec 作为统一情感锚点**。但在解耦精细度和数据规模上不如前沿系统。

> 检索命中: [[Prosody Modeling]]✓, [[Speech Factorization]]✓, [[Zero-shot Speech Synthesis]]✓, [[Speaker Embedding]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[Natural Language Description for TTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出多模态情感提示 (文本/图像/语音) 的零样本 TTS,通过层级解耦 + Emotion2Vec 锚定 + AR prosody predictor 实现情感与音色的独立定制
> - **路线**: 文本→Conformer content encoder→LR→Diffusion decoder→mel→HiFi-GAN; 情感提示→MPEE(Emotion2Vec/CLIP+adapter)→emotion code; 参考语音→ECAPA-TDNN timbre encoder→speaker vector; (content+emotion+timbre)→AR prosody predictor→prosody code
> - **指标**: MOS 3.73 (speech prompt), ESMOS 4.05, EACC 48% vs GT 54% on MEAD-TTS [Table 1]; 文本提示 MOS 3.65, EACC 46% [Table 2]; 图像提示 ESMOS 3.90, EACC 47% [Table 3]
> - **可借鉴**: (1) 用预训练 Emotion2Vec 作为跨模态情感对齐的锚点,MSE loss 拉齐 CLIP text/image encoder 到语音情感空间; (2) Emotion Consistency Loss (ECL) — 在 prosody predictor 后加情感分类器,确保预测的韵律保留情感信息
> - **局限**: 数据规模小 (MEAD-TTS 仅 36h/48 人/8 情感); EACC 上限 54% (GT 的 SER 识别率本身不高); 无开源代码; 仅复现 MM-TTS 作为 baseline (无官方实现); 未与现代大规模 ZS-TTS 系统对比

## 核心问题

现有零样本 TTS 系统大多依赖 **单一模态** 的风格提示 — 参考语音或文本描述,限制了灵活性。参考语音中各属性 (timbre/emotion/prosody) 深度纠缠,难以实现 **独立定制** (如"用 A 的声音说 B 情感的话")。文本描述虽灵活,但用户往往难以精确描述所需情感。MPE-TTS 的核心问题是: **如何让用户从文本、图像、语音三种模态中任选一种作为情感提示,同时保持音色与情感的独立可控?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MPE-TTS 由三个核心组件构成 [§2, Fig 1]:
1. **Multi-Modal Prompt Emotion Encoder (MPEE)**: 从文本/图像/语音提示中提取统一情感编码
2. **Diffusion-based Acoustic Model**: conformer content encoder + timbre encoder + prosody encoder + duration predictor + diffusion decoder,生成 mel-spectrogram
3. **LLM-like Prosody Predictor**: AR Transformer 基于 content + emotion + timbre 预测韵律编码

训练分三阶段 [§2, Fig 1(a)]:
1. Emotion Training: 训练 MPEE (对齐文本/图像 encoder 到语音情感空间)
2. Acoustic Model Training: 训练 AM (content encoder + timbre encoder + prosody encoder + diffusion decoder)
3. Prosody Training: 训练 AR prosody predictor (使用 AM 中已训练好的 prosody encoder 生成 target)

### 关键设计选择

#### 1. 层级解耦策略 (Hierarchical Disentangling)

[论文原文] 将语音特征按粒度分为两层 [§2.1]:
- **粗粒度 (全局向量)**: timbre 和 emotion — 在一段语音内不会剧烈变化,建模为全局向量更高效
- **细粒度 (帧级序列)**: content 和 prosody — content 是逐帧的时序信息,prosody 有高动态范围需帧级建模

[论文原文] 解耦手段 [§2.1]:
- Timbre: 用同一说话人的 **另一句话** 提取全局 speaker vector (来自 ECAPA-TDNN-like encoder),因此 timbre encoder 只能获取说话人身份,无法获取当前句子的 content 信息 — 这是经典的 bottleneck 解耦
- Prosody: VQ-based prosody encoder 仅取 mel-spectrogram 的 **低 20 bins** 作为输入,该频段包含几乎完整的韵律信息但大幅削弱了 timbre 和 content [§2.1, 引用 ProsoSpeech]
- Emotion: 训练 prosody predictor 时,emotion code 来自与目标语音 **同情感标签但不同说话人和内容** 的另一条语音 [§2.3],进一步切断 emotion 与 speaker/content 的关联

[agent 解读] 这种"输入设计"式解耦 (控制每个 encoder 能"看到"什么) 是一种轻量方案,不需要对抗训练或 gradient reversal,但其解耦程度依赖数据中的属性分布 — 如果训练数据中某些说话人只出现特定情感,解耦可能不完全。

#### 2. Multi-Modal Prompt Emotion Encoder (MPEE)

[论文原文] 设计思路 [§2.2, Fig 2]:
- 语音情感编码: 直接使用预训练 **Emotion2Vec+ Large** (冻结),输出作为情感空间的锚点
- 文本情感编码: CLIP text encoder (冻结) + 可学习 adapter layer
- 图像情感编码: CLIP image encoder (冻结) + 可学习 adapter layer
- 对齐: MSE loss 将文本/图像 adapter 输出拉齐到 Emotion2Vec 的语音情感空间 [公式 1]

$$Loss_{MPEE} = MSE(E_t, E_s) + MSE(E_i, E_s)$$

[agent 解读] 这种"锚定到语音情感空间"的设计是合理的 — TTS 的目标本就是生成语音,以语音情感表示为标准比以文本或图像为标准更直接。但 MSE loss 是逐维对齐,不考虑情感空间的结构 (如类间距离),可能导致不同情感的 embedding 过于接近。

#### 3. LLM-like Prosody Predictor

[论文原文] 架构 [§2.3]:
- 8 层 Transformer, 8 attention heads, 768 embedding dim
- 输入: content encoder 输出 + timbre embedding + emotion code
- 目标: 预测 VQ prosody code (teacher-forcing + cross-entropy loss)
- 额外约束: **Emotion Consistency Loss (ECL)** — 在 prosody predictor 后加分类器,识别预测韵律中的情感类别,用 cross-entropy loss 确保预测韵律保留情感信息

[论文原文] ECL 的动机: prosody 与 emotion 高度相关 (情感通过韵律变化表现),但 AR 预测过程可能丢失情感信息,ECL 作为显式约束防止这种丢失 [§2.3]

[agent 解读] ECL 本质上是一个辅助分类损失,类似于 GRL 的"正向"版本 — GRL 让 encoder 隐藏特定信息,ECL 让 predictor 保留特定信息。这比单纯的 cross-entropy 重建更强,因为重建目标 (VQ code) 可能编码了多种信息而不专注于情感。

#### 4. Diffusion-Based Acoustic Model

[论文原文] 标准配置 [§2.4]:
- Content encoder: 5 层 Conformer, 512 dim
- Timbre encoder: ECAPA-TDNN-like
- Duration predictor: 5 层 Conv-1D, 融合 content + timbre + prosody
- Diffusion decoder: U-Net, 512 hidden size (基于 Grad-TTS)
- Vocoder: 预训练 HiFi-GAN (16kHz LibriTTS)

### 训练策略

三阶段顺序训练 [§3.2]:
1. **MPEE 训练**: 在 MEAD-TTS 上训练 100 epochs (对齐 text/image adapter 到 Emotion2Vec 空间)
2. **AM 预训练 + 微调**: LibriTTS 上预训练 500K steps (获得基础生成能力) → MEAD-TTS 上微调 50 epochs (获得情感生成能力)。Prosody encoder codebook warmup 40K steps
3. **Prosody Predictor 训练**: 在 MEAD-TTS 上训练 50 epochs,使用 AM 中已训练好的 prosody encoder 提取 target prosody code

[agent 解读] 三阶段训练增加了工程复杂度但保证了各组件的稳定性。AM 先在大规模 LibriTTS 上学习基础能力再在小规模情感数据上微调,是数据有限时的标准做法。

## 实验

| 指标 | MPE-TTS (speech prompt) | MM-TTS | GenerSpeech | Meta-StyleSpeech | GT(mel) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOS | 3.73 | 3.55 | 3.53 | 3.43 | 4.05 | MEAD-TTS | [Table 1] |
| ESMOS | 4.05 | 3.75 | 3.63 | 3.45 | - | MEAD-TTS | [Table 1] |
| SSMOS | 3.73 | 3.55 | 3.65 | 3.53 | - | MEAD-TTS | [Table 1] |
| WER | 23.4% | 31.2% | 22.8% | 26.0% | 18.8% | MEAD-TTS | [Table 1] |
| EACC | 48% | 41% | 38% | 24% | 54% | MEAD-TTS | [Table 1] |
| MOS (text prompt) | 3.65 | 3.53 | - | - | 4.05 | MEAD-TTS | [Table 2] |
| EACC (text prompt) | 46% | 37% | - | - | 54% | MEAD-TTS | [Table 2] |
| MOS (image prompt) | 3.63 | 3.55 | - | - | 4.05 | MEAD-TTS | [Table 3] |
| ESMOS (image) | 3.90 | 3.53 | - | - | - | MEAD-TTS | [Table 3] |
| EACC (image) | 47% | 35% | - | - | 54% | MEAD-TTS | [Table 3] |

**消融实验** [Tables 1-3]:
- **w/o MPEE** (使用 MM-TTS 的 AMPE 代替): 所有指标显著下降,尤其 ESMOS (speech: 4.05→3.35),说明 Emotion2Vec-based MPEE 提取情感信息更准确
- **w/o ECL**: ESMOS 下降 (speech: 4.05→3.83),在 text/image prompt 场景下 ECL 的作用更显著 (text EACC: 46%→36%, image EACC: 47%→34%) [Tables 2-3],说明 ECL 对非语音模态的情感保持尤为重要

## 局限性

1. **数据规模极小**: MEAD-TTS 仅 36h/48 人/8 情感,与现代 ZS-TTS 系统 (100K+ 小时) 差距巨大,论文自述 "significant potential for improvement in scalability" [§4]
2. **EACC 天花板低**: GT mel 的 SER 识别率仅 54%,说明评估工具本身的情感分类能力有限,48% 的绝对数值难以直接解读
3. **Baseline 公平性存疑**: MM-TTS 无官方开源,作者自行复现,可能无法完全还原原始性能 [§3.4]
4. **未与现代系统对比**: 无 CosyVoice/Seed-TTS/MaskGCT 等当前 SOTA 的对比
5. **多模态融合未探索**: 论文仅支持"三选一"单模态输入,未探索多模态联合输入 [§4]
6. **仅 16kHz**: 采样率较低,限制生成质量
7. **WER 偏高**: 即使是 GT mel + HiFi-GAN 也有 18.8% WER,说明评估 pipeline 或数据质量有问题

## 点评

MPE-TTS 提出了一个完整的多模态情感零样本 TTS 系统,核心思路 — 用 Emotion2Vec 锚定跨模态情感空间 + 层级解耦 + ECL 韵律情感保持 — 在方法论上是清晰合理的。MPEE 的设计利用了 Emotion2Vec 作为强大的语音情感预训练模型,通过 MSE loss 将 CLIP 空间拉齐过来,是一种简洁有效的跨模态对齐方案。

但从 TTS 领域当前发展水平看,这项工作存在明显的规模和基线差距。MEAD-TTS 36h 的训练数据、48 个说话人、8 类情感的设定,与当前大规模零样本 TTS 的数据规模 (万小时级) 和开放域情感能力差距巨大。Baseline 仅包含 2022-2024 年的中小型系统 (Meta-StyleSpeech, GenerSpeech, MM-TTS),未与任何 LLM-based TTS 系统对比,使得结论的说服力受限。

ECL 是本文最具迁移价值的设计 — 在 prosody prediction 后通过分类器显式约束情感保持,尤其在非语音模态 (text/image) prompt 场景下效果显著。这一 trick 可以直接应用到其他需要在中间表示中保持特定属性的场景。

## 可复用的 idea

1. **Emotion2Vec 作为跨模态情感锚点**: 将 Emotion2Vec 的语音情感空间作为统一目标,用 MSE loss 对齐其他模态 (CLIP text/image) 的 adapter 输出。这种"选择一个强模态的预训练模型作为锚点"的对齐策略可推广到其他跨模态场景
2. **Emotion Consistency Loss (ECL)**: 在序列生成器 (如 prosody predictor) 后加属性分类器,用分类 loss 确保生成结果保留目标属性。本质上是一种"正向 GRL" — GRL 让 encoder 丢弃信息,ECL 让 generator 保留信息
3. **低频 mel bins 作为韵律信号**: 取 mel-spectrogram 的低 20 bins 作为 prosody encoder 输入,利用低频段包含韵律但削弱 timbre/content 的物理特性实现轻量解耦 (来自 ProsoSpeech)
4. **输入设计式解耦**: 通过控制每个 encoder "能看到什么" (timbre encoder 看不同句子、emotion encoder 看不同说话人的同情感语音) 实现解耦,不需要对抗训练,工程实现简单

> [!review] 审阅 (auto)
> 详见 [[_review/MPE-TTS-review.yml]]
