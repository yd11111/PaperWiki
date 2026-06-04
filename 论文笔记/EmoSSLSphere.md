---
type: paper
tier: deep
title: "EmoSSLSphere: Multilingual Emotional Speech Synthesis with Spherical Vectors and Discrete Speech Tokens"
arxiv_id: "2508.11273"
source: "Sources/EmoSSLSphere.pdf"
authors: [Joonyong Park, Kenichi Nakamura]
year: 2025
venue: "13th ISCA Speech Synthesis Workshop"
tags: [TTS, emotion, multilingual, self-supervised-learning, spherical-coordinates, discrete-tokens, prosody, HuBERT]
concepts: ["[[Emotion Control in TTS]]", "[[Self-Supervised Speech Representation]]", "[[Prosody Modeling]]", "[[Mel Spectrogram]]", "[[Speaker Embedding]]", "[[Non-autoregressive TTS]]"]
models: ["[[模型库/HuBERT]]"]
tasks: ["[[任务库/Cross-lingual Voice Cloning]]"]
datasets: ["ESD", "JVNV"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 3 个待确认实体页: [[Prosody Modeling]]$\checkmark$, [[Speaker Embedding]]$\checkmark$, [[Cross-lingual Voice Cloning]]$\checkmark$, [[Emotion Control in TTS]][待确认], [[Self-Supervised Speech Representation]][待确认], [[模型库/HuBERT|HuBERT]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Prosody Modeling]], [[Speaker Embedding]], [[Cross-lingual Voice Cloning]], [[Emotion Control in TTS]], [[Self-Supervised Speech Representation]], [[HuBERT]] | 过滤: 3 页 pending-review | 未命中但可能相关: [[Style Transfer in TTS]], [[Global Style Tokens]]

**谱系定位**: EmoSSLSphere 处于情感可控 TTS 演进线的 EmoSphere-TTS (Interspeech 2024) 之后,是其向多语言+SSL 特征方向的扩展。与 [[Emotion Control in TTS]][待确认] 中记录的演进线对照: EmoSphere-TTS 引入了 AVD 球面坐标变换实现风格/强度解耦; EmoSphere++ 进一步升级为 emotion-adaptive centroid + CFM decoder + 零样本能力。EmoSSLSphere 则沿另一条轴线扩展 -- 保留 EmoSphere-TTS 的球面情感编码器和 FastSpeech 2 decoder,但新增 SSL 离散 token (HuBERT) 和语义文本编码器 (DeBERTaV3),聚焦多语言场景。在 KB 中已有 [[论文笔记/EmoSphere-TTS|EmoSphere-TTS]] 和 [[论文笔记/EmoSphere++|EmoSphere++]] 两篇精读笔记。

**已有认知**: [[Self-Supervised Speech Representation]][待确认] 记录了 HuBERT 中间层 (8-9) 对超音段韵律特征 (stress/tone/accent) 的编码最强的发现 [de la Fuente & Jurafsky, 2024],这为 EmoSSLSphere 选择第 9 层特征提供了理论支撑。[[Prosody Modeling]]$\checkmark$ 涵盖了从显式韵律预测到 SSL 隐式韵律表征的完整方法谱。[[Speaker Embedding]]$\checkmark$ 说明了 speaker lookup table 和 speaker encoder 两种范式在多说话人 TTS 中的角色。

**创新判断**: 与 KB 已有工作相比,EmoSSLSphere 的贡献在于将球面情感向量与 SSL 离散 token 融合用于多语言情感 TTS。核心新颖性是:(1) 用语言分别聚类的 HuBERT k-means token 提供跨语言一致的韵律控制信号; (2) DeBERTaV3 语义编码器通过 cross-attention 条件化情感/韵律模块。但架构整体仍在 FastSpeech 2 + mel-spectrogram 范式内,未触及 LLM-TTS 或 CFM decoder 的新范式。

## 速查

> [!summary] 速查
> - **一句话**: 在 EmoSphere-TTS 基础上融合 HuBERT 离散韵律 token 和 DeBERTaV3 语义编码,实现英日双语情感可控 TTS
> - **路线**: 文本 → Phoneme Encoder (MFA/pyopenjtalk) + Semantic Encoder (DeBERTaV3, frozen) || 参考音频 → AVD Encoder → 球面坐标变换 → 球面情感向量 || 参考音频 → HuBERT (layer 9) → k-means (K=200) → 离散 token || Speaker Embedding → 四路 concat + 线性投影 → FastSpeech 2 Mel Decoder → Mel → Waveform
> - **指标**: EN: WER 19.58% / MCD 7.282 / nMOS 4.13+-0.08 vs EmoSphere-TTS WER 20.96% / MCD 7.979 / nMOS 4.05+-0.12; JA: CER 18.33% / MCD 8.719 / nMOS 3.94+-0.15 vs EmoSphere-TTS CER 19.26% / MCD 9.131 / nMOS 3.63+-0.16 [Table 1, 2]; AVD RMSE: EN 0.0773 vs 0.0798, JA 0.0783 vs 0.0867 [Table 3]
> - **可借鉴**: (1) 语言分别聚类的 HuBERT token 作为跨语言韵律控制信号,避免不同语言 phone 体系混淆 codebook; (2) DeBERTaV3 语义编码通过 cross-attention 条件化韵律模块而非直接作为 decoder 输入,间接增强语义-韵律对齐
> - **局限**: 仅单说话人数据 (EN 80 句 / JA 60 句); 无零样本能力; 未与 EmoSphere++ 或 LLM-TTS 对比; 代码未开源; 评估规模极小 (9/21 人主观评估)

## 核心问题

多语言情感 TTS 面临三重交叉挑战 [S1]:

1. **情感表达与语言特异性的耦合**: 不同语言的韵律模式差异显著 (如日语音高模式 vs 英语重音模式),同一情感在不同语言中的声学实现不同,导致直接跨语言迁移时出现不自然的语调和外国口音 [S1, S2]
2. **情感控制的精细度不足**: 前作 EmoSphere-TTS 的球面向量虽解耦了风格/强度,但缺乏对细粒度韵律模式 (pitch contour, rhythm pattern) 的建模能力,在多语言场景下这一不足被放大 [S1]
3. **语义-韵律一致性**: 情感韵律应与语义内容一致 (如强调句中关键词时 pitch 上升),但现有方法缺乏将语义信息注入韵律生成的机制 [S3.3]

**核心提问**: 能否在保留球面情感控制的直觉性的同时,通过 SSL 语音表征引入语言无关的韵律控制信号,实现多语言场景下的精细情感合成?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

EmoSSLSphere 由四个主模块组成,各自独立预训练后冻结,仅训练 Mel Decoder [S3, Fig 1-2]:

```
Text → Phoneme Encoder (MFA/pyopenjtalk) → phoneme embedding ─────────┐
     → Semantic Encoder (DeBERTaV3, frozen) → semantic embedding ──────┤
                                                                       │
Reference Audio → AVD Encoder → Cartesian→Spherical → spherical       │
                  emotion vector ─────────────────────────────────────┤
               → Emotion ID → one-hot embedding ─────────────────────┤
                                                                       │
Reference Audio → HuBERT (layer 9, frozen) → k-means (K=200)         │
                  → discrete token feature ───────────────────────────┤
                                                                       │
Speaker Embedding ─────────────────────────────────────────────────────┤
                                                                       ↓
                               Concat → Linear Projection → FastSpeech 2 Decoder → Mel
```

关键设计: 每种目标语言各自实例化一套编码器 (Emotional Encoder, SSL Feature Encoder, Text Encoder),避免跨语言干扰 [S4.1] [论文原文]。

### 关键设计选择

#### 1. 为什么用 HuBERT 第 9 层而非其他层?

[论文原文] 作者引用了分析研究表明 "lower-middle transformer layers encode rhythm and pitch trajectories reliably across languages",第 9 层在 12 层 base model 中提供了细粒度语音细节 (低层) 与高层语义 (高层) 之间的最佳平衡 [S3.2]。

[agent 解读] 这与 KB 中 [[Self-Supervised Speech Representation]][待确认] 记录的 de la Fuente & Jurafsky (2024) 发现完全一致 -- HuBERT 中间层 (8-9) 对超音段韵律特征的编码最强。但论文并未引用该分析工作,说明作者可能是独立得到类似结论的。

#### 2. 为什么对 SSL 特征做语言分别聚类?

[论文原文] 聚类在每种语言上分别进行 (separately for each language),因为不同语言的音素清单 (phone inventory) 差异会模糊共享 codebook。然而作者发现生成的 token 虽然是语言分别聚类的,实际上捕获了通用的韵律模式 (universal prosodic patterns),语言特异性较弱 [S3.2]。

[agent 解读] 这一设计选择暗含一个假设: HuBERT 第 9 层特征已经部分抽象掉了语言特异的音素信息,保留了更通用的韵律模式。语言分别聚类是为了确保 codebook 质量 (避免日语和英语的 phone inventory 在 k-means 空间中相互干扰),而非因为底层特征本身强烈语言相关。

#### 3. 语义编码器如何影响情感合成?

[论文原文] DeBERTaV3 语义编码器生成的 semantic embeddings "first used to condition the emotional and prosodic modules through cross-attention mechanisms",在线性投影后也作为 decoder 输入的一部分 concat。仅对 tokenizer 做语言适应,核心 encoder 保持冻结 [S3.3]。

[agent 解读] 语义编码器的角色是间接的 -- 不是直接生成韵律,而是通过 cross-attention 告诉情感/韵律模块"当前语义上下文是什么",从而实现语义感知的韵律生成。例如,当语义编码器检测到疑问句时,可以引导情感模块产生适当的升调。消融实验 (Table 2) 显示去掉语义编码器后 MCD 几乎不变但 nMOS 轻微下降 (但差异不大),说明其主要贡献在韵律自然度而非频谱精度。

#### 4. 球面情感编码器的设计 (沿用 EmoSphere-TTS)

[论文原文] 沿用 EmoSphere-TTS 的方案: AVD 向量从参考音频 Mel 谱提取,经笛卡尔→球面坐标变换 (r, theta, phi)。此外加入 Emotion ID 的 one-hot 嵌入作为类别先验,与球面 AVD 向量 concat 后共同编码 [S3.1]。

[agent 解读] 与 EmoSphere++ 的 emotion-adaptive centroid transform (EACT) 不同,EmoSSLSphere 似乎使用了 EmoSphere-TTS 原始的固定中性中心方案。论文未提及 EACT 或 max-ratio centroid 计算,这可能是一个有意的简化选择 (聚焦多语言扩展而非情感建模精度提升) 或者是与 EmoSphere++ 独立平行发展的结果。

#### 5. 推理时的控制方式

[论文原文] 推理需要两个输入: 目标文本和参考语音波形。参考波形提供情感和韵律线索,不需要与文本内容匹配但应反映期望的情感和韵律风格。说话人身份通过固定 speaker embedding 保持,独立于参考音频 [S3.5]。

[agent 解读] 这意味着参考音频实际承担了"情感指令"和"韵律模板"的双重角色。消融实验中 content mismatch (Table 1, 2) 对性能影响相对较小,说明系统能较好地分离内容与风格/韵律信息。

### 训练策略

**两阶段训练** [S3.4]:
1. **Pre-training**: 三个编码器独立预训练:
   - Emotional Encoder: 按 EmoSphere-TTS 方案在情感语料上训练
   - SSL Feature Encoder: HuBERT 预训练特征 + 语言分别 k-means (K=200)
   - Semantic Text Encoder: 冻结 DeBERTaV3 (EN: base; JA: Wikipedia+Aozora Bunko 微调版)
2. **Fine-tuning**: 冻结所有编码器,仅训练 FastSpeech 2 Mel Decoder。输入 = concat(phoneme, spherical emotion, emotion ID, discrete tokens, speaker embedding, semantic embedding) → linear projection → decoder

**数据** [S4.2]:
- English: ESD 数据集 80 句 (4 情感 x 20 句, 单女性说话人) [S4.2]
- Japanese: JVNV 数据集 60 句 (6 情感 x 10 句, 单女性说话人) [S4.2]
- 推理参考: 同说话人同语言的 GT 语音 [S4.2]

## 实验

### 语音可理解性 (Table 1)

| 指标 | EmoSSLSphere | EmoSphere-TTS | NATSpeech w/ EL | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER (EN, %) ↓ | 19.58 | 20.96 | 21.50 | 13.65 | ESD | [Table 1] |
| SpeechBERTScore (EN) ↑ | 0.880 | 0.875 | 0.870 | - | ESD | [Table 1] |
| SpeechBLEU (EN) ↑ | 0.225 | 0.209 | 0.200 | - | ESD | [Table 1] |
| CER (JA, %) ↓ | 18.33 | 19.26 | 20.30 | 16.37 | JVNV | [Table 1] |

### 语音质量与自然度 (Table 2)

| 指标 | EmoSSLSphere | EmoSphere-TTS | NATSpeech w/ EL | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MCD (EN) ↓ | 7.282 | 7.979 | 8.060 | - | ESD | [Table 2] |
| LogF0RMSE (EN) ↓ | 0.341 | 0.359 | 0.357 | - | ESD | [Table 2] |
| UTMOSv2 (EN) ↑ | 2.427 | 2.403 | 2.348 | - | ESD | [Table 2] |
| nMOS (EN) ↑ | 4.13+-0.08 | - | - | 4.30+-0.09 | ESD | [Table 2] |
| MCD (JA) ↓ | 8.719 | 9.131 | 9.287 | - | JVNV | [Table 2] |
| nMOS (JA) ↑ | 3.94+-0.15 | - | - | 4.40+-0.12 | JVNV | [Table 2] |

### 情感表现力 (Table 3)

| 指标 | EmoSSLSphere | EmoSphere-TTS | NATSpeech w/ EL | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| AVD RMSE Avg (EN) ↓ | 0.0773 | 0.0798 | 0.0938 | ESD | [Table 3] |
| AVD RMSE Avg (JA) ↓ | 0.0783 | 0.0867 | 0.0936 | JVNV | [Table 3] |

**值得注意**: EmoSSLSphere 在日语上的改进幅度大于英语 (JA AVD RMSE: -9.7% vs EN: -3.1%),暗示 SSL 离散 token 对情感韵律建模的增益在跨语言场景中更显著 [agent 解读]。

### 消融实验 (Table 1, 2)

| 消融条件 | WER (EN) | MCD (EN) | nMOS (EN) | CER (JA) | MCD (JA) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Full model | 19.58 | 7.282 | 4.13 | 18.33 | 8.719 | [Table 1,2] |
| w/o Reference waveform | 20.37 | 7.732 | - | 19.18 | 9.134 | [Table 1,2] |
| Ref content mismatch | 20.31 | 7.681 | - | 19.11 | 9.081 | [Table 1,2] |
| Ref emotion mismatch | 19.83 | 7.405 | - | 18.79 | 8.997 | [Table 1,2] |
| Ref speaker mismatch | 19.69 | 7.523 | - | 18.42 | 8.846 | [Table 1,2] |
| w/o Semantic Text Encoder | 19.72 | 7.293 | - | 18.46 | 8.809 | [Table 1,2] |
| w/o k-means discretization | 20.12 | 7.418 | - | 18.89 | 8.812 | [Table 1,2] |

**消融关键发现** [S5.6]:
- **参考波形移除**影响最大: WER 从 19.58→20.37, MCD 从 7.282→7.732,证实 SSL-derived token 的韵律条件化对韵律保真度至关重要 [S5.6] [论文原文]
- **speaker mismatch 影响最小**: 某些指标甚至略优于完整模型 (JA CER 18.42 vs 18.33),说明韵律建模具有说话人无关性 [S5.6] [论文原文]
- **k-means 离散化 vs 连续特征**: 去掉离散化使用连续 HuBERT 特征后性能下降 (WER 20.12, MCD 7.418),表明离散 token 提供了更稳定的韵律/音段对齐 [S5.6] [论文原文]
- **语义编码器**: 影响较小但一致 (WER 19.72, MCD 7.293),主要贡献在韵律一致性而非频谱精度 [S5.6] [论文原文]

### 共振峰分析 (Fig 5)

[论文原文] 对 F1/F2/F3 共振峰在不同情感下的变化分析显示 [S5.5]:
- F1 (元音开口度): EmoSSLSphere 在高唤起情感下产生更高 F1,与 GT 一致
- F2 (舌位): 日语中差异较小,EmoSSLSphere 与 GT 轨迹接近
- F3 (音色): 不同情感间保持清晰的音色区分

## 局限性

1. **极小评估规模**: EN 仅 80 句 (4 情感 x 20 句)、JA 仅 60 句 (6 情感 x 10 句),均为单说话人 [S4.2]。这使统计显著性存疑,且完全无法评估多说话人泛化能力。作者也在结论中承认这一局限 [S6]
2. **无零样本能力**: 推理时要求参考语音与目标说话人匹配,不支持 unseen speaker [S3.5]。而同系列的 EmoSphere++ 已实现零样本情感 TTS [agent 解读]
3. **未与 EmoSphere++ 对比**: EmoSphere++ (arXiv 2024.11) 比本文 (arXiv 2025.08) 更早发表,但未在 baseline 中出现。考虑到 EmoSphere++ 在 ESD 上的 ECA 93.53%、nMOS 3.92,缺少这一对比使本文贡献的增量价值难以准确评估 [agent 解读]
4. **架构保守**: 仍基于 FastSpeech 2 mel-spectrogram 范式,未采用 CFM/Diffusion decoder 或 LLM-based 生成。在 KB 中 [[Emotion Control in TTS]] 的演进线上,这一架构已被后续工作 (TTS-CtrlNet, UDDETTS) 超越 [agent 解读]
5. **nMOS 未提供 baseline 数据**: Table 2 的 nMOS 仅报告了 proposed method 和 GT,EmoSphere-TTS 和 NATSpeech 的 nMOS 数据以脚注形式出现 (EN: 4.05, JA: 3.63),未提供置信区间。Baseline 只进行了单次 MOS 评估 [agent 解读]
6. **语义编码器贡献存疑**: 消融显示去掉语义编码器的影响很小 (MCD EN 7.293 vs 7.282),其跨语言价值未被充分验证 [Table 2]
7. **无跨语言情感迁移实验**: 虽然标题含 "Multilingual",但未做 cross-lingual emotion transfer (如用英语参考的情感合成日语),所有实验均在同语言内进行 [S4.2]

## 点评

**优点**:
- 将 SSL 离散 token 引入球面情感 TTS 是合理且有效的扩展方向。HuBERT k-means token 提供了与球面 AVD 向量互补的控制信号 -- AVD 管全局情感方向/强度,token 管局部韵律节奏,两者分工清晰 [论文原文]
- 语言分别聚类的设计选择符合 SSL 表征的特性 (中间层编码通用韵律但有语言特异的 phone 分布),同时论文发现 token 实际上是弱语言相关的,这一观察本身有价值 [S3.2]
- 消融实验设计全面,特别是 reference waveform 的四种 mismatch 条件 (content/emotion/speaker/全部移除) 提供了系统的诊断信息 [S5.6]

**不足**:
- 评估规模 (80+60 句,单说话人) 是最大弱点,严重限制了结论的可推广性。与 EmoSphere-TTS (17,500 句, 10 说话人) 和 EmoSphere++ (同规模) 相比,EmoSSLSphere 的训练和评估数据都小一个数量级
- 与前作 EmoSphere++ 的定位关系不清: 两者都是 EmoSphere-TTS 的扩展但方向不同 (EmoSphere++ 升级情感建模精度+零样本; EmoSSLSphere 升级韵律建模+多语言)。缺少 EmoSphere++ 作为 baseline 使读者无法判断哪个方向更有价值
- "Multilingual" 的 claim 过强: 实际只是"双语",且无跨语言迁移实验。真正的 multilingual 应展示语言间的知识共享或迁移效果
- 在 KB 的情感 TTS 演进线中,EmoSSLSphere 的贡献是增量性的 -- 将已有组件 (球面 AVD + HuBERT k-means + DeBERTaV3) 组合应用于多语言场景,但未在任何单一维度上实现突破

## 可复用的 idea

1. **语言分别聚类的 SSL 韵律 token**: 对不同语言分别训练 k-means codebook 以避免 phone inventory 混淆,但利用 SSL 中间层的语言不变韵律编码实现跨语言一致的韵律控制。可迁移到任何需要多语言韵律建模的系统
2. **球面情感向量 + SSL 韵律 token 的双控制范式**: 全局情感方向/强度 (球面 AVD) + 局部韵律细节 (离散 token) 的组合控制思路,可推广到需要同时进行宏观和微观韵律控制的系统
3. **语义编码器通过 cross-attention 间接条件化**: DeBERTaV3 不直接作为 decoder 输入而是通过 cross-attention 条件化韵律模块的设计,使语义信息以"建议"而非"命令"的方式影响韵律生成,避免过度耦合

> [!review] 审阅 (2026-06-04, agent)
> **结论**: pass-with-fixes | 3 issues (0 high, 2 medium, 1 low)
> - **medium/template-compliance**: frontmatter models 字段未列 baseline 模型 (NATSpeech),但 NATSpeech 无独立模型页,边界情况
> - **medium/traceability-gap**: S3.3 语义编码器的 cross-attention 机制细节在原文中描述简略 ("used to condition... through cross-attention mechanisms"),笔记中的推断已标注 [agent 解读]
> - **low/weak-reusability**: 可借鉴第 3 点 (cross-attention 间接条件化) 是通用设计模式,具体实现细节不足
> 详见 `_review/EmoSSLSphere-review.yml`

---

检索命中: [[Prosody Modeling]]$\checkmark$, [[Speaker Embedding]]$\checkmark$, [[Cross-lingual Voice Cloning]]$\checkmark$, [[Emotion Control in TTS]](pending-review), [[Self-Supervised Speech Representation]](pending-review), [[HuBERT]](pending-review) | 过滤: 无 | 未命中但可能相关: [[Style Transfer in TTS]], [[Global Style Tokens]]
