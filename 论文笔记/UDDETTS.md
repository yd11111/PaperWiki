---
type: paper
tier: deep
title: "UDDETTS: Unifying Discrete and Dimensional Emotions for Controllable Emotional Text-to-Speech"
arxiv_id: "2505.10599"
source: "Sources/UDDETTS.pdf"
authors: [Jiaxuan Liu, Yang Xiang, Han Zhao, Xiangang Li, Yingying Gao, Shilei Zhang, Zhenhua Ling]
year: 2025
venue: "Preprint"
tags: [TTS, emotion, LLM-based, flow-matching, controllable, semi-supervised, ADV-space]
concepts: ["[[EmotionControlinTTS]]", "[[ConditionalFlowMatching]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[FiniteScalarQuantization]]", "[[ProsodyModeling]]", "[[SpeakerEmbedding]]"]
models: ["[[CosyVoice3]]", "[[CosyVoice]]", "[[CosyVoice2]]", "[[MinMo]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[EmotionControlinTTS]][待确认], [[FiniteScalarQuantization]][待确认], [[CosyVoice3]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: UDDETTS 位于 LLM-based Emotional TTS 的交叉点上。在 [[LLM-basedTTS]] 的演进中,情感控制一直是薄弱环节 — 现有 LLM-TTS (CosyVoice, Spark-TTS, VALL-E 等) 主要依赖离散标签提示,无法捕捉情感的连续性和维度解耦。[[EmotionControlinTTS]] 概念页 [待确认] 记录了从 emotion embedding → 层级建模 → 对抗解耦 → DPO/RLHF → 球面空间(EmoSphere-TTS/EmoSphere++) 的演进线。UDDETTS 的直接前驱是 EmoSphere-TTS/EmoSphere++,它们首次在 non-LLM TTS 中引入 ADV 空间,但使用笛卡尔→球面坐标变换,存在情感簇扭曲和重叠问题。UDDETTS 的核心创新在于: (1) 在 LLM-TTS 中首次引入 ADV 空间并用非线性分箱替代球面变换; (2) 用半监督训练统一异构情感标注数据集。

**已有认知**: [[ConditionalFlowMatching]]✓ 记录了 OT-CFM 在 TTS 中作为 "fine stage 渲染器" 的典型用法 (CosyVoice 系列使用 CFM 将离散 token → mel spectrogram)。[[SpeechTokenizer]]✓ 记录了监督式 semantic tokenizer 的设计 (CosyVoice 3 的 FSQ-MinMo)。[[FiniteScalarQuantization]][待确认] 记录了 FSQ 无码本、100% 利用率的优势。UDDETTS 的 speech tokenizer 直接受 [[CosyVoice3]][待确认] 启发,在 MinMo 的 FSQ 模块基础上增加了情感识别 (SELR) 和 ADV 识别 (SADVR) 多任务训练,使 token 富含情感副语言信息。

**创新判断**: UDDETTS 相对 KB 已有知识的新增贡献在于: (a) ADV 空间的非线性分箱量化方案 (基于聚类的自适应分箱,解决情感分布不均衡); (b) 半监督训练框架统一 4 种标注类型的数据集; (c) OT-CFM 中的 emotional mixture encoder (融合 ADV 和 label 条件)。这些在 KB 中均无对应记录。

> 检索命中: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓ | 过滤: [[EmotionControlinTTS]](pending-review), [[FiniteScalarQuantization]](pending-review), [[CosyVoice3]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个在 LLM-TTS 中引入 Arousal-Dominance-Valence (ADV) 空间的框架,通过非线性分箱 + 半监督训练统一离散标签与维度情感,实现三维解耦的可解释情感控制
> - **路线**: Text + (ADV tokens / Label / 无) → Neural Codec LM (0.70B, Transformer AR) → semantic tokens + label → OT-CFM (0.35B, U-Net + emotional mixture encoder) → mel → HiFi-GAN → waveform
> - **指标**: Label-controlled MOS 4.29, ES 0.833, WER 2.40% [Table 1]; ADV-controlled SRC 0.92/0.85/0.78 (V/A/D) [Table 2]; 非线性分箱 ADV 覆盖率 77.89% → 半监督后 89.35% [Fig 4]; 端到端偏好率 67.33% vs CosyVoice2 [Table 3]
> - **可借鉴**: (1) 非线性分箱: 对任意不均衡连续空间做聚类→自适应分箱→CLT 选 bin 数,可泛化到 pitch/energy 等其他连续属性控制; (2) 半监督 label↔ADV 知识迁移策略 — 用有 ADV 标注的小数据集带动仅有 label 的大数据集; (3) emotional mixture encoder 中 label-ADV 注意力融合 + 门控的条件注入方式
> - **局限**: ADV 标注主观性大,标注者间不一致影响线性控制; ADV predictor 对情感模糊文本表现差; 缺乏多模态/对话上下文建模; 仅支持英语; 训练需 24×A800 GPU

## 核心问题

UDDETTS 要解决的核心问题是: **LLM-based TTS 中的情感控制仍然停留在离散标签层面,无法实现连续、细粒度、可解释的情感维度控制。** 具体来说有三个子问题:

1. **离散标签表达力不足**: 现有 LLM-TTS 用 "happy/sad/angry" 等预定义标签控制情感,每个类别只能生成该类别的"平均表达",无法捕捉情感的连续分布和微妙变化 [§1]
2. **ADV 标注稀缺且分布不均**: 大规模情感语音数据集多数只有离散标签,同时具备 ADV 维度标注的数据集很少,且 ADV 空间中中性区域过度表示、极端情感区域数据稀疏 [§1]
3. **异构数据集的统一利用**: 不同数据集有不同标注类型 (仅 label / label+ADV / 自发 vs 诱发),如何在统一框架中利用所有数据是开放挑战 [§3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UDDETTS 由三个主要模块组成 [§3, Fig 1]:

1. **Neural Codec Language Model** (0.70B, Transformer): 自回归生成 speech semantic tokens。输入序列包含 text tokens (BPE + Conformer text encoder), speaker embedding (3D-Speaker voiceprint 的平均中性语音 embedding), ADV tokens (3 个量化 token, 可 mask), label token (情感类别 token, 可 mask)。输出序列为 label token + semantic tokens + EOS [Eq. 1-3]
2. **OT-CFM Module** (0.35B, U-Net): 将 semantic tokens 重建为 mel spectrogram,条件包括 speaker embedding、semantic embedding (Conformer encoder 编码)、emotion conditions (emotional mixture encoder 生成) [§3.2]
3. **HiFi-GAN Vocoder**: mel → waveform,在情感数据集上 fine-tune 5 epochs [§4.2]

**Speech Tokenizer**: 基于 MinMo 模型,在 encoder 中插入 FSQ 模块,联合训练 ASR + SELR (speech emotion label recognition) + SADVR (speech ADV recognition) 三个任务 [§3.1.1, Fig 2]。[论文原文] 作者指出这一设计受 CosyVoice 3 启发,目的是让 semantic token 富含情感副语言信息。

### 关键设计选择

#### 1. 数据集分类与序列设计

[论文原文] 作者将所有数据集分为两大类 [§3]:
- **自发情感 (DS)**: 自然场景录制 (对话、演讲),文本情感与语音情感通常一致,LLM 可从文本学到情感映射
- **诱发情感 (DE)**: 演员用相同文本表达不同预设情感,文本情感与语音情感不匹配,必须依赖 ADV/label 指导

再按标注类型分 4 种: DS,AL (有 label+ADV)、DS,L (仅 label)、DE,AL (有 label+ADV)、DE,L (仅 label)。

**为什么这样分?** [论文原文] 因为 DS 中文本可预测情感(text→label 映射有意义),而 DE 中文本无法预测情感(同一文本可对应多种情感) [§3]。这直接决定了训练时 label loss 是否需要 mask: DS,L 的 label 不 mask (文本可预测),DE,L 的 label 必须 mask (文本不可预测) [Eq. 2-3, §3.1.4]。

#### 2. ADV 非线性分箱 (Nonlinear Binning)

[论文原文] 将连续 ADV 值 (归一化到 [1,7]) 量化为离散 token xadv = [xa, xd, xv] ∈ Z³[1,m],选用 m=14 bins [§3.1.2]。

**为什么不用线性等距分箱?** [论文原文] 因为 ADV 值在三个维度上呈近正态分布 [Appendix D],线性分箱导致密集区域过度分辨、稀疏区域严重欠表示。作者探索了多种非线性分箱算法,最终选择**聚类分箱** (clustering-based binning): 先对样本做 k-means 聚类,以聚类中心为依据划分 bin 边界,再用方差比调整边界位置 [Table 6]。

**bin 数量如何确定?** [论文原文] 使用中心极限定理 (CLT) 确定最大 bin 数 Kmax ≤ ⌊³√N⌋,通过 silhouette score 搜索最优 K [Table 6]。

**效果**: 非线性分箱将 ADV 空间覆盖率从 60.83% 提升到 77.89% [Fig 4]。

#### 3. 半监督训练策略

[论文原文] 核心是动态 masking [§3.1.4]:

| 数据来源 | 输入 ADV | 输出 label | 原因 |
| --- | --- | --- | --- |
| DS/DE,AL | 真实 ADV | 不 mask | ADV 可预测 label |
| DS,L | mask (xign) | 不 mask | 文本可预测 label |
| DE,L | mask (xign) | mask | 文本无法预测 label |
| label=Unknown | 任意 | mask | 情感模糊 |

**位置感知加权损失**: label token 位置的 loss 权重 wemo=5.0 (当 label 有效时),加速情感收敛 [Eq. 5-6]。同时使用 label smoothing (ε) 防止过拟合 [Eq. 6]。

**效果**: 半监督训练将 ADV 覆盖率从 77.89% 进一步提升到 89.35%,且可在完全无训练样本的 ADV 区域合成合理语音 (如 [14,1,1] 处合成啜泣式语音) [§4.4, Fig 4]。[论文原文] 作者认为半监督训练促进了 label 知识向 ADV 空间的迁移。

#### 4. ADV Predictor

[论文原文] 仅用文本预测情感时,LLM 倾向于生成中性语音 [§3.1.3]。因此引入 ADV predictor: 用 RoBERTa encoder 从文本预测 pseudo-ADV 值,量化后作为 LLM 输入的中间 token。

**损失函数** [Eq. 4]: MSE loss (三维分别) + bin center distance loss (将预测值拉向真实 ADV 所在 bin 的中心)。[agent 解读] 第二项相当于对量化后的离散目标做近似,使预测值不仅在连续空间接近真值,还在量化后落入正确的 bin。

#### 5. Emotional Mixture Encoder (OT-CFM 中的情感条件)

[论文原文] OT-CFM 需要情感条件 Eemo 来指导 mel spectrogram 生成 [§3.2, Fig 3]:

1. ADV encoder 分别编码 xa, xd, xv → 拼接 → 交互层 → Eadv
2. Label encoder: xlbl → Elbl
3. Multi-head attention: Elbl (query) 与 Eadv (key/value) → Eemo_attn
4. 半监督门控 [Eq. 7]: 根据 label 和 ADV 的可用性,三路选择输出 Eemo

[agent 解读] 这种设计让 OT-CFM 在不同推理模式下都能获得最优的情感条件: 仅有 label 时依赖 label embedding; label+ADV 均有时通过注意力融合两者; label=Unknown 时退回纯 ADV 条件。

### 训练策略

- **第一阶段**: 在 49400+ 小时无情感标注的通用英语语音上预训练 LLM (0.70B) + OT-CFM (0.35B),lr=1e-3, warmup 5000 steps, 15 epochs [§4.2]
- **第二阶段**: 在 ~551 小时情感语音数据集上半监督微调,text encoder 冻结,lr=1e-4, warmup 2500 steps, 30 epochs [§4.2]
- **Speech tokenizer**: 在完整训练集上预训练,500K steps 收敛 [§4.2]
- **Vocoder**: HiFi-GAN 在情感数据集上微调 5 epochs [§4.2]
- **硬件**: 24× NVIDIA A800-80GB, Adam, gradient accumulation 2, max frame length 5000/batch [§4.2]

## 实验

### Label-Controlled TTS [Table 1]

| 指标 | UDDETTS | CosyVoice3 | IndexTTS2 | CosyVoice2 | F5-TTS | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS↑ | 4.29±0.12 | 4.35±0.10 | 4.29±0.10 | 4.20±0.10 | 4.18±0.07 | [Table 1] |
| Pm (macro-Precision)↑ | 0.94 | 0.85 | 0.87 | 0.85 | 0.88 | [Table 1] |
| Rm (macro-Recall)↑ | 0.90 | 0.82 | 0.80 | 0.75 | 0.78 | [Table 1] |
| ES (Emotion Similarity)↑ | 0.833 | 0.790 | 0.778 | 0.720 | 0.709 | [Table 1] |
| WER↓ | 2.40% | 1.45% | 1.69% | 2.42% | 1.82% | [Table 1] |
| SS (Speaker Similarity)↑ | 0.702 | 0.784 | 0.792 | 0.733 | 0.723 | [Table 1] |
| UTMOS↑ | 4.25 | 4.48 | 4.20 | 4.10 | 4.30 | [Table 1] |

**注意**: UDDETTS 在情感控制指标 (Pm/Rm/ES) 上全面领先,但在自然度 (MOS/UTMOS) 和 speaker similarity (SS) 上低于 CosyVoice3/IndexTTS2。CosyVoice3 和 FireRedTTS2 未在情感数据上微调 (代码不开源),直接用预训练 checkpoint 推理 [Appendix B]。

### ADV-Controlled TTS [Table 2]

| 维度 | ADV 设置 | SRC (非线性) | KW (非线性) | SRC (线性) | KW (线性) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Arousal | [1-14, 7, 7] | 0.85 | 0.70 | 0.52 | 0.48 | [Table 2] |
| Dominance | [14, 1-14, 1] | 0.78 | 0.68 | 0.48 | 0.50 | [Table 2] |
| Valence | [14, 14, 1-14] | 0.92 | 0.83 | 0.57 | 0.58 | [Table 2] |

SRC 接近 1.0 表明感知情感与 ADV 值线性相关 [§4.4]。KW > 0.6 表示强评估者间一致性 [§4.4]。非线性分箱全面碾压线性分箱。

### End-to-End TTS [Table 3]

| 对比对 | UDDETTS | Similar | Baseline | p-value | 出处 |
| --- | --- | --- | --- | --- | --- |
| vs CosyVoice2 | 67.33% | 19.45% | 13.22% | 0.001 | [Table 3] |
| vs IndexTTS2 | 58.60% | 29.23% | 12.17% | 0.012 | [Table 3] |
| w/o ADV pred. vs CosyVoice2 | 46.88% | 24.30% | 28.82% | 0.035 | [Table 3] |

去掉 ADV predictor 后偏好率显著下降,表明 pseudo-ADV 对端到端情感合成至关重要 [§4.6]。

### 消融实验 [§4.6]

| 消融 | 影响 | 出处 |
| --- | --- | --- |
| 去掉 ADV predictor | 端到端偏好率下降,语音偏中性 | [Table 3] |
| 去掉 emotional mixture encoder (w/o EME) | ES 从 0.833 降至 0.820, MOS 从 4.29 降至 4.20 | [Table 1 最后一行] |
| 非线性→线性分箱 | SRC/KW 大幅下降 (Valence 0.92→0.57) | [Table 2 右侧] |
| 仅用 DS,AL (无半监督) | ADV 覆盖率降至 70%,无法合成 [14,1,1] 等未见情感 | [§4.6] |

## 局限性

1. **ADV 标注质量依赖**: 标注者间主观差异导致 ADV 标签不一致,直接影响线性控制精度 [§5]
2. **ADV predictor 对情感模糊文本不佳**: 同一文本在不同上下文可表达不同情感,纯文本无法消歧 [§5]
3. **无多模态上下文**: 缺乏对话历史、视觉等多模态信息的利用 [§5]
4. **仅英语**: 所有数据集和评估均为英语,未验证跨语言泛化
5. **speaker similarity 偏低**: SS 0.702 低于 CosyVoice3 (0.784) 和 IndexTTS2 (0.792) [Table 1],情感控制与音色保持存在 trade-off
6. **硬 ADV 区域质量下降**: 远离所有训练分布的 ADV 值 (如 [1,14,14], [1,14,7]) MOS 降至 3.56-3.60, UTMOS 降至 3.20-3.43 [Table 9]

## 点评

**优势**:
- ADV 空间 + 非线性分箱的组合是一个优雅的解决方案: 既保持了维度可解释性,又解决了数据不均衡问题。特别是 SRC 高达 0.85-0.92 的线性控制能力,表明 ADV 维度确实被有效解耦 [Table 2]
- 半监督训练框架的设计非常务实 — 认识到大多数情感数据集只有离散标签,通过动态 masking 最大化利用异构数据,ADV 覆盖率从 60.83% 提升到 89.35% 是显著进步 [Fig 4]
- 三种推理模式 (label/ADV/end-to-end) 覆盖了不同应用场景的需求

**不足**:
- 评估中 CosyVoice3 和 FireRedTTS2 未在情感数据上微调 (训练代码不开源),导致这些 baseline 在情感指标上处于不公平劣势 [Appendix B]。更公平的对比应仅限于可微调的模型
- MOS 4.29 虽然不低,但低于 CosyVoice3 (4.35) 和 IndexTTS2 (4.29 持平),说明情感控制增强是以一定自然度和音色保持为代价的
- 主观评估仅 12 名参与者,样本量偏小
- 论文未分析 ADV predictor 的预测精度 (预测 ADV 与真实 ADV 的误差分布),仅通过下游偏好测试间接验证

## 可复用的 idea

1. **非线性分箱方法**: 对任意不均衡分布的连续控制维度 (pitch range, energy, speaking rate 等),可用聚类→自适应分箱→CLT 选 bin 数的通用流程,替代简单的线性等距量化
2. **半监督异构数据统一**: 当不同数据集有不同粒度的标注 (如部分有细粒度标注、部分仅有粗粒度标签),可通过动态 masking + position-aware loss weighting 在统一模型中同时利用
3. **Emotional mixture encoder 的门控融合**: label 作 query、ADV 作 key-value 的注意力融合 + 三路门控,是一种灵活的多条件注入方式,可泛化到其他多条件生成场景
4. **数据集分类策略 (spontaneous vs elicited)**: 根据文本-语音情感一致性决定训练信号的 masking 策略,这种数据感知的训练设计思路可迁移到其他 multi-source 训练场景

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes (2 low issues, 0 high/medium)
> - [low] frontmatter models 补充 CosyVoice 2 (已修正)
> - [low] 消融 w/o EME 描述改为明确的对比数字 (已修正)
> 详见 `_review/UDDETTS-review.yml`
