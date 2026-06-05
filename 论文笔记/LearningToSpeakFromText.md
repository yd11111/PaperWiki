---
type: paper
tier: deep
title: "Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining"
arxiv_id: "2301.12596"
source: "Sources/LearningToSpeakFromText.pdf"
authors: [Takaaki Saeki, Soumi Maiti, Xinjian Li, Shinji Watanabe, Shinnosuke Takamichi, Hiroshi Saruwatari]
year: 2023
venue: "IJCAI 2023"
tags: [TTS, zero-shot, multilingual, cross-lingual, pretraining, masked-language-model, low-resource]
concepts: ["[[PhonemeRepresentation]]", "[[SpeakerEmbedding]]", "[[Attention-basedTTS]]", "[[MelSpectrogram]]", "[[Text-to-SpeechPipeline]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[Zero-shotSpeechSynthesis]], [[Cross-lingualVoiceCloning]], [[SpeakerEmbedding]]; 3 个待确认页: [[PhonemeRepresentation]], [[Attention-basedTTS]], [[MelSpectrogram]])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文发表于 2023 年初,处于 LLM-TTS 范式兴起之前的 Transformer TTS 时代。当时主流零样本 TTS (如 VALL-E, 2023 年 1 月同期) 依赖大量配对数据,且多聚焦于说话人泛化而非语言泛化。多语言 TTS 的标准做法是使用 IPA 或 bytes 作为跨语言共享 token,但仍需目标语言的配对数据。本文的独特定位在于: **只用目标语言的纯文本数据 (无语音)** 实现零样本跨语言 TTS,填补了"无语音低资源语言"这一 gap。

**已有认知**:
- [[Zero-shotSpeechSynthesis]]: 当前 SOTA 方案 (CosyVoice 3, Seed-TTS 等) 通过 LLM + 离散 token 实现零样本说话人泛化,但它们的"zero-shot"指说话人维度,本文的"zero-shot"指语言维度 — 二者正交。
- [[Cross-lingualVoiceCloning]]: 现代跨语言方案 (CosyVoice 3, X-Voice) 通过多语言大规模数据训练自然获得跨语言能力,而本文在极低资源场景 (只有文本) 下探索跨语言迁移。
- [[SpeakerEmbedding]]: 本文使用 x-vector 作为说话人表征,属于 speaker encoder 范式的早期标准选择。
- [[PhonemeRepresentation]] [待确认]: 本文核心 contribution 之一是在无 G2P 的 byte 表示下,通过 MLM 预训练达到甚至超过 IPA baseline 的性能。
- [[Attention-basedTTS]] [待确认]: 本文基于 Transformer TTS (Li et al., 2019),6 层 encoder + 6 层 decoder 的标准自回归架构。

**创新判断**: 本文的核心创新在于将 NLP 的多语言 MLM 预训练 (类似 mBERT) 迁移到 TTS 领域,通过 frozen language-aware embedding layer 实现跨语言的发音和韵律迁移。这一思路在 2023 年是新颖的,后续被大规模多语言 TTS 系统部分吸收。

> 检索命中: [[Zero-shotSpeechSynthesis]]✓, [[Cross-lingualVoiceCloning]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[PhonemeRepresentation]](pending-review), [[Attention-basedTTS]](pending-review), [[MelSpectrogram]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用多语言纯文本做 MLM 预训练 + 冻结 language-aware embedding,实现只需文本 (无语音) 的零样本多语言 TTS
> - **路线**: 多语言文本 → MLM 预训练 (token embed + language embed + bottleneck + encoder) → 少量配对数据微调 (冻结 language-aware embedding,训练 encoder + decoder) → 推理时输入未见语言文本 + language ID → mel → HiFi-GAN
> - **指标**: 未见语言 (西班牙语) CER 18.27% (bytes) / 11.69% (IPA) [Table 3],已见语言 byte 模型全面超 IPA baseline [Table 2];MOS 3.44 (bytes zero-shot) vs 3.29 (baseline) [Fig 5]
> - **可借鉴**: (1) 冻结 language-aware embedding 的跨语言迁移策略; (2) 纯文本 MLM 预训练提升 byte-based TTS 的思路; (3) bottleneck layer 对跨语言泛化的重要性
> - **局限**: 仅在 7+1 欧洲语言上验证,语种跨度有限; 模型规模小 (6 层 Transformer); 与 oracle 仍有明显 gap; 语言依赖性强 (相似语种效果好,差异大的语种改善有限)

## 核心问题

**动机**: 现有多语言 TTS 依赖每种语言的配对语音-文本数据,但世界上 6000+ 语言中大多数没有高质量语音数据。能否仅用文本数据就为新语言构建 TTS 系统?

**核心假设**: 多语言 BERT 在 NLP 任务上展现了强跨语言迁移能力,类似的 MLM 预训练策略能否迁移到 TTS — 让模型从文本中学习跨语言的发音和韵律知识?

**关键创新**:
1. 将多语言 MLM 预训练引入 TTS: 在多语言文本上做 masked language model 预训练,学习跨语言的 token 表征
2. Language-aware embedding layer: 设计 token embedding + language embedding + bottleneck 三层结构,预训练后冻结,保留跨语言迁移能力
3. Text-seen zero-shot TTS: 提出新的零样本定义 — 目标语言仅出现在文本预训练中,不出现在配对训练数据中

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

基于 Transformer TTS [Li et al., 2019b] 的自回归 mel-spectrogram 生成模型,由三部分组成 [§2]:

1. **Language-aware embedding layer**: token embedding + language embedding + bottleneck layer
2. **Encoder**: 6 层 Transformer encoder
3. **Decoder**: 6 层 Transformer decoder + 5 层 CNN postnet

训练分两阶段:
- **Stage 1 (无监督文本预训练)**: 在 19 种语言的纯文本上做 MLM 预训练 [§2.1]
- **Stage 2 (有监督 TTS 训练)**: 在 7 种语言的配对数据上训练,冻结 language-aware embedding layer [§2.2]

### 关键设计选择

**1. Language-aware embedding layer 的设计 [§2.1]**

三层结构:
- **Token embedding** $\theta_T$: 将输入 token (bytes 或 IPA) 映射为 $d$ 维向量
- **Language embedding** $\theta_L$: 将语言 ID 映射为 $d$ 维向量
- **Bottleneck layer** $\theta_B$: 由 LayerNorm + down projection (d→256) + ReLU + up projection (256→d) 的残差网络组成 [§2.4]

计算: $H_{in} = \text{Bottleneck}(Z^m + e_l; \theta_B)$ [Eq. 2]

**Why bottleneck?** [论文原文] bottleneck 通过非线性降维将 token 和 language embedding 投射到隐藏输入表示,有助于改善零样本 TTS 的泛化能力 [§3.4]。[agent 解读] 类似于 NLP 中 adapter layer 的思想 (引用 Bapna & Firat, 2019),bottleneck 在低维空间中融合 token 和 language 信息,可能迫使模型学习更紧凑、更可迁移的表征。

**2. 冻结策略 [§2.2]**

预训练后进入有监督阶段时,冻结整个 language-aware embedding layer ($\theta_T, \theta_L, \theta_B$),仅更新 encoder ($\theta_E$) 和 decoder ($\theta_D$) [Eq. 9]。

**Why freeze?** [论文原文] 与简单 fine-tuning (更新全部参数,如 Eq. 8) 相比,冻结方案保留了预训练获得的跨语言表征。消融实验表明冻结方案在大多数语言和指标上优于更新方案,平均 MCD 差 0.29,CER 差 1.04% [§3.4]。

[agent 解读] 关键 insight: 如果在 TTS fine-tuning 时更新 embedding 层,配对数据只包含少数语言,embedding 会被拉向这些语言的语音特性方向,丢失预训练获得的跨语言共性。冻结则保持 embedding 空间不变,让 encoder/decoder 学习将这个"多语言文本空间"映射到语音。

**3. 输入 token 类型 [§2.1]**

两种 token 方案:
- **UTF-8 bytes**: 直接使用字符的字节编码,无需任何语言学知识
- **IPA symbols**: 使用 eSpeak-NG 提取国际音标,需 G2P 工具

MLM 预训练使用与 BERT 相同的 masking 策略: 12% token 替换为 [MASK],1.5% 替换为随机 token,1.5% 保持不变 [§2.1]。

**4. 说话人表征 [§2.4]**

使用 x-vector (在 VoxCeleb1/2 上预训练,SpeechBrain 实现),通过投影层加到 encoder 输出 [§2.4]。训练时使用该语言训练数据的平均 x-vector;零样本推理时使用目标语言测试数据的平均 x-vector (或使用相近语言的 x-vector) [§2.4]。

### 训练策略

- **预训练**: 使用 VoxPopuli + M-AILABS + CSS10 的文本 (共 2.8GB,19 种语言),Noam optimizer,1.2M iterations [§3.1]
- **TTS 训练**: CSS10 的 7 种欧洲语言配对数据 (每语言单说话人,4-21h 不等),200 epochs (2.47M iterations),冻结 language-aware embedding [§3.1]
- **Vocoder**: HiFi-GAN,在 LibriTTS + VCTK + CSS10 上训练 [§3.1]
- **声学特征**: 16kHz,80 维 mel filterbank,FFT 1024,帧移 256 samples [§3.1]

## 实验

### 已见语言 (Seen Languages)

| 指标 | 本文 (Bytes) | 本文 (IPA) | Baseline (IPA multi w/ LIDs) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MCD (de) | **5.65** | 5.88 | 6.16 | CSS10 | [Table 2] |
| CER (de) | **3.79** | 5.52 | 9.76 | CSS10 | [Table 2] |
| MCD (fr) | **6.48** | 6.61 | 6.88 | CSS10 | [Table 2] |
| CER (fr) | **7.15** | 7.72 | 14.97 | CSS10 | [Table 2] |
| AMOS avg | **2.89** | 2.84 | 2.83 | CSS10 | [Fig 4] |

关键发现: 预训练后的 byte 模型全面超过 IPA baseline,**甚至超过使用 G2P 的 IPA 模型** [§3.2]。法语 (deep orthography) 改善尤其显著: baseline bytes CER 91.82% → proposed bytes CER 7.15% [Table 2]。

### 未见语言 (Unseen Language — 西班牙语)

| 指标 | 本文 (Bytes) | 本文 (IPA) | Baseline (Bytes) | Baseline (IPA) | Oracle (IPA multi) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CER (es x-vector) | 18.27 | **11.69** | 64.07 | 44.75 | 5.32 | CSS10 | [Table 3] |
| MCD (es x-vector) | 9.05 | 9.44 | 11.22 | 10.75 | 6.20 | CSS10 | [Table 3] |
| CER (fr x-vector) | **13.74** | 13.33 | 66.45 | 44.37 | — | CSS10 | [Table 3] |
| MOS (es) | **3.44** | 3.32 | 3.29 | — | 3.96 | CSS10 | [Fig 5] |

关键发现:
1. 预训练将 CER 降低了 60-70% (绝对值),从 baseline 的 44-64% 降到 11-18% [Table 3]
2. 在未见语言上,IPA 模型优于 byte 模型 (与已见语言上相反) [§3.3]
3. 使用法语 x-vector 替代西班牙语 x-vector 时 byte 模型 CER 反而更低 (13.74% vs 18.27%),可能因为 fr x-vector 出现在训练数据中 [§3.3]
4. AB test 显示 proposed bytes 显著优于 baseline IPA (p=0.011) [Fig 5]

### 消融实验 [Table 4]

| 消融项 | 平均 CER 变化 | 平均 MCD 变化 | 出处 |
| --- | --- | --- | --- |
| W/o bottleneck layer | +4.16% CER | +0.53 MCD | [Table 4] |
| W/o language ID | +4.48% CER | +0.50 MCD | [Table 4] |
| W/o initializing encoder | +2.27% CER | +0.04 MCD | [Table 4] |
| Updating language-aware embedding | +1.04% CER | +0.29 MCD | [Table 4] |

关键发现:
1. **Bottleneck layer 对未见语言最重要**: 移除后未见语言 MCD 增加 1.21 [§3.4]
2. **预训练收益主要来自 embedding layer 而非 encoder**: 不初始化 encoder 仅增加 2.27% CER,而不用 language ID 增加 4.48% CER [§3.4]
3. **冻结优于更新**: 冻结 language-aware embedding 在大多数条件下更优 [§3.4]

### 附录实验

- **文本域影响** [Table 6]: spoken text 预训练优于 written text (平均 CER 低 2.94%),但混合使用可改善未见语言的泛化
- **Bottleneck 架构** [Table 7]: Transformer encoder 做 bottleneck 在未见语言上 CER 改善 4.12%,但平均指标相当
- **语言依赖性** [Table 5, §3.5]: 将 de (日耳曼语系) 和 hu (乌拉尔语系) 分别作为未见语言,de 改善明显 (CER -10%),hu 改善有限 (CER -2%)
- **Cross-attention 可视化** [Fig 6]: 预训练模型在未见语言上的 cross-attention 更连续、更对角化,baseline 出现明显不连续

## 局限性

1. **语言覆盖有限**: 仅在欧洲语言上验证,所有语言都是印欧语系或乌拉尔语系,未涉及声调语言 (中文)、阿拉伯语系等差异更大的语言 [§5]
2. **与 oracle 仍有显著 gap**: 零样本 CER 11.69% vs oracle 5.32%,MOS 3.44 vs 3.96 [Table 3, Fig 5]
3. **语言依赖性未解决**: 性能强烈依赖已见语言中是否有与未见语言相似的语言 [§3.5]
4. **模型规模小**: 6 层 Transformer,在当今标准下较小;更大模型是否能缩小 gap 未知
5. **单说话人评估**: 每语言仅 1 个说话人 (CSS10),无法评估多说话人泛化能力
6. **x-vector 依赖**: 零样本推理仍需目标语言的 x-vector (或相近语言的),并非完全"零资源"

## 点评

**意义**: 本文提出了一个简洁而有效的框架,将 NLP 中 mBERT 的跨语言迁移思想引入 TTS。在 2023 年 LLM-TTS 兴起之前,这是少有的关注"语言维度零样本"的工作 (不同于 VALL-E 等关注"说话人维度零样本")。仅用文本数据就能为新语言构建可理解的 TTS 系统,对低资源语言有实际价值。

**方法论价值**:
1. frozen embedding 策略的有效性值得关注 — 它说明跨语言迁移的关键在于保持预训练的 embedding 空间不被下游任务破坏,这与 NLP 中的发现一致
2. byte-based 模型通过预训练即可超过 IPA baseline,表明 G2P 工具并非多语言 TTS 的必需品 — 这一发现被后续工作 (如 XTTS, X-Voice 仍选择 IPA) 部分验证但也受到挑战

**局限性评估**: 实验设计的一个弱点是仅用 CSS10 这个小数据集 (每语言 4-21h 单说话人),且仅在欧洲语言间迁移。这些语言共享相似的字母体系和语法结构,跨语言迁移的难度相对较低。在真正的"低资源"场景 (如非洲语言、太平洋岛屿语言) 中效果如何是未知的。

**在当今视角下的定位**: 本文的贡献更多是概念性的 (proof of concept) 而非实用性的。现代多语言 TTS 系统 (CosyVoice 3 支持 9 语言,X-Voice 支持 30 语言) 通过大规模多语言数据训练直接获得跨语言能力,不需要分阶段预训练。但对于真正无语音数据的极低资源语言,本文的思路仍有参考价值。

## 可复用的 idea

1. **Frozen pretrained embedding for cross-domain transfer**: 预训练后冻结 embedding 层、只训练上层网络的策略可迁移到其他跨域迁移场景 (如从文本到歌唱合成)
2. **Bottleneck as language adapter**: 用 bottleneck layer 融合 token 和 language embedding 的设计可借鉴到需要风格/说话人/情感条件注入的场景
3. **Text-only pretraining for TTS**: 利用大量无标注文本预训练 TTS encoder 的思路,在低资源场景下可能比从头训练更高效
4. **Byte-based multilingual TTS without G2P**: 证明通过预训练可以绕过 G2P,对不支持 G2P 的语言有价值

## 审阅

> [!review] 审阅 (2026-06-06, inline — no subagent available)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节因果解释充分,设计选择 WHY 明确 |
> | 可信赖 | pass | 所有关键数字经 PDF 交叉验证正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注一致 |
> | 可定位 | pass | KB 背景谱系定位具体,创新判断有对比 |
> | 不污染 | pass | 反向更新为纯 append,无 overclaim |
> 
> Issues: 2 (high: 0, medium: 0, low: 2)
> 详见 `_review/LearningToSpeakFromText-review.yml`
