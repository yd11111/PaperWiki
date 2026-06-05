---
type: paper
tier: deep
title: "Continuous Speech Tokenizer in Text To Speech"
arxiv_id: "2410.17081"
source: "Sources/Cont-SPT.pdf"
authors: [Yixing Li, Ruobing Xie, Xingwu Sun, Yu Cheng, Zhanhui Kang]
year: 2024
venue: "NAACL 2025 Findings"
tags: [TTS, continuous-token, speech-tokenizer, autoregressive, flow-matching, information-retention]
concepts: ["[[SpeechTokenizer]]", "[[ResidualVectorQuantization]]", "[[ConditionalFlowMatching]]", "[[VariationalAutoencoderforTTS]]", "[[CodecLanguageModel]]", "[[SemanticvsAcousticTokens]]", "[[MelSpectrogram]]", "[[TTSEvaluation]]"]
models: ["[[EnCodec]]", "[[MELLE]]", "[[论文笔记/VALL-E|VALL-E]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["LibriSpeech"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认参考: [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓, [[ConditionalFlowMatching]]✓, [[SemanticvsAcousticTokens]]✓, [[CodecLanguageModel]][待确认], [[VariationalAutoencoderforTTS]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓, [[ConditionalFlowMatching]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[VariationalAutoencoderforTTS]](pending-review) | 未命中但可能相关: 无

**谱系定位**: Cont-SPT 属于 Speech Tokenizer 页记录的 "Continuous VAE Tokenizer" 新路线的早期探索之一。Speech Tokenizer 页的演进线已标注: "Continuous VAE tokenizer (sigma-VAE, LatentLM 2024; shortcut-VAE, CLEAR 2025): 绕过离散量化,直接用 VAE 编码为连续 latent vectors"。Cont-SPT 提出时间 (2024-10) 略早于 LatentLM (2024-12) 和 CLEAR (2025),是同期独立探索连续语音表征路线的工作之一。

**已有认知**: RVQ 页详细记录了离散量化的信息损失机制 — 有限 codebook 大小约束表示空间,造成 high-frequency 信息丢失。Semantic vs Acoustic Tokens 页的核心 trade-off 表指出 semantic tokens "缺高频细节"而 acoustic tokens 因 RVQ 多级量化可保留声学保真度,但序列更长。Cont-SPT 试图从根本上绕过离散化带来的信息损失。

**创新判断**: 与同期的 MELLE (直接在 mel-spectrogram 空间做 AR,完全去掉 tokenizer) 相比,Cont-SPT 保留了 tokenizer 概念但将其改为连续版本,并进行了 tokenizer 预训练和联合训练。相比后续更完整的 LatentLM (sigma-VAE + diffusion head) 和 CLEAR (enhanced wav-VAE + rectified flow),Cont-SPT 的方案更简洁(encoder 直出连续 embedding + MSE loss),但在架构设计和实验规模上相对初步。

> [!summary] 速查
> - **一句话**: 用连续 speech tokenizer (Cont-SPT) 替代 RVQ 离散 tokenizer,在 VALL-E 架构上减少量化信息损失,提升 TTS 连续性和 MOS
> - **路线**: Audio → Encoder(24kHz resample) → Continuous embedding → AR Language Model → Codec Decoder + Flow Matching + De-noising → Output Audio
> - **指标**: WER 6.59% (vs VALL-E 12.73%), SIM 0.73 (vs 0.53), EMoS 1.32 (vs 0.83), Continuity 3.61 (vs 1.80), LibriSpeech test-clean [Table 1, 2]
> - **可借鉴**: (1) tokenizer encoder-decoder 先以 VAE-like 方式预训练再联合微调(LR=0.05x LM); (2) 用频域 transfer function 分析 tokenizer 信息保留能力; (3) 连续 token 对采样率更鲁棒
> - **局限**: 仅在 LibriSpeech (~1000h) 上实验,未验证大规模数据; 无 zero-shot speaker cloning 评估; 未在多模态 LLM 场景验证; EMoS 绝对值偏低(1.32),可能与评估工具有关

## 核心问题

1. **离散 speech tokenizer 的信息损失有多大?** 现有 RVQ-based tokenizer 将 encoder 输出量化为离散 codebook index,有限 codebook 大小导致信息丢失 — 尤其在高频段。这个损失对下游 TTS 的影响如何量化?
2. **连续 token 能否作为 LM 的直接输入/输出?** 语言模型通常操作离散 token (next-token prediction),连续表征需要不同的训练目标 (MSE 而非 cross-entropy)。这是否可行?
3. **tokenizer 的预训练和联合训练各自贡献多大?** 连续 tokenizer 的 encoder-decoder 需要先预训练再联合训练 LM,两阶段各自的必要性是什么?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Cont-SPT 系统包含四个组件 [Fig 1d]:

1. **Continuous Speech Tokenizer** [Fig 1b]: 与离散版 [Fig 1a] 共享 encoder 架构,区别在于去掉了 RVQ 模块,直接使用 encoder 输出的连续向量作为 speech token。输入音频先 resample 到 24kHz,再通过 encoder 得到连续 embedding [§2.1]。
2. **Text Tokenizer + Embedding**: 文本经 text tokenizer 和 embedding 层得到文本 embedding [§2.2, Eq. 2]。
3. **Autoregressive Language Model**: 连续 speech token 和 text embedding 拼接后输入 AR transformer,按 VALL-E 架构构建 [§3.1, "the number of model blocks are set to the default values"]。LM 输出为预测的连续 speech token(非离散 code)。
4. **Audio Decoder** [Fig 1c]: 包含 codec decoder、OT-CFM flow matching 和 de-noising 三个子模块,将 LM 输出的连续 token 转为 mel spectrogram 再合成波形 [§2.3]。

### 关键设计选择

**为什么去掉 RVQ 保留 encoder?** [论文原文] RVQ 的量化过程 "involves distance grounding based on the content within the codebook, which suffers from great information loss" [§2.1]。作者的方案是最小化改动 — 仅去除 RVQ,保留完整的 encoder + decoder 架构,用连续 embedding 直接替代离散 code [§2.1]。[agent 解读] 这与 MELLE 的策略不同:MELLE 完全去掉了 tokenizer 而直接操作 mel-spectrogram;Cont-SPT 认为 tokenizer encoder 提供的压缩表征仍有价值,只是不应该离散化。

**为什么用 VAE-like 预训练?** [论文原文] tokenizer 的 encoder 和 decoder "can be pre-trained together in a VAE-like form, and our experiments show that this is beneficial for TTS tasks" [§2.1]。预训练损失包含两部分:重建损失 (encoder → decoder → 重建音频与原始的相似度) + ASR CTC 损失 (重建音频经 ASR decoder 的文本识别准确度) [§2.4.1, Eq. 5]。[agent 解读] ASR 损失的加入确保了连续 embedding 不仅保留声学信息还隐含语义信息,类似于一种弱监督约束。

**为什么用 MSE 而非 cross-entropy?** [论文原文] 由于 LM 输出是连续向量而非离散 code,损失函数改为 MSE:"the loss function is calculated by the distance between the continuous speech token of the label and the model output" [§2.4.2, Eq. 7]。[agent 解读] 这是连续 token 路线的必然选择。后续 MELLE 也用了类似的回归损失但额外引入了 spectrogram flux loss 和 latent sampling module 来解决连续 AR 的 static frame 和采样问题,Cont-SPT 未涉及这些。

**tokenizer 联合训练的学习率策略**: tokenizer 在第二阶段 (与 LM 联合训练时) 的学习率设为 LM 的 0.05 [§2.4.3]。[agent 解读] 较低的 LR 防止预训练好的 tokenizer 在联合训练中被过度改变,类似于 fine-tuning 中冻结大部分层的策略。

### 训练策略

两阶段训练 [§2.4.3]:

1. **Stage 1 — Tokenizer 预训练**: encoder 连接 decoder,以 VAE-like 方式训练。损失 = SIM(A, A_hat) + CTC(Y, Y_hat) [Eq. 5]。
2. **Stage 2 — 联合训练**: 冻结 decoder,encoder(tokenizer) + LM 联合训练。tokenizer LR = 0.05 × LM LR。损失 = MSE(O, M) [Eq. 7],其中 M 是 label 音频经 tokenizer encode 的连续 token。

## 实验

### TTS 基本指标 [Table 1]

| 指标 | Origin | VALL-E | MELLE | Cont-SPT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER ↓ | 2.61% | 12.73% | 7.26% | **6.59%** | LibriSpeech test-clean | [Table 1] |
| SIM ↑ | 0.86 | 0.53 | 0.67 | **0.73** | LibriSpeech test-clean | [Table 1] |

### 语音生成质量 [Table 2]

| 指标 | VALL-E | Cont-SPT | 出处 |
| --- | --- | --- | --- |
| EMoS ↑ | 0.83 | **1.32** | [Table 2] |
| CLVP ↓ | 3.52 | **2.94** | [Table 2] |
| STOI ↑ | 0.20 | **0.42** | [Table 2] |
| Noisiness Quality ↑ | 1.27 | **1.82** | [Table 2] |
| Continuity Quality ↑ | 1.80 | **3.61** | [Table 2] |
| Loudness Quality ↑ | 1.56 | **1.65** | [Table 2] |
| Naturalness ↑ | 1.31 | **1.45** | [Table 2] |

Cont-SPT 在所有指标上优于 VALL-E baseline。**Continuity Quality 提升最显著 (1.80 → 3.61)**,作者归因于连续 token 更好的信息保留 [§3.2]。

### 频域信息保留分析 [Table 3]

| 频率 | Discrete | Continuous | 出处 |
| --- | --- | --- | --- |
| 2kHz | 0.95 | 0.94 | [Table 3] |
| 5kHz | 0.78 | 0.81 | [Table 3] |
| 8kHz | 0.34 | **0.55** | [Table 3] |

[论文原文] "The continuous speech tokenizer has high information retention in all frequency bands, especially in the high frequency part" [§3.3]。离散 tokenizer 在 8kHz 处 retention rate 仅 0.34,连续版为 0.55。[agent 解读] 这直接验证了 RVQ codebook 有限大小导致高频信息丢失的假说。低频段两者差异不大 (2kHz: 0.95 vs 0.94),说明 RVQ 对低频编码已足够。

### 消融实验 [Table 4]

| 配置 | WER ↓ | SIM ↑ | 出处 |
| --- | --- | --- | --- |
| Full (Ours) | 6.59% | 0.73 | [Table 4] |
| w/o SPT Joint Training | 6.83% | 0.71 | [Table 4] |
| w/o L_spt Pre-train | 8.42% | 0.64 | [Table 4] |

两项消融均导致性能下降,其中去掉预训练 (w/o L_spt Pre-train) 影响更大 (WER: 6.59 → 8.42, SIM: 0.73 → 0.64),说明 tokenizer 预训练是整个框架的关键 [§A.1]。

### 采样率鲁棒性 [Fig 2]

连续 tokenizer 对不同 window length / sampling rate 的变化更鲁棒,尤其在 window length ratio > 1 时优势明显 [§3.4]。[论文原文] "This is due to the better information preservation of continuous speech tokenizer" [§3.4]。

## 局限性

1. **实验规模有限**: 仅在 LibriSpeech (~1000h) 上训练和评估,未验证在大规模数据 (如 Emilia 100k+h) 上的表现。同期 MELLE 也在 LibriSpeech 上实验,但 Seed-TTS 等已证明大规模数据对连续 token 路线的重要性 [论文原文, Limitations]。
2. **未验证多模态场景**: 作者承认未在 MLLM 场景验证,"The continuous speech tokenizer proposed in our work could improve the information carrying capacity, but it may also bring difficulties of training" [Limitations]。
3. **EMoS 绝对值偏低**: Cont-SPT 的 EMoS 仅 1.32 (满分通常 5),虽优于 VALL-E 的 0.83,但与后续系统 (如 MELLE MOS 4.20 [Table 3]) 差距巨大。[agent 解读] 可能与使用的 MOSNet 评估工具版本有关,或与实验设置差异有关;论文未报告 human MOS。
4. **无 zero-shot 评估**: 未评估未见说话人的零样本语音合成能力,而这是当前 TTS 研究的核心场景。
5. **连续 AR 的固有问题未讨论**: 论文未讨论连续空间 AR 的 exposure bias、monotonic output 等问题 (MELLE 通过 spectrogram flux loss 和 latent sampling module 显式解决)。
6. **与 MELLE 的不公平对比**: Table 1 中 Cont-SPT WER 6.59% vs MELLE 7.26%,但两者训练设置可能不同 (MELLE 使用自己的训练 pipeline 和更大模型)。

## 点评

**定位**: Cont-SPT 是连续 speech token 路线在 LLM-based TTS 中的早期简洁探索。核心贡献是提出了一个完整的框架 (continuous tokenizer + AR LM + flow matching decoder) 并通过频域分析提供了理论支撑 (离散 tokenizer 的高频信息损失)。

**优势**: (1) 改动最小 — 仅去掉 RVQ,保留完整的 codec encoder-decoder 架构,便于与现有离散系统公平对比;(2) 频域 transfer function 分析提供了直观的信息损失可视化;(3) 消融清晰展示了预训练和联合训练各自的贡献。

**不足**: (1) 方法设计相对简单,未处理连续 AR 的已知问题 (static frames, sampling mechanism);(2) 实验规模和评估维度有限;(3) 与同期 MELLE 的对比缺乏统一设置。

**在知识库中的位置**: 在 Speech Tokenizer 页的 "Continuous VAE Tokenizer" 演进线中,Cont-SPT 可视为从 "离散 RVQ tokenizer" 到 "连续 VAE tokenizer" 过渡的桥梁工作 — 它保留了 tokenizer 概念并证明了连续化的价值,但尚未引入 sigma-VAE (LatentLM) 或 enhanced wav-VAE (CLEAR) 等更成熟的连续编码设计。

## 可复用的 idea

1. **频域 transfer function 分析 tokenizer 质量**: 在不同频率点 (2k/5k/8k Hz) 测量 tokenizer 的信息保留率,可快速定位 tokenizer 的信息瓶颈,适用于任何 codec 评估。
2. **tokenizer 预训练 + 联合训练的两阶段策略**: 先 VAE-like 预训练 encoder-decoder,再以较低 LR (0.05x) 与 LM 联合训练。消融显示这两步都不可或缺 [Table 4]。
3. **ASR CTC loss 作为 tokenizer 预训练的辅助约束**: 在重建损失外加入 ASR 解码损失,确保连续 embedding 保留语义信息。

---

> [!review] 审阅: pass-with-fixes (2026-06-03)
> - **可复述**: 4/5 — 方法节包含因果解释和关键设计选择的 WHY
> - **可信赖**: 4/5 — claim 标注覆盖率 94%
> - **可区分**: 4/5 — [论文原文]/[agent 解读] 标注清晰
> - **可定位**: 4/5 — KB 背景准确定位 continuous tokenizer 路线
> - **不污染**: 5/5 — 反向更新均为追加,无 factual error
> - Issues: 2 medium (frontmatter datasets 为空 → 已补; models 缺 VALL-E → 已补), 2 low
> - 详见 `_review/Cont-SPT-review.yml`

检索命中: [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓, [[ConditionalFlowMatching]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[VariationalAutoencoderforTTS]](pending-review) | 未命中但可能相关: 无
