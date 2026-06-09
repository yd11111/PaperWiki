---
type: paper
tier: deep
title: "XEUS: Towards Robust Speech Representation Learning for Thousands of Languages"
arxiv_id: "2407.00837"
source: "Sources/XEUS.pdf"
authors: [William Chen, Wangyou Zhang, Yifan Peng, Xinjian Li, Jinchuan Tian, Jiatong Shi, Xuankai Chang, Soumi Maiti, Karen Livescu, Shinji Watanabe]
year: 2024
venue: "arXiv (CMU / Shanghai Jiaotong / Toyota Tech)"
tags: [self-supervised-learning, speech-representation, multilingual, massively-multilingual, E-Branchformer, dereverberation, SSL, ASR, speech-translation, ML-SUPERB, SUPERB]
concepts: ["[[Self-SupervisedSpeechRepresentation]]", "[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]"]
models: ["[[模型库/HuBERT|HuBERT]]", "[[模型库/WavLM|WavLM]]", "[[模型库/w2v-BERT|w2v-BERT]]"]
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[SpeechTokenizer]], [[SemanticvsAcousticTokens]], [[CodebookCollapse]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓(confirmed), [[SemanticvsAcousticTokens]]✓(confirmed), [[CodebookCollapse]]✓(confirmed) | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review) | 未命中但可能相关: 无

**已有知识要点**:
- [[SpeechTokenizer]]: 自监督 tokenizer (HuBERT, w2v-BERT 2.0) 通过 masked prediction 学习表征后做 k-means 聚类离散化;w2v-BERT 2.0 tokens 被 AudioLM 采用,MaskGCT 用 VQ-VAE 量化其第 17 层特征 ✓
- [[SemanticvsAcousticTokens]]: SSL 表征产生 semantic tokens,语义连贯但缺乏声学细节;HuBERT 25Hz 在 SALMon benchmark 中语义任务最强 ✓
- [[CodebookCollapse]]: w2v-BERT 首次发现 contrastive loss 是防止端到端量化中 codebook collapse 的必要条件 ✓
- [[Self-SupervisedSpeechRepresentation]] [待确认]: XEUS 结合了 HuBERT 的 masked prediction + WavLM 的 denoising + 新提出的 dereverberation,属于 Combined 范式的进一步演进

## 速查

> [!summary] 速查
> - **一句话**: 提出 XEUS (Cross-lingual Encoder for Universal Speech),一个在 1M 小时、4057 种语言数据上预训练的 E-Branchformer 语音 SSL 模型,结合 masked prediction + denoising + 新颖的 dereverberation 目标,在 ML-SUPERB 上设立新 SOTA
> - **路线**: Waveform → (noise/reverb augmentation) → Random Masking → XEUS (Student, E-Branchformer 19层) → Predicted Pseudo-labels; WavLabLM (Teacher) → K-Means → Target Pseudo-labels
> - **指标**: ML-SUPERB SUPERB_s 956/956 (SOTA) [Table 3]; SUPERB 英语 4 项第一 (KS, SD, ER, ASR) [Table 5]; FLEURS ASR CER 8.9 (接近 w2v-BERT 2.0 v2 8.7) [Table 4]; VCTK resynthesis MOS 3.23 / WER 10.0 (均最优) [Table 6]
> - **可借鉴**: (1) Dereverberation 作为 SSL 预训练目标,增强噪声/混响鲁棒性; (2) 7413 小时新语料覆盖 4057 种语言的数据收集方法 (WikiTongues + Jesus Dramas + MMS-unlab v2); (3) E-Branchformer 替代 Transformer/Conformer 提升训练效率
> - **局限**: 长尾语言 (<1h 数据) 下游性能仍显著差于高资源语言 [§7]; 数据/训练代码尚未完全公开; 577M 参数量未做更大规模的 scaling 实验

## 核心问题

**XEUS 要解决什么问题?** [论文原文]

现有多语言 SSL 语音模型面临三大瓶颈 [§1, §2]:
1. **语言覆盖不足**: 大多数多语言 SSL 模型仅覆盖 50-150 种语言 (XLS-R 128种, w2v-BERT 51种, WavLabLM 136种),远未覆盖世界 7000+ 种语言 [Table 1]
2. **噪声鲁棒性差**: 低资源语言的录音往往特别嘈杂,且录音条件多样 (混响、背景噪音等),但现有 SSL 模型大多仅在干净语音上训练 [§2.2]
3. **可复现性差**: 性能最好的模型 (Whisper, USM, w2v-BERT 2.0) 均基于未公开数据训练,阻碍了研究社区的进展 [Table 1, §2.3]

**核心思路**: [论文原文] "Our goal is to thus build a universal speech encoder that can handle both linguistically and acoustically diverse speech" [§1] — 通过 (a) 大幅扩展语言覆盖 (4057种), (b) 引入 dereverberation 作为新 SSL 任务增强鲁棒性, (c) 全面开源数据+代码+checkpoints。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

XEUS 基于 HuBERT 架构并做了多项修改 [§4.3, Fig 2]:
1. **Feature Extractor**: Convolutional feature extractor (沿用 HuBERT 设计) [§4.3]
2. **E-Branchformer Encoder**: 19 层,替代原始 Transformer [§4.3]
   - Hidden dimension: 1,024; FFN size: 4,096; 8 attention heads; kernel size: 31
   - 总参数量: **577M** [§4.3]
3. **Teacher Model**: 预训练的 WavLabLM (Chen et al., 2023b),其输出经 k-means (k=2048) 聚类产生 target pseudo-labels [§4.1]
4. **Loss**: Cross-entropy loss (替代原始 HuBERT 的 cosine similarity loss),更快且下游性能更好 [§4.3]

[论文原文] 选择 E-Branchformer 而非 Conformer 的原因: "convolution-augmented models achieve superior SSL performance" + E-Branchformer 训练比 Conformer 更容易 [§4.3]

### 关键设计选择

#### 1. Masked Prediction + Denoising [§4.1]

[论文原文] 沿用 HuBERT 的 masked prediction 框架: 输入语音经随机 masking 后,模型预测 WavLabLM teacher 产生的 pseudo-labels [§4.1]

同时整合 WavLM 的 acoustic denoising: 训练时以概率 $p=0.2$ 对输入施加噪声干扰 (随机噪声或 batch 内另一条语音),模型需从含噪输入预测干净语音的 pseudo-labels [§4.1, Algorithm 1]

#### 2. 新颖的 Dereverberation 目标 [§4.2]

[论文原文] XEUS 提出将混响去除作为新的 SSL 预训练任务,扩展了 WavLM 的 denoising 思路 [§4.2]:

- 训练时以概率 $p_r=0.3$ 对输入施加模拟混响 (Room Impulse Response 卷积) [§4.2, Algorithm 1]
- Target pseudo-labels 始终来自**干净语音**,模型被迫隐式学习去混响 [§4.2]
- 关键技术: 混响后需重新对齐 (argmax(RIR) 偏移估计 + 能量归一化),否则音频会错位导致 pseudo-labels 不匹配 [§4.2]
- [agent 解读] 去混响目标特别适合多语言场景:低资源语言的录音往往在非标准录音环境中完成 (教堂、户外、简陋录音室),混响是主要的录音质量瓶颈

#### 3. 大规模多语言预训练数据 [§3]

[论文原文] XEUS 的预训练数据分为两部分 [§3]:

**已有公开数据集 (1.074M hours, 150+ 语言)** [§3.1, Table 2]:
- 37 个公开可获取的语音数据集,包括 YODAS (422K h), VoxPopuli (400K h), LibriLight (60K h) 等
- 涵盖多种说话风格: 自发语音、口音语音、code-switching、歌唱 [§3.1]

**新收集语料 (7,413 hours, 4,057 语言)** [§3.2-3.4]:
- **MMS-unlab v2** (6,700 h, 4,023 语言): 复现并扩展 MMS 数据集,从 Global Recordings Network 爬取宗教有声读物 [§3.2]
- **WikiTongues** (70 h, ~700 语言): 草根语言保护项目的录音 [§3.3]
- **Jesus Dramas** (645 h, 430 语言): 多角色宗教音频剧 [§3.4]

总计: **1.081M hours** 预训练数据,**4,057 ISO3 语言**,语言分布为长尾分布 (top 50 语言占 99.5% 数据) [§3.5, Fig 1]

### 训练策略

[论文原文] 预训练配置 [§4.4]:
- **硬件**: 64 张 40GB NVIDIA A100 GPUs [§4.4]
- **Batch size**: 每 GPU 最大 100 秒,总 batch 约 106 分钟 [§4.4]
- **噪声概率**: $p=0.2$,随机噪声和 batch 内语音各 50% [§4.1]
- **混响概率**: $p_r=0.3$ [§4.2]
- **训练轮次**: 2 passes through training set,共 670K steps [§4.4]
- **工具链**: 基于 ESPnet 并做了大量优化 [§4.4]

## 实验

| 指标 | 本文 (XEUS) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ML-SUPERB SUPERB_s (10min/1h) | **956/956** | w2v-BERT 2.0 v2: 826/916 | ML-SUPERB | [Table 3] |
| Mono ASR CER | 30.3/25.1 | XLS-R 1B: 39.7/30.6 | ML-SUPERB | [Table 3] |
| Multi ASR CER (Normal) | **21.1/20.1** | w2v-BERT 2.0 v2: 24.6/20.3 | ML-SUPERB | [Table 3] |
| LID ACC (Normal) | **81.5/87.3** | MMS 1B: 84.8/86.1 | ML-SUPERB | [Table 3] |
| FLEURS CER | 8.9 | w2v-BERT 2.0 v2: 8.7 | FLEURS 102-lang | [Table 4] |
| JesusFilm chrF | **22.1** | MMS 1B: 15.5 | Low-resource ST | [Table 4] |
| SUPERB PR/ASR | 3.21/3.34 | WavLM Large: 3.06/3.44 | SUPERB | [Table 5] |
| SUPERB KS/SD/ER | **98.32/3.11/71.08** | WavLM: 97.86/3.24/70.62 | SUPERB | [Table 5] |
| VCTK MOS | **3.23** | w2v-BERT 2.0 v2: 3.21 | VCTK resynthesis | [Table 6] |
| VCTK WER | **10.0** | w2v-BERT 2.0 v2: 15.5 | VCTK resynthesis | [Table 6] |

**关键实验发现**:

1. **ML-SUPERB 全面 SOTA** [Table 3]: [论文原文] XEUS 在 ML-SUPERB 10min 和 1h 两个设定上均取得最高 SUPERB_s 分数 956,超越参数更多的 w2v-BERT 2.0 v2 (580M, 826/916) 和 MMS 1B (316M/1B, 953/948),且仅使用 22% 的数据量 (1M vs 4.5M hours) [§5.1.1]

2. **低资源语言语音翻译** [Table 4]: [论文原文] XEUS 在 JesusFilm ST 上 chrF 22.1,相对 MMS 1B 的 15.5 提升 35%,主要得益于更广的语言覆盖 (4057 vs 1406 语言) [§5.1.3]

3. **英语任务同样强劲** [Table 5]: [论文原文] 尽管 XEUS 聚焦多语言,在英语 SUPERB 上仍在 KS/SD/ER/ASR 4 项任务上取得第一,证明多语言预训练不损害英语性能 [§5.2]

4. **语音重合成质量最佳** [Table 6]: [论文原文] XEUS 重合成语音 MOS 3.23、WER 10.0,均优于 WavLM Large (3.20/27.8) 和 w2v-BERT 2.0 v2 (3.21/15.5),表明其声学表征质量最高 [§5.3]

## 局限性

1. **长尾语言瓶颈** [§7]: [论文原文] 4000+ 种语言中许多仅有 <1 小时数据,这些语言的下游性能仍显著差于高资源语言
2. **录音质量不一**: [agent 解读] 新收集的 WikiTongues/Jesus Dramas 语料录音质量参差不齐,可能引入 noisy supervision
3. **评估覆盖有限** [§7]: [论文原文] 由于任务和领域覆盖广,主要使用 SUPERB 等轻量 benchmark + 有限超参调优,未做大规模 fine-tuning
4. **未公开完整训练数据**: [agent 解读] 虽然模型/代码/配置已公开,但部分数据源 (如需要许可的 BABEL) 可能限制完全复现
5. **模型规模局限**: [论文原文] 577M 参数未探索更大规模 (如 1B+) 的 scaling 效果

## 点评

**历史地位**: [agent 解读] XEUS 代表了多语言 SSL 语音模型在语言覆盖维度的最大突破 — 从 MMS 的 1406 种语言一跃至 4057 种,同时保持甚至超越 SOTA 性能。更重要的是,XEUS 在开放性方面树立了标杆: 全部数据/代码/checkpoints/200+ 中间 checkpoints 公开,这在 speech foundation model 领域前所未有。

**方法论贡献**:
1. **Dereverberation 作为 SSL 目标**: 扩展了 WavLM 的 denoising 思路,针对多语言录音的真实瓶颈 (混响) 设计了新的预训练任务
2. **长尾语言数据收集**: WikiTongues + Jesus Dramas 的数据源策略展示了从非传统来源获取低资源语言数据的可行路径
3. **E-Branchformer 替代**: 证明了 E-Branchformer 在大规模 SSL 中的有效性,提供了 Conformer/Transformer 之外的选择

**与已有知识的关系**: [agent 解读]
- XEUS 是 [[Self-SupervisedSpeechRepresentation]] 演进链中"大规模多语言"方向的最新里程碑
- 其 teacher model (WavLabLM) 和预训练目标 (masked prediction + denoising) 直接继承自 [[模型库/HuBERT|HuBERT]] 和 [[模型库/WavLM|WavLM]]
- 与 w2v-BERT 2.0 v2 的直接对比显示: 更好的预训练目标设计 (dereverberation) + 更多样的数据可以弥补训练数据总量的差距 (1M vs 4.5M hours)

## 可复用的 idea

1. **Dereverberation 预训练**: RIR 卷积 + 时间对齐 + 能量归一化的完整 pipeline,可迁移到任何 SSL 模型的鲁棒性增强
2. **非传统数据源**: WikiTongues (草根语言保护项目) 和 Jesus Dramas (宗教音频剧) 展示了低资源语言数据获取的创新思路
3. **E-Branchformer 架构选择**: 在训练稳定性和下游性能间取得良好平衡,适合大规模 SSL 预训练
4. **Cross-entropy 替代 cosine similarity**: 比原始 HuBERT loss 更快收敛且下游性能更好 [§4.3]
5. **全面开放策略**: 释放中间 checkpoints + 训练日志,支持社区研究训练动态

---

检索命中: [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[CodebookCollapse]]✓ | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[AudioUnderstanding]](pending-review) | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
