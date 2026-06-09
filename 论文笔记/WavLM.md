---
type: paper
tier: deep
title: "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"
arxiv_id: "2110.13900"
source: "Sources/WavLM.pdf"
authors: [Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, Jian Wu, Long Zhou, Shuo Ren, Yanmin Qian, Yao Qian, Jian Wu, Michael Zeng, Xiangzhan Yu, Furu Wei]
year: 2022
venue: "IEEE Journal of Selected Topics in Signal Processing"
tags: [self-supervised-learning, speech-representation, masked-prediction, speech-denoising, multi-task, speaker-verification, speech-separation, diarization, ASR, full-stack]
concepts: ["[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[SpeakerEmbedding]]", "[[SpeechLanguageModel]]", "[[SpeechFactorization]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: ["[[模型库/WavLM|WavLM]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[SpeechTokenizer]], [[SemanticvsAcousticTokens]], [[SpeakerEmbedding]], [[SpeechLanguageModel]], [[SpeechFactorization]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓(confirmed), [[SemanticvsAcousticTokens]]✓(confirmed), [[SpeakerEmbedding]]✓(confirmed), [[SpeechLanguageModel]]✓(confirmed), [[SpeechFactorization]]✓(confirmed) | 过滤: 无 | 未命中但可能相关: 无

**已有知识要点**:
- [[SpeechTokenizer]]: Survey benchmark 中 WavLM (DWavL-S-16) 在声学属性建模上表现最佳;在 SpeechLM 中 Mimi tokenizer 使用 WavLM 作为 semantic 信号源 ✓
- [[SemanticvsAcousticTokens]]: WavLM 在 SALMon benchmark 中声学一致性最强 (Gender 92.0%, Spk 86.5%),但语义任务不如 HuBERT ✓
- [[SpeakerEmbedding]]: WavLM 表征可用于 speaker verification,在 VoxCeleb 上超越 ECAPA-TDNN SOTA ✓
- [[SpeechLanguageModel]]: WavLM 作为通用 speech foundation model,覆盖 content/speaker/semantic/paralinguistic 多维度 ✓
- [[SpeechFactorization]]: WavLM 不同层自然分离内容和说话人信息 — bottom layers 编码 speaker,top layers 编码 content ✓

## 速查

> [!summary] 速查
> - **一句话**: 首个面向 full-stack 语音处理的大规模自监督预训练模型,通过 masked speech denoising + gated relative position bias + 94k 小时多样化数据,在 SUPERB 19 个子任务和 speaker verification/diarization/separation 等非 ASR 任务上全面达到 SOTA [Abstract, §I]
> - **路线**: Waveform (+ simulated noisy/overlapped mixing) → CNN Encoder → Masking → Transformer with Gated Relative Position Bias → Masked Prediction (HuBERT-style pseudo-labels from k-means) [Fig 1, §IV]
> - **指标**: SUPERB 14/15 subtasks 超越 HuBERT Large (absolute +2.4 overall) [Table I]; VoxCeleb1 EER 0.383%/0.480%/0.986% (Vox1-O/E/H) [Table II]; LibriCSS separation avg WER 6.0 [Table IV]; CALLHOME DER 10.35 (all) [Table III]; ASR WER 1.8/3.2 (960h, Transf. LM) [Table VI]
> - **可借鉴**: (1) Masked speech denoising — 在输入中混合噪声/重叠语音,迫使模型学习去噪+说话人分离; (2) Gated relative position bias — 可微门控机制动态调整位置编码; (3) 多样化预训练数据 (94k hrs from LibriLight+GigaSpeech+VoxPopuli) 显著提升泛化
> - **局限**: 预训练 700k steps on 64 V100 (Large); 仍依赖 HuBERT 的迭代 k-means 伪标签; 未探索多语言/跨语言

## 核心问题

**WavLM 要解决什么问题?** [论文原文]

现有 SSL 语音模型 (wav2vec 2.0, HuBERT) 主要关注 ASR,在非 ASR 任务上表现不佳 [§I]:
1. **Multi-speaker 任务**: speech separation, diarization — 现有预训练不强制学习 speaker discrimination [§I]
2. **数据偏差**: 现有模型主要使用 LibriLight (audiobook),域分布单一,与真实场景不匹配 [§I]
3. **位置编码局限**: wav2vec 2.0 / HuBERT 的卷积相对位置编码不够灵活 [§I]

**核心洞察**: [论文原文] 通过在输入中引入 simulated noisy/overlapped speech 进行 masked speech denoising,模型不仅学习内容预测 (ASR 相关),还隐式学习去噪、说话人分离和说话人身份建模 — 一个预训练目标覆盖 full-stack 语音处理 [§I, §IV-B]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

[论文原文] WavLM 沿用 HuBERT 架构 (CNN Encoder + Transformer) 并做三项关键改进 [§IV, Fig 1]:

1. **CNN Encoder**: 与 wav2vec 2.0 / HuBERT 完全相同 — 7 层 512-ch CNN, strides (5,2,2,2,2,2,2), kernel (10,3,3,3,3,2,2), 输出 20ms/帧 [§IV-A]
2. **Transformer with Gated Relative Position Bias**: 替换原来的 convolution-based relative PE [§IV-A, Eq 2-6]
3. **Masked Speech Denoising**: 输入混合模拟噪声/重叠语音,目标是预测原始干净语音的伪标签 [§IV-B]

**模型配置** [§V-A]:

| 配置 | Transformer 层 | Hidden dim | Attention heads | 参数量 | 预训练数据 |
|------|---------------|------------|-----------------|-------|----------|
| Base | 12 | 768 | 8 | 94.70M | LS-960h |
| Base+ | 12 | 768 | 8 | 94.70M | Mix 94k |
| Large | 24 | 1024 | 12 | 316.62M | Mix 94k |

### 关键设计选择

#### 1. Gated Relative Position Bias [§IV-A, Eq 2-6]

[论文原文] 替代 wav2vec 2.0 / HuBERT 的卷积相对位置嵌入,使用可学习的门控机制动态调整位置偏差 [§IV-A]:

Attention 计算: $a_{ij} \propto \exp\{\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}} + r_{i-j}\}$ [Eq 3]

Gated position bias:
$$r_{i-j} = d_{|i-j|} + g_i^{(\text{update})} d_{i-j} + (1 - g_i^{(\text{update})}) \tilde{r}_{i-j}$$

其中 $g_i^{(\text{update})}, g_i^{(\text{reset})} = \sigma(\mathbf{q}_i \cdot \mathbf{u}), \sigma(\mathbf{q}_i \cdot \mathbf{w})$ [Eq 5]

- Bucket relative PE ($n=320$ embeddings), shared across layers [§IV-A]
- Logarithmic binning: max offset $m=800$, 超出则共用同一 embedding [Eq 5, 6]
- [论文原文] 门控机制让位置偏差根据当前语音内容自适应调整 — 同样的帧间距在 silence 和 speech 中应有不同权重 [§IV-A]

[agent 解读] 这一设计的核心优势是 content-aware positional encoding: 传统固定位置编码对所有内容一视同仁,但语音中 silence/speech/overlap 的时序结构差异很大,门控机制让模型学到这些差异

#### 2. Masked Speech Denoising [§IV-B, Algorithm 1]

[论文原文] 核心创新 — 在 HuBERT 的 masked prediction 基础上引入输入端噪声/重叠模拟 [§IV-B]:

**Noisy/Overlapped Speech Simulation** [Algorithm 1]:
1. 以概率 $p$ 从 batch 中选择 utterances 进行混合
2. 若 $v > p_n$ (mixing noise probability): 从同 batch 采样 secondary utterance,mixing energy ratio $\mathcal{U}(-5, 5)$ dB
3. 若 $v \leq p_n$: 混合 DNS 噪声 (Deep Noise Suppression),mixing energy ratio $\mathcal{U}(-5, 20)$ dB
4. 混合区域长度 $l \sim \text{uniform}\{1, ..., L/2\}$,确保 overlap < 50% (main speaker 始终占主导) [Algorithm 1]
5. 取 main speaker 的伪标签作为 prediction target [§IV-B]

**伪标签生成** [§V-A]:
- Base: 960h data, 400k steps, 使用 1st-iter HuBERT Base 第 6 层 k-means [§V-A]
- Base+/Large: 94k data, 1.2M/700k steps, 使用 2nd-iter HuBERT Base 第 9 层 k-means [§V-A]
- speech denoising: Base 设 $p_n=0$, Base+/Large 设 $p_n=10\%$ [§V-A]

[agent 解读] 这一设计的 insight 是: 当输入含噪声/重叠语音但目标是干净语音的伪标签时,模型被迫:
- 分辨 main speaker vs secondary speaker (→ speech separation 能力)
- 过滤噪声 (→ speech enhancement 能力)
- 保留 speaker identity 信息 (→ speaker verification 能力)
这些能力在传统 SSL 预训练中不会出现,因为输入都是干净语音

#### 3. Training Stabilization [§IV-D, Eq 8]

[论文原文] fp16 训练中 attention score $\frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d}}$ 可能超过 fp16 上界导致 NaN loss [§IV-D]:

解决: 引入缩放常数 $c=32$,使 $a_{i,j} \propto \exp\{(\frac{\mathbf{q}_i}{c\sqrt{d}} \cdot \mathbf{k}_j - \max_{j'}\frac{\mathbf{q}_i}{c\sqrt{d}} \cdot \mathbf{k}_{j'}) \times c + r_{i-j}\}$ [Eq 8]

保证 max value $< 2^{16}$,消除 overflow [§IV-D]

### 训练策略

**预训练** [§V-A, Table VII]:

| 模型 | 数据 | Steps | LR | Warmup | Batch | GPUs |
|------|------|-------|-----|--------|-------|------|
| Base | 960h | 400k | 5e-4 | 32k | 350s | 32 V100 |
| Base+ | 94k | 1.2M | 5e-4 | 96k | 350s | 32 V100 |
| Large | 94k | 700k | 1.5e-3 | 32k | 720s | 64 V100 |

**Fine-tuning**: CTC loss, SpecAugment, LayerDrop 0.05 (Base/Base+) / 0.1 (Large); 前 10k steps 冻结 Transformer [§V-F]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SUPERB overall score | 74.6 (Large) | HuBERT Large: 72.2 | SUPERB 15 tasks | [Table I] |
| EER Vox1-O/E/H | 0.383/0.480/0.986 (Large*) | ECAPA-TDNN: 1.010/1.240/2.320 | VoxCeleb1 | [Table II] |
| DER (all speakers) | 10.35 (Large) | HuBERT Large: 12.40 | CALLHOME | [Table III] |
| Separation WER avg | 6.0 (Large) | HuBERT Base: 8.1 | LibriCSS | [Table IV] |
| ASR WER (test-clean/other, 960h) | 1.8/3.2 (Large, Transf. LM) | wav2vec 2.0 Large: 1.8/3.3 | LibriSpeech | [Table VI] |
| ASR WER (test-clean/other, 1h) | 3.8/6.6 (Large, 4-gram LM) | wav2vec 2.0 Large: 3.8/7.1 | LibriSpeech | [Table V] |

**关键实验发现**:

1. **Full-stack SOTA** [Table I]: WavLM Large 在 SUPERB 14/15 subtasks 上超越 HuBERT Large,overall score +2.4 绝对提升;Base+ (仅 94.7M 参数) 已超越 HuBERT Large,证明三项改进的有效性 [§V-B]

2. **Speaker verification 突破** [Table II]: [论文原文] WavLM Large 在 VoxCeleb1 三个 trial 上 EER 0.383%/0.480%/0.986%,比 Fbank ECAPA-TDNN 相对降低 >35% [§V-C] — 证明 SSL representation 可以替代人工特征用于 speaker task

3. **Speech separation 显著提升** [Table IV]: [论文原文] WavLM Large 在 LibriCSS 上 avg WER 6.0,相比 Conformer baseline (冻结参数) 相对降 27.7%;关键是 WavLM 预训练时见过 overlapped speech [§V-E]

4. **层级信息分离** [Fig 2]: [论文原文] 不同层学到不同信息 — bottom layers 贡献 speaker-related tasks (SID, ASV, SD),top layers 贡献 content/semantic tasks (ASR, PR, IC);middle layers 对 speaker tasks 也很重要 [§V-B-3]

5. **Denoising 贡献消融** [Table I]: [论文原文] "w/o denoising task" 行:移除 denoising 后 speaker diarization 显著下降 (12.4→12.63, LibriCSS separation 8.1→8.2 vs 7.4),证实 denoising 对 multi-speaker tasks 关键 [Table I]

6. **多样化数据贡献** [Table V]: [论文原文] Base+ (94k hrs) vs Base (960h) 在所有 fine-tuning 设定下持续改善,尤其 test-other;说明 audiobook-only 数据不足以训练通用 speech model [§V-F]

## 局限性

1. **仍依赖 HuBERT 迭代 k-means**: [agent 解读] WavLM 使用 HuBERT 产生的 k-means 伪标签作为训练目标,继承了 HuBERT 的迭代训练复杂度
2. **仅英语**: [agent 解读] 94k 小时预训练数据全为英语;未探索多语言迁移能力
3. **Large 模型训练成本**: 64 V100 × 700k steps [§V-A, Table VII] [论文原文]
4. **Speaker verification 依赖额外技巧**: [论文原文] Large margin fine-tuning + quality-aware score calibration 才达到最佳 EER [Table II, §V-C]
5. **denoising simulation 简化**: [agent 解读] 噪声/重叠模拟使用简单的 energy ratio mixing,未包含真实声学环境的复杂性 (reverberation, distance effects 等)

## 点评

**历史地位**: [agent 解读] WavLM 是 SSL 语音模型从 "ASR-focused" 到 "full-stack" 的里程碑。通过一个简单但有效的改动 (masked speech denoising),将 SSL 预训练的适用范围从 ASR 扩展到 speaker verification、speech separation、diarization 等全部语音处理任务,成为 "next-generation backbone network" [§VI]。

**与前序工作的关系**:
- wav2vec 2.0: WavLM 继承了其 CNN encoder 和 Transformer backbone,但将 contrastive loss 替换为 HuBERT 的 masked prediction [§III]
- HuBERT: WavLM 直接扩展 HuBERT 框架,三项改进 (denoising + gated PE + diverse data) 在 ASR 性能不降的前提下全面提升非 ASR 任务 [§V]
- w2v-BERT: 同期工作,w2v-BERT 选择结合 contrastive + masked prediction;WavLM 选择 masked prediction + denoising;两者从不同角度改进 HuBERT

**方法论贡献**:
1. **Masked speech denoising**: 预训练时输入加噪、目标不变的简单思路,让模型隐式学会 speaker discrimination — 成本几乎为零
2. **Layer-wise information separation**: 实验证实底层→说话人信息,顶层→内容信息 [Fig 2],为后续 "加权层融合" 方案提供理论基础
3. **多样化数据的重要性**: 94k hrs (LibriLight + GigaSpeech + VoxPopuli) vs 960h (LibriSpeech only) 的对比清楚展示了数据多样性对泛化的影响

## 可复用的 idea

1. **Input perturbation + clean target**: 在自监督预训练中对输入做简单扰动 (noise, overlap),但保持 target 不变,迫使模型学习鲁棒表征 — 可迁移到任何 SSL 框架
2. **Gated relative position bias**: content-aware 位置编码,同样的帧间距根据内容动态调整权重
3. **Multi-source diverse pretraining data**: audiobook + podcast + EP recording 混合,覆盖不同 speaking styles、acoustic environments、speaker types
4. **Training stabilization (scaled attention)**: fp16 训练时的简单 attention scaling trick,对大模型训练有普遍价值
5. **Layer-wise weighted sum**: 用可学习权重融合不同层表征,不同下游任务自动选择最相关的层

---

检索命中: [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeakerEmbedding]]✓, [[SpeechLanguageModel]]✓, [[SpeechFactorization]]✓ | 过滤: 无 | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
