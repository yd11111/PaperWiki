---
type: paper
tier: deep
title: "MPO: Multidimensional Preference Optimization for Language Model-based Text-to-Speech"
arxiv_id: "2509.00685"
source: "Sources/MPO.pdf"
authors: [Kangxiang Xia, Xinfa Zhu, Jixun Yao, Lei Xie]
year: 2025
venue: "arXiv (Interspeech format)"
tags: [RLHF, DPO, preference-optimization, zero-shot-TTS, multidimensional-alignment, regularization, codec-LM, post-training]
concepts: ["[[LLM-basedTTS]]", "[[DifferentiableRewardOptimization]]", "[[TTSEvaluation]]", "[[SpeechTokenizer]]", "[[SpeakerEmbedding]]", "[[Single-codebookvsMulti-codebook]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[LLM-basedTTS]], [[SpeechTokenizer]], [[SpeakerEmbedding]])
> **谱系定位**: TTS 偏好优化已形成清晰的演进线: (1) SpeechAlign (2024) 首次将 DPO 引入 codec LM,用 golden vs synthetic AR tokens 构建偏好对; (2) UNO (2024) 处理 unpaired 主观评估中的标注不确定性; (3) RIO (2024) 用反向推理自动选择偏好样本; (4) FPO (2025, 同一实验室 NWPU) 将 loss 下沉到 error segment tokens 实现 3-4x 数据效率; (5) DiffRO/CosyVoice 3 (2025) 在 token 层做端到端可微 reward 优化; (6) Multi-Reward GRPO (2025) 在 audio-level 用多奖励 GRPO。MPO 处于路线 (1)-(3) 的同期工作,解决两个这些前驱未系统性解决的问题: (a) 如何在多个评估维度上同时优化 (已有方法多为单维度或简单 ranking 合并); (b) DPO 训练中的过拟合退化问题。
>
> **已有认知**: LLM-based TTS (confirmed) 将 TTS 重构为条件语言建模任务,decoder-only transformer 自回归生成 speech tokens; MPO 的 base model 使用 LLaMA 架构 + 单码本 neural codec (codebook 8192),属于典型 single-codebook codec LM 路线。Speaker Embedding (confirmed) 中 WavLM-large speaker verification 是标准 SIM 评估工具。TTS Evaluation [待确认] 中 WER/CER + SIM + MOS 是标准三件套。
>
> **创新判断**: 与已有工作对比,MPO 的两个贡献是互补的: (1) preference set 将"每维度选最优/最差"替代"所有维度综合 ranking",使多维度对比更 sharp (FPO 同实验室但聚焦 token-level,不涉及多维度); (2) CE loss 正则化解决 DPO 退化是工程上有效但理论贡献有限的方案 (NLP 中已有类似做法如 DPOP, SimPO 等)。
>
> 检索命中: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[DifferentiableRewardOptimization]](pending-review), [[TTSEvaluation]](pending-review), [[CodecLanguageModel]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review), [[SpeakerVerification]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 Multidimensional Preference Optimization (MPO),通过 preference set (每维度独立选最优/最差) + CE loss 正则化,实现 LM-based TTS 在 intelligibility/speaker similarity/prosody 三维度的同时偏好对齐 [§3]
> - **路线**: Text → BPE → LLaMA decoder (24L/16H/1024d) → single-codebook acoustic tokens (8192) → neural codec → waveform; 生成 10 candidates per input → 各维度选 max/min 构建 preference set (wset/lset) → DPO + CE loss 联合训练
> - **指标**: CER 3.90↓ (base 4.72), SPK SIM 0.577↑ (base 0.548), Prosody RMSE 0.279↓ (base 0.337) [Table 2]; ABX: MPO vs baseline 52.3% win [Fig 3]; vs combined rankings 40.2% win [Fig 3]
> - **可借鉴**: (1) preference set 构建: 多维度独立选极值 + 冲突时选次优/次差,比 combined ranking 更有效; (2) DPO + CE loss 联合训练防退化,简单有效 (λ=10)
> - **局限**: (1) 仅在 100h 中文数据上做偏好优化,规模偏小; (2) 无 MOS 主观评价,仅 ABX preference test; (3) 未与 SpeechAlign/UNO/RIO 等已有偏好优化方法直接对比; (4) 三个评估维度由客观指标定义 (CER/SIM/F0 RMSE),未涉及自然度/情感等更主观的维度; (5) 代码/模型未开源

## 核心问题

1. **多维度偏好优化困难**: DPO 的标准形式要求每个输入一对 (preferred, dispreferred),但 TTS 质量涉及多个维度 (可懂度/说话人相似度/韵律),一个样本可能在某维度好但在另一维度差,简单合并 ranking 会稀释各维度的对比信号 [§1, §3.1] [论文原文]

2. **DPO 过拟合退化**: DPO loss 的全局最优要求 P(yw ≻ yl) = 1,导致 πθ(yl) → 0; 在偏好数据有限时,模型会将概率质量移到训练集外的 response,甚至给 preferred response 赋近零概率,语音合成能力彻底退化 [§3.2] [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MPO 基于 LLaMA 架构的 decoder-only LM,24 层 transformer、16 attention heads、1024d embedding [§4.2]。使用 BPE 做文本 tokenization,单码本 neural codec (codebook size 8192) 做语音 tokenization [§4.2]。Base model 在 160,000 小时数据 (WenetSpeech4TTS + LibriHeavy + internal) 上从头训练 2M steps,再在 2000 小时 TTS 数据上 SFT 130K steps [§4.1-4.2]。偏好优化在 100 小时高质量中文 TTS 语料上进行 [§4.1]。

### 关键设计选择

#### 1. Preference Set 构建 (§3.1)

**为什么不直接用 combined ranking**: 传统方法将多个指标合并为单一排名 (如给各指标 0-9 分再求和),但这样做会在各维度间产生平均效应,弱化各维度的差异信号 [论文原文]。实验证实: combined ranking 训练出的模型在各维度上优化效果平均,不如 preference set [Table 2] [论文原文]。

**MPO 的做法**: 对每个 text input x,用 SFT model 生成 N=10 个 candidate 语音序列 y1...yn。设 A, B 为两个评估维度:
- wset (preferred set) = {ymaxA, ymaxB}: 各维度最优样本的并集
- lset (dispreferred set) = {yminA, yminB}: 各维度最差样本的并集

训练时,从 wset 和 lset 各随机抽一个样本组成 preference pair,仍使用标准 DPO 框架训练 [§3.1]。

**冲突处理**: 当 wset ∩ lset ≠ ∅ (同一样本既是某维度最优又是另一维度最差) 时,用该维度的次优/次差样本替代 [§3.1] [论文原文]。

**为什么这样更好** [agent 解读]: preference set 本质上是通过"各维度独立选极值"来增大 preferred 和 dispreferred 在每个维度上的差距,使 DPO loss 的梯度更聚焦。combined ranking 则因为求和操作导致极值被拉平。

#### 2. CE Loss 正则化 (§3.2)

**为什么 DPO 会退化**: DPO loss 的全局最小要求 πθ*(yl|x)/πref(yw|x) / (πθ*(yw|x)/πref(yl|x)) → ∞ [§3.2]。由于 πref 是 SFT 模型,对所有 y 都有 0 < πref(y) < 1,因此只需 πθ(yl) = 0 即可满足,即模型可以通过"杀死 dispreferred response 的概率"来最小化 loss,而不是"提升 preferred response 的概率" [§3.2] [论文原文]。

**实验验证** [§4.4, Fig 2]: 不加 CE loss 时,DPO loss 后期收敛到 0,CE loss 飙升到 ~10 (接近预训练初始水平),说明模型丧失了语音合成能力。CER 从 SFT 的 4.72 → 5k steps 4.57 → 10k steps 6.41 → 15k steps 14.52 [Table 1]。

**解决方案**: 在 DPO loss 上叠加 CE loss (即标准语言建模 loss):
L = λ·Ldpo + Lce, λ=10 [§3.2, Eq. 6]

**为什么 CE loss 能防退化** [agent 解读]: CE loss 持续强制模型维持对训练数据的生成能力,相当于对 DPO 优化施加了一个"不能太偏离 SFT 模型"的隐式约束。这与 NLP 中的 DPOP (Amini et al., 2024)、SimPO (Meng et al., 2024) 等工作中使用参考模型正则化的思路一致。

### 训练策略

- Optimizer: AdamW [§4.2]
- LR: 1×10⁻⁶ (偏好优化阶段) [§4.2]
- λ = 10 (DPO loss 权重) [§4.2]
- Hardware: 单张 NVIDIA A6000 (偏好优化) [§4.2]
- 训练约 20k steps (with CE loss) [Table 1]

### Preference Set 具体构建 (§4.3)

三个评估维度:
1. **Intelligibility**: Paraformer ASR 计算 CER,preferred 样本 CER 必须为 0 [§4.3]
2. **Speaker Similarity**: WavLM-large fine-tuned on SV 提取 speaker embedding,计算与 real audio 的 cosine similarity; preferred vs dispreferred 差距 ≥ 0.1 [§4.3]
3. **Prosody**: Log F0 RMSE (via DTW alignment, ESPnet script); preferred vs dispreferred 差距 ≥ 0.1 [§4.3]

## 实验

| 指标 | MPO | Base model | Train on CER only | Train on SIM only | Train on Prosody only | Combined Rankings | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CER↓ | **3.90** | 4.72 | 4.24 | 5.50 | 4.86 | 4.30 | 100h Mandarin | [Table 2] |
| SPK SIM↑ | **0.577** | 0.548 | 0.549 | **0.576** | 0.537 | 0.564 | 100h Mandarin | [Table 2] |
| Prosody RMSE↓ | 0.279 | 0.337 | 0.322 | 0.283 | **0.237** | **0.218** | 100h Mandarin | [Table 2] |

**关键发现**:
1. 单维度 DPO 只在目标维度改善,可能在其他维度退化 (如 Train on SIM: CER 5.50↑, Prosody 0.283↓) [Table 2]
2. Combined rankings 各维度平均改善,但 CER 和 SIM 均不如 MPO [Table 2]
3. MPO 在 CER 和 SIM 上同时达到最优,Prosody 虽不是最优但比 baseline 显著改善 [Table 2]

**ABX Preference Test** [Fig 3]:
- MPO vs baseline: 52.3% win, 22.4% tie, 25.3% lose
- MPO vs combined rankings: 40.2% win, 29.1% tie, 30.7% lose
- MPO vs ground truth: 35.3% win, 30.0% tie, 34.7% lose (接近持平)

**CE Loss 正则化效果** [Table 1, Fig 2]:
- 无 CE loss 约束: 5k steps CER 4.57 (略改善) → 10k steps CER 6.41 → 15k steps CER 14.52 (严重退化)
- 有 CE loss 约束: ~20k steps CER 4.24 (持续改善,无退化)

## 局限性

1. **实验规模有限**: 偏好优化仅在 100h 中文数据上进行,未在英文/多语言/大规模数据上验证 [agent 解读]
2. **缺少与已有方法的对比**: 未与 SpeechAlign/UNO/RIO/FPO 等同期或前驱方法直接比较,难以判断 MPO 的相对优势 [agent 解读]
3. **评估维度受限**: 仅考虑可懂度/说话人相似度/韵律三个客观可测的维度,未涉及自然度、情感表达、流利度等更主观的维度 [agent 解读]
4. **理论贡献有限**: CE loss 防退化在 NLP 中已有多种成熟方案 (DPOP, IPO 等);preference set 的改进主要是工程层面的数据构建策略 [agent 解读]
5. **无 MOS 评价**: 仅有 ABX preference test 作为主观评估,缺少 MOS 分数 [agent 解读]
6. **preference set 中维度数扩展性**: 论文仅展示 3 个维度,当维度增多时 wset/lset 大小线性增长,冲突处理可能变复杂 [agent 解读]

## 点评

MPO 是一篇解决实际工程问题的论文,两个贡献 (preference set + CE loss 正则化) 都是简洁、直觉的方案。Preference set 的核心洞见 -- "各维度独立选极值比综合排名更有效" -- 在实验中得到了验证,但缺少理论分析说明为什么。CE loss 正则化解决 DPO 退化是有效的,但在 NLP 社区已有大量类似工作 (DPO + SFT loss, DPOP 等),novelty 有限。

与同实验室的 FPO (Yao et al., 2025) 对比: FPO 聚焦 loss 计算粒度 (utterance → token-level),MPO 聚焦 data 构建策略 (single-pair → multi-dimensional set),两者是正交的,理论上可组合使用。MPO 的论文未提及 FPO,但两篇的第三作者 Jixun Yao 均为共同作者,说明可能是同一团队的并行探索。

**缺少的关键对比**: 论文仅与 "combined rankings" 对比,未与 DPO 社区的其他多目标优化方案 (如 multi-objective DPO) 或 TTS 社区的 SpeechAlign/UNO/RIO 对比,削弱了 contribution 的可信度。

## 可复用的 idea

1. **Preference set 构建策略**: 多维度评估时,与其用加权求和合并为单一排名,不如各维度独立选极值,训练时随机组合 -- 这个策略可迁移到任何多维度 DPO 场景
2. **DPO + CE loss 联合训练**: λ·Ldpo + Lce 防退化,简单有效,适用于任何基于 DPO 的 TTS 后训练 (λ=10 可作为起点)
3. **Preference data 质量约束**: 要求 preferred 样本 CER=0,preferred/dispreferred 差距 ≥ 0.1,这种 quality gating 可提升偏好数据的信噪比

---

> [!review] 审阅状态
> 待审阅 — 见 `_review/MPO-review.yml`

---

检索命中: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[DifferentiableRewardOptimization]](pending-review), [[TTSEvaluation]](pending-review), [[CodecLanguageModel]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review), [[SpeakerVerification]](pending-review) | 未命中但可能相关: 无
