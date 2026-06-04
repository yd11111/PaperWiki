---
type: paper
tier: deep
title: "BridgeCode: A Dual Speech Representation Paradigm for Autoregressive Zero-Shot Text-to-Speech Synthesis"
arxiv_id: "2510.11646"
source: "Sources/BridgeCode.pdf"
authors: [Jingyuan Xing, Mingru Yang, Zhipeng Li, Xiaofen Xing, Xiangmin Xu]
year: 2025
venue: "arXiv"
tags: [zero-shot-TTS, autoregressive, speech-representation, token-rate, discrete-continuous, RVQ, dual-representation]
concepts: ["[[Residual Vector Quantization]]", "[[Token Rate and Bitrate Trade-offs]]", "[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[LLM-based TTS]]", "[[Neural Vocoder]]", "[[Codec Language Model]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]", "[[模型库/wav2vec 2.0|wav2vec 2.0]]"]
tasks: ["[[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]"]
datasets: [LibriTTS]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 1 个待确认实体页: [[Residual Vector Quantization]]✓, [[Speech Tokenizer]]✓, [[LLM-based TTS]]✓, [[Semantic vs Acoustic Tokens]]✓, [[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]✓, [[Token Rate and Bitrate Trade-offs]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: BridgeCode/BridgeTTS 位于 LLM-based TTS 的 AR codec language model 路线上,直接针对 AR TTS 中 token rate 与合成质量的固有 trade-off。在 [[Token Rate and Bitrate Trade-offs]][待确认] 概念页记录的典型参数范围中,现有 AR TTS 系统的 token rate 通常在 25-50 Hz (CosyVoice 25 Hz, VALL-E/EnCodec 50 Hz),BridgeTTS 的 10 Hz 是已知最低的 AR TTS token rate。

**已有认知**:
- [[Residual Vector Quantization]] 覆盖了 RVQ 的多种变体 (GVQ/MSRVQ/CSRVQ 等),BridgeCode 使用的 hierarchical RVQ (将 2304d 向量分 3 组,每组 3 级 RVQ) 接近 GVQ 的分组量化思路,但独特之处在于只保留每组第一层 index 实现极端压缩。
- [[Speech Tokenizer]] 中的 continuous VAE tokenizer 路线 (LatentLM/CLEAR/VibeVoice) 是解决 rate-quality trade-off 的另一条路线——完全绕过离散量化,BridgeCode 则保留离散 token 但通过 bridging module 恢复连续特征,是一种"两者共存"的折中。
- [[Semantic vs Acoustic Tokens]] 的核心二分法在 BridgeCode 中以另一种形式出现: sparse tokens (压缩自监督表征) vs dense continuous features (完整声学),通过 SparseBridge/DenseBridge 双向转换。

**创新判断**: BridgeCode 的核心创新不是设计新 codec/tokenizer,而是在已有表征 (wav2vec 2.0 features) 基础上增加双向 bridging 模块,使 AR 循环内可以用 10 Hz sparse tokens 做高效预测,同时通过 DenseBridge 恢复 50 Hz dense features 供 vocoder 使用。feature loss 作为补充监督信号的思路与 MELLE (连续 mel 预测) 异曲同工,但 BridgeTTS 保留了离散+连续双重训练目标。相比 [[LLM-based TTS]] 中记录的 Hybrid 架构 (LLM + Flow),BridgeTTS 不使用 flow/diffusion 后处理,而是直接用 learned bridging module 实现 sparse-to-dense 转换。

> 检索命中: [[Residual Vector Quantization]]✓, [[Speech Tokenizer]]✓, [[LLM-based TTS]]✓, [[Semantic vs Acoustic Tokens]]✓, [[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]✓ | 过滤: [[Token Rate and Bitrate Trade-offs]](pending-review) | 未命中但可能相关: [[Codec Language Model]]

## 速查

> [!summary] 速查
> - **一句话**: 提出 BridgeCode 双表征范式 (10 Hz sparse tokens + 50 Hz dense continuous features) 和双向 bridging module,让 AR TTS 以 5 倍更少的迭代步数达到竞争力质量
> - **路线**: wav2vec 2.0 连续特征 → SparseBridge (multi-scale conv + 5x downsample + hierarchical RVQ + code selection) → 10 Hz sparse tokens → GPT-2 AR 预测 → DenseBridge (code prediction + RVQ decoder + upsample + inverse conv) → 50 Hz dense features → HiFi-GAN vocoder → 波形
> - **指标**: LibriTTS test: WER 4.9% (最优) / SMOS 4.01 / QMOS 4.11 / UTMOS 3.894; token rate 10 Hz (CosyVoice 25 Hz, VALL-E 50 Hz); RTF 0.37x (2.7x 加速 vs baseline AR) [Table 1, Table 3]
> - **可借鉴**: (1) 在 AR 循环中输入 5 帧连续特征预测下一个 token——让模型看到比 token 更丰富的上下文; (2) feature loss 作为 cross-entropy 的补充监督,惩罚声学距离而非均匀 token 错误; (3) 层间对齐 (layer-wise alignment) 确保 SparseBridge/DenseBridge 中间表征一一对应
> - **局限**: 仅在 LibriTTS (585h English) 上验证,未测试大规模数据/多语言; 质量与 CosyVoice (25 Hz) 持平但未超越; 未开源; 无 SEED-TTS-Eval 等标准 benchmark 结果

## 核心问题

1. AR TTS 的 token rate-quality trade-off: 降低 token rate 减少 AR 迭代次数加速推理,但低 rate token 信息不足导致合成质量下降——如何"鱼与熊掌兼得"? [§1]
2. Cross-entropy loss 的 supervision mismatch: 语音 token 空间中相邻 token 声学差异很小,但 CE loss 对所有 token 错误施加等量惩罚——如何为语音 token 提供更精细的监督? [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

BridgeTTS 由两个阶段组成 [§2, Fig 2-3]:

**阶段 1: BridgeCode 训练** — 学习 sparse tokens ↔ dense features 的双向转换
- 输入: wav2vec 2.0 (frozen) 提取的 F0 ∈ R^{T×768} 连续特征 (来自 GPT-Talker 的预训练 encoder) [§2.1]
- SparseBridge: F0 → multi-scale conv (kernel 1/3/5) → F1 ∈ R^{T×2304} → 5x temporal downsample → F2 ∈ R^{T/5×2304} → hierarchical RVQ (分 3 组 × 3 级 RVQ) → code selector (仅保留每组第一层 index) → 3 个 sparse tokens per frame [§2.1]
- DenseBridge: sparse tokens → code predictor (预测缺失的 2nd/3rd RVQ codes) → hierarchical RVQ decoder → upsample → multi-scale inverse conv → 重建 dense features [§2.1]
- 训练损失: L_total = L_code + L_feat + L_adv [Eq. 1]

**阶段 2: BridgeTTS AR Generator 训练** — 在 sparse token 空间中做高效 AR 生成
- GPT-2-based AR 模型,输入 5 帧连续特征预测下一个 sparse token [§2.2]
- 训练损失: L_AR = L_token (CE on sparse tokens) + L_features (MSE on dense features) [§2.2]
- 推理时: AR 逐步预测 sparse token → DenseBridge (frozen) 转换为 dense features → 反馈到 AR 输入 → 直到 EOS → HiFi-GAN 合成波形 [§2.2, Fig 3(B)]

### 关键设计选择

**为什么用 hierarchical RVQ 而非标准 RVQ?**
[论文原文] 将 2304d 特征分为 3 组 768d,每组独立 3 级 RVQ,实现了对高维特征的分组量化。code selector 仅保留每组第一层 index (共 3 个 token/frame),基于 VALL-E 的发现——"only the first RVQ indices are crucial while other indices can be discarded without significant information loss" [§2.1]。
[agent 解读] 这种分组+只取第一层的策略兼顾了表达力 (3 个 token 覆盖 2304d) 和压缩率 (丢弃 6 个辅助层 index),代价是需要 DenseBridge 的 code predictor 来恢复缺失信息。

**为什么 AR 输入是 5 帧连续特征而非 token?**
[论文原文] "This enables the AR model to make more informed predictions based on richer contextual information" [§2.2]。同时 AR 模型可以"observe the continuous features directly used for speech generation when predicting the next token, allowing it to adjust output tokens to control the synthesis of subsequent continuous features" [§2.2]。
[agent 解读] 这打破了传统 AR 的"token-in, token-out"范式。输入连续特征而非离散 token 消除了量化误差在 AR 链中的累积: 每步输入的是 DenseBridge 重建的高保真特征,而非有损的 discrete token embedding。5 帧对应 sparse token 的一个时间步 (因为 5x downsample),所以每步 AR 实际上看到了完整的原始时间分辨率上下文。

**为什么需要 layer-wise alignment?**
[论文原文] 灵感来自 VDVAE,通过在 SparseBridge 和 DenseBridge 的中间层之间强制对齐,确保"high-fidelity bidirectional conversion between sparse tokens and dense continuous features" [§2.1]。
[agent 解读] 这种层间约束可以理解为一种正则化: 不仅要求最终输出匹配,还要求中间表征空间对齐,减少了 DenseBridge 在恢复缺失 RVQ codes 时的自由度,降低了重建误差的传播。

**为什么加 feature loss (L_features)?**
[论文原文] "token loss computes prediction accuracy through cross-entropy loss, treating any mismatch between predicted and ground truth tokens as equally incorrect, regardless of their acoustic similarity" [§2.2]。feature loss 通过 MSE 在连续特征空间计算距离,"providing fine-grained, hierarchical supervision" [§2.2]。
[agent 解读] 这解决了离散 token 空间中度量学习的缺失: CE loss 只区分对错,feature loss 提供了"距离"概念——预测声学相近的 token 比声学远离的 token 受到更小的惩罚。

### 训练策略

**BridgeCode 阶段** [§3.2]:
- wav2vec 2.0 Base 权重 frozen
- SparseBridge + DenseBridge 联合训练 700K steps
- A800 GPU, batch size 16, AdamW, lr=1e-4, decay 0.9991/8 per epoch
- HiFi-GAN vocoder 同时训练 (L_adv 约束)

**BridgeTTS AR Generator 阶段** [§3.2]:
- SparseBridge + DenseBridge frozen
- GPT-2-based AR generator 训练 600K steps
- 相同训练条件

## 实验

| 指标 | BridgeTTS | CosyVoice | GPT-Talker | VALL-E | UniAudio | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Token Rate (Hz) ↓ | **10** | 25 | 50 | 50 | 50 | / | — | [Table 1] |
| WER ↓ (dev) | **3.4%** | 6.8% | 5.9% | — | 11.4% | 2.3% | LibriTTS dev | [Table 1] |
| WER ↓ (test) | **4.9%** | 8.0% | 16.4% | 18.5% | 12.9% | 3.1% | LibriTTS test | [Table 1] |
| SMOS ↑ (dev) | 4.07 | **4.13** | 3.78 | — | 3.81 | 4.41 | LibriTTS dev | [Table 1] |
| SMOS ↑ (test) | 4.01 | **4.12** | 3.78 | 3.64 | 3.62 | 4.33 | LibriTTS test | [Table 1] |
| QMOS ↑ (dev) | 4.15 | **4.36** | 3.96 | — | 3.92 | 4.41 | LibriTTS dev | [Table 1] |
| QMOS ↑ (test) | 4.11 | **4.29** | 3.84 | 3.49 | 3.83 | 4.32 | LibriTTS test | [Table 1] |
| UTMOS ↑ (dev) | 4.050 | **4.253** | 3.693 | — | 3.676 | 4.258 | LibriTTS dev | [Table 1] |
| UTMOS ↑ (test) | 3.894 | **4.148** | 3.566 | 2.728 | 3.663 | 4.275 | LibriTTS test | [Table 1] |
| RTF ↓ | **0.37x** | — | 1x (baseline) | — | — | — | — | [Table 3] |

### 消融实验 [Table 2, LibriTTS test]

| 配置 | WER ↓ | SMOS ↑ | QMOS ↑ | UTMOS ↑ |
| --- | --- | --- | --- | --- |
| BridgeTTS (full) | **4.9%** | **4.01** | **4.11** | **3.894** |
| w/o DenseBridge | 13.8% | 3.74 | 3.74 | 3.443 |
| w/o L_features | 7.1% | 3.92 | 3.96 | 3.471 |

**关键观察**:
1. **DenseBridge 是核心组件**: 去掉 DenseBridge 后 WER 从 4.9% 暴涨到 13.8%,证明仅用压缩 token 做 AR 生成(缺少 rich context 输入)会严重退化 [Table 2, agent 解读]
2. **Feature loss 提供实质改善**: 去掉 L_features 后 WER 从 4.9% 升到 7.1%,UTMOS 从 3.894 降到 3.471,说明连续特征空间的 MSE 监督确实比纯 CE loss 提供了更有效的梯度信号 [Table 2, agent 解读]
3. **BridgeTTS vs CosyVoice**: BridgeTTS 在 WER 上显著优于 CosyVoice (4.9% vs 8.0%),论文归因于更低的 token rate 减少了 AR 推理中的 error accumulation [论文原文, §3.3]; 但 SMOS/QMOS/UTMOS 低于 CosyVoice,CosyVoice 使用了"updated codec models, advanced AR architectures, and larger training datasets" [论文原文, §3.3]
4. **速度优势明显**: RTF 0.37x 意味着 2.7 倍加速,直接源于 5x 更低的 token rate [Table 3]

## 局限性

1. **实验规模有限**: 仅在 LibriTTS (585h English) 上验证,未测试大规模数据 (如 Emilia 100K+ h) 或多语言场景,无法判断 BridgeCode 在更大规模下的扩展性 [agent 解读]
2. **质量未超越 CosyVoice**: 在 SMOS/QMOS/UTMOS 上均低于 CosyVoice (25 Hz),论文将差距归因于 CosyVoice 的"更先进 codec 和更大训练数据",但这也暗示 10 Hz 的极端压缩确实丢失了部分声学信息 [Table 1]
3. **无标准 benchmark 结果**: 缺少 SEED-TTS-Eval、LibriSpeech test-clean 等社区标准评测,限制了与更多近期方法的可比性 [agent 解读]
4. **Speaker Similarity 度量不足**: 仅用 SMOS (主观评测),缺少客观的 speaker embedding 余弦相似度 (如 ERes2Net/WavLM-based),难以量化 speaker cloning 能力 [agent 解读]
5. **未开源**: 无代码/模型公开,复现难度高 [agent 解读]
6. **Feature encoder 依赖**: 依赖 GPT-Talker 预训练的 wav2vec 2.0 frozen encoder,如果 feature 质量有上限,BridgeCode 的恢复质量也受限 [agent 解读]

## 点评

BridgeCode 提出了一个简洁优雅的方案来解决 AR TTS 的 token rate-quality trade-off: 不去设计新的低帧率 codec (如 WavTokenizer 40 Hz 单码本),而是在已有的 50 Hz dense representation 之上学习一个 10 Hz sparse representation,通过 bridging module 实现双向转换。这种"既离散又连续"的双表征思路在概念上与 LatentLM/CLEAR 的 continuous tokenizer 路线互补: 后者完全放弃离散 token,BridgeCode 则让 discrete tokens 和 continuous features 在 AR 循环中共存。

论文最大的贡献在于实验验证了两个直觉: (1) 即使压缩到 10 Hz,AR 模型只要能看到 DenseBridge 恢复的 rich features 就能保持合成质量; (2) feature loss 确实比纯 CE loss 提供更好的监督信号 (UTMOS 3.894 vs 3.471)。但实验的规模和范围是明显短板——仅 LibriTTS 585h,缺少与 Llasa/Qwen3-TTS/Spark-TTS 等最新 AR 系统的对比,也没有流式/streaming 场景的讨论。

## 可复用的 idea

1. **AR 循环中输入连续特征而非 token**: 打破"token-in token-out"的 AR 范式,让每步预测基于更丰富的上下文,同时避免量化误差在 AR 链中累积。这个思路可以推广到任何 AR codec LM——只要有一个 token→feature 的 bridge 模块。

2. **Feature loss 作为 CE loss 的补充监督**: 对于语音 token 这类"连续空间中的离散化"问题,在 token-level CE loss 之外加 feature-level MSE loss,为模型提供声学距离感知。适用于任何基于离散 token 的语音生成系统。

3. **Hierarchical RVQ + code selection**: 将高维特征分组 RVQ 后只保留每组第一层 code,实现极端压缩,丢失的信息由 code predictor 恢复。这种"先压缩再恢复"的设计可用于任何需要降低 AR 序列长度的场景。

> [!review] 审阅: pass-with-fixes (2026-06-04)
> - **结论**: pass-with-fixes, 2 low issues, 0 medium/high
> - **issue 1** (low, fact-inference-mixing): KB 背景中 sparse tokens 描述用词已修正 (压缩语义→压缩自监督表征)
> - **issue 2** (low, template-compliance): frontmatter models 未列无页面的基线模型 (VALL-E/UniAudio/GPT-Talker),可接受
> - 详见 `_review/BridgeCode-review.yml`
