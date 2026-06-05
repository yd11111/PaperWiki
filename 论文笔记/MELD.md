---
type: paper
tier: deep
title: "MELD: Mel-Spectrogram-Based Speech Language Modeling with Discrete Latent Variables"
arxiv_id: "2605.29859"
source: "Sources/MELD.pdf"
authors: [Sung-Lin Yeh, Wei Zhou, Gil Keren, Duc Le, Zhong Meng, Hao Tang, Jay Mahadeokar, Ozlem Kalinli, Alexandre Mourachko]
year: 2026
venue: "arXiv"
tags: [TTS, STT, zero-shot, autoregressive, mel-spectrogram, discrete-latent, variational-inference, joint-modeling, codec-free]
concepts: ["[[MelSpectrogram]]", "[[SpeechLanguageModel]]", "[[CodecLanguageModel]]", "[[VariationalAutoencoderforTTS]]", "[[ResidualVectorQuantization]]"]
models: ["[[MELLE]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["LibriSpeech"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: MELD 处于 Speech Language Model 领域中 **mel-spectrogram-based AR 建模** 这一分支。当前 SpeechLM 的主流路线是 Codec Language Model (codec-based LM),使用 RVQ 产生的离散 tokens 做 next-token prediction [CodecLanguageModel]。MELD 不走 codec 路线,而是直接在连续 mel-spectrogram 上建模,引入离散隐变量作为采样中介,与 MELLE 形成同一分支内的直接竞争。

**已有认知**:
- [[SpeechLanguageModel]]: 端到端处理语音的自回归模型,核心三组件为 tokenizer + LM + vocoder。MELD 的贡献在于消除独立 tokenizer 阶段,将编码器与 AR 模型联合优化。
- [[CodecLanguageModel]] [待确认]: codec-based LM 使用 RVQ 多层码本,面临序列长度爆炸和多层建模复杂性问题。MELD 每步仅预测单个隐变量 token,避免了这些问题。
- [[MelSpectrogram]] [待确认]: mel-spectrogram 是 TTS 中最广泛使用的中间声学特征,但丢失相位信息,且 AR 建模时容易陷入静音循环。
- [[MELLE]] [待确认]: 首个在连续 mel 空间做 AR 建模的零样本 TTS,使用高斯采样,是 MELD 的最直接 baseline。
- [[ResidualVectorQuantization]]: RVQ 是 codec 路线的核心量化模块,MELD 明确不采用 RVQ,转而使用单层 soft VQ + 冻结 k-means codebook。
- [[Zero-shotSpeechSynthesis]]: 给定短参考语音合成保持音色的语音,当前 SOTA 系统 (CosyVoice 3, IndexTTS2) 主要使用 codec tokens + LLM。

**创新判断**: 相对于 MELLE 的高斯采样 + ad-hoc 目标函数,MELD 提出在离散隐空间做采样 + 严格的变分下界目标,解决了 mel AR 建模的静音循环问题。相对于 codec-based LM,MELD 通过联合优化消除了 encoder 与 AR 模型之间的梯度断裂,且无 RVQ 多层建模复杂性。

> 检索命中: [[SpeechLanguageModel]]✓, [[ResidualVectorQuantization]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[MelSpectrogram]](pending-review), [[MELLE]](pending-review) | 未命中但可能相关: FiniteScalarQuantization (dMel 对比)

## 速查

> [!summary] 速查
> - **一句话**: 在 mel-spectrogram 上引入离散隐变量做变分自回归建模,联合优化编码器和 AR 模型,同时解决 mel AR 的静音循环问题和 codec 两阶段的信息损失问题
> - **路线**: text(BPE) + mel prompt → decoder-only Transformer → 预测离散隐变量 z_t (K=8192 soft VQ) → 重建 mel frame x_t (MLP+PostNet) → HiFi-GAN vocoder → waveform
> - **指标**: TTS WER 2.4/1.9 (Conformer/Whisper), SIM 0.872, SMOS 3.89 on LibriSpeech test-clean [Table 2,4]; STT WER 4.2/10.0 (clean/other) [Table 6]; 均优于 Codec-LM 和 MELLE
> - **可借鉴**: (1) 冻结 k-means codebook 做 soft VQ,避免 VQ 训练不稳定; (2) 将 EOS 合并到离散词表,免去 stop predictor; (3) 离散采样天然支持 repetition penalty 抑制静音循环
> - **局限**: 仅在 LibriSpeech 960h 上验证; 依赖 HiFi-GAN vocoder (与 codec decoder 不完全可比); 未探索更多语音任务 (QA, 翻译等); 200M 参数较小

## 核心问题

当前 speech language model 普遍采用两阶段策略: 先训练独立的 speech codec/VAE 提取表征,再在这些表征上训练 AR 模型。这造成两个问题:

1. **编码器与下游任务脱节**: 独立训练的 encoder 不知道下游任务需要保留什么信息,导致离散化表征可能丢失任务关键信息 [§1]
2. **Mel AR 建模的采样困境**: 直接在连续 mel 空间做 AR 建模缺乏良定义的采样分布,导致生成时陷入无限静音或产生恒定伪影 [§1, §2]

MELD 的核心思路是: 在 mel-spectrogram 上引入离散隐变量,通过变分框架联合优化编码器和 AR 模型,同时获得离散采样的可控性和连续 mel 空间的信息完整性。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MELD 的生成过程分解为两步 [§2.1, Fig 1]:

$$p(x_t, z_t | x_{<t}, y) = p(x_t | z_t, x_{<t}, y) \cdot p(z_t | x_{<t}, y)$$

其中 $x_t$ 是当前 mel frame, $z_t$ 是离散隐变量, $x_{<t}$ 是历史 mel frames, $y$ 是文本 BPE tokens。

三个核心组件 [§3, Fig 3]:

1. **Quantization Network** $q(z_t | x_t)$: 将当前 mel frame 软量化为离散隐变量。使用冻结的 k-means codebook (K=8192, 从 mel frames 初始化),基于欧氏距离的 softmax 分配 [§3.1, Eq 6]。仅在训练时使用,推理时不可访问未来帧。

2. **Autoregressive Latent Predictor** $p(z_t | x_{<t}, y)$: decoder-only Transformer (12 层, 1024 dim, 16 heads, ~200M params) 预测下一个离散隐变量。Mel frames 通过 3 层 MLP ($g_{Mel}$) 编码为 $d_{model}$ 维 embedding [§3.2]。

3. **Reconstruction Network** $p(x_t | z_t, x_{<t}, y)$: 从隐变量嵌入 $c_{z_t}$ + Transformer 输出 $h_t$ 重建 mel frame。包含 Linear + MLP 残差 + conv PostNet (Tacotron 2 风格) [§3.3, Eq 7-8]。

### 关键设计选择

**为什么用离散隐变量而非连续高斯 (vs MELLE)?**
[论文原文] MELLE 的高斯采样空间受限于 reverse KL divergence 的性质 -- 当先验 $\mathcal{N}(x_t, I)$ 对应静音帧时,采样无法逃离静音区域。而离散隐空间包含静音和非静音码字,top-k/top-p 采样可以跳出静音循环 [§5.3]。

**为什么冻结 codebook?**
[论文原文] 用 k-means 初始化后冻结 codebook,让 reconstruction network 来精细化重建,避免了 VQ 训练中的梯度估计难题 (straight-through estimator 等) [§3.1, citing Huh et al. 2023]。
[agent 解读] 这是一个巧妙的权责分离: codebook 负责提供离散采样的"锚点",重建网络负责精确恢复细节,两者解耦降低了训练难度。

**为什么 EOS 合并到离散词表?**
[论文原文] MELD 将 <EOS> 作为离散码词的一部分,与 BPE tokens 和 K 个隐变量共享 softmax 预测,因此不需要额外的 stop predictor [§2.3]。这与 MELLE/Mel-LM 必须训练单独的 stop predictor 形成对比。
[agent 解读] 这简化了架构,也使停止决策参与了端到端优化,可能减少过早/过晚停止的问题。

**为什么用 BPE 而非 phoneme?**
[论文原文] BPE tokenization (|V_text|=4096) 使同一模型可同时处理 TTS 和 STT 任务,而 phoneme 需要额外的 G2P 工具 [§5.1]。实验显示 phoneme → BPE 切换仅导致 speaker similarity 微降 (~0.008) [Table 2]。

**变分目标 vs MELLE 的 ad-hoc 目标**:
[论文原文] MELD 的训练目标是严格的 Variational Lower Bound (VLB) [§2.2, Eq 3]:
$$\mathcal{L}_{VLB} = \sum_t \left[ KL(q(z_t|x_t) \| p(z_t|x_{<t}, y)) - \mathbb{E}_{z_t \sim q} \log p(x_t | z_t, x_{<t}, y) \right]$$
而 MELLE 的目标是 ad-hoc 设计的多项损失组合 (regression + KL + stop + flux) [§4, Appendix A.6]。

**Joint TTS-STT 建模**:
[论文原文] TTS 和 STT 共享同一个 AR 网络,通过特殊 token (<TTS>/<STT>) 控制任务模式 [§2.3, Fig 2]。统一词表 $V = V_{text} \cup V_{latent}$。训练策略: 先 TTS-only 80k steps 收敛,再混合 50/50 TTS+STT 继续训练 [§5.6]。

### 训练策略

- **Loss**: $\mathcal{L}_{VLB}$ (KL + MSE reconstruction) + $\mathcal{L}_{slow}$ (slowness penalty, weight 0.2) [§3.4, Eq 9]
- **Slowness penalty**: $-\frac{1}{T-1} \sum_{t=1}^{T-1} \| \hat{x}_t - \hat{x}_{t+1} \|_2$,惩罚预测帧变化过慢,促进多样性 [§3.4]
- **Codebook**: K=8192, 从训练数据 mel frames 的 k-means 聚类初始化,训练全程冻结 [§5.1]
- **Mel encoder $g_{Mel}$**: 3 层 MLP, dim=1024, GELU + dropout 0.5, 训练和推理均使用 dropout (test-time dropout 缓解 train-inference mismatch) [§5.1]
- **推理采样**: top-p (p=0.9) + top-k (k=60) + repetition penalty (对前一步 top-p 候选码词施加 -1 分) [§5.2]
- **训练**: 200k steps, 16x V100, Adam, 最大 50k frames/batch [Appendix A.2]

## 实验

| 指标 | MELD | Codec-LM (BPE) | MELLE | Mel-LM | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| TTS WER (Conf/Whisper) | 2.4/1.9 | 5.3/4.8 | 4.8/4.2 | 4.7/4.2 | LS test-clean | [Table 2,3] |
| TTS SIM | 0.872 | 0.864 | 0.826 | 0.825 | LS test-clean | [Table 2,3] |
| SMOS | 3.89 | 3.72 | - | - | LS test-clean (43 samples) | [Table 4] |
| CMOS (vs MELD) | 0.0 | -0.31 | - | - | LS test-clean | [Table 4] |
| STT WER clean/other | 4.2/10.0 | 6.4/16.4 | - | - | LS test-clean/other | [Table 6] |
| STT WER (260M) | 3.5/9.2 | - | - | - | LS test-clean/other | [Table 6] |
| Joint TTS WER | 2.8/2.2 | - | - | - | LS test-clean | [Table 7] |
| Joint STT WER clean/other | 4.9/12.1 | - | - | - | LS test-clean/other | [Table 7] |

**关键发现**:

1. **MELD 在 TTS 和 STT 上均显著优于 MELLE 和 Codec-LM**: TTS WER 降低约 50% (vs MELLE) 和 55% (vs Codec-LM),同时 SIM 提升 0.046 (vs MELLE) [Table 2,3]。

2. **离散隐空间有效抑制静音循环**: MELLE 和 Mel-LM 频繁出现长静音,即使使用 slowness penalty 也无法完全消除。MELD 的离散采样 + repetition penalty 有效解决了这一问题 [§5.3]。

3. **离散隐变量对推理必不可少**: 将 $c_{z_t}$ 置零时 WER 飙升至 52.3/51.7, SIM 降至 0.520,且合成时长远超 GT (>200 min vs 131.8 min),说明隐变量不是训练 artifact 而是实际编码了关键信息 [Table 5]。

4. **Repetition penalty 有效但非必须**: 去除后 WER 仅升 0.7%,但合成时长增加 ~6 min,deletion error 显著上升 [Table 5]。

5. **联合优化对 STT 的改进更显著**: MELD 在 STT 上比 dMel 低 0.4% (test-other) [Table 6],在 joint TTS-STT 模式下比 dMel joint 低 3.2% (test-other) [Table 7]。

6. **Codebook 初始化对 Codec-LM 的 STT 至关重要**: 无初始化的 Codec-LM STT WER 超过 100%,表明 codec token 空间难以从零学习语义对齐 [Table 6]。

## 局限性

1. **公平对比困难**: MELD 用 HiFi-GAN vocoder, Codec-LM 用 DAC decoder,不同声码器引入了混淆因素 [§7]。

2. **MELLE 复现不完全**: 作者按原文配置复现 MELLE 但未达到原始报告性能,怀疑 VAD 预处理步骤是关键但未在原文中强调 [§7, Appendix A.6]。

3. **规模有限**: 仅在 LibriSpeech 960h 上验证,200M 参数,未测试大规模数据或更大模型。当前 SOTA 系统使用 100K+ 小时数据和 1B+ 参数。

4. **任务覆盖有限**: 只验证了 TTS 和 STT,未探索 QA、翻译、情感等更广泛的语音任务 [§7]。

5. **Speaker similarity 仍有提升空间**: SIM 0.872 在 LibriSpeech 内 benchmark 尚可,但与当前 SOTA 系统使用不同评估体系 (不同数据集、不同 speaker embedding 模型),无法直接比较。

## 点评

MELD 的核心贡献是在 mel-spectrogram AR 建模中引入了一种优雅的解决方案: 用离散隐变量同时解决采样困难和信息损失两个问题。

**最有价值的洞察**: 静音循环问题的根因不是采样策略而是采样空间的拓扑。高斯空间中静音区域是连通的、难以逃离的 [§5.3 的分析]。离散空间将连续流形打碎为离散码字,top-k/top-p 采样天然具有"跳出当前模式"的能力。这个洞察对理解 codec-based 和 mel-based 方法各自的优劣很有价值。

**冻结 codebook 的巧妙之处**: 将 VQ 训练问题完全回避,让 k-means 提供一个"足够好"的离散化锚点,reconstruction network 负责精细恢复。这是一种实用主义的工程选择,降低了训练复杂度。

**Joint TTS-STT 的意义**: 证明了在同一个 mel-spectrogram AR 框架下可以同时做 TTS 和 STT,且相比独立训练只有很小的性能折损 (TTS WER +0.4/+0.3, STT WER +0.7/+2.1)。在 LibriSpeech 960h / 200M 规模下,这暗示 mel-spectrogram 作为中间表征可能比 codec tokens 更适合多任务统一建模 [agent 解读]。

**不足**: 实验规模偏小 (960h, 200M),在大数据大模型时代说服力有限。与 MELLE 的对比可能受到复现质量的影响。没有与 flow-matching 等非 AR 方法对比。

## 可复用的 idea

1. **冻结 k-means codebook 做 soft VQ**: 避免 VQ 训练不稳定问题,可应用于任何需要离散化连续表征的场景。核心是将"离散化"和"重建"解耦,让 codebook 只提供粗粒度锚点。

2. **EOS 合并到离散码词空间**: 将停止决策与内容预测统一到同一个 softmax,消除独立 stop predictor 的需求。适用于任何 AR mel-spectrogram 或 AR latent generation 系统。

3. **Repetition penalty 抑制静音循环**: 对前一步 top-p 候选码词施加固定惩罚分 (-1),简单有效地减少重复/静音。可迁移到任何使用离散 token 的 AR 生成系统。

4. **变分框架统一 TTS+STT**: 通过将 BPE tokens 和离散隐变量合并到统一词表,用特殊 token 切换任务模式,实现单模型多任务。训练策略为 TTS-first (80k) + joint fine-tuning。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 因果解释充分,每个设计选择有 WHY 分析,可借鉴具体 |
> | 可信赖 | pass | 关键数字全部交叉验证通过,出处标注覆盖>90% |
> | 可区分 | pass | [论文原文]/[agent 解读]标注一致,覆盖>80% |
> | 可定位 | pass | KB 背景定位清晰,创新判断有对比基准 |
> | 不污染 | pass | 无新建概念页需求,反向更新均为追加 |
> 
> Issues: 4 (high: 0, medium: 2, low: 2)
> 详见 `_review/MELD-review.yml`
