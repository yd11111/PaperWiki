---
type: paper
tier: deep
title: "StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs"
arxiv_id: ""
source: "Sources/StableToken.pdf"
authors: [Anonymous]
year: 2026
venue: "Under review at ICLR 2026"
tags: [speech-tokenizer, noise-robustness, voting-LFQ, consensus-training, SpeechLLM, token-stability, multi-branch-quantization]
concepts: ["[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[SpeechLanguageModel]]", "[[ResidualVectorQuantization]]", "[[Self-SupervisedSpeechRepresentation]]", "[[FiniteScalarQuantization]]"]
models: ["[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeechTokenizer]], [[SemanticvsAcousticTokens]], [[SpeechLanguageModel]], [[ResidualVectorQuantization]])
> 检索命中: [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechLanguageModel]]✓, [[ResidualVectorQuantization]]✓ | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[FiniteScalarQuantization]](pending-review), [[Gumbel-Softmax]](pending-review) | 未命中但可能相关: 无

**[[SpeechTokenizer]]**: StableToken 属于第二类"监督式 semantic tokenizer" — 通过 ASR 监督在 encoder 中插入量化层, 直接优化语义编码。与 CosyVoice 系列的 S3/FSQ tokenizer 同族, 但 StableToken 的核心创新在量化架构 (Voting-LFQ 替代 FSQ/VQ) 和训练策略 (Noise-Aware Consensus Training)。KB 显示现有 semantic tokenizer 在 SpeechLM 中被广泛采用 (CosyVoice, GLM-4-Voice, S^3 Tokenizer), 但无一专门解决噪声鲁棒性问题。

**[[SemanticvsAcousticTokens]]**: StableToken 是纯 semantic tokenizer (单码本, 25Hz, 8192 entries), 不编码声学细节。其核心贡献是发现现有 semantic tokenizer 的 token stability 问题: 微小声学扰动 → 剧烈 token 变化 → 下游 SpeechLLM 性能退化。这揭示了 semantic tokens 的一个被忽视的脆弱性。

**[[SpeechLanguageModel]]**: StableToken 直接面向 SpeechLLM 下游任务优化。KB 显示 SpeechLM 高度依赖 tokenizer 的 token 质量 — token 不稳定会破坏 speech-text alignment, 迫使 LLM 从不一致的输入中学习。StableToken 在 ASR (CHiME-4), SER, TTS (SEED-TTS) 三大下游任务上验证了 token stability 对 SpeechLLM resilience 的直接贡献。

**[[ResidualVectorQuantization]]**: StableToken 不使用 RVQ, 而是使用 LFQ (Look-up Free Quantization) 的多分支变体 Voting-LFQ。与 RVQ 的层级量化残差不同, Voting-LFQ 通过 n 个并行分支独立二值化投影, 再 bit-wise majority vote 合并。这是一条与 RVQ 平行的量化路线, 通过冗余和投票实现鲁棒性而非精度。

## 速查

> [!summary] 速查
> - **一句话**: 首个系统性解决 semantic speech tokenizer 噪声脆弱性的工作, 通过 Voting-LFQ (多分支 bit-wise majority vote 量化) + Noise-Aware Consensus Training 将 token 稳定性 (UED) 降低 >60% 相对, 同时保持 SOTA 重建质量和下游 SpeechLLM 性能 [论文原文]
> - **路线**: Speech (16kHz) → Whisper-large-v3 Encoder (frozen bottom, trainable top) → Average Pooling (25Hz) → n=5 parallel linear projections (d=13 dims each) → Sign binarization → Bit-wise majority vote → Token index (0 to 2^13-1 = 8192) → Text Decoder (CTC loss); Training: majority branches get clean input (H_clean), minority branches get perturbed input (H_perturbed) → Consensus Loss [§2, Fig.2]
> - **指标**: UED avg 10.17% (vs CosyVoice2 38.66%, GLM-4-Voice 28.62%, S^3 Tokenizer 33.09%) [Table 1]; WER: LS-clean 3.84 / LS-other 7.99 / SEED-en 3.44 / SEED-zh 2.62 (均为 SOTA 或 competitive) [Table 2]; 下游 ASR: CHiME-4 real 35.90% / sim 30.61% WER (vs CosyVoice 54.63% / 47.71%) [Table 3]; 下游 TTS: SEED-TTS WER 3.02% / MOS 4.08 ZH [Table 3]
> - **可借鉴**: (1) Voting-LFQ 多分支 bit-level voting — 低开销的量化鲁棒化方案, 可迁移到任何 LFQ/FSQ tokenizer; (2) Noise-Aware Consensus Training — 通过 clean majority + perturbed minority 的 multi-view 训练 + consensus loss 显式学习噪声不变性; (3) Token stability (UED) 作为 tokenizer 评估新维度
> - **局限**: 仅关注 semantic tokenizer (无 acoustic reconstruction); vocab 8192 比 CosyVoice2 (4096) 大, 使 token-level invariance 更难维持 (但仍 SOTA); 推理时所有 n 分支都用 clean input [§2.2]; 代码未开源

## 核心问题

现有 semantic speech tokenizer (CosyVoice, GLM-4-Voice, S^3 Tokenizer) 存在一个被忽视的关键脆弱性: **token 不稳定性** [§1]:

1. **问题**: 即使在高 SNR (语音完全可听) 下, 微小声学扰动就能导致输出 token 序列发生剧烈变化 [§1, Fig.1] [论文原文]
2. **根因 1 — 架构脆弱**: 单路径量化 (single-path quantization), 量化边界附近的微小扰动被不可逆地放大为完全不同的 output token [§1] [论文原文]
3. **根因 2 — 训练信号遥远**: 标准 ASR loss 仅监督最终文本转写, 对中间 token 表示的稳定性无感知 — 模型可以收敛到"功能正确但表征脆弱"的解 [§1] [论文原文]
4. **下游危害**: token 不稳定 → 破坏 speech-text alignment → 增加 LLM 学习负担 → 噪声环境下 SpeechLLM 性能骤降 [§1] [论文原文]

为什么简单方案不行 [§1]:
- **Offline ensemble**: 推理成本翻倍, 独立训练模型的量化边界无法对齐, token-level vote 粒度太粗 [论文原文]
- **Token-level consistency loss**: 离散 code 的梯度不稳定, 难以训练 [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

StableToken 基于 encoder-decoder ASR 架构 (initialized from whisper-large-v3), 用 Voting-LFQ 模块替代标准 single-path quantizer [§2.1, Fig.2]:

```
Speech → Whisper Encoder → Average Pooling (25Hz) → h ∈ R^D
  ↓
Voting-LFQ Module:
  Branch 1: p_1 = W_1·h + b_1 ∈ R^d → B_1 = sign(p_1) ∈ {-1,+1}^d
  Branch 2: p_2 = W_2·h + b_2 ∈ R^d → B_2 = sign(p_2)
  ...
  Branch n: p_n = W_n·h + b_n ∈ R^d → B_n = sign(p_n)
  ↓
  Training: s_final_j = (1/n) Σ (B_i)_j  (real-valued average) [Eq.2]
  Inference: B_final = sign(s_final)  (hard majority vote) [Eq.3]
  ↓
  Map B_final to token index k ∈ {0, ..., 2^d - 1} [§2.2]
  ↓
Encoder Layer → Projection → Text Decoder → CTC Loss (L_ASR)
```

### 关键设计选择

**1. Voting-LFQ Module [§2.2]**

核心: 用 n 个独立并行分支创建输入的 n 个"perspectives", 每个分支独立做 binary quantization, 然后通过 bit-wise majority vote 合并 [§2.2]:

- n 个 linear projections: p_i = W_i·h + b_i (R^D → R^d) [Eq.1]
- 二值化: B_i = sign(p_i) ∈ {-1, +1}^d, 使用 STE 反向传播 [§2.2]
- **训练时**: 实值平均 s_final_j = (1/n)Σ(B_i)_j → 提供梯度信号 [Eq.2]
- **推理时**: 硬投票 B_final = sign(s_final) → 确定性 token [Eq.3]

为什么 bit-level vote 优于 token-level vote: bit-level 操作粒度更细, 即使多数 branch 预测错误的 token, 只要 bit-level 错误分散 (sparse), 投票仍能恢复正确 token [§2.2, Table 6] [论文原文]

Case study [Table 6]: Position 80, 3/5 voters 产生错误 token, 但 bit-wise vote 在 bit#5 以 3:2 多数票选 '0', bit#7 以 4:1 多数票选 '1', 成功恢复原始 token 3485 [§4.3.3]

**为什么用 LFQ 而非 VQ/FSQ**: LFQ 的 binary 量化天然适合 bit-wise voting — 每个 bit 独立且只有两个取值, 多数投票有明确数学保证。VQ 的高维 codebook lookup 无法做如此精细的投票 [agent 解读]。

**2. Noise-Aware Consensus Training [§2.3]**

核心: 显式训练 tokenizer 对噪声不变:

1. 对 input audio w, 生成扰动版本 w' = A(w) (Gaussian noise etc.) [§2.3]
2. Encoder 分别处理 w 和 w' → clean hidden h 和 perturbed hidden h' [§2.3]
3. 随机选 k < n/2 个 branch 接收 h' (少数), 其余 n-k 个 branch 接收 h (多数) [§2.3]
4. 所有 branch 的 pre-quantization vectors p_i 与全局均值 p̄_all 做 L2 约束 [§2.3]:

$$\mathcal{L}_{\text{consensus}} = \frac{1}{n} \sum_{i=1}^{n} \|p_i - \bar{p}_{\text{all}}\|_2^2, \quad \bar{p}_{\text{all}} = \frac{1}{n} \sum_{j=1}^{n} p_j$$

为什么 clean majority 是锚点: 多数 branch 的 clean input 作为稳定参照, 全局均值 p̄_all 不会被少数 noisy branch 拉偏, 迫使 noisy branch 主动靠拢 clean consensus [§2.3] [论文原文]

为什么在 continuous p_i 上做 consensus 而非 binary B_i: continuous 空间梯度更平滑, binary 的 sign 函数梯度不稳定 [§2.3] [论文原文]

**3. Training Objective [§2.4]**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ASR}} + \lambda_1 \mathcal{L}_{\text{consensus}} + \lambda_2 \mathcal{L}_{\text{commitment}} + \lambda_3 \mathcal{L}_{\text{codebook}}$$

- L_ASR: Cross-entropy loss on ground-truth transcripts [§2.4]
- L_consensus: 上述多分支一致性约束 [§2.3]
- L_commitment: 鼓励 hidden states 靠近量化表征 [§2.4]
- L_codebook: 促进 codebook 均匀使用 (entropy maximization) [§2.4]

### 训练策略

- **初始化**: whisper-large-v3 (Radford et al., 2023), Voting-LFQ 插入 encoder mid-point [§3]
- **预训练数据**: 150K 小时多样语音语料 [§3]
- **Vocab**: 8192 (d=13 → 2^13) [§3]
- **Frame rate**: 25Hz [§3]
- **Voters**: N=5 (optimal, N=7 marginal gain) [§4.3.2, Table 5]
- **Noise augmentation**: Gaussian noise + 多种真实噪声 [Appendix B.3]

## 实验

### Tokenizer-Level: Noise Robustness (UED%) [Table 1]

| Model | Type | Vocab | Gauss | Pink | Brown | Bit Crush | Real | Real(OOD) | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CosyVoice2 | Supervised | 4096 | 54.67 | 42.57 | 39.96 | 30.87 | 31.76 | 32.13 | 38.66 |
| GLM-4-Voice | Supervised | 16384 | 42.44 | 32.12 | 30.22 | 25.53 | 27.67 | 28.62 | 31.10 |
| S^3 Tokenizer | Supervised | 4096 | 35.40 | 27.09 | 25.45 | 20.64 | 23.88 | 24.58 | 26.17 |
| **StableToken** | **Supervised** | **8192** | **12.93** | **9.76** | **9.37** | **7.32** | **10.65** | **10.96** | **10.17** |

UED (Unit Edit Distance) 降低 >60% 相对 vs best supervised baseline (S^3) [Table 1]。

### Tokenizer-Level: Reconstruction Quality [Table 2]

| Model | #C | Frame Rate | BPS | LS-clean WER↓ | LS-other WER↓ | SEED-en WER↓ | SEED-zh WER↓ | LS-clean MOS↑ | SEED-zh MOS↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CosyVoice2 | 1 | 25Hz | 325 | 4.25 | 9.68 | 4.34 | 2.75 | 3.25 | 3.58 |
| **StableToken** | **1** | **25Hz** | **325** | **3.84** | **7.99** | **3.44** | **2.62** | **4.09** | **4.18** |

### Downstream SpeechLLM Performance [Table 3]

| Tokenizer | ASR Real WER↓ | ASR Sim WER↓ | TTS SEED-EN WER↓ | TTS SEED-EN MOS↑ | TTS SEED-ZH WER↓ | TTS SEED-ZH MOS↑ |
| --- | --- | --- | --- | --- | --- | --- |
| CosyVoice | 54.63 | 47.71 | 7.80 | 3.52 | 8.73 | 3.47 |
| CosyVoice2 | 59.83 | 55.01 | 7.22 | 3.75 | 9.89 | 3.37 |
| GLM-4-Voice | 51.08 | 43.09 | 6.19 | **4.19** | 5.26 | 3.85 |
| **StableToken** | **35.90** | **30.61** | **4.43** | 4.12 | **3.02** | **4.08** |

### Component Ablation [Table 4]

| Configuration | Gauss UED↓ | Brown UED↓ | Real OOD UED↓ | LS-Clean WER↓ | LS-Other WER↓ |
| --- | --- | --- | --- | --- | --- |
| **StableToken (Full)** | **12.93** | **9.76** | **10.96** | **2.03** | **4.68** |
| w/o Consensus Loss | 24.80 | 19.06 | 17.43 | 2.03 | 4.88 |
| w/o Noise-Aware Training | 30.77 | 23.05 | 21.51 | 2.19 | 5.52 |
| w/o Multi-Branch | 34.53 | 25.44 | 24.47 | 2.39 | 5.85 |

每个组件都是关键的: Multi-Branch 是结构基础, Noise-Aware Training 是训练使能, Consensus Loss 是最终精化 [Table 4] [论文原文]。

## 局限性

1. **仅 semantic tokenizer**: StableToken 不编码声学细节, 需要外部 acoustic model (如 flow matching) 重建波形; 噪声鲁棒性仅针对 semantic token 层面 [agent 解读]
2. **Frame rate 25Hz**: 与 CosyVoice2 同帧率, 但高于 Mimi (12.5Hz) 和 FlexiCodec (3-12.5Hz), 在 LLM 序列效率方面仍有改进空间 [agent 解读]
3. **Noise augmentation 种类**: 训练时使用 Gaussian + 真实噪声, 但极端失真 (如 bit crush 在低 bit 下) 仍有改进空间 — 虽然已大幅优于 baselines [Table 1]
4. **未开源**: 代码和模型将在 acceptance 后发布 [Reproducibility Statement]
5. **Vocab 大小 trade-off**: 8192 vocab 比 CosyVoice2 (4096) 大, 在更细粒度的 decision space 上维持 token invariance 更困难 — 但 StableToken 仍显著优于所有 baselines [§4.1.1] [论文原文]

## 点评

**为什么这篇值得关注**: StableToken 首次揭示并系统性解决了 semantic speech tokenizer 的噪声脆弱性问题。这个问题之前被忽视, 但直接影响 SpeechLLM 在真实环境中的可靠性。Voting-LFQ 的 bit-level majority voting 是优雅且高效的解决方案。

**关键洞见**:
1. **Token stability 是被忽视的评估维度**: 现有 tokenizer 评估集中在 reconstruction quality (WER, MOS) 和 downstream task performance, 但 token stability (UED) 在噪声环境下与下游性能高度相关 [§1] [论文原文]
2. **Architecture-training co-design 是关键**: 多分支架构 (容错结构) + consensus 训练 (学习不变性) + noise-aware training (显式噪声暴露) 三者缺一不可, 消融证明去除任一组件 UED 急剧退化 [Table 4] [论文原文]
3. **Bit-level > Token-level**: 投票粒度从 token 降到 bit 带来本质提升 — 即使多数 branch 产生错误 token, 只要错误在 bit 层面分散, 仍可恢复 [Table 6 case study] [论文原文]
4. **Token stability 直接传导到下游**: StableToken 在 CHiME-4 ASR 上 WER 35.90 vs CosyVoice2 59.83, 差距随噪声增强而扩大 [Fig.3] — 这验证了"token 质量决定 SpeechLLM 上限"的论点 [论文原文]

**与 CosyVoice 系列的关系**: StableToken 可视为 CosyVoice tokenizer 的鲁棒化版本 — 同族架构 (Whisper encoder + quantizer + ASR loss), 但用 Voting-LFQ 替代 FSQ, 加入 Noise-Aware Consensus Training [agent 解读]。

**与 FSQ 的对比**: FSQ 用固定网格量化 (round + bound), 天然避免 codebook collapse 但无噪声鲁棒机制; Voting-LFQ 用多分支二值化 + majority vote, 在保留 lookup-free 优势的同时通过冗余和投票添加噪声容错 [agent 解读]。

## 可复用的 idea

1. **Voting-LFQ**: bit-level majority voting over n parallel branches — 可直接应用于任何 LFQ/FSQ tokenizer, 增加噪声鲁棒性, 推理开销可忽略 [§2.2]
2. **Noise-Aware Consensus Training**: clean majority + perturbed minority + consensus loss — 通用的噪声不变性训练范式, 可应用于 speech/image/video 的离散表征学习
3. **UED 作为 tokenizer 评估指标**: Unit Edit Distance 在多种噪声条件下量化 token stability — 应成为 tokenizer 标准评测的一部分

---

检索命中: [[SpeechTokenizer]], [[SemanticvsAcousticTokens]], [[SpeechLanguageModel]], [[ResidualVectorQuantization]] | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review), [[FiniteScalarQuantization]](pending-review), [[Gumbel-Softmax]](pending-review) | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
