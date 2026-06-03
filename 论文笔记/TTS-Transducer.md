---
type: paper
tier: deep
title: "TTS-Transducer: End-to-End Speech Synthesis with Neural Transducer"
arxiv_id: "2501.06320"
source: "Sources/TTS-Transducer.pdf"
authors: [Vladimir Bataev, Subhankar Ghosh, Vitaly Lavrukhin, Jason Li]
year: 2025
venue: "arXiv"
tags: [TTS, neural-transducer, RNNT, audio-codec, monotonic-alignment, end-to-end, zero-shot]
concepts: ["[[Residual Vector Quantization]]", "[[Duration Predictor]]", "[[Global Style Tokens]]", "[[Non-autoregressive TTS]]", "[[Codec Language Model]]", "[[LLM-based TTS]]", "[[Speaker Embedding]]"]
models: ["[[模型库/EnCodec|EnCodec]]", "[[模型库/SoundStream|SoundStream]]"]
tasks: ["[[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]"]
datasets: ["LibriTTS-R", "VCTK"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: TTS-Transducer 处于 LLM-based TTS 范式 (VALL-E 系列) 与传统 NAR TTS (FastSpeech 系列) 之间的独特位置。它借鉴了 VALL-E 的"预测 audio codec tokens"思路,但用 ASR 领域成熟的 neural transducer (RNNT) 替代自回归 decoder-only Transformer,以获得单调对齐约束。这与 VALL-E 系列的核心差异在于: VALL-E 隐式学习文本-语音对齐 (自回归 + cross-attention),TTS-Transducer 则通过 RNNT lattice 显式强制单调对齐,从而避免了 LLM-based TTS 常见的跳字/重复问题,也避免了 NAR TTS 对显式 [[Duration Predictor]] 的依赖。
>
> **已有认知**:
> - [[Residual Vector Quantization]] (confirmed): TTS-Transducer 直接操作 RVQ 产生的多层 codec tokens,第一层由 transducer 预测,剩余层由 NAR Transformer 迭代预测。这与 VALL-E 的 AR+NAR 两阶段结构类似,但第一阶段的建模方式从 decoder-only LM 变为 transducer。
> - [[LLM-based TTS]] (confirmed): VALL-E 系列是当前主流 zero-shot TTS 范式,核心是 codec LM + in-context learning。TTS-Transducer 提供了一种不依赖大规模预训练的替代路线,仅用 464h 数据即可达到可比性能。
> - [[Speaker Embedding]] (confirmed): TTS-Transducer 使用 [[Global Style Tokens]] [待确认] 从参考语音提取说话人 embedding,通过 conditional LayerNorm 注入 encoder 和 RCH,而非 VALL-E 的 in-context prompt 方式。
> - [[模型库/EnCodec|EnCodec]] (confirmed): 本文使用的三种 codec 之一,也是 VALL-E 的默认 codec。
> - [[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]] (confirmed): TTS-Transducer 在 unseen speakers 上展示了零样本能力,speaker similarity 0.868-0.881,与 VALL-E-X 等系统可比。
> - [[Codec Language Model]] [待确认]: TTS-Transducer 是 CodecLM 的一个替代方案——同样预测 codec tokens,但用 transducer 而非 language model 建模。
>
> **创新判断**: 将 RNNT 引入 TTS codec token 预测是新颖的组合。已有 Transduce-and-Speak (ASRU 2023) 和 VALL-T (arXiv 2024) 探索过 transducer+TTS,但前者需要两阶段分开训练且依赖 semantic tokens,后者内存开销极大;TTS-Transducer 通过 transducer (第一码本) + RCH (剩余码本) 的分离设计解决了多码本 RNNT 的内存问题,且支持端到端联合训练。
>
> 检索命中: [[Residual Vector Quantization]]✓, [[LLM-based TTS]]✓, [[Speaker Embedding]]✓, [[模型库/EnCodec|EnCodec]]✓, [[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]✓ | 过滤: [[Codec Language Model]](pending-review), [[Global Style Tokens]](pending-review), [[Duration Predictor]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 将 ASR 领域的 neural transducer (RNNT) 引入 TTS,端到端预测 audio codec tokens,利用单调对齐约束避免显式 duration predictor 和自回归 hallucination。
> - **路线**: Tokenized Text → Transformer Encoder (+ GST speaker cond.) → RNNT Joint Network → 第一码本 codes → Aligned Encoder Output + Predicted Codes → NAR Transformer (RCH) → 剩余码本 codes → Audio Codec Decoder → Waveform
> - **指标**: CER 3.94% / WER 13.83% (challenging texts, IPA+NeMo-Codec) vs Bark CER 11.67% / SpeechT5 CER 6.00%; MOS 3.82 (可比 SpeechT5 3.84); SSIM 0.868-0.908 (LibriTTS-R) [Table I, III]
> - **可借鉴**: (1) 用 RNNT loss + k2 shortest_path 同时获得对齐和预测,零额外计算开销;(2) 只用 transducer 预测第一码本、剩余码本用 NAR 头迭代预测,优雅解决多码本 RNNT 内存爆炸问题;(3) codec-agnostic 设计——同一架构适配 EnCodec/DAC/NeMo-Codec 无需修改
> - **局限**: 仅在 LibriTTS-R (464h) 上训练,未在大规模数据 (60K+ h) 上验证 scaling;MOS 仅 3.82 (低于 ground truth 通常 4.0+);未开源模型权重;未与同期强 baseline (如 VALL-E 2, CosyVoice) 对比;streaming 模式仅作为 future work 提及

## 核心问题

1. **如何在 TTS 中实现鲁棒的文本-语音单调对齐,同时避免显式 duration predictor 和 AR decoder 的 hallucination?** — 用 RNNT 的 blank 机制强制单调对齐 [§III]
2. **如何解决 RNNT 应用于 RVQ 多码本目标时的内存爆炸问题?** — 将预测拆分为 transducer (第一码本) + NAR head (剩余码本) [§III]
3. **Transducer-based TTS 能否达到与主流 LLM-based TTS 可比的质量和鲁棒性?** — 在 challenging texts 上 CER 优于 Bark 和 VALL-E-X,MOS 可比 [Table III]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TTS-Transducer 由两个端到端联合训练的组件构成 [§III, Fig 1]:

**组件 1: Neural Transducer (预测第一码本)**
- **Encoder**: 非自回归 Transformer (12 层),将 tokenized text (BPE 或 IPA) 转换为隐表示序列 e_i。通过 conditional LayerNorm 注入 GST speaker embedding [§III]。
- **Prediction Network (Predictor)**: 自回归 Transformer-Decoder (6 层),以 `<SOS>` 开头,将已预测的第一码本 tokens c_{0,j} 转换为预测向量 p_j [§III]。
- **Joint Network**: 对每对 (e_i, p_j) 计算 `j_{i,j} = Softmax(Linear(ReLU(e_i + p_j)))`,输出第一码本词表 + `<blank>` 的概率分布 [§III]。
- **Loss**: RNNT loss (基于 k2 框架的 WFST 实现),优化所有可能对齐路径的总概率 [§III, ref 33-34]。

**组件 2: Residual Codebook Head (RCH, 预测剩余码本)**
- NAR Transformer-Encoder (12 层) [§III]。
- 输入: 前 i-1 个码本的 embedding 之和,拼接对齐后的 encoder 输出 e_k [§III]。对齐信息从 RNNT lattice 中通过 k2.shortest_path 提取 [§III, footnote 2]。
- 训练时: 随机选择一个残差码本 i,预测第 i+1 个码本的 codes [§III]。
- Loss: cross-entropy loss (λ_CE) [§III]。
- 同样通过 conditional LayerNorm 注入 GST speaker embedding [§III]。

**总 loss**: λ_total = (1-α)·λ_RNNT + α·λ_CE, α=0.4 [§III]。

**解码阶段**: 先运行 RNNT 解码获得第一码本预测 + 对齐 (label-looping greedy decoding + nucleus sampling, p=0.95),再迭代运行 RCH 从第 1 到第 n 个残差码本逐层 greedy 预测 [§III]。最后用 audio codec decoder 合成波形。

### 关键设计选择

**为什么用 RNNT 而不是 AR decoder-only?**

[论文原文] Transducers 通过 blank 符号天然强制单调对齐,解决了 AR TTS 的 hallucination/跳字/重复问题 [§I]。而 AR encoder-decoder 虽然语音更自然,但存在这些稳定性缺陷 [§I, ref 5]。[agent 解读] 这是将 ASR 领域对 RNNT 鲁棒性的认知迁移到 TTS 的典型思路——RNNT 在 ASR 中已被验证在单调序列任务上优于 attention-based 模型。

**为什么不直接用 RNNT 预测所有码本?**

[论文原文] 直接用 RNNT 预测所有码本的全部 codes 序列化排列,会导致内存复杂度爆炸——RNNT loss 的内存复杂度取决于输入和目标序列长度的乘积;将 N 个码本展开后目标长度增大 N 倍 [§I]。

[agent 解读] 以 EnCodec 8 codebooks 为例,若每帧 8 个 token 全部序列化,目标序列长度增加 8 倍,RNNT lattice 内存占用约为 8 倍,在长句上会超出 GPU 内存。拆分为 transducer (1 codebook) + NAR head (7 codebooks) 完全避免了这个问题,代价是残差码本间失去了 RNNT 的对齐建模能力——但 [论文原文] 认为残差码本只需要在已知对齐下预测 fine details,不需要重新学习对齐 [§III]。

**为什么用 GST 而不是固定 speaker embedding?**

[论文原文] 固定 speaker verification embedding 在未见说话人上泛化能力差 [§III, ref 31];GST 通过 CNN-RNN encoder + multi-head attention over learnable style tokens 自适应提取说话人风格,泛化性更好 [§III, ref 14, 32]。

**为什么同时实验 BPE 和 IPA?**

[agent 解读] 论文虽未明确说明动机,但实验结果清晰表明: IPA 在所有场景下一致优于 BPE 的 CER/WER (如 seen speakers: BPE 最低 CER 2.18% vs IPA 最低 CER 1.31%),代价是 SSIM 略低 [Table I]。这可能因为 IPA phonemes 与声学特征有更直接的对应关系,减少了 transducer 的对齐学习难度。

### 训练策略

- **数据**: LibriTTS-R train-clean-100 + train-clean-360 + train-other-500,过滤 >15s,共 464 小时 [§IV-A]。
- **模型规模**: 199-200M 参数 (EnCodec/NeMo-Codec 199M, DAC 200M 因 9 codebooks 多一个 embedding table) [§IV-B]。
- **Codec**: EnCodec (8 codebooks, 6 kbps), NeMo-Codec RVQ (8 codebooks, 6.9 kbps), DAC (9 codebooks, 8 kbps) [§IV-B]。
- **Speaker embedding**: 1024 个 640 维 GST learnable embeddings [§IV-B]。
- **Optimizer**: AdamW, cosine annealing, 2000 warmup steps, max lr 1e-3 [§IV-B]。
- **Hardware**: 32 NVIDIA A100 GPUs, global batch 2048, 200 epochs [§IV-B]。
- **Decoding**: nucleus sampling (p=0.95) for 第一码本, greedy for 残余码本 [§IV-B]。

## 实验

| 指标 | 本文 (best) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER (seen) | 1.31% (IPA+NeMo) | 0.93% (GT) | LibriTTS-R train-clean-100 held-out | [Table I] |
| CER (unseen) | 2.11% (IPA+NeMo) | 0.87% (GT) | LibriTTS-R dev-clean | [Table I] |
| CER (OOD) | 1.48% (IPA+NeMo) | 0.44% (GT) | VCTK | [Table I] |
| SSIM (seen) | 0.908 (BPE+NeMo) | — | LibriTTS-R | [Table I] |
| SSIM (unseen) | 0.881 (BPE+NeMo) | — | LibriTTS-R | [Table I] |
| CER (challenging) | 3.94% (IPA+NeMo) | Bark 11.67% / SpeechT5 6.00% | 184 utterances | [Table III] |
| WER (challenging) | 13.83% (IPA+NeMo) | Bark 22.92% / SpeechT5 16.24% | 184 utterances | [Table III] |
| MOS (challenging) | 3.82±0.04 (IPA+NeMo) | Bark 3.82±0.04 / SpeechT5 3.84±0.04 | 184 utterances | [Table III] |

### 消融实验 [Table II]

模型大小对 WER 的影响 (EnCodec, BPE):
- 6+3+6 层 (最小): WER 6.66% (seen) / 6.93% (unseen)
- 12+6+12 层 (最大): WER 4.90% (seen) / 6.25% (unseen)
- 结论: encoder、predictor、RCH 三个组件的层数增加都对 intelligibility 有贡献 [§IV-C]

### BPE vs IPA

- IPA 在所有 codec 和所有 speaker 条件下一致优于 BPE 的 CER/WER [Table I]
- BPE 在 SSIM 上略优于 IPA (0.903 vs 0.900 for DAC seen) [Table I]
- [agent 解读] IPA 提供了更接近声学的输入表示,降低了对齐学习的负担;BPE 的 subword 粒度在捕获说话人风格方面可能略有优势

## 局限性

1. **数据规模**: 仅在 464h LibriTTS-R 上训练,未在大规模数据 (VALL-E 用 60K h, Bark 类似) 上验证 scaling 行为 [agent 解读]。论文声称无需大规模预训练 [§V],但未提供 scaling 实验佐证。
2. **Baseline 选择**: 仅与 Bark (开源), VALL-E-X (非官方实现), SpeechT5 对比;未与同期的 VALL-E 2, CosyVoice 等强 baseline 比较,难以判断真实竞争力 [agent 解读]。
3. **MOS 天花板**: MOS 3.82-3.84,与 baseline 可比但距 human 水平 (通常 >4.0) 仍有差距 [Table III]。
4. **未开源权重**: 论文称将在 NeMo toolkit 中开源代码 [§I],但截至论文发布仅有代码框架,模型权重未公开。
5. **Streaming 未实现**: 论文提到可将 RCH 替换为 AR Transformer 实现 streaming [§III],但仅作为 future work,未提供实验。
6. **Speaker conditioning**: 使用 GST 而非现代 prompt-based 方案 (如 in-context codec tokens),在极端 zero-shot 场景下的泛化能力有待验证 [agent 解读]。

## 点评

**优势**:
1. 动机清晰且合理: 将 RNNT 的单调对齐优势从 ASR 迁移到 TTS,在理论上优雅地避免了 AR hallucination 和显式 duration predictor 的两个痛点。
2. 工程设计精巧: transducer (第一码本) + NAR head (剩余码本) 的拆分既解决了内存问题,又保持了端到端联合训练,k2 shortest_path 提取对齐的零额外计算开销是实用的工程贡献。
3. Codec-agnostic: 在三种不同 codec 上均验证有效,证明了架构的通用性。

**不足**:
1. 实验对比不够全面: 仅与 2023 年的 Bark、VALL-E-X (非官方)、SpeechT5 对比,未涉及 VALL-E 2、RALL-E、CosyVoice 等同期或更强系统。
2. 评估指标依赖 ASR 转写的 CER/WER 和 WavLM-based SSIM,缺少 PESQ、UTMOS 等直接音质指标。
3. 论文未讨论 inference latency / RTF,无法判断实际可用性。

**定位**: TTS-Transducer 是一个有趣的"跨界"工作,将 ASR 领域的 transducer 技术引入 TTS,提供了一种不依赖大规模 LLM 预训练的鲁棒 TTS 方案。其核心价值在于鲁棒性 (CER 3.94% 在 challenging texts 上确实优异),但在自然度和 zero-shot 质量上尚未展现超越当前 SOTA 的潜力。

## 可复用的 idea

1. **RNNT + k2 shortest_path 提取对齐**: 在 transducer 解码过程中零开销获得文本-语音对齐,可用于任何需要单调对齐信息的下游任务 (如 forced alignment, duration extraction)。
2. **第一码本 + 剩余码本分离预测**: 将多码本 RVQ 的预测拆为"对齐敏感"和"对齐无关"两部分,是处理 RVQ 多码本建模的通用策略,可移植到其他架构。
3. **随机残差码本采样训练**: 训练时随机选择第 i 个码本预测第 i+1 个,增加数据效率和泛化——类似于 Quantizer Dropout 的思想,可用于任何迭代 RVQ 预测场景。
4. **Label-looping + nucleus sampling**: 将 RNNT 的高效 label-looping decoding 与 nucleus sampling 结合,是 transducer 用于生成任务时的实用解码策略。

---

> [!review] 审阅状态
> 待审阅 — 见 `_review/TTS-Transducer-review.yml`

检索命中: [[Residual Vector Quantization]], [[LLM-based TTS]], [[Speaker Embedding]], [[模型库/EnCodec|EnCodec]], [[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]] | 过滤: [[Codec Language Model]](pending-review), [[Global Style Tokens]](pending-review), [[Duration Predictor]](pending-review) | 未命中但可能相关: 无
