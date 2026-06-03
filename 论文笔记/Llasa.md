---
type: paper
tier: deep
title: "Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis"
arxiv_id: "2502.04128"
source: "https://arxiv.org/abs/2502.04128"
authors: [Zhen Ye, Xinfa Zhu, Chi-Min Chan, Xinsheng Wang, Xu Tan, Jiahe Lei, Yi Peng, Haohe Liu, Yizhu Jin, Zheqi Dai, Hongzhan Lin, Jianyi Chen, Xingjian Du, Liumeng Xue, Yunlin Chen, Zhifei Li, Lei Xie, Qiuqiang Kong, Yike Guo, Wei Xue]
year: 2025
venue: "ICML 2025"
tags: [TTS, LLM-based-TTS, scaling-law, inference-time-compute, speech-codec, single-codebook, autoregressive, zero-shot, FSQ, open-source]
concepts: ["[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Finite Scalar Quantization]]", "[[Residual Vector Quantization]]", "[[Semantic vs Acoustic Tokens]]", "[[Single-codebook vs Multi-codebook]]", "[[Token Rate and Bitrate Trade-offs]]"]
models: ["[[CosyVoice]]", "[[EnCodec]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[Semantic vs Acoustic Tokens]]✓, [[Residual Vector Quantization]]✓, [[CosyVoice]]✓, [[Zero-shot Speech Synthesis]]✓)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: LLM-based TTS 领域已形成两条主流路线: (1) AR+NAR 多阶段 (VALL-E 系列) 或 AR+Diffusion 混合 (CosyVoice, Seed-TTS, FireRedTTS),使用 RVQ 多层 codec 或 semantic+acoustic 两阶段 token; (2) 连续值 AR (LatentLM, CLEAR),绕过离散量化。Llasa 提出第三条路线: **单层 VQ codec + 单 Transformer**,完全对齐文本 LLM 范式,是目前该方向最系统的尝试。
>
> **已有认知**: 概念库确认 semantic vs acoustic tokens 的 trade-off 核心是语义连贯性 vs 声学保真度; 单层 codec 在声学重建上受限 (SPK-SIM < 0.85, PESQ ~3); RVQ 多层方案声学质量高但序列长度大,不利于 LLM 自回归建模。FSQ 已被 CosyVoice 系列验证为比 VQ 更稳定的量化方案。
>
> **创新判断**: Llasa 的核心创新不在架构本身 (单 Transformer + codec LM 并不新),而在两个实验性贡献: (a) 系统验证了 TTS 领域的 train-time scaling law (1B→3B→8B, 80k→160k→250k hours); (b) 首次将 inference-time compute scaling (best-of-N, beam search with verifiers) 引入 TTS,提出 partial PRM 策略平衡 speaker similarity 和内容准确性。
>
> 检索命中: [[LLM-based TTS]], [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Residual Vector Quantization]], [[CosyVoice]], [[Zero-shot Speech Synthesis]] | 过滤: [[Finite Scalar Quantization]](pending-review), [[Codec Language Model]](pending-review), [[Single-codebook vs Multi-codebook]](pending-review), [[Token Rate and Bitrate Trade-offs]](pending-review), [[MELLE]](pending-review), [[Emilia]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 将 TTS 完全对齐标准 LLM 范式 (单层 VQ codec X-codec2 + LLaMA 初始化的单 Transformer),系统验证 train-time 和 inference-time scaling 效果
> - **路线**: Text tokens → [LLaMA Transformer] → Speech tokens (单层 FSQ, 50Hz) → [X-codec2 Decoder] → Waveform
> - **指标**: Seed-TTS-eval test-hard WER 4.38% / SIM-O 0.767 (8B+partial PRM+ORM) [Table 2]; LibriSpeech test-clean WER-H 1.49% / SIM-R 0.740 (8B+PRM+ORM) [Table 5]; X-codec2 UTMOS 4.13 / WER 2.47 / SPK-SIM 0.82 at 50 token/s [Table 1]
> - **可借鉴**: Partial PRM 策略 (前 n 秒用 beam search + speaker verifier,后段切换到 best-of-N + WER verifier) 平衡了 SIM 和 WER;X-codec2 用 semantic encoder (Wav2Vec2-BERT) + acoustic encoder concat → 单层 FSQ 融合语义声学信息
> - **局限**: 单层 codec 的 SPK-SIM 上限受限 (resynthesis SIM-O 仅 0.677 on Seed-TTS-eval);test-time scaling 的计算量极大 (beam width 16, N=16, 即 256x 候选);未开源训练数据;中文 test-hard WER 仍达 4.38%

## 核心问题

当前 LLM-based TTS 系统大多采用多阶段架构 (如 AR+Diffusion, AR+NAR),每个阶段有独立的模型和设计选择,这使得研究者难以像文本 LLM 社区那样系统探索 scaling law。Llasa 试图回答: **如果把 TTS 彻底简化为与文本 LLM 完全对齐的范式 (单 tokenizer + 单 Transformer),训练规模和推理计算量的增加是否也能带来一致的质量提升?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Llasa 的架构极其简洁: **一个 speech tokenizer (X-codec2) + 一个 LLaMA Transformer**。

训练时,文本 token 序列 {x_1, ..., x_T} 和语音 token 序列 {y_1, ..., y_S} 拼接为一个序列,模型学习条件概率 P(y_1, ..., y_S | x_1, ..., x_T),loss 仅计算在语音 token 上 [§3.1]。推理时,给定文本 token,自回归生成语音 token,再由 X-codec2 decoder 解码为波形。

**与标准文本 LLM 的完全对齐**: Transformer 参数直接从 LLaMA (3.2-1B/3B, 3.1-8B) 初始化,只扩展 vocabulary 加入 speech tokens。训练用标准 next-token prediction,无任何 TTS 特化模块 (无 duration predictor, 无 pitch predictor, 无 diffusion decoder) [§3.1]。[论文原文]

**为什么这样设计**: 作者认为多阶段 TTS 系统 (如 AR LLM + Diffusion) 阻碍了 scaling 研究,因为无法确定应该 scale 哪个模块。单 Transformer 架构使 scaling 的效果可以被干净地度量 [§1]。[论文原文]

### 关键设计选择

#### X-codec2: 单层 VQ Speech Tokenizer

X-codec2 是 Llasa 的核心组件,基于 X-codec 改进,目标是用单层 codebook 同时编码 semantic 和 acoustic 信息:

**Encoder (双路融合)**:
- Semantic Encoder (Enc_s): 预训练 Wav2Vec2-BERT,提取多语言语义特征 (内容+情感) [§3.2]
- Acoustic Encoder (Enc_a): 多层 residual convolutional blocks + Snake activation,编码低级声学细节 [§3.2]
- 两路 concat: H = [Enc_s(Y), Enc_a(Y)]

**Vector Quantization**: 使用 FSQ (Finite Scalar Quantization) 而非传统 VQ 或 RVQ [§3.2]。

**为什么用 FSQ 而非 RVQ**: RVQ 产生多层 codebook index,需要多阶段建模 (如 VALL-E 的 AR+NAR);FSQ 用单层量化确保 **1D 因果依赖**,与 LLM 的从左到右自回归机制天然对齐。此外 FSQ 不需要 codebook commitment loss,训练更简单 [§3.2]。[论文原文] [agent 解读: 这是 Llasa "完全对齐 LLM" 理念的关键技术选择 -- 牺牲多层 RVQ 的重建质量换取架构简洁性]

**Decoder**: Transformer-based decoder (替代 ConvNeXt) 预测 STFT magnitude + phase,通过 iSTFT 重建波形。语义重建分支 (L2 loss) 仅在训练时使用,推理时不需要 [§3.2]。[论文原文]

**Codec 训练**: 使用 multi-period discriminator (MPD) + multi-scale STFT discriminator + spectral discriminator 对抗训练。最后阶段加入 perceptual loss 增强可懂度 [§3.2]。训练数据约 150k hours 多语言语音 (Emilia + MLS),16kHz,codebook size 65536,下采样率 320 (即 50Hz token rate) [§4.1.1]。

**关键参数**: codebook size = 65536, projection dim = 8, frame rate = 50Hz, token rate = 50 tokens/s [Table 1]

#### 为什么不用 prompt 机制做 zero-shot?

[agent 解读] 论文未详细讨论 prompt 机制,但从实验设置 (Seed-TTS-eval, LibriSpeech continuation) 看,zero-shot TTS 应通过 in-context learning 实现: 将参考语音的 codec tokens 作为序列前缀,模型从中隐式学习说话人特征。这与 VALL-E 系列的做法一致。

### 训练策略

**Train-time Scaling 实验设计** [§3.3]:
- 固定数据 (250k hours), 变模型规模: LLaMA 3.2 1B / 3B, LLaMA 3.1 8B
- 固定模型 (1B), 变数据量: 80k / 160k / 250k hours (随机采样 1/3, 2/3, 全量)

**训练超参**: 3 epochs, batch size 2M tokens, max lr 5e-5, cosine schedule (3% warmup), 最终 lr = 10% peak, 最大序列长度 2048 tokens [§4.2.1]

**Inference-time Scaling 策略** [§3.4]:

核心思路: 生成多个候选,用 speech understanding models (verifiers) 评分选优。

1. **ORM (Output Reward Model) + Best-of-N**: 完整生成 N 个候选,用 verifier 选最优。简单但无中间引导 [§3.4]。
2. **PRM (Process Reward Model) + Beam Search**: 每 M=25 tokens (0.5秒) 扩展 B 个 beam,每个 beam 生成 N=16 个候选,按 verifier 分数选 top-B。逐步引导生成方向,但可能陷入局部最优 [§3.4]。
3. **Partial PRM (创新策略)**: 前 n=2 秒用 PRM (speaker similarity verifier),之后切换到 ORM (WER verifier)。平衡 speaker similarity 和内容准确性 [§4.2.3]。[论文原文]

**为什么 PRM 单独用效果不好**: 实验发现纯 PRM beam search 虽提升 SIM,但 WER 反而变差 -- 因为逐步最大化 SIM 导致生成陷入局部最优,多样性不足 [§4.2.3, Fig 2 红线]。[论文原文]

## 实验

### Codec 评估 (X-codec2 vs baselines, LibriSpeech test-clean) [Table 1]

| 指标 | X-codec2 (ours) | BigCodec | X-codec | DAC (600) | EnCodec (600) | Ground Truth | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Token Rate | 50 | 80 | 50 | 600 | 600 | - | [Table 1] |
| Codebook Layer | 1 | 1 | 1 | 12 | 8 | - | [Table 1] |
| WER ↓ | 2.47 | 2.76 | 3.42 | 2.00 | 2.15 | 1.96 | [Table 1] |
| SPK-SIM ↑ | 0.82 | 0.84 | 0.52 | 0.95 | 0.89 | 1.00 | [Table 1] |
| UTMOS ↑ | 4.13 | 4.11 | 4.05 | 4.00 | 3.09 | 4.09 | [Table 1] |
| PESQ-WB ↑ | 2.43 | 2.68 | 1.84 | 4.01 | 2.77 | 4.64 | [Table 1] |

**发现**: X-codec2 在 50 token/s 单层 codebook 下取得最优综合表现 (WER 最低, UTMOS 接近 GT)。但 SPK-SIM (0.82) 和 PESQ (2.43) 仍明显低于高 token rate 的多层 RVQ 方案 (DAC 600: 0.95/4.01),说明单层 codec 的声学重建上限确实受限 [§4.1.3]。

### TTS Train-time Scaling (Seed-TTS-eval) [Table 2]

| 模型 | test-zh CER↓ | test-zh SIM-O↑ | test-en WER↓ | test-en SIM-O↑ | test-hard WER↓ | test-hard SIM-O↑ | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Seed-TTS (closed) | 1.12 | 0.796 | 2.25 | 0.762 | 7.59 | 0.776 | [Table 2] |
| CosyVoice 2 | 1.45 | 0.748 | 2.57 | 0.652 | 6.83 | 0.724 | [Table 2] |
| MaskGCT | 2.27 | 0.774 | 2.62 | 0.714 | 10.27 | 0.748 | [Table 2] |
| llasa 1B 80k | 2.69 | 0.648 | 3.71 | 0.541 | 17.11 | 0.618 | [Table 2] |
| llasa 1B 250k | 1.89 | 0.669 | 3.22 | 0.572 | 12.13 | 0.638 | [Table 2] |
| llasa 8B 250k | 1.59 | 0.684 | 2.97 | 0.574 | 11.09 | 0.660 | [Table 2] |

**发现**: 增大模型和数据一致带来改善。但直接推理时 SIM-O 较低 (8B: 0.684),这主要受限于单层 codec 的重建上限 (resynthesis SIM-O 仅 0.677) [§4.2.4]。

### TTS Test-time Scaling (Seed-TTS-eval) [Table 2]

| 模型 | test-zh CER↓ | test-zh SIM-O↑ | test-hard WER↓ | test-hard SIM-O↑ | 出处 |
| --- | --- | --- | --- | --- | --- |
| llasa 8B (direct) | 1.59 | 0.684 | 11.09 | 0.660 | [Table 2] |
| llasa 8B + Partial PRM (SIM) | 1.04 | 0.827 | 10.59 | 0.785 | [Table 2] |
| llasa 8B + Partial PRM + ORM (WER) | 0.47 | 0.825 | 4.38 | 0.767 | [Table 2] |
| Seed-TTS (closed) | 1.12 | 0.796 | 7.59 | 0.776 | [Table 2] |

**关键发现**: Test-time scaling 效果显著 -- 8B 模型通过 partial PRM + ORM,SIM-O 从 0.684→0.825 (超越 Seed-TTS 的 0.796),test-hard WER 从 11.09→4.38 (远超 Seed-TTS 的 7.59) [Table 2]。但代价是 256x 计算量。

### Emotion Similarity (ESD dataset) [Table 4]

| 模型 | en EMO-SIM | zh EMO-SIM | 出处 |
| --- | --- | --- | --- |
| GT | 0.94 | 0.94 | [Table 4] |
| llasa 8B (direct) | 0.778 | 0.861 | [Table 4] |
| llasa 8B + PRM (emotion) | 0.951 | 0.974 | [Table 4] |

**发现**: 用 Emotion2Vec 作为 PRM verifier 时,emotion similarity 接近甚至超过 GT,说明 test-time scaling 可以将生成分布偏向特定 verifier 的偏好 [§4.2.3, Table 4]。

### ASR 扩展 (LibriSpeech) [Table 3]

| 模型 | Test Clean WER | Test Other WER | 出处 |
| --- | --- | --- | --- |
| Whisper large v3 | 1.8 | 3.6 | [Table 3] |
| llasa ASR 3B | 1.9 | 5.9 | [Table 3] |

**发现**: 反转 speech/text token 顺序即可做 ASR,证明 "单 Transformer + tokenizer" 范式的灵活性 [§4.3]。

### Text Understanding Ability [Fig 1]

增大模型 (1B→8B) 和数据 (80k→250k) 显著提升中英文文本理解能力。情感表达、中文古诗、绕口令等需要深层语义理解的任务获益最大;简单任务 (如疑问句) 提升有限;生僻字主要受益于数据量增加而非模型增大 [§4.2.2, Fig 1]。

## 局限性

1. **单层 codec 的声学重建上限**: X-codec2 的 SPK-SIM (0.82) 和 PESQ-WB (2.43) 在 50 token/s 下虽然是单层最优,但与多层 RVQ (DAC 600: SPK-SIM 0.95, PESQ 4.01) 差距明显。这直接限制了 TTS 的 SIM-O 天花板 [Table 1, §4.2.4]
2. **Test-time scaling 计算代价极高**: beam width 16 + N=16 意味着每步生成 256 个候选,加上 verifier 推理,推理成本是 baseline 的数百倍。实用性存疑
3. **Verifier 偏好的风险**: 用 PRM 最大化 SIM 会导致 WER 恶化 [Fig 2 红线],说明 test-time scaling 本质上是在 verifier 的偏好方向上偏移分布,而非真正提升质量。不同 verifier 之间可能冲突
4. **训练数据未开源**: 250k hours 的数据规模难以复现
5. **16kHz 采样率**: 限制了合成语音的频谱上限
6. **SIM-O vs SIM-R 差异**: codec resynthesis 本身就丢失了 speaker similarity (SIM-O 0.677 vs GT 0.755 on Seed-TTS-eval zh),SIM-R (resynthesis-reference) 高但 SIM-O (original-reference) 低,说明 codec bottleneck 而非 LM 能力是 SIM 的主要瓶颈

## 点评

**优势**:
1. **范式简洁性的极致追求**: 单 tokenizer + 单 Transformer 彻底消除了 TTS 的模块化复杂度,使 scaling 实验的结论干净可解释。这种 "不做加法" 的设计哲学值得尊重
2. **首个系统的 TTS scaling 研究**: 从 1B 到 8B、从 80k 到 250k hours 的系统实验,清晰展示了模型规模和数据规模的分别贡献,为 TTS 领域的 scaling 研究奠定了 baseline
3. **Partial PRM 的实用创新**: 前段 PRM + 后段 ORM 的混合策略优雅地解决了 SIM vs WER 的矛盾,是 test-time compute 在 TTS 中的有价值探索
4. **开源**: 1B/3B/8B 模型 + codec + 训练代码全部开源

**不足**:
1. **Codec 是瓶颈但投入不够**: 论文的核心创新声称是 scaling,但实际上直接推理时性能受限于 codec 重建质量。X-codec2 相比 X-codec 的改进 (SPK-SIM 0.52→0.82) 幅度大于任何 train-time scaling 的改进,暗示 **tokenizer 质量才是此类系统的第一性约束**
2. **Test-time scaling 的公平性**: 与 Seed-TTS 等基线的对比,Llasa 使用了 256x 候选 + 外部 verifier,而 baseline 都是单次推理。论文承认 "this comparison might not be entirely fair" [§4.2.4],但结论标题仍用 "state-of-the-art"
3. **缺乏 perception-based 评估**: 无 MOS 评测,仅靠 text understanding 的 expert score (3-point scale) 和客观指标,对合成质量的判断不完整
4. **Zero-shot prompt 机制细节缺失**: 论文未详细说明 in-context learning 的实现方式 (prompt length, prompt token 的处理)

## 可复用的 idea

1. **Partial PRM 策略**: 将 test-time search 分为两阶段 -- 前段用 process-level verifier 确定方向 (如 speaker identity),后段用 output-level verifier 保证内容质量 (如 WER)。适用于任何需要平衡多个目标的生成任务
2. **Semantic + Acoustic encoder concat → 单层 VQ**: X-codec2 的双路编码器 concat 后做单层 FSQ 量化,是在单 codebook 约束下融合语义和声学信息的实用方案。可用于需要 1D 因果序列的场景
3. **LLM 初始化 + vocabulary 扩展**: 从预训练文本 LLM (LLaMA) 直接初始化,仅扩展 vocabulary 加入 speech tokens,比从头训练更高效
4. **Text understanding evaluation protocol**: 用 7 类复杂文本 (情感/多音字/古诗/绕口令等) 评估 TTS 系统的文本理解能力,是一个有价值的评估框架

> [!review] 审阅 (2026-06-03, agent)
> **结论**: pass-with-fixes (3 issues: 0 high, 1 medium, 2 low)
> - [medium] frontmatter models 删除 MELLE (关联弱),保留 CosyVoice + EnCodec (已修正)
> - [low] zero-shot prompt 实现的推测可补充论文中的 prompt 设置描述
> - [low] 可复用 idea #3 (LLM 初始化) 稍显泛化
> 详见 `_review/Llasa-review.yml`
