---
type: paper
tier: deep
title: "MELA-TTS: Joint transformer-diffusion model with representation alignment for speech synthesis"
arxiv_id: "2509.14784"
source: "Sources/MELA-TTS.pdf"
authors: [Keyu An, Zhiyu Zhang, Changfeng Gao, Yabin Li, Zhendong Peng, Haoxu Wang, Zhihao Du, Han Zhao, Zhifu Gao, Xiangang Li]
year: 2025
venue: "arXiv"
tags: [TTS, diffusion, autoregressive, mel-generation, continuous-features, representation-alignment, zero-shot, streaming]
concepts: ["[[Diffusion-basedTTS]]", "[[MelSpectrogram]]", "[[Classifier-FreeGuidance]]", "[[SpeakerEmbedding]]", "[[SemanticvsAcousticTokens]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[NeuralVocoder]]"]
models: ["[[CosyVoice]]", "[[CosyVoice2]]", "[[CosyVoice3]]", "[[SenseVoice]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[SemanticvsAcousticTokens]]✓, [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓, [[Diffusion-basedTTS]][待确认], [[CosyVoice]]✓, [[SpeechTokenizer]]✓)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SemanticvsAcousticTokens]], [[LLM-basedTTS]], [[SpeakerEmbedding]], [[Diffusion-basedTTS]], [[CosyVoice]], [[SpeechTokenizer]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: MELA-TTS 属于 **continuous-valued AR** 路线,与 LatentLM、CLEAR、DiTAR 同属一个新兴分支 — 直接自回归生成连续表征,绕过离散 speech tokenizer。这条路线在 KB 中被 [[SpeechTokenizer]] 页面记录为 "Continuous VAE Tokenizer" 演进线,在 [[LLM-basedTTS]] 中被归为 "Continuous-valued AR (Next-Token Diffusion) 路线"。

**已有认知**:
- [[SemanticvsAcousticTokens]]: 传统 discrete-token TTS 面临 semantic-acoustic 两阶段的信息损失和级联误差。MELA-TTS 的核心立场是完全绕过离散化,直接从 mel-spectrogram 端到端生成。
- [[CosyVoice]]: 同一团队(阿里巴巴)的前作,采用监督式 S3 semantic tokens + LLM + OT-CFM 的 coarse-to-fine 两阶段架构。MELA-TTS 是该团队对 tokenizer-free 路线的探索,与 CosyVoice 系列形成对照实验。
- [[Diffusion-basedTTS]]: MELA-TTS 的 diffusion module 用 DiT 架构在 mel chunk 上做去噪,属于 "两阶段中的声学生成器" 角色,但与 Diff-TTS/Grad-TTS 不同的是它接收的条件来自自回归 transformer 而非 text encoder。
- [[SpeakerEmbedding]]: MELA-TTS 使用双 embedding (speaker embedding via 3D-Speaker + utterance embedding via transformer encoder),与 CosyVoice 的 x-vector 方案类似但增加了 utterance-level 建模。

**创新判断**: MELA-TTS 的核心创新 — representation alignment module (用 ASR encoder 输出对齐 AR decoder 隐表征) — 是对 "连续表征 AR 建模困难" 这一公认挑战的直接回应。相比 DiTAR 用 flow matching head、CLEAR/LatentLM 用 VAE tokenizer,MELA-TTS 选择保留 raw mel + 用 ASR 语义监督中间表征,思路更接近 CosyVoice 的监督式 token 理念,但在连续空间而非离散空间实现。

> [!summary] 速查
> - **一句话**: 用自回归 transformer 生成连续向量 + diffusion 生成 mel chunk 的 tokenizer-free TTS,通过 ASR encoder 对齐中间表征显著提升内容一致性和训练收敛速度
> - **路线**: Text (BPE) + Speaker/Utterance Embedding → Qwen2-0.5B AR Transformer Decoder (6.25Hz) → 连续向量 h → DiT Diffusion (10-step DDIM) → Mel Chunk (50Hz, 80-dim) → HiFTNet Vocoder → Waveform
> - **指标**: CER 0.9% (test-zh), WER 2.4% (test-en, 170K h), SS1 0.59/SS2 0.68 (test-en); 主观偏好 66.7% vs CosyVoice 2, 57.3% vs CosyVoice 3 [Table 2, Fig 5]
> - **可借鉴**: (1) 用预训练 ASR encoder 的表征对齐 AR decoder 中间状态,可加速训练 3.3x 且提升内容一致性; (2) 自回归帧率 6.25Hz 远低于 discrete-token 系统的 25-75Hz,大幅降低序列长度
> - **局限**: Speaker similarity 在 test-en 上不如 CosyVoice 系列 (SS2 0.68 vs 0.78-0.79),作者归因于 diffusion 模块只能利用 local context; 未开源

## 核心问题

这篇论文试图解决 **end-to-end 连续表征 TTS 的两个瓶颈**:

1. **内容一致性差**: 直接自回归生成连续 mel 特征时,模型缺乏显式语义引导,导致内容错误率 (WER/CER) 明显高于 discrete-token 方案 [§1]
2. **训练收敛慢**: 连续高维特征的自回归建模比离散 token 需要更多训练迭代才能收敛 [§1]

论文的核心假设是: 如果让 AR decoder 的中间表征 h 对齐预训练 ASR encoder 的语义表征,就能同时解决这两个问题 — h 被迫编码更丰富的语义信息,从而提升下游 diffusion 模块的内容一致性,并加速语义-声学的解耦学习。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MELA-TTS 由三个核心模块组成 [§2, Fig 1]:

1. **Autoregressive Transformer Decoder**: 以 Qwen2-0.5B 初始化,自回归生成连续向量 h,工作在 6.25 Hz (每步对应 160ms mel chunk) [§3.1.2]
2. **Diffusion Module**: 22 层 DiT (1024-dim, 16 heads),以 h + speaker/utterance embedding 为条件,去噪生成 mel-spectrogram chunk [§2.2]
3. **Representation Alignment Module**: 将 h 对齐到 SenseVoice-Large ASR encoder 的输出表征 [§2.3]

工作流程: Text (BPE from Qwen2) + Mel History → Transformer Decoder → h → DiT Denoising → Mel Chunk → HiFTNet Vocoder → Waveform

### 关键设计选择

**为什么选择 mel-spectrogram 而非 latent/token?**
[论文原文] 离散化 speech signals 天然带来信息损失,且两阶段 pipeline (tokenizer + decoder) 引入级联误差 [§1]。直接生成 mel 消除了这两层损失。
[agent 解读] 这一选择让 MELA-TTS 的位置介于 LatentLM/CLEAR (VAE latent) 和 DiTAR (也是连续,但更隐式) 之间 — MELA-TTS 选择了最 "原始" 的连续表征 mel-spectrogram 作为生成目标。

**为什么用 ASR encoder 表征而非 mel 本身作为 alignment target?**
[论文原文] 直接用 mel-spectrogram 作为对齐目标不仅无正面增益,反而显著恶化 WER 和 speaker similarity (Exp 2 vs Exp 0: WER 6.7 vs 6.3, SS2 0.48 vs 0.55) [§3.2]。作者解释这是因为 ASR encoder 表征鼓励了解耦的 semantic-acoustic 建模: AR transformer 生产语义信息丰富的 h,声学细节交给 diffusion 重建 [§3.2]。
[论文原文] 这一发现与 discrete-token TTS 中的认知一致: CosyVoice 等系统中 semantic-acoustic 解耦被证明同时有利于内容一致性和声音克隆 [§3.2, ref 9, 13]。

**Mel chunk 设计 (chunk size = 8)**:
[论文原文] 每个 mel chunk 包含 8 帧 (160ms),AR transformer 以 6.25 Hz 生成 h,远低于 discrete-token 系统的 25-75 Hz [§3.1.2]。
[agent 解读] 这意味着相同时长的语音,MELA-TTS 的 AR 序列长度仅为 CosyVoice 3 的 1/4 或 VALL-E 的 1/12,显著降低了自回归推理的计算量。

**Utterance Embedding 的作用**:
[论文原文] Utterance embedding 从语音中随机裁剪的片段经 transformer encoder 提取,与 decoder 联合训练。它增强了 speaker information 的建模能力 (SS1 0.46→0.47, SS2 0.55→0.57) [§3.2, Table 1]。
[论文原文] 当与 representation alignment 联合使用时,两者互补: alignment 负责语义一致性,utterance embedding 专注细粒度声学建模 (如 speaker identity),取得最优综合性能 [§3.2]。

**Stop Prediction Module**:
[论文原文] 不同于 discrete-token TTS 使用 EOS token,MELA-TTS 用二分类器 (BCE loss) 在每步判断是否终止合成 [§2.1]。
[agent 解读] 这是连续表征 AR 系统的必然选择 — 没有离散 codebook 就没有 EOS token 可预测。

### 训练策略

- **Loss**: $L = L_{diff} + L_{stop} + L_{align}$,其中 $L_{diff}$ 为 L2 mel reconstruction loss,$L_{stop}$ 为 BCE stop loss,$L_{align}$ 为 cosine similarity loss [§2.3]
- **ASR Encoder**: SenseVoice-Large encoder,输入 16kHz/128-dim mel (与 TTS 的 24kHz/80-dim mel 不同),输出 25Hz 语义表征 [§3.1.2]
- **Time Alignment Module (TAM)**: 线性层 + reshape,将 h 上采样 4x 匹配 ASR encoder 25Hz 输出的时间分辨率 [§2.3]
- **Diffusion 训练**: VP formulation ($\alpha_t = \cos(\pi t/2)$, $\sigma_t = \sin(\pi t/2)$),条件和无条件联合训练以支持 CFG [§3.1.2]
- **CFG**: $\alpha = 0.7$,DDIM 采样,默认 NFE=10 [§3.1.2]
- **Diffusion 上下文**: 前一个 mel chunk $X_0^{(i-1)}$ 和前一个隐向量 $h_{i-1}$ 作为 prefix context [§2.2]
- **Streaming**: text tokens 和 mel chunks 以 n:m = 4:3 交错排列,同一模型同时训练 interleaved 和 non-interleaved 序列 [§2.4]
- **数据**: LibriTTS 585h (消融) + 170K h in-house (130K 中文 + 30K 英文 + 10K 其他语言) [§3.1.1]

## 实验

| 指标 | MELA-TTS (w/ rep align) | CosyVoice (同数据) | CosyVoice 2 (同数据) | CosyVoice 3-0.5B (同数据) | DiTAR | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CER ↓ | 0.9 | 3.6 | 1.5 | 1.3 | 1.0 | seed-tts-eval test-zh | [Table 2] |
| WER ↓ | 2.4 | 4.3 | 2.6 | 2.5 | 2.3 | seed-tts-eval test-en | [Table 2] |
| CER ↓ | 4.0 | 7.6 | 7.0 | 7.7 | 6.8 | seed-tts-eval test-hard | [Table 2] |
| SS1 ↑ | 0.59 | 0.61 | 0.65 | 0.65 | 0.60 | seed-tts-eval test-en | [Table 2] |
| SS2 ↑ | 0.68 | 0.68 | 0.75 | 0.75 | 0.59 | seed-tts-eval test-en | [Table 2] |
| SS1 ↑ | 0.72 | 0.72 | 0.72 | 0.72 | 0.71 | seed-tts-eval test-en (streaming) | [Table 2] |

**消融实验 (LibriTTS, seed-tts-eval test-en)** [Table 1]:

| 配置 | WER ↓ | SS1 ↑ | SS2 ↑ |
| --- | --- | --- | --- |
| Baseline (无 rep align, 无 utt emb, offline) | 6.3 | 0.46 | 0.55 |
| + Rep Align | 5.3 | 0.46 | 0.54 |
| + Mel Align (替代 ASR) | 6.7 | 0.41 | 0.48 |
| + Utt Emb | 6.0 | 0.47 | 0.57 |
| + Rep Align + Utt Emb | **5.2** | **0.48** | **0.58** |
| Streaming baseline | 6.6 | 0.46 | 0.55 |
| Streaming + Rep Align + Utt Emb | **5.0** | **0.48** | **0.58** |

**训练收敛加速**: Representation alignment 使模型在 30 epochs 内达到无 alignment 训练 100+ epochs 的 WER 水平,加速 >3.3x [§3.2, Fig 4]

**主观评估**: A/B 偏好测试 (15 人, 各 15 对), MELA-TTS 对 CosyVoice 2 偏好率 66.7%, 对 CosyVoice 3 偏好率 57.3% [Fig 5]

**Scaling**: LibriTTS 585h → 170K h 带来显著提升: WER 5.2→2.4, SS1 0.48→0.59, SS2 0.58→0.68 (test-en) [§3.3]

## 局限性

1. **Speaker similarity 不足**: 在 test-en 上 SS2 0.68,明显低于 CosyVoice 系列的 0.75-0.79。[论文原文] 作者认为这是因为 diffusion 模块只能利用 local context (当前和前一个 chunk 的 h 和 mel),而 CosyVoice 的 flow-matching 模块可以 attend to 所有历史 tokens + prompt speech [§3.3]。类似的 speaker similarity gap 在其他 continuous-representation 系统 (如 MELLE, CLEAR) 中也被观察到 [§3.3, ref 3, 25]。

2. **Hard cases 鲁棒性缺乏充分对照**: MELA-TTS 在 test-hard 上 CER 4.0%,优于同数据 CosyVoice 系列 (7.0-7.7%) [Table 2]。但最直接的竞争对手 DiTAR 未在 test-hard 上评估 [§3.3],因此无法判断 continuous-AR 路线在 hard cases (重复/绕口令) 上的相对优劣。

3. **Diffusion 推理成本**: 每个 mel chunk 需 10 步 DDIM 采样,总推理步数 = AR 步数 x 10。论文未报告实际 RTF。[agent 解读] 对比 CLEAR 报告的 RTF 0.18,MELA-TTS 的推理效率可能不占优势。

4. **未开源**: 代码和模型未公开。

## 点评

**优势**:
- **Representation alignment 是一个简洁有效的创新**: 用已有的 ASR encoder 作为"语义锚点"对齐 AR decoder 中间表征,不增加推理成本 (只在训练时用),但同时解决了内容一致性和训练效率两个问题。这个 idea 可迁移到其他 continuous AR 系统。
- **同数据对比有说服力**: 与 CosyVoice 系列使用完全相同的训练数据 (170K h),证明 tokenizer-free 路线在内容一致性 (CER/WER) 上确实可以超越 discrete-token 路线 — CER 0.9 vs CosyVoice 3 的 1.3 (test-zh)。
- **极低的 AR 帧率**: 6.25 Hz 是目前 AR-based TTS 中最低之一,意味着 AR 序列极短,长文本合成的序列长度优势明显。

**不足**:
- **Speaker similarity 的 gap 仍然显著**: SS2 0.68 vs 0.75-0.79 (CosyVoice 系列),这在实际应用中可能是致命的。作者归因于 diffusion 的 local context 限制,但没有提出解决方案。
- **缺乏 RTF/latency 报告**: 作为一个 joint AR+diffusion 系统,推理效率是关键问题,但论文完全没有报告。
- **与 DiTAR 的对比不够充分**: DiTAR 是最直接的竞争对手 (同为 continuous AR + diffusion),但论文仅引用其数字,未深入分析两者架构差异的影响。

**定位**: MELA-TTS 为 CosyVoice 团队从 "monitored discrete token" 路线向 "continuous mel + semantic supervision" 路线的探索。它证明了连续表征 + 语义对齐可以在内容一致性上超越离散 token 方案,但在 speaker similarity 上仍有差距。这为未来可能的 CosyVoice 架构演进 (如将 representation alignment 与 flow-matching 全局条件结合) 提供了实验基础。

## 可复用的 idea

1. **Representation Alignment via ASR Encoder**: 在任何连续表征的 AR 系统中,用预训练 ASR encoder 输出作为中间表征的对齐目标。核心是 cosine similarity loss,不需要帧级对齐 (TAM 处理时间分辨率差异)。可直接应用于 LatentLM/CLEAR 等系统。
2. **Mel Chunk 作为 AR 单位**: 将 mel spectrogram 按固定帧数 (如 8 帧) 分 chunk,每个 chunk 经卷积降采样后作为 AR 的一个"token"。这将 AR 帧率降到 6.25 Hz,大幅缩短序列。
3. **Utterance Embedding 补充 Speaker Embedding**: 从参考语音的随机裁剪片段经独立 encoder 提取 utterance-level embedding,与 speaker embedding 互补 — 前者捕捉局部声学特征,后者捕捉全局身份信息。
4. **Diffusion Prefix Context**: 将前一个 chunk 的 clean mel + hidden vector 作为 diffusion 的 prefix,提供时序连续性。

---

> [!review] 审阅结论: pass-with-fixes (2026-06-04)
> 3 个 low 问题,不阻塞反向更新。方法的 WHY 解释充分,数据标注覆盖率高,KB 背景有具体谱系定位。
> 详见 `_review/MELA-TTS-review.yml`

检索命中: [[SemanticvsAcousticTokens]], [[LLM-basedTTS]], [[SpeakerEmbedding]], [[Diffusion-basedTTS]], [[CosyVoice]], [[SpeechTokenizer]] | 过滤: 无 | 未命中但可能相关: 无
