---
type: paper
tier: deep
title: "Continuous-Token Diffusion for Speaker-Referenced TTS in Multimodal LLMs"
arxiv_id: "2510.12995"
source: "Sources/CTDiffusion.pdf"
authors: [Xinlu He, Swayambhu Nath Ray, Harish Mallidi, Jia-Hong Huang, Ashwin Bellur, Chander Chandak, M. Maruf, Venkatesh Ravichandran]
year: 2025
venue: "NeurIPS 2025 Workshop SPIGM"
tags: [TTS, diffusion, autoregressive, continuous-representation, MLLM, zero-shot, speaker-cloning, exposure-bias]
concepts: ["[[Next-TokenDiffusion]]", "[[LLM-basedTTS]]", "[[Diffusion-basedTTS]]", "[[VariationalAutoencoderforTTS]]", "[[MaskedGenerativeModeling]]", "[[Classifier-FreeGuidance]]", "[[SpeakerEmbedding]]"]
models: ["[[模型库/NaturalSpeech3|NaturalSpeech 3]]", "[[模型库/MELLE|MELLE]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 7
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文属于 **Next-Token Diffusion** 路线中的一员,即在 causal LM 的每个 token 位置挂载轻量 diffusion head 生成连续 latent [待确认]。这条路线由 LatentLM (Sun et al., 2024) 开创,后续有 CLEAR、VibeVoice、SemaVoice 等跟进。本文的特点在于: (1) 使用 OPT-125M 作为 backbone 而非更大的 LLM; (2) 提出 dual-head 架构保留 LM head 用于 EOS 控制; (3) 系统性地研究了 masked training 和 two-stage training 两种训练策略来解决 exposure bias 和 distribution drift 问题。

**已有认知**:
- **LLM-based TTS** (confirmed): 将 TTS 重构为条件语言建模任务,主流使用离散 codec/semantic tokens + AR 生成。本文延续 LLM-based 范式但用连续表示替代离散 tokens。
- **Semantic vs Acoustic Tokens** (confirmed): 离散 tokens 存在量化损失,丢失细粒度声学信息。本文正是针对这一痛点,用 VAE 连续 latent 替代离散 codebook tokens。
- **Zero-shot Speech Synthesis** (confirmed task): 本文的目标任务,通过 3 秒参考音频实现 speaker-referenced 合成。当前 SOTA 在 SEED-TTS-Eval 上,WER 已低至 ~1%。
- **Speaker Embedding** (confirmed): 本文使用 LAM (768 维) 提取 speaker embedding 作为 LLM 条件。
- **Next-Token Diffusion** [待确认]: 本文的 diffusion head 是逐帧 MLP denoiser,与 LatentLM 的 per-token DDPM head 同源。区别在于 LatentLM 用 sigma-VAE + 3-6 层 head,本文用标准 VAE vocoder + 12 层 MLP。
- **Diffusion-based TTS** [待确认]: 本文属于 "diffusion 嵌入 AR" 而非独立 diffusion 声学模型。
- **Variational Autoencoder for TTS** [待确认]: 本文的 VAE vocoder (64-dim, 25 fps) 既是 target encoder 也是 waveform decoder,与 LatentLM 的 sigma-VAE 设计不同(本文未采用 sigma-VAE)。

**创新判断**: 相比 LatentLM/CLEAR 等前作,本文的主要增量在于 (1) dual-head EOS 控制机制和 (2) two-stage training 策略,后者带来 46% 相对 WER 降低。模型仅 160M 参数,远小于 LatentLM (1B+) 和 CLEAR (规模未公开) 等同路线工作。

> 检索命中: [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeakerEmbedding]]✓, [[Zero-shotSpeechSynthesis]]✓ | 参考(待确认): [[Next-TokenDiffusion]], [[Diffusion-basedTTS]], [[VariationalAutoencoderforTTS]] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 MLLM 中引入逐帧 continuous-token diffusion head + dual-head 架构 + two-stage 训练,以 160M 参数实现 SOTA AR zero-shot TTS (WER 1.95%, SIM 0.54)
> - **路线**: Text + 3s ref audio → OPT-125M (causal LM) → dual-head: LM head 控制 EOS + diffusion head (12-layer MLP, DDPM) 逐帧去噪 64-dim VAE latent → VAE decoder → waveform
> - **指标**: WER 1.95% / SIM-R 0.54 / UTMOS 4.00 on LibriSpeech(PC) test-clean; two-stage vs one-stage: 46% 相对 WER 降低 (3.61%→1.95%) [Table 1]
> - **可借鉴**: (1) Two-stage 训练策略(先联合训练后冻结 LM 单独训练 diffusion head)解决 distribution drift,通用于所有 LM + diffusion head 联合训练场景; (2) Masked training (30% mask ratio) 缓解连续 AR 的 exposure bias; (3) 用 LM head 预测 `<cont_speech_gen>` + `<eos>` 控制变长语音生成,无需外部 duration predictor
> - **局限**: (1) 仅在 LibriSpeech(PC) 评估,无中文/多语言验证; (2) 模型仅 160M,scalability 未验证; (3) 推理需 100 步 diffusion denoising,速度可能是瓶颈; (4) 未开源

## 核心问题

本文要解决的核心问题是: **如何在 MLLM 框架内,用连续语音表示替代离散 tokens 进行自回归 TTS,同时解决 exposure bias 和联合训练不稳定两个关键难题?**

具体而言:
1. 离散 token 的量化瓶颈: 当前 MLLM-based TTS 依赖离散 codec tokens,量化过程丢失细粒度声学信息 [§1]
2. 连续 AR 的 exposure bias: 训练时用 teacher-forcing (ground-truth 历史),推理时用自己的预测,小误差在帧级别逐帧累积 [§3.3]
3. 联合训练的 distribution drift: LLM 输出分布随训练动态变化,导致 diffusion head 的输入分布非稳态,收敛不可靠 [§3.4]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文提出 **dual-head MLLM** 架构 [§3, Fig 1]:

```
Text prompt ("Speak {TEXT} in the voice of [REF_AUDIO]")
    ↓
Speaker ref audio → LAM encoder → 768-dim embedding → projector → LLM input space
    ↓
OPT-125M (causal LLM backbone)
    ↓ hidden state z_i
    ├── LM Head: 预测控制 token (<sbos>, <cont_speech_gen>, <eos>)
    └── Diffusion Head: 12-layer MLP denoiser, DDPM, z_i 条件化
                         ↓ 从 Gaussian noise 迭代去噪
                    64-dim continuous embedding x̂_i
                         ↓
                    VAE decoder → waveform
```

**语音表示**: VAE-based vocoder 将 48kHz 立体声编码为 64-dim embedding @ 25 fps (5 个 ResNet downsampling blocks) [§4.2]。

### 关键设计选择

**1. Dual-head 架构 — 为什么不直接只用 diffusion head?**

[论文原文] 保留 LM head 有两个目的 [§3.2]:
- **变长控制**: LM head 预测 `<speech_bos>` 启动语音生成,每帧预测 `<cont_speech_gen>` 表示继续生成,预测 `<eos>` 终止。这让模型自主决定语音长度,无需外部 duration predictor 或 oracle endpoint。
- **多任务兼容**: 保留 LM head 使模型可在 MLLM 框架内与其他模态任务共存,不需要额外的 classifier 模块。

[agent 解读] `<cont_speech_gen>` token 虽然推理时不被输出到序列中,但训练时对每帧都提供 dense cross-entropy supervision,可能缓解模型在只有首尾 boundary token 监督时的 premature EOS 倾向(论文未做该消融,仅称"reduces the risk" [§3.2])。这个设计比 VALL-E 的外部 classifier 更统一。

**2. Masked training — 为什么 30% 是最优?**

[论文原文] 将一定比例的输入帧替换为零向量,模拟推理时自身预测可能出错的场景 [§3.3]。每帧独立以概率 p_mask 被替换: v^t ~ Bernoulli(1 - p_mask), x̃ = x ⊙ v。

消融 [Table 2]: 0% mask → WER 15.06% (严重 exposure bias); 15% → 12.65%; **30% → 6.17%** (最优); 50% → 8.13% (过度破坏语义对齐) [§5.2]。

[agent 解读] 30% 的甜蜜点可以理解为: 太少则训练/推理条件差距仍大,太多则模型无法学到有效的上下文依赖。这与 BERT 的 15% masking 和 audio generation 中的 masked training (Yang et al., 2025) 的发现一致,但具体比例受连续表示的帧间依赖特性影响。

**3. Two-stage training — 为什么不一直端到端?**

[论文原文] 作者观察到端到端联合训练时,虽然 loss 单调下降,但 WER 先降后升 [§3.4]。假设原因是 LLM 输出分布 p_θ(z) 随参数 θ 更新而变化,使 diffusion head 需要适应不断漂移的输入分布 (distribution drift)。

解决方案 [§3.4, Fig 2c]:
- **Stage 1**: 联合训练 LLM backbone + LM head + diffusion head,使用 early stopping 选择最低 validation WER 的 checkpoint
- **Stage 2**: 冻结 LLM backbone + LM head + speech projector,仅训练 diffusion head。此时 p_θ(z) 固定,diffusion head 可在稳态分布上精细优化

效果 [Table 1]: Stage-1 baseline WER 3.61% → Stage-2 WER 1.95% (46% 相对降低), SIM 0.49→0.54, UTMOS 3.21→4.00。

[agent 解读] 这个发现与 LatentLM/CLEAR 等工作形成互补 — LatentLM 使用端到端训练但未报告 distribution drift 问题,可能因为其 sigma-VAE 的 variance 设计提供了更好的鲁棒性。本文使用标准 VAE (非 sigma-VAE),因此对 distribution drift 更敏感。Two-stage 策略本质上是将"学习 LM 表示"与"学习 latent-to-speech 映射"解耦,类似 curriculum learning 的思路。

### 训练策略

- **Diffusion 过程**: DDPM, T=1000 步, cosine noise schedule [§4.2]
- **Denoiser**: MLP with residual blocks, 12 层, layer normalization + SiLU + adaptive layer normalization modulation (AdaLN), 无 dropout [§4.2]
- **损失函数**: L = L_LM (cross-entropy) + L_diff (noise prediction MSE) [§3.2]
- **Stage 1**: lr warmup 3e-5→3e-4, cosine decay, 300k steps [§4.2]
- **Stage 2**: constant lr 2e-4, 300k steps [§4.2]
- **硬件**: NVIDIA A100, batch size 2048, FP16 [§4.2]
- **推理**: 100 denoising steps, temperature 0.9, CFG=1 [§4.2]

## 实验

| 指标 | 本文 (CTDiffusion) | Stage-1 Baseline | VALL-E (AR+NAR, 400M) | MegaTTS (AR+NAR, 500M) | Voicebox (NAR, 400M) | StyleTTS2 (NAR, 700M) | Ground Truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER (%) ↓ | **1.95** | 3.61 | 6.11 | 2.32 | 2.14 | 2.49 | 2.84 | LibriSpeech(PC) test-clean | [Table 1] |
| SIM ↑ | **0.54** | 0.49 | 0.47 | 0.53 | 0.48 | 0.38 | 0.69 | LibriSpeech(PC) test-clean | [Table 1] |
| UTMOS ↑ | 4.00 | 3.21 | 3.68 | **4.02** | 3.73 | 3.94 | 4.16 | LibriSpeech(PC) test-clean | [Table 1] |

**关键消融结果**:

| 实验 | 设置 | WER (%) | SIM-R | UTMOS | 出处 |
| --- | --- | --- | --- | --- | --- |
| Masking ratio | 0% (无 mask) | 15.06 | 0.45 | 2.00 | [Table 2] |
| Masking ratio | 30% (最优) | 6.17 | 0.46 | 3.21 | [Table 2] |
| Masking ratio | 50% | 8.13 | 0.46 | 2.84 | [Table 2] |
| Diffusion depth | 3-layer MLP, no stage-2 | 6.17 | 0.46 | 3.10 | [Table 3] |
| Diffusion depth | 12-layer MLP, no stage-2 | 3.61 | 0.49 | 3.21 | [Table 3] |
| Diffusion depth | 12-layer MLP, with stage-2 | **1.95** | **0.54** | **4.00** | [Table 3] |
| Stopping criteria | GT duration | 29.36 | 0.48 | 2.55 | [Table 4] |
| Stopping criteria | EOS token (本文) | 3.61 | 0.49 | 3.21 | [Table 4] |
| Inference temp | 1.0, 100 steps | 7.53 | 0.48 | 3.27 | [Table 5] |
| Inference temp | 0.9, 100 steps | **1.95** | **0.54** | **4.00** | [Table 5] |
| Inference temp | 0.8, 100 steps | 16.11 | 0.45 | 3.01 | [Table 5] |

**关键发现**:
- 使用 ground-truth duration 反而 WER 暴增至 29.36% [Table 4],作者解释为"unstable outputs" [§5.2]。[agent 解读] 这可能是因为强制使用固定长度导致 diffusion head 无法自适应调节生成节奏。
- Temperature 对结果影响极大: 0.9 最优,0.8 和 1.0 都显著恶化 [Table 5]。[agent 解读] 这表明 diffusion sampling 的方差控制至关重要,模型对采样参数敏感性较高,在实际部署中需要精细调参。

## 局限性

1. **评估范围有限**: 仅在 LibriSpeech(PC) test-clean 评估,无中文/多语言/大规模 benchmark (如 SEED-TTS-Eval) 验证 [Limitations 节]。与领域 SOTA (CosyVoice 3 WER 1.45% on SEED-TTS-Eval test-en) 不具直接可比性。
2. **模型规模受限**: 仅验证 OPT-125M (0.5B→160M),未探索 scaling behavior [Limitations 节]。同路线的 LatentLM 使用更大 LM,CLEAR 等未公开规模。
3. **推理效率**: 100 步 diffusion denoising per frame,加上 25 fps,生成 10 秒语音需要 250 帧 × 100 步 = 25000 次 MLP forward。虽然 MLP 轻量,但与 CLEAR 的 MLP rectified flow (单步或少步) 相比,仍有差距。
4. **Temperature 敏感性**: 最优 temperature (0.9) 与次优 (1.0) 间 WER 差距巨大 (1.95% vs 7.53%),鲁棒性存疑 [Table 5]。
5. **未采用 sigma-VAE**: 与 LatentLM/CLEAR 不同,本文使用标准 VAE vocoder,未讨论 variance collapse 问题,这可能是需要 two-stage training 的根因之一。
6. **未开源**: 代码和模型权重未发布。

## 点评

**贡献层面**:
- Two-stage training 是本文最有价值的发现。Distribution drift 问题在 LM + diffusion head 联合训练中具有普遍性,这一策略可推广到其他类似架构。46% 相对 WER 降低的幅度令人信服 [Table 1]。
- Dual-head 的 `<cont_speech_gen>` token 设计比 VALL-E 的外部 classifier 更统一,且实验证明 EOS token 控制优于 oracle duration [Table 4]。
- 消融实验全面,covering masking ratio、diffusion depth、stopping criteria、inference hyperparameters,为后续工作提供了有用的参考。

**不足层面**:
- 仅在 LibriSpeech(PC) 上评估,这是一个相对简单的英语朗读数据集,无法代表实际 zero-shot TTS 的挑战 (多语言、情感、噪声、out-of-domain text)。缺少与 SEED-TTS-Eval 等标准 benchmark 的对比。
- 160M 的模型规模远小于当前主流 (Qwen2.5-1.5B for SemaVoice, 1.5B for CosyVoice 3),结果的可推广性不确定。
- WER 1.95% 虽然在 LibriSpeech(PC) 上超越了 baselines,但低于 ground-truth 的 WER 2.84%,说明存在非自然过度 articulation 的可能 [agent 解读]。
- 未与同路线的 LatentLM 在同一 benchmark 上直接对比,虽然引用了 LatentLM 但未做实验对比。

## 可复用的 idea

1. **Two-stage training for LM + diffusion head**: 先联合训练使 LM 适应任务,再冻结 LM 单独优化 diffusion head。这个策略对任何 "backbone + lightweight generative head" 架构都适用,核心洞察是 head 需要稳态输入分布才能精细优化。
2. **Dense control token supervision**: 在每帧位置用 `<cont_speech_gen>` 提供 cross-entropy supervision 信号,而非仅在 boundary 提供。可推广到所有需要变长生成的场景。
3. **Masking ratio 调参**: 在本文的 3-layer diffusion head + 64-dim VAE 设置下,30% 是连续 AR 语音生成中 exposure bias mitigation 的经验最优点,比 BERT 的 15% 高,比 50% 低。不同 head depth 或 latent 维度下最优值可能不同。

> [!review] 审阅
> 待审阅 — 见 `_review/CTDiffusion-review.yml`
