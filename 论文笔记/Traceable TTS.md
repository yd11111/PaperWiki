---
type: paper
tier: deep
title: "Traceable TTS: Toward Watermark-Free TTS with Strong Traceability"
arxiv_id: "2507.03887"
source: "Sources/TraceableTTS.pdf"
authors: [Yuxiang Zhao, Yunchong Xiao, Yushen Chen, Zhikang Niu, Shuai Wang, Kai Yu, Xie Chen]
year: 2025
venue: "arXiv preprint"
tags: [TTS, traceability, deepfake-detection, model-attribution, joint-training, watermark-free, security, speech-synthesis]
concepts: ["[[Anti-spoofing and Deepfake Detection]]", "[[Conditional Flow Matching]]", "[[TTS Evaluation]]", "[[Voice Cloning Taxonomy]]", "[[Self-Supervised Speech Representation]]"]
models: ["[[wav2vec 2.0]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页: [[Conditional Flow Matching]]✓, [[Anti-spoofing and Deepfake Detection]][待确认], [[TTS Evaluation]][待确认], [[Voice Cloning Taxonomy]][待确认], [[wav2vec 2.0]][待确认], [[Self-Supervised Speech Representation]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Conditional Flow Matching]]✓ | 参考: [[Anti-spoofing and Deepfake Detection]](pending-review), [[TTS Evaluation]](pending-review), [[Voice Cloning Taxonomy]](pending-review), [[wav2vec 2.0]](pending-review), [[Self-Supervised Speech Representation]](pending-review) | 未命中但可能相关: Model Attribution, GAN Fingerprint(概念库中无独立页)

**谱系定位**: 本文处于 Anti-spoofing and Deepfake Detection 的主动溯源分支。Anti-spoofing 页记录了从被动检测 (deepfake detection) → 主动防护 (SafeSpeech) → 主动溯源 (TraceableSpeech watermarking) 的演进线。本文提出 **watermark-free traceability**,是在 TraceableSpeech (Zhou et al., 2024) 的 codec-watermark 方案之后的更激进一步: 完全不嵌入显式水印,而是通过联合训练让 TTS 模型自动产生可被配对判别器识别的隐式特征。这代表了从 "主动嵌入标记" 到 "自然产生指纹" 的范式跃迁。

**已有认知**: Conditional Flow Matching 页 (confirmed) 记录了 F5-TTS 使用 CFM + DiT 实现端到端 TTS,支持 zero-shot 生成;wav2vec 2.0 页记录了其作为自监督语音表征提取器的能力,在 deepfake detection 领域已有广泛应用 (wav2vec 2.0 + LCNN 是竞争性 baseline)。Anti-spoofing 页记录了 vocoder fingerprint 的可区分性 (Yan et al., 2022) 和 acoustic model fingerprint 被 vocoder fingerprint 遮蔽的问题 (Zhang et al., 2024)。

**创新判断**: 本文的核心创新在于 **反转 GAN 的 generator loss**: 传统 GAN 中 generator 和 discriminator 对抗 (adversarial),本文让两者 **同向优化** (collaborative),使 generator 主动产生更易被 discriminator 识别的特征。这是概念级创新,与 TraceableSpeech 的 codec-level watermark embedding 完全不同。

> [!summary] 速查
> - **一句话**: 通过反转 GAN generator 的 loss 方向实现 TTS 模型与判别器的协同训练,让模型自然产生可追溯的隐式指纹,无需嵌入任何显式水印
> - **路线**: Text → F5-TTS (CFM+DiT, Mel) → Vocos (vocoder) → Speech; 判别器: Speech → wav2vec 2.0 (feature) → LCNN (classifier) → Binary (yes/no this model); 训练 10 loops 交替 finetune TTS + 生成数据 + 训练判别器
> - **指标**: 域内 AUC 0.9999 / EER 0.29% [Table 2]; 域外泛化 AUC 0.9421 vs baseline 0.8823, EER 11.50% vs 18.99% [Table 3]; 语音质量 WER 2.033% vs 原始 2.202%, UTMOS 3.958 vs 3.926 [Table 5] (LibriSpeech-PC test-clean)
> - **可借鉴**: (1) 反转 GAN loss 实现协同训练的思路,可推广到任何需要让生成器输出携带可识别特征的场景 (如品牌音色标记、生成内容溯源); (2) 三阶段循环训练 (finetune→infer→train-disc) 的迭代增强策略
> - **局限**: 仅适用于端到端可微的 TTS (不支持 VALL-E 等 discrete token 模型); 域外 EER 仍 11.5%; 噪声/变调攻击下精度大幅下降 (~13%); 仅在 F5-TTS + LibriTTS 上验证; 未讨论多模型共存场景

## 核心问题

TTS 合成语音的安全溯源面临三层递进困境 [§1]:

1. **被动检测不可靠** — Deepfake detection (real vs fake 二分类) 依赖已知模型的伪影,面对新型 TTS 模型泛化困难,容易在 out-of-domain 场景失败 [§1]
2. **显式水印有代价** — 传统 audio watermark (如 WavMark) 和 model watermark (如 TraceableSpeech, WMCodec) 嵌入 n-bit 水印信息,不可避免地影响语音质量;且水印本身可被伪造或去除,引入新的安全风险 [§1]
3. **水印方案受限于架构** — Model watermark (如 TraceableSpeech) 仅适用于 codec-based TTS,且无法应对 vocoder 替换场景 [§1]

本文的核心洞察: 如果不嵌入显式水印,而是让 TTS 模型在联合训练过程中自然产生配对判别器可识别的隐式特征 (implicit fingerprint),就能同时消除水印对质量的影响和水印泄露的安全风险 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由两个核心组件构成 [§3.1, Fig 2]:

**Generator**: F5-TTS — 基于 Conditional Flow Matching + DiT 的全 NAR TTS 系统 [§3.2, Fig 3]
- 使用 ConvNeXt V2 处理文本输入,增强 text-speech 对齐
- 通过 Flow Matching 从 Gaussian noise 逐步生成 Mel spectrogram
- 预训练 Vocos vocoder (冻结) 将 Mel 转为波形
- 选择 F5-TTS 的理由: 从输出波形到输入文本端到端可微,BCE loss 可回传优化整个模型 [§3.2] [论文原文]

**Discriminator**: wav2vec 2.0 (feature extractor) + LCNN (classifier) [§3.2, Fig 3]
- wav2vec 2.0: 7 层 1D CNN encoder + Gumbel-Softmax 量化 + Transformer encoder, 提取自监督语音表征
- LCNN: 9 层轻量 CNN + max pooling + batch norm + global pooling + FC + sigmoid → 二分类
- 选择依据: 先前 deepfake detection 工作表明 wav2vec 2.0 + LCNN 是性能优异的组合 [§2.3, §3.2] [论文原文]

### 关键设计选择

**1. 反转 GAN Generator Loss (核心创新)** [§3.1, Eq 3-4]

传统 GAN [Eq 1-2]:
- Discriminator: max log D(x) + log(1 - D(G(z))) (识别真假)
- Generator: min log(1 - D(G(z))) (欺骗判别器)

本文 [Eq 3-4]:
- Discriminator: max log D(x) + log(1 - D(G(z))) (与传统相同)
- Generator: **max** log(1 - D(G(z))) (与 discriminator 同向, **帮助** 判别器识别自己)

WHY: 传统 GAN 中 generator 和 discriminator 目标对立 — generator 想让 discriminator 分不清;但本文的目标恰好相反: 希望 generator 产生的语音更容易被自己的配对 discriminator 识别。通过对齐优化方向,discriminator 学习 generator 特有的判别特征,generator 在 TTS loss 约束下产生人耳不可察但 discriminator 可检测的隐式指纹 [§3.1] [论文原文]。

[agent 解读] 这本质上是把 GAN 从对抗博弈变成了合作博弈。TTS loss (L_TTS) 充当质量约束,防止 generator 产生人耳可感知的差异;BCE loss (L_BCE) 驱动 generator 在质量约束内最大化判别器的识别能力。两者的平衡由 λ_TTS 和 λ_BCE 控制。

**2. 三阶段循环训练** [§3.3]

每个 loop 包含三个阶段:

Stage 1 — **Finetune F5-TTS**: 在 LibriTTS train-clean-100 上微调,L = λ_TTS · L_TTS + λ_BCE · L_BCE (λ_TTS = λ_BCE = 1),每个 loop 训练 2 epochs,从前一个 loop 的 checkpoint 继续 [§3.3, §4.3]

Stage 2 — **Inference**: 用微调后的 F5-TTS 重建 LibriTTS dev-clean (5734 条),采用 voice cloning 模式 (ref-audio + ref-text + gen-text) 以排除说话人/文本信息干扰 [§3.3]

Stage 3 — **Train Discriminator**: 训练数据 = {微调 F5-TTS 生成 (label 1), ground truth (label 0), **原始** F5-TTS 生成 (label 0)}。20 epochs, Adam lr=5e-6, L2 reg=1e-4, gradient clip=1.0 [§3.3, §4.2-4.3]

WHY 引入原始 F5-TTS 数据 (label 0): 如果只区分合成 vs 真实,这是一个太简单的任务,discriminator 会快速收敛但学不到微调后模型的特有特征。加入原始 F5-TTS (label 0) 迫使 discriminator 学习区分"微调后 F5-TTS"和"未微调 F5-TTS",从而聚焦于联合训练引入的隐式指纹 [§3.3] [论文原文]。

**训练后处理**: 10 个 loop 完成后,丢弃原始 F5-TTS 数据,再训练 discriminator 10 epochs,使其不再关注微调前后的差异,而专注于微调后 F5-TTS 本身的特征 [§4.3] [论文原文]。

**3. Vocoder-robust 设计** [§1]

水印嵌入在 vocoder 层面的方案 (如 TraceableSpeech, WMCodec) 在 vocoder 被替换后失效。本文的 traceability 特征嵌入在 Mel spectrogram 生成阶段 (TTS model 层面),vocoder 仅做 Mel → waveform 转换,因此更换 vocoder 不影响可追溯性 [§1] [论文原文]。

[agent 解读] 但论文没有实验验证这一声称 — 所有实验均使用 Vocos 作为 vocoder,未测试替换为 BigVGAN 或 HiFi-GAN 后的鲁棒性。

### 训练策略

**总 Loss** [§3.3, Eq 5]:

$$L = \lambda_{TTS} L_{TTS} + \lambda_{BCE} L_{BCE}$$

- L_TTS: F5-TTS 原始训练 loss (Flow Matching loss)
- L_BCE: Binary Cross-Entropy loss,来自 discriminator 对 generator 输出的判定 (optimization goal 与 discriminator 训练一致,即 generator 输出应被判为 label 1)
- λ_TTS = 1, λ_BCE = 1 [§3.3]

**训练规模**:
- F5-TTS finetune: 基于开源 1,200,000-step checkpoint, 每 loop 2 epochs on train-clean-100 [§4.3]
- Discriminator: 每 loop 20 epochs, 最终 +10 epochs [§4.3]
- 总共 10 loops [§4.3]

## 实验

### 实验设置 [§4.1-4.3]
- 数据集: LibriTTS (586h, 2456 speakers, 24kHz) [§4.1]
- F5-TTS: 开源 checkpoint (trained on Emilia 95kh),finetune on train-clean-100 [§4.1]
- Discriminator: 训练数据从 dev-clean 重建,测试数据从 test-clean 重建 [§4.1]
- Baseline: 独立训练 discriminator (wav2vec 2.0 + LCNN),不做 F5-TTS 联合训练 [§4.2]
- 域外测试: 混入 CosyVoice, CosyVoice 2, E2-TTS 生成的音频 (label 0) [Table 1]

### 域内二分类 [Table 2]

| 指标 | Baseline | Ours | 出处 |
| --- | --- | --- | --- |
| AUC ↑ | 0.9990 | 0.9999 | [Table 2] |
| EER ↓ | 0.17% | 0.29% | [Table 2] |
| ACC ↑ | 99.90% | 99.74% | [Table 2] |

[agent 解读] 域内结果两者几乎相同,说明区分特定 TTS vs 真实语音本身就是简单任务;联合训练的价值不在域内。

### 域外泛化 (核心结果) [Table 3]

| 指标 | Baseline | Ours | 出处 |
| --- | --- | --- | --- |
| AUC ↑ | 0.8823 | **0.9421** | [Table 3] |
| EER ↓ | 18.99% | **11.50%** | [Table 3] |
| ACC ↑ | 85.60% | **89.38%** | [Table 3] |

域外数据包含 F5-TTS (label 1) + ground truth + CosyVoice + CosyVoice 2 + E2-TTS (label 0) [Table 1]。联合训练将 EER 从 18.99% 降至 11.50%,AUC 从 0.8823 提升至 0.9421 [Table 3]。

### 鲁棒性 (攻击抵抗) [Table 4]

| 攻击类型 | 域内 Diff ACC ↑ | 域外 Diff ACC ↑ | 出处 |
| --- | --- | --- | --- |
| Sample rate (24k→8k→16k) | -1.79% | -4.24% | [Table 4] |
| 1.2x speed | -0.01% | -2.48% | [Table 4] |
| 0.8x speed | -0.02% | -4.28% | [Table 4] |
| MUSAN noise (0.01) | **-9.37%** | **-13.74%** | [Table 4] |
| Reverb | +0.02% | -6.29% | [Table 4] |
| Pitch (+4 semitones) | -1.85% | **-7.09%** | [Table 4] |
| Volume (0.5x) | +0.10% | -1.38% | [Table 4] |
| MP3 compression | -0.01% | -1.27% | [Table 4] |
| WAV (MP3→WAV) | +0.02% | -1.76% | [Table 4] |

MUSAN 加噪和 pitch shift 是主要挑战,域外场景下分别导致 ~13.7% 和 ~7.1% 的精度下降 [Table 4]。

### 语音质量 [Table 5]

| 指标 | F5-TTS (原始) | Ours (联合训练后) | 出处 |
| --- | --- | --- | --- |
| WER ↓ | 2.202% | **2.033%** | [Table 5] |
| SIM ↑ | 0.659 | **0.661** | [Table 5] |
| UTMOS ↑ | 3.926 | **3.958** | [Table 5] |

联合训练后三项指标均略有提升 [Table 5],验证了 watermark-free 方案不损害甚至轻微改善语音质量 [§5.6]。

[agent 解读] 质量提升可能来自额外 finetune 的正则化效果,而非 joint training 本身的功劳。没有消融实验区分"仅 finetune"和"finetune + joint training"的贡献。

## 局限性

1. **仅适用于端到端可微模型** — 明确指出 VALL-E 等 discrete token-based TTS 不适用,因为 BCE loss 无法回传通过离散量化步骤 [§6] [论文原文]; 这意味着当前主流 codec LM TTS (VALL-E, CosyVoice, Spark-TTS) 均被排除
2. **域外泛化仍不充分** — EER 11.5% 意味着每 9 个判断中约有 1 个错误;在安全关键场景中这不够可靠 [Table 3] [agent 解读]
3. **噪声/变调攻击脆弱** — MUSAN 加噪导致域外精度下降 13.7%,pitch shift 下降 7.1% [Table 4]; 论文承认这是主要挑战 [§5.5]
4. **缺少关键消融** — 未区分"仅 finetune F5-TTS"和"finetune + joint training"的贡献;质量提升可能来自额外 finetune 而非协同训练 [agent 解读]
5. **单模型验证** — 仅在 F5-TTS 上验证,未测试 FastSpeech 2、VITS 等其他端到端可微 TTS [§4] [agent 解读]
6. **未讨论多模型共存** — 如果 N 个 TTS 厂商各自部署联合训练,N 个 discriminator 之间是否会互相干扰? 一段语音被多个 discriminator 同时识别为"自己的"怎么处理? [agent 解读]
7. **Vocoder 替换未实验验证** — 声称对 vocoder 替换鲁棒但所有实验仅用 Vocos [§1 vs §4] [agent 解读]
8. **仅英文单数据集** — 所有实验在 LibriTTS 上完成 [§4.1]

## 点评

本文提出了一个概念上优雅的框架: 通过反转 GAN generator 的 loss 方向,将对抗博弈变为合作博弈,使 TTS 模型主动产生可追溯的隐式指纹而非依赖显式水印。这种 "watermark-free traceability" 的定位在安全领域有独特价值 -- 消除了水印带来的质量损失和水印泄露的安全风险。

与 vault 中已有的 TraceableSpeech (Zhou et al., 2024) 相比,两者代表了 TTS 溯源的两条不同路线: TraceableSpeech 将水印视为一等公民并联合优化,本文则彻底抛弃显式水印。TraceableSpeech 在其方案内做到了极高的提取精度 (resplicing 攻击后仍 100%),而本文的域外 EER 仍有 11.5%。这说明 watermark-free 方案虽然概念更激进,但在当前实验条件下精度远不及 watermark-based 方案。

本文的主要弱点是实验验证的广度和深度不足: (1) 仅在一个 TTS 模型上验证; (2) 缺少关键消融 (joint training vs pure finetuning); (3) 未验证 vocoder 替换鲁棒性这一核心声称。"first work" 的定位为其争取到了一定的宽容度,但如果要推动这个方向,后续工作需要在多模型、跨数据集、更强攻击场景下进行系统验证。

从领域视角看,本文最有价值的贡献不是具体方法,而是问题定义: 将 TTS 溯源从 "在语音中嵌入什么" 转变为 "让模型本身产生什么",为后续研究打开了新的设计空间。

## 可复用的 idea

1. **反转 GAN Generator Loss 实现协同训练**: 在任何需要让生成器输出携带可识别特征但不损害主任务质量的场景中适用 — 如品牌音色嵌入、版权保护、生成内容分级标记
2. **三阶段循环训练 (finetune→infer→train-disc)**: 解决了联合训练中 generator 和 discriminator 数据分布漂移的问题。每个 loop 用最新 generator 重新生成训练数据,保证 discriminator 始终对当前 generator 有效
3. **引入 "原始模型" 作为 hard negative**: 在 discriminator 训练中加入未微调模型的输出 (label 0),迫使 discriminator 学习微调引入的特有特征而非通用合成伪影 — 这个 trick 可推广到任何模型指纹/归因任务

---

> [!review] 审阅结论: pass (2026-06-03)
> 五个原则均满足, 2 个 low issue (traceability-gap ×1, template-compliance ×1) 均不阻塞.
> 详见 `_review/Traceable TTS-review.yml`

检索命中: [[Conditional Flow Matching]]✓ | 参考: [[Anti-spoofing and Deepfake Detection]](pending-review), [[TTS Evaluation]](pending-review), [[Voice Cloning Taxonomy]](pending-review), [[wav2vec 2.0]](pending-review), [[Self-Supervised Speech Representation]](pending-review) | 未命中但可能相关: Model Attribution, GAN Fingerprint(概念库中无独立页)
