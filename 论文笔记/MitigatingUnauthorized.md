---
type: paper
tier: deep
title: "Mitigating Unauthorized Speech Synthesis for Voice Protection"
arxiv_id: "2410.20742"
source: "Sources/MitigatingUnauthorized.pdf"
authors: [Zhisheng Zhang, Qianyi Yang, Derui Wang, Pengyang Huang, Yuxin Cao, Kai Ye, Jie Hao]
year: 2024
venue: "arXiv preprint"
tags: [voice-protection, adversarial-perturbation, unlearnable-examples, data-poisoning, voice-cloning, deepfake-defense, error-minimizing]
concepts: ["[[Anti-spoofingandDeepfakeDetection]]", "[[SpeakerVerification]]", "[[MelSpectrogram]]", "[[VoiceCloningTaxonomy]]", "[[VariationalAutoencoderforTTS]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeakerEmbedding]], [[Anti-spoofingandDeepfakeDetection]], [[SpeakerVerification]], [[VoiceCloningTaxonomy]], [[MelSpectrogram]], [[VariationalAutoencoderforTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 voice protection / proactive defense 方向,是 [[Anti-spoofingandDeepfakeDetection]] 中"主动防御"分支的早期工作。KB 中已有的 SafeSpeech(同一第一作者,USENIX Security 2025)是本文的直接后续升级版,POP 可视为 SafeSpeech 的技术原型。与 KB 中 deepfake detection(被动检测)和 watermarking(可追踪)路线不同,POP 走的是"让数据在训练阶段不可学"的主动扰动路线,但仅覆盖 fine-tuning 场景,尚未扩展到 zero-shot。
>
> **已有认知**: [[VITS]] 模型库页记录了 VITS 的 VAE+Flow+GAN 端到端架构和其多目标 loss 函数(reconstruction + KL + duration + adversarial + feature-matching),POP 正是基于对 VITS loss 结构的分析提出 pivotal objective 选择策略。[[VariationalAutoencoderforTTS]] 页记录了 VAE 在 TTS 中的 encoder-decoder 结构和训练目标,POP 的核心洞察是利用 VAE 重建损失的通用性。[[MelSpectrogram]] 页记录了 mel 频谱作为 TTS 中间表示的标准流程,POP 的 pivotal objective 正是 mel 距离。[[SpeakerVerification]] 页记录了 ECAPA-TDNN 等 speaker encoder 的标准做法,但本文未直接针对 SV 做优化。
>
> **创新判断 (对比 SafeSpeech)**: POP 提出了 pivotal objective selection 的核心思想(SafeSpeech 直接继承),但缺少 SPEC(KL 引导输出趋近噪声)和感知优化(STOI+STFT)两个关键组件。POP 的防护场景仅限 fine-tuning(3 个 TTS 模型),SafeSpeech 扩展到 10 个模型(含 5 个 zero-shot)。
>
> 检索命中: [[SpeakerEmbedding]]✓, [[Anti-spoofingandDeepfakeDetection]][待确认], [[SpeakerVerification]][待确认], [[VoiceCloningTaxonomy]][待确认], [[MelSpectrogram]][待确认], [[VariationalAutoencoderforTTS]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 Pivotal Objective Perturbation (POP),在原始语音中嵌入不可感知的 error-minimizing 扰动,使 TTS 模型在 fine-tuning 时无法学到有效的语音分布,从而阻止高质量仿冒语音合成
> - **路线**: 原始音频 → 代理 TTS 模型(VITS)→ 分析多目标 loss → 选择 mel reconstruction loss 作为 pivotal objective → PGD 生成 error-minimizing 扰动 → 受保护音频(position-fixed patch perturbation)→ 攻击者 fine-tune 任意 TTS 模型均生成低质量语音
> - **指标**: MB-iSTFT-VITS LibriTTS WER 21.9%→127.3%, MCD 5.83→13.65 [Table 1]; 跨模型迁移 VITS WER 27.0%→124.0% [Table 3]; 受保护音频 SNR 17.9dB, PESQ 3.55 [Fig 4]; MOS 降至 0.13-0.18(VITS/MB-iSTFT-VITS) [Table 2]
> - **可借鉴**: pivotal objective selection -- 面对多目标 TTS loss,分析哪个 loss 与音频输入直接相关且跨模型通用,只优化它做扰动生成;position-fixed perturbation 减少计算开销同时提高不可感知性
> - **局限**: 仅覆盖 fine-tuning 场景(不含 zero-shot);无感知优化(仅用 L_p norm);MOS 评估仅 61 人;未测试 codec LM 架构;被同作者 SafeSpeech (2025) 全面升级

## 核心问题

POP 要解决的核心问题是: **公开可获取的语音数据被攻击者非法获取后,可通过 TTS 模型 fine-tuning 生成高质量仿冒语音,造成隐私泄露和安全威胁** [§1]。

具体而言,作者识别了现有防护方法的三个不足 [§1, §2.2]:

1. **防护层次不对**: 现有方法(AntiFake, VSMask, AttackVC)基于 adversarial examples,仅在推理阶段干扰 speaker verification 系统,实现 timbre dissimilarity,但合成语音本身仍然高质量且可用 [§2.2]
2. **TTS 多模态输入复杂**: 不同于图像分类任务,TTS 模型的输入包括文本、音频、频谱图,扰动只能加在音频上而不能改变文本内容 [§1, Challenge 1]
3. **多目标函数交叉**: TTS 模型有多个加权优化目标(duration, timbre, style),并非所有目标都能被音频扰动影响,需要选择性优化 [§1, Challenge 3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

POP 的整体思路是: 利用 unlearnable examples 的框架,在用户公开音频前嵌入不可感知的 error-minimizing 扰动,使得攻击者用受保护数据 fine-tune TTS 模型时,模型学到的信息被扰动"短路",只能生成充满噪声的低质量语音 [§4, Fig 2]。

核心优化目标 [§4.1, Eq. 1]:

```
arg min L(G_l(x + delta), x)
s.t. H(x + delta) ≈ H(x) and ||delta||_p <= epsilon
```

其中 L 是目标函数,H 是感知函数,G_l 是 TTS 模型在位置 l 的生成器,epsilon=8/255 [§5.1]。

### 关键设计选择

**1. Speaker Selection -- 为什么要选择与预训练模型相似的说话人** [§4.2]

[论文原文] 基于预训练模型做 fine-tuning 比从头训练效率更高。为了模拟最有效的攻击场景,作者用 ECAPA-TDNN speaker encoder 计算候选说话人与预训练模型说话人的 cosine similarity,选择最相似的说话人 [Eq. 2]。每位说话人至少有 50 个样本 [§4.2]。

[agent 解读] 这个设计确保了实验在"最强攻击者"假设下验证防护效果 -- 如果连最相似的说话人都保护住了,其他说话人更不在话下。

**2. Unlearnable Audio 的简化 -- 为什么不用 bi-level 优化** [§4.3]

[论文原文] 原始的 unlearnable examples (Huang et al., 2021) 采用 bi-level 优化: 外层优化模型参数,内层生成扰动 [Eq. 3]。但在 TTS 这种大参数量、大数据集场景下,bi-level 优化的计算开销太大。作者简化为仅使用内层优化(固定模型参数,只优化扰动) [Eq. 4],因为核心机制在于"用扰动减小模型误差",模拟训练过程让模型学到更多噪声信息 [§4.3]。

[agent 解读] 这个简化成立的前提是: TTS 模型作为生成模型,学的是输入数据的分布,而不是分类的决策边界。因此固定模型参数生成的扰动已经足够有效,不需要 bi-level 来提高泛化性。SafeSpeech 继承了这一简化。

**3. Pivotal Objective Perturbation -- 为什么只优化 reconstruction loss** [§4.4]

[论文原文] 以 VITS 为例,其 generator loss 包含 5 个组件 [Eq. 6]: L_recon (mel 重建), L_kl (KL 散度), L_dur (duration), L_adv (对抗), L_fm (feature matching)。

作者逐一分析 [§4.4]:
- L_dur: 完全依赖文本,与音频扰动无关 → 无法优化(实验中生成零向量) [Fig 3, c]
- L_kl: 可被扰动影响,但学习的是 phoneme 与文本的关系 [Eq. 7] → 效果有限
- L_adv, L_fm: 可被扰动影响且在 VITS 上效果不错,但其他 TTS 模型(如 GlowTTS, Transformer-TTS)没有这些 loss → 迁移性差 [§4.4]
- L_recon: mel 距离(L1)与音频直接相关,且所有生成式 TTS 模型都会输出波形计算重建误差 → 通用性最强 [§4.4]

[论文原文] 选择 L_recon 作为 pivotal objective 的三个理由: (a) 与音频直接相关,可被扰动优化; (b) 所有生成式 TTS 模型都有此类 loss; (c) 相比优化全部 loss 需要平衡超参数,只优化一个函数更稳定,还能减少计算资源 [§4.4]。

[agent 解读] 这是 POP 最核心的洞察。它利用了生成式 TTS 的共性: 无论 VAE/Flow/GAN,最终都要最小化合成语音与真实语音的距离。在这个共享目标上做 error-minimizing 等价于让模型"以为输入已经完美了,不需要学习了"。SafeSpeech 在此基础上增加了 SPEC(引导输出趋近噪声),但 pivotal objective 的选择策略完全继承自本文。

**4. Position-fixed Perturbation -- 为什么只扰动固定位置的片段** [§4.4]

[论文原文] VITS 和 MB-iSTFT-VITS 使用 windowed generator training (WGT) [§4.4],训练时每次只取音频的一小段。如果对整段音频生成扰动,计算开销大且可感知性高。参考 PosCUDA [15],POP 在采用 WGT 的模型上只对位置 l(如 0)的固定 patch 生成扰动 [§4.4]。对不使用 WGT 的轻量模型(如 GlowTTS),则对整段音频生成扰动 [§4.4]。

[agent 解读] Position-fixed perturbation 是一个巧妙的工程优化: 利用 WGT 的训练策略,在不牺牲保护效果的前提下大幅降低计算成本和扰动的可感知性。但当攻击者使用 random segment (RS) 训练时,这种策略会被部分规避(Table 4 中 POP w/ RS 的 WER 从 109.6% 降到 63.0%)。

### 训练策略

**代理模型**: 使用 VITS 作为代理模型生成扰动,测试迁移到 MB-iSTFT-VITS 和 GlowTTS(+WaveGlow/HiFiGAN 两种 vocoder) [§5.1]。

**数据集**: LibriTTS (train-clean-100, 50 speakers) 和 CMU ARCTIC (18 speakers, 每人 300 样本),80/20 train/eval split [§5.1]。

**扰动生成**: PGD 算法,200 迭代,epsilon=8/255,batch size 15,单卡 A800 80GB [§5.1]。

## 实验

| 指标 | POP | EM baseline | Clean | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER (↑) | 127.310% | 99.681% | 21.939% | LibriTTS, MB-iSTFT-VITS | [Table 1] |
| MCD (↑) | 13.646 | 11.463 | 5.830 | LibriTTS, MB-iSTFT-VITS | [Table 1] |
| WER (↑) | 105.596% | 116.091% | 27.033% | LibriTTS, VITS | [Table 1] |
| MOS (↓) | 0.18±0.13 | 0.45±0.14 | 4.50±0.25 | LibriTTS, MB-iSTFT-VITS | [Table 2] |
| MOS (↓) | 0.13±0.09 | 0.18±0.14 | 3.68±0.45 | LibriTTS, VITS | [Table 2] |
| WER (↑, transfer) | 124.013% | 106.829% | - | LibriTTS, MB-iSTFT-VITS→VITS | [Table 3] |
| WER (↑, transfer) | 81.110% | 74.492% | - | LibriTTS, MB-iSTFT-VITS→GlowTTS-a | [Table 3] |
| SNR (↑, imperceptibility) | 17.894 dB | 9.026 dB | - | VITS | [Fig 4a] |
| PESQ (↑, imperceptibility) | 3.551 | 2.046 | - | VITS | [Fig 4b] |

**鲁棒性实验** [§5.6, Table 5]:

| 攻击手段 | WER (↑) | vs 无攻击 | 出处 |
| --- | --- | --- | --- |
| 无攻击 | 109.571% | - | [Table 4, w/o RS] |
| 频谱门控去噪 | 83.419% | WER 降 26pp | [§5.6] |
| Low-Pass Filter | 90.317% | WER 降 19pp | [Table 5, 13] |
| Hybrid Transformation | 88.009% | WER 降 21pp | [Table 5, 9] |
| Random Segment Training | 63.021% | WER 降 47pp | [Table 4, RS] |
| POP+ESP (entire) w/ RS | 89.930% | 恢复至 90% | [Table 4] |

## 局限性

1. **仅覆盖 fine-tuning 场景**: POP 仅在 fine-tuning 场景下验证,未考虑 zero-shot TTS(如 VALL-E, F5-TTS)的推理时保护 [§4]。SafeSpeech (2025) 后来补上了这一关键缺口
2. **无感知优化**: 仅用 L_p norm 约束扰动幅值(epsilon=8/255),未引入感知优化(如 STOI, STFT)来确保受保护音频的听觉质量 [§6]。SNR 17.9dB 和 PESQ 3.55 尚可,但存在改进空间
3. **Position-fixed 对 RS 训练脆弱**: 当攻击者使用 random segment 训练时,position-fixed perturbation 的 WER 从 109.6% 降至 63.0% [Table 4],保护效果显著下降。虽然 POP+ESP(entire perturbation)可恢复到 89.9%,但计算开销增加 3.8x(7.128s vs 1.855s) [Table 4]
4. **评估模型范围窄**: 仅测试 3 个 TTS 模型(MB-iSTFT-VITS, VITS, GlowTTS),均为 VITS 家族或相近架构,未测试 codec LM 或 diffusion-based TTS
5. **主观评估规模小**: MOS 评估仅 61 有效参与者 [§5.4],且未报告受保护音频本身的听觉质量 MOS(仅报告了合成语音的 MOS)
6. **无 SIM 指标**: 未报告 speaker similarity (SIM/SECS) 指标,仅用 WER 和 MCD 评估保护效果,无法直接衡量 timbre 保护程度

## 点评

**优势**:
- POP 的 pivotal objective selection 思想是本文最核心的贡献。通过分析 VITS 的 5 个 loss 函数,发现 mel reconstruction loss 是唯一同时满足"可通过扰动优化"和"跨模型通用"的目标,这个分析框架清晰且可复用 [§4.4, Fig 3]
- 迁移性实验(Table 3)出乎意料地好: 用 MB-iSTFT-VITS 生成的扰动在 VITS 上的 WER(124.0%)甚至高于 self-protection(105.6%),这从侧面验证了 pivotal objective 的跨模型通用性
- 对 13 种数据增强/降噪技术的鲁棒性测试(Table 5)非常全面,为后续工作建立了 robustness 评估基准
- 不可感知性指标(SNR 17.9dB, PESQ 3.55)显示 POP 的扰动在听觉上几乎不可察觉,远优于 EM baseline

**不足**:
- 与同作者后续的 SafeSpeech 相比,POP 缺少了三个关键改进: (a) SPEC -- 引导输出趋近噪声(从"学不到东西"升级为"学到噪声"); (b) 感知优化(STOI+STFT); (c) zero-shot 场景覆盖
- 论文声称"根本性解决 deepfake 问题"[§1, "radically mitigate the threat"],但实际上仅在 fine-tuning 场景下有效,且对 RS 训练有明显弱点(WER 降至 63%),这一 claim 过于强势
- 虽然实验在 MB-iSTFT-VITS→VITS 迁移时效果好,但这两个模型架构非常接近(都是 VAE+GAN 端到端),不能充分证明对架构差异大的模型(如 diffusion TTS, codec LM)的迁移性

**在 KB 语境下的定位**:
从 [[Anti-spoofingandDeepfakeDetection]] 的视角看,POP 是 proactive voice protection 方向的早期工作,提出了 pivotal objective selection 的核心思想,但覆盖面和鲁棒性不如后续的 [[论文笔记/SafeSpeech|SafeSpeech]]。POP 的主要历史价值在于: (a) 首次将 unlearnable examples 从图像分类迁移到 TTS 生成任务,并解决了多模态输入和多目标 loss 的技术挑战; (b) 为 SafeSpeech 的 SPEC 和感知优化提供了技术基础。对比 KB 中已有的安全方向工作,POP/SafeSpeech 走"数据端防护"路线,与 SpeakerIdentityUnlearning (模型端遗忘)和 TraceableTTS (事后溯源)互补。

## 可复用的 idea

1. **Pivotal objective selection**: 面对多目标 TTS loss,逐一分析每个 loss 的三个属性(是否可被扰动影响、是否跨模型通用、收敛速度),选出最优单一目标。这个分析框架可迁移到任何需要跨模型 transferable perturbation 的场景 [§4.4, Fig 3]
2. **Position-fixed perturbation**: 利用 TTS 模型 WGT 训练策略的特性,只在固定位置生成扰动,大幅降低计算成本(1.855s vs 7.128s)和提高不可感知性(SNR 17.9dB vs 11.0dB) [§4.4, Table 4]
3. **Error-minimizing for generative models**: 将 unlearnable examples 从分类任务迁移到生成任务时,不需要 bi-level 优化,固定模型参数只优化扰动即可,因为生成模型学的是输入分布而不是决策边界 [§4.3]
