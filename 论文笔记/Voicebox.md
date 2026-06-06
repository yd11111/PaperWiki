---
type: paper
tier: deep
title: "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale"
arxiv_id: "2306.15687"
source: "Sources/Voicebox.pdf"
authors: [Matthew Le, Apoorv Vyas, Bowen Shi, Brian Karrer, Wei-Ning Hsu, Vimal Manohar, Leda Sari, Mary Williamson, Yossi Adi, Jay Mahadeokar, Rashel Moritz]
year: 2023
venue: "arXiv"
tags: [TTS, zero-shot, flow-matching, non-autoregressive, speech-infilling, cross-lingual, multilingual, in-context-learning]
concepts: ["[[ConditionalFlowMatching]]", "[[Classifier-FreeGuidance]]", "[[Non-autoregressiveTTS]]", "[[DurationPredictor]]", "[[MelSpectrogram]]"]
models: ["[[NaturalSpeech2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[Classifier-FreeGuidance]](pending-review), [[Non-autoregressiveTTS]](pending-review), [[NaturalSpeech2]](pending-review), [[DurationPredictor]](pending-review) | 未命中但可能相关: 无

**谱系定位**: Voicebox 是 flow matching 在大规模语音生成中的开创性应用。在 [[ConditionalFlowMatching]] 概念页的演进线中,Voicebox (2023) 被明确标记为 "Flow Matching (Voicebox, 2023)" 这一里程碑节点,位于 Grad-TTS (2021, diffusion-based) 之后、CosyVoice 系列 (2024-2025, CFM+DiT) 之前。它首次证明了 OT-CFM 在 speech generation 领域的可行性和优越性。

**已有认知**: 概念库已记录 CFM 的技术发展已远超 Voicebox 时代 -- 从 rectified flow (VoiceFlow, 2024) 到 single-step distillation (DSFlow, OZSpeech) 到 discrete flow matching (DiFlow-TTS) 到 learned prior (Flamed-TTS)。Voicebox 作为这条技术线的源头,其核心贡献 (speech infilling formulation + OT path + masked CFM loss) 在后续工作中被反复继承和改进。

**创新判断**: 从当前知识库视角看,Voicebox 的独特贡献不在于 flow matching 本身 (Lipman et al., 2023 已提出),而在于: (1) 将 speech 的多种 task 统一为 text-guided infilling problem, (2) 证明 NAR flow matching 在大规模数据 (60K hrs) 下可超越 AR 方案 (VALL-E), (3) 将 CFG 从 diffusion 扩展到 flow matching。后续系统 (CosyVoice, F5-TTS, Seed-TTS_DiT) 均沿用了这些设计选择。

**零样本 TTS 对比**: [[Zero-shotSpeechSynthesis]] 任务页记录的当前 SOTA (CosyVoice 3, Seed-TTS, IndexTTS2) 在 SEED-TTS-Eval 上 WER < 1.5%, SS > 0.8。Voicebox 评测基于 Librispeech test-clean,指标体系不同 (WER 1.9%, SIM-o 0.662),但在 2023 年 VALL-E 基准下是巨大进步。

## 速查

> [!summary] 速查
> - **一句话**: 首个大规模非自回归 flow matching 语音生成模型,通过 text-guided speech infilling 统一 zero-shot TTS / 去噪 / 编辑 / 多样化采样等多种任务
> - **路线**: (文本, 音频上下文) → phonemize + forced alignment → duration model (预测帧长) → audio model (OT-CFM Transformer, 80-dim log mel) → HiFi-GAN vocoder → waveform
> - **指标**: Zero-shot TTS WER 1.9% vs VALL-E 5.9%, SIM-o 0.662 vs 0.580 [Table 2]; 推理速度 20x vs VALL-E (NFE=2); 跨语言 TTS 平均 WER 5.2% vs YourTTS 10.9% [Table 3]
> - **可借鉴**: speech infilling 作为 universal task formulation; masked CFM loss 聚焦预测区域; duration model 解耦文本-音频建模; CFG 扩展到 flow matching
> - **局限**: 依赖 phonemizer + forced aligner (MFA); 仅训练于有声读物 (read speech); 不支持属性解耦控制; 未开源模型

## 核心问题

Voicebox 要解决的核心问题是: **如何构建一个通用的语音生成模型,使其能通过 in-context learning 泛化到多种未显式训练的任务,同时达到或超越各任务专用模型的性能?**

具体动机:
1. 当时的 speech generative models 仍停留在 task-specific 阶段,每种任务 (TTS, 去噪, 编辑) 需要独立模型和独立数据集 [§1]
2. 大规模 in-the-wild 数据上的训练效果远不如 curated 小数据集,说明模型 capacity 和训练方法存在严重瓶颈 [§1]
3. AR 方案 (如 VALL-E) 只能条件于过去上下文,无法利用未来上下文,且推理速度慢 [§2]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Voicebox 由两个解耦组件构成:

1. **Duration Model**: 预测 phoneme-level 时长 l, 输入为 phone 序列 y + 上下文时长 lctx [§3.3]
2. **Audio Model**: 给定 frame-level phone 转录 z 和音频上下文 xctx, 用 OT-CFM 生成缺失语音 xmis [§3.3]

**统一 formulation**: 所有任务被归结为 text-guided speech infilling -- 给定 mask m, 将原始音频分为 xmis (masked, 要生成) 和 xctx (context, 已知), 模型学习 p(xmis | y, xctx) [§3.2]。[agent 解读] 这一 formulation 的巧妙之处在于,通过改变 mask 的位置和大小,可以自然地实现不同任务: mask 在末尾 = TTS, mask 在中间 = 编辑/去噪, 全部 mask = 无条件采样。

### 关键设计选择

**1. OT-CFM 训练目标 (而非 score matching 或 diffusion path)**

论文采用 Lipman et al. (2023) 的 Conditional Flow Matching with Optimal Transport path。关键方程:

- 条件路径: pt(x | x1) = N(x | tx1, (1-(1-sigma_min)t)^2 I) [§3.1]
- 条件向量场: ut(x | x1) = (x1 - (1-sigma_min)x) / (1-(1-sigma_min)t) [§3.1]
- ODE 轨迹为近似直线 (constant speed and direction) [§3.1]

[论文原文] "The flow is arguably simple because points move with a constant speed and direction. We adopt it for Voicebox." [§3.1]

**为什么选 OT path 而非 diffusion path?** 消融实验 (Table 8, 9) 给出了定量答案:
- **训练效率**: FM w/ OT 在 50K updates 就达到 WER 2.5 / SIM-o 0.424; FM w/ diff 在 100K updates 才达到 WER 3.1 / SIM-o 0.344; SM w/ diff 在 150K 才达到 WER 5.1 / SIM-o 0.349 [Table 8]
- **推理效率**: FM w/ OT 在 NFE=4 就达到 WER 2.4 / SIM-o 0.410; FM w/ diff 需要 NFE=16 (WER 2.7), SM w/ diff 需要 NFE>64 [Table 9]

[agent 解读] 这一消融结论对后续工作影响深远 -- 之后几乎所有 flow-based TTS (CosyVoice, F5-TTS, Matcha-TTS) 都直接采用 OT path 而非 diffusion path。

**2. Masked CFM Loss**

标准 CFM loss 在所有帧上计算 (Eq. 5)。Voicebox 提出只在 masked 帧上计算 loss 的变体 (Eq. 6):

L_audio-CFM-m = E[||m * (ut - vt)||^2]

[论文原文] masked loss "leads to better results" [§3.3, Appendix B.1]。具体消融: masked loss 在 zero-shot TTS 上 SIM-r 0.597 vs all-frame loss 0.528, FSD 242.5 vs 243.1 [Table B2]。

[agent 解读] 直觉上, masked loss 让模型集中学习"根据上下文预测缺失部分"的能力,而非在已知区域浪费容量。这与 BERT 的 masked language modeling 类比: 只在 [MASK] 位置计算 loss 效果更好。

**3. 音频-时长解耦建模**

[论文原文] "Motivated by the need that some applications require fine-grained alignment control between speech and text, we decouple Voicebox into two components" [§3.3]

Duration model 有两种变体:
- **Flow matching duration**: 用 CFM 建模 q(l | y, lctx), 可生成多样化时长 [§3.3]
- **Regression duration**: L1 回归 masked 时长, 类似 FastSpeech 2 [§3.3, Eq. 7]

[agent 解读] 回归 duration 默认用于 TTS (更稳定), flow matching duration 用于多样化采样 (更多样)。这一解耦让 Voicebox 比 NaturalSpeech 2 更灵活 -- NS2 的 pitch + duration predictor 都是确定性回归, 无法生成多样化样本。

**4. Classifier-Free Guidance 扩展到 Flow Matching**

训练时以 puncond = 0.2 的概率丢弃条件 (z, xctx) [§3.5]。推理时的 guided vector field:

v_tilde = (1 + alpha) * vt(w, xctx, z) - alpha * vt(w) [Eq. 8]

alpha (音频 CFG strength) 和 alpha_dur (时长 CFG strength) 分开调控 [§3.5]。

**5. Transformer 架构细节**

Audio model: 24 层, 16 heads, 1024/4096 embed/FFN, 330M 参数 [§5.1]。关键特征:
- Convolutional positional embedding (from wav2vec 2.0) [§5.1]
- Symmetric bi-directional ALiBi self-attention bias [§5.1, Appendix A.6]
- UNet-style skip connections (对称层连接) [§5.1]
- Flow step embedding 的 ALiBi bias 设为 0 [Appendix A.6]

输入拼接方式: (xt, xctx, zemb) 逐帧拼接后投影到 D 维 [§3.3]

**6. Ghost Silence 和 Word-Position-Dependent Phone**

[agent 解读] 这两个工程技巧被后续工作较少提及,但实际上很关键:
- Ghost silence: 在词间插入 duration=0 的 silence token, 让 duration model 学习"是否需要停顿" [Appendix A.2]
- Word-position phone: 添加 _B/_E/_I/_S 后缀标记词内位置, 帮助 audio model 识别词边界 [Appendix A.2]

### 训练策略

- 数据: VB-En 60K hrs 英语有声读物, VB-Multi 50K hrs 6 语言有声读物 [§5.1]
- 训练: VB-En 500K updates, VB-Multi 750K updates, effective batch size 240K frames [§5.1]
- 音频上限: 1600 frames (16 秒), 超长则随机截断 [§5.1]
- Masking: 音频连续 mask r% (r ~ U[70,100]), 时长 mask r% (r ~ U[10,100]) [§5.1]
- 低资源语言上采样: beta=0.25 指数上采样 [§5.1]
- 特征: 80-dim log mel spectrogram @ 100Hz, 全局 mean/std 归一化 [§5.1, Appendix A.3]
- Forced alignment: Montreal Forced Aligner (MFA), modified IPA phone set [§5.1]

## 实验

| 指标 | 本文 (VB-En) | VALL-E | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ZS-TTS WER (cross-sent) | 1.9% | 5.9% | LS test-clean | [Table 2] |
| ZS-TTS SIM-o (cross-sent) | 0.662 | - | LS test-clean | [Table 2] |
| ZS-TTS SIM-r (cross-sent) | 0.681 | 0.580 | LS test-clean | [Table 2] |
| ZS-TTS WER (continuation) | 2.0% | 3.8% | LS test-clean | [Table 2] |
| ZS-TTS SIM-o (continuation) | 0.593 | 0.452* | LS test-clean | [Table 2] |
| QMOS | 3.78 | - | LS test-clean | [Table 2] |
| SMOS | 3.71 | - | LS test-clean | [Table 2] |
| Cross-lingual avg WER (6 lang) | 5.2% | - | MLS test | [Table 3] |
| Cross-lingual avg SIM-o | 0.481 | - | MLS test | [Table 3] |
| vs YourTTS avg WER (En/Fr/Pt) | 5.2% | 10.9% (YT) | MLS test | [Table 3] |
| Denoising WER (-10dB, 50%) | 2.0% | - | LS test-clean + noise | [Table 5] |
| Denoising SIM-o | 0.612 | - | LS test-clean + noise | [Table 5] |
| Diverse sampling FSD | 155.7 | - | LS test-other | [Table 6] |
| ASR data gen: WER test-clean | 2.6% | - | LS train-960 synth | [Table 7] |
| ASR data gen: WER test-other | 6.7% | - | LS train-960 synth | [Table 7] |

关键发现:

1. **NFE 与质量 trade-off**: NFE=2 时推理速度是 VALL-E 的 20x, WER 仍优于 VALL-E [Fig 5a, §5.6]。NFE>=8 即可获得接近最优的 SIM-r [Fig 5c]。

2. **上下文长度效应**: SIM-r 随 prompt 长度增加快速饱和, Voicebox 约需 VALL-E 2/3 的 prompt 长度即可达到同等 SIM [Fig 6b] [§5.7]。

3. **跨语言 prompt 长度**: 更长 prompt 提高 SIM 但 WER 恶化, 尤其 En→非 En 方向, 因训练数据中英语占 90%+ [§5.7, Fig 8]。

4. **数据规模 scaling**: 从 60h 到 60Kh, zero-shot SIM-r 从 0.151 升至 0.645, WER 从 2.30 降至 2.05 [Table B3]。

5. **合成数据训练 ASR**: Voicebox (FM duration) 生成的合成数据训练 ASR, WER 仅比真实数据高 0.4%/1.7% (test-clean/other), 远优于 VITS-LJ (+49.4%), YourTTS (+18.2%) [Table 7]。

## 局限性

1. **依赖外部工具链**: 需要 phonemizer + forced aligner (MFA), 限制语言覆盖和端到端优化 [§7]。[agent 解读] 这一依赖在后续工作中逐步被消除: E2 TTS 完全去掉 phonemizer, CosyVoice 用 LLM 隐式处理对齐。

2. **仅训练于 read speech**: 有声读物数据缺乏对话、情感、非言语声音 (笑声、嗯嗯等) 的覆盖 [§7]。

3. **不支持属性解耦控制**: 无法独立控制音色、情感、韵律等属性 — 只能整体模仿参考音频的风格 [§7]。[agent 解读] NaturalSpeech 3 (2024) 通过 factorized codec 解决了这一问题。

4. **基于 word-based phonemizer**: 不考虑上下文依赖的发音 (如法语连诵 liaisons), 影响跨语言性能 [§7]。

5. **未开源**: 仅提供 demo 页面, 无代码或模型权重。后续 Meta 的 Audiobox 延续了类似设计但仍未完全开源。

6. **评测局限**: 使用 Librispeech test-clean 评测 zero-shot TTS, 数据偏 clean 且说话人有限。后续 SEED-TTS-Eval 等 benchmark 提供了更严格的评测。

## 点评

**历史地位**: Voicebox 是 TTS 领域的分水岭工作。它同时在三个维度上突破了既有范式:
1. **方法论**: 首次在大规模语音生成中验证 flow matching 优于 diffusion 和 score matching [Table 8, 9]
2. **任务统一**: 将 TTS / 去噪 / 编辑 / 采样统一为 text-guided infilling, 证明 speech 的 in-context learning 可行
3. **规模化**: 在 60K hrs in-the-wild 数据上训练, 证明 NAR 方案也能处理大规模非 curated 数据

**对后续工作的影响**:
- CosyVoice / F5-TTS / Matcha-TTS 直接采用 OT-CFM 作为声学模型
- VALL-E 之后的 AR 方案 (如 Seed-TTS) 也开始探索 diffusion/flow 变体
- Speech infilling formulation 被 SoundStorm, AudioLDM 2 等继承
- FSD (Frechet Speech Distance) 指标被后续工作广泛采用

**与 NaturalSpeech 2 的对比** (同期工作):
- NS2 用 latent diffusion + learned codec latent; Voicebox 用 OT-CFM + mel spectrogram
- NS2 需要 pitch predictor; Voicebox 不需要
- NS2 always conditions on prompt (无法无条件采样); Voicebox 可以
- NS2 用 150 diffusion steps; Voicebox 仅需 16 ODE steps (甚至 8 步可用)
- NS2 不支持 infilling; Voicebox 的 infilling formulation 是核心

**今天看来的不足**: 从 2026 年视角回看, Voicebox 的 phonemizer + forced aligner 依赖已被证明非必要 (E2 TTS, F5-TTS 无需 alignment); 330M 的模型规模在今天偏小 (CosyVoice 3 为 1.5B); 仅支持 read speech 的泛化能力有限。但其核心 insight (speech infilling + flow matching + masked loss) 至今仍是主流方案的基础。

## 可复用的 idea

1. **Speech infilling 作为 universal task**: 通过改变 mask 位置/大小统一多种任务, 避免为每个任务训练独立模型。可迁移到音乐编辑、音频修复等场景。

2. **Masked CFM loss**: 只在需要预测的区域计算 loss, 提升模型对目标区域的专注度。可用于任何条件生成任务。

3. **Duration model 解耦**: 分别建模时长和声学特征, 使不同应用可以选择确定性 (回归) 或多样性 (flow matching) 的时长生成。

4. **Ghost silence 机制**: 在词间插入 duration=0 的显式 silence token, 让模型自主决定是否插入停顿, 比硬编码规则更灵活。

5. **FSD (Frechet Speech Distance)**: 用 wav2vec 2.0 特征计算 Frechet distance 评估语音多样性和质量, 论文详细验证了 FSD 相比 FAD 对 speech diversity 更敏感 [Appendix C.1]。

6. **合成数据验证方式**: 用 TTS 合成数据训练 ASR 模型并在真实数据上评测, 作为衡量合成语音真实性的间接指标 [§5.5]。
