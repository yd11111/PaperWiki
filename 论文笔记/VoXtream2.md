---
type: paper
tier: deep
title: "VoXtream2: Full-stream TTS with dynamic speaking rate control"
arxiv_id: "2603.13518"
source: "Sources/VoXtream2.pdf"
authors: [Nikita Torgashov, Gustav Eje Henter, Gabriel Skantze]
year: 2026
venue: "Under review at INTERSPEECH 2026"
tags: [TTS, streaming, zero-shot, autoregressive, speaking-rate-control, full-stream, CFG, distribution-matching]
concepts: ["[[Classifier-FreeGuidance]]", "[[SpeakerEmbedding]]", "[[ProsodyModeling]]", "[[DurationPredictor]]", "[[LLM-basedTTS]]", "[[SemanticvsAcousticTokens]]", "[[SpeechTokenizer]]", "[[PhonemeRepresentation]]", "[[Speech-TextAlignment]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[LLM-basedTTS]], [[SpeakerEmbedding]], [[ProsodyModeling]], [[模型库/CosyVoice2|CosyVoice 2]] + 2 个待确认: [[Classifier-FreeGuidance]], [[DurationPredictor]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: VoXtream2 是 [[论文笔记/VoXtream|VoXtream]] 的直接后继,属于 [[LLM-basedTTS]] 中的 full-stream AR 路线。与主流 LLM-TTS (VALL-E/CosyVoice) 不同,VoXtream 系列不使用通用 text LLM 做骨干,而是设计三层专用 transformer (PT/TT/DT)。VoXtream2 的核心新增是动态语速控制 (Dynamic SRC),这在 [[ProsodyModeling]] 中属于 duration 维度的精细控制 -- 目前概念页记录了多种 duration predictor 技术路线 (DDP/SDP/MoE-DP/GRPO/DPO),但分布匹配 (distribution matching) 用于帧级语速控制是全新的方案。
>
> **已有认知**: [[Classifier-FreeGuidance]][待确认] 页面记录了 CFG 从连续 diffusion/flow 扩展到离散 token 空间的趋势 (OmniVoice 的 log-softmax CFG);VoXtream2 进一步将 CFG 应用到多条件 AR TTS (text/audio/speaker) 并发现 CFG 的 γ 参数间接影响语速 (不直接应用于 duration 时也通过 intelligibility 影响 SRC 效果)。[[DurationPredictor]][待确认] 页面追踪了从 FastSpeech DDP 到 MoE-DP/GRPO/DPO 的演进,VoXtream2 的 distribution matching 是另一个正交方向 -- 不修改 duration predictor 本身,而是在采样阶段通过直方图匹配校正 duration 分布。[[SpeakerEmbedding]] 页面记录了 ECAPA-TDNN 等 encoder,VoXtream2 使用 ReDimNet (100k+ identities),是该页未覆盖的新型 speaker encoder。[[模型库/CosyVoice2|CosyVoice 2]] 是 VoXtream2 的关键 baseline,CosyVoice2 的 instructed generation 仅支持有限范围语速控制 (3-4 SPS),而 VoXtream2 实现了 1-7 SPS 的连续动态控制。
>
> **创新判断**: VoXtream2 的核心创新是 distribution matching over duration states 实现帧级动态 SRC -- KB 中没有对应概念。这是一种轻量级、采样时的控制方案,不需要额外训练或模型修改,与现有 duration predictor 优化路线 (GRPO/DPO/MoE) 完全正交。
>
> 检索命中: [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓, [[ProsodyModeling]]✓, [[模型库/CosyVoice2|CosyVoice 2]]✓ | 过滤: [[Classifier-FreeGuidance]](pending-review), [[DurationPredictor]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 在 VoXtream 基础上通过 distribution matching 实现帧级动态语速控制 (1-7 SPS,支持 mid-utterance 变速),配合多条件 CFG 和 prompt text masking,以 462M 参数 + 40k 小时数据达到公开 baseline 水平,FPL 74ms
> - **路线**: Text (word-by-word) → G2P (IPA) → Phoneme Transformer (25 look-ahead) → Temporal Transformer (semantic token + duration token 联合预测,CFG γ=1.5) + distribution matching SRC → Depth Transformer (16 codebooks,冻结 CSM 权重,CFG γ=3.0) + ReDimNet speaker embedding → Mimi decoder → waveform
> - **指标**: SEED-en WER 1.32% / SPK-SIM 0.656 / UTMOS 4.05; LS-PC WER 2.12 / SPK-SIM 0.638 / UTMOS 4.24; Naturalness MUSHRA 68.8±2.3 (最高); FPL 74ms, RTF 0.256 (RTX3090); Dynamic SRC Pearson corr 0.62-0.83 [Table 2, 4, 5]
> - **可借鉴**: (1) Distribution matching 做采样时语速控制: 不改模型,只在 duration token 采样时用目标分布重加权,轻量且可即时切换; (2) CFG 不应用于 duration 但仍间接影响语速,γ 可作为 SRC 辅助调节杠杆; (3) Prompt text masking 消除对外部 aligner 的推理依赖; (4) Acoustic prompt enhancement 对冲 CFG 带来的音质退化
> - **局限**: 语速仍受 prompt speaking rate 影响 (未完全解耦); 慢速语速 (1 SPS) WER 大幅上升 (16.5%); 训练依赖外部 forced aligner (~35% 数据因对齐失败丢弃); 仅英语

## 核心问题

VoXtream2 要解决的核心问题是: **如何在 full-stream TTS 中实现动态语速控制,使合成语音的语速可以在生成过程中实时调整?** 这个问题有两个层面:

1. **静态语速控制**: 现有系统要么不支持语速控制 (VoXtream, Kyutai-TTS),要么只支持全句级控制 (CosyVoice2 instructed generation, Spark-TTS),要么需要精确指定 duration (MaskGCT, VoiceStar) 但代价是质量下降 [§1, Table 1]
2. **动态语速控制**: 人类说话时语速是动态变化的 -- 思考时变慢插入 "uhm",表达熟悉内容时加速 [§1]。仅有 WeSCon 探索了 word-level 动态 SRC,但依赖复杂的多轮推理且降采样超过 40% 就不可用 [§2.1]

VoXtream2 同时解决了 VoXtream 的几个遗留问题:
- 需要 prompt 文本转录和外部 phoneme aligner [§2.2]
- 无 CFG 导致 speaker similarity 较低 [§1]
- 无语速控制机制 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VoXtream2 沿用 VoXtream 的三层 transformer 架构,做了以下关键修改 [§3.1, Fig 1]:

1. **Phoneme Transformer (PT)**: 6 层 8 头。改用 IPA 音标字典 (支持多语言) [§3.1],最大 look-ahead 从 10 增至 25 phonemes,最小 look-ahead 仍为 3 phonemes [§3.1]。增加标点符号处理: 每个标点作为独立 phoneme token 输入 PT,输出前移除,使 TT 只建模 phoneme duration 但获得标点上下文 [§3.1]。

2. **Temporal Transformer (TT)**: 12 层 16 头,embedding 1024,FFN 4096。输出维度为 N×D (N = Mimi vocab size, D = 6 duration tokens),semantic token 和 duration token 联合建模、联合采样 [§3.1]。Duration token 编码 3 种 shift state (前进 0/1/2 个 phoneme) × 2 种 phoneme count (1 或 2),共 6 个状态 (VoXtream 仅 4 个) [§3.1]。CFG γ_temp = 1.5 [§3.3]。

3. **Depth Transformer (DT)**: 4 层 8 头,FFN 8192 (参照 Sesame-CSM [55])。生成 15 个 acoustic tokens per step (VoXtream 为 11 个, 即 16 codebooks - 1 semantic) [§3.1]。CFG γ_depth = 3.0,speaker embedding conditioning 权重增加 50% [§3.3]。DT 权重冻结,预训练于大规模会话语音数据集 [§3.1]。

4. **Speaker Encoder**: ReDimNet [56],训练于 100k+ speaker identities [§4.2]。

5. **Audio Codec**: Mimi [45],12.5 Hz,24 kHz,16 codebooks [§3.1]。

### 关键设计选择

**1. 为什么用 distribution matching 做语速控制,而非修改 duration predictor?**

VoXtream2 的 SRC 不改变模型本身,只在采样阶段对 duration token 分布做重加权 [论文原文]。核心机制 [§3.5, Fig 2]:

(a) **目标分布 Ptarget**: 给定目标 syllables-per-second (SPS) 值,查找训练数据中该 SPS 对应的 duration state 直方图 (6-bin) [§3.5]

(b) **当前分布 Pcurrent**: 从 TT 输出的 N×D 矩阵中,marginalize over semantic tokens 得到 D 维 duration 分布 [§3.5, Eq. 1]

(c) **累积分布 Pacc**: 基于过去 3 秒生成的 duration counter 在线估计 [§3.5]。初始化为均匀分布。3 秒窗口的设计: "long enough for a robust estimate and short enough to react to changes" [论文原文]

(d) **校正权重**: W = exp(β * (log10(Ptarget) - log10(Pacc))) [§3.5, Eq. 2]

(e) **更新采样分布**: Pupdated = (Pcurrent ⊙ W) / Σ(Pcurrent_i * W_i) [§3.5, Eq. 3]

参数 β 控制 SRC 强度: β=1 控制弱但质量好,β=10 控制强但 WER 增加,β=5 为 trade-off [§3.5]。

[agent 解读: 这个设计的精妙之处在于它是 self-correcting 的 -- Pacc 是滑动窗口的在线估计,如果实际语速偏离目标,Pacc 与 Ptarget 的差异会增大,校正力度自动增强。这比一次性设定 duration 更鲁棒,因为它在生成过程中持续调整。同时 distribution matching 是在 log 域操作的,避免了极端权重。]

为什么不直接对 duration token 施加 CFG? 因为 Parakeet [37] 和 SSR-Speech [38] 发现 CFG 会加速语速,不适合做精细控制。VoXtream2 不对 duration state 应用 CFG,而是用 distribution matching 做正交控制 [§3.5]。

**2. 为什么 CFG 同时应用于 text/audio/speaker 三个条件?**

VoXtream2 将 CFG 从 Koel-TTS [39] 的做法推广到所有条件 [§3.3]:

- Text conditioning (PT 输入): 10% mask → 生成可以不依赖文本
- Audio conditioning (TT 输入): 10% mask → 生成可以不依赖 prompt audio
- Speaker embedding (DT 输入): 10% drop → 生成可以不依赖 speaker identity

γ_temp = 1.5 (较小,允许韵律变化), γ_depth = 3.0 (较大,精确控制音色) [§3.3]。Speaker embedding conditioning weight 增加 50% 以进一步增强 voice cloning [§3.3]。

Ablation (Table 6) 显示逐步添加 CFG 的效果:
- + CFG text: WER 从 2.29 降至 1.37 (显著改善 intelligibility)
- + CFG audio: SPK-SIM 从 0.578 升至 0.661 (WER 略增至 1.62,UTMOS 降至 3.65)
- + CFG speaker: SPK-SIM 继续升至 0.674
- + Speaker weight: SPK-SIM 最高 0.679
- + Prompt enhancement: WER 最低 1.32,UTMOS 最高 4.05,SPK-SIM 微降至 0.656

[agent 解读: Audio CFG 引入了 similarity-quality trade-off: 越强调 speaker similarity,生成越趋近 prompt 的声学条件 (包括噪声和瑕疵),因此 UTMOS 下降。Prompt enhancement 是对冲这个退化的关键设计 -- 增强 prompt 质量后,即使 CFG 把生成"拉向" prompt,音质也有保障。]

**3. 为什么需要 prompt text masking?**

VoXtream (v1) 的一个实用性瓶颈: 需要 prompt 音频的文本转录 + 外部 phoneme aligner [§3.2]。问题:
- 高语速 prompt 难以转录和对齐 [§2.2]
- Aligner 错误会退化性能 [§3.2]

VoXtream2 的做法 [§3.2]:
- 训练时: 随机选 3-10s prompt 音频,将对应文本替换为 \<UNK\> token 序列
- 推理时: 每个 prompt 音频帧分配一个 \<UNK\> token
- 不同于 Cross-Lingual F5-TTS [46] 的 text prefix dropping,VoXtream2 保留 masked tokens 参与梯度计算,这使 CFG 可以应用于 prompt [§3.2]

Table 3 显示 prompt masking 的效果:
- WER 在不同 prompt 语速下更稳定 (慢 prompt: 2.14 vs baseline 2.80,快 prompt: 2.13 vs 4.21)
- 标准差显著降低 (慢: 3.27 vs 5.64,快: 3.73 vs 8.10)
- 但生成语速仍受 prompt 影响 [§5, Table 3]

**4. Acoustic Prompt Enhancement 的作用**

CFG audio 增强 speaker similarity 的副作用: 生成质量趋近 prompt 质量 [§3.4]。如果 prompt 有背景噪声或录音瑕疵,会传播到生成音频 [论文原文]。解决方案: 用 Sidon [48] 增强 prompt 音频。仅处理 prompt (不处理生成),不增加推理延迟 [§3.4]。

**5. Filler word 生成机制**

为增强慢速语音的自然度,训练时将 filler words (uh, uhm, yeah) 从转录中移除,其 phoneme 时长合并到相邻 phoneme [§3.5, §4.1]。模型必须从韵律模式 (主要是 elongated durations) 推断 filler 插入位置 [论文原文]。结果: 慢速时自动插入 fillers,快速时几乎不插入,且 prompt 含 fillers 会进一步增加出现频率 [§5.2, Fig 5]。

### 训练策略

- **数据**: Emilia 30k h (English subset from 47k, filtered) + HiFiTTS-2 10k h (22kHz, filtered WER<10% + >5s) = 40k h total [§4.1]
- Filler words 移除 + phoneme timestamps merged with preceding phoneme [§4.1]
- Phoneme 对齐: Clap-IPA forced aligner [53],~35% 数据因对齐或转录无效被丢弃 [§7]
- 样本拼接: speaker-level 拼接到 55s 目标长度,短 clip 用 silence padding [§4.1]
- **Model**: Llama-3.2 [54] transformer 作为 backbone [§4.2]
- **训练**: 2×NVIDIA H200, batch 64/GPU, 10 epochs, 28 hours [§4.2]
- 随机裁剪 50s audio segments [§4.2]
- AdamW, 1 epoch warmup from 1e-5 to 2e-4 [§4.2]
- torch.compile 加速训练; CUDA Graphs 加速推理 [§4.2]
- **Loss**: TT + DT 均为 cross-entropy loss [§3.1]

## 实验

### Zero-shot TTS (Table 2)

| 指标 | VoXtream2 | VoXtream2-B | CosyVoice2 | VoiceStar | MaskGCT | Spark-TTS | Kyutai-TTS | F5-TTS | Human | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER (%) | 1.32 | 1.45 | 2.27 | 2.20 | 2.03 | 1.85 | 1.34 | 1.26 | 1.50 | SEED-en | [Table 2] |
| SPK-SIM | 0.656 | 0.679 | 0.658 | 0.605 | 0.712 | 0.572 | 0.689 | 0.670 | 0.734 | SEED-en | [Table 2] |
| UTMOS | 4.05 | 3.56 | 4.16 | 3.92 | 3.56 | 3.94 | 3.63 | 3.69 | 3.53 | SEED-en | [Table 2] |
| WER (%) | 2.12 | 2.14 | 1.93 | 2.14 | 2.58 | 2.11 | 1.63 | 2.05 | 1.95 | LS-PC | [Table 2] |
| SPK-SIM | 0.638 | 0.658 | 0.655 | 0.603 | 0.691 | 0.578 | 0.660 | 0.653 | 0.695 | LS-PC | [Table 2] |
| UTMOS | 4.24 | 4.02 | 4.38 | 4.25 | 3.92 | 4.35 | 4.04 | 3.88 | 4.10 | LS-PC | [Table 2] |
| Naturalness (MUSHRA) | 68.8±2.3 | 60.2±2.5 | 66.2±2.5 | 60.0±2.6 | 57.6±2.6 | 63.2±2.5 | 60.5±2.4 | 65.6±2.4 | 56.7±2.6 | — | [Table 2] |

### Full-stream 性能 (Table 4, RTX3090)

| 模型 | FPL (ms) | RTF | 出处 |
| --- | --- | --- | --- |
| CosyVoice2:TRT | 837 | 0.546 | [Table 4] |
| Kyutai-TTS:CG | 277 | 0.272 | [Table 4] |
| VoXtream:TC | 78 | 0.223 | [Table 4] |
| VoXtream2 (CUDA Graphs) | 74 | 0.256 | [Table 4] |
| VoXtream2 (torch.compile) | 63 | 0.173 | [Table 4] |

### 动态语速控制 (Table 5)

| 设置 | Corr. | WER (%) | SPK-SIM | UTMOS | 出处 |
| --- | --- | --- | --- | --- | --- |
| No SRC | - | 3.26 | 0.624 | 3.86 | [Table 5] |
| Static: Slow (1 SPS) | - | 16.51 | 0.620 | 3.53 | [Table 5] |
| Static: Normal (4 SPS) | - | 3.20 | 0.626 | 3.83 | [Table 5] |
| Static: Fast (7 SPS) | - | 3.52 | 0.609 | 3.81 | [Table 5] |
| Dynamic: S→N→F | 0.701 | 4.73 | 0.628 | 3.75 | [Table 5] |
| Dynamic: F→N→S | 0.826 | 11.25 | 0.626 | 3.71 | [Table 5] |
| Dynamic: S↔F | 0.619 | 9.55 | 0.629 | 3.66 | [Table 5] |
| Dynamic: F↔S | 0.660 | 11.20 | 0.627 | 3.64 | [Table 5] |

### 关键发现

1. **VoXtream2 在主观评估中排名第一**: MUSHRA 68.8±2.3,超过 CosyVoice2 (66.2), F5-TTS (65.6),甚至超过 Human (56.7) [Table 2]。[agent 解读: Human 得分低是因为录音含真实背景噪声,符合 MOS 评估设计敏感性的已知问题 [65]]

2. **Prompt enhancement 是主观评估的关键**: VoXtream2-B (无增强) MUSHRA 60.2 → VoXtream2 (有增强) 68.8,提升 8.6 分 [Table 2]。UTMOS 3.56 → 4.05 [Table 2]

3. **FPL 达到 74ms**: 比 CosyVoice2:TRT (837ms) 快 11 倍,比 Kyutai-TTS (277ms) 快 3.7 倍 [Table 4]

4. **动态 SRC 可行但有代价**: 渐变场景 (S→N→F) Corr=0.70-0.83,突变场景 (S↔F) Corr=0.62-0.66。快→慢的 controllability (0.826) 好于慢→快 (0.701),但快→慢的 WER (11.25%) 高于慢→快 (4.73%) [Table 5]

5. **慢速 SRC 是弱点**: 1 SPS 静态 WER 高达 16.51%,原因是 (a) 训练数据中超慢 duration state 不足导致 hallucination; (b) filler/重复词插入增多推高 WER [§5.3]

6. **CFG γ_temp 与 SRC 存在交互**: 虽然 CFG 不直接应用于 duration,但 γ_temp 越大快速语音 WER 越低而慢速 WER 越高,反之亦然 [§6, Fig 9]。这意味着 γ_temp 可以作为 SRC 的辅助调节参数

7. **Prompt text masking 降低了对 prompt 语速的敏感度**: 快 prompt WER 从 4.21 降至 2.13,标准差从 8.10 降至 3.73 [Table 3]。但生成语速仍受 prompt 影响 (未完全解耦) [§5, Table 3, Fig 7]

8. **Filler 插入频率与语速负相关**: 2 SPS 时生成 ~50 个 fillers (有 filler prompt) / ~20 个 (无 filler prompt),5+ SPS 时几乎为 0 [Fig 5]。VoXtream2 是评估系统中唯一展示此自适应 filler 行为的 [§5.2]

## 局限性

1. **语速与 prompt 未完全解耦**: 尽管 SRC 显著减弱了 prompt 语速的影响,生成语速仍部分由 prompt rate 决定,且某些 prompt-target 组合 WER 增加 (慢 prompt + 快 target, 快 prompt + 慢 target) [§7, Fig 7]。根本原因: prompt 和生成共享同一个 encoder,next-token prediction 倾向于延续 prompt 的说话风格 [论文原文]

2. **慢速 SRC 质量不稳定**: 1 SPS 时 WER 16.51%,hallucination 和过度重复是主因,训练数据中超慢 duration state 的不足是根源 [§5.3, Table 5]

3. **训练依赖外部 forced aligner**: 虽然推理时不再需要 aligner (通过 prompt text masking 解决),但训练数据仍需 Clap-IPA 对齐,约 35% 数据因对齐或转录无效被丢弃 [§7]

4. **仅英语**: 40k 小时全部为英语数据。虽然 IPA phonemizer 理论上支持多语言,但未做定量评估 [§3.2]

5. **动态 SRC 突变场景质量下降**: S↔F / F↔S 的 WER (9.55-11.20%) 远高于 no SRC (3.26%) 和静态 normal (3.20%) [Table 5]

## 点评

VoXtream2 在 VoXtream 的 full-stream 基础上做了三个实质性推进: (1) distribution matching SRC, (2) multi-condition CFG, (3) prompt text masking。这三个改进分别解决了可控性、音质/相似度、实用性问题,使系统从一个 proof-of-concept 演进为更接近实际部署的方案。

**Distribution matching 是本文最有启发性的设计。** 与 KB 中记录的 duration predictor 优化路线 (GRPO/DPO/MoE-DP) 不同,distribution matching 完全是 inference-time 的方案: 不需要额外训练,通过直方图匹配在线校正 duration 采样。这个思路有几个值得注意的属性:
- **Self-correcting**: 滑动窗口 Pacc 使控制信号自动适应当前状态,不需要显式规划
- **Composable**: 可以与任何 AR TTS 的 duration token 系统组合
- **Dynamic**: β 和 Ptarget 可以在生成过程中随时改变,实现 mid-utterance 变速

但这个方案也有明显天花板: 它只能在训练数据覆盖的 duration state 范围内工作,超出范围 (如 1 SPS) 就会 hallucinate。

**与 VoXtream (v1) 的对比** 清晰展示了从 9k→40k 小时数据 + 架构改进的综合效果: SPK-SIM 从 0.529 (v1, SEED-en) 提升至 0.656 (v2),UTMOS 从 3.88 提升至 4.05,WER 从 3.82% 降至 1.32%。FPL 从 102ms 降至 74ms。这些改进中,CFG (特别是 audio + speaker) 和 prompt enhancement 对 SPK-SIM 和 UTMOS 贡献最大,而数据扩展和架构更新对 WER 贡献最大 (Table 6 ablation)。

**与 CosyVoice 2 的 SRC 对比**特别有意义: CosyVoice 2 的 instructed generation 仅在 3-4 SPS 范围内有效调节 [Fig 6],而 VoXtream2 实现 2-5 SPS 连续控制。但 CosyVoice2 的 SPK-SIM (0.658) 仍略优于 VoXtream2 (0.656),差距从 v1 的 0.127 缩小至 0.002,基本追平。这说明多条件 CFG + prompt enhancement 有效弥补了纯 AR 方案在音色保真上的劣势。

## 可复用的 idea

1. **Distribution matching 做采样时语速/韵律控制**: 在 AR 模型的 duration/style token 采样阶段,用目标分布直方图重加权当前预测分布 + 滑动窗口在线校正。不需要额外训练,可以即时切换目标。可泛化到任何有离散 style/duration token 的 AR 系统。

2. **CFG 的间接韵律影响**: 即使 CFG 不直接作用于 duration 采样,γ 参数仍然通过 conditioning 强度间接影响语速行为。这意味着 CFG γ 可以作为"第二调节杠杆"辅助 SRC,在需要时偏向慢速或快速。

3. **Prompt text masking + CFG 的组合**: 保留 masked tokens (而非直接 drop) 使 CFG 在 prompt 条件上也能工作。这比简单的 text prefix dropping 更灵活,因为它允许在推理时通过 γ 调节 prompt 依赖强度。

4. **Acoustic prompt enhancement 对冲 CFG 音质退化**: 当 CFG 增强 speaker similarity 时,生成质量趋近 prompt 质量。预处理增强 prompt 是一个低成本的缓解方案 (仅处理 prompt,不增加推理延迟)。

5. **隐式 filler 生成**: 训练时从转录中移除 filler words 并合并时长,模型自动学会根据 duration state (慢速时长 phoneme → 插入 filler) 决定 filler 位置。这比显式 filler 标注更自然、更可扩展。

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 5 个设计选择均有因果解释,distribution matching 机制解释详尽 |
> | 可信赖 | pass | 数字出处标注 >90%,与 PDF 交叉验证一致 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注贯穿方法节,覆盖率 ~90% |
> | 可定位 | pass | KB 背景引用 DurationPredictor 演进线 + CosyVoice 2 SRC 做对比基准 |
> | 不污染 | pass | 笔记内容无 overclaim,KB 更新待执行 |
> 
> Issues: 2 (high: 0, medium: 0, low: 2)
> 详见 `_review/VoXtream2-review.yml`
