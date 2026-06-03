---
type: paper
tier: deep
title: "E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS"
arxiv_id: "2406.18009"
source: "Sources/E2_TTS.pdf"
authors: [Sefik Emre Eskimez, Xiaofei Wang, Manthan Thakker, Canrun Li, Chung-Hsien Tsai, Zhen Xiao, Hemin Yang, Zirun Zhu, Min Tang, Xu Tan, Yanqing Liu, Sheng Zhao, Naoyuki Kanda]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, zero-shot, flow-matching, non-autoregressive, mel-spectrogram, simplification]
concepts: ["[[Conditional Flow Matching]]", "[[Non-autoregressive TTS]]", "[[Classifier-Free Guidance]]", "[[Duration Predictor]]", "[[Mel Spectrogram]]"]
models: ["[[模型库/NaturalSpeech 3|NaturalSpeech 3]]", "[[模型库/NaturalSpeech 2|NaturalSpeech 2]]", "[[模型库/BigVGAN|BigVGAN]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: E2 TTS 处于 NAR zero-shot TTS 的 flow-matching 路线上,是 Voicebox 的极简化后继。与 [[Conditional Flow Matching]] 页记录的 CosyVoice 系列(LLM + CFM 两阶段)不同,E2 TTS 是纯 CFM 单阶段系统,不依赖 LLM 或离散 token。与 [[Non-autoregressive TTS]] 页记录的 FastSpeech 系列依赖 [[Duration Predictor]] + phoneme alignment 的路线相比,E2 TTS 通过 filler token 机制彻底消除了显式 duration 预测和 phoneme alignment 的需求,是 NAR TTS 简化的一个极端案例。

**已有认知**: [[Zero-shot Speech Synthesis]] 任务页记录了三大路线(LLM+离散token / Diffusion&Flow / Coarse-to-fine hybrid),E2 TTS 属于第二条 Diffusion/Flow-based 路线,与 NaturalSpeech 3 同属此类但架构极度简化。[[Classifier-Free Guidance]] 页记录了 CFG 在 TTS 中通过条件 dropout 增强生成质量的标准用法,E2 TTS 延续了这一做法(20% dropout)。

**创新判断**: 对比 KB 中已有方法,E2 TTS 的核心创新在于证明了 flow matching + speech infilling 框架下,连 phoneme alignment、G2P converter、duration predictor 这些传统被认为"必要"的组件都可以去掉,仅用字符序列 + filler token 就能达到人类水平的零样本 TTS。这直接挑战了 [[Duration Predictor]] 页记录的"duration predictor 是 NAR TTS 核心组件"这一共识。

> 检索命中: [[Conditional Flow Matching]]$\checkmark$, [[Zero-shot Speech Synthesis]]$\checkmark$ | 过滤: [[Non-autoregressive TTS]](待确认), [[Classifier-Free Guidance]](待确认), [[Duration Predictor]](待确认), [[Mel Spectrogram]](待确认) | 未命中但可能相关: Voicebox(无实体页)

## 速查

> [!summary] 速查
> - **一句话**: 用字符序列+filler token 替代 phoneme alignment/G2P/duration model,使 flow-matching TTS 简化到只有 mel spectrogram generator + vocoder 两个模块,仍达 SOTA 零样本 TTS
> - **路线**: 字符文本+filler token padding → Flow-matching Transformer (U-Net skip connections) → mel spectrogram → BigVGAN vocoder → waveform
> - **指标**: WER 1.9% / SIM-o 0.708 (预训练初始化, LibriSpeech-PC test-clean) [Table 1]; CMOS -0.05 vs GT (不可区分) [Table 2]
> - **可借鉴**: filler token 机制 -- 用特殊填充 token 将短文本序列 pad 到与 mel 等长,让模型自己学文本-音频对齐,无需任何外部对齐工具; speech infilling 训练范式无需 paired phoneme alignment 数据
> - **局限**: 收敛极慢(需完整 800K 步训练才超过 Voicebox,早期阶段 WER 远高于有 alignment 的模型) [Fig 4]; 推理仍需指定目标时长; 训练数据要求高(50K+ hours); 未开源

## 核心问题

E2 TTS 要解决的核心问题是: **现有 NAR zero-shot TTS 系统过于复杂**。

具体而言 [§1]:
1. NaturalSpeech 2/3 和 Voicebox 需要 frame-wise phoneme alignment(需外部 aligner 工具)
2. Matcha-TTS 需要 monotonic alignment search + 独立 duration model
3. E3 TTS 需要精心设计的 cross-attention U-Net 架构
4. 几乎所有系统都需要 grapheme-to-phoneme (G2P) converter 和 text normalizer

作者的核心洞察是: **这些复杂组件不仅不必要,有时甚至有害**。论文通过实验证明,去掉 phoneme alignment 后自然度反而提升 [§3.4],因为联合建模让模型学到了比外部工具更好的 grapheme-to-phoneme 映射 [§3.6.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

E2 TTS 的架构极度简化,只包含两个模块 [§2]:
1. **Flow-matching-based mel spectrogram generator**: 一个带 U-Net skip connections 的 vanilla Transformer
2. **Vocoder**: BigVGAN,将 mel spectrogram 转为 waveform

没有 G2P converter、phoneme aligner、duration predictor、text normalizer。

### 关键设计选择

#### 1. Filler Token 机制 (核心创新)

**问题**: NAR TTS 需要解决文本序列(短)和 mel 序列(长)之间的长度不匹配。传统方法用 phoneme alignment 或 duration predictor 来桥接。

**E2 TTS 的方案** [§2.1]: 在字符序列末尾追加特殊 filler token `<F>` 使其长度等于 mel 帧数:
```
y_hat = (c1, c2, ..., cM, <F>, <F>, ..., <F>)    # 总长度 = T (mel 帧数)
```

**为什么这能 work** [论文原文]: 模型通过 speech infilling 训练任务自动学会了字符与音频帧的对应关系,无需显式对齐。这意味着 E2 TTS 的 mel spectrogram generator 可以被视为 G2P converter + duration model + audio model 的联合模型 [§2.4]。

**为什么联合建模更好** [论文原文]: 作者认为联合建模避免了外部工具引入的错误传播,并且在大规模训练数据上,模型能学到比外部 G2P 和 aligner 更准确的映射 [§3.6.1]。[agent 解读]: 这与 end-to-end 学习的一般规律一致 -- 分离的模块各自优化局部目标,联合训练则直接优化全局目标(生成质量)。

#### 2. Speech Infilling 训练

模型学习条件分布 P(m * s | (1-m) * s, y_hat) [§2.1],其中:
- m 是二值时间掩码(随机掩码 70%-100% 的 mel 帧)
- s 是 mel spectrogram
- y_hat 是带 filler token 的字符序列

这与 Voicebox 的训练方式完全相同 [§2.4],区别仅在于条件信息从 frame-wise phoneme sequence 变为 character + filler token sequence。

#### 3. 推理流程

推理时 [§2.2]:
1. 将 audio prompt 的 mel 和 transcription 与目标文本拼接
2. 目标文本部分用全零矩阵 z_gen 作为 mel 输入(待生成区域)
3. 字符序列: y'_hat = (y_aud, y_text, <F>, ..., <F>),总长 = T_aud + T_gen
4. 需要指定 T_gen(目标语音时长),可通过训练的 duration model 或任意指定

#### 4. 与 Voicebox 的关系

从 Voicebox 视角 [§2.4]: E2 TTS 将 Voicebox 的 frame-wise phoneme sequence 替换为 character + filler token sequence。这一替换消除了 G2P、phoneme aligner 和 duration model 三个组件。

[agent 解读]: E2 TTS 本质上是对 Voicebox 的"减法创新" -- 不是增加新模块,而是通过发现某些被认为必要的模块其实不必要,从而极大简化系统。

### 训练策略

**模型配置** [§3.2]:
- Transformer: 24 layers, 16 heads, dim=1024, FFN dim=4096, 335M parameters
- U-Net style skip connections
- 100-dim log mel-filterbank, 10.7ms hop size, 24kHz
- 字符词表: 399 个(训练数据中出现的所有字符/符号,无过滤)
- CFG: 20% 概率丢弃所有条件信息 [§3.2]
- 训练: 800K steps, batch size 307,200 frames, linear decay LR (peak 7.5e-5)

**数据** [§3.1]:
- 主实验: Libriheavy 50K hours
- 扩展实验: 私有 200K hours

**预训练** [§3.2]: 可选的无监督预训练(200K hours 无标注数据, 800K steps),显著提升性能 ((P1) vs (P2))。

### 扩展

#### E2 TTS X1: 推理时无需 prompt 转录 [§2.5.1]

训练时使用被掩码区域的转录作为 y(而非完整音频转录)。推理时 y'_hat 不包含 y_aud,只有目标文本 + filler tokens。通过 MFA 确定训练时掩码区域的边界(不切断词中间)。

**效果**: 与基本 E2 TTS 几乎持平(预训练版 WER 2.0% vs 1.9%, SIM-o 0.705 vs 0.708) [Table 3]。

#### E2 TTS X2: 可指定部分词的发音 [§2.5.2]

训练时以 15% 概率将词替换为括号包裹的 CMU 音素序列。推理时可对特定词(如外国人名)指定发音。

**效果**: 即使 50% 的词替换为音素,WER 仅从 1.9% 升至 2.1% [Table 4]。

## 实验

| 指标 | E2 TTS (P2, best) | Voicebox (B5) | NaturalSpeech 3 (B2) | VALL-E (B1) | Ground Truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER (%) | **1.9** | 2.2 | 2.6 | 4.9 | 2.0 | LibriSpeech-PC test-clean | [Table 1] |
| SIM-o | **0.708** | 0.695 | 0.632 | 0.500 | 0.695 | LibriSpeech-PC test-clean | [Table 1] |
| CMOS | **-0.05** | -0.78 | -0.98 | - | 0.00 | LibriSpeech-PC test-clean | [Table 2] |
| SMOS | 4.65 | 4.73 | 4.76 | - | 3.91 | LibriSpeech-PC test-clean | [Table 2] |

**关键发现**:

1. **Phoneme alignment 是自然度瓶颈** [§3.4]: (B4) Voicebox vs (P1) E2 TTS 的对比(同训练集,同配置,仅差 alignment),E2 TTS 在 WER (2.0 vs 2.2)、SIM-o (0.675 vs 0.667)、CMOS (-0.14 vs -0.78) 全面优于 Voicebox。[论文原文] 作者明确指出"phoneme alignment was the major bottleneck in achieving better naturalness"。

2. **预训练的显著效果** [Table 1]: (P1) vs (P2) 对比,无监督预训练将 SIM-o 从 0.675 提升至 0.708,同时 WER 从 2.0 降至 1.9。

3. **数据 scaling 有效** [Table 1]: (P3) 使用 200K hours 从头训练,达到与 (P2) 预训练相同的 WER 1.9%,SIM-o 0.707。

4. **收敛速度慢但终态更好** [Fig 4]: E2 TTS 在训练前 50% 阶段 WER 远高于 Voicebox(因为模型需要同时学习 G2P mapping + duration + 声学生成),但最终收敛后超过 Voicebox。[论文原文] 作者推测这是因为 E2 TTS 在大规模数据上学到了更好的 grapheme-to-phoneme 映射。

5. **对 prompt 长度鲁棒** [Fig 5]: 4-10 秒 prompt 长度范围内 WER 无明显变化,SIM-o 随 prompt 变长显著提升。

6. **对语速变化鲁棒** [Fig 6]: 语速从 0.7x 到 1.3x,WER 仅中等程度上升,SIM-o 基本稳定。

## 局限性

1. **收敛极慢** [Fig 4]: 需要完整的 800K 步训练才能超越有 alignment 的 Voicebox。在训练资源受限时,Voicebox 的 early-stage 表现更好。[agent 解读] 这可能限制了该方法在小数据/低计算预算场景的应用。

2. **需要指定目标时长**: 推理时仍需要提供 T_gen(目标语音帧数),论文使用了独立训练的 duration model 进行公平对比 [§3.2]。虽然实验表明模型对时长变化鲁棒 [§3.6.3],但这仍是一个需要外部指定的超参数。

3. **评估局限**: 仅在英语朗读风格数据(LibriSpeech)上评估,未验证对话场景、多语言、情感语音等更复杂的场景。

4. **SMOS 不高**: 虽然 CMOS 接近 GT(自然度优秀),但 SMOS (4.65) 低于 Voicebox (4.73) 和 NaturalSpeech 3 (4.76) [Table 2],说话人相似度的主观评价并不领先。[agent 解读] 不过所有系统的 SMOS 都高于 GT (3.91),论文解释这是 LibriSpeech 中说话人用不同声线朗读不同角色导致的 [§3.4, footnote 8]。

5. **大数据依赖**: 50K hours 起步,scaling 到 200K hours 才充分发挥优势 [Table 1]。

6. **未开源**: 论文仅提供 demo samples 和测试集,未开源模型或训练代码。

## 点评

E2 TTS 是一个典型的"减法创新"案例。在 TTS 领域,从 Tacotron 到 FastSpeech 到 VITS,复杂度不断增加(attention → alignment → duration predictor → VAE → flow → ...),E2 TTS 反其道而行,问"哪些组件其实不必要?",并通过实验证明答案是"G2P、phoneme aligner、duration model 都不必要"。

这篇工作的价值不仅在于系统简化,更在于提供了一个反直觉的实证:phoneme alignment 不仅不必要,实际上**有害于自然度**(CMOS 对比)。这推翻了 NAR TTS 领域多年的默认假设。

从方法论角度,filler token 机制的成功说明:在足够大的训练数据和模型容量下,Transformer 能自动学会隐式的文本-音频对齐,且这种学到的对齐比外部工具提供的 forced alignment 更优。这与 NLP 领域"scaling + 端到端学习 > 显式规则/工具"的趋势一致。

不过值得注意的是,收敛速度是显著代价。E2 TTS 的 simplicity 来自于将对齐学习的负担从外部工具转移到了训练过程本身,这需要大量数据和计算。在数据/计算受限场景下,传统 alignment 路线可能仍有优势。

[agent 解读] 后续工作 F5-TTS 在此基础上进一步发展,使用 DiT 架构和更高效的训练,成为了开源社区广泛使用的方案,验证了 E2 TTS 提出的"无 alignment"路线的可行性和影响力。

## 可复用的 idea

1. **Filler token for length matching**: 用特殊 token 将短序列 pad 到长序列长度,让模型自己学习对应关系。可迁移到其他需要序列长度对齐的生成任务(如 text-to-image, text-to-music 中的时间轴对齐)。

2. **Speech infilling as unified training objective**: 将 TTS 统一为 speech infilling 任务(prompt 部分不掩码,生成部分全掩码),自然支持零样本能力,无需额外的 speaker conditioning 设计。

3. **"减法"设计思路**: 在系统变得越来越复杂时,回头问"哪些组件真的必要?"并用实验验证。这种方法论在其他领域(如 ASR 从 pipeline 到 end-to-end)也反复被证明有效。

4. **渐进式扩展 (X1/X2)**: 在基础系统上通过训练数据构造(而非架构修改)添加功能(免转录推理、发音指定),保持架构不变。

> [!review] 审阅: pass-with-fixes (2026-06-03)
> 3 issues (0 high, 1 medium, 2 low): models 字段含评估工具已修正; Speech-Text Alignment 概念挂接语义不匹配已修正; 点评节外部知识标注已补充。
> 详见 `_review/E2 TTS-review.yml`

---

*检索命中: [[Conditional Flow Matching]], [[Zero-shot Speech Synthesis]] | 过滤: [[Non-autoregressive TTS]](待确认), [[Classifier-Free Guidance]](待确认), [[Duration Predictor]](待确认), [[Mel Spectrogram]](待确认) | 未命中但可能相关: Voicebox(无实体页)*
