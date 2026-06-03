---
type: paper
tier: deep
title: "FlexSpeech: Towards Stable, Controllable and Expressive Text-to-Speech"
arxiv_id: "2505.05159"
source: "Sources/FlexSpeech.pdf"
authors: [Linhan Ma, Dake Guo, He Wang, Jin Xu, Lei Xie]
year: 2025
venue: "arXiv (eess.AS)"
tags: [TTS, zero-shot, flow-matching, duration-prediction, DPO, style-transfer, NAR, preference-alignment]
concepts: ["[[Conditional Flow Matching]]", "[[Duration Predictor]]", "[[Non-autoregressive TTS]]", "[[Style Transfer in TTS]]", "[[Prosody Modeling]]", "[[Classifier-Free Guidance]]", "[[Speaker Embedding]]", "[[Differentiable Reward Optimization]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]", "[[模型库/NaturalSpeech 3|NaturalSpeech 3]]", "[[模型库/BigVGAN|BigVGAN]]"]
tasks: ["[[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]"]
datasets: ["[[数据集/Emilia|Emilia]]", "[[数据集/SEED-TTS-Eval|SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[Conditional Flow Matching]]、[[Prosody Modeling]]、[[Speaker Embedding]] + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: FlexSpeech 位于 NAR TTS 与 AR prosody modeling 的交叉点。在已有知识库中,TTS 的 duration prediction 经历了 "FastSpeech 显式 duration → Glow-TTS MAS → MaskGCT T2D model → DMOSpeech 2 GRPO-optimized duration" 的演进。FlexSpeech 提出了一条新路线: **AR duration predictor + DPO 偏好对齐**,与 DMOSpeech 2 的 GRPO-optimized duration 理念相近(都是对 duration 组件做 RL/偏好优化),但优化方法不同(DPO vs GRPO)。
>
> **已有认知**: KB 中 [[Conditional Flow Matching]] (confirmed) 详细记录了 OT-CFM 在 TTS 中的应用谱系(CosyVoice 系列、F5-TTS、VoiceFlow 等)。[[Duration Predictor]] [待确认] 梳理了从 FastSpeech 到 DMOSpeech 2 的 duration 建模演进,特别指出 DMOSpeech 2 的发现: "最优 duration 不等于真实 duration"(RL-optimized WER 1.752 < GT duration WER 1.821)。[[Prosody Modeling]] (confirmed) 将韵律信息分为 duration/pitch/energy/pause 四维度,FlexSpeech 专注于 duration 维度的 AR 建模。
>
> **创新判断**: FlexSpeech 的核心创新在于将 AR + DPO 引入 duration predictor(而非整个 TTS pipeline),实现了 duration 的 Markov 依赖建模 + 人类偏好对齐。与 [[Differentiable Reward Optimization]] [待确认] 中记录的 DiffRO/GRPO 方案对比,FlexSpeech 的 DPO 是最轻量的偏好优化(不需要 reward model,仅需人工标注的 win-lose 对)。与 MegaTTS 系列的 prosody/timbre 解耦思路类似,但 FlexSpeech 的解耦更彻底: 完全独立的 duration model + acoustic model,且 DPO 仅作用于 duration 不影响 acoustic model。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Prosody Modeling]]✓, [[Speaker Embedding]]✓ | 过滤: [[Duration Predictor]](待确认), [[Non-autoregressive TTS]](待确认), [[Style Transfer in TTS]](待确认), [[Differentiable Reward Optimization]](待确认) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 TTS 解耦为 AR duration predictor + NAR flow-matching acoustic model,通过 DPO 对 duration predictor 做偏好对齐,以约 100 条数据实现快速风格迁移
> - **路线**: Text → Phoneme + MFA duration → AR Duration Predictor (encoder-decoder, next-token prediction) → Expanded Phoneme Sequence → Flow-Matching DiT Acoustic Model (+ speaker embedding) → Mel Spectrogram → BigVGAN → 48kHz Waveform
> - **指标**: WER 1.20% (Seed-TTS test-zh) / 1.81% (Seed-TTS test-en), SMOS 3.93/3.92; CMOS 与 MegaTTS 3 持平或略优 [Table 1]
> - **可借鉴**: (1) 对 duration predictor 单独做 DPO 是极轻量的偏好对齐路线,仅 50 对数据就有效; (2) duration + acoustic 完全解耦使风格迁移只需微调 duration 模型(~100 samples),acoustic 模型不动; (3) duration 离散化为分类问题(帧数 0-99)而非回归,适合 next-token prediction
> - **局限**: (1) 依赖外部 forced alignment (MFA) 提取训练 duration 标签; (2) SIM-O 偏低(0.60-0.68 vs MaskGCT 0.717-0.774),可能因 ECAPA-TDNN speaker embedding 表达力有限; (3) 未开源代码和模型权重; (4) 韵律控制仅限 duration 维度,pitch/energy 无显式控制

## 核心问题

FlexSpeech 试图同时解决 TTS 中的两个长期矛盾:
1. **稳定性 vs 自然度**: NAR 方法(如 FastSpeech)通过显式 duration 保证稳定但韵律单调;AR 方法(如 VALL-E)通过隐式 duration 建模获得自然韵律但容易跳字/重复 [§1]
2. **风格迁移的效率**: 传统方法需要大量目标风格数据微调整个模型;FlexSpeech 希望仅用约 100 条数据微调 duration 即可完成风格迁移 [§5]

核心假设: **duration 是连接稳定性和自然度的关键变量** — 只要 duration 预测足够好(有 Markov 依赖 + 人类偏好对齐),NAR acoustic model 就能稳定地生成高质量语音 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

FlexSpeech 将 TTS 完全解耦为两个独立模块 [§3, Fig 1]:

```
[Text] → G2P → Phoneme Seq
                  ↓
[Reference Audio] → ECAPA-TDNN → Speaker Embedding (192-dim)
                  ↓
[Duration Model] (AR encoder-decoder)
    Encoder: 8-layer Transformer (bidirectional attention) + mask learning loss
    Decoder: 8-layer Transformer (causal attention) + next-token prediction
    Input: phoneme + reference mel + prompt duration
    Output: target phoneme duration d_1, d_2, ..., d_N (discrete class 0-99)
                  ↓
[Length Regulator] 按 duration 扩展 phoneme → expanded sequence (长度 = mel 帧数)
                  ↓
[Acoustic Model] (NAR, flow matching)
    Backbone: 22-layer DiT (Diffusion Transformer), 330M params
    Input: expanded phoneme embedding + repeated speaker embedding + Gaussian noise (channel concat)
    Condition: timestep t → adaLN-zero
    Training: OT-CFM loss (Eq.5) + CFG (drop rate 0.3)
    Output: 80-dim mel spectrogram
                  ↓
[BigVGAN] → 48kHz waveform
```

### 关键设计选择

**1. 为什么用 AR 预测 duration 而不是传统回归?**

[论文原文] NAR duration predictor 独立预测每个音素的 duration,不建模音素间的依赖关系,导致韵律缺乏丰富性 [§1]。AR next-token prediction 让每个 duration 的预测依赖前一个 duration(Markov 依赖),捕捉自然语音中的节奏模式。

[agent 解读] 将 duration 离散化为 0-99 的分类问题(每个 class 对应一帧 mel)是使 AR next-token prediction 可行的关键。如果保持回归目标,AR 模型难以直接预测连续值的分布。这与 MaskGCT 的 T2D model(同样离散化 duration)思路一致,但 MaskGCT 预测的是总时长,FlexSpeech 预测的是每个音素的帧数。

**2. 为什么 acoustic model 不用 in-context learning(如 F5-TTS)?**

[论文原文] In-context learning 方式(如 F5-TTS)会让 acoustic model 学到参考音频的 duration 特征,与 expanded phoneme sequence 中已有的 duration 信息冲突 [§3.1]。因此 FlexSpeech 用 ECAPA-TDNN 提取 utterance-level speaker embedding(仅编码音色),与 duration 完全解耦。

[agent 解读] 这是一个重要的架构决策: 它牺牲了 in-context learning 的灵活性(F5-TTS 可以从参考音频自动学韵律),换来了 duration 的完全可控性。这也解释了为什么 FlexSpeech 的 SIM-O 相对偏低 — ECAPA-TDNN 的 utterance-level embedding 可能丢失了细粒度的说话人特征。

**3. 为什么 DPO 作用于 duration 而非 acoustic model?**

[论文原文] SFT 后的 duration model 统计上能捕捉 duration 分布,但生成的韵律模式不完全符合人类偏好(如不自然的停顿、过于机械的节奏) [§3.4]。DPO 通过 win-lose duration 对直接优化 duration 预测,使其更贴近人类偏好。

[agent 解读] 这是 FlexSpeech 与 SpeechAlign/DiffRO/GRPO 等全 pipeline RL 方案的根本区别。FlexSpeech 的 DPO 只优化 duration 这一个组件,acoustic model 完全不动。这使得: (1) 优化空间极小,50 对数据就有效; (2) 风格迁移只需微调 duration model; (3) 不存在 reward hacking 对音质的副作用。但代价是: DPO 无法改善 acoustic model 本身的音质问题。

**4. Encoder 的 mask learning loss 的作用**

[论文原文] 随机选择句子,对部分音素做 mask,用线性投影从 hidden representation 预测被 mask 的音素,用 cross-entropy loss 约束 [§3.2, Eq.12]。目的是让 encoder 更好地编码整个文本的语义信息。

[agent 解读] 这本质上是 BERT-style 的 masked language modeling 预训练策略,但应用于 duration model 的 encoder。mask 概率为 0.15,与 BERT 一致。这有助于 encoder 学到更好的 phoneme-level contextual representations,使 decoder 在预测 duration 时有更丰富的上下文。

**5. Reference mel-spectrogram 的双重使用**

Reference mel-spectrogram 同时用于 [§3.2]:
- Duration model: 通过 cross-attention 提供 acoustically relevant feature,影响 duration 预测的韵律风格
- Acoustic model: 通过 ECAPA-TDNN 提取 speaker embedding,仅提供音色信息

[agent 解读] 这种设计实现了 "风格来自 duration,音色来自 speaker embedding" 的解耦。在 inference 时,duration model 的 reference 决定了 "怎么说"(韵律节奏),acoustic model 的 reference 决定了 "谁在说"(音色)。两者可以来自不同的参考音频,实现 timbre/style 的独立控制。

### 训练策略

三阶段训练 [§4.1-4.2]:

| 阶段 | 目标 | 数据 | 步数 |
| --- | --- | --- | --- |
| Stage 1: Pretrain | Acoustic model + Duration model | ~90K hours Emilia (EN+ZH) | AM: 800K steps; DM: 1M steps [§4.2] |
| Stage 2: SFT | 提升音质和自然度 | 1K hours 高质量内部数据 | AM: 50K steps; DM: 100K steps [§4.2] |
| Stage 3: DPO | Duration 偏好对齐 | ~50-1000 win-lose duration 对 | 未明确报告步数 [§3.4] |

DPO 的偏好数据构建 [§3.4, Appendix A]:
1. 自动预筛选: 计算 WER + 检测异常停顿,过滤明显错误样本
2. 人工标注: 标注员听 audio 对,判断哪个更自然(60 对/小时效率)
3. Guidelines 强调: 自然度、异常停顿、韵律相似性

Inference 配置 [§4.2]:
- Euler solver, 32 time steps, CFG strength 2
- Duration model: top-k=6, top-p=0.5, temperature=0.9, repetition penalty=1.0
- EMA weights for acoustic model

## 实验

### 主要结果 (Zero-shot TTS)

| 指标 | FlexSpeech (#1k DPO) | F5-TTS | MaskGCT | CosyVoice | MegaTTS 3 | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER(%)↓ | **1.20** | 1.56 | 2.273 | 3.10 | - | 1.26 | Seed-TTS test-zh | [Table 1] |
| WER(%)↓ | **1.81** | 1.83 | 2.623 | 3.39 | 2.31 | 2.06 | Seed-TTS test-en | [Table 1] |
| WER(%)↓ | 2.64 | 2.42 | - | 3.59 | - | 2.23 | LibriSpeech-PC | [Table 1] |
| SIM-O↑ | 0.68 | 0.76 | **0.774** | 0.75 | 0.70 | 0.76 | Seed-TTS test-zh | [Table 1] |
| SIM-O↑ | 0.62 | 0.67 | **0.717** | 0.64 | - | 0.73 | Seed-TTS test-en | [Table 1] |
| SMOS↑ | **3.93** | 3.85 | 3.80 | 3.63 | - | 3.78 | Seed-TTS test-zh | [Table 1] |
| SMOS↑ | **3.92** | 3.93 | 3.81 | 3.59 | - | 3.94 | Seed-TTS test-en | [Table 1] |
| CMOS↑ | 0.00 (anchor) | -0.10 | -0.15 | -0.29 | - | -0.21 | Seed-TTS test-zh | [Table 1] |

**关键发现**:
- WER 达到 SOTA: test-zh 1.20% 甚至低于 GT 的 1.26%,说明 FlexSpeech 的稳定性极强 [Table 1]
- SIM-O 偏低: 0.62-0.68,显著低于 MaskGCT (0.717-0.774) 和 F5-TTS (0.67-0.76) [Table 1]
- SMOS 和 CMOS 都领先: 主观评估显示 FlexSpeech 的韵律自然度优于所有 baseline [Table 1]
- #50 vs #1k DPO 几乎无差异: 50 对 DPO 数据与 1000 对效果相当 [Table 1]

### 消融实验

| 方法 | WER(%) test-en | SMOS test-en | WER(%) test-zh | SMOS test-zh | 出处 |
| --- | --- | --- | --- | --- | --- |
| FlexSpeech (full) | 1.86 | 3.91 | 1.20 | 3.89 | [Table 2] |
| w/o DM-DPO | 2.94 | 3.71 | 2.54 | 3.72 | [Table 2] |
| w/o DM-DPO+SFT | 6.65 | 3.31 | 5.93 | 3.42 | [Table 2] |
| w/o AM-SFT | 3.44 | 3.88 | 3.17 | 3.80 | [Table 2] |

**消融发现** [§4.5]:
- DPO 对 WER 改善显著: 2.94 → 1.86 (test-en), 2.54 → 1.20 (test-zh) [Table 2]
- Duration SFT 极为关键: 无 SFT 时 WER 暴涨至 6.65/5.93 [Table 2],原因是 Emilia 预训练数据有转录错误导致的对齐偏差 [§4.5]
- Acoustic SFT 也重要但影响较小: WER 3.44/3.17 [Table 2],原因是 phoneme-audio 偏差会混淆 acoustic model [§4.5]

### 快速风格迁移 (StoryTTS)

在 StoryTTS 故事风格数据上做 DPO 迁移 [§5, Fig 2]:
- 10 对: WER 略降,SMOS 提升有限
- 50 对: WER 稳定,基本收敛
- 100 对: SMOS 达到平台,风格迁移基本完成
- 200-1000 对: 几乎无额外收益

[论文原文] 约 100 对数据即可完成快速风格迁移,无需调整 acoustic model [§5]。

## 局限性

1. **SIM-O 偏低**: 0.60-0.68 显著落后于 MaskGCT (0.717-0.774) 和 F5-TTS (0.67-0.76) [Table 1]。[agent 解读] 根本原因可能是 ECAPA-TDNN utterance-level embedding 的信息瓶颈 — 相比 in-context learning 方式,固定维度的 speaker embedding 丢失了细粒度的说话人特征。
2. **依赖外部对齐工具**: 训练需要 MFA 提取的 phoneme-level duration 标签 [§4.2],增加了数据准备成本,且 forced alignment 在 noisy data 上不可靠(消融实验已证实) [§4.5]。
3. **仅控制 duration 维度**: 韵律的 pitch/energy/pause 维度没有显式控制通道 [agent 解读],所有非 duration 韵律信息只能靠 acoustic model 隐式恢复。
4. **未开源**: 代码和模型权重未公开,仅提供 demo page。
5. **评估偏差风险**: CMOS 评估中,与非开源系统(NS3, MegaTTS 2/3)的对比是从 demo page 下载样本做平行评估 [§4.4],样本可能非对应测试集。

## 点评

**与 MegaTTS 系列的关系**: FlexSpeech 论文明确承认与 MegaTTS 的密切关系 [§1],核心差异在于: (1) MegaTTS 没有 DPO 机制; (2) FlexSpeech 通过 duration 解耦实现了更灵活的风格迁移。但两者共享了 "phoneme duration 决定稳定性,prosody latent 决定自然度" 的思路。

**DPO on duration 的意义**: 这是一条非常轻量的偏好优化路线,与 DMOSpeech 2 的 GRPO on duration 形成对照。DMOSpeech 2 用自动 reward (WER+SIM) 优化 duration,FlexSpeech 用人工标注的 win-lose 对做 DPO。两者的共同发现是: **duration predictor 是 TTS pipeline 中偏好优化最有效的作用点**。

**SIM-O 的代价**: FlexSpeech 为了实现 duration/timbre 完全解耦,放弃了 in-context learning,用 ECAPA-TDNN speaker embedding 替代。这牺牲了 speaker similarity(SIM-O 0.60-0.68),但获得了 duration 的完全可控性。这是一个明确的 trade-off 选择。

**50 对 DPO 数据的效率**: 论文中最令人印象深刻的发现之一 — 仅 50 对 win-lose duration 数据就能显著提升稳定性和自然度 [Table 1]。这暗示 duration predictor 的优化空间非常 "窄"(比整个 TTS 的优化空间小得多),少量人类偏好信号就能有效引导。

## 可复用的 idea

1. **Duration 离散化 + AR next-token prediction**: 将连续的 duration 离散为帧数类别(0-99),转化为分类问题,使 Transformer decoder 的 next-token prediction 范式适用于 duration 建模。比回归目标更适合 AR 架构。
2. **Component-level DPO**: 不对整个 TTS pipeline 做偏好优化,而是精确靶向 duration predictor 这一组件。优化空间极小(~50 对数据即有效),无副作用(acoustic model 不动)。
3. **Duration/Timbre 完全解耦用于风格迁移**: Duration model 控制韵律风格,acoustic model + speaker embedding 控制音色。迁移新风格时只需微调 duration model(~100 samples),acoustic model 完全不动。
4. **Encoder mask learning**: 在 duration model 的 encoder 上加 BERT-style mask prediction loss,增强 phoneme-level contextual representation 的质量,间接改善 duration prediction。

> [!review] 审阅: pass (2026-06-03)
> **结论**: pass — 5 个原则均满足,2 个 low issues,0 个 high/medium。
> - low: 消融表未标明 DPO 数据量版本(原文未明确)
> - low: models 字段可扩展(保持当前即可)
> 详见 `_review/FlexSpeech-review.yml`
