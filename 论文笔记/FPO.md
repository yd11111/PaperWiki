---
type: paper
tier: deep
title: "FPO: Fine-grained Preference Optimization Improves Zero-shot Text-to-Speech"
arxiv_id: "2502.02950"
source: "Sources/2502.02950.pdf"
authors: [Jixun Yao, Yang Yuguang, Yuan Feng, Yu Pan, Ziqian Ning, Jianhao Ye, Hongbin Zhou, Lei Xie]
year: 2025
venue: "arXiv (journal submission)"
tags: [RLHF, DPO, preference-optimization, zero-shot-TTS, robustness, fine-grained-alignment, token-level-optimization, data-efficiency, codec-LM]
concepts: ["[[LLM-based TTS]]", "[[Semantic vs Acoustic Tokens]]", "[[Differentiable Reward Optimization]]", "[[TTS Evaluation]]"]
models: ["[[CosyVoice]]", "[[CosyVoice 2]]", "[[论文笔记/Llasa|Llasa]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个实体页: [[LLM-based TTS]]confirmed, [[Semantic vs Acoustic Tokens]]confirmed, [[CosyVoice]]confirmed, [[CosyVoice 2]]confirmed, [[TTS Evaluation]]pending-review)
> **谱系定位**: TTS 领域的 RL/偏好优化已形成三条路线: (1) 音频级 RL (Seed-TTS 的 REINFORCE, 2024); (2) utterance-level DPO/KTO (SpeechAlign, UNO, RIO, 2024); (3) token-level 可微优化 (DiffRO/CosyVoice 3, 2025)。FPO 处于路线 (2) 和 (3) 之间 -- 仍使用 DPO 框架,但将 loss 计算从 utterance-level 下沉到 token-level 的 problematic segments。
>
> **已有认知**: SpeechAlign 首次将 DPO 引入 codec LM,通过 golden vs synthetic AR tokens 构建偏好数据集; RIO 提出 reverse inference 自动选择偏好样本,无需人工标注; 两者均为 utterance-level 优化。概念库中 Differentiable Reward Optimization 页记录了 CosyVoice 3 的 DiffRO 路线 (token-level Gumbel-Softmax + reward model),以及 Multi-Reward GRPO 路线 (audio-level 多奖励)。NLP 中 Mask-DPO 和 Rho-1 已探索 token-level 选择性训练。
>
> **创新判断**: FPO 的核心贡献不是 token-level DPO 本身 (Mask-DPO 已存在),而是 (a) 针对 TTS 特有错误类型 (temporal modeling errors vs semantic-phonetic alignment errors) 设计了差异化的 fine-grained annotation 策略; (b) 在 CosyVoice/CosyVoice2/Llasa 三个 backbone 上验证了 3-4x 数据效率优势。
>
> 检索命中: [[LLM-based TTS]], [[Semantic vs Acoustic Tokens]], [[CosyVoice]], [[CosyVoice 2]], [[TTS Evaluation]] | 过滤: [[Differentiable Reward Optimization]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 TTS 偏好优化从 utterance-level 下沉到 token-level,通过分类 segmental errors (temporal modeling / semantic-phonetic alignment) 并选择性计算 loss,以 3-4x 数据效率显著降低 bad case ratio
> - **路线**: 多次采样生成 candidate pairs -> 自动评分选择 preference pairs -> fine-grained annotation (Whisper-timestamped + forced alignment 定位 error segments) -> token-level indicator function 选择性 DPO loss
> - **指标**: CosyVoice CER 3.63->1.83 (-49.6%), bad case 21%->9%; CosyVoice2 CER 1.45->1.32, bad case 14%->8%; Llasa CER 1.89->1.47, bad case 25%->11% [Table I]; NMOS +0.19/+0.10/+0.19 [Table II]; 200 utterances 即达 UNO 600 utterances 的 WER [Fig 4]
> - **可借鉴**: (1) 对 TTS segmental errors 的二分类 (temporal vs semantic-phonetic) + 不同 annotation 策略 (局部 vs 级联); (2) token-level indicator function 选择性 loss 计算,可迁移到任何序列模型的 fine-grained alignment
> - **局限**: fine-grained 人工标注成本高 (虽然数据量少但每条更复杂); SECS 无显著提升 (FPO 不优化 speaker similarity); 仅在 AR 模型上验证,连续表征 TTS 的适用性未探索

## 核心问题

现有 TTS 偏好优化方法 (SpeechAlign/UNO/RIO) 在 utterance-level 操作,但 TTS 的质量问题往往是局部的: 一段话中只有某几个词发音异常或韵律不自然,其余部分质量良好。utterance-level loss 将梯度均匀分配到整个序列,导致: (1) well-learned tokens 被不必要地扰动; (2) error tokens 的学习信号被稀释; (3) 需要更多数据才能统计性地 "过滤" 无关梯度 [§I]。

FPO 的核心假设: **targeted intervention on error-prone segments leads to more efficient and effective error correction** [§VII]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

FPO 是一个 post-training 框架,作用于已预训练的 neural codec LM (CosyVoice/CosyVoice2/Llasa)。Pipeline:

```
Step 1: Preference Data Collection (采样 + utterance-level 评分)
Step 2: Fine-grained Annotation (error segment 定位 + token-level 标签)
Step 3: Selective Training Loss (仅在 error tokens 上计算 DPO loss)
```

### 关键设计选择

**1. Segmental Error 分类 (WHY: 不同错误需要不同 annotation 策略)**

论文将 TTS 生成错误分为两类 [§III-C, Fig 1]:

| 类型 | 具体表现 | Annotation 范围 | 原因 |
|------|----------|----------------|------|
| **Temporal modeling errors** | 发音异常、异常静音、不自然韵律 (停顿) | 仅标注 problematic segment | 错误局限于特定时间段,后续 tokens 不受影响 [论文原文] |
| **Semantic-phonetic alignment errors** | 重复 (repetition)、截断 (truncation) | 从 error onset 到序列结束全部标注 | 一旦出现 alignment 错误,后续所有 tokens 都会受级联影响 [论文原文] |

[agent 解读] 这一分类的关键洞察是: temporal errors 是 "点故障" (局部修复即可),而 alignment errors 是 "链式故障" (需要从断点处重新学习整个后续序列)。这解释了为什么不能对所有错误使用同一种 annotation 策略。

**2. Preference Data Collection (WHY: 自动化降低人工标注成本)**

采用多指标综合评分自动选择 preference pairs [§III-B]:

$$s(y) = \lambda_w \cdot w(y)^p + \lambda_m \cdot m(y)^p + \lambda_c \cdot c(y)^p + \lambda_d \cdot \text{Dur}(y)^p$$

其中 w(y) = intelligibility (WER/CER), m(y) = quality (UTMOS), c(y) = speaker similarity (SECS), Dur(y) = duration difference [Eq.2]。设置阈值 tau=0.3 确保 preferred/dispreferred 之间有足够差异 [§III-B]。

[agent 解读] 虽然论文称"human annotators evaluate",但实际上大部分评估是通过预训练系统 (Whisper, UTMOS, WavLM-SV) 自动化完成的,人工主要参与 fine-grained prosody 错误的标注。

**3. Fine-grained Annotation Pipeline (HOW: 如何定位 error segments)**

Temporal modeling errors:
- 发音异常/异常静音: 使用 Whisper-timestamped + VAD 自动标注时间戳 [§III-C]
- 不自然韵律: ASR 转录 -> text LLM 确定最优停顿位置 -> 人工评估 -> forced alignment 获取时间戳 [§III-C]

Semantic-phonetic alignment errors:
- 通过 ASR 转录与 ground truth transcript 对比 + forced alignment 定位错误起始点 [§III-C]

Token-level indicator function [Eq.4]:
$$I(y^i) = \begin{cases} 1 & \text{if } y^i \text{ in error segment} \\ 0 & \text{otherwise} \end{cases}$$

**4. Selective Training Loss (WHY: 避免在 well-learned tokens 上浪费梯度)**

标准 DPO loss 计算整个序列的 log-ratio;FPO 引入 indicator function 仅在 error tokens 上计算 [Eq.8]:

$$L_{\text{FPO}} = -\mathbb{E}\left[\sum_i I(y^i) \cdot \log \sigma\left(\beta \log \frac{\pi_\theta(y_w^i|x)}{\pi_{\text{ref}}(y_w^i|x)} - \beta \log \frac{\pi_\theta(y_l^i|x)}{\pi_{\text{ref}}(y_l^i|x)}\right)\right]$$

[论文原文] 这样做的效果是: "fit a token-level implicit reward to optimize the TTS model using preference data while avoiding computational waste on well-trained tokens" [§III-D]。

[agent 解读] 与 NLP 中 Mask-DPO 的区别: Mask-DPO 在 sentence-level 做 masking (基于 factuality),FPO 在 token-level 做 masking (基于 TTS-specific error types)。FPO 的 masking 粒度更细且错误分类更贴合 TTS 场景。

### 训练策略

- Backbone: CosyVoice, CosyVoice2, Llasa (单码本 codec + AR LM)
- 训练数据: LibriTTS + AISHELL-3 (已在 base model 训练集中,排除数据增量影响)
- 采样: 100 speakers pool, 多次推理构建 candidate set, 选出 1000 utterances
- 超参: lr=2e-6, batch_size=64, AdamW, 仅 2 epochs [§IV-A]
- 评估: Seed-TTS eval dataset (EN + ZH + hard set) [§IV-B]

## 实验

| 指标 | FPO (CosyVoice) | Base CosyVoice | RIO | SDPO | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| CER (%) test-zh | **1.83** (-49.6%) | 3.63 | 2.21 (-39.1%) | 2.69 (-25.9%) | SEED-TTS-Eval | [Table I] |
| WER (%) test-en | **2.77** (-35.4%) | 4.29 | 2.89 (-32.6%) | 3.41 (-20.5%) | SEED-TTS-Eval | [Table I] |
| WER (%) test-hard | **7.98** (-32.1%) | 11.75 | 9.12 (-22.4%) | 9.84 (-16.3%) | SEED-TTS-Eval | [Table I] |
| Bad case (%) | **9%** | 21% | 12% | 17% | 综合 | [Table I] |
| NMOS | **3.83**+-0.09 | 3.64+-0.11 | 3.76+-0.08 | 3.68+-0.09 | 主观 | [Table II] |
| SECS | 0.728 | 0.723 | 0.744 | 0.731 | test-zh | [Table I] |
| F0 Corr. | **0.78** | 0.71 | 0.73 | 0.70 | 综合 | [Table I] |

| 指标 | FPO (CosyVoice2) | Base CosyVoice2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER (%) test-zh | **1.32** (-9.0%) | 1.45 | SEED-TTS-Eval | [Table I] |
| WER (%) test-en | **2.24** (-12.8%) | 2.57 | SEED-TTS-Eval | [Table I] |
| Bad case (%) | **8%** | 14% | 综合 | [Table I] |
| NMOS | **3.91**+-0.08 | 3.81+-0.07 | 主观 | [Table II] |

| 指标 | FPO (Llasa) | Base Llasa | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER (%) | **1.47** (-22.2%) | 1.89 | test-zh | [Table I] |
| WER (%) | **2.46** (-23.6%) | 3.22 | test-en | [Table I] |
| Bad case (%) | **11%** | 25% | 综合 | [Table I] |
| NMOS | **3.93**+-0.10 | 3.74+-0.12 | 主观 | [Table II] |

### 数据效率

FPO 仅用 200 utterances 即达 UNO 需 600 utterances 的 WER 性能 (3x 数据效率); bad case ratio 上 FPO 达到约 4x 加速 [Fig 4]。

在更大数据量对比中 (Table III), FPO 用 1000 utterances 达到 SDPO 用 5000 utterances 的 bad case ratio (8% vs 7%) [Table III]。当 FPO 也使用 5000 utterances 时,bad case ratio 降至 3% [Table III]。

### SECS 和 UTMOS 的 trade-off

FPO 在 SECS 和 UTMOS 上仅有 marginal improvement,RIO 和 UNO 在这些指标上更优。论文解释: FPO 靶向 segmental errors (局部修复),而 RIO/UNO 优化全局属性 (speaker similarity/overall quality); base model 在 speaker similarity 上已经很强,改进空间有限 [§V-A]。

[agent 解读] 这是 FPO 设计的必然 trade-off: 选择性 loss 只关注 error segments,自然不会改善 non-error segments 的属性 (如 speaker similarity)。如果需要同时优化全局和局部属性,可能需要将 FPO 与 utterance-level 方法组合使用。

## 局限性

1. **标注成本**: Fine-grained annotation 虽然数据量少,但每条样本的标注复杂度高于 utterance-level,需要时间戳级别的 error segment 定位 [§VI]
2. **仅验证 AR 模型**: 论文仅在 autoregressive TTS (CosyVoice 系列 + Llasa) 上验证; 对连续表征的 NAR/diffusion-based TTS (如 F5-TTS, MaskGCT) 的适用性存在挑战,因为构建 diverse paired samples 更困难 [§VI]
3. **Speaker similarity 无提升**: FPO 的 SECS 与 baseline 基本持平,而 RIO/UNO 有明显提升 [Table I]
4. **缺乏 iterative 训练**: FPO 仅训练 2 epochs,未探索多轮迭代的 self-improvement (对比 SpeechAlign 的 iterative SFT + DPO)
5. **未与 DiffRO/GRPO 对比**: 论文发表时间早于 CosyVoice 3 的 DiffRO,未能与 token-level 可微优化方法直接对比

## 点评

FPO 的核心贡献在于将 "fine-grained alignment" 这一在 NLP 中已有探索 (Mask-DPO, Rho-1) 的思路系统性地适配到 TTS 场景,并找到了 TTS-specific 的错误分类 (temporal vs semantic-phonetic) 和对应的 annotation 策略。这一分类体现了对 TTS 生成错误模式的深刻理解。

3-4x 的数据效率优势在实际部署中极具价值: 语音偏好数据的标注成本远高于文本,减少数据需求直接降低成本。

然而,FPO 在概念上的新颖性有限: 核心公式 (Eq.8) 本质上就是 DPO + masking,技术门槛不高。其真正的贡献在于 "如何定义 mask" (error type classification + annotation pipeline) 而非 "如何使用 mask"。此外,与 CosyVoice 3 的 DiffRO 相比,FPO 仍需显式的偏好数据标注,而 DiffRO 通过可微 reward model 实现了 end-to-end 优化。

## 可复用的 idea

1. **TTS segmental error taxonomy**: temporal modeling errors (局部) vs semantic-phonetic alignment errors (级联) 的二分法,可用于任何 TTS 系统的错误分析和诊断
2. **差异化 annotation 策略**: 局部错误只标局部,级联错误标到结尾 -- 这个思路可迁移到其他序列生成任务 (如 music generation, video generation)
3. **Selective loss with indicator function**: 在任何 DPO/KTO/RLHF 框架中加入 token-level masking,只需一个 indicator function 即可实现 fine-grained optimization,实现成本极低
4. **多指标综合评分自动选 preference pairs**: Eq.2 的加权评分方法可替代人工标注,减少标注成本

---

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes
> - [medium] frontmatter models 缺少 Llasa -> 已修正
> - [low] SECS 数据仅列 test-zh (可接受,正文已说明 trade-off)
> - [low] 可复用 idea #4 非 FPO 独创 (已知,保留作参考)
> 详见 `_review/FPO-review.yml`

检索命中: [[LLM-based TTS]], [[Semantic vs Acoustic Tokens]], [[CosyVoice]], [[CosyVoice 2]], [[TTS Evaluation]] | 过滤: [[Differentiable Reward Optimization]](pending-review) | 未命中但可能相关: 无
