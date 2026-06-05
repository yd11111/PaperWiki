---
type: paper
tier: deep
title: "WAND: Windowed Attention and Knowledge Distillation for Efficient Autoregressive Text-to-Speech Models"
arxiv_id: "2604.08558"
source: "Sources/WAND.pdf"
authors: [Hanna Lee, Tan Dat Nguyen, Jaehoon Kang, Kyuhong Shim]
year: 2026
venue: "arXiv"
tags: [TTS, efficient-inference, sliding-window-attention, knowledge-distillation, KV-cache, autoregressive, LLM-based, curriculum-learning]
concepts: ["[[LLM-basedTTS]]", "[[CodecLanguageModel]]", "[[FiniteScalarQuantization]]", "[[Single-codebookvsMulti-codebook]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[LibriTTS]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页: [[LLM-basedTTS]], [[模型库/CosyVoice2|CosyVoice 2]], [[SpeechLanguageModel]], [[CodecLanguageModel]], [[FiniteScalarQuantization]], [[Single-codebookvsMulti-codebook]])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: WAND 属于 LLM-based TTS 范式中"推理效率优化"方向。LLM-based TTS 的核心局限之一就是"高计算成本: 长序列自回归推理慢"(概念页明确列出)。当前解决路径包括: (1) 线性注意力/Mamba 替代(从头训练,质量有 gap); (2) 模型压缩/剪枝(不解决注意力本身); (3) speculative decoding(加速但不省内存); (4) KV caching(接近线性推理但内存持续增长)。WAND 开辟了第五条路: 在保留预训练权重的前提下,通过注意力分区+蒸馏将内存/计算开销从线性降到常数。

**已有认知**: CosyVoice 2 使用 Qwen2.5-0.5B backbone + FSQ 单码本 25Hz,已经是高效架构(GQA)的代表。IndexTTS 使用传统 MHA,SparkTTS 使用 BiCodec + 50Hz 更高 token rate。这三个系统覆盖了不同 codec 设计和注意力机制,构成了对 WAND 通用性的良好测试床。

**创新判断**: 知识库中尚无专门关于"注意力窗口限制"用于 TTS 推理优化的概念页。[[模型库/CosyVoice2|CosyVoice 2]] 被大量工作用作 baseline(已有 30+ 引用记录),但均围绕质量/可控性,尚无工作从注意力效率角度优化其推理。WAND 是首个在 pretrained AR-TTS 上验证 sliding-window + KD 的工作。

> 检索命中: [[LLM-basedTTS]]✓, [[模型库/CosyVoice2|CosyVoice 2]]✓, [[SpeechLanguageModel]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[FiniteScalarQuantization]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过将 AR-TTS 的注意力分为 global (conditioning) + local sliding-window (generated tokens),配合 KD 和 curriculum learning,实现常数内存/计算推理,且几乎无质量损失
> - **路线**: Pretrained AR-TTS → 注意力分区 (global + local window W) → curriculum scheduling (W_start→W) → KD (L_CE + λL_KL) from full-attention teacher → 1 epoch fine-tuning on 100h
> - **指标**: KV cache -66.2% (IndexTTS), GFLOPs -46.9%, speedup 1.51-1.89x, WER 持平或改善 (CosyVoice 2: 1.94→1.72%), 跨语言 CER 退化 <0.1% [Table 1, Table 3]
> - **可借鉴**: (1) global/local 注意力分区策略可推广到所有 prefix-conditioned AR 生成模型; (2) 注意力质量分布分析方法(量化 prompt vs generated 的注意力占比)可用于验证窗口大小选择; (3) curriculum + soft mask 渐进训练策略
> - **局限**: 仅验证 ≤0.5B 模型 + 10s 生成; window size 需根据 token rate 手动选择; 未在 >1B 模型或分钟级生成上验证; 无开源代码

## 核心问题

1. **AR-TTS 模型是否真的需要全序列注意力来维持高保真合成?** 论文假设不需要: conditioning tokens 提供全局语义/声学上下文,generated tokens 只需局部时序连贯性 [§1]。
2. **如何在不从头训练的前提下将注意力复杂度从 O(L) 降到 O(1)?** 通过 global-local 注意力分区 + KD + curriculum fine-tuning [§2]。
3. **这种优化是否跨架构、跨语言通用?** 在三种不同架构/codec/token rate 的模型上验证,仅用英语数据训练却保留中文能力 [§4.3]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

WAND 不修改模型架构本身,而是修改注意力 mask 和 KV cache 策略 [§2, Fig 1]:

```
Input: [System Prompt | Text | Ref Audio | Generated Tokens y_1:T]
       ←——— Conditioning Prefix ———→  ←— Generated Region —→
       
Attention:
  - Global (红色): 所有 query 位置都能 attend 到 conditioning prefix (固定)
  - Local (绿色): generated token 只能 attend 到最近 W 个 generated tokens (滑动窗口)
  
KV Cache:
  - 固定部分: conditioning prefix 的 KV (大小不变)
  - 滚动部分: 最近 W 个 generated tokens 的 KV (大小固定)
  → 总 KV cache = O(1),不随生成长度增长
```

[论文原文] 将 AR formulation 从 p(y_t | y_{<t}, s, x, a_pr) 近似为 p(y_t | y_{t-W:t-1}, s, x, a_pr) [§2.1, Eq. 2],理由是"声学信号局部连贯且单调,远距离过去 token 的影响在全局条件固定后消失" [§2.1]。

### 关键设计选择

**1. Global vs Local 的分割依据** [§4.4, Table 2]

[论文原文] 通过分析 vanilla full-attention 模型的注意力分布验证了这一分割:
- CosyVoice 2: 58.5% 注意力在 prompt,41.5% 在 generated;generated 中 70.2% 集中在 W=32 窗口内 → 总覆盖 87.6%
- IndexTTS 1.5: 64.6% prompt + 35.4%×57.1% local → 覆盖 84.8%
- SparkTTS: 47.9% prompt + 52.1%×82.8% local → 覆盖 91.0%

[agent 解读] 这与 text LLM 中的"attention sink"现象一致 [23]: 大部分注意力集中在少数 prefix token 和局部窗口。TTS 的 prefix = conditioning prompt,天然是注意力锚点。SparkTTS 分配更多注意力给 decode region (52.1%),对应其更高的 token rate (50 Hz vs 25 Hz),因此需要更宽的窗口 (W=64)。

**2. Knowledge Distillation 的两个互补目标** [§2.2, Eq. 3]

- L_CE: 交叉熵,锚定 student 到 ground-truth token → 确保基本对齐
- L_KL: Skew KL divergence,让 student 的 token 分布模仿 full-attention teacher → 补偿丢失的远距离上下文

L = L_CE + λ·L_KL

[论文原文] 消融显示两者互补: 仅 SW (无蒸馏) CosyVoice 2 WER 退化到 3.40%,加 L_CE 降到 2.02%,加 L_KL 降到 2.37%,组合后达 1.72% (优于 baseline 1.94%) [Table 4]。

[agent 解读] L_CE 提供 hard target supervision,L_KL 提供 soft target supervision。这类似于经典 Hinton KD 的设计: hard label 保证下界,soft label 传递 teacher 的 dark knowledge (分布细节)。在 TTS 场景中,L_KL 的价值在于传递 teacher 在远距离 token 上编码的上下文模式。

**3. Curriculum Scheduling** [§2.3, Eq. 4-5]

渐进式缩小窗口,避免突然截断造成的训练不稳定:
- 窗口大小: W(t) = W_start - α(t)·(W_start - W),α(t) 为 cosine schedule
- Soft mask: A' = A - τ(t)·M,τ(t) 从 τ_start 到 τ_end 做 log-scale interpolation
- 早期 τ 小,允许部分注意力穿透 mask → 保留梯度流
- 后期 τ 大,逼近推理时的 hard mask

[论文原文] 消融证实 curriculum 优于 direct training: CosyVoice 2 WER 1.86%→1.72%, IndexTTS 1.03%→0.91% [Table 5]。

### 训练策略

- 数据: LibriTTS train-clean-100 (100h,约为 baseline 训练数据的 1%)
- 优化器: AdamW, cosine LR schedule, peak LR 1e-5
- 训练量: 1 epoch
- 硬件: 单张 NVIDIA A100 MIG (20GB)
- 窗口: W_start=128,target W=32 (CosyVoice 2, IndexTTS) 或 64 (SparkTTS)

[agent 解读] 训练成本极低 — 100h 数据、1 epoch、单卡,这使得 WAND 实际可部署。关键在于它不是从头训练新架构,而是在预训练模型上做轻量级 adaptation。

## 实验

| 指标 | WAND (CosyVoice 2) | Baseline (CosyVoice 2) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (%) | 1.72 | 1.94 | Seed-TTS test-en | [Table 1] |
| UTMOS | 4.21 (+0.03) | 4.18 | Seed-TTS test-en | [Table 1] |
| NMOS | 4.02 +/- 0.16 | 4.01 +/- 0.17 | Seed-TTS test-en | [Table 1] |
| SSIM | 94.5 | 94.6 | Seed-TTS test-en | [Table 1] |
| KV Cache | 5.25 MB | 10.48 MB | 10s generation (fp32) | [Table 1] |
| GFLOPs | 7.44 | 11.55 | 10s generation | [Table 1] |
| Speedup | 1.55x | - | 10s generation | [Table 1] |
| CER (%) zh | 1.53 | 1.59 | Seed-TTS test-zh | [Table 3] |

| 指标 | WAND (IndexTTS 1.5) | Baseline (IndexTTS 1.5) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (%) | 0.91 | 0.98 | Seed-TTS test-en | [Table 1] |
| KV Cache | 13.01 MB | 38.44 MB | 10s generation (fp32) | [Table 1] |
| Cache Reduction | 66.2% | - | - | [Table 1] |
| Speedup | 1.89x | - | - | [Table 1] |
| CER (%) zh | 0.91 | 0.82 | Seed-TTS test-zh | [Table 3] |

| 指标 | WAND (SparkTTS) | Baseline (SparkTTS) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (%) | 3.11 | 3.27 | Seed-TTS test-en | [Table 1] |
| KV Cache | 7.15 MB | 18.09 MB | 10s generation (fp32) | [Table 1] |
| Cache Reduction | 60.5% | - | - | [Table 1] |
| Speedup | 1.51x | - | - | [Table 1] |
| CER (%) zh | 2.35 | 2.14 | Seed-TTS test-zh | [Table 3] |

**注意力分布分析** [Table 2]: 三个模型的 conditioning prefix 平均占 47.9-64.6% 的注意力质量,local window 内的 generated tokens 覆盖额外 20-43%,prefix + local window 联合覆盖 85-91% 的总注意力 → 证实了 global-local 分区的合理性。

**消融**: 
- KD 损失分解 [Table 4]: L_CE + L_KL 组合一致优于单独使用任一;无蒸馏的纯 SW 在 CosyVoice 2 上 WER 退化到 3.40%
- Curriculum [Table 5]: 渐进训练一致优于直接使用目标窗口

## 局限性

1. **规模限制**: 仅在 ≤0.5B 模型上验证,未覆盖更大模型 (如 1B+ 或 7B 级 LLM backbone)。更大模型的注意力分布模式可能不同。
2. **长度限制**: 评估仅覆盖 10s 生成,未验证分钟级长语音场景。虽然理论上 O(1) 应扩展良好,但 curriculum 是否需要更长窗口在长语音场景中值得检验。
3. **窗口大小选择**: W 的选择依赖于 token rate (25Hz→W=32, 50Hz→W=64),但缺乏系统的 W 与 token rate 关系分析,需要人工调参。
4. **跨语言退化**: 虽然声称 CER 退化 <0.1%,但 SparkTTS 的中文 CER 从 2.14% 退化到 2.35% (+0.21%),IndexTTS 从 0.82% 到 0.91% (+0.09%),后者接近但前者已超过 0.1% 界限 [Table 3]。
5. **未开源**: 无代码/模型公开,复现性受限。
6. **仅测试单码本/少码本系统**: 三个 baseline 均为单码本 (FSQ/VQ) 或 BiCodec,未覆盖 8-layer RVQ 等多码本 codec,其注意力模式可能不同。

## 点评

**核心贡献的价值**: WAND 抓住了 AR-TTS 的一个关键结构性冗余 — 大部分注意力其实集中在 conditioning prefix + 局部窗口,远距离 generated tokens 贡献的注意力 <15%。这个发现本身比具体方法更有价值,因为它指明了优化方向。

**方法的巧妙之处**: 不改架构、不从头训练、仅用 100h/1 epoch — 这让 WAND 成为一个极低成本的"即插即用"优化。与 MamTra (替换 Transformer 层为 Mamba) 或 RWKVTTS (替换整个 backbone) 等替代方案相比,WAND 的侵入性最小。

**与 LLM 领域的连接**: attention sink + sliding window 的思路在 text LLM 中已有广泛研究 (StreamingLLM, Mistral),但 WAND 是首次系统性地将其适配到 TTS,并通过 KD + curriculum 解决了 TTS 特有的质量敏感性问题。

**WER 改善的解释**: [论文原文] 将 WAND 有时优于 baseline 的 WER 归因于"sliding window 的隐式正则化效果 — 隔离解码上下文,抑制远距离采样伪影和幻觉的传播" [§4.1]。[agent 解读] 这个解释合理: AR 生成中一个 bad sample 可以通过 full attention 被后续所有 step 看到并放大,window 限制了这种 error propagation 的范围。

**不足**: 缺乏长语音和大模型的验证是最大遗憾。TTS 的效率问题在长语音 (audiobook, podcast) 场景中最痛,正是 O(1) 方法最能展示优势的场景,但论文仅评估了 10s。

## 可复用的 idea

1. **Global-Local 注意力分区策略**: 任何 prefix-conditioned AR 生成任务 (TTS, 音乐生成, 视频生成) 都可以采用相同思路 — 将 KV cache 分为固定 prefix 部分和滚动 generated 部分。关键前提是验证 prefix 注意力占比足够高。
2. **注意力质量分布分析**: Table 2 的分析方法 (分解 prompt vs generated 的注意力占比,计算 local window 覆盖率) 可作为通用的"window size 选择依据"。
3. **Curriculum + Soft Mask 渐进训练**: 任何需要将 soft constraint 转为 hard constraint 的训练场景 (如 quantization-aware training, pruning) 都可以借鉴 τ(t) 的 log-scale interpolation 策略,比 hard cutoff 更稳定。
4. **KD 作为 attention adaptation 的桥梁**: 当修改注意力模式时,用原始 full-attention 模型作为 teacher 进行蒸馏,可以在极少数据下恢复质量。这个策略可推广到注意力剪枝、稀疏注意力等场景。

## 审阅

(待独立审阅 agent 填写)
