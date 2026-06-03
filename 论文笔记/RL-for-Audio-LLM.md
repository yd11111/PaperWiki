---
type: paper
tier: deep
title: "Explore the Reinforcement Learning for the LLM based ASR and TTS system"
arxiv_id: "2509.18569"
source: "https://arxiv.org/abs/2509.18569"
authors: [Changfeng Gao, Yabin Li, Keyu An, Zhifu Gao, Zhihao Du, Han Zhao, Xiangang Li]
year: 2025
venue: "arXiv"
tags: [reinforcement-learning, GRPO, DiffRO, ASR, TTS, reward-design, audio-LLM, CosyVoice, FunAudio-ASR]
concepts: ["[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Residual Vector Quantization]]", "[[TTS Evaluation]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[LLM-based TTS]], [[Speech Tokenizer]], [[Residual Vector Quantization]], [[TTS Evaluation]])
> LLM-based TTS 使用 decoder-only transformer 生成离散 speech tokens,CosyVoice 系列采用 semantic tokens + CFM 解码。Differentiable Reward Optimization (DiffRO) [待确认] 是 CosyVoice 3 提出的 token-level 可微 RL 方法。TTS Evaluation 中 WER 和 SIM 是标准客观指标,但 WER 作为 reward 可能导致韵律坍缩。
> 检索命中: [[LLM-based TTS]], [[Speech Tokenizer]], [[Residual Vector Quantization]], [[TTS Evaluation]] | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Codec Language Model]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 设计轻量级 RL 训练框架统一 audio LLM 的 ASR 和 TTS 后训练,对比 GRPO 与 DiffRO 并提出 Combined+Filter 组合方案 [论文原文]
> - **路线**: Audio Encoder → SGLang LLM (rollout) → FSDP LLM (policy update); ASR: 规则 reward (WER+Hallucination+Keyword); TTS: GRPO + DiffRO + Sample Filter
> - **指标**: ASR: WER 9.71 (短) / 6.03 (长), 相对改进 5.3% [Table 1]; TTS: WER 3.414 (zh), SS 77.11 (zh) (Combined+Filter R1,2,3) [Table 2]
> - **可借鉴**: (1) GRPO 与 DiffRO 各有优劣,sample filter 使两者兼容; (2) 规则式 reward 设计 (hallucination detection + keyword) 对 ASR 有效; (3) GPU 交替调度框架 (SGLang rollout + FSDP policy) 提升 RL 训练效率
> - **局限**: (1) TTS 实验仅在 CosyVoice2-0.5B 上; (2) 未报 MOS 主观评估; (3) 组合方法 (GRPO+DiffRO) 直接合并反而变差,需 sample filter 才有效,机制未深入分析

## 核心问题

1. **Audio LLM 的 RL 框架**: 与纯文本 LLM 不同,audio LLM 有 audio encoder/decoder 等前后端模块,如何高效管理 GPU 资源做 RL 训练? [§3]
2. **ASR reward 设计**: 如何设计既能降低 WER 又能抑制 hallucination 的 reward function? [§4.1]
3. **TTS 的 GRPO vs DiffRO**: 两种 RL 方法各有什么优劣?能否组合? [§5]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

提出一个统一 RL 框架 [Fig 1],通过 GPU 交替调度 (alternate GPU utilization) 实现高效训练:

**ASR 分支** (左侧):
1. Audio Encoder (PyTorch) 提取 audio embeddings
2. SGLang LLM rollout engine 生成 G 组文本 hypotheses
3. 规则 reward model 对每组打分
4. FSDP LLM 执行 GRPO 策略更新

**TTS 分支** (右侧):
1. SGLang LLM 生成 acoustic token sequences
2. Tokens 送入 PyTorch Flow-matching model + vocoder 生成 Mel/waveform
3. 可选: DiffRO reward model (基于 ASR 的文本后验概率) 或 规则 reward model
4. FSDP LLM 执行 GRPO 和/或 DiffRO 策略更新

**训练效率** [§6.1]: 在 8xA100 上,ASR 一步约 54.6s (RTF 0.015); TTS 瓶颈在 FM decoder (16.73s per step, batch 128) [Fig 2] [论文原文]

### 关键设计选择

**1. ASR Reward Functions** [§4.1]

| Reward | 定义 | 目的 |
|--------|------|------|
| R^1: ASR Accuracy | 1 - WER(y*, y) | 降低 WER [论文原文] |
| R^2: Hallucination Detection | 检测重复/不存在/翻译内容 → reward=-1 | 抑制 hallucination [论文原文] |
| R^3: Keyword Accuracy & Recall | keyword 的 precision/recall 均值 | 关键词准确性 [论文原文] |

**2. ASR 训练数据构建** [§4.2]
- D^1: hard case samples (不同 ASR 系统输出不一致 + 长重复句) [论文原文]
- D^2: 限制音频长度 < 20s [论文原文]
- D^3: keyword-heavy 样本 (品牌名、专有名词) [论文原文]
- D^0: random control set [论文原文]

**3. TTS Reward Functions** [§5.1]

| Reward | 定义 | 操作层面 |
|--------|------|---------|
| R^1: ASR Accuracy | ASR 对合成音频的识别准确率 | waveform [论文原文] |
| R^2: Audio Duration | -abs((|o_i| - T_m) / T_m), 惩罚偏离中位长度 | token/waveform [论文原文] |
| R^3: Token & Pitch Diversity | edit distance (token diversity) + std(F0) (pitch diversity) | token + waveform [论文原文] |

**4. Combined GRPO + DiffRO with Sample Filter** [§5.2]
- 关键发现: 直接组合 GRPO 和 DiffRO loss 反而变差 [Table 2: Combined R^1, WER 3.782/SS 77.04 vs DiffRO R^1, WER 3.418/SS 77.00] [论文原文]
- 原因: 高温采样时某些 response 包含大量重复 token 无法预测 stop token,直接对这些 bad case 计算 DiffRO loss 有害 [论文原文]
- **Sample filter**: 按 advantage A_i 划分 positive (A_i > 0) 和 negative (A_i <= 0) 样本,仅对 positive 样本计算 DiffRO loss [论文原文]
- Filter 后: WER 3.381 → 3.330 → **3.414** (R^{1,2,3}), SS 77.37 → 77.11 [Table 2] [论文原文]

### 训练策略

- ASR: FunAudio-ASR (0.7B encoder + Qwen2.5-7B LLM), batch 32, group size 12, lr 1e-5, KL coeff β=0.1, 1 天完成 [§6.2.1]
- TTS: CosyVoice2-0.5B, CommonVoice zh+en 文本数据, batch 16, group size 8, temperature 1.0, KL coeff β=0.1 [§6.3.1]
- 评估: CV3-Eval test set [§6.3.1]

## 实验

| 指标 | Combined+Filter R^{1,2,3} | Baseline (no RL) | DiffRO R^1 | GRPO R^1 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| TTS WER (zh) | 3.414 | 4.280 | 3.418 | 3.710 | CV3-Eval | [Table 2] |
| TTS SS (zh) | 77.11 | 77.64 | 77.00 | 77.26 | CV3-Eval | [Table 2] |
| TTS WER (en) | 5.579 | 6.074 | 5.690 | 5.974 | CV3-Eval | [Table 2] |
| ASR WER (short) | 9.71 | 10.25 | - | - | in-house | [Table 1] |
| ASR WER (long) | 6.03 | 6.35 | - | - | in-house | [Table 1] |

**关键发现**:
1. **RL 对 ASR 有效**: 相对 WER 改进 5.3%,且 training data 和 reward 设计至关重要 [Table 1] [论文原文]
2. **DiffRO 降 WER 更强但伤 SS**: DiffRO R^1 WER 3.418 < GRPO R^1 WER 3.710,但 SS 也更低 (77.00 vs 77.26) [Table 2] [论文原文]
3. **Sample filter 关键**: 过滤 bad case 后 Combined 方案比单独使用 GRPO 或 DiffRO 更好 [论文原文]
4. **DiffRO 训练更稳定**: Fig 3 显示 DiffRO 在超过 1500 步后仍保持稳定,而 GRPO 和 Combined (无 filter) 迅速退化 [论文原文]
5. **Duration reward 有效**: R^2 (audio duration) 防止语速过慢,R^3 (diversity) 改善主观体验但不改善 WER/SS [论文原文]

## 局限性

1. TTS 实验仅在 CosyVoice2-0.5B 单一模型上验证 [agent 解读]
2. 未报 MOS 主观评估指标,仅有 WER 和 Speaker Similarity [agent 解读]
3. Sample filter 的阈值选择 (A_i > 0 vs A_i <= 0) 似乎是启发式的,未做 sensitivity analysis [agent 解读]
4. ASR 实验使用内部工业测试集,不易复现 [agent 解读]
5. Hallucination detection 依赖规则 (重复/不存在/翻译),可能遗漏其他 hallucination 类型 [agent 解读]

## 点评

本文是 **通义团队对 audio LLM RL 后训练的系统性探索**,最大贡献在于:

1. **统一 RL 框架**: 首次提出可同时处理 ASR 和 TTS 的 audio LLM RL 框架,GPU 交替调度设计实用
2. **GRPO vs DiffRO 对比**: 首次在同一框架下公平对比两种方法,揭示了 DiffRO 更稳定但可能伤 speaker similarity 的 trade-off
3. **Sample filter 的工程价值**: 虽然简单,但解决了 GRPO+DiffRO 直接组合的实际问题

**与 KB 已有知识的关系**: 本文的 DiffRO 实现延续了 CosyVoice 3 的方案 (Gumbel-Softmax + ASR reward),但首次将其与 GRPO 在同一框架下对比。R^2/R^3 reward 设计 (duration + diversity) 是对纯 WER reward 的有益补充,与 [[TTS Evaluation]] 中关于 WER 作为 reward 的局限性讨论一致。

## 可复用的 idea

1. **GPU 交替调度**: SGLang rollout + FSDP policy update 交替使用 GPU,适用于各种 audio LLM RL 场景
2. **Rule-based reward 组合**: Accuracy + Hallucination + Keyword 的三元 reward 设计可推广到其他 ASR RL 场景
3. **Sample filter for combined RL**: 对 positive samples 做 DiffRO、对全部做 GRPO,解决两种 loss 的兼容性问题
4. **Duration reward**: 简单有效地防止 RL 训练中语速变慢的问题
