---
type: paper
tier: deep
title: "FlowTTS-GRPO: Online Reinforcement Learning with Multi-Objective Reward Optimization for Flow-Matching Based Text-to-Speech"
arxiv_id: "2606.23190"
source: "Sources/FlowTTS-GRPO.pdf"
authors: [Haoxu Wang, Biao Tian, Weiqing Li, Xiang Lv, Han Zhao, Xiangang Li]
year: 2026
venue: "Interspeech 2026"
tags: [TTS, reinforcement-learning, flow-matching, GRPO, post-training, zero-shot, speaker-similarity]
concepts: ["[[ConditionalFlowMatching]]", "[[DifferentiableRewardOptimization]]", "[[Classifier-FreeGuidance]]", "[[LLM-basedTTS]]"]
models: ["[[CosyVoice3]]", "[[CosyVoice2]]", "[[CosyVoice]]", "F5-TTS", "[[F5R-TTS]]", "[[MaskGCT]]", "[[Seed-TTS]]", "[[Llasa]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[CV3-Eval]]", "WenetSpeech4TTS", "LibriTTS-960"]
kb_context_sources: 6
status: draft
created: 2026-06-23
updated: 2026-06-23
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 1 个待确认页: [[ConditionalFlowMatching]], [[LLM-basedTTS]], [[SEED-TTS-Eval]], [[CV3-Eval]], [[Classifier-FreeGuidance]] + [[DifferentiableRewardOptimization]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文处于 TTS RL post-training 演进线的 FM 侧。已有工作主要集中在 LLM 侧的 RL: DiffRO (CosyVoice 3) 在 token 空间做可微优化, Multi-Reward GRPO (Tencent) 在 audio-level 用 5 维 reward 优化 AR LM, GRPO-TTS (USTC/iFLYTEK) 用 CER+NLL composite reward 微调 LLM。FM 侧的 RL 仅有 F5R-TTS (Tencent, 2025) 一项前驱工作,通过训练独立的 Gaussian FM generator 引入随机性。本文通过 ODE→SDE 转换直接在现有 FM 模型上做 GRPO,消除了对独立生成器的依赖。
>
> **已有认知**: CFM 概念页记录了 flow matching 的 ODE 路径特性(确定性推理);DifferentiableRewardOptimization 页详细记录了 GRPO vs DiffRO 对比(Tongyi RL-for-Audio-LLM 结论: DiffRO 降 WER 更强但可能伤 speaker similarity, GRPO 超 1500 步迅速退化);CosyVoice 3 模型页记录了 LLM+FM hybrid 架构和 DiffRO post-training。
>
> **创新判断**: 相比 F5R-TTS 的"训练独立 Gaussian FM generator + 单 rollout + 单 reward",本文的核心差异在于 (1) 通过 SDE 转换直接复用开源 FM 模型, (2) 多 rollout + 多目标 reward, (3) 多项工程优化(去 CFG、hard case、window training)。相比 DiffRO 只作用于 LLM token 空间,本文作用于 FM 的连续声学空间,两者优化对象互补。
>
> 检索命中: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SEED-TTS-Eval]]✓, [[CV3-Eval]]✓, [[Classifier-FreeGuidance]]✓ | 参考(待确认): [[DifferentiableRewardOptimization]] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 Flow-GRPO (ODE→SDE 转换 + GRPO) 首次引入 FM-based TTS,直接微调开源 FM 模型提升说话人相似度和感知质量
> - **路线**: 文本+prompt → (LLM/text encoder → speech tokens) → FM 模型 (ODE→SDE 采样, GRPO 优化) → vocoder → waveform; reward 来自 ERes2Net (SS) + ASR (WER/CER) + DNSMOS (MOS)
> - **指标**: CV3 SS2 0.830→0.859 (+0.029), SS1 0.777→0.804 (超越闭源 Seed-TTS, 首次 SOTA) [Table 2, test-zh]; F5-TTS SS2 0.796→0.827, CER 1.81→1.55 [Table 2, test-zh]
> - **可借鉴**: (1) ODE→SDE 转换公式(Eq.6-7)可直接复用于任何 CFM 模型做 RL; (2) 加权组合 + std 归一化的多目标 reward 设计(Eq.12); (3) 训练时去掉 CFG 加速 RL 收敛的实用发现
> - **局限**: F5-TTS 仅训练 1289 步(受计算资源限制); FM-GRPO 对 LLM+FM hybrid 系统的 WER 改进有限(CER 从 1.20→1.26 略升); 未与 DiffRO/DPO 等其他 RL 方法做同条件对比

## 核心问题

FlowTTS-GRPO 要解决的核心问题是: **如何在 flow-matching (FM) based TTS 模型上应用强化学习 post-training?**

FM 使用确定性 ODE 推理,缺乏 RL 所需的随机性(策略概率无法定义)。已有 TTS RL 工作要么只针对 LLM 组件(DiffRO, GRPO-TTS),要么需要训练独立的随机生成器(F5R-TTS)。本文通过将 ODE 转换为等价的 SDE,在保持边际分布不变的前提下引入 GRPO 所需的随机性,实现了对开源 FM 模型的直接 RL 微调。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

FlowTTS-GRPO 是一个 post-training 框架,在预训练 FM 模型之上叠加 GRPO 优化。Pipeline (Fig.1):

1. **推理采样**: 给定 prompt waveform + target text,TTS 模型生成 G 组候选语音
2. **Reward 计算**: 对每组输出计算 SS (ERes2Net cosine similarity) + ASR (1-CER/WER) + MOS (P.835 DNSMOS)
3. **Advantage 估计**: 组内相对优势 (Eq.9),标准化 reward 后加权组合
4. **策略更新**: GRPO loss (Eq.10) 对 FM velocity field 做梯度更新

应用于两个模型: CosyVoice 3.0 (LLM+FM hybrid,仅优化 FM) 和 F5-TTS (FM-only,优化整个 FM)。

### 关键设计选择

**1. ODE→SDE 转换 (核心创新)**

FM 的标准推理是确定性 ODE: dxt = vt dt [论文原文]。GRPO 需要策略概率密度,但确定性推理下策略退化为 Dirac delta,无法探索 [论文原文, §2.2.1]。

借鉴 Flow-GRPO (图像生成, [27]) 和 FlowSE-GRPO (语音增强, [28]),本文将 ODE 转换为反向 SDE (Eq.4-5),引入噪声项 σt·dw。最终更新公式 (Eq.7):

```
xt,mean = xt + vθ(xt,t) + σt²/(2(1-t)) · (-xt + t·vθ(xt,t)) · Δt
xt+Δt = xt,mean + σt·√Δt · ε
```

其中 σt = a·√((1-t)/t), a 为噪声超参。转换后的 SDE 保持与原 ODE 相同的边际分布 [论文原文, 引用 [27]]。这使得策略概率 pθ(xt+Δt|xt,c) 可按高斯密度计算 (Eq.11),GRPO 的 importance ratio rti(θ) 可以正常定义 [论文原文, §2.2.3]。

**为什么不用 F5R-TTS 的方案?** F5R-TTS 需要独立训练一个 Gaussian FM generator 来产生随机性,无法直接利用开源模型;且只用单 rollout (rollout=1),无法利用组内优势差异和多目标优化 [论文原文, §1]。

**2. Window Training**

仅对 SDE 的早期步骤子集 t ∈ [Smin, Smin+ws] 做 SDE 采样和优化,其余步骤用确定性 ODE [论文原文, §2.2.2]。CV3 使用 2-step window (denoising step [5,8] 中的 Smin ∈ [1,3]); F5-TTS 使用 2-step window (denoising step [8,16] 中的 Smin ∈ [1,3]) [§3.3]。

[agent 解读] 这减少了需要计算策略概率的步数,降低训练成本;同时早期步骤的噪声更大,探索效果更显著。

**3. 多目标 Reward 组合**

论文比较了两种策略 [§2.4]:

- **概率组合** (Probabilistic, [36]): 每个 prompt 概率性地分配给单一 reward 函数,组内 advantage 由单一 reward 驱动。优点是避免 reward 尺度差异;缺点是早期振荡,收敛慢 [论文原文, Fig.5]。

- **加权组合** (Weighted, Eq.12): 先按当前 batch 的 std 归一化每个 reward (使 std=1),再加权求和: R = λ1·RSS/std(RSS) + λ2·RASR/std(RASR) + λ3·RMOS/std(RMOS)。std 归一化的动机: 不同 reward 的 std 差异大 (Fig.2 显示 MOS std 均值=0.199 >> SS std=0.084 >> ASR std=0.057),直接加权无法实现预期权重 [论文原文, §2.4.2]。

最终选择加权组合, λ1=λ2=1.0, λ3=0.4 [§4.4.3]。λ3=0.4 的选择理由: SS 对零样本 TTS 最关键,过高的 MOS 权重会拖慢 SS 增长 [论文原文, Fig.6]。

**4. Hard Case Synthesis**

通过三种启发式文本增强构造困难训练样本 [§2.5, Table 1]:
- LWR (Local Word Repetition): 随机重复某个词 3-5 次
- SMR (Sparse Multi-word Repetition): 2-3 个词各重复 2-3 次
- GSR (Global Sentence Repetition): 整句重复

[agent 解读] 这借鉴了 Seed-TTS-Eval hard-zh 中的失败模式,相当于在 RL 训练中做 curriculum learning,让模型在困难样本上建立鲁棒性。

### 训练策略

- **LoRA 微调**: 两个模型均使用 LoRA rank=32, alpha=64,可训练参数约 10M (~2.8% of FM) [§3.3]
- **训练数据**: WenetSpeech4TTS Premium (中文) + LibriTTS-960 (英文), 各 20K 样本 (train-easy-40k); 额外 20K 中文困难样本 (train-hard-20k) [§3.1]
- **CV3 特殊处理**: 先用 0.5B LM (不做 RL) 预解码训练集,提取 prompt/generated tokens 作为 FM 的输入条件 [§3.3]。仅微调 FM,LM/tokenizer/vocoder 冻结
- **训练时去 CFG**: 两个模型训练采样时不使用 CFG,推理时仍使用 CFG [§3.3, §4.4.4]
- **噪声水平**: a=0.5 (a 太小探索不足,a 太大训练不稳定) [Fig.8]

## 实验

### 主要结果 (Seed-TTS-Eval)

| 指标 | 模型 | 无 RL | 有 FM-GRPO | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| SS2 (ERes2Net) | CV3 | 0.830 | 0.859 (+0.029) | test-zh | [Table 2] |
| SS1 (WavLM) | CV3 | 0.777 | 0.804 (+0.027) | test-zh | [Table 2] |
| P835 DNSMOS | CV3 | 3.353 | 3.536 (+0.183) | test-zh | [Table 2] |
| CER | CV3 | 1.20 | 1.26 (+0.06) | test-zh | [Table 2] |
| SS2 | F5-TTS | 0.796 | 0.827 (+0.031) | test-zh | [Table 2] |
| SS1 | F5-TTS | 0.760 | 0.777 (+0.017) | test-zh | [Table 2] |
| P835 DNSMOS | F5-TTS | 3.313 | 3.514 (+0.201) | test-zh | [Table 2] |
| CER | F5-TTS | 1.81 | 1.55 (-0.26) | test-zh | [Table 2] |
| WER | F5-TTS | 1.88 | 1.73 (-0.15) | test-en | [Table 2] |

**关键发现**:
- CV3 FM-GRPO 的 SS1 在 test-zh 上达到 0.804,首次超越闭源 Seed-TTS (0.796) [§4.2.1]
- SS1 不是 proxy reward (未用于训练优化),其同步提升表明 RL 未 overfit 到 ERes2Net reward model [论文原文, §4.2.1]
- CV3 CER 略升 (1.20→1.26),因为在 LLM+FM hybrid 中 intelligibility 主要受 LLM 控制,FM-RL 难以改善 [论文原文, §4.2.1]
- F5-TTS 作为 FM-only 系统,CER 同时改善,因为 FM 同时控制 intelligibility 和 acoustic details [论文原文, §4.2.2]

### 跨语言泛化 (CV3-Eval)

仅在中英文数据上训练的 FM-GRPO,在日/韩/德/法/俄/意/西等语言上也展现 SS 和 P808 提升 [Table 3]。

### 消融实验

**多目标 Reward 组合**: 加权组合 vs 概率组合 vs 加权无 std 归一化 (Fig.5)
- 加权组合: SS2 和 P835 增长最快最稳定
- 概率组合: 早期振荡,最终收敛但慢
- 无 std 归一化: DNSMOS 增长过快 (因 std 大导致 advantage 贡献大),SS 增长受抑制

**DNSMOS 权重 λ3** (Fig.6): λ3=0.2 时 DNSMOS 基本不涨; λ3=1.0 时 DNSMOS 涨快但 SS 涨慢; λ3=0.4 取平衡

**CFG 对 RL 训练的影响** (Fig.7): 训练时不用 CFG 的 SS proxy reward 增长明显快于用 CFG 的
[论文原文] 去 CFG 增加探索多样性,加速 RL 收敛; 优化条件速度场仍能在 CFG 推理下有效 [§4.4.4]

**噪声水平** (Fig.8): a=0.1 几乎无增长; a=0.5 达到训练饱和点; 选择 a=0.5

**Hard Case** (Fig.4):
- CV3 加入 hard case 对 ASR reward 几乎无效 (intelligibility 由 LM 控制)
- F5-TTS 加入 hard case 显著加速 ASR reward 增长

### 主观评估

A/B preference test (30 样本, 10 评审员) [Fig.9, Fig.10]:
- F5-TTS-RL vs F5-TTS: Overall MOS 65.4% prefer RL, Timbre Similarity 64.2% prefer RL (英文)
- CV3-FM-RL vs CV3: Overall MOS 56.4% prefer RL, Timbre Similarity 54.0% prefer RL (英文)
- 中文结果趋势一致但 margin 更小 (CV3: MOS 53.0%, Timbre 52.3%)

### Functional Decoupling 发现

在 LLM+FM hybrid 系统中 [论文原文, §4.2.1]:
- **LM 上做 RL** → 主要改善 intelligibility (WER/CER 下降)
- **FM 上做 RL** → 主要改善 audio details (SS, DNSMOS 提升)
- 两者功能解耦,FM-GRPO 模型可泛化到 LM-RL 变体 (CV3-LM-RL 上 FM-GRPO 仍有效) [Table 2]

## 局限性

1. **F5-TTS 训练不充分**: 仅 1289 步 (资源受限),CV3 训练了 9545 步; F5-TTS 的潜力可能未充分发掘 [§4.2.2]
2. **CV3 CER 略升**: FM-GRPO 对 LLM+FM hybrid 的 intelligibility 无正面作用,CER 从 1.20 微增至 1.26 [Table 2]
3. **缺少与其他 FM-RL 方法的同条件对比**: 仅引用了 F5R-TTS 的数字但未在相同设置下重跑 [Table 2 引用原文数字]
4. **缺少与 DiffRO 的组合分析**: 论文作者来自同一团队 (Tongyi Lab),已有 RL-for-Audio-LLM 做过 GRPO vs DiffRO 对比,但本文未探讨 FM-GRPO + DiffRO 的联合效果
5. **主观评估规模较小**: 30 样本 × 10 评审员,统计功效有限

## 点评

**优势**: 本文的核心价值在于为 FM-based TTS 的 RL post-training 提供了一个简洁有效的框架。ODE→SDE 转换虽非本文原创 (来自 Flow-GRPO [27]),但将其成功适配到 TTS 并发现了多个实用优化(去 CFG、hard case、std 归一化)是实质性贡献。"RL on FM 改善 audio details, RL on LM 改善 intelligibility"的 functional decoupling 发现对后续工作有指导意义。

**与 F5R-TTS 的对比**: F5R-TTS 需要训练独立 Gaussian FM generator 且仅用单 rollout/单 reward,本文消除了这些限制。但两者未在完全相同条件下对比 (不同训练数据/步数/评估集子集)。

**与 DiffRO 的互补性**: 本文作用于 FM 的连续声学空间,DiffRO 作用于 LM 的离散 token 空间。两者优化对象不同,理论上可叠加。RL-for-Audio-LLM 工作已初步验证 GRPO+DiffRO 组合的可行性(需 sample filter),但本文未进一步探讨。

**工程贡献**: LoRA rank=32 仅需 ~10M 可训练参数即可有效优化 ~360M FM 模型,训练成本低 (8 GPU);去 CFG 训练加速收敛是一个反直觉但有实验支撑的发现。

## 可复用的 idea

1. **ODE→SDE 转换公式 (Eq.6-7)**: 任何基于 flow matching 的模型 (TTS/vocoder/speech enhancement/image) 都可用此公式引入 GRPO 训练,核心只需实现 SDE 采样和高斯对数概率计算
2. **多目标 reward 的 std 归一化 (Eq.12)**: 当不同 reward 的 std 差异大时 (本文 MOS std ≈ 3.5× SS std),先归一化再加权,否则高方差 reward 会主导 advantage
3. **训练时去 CFG**: FM-based 模型做 RL 时去掉 CFG 增加探索,推理时恢复 CFG,对条件速度场的优化仍然有效
4. **Functional decoupling**: LLM+FM hybrid 系统中,分别对 LM 和 FM 做 targeted RL 可以更高效地改善不同维度的指标
5. **Hard case synthesis 策略** (LWR/SMR/GSR): 简单的文本重复增强即可构造有效的困难训练样本,对 FM-only 系统的 intelligibility 改善尤为有效

> [!review] 审阅结论: pass-with-fixes (2026-06-23)
> reproducible: 8 | trustworthy: 8 | distinguishable: 9 | locatable: 9 | no_pollution: 8
> issues: 3 (1 medium, 2 low) — 详见 `_review/FlowTTS-GRPO-review.yml`
> medium fix applied: frontmatter models 字段已补充 wikilink
