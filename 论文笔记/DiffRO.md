---
type: paper
tier: deep
title: "Differentiable Reward Optimization for LLM based TTS system"
arxiv_id: "2507.05911"
source: "Sources/2507.05911.pdf"
authors: [Changfeng Gao, Zhihao Du, Shiliang Zhang]
year: 2025
venue: "arXiv (Interspeech format)"
tags: [reinforcement-learning, DiffRO, TTS, reward-model, Gumbel-Softmax, RLHF, multi-task-reward, emotion-control, CosyVoice]
concepts: ["[[Differentiable Reward Optimization]]", "[[Gumbel-Softmax]]", "[[Emotion Control in TTS]]", "[[TTS Evaluation]]", "[[Speech Language Model]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]", "[[模型库/SenseVoice|SenseVoice]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> **谱系定位**: 本文是 DiffRO 方法的原始独立论文 (2025-07),出自阿里通义实验室 Speech Team。[[Differentiable Reward Optimization]] [待确认] 概念页已记录了该方法在 CosyVoice 3 中的集成及其与 GRPO 的后续对比 (RL-for-Audio-LLM, 同一第一作者)。本文聚焦于 DiffRO 的核心机制提出和独立验证。
> **已有认知**: [[Speech Language Model]] (confirmed) 定义了 neural codec LM TTS 的标准四组件架构 (tokenizer + LM + FM + vocoder); [[模型库/CosyVoice 2|CosyVoice 2]] (confirmed) 是本文的 baseline 系统; [[Gumbel-Softmax]] [待确认] 是 DiffRO 实现可微采样的核心技术; [[Emotion Control in TTS]] [待确认] 已记录 Emo-DPO 等情感控制路线; [[TTS Evaluation]] [待确认] 讨论了 WER 作为 reward 的局限性。
> **创新判断**: 对比已有 RL for TTS 方法 (Seed-TTS 的 audio-level REINFORCE, Emo-DPO 的 preference-based),DiffRO 的核心新意在于: (1) 在 token 空间而非 audio 空间计算 reward; (2) 通过 Gumbel-Softmax 实现端到端可微优化,无需 PPO/DPO 的 RL loop; (3) 首次提出 Multi-Task Reward (MTR) 模型统一多维度反馈。
> 检索命中: [[Speech Language Model]]✓, [[模型库/CosyVoice 2|CosyVoice 2]]✓ | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Emotion Control in TTS]](pending-review), [[TTS Evaluation]](pending-review), [[Gumbel-Softmax]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 codec token 空间直接计算可微 reward 并反向传播梯度优化 TTS LM,绕过 FM+vocoder 的计算瓶颈,同时用 Multi-Task Reward 模型实现多维度音频属性控制
> - **路线**: Text → LLM(codec token prediction, Gumbel-Softmax 采样) → Token2Reward Model(ASR/SER/SQA/AED) → reward loss → 反向传播更新 LM
> - **指标**: WER 0.78%(zh, Seed-TTS-eval, 基于 CosyVoice2.0 base) [Table 2]; 情感 accuracy HAPPY 1.00(zh)/0.92(en), SAD 0.76/0.96, ANGRY 0.84/0.92 [Table 3]
> - **可借鉴**: (1) Token2Reward 范式: 用类 ASR 模型在 token 空间计算 reward,省去 FM+vocoder 推理; (2) Gumbel-Softmax 替换 argmax 实现端到端可微; (3) MTR 模型: 一个多任务 reward 模型同时提供 ASR/SER/SQA/AED 反馈,无需显式情感标签数据即可实现情感控制
> - **局限**: (1) MTR-based DiffRO 的 ASR WER 劣于 ASR-only DiffRO,多任务目标存在冲突 [Table 2]; (2) MOS/age/gender 控制效果受限于 FM+vocoder 的去噪能力,LM 层面的控制无法完全传递到最终音频 [Table 4]; (3) 仅在 CosyVoice 2.0 上验证,未测更多 backbone

## 核心问题

1. **TTS RLHF 的计算瓶颈**: 传统方法需通过 FM+vocoder 将 codec tokens 合成为音频后才能计算 reward,计算成本高,如何绕过? [§1]
2. **正负样本区分度不足**: TTS 生成的多次采样音频高度相似,难以构造有意义的偏好数据对; DPO 的 preferred/dispreferred 二分法对多维度 TTS 评估不够充分 [§2.2]
3. **多维度控制**: 如何用统一方法同时优化发音准确性、情感表达、音质等多个维度? [§3.2]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DiffRO 的核心思想是在 **codec token 空间** 直接计算 reward 并端到端反向传播梯度优化 LM [§3, Fig 1]。与 DPO 的两个关键区别:
1. **reward 计算跳过 FM+vocoder**: DPO 需要将 codec tokens 合成为 audio 才能评估质量; DiffRO 用一个 Token2Reward 模型直接从 tokens 预测 reward [论文原文]
2. **无需 RL training loop**: DPO 仍需构造偏好数据对; DiffRO 通过 Gumbel-Softmax 使整个 pipeline 可微,可直接用梯度下降优化 [论文原文]

### 关键设计选择

#### 1. Token2Reward Prediction [§3.1]

DiffRO 的第一个核心洞察: 如果 codec tokens 编码了语音的主要信息,那么 **可以直接从 tokens 预测 reward,不需要先合成音频** [论文原文]。

具体方式: 将 SenseVoice 的前端 CNN 模块替换为 embedding layer 以接受 speech token 输入,训练一个类 ASR 模型,输入 codec token 序列,输出文本后验概率 P_ASR(Y|U) [§3.1, Eq 7-8]。这个后验概率即 ASR reward: token 序列越能被正确转写回原文,说明 tokens 越好地保留了内容信息。

**为什么这样设计而非直接在 audio 上计算**: FM+vocoder 推理成本占 TTS pipeline 的主要计算量,跳过这一步可大幅降低 RL 数据生产成本 [论文原文, §1]。Table 1 验证了 token2reward 模型在 ASR/SER/SQA/Age&Gender 各任务上都有合理性能,证明 codec tokens 包含充足信息 [§4.1.2]。

#### 2. Gumbel-Softmax 可微采样 [§3.1]

DiffRO 的第二个核心洞察: LM 预测 token 时使用 argmax (Eq 6),这是不可微的。用 **Gumbel-Softmax 替代 argmax** 后,token 选择变为可微操作,reward 损失的梯度可以直接反向传播到 LM 参数 [论文原文, §3.1]。

[agent 解读] 这一设计将 RLHF 从 "采样→评估→策略更新" 的 RL loop 简化为标准的 "前向→loss→反向" 的监督学习式训练,训练稳定性和效率都优于 PPO/DPO。但 Gumbel-Softmax 是 argmax 的连续松弛,温度参数的选择会影响近似质量。

#### 3. Multi-Task Reward (MTR) Model [§3.2]

在 ASR reward 基础上,DiffRO 引入 **多任务 reward 模型** [§3.2, Fig 2],在一个模型中集成:
- **ASR**: 内容一致性 reward (发音准确性)
- **SER**: 情感识别 reward (情感表达)
- **SQA**: 语音质量评估 reward (MOS 预测)
- **AED**: 音频事件检测 reward (笑声/呼吸等)
- **Age & Gender**: 说话人属性预测

MTR 模型以 SenseVoice 架构为基础,前端 CNN 替换为 embedding layer,每个任务加一个 attention pooling layer [§4.1.2]。总 reward 为各任务的 log 后验概率之和 (Eq 9): R_MTR = sum_i log P_task_i(A_i|U_tilde) [§3.2]。

**为什么用多任务而非单任务**: 单一 ASR reward 只优化发音,不关注情感、音质等属性; MTR 模型通过多个 reward 信号同时提供多维度反馈,实现零样本属性控制 [论文原文, §3.2]。

#### 4. 情感控制的指令模板 [§4.3.1]

情感控制通过特定指令模板实现:
```
Your emotion is {E} <endofprompt> Y_{1:N} <s> U_{1:T} </s>
```
SER reward 为 log P(A_emo = E | U_tilde),同时保留 ASR reward [§4.3.1]。

[agent 解读] 这意味着情感知识完全来自 MTR 模型(其 SER 任务在 13000+ 小时伪标签数据上训练),而非显式情感标签数据。这是一种知识蒸馏式的零样本控制: reward 模型的知识通过梯度传递到 TTS LM。

### 训练策略

- **Baseline**: CosyVoice 2.0-0.5B [§4.1.1]
- **SFT**: 4000 音频样本, 5 speakers (4F+1M), Mandarin [§4.1.1]
- **RL data**: 10000 文本 (90% 中英, 10% 日韩) [§4.1.1]
- **Reward model 训练**: 13000+ 小时音频, 伪标签 (ASR/emotion/MOS/age&gender/event) [§4.1.2]
- **RL 设置**: beta=0.1, lr=1e-5, 4x A800 GPU [§4.1.3]
- **DPO 对比**: 每条文本合成 5 次,按 WER+speaker similarity 选正负样本 [§4.1.3]

## 实验

| 指标 | 本文 (DiffRO-ASR, base) | 本文 (DiffRO-ASR, SFT) | DPO (SFT) | Baseline (CosyVoice2.0) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER-zh (%) | 0.78 | 1.09 | 1.27 | 1.56 | Seed-TTS-eval | [Table 2] |
| WER-en (%) | 1.89 | 2.57 | 3.28 | 2.75 | Seed-TTS-eval | [Table 2] |
| WER-hard (%) | 5.58 | 5.83 | 6.74 | 6.91 | Seed-TTS-eval | [Table 2] |
| WER-ja (%) | 6.36 | 8.38 | 10.4 | 9.13 | CV3-Eval | [Table 2] |
| WER-ko (%) | 5.41 | 6.35 | 9.12 | 7.43 | CV3-Eval | [Table 2] |
| Emotion-HAPPY-zh | -- | 1.00 (MTR) | -- | 0.92 | CV3-Eval | [Table 3] |
| Emotion-SAD-en | -- | 0.96 (MTR) | -- | 0.84 | CV3-Eval | [Table 3] |
| Emotion-ANGRY-zh | -- | 0.84 (MTR) | -- | 0.76 | CV3-Eval | [Table 3] |

**关键发现**:

1. **DiffRO-ASR 全面优于 DPO**: 在所有 5 个测试集上 WER 均更低 [Table 2],且 DiffRO 可直接应用于 base model (无需 SFT),在 zh 和 hard 上达到 SOTA [论文原文, §4.2]
2. **跨语言迁移**: DiffRO-ASR 在 ja/ko 上甚至超过 base model,尽管 RL 数据中日韩文本很少; 作者解释为 TTS 通过 ASR reward model 学会生成多语言 codec tokens [论文原文, §4.2]
3. **MTR 与 ASR-only 的 trade-off**: DiffRO-MTR 在 ASR WER 上劣于 DiffRO-ASR,因为 MTR 鼓励 tokens 携带非 ASR 信息 (情感、事件等),但这些额外信息支撑了属性控制能力 [论文原文, §4.2]
4. **零样本情感控制**: DiffRO-MTR 在所有情感类别上大幅超越 CosyVoice 2.0/F5-TTS/GPT-SoVITS,且无需情感标注数据 [Table 3]。系统还学会了合成笑声、抽泣、呼吸等音频事件来传达情感 [Fig 3]
5. **MOS/属性控制的瓶颈**: codec token 层面的 MOS 可接近目标值 (MOSt=2→2.20),但最终音频 MOS 变化有限 (3.77 vs baseline 3.84),因为 FM+vocoder 在清洁音频上训练,具有去噪能力,会"纠正"LM 层面的质量降低 [论文原文, §4.3.2, Table 4]

## 局限性

1. **MTR 多目标冲突**: ASR-only DiffRO 的发音准确性优于 MTR DiffRO,说明多维度 reward 之间存在 trade-off,目前仅通过简单求和组合 [§4.2]
2. **控制边界受限于 pipeline 架构**: DiffRO 仅优化 LM,而 speaker 信息在 CosyVoice 2.0 的 FM 阶段集成,因此 age/gender/MOS 等属性控制效果有限; 需要将 DiffRO 扩展到 FM 模块才能突破 [§4.3.2, §5]
3. **Token2Reward 的信息损失**: Token2Reward 模型的 ASR WER (11.3%) 显著差于直接在 audio 上的 SenseVoice (8.67%),表明 codec tokens 丢失了部分信息 [Table 1]
4. **实验规模**: 仅在 CosyVoice 2.0 (0.5B) 上验证,未测试更大模型或其他 TTS backbone
5. **缺少主观评估**: 未报告 MOS/CMOS 等主观质量指标,仅用 ASR WER 和情感分类 accuracy

## 点评

本文是 DiffRO 方法的原始独立论文,核心贡献是将 TTS RLHF 从 audio-level 下沉到 token-level,通过 Gumbel-Softmax 实现端到端可微训练。这一设计巧妙地利用了 neural codec TTS 的 pipeline 特性: codec tokens 已经编码了语音的主要信息,无需每次都 decode 到 audio 才能评估。

与后续的 RL-for-Audio-LLM (同一第一作者, 2509.18569) 对比: 该后续工作在本文基础上加入 GRPO 对比、Sample Filter 组合方案、以及 ASR 方向的 RL,是更完整的系统性工作。本文的独特价值在于清晰阐述 DiffRO 的核心机制和 MTR 模型的设计,以及 token2reward 范式的验证。

一个值得关注的发现是 MOS 控制实验 [Table 4] 暴露的 "FM+vocoder 去噪瓶颈": LM 层面可以精确控制 codec token 的质量分布,但后端模块的去噪能力会"抵消"这种控制。这意味着 DiffRO 在控制 LM 直接决定的属性 (发音、情感) 上效果最好,而在 FM+vocoder 主导的属性 (音质、说话人) 上受限。

## 可复用的 idea

1. **Token2Reward 范式**: 在离散 token 空间建 reward model,绕过 decoder 推理。可推广到任何 "encoder → discrete tokens → decoder" 架构的系统 (不限于 TTS)
2. **Gumbel-Softmax 端到端 RL**: 将 RL 的 "采样-评估-更新" loop 简化为标准监督训练,降低工程复杂度和训练不稳定性
3. **多任务 Reward 蒸馏**: 一个 reward model 提供多维度反馈,实现零样本属性控制; 属性知识从已有的下游任务模型通过伪标签蒸馏到 reward model,再通过梯度传递到 TTS LM
4. **情感控制无需标注数据**: 只需 SER reward model (在通用数据上训练) + 指令模板,即可让 TTS 学会情感表达甚至合成笑声/呼吸等事件

---

> [!review] 审阅 (2026-06-04, agent)
> **结论: pass-with-fixes** | 3 issues (0 high, 1 medium, 2 low)
> - [medium] traceability-gap: SOTA claim 仅在 Seed-TTS-eval 上基于 CosyVoice 2.0 对比,已标注来源可接受
> - [low] template-compliance: datasets 字段为空,可选补充评估集
> - [low] traceability-gap: 训练超参数段落标注可更精确
> 详见 `_review/DiffRO-review.yml`

检索命中: [[Speech Language Model]]✓, [[模型库/CosyVoice 2|CosyVoice 2]]✓ | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Emotion Control in TTS]](pending-review), [[TTS Evaluation]](pending-review), [[Gumbel-Softmax]](pending-review) | 未命中但可能相关: 无
