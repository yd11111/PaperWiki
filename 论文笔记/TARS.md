---
type: paper
tier: deep
title: "TARS: Closing the Modality Reasoning Gap for Speech Large Language Models"
arxiv_id: "2601.05543"
source: "Sources/TARS.pdf"
authors: [Chaoren Wang, Heng Lu, Xueyao Zhang, Shujie Liu, Yan Lu, Jinyu Li, Zhizheng Wu]
year: 2026
venue: "ACL 2026"
tags: [speech-LM, reasoning, reinforcement-learning, modality-alignment, GRPO, trajectory-alignment, representation-drift]
concepts: ["[[SpeechLanguageModel]]", "[[Speech-TextAlignment]]", "[[ModalityAdaptationforSpeechLLM]]", "[[AudioUnderstanding]]"]
models: ["Qwen2.5-Omni", "Phi-4-MM"]
tasks: ["speech-reasoning", "multiple-choice-QA"]
datasets: ["VoiceBench-MMSU", "VoiceBench-OBQA", "UnifiedQA", "LibriSpeech", "SD-QA"]
kb_context_sources: 1
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页: [[SpeechLanguageModel]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓ | 过滤: [[Speech-TextAlignment]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review), [[AudioUnderstanding]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review), [[SpokenDialogueEvaluation]](pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: KB 记录了 SpeechLM 的三阶段架构 (speech encoder + adapter + decoder-only LLM) 和从 GSLM→Moshi 的演进线。TARS 不改变架构,而是针对这一架构的固有缺陷 — 语音模态在 Transformer 层间发生 representational drift,导致推理能力显著低于文本模态 — 提出后训练解决方案。KB 中已记录 Post-alignment 阶段 (Align-SLM, SpeechAlign) 通过 DPO 对齐语义/声学,TARS 将这一阶段从 off-policy supervision 推进到 on-policy RL,用 dense alignment reward 替代 sparse binary reward。在 KB 分类中,TARS 属于 "Post-alignment" 训练阶段,但对齐目标从输出行为扩展到了中间层表征的全程轨迹。[agent 解读]

> [!summary] 速查
> - **一句话**: 提出 on-policy RL 框架,通过 layer-wise representation alignment + output-level behavior alignment 两个 dense reward 信号,在 GRPO 训练中将 Speech LLM 的语音推理轨迹对齐到文本推理轨迹,实现 7B 模型 SOTA 的跨模态推理恢复率 (MRR ~100%) [§1, §3]
> - **路线**: speech/text input → Speech LLM (frozen encoder + LoRA on LLM) → GRPO 生成 G=8 completions (4 speech + 4 text) → 非对称 reward: text 用 R_base, speech 用 R_base + R_rep + R_beh → modality-specific normalization → DAPO loss update [§3, Fig 1]
> - **指标**: Qwen2.5-Omni: avg audio 76.84%, MRR 98.89% (vs base 71.30%/91.76%); Phi-4-MM: avg audio 79.80%, MRR 100.45% (vs base 63.23%/79.59%); 超越 cascaded ASR+LLM baselines; WER 不变 (~4.2%) [Table 1, Table 2]
> - **可借鉴**: (1) 用文本模态的 on-policy completion 作为 moving reference 而非 static teacher,随训练共同进化 [§3.2]; (2) modality-specific normalization 解决混合模态 GRPO 中弱模态被持续压制的问题 [§3.3]; (3) 即使 accuracy reward 全零 (语音推理太难),alignment reward 仍提供有效梯度 [§3.3]; (4) layer-wise representation reward 的 middle layers (11-20) 最关键,但全层最优 [§5.2, Fig 3]
> - **局限**: 仅评估 7B 规模 [Limitations]; 仅单轮推理,未验证多轮对话 [Limitations]; 依赖 text-only reference,无法对齐文本中无对应的副语言信息 [Limitations]; 仅 TTS 合成训练数据,real speech 仅在评估中验证 [§4.1]

## 核心问题

TARS 要解决 Speech LLM 中的 **modality reasoning gap**: 同一模型处理语音输入时推理能力显著低于文本输入 [§1]。

1. **现象量化**: 通过 Modality Recovery Rate (MRR) 衡量。Qwen2.5-Omni base 的 MRR 仅 91.76% (MMSU 61.51% audio vs 67.94% text),意味着语音输入导致 ~8% 的推理能力丧失 [§3.1, Eq 1, Table 1]
2. **根因定位**: 论文将 gap 归因于两层:
   - **Representational drift**: 语音和文本的 hidden states 在 Transformer 层间逐渐发散,minor modality-specific differences 在深层累积放大 [§1, §5.1] [论文原文]
   - **Behavioral divergence**: 长链推理中,off-policy supervision 导致 exposure bias — 单个 token 预测错误即将模型推入未见过的状态,reasoning 彻底崩溃 [§1] [论文原文]
3. **已有方法的不足**: 
   - **Input-side fusion** (AlignChat, DeSTA, OTReg): 冻结 LLM backbone,仅优化 adapter/projector,只做 surface-level alignment,无法修正深层 representation drift [§1, §2] [论文原文]
   - **Output-side supervision** (KD, prompt-switching): 强制 token-level 模仿 text completion,但 speech-conditioned 分布本质不同于 text,forcing exact token matching 是 unreachable objective; 加上 exposure bias,一步错步步错 [§1] [论文原文]

**WHY not 已有 RL 方法**: 同期 SoundMind-RL 也用 RL 于 Speech LLM,但仅用 sparse rule-based reward (format + correctness)。当 speech 推理太难导致 accuracy 全零时,sparse reward 无法提供学习信号。TARS 的 dense alignment reward 即使在 accuracy=0 时仍有效 [§2, §3.3] [论文原文]

## 方法: TARS (Trajectory Alignment for Reasoning in Speech)

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体框架 [§3, Fig 1]

TARS 在现有 Speech LLM 上施加后训练,不修改架构。核心思想: 用 text-conditioned reasoning trajectory 作为 reference,通过 RL 引导 speech-conditioned trajectory 向其对齐 [§3] [论文原文]。

训练流程:
1. 对每个 prompt q,同时输入 speech 版本 q_speech 和 text 版本 q_text [§3.1]
2. 用当前策略 π_θ 生成 G=8 个 completions (4 speech + 4 text) [§3.3]
3. 对 text completions 仅计算 R_base; 对 speech completions 计算 R_total = R_base + α·R_rep + β·R_beh [§3.2, Eq 2]
4. 用 modality-specific normalization 计算 advantage [§3.3, Eq 7]
5. 用 DAPO loss 更新 LoRA 参数 [§3.3]

**关键设计: asymmetric reward** — text 和 speech 的 reward 不对称。Text branch 只需任务准确率,speech branch 额外需要向 text 对齐。并且 text reference 来自当前策略 (非冻结 teacher),随训练同步进化 [§3.2] [论文原文]。

### Reward 设计 [§3.2]

#### 1. Base Reward R_base [§3.2, Eq 3]

```
R_base = R_acc + λ · R_fmt
```

- R_acc ∈ {0,1}: 通过 xFinder 提取答案,判断是否匹配 ground truth [§3.2]
- R_fmt ∈ {0,1}: 正则表达式匹配 `<think>...</think><answer>...The answer is [ABCD]...</answer>` 格式 [Appendix A]
- λ = 0.5 [§3.2]

#### 2. Representation Alignment Reward R_rep [§3.2, Eq 4-5]

核心: 度量 speech 和 text completion 在每一层的 hidden state 相似度。

**步骤**:
1. 对第 l 层的 hidden states H^(l) ∈ R^{T×d},mean-pool 生成部分 (排除 prompt 前 n 个 token) 为 h̄^(l) [§3.2, Eq 4]
2. 对每个 speech completion,从同 group 随机选一个 correct text completion (R_acc=1) 作为 reference [§3.2]
3. 计算跨 L 层的平均 cosine similarity [§3.2, Eq 5]:
```
R_rep = (1/L) Σ_{l=1}^{L} CosSim(h̄^(l)_speech, h̄^(l)_text)
```

**WHY mean-pool 而非 token-level**: speech 和 text 的序列长度不同 (语音 prompt 更长),mean-pool 消除长度差异,提取段级表征 [§3.2] [论文原文]

**WHY 选择 correct text completion 作为 reference**: 只选 R_acc=1 的 text completion,确保 speech 对齐的目标是正确推理的表征轨迹,而非错误的 [§3.2] [论文原文]

**Fallback**: 若 group 中无 correct text completion,设 R_rep=0。此情况出现频率仅 5.9%-10.7% [§3.2]

#### 3. Behavior Alignment Reward R_beh [§3.2, Eq 6]

核心: 度量 speech 和 text completion 最终输出的语义一致性。

```
R_beh = CosSim(E(y_speech), E(y*_text))
```

- E: 外部 embedding model (Qwen3-Embedding-0.6B) [§3.2]
- y*_text: 从 group 中选取的 correct text completion [§3.2]

**WHY 用 embedding similarity 而非 token-level matching**: token-level KL/CE 强制精确复制 text 输出,但 speech-conditioned model 的分布本质上不同,应允许语义等价但措辞不同的输出 [§1, §3.2] [论文原文]

**两个 reward 的互补性**: R_rep 是 coarse-grained representation-level 的,提供 dense 信号 (每层都有梯度); R_beh 是 token-level 但 sparser 的 (仅在最终输出上计算)。前者对齐内部推理路径,后者约束最终行为 [§3.2] [论文原文]

### Modality-Specific Normalization [§3.3, Eq 7]

**问题**: GRPO 按整个 group 归一化 reward 计算 advantage。但 text completions 的 R_base 天然高于 speech,导致 speech completions 的 advantage 持续为负,学习被压制 [§3.3] [论文原文]

**解决**: 分模态归一化 —

```
Â_{i,m} = r_{i,m} - μ_m,  m ∈ {speech, text}
```

μ_m 是模态 m 内的 reward 均值。每个模态相对自己的 baseline 优化,而非与另一模态比较 [§3.3, Eq 7]

**WHY 不是简单分组训练**: speech 和 text 仍在同一 group 中生成,共享 text reference。分的是 normalization,不是 generation [§3.3] [agent 解读]

### 训练细节 [§4.1, Appendix C]

- **数据**: UnifiedQA auxiliary_train (来自 MMLU repository),9,953 samples (203 hours) [§4.1]
- **语音合成**: CosyVoice2 + openaudio-s1-mini,参考说话人从 Emilia-YODAS EN 子集采样 [§4.1]
- **质量过滤**: whisper-medium 10% WER 阈值 [§4.1]
- **微调方式**: LoRA (rank=8, alpha=32) on all linear layers,冻结 audio encoder 和 projector [§4.1, Table 5]
- **RL 参数**: GRPO G=8, temperature=1.0, max completion length=1024, DAPO loss, α=1.0, β=1.0 [Table 5]
- **硬件**: 4×A100 或 8×H200, ~55h (Qwen2.5-Omni), ~35h (Phi-4-MM) [§4.1]
- **框架**: ms-swift (Qwen2.5-Omni), HuggingFace TRL (Phi-4-MM) [§4.1]

## 实验

### 主要结果 [§4.2, Table 1]

| 指标 | TARS (Qwen) | Base Qwen | TARS (Phi) | Base Phi | AlignChat | KD | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Audio Acc (avg) | 76.84% | 71.30% | 79.80% | 63.23% | 77.57% | 72.87% | MMSU+OBQA | Table 1 |
| Text Acc (avg) | 78.56% | 76.17% | 83.82% | 78.39% | 77.70% | 76.89% | MMSU+OBQA | Table 1 |
| MRR | 98.89% | 91.76% | 100.45% | 79.59% | 99.83% | 93.78% | MMSU+OBQA | Table 1 |
| MMSU Audio | 67.96% | 61.51% | 70.14% | 54.81% | 69.65% | 63.09% | MMSU | Table 1 |
| OBQA Audio | 85.71% | 81.09% | 89.45% | 71.65% | 85.49% | 82.64% | OBQA | Table 1 |

**关键发现**:

1. **Phi-4-MM MRR=100.45%**: 语音准确率 (79.80%) 甚至超过原始文本准确率 (78.39%),完全消除了 modality gap [Table 1] [论文原文]
2. **超越 cascaded pipeline**: TARS on Qwen (76.84%) 和 Phi (79.80%) 均超过对应的 ASR+LLM pipeline (75.55% 和 73.40%),表明端到端处理可避免 ASR 引入的推理错误 [§4.2] [论文原文]
3. **文本性能不降反升**: Qwen text 76.17%→78.56%, Phi text 78.39%→83.82%。语音对齐学到的知识反向增强了文本推理 [§4.2] [论文原文]
4. **ASR 能力不受影响**: WER 维持 ~4.16%-4.24%,说明收益来自推理对齐而非更好的语音识别 [Table 2] [论文原文]

### 训练策略消融 [§4.3, Table 2]

在 Phi-4-MM 上对比不同后训练策略:

| 方法 | Avg Audio | MRR | WER | 出处 |
| --- | --- | --- | --- | --- |
| Base (no CoT) | 63.16% | 79.60% | 4.16% | Table 2 |
| + CoT prompting | 70.06% | 88.29% | - | Table 2 |
| SFT | 72.52% | 91.37% | 4.18% | Table 2 |
| DPO | 75.37% | 94.98% | 4.23% | Table 2 |
| Standard GRPO (R_base only) | 73.17% | 92.21% | 4.24% | Table 2 |
| + R_rep | 75.83% | 95.56% | 4.18% | Table 2 |
| + R_beh | 78.73% | 99.22% | 4.20% | Table 2 |
| TARS (both) | 79.57% | 100.28% | 4.20% | Table 2 |

**WHY DPO > Standard GRPO**: Standard GRPO 仅用 sparse binary reward (accuracy),在语音推理经常全错的情况下缺乏学习信号; DPO 的偏好信号相对更稳定 [§4.3] [agent 解读]

**WHY TARS > DPO**: DPO 是 off-policy 的静态偏好优化,不能动态探索和修正推理轨迹; TARS 的 on-policy exploration 允许模型在训练中试错和自我纠正 [§1, §2] [论文原文]

### Reward 组件消融 [§4.4, Table 2]

从 Standard GRPO 逐步添加 reward 组件:

- **+R_rep**: MRR 92.21%→95.56%, layer-wise hidden state alignment 提供了比 R_base 更 dense 的信号 [§4.4] [论文原文]
- **+R_beh**: MRR 92.21%→99.22%, behavior alignment 接近完全恢复,语义一致性约束对输出行为的引导更直接 [§4.4] [论文原文]
- **Both**: MRR 100.28%, 两者互补 — R_rep 减少内部表征漂移,R_beh 约束外部输出一致性 [§4.4] [论文原文]

### Qwen2.5-Omni 消融 [§5.3, Table 3]

重要发现: 对高度优化的 Qwen2.5-Omni,R_rep 反而有害 (MRR 99.23%→98.39%)。论文将此归因为 **过度正则化**: 强迫语音表征在每层都模仿文本表征,剥夺了模型处理模态特有特征的自由度 [§5.3] [论文原文]。

**规律**: 训练程度低的模型 (Phi-4-MM, 仅 SFT) 对显式 layer-wise guidance 更敏感; 高度优化的模型 (Qwen2.5-Omni) 只需 R_beh 即足够 (100.01%) [§5.3] [论文原文]。这与知识蒸馏文献中 "过度严格的模仿在容量受限模型上反而有害" 的发现一致 [§5.3] [论文原文]。

### Layer-wise 分析 [§5.1, §5.2]

**层间对齐分析 (Fig 2)**: cosine similarity 随层深度自然下降 (modality-specific differences 累积)。关键是方法间的相对差距:
- R_base only: 最低 similarity
- +R_rep: 全程提升
- +R_rep+R_beh: 最高, behavior alignment 作为互补约束引导 speech branch 走语义一致的推理路径 [§5.1] [论文原文]

**层深度敏感性 (Fig 3)**: 将 32 层分为 Shallow/Middle/Deep/Last 四组:
- Middle (layers 11-20): 75.48%, 最关键的局部区域
- Shallow (1-10): 72.62%, Deep (21-30): 73.31%, Last (31-32): 72.34%
- All layers: 75.83%, 全层最优 [§5.2, Fig 3]

**解读**: representation drift 主要发生在中间推理阶段,而非早期感知或最终 logit 对齐阶段。这暗示语音和文本的感知编码差异在浅层尚可容忍,在中层的推理变换中才被放大 [§5.2] [论文原文]

### 泛化验证 [§5.5, Table 4]

在真实语音和副语言任务上验证:
- **SD-QA** (真实口语 QA, 多口音): TARS 41.95% vs base 39.42% [Table 4]
- **MMSU** (副语言感知): TARS 57.40% vs base 56.70%。CoT prompting 反而降到 51.50% [Table 4]

**重要**: CoT prompting 伤害副语言感知,但 TARS 不会。说明 TARS 的轨迹对齐保留了语音特有属性 (prosody, accent),而非用文本模式覆盖 [§5.5] [论文原文]

### 计算效率 [§5.7, Table 8]

TARS 相比 Standard GRPO 仅增加 4.4% 训练时间开销 (28.7h vs 27.5h on 8×H200)。对比 SFT/DPO 需要额外 10.9h+ 的 offline 数据生成,TARS 完全在线 [Table 8]

## 局限性

1. **规模受限**: 仅在 7B 模型验证,更小/更大模型的 reward 设计可能需要调整 [Limitations]
2. **单轮推理**: 未验证多轮对话或交互式推理场景 [Limitations]
3. **Text-only reference 的固有瓶颈**: alignment 依赖 text completion 作为 reference,无法对齐文本中无对应物的副语言信息 (emotion, prosody, intent) [Limitations]
4. **合成训练数据**: 训练语音全部由 TTS 合成,可能存在 domain shift; 真实语音仅在评估时验证 [§4.1]
5. **评估范围有限**: 仅 multiple-choice QA (MMSU, OBQA),未验证开放式生成推理 [agent 解读]

## 核心设计选择分析

### WHY on-policy RL 而非 off-policy distillation [§1, §2]

Off-policy 方法 (KD, prompt-switching) 强迫 speech model 在每个 token 上模仿 text teacher 的分布。但 speech-conditioned 分布与 text-conditioned 分布本质不同 (paralinguistic cues, encoder noise 等),forcing token-level matching 是 unreachable objective。更严重的是 exposure bias: 一旦某步预测偏离 teacher 轨迹,模型进入未训练区域,reasoning 彻底崩溃。On-policy RL 让模型自己探索,只要最终对齐即可,允许路径上的合理偏差 [§1] [论文原文]

### WHY 用 current policy 而非 frozen teacher [§3.2]

Text completion 来自当前策略 π_θ (非冻结的 teacher model)。好处: text branch 也在 R_base 下持续改进,speech 对齐的目标是 "越来越强的 text reasoning",实现 co-evolution [§3.2] [论文原文]。

### WHY α=1.0, β=1.0 [§3.2]

论文未做 α/β 的超参搜索,直接设 1.0。消融显示 R_rep 和 R_beh 各自贡献清晰且互补 (Table 2),但最优比例可能因模型而异 (Qwen 上 R_rep 有害) [§5.3] [agent 解读]

## 点评

**核心贡献的价值**: TARS 的核心贡献不在于 "Speech LLM + RL" (SoundMind-RL 已做),而在于 **dense alignment reward 的设计** — 将 text modality 作为 "内部参考轨迹" 嵌入 RL reward,解决了 speech reasoning 中 reward 过于 sparse 的根本问题。这一思路可泛化到任何 "弱模态对齐到强模态" 的场景 [agent 解读]。

**Modality-specific normalization 的巧妙**: 这是一个看似简单但实际关键的工程贡献。在混合模态 GRPO 中,如果不分组归一化,弱模态 (speech) 的 advantage 永远为负,等于没有训练。这个问题在任何多模态 RL 场景中都会出现 [agent 解读]。

**Stage-dependent regularization 的发现值得关注**: R_rep 对 Phi-4-MM 有效但对 Qwen2.5-Omni 有害 (Table 3),揭示了后训练 reward 设计需要考虑 base model 的优化程度。这暗示 "one-size-fits-all" 的 alignment reward 不存在,需要根据 model maturity 调整 [agent 解读]。

**实验设计的局限**: 评估仅限 multiple-choice QA (MMSU, OBQA)。这类任务有明确的正确答案和 xFinder 可提取的格式,但开放式推理 (如 "解释为什么...") 的 modality gap 可能表现不同,且 behavior alignment 的 embedding similarity 在开放式场景中是否有效存疑 [agent 解读]。

## 可复用的 idea

1. **Dense alignment reward 模式**: 当目标模态 (speech) 的 task reward 极度 sparse 时,用强模态 (text) 的 intermediate representation 和 final output 作为 continuous reference signal。可迁移到 video-text、image-text 等跨模态 RL 场景 [§3.2]
2. **Modality-specific normalization**: 多模态 RL 中按模态分组计算 advantage,防止强模态压制弱模态学习。直接可用于任何混合输入的 GRPO/PPO 训练 [§3.3]
3. **Moving reference (co-evolving teacher)**: 用当前策略的 text output 而非 frozen teacher,避免 static target 的 capacity gap 问题。尤其适用于 teacher 和 student 是同一模型不同模态的场景 [§3.2]
4. **Layer sensitivity 诊断**: 在 RL 对齐前,先分层组检查 representation drift 在哪个深度最严重,以此指导 reward 设计。对于已高度优化的模型,可能只需 output-level alignment [§5.2, §5.3]

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节详细解释了 WHY on-policy 而非 off-policy、WHY moving reference、WHY modality-specific normalization |
> | 可信赖 | pass | 数字 claim 均标注至 Table/Section,指标方向正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率 >80% |
> | 可定位 | pass | KB 背景定位到 Post-alignment 阶段,与 Align-SLM/SpeechAlign 对比 |
> | 不污染 | pass | 无新建概念页需求,现有概念覆盖充分 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/TARS-review.yml`
