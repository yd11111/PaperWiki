---
type: paper
tier: deep
title: "ADAPTERMIX: Exploring the Efficacy of Mixture of Adapters for Low-Resource TTS Adaptation"
arxiv_id: "2305.18028"
source: "Sources/ADAPTERMIX.pdf"
authors: [Ambuj Mehrish, Abhinav Ramesh Kashyap, Li Yingting, Navonil Majumder, Soujanya Poria]
year: 2023
venue: "Interspeech 2023"
tags: [TTS, speaker-adaptation, parameter-efficient, adapter, mixture-of-experts, low-resource, few-shot]
concepts: ["[[SpeakerAdaptation]]", "[[SpeakerEmbedding]]", "[[Non-autoregressiveTTS]]", "[[TTSEvaluation]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: ADAPTERMIX 属于 [[SpeakerAdaptation]] 的 "Parameter Efficiency" 子课题。SpeakerAdaptation 概念页 [待确认] 列出了参数高效方法的演进: CLN tuning (AdaSpeech) → Residual adapters (Morioka et al.) → Structured pruning (LightClone) → Hypernetwork (HyperTTS) → MoA (Mixture of Adapters)。ADAPTERMIX 正是 MoA 这一路线上较早的实践之一,但概念页中引用的 MoA 代表工作为 Fujita et al.,非本文。

**已有认知**:
- [[SpeakerEmbedding]] (confirmed) 详述了 speaker embedding 的两种范式 (lookup table vs encoder) 及多种注入方式。ADAPTERMIX 使用 speaker lookup table + 微调目标说话人 embedding 来 condition variance adapter,属于 "Speaker ID one-hot → Learnable speaker embedding" 阶段的方法。
- [[Non-autoregressiveTTS]] [待确认] 中 Transformer TTS (Li et al., 2019) 作为本文 backbone,是 NAR TTS 的早期代表,4 encoder + 6 decoder layers, d_model=256。
- [[SpeakerAdaptation]] [待确认] 的 AdaSpeech 系列 (CLN-based) 是本文主要对比参照系: AdaSpeech 通过 Conditional Layer Norm 实现参数高效适应,而 ADAPTERMIX 走 adapter 路线。
- [[VoiceCloningTaxonomy]] [待确认] 将本文归入 "Few-shot Voice Cloning → Parameter Efficiency" 分支。
- [[TTSEvaluation]] [待确认] 中 MOS + WER + Cosine Similarity 是本文使用的评估三件套,2023 年尚处于标准期,SpeakerEmbedding 页 (confirmed) 指出 cosine similarity 高度依赖 speaker encoder 选择。

**创新判断**: 相对于已有 KB 知识,ADAPTERMIX 的核心新增在于: (1) 将 MoE 的 expert choice routing 引入 TTS adapter; (2) 多个 adapter 跨说话人共享而非每人一套 (vs Hsieh/Morioka per-speaker adapter)。但从 2023 年至今,LLM-based TTS (in-context learning) 已大幅改变 speaker adaptation 格局,adapter-based 方法在当前范式中的直接适用性有限。

> 检索命中: [[SpeakerAdaptation]][待确认], [[SpeakerEmbedding]]✓, [[VoiceCloningTaxonomy]][待确认], [[Non-autoregressiveTTS]][待确认], [[TTSEvaluation]][待确认], [[ProsodyModeling]]✓ | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 Mixture of Experts 中的 expert choice routing 引入 TTS decoder adapter,实现跨说话人共享的参数高效 speaker adaptation (仅 11.62% 参数)
> - **路线**: Text → Transformer TTS Encoder → Duration/Pitch Predictor → Decoder (每层含 N 个并行 residual adapter + routing) → Mel Spectrogram → Vocoder
> - **指标**: 1min 数据 MOS 3.33 vs Fine-tune 3.45 vs Adapter 2.82; XAB speaker similarity 偏好 43.63% (10min); Cosine Sim 0.7324 (10min) 接近 Fine-tune 0.7362 [Table 1, Fig 3]
> - **可借鉴**: expert choice routing 让每个 adapter 自主选择 top-k token 处理,实现 token 级别的细粒度分工;adapter 跨说话人共享的设计比 per-speaker adapter 更可扩展
> - **局限**: 仅在 LibriTTS → VCTK 英语场景验证; backbone 3.6M 参数偏小; 与 CLN-based 方法 (AdaSpeech) 未直接对比; 无 ablation 解释各 adapter 学到了什么; 已开源但 2023 年后无后续

## 核心问题

1. **Few-shot speaker adaptation 中,单个 adapter 能否捕获说话人声音的多面特征?** 作者认为不行 -- 单个 adapter 只能学到一种维度的 speaker-specific 信息,多个并行 adapter 能捕获互补的 fine-grained 特征 (如 prosody, speaking rate, accent) [§1]。

2. **Per-speaker adapter (如 Hsieh et al.) 的扩展性瓶颈如何解决?** 为每个新说话人训练独立 adapter 导致参数量随说话人线性增长。ADAPTERMIX 让 N 个 adapter 跨说话人共享,通过 routing 机制让 adapter 按 token 内容自动选择,不再绑定到具体说话人 [§2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

两阶段训练:
1. **预训练阶段**: 在 LibriTTS train-clean-100 (100h, 251 speakers, 22.05kHz) 上训练标准 Transformer TTS backbone (4 encoder + 6 decoder layers, d_model=256, 含 post-net/pre-net/variance adapter),训练 900k steps [§4.2]
2. **适应阶段**: 冻结 backbone 全部参数 (含 BatchNorm),在 decoder 每层 feed-forward 之后插入 ADAPTERMIX 模块 (N 个 parallel residual adapters + routing),仅训练 adapter 参数 + 目标说话人 embedding [§3.2, §4.2]

### 关键设计选择

#### 1. Residual Adapter 结构 [§3.2]

每个 adapter 由以下组件构成:
```
ĥ_l = h_l + ReLU(LayerNorm(h_l) · W_down) · W_up
```
- `W_down ∈ R^{d_model × r}`: 降维到 bottleneck r=128
- `W_up ∈ R^{r × d_model}`: 升维回 d_model=256
- LayerNorm → 降维 → ReLU → 升维 → 残差连接

[agent 解读] 这是标准的 Houlsby adapter 变体,残差连接保证在 adapter 未充分训练时不破坏 backbone 输出。

#### 2. Expert Choice Routing [§3.2]

借鉴 Zhou et al. (2022) 的 expert choice routing:
- Token-to-adapter 亲和度矩阵: `S = Softmax(h_l · W_g)`, 其中 `W_g ∈ R^{d_model × N}` [Eq. 2]
- 每个 adapter 独立选择 top-k 个亲和度最高的 token: `G, I = TopK(S^T, k)` [Eq. 3]
- k 由动态公式决定: `k = n × c / N` (n=序列长度, c=capacity 超参, N=adapter 数) [§3.2]
- 一个 token 可被多个 adapter 处理,也可不被任何 adapter 处理

[agent 解读] Expert choice (adapter 选 token) 与 token choice (token 选 expert) 的关键差异: expert choice 保证每个 adapter 处理相同数量的 token,避免 load balancing 问题,但代价是某些 token 可能被忽略。论文未讨论 token drop 对 TTS 质量的影响。

#### 3. 输出聚合 [§3.2]

通过 Permutation matrix P = OneHot(I) 选取 token 表示,各 adapter 输出用 Einstein summation 聚合:
```
ĥ_l = h_l + Σ_i P · G · h^out_l[i]   [Eq. 5]
```
G 中包含的 softmax 门控值起到加权作用,[论文原文] "outputs of the adapters are combined" [§3.2]。

#### 4. Variance Adapter [§4.2]

目标说话人的 duration/pitch/energy 可能与预训练说话人不同。在 variance adapter 输出后额外加一个 r=64 的 residual adapter,同时微调目标说话人的 speaker embedding 来 condition variance adapter [§4.2]。

[agent 解读] 这是一个容易被忽略但重要的设计: 不仅 decoder 层有 adapter,variance predictor 也有独立 adapter,保证韵律也能适应新说话人。

### 训练策略

- 目标说话人: VCTK 中随机选 10 人 (5M+5F),按 1min/10min/15min 三种数据量分组 [§4.2]
- 所有模型训练 10k steps, Adam optimizer, warmup 4k steps, 学习率退火 (6k/7k/8k steps, rate=0.3) [§4.2]
- Batch size: 64 (10min/15min), 16 (1min) [§4.2]
- 训练设备: 单张 NVIDIA Tesla A6000 GPU [§4.2]
- Adapter 可训练参数: 11.62% of backbone; 单 Adapter baseline 仅 1.57% [Table 1]

## 实验

| 指标 | ADAPTERMIX | Finetune (100%) | Adapter (1.57%) | 数据量 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS↑ | 3.33 | 3.45 | 2.82 | 1min | [Table 1] |
| MOS↑ | 3.66 | 3.18 | 3.13 | 10min | [Table 1] |
| MOS↑ | 3.53 | 3.54 | 3.19 | 15min | [Table 1] |
| WER↓ | 0.2987 | 0.2489 | 0.3911 | 1min | [Table 1] |
| WER↓ | 0.2270 | 0.2228 | 0.2445 | 10min | [Table 1] |
| Cosine Sim↑ | 0.7037 | 0.7470 | 0.6733 | 1min | [Table 1] |
| Cosine Sim↑ | 0.7324 | 0.7362 | 0.7091 | 10min | [Table 1] |
| XAB Preference | 43.63% | - | 21.05% | 10min | [Fig 3] |
| XAB Preference | 43% | - | 31% | 1min | [Fig 3] |
| Avg Rank↓ | 2.18 | 2.60 | - | all | [Fig 2] |

**关键发现**:
1. **10min 场景 MOS 反超 Fine-tune**: ADAPTERMIX 3.66 vs Fine-tune 3.18 [Table 1]。[agent 解读] Fine-tune 在 10min 时 MOS 反而低于 1min (3.18 vs 3.45),这不太符合直觉,可能与 overfitting 或评估方差有关,论文未解释这一异常。
2. **Speaker similarity 接近 Fine-tune**: 10min Cosine Sim 0.7324 vs 0.7362,差距极小 [Table 1]
3. **XAB 偏好显著优于单 Adapter**: 10min 偏好比 43.63% vs 21.05% [Fig 3]
4. **参数效率**: 仅用 11.62% 参数达到接近甚至超过 Full Fine-tune 的效果 [§4.4]

## 局限性

1. **实验规模有限**: 仅 10 个说话人 (VCTK), 单一语言 (英语), backbone 仅 3.6M 参数,无法确认方法在大规模场景下的表现 [agent 解读]
2. **缺少与 CLN-based 方法的对比**: 未与 AdaSpeech 系列 (同为参数高效方法) 直接比较,无法判断 MoA 是否优于 CLN [agent 解读]
3. **无 adapter 专业化分析**: 论文声称多个 adapter 捕获互补特征 (prosody, speaking rate, accent),但未提供任何可视化或 ablation 证据 [§1 vs 实验部分]
4. **Fine-tune 在 10min 时 MOS 异常下降**: 3.45 (1min) → 3.18 (10min) → 3.54 (15min),论文未讨论此现象 [Table 1]
5. **WER 偏高**: 所有方法 WER 均 > 0.19,说明基线 TTS backbone 本身可懂度有限 [Table 1]
6. **评估规模小**: 20 个评估者, 60 个合成样本,主观评估统计效力有限 [§4.4]
7. **Vocoder 未说明**: 论文未明确使用哪个 vocoder,影响可复现性 [agent 解读]

## 点评

ADAPTERMIX 是将 MoE routing 机制引入 TTS speaker adaptation adapter 的早期尝试,思路清晰: 多个 shared adapter + expert choice routing 解决了 per-speaker adapter 的扩展性问题。在 2023 年的 Transformer TTS 框架下,用 11% 参数达到接近 full fine-tune 的效果是有意义的。

然而,本文的实验设计有明显不足: (1) backbone 太小 (3.6M),无法证明方法在现代规模模型上的有效性; (2) 缺少与同期最相关的 CLN-based 方法 (AdaSpeech) 的对比; (3) 核心卖点 "adapter 捕获互补特征" 缺乏证据支持。Fine-tune 在 10min 场景 MOS 异常低值也令人担忧。

从 2023 到 2026 年的发展来看,LLM-based TTS (VALL-E → CosyVoice → Seed-TTS) 通过 in-context learning 直接绕过了 adapter fine-tuning 的范式,zero-shot 能力使 few-shot adaptation 的需求大幅减少。ADAPTERMIX 的 MoE routing 思路在 DiaMoE-TTS 等后续工作中得到更成熟的发展,但已不再用于 speaker adaptation 而是用于模型整体的条件建模。

## 可复用的 idea

1. **Expert choice routing 用于 adapter 分工**: 让每个 adapter 自主选择处理哪些 token,避免人工指定分工。这个思路可迁移到任何需要多模块协作的场景 (e.g., multi-style adapter, multi-emotion adapter)。

2. **Adapter 跨说话人/跨任务共享**: 不为每个新条件训练独立模块,而是共享一组 adapter + routing,以 O(1) 参数支持 O(N) 个条件。对现代 LoRA-based adaptation 也有参考价值。

3. **Variance adapter 独立适应**: 在 variance predictor (duration/pitch/energy) 后也加 adapter,不仅适应音色还适应韵律。这提醒在做 speaker adaptation 时不能只关注 decoder/acoustic model。
