---
type: paper
tier: deep
title: "Zoology: Measuring and Improving Recall in Efficient Language Models"
arxiv_id: "2312.04927"
source: "Sources/Zoology.pdf"
authors: [Simran Arora, Sabri Eyuboglu, Aman Timalsina, Isys Johnson, Michael Poli, James Zou, Atri Rudra, Christopher Re]
year: 2023
venue: "arXiv preprint"
tags: [sequence-modeling, attention, gated-convolution, SSM, associative-recall, hybrid-architecture, efficiency, language-model, Transformer, Hyena, RWKV, H3, in-context-learning]
concepts: []
models: []
tasks: []
datasets: []
kb_context_sources: 0
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (未找到相关知识库背景)
> 本 vault 以 TTS/语音合成为核心,概念库中无"associative recall"、"gated convolution"、"state space model"等通用序列建模概念页。本文属于通用语言建模架构分析,与 vault 中已有的 [[论文笔记/RWKVTTS|RWKVTTS]] (将 RWKV-7 用于 TTS) 和 [[论文笔记/MambaVoiceCloning|MambaVoiceCloning]] (将 Mamba 用于 TTS conditioning) 形成上游理论支撑关系:Zoology 从根本上解释了为什么 RWKV/Hyena 等 sub-quadratic 架构在需要 recall 的任务上弱于 attention,以及如何通过混合架构弥补。
>
> 检索命中: 无 | 过滤: 无 | 未命中但可能相关: [[论文笔记/RWKVTTS]], [[论文笔记/MambaVoiceCloning]]

## 速查

> [!summary] 速查
> - **一句话**: 系统性证明 gated convolution 架构 (Hyena/RWKV/H3) 在 associative recall 上显著弱于 attention (70M attention > 1.4B Hyena),提出 Multi-Query Associative Recall (MQAR) 形式化框架并证明 gated convolution 的 model dimension 需随序列长度线性增长,而 attention 不需要;最终用 <10% 层的 input-dependent sparse attention hybrid 关闭 85% 的 AR gap (programmatic selection, 360M) [Table 2]
> - **路线**: 17 个语言模型预训练 (Pile, 70M-1.4B) → fine-grained perplexity 分析发现 AR gap → 定义 MQAR 合成任务 → 理论分析 BaseConv capacity → 证明 input-dependence 必要性 → sparse attention hybrid 验证
> - **指标**: 360M BaseConv+3层 programmatic attention hybrid: 整体 PPL 9.54 (vs attention 9.44), AR slice PPL 2.35 (vs attention 1.98), 关闭 85% AR gap; 360M BaseConv+full attention hybrid: PPL 8.59, 比纯 attention 好 0.85 PPL 且 FLOPs 少 18% [Table 2]; 1.4B 规模: 70M attention AR PPL 2.41 vs 1.4B Hyena AR PPL 3.43 [Table 5]
> - **可借鉴**: (1) 用 n-gram 重复率作为 real data 上 AR 能力的 proxy metric,简单有效 [§3.1]; (2) MQAR 合成任务设计:大词表 (8192) + 多查询 + power-law 距离分布,比旧的单查询合成任务更能预测 downstream gap [§3.2, Alg.1]; (3) BaseConv 作为 gated convolution 的 canonical representation,19 行 PyTorch,便于理论和实验分析 [§4.1, Appendix B]; (4) 极少量 input-dependent 层 (<10%) 即可大幅提升 recall,为 speech/audio 中的 SSM-attention 混合比例提供理论依据 [§5]
> - **局限**: 仅在 Pile 上验证,未涉及 speech/audio domain 的 recall 特性 [§1]; 最大规模止于 1.4B (50B tokens) + 7B (开源模型评估),未覆盖当前 7B+ 训练规模 [§G]; AR heuristic 仅测量显式 bigram 重复,不覆盖 fuzzy/semantic recall [§C.1]; 理论结果依赖 BaseConv 到 Hyena 的 poly-log 等价,实际训练动态未考虑 [Thm 4.2]

## 核心问题

本文回答三个层次递进的问题:

1. **现象**: 当前 SoTA 的 gated convolution 语言模型 (Hyena, RWKV, H3) 真的追上了 attention 吗? → 没有,仍有 0.35-2.1 PPL 的差距 [Table 1]

2. **归因**: 差距来自哪里? → 82% 的差距来自 associative recall (AR) — 即模型回忆 context 中先前出现过的关联的能力。仅占 6.4% tokens 的 AR Hits 解释了绝大部分 perplexity gap [Table 1, §3.1]

3. **解法**: 如何在保持 sub-quadratic 效率的同时关闭 AR gap? → 在 gated convolution backbone 中插入极少量 (<10%) 的 input-dependent sparse attention 层 [§5, Table 2]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文不提出单一新架构,而是一个 **分析框架 + 最小修复方案**:

```
Step 1: 预训练 17 个语言模型 (5 架构 × 3-4 规模) → 发现 perplexity gap
Step 2: 定义 AR Hits heuristic → 归因 82% gap 到 associative recall
Step 3: 定义 MQAR 合成任务 → 在合成数据上复现 gap
Step 4: 定义 BaseConv (canonical gated convolution) → 理论分析 capacity
Step 5: 证明 input-dependence 的必要性 → 设计 sparse hybrid
```

### 关键设计选择

**1. 为什么用 bigram 重复率来衡量 AR?**

作者需要一个能在 10M+ tokens 规模上自动标注 "哪些 token 需要 recall" 的 heuristic。他们定义 AR Hit 为 "context 中先前出现过的 bigram 的最后一个 token,且该 bigram 在训练数据中出现 < 1250 次" [§3.1, §C.1]。频率阈值的设计考虑是:高频 bigram (如 "of the") 可能已被记忆,不需要 in-context recall [论文原文]。这个 heuristic 虽然粗糙 (不覆盖 fuzzy recall 和语义级 recall),但足够揭示架构间的系统性差异 [§C.1]。

**2. 为什么已有的单查询 AR 合成任务无法预测 downstream gap?**

Prior work (H3, Hyena) 的合成 AR 任务假设 (a) 每个输入只有一个 query,(b) query 在序列末尾固定位置,(c) 词表很小 (|V| < 50 < model dim) [§3.2]。在这些设置下 gated convolution 可以完美解决 AR。但 real language 中 (a) 每个 forward pass 需要多次 recall (如 "Hakuna Matata" 和 "no worries" 在同一段中),(b) query 在任意位置,(c) 词表远大于 model dimension (30K-50K) [论文原文]。MQAR (Def. 3.1) 正是为了捕捉这些差异而设计 [§3.2]。

**3. BaseConv 作为 canonical representation 的意义**

BaseConv 定义为 `y := (uW + b1) ⊙ (h * u + b2)` [Eq. 1],即一个 linear projection 和一个 convolution 的 element-wise 乘积。它的关键性质是 **universality**: 任何 gated convolution 架构 (H3, Hyena, RWKV) 都可以被 BaseConv 以 poly-log blowup 模拟 (Thm 4.2),因此对 BaseConv 的理论结果可以推广到整个 gated convolution 家族 [论文原文]。具体地,BaseConv 与 Hyena 之间仅有常数倍的参数 blowup (Prop. H.12) [§4.1]。

**4. 为什么 gated convolution 在 MQAR 上效率低?**

核心论点:gated convolution 使用 **input-independent** 的卷积 filter — filter 由模型权重定义,不随输入变化 [论文原文]。但 MQAR 要求模型处理 **variable distance** 的 token-to-token 交互 (如 "Hakuna Matata" 间距 10 tokens, "no worries" 间距 9 tokens)。理论结果 (Thm 4.4): BaseConv 解 MQAR 需要 model dimension d = O(N log c),即随序列长度线性增长;而 attention 仅需 d = O(c^2),与序列长度无关 (Prop. 4.3) [§4.2]。

[agent 解读] 直观理解:attention 的 QK^T 计算天然是 input-dependent 的 — 每个 token pair 的交互权重由它们的内容决定。gated convolution 的 filter 是固定的,要覆盖所有可能的交互距离,必须用更多参数来 "记忆" 每个距离上的匹配模式。

**5. 为什么 input-dependent filter 能改善 scaling?**

Thm 4.5 证明:如果 BaseConv 使用 input-dependent kernels (filter 是输入的函数),则在交互距离种类数 t 有限时,可以用 O(t * Nc) 参数、O(1) 层解决 MQAR [§4.2]。实验中两种实现方式验证了这一点:(a) programmatic filter:在 token 重复的位置构造 spike;(b) autocorrelation filter:用自相关学习 fuzzy matching [§4.3, Fig. 2 bottom]。

**6. Sparse hybrid 的设计: 为什么只需 <10% 的 attention 层?**

作者在 BaseConv backbone 中只替换 3 层 (48 层中的 6.3%) 为 attention [§5]。这些 attention 层通过 selection function f(u) 决定哪些 token 需要 attend:
- Full attention: f(u)[i] = 1 for all i — O(N^2) [Eq. 2]
- Random: Bernoulli 采样 — 控制实验
- Programmatic: f(x)[i] = 1 iff x_i 在之前出现过 — 基于 token id 匹配 [Eq. 3]
- Learned: f(u)[i] = σ(u[i,:] · W),取 top-k — sub-quadratic,O(Ndk) [§5]

[论文原文] Programmatic selection 关闭了 85% 的 AR gap (360M),learned selection 关闭了 72% 并保持 sub-quadratic [Table 2]。关键 insight: 大部分 token 不需要 recall,只在少数 AR hit 位置引入 attention 即足够 [§5]。

### 训练策略

- 所有模型使用相同数据 (Pile)、相同训练基础设施 (GPT-NeoX)、相同 tokenizer (GPT2BPE)、相同数据顺序 [§C]
- Attention baseline 使用 LLaMA 架构 (RoPE + SwiGLU MLP) [§3.1]
- 合成实验: 2 层模型,AdamW,WD=0.1,sweep LR 从 1e-4 到 1e-2,64 epochs [§E.2]
- Downstream: 10B tokens 预训练 (main), 50B tokens (1.4B scale) [§3.1, §G]

## 实验

| 指标 | 本文 (BaseConv+hybrid) | Baseline (Attention) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Overall PPL (360M) | 9.54 (programmatic) / 8.59 (full attn) | 9.44 | Pile | [Table 2] |
| AR Hits PPL (360M) | 2.35 (programmatic) / 1.95 (full attn) | 1.98 | Pile | [Table 2] |
| Other Tokens PPL (360M) | 10.50 (programmatic) / 8.91 (full attn) | 10.62 | Pile | [Table 2] |
| FLOPs (360M) | 5.06 TFLOPs (programmatic) | 6.23 TFLOPs | Pile | [Table 2] |
| AR PPL gap closure (360M programmatic) | 85% of AR gap | -- | Pile | [Table 2] |
| AR PPL gap closure (360M learned, k=256) | 72% of AR gap | -- | Pile | [Table 2] |
| Overall PPL (125M) | 11.06 (programmatic) | 11.01 | Pile | [Table 2] |
| 70M Attn AR PPL vs 1.4B Hyena AR PPL | 2.41 (70M Attn) | 3.43 (1.4B Hyena) | Pile | [Table 5] |
| MQAR synth accuracy (d=64, N=512) | ~100% (Attention) | <10% (BaseConv, d<N) | Synth (|V|=8192) | [Fig 2] |

**Key experimental findings:**

1. **82% 归因**: AR Hits 仅占 6.4% tokens 但解释了 82% 的平均 perplexity gap (across H3/Hyena/RWKV at 125M-360M) [Table 1, §3.1]

2. **20x 规模差**: 70M attention 的 AR slice PPL (2.41) 优于 1.4B Hyena (3.43) [Table 5]。7B 开源对比 (RWKV vs LLaMA-2) 同样显示 RWKV 在增加 query 数时 AR 性能急剧下降 [Fig 7, §G]

3. **MQAR 合成实验**: Attention 在 d=64 即可完美解决所有序列长度的 MQAR;gated convolution (BaseConv/Hyena/RWKV/H3) 需要 d >= N 才能达到 >0.9 accuracy [Fig 2 top, §4.3]

4. **Input-dependent 的效果**: Programmatic input-dependent filter 使 BaseConv 实现接近 constant scaling (d vs N);autocorrelation filter 改善了 scaling 但不如 programmatic [Fig 2 bottom]

5. **Hybrid 的效率**: 360M BaseConv + 3 层 full attention hybrid 比纯 attention 好 0.85 PPL,同时 FLOPs 减少 18% (5.10 vs 6.23 TFLOPs) [Table 2]

6. **AR hit 距离分布**: Pile 中 AR hit 与先前 bigram 的距离服从 power law,大部分在 100 tokens 以内,少量 long-range [Fig 3, §D.1.2] — 这解释了为什么 sliding window attention 也能部分关闭 gap [Table 4]

## 局限性

1. **仅文本域**: 所有实验在 Pile 上进行,未验证在 speech/audio token 序列上是否有相同的 AR gap pattern。Speech token 的 vocabulary 结构和重复模式可能不同 [agent 解读]

2. **AR heuristic 的覆盖不全**: 仅测量显式 bigram 重复,不覆盖 (a) synonym-level recall ("iPhone" → "Android phone" → predict "cheaper"),(b) 高阶 n-gram recall,(c) 语义级 recall [§C.1]

3. **规模限制**: Main results 止于 360M (10B tokens);1.4B 实验仅 attention vs Hyena 单对比;7B 仅用开源模型零样本评估,无受控预训练对比 [§G]

4. **理论-实践 gap**: Thm 4.4 的 O(N log c) 下界依赖 arithmetic circuit 到 BaseConv 的转换,实际训练中模型可能学到近似解而非精确解;实验中的 scaling law 更像 d >= N 而非 d = O(N log c) [agent 解读]

5. **Hybrid 设计未优化**: Programmatic selection 需要运行时比较 token id (非标准操作);learned selection 需要额外的辅助损失和 top-k 计算;均未与后续的 Mamba (selective SSM) 或 Jamba (Mamba-Transformer hybrid) 做对比 [§5]

6. **未考虑 causal constraint**: Input-dependent filter (autocorrelation) 不天然满足 causality,作者提到但未解决 [§4.3]

## 点评

**意义**: 这是 sub-quadratic 序列模型研究中的一篇 **基础性分析工作** — 不是提出新架构,而是用严格的控制实验 + 理论分析解释了"为什么 Hyena/RWKV 还差 attention 一截"这个当时的核心问题。MQAR 框架后来被广泛采用 (包括 Mamba 论文的评估),BaseConv 作为 canonical representation 的构造也为后续理论分析提供了工具。

**方法论价值**: 本文展示了 **机制导向的架构分析** 的典范:不是盲目 scale up 看 perplexity,而是 (1) 归因到具体 capability (AR),(2) 用合成任务隔离变量,(3) 用理论预测 scaling,(4) 用最小修改验证假说。这种方法论对 TTS/speech 领域的 SSM-attention 混合架构选择有直接参考价值。

**对 TTS 方向的启示**: vault 中 [[论文笔记/RWKVTTS|RWKVTTS]] 将 RWKV-7 完整替换 CosyVoice 2 的 Transformer,但未报告 in-context recall 相关指标。[[论文笔记/MambaVoiceCloning|MambaVoiceCloning]] 用 Mamba 替换 attention conditioning path。Zoology 的理论预测暗示:(a) 纯 SSM/RWKV 的 TTS 系统在需要 long-range recall 的场景 (如重复说话人特征、韵律模式回忆) 可能存在隐藏的质量上限;(b) 少量 attention 层的 hybrid 可能是效率-质量的最优折中 — 这与 [[论文笔记/MamTra|MamTra]] (Mamba-Transformer hybrid for TTS) 的实证发现一致。[agent 解读]

**时间定位**: 发表于 2023.12,早于 Mamba (2023.12 同月) 和 Jamba (2024.03)。Mamba 通过 selective state spaces (input-dependent A, B matrices) 正是沿着 Zoology 指出的"input-dependence 是关键"这一方向的具体架构实现。[agent 解读]

## 可复用的 idea

1. **n-gram overlap 作为 recall proxy**: 在任何 token-based 生成系统 (包括 codec LM-based TTS) 中,可以用 context 内 n-gram 重复率来快速诊断模型的 in-context recall 能力,无需额外标注

2. **MQAR 合成 benchmark**: 评估新的 speech LM backbone (如 RWKV-7, Mamba-2, xLSTM) 时,可以先在 MQAR 合成任务上验证 recall capacity,避免昂贵的 full TTS pipeline 训练后才发现 backbone 能力不足

3. **极少量 attention 层的 hybrid 策略**: <10% 的 attention 层即可恢复大部分 recall 能力 — 这为 TTS 中的 streaming/low-latency 场景提供了具体的效率-质量 trade-off 参考:不需要全 attention,也不应该零 attention

4. **BaseConv 作为分析工具**: 对任何 gated convolution 变体 (包括 TTS 中的 ConvNeXt blocks, Mamba blocks 等),可以用 BaseConv 等价性来推理其理论 capacity 上限

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节 6 个设计选择含因果解释,可借鉴字段具体可操作 |
> | 可信赖 | pass | claim 标注覆盖率~90%,指标使用正确;速查一句话的 97.4% 已修正为 85% |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率~85% |
> | 可定位 | pass | KB 无匹配,诚实标注并链接相关笔记;非 TTS 核心领域故无概念页匹配 |
> | 不污染 | pass | 按指令不做反向更新和 git commit |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> 详见 `_review/Zoology-review.yml`
