---
type: paper
tier: deep
title: "Streaming Speech Recognition with Decoder-Only Large Language Models and Latency Optimization"
arxiv_id: "2601.22779"
source: "Sources/StreamingASR.pdf"
authors: [Genshun Wan, Wenhui Zhang, Jing-Xuan Zhang, Shifu Xiong, Jianqing Gao, Zhongfu Ye]
year: 2026
venue: "arXiv preprint"
tags: [ASR, streaming, LLM, decoder-only, MoChA, latency-optimization, LoRA, Conformer, Mandarin]
concepts: ["[[Speech-TextAlignment]]", "[[ModalityAdaptationforSpeechLLM]]", "[[StreamingSpokenDialogue]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文属于 latent-representation-based Speech-LLM Integration [待确认] 路线中的 streaming ASR 方向。与 text-based integration 的 LLM Rescoring/GER [待确认] 不同,本文让 LLM 直接基于连续语音表征生成转写,而非后处理文本假设。

**已有认知**:
- [[SpeechLanguageModel]] 将 SpeechLM 定义为端到端处理语音的自回归模型,其中 LLM-based ASR 属于 latent-representation-based 集成路线的语音理解子任务。本文正是此路线在 streaming 场景的探索。
- [[Speech-TextAlignment]] 记录了四种 speech-text token 排列方式(speech-only / text-only / concatenated / alternating),其中 alternating 方式被 SUTLM 证明在跨模态任务表现最佳。本文的 interleaved speech-text 序列本质上属于 alternating 方式,但关键区别在于对齐边界由 MoChA 动态决定而非预设。
- [[ModalityAdaptationforSpeechLLM]] 总结了三种 adaptor 方法(Conv downsampling / CTC compression / Q-Former)。本文的 adaptor 是简单的 feed-forward network(最基础变体),但创新点不在 adaptor 本身而在其上游的 MoChA policy network。
- [[StreamingSpokenDialogue]] 汇总了 E2E streaming 的多种技术路线(causal convolution / chunk-wise encoding / block-by-block processing 等)。本文采用 context-sensitive chunking 进行 streaming encoding,与 Freeze-Omni 的 chunk-wise streaming 思路接近,但用 MoChA 替代固定分块策略实现动态 speech-text 同步。

**创新判断**: 与已有 streaming LLM-ASR 方法(BESTOW 的 wait-k、BTI 的 discrete token + hybrid boundary、SpeechLLM-XL 的 CTC force-alignment)相比,本文的 MoChA policy network 提供了一种无需预设 CTC/hybrid 对齐的自适应分割方案,且 minLT loss 进一步优化延迟。这是一条不依赖外部对齐器的端到端 streaming 路线。

> 检索命中: [[SpeechLanguageModel]]✓, [[Speech-TextAlignment]][待确认], [[ModalityAdaptationforSpeechLLM]][待确认], [[Speech-LLMIntegrationTaxonomy]][待确认], [[StreamingSpokenDialogue]][待确认], [[LLM-enhancedASR]][待确认] | 过滤: 后 5 个均为 pending-review | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 MoChA 作为 read/write policy network 自适应分割语音流,使 decoder-only LLM 能以交错序列方式进行 streaming ASR,并通过 minLT loss 大幅降低延迟
> - **路线**: 语音 → Conformer chunked encoder → FFN adaptor → MoChA policy network 动态分割 → 语音段与文本 token 交错 → Qwen 2.5-1.5B (LoRA) → 转写
> - **指标**: AISHELL-1 CER 5.1% (streaming) / 4.9% (non-streaming) [Table 1]; AISHELL-2 CER 5.5% / 5.0% [Table 2]; minLT 使平均延迟从 16 帧降至 6 帧 (62.5% reduction),CER 仅从 5.4% 升至 5.5% [Table 4]
> - **可借鉴**: (1) MoChA 的 stop-and-decode 机制可迁移至其他需要动态 speech-text 对齐的场景(如 streaming speech translation); (2) streaming/non-streaming 共享参数联合训练简化系统开发; (3) minLT loss 作为可插拔的延迟正则项
> - **局限**: 仅在 Mandarin 数据上验证; LLM 仅用 1.5B 参数; 依赖 HMM forced alignment 生成 minLT 的 gold boundary; 未报告实际 RTF 或端到端延迟(仅报告 token-level 帧延迟)

## 核心问题

本文要解决的核心问题是: **如何让 decoder-only LLM 在 streaming 场景下进行语音识别,同时避免依赖外部 CTC/hybrid 模型进行 speech-text 对齐?**

现有 streaming LLM-ASR 方法的痛点 [§1]:
1. **依赖外部对齐**: BESTOW 用 wait-k 策略(固定 k 个 chunk 后预测), BTI 用 hybrid ASR 系统提取边界, SpeechLLM-XL 用 CTC force-alignment — 这些都引入了额外的级联模块,阻碍端到端优化 [论文原文]
2. **固定分块的延迟问题**: 固定 chunk size 的方法无法自适应地最小化 token 生成延迟 [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由四个模块组成 [§4.1, Fig 2]:

1. **Conformer Speech Encoder**: 12 层 Conformer,采用 context-sensitive chunking [17] 进行 streaming encoding。每个 chunk 0.4s,左侧上下文窗口 1.6s,**不使用任何未来上下文**以避免编码延迟 [§5.1]。训练时各 chunk 并行处理,丢弃上下文帧输出后拼接得到 utterance-level 特征。

2. **Adaptor (FFN)**: 一个前馈网络(hidden dim 1024, GELU activation),将语音表征投射到 LLM 的 word embedding space [§3]。作者选择 FFN 而非更复杂的 Q-Former/CTC compression,论文给出的理由是 "simplicity and competitive performance" [§3] [论文原文]。[agent 解读] 这可能因为本文的核心对齐机制已由 MoChA 承担,adaptor 只需做维度映射,无需承担序列压缩职责。

3. **Read/Write Policy Network (MoChA)**: 核心创新组件。基于 Monotonic Chunkwise Attention [6],配合一个轻量级 decoder:
   - 在每个 decoder timestep i,MoChA 从上一次停止位置 t_{i-1} 开始逐帧扫描 encoder 输出 [§4.1]
   - 对每帧 j 计算 selection probability p_{i,j} [§4.1]
   - 当 p_{i,j} 超过预设阈值时触发 stop-and-decode 信号,设 t_i = j [§4.1]
   - 在 hard alignment 之上再施加 soft chunkwise attention 聚合局部信息,增加灵活性 [§4.1]
   - 最终产生 encoder index 序列 [t_1, ..., t_L],每个 token y_i 基于语音段 h_{t_{i-1}+1:t_i} 解码 [§4.1]

4. **Decoder-only LLM (Qwen 2.5-1.5B)**: 28 层 Transformer,12 attention heads,hidden dim 1536。使用 LoRA (rank 32, alpha 64) 适配 Q/K/V/O 投影层 [§5.1]。保留原始 tokenizer 和词表以减轻灾难性遗忘 [§5.1]。

**训练时序列组织** [§4.1, Eq.2]: MoChA 产生的对齐将语音段和文本 token 交错排列:

```
H_y = [h_{t1+1:t2}, y1, h_{t2+1:t3}, y2, ..., h_{tL-1+1:tL}, y_{L-1}]
```

LLM 在这个交错序列上做 next-token prediction,cross-entropy loss 仅在每个语音段末帧(即预测 y_i 的位置)计算 [§4.1, Eq.3]。

**推理流程** [§4.3]: 输入语音先经 chunk-level 编码 → policy network 逐帧扫描直到触发 selection signal → 缓冲的语音段 + 上一个 token 送入 LLM 预测下一个 token → 预测 token 同时反馈给 LLM 和 MoChA → 重复直到 EOS。

### 关键设计选择

**为什么选 MoChA 而非 CTC/hybrid 对齐?** [论文原文] 现有方法依赖 CTC 或 hybrid 模型预先提取 speech-text 对齐,这种级联设计 "complicate end-to-end optimization" [§1]。MoChA 的优势在于: (1) 可微分,支持端到端训练; (2) 自适应边界(非固定 chunk); (3) 不需要额外的对齐模型。[agent 解读] MoChA 最初为 attention-based encoder-decoder 的 streaming 设计 [6],本文将其从 AED 框架迁移至 decoder-only LLM 框架是主要工程创新。

**为什么用 interleaved 而非 prepend?** [agent 解读] 标准 LLM-ASR (non-streaming) 将所有语音 embedding 作为 prefix prompt 一次性送入 LLM [§3, Fig 1]。但 streaming 要求边听边写,因此必须将语音段和文本交错排列,使 LLM 能在接收部分语音后就开始解码。这本质上是将 non-streaming 的 "先听完再说" 改为 streaming 的 "听一段说一字"。

**为什么联合训练 streaming + non-streaming?** [论文原文] 两个模型共享所有参数,仅前向计算路径不同。训练时每个 batch 随机分配为 streaming 或 non-streaming 模式 [§4.2]。作者认为这简化了训练 pipeline 并降低开发成本 [§1]。[Table 5] 证实联合训练不损害任一模式的性能(non-streaming: 5.0% vs 5.1%; streaming: 5.5% vs 5.6%)。

### 训练策略

总损失函数 [§4.2, Eq.5]:

```
L_total = L_LLM + L_MoChA + lambda * L_minLT
```

1. **L_LLM**: LLM 输出的标准 cross-entropy loss,在语音段末帧位置计算 [§4.1]
2. **L_MoChA**: Policy network 内部轻量 decoder 的 cross-entropy loss,与 LLM 共享词表。注意: policy network 的 decoder 输出仅用于训练,推理时丢弃 [§4.2]
3. **L_minLT (Minimal Latency Training)**: 基于 HMM forced alignment 的可微期望延迟损失 [§4.2, Eq.4]:

```
L_minLT = (1/L) * sum_i sum_j |j * alpha_{i,j} - b_i|
```

其中 alpha_{i,j} 是 MoChA 的边缘化对齐概率,b_i 是 forced alignment 提供的 gold boundary。lambda 设为 0.1 [§5.1]。

**优化器**: AdamW + triangular cyclic learning rate scheduler (max LR 1.5e-4, min LR 0, cycle 25k updates, total 100k steps) [§5.1]。推理用 beam search, beam size 10 [§5.1]。

## 实验

| 指标 | 本文 (streaming) | 本文 (non-stream) | BESTOW (stream) | BTI (stream) | WeNet-U2 (non-stream) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CER (%) | 5.1 | 4.9 | 5.3 | 5.9 | 5.0 | AISHELL-1 | [Table 1] |
| CER (%) | 5.5 | 5.0 | 5.6 | 7.2 | 6.1 | AISHELL-2 | [Table 2] |
| CER (%) | 7.6 | 6.7 | — | — | — | In-house MD | [Table 3] |

**延迟分析** [Table 4] (AISHELL-2, 1 frame = 40ms):

| 方法 | CER (%) | First (帧) | Mid. (帧) | Last (帧) | Avg. (帧) |
| --- | --- | --- | --- | --- | --- |
| Baseline-stream (MoChA enc-dec) | 6.1 | 19 | 15 | 7 | 15 |
| Proposed w/o minLT | 5.4 | 18 | 15 | 9 | 16 |
| Proposed w/ minLT | 5.5 | 10 | 5 | 2 | 6 |

minLT 使平均延迟从 16 帧 (640ms) 降至 6 帧 (240ms),CER 仅从 5.4% 微升至 5.5% [§5.3]。

**消融实验** [Table 5] (AISHELL-2):

| 配置 | Non-streaming CER (%) | Streaming CER (%) |
| --- | --- | --- |
| Proposed (完整) | 5.0 | 5.5 |
| w/o joint-train | 5.1 | 5.6 |
| w/o LoRA (冻结 LLM) | 5.4 | 5.7 |
| w/o Qwen init. (随机初始化) | 6.5 | 7.2 |

关键发现 [§5.4]:
- 联合训练不损害性能,单模型支持两种模式 [论文原文]
- LoRA 适配至关重要,冻结 LLM 导致性能下降 [论文原文]
- 预训练 LLM 初始化贡献最大(随机初始化 CER 暴增 1.7 个绝对点) [论文原文]

## 局限性

1. **仅验证 Mandarin**: 所有实验基于 AISHELL-1/2 和内部中文数据集,缺乏英文等多语言验证 [agent 解读]
2. **LLM 规模较小**: 仅用 Qwen 2.5-1.5B,未探索更大模型(如 7B)的 scaling behavior [agent 解读]
3. **minLT 依赖外部 HMM 对齐**: L_minLT 需要 HMM-based hybrid ASR 系统生成 forced alignment 作为 gold boundary [§4.2],部分削弱了"无需外部对齐器"的优势 [agent 解读]
4. **延迟评估不完整**: 仅报告 token-level 帧延迟(以 40ms 帧为单位),未报告实际 RTF 或端到端 wall-clock 延迟 [agent 解读]
5. **非实时 beam search**: 推理用 beam size 10 的 beam search [§5.1],在实际 streaming 部署中可能引入额外延迟 [agent 解读]
6. **训练数据规模有限**: AISHELL-1 仅 165h,AISHELL-2 为 1000h,未探索更大规模数据的效果 [agent 解读]

## 点评

本文的核心贡献是将 MoChA 从 AED 框架迁移到 decoder-only LLM 框架,实现了无需外部 CTC/hybrid 对齐的自适应 streaming ASR。这个迁移思路本身是合理且有价值的 -- MoChA 的 stop-and-decode 机制天然适配 streaming 的"边听边写"需求。

**亮点**:
- streaming/non-streaming 联合训练是一个实用的工程设计,使部署成本减半
- minLT loss 效果显著(延迟降 62.5%,CER 仅升 0.1%),且作为正则项可插拔
- 消融实验充分,特别是证明了 LLM 预训练初始化的关键作用

**不足**:
- minLT 仍需 HMM forced alignment,与"端到端"定位有矛盾。一个真正端到端的 latency loss 应可基于学习到的对齐自身计算,而非外部对齐器
- 实验仅限 Mandarin 且 LLM 仅 1.5B,泛化性存疑
- 与同期 streaming LLM-ASR 方法(如 SpeechLLM-XL [14])的对比不够完整,BESTOW 和 BTI 是作者自行复现的,可能存在实现差异

**在领域中的位置**: 本文是 streaming LLM-ASR 的一个稳健方案,但更像是对已有技术(MoChA + LLM + LoRA)的组合而非方法论突破。核心创新集中在"如何在 decoder-only LLM 框架下复用 MoChA"这一工程问题上。

## 可复用的 idea

1. **MoChA 作为通用 read/write policy**: 任何需要将变长连续信号动态对齐到离散 token 序列的场景都可考虑 MoChA policy network,例如 streaming speech translation 或 streaming speech-to-speech
2. **minLT loss 作为可插拔延迟正则**: 当有 forced alignment 可用时,minLT 提供了一种简单有效的延迟约束方法,且对识别精度影响极小
3. **Streaming/non-streaming 参数共享**: 两种模式仅在前向路径不同,共享所有参数 -- 这种设计可推广至 streaming/non-streaming TTS 或 speech translation 系统
4. **Cross-entropy loss 仅在段末帧计算**: 避免在语音帧位置计算无意义的文本预测 loss,减少噪声梯度

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节包含 WHY 解释,MoChA 迁移逻辑清晰 |
> | 可信赖 | pass | 数字均有出处标注,指标使用正确 |
> | 可区分 | pass | 论文原文 vs agent 解读标注充分 |
> | 可定位 | pass | KB 背景定位准确,与 BESTOW/BTI/SpeechLLM-XL 的差异明确 |
> | 不污染 | pass | 无新建概念页请求,反向更新安全 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> 详见 `_review/StreamingASR-review.yml`
