---
type: paper
tier: deep
title: "Bridging the Gap Between Training and Inference in LM-Based TTS Models"
arxiv_id: "2509.17021"
source: "Sources/BridgingLM-TTSgap.pdf"
authors: [Ruonan Zhang, Lingzhou Mu, Xixin Wu, Kai Zhang]
year: 2025
venue: "arXiv"
tags: [TTS, LLM-based, exposure-bias, training-strategy, autoregressive, long-form-synthesis]
concepts: ["[[LLM-basedTTS]]", "[[SpeechTokenizer]]"]
models: ["[[CosyVoice]]", "[[CosyVoice2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 4
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[LLM-basedTTS]], [[CosyVoice]], [[CosyVoice2]], [[SpeechTokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文处理的是 [[LLM-basedTTS]] 范式中一个被广泛忽视但根本性的训练缺陷 -- exposure bias (训练用 GT token, 推理用自生成 token 的不一致)。LLM-based TTS 将语音合成重构为自回归 next-token prediction, 依赖 teacher forcing 训练。已有研究如 RLHF/DPO (FPO, GRPO-TTS, SpeechAlign) 从输出质量角度优化, 但本文从训练机制层面直接缩小 training-inference gap, 是互补的改进方向。
>
> **已有认知**: [[CosyVoice]] 采用 LLM + OT-CFM 的 coarse-to-fine 架构, 用监督式 semantic tokens (S3); [[CosyVoice2]] 在此基础上增加 streaming + instruction-following, 以 FSQ-SenseVoice tokenizer + text-based LLM 初始化为核心改进。两者均使用标准 teacher forcing 训练, 已被大量后续工作作为 baseline。[[SpeechTokenizer]] 将语音离散化为 50-100 tokens/sec, 这种高时间分辨率使 exposure bias 的影响远大于文本领域。
>
> **创新判断**: 此前 scheduled sampling (Bengio et al., 2015) 已被提出用于缓解 exposure bias, 但未针对 TTS 的高密度 token 特性进行适配。本文的 prompt-guided hybrid training + EOS-adaptive scheduling 组合是首次在 LM-based TTS 上系统性地量化和解决 exposure bias 的工作。
>
> 检索命中: [[LLM-basedTTS]], [[CosyVoice]], [[CosyVoice2]], [[SpeechTokenizer]] | 过滤: [[CodecLanguageModel]](pending-review), [[Non-autoregressiveTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 prompt-guided hybrid training 方案, 在训练中混入模型自生成 token 模拟推理分布, 配合 EOS-adaptive scheduling, 系统性缓解 LM-based TTS 的 exposure bias
> - **路线**: GT token 序列 → iteration 1 teacher forcing → iteration 2+ 替换部分 GT 为自生成 token (prompt protection 保留前 T1 个 GT) → EOS 预测反馈调节迭代次数 → 单次 backward (L_TF + sum(w_n * L_FR))
> - **指标**: CosyVoice2 WER 6.23→4.21 (LibriSpeech), SIM 0.74→0.78; A/B preference 54.3% win vs TF; 仅 1.5x 训练计算量
> - **可借鉴**: EOS 预测作为 exposure bias 严重度指标, 用于动态调节 free running 程度; prompt protection 策略可迁移到任何 AR 语音生成模型的训练
> - **局限**: 仅在 CosyVoice/CosyVoice2 上验证, 未测试其他 LM-TTS 架构; 短语音 (Seed-TTS, <10s) 改善有限; 未报告长语音专项评估 (如 >30s); 未开源

## 核心问题

LM-based TTS 在训练时使用 ground-truth (GT) speech tokens 作为 prefix (teacher forcing), 但推理时必须依赖模型自身生成的 token 进行自回归解码。这种 training-inference gap (即 exposure bias) 在 TTS 中被放大, 原因有三 [S1]:

1. **高密度 token**: 语音需要 50-100 tokens/sec, 远高于文本。一个音节对应多个 token, 单个预测错误会快速级联传播
2. **额外属性**: 语音携带 speaking rate 和 prosody 等文本中不存在的属性, 这些属性对 TTS 评估至关重要
3. **三种临床表现**: (a) 长语音生成时语速加快, (b) EOS 误预测导致提前终止或重复, (c) 韵律过度平坦化

核心研究问题: 如何在不显著增加训练成本的前提下, 让 LM-based TTS 的训练分布逼近推理分布, 从而缓解 exposure bias?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

方法的核心思想是在单个训练 step 内进行多次迭代, 逐步将 GT token 替换为模型自生成的 token, 从而让训练过程模拟推理时的自回归生成过程 [S2.1]。框架由三个组件构成:

1. **Prompt-guided hybrid training** (核心): 训练中混合 teacher forcing 和 free running
2. **Adaptive free running scheduling**: 基于 EOS 预测动态调节 free running 迭代次数
3. **统一损失函数**: L_TF + weighted sum of L_FR, 单次 backward

### 关键设计选择

**为什么不直接用 scheduled sampling?**
[论文原文] Scheduled sampling (Bengio et al., 2015) 虽然被提出用于缓解 exposure bias, 但在 TTS 中效果有限。因为语音 token 的高密度意味着 error 传播更快, 而 scheduled sampling 的随机替换缺乏针对性 [S1]。
[agent 解读] 本文的方案与 scheduled sampling 的本质区别在于: (1) 在单个 training step 内做多次前向传播, 产生"真实"的自生成 token 而非随机采样; (2) prompt protection 保证了训练稳定性; (3) EOS-adaptive 机制提供了基于质量的动态调节。

**Prompt-guided hybrid training [S2.1]**:
- Iteration 1: 标准 teacher forcing, 用 GT token 作为 prefix, 计算 L_TF
- Iteration 2~N: 取上一次迭代的模型输出, 将前 T1 个 token 替换为 GT (prompt protection), 其余保留为自生成 token, 送入模型计算 L_FR
- 随训练推进, 逐渐增加 GT 替换量 (T1 增大), 实现从 TF-dominated 到 self-generation-dominated 的平滑过渡 [S2.1]
- [agent 解读] Prompt protection 的双重作用: (a) 锚定 prompt following 能力, 因为 TTS 的 speaker prompt 对应前几个 speech token; (b) 被替换的 GT 部分也贡献额外的 TF loss, 进一步稳定训练

**Adaptive free running scheduling via EOS prediction [S2.2]**:
- EOS token 是否被正确预测 → 作为 exposure bias 严重程度的可靠指标
- EOS 预测错误 (premature EOS) → 增加 GT supervision, 减少 free running 迭代
- EOS 连续正确预测 → 增加 free running 迭代次数, 更充分模拟推理过程
- [论文原文] "too many iterations of free running may interfere with model convergence" [S2.2] — 这是引入 EOS-adaptive 的直接动机

**损失函数 [S2.3]**:

Teacher forcing loss (iteration 1):
L_TF = -sum_t log P(y_t | y^gt_{i<t}, X) [Eq. 1]

Free running loss (iteration 2+):
L_FR = -sum_{t=1}^{T1} log P(y_t | y^gt_{i<t}, X) - sum_{t=T1+1}^{T2} log P(y_t | y^gt_{i<T1}, y^pred_{T1<i<t}, X) [Eq. 2]

L_FR 由两部分组成: 前 T1 个 token 仍基于 GT prefix (prompt protection 区域), 后面的 token 基于 GT prefix + 自生成 token 的混合前缀。

总损失:
L_total = L_TF + sum_{n=1}^{N} w_n * L_FR^{(n)} [Eq. 3]

**关键效率设计**: 尽管每个 training step 有多次前向传播, 但只执行**一次** backward pass (累积所有迭代的 loss), 总训练成本仅为 baseline 的 1.5x [S3.5]。

### 训练策略

- Base models: CosyVoice 和 CosyVoice2 (预训练模型)
- 训练数据: LibriSpeech (~40K hours) [S3.1] (注: 论文引用 MLS [18] 但称 "LibriSpeech corpus", 实际为 Multilingual LibriSpeech ~40K hours, 非标准 LibriSpeech 960h)
- Optimizer: AdamW, constant lr=1e-5 [S3.1]
- Speech tokenizer: 单码本 VQ, 4096 entries [S3.1]
- 10K warm-up steps 后才引入 hybrid training [S3.1]
- [agent 解读] 10K warm-up 是关键: 让模型先通过纯 teacher forcing 建立基础能力, 再引入 free running, 否则模型初期自生成 token 质量太差, 会导致训练不稳定

## 实验

| 指标 | 本文 (CosyVoice2) | Baseline (CosyVoice2-TF) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER ↓ | **4.21** | 6.23 | LibriSpeech test-clean | [Table 1] |
| SIM ↑ | **0.78** | 0.74 | LibriSpeech test-clean | [Table 1] |
| WER ↓ | **4.64** | 4.83 | Seed-TTS | [Table 1] |
| SIM ↑ | **0.84** | 0.83 | Seed-TTS | [Table 1] |
| WER ↓ (CosyVoice) | **5.38** | 8.17 | LibriSpeech test-clean | [Table 1] |
| A/B preference (vs TF) | **54.3%** win | 13.3% win | LibriSpeech 30 samples | [Fig 2] |
| A/B preference (vs GT) | 26.9% win | — | LibriSpeech 30 samples | [Fig 2] |

**关键发现**:

1. **长语音改善更显著**: Seed-TTS (utterances <10s) 上改善较小 (WER 4.83→4.64), LibriSpeech 上改善大 (6.23→4.21)。[论文原文] "our method tends to perform better on longer speech samples by mitigating the accumulated error caused by exposure bias" [S3.2]

2. **Exposure bias 可视化** [Fig 3]: 计算 token-level accuracy (预测 token 与 GT 完全匹配的比例)。Baseline 中 teacher forcing 和 free running 之间有明显 accuracy gap; hybrid training 同时提升两种模式的绝对 accuracy 并显著缩小差距。注意: speech token accuracy 本身就很低 (~20% median), 因为相似音可对应不同 token (非一对一映射) [S3.4]

3. **EOS scheduling 验证** [Fig 4]: 训练初期 premature EOS 多发于前几次迭代; 训练推进后 premature EOS 延迟到更晚的迭代, 说明 free running 鲁棒性提升

**Ablation** [Table 2] (CosyVoice2, LibriSpeech):

| 配置 | WER ↓ | SIM ↑ |
| --- | --- | --- |
| w/o Prompt Protection | 7.25 | 0.65 |
| w/o EOS Adaptive | 4.98 | 0.72 |
| Full (w) | **4.21** | **0.80** |

- Prompt protection 是更关键的组件: 移除后 WER 退化到 7.25 (甚至比 baseline TF 6.23 更差), SIM 降至 0.65
- [agent 解读] 这说明无 prompt protection 的 free running 会破坏模型的 prompt following 能力, 导致 speaker identity 丢失

## 局限性

1. **验证范围窄**: 仅在 CosyVoice/CosyVoice2 两个模型上验证, 未测试 VALL-E、Seed-TTS、IndexTTS2 等其他主流 LM-based TTS 架构。方法能否推广到不同 tokenizer (如 multi-codebook RVQ) 和不同 LM 架构 (如 non-causal masked LM) 未知

2. **长语音评估不充分**: 论文强调方法对长语音合成尤其有效, 但未报告 >30s 的专项评估。LibriSpeech test-clean 的平均时长较短, 无法充分体现长语音优势

3. **缺少语速/韵律定量评估**: 论文 claim exposure bias 导致语速加快和韵律平坦 [S1], 但实验仅报告 WER 和 SIM, 未提供 speaking rate deviation 或 pitch variance 等定量指标

4. **训练超参数敏感性未探讨**: 10K warm-up steps、w_n 权重、T1 增长策略、最大迭代次数等超参数如何影响性能, 缺乏系统分析

5. **未开源**: 代码和模型未公开

## 点评

**优点**:
- 问题定义精准: exposure bias 在 TTS 领域被广泛忽视, 本文首次在 LM-based TTS 上进行系统量化 (Fig 3 的 accuracy gap 可视化是很好的分析工具)
- 方法设计巧妙: EOS 预测作为 exposure bias 严重度的 proxy 信号, 用于动态调节训练策略, 这个观察很有实用价值
- 训练效率合理: 单次 backward + 多次 forward 的设计只需 1.5x 计算量, 工程上可行
- 消融实验清晰地展示了每个组件的贡献

**不足**:
- 缺乏与 RLHF/DPO 类方法 (如 FPO, GRPO-TTS) 的对比或组合实验。exposure bias 和 preference alignment 是互补的改进方向, 组合效果值得期待
- SIM 指标在 ablation 中波动较大 (0.65-0.80), 但论文未讨论 prompt protection 为何对 speaker similarity 影响如此显著
- LibriSpeech 上 IndexTTS baseline WER 8.30 较高, 论文未解释是否为官方结果 (IndexTTS 原文报告 WER 更低)

**在 LLM-based TTS 演进中的位置**:
本文与 RLHF/DPO 路线 (FPO, GRPO-TTS, SpeechAlign) 形成互补: 后者优化"生成什么质量的输出", 本文解决"训练怎么更接近推理"。两者可叠加。从方法论角度, 本文更接近 NMT 领域的 scheduled sampling 传统, 但通过 prompt protection + EOS-adaptive 做了 TTS-specific 的适配。

## 可复用的 idea

1. **EOS 预测作为训练质量 probe**: 在任何 AR 生成模型中, EOS 的正确预测率可以作为模型是否"准备好" free running 的实时信号。这可以用于: (a) 动态调节训练策略, (b) 早期诊断 exposure bias 严重程度, (c) 作为 curriculum learning 的 gate

2. **Prompt protection + 渐进式 free running**: 保留 prompt 区域的 GT token 这个策略可以直接迁移到任何需要 prompt following 的生成任务 (如 voice cloning, style transfer)。核心insight: 不是"全部换成自生成"而是"保护 prompt, 只在生成区域 free run"

3. **单次 backward + 多次 forward 的训练效率设计**: 对于需要 self-play 或 self-generated data 的训练方案, 这种"在一个 step 内多次 forward, 只做一次 backward"的设计是一个通用的效率模式

4. **Exposure bias 可视化方法**: Fig 3 的 token-level accuracy gap (TF mode vs free running mode) 是一种简单有效的诊断工具, 可以用于评估任何 AR 模型的 exposure bias 严重程度

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法含充分 WHY 解释, 设计选择有对比基准 |
> | 可信赖 | pass | 数字标注覆盖率 >90%, 指标名正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注一致, 覆盖率 ~90% |
> | 可定位 | pass | KB 背景具体引用范式/模型/互补路线 |
> | 不污染 | N/A | no-kb-update 模式 |
> 
> Issues: 1 (high: 0, medium: 1, low: 0)
> - (medium, traceability-gap) 论文称 "LibriSpeech" 但引用 MLS [18] (~40K hours), 已在笔记中加注说明
> 详见 `_review/BridgingLM-TTSgap-review.yml`
