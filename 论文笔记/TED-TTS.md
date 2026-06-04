---
type: paper
tier: deep
title: "TED-TTS: Training-Free Intra-Utterance Emotion and Duration Control for Text-to-Speech Synthesis"
arxiv_id: "2601.03170"
source: "Sources/TED-TTS.pdf"
authors: [Qifan Liang, Yuansen Liu, Ruixin Wei, Nan Lu, Junchuan Zhao, Ye Wang]
year: 2026
venue: "arXiv"
tags: [TTS, emotion-control, duration-control, training-free, inference-time, zero-shot, autoregressive, controllable-TTS, intra-utterance]
concepts: ["[[Emotion Control in TTS]]", "[[Prosody Modeling]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 1 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: TED-TTS 属于 inference-time controllable TTS 分支,与 [[论文笔记/EmoSteer-TTS|EmoSteer-TTS]] (activation steering)、[[论文笔记/WeSCon|WeSCon]] (self-training word-level emotion) 构成当前 intra-utterance 情感控制的三条主要路线:
- 与 EmoSteer-TTS 都是 training-free,但 EmoSteer-TTS 通过 difference-in-means 激活 steering 操作 DiT 层内部表征,TED-TTS 通过注意力掩码和对齐算法在自回归解码器层面操作
- 与 WeSCon 都追求 intra-utterance 粒度,但 WeSCon 需 self-training (~500h ESD),TED-TTS 完全不需要训练
- 与 EmoCtrl-TTS (27kh 帧级控制) 和 TTS-CtrlNet (ControlNet 旁挂) 相比,代价最低但控制粒度受限于 segment 级

**已有认知**:
- [[Emotion Control in TTS]] [待确认]: 情感控制演进线从 emotion embedding 到 training-free activation steering 到 self-training word-level,TED-TTS 代表 training-free 路线中面向 AR TTS 的分支
- [[Prosody Modeling]]: LLM-TTS 的核心局限是"隐式建模使细粒度韵律控制困难",TED-TTS 通过 inference-time 干预回应此局限
- [[Zero-shot Speech Synthesis]]: [[论文笔记/IndexTTS2|IndexTTS2]] 是 TED-TTS 的 baseline,已是 AR zero-shot TTS 中 duration control 的 SOTA
- [[LLM-based TTS]]: AR TTS 中语义 token 连续生成,缺少显式 segment boundary,使 intra-utterance 控制成为根本性挑战
- [[Speech Tokenizer]]: IndexTTS2 使用 MaskGCT semantic codec 作为 speech tokenizer,TED-TTS 继承此设计
- [[模型库/CosyVoice 2|CosyVoice 2]]: 作为对比模型,在 instruction-following 情感控制上表现较弱

**创新判断**: 首个在 AR TTS 上实现 training-free segment-level emotion+duration 联合控制的框架。相比 EmoSteer-TTS 操作 flow-matching DiT 激活,TED-TTS 操作 AR 解码器的注意力掩码和对齐信念,属于不同技术路线。

> 检索命中: [[Zero-shot Speech Synthesis]]✓, [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[Prosody Modeling]]✓, [[CosyVoice 2]]✓ | 过滤: [[Emotion Control in TTS]](pending-review) | 未命中但可能相关: [[Duration Predictor]]

## 速查

> [!summary] 速查
> - **一句话**: 首个 training-free 框架,在预训练自回归 zero-shot TTS 上实现 intra-utterance 多段情感切换和时长控制,通过 2D causal attention mask + 单调流对齐(MSA)隔离 segment 条件并追踪对齐,通过 duration embedding steering + EOS logit 调制实现局部时长控制
> - **路线**: 文本分段 + 每段 emotion/duration spec → LLM (Qwen3-8B fine-tuned) 自动 prompt → IndexTTS2 T2S 模块 + 2D causal mask + MSA 对齐 → segment-aware emotion conditioning + duration steering → semantic tokens → waveform
> - **指标**: SMOS 4.00-4.22 / NMOS 4.07-4.22 (EN, SOTA) [Table 1, 2]; Duration error 3.21-3.39% (全设置, vs baseline 5.78-12.03%) [Table 4]; RTF < 1.0 (40% overhead vs baseline) [Fig 6]
> - **可借鉴**: 2D causal attention mask 实现 segment-local conditioning 的思路可迁移到任何需要 intra-sequence 条件切换的 AR 生成任务; MSA 的 Bayesian 对齐追踪是在线 text-semantic alignment 的通用方案
> - **局限**: 不建模 segment 间的渐进情感过渡(仅 segment-wise 切换); duration 精度受限于预训练模型的 duration embedding 表示能力; 仅在 IndexTTS2 上验证,跨模型泛化性未证明; 代码开源但 MED-TTS 数据集未见公开

## 核心问题

**问题**: 现有可控 TTS 方法在 intra-utterance level 的情感和时长控制上存在根本限制 — 多数方法仅支持 utterance-level 全局情感,少数支持 intra-utterance 控制的方法 (如 WeSCon) 需要专用训练数据和多阶段训练,严重限制了跨模型迁移性和部署灵活性 [§1]。

**核心追问**: 能否在不重新训练模型的前提下,仅通过推理时的条件注入和解码策略改造,实现稳定的 segment-level 情感转换和时长控制?

**解决思路**: 将问题从"训练更好的模型"重构为"更好地使用现有模型" — 通过重新组织 conditioning information 在 autoregressive decoding 中的可见性和更新方式,实现 training-free 的 intra-utterance 控制 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TED-TTS 基于 IndexTTS2 的 text-to-semantic (T2S) 模块,不改动模型参数,仅在推理时引入三个组件 [§3]:

1. **MED-TTS 数据集 + Qwen3-8B**: 自动将用户输入文本转换为结构化多段 prompt (每段包含文本、情感描述、目标时长),消除手动 prompt 工程 [§3.1]
2. **Segment-aware Emotion Conditioning**: 2D causal attention mask + Monotonic Stream Alignment (MSA),在 AR 解码中实现 segment-local 情感条件注入和平滑切换 [§3.2]
3. **Segment-aware Duration Steering**: Local duration embedding steering + Global EOS logit modulation,实现 segment-level 时长控制 [§3.3]

### 关键设计选择

#### 设计1: 2D Causal Attention Mask

**WHY**: AR T2S 生成 semantic tokens 作为连续流,没有显式 segment 边界,无法直接将 segment-level emotion conditions 映射到对应 tokens [论文原文, §3.2]。

**HOW**: 在标准 causal attention 之上叠加 segment-local condition visibility 约束 [§3.2, Fig 2, Appendix B]:
- 文本 token $x_t$ 只能 attend 到其所属 segment 的 condition embedding $C_{seg_x[t]}$,其他 segment 的 condition 被 mask 为 $-\infty$
- 已生成的 semantic token $s_r$ 同样只能 attend 到其生成时 active segment 的 condition
- Condition embedding 之间互相隔离,不存在 cross-condition 信息泄露
- **关键**: 文本和 semantic token 之间保留标准 causal attention,保证全局语义连贯

[agent 解读] 这个设计本质上是将 AR transformer 的 attention 空间分割为"全局语义层"(text/semantic 之间正常 causal) + "局部条件层"(condition 只在 segment 内可见),二者通过同一 attention matrix 实现。这比 WeSCon 的 Dynamic Emotional Attention Bias 更优雅 — DEAB 是加性 bias,这里是硬 mask,条件隔离更彻底。

#### 设计2: Monotonic Stream Alignment (MSA)

**WHY**: 2D mask 需要知道当前 semantic token 对齐到哪个 text segment,才能触发 mask 切换。直接用 raw attention maps 做对齐不可靠 — "raw attention maps are often noisy, head-dependent, and non-monotonic" [论文原文, §3.2]。

**HOW**: MSA 是一个在线 Bayesian 对齐追踪算法,维护 text positions 上的 belief distribution [§3.2, Fig 4, Algorithm 1]:

1. **Predict**: 将前一步后验 $\pi_{i-1}$ 通过 monotonic transition operator $P$ (鼓励前进、抑制后退) 传播,得到先验 $\hat{\pi}_i$ [Eq. implicit]
2. **Select**: 从所有 attention heads 中选择与先验最一致的 head: $(l^*, h^*) = \arg\max_{l,h} \hat{\pi}_i^\top \log A_i^{(l,h)}$ [Eq. 1]
3. **Update**: 将选定 head 的 attention 与先验融合: $\pi_i = (\hat{\pi}_i \odot G_\sigma(A_i^{(l^*,h^*)})) / Z$ [Eq. 2]

当后验的期望位置超过当前 segment 边界时,触发 segment 切换: $\sum_t t \cdot \pi_i[t] > b_m$ → $m \leftarrow m+1$ [Algorithm 1, line 38-40]

[agent 解读] MSA 的核心贡献在于将 noisy multi-head attention 转化为 stable monotonic alignment,三个设计选择都很关键: (1) monotonic prior 保证对齐只能前进; (2) 动态 head selection 避免依赖固定 head (不同层/head 对齐质量可能随 decoding 位置变化); (3) Gaussian smoothing 平滑 attention noise。与 MAS (Glow-TTS) 的区别: MAS 是离线全局搜索,MSA 是在线逐步追踪。

#### 设计3: Local Duration Embedding Steering

**WHY**: AR 模型的实际生成速度可能偏离用户指定时长 [论文原文, §3.3]。

**HOW**: 利用 IndexTTS2 的 duration embedding table $W_{dur}$ (与 semantic positional embedding 共享),在 AR 解码中根据 MSA 估计的 text/semantic progress 差异动态修正 [§3.3]:
- 计算 $\Delta r = r_{text} - r_{sem}$ (text progress - semantic progress, 正值表示 semantic 生成落后)
- 比例控制器调整: $\Delta\hat{D}_i = \text{clip}(\lfloor k \cdot \Delta r \rceil, -\Delta_{max}, \Delta_{max})$ [Eq. 3]
- 重新查询 duration table 更新当前 segment 的 duration embedding
- 更新频率: 每 5 步一次; 最大调整幅度: $\Delta_{max} = 10$ tokens [Appendix C]

#### 设计4: Global EOS Steering

**WHY**: Local steering 控制局部节奏,但不控制何时终止生成 [论文原文, §3.3]。

**HOW**: 对 EOS logit 施加自适应 bias [§3.3, Appendix C]:
- 非最终 segment: EOS logit 被完全抑制 (防止提前终止)
- 最终 segment: 根据已生成 token 与目标预算的比值动态调整 — ratio < 0.5 强抑制, [0.8, 1.1] 中性, > 1.2 强鼓励; bias 范围 $[-5.0, +15.0]$

#### 设计5: MED-TTS 数据集 + LLM 自动 Prompt

**WHY**: Intra-utterance 控制需要将文本分段并为每段指定情感和时长,手动标注不现实 [论文原文, §3.1]。

**HOW**: 构建 30,000 样本 (15k 中文 + 15k 英文) 的多情感时长标注文本数据集 MED-TTS [§3.1]:
- Step 1: GPT-4o 生成情感丰富的内容文本 (7 种情感 × 3 种文本类别)
- Step 2: DeepSeek-Chat 进行 segment-level 情感标注和时长估计
- Step 3: 自动检查 + 人工验证 (1000 样本随机抽样审核)
- 基于 MED-TTS 对 Qwen3-8B 做 LoRA fine-tuning (rank=32, 4 epochs) [Appendix A.4]

### 训练策略

**TED-TTS 本身不需要训练** — 仅 Qwen3-8B prompt 构造器需要 LoRA fine-tuning。核心的 emotion conditioning 和 duration steering 完全是 inference-time 操作,不修改 IndexTTS2 的任何参数 [§3]。

## 实验

| 指标 | 本文 (TED-TTS) | Baseline (IndexTTS2) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SMOS (EN, speech prompt) | 4.00±0.24 | 3.20±0.36 | MED-TTS eval (500) | [Table 1] |
| NMOS (EN, speech prompt) | 4.20±0.23 | 2.98±0.30 | MED-TTS eval (500) | [Table 1] |
| EMOS (EN, speech prompt) | 3.42±0.30 | 4.07±0.26 | MED-TTS eval (500) | [Table 1] |
| SMOS (ZH, speech prompt) | 4.13±0.23 | 3.67±0.33 | MED-TTS eval (500) | [Table 1] |
| NMOS (ZH, speech prompt) | 4.07±0.30 | 3.02±0.30 | MED-TTS eval (500) | [Table 1] |
| DNSM (EN, speech prompt) | 3.925 | 3.871 | MED-TTS eval (500) | [Table 1] |
| SSIM (EN, speech prompt) | 0.485 | 0.457 | MED-TTS eval (500) | [Table 1] |
| NISQA (EN, speech prompt) | 4.706 | 4.465 | MED-TTS eval (500) | [Table 1] |
| SMOS (EN, duration) | 4.22±0.22 | 3.89±0.33 | MED-TTS eval (500) | [Table 2] |
| NMOS (EN, duration) | 4.20±0.25 | 3.87±0.32 | MED-TTS eval (500) | [Table 2] |
| Duration error 1.0x (EN) | 3.218% | 7.100% (baseline) | MED-TTS eval (500) | [Table 4] |
| Duration error 0.75x (EN) | 3.387% | 5.778% (baseline) | MED-TTS eval (500) | [Table 4] |
| Duration error 1.25x (EN) | 3.211% | 12.032% (baseline) | MED-TTS eval (500) | [Table 4] |
| RTF (EN, full system S3) | ~0.87 | ~0.62 (S1) | N/A | [Fig 6] |

**关键实验发现**:

1. **Emotion 控制**: TED-TTS 在 SMOS/NMOS 上全面 SOTA (EN speech prompt: 4.00/4.20),但 EMOS 略低于 IndexTTS2 (3.42 vs 4.07)。[论文原文] 认为这是 training-free 框架的预期代价 — 情感识别准确率受限于不修改模型参数 [§5.1]。[agent 解读] EMOS 下降反映了一个 trade-off: intra-utterance multi-emotion 任务中,每段情感受相邻段干扰,纯粹的情感准确度不如独立合成每段; 但 NMOS (过渡自然度) 大幅提升说明连续生成比拼接更自然。

2. **Duration 控制**: 全设置下 semantic token number error 3.21-3.39%,比去除 local steering (3.03-11.59%) 和去除 global EOS (1.94-9.16%) 都显著更低更稳定 [Table 4]。关键: 不同 scaling factor 下误差保持一致 (~3.2%),而 baseline 和 ablated 版本在极端 scaling (1.25x) 时误差急剧上升。

3. **MSA 对齐质量**: 完整 MSA 的 segment boundary MAE 为 0.157 (vs greedy monotonic 0.216, raw attention 更高) [§5.2, Fig 5]。Top-3 head selection 优于 single max head [Table 7]。

4. **效率开销**: 完整系统 RTF 增加 ~40% (EN: 0.62→0.87),但仍 < 1.0,保持实时推理 [Fig 6]。

5. **公平性说明**: 对比模型 (MaskGCT, F5-TTS, CosyVoice2, Spark-TTS) 都不支持 intra-utterance 控制,因此需要独立合成每段再拼接评估 [§5.1]。这意味着 TED-TTS 的 DNSM/SSIM 优势部分来源于"单次连续生成 vs 拼接"的天然优势。

## 局限性

1. **Segment-wise 而非连续情感**: 情感控制是 segment-level 离散切换,不建模 segment 间的渐进过渡 (如从 happy 平滑过渡到 sad)。"emotional variation is controlled in a segment-wise manner rather than through a continuous emotion trajectory" [Limitations]

2. **Duration 精度受限于预训练模型**: TED-TTS 不修改模型参数,duration embedding 的表示能力是固定的。"the duration embedding may not always support strictly linear or fine-grained timing control, particularly under highly expressive or out-of-domain conditions" [Limitations]

3. **仅在 IndexTTS2 上验证**: 虽然论文声称"readily applicable to a broad class of autoregressive TTS backbones" [§6],但实验仅在 IndexTTS2 上进行。MSA 依赖 attention maps 的质量,不同 AR 模型的 attention 行为可能差异很大。

4. **EMOS 下降**: 在 speech prompt 设置下,EMOS (情感准确率) 低于 IndexTTS2 baseline (3.42 vs 4.07 EN) [Table 1],说明 intra-utterance 情感控制的准确性仍有提升空间。

5. **依赖 IndexTTS2 的 duration embedding**: Local duration steering 利用 IndexTTS2 特有的 duration embedding table,这一设计不一定可迁移到其他 AR TTS 模型。

## 点评

**创新性**: TED-TTS 的核心贡献在于将 intra-utterance 控制问题从训练侧转移到推理侧,且方案足够优雅 — 2D causal mask + MSA 的组合在概念上简洁,实现上与 AR decoding 无缝集成。这是 training-free controllable TTS 方向的重要一步。与同期 EmoSteer-TTS 的对比尤为有意思: EmoSteer-TTS 操作激活空间 (steering vectors),TED-TTS 操作 attention 空间 (mask + alignment); 前者做全局强度控制,后者做局部 segment 切换,二者互补。

**方法论**: MSA 算法是本文最有技术含量的部分。将 Bayesian 在线推断引入 attention-based alignment tracking,通过 monotonic prior + dynamic head selection + Gaussian smoothing 三重机制实现稳定的在线对齐,是独立于 TTS 应用的通用贡献。

**实验设计**: 公平性问题值得注意。对比模型需要独立合成每段再拼接,TED-TTS 是单次连续生成。DNSM (transition smoothness) 在这种设置下天然有利于连续生成方法。不过这也正说明了 TED-TTS 的实际价值 — 消除拼接瑕疵。

**实用性**: 40% 的 RTF 开销对实时应用是可接受的,但 MED-TTS 数据集的质量和 Qwen3-8B 自动 prompt 的准确性需要进一步验证。此外,仅在 IndexTTS2 上验证是一个显著的泛化性风险。

## 可复用的 idea

1. **2D Causal Attention Mask for Segment-local Conditioning**: 在保持全局语义 causal attention 的同时限制 condition embedding 的可见性到局部 segment。可迁移到任何需要在 AR 生成中实现 intra-sequence 条件切换的场景 (如对话 TTS 中的说话人切换、音乐生成中的风格切换)。

2. **Monotonic Stream Alignment (MSA)**: 基于 Bayesian 推断的在线 text-semantic 对齐追踪,将 noisy multi-head attention 转化为 stable monotonic alignment。可独立用于 AR TTS 的对齐监控、streaming TTS 的文本进度追踪、或任何需要从 attention maps 提取可靠对齐的任务。

3. **EOS Logit Modulation for Duration Control**: 通过对 EOS token 的 logit 施加基于 progress ratio 的自适应 bias,在 AR 生成中实现全局时长控制。方案简单有效,可直接迁移到其他 AR TTS/语言模型的长度控制。

4. **LLM-based Automatic Prompt Construction**: 用 LLM fine-tuning 将自由文本自动转换为结构化控制 prompt,消除 segment-level 手动标注需求。可用于任何需要结构化输入但用户期望自由文本输入的系统。

## 审阅

> [!review] 审阅 (2026-06-04, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 5 个设计选择均有 WHY+HOW,速查可借鉴具体可迁移 |
> | 可信赖 | pass | 14 个数据点全标注出处,指标命名正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 覆盖率 100%,推断标注清晰 |
> | 可定位 | pass | 三维度 KB 定位 (Emotion/Duration/Prosody),比较具体 |
> | 不污染 | pass | 仅 append 操作,无新建页,无 overclaim 风险 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/TED-TTS-review.yml`
