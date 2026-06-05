---
type: paper
tier: deep
title: "Kinetic-Optimal Scheduling with Moment Correction for Metric-Induced Discrete Flow Matching in Zero-Shot Text-to-Speech"
arxiv_id: "2605.09386"
source: "Sources/KineticOptimalTTS.pdf"
authors: [Dong Yang, Yiyi Cai, Haoyu Zhang, Yuki Saito, Hiroshi Saruwatari]
year: 2026
venue: "arXiv preprint"
tags: [discrete-flow-matching, Fisher-Rao, kinetic-optimal, CTMC, zero-shot-TTS, NAR, codec-based, DiT, moment-correction]
concepts: ["[[ConditionalFlowMatching]]", "[[MaskedGenerativeModeling]]", "[[Classifier-FreeGuidance]]", "[[ResidualVectorQuantization]]", "[[Non-autoregressiveTTS]]", "[[DiffusionModel]]", "[[DurationPredictor]]"]
models: ["[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/F5-TTS|F5-TTS]]", "[[论文笔记/OmniVoice|OmniVoice]]", "[[论文笔记/CosyVoice2|CosyVoice 2]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[论文笔记/DiFlow-TTS|DiFlow-TTS]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: ["[[数据集/Emilia|Emilia]]", "[[数据集/SEED-TTS-Eval|SEED-TTS-Eval]]", "[[数据集/CV3-Eval|CV3-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文位于离散空间 flow matching 在 TTS 中的应用前沿。连续空间 flow matching ([[ConditionalFlowMatching]]) 已在 TTS 中广泛应用(CosyVoice 系列、F5-TTS 等),主要用于连续 mel spectrogram 生成。离散空间 flow matching (discrete flow matching, DFM) 的 TTS 应用此前仅有 [[论文笔记/DiFlow-TTS|DiFlow-TTS]]。本文是首个将 metric-induced DFM (MI-DFM) 应用于 TTS 的工作,与 mask-source DFM 不同,MI-DFM 利用 codec token embedding 的几何结构定义概率路径。

**已有认知**:
- [[MaskedGenerativeModeling]] [待确认]: MaskGCT 等 masked generative 方法是本文的主要对比 baseline。MaskGCT 使用 per-layer masked generation + semantic-to-acoustic 两阶段,本文的 GibbsTTS 则是单阶段 full-codebook 直接生成。
- [[ResidualVectorQuantization]]: 本文使用 MaskGCT 的 codec (DAC encoder + Vocos decoder, 12 层 RVQ, codebook 1024, dim 8)。RVQ 的层级结构是 full-codebook vs per-layer 策略对比的基础。
- [[Zero-shotSpeechSynthesis]]: 评估遵循标准 protocol,使用 [[SEED-TTS-Eval]] 和 [[CV3-Eval]]。当前 SOTA 以 AR 模型为主(CosyVoice 3, Qwen3-TTS),NAR 模型在 speaker similarity 上有优势。

**创新判断**: 本文的核心贡献是算法层面的(kinetic-optimal scheduling + moment correction),而非系统层面的。与 OmniVoice(同期, 系统级设计)互补。相比 DiFlow-TTS(首个 DFM for TTS, mask-source DFM),本文使用 metric-induced 路径并解决了其 scheduler 超参搜索问题。

> 检索命中: [[ConditionalFlowMatching]]✓, [[ResidualVectorQuantization]]✓, [[Zero-shotSpeechSynthesis]]✓, [[SEED-TTS-Eval]]✓ | 过滤: [[MaskedGenerativeModeling]](pending-review), [[Non-autoregressiveTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 为 metric-induced discrete flow matching 推导了 training-free 的 kinetic-optimal 时间调度器 + finite-step moment correction,在 codec-based zero-shot TTS 上全面超越 masked generative baselines
> - **路线**: Text+Prompt → DiT backbone (全 codebook 联合预测) → CTMC sampling (KO scheduler + moment corrector) → Codec decoder → Waveform
> - **指标**: UTMOS 3.651 / WER 1.777% / SIM 0.743 (Seed-TTS test-en, 32 NFE) [Table 2]; CMOS 0 (ref) vs all baselines negative [Table 3]; SIM highest on 3/4 SOTA test sets [Table 4,5]
> - **可借鉴**: (1) Fisher-Rao 等速遍历作为 training-free scheduler 设计原则, 可迁移到其他 discrete generative task; (2) moment correction 不改 jump 方向只调 jump 概率的轻量 corrector 设计; (3) codebook-wise linearly decayed loss weight 稳定 full-codebook 训练
> - **局限**: 仅验证了 L2 cosine distance,未探索其他 token distance; 399M 参数在 WER 上不敌大规模 AR 系统; 仅验证 TTS 未推广到其他域; text frontend 较简单影响 WER

## 核心问题

MI-DFM 相比 mask-source DFM 有独特优势——利用 token embedding 的几何结构定义概率路径(Gibbs 分布),使得距目标 token 更近的 token 被优先采样。但其实际应用受两个问题限制:

1. **Scheduler 超参搜索**: β_t = c(t/(1-t))^a 中的 c, a 需要大量实验搜索,且最优值高度依赖任务和解码方式 [§7, Table 6]
2. **有限步路径跟踪误差**: 一阶 CTMC solver 的 jump probability ρ_base = 1-exp(-hλ_t) 仅在无穷小步长极限下精确,有限步数时偏离参考路径 [§4]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

GibbsTTS 由三部分组成:

1. **Kinetic-optimal scheduler**: 从 token 距离矩阵数值构建,无需超参搜索
2. **DiT backbone**: 预测目标 token 分布 p_{1|t}^θ
3. **CTMC sampler + moment corrector**: 有限步采样,corrector 调整 jump 概率

**生成范式**: 不同于 MaskGCT 的 text→semantic→acoustic 两阶段 + per-layer 生成,GibbsTTS 采用 text→codec 单阶段 + full-codebook 联合生成。从 uniform random token 初始化(非 mask 初始化),通过 CTMC 迭代跳转到目标 token [§5, Fig 1]。

### 关键设计选择

#### 1. Kinetic-Optimal Scheduler (核心贡献一)

**问题**: 给定 MI-DFM 的概率路径 p_t(x|x_1) = softmax(-β_t · d(x,x_1)),如何选择 β_t 使得采样质量最优?

**解法**: 最小化 Fisher-Rao path energy。论文证明三个关键结论 [§3, Appendix B]:

- **Lemma 1**: Fisher-Rao 路径长度不随 scheduler 变化(几何不变量)
- **Lemma 2**: 最优 scheduler 等价于恒速遍历(constant Fisher-Rao speed)
- **Proposition 1**: κ_t* = F^{-1}(t),其中 F 是归一化的 Fisher-Rao 累积弧长

**为什么是恒速?** [论文原文] 改变 scheduler 只是重参数化同一条几何路径,长度不变但 energy (速度的平方积分) 依赖速度分布。由 Cauchy-Schwarz 不等式,恒速时 energy 取下界 L^2 [§3.1, Eq. 34-36]。

**MI-DFM 的 Fisher information**: I_{x_1}(β) = Var_{x~p(·|x_1;β)}[d(x,x_1)] — 即距离到目标 token 的方差 [§3.2, Eq. 17]。这个结果的物理含义是: 当分布已经高度集中时(方差小、Fisher information 小),scheduler 应该"慢下来";当分布仍然分散时(方差大),应该"快走" [agent 解读]。

**数值构建**: 由于 MI-DFM 的 Fisher information 依赖完整的 token 距离分布,没有解析解。论文通过 Algorithm 1 确定 β_max,Algorithm 2 在 β 网格上计算累积弧长并反插值得到 lookup table {β_j*, β̇_j*} [§3.2]。这使得 scheduler 完全由 codebook embedding 决定,不需要下游训练搜索。

**一致性验证**: 对 mixture path (mask-source DFM),Proposition 1 恢复了已知的闭式解 κ_t = 1 - sin^2((1-t)Ω)/sin^2(Ω) [Appendix D, Eq. 59]。

#### 2. Finite-Step Moment Correction (核心贡献二)

**问题**: 一阶 CTMC solver 在有限步数下产生路径跟踪误差。

**设计哲学**: 不做高阶积分器,不改变 jump destination 分布 π_t,只调整 jump probability ρ [论文原文, §4]。这样做的好处是 (1) 保持 CTMC 结构 (2) 只需一维优化 [agent 解读]。

**具体做法**: 选择一个标量统计量 φ_t(x|x̂_1) 和参考 moment m_{t+h},求解使 post-step moment 匹配参考的 ρ* [§4.1, Eq. 20-22]:

ρ* = [φ_t(z|x̂_1) - m_{t+h}(z,x̂_1)] / [φ_t(z|x̂_1) - φ̄_t(z,x̂_1)]

**MI-DFM 的实例化**: φ_t = ∂_t log p_t (local Fisher-Rao tangent statistic), m_{t+h} = E_{y~p_{t+h}}[φ_t(y|x̂_1)]。代入后 β̇_t 消去,最终 [§4.2, Eq. 26]:

ρ* = [d(z,x̂_1) - E_{y~p_{t+h}}[d(y,x̂_1)]] / [d(z,x̂_1) - E_{y~π_t}[d(y,x̂_1)]]

**为什么 β̇_t 消去很重要?** [agent 解读] 消去意味着 corrected jump probability 不依赖于 scheduler 的导数,仅依赖于期望距离,数值上更稳定。

**一致性验证**: 对 mixture path,使用 target-state indicator moment 时 correction 恢复精确的有限步转移概率 ρ_exact = (κ_{t+h}-κ_t)/(1-κ_t) [Appendix F]。

#### 3. Full-Codebook Training & Codebook-wise Loss

**为什么 full-codebook?** [论文原文, Appendix I] MaskGCT 按 RVQ layer 逐层生成,本文所有 12 层 codebook 在每步同时预测。实验表明 full-codebook 在相同 NFE 下优于 per-layer,因为 per-layer 把步数分散到各层,而 full-codebook 每步联合更新所有层。

**Codebook-wise loss weight**: w_c = 1 - (c-1)/C,线性衰减。早期 codebook 编码 coarse 信息权重更大,后期 codebook 编码残差细节权重更小。论文报告不加 weighting 时梯度爆炸 [Appendix I]。

#### 4. 其他工程选择

- **DiT backbone**: RoPE + SwiGLU + RMSNorm + adaLN-Zero,timestep 和 language embedding concatenate 作为 conditioning [§5, Fig 1]
- **Prompt 构造**: 训练时随机 U(0, 0.3) 比例切分 prefix 为 prompt,加 learnable prompt embedding [§5]
- **Duration predictor**: 改进 MaskGCT 的 rule-based 方案,加入 ratio clipping (γ=0.8) 提升鲁棒性 [Appendix J, Eq. 61-62]
- **CFG**: condition drop rate 0.15, scale 2.5, rescale 0.75 [§6.2]
- **温度**: MI-DFM 用 0.6,masked baselines 用 0.1-0.2 [Appendix K]

### 训练策略

- AdamW, peak LR 2e-4, cosine decay to 10%, linear warmup 5%
- EMA 0.9999
- 10 epochs
- Emilia en (46k hrs) + zh (45k hrs) [§6.2]
- Base: 178M, 768d, 12 layers, 8 H100, 33 hours
- Large: 399M, 1024d, 16 layers, 32 H100, 46 hours [Table 1]

## 实验

### 主实验 (Controlled Comparison, Large)

| 指标 | GibbsTTS (MI-DFM + KO + corr) | MI-DFM w/o corr | Masked DFM (KO) | Masked DFM (DiFlow) | Masked DD (MaskGCT sched) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UTMOS↑ | **3.651** | 3.403 | 3.639 | 3.546 | 3.415 | Seed test-en | [Table 2a] |
| WER↓ | **1.777%** | 2.120% | 1.969% | 1.827% | 2.338% | Seed test-en | [Table 2a] |
| SIM↑ | **0.743** | 0.723 | 0.742 | 0.728 | 0.721 | Seed test-en | [Table 2a] |
| UTMOS↑ | **2.712** | 2.447 | 2.656 | 2.559 | 2.387 | Seed test-zh | [Table 2a] |
| CER↓ | 1.327% | 1.777% | 1.536% | **1.308%** | 1.583% | Seed test-zh | [Table 2a] |
| SIM↑ | **0.790** | 0.775 | 0.788 | 0.785 | 0.776 | Seed test-zh | [Table 2a] |

### SOTA 对比 (Large)

| 指标 | GibbsTTS (0.4B) | MaskGCT (1.5B) | F5-TTS (0.3B) | OmniVoice (0.6B) | Qwen3-TTS (1.7B) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UTMOS↑ | 3.651 | 3.582 | 3.762 | **3.896** | 4.178 | Seed test-en | [Table 4] |
| WER↓ | 1.777% | 3.763% | 2.020% | **1.475%** | 1.434% | Seed test-en | [Table 4] |
| SIM↑ | **0.743** | 0.716 | 0.654 | 0.741 | 0.712 | Seed test-en | [Table 4] |
| SIM↑ | **0.790** | 0.774 | 0.747 | 0.778 | 0.770 | Seed test-zh | [Table 4] |

### 主观评估

| 指标 | GibbsTTS | MI-DFM w/o corr | Grid-searched | Masked DFM (KO) | Masked DD (MaskGCT) | MaskGCT (orig) | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CMOS en↑ | 0 (ref) | -0.362‡ | -0.257‡ | -0.229† | -0.743‡ | -0.762‡ | [Table 3] |
| SMOS en↑ | **4.18** | 3.86‡ | 4.05 | 3.94 | 3.56‡ | 4.06 | [Table 3] |
| CMOS zh↑ | 0 (ref) | -0.459‡ | -0.153 | -0.224† | -0.647‡ | -0.612‡ | [Table 3] |
| SMOS zh↑ | **4.28** | 4.03† | 4.21 | 4.11 | 3.87‡ | 4.19 | [Table 3] |

### Ablation: KO Scheduler vs Grid-searched

KO scheduler 在 Seed-TTS test-en 上 UTMOS +0.034, WER -0.016pp, SIM +0.014 相比 grid-searched (a=5, c=1)。关键是无需搜索,直接从 codebook 距离矩阵计算 [§7]。

### Ablation: Moment Correction

去掉 corrector 后全部指标一致下降(UTMOS -0.248, WER +0.343pp, SIM -0.020 on Seed test-en)。在不同 NFE (16/32/64) 和不同 scheduler 下均观察到一致增益 [Tables 11,12]。

### Ablation: Full-codebook vs Per-layer

Full-codebook (32 NFE) vs Per-layer 最佳配置 (66 NFE): SIM 0.711 vs 0.675, WER 1.961% vs 3.377% on Seed test-en [Table 8]。Full-codebook 在更少 NFE 下实现更好的 intelligibility 和 similarity。

## 局限性

1. **Token distance 未充分探索**: 仅使用 L2 cosine distance(因 codec 的 L2 norm),未尝试其他 metric [§9]
2. **距离矩阵优化**: scheduler 确定后,距离矩阵本身成为下一个优化目标 [§9]
3. **Text frontend 简单**: 使用 G2P-based frontend,影响 WER/CER 指标 [§8]
4. **规模有限**: 0.4B 参数在 UTMOS/WER 上不敌 1.7B Qwen3-TTS [Table 4]
5. **仅验证 TTS**: 算法是通用的但未推广到其他 discrete generation 任务 [§9]
6. **Moment correction 选择空间**: 仅用了 Fisher-Rao tangent statistic,其他 moment 选择可能更优 [§9]

## 点评

**核心价值**: 本文的最大贡献不是"又一个 TTS 系统",而是为 MI-DFM 提供了两个有理论支撑的算法改进。kinetic-optimal scheduler 从 Fisher-Rao geometry 自然推导,moment correction 从有限步误差分析出发,二者都有清晰的数学动机和一致性验证(恢复已知的闭式解)。

**与 DiFlow-TTS 的区别**: DiFlow-TTS 使用 mask-source DFM,概率路径是 (1-κ_t)·mask + κ_t·target,有闭式 KO scheduler。本文使用 MI-DFM,路径是 softmax(-β_t·d(x,x_1)),没有闭式 scheduler,因此需要数值构建。

**与 OmniVoice 的关系**: 两篇同期工作,关注点互补。OmniVoice 用 LLM 初始化 + 多语言 scaling 等系统级设计;本文专注算法层面(scheduler + corrector)。OmniVoice 也用 full-codebook 策略,但用 mask-source DFM 而非 MI-DFM。两者的工程设计可相互迁移。

**Scheduler 解耦**: Table 2 的一个重要发现是 scheduler 选择与解码方式强耦合——适合 masked DFM 的 scheduler 不一定适合 masked DD,反之亦然。这暗示离散生成模型的 scheduler 设计不是 universal 的,需要 formulation-specific 优化。

**Speaker similarity 优势**: GibbsTTS 在 4/4 SOTA 对比 test set 中 SIM 排名第一或第二(仅 CosyVoice 3 en 次于 OmniVoice)。这可能与 MI-DFM 利用 codec embedding 几何结构的特性有关——距离目标 token 近的 token 被优先采样,有助于保持声学一致性 [agent 解读]。

## 可复用的 idea

1. **Fisher-Rao 等速原则**: 任何需要 scheduler 的离散生成模型都可考虑 — 在 token embedding 空间计算距离方差作为 Fisher information,数值构建等速 scheduler。特别适用于 VQ-based 生成任务(图像、视频、audio)。

2. **不改方向只调概率的 corrector**: 轻量级 finite-step 修正通用框架。保留 jump destination π_t 不变,仅调 ρ ∈ [0,1],单维最小二乘即可。可直接应用于任何 CTMC-based discrete sampler。

3. **Full-codebook + linearly decayed loss weight**: 多层 RVQ 的联合建模方案。w_c = 1-(c-1)/C 简单有效,避免了 per-layer 训练+推理的复杂度,且在相同 NFE 下性能更好。

4. **Duration ratio clipping**: r = clip(r_prompt, γ·r̄, r̄/γ),用训练集均值限制 prompt-derived ratio,γ=0.8 作为通用默认值。

## 审阅

