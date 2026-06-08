---
type: paper
tier: deep
title: "Moonlight: Muon is Scalable for LLM Training"
arxiv_id: "2502.16982"
source: "Sources/Moonlight.pdf"
authors: [Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, Yanru Chen, Huabin Zheng, Yibo Liu, Shaowei Liu, Bohong Yin, Weiran He, Han Zhu, Yuzhi Wang, Jianzhou Wang, Mengnan Dong, Zheng Zhang, Yongsheng Kang, Hao Zhang, Xinran Xu, Yutao Zhang, Yuxin Wu, Xinyu Zhou, Zhilin Yang]
year: 2025
venue: "arXiv"
tags: [LLM, optimizer, Muon, MoE, scaling-law, distributed-training, AdamW, matrix-orthogonalization, training-efficiency]
concepts: []
models: []
tasks: [language-modeling, LLM-pretraining]
datasets: []
kb_context_sources: 0
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 0 个直接相关已确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: 无直接匹配 | 间接相关: [[SpeechLanguageModel]]、[[LLM-basedTTS]] (Muon 优化器可用于训练 speech LM 的 backbone LLM,但本文不涉及语音) | 未命中但可能相关: MoE (无独立概念页)

- **间接相关性**: 本文聚焦 LLM 优化器研究和 MoE 模型训练,不涉及语音/TTS。但 Muon 作为通用 LLM 优化器,理论上可用于训练 speech LM 的 backbone (如 Qwen、Llama 系列),其 ~2x 训练效率提升对大规模 speech foundation model 训练具有实际价值。
- **MoE 架构**: Moonlight 采用 DeepSeek-V3-Small 架构 (2.24B activated / 15.29B total),MoE 路由器权重在 Muon 下的 SVD entropy 提升最为显著 [§3.4],暗示 MoE 架构特别受益于 Muon 优化。这对 Step-Audio 等采用 MoE 架构的 speech 系统有参考意义。

> [!summary] 速查
> - **一句话**: 首次证明 Muon 优化器可扩展到大规模 LLM 训练,通过添加 weight decay 和调整 per-parameter update scale 实现 ~2x 计算效率优于 AdamW,训练出 3B/16B MoE 模型 Moonlight 推进 Pareto frontier [论文原文]
> - **路线**: Muon optimizer (gradient momentum → Newton-Schulz orthogonalization → weight update) + weight decay + sqrt(max(A,B)) update RMS scaling → Distributed Muon (ZeRO-1 style) → 3-stage pretraining (warmup → cosine decay → cooldown) [§2, §3.3]
> - **指标**: Moonlight@5.7T: MMLU 70.0 / HumanEval 48.1 / GSM8K 77.4 (vs DSV2-Lite MMLU 58.3 / HumanEval 29.9 / GSM8K 41.1, 同参数同 token 量); Scaling law: Muon 仅需 ~52% FLOPs 达到 AdamW 同等性能 [Table 5, Fig 1a]
> - **可借鉴**: (1) 通过 sqrt(max(A,B)) scaling 统一不同形状矩阵的 update RMS,消除超参数调节; (2) 将 Muon update RMS 对齐 AdamW 范围 (0.2) 使 LR/WD 直接复用; (3) Distributed Muon 的 ZeRO-1 集成仅增加 ~0-25% 通信开销; (4) SVD entropy 作为优化器质量的诊断指标
> - **局限**: (1) 预训练-微调优化器不匹配时性能下降,无法直接利用现有 AdamW pretrained checkpoints [§4]; (2) 非矩阵参数仍需 AdamW 处理; (3) 未开源训练数据,复现依赖 Kimi 内部数据集

## 核心问题

Muon 优化器在小规模 LM 训练上表现优异,但扩展到大规模 (数十亿参数 + 万亿 token) 面临三个未解决的挑战 [§1]:

1. **可扩展性**: 如何将基于 matrix orthogonalization 的优化器有效扩展到数十亿参数? 直接使用 vanilla Muon 训练大模型时,weight 和 layer output 的 RMS 持续增长到超出 bf16 精度范围 [§2.2] [论文原文]
2. **分布式计算**: Muon 的 Newton-Schulz 迭代需要完整梯度矩阵,与 ZeRO-1 的参数分片策略冲突。如何在分布式环境中高效计算近似正交化? [§1] [论文原文]
3. **阶段通用性**: Muon 能否泛化到不同训练阶段 (预训练 + SFT)? [§1] [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Moonlight 不是一个新架构,而是在 DeepSeek-V3-Small 架构上用改进后的 Muon 优化器训练的 MoE 模型。核心贡献在优化器层面。

**Muon 基础** [§2.1]: 对矩阵参数 W,Muon 用 Newton-Schulz 迭代近似计算梯度动量的正交化:
```
M_t = μ·M_{t-1} + ∇L_t(W_{t-1})      # 梯度动量
O_t = Newton-Schulz(M_t)               # 正交化: UV^T (SVD)
W_t = W_{t-1} - η_t · O_t              # 参数更新
```
正交化的直觉: 确保更新矩阵是 isomorphic 的,防止权重只沿少数主方向学习 [§2.1] [论文原文]。

**理论视角**: Bernstein et al. (2024) 将深度学习优化视为范数约束下的 steepest descent。Adam 的范数约束来自动态调整的 Max-of-Max norm; Muon 提供的是 spectral norm (Schatten-p norm 的极限)。对于作用在 Euclidean 空间上的权重矩阵,spectral norm 作为 induced operator norm 比 Adam 的约束更合理 [§2.1] [论文原文]。

### 关键设计选择

#### 1. Weight Decay [§2.2]

**问题**: Vanilla Muon 在小规模表现好,但扩展时性能优势减弱。原因: weight 和 layer output 的 RMS 持续增长,超出 bf16 精度范围 [§2.2] [论文原文]。

**解决方案**: 引入 AdamW 式 weight decay:
```
W_t = W_{t-1} - η_t · (O_t + λ·W_{t-1})
```

**WHY weight decay 特别重要于 Muon**: Adam/AdamW 的 second moment normalization 天然抑制 weight growth (update ∝ sign(gradient)),但 Muon 的正交化更新没有这种隐式正则化效果 -- 正交化保证 update 的方向均匀分布,但不限制 weight 的绝对大小 [agent 解读]。

**实证验证** [Fig 2]: 800M 模型训 100B token (5x compute-optimal)。Vanilla Muon 前期收敛更快,但后期被 Muon+WD 反超。在 overtrain regime (iteration 66K) Muon+WD 的 val loss 比 vanilla Muon 低 0.017。

#### 2. Consistent Update RMS [§2.2, §3.1]

**问题**: Muon 的 update RMS 依赖矩阵形状:

**Lemma 1**: 对形状 [A, B] 的满秩矩阵参数,Muon 的理论 update RMS = 1/sqrt(max(A,B)) [§2.2, Appendix A 有证明]。

这导致:
- max(A,B) 太大时 (如 dense MLP 矩阵), 更新过小 → 表达能力受限 [论文原文]
- max(A,B) 太小时 (如 GQA/MLA 的 KV head), 更新过大 → 训练不稳定 [论文原文]

**解决方案 (Adjusted LR)** [Eq. 7]:
```
W_t = W_{t-1} - η_t · (0.2 · O_t · sqrt(max(A,B)) + λ·W_{t-1})
```

三个设计决策叠加:
1. **sqrt(max(A,B))** 消除形状依赖的 update RMS 差异 (Lemma 1 的逆操作)
2. **0.2** 将 Muon 的 update RMS 对齐到 AdamW 的经验范围 (0.2~0.4) [Table 8, Appendix A]
3. **共享 η 和 λ**: 对齐后 Muon 可直接复用 AdamW 的最优 LR 和 weight decay,无需额外调参 [§2.2] [论文原文]

**WHY 选 Adjusted LR 而非 Update Norm** [§3.1, Table 1]: 两者性能相当,但 Adjusted LR 计算成本更低 (不需要额外的 RMS 计算)。Update Norm 强制归一化所有矩阵的 update RMS = 0.2,而 Adjusted LR 仅对形状差异做补偿,保留了 shape [H, H] 矩阵的原始行为。

#### 3. Distributed Muon [§2.3]

**问题**: ZeRO-1 将 optimizer states 分片到 DP workers,但 Muon 的 Newton-Schulz 需要完整梯度矩阵。

**解决方案** (Algorithm 1): 在 vanilla ZeRO-1 基础上增加两步 (蓝色标注):
1. **DP Gather**: 将分片梯度 gather 成完整梯度矩阵 (bf16)
2. **Full Update**: 对完整矩阵做 Newton-Schulz,然后丢弃非本地分片对应的更新

**开销分析**:
- **Memory**: Muon 仅需 1 个 momentum buffer (Adam 需 2 个) → 优化器状态内存为 AdamW 的一半 [§2.3] [论文原文]
- **Communication**: 额外 gather 仅需处理本地 1/DP 的参数量,且 bf16 进一步减半 → 通信开销为 AdamW 的 [1, 1.25] 倍,实践中接近 1x [§2.3] [论文原文]
- **Latency**: Newton-Schulz 仅需 5 步迭代,端到端优化器时间占 forward-backward 的 1-3%,可忽略 [§2.3] [论文原文]

### 训练策略

**模型**: DeepSeek-V3-Small 架构,2.24B activated / 15.29B total 参数 (含 embedding 为 3B/16B) [§3.3]。微调: 无 MTP 层; 修改 auxfree bias 更新为零均值版本; gate scaling factor 2.446 [Appendix C]。

**三阶段预训练** [§3.3]:
1. **Stage 1** (0→33B tokens): LR warmup 至 4.2e-4 (2K steps), batch size 2048, auxfree bias update rate 1e-3
2. **Stage 2** (33B→5.2T tokens): LR cosine decay 4.2e-4→4.2e-5, batch size 2048 (200B后翻倍至4096), auxfree bias update rate 1e-3
3. **Stage 3 / Cooldown** (5.2T→5.7T tokens): LR bump至1e-4再线性衰减至0 (500B tokens), batch size 4096, 聚焦 math/code/reasoning 高质量数据, auxfree bias rate 0.0

Weight decay 全程 λ = 0.1。

**训练稳定性** [Appendix D]:
- 无 loss spike 或 gradient norm spike [Fig 7a, 7b]
- Max attention logit 在初始阶段上升至 >100,但 large attention logits ratio 维持在 ~1e-4,随训练逐渐下降 [Fig 7c, 7d]
- RMSNorm gamma 的 weight decay 是训练稳定性的关键 [Appendix D] [论文原文]

## 实验

### Scaling Law [§3.2, Fig 3, Table 3]

| 模型规模 | Params (w/o embed) | Tokens | LR | Batch Size |
| --- | --- | --- | --- | --- |
| 399M | 399M | 8.92B | 9.503e-4 | 96 |
| 545M | 545M | 14.04B | 9.143e-4 | 128 |
| 822M | 822M | 20.76B | 8.825e-4 | 160 |
| 1.1B | 1.1B | 28.54B | 8.561e-4 | 192 |
| 1.5B | 1.5B | 38.91B | 8.305e-4 | 256 |

拟合结果: Muon LM loss = 2.506 x C^{-0.052}; AdamW LM loss = 2.608 x C^{-0.054}。Muon 仅需 ~52% 的训练 FLOPs 达到 AdamW 的同等 loss [Fig 1a] [论文原文]。

### 同架构 1.2T 对比 [§3.3, Table 4]

| 指标 | DSV3-Small (1.33T, AdamW) | Moonlight-A@1.2T (AdamW) | Moonlight@1.2T (Muon) | 出处 |
| --- | --- | --- | --- | --- |
| MMLU | 53.3 | 60.2 | 60.4 | Table 4 |
| MMLU-pro | 41.4 | 26.8 | 28.1 | Table 4 |
| HumanEval | 26.8 | 29.3 | **37.2** | Table 4 |
| MBPP | 36.8 | 49.2 | **52.9** | Table 4 |
| GSM8K | 31.4 | 43.8 | **45.0** | Table 4 |
| MATH | 10.7 | 16.1 | **19.8** | Table 4 |

Muon 在 Math 和 Code 任务上优势尤为突出: HumanEval +7.9, MATH +3.7 (vs Moonlight-A) [论文原文]。

### 完整训练 5.7T 对比 [§3.3, Table 5]

| 指标 | Llama3.2-3B (9T) | Qwen2.5-3B (18T) | DSV2-Lite (5.7T, AdamW) | Moonlight (5.7T, Muon) | 出处 |
| --- | --- | --- | --- | --- | --- |
| MMLU | 54.7 | 65.6 | 58.3 | **70.0** | Table 5 |
| HumanEval | 28.0 | 42.1 | 29.9 | **48.1** | Table 5 |
| MBPP | 48.7 | 57.1 | 43.2 | **63.8** | Table 5 |
| GSM8K | 34.0 | 79.1 | 41.1 | **77.4** | Table 5 |
| MATH | 8.5 | 42.6 | 17.1 | **45.3** | Table 5 |

Moonlight 以 2.24B activated / 5.7T tokens 超越同架构 DSV2-Lite (AdamW),并与使用 3x token 量的 Qwen2.5-3B 竞争 [论文原文]。

### 与更大 dense 模型对比 [Appendix E, Table 10]

| 指标 | Moonlight (2.24B act, 5.7T) | Llama3.1-8B (7.38B, 15T) | Gemma2-9B (8.32B, 8T) | Qwen2.5-7B (6.83B, 18T) | 出处 |
| --- | --- | --- | --- | --- | --- |
| MMLU | 70.0 | 66.7 | 71.3 | 74.2 | Table 10 |
| HumanEval | **48.1** | 37.2 | 37.8 | 57.9 | Table 10 |
| GSM8K | **77.4** | 57.2 | 70.7 | 85.4 | Table 10 |

Moonlight 在 HumanEval 和 GSM8K 上超越 3x 参数量的 Llama3.1-8B 和 Gemma2-9B [论文原文]。

### SVD Entropy 分析 [§3.4, Fig 4]

Muon 优化的权重矩阵在所有训练检查点和所有权重组中 SVD entropy 均高于 AdamW [Fig 4]。差异在 **Router weights** 上最为显著,表明 MoE 模型的路由机制特别受益于 Muon 的均匀方向更新 [§3.4] [论文原文]。

超过 90% 的权重矩阵在 Muon 下有更高的 SVD entropy [§3.4, Appendix F],为 Muon "在更多样化的方向上优化"的直觉提供了实证支持。

### SFT Ablation [§3.5, Table 6]

| Pretrain Optimizer | SFT Optimizer | MMLU (0-shot CoT) | HumanEval | GSM8K (5-shot) | 出处 |
| --- | --- | --- | --- | --- | --- |
| Muon | Muon | **55.7** | **57.3** | **68.0** | Table 6 |
| AdamW | Muon | 55.3 | 53.7 | 62.1 | Table 6 |
| Muon | AdamW | 50.2 | 52.4 | 64.9 | Table 6 |
| AdamW | AdamW | 52.0 | 53.1 | 64.6 | Table 6 |

关键发现: Muon-pretrained + Muon-SFT 性能最优; 预训练和 SFT 使用不同优化器时,Muon SFT 不优于 AdamW SFT [§3.5] [论文原文]。这暗示优化器在权重空间中创造了不同的 "landscape topology",跨优化器微调无法充分利用预训练阶段的优化方向 [agent 解读]。

## 局限性

1. **预训练-微调优化器不匹配** [§4]: AdamW pretrained checkpoint 无法通过 Muon SFT 获得额外收益,反之亦然。这意味着现有大量 AdamW pretrained models 无法直接受益于 Muon,必须从头训练 [论文原文]
2. **非矩阵参数仍需 AdamW** [§4]: RMSNorm, LM head, embedding 等不能用 Muon 优化,导致混合优化器的复杂性。将所有参数统一到 Muon 框架是开放问题 [论文原文]
3. **Scaling law 验证范围有限**: 仅在 399M-1.5B dense models 上拟合 scaling law,MoE 架构的 scaling law 未单独验证 [agent 解读]
4. **训练数据未开源**: 预训练数据集引用 Kimi k1.5 内部数据,复现无法使用相同数据 [§3.3]
5. **SFT 场景收益不明确** [§3.5.2, Table 7]: 在 Qwen2.5-7B (AdamW pretrained) 上 Muon SFT 与 Adam SFT 性能持平,Muon 的收益主要来自预训练阶段

## 点评

**优化器研究的扎实工程文** Moonlight 的核心贡献不在于提出新算法,而在于系统性地解决 Muon 从玩具规模到工业规模的所有 engineering barriers: weight growth → weight decay; shape-dependent update → sqrt(max(A,B)) scaling; distributed incompatibility → ZeRO-1 集成。每个解决方案都有清晰的动机-方案-验证闭环。

**~2x FLOPs 效率的含义** 如果 scaling law 结论成立,Muon 相当于将训练预算翻倍。对于一个 $100M 的训练 run,这意味着节省 $50M 或在同等预算下训练出更强模型。但需注意: 这个倍率来自 dense model 的 scaling law 拟合,MoE 上的实际倍率未单独验证。

**SVD entropy 分析是重要的诊断工具** Fig 4 不仅验证了 Muon 的直觉,还揭示了 MoE router weights 是最大受益者。这提示 MoE 路由的多样性可能是被 AdamW 系统性低估的维度。这个发现比主结果更有启发性 [agent 解读]。

**预训练-SFT 不匹配问题值得关注** Table 6 的 ablation 表明,优化器选择并非独立于训练阶段 -- 整个训练流程需要 optimizer-consistent。这对 open-source 生态是个挑战: 如果 Muon 成为 pretrain 标准,那么社区需要重新开发 Muon-native 的 SFT/RLHF 方法 [agent 解读]。

## 可复用的 idea

1. **Update RMS 对齐策略** [§2.2, Eq. 4]: 将新优化器的 update RMS 对齐到 Adam 的范围,使得所有超参数可直接复用,避免大规模重新调参。这个策略可迁移到任何替代优化器的开发中。

2. **sqrt(max(A,B)) per-parameter scaling** [Lemma 1, Eq. 7]: 消除矩阵形状对 update scale 影响的通用技巧。任何使用矩阵级别操作的优化器 (如 Shampoo, SOAP) 都可能面临类似问题。

3. **SVD entropy 作为优化诊断指标** [§3.4]: 比较不同优化器/训练策略对权重矩阵奇异值分布的影响。可用于诊断 speech tokenizer 的 codebook 利用率、attention 矩阵的多样性等。

4. **Distributed Muon 的 ZeRO-1 集成模式** [Algorithm 1]: "先 gather → 全矩阵操作 → 丢弃非本地分片" 的模式,可迁移到需要全局信息的其他分布式算法 (如分布式 spectral normalization)。

5. **Cooldown stage 数据配方** [§3.3 Stage 3]: 最后 500B tokens 使用高质量 math/code/reasoning 数据 + LR bump,这个 "annealing on curated data" 策略在 Moonlight 的 math/code 性能跃升中可能扮演重要角色 [agent 解读]。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 核心方法 (weight decay + update RMS scaling + distributed Muon) 有完整数学推导和算法伪代码 |
> | 可信赖 | pass | [论文原文] vs [agent 解读] 标注清晰; scaling law 拟合公式和实验配置完整; 局限性部分涵盖了 optimizer mismatch 等关键问题 |
> | 可区分 | pass | 与 vanilla Muon、AdamW 的多维度对比清晰; 与 DSV3-Small/DSV2-Lite/Llama/Qwen 的 benchmark 对比完整 |
> | 可定位 | pass | 明确标注本文为 LLM 优化器研究,非语音/TTS 方向; KB 背景说明间接相关性 |
> | 不污染 | pass | 未将论文未涉及的 speech 领域信息混入方法描述; 所有跨域推断标注 [agent 解读] |
> 
> Issues: 1 (high: 0, medium: 0, low: 1)
> 详见 `_review/Moonlight-review.yml`
