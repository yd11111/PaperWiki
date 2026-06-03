---
type: paper
tier: deep
title: "LLaDA-TTS: Unifying Speech Synthesis and Zero-Shot Editing via Masked Diffusion Modeling"
arxiv_id: "2603.26364"
source: "Sources/LLaDA-TTS.pdf"
authors: [Xiaoyu Fan, Huizhi Xie, Wei Zou, Yunzhang Chen]
year: 2026
venue: "arXiv preprint"
tags: [TTS, masked-diffusion, non-autoregressive, zero-shot, speech-editing, discrete-diffusion, LLM-based]
concepts: ["[[Masked Generative Modeling]]", "[[LLM-based TTS]]", "[[Non-autoregressive TTS]]", "[[Diffusion Model]]", "[[Conditional Flow Matching]]", "[[Semantic vs Acoustic Tokens]]", "[[Speech Tokenizer]]"]
models: ["[[CosyVoice 3]]", "[[模型库/Whisper|Whisper]]", "[[模型库/WavLM|WavLM]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: LLaDA-TTS 位于 [[LLM-based TTS]] 范式的 NAR 分支。主流 LLM-based TTS (VALL-E, Seed-TTS, CosyVoice 系列) 均采用自回归 decoder-only transformer 生成离散 speech tokens,然后用 vocoder 合成波形。LLaDA-TTS 保留了这一 pipeline 的 tokenizer + vocoder 部分,仅将 AR LLM 替换为 masked discrete diffusion model,属于对 LLM-based TTS 的 "解码策略替换" 而非 "架构重设计"。
>
> **已有认知**:
> - [[Masked Generative Modeling]] [待确认] 页记录了 MaskGIT-style mask-and-predict 范式在 TTS 中的应用 (SoundStorm 用于 S2A, MaskGCT 扩展到 T2S+S2A),但 LLaDA-TTS 使用的是基于 ELBO 的 1/t-weighted masked diffusion objective (来自 LLaDA/Dream),与 MaskGIT-style confidence-based decoding 有理论区别。
> - [[CosyVoice 3]] [待确认] 是 LLaDA-TTS 的直接 baseline (Qwen2-0.5B backbone + FSQ tokenizer + CFM vocoder)。CosyVoice 3 在 SEED-TTS-Eval 上报告 CER 0.71% (zh) / WER 1.45% (en),LLaDA-TTS 论文中引用的 baseline 数据为 CER 1.21% (zh) / WER 2.24% (en),对应的是 CosyVoice 3-0.5B 的未经 DiffRO post-training 的版本。
> - [[Non-autoregressive TTS]] [待确认] 页覆盖了传统 NAR (FastSpeech/VITS) 和 masked generation (MaskGCT),但尚未涵盖 masked discrete diffusion 这一新子类。
> - [[Diffusion Model]] [待确认] 页主要覆盖连续空间 diffusion (DDPM, SDE),对离散空间 diffusion (D3PM, MDLM, LLaDA) 的覆盖有限。
>
> **创新判断**: LLaDA-TTS 的核心贡献在于证明 AR-pretrained weights 可以高效迁移到 masked diffusion 范式 (仅需 50 小时微调数据 + bidirectional attention + label shift),并给出了理论解释 (epsilon-forward dependence)。这与 MaskGCT 的区别在于: MaskGCT 从头训练,使用 MaskGIT-style confidence decoding;LLaDA-TTS 从 AR checkpoint 初始化,使用 ELBO-derived 1/t objective。同时 LLaDA-TTS 作为 "method paper" 展示了该方法可推广到任意 LLM-based AR TTS 系统。
>
> 检索命中: [[LLM-based TTS]]✓, [[SEED-TTS-Eval]]✓ | 过滤: [[CosyVoice 3]](pending-review), [[Masked Generative Modeling]](pending-review), [[Non-autoregressive TTS]](pending-review), [[Diffusion Model]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 AR-pretrained TTS LLM 转换为 masked discrete diffusion decoder,仅改 attention mask + 训练目标,50 小时数据即可匹配原 AR 性能并获得 2x 加速 + 免训练 speech editing。
> - **路线**: Text + Prompt → [Qwen2-0.5B, bidirectional attention + 1/t masked diffusion] → Discrete Speech Tokens (T 步 iterative unmasking) → [Flow Matching + HiFi-GAN (CosyVoice 3)] → Waveform
> - **指标**: CER 0.98% (zh) / WER 1.96% (en) @ 64 steps on SEED-TTS-Eval,匹配 CosyVoice 3-0.5B baseline (CER 1.21%)，2x LLM-stage speedup [Table 1, Fig 3]
> - **可借鉴**: AR→masked diffusion 的迁移配方 (bidirectional attention + label shift + 1/t objective + AR init) 可直接应用于任何 LLM-based AR TTS 系统;attention head 中涌现的 text-speech alignment 可用于免训练 speech editing
> - **局限**: 需预先指定输出长度 (token-text ratio)；非流式；无 KV cache 所以单步成本高于 AR (靠步数少补偿)；仅在 CosyVoice 3 一个 backbone 上验证

## 核心问题

LLM-based TTS 的 AR 解码器需要 N 步生成 N 个 speech token,推理延迟随输出长度线性增长。能否将 AR LLM 替换为并行生成的 masked diffusion model,在保持语音质量的同时解耦推理成本与序列长度?进一步地,能否利用 AR 预训练的知识加速收敛,并利用 bidirectional attention 的副产品实现免训练 speech editing?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

LLaDA-TTS 保留标准 LLM-based TTS pipeline (supervised speech tokenization → LLM-based token generation → flow matching vocoder),仅替换 LLM 的解码策略 [§3]:
- **Tokenizer + Vocoder**: 完全沿用 CosyVoice 3 的 FSQ speech tokenizer (25 Hz) 和 flow matching + HiFi-GAN vocoder,不做任何修改 [§3, Fig 1]
- **LLM backbone**: Qwen2-0.5B,从 CosyVoice 3 的 AR checkpoint 加载所有参数 [§3.2]
- **核心修改**: (1) causal attention mask → bidirectional attention mask; (2) AR next-token prediction → 1/t-weighted masked diffusion objective [§3.1, §3.2]

序列格式不变: `[SOS, t_instruct, t_text, SEP, s_prompt, s_target, EOS]`,其中只有 `s_target` 参与 masking,text/prompt/special tokens 始终可见 [§3.1]。

### 关键设计选择

**1. Label Shift -- 从 AR 到 bidirectional 的桥梁**

LLaDA-TTS 保留 AR 的预测约定: hidden state i 预测 token i+1,通过 logit 位移实现 `y_hat_i = logits_{i-1}` [§3.1]。[论文原文] 这是 Dream (Ye et al., 2025) 引入的技巧,使得 AR 预训练的 output projection 可以直接复用,无需重新学习预测映射。消融实验证实这是关键: 去掉 label shift 后 CER 从 0.98% 恶化到 1.45%,zh-hard 从 7.04% 恶化到 12.33% [Table 2]。

[agent 解读] Label shift 的本质是保持 weight geometry 的一致性。AR 模型中 position i 的 hidden state 经过长期训练已经学会 "从左侧上下文预测下一个 token" 的映射。如果不做 label shift 就改为 bidirectional attention,hidden state 的含义会发生错位,output projection 矩阵需要从头学习新的映射关系,解释了消融中的性能差距。

**2. AR 初始化 -- 为什么 50 小时就够**

所有参数从 AR checkpoint 加载,仅新增 `[MASK]` token embedding 随机初始化 [§3.2]。[论文原文] 作者通过 Theorem 1 (Bounded Suboptimality) 给出理论解释: 在 epsilon-forward dependence 假设下 (语音 token 的因果方向主导信息流),AR predictor 的 per-token KL divergence 与 optimal bidirectional predictor 之差有界于 epsilon [§3.3, Eq. 3]。

[论文原文] 物理直觉: 在 25 Hz frame rate 下 (40 ms/frame),发声器官参数 (舌位、唇形、声门状态) 的变化时间尺度为 50-100 ms,连续帧高度冗余,因果方向主导信息流 [§3.3]。因此 AR 模型的 left-to-right 预测已经接近 bidirectional optimal,只需少量微调适应 low-masking-rate 区间的 bidirectional context。

消融数据: AR init + label shift = 0.98% CER; AR init, no label shift = 1.45%; from scratch = 45.27%; AR weights 不训练 = 99.97% [Table 2]。

**3. 1/t-weighted Masked Diffusion Objective**

训练时对每个 sample 采样 t ~ U(0,1),以概率 t 独立 mask 每个 target speech token,计算 1/t-weighted cross-entropy loss (仅在 masked positions 上) [§3.2, Eq. 1]:

```
L(theta) = E_t [ (1/t) * sum_{i: s_t^i = [MASK]} -log p_theta(s_0^i | s_t) ]
```

[论文原文] 1/t weighting 对应 LLaDA (Nie et al., 2025) 推导的 ELBO,upweight 接近完成时的预测 (t 小,每个剩余 token 携带更多信息) [§3.2]。

[agent 解读] 这与 MaskGIT-style 的 uniform cross-entropy loss 不同。1/t weighting 让模型在 "大部分 token 已 unmask" 的阶段投入更多学习能力,这也恰好是 AR init 优势最小的区间 (更多 bidirectional context 可用),形成互补。

**4. 推理: T 步 Iterative Unmasking**

从全 mask 序列出发,T 步完成生成 (Algorithm 1) [§3.4]:
1. 对所有 masked positions 执行一次 forward pass,预测 logits
2. Temperature-scaled nucleus sampling (tau=0.986, top-p=0.586) 得到候选 token
3. 计算 confidence (top-k margin: p1(i) - p2(i), confidence temperature tau_c=0.424)
4. Unmask top-K_k most confident positions (线性 schedule)
5. 总推理成本 = T 次 forward pass,与序列长度无关

超参数由 300 次 Optuna trials 选定 [§3.4]。

### 训练策略

- **数据**: 6,000 小时 Emilia 数据集 (58% 中文, 42% 英文, 250 万句) [§3.2]
- **硬件**: 7x A100 GPU, DDP [§3.2]
- **优化**: Adam, lr=1e-5, gradient clipping=5.0, FP16 mixed precision [§3.2]
- **微调数据量**: 论文摘要称 "only 50 hours of fine-tuning data",但 §3.2 描述的是 6000 小时训练集 [agent 解读] 推测 50 小时指有效训练 compute 而非数据规模,或指快速收敛所需的最少数据量

### 涌现现象: AR-like Unmasking 与 Attention Alignment

**AR-like 左到右 unmasking**: 尽管使用完全 bidirectional attention,unmasking 过程呈现主要的 left-to-right progression [Fig 4]。序列起始位置的 token 最先被 unmask,形成向右扫过的 "wavefront"。但并非严格顺序: 韵律边界、静音等高置信位置提前 unmask,歧义 token 延后 [§4.5]。

[论文原文] Theorem 1 + Corollary 1 提供解释: 左侧 token 受益于始终可见的 text + prompt prefix,提供强 left conditioning → low KL → high confidence → early unmasking → 级联 wavefront 效应 [§4.5]。

**Attention-based alignment 涌现**: Layer 11 的 H1 和 H5 自发发展出精确的 text-to-speech monotonic alignment,MAE=1.29 tokens (约 52 ms @ 25 Hz),远优于 proportional alignment (MAE=2.99) [§4.5, Fig 5]。这在 causal masking 下不可能出现。

[agent 解读] 这一涌现可能源于 bidirectional attention 让 text token 和 speech token 的交互不再受因果约束,某些 head 自然发展出 "对齐" 功能,类似 Tacotron 中 attention 的功能但无需显式监督。

### Speech Editing (零成本副产品)

基于 bidirectional architecture 的 speech editing pipeline [§3.5, Fig 2]:
1. 用 LLaDA-TTS 特定 attention head (L16-H2 或 L11-H5) 的 attention 进行 text-to-speech alignment
2. 根据文本编辑操作,mask 受影响区域 + context margins (substitution: C=5 tokens, insertion/deletion: S=3 tokens)
3. 冻结周围 token,通过 iterative unmasking 重新生成被 mask 区域

[论文原文] 数学上等价于计算 posterior P(s_edit | s_prefix, s_suffix, t_new),bidirectional attention 在每步自然 marginalize over 两侧 context [§3.5]。

## 实验

| 指标 | 本文 (64 steps) | CosyVoice 3-0.5B (AR) | MaskGCT (NAR) | F5-TTS (Flow) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| CER (%) test-zh | **0.98** | 1.21 | 2.27 | 1.52 | SEED-TTS-Eval | [Table 1] |
| WER (%) test-en | **1.96** | 2.24 | 2.62 | 2.00 | SEED-TTS-Eval | [Table 1] |
| CER (%) test-zh-hard | 7.04 | **6.71** | 10.27 | 8.67 | SEED-TTS-Eval | [Table 1] |
| SS (%) test-zh | 74.6 | **78.0** | 77.4 | 74.1 | SEED-TTS-Eval | [Table 1] |
| SS (%) test-en | 71.1 | **71.8** | 71.4 | 64.7 | SEED-TTS-Eval | [Table 1] |
| LLM-stage RTF (A100) | ~0.15 (64步) | ~0.30 (est.) | — | — | — | [Fig 3] |

**Speed-quality tradeoff** [Fig 3]:
- 48 steps: CER 1.09%, 超越 AR baseline, ~2.6x LLM-stage speedup
- 64 steps: CER 0.98%, 1.9x speedup
- 96 steps: CER 0.74%, 1.4x speedup

**消融** [Table 2] (mini set):

| 配置 | zh CER | zh-hard CER | en WER |
| --- | --- | --- | --- |
| AR init + label shift (full) | **0.98** | **7.04** | **1.96** |
| AR init, no label shift | 1.45 | 12.33 | 2.30 |
| From scratch | 45.27 | 68.25 | 62.58 |
| AR weights only (no training) | 99.97 | 99.94 | 99.97 |

## 局限性

1. **需预指定输出长度**: 通过 token-text ratio 估算,无法动态确定 [§5]
2. **非流式**: 当前实现不支持 streaming inference [§5]
3. **无 KV cache**: Bidirectional attention 不支持 KV cache,单步 forward pass 成本高于 AR,靠减少总步数补偿 [§4.3]
4. **Speaker similarity 略低于 AR**: SS 下降 3-4 个百分点 (74.6 vs 78.0 test-zh),说明 bidirectional unmasking 可能损失部分说话人一致性 [Table 1]
5. **仅验证 CosyVoice 3 一个 backbone**: 虽声称方法通用于任何 LLM-based AR TTS,但未在 VALL-E/Seed-TTS/Spark-TTS 等其他系统上验证 [§5]
6. **理论假设的局限**: epsilon-forward dependence 在 expressive speech (大幅韵律变化) 或极低 token rate 下可能不成立 [agent 解读]

## 点评

**核心价值**: LLaDA-TTS 最大的贡献不是 "又一个 NAR TTS",而是证明了 **AR→masked diffusion 的迁移是高效的** -- 仅改 attention mask + objective,不改 tokenizer/prompt/vocoder,50 小时微调即可收敛。这为整个 LLM-based TTS 社区提供了一条低成本加速路径。

**理论与实践的结合**: Theorem 1 (epsilon-forward dependence → AR prediction near-optimal for bidirectional) 虽然在严格意义上是一个较强的假设 (需要对所有 partial past observation 成立),但为 AR→diffusion 迁移提供了直觉上令人信服的解释,也与涌现的 left-to-right unmasking 行为一致。

**与 MaskGCT 的定位差异**: MaskGCT 是从头设计的 masked generative TTS,需要完整训练;LLaDA-TTS 是 "AR 模型的即插即用改造",强调的是迁移效率而非架构创新。两者在方法论上属于不同路线。

**Speech editing 的实用性**: 利用涌现的 attention alignment 实现免训练 speech editing 是一个 elegant 的副产品,但实用价值取决于 alignment 精度 (MAE=52ms 在 word-level editing 中可能引入可感知的边界伪影)。

**存疑点**: (1) 摘要称 "50 hours fine-tuning data" 但方法节描述 6000 小时训练集,表述有歧义; (2) Speaker similarity 的下降 (3-4%) 在实际应用中可能显著; (3) 与 CosyVoice 3 的完整版 (含 DiffRO post-training, CER 0.71%) 对比缺失。

## 可复用的 idea

1. **AR→Masked Diffusion 迁移配方**: bidirectional attention + label shift + 1/t objective + AR checkpoint init。任何 LLM-based AR TTS 系统只需这 4 项改动就可转为 NAR masked diffusion decoder,无需修改 tokenizer/vocoder。这个配方的 "最小改动原则" 值得借鉴。

2. **Attention head 作为免费 alignment 工具**: 在 bidirectional transformer 中,特定 layer-head 自发涌现 monotonic text-speech alignment。可用于 speech editing、duration control、可视化分析等,无需外部 forced alignment 工具 (如 MFA)。

3. **epsilon-forward dependence 假设**: 为评估 "某个序列是否适合从 AR 迁移到 bidirectional diffusion" 提供了理论框架。如果目标模态的 token 满足 "因果方向主导信息流",则 AR init 就是一个好的起点。这可推广到 audio/video/motion 等其他序列生成任务。

4. **Optuna 超参数搜索用于 sampling 策略**: 用 300 trials 系统搜索 temperature + top-p + confidence temperature 三个推理超参,比手动调参更可靠。

---

> [!review] 审阅状态
> 待审阅。见 `_review/LLaDA-TTS-review.yml`。
