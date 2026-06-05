---
type: paper
tier: deep
title: "DUET: Unified Dual-Space Emotion Control for Diffusion/Flow TTS"
arxiv_id: "2606.00066"
source: "Sources/DUET.pdf"
authors: [Xu Zhang, Longbing Cao, Zhangkai Wu]
year: 2026
venue: "arXiv"
tags: [emotion-control, TTS, diffusion, flow-matching, activation-steering, plug-and-play, training-free, classifier-guidance, mel-space-guidance]
concepts: ["[[EmotionControlinTTS]]", "[[ConditionalFlowMatching]]", "[[Diffusion-basedTTS]]", "[[Classifier-FreeGuidance]]", "[[MelSpectrogram]]", "[[NeuralVocoder]]"]
models: ["[[CosyVoice2]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页: [[ConditionalFlowMatching]], [[NeuralVocoder]], [[EmotionControlinTTS]], [[Diffusion-basedTTS]], [[Classifier-FreeGuidance]], [[MelSpectrogram]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]]✓, [[NeuralVocoder]]✓ | 过滤: [[EmotionControlinTTS]][待确认], [[Diffusion-basedTTS]][待确认], [[Classifier-FreeGuidance]][待确认], [[MelSpectrogram]][待确认] | 未命中但可能相关: 无

**谱系定位**: DUET 位于 training-free 情感控制的演进线上,直接继承 [[论文笔记/EmoSteer-TTS|EmoSteer-TTS]] (Xie et al., 2025) 的 activation steering 思路,但做了两项关键扩展:(1) 从 DiT-only 扩展到跨架构 (DiT + Transformer + U-Net, 覆盖 diffusion 和 flow-matching 两大范式);(2) 在 hidden space steering 之外增加 mel-space guidance (通过可微 vocoder 反传 SER 梯度),形成双空间联合控制。

**已有认知**:
- **EmotionControlinTTS** [待确认] 梳理了情感控制的完整路线图: embedding → 层级建模 → 对抗解耦 → DPO → 球面表示 → LLM prompt → training-free steering。EmoSteer-TTS 首次证明 flow-matching TTS 的 DiT 激活中隐式编码了情感,可通过 difference-in-means 提取 steering vector。CoCoEmo (2026) 将 steering 从 flow-matching 转移到 SLM 阶段。
- **ConditionalFlowMatching** ✓ 是 DUET 支持的 backbone 范式之一 (F5-TTS, Matcha-TTS, StableTTS),生成轨迹遵循 ODE。
- **Diffusion-basedTTS** [待确认] 是 DUET 支持的另一范式 (GradTTS 基于 SDE, ProDiff 基于 DDPM),生成轨迹遵循 reverse SDE。
- **Classifier-FreeGuidance** [待确认] 概念页记录了 classifier guidance 与 CFG 的对比: DUET 的 mel-space guidance 属于 classifier guidance 范式 (用外部 SER 梯度引导),而非 CFG。EmoDiff (2022) 曾用类似的 classifier guidance 控制情感但效果有限。
- **NeuralVocoder** ✓ 中 Vocos (Siuzdak, 2024) 是可微 vocoder,DUET 正是利用其可微性将 SER 梯度从 waveform 反传到 mel spectrogram。
- **MelSpectrogram** [待确认] 是 DUET mel-space guidance 的操作空间,mel→waveform 的映射由可微 vocoder 完成。

**创新判断**: 相对于 EmoSteer-TTS (仅 hidden space steering, 仅 DiT), DUET 的核心新增是双空间联合控制和跨架构泛化。相对于 TTS-CtrlNet (需训练 ControlNet 旁挂), DUET 完全不需训练。相对于 DiffRO/RLAIF-SPA (需 reward model 训练/GRPO 优化), DUET 只需一次 linear probing。

## 速查

> [!summary] 速查
> - **一句话**: 发现预训练 diffusion/flow-matching TTS 的 hidden states 中情感方向与说话人方向近正交,由此设计双空间 plug-and-play 情感控制框架 — hidden space steering 偏移韵律轨迹 + mel-space guidance 经可微 vocoder 修正频谱细节
> - **路线**: 冻结 TTS backbone → linear probe 找到情感最可分层 l* 和判别方向 de → 每步去噪: (1) hidden state 沿 de steering (norm-adaptive) → (2) clean mel estimate 经 Vocos 反传 SER 梯度修正 → 下一步
> - **指标**: ESD 上 DUET+GradTTS Avg 75.5% vs 最强 baseline Qwen3-TTS 46.8% (+28.7%); 主观 EMOS 3.93 (最高), NMOS 3.83 [Table 1, Table 3]
> - **可借鉴**: (1) linear probe 同时检测情感可分层 + 验证 speaker/emotion 正交性的方法论; (2) norm-adaptive steering (perturbation 按 ||h|| 缩放) 解决去噪步间尺度变化问题; (3) cosine-scheduled guidance + trust-region 约束组合,控制梯度介入的时间窗口和步长上限
> - **局限**: 仅支持离散类别情感 (angry/happy/sad); angry 表现显著弱于 happy/sad (仅达 GT ceiling 的 49%); 未验证连续 arousal-valence 控制; 需要少量情感标注语音做 probing (~数千条); 代码/模型未开源

## 核心问题

1. **情感信号在预训练 TTS 模型中存在吗?** — 模型从未接受情感监督,但论文发现 hidden states 中情感是 linearly decodable 的,且与 speaker identity 方向近正交 (|cos theta| = 0.029 on F5-TTS) [§1, Fig 1c-d]。
2. **为什么现有情感 TTS 方法需要监督再训练?** — 情感仅占 hidden state variance 的 8.5% [Fig 1a],在 mel 空间中同一情感跨说话人差异大 (cosine similarity 仅 0.03-0.20) [Fig 1b],所以直接在 mel 空间操作效果差。
3. **如何在不训练的前提下精确控制情感?** — 双空间互补: hidden space steering 处理全局韵律方向 (prosodic trajectory), mel-space guidance 修正细粒度频谱纹理 (spectral texture),两者在每个去噪步内顺序执行。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DUET 是一个插件式框架,冻结预训练 TTS backbone 的所有参数,在推理时的每个去噪步插入两个干预:

```
每个去噪步 t:
  ① Hidden state steering: h → h* (沿 de 偏移)
  ② 前向传播剩余层 → clean mel estimate x̂₀
  ③ Mel-space guidance: x̂₀ → x̂'₀ (SER 梯度修正)
  ④ Re-noise x̂'₀ → x_{t-Δt}
```

适用于两大范式:
- **Flow matching** (F5-TTS, Matcha-TTS, StableTTS): velocity prediction → `x̂₀ = xt - t·vθ(xt,t;c)` [Eq 9]
- **Diffusion** (GradTTS, ProDiff): noise prediction → `x̂₀ = (xt - √(1-ᾱt)·εθ) / √ᾱt` [Eq 7]

### 关键设计选择

**1. 为什么用 linear probe 而不是 difference-in-means?**

EmoSteer-TTS 使用 difference-in-means 提取 steering vector,只捕获了 centroid 方向。DUET 认为单一 centroid 方向不足以覆盖 probe weight matrix W 定义的完整判别子空间 [论文原文, §3.1.2]。因此 DUET 对 W 做 SVD 取 top-k 右奇异向量 (正交于 centroid 方向),组合成更完整的 steering direction [Eq 3]:

```
de = δ̂e + β·Σvi  (δ̂e = centroid, vi = SVD 方向, β 控制相对权重)
```

[agent 解读] 这本质上是用 Fisher 线性判别分析的多方向版本替代了简单的均值差,理论上能捕获更丰富的情感子空间结构。

**2. 为什么需要两种 probing 模式?**

对于 reference-speech-conditioned backbone (如 F5-TTS): 可直接用情感标注的参考语音,通过冻结模型提取 hidden states + 已知 label 训练 probe [论文原文, §3.1.1]。

对于 text-only backbone (如 GradTTS, Matcha-TTS): 参考语音的 hidden states 与生成时的 hidden states 近正交 (cosine similarity ~0.05) [§3.1.1, §4.3]。因此必须在 generation time 采样: 先用冻结模型生成语音,再用外部 SER 标注伪标签,在 (hidden state, pseudo-label) 对上训练 probe [论文原文]。

[agent 解读] 这解决了 EmoSteer-TTS 无法泛化到 text-only backbone 的根本限制 — EmoSteer-TTS 的 difference-in-means 需要情感参考语音作为 anchor。

**3. 为什么用 norm-adaptive steering?**

Hidden state 的 norm ||h|| 在去噪过程中变化很大 (早期大、后期小) [论文原文, §3.1.3]。固定幅度的 perturbation 在早期会被 h 淹没、在后期又会主导 h。因此 DUET 将 perturbation 按 ||h|| 缩放 [Eq 4]:

```
h* = h + λ · (de/||de||) · ||h||
```

λ 是相对强度比,跨步恒定。[agent 解读] 这类似于 NLP activation steering 中的 norm-matching 策略,确保干预不会破坏 hidden state 的 learned manifold。

**4. 为什么需要 mel-space guidance?**

Hidden state steering 在表示层偏移了全局韵律方向,但不直接塑造 mel spectrogram [论文原文, §3.2]。而 mel 到 waveform 的映射是 vocoder 的非线性变换,仅在 hidden space 操作无法保证最终波形体现目标情感的细粒度声学纹理 [论文原文]。

[agent 解读] 这类似 EmoDiff (2022) 的 classifier guidance 思路,但 DUET 的创新在于将梯度从 waveform 经可微 vocoder 反传到 mel,而非直接在 mel 上做梯度 (后者忽略了 mel→wave 的非线性失真)。

**5. Cosine schedule + trust region: 为什么不全程做 guidance?**

Guidance 梯度的可靠性随去噪进度变化 [论文原文, §3.2]:
- 早期步: mel 估计噪声大,梯度不可靠
- 中间步: mel 足够干净又足够灵活,梯度最有效
- 后期步: mel 几乎收敛,改动空间小

因此用 cosine schedule `w(t) = 0.5(1+cos(π|t-tpeak|/twidth))` 集中在中间步 [Eq 6],并用 trust region `||x̂'₀ - x̂₀|| ≤ γ||x̂₀||` 防止 overshoot [§3.2]。

### 训练策略

DUET **不训练 TTS backbone** — 唯一需要训练的是 lightweight linear probe (一个线性层),在少量情感标注/伪标注语音上用 cross-entropy 训练。Probe 参数量极小 (D × C,D 为 hidden dim, C 为情感类别数)。

steering 方向一次提取,固定用于全部评估 [§4.1]。

## 实验

| 指标 | 本文 (DUET+GradTTS) | 最强 Baseline (Qwen3-TTS) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Emotion Acc Avg (SER) | 75.5% | 46.8% | ESD | [Table 1] |
| Emotion Acc Avg (SER) | 60.2% | 38.0% | CREMA-D | [Table 1] |
| Emotion Acc Avg (SER) | 55.1% | 44.2% | IEMOCAP | [Table 1] |
| EMOS (subjective) | 3.93 | 3.75 (Qwen3-TTS) | 36 samples | [Table 3] |
| NMOS (subjective) | 3.83 | 4.18 (Qwen3-TTS) | 36 samples | [Table 3] |

**跨 backbone 结果** [Table 1]:
| Backbone | ESD Avg | CREMA-D Avg | IEMOCAP Avg |
| --- | --- | --- | --- |
| GradTTS (diffusion, U-Net) | 75.5% | 60.2% | 55.1% |
| F5-TTS (flow, DiT) | 64.9% | 50.2% | 64.1% |
| Matcha-TTS (flow, Transformer) | 64.3% | 47.4% | 75.8% |
| ProDiff (diffusion, DDPM) | 63.6% | 75.5% | 66.9% |
| StableTTS (flow, lightweight) | 43.3% | 32.8% | 48.8% |

**消融** (F5-TTS on ESD) [Table 2]:
| 变体 | Angry | Happy | Sad | Avg |
| --- | --- | --- | --- | --- |
| Full DUET | 40.9% | 75.2% | 78.7% | 64.9% |
| w/o mel guidance | 23.5% | 50.9% | 61.8% | 45.4% |
| w/o hidden steering | 16.5% | 46.5% | 58.8% | 40.6% |

**关键发现**:
1. 4/5 backbones 的 DUET 超过全部 10 个监督 baseline 的平均准确率 [Table 1]
2. Hidden state steering 是主要贡献者 (去掉后 -24.3%),mel guidance 提供互补增益 (去掉后 -19.5%) [Table 2]
3. Angry 表现显著弱于 happy/sad — 在 F5-TTS 上仅达 GT ceiling 的 49% (vs happy 83%, sad 91%) [Fig 2a]。作者归因于 anger 依赖时间局部化的 cue (sharp onset, rhythm change),而 uniform steering 和 utterance-level guidance 无法捕捉 [论文原文, §4.2]
4. Emotion 和 speaker 方向的 cosine similarity 仅 0.029 (F5-TTS),确认近正交 [§4.3, Fig 2b]
5. Probe accuracy 与 steering SER accuracy 跨层高度相关,均在 layer 16 达到峰值 [Fig 2c]

## 局限性

1. **仅支持离散类别情感**: angry/happy/sad 三类,未验证连续 arousal-valence 控制 (作者在结论中提到这是自然扩展方向) [§6]
2. **Angry 表现较弱**: uniform steering 对时间局部化情感 (anger) 效果差,作者承认需要 temporally adaptive variant [§6]
3. **需要少量标注数据**: linear probe 需要情感标注语音 (或 generation-time probing 需要 SER 伪标签),不是完全零资源
4. **NMOS 略低于 Qwen3-TTS**: 3.83 vs 4.18 [Table 3],作者归因于 mel-space guidance 引入的微小频谱扰动 [§4.5]
5. **评估用 macro-average across 2 SER models**: 可能与人感知有偏差,尤其对 anger 的机器评估可能高估
6. **代码/模型未开源**: 可复现性依赖论文描述
7. **未与最直接竞品 EmoSteer-TTS 做受控对比**: 虽然在 baselines 中列了 supervised 方法,但未将 EmoSteer-TTS 作为 baseline 直接比较 (两者最相似)

## 点评

**优势**:
- 关键 insight 有深度: 情感在 hidden states 中 linearly decodable + 与 speaker 近正交的发现,为整个框架提供了可靠的理论基础。这不只是工程 trick,而是对预训练 TTS 模型表示结构的新理解。
- 双空间互补设计合理: hidden steering 做全局韵律,mel guidance 做局部频谱,消融实验清晰验证了互补性 [Table 2]。
- 跨架构泛化是实质性进步: 5 个架构差异巨大的 backbone (DiT/Transformer/U-Net, flow/diffusion) 都 work,说明发现的表示结构是 pretrained TTS 的通用属性。

**不足**:
- 与 EmoSteer-TTS 缺少 head-to-head 对比,使得 probing 方向 vs difference-in-means 的增益难以量化。
- Angry 的 performance gap 暴露了 uniform steering 的根本限制,但论文仅提出"temporally adaptive variant"作为未来方向,未给出具体方案。
- 主观评估规模偏小 (20 listeners, 36 samples),且未报告 95% CI 或 p-values。

## 可复用的 idea

1. **Linear probe 作为 representation structure 的诊断工具**: 不仅定位最可分层,还通过 probe weight matrix 的 SVD 提取多方向判别子空间 + 通过 speaker probe 验证正交性。这套方法论可推广到任何需要理解 frozen model 内部表示的场景。
2. **Norm-adaptive perturbation**: 按 ||h|| 缩放干预幅度,解决 iterative generative model 中步间尺度变化问题。可迁移到其他 activation steering 场景 (LLM decoding intervention, image diffusion control 等)。
3. **Generation-time probing**: 对 text-only backbone,用模型自身生成 + 外部标注器伪标签做 probe 训练。解决了"offline anchor 与 online generation 表示不对齐"的通用问题。
4. **Cosine schedule + trust region 组合**: 控制 gradient-based guidance 的时间窗口和步长上限,比单一超参 η 更鲁棒。可用于任何 classifier-guidance 场景。
5. **Dual-space 互补框架**: 将 representation-level intervention 与 output-level gradient guidance 组合的思路,可推广到其他可控生成场景 (如 image style transfer + pixel-level refinement)。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 5 个编号设计选择各回答 WHY; 可借鉴给出 3 个具体 trick |
> | 可信赖 | pass | 所有关键数字经 PDF 交叉验证; 出处标注覆盖率 ~95% |
> | 可区分 | pass | 因果解释一致标注 [论文原文]/[agent 解读]; 来源覆盖率 ~90% |
> | 可定位 | pass | KB 背景含 EmoSteer-TTS/TTS-CtrlNet/DiffRO 对比谱系 |
> | 不污染 | pass | 无新建概念页; 反向更新仅追加 |
> 
> Issues: 2 (high: 0, medium: 0, low: 2)
> 详见 `_review/DUET-review.yml`
