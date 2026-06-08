---
type: paper
tier: deep
title: "video-SALMONN-o1: Reasoning-enhanced Audio-visual Large Language Model"
arxiv_id: "2502.11775"
source: "Sources/video-SALMONN-o1.pdf"
authors: [Guangzhi Sun, Yudong Yang, Jimin Zhuang, Changli Tang, Yixuan Li, Wei Li, Zejun Ma, Chao Zhang]
year: 2025
venue: "Preprint"
tags: [audio-visual-LLM, reasoning, process-DPO, chain-of-thought, video-understanding, benchmark, multimodal, reinforcement-learning, pairwise-preference]
concepts: ["[[SpeechLanguageModel]]", "[[ModalityAdaptationforSpeechLLM]]", "[[AudioUnderstanding]]", "[[DifferentiableRewardOptimization]]", "[[Audio-LanguagePretraining]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: ["VideoMME", "NExT-QA", "RivaBench", "LibriSpeech-960h", "AudioCaps"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeechLanguageModel]], [[ModalityAdaptationforSpeechLLM]], [[AudioUnderstanding]], [[Whisper]], [[DifferentiableRewardOptimization]], [[Audio-LanguagePretraining]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[ModalityAdaptationforSpeechLLM]][待确认], [[AudioUnderstanding]][待确认], [[Whisper]][待确认], [[DifferentiableRewardOptimization]][待确认], [[Audio-LanguagePretraining]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: video-SALMONN-o1 属于 latent-representation-based 多模态 LLM 路线 -- 不同于 audio-token-based SpeechLM (如 Moshi, AudioLM),它使用连续表征 + 模态适配器桥接编码器与 LLM [参见 [[Speech-LLMIntegrationTaxonomy]]]。SALMONN 系列在 [[ModalityAdaptationforSpeechLLM]] 概念页中被列为 Q-Former 适配器的代表系统。

**已有认知**:
- [[SpeechLanguageModel]]: 演进线中 SALMONN 定位在 latent-representation 路线,与 audio-token 路线(GSLM→Moshi)平行发展。video-SALMONN-o1 扩展到视觉模态,且聚焦推理能力而非生成。
- [[ModalityAdaptationforSpeechLLM]]: Q-Former 在 Yu et al. (2024) 实验中性能优于 Conv downsampling 和 CTC compression。video-SALMONN-o1 的音频侧使用 window-level Q-Former。
- [[DifferentiableRewardOptimization]]: KB 中已有大量 TTS 领域 DPO/RL 工作 (DiffRO, SpeechAlign, GRPO-TTS 等),但均作用于语音生成质量优化;本文的 pDPO 作用于视频推理的步骤级偏好,是 DPO 在多模态推理方向的应用。
- [[Whisper]]: Whisper-Large-v3 encoder 是本文的音频编码器,也是 SpeechLM 领域最流行的 speech feature extractor。

**创新判断**: 本文的核心创新 -- process DPO (pDPO) 和 contrastive step selection -- 是推理优化领域的贡献,与 KB 中已有的 TTS RL 优化方法(token 级 reward、Gumbel-Softmax 采样)在技术层面有共性(都是 DPO 变体),但应用场景完全不同(视频推理 vs 语音生成)。

## 速查

> [!summary] 速查
> - **一句话**: 首个开源推理增强音视频 LLM,通过 process DPO (pDPO) + contrastive step selection 实现步骤级推理优化,在 VideoMME/NExT-QA/RivaBench 上超越同规模所有开源模型
> - **路线**: 视频帧 → SigLIP encoder + Whisper-Large-v3 encoder → 各自 aligner → interleaved synchronization → Qwen2-7B (LoRA) → 推理步骤 + 答案
> - **指标**: VideoMME 65.6%, NExT-QA 82.3%, RivaBench-StandUp 76.7%(超 Gemini-1.5-pro 的 75.8%）; pDPO 比 SFT 提升 3-8% absolute [Table 2]
> - **可借鉴**: (1) pDPO 的步骤级偏好建模思路可迁移到 TTS 推理链优化; (2) contrastive step selection 用输入扰动+KL 散度定位易错步骤,计算效率高; (3) 推理路径 + 直接回答混合 SFT 策略
> - **局限**: 7B 规模限制; SynthDec F1 仅 17.8% 说明合成视频检测仍很难; 依赖 Gemini-1.5-pro/GPT-4o 生成训练数据

## 核心问题

本文要解决的核心问题: **现有多模态 LLM 的推理优化集中在数学/图像领域,忽略了通用视频理解中音视频多模态交互的推理需求** [§1]。具体而言:

1. 视频理解模型在 SFT 后普遍丧失 step-by-step 推理能力,倾向于直接输出答案 [§3.2]
2. 现有 reward model (ORM/PRM) 在视频理解任务上训练困难 -- loss 仅下降约 5% [§7.3],因为给视频推理步骤打绝对分数远比数学题困难
3. 缺乏需要深度音视频推理的评测基准

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

video-SALMONN-o1 沿用 video-SALMONN 2 的双分支编码架构 [§3.1, Fig 1]:

```
输入视频 ──┬── Visual Branch: SigLIP encoder → 2-layer Linear+GELU aligner → visual tokens
           │   (2 fps, max 60 frames)
           │
           └── Audio Branch: Whisper-Large-v3 encoder → window-level Q-Former aligner → audio tokens
               (0.2s window, 150 tokens per 30s)
               ↓
           Interleaved Synchronization (视觉帧等距排列, 音频 tokens 插入对应时间段)
               ↓
           H^AV = Concat(..., H^V_t1, H^A_{t1:t2}, H^V_t2, ...)   [Eqn. 1]
               ↓
           Qwen2-7B LLM (LoRA r=64, α=256) → 推理步骤 {s1, s2, ..., sK} → 答案 A
```

**音频处理细节**: 音频被切成 t_max 长度的段,每段独立通过 Whisper 编码,加上 segment-level positional embedding 后拼接,再通过 Q-Former aligner [Eqn. 2-5]。

**训练时冻结策略**: visual encoder + visual aligner + audio encoder + LLM backbone 全部冻结,仅训练 audio aligner (从头) 和 LoRA 模块 [§6.1]。

### 关键设计选择

#### 1. Reasoning-intensive SFT 数据构建 [§3.2, Fig 2]

[论文原文] 视频理解模型 SFT 后丧失推理能力的原因: 模型学会直接输出答案,不再生成推理步骤。

数据构建流水线:
1. 从 13k 视频 + 音频的训练集中,用 Gemini-1.5-pro 生成 question-answer-reasoning 三元组
2. GPT-4o 做质量检查,不合格的重新生成
3. 除新创建的推理题外,还为原始训练集生成推理路径(同样 Gemini-1.5-pro 生成 + GPT-4o 检查)

**关键决策**: 推理路径和直接回答混合训练 -- 避免模型学习两种不同的回答机制 [§3.2]。[论文原文] 这一点在实验中被证明对保持竞争力的推理性能至关重要。

最终 SFT 数据: 150k normal QA + 30k reasoning-intensive QA,每个 QA 都带推理步骤 [§6.2]。

#### 2. Process DPO (pDPO) [§4.2, Fig 3]

pDPO 是本文核心技术贡献。与已有方法的关键区别:

| 方法 | 粒度 | 建模方式 | 问题 |
|------|------|---------|------|
| ORM | 整条路径 | 绝对分数 (0/1) | 视频推理中分数标准模糊 [§4.2] |
| PRM | 每一步 | 绝对分数 (p_sk) | 同上,且标注成本高 |
| PPRM | 整条路径 | 成对偏好 | 粒度太粗,无法优化单步 |
| **pDPO** | **每一步** | **成对偏好** | -- |

pDPO 的工作流程:
1. 对推理步骤 sk,生成替代步骤 s'k (相同 prefix,不同续写)
2. 分别从 sk 和 s'k 出发做 rollout (各 6 条),得到最终答案
3. 通过 GPT-4o 比对答案正确性,计算 expected correctness p_sk 和 p_{s'k} [Eqn. 2]
4. 用 Bradley-Terry 模型定义偏好概率 p(sk > s'k) = sigma(r(sk) - r(s'k)) [Eqn. 5]
5. pDPO loss: L = -E[alpha_k * log p(sk > s'k) + (1-alpha_k) * log p(s'k > sk)] [Eqn. 6]

其中 alpha_k = 1(p_sk > p_{s'k}) 作为硬标签,或 sigma((p_sk - p_{s'k})/mu) 作为软标签以容忍估计噪声 [§4.2]。

**与 PPRM 的协同**: pDPO 在步骤级提供细粒度优化,PPRM 在完整路径级确保整体解的生成能力。两者联合使用 [§4.2]。

[agent 解读] pDPO 的核心洞察是: 在视频推理中,打绝对分数极其困难(ORM/PRM 的 loss 仅下降 5%),但判断"哪个步骤更好"相对容易,因为可以通过 rollout 到最终答案来间接评估。这与 TTS 领域的 DPO 思路类似 -- 比较两段语音的相对偏好比给绝对 MOS 分更容易。

#### 3. Contrastive Step Selection [§4.3, Fig 3 top]

[论文原文] 通过分析验证集上的错误推理路径,发现 **>70% 的推理错误发生在模型误解或幻觉视频内容的步骤** [§4.3]。

定位这些步骤的方法:
1. 对输入视频施加微小扰动 (tiny perturbation)
2. 计算每步的 length-normalized per-token KL 散度:
   d_sk = (1/|sk|) * sum_{yi in sk} DKL(P(yi|y<i, H^AV) || P(yi|y<i, H_tilde^AV))
3. d_sk 越高 → 该步对视频输入越敏感 → 越容易出错
4. 选择 top T 步 (T=3 最优 [Fig 5]) 做 pairwise rollout

[agent 解读] 这个设计巧妙之处在于: 它不需要任何人工标注就能自动识别"最值得优化的推理步骤"。高 KL 散度意味着该步骤高度依赖视频输入,而这恰好是视频推理中最容易出错的地方(幻觉/误读)。文本逻辑错误则由 PPRM 在整条路径级别处理。

**训练偏向**: contrastive step selection 使 pDPO 训练偏向视频依赖型错误 (video-dependent errors),文本逻辑错误由 PPRM 覆盖 [§4.3]。

### 训练策略

三阶段训练 [§3.1, §6.1]:

| 阶段 | 数据 | 训练模块 | 冻结模块 | 硬件 |
|------|------|---------|---------|------|
| Audio modality alignment | LibriSpeech-960h + AudioCaps | Audio aligner (从头) | Visual encoder/aligner, Audio encoder, LLM | - |
| Audio-visual SFT | 13k videos, 150k QA + 30k reasoning QA | Audio aligner + LoRA | Visual encoder/aligner, Audio encoder, LLM backbone | 16x A100, 48h |
| pDPO | ~100k full path pairs + ~100k step pairs from 5k videos | LoRA | 其余全部 | 8x A100, 24h |

pDPO 数据构建: 从 reasoning-intensive subset 采样 10 条路径/QA → 保留 SFT 模型答错的 QA → 对完整路径用 GPT-4o 比对选 preferred → 对 top 3 步做 6 rollouts 构建步骤级 pairs [§6.2]。

### RivaBench 基准 [§5, Table 1]

| 分区 | 规模 | 格式 | 来源 | 考察能力 |
|------|------|------|------|---------|
| Academic | 1,912 QA | 5-way MCQ | M3AV 会议/讲座录像 | 学术内容理解 + 跨模态推理 |
| StandUp | 2,128 QA | 5-way MCQ | YouTube 脱口秀 | 幽默理解 + 语音+表情+语境推理 |
| SynthDec | 200 QA | Yes/No | Hunyuan-large 生成 + 对应真实视频 | 合成视频检测 (零样本) |

[论文原文] RivaBench 与 VideoMME 的区别: 需要 **更长的推理链、更广的世界知识、更紧密的音视频信息结合** [§2.3]。GPT-4o 在 StandUp/Academic 上因缺乏音频信息而落后 Gemini-1.5-pro,说明这些任务确实需要音视频联合理解 [§7.1]。

## 实验

| 指标 | video-SALMONN-o1 (pDPO) | LLaVA-OneVision | video-SALMONN | Gemini-1.5-pro | GPT-4o | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VideoMME (acc) | **65.6%** | 58.2% | 43.3% | 75.0% | 71.9% | VideoMME | [Table 2] |
| NExT-QA (acc) | **82.3%** | 79.4% | 49.2% | 79.2% | 81.7% | NExT-QA | [Table 2] |
| StandUp (acc) | **76.7%** | 67.2% | 47.8% | 75.8% | 63.3% | RivaBench-StandUp | [Table 2] |
| Academic (acc) | **48.3%** | 45.8% | 33.6% | 67.1% | 60.0% | RivaBench-Academic | [Table 2] |
| SynthDec (F1/P/R) | **17.8%** (87%/13%) | 0.0% (97%/0%) | 0.0% (100%/0%) | 34.1% (90%/21%) | 34.1% (90%/21%) | RivaBench-SynthDec | [Table 2] |

**粗体**: 开源模型最优。video-SALMONN-o1 在推理时执行 step-by-step reasoning,其他开源模型直接输出答案 [Table 2 caption]。

### Ablation: SFT 数据效果 [Table 3]

| 配置 | 推理 | VideoMME | NExT-QA | Academic | StandUp |
|------|------|----------|---------|----------|---------|
| Full SFT data | No | 63.7% | 80.7% | 45.2% | 72.3% |
| Full SFT data | Yes | 62.9% | 78.2% | 42.5% | 68.6% |
| Full SFT + pDPO | Yes | **65.6%** | **82.3%** | **48.3%** | **76.7%** |
| w/o reasoning-intensive | Yes | 61.6% | 76.6% | 42.3% | 67.5% |
| Reasoning-intensive only | Yes | 58.8% | 75.2% | 40.1% | 63.5% |

**关键发现**:
1. **SFT 后直接回答优于推理** (row 1 vs row 2) -- 原因是 teacher forcing 的 exposure bias 在长序列推理路径上更严重 [§7.2, 论文原文]
2. **pDPO 解决 exposure bias** -- 通过在自身样本上学习,实现一致性提升 [§7.2]
3. **Reasoning-intensive 数据必不可少** -- 去掉后推理性能明显下降 [§7.2]
4. **仅用 reasoning-intensive 数据不行** -- 模型无法获得基础的音视频感知能力 [§7.2]

### Ablation: Reward 建模方法 [Table 4]

| 方法 | 推理方式 | VideoMME | NExT-QA | StandUp | Academic |
|------|---------|----------|---------|---------|----------|
| SFT | 1-best | 62.9% | 78.2% | 68.6% | 42.5% |
| SFT | Major@20 | 63.5% | 81.5% | 73.5% | 45.3% |
| SFT + ORM | RM@20 | 62.7% | 78.5% | 69.0% | 42.6% |
| SFT + PRM | RM@20 | 63.5% | 79.3% | 72.1% | 43.9% |
| SFT + pDPO | 1-best | **65.6%** | **82.3%** | **76.7%** | **48.3%** |

[论文原文] ORM 表现混杂,PRM 一致但微弱改进(仅与 majority voting 持平)。原因: ORM/PRM 的训练 loss 仅下降约 5%,反映了在通用视频理解任务上学习绝对分数的困难 [§7.3]。pDPO 通过成对偏好避免了这一问题。

### Ablation: Contrastive Step Selection [Fig 5]

| 配置 | VideoMME | StandUp | Academic |
|------|----------|---------|----------|
| Full paths only | 65.6% | 75.1% | 46.5% |
| Full paths + Top 3 steps | 65.7% | 76.7% | 48.3% |
| Full paths + All steps | 65.6% | 76.5% | 48.3% |

[论文原文] 使用中间步骤 pairs 在需要频繁参考视频/音频信息的问题上改进尤为明显 [§7.3]。Top 3 步与全部步骤效果接近,但计算成本大幅降低。

### 零样本合成视频检测 [§7.1]

[论文原文] video-SALMONN-o1 展现出零样本合成视频检测能力(训练中从未见过此任务),而所有其他开源模型 100% 输出 "real" [Table 2]。模型能在推理过程中主动寻找视频中的扭曲异常 [Fig 16-17, Appendix G]。

[agent 解读] 这是一个有趣的涌现能力: 增强推理能力后,模型可以逐步分析视频中的物理违规现象,而不是简单输出默认答案。但 F1 仅 17.8% 说明这项能力仍然很初级。

## 局限性

1. **模型规模**: 仅 7B 参数,与 Gemini-1.5-pro 等大模型有明显差距(Academic 48.3% vs 67.1%）[Table 2]
2. **合成视频检测仍很难**: 即使推理增强,F1 也仅 17.8%;即使是违反物理规则的视频,SOTA 模型仍难检测 [§7.1]
3. **训练数据依赖闭源模型**: reasoning SFT 数据依赖 Gemini-1.5-pro 生成 + GPT-4o 质量检查 [§3.2]
4. **Exposure bias 仅部分解决**: SFT 后直接回答仍优于推理(63.7% vs 62.9% on VideoMME),pDPO 才扭转,说明推理能力的获取仍不够自然 [Table 3]
5. **评估局限**: RivaBench 仅覆盖三个场景,且 SynthDec 仅 200 条数据
6. **固有偏差**: 继承 SigLIP/Whisper/Qwen2 预训练模型的偏差,可能对特定人群表现更差 [Impact Statement]

## 点评

**优势**:
- pDPO 的设计理念清晰: 在绝对分数难以定义的视频推理任务中,成对偏好是更自然的优化信号。ORM/PRM loss 仅降 5% vs pDPO 的一致性提升,这个对比非常有说服力 [Table 4]
- Contrastive step selection 用信息论方法(KL 散度)替代了人工标注,既高效又有理论依据(高 KL = 高视频依赖 = 高错误风险)
- RivaBench 的 StandUp 分区设计新颖 -- 理解幽默需要同时理解语音内容、面部表情和语境,是真正的多模态推理任务
- 在 StandUp 上超过 Gemini-1.5-pro (76.7% vs 75.8%),说明推理优化确实有效

**不足**:
- 与 [[论文笔记/video-SALMONN|video-SALMONN]] 的关系没有充分讨论 -- video-SALMONN-o1 使用 video-SALMONN 2 架构但似乎跳过了 video-SALMONN 2 的 MrDPO captioning 优化,直接在 SFT 基础上做推理优化
- pDPO 的 rollout 过程仍然计算昂贵(10 paths/QA sampling + 6 rollouts/step),只是通过 contrastive step selection 缓解而非根本解决
- SynthDec 的 evaluation protocol 值得商榷 -- 仅 200 条数据,且 F1 分数的 precision/recall 极不平衡(87%/13%)

## 可复用的 idea

1. **步骤级偏好优化替代绝对评分**: 当任务难以定义绝对质量分数时(如 TTS 韵律、对话自然度),可考虑 pDPO 式的成对偏好建模
2. **输入扰动 + KL 散度定位关键步骤**: 适用于任何多模态推理场景 -- 通过微扰输入模态并测量输出敏感度来自动识别"最值得优化的决策点"
3. **推理路径 + 直接回答混合训练**: 避免模型学习两种不同的回答机制,在 SFT 阶段同时教授推理和直答
4. **Proprietary LLM 生成 + Proprietary LLM 质检**: 双重闭源 LLM 管线(Gemini 生成 + GPT-4o 检查)用于构建高质量训练数据,可迁移到其他数据稀缺场景
5. **零样本能力作为推理增强的副产品**: 增强推理能力可能在未见任务上产生涌现能力(如合成视频检测)

## 审阅

> [!review] 审阅: pass-with-fixes (0 high, 1 medium, 3 low)
> - **medium**: frontmatter datasets 字段为空,应补充 VideoMME/NExT-QA/RivaBench/LibriSpeech-960h/AudioCaps
> - low: frontmatter tasks 为空 (vault 无对应视频任务页,可接受)
> - low: contrastive step selection 的扰动类型未具体说明
> - low: frontmatter models 仅列 Whisper,缺少其他组件/baseline (无对应模型页,可接受)
> - 详见 `_review/video-SALMONN-o1-review.yml`
