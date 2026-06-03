---
type: paper
tier: deep
title: "Fish Audio S2 Technical Report"
arxiv_id: "2603.08823"
source: "Sources/FishAudioS2.pdf"
authors: [Shijia Liao, Yuxuan Wang, Songting Liu, Yifan Cheng, Ruoyi Zhang, Tianyu Li, Shidong Li, Yisheng Zheng, Xingwei Liu, Qingzheng Wang, Zhizhuo Zhou, Jiahua Liu, Xin Chen, Dawei Han]
year: 2026
venue: "Technical Report"
tags: [TTS, LLM, dual-AR, RVQ, audio-codec, GRPO, reinforcement-learning, zero-shot, voice-cloning, instruction-following, multi-speaker, open-source, streaming, controllable-TTS]
concepts: ["[[LLM-based TTS]]", "[[Residual Vector Quantization]]", "[[Semantic vs Acoustic Tokens]]", "[[Speech Tokenizer]]", "[[Differentiable Reward Optimization]]"]
models: ["[[论文笔记/Fish-Speech|Fish-Speech]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 1 个待确认: [[LLM-based TTS]], [[Speech Tokenizer]], [[Residual Vector Quantization]], [[Semantic vs Acoustic Tokens]], [[Differentiable Reward Optimization]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[Residual Vector Quantization]]✓, [[Semantic vs Acoustic Tokens]]✓ | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Codec Language Model]](pending-review) | 未命中但可能相关: 无

**谱系定位**: Fish Audio S2 是 [[论文笔记/Fish-Speech|Fish-Speech]] (Fish Audio S1, 2024) 的直接后继。S1 首次提出了 Dual-AR (Slow+Fast Transformer) + GFSQ 量化的 LLM-based TTS 架构。S2 保留了 Dual-AR 骨架,但做了四个关键升级:(1) GFSQ→RVQ 量化回归;(2) Qwen3-4B 作为 Slow AR backbone 替代自训练模型;(3) 引入 GRPO-based RL 后训练;(4) 增加自然语言 instruction following 能力。

**在 [[LLM-based TTS]] 谱系中**: S2 属于 decoder-only autoregressive codec LM 路线 (VALL-E → Fish-Speech → Fish Audio S2),与 CosyVoice 系列的 "LLM→semantic tokens→CFM" 混合路线形成对比。S2 的 RL 后训练使用 GRPO (而非 CosyVoice 3 的 DiffRO),是 TTS 领域 GRPO 应用的代表性工作。

**在 [[Residual Vector Quantization]] 演进中**: S1 的 GFSQ (Grouped Finite Scalar Quantization) 声称 100% codebook 利用率优于 RVQ,但 S2 放弃 GFSQ 回归 10 层 RVQ,表明在大规模训练下 RVQ 的重建质量仍具优势。S2 通过 semantic distillation (蒸馏 w2v-BERT 2.0 到 RVQ 第一层) 实现了 mixed token 效果。

**在 [[Semantic vs Acoustic Tokens]] 框架中**: S2 的 audio tokenizer 属于 mixed tokenizer 路线 — RVQ 第一层通过语义蒸馏编码 semantic 信息,后续 9 层编码 acoustic details。这与 Mimi (Moshi) 的设计思路一致,但实现方式不同 (Mimi 用独立 VQ 模块)。

**与 [[Differentiable Reward Optimization]][待确认] 对比**: S2 使用 GRPO (在 audio 层面计算 reward 后通过 group-level advantage 优化), 而非 CosyVoice 3 的 DiffRO (在 token 层面直接计算可微 reward)。二者都是 TTS RL 后训练方案,但 GRPO 在 audio 层面操作更直接,DiffRO 在 token 层面更高效。根据 Tongyi (2025) 的对比,DiffRO 降 WER 更强但可能伤 speaker similarity,GRPO 更均衡。

> [!summary] 速查
> - **一句话**: Fish-Speech S1 的全面升级版 — 用 Qwen3-4B 作为 Slow AR backbone + RVQ 语义蒸馏 tokenizer + GRPO 多维度 RL 对齐,实现指令控制、多说话人对话和长文本合成
> - **路线**: Text + Reference Audio → Qwen3-4B (Slow AR, semantic tokens) → 4-layer Fast AR (acoustic tokens, depth-wise) → Multi-Codebook Fusion → 10-layer RVQ Codec (21 Hz, 44.1kHz) with EVA-GAN Decoder → Waveform
> - **指标**: Seed-TTS-Eval WER 0.54%/0.99% (zh/en) [Table 1]; ATT posterior mean 0.515 (w/ instruction) [Table 5]; EmergentTTS-Eval win rate 81.88% [Table 6]; RTF 0.195, TTFA <100ms [§5]
> - **可借鉴**: (1) 数据 pipeline 的 dual-purpose 设计 — 质量模型和 ASR 模型既做预训练过滤又做 RL reward,消除分布偏移; (2) Dr.GRPO 去除标准差归一化避免难度偏差; (3) LoRA weight-swap 机制节省 RL 中 reference model 的显存; (4) SGLang RadixCache 对 reference audio KV 缓存实现 86.4% 命中率
> - **局限**: (1) Qwen3-4B backbone 非自研,受限于其 tokenizer 和架构; (2) RL reward 中的 ASR 模型 (Qwen3-Omni-30B-A3B) 远大于 TTS 模型本身,部署成本高; (3) 消融实验极少,难以归因各组件贡献; (4) Fish Audio Instruction Benchmark 是自建,缺乏第三方验证

## 核心问题

Fish Audio S2 要解决三个核心问题:

1. **指令可控性缺失**: S1 仅支持 reference audio 驱动的 voice cloning,无法通过自然语言描述控制情感、韵律、语气等细粒度属性 [§1]
2. **数据 annotation 瓶颈**: 大规模语音数据的细粒度声学属性标注 (emotion, prosody, vocal events) 无法靠人工扩展 [§3]
3. **生成鲁棒性不足**: 自回归 TTS 常见 hallucination、token skipping、timbre drift 等问题,需要 RL 后训练解决 [§4.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Fish Audio S2 沿用 S1 的 Dual-AR 架构 [Fig 2],包含四个核心组件:

1. **Audio Tokenizer**: 基于 DAC 改进的 RVQ codec (10 codebooks, ~21 Hz, 44.1kHz),将语音压缩为离散 token
2. **Slow AR (Temporal Semantic Backbone)**: Qwen3-4B,在时间轴上自回归生成 RVQ 第一层 semantic tokens
3. **Fast AR (Depth-wise Acoustic Decoder)**: 4 层 Transformer,在 codebook 深度轴上自回归生成剩余 9 层 acoustic tokens
4. **EVA-GAN Decoder**: 从 RVQ tokens 重建 44.1kHz 波形

训练分四阶段: Tokenizer Training → Pre-training (8K→16K context) → SFT → RL Post-training [§4]

### 关键设计选择

#### 1. 从 GFSQ 回归 RVQ + 语义蒸馏 [§2.1]

**WHY**: [agent 解读] S1 使用 GFSQ 声称 100% codebook 利用率,但 S2 改用 10 层 RVQ。论文未直接解释这一变化,但从架构看,RVQ 的层级结构天然适配 Dual-AR (第一层→Slow AR, 后续层→Fast AR),且 semantic distillation 在 RVQ 上有成熟先例 (Mimi/Moshi)。

**HOW**: 在 DAC encoder 基础上扩展:
- **Causal Convolutions**: 替换标准卷积实现严格因果推理,支持流式生成 [§2.1]
- **Transformer Bottleneck**: RVQ 前后加滑动窗口 causal Transformer (类 Mimi 设计),解决长序列 OOM [§2.1]
- **Extended Downsampling**: 额外 ConvNeXt V2 层实现 4x 降采样,总降采样率 2048x,帧率 ~21 Hz [§2.1]
- **EVA-GAN Decoder**: 替代原始 DAC decoder,提升参数效率和重建质量 [§2.1]
- **Semantic Distillation**: 辅助 head 回归 w2v-BERT 2.0 第 16 层激活值,强制 RVQ 第一层编码 semantic 信息 [§2.1]

**Tokenizer 总参数**: 446M,训练 1M steps [§4.1]

#### 2. Dual-AR: Slow + Fast 非对称设计 [§2.2]

**WHY**: [论文原文] 直接将 10 层 RVQ 展平到时间轴会导致序列长度增加 10 倍,严重限制 LLM 处理长上下文的能力 [§2.2]

**Slow AR** (Qwen3-4B):
- 在完整 token 序列上自回归,交错文本 token 和音频 token [§2.2]
- 每个时间步 t 预测 RVQ 第一层 semantic token q_t^(0) [§2.2]
- [agent 解读] 选择 Qwen3-4B 作为 backbone 而非自训练 LLM,说明 Fish Audio 认为预训练文本能力 (多语言理解、指令跟随) 比语音领域专用架构更重要

**Fast AR** (4 层 Transformer):
- 接收 Slow AR 的 hidden state h_slow_t 作为 conditioning prefix [§2.2]
- 从 semantic token q_t^(0) 开始,depth-wise 自回归生成 q_t^(1) 到 q_t^(N-1) [§2.2]
- 所有 codebook 层共享同一 embedding table,用 RoPE 编码层级位置 [§2.2]
- [论文原文] 这种高度非对称设计 (4B vs 4层) 保证了推理效率 [§2.2]

**Multi-Codebook Fusion (MCF)**: 每个时间步的 N 个 codebook token 生成完毕后,通过 N+1 个 embedding 求和聚合为下一时间步的输入 [Eq 1]:
x_{t+1} = e^LM_t + sum(E^(k)(q_t^(k))) [§2.2]

[agent 解读] MCF 的 N+1 设计值得注意: semantic token 同时贡献两个独立参数化的 embedding (LM embedding 和 codebook embedding),分别捕获 semantic token 的不同方面。

#### 3. Dual-Purpose 数据 Pipeline [§3]

**WHY**: [论文原文] RL 对齐通常使用独立训练的 reward model,会引入预训练和后训练之间的分布偏移;同时,大规模语音数据的细粒度 vocal annotation 无法人工扩展 [§3]

**HOW**: 围绕两个核心引擎构建三阶段 pipeline [Fig 3]:

**核心引擎 1 — Speech Quality Model** [§3.1]:
- 架构: w2v-BERT 2.0 backbone + MLP head (类 Uni-VERSA) [§3.1]
- 训练: 人工标注的语音质量标签, MSE + focal loss [§3.1]
- Pre-training 阶段: 作为质量过滤器,去除噪声、重叠语音、残留背景音乐 [§3.1]
- RL 阶段: 直接复用为 acoustic preference reward R_Pref [§4.3]

**核心引擎 2 — Rich-Transcription ASR Model** [§3.2]:
- 架构: 基于 Qwen3-Omni-30B-A3B fine-tune [§3.2]
- 功能: 同时转录文本 + 标注说话人轮换 (<|speaker:0|>) + 注入 vocal instructions ([laugh], [whispers], [emphasis] 等) [§3.2, Fig 4]
- Pre-training 阶段: 作为 Rich Transcription 标注器,生成细粒度自然语言指令 [§3.2]
- RL 阶段: 复用为 intelligibility + instruction-following reward R_STT [§4.3]

[agent 解读] 这是本文最核心的设计创新 — "同一个模型既做数据标注又做 RL reward" 的 dual-purpose 思路,从根源上消除了 reward model 与 pre-training data 之间的分布偏移。但代价是 reward model 的 capability 直接受限于 pre-training annotator 的能力上限。

#### 4. GRPO 多维度 RL 对齐 [§4.3]

**WHY**: [论文原文] 自回归音频生成存在 hallucination、token skipping、timbre drift,需要 RL 后训练纠正 [§4.3]

**HOW**: 采用 Dr.GRPO 变体 (去除标准差归一化的 GRPO):

1. **Sampling**: 对每个 prompt 独立采样 G 个候选输出 {y_1,...,y_G} [§4.3]
2. **Advantage**: A_i = R_i - R_mean (不除以标准差,避免难度偏差) [Eq 5]
3. **Slow AR Policy Loss**: 标准 GRPO + per-token KL 散度约束 [Eq 6]
4. **Fast AR Policy Loss**: 共享同一 advantage signal,独立在 codebook 维度优化 [Eq 7]
5. **Total Loss**: L_RL = L_RL_slow + gamma * L_RL_fast [Eq 8]

**多维度 Reward** [Eq 9]:
R_total = lambda_STT * R_STT + lambda_Pref * R_Pref + lambda_SIM * R_SIM

- R_STT (Semantic Accuracy): ASR caption model 的 per-token confidence,对 speaker ID tag 错误加大惩罚,对 missed vocal instructions 追加惩罚 [§4.3]
- R_Pref (Acoustic Preference): Speech quality model 评分 [§4.3]
- R_SIM (Timbre Similarity): 外部 voiceprint model 的 cosine similarity [§4.3]

**工程优化** [§4.3]:
- **异步 scoring 架构**: 将 reward 计算解耦,防止 scoring model 空转 [§4.3]
- **集中式波形缓存**: 最大化 rollout 吞吐量 [§4.3]
- **LoRA weight-swap**: reference policy 以 LoRA 权重备份存于 CPU,动态 swap 到 GPU 做 KL 前向,避免常驻 reference model 占 VRAM [§4.3]
  - rsLoRA (r=16, alpha=64),仅更新 MLP 层 [§4.3]

#### 5. Reference Audio 位置与 Instruction Following [§4.2]

**S1→S2 变化**: S1 将 reference audio 追加到 user input 末尾; S2 改为前置到 system prompt [§4.2]

**WHY**: [agent 解读] 前置到 system prompt 使 reference audio 的 KV cache 可被 RadixCache 缓存复用,多次请求同一 voice 时跳过 prefill,这是 S2 实现 TTFA <100ms 的关键。

**Instruction Following**: 不使用全局 style prompt,而是在对话上下文的特定位置嵌入自然语言指令 (如 [whisper], [angry], [laugh]) [§4.2]

[论文原文] 通过大规模数据上的自回归训练,模型自然内化文本指令到声学变化的映射,无需专用控制 token [§4.2]

### 训练策略

**Audio Tokenizer**: 446M params, 1M steps, multi-discriminator GAN loss (multi-period + multi-resolution + multi-scale STFT) [§4.1]

**Pre-training + SFT** [§4.2]:
- 两阶段渐进: Stage 1 (8192 context) → Stage 2 (16384 context, 支持长文本和多轮多说话人) [§4.2]
- 数据量: 10M+ 小时原始音频,~80 种语言和方言 [§4.2]
- Vocabulary: 扩展 Qwen3-4B + 4096 semantic tokens + 结构控制 token [§4.2]
- 新 token embedding 初始化: 从现有文本 embedding 矩阵的经验均值和协方差采样 [§4.2]
- Reference audio tokens 的 loss 被 mask (防止逐字记忆) [§4.2]

**Slow AR Loss** [Eq 2]:
L_slow = -sum(m_t * lambda_t * log P(x_t | x_{<t})),m_t=0 for system prompt/reference tokens

**Fast AR Loss** [Eq 3]:
- Pre-training: w^(k)=1 均匀权重,k=0 的 semantic token 预测作为辅助任务帮助 Fast AR 学习从 Slow AR hidden state 提取信息 [§4.2]
- SFT: 去掉 semantic token 预测,对剩余 codebook 应用渐进衰减权重,集中模型容量于对感知质量贡献最大的 coarse acoustic codebook [§4.2]

**FSDP + 差异化学习率**: 文本基础参数低学习率,音频模块高学习率 + WSD scheduler [§4.2]

## 实验

### Seed-TTS-Eval [Table 1]

| 指标 | Fish Audio S2 | Fish Audio S1 | CosyVoice 3-1.5B | Qwen3-TTS | Seed-TTS | MiniMax-Speech-02 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER test-zh (%) | **0.54** | 0.54 | 1.12 | 0.77 | 1.12 | 0.99 | [Table 1] |
| WER test-en (%) | **0.99** | 1.07 | 2.21 | 1.24 | 2.25 | 1.90 | [Table 1] |
| WER zh-hard (%) | 5.99 | 17.00 | **5.83** | - | 7.59 | - | [Table 1] |

### CV3-Eval [Table 2]

| 指标 | Fish Audio S2 | Fish Audio S1 | CosyVoice 3-1.5B | 出处 |
| --- | --- | --- | --- | --- |
| WER zh (%) | **2.65** | 2.98 | 3.01 | [Table 2] |
| WER en (%) | **2.43** | 3.00 | 3.71 | [Table 2] |
| WER ja (%) | **3.96** | 4.54 | 5.27 | [Table 2] |
| 平均 WER | **~3.01** (估算) | 3.96 | - | [Table 2] |

### Long-Audio Benchmark [Table 4]

| 指标 | Fish Audio S2 | Fish Audio S1 | Qwen3-TTS | VibeVoice | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER en (%) | **4.38** | 6.26 | 7.69 | 28.0 | [Table 4] |
| CER zh (%) | **5.95** | 6.44 | 8.09 | 26.2 | [Table 4] |
| SIM-Mean en | 0.523 | 0.436 | 0.390 | **0.530** | [Table 4] |
| SIM-Std en | 0.0761 | 0.108 | 0.0737 | **0.0572** | [Table 4] |

### Audio Turing Test [Table 5]

| 指标 | Fish Audio S2 | S2 (w/ instruction) | S1 | Seed-TTS | MiniMax-Speech | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| ATT Mean | 0.483 | **0.515** | 0.479 | 0.417 | 0.387 | [Table 5] |
| 95% HDI | [0.477, 0.489] | [0.510, 0.521] | [0.471, 0.486] | [0.398, 0.438] | [0.368, 0.407] | [Table 5] |

### EmergentTTS-Eval [Table 6]

| 指标 | Fish Audio S2 | Gemini-2.5-Flash | gpt-4o-audio | 出处 |
| --- | --- | --- | --- | --- |
| Overall Win Rate (%) | **81.88** | 79.10 | 77.36 | [Table 6] |
| Paralinguistics Win (%) | **91.61** | 91.25 | 89.46 | [Table 6] |
| Questions Win (%) | **84.41** | 74.82 | 62.50 | [Table 6] |
| WER overall (%) | 8.15 | **6.35** | 7.75 | [Table 6] |

### Fish Audio Instruction Benchmark [Table 7]

| 指标 | Fish Audio S2 | Fish Audio S1 | 出处 |
| --- | --- | --- | --- |
| TAR zh | **0.984** | 0.942 | [Table 7] |
| TAR en | **0.881** | 0.626 | [Table 7] |
| Naturalness zh | **4.40** | 4.15 | [Table 7] |
| Expressiveness zh | **4.94** | 4.65 | [Table 7] |

### Inference Performance [§5]

| 指标 | 数值 | 设备 | 出处 |
| --- | --- | --- | --- |
| RTF | 0.195 | H200 | [§5] |
| TTFA | <100 ms | H200 | [§5] |
| Max throughput | 3000+ tokens/s | H200 | [§5] |
| Prefix cache hit rate | 86.4% avg, >90% peak | - | [§5] |

## 局限性

1. **消融实验缺失**: 论文不包含任何消融实验。无法区分 RVQ→GFSQ 的变化、Qwen3-4B backbone 的贡献、RL 各维度 reward 的独立贡献、dual-purpose pipeline 的实际效果。这是最大的学术短板 [agent 解读]

2. **自建 benchmark 自评**: Fish Audio Instruction Benchmark 是作者自己设计、自己评估的 benchmark [§6.3, Appendix A]。虽然做了 human-model alignment validation (Cohen's kappa 0.47),但这仅为 "moderate agreement",且未与其他系统对比 [Appendix A.3]

3. **RL 基础设施成本**: Reward 中的 ASR 模型基于 Qwen3-Omni-30B-A3B fine-tune [§3.2],远大于 TTS 模型本身 (4B)。RL 训练需要同时运行 policy model + reward scoring 系统,部署成本高 [agent 解读]

4. **Speaker Similarity 未见优势**: 在 Long-Audio benchmark 中 SIM-Mean 和 SIM-Std 均不如 VibeVoice [Table 4]。在 Minimax Multilingual Testset 中部分语言 SIM 不如 MiniMax-Speech [Table 3]

5. **Long-form 受 context 限制**: 最大 16384 context 导致需要截断超长文本,Long-TTS-Eval 的样本被限制在约 185 秒以内 [§6.1.3]

6. **非开放 reward model**: 虽然 TTS 模型开源,但 speech quality model 和 rich-transcription ASR model 均未开源,限制了社区复现 RL 训练流程 [agent 解读]

## 点评

Fish Audio S2 是一个工程完成度很高的 TTS 系统,在多个公开 benchmark 上展现了领先的 WER 和指令跟随能力。其核心学术贡献在于 **dual-purpose 数据 pipeline** — 用同一套模型同时做预训练数据过滤和 RL reward 信号,从设计上消除了两阶段之间的分布偏移。这个思路简洁有效,值得其他 TTS RL 工作借鉴。

然而,论文的学术严谨性有明显不足:
- **消融缺失是硬伤**: 无法判断性能提升来自 Qwen3-4B backbone 的语言能力、10M 小时数据的规模优势、RL 对齐、还是 tokenizer 改进。与同期 CosyVoice 3 (提供了 DiffRO 消融)、VibeVoice (提供了 tokenizer 消融) 相比,实验深度差距明显。
- **S1→S2 的技术决策未解释**: GFSQ 在 S1 中被宣传为 RVQ 的优越替代,S2 却静默回归 RVQ,没有给出任何解释或对比数据。
- **ATT benchmark 的 0.515 posterior mean** 表明在 instruction rewriting 后确实接近人类水平,但 95% HDI 仅窄幅超过 0.5,统计意义有限。

从工程角度看,S2 的部署方案 (SGLang + RadixCache + LoRA weight-swap) 设计精巧,RTF 0.195 和 TTFA <100ms 在开源系统中处于领先水平。开源 model weights + fine-tuning code + inference engine 的做法也值得肯定。

**与领域内同期工作的定位**: S2 与 CosyVoice 3、Qwen3-TTS、VibeVoice 构成了 2025-2026 年 TTS 的四个代表性路线 — S2 是 "LLM-backbone + RVQ codec + GRPO" 路线; CosyVoice 3 是 "LLM→semantic tokens→CFM + DiffRO" 路线; Qwen3-TTS 是 "dual-track LM + dual tokenizer" 路线; VibeVoice 是 "continuous VAE tokenizer + next-token diffusion" 路线。

## 可复用的 idea

1. **Dual-purpose 数据 pipeline**: 同一模型先做数据标注/过滤,再做 RL reward,是消除 pre-training 和 post-training 分布偏移的通用范式。可推广到任何需要 RL 后训练的生成模型 [§3]

2. **Dr.GRPO 去标准差归一化**: 去除 intra-group standard deviation normalization 避免低 reward variance 样本获得不成比例的梯度更新。任何使用 GRPO 的场景都应考虑此变体 [§4.3, Eq 5]

3. **LoRA weight-swap for RL**: 将 reference policy 以 LoRA weight backup 存于 CPU memory,需要时动态 swap 做 KL 前向,避免双模型常驻 VRAM。适用于任何 RLHF/GRPO 训练流程 [§4.3]

4. **RadixCache 用于 voice reuse**: Reference audio 前置到 system prompt 后,SGLang 的 RadixCache 可缓存 KV states,同一 voice 多次请求时跳过 reference audio prefill (86.4% hit rate)。适用于任何基于 in-context prompt 的语音系统 [§5]

5. **Rich-transcription ASR 做 RL reward**: 用 ASR re-transcription + confidence scoring 作为 semantic accuracy reward,对 speaker ID 和 vocal instruction 加权惩罚。这比简单的 WER 计算提供更精细的 reward signal [§4.3]

6. **Token-weighted reward mask**: 在 instruction-following reward 中对不同类型的 token 施加不同权重 (speaker ID tag 错误惩罚更重,missed vocal instructions 追加惩罚),实现差异化的 reward granularity [§4.3]

---

检索命中: [[LLM-based TTS]], [[Speech Tokenizer]], [[Residual Vector Quantization]], [[Semantic vs Acoustic Tokens]] | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Codec Language Model]](pending-review) | 未命中但可能相关: 无
