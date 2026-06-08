---
type: paper
tier: deep
title: "ERNIE 5.0 Technical Report"
arxiv_id: "2602.04705"
source: "Sources/ERNIE5.0.pdf"
authors: [ERNIE Team, Baidu]
year: 2026
venue: "arXiv"
tags: [multimodal, unified-model, MoE, autoregressive, audio-generation, audio-understanding, TTS, ASR, elastic-training, reinforcement-learning, trillion-parameter, omni-model]
concepts: ["[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[LLM-basedTTS]]", "[[ResidualVectorQuantization]]"]
models: ["[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[LLM-basedTTS]], [[ResidualVectorQuantization]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[LLM-basedTTS]]✓, [[ResidualVectorQuantization]]✓ | 过滤: [[ModalityAdaptationforSpeechLLM]](pending-review), [[AudioUnderstanding]](pending-review), [[CodecLanguageModel]](pending-review) | 未命中但可能相关: DiffusionModel (用于视觉 refiner)
>
> **Speech Language Model**: ERNIE 5.0 超越了 KB 中典型 SpeechLM 的范畴 -- 它不是语音专用基础模型,而是一个原生多模态 (text+image+video+audio) 统一自回归模型。KB 中 SpeechLM 的核心范式是 "speech tokenizer → LM → vocoder" 三组件管线,ERNIE 5.0 将音频组件作为统一 backbone 中的一个模态分支,所有模态共享同一个 MoE backbone,没有独立的 speech LM。这代表了 SpeechLM 演进中 "omni-model" 路线的工业级落地,与 Moshi/VITA 等系统走同一方向但规模大得多 (万亿参数)。
>
> **Speech Tokenizer**: ERNIE 5.0 的 audio tokenizer 采用 12.5 Hz RVQ 设计,第一层 token 通过 Whisper 蒸馏对齐语义表征,后续层编码残差声学信息。这在 KB 的 tokenizer 分类中属于 "混合 tokenizer" 路线 (类 SpeechTokenizer/Mimi 的思路: 第一层语义 + 后续层声学),但采用蒸馏而非自监督学习来获取语义信息。12.5 Hz 帧率与 FireRedTTS 2 相同,是当前主流 25 Hz 方案的一半,进一步压缩序列长度以适配万亿参数 backbone。
>
> **LLM-based TTS**: ERNIE 5.0 的 TTS 能力通过 NCP (Next-Codec Prediction) 实现,在 SEED-TTS 上 WER 1.35% (zh) / 1.54% (en),与 Qwen3-Omni (1.07%/1.39%) 相当,但弱于 CosyVoice 3 (0.71%/1.45%)。这符合 KB 中记录的 "统一 omni-model 在 TTS 上通常不及专用 TTS 系统" 的规律 -- 统一训练牺牲了部分 TTS 专精性能,换取了跨模态能力。
>
> **Residual Vector Quantization**: ERNIE 5.0 的 audio quantization 采用标准 RVQ 设计,但生成方式独特 -- 不像 VALL-E 那样将 RVQ 层展开为 AR+NAR 两阶段序列,而是采用 depth-wise 架构: 不同 RVQ 层的预测分布在 backbone 的不同 transformer 层中,每个 transformer 层负责一个 codec level。这与 KB 中记录的 RVQ 建模范式 (时间轴 AR + codebook 轴 NAR) 是正交的设计选择。

> [!summary] 速查
> - **一句话**: 首个公开的万亿参数统一自回归基础模型,通过超稀疏 MoE + 模态无关路由 + 弹性训练,在单一 next-group-of-tokens prediction 框架下实现 text/image/video/audio 的理解和生成,且可从单次预训练中提取多种规模的子模型 [§Abstract]
> - **路线**: Text/Vision/Audio → 模态专用 Tokenizer (text BPE / causal 3D visual VQ / 12.5Hz RVQ audio codec) → 统一 embedding → 超稀疏 MoE Backbone (<3% activation, modality-agnostic routing) → 模态专用 Head/Decoder → 输出 (text / image / video / waveform) [§2, Fig 1]
> - **指标**: TTS WER 1.35% zh / 1.54% en (SEED-TTS, post-trained) [Table 8]; ASR WER 0.31 AISHELL-1 / 1.16|2.61 LibriSpeech (post-trained, best among omni-models) [Table 7]; VoiceBench AlpacaEval 4.94 / MMSU 81.95 [Table 7]; Audio Understanding MMAU 75.90 [Table 7]; Text MMLU-Pro 83.80 / AIME 2025 89.06 (post-trained) [Table 2]; Vision GenEval 90.1 / VBench-Semantic 83.40 [Table 5,6]; 弹性训练: 35.8% 参数子模型 avg score 75.17 vs 全模型 75.55 [Table 12]; 25% top-k routing 获 >15% decoding speedup,精度损失极小 [§6.4.2]
> - **可借鉴**: (1) **Depth-wise NCP**: 将 RVQ 多层 codec 的预测分布到 backbone 不同 transformer 层,避免展平为长序列,适用于任何 RVQ-based 音频生成 [§2.3.2]; (2) **弹性训练三维度**: depth/width/sparsity 同时训练,单次预训练出多规格子模型,对 MoE 模型部署灵活性极有价值 [§3.3]; (3) **AHRL (Adaptive Hint-based RL)**: 对难题注入部分 think skeleton 引导,随训练进展渐退,解决 RL 中困难任务零奖励问题 [§4.3]; (4) **U-RB (Unbiased Replay Buffer)**: 解决异步 RL 中长尾 rollout 导致的数据分布偏差 [§4.1]; (5) **音频 semantic token Whisper 蒸馏**: 将 Whisper encoder 输出对齐到第一层 RVQ token,使语义信息集中于第一层,对设计 semantic-aware codec 有参考价值 [§2.3.1]
> - **局限**: 非开源,无法复现; TTS 质量弱于 CosyVoice 3 等专用系统 (zh WER 1.35 vs 0.71) [Table 8]; 音频评测仅覆盖 ASR/VoiceBench/MMAU 三类,缺乏 SIM/MOS/PESQ 等声学质量指标; 弹性训练的 ablation 仅在小模型 (3.2B) 上验证,未展示万亿参数上的详细消融; 论文未公开具体参数量、层数、专家数等关键架构细节; 音频只覆盖 speech,对 music/sound effect 等通用音频的生成能力未评测

## 核心问题

ERNIE 5.0 试图解决当前多模态模型的三个核心瓶颈 [§1]:

1. **理解-生成脱节**: 现有系统通常将多模态理解和生成解耦,通过 late-fusion 接入模态专用解码器,导致跨模态集成深度不足,且常出现 "ability seesaw" (一个模态能力提升时另一个下降) [论文原文]
2. **模型-部署刚性**: 万亿参数模型训练成本极高,但部署场景的算力/内存/延迟约束各异。传统 "先训练再压缩" 的流程需要为每种部署规格重复压缩过程 [论文原文]
3. **多模态 RL 不稳定**: 将 RL 扩展到统一多模态模型面临采样偏差、稀疏奖励和熵坍塌等问题,超稀疏 MoE 进一步放大了训练-推理不一致性 [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ERNIE 5.0 是一个原生统一自回归模型,所有模态 (text/image/video/audio) 在一个共享的 MoE backbone 中从头联合训练 [§2, Fig 1]:

- **Backbone**: 超稀疏 MoE Transformer,激活率 < 3%,模态无关路由 (routing 仅看 token embedding,不看模态标识) [§2.1]
- **统一目标**: Next-Group-of-Tokens Prediction -- text 用 NTP+MTP,vision 用 NFSP (Next-Frame-and-Scale Prediction),audio 用 NCP (Next-Codec Prediction) [§2.1]
- **Input/Output**: 模态专用 tokenizer + embedding 层 → 共享 backbone → 模态专用 head + decoder [Fig 1]

**为什么不用 late-fusion?** [§1] 论文认为后融合 (先训好 LLM 再接视觉/音频解码器) 存在两个根本问题:
- 模态解耦导致理解和生成使用不同的优化目标,阻碍深层跨模态交互 [论文原文]
- 后接模态导致 "ability seesaw": 新增模态时核心文本能力常下降 [论文原文]

**为什么选 modality-agnostic routing 而非 modality-specific?** [§2.1] 论文指出:
- 模态隔离路由 (ERNIE 4.5 采用的策略) 需要人工预分配专家给各模态,涉及 >2 个模态时非常困难 [论文原文]
- 无关路由促进跨模态知识泛化,且通过共享专家的涌现特化提升单模态性能 [论文原文]
- 实验验证: 模型自发形成了按任务 (而非模态) 分化的专家使用模式 [§6.4.1, Fig 8]

[agent 解读] 这个设计选择的深层逻辑是: 模态间共享的知识 (如语义理解) 应由共享专家处理,模态特有的知识 (如声学细节) 应由专有专家处理,而这种分工应让路由器自动发现,而非人工指定。

### 关键设计选择

#### Audio Tokenization [§2.3.1]

音频信号通过 codec-style tokenizer 离散化,帧率 12.5 Hz,采用 RVQ 多层量化:

- **第一层**: 显式编码高层语义信息 (linguistic + phonetic),通过从预训练 Whisper encoder 蒸馏对齐 [论文原文]
- **后续层**: 编码残差声学信息,逐层细化 timbre/prosody 等 [论文原文]
- Whisper 表征通过 average pooling 降采样到 12.5 Hz 以匹配 tokenizer 帧率 [§2.3.1]

**为什么蒸馏 Whisper 到第一层?** [§2.3.1] 论文的理由是: 确保第一层 token 包含足够丰富的语义信息 (linguistic/phonetic cues),使其可直接参与 audio-text 联合建模 [论文原文]

[agent 解读] 这实质上复用了 semantic vs acoustic token 分层思想 (如 SpeechTokenizer 的第一层语义 + 后续层声学),但用蒸馏替代对比学习,且蒸馏来源是 ASR 模型而非 SSL 模型,可能提供更强的 linguistic structure。

#### Audio Understanding: Depth-wise Additive Embedding [§2.3.2]

理解任务中,音频输入的多层 RVQ code 通过 level-specific embedding 层映射后逐层求和:

```
Audio token = sum(Embed_1(code_1), Embed_2(code_2), ..., Embed_N(code_N))
```

**为什么用加法而非拼接?** [§2.3.2] 加法聚合反映了 RVQ 的残差本质: 每一层都是对前一层残差的补充信息,加法自然地将多层信息融合为统一表征 [论文原文]

#### Audio Generation: Next-Codec Prediction (NCP) [§2.3.2]

NCP 是 ERNIE 5.0 音频生成的核心创新。与传统 AR+NAR 两阶段 (如 VALL-E 将 RVQ 层展开为序列) 不同,NCP 将多层 codec 的预测分布到 backbone 的不同 transformer 层 [§2.3.2, Fig 3]:

1. 多个 audio head 插入到 backbone 的顶部 transformer 层 [论文原文]
2. 模型首先预测第一层 semantic code,然后依次预测后续层 [论文原文]
3. 每预测一层 code,将其 embedding 加回 hidden state,作为下一层预测的条件 [论文原文]
4. 训练时使用 teacher forcing (ground-truth code 的 embedding 反馈) [论文原文]
5. TTS 中,speaker embedding 作为 conditioning context 的一部分注入,控制音色但不改变 NCP 结构 [§2.3.2]

**NCP 为什么优于展平序列?** [§2.3.2] 展平多 codebook tokens 为单一序列会导致序列长度成倍增长 (prohibitive sequence length),NCP 通过在 depth 维度结构化预测避免了这个问题 [论文原文]

[agent 解读] NCP 的思想与 Moshi 的 RQ-Transformer 和 AudioLM 的 coarse-to-fine 有相似之处,但关键区别在于: NCP 利用了 backbone transformer 的层级结构本身来实现深度方向的依赖,而非引入额外的生成步骤或独立模型。这在超大 backbone 中特别高效,因为顶部层的计算资源被复用于多层 codec 预测。

#### 弹性训练 (Elastic Training) [§3.3]

一次预训练同时优化全模型和多种子网络配置,在三个正交维度引入弹性 [Fig 4]:

| 维度 | 训练策略 | 推理效果 |
|------|----------|----------|
| **Elastic Depth** | 75% 全深度, 25% 随机减层 | 减少层数的子模型可直接部署 |
| **Elastic Width** | 80% 全专家, 20% 随机子集专家 | 减少总专家数以降低内存 |
| **Elastic Sparsity** | 80% 默认 top-k, 20% 随机减小 k | 减少激活专家数以加速推理 |

**为什么弹性训练优于 "先训练再压缩"?** [§3.3] 论文给出两个理由:
- 压缩需要专门基础设施和大量计算,且每种目标规格都要重复 [论文原文]
- 弹性训练使子网络在预训练中就获得了全模型的知识,子网络可直接作为 mid-training/fine-tuning 的起点 [论文原文]

### 训练策略

#### 预训练 [§3.1-3.2]

- **数据**: 所有模态从训练起步就同时参与 (非分阶段加入),包含 text/image-text/video-text/audio-text 以及 interleaved multimodal 序列 [§3.1]
- **分阶段扩展**: Stage 1 (8K context, WSD schedule) → Stage 2 (32K → 128K context, cosine decay) [§3.2]
- **MoE 特定优化**: 使用 auxiliary-loss-free load balancing,bias update speed 在 mid-training 降 10x 以抑制震荡 [§3.2]
- **多模态平衡**: posterior-based loss weighting 将不同模态的 AR loss 归一化到同一区间 [§3.2]
- **Visual tokenizer 渐进替换**: 从低 bit (小词表) tokenizer 起步,渐进切换到高 bit (大词表) tokenizer,使视觉生成训练更稳定 [§2.2.1]

#### 后训练 (Post-Training) [§4]

两阶段: SFT → 统一多模态 RL (UM-RL)。RL 阶段面临的三个核心挑战及解决方案:

**挑战 1: Rollout 效率低** [§4.1]
- 问题: 长尾 rollout 阻塞整个 batch (>90% 训练时间在 rollout) [论文原文]
- 解决: U-RB (Unbiased Replay Buffer) -- 在等待长尾 query 完成时准备未来 batch 的 rollout,保持 query 顺序以避免数据难度分布偏移 (区别于 APRIL 的截断策略) [§4.1, Fig 5]

**挑战 2: 熵坍塌** [§4.2]
- 问题: MoE 模型的训练-推理不一致放大了 importance sampling 偏差; 早期过拟合简单 query 加速熵坍塌 [论文原文]
- 解决 1: MISC (Multi-granularity Importance Sampling Clipping) -- 在 GSPO+IcePop 基础上引入 token 级 masking 校准 (而非仅 sequence 级),避免低熵 response 被大量裁剪 [§4.2, Eq. 3]
- 解决 2: WPSM (Well-learned Positive Sample Mask) -- 当 query 的成功率超过阈值 τ 且 response 熵低于 η 时,标记为 "well-learned" 并降低梯度权重,将 gradient budget 转向更难的样本 [§4.2, Eq. 4]

**挑战 3: 困难任务稀疏奖励** [§4.3]
- 问题: 所有 rollout 获得零奖励时,GRPO 无法提供有效梯度信号 [论文原文]
- 解决: AHRL (Adaptive Hint-based RL) -- 对困难 query 注入部分 think skeleton (前 p_hint 比例的 thinking 过程),按退火 schedule 随训练进展渐减 [§4.3, Eq. 5]

[agent 解读] AHRL 本质上是一种 curriculum learning 策略: 难题先给 "提示",模型从提示继续推理,随着能力增长逐步撤除提示。相比 reward shaping 或 rejection sampling,这种方式更直接且不改变奖励函数。

## 实验

### 音频相关结果

| 指标 | ERNIE 5.0 | Kimi Audio | GPT-4o Audio | Qwen3-Omni | Gemini-3-Pro | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ASR WER ↓ | 0.31 | 0.60 | 3.52 | 0.84 | 0.75 | AISHELL-1 | Table 7 |
| ASR WER ↓ | 1.16 / 2.61 | 1.28 / 2.42 | 1.39 / 3.75 | 1.22 / 2.48 | 1.47 / 3.73 | LibriSpeech c/o | Table 7 |
| VoiceBench AlpacaEval ↑ | 4.94 | 4.46 | 4.73 | 4.74 | 4.80 | VoiceBench | Table 7 |
| VoiceBench MMSU ↑ | 81.95 | 62.17 | 78.90 | 69.00 | 80.80 | VoiceBench | Table 7 |
| MMAU ↑ | 75.90 | 65.20 | 68.40 | 77.50 | 80.40 | MMAU | Table 7 |
| TTS WER ↓ | 1.35 / 1.54 | - | - | 1.07 / 1.39 | - | SEED-TTS zh/en | Table 8 |

### TTS 详细对比

| 模型 | SEED-TTS test-zh WER ↓ | SEED-TTS test-en WER ↓ | 出处 |
| --- | --- | --- | --- |
| ERNIE 5.0 | 1.35 | 1.54 | Table 8 |
| ERNIE 5.0-Base | 3.41 | 2.44 | Table 8 |
| CosyVoice 3 | **0.71** | 1.45 | Table 8 |
| Qwen3-Omni | 1.07 | **1.39** | Table 8 |
| MaskGCT | 2.27 | 2.62 | Table 8 |
| F5-TTS | 1.56 | 1.83 | Table 8 |
| CosyVoice 2 | 1.45 | 2.57 | Table 8 |
| Seed-TTS (RL) | 1.00 | 1.94 | Table 8 |

### 弹性训练

| 配置 | 激活参数比 | 总参数比 | Avg Score | vs 全模型 | 出处 |
| --- | --- | --- | --- | --- | --- |
| ERNIE 5.0-Exp (full) | 100% | 100% | 75.55 | baseline | Table 12 |
| ES 25% (仅减 top-k) | ~25% routing | 100% | 74.43 | -1.12 | Table 12 |
| EA 35.8% (depth+width+sparsity) | 53.7% | 35.8% | 75.17 | -0.38 | Table 12 |

### Expert Routing 分析 [§6.4.1]

- **专家利用非均匀**: 部分专家跨所有模态被频繁激活 (共享知识),其余专家呈现强模态专有模式 [Fig 8]
- **视觉/音频专家更集中**: image/video/audio 的专家激活比 text 更集中于少数专家 [Fig 8]
- **理解-生成分离**: 视觉理解和视觉生成的 top-25% 专家重叠度低 (IoU 低),没有倾向于共享专家 [Fig 9]
- **跨模态协作随深度增加**: text 与 audio 的专家重叠在深层更高,表明深层表征趋向统一语义 [Fig 9]
- **负载均衡**: text 跨层 NE (normalized entropy) 稳定且高; 视觉生成和音频任务在低层和高层 NE 下降,中间层恢复 [Fig 10]

## 局限性

1. **TTS 质量差距**: 统一模型 TTS 显著弱于 CosyVoice 3 (zh WER 1.35 vs 0.71),仅报告了 content consistency (WER),未报告 speaker similarity、MOS、PESQ 等声学质量指标,无法评估实际听感 [Table 8]
2. **音频评测覆盖不足**: 仅评测了 ASR、VoiceBench (对话) 和 MMAU/TUT/CochlScene (理解),缺少音乐生成/理解、语音编辑、多说话人等评测 [Table 7]
3. **架构细节缺失**: 论文未公开 backbone 的层数、隐藏维度、专家数量、每个模态的 head 数等关键架构参数,可复现性为零 [agent 解读]
4. **弹性训练验证不充分**: 三维弹性的 ablation 在 3.2B 小模型上完成,万亿参数上仅展示了最终结果,不确定 scaling behavior 是否一致 [§6.4.2]
5. **Base 模型 TTS 极弱**: ERNIE 5.0-Base 的 TTS WER 3.41% (zh) 远差于大多数专用系统和 post-trained 版本,说明 TTS 能力高度依赖 SFT/RL 后训练,预训练阶段对音频生成的效果有限 [Table 8]
6. **speaker embedding 方式未详述**: 仅提到 "speaker embedding is inserted as part of conditioning context" [§2.3.2],未说明 embedding 来源、维度、注入方式等关键 TTS 实现细节

## 点评

ERNIE 5.0 的核心贡献在工程规模上: 首次公开披露万亿参数级统一自回归模型实现文本/图像/视频/音频的理解和生成。从 TTS/音频视角看:

**NCP (Next-Codec Prediction) 是最有技术启发性的设计**。将 RVQ 多层预测映射到 backbone transformer 的不同层,是一种优雅的 "复用计算资源" 思路。传统方案 (VALL-E 的 AR+NAR、Moshi 的 RQ-Transformer) 要么需要额外模型,要么展平为长序列,NCP 利用了深度 transformer 本身就有的层级抽象能力。这个思路可以迁移到任何多层 codec 的生成场景。

**弹性训练的实用价值高于学术新颖性**。三维弹性 (depth/width/sparsity) 各自并非新概念,但首次在万亿参数 MoE 上联合应用并验证了 "35.8% 参数保持 99.5% 性能" 的结果,对工业部署有实际参考价值。

**RL 三件套 (U-RB + MISC + AHRL) 值得关注**。这些技术虽然各自增量改进不大,但组合后解决了大规模多模态 RL 的稳定性问题。AHRL 的 think skeleton 注入思路对其他需要 RL 的语音模型 (如 Step-Audio 的 RLHF) 可能有参考价值。

**Modality-agnostic routing 的实证分析 (§6.4.1) 是论文最有价值的贡献之一**。Fig 8-10 展示了专家自发形成按任务分化 (而非按模态分化) 的模式,以及 audio 与 text 在深层趋向共享专家,这些发现对未来设计统一多模态架构有指导意义。

**不足之处**: 作为 technical report,论文在音频方面的评测深度明显不如文本和视觉。缺少 MOS/SIM 等主观评测、缺少多说话人/情感控制/音乐生成等细分评测,使得 "统一模型是否真正保留了细粒度音频生成能力" 这个关键问题无法回答。

## 可复用的 idea

1. **Depth-wise NCP for RVQ generation**: 将多层 codec 预测分散到不同 transformer 层,避免长序列问题。实现要点: 在 backbone 顶部 N 层各加一个 prediction head,每层 head 预测一个 RVQ level,预测结果的 embedding 加回 hidden state 作为下一层条件。适用于任何使用 RVQ 且 backbone 足够深的系统 [§2.3.2]

2. **Audio semantic token Whisper distillation**: 用 Whisper encoder 输出作为 teacher,通过 average pooling 对齐帧率后蒸馏到 RVQ 第一层,使第一层 token 集中编码 linguistic/phonetic 信息。比自监督 k-means 聚类 (HuBERT) 更直接,且继承了 Whisper 680K 小时多语言训练的知识 [§2.3.1]

3. **弹性 top-k routing for inference speedup**: 在 MoE 训练中以 80/20 比例混入不同 top-k 配置,推理时降低 top-k 到 25% 即获 >15% speedup,性能损失极小。这是最易迁移的 MoE 加速技巧,无需改模型结构,只需修改训练时的采样策略 [§3.3, Table 12]

4. **AHRL think skeleton injection**: 对 RL 中零奖励的困难 query,在 prompt 后注入部分 thinking 过程 (前 p_hint 比例),按退火 schedule 渐减。可迁移到任何 RLHF/GRPO 训练中遇到 sparse reward 的场景 [§4.3, Eq. 5]

5. **Posterior-based loss weighting for multimodal balance**: 将不同模态的 AR loss 通过后验统计归一化到同一区间,避免某个模态 loss 过大主导优化方向。对联合训练多模态 (如 speech+text) 的系统有参考价值 [§3.2]

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | NCP/弹性训练/RL 三件套的因果解释清晰,含 WHY |
> | 可信赖 | pass | 数字型 claim 标注率 >80%,指标方向正确 |
> | 可区分 | pass | [论文原文] vs [agent 解读] 标注覆盖率 >80% |
> | 可定位 | pass | KB 背景含具体谱系对比 (vs VALL-E/Moshi/CosyVoice 3) |
> | 不污染 | pass | 无新建概念页需求,不触发反向更新 |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> 详见 `_review/ERNIE5.0-review.yml`
