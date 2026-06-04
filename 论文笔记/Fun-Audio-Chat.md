---
type: paper
tier: deep
title: "Fun-Audio-Chat Technical Report"
arxiv_id: "2512.20156"
source: "Sources/Fun-Audio-Chat.pdf"
authors: [Tongyi Fun Team, Alibaba Group]
year: 2026
venue: "arXiv"
tags: [speech-LM, LALM, full-duplex, dual-resolution, DPO, model-merging, spoken-dialogue, audio-understanding, post-training]
concepts: ["[[Speech Language Model]]", "[[Full-duplex Spoken Dialogue]]", "[[Speech Tokenizer]]", "[[Modality Adaptation for Speech LLM]]", "[[Speech-LLM Integration Taxonomy]]", "[[Conditional Flow Matching]]"]
models: ["[[CosyVoice 3]]", "[[Whisper]]"]
tasks: []
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[Speech Language Model]]✓, [[Speech Tokenizer]]✓, [[Full-duplex Spoken Dialogue]], [[Modality Adaptation for Speech LLM]], [[CosyVoice 3]], [[Speech-LLM Integration Taxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Speech Language Model]], [[Speech Tokenizer]], [[Full-duplex Spoken Dialogue]], [[Modality Adaptation for Speech LLM]], [[CosyVoice 3]], [[Speech-LLM Integration Taxonomy]] | 过滤: [[Full-duplex Spoken Dialogue]](pending-review), [[Modality Adaptation for Speech LLM]](pending-review), [[CosyVoice 3]](pending-review), [[Speech-LLM Integration Taxonomy]](pending-review) | 未命中但可能相关: 无

**谱系定位**: Fun-Audio-Chat 属于 Speech Language Model 中的 **Parallel Joint Speech-Text Model** 路线,与 Moshi 同属在 LLM 内同时生成 text 和 speech tokens 的范式。在 [[Speech-LLM Integration Taxonomy]] 中,它处于 audio-token-based integration 类别。与近期同类 LALM (Kimi-Audio, Step-Audio2-Mini, MiMo-Audio, Baichuan-Audio) 的区别在于: (1) 不做大规模 audio-text 预训练,仅做 post-training; (2) 采用 dual-resolution 架构 (5Hz LLM + 25Hz SRH) 降低计算成本。

**已有认知**: CosyVoice 3 的 S3Tokenizer (FSQ 量化, 25Hz, MinMo backbone, 530K 小时多任务监督训练) 被本文直接采用且冻结,验证了该 tokenizer 的跨场景迁移能力。Full-duplex Spoken Dialogue 领域已有 Moshi (RQ-Transformer)、FreezeOmni (chunk-level state prediction)、OmniFlatten 等系统,Fun-Audio-Chat-Duplex 的 parallel input stream 方案是又一种实现路径。

**创新判断**: DRSR 架构来自前作 DrVoice,本文的增量贡献在于 (1) 规模验证 (8B dense + 30B MoE); (2) Multi-Task DPO Training 增加鲁棒性/指令跟随/共情能力; (3) 全双工变体。

> [!summary] 速查
> - **一句话**: 阿里通义的 Parallel LALM,用 dual-resolution 架构 (5Hz LLM + 25Hz SRH) 和 Core-Cocktail Training 实现计算高效的语音对话,并引入 Multi-Task DPO 增强真实场景能力
> - **路线**: Speech(Whisper encoder → Adapter → S3Tokenizer) → Shared LLM(5Hz grouped tokens + text) → Text Head(text) + SRH(25Hz speech tokens) → Speech Detokenizer(FlowMatching + HiFi-GAN)
> - **指标**: OpenAudioBench S2T 76.61%(8B, 同规模最优) [Table 2]; VoiceBench S2T 83.21%(8B) [Table 2]; MMAU 77.9%(30B-A3B, 全模型最优) [Table 3]; 全双工 Turn-taking 100%(30B-A3B) [Table 7]; UTMOS 4.37 / ASR-WER 4.32% on Llama Q. [§3.2]; GPU hours ~50% reduction [§1]
> - **可借鉴**: Core-Cocktail Training 的中间模型合并策略 ($M_r = \alpha M_1 + (1-\alpha)M_0$, $\alpha$=0.5) 可迁移到任何多模态微调场景防灾难性遗忘; 分辨率解耦 (LLM 低帧率 + 生成头高帧率) 是通用的效率-质量平衡思路
> - **局限**: 多轮对话记忆丢失 [§5]; 语音指令跟随稳定性不足 [§5]; 共情表现不一致 [§5]; 仅做 post-training 无大规模预训练,理论上限可能受限

## 核心问题

Fun-Audio-Chat 要解决的核心问题是: 现有 joint speech-text 模型面临三个瓶颈 [§1]:
1. **时序分辨率失配**: speech tokens 通常 25Hz 而 text tokens ~3Hz,直接拼接导致 LLM 被稀释的语音帧淹没,语义建模效率低
2. **计算成本高**: 高帧率 (12.5-25Hz) 的音频 token 序列使 LLM 训练和推理成本居高不下
3. **灾难性遗忘**: 多模态训练 (continual pre-training + post-training) 常导致原始 text LLM 知识被覆盖

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Fun-Audio-Chat 是一个 **Parallel Joint Speech-Text Model**,架构包含三个主要模块 [§2, Fig 2]:

1. **语音输入端**: Speech Encoder (Whisper-Large-v3, frozen after pre-alignment) + Adapter → 将音频波形转为连续表征; S3Tokenizer (CosyVoice 3, frozen) → 将音频转为 25Hz 离散 speech tokens [§2.1]
2. **Shared LLM Layer**: 接收 text + grouped speech tokens (5Hz) 的混合序列,同时生成两种模态输出 [§2.3]
3. **输出端**: Text Head (1x forward) → text tokens; Speech Refined Head (SRH, 5x forward) → 25Hz speech tokens → Speech Detokenizer (Flow Matching + HiFi-GAN) → 语音波形 [§2.1, §2.2]

**Parallel 设计的关键**: 模态对齐仅在 assistant 侧执行 — 用户通常提供单模态输入 (speech 或 text),而 assistant 同时输出 speech + text [论文原文, §2.3]。两种 token 的 embedding 通过加法融合: $c_t = E_{speech}(s_t) + E_{text}(t_t)$ [Eq. 5],长度不匹配时用 `<|SIL|>` 填充 [§2.3]。

### 关键设计选择

#### 1. Dual-Resolution Speech Representations (DRSR) [§2.2]

**问题**: 25Hz speech token 的序列长度远超 text token (~3Hz),LLM 处理效率低 [论文原文]。

**方案**: 将 25Hz speech tokens 按 k=5 分组,通过线性投影映射为 5Hz grouped representations 供 LLM 处理:
$$\mathbf{g}_i = \text{Linear}(\text{Concat}_{j=ik}^{(i+1)k-1}(\mathbf{s}_j)) \in \mathbb{R}^{d_{text}}$$
[Eq. 1]

这使 LLM backbone 的输入帧率降到 5Hz,与 text token rate 更匹配,同时序列长度缩短 5 倍 [论文原文]。

**生成端恢复精度**: SRH 执行逆操作 (ungrouping) — 将 LLM 最终隐状态线性投影后拆分为 k=5 个 segment,每个 segment 自回归地预测 25Hz speech tokens [Eq. 2-4]:
$$\mathbf{h}_{ug} = \mathbf{W}_p \mathbf{h}_L^{[SLLM]}$$
$$\mathbf{H} = \text{Split}_k(\mathbf{h}_{ug}) = [\mathbf{h}_{ug}^{(1)}, ..., \mathbf{h}_{ug}^{(k)}]$$
SRH 的训练损失是标准的 next-token prediction loss: $\mathcal{L}_{SRH} = -\sum_{i=1}^T \log P(s_i | s_{<i}, \mathbf{H}_{<i})$ [Eq. 4]

**为什么这样设计**: [agent 解读] 这种 "LLM 做粗粒度语义规划, SRH 做细粒度声学细化" 的分工本质上是 coarse-to-fine 思路在实时 LALM 中的自然延伸,与 CosyVoice 的 LLM + CFM 两阶段在精神上相通,但将两阶段压缩到单次 forward pass 中。

#### 2. Core-Cocktail Training [§2.4]

**问题**: 多模态微调中,高 LR 损害 text LLM 知识,低 LR 收敛慢 [论文原文]。

**方案** (来自 DrVoice, Tan et al. 2025): 两阶段训练 + 中间模型合并:
- **Stage 1**: 高 LR 微调 (1e-4 → 1e-5, cosine annealing),快速适应多模态 [§2.4]
- **中间模型合并**: $M_r \leftarrow \alpha M_1 + (1-\alpha) M_0$,其中 $\alpha=0.5$,将 Stage 1 模型与原始预训练 LLM 加权混合,恢复被冲刷的知识 [Eq. 7, §2.4]
- **Stage 2**: 低 LR 精调 (1e-5 → 1e-6),在合并模型上稳定优化 [§2.4]

**为什么 α=0.5**: [论文原文] "Lower α values favor stronger retention of the base LLM's knowledge" [§2.4]。α=0.5 在知识保留和多模态适应之间取折中。

#### 3. Multi-Task DPO Training [§2.4]

Core-Cocktail 之后的对齐阶段,使用四维度偏好学习:
1. **Robustness preference**: 在噪声/多样输入下保持质量 [§2.4]
2. **Instruction-following preference**: 准确执行情感/风格/韵律等语音指令 [§2.4]
3. **Audio understanding preference**: 准确理解音频内容 [§2.4]
4. **Voice empathy preference**: 适当的情感理解和共情响应 [§2.4]

**为什么做 Multi-Task DPO 而不是单任务 SFT**: [agent 解读] SFT 只能教模型 "做什么",DPO 教模型 "哪个更好",后者更适合需要在多维度 trade-off 的对话系统。通过统一 DPO loss 跨维度计算,模型学到的是一个平衡的偏好信号而非针对单一指标优化。

### 训练策略

完整训练管线包含三阶段 [§2.4]:

1. **Pre-alignment**: 大规模 speech-text 配对数据,训练 Speech Encoder + Adapter + SRH,Shared LLM 冻结 [§2.4]
   - Speech Encoder: 初始化自 Whisper-Large-v3
   - Shared LLM: 初始化自 Qwen3-30B-A3B 或 Qwen3-VL-8B
   - S3Tokenizer + Speech Detokenizer: 来自 CosyVoice 3,全程冻结
   
2. **Core-Cocktail Training**: 全参数微调,使用 CosyVoice 3 合成的高质量数据 (按 WER 筛选) + 开源数据 + 内部数据 [§2.4]
   - 数据: 百万小时级多样语音 (对话/多语言/ASR/TTS/音频理解/指令跟随/共情)
   - 最大上下文: 2048 tokens (~6 分钟语音) [§2.4]
   
3. **Multi-Task DPO Training**: 真实语音数据的偏好对齐 [§2.4]

**Full-duplex 训练** [§2.4]: 从 Core-Cocktail checkpoint 继续,引入 parallel speech-text input stream (用户和 assistant 双通道),用 OmniFlatten 方式将半双工对话数据增强为全双工训练数据。

## 实验

### Spoken Question Answering

| 指标 | Fun-Audio-Chat-8B | Fun-Audio-Chat-30B-A3B | Best Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| OpenAudioBench Overall (S2T) | **76.61** | 80.13 | 65.45 (MiMo-Audio) | OpenAudioBench | [Table 2] |
| VoiceBench Overall (S2T) | **83.21** | 88.72 | 74.06 (Kimi-Audio) | VoiceBench | [Table 2] |
| UltraEval-Audio Overall (S2S) | **59.56** | — | 48.52 (Kimi-Audio) | UltraEval-Audio | [Table 2] |
| OpenAudioBench Overall (S2T, large) | — | **80.13** | 84.94 (GPT-Audio) | OpenAudioBench | [Table 1] |
| VoiceBench Overall (S2T, large) | — | **88.72** | 90.06 (GPT-Audio) | VoiceBench | [Table 1] |

### Audio Understanding

| 指标 | Fun-Audio-Chat-8B | Fun-Audio-Chat-30B-A3B | Best Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MMAU | 76.6 | **77.9** | 74.9 (MiMo-Audio) | MMAU | [Table 3] |
| MMAU-Pro | 58.0 | **59.9** | 53.4 (MiMo-Audio) | MMAU-Pro | [Table 3] |
| MMSU | 67.8 | **70.1** | 61.7 (Kimi-Audio) | MMSU | [Table 3] |

### Speech Function Calling

| 指标 | Fun-Audio-Chat-8B | Fun-Audio-Chat-30B-A3B | Best Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Overall | 76.19 | **79.63** | 74.04 (Gemini-2.5-Pro) | 3 benchmarks | [Table 4] |
| ACEBench (Single) | 66.30 | **76.40** | 68.30 (GPT-Audio) | Speech-ACEBench | [Table 4] |
| BFCL (Parallel) | **87.63** | 86.29 | 80.91 (Doubao) | Speech-BFCL | [Table 4] |

### Speech Quality

| 指标 | Fun-Audio-Chat-8B | 数据集 | 出处 |
| --- | --- | --- | --- |
| UTMOS | 4.37 | UltraEval-Audio Llama Q. | [§3.2] |
| ASR-WER (%) | 4.32 | UltraEval-Audio Llama Q. | [§3.2] |

### Full-Duplex Interaction

| 指标 | Fun-Audio-Chat-Duplex-8B | Fun-Audio-Chat-Duplex-30B-A3B | Moshi | FreezeOmni | 出处 |
| --- | --- | --- | --- | --- | --- |
| S2M-T Avg | 49.03 | **54.89** | 33.17 | 47.58 | [Table 7] |
| S2M-S Avg | 43.44 | **49.28** | 29.86 | 34.49 | [Table 7] |
| Turn-taking Success Rate | 99.94% | **100.00%** | 99.77% | 93.87% | [Table 7] |

### 计算效率

Frame Rate-In/Out 均为 **5 Hz** [Table 1, Table 2],比其他模型 (6.25-25Hz) 低 1.25x-5x,实现 ~50% GPU hours 减少 [§3.7, §1]。

## 局限性

1. **多轮对话记忆丢失**: 复杂多轮 QA 中早期轮次信息丢失,长上下文理解受限 [§5]
2. **语音指令跟随不稳定**: 情感、风格、韵律等细粒度控制有时无法准确执行 [§5]
3. **共情能力波动**: 跨场景和情感类型的共情一致性不够可靠 [§5]
4. **[agent 解读] 仅做 post-training 的局限**: 不做大规模 audio-text 预训练,而是依赖 pre-trained text LLM + 后训练,与 Kimi-Audio/Step-Audio 的预训练路线不同。这减少了训练成本,但可能限制了音频理解的深度整合上限
5. **[agent 解读] 最大上下文 2048 tokens (~6 min)**: 在长对话场景下可能不足,但对多数语音交互足够

## 点评

**优势**:
- DRSR 架构是一个优雅的效率-质量折中方案。将 LLM 帧率降到 5Hz 而 SRH 保持 25Hz 生成精度,这种 "规划与执行分离" 的思路在其他多模态模型中也有应用前景
- Core-Cocktail Training 的模型合并策略简单有效,α=0.5 的中间点虽非最优但足够通用
- Multi-Task DPO 的四维度设计比单纯 SFT 更系统,特别是 robustness 和 empathy 维度的引入
- 全双工变体 100% turn-taking 成功率是强结果

**不足**:
- 作为 technical report,方法细节偏少: DPO 的具体数据构造(reject/chosen 样本如何生成)、Pre-alignment 的详细训练超参、SRH 的具体架构(层数/注意力机制)均未详述
- DRSR 和 Core-Cocktail Training 均来自前作 DrVoice,本文的独立贡献主要是 scale-up + DPO + full-duplex,但 DPO 部分实验不够充分(无 ablation 对比有无 DPO 的效果)
- 8B dense 模型在 VStyle 指令跟随 benchmark 上与 GPT-4o/GPT-Audio 仍有差距 (3.35/3.46 vs 4.05/3.84) [Table 5]

## 可复用的 idea

1. **分辨率解耦**: LLM 低帧率 (5Hz) + 专用生成头高帧率 (25Hz) 的思路可推广到任何 speech-text 联合模型,用于降低 LLM 处理语音序列的计算负担
2. **中间模型合并防遗忘**: $M_r = \alpha M_1 + (1-\alpha) M_0$ 在多模态微调后将模型参数与原始 LLM 混合,比 replay buffer 或 EWC 等正则化方法更简单,适合工程部署
3. **Multi-Task DPO 维度设计**: 将对齐目标拆分为 robustness / instruction-following / understanding / empathy 四维度,每个维度独立构造偏好数据,通过统一 loss 联合优化,比单一目标更全面
4. **Post-training-only 路线**: 不做大规模 audio-text 预训练,而是通过精心设计的后训练管线 (pre-alignment → SFT → DPO) 达到竞争力,显著降低训练总成本
