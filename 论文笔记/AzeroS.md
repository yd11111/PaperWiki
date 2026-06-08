---
type: paper
tier: deep
title: "AzeroS: Extending LLM to Speech with Self-Generated Instruction-Free Tuning"
arxiv_id: "2601.06086"
source: "Sources/AzeroS.pdf"
authors: [Yiwen Shao, Wei Liu, Jiahong Li, Tianzi Wang, Kun Wei, Meng Yu, Dong Yu]
year: 2025
venue: "arXiv preprint"
tags: [speech-LM, instruction-free, modality-adaptation, paralinguistic, dual-encoder, self-generated-data, alignment, audio-understanding, Qwen2.5, projection, speech-aware]
concepts: ["[[SpeechLanguageModel]]", "[[ModalityAdaptationforSpeechLLM]]", "[[AudioUnderstanding]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

> [!review] 审阅: pass, 0 high / 0 medium / 3 low (2026-06-08, auto inline)
> 详见 `_review/AzeroS-review.yml`

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: AzeroS 属于 latent-representation-based Speech-LLM integration 路线 [Speech-LLMIntegrationTaxonomy],采用 speech-aware 架构 (speech+text in, text out),而非 audio-token-based 路线。与 SALMONN / Qwen-Audio 等系统同属 "冻结 LLM + 可训练 adapter" 的轻量适配范式 [ModalityAdaptationforSpeechLLM],但在训练策略上根本不同: 这些系统依赖大规模 task-specific instruction tuning (TSIT),AzeroS 则完全移除 instruction,依靠冻结 LLM 自身生成监督信号。
>
> **已有认知**: KB 的 Modality Adaptation 页记录了三种主要适配方法 (Conv downsampling, CTC compression, Q-Former)。AzeroS 使用最简单的 linear projector (2-layer MLP with ReLU, 23.8M),不同于 SALMONN 的 Q-Former。KB 中 AudioUnderstanding 页区分了 semantic / speaker / paralinguistic 三类理解任务; AzeroS 是少数同时处理 semantic 和 paralinguistic 的系统。SpeechLanguageModel (confirmed) 页记录了 SpeechLM 的三大组件 (tokenizer/LM/vocoder) 和训练阶段分类; AzeroS 对应 "instruction-tuning" 阶段但去掉了 instruction,构成新范式。
>
> **创新判断**: AzeroS 的核心创新不在架构 (标准 encoder-projector-LLM) 而在训练范式: Self-Generated Instruction-Free Tuning (SIFT) 是一个原理性贡献,论证了去掉 instruction 后 encoder 被迫做 "全信息对齐" 而非 "task-specific 部分对齐",从而获得更好的零样本泛化。KB 中尚无 SIFT 或 instruction-free alignment 的专门概念页。与 SALMONN 的关键对比: SALMONN 用 TSIT + activation tuning 解决 task over-fitting; AzeroS 从根源消除 instruction bias,路线截然不同。
>
> 检索命中: [[SpeechLanguageModel]]✓, [[ModalityAdaptationforSpeechLLM]][待确认], [[AudioUnderstanding]][待确认], [[Speech-LLMIntegrationTaxonomy]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认], [[Whisper]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过 Self-Generated Instruction-Free Tuning (SIFT) 范式,仅训练两个轻量 projector (共 47.6M 参数) 就让冻结的 Qwen2.5-7B 同时理解语音语义和副语言信息,在 VoiceBench 和 AIR-Bench 上取得 SOTA
> - **路线**: Speech → TTA encoder (semantic, 153M, frozen) + Auden-Voice encoder (paralinguistic, 153M, frozen) → 2x linear projector (23.8M each, trainable) → Qwen2.5-7B-Instruct (frozen) → text response
> - **指标**: VoiceBench overall 73.13 (vs text-only 上界 77.52, cascaded Whisper+Qwen2.5 76.05); AIR-Bench Foundation avg 77.25; AIR-Bench Chat 8.28 (超过 GPT-4o 的 7.53 和 Qwen2.5-Omni 的 7.85) [Table 5]
> - **可借鉴**: (1) SIFT 范式 -- 去掉 instruction 强迫 encoder 做全信息对齐,理论简洁且实验有效; (2) 冻结 LLM 的 self-generated data 替代人工标注; (3) 两阶段渐进对齐 (先 semantic 后 paralinguistic) 比一阶段效果好
> - **局限**: 仅做 speech-aware (text output),不生成语音; 公开数据仅 ~25k 小时; 未验证更大 LLM backbone 的 scaling; SIFT 在 self-elicit 条件不满足时退化 (需 SIT fallback)

## 核心问题

本文要解决的核心问题是: **如何让 Speech-LLM 在不依赖大规模 task-specific instruction data 的情况下,同时实现强的语义理解和副语言理解,并泛化到未见过的任务?**

现有 Speech-LLM 训练范式 (Task-Specific Instruction Tuning, TSIT) 存在两个根本缺陷 [§3.1]:
1. **Instruction-Induced Feature Collapse**: 按特定任务 instruction 优化时,encoder 只学提取该任务所需的最小特征子集 (如 ASR 只学 phonemes),丢弃其余信息 (emotion, speaker identity)
2. **Instruction Overfitting / Modality Neglect**: 模型学到 instruction 和 response 之间的 spurious correlation (P(y|i)),绕过实际 audio 输入

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

AzeroS 采用标准的 encoder-projector-LLM 三层架构 [§4.1, Fig 1]:

```
Speech signal x
    ├── TTA encoder (semantic, 153M Zipformer, 358k hrs, frozen)
    │     → 768-dim @ 25 Hz → 4x downsample
    │     → Linear Projector (23.8M, 2-layer MLP + ReLU) → LLM embedding space @ 6.25 Hz
    │
    └── Auden-Voice encoder (paralinguistic, 153M Zipformer, 8k hrs, frozen)
          → 768-dim @ 25 Hz → 4x downsample
          → Linear Projector (23.8M, 2-layer MLP + ReLU) → LLM embedding space @ 6.25 Hz
          
    → Qwen2.5-7B-Instruct (frozen) → text response
```

LLM backbone: Qwen2.5-7B-Instruct,完全冻结。两个 encoder 也完全冻结。**唯一可训练的参数是两个 projector,共 47.6M** [论文原文, §4.1]。

### 关键设计选择

#### 1. SIFT: Self-Generated Instruction-Free Tuning [§3.3-3.5]

这是本文的核心方法论贡献。

**数据构造**: 给定 speech-text pair (x, h_tilde(x)),其中 h_tilde(x) 是语音的 oracle 文本表示 (如 transcript + metadata),将 h_tilde(x) 直接送入冻结 LLM 生成 supervision target y:

```
y = g_phi(h_tilde(x), i=∅)   [Eq. 1, i=∅ means no instruction]
```

**训练**: 优化 projector 参数 theta 使 audio 投影后的表征"骗过" LLM 产生相同的 y:

```
theta* = argmin L_CE(y, g_phi(h_theta(x), ∅))   [Eq. 3]
```

**为什么 instruction-free 泛化最好** [论文原文, §3.5]: 移除 instruction 后,loss 的唯一梯度通道是 encoder → projector。要最小化 loss,projector 必须实现 h_theta(x) ≈ h_tilde(x) 的 **全局对齐** (global alignment) [Eq. 4],而非 task-dependent 的部分对齐。数学上,阻断 instruction channel 迫使 encoder 解决最难的对齐问题 [论文原文]。

**两个设计目标** [§3.3]:
- **Goal 1: Maximizing Information in h_tilde(x)** -- oracle text 应尽可能包含语音的完整信息 (内容 + 情感 + 说话人风格)
- **Goal 2: Self-Elicit Capability** -- 冻结 LLM 在无 instruction 时能自发生成覆盖 h_tilde(x) 全部信息的 response

[agent 解读] SIFT 的本质是将 instruction-tuning 问题转化为 input-level modality alignment 问题。传统方法用 instruction 告诉模型"做什么",导致 encoder 学会偷懒; SIFT 不告诉模型做什么,逼迫 encoder 把所有信息传达给 LLM,让 LLM 自行决定输出什么。这在概念上类似于信息瓶颈理论 -- 移除 shortcut 后模型被迫学更完整的表征。

#### 2. SIT Fallback [§3.3, §5.7]

当 Self-Elicit (Goal 2) 不满足时 -- 例如注入 system prompt 使 LLM 倾向简洁对话而非详细描述 -- SIFT 的训练 target 信息量退化。此时需要 SIT (Self-Generated Instruction Tuning) 作为 fallback,用显式 instruction (如 "Describe all the information you can hear") 强制 LLM 输出完整信息 [§5.7, Fig 4]。

实验验证: Dialog-LLM case study [Table 7] 显示 SIFTssp 在 age recognition 和 MMSU 等细粒度任务上退化,混合 SIFT+SIT 可修复 [论文原文]。

#### 3. 统一框架下的先行工作定位 [Table 1]

[论文原文] 将先行工作视为 SIFT 的受限特例:

| 方法 | Oracle Info (Goal 1) | Instruction (Goal 2) | 局限 |
|------|---------------------|---------------------|------|
| AudioChatLlama | Transcript only | ∅ | 仅 semantic 对齐,忽略其他信息 |
| DeSTA-2/2.5 | Transcript + attributes | Required | 仍需 instruction,引入 instruction bias |
| **SIFT (本文)** | Full info (transcript + attributes 或 natural description) | ∅ | 理论最优: 无 instruction bias,全信息对齐 |

### 训练策略

两阶段渐进对齐 [§4.2]:

**Stage 1 (Semantic)**: 仅训练 TTA encoder 的 projector
- 数据: Data-S (22k hrs ASR corpora: WenetSpeech + GigaSpeech + CommonVoice)
- Oracle text: transcript only
- 训练配置: SIFTs (instruction-free)
- 结果: VoiceBench overall 73.63 [Table 4]

**Stage 2 (Semantic + Paralinguistic)**: 冻结 Stage 1 的 TTA branch,仅训练 Auden-Voice encoder 的 projector
- 数据: Data-SP (3.5k hrs with paralinguistic labels: IEMOCAP, CREMA-D, VoxCeleb2 等 [Table 2])
- Oracle text: transcript + paralinguistic metadata (gender, age, emotion)
- 训练配置: SIFTs + SIFTsp 混合 (同时在 Data-SP 上训练两种 SIFT)
- 结果: AIR-Bench Foundation avg 77.25, VoiceBench 维持 73.13 [Table 4]

[agent 解读] Stage 2 的一个关键发现是: 只训练 paralinguistic projector 仍会导致 VoiceBench 轻微退化 (73.63 → 72.27),混合 SIFTs + SIFTsp 训练可以修复。这说明两阶段之间存在微妙的干扰,需要通过 mixed training 缓解。

## 实验

| 指标 | AzeroS | 最强开源对比 | 闭源 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| VoiceBench Overall | 73.13 | Qwen3-Omni-30B: 68.10 | GPT-4o: 80.25 | VoiceBench | [Table 5] |
| VoiceBench (vs 同 backbone) | 73.13 | Qwen2.5-Omni: 61.45 | - | VoiceBench | [Table 5] |
| VoiceBench (vs cascaded) | 73.13 | Whisper+Qwen2.5: 76.05 | - | VoiceBench | [Table 5] |
| VoiceBench (vs text-only 上界) | 73.13 | Qwen2.5 text-only TN: 77.52 | - | VoiceBench | [Table 5] |
| AIR-Bench Foundation Avg | 77.25 | Qwen2.5-Omni: 76.50 | GPT-4o: * (rejected) | AIR-Bench | [Table 5] |
| AIR-Bench Chat | 8.28 | Qwen3-Omni-30B: 8.52 | GPT-4o: 7.53 | AIR-Bench Chat | [Table 5] |
| AIR-Bench Gender | 86.75 | Qwen2.5-Omni: 91.11 | GPT-4o: 90.7 | AIR-Bench | [Table 5] |
| AIR-Bench Emotion | 71.45 | Qwen3-Omni-30B: 62.20 | GPT-4o: 49.10 | AIR-Bench | [Table 5] |
| AIR-Bench Age | 61.30 | Qwen3-Omni-30B: 67.00 | - | AIR-Bench | [Table 5] |

### 关键实验发现

**1. TSIT vs SIT vs SIFT (Stage 1 ablation)** [Table 3]:
- TSITs (传统 ASR instruction tuning): VoiceBench overall 22.26
- SITs (self-generated + instruction): 57.89
- **SIFTs (self-generated, instruction-free): 68.70**
- 差距巨大,验证了 instruction-free 的泛化优势 [论文原文]

**2. Two-stage vs one-stage** [Table 6]:
- 2-stage (TTA-Voice): VoiceBench 73.13, AIR-Bench avg 77.25
- 1-stage (TTA-Voice): VoiceBench 72.46, AIR-Bench avg 70.50
- 1-stage (Whisper): VoiceBench 69.86, AIR-Bench avg 68.99
- 两阶段略优,但差距不大; 即使单阶段 + 单编码器 (Whisper),SIFT 也有竞争力 [论文原文]

**3. SIFT 的增益主要来自训练范式而非架构** [Table 6]:
- 单编码器 Whisper + 一阶段 SIFT 的 VoiceBench (69.86) 已超过大多数端到端 Speech-LLM [论文原文]
- [agent 解读] 这是一个重要结论: SIFT 的增益是方法论层面的,不是因为双编码器或两阶段的架构优势。

**4. Self-Elicit 失败时的退化** [Table 7]:
- SIFTssp (with system prompt): VoiceBench 70.85, AIR-Bench age 53.40 (退化)
- SIFTssp + SITssp (mixed): VoiceBench 71.42, AIR-Bench age 67.00 (修复)
- [论文原文] System prompt 改变了 LLM 默认行为 (简洁对话而非详细描述),导致 SIFT target 信息量减少,encoder 被迫对齐到信息稀疏的 target

## 局限性

1. **仅 speech-aware,不生成语音**: AzeroS 只做理解 (speech→text),不做生成。作者指出可以通过接入 streaming TTS (如 CosyVoice) 升级为 general speech-text 系统 [§1],但未实验验证 [agent 解读]。

2. **Self-Elicit 条件的脆弱性**: SIFT 的最优性依赖于 LLM 在无 instruction 时能自发输出完整信息 (Goal 2)。一旦 LLM 默认行为被 system prompt 改变,SIFT 就退化,需要 SIT fallback [§5.7]。[agent 解读] 这限制了 SIFT 在需要特定交互风格 (如对话式、简洁式) 的实际部署场景中的纯净使用。

3. **数据规模有限**: 训练仅用 ~25k 小时公开数据 (Data-S 22k + Data-SP 3.5k),远小于 Qwen2-Audio 等系统。作者 claim 这证明了 SIFT 的数据效率,但也意味着在更大数据规模下的表现尚未验证 [论文原文]。

4. **Backbone scaling 未验证**: 仅在 Qwen2.5-7B 上实验。作者观察到 Qwen3-Omni-30B (更大 backbone) 在 semantic 任务上有显著提升,推测 SIFT 在更大 LLM 上会更好 [§5.5],但未做实验 [agent 解读]。

5. **评估覆盖有限**: 仅在 VoiceBench 和 AIR-Bench 上评估。缺少 ASR benchmark (LibriSpeech 等)、翻译 benchmark、以及更多 paralinguistic benchmark 的结果。[agent 解读] 尤其是,SIFT 声称能保留所有信息,但未用 ASR WER 等客观指标验证 semantic 信息是否真的完整保留。

6. **理论分析缺少形式化证明**: §3.5 的 "为什么 instruction-free 泛化更好" 主要是直觉论证和类比 ("Packed Suitcase" analogy),缺乏严格的数学证明 [agent 解读]。

## 点评

**正面评价**:

AzeroS 最有价值的贡献是 **SIFT 范式的概念化和统一框架**。将 AudioChatLlama、DeSTA-2/2.5 等先行工作统一到同一个 (Oracle Info, Instruction) 坐标系中 [Table 1],清晰展示了为什么 instruction-free + full information 是理论最优配置。这种 "把已有方法统一为同一框架的特例" 的分析方式,比单纯提出新架构更有长远价值。

实验设计也很到位: Table 3 的 TSIT→SIT→SIFT 三级对比和 Table 6 的架构 ablation 共同证明了增益来自 SIFT 范式而非架构复杂度。§5.7 的 Dialog-LLM case study 则诚实地展示了 SIFT 的失败条件,增强了论文的可信度。

**训练效率惊人**: 仅 47.6M 可训练参数 + 25k 小时数据就达到了 SOTA,远小于 Qwen-Audio、Qwen2.5-Omni 等系统的投入。这对资源有限的团队有很强的实用意义。

**批判性分析**:

1. "Universal decoder" 假设 [§3.1] 过于理想化。现实中 LLM 的泛化能力并非无限,不同 instruction 下的 reasoning pattern 差异不仅仅是 "feature filter" 导致的。[agent 解读] SIFT 的成功可能部分依赖于 Qwen2.5-7B-Instruct 恰好有很强的 open-ended 描述能力; 换一个倾向简洁的 LLM backbone 可能效果不同 (Dialog-LLM case study 间接印证了这一点)。

2. 与 SALMONN 的对比不够公平: SALMONN 用 Q-Former (更强的 adapter) + LoRA (微调 LLM),而 AzeroS 只用 linear projector + 完全冻结 LLM。Table 5 中 AzeroS 大幅领先 SALMONN 和 Qwen2-Audio,但这些系统是 2023-2024 年的工作,backbone 更旧更小。更公平的对比是用相同 backbone 和 adapter 对比 SIFT vs TSIT,但论文未提供。

3. VoiceBench 依赖 GPT-4o 评估,AIR-Bench Chat 也是 GPT-4 评估,这引入了评估器偏差。缺少客观指标 (WER, accuracy) 的补充验证。

## 可复用的 idea

1. **SIFT 范式可推广到其他模态**: 论文结尾提到计划将 SIFT 应用于 vision-language models [§6]。[agent 解读] 核心洞察 "去掉 instruction 强迫 encoder 做全信息对齐" 适用于任何 modality adapter + frozen LLM 的组合场景。

2. **Self-generated data 替代人工标注**: 用冻结 LLM 从 oracle text 生成训练 target,完全避免了 instruction data 的采集成本。这是一种通用的数据增强思路。

3. **两阶段渐进对齐**: 先对齐 semantic (信息量大、数据多),再对齐 paralinguistic (信息量小、数据少),避免信号冲突。[agent 解读] 类似于 curriculum learning 的思路,但用在 modality alignment 上。

4. **诊断 Self-Elicit 条件**: 通过观察 SIFT target 的信息密度,可以判断某个 LLM backbone 是否适合 SIFT。如果 LLM 在无 instruction 时输出过于简洁或偏向特定风格,就需要 SIT fallback。
