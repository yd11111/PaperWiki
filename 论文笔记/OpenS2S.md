---
type: paper
tier: deep
title: "OpenS2S: Advancing Fully Open-Source End-to-End Empathetic Large Speech Language Model"
arxiv_id: "2507.05177"
source: "Sources/OpenS2S.pdf"
authors: [Chen Wang, Tianyu Peng, Wen Yang, Yinan Bai, Guangfu Wang, Jun Lin, Lanpeng Jia, Lingxiang Wu, Jinqiao Wang, Chengqing Zong, Jiajun Zhang]
year: 2025
venue: "arXiv preprint (Technical Report)"
tags: [speech-LM, empathetic, end-to-end, streaming, interleaved-decoding, open-source, data-construction]
concepts: ["[[Speech Language Model]]", "[[Streaming Spoken Dialogue]]", "[[Emotion Control in TTS]]", "[[Speech Tokenizer]]", "[[Modality Adaptation for Speech LLM]]", "[[Conditional Flow Matching]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]", "[[模型库/Whisper|Whisper]]", "[[模型库/MinMo|MinMo]]", "Kimi-Audio", "GLM-4-Voice", "LLaMA-Omni2", "Qwen2-Audio"]
tasks: []
datasets: ["[[数据集/Emilia|Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[Speech Language Model]], [[Speech Tokenizer]], [[CosyVoice 2]]; 3 个待确认实体页: [[Streaming Spoken Dialogue]][待确认], [[Modality Adaptation for Speech LLM]][待确认], [[Emotion Control in TTS]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: OpenS2S 属于 modular SpeechLM 路线 — 用预训练语音编码器 + LLM + 流式语音解码器构建端到端对话系统。这条路线的代表系统包括 LLaMA-Omni 2、Freeze-Omni、MinMo。与 Moshi (全因果 RQ-Transformer) 路线不同,OpenS2S 沿用 LLaMA-Omni 2/MinMo 的 interleaved text-speech streaming decoding 架构。
>
> **已有认知**:
> - Speech Language Model: SpeechLM 的三大组件 (tokenizer + LM + vocoder),分类体系 (cold init → continued pretraining → instruction tuning → post-alignment),以及 modular vs end-to-end 两条主线已有系统性总结 [confirmed]
> - Streaming Spoken Dialogue: 已总结 Mini-Omni (delayed parallel)、LLaMA-Omni (NAR CTC)、Moshi (fully causal)、LLaMA-Omni 2 (AR interleaved R:W) 等多种流式方案 [待确认]
> - Speech Tokenizer: 三类 tokenizer (自监督/监督式/声学) 及演进脉络已清晰,GLM-4-Voice 的监督 semantic tokenizer 属于 Whisper-based 量化路线 [confirmed]
> - CosyVoice 2: FSQ-SenseVoice tokenizer + chunk-aware flow matching + HiFi-GAN 的完整技术栈已有详细记录;被 LLaMA-Omni 2 直接复用作为 streaming speech decoder [confirmed]
> - Emotion Control in TTS: 情感控制从 embedding → 层级建模 → DPO 优化 → LLM 自由文本情感的演进路线清晰,但 **empathetic dialogue** (理解对方情感 + 生成情感回应) 作为系统级能力在知识库中尚无专门页面 [待确认]
>
> **创新判断基准**: OpenS2S 的核心差异化在于 (1) 将 empathy 作为第一优先级设计目标(区别于侧重指令遵循的 LLaMA-Omni 2),(2) 提出自动化共情对话数据构建 pipeline(区别于手动标注或简单 TTS 合成的已有数据集),(3) 完全开源(数据 + 代码 + 权重)。架构本身与 LLaMA-Omni 2/MinMo 高度相似。
>
> 检索命中: [[Speech Language Model]]✓, [[Speech Tokenizer]]✓, [[CosyVoice 2]]✓ | 过滤: [[Streaming Spoken Dialogue]](pending-review), [[Modality Adaptation for Speech LLM]](pending-review), [[Emotion Control in TTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 完全开源的端到端共情语音对话模型,通过自动化数据构建 pipeline 以低成本实现副语言情感理解与表达
> - **路线**: 语音输入 → Qwen2-Audio encoder (25Hz → 6.25Hz via CNN adapter) → Qwen3-8B LLM (interleaved text+speech hidden states) → Qwen3-1.8B streaming speech decoder (GLM-4-Voice tokenizer, 12.5 tok/s) → chunk-aware causal flow matching + HiFi-GAN → 语音输出
> - **指标**: VoiceBench alpaca 4.51 / URO-Bench UnderEmo-en 59.32 (V1.5, 接近 Kimi-Audio 的 59.22/76.96) [Table 2]; 预训练用 Emilia ~12k 小时 TTS 数据 [§3.1] + ~3.8M ASR 对 + 70k SER 对, 远少于 Kimi-Audio 的 1300 万小时 [§1]
> - **可借鉴**: 自动化共情数据构建三步法 (seed audio → LLM self-instruct 生成带副语言标注的 query → CosyVoice2 情感可控合成 response); 用 thinking mode 推断回复应带的情感色调
> - **局限**: 语音到语音评估仅有定性分析无定量指标; 架构与 LLaMA-Omni 2 高度同构,创新主要在数据侧; V1 版本多项指标落后 Kimi-Audio 较多, V1.5 通过扩数据追平

## 核心问题

1. **如何在不使用海量预训练数据的前提下,让 SpeechLM 具备共情对话能力?** Kimi-Audio 等模型用 1300 万小时音频预训练获得强共情能力,OpenS2S 试图用低成本替代方案达到可比效果。
2. **如何系统性地构建高质量共情语音对话数据?** 现有开源数据集要么缺乏说话人多样性,要么忽略副语言信息,要么标注维度单一。
3. **如何在开源框架下实现完整的端到端语音对话?** 大多数强共情 LSLM 是闭源的(GPT-4o、Gemini、Kimi-Audio 的训练数据均未公开)。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

OpenS2S 是一个四组件 modular SpeechLM [§2.1, Fig 1]:

1. **Audio Encoder**: Qwen2-Audio encoder 提取 25Hz 帧级特征 → 2 层 CNN downsampling (4x) → FFN speech adapter → 6.25Hz 连续表征输入 LLM [§2.1]
2. **Instruction-Following LLM**: Qwen3-8B-Instruct 作为核心语言模型,接收 audio embeddings + text embeddings 的拼接序列 [§2.1]
3. **Streaming Speech Decoder**: 基于 Qwen3-1.8B 初始化的 decoder-only Transformer,词表扩展 16384 个 speech tokens。输入为 LLM 最终层 hidden states (经 linear projection),按 M:N = 4:8 的比例与生成的 speech tokens 交替排列 [§2.1]
4. **Token2Wav Decoder**: chunk-aware causal flow matching + HiFi-GAN vocoder,均来自 GLM-4-Voice 预训练组件 [§2.1]

**Speech Tokenizer**: 采用 GLM-4-Voice 的监督 semantic tokenizer — 在 Whisper-large-v3 encoder 中插入量化模块,产生 12.5 tok/s、词表 16384 的 token 序列 [§2.1]。[agent 解读] 这与 CosyVoice 系列使用的 FSQ-SenseVoice tokenizer 不同,GLM-4-Voice 的 tokenizer 基于 Whisper 而非 SenseVoice。

### 关键设计选择

**为什么选择 interleaved streaming decoding?** [论文原文] OpenS2S 采用受 MinMo 和 LLaMA-Omni2 启发的框架 [§2.1],每消耗 M=4 个 LLM hidden states 就生成 N=8 个 speech tokens。所有 hidden states 消耗完后,继续自回归生成剩余 speech tokens。[agent 解读] 这种 interleaved 方案的优势在于: (1) LLM 不需要生成完整 text 后再开始 speech 生成(避免了 chain-of-modality 的高延迟),(2) 1:2 的 R:W 比例意味着 speech decoder 在 LLM 推理过程中就能开始输出,降低首字节延迟。

**为什么用 Qwen2-Audio 做 encoder 而非 Whisper?** [论文原文] 因其"编码语义内容和副语言信息的强大能力" [§2.1]。[agent 解读] Qwen2-Audio 是在 Whisper 基础上通过多任务指令训练增强的,比原始 Whisper 更擅长捕捉情感、说话人等副语言特征,这对共情对话至关重要。

**为什么 Token2Wav 复用 GLM-4-Voice 的组件?** [论文原文] 论文未解释选择原因,直接说"adopted from the pretrained components in GLM-4-Voice" [§2.1]。[agent 解读] 由于 speech tokenizer 也来自 GLM-4-Voice,复用配套的 flow matching + vocoder 可以确保 token→wav 的重建质量,避免 tokenizer 与 decoder 不匹配的问题。

### 训练策略

三阶段训练 [§2.2, Fig 2]:

**Stage 1: Speech Understanding Pretraining** — 分两个子阶段:
- **1.1 Semantic Alignment**: 基于 BLSP-Emo 的 behavioral alignment 方法 — 先让 LLM 根据 ASR 文本生成 continuation,训练时要求模型从语音输入生成相同的 continuation [§2.2]。冻结 audio encoder 和 LLM,只训练 speech adapter。
- **1.2 Emotional Alignment**: 使用 SER 数据集(含情感标签),LLM 先根据 transcript + 情感标签生成 emotion-aware continuation,再训练模型从语音直接生成相同输出 [§2.2]。同样只训练 adapter。

[agent 解读] 这种"对齐续写"范式比直接在语音-文本对上训练更巧妙: 它利用 LLM 本身的文本能力来生成训练 target,从而将"语音→语义理解"和"语义→情感理解"解耦为两个子任务。Stage 1.2 的创新在于引入情感标签作为额外条件,使 adapter 学会从语音中提取副语言信息。

**Stage 2: Speech Generation Pretraining** — 分两个子阶段:
- **2.1 Offline TTS**: 扩展 Qwen3-1.8B 词表加入 16384 speech tokens,用文本→speech token 的 TTS 数据训练,学习基本的 text→speech token 映射 [§2.2]
- **2.2 Streaming Interleaved**: 文本不再直接输入 speech decoder,改为先通过 LLM 处理,提取最终层 hidden states 后与 speech tokens 交替训练 [§2.2]。冻结 LLM,训练 linear projection + speech decoder。

[agent 解读] Stage 2.1 → 2.2 的过渡是关键: 先让 decoder 学会"朗读"(text→speech),再让它学会"从 LLM 的语义表征生成语音"(hidden states→speech)。这种渐进式训练避免了直接从 hidden states 训练时的冷启动困难。

**Stage 3: Empathetic Speech Instruction Tuning** — 使用构建的共情数据集进行全模型微调(冻结 speech encoder,训练 adapter + LLM + projection + speech decoder)[§2.2]。

[论文原文] 作者发现仅用 speech-to-speech 指令数据会导致 speech decoder 在处理文本指令时失败,将此归因于"TTS 预训练阶段的过拟合,模型学会依赖于 TTS 任务定义的狭窄表征子空间" [§2.2]。解决方案: 额外加入 text-to-speech 指令数据,使模型能同时处理语音和文本输入。

### 数据构建 Pipeline

OpenS2S 的核心贡献之一是自动化共情数据构建流程 [§3.2, Fig 3]:

**Step 1: Seed Audio Collection** — 从公开 SER 数据集中选取 1000 英文 + 1000 中文种子音频,手动标注 transcript、gender、age、emotion [§3.2]

**Step 2: Speech Instruction Generation** — 使用 Qwen3-32B-Instruct 的 self-instruct 生成副语言敏感的任务指令(如"你觉得我能跑马拉松吗?",标注为 age-sensitive),然后匹配对应标签的种子音频,用 CosyVoice2 做 voice cloning 生成语音版指令 [§3.2]

**Step 3: Speech Response Generation** — 对每条指令,将语义内容 + 副语言标签输入 Qwen3-32B-Instruct (thinking mode 开启),生成简洁共情文本回复,并推断回复应带的情感色调。最后用 CosyVoice2 通过 instruction prompts 合成情感表达丰富的语音回复 [§3.2]

最终产出: V1 版本 50k EN + 50k ZH 共情样本 + 100k 通用双语样本; V1.5 版本扩展至 200k EN + 200k ZH 共情 + 400k 通用 [§3.2]

## 实验

| 指标 | OpenS2S | OpenS2S_V1.5 | Kimi-Audio | GLM-4-Voice | LLaMA-Omni2 | Qwen2-Audio | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VoiceBench-alpaca | 4.09 | 4.51 | 4.46 | 3.97 | 3.96 | 3.74 | VoiceBench | [Table 2] |
| VoiceBench-common | 3.65 | 3.88 | 3.97 | 3.46 | 3.42 | 3.43 | VoiceBench | [Table 2] |
| VoiceBench-ifeval | 42.89 | 41.75 | 61.10 | 25.92 | 17.36 | 26.33 | VoiceBench | [Table 2] |
| VoiceBench-wildvoice | 3.66 | 3.78 | 4.20 | 3.18 | 3.07 | 3.01 | VoiceBench | [Table 2] |
| URO-Bench UnderEmo-en | 46.90 | 59.32 | 59.22 | 52.41 | 39.46 | 35.38 | URO-Bench | [Table 2] |
| URO-Bench UnderEmo-zh | 67.68 | 72.83 | 76.96 | 74.51 | 63.79 | 69.62 | URO-Bench | [Table 2] |

**关键发现**:
1. V1 版本整体排名第三(仅次于 Kimi-Audio 和 GLM-4-Voice),考虑到其训练数据量远少于 Kimi-Audio (1300 万小时 vs ~12k 小时预训练),数据效率显著 [Table 2]
2. V1.5 版本仅扩展 Stage 3 的数据规模即实现大幅提升: VoiceBench-alpaca 和 URO-Bench UnderEmo-en 两项超越所有 baseline [Table 2]
3. ifeval 子集上的差距较大 (41.75 vs Kimi-Audio 61.10),表明 OpenS2S 在复杂指令遵循方面仍有不足 [Table 2]
4. Speech-to-speech 评估仅有定性 demo,无定量指标 [§4.2]

## 局限性

1. **S2S 评估缺失**: 作为一个 speech-to-speech 系统,论文仅评估了 speech-to-text 性能,speech 输出质量(MOS、speaker similarity、emotion accuracy 等)完全缺失 [§4.2]
2. **架构同质化**: 整体架构(Qwen2-Audio encoder + LLM + interleaved streaming decoder + GLM-4-Voice tokenizer/vocoder)与 LLaMA-Omni 2 高度相似,技术创新主要集中在数据侧 [§2.1]
3. **合成数据偏差**: 回复端固定为年轻女性声音 [§3.2],speaker diversity 仅体现在输入端,这可能导致模型在生成多样化声音时的能力受限
4. **ifeval 指标显著落后**: 复杂指令遵循能力 (ifeval 41.75 vs Kimi-Audio 61.10) 差距较大,可能限制实际部署场景 [Table 2]
5. **无延迟/实时性评测**: 虽然采用了 streaming 架构,但论文未报告首包延迟、RTF 等时效性指标

## 点评

OpenS2S 的核心价值在于**完全开源**和**低成本共情数据构建方法**,而非架构创新。在已有 LLaMA-Omni 2 提供了成熟的 modular streaming SpeechLM 架构的前提下,OpenS2S 的贡献可以理解为: 验证了"在该架构上,通过精心设计的共情数据 pipeline,可以用显著更少的资源获得接近 SOTA 的共情交互能力"。

数据构建方法是本文最值得关注的部分: seed audio → LLM self-instruct (副语言敏感) → CosyVoice2 情感可控合成的三步流程,系统性地解决了现有数据集的三大问题(speaker diversity、paralinguistic information、label granularity)。特别是用 LLM thinking mode 推断回复应带的情感色调,这一步将共情从"理解情感"延伸到"情感推理+表达",是合理的设计。

但也需注意: V1→V1.5 的大幅提升主要来自数据规模扩展(100k→800k 样本),这既验证了数据方法的 scalability,也暗示 V1 的训练数据可能不够充分。S2S 评估的完全缺失是一个明显的不足 — 对于一个声称实现"empathetic speech interaction"的系统,不评估语音输出的情感表达质量是不完整的。

## 可复用的 idea

1. **Behavioral alignment 训练策略** (来自 BLSP-Emo): 用 LLM 根据文本生成 continuation 作为训练 target,再让模型从语音输入复现相同输出 — 这种"续写对齐"方法可以零成本地将 LLM 的文本能力迁移到语音模态理解中,可推广到其他跨模态适配场景
2. **副语言敏感的 self-instruct**: 在 LLM self-instruct 生成指令时,显式标注哪些指令对 age/emotion/gender 敏感,并据此匹配种子音频特征 — 这种有意识的副语言多样性设计比随机 TTS 合成更有针对性
3. **用 LLM thinking mode 推断回复情感**: 不直接规定回复情感,而是让 LLM 根据上下文 + 输入副语言特征推断应表达的情感 — 这比硬编码情感映射规则更灵活,可迁移到任何需要情感推理的对话系统
4. **TTS 过拟合的诊断和修复**: Stage 2 TTS 预训练后 speech decoder 只能处理 TTS 任务的窄表征空间,解法是混入 text-to-speech 指令数据 — 这个"过拟合→加任务多样性"的模式对所有多阶段训练系统有参考价值

> [!review] 审阅 (2026-06-03, auto)
> **结论: pass-with-fixes** (0 high / 2 medium / 1 low)
> - [medium] frontmatter.models: 已补充 baseline 模型 (Kimi-Audio, GLM-4-Voice, LLaMA-Omni2, Qwen2-Audio)
> - [medium] 速查卡片指标: 已明确 ~12k 小时来源为 §3.1 Emilia 数据
> - [low] 可复用 idea #1: 已注明 behavioral alignment 来自 BLSP-Emo
> 详见 `_review/OpenS2S-review.yml`。
