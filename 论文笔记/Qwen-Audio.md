---
type: paper
tier: deep
title: "Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models"
arxiv_id: "2311.07919"
source: "Sources/Qwen-Audio.pdf"
authors: [Yunfei Chu, Jin Xu, Xiaohuan Zhou, Qian Yang, Shiliang Zhang, Zhijie Yan, Chang Zhou, Jingren Zhou]
year: 2023
venue: "arXiv"
tags: [speech-LM, audio-understanding, multimodal, multi-task-pretraining, hierarchical-tags, SRWT, universal-audio, instruction-tuning, LALM]
concepts: ["[[AudioUnderstanding]]", "[[Audio-LanguagePretraining]]", "[[ModalityAdaptationforSpeechLLM]]", "[[SpeechLanguageModel]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Qwen-Audio 属于 latent-representation-based Speech-LLM integration 路线 [Speech-LLMIntegrationTaxonomy],使用 Whisper-large-v2 作为单一音频编码器,其连续特征通过下采样后直接送入 Qwen-7B LLM。在 KB 的 SpeechLM 分类中,Qwen-Audio 使用连续编码器特征 (continuous features),属于 "continuous features" 类别 [SpeechLanguageModel]。与同期的 SALMONN 相比,SALMONN 使用双编码器 (Whisper+BEATs) + Q-Former,而 Qwen-Audio 选择单一编码器 + 层级标签框架。在 ALM 架构分类中属于 "Two Heads" 架构 [Audio-LanguagePretraining]。
>
> **已有认知**: KB 中 AudioUnderstanding 页面已记录 SpeechLM 理解能力的三大分类 (语义/说话人/副语言),Qwen-Audio 的任务覆盖远超这三类,扩展到音乐和环境声。ModalityAdaptationforSpeechLLM 页面记录了三种适配方法 (Conv downsampling, CTC compression, Q-Former),Qwen-Audio 使用最简单的池化层 (stride=2 pooling) 下采样,而非 Q-Former 等复杂方案。Self-SupervisedSpeechRepresentation 页面记录了 Whisper encoder 作为弱监督替代自监督的定位 — Qwen-Audio 直接复用 Whisper encoder 而非 HuBERT/WavLM 等 SSL 模型。
>
> **创新判断**: 相对于已有 KB 知识,Qwen-Audio 的独特性在于: (1) 用层级标签 (hierarchical tags) 解决多任务/多数据集混训的 one-to-many 干扰问题,而非 Q-Former 或 dataset ID 等已有方案; (2) 提出 SRWT (word-level timestamp) 训练任务,发现其对非语音任务 (AQA/音乐 QA) 也有增益; (3) 任务覆盖 30+ 种,远超当时的同类工作。KB 中尚无专门讨论 hierarchical tag / multi-task interference 的概念页。
>
> 检索命中: [[SpeechLanguageModel]]✓, [[AudioUnderstanding]][待确认], [[Audio-LanguagePretraining]][待确认], [[ModalityAdaptationforSpeechLLM]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个覆盖 30+ 任务、多种音频类型 (语音/环境声/音乐/歌曲) 的统一音频-语言模型,通过层级标签框架解决多任务混训干扰,无需任务特定微调即超越同类模型
> - **路线**: Audio (any type) → 16kHz resample → 80-ch mel spectrogram → Whisper-large-v2 encoder (640M, 可训练) → stride-2 pooling (~40ms/frame) → Qwen-7B LLM (7.7B, Stage 1 冻结 / Stage 2 可训练) → text response
> - **指标**: Aishell1 test WER 1.3% (SOTA); CochlScene ACC 79.5% (SOTA); ClothoAQA ACC 57.9% (SOTA); VocalSound ACC 92.9% (SOTA); LibriSpeech test-clean WER 2.0%; CoVoST2 七方向平均 BLEU 显著优于 SpeechLLaMA/SALMONN [Table 3]
> - **可借鉴**: (1) 层级标签框架 (transcription/audio-lang/task/text-lang/timestamps/output-instruction) 作为多数据集混训的轻量级干扰缓解方案; (2) SRWT (word-level timestamp) 训练同时提升 ASR 和非语音 QA 性能的发现; (3) 两阶段训练 (Stage 1 冻结 LLM 训编码器, Stage 2 冻结编码器训 LLM) 的模块化对齐策略
> - **局限**: 论文未开源训练数据配比和混训细节; Chat 模型仅 20k instruction 数据,与后续工作量级差距大; 无生成能力 (仅理解); 无消融实验验证各标签层级的独立贡献

## 核心问题

**想解决什么**: 2023 年底,现有音频-语言多任务模型 (SpeechT5, Whisper, Pengi) 只能处理特定类型的音频 (人声或环境声) 或特定任务 (ASR 或 captioning)。构建 instruction-following 的音频交互模型需要一个能覆盖所有音频类型和任务的预训练基座,但这样的基座并不存在 [§1]。

**为什么难**: 直接混合 30+ 任务和数十个数据集会产生严重的 **one-to-many 干扰**: 同一段音频可能对应完全不同格式的文本标签 — ASR 需要逐字转写,captioning 需要描述性文本,分类需要类别标签。标签之间在任务目标、语言、标注粒度和文本结构上差异巨大,模型无法区分该产生什么类型的输出 [§1, §3.2]。

**怎么切入**: 设计一套层级标签 (hierarchical tags) 作为解码器的条件信号: 通过共享标签促进知识迁移,通过特定标签消除干扰。同时引入 SRWT 任务增强细粒度对齐能力 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Qwen-Audio 由两个组件构成 [§3.1, Fig 3]:

1. **Audio Encoder**: 基于 Whisper-large-v2 初始化,32 层 Transformer + 2 层卷积下采样 stem,共 640M 参数。将 16kHz 音频转为 80 通道 mel spectrogram (window 25ms, hop 10ms),编码器后加 stride-2 pooling 层将帧率减半,每帧约对应 40ms 原始音频 [§3.1]。训练时应用 SpecAugment 数据增强 [§3.1]。

2. **Large Language Model**: 基于 Qwen-7B 初始化,32 层 Transformer decoder,hidden size 4096,共 7.7B 参数。训练目标为标准 next token prediction:
   ```
   P_θ(x_t | x_{<t}, Encoder_φ(a))
   ```
   其中 θ 和 φ 分别为 LLM 和 encoder 的可训练参数 [Eq. 1]。

**架构简洁性**: [agent 解读] 与 SALMONN 的双编码器+Q-Former 设计相比,Qwen-Audio 的架构极其简洁 — 没有额外的 adapter/projector 模块,仅用 pooling 层做下采样。核心创新不在架构上,而在训练框架和数据组织上。

### 关键设计选择

**为什么选单一编码器而非多编码器?** [论文原文] 论文未显式讨论此选择。[agent 解读] Whisper-large-v2 虽然在 ASR/翻译上训练,但 Gong et al. (2023a) 已证明其编码器的表征包含丰富的背景噪声和非语音信息。单一编码器在推理效率上优于双编码器 (SALMONN),且训练更简单 — 无需对齐两个编码器的帧率和特征维度。代价是没有专用非语音编码器的互补优势。

**为什么用层级标签而非 dataset ID?** [论文原文] 之前的方法要么按任务分组,要么给每个数据集分配 ID。这两种方法的问题是: 分组无法区分组内不同格式的输出; dataset ID 导致每个数据集完全独立,无法利用任务间的知识共享 [§3.2]。层级标签的设计原则是在 "最大化知识共享" 和 "消除 one-to-many 歧义" 之间取得平衡: 共享标签 (如 transcription tag) 让相关任务共享底层能力,而特定标签 (如 output instruction) 区分输出格式 [§3.2]。

**为什么引入 SRWT?** [论文原文] SRWT 的目的有两个: (1) 增强模型对音频信号和细粒度时间戳的对齐能力; (2) 支持 Qwen-Audio-Chat 中的 grounding 和 grounding-based QA 任务,如定位音频中某人名的起止时间 [§4.5]。[论文原文] 时间戳预测与转写交替进行: 每个词前预测开始时间,词后预测结束时间 [§3.2, timestamps tag]。[agent 解读] SRWT 之所以能提升非语音 QA 任务 (ClothoAQA, MusicAVQA),可能是因为细粒度时间对齐迫使编码器学习更精确的帧级表征,这种精确表征对需要定位音频事件的 QA 任务也有益。

### 多任务训练格式框架 (Multi-task Training Format Framework)

层级标签框架将条件信号组织为六层,按序拼接在解码器输入 [§3.2, Fig 3]:

| 层级 | 标签 | 作用 | 示例 |
|------|------|------|------|
| 1. Transcription Tag | `<\|startoftranscripts\|>` 或 `<\|startofanalysis\|>` | 区分语音转写类任务和分析类任务 | ASR/S2TT → transcripts; AAC/ER → analysis |
| 2. Audio Language Tag | 8 种语言 + `<\|unknown\|>` | 指示音频中的口语语言; 非语音用 unknown | `<\|zh\|>`, `<\|en\|>`, `<\|unknown\|>` |
| 3. Task Tag | 5 类: transcribe, translate, caption, analysis, question-answer | 指定任务类型; QA 任务后追加问题文本 | `<\|transcribe\|>`, `<\|question-answer\|>` |
| 4. Text Language Tag | 对应输出文本的语言 | 控制输出语言 | `<\|en\|>` |
| 5. Timestamps Tag | `<\|timestamps\|>` 或 `<\|notimestamps\|>` | 是否预测 word-level 时间戳 (SRWT) | — |
| 6. Output Instruction | 自然语言指令 | 进一步指定子任务和输出格式 | "WORD LEVEL TRANSCRIPTION" |

[agent 解读] 这套框架本质上是对 Whisper 多任务格式的大幅扩展: Whisper 仅支持 ASR+翻译+VAD+语言识别+句级时间戳,且只处理语音; Qwen-Audio 将此扩展到所有音频类型和 30+ 任务。层级设计的核心思想类似数据库的层次索引 — 上层标签做粗分类以共享知识 (所有 transcription 类任务共享 transcripts 标签),下层标签做细分以消歧。

### 训练策略

两阶段训练 [§4.1, Table 6]:

**Stage 1 — Multi-task Pre-training**:
- **冻结 LLM,仅训练 audio encoder** [§4.1]
- 500k 步,batch size 120,峰值学习率 5e-5,cosine decay
- 编码器学习率衰减因子 0.95
- 覆盖 30+ 任务,训练数据总量约 12 万小时 (估算来自 Table 1: ASR 30k + S2TT 3.7k + AAC 8.4k + 音乐 caption 25k + 其他)

[论文原文] 冻结 LLM 的原因: 先让编码器学会将音频表征映射到 LLM 可理解的空间,避免两个模块同时训练的不稳定性 [§4.1]。[agent 解读] 这与 SALMONN 的策略一致 — 分阶段训练降低优化难度。但 Qwen-Audio 选择在 Stage 1 训练编码器 (SALMONN 冻结编码器训 Q-Former),原因可能是 Qwen-Audio 没有独立的 adapter 模块,编码器本身就承担了表征对齐的角色。

**Stage 2 — Supervised Fine-tuning (Qwen-Audio-Chat)**:
- **冻结 audio encoder,仅训练 LLM** [§4.1]
- 8k 步,batch size 128,峰值学习率 1e-5
- 使用 ChatML 格式 (`<im_start>` / `<im_end>`)
- 约 20k instruction tuning 数据,由人工标注 + GPT-3.5 生成 [§3.3]
- 支持多音频输入,通过 "Audio id:" 标注不同音频 [§3.3]

### 数据覆盖

[Table 1] 训练数据覆盖三大类音频和 30+ 任务:

| 类型 | 主要任务 | 数据量 |
|------|----------|--------|
| 语音 (Speech) | ASR (30k h), S2TT (3.7k h), SRWT (21k h), LID (11.7k h), SV (1.2k h) 等 18 类 | ~85k h |
| 环境声 (Sound) | AAC (8.4k h), SEC (5.4k h), ASC, SED, AQA | ~15k h |
| 音乐/歌曲 | MC (25k h), MGR (9.5k h), SID, SMER, MNA 等 8 类 | ~37k h |

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER ↓ | 2.0 / 4.2 | SpeechT5: 2.4 / 5.8; SLM-FT: 2.1 / 4.9 | LibriSpeech test-clean / test-other | [Table 3] |
| WER ↓ | 1.2 / 1.3 | Paraformer-large: 2.0; MMSpeech-large: 1.9 | Aishell1 dev / test (SOTA) | [Table 3] |
| WER ↓ | 3.3 / 3.1 / 3.3 | Paraformer-large: 2.9 / 3.3; MMSpeech-base: 4.5 / 3.9 / 4.0 | Aishell2 Mic / iOS / Android | [Table 3] |
| BLEU ↑ | 25.1 / 33.9 / 41.5 / 15.7 | SpeechLLaMA: 14.1 / - / - / 12.3 | CoVoST2 (de-en, en-zh, zh-en, en-de) | [Table 3] |
| BLEU ↑ | 39.7 / 38.5 / 36.0 | SpeechLLaMA: 27.9 / 25.2 / 25.9 | CoVoST2 (es-en, fr-en, it-en) | [Table 3] |
| SPIDEr ↑ | 0.288 | Pengi: 0.271 | Clotho (AAC) | [Table 3] |
| AAS (ms) ↓ | 51.5 | Force-aligner: 60.3; Paraformer-TP: 65.3 | Industrial Data (SRWT) | [Table 3] |
| ACC ↑ | 0.795 | CochlScene baseline: 0.669 | CochlScene (ASC, SOTA) | [Table 3] |
| ACC ↑ | 0.649 | Pengi: 0.353 | TUT2017 (ASC) | [Table 3] |
| ACC ↑ | 0.557 | WavLM-large: 0.542 | Meld (SER) | [Table 3] |
| ACC ↑ | 0.579 / 0.749 | Pengi: -/0.645; ClothoAQA: 0.542/0.627 | ClothoAQA (AQA, SOTA) | [Table 3] |
| ACC ↑ | 0.9289 | Pengi: 0.6035; CLAP: 0.4945 | VocalSound (VSC, SOTA) | [Table 3] |
| ACC ↑ | 0.7882 | Pengi: 0.5007 | NSynth Instrument (MNA) | [Table 3] |

### SRWT 消融实验

[Table 4, Table 5] 移除 SRWT 训练任务 (其他任务不变,数据集不变) 后:

| 任务 | w/ SRWT | w/o SRWT | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER ↓ | 1.79 / 4.00 / 2.04 / 4.19 | 1.93 / 4.18 / 2.22 / 4.21 | LibriSpeech dev-clean/other, test-clean/other | [Table 4] |
| WER ↓ | 1.22 / 1.29 | 1.54 / 1.71 | Aishell1 dev / test | [Table 4] |
| AQA ACC ↑ | 0.5795 / 0.7491 | 0.5648 / 0.7418 | ClothoAQA test / test-binary | [Table 5] |
| Music QA ↑ | 0.7211 | 0.7027 | MusicAVQA audio question | [Table 5] |

**关键观察**:
1. SRWT 对 ASR 的提升在中文 (Aishell1) 上更显著 (WER 1.71→1.29, 相对降低 24.6%) 而在英文上较小 [Table 4]。[agent 解读] 这可能是因为中文 SRWT 数据量 (11k h) 大于英文 (10k h),且中文 ASR 从更准确的帧对齐中获益更多 (中文分词边界更模糊)。
2. SRWT 对非语音 QA (ClothoAQA, MusicAVQA) 也有提升 [Table 5]。[论文原文] 这说明细粒度时间戳预测增强了模型对音频信号的通用 grounding 能力,这种能力跨越了语音/非语音的边界 [§4.5]。
3. SRWT 的时间戳精度 (AAS 51.5ms) 优于 Force-aligner (60.3ms) 和 Paraformer-TP (65.3ms) [Table 3]。[agent 解读] 值得注意的是 Force-aligner 使用 ground-truth 转写作为输入,而 Qwen-Audio 是同时生成转写和时间戳,在更难的条件下取得更好精度,说明端到端建模的优势。

## 局限性

1. **无消融实验验证层级标签的各层独立贡献**: 论文证明了完整框架的有效性,但未测试去掉某一层标签 (如 audio language tag) 的影响 [agent 解读]。
2. **仅理解不生成**: Qwen-Audio 只输出文本,不输出语音/音频 [agent 解读]。
3. **Chat 模型数据量极小**: 仅 20k instruction 数据 [§3.3],且依赖 GPT-3.5 生成,与后续工作 (如 Qwen2-Audio 的 200k+) 相比量级差距大。
4. **Stage 1 冻结 LLM 的代价**: 编码器需要单独学会映射到 LLM 空间,但 LLM 参数未被调整以适应音频输入。这可能限制了复杂推理任务的表现 [agent 解读]。
5. **评估局限**: 论文未测试多音频输入场景 (只在 demo 中展示); Chat 模型未在标准 benchmark 上系统评估 [agent 解读]。
6. **训练成本不透明**: 论文未报告总 GPU 小时和训练成本 [agent 解读]。

## 点评

**数据工程 > 架构创新**: Qwen-Audio 的核心贡献不在架构 (encoder + LLM 是标准范式) 而在数据和训练组织上。30+ 任务的混训、层级标签框架、SRWT 任务的引入 — 这些都是系统级工程的产出。在后来的 Qwen2-Audio 中架构基本未变,进一步证明了数据和训练策略的重要性。

**层级标签 vs Whisper 的关系**: Qwen-Audio 的多任务格式直接继承自 Whisper (Radford et al., 2023) 的 special token 方案,但做了实质性扩展 — 从 2 种任务扩展到 30+,从语音扩展到全音频类型。论文对此归功说明较少,但这种"站在 Whisper 肩上"的路线非常有效。

**SRWT 的意外发现**: 最有洞察力的实验发现是 SRWT 对非语音 QA 的跨域增益 [Table 5]。这暗示了一个更普遍的原则: 细粒度时序监督信号可以作为"免费的"通用表征增强,即使目标任务不需要时间戳。

**与 SALMONN 的互补对比**: 两篇同期工作 (Qwen-Audio: 2023.11, SALMONN: 2023.10) 代表了 Audio-LLM 的两种设计哲学 — SALMONN 在架构上创新 (dual encoder + Q-Former + activation tuning),Qwen-Audio 在训练框架和数据覆盖上创新 (hierarchical tags + 30+ tasks)。最终 Qwen-Audio 在更多 benchmark 上取得更好性能,部分归因于其更大的训练数据规模。

**历史定位**: Qwen-Audio 发表于 2023 年 11 月,处于 Audio-LLM 范式确立期。它与 SALMONN, LTU 共同证明了 "冻结/微调编码器 + LLM" 范式的有效性,并通过 Qwen-Audio-Chat 展示了从基座模型到交互模型的标准路径 (pre-training → SFT)。Qwen 团队后续的 Qwen2-Audio 和 Qwen3.5-Omni 在此基础上不断扩展。

## 可复用的 idea

1. **层级标签框架用于多数据集混训**: 当需要在标签格式差异大的多个数据集上联合训练时,可设计多层条件标签 — 上层共享标签促进知识迁移,下层特定标签消歧。这比 dataset ID (完全隔离) 或简单混合 (完全不区分) 都更优。适用于任何多任务/多数据集的统一模型训练。

2. **SRWT 作为辅助任务增强表征**: 在主任务之外加入细粒度时序预测 (word-level timestamp) 可以"免费"增强编码器的帧级表征质量,且对非语音任务也有跨域增益。可考虑在其他连续信号建模任务中引入类似的时序辅助任务。

3. **两阶段模块化训练**: Stage 1 冻结 LLM 训编码器 → Stage 2 冻结编码器训 LLM。这种"先对齐表征空间,再释放 LLM 能力"的策略适用于任何 encoder+LLM 架构,避免了同时训练两个大模块的不稳定性。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节因果解释充分,层级标签设计原则、SRWT 引入动机、两阶段训练逻辑均回答 WHY |
> | 可信赖 | pass | 数字标注覆盖率>90%,指标方向正确,实验表格含完整 baseline+数据集+出处 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率>90%,推断均标记为 agent 解读 |
> | 可定位 | pass | KB 背景谱系清晰,与 SALMONN 对比定位明确,历史节点准确 (2023.11 Audio-LLM 范式确立期) |
> | 不污染 | pass | 反向更新为 append-only (key_papers 追加),无新概念页创建需求 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/Qwen-Audio-review.yml`
