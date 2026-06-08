---
type: paper
tier: deep
title: "VoxInstruct: Expressive Human Instruction-to-Speech Generation with Unified Multilingual Codec Language Modelling"
arxiv_id: "2408.15676"
source: "Sources/VoxInstruct.pdf"
authors: [Yixuan Zhou, Xiaoyu Qin, Zeyu Jin, Shuoyi Zhou, Shun Lei, Songtao Zhou, Zhiyong Wu, Jia Jia]
year: 2024
venue: "ACM MM 2024"
tags: [instruction-to-speech, codec-LM, multilingual, CFG, semantic-token, LLaMA, expressive-TTS, voice-cloning, fine-grained-control]
concepts: ["[[Instruction-GuidedSpeechSynthesis]]", "[[NaturalLanguageDescriptionforTTS]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[CodecLanguageModel]]", "[[Classifier-FreeGuidance]]"]
models: ["[[模型库/EnCodec|EnCodec]]"]
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[LLM-basedTTS]], [[SpeechTokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓ | 过滤: [[Instruction-GuidedSpeechSynthesis]](pending-review), [[NaturalLanguageDescriptionforTTS]](pending-review), [[CodecLanguageModel]](pending-review), [[Classifier-FreeGuidance]](pending-review) | 未命中但可能相关: 无
>
> **LLM-based TTS**: VoxInstruct 属于 LLM-based TTS 中的 codec LM 路线,采用 VALL-E 开创的 AR+NAR 两阶段范式,但将输入从 phoneme sequence 替换为 unified natural language instruction。KB 记录了该范式的典型架构: text→token encoder + decoder-only transformer + speech decoder。VoxInstruct 在此基础上增加了 instruction understanding 层 (MT5 encoder) 和 semantic token 中间表征。在 KB 的演进线中,VoxInstruct 被定位为从 "VALL-E codec LM" 到 "Instruction-aware 架构" 的过渡节点,与 CosyVoice (LLM + Flow hybrid) 同期但走了不同技术路线。
>
> **Speech Tokenizer**: VoxInstruct 同时使用两类 speech tokenizer: (1) HuBERT + k-means 的自监督 semantic tokenizer,用于提取去除韵律/时长信息的高层内容表征,作为 instruction-to-content 的中间桥梁; (2) EnCodec 的 RVQ acoustic tokenizer (8 层),编码完整声学信息。KB 记录了这两类 tokenizer 的本质区别: semantic tokens 编码语言/内容,acoustic tokens 编码声学细节。VoxInstruct 的三阶段 pipeline (instruction→ST→coarse AT→fine AT) 正是利用了这种层级性。

> [!summary] 速查
> - **一句话**: 提出首个统一的多语言 instruction-to-speech 框架,用一条自然语言指令同时传达内容和风格,通过 semantic token 中间表征 + 多重 CFG 策略替代传统的 content/description 分离输入 [§1]
> - **路线**: Instruction Text → MT5 Encoder (with LoRA) → Text Embedding → AR LLaMA (Stage I: →Language ID + Semantic Tokens; Stage II: →Coarse AT Q1) → NAR LLaMA (Stage III: →Fine AT Q2-Q8) → Vocos Decoder → Waveform [§4.1-4.2, Fig 2]
> - **指标**: EN MOS-Q 4.22 / MOS-I 3.76, WER 2.5%, attribute accuracy 80.54% [Table 2]; stress Acc_word 88.29% / Acc_sentence 87.17% [Table 4]; voice cloning MOS-Q 4.06 / MOS-S 4.01 (mono, vs VALL-E 3.48/3.99) [Table 5]
> - **可借鉴**: (1) Semantic token 做 instruction-to-content 桥梁: 去除连续重复 token 后的 HuBERT ST 序列让模型先"理解"要说什么再生成声学细节,去掉 ST 后 WER 从 3.0 骤升到 26.2 [Table 6]; (2) 多重 CFG 在 codec LM 中的应用: 对 instruction→ST 和 instruction/ST→AT 分别做 CFG,各自增强不同维度的控制力 [§4.3]; (3) 渐进式 fine-tuning: pre-train (transcript-only) → fine-tune (instruction) → fine-grained fine-tune (stress) 三阶段逐步解锁控制粒度 [§4.4]
> - **局限**: (1) Emotion accuracy 仅 59.81%,是所有属性中最低的,论文未深入分析原因 [Table 2]; (2) 训练数据中 instruction-speech 对仅 2.4K 小时,fine-grained 仅 200 小时,规模较小; (3) 仅支持英文和中文; (4) 论文发表于 2024 年,其 WER 2.5% 和 MOS 4.22 已被后续系统 (如 CosyVoice2, Seed-TTS) 超越

## 核心问题

现有 TTS 系统在处理自然语言指令方面存在两个结构性限制 [§1] [论文原文]:

1. **输入格式不自然**: PromptTTS、InstructTTS、Salle 等方法要求将输入拆分为 content prompt (transcript) 和 description prompt (style/speaker)。这与 text-to-image 等其他 AIGC 模态不一致 -- 在图像生成中,用户用一条自然语言 prompt 同时描述内容和风格 [§1] [论文原文]。

2. **控制粒度受限**: 使用独立 description prompt 建模语音风格时,description 与 transcript 是解耦的,只能实现全局风格控制 (如整句语速快),无法做到词级控制 (如强调某个特定词) [§1] [论文原文]。

**为什么之前没人做**: 核心瓶颈在于数据 -- 大规模 instruction-speech 配对数据稀缺,现有公开数据集仅有 transcript 标注,没有丰富的风格描述 [§4.4] [agent 解读]。此外,直接从自然语言指令中提取内容 (什么话要说) 本身就是一个困难的序列理解问题,phoneme 输入通过 G2P 预处理绕过了这个问题 [§4.2] [agent 解读]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VoxInstruct 由四个组件构成 [§4.1, Fig 2]:
- **Text encoder**: MT5-base (多语言 T5) 的预训练 encoder,插入 LoRA (r=16) 适配器微调 [§4.1]
- **Acoustic encoder**: EnCodec,提取 8 层 RVQ 离散 acoustic tokens [§4.1]
- **Codec language model**: 基于 LLaMA 架构的 AR + NAR transformer,12 层,hidden dim 1024,FFN dim 4096 [§5.1]
- **Acoustic decoder**: Vocos (替代原始 EnCodec decoder,重建质量更好) [§4.1]

生成过程分三个阶段,前两个阶段共享同一个 AR 模型 [§4.2]:

**Stage I (AR): Instruction → Semantic Tokens** [§4.2]
MT5 将 instruction 编码为 text embedding 序列 E_ins。AR 模型先预测 language label l,再逐步生成 semantic token 序列 S (由 HuBERT + k-means 提取,去除连续重复) [§4.2]。

**Stage II (AR): Semantic Tokens → Coarse Acoustic Tokens** [§4.2]
同一个 AR 模型继续,以 text embedding + language label + semantic tokens 为条件,逐步生成 EnCodec 第一层 quantizer 的 acoustic tokens A(:,1)。训练时 Stage I 和 II 的输入以拼接形式 <E_ins, l, S, A(:,1)> 一次性输入 AR 模型,通过 causal attention mask 同时训练两个阶段 [§4.2]。

**Stage III (NAR): Coarse → Fine Acoustic Tokens** [§4.2]
NAR 模型 (同样基于 LLaMA 但无 causal mask) 逐层生成 quantizer 2-8 的 acoustic tokens,采用 MaskGIT 式迭代并行解码 [§4.2]。NAR 模型可选择性地接受 speech prompt A_tilde 作为额外输入,训练时以 0.3 概率不使用 speech prompt [§4.2]。

### 关键设计选择

**设计 1: Semantic Token 作为 Instruction-to-Content 桥梁**

**为什么需要**: 语音比图像对内容对齐要求更严格 -- 发音单元的正确出现和顺序直接影响可懂度。直接从 instruction text embedding 映射到 acoustic tokens 较困难 [§4.2] [论文原文]。

**怎么做**: 引入 HuBERT semantic tokens 作为中间表征。这些 tokens 编码语音的高层内容信息,但不包含韵律 (如时长) 或说话风格,通过去除连续重复 tokens 进一步压缩 [§4.2] [论文原文]。模型先学会从 instruction 中提取"说什么",再学会"怎么说"。

**证据**: 消融实验 [Table 6] 中,移除 ST guidance 后 WER 从 3.0% 骤升至 26.2%,attribute accuracy 从 71.68% 降至 65.08%。这证明 ST 不仅帮助内容理解,对整体生成质量也有间接贡献 [论文原文]。

**设计 2: 多重 Classifier-Free Guidance**

**为什么引入**: CFG 在 text-to-image (diffusion) 中已证明有效,Sanchez et al. (2023) 进一步展示 CFG 也适用于 LLM 的 text generation。VoxInstruct 首次将 CFG 引入 speech codec language model [§4.3] [论文原文]。

**怎么做**: 训练时以 0.1 概率 mask 条件序列实现无条件生成。推理时使用三种 CFG [§4.3]:
1. **Instruction CFG on ST** (强度 gamma): 增强 instruction 对 semantic token 的引导 → 改善属性控制 [Eq 4]
2. **Instruction CFG on AT** (强度 alpha): 增强 instruction 对 coarse AT 的引导 → 更显著改善属性控制 [Eq 5]
3. **ST CFG on AT** (强度 beta): 增强 semantic tokens 对 coarse AT 的引导 → 改善可懂度 [Eq 6]

**证据**: 消融 [Table 6] 单独设置各 CFG=2.0 (其余=1.0),Instruction CFG on AT 最有效地提升 attribute accuracy (75.70% vs baseline 71.68%); ST CFG on AT 最有效降低 WER (2.5% vs baseline 3.0%) [论文原文]。

**设计 3: 统一指令格式**

**为什么这样做**: 与 text-to-image 的单 prompt 输入对齐。传统方法要求 content 和 description 分开输入,VoxInstruct 允许 spoken content 自由嵌入 instruction 的任意位置 (前/后/中间),类似小说或文章的写作方式 [§3] [论文原文]。

**实际效果**: 统一指令格式不仅更自然,还使细粒度控制成为可能。例如可以写"giving 'little' a pronounced stress"来指定特定词的重音 [Fig 1, Fig 2] [论文原文]。这是传统 content/description 分离方法无法实现的,因为 description 不知道 content 中哪个词需要强调 [agent 解读]。

**设计 4: Speech Prompt + Instruction 组合**

VoxInstruct 的 NAR 阶段支持 optional speech prompt 输入,使模型可以同时利用 speech prompt 的音色特征和 instruction 的风格/内容描述 [§3, §4.2] [论文原文]。当 instruction 仅包含 spoken content (无 description) 时,模型退化为标准 zero-shot voice cloning TTS [§3] [论文原文]。这是首次实现 speech prompt + text instruction 联合输入 [论文原文]。

### 训练策略

三阶段渐进训练 [§4.4]:

1. **Pre-training** (transcript-only, 13.4K h): GigaSpeech-xl (EN, 7,117h) + WenetSpeech (ZH, 6,319h),instruction 仅包含引号括起的 transcript (即 x_ins = x_con)。目的: 建立 text-to-speech 的基础能力和 zero-shot voice cloning 能力 [§4.4, Table 1]。1M iterations, batch 64, 8xA100 [§5.1]。

2. **Fine-tuning** (instruction, 2.4K h): GigaSpeech-m + LibriTTS-R + TextrolSpeech + in-the-wild corpus (EN, 1,331h) + AISHELL3 + Zhvoice + in-the-wild corpus (ZH, 1,116h)。指令包含 spoken content + 声学属性 + 说话人 + 情感 + 场景背景描述。标注系统详见 SpeechCraft (Jin et al., 2024) [§5.1, Table 1]。800K iterations, batch 32 [§5.1]。

3. **Fine-grained fine-tuning** (stress, 200 h): LibriTTS-stress (EN, 149h) + AISHELL3-stress (ZH, 51h)。指令包含词级重音信息 [§5.1, Table 1]。100K iterations [§5.1]。

**为什么要三阶段而非一阶段**: instruction-speech 配对数据规模远小于 transcript-only 数据 (2.4K vs 13.4K h)。先用大量 transcript 数据建立语音合成基础能力,再用小量但信息更丰富的 instruction 数据微调,最后用更少量的 fine-grained 数据解锁词级控制 [§4.4] [论文原文]。消融 [Table 2] 显示 pre-training 阶段对 WER 有显著帮助 (3.3→2.5) [论文原文]。

## 实验

### 主要结果

| 指标 | VoxInstruct | VoxInstruct (w/o PT) | Salle | PromptTTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS-Q ↑ | **4.22** | 4.11 | 3.67 | 1.82 | GigaSpeech-s (EN) | [Table 2] |
| MOS-I ↑ | **3.76** | 3.66 | 3.18 | 2.07 | GigaSpeech-s (EN) | [Table 2] |
| WER ↓ | **2.5** | 3.3 | 7.2 | 11.6 | GigaSpeech-s (EN) | [Table 2] |
| Attr Acc (mean) ↑ | **80.54** | 79.96 | 67.88 | 67.14 | GigaSpeech-s (EN) | [Table 2] |
| MCD ↓ | **11.864** | 16.273 | 12.768 | - | GigaSpeech-s (EN) | [Table 2] |
| SECS (GT) ↑ | **0.641** | 0.622 | 0.595 | 0.577 | GigaSpeech-s (EN) | [Table 2] |

中文结果 (LLM-aided instruction generation) [Table 3]:
- VoxInstruct: MOS-Q **4.01** / MOS-I **3.83** (vs Salle 2.77/2.68, PromptTTS 2.36/2.17)
- 客观指标与 baseline 可比或略低,但主观评价显著领先 [Table 3]

### Fine-grained Control (Stress)

| 指标 | VoxInstruct | Salle | PromptTTS | 出处 |
| --- | --- | --- | --- | --- |
| Acc_word ↑ | **88.29** | 81.75 | 76.46 | [Table 4] |
| Acc_sentence ↑ | **87.17** | 71.96 | 65.59 | [Table 4] |

统一指令格式的优势在 fine-grained control 上尤为明显: VoxInstruct 的 Acc_sentence 比 Salle 高 15.21 个百分点,因为 Salle 将 content 和 style 分开建模,难以关联"哪个词需要强调" [Table 4] [agent 解读]。

### Voice Cloning

| 指标 | VoxInstruct (mono) | VALL-E | VoxInstruct (cross) | VALL-E X | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS-Q ↑ | **4.06** | 3.48 | 3.68 | **4.01** | [Table 5] |
| MOS-S ↑ | **4.01** | 3.99 | 3.86 | 3.85 | [Table 5] |
| SECS ↑ | 0.824 | **0.839** | 0.816 | 0.811 | [Table 5] |

VoxInstruct 在 instruction fine-tuning 后仍保留了竞争力强的 voice cloning 能力,MOS-Q 显著优于 VALL-E,归因于 Vocos decoder 和 MT5 encoder 的语义信息增强 [§5.5] [论文原文]。

### 消融实验

| 配置 | WER ↓ | MCD ↓ | SECS ↑ | Attr Acc ↑ | 出处 |
| --- | --- | --- | --- | --- | --- |
| w/o CFG (baseline) | 3.0 | 11.861 | 0.609 | 71.68 | [Table 6] |
| + CFG Instruction on ST | 3.5 | 12.441 | 0.607 | 72.30 | [Table 6] |
| + CFG Instruction on AT | 2.7 | 11.692 | 0.615 | **75.70** | [Table 6] |
| + CFG ST on AT | **2.5** | 11.704 | 0.614 | 71.68 | [Table 6] |
| - ST guidance | 26.2 | 14.146 | 0.599 | 65.08 | [Table 6] |

关键发现 [§5.6] [论文原文]:
- Instruction CFG on AT 对属性控制最有效 (+4.02 accuracy)
- ST CFG on AT 对可懂度最有效 (WER 2.5 vs 3.0)
- ST guidance 是不可或缺的: 移除后 WER 从 3.0 暴涨至 26.2

## 局限性

1. **Emotion 控制最弱**: Emotion accuracy 仅 59.81% (Ground Truth 也仅 62.03%),是所有属性因子中最低的 [Table 2]。论文未分析原因,可能与情感本身的主观性和标注噪声有关 [agent 解读]。

2. **数据规模受限**: instruction-speech 配对数据仅 2.4K 小时 (fine-grained 仅 200 小时),与后续系统 (如 Seed-TTS 的几十万小时) 相比差距悬殊。pre-training 阶段的 13.4K 小时也不算大规模 [Table 1] [agent 解读]。

3. **语言覆盖有限**: 仅支持英文和中文,尽管框架设计 (MT5 多语言 encoder) 理论上可扩展 [论文原文]。

4. **评价存在局限**: 主观评价仅 20 名参评者,5 分 MOS 量表,无置信区间报告 [§5.1]。voice cloning 对比使用 VALL-E 和 VALL-E X demo 页的样本而非同条件生成 [§5.2] [论文原文]。

5. **Semantic token 依赖 HuBERT**: 使用外部预训练的 HuBERT 提取 semantic token,其质量和语言覆盖直接制约系统表现。HuBERT 主要在英文数据上训练,中文 semantic token 的质量可能较低 [agent 解读]。

## 点评

**VoxInstruct 的核心贡献在于重新定义了 TTS 的输入格式**: 从 (content, description) 的结构化二元输入,走向了单条自由格式 instruction 的统一输入。这不仅是 UX 层面的改进,更重要的是打开了 fine-grained control 的可能性 -- 当 content 和 style description 在同一条指令中交织时,模型可以建立词级别的内容-风格关联 (如 "stress the word 'always'")。

**Semantic token 作为中间表征是精巧的工程选择**: WER 从 3.0 到 26.2 的消融差距说明,对于 instruction-to-speech 这种高度灵活的输入格式,让模型先"想清楚要说什么"再"决定怎么说"的分步策略比端到端映射有效得多。这个 insight 对其他 instruction-guided 生成任务也有参考价值。

**多重 CFG 策略的设计反映了对 codec LM 条件分解的理解**: 对不同条件 (instruction embedding, semantic tokens) 在不同生成阶段 (ST, AT) 分别做 CFG,而非简单地对所有条件统一做 CFG。消融实验清晰展示了各个 CFG 通道各自增强的维度。

**历史定位**: VoxInstruct 发表于 2024 年 ACM MM,在 timeline 上位于 VALL-E (2023) 之后、CosyVoice (2024)和 Seed-TTS (2024) 同期。它的主要价值不在绝对性能指标 (后续系统已超越),而在于率先提出了 instruction-to-speech 这一新任务定义和完整的技术路线。后续的 CosyVoice 等系统也采纳了 instruction-guided 的输入范式,但使用了不同的技术栈 (flow matching 等)。

## 可复用的 idea

1. **Semantic token 做 content bridge**: 在任何需要从复杂自然语言输入中提取"要做什么"的生成任务中,先用信息量化的中间表征锁定核心内容,再生成细节。不限于 TTS,也适用于 instruction-guided music generation 等 [Table 6 消融支撑]。

2. **多条件分离 CFG**: 当生成任务有多种条件信号时,对每种条件单独做 CFG 并调整各自的 guidance strength,比统一 CFG 更精细。特别适用于有 content + style 两类条件的场景 [§4.3, Table 6]。

3. **渐进式 fine-tuning 解锁控制粒度**: 数据量按控制粒度分层: 大量粗粒度数据建立基础 → 中量中粒度数据添加可控性 → 少量细粒度数据实现精细控制。解决了细粒度标注数据稀缺的问题 [§4.4, Table 1]。

4. **Input format unification for fine-grained control**: 将 content 和 style 信息放在同一个序列中,使模型能建立跨信息类型的局部关联,实现传统分离输入无法达到的细粒度控制 [Table 4: Acc_sentence 87.17 vs Salle 71.96]。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 三阶段 pipeline 的 WHY 和 HOW 均有充分解释,每个设计选择有因果分析 |
> | 可信赖 | pass | 数字型 claim 有 [Table N]/[§X.X] 标注,覆盖率 >80%;指标名使用正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖方法节所有因果解释 |
> | 可定位 | pass | KB 背景有具体谱系定位 (对标 VALL-E, CosyVoice),创新判断有对比基准 |
> | 不污染 | pass | 无新建概念页;反向更新仅需 key_papers 追加 |
> 
> Issues: 2 (high: 0, medium: 0, low: 2)
> 详见 `_review/VoxInstruct-review.yml`
