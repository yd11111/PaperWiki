---
type: paper
tier: deep
title: "TextrolSpeech: A Text Style Control Speech Corpus with Codec Language Text-to-Speech Models"
arxiv_id: "2308.14430"
source: "Sources/TextrolSpeech.pdf"
authors: [Shengpeng Ji, Jialong Zuo, Minghui Fang, Ziyue Jiang, Feiyang Chen, Xinyu Duan, Baoxing Huai, Zhou Zhao]
year: 2023
venue: "arXiv"
tags: [dataset, controllable-TTS, text-style-prompt, codec-LM, emotion, style-control, RVQ, autoregressive]
concepts: ["[[NaturalLanguageDescriptionforTTS]]", "[[CodecLanguageModel]]", "[[EmotionControlinTTS]]", "[[StyleTransferinTTS]]", "[[ResidualVectorQuantization]]", "[[LLM-basedTTS]]"]
models: ["[[模型库/EnCodec|EnCodec]]"]
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[LLM-basedTTS]], [[SpeechTokenizer]], [[ResidualVectorQuantization]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓ | 过滤: [[NaturalLanguageDescriptionforTTS]](pending-review), [[CodecLanguageModel]](pending-review), [[EmotionControlinTTS]](pending-review), [[StyleTransferinTTS]](pending-review) | 未命中但可能相关: 无
>
> **LLM-based TTS**: TextrolSpeech/Salle 是 LLM-based TTS 早期探索文本风格控制的工作。KB 记录了 PromptTTS (Guo et al., 2023) 作为开创性工作使用 5 属性文本描述控制 TTS。TextrolSpeech 在数据集规模和描述多样性上超越了 PromptTTS 的 PromptSpeech 数据集 (27,893 descriptions vs 236,220)。Salle 模型采用 codec LM 路线,与 VALL-E 的 AR+NAR 两阶段类似但增加了 style prompt 输入。
>
> **Speech Tokenizer**: Salle 使用 EnCodec [9] 作为 acoustic tokenizer,产生 T x n 的二维码矩阵 (n=8 layers) [§3]。模型利用 RVQ 的层级特性: 第一层编码 speaker identity 等主要信息,后续层编码细粒度声学细节 [论文原文]。
>
> **Residual Vector Quantization**: Salle 在 RVQ 的第一层使用 AR 生成 (建模语义和说话人),后续层使用 NAR 并行生成 (补充声学细节) [§3],这与 VALL-E 的 AR+NAR 架构一致。

> [!summary] 速查
> - **一句话**: 发布首个大规模开源文本风格控制语音数据集 TextrolSpeech (330h, 236K style descriptions, 5 style factors),并提出 codec LM 模型 Salle 用文本描述直接引导 acoustic token 生成 [§Abstract]
> - **路线**: Text + Style Prompt → Phoneme Conversion → [Style Token Embeddings, Text Token Embeddings] → AR Codec LM (theta_SAR, 生成第 1 层 RVQ) → NAR Codec LM (theta_SNAR, 生成第 2-8 层 RVQ) → Audio Codec Decoder → Waveform [§3, Fig 2]
> - **指标**: Style factor accuracy: Gender 95.5%, Pitch 90.5%, Speech 85%, Volume 86%, Emotion 81% (avg 87.6% vs PromptTTS 82.3%) [Table 2]; MOS-Q 3.78±0.09 / MOS-S 3.88±0.10 (vs PromptTTS 3.76/3.74) [Table 3]
> - **可借鉴**: (1) Multi-stage prompt programming: 用 GPT-3.5-Turbo 通过 Base→Diversity→Constraint→Few-shot 四阶段生成 500 种自然文本描述/style group [§2.3]; (2) 数据集构建方法论: LibriTTS + VCTK + emotion datasets (ESD/TESS/MEAD/SAVEE/MESS) → acoustic feature extraction + text-speech alignment → 自动标注 5 factors [§2.2]
> - **局限**: (1) MOS-Q 3.78 (满分 5),音质仍有明显差距; (2) Emotion accuracy 最低 (81%),论文归因于数据有限和情感分类固有困难 [§4.2.1]; (3) 模型架构较简单 (6 层 Transformer),未采用更强的 LLM backbone; (4) 数据集仅英文; (5) 未开源模型权重

## 核心问题

可控 TTS 的传统方法依赖: (1) 参考音频风格迁移 [4],需要用户提供符合要求的参考音频,操作不便; (2) 显式数值参数 (pitch/speed/energy) [5,6,7],需要专业声学知识 [§1] [论文原文]。

**自然语言文本描述** 是更直觉的控制方式,但面临两个障碍 [§1] [论文原文]:
1. **数据稀缺**: 现有文本风格数据集要么不开源 (StylePrompt, NLSpeech),要么规模小、描述模板化 (PromptSpeech 仅 27,893 descriptions, 5 types/style group) [Table 1]
2. **模型不足**: 缺乏能有效利用丰富文本描述的 TTS 架构

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### TextrolSpeech 数据集

**数据来源** [§2.2]:
- 语音: LibriTTS clean + VCTK (语音内容) + ESD/TESS/MEAD/SAVEE/MESS (情感) → 共 42,909 条情感语音 + 330h 总时长 [论文原文]
- 采样率统一为 24kHz [§2.1]
- 分割: 200 validation + 200 test + 其余 training [§2.1]

**5 个风格因子** [§2.1]:
- Gender: male/female (2 类)
- Pitch: high/low/normal (3 类)
- Speaking Speed: high/low/normal (3 类)
- Volume: high/low/normal (3 类)
- Emotion: angry/contempt/disgusted/fear/happy/sad/surprised/neutral (8 类)

**标注方法** [§2.2]: 对无情感内容的样本标注为 neutral; 使用 librosa 分析 energy/pitch + WORLD vocoder 提取 pitch + forced alignment 获取 duration; 将连续值三分类 (high/low/normal) [论文原文]

**Prompt Programming (文本描述生成)** [§2.3, Fig 1]:
使用 GPT-3.5-Turbo 通过四阶段 prompt 生成 500 种自然描述/style group [论文原文]:
1. **Base**: 明确任务 — "Generate one sentence that describe different, natural and brief speaking style based on four keywords" [论文原文]
2. **Increasing diversity**: 允许同义词替换和自由发挥 [论文原文]
3. **Reducing irrelevant descriptions**: 限制无关内容 — "Please remember do not include scene descriptions such as 'churches'" [论文原文]
4. **Few-shot templates**: 提供模板参考 — "The rapid, loud and high-keyed voice belongs to the girl" [论文原文]

这使描述数从先前工作的 ~5 types/style group 扩展到 **500 types/style group** [§2.3]。

### Salle 模型架构

**基于 EnCodec [9]** 的两阶段 codec LM [§3]:

**Stage 1 — AR model (theta_SAR)** [§3, Eq.2]:
- Decoder-only 架构,不需要 text-speech alignment [论文原文]
- 输入: text style prompt S + text token embeddings x
- 输出: 第 1 层 RVQ codes A_(1:T,1)
- 公式: p(A_(1:T,1)|S, x; theta_SAR) = prod_{t=0}^T p(A_(t,1)|A_(<t,1), S, x; theta_SAR) [Eq.2]
- **关键设计**: style prompt 直接参与第 1 层 token 生成的 conditioning,因为第 1 层主要编码 speaker identity 等主信息 [论文原文]

**Stage 2 — NAR model (theta_SNAR)** [§3, Eq.3]:
- 并行生成第 2-8 层 RVQ codes
- 输入: text information x + 已生成的前层 acoustic tokens
- **不包含** text style prompt (风格信息已通过第 1 层 token 传递) [论文原文]
- 公式: p(A_(:,2:n)|x; theta_SNAR) = prod_{i=2}^n p(A_(:,i)|A_(:,<i), x; theta_SNAR) [Eq.3]

**训练细节** [§4.1]:
- 两个模型共享架构: 6 层 Transformer, 16 attention heads, embedding dim 512, FFN dim 2048, dropout 0.1
- 仅对 A_(1:T,n) 计算 cross-entropy loss [论文原文]
- 训练中随机选择 A_(:,i) 作为 ground truth 以缓解 NAR 的 learned bias [论文原文]
- 4x NVIDIA V100 32GB, batch size 6k tokens/GPU, 200K steps
- AdamW, warm up 32K steps (1e-7 → 5e-4), then linear decay

### 关键设计选择

**为什么在 NAR 阶段去掉 style prompt?** [agent 解读]
RVQ 的层级结构使得第 1 层包含主要语义和说话人信息。style prompt 引导第 1 层生成后,风格信息已隐含在 token 中,后续层只需基于前层补充声学细节 [agent 解读]。这也减少了 NAR 模型的输入复杂度。

**为什么用 codec tokens 而非 Mel spectrogram?** [§1]
论文明确指出 "utilizing audio codec codes as an intermediate representation to replace the conventional mel-spectrogram" [论文原文]。Codec tokens 具有层级结构 (RVQ),可利用不同层的信息粒度进行分阶段建模; 且 codec tokens 与 LM 的 token-based paradigm 天然兼容 [agent 解读]。

## 实验

### Style Factor Accuracy [Table 2]

| Style Factor | PromptTTS | Salle | 出处 |
| --- | --- | --- | --- |
| Gender | 92.5 | **95.5** | [Table 2] |
| Pitch | 82.5 | **90.5** | [Table 2] |
| Speech speed | 83 | **85** | [Table 2] |
| Volume | 82 | **86** | [Table 2] |
| Emotion | 71.5 | **81** | [Table 2] |
| **Mean** | 82.3 | **87.6** | [Table 2] |

### Speech Quality [Table 3]

| Model | MOS-Q | MOS-S | 出处 |
| --- | --- | --- | --- |
| GT | 4.21±0.09 | 4.25±0.08 | [Table 3] |
| GT-codec | 4.10±0.11 | 4.13±0.10 | [Table 3] |
| PromptTTS | 3.76±0.12 | 3.74±0.09 | [Table 3] |
| Salle | **3.78±0.09** | **3.88±0.10** | [Table 3] |

Gender accuracy 最高 (95.5%),emotion accuracy 最低 (81%)。论文解释: gender 信息容易学习,emotion 受限于数据量和分类固有困难 [§4.2.1] [论文原文]。

## 局限性

1. **音质差距**: MOS-Q 3.78 vs GT 4.21,差距 0.43 分 [Table 3]。论文承认 "the overall sound quality is not exceptionally high" [§4.2.2] [论文原文]
2. **Emotion 控制最弱**: 81% accuracy,论文归因于 "limited data, interference from neutral emotions, and the inherent difficulty of emotion classification" [§4.2.1] [论文原文]
3. **数据集仅英文**: 缺少多语言支持 [agent 解读]
4. **模型较小**: 6 层 Transformer (dim 512),未利用更强的预训练 LLM [agent 解读]
5. **评估有限**: 仅 MOS 和 classification accuracy,缺少 speaker similarity、WER 等指标 [agent 解读]

## 点评

TextrolSpeech 的主要贡献在**数据集**而非模型。236,220 条自然语言风格描述配对语音,相比 PromptSpeech 的 27,893 条是接近 10x 的提升 [Table 1]。特别是 prompt programming 方法论 (4 阶段 GPT 生成) 是一个可复用的数据构建范式 [agent 解读]。

Salle 模型本身相对简单: 6 层 Transformer + EnCodec tokens 的 AR+NAR 两阶段。但其设计选择有教育意义: 在 AR 阶段用 style prompt 引导第 1 层 token 生成,在 NAR 阶段去掉 style prompt,利用了 RVQ 的自然层级结构 [agent 解读]。

**历史定位**: 这篇 2023 年的工作是 description-based TTS 数据集领域的重要贡献,填补了开源大规模文本风格标注数据集的空白。后续工作如 Parler-TTS (2024) 和 SpeechCraft (2024) 在此基础上进一步扩展了描述的丰富度和多样性 [agent 解读]。

## 可复用的 idea

1. **Multi-stage prompt programming**: Base→Diversity→Constraint→Few-shot 四阶段 GPT 生成描述,可用于任何需要大量自然语言标注的数据集构建 [§2.3]
2. **Style prompt 仅参与 AR 第 1 层**: 利用 RVQ 层级结构,将风格信息注入最核心的 token 层,后续层自动继承 [§3]
3. **大规模自动标注 + 人工质检**: acoustic feature extraction → 三分类 → GPT 生成自然描述,自动化程度高 [§2.2]

---

检索命中: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓ | 过滤: [[NaturalLanguageDescriptionforTTS]](pending-review), [[CodecLanguageModel]](pending-review), [[EmotionControlinTTS]](pending-review), [[StyleTransferinTTS]](pending-review) | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
