---
type: paper
tier: deep
title: "XTTS: a Massively Multilingual Zero-Shot Text-to-Speech Model"
arxiv_id: "2406.04904"
source: "Sources/XTTS.pdf"
authors: [Edresson Casanova, Kelly Davis, Eren Golge, Gorkem Goknar, Iulian Gulea, Logan Hart, Aya Aljafari, Joshua Meyer, Reuben Morais, Samuel Olayemi, Julian Weber]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, zero-shot, multilingual, cross-lingual, voice-cloning, language-model, VQ-VAE, GPT-2, open-source]
concepts: ["[[CodecLanguageModel]]", "[[SpeakerEmbedding]]", "[[SpeakerAdaptation]]", "[[Single-codebookvsMulti-codebook]]", "[[VoiceCloningTaxonomy]]"]
models: ["[[模型库/VITS|YourTTS]]", "[[模型库/HierSpeech++|HierSpeech++]]", "[[模型库/EnCodec|EnCodec]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: XTTS 属于 Codec Language Model 路线中的 VQ-VAE + AR LM 范式,是 Tortoise 的多语言扩展版。在 [[VoiceCloningTaxonomy]] 的四分类中,XTTS 同时覆盖 Zero-shot VC (推理时无需微调) 和 Multilingual VC (16 语言) 两个类别,且支持 Speaker Adaptation (10 min fine-tuning)。Survey (Azzuni & El Saddik, 2025) 在 Multilingual VC 的 "5+ languages" 子类中明确提及 XTTS (VQ-VAE + GPT-2)。

**已有认知**:
- [[Zero-shotSpeechSynthesis]] (confirmed): 当前 SOTA 已演进到 CosyVoice 3、IndexTTS2、Qwen3-TTS 等系统。XTTS 使用的 language model + discrete token 路线是三大主流方法之一,但其 VQ-VAE 单码本方案在 [[Single-codebookvsMulti-codebook]] [待确认] 的演进中处于较早期位置。
- [[Cross-lingualVoiceCloning]] (confirmed): 当前 SOTA (CosyVoice 3, Qwen3-TTS) 支持 4-9 语种,XTTS 覆盖 16 语种的广度在当时(2024)是最大规模。
- [[SpeakerEmbedding]] (confirmed): XTTS 使用 H/ASP speaker encoder (多层注意力统计池化),结合 Speaker Consistency Loss (SCL),在跨语言场景中通过 fine-tuning 提升 SECS 从 0.5852 到 0.7166。
- [[CodecLanguageModel]] [待确认]: XTTS 的 VQ-VAE + GPT-2 属于 codec LM 范式的早期形态,使用单码本 (8192→1024 codes) 而非 RVQ,帧率 21.53 Hz 远低于 EnCodec 的 75 Hz,天然适合 AR 建模。

**创新判断**: 相对已有认知,XTTS 的核心差异化在于:
1. 语言覆盖规模 (16 语种 vs 前人最多 6 语种),尤其包含低资源语言
2. Perceiver Resampler 替代单 embedding 进行说话人条件注入 (32 embeddings vs 1)
3. 码本过滤策略 (8192→1024 最高频 codes)
4. CJK 文字罗马化预处理

> 检索命中: [[Zero-shotSpeechSynthesis]]✓, [[Cross-lingualVoiceCloning]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[VoiceCloningTaxonomy]](pending-review), [[SpeakerAdaptation]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 Tortoise 改进的多语言零样本 TTS,通过 VQ-VAE + GPT-2 + HiFi-GAN 三组件架构实现 16 语种的语音克隆与跨语言合成
> - **路线**: Text (BPE) + Reference Mel → Conditioning Encoder (Perceiver Resampler) → GPT-2 AR decoder (预测 VQ-VAE codes) → GPT-2 latent → HiFi-GAN (conditioned on H/ASP speaker embedding) → 24kHz Waveform
> - **指标**: 英语 CER 0.54, UTMOS 4.007, SECS 0.642 [Table 2]; 16 语种平均 CER 2.06, SECS 0.505 [Table 4]; CMOS +0.41/+0.92 vs HierSpeech++/Mega-TTS 2 [Table 3]; Fine-tuning 后跨语言 SECS 0.585→0.717 [§5]
> - **可借鉴**: (1) 码本频率过滤 (保留 Top-1024) 提升表现力; (2) Perceiver Resampler 生成固定数量 speaker embeddings,解耦参考音频长度; (3) 语言 batch balancer 平衡多语言训练
> - **局限**: SMOS 略低于 HierSpeech++/Mega-TTS 2 (speaker similarity 不如单语模型); VQ-VAE decoder 重建有 artifacts 需额外 HiFi-GAN; 低资源语言数据量极少 (如 Czech 52h, Japanese 57h)

## 核心问题

XTTS 要解决的核心问题是: **现有零样本 TTS 系统支持的语言数量极其有限** (YourTTS 3 语种, VALL-E X 2 语种, Voicebox 6 语种),导致大量中低资源语言无法受益于零样本语音克隆技术 [§1]。

具体挑战包括:
1. 如何在大规模多语言训练中保持说话人克隆能力 (多语言训练容易稀释 speaker identity 信息) [§2]
2. 如何处理不同书写系统 (CJK 文字) 的文本输入 [§2]
3. 如何在保持推理效率的同时扩展语言覆盖 [§6]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

XTTS 由三个组件构成 [§2, Fig 1]:

**1. VQ-VAE (13M params)**
- 输入 mel spectrogram,每帧编码为 1 个 codebook (8192 codes),帧率 21.53 Hz [§2]
- 架构和训练流程与 Tortoise 相同,但训练后过滤码本,仅保留最高频的 1024 个 codes [§2]
- [论文原文] 过滤低频 codes 能提升模型的表现力 (expressiveness) [§2]
- [agent 解读] 低频 codes 可能对应噪声或极端边缘情况,过滤后相当于减小 LM 的词表大小,降低建模难度并减少 codebook collapse 问题

**2. GPT-2 Encoder (443M params)**
- Decoder-only transformer,与 Tortoise 类似 [§2]
- 输入: 文本 BPE tokens (6681 词表的自定义分词器) [§2]
- 输出: 预测 VQ-VAE audio codes [§2]
- **条件注入**: Conditioning Encoder 处理参考 mel spectrogram → 6 层 16-head attention + **Perceiver Resampler** → 生成 32 个 1024 维 embeddings [§2]
- [论文原文] Tortoise 仅使用单个 1024 维 embedding,但在大规模多语言训练中,单 embedding 会导致说话人克隆能力下降。Perceiver Resampler 生成固定数量 embeddings,不受参考音频长度影响 [§2]
- [agent 解读] 32 个 embeddings 相比 1 个能编码更丰富的说话人特征 (如音色细节、韵律特征、口音信息),这在多语言场景中尤为关键,因为同一说话人在不同语言间的表现差异需要更细粒度的表示

**3. HiFi-GAN Decoder (26M params)**
- 输入: GPT-2 encoder 的 latent vectors (非 VQ-VAE codes) [§2]
- [论文原文] 直接从 VQ-VAE codes 重建音频会导致发音问题和 artifacts,因此使用 GPT-2 latent 作为 decoder 输入 [§2, 引用 Tortoise]
- 通过 H/ASP speaker embedding 在每个上采样层进行 linear projection 条件注入 [§2]
- 加入 Speaker Consistency Loss (SCL),借鉴自 YourTTS,提升说话人相似度 [§2]
- 训练时 VQ-VAE 和 encoder 使用 22.5 kHz,decoder 线性上采样至 24 kHz 输出 [§2]

### 关键设计选择

**为什么选单码本 VQ-VAE 而非多码本 RVQ?**
- [agent 解读] Tortoise 原始设计即使用单码本 VQ-VAE。XTTS 继承这一选择,21.53 Hz 帧率使得每秒仅约 22 个 tokens,而 VALL-E 使用 EnCodec 的 75 Hz x 8 codebooks = 600 tokens/s。极低的 token rate 使 AR 建模高效且序列短,适合 GPT-2 的 context window。这是 XTTS 推理速度快于 VALL-E 的关键原因 [§6]

**为什么要过滤码本 (8192→1024)?**
- [论文原文] 初步实验验证了过滤低频 codes 能提升模型 expressiveness [§2]
- [agent 解读] 大码本中低频 codes 的 embedding 在训练中更新不充分,可能引入噪声。保留高频 codes 实质上是一种数据驱动的码本剪枝,与 codebook collapse 的缓解策略异曲同工

**为什么 CJK 需要罗马化?**
- XTTS 对韩文用 hangul-romanize,日文用 Cutlet,中文用 Pypinyin 进行罗马化后再 BPE 分词 [§2]
- [agent 解读] CJK 字符集极大,直接 BPE 会导致词表膨胀或 OOV 问题;罗马化将所有语言统一到拉丁字母空间,使 6681 token 的小词表能覆盖 16 语种,代价是丢失部分字形级信息

**为什么 decoder 不直接用 VQ-VAE codes?**
- [论文原文] VQ-VAE 的高压缩率意味着 codes 信息损失大,直接重建会有发音问题和 artifacts [§2]
- [agent 解读] GPT-2 的 latent space 是连续的高维向量,包含比离散 codes 更丰富的信息,特别是 codes 间的过渡和上下文信息,使 HiFi-GAN 能生成更平滑的波形

### 训练策略

- **VQ-VAE**: 先独立训练 (与 Tortoise 同),训练后码本过滤 [§2]
- **GPT-2 Encoder**: AdamW (beta 0.9/0.96), LR 5e-5, batch 4 x 4 GPU x 16 grad accum, MultiStepLR (milestones 5k/150k/300k, gamma 0.5), 约 2.5M steps [§3.3]
- **语言 batch balancer**: 平衡各语言在 batch 中的比例 [§3.2]
- [论文原文] 语言 batch balancer 对多语言训练至关重要。YourTTS 原始多语言模型因法语仅 5 speakers、葡萄牙语仅 1 speaker,使用 balancer 时 66% batch 来自仅 6 speakers,导致过拟合并降低英语性能 [§4.1]
- 硬件: 4x NVIDIA A100 80GB [§3.3]

## 实验

| 指标 | 本文 (XTTS) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER (en) ↓ | **0.5425** | 0.6789 (Tortoise), 0.7741 (HierSpeech++), 1.4269 (Mega-TTS 2) | FLORES+ (240 sentences) + DAPS (20 speakers) | [Table 2] |
| UTMOS (en) ↑ | 4.007 | 4.457 (HierSpeech++), 4.426 (StyleTTS 2), 4.184 (Mega-TTS 2) | FLORES+ + DAPS | [Table 2] |
| SECS (en) ↑ | 0.6423 | 0.6530 (HierSpeech++), 0.5492 (Tortoise) | FLORES+ + DAPS, ECAPA2 encoder | [Table 2] |
| CMOS vs HierSpeech++ ↑ | +0.41 | 0 (reference) | English, 15 evaluators | [Table 3] |
| CMOS vs Mega-TTS 2 ↑ | +0.92 | 0 (reference) | English, 15 evaluators | [Table 3] |
| SMOS vs HierSpeech++ ↑ | -0.31 | 0 (reference) | English, 15 evaluators | [Table 3] |
| SMOS vs Mega-TTS 2 ↑ | -0.39 | 0 (reference) | English, 15 evaluators | [Table 3] |
| CER (16-lang avg) ↓ | **2.0646** | 4.7949 (YourTTS) | FLORES+ + DAPS (cross-lingual) | [Table 4] |
| SECS (16-lang avg) ↑ | **0.5046** | 0.4704 (YourTTS) | FLORES+ + DAPS, ECAPA2 | [Table 4] |
| SECS (fine-tuned, cross-lingual) ↑ | 0.7166 | 0.5852 (zero-shot) | 8 speakers, ~10 min each | [§5] |

**关键发现**:
1. XTTS 在 16 语种训练下的英语 CER 优于所有对比模型 (含单语模型),但 UTMOS 和 SECS 低于 HierSpeech++ [Table 2]
2. 主观评估中,XTTS 在自然度/音质 (CMOS) 上显著优于 HierSpeech++ 和 Mega-TTS 2,但在说话人相似度 (SMOS) 上略逊 [Table 3]
3. [论文原文] SMOS 略低是多语言训练复杂性的预期结果 [§4.1]
4. YourTTS 原始多语言 checkpoint 因数据分布不均 (66% batch 来自仅 6 non-English speakers) 导致过拟合,性能远低于重新训练的单语版本 (Exp 1 CER 1.091 vs Original CER 2.874);多语言扩展到 16 语种后英语进一步劣化 (Exp 2 CER 3.480),暴露了之前论文比较的不公平性 [§3.2, §4.1]
5. Fine-tuning (~10 min 数据) 可将跨语言 SECS 从 0.585 提升至 0.717,且能跨语言迁移风格 (如耳语) [§5]

## 局限性

1. **Speaker similarity 不如单语模型**: SMOS 评分低于 HierSpeech++ 和 Mega-TTS 2,多语言训练稀释了说话人表示能力 [Table 3, §4.1]
2. **低资源语言数据极不均衡**: Czech 52h, Japanese 57h vs English 14.5k h,数据量差距达 250 倍+ [Table 1],低资源语言质量难以保证
3. **VQ-VAE 重建瓶颈**: 需要额外 HiFi-GAN decoder 从 GPT-2 latent 重建,增加了系统复杂度。作者自己也指出未来要改进 VQ-VAE 使其 decoder 可直接用 [§6]
4. **CJK 罗马化损失**: 日韩中文经罗马化后丢失原始字形信息,可能影响同音异字场景的发音准确性 (如中文多音字)
5. **评估局限**: 跨语言评估使用英语说话人作为所有语言的参考 (DAPS),无法反映母语说话人的真实效果 [§4]

## 点评

XTTS 的主要价值在于**规模**而非**深度创新**: 将 Tortoise 的 AR language model 方案成功扩展到 16 语种,证明了 VQ-VAE + GPT-2 框架在多语言场景中的可行性。其开源 (Coqui TTS / HuggingFace) 也对社区有实际贡献。

然而从技术路线看,XTTS 处于 Codec LM 范式的较早期阶段:
- **单码本 VQ-VAE + 码本过滤**虽然简化了 LM 建模,但信息损失大,需要额外 HiFi-GAN 补偿,不如后来的 semantic + acoustic 两阶段方案 (CosyVoice) 或 single-codebook + high-quality vocoder 方案 (BigCodec + LLM)
- **Perceiver Resampler 替代单 embedding**是一个有效的工程改进,但相比后来 VALL-E 的 in-context prompt 方式仍显得较为固定
- **罗马化预处理**是权宜之计,后来的多语言模型 (CosyVoice 3, Qwen3-TTS) 通过 text LLM 的 tokenizer 或 byte-level encoding 更优雅地解决了多文字系统问题

论文在实验设计上有一个有价值的贡献: 揭示了之前论文用 YourTTS 原始多语言 checkpoint 作为 baseline 的不公平性 (因为数据分布严重不均导致过拟合),并通过控制变量实验证明了这一点 [§3.2, §4.1]。

## 可复用的 idea

1. **码本频率过滤**: 训练后根据 code 频率保留 Top-K codes,简单但有效的码本优化策略,可应用于任何 VQ-VAE/RVQ 系统
2. **Perceiver Resampler 用于 speaker conditioning**: 将可变长度参考音频映射为固定数量的 embeddings,解耦了音频长度与条件信号维度
3. **语言 batch balancer + 公平比较实验设计**: 在多语言/多任务训练中使用 batch balancer,并通过控制数据量的对比实验暴露先前工作的比较偏差
4. **Fine-tuning 实现风格跨语言迁移**: 用 10 min 单语数据 fine-tune 后,耳语风格可自动迁移到其他 15 种语言 [§5]

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes | 分数: 理解 8 / 溯源 8 / 严谨 8 / 导航 8 / KB安全 8
> **已修正**: Table 2 中 HierSpeech++ 与 Mega-TTS 2 数值混淆 (PDF 表格提取行对齐错误,通过 Table 4 交叉验证修正)
> **残留**: 速查指标 fine-tuning SECS 出处仅标 [§5] (正文 prose,无独立 table); YourTTS 链接指向 VITS 模型页 (vault 无独立 YourTTS 模型页)
> 详见 `_review/XTTS-review.yml`
