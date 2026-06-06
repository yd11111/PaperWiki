---
type: paper
tier: deep
title: "MunTTS: A Text-to-Speech System for Mundari"
arxiv_id: "2401.15579"
source: "Sources/MunTTS.pdf"
authors: [Varun Gumma, Rishav Hada, Aditya Yadavalli, Pamir Gogoi, Ishani Mondal, Vivek Seshadri, Kalika Bali]
year: 2024
venue: ""
tags: [TTS, low-resource, VITS, XTTS, Mundari, Indian-language, data-collection, endangered-language, end-to-end]
concepts: ["[[VariationalAutoencoderforTTS]]", "[[Text-to-SpeechPipeline]]", "[[TTSEvaluation]]", "[[MelSpectrogram]]", "[[SpeakerEmbedding]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeakerEmbedding]], [[VITS]], [[TTSEvaluation]], [[VariationalAutoencoderforTTS]], [[Text-to-SpeechPipeline]], [[MelSpectrogram]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文使用的 VITS 是 E2E TTS pipeline 的代表 (Stage 4: text → waveform),结合 conditional VAE + normalizing flow + adversarial training。VITS 在知识库中有丰富记录,其 LJ Speech MOS 4.43,VCTK multi-speaker MOS 4.38,推理速度 67.12x 实时 [VITS Table 1/3/4]。XTTS v2 是 Coqui 的跨语言 TTS 模型,KB 中模型库无独立页但 VITS 页有提及。
>
> **已有认知**: KB 中 SpeakerEmbedding (confirmed) 详细覆盖了 multi-speaker TTS 的两种范式 (lookup table vs speaker encoder)。TTSEvaluation (pending-review) 系统讨论了 MOS 评估的局限性 (ceiling effect, non-transferability, listener bias)。VariationalAutoencoderforTTS (pending-review) 解释了 VITS 的 VAE + flow 结合机制。
>
> **创新判断**: 本文不提出新架构,而是将成熟的 E2E TTS 模型 (VITS) 应用于极低资源语言 Mundari。核心贡献在于数据收集方法论和在低资源条件下的系统比较。KB 中 VITS 模型页已记录类似的低资源应用案例 (如 ShanghainTTS 做上海话 TTS、XPhoneBERT 在低资源越南语上的改进),本文提供了 E2E vs 多语言模型在极低资源场景的新对比证据。
>
> 检索命中: [[SpeakerEmbedding]]✓, [[VITS]][待确认], [[TTSEvaluation]][待确认], [[VariationalAutoencoderforTTS]][待确认], [[Text-to-SpeechPipeline]][待确认], [[MelSpectrogram]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 为极低资源印度语言 Mundari 构建首个专用多说话人 E2E TTS 系统,通过社区参与式数据收集 (27.5h) + VITS-44K 实现 MOS 3.69
> - **路线**: 印地语句子 → 人工翻译为 Mundari → 2 位说话人录制 (44.1kHz) → VITS/XTTS v2 训练 → 母语者 MOS 评估
> - **指标**: VITS-44K MOS 3.69+-1.18 / MCD 7.60 (最佳); GT-22k MOS 4.62; XTTS-finetuned MOS 0.05 (灾难性遗忘) [Table 2, 3]
> - **可借鉴**: 社区参与式低资源语言数据收集流程; 单语言 E2E 模型 vs 多语言模型的系统比较方法论; speaker-weighted sampler 处理说话人不平衡
> - **局限**: 仅 2 位说话人 (1M/1F, 严重不平衡 74% F); 无对比两阶段模型 (AM+Vocoder); 未开放训练数据; MOS 评价者仅 5 人/样本; 未报 CI 统计显著性检验

## 核心问题

1. **如何为一种只有 ~100 万母语者、数字资源几乎为零的语言构建 TTS 系统?** 这涉及数据收集 (文本获取、录音、质量控制) 和模型选择两个维度。
2. **对于极低资源语言, E2E 单语言模型 vs 多语言预训练模型哪个更优?** 特别是 XTTS v2 这类跨语言模型是否能通过微调有效迁移。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文不提出新架构,而是对比评估了多种现有 E2E TTS 模型在 Mundari 上的表现:

1. **VITS-22K**: 从头训练的 VITS (22.05kHz 下采样数据) [§4.2]
2. **VITS-44K**: 从头训练的 VITS (原始 44.1kHz 数据) [§4.2]
3. **XTTS v2 (finetuned)**: 在 Mundari 数据上微调的 XTTS v2 (22.05kHz) [§4.2]
4. **XTTS v2 (pretrained)**: 零样本使用预训练 XTTS v2 (Hindi 预训练) [§4.2]
5. **MMS-UNR**: Facebook MMS 项目中的 Mundari VITS 模型 [§4.2]

所有模型均为端到端架构,直接从文本生成波形 [论文原文: "we suggest the usage of single E2E models, as they are found to be significantly faster than two-stage models" §4.2]。

### 数据收集流程

这是本文的核心贡献之一 [§3]:

1. **文本获取**: 从 Karya 数据库获取 100,000 句印地语句子,随机选取 20,000 句,人工翻译为 Mundari (Devanagari 书写) [§3]
2. **翻译策略**: 翻译员被指示优先保证流畅性而非忠实度 [§3] [agent 解读: 这是合理的选择,因为 TTS 训练需要自然的语句而非逐词翻译]
3. **说话人选择**: 从 12 位候选人 (6M+6F) 中经阅读任务评估 → 6 人短名单 → 最终 1M+1F [§3]
4. **录音条件**: 录音棚环境, 44.1kHz, 32-bit, 通过 Karya 众包平台 [§3]
5. **最终数据**: 15,656 unique sentences, 26,868 recordings (含跨说话人重复), 27.51 hours [§3, Table 1]

**数据分布不平衡**: 女性录音占 74%, 男性仅 26% [§3]。训练时使用 speaker-weighted sampler 缓解 [§4.3]。

### 关键设计选择

1. **选择 E2E 模型而非两阶段**: [论文原文] 理由是 "construct a simple unified system for speech synthesis, designed for straightforward deployment and ease of use by the general public" [§7]。[agent 解读] 但这也意味着没有对比 GlowTTS + HiFi-GAN 等已在印度语言上验证过的组合 (EkStep Foundation, 2021; Prakash & Murthy, 2020),这是一个遗漏。

2. **44.1kHz vs 22.05kHz**: 原始数据为 44.1kHz,同时训练了下采样到 22.05kHz 的版本。结果显示保留原始采样率的 VITS-44K 全面优于 VITS-22K [Table 2, 3]。[agent 解读] 这表明对于高质量录音数据,避免下采样可以保留更多声学细节。

3. **学习率策略**: VITS 使用较高 LR 5e-4,XTTS v2 微调使用极低 LR 5e-6 + weight_decay 1e-2 [§4.3]。[agent 解读] XTTS v2 的低 LR 策略意图是避免灾难性遗忘,但结果表明即便如此仍然失败。

### 训练策略

- **VITS**: LR=5e-4, batch_size=128, ExponentialLR scheduler, AdamW optimizer [§4.3]
- **XTTS v2**: LR=5e-6, batch_size=256, MultiStepLR scheduler, AdamW (weight_decay=1e-2) [§4.3]
- 所有模型在单块 A100 80GB 上训练 2500 epochs,约 5 天收敛 [§4.3]
- 每 epoch 基于 dev set loss_1 做 checkpoint,取最优 [§4.3]
- Speaker-weighted sampler 处理说话人分布不平衡 [§4.3]

## 实验

| 指标 | 本文 (VITS-44K) | VITS-22K | XTTS-pretrained | XTTS-finetuned | MMS | GT-22k | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOS (Full) | **3.69+-1.18** | 3.04+-1.29 | 2.20+-1.32 | 0.05+-0.30 | 0.79+-1.02 | 4.62+-0.68 | Mundari test (n=100) | [Table 2] |
| MOS (Male) | **3.39+-1.25** | 2.65+-1.34 | 2.10+-1.36 | 0.13+-0.52 | 0.79+-1.02 | 4.59+-0.65 | Mundari test (n=26) | [Table 2] |
| MOS (Female) | **3.79+-1.13** | 3.18+-1.25 | 2.23+-1.31 | 0.02+-0.16 | - | 4.63+-0.69 | Mundari test (n=74) | [Table 2] |
| MCD (Full) | **7.60+-3.99** | 9.45+-3.71 | 15.80+-7.03 | 13.65+-5.92 | 15.13+-4.19 | - | Mundari test (n=1344) | [Table 3] |
| MCD (Male) | **7.27+-3.08** | 10.03+-4.05 | 13.89+-5.87 | 10.73+-5.33 | 15.13+-4.19 | - | Mundari test (n=350) | [Table 3] |
| MCD (Female) | **7.72+-4.25** | 9.24+-3.56 | 16.48+-7.27 | 14.69+-5.77 | - | - | Mundari test (n=996) | [Table 3] |

**关键发现**:

1. **VITS-44K 全面最优**: MOS 和 MCD 均显著优于所有其他模型 [Table 2, 3]
2. **XTTS v2 微调灾难性失败**: 微调后 MOS 从 2.20 暴跌至 0.05,生成 "nonsensical outputs" [§5.3]。[论文原文] 归因于 catastrophic forgetting [§5.3]
3. **XTTS v2 零样本存在 phantom speech**: 基于 GPT-2 的 XTTS v2 会在句末生成随机 Hindi 词和乱码,类似 LLM 幻觉 [§5.3]
4. **MMS 表现极差**: MOS 0.79 [Table 2]。[agent 解读] MMS 覆盖 1000+ 语言,每语言数据极少,质量自然不如专用模型
5. **男性合成质量显著低于女性**: VITS-44K Male MOS 3.39 vs Female MOS 3.79 [Table 2],反映训练数据的说话人不平衡 (男性仅 26%)

## 局限性

1. **仅使用 E2E 模型**: 论文明确承认排除了 acoustic model + vocoder 的两阶段方案 [§7]。但 EkStep 和 Kumar et al. 的工作表明两阶段方案在印度语言上可行,缺少这一对比使结论不完整。

2. **说话人严重不平衡**: 74% 女性 / 26% 男性,导致男性合成质量明显较低 [§7, Table 2]。论文承认"the challenge of recruiting native proficient speakers" [§7]。

3. **评估规模有限**: MOS 仅在 100 个样本上评估,每样本 5 位评价者。未报告评价者间一致性 (inter-rater agreement)。MOS 标准差很大 (1.02-1.32),暗示评分变异性高。

4. **缺乏统计显著性检验**: 模型间的 MOS 差异未做显著性检验。VITS-44K (3.69+-1.18) 与 VITS-22K (3.04+-1.29) 的差距是否显著需要验证。

5. **XTTS v2 微调策略单一**: 仅尝试了一种 LR 和 schedule,未探索 LoRA、adapter 等参数高效微调方案。"nonsensical outputs despite successful convergence" [§5.3] 暗示可能是 evaluation 或 checkpoint 选择问题而非必然的灾难性遗忘。

6. **未开放训练数据**: 仅开放模型,但数据不开放,限制了可复现性。

7. **缺乏消融实验**: 未分析 speaker-weighted sampler 的效果、数据量对质量的影响曲线、以及不同训练 epoch 的效果。

## 点评

这是一篇以**实践价值**为主的工作,核心贡献在于为一种真正的极低资源濒危语言 (Mundari, ~100 万母语者) 构建了 TTS 系统,并开源了模型。从技术深度看,本文没有方法创新,而是标准的"应用 + 比较"模式。

**值得注意的经验**:

1. **单语言从头训练 > 多语言微调**: 即使 XTTS v2 有 Hindi 的预训练知识 (Mundari 使用 Devanagari 书写,与 Hindi 共享字符和发音),微调仍然灾难性失败。这与 Kumar et al. (2023) "single-language models are preferable" 的发现一致,进一步在极端低资源场景下验证。

2. **数据收集是瓶颈而非模型**: 27.5 小时的高质量单语 TTS 数据可以让标准 VITS 达到可用质量 (MOS 3.69),但招募母语者、组织录音、确保质量是真正的挑战。

3. **采样率保留有价值**: 44.1kHz 直接训练优于下采样到 22.05kHz,这是一个简单但实用的发现。

**与 KB 已有知识的对比**: KB 中 VITS 模型页记录了类似的低资源应用 (ShanghainTTS 做上海话,XPhoneBERT 在低资源越南语上)。MunTTS 的 MOS 3.69 显著低于 VITS 在 LJ Speech 上的 4.43 [VITS Table 1],但考虑到数据量 (27.5h vs LJ Speech 24h) 和语言复杂性,这是合理的。ShanghainTTS 在更小数据上取得 MOS 4.14,但那是使用了语言学先验知识 (声调变调域标注) 的结果。

## 可复用的 idea

1. **社区参与式数据收集流程**: Hindi → 目标语言翻译 + 众包录音 + 分层说话人筛选,可迁移至其他低资源语言 TTS 项目。
2. **Speaker-weighted sampler**: 简单有效地处理多说话人数据不平衡,适用于任何多说话人 TTS 训练。
3. **保留原始采样率训练**: 如果录音质量高,避免下采样可获得一致性改进。
4. **反面教训 -- 多语言模型微调风险**: XTTS v2 在仅 27.5h 低资源语言数据上微调会灾难性遗忘,提示在迁移学习时需要更谨慎的策略 (如冻结更多层、使用 adapter)。
