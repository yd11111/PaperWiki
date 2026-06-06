---
type: paper
tier: deep
title: "IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS"
arxiv_id: "2409.05356"
source: "Sources/IndicVoices-R.pdf"
authors: [Ashwin Sankar, Mehak Singal, Srija Anand, Praveen Srinivasa Varadhan, Shridhar Kumar, Deovrat Mehendale, Sherry Thomas, Aditi Krishana, Giri Raju, Mitesh Khapra]
year: 2024
venue: "arXiv preprint"
tags: [TTS, dataset, multilingual, Indian-languages, speech-enhancement, data-pipeline, zero-shot, multi-speaker, low-resource]
concepts: ["[[SpeakerAdaptation]]", "[[TTSEvaluation]]", "[[SpeakerEmbedding]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 TTS 数据集工程范畴,目标是通过语音增强管道将 ASR 语料转化为 TTS 训练数据,进而实现印度语言的零样本说话人泛化。这与 [[Zero-shotSpeechSynthesis]] 任务页中记录的"数据规模决定泛化能力"的共识一致 -- NaturalSpeech 3 使用 200K 小时数据,Emilia 提供 101K 小时开源数据。本文走的是 LibriTTS-R 路线(增强已有 ASR 数据集),而非 Emilia 路线(从 in-the-wild 视频源头采集)。
>
> **已有认知**: [[Cross-lingualVoiceCloning]] 记录了低资源语言方向的跨语言 TTS 仍是开放问题; [[SpeakerAdaptation]][待确认] 中的非英语适应子课题指出"多数工作集中于英语,非英语仍待发展"; [[SpeakerEmbedding]] 中 XLSR-53 等跨语言自监督模型已被用于多语言 TTS。[[TTSEvaluation]][待确认] 记录了 NORESQA-MOS、SNR、C50 等本文使用的评估指标及其局限性。
>
> **创新判断**: 本文的核心贡献在数据而非模型 -- 首次为全部 22 种印度官方语言提供开源 TTS 数据集,且说话人数量 (10,496) 远超所有已知印度 TTS 数据集。数据增强管道(demixing→dereverberation→enhancement→filtering)是该工作的技术核心,但各组件(HTDemucs/VoiceFixer/DeepFilterNet3)均为现成工具,创新在于组合与跨语言验证。
>
> 检索命中: [[Zero-shotSpeechSynthesis]]✓, [[Cross-lingualVoiceCloning]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[TTSEvaluation]](pending-review), [[SpeakerAdaptation]](pending-review), [[Emilia]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过语音增强管道将 ASR 数据集 IndicVoices 转化为 1,704 小时、10,496 说话人、22 种印度语言的高质量 TTS 数据集,并配套首个印度语 zero-shot/few-shot/many-shot TTS benchmark
> - **路线**: IndicVoices (ASR, 44.1kHz) → HTDemucs (demixing) → VoiceFixer (dereverberation) → DeepFilterNet3 (enhancement) → 多指标过滤 (C50/SNR/pitch/speaking rate/CER) → 音量归一化 → IndicVoices-R
> - **指标**: N-MOS 3.38 (vs LJSpeech 4.36 / IndicTTS 4.29) | SNR 60.47 dB | C50 53.45 dB | speaker similarity 88.18% (VoiceCraft on IV-R) vs 78.93% (VoiceCraft on IndicTTS) [Table 1, Table 5]
> - **可借鉴**: 英语预训练的语音增强模型 (HTDemucs/VoiceFixer/DeepFilterNet3) 具有良好的跨语言泛化能力,可直接用于非英语语音数据处理; ASR→TTS 数据转化管道可迁移到其他低资源语言
> - **局限**: N-MOS 仍低于录音室质量数据集 (3.38 vs 4.29); 8kHz 会话子集未被利用; 仅使用 grapheme (无 phonemizer),对复杂音素映射的语言可能不够; 仅验证了 VoiceCraft 一个模型

## 核心问题

1. **印度语 TTS 的数据瓶颈是什么?** 现有印度 TTS 数据集仅覆盖最多 14 种语言,每种语言 1-2 个说话人,以朗读风格为主,总量有限(IndicTTS 284h / 27 speakers)。缺乏 YouTube 手动字幕等高质量网络数据源。[§1]
2. **为何选择 IndicVoices 作为源数据集?** 四方面优势: (a) 唯一覆盖全部 22 种印度官方语言; (b) 44.1kHz 高采样率; (c) 手动标注 + 质量保证; (d) 获得知情同意,符合伦理规范。[§3.1]
3. **英语预训练的语音增强模型能否跨语言泛化到印度语言?** 能。HTDemucs/VoiceFixer/DeepFilterNet3 均在英语数据上训练,但在印度语言上有效去噪/去混响/增强而不损害可懂度。[§1, §3.2]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文的核心贡献不是模型而是**数据管道**: 将 ASR 质量的 IndicVoices 语料增强为 TTS 质量的 IndicVoices-R。管道分 6 步 [§3.2, Fig 2]:

**Step 1 - 预处理**: 仅保留 44.1kHz 采样率的 extempore + read-speech 音频(排除 8kHz 会话数据),过滤 >30s 的样本,用 ffmpeg 将 mono 上混为 stereo。[论文原文]

**Step 2 - Demixing (HTDemucs)**: IndicVoices 在自然环境录制,包含背景噪声、重叠语音、回声等。使用 HTDemucs(混合时频域源分离模型)进行音频分离和降噪。[论文原文]

**Step 3 - 去混响 (VoiceFixer)**: 去噪后仍存在高混响和数字伪影。VoiceFixer 有效减少混响且不损失可懂度。[论文原文]

**Step 4 - 语音增强 (DeepFilterNet3)**: VoiceFixer 引入的数字伪影通过 DeepFilterNet3(全频带频谱图操作)消除,同时保持语音质量。[论文原文]

**Step 5 - 过滤**:

音频过滤条件 [§3.2]:
| 指标 | 阈值 | 目的 |
|------|------|------|
| C50 | >= 30 dB | 最小混响 |
| SNR | >= 25 dB | 语音清晰度 |
| Duration | 0.2-30 s | 适当长度 |
| Pitch mean | <= 350 Hz | 自然音高 |
| Pitch std | <= 150 Hz | 音高一致性 |
| Speaking rate | <= 30 chars/s | 正常语速 |

文本过滤: 排除两级转写(verbatim vs standardized)之间 CER > 5% 的样本,确保对齐可靠。[§3.2]

**Step 6 - 后处理**: 使用 PyDub normalize 统一音量,headroom 设为 0.1 dB 避免削波。使用 verbatim 转写保留口语特征(语气词、方言表达)。[§3.2]

### 关键设计选择

1. **选择增强 ASR 数据而非网络挖掘**: [论文原文] 印度语言在 YouTube 等平台缺乏手动字幕的高质量数据,且 in-the-wild 数据包含重叠说话人、音乐、低信噪比。ASR→TTS 转化路线在英语已有成功先例(LibriTTS-R, LibriLight)。[agent 解读] 这一选择还回避了网络数据的授权问题,IndicVoices 有明确的知情同意。

2. **三阶段级联增强 (HTDemucs → VoiceFixer → DeepFilterNet3)**: [论文原文] 每个模型解决不同层面的问题 -- 源分离、去混响、伪影消除。VoiceFixer 修复了 HTDemucs 遗留的混响,但自身引入数字伪影,再由 DeepFilterNet3 清理。[agent 解读] 这种"修复-引入新问题-再修复"的级联虽然笨拙,但每一步都可独立评估和替换,工程上比端到端方案更可控。

3. **Grapheme 而非 Phoneme**: [论文原文] 由于 22 种印度语言缺乏统一的 phonemizer,选择 grapheme 作为文本表示。[agent 解读] 这是一个务实的折衷 -- 避免了音素化的工程开销,但可能影响发音准确性,尤其对书写系统复杂的语言(如梵语、乌尔都语)。

4. **仅使用 44.1kHz 子集**: [论文原文] 排除 8kHz 会话数据以保证高保真度。[agent 解读] 这意味着 IndicVoices 中最自然、最具对话性的部分(8kHz 会话录音)未被利用,牺牲了数据量和对话多样性以换取音频质量。

### 训练策略

TTS 实验使用 VoiceCraft (830M 参数,GigaSpeech 预训练) 进行 fine-tuning:

- **Vocabulary 扩展**: tokenizer 从 100 tokens 扩展到 1089 tokens (22 种语言的 grapheme 合集),padding 到 1536。新 token embedding 用旧 embedding 的均值/方差初始化高斯分布。[§5]
- **训练配置**: 8x A100 40GB, AdamW (beta1=0.99, beta2=0.999), weight decay 1e-2, Lambda LR schedule, lr=1e-5, 50K steps, 动态 batch (max 20K tokens/GPU)。[§5]
- **两组对比实验**: (1) VoiceCraft fine-tuned on IndicTTS (284h, 27 speakers) vs (2) VoiceCraft fine-tuned on IV-R (1704h, 10496 speakers),在 IV-R Benchmark 上评估 zero-shot 性能。[§5]

## 实验

| 指标 | IV-R (本文) | IndicTTS (baseline) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| N-MOS (数据集质量) | 3.38 | 4.29 | 各自数据集 | [Table 1] |
| SNR (dB) | 60.47 | 65.38 | 各自数据集 | [Table 1] |
| C50 (dB) | 53.45 | 58.78 | 各自数据集 | [Table 1] |
| N-MOS (VoiceCraft合成) | 3.64 | 3.83 | IV-R Benchmark (zero-shot) | [Table 5] |
| S-SIM (说话人相似度) | 88.18% | 78.93% | IV-R Benchmark (zero-shot) | [Table 5] |

**数据集对比 (Table 1)**:
- IV-R 在规模上远超所有对比数据集: 1704h / 10496 speakers / 22 languages (vs LibriTTS 585h / 2456 spk / 1 lang; IndicTTS 284h / 27 spk / 14 lang)
- N-MOS 3.38 低于录音室数据集 (LJSpeech 4.36, IndicTTS 4.29),但 N-MOS 分布图 [Fig 3] 显示 IV-R 有比 IndicTTS 和 LibriTTS 更多的 N-MOS>4 样本

**语言覆盖**: IV-R 首次为 9 种语言提供开源 TTS 数据: Dogri, Kashmiri, Konkani, Maithili, Nepali, Sanskrit, Santali, Sindhi, Urdu (Table 3 中标 * 的语言) [§3.4]

**多样性优势**:
- 说话人: 10,496 speakers,比此前最多的 Google-CS (261 speakers) 多 40 倍 [§3.3.2]
- 年龄/性别均衡: 男女比例接近 (5030:5466), 覆盖 18-60+ 四个年龄段 [Table 2]
- 风格: 93.25% extempore 录音(vs IndicTTS 全部为 read-speech),pitch 分布更宽(最高 350Hz vs IndicTTS 更窄),speaking rate 更分散(均值 12 vs IndicTTS 9.88 words/s)[§3.3.2, Fig 4]

**Benchmark 评估 (Table 5)**:
- VoiceCraft on IV-R 的 speaker similarity (88.18%) 显著优于 IndicTTS (78.93%),说明更大的说话人多样性确实改善了 zero-shot 泛化
- N-MOS 略低 (3.64 vs 3.83),可能反映 IV-R 数据本身的音频质量略低于录音室数据的影响

**IV-R Benchmark 设计 [§4, Table 4]**:
- 352 zero-shot + 541 few-shot (<5min) + 1324 medium-shot (<10min) + 2126 many-shot (>10min) 说话人
- 覆盖双性别 x 4 年龄段,测试集总时长 ~39.5h
- 设计原则: 最大化 zero-shot/few-shot 说话人覆盖,同时不过度损耗训练数据

## 局限性

1. **音频质量仍有差距**: N-MOS 3.38 低于录音室数据集 (LJSpeech 4.36, IndicTTS 4.29),说明增强管道未能完全弥合自然环境录音与录音室录音的质量差距。[Table 1]
2. **8kHz 会话数据未利用**: IndicVoices 的会话子集采样率仅 8kHz,被排除在外。这部分数据可能包含最丰富的对话韵律和自然口语特征。[§6]
3. **仅验证一个 TTS 模型**: 实验仅使用 VoiceCraft,未在其他架构(如 F5-TTS, CosyVoice 等现代系统)上验证数据效用。
4. **评估维度有限**: 仅报告 N-MOS 和 speaker similarity,未评估 WER/CER(可懂度)、韵律自然度、跨语言泛化等关键维度。
5. **Grapheme 局限**: 不使用 phoneme 可能导致发音歧义,尤其对书写系统与发音映射复杂的语言。[§5]
6. **增强管道缺乏消融实验**: 未逐步评估 HTDemucs / VoiceFixer / DeepFilterNet3 各自的贡献,难以判断哪个环节最关键。

## 点评

**优势**:
- 填补了印度语 TTS 的关键数据空白 -- 首次覆盖全部 22 种官方语言,且说话人数量比此前最大印度 TTS 数据集 (Google-CS, 261 speakers) 多约 40 倍
- ASR→TTS 数据转化管道思路清晰,且验证了英语增强模型的跨语言能力,为其他低资源语言提供了可复制的方法论
- Benchmark 设计考虑周到(zero/few/many-shot x gender x age),为印度 TTS 社区提供了标准化评估框架
- 全部数据和代码开源,符合可复现性要求

**不足**:
- 实验设计偏简单: 仅对比 IndicTTS vs IV-R fine-tuning,且仅用 VoiceCraft。应至少验证 2-3 个模型架构,并评估更多指标
- 缺乏对增强管道本身的系统消融。每个组件(HTDemucs/VoiceFixer/DeepFilterNet3)的独立贡献和交互效应未被量化
- 虽声称匹配录音室质量,但 N-MOS 差距明显 (3.38 vs 4.29),这一 claim 更多由分布图 [Fig 3] 支撑,解读上有选择性
- 论文标题和摘要强调"matching quality",但 Table 1 的平均值明显低于对比数据集,表述不够严谨

**与 Emilia 对比** [agent 解读]: Emilia (101K h, 6 语言) 和 IV-R (1.7K h, 22 语言) 走不同路线 -- Emilia 从 in-the-wild 视频采集,规模巨大但语言集中在中英; IV-R 增强已有 ASR 数据,规模较小但语言覆盖广。两者的预处理 pipeline 也不同: Emilia 用 UVR-MDX-Net + pyannote + WhisperX; IV-R 用 HTDemucs + VoiceFixer + DeepFilterNet3。IV-R 的优势在于获得了原始说话人的知情同意,伦理合规性更强。

## 可复用的 idea

1. **ASR→TTS 数据转化管道**: 三阶段级联增强 (source separation → dereverberation → enhancement → filtering) 可直接迁移到其他低资源语言。关键是验证了英语预训练增强模型的跨语言能力。
2. **多维度过滤标准**: C50/SNR/pitch/speaking rate/CER 的组合阈值过滤策略,比单纯依赖 DNSMOS 更细粒度。
3. **Benchmark 设计方法**: zero/few/many-shot x gender x age 的交叉设计,可作为其他低资源语言 TTS benchmark 的模板。
4. **Vocabulary 扩展策略**: 新 token embedding 用旧 embedding 统计量初始化 ($\mu_{old}$, $\sigma_{old}^2$),简单但有效的迁移方法。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 管道每步有 WHY 解释,设计选择有对比 |
> | 可信赖 | pass | 关键数字与 Table 1/5 交叉验证一致 |
> | 可区分 | pass | 来源标注覆盖率 >90%, 已修正 overclaim |
> | 可定位 | pass | KB 背景谱系定位清晰 (LibriTTS-R vs Emilia 路线) |
> | 不污染 | pass | 反向更新为 append 操作,无 factual error 风险 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1) — 已当场修正
> 详见 `_review/IndicVoices-R-review.yml`
