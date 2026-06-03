---
type: paper
tier: deep
title: "VoXtream: Full-Stream Text-to-Speech with Extremely Low Latency"
arxiv_id: "2509.15969"
source: "Sources/VoXtream.pdf"
authors: [Nikita Torgashov, Gustav Eje Henter, Gabriel Skantze]
year: 2025
venue: "IEEE ICASSP 2026"
tags: [TTS, streaming, zero-shot, autoregressive, low-latency, full-stream]
concepts: ["[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Speaker Embedding]]", "[[LLM-based TTS]]", "[[Duration Predictor]]", "[[Phoneme Representation]]", "[[Speech-Text Alignment]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Speaker Embedding]], [[LLM-based TTS]], [[Zero-shot Speech Synthesis]], [[模型库/CosyVoice 2|CosyVoice 2]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: VoXtream 属于 LLM-based TTS 的自回归路线,但与 VALL-E/CosyVoice 等主流方案有根本分歧 -- 它不使用 text LLM 做骨干,而是设计了三层专用 transformer (Phoneme/Temporal/Depth),且核心创新在于输入侧流式处理(full-stream)。在 [[LLM-based TTS]] 概念页中,streaming 方向主要由 CosyVoice 2 和 XTTS 代表,但它们都是 output-streaming(全文就绪后流式输出音频)或依赖 NAR flow-matching 解码器引入延迟。VoXtream 是首个在 input-side 实现增量处理的全 AR 系统。
>
> **已有认知**: [[Speech Tokenizer]] 记录了 Mimi codec 作为 mixed tokenizer 的设计(单 VQ 语义 + 额外 RVQ 声学,12.5Hz),VoXtream 直接使用 Mimi 的 12 层 codebook。[[Semantic vs Acoustic Tokens]] 的层级建模方案中,VoXtream 的 Temporal Transformer 预测 semantic tokens(第1层 codebook),Depth Transformer 预测 acoustic tokens(第2-12层),是典型的 coarse-to-fine 策略。[[Speaker Embedding]] 页提到 ECAPA-TDNN 等 encoder,VoXtream 使用 ReDimNet speaker encoder (100K+ identities 预训练),属于该页未记录的新型 speaker encoder。[[模型库/CosyVoice 2|CosyVoice 2]] 是 VoXtream 的关键 baseline,其 full-stream 模式下 FPL=1643ms,远高于 VoXtream 的 102ms。
>
> **创新判断**: VoXtream 的核心创新 -- incremental phoneme transformer + 受限 look-ahead 的 full-stream 设计 -- 在 KB 中没有对应概念页。将输入流式化(word-by-word 接收并立即开始生成)是对现有 streaming TTS 方案的实质性推进,而非增量改进。
>
> 检索命中: [[Speech Tokenizer]]✓, [[Semantic vs Acoustic Tokens]]✓, [[Speaker Embedding]]✓, [[LLM-based TTS]]✓, [[Zero-shot Speech Synthesis]]✓, [[模型库/CosyVoice 2|CosyVoice 2]]✓ | 过滤: [[Duration Predictor]](pending-review), [[Speech-Text Alignment]](pending-review), [[Phoneme Representation]](pending-review), [[Streaming Spoken Dialogue]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 首个 full-stream(输入侧+输出侧同时流式) zero-shot TTS,通过 incremental phoneme transformer + 有限 look-ahead 实现 102ms 首包延迟,以 9k 小时数据比肩大规模非流式系统
> - **路线**: Phoneme stream → Incremental Phoneme Transformer (有限 look-ahead) → Temporal Transformer (semantic token + duration token) → Depth Transformer (acoustic tokens, 冻结 CSM 权重) + ReDimNet speaker embedding → Mimi decoder → waveform
> - **指标**: FPL 102ms (torch.compile, A100); WER 3.82%/3.09% (SEED-en/LibriSpeech), SPK-SIM 0.529/0.461, UTMOS 3.88/4.08; Full-stream naturalness preference 57% vs CosyVoice2 31% [Table 1-3]
> - **可借鉴**: (1) Incremental phoneme encoding with limited look-ahead: 不等全文,每来一个词就开始编码,过去状态随新上下文更新; (2) 冻结大规模预训练的 Depth Transformer (CSM-DT) 做知识迁移,用外部 speaker encoder 补偿 speaker similarity; (3) Duration token = shift flag + phoneme count 的联合编码方式,实现单一 classification head 同时预测语义和时长
> - **局限**: SPK-SIM 明显低于 CosyVoice 2 等使用 NAR flow-matching decoder 的系统 (0.529 vs 0.656 on SEED-en); 仅训练于 9k 小时英语数据,无多语言支持; 仅 441M 参数,模型较小; 无显式 speaking-rate control

## 核心问题

VoXtream 要解决的核心问题是: **如何让 TTS 系统在输入文本逐词到达时就立即开始说话,而非等到全部文本就绪?** 这在 voice assistant/同声传译/conversational AI 场景中至关重要。现有方案的问题:
- **非流式系统** (VALL-E, CosyVoice, Spark-TTS): 需要全部文本才能开始,完全不适用于实时场景 [§1]
- **输出流式系统** (XTTS, CosyVoice2:Out, FireRedTTS-1S): 仅在解码侧分块输出,但仍需全部输入文本 [§1]
- **Full-stream 尝试** (CosyVoice2:Full, IST-LM): 用 interleaved text-speech 实现输入流式,但依赖 NAR flow-matching decoder 做 chunk 处理,FPL 仍高达 1643ms [§1, Table 3]
- **SpeakStream**: 低 FPL 但未探索 zero-shot [§1]
- **SyncSpeech**: 需要积累 token 后才能解码 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VoXtream 由三个级联 transformer 组成 [§2, Fig 1]:

1. **Phoneme Transformer (PT)**: decoder-only transformer,6 层 8 头。接收逐词到达的 phoneme 序列,增量编码。输入随每个新词增长,允许最多 10 phoneme 的 look-ahead,但不等待 look-ahead 累积就立即开始生成 [论文原文]。g2pE 负责 word-level 的 grapheme-to-phoneme 转换。

2. **Temporal Transformer (TT)**: AR transformer,12 层 16 头,embedding 1024,FFN 4096。以音频 token 和对应 phoneme 序列为条件。核心输出: (a) Mimi 第 1 层 codebook 的 semantic token; (b) duration token。两者通过单一 classification head 从联合分布中采样 [§2]。

3. **Depth Transformer (DT)**: 4 层 8 头,FFN 8192。以 TT 输出 embedding + semantic token + ReDimNet speaker embedding 为条件,自回归生成 Mimi 第 2-12 层 acoustic tokens。直接借用 CSM 模型的预训练权重,训练时冻结 [§3.1]。

最终,Mimi decoder 将每帧的 semantic + acoustic tokens 流式转为 80ms 语音。

### 关键设计选择

**1. 为什么用 incremental phoneme encoding 而非 interleaved text-speech?**

现有 full-stream 方案(如 CosyVoice 2, IST-LM)将文本和语音 token 交错排列在同一序列中,按固定比例交替 [论文原文]。但这要求 NAR decoder 积累足够 chunk 才能解码,造成高 FPL [论文原文]。VoXtream 的方案是让 PT 单独编码文本,且随新词到达增量更新过去状态。这样 TT 可以在仅有第一个词时就开始预测 audio tokens,不需等待任何最小 chunk [论文原文]。过去的 phoneme embedding 会随更多上下文的到来而被更新(即每次新词到达时 PT 重新编码整个已有序列),这使有限上下文下的韵律仍然自然 [agent 解读: 论文说 "past text states are updated as more context is available" [§1], 推测每次新词到达触发 PT 对已有序列的重计算]。

**2. 为什么用 duration token 的 shift-flag 设计?**

灵感来自 VALL-E R 的 monotonic alignment [ref 21],但不预测具体 phoneme,而是预测 duration token 编码两个信息 [§2]:
- **shift flag** (stay/go): 下一帧继续当前 phoneme 还是切换到下一个
- **phoneme count** (1 or 2): 对应慢速还是快速发音

这与 semantic token 共用同一 classification head,从联合分布采样 [论文原文]。为什么联合建模? [agent 解读: 因为某个 semantic token 的选择与 duration 是耦合的 -- 一个 phoneme 结束时和持续中的 acoustic 特征不同,联合采样避免了两者不一致的问题]。每帧最多 2 个 phoneme (12.5 Hz 帧率下每帧 80ms,对应语速较快时一帧承载两个短 phoneme) [论文原文]。

**3. 为什么冻结 CSM 的 Depth Transformer?**

DT 的任务(从 semantic token 生成 acoustic tokens)是一个相对通用的能力 [agent 解读]。CSM 在大规模数据上训练,其 DT 已学会高质量的 semantic→acoustic 映射 [§3.1]。冻结 CSM-DT 带来两个好处: (a) 知识迁移,CSM-DT 显著提升 UTMOS (3.39→3.90) 和 SPK-SIM (0.471→0.504) [Table 4]; (b) 减少训练参数,9k 小时中等规模数据不足以从头训练高质量 DT [agent 解读]。

**4. 为什么用 ReDimNet speaker encoder 而非 prompt-based?**

VoXtream 不使用 in-context prompt 的方式传递 speaker identity(不像 VALL-E 把参考音频 token 作为前缀),而是用外部 ReDimNet [ref 22] 提取 speaker embedding 注入 DT [§2]。原因: [agent 解读: full-stream 场景下输入逐词到达,很难在序列开头附加完整的参考音频 token 前缀; 外部 speaker encoder 可以独立处理参考音频,不影响输入流式 pipeline]。Ablation 显示 SPK-ENC 在 DT unfrozen 时提升 SPK-SIM 19% (0.471→0.558),即使 DT frozen 也提升 6% (0.504→0.537) [Table 4]。

### 训练策略

- 数据: Emilia (4.5k h 自发风格) + HiFiTTS-2 (4.5k h 朗读风格),共 9k 小时英语 [§3.1]
- Emilia 额外经过 diarization 去除多说话人片段 + NISQA 过滤低质量 [§3.1]
- 训练: 2x A100-80GB, batch size 128/GPU, 9 epochs [§3.1]
- 输入: 固定 20s 音频 chunk + 对应 phoneme 序列,短音频通过同说话人拼接 [§3.1]
- 优化: AdamW, 1 epoch warmup, peak lr 5e-4 [§3.1]
- 对齐: MFA (Montreal Forced Aligner) 获取 acoustic prompt 与 phoneme 的初始对齐 [§3.1]
- Loss: TT 和 DT 均为 negative log-likelihood [§2]
- Mimi 使用 12 个 codebook (latency-quality trade-off), 12.5 Hz, 24 kHz [§2, §3.1]
- Acoustic delay: 1 step delay for stability, following Mimi 的做法 [§2]

## 实验

| 指标 | VoXtream:Full | VoXtream:Out | VoXtream-NS | CosyVoice2 | CosyVoice2:Out | XTTS-v2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER (%) | 3.81 | 3.82 | 3.64 | 2.87 | 2.70 | 3.64 | SEED test-en | [Table 1] |
| SPK-SIM | 0.529 | 0.529 | 0.537 | 0.656 | 0.662 | 0.467 | SEED test-en | [Table 1] |
| UTMOS | 3.90 | 3.88 | 3.89 | 4.18 | 4.05 | 3.57 | SEED test-en | [Table 1] |
| WER (%) | 3.15 | 3.09 | 2.99 | 2.97 | 2.65 | 3.90 | LS test-clean | [Table 1] |
| SPK-SIM | 0.458 | 0.461 | 0.465 | 0.587 | 0.592 | 0.444 | LS test-clean | [Table 1] |
| UTMOS | 4.07 | 4.08 | 4.07 | 4.23 | 4.19 | 3.72 | LS test-clean | [Table 1] |
| Naturalness (MUSHRA) | 53.4 ± 2.5 | — | — | — | — | 51.9 ± 2.6 | — | [Table 1] |

**Full-stream 对比 (LibriSpeech long, >10s utterances):**

| 指标 | VoXtream:Full | CosyVoice2:Full | Human | 出处 |
| --- | --- | --- | --- | --- |
| WER (%) | 3.24 | 6.11 | 1.97 | [Table 2] |
| SPK-SIM | 0.564 | 0.685 | 0.784 | [Table 2] |
| UTMOS | 4.23 | 4.19 | 4.16 | [Table 2] |
| Naturalness Pref (%) | 57 | 31 | — | [Table 2] |

**延迟对比 (A100, FP16):**

| 模型 | FPL (ms) | RTF | 出处 |
| --- | --- | --- | --- |
| VoXtream:TC | 102 | 0.17 | [Table 3] |
| VoXtream | 171 | 1.00 | [Table 3] |
| XTTS-v2:DS | 196 | 0.26 | [Table 3] |
| XTTS-v2 | 295 | 0.37 | [Table 3] |
| CosyVoice2 | 1643 | 0.85 | [Table 3] |

**关键发现:**
1. VoXtream 在 mid-scale (9k h) 数据组中达最佳 SPK-SIM 和 UTMOS [Table 1]
2. Full-stream → output-stream → non-stream 的 WER 递增极小 (3.81 → 3.82 → 3.64 on SEED-en),说明流式引入的质量损失可控 [Table 1]
3. 在长句 full-stream 场景,VoXtream WER (3.24%) 远低于 CosyVoice2 (6.11%),自然度偏好 57% vs 31% (p < 5e-10) [Table 2]
4. FPL 102ms 是所有公开流式 TTS 中最低,比 CosyVoice2 快 16 倍 [Table 3]
5. RTF 0.17 (with torch.compile),比实时快 5 倍以上 [Table 3]

**Ablation (CSM-DT + SPK-ENC, SEED test-en):**

| CSM-DT | SPK-ENC | WER (%) | SPK-SIM | UTMOS | 出处 |
| --- | --- | --- | --- | --- | --- |
| No | No | 3.53 | 0.471 | 3.39 | [Table 4] |
| Yes | No | 3.70 | 0.504 | 3.90 | [Table 4] |
| No | Yes | 3.65 | 0.558 | 3.39 | [Table 4] |
| Yes | Yes | 3.64 | 0.537 | 3.89 | [Table 4] |

CSM-DT 对 UTMOS 贡献最大 (+0.51),SPK-ENC 对 SPK-SIM 贡献最大 (+0.087); 两者组合的 SPK-SIM (0.537) 比单独 SPK-ENC (0.558) 低,这是因为冻结 DT 限制了 speaker embedding 的利用 [agent 解读]。Baseline (无 CSM-DT, 无 SPK-ENC) WER 最低 (3.53%),说明基础架构本身的 intelligibility 已经很好 [Table 4]。

## 局限性

1. **SPK-SIM 差距明显**: VoXtream 的 SPK-SIM (0.529/0.461) 系统性低于 CosyVoice2 (0.656/0.587),主要因为 CosyVoice2 使用 NAR flow-matching decoder 可以更好地重建声学细节和音色 [Table 1] [agent 解读: 全 AR 方案在 acoustic fidelity 上天然劣于 NAR decoder]
2. **仅英语**: 训练数据 9k 小时全部为英语,无多语言支持,而 CosyVoice2 覆盖 167k 小时多语种数据 [§3.1, Table 1]
3. **依赖外部组件**: CSM-DT 和 ReDimNet 均为外部预训练组件,系统的 end-to-end 可控性受限 [§3.1]
4. **无显式语速控制**: 论文未来工作中提到计划探索 explicit speaking-rate control [§5]
5. **长文本稳定性未知**: 虽然 LibriSpeech long (avg 15s) 表现良好,但更长文本 (>30s) 的稳定性未评估 [agent 解读]
6. **评估局限**: Naturalness 评估仅 100 句,MUSHRA 评估 40 人,样本量有限 [§3.2]

## 点评

VoXtream 在流式 TTS 领域做出了实质性推进。它的核心贡献不是某个单一技巧,而是一套完整的 full-stream 架构设计: incremental phoneme encoding + limited look-ahead + depth transformer 知识迁移。这套设计使得系统可以在第一个词到达时就开始说话 (102ms FPL),这在 voice assistant 场景中具有重要实用价值。

从技术选择看,VoXtream 做了一个清晰的 trade-off: 用 SPK-SIM 换取极低延迟。全 AR pipeline 避免了 NAR flow-matching decoder 的 chunk 积累延迟,但也放弃了其在 acoustic fidelity 上的优势。在 mid-scale 数据条件下 (9k h),VoXtream 的质量已经非常有竞争力 (UTMOS 3.88/4.08 与训练数据量大 10-50 倍的系统相当)。

CSM-DT 冻结迁移是一个聪明的工程选择,它本质上是利用大模型的知识弥补中等规模数据的不足。但这也引入了对 CSM 的依赖,且冻结 DT 限制了 speaker embedding 的充分利用 (ablation 中 SPK-ENC 单独使用的 SPK-SIM 反而更高)。

与 KB 中的系统对比: CosyVoice 2 的 full-stream 模式 FPL 高达 1643ms (因 flow-matching chunk 积累),VoXtream 直接快了 16 倍。但 CosyVoice 2 的 SPK-SIM 显著更高,说明在 "像谁" 这个维度上 NAR decoder 的优势难以被纯 AR 方案取代。这启示未来可能需要 AR (低延迟) + 轻量 NAR (声学增强) 的混合方案来同时兼顾延迟和音色保真。

## 可复用的 idea

1. **Incremental phoneme encoding with state update**: 文本编码器不等全文,每来一个词就增量处理,且随新上下文更新已编码状态。这个 idea 可泛化到任何需要 streaming input 的 seq2seq 任务。
2. **Duration token = shift flag + phoneme count 的联合编码**: 将时长信息压缩为离散 token 与 semantic token 联合预测,避免独立 duration predictor 的误差传播。
3. **冻结大模型的 Depth Transformer 做知识迁移**: 对于数据量有限的场景,借用大规模预训练模型的子模块 (冻结权重) 可以显著提升音质。
4. **External speaker encoder 而非 in-context prompt**: 在 full-stream 场景下,外部 speaker encoder 比 prompt token 前缀更灵活,不占序列长度,可随时提供 speaker conditioning。
5. **Limited look-ahead 而非 forced delay**: 允许最多 N 个 phoneme 的 look-ahead 但不等待积累,第一个词到达就立即开始,兼顾韵律 (look-ahead 改善) 和延迟 (不 block)。

> [!review] 审阅待完成
> 本笔记为 draft 状态,待审阅。
