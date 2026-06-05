---
type: paper
tier: deep
title: "MOSS-TTS Technical Report"
arxiv_id: "2603.18090"
source: "Sources/MOSS-TTS.pdf"
authors: [Yitian Gong, Botian Jiang, Yiwei Zhao, Yucheng Yuan, Kuangwei Chen, Yaozhou Jiang, Cheng Chang, Dong Hong, Mingshu Chen, Ruixiao Li, Yiyang Zhang, Yang Gao, Hanfu Chen, Ke Chen, Songlin Wang, Xiaogui Yang]
year: 2026
venue: "arXiv"
tags: [TTS, autoregressive, discrete-token, audio-codec, foundation-model, voice-cloning, RVQ, Transformer-tokenizer, long-form-generation, duration-control]
concepts: ["[[ResidualVectorQuantization]]", "[[CodecLanguageModel]]", "[[SemanticvsAcousticTokens]]", "[[QuantizerDropout]]", "[[LLM-basedTTS]]", "[[AudioTokenizerTaxonomy]]", "[[CodecTrainingObjectives]]"]
models: ["[[SoundStream]]", "[[EnCodec]]", "[[CosyVoice3]]", "[[Whisper]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[CV3-Eval]]", "[[AudioSet]]", "[[MUSDB]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: MOSS-TTS 属于 [[LLM-basedTTS]] 的 codec language model 路线 ([[CodecLanguageModel]]),即直接在 neural audio codec 的离散 token 上做 next-token prediction 生成语音。在这一谱系中,它的核心区分点在于 tokenizer 设计: MOSS-Audio-Tokenizer 采用纯 Transformer 架构 (非 CNN/CNN+RNN),端到端联合训练语义和声学,无需外部 SSL encoder (HuBERT/Whisper) 蒸馏。这在 [[AudioTokenizerTaxonomy]] [待确认] 的五轴分类中属于: Transformer 架构 + RVQ 量化 + 联合训练 + 多域 + 流式,与主流 CNN-based codec (SoundStream/EnCodec/DAC) 和 CNN+T hybrid (Mimi) 均有区别。

**已有认知**:
- [[ResidualVectorQuantization]]: MOSS-TTS 使用 32 层 RVQ,12.5 fps,与 Mimi 帧率相同但码本数远多。KB 已记录 DAC 的 factorized codes + L2-norm 改进,MOSS-TTS 亦采用 (dim 8, L2-norm),配合 [[QuantizerDropout]] (p=1.0) 实现可变比特率。p=1.0 比 DAC 的 p=0.5 更激进。
- [[SemanticvsAcousticTokens]]: MOSS-TTS 走"统一 semantic-acoustic"路线,但实现方式与 SpeechTokenizer (RVQ 第一层蒸馏 HuBERT) / Mimi (单 VQ 语义 + RVQ 声学) 不同 — 它用 0.5B decoder-only LLM 做 audio-to-text 监督 (ASR/captioning),让 RVQ tokens 自然携带语义信息,无需外部 SSL teacher。
- [[CodecLanguageModel]] [待确认]: MOSS-TTS 提出了两种 token pattern 对比 — Delay Pattern (single backbone, 源自 MusicGen) 和 Local Transformer (hierarchical, 源自 Moshi RQ-Transformer)。KB 已记录 delay pattern 和 RQ-Transformer 作为多层 RVQ 建模的关键策略。

**创新判断**: 相对 KB 已有知识,MOSS-TTS 的主要贡献在于: (1) 纯 Transformer tokenizer 架构 (1.6B params),与 CNN-based 主流不同; (2) 在同一 tokenizer + 数据下系统对比 Delay Pattern vs Local Transformer 两种 AR 策略; (3) 4 阶段训练 curriculum 的详细工程经验; (4) 小时级超长文本合成的实验数据。

> 检索命中: [[ResidualVectorQuantization]]✓, [[SemanticvsAcousticTokens]]✓, [[QuantizerDropout]]✓, [[LLM-basedTTS]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[AudioTokenizerTaxonomy]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用纯 Transformer tokenizer (1.6B, 12.5 fps, 32-layer RVQ) + 大规模 AR 预训练 (8B Delay / 1.7B Local-Transformer),在 zero-shot voice cloning、duration control 和小时级长文本生成上均取得竞争力
> - **路线**: Text + Optional Prompt → Interleaved Text/Audio Tokens → Single AR Backbone (Delay Pattern: 33 heads) or Backbone + Local AR (Local Transformer) → 32-layer RVQ Tokens → MOSS-Audio-Tokenizer Decoder → 24kHz Waveform
> - **指标**: Seed-TTS-eval ZH SIM 79.62% / EN SIM 73.28% (Local-Transformer, Continuation) [Table 3]; Duration control AbsErr Mean 0.7% [Table 5]; Ultra-long 中文 CER 1.86% @ 10000+ tokens (Continuation) [Table 6]
> - **可借鉴**: (1) Tokenizer 内嵌 LLM semantic head 做 audio-to-text 监督,无需外部 SSL teacher; (2) 4 阶段 WSD curriculum: 先纯 TTS → 重采样 clone 数据 → 恢复均衡 + LR decay → 长上下文扩展; (3) Continuation 模式 (把 prompt 当 speech prefix 续写) 比显式 Clone 模式 SIM 更高
> - **局限**: 开源但部分数据未开放; 超长英文 (50000+ tokens ≈ 67 min) WER 劣化严重 (29.52%); speaker drift 随时间累积是主要瓶颈; 无主观 MOS 评估

## 核心问题

本文要回答的核心问题: **"离散 token + AR 建模 + 大规模预训练" 这条最简洁的路线,能否在不引入多阶段 cascade / 外部 semantic teacher / flow/diffusion 后处理的情况下,达到 foundation-model 级别的 TTS 质量和可控性?**

论文的立场是明确的: 足够好的 tokenizer 把语音生成转化为单一的 token prediction 问题,从而可以像 LLM 一样扩展数据、算力和下游能力,不需要不断扩展模型栈 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MOSS-TTS 由三个核心组件构成 [§1]:

1. **MOSS-Audio-Tokenizer** (1.6B): 纯 causal Transformer tokenizer,24kHz → 12.5 fps,32 层 RVQ,0.125-4 kbps 可变比特率
2. **AR Generator** (两种架构):
   - MOSS-TTS (Delay Pattern, 8B): 单 Transformer backbone + 33 个 prediction heads
   - MOSS-TTS-Local-Transformer (1.7B): backbone + 轻量 local AR module
3. **大规模数据 pipeline**: 百万小时级多语言预训练数据

### 关键设计选择

#### 1. 为什么用纯 Transformer tokenizer 而非 CNN?

论文的核心论点是: 现有 codec (SoundStream/EnCodec/DAC) 的 CNN/CNN+RNN 架构引入了特定的 inductive bias 和架构约束,阻碍了与模型容量、数据量、量化层数的无缝扩展 [论文原文, §3.1]。MOSS-Audio-Tokenizer 采用极简 causal Transformer 设计,68 层 encoder + 68 层 decoder,刻意不引入特化 CNN 结构,使其"remarkably simple to implement and highly efficient to scale up" [论文原文, §3.1]。

**架构细节** [§3.2]:
- Encoder: 4 阶段 (hidden dim 768/768/768/1280, 各 12/12/12/32 层),用 patchify 操作逐步降采样 (patch sizes 240, 2, 2, 2),最终从 24kHz 降到 12.5 fps
- Decoder: 镜像 encoder 的 causal Transformer
- 每个阶段 FFN dim = 4x hidden dim,RoPE 位置编码,10 秒滑动窗口注意力 [§3.2]
- Encoder + Decoder 共 ~1.6B 参数,从零训练

[agent 解读] 这是一个大胆的设计赌注: 放弃 CNN 的局部感受野归纳偏置,转而依赖大规模数据 + 大模型容量来学习音频的局部结构。1.6B 参数远超主流 codec (EnCodec ~14M, DAC ~70M),代价是推理成本更高,但收益是表征能力的天花板更高。

#### 2. 如何在不依赖外部 SSL encoder 的情况下实现 semantic-acoustic 统一?

MOSS-Audio-Tokenizer 附加一个 0.5B decoder-only causal LM 作为 semantic head [§3.2-3.3]。该 LM 接收 RVQ 量化后的 hidden states,自回归预测文本 token。监督任务包括 ASR、多说话人 ASR 和 audio captioning [§3.3]。

语义损失 [§3.3, Eq 1]:
```
L_sem = -sum_t log p_LLM(s_t | T, q, s_{<t})
```
其中 q 是量化表征,T 是任务标签,s 是目标文本。

[论文原文] "Unlike approaches that depend on external pretrained audio encoders or multi-stage distillation, MOSS-Audio-Tokenizer is trained end-to-end to jointly optimize acoustic reconstruction and semantic alignment" [§1]。

[agent 解读] 与 SpeechTokenizer (蒸馏 HuBERT 到 RVQ 第一层) 和 Mimi (单 VQ 语义 + RVQ 声学) 的本质区别是: MOSS 不指定哪一层 RVQ 负责语义,而是通过全局 semantic loss 让所有层的联合表征自然携带语义信息。这避免了"第一层必须 = 语义"的硬约束,但也意味着 token 的语义-声学分离度较低。

#### 3. Delay Pattern vs Local Transformer: 设计 trade-off

两种架构在同一 tokenizer + 数据下对比,隔离 token modeling pattern 的效果 [§4]:

**Delay Pattern (MOSS-TTS, 8B)** [§4.1]:
- 32 层 RVQ 的第 j 层向前延迟 j-1 帧,使不同层在不同时间步预测
- 单 backbone 的 hidden state x_t 通过 33 个轻量 head 直接投影出各层预测
- 输入是所有层 embedding 的求和: h_t = sum_j Emb_j(ã_{j,t}) [Eq 11]
- Head-wise 加权 CE loss,前几层权重更高: λ = (1, 3, 3, 3, 2, 2, 2, 1,...,1) [Eq 9]

[论文原文] "This simplicity is one of the main reasons it is easier to implement, scale, and deploy" [§4.1]。适合长上下文和 control-oriented 部署。

**Local Transformer (MOSS-TTS-Local-Transformer, 1.7B)** [§4.2]:
- Backbone 产生每步一个 global latent,轻量 Local Transformer 自回归展开为 N_q+1 个 token
- Local Transformer 输入: 第 0 步用 backbone hidden state x_t,后续步用前一个 channel 的 embedding [Eq 14]
- 无 temporal shift,但每帧内部有长度 N_q+1 的 AR loop

[论文原文] "it can start emitting audio earlier because it does not need to wait for delayed offsets to materialize the first frame" [§4.2]。主要优势是更高建模效率和更强 speaker preservation,而非架构简洁。

[agent 解读] 这实际上是 MusicGen delay pattern 和 Moshi RQ-Transformer 在同一 tokenizer 下的公平对比。1.7B Local-Transformer 在 SIM 上超过 8B Delay,说明 frame-local AR 的归纳偏置比模型规模对 speaker preservation 更重要。但 Delay 的优势在工程侧: 单 backbone → 更容易做 KV-cache 优化、流式推理、长序列推理。

#### 4. 训练策略

**Tokenizer 训练** [§3.3]:
- 百万小时多域数据 (speech + music + sound effects)
- 总损失 [Eq 8]: L_G = 20·L_sem + 15·L_rec + 0.25·L_cmt + 1.0·L_code + 1.0·L_adv + 2.0·L_feat
- 两阶段: 非对抗预训练 520k steps (batch 1536) → 对抗微调 500k steps (batch 768)
- 使用 multi-period discriminator + complex STFT discriminator
- Factorized VQ (dim 8) + L2-normalized codes, quantizer dropout p=1.0

**Generator 4 阶段 curriculum** [§5.2, Table 1]:

| Phase | LR | 数据 | Max Seq | 目的 |
|---|---|---|---|---|
| P1 | warmup→2e-4 | D_basic only | 32k | 基础 text-speech 对齐 |
| P2 | 2e-4 hold | 全数据, 强上采样 D_clone | 32k | Voice cloning 能力扩展 |
| P3 | 2e-4→2e-6 decay | 全数据, 恢复均衡比例 | 32k | 质量巩固 + 分布再平衡 |
| P4 | 2e-6 hold | 全数据, 重上采样长文本 | 64k | 长上下文扩展 |

[论文原文] P2 刻意过度上采样 clone 数据的原因: "prompt-conditioned timbre transfer is both harder and more fragile than ordinary text-to-speech, and if it is introduced too weakly it tends to remain a tail capability" [§5.2]。P3 恢复均衡是因为"oversampling timbre-cloning data for too long biases the model toward prompt copying" [§5.2]。P4 推迟长上下文的原因: "Training with a very long window from the beginning is significantly less efficient" [§5.2]。

### 训练策略

**数据 pipeline** [§5.1] 分三阶段:
1. **预处理** [§5.1.1]: MossFormer2-SE-48K 降噪 → FLAC 格式统一 → RMS 音量归一化 (target -20 dBFS, ±3 dB clamp) → DiariZen 说话人分割 + 合并 (同说话人相邻段合并,上限 1h)
2. **过滤** [§5.1.2]: ASR 转写 (MOSS-Transcribe-Diarize) → 规则过滤 (空/重复/非语音) → LLM 精修 (诊断 + 清洗) → 单说话人验证 → 联合过滤 (DNSMOS > 2.8, Meta PQ > 6.5, 语言一致性, 时长-文本一致性)
3. **数据合成** [§5.1.3]:
   - Timbre-cloning pairs: 同说话人不同段,随机裁剪 ≤30s,WavLM-Large cosine similarity 选最佳 prompt
   - Input robustness: 标点噪声/空格伪影/标点丢弃/脏字符注入
   - Phonetic input: 拼音 (Chinese) / IPA (English),支持 partial 和 full replace
   - Short-form data: 单字/单词补充

**Duration control 训练** [§5.1.3 末]: 每个样本序列化为两个变体 — duration-conditioned (prompt 包含目标 token 数) 和 free-duration (prompt=None),在预训练全程均存在。

## 实验

### Audio Tokenizer [Table 2, Fig 5]

| 指标 | MOSS-AT@1000bps | XY-Tokenizer@1000bps | Mimi@1100bps | 数据集 | 出处 |
|---|---|---|---|---|---|
| SIM (EN/ZH) | 0.88/0.81 | 0.85/0.79 | 0.74/0.59 | LibriSpeech/AISHELL-2 | [Table 2] |
| STOI (EN/ZH) | 0.94/0.91 | 0.92/0.87 | 0.91/0.85 | LibriSpeech/AISHELL-2 | [Table 2] |
| PESQ-WB (EN/ZH) | 2.87/2.43 | 2.50/2.12 | 2.25/1.78 | LibriSpeech/AISHELL-2 | [Table 2] |

MOSS-Audio-Tokenizer 在各比特率段 (750-4000 bps) 一致优于开源 baseline 的语音重建指标 [Table 2]。音频/音乐重建维持竞争力 [Table 2]。

### Voice Cloning [Table 3]

| 模型 | Mode | Params | EN WER↓ | EN SIM↑ | ZH CER↓ | ZH SIM↑ | 出处 |
|---|---|---|---|---|---|---|---|
| MOSS-TTS | Clone | 8B | 1.92 | 69.31 | 1.46 | 76.21 | [Table 3] |
| MOSS-TTS | Continuation | 8B | 1.84 | 70.86 | 1.37 | 76.98 | [Table 3] |
| MOSS-TTS-LT | Clone | 1.7B | 1.87 | 71.74 | 1.33 | 77.24 | [Table 3] |
| MOSS-TTS-LT | Continuation | 1.7B | 1.93 | 73.28 | 1.44 | 79.62 | [Table 3] |
| Qwen3-TTS | — | 1.7B | 1.50 | 71.45 | 1.33 | 76.72 | [Table 3] |
| CosyVoice3 (open) | — | 0.5B | 2.02 | 71.80 | 1.16 | 78.00 | [Table 3] |
| CosyVoice3 (closed) | — | 1.5B | 2.22 | 72.00 | 1.12 | 78.10 | [Table 3] |
| Seed-TTS | — | — | 2.25 | 76.20 | 1.12 | 79.60 | [Table 3] |

MOSS-TTS-LT Continuation 在开源模型中 ZH SIM 最高 (79.62%),EN SIM 73.28% 亦具竞争力 [Table 3]。但 Seed-TTS (closed) 仍在 EN SIM (76.20%) 上领先。

### Duration Control [Table 5]

| 语言 | Bucket | AbsErr Mean(%)↓ | AbsErr P50(%)↓ | RMSE(%)↓ | 出处 |
|---|---|---|---|---|---|
| zh | overall | 0.712 | 0.284 | 1.141 | [Table 5] |
| en | overall | 0.723 | 0.288 | 1.160 | [Table 5] |
| zh | 10m-30m | 0.678 | 0.061 | 1.228 | [Table 5] |

仅通过预训练实现,无需额外 duration-control 微调 [§6.4]。

### Ultra-Long Generation [Table 6, Fig 6]

| 语言 | Bucket | Clone CER/WER↓ | Continuation CER/WER↓ | Clone SIM↑ | Cont. SIM↑ | 出处 |
|---|---|---|---|---|---|---|
| zh | 10000+ | 3.41 | 1.86 | 60.1 | 63.0 | [Table 6] |
| en | 50000+ | 17.49 | 29.52 | 44.4 | 51.2 | [Table 6] |
| en | 2500-12500 | 3.75 | 4.05 | 60.3 | 60.0 | [Table 6] |

主要发现: Continuation 在中文长文本上显著优于 Clone (CER 1.86% vs 3.41% @ 10000+) [Table 6]。SIM 随时间累积下降是主要瓶颈而非词汇错误 [§6.5, Fig 6]。英文超长段 (50000+ ≈ 67min) 严重劣化。

### Pronunciation Control [Table 7]

| 语言 | Setting | Span CER/WER↓ | 出处 |
|---|---|---|---|
| zh | partial-replace | 1.00 | [Table 7] |
| zh | full-replace | 1.65 | [Table 7] |
| en | partial-replace | 4.32 | [Table 7] |
| en | full-replace | 5.84 | [Table 7] |

## 局限性

1. **超长英文劣化严重**: 50000+ tokens (约 67 分钟) WER 达 29.52% (Continuation),远超可用范围 [Table 6]
2. **Speaker drift 是核心瓶颈**: Fig 6 明确显示 SIM 随时间单调递减,英文比中文更严重,50000+ bucket 在几分钟后即与短 bucket 分离 [§6.5]
3. **无主观评估**: 全文无 MOS/CMOS,仅依赖客观指标。论文对 WER<2% 后残余差异的可解读性表示谨慎 [§6.2],但缺少 naturalness/expressiveness 评估
4. **tokenizer 推理成本高**: 1.6B Transformer tokenizer 远大于主流 codec (EnCodec ~14M),streaming 推理的实际 RTF 未报告
5. **训练数据未完全开放**: "millions of hours" 的多数是内部数据,包括 proprietary ASR 模型 (MOSS-Transcribe-Diarize),可复现性受限
6. **多语言覆盖不均**: CV3-Eval 上 ja/ko 的 CER 明显高于 zh/en/de/es,部分低资源语言的效果存疑 [Table 4]
7. **Delay vs Local Transformer 对比条件不完全匹配**: 8B vs 1.7B,虽然论文论证了 Local-Transformer 在小规模下更高效,但相同参数下的对比缺失

## 点评

**优势**:
- **清晰的工程叙事**: 不同于许多"堆料"式技术报告,MOSS-TTS 围绕"离散 token + AR + 大规模预训练"这一最简路线展开,并系统对比两种 AR 策略。这种"少做但做深"的方法论使结论更可信
- **纯 Transformer tokenizer 的验证**: 首次在 1.6B 规模验证纯 Transformer 架构在 audio codec 领域的可行性和优势,打破了 CNN/CNN+RNN 作为 audio codec encoder-decoder 的默认选择
- **Continuation 模式的发现**: 将 prompt audio 作为 speech prefix 续写而非显式 clone conditioning,在 SIM 上一致优于 Clone 模式。这是一个简单但有价值的工程 insight
- **4 阶段 curriculum 的详细工程分析**: 特别是 P2 过采样 clone 数据 → P3 恢复均衡的策略,揭示了 voice cloning 作为"fragile tail capability"需要前期重点投入的实践经验

**不足**:
- **缺少消融实验**: semantic head (0.5B LLM) 的贡献、patchify 策略的影响、quantizer dropout p=1.0 vs p=0.5 的对比均未报告
- **tokenizer 与 generator 分离评估**: Table 2 证明 tokenizer 重建质量强,但 tokenizer 最优 ≠ 下游生成最优 (Mousavi et al. 2025 survey 已指出)。tokenizer 对 TTS 生成质量的 end-to-end 贡献度不明确
- **与非 AR 路线 (CFM/Diffusion) 缺乏对比**: CosyVoice 系列是 LLM + CFM hybrid,F5-TTS 是 flow matching,论文仅比 SIM/WER 数字但未讨论为什么纯 AR 不需要 post-hoc flow 阶段

## 可复用的 idea

1. **Tokenizer 内嵌 LLM semantic head**: 用 decoder-only LM 做 audio-to-text 监督实现 semantic-acoustic 联合,不依赖外部 SSL encoder。可以迁移到任何 RVQ-based codec 的训练中,特别是当不想引入 HuBERT/Whisper 依赖时
2. **Continuation 替代 Clone**: 在 voice cloning 场景中,把 prompt audio 拼到 assistant speech prefix + ASR transcript 拼到 text prefix,让模型做 speech continuation 而非显式 clone。实现简单,效果好于显式 reference conditioning
3. **4 阶段 curriculum 设计模式**: P1 (基础对齐) → P2 (重采样难任务) → P3 (恢复均衡 + LR decay) → P4 (上下文扩展)。"先让难能力成为 first-class,再恢复分布平衡"的策略对任何多任务预训练都适用
4. **Duration control 通过双变体训练**: 同一数据序列化为 duration-conditioned 和 free-duration 两个变体,全程预训练,无需专门 fine-tune 阶段。原理是让 duration 成为一个 optional conditioning signal
5. **Head-wise weighted CE loss**: 对 33 个 prediction head 使用不等权重 λ = (1,3,3,3,2,2,2,1,...),前几层 coarse RVQ 权重更高。可借鉴用于任何 multi-codebook AR 系统
