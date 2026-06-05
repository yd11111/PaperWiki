---
type: paper
tier: deep
title: "LM-SPT: LM-Aligned Semantic Distillation for Speech Tokenization"
arxiv_id: "2506.16738"
source: "Sources/LM-SPT.pdf"
authors: [Daejin Jo, Jeeyoung Yun, Byungseok Roh, Sungwoong Kim]
year: 2025
venue: "arXiv preprint"
tags: [speech-tokenizer, semantic-distillation, RVQ, speech-LM, low-frame-rate, codec-design]
concepts: ["[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[ResidualVectorQuantization]]", "[[TokenRateandBitrateTrade-offs]]", "[[Self-SupervisedSpeechRepresentation]]", "[[SpeechLanguageModel]]", "[[CodecTrainingObjectives]]"]
models: ["[[模型库/EnCodec|EnCodec]]", "[[模型库/HuBERT|HuBERT]]", "[[模型库/WavLM|WavLM]]", "[[模型库/Whisper|Whisper]]"]
tasks: ["[[NeuralAudioCompression]]"]
datasets: ["LibriSpeech", "LJSpeech", "GigaSpeech", "VCTK"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页: [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[ResidualVectorQuantization]]✓, [[SpeechLanguageModel]]✓, [[TokenRateandBitrateTrade-offs]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: LM-SPT 属于 mixed objective tokenizer 路线 (SpeechTokenizer → Mimi → LM-SPT),即在单一 codec 内同时编码语义和声学信息。KB 中记录了该路线的两个先驱:
- SpeechTokenizer: RVQ 第一层蒸馏 HuBERT 语义,后续层量化声学残差
- Mimi: 单 VQ 模块提取 WavLM 语义 + 额外 RVQ 声学,Split RVQ 结构

**已有认知对比**: 
1. KB [[SpeechTokenizer]] 记录了"SSL teacher 实际编码的是 phonetic 而非 semantic 信息"这一认知 (Choi et al. 2024),LM-SPT 正是从这一局限出发,改用 ASR teacher (Whisper)
2. KB [[SemanticvsAcousticTokens]] 的 Survey 发现"没有任何 tokenizer 在 semantic-acoustic alignment 上取得实质性成果",LM-SPT 的 reconstruction-driven distillation 可视为对此的一种回应
3. KB [[TokenRateandBitrateTrade-offs]] 记录了"低 token rate 对 LM 建模有巨大优势",LM-SPT 支持 25/12.5/6.25 Hz 三档超低帧率,正对应此趋势

**创新判断**: 相对于 KB 中已有的 mixed tokenizer (SpeechTokenizer, Mimi),LM-SPT 的核心新意在于:(a) 将 semantic distillation 从 feature-level 改为 reconstruction-driven,绕过帧率对齐问题;(b) 从 SSL teacher 换为 ASR teacher (Whisper),更贴近 LM 语义;(c) dual encoder 架构显式分离 semantic 和 acoustic 编码。这些设计与 KB 中 FireRedTTS 2 的思路有相似性(也用 Whisper encoder + 声学 encoder + Vocos decoder),但 LM-SPT 的 distillation 机制不同。

> 检索命中: [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[ResidualVectorQuantization]]✓, [[SpeechLanguageModel]]✓ | 过滤: [[TokenRateandBitrateTrade-offs]](pending-review), [[Self-SupervisedSpeechRepresentation]](pending-review) | 未命中但可能相关: [[CodecTrainingObjectives]]

## 速查

> [!summary] 速查
> - **一句话**: 提出 reconstruction-driven semantic distillation — 不直接对齐 teacher-student 特征,而是用 ASR encoder (Whisper) 对比原始与语义重建波形的表征差异,使 speech token 在低帧率下更好对齐 LM 语义
> - **路线**: 语音波形 → dual encoder (semantic encoder + acoustic encoder) → Split VQ (semantic VQ + acoustic RVQ) → Vocos decoder 重建; 同时 semantic tokens → 轻量 auxiliary decoder → 重建波形 → Whisper encoder 表征 → 与原始波形 Whisper 表征 MSE loss
> - **指标**: 重建 WER 3.19% (25Hz) / 3.92% (12.5Hz) / 8.35% (6.25Hz); TTS WER 4.94% (25Hz, 0.5B LM), UTMOS 3.82; 全面超越 SpeechTokenizer 和 Mimi baseline [Table 1, 2]
> - **可借鉴**: (1) reconstruction-driven distillation 绕过帧率对齐约束,可泛化到任意 teacher-student 帧率组合; (2) 轻量 auxiliary decoder 作为 information bottleneck 迫使 semantic encoder 学习更紧凑表征; (3) dual encoder 显式分离 semantic/acoustic 编码,比 shared encoder 提升约 0.3% WER
> - **局限**: 仅在 LibriSpeech (1K h) 上训练,泛化能力有待验证; 仅测试 16kHz; 6.25Hz 帧率下 TTS WER 仍>18%,极低帧率性能待改善; 未开源代码/权重

## 核心问题

LM-SPT 要解决的核心问题是: **现有 speech tokenizer 在降低帧率时如何保持与 LM 的语义对齐?**

具体而言:
1. **SSL teacher 的语义局限**: HuBERT/WavLM 等 SSL 模型学到的是 phonetic 而非 semantic 表征 [§1, Choi et al. 2024],与 LM 的语义空间存在 gap
2. **帧率下降时的对齐困难**: 传统 feature-level distillation 要求 teacher-student 帧对齐,当 student 帧率 (如 12.5Hz) 远低于 teacher (50Hz) 时,需用 average pooling 下采样 teacher 输出,这种刚性对齐会扭曲语义结构 [§3.1]
3. **语义-声学干扰**: 共享 encoder 同时优化语义蒸馏和声学重建时,两个目标可能冲突 [§4.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

LM-SPT 采用 encoder-decoder + Split RVQ 架构,核心创新在于语义蒸馏方式和编码器设计 [§3, Fig 1b]:

1. **Dual Encoder**: 两个独立的 convolutional encoder,分别处理语义和声学
   - Semantic encoder: h_sem(x) → Q_sem (单层 VQ)
   - Acoustic encoder: h_ac(x) → Q_ac (7 层 RVQ)
   - 两个 encoder 使用相同的卷积结构 (4 个卷积块),后接共享的 8 层 Transformer bottleneck (hidden dim 512) [Appendix D]

2. **Split RVQ**: 语义 VQ (1 层) 与声学 RVQ (7 层) 并行,非级联
   - 沿袭 Mimi 的设计,但配合 dual encoder 使分离更彻底

3. **Vocos Decoder**: 12 层 backbone (hidden 768),基于 inverse STFT 重建波形,避免转置卷积上采样的 tonal artifacts [§3.2, Siuzdak 2024]

4. **Auxiliary Semantic Decoder**: 1 层轻量 Vocos decoder (G_aux),仅用于语义蒸馏训练,不共享参数 [§3.1]

### 关键设计选择

#### 1. Reconstruction-driven Semantic Distillation (核心创新)

**为什么不用 feature-level distillation?** [论文原文] 传统方法直接最小化 teacher 和 student 特征的距离,但这要求两者帧对齐。当 student 帧率远低于 teacher 时,必须用 average pooling 下采样 teacher 输出,这种刚性操作"can distort or dilute the semantic structure" [§3.1]。

**为什么选 ASR teacher (Whisper) 而非 SSL teacher?** [论文原文] SSL 模型 (HuBERT, WavLM) 的表征"trained to capture phonetic regularities rather than semantic abstraction" [§3.1, citing Choi et al. 2024, Wells et al. 2022],而 Whisper encoder 虽是 ASR 训练,"its encoder outputs are optimized for text prediction and implicitly capture high-level linguistic structures" [§3.1]。

**怎么做?** 不直接比较 teacher-student 特征,而是:
- 从 semantic tokens 重建波形: x̂_sem = G_aux(z_sem) [§3.1, Eq 1]
- 用冻结的 Whisper encoder 分别编码原始波形和重建波形
- 最小化两者表征的 MSE: L_distill = ||f_T(x) - f_T(x̂_sem)||² [§3.1, Eq 1]

这种间接监督绕过了帧对齐约束,因为 Whisper encoder 本身处理整段波形,输出与输入帧率无关 [agent 解读]。

#### 2. Decoupled Auxiliary Decoder (信息瓶颈)

**为什么不用 shared decoder?** [论文原文] 共享 decoder 同时优化重建和蒸馏会导致目标冲突;冻结 decoder 可避免冲突但"weakens the effect of distillation" [§3.1]。

**为什么用轻量 decoder?** [论文原文] 2.4M 参数的轻量 auxiliary decoder 比 47.4M 参数的大 decoder 在下游 TTS 任务上更好 (WER 4.94% vs 7.89%),因为它"functioning as an information bottleneck" [§3.1],迫使 semantic encoder 自身学习更紧凑的表征。

#### 3. Dual Encoder

**为什么分离编码器?** [论文原文] 受 disentangled representation learning 启发 [Qu et al. 2024, Ye et al. 2025],显式分离编码过程可避免共享 encoder 中语义和声学信号的干扰 [§3.2]。

**效果**: 相比 shared encoder,dual encoder 在重建 WER (3.19 vs 3.47)、UTMOS (3.75 vs 3.68)、Speaker Similarity (0.98 vs 0.98) 上均有提升 [Table 5]。dual encoder 增加约 22M 参数,但即使将 Mimi baseline 增大至 166.2M (Mimi (R) large) 仍不如 99.7M 的 LM-SPT [Table 5]。

#### 4. 理论解释: Reconstruction-driven 的 KL 优势

[论文原文] 在高斯假设下,feature-level distillation 的 KL 散度包含一个不可消除的 variance gap 项 (Eq 3),因为 MSE 训练的 student 倾向于低估 teacher variance (σ_fS < σ_fT) [§3.1]。而 reconstruction-driven distillation 在理想情况下 KL 散度可降至零 (Eq 4),因为它在同一个 encoder (Whisper) 的输出空间中比较,两者具有相同的 variance [§3.1]。

### 训练策略

- **数据**: LibriSpeech ~1000h (16kHz); 大规模版 LM-SPT (L) 使用 60K h (LibriHeavy + 10K h 韩语) [§4.1]
- **帧率配置**: 25Hz / 12.5Hz / 6.25Hz 分别训练独立模型,codebook size 随帧率降低而增大 (1024/2048/4096) [§4.1]
- **训练细节**: 20 epochs, lr=2e-4, batch 128, 随机裁剪 6s 片段, 8x A100 约 40 小时 [§4.1]
- **Loss**: 标准 SpeechTokenizer loss (重建 + 判别 + feature matching + commitment) + 语义蒸馏 loss,λ_distill=100, 其余 λ 详见 Table 9 [Appendix C]
- **Semantic teacher**: Whisper Small (50Hz) [§4.1]
- **SLM 评估**: Qwen2.5-0.5B-Instruct 和 LLaMA3.2-3B-Instruct,训练 10K steps [§4.1]

## 实验

### 重建性能 [Table 1]

| 指标 | LM-SPT (25Hz) | SpeechTokenizer (25Hz) | Mimi (R) (25Hz) | LM-SPT (12.5Hz) | Mimi (12.5Hz, official) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER ↓ | 3.19 | 3.59 | 3.28 | 3.92 | 4.08 | [Table 1] |
| UTMOS ↑ | 3.75 | 3.37 | 3.22 | 3.78 | 3.56 | [Table 1] |
| DNSMOS ↑ | 3.27 | 3.24 | 3.18 | 3.26 | 3.16 | [Table 1] |
| SS ↑ | 0.98 | 0.96 | 0.97 | 0.97 | 0.97 | [Table 1] |

### SLM 下游性能 (0.5B LM) [Table 2]

| 指标 | LM-SPT (25Hz) | SpeechTokenizer (25Hz) | Mimi (R) (25Hz) | LM-SPT (12.5Hz) | 出处 |
| --- | --- | --- | --- | --- | --- |
| STT WER ↓ | 15.34 | 13.32 | 17.92 | 18.56 | [Table 2] |
| TTS WER ↓ | 4.94 | 8.97 | 8.21 | 5.82 | [Table 2] |
| TTS UTMOS ↑ | 3.82 | 2.82 | 3.52 | 3.42 | [Table 2] |
| TTS DNSMOS ↑ | 3.35 | 3.06 | 3.18 | 3.32 | [Table 2] |

### 关键消融 [Tables 3-5]

| 消融项 | 变体 | TTS WER ↓ (0.5B) | 说明 | 出处 |
| --- | --- | --- | --- | --- |
| Distillation | feature-level (Whisper) | 13.97 | 同一 teacher,直接特征对齐 | [Table 3] |
| Distillation | reconstruction-driven (Whisper) | **4.94** | 本文方法 | [Table 3] |
| Sem. decoder | shared decoder | 8.97 | 共享导致目标冲突 | [Table 4] |
| Sem. decoder | w/o gradient | 39.55 | 冻结 decoder 蒸馏失效 | [Table 4] |
| Sem. decoder | decoupled (本文) | **4.94** | 解耦 + 轻量 | [Table 4] |
| Sem. decoder cap. | 47.4M | 7.89 | 大 decoder 瓶颈不够紧 | [Table 4] |
| Sem. decoder cap. | 2.4M (本文) | **4.94** | 小 decoder 迫使 encoder 学更好 | [Table 4] |
| Encoder | w/o dual encoder | TTS WER 未列 | 重建 WER 3.47 vs 3.19 | [Table 5] |
| Decoder | w/o Vocos | TTS WER 未列 | 重建 WER 3.12 但 UTMOS 3.53 vs 3.75 | [Table 5] |

### Held-out 数据集验证 [Tables 6-7]

- **LJSpeech 重建**: LM-SPT 在所有帧率上均达最低 WER 和最高 UTMOS [Table 6]
- **GigaSpeech SLM**: LM-SPT 在 STT 和 TTS 任务上均优于 baseline; LM-SPT (L) 进一步提升 [Table 7]

### SNMI 语义对齐分析 [Table 8, Appendix B]

- 提出 Sequence-Normalized Mutual Information (SNMI) 衡量 token 与文本的语义对齐度
- LM-SPT 在 25Hz 和 6.25Hz 上 semantic SNMI 最高 (0.753, 0.749) [Table 8]
- LM-SPT 的 acoustic/semantic SNMI ratio 在所有帧率上最低 (0.940, 1.019, 1.016),表明语义-声学分离最干净 [Table 8]

## 局限性

1. **训练数据规模有限**: 主实验仅用 LibriSpeech 1K h (英语朗读语音),泛化到更多语言/风格需更多验证 [§5]
2. **采样率限制**: 仅测试 16kHz,未验证 24kHz 等更高采样率 [§5]
3. **极低帧率瓶颈**: 6.25Hz 下 TTS WER 仍>18% (0.5B LM),压缩与保真度的 trade-off 在极端压缩下仍未解决 [§5, Table 2]
4. **STT 未一致领先**: 在 STT 任务上 LM-SPT 表现 "competitive" 但未显著超越 SpeechTokenizer (25Hz: 15.34 vs 13.32) [Table 2],说明 ASR teacher 的语义蒸馏更有利于生成而非理解 [agent 解读]
5. **任务覆盖有限**: 仅评估基础 STT/TTS,未涉及 instruction-following、speech editing 等复杂任务 [§5]
6. **未开源**: 代码和权重未公开发布

## 点评

**优势**:
- Reconstruction-driven distillation 是一个优雅的设计: 绕过帧率对齐约束的方式既简洁又有效,且配有理论分析 (KL 散度下界) 支撑
- 消融实验全面且系统: 每个设计选择 (distillation 方式、decoder 共享/解耦/容量、encoder 共享/分离、Vocos vs mirrored decoder) 都有对照实验
- 多帧率验证 (25/12.5/6.25Hz) 展示了方法的一致优势
- SNMI 指标是有价值的新分析工具,补充了传统 PNMI 的语义对齐评估

**不足**:
- 仅在 LibriSpeech 上训练,缺乏大规模/多语言验证 (LM-SPT (L) 只测了 12.5Hz 一个帧率)
- 与 CosyVoice 系列的监督式 semantic tokenizer 路线缺乏对比 — 后者直接用 ASR loss 训练 token,可能在语义对齐上更直接
- 理论分析基于简化的高斯假设 (Eq 2-5),实际 encoder 输出分布可能复杂得多
- Whisper Small 作为 teacher 的选择缺乏消融 (Large? Medium?)

## 可复用的 idea

1. **Reconstruction-driven distillation 范式**: 当 teacher 和 student 帧率不同时,不直接对齐特征,而是"通过重建绕道对齐"。这个思路可泛化到任何需要跨分辨率知识蒸馏的场景 (视觉、多模态等)。
2. **轻量 auxiliary decoder 作为 information bottleneck**: 刻意使用低容量 decoder 迫使 encoder 学习更紧凑表征,比冻结 decoder 更有效。参数量 2.4M vs 47.4M 的对比 (Table 4) 是很好的参考。
3. **SNMI 语义对齐度量**: 将 HuBERT 的 PNMI (phoneme-level) 扩展到 sequence-level,用 VCTK 重复句子控制语义变量,配合 LSH 哈希处理变长序列。可用于评估任何 speech tokenizer 的语义质量。
4. **Dual encoder 显式分离**: 用独立 encoder 分别处理 semantic 和 acoustic,比 shared encoder + VQ 分离更彻底。额外 22M 参数换来一致性提升,性价比合理。

> [!review] 审阅状态
> 待审阅 — 见 `_review/LM-SPT-review.yml`
