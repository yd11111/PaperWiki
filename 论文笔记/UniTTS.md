---
type: paper
tier: deep
title: "UniTTS: An end-to-end TTS system without decoupling of acoustic and semantic information"
arxiv_id: "2505.17426"
source: "Sources/UniTTS.pdf"
authors: [Rui Wang, Qianguo Sun, Tianrong Chen, Zhiyun Zeng, Junlong Wu, Jiaxing Zhang]
year: 2025
venue: "arXiv preprint"
tags: [TTS, LLM-based-TTS, audio-codec, single-codebook, codec-distillation, universal-audio, DPO, alignment]
concepts: ["[[ResidualVectorQuantization]]", "[[LLM-basedTTS]]", "[[SemanticvsAcousticTokens]]", "[[Single-codebookvsMulti-codebook]]", "[[SpeechTokenizer]]", "[[CodebookCollapse]]", "[[CodecTrainingObjectives]]"]
models: ["[[CosyVoice2]]", "[[EnCodec]]", "[[SoundStream]]"]
tasks: ["[[NeuralAudioCompression]]", "[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 1 个待确认实体页: [[ResidualVectorQuantization]], [[LLM-basedTTS]], [[SemanticvsAcousticTokens]], [[SpeechTokenizer]], [[CodebookCollapse]], [[Single-codebookvsMulti-codebook]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ResidualVectorQuantization]]✓, [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechTokenizer]]✓, [[CodebookCollapse]]✓ | 过滤: [[Single-codebookvsMulti-codebook]](pending-review) | 未命中但可能相关: 无

**谱系定位**: UniTTS 位于 LLM-based TTS 的 single-codebook + 不做 semantic-acoustic 解耦路线上。与主流方案(CosyVoice 系列的监督式 semantic tokens + CFM, VALL-E 系列的多码本 AR+NAR, Spark-TTS/Llasa 的 X-codec2/BiCodec 语义对齐单码本)不同, UniTTS 明确放弃 semantic alignment, 使用知识蒸馏将多码本 NAC 压缩为超大单码本(32768 codes)的 DistilCodec, 让 LLM 直接建模完整声学信息。

**已有认知**: KB 中 [[Single-codebookvsMulti-codebook]] [待确认] 记录了从多码本向单码本回归的趋势(BigCodec/WavTokenizer/Llasa), 但已有单码本方案通常仍依赖 semantic distillation(如 SpeechTokenizer 蒸馏 HuBERT, X-codec2 蒸馏 semantic 信息)。[[CodebookCollapse]] 页面记录了大码本的核心挑战: DAC 用 factorized codes + L2-norm 将利用率从 62% 提升到 99%, FSQ 从结构上消除码本。UniTTS 的 DistilCodec 声称 32768 大码本实现接近 100% 利用率, 这在已有 KB 中属于新方案。

**创新判断**: 相对于 KB 已记录的工作, UniTTS 的核心创新在于: (1) 提出 DMS (多码本→单码本蒸馏)算法, 通过 encoder/decoder 参数继承实现高效蒸馏; (2) 不做 semantic alignment, 用 universal audio 训练 NAC, 支持非语音音频; (3) 三任务预训练(audio AR + text AR + cross-modal)在同一 LLM 中统一文本和音频能力。这是一条与 CosyVoice/Llasa/Spark-TTS 都不同的路线。

## 速查

> [!summary] 速查
> - **一句话**: 提出 DistilCodec(多码本→单码本蒸馏, 32768 codes, ~100% 利用率)+ UniTTS(Qwen2.5-7B 三任务预训练), 不做 semantic-acoustic 解耦, 端到端建模完整音频信息
> - **路线**: Audio → DistilCodec Encoder → VQ(1 codebook, 32768, dim=3584) → Audio Tokens; Text → Qwen2.5 Word Embedding; 两者拼接为 ~180K 词表 → Qwen2.5-7B(Pretrain→SFT→LPO) → Audio Tokens → DistilCodec Decoder → Waveform
> - **指标**: MOS Emotional expressiveness 4.60(最高), Naturalness 4.94(最高), Fidelity 4.80, Stability 4.97; SFT CER 3.43%→AB-1 CER 3.466%; DistilCodec codebook usage 98.2-99.9%, PPL 21660-26999
> - **可借鉴**: (1) DMS 蒸馏策略(继承 teacher encoder/decoder 参数, 仅重训 VQ 层)降低单码本训练难度; (2) 码本维度对齐 LLM word embedding(3584 = Qwen2.5-7B embedding dim), 直接用 codebook 初始化 audio embedding; (3) LPO(Linear Preference Optimization)替代 DPO 避免超长序列下的 mode collapse
> - **局限**: (1) 预训练 loss 未收敛, 计算资源不足 [Fig 7]; (2) SFT 仅 948h 音频数据(vs Llasa 250Kh, Spark-TTS 100Kh); (3) MOS 评估基于自建 test set, 非标准 benchmark(无 LibriSpeech/SEED-TTS-Eval WER/SIM); (4) 代码开源但训练数据部分为自有

## 核心问题

UniTTS 要回答的核心问题: **多码本 NAC 通过 semantic-acoustic 解耦获得了 LLM-TTS 的良好性能, 但 semantic alignment 会丢失非语言声学信息(笑声、哭声、音效); 能否不做解耦, 用单码本完整建模所有音频信息, 同时保持 LLM-based TTS 的可用性?**

具体拆解为三个子问题 [§1]:
1. 如何在不依赖 semantic prior 的前提下, 训练低比特率高性能的 universal audio NAC?
2. 如何在 NAC 和 LLM-based TTS 训练中充分利用无标注的 universal audio 数据?
3. LLM 配合 universal audio NAC 能否有效实现 text-audio 对齐?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UniTTS 由两个核心组件组成 [§3.1, Fig 1]:
1. **DistilCodec**: 单码本 audio tokenizer, 负责 audio ↔ tokens 转换
2. **Qwen2.5-7B**: LLM backbone, 建模 audio tokens 和 text tokens 的关系

工作流: Audio → DistilCodec Encoder → VQ(32768 codes, dim 3584) → Audio Tokens → LLM → Audio Tokens → DistilCodec Decoder → Waveform

### 关键设计选择

**设计选择 1: 不做 semantic-acoustic 解耦**

[论文原文] 论文认为 semantic 和 acoustic 信息无法完全对齐, 尤其对韵律显著的非语言发声(笑声、哭声)和含复杂背景的高保真通用音频, 解耦会导致 LLM 无法获取完整音频信息 [§1]。因此 DistilCodec 放弃 semantic alignment, 直接用 universal audio 数据训练纯重建目标的 codec。

[agent 解读] 这一选择的本质权衡是: 放弃 semantic alignment 带来的 LM 建模便利(semantic token 与文本对齐好, LM 更容易学), 换取更完整的音频信息保留。代价是需要更多预训练数据和计算资源来让 LLM 自行学习 text-audio 对齐, 论文自己也承认预训练不够充分 [§3.3.3, Fig 7]。

**设计选择 2: DMS(多码本→单码本蒸馏)**

核心算法 [§3.2, Algorithm 1]:
1. 训练 TeacherCodec: 8 residual layers x 4 groups, codebook size 1024, dim 512
2. 初始化 StudentCodec(DistilCodec): 继承 Teacher 的 Encoder 和 Decoder 参数, 仅重新初始化 VQ 层(1 residual x 1 group, codebook size 32768, dim 3584)
3. 用 DLF(DistilCodec-LSGAN-Training)训练 StudentCodec

[论文原文] 参数继承使 Student 从已学好的 encoder/decoder 出发, 只需重训 VQ 层, 大幅降低训练难度 [§3.2]。DMS 同时允许码本维度超过传统上限 2048 [§3.1]。

[agent 解读] 这是一种标准的 knowledge distillation 策略应用于 codec 架构。关键 insight 是: teacher 的 encoder 已学会好的 latent space, student 继承后 VQ 层只需找到在这个空间中的好量化点, 比从零训练容易得多。但论文未做消融验证"不继承参数, 直接训练单码本"的效果差异。

**设计选择 3: 码本维度 = LLM embedding 维度**

DistilCodec 的码本维度设为 3584, 与 Qwen2.5-7B 的 word embedding 维度完全一致 [§3.1]。这样 codebook embedding 可以直接用来初始化 LLM 的 audio embedding 层, 词表从 ~150K 扩展到 ~180K。

[agent 解读] 这一设计消除了 audio embedding projector(如 MLP adapter)的需要, 减少参数和训练复杂度。类似做法在 Llasa 中也有(X-codec2 的码本维度对齐 LLaMA word embedding)。

**设计选择 4: 三任务预训练**

基于 Bayes 分解 P(A,T) = P(A|T)P(T) = P(T|A)P(A) [§3.3.1, Eq. 3], 设计三个预训练任务:
- P(A): Audio AR — 用无标注 universal audio 做音频自回归
- P(T): Text AR — 保持 LLM 原有文本能力
- P(A|T) & P(T|A): Cross-modal — text-audio 对齐

[论文原文] 实验发现引入 audio 训练数据后出现 modal competition, 文本生成能力退化 [§3.3.1, Stage 1]。Stage 2 增加文本 instruction 数据恢复文本能力 [Appendix B.9]。

[agent 解读] Modal competition 是多模态 LLM 训练的常见问题(GPT-4o 等也有类似报告)。UniTTS 的解决方案(在 Stage 2 补充文本 instruction 数据)是 straightforward 但有效的。表 21 显示恢复后 MMLU 47→52(仍低于原始 74), 说明文本能力损失不可完全恢复。

**设计选择 5: LPO 替代 DPO**

[论文原文] DPO 在超长序列(如 UniTTS 的 audio token 序列)中容易 mode collapse [§3.3.3]。LPO(Linear Preference Optimization)通过线性化替代 DPO 的 log-sigmoid [Eq. 5-11]。

[agent 解读] LPO 的具体公式涉及 STE(straight-through estimator)和 hinge loss 形式的 margin 项, 核心思想是让正负样本对的 reward 差距线性化而非 sigmoid 压缩, 避免超长序列的数值不稳定。但论文未提供 DPO vs LPO 的直接消融对比。

### 训练策略

DistilCodec 训练 [§4.1]:
- 数据: 100Kh universal audio(中英有声书 + 语音 + 音乐) [Table 15]
- 硬件: 5x8 A100
- Teacher: 5 epochs; Student: 3 epochs
- Loss: Mel loss + GAN adversarial loss + feature matching loss [Eq. 1]
- Discriminators: MPD + MSD + Multi-STFT [Tables 11-13]

UniTTS 训练 [§4.3.1]:
- **Pretrain**: 322B tokens(Text 140B + Audio 100B + Text-Audio 82B); 两阶段, context 8192→16384, LR 1e-4→2e-5→9e-6
- **SFT**: 仅 948h 音频 + 181K text + 55K long-CoT; LR 9e-6→5e-6, context 8192, batch 128
- **LPO**: 445K 样本(300K text-audio + 100K general SFT + 45K long-CoT); 每个 prompt 生成 3 个候选, 与 reference 配对

## 实验

### DistilCodec 评估

| 指标 | DistilCodec | BigCodec | WavTokenizer | X-codec2 | DAC(Nq=2) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Codebook Size | 32768 | 8192 | 4096 | 65536 | 1024 | [Table 3] |
| Nq | 1 | 1 | 1 | 1 | 2 | [Table 3] |
| Bandwidth (bps) | 1300 | 1040 | 900 | 800 | 1000 | [Table 3] |
| STOI↑ | 0.93 | 0.94 | 0.89 | 0.92 | 0.73 | [Table 3] |
| PESQ↑ | 2.02 | 2.68 | 2.14 | 2.43 | 1.14 | [Table 3] |
| UTMOS↑ | 3.75 | 4.11 | 3.94 | 4.13 | 1.29 | [Table 3] |
| Codebook Usage(%) | 98.2(speech)/99.9(universal) | - | - | - | - | [Table 2] |

[agent 解读] DistilCodec 的重建指标(PESQ 2.02, UTMOS 3.75)明显低于 BigCodec(2.68, 4.11)和 X-codec2(2.43, 4.13), 尽管码本更大。这可能因为 DistilCodec 用 universal audio 训练(含音乐/音效), 而其他方案主要用 speech 数据; 也可能蒸馏方案本身有信息损失。高码本利用率(98-99%)是亮点。

Universal Audio MOS [Table 4]:
- Speech Clarity: DistilCodec 4.689 vs GT 4.945
- Background Audio Clarity: DistilCodec 4.768 vs GT 4.927
- Average: 4.728 vs 4.936

### TTS 评估

| 指标 | UniTTS-LPO | UniTTS-SFT | CosyVoice2 | Spark-TTS | Llasa | F5-TTS | Fish Speech | IndexTTS | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fidelity | 4.80 | 4.43 | 4.80 | 4.89 | 4.74 | 4.94 | 4.89 | 4.69 | [Table 5] |
| Stability | 4.97 | 5 | 5 | 5 | 4.91 | 5 | 5 | 4.83 | [Table 5] |
| Naturalness | 4.94 | 4.77 | 4.89 | 4.89 | 4.91 | 4.89 | 4.83 | 4.89 | [Table 5] |
| Emotional expressiveness | 4.60 | 4.23 | 4.11 | 4.26 | 4.11 | 3.97 | 4.29 | 4.31 | [Table 5] |

[agent 解读] UniTTS-LPO 在 Emotional expressiveness 上的优势(4.60 vs 其他最高 4.31)是最显著的差异化结果, 符合论文"不解耦可保留更完整情感信息"的假设。但需注意: (1) 评估用的是自建 diversity evaluation dataset(含笑声、哭声等), 可能有利于 universal audio codec; (2) 未使用标准 benchmark(LibriSpeech, SEED-TTS-Eval)做 WER/SIM 评估, 可比性有限。

### 消融分析

| 实验 | CER | 关键变量 | 出处 |
| --- | --- | --- | --- |
| AB-1 (PROMPT1 + text+CoT) | 3.466% | 完整设置, text先audio后 | [Table 6] |
| AB-2 (PROMPT3, 无text) | 5.505% | prompt 仅有audio | [Table 6] |
| AB-3 (加 ASR 任务) | 3.582% | TTS+ASR 任务混训 | [Table 6] |
| AB-4 (仅 TTS) | 3.703% | 无 text/CoT data | [Table 6] |
| AB-5 (PROMPT2, audio先text后) | 3.740% | 顺序影响 | [Table 6] |

关键发现 [§4.3.3]:
- 加入文本 instruction 数据可改善 audio 质量(AB-1 vs AB-4: 3.47% vs 3.70%) [论文原文]
- Prompt 中包含 reference text 很重要(AB-1 vs AB-2: 3.47% vs 5.50%) [论文原文]
- Text-先-audio-后的顺序优于反序(AB-1 vs AB-5: 3.47% vs 3.74%) [论文原文]
- ASR 与 TTS 混训在当前预训练规模下有 task confusion 问题(AB-3 部分输出含非 audio tokens) [论文原文]

### 预训练规模影响

| 数据量/设置 | CER | 出处 |
| --- | --- | --- |
| AB_SFT (仅 SFT, 6.2M pairs, 无预训练audio) | 18.18% | [Table 22] |
| UniTTS-SFT (预训练 322B tokens + SFT 401K pairs) | 3.43% | [Table 22] |

[论文原文] 仅 SFT 时 CER 高达 18.18%, 加上预训练后降至 3.43%, 验证了 universal audio 预训练的必要性 [Appendix B.10]。生成 audio tokens 的空间复杂度远大于 text tokens(同一文本两次合成仅 5% 重复率), 因此需要大量预训练 [Appendix B.10]。

## 局限性

1. **预训练不充分**: Loss 曲线仍在明显下降 [Fig 7], 论文承认计算资源不足 [§3.3.3]。这意味着当前结果可能远未达到该架构的上限。

2. **SFT 数据量极小**: 仅 948h(vs Llasa 250Kh, Spark-TTS 100Kh), 论文认为 universal audio 预训练弥补了这一点, 但缺乏等规模对比。

3. **评估不标准**: 未在 LibriSpeech/SEED-TTS-Eval 上报告 WER/SIM, MOS 评估用自建 test set, 难以与已有工作直接比较。消融实验用 seed-tts-eval 的 CER, 但主实验改用 MOS, 评估标准不一致。

4. **文本能力损失**: 引入 audio 模态后 MMLU 从 74→47→52 [Table 21], 数学和代码能力显著退化, 说明 modal competition 未完全解决。

5. **重建质量**: DistilCodec 在标准 speech benchmark 上的 PESQ/UTMOS 低于 BigCodec/X-codec2 等同类单码本方案 [Table 3]。Universal audio 训练可能牺牲了 speech-specific 性能。

6. **Codec 带宽偏高**: 1300 bps, 高于 X-codec2(800 bps), WavTokenizer(900 bps), BigCodec(1040 bps), 部分抵消了单码本的 token rate 优势。

## 点评

**优点**:
1. 提出了一条与主流不同的技术路线 — 不做 semantic-acoustic 解耦, 完全依赖 LLM 自身的建模能力从 raw audio tokens 中学习 text-audio 对齐。如果预训练足够充分, 这条路线理论上更简洁统一。
2. DMS 蒸馏策略简洁有效, 码本维度对齐 LLM embedding 是值得借鉴的工程优化。
3. 情感表现力指标确实领先, 支持"完整音频信息建模有利于情感/韵律"的假设。
4. 开源代码和模型。

**不足**:
1. 核心假设"semantic alignment 导致信息丢失"缺乏严格验证 — 未做 DistilCodec+semantic alignment vs DistilCodec(no alignment) 的控制实验。
2. 评估体系不完整: 缺少标准 benchmark(WER, SIM, SEED-TTS-Eval), 仅凭自建 test set 的 MOS 难以令人信服地支持"SOTA"结论。
3. 与 Llasa(相似路线, X-codec2 单码本 + LLaMA)的对比不公平 — UniTTS 用 7B backbone + 322B pretrain tokens, Llasa 从 1B-8B 均有, 且 Llasa 在更标准的 benchmark 上有成绩。
4. 训练规模与结论之间有 gap: 论文承认预训练不足但声称"验证了该路线的可行性", 这个 claim 的强度值得商榷。

**在领域中的定位**:
UniTTS 代表了 LLM-based TTS 中"去解耦"路线的一个有意义的探索。与 CosyVoice 系列(监督 semantic + CFM)、Spark-TTS/Llasa(语义对齐单码本 + LLM)形成三角关系。当前结果显示该路线在情感表现力上有潜力, 但在标准 TTS 指标上尚未证明优势。该路线的最终价值取决于预训练规模能否有效缩小 text-audio alignment gap。

## 可复用的 idea

1. **DMS 蒸馏框架**: 多码本→单码本的蒸馏策略(继承 encoder/decoder, 仅重训 VQ)可应用于其他需要码本压缩的场景。关键是 teacher 提供了好的 latent space, student 在此基础上做更高效的量化。

2. **码本维度对齐 LLM embedding**: 将 codec codebook dim 设为与目标 LLM 的 word embedding dim 相同, 直接用 codebook 初始化 audio embedding, 避免额外的 adapter/projector。这是一个可迁移的工程简化。

3. **Universal audio AR 预训练**: 在 LLM-TTS 预训练中加入无标注 universal audio 的自回归任务(P(A))。即使最终目标是 speech, 这种数据扩充可以帮助 LLM 更好地理解音频 token 空间的分布。

4. **LPO 替代 DPO**: 在超长序列(audio token 序列可达数千)的 preference optimization 中, 线性化的 LPO 可能比 DPO 更稳定, 值得在其他音频生成任务中尝试。

5. **质量分数数据过滤**: quality(i) = dnsmos(i) - cer(i) 的复合质量分数 [Eq. 4] 用于 SFT 数据筛选, 简洁且同时考虑了音频质量和文本准确性, 可直接复用。

---

> [!review] 审阅结论: pass (2026-06-03)
> - **可复述** ✓ 5 个设计选择均有因果解释
> - **可信赖** ✓ 数字标注覆盖率 >90%
> - **可区分** ✓ [论文原文]/[agent 解读] 标注覆盖率 >80%
> - **可定位** ✓ KB 背景谱系定位具体, frontmatter 完整
> - **不污染** ✓ 反向更新仅追加 key_papers, 无新建概念页
> - Issues: 3 low (速查指标密度高 / MOS 来源 / LPO 细节)
> - 详见 `_review/UniTTS-review.yml`

检索命中: [[ResidualVectorQuantization]], [[LLM-basedTTS]], [[SemanticvsAcousticTokens]], [[SpeechTokenizer]], [[CodebookCollapse]] | 过滤: [[Single-codebookvsMulti-codebook]](pending-review) | 未命中但可能相关: 无
