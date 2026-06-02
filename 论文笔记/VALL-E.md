---
type: paper
tier: deep
title: "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers"
arxiv_id: "2301.02111"
source: "https://arxiv.org/abs/2301.02111"
authors: [Chengyi Wang, Sanyuan Chen, Yu Wu, Ziqiang Zhang, Long Zhou, Shujie Liu, Zhuo Chen, Yanqing Liu, Huaming Wang, Jinyu Li, Lei He, Sheng Zhao, Furu Wei]
year: 2023
venue: "arXiv preprint"
tags: [TTS, zero-shot, codec-LM, in-context-learning, large-scale-training, AR-NAR]
concepts: ["[[Residual Vector Quantization]]", "[[Speech Tokenizer]]"]
models: ["[[EnCodec]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-01
updated: 2026-06-01
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[Residual Vector Quantization]], [[Speech Tokenizer]], [[Zero-shot Speech Synthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Residual Vector Quantization]]✓, [[Speech Tokenizer]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Neural Audio Compression]](pending-review) | 未命中但可能相关: 无

**谱系定位**: VALL-E 是 Zero-shot TTS 任务页中记录的 "LLM + 离散 token" 路线的开山之作。在 Speech Tokenizer 页中,VALL-E 属于使用"声学 tokenizer"(EnCodec/SoundStream 的 RVQ codes)作为 LM 生成目标的首个 TTS 系统。

**已有认知**: RVQ 页已记录 EnCodec 的 codebook 训练方式(EMA update)。Speech Tokenizer 页将 codec tokens 归类为第 3 类"声学 tokenizer",并指出后续 CosyVoice 系列转向了监督式 semantic tokenizer 以提升 speaker similarity。Zero-shot TTS 页记录了 VALL-E 后继系统(CosyVoice 3, Seed-TTS, IndexTTS2, MaskGCT)的 SOTA 结果。

**创新判断**: VALL-E 是首个将 TTS 重构为条件语言建模任务(而非连续信号回归)的工作,开创了 codec LM TTS 范式。本文对理解后续所有 LLM-based TTS 系统的设计动机和演进方向至关重要。

## 速查

> [!summary] 速查
> - **一句话**: 首个将 TTS 重构为 neural codec 条件语言建模任务,60K 小时训练使模型涌现 in-context learning 能力,3 秒 prompt 即可零样本克隆
> - **路线**: Text → G2P → Phoneme + 3s Acoustic Prompt(EnCodec codes） → AR model（生成 1st quantizer codes） → NAR model（生成 2-8th quantizer codes） → EnCodec Decoder → Waveform
> - **指标**: WER 5.9% / SPK 0.580 (LibriSpeech) vs YourTTS WER 7.7 / SPK 0.337 [Table 2]; SMOS 4.38 vs YourTTS 3.45 [Table 3]; CMOS +0.04 vs GroundTruth on VCTK [Table 7]
> - **可借鉴**: 将 TTS 重构为 LM task 的范式转换思路; AR(coarse) + NAR(fine) 的层级生成匹配 RVQ 结构; 60K h 半监督数据 scaling; 3s prompt 作为 in-context example
> - **局限**: 鲁棒性问题(词漏/重复/错序); 仅英文; 需两个独立模型(AR+NAR); 未开源; SPK 相对 GT 仍有明显 gap (0.580 vs 0.754)

## 核心问题

传统 TTS 系统面临三个结构性瓶颈 [§1]:
1. **表征瓶颈** — 使用 mel spectrogram 作为中间表征,需要连续信号回归,无法利用 LM 的强大生成能力
2. **数据瓶颈** — 训练数据仅百~千小时(录音棚数据),无法支持零样本泛化
3. **方法瓶颈** — 零样本依赖 speaker adaptation/encoding,需要额外 fine-tuning 或复杂特征工程

VALL-E 的核心洞察: 如果用 neural codec 的离散 codes 替代 mel spectrogram,TTS 就变成了 conditional language modeling — 可以直接借用 GPT 的范式(大数据 + in-context learning）[§1, Table 1]。

## 方法: 它怎么 work

### 整体架构

给定文本 x 和 3 秒注册语音,VALL-E 生成对应的 acoustic code matrix C ∈ R^{T×8} [§4.1, Fig 1]:
1. 文本 → G2P → phoneme 序列 x
2. 注册语音 → EnCodec → acoustic prompt matrix C_tilde ∈ R^{T'×8}
3. AR model: p(c_{:,1} | x, C_tilde_{:,1}) — 自回归生成第 1 层 codes
4. NAR model: p(C_{:,2:8} | x, C_tilde, C_{:,<j}) — 非自回归逐层生成 2-8 层 codes
5. EnCodec decoder: C → waveform

### 关键设计选择

#### 1. EnCodec 作为 Speech Tokenizer [§3]

使用预训练 EnCodec (Défossez et al., 2022):
- 24 kHz 音频, 320x 下采样 → 75 Hz frame rate
- 8 层 RVQ, 每层 1024 entries (10 bits)
- 6 kbps bitrate, 10 秒音频 → 750 × 8 code matrix [§3]

**为什么选 codec codes 而非 HuBERT/k-means**: codec codes 保留完整声学信息(speaker identity + acoustic details),而 HuBERT codes 丢弃 speaker identity [§3]。同时 codec decoder 直接生成 waveform,无需额外 vocoder。

#### 2. 层级 AR + NAR 设计 [§4.2, Fig 3]

**为什么分两阶段**: RVQ 的层级结构天然适合分层生成 — 第 1 层 codes 编码 coarse 信息(内容+韵律+说话人大致特征),后续层编码 fine acoustic details [§4.2]。

**AR model** (第 1 层) [§4.2.1]:
- Decoder-only Transformer: 12 层, 16 heads, dim 1024, FFN 4096
- 输入: [phoneme_prompt; phoneme_target; <EOS>; acoustic_prompt_c1; acoustic_target_c1; <EOS>]
- 训练: 标准 causal LM, prompt prefix 作为 in-context example
- Embedding: phoneme embedding W_x + acoustic embedding W_a (共享 output projection)

**NAR model** (第 2-8 层) [§4.2.2]:
- 相同架构但非自回归: 每个 token 可 attend to 所有位置
- 每步训练随机采样 stage i ∈ [2,8], 输入前 i-1 层 codes 的 embedding sum
- 使用 Adaptive Layer Normalization (AdaLN) 注入 stage embedding
- 8 个 acoustic embedding 层, j-th prediction layer 权重 = (j+1)-th embedding layer 权重

**为什么 AR+NAR 优于纯 AR**: AR 处理时长可变性(说话人语速不同), NAR 将后续层从 O(T) 降为 O(1) [§4.2]。

#### 3. In-Context Learning via Prompting [§4.3]

- 推理: text → phoneme; enrolled 3s speech → EnCodec → prompt codes
- AR: sampling-based decoding (避免 beam search 导致的 loop)
- NAR: greedy decoding (取最高概率 token)
- 两种模式: VALL-E (完全零样本) 和 VALL-E-continual (续写模式)

### 训练策略

- 数据: LibriLight 60K hours, ~7000 speakers, ASR 自动标注 phoneme [§5.1]
- 16 × V100 32GB GPUs, batch size 6K acoustic tokens/GPU, 800K steps
- AdamW, warmup 32K steps → peak 5e-4 → linear decay
- 音频裁剪: 随机 10-20 秒片段训练

## 实验

| 指标 | 本文 (VALL-E) | Baseline (YourTTS) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (%) | 5.9 | 7.7 | LibriSpeech test-clean | [Table 2] |
| Speaker Similarity | 0.580 | 0.337 | LibriSpeech test-clean | [Table 2] |
| WER (VALL-E-continual) | 3.8 | - | LibriSpeech test-clean | [Table 2] |
| SMOS | 4.38±0.10 | 3.45±0.09 | LibriSpeech (40 speakers) | [Table 3] |
| CMOS vs baseline | +0.12 | 0.00 | LibriSpeech | [Table 3] |
| SMOS (VCTK) | 3.81±0.09 | 3.70±0.09 | VCTK (60 speakers) | [Table 7] |
| CMOS (VCTK) | +0.23 vs YourTTS | - | VCTK | [Table 7] |
| CMOS vs GT (VCTK) | -0.04 (无显著差异) | - | VCTK | [Table 7] |
| SPK 10s prompt (VCTK) | 0.484 | 0.394 | VCTK 108 speakers | [Table 6] |

**关键发现**:
- VALL-E 是首个在 VCTK 上达到与 ground truth 无统计显著差异的零样本 TTS [Table 7]
- Speaker similarity 随 prompt 长度增加而提升: 3s→5s→10s (0.382→0.423→0.484) [Table 6]
- 纯 NAR(去掉 acoustic prompt)speaker similarity 从 0.585 骤降至 0.236 → acoustic prompt 对音色克隆至关重要 [Table 5]
- VALL-E 能保持 prompt 的混响环境和情感 [§5.4]
- 合成输出具有多样性(同一输入不同 seed 产生不同韵律/语速) [Fig 4]

## 局限性

1. **鲁棒性不足** — AR 模型存在词漏/重复/错序问题(attention alignment 无约束) [§6]
2. **仅英文** — 60K h 仅含英语,accent speakers 覆盖不足 [§6]
3. **双模型结构** — AR+NAR 分开训练,未来可统一为单模型 [§6]
4. **Speaker similarity gap** — 与 ground truth 仍有明显差距 (0.580 vs 0.754) [Table 2]
5. **未开源** — 无官方代码和权重
6. **隐私风险** — 可被用于 voice spoofing [§6]

## 点评

**历史地位**: VALL-E 是 TTS 领域的 "GPT 时刻" — 将 TTS 从信号处理问题重构为语言建模问题,打开了用 LLM scaling laws 解决 TTS 的大门。后续 Seed-TTS、CosyVoice、F5-TTS、MaskGCT 等均沿用此范式。

**优势**:
- 范式转换的简洁性: "mel spectrogram regression → codec LM" 一步到位解决了表征/数据/方法三个瓶颈
- In-context learning 的巧妙利用: 将 speaker cloning 转化为 prompt engineering,无需任何 adaptation 机制
- AR+NAR 层级设计与 RVQ 天然匹配: 第 1 层控制内容+时长(AR), 后续层填充声学细节(NAR)

**不足**:
- 鲁棒性是硬伤: 无 duration predictor 或 alignment constraint 导致合成不稳定
- 实验评估偏简单: 仅在英文 LibriSpeech/VCTK 上测试,无多语言/噪声场景
- 与 AudioLM 的差异未充分讨论: AudioLM 也用 codec codes 但做 speech-to-speech

## 可复用的 idea

1. **Codec LM 范式**: 将任何 "continuous signal generation" 问题转化为 "discrete token LM" — 适用于 music generation, sound effect synthesis 等
2. **AR + NAR 层级生成**: 对多层离散表征 (RVQ/multi-stream tokens), coarse 层用 AR 保证全局一致性, fine 层用 NAR 加速
3. **In-context learning for speaker cloning**: 将 reference audio 编码为 prefix tokens, 让 LM 自动学习 speaker 特征提取,无需显式 speaker encoder
4. **半监督大规模训练**: 用 ASR 自动标注 unlabeled audio-only corpus, 以噪声标注换取数据量级提升 (60K h vs 百小时)
5. **AdaLN for stage conditioning**: 在 NAR model 中用 AdaLN 注入当前 stage 信息,使单模型处理多层生成

> [!review] 自动审阅 (2026-06-02)
> **结论:** pass-with-fixes
> **评分:** 理解 4 | 溯源 4 | 严谨 4 | 导航 3 | 安全 4
> **Claim 标注率:** 100% (20/20)
> **问题:** 0 high, 2 medium, 1 low
> - [medium/template-compliance] frontmatter > datasets: 字段为空但论文使用 LibriLight/LibriSpeech/VCTK 三个核心数据集
> - [medium/bad-linking] frontmatter > concepts: 仅 2 个概念,缺少 [[LLM-based TTS]] 等核心概念挂接
> **反向更新:** ✅
