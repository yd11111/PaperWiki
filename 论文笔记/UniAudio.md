---
type: paper
tier: deep
title: "UniAudio: An Audio Foundation Model Toward Universal Audio Generation"
arxiv_id: "2310.00704"
source: "Sources/UniAudio.pdf"
authors: [Dongchao Yang, Jinchuan Tian, Xu Tan, Rongjie Huang, Songxiang Liu, Xuankai Chang, Jiatong Shi, Sheng Zhao, Jiang Bian, Xixin Wu, Zhou Zhao, Shinji Watanabe, Helen Meng]
year: 2023
venue: "Under review, ICLR 2024"
tags: [audio-generation, LLM, multi-task, TTS, voice-conversion, speech-enhancement, singing-voice, text-to-sound, text-to-music, audio-edit, speech-dereverberation, multi-scale-transformer, neural-codec, RVQ, universal-model]
concepts: ["[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[ResidualVectorQuantization]]", "[[SemanticvsAcousticTokens]]", "[[SpeechLanguageModel]]", "[[CodecLanguageModel]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[LLM-basedTTS]], [[SpeechTokenizer]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]], [[SpeechLanguageModel]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechLanguageModel]]✓ | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无

**[[LLM-basedTTS]]**: UniAudio 将 LLM-based TTS 的思想从 TTS 单任务推广到 11 种音频生成任务。与 VALL-E 专注于 TTS 不同,UniAudio 验证了 codec LM 范式在 universal audio generation 中的可行性。其 TTS 性能 (SIM 0.708, WER 2.0) 在当时与 NaturalSpeech 2 和 VoiceBox 可比 [Table 10]。

**[[SpeechTokenizer]]**: UniAudio 自建 neural codec (RVQ, n_q=3) 作为通用 audio tokenizer,将 speech/sound/music/singing 统一编码为同一类 audio tokens。同时使用 HuBERT 的 semantic tokens 作为 condition (k-means 500 clusters) [§2.1]。这是典型的 acoustic tokenizer + semantic condition 双路设计。

**[[ResidualVectorQuantization]]**: UniAudio 使用 3 层 RVQ (n_q=3) 量化 audio codec。RVQ 的多层结构直接导致了序列长度挑战: T 帧 x 3 层 = 3T tokens,这是 multi-scale transformer 设计的直接动因 [§2.3]。

**[[SemanticvsAcousticTokens]]**: UniAudio 同时使用两类 token: HuBERT semantic tokens 作为 voice conversion 的条件输入,RVQ acoustic tokens 作为所有任务的目标输出。这体现了 semantic tokens 用于条件控制、acoustic tokens 用于高保真生成的分工 [Table 1]。

**[[SpeechLanguageModel]]**: UniAudio 是 SpeechLM 向 "universal audio generation" 的泛化尝试。它支持 speech/sound/music/singing 四种音频类型的 11 种生成任务,超越了 SpeechLM 通常聚焦的 speech-only 范围。Multi-scale Transformer 设计兼顾了 inter-frame (global) 和 intra-frame (local) 两个层级的建模。

**[待确认]** [[CodecLanguageModel]]: UniAudio 直接在 neural codec 的 RVQ tokens 上做 next-token prediction,是 CodecLM 范式在多任务场景的代表。与 VioLA (ASR+TTS+ST) 类似但任务覆盖更广。

> [!summary] 速查
> - **一句话**: 首个支持 11 种音频生成任务的统一 LLM-based 音频基础模型, 通过统一 tokenization + 统一 task formulation + multi-scale transformer 实现跨任务迁移
> - **路线**: Conditions (Phoneme/MIDI/Text/Semantic Token/Audio) → Tokenization → [condition, target] sequence → Multi-Scale Transformer (Global + Local) → Audio Tokens (RVQ) → Neural Codec Decoder → Waveform
> - **指标**: TTS SIM 0.708 / WER 2.0 / MOS 3.81 / SMOS 3.56; VC SIM 0.868 / MOS 3.54; Text-to-Sound FAD 3.12 / KL 2.6 [Table 2, 10]
> - **可借鉴**: (1) Multi-scale Transformer 降低 RVQ 长序列复杂度 O(T*n_q)² → O(T²); (2) 统一 [condition, target] 序列格式处理多任务; (3) 多任务联合训练带来互利
> - **局限**: n_q 受限于 flattening 长度无法超过 4; 不含 noise removal / speech translation; 1B 参数+165K h 数据在当前标准下较小

## 核心问题

UniAudio 要解决的核心问题 [§1]:
1. **任务隔离**: 现有音频生成模型各自针对单一任务设计,无法共享跨任务知识 [§1]
2. **RVQ 长序列**: Neural codec 的多层 RVQ 展开后序列极长 (T × n_q),标准 Transformer 的 O((T*n_q)²) 复杂度不可接受 [§2.3]
3. **新任务扩展**: 如何让模型支持训练时未见的新任务? [§3.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UniAudio 由三大组件构成 [Fig 1 left]:

**1. Multi-Modal Tokenization** [§2.1]: 将所有输入模态离散化
- Audio (speech/sound/music/singing): 自建 neural codec + RVQ (n_q=3)
- Phoneme: 音素序列 (来自 CMUDict 或 forced alignment)
- MIDI: F0 序列 (duration 信息展平)
- Text: T5 encoder 连续 embeddings
- Semantic Token: HuBERT 9th layer + k-means 500 clusters

**2. Unified Task Formulation** [§2.2, Table 1]:
所有 11 种任务统一为 [conditions, target] 序列格式:
```
<start> <task_id> <cond_start> condition_tokens <cond_end> <audio_start> audio_tokens <audio_end> <end>
```
特殊 token 标记序列/子序列/模态的边界,以及任务身份 [§2.2]

**3. Multi-Scale Transformer** [§2.3, Fig 1 right]:
- **Global Transformer**: 24 layers, dim 1536, 12 heads, 744M params; 以 patch (n_q 个 token) 为单位处理 inter-frame 相关性 [§2.3, Table 8]
- **Local Transformer**: 8 layers, dim 1536, 12 heads, 238M params; 处理 patch 内 n_q 个 token 的 intra-frame 相关性 [§2.3, Table 8]
- 总参数: ~1B [§3.1]

### 关键设计选择

#### 1. 通用 Audio Tokenization [§2.1.1]

**WHY**: [论文原文] 现有 neural codec (EnCodec, SoundStream) 是 domain-specific 的,不适合统一建模 speech+sound+music [§2.1.1]

**HOW**: 自建 neural codec,使用更广域的训练数据和 broader data coverage [§3.1, Appendix E]
- Encoder-Decoder + RVQ (n_q=3)
- Audio → h = Encoder(x) → ĥ = Quantization(h) → x̂ = Decoder(ĥ) [Eq 1]
- 每帧 n_q 个 discrete index: z_t = [z¹_t, z²_t, ..., z^{n_q}_t] [§2.1.1]
- RVQ 过程: 迭代量化残差, ĥ_t = Σ q_j^{z_t^j} [Eq 2]
- 序列展平: z ∈ Z^{T×n_q} → 1D 序列, 每 n_q 个 token 为一个 patch [§2.1.1]

#### 2. Multi-Scale Transformer 设计 [§2.3, Fig 1 right]

**WHY**: [论文原文] Flattening prediction (如 MusicGen) 虽然保持自回归性质获得最优生成质量,但序列长度 T×n_q 导致 O((T*n_q)²) 复杂度,n_q≥4 时训练困难 [§3.4.2]

**HOW**: 层级建模,分离 inter-frame 和 intra-frame:

**(a) Global Transformer 处理 frame-level**:
- 每帧的 n_q 个 token 的 embeddings 求和,作为一个 patch 的表征 [§2.3]
- Global Transformer 预测帧 x_t 时,条件是所有 previous frames x_{t-1} 和所有之前内容 [§2.3]
- 输出 hidden representation h_t,线性变换后作为 patch-level context [§2.3]
- [agent 解读] h_t 独立于 z_t 中的具体 token 选择,只依赖帧级语义

**(b) Local Transformer 预测 patch 内 token**:
- 给定 Global Transformer 的 h_t,Local Transformer 自回归预测 z_t^k (k=1..n_q) [§2.3]
- 每个 token z_t^k 依赖前 k-1 个 token: z_t^k depends on {z_t^j | j < k} [§2.3]
- [论文原文] Local Transformer 仅处理长度为 n_q 的短序列,计算量轻

**(c) 非 audio token 的处理**:
- Phoneme/Semantic/MIDI 等离散 token: 重复 n_q 次填充 patch [§2.3]
- Text 连续 embedding: 也重复 n_q 次,embedding 过程替换为线性变换,预测目标为 <continuous_token> 特殊 token [§2.3]

**复杂度分析** [§3.4.2]:
- Flattening: O((T × n_q)²) — n_q≥4 时不可行
- Multi-Scale Transformer: O(T²) — global 建模独立于 n_q [§3.4.2]
- [论文原文] 因此可以支持更大的 n_q (如 n_q=8),这是 flattening 做不到的

#### 3. 统一 Task Formulation [§2.2, Table 1]

**WHY**: [论文原文] 各音频生成任务的输入条件不同 (phoneme/text/MIDI/audio/semantic),但目标都是生成音频 [§2.2]

**HOW**: 所有任务统一为序列到序列格式 [Table 1]:

| Task | Conditions | Target |
|------|-----------|--------|
| TTS | phoneme, speaker prompt | speech |
| Voice Conversion | semantic token, speaker prompt | speech |
| Speech Enhancement | noisy speech | speech |
| Target Speech Extraction | mixed speech, speaker prompt | speech |
| Singing Voice Synthesis | phoneme (w/ duration), speaker prompt, MIDI | singing |
| Text-to-Sound | textual description | sounds |
| Text-to-Music | textual description | music |
| Audio Edit | textual description, original sounds | sounds |
| Speech Dereverberation | reverberant speech | speech |
| Instructed TTS | phoneme, textual instruction | speech |
| Speech Edit | phoneme (w/ duration), original speech | speech |

前 7 个任务为 training stage,后 4 个为 fine-tuning stage [Table 1]

#### 4. 两阶段训练: Training + Fine-tuning [§3.1]

**WHY**: [论文原文] 为验证模型扩展到新任务的能力 [§3.3]

**HOW**: 
- Training stage: 7 tasks, 165K hours, 16 AMD MI200 GPUs [§3.1]
- Fine-tuning stage: 新增 4 tasks, re-sampling α=0.05 保持旧任务性能 [§3.1]
- Top-k=30, temperature=0.8 [§3.1]
- Pre-training: batch 8k patches/GPU, LR 1e-4, warmup 10K steps, 800K steps total [Table 9]
- Fine-tuning: batch 8k patches/GPU, LR 1e-5, warmup 1K steps, 50K steps total [Table 9]

### 训练策略

**数据** [§3.1, Table 6-7]:
- 12 public datasets, 总量 165K hours [Table 6]
- TTS/VC: LibriLight 60K h
- Sound: AudioCaps + WavCaps 7.5K h
- Music: Million Song Dataset 7K h
- Singing: OpenCPOP + OpenSinger + AISHELL3 150 h
- Evaluation: LibriSpeech, VCTK, Cloth, MusicCaps, M4Singer 等 [Table 7]

**模型配置** [Table 8]:
- Global Transformer: 24 layers, 1536 dim, 12 heads, 6144 FFN, 744M params, max context 3000 tokens [Table 8]
- Local Transformer: 8 layers, 1536 dim, 12 heads, 6144 FFN, 238M params, max context 3 tokens (= n_q) [Table 8]
- 总词表: 4212 (含所有特殊 token) [§3.1]
- Both transformers: causal, vanilla decoder [§3.1]

## 实验

### Training Stage 任务 (7 tasks) [Table 2]

| Task | Metric | UniAudio | Best Prior | 出处 |
| --- | --- | --- | --- | --- |
| TTS | SIM↑/WER↓ | **0.71**/2.0 | NatSpeech2: 0.62/2.3 | [Table 2] |
| TTS | MOS/SMOS | 3.81/3.56 | NatSpeech2: **3.83**/3.11 | [Table 2] |
| Voice Conversion | SIM↑/WER↓ | **0.87**/4.8 | LM-VC: 0.82/4.91 | [Table 2] |
| Speech Enhancement | MOS↑ | **3.68** | SGMSE+: 3.56 | [Table 2] |
| Target Speaker Extraction | MOS↑ | **3.72** | VoiceFilter: 3.43 | [Table 2] |
| Singing Voice Synthesis | MOS↑/SMOS↑ | **4.08**/4.04 | DiffSinger: 3.94/4.05 | [Table 2] |
| Text-to-Sound | FAD↓/KL↓ | **3.12**/2.6 | AudioGen: 2.55/2.5 | [Table 2] |
| Text-to-Music | FAD↓/KL↓ | 3.65/1.87 | MusicGen: **4.52**/1.41 | [Table 2] |

### Fine-tuning Stage 任务 (4 tasks) [Table 3]

| Task | Metric | UniAudio | Best Prior | 出处 |
| --- | --- | --- | --- | --- |
| Audio Edit | FD↓/KL↓ | **17.78**/0.77 | AUDIT: 20.78/0.86 | [Table 3] |
| Speech Dereverberation | PESQ↑/DNSMOS↑ | 2.13/**3.51** | SGMSE+: 2.87/3.42 | [Table 3] |
| Instructed TTS | MOS↑/SMOS↑ | 3.61/3.71 | GT: **3.77**/3.85 | [Table 3] |
| Speech Edit | MCD↓/MOS↑ | **5.12**/3.82 | TTS regen: 6.98/3.69 | [Table 3] |

### Multi-Scale Transformer 消融 [Table 4-5]

| Structure | n_q | MOS↑ | MCD↓ | GPU Mem (GB) | Time (s)/Iter | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Flattening | 3 | 3.80±0.09 | 6.56 | 36.7 | 1.63 | [Table 4] |
| Multi-Scale (ours) | 3 | 3.77±0.05 | **6.52** | 19.4 | **0.73** | [Table 4] |
| Multi-Scale (ours) | 8 | **3.84±0.06** | 6.27 | 24.0 | 1.10 | [Table 4] |
| Coarse first | 8 | 3.48±0.05 | 7.37 | 18.7 | 0.58 | [Table 4] |
| Parallel | 3 | 3.14±0.07 | 7.89 | 13.56 | 0.53 | [Table 4] |
| Delay | 3 | 3.48±0.05 | 6.95 | 13.65 | 0.59 | [Table 4] |

**关键发现**:
1. Multi-Scale Transformer (n_q=3) 性能与 Flattening 相当 (MOS 3.77 vs 3.80) 但 GPU 内存减半 (19.4 vs 36.7 GB)、训练速度快 2.2x [Table 4]
2. Multi-Scale 可以支持 n_q=8 (MOS 3.84, 最优),而 Flattening 无法扩展到 n_q≥4 [§3.4.2, Table 4]
3. 自回归性质很重要: Flattening 和 Multi-Scale (均为 AR) 显著优于 Parallel/Delay (非 AR) [Table 4]
4. 联合训练多任务一致优于单任务训练 [Appendix C.1]
5. Fine-tuning 新任务不影响原有任务性能 [Appendix C.2]
6. 数据量增加持续带来性能提升 [Appendix C.3]

## 局限性

1. **n_q 限制的变通而非根本解决**: Multi-Scale Transformer 降低了 global 复杂度,但 n_q 增大仍增加 local 计算和总推理时间 [Table 4] [agent 解读]
2. **任务覆盖不完全**: 不含 noise removal、speech-to-speech translation 等任务 [§5]
3. **Fine-tuning 仅支持已知模态**: 新任务必须用已有模态表示,无法引入训练时未见的新模态 [§5]
4. **未使用无标注数据**: 所有训练基于标注数据集,未利用大量可用的无标注音频 [§5]
5. **未使用 domain-specific foundation models**: 如预训练 speech encoder、music encoder 等 [§5]
6. **评估局限**: 部分任务使用非官方实现的 baseline (footnote 11); signal-level metrics (PESQ, VISQOL) 可能不适合生成式方法 [§B.2]
7. **规模较小**: 1B 参数 + 165K h 数据,在 2025 标准下偏小 (CosyVoice 3: 1M+ h, GLM-TTS: 100K h 但 1.5B params)

## 点评

UniAudio 的核心贡献是验证了 "一个模型覆盖多种音频生成任务" 的可行性,并提出了 multi-scale transformer 解决 RVQ 长序列问题。作为 2023 年的工作,它在理念上领先于同期大多数 task-specific 系统。

Multi-Scale Transformer 的设计思路 (global inter-frame + local intra-frame) 简洁有效,后续被 Moshi 等系统采用类似思路。统一 [condition, target] 序列格式的简洁性也值得借鉴。

但从当前视角看,UniAudio 的规模 (1B params, 165K h) 已显不足,且其 codec 是自建的 general-purpose 版本,在特定任务上不如 domain-specific tokenizer (如 Survey 中 ESPnet EnCodec speech-only 在 TTS 上的优势)。联合训练的互利效应虽然存在 (Appendix C.1),但在音乐和 SE 等任务上的性能仍有差距。

总体而言,UniAudio 是 universal audio generation 方向的重要开创性工作,其 multi-scale transformer 和统一 task formulation 的设计对后续研究有深远影响。

## 可复用的 idea

1. **Multi-Scale Transformer**: Global (frame-level) + Local (token-level) 分离处理 RVQ 长序列,O(T²) 替代 O((T*n_q)²)
2. **统一 [condition, target] 序列格式**: 用特殊 token 标记任务/模态/边界,一个模型处理多任务
3. **联合多任务训练**: 不同音频任务共享 audio representation 带来互利,数据效率更高
4. **两阶段扩展**: Training (核心任务) + Fine-tuning (新增任务),re-sampling 保持旧任务性能
5. **通用 audio codec**: 跨 speech/sound/music/singing 的统一 tokenizer

---

检索命中: [[LLM-basedTTS]], [[SpeechTokenizer]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]], [[SpeechLanguageModel]] | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无
