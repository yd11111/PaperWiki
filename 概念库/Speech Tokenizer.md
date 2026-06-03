---
type: concept
title: "Speech Tokenizer"
aliases: [语音分词器, Semantic Token, Discrete Speech Token]
category: "representation"
tags: [speech-representation, tokenization, discrete-token, TTS]
key_papers: ["[[论文笔记/VALL-E|VALL-E, "[[论文笔记/UDDETTS|UDDETTS]]"]]", "[[论文笔记/Spark-TTS|Spark-TTS]]", "[[论文笔记/SoundStream|SoundStream]]", "[[论文笔记/CosyVoice|CosyVoice]]", "[[论文笔记/CosyVoice 2|CosyVoice 2]]", "[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/NaturalSpeech 3|NaturalSpeech 3]]", "[[论文笔记/SoundStorm|SoundStorm]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Seed-TTS|Seed-TTS]]", "[[论文笔记/Survey-Discrete Audio Tokens|Survey-Discrete Audio Tokens]]", "[[论文笔记/AudioLM|AudioLM]]", "[[论文笔记/SPEAR-TTS|SPEAR-TTS]]", "[[论文笔记/HuBERT|HuBERT]]", "[[论文笔记/Whisper|Whisper]]", "[[论文笔记/Fish-Speech|Fish-Speech]]", "[[论文笔记/GLM-TTS|GLM-TTS]]", "[[论文笔记/UniAudio|UniAudio]]", "[[论文笔记/FireRedTTS|FireRedTTS]]", "[[论文笔记/FireRedTTS 2|FireRedTTS 2]]", "[[论文笔记/BASE TTS|BASE TTS]]", "[[论文笔记/Moshi|Moshi]]", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/wav2vec 2.0|wav2vec 2.0]]", "[[论文笔记/WavLM|WavLM]]", "[[论文笔记/w2v-BERT|w2v-BERT]]", "[[论文笔记/SNAC|SNAC]]", "[[论文笔记/RepCodec|RepCodec]]", "[[论文笔记/VQ-VAE|VQ-VAE]]", "[[论文笔记/LatentLM|LatentLM]]", "[[论文笔记/CLEAR|CLEAR]]", "[[论文笔记/VibeVoice|VibeVoice]]", "[[论文笔记/FlexiCodec|FlexiCodec]]", "[[论文笔记/StableToken|StableToken]]", "[[论文笔记/SiTok|SiTok]]", "[[论文笔记/NAC Token Language Analysis|NAC Token Language Analysis]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/XEUS|XEUS]]", "[[论文笔记/BEATs|BEATs]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/VoXtream|VoXtream]]", "[[论文笔记/Voxtral TTS|Voxtral TTS]]", "[[论文笔记/Fish Audio S2|Fish Audio S2]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/Cont-SPT|Cont-SPT]]", "[[论文笔记/TADA|TADA]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/OmniVoice|OmniVoice]]", "[[论文笔记/NAST|NAST]]", "[[论文笔记/Vec-Tok Speech|Vec-Tok Speech]]", "[[论文笔记/PALLE|PALLE]]", "[[论文笔记/GOAT-TTS|GOAT-TTS]]", "[[论文笔记/LLMVoX|LLMVoX]]", "[[论文笔记/OpenOmni|OpenOmni]]", "[[论文笔记/IndexTTS|IndexTTS]]", "[[论文笔记/LLaMA-Omni 2|LLaMA-Omni 2]]", "[[论文笔记/LM-SPT|LM-SPT]]", "[[论文笔记/UniTTS|UniTTS]]", "[[论文笔记/C2F-LM|C2F-LM]]", "[[论文笔记/SpeechAccentLLM|SpeechAccentLLM]]"]
origin_paper: ""
related_concepts: ["[[Finite Scalar Quantization]]", "[[Conditional Flow Matching]]", "[[Residual Vector Quantization]]", "[[Semantic vs Acoustic Tokens]]", "[[Speech Language Model]]", "[[Codec Language Model]]", "[[Audio Tokenizer Taxonomy]]", "[[Token Rate and Bitrate Trade-offs]]", "[[Single-codebook vs Multi-codebook]]", "[[Codec Training Objectives]]", "[[Self-Supervised Speech Representation]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Speech Tokenizer 将连续语音波形转换为离散 token 序列的模块,是 LLM-based TTS 系统的关键组件。根据训练方式可分为三类:

1. **自监督 tokenizer**: 如 HuBERT、W2v-BERT 2.0,通过 masked prediction 学习表征后做 k-means 聚类
2. **监督式 semantic tokenizer**: 如 CosyVoice 系列,通过 ASR 等下游任务监督训练,token 主要编码语义/语言信息
3. **声学 tokenizer**: 如 SoundStream、EnCodec,通过 RVQ 重建波形,token 编码全部声学信息

## 在 TTS 中的应用

在 coarse-to-fine TTS pipeline 中,semantic speech tokenizer 处于核心位置:
- **编码时**: 将参考语音 → token 序列(训练目标)
- **解码时**: LLM 生成 token 序列 → CFM/vocoder 合成语音

CosyVoice 3 的 speech tokenizer 基于 MinMo 构建,通过 FSQ 量化,以 25 Hz token rate 工作。其监督多任务训练(ASR + LID + SER + AED + SA,共 530K 小时)使 token 富含副语言信息,同时排除底层声学细节,让 CFM 可以独立控制音色。

关键 trade-off: token 编码越多语义信息(排除声学)→ speaker similarity 越高(CFM 从 prompt 学音色)+ content consistency 越好;但也可能丢失韵律细节。

## 关键论文

- CosyVoice 3 (2025): 监督多任务 FSQ-MinMo tokenizer
- CosyVoice (2024): FSQ-SenseVoice tokenizer
- HuBERT (2021): 自监督 speech representation
- SoundStream (2021): 声学 RVQ tokenizer
- DAC (Kumar et al., NeurIPS 2023): 改进 RVQ-GAN 的声学 codec,可作为 drop-in replacement 用于 AudioLM/VALL-E/MusicLM 等生成模型的 audio tokenizer,因更高 bitrate efficiency (99%) 和保真度提升下游生成质量
- IndexTTS2 (Zhou et al., 2025): 采用 MaskGCT 的 semantic codec 作为 speech tokenizer,在 T2S 模块中生成 semantic token 序列,并通过共享位置编码表(W_sem = W_num)实现精确 duration control
- MaskGCT (Wang et al., 2024): 提出 VQ-VAE semantic codec,用单层 codebook (8192 entries, dim 8) 量化 W2v-BERT 2.0 第 17 层 hidden states,相比 k-means 保留更多韵律信息,被后续 IndexTTS2 等采用
- [[论文笔记/Seed-TTS|Seed-TTS]] (ByteDance, 2024): 探索 continuous 和 discrete 两种 speech tokenizer 设计,发现 tokenizer 是全系统性能瓶颈;使用类似 Betker (2023) 的方案,在大规模数据上验证了 tokenizer 质量对零样本 TTS 的决定性影响
- [[论文笔记/FireRedTTS|FireRedTTS]] (小红书, 2024): 提出 Semantic-Aware Speech Tokenizer (SAST),结合 HuBERT semantic encoder + ECAPA-TDNN acoustic encoder (Clip&Shuffle 防 content 泄漏) + VQ decoder (codebook 16384, frameshift 40ms),属于自监督 semantic + 全局 acoustic embedding 的混合路线
- [[论文笔记/FireRedTTS 2|FireRedTTS 2]] (小红书, 2025): 重新设计为 12.5Hz streaming tokenizer,结合 Whisper encoder (semantic) + 可训练 acoustic encoder → concat + 4x downsample → 16 层 RVQ (2048 entries) + dual decoder (semantic + Vocos-based acoustic);帧率仅为主流方案的一半,适用于长对话场景

## 相关概念

- [[Finite Scalar Quantization]]: CosyVoice 系列使用的量化方法
- [[Conditional Flow Matching]]: tokenizer 的下游,从 token 恢复声学细节
- BPE Text Tokenizer: 文本侧的 tokenizer,speech tokenizer 是其语音对应物

## SpeechLM 视角下的三类体系

Cui et al. (2024) 从 SpeechLM 角度将 speech tokenizer 重新分类为三类 (Figure 3):

### Mixed Objective Tokenizer
兼顾语义理解和声学生成的第三类 tokenizer,目前处于早期阶段但前景显著:
- **SpeechTokenizer** (Zhang et al., ICLR 2024): 采用 RVQ-GAN 架构,但将第一层 RVQ 通过蒸馏对齐 HuBERT 语义表征,后续层量化声学残差。实现了单一 tokenizer 同时编码高层语义和底层声学。
- **Mimi** (Defossez et al., 2024, Moshi): 使用单个 VQ 模块提取语义信息 (来自 WavLM),外加额外 RVQ 模块编码声学信息。被 Moshi 全双工系统采用。
- **LM-SPT** (Jo et al., 2025): 在 Mimi Split RVQ 基础上引入 reconstruction-driven semantic distillation (用 Whisper ASR encoder 对比原始与语义重建波形的表征) + dual encoder 架构,在 25/12.5/6.25 Hz 三帧率下重建保真度和 TTS 下游性能全面超越 SpeechTokenizer/Mimi baseline。

这种混合路线在 SpeechLM 中越来越受关注,因为它避免了 semantic-only 或 acoustic-only 的局限性。详见 [[Semantic vs Acoustic Tokens]]。

### 在 SpeechLM 中的角色分布
Survey (Table II) 统计了 50+ SpeechLM 系统的 tokenizer 选择:
- **Whisper encoder** (最流行): Kimi-Audio, Qwen2.5-Omni, Mimmo, Lyra, Flow-Omni, SLAM-Omni, Mini-Omni 2, IntrinsicVoice
- **HuBERT**: SynCLLM, SpeechGPT, dGSLM, SUTLM, pGSLM, GSLM, TWIST, PSLM
- **SpeechTokenizer**: SpeechGPT-Gen, ICoT, AnyGPT
- **EnCodec**: GPST, VoiceBox, VioLA, UmAudio
- **w2v-BERT**: AudioLM

## Survey 五轴 Taxonomy 视角 [Mousavi et al. 2025]

Mousavi et al. (2025) 提出了比 semantic/acoustic/mixed 三分法更精细的五轴分类体系 (详见 [[Audio Tokenizer Taxonomy]]):
1. **Encoder-Decoder 架构**: CNN / CNN+RNN / Transformer / CNN+T
2. **量化方法**: RVQ / SVQ / GVQ / MSRVQ / CSRVQ / PQ / FSQ / K-means
3. **训练范式**: Separate (post-training) vs Joint (end-to-end); 训练目标组合
4. **目标领域**: Speech / Music / General Audio / Multi-domain
5. **流式能力**: Streamable / Non-streamable

Survey Table 1 覆盖 50+ tokenizer 的完整设计参数矩阵,是选型的重要参考。

### Survey Benchmark 关键结论 [§3]

| 评估维度 | 最优 tokenizer | 关键发现 |
|---------|---------------|----------|
| 判别式下游 (ASR等) | Discrete WavLM | SSL semantic tokenizer 在 phonetic 任务上领先 |
| Speaker 保持 | DAC | 重建目标保留 speaker identity 更好 |
| TTS | ESPnet EnCodec (speech-only) | domain-specific 训练关键 |
| SLM 语义 | HuBERT 25Hz | SSL tokenizer 语义理解仍最强 |
| SLM 声学 | WavLM (DWavL-S-16) | 最佳声学属性建模 |

**核心发现**: "no single tokenizer excels across all spoken and acoustic tasks" [§3.3.1] — 没有万能 tokenizer。

## NAC Token 的语言学统计特性 [Park et al., 2025]

Park et al. (2025) 首次系统分析了 neural audio codec (NAC) token 序列的语言学统计特性。核心发现:
- **3-gram NAC tokens 最接近自然语言分布**: 在 Zipf alpha, Heaps' beta/k, entropy/redundancy 上,3-gram 级别的 NAC tokens 与自然语言词汇最相似 [Fig 2-6]
- **"更像语言的 token = 更好的语音"**: alpha↓ (更 Zipfian), beta↑ (更线性 vocab growth), redundancy↓ 与 WER↓ / UTMOS↑ 正相关 [Fig 7-9]
- **高维 codec (大 n_d) 更接近自然语言**: dimension size 是影响统计特性的关键因素

这些发现有 tokenizer 设计启示: NAC token 的 Zipf/Heaps 统计量可作为 tokenizer 质量的快速 proxy 指标。详见 [[论文笔记/NAC Token Language Analysis|NAC Token Language Analysis]]。

## 演进

Mel spectrogram (传统 TTS) → VQ-VAE acoustic tokens (2019) → HuBERT semantic tokens (2021) → 监督式 semantic tokens (CosyVoice, 2024) → Mixed tokenizer (SpeechTokenizer/Mimi, 2024) → 多任务监督 + 大模型 backbone (CosyVoice 3, 2025) → 五轴精细化 taxonomy + 统一 benchmark (Mousavi et al., 2025) → **Continuous VAE tokenizer** (sigma-VAE, LatentLM 2024; shortcut-VAE, CLEAR 2025): 绕过离散量化,直接用 VAE 编码为连续 latent vectors,压缩比可达 1600-6400x (帧率 3.75-15 Hz),重建质量优于同压缩比离散方案

### Continuous VAE Tokenizer (连续 tokenizer 新路线)

以 LatentLM (Sun et al., 2024), CLEAR (Wu et al., 2025), VibeVoice (Peng et al., 2025) 为代表的新路线**完全绕过离散量化**:

- **sigma-VAE** (LatentLM): 固定 variance sigma ~ N(0, C_sigma) 防止 AR 场景下的 variance collapse; ConvNeXt encoder; 压缩比 1600x/3200x/6400x; PESQ 3.068 / UTMOS 4.181 at 7.5 Hz [LatentLM Table 5]
- **Enhanced wav-VAE** (CLEAR): 7-stage oobleck encoder + snake activation + **非参数化 shortcut connections** (space-to-channel); 压缩比 2048x (strides [2,4,4,8,8]); WER 2.89%, UTMOS 4.08 接近 ground truth [CLEAR Appendix C.1]
- **VibeVoice tokenizer**: 基于 sigma-VAE, hierarchical Transformer blocks (7 stages), 1D causal convolution, 3200x 压缩 (7.5 Hz), 340M params; PESQ 3.068, UTMOS 4.181 领先所有 multi-quantizer baseline [VibeVoice Table 3]

**关键优势**: speech-to-text token ratio 约 2:1 (接近 BPE 粒度), 使 LLM 可自然处理语音和文本的交错序列。90 分钟对话仅需 ~40K tokens (传统 50Hz codec 需 ~270K)。

- **Cont-SPT** (Li et al., NAACL 2025 Findings): 最简洁的连续 tokenizer 方案 — 直接去掉 RVQ,保留 codec encoder 的连续 embedding 输出作为 speech token。VAE-like 预训练 (重建 + ASR CTC loss) + 与 LM 联合训练 (LR=0.05x)。频域分析证明连续 tokenizer 在高频段信息保留显著优于离散版 (8kHz: 0.55 vs 0.34)。在 LibriSpeech 上 WER 6.59% / SIM 0.73 优于 VALL-E baseline
