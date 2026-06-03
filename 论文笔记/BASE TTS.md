---
type: paper
tier: deep
title: "BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data"
arxiv_id: "2402.08093"
source: "https://arxiv.org/abs/2402.08093"
authors: [Mateusz Lajszczak, Guillermo Cambara, Yang Li, Fatih Beyhan, Arent van Korlaar, Fan Yang, Arnaud Joly, Alvaro Martin-Cortinas, Ammar Abbas, Adam Michalski, Alexis Moinet, Sri Karlapati, Ewa Muszynska, Haohan Guo, Bartosz Putrycz, Soledad Lopez Gambino, Kayeon Yoo, Elena Sokolova, Thomas Drugman]
year: 2024
venue: "arXiv (Amazon AGI)"
tags: [TTS, large-scale, LLM-TTS, emergent-abilities, scaling, discrete-speech-tokens, WavLM, BPE, speechcode, streamable, billion-parameter, multilingual]
concepts: ["[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Speaker Embedding]]", "[[Speech Factorization]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[LLM-based TTS]], [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Speaker Embedding]], [[Speech Factorization]])
> BASE TTS 是 [[LLM-based TTS]] 的规模化验证: 将 GPT-2 式 decoder-only Transformer 扩展到 1B 参数 + 100K 小时数据,验证 TTS 中的 "emergent abilities" 假设。其 speechcode 设计与 [[Semantic vs Acoustic Tokens]] 页所述的 token 层级直接相关: BASE TTS 的 WavLM-based speechcodes 是一种 **speaker-disentangled semantic token**,通过 [[Speech Factorization]] 中的对抗训练 (gradient reversal + contrastive loss) 将 speaker identity 从 content representation 中移除,让 speechcode 只编码 phonetic + prosodic 信息。[[Speech Tokenizer]] 页中的"监督式 semantic token"路线 (CosyVoice) 与 BASE TTS 的 SSL-based 路线形成对比: 前者用 ASR loss 监督,后者用 WavLM 自监督表征。
> 检索命中: [[LLM-based TTS]], [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Speaker Embedding]], [[Speech Factorization]] | 过滤: [[Codec Language Model]](pending-review), [[Diffusion-based TTS]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 1B 参数 GPT-2 + 100K 小时公共数据训练的最大 TTS 模型,验证了 TTS 中的 "emergent abilities" (10K+ hours / 500M+ params 时复杂韵律开始涌现) / Largest TTS model at 1B params trained on 100K hours of public data, demonstrating emergent abilities in TTS prosody at scale
> - **路线**: Speech → WavLM → Content Regressor + Speaker Regressor → VQ → Speechcodes (+BPE) → SpeechGPT (GPT-2, 1B, AR, text+speechcodes joint modeling) → Speechcode Decoder (Conv+BigVGAN) → Waveform [§2, Fig 1-3]
> - **指标**: MUSHRA 71.7-73.7 (seen/unseen, vs Tortoise 68.8, Bark 46-49, YourTTS 39-47) [Table 6]; WER 6.5% / SIM 92.7% (vs Bark 19.2/91.7, Tortoise 8.0/90.3, YourTTS 16.3/90.1) [Table 7]; Speechcode decoder 3x faster than diffusion decoder + streamable [§4.5]; Emergent abilities visible at BASE-medium (10K hrs, 400M params) [§4.3, Fig 4]
> - **可借鉴**: (1) WavLM-based speaker-disentangled speechcodes — contrastive loss + gradient reversal 将 speaker info 推到 speaker embedding 侧,speechcode 只保留 phonetic+prosodic [§2.2.2, Fig 2]; (2) BPE on speechcodes — 序列长度减少 40%,使 AR Transformer 可处理更长音频 [§2.2.3]; (3) Speechcode decoder (Conv+BigVGAN) 替代 diffusion decoder — 3x 加速 + streamable + 质量不降 [§2.4, §4.2]; (4) "Emergent abilities" testset: 7 categories of challenging texts 作为 TTS benchmark [§3.3, Table 2]
> - **局限**: 偶有 hallucination/cutoff [§6]; 未开源 (ethical concerns) [§7]; 仅报告一种 speechcode 配置(WavLM)的完整结果 [§6]; ASR 生成的转录有噪声(已通过 text restoration 部分缓解) [§3.1]

## 1. 核心问题与动机 / Core Problem & Motivation

**Why this paper?** BASE TTS (Big Adaptive Streamable TTS with Emergent abilities) 回答一个核心问题: **TTS 是否也存在类似 LLM 的 "emergent abilities"?** [§1]

[论文原文] "We test the hypothesis that this also holds for LTTS, we propose an evaluation scheme to assess potential emergent abilities in TTS, identifying seven categories that are challenging from the literature: compound nouns, emotions, foreign words, paralinguistics, punctuations, questions, and syntactic complexities" [§1]

[agent解读] 这是 TTS 领域首次系统性地验证 scaling law / emergent abilities 假设。与 NLP 中 Wei et al. (2022) 定义的 "abilities that are not present in smaller-scale models but are present in large-scale models" 直接对标。BASE TTS 用三个规模变体 (small/medium/large) 和专门设计的 emergent abilities testset 来验证。

**三大贡献** [§1]:
1. 最大 TTS 模型: 1B params, 100K hours
2. Emergent abilities benchmark: 7 类困难文本
3. WavLM-based speaker-disentangled speechcodes + streamable decoder

## 2. 方法详解 / Method

### 2.1 System Overview [§2.1, Fig 1]

[论文原文] BASE TTS 的三阶段 pipeline [§2.1]:

```
1. Speech tokenizer → Speechcodes (discrete)
2. SpeechGPT → AR generation of speechcodes conditioned on text + reference
3. Speechcode decoder → Waveform (streamable)
```

### 2.2 Discrete Speech Representations [§2.2]

BASE TTS 探索了两种 speechcode:

#### VQ-VAE Speechcodes [§2.2.1]
- 标准 VQ-VAE: convolutional encoder+decoder + VQ bottleneck
- Codebook size 256 for WavLM, 8192 for VQ-VAE [§3.2]
- 25 Hz frame rate → 325 bits/s (VQ-VAE) 或 400 bits/s (WavLM) [§2.2]
- 加入 global reference encoder 部分解耦 speaker [§2.2.1]

[论文原文] "From informal listening, we find that speechcodes produced by the autoencoder-based speech tokenizer still contain speaker information" [§2.2.1] — 这促使了 WavLM-based 方案。

#### WavLM-based Speechcodes [§2.2.2, Fig 2]

[论文原文] 这是 BASE TTS 最重要的技术创新之一。架构 [§2.2.2, Fig 2]:

```
Waveform → WavLM (frozen) → Hidden States
                              ├── Content Regressor → Encoder → VQ → Speechcodes
                              └── Speaker Regressor → Speaker Extractor → Speaker Embedding
                                                       ↑ grad. rev.        ↑ contrastive loss
```

**Speaker disentanglement 三重机制** [§2.2.2]:
1. **Content/Speaker regressors**: 将 WavLM hidden states 分为 content 和 speaker 分支
2. **Contrastive loss**: 同说话人样本 speaker embedding 相似,不同说话人样本远离 [§2.2.2]
3. **Gradient reversal**: content regressor 输出 → frozen speaker extractor → cosine distance 最大化 → 迫使 content 表示不含 speaker 信息 [§2.2.2]

[agent解读] 这三重机制与 [[Speech Factorization]] 中的对抗训练方案直接对应。BASE TTS 的 insight 是: 如果 speechcode 不含 speaker 信息,那么 autoregressive model (SpeechGPT) 就只需要建模 content + prosody,而 speaker identity 可以完全由解码器从 speaker embedding 注入。这大幅简化了 AR model 的建模负担。

**WavLM 的选择理由** [§2.2.2]: [论文原文] "Our intuition is that WavLM was trained with data augmentation to encourage disentanglement from background noise" — WavLM 的自监督预训练本身就有一定的环境噪声鲁棒性。

**训练损失** [§2.2.2, Eq 2]:
```
L = L_recon + alpha * L_commitment + beta * L_contrastive + gamma * L_cosine
```

### 2.3 BPE on Speechcodes [§2.2.3]

[论文原文] "We apply Byte-Pair Encoding to reduce the average sequence length of speechcodes by around 40%" [§2.2.3]

[agent解读] BPE 在 speechcodes 上的应用是新颖的: 将最频繁的 speechcode 对合并为新 token,vocabulary 扩展到预定大小 (8192)。这与 NLP 中的 BPE 完全类似,但作用于语音 token 而非文字。效果: 50 Hz WavLM frames → BPE 后平均 ~30 tokens/s → Transformer 可建模更长音频。

### 2.4 SpeechGPT: Autoregressive Speech Modeling [§2.3]

[论文原文] "We train a GPT2-architecture autoregressive model that we call 'SpeechGPT' to predict the speechcodes conditioned on text and reference speech" [§2.3]

| 变体 | Data | Params | Layers | Dims | Heads |
|------|------|--------|--------|------|-------|
| BASE-small | 1K hrs | 150M | 16, 768 | 3072 | 12 [Table 1] |
| BASE-medium | 10K hrs | 400M | 30, 1024 | 4096 | 16 [Table 1] |
| BASE-large | 100K hrs | 980M | 32, 1536 | 6144 | 24 [Table 1] |

**训练细节** [§3.2]:
- Cross-entropy loss: text weight 0.01, speech weight 1.0 [§2.3]
- 从零训练,无文本预训练 [§2.3]
- 输入: reference speech embedding + text tokens + speechcodes
- 两个 prediction heads: text 和 speech 分开 [§2.3]

[agent解读] 与 VALL-E 的关键区别: (1) SpeechGPT 不做 text 预训练,完全从 speech data 学习; (2) 同时预测 text next token 和 speech next token (multi-task); (3) 用 speaker-disentangled speechcodes 而非 full codec tokens。

### 2.5 Speechcode Decoder [§2.4, Fig 3]

[论文原文] BASE TTS 提出两种 decoder [§2.4]:
1. **Baseline: Diffusion decoder** — UnivNet + vocoder,非流式 [§2.4]
2. **Proposed: Speechcode decoder** — Conv-based + BigVGAN,**流式** [§2.4, Fig 3]

Speechcode decoder 架构 [§2.4]:
```
SpeechGPT last hidden state (50Hz) → Linear → Decoder block (upsample 2x → 100Hz)
                                                                    ↓
Reference speech → Reference encoder → ─────────────────────────── →
                                                                    ↓
                                                      BigVGAN-base → Waveform (24kHz)
```

[论文原文] "We show that this convolution-based speechcode decoder is compute-efficient and reduces the whole-system synthesis time by over 70% compared to the baseline diffusion-based decoder" [§1]

**实际性能** [§4.5]:
- Diffusion decoder: 69.1 seconds / 1000 utterances (~20s each)
- Speechcode decoder: **17.8 seconds** → **3x 加速** [§4.5]
- First-byte latency: **100ms** (vs diffusion: must generate entire sequence) [§4.5]
- Total params for decoder: ~150M [§3.2]

### 2.6 数据准备 [§3.1]

[论文原文] 100K hours 公共领域数据的准备 [§3.1]:
- 英语为主 (>90%), 含 German, Dutch, Spanish [§3.1]
- 24 kHz, 16-bit mono LPCM [§3.1]
- 非录音室条件,含噪声 [§3.1]
- **Text restoration**: 匹配网络源文本到 ASR 转录,恢复约 1/2 文本为 "original" → 引号增加 300%,括号增加 20000% [§3.1]

[agent解读] Text restoration 是一个重要但容易被忽视的细节: ASR 转录几乎不会生成引号、括号等标点,但 TTS 需要理解这些符号来控制韵律。BASE TTS 通过搜索源文本并逐句匹配,大幅提升了标点多样性。这可能是 emergent abilities 涌现的一个隐含条件 — 没有丰富的文本标点,模型可能无法学会问句语调等能力。

## 3. 实验结果与分析 / Experiments

### 3.1 VQ-VAE vs WavLM Speechcodes [§4.1, Table 3]

| Metric | VQ-VAE | WavLM |
|--------|--------|-------|
| Avg MUSHRA (English) | 74.8 | 74.7 [Table 3] |
| Avg MUSHRA (Spanish) | 73.3 | **74.7** [Table 3] |

[论文原文] "For Spanish voices, WavLM based model outperforms the VQ-VAE one in a statistically significant way" [§4.1]

[agent解读] 英语上两者持平,但西班牙语 (仅占训练数据 ~2%) 上 WavLM 显著更好。假设: SSL 预训练带来的跨语言鲁棒性在低资源语言上更有价值。

### 3.2 Diffusion Decoder vs Speechcode Decoder [§4.2, Table 4]

| Speaker | One-shot/seen | Diffusion | Speechcode |
|---------|---------------|-----------|------------|
| Male A | seen | 73.3 | **74.9** [Table 4] |
| Male B | one-shot | 74.6 | **75.5** [Table 4] |
| Female A | seen | 77.9 | 77.4 [Table 4] |
| Female C | one-shot | 69.2 | **71.1** [Table 4] |

[论文原文] "For 4 voices out of 6, the BASE TTS variant with speechcode decoder outperforms the diffusion-based baseline in terms of average MUSHRA score" [§4.2]

[agent解读] Speechcode decoder 不仅 3x 更快,在多数 speaker 上质量还略好。这强烈支持了"两个强生成模型串联是冗余的"(diffusion decoder + vocoder → 合并为一个 decoder)。

### 3.3 Emergent Abilities [§4.3, Table 5, Fig 4]

| Speaker | BASE-small (1K/150M) | BASE-medium (10K/400M) | BASE-large (100K/980M) |
|---------|---------------------|----------------------|----------------------|
| Male A | 61.5 | 70.5 | **72.2** [Table 5] |
| Female A | 68.1 | **72.0** | **72.3** [Table 5] |

**Linguistic Expert Evaluation (7 categories)** [§4.3, Fig 4]:

[论文原文] 关键发现:
1. **BASE-small → medium: 跳跃式提升** — "we see a universal jump from BASE-small to BASE-medium across categories" [§4.3]
2. **BASE-small 几乎无法理解 emotions/paralinguistics/foreign words**: average score < 1.25, never above 1.75 [§4.3]
3. **BASE-medium 已掌握 compound nouns**: "at BASE-medium, the model has mastered compound nouns" [§4.3]
4. **BASE-medium → large: 递减提升**: "continued but diminishing improvement" except compound nouns [§4.3]
5. **Emotions 和 Paralinguistics 最难**: "even BASE-large performs at around an average score of 2.0" [§4.3]

[agent解读] 这是论文的核心结论: TTS 的 emergent abilities 大约在 **10K hours + 400M params** 开始涌现。这与 LLM 的 emergent abilities (10^22 → 10^24 training FLOPs) 形成有趣对应。关键质变指标: compound nouns (phrasal stress) 和 questions (intonation) 在 BASE-medium 时质变,而 emotions 和 paralinguistics 即使在 BASE-large 也仍不完美。

### 3.4 vs Industry Baselines [§4.4, Table 6-7]

| Model | MUSHRA (Male A) | MUSHRA (Female A) |
|-------|----------------|-------------------|
| **BASE TTS** | **71.7-73.7** | **67.8-68.2** [Table 6] |
| Tortoise | 68.8 | 58.5 [Table 6] |
| Bark | - (46.0) | - (49.7) [Table 6] |
| YourTTS | 47.6 | 39.7 [Table 6] |

| | BASE TTS | Bark | Tortoise | YourTTS |
|------|----------|------|----------|---------|
| WER↓ | **6.5** | 19.2 | 8.0 | 16.3 [Table 7] |
| SIM↑ | **92.7** | 91.7 | 90.3 | 90.1 [Table 7] |

[论文原文] "Overall, BASE TTS produces the most natural speech, the least amount of misalignment with input text, and the most similar speech to the reference speaker" [§4.4]

[agent解读] BASE TTS 在 MUSHRA (自然度) 和 WER+SIM (客观指标) 上全面领先所有公开可用的大规模 TTS 系统。特别值得注意: BASE TTS 的 WER 6.5% 远低于 Bark 的 19.2% 和 YourTTS 的 16.3%,说明 scale + speaker disentanglement 显著提升了 intelligibility。

## 4. 与已有方法的差异 / Comparison with Existing Work

| 维度 | BASE TTS | VALL-E | NaturalSpeech 2 | Mega-TTS 2 |
|------|----------|--------|-----------------|-----------|
| 规模 | 1B, 100K hrs | ~300M, 60K hrs | 435M, 44K hrs | 367M, 60K hrs |
| Token 类型 | Speaker-disentangled semantic (WavLM+VQ) | Full codec (EnCodec RVQ) | Continuous latent (RVQ sum) | Content + Prosody VQ codes |
| AR model | GPT-2 (text+speech joint) | AR (1st layer) + NAR (rest) | Non-AR (diffusion) | AR (prosody only) |
| Decoder | Conv+BigVGAN (streamable) | Codec decoder | Codec decoder | HiFi-GAN |
| Speaker 处理 | Disentangled (GRL+contrastive) | In-context (codec prefix) | In-context (speech prompt) | MRTE (multi-ref attention) |
| Streamable | Yes (100ms latency) | No | No | No |
| 多语言 | Yes (EN+DE+NL+ES) | No | No | No |
| Emergent abilities | Systematically verified | Not studied | Not studied | Not studied |

## 5. 局限与未来方向

[论文原文] [§6]:
1. "BASE TTS occasionally produces hallucinations and cutoffs" — AR 固有问题
2. "an inherent problem with the autoregressive LM approach, made worse by the misalignment between audio data and the ASR-generated text" — ASR 转录噪声
3. "Selecting the right discrete representation for GPT-style TTS is crucial" — 仅测试了一种 speechcode 配置

[agent解读] 未说明但重要的局限:
1. 未开源,无法验证 emergent abilities 结论的可复现性
2. Emergent abilities 的评估基于 linguistic expert judgment (主观),可能存在偏差
3. 100K hours 中 90%+ 为英语,多语言能力验证有限(仅 Spanish 结果)
4. 与 VALL-E 的对比是间接的(不同数据/配置)

---

> [!review] 审阅摘要
> - **结论**: pass
> - **主要优点**: 首次系统验证 TTS emergent abilities; speechcode decoder 的实用价值高 (3x加速+流式); speaker disentanglement 方案清晰; 与多个 baseline 直接 MUSHRA 对比
> - **次要建议**: 可补充 BASE-large 在 emergent abilities testset 上的具体数值(论文只提供了图表)

检索命中: [[LLM-based TTS]], [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Speaker Embedding]], [[Speech Factorization]] | 过滤: [[Codec Language Model]](pending-review), [[Diffusion-based TTS]](pending-review) | 未命中但可能相关: 无
