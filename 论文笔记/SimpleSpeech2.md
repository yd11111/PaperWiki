---
type: paper
tier: deep
title: "SimpleSpeech 2: Towards Simple and Efficient Text-to-Speech with Flow-based Scalar Latent Transformer Diffusion Models"
arxiv_id: "2408.13893"
source: "Sources/SimpleSpeech2.pdf"
authors: [Dongchao Yang, Rongjie Huang, Yuanyuan Wang, Haohan Guo, Dading Chong, Songxiang Liu, Xixin Wu, Helen Meng]
year: 2024
venue: "arXiv"
tags: [TTS, non-autoregressive, flow-matching, scalar-quantization, speech-tokenizer, sentence-duration, diffusion-transformer]
concepts: ["[[ConditionalFlowMatching]]", "[[FiniteScalarQuantization]]", "[[Non-autoregressiveTTS]]", "[[Classifier-FreeGuidance]]", "[[DurationPredictor]]", "[[SpeechTokenizer]]"]
models: ["[[NaturalSpeech2]]", "[[NaturalSpeech3]]", "[[EnCodec]]", "[[SoundStream]]", "[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: SimpleSpeech 2 处于 NAR TTS 的 flow matching 路线。与 Voicebox/F5-TTS/CosyVoice 系列的 CFM-based TTS 属于同一代际,但在 speech tokenizer 设计上走了一条独特的路: 不使用 RVQ discrete tokens 也不使用 VAE continuous latent,而是用 Scalar Quantization 构造有限离散标量空间作为 flow matching 的目标。这一设计与 [[FiniteScalarQuantization]] 概念页描述的 FSQ 高度相关 — SQ-Codec 的 `round(tanh(z)*S)/S` 公式与 FSQ 的 `round(f(z_i))` 形式相同,但 SimpleSpeech 2 的动机是压缩 diffusion 搜索空间(有限+紧凑),而 FSQ 原始论文的动机是消除 codebook collapse。
>
> **已有认知**:
> - [[ConditionalFlowMatching]]: SimpleSpeech 2 采用 flow matching (linear interpolation schedule) 替代前作 SimpleSpeech 的 DDPM。KB 中已有大量 CFM-based TTS 案例 (CosyVoice/F5-TTS/Seed-TTS 等),但 SimpleSpeech 2 的特殊之处在于 flow matching 的目标空间是有限标量空间而非连续 mel/latent 空间。
> - [[Classifier-FreeGuidance]]: SimpleSpeech 2 提出 noisy ASR labels 等效于 CFG 训练的理论分析,这提供了一个新的 CFG 理论视角。
> - [[DurationPredictor]]: SimpleSpeech 2 探索了 4 种 sentence-level duration predictor,与 KB 中 phone-level duration 的主流方案形成对比,代表了 "粗粒度 duration + diffusion 隐式对齐" 的路线。
> - [[SpeechTokenizer]]: SQ-Codec 是一种新型 speech tokenizer,介于 discrete codec (RVQ) 和 continuous latent (VAE) 之间。
>
> **创新判断**: 相对于 KB 中已有的 flow matching TTS 系统 (大多在连续 mel 或 RVQ latent 空间操作), SimpleSpeech 2 的核心差异化在于: (1) 有限标量空间作为生成目标 (completeness + compactness 双准则); (2) flow matching + SQ regularization 保证输出落在标量空间; (3) Time-MoE 为不同去噪阶段分配专用专家。
>
> 检索命中: [[ConditionalFlowMatching]]✓, [[SpeechTokenizer]]✓ | 过滤: [[FiniteScalarQuantization]][待确认], [[Non-autoregressiveTTS]][待确认], [[Classifier-FreeGuidance]][待确认], [[DurationPredictor]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 scalar quantization codec (SQ-Codec) 将语音压缩到有限紧凑的标量空间,再用 flow-based diffusion transformer 在该空间生成,结合 sentence-level duration 控制实现简单高效的 NAR TTS。
> - **路线**: Text → ByT5 encoder → [Timbre&Style from FACodec] → Flow-based Scalar Transformer Diffusion (LLAMA-style, Time-MoE) → SQ-Codec Decoder → Waveform
> - **指标**: MOS 4.28 (vs GT 3.96), WER 7.55% (Whisper+HuBERT avg), SIM 0.93, RTF 0.25 (25 steps) [Table III]; SQ-Codec PESQ 4.16, STOI 0.95 [Table II]
> - **可借鉴**: (1) Scalar quantization 思路: 用 tanh + round 将连续特征映射到有限离散空间,既保证重建质量又压缩 diffusion 搜索空间; (2) Time-MoE: 按时间步分段使用不同 denoising expert; (3) Noisy label ≈ CFG 的理论分析可为大规模 ASR 数据训练 TTS 提供理论依据; (4) Sentence-level duration 足以控制 NAR TTS 语音长度
> - **局限**: 仅 7K 小时训练数据 (vs 60K+ 主流), speaker similarity 落后于大数据方案; 中文 WER 13.98% (高于 ChatTTS 8.88%); 未开源; 与 SeedTTS/E2TTS 等并发工作的差异化有限

## 核心问题

本文要回答三个研究问题:

1. **什么是好的 speech tokenizer?** 论文提出 completeness (重建质量) 和 compactness (搜索空间大小) 两个准则,认为有限且紧凑的标量空间最适合 diffusion 生成 [§I]。
2. **为什么 ASR 转录的 noisy labels 可以训练好 TTS?** 论文给出理论分析: noisy labels 等价于自然引入 CFG 训练 [§III-E]。
3. **如何设计简单高效的 NAR TTS?** 结合 sentence-level duration (无需 phoneme 对齐) + flow matching (比 DDPM 更快更稳) + LLAMA-style transformer 架构。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SimpleSpeech 2 由三个预训练/冻结的编码器 + 一个可训练的 flow-based diffusion transformer 组成 [§III, Fig 1]:

1. **Text Encoder** (冻结): ByT5,直接从 byte-level 文本提取表征,不依赖 phoneme 转换 [§III-A]
2. **Timbre & Style Encoder** (冻结): 基于 FACodec 的 timbre encoder (全局 embedding) + prosody encoder (3 段 style vectors,通过 segment-based average pooling 提取) [§III-B]
3. **SQ-Codec** (冻结): 将语音编码为标量量化后的 latent,decoder 将 latent 恢复为波形 [§III-C]
4. **Flow-based Scalar Transformer Diffusion** (可训练): LLAMA-style transformer,16 层,hidden 768,32 heads,以 text + speaker + timestep 为条件生成标量 latent [§III-D]

### 关键设计选择

**为什么选 SQ-Codec 而不是 RVQ Codec 或 VAE?** [论文原文]

论文的核心论点是: diffusion 的建模难度与目标空间的大小和复杂度正相关。三种 tokenizer 路线的目标空间对比 [§III-C, §V-C]:

| Tokenizer | 目标空间 | 空间性质 | 建模难度 |
|-----------|---------|---------|---------|
| RVQ Codec (SoundStream) | 连续 encoder 输出 | **无限**,分布复杂 | 高 |
| VAE | 连续 latent | **无限**, 近正态但需 KL 约束 | 中 |
| SQ-Codec | 标量量化后的 latent | **有限**, 2S+1 个离散值 | 低 |

SQ-Codec 的量化操作 [Eq. 1]:
```
h̃ = Tanh(h)          # 映射到 [-1, 1]
s = Round(h̃ * S) / S  # 量化到 2S+1 个离散值
```

当 S=9 时,每个标量维度只有 19 个可能取值 (从 -1 到 1 等间距)。这大幅压缩了 diffusion 的搜索空间。[agent 解读] 这里的关键 insight 与 [[FiniteScalarQuantization]] 相同: 将复杂性从量化层推给 encoder/decoder,量化层本身做到极致简单。

**为什么从 DDPM 换到 Flow Matching?** [论文原文]

SimpleSpeech 1 使用 DDPM,需要 100 步推理。SimpleSpeech 2 换用 flow matching (linear interpolation schedule) [Eq. 2-6],原因: (1) flow matching 的直线轨迹使得更少步数即可高质量生成 (25 步 vs 100 步); (2) DDPM 在标量空间中收敛更慢 [Table VIII: DDPM WER 9.0 vs Flow 7.5]。

Flow matching 目标:
```
x_t = t*x + (1-t)*ε,  t ∈ [0,1]
L_FM = min_θ E_{t, p_t(x)} ||v_θ(x,t) - (x - ε)||²
```

**SQ Regularization** [Eq. 7]: 推理最后一步对输出施加 SQ 操作,确保生成结果落在标量空间内。[agent 解读] 这是一个简单但有效的约束: flow matching 在连续空间做插值,最终 SQ 操作将输出 "snap" 到最近的合法标量值,避免 decoder 接收到 out-of-distribution 的输入。

**为什么用 Time-MoE?** [论文原文]

论文观察到 flow matching 的不同时间步有不同任务特性 [§III-D(4)]:
- t 接近 0 (高噪声): 网络主要从条件信息 (text, timbre) 中提取语义
- t 接近 1 (低噪声): 网络主要精炼声学细节

因此将时间步均匀分为 4 段,每段分配一个专用 expert (FFN 层)。[agent 解读] 这一设计来自图像生成领域的 ERNIE-ViLG 2.0 [59],在 TTS 中的适用性由消融实验验证 (去掉 Time-MoE: WER 8.2 vs 7.5) [Table VIII]。

**Noisy Labels ≈ CFG 训练** [论文原文]

论文证明: 当训练数据中包含少量 noisy labels (ASR 错误转录) 时,这些样本的条件概率 p(y_i|x_i) ≈ 0,相当于自然地进行无条件训练,等效于 CFG 的条件 dropout 机制 [Eq. 9-13, §III-E]。因此无需刻意构造 mask samples。

[agent 解读] 这个分析有其启发性,但有一个关键假设限制: 它假设各 word-piece 是独立的 (Eq. 11),实际中连续文本的相邻词显然不独立。此外,"noisy label" 导致的是固定样本的条件失效,而 CFG 是随机 dropout,两者在优化动态上并不完全等价。论文在 Table IX 中仍然使用了显式 CFG (λ=5 最优),说明 noisy label 的隐式 CFG 效果不足以完全替代显式 CFG。

### 训练策略

- **SQ-Codec**: LibriTTS 数据集训练,S=9, d=32, 200K steps, lr=2e-3 [§IV-B]
- **SimpleSpeech 2 主模型**: 冻结所有编码器和 SQ-Codec,仅训练 transformer diffusion,400K steps, lr=1e-4, cosine scheduler, warmup 1K [§IV-B]
- **训练数据**: 7K 小时英语 (MLS 数据集, Whisper-base 转录); 多语言版本额外加 4K 小时中文 (WenetSpeech) [§IV-A]
- **CFG**: 推理时 λ=5 [Table IX]
- **推理**: 25 步 flow matching, sentence duration 由 ByT5-based predictor 预测 [§IV-B]

**Sentence Duration Predictors** [§III-F]:

论文探索 4 种 sentence-level duration predictor:
1. **ByT5-based** (推荐): 在冻结 ByT5 上加 attention + linear layers,MSE loss 训练 [Fig 4a]
2. **ChatGPT in-context learning**: 调 5 次取平均,不稳定且成本高 [Fig 4b]
3. **FastSpeech 2 teacher**: 用 FS2 的 phoneme duration predictor,逐 phoneme 求和,加随机 scale [0.9, 1.3] [Fig 4c]
4. **AR-based phoneme predictor**: 自回归预测 phoneme duration 后求和 [Fig 4d]

结果: ByT5-based 最优 (WER 7.55), ChatGPT 最差 (WER 9.61) [Table VII]。

[agent 解读] 使用 sentence-level duration 而非 phone-level duration 是 SimpleSpeech 系列的核心设计取舍。优势: 完全避免 phoneme alignment 的复杂性; 劣势: 放弃了对 phoneme-level 韵律的显式控制。这一路线后来被 SeedTTS、E2 TTS 等并发工作验证。

## 实验

### SQ-Codec 重建性能 [Table II]

| 模型 | Size (M) | Bitrate | PESQ | STOI | SSIM | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| EnCodec | 14 | 12 kbps | 3.76 | 0.90 | 0.72 | [Table II] |
| DAC | 70 | 8 kbps | 3.99 | 0.95 | 0.85 | [Table II] |
| SoundStream* | 14 | 4 kbps | 3.25 | 0.90 | 0.77 | [Table II] |
| VAE* | 15 | - | 3.93 | 0.94 | 0.86 | [Table II] |
| **SQ-Codec** | **5** | **8 kbps** | **4.16** | **0.95** | **0.86** | [Table II] |

SQ-Codec 在最小参数量 (5M) 下取得最优 PESQ。

### TTS 主实验 [Table III]

| 模型 | MOS | SMOS | WER (avg) | SIM (avg) | DNSMOS | RTF | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ground Truth | 3.96 | 4.13 | 7.37 | 0.96 | - | - | [Table III] |
| HierSpeech++ | 3.85 | 3.81 | 5.83 | 0.91 | 3.71 | 0.57 | [Table III] |
| NaturalSpeech 3⋆ | 3.96 | 3.89 | 8.96 | 0.93 | 3.48 | 0.73 | [Table III] |
| SimpleSpeech | 3.97 | 3.75 | 13.5 | 0.91 | 3.79 | 1.6 | [Table III] |
| **SimpleSpeech 2** | **4.28** | **3.90** | **7.55** | **0.93** | **3.85** | **0.25** | [Table III] |

SimpleSpeech 2 的 MOS 4.28 甚至超过 GT 3.96,但这可能反映测试集 GT 质量参差 [agent 解读]。与 SimpleSpeech 1 相比: WER 从 13.5 降至 7.55,RTF 从 1.6 降至 0.25,全面提升。

### Tokenizer 影响消融 [Table V, Table VI]

| Tokenizer | WER | SIM | DNSMOS | 出处 |
| --- | --- | --- | --- | --- |
| SoundStream (d=32) | 9.7 | 0.91 | 3.69 | [Table V] |
| VAE (d=32) | 10.5 | 0.91 | 3.77 | [Table V] |
| SQ-Codec (S=9, d=32) | 7.5 | 0.93 | 3.85 | [Table V] |
| SQ-Codec (S=9, d=50) | 18.9 | 0.92 | 3.82 | [Table VI] |
| SQ-Codec (S=9, d=10) | 11.5 | 0.91 | 3.81 | [Table VI] |

关键发现: (1) SQ-Codec 优于 VAE 和 SoundStream [Table V]; (2) d=50 虽然重建更好 (PESQ 4.23),但生成性能大幅下降 (WER 18.9),因为搜索空间变大 [Table VI]; (3) S=9, d=32 是 completeness 和 compactness 的最佳平衡点。

### Architecture/Formulation 消融 [Table VIII]

| 设置 | WER | SIM | DNSMOS | 出处 |
| --- | --- | --- | --- | --- |
| SimpleSpeech 2 (full) | 7.5 | 0.93 | 3.85 | [Table VIII] |
| w/o Time-MoE | 8.2 | 0.92 | 3.80 | [Table VIII] |
| DDPM formulation | 9.0 | 0.92 | 3.82 | [Table VIII] |

Time-MoE 和 flow matching 都有正向贡献。

### CFG 影响 [Table IX]

λ=1 (无 CFG): WER 50.4, DNSMOS 3.58 — 性能灾难性下降。λ=5: WER 7.5, DNSMOS 3.85 — 最优。λ>5: 性能开始退化。这证实 CFG 对 SimpleSpeech 2 的生成质量至关重要。

### Diffusion 步数影响 [Table X]

25 步 (RTF 0.25) 与 50 步 (RTF 0.42) 性能接近 (WER 7.55 vs 7.43, DNSMOS 3.85 vs 3.85),但 5 步 (RTF 0.12) 质量明显下降 (WER 13.85)。25 步是质量-速度的最佳折中。

## 局限性

1. **训练数据规模有限**: 仅 7K 小时,远小于 SeedTTS (60K+)、Voicebox (60K+) 等。论文承认 speaker similarity 与大数据方案有差距,认为"scaling data can eliminate the gap" [§V-B],但未验证。
2. **中文性能不足**: 多语言版本中文 WER 13.98% vs ChatTTS 8.88% [Table XI]。论文归因于 ByT5 直接处理中文 byte 导致多音字问题 [§V-I],但未探索 phoneme 输入的替代方案。
3. **与并发工作的差异化**: SeedTTS、E2 TTS、DiTTo-TTS 均在同期采用类似的 sentence-level duration + NAR diffusion 策略。论文声称"one of the earliest works" [§II-B],但差异化论证不够充分。
4. **SQ-Codec 评估不完整**: 仅在 LibriTTS 16kHz 上评估重建,未验证更高采样率或更多样化数据。[[FiniteScalarQuantization]] 概念页记录了 Survey 发现 FSQ 在 44.1kHz 时性能反而退化的现象,SQ-Codec 是否存在同样问题未知。
5. **Noisy label 理论的假设限制**: 独立性假设 (Eq. 11) 在连续文本中不成立,且论文仍需显式 CFG (λ=5) 才能达到最优性能,说明理论分析的实际贡献有限。
6. **未开源**: 无法复现验证。

## 点评

SimpleSpeech 2 的核心贡献是提出了 "有限标量空间作为 diffusion 生成目标" 的思路,以及围绕这一思路的系统性验证 (SQ-Codec 设计 + tokenizer completeness/compactness 准则 + flow matching 适配)。

**最值得注意的 insight**: 增大 tokenizer latent dimension (d=50) 虽然提升重建质量,却严重损害生成质量 [Table VI]。这说明 speech tokenizer 设计不能只看重建指标,必须同时考虑下游生成模型的建模难度。这一发现对 tokenizer 设计有实际指导意义。

**与 KB 中 FSQ 的关系**: SQ-Codec 的 tanh + round 操作与 FSQ 形式相同,但论文未引用 Mentzer et al. (ICLR 2024) 的 FSQ 工作,也未使用 FSQ 术语。两者动机不同 (FSQ: 消除 codebook collapse; SQ-Codec: 压缩 diffusion 搜索空间),但本质上解决了相同的问题。

**Time-MoE 的贡献**: 消融显示 WER 从 7.5 → 8.2 [Table VIII],贡献实在但不算革命性。这一设计后来在 TTS 领域未见广泛采纳。

**历史定位**: SimpleSpeech 2 (Aug 2024) 与 SeedTTS (Jun 2024)、E2 TTS (Jun 2024)、DiTTo-TTS (Jun 2024) 属于同一波 "sentence-level duration + NAR diffusion" 的浪潮。其中 SeedTTS 在工业界影响最大 (ByteDance 大数据+大算力),E2 TTS 在学术界影响更广 (F5-TTS 开源生态)。SimpleSpeech 2 的 SQ-Codec 方向提供了独特视角,但因未开源+数据规模小,后续影响有限。

## 可复用的 idea

1. **Completeness + Compactness 作为 tokenizer 评估准则**: 不仅看重建质量,还要看生成目标空间的紧凑度。这个双准则可用于任何涉及 tokenizer → generative model 的 pipeline 评估。
2. **SQ Regularization (推理时 SQ 操作)**: 在生成模型输出端施加 domain-specific 约束,确保输出落在合法空间。类似思路可用于其他有限空间的生成任务。
3. **Sentence-level duration 足以工作**: 无需 phoneme-level alignment 即可实现合理的 NAR TTS,大幅简化数据处理。适合大规模 in-the-wild 数据场景。
4. **Time-MoE 的按时间步分专家**: 对于多步生成过程中不同阶段有不同任务特性的场景 (去噪前期重语义/后期重细节),可用 MoE 分工。

## 审阅

> [!review] 审阅 (2026-06-08, self-review)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节包含 WHY 解释 (为什么选 SQ/Flow/Time-MoE),关键设计选择有对比论证 |
> | 可信赖 | pass | 数字标注覆盖充分,指标方向正确,关键实验数据均有 Table 出处 |
> | 可区分 | pass-with-fixes | 因果解释来源标注 [论文原文]/[agent 解读] 基本覆盖,noisy label 分析的批评有标注 |
> | 可定位 | pass | KB 背景有谱系定位 (CFM 路线中的标量空间变体),与 FSQ/CFG/Duration Predictor 的关系明确 |
> | 不污染 | pass | 未新建概念页;反向更新内容 (SQ-Codec 作为 FSQ 变体案例) 属于追加类型 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> - ⚠️ [medium/traceability-gap] 速查卡片"指标"字段列出了 Table III 和 Table II 的数字但未标注 baseline 数据集名 (LibriTTS/VCTK/CommonVoice/RAVDESS/SwitchBoard 混合测试集)
> - ℹ️ [low/weak-reusability] "可复用的 idea" 第 4 条 Time-MoE 在 TTS 后续工作中未见广泛采纳,实际可迁移性待验证
