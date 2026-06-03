---
type: paper
tier: deep
title: "TraceableSpeech: Towards Proactively Traceable Text-to-Speech with Watermarking"
arxiv_id: "2406.04840"
source: "Sources/TraceableSpeech.pdf"
authors: [Junzuo Zhou, Jiangyan Yi, Tao Wang, Jianhua Tao, Ye Bai, Chu Yuan Zhang, Yong Ren, Zhengqi Wen]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, watermarking, proactive-traceability, neural-codec, codec-LM, security, speech-synthesis]
concepts: ["[[Codec Language Model]]", "[[Residual Vector Quantization]]", "[[Anti-spoofing and Deepfake Detection]]", "[[TTS Evaluation]]", "[[Speaker Embedding]]"]
models: ["[[EnCodec]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 3 个待确认实体页: [[Residual Vector Quantization]]✓, [[Speaker Embedding]]✓, [[Codec Language Model]][待确认], [[Anti-spoofing and Deepfake Detection]][待确认], [[TTS Evaluation]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Residual Vector Quantization]]✓, [[Speaker Embedding]]✓ | 参考: [[Codec Language Model]](pending-review), [[Anti-spoofing and Deepfake Detection]](pending-review), [[TTS Evaluation]](pending-review) | 未命中但可能相关: Audio Watermarking(概念库中无独立页)

**谱系定位**: TraceableSpeech 处于 Codec Language Model (VALL-E 范式) 与 Anti-spoofing/Deepfake Detection 的交叉地带。在 Codec LM 页中,VALL-E 是首个大规模 codec LM TTS,使用 EnCodec tokens + AR/NAR 两阶段生成;TraceableSpeech 在此基础上将水印机制嵌入 codec 训练,属于 codec LM 的安全性扩展。Anti-spoofing 页提到了 watermarking 作为 proactive traceability 手段,但缺乏技术细节,TraceableSpeech 是该方向的具体实现。TTS Evaluation 页的 Security & Traceability 一节直接引用了本文 (TraceSpeech [Zhou et al., 2024]) 作为 imperceptible audio watermarking 的代表工作。

**已有认知**: RVQ 页已确认 HiFiCodec 使用 Group VQ (GVQ) 变体,1 group × 8 codebooks。Speaker Embedding 页记录了 ResNet 作为 speaker encoder 架构之一。Codec LM 页记录了 VALL-E 的 AR+NAR 两阶段框架。

**创新判断**: 本文的核心创新不在 TTS 架构 (沿用 VALL-E + HiFiCodec),而在于将水印嵌入从"生成后处理"变为"生成过程内联"(joint optimization),以及 frame-wise broadcast 实现时域灵活性。这是 proactive speech traceability 在 codec LM TTS 中的首次端到端实现。

> [!summary] 速查
> - **一句话**: 将水印嵌入从 TTS 的后处理阶段前移到 codec 联合训练阶段,通过 frame-wise broadcast 实现对任意时长语音的鲁棒水印
> - **路线**: Text → Phoneme → VALL-E (AR+NAR) → Discrete Codes → Imprint Module (Watermark Encoder broadcast + merge with latent z) → Speech Decoder → Watermarked Speech; 提取: Mel-spectrogram → ResNet → r-vector → Linear layers → Predicted Watermark
> - **指标**: PESQ 3.641 (4@10) / 3.569 (4@16) vs baseline 3.197 [Table 1]; MOS 3.959 (4@10) vs 3.554 baseline [Table 2]; 提取精度在双重 resplicing 攻击后仍 100% (4@10) vs 49-86% baseline [Table 3]; 4@64 在 0.3s 语音上精度 >95% [Table 4]
> - **可借鉴**: (1) 水印与 codec 端到端联合训练消除误差累积的思路,可推广到其他需要在生成过程中嵌入元信息的场景; (2) frame-wise broadcast 将固定维度的信息均匀扩展到可变长序列的方法
> - **局限**: 仅在 LibriTTS 上实验; 语音质量仍低于无水印 codec (未报告无水印 HiFiCodec 重建指标作对照); 水印容量有限 (最高 4@64 ≈ 24 bit); 未讨论对 vocoder 攻击 (如重新合成) 的鲁棒性; 仅英文

## 核心问题

当前 TTS 系统面临的安全威胁 (deepfake、版权侵犯) 要求对合成语音进行溯源 [§1]。现有方案的核心矛盾:

1. **后处理水印损害质量** — 传统做法是先生成语音再叠加水印,两步串联导致误差累积,水印不可察觉性和语音质量双双下降 [§1]
2. **时域灵活性差** — WavMark 只能在固定 1 秒片段上操作,TTS 输出时长不可预知,强行适配导致可用容量减半 (32→16 bit) 且短语音鲁棒性差 [§1]
3. **Resplicing 攻击脆弱** — WavMark 采用固定间隔重复嵌入同一水印,一旦被裁剪拼接,pattern bits 被破坏即无法提取 [§1]

TraceableSpeech 的核心洞察: 如果水印嵌入发生在 codec 的 latent space 而非 waveform,就能与语音合成端到端联合优化,同时利用 frame-level 粒度实现时域灵活性 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统分为两个训练阶段 [§2.1, Fig 1, Fig 2]:

**Stage 1: Neural Codec + Watermarking 联合训练** [Fig 1]
- Speech Encoder (来自 HiFiCodec): 波形 x ∈ R^T → 下采样 240× → latent z ∈ R^{t×512}
- Watermark Encoder: m-digit base-b 水印 → embedding (b×16 weight matrix per digit) → 拼接 → 两层 Linear → wo ∈ R^{1×512}
- Imprint Module: wo 沿时间轴 broadcast → w ∈ R^{t×256},与 z merge
- Speech Decoder: 从 merge 后的特征解码出带水印的语音
- Watermark Extractor: Mel-spectrogram → ResNet → r-vector → m 组双层 Linear → softmax → 预测水印
- Attack Simulation + Discriminators: 端到端训练

**Stage 2: VALL-E Language Model** [Fig 2]
- 与原始 VALL-E 结构相同,训练方式也相同 [§2.3.2] [论文原文]
- 推理时: LM 预测 discrete codes → Imprint Module 嵌入水印 → Speech Decoder 合成带水印语音

[agent 解读] 将水印嵌入点选在 codec latent space 而非 waveform,是因为 latent z 是连续高维特征,信息冗余度高于离散 waveform,更适合承载额外信息且不影响可感知质量。

### 关键设计选择

**1. Frame-wise Broadcast (核心创新)** [§2.2]

传统做法 (WavMark): 将水印向量通过 Linear 直接拉伸到与波形等长 → 训练时长固定,推理时只能处理等长片段 [论文原文]。

TraceableSpeech 的做法: 水印编码为固定维度 wo ∈ R^{1×512},通过 broadcast 沿时间轴复制到每一帧 → w ∈ R^{t×256} [§2.2]。

WHY: broadcast 保证了三个性质 [论文原文]:
- **时域灵活性**: t 可以是任意值,因此支持任意时长语音 [§2.2]
- **均匀分布**: 每一帧都携带完整水印信息,即使部分帧被截断,剩余帧仍可提取 [§2.2]
- **精确控制**: 可以通过控制 broadcast 的起止位置,精确选择哪些段被水印化 [§2.2]

**2. ResNet-based r-vector 提取** [§2.3.1, Fig 1(b)]

提取器输入为 Mel-spectrogram (而非 waveform) [论文原文]。使用 ResNet (实验中为 ResNet34/101) 提取 r-vector,然后接 m 组独立的双层 Linear → softmax 预测每一位水印 [§2.3.1]。

[agent 解读] 将提取器建立在 Mel-spectrogram 上而非 waveform 上可能是为了鲁棒性: Mel 表征对小幅波形扰动 (如加噪、重采样) 有天然的抗干扰能力。使用 ResNet 提取全局 r-vector 类似于 speaker verification 中提取 speaker embedding 的做法,将时变信号压缩为固定向量。

**3. 攻击模拟 (Attack Simulation)** [§2.3.2]

训练时随机施加 7 种攻击之一 [§2.3.2]:
- Normal (无攻击), Resample 90%, White Noise SNR=35dB, Sample Dropout 0.1%, Amplitude Reduce 90%, Echo Addition, Low-pass 5kHz
- 权重分配: Normal 0.45, Noise 0.25, Echo 0.14, 其余各 0.04 [§2.3.2]

WHY 权重不均: TraceableSpeech 对 Noise 和 Echo 攻击更敏感,因此训练中增大其权重 [论文原文]。

### 训练策略

**Stage 1 Loss** [§2.3.2, Eq 1-2]:

$$L = \lambda_f L_f + \lambda_g L_g + \lambda_{feat} L_{feat} + \lambda_{qz} L_{qz} + \lambda_c L_c$$

- L_f: 频域重建损失 (与 HiFiCodec 相同)
- L_g: GAN adversarial loss (generator)
- L_feat: feature matching loss
- L_qz: 量化损失
- L_c: 水印交叉熵损失 (Eq 1)

超参设置: λ_f=1, λ_g=1, λ_feat=1, λ_qz=10, λ_c=5 [§2.3.2]

[agent 解读] λ_qz 和 λ_c 的权重远高于其他项,说明训练策略显著偏向量化精度和水印嵌入质量,这与将水印视为一等公民的设计理念一致。

**Stage 2**: 与 VALL-E 训练完全相同,AR 20 epochs + NAR 40 epochs [§3.2]。

## 实验

### 实验设置 [§3.1-3.2]
- 数据集: LibriTTS (585h, 2456 speakers, 24kHz) [§3.1]
- Codec: 1 group, 8 codebooks, 训练数据截断至 0.5s, 150k steps [§3.2]
- LM: 最大 batch duration 100 [§3.2]
- Baseline: VALL-E/HiFiCodec + WavMark (16-bit binary, 实际可用仅 16 bit) [§3.2]

### 水印不可察觉性 (语音重建) [Table 1]

| 指标 | HiFiCodec+WavMark(16bit) | TraceableSpeech(4@10) | TraceableSpeech(4@16) | 出处 |
| --- | --- | --- | --- | --- |
| PESQ ↑ | 3.197 | **3.641** | 3.569 | [Table 1] |
| STOI ↑ | 0.947 | **0.950** | 0.948 | [Table 1] |
| ViSQOL ↑ | 3.880 | **4.060** | 3.985 | [Table 1] |

### 零样本语音合成质量 [Table 2]

| 指标 | VALL-E+WavMark(16bit) | TraceableSpeech(4@10) | TraceableSpeech(4@16) | 出处 |
| --- | --- | --- | --- | --- |
| WER(%) ↓ | 10.80 | **9.61** | 10.47 | [Table 2] |
| MOS ↑ | 3.554±0.19 | **3.959±0.18** | 3.905±0.17 | [Table 2] |

### 水印提取精度 — 攻击鲁棒性 [Table 3]

| 模型 | Resplicing | Normal | RSP-90 | Noise-W35 | SD-01 | AR-90 | EA-0315 | LP5000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VALL-E+WavMark | No | 100 | 99.76 | 91.41 | 100 | 100 | 94.53 | 100 |
| TraceableSpeech(4@10) | No | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| VALL-E+WavMark | Once | 91.10 | 91.46 | 63.53 | 95.95 | 93.61 | 88.58 | 89.66 |
| TraceableSpeech(4@10) | Once | 100 | 100 | 100 | 99.90 | 100 | 100 | 100 |
| VALL-E+WavMark | Twice | 76.65 | 77.74 | 49.14 | 79.47 | 85.46 | 68.19 | 75.32 |
| TraceableSpeech(4@10) | Twice | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

[Table 3] 出处

### 容量与时长分析 [Table 4]

| 模型 | 1.0s | 0.5s | 0.3s | 0.2s | 0.1s |
| --- | --- | --- | --- | --- | --- |
| TraceableSpeech(4@32) | 100 | 99.74 | 99.23 | 94.13 | 50.51 |
| TraceableSpeech(4@64) | 100 | 99.86 | 95.57 | 80.59 | 17.01 |

[Table 4] 出处。无攻击模拟条件下测试。4@64 在 0.3s 上仍 >95% [§3.5]。

## 局限性

1. **缺少无水印基准对照** — Table 1 比较的是 HiFiCodec+WavMark vs TraceableSpeech,但未报告无水印 HiFiCodec 的重建指标,无法量化水印嵌入本身对质量的影响 [agent 解读]
2. **仅英文单数据集** — 所有实验在 LibriTTS 上完成,未验证跨语言/跨数据集泛化性 [§3.1]
3. **水印容量有限** — 最高 4@64 ≈ 24 bit,对于需要嵌入复杂元数据 (如模型 ID + 时间戳 + 用户 ID) 的场景可能不足 [agent 解读]
4. **未讨论高级攻击** — 未考虑 re-synthesis 攻击 (将带水印语音经另一 TTS 重新合成)、neural codec re-encoding、对抗性扰动等现代攻击手段 [agent 解读]
5. **MOS 评估规模有限** — 仅 7 名评估者,MOS 统计显著性存疑 [§3.4]
6. **与 codec 绑定** — 水印机制与 HiFiCodec 的 encoder-decoder 结构深度耦合,迁移到其他 codec (如 EnCodec, DAC) 需要重新训练 [agent 解读]

## 点评

TraceableSpeech 提出了一个清晰且有说服力的思路: 将水印从 TTS 的"后处理附件"变为"生成过程的一等公民"。这种端到端联合优化消除了传统后处理水印的误差累积问题,实验结果在所有指标上均优于后处理基线。

frame-wise broadcast 是一个优雅的工程设计: 通过在 latent space 的帧级别复制水印信息,同时解决了时域灵活性和 resplicing 鲁棒性两个问题。特别是在双重 resplicing 攻击下,4@10 模型仍保持 100% 提取精度,而 baseline 已降至 49-86% [Table 3],差距悬殊。

但本文也有明显的工程导向特征: 它没有深入分析水印信息在 latent space 中的编码方式、水印与语音信息的交互机制、以及水印容量的理论上界。此外,缺少无水印 codec 的对照实验使得"联合训练不损害质量"的论断缺乏直接证据。

从领域视角看,proactive speech traceability 是一个重要但尚未成熟的方向。本文是 codec LM TTS 框架下的首次端到端尝试,具有开创意义,但距离实用 (容量、多攻击鲁棒性、跨模型泛化) 仍有距离。

## 可复用的 idea

1. **Codec-watermark 联合训练**: 将需要嵌入的辅助信息 (不限于水印,也可以是情感标签、语言 ID 等) 作为 codec 训练的一部分,与重建损失联合优化,避免后处理引入的质量退化
2. **Frame-wise broadcast**: 将固定维度的控制信号 (条件信息) 均匀复制到每一帧,是一种简单有效的 conditioning 策略,兼顾可变长度和局部完整性
3. **ResNet r-vector 提取**: 借鉴 speaker verification 的全局 embedding 提取思路做信息恢复,在需要从变长信号中提取固定信息的场景通用适用

---

> [!review] 审阅结论: pass (2026-06-03)
> 五个原则均满足, 3 个 low issue (template-compliance ×1, traceability-gap ×2) 均不阻塞.
> 详见 `_review/TraceableSpeech-review.yml`

检索命中: [[Residual Vector Quantization]]✓, [[Speaker Embedding]]✓ | 参考: [[Codec Language Model]](pending-review), [[Anti-spoofing and Deepfake Detection]](pending-review), [[TTS Evaluation]](pending-review) | 未命中但可能相关: Audio Watermarking(概念库中无独立页)
