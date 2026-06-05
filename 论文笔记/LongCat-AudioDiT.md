---
type: paper
tier: deep
title: "LongCat-AudioDiT: High-Fidelity Diffusion TTS in the Waveform Latent Space"
arxiv_id: "2603.29339"
source: "Sources/LongCat-AudioDiT.pdf"
authors: [Detai Xin, Shujie Hu, Chengzuo Yang, Chen Huang, Guoqiao Yu, Guanglu Wan, Xunliang Cai]
year: 2026
venue: "arXiv"
tags: [TTS, diffusion, NAR, flow-matching, waveform-latent, VAE, DiT, zero-shot, voice-cloning]
concepts: ["[[ConditionalFlowMatching]]", "[[Classifier-FreeGuidance]]", "[[Diffusion-basedTTS]]", "[[VariationalAutoencoderforTTS]]", "[[MelSpectrogram]]", "[[Non-autoregressiveTTS]]"]
models: ["[[模型库/CosyVoice3|CosyVoice 3]]", "[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[ConditionalFlowMatching]]✓, [[Diffusion-basedTTS]][待确认], [[Classifier-FreeGuidance]][待确认], [[VariationalAutoencoderforTTS]][待确认], [[MelSpectrogram]][待确认], [[Non-autoregressiveTTS]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]], [[Diffusion-basedTTS]], [[Classifier-FreeGuidance]], [[VariationalAutoencoderforTTS]], [[MelSpectrogram]], [[Non-autoregressiveTTS]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: LongCat-AudioDiT 属于 Diffusion-based NAR TTS 中的纯扩散路线。在 KB 谱系中,其位置为:

- **生成范式**: 采用 CFM (Conditional Flow Matching) 框架,与 F5-TTS、Matcha-TTS、VoiceBox 同族。不同于 CosyVoice 系列的 hybrid LLM+Flow 路线,LongCat-AudioDiT 是纯 NAR diffusion 系统。
- **建模目标**: 与几乎所有已有 diffusion TTS 不同(它们以 mel spectrogram 或 Mel-VAE latent 为生成目标),LongCat-AudioDiT 直接在 waveform latent 空间建模。KB 中 [[MelSpectrogram]] 指出 mel 丢失相位信息且需要额外 vocoder,[[VariationalAutoencoderforTTS]] 中 Semantic-VAE 已发现 VAE 维度与 TTS 质量间的非单调关系(dim=64 > dim=16 for WER),与本文的 dimension-capacity trade-off 发现一致。
- **推理改进**: 本文用 Adaptive Projection Guidance (APG) 替代标准 CFG。KB 中 [[Classifier-FreeGuidance]] 记录了 CFG 在高 guidance scale 下的 oversaturation 问题,APG 正是针对此问题的解法(源自图像生成领域)。
- **竞争格局**: KB 中 [[ConditionalFlowMatching]] 记录了 Seed-DiT 作为唯一报告超越 hybrid 方案的纯扩散系统,但架构未公开。LongCat-AudioDiT 填补了这一空白,是首个公开的、在 Seed 基准上 SOTA 的纯 diffusion TTS。

**已有认知**: KB 已覆盖 CFM 框架原理、diffusion TTS 演进(Grad-TTS → FM → DiT)、CFG 机制及变体(离散空间 CFG、多条件 CFG)、VAE 在 TTS 中的角色(sigma-VAE、Semantic-VAE)。本文的 Wav-VAE 继承了这些概念但提出新的 waveform-direct 编码方式。

**创新判断**: 相对于 KB 已有知识,本文的核心新贡献是 (1) waveform latent space 直接建模(消除 mel 中间表示)、(2) 训练-推理不匹配的诊断与修复(prompt noisy latent drift)、(3) APG 替代 CFG。其中 (1) 是架构层面创新,(2)(3) 是推理层面的工程/理论改进。

## 速查

> [!summary] 速查
> - **一句话**: 纯扩散 NAR TTS,通过在 waveform latent 空间直接建模消除 mel 中间表示引入的 compounding error,在 Seed 基准上 SOTA
> - **路线**: 文本 → UMT5 (dual embedding: last hidden + raw word emb) → ConvNeXt V2 refinement → DiT (CFM backbone, cross-attention alignment) → Wav-VAE decoder → waveform
> - **指标**: SIM 0.818 (Seed-ZH), 0.797 (Seed-Hard), CER 1.09% (Seed-ZH), WER 1.50% (Seed-EN); 3.5B 参数, 1M 小时数据 [Table 1]
> - **可借鉴**: (1) prompt noisy latent 在推理时需要用 GT 值覆盖,否则 drift 导致质量下降; (2) UMT5 的 raw word embedding + last hidden state 双通道文本表示改善可懂度; (3) APG 消除 CFG 高 scale 下的 oversaturation; (4) VAE 维度越高重建越好但 TTS 生成反而变差 — 维度选择需联合优化
> - **局限**: 可懂度 (CER/WER) 仍不及 Qwen3-TTS、CosyVoice3.5 等重工程系统; 仅支持中英双语; 推理速度未报告 RTF; 无情感/韵律控制能力的讨论

## 核心问题

**解决什么问题**: 现有 diffusion-based TTS 系统依赖 mel spectrogram 作为中间表示,需要额外 vocoder 将 mel 转为波形。这个两阶段过程引入 **compounding error** — 声学模型预测的 mel 误差被 vocoder 放大,尤其影响 speaker similarity(高频声学细节在 mel 转换中丢失)[§1]。而唯一声称超越 hybrid 方案的 Seed-DiT 架构未公开,纯扩散 TTS 路线缺乏可复现的 SOTA 方案。

**核心假设**: 在 waveform latent 空间直接建模优于 mel spectrogram latent 空间,因为 waveform VAE 保留了 mel 丢弃的相位和高频信息,且统一了声学建模与波形生成为单一连续空间 [§1, §2.2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由两个组件构成 [§1, Fig 1]:

1. **Wav-VAE** (157M 参数): 将原始波形 x ∈ R^(1×T) 编码为连续 latent z ∈ R^(D×T/R),默认 D=64, R 对应 11.72 Hz 帧率。解码器直接从 latent 重建波形,不经过 mel spectrogram [§3]。
2. **Diffusion Transformer (DiT)**: CFM backbone,1B 或 3.5B 参数。在 Wav-VAE 的 latent 空间中用 flow matching 学习文本→语音的生成 [§4]。

[论文原文] 这种设计"elegantly bypasses intermediate transformations and mitigates the compounding error problem" [§2.2]。

### 关键设计选择

**1. Wav-VAE 架构** [§3.1]

Encoder 采用 hierarchical downsampling: 原始波形经 weight-normalized 1D conv 投射到高维特征空间,然后通过 N 个级联 Oobleck block(每个 block 以 stride s_i 降采样),每个 block 包含 dilated residual units(使用 Snake activation)和 **non-parametric shortcut path**(space-to-channel reshape + channel averaging,无参数)。

[论文原文] shortcut path 的设计动机: "to stabilize the training process under aggressive downsampling" [§3.1],来自 CLEAR (Wu et al., 2025)。[agent 解读] 这类似 ResNet 的残差思想,但因维度变化使用了 reshape 而非简单 identity,本质是为极端降采样率(从 24kHz 到 ~12Hz,约 2000x)提供绕过非线性变换的线性通路。

Decoder 对称设计,用 channel-to-space rearrangement 上采样。

**2. 为什么选 waveform latent 而非 mel latent** [§2.2, §5.3.1]

[论文原文] Mel spectrogram "inherently discards phase information and fine-grained high-frequency details" [§3],导致:
- latent → mel → waveform 的两次转换引入 compounding error
- 高频声学细节(对 speaker similarity 至关重要)在 mel 转换中丢失

实验验证 (Table 3): Wav-VAE vs Mel-VAE 在相同 1B DiT 下,SIM 从 0.706→0.812 (ZH), 0.714→0.762 (EN), 0.696→0.787 (ZH-Hard)。CER/WER 也改善(1.29→1.18, 2.20→1.78)[§5.3.1]。

[论文原文] SIM 的巨大提升"elegantly corroborates our hypothesis: fine-grained, high-frequency acoustic details—which are essential for zero-shot voice cloning—are intrinsically fragile and easily lost during the cascading conversions" [§5.3.1]。

**3. UMT5 双通道文本表示** [§4.2]

[论文原文] 仅用 UMT5 最后一层 hidden state 作为文本表示导致可懂度差。作者假设"the last hidden state is rich in high-level semantic information, it abstracts away the low-level lexical and phonetic cues that are crucial for precise acoustic mapping" [§4.2]。

解决方案: q = LayerNorm(last_hidden_state) + LayerNorm(raw_word_embedding) [Eq. 5]。两个 LayerNorm 均为 non-parametric,用于平衡两个表示空间的 scale 差异。

[agent 解读] 这本质上是 skip connection 的思想 — 将 Transformer 底层的表面词汇信息(发音线索)直接传递给 TTS 模型,绕过深层语义抽象对音素信息的稀释。后续经 ConvNeXt V2 做 local refinement 加速对齐收敛。

**4. 修复训练-推理不匹配** [§4.3]

[论文原文] 在推理时使用 Euler method 迭代 ODE,noisy latent z_t 由 prompt 部分 z_t^ctx 和生成部分 z_t^gen 组成。训练时 z_t 由线性插值 (Eq. 3) 构造。但推理时:
- z_t^gen: 由 velocity prediction 迭代更新,与 GT 轨迹一致
- z_t^ctx: flow matching loss (Eq. 4) 仅在 masked region 计算,prompt 部分的 velocity 预测"essentially unconstrained and arbitrary" — 迭代累积导致 z_t^ctx 偏离 GT 轨迹 [§4.3]

修复: 每一推理步强制覆盖 z_t^ctx = tz^ctx + (1-t)z_0^ctx [Eq. 7]。

推论: 计算 unconditional velocity 时,不仅要 drop z^ctx 条件,还必须 drop 显式构造的 noisy prompt latent z_t^ctx,因为它"inherently leaks acoustic information about the prompt" [§4.3]。

[agent 解读] 这是一个重要但容易被忽略的工程细节。VoiceBox/F5-TTS 等使用同样的 masked conditioning 方式但似乎未报告此问题。原因可能是: (1) mel-level 的 drift 影响较小(mel 是 framewise independent 的),而 waveform latent 的时域相关性使 drift 更显著; (2) 较少的推理步数(如 NFE=16)放大了每步 drift 的影响。

**5. Adaptive Projection Guidance (APG) 替代 CFG** [§4.4]

[论文原文] 标准 CFG (Eq. 8) 在高 guidance scale 下引入"oversaturation phenomenon" — 产生 audible artifacts [§4.4]。

APG 的核心思想(来自 Sadat et al., 2024): 将 guidance residual 分解为平行于 conditional prediction 的分量和正交分量。oversaturation 主要由平行分量导致,因此选择性衰减它 [§4.4]:

1. 从 velocity 域转到 sample 域: μ_t = z_t + (1-t)v_t
2. 计算 guidance term ∆μ_t 的平行/正交分解
3. 衰减平行分量: μ_t^APG = μ_t + α∆μ_t^⊥ + η∆μ_t^∥ (η=0.5, α=4.0) [Eq. 9]
4. 映射回 velocity 域 [Eq. 10]
5. 加入负动量 reverse momentum trick (β=-0.3) [§4.4]

**6. 其他技术选择**

- **REPA** (Representation Alignment): 用预训练 mHuBERT 第 8 层特征对齐 DiT 第 8 层输出,L1 距离。[论文原文] "does not enhance the generation quality" 但"substantially accelerates the convergence during training" [§4.1]
- **Long-skip connections**: 将网络输入直接加到最终层 hidden state (来自 DiTTo-TTS)
- **Global AdaLN**: 所有 DiT 层共享一个 AdaLN projection (来自 Gentron),减少参数不降性能 [§4.1]
- **Cross-attention alignment**: 不使用 duration predictor,文本-语音对齐由 cross-attention 隐式学习 (来自 DiTTo-TTS) [§4.1]
- **QK-Norm**: 在 attention 中使用 RMSNorm 做 query-key normalization 稳定训练 [§4.1]

### 训练策略

**Wav-VAE 训练** [§3.2]:
- 两阶段对抗训练: warmup phase 仅用重建 loss (L_spec + L_mel + L_time + L_KL),之后加入 adversarial loss (L_adv + L_fm) 和 multi-scale STFT discriminator
- 200K 小时中英语语音, 32 H800 GPU, batch=384

**DiT 训练** [§5.1]:
- CFM 目标 (Eq. 4),VoiceBox-style random span masking 构造 prompt
- 训练时 10% 概率 joint drop audio context + text (为 CFG)
- 1B: 100K hours, 16 GPU, batch=256; 3.5B: 1M hours, 64 GPU, batch=1024
- AdamW, lr 1e-4→1e-5 linear decay, 1K warmup steps
- 最大音频时长 60s, 采样率 24kHz

## 实验

| 指标 | 本文 (3.5B) | 本文 (1B) | Seed-DiT | F5-TTS | MaskGCT | CosyVoice3.5 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CER (ZH) ↓ | 1.09% | 1.18% | 1.18% | 1.56% | 2.27% | 0.87% | Seed-ZH | [Table 1] |
| SIM (ZH) ↑ | **0.818** | 0.812 | 0.809 | 0.741 | 0.774 | 0.797 | Seed-ZH | [Table 1] |
| WER (EN) ↓ | 1.50% | 1.78% | 1.73% | 1.83% | 2.62% | 1.57% | Seed-EN | [Table 1] |
| SIM (EN) ↑ | **0.786** | 0.762 | 0.790 | 0.647 | 0.714 | 0.738 | Seed-EN | [Table 1] |
| CER (ZH-Hard) ↓ | 6.04% | 6.33% | 10.27% | 8.79% | 8.67% | 5.71% | Seed-Hard | [Table 1] |
| SIM (ZH-Hard) ↑ | **0.797** | 0.787 | 0.748 | - | 0.713 | 0.786 | Seed-Hard | [Table 1] |

**Wav-VAE 重建** [Table 2]:
- PESQ 3.237, STOI 0.967, UTMOS 4.013 (11.72 Hz, dim=64)
- 优于 VibeVoice (PESQ 3.068, STOI 0.828) 和多数 discrete codecs

**Ablation: Wav-VAE vs Mel-VAE** [Table 3]:
- SIM 提升巨大: ZH 0.706→0.812 (+0.106), EN 0.714→0.762 (+0.048), ZH-Hard 0.696→0.787 (+0.091)
- CER/WER 也改善: ZH 1.29→1.18, EN 2.20→1.78, ZH-Hard 7.70→6.33

**Ablation: Latent dimension** [Fig 3]:
- VAE 重建: dim 越高越好 (PESQ/STOI/SIM/UTMOS 单调提升)
- TTS 生成: dim 越高越差 (WER/SIM/UTMOS/DNSMOS 单调下降)
- 即使 3.5B 参数 + dim=128 的 VAE,仍不如 3.5B + dim=64 [§5.3.2]

**Ablation: Frame rate (FPS)** [Fig 4]:
- 低 FPS (7.81 Hz): VAE 可懂度/自然度好但 SIM/PESQ 差; TTS 质量全面提升
- 高 FPS (23.44 Hz): VAE 细节保留好但 TTS 不稳定
- 最优: 11.72 Hz (兼顾 VAE 重建和 TTS 生成) [§5.3.2]

**Ablation: 推理技术** [Table 4, Seed-ZH]:
- 不修复 training-inference mismatch: SIM 从 0.812→0.769, UTMOS 3.16→2.83, CER 基本不变 (1.18→1.21)
- 不用 APG (标准 CFG): SIM 不变 (0.812), CER 不变 (1.18), UTMOS 3.16→3.06, DNSMOS 3.40→3.38

## 局限性

1. **可懂度不及顶尖**: CER/WER 仍落后于 Qwen3-TTS (WER 1.23%)、CosyVoice3.5 (CER 0.87%)、VoxCPM (CER 0.93%) 等系统 [Table 1]。[论文原文] 这些系统依赖"complex multi-stage training pipelines and massive amounts of high-quality, human-annotated data" [§5.2]
2. **语言覆盖有限**: 仅支持中文和英文,虽然使用了 UMT5 (107 语言),但训练数据仅含中英
3. **推理速度缺失**: 论文未报告 RTF 或实时率,3.5B 参数的 DiT 推理开销可能较大。NFE=16 步 [§4.3]
4. **VAE 维度-生成质量的反直觉关系**: dim=64 虽然是最优,但其 VAE 重建 PESQ 仅 3.237(远低于 dim=256 的性能),意味着系统受限于 VAE 表达力的瓶颈。[论文原文] 3.5B 参数也无法弥补 dim=128 的 modeling burden [§5.3.2]
5. **无可控性讨论**: 未涉及情感、韵律、语速等可控 TTS 能力
6. **Wav-VAE 训练数据规模**: 200K 小时,仅在 3s 片段上训练,长音频重建效果未验证

## 点评

**优势**:
- **简洁的系统设计**: 仅 Wav-VAE + DiT 两个组件,无需 duration predictor、语言 token、discrete tokenizer、额外 vocoder,是目前 SOTA 级 TTS 中最简洁的 pipeline 之一
- **深入的实证分析**: VAE dimension 和 frame rate 的系统性消融揭示了 "更好的 VAE ≠ 更好的 TTS" 这一 counterintuitive finding,对领域理解有实质贡献
- **training-inference mismatch 的诊断**: 首次明确指出 VoiceBox-style masked conditioning 中 prompt noisy latent 的 drift 问题,可能对所有使用类似方案的系统都有参考价值
- **完全开源**: 代码 + 模型权重(1B/3.5B)均已发布

**不足**:
- SIM SOTA 但 CER/WER 仍有差距,说明 waveform latent 在可懂度方面不一定优于 mel + 精细工程
- APG 效果有限(仅提升 UTMOS/DNSMOS),不如 training-inference mismatch 修复那样 dramatic
- 与 hybrid 系统 (CosyVoice3.5, MiniMax-Speech) 的对比不够公平 — 后者有 LLM 组件提供语义理解

**定位**: LongCat-AudioDiT 证明了纯 NAR diffusion 路线可以在 speaker similarity 上超越所有 hybrid 系统(包括此前未公开的 Seed-DiT),同时保持竞争力的可懂度。其核心贡献不在于任何单一技术(CFM/DiT/APG 都是已有方法),而在于 **waveform latent space 这一建模选择** 及其配套的系统工程。

## 可复用的 idea

1. **Prompt noisy latent 覆盖**: 在任何使用 VoiceBox-style masked conditioning 的 CFM 系统中,推理时强制用 GT 值覆盖 prompt 部分的 noisy latent,防止 drift。同时在 CFG 的 unconditional branch 也要 drop prompt noisy latent 以避免信息泄漏
2. **UMT5 dual embedding**: 当使用预训练 LM 作为文本 encoder 时,将 raw word embedding 与 last hidden state 相加,可改善 TTS 可懂度。适用于其他使用 pretrained LM 的 TTS 系统
3. **APG 替代 CFG**: 当 CFG 在高 scale 下产生 artifacts 时,APG 通过几何正交分解选择性衰减平行分量,在 TTS 和音频生成中值得尝试
4. **VAE 维度需联合优化**: 选择 VAE latent dimension 时不能单独优化重建质量,必须与下游生成模型联合评估。dim=64 在 11.72 Hz 下对 TTS 最优
5. **Non-parametric shortcut for aggressive downsampling**: 在高压缩率 VAE 中使用 space-to-channel reshape 的参数无关 shortcut path 稳定训练

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节 WHY 解释充分,速查卡片可借鉴字段具体 |
> | 可信赖 | pass | 数字标注覆盖率 >90%; 发现 1 处 SIM 值串行 (已修正) |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注清晰 |
> | 可定位 | pass | KB 背景谱系定位具体,含对比模型名 |
> | 不污染 | pass | 反向更新均为 append,无 overclaim |
> 
> Issues: 4 (high: 0, medium: 1, low: 3)
> 详见 `_review/LongCat-AudioDiT-review.yml`
