---
type: paper
tier: deep
title: "WavTTS: Towards High-Quality Zero-Shot TTS via Direct Raw Waveform Modeling"
arxiv_id: "2606.03455"
source: "Sources/WavTTS.pdf"
authors: [Wenxi Chen, Dongya Jia, Yushen Chen, Zhikang Niu, Yuzhe Liang, Xiquan Li, Ruiqi Yan, Ziyang Ma, Guanrou Yang, Sanyuan Chen, Yue Wang, Zhuo Chen, Kai Yu, Xie Chen]
year: 2026
venue: "arXiv"
tags: [TTS, flow-matching, waveform-generation, end-to-end, zero-shot, diffusion-transformer, NAR]
concepts: ["[[ConditionalFlowMatching]]", "[[Diffusion-basedTTS]]", "[[Non-autoregressiveTTS]]", "[[Classifier-FreeGuidance]]", "[[NeuralVocoder]]", "[[MelSpectrogram]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: WavTTS 属于 [[Diffusion-basedTTS]] 家族中的端到端分支。与主流路线不同的是,当前 diffusion/flow matching TTS 系统均在 mel spectrogram 空间 (F5-TTS, E2 TTS) 或 VAE latent 空间 (MaskGCT, LongCat-AudioDiT) 上建模,再依赖 [[NeuralVocoder]] 或 decoder 还原波形。WavTTS 跳过了这一中间表示,直接在原始波形空间上做 [[ConditionalFlowMatching]]。这与 LongCat-AudioDiT 的 waveform latent 路线 (先用 Wav-VAE 编码再做 diffusion) 形成对比: WavTTS 完全不需要任何预训练 codec/VAE/vocoder。
>
> **已有认知**: KB 中 [[ConditionalFlowMatching]] 页记录了 FM 在 TTS 中已从 vocoder 级 (PeriodWave) 扩展到 mel 级 (F5-TTS) 再到 latent 级 (LongCat-AudioDiT),但直接在原始波形空间上的 FM-TTS 尚无成功先例。[[NeuralVocoder]] 页的演进线显示 vocoder 从 WaveNet 到 HiFi-GAN 到 BigVGAN 的路径,WavTTS 的意义在于尝试彻底消除这一环节。[[MelSpectrogram]] 页 [待确认] 记录了 LongCat-AudioDiT 量化的 mel compounding error (SIM 0.706→0.812 换用 Wav-VAE),与 WavTTS 的动机一致。
>
> **创新判断**: 在 KB 已有知识图谱中,原始波形空间的 flow matching TTS 此前为空白。E3-TTS (2023) 曾探索但未达到 SOTA 水准。WavTTS 的核心贡献在于通过一组针对性设计 (signal-noise variance alignment + noise-shifted scheduling + x-prediction + multi-scale mel loss) 使得原始波形空间建模首次接近 latent/mel 空间模型的性能。
>
> 检索命中: [[ConditionalFlowMatching]]✓, [[NeuralVocoder]]✓ | 过滤: [[Diffusion-basedTTS]](pending-review), [[Non-autoregressiveTTS]](pending-review), [[Classifier-FreeGuidance]](pending-review), [[MelSpectrogram]](pending-review), [[Emilia]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个在原始波形空间上用 flow matching + DiT 实现接近 SOTA 水准的零样本 TTS 系统,消除了对 vocoder/VAE/codec 的依赖
> - **路线**: Text + Audio Prompt → ConvNeXt V2 encoder + Patchified waveform (F=160) → DiT (flow matching, x-prediction) → Unpatchify → Raw waveform (16kHz)
> - **指标**: Seed-TTS test-en WER 1.50% (best) / SIM-o 0.65 / UTMOS 3.92 (best); test-zh CER 1.59% / SIM-o 0.73 [Table 1]
> - **可借鉴**: Signal-noise variance alignment (k=9) 解决波形低方差与 Gaussian prior 的 20dB SNR 失配; PolyShift 推理时间表; x-prediction + multi-scale mel loss 组合
> - **局限**: Speaker similarity 仍落后于 latent 模型 (0.65 vs ZipVoice 0.70); 仅 16kHz; 50 NFE 推理步; 需 100K 小时数据 + 673M 参数才有竞争力

## 核心问题

当前 zero-shot TTS 的主流范式依赖压缩的中间表示:
1. **Mel spectrogram 路线** (F5-TTS, E2 TTS, ZipVoice): 丢弃相位和高频信息,需额外 vocoder 还原波形
2. **VAE latent 路线** (MaskGCT, LongCat-AudioDiT): 高度压缩导致信息丢失,多阶段训练引入累积误差

WavTTS 提出的核心问题: **能否通过直接建模原始波形来实现高质量零样本 TTS,从而绕过所有有损中间表示?** [§1]

这个方向此前被认为不可行,因为原始音频的极长序列长度 (16kHz = 每秒 16000 个采样点) 给模型的长程依赖建模和计算效率带来巨大挑战 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

WavTTS 基于 flow matching + DiT 框架,采用 speech infilling 任务实现零样本语音克隆 (与 Voicebox/F5-TTS 相同范式) [§3.1]:

1. **输入处理**: 给定文本 y 和参考音频 x_ctx (通过 span mask 截取),将原始波形通过 non-overlapping patchification (patch size F=160) 转换为 N x F 的 patch 序列,序列速率 100Hz [§3.1]
2. **条件编码**: 文本 (双语拼音+字母 token) 通过 4 层 ConvNeXt V2 blocks 编码,填充 filler tokens 对齐到音频 patch 长度,实现隐式 text-audio 对齐 (无需 duration predictor) [§3.1]
3. **Flow matching**: DiT backbone (28 层, hidden dim 1152, 16 heads, 673M) 学习从 Gaussian noise 到目标波形的 flow,使用 adaLN-Zero 注入时间步 t [§3.1, Appendix A]
4. **输出**: 线性投影 + unpatchify 恢复预测波形 x_θ [§3.1]

### 关键设计选择

#### 1. x-prediction 替代 v-prediction [§3.1]

[论文原文] 标准 flow matching 训练网络预测速度场 v_t = x_1 - x_0,但在高维波形空间中,预测目标包含随机噪声分量 x_0,在静音段和低能量区域导致不稳定优化。受 JiT (Li & He, 2025) 启发,WavTTS 改为直接预测 clean waveform x_θ = net_θ(x_t, t),等价地将 FM loss 变为 ||x_θ - x_1||^2 / (1-t)^2 [Eq. 2-3]。

[agent 解读] x-prediction 的另一个关键好处是可以直接对预测波形计算 mel loss,而 v-prediction 则需要先从速度场反推波形再计算,增加了实现复杂度。消融实验也验证 x-prediction 在 SIM-o 上明显优于 v-prediction (0.65 vs 0.61) [Table 3]。

#### 2. Multi-scale Mel-Spectrogram Loss [§3.2]

[论文原文] 仅靠时域 FM loss 会迫使模型拟合感知上不重要的样本级变化,阻碍高效优化。因此引入 7 个尺度 (window sizes [32, 64, 128, 256, 512, 1024, 2048]) 的 multi-scale mel loss 作为辅助监督 [Eq. 6-7, Appendix A]:

L = L_FM + λ_mel * L_mel, λ_mel = 0.05

[论文原文] 这个 loss 源自 DAC (Defossez et al., 2022) 的设计 [Appendix A],只在 masked target region 上计算。

[agent 解读] 关键的是,这个 mel loss 在**原始尺度**上计算 (xθ/k vs x1),不受 variance alignment scaling 影响 [§3.3.1]。这是一个精巧的设计: FM 优化在 scaled 空间进行,而感知质量监督在原始空间进行。消融显示去掉 mel loss (λ=0) 导致全面退化,且在 200K 步时模型仍无法生成可理解语音 [§5.2.1]。

#### 3. Signal-Noise Variance Alignment [§3.3.1]

[论文原文] 这是 WavTTS 最关键的设计。原始波形的标准差很小 (Emilia 上 σ ≈ 0.12),而 Gaussian noise σ = 1,导致 rectified flow 的线性插值路径 x_t = (1-t)x_0 + tx_1 中信号被噪声淹没。从 Log-SNR 分析 [Eq. 8]:

Log-SNR(t) = 20*log10(t/(1-t)) + 20*log10(σ_x1/σ_x0)

当 σ_x1 = 0.12, σ_x0 = 1 时,后一项为 -18.4 dB,意味着整个 SNR 曲线下移约 20dB [Fig 3]。

**解决方案**: 在训练前将波形乘以 k = 9 (使 σ ≈ 1),推理后除以 k 恢复原始幅度。

[agent 解读] 这个设计虽然简单 (一个标量缩放),但消融结果极为显著: k=1 时 SIM-o 仅 0.32, UTMOS 2.40; k=9 时 SIM-o 0.65, UTMOS 3.93 [Table 4]。这揭示了一个此前被忽视的问题: rectified flow 的隐含假设 (信号和噪声方差匹配) 在波形空间严重违反。

#### 4. Noise-Shifted Temporal Scheduling [§3.3.2]

**训练时间步采样**: 采用 logit-normal 分布 (μ=-0.8, σ=0.8) 替代均匀采样,将更多训练权重分配给高噪声 (低 t) 区域 [Fig 4]。从 loss 重新加权的角度看,这等价于在均匀采样下对低 t 区域施加更大的隐式权重 π(t)/(1-t)^2 [Eq. 9] [§3.3.2]。

**推理时间步分配 — PolyShift** [§3.3.2, Eq. 10]:
t = τ^p / (τ^p + s(1-τ^p)), 其中 p=2, s=3

[论文原文] Sway Sampling (F5-TTS) 对波形空间的偏移力度不足,PolyShift 通过 polynomial + time-shift 组合实现更灵活的高噪声区域密集采样 [Fig 7]。

[agent 解读] 训练和推理都向高噪声区域偏移的一致性很重要 — 训练时在困难的高噪声区域投入更多样本学习粗粒度结构,推理时在相同区域投入更多 ODE 步减少截断误差。但过度偏移会牺牲细节 (aggressive μ=-1.2 导致 SIM-o 和 UTMOS 退化) [Fig 5]。

### 训练策略

- **数据**: Emilia 95K 小时英中双语语音, 16kHz [§4]
- **训练**: 1.2M 步, 8x A100 80GB, batch size 153,600 audio patch frames (≈ 0.43h audio/batch) [§4]
- **优化器**: AdamW, 峰值 lr 7.5e-5, 20K 步 warmup 后恒定 [§4]
- **CFG**: 训练时以 0.1 概率同时丢弃文本和音频提示 [§3.1]
- **推理**: 50 NFE, CFG scale α=3, PolyShift (p=2, s=3) [§4]
- **Infilling**: 随机 mask 70%-100% 的连续音频段 [Appendix A]

## 实验

### 主实验: Zero-Shot TTS (Seed-TTS benchmark)

| 指标 | WavTTS | F5-TTS | ZipVoice | MaskGCT | LongCat-AudioDiT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER (%) ↓ | **1.50** | 1.65 | 1.60 | 2.36 | 1.94 | Seed-TTS test-en | [Table 1] |
| SIM-o ↑ | 0.65 | 0.66 | **0.70** | 0.71 | 0.76 | Seed-TTS test-en | [Table 1] |
| UTMOS ↑ | **3.92** | 3.73 | 3.83 | 3.57 | 3.80 | Seed-TTS test-en | [Table 1] |
| CER (%) ↓ | 1.59 | 1.55 | 1.40 | 2.48 | **1.10** | Seed-TTS test-zh | [Table 1] |
| SIM-o ↑ | 0.73 | 0.75 | 0.75 | 0.77 | **0.81** | Seed-TTS test-zh | [Table 1] |

### 与端到端模型对比

| 指标 | WavTTS | VITS_LJ | JETS | WaveGrad 2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER (%) ↓ | **3.43** | 3.72 | 3.73 | 25.19 | LJSpeech | [Table 2] |
| UTMOS ↑ | **4.39** | 4.37 | 4.36 | 3.24 | LJSpeech | [Table 2] |
| WER (%) ↓ | **2.02** | 2.23 | 3.00 | 33.77 | LibriSpeech-PC | [Table 2] |

### 关键消融

**训练目标** [Table 3]:
- v-prediction + mel loss: WER 1.67, SIM-o 0.61, UTMOS 3.94
- x-prediction + mel loss (default): WER 1.65, SIM-o **0.65**, UTMOS 3.93
- x-prediction 无 mel loss: WER 1.92, SIM-o 0.56, UTMOS 3.77

**Variance alignment (k)** [Table 4]:
- k=1 (无缩放): WER 4.18, SIM-o 0.32, UTMOS 2.40 — 严重退化
- k=9 (default): WER 1.65, SIM-o **0.65**, UTMOS **3.93**

**表示空间对比** [Fig 6]:
- 波形: 收敛最快, WER 4.10% @ 200K, UTMOS 3.93 @ 1M
- Mel spectrogram: WER 9.76% @ 200K, UTMOS 3.68 @ 1M
- STFT: ~600K 步才开始生成可识别语音
- MDCT: ~400K 步开始生成可识别语音

**数据和模型 scaling** [Table 6]:
- LibriTTS 585h + 673M: SIM-o 0.31 (零样本失败)
- Emilia 100K + 340M: SIM-o 0.56
- Emilia 100K + 673M: SIM-o **0.65** (数据+模型都需要)

## 局限性

1. **Speaker similarity 差距**: SIM-o 0.65 vs 最佳 latent 模型 0.76 (LongCat-AudioDiT), 差距约 0.11。论文假设波形空间的高维性使有限容量的模型难以优先学习说话人音色 [§5.1.1], [agent 解读] 但也可能是波形 patchification (F=160) 的压缩粒度不够 — 相比 VAE latent 的高度抽象表示,160 点的 patch 仍包含大量冗余的低级信号变化。
2. **仅 16kHz**: 未探索更高采样率 (如 24kHz/44.1kHz),这对实际部署是明显限制。
3. **推理效率**: 50 NFE 在波形空间意味着 50 次完整 DiT forward pass,实际推理速度未报告。[agent 解读] 考虑到 673M 模型 + 波形级序列长度,推理成本可能显著高于 latent 模型。
4. **数据依赖**: 需要 100K 小时训练才有竞争力; 585 小时 LibriTTS 上完全无法做零样本 (SIM-o 0.31) [Table 6]。

## 点评

**正面**:
- 令人信服的概念验证 (proof-of-concept): 通过 4 个精心设计的组件 (x-prediction, mel aux loss, variance alignment, noise-shifted scheduling) 使得此前被认为不可行的波形空间 TTS 首次接近 SOTA。每个组件都有清晰的消融支持。
- Variance alignment 是一个优雅且通用的 insight — 任何需要在低方差信号上做 flow matching 的任务 (如环境声音、音乐) 都可能受益。
- 表示空间对比实验 (Fig 6) 特别有价值: 波形直接建模竟然优于 mel spectrogram 且收敛更快,这颠覆了"mel 更容易学"的普遍假设。

**保留意见**:
- Speaker similarity 的差距 (0.65 vs 0.76) 不是小问题 — 在实际 voice cloning 应用中,这个差距可能是 deal-breaker。论文归因于"高维性",但缺乏更深入的分析。
- 与 LongCat-AudioDiT (同样消除 mel 中间表示,使用 Wav-VAE latent) 相比,WavTTS 在 SIM-o 上大幅落后 (0.65 vs 0.76 en, 0.73 vs 0.81 zh),表明 latent 压缩对说话人信息的保留反而优于直接波形建模。这挑战了"无损 = 更好"的直觉。
- 缺少推理速度对比 (RTF),无法评估端到端范式是否真正带来实际效率提升。

## 可复用的 idea

1. **Signal-Noise Variance Alignment**: 对任何方差不匹配的 flow matching 任务通用。可应用于环境声/音乐生成 (低方差), 也可能用于某些图像任务 (HDR 图像等)。实现极简: 训练前乘 k,推理后除 k。
2. **PolyShift 推理时间表**: t = τ^p / (τ^p + s(1-τ^p)) 比 Sway Sampling 更灵活,且可通过 (p, s) 二维调参适配不同任务。可直接插入任何 flow matching 系统。
3. **x-prediction + frequency-domain auxiliary loss 组合**: x-prediction 使 clean 预测可用于任意域的辅助 loss (mel, STFT, 感知 loss 等),而 v-prediction 无此便利。这个组合范式可推广到其他 raw signal generation 任务。
4. **波形 patchification 策略**: 简单的 non-overlapping 1D patchification (F=160 @ 16kHz = 10ms) 将序列压缩 160 倍,使 DiT 可以处理原始波形。可作为所有波形级 Transformer 模型的 baseline 方案。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 4 个设计选择均有因果解释 + 消融支持,速查可借鉴字段具体可操作 |
> | 可信赖 | pass | 数字标注覆盖率 >90%, Table 1/3/4/6 交叉验证全部正确 |
> | 可区分 | pass | 方法节 [论文原文]/[agent 解读] 标注覆盖充分,无未标注推断 |
> | 可定位 | pass | KB 背景谱系定位具体 (mel/latent/waveform 三路线对比),创新判断有基准 |
> | 不污染 | pass | 仅追加 key_papers,无实质修改,污染风险低 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/WavTTS-review.yml`
