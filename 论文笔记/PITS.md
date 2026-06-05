---
type: paper
tier: deep
title: "PITS: Variational Pitch Inference without Fundamental Frequency for End-to-End Pitch-controllable TTS"
arxiv_id: "2302.12391"
source: "Sources/PITS.pdf"
authors: [Junhyeok Lee, Wonbin Jung, Hyunjae Cho, Jaeyeon Kim, Jaehwan Kim]
year: 2023
venue: "Preprint (maum.ai / KAIST / SNU)"
tags: [TTS, pitch-control, end-to-end, VITS, VAE, adversarial-training, Yingram, prosody, disentanglement, voice-conversion]
concepts: ["[[F0Modeling]]", "[[VariationalAutoencoderforTTS]]", "[[ProsodyModeling]]", "[[SpeechFactorization]]", "[[DurationPredictor]]"]
models: ["[[VITS]]"]
tasks: []
datasets: ["VCTK"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: PITS 直接构建在 [[VITS]] 之上,继承 VITS 的 conditional VAE + normalizing flow + HiFi-GAN decoder 端到端架构 [待确认]。核心创新方向是**pitch 可控性** — 这在 [[ProsodyModeling]] 中属于 prosody 的 pitch/F0 维度,但 PITS 选择了一条非主流路径:不用 [[F0Modeling]] 中的显式 F0 预测(FastSpeech 2 variance adaptor 路线),也不隐式依赖 VITS 的单一 posterior encoder 来编码 pitch,而是引入 Yingram 作为 F0 的替代表示,用双 posterior encoder 架构实现 pitch-linguistic 解耦。

**已有认知对比**: 
- [[SpeechFactorization]] 记录了多种 content-pitch 解耦方法(对抗训练、信息瓶颈、self-distillation 等)。PITS 的解耦思路源自 voice conversion 领域(NANSY, Choi et al., 2021),用 Yingram encoder 编码 pitch + harmonics,STFT encoder 编码 linguistic,加可选 Q-VAE 限制 STFT encoder 的 pitch 泄露 — 属于信息瓶颈 + 量化混合路线。
- [[VariationalAutoencoderforTTS]] 记录了 VAE 在 TTS 中的 one-to-many mapping 建模。PITS 的变分 pitch 推断本质是对 VITS VAE 框架的扩展:将单一 posterior encoder 拆为两个(STFT + Yingram),分别编码 linguistic 和 pitch 的 latent variables [待确认]。
- [[DurationPredictor]] 记录了 VITS 的 stochastic duration predictor。PITS 沿用此设计,duration 建模方面无创新 [待确认]。

**创新判断**: 核心创新在于用 Yingram(自相关 pitch 表示)替代 F0,避免 F0 在 unvoiced segments 未定义的问题,并通过 scope-shift 机制实现推理时的连续 pitch 控制。这是一种比 F0 predictor 更底层的 pitch 建模思路,从表示层面解决 F0 的固有局限。

> 检索命中: [[ProsodyModeling]]✓, [[SpeechFactorization]]✓ | 过滤: [[VITS]](pending-review), [[F0Modeling]](pending-review), [[VariationalAutoencoderforTTS]](pending-review), [[DurationPredictor]](pending-review) | 未命中但可能相关: [[NeuralVocoder]]

## 速查

> [!summary] 速查
> - **一句话**: 在 VITS 上添加 Yingram encoder + adversarial pitch-shifted training,实现无需 F0 提取器的端到端 pitch 可控 TTS
> - **路线**: Phoneme → Text Encoder + MAS → Prior; Linear Spec → STFT Encoder (zspec) + Yingram → Yingram Encoder (zyin) → scope crop → concat → HiFi-GAN Decoder → Waveform; 推理时 scope-shift 控制 pitch
> - **指标**: MOS 4.01+-0.04 (vs VITS 3.89, GT 4.04), CER 3.27% (vs VITS 4.35%), EER 0.926% (vs VITS 1.08%), VCTK [Table 1]
> - **可借鉴**: Yingram 作为 F0 替代的 pitch 表示; scope-shift 机制实现推理时无需重训练的连续 pitch 控制; adversarial pitch-shifted training 确保 shifted speech 质量
> - **局限**: Q-VAE 解耦路线失败(CER 42-50%); 仅验证 VCTK 多说话人场景; pitch-shift 影响说话人身份(EER 随 shift 增大); 无大规模数据/语言泛化验证

## 核心问题

PITS 要解决的核心问题是:**现有 pitch-controllable TTS 依赖显式 F0 建模,导致合成语音的 pitch 多样性低,且 F0 在部分语音段(unvoiced)上没有良好定义** [§1]。

具体而言:
1. FastSpeech 2、FastPitch 等直接回归 ground truth F0,是确定性预测,pitch 方差低 [§1]
2. VarianceFlow 虽用 normalizing flow 建模 F0,仍因直接建模 F0 而方差低 [§1]
3. VISinger、Period VITS 在 VITS 上加 F0 predictor,但需要外部 duration 标注和 pitch contour,降低了合成多样性 [§1]
4. F0 本身在 unvoiced segments 没有良好定义 [§1, citing Ardaillon & Roebel 2020]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

PITS 在 VITS 基础上做了四项结构性修改 [§2]:

1. **Yingram Encoder** — 新增的第二 posterior encoder,专门编码 pitch 信息 [§2.1]
2. **Yingram Decoder** — 辅助解码器,强制 Yingram latent 的 channel 方向平移等变性 [§2.2]
3. **Q-VAE** (可选) — 量化 STFT encoder 输出以增强 pitch-linguistic 解耦 [§2.3]
4. **Pitch-shifted waveform synthesis** — 训练时合成 pitch-shifted 波形并对其做对抗训练 [§2.4]

训练时数据流 [Fig 1a]:
```
Phonemes → Text Encoder → htext
Linear Spec → STFT Encoder → zspec → (optional VQ) → zling
Yingram → Yingram Encoder → zyin → crop → zcrop (default scope)
                                    → shift+crop → zcrop_shift (shifted scope)
concat(zling, zcrop) → zdec → Decoder → normal waveform
concat(zling_sg, zcrop_shift) → zdec_shift → Decoder → pitch-shifted waveform
zcrop_shift → Yingram Decoder → reconstructed shifted Yingram
concat(zspec, zyin) → Flow → MAS → alignment → Duration Predictor
```

推理时 [Fig 1b]:
```
Phonemes → Text Encoder → htext → Duration Predictor → Length Regulator
→ Prior (mu, sigma) → Flow^{-1} → z → split → (zspec, zyin)
→ scope-shift zcrop → concat → Decoder → waveform
```

### 关键设计选择

**为什么用 Yingram 而不是 F0?** [论文原文] F0 在 unvoiced segments 没有良好定义 [§1]。Yingram 基于 YIN 自相关算法,是一种多通道 pitch 表示,包含基频和谐波信息,在所有语音段都有定义 [§1, citing Choi et al., 2021]。此外,NANSY 中基于 Yingram 的模型在 voice conversion 中被偏好优于基于 F0 的模型 [§1]。

**为什么需要两个 posterior encoder?** [论文原文] 类似 voice conversion 中分离 linguistic 和 pitch 信息的做法(NANSY, Polyak et al., 2021),PITS 用 STFT encoder 编码 linguistic 信息,Yingram encoder 编码 pitch 信息 [§2.1]。[agent 解读] 单一 encoder 中 pitch 和 content 纠缠,无法实现推理时独立控制 pitch。

**Scope-shift 如何实现 pitch 控制?** [论文原文] Yingram encoder 输出 80 通道 latent,默认取中间 50 通道(16th-65th)作为 zcrop [§2.1]。推理时,通过改变 crop 的起始位置(scope-shift s),等效于在 Yingram 的频率轴上平移,实现 pitch 的半音级控制 [§2.1]。s 的范围 [-15, 15] 对应约 ±7.5 半音的 pitch shift。[agent 解读] 这利用了 Yingram 在频率域的结构化特性 — 相邻通道对应相邻半音,scope 的平移等价于 pitch 的平移。

**为什么需要 Yingram Decoder?** [论文原文] Scope-shift 机制要求 Yingram encoder 在 channel 方向具有平移等变性(translation equivariance)。但实验发现默认 scope 之外的 latent 区域未被训练,仍为 Gaussian 噪声 [§2.2]。Yingram decoder 通过重建 shifted scope 的 Yingram,强制这些区域也学到有意义的 pitch 表示。

**为什么 Q-VAE 失败了?** [论文原文] 将 STFT encoder 的输出量化(Q-VAE)旨在通过离散化去除连续 pitch 信息,增强解耦 [§2.3]。但实验显示 Q-VAE 导致质量和可懂度严重下降(CER 42-50%) [§4, Table 1]。[论文原文] 作者观察到 codebook loss 和 commit loss 发散,推测是因为 zspec 同时接收来自 flow 和 codebook 的多个梯度 [§2.3]。[agent 解读] Q-VAE 的失败可能反映了 VITS 训练框架中 VQ 和 normalizing flow 的不兼容性 — flow 需要连续可微的 latent,VQ 引入的离散化破坏了梯度传播。

**为什么需要 adversarial pitch-shifted training?** [论文原文] 仅用 Yingram reconstruction loss (Lyin) 训练 pitch-shifted 合成会导致质量退化 [§2.4]。将 pitch-shifted 波形也送入 discriminator 做对抗训练,利用"shifted 和 original 共享相同 time-aligned linguistic features"的特性,提供可靠的配对对抗训练 [§2.4]。对比 (A+D) vs (D+Q) 证实了 adversarial loss 对 pitch-shifted 合成的自然度至关重要 [§4, Table 2]。

### 训练策略

**Loss 函数** [§2.5, Eq. 3]:
```
L_total = L_mel + L_KL + L_dur + L_adv(G) + L_fm(G)           [VITS 原有]
        + L_yin + L_adv_shift(G) + L_fm_shift(G)               [pitch-shift 相关]
        + (L_yd)                                                 [可选: Yingram decoding]
        + (L_vq + L_commit)                                      [可选: Q-VAE]
```

**Yingram reconstruction loss** [§2.4, Eq. 2]: 对 Yingram 取负指数后计算 L1 loss。[论文原文] 负指数变换是因为接近零的值对谐波表示更关键,且 pitch-shifted Yingram 因含 linguistic 信息不能完美平移等变 [§2.4]。

**Discriminator**: 使用 Avocodo 的 CoMBD (Collaborative Multi-Band Discriminator) + SBD (Sub-Band Discriminator) 替代 VITS 原有的 MPD [§2.4]。[论文原文] CoMBD 的协作式多尺度层级结构有助于提升语音质量,且也接收 pitch-shifted 波形的层级输出 [§2.4]。

**PhaseAug**: 训练时应用可微相位增强以防止周期性伪影 [§2.4, citing Lee et al., 2022a]。

**训练细节**: 4 x V100, batch size 48/GPU, 3000 epochs, 14-18 天 [§3.2]。Yingram 80 通道, 24 notes/octave, note 69 = 440 Hz, 频率范围 30.8-508 Hz [§3.3]。

## 实验

| 指标 | PITS (A+D) | VITS | FS2+HiFi-GAN | Ground Truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS | 4.01 ± 0.04 | 3.89 ± 0.04 | 3.84 ± 0.04 | 4.04 ± 0.04 | VCTK | [Table 1] |
| CER (%) | 3.27 | 4.35 | 3.78 | 2.63 | VCTK | [Table 1] |
| EER (%) | 0.926 | 1.08 | 2.62 | 1.54 | VCTK | [Table 1] |

**Architecture search** (pitch-shifted speech, scope-shift=0) [Table 2]:

| 变体 | MOS | CER (%) | EER (%) | 配置 |
| --- | --- | --- | --- | --- |
| A+D+Q | 3.70 ± 0.05 | 42.3 | 2.16 | 全部 |
| **A+D** | **4.01 ± 0.04** | **3.27** | **0.926** | **最优** |
| A+Q | 3.90 ± 0.04 | 18.8 | 1.39 | 无 Yingram decoder |
| D+Q | 3.65 ± 0.05 | 49.6 | 2.16 | 无 adversarial |

**关键发现**:
1. **Q-VAE 有害**: 含 Q-VAE 的模型 CER 飙升至 18.8-49.6%,表明量化严重损害可懂度 [§4, Table 1]
2. **Adversarial loss 关键**: (A+D) vs (D+Q) 对比显示 adversarial loss 显著提升 pitch-shifted 合成的自然度 [§4, Table 2]
3. **Pitch-shift 影响说话人身份**: EER 随 pitch-shift 增大而增加,但 (A+D) 在所有 shift 值下 EER 最小 [§4, Table 2]
4. **PITS (A+D) 与 GT 无统计显著差异**: MOS 4.01 vs 4.04 [§4, Table 1]
5. **Pitch contour 平行**: [Fig 2] 展示不同 scope-shift 下的 F0 contour 在 log scale 上近似平行,验证 pitch 控制的有效性

## 局限性

1. **Q-VAE 解耦失败**: 论文承认 Q-VAE 显著降低质量和可懂度 [§5]。这意味着 PITS 的 pitch-linguistic 解耦依赖于双 encoder 的隐式分离,缺少显式解耦机制保证
2. **仅验证 VCTK**: 44 小时多说话人数据集,未验证单说话人、大规模数据或非英语场景
3. **Pitch-shift 改变说话人身份**: EER 随 shift 增大,说明 pitch 和 speaker identity 仍有耦合 [Table 2]
4. **Yingram 频率范围有限**: 30.8-508 Hz,对高音域说话人或歌声可能不够 [§3.3]
5. **Voice conversion 质量受限**: 附录 A 中 VC 需要迭代合成(3 次迭代)且质量仍不如 TTS 用途 [Appendix A]
6. **训练成本**: 14-18 天 4xV100,Yingram encoder 增加了约 50% 的参数量 [agent 解读]

## 点评

PITS 的核心贡献是提出了一种**从表示层面解决 F0 局限性**的思路:不是改进 F0 predictor,而是用 Yingram 这种更鲁棒的 pitch 表示替代 F0。这在 [[F0Modeling]] 的演进线中开辟了一条独立路径 — 大多数工作在"如何更好地预测/生成 F0"上努力,PITS 问的是"为什么一定要用 F0"。

Scope-shift 机制是一个优雅的设计:利用 Yingram 在频率域的结构化特性,通过简单的通道偏移实现 pitch 控制,无需额外的 pitch predictor 或 conditioning。但这种优雅性建立在 Yingram 的特定频率-通道映射上,泛化到其他 pitch 表示可能需要重新设计。

Q-VAE 的失败是一个有价值的负面结果。它揭示了在 VITS 的 flow-based 训练框架中集成 VQ 的困难 — zspec 同时接收 flow 和 codebook 的梯度导致训练不稳定。这与后来 codec-based TTS 中 VQ 的成功形成对比:codec-based 系统(如 EnCodec)在独立的 encoder-decoder 框架中训练 VQ,不存在 flow 的梯度干扰。

**局限判断**: 论文发表于 2023 年初,正值 TTS 从 VITS-based 向 codec/LLM-based 范式转型的时期。PITS 的 pitch 控制思路(双 encoder 解耦 + scope-shift)绑定了 VITS 的架构假设,在 LLM-based TTS 中不直接适用。但 Yingram 作为 pitch 表示的价值是架构无关的,后续工作可在任何框架中使用。

## 可复用的 idea

1. **Yingram 替代 F0 作为 pitch 表示**: 自相关 pitch 表示在 unvoiced 段有定义且包含谐波信息,可用于任何需要 pitch 建模的场景
2. **Scope-shift pitch 控制**: 利用频率域结构化表示的通道偏移实现推理时 pitch 控制,无需重训练或额外 predictor
3. **Adversarial training of augmented outputs**: 对训练时生成的 pitch-shifted/augmented 样本做对抗训练,确保变换后的输出质量,可迁移到其他受控生成任务
4. **负指数 Yingram loss**: 对 pitch 表示取负指数后计算重建 loss,强调谐波区域的低值,适用于自相关类 pitch 特征的重建
5. **Q-VAE 的负面教训**: 在含 normalizing flow 的 VAE 框架中直接量化 latent 可能破坏训练稳定性,需要更精心的梯度隔离设计
