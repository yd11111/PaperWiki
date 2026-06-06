---
type: paper
tier: deep
title: "CM-TTS: Enhancing Real Time Text-to-Speech Synthesis Efficiency through Weighted Samplers and Consistency Models"
arxiv_id: "2404.00569"
source: "Sources/CM-TTS.pdf"
authors: [Xiang Li, Fan Bu, Ambuj Mehrish, Yingting Li, Jiale Han, Bo Cheng, Soujanya Poria]
year: 2024
venue: "arXiv"
tags: [TTS, diffusion, consistency-model, NAR, single-step-generation, weighted-sampling, mel-spectrogram]
concepts: ["[[Diffusion-basedTTS]]", "[[DiffusionModel]]", "[[ScoreMatching]]", "[[Non-autoregressiveTTS]]", "[[NeuralVocoder]]", "[[MelSpectrogram]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[NeuralVocoder]]✓, [[Diffusion-basedTTS]], [[DiffusionModel]], [[ScoreMatching]], [[Non-autoregressiveTTS]], [[MelSpectrogram]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[NeuralVocoder]]✓, [[Diffusion-basedTTS]][待确认], [[DiffusionModel]][待确认], [[ScoreMatching]][待确认], [[Non-autoregressiveTTS]][待确认], [[MelSpectrogram]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: CM-TTS 位于 diffusion-based TTS 的"高效加速"分支。KB 中记录了该分支的两条既有路线: (1) DiffGAN-TTS 通过 GAN 学习大步长去噪分布实现 1 步生成,但引入对抗训练的收敛困难 [Diffusion-basedTTS]; (2) ProDiff 通过知识蒸馏将 teacher 压缩,但依赖预训练 teacher model [Diffusion-basedTTS]。CM-TTS 提出第三条路线: 基于 consistency model (Song et al., 2023) 的 consistency training,无需对抗训练也无需 teacher 蒸馏。

**已有认知**: Diffusion model 通过 SDE/ODE 框架统一了 DDPM 和 score-based 方法 [DiffusionModel, ScoreMatching]。Consistency model 在此 ODE 轨迹上施加自一致性约束: 轨迹上任意两点经 consistency function 映射后应得到相同的起始点 [§3]。CM-TTS 的 CM-Decoder 采用与 DiffGAN-TTS 相同的非因果 WaveNet 结构 [Diffusion-basedTTS],前端复用 FastSpeech2 的 phoneme encoder + variance adaptor [Non-autoregressiveTTS],vocoder 使用 HiFi-GAN [NeuralVocoder],输出 80-bin mel spectrogram [MelSpectrogram]。

**创新判断**: 相对于 KB 中已记录的 DiffGAN-TTS (GAN 加速) 和 CoMoSpeech (consistency distillation),CM-TTS 的核心差异在于使用 consistency training 而非 consistency distillation,消除了对 teacher model 的依赖。同时提出 weighted sampler 机制(尤其是 importance sampler)改善训练中的采样偏差。

## 速查

> [!summary] 速查
> - **一句话**: 将 consistency training (非 distillation) 应用于 mel-spectrogram 生成,配合 importance sampler 消除对抗训练和 teacher 依赖,实现单步高质量 TTS
> - **路线**: Phoneme → FFT Encoder → Variance Adaptor (pitch/energy/duration) → CM-Decoder (WaveNet, online+target EMA) → Mel → HiFi-GAN → Waveform
> - **指标**: VCTK MOS 3.96 (T=1) vs DiffGAN-TTS 3.45 (T=1) [Table 1]; LJSpeech MOS 3.84 (T=1) vs DiffGAN-TTS 3.71 [Table 8]; 28.6M params
> - **可借鉴**: Importance sampler 根据历史 loss 动态调整 ODE 轨迹采样权重,可迁移到任何基于 ODE 轨迹采样的训练(flow matching 等); padding 纳入 loss 计算提升变长序列建模
> - **局限**: 单说话人零样本场景不如 DiffGAN-TTS [Table 9]; 仅探索 mel 生成未扩展到其他音频任务; 网络结构本身未优化(层数/残差模块); 代码开源但影响力有限

## 核心问题

CM-TTS 要解决的核心问题是: **diffusion-based TTS 在推理时需要多步采样导致速度慢,而现有加速方案要么引入对抗训练不稳定(DiffGAN-TTS),要么依赖预训练 teacher 模型增加复杂度(CoMoSpeech/consistency distillation)**。

具体来说:
1. DMs 的多步迭代采样是实时 TTS 的瓶颈 [§1]
2. GAN+DM 混合方法虽然减少步数,但 discriminator 的额外训练阻碍模型收敛 [§1]
3. Consistency distillation 依赖预训练好的 diffusion teacher model,增加 pipeline 复杂度 [§1]
4. 之前的 consistency-based TTS (CoMoSpeech, Ye et al. 2023) 只在单说话人 LJSpeech 上验证,多说话人场景适用性未知 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CM-TTS 由四个组件构成 [§4.1, Fig 1]:
1. **Phoneme Encoder**: 多层 FFT blocks,结构与 FastSpeech2 一致(4 FFT blocks, hidden=256, kernel=9, filter=1024) [§5.3]
2. **Variance Adaptor**: 预测 pitch/energy/duration,与 FastSpeech2 设计一致 [§4.2]
3. **CM-Decoder**: 基于 consistency model 的 mel-spectrogram 生成器,采用非因果 WaveNet 结构 [§4.3]
4. **Vocoder**: HiFi-GAN 将 mel 转为波形 [§4.1]

总参数量: 28.6M [§5.3]

### 关键设计选择

#### 1. Consistency Training vs Consistency Distillation

[论文原文] 作者选择 consistency training 而非 distillation,原因是 distillation 依赖预训练 teacher model,增加了 pipeline 复杂度 [§3, §4]。

Consistency model 的核心是在 probability flow ODE 轨迹上施加自一致性约束 [§3, Eq. 4]:
- 对轨迹上任意时间步 t 和 t' 的样本,consistency function 映射到同一起始点: f(x_t, 0) = f(x_{t'}, t')
- 这使得单步生成成为可能: 直接从 f(x_T, T) 得到 x_0

参数化方式 [§4.3, Eq. 5]:
```
f_theta(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)
```
其中 c_skip(epsilon) = 1, c_out(epsilon) = 0,确保 f_theta(x_0, epsilon) = x_0。

#### 2. Online-Target EMA 训练

[论文原文] 使用双网络架构(类似 BYOL, Grill et al., 2020): online network f_theta 通过梯度更新, target network f_{theta-} 通过 EMA 更新 [§4.3, Eq. 6]:
```
theta- <- stopgrad(mu * theta- + (1 - mu) * theta)
```

[agent 解读] 这种设计避免了模式崩塌: target network 提供稳定的训练目标,而 online network 逐步学习。与 DiffGAN-TTS 的对抗训练相比,EMA 更新不存在 discriminator 训练不平衡的问题。

Consistency loss [§4.3, Eq. 7]:
```
L_CT = sum_{n>=1} E[lambda(t_n) * d(f_theta(x_{t+1}), f_{theta-}(x_t))]
```
其中 x_{t+1} = x_0 + t_{n+1}*z, x_t = x_0 + t_n*z (z ~ N(0,I)) [Eq. 8]

#### 3. WaveNet Decoder 结构

[论文原文] CM-Decoder 使用非因果 WaveNet 结构(1D conv + ReLU),与 DiffGAN-TTS 相同 [§4.3, §5.3]。Speaker-ID 通过 WaveNet residual blocks 转为 embedding。Diffusion step t 用正弦位置编码。mel decoder 包含 4 FFT blocks [§5.3]。

[agent 解读] 选择与 DiffGAN-TTS 相同的 decoder 结构使得两者的性能差异可以更纯粹地归因于训练范式(consistency training vs GAN)的不同,而非网络架构差异。

#### 4. Weighted Sampler

[论文原文] 训练中需要从 [1, N-1] 中采样索引 n 来计算 t_n [Eq. 9]。不同采样策略影响模型对 ODE 轨迹不同位置的学习 [§4.3.2]:

| Sampler | 权重 c_n | 设计思路 |
|---|---|---|
| Uniform | c_n = 1 | 等概率采样,baseline |
| Linear (ascending) | c_n = alpha * n | 大噪声位置采样更多 |
| Linear (descending) | c_n = alpha * (N-n) | 小噪声位置采样更多 |
| **Importance Sampling (IS)** | 基于历史 loss | 高 loss 位置采样更多 |

IS 公式 [§4.3.2]: c_n = (1-phi) * sum(L(t,j)) / sum(sum(L(i,j))) + phi,其中 L 为 (N-1)*H 的历史 loss 矩阵,H=10 为存储的历史 loss 数,phi 为平衡因子。

[论文原文] IS 根据历史 loss 动态调整采样概率,优先训练 loss 较大的位置,减轻随机采样带来的偏差 [§4.3.2]。

### 训练策略

总体损失 [§4.3.1, Eq. 11]:
```
L_CM-TTS = L_CT(theta, theta-) + L_recon
```

其中重建损失 [Eq. 10]:
```
L_recon = L_mel(x_0, x_hat_0) + 0.1*L_duration + 0.1*L_pitch + 0.1*L_energy
```
- L_mel: MAE (ground truth vs generated mel)
- L_duration/pitch/energy: MSE

训练配置 [§5.4]:
- 单张 V100 32GB
- Batch size 32, 300K steps
- 指数学习率衰减 (rate 0.999, 初始 10e-4)
- 按 Song et al. (2023) 的 schedule 周期性调整 N 和 mu
- VCTK 训练 ~34.2h, LJSpeech ~42.8h, LibriSpeech ~45.6h

推理 [§4.3.1, Fig 2]:
- 单步: 直接 f_theta(x_T, T) → x_0
- 多步: 交替去噪和注入噪声,可提升质量

## 实验

### 主实验 (VCTK, 多说话人)

| 指标 | CM-TTS (T=1) | DiffGAN-TTS (T=1) | FastSpeech2 | VITS | DiffSpeech | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOS | 3.96 | 3.45 | 3.68 | 3.67 | 2.92 | VCTK | [Table 1] |
| S.Cos | 0.8396 | 0.8284 | 0.8236 | 0.8154 | 0.7400 | VCTK | [Table 1] |
| melFID | 7.58 | 20.01 | 8.82 | 15.40 | 11.55 | VCTK | [Table 1] |
| WER | 0.0688 | 0.0809 | 0.0677 | 0.0451 | 0.5708 | VCTK | [Table 1] |
| RTF | 0.02 | 0.02 | 0.02 | 0.23 | 9.19 | VCTK | [Table 1] |

CM-TTS (T=2) 达到最高 S.Cos 0.8401 [Table 1]。

### LJSpeech (单说话人)

| 指标 | CM-TTS (T=1) | DiffGAN-TTS (T=1) | FastSpeech2 | CoMoSpeech | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS | 3.84 | 3.71 | 3.57 | 3.56 | [Table 8] |
| S.Cos | 0.9009 | 0.8959 | 0.8825 | 0.8666 | [Table 8] |
| melFID | 2.97 | 3.70 | 5.28 | 17.81 | [Table 8] |

### 零样本 (LibriTTS train → VCTK test)

| 指标 | CM-TTS (T=1) | DiffGAN-TTS (T=1) | 出处 |
| --- | --- | --- | --- |
| MOS | 3.87 | 3.46 | [Table 5] |
| S.Cos | 0.7108 | 0.6874 | [Table 5] |
| mfccFID | 157.91 | 283.77 | [Table 5] |

但在单说话人 LJSpeech 零样本场景 [Table 9, Table 10], DiffGAN-TTS 在 prosody similarity 上更优。

### Weighted Sampler 对比 (VCTK, T=1)

| Sampler | MOS | melFID | WER | S.Cos | 出处 |
| --- | --- | --- | --- | --- | --- |
| Uniform | 3.81 | 10.08 | 0.0872 | 0.8333 | [Table 3] |
| Linear (asc.) | 3.33 | 11.35 | 0.0822 | 0.8356 | [Table 3] |
| Linear (desc.) | 3.57 | 11.05 | 0.0861 | 0.8315 | [Table 3] |
| **IS** | **3.91** | **7.58** | **0.0688** | **0.8396** | [Table 3] |

IS 在 MOS (+0.10)、melFID (-2.50)、WER (-0.018) 上均显著优于 uniform。

### Ablation (VCTK, T=1)

| 配置 | melFID | WER | 出处 |
| --- | --- | --- | --- |
| CM-TTS (full) | 7.58 | 0.0688 | [Table 2] |
| w/o CM (无 consistency) | 10.74 | 0.0832 | [Table 2] |
| w/o IS (无 importance sampling) | 10.08 | 0.0872 | [Table 2] |

### IS 迁移到 DiffGAN-TTS

| 配置 | S.Cos | WER | 出处 |
| --- | --- | --- | --- |
| DiffGAN-TTS (T=2) | 0.8333 | 0.0827 | [Table 6] |
| DiffGAN-TTS (T=2) + IS | 0.8397 | 0.0720 | [Table 6] |

IS sampler 可迁移: 应用到 DiffGAN-TTS 后 WER 降低 13%,S.Cos 提升 [Table 6]。

### Padding in Loss

包含 padding 部分计算 L_mel 比不包含效果更好(l1 with padding MOS 3.91 vs without 3.81) [Table 4]。

## 局限性

1. **单说话人零样本不如 DiffGAN-TTS**: 多说话人场景一致优于 DiffGAN-TTS,但单说话人零样本(LJSpeech)中 DiffGAN-TTS 的 prosody similarity 更好 [Table 9]。[论文原文] 作者在结论中承认此限制 [§Conclusion]。
2. **网络结构未探索**: 论文专注于训练范式优化,未对 decoder 层数、残差模块等网络结构进行探索 [§Limitations]。
3. **仅限 TTS 任务**: 未扩展到声音生成等其他音频任务 [§Limitations]。
4. **评估局限**: MOS 评估仅 20 名听众、30 个样本,统计显著性有限 [§5.5]。不同 table 中 MOS 评估独立进行导致同一配置数值略有差异。
5. **时代局限**: [agent 解读] 2024 年的工作,未对比 flow matching (Voicebox, Matcha-TTS, F5-TTS) 等更现代的加速方法。且 consistency model 在 TTS 中的后续影响力远不如 flow matching 路线。
6. **开源但影响有限**: GitHub 开源,但相比同时期的 flow matching 方案,社区采用度有限。

## 点评

**优势**:
- 提供了 diffusion TTS 加速的第三条路线(consistency training),与 GAN 加速和蒸馏加速形成互补
- Weighted sampler 特别是 IS 的设计具有通用性,实验证明可迁移到 DiffGAN-TTS [Table 6]
- 实验覆盖 12 个指标,包含 ablation、sampler 对比、loss 设计、零样本等多角度验证
- 对 padding 在 loss 计算中的影响提供了有价值的实证 [Table 4]

**不足**:
- 未对比同时期的 flow matching 方案(Voicebox 2023, Matcha-TTS 2024 等)
- Decoder 结构直接沿用 DiffGAN-TTS,未探索是否有更适合 consistency training 的架构
- 零样本实验仅对比 DiffGAN-TTS 一个 baseline,不够充分
- [agent 解读] 从历史发展看,flow matching 而非 consistency model 最终成为 diffusion TTS 加速的主流路线。CM-TTS 代表了一条有价值但未成为主流的技术路线。

## 可复用的 idea

1. **Importance Sampler for ODE Training**: 根据历史 loss 动态调整 ODE 轨迹采样权重的方法具有通用性。可直接应用于 flow matching 训练中 t 的采样策略,重点训练 loss 较高的时间步。
2. **Padding-aware Loss**: 变长序列训练中,将 padding 部分纳入 loss 计算(引导模型学习静音生成)可能优于忽略 padding。
3. **Consistency Training for Single-step Generation**: 虽然 flow matching 成为主流,但 consistency training 无需 teacher model 的特性在资源受限场景仍有价值。
4. **IS 可迁移性验证方法**: 将新 training trick 迁移到已有 baseline (DiffGAN-TTS) 验证通用性的实验设计值得借鉴。
