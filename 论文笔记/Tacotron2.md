---
type: paper
tier: deep
title: "Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions"
arxiv_id: "1712.05884"
source: "Sources/Tacotron2.pdf"
authors: [Jonathan Shen, Ruoming Pang, Ron J. Weiss, Mike Schuster, Navdeep Jaitly, Zongheng Yang, Zhifeng Chen, Yu Zhang, Yuxuan Wang, RJ Skerry-Ryan, Rif A. Saurous, Yannis Agiomyrgiannakis, Yonghui Wu]
year: 2017
venue: "ICASSP 2018 (Google)"
tags: [TTS, seq2seq, attention, mel-spectrogram, WaveNet-vocoder, autoregressive]
concepts: ["[[MelSpectrogram]]", "[[NeuralVocoder]]", "[[DurationPredictor]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 0
status: draft
created: 2026-06-10
updated: 2026-06-10
---

## 速查

> [!summary] 速查
> - **一句话**: 将 Tacotron 的 seq2seq 思想与 WaveNet vocoder 结合,建立 text -> mel spectrogram -> waveform 的经典两阶段 TTS pipeline,MOS 4.53 接近人类语音 (4.58)
> - **路线**: Characters → Embedding → 3 Conv + BiLSTM Encoder → Location-Sensitive Attention → 2 LSTM Decoder + Pre-net → Mel Spectrogram → 5 Conv Post-net → Modified WaveNet (MoL) → 24kHz Waveform
> - **指标**: MOS 4.526 +- 0.066 (GT 4.582 +- 0.053) [Table 1]; Side-by-side vs GT: -0.270 +- 0.155 [Section 3.2]
> - **可借鉴**: (1) mel spectrogram 作为中间表示 — 比 linguistic features 简单且有效; (2) location-sensitive attention 稳定对齐; (3) Pre-net (always dropout) 作为信息瓶颈; (4) Post-net 残差细化 mel
> - **局限**: 两阶段分别训练,无联合优化; 自回归解码推理慢; 偶有发音错误 (尤其人名); 需要 WaveNet vocoder (也慢)

## 核心问题

**Tacotron 2 要解决什么问题?** [论文原文]

原始 Tacotron 使用 Griffin-Lim 合成波形,产生 "characteristic artifacts and lower audio quality" [Section 1]。直接以 WaveNet 作为 TTS 系统需要复杂的 linguistic features、F0 和 duration 预测 [Section 1]。

**目标**: 设计一个统一的全神经网络 TTS 系统,结合:
- Tacotron 风格的 seq2seq 模型 (简化文本前端)
- 修改版 WaveNet (基于 mel spectrogram 条件化,简化 vocoder 输入)

**核心假设**: [agent 解读] Mel spectrogram 作为中间表示,既保留了足够的声学信息供 WaveNet 重建高质量波形,又足够紧凑使 seq2seq 模型可以端到端学习从文本到声学特征的映射,从而消除对复杂 linguistic features 的依赖。

## 方法: 它怎么 work

> [!important] 区分来源
> [论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构 [Section 2, Fig 1]

**Stage 1 - Spectrogram Prediction Network (seq2seq)**:
1. **Character Embedding**: 512-dim learned embedding [Section 2.2]
2. **Encoder**: 3 x Conv1D (512 channels, kernel 5, BN + ReLU) → BiLSTM (512 units, 256/direction) [Section 2.2]
3. **Attention**: Location-sensitive attention (additive attention + cumulative attention weights + 32 location filters, kernel 31) [Section 2.2]
4. **Decoder**: 2-layer Pre-net (256 ReLU + dropout 0.5 always-on) → Attention LSTM (1024) → Decoder LSTM (1024) → Linear projection → Mel frame; 同时预测 stop token (sigmoid > 0.5) [Section 2.2]
5. **Post-net**: 5 x Conv1D (512 channels, kernel 5, BN + tanh, 最后一层 linear) → 残差加到 mel 输出 [Section 2.2]

**Stage 2 - Modified WaveNet Vocoder**:
- 30 dilated conv layers, 3 dilation cycles (1 to 2^9), 2 upsampling layers [Section 2.3]
- MoL output (10 components) at 24 kHz [Section 2.3]
- 条件输入: predicted mel spectrogram (非 linguistic features) [Section 2.3]

### 关键设计选择

#### 1. Mel Spectrogram 作为中间表示 [Section 2.1]
[论文原文] "a low-level acoustic representation: mel-frequency spectrograms" — 相比 WaveNet 原本使用的 linguistic features,mel spectrogram 是更简洁的声学表示,可以直接从波形计算,不需要文本分析系统。

STFT: 50ms frame, 12.5ms hop, Hann window → 80-channel mel filterbank (125-7600 Hz) → log compression (clip min 0.01) [Section 2.2]

**消融验证** [Table 3]: Mel+WaveNet (4.526) vs Linear+WaveNet (4.510) — mel 略优且更紧凑 (80 vs 1025 维)

#### 2. Location-Sensitive Attention [Section 2.2]
[论文原文] 基于 Chorowski et al. (2015) 的 additive attention,额外使用 cumulative attention weights 作为特征,鼓励模型 "move forward consistently through the input"。

- [agent 解读] 累积注意力权重是对齐稳定性的关键。没有它,decoder 可能重复或跳过子序列。后续 Tacotron 系列中,对齐问题一直是核心挑战。

#### 3. Pre-net 的 Always-on Dropout [Section 2.2]
[论文原文] Pre-net (2 FC layers, 256 units, ReLU) 在训练和推理时都使用 dropout 0.5。"we found that the pre-net acting as an information bottleneck was essential for learning attention"。

[agent 解读] 推理时保持 dropout 是一个非常规设计,它引入了输出随机性,使合成语音更具变化性,同时强迫 attention 机制学习稳健的对齐。

#### 4. 两阶段独立训练 [Section 3.1]
[论文原文] 先训练 spectrogram prediction network (64 batch, 1 GPU, Adam, teacher-forcing),再独立训练 WaveNet (128 batch, 32 GPUs, teacher-forcing on ground truth mel spectrograms)。

**消融验证** [Table 2]: WaveNet 在 predicted mel 上训练 + predicted mel 推理 = 4.526 (最佳); 在 GT mel 训练 + predicted mel 推理 = 4.362 — 说明 WaveNet 需要在 predicted (over-smoothed) features 上训练才能最优

## 实验

| 指标 | 本文 (Tacotron 2) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| MOS | 4.526 +- 0.066 | Parametric: 3.492; Concatenative: 4.166; WaveNet(Ling): 4.341; GT: 4.582 | 24.6h internal EN | [Table 1] |
| Side-by-side vs GT | -0.270 +- 0.155 | - | 同上 | [Section 3.2] |
| 100-sent custom MOS | 4.354 | - | 同上 | [Section 3.2] |
| WaveNet simplified (24L,4cyc,6dil) | 4.547 +- 0.056 | 30L baseline: 4.526 | - | [Table 4] |

**关键实验发现**:
1. **接近人类语音** [Table 1]: MOS 4.526,与 GT 4.582 差距仅 0.056,是当时最接近人类的 TTS 系统
2. **Mel 优于 linguistic features** [Table 1]: Tacotron2 (4.526) > WaveNet conditioned on linguistic features (4.341)
3. **Post-net 重要** [Section 3.3.3]: 去掉 post-net MOS 从 4.526 降至 4.429
4. **WaveNet 可简化** [Table 4]: 24 层 4 cycle 6 dilation 的小模型 (感受野仅 21ms) 仍达 4.547 MOS

## 局限性

1. **两阶段分别训练**: 无法联合优化全局目标,acoustic model 的 over-smoothing 传播到 vocoder [agent 解读]
2. **推理慢**: Decoder 自回归生成 mel + WaveNet 自回归生成波形,双重瓶颈 [agent 解读]
3. **发音鲁棒性**: 偶有 mispronunciation (6/100), skipped words (1/100), unnatural prosody (23/100) [Section 3.2]
4. **Attention 不稳定**: 长句或域外文本可能失败 (end-point prediction failed in 1 case) [Section 3.2]

## 点评

**历史地位**: [agent 解读] Tacotron 2 定义了现代 TTS 的标准 pipeline: text → mel → vocoder。这个两阶段架构影响了后续 3-4 年的主流研究方向,直到 VITS (2021) 将其统一为端到端系统:
- FastSpeech/FastSpeech 2: 替换自回归 decoder 为并行生成
- Glow-TTS: 用 normalizing flow 替换 attention
- HiFi-GAN: 替换 WaveNet vocoder 为并行 GAN vocoder
- VITS: 将 Tacotron 2 的两阶段彻底统一

**方法论贡献**:
1. 证明 mel spectrogram 是 TTS 的最佳中间表示 (简单、有效、通用)
2. Location-sensitive attention 成为后续 TTS attention 的基线
3. WaveNet vocoder 在 predicted features 上训练的重要性

## 可复用的 idea

1. **Mel spectrogram 作为通用中间表示**: 在 acoustic model 和 vocoder 之间提供标准化接口
2. **Location-sensitive attention**: 累积注意力权重促进单调前进,适用于任何语音合成对齐任务
3. **Pre-net always-on dropout**: 信息瓶颈 + 推理时随机性,提升合成多样性和对齐学习
4. **Post-net 残差细化**: 对粗糙的初始预测做局部细化,成本低收益明确
5. **在 predicted features 上训练 vocoder**: 缓解训练-推理特征分布不匹配

---

## 代码级分析

> [!info] 代码来源
> - 仓库: https://github.com/NVIDIA/tacotron2
> - commit: 185cd24
> - 分析日期: 2026-06-10

### 架构验证

代码与论文 Fig 1 完全一致:
- `model.py:Encoder` = 3 Conv + BiLSTM (encoder_embedding_dim=512, kernel=5)
- `model.py:Decoder` = Pre-net (2x256) + Attention LSTM (1024) + Decoder LSTM (1024) + Linear Projection + Gate
- `model.py:Attention` = Location-sensitive attention (additive + location conv 32 filters, kernel 31)
- `model.py:Postnet` = 5 Conv (512 channels, kernel 5, BN + tanh, 最后一层 linear)
- WaveNet vocoder 在 `waveglow/` 子模块中 (实际使用 WaveGlow 替代)

### 论文未写的实现细节

1. **Embedding 初始化** (`model.py:466`): 使用均匀分布 $U(-\sqrt{3} \cdot std, \sqrt{3} \cdot std)$, 其中 $std = \sqrt{2/(n_{symbols} + dim)}$,这是 Xavier uniform 的变体
2. **Attention score masking** (`model.py:80`): padding 位置用 `-inf` mask,确保 softmax 后权重为 0
3. **Gate threshold** (`model.py:443`): 推理时 sigmoid(gate_output) > 0.5 即停止,最大 1000 步保护
4. **Loss = MSE(mel) + MSE(mel_postnet) + BCE(gate)** (`loss_function.py:16-18`): pre-postnet 和 post-postnet 的 mel 都计算 MSE loss,gate 用 BCEWithLogitsLoss
5. **Zoneout 未使用**: 论文提到 LSTM 用 zoneout 0.1,但代码中用标准 dropout (`p_decoder_dropout=0.1`)
6. **Mask padding** (`model.py:487-495`): 训练时将 padding 位置的 mel 输出强制置 0,gate 置 1e3 (强制 sigmoid 接近 1)

### 训练 pipeline 拆解

数据流: 文本 → character indices → embedding → encoder → decoder (teacher-forcing with GT mel) → mel_out + mel_postnet + gate → MSE + BCE loss → Adam → checkpoint

- Teacher forcing: decoder 始终接收 GT mel 的前一帧作为输入
- 无 scheduled sampling, 无 attention guidance loss
- `hparams.py`: epochs=500, batch_size=64, lr=1e-3, weight_decay=1e-6, grad_clip=1.0
- 使用 NVIDIA Apex FP16 训练 (可选)

### 推理 pipeline 拆解

文本 → character indices → embedding → encoder.inference → decoder.inference (自回归,直到 gate > 0.5 或 max_steps=1000) → mel_out + mel_postnet → WaveGlow/WaveNet vocoder → 波形

- Pre-net 在推理时仍然开启 dropout (training=True 在 forward 中)
- 无 beam search, 无 length penalty

### 关键超参数表

| 参数 | 论文值 | 代码实际值 | 备注 |
|------|--------|-----------|------|
| encoder_embedding_dim | 512 | 512 | 一致 |
| encoder_n_convolutions | 3 | 3 | 一致 |
| encoder_kernel_size | 5 | 5 | 一致 |
| decoder_rnn_dim | 1024 | 1024 | 一致 |
| attention_rnn_dim | 1024 | 1024 | 一致 |
| prenet_dim | 256 | 256 | 一致 |
| attention_dim | 128 | 128 | 一致 |
| attention_location_filters | 32 | 32 | 一致 |
| attention_location_kernel | 31 | 31 | 一致 |
| n_mel_channels | 80 | 80 | 一致 |
| sampling_rate | 24000 (论文) | 22050 (代码) | 差异: 代码用 LJSpeech 标准 |
| batch_size | 64 (论文单GPU) | 64 | 一致 |
| learning_rate | 1e-3 (decay) | 1e-3 (固定) | 论文有 decay,代码无 |
| optimizer | Adam | Adam | 一致 |

### 复现 checklist (基于代码)

- [ ] 环境依赖: PyTorch, tensorflow (仅用于 hparams), numpy, scipy, librosa, unidecode, inflect
- [ ] 数据准备: LJSpeech 下载, filelists/ 中有预生成的 train/val split
- [ ] 预训练模型依赖: 无 (从头训练); vocoder 需要 WaveGlow 预训练权重
- [ ] 训练命令: `python train.py --output_directory=outdir --log_directory=logdir`
- [ ] 推理命令: 通过 `inference.ipynb` Jupyter notebook
- [ ] 已知坑: tensorflow 依赖仅用于 HParams 类 (可用 tf.compat.v1); WaveGlow 子模块需要额外 clone; 论文的 WaveNet vocoder 与代码附带的 WaveGlow 不同

### 代码质量与可复现性评估

- **工程质量**: 4/5 - NVIDIA 官方实现,代码简洁直接,无过度封装
- **文档完善度**: 3/5 - README 基本够用,但缺少详细的训练指南和预训练模型下载
- **社区活跃度**: 2/5 - 2019 年后不再维护,但仍有大量 fork
- **复现难度**: 2/5 - 流程简单,LJSpeech 数据易获取,单 GPU 可训练
