---
type: paper
tier: deep
title: "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"
arxiv_id: "2006.04558"
source: "Sources/FastSpeech2.pdf"
authors: [Yi Ren, Chenxu Hu, Xu Tan, Tao Qin, Sheng Zhao, Zhou Zhao, Tie-Yan Liu]
year: 2020
venue: "ICLR 2021 (Zhejiang University + Microsoft)"
tags: [TTS, non-autoregressive, variance-adaptor, duration-prediction, pitch-prediction, energy-prediction]
concepts: ["[[Non-autoregressiveTTS]]", "[[DurationPredictor]]", "[[MelSpectrogram]]", "[[NeuralVocoder]]"]
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
> - **一句话**: 非自回归 TTS 系统,通过 variance adaptor (duration + pitch + energy predictor) 直接从 phoneme 并行生成 mel spectrogram,消除了 FastSpeech 的 teacher-student distillation 依赖,质量超越自回归基线
> - **路线**: Phonemes → Encoder (4 FFT blocks) → Variance Adaptor [Duration (MFA) → Length Regulator → Pitch (CWT) → Energy] → Decoder (6 FFT blocks) → Mel Spectrogram → Parallel WaveGAN → Waveform; FastSpeech 2s 变体直接生成波形
> - **指标**: MOS 3.83 +- 0.08 (GT 4.30, Tacotron2 3.70) [Table 1a]; CMOS +0.885 vs FastSpeech [Table 1b]; 训练时间 17.02h (FastSpeech 53.12h, 3.12x speedup) [Table 2]; 推理 47.8x 加速 vs Transformer TTS [Table 2]
> - **可借鉴**: (1) 用 MFA 替代 teacher model 提取 duration,更准确且训练更简单; (2) Continuous Wavelet Transform 建模 pitch 变化; (3) pitch + energy 作为额外条件缓解 one-to-many mapping; (4) FastSpeech 2s 端到端 NAR text-to-waveform
> - **局限**: 依赖外部 MFA 对齐工具和 pitch 提取; MOS 仍低于 GT 较多 (3.83 vs 4.30); 多说话人未验证; vocoder 仍独立训练

## 核心问题

**FastSpeech 2 要解决什么问题?** [论文原文]

FastSpeech (2019) 虽然实现了非自回归并行合成,但有三个问题 [Section 1]:
1. **Teacher-student pipeline 复杂**: 需要先训练自回归 teacher,再蒸馏 student,两阶段
2. **Teacher 的 mel 信息损失**: teacher 生成的 mel 比 GT 简单化,丢失了 pitch/energy/prosody 细节
3. **Teacher attention 提取的 duration 不够准确**: attention map 的对齐不如专业对齐工具

**核心假设**: [agent 解读] TTS 的 one-to-many mapping 问题 (同一文本 → 多种可能的语音) 可以通过显式引入 variance information (duration + pitch + energy) 来缓解,无需依赖 teacher model 的知识蒸馏。

## 方法: 它怎么 work

> [!important] 区分来源
> [论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构 [Section 2.2, Fig 1]

1. **Encoder**: 4 个 Feed-Forward Transformer (FFT) blocks, 每个含 multi-head self-attention + 2 Conv1D layers [Section 2.2, Section 3.1]
2. **Variance Adaptor**: Duration Predictor → Length Regulator → Pitch Predictor → Energy Predictor [Section 2.3, Fig 1b]
3. **Decoder**: 6 个 FFT blocks [Section 3.1]
4. **Linear Output Layer**: 投影到 80-dim mel spectrogram [Section 3.1]
5. **PostNet** (可选): 残差细化 [代码实现]

### 关键设计选择

#### 1. 直接使用 GT Mel + MFA Duration [Section 2.2]
[论文原文] 与 FastSpeech 不同,"we directly train the FastSpeech 2 model with ground-truth target instead of the simplified output from teacher"。
- Duration: 使用 Montreal Forced Alignment (MFA) 工具提取 phoneme-level duration,比 teacher attention map 更准确
- **消融验证** [Table 5]: MFA duration (delta=12.47ms) vs teacher duration (delta=19.68ms); CMOS +0.195 使用 MFA

#### 2. Pitch Predictor + CWT [Section 2.3]
[论文原文] 直接预测 pitch contour 的分布与 GT 差异大。使用 Continuous Wavelet Transform (CWT) 将连续 pitch 序列分解为 pitch spectrogram,在频率域预测:
- 训练: GT pitch (F0) → quantize to 256 bins (log scale) → pitch embedding → add to hidden; 同时 GT pitch → CWT → pitch spectrogram → MSE target for pitch predictor
- 推理: pitch predictor → predicted pitch spectrogram → iCWT → pitch contour → quantize → embedding
- **消融验证** [Table 6]: 去掉 pitch CMOS -0.245 (FS2) / -1.130 (FS2s); 不用 CWT CMOS -0.185 (FS2)

#### 3. Energy Predictor [Section 2.3]
[论文原文] Energy = L2 norm of STFT frame amplitude, quantized to 256 bins, embedded and added to hidden sequence.
- 训练: GT energy → MSE target; 推理: predicted energy
- **消融验证** [Table 6]: 去掉 energy CMOS -0.040 (FS2) / -0.160 (FS2s)

#### 4. FastSpeech 2s (端到端变体) [Section 2.4]
[论文原文] 不经过 mel spectrogram 中间表示,直接从 hidden sequence 生成波形:
- Waveform decoder: 基于 WaveNet 结构 (non-causal conv + gated activation), 用转置卷积上采样
- 对抗训练: Multi-resolution STFT loss + LSGAN discriminator (Parallel WaveGAN)
- 推理时丢弃 mel decoder,仅用 waveform decoder

### 训练策略

**总损失** [代码 model/loss.py]:
$$L = L_{mel} + L_{mel\_postnet} + L_{duration} + L_{pitch} + L_{energy}$$

- $L_{mel}$, $L_{mel\_postnet}$: MAE (L1 loss) on mel spectrogram
- $L_{duration}$: MSE on log(duration + 1)
- $L_{pitch}$: MSE on pitch (spectrogram or contour)
- $L_{energy}$: MSE on energy values

**训练细节** [Section 3.1, Appendix]:
- Optimizer: Adam
- 数据集: LJSpeech (13,100 clips, ~24h)
- 外部对齐: MFA (Montreal Forced Alignment)
- 外部 pitch: 未在论文正文明确,代码中使用 pyworld

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| MOS | FastSpeech 2: 3.83 +- 0.08 | GT: 4.30; Tacotron2: 3.70; FastSpeech: 3.68 | LJSpeech | [Table 1a] |
| MOS | FastSpeech 2s: 3.71 +- 0.09 | (同上) | LJSpeech | [Table 1a] |
| CMOS (vs FastSpeech) | FastSpeech 2: 0.000 (ref) | FastSpeech: -0.885 | LJSpeech | [Table 1b] |
| Training Time | 17.02h | Transformer TTS: 38.64h; FastSpeech: 53.12h | - | [Table 2] |
| Inference Speedup | 47.8x (FS2); 51.8x (FS2s) | FastSpeech: 48.5x | vs Transformer TTS | [Table 2] |

**关键实验发现**:
1. **质量超越 FastSpeech 和自回归基线** [Table 1]: MOS 3.83 > Tacotron2 (3.70) > FastSpeech (3.68)
2. **训练加速 3.12x** [Table 2]: 消除 teacher-student pipeline 后训练时间从 53h 降至 17h
3. **Pitch 是最重要的 variance** [Table 6]: 去掉 pitch 导致 CMOS -0.245 (FS2), -1.130 (FS2s)
4. **CWT 有效** [Table 3, Table 6]: pitch 的统计矩 ($\sigma$, $\gamma$, $\mathcal{K}$) 和 DTW 距离更接近 GT
5. **FastSpeech 2s 端到端可行** [Table 1a]: MOS 3.71,虽低于 FS2+PWG (3.83) 但推理更快 (51.8x)

## 局限性

1. **依赖外部工具**: MFA 对齐 + pitch extraction (pyworld) 都是外部依赖 [Section 4]
2. **MOS 距 GT 仍有差距**: 3.83 vs 4.30 GT,差距 0.47 仍然明显 [Table 1a]
3. **仅单说话人实验**: 论文只在 LJSpeech (单人 24h) 上验证 [Section 3.1]
4. **Pitch/energy embedding 是 quantized**: 256-bin 量化引入信息损失 [agent 解读]
5. **无 attention 可视化**: 使用外部 duration,丧失了可解释的对齐可视化 [agent 解读]

## 点评

**历史地位**: [agent 解读] FastSpeech 2 确立了 NAR TTS 的标准范式: encoder + variance adaptor + decoder。其设计理念 (显式建模 duration/pitch/energy 来缓解 one-to-many mapping) 被广泛采用,影响了后续大量工作:
- PortaSpeech, AdaSpeech: 在 FS2 基础上扩展
- VITS: 用 VAE + MAS 替代外部 duration/alignment
- NaturalSpeech: 在 FS2 架构上引入 VAE
- 大量工业级 TTS 系统以 FS2 为 backbone

**方法论贡献**:
1. 证明了直接用 GT target 训练 NAR TTS 优于 teacher distillation
2. CWT pitch modeling 成为后续 pitch 建模的参考方案
3. FastSpeech 2s 是早期 fully end-to-end NAR TTS 的重要尝试

## 可复用的 idea

1. **Variance Adaptor 架构**: 将 TTS 中的 one-to-many mapping 分解为可控的 variance factors,通用思路
2. **MFA 替代 teacher model**: 用专业对齐工具获取 duration 比 attention map 更准确
3. **CWT pitch modeling**: 在频率域预测 pitch 变化比直接预测 contour 更稳定
4. **Quantized embedding for continuous features**: 将 pitch/energy 量化后用 embedding 表示,简单有效
5. **Length Regulator**: 按 duration 重复 hidden states 实现 phoneme→frame 扩展,几乎零成本

---

## 代码级分析

> [!info] 代码来源
> - 仓库: https://github.com/ming024/FastSpeech2
> - commit: d4e79eb
> - 分析日期: 2026-06-10
> - 说明: 社区最高星第三方实现,非作者官方代码

### 架构验证

代码与论文架构基本一致,但有简化:
- `model/fastspeech2.py:FastSpeech2`: Encoder → VarianceAdaptor → Decoder → mel_linear → PostNet
- `model/modules.py:VarianceAdaptor`: DurationPredictor + LengthRegulator + PitchPredictor + EnergyPredictor
- `model/modules.py:VariancePredictor`: 2 Conv1D + ReLU + LayerNorm + Dropout + Linear (共享结构)
- `transformer/`: Encoder 和 Decoder 使用标准 FFT blocks (self-attention + conv)

**与论文差异**:
1. **CWT pitch 未实现**: 代码直接预测原始 pitch 值 (frame-level 或 phoneme-level),未使用 CWT/iCWT 变换
2. **Pitch/Energy quantization**: 使用 `torch.bucketize` 将连续值量化到 256 bins,然后 embedding
3. **PostNet**: 论文未详细描述,代码实现为标准 5-layer conv postnet (类似 Tacotron2)

### 论文未写的实现细节

1. **Duration 取 log** (`model/loss.py:42`): `log_duration_targets = torch.log(duration_targets.float() + 1)`,loss 在 log domain 计算,防止长 duration 主导梯度
2. **Pitch/Energy bins 初始化** (`model/modules.py:48-71`): 从预处理统计 (stats.json) 读取 min/max,线性或对数均匀划分 256 bins,bins 作为 nn.Parameter(requires_grad=False)
3. **Length Regulator 实现** (`model/modules.py:167-193`): 简单的 for 循环,逐 batch 逐 phoneme repeat hidden vector,非常高效但无法 GPU 并行 (batch 内长度不同)
4. **Mel loss 用 L1** (`model/loss.py:74-75`): mel_loss 和 postnet_mel_loss 都用 MAE (L1),不是 MSE
5. **Mask 处理** (`model/modules.py:247-248`): variance predictor 输出在 mask 位置强制置 0
6. **Multi-speaker 支持** (`model/fastspeech2.py:29-41`): 可选 speaker embedding,加到 encoder output 上

### 训练 pipeline 拆解

数据流: Phonemes → TextDataset (phoneme + mel + duration + pitch + energy) → FastSpeech2 forward → (mel_out, postnet_out, pitch_pred, energy_pred, duration_pred) → L1(mel) + MSE(duration,pitch,energy) → Adam → checkpoint

- 预处理: `prepare_align.py` + MFA 对齐 + `preprocess.py` 提取 pitch/energy/duration/mel
- 训练: `train.py` 统一训练所有预测头

### 推理 pipeline 拆解

Phonemes → Encoder → Duration Predictor → exp(pred) - 1 → round → clamp(min=0) → Length Regulator → Pitch/Energy Predictor → Decoder → mel_linear → PostNet → Mel Spectrogram → HiFi-GAN/PWG → 波形

- 可通过 p_control, e_control, d_control 参数控制 pitch/energy/duration 的缩放
- 推理完全并行 (每句一次前向)

### 关键超参数表

| 参数 | 论文值 | 代码实际值 (LJSpeech) | 备注 |
|------|--------|-----------|------|
| encoder_layer | 4 FFT | 4 | 一致 |
| decoder_layer | 4 FFT (论文) | 6 | 代码多 2 层 |
| encoder_hidden | 256 | 256 | 一致 |
| n_heads | 2 | 2 | 一致 |
| conv_filter_size | 1024 | 1024 | 一致 |
| conv_kernel_size | [9,1] | [9,1] | 一致 |
| variance_filter | 256 | 256 | 一致 |
| variance_kernel | 3 | 3 | 一致 |
| variance_dropout | 0.5 | 0.5 | 一致 |
| n_bins | 256 | 256 | 一致 |
| n_mel_channels | 80 | 80 | 一致 |
| vocoder | Parallel WaveGAN | HiFi-GAN/MelGAN | 代码支持多种 |
| pitch modeling | CWT | 直接预测 | 差异: 代码未实现 CWT |

### 复现 checklist (基于代码)

- [ ] 环境依赖: PyTorch, numpy, pyworld, librosa, g2p_en, tqdm, yaml
- [ ] 数据准备: LJSpeech 下载 → `prepare_align.py` → MFA 安装和运行 → `preprocess.py`
- [ ] 预训练模型依赖: HiFi-GAN 预训练 vocoder (需另行下载)
- [ ] 训练命令: `python train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml`
- [ ] 推理命令: `python synthesize.py --text "..." --restore_step XXXX -p ... -m ... -t ...`
- [ ] 已知坑: MFA 安装可能有依赖冲突; CWT pitch 未实现 (与论文有差距); `preprocessed_data/` 目录结构必须正确; pyworld pitch 提取有时不稳定

### 代码质量与可复现性评估

- **工程质量**: 3/5 - 代码组织合理但有些硬编码路径,config 系统用 YAML 较清晰
- **文档完善度**: 3/5 - README 提供基本步骤,但 MFA 安装和数据预处理细节不够
- **社区活跃度**: 3/5 - 仍有 PR 和 Issue 活动,支持多数据集
- **复现难度**: 3/5 - 预处理步骤较多 (MFA + pitch + energy),但每步都有脚本; 注意 CWT 未实现
