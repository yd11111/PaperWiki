---
type: concept
title: "Duration Predictor"
aliases: [时长预测器, Length Regulator, Duration Model, 音素时长预测]
category: "architecture-component"
tags: [TTS, duration, alignment, non-autoregressive, acoustic-model]
key_papers: ["[[论文笔记/VITS|VITS]]", "[[论文笔记/E2 TTS|E2 TTS]]", "[[论文笔记/DMOSpeech 2|DMOSpeech 2]]", "[[论文笔记/SESD|SESD]]", "[[论文笔记/VoiceFlow|VoiceFlow]]", "[[论文笔记/Very Attentive Tacotron|Very Attentive Tacotron (Battenberg et al., 2025)]]", "[[论文笔记/Bridge-TTS|Bridge-TTS]]", "[[论文笔记/FlexSpeech|FlexSpeech]]", "[[论文笔记/TTS-Transducer|TTS-Transducer]]", "[[论文笔记/OZSpeech|OZSpeech]]", "[[论文笔记/RapFlow-TTS|RapFlow-TTS]]", "[[论文笔记/FMSD-TTS|FMSD-TTS]]", "[[论文笔记/ZipVoice|ZipVoice]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Non-autoregressive TTS]]", "[[Attention-based TTS]]", "[[Text-to-Speech Pipeline]]", "[[Prosody Modeling]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Duration Predictor 是 TTS 系统中预测每个输入音素(或字符)对应多少帧声学特征的模块。它是非自回归 TTS 的核心组件,用于替代 encoder-decoder attention 来桥接文本序列与声学序列之间的长度不匹配。

**数学表达**: 给定音素序列 $p_1, p_2, ..., p_N$, duration predictor 预测 $d_1, d_2, ..., d_N$, 其中 $d_i$ 为第 $i$ 个音素对应的帧数, $\sum d_i = T$ (总帧数)。

**典型架构**:
```
Phoneme Hidden → [Conv1D × 2-3] → [LayerNorm] → [ReLU] → [Linear] → Duration (integer)
```

## 工作机制

### Length Regulator (FastSpeech)
将 encoder 输出的 phoneme hidden states 按预测 duration 扩展:
- 若 phoneme $p_i$ 的 duration 为 $d_i = 3$, 则将 $h_i$ 复制 3 次
- 扩展后的序列长度 = mel spectrogram 帧数
- Decoder 在此扩展序列上并行生成所有帧

### Duration 标签获取

| 方法 | 类型 | 代表工作 | 特点 |
|------|------|----------|------|
| AR teacher attention | 外部 | FastSpeech | 从 Tacotron attention 提取 |
| CTC alignment | 外部 | TalkNet | ASR 的 CTC 对齐提供 duration |
| HMM forced alignment | 外部 | FastSpeech 2, DurIAN | MFA 工具, 最常用 |
| Dynamic programming | 内部 | AlignTTS | DP 搜索最优对齐 |
| Monotonic alignment search | 内部 | Glow-TTS | 单调约束下搜索对齐 |
| Soft DTW | E2E | EATS | 可微 duration, 端到端优化 |

### 端到端 vs 非端到端优化

**非端到端** (大多数): duration 标签由外部工具获取(MFA, CTC, teacher attention), predictor 仅学习预测该标签, 不接收 mel loss 梯度。

**端到端**: EATS 通过 duration interpolation + soft DTW loss 让 duration 直接接收来自波形生成的梯度。Parallel Tacotron 2 也采用类似思路。

## Duration 的重要性

Duration prediction 的质量直接影响:
1. **韵律自然度**: duration 是韵律的核心维度之一
2. **鲁棒性**: 显式 duration 完全消除跳字/重复问题
3. **可控性**: 通过缩放 duration 直接控制语速
4. **生成速度**: 有了 duration 就能并行生成

**Survey 观点**: "It is very interesting that the early SPSS uses duration for alignments, and then the sequence-to-sequence models remove duration and use attention instead, and the later TTS models discard attention and use duration again, which is a kind of technique renaissance."

## 在 TTS 中的应用

### FastSpeech 系列
- FastSpeech: duration from teacher attention → MSE loss
- FastSpeech 2: duration from MFA → MSE loss on log-duration

### Variance Adaptor (FastSpeech 2)
Duration predictor 作为 variance adaptor 的一部分:
```
Encoder Output → Duration Predictor → Length Regulator → Pitch Predictor → Energy Predictor → Decoder
```

### 现代 LLM-TTS 中的 duration
- MaskGCT: 专门的 duration prediction 阶段 (T2D model)
- IndexTTS2: 通过共享位置编码实现隐式 duration control
- CosyVoice: flow matching 内部隐式处理 duration
- SESD (Lovelace et al., 2024): 仅预测 utterance-level 总时长 (fine-tune ByT5 为 seq2seq duration predictor, RMSE 1.4s),diffusion 内部隐式解决 phoneme alignment,完全避免 phoneme duration 标注

### VITS Stochastic Duration Predictor (Kim et al., ICML 2021)
- **概率 duration 建模**: 首个用 flow-based 模型学习音素时长的概率分布 (非确定性预测) [§2.2.2]
- **核心设计**: 引入 variational dequantization (u) + variational data augmentation (v) 将离散 duration 转为连续分布 [§2.2.2]
- **架构**: DDSConv residual blocks + neural spline flows [§2.5.5]
- **与 Glow-TTS 对比**: Glow-TTS 用 MAS 估计对齐后训练确定性 duration predictor (MSE loss),VITS 则学习 duration 分布的变分下界
- **实验验证**: stochastic vs deterministic (DDP) 版本 MOS 4.43 vs 4.39 [Table 1],stochastic 产生更多样化的 F0 和时长分布 [Fig 2, Fig 3]
- 详见 [[论文笔记/VITS|VITS]]

## 在 SVS 中的时长预测

SVS 中的时长预测与 TTS 有本质差异 [Pan et al., 2026, §4.1]:

**乐谱约束**: SVS 的音符时长由乐谱 (note duration, BPM) 显式提供,duration predictor 的任务从"预测时长"变为"在乐谱约束下细化音素级时长"。

**Melisma 处理**: 一个音节跨越多个不同音高的音符 (花腔),形成一对多映射,TTS 中不存在此问题。

**三种对齐方式**:
1. 外部强制对齐 (MFA/Praat) → 音素/音节时长标签
2. 可学习单调对齐 (VISinger) → 随机 duration 建模节奏不确定性
3. 可学习上采样 (He et al., 2023) → 与 FastSpeech length regulator 类似但接受乐谱约束

**评估指标**: Duration RMSE/MAE 和 Duration Prediction Accuracy 是 SVS 独有的评估维度。

详见 [[Musical Score Encoder]]。

## 关键论文

- FastSpeech (Ren et al., NeurIPS 2019): 首次引入 duration predictor 到端到端 TTS
- FastSpeech 2 (Ren et al., ICLR 2021): MFA-based duration + log-domain loss
- DurIAN (Yu et al., IS 2019): AR 解码 + duration predictor (混合形态)
- Glow-TTS (Kim et al., NeurIPS 2020): Monotonic Alignment Search 自动获取 duration
- EATS (Donahue et al., ICLR 2021): 端到端可微 duration optimization

## 相关概念

- [[Non-autoregressive TTS]]: duration predictor 使 NAR 生成成为可能
- [[Attention-based TTS]]: duration predictor 替代 attention 的对齐功能
- [[Prosody Modeling]]: duration 是韵律的核心维度之一
- Montreal Forced Alignment (MFA): 最常用的 duration 标签提取工具

## 演进

HMM state duration (SPSS) → Attention alignment (Tacotron, 2017) → Duration Predictor (FastSpeech, 2019; 回归显式 duration) → Monotonic Alignment Search (Glow-TTS, 2020; 内部对齐) → E2E differentiable duration (EATS, 2021) → T2D model (MaskGCT, 2024; 独立 duration 生成阶段) → RL-optimized duration policy (DMOSpeech 2, 2025; GRPO 优化总时长预测) → AR duration + DPO (FlexSpeech, 2025; phone-level AR next-token prediction + DPO 偏好对齐)

### DMOSpeech 2 RL-based Duration Optimization (Li et al., AAAI 2026)

- **GRPO 优化 duration policy**: 将 duration predictor 建模为 300-class 分类 (100ms bins) 的 stochastic policy,用 GRPO 以 SIM+WER 为 reward 优化。仅需 1.5K 额外训练步,在 Seed-TTS-en 上 WER 从 3.750 降到 1.752 [Table 1]
- **关键发现**: RL-optimized duration 的 WER (1.752) 甚至优于使用 ground truth duration (1.821),说明最优 duration 不等于真实 duration [Table 3]
- **计算效率**: 利用 4-step DMD-distilled student 生成样本计算 reward,避免传统 RL 需数百步采样的开销
- 详见 [[论文笔记/DMOSpeech 2|DMOSpeech 2]]

### FlexSpeech AR Duration + DPO (Ma et al., 2025)

- **AR next-token prediction for duration**: 将 phone-level duration 离散化为 0-99 帧的分类标签,用 encoder-decoder Transformer 做 next-token prediction,显式建模 Markov 依赖
- **DPO 偏好对齐**: 用人工标注的 win-lose duration 对做 Direct Preference Optimization,仅 50 对数据即可显著改善自然度和稳定性
- **与 DMOSpeech 2 对比**: DMOSpeech 2 用 GRPO + 自动 reward;FlexSpeech 用 DPO + 人工标注。两者共同结论: duration predictor 是 TTS pipeline 中偏好优化最有效的作用点
- **WER**: Seed-TTS test-zh 1.20% (低于 GT 1.26%), test-en 1.81% [Table 1]
- **风格迁移**: 仅微调 duration model (~100 samples DPO),acoustic model 不动,即可完成风格迁移
- 详见 [[论文笔记/FlexSpeech|FlexSpeech]]
