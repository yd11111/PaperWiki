---
type: concept
title: "Duration Predictor"
aliases: [时长预测器, Length Regulator, Duration Model, 音素时长预测]
category: "architecture-component"
tags: [TTS, duration, alignment, non-autoregressive, acoustic-model]
key_papers: ["[[论文笔记/VITS|VITS]]", "[[论文笔记/E2TTS|E2 TTS]]", "[[论文笔记/DMOSpeech2|DMOSpeech 2]]", "[[论文笔记/SESD|SESD]]", "[[论文笔记/VoiceFlow|VoiceFlow]]", "[[论文笔记/VeryAttentiveTacotron|Very Attentive Tacotron (Battenberg et al., 2025)]]", "[[论文笔记/Bridge-TTS|Bridge-TTS]]", "[[论文笔记/FlexSpeech|FlexSpeech]]", "[[论文笔记/TTS-Transducer|TTS-Transducer]]", "[[论文笔记/OZSpeech|OZSpeech]]", "[[论文笔记/RapFlow-TTS|RapFlow-TTS]]", "[[论文笔记/FMSD-TTS|FMSD-TTS]]", "[[论文笔记/ZipVoice|ZipVoice]]", "[[论文笔记/DS-TTS|DS-TTS]]", "[[论文笔记/SMLLE|SMLLE]]", "[[论文笔记/FNH-TTS|FNH-TTS]]", "[[论文笔记/DiFlow-TTS|DiFlow-TTS]]", "[[论文笔记/Cross-LingualF5-TTS|Cross-Lingual F5-TTS]]", "[[论文笔记/VARSTok|VARSTok]]", "[[论文笔记/TMD-TTS|TMD-TTS]]", "[[论文笔记/BFA|BFA]]", "[[论文笔记/Flamed-TTS|Flamed-TTS]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Non-autoregressiveTTS]]", "[[Attention-basedTTS]]", "[[Text-to-SpeechPipeline]]", "[[ProsodyModeling]]"]
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

详见 [[MusicalScoreEncoder]]。

## 关键论文

- FastSpeech (Ren et al., NeurIPS 2019): 首次引入 duration predictor 到端到端 TTS
- FastSpeech 2 (Ren et al., ICLR 2021): MFA-based duration + log-domain loss
- DurIAN (Yu et al., IS 2019): AR 解码 + duration predictor (混合形态)
- Glow-TTS (Kim et al., NeurIPS 2020): Monotonic Alignment Search 自动获取 duration
- EATS (Donahue et al., ICLR 2021): 端到端可微 duration optimization

## 相关概念

- [[Non-autoregressiveTTS]]: duration predictor 使 NAR 生成成为可能
- [[Attention-basedTTS]]: duration predictor 替代 attention 的对齐功能
- [[ProsodyModeling]]: duration 是韵律的核心维度之一
- Montreal Forced Alignment (MFA): 最常用的 duration 标签提取工具
- BFA (Rehman et al., 2025): CTC-based forced aligner,比 MFA 快 240 倍,支持多语言;详见 [[论文笔记/BFA|BFA]]

## 演进

HMM state duration (SPSS) → Attention alignment (Tacotron, 2017) → Duration Predictor (FastSpeech, 2019; 回归显式 duration) → Monotonic Alignment Search (Glow-TTS, 2020; 内部对齐) → E2E differentiable duration (EATS, 2021) → T2D model (MaskGCT, 2024; 独立 duration 生成阶段) → RL-optimized duration policy (DMOSpeech 2, 2025; GRPO 优化总时长预测) → AR duration + DPO (FlexSpeech, 2025; phone-level AR next-token prediction + DPO 偏好对齐) → MoE-DP (FNH-TTS, 2026; Switch-Transformer 多专家结构 + speaker-conditioned routing) → Inference-time AR duration steering (TED-TTS, 2026; training-free segment-level duration embedding + EOS logit modulation) → Sampling-time distribution matching (VoXtream2, 2026; 在线直方图匹配 + 滑动窗口自校正,支持动态 mid-utterance 变速) → Duration-as-external-condition (MAGIC-TTS, 2026; token-level duration+pause 作为外部数值 conditioning,而非内部预测/对齐变量)

### DMOSpeech 2 RL-based Duration Optimization (Li et al., AAAI 2026)

- **GRPO 优化 duration policy**: 将 duration predictor 建模为 300-class 分类 (100ms bins) 的 stochastic policy,用 GRPO 以 SIM+WER 为 reward 优化。仅需 1.5K 额外训练步,在 Seed-TTS-en 上 WER 从 3.750 降到 1.752 [Table 1]
- **关键发现**: RL-optimized duration 的 WER (1.752) 甚至优于使用 ground truth duration (1.821),说明最优 duration 不等于真实 duration [Table 3]
- **计算效率**: 利用 4-step DMD-distilled student 生成样本计算 reward,避免传统 RL 需数百步采样的开销
- 详见 [[论文笔记/DMOSpeech2|DMOSpeech 2]]

### FNH-TTS MoE Duration Predictor (Meng et al., 2026)

- **MoE 结构化 duration modeling**: 首次将 Mixture-of-Experts 引入 Duration Predictor,用 1D Conv + Switch-Transformer blocks (8 experts) 构建 MoE-DP。属于 DDP 类别的结构创新,与 SDP 概率化和 GRPO/DPO RL 优化方向正交 [§2.1]
- **Speaker-conditioned routing**: router 输入为 x + speaker_embedding,使路由决策本身就是说话人感知的,不同说话人的 phoneme 可路由到不同专家 [Eq. 2]
- **Duration-Vocoder 耦合发现**: MoE-DP 单独使用反而降低 MOS (LJ: 3.92 vs VITS 4.26),因为更丰富的 duration 变化超出 HiFi-GAN 合成能力;配合 VOCOS vocoder + CoMBD/SBD 判别器后恢复并超越 (LJ: 4.48) [Table 1]
- **JS divergence 评估**: 用 phoneme-level Jensen-Shannon divergence 衡量 duration 对齐质量,MoE-DP (0.053/0.039) 优于 DDP (0.057/0.044) 和 SDP (0.087/0.066) [Table 4]
- 详见 [[论文笔记/FNH-TTS|FNH-TTS]]

### FlexSpeech AR Duration + DPO (Ma et al., 2025)

- **AR next-token prediction for duration**: 将 phone-level duration 离散化为 0-99 帧的分类标签,用 encoder-decoder Transformer 做 next-token prediction,显式建模 Markov 依赖
- **DPO 偏好对齐**: 用人工标注的 win-lose duration 对做 Direct Preference Optimization,仅 50 对数据即可显著改善自然度和稳定性
- **与 DMOSpeech 2 对比**: DMOSpeech 2 用 GRPO + 自动 reward;FlexSpeech 用 DPO + 人工标注。两者共同结论: duration predictor 是 TTS pipeline 中偏好优化最有效的作用点
- **WER**: Seed-TTS test-zh 1.20% (低于 GT 1.26%), test-en 1.81% [Table 1]
- **风格迁移**: 仅微调 duration model (~100 samples DPO),acoustic model 不动,即可完成风格迁移
- 详见 [[论文笔记/FlexSpeech|FlexSpeech]]

### TED-TTS Inference-time AR Duration Steering (Liang et al., 2026)

- **Training-free segment-level duration control**: 首次在 AR TTS 中实现推理时 segment-level 时长控制,不修改模型参数。利用 IndexTTS2 的 duration embedding table (与 semantic positional embedding 共享),在 AR 解码中根据在线 text-semantic progress 差异动态重新查询 duration table 修正 embedding
- **双层控制**: (1) Local duration embedding steering — 比例控制器根据 MSA 对齐估计的 text/semantic progress 差异调整 duration embedding,每 5 步更新一次,最大调整 10 tokens; (2) Global EOS logit modulation — 非最终 segment 抑制 EOS,最终 segment 根据 progress ratio 动态调整 EOS logit (bias 范围 [-5.0, +15.0])
- **与已有方法的根本区别**: DMOSpeech 2/FlexSpeech/FNH-TTS 都是训练/优化 duration predictor 模块; TED-TTS 完全不训练,在 AR 解码过程中通过 embedding lookup 和 logit 修改实现控制
- **Duration error**: 全设置 3.21-3.39% (vs baseline 5.78-12.03%),且不同 scaling factor 下误差保持稳定 [Table 4]
- 详见 [[论文笔记/TED-TTS|TED-TTS]]

### T5Gemma-TTS PM-RoPE Multilingual Validation (Arata & Kurihara, 2026)

- **PM-RoPE 多语言验证**: 首次在预训练多语言 encoder-decoder 骨干(T5Gemma 4B)上验证 VoiceStar 的 PM-RoPE 对英语以外语言的泛化性。170K 小时三语言(EN/ZH/JA)训练后,日语 DA 0.79(phoneme-count 估计)/ 1.00(oracle target);关闭 PM-RoPE 导致 CER 0.129→0.982,近乎完全合成失败 [Table 3]
- **Subword 输入的 open question**: 与 VoiceStar 使用 phoneme 不同,T5Gemma-TTS 使用 SentencePiece subword 输入,牺牲单调对齐特性;phoneme vs subword 对 PM-RoPE 效果的影响未做消融
- **与 VoiceStar 的区别**: VoiceStar 从零训练 + phoneme 输入 + 英语 only; T5Gemma-TTS 预训练初始化 + subword + 多语言
- 详见 [[论文笔记/T5Gemma-TTS|T5Gemma-TTS]]

### VoXtream2 Distribution Matching SRC (Torgashov et al., 2026)

- **Sampling-time distribution matching**: 不修改 duration predictor 本身,在 AR 推理时通过直方图匹配校正 duration token 采样分布。给定目标 SPS 对应的 duration state 直方图 (Ptarget),计算当前预测分布 (Pcurrent) 与过去 3s 窗口累积分布 (Pacc) 的差异,用 W = exp(β * (log10(Ptarget) - log10(Pacc))) 重加权采样 [§3.5, Eq. 2-3]
- **Self-correcting 机制**: 滑动窗口 Pacc 使控制信号自动适应实际生成状态,偏差越大校正力度越强;β=5 为 controllability-intelligibility trade-off
- **动态 mid-utterance 变速**: Ptarget 可在生成过程中随时改变,实现帧级语速控制;渐变场景 Pearson corr 0.70-0.83,突变场景 0.62-0.66 [VoXtream2 Table 5]
- **与已有方法的根本区别**: DMOSpeech 2/FlexSpeech/FNH-TTS 修改/优化 duration predictor 模块; TED-TTS 通过 embedding steering 控制; VoXtream2 在 token 采样概率空间做在线分布匹配,三者正交
- 详见 [[论文笔记/VoXtream2|VoXtream2]]

### MAGIC-TTS Duration-as-External-Condition (Mai et al., 2026)

- **Duration 从预测变量变为外部条件**: 与上述所有方法根本不同 — MAGIC-TTS 不优化 duration 的预测准确性,而是训练 acoustic generator 可靠地遵从外部给定的 token-level content duration 和 pause 数值。将 duration 视为 conditioning signal 而非 alignment intermediate [§3.1]
- **Zero-value correction**: 零值 pause 贡献零残差 (g(x)-g(0)),防止大量 pause=0 的 token 产生 dense bias 淹没 content duration 控制信号 [§3.4]
- **Cross-validated supervision**: Stable-ts × MFA 交叉验证筛选高置信度时长标签 (B@150, 202K utterances / 230h from 13.6M entries),解决 content duration 对标注边界精度的高敏感性 [§3.3]
- **控制效果**: C-MAE 10.56ms / C-Corr 0.918 (controlled),spontaneous 模式 C-MAE 36.88ms / C-Corr 0.588 (与 F5-TTS Base 相当) [Table 1]
- **Quality trade-off**: Seed-TTS-Eval EN WER 1.993→3.434,ZH CER 1.665→2.215 (spontaneous mode) [Table 7]
- **与本页其他方法的互补性**: DMOSpeech/FlexSpeech/TED-TTS 优化"预测更好的 duration",MAGIC-TTS 优化"遵从外部给定的 duration"。两个方向可组合: predictor 提供 default,用户通过 MAGIC-style interface 局部编辑
- 详见 [[论文笔记/MAGIC-TTS|MAGIC-TTS]]
