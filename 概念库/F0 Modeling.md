---
type: concept
title: "F0 Modeling"
aliases: [基频建模, Pitch Modeling, Fundamental Frequency, 音高建模, F0 Prediction, Pitch Contour]
category: "technique"
tags: [SVS, TTS, pitch, F0, vibrato, prosody, acoustic-feature]
key_papers: ["[[论文笔记/Survey-Synthetic Singers|Synthetic Singers (Pan et al., 2026)]]", "[[论文笔记/HierSpeech++|HierSpeech++]]", "[[论文笔记/Make-A-Voice|Make-A-Voice]]", "[[论文笔记/Seed-VC|Seed-VC]]", "[[论文笔记/MambaVoiceCloning|MambaVoiceCloning (2026)]]", "[[论文笔记/StableForm-TTS|StableForm-TTS]]", "[[论文笔记/TechSinger|TechSinger]]", "[[论文笔记/Spotlight-TTS|Spotlight-TTS]]", "[[论文笔记/MiSTR|MiSTR]]"]
origin_paper: "Pan et al., Synthetic Singers: A Review of Deep-Learning-based SVS Approaches, 2026"
related_concepts: ["[[Singing Voice Synthesis]]", "[[Prosody Modeling]]", "[[Musical Score Encoder]]", "[[Duration Predictor]]", "[[Diffusion-based TTS]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

F0 (Fundamental Frequency, 基频) 是声带振动频率的物理量,直接决定语音/歌声的音高。F0 Modeling 是对 F0 轮廓 (contour) 进行预测、控制和合成的技术。在 TTS 中 F0 是韵律的一个维度 (见 [[Prosody Modeling]]),但在 SVS 中 F0 是核心约束维度 — 合成歌声必须精确跟随乐谱指定的音高 [Pan et al., 2026, §2]。

**F0 在 TTS vs SVS 中的地位差异**:

| 维度 | TTS 中的 F0 | SVS 中的 F0 |
|------|-----------|-----------|
| 约束来源 | 语言韵律 (柔性) | 乐谱 MIDI pitch (刚性) |
| 精度要求 | 相对自然即可 | 绝对音高精确 (半音级) |
| 评估指标 | 无专用 F0 指标 | FFE, F0 RMSE, F0 相关系数 |
| Vibrato | 不涉及 | 必须建模 (振幅、频率、相位) |
| UV 边界 | 简单处理 | 影响歌声自然度的关键 |
| 建模地位 | Variance adaptor 的一个分支 | 独立核心模块 |

## F0 的物理维度

- **Voiced/Unvoiced (V/UV)**: 声带是否振动 — 元音有 F0,清辅音无 F0
- **Pitch contour**: F0 随时间的变化轨迹
- **Vibrato**: 歌声特有的周期性音高波动 (频率 5-8 Hz, 幅度 0.5-2 半音)
- **Portamento/Glide**: 音符间的滑音过渡

## 在 SVS 中的建模方法

### 1. 显式 F0 预测 (Explicit Prediction)
级联 SVS 系统中将 F0 作为独立预测目标:

- **简单回归**: 类似 FastSpeech 2 的 pitch predictor,但条件化乐谱 MIDI pitch
- **Diffusion-based**: RMSSinger (He et al., 2023) 使用扩散模型预测 F0 轮廓,生成更自然、可控的 F0 轨迹 [§2]
- **源滤波器模型**: SiFiSinger (Cui et al., 2024) 引入源模块生成 F0 控制的激励信号 (excitation signal),通过滤波器模块整形为频谱,实现物理可解释的 F0 控制 [§3.2]
- **Flow Matching-based**: [[论文笔记/TechSinger|TechSinger]] (Guo et al., 2025) 将 F0 视为一维连续数据,用 rectified flow matching 训练向量场估计器预测 F0 轮廓,以乐谱+技巧编码为条件,比 L1 回归更好地建模技巧→F0 的复杂映射

### 2. Vibrato 建模 [§2, §A.1]
歌声中的 vibrato 需要专门建模:

- **参数化 vibrato**: Song et al. (2022) 的 DL 模型控制 vibrato 的多个方面 (振幅、频率、起始延迟)
- **数据增强**: F0 扰动策略 — 向训练数据添加小 vibrato 并微调 UV 边界,改善跨域泛化 [§A.1]
- Vibrato 是 SVS 区别于 TTS 的标志性表现力维度

### 3. 对抗解耦 (Adversarial Disentanglement)
Kim et al. (2022) 引入对抗多任务学习框架,解耦 timbre 和 pitch 特征 [§3.1]。使 F0 控制不影响音色,音色迁移不影响音高。

### 4. 端到端隐式 F0
VITS 类端到端系统 (VISinger) 中 F0 被隐式编码在 VAE latent 中,不作为显式中间表示。

## 在 TTS 中的 F0 建模

### Variance Adaptor (FastSpeech 2)
Pitch predictor 作为 variance adaptor 的一个分支:
```
Encoder → Duration Predictor → Length Regulator → [Pitch Predictor + Energy Predictor] → Decoder
```
预测帧级 F0 值,训练目标为 ground-truth F0 的 MSE loss。

### Flow/Diffusion 隐式建模
VITS, Glow-TTS 等系统中 F0 被生成模型隐式捕获,不作为显式预测目标。

### LLM 时代
VALL-E / CosyVoice 等系统中 F0 信息编码在 codec token 中,由 in-context learning 隐式复制。

## SVS 的 F0 评估指标

| 指标 | 定义 | 用途 |
|------|------|------|
| F0 Frame Error (FFE) | F0 帧与 ground-truth 的误差率 | 音高精度评估 [§5.3] |
| F0 RMSE | F0 帧级均方根误差 | 音高精度评估 [§5.3] |
| F0 Pearson Correlation | F0 轮廓的线性相关系数 | F0 轮廓形状匹配 [§5.3] |
| Pitch Accuracy | 合成 F0 与乐谱目标 pitch 的吻合度 | SVS 核心指标 [§5.3] |

详见 [[SVS Evaluation Metrics]]。

## 关键论文

- FastSpeech 2 (Ren et al., 2021): TTS 中显式 pitch predictor 的标杆
- DiffSinger (Liu et al., 2022a): 强调 F0 在歌声合成中的核心地位
- RMSSinger (He et al., 2023): 扩散 pitch predictor,F0 建模里程碑
- SiFiSinger (Cui et al., 2024): 源滤波器模型实现物理可解释的 F0 控制
- Song et al. (2022): DL-based vibrato 多维度控制
- Kim et al. (2022): 对抗多任务解耦 timbre 与 pitch

## 相关概念

- [[Prosody Modeling]]: F0 是韵律的 pitch 维度,SVS 中独立性更强
- [[Singing Voice Synthesis]]: F0 是 SVS 的核心约束
- [[Musical Score Encoder]]: 提供 F0 的目标 MIDI pitch
- [[Duration Predictor]]: 与 F0 共同决定歌声的时间-频率结构
- [[Diffusion-based TTS]]: 扩散模型在 F0 预测中的应用 (RMSSinger)

## 演进

SPSS 参数化 F0 → FastSpeech 2 pitch predictor (TTS, 2020) → DiffSinger 强调 F0 核心性 (SVS, 2022) → 对抗 timbre-pitch 解耦 (Kim et al., 2022) → Vibrato 参数化控制 (Song et al., 2022) → Diffusion pitch predictor (RMSSinger, 2023) → 源滤波器 F0 激励 (SiFiSinger, 2024)
