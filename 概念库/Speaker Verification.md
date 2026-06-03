---
type: concept
title: "Speaker Verification"
aliases: [说话人验证, Speaker Recognition, SV, 说话人识别, Speaker Identification]
category: "evaluation-and-security"
tags: [speaker-identity, evaluation, voice-cloning, anti-spoofing, security, SECS, SV-EER]
key_papers: ["[[论文笔记/Survey-Voice Cloning|Azzuni & El Saddik 2025]]", "[[论文笔记/TTSDS|TTSDS]]", "[[论文笔记/TITW|TITW]]", "[[论文笔记/SafeSpeech|SafeSpeech]]", "[[论文笔记/DMOSpeech|DMOSpeech]]", "[[论文笔记/DiVISe|DiVISe (Liu et al., 2025)]]", "[[论文笔记/Koel-TTS|Koel-TTS]]"]
origin_paper: ""
related_concepts: ["[[Speaker Embedding]]", "[[Voice Cloning Taxonomy]]", "[[Anti-spoofing and Deepfake Detection]]", "[[Speaker Adaptation]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Speaker Verification (SV) 是判断一段语音是否属于特定目标说话人的技术。在 voice cloning 研究中,SV 具有双重角色:

1. **评估工具**: 衡量合成语音与目标说话人的相似度 (SECS, SV-EER)
2. **训练/安全组件**: 作为训练信号提升 speaker similarity,或作为防线检测伪造语音

## 作为评估指标

### SECS (Speaker Embedding Cosine Similarity)
- 计算生成语音与参考语音的 speaker embedding 余弦相似度
- 公式: $\text{SECS} = \frac{E(\text{generated}) \cdot E(\text{reference})}{\|E(\text{generated})\| \cdot \|E(\text{reference})\|}$
- 值域 [-1, 1],越高越好 (↑)
- 主观性: 依赖所选 speaker encoder 的品质

**常用 Speaker Encoder 架构** (Survey 汇总):

| 架构 | 特点 | 来源 |
|------|------|------|
| X-vector | TDNN + 统计池化 | Snyder et al., 2018 |
| GE2E | 端到端 speaker verification loss | Wan et al., 2018 |
| ECAPA-TDNN | 强调通道注意力与传播聚合 | Desplanques et al., 2020 |
| TitaNet-L | 1D 深度可分离卷积 + global context | Koluguri et al., 2022 |
| WavLM | 大规模自监督预训练表示 | Chen et al., 2022 |
| XLSR-53 | 跨语言自监督表示 | Conneau et al., 2021 |
| CAM++ | 上下文感知掩码 + 高效 speaker verification | Wang et al., 2023 |

**报告差异**: Survey 指出 SECS 结果因 speaker encoder 不同而差异显著。Tables V/VI 中同一系统使用 x-vector vs GE2E vs ECAPA-TDNN 报告的 SECS 值可差 0.1-0.3。因此 SECS 的可比性受限于所用 encoder 的统一性。

### SV-EER (Speaker Verification Equal Error Rate)
- 衡量 speaker verification 系统性能的标准指标
- EER: False Acceptance Rate (FAR) = False Rejection Rate (FRR) 时的错误率
- 越低越好 (↓)
- 比 SECS 更严格,考虑了决策阈值

## 作为训练组件

Voice cloning survey 揭示了 SV 在训练中的三种用途:

### 1. Feedback Constraint (反馈约束)
Cai et al. 提出从 speaker verification 网络向 TTS 框架迁移知识,用 feedback constraint 机制促进声音克隆训练。将 Tacotron-based 模型与 LDE-based verification 模型结合,增强 unseen speakers 的 speaker similarity。

### 2. Adversarial Training Signal (对抗训练信号)
- Nakai et al.: 引入多任务对抗训练,通过 speaker verification 判别真实/合成语音 + 判别说话人是否存在
- ACAI (Adversarially Constrained Autoencoder Interpolation): 正则化项提升 unseen speakers 性能
- DINO-VITS: 集成 self-supervised DINO loss 提升 speaker representation learning 和噪声鲁棒性

### 3. Loss Function Component (损失函数)
- GE2E loss: 多个系统用于训练 speaker encoder (VStyclone, SC-GlowTTS)
- Angular softmax loss / Angular prototypical loss: 用于学习 discriminative speaker embedding (Cooper et al.)
- Speaker consistency loss: Latent Filling 用于跨语言 ZS-TTS 中保持说话人一致性

## Speaker Encoder 在 Voice Cloning 中的角色差异

| Cloning 类型 | Speaker Encoder 角色 | 训练/推理 | 冻结/可训练 |
|-------------|---------------------|---------|-----------|
| Speaker Adaptation | 提取初始 embedding 或验证质量 | 主要训练时 | 通常冻结 |
| Few-shot VC | 提取 representation + 验证 | 训练+推理 | 部分可训练 |
| Zero-shot VC | 核心组件: 推理时提取 speaker identity | 推理时必须 | 通常冻结 (预训练) |
| Evaluation | 计算 SECS 相似度 | 评估时 | 冻结 |

## 在 TTS 中的应用

- **Zero-shot TTS 评估**: 所有 ZS-TTS 论文均报告 SECS (Tables V/VI 中覆盖 30+ 系统)
- **Few-shot TTS 评估**: Table IV 中所有 FS-TTS 系统报告 SIM 指标
- **训练信号**: 提升合成语音与目标说话人的相似度
- **安全检测**: 检测语音是否为合成/伪造 (→ [[Anti-spoofing and Deepfake Detection]])
- **多说话人 TTS**: 开集 (open-set) 说话人 TTS 的核心使能技术

## 关键论文

- Wan et al., "GE2E: Generalized End-to-End Loss for Speaker Verification" (2018): 奠基性 loss
- Desplanques et al., "ECAPA-TDNN" (Interspeech 2020): 当前最常用的 speaker encoder
- Cai et al., "From Speaker Verification to Multi-speaker TTS" (Interspeech 2020): SV → TTS 知识迁移
- Cooper et al., "Zero-shot multi-speaker TTS with state-of-the-art neural speaker embeddings" (ICASSP 2020): 系统比较不同 speaker embedding 对 ZS-TTS 的影响
- Chung, "In Defence of Metric Learning for Speaker Recognition" (Interspeech 2020): Angular prototypical loss

## 相关概念

- [[Speaker Embedding]]: SV 的输出表示,用于 TTS 中的说话人条件化
- [[Voice Cloning Taxonomy]]: SV 在四类 cloning 方法中均有应用
- [[Anti-spoofing and Deepfake Detection]]: SV 的安全对偶应用
- [[Speaker Adaptation]]: SV 作为训练信号增强适应效果
- [[Speech Factorization]]: SV 训练的 speaker encoder 常用于 disentanglement 的 speaker branch

## 演进

GMM-UBM i-vector (传统方法) → DNN d-vector (Variani et al., 2014) → X-vector TDNN (Snyder et al., 2018) → GE2E end-to-end (Wan et al., 2018) → ECAPA-TDNN (Desplanques, 2020) → Self-supervised (WavLM/XLSR, 2022) → CAM++ context-aware (Wang et al., 2023)
