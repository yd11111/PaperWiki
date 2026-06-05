---
type: concept
title: "Diffusion Model"
aliases: [DDPM, Denoising Diffusion Probabilistic Model, Score-based Generative Model, Diffusion Probabilistic Model, DPM, 扩散模型]
category: "generative-model"
tags: [generative-model, diffusion, DDPM, score-matching, SDE, ODE, audio-generation]
key_papers: ["[[论文笔记/Survey-AudioDiffusionModels|Survey-Audio Diffusion Models]]", "[[论文笔记/NaturalSpeech2|NaturalSpeech 2]]", "[[论文笔记/Seed-VC|Seed-VC]]", "[[论文笔记/USM-VC|USM-VC]]", "[[论文笔记/TortoiseTTS|Tortoise TTS]]", "[[论文笔记/LatentLM|LatentLM]]", "[[论文笔记/CLEAR|CLEAR]]", "[[论文笔记/VibeVoice|VibeVoice]]", "[[论文笔记/SiTok|SiTok]]", "[[论文笔记/VoiceFlow|VoiceFlow]]", "[[论文笔记/Bridge-TTS|Bridge-TTS]]", "[[论文笔记/DiffCSS|DiffCSS]]", "[[论文笔记/DMOSpeech|DMOSpeech]]", "[[论文笔记/RapFlow-TTS|RapFlow-TTS]]", "[[论文笔记/DLPO|DLPO]]", "[[论文笔记/HiStyle|HiStyle]]"]
origin_paper: "Sohl-Dickstein et al., Deep Unsupervised Learning Using Nonequilibrium Thermodynamics, 2015"
related_concepts: ["[[ConditionalFlowMatching]]", "[[ScoreMatching]]", "[[NeuralVocoder]]", "[[Diffusion-basedVocoder]]", "[[Diffusion-basedTTS]]", "[[Classifier-FreeGuidance]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Diffusion Model 是一类基于逐步加噪-去噪过程的生成模型。其核心思想源自非平衡热力学 [§2.2]: 通过 **前向过程 (forward process)** 将数据逐步加噪至纯噪声,再通过学习的 **反向过程 (reverse process)** 从噪声中恢复数据。

### 前向过程 (Forward Process)

定义数据分布 x_0 ~ q(x_0),前向过程构成 Markov chain,逐步添加 Gaussian 噪声:

```
q(x_{1:T} | x_0) := prod_{t=1}^{T} q(x_t | x_{t-1})
q(x_t | x_{t-1}) := N(x_t; sqrt(1 - beta_t) * x_{t-1}, beta_t * I)
```

其中 beta_t 为 **噪声调度 (noise schedule)**,控制每步加噪程度。经 T 步后 x_T 近似为标准 Gaussian 噪声。

### 反向过程 (Reverse Process)

DDPM (Ho et al., 2020) 通过训练神经网络估计噪声,从 x_T 逐步恢复 x_0。训练目标为预测每步添加的噪声 epsilon,损失函数为简单的均方误差 (MSE)。

### 两大理论分支

虽然从不同角度出发,DDPM 和 score-based generative model 在一定条件下等价 [§2.2]:

| 分支 | 代表工作 | 核心思想 | 数学工具 |
|------|----------|----------|----------|
| **DDPM** | Ho et al. (2020) | 离散步马尔科夫链去噪 | 变分下界 (VLB) |
| **Score-based** | Song & Ermon (2019, 2020) | 估计数据分布的 score function | 随机微分方程 (SDE) |

### SDE/ODE 统一视角

Song et al. (2020) 将 diffusion 统一为连续时间 SDE 框架:
- **SDE (随机)**: 前向为 Ito SDE,反向需知 score function
- **ODE (确定)**: probability flow ODE,与 SDE 具有相同边际分布,但路径确定

ODE 视角是 [[ConditionalFlowMatching]] 的理论桥梁: CFM 直接学习 ODE 向量场,而非通过 SDE 的 score function 间接求解。

## 在 TTS 中的应用

Zhang et al. (2023) 将 diffusion 在 TTS 中的应用分为三个阶段 [§3, Table 1]:

### 1. 声学模型 (Acoustic Model)
从文本生成 mel spectrogram,是 diffusion TTS 最主要的应用方式:
- **开创性工作**: Diff-TTS (2021), Grad-TTS (2021)
- **高效加速**: ProDiff (知识蒸馏), DiffGAN-TTS (GAN 加速, 1 步生成)
- **多说话人**: Grad-TTS with ILVR, Grad-StyleSpeech, Guided-TTS/2

### 2. 声码器 (Vocoder)
从 mel spectrogram 生成波形:
- **开创性工作**: WaveGrad (2020), DiffWave (2020)
- **高效化**: BDDM (7 步), InferGrad (3x 加速)
- **统计改进**: PriorGrad (自适应先验), DDGM (Gamma 噪声)

### 3. 端到端 (End-to-End)
直接从文本/音素生成波形:
- WaveGrad 2: 音素 → 波形,无需 mel spectrogram
- CRASH: score-based SDE 端到端鼓声合成
- DAG: 全频段音频端到端生成

### 4. 语音增强 (Speech Enhancement)
- **去噪**: DiffuSE, CDiffuSE, SGMSE/SGMSE+ (score-based, STFT 域)
- **超分辨率**: NU-Wave, NU-Wave 2
- **去混响**: UVD (基于 DDRM 的无监督方法)
- **源分离**: DiffSep (SDE-based)

## 与 Flow Matching 的关系

Diffusion Model 和 [[ConditionalFlowMatching]] 是近亲关系:

| 维度 | Diffusion (SDE) | Flow Matching (ODE) |
|------|-----------------|---------------------|
| 路径类型 | 随机 (Stochastic) | 确定 (Deterministic) |
| 训练目标 | score function / noise | velocity field |
| 推理步数 | 典型 50-1000 | 典型 4-20 |
| 理论连接 | SDE 的 probability flow ODE | 直接回归 ODE 路径 |
| 代表 TTS | Grad-TTS, DiffWave | Voicebox, F5-TTS, Matcha-TTS |

**演进关系**: Diffusion (DDPM/SDE, 2020) --> Probability Flow ODE --> Rectified Flow --> Flow Matching (2023)

## 关键论文

- Sohl-Dickstein et al. (2015): Deep Unsupervised Learning Using Nonequilibrium Thermodynamics — 首次提出 DPM
- Ho et al. (2020): Denoising Diffusion Probabilistic Models (DDPM) — 奠基性工作
- Song & Ermon (2019): Generative Modeling by Estimating Gradients of the Data Distribution — score-based 分支
- Song et al. (2020): Score-Based Generative Modeling through SDEs — SDE 统一框架
- Song et al. (2020): DDIM — 确定性采样加速
- Zhang et al. (2023): A Survey on Audio Diffusion Models — 音频 diffusion 综述

## 相关概念

- [[ConditionalFlowMatching]]: diffusion SDE 的 ODE 近亲,更少推理步数
- [[ScoreMatching]]: diffusion 的理论基础之一
- [[Diffusion-basedVocoder]]: diffusion 在声码器中的应用
- [[Diffusion-basedTTS]]: diffusion 在 TTS 声学模型/端到端中的应用
- [[Classifier-FreeGuidance]]: diffusion 条件生成的主流引导方法
- [[NeuralVocoder]]: 声码器家族,diffusion 是其中一个分支

## 演进

Boltzmann Machine (能量模型) --> DPM (Sohl-Dickstein, 2015) --> DDPM (Ho et al., 2020) + NCSN (Song & Ermon, 2019) --> SDE 统一框架 (Song et al., 2020) --> DDIM 加速 (2020) --> Classifier-Free Guidance (2022) --> Flow Matching (Lipman, 2023) --> Audio 领域: DiffWave/Grad-TTS (2020-21) --> 高效化: ProDiff/BDDM (2022) --> CFM 取代 diffusion 成为 TTS 主流 (2023-)
