---
type: concept
title: "Score Matching"
aliases: [Score Function, Score-based Model, NCSN, Noise Conditional Score Network, 分数匹配]
category: "generative-model"
tags: [generative-model, score-matching, diffusion, SDE, speech-enhancement]
key_papers: ["[[论文笔记/Survey-AudioDiffusionModels|Survey-Audio Diffusion Models]]", "[[论文笔记/FlowDec|FlowDec]]", "[[论文笔记/VoiceFlow|VoiceFlow]]", "[[论文笔记/StableForm-TTS|StableForm-TTS]]", "[[论文笔记/Bridge-TTS|Bridge-TTS]]", "[[论文笔记/DMOSpeech|DMOSpeech]]", "[[论文笔记/RapFlow-TTS|RapFlow-TTS]]"]
origin_paper: "Song & Ermon, Generative Modeling by Estimating Gradients of the Data Distribution, NeurIPS 2019"
related_concepts: ["[[DiffusionModel]]", "[[ConditionalFlowMatching]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Score Matching 是一类通过估计数据分布的 **score function**(即对数概率密度的梯度 nabla_x log p(x))来进行生成建模的方法。它是 diffusion model 的理论基础之一,与 DDPM 虽从不同视角出发,但在一定条件下等价 [§2.2]。

### Score Function

Score function 定义为数据分布对数密度的梯度:
```
s(x) = nabla_x log p(x)
```

它指向数据密度增长最快的方向。知道 score function 后,可通过 Langevin dynamics 从任意初始点迭代采样,逐步移向高概率区域,从而生成新数据。

### 核心思想

直接估计 nabla_x log p(x) 面临两个困难:
1. p(x) 未知(这正是要学的)
2. 低密度区域 score 估计不准确

**解决方案 (Song & Ermon, 2019)**: 使用多尺度噪声扰动:
- 对数据加不同强度的 Gaussian 噪声,得到一系列噪声分布
- 在每个噪声水平上用神经网络 (NCSN) 估计 score function
- 高噪声填充低密度区域,低噪声保留数据精细结构

### 与 DDPM 的统一

Zhang et al. (2023) 综述指出 [§2.2]:

| 分支 | 训练目标 | 采样方式 | 等价条件 |
|------|----------|----------|----------|
| DDPM | 预测噪声 epsilon | 离散步去噪 | 当步数 T → 无穷 |
| Score-based | 估计 score function | Langevin dynamics / SDE | 连续时间 SDE |

Song et al. (2020) 的 SDE 统一框架表明:
- DDPM 的离散去噪过程对应离散化的反向 SDE
- NCSN 的 Langevin 采样对应连续 SDE 的数值求解
- 两者共享相同的 score function 作为核心

## 在 TTS 中的应用

### Grad-TTS (Popov et al., 2021)

Grad-TTS 基于 SDE 形式化 diffusion 过程 [§3.2.1]:
- 采用 U-Net (来自 WaveGrad) 作为 mel spectrogram 生成的 score 估计器
- 基于 SDE 而非 DDPM,是 score-based 方法在 TTS 中的代表性应用

### SGMSE / SGMSE+ (语音增强)

Score-based 方法在语音增强领域尤为成功 [§4.1.1]:
- SGMSE (Welker et al., 2022): 在 STFT 域使用 score function 引导去噪
- SGMSE+ (Richter et al., 2022): 使用 NCSN++ 架构,在语音增强和去混响上达到 SOTA
- 相比 DDPM-based 方法 (DiffuSE, CDiffuSE),score-based 方法更自然、更少 artifacts

### CRASH (Rouard & Hadjeres, 2021)

CRASH 使用 score-based SDE 实现端到端鼓声合成 [§3.4]:
- 基于 score function 估计的 U-Net
- 支持 class-mixing sampling 生成混合鼓声

### UNIVERSE (Serra et al., 2022)

UNIVERSE 使用 score-based diffusion 构建通用语音增强系统 [§4.3]:
- 可处理 55 种不同类型的音频退化
- 包含 conditioner network + score-based generator

## 关键论文

- Song & Ermon (2019): Generative Modeling by Estimating Gradients of the Data Distribution — 首次提出 NCSN
- Song & Ermon (2020): Improved Techniques for Training Score-Based Generative Models — 改进训练
- Song et al. (2020): Score-Based Generative Modeling through SDEs — SDE 统一框架
- SGMSE+ (Richter et al., 2022): 语音增强 SOTA [§4.1.1]
- Grad-TTS (Popov et al., 2021): score-based TTS [§3.2.1]

## 相关概念

- [[DiffusionModel]]: Score matching 是 diffusion 的理论基础之一
- [[ConditionalFlowMatching]]: 与 score-based SDE 通过 probability flow ODE 相连

## 演进

Hyvarinen (2005, 原始 score matching) --> Denoising Score Matching (Vincent, 2011) --> NCSN (Song & Ermon, 2019, 多尺度噪声) --> SDE 统一框架 (Song et al., 2020) --> 音频应用: Grad-TTS (2021), SGMSE (2022) --> Flow Matching 吸收 ODE 思路,减少采样步数 (2023-)
