---
type: paper-note
title: "A Survey on Audio Diffusion Models: Text To Speech Synthesis and Enhancement in Generative AI"
authors: [Chenshuang Zhang, Chaoning Zhang, Sheng Zheng, Mengchun Zhang, Maryam Qamar, Sung-Ho Bae, In So Kweon]
affiliation: KAIST, Kyung Hee University, Beijing Institute of Technology
year: 2023
venue: "arXiv:2303.13336"
tier: card
tags: [survey, diffusion, TTS, speech-enhancement, vocoder, DDPM, score-matching, cold-start]
created: 2026-06-02
updated: 2026-06-02
---

## 概要

首篇聚焦于 audio diffusion model 的综述,覆盖 diffusion 在语音合成 (TTS) 和语音增强 (speech enhancement) 两大任务中的应用。互补于已有综述: 既不像 Tan et al. (2021) 局限于 TTS 整体架构,也不像通用 diffusion 综述 (Cao et al., 2022) 忽视音频领域的进展。

## 核心贡献

1. **统一背景**: 简明介绍 DDPM 前向/反向过程和 score-based generative model,建立 diffusion 在音频中的理论基础 [§2.2]
2. **TTS 三阶段分类**: 将 diffusion TTS 按 diffusion 介入阶段分为 acoustic model / vocoder / end-to-end 三类 [§3, Table 1]
3. **语音增强分类**: 按功能分为 enhancement by removing (去噪/去混响) 和 enhancement by adding (超分辨率/带宽扩展),加上 source separation / voice conversion 等任务 [§4, Table 4]
4. **实验汇总**: 汇集 acoustic model [Table 2]、vocoder [Table 3]、speech enhancement [Table 5] 的实验结果对比

## 知识提取成果

本文用于概念库冷启动 #4,提取 diffusion model 在音频/语音领域的核心领域知识。

### 新建概念页 (5 个)

| 概念 | 核心内容 |
|------|----------|
| [[DiffusionModel]] | DDPM 前向/反向过程、score-based 分支、SDE/ODE 统一框架、与 flow matching 的关系 |
| [[Classifier-FreeGuidance]] | 无分类器引导方法,Guided-TTS 2 等在音频中的应用 |
| [[ScoreMatching]] | Score function 估计、NCSN、SDE 统一、SGMSE 语音增强应用 |
| [[Diffusion-basedVocoder]] | DiffWave/WaveGrad/BDDM/PriorGrad 等 diffusion 声码器家族及实验对比 |
| [[Diffusion-basedTTS]] | Diff-TTS/Grad-TTS/ProDiff/Guided-TTS 2 等 diffusion 声学模型及端到端系统 |

### 更新概念页 (2 个)

| 概念 | 更新内容 |
|------|----------|
| [[NeuralVocoder]] | 追加 diffusion vocoder 家族详述 (BDDM/PriorGrad/InferGrad/SpecGrad) 及 survey 引用 |
| [[ConditionalFlowMatching]] | 补充与 diffusion SDE 的理论连接 (probability flow ODE 桥梁) |

## Survey 结构与关键内容

### Section 2: Background
- **2.1 Audio & Speech**: STFT → mel spectrogram 基础
- **2.2 Diffusion Model**: DDPM 前向/反向过程 (Eq.1-2),score-based model 与 DDPM 的等价性

### Section 3: TTS with Diffusion (核心)
- **3.1**: TTS 从三阶段 → 两阶段的发展 (Fig.1)
- **3.2 Acoustic Model**: Diff-TTS/Grad-TTS (pioneering) → ProDiff/DiffGAN-TTS (acceleration) → Guided-TTS 2 (multi-speaker) → EmoDiff (emotional control)
- **3.3 Vocoder**: WaveGrad/DiffWave (pioneering) → BDDM/InferGrad (efficient) → PriorGrad/DDGM/SpecGrad (statistical)
- **3.4 End-to-End**: WaveGrad 2, CRASH, FastDiff, DAG, Iton

### Section 4: Speech Enhancement with Diffusion
- **4.1 Enhancement by Removing**: SGMSE/SGMSE+ (STFT域), DiffuSE/CDiffuSE (时域), UVD (去混响)
- **4.2 Enhancement by Adding**: NU-Wave/NU-Wave 2 (超分辨率)
- **4.3 Other Tasks**: DiffSep (源分离), DiffSVC (声音转换), CQT-Diff/UNIVERSE (统一框架)

## 关键模型速览

### Diffusion TTS 模型 (Table 1)

```
Acoustic Model:
  开创: Diff-TTS, Grad-TTS
  加速: ProDiff (蒸馏), DiffGAN-TTS (GAN 1步)
  多说话人: Guided-TTS 2 (CFG), Grad-StyleSpeech
  离散空间: Diffsound (VQ-VAE), NoreSpeech
  情感: EmoDiff (classifier guidance)

Vocoder:
  开创: WaveGrad (score-based), DiffWave (DDPM)
  高效: BDDM (7步), InferGrad (联合训练)
  统计改进: PriorGrad (自适应先验), DDGM (Gamma噪声)

End-to-End:
  WaveGrad 2, CRASH, FastDiff, DAG, Iton
```

## 演进脉络

```
DDPM (2020) + NCSN (2019) → SDE 统一 (2020)
    ↓ Audio 应用
DiffWave + WaveGrad (vocoder, 2020)
Diff-TTS + Grad-TTS (acoustic model, 2021)
DiffuSE + SGMSE (speech enhancement, 2021-22)
    ↓ 高效化
ProDiff + DiffGAN-TTS (1步生成, 2022)
BDDM (7步vocoder, 2022)
    ↓ 被 Flow Matching 取代 (2023-)
Voicebox, Matcha-TTS, F5-TTS (CFM-based TTS)
```
