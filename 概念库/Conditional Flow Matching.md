---
type: concept
title: "Conditional Flow Matching"
aliases: [CFM, Flow Matching]
category: "generative-model"
tags: [generative-model, flow-based, diffusion-alternative, TTS]
key_papers: ["[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/IndexTTS2|IndexTTS2]]"]
related_concepts: ["[[Finite Scalar Quantization]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Conditional Flow Matching (CFM) 是一种基于连续正规化流 (Continuous Normalizing Flow, CNF) 的生成模型训练方法。它通过学习一个向量场将简单先验分布(如高斯)转换为目标数据分布,训练时仅需回归条件概率路径上的向量场,比传统 diffusion 模型推理步数更少、效率更高。

与 diffusion model 的关键区别: CFM 直接学习确定性 ODE 路径(flow),而非随机 SDE;可使用更少步数完成生成,且支持 rectified flow 等加速变体。

## 在 TTS 中的应用

在 coarse-to-fine TTS 系统(如 CosyVoice 系列、Voicebox、F5-TTS)中,CFM 用于将离散 speech token 序列转换为连续 Mel spectrogram。它作为 "fine stage" 渲染器,负责恢复 speech token 中被丢弃的声学细节(音色、韵律微观结构)。

CosyVoice 3 中 CFM 采用 DiT (Diffusion Transformer) 架构作为 backbone,参数从 100M 扩至 300M,去掉了 CosyVoice 2 的 text encoder 和 length regularization module。

## 关键论文

- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
- CosyVoice 3 (2025): 使用 DiT-based CFM,300M 参数
- F5-TTS (2024): Flow matching for fluent and faithful speech
- Matcha-TTS (2024): CFM for fast TTS
- IndexTTS2 (Zhou et al., 2025): 在 S2M 模块中使用 flow matching 从 semantic tokens + speaker embedding 生成 mel spectrogram,并引入 GPT latent enhancement 融合上游 AR 隐状态以提升高情感语音的发音清晰度

## 相关概念

- Diffusion Model: CFM 的概念近亲,通过 SDE 而非 ODE
- DiT (Diffusion Transformer): CosyVoice 3 CFM 的 backbone
- Vocoder: CFM 输出 Mel spectrogram 后仍需 vocoder 合成波形
- [[Finite Scalar Quantization]]: CFM 的输入(speech token 的条件)

## 演进

WaveNet (2016, autoregressive vocoder) → Diffusion-based TTS (Grad-TTS, 2021) → Flow Matching (Voicebox, 2023) → CFM + DiT (CosyVoice 3, 2025)
