---
type: concept
title: "Next-Token Diffusion"
aliases: [Per-Token Diffusion, Token-Level Diffusion Head, 逐token扩散, Next-Token Diffusion Head]
category: "technique"
tags: [diffusion, autoregressive, continuous-representation, language-model, TTS, multimodal]
key_papers: ["[[论文笔记/LatentLM|LatentLM]]", "[[论文笔记/CLEAR|CLEAR]]", "[[论文笔记/VibeVoice|VibeVoice]]", "[[论文笔记/SemaVoice|SemaVoice]]", "[[论文笔记/TADA|TADA]]", "[[论文笔记/Dragon-FM|Dragon-FM]]", "[[论文笔记/MELA-TTS|MELA-TTS]]"]
origin_paper: "Sun et al., Multimodal Latent Language Modeling with Next-Token Diffusion, 2024 (arXiv:2412.08635)"
related_concepts: ["[[Diffusion Model]]", "[[Conditional Flow Matching]]", "[[LLM-based TTS]]", "[[Classifier-Free Guidance]]", "[[Variational Autoencoder for TTS]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-03
updated: 2026-06-03
---

## 定义

Next-Token Diffusion 是一种将 diffusion/flow matching 过程嵌入自回归语言模型中的生成技术。其核心思想是: **在 causal Transformer 的每个连续 token 位置上,挂一个轻量 diffusion (或 rectified flow) head,以该位置的 hidden state 为条件,独立生成该位置的连续 latent vector**。

与传统 sequence-level diffusion (如 DiT, Stable Diffusion) 的根本区别:

| 维度 | Sequence-Level Diffusion | Next-Token Diffusion |
|------|--------------------------|----------------------|
| 去噪范围 | 整个序列同时去噪 | 每个 token 位置独立去噪 |
| 注意力 | 需要双向 (全序列交互) | 兼容 causal (单向) |
| Backbone | 独立 diffusion 模型 (DiT/U-Net) | 共享 LM Transformer backbone |
| 推理 | LM backbone 需多次 forward | LM backbone 仅一次 forward,仅 head 迭代 |
| 流式能力 | 需等全序列 | 支持 (逐 token 生成) |

**数学形式** (以 DDPM 变体为例, LatentLM [§2.1]):
- Transformer 输出 hidden state h_i
- Diffusion head epsilon_theta(x_i^t, t, h_i) 预测 noise
- 损失: L_Diff = E[||epsilon - epsilon_theta(x_i^t, t, h_i)||^2]
- 推理: 从纯 Gaussian noise 出发,以 h_i 为条件迭代去噪

**Head 架构**: 轻量残差网络 (3-6 层),包含 AdaLN-Zero 条件化 timestep t 和 h_i [LatentLM §2.1]。CLEAR 使用 ResBlock1D [CLEAR §3.2]。

## 在 TTS 中的应用

Next-Token Diffusion 在 TTS 中实现了**单阶段连续值 AR 生成**,绕过了两阶段 (AR discrete tokens → flow matching mel) 的复杂度:

- **LatentLM** (Sun et al., 2024): 奠基性工作。sigma-VAE + per-token DDPM head + causal Transformer。TTS 以 15 fps (1600x 压缩) 超越 VALL-E 2,解码步数仅为其 1/10 [LatentLM Table 4]
- **CLEAR** (Wu et al., 2025): 用 MLP-based rectified flow head (非 DDPM) + enhanced wav-VAE (2048x)。核心优势: MLP 不需全序列注意力,支持流式 (96ms FFL)。RTF 0.18,78 步 AR decoding [CLEAR Table 2, 4]
- **VibeVoice** (Peng et al., 2025): 直接复用 LatentLM 的 DDPM head (4 layers) + Qwen2.5 LLM backbone。3200x causal tokenizer + dual tokenizer (acoustic + semantic)。实现 90 分钟多说话人对话 [VibeVoice §2.2]
- **SemaVoice** (Wang et al., 2026): 在 sigma-VAE 训练中引入 WavLM guided alignment (frame-wise cosine + pair-wise 自相似矩阵匹配),提升连续表示的语义一致性。Qwen2.5-1.5B + patch-wise LocDiT (L=2, 含 previous-patch conditioning)。150K h 双语训练, Seed-TTS-Eval EN WER 1.71% [SemaVoice Table 1]

**共同模式**: VAE encoder → 连续 latent → causal Transformer → per-token diffusion/flow head → VAE decoder → waveform

## 与 sequence-level diffusion 的互补

Next-Token Diffusion 不是要取代 sequence-level diffusion,而是用于**与 causal LM 兼容**的场景:
- Sequence-level 更适合需要全局协调的任务 (如图像生成的全局构图)
- Next-Token 更适合需要流式、因果、长序列的任务 (如 TTS、对话)
- 两者可用同一 Transformer backbone,通过 softmax head (离散) 和 diffusion head (连续) 共存

## 关键论文

- LatentLM (Sun et al., 2024, arXiv:2412.08635): 提出 next-token diffusion 概念和 sigma-VAE [origin paper]
- CLEAR (Wu et al., 2025, arXiv:2508.19098): MLP rectified flow 变体,强调流式低延迟
- VibeVoice (Peng et al., 2025, arXiv:2508.19205): 工业级落地,90 分钟多说话人对话
- GIVT (Tschannen et al., 2023): 使用 Gaussian mixture 直接预测 VAE latent (无 diffusion),LatentLM 的对比基线

## 相关概念

- [[Diffusion Model]]: Next-Token Diffusion 的理论基础,但从 sequence-level 改为 per-token
- [[Conditional Flow Matching]]: CLEAR 使用 rectified flow 作为 per-token head 的具体实现
- [[LLM-based TTS]]: Next-Token Diffusion 是 LLM-based TTS 从离散 token 扩展到连续表示的技术桥梁
- [[Classifier-Free Guidance]]: 在 next-token diffusion 中同样适用,通过 text embedding dropout 实现
- [[Variational Autoencoder for TTS]]: sigma-VAE 作为连续数据的 tokenizer 与 next-token diffusion 配合使用

## 演进

MELLE (2024, continuous mel AR, Gaussian assumption, no diffusion) → LatentLM (2024, per-token DDPM head, sigma-VAE, multimodal) → CLEAR (2025, per-token rectified flow head, enhanced VAE, streaming TTS) → VibeVoice (2025, industrial-scale, long-form multi-speaker, Qwen2.5 backbone) → SemaVoice (2026, SFM-guided VAE alignment, patch-wise LocDiT)
