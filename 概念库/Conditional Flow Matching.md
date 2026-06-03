---
type: concept
title: "Conditional Flow Matching"
aliases: [CFM, Flow Matching]
category: "generative-model"
tags: [generative-model, flow-based, diffusion-alternative, TTS]
key_papers: ["[[论文笔记/CosyVoice|CosyVoice]]", "[[论文笔记/CosyVoice 2|CosyVoice 2]]", "[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Seed-TTS|Seed-TTS]]", "[[论文笔记/Survey-Audio Diffusion Models|Survey-Audio Diffusion Models]]", "[[论文笔记/GLM-TTS|GLM-TTS]]", "[[论文笔记/FireRedTTS|FireRedTTS]]", "[[论文笔记/Seed-VC|Seed-VC]]", "[[论文笔记/FELLE|FELLE]]", "[[论文笔记/FlowDec|FlowDec]]", "[[论文笔记/CLEAR|CLEAR]]", "[[论文笔记/FlexiVoice|FlexiVoice]]", "[[论文笔记/SiTok|SiTok]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/Multi-Reward GRPO|Multi-Reward GRPO]]", "[[论文笔记/Voxtral TTS|Voxtral TTS]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/Cont-SPT|Cont-SPT]]", "[[论文笔记/E2 TTS|E2 TTS]]", "[[论文笔记/TADA|TADA]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/DMOSpeech 2|DMOSpeech 2]]", "[[论文笔记/VoiceFlow|VoiceFlow]]", "[[论文笔记/PeriodWave|PeriodWave]]", "[[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]", "[[论文笔记/Bridge-TTS|Bridge-TTS]]", "[[论文笔记/EmoSphere++|EmoSphere++]]", "[[论文笔记/GOAT-TTS|GOAT-TTS]]", "[[论文笔记/FlexSpeech|FlexSpeech]]"]
origin_paper: ""
related_concepts: ["[[Finite Scalar Quantization]]", "[[Diffusion Model]]", "[[Score Matching]]"]
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
- CosyVoice (Du et al., 2024): 最早在 LLM-TTS 中采用 OT-CFM 替代 DDPM,使用 cosine scheduler + CFG (β=0.7) + masked mel conditioning 的工程组合
- CosyVoice 3 (2025): 使用 DiT-based CFM,300M 参数
- F5-TTS (2024): Flow matching for fluent and faithful speech
- [[论文笔记/VoiceFlow|VoiceFlow]] (Guo et al., ICASSP 2024): 首次将 rectified flow matching 引入 TTS 声学模型,与 GradTTS 保持相同 U-Net 架构仅替换生成算法,2 步 MOS 3.92 vs GradTTS 2.98 (LJSpeech); 通过 flow rectification 自蒸馏进一步拉直 ODE 轨迹,CMOS +0.78/+1.21
- Matcha-TTS (2024): CFM for fast TTS
- IndexTTS2 (Zhou et al., 2025): 在 S2M 模块中使用 flow matching 从 semantic tokens + speaker embedding 生成 mel spectrogram,并引入 GPT latent enhancement 融合上游 AR 隐状态以提升高情感语音的发音清晰度
- MaskGCT (Wang et al., 2024): 使用 flow matching 训练 total duration predictor (非 phone-level),12 层 Transformer + in-context learning + midpoint ODE solver (4 steps 推理)
- [[论文笔记/Seed-TTS|Seed-TTS]] (ByteDance, 2024): 使用 token diffusion model (diffusion transformer) 将 AR LM 生成的离散 token 转为连续声学表征;Seed-TTS_DiT 变体为完全 diffusion-based NAR TTS,直接从 Gaussian noise 预测 vocoder latent,无需 duration predictor;部署时使用 consistency distillation + modified flow matching 加速

## 相关概念

- [[Diffusion Model]]: CFM 的概念近亲,通过 SDE 而非 ODE; Zhang et al. (2023) audio diffusion survey 详细描述了 SDE → probability flow ODE → flow matching 的理论连接
- DiT (Diffusion Transformer): CosyVoice 3 CFM 的 backbone
- Vocoder: CFM 输出 Mel spectrogram 后仍需 vocoder 合成波形
- [[Finite Scalar Quantization]]: CFM 的输入(speech token 的条件)

- CosyVoice 2 (Du et al., 2024): 提出 chunk-aware causal flow matching,通过四种 attention mask (non-causal/full-causal/chunk-M/chunk-2M) 统一训练实现隐式自蒸馏,首次在 flow matching 框架下实现近无损流式 TTS; CFG β=0.7, NFE=10, cosine scheduler

## 演进

WaveNet (2016, autoregressive vocoder) → Diffusion-based TTS (Grad-TTS, 2021) → Flow Matching (Voicebox, 2023) → CFM + DiT (CosyVoice 3, 2025)

- [[论文笔记/CLEAR|CLEAR]] (Wu et al., 2025): 将 rectified flow 作为轻量 MLP head 直接挂在 AR language model 的每个 token 位置上,以 hidden state 为条件逐 token 生成连续 VAE latent。不同于 CosyVoice 系列将 CFM 作为独立的 second-stage renderer,CLEAR 实现了 **单阶段** AR + flow 联合训练。使用 logit-normal timestep sampling 和 auxiliary velocity direction loss 加速收敛。RTF 0.18, 仅需 78 步 AR decoding (对比 VALL-E 750 步)
- [[论文笔记/PeriodWave|PeriodWave]] (Lee et al., 2024): 首次将 OT-CFM 成功应用于 **波形级 (vocoder)** 生成,而非 mel spectrogram 级。提出 period-aware flow matching estimator,通过 multi-period reshaping (periods=[1,2,3,5,7]) 让 2D UNet 显式捕获不同周期特征。在 pitch/periodicity 指标上大幅超越 GAN vocoder (BigVGAN),且仅用单一 CFM loss 训练 3 天
