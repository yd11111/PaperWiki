---
type: model
title: "VITS"
aliases: [Variational Inference with adversarial learning for end-to-end Text-to-Speech, VITS TTS]
org: "Kakao Enterprise / KAIST"
year: 2021
tags: [TTS, end-to-end, VAE, normalizing-flow, GAN, parallel-synthesis]
key_concepts: ["[[VariationalAutoencoderforTTS]]", "[[Non-autoregressiveTTS]]", "[[DurationPredictor]]", "[[NeuralVocoder]]", "[[Speech-TextAlignment]]"]
tasks: []
key_papers: ["[[论文笔记/VITS|VITS]]", "[[论文笔记/YourTTS|YourTTS]]", "[[论文笔记/IDEA-TTS|IDEA-TTS]]", "[[论文笔记/Llama-VITS|Llama-VITS]]", "[[论文笔记/TITW|TITW]]", "[[论文笔记/UMETTS|UMETTS]]", "[[论文笔记/Muyan-TTS|Muyan-TTS]]", "[[论文笔记/MathReader|MathReader]]", "[[论文笔记/SafeSpeech|SafeSpeech]]", "[[论文笔记/FaceSpeak|FaceSpeak]]", "[[论文笔记/FMSD-TTS|FMSD-TTS]]", "[[论文笔记/FNH-TTS|FNH-TTS]]", "[[论文笔记/TMD-TTS|TMD-TTS]]", "[[论文笔记/ParaStyleTTS|ParaStyleTTS]]"]
supersedes: []
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

VITS (Kim et al., ICML 2021) 是首个将 conditional VAE + normalizing flow + adversarial training 统一为端到端并行 TTS 系统的工作。单模型直接从 phoneme 序列生成高质量波形,无需外部 vocoder,质量超越所有当时的两阶段系统并接近真实语音 [§1, §4.1]。

## 核心方法

- **Conditional VAE**: posterior encoder (WaveNet blocks, linear spectrogram input) + prior encoder (Transformer + normalizing flow) + HiFi-GAN decoder [§2.1, §2.5]
- **Normalizing Flow 增强 prior**: affine coupling layers 将 factorized Gaussian 变换为复杂分布,消融显示去掉 flow MOS 下降 1.52 [§2.1.3, Table 2]
- **Monotonic Alignment Search**: 复用 Glow-TTS 的 MAS 在 ELBO 框架下自动估计 text-speech 对齐 [§2.2.1]
- **Stochastic Duration Predictor**: flow-based 概率 duration 建模,生成多样化的韵律 [§2.2.2]
- **Adversarial Training**: HiFi-GAN MPD + feature matching loss 提升波形质量 [§2.3]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| MOS | 4.43 (+-0.06) | LJ Speech | [Table 1] |
| MOS (multi-spk) | 4.38 (+-0.06) | VCTK | [Table 3] |
| 合成速度 | 67.12x 实时 | LJ Speech | [Table 4] |
| CMOS vs GT | -0.106 | LJ Speech | [Table 5] |

## 演进线

Glow-TTS (Kim et al., 2020; flow-based NAR) → **VITS** (2021; VAE+Flow+GAN E2E) → VITS 2 (Kim et al., 2023; 多说话人扩展) → NaturalSpeech (Tan et al., 2022; memory-based VAE 扩展) → VALL-E / LLM-based TTS (2023; 范式转换)

## 关键贡献

1. 首次证明端到端单模型 TTS 可超越两阶段系统 [Table 1]
2. Normalizing flow 增强 VAE prior 的有效性 (贡献最大的单一因素) [Table 2]
3. Flow-based stochastic duration predictor 建模韵律多样性 [§2.2.2]
4. Linear spectrogram 作为 posterior 高分辨率输入的设计 [§2.1.3]

## 相关工作

- [[论文笔记/ZeSTA|ZeSTA]] (Choi et al., 2026): 以 VITS 为 target model,验证 ZS-TTS 合成数据增强 + domain-conditioned training 的低资源个性化策略; 复用 VITS speaker embedding matrix (缩减至 64-dim) 实现 domain conditioning,DC+OS 使 SECS 从 0.765 (naive mixing) 恢复到 0.815 (接近 Real 100% 的 0.832) [ZeSTA Table 3]
- [[论文笔记/SelfTTS|SelfTTS]] (Ueda et al., 2026): 在 VITS 上扩展双 Reference Encoder (speaker + emotion) + cosine-based GRL 解耦 + MPCL 对比聚类 + Self-Augmentation (利用 normalizing flow 可逆性做 VC 生成合成情感数据); 在 ESD 上 eMOS 2.853 超越 E3-VITS/VECL baseline [SelfTTS Table 1]
- [[论文笔记/PITS|PITS]] (Lee et al., 2023): 在 VITS 上增加 Yingram encoder (第二 posterior encoder) 编码 pitch 信息,用 scope-shift 实现无 F0 的 pitch 可控 TTS; Adversarial pitch-shifted training 确保 shifted speech 质量; VCTK 上 MOS 4.01 (vs VITS 3.89, GT 4.04), CER 3.27% (vs 4.35%), EER 0.926% (vs 1.08%) [PITS Table 1]
- [[论文笔记/XPhoneBERT|XPhoneBERT]] (Nguyen et al., INTERSPEECH 2023): 用多语言预训练 phoneme BERT (330M sentences, 94 languages) 替换 VITS 的 Transformer encoder; EN MOS 4.00→4.14 (+0.14, LJSpeech), VN MOS 3.74→3.89 (+0.15); 低资源 VN (5% data) MOS 1.59→3.35 (+1.76),证明预训练 phoneme encoder 可大幅提升 VITS 性能尤其在低资源场景 [XPhoneBERT Table 2, 3]
- [[论文笔记/ShanghainTTS|ShanghainTTS]] (Chen, 2023): 用 VITS 做上海话 (Wu Chinese) 低资源 TTS,利用 stochastic duration predictor 建模声调变调 (tone sandhi) 的 one-to-many mapping; 通过 jieba 分词标注左主导变调域 (LD domain) 边界作为韵律代理; MOS 4.14 与 Apple VoiceOver 4.19 无显著差异 (p=0.64),但在多音节 LD 域处理上显著优于 VoiceOver (4.53 vs 3.08, p<<0.001) [ShanghainTTS Table 1, 3]
- [[论文笔记/MunTTS|MunTTS]] (Gumma et al., 2024): 用 VITS 为极低资源印度语言 Mundari (~1M 母语者) 从头训练多说话人 TTS; 27.5h 社区参与式数据收集 (Hindi 翻译 + 众包录音); VITS-44K MOS 3.69 全面优于 VITS-22K (3.04)、XTTS v2 pretrained (2.20) 和 MMS (0.79); XTTS v2 微调灾难性失败 (MOS 0.05),佐证单语言 E2E 模型在极低资源场景优于多语言微调 [MunTTS Table 2, 3]
- [[论文笔记/USAT|USAT]] (Wang et al., 2024): 在 VITS 基础上构建统一 zero-shot + few-shot speaker adaptation 框架; 继承 VITS 的 posterior encoder/stochastic duration predictor,扩展为 memory-augmented VAE + timbre converter (normalizing flow + 两个解耦判别器); 冻结预训练参数插入 flow adapter + phoneme adapter 实现 0.64M 参数 (~1%) 的 few-shot 适应; LibriTTS unseen SMOS 4.06, SMCS 0.780, SVR 88.7% 超越 YourTTS [USAT Table I]
