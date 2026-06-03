---
type: model
title: "WavLM"
aliases: [WavLM Base, WavLM Base+, WavLM Large]
org: "Microsoft"
year: 2022
tags: [self-supervised-learning, speech-representation, masked-prediction, speech-denoising, full-stack, speaker-verification, speech-separation, diarization, ASR]
key_concepts: ["[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Speaker Embedding]]", "[[Speech Factorization]]", "[[Self-Supervised Speech Representation]]"]
tasks: []
key_papers: ["[[论文笔记/WavLM|WavLM]]", "[[论文笔记/Vec-Tok Speech|Vec-Tok Speech]]", "[[论文笔记/LM-SPT|LM-SPT]]"]
supersedes: ["[[模型库/HuBERT|HuBERT]]"]
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-03
updated: 2026-06-03
---

## 概述

WavLM (Chen et al., IEEE JSTSP 2022) 是首个面向 full-stack 语音处理的大规模自监督预训练模型。通过三项关键改进 — masked speech denoising、gated relative position bias、多样化 94k 小时预训练数据 — 在 SUPERB 19 个子任务和 speaker verification/diarization/separation 等非 ASR 任务上全面达到 SOTA。

## 核心方法

- **架构基础**: 沿用 HuBERT (CNN Encoder + Transformer),三种配置: Base (94.7M), Base+ (94.7M, 94k data), Large (316.6M) [§V-A]
- **Masked Speech Denoising**: 预训练时在输入中混合模拟噪声/重叠语音 (Algorithm 1),目标仍是干净语音的 HuBERT k-means 伪标签;模型被迫学习去噪+说话人分离+说话人识别 [§IV-B]
- **Gated Relative Position Bias**: 替代卷积位置编码,使位置偏差根据当前语音内容自适应调整 (content-aware PE) [§IV-A]
- **多样化数据**: 94k hrs (LibriLight 60k + GigaSpeech 10k + VoxPopuli 24k),覆盖 audiobooks + podcasts + EP recordings [§IV-C]
- **Training stabilization**: scaled attention (c=32) 解决 fp16 overflow [§IV-D]

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| SUPERB overall score (Large) | 74.6 | SUPERB 15 tasks | [Table I] |
| EER Vox1-O/E/H (Large*) | 0.383/0.480/0.986 | VoxCeleb1 | [Table II] |
| DER all speakers (Large) | 10.35 | CALLHOME | [Table III] |
| Separation avg WER (Large) | 6.0 | LibriCSS | [Table IV] |
| ASR WER test-clean/other (960h, Transf. LM) | 1.8/3.2 | LibriSpeech | [Table VI] |

## 演进线

wav2vec 2.0 (2020; contrastive only) → HuBERT (2021; masked prediction + k-means) → w2v-BERT (2021; contrastive + masked prediction) → **WavLM** (2022; masked speech denoising + gated PE + diverse data, full-stack SOTA) → WavLM 2.0 (if any) / UniSpeech-SAT (2021; speaker-aware)

## 关键贡献

1. 首次在单一 SSL 模型中统一 ASR 和非 ASR (speaker verification, diarization, separation) 任务 [Table I]
2. Masked speech denoising: 预训练时加噪 → 模型学会 speaker discrimination + denoising [§IV-B, Table I ablation]
3. 层级信息分离: bottom layers → speaker info, top layers → content info [Fig 2] — 为 layer-wise weighted sum 方案提供理论基础
4. Gated relative position bias: content-aware positional encoding [§IV-A]
5. 多样化数据 (94k hrs) 显著提升 out-of-domain 泛化 [Table V, Table VI]
