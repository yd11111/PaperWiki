---
type: paper-note
title: "Audio-Language Models for Audio-Centric Tasks: A Systematic Survey"
authors: [Yi Su, Jisheng Bai, Qisheng Xu, Kele Xu, Yong Dou]
year: 2025
venue: "arXiv:2501.15177"
arxiv_id: "2501.15177"
tier: card
tags: [survey, audio-language, CLAP, ALM, LALM, pretraining, captioning, QA, cold-start]
concepts: ["[[Audio-Language Pretraining]]", "[[Audio Understanding]]", "[[Speech Language Model]]"]
created: 2026-06-02
updated: 2026-06-02
---

## 概要

首篇全面系统综述 Audio-Language Models (ALMs) 的 survey。ALMs 在配对 audio-text 数据上训练,通过自然语言监督学习音频表征,覆盖语音、环境声、音乐三大域。论文从 (1) ALM 基础架构与训练目标, (2) 表征预训练, (3) 下游迁移, (4) 数据集与评估 四个维度全面梳理该领域。

## 核心贡献

1. **ALM 架构分类**: Two Towers / Two Heads / One Head / Cooperated Systems 四类
2. **预训练目标体系**: Contrastive (infoNCE) + Generative (masked reconstruction, LM) + Discriminative (matching, classification)
3. **CLAP 模型族谱**: MS-CLAP → LAION-CLAP → MS-CLAP V2 → WavCaps → COMPA → MGA-CLAP → T-CLAP
4. **LALM 体系**: Pengi, LTU, SALMONN, Audio Flamingo (1/2/3), GAMA, Moshi, SHANKS, STITCH
5. **数据集全景**: 16+ audio-text paired datasets (57K-6M pairs), 10+ AQA datasets
6. **核心局限**: Hallucination, adversarial vulnerability, linguistic bias, training cost, data contamination

## 知识提取成果

本文用于概念库冷启动 T2 补充。

### 新建概念页 (1 个)

| 概念 | 核心内容 |
|------|----------|
| [[Audio-Language Pretraining]] | CLAP 范式、ALM 架构、训练目标、模型族谱、下游任务、LALM |

### 追加更新 (1 个)

| 页面 | 更新内容 |
|------|----------|
| [[Audio Understanding]] | 补充 ALM 视角: audio captioning, AQA, ATR; LALM 模型; 扩展 benchmark |

## 关键观点

- ALM 与 SpeechLM 是互补路线: SpeechLM 侧重 speech-in-speech-out 端到端, ALM 侧重 audio-text cross-modal 理解 [§I]
- CLAP 的 batch size bias 问题尚未充分解决: 有限负样本导致 systematic gradient bias [§III-B]
- Hallucination 在结构化 QA 中比 free-form captioning 更严重 [§VIII-A]
- 跨数据集音频重叠 (WavCaps/Clotho) 导致评估不可靠,需 de-duplication pipeline [§VII-B]
- LALM 中 instruction tuning 和 in-context learning 是两条互补的 multi-task 路线 [§V-B]
- Agent Systems (AudioGPT, SpeechAgents, MusicAgents) 利用 LLM 协调多模型是第四种 ALM 架构范式 [§III-A]
