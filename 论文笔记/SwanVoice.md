---
type: paper
tier: enhanced-card
title: "SwanVoice: Expressive Long-Form Zero-Shot Speech Synthesis for Both Monologue and Dialogue"
arxiv_id: "2605.30993"
source: "https://arxiv.org/abs/2605.30993"
authors: [Ruiqi Li, Yu Zhang, Changhao Pan, Ke Lei, Xiang Yin]
year: 2025
venue: ""
tags: [zero-shot-tts, long-form, dialogue, flow-matching, expressive-tts, diffusion-post-training]
concepts: []
models: []
tasks: []
datasets: []
status: draft
created: 2026-06-01
updated: 2026-06-01
---

## 一句话定位

零样本长篇幅语音合成系统,同时支持 1-4 人独白和对话场景,结合 25Hz VAE + Flow Matching DiT + DiffusionNFT post-training。

## 方法概述

三阶段架构:
1. **25Hz VAE** 将语音压缩为连续 latent(不走离散 token)
2. **Flow Matching DiT** 以 raw text(含 pause-aware symbols + pinyin)为条件,生成 VAE latent;支持 speaker-turn conditioning 实现多人切换
3. **DiffusionNFT post-training** 用 phone-level 和 speaker-similarity rewards 做 diffusion 后训练优化

训练策略:monologue → mixed → real dialogue 渐进;数据来自 SwanData-Speech(wild audio + forced alignment)。

## 关键结果

- 在 SwanBench-Speech 上 richness 和 hierarchy scores 超过所有开源 baseline(monologue + dialogue)
- 主要局限:content accuracy(CER/WER)仍是短板
- 支持 1-4 speaker zero-shot

## 与已知方法简单对比

- vs CosyVoice 3:CosyVoice 走离散 token + CFM,SwanVoice 走连续 VAE + Flow DiT;CosyVoice 侧重短句精度(CER 0.71%),SwanVoice 侧重长篇幅表达力
- vs 拼接方案(逐句合成+拼接):SwanVoice 原生多人对话建模,避免拼接带来的声学不连续

## 为何值得关注

1. 连续 VAE(非离散 token)路线在 TTS 的新探索
2. DiffusionNFT post-training 是 DiffRO 之外的另一种 reward-based 后训练思路
3. 多人对话 TTS 的原生建模(不靠拼接)
