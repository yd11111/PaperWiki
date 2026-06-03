---
title: "NLLB"
aliases: [No Language Left Behind, NLLB-200]
authors: [NLLB Team, Marta R. Costa-jussa, James Cross, et al.]
year: 2022
venue: arXiv
arxiv_id: "2207.04672"
source: "https://arxiv.org/abs/2207.04672"
tags: [machine-translation, multilingual, low-resource, MoE, FLORES-200, text-only]
level: enhanced-card
status: draft
created: 2026-06-03
updated: 2026-06-03
concepts: []
models: []
datasets: []
---

# No Language Left Behind: Scaling Human-Centered Machine Translation

## 为什么是 enhanced-card

NLLB 是纯文本机器翻译工作,不涉及语音/TTS 技术。与本知识库（以 TTS/语音合成为核心）的关联仅限于：
1. Seamless 系列以 NLLB 为文本翻译基座 [§3.2.2 of Seamless paper]
2. SeamlessM4T 的 T2TT 模型在 NLLB 数据上预训练 [论文原文]
3. FLORES-200 benchmark 被后续语音翻译工作广泛引用

因此按 AGENTS.md 规则，不触发 KB 检索，生成 enhanced-card 而非 deep note。

## 摘要 / Summary

NLLB (No Language Left Behind) 是 Meta AI 的大规模多语言机器翻译项目，目标是为 200+ 低资源语言提供高质量翻译。核心模型 NLLB-200 基于 54.5B Sparsely Gated Mixture-of-Experts (MoE) 架构，支持 202 种语言、40,000+ 翻译方向，在 FLORES-101 上相比前 SOTA 提升 44% BLEU [§Abstract]。

## 核心方法 / Core Method

**四大支柱** [Fig 1]:
1. **以人为本**: 访谈 44 位低资源语言使用者，理解翻译需求与痛点 [§2.1]
2. **自动数据创建**: 语言识别 + 单语数据收集 + LASER3 双文本挖掘 [§5]
3. **建模**: Sparsely Gated MoE + 课程学习 + 自监督预训练 + 反向翻译 + NLLB-Seed 引导 [§6]
4. **评估**: FLORES-200 (204 语言 many-to-many benchmark) + Toxicity-200 安全评估 [§7]

**模型家族** [§1]:
- NLLB-200: 54.5B MoE 主模型
- 3.3B / 1.3B Dense Transformer
- 1.3B / 600M 蒸馏模型
- 全部开源: github.com/facebookresearch/fairseq/tree/nllb

## 关键结果 / Key Results

| 指标 | 数值 | 说明 |
|------|------|------|
| BLEU 提升 | +44% | 相比前 SOTA (FLORES-101) [§Abstract] |
| 语言覆盖 | 202 种 | 含 100+ 新增低资源语言 [Table 1] |
| 翻译方向 | 40,602 | many-to-many [§Abstract] |
| 低资源语言占比 | 75% | 150/200 语言为 low-resource [§3] |

## 与语音领域的关联 / Speech Relevance

- **SeamlessM4T 基座**: SeamlessM4T v2 的文本翻译模块 (SeamlessM4T-NLLB) 直接使用 NLLB 预训练的 Dense Transformer encoder-decoder [Seamless §3.2.2, Fig 2]
- **FLORES-200**: 被 Seamless 等语音翻译工作用作评估基准
- **语言 ID 模型**: NLLB 的 LID 模型被 Seamless 数据处理流程复用 [Seamless §3.1.3]
- **Toxicity-200**: 被 Seamless 的 Responsible AI 框架引用

## 数据集贡献

| 数据集 | 内容 | 规模 |
|--------|------|------|
| FLORES-200 | 204 语言评估集 | 3001 句 x 204 语言 [§4.1] |
| NLLB-Seed | 39 种低资源语言种子数据 | 人工翻译 [§4.2] |
| NLLB-MD | 6 语言多领域数据 | 泛化评估 [§4.3] |
| Toxicity-200 | 200 语言毒性词表 | 安全评估 [§7.3] |

## 局限 / Limitations

- 纯文本翻译,不处理语音模态
- MoE 模型推理成本高,需蒸馏才能实用部署
- 低资源语言数据质量仍不均衡
