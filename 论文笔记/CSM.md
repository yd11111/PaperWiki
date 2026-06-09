---
type: paper
tier: card
title: "CSM: Conversational Speech Model"
arxiv_id: ""
source: "https://github.com/SesameAILabs/csm"
authors: [SesameAILabs]
year: 2025
venue: "GitHub (no paper)"
tags: [speech-LM, conversational-TTS, multi-turn, open-source, codec-LM]
concepts: ["[[SpeechLanguageModel]]", "[[CodecLanguageModel]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 0
status: draft
created: 2026-06-09
updated: 2026-06-09
---

## 速查卡片

- **一句话**: Sesame 开源的对话语音生成模型,基于 Llama backbone + 多 codebook 语音生成,支持多轮对话上下文
- **路线**: 文本/音频 token → Llama Transformer → 多 codebook AR 生成 → Mimi codec 解码
- **GitHub Stars**: 14.7K (截至 2026-06)
- **限制**: 无 arXiv 论文,技术细节仅通过 GitHub README 和 HuggingFace 模型卡发布;训练数据/推理延迟未公开;英语为主

## 技术要点

[基于公开信息, 非论文精读]

- 基于 Mimi codec (借鉴 Kyutai) 做语音 tokenization
- 使用 Llama 架构 Transformer backbone 进行语音 token 预测
- 多 codebook 自回归生成: semantic tokens → acoustic tokens 逐层生成
- 多轮对话上下文建模: 将对话历史 (多个说话人的 audio tokens) 作为条件输入
- 以 "Voice Presence" (对话韵律真实感) 为核心差异化卖点

## 备注

本条目为 card 级笔记。Sesame 目前没有公开发表的 arXiv 论文,CSM 的技术细节主要通过 GitHub README 和 HuggingFace 模型卡发布。如未来发布技术报告,应升级为 deep 级笔记。
