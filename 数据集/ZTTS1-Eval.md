---
type: dataset
title: "ZTTS1-Eval"
aliases: [ZTTS1-Eval benchmark, Zyphra TTS Eval]
domain: "TTS evaluation (multilingual, prosody)"
scale: "9 languages x 500 utterances (Clean) + 17 languages x 1618 total utterances (ITW)"
tags: [benchmark, TTS, multilingual, zero-shot, evaluation, in-the-wild, prosody]
used_by: ["[[论文笔记/ZONOS2|ZONOS2]]"]
metrics_reported_on: [WER, Speaker Similarity, UTMOS, TTSDS2-Prosody, DS-WED]
url: "https://github.com/Zyphra/ZTTS1-Eval"
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-25
updated: 2026-06-25
---

## 概述

ZTTS1-Eval 是 Zyphra 随 [[论文笔记/ZONOS2|ZONOS2]] 论文发布的多语言零样本 TTS 评估 benchmark,设计用于弥补 [[数据集/SEED-TTS-Eval|SEED-TTS-Eval]] 和 [[数据集/CV3-Eval|CV3-Eval]] 的三个主要局限:
1. 语言覆盖有限(SEED-TTS-Eval 仅中英)
2. 评分模型过时(Whisper-Large / Paraformer / WavLM)
3. 缺少韵律多样性评估维度

### 评估子集

- **Clean set**: 9 种语言(en, zh, de, es, fr, it, ja, ko, ru),每语种 500 utterances,来自 FLEURS-R,约 13 小时,read-aloud speech;含 hard 子集(英文和中文复杂句)
- **In-the-Wild (ITW) set**: 17 种语言(在 Clean 的 9 语种基础上增加 ar, hi, id, pl, pt, th, tl, tr),共 1618 utterances(每语种 84-120 条不等),来自 VoxBlink2,约 3 小时,自发对话语音

### 评分模型栈

| 维度 | 评分模型 | 替代对象 |
|------|---------|---------|
| Content (WER) | Qwen3-ASR | Whisper-Large / Paraformer |
| Speaker Similarity | ReDimNet | WavLM / ERes2Net |
| Quality | MSR-UTMOS | DNSMOS / 无 |
| Prosody Distribution | TTSDS2 | 无 |
| Generation Diversity | DS-WED | 无 |

## 用途

面向多语言、自发语音场景的全维度 TTS 评估,特别适用于评估韵律多样性和生成一致性。相比 SEED-TTS-Eval 和 CV3-Eval 的关键差异化优势在于 prosody/diversity 评估维度(TTSDS2 + DS-WED)和更新的评分模型栈。

## 使用此数据集的模型

- [[论文笔记/ZONOS2|ZONOS2]]: ZTTS1-Clean en WER 2.76% / Spk.sim 78.6; ITW en WER 4.70% / Spk.sim 67.0; Quality Mode 显著改善非英语 WER (zh: 15.62→6.73) [Table IV, V]

## 注意事项

- ZONOS2 作者声明模型未在 ZTTS1-Eval 数据上训练,但无法确认其他对比模型是否训练过该数据
- Benchmark 由 ZONOS2 论文作者提出,ZONOS2 在其上的优势(尤其是 prosody metrics)需要独立验证
- Apache 2.0 许可,已在 GitHub 开源
