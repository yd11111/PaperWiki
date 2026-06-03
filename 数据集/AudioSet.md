---
type: dataset
title: "AudioSet"
aliases: []
domain: "general-audio"
tags: [environmental-sound, general-audio, large-scale, multi-label]
key_papers: ["[[论文笔记/DAC|DAC]]", "[[论文笔记/UmbraTTS|UmbraTTS]]"]
status: pending-review
lifecycle: active
created: 2026-06-01
updated: 2026-06-01
---

## 概述

AudioSet 是 Google 发布的大规模音频事件数据集,包含超过 200 万条 10 秒 YouTube 音频片段,覆盖 632 种音频事件类别。是环境声/通用音频研究的标准数据集。

## 在 DAC 中的使用

- **训练**: balanced + unbalanced train segments 作为 environmental sound 域的数据源 [§4.1]
- **评估**: evaluation segments 作为环境声测试集 (1000 个 10 秒片段) [§4.1]
- **注意**: AudioSet 数据可能是 band-limited (非 full-band),属于 balanced sampling 中需要特殊处理的类别 [§4.2]

## 在 BEATs 中的使用

- **预训练 + 微调**: BEATs 在 AudioSet full training set (AS-2M, ~5K hours) 上预训练和微调 acoustic tokenizer 及 audio SSL 模型 [BEATs §4.1]
- **评估任务**: AS-2M (19K eval set, mAP) 和 AS-20K (21K balanced train set, mAP) [BEATs §4.1]
- **SOTA 结果**: BEATs_iter3+ 在 AS-2M 上取得 48.6 mAP (single model SOTA),ensemble 10 models 达 50.6 mAP [BEATs Table 1, Table 3]
- 详见 [[论文笔记/BEATs|BEATs]]

## 在 UmbraTTS 中的使用

- **训练**: AudioSet 中自然包含语音+环境音的录音,作为 UmbraTTS 环境感知 TTS 的主要训练数据源 [§4]
- **Self-supervised 数据构建**: 通过 VAD 或 source separation 从 AudioSet 录音中分离语音和环境音,构建 (speech, env, transcript) 三元组 [§3]
