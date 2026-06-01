---
type: dataset
title: "AudioSet"
aliases: []
domain: "general-audio"
tags: [environmental-sound, general-audio, large-scale, multi-label]
key_papers: ["[[论文笔记/DAC|DAC]]"]
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
