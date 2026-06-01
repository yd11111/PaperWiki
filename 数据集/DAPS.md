---
type: dataset
title: "DAPS"
aliases: [Device and Produced Speech Dataset]
domain: "speech"
tags: [speech, high-quality, studio-recording]
key_papers: ["[[论文笔记/DAC|DAC]]"]
status: pending-review
lifecycle: active
created: 2026-06-01
updated: 2026-06-01
---

## 概述

DAPS (Device and Produced Speech Dataset) 是高质量录音棚语音数据集,包含多位说话人在专业环境下的录音。因其 full-band 高保真特性,常用于 neural audio codec 的训练和评估。

## 在 DAC 中的使用

- **训练**: 作为 speech 域的高质量全带宽数据源 [§4.1]
- **评估**: F10, M10 两位说话人的数据被 held out 作为测试集 [§4.1]
- **Balanced sampling**: 属于 full-band 数据类别,确保模型学习全频带重建 [§4.2]
