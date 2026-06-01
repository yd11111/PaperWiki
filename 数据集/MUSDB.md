---
type: dataset
title: "MUSDB"
aliases: [MUSDB18, MUSDB-HQ]
domain: "music"
tags: [music, source-separation, multi-track]
key_papers: ["[[论文笔记/DAC|DAC]]"]
status: pending-review
lifecycle: active
created: 2026-06-01
updated: 2026-06-01
---

## 概述

MUSDB (Music Source Database) 是音乐源分离领域的标准数据集,包含全带宽多轨音乐录音。MUSDB18 包含 150 首完整歌曲 (约 10 小时),分为训练集和测试集。

## 在 DAC 中的使用

- **训练**: 作为 music 域的数据源 [§4.1]
- **评估**: test split 被用作音乐类测试集 (1000 个 10 秒片段) [§4.1]
