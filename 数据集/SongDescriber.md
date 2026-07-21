---
type: dataset
title: "Song Describer Dataset"
aliases: [SongDescriber, Song Describer, SDD]
domain: "music"
tags: [music, audio-captioning, music-and-language, evaluation]
key_papers: ["[[论文笔记/STAR-VAE|STAR-VAE]]"]
status: pending-review
lifecycle: active
created: 2026-07-21
updated: 2026-07-21
---

## 概述

Song Describer Dataset (Manco et al., 2023) 是音乐-语言评测语料,包含音乐音频及其自然语言描述,用于 music-and-language 任务评估。在音频重建/生成研究中常作为 music 域测试集。

## 在 STAR-VAE 中的使用

- **评测**: 作为 music 域重建评测集,报告 STFT-D/MSD/SI-SDR/FAD/LC [STAR-VAE Table 1];架构消融也在此集上进行 [STAR-VAE Table 3]
- STAR-VAE 在此集重建 FAD 0.25(vs Stable Audio Open 0.69,同 21.5Hz)[STAR-VAE Table 1]
