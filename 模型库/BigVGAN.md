---
type: model
title: "BigVGAN"
aliases: [BigVGANv2, BigVGAN v2]
org: "NVIDIA"
year: 2023
tags: [vocoder, neural-vocoder, GAN-based, waveform-generation]
key_concepts: []
tasks: ["[[Zero-shot Speech Synthesis]]"]
key_papers: ["[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/DAC|DAC]]", "[[论文笔记/E2 TTS|E2 TTS]]", "[[论文笔记/FMSD-TTS|FMSD-TTS]]", "[[论文笔记/DeepDubbing|DeepDubbing]]", "[[论文笔记/Audiobook-CC|Audiobook-CC]]"]
supersedes: [HiFi-GAN]
superseded_by: []
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

BigVGAN 是 NVIDIA 提出的通用神经声码器,通过大规模训练实现对未见说话人、语言、录音条件的强泛化能力。BigVGANv2 (2023) 进一步提升了音质和效率。

## 核心方法

1. **大规模训练**: 在大量多样化语音数据上训练,获得 universal vocoder 能力
2. **Snake activation**: 使用周期性激活函数建模语音波形的周期结构
3. **Anti-aliased representation**: 减少频谱混叠
4. **多分辨率判别器**: 保证不同尺度的音质

## 在 TTS pipeline 中的角色

作为级联 TTS 系统的最后一环,将 mel spectrogram 转换为 audio waveform。在 IndexTTS2 中,BigVGANv2 接收 S2M 模块输出的 mel spectrogram 并生成最终语音波形。

## 性能

以高保真度和泛化能力著称,被多个 zero-shot TTS 系统采用(IndexTTS, IndexTTS2 等)。

## 关键贡献

- 首个真正意义上的 "universal" neural vocoder
- 证明 vocoder 的泛化能力可通过数据规模和架构设计同时提升

## 使用此模型的系统

- [[论文笔记/IndexTTS2|IndexTTS2]]: 使用 BigVGANv2 作为 vocoder
- IndexTTS (Deng et al., 2025): 同样使用 BigVGAN
- [[论文笔记/DAC|DAC]] (Kumar et al., NeurIPS 2023): 沿用 BigVGAN 的 Snake activation 和训练 recipe,将其扩展到 universal audio codec 领域
