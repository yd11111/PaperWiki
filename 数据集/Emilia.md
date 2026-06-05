---
type: dataset
title: "Emilia"
aliases: [Emilia Dataset]
domain: "Large-scale speech generation training"
scale: "101K+ hours, multilingual"
tags: [training-data, large-scale, multilingual, TTS]
used_by: ["[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Seed-VC|Seed-VC]]", "[[论文笔记/NVSpeech|NVSpeech]]", "[[论文笔记/FlexiCodec|FlexiCodec]]", "[[论文笔记/FlexiVoice|FlexiVoice]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/DiSTAR|DiSTAR]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/LLaDA-TTS|LLaDA-TTS]]", "[[论文笔记/SemaVoice|SemaVoice]]", "[[论文笔记/CapSpeech|CapSpeech]]", "[[论文笔记/OpenS2S|OpenS2S]]", "[[论文笔记/Cross-LingualF5-TTS|Cross-Lingual F5-TTS]]", "[[论文笔记/MBCodec|MBCodec]]", "[[论文笔记/DiaMoE-TTS|DiaMoE-TTS]]", "[[论文笔记/Vox-Evaluator|Vox-Evaluator]]", "[[论文笔记/Ming-UniAudio|Ming-UniAudio]]", "[[论文笔记/DisCo-Speech|DisCo-Speech]]", "[[论文笔记/Training-freeSpeakerUnlearning|TruS (Lee et al., 2026)]]", "[[论文笔记/ARCHI-TTS|ARCHI-TTS]]", "[[论文笔记/KineticOptimalTTS|GibbsTTS]]", "[[论文笔记/UNISON|UNISON]]"]
metrics_reported_on: []
url: ""
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

Emilia 是首个大规模、多语言、多样化的开源 in-the-wild 语音生成数据集,由 He et al. (2024) 发布。配套开源预处理 pipeline Emilia-Pipe,可将原始 in-the-wild 音频转换为高质量带标注训练数据。详见 [[论文笔记/Emilia|Emilia 论文笔记]]。

## 规模与特点

- **总量**: 101,654 小时 (初始版本)
- **语种分布**: 英语 46.8K h (46.77%), 中文 49.9K h (49.83%), 德语 1.6K h, 法语 1.8K h, 日语 1.7K h, 韩语 0.2K h
- **采样率**: 24 kHz, mono, 16-bit
- **数据来源**: 多样化视频平台和播客 — 访谈、辩论、体育解说、有声书等 in-the-wild 录音
- **质量**: DNSMOS P.835 OVRL 3.26 ± 0.14,在 9 个对比数据集中排第三 (仅次于 MLS 3.33, Libri-Light 3.25)
- **多样性**: acoustic (WavLM features) 和 semantic (Sentence-BERT) 特征空间上均显著优于 audiobook 数据集 (MLS)

## Emilia-Pipe 预处理 Pipeline

首个全开源的 in-the-wild 语音数据预处理 pipeline,6 步流程:
1. **Standardization**: WAV, mono, 24kHz, -20 dBFS 归一化
2. **Source Separation**: UVR-MDX-Net Inst 5 去除背景音乐/噪声
3. **Speaker Diarization**: pyannote 3.1 说话人分割
4. **Fine-grained Segmentation**: Silero-VAD 切分为 3-30 秒段落
5. **ASR**: WhisperX (Whisper-Medium + faster-whisper + CTranslate2)
6. **Filtering**: 语种过滤 + DNSMOS OVRL ≥ 3.0 + duration outlier 过滤

处理效率: ~2.5 小时数据/分钟 (8×RTX 4090)

## 使用此数据集的模型

- [[论文笔记/IndexTTS2|IndexTTS2]]: 使用 Emilia 作为主要训练数据来源,55K 小时训练数据中大部分来自 Emilia (30K 中文 + 25K 英文)
- [[论文笔记/MaskGCT|MaskGCT]]: 使用 Emilia 100K 小时 (50K 英文 + 50K 中文) 训练全部模型组件
- [[论文笔记/Seed-VC|Seed-VC]]: 使用 Emilia 训练 voice conversion 模型
- [[论文笔记/NVSpeech|NVSpeech]]: 使用 Emilia 子集作为副语言感知 ASR 自动标注的数据来源之一
- [[论文笔记/X-Voice|X-Voice]]: 使用 Emilia 中英数据作为 420K 小时 30 语言训练语料的重要组成部分; F5-TTS-v1-Base checkpoint (在 Emilia 上预训练) 作为 DiT 初始化

## 来源

He et al., "Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation", IEEE SLT 2024. arXiv: 2407.05361.
