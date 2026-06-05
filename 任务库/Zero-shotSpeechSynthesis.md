---
type: task
title: "Zero-shot Speech Synthesis"
aliases: [零样本语音合成, Zero-shot TTS, Zero-shot Voice Cloning]
tags: [TTS, zero-shot, voice-cloning]
key_approaches: ["LLM + discrete tokens", "Diffusion-based", "Coarse-to-fine hybrid"]
key_models: ["[[论文笔记/VALL-E|VALL-E]]", "[[论文笔记/CosyVoice|CosyVoice]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[CosyVoice2]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Seed-TTS|Seed-TTS]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/Chatterbox-Flash|Chatterbox-Flash]]", "[[论文笔记/VoXtream|VoXtream]]", "[[论文笔记/DiSTAR|DiSTAR]]", "[[论文笔记/Spark-TTS|Spark-TTS]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/IDEA-TTS|IDEA-TTS]]", "[[论文笔记/E2TTS|E2 TTS]]", "[[论文笔记/SemaVoice|SemaVoice]]", "[[论文笔记/TADA|TADA]]", "[[论文笔记/OmniVoice|OmniVoice]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/LiveSpeech2|LiveSpeech 2]]", "[[论文笔记/TacoLM|TacoLM]]", "[[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]", "[[论文笔记/StableForm-TTS|StableForm-TTS]]", "[[论文笔记/XTTS|XTTS]]", "[[论文笔记/Muyan-TTS|Muyan-TTS]]", "[[论文笔记/PALLE|PALLE]]", "[[论文笔记/GOAT-TTS|GOAT-TTS]]", "[[论文笔记/Survey-ControllableTTS|Xie et al. Survey 2024]]", "[[论文笔记/TTS-Transducer|TTS-Transducer]]", "[[论文笔记/DMOSpeech|DMOSpeech]]", "[[论文笔记/Koel-TTS|Koel-TTS]]", "[[论文笔记/F5R-TTS|F5R-TTS]]", "[[论文笔记/IndexTTS|IndexTTS]]", "[[论文笔记/OZSpeech|OZSpeech]]", "[[论文笔记/GSA-TTS|GSA-TTS]]", "[[论文笔记/MPE-TTS|MPE-TTS]]", "[[论文笔记/TTS-CtrlNet|TTS-CtrlNet]]", "[[论文笔记/ZipVoice|ZipVoice]]", "[[论文笔记/DS-TTS|DS-TTS]]", "[[论文笔记/SMLLE|SMLLE]]", "[[论文笔记/StreamMel|StreamMel]]", "[[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning]]", "[[论文笔记/Llasa+|Llasa+]]", "[[论文笔记/DAIEN-TTS|DAIEN-TTS]]", "[[论文笔记/LibriQuote|LibriQuote]]", "[[论文笔记/DiFlow-TTS|DiFlow-TTS]]", "[[论文笔记/ZipVoice-Dialog|ZipVoice-Dialog]]", "[[论文笔记/Cross-LingualF5-TTS|Cross-Lingual F5-TTS]]", "[[论文笔记/MELA-TTS|MELA-TTS]]", "[[论文笔记/TaDiCodec|TaDiCodec]]", "[[论文笔记/TTS-1|TTS-1 (Inworld, 2025)]]", "[[论文笔记/DiaMoE-TTS|DiaMoE-TTS]]", "[[论文笔记/CTDiffusion|CTDiffusion]]", "[[论文笔记/BridgeCode|BridgeCode]]", "[[论文笔记/Flamed-TTS|Flamed-TTS]]", "[[论文笔记/DialoSpeech|DialoSpeech]]", "[[论文笔记/MAVE|MAVE]]", "[[论文笔记/UniVoice|UniVoice]]", "[[论文笔记/WeSCon|WeSCon]]", "[[论文笔记/GRPO-TTS|GRPO-TTS]]", "[[论文笔记/ARDM-DPO|ARDM-DPO]]", "[[论文笔记/Semantic-VAE|Semantic-VAE]]"]
benchmarks: ["[[SEED-TTS-Eval]]", "[[CV3-Eval]]"]
metrics: [WER, CER, Speaker Similarity, MOS, DNSMOS]
status: confirmed
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 问题定义

给定任意说话人的短参考语音(prompt,通常 3~10 秒)和目标文本,合成保持该说话人音色、韵律风格的语音,且无需该说话人的专用训练数据。

核心挑战:
- **内容一致性**: 合成语音的文字内容需与目标文本完全一致(CER/WER)
- **说话人相似度**: 合成语音的音色需与参考语音高度相似(Speaker Similarity)
- **自然度**: 韵律、情感、语速等需自然流畅(MOS)
- **泛化能力**: 对未见说话人、罕见文本、噪声环境的鲁棒性

## 主流方法

1. **LLM + 离散 token**: 自回归 LLM 生成 semantic/acoustic token → 声码器/flow model 合成(VALL-E, Seed-TTS, CosyVoice 系列, Spark-TTS)
2. **Diffusion/Flow-based**: 扩散模型直接学习文本-语音对齐(Voicebox, NaturalSpeech 3)
3. **Coarse-to-fine hybrid**: LLM 生成粗粒度 token + 非自回归模型渲染细节(CosyVoice, F5-TTS, MaskGCT)

## 代表模型

- [[论文笔记/CosyVoice3|CosyVoice 3]] (2025): Alibaba, 1.5B, 9 languages, SOTA
- [[论文笔记/Seed-TTS|Seed-TTS]] (2024): ByteDance, 自回归 + token diffusion, CMOS -0.07/-0.08 vs Human (首次达人类水平), 附带 Seed-TTS_DiT (全 diffusion NAR 变体)
- F5-TTS (2024): 非自回归 flow matching
- MaskGCT (2024): Masked generative codec transformer
- [[论文笔记/IndexTTS2|IndexTTS2]] (2025): bilibili, AR-based, 首个同时实现精确 duration control 和自然时长生成的自回归 zero-shot TTS,情感-音色解耦
- [[论文笔记/MaskGCT|MaskGCT]] (2024): CUHK-SZ, 完全非自回归 masked generative TTS,无需 text-speech alignment 和 phone-level duration prediction,100K h 数据,多语言支持

## 评估

### Benchmarks

- [[SEED-TTS-Eval]]: 中/英/hard 三子集
- [[CV3-Eval]]: 9 语种多语言 + 跨语言 + 情感

### Metrics

- **CER/WER**: ASR 转写后与目标文本比较,衡量内容一致性
- **Speaker Similarity (SS)**: 说话人 embedding 余弦相似度(ERes2Net 或 WavLM-based)
- **MOS**: 人工主观评分(1-5 分)
- **DNSMOS**: 自动化音质评估

### 当前 SOTA

| 模型 | 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CosyVoice 3-1.5B_RL | CER test-zh | 0.71% | SEED-TTS-Eval | CosyVoice 3 Table 4 |
| CosyVoice 3-1.5B_RL | WER test-en | 1.45% | SEED-TTS-Eval | CosyVoice 3 Table 4 |
| Seed-TTS | SS test-zh | 0.796 | SEED-TTS-Eval | CosyVoice 3 Table 4 |
| IndexTTS2 | WER test-zh | 1.008% | SEED-TTS-Eval | IndexTTS2 Table 1 |
| IndexTTS2 | SS test-zh | 0.865 | SEED-TTS-Eval | IndexTTS2 Table 1 |
| Qwen3-TTS-12Hz-1.7B | WER test-en | 1.24% | SEED-TTS-Eval | Qwen3-TTS Table 5 |
| Qwen3-TTS-12Hz-1.7B | WER test-zh | 0.77% | SEED-TTS-Eval | Qwen3-TTS Table 5 |
| PilotTTS | SS test-zh | 0.862 | SEED-TTS-Eval | PilotTTS Table 1 |
| PilotTTS | SS test-en | 0.815 | SEED-TTS-Eval | PilotTTS Table 1 |
| PilotTTS | WER test-en | 1.50% | SEED-TTS-Eval | PilotTTS Table 1 |

## 开放问题

- 音色可控性: 能否通过文本指令编辑音色而非仅克隆
- 歌唱合成: 当前零样本系统对歌唱场景支持有限
- 极端鲁棒性: 罕见词、绕口令、domain-specific 术语仍是难点
- 实时流式: 在保持质量前提下实现极低延迟
