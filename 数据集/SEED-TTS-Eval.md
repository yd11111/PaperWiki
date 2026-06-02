---
type: dataset
title: "SEED-TTS-Eval"
aliases: [SEED-TTS Eval, SEED TTS Eval]
domain: "TTS evaluation"
scale: "3 subsets (test-zh, test-en, test-hard)"
tags: [benchmark, TTS, zero-shot, evaluation]
used_by: ["[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[CosyVoice 2]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Seed-TTS|Seed-TTS]]"]
metrics_reported_on: [CER, WER, Speaker Similarity]
url: ""
status: confirmed
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

SEED-TTS-Eval 是由 ByteDance 随 [[论文笔记/Seed-TTS|Seed-TTS]] 论文发布的广泛使用的零样本 TTS 评估集,包含三个子集:
- **test-zh**: 中文普通话测试集
- **test-en**: 英文测试集
- **test-hard**: 中文高难度测试集(包含罕见词、绕口令、领域术语等)

评估维度: 内容一致性(CER/WER)和说话人相似度(Speaker Similarity)。

## 用途

作为零样本 TTS 模型的标准化评估 benchmark,被 Seed-TTS、CosyVoice 系列、F5-TTS、MaskGCT、FireRedTTS 等众多模型采用。原始论文使用 Whisper-large-v3 (EN) 和 Paraformer-zh (ZH) 计算 WER,使用 WavLM-large fine-tuned speaker verification 计算 SIM [Seed-TTS §3.1]。objective set 包含 1000 条 Common Voice + 2000 条 DiDiSpeech 样本。

## 使用此数据集的模型

- [[论文笔记/CosyVoice 3|CosyVoice 3]]: CER 0.71% (zh), WER 1.45% (en), CER 5.09% (hard)
- [[CosyVoice 2]]: CER 1.45% (zh), WER 2.57% (en), CER 6.83% (hard)
- Seed-TTS: CER 1.12% (zh), WER 2.25% (en), CER 7.59% (hard)
- F5-TTS: CER 1.56% (zh), WER 1.83% (en), CER 8.67% (hard)

## 注意事项

- Speaker Similarity 有两种评估方式: ERes2Net-based 和 WavLM-based,需注明使用哪种
- 随着模型进步,各系统在此 benchmark 上的分数趋于接近,区分度下降,这促使了 CV3-Eval 等新 benchmark 的提出

## 最新结果

- [[论文笔记/IndexTTS2|IndexTTS2]] (2025): WER 1.008% (test-zh), WER 1.521% (test-en), SS 0.865 (test-zh), SS 0.860 (test-en); 同时在 duration control 设定下 token number error rate <0.02%
- [[论文笔记/MaskGCT|MaskGCT]] (2024): SIM-O 0.728 (test-en), WER 2.466 (test-en), SIM-O 0.777 (test-zh), WER 2.183 (test-zh); 非自回归 masked generative 方法
