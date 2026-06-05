---
type: dataset
title: "SEED-TTS-Eval"
aliases: [SEED-TTS Eval, SEED TTS Eval]
domain: "TTS evaluation"
scale: "3 subsets (test-zh, test-en, test-hard)"
tags: [benchmark, TTS, zero-shot, evaluation]
used_by: ["[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[CosyVoice2]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Seed-TTS|Seed-TTS]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/Multi-RewardGRPO|Multi-Reward GRPO]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/MamTra|MamTra]]", "[[论文笔记/SemaVoice|SemaVoice]]", "[[论文笔记/TADA|TADA]]", "[[论文笔记/LLaDA-TTS|LLaDA-TTS]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/DMOSpeech2|DMOSpeech 2]]", "[[论文笔记/GOAT-TTS|GOAT-TTS]]", "[[论文笔记/F5R-TTS|F5R-TTS]]", "[[论文笔记/JoyTTS|JoyTTS]]", "[[论文笔记/Llasa+|Llasa+]]", "[[论文笔记/Cross-LingualF5-TTS|Cross-Lingual F5-TTS]]", "[[论文笔记/MELA-TTS|MELA-TTS]]", "[[论文笔记/TTS-1|TTS-1 (Inworld, 2025)]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/Vox-Evaluator|Vox-Evaluator]]", "[[论文笔记/Ming-UniAudio|Ming-UniAudio]]", "[[论文笔记/ARDM-DPO|ARDM-DPO]]", "[[论文笔记/GRPO-TTS|GRPO-TTS]]", "[[论文笔记/DisCo-Speech|DisCo-Speech]]", "[[论文笔记/ARCHI-TTS|ARCHI-TTS]]", "[[论文笔记/MOSS-TTS|MOSS-TTS]]", "[[论文笔记/IterateDifferentiate|I2D]]", "[[论文笔记/KineticOptimalTTS|GibbsTTS]]", "[[论文笔记/RobustSpeechFlow|RobustSpeechFlow]]", "[[论文笔记/Raon-Speech|Raon-Speech]]"]
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

- [[论文笔记/CosyVoice3|CosyVoice 3]]: CER 0.71% (zh), WER 1.45% (en), CER 5.09% (hard)
- [[CosyVoice2]]: CER 1.45% (zh), WER 2.57% (en), CER 6.83% (hard)
- Seed-TTS: CER 1.12% (zh), WER 2.25% (en), CER 7.59% (hard)
- F5-TTS: CER 1.56% (zh), WER 1.83% (en), CER 8.67% (hard)

## 注意事项

- Speaker Similarity 有两种评估方式: ERes2Net-based 和 WavLM-based,需注明使用哪种
- 随着模型进步,各系统在此 benchmark 上的分数趋于接近,区分度下降,这促使了 CV3-Eval 等新 benchmark 的提出

## 最新结果

- [[论文笔记/IndexTTS2|IndexTTS2]] (2025): WER 1.008% (test-zh), WER 1.521% (test-en), SS 0.865 (test-zh), SS 0.860 (test-en); 同时在 duration control 设定下 token number error rate <0.02%
- [[论文笔记/MaskGCT|MaskGCT]] (2024): SIM-O 0.728 (test-en), WER 2.466 (test-en), SIM-O 0.777 (test-zh), WER 2.183 (test-zh); 非自回归 masked generative 方法
- [[论文笔记/MamTra|MamTra]] (2026): WER 2.28% (test-en, 1:1 配置), SSIM 0.72 (test-en), UTMOS 4.16 (test-en); Mamba-Transformer 混合架构,VRAM 降低 34% vs CosyVoice 2
- [[论文笔记/LLaDA-TTS|LLaDA-TTS]] (2026): CER 0.98% (test-zh), WER 1.96% (test-en), CER 7.04% (test-hard), SS 74.6% (test-zh); masked discrete diffusion 替代 AR decoder,基于 CosyVoice 3-0.5B backbone,64 步推理实现 2x LLM-stage speedup
- [[论文笔记/PilotTTS|PilotTTS]] (2026): CER 0.87% (test-zh), WER 1.50% (test-en), SIM 0.862 (test-zh), SIM 0.815 (test-en); Q-Former + CAMPPlus 双路径 conditioning,仅用 200K h 数据,SIM 刷新 SEED-TTS-Eval 最高记录
- [[论文笔记/DMOSpeech2|DMOSpeech 2]] (2025/AAAI 2026): WER 1.752% (test-en), CER 1.527% (test-zh), SIM 0.698 (test-en), SIM 0.760 (test-zh), RTF 0.032; 0.3B params, GRPO 优化 duration predictor, 4-step DMD-distilled flow matching
- [[论文笔记/F5R-TTS|F5R-TTS]] (Tencent, 2025): WER 1.48% (test-cn general), WER 10.63% (test-cn hard), SIM 0.730 (test-cn general), SIM 0.711 (test-cn hard); 首次在 NAR flow-matching TTS 上集成 GRPO,通过 output probabilization 使 RL 兼容 CFM 架构
- [[论文笔记/MOSS-TTS|MOSS-TTS]] (OpenMOSS, 2026): CER 1.44% (test-zh), WER 1.93% (test-en), SIM 79.62% (test-zh), SIM 73.28% (test-en); MOSS-TTS-Local-Transformer 1.7B Continuation 模式,纯 AR 离散 token 路线,ZH SIM 为开源模型最高
- [[论文笔记/X-Voice|X-Voice]] (Xu et al., 2026): X-Voice_s2 WER 1.28% (test-zh), WER 1.30% (test-en), SIM 0.76 (test-zh), SIM 0.65 (test-en), RTF 0.073; 0.4B NAR flow-matching,30 语言 transcript-free 零样本克隆
- [[论文笔记/KineticOptimalTTS|GibbsTTS]] (Yang et al., 2026): UTMOS 3.651 (test-en), WER 1.777% (test-en), SIM 0.743 (test-en), UTMOS 2.712 (test-zh), CER 1.327% (test-zh), SIM 0.790 (test-zh); 0.4B MI-DFM NAR, kinetic-optimal scheduler + moment correction, SIM 在 test-en/test-zh 均为评估系统中最高
- [[论文笔记/RobustSpeechFlow|RobustSpeechFlow]] (Yang et al., 2026): WER 1.38% (test-en), SIM 0.60 (test-en); 0.06B compact flow-matching TTS (SupertonicTTS backbone),仅通过 contrastive FM + failure-mode hard negatives 训练策略改进达到 benchmark 最低 WER,但 SIM 偏低
