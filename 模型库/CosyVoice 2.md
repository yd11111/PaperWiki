---
type: model
title: "CosyVoice 2"
aliases: [CosyVoice2]
org: "Alibaba (Tongyi Lab)"
year: 2024
tags: [TTS, zero-shot, streaming, LLM-based, coarse-to-fine]
key_concepts: ["[[Speech Tokenizer]]", "[[Finite Scalar Quantization]]", "[[Conditional Flow Matching]]"]
tasks: ["[[Zero-shot Speech Synthesis]]", "[[Instructed Speech Generation]]"]
key_papers: ["[[论文笔记/CosyVoice 2|CosyVoice 2]]", "[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/MamTra|MamTra]]", "[[论文笔记/RWKVTTS|RWKVTTS]]", "[[论文笔记/EmoVoice|EmoVoice]]", "[[论文笔记/FPO|FPO]]", "[[论文笔记/LLaMA-Omni 2|LLaMA-Omni 2]]", "[[论文笔记/JoyTTS|JoyTTS]]", "[[论文笔记/OpenS2S|OpenS2S]]", "[[论文笔记/StreamMel|StreamMel]]"]
supersedes: ["[[模型库/CosyVoice|CosyVoice]]"]
superseded_by: ["[[模型库/CosyVoice 3|CosyVoice 3]]"]
status: confirmed
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 概述

CosyVoice 2 是阿里巴巴通义实验室开发的可扩展流式语音合成模型,集成 LLM 和 chunk-aware flow matching 模型,实现低延迟双向流式合成且质量接近人类水平。主要面向中英文场景。

## 核心方法

1. **FSQ-SenseVoice tokenizer**: 将 FSQ 插入 SenseVoice-Large ASR 编码器
2. **Text-based LLM 初始化**: 利用文本 LLM 的语言知识
3. **双向流式方案**: 实现超低延迟且几乎无损的流式合成
4. **统一指令能力建模**: instruction-following 支持

## 性能

| 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- |
| CER (%) test-zh | 1.45 | SEED-TTS-Eval | CosyVoice 3 Table 4 |
| WER (%) test-en | 2.57 | SEED-TTS-Eval | CosyVoice 3 Table 4 |
| MOS 平均 | 4.36 | 主观评估 | CosyVoice 3 Fig.4 |

## 演进线

CosyVoice (2024) → CosyVoice 2 (2024, streaming + instruction) → [[论文笔记/CosyVoice 3|CosyVoice 3]] (2025)

## 关键贡献

- 首个实现几乎无损双向流式零样本 TTS 的系统
- 验证了 text-based LLM 初始化对 TTS LM 的有效性

## 作为 Baseline 被引用

- [[论文笔记/IndexTTS2|IndexTTS2]] (2025): 在基础性能和情感表达两方面均以 CosyVoice2 为 baseline,IndexTTS2 在 SS 和 WER 上全面超越 CosyVoice2;在自然语言情感控制对比中,IndexTTS2 的 EMOS (3.786) 显著优于 CosyVoice2 (3.339)
- [[论文笔记/NVSpeech|NVSpeech]] (2025): 以 CosyVoice2 为 TTS backbone,通过词表扩展+微调添加 18 类副语言发声的显式控制能力;微调后 CER_w/o_para 3.73% (in-domain),listener win rate 75.4% (vs pre-trained)
- [[论文笔记/VoXtream|VoXtream]] (2025): 在 full-stream 场景直接对比,VoXtream FPL 102ms vs CosyVoice2 1643ms (快 16 倍); 在 LibriSpeech long 上 VoXtream WER 3.24% vs CosyVoice2 6.11%,naturalness preference 57% vs 31%; 但 CosyVoice2 的 SPK-SIM 显著更高 (0.685 vs 0.564),归因于 NAR flow-matching decoder 的声学优势
- [[论文笔记/MamTra|MamTra]] (2026): 以 CosyVoice 2 为 teacher backbone,将部分 Transformer 层替换为 Mamba-2 层; MamTra 1:1 在 SEED-TTS-eval test-en 上 WER 2.28% (vs teacher 2.03%),NMOS 3.66 vs 3.68,VRAM 降低 34%; 仅用 LibriTTS 0.5kh (teacher 数据量的 0.3%) 训练
- [[论文笔记/SemaVoice|SemaVoice]] (2026): 连续 AR 路线的 baseline 对比; SemaVoice EN WER 1.71% vs CosyVoice 2 2.57%, ZH CER 1.18% vs 1.45%; 但 CosyVoice 2 在 Hard 子集 CER (6.83% vs 8.09%) 和 SIM (0.724 vs 0.711) 上仍占优 [SemaVoice Table 1]
- [[论文笔记/RWKVTTS|RWKVTTS]] (2025): 将 CosyVoice 2.0 的 Transformer LLM backbone 完整替换为 RWKV-7 (RNN-based); 声称 Production Quality 7.73 接近 GT 7.80,但仅与 FireRedTTS-1S 对比,未报告标准 TTS 指标 (WER/CER/MOS) 且无效率数据 [RWKVTTS Fig 1]
- [[论文笔记/Muyan-TTS|Muyan-TTS]] (2025): 以 CosyVoice2 为主要 baseline 之一; LibriSpeech WER Muyan-TTS 3.44% vs CosyVoice2 2.91%, MOS 4.58 vs 4.81, SIM 0.37 vs 0.70; 推理速度 Muyan-TTS (r=0.33) 显著快于 CosyVoice2 (r=2.19) [Muyan-TTS Table 3/5]
- [[论文笔记/EmoVoice|EmoVoice]] (2025): 以 CosyVoice 语义 token + flow matching + HiFi-GAN 作为音频后端; 在情感控制对比中 EmoVoice(1.5B) 情感 MOS 3.507 vs CosyVoice2 2.138, Emo_Sim 0.9118 vs 0.8647 [EmoVoice Table 2/3]; 中文 Secap 上 EmoVoice-PP WER 7.60 vs CosyVoice2 9.13 [EmoVoice Table 4]
- [[论文笔记/FPO|FPO]] (2025): 以 CosyVoice2 为 backbone 之一,通过 token-level 选择性 DPO 优化; FPO 将 CER 从 1.45 降至 1.32 (-9.0%), WER 从 2.57 降至 2.24 (-12.8%), bad case ratio 从 14% 降至 8%, NMOS 从 3.81 提升至 3.91 [FPO Table I/II]
- [[论文笔记/LLaMA-Omni 2|LLaMA-Omni 2]] (2025): 直接复用 CosyVoice 2 的 FSQ-SenseVoice tokenizer + chunk-aware causal flow matching + HiFi-GAN 作为 streaming speech decoder,嫁接到 Qwen2.5 LLM 构建 modular SpeechLM; 用 200K 合成多轮对话训练即超越 GLM-4-Voice; UTMOS 4.19-4.20 (R=3 W=10),延迟 ~583ms [LLaMA-Omni 2 Table 1]
- [[论文笔记/JoyTTS|JoyTTS]] (2025): 用 CosyVoice2 替换 MiniCPM-o 原有 GPT-SoVITS TTS 模块,通过 MLP 映射 LLM hidden states (3584→768) 桥接; SEED-TTS-zh SS 0.73 vs CosyVoice2 独立 0.748 (-2.4%), WER 5.09 vs 1.45 (3.5x 退化); 开源训练代码 [JoyTTS Table 1]
- [[论文笔记/OpenS2S|OpenS2S]] (2025): 在共情数据构建 pipeline 中使用 CosyVoice2 进行 voice cloning (输入端种子音频克隆) 和 instruction-controlled emotional speech synthesis (输出端情感可控合成); 50k+50k 双语共情样本均通过 CosyVoice2 合成 [OpenS2S §3.2]
- [[论文笔记/StreamMel|StreamMel]] (2025): 作为两阶段流式 baseline 对比; StreamMel 单阶段连续 mel 路线 FPL-A 0.01s vs CosyVoice* 0.22s (快 22 倍); cross-sentence WER-W 2.77 vs CosyVoice* 3.47 [StreamMel Table IV]
