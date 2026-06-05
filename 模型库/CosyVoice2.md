---
type: model
title: "CosyVoice 2"
aliases: [CosyVoice2]
org: "Alibaba (Tongyi Lab)"
year: 2024
tags: [TTS, zero-shot, streaming, LLM-based, coarse-to-fine]
key_concepts: ["[[SpeechTokenizer]]", "[[FiniteScalarQuantization]]", "[[ConditionalFlowMatching]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[InstructedSpeechGeneration]]"]
key_papers: ["[[论文笔记/CosyVoice2|CosyVoice 2]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[论文笔记/DiffRO|DiffRO]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/MamTra|MamTra]]", "[[论文笔记/RWKVTTS|RWKVTTS]]", "[[论文笔记/EmoVoice|EmoVoice]]", "[[论文笔记/FPO|FPO]]", "[[论文笔记/LLaMA-Omni2|LLaMA-Omni 2]]", "[[论文笔记/JoyTTS|JoyTTS]]", "[[论文笔记/OpenS2S|OpenS2S]]", "[[论文笔记/StreamMel|StreamMel]]", "[[论文笔记/NonverbalTTS|NonverbalTTS]]", "[[论文笔记/EmoSteer-TTS|EmoSteer-TTS]]", "[[论文笔记/MELA-TTS|MELA-TTS]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/DiaMoE-TTS|DiaMoE-TTS]]", "[[论文笔记/TKTO|TKTO]]", "[[论文笔记/HD-PPT|HD-PPT]]", "[[论文笔记/WeSCon|WeSCon]]", "[[论文笔记/BatonVoice|BatonVoice]]", "[[论文笔记/Audiobook-CC|Audiobook-CC]]", "[[论文笔记/GRPO-TTS|GRPO-TTS]]"]
supersedes: ["[[模型库/CosyVoice|CosyVoice]]"]
superseded_by: ["[[模型库/CosyVoice3|CosyVoice 3]]"]
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

CosyVoice (2024) → CosyVoice 2 (2024, streaming + instruction) → [[论文笔记/CosyVoice3|CosyVoice 3]] (2025)

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
- [[论文笔记/LLaMA-Omni2|LLaMA-Omni 2]] (2025): 直接复用 CosyVoice 2 的 FSQ-SenseVoice tokenizer + chunk-aware causal flow matching + HiFi-GAN 作为 streaming speech decoder,嫁接到 Qwen2.5 LLM 构建 modular SpeechLM; 用 200K 合成多轮对话训练即超越 GLM-4-Voice; UTMOS 4.19-4.20 (R=3 W=10),延迟 ~583ms [LLaMA-Omni 2 Table 1]
- [[论文笔记/JoyTTS|JoyTTS]] (2025): 用 CosyVoice2 替换 MiniCPM-o 原有 GPT-SoVITS TTS 模块,通过 MLP 映射 LLM hidden states (3584→768) 桥接; SEED-TTS-zh SS 0.73 vs CosyVoice2 独立 0.748 (-2.4%), WER 5.09 vs 1.45 (3.5x 退化); 开源训练代码 [JoyTTS Table 1]
- [[论文笔记/OpenS2S|OpenS2S]] (2025): 在共情数据构建 pipeline 中使用 CosyVoice2 进行 voice cloning (输入端种子音频克隆) 和 instruction-controlled emotional speech synthesis (输出端情感可控合成); 50k+50k 双语共情样本均通过 CosyVoice2 合成 [OpenS2S §3.2]
- [[论文笔记/StreamMel|StreamMel]] (2025): 作为两阶段流式 baseline 对比; StreamMel 单阶段连续 mel 路线 FPL-A 0.01s vs CosyVoice* 0.22s (快 22 倍); cross-sentence WER-W 2.77 vs CosyVoice* 3.47 [StreamMel Table IV]
- [[论文笔记/NonverbalTTS|NonverbalTTS]] (2025): 以 CosyVoice2 为 NV 生成 baseline; 用 17h 开源 NVTTS 数据微调 CosyVoice-300M 后,人类偏好测试 33.4% vs CosyVoice2 35.4% (p>0.05 无显著差异); NV Jaccard 0.80 vs 0.78; NVTTS SIM-o 0.89 显著优于 CosyVoice2 0.75,但 DNSMOS 3.82 略低于 3.93; laughter Jaccard NVTTS 0.25 vs CosyVoice2 0.34,归因于 CosyVoice2 的细粒度 laughter tokenization [NonverbalTTS Table 8, Fig 1]
- [[论文笔记/EmoSteer-TTS|EmoSteer-TTS]] (2025): 以 CosyVoice2 为三个测试模型之一,通过 training-free activation steering 实现情感控制; CosyVoice2+EmoSteer WER 2.83, S-SIM 0.65, E-SIM 0.26, N-MOS 3.65; CosyVoice2 (56 DiT layers, 10 CFM steps) 每隔 5 层 steering (层 1,6,11,...) [EmoSteer-TTS Table 1, Table 2]
- [[论文笔记/DiffRO|DiffRO]] (Gao et al., 2025): 以 CosyVoice 2.0-0.5B 为 baseline 验证 DiffRO (Differentiable Reward Optimization); DiffRO-ASR WER-zh 0.78% vs CosyVoice2 1.56% (Seed-TTS-eval), 跨语言迁移 ja 6.36% vs 9.13%, ko 5.41% vs 7.43%; DiffRO-MTR 实现零样本情感控制,accuracy 全面超越 CosyVoice2 baseline [DiffRO Table 2, Table 3]
- [[论文笔记/TKTO|TKTO]] (Kotoge & Sasaki, 2025): 以 CosyVoice 2 (0.5B) 为 base model,在 20K 小时日语数据微调后应用 TKTO (token-level KTO); 日语歧义发音 Acc 0.668→0.958, CER 0.138→0.066 (-52%), NMOS 4.09→4.21; 超越 gpt-4o-mini-tts 和 gemini-2.5-pro-preview-tts [TKTO Table 1/2]
- [[论文笔记/HD-PPT|HD-PPT]] (Nie et al., 2026): 以 CosyVoice 2 为主要 baseline 和基础设施(复用其 speech tokenizer 和 vocoder); HD-PPT 在 TextrolSpeech+EmoVoice-DB 联合测试上 MOS-N 4.108 vs CosyVoice2 3.920, MOS-S 4.167 vs 3.885, EMO-SIM 0.753 vs 0.714, WER 5.18% vs 5.71%; 通过 hierarchical preference token 分解实现更精细的指令遵循 [HD-PPT Table 1]
- [[论文笔记/BatonVoice|BatonVoice]] (Wang et al., 2025): 直接复用 CosyVoice2 的 speech decoder (flow matching + HiFi-GAN vocoder) 作为冻结的 "orchestra"; 以 Qwen3-1.7B/Qwen2.5-0.5B 替换 CosyVoice2 的 LLM backbone,额外以 verbalized vocal features 作为条件; Emotion Acc. 57.6% 超越 CosyVoice (43.8%) 和 CosyVoice2 (37.8%),但 WER 2.5 略逊于 CosyVoice2 (2.1) [BatonVoice Table 1]
- [[论文笔记/WeSCon|WeSCon]] (Wang et al., NeurIPS 2025): 以 CosyVoice2 为 backbone,通过两阶段 self-training 实现 word-level 情感和语速控制; Emo2v. 0.882 vs CosyVoice2 0.866, DNSV 4.361 vs 7.894 (EN), EMOS 3.70 vs 3.61; 零样本 TTS 性能几乎无损 CER 1.47 vs 1.45, S-SIM 0.744 vs 0.748 [WeSCon Table 1/2/3]
- [[论文笔记/GRPO-TTS|GRPO-TTS]] (Liu et al., USTC/iFLYTEK, 2025): 以 CosyVoice2 为两类 LLM-TTS 之一验证 GRPO (CER+NLL composite reward from Whisper); CER zh 1.41→1.07 (-24%), WER en 2.46→2.30 (-6.5%), MOS zh 4.42→4.58 (p<0.05), MOS en 4.22→4.43 (p<0.05); 仅 4000 句 GRPO 训练数据; 同时在 Llasa-1B 上验证,CER 改善更大但 MOS 无显著提升 [GRPO-TTS Table 1, Table 2]
- [[论文笔记/RRPO|RRPO]] (Wang et al., Tongyi Lab, 2026): 以 CosyVoice2 为 baseline 验证 Robust Reward Policy Optimization; 在 DiffRO 框架上通过三层混合正则化 (LS+EAM+Adv) 强化 SER reward model,解决 reward hacking; RRPO E-MOS 3.78 vs CosyVoice2 3.27, N-MOS 3.81 vs 3.65; DiffRO baseline N-MOS (3.61) 低于 SFT (3.72) 体现 reward hacking,RRPO 同时提升两个维度 [RRPO Table 1]
- [[论文笔记/InstructAudio|InstructAudio]] (Qiang et al., Kuaishou/Tianjin Univ., 2025): 以 CosyVoice2 为 instruction-based TTS 的主要 baseline; InstructAudio 纯文本控制在 Gender Acc 100% vs CosyVoice2 不支持, Emotion Acc 83.33% vs 58.33%, Style Acc 86.67% vs 65.00%, Speaker SIM 0.76 vs 0.68, Emotion SIM 0.71 vs 0.53; 但 CosyVoice2 使用额外参考音频,QMOS 3.90 vs 3.73, NMOS 3.65 vs 3.46; WER EN CosyVoice2 2.57% vs InstructAudio 1.52%, ZH 1.45% vs 1.35% [InstructAudio Table 1, 2]
- [[论文笔记/TED-TTS|TED-TTS]] (Liang et al., NUS, 2026): 以 CosyVoice2 为 intra-utterance emotion control 的对比模型之一; TED-TTS training-free 框架在 IndexTTS2 上实现 segment-level 多情感控制; EN speech prompt SMOS TED-TTS 4.00 vs CosyVoice2 3.33, NMOS 4.20 vs 2.87, NISQA 4.706 vs 4.535; CosyVoice2 WER 1.411 优于 TED-TTS 2.519; ZH speech prompt SMOS 4.13 vs 3.04, NMOS 4.07 vs 2.71 [TED-TTS Table 1]
- [[论文笔记/VoiceSculptor|VoiceSculptor]] (Hu et al., ASLP@NPU, 2026): 以 CosyVoice2-0.5B-LLM 作为 voice clone 后端,接收 LLaSA-3B voice design 模块生成的 prompt waveform 进行 timbre transfer; InstructTTSEval-Zh VD&VC AVG 67.3% 与 VD-only 67.6% 接近,证明 VD→VC 风格传递几乎无损 [VoiceSculptor Table 1]
- [[论文笔记/TaskVectorTTS|TaskVectorTTS]] (Feng et al., SJTU, 2025): 以 CosyVoice2 为方言合成和情感方言合成的 baseline; 方言合成 E-Vector MOS 3.18 vs CosyVoice2 2.62, 情感方言合成 HE-Vector MOS 2.83 vs CosyVoice2 1.87; 作者指出 E-Vector 方法应用于 CosyVoice (v1) 时质量下降,因 LLM+flow matching 组件间协调被破坏 [TaskVectorTTS Table 2, Table 3, §5]
- [[论文笔记/OV-InstructTTS|OV-InstructTTS]] (Ren et al., CASIA/Tsinghua, 2026): 以 CosyVoice2 (No-Instruct) 为 baseline 之一,同时用 CosyVoice2 的 tokenizer 和 flow matching 为其他 baseline (GPT4o, Higgs Audio V2) 做音色转换; OV-InstructTTS-TEP Gemini Score 70.42 vs CosyVoice2 66.99, MOS 4.28 vs 3.84, ICMOS 3.91 vs 2.94; CosyVoice2 CER 3.09% 优于 OV-InstructTTS-TEP 3.61% [OV-InstructTTS Table 2]
- [[论文笔记/CoCoEmo|CoCoEmo]] (Wang et al., Univ. Melbourne, 2026): 以 CosyVoice2 为主要实验 backbone,通过 cross-conditioning diagnostic 证明情感韵律主要编码在 SLM (Qwen2-based decoder) 而非 flow-matching 模块; 在 SLM layers 10-17 的 attn_output 发现最高线性可分性; Mixed-emotion CoCoEmo (alpha=5.0) E-SIM 0.795 / TEP 0.315 全面超越 Instruction2 (0.762/0.169); High-mismatch E-SIM 0.862 / TEP 0.504 (alpha=6.0); S-SIM 0.870 几乎无损 (vs no-steer 0.871); 可叠加在 Instruction1/2 之上进一步提升 [CoCoEmo Table 2, Table 3]
- [[论文笔记/ZeSTA|ZeSTA]] (Choi et al., Maum AI, 2026): 以 CosyVoice 2 为两个 ZS-TTS 数据源之一,为 VITS 微调生成合成训练数据; CosyVoice 2 生成的合成语音 SECS 0.794 (LibriTTS) / 0.788 (YoBind); ZeSTA DC+OS 使用 CV2 合成数据后 VITS SECS 0.815 (vs naive mixing 0.789), CER 4.943→4.560; ABX preference 61.8% (p<0.05) [ZeSTA Table 2, Table 3, Table 4]
- [[论文笔记/NV-Bench|NV-Bench]] (Ni et al., 2026): 以 CosyVoice 2 的两个 NV 微调变体 (SMIIP-NV-CV2, Emilia-NV-CV2) 作为 benchmark baseline; SMIIP-NV-CV2 PCER 75.64% (ZH single), Emilia-NV-CV2 PCER 40.00%; 两者均显著弱于 NV-CV3 (27.69%),但 Emilia-NV-CV2 IMOS 3.89 接近 NV-CV3 (3.95); NV-Bench 验证了 CosyVoice 架构在 NV 微调后的可控性提升空间 [NV-Bench Table 4, Table 5]
- [[论文笔记/DSFlow|DSFlow]] (Lin et al., StepFun, 2026): 以 CosyVoice2 为跨架构蒸馏验证目标之一 (U-Net without adaLN-Zero); 仅用 dual supervision + weak CFG (不含 step-aware token,因无 adaLN) 即获 1-step MOS-N 4.23 vs CosyVoice2 teacher 10-step 4.41, SIM-o 0.63 vs 0.64; 证明 DSFlow 的核心蒸馏组件在非 DiT 架构上通用 [DSFlow Table 1, Table 7]
- [[论文笔记/VoXtream2|VoXtream2]] (Torgashov et al., KTH, 2026): 在 zero-shot TTS 和 SRC 两个维度对比 CosyVoice2; zero-shot: VoXtream2 SEED-en WER 1.32% vs CosyVoice2 2.27%, SPK-SIM 0.656 vs 0.658 (基本持平), MUSHRA 68.8 vs 66.2; full-stream: FPL 74ms vs CosyVoice2:TRT 837ms (快 11 倍); SRC: CosyVoice2 instructed generation 仅在 3-4 SPS 范围有效,VoXtream2 通过 distribution matching 实现 2-5 SPS 连续控制 + 动态 mid-utterance 变速 [VoXtream2 Table 2, 4, Fig 6]
- [[论文笔记/WAND|WAND]] (Lee et al., KAIST/SKKU, 2026): 在 CosyVoice 2-0.5B 上应用 windowed attention + KD 实现常数推理开销; KV cache 5.25 MB vs baseline 10.48 MB (-49.9%), GFLOPs 7.44 vs 11.55 (speedup 1.55x); WER en 1.72% vs 1.94% (改善), CER zh 1.53% vs 1.59% (改善); 仅用 100h LibriTTS 1 epoch 微调,不改架构; 注意力分析显示 CosyVoice 2 的 58.5% 注意力在 conditioning prefix [WAND Table 1, Table 2, Table 3]
- [[论文笔记/CosyEdit2|CosyEdit2]] (Chen et al., Nankai, 2026): 以 CosyVoice2 为 backbone 构建端到端 speech editing 系统,通过两阶段 post-training (SFT→editing-oriented GRPO) 解锁编辑能力; GRPO 仅更新 LLM 而冻结 Flow+BigVGAN; 编辑性能: WER 1.43% (substitution, Ming-Freeform), SS 0.89-0.93, MAE_DNSMOS 0.107-0.137 (best acoustic consistency); 关键发现: editing-oriented GRPO 反哺 zero-shot TTS,SEED-TTS CER zh 1.36→1.16, WER en 3.10→1.95; CV3-Eval hard-zh CER 15.70→8.06 [CosyEdit2 Table 1, Table 2, Table 9]
