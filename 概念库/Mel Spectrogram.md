---
type: concept
title: "Mel Spectrogram"
aliases: [梅尔频谱图, Mel-spectrogram, MelS, 梅尔谱]
category: "representation"
tags: [acoustic-feature, signal-processing, TTS, vocoder]
key_papers: ["[[论文笔记/VITS|VITS]]", "[[论文笔记/HuBERT|HuBERT]]", "[[论文笔记/Whisper|Whisper]]", "[[论文笔记/MELLE|MELLE]]", "[[论文笔记/FELLE|FELLE]]", "[[论文笔记/E2 TTS|E2 TTS]]", "[[论文笔记/STTATTS|STTATTS]]", "[[论文笔记/Bridge-TTS|Bridge-TTS]]", "[[论文笔记/Very Attentive Tacotron|Very Attentive Tacotron (Battenberg et al., 2025)]]", "[[论文笔记/SafeSpeech|SafeSpeech]]", "[[论文笔记/DiVISe|DiVISe (Liu et al., 2025)]]", "[[论文笔记/Spotlight-TTS|Spotlight-TTS]]", "[[论文笔记/SMLLE|SMLLE]]", "[[论文笔记/StreamMel|StreamMel]]", "[[论文笔记/Shallow Flow Matching|Shallow Flow Matching]]", "[[论文笔记/MELA-TTS|MELA-TTS]]", "[[论文笔记/MiSTR|MiSTR]]", "[[论文笔记/EmoSSLSphere|EmoSSLSphere]]", "[[论文笔记/TMD-TTS|TMD-TTS]]"]
origin_paper: ""
related_concepts: ["[[Speech Tokenizer]]", "[[Neural Vocoder]]", "[[Text-to-Speech Pipeline]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Mel Spectrogram 是语音信号经短时傅里叶变换(STFT)后,将频率轴映射到 Mel 尺度(模拟人耳对频率的非线性感知)得到的二维时频表示。它是现代神经 TTS 系统中最广泛使用的中间声学特征。

**计算流程**:
1. 对语音波形做分帧(典型: 帧长 50ms, 帧移 12.5ms / hop size 256 @ 22.05kHz)
2. 每帧做 FFT 得到线性频谱
3. 通过 Mel 滤波器组(典型: 80 个三角滤波器)映射到 Mel 尺度
4. 取对数压缩动态范围 → log-mel spectrogram

**典型参数**:
- 采样率: 16kHz / 22.05kHz / 24kHz
- FFT 点数: 1024 / 2048
- Mel 频带数: 80 (主流) / 128
- Hop size: 256 samples (约 10ms @ 24kHz)

## 在 TTS 中的应用

Mel spectrogram 在 neural TTS pipeline 中扮演**声学模型与声码器之间的桥梁**:

1. **作为声学模型的输出目标**: Tacotron 2, FastSpeech 1/2, TransformerTTS 等主流声学模型都预测 mel spectrogram
2. **作为声码器的输入**: WaveGlow, HiFi-GAN, WaveRNN 等神经声码器从 mel spectrogram 合成波形
3. **损失函数**: L1/L2 mel-spectrogram loss 是训练声学模型的主要监督信号

**替代方案演进**:
- 早期 SPSS: 使用 MGC (mel-generalized cepstral coefficients) + BAP + F0
- 中期 neural TTS: 使用 mel spectrogram (高维但信息更丰富)
- 近期 LLM-TTS: 使用离散 speech tokens (VQ-VAE / FSQ 量化后的表示)

**优势**: 相比原始波形,mel spectrogram 压缩了时间维度(约 100x),保留了人耳可感知的主要频谱信息,且连续值便于梯度优化。

**局限**: mel spectrogram 丢失了相位信息,从 mel 恢复波形需要神经声码器或 Griffin-Lim 算法(质量较差)。

## 关键论文

- Tacotron (Wang et al., Interspeech 2017): 首个直接预测 linear spectrogram 的端到端模型
- Tacotron 2 (Shen et al., ICASSP 2018): 转向预测 mel spectrogram + WaveNet vocoder 的标准范式
- FastSpeech (Ren et al., NeurIPS 2019): NAR 并行生成 mel spectrogram
- HiFi-GAN (Kong et al., NeurIPS 2020): mel → waveform 的高效 GAN vocoder

## 相关概念

- [[Neural Vocoder]]: 从 mel spectrogram 合成波形
- [[Speech Tokenizer]]: mel spectrogram 的离散化替代方案
- [[Text-to-Speech Pipeline]]: mel spectrogram 作为 pipeline 中间表示
- Linear Spectrogram: 未经 Mel 映射的原始 STFT 频谱
- MFCC: 从 mel spectrogram 进一步做 DCT 得到的紧凑表示

## 演进

MGC/MCC+BAP+F0 (SPSS时代) → Linear Spectrogram (Tacotron, 2017) → **Mel Spectrogram** (Tacotron 2, 2018; 成为主流) → Discrete Speech Tokens (VALL-E, 2023; LLM-TTS 时代)
