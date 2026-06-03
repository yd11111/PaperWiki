---
type: concept
title: "Modality Adaptation for Speech LLM"
aliases: [模态适配, Speech-LLM Adapter, Speech Modality Adapter, 语音模态适配器, Bridge Network, Module Connector, Speech Encoder Adapter]
category: "technique"
tags: [speech-LM, adapter, modality-adaptation, downsampling, CTC, Q-Former, PEFT, LoRA]
key_papers: ["Hono et al., 2023", "Yu et al., 2024", "Pham et al., 2024", "Wu et al., 2023", "Fathullah et al., 2024", "Li et al., 2023a (BLIP-2)", "[[论文笔记/Step-Audio 2.5|StepAudio 2.5]]", "[[论文笔记/GOAT-TTS|GOAT-TTS]]", "[[论文笔记/OpenS2S|OpenS2S]]"]
origin_paper: "Yang et al., When LLM Meet Speech, 2025"
related_concepts: ["[[Speech-LLM Integration Taxonomy]]", "[[Speech-Text Alignment]]", "[[Speech Language Model]]", "[[Speech Tokenizer]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Modality Adaptation for Speech LLM 是 latent-representation-based 集成方法中连接语音编码器和大语言模型的适配机制。其核心目标是将语音编码器输出的帧级 (frame-wise) 连续表征映射为 LLM 可处理的 token 级 (token-wise) 表征,解决两大问题 [§4.2]:

1. **序列长度不匹配**: 语音帧率 50-100 fps 远高于文本 token 率,需要 downsampling
2. **表征空间不对齐**: 语音表征与 LLM 文本 embedding 空间分布不同,需要投影对齐

**与 Speech-Text Alignment 的区别**: Speech-Text Alignment (Cui et al. 2024) 关注 token 序列的排列方式 (交替/拼接/并行); Modality Adaptation 关注从连续帧表征到 LLM 输入的信号处理和映射。两者互补,分别解决 "序列怎么组织" 和 "表征怎么转换" 的问题。

## 三种主要方法

### 1. Convolutional Downsampling [§4.2.1]

最基础的策略,用卷积层对帧级表征降采样:

```
Frame-wise features → Conv layers (stride=N) → Linear Projection → Token-wise input
```

- **简单变体**: kernel size = stride → 等价于堆叠相邻帧 (Fathullah et al., 2024)
- **增强变体**: Conv + FC layer 或 Multi-head Transformer (Yu et al., 2024) → 更好对齐 LLM embedding 空间

**代表系统**: Hono et al. (2023), Fathullah et al. (2024), BLSP, SALM, SpeechVerse, SLAM-ASR, Seed-ASR, WavLLM, LFI, Llama-Omni

**优势**: 实现简单,参数少
**劣势**: 固定降采样率,不适应语音内容的信息密度变化

### 2. CTC Compression [§4.2.2]

基于 CTC (Connectionist Temporal Classification) 的内容感知压缩,两步实现:
1. 在语音编码器上训练 CTC 对齐 (ASR 任务)
2. 根据 CTC 预测结果压缩表征序列

三种具体策略 [Fig 4]:

| 策略 | 机制 | 特点 |
|------|------|------|
| **Blank-removal** [Fig 4b] | 丢弃所有被 CTC 预测为 blank 的帧 | 最激进压缩,可能丢失停顿信息 |
| **Frame-averaging** [Fig 4c] | 对连续相同 CTC 预测的帧取平均 | 保留段级信息,平滑噪声 |
| **Blank-probability** [Fig 4d] | 丢弃 blank 概率超过阈值的帧 | 可调阈值控制压缩率 |

**代表系统**: Hono et al. (2023), Wu et al. (2023), Ling et al. (2024), Speech-Llama

**优势**: 内容感知 (content-aware),根据语音内容动态调整序列长度
**劣势**: 需要额外 CTC 训练; blank-removal 可能丢失副语言信息

### 3. Q-Former [§4.2.3]

Transformer-based 模块,使用 learned queries 将变长输入映射为固定长度输出:

```
Frame-wise features → Q-Former (Learned Queries + Cross-Attention) → Fixed-length output
```

- 源自视觉-语言预训练的 BLIP-2 (Li et al., 2023a)
- 一组可学习的 query tokens 通过 cross-attention 从语音表征中提取信息
- 输出长度由 query 数量决定,与输入长度无关

**代表系统**: Yu et al. (2024), SALMONN (Tang et al., 2024), COSMIC (Pan et al., 2023), XLLM, DesSTA, Secap

**优势**: 输出长度固定,效率最高; 性能最优 (Yu et al., 2024 实验证实)
**劣势**: 额外 query tokens 需要学习; 可能丢失细粒度时序信息

## 性能对比

Hono et al. (2023) 和 Yu et al. (2024) 的实验表明:

```
Q-Former > CTC Compression > Convolutional Downsampling
```

但需注意此排序可能因任务和数据量而变化。

## 训练策略 [§4.3]

Modality adaptation 的训练涉及三个模块的更新策略:

### 全模型微调 (Full Fine-tuning)
- 同时训练 speech encoder + adapter + LLM
- 性能最优但计算成本最高

### PEFT (Parameter-Efficient Fine-Tuning)
- **LoRA for LLM**: Pham et al. (2024) 系统实验证明 LoRA 应用于 LLM 显著提升性能
- **QA-LoRA** (Xu et al., 2024c): 量化感知 LoRA,进一步降低资源需求
- 对 encoder 模块: 全微调 > 部分微调 (但部分微调更具性价比)

### 仅训练 Adapter
- 冻结 speech encoder 和 LLM,只训练中间 adapter
- 最轻量但性能可能受限

### 两阶段训练 (Wu et al., 2023)
- Stage 1: 先训练 speech encoder (此时不启动 PEFT)
- Stage 2: encoder 稳定后启动 LLM 的 PEFT
- 目的: 避免 encoder 不稳定的梯度干扰 LLM 训练

## Speech Encoder 选择 [§4.1]

Adapter 之前的语音编码器有两种来源:

1. **预训练 S3M**: HuBERT (Hsu et al., 2021), Whisper encoder (Radford et al., 2023) → 利用大规模语音预训练知识
2. **从头训练**: Multi-layer Transformer 或 Conformer (Gulati et al., 2020) → 为特定 LLM 集成定制

两者通过 adapter (也称 bridge network / module connector) 与 LLM 连接。

## 在 TTS 中的应用

Modality adaptation 主要服务于语音理解任务 (ASR, S2TT 等)。在 TTS 中的直接应用较少,因为 TTS 的输入通常是文本而非语音。但在以下场景中相关:
- **语音到语音翻译 (S2ST)**: 输入端需要 modality adaptation
- **语音克隆/参考编码**: 参考语音的编码可使用类似 adapter 机制
- **Omni-model**: 同时支持输入输出语音的模型 (如 Moshi) 中输入侧使用 modality adaptation

## 关键论文

- Hono et al. (2023): 首个系统对比 Conv downsampling vs CTC compression
- Yu et al. (2024): 加入 Q-Former 对比,证实 Q-Former 最优
- Pham et al. (2024): LoRA 在 LLM module 上的系统性实验
- Wu et al. (2023): 两阶段训练策略 (encoder-first, then PEFT)
- BLIP-2 / Li et al. (2023a): Q-Former 的原始提出 (视觉-语言)
- Fathullah et al. (2024): 探索 prompting LLM 的语音识别能力

## 相关概念

- [[Speech-LLM Integration Taxonomy]]: modality adaptation 是 latent-representation 路线的核心
- [[Speech-Text Alignment]]: 互补关系 -- alignment 管 token 排列, adaptation 管表征转换
- [[Speech Language Model]]: SpeechLM 中的组件之一
- [[Speech Tokenizer]]: audio-token 路线的替代方案,将语音离散化而非连续映射

## 演进

Random downsampling (Wang et al., 2023c, 早期) → Convolutional downsampling (Hono et al., 2023) → CTC compression (blank-removal/frame-averaging, 2023) → Q-Former 引入 (BLIP-2 → 语音, 2024) → PEFT + adapter 联合训练 (LoRA, 2024) → 两阶段训练策略 (Wu et al., 2023) → 开放问题: 最优 adapter 架构因任务而异
