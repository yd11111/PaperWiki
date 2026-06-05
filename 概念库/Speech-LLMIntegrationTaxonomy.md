---
type: concept
title: "Speech-LLM Integration Taxonomy"
aliases: [语音-LLM集成分类, Speech-LLM Integration, LLM-Speech Integration Approaches, 语音与大模型集成方法]
category: "taxonomy"
tags: [speech-LM, integration, taxonomy, multimodal, architecture, survey]
key_papers: ["Yang et al., When LLM Meet Speech, 2025", "[[论文笔记/LLMVoX|LLMVoX]]", "[[论文笔记/Raon-Speech|Raon-Speech]]"]
origin_paper: "Yang et al., When LLM Meet Speech, 2025"
related_concepts: ["[[SpeechLanguageModel]]", "[[LLM-basedTTS]]", "[[ModalityAdaptationforSpeechLLM]]", "[[LLM-enhancedASR]]", "[[Speech-TextAlignment]]", "[[AudioUnderstanding]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Speech-LLM Integration Taxonomy 是 Yang et al. (2025) 提出的语音与大语言模型集成方法的分类框架。与 Cui et al. (2024) 从统一 SpeechLM 视角出发不同,本分类从 **集成接口** 角度将现有方法分为三大类 [Fig 1]:

1. **Text-based Integration**: LLM 以文本为输入输出,通过外部 ASR/TTS 处理语音
2. **Latent-representation-based Integration**: 语音编码器产生连续潜在表征直接送入 LLM
3. **Audio-token-based Integration**: 语音被离散化为 audio tokens 作为 LLM 的输入/输出

**与 SpeechLM 分类的互补关系**: Cui et al. 按 "建模什么特征" 和 "训练阶段" 分类,Yang et al. 按 "语音如何进入/离开 LLM" 分类。两种视角互补,前者是 model-centric,后者是 interface-centric。

## 三大集成方式详述

### 1. Text-based Integration [§3]

LLM 保持原始文本输入输出模式,通过外部模块桥接语音:

| 子方法 | 机制 | 代表系统 |
|--------|------|----------|
| Cascaded Integration [§3.1] | ASR + LLM + TTS 管线串联 | AudioGPT, HuggingGPT |
| LLM Rescoring [§3.2] | LLM 对 ASR N-best 假设重排序 | Chen et al. 2023c, Udagawa et al. 2022 |
| LLM GER [§3.3] | LLM 基于假设列表生成新转写 | HyPoradise, Whispering Llama, MMGER |

**优势**: 解释性最强,对 LLM 适配需求最小
**劣势**: 信息丢失 (副语言信息在 ASR 阶段丢弃), 延迟高, 错误累积

### 2. Latent-representation-based Integration [§4]

语音编码器生成连续帧级表征,经 modality adaptation 后直接送入 LLM embedding space:

**组件**: Speech Encoder (预训练 S3M 或从头训练) → Modality Adapter → LLM

**模态适配方法** [§4.2, Fig 4]:
- Convolutional Downsampling [§4.2.1]
- CTC Compression (blank-removal / frame-averaging / blank-probability) [§4.2.2]
- Q-Former [§4.2.3]

**性能排序** (Yu et al., 2024; Hono et al., 2023): Q-Former > CTC Compression > Convolutional Downsampling

**训练策略** [§4.3]:
- 全模型微调 (最优但昂贵)
- LoRA 等 PEFT (Xu et al., 2024c; Pham et al., 2024: LoRA 用于 LLM 显著提升性能)
- 仅训练 adapter (轻量)
- 两阶段训练: 先训练 encoder 稳定后再启动 PEFT (Wu et al., 2023)

**代表系统**: BLSP, SALM, SALMONN, Qwen-Audio, Qwen2-Audio, LLaST, SpeechVerse, WavLLM, SLAM-ASR, BESTOW

### 3. Audio-token-based Integration [§5]

语音被离散化为 tokens 直接参与 LLM 的语言建模:

| 子方法 | Token 类型 | 代表系统 |
|--------|-----------|----------|
| Semantic Tokens [§5.1.1] | HuBERT/w2v-BERT k-means tokens | VoxtLM, SpeechGPT, TWIST, Spirit-LM, BASE TTS, GPT-Talker |
| Acoustic Tokens [§5.1.2] | Neural codec tokens (EnCodec等) | VALL-E, LauraGPT, Neekhara et al. |
| Semantic + Acoustic [§5.1.3] | 两阶段: semantic → acoustic | AudioPaLM, Moshi, Hibiki, Emova |

## 对比分析

### 优劣排序 [§6.1]

| 维度 | 排序 |
|------|------|
| 集成深度 | Latent-representation > Audio-token > Text-based |
| 可解释性 | Text-based > Audio-token > Latent-representation |
| 语音生成能力 | Text-based, Audio-token 可以; Latent-representation 通常不行 |

### 适用场景

- **资源充足 + 实时需求**: Latent-representation / Audio-token (更深集成,更低延迟)
- **资源有限 + 需解释性**: Text-based (简单管线,易调试)
- **需要语音输出**: Audio-token > Latent-representation (从连续表征生成语音仍具挑战性)
- **纯理解任务**: Latent-representation > Audio-token (语义保留更好)

### 定量对比 [Table 1]

Survey Table 1 对比了不同集成方法在 ASR (LibriSpeech, Fleurs, AISHELL-2, VoxPopuli), S2TT (CoVoST2), S2ST (CVSS), TTS (LibriTTS) 上的表现。关键发现:
- 更深集成 (latent-representation) 在充足资源下通常优于 text-based
- 但 backbone LLM 大小、训练数据量等变量使直接对比困难
- 需要统一实验设置的 fair comparison (目前缺乏) [§7.4]

## 开放挑战 [§7]

1. **Text-based**: 如何传递 ASR 丢失的副语言信息 (prosody, emotion) [§7.1]
2. **Latent-representation**: 语音-文本表征对齐仍不充分,synthetic data 可能缓解 [§7.2]
3. **Audio-token-based**: semantic token 仅捕获语音学信息,acoustic token 语义弱;两者融合研究仍不足 [§7.3]
4. **Fair Comparison**: 缺乏跨方法的统一 benchmark [§7.4]
5. **Multilingualism**: 多数系统仅支持英语,多语言能力受限于 backbone LLM [§7.5]
6. **Real-time Processing**: LLM 体积大导致延迟高,需要更好的在线推理方法 [§7.6]

## 关键论文

- Yang et al., When LLM Meet Speech, 2025: 提出本分类框架
- Cui et al., Speech Language Models Survey, 2024: 互补的 model-centric 分类
- Hono et al., 2023: 对比 modality adaptation 方法
- Yu et al., 2024: Q-Former vs CTC vs Conv downsampling 实验对比
- Pham et al., 2024: LoRA 对 LLM 的系统性对比 (LoRA > partial FT for LLM module)

## 相关概念

- [[SpeechLanguageModel]]: 本分类的 audio-token-based 路线与 SpeechLM 概念高度重叠
- [[LLM-basedTTS]]: 属于本分类中的 text-based (cascaded) 或 audio-token-based 路线
- [[ModalityAdaptationforSpeechLLM]]: latent-representation-based 路线的核心技术
- [[LLM-enhancedASR]]: text-based 路线中的 LLM Rescoring 和 GER
- [[Speech-TextAlignment]]: 与 latent-representation 路线的对齐机制相关
- [[AudioUnderstanding]]: 各集成方式下的理解能力差异

## 演进

Cascaded ASR+LLM+TTS (AudioGPT/HuggingGPT, 2023) → LLM Rescoring/GER (2023) → Latent-representation (BLSP/SALM/Qwen-Audio, 2023-2024) → Audio-token (SpeechGPT/TWIST/Spirit-LM, 2023-2024) → Hybrid integration (Moshi/AudioPaLM, 2024) → 开放问题: 统一评估 + 多语言 + 实时 (2025+)
