---
type: concept
title: "Speech Language Model"
aliases: [SpeechLM, 语音语言模型, Spoken Language Model, Multimodal Speech LLM, Speech Foundation Model, Omni-model]
category: "model-family"
tags: [speech-LM, multimodal, foundation-model, end-to-end, autoregressive, speech-interaction]
key_papers: ["GSLM (Lakhotia et al., 2021)", "[[论文笔记/AudioLM|AudioLM]]", "TWIST (Hassid et al., 2024)", "SPIRIT-LM (Nguyen et al., 2024)", "[[论文笔记/Moshi|Moshi]]", "SpeechGPT (Zhang et al., 2023)", "Mini-Omni (Xie & Wu, 2024)", "VITA (Fu et al., 2024)", "AudioPaLM (Rubenstein et al., 2023)", "Yang et al., When LLM Meet Speech, 2025", "[[论文笔记/UniAudio|UniAudio]]", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/Step-Audio 2.5|StepAudio 2.5]]", "[[论文笔记/wav2vec 2.0|wav2vec 2.0]]", "[[论文笔记/WavLM|WavLM]]", "[[论文笔记/w2v-BERT|w2v-BERT]]", "[[论文笔记/StableToken|StableToken]]", "[[论文笔记/STITCH|STITCH (Chiang et al., ICLR 2026)]]", "[[论文笔记/CoT-ST|CoT-ST]]", "[[论文笔记/SpeechWorldModel|SpeechWorldModel]]", "[[论文笔记/TADA|TADA]]", "[[论文笔记/NAST|NAST]]", "[[论文笔记/Vec-Tok Speech|Vec-Tok Speech]]", "[[论文笔记/GOAT-TTS|GOAT-TTS]]", "[[论文笔记/OpenOmni|OpenOmni]]", "[[论文笔记/LLaMA-Omni 2|LLaMA-Omni 2]]", "[[论文笔记/LM-SPT|LM-SPT]]", "[[论文笔记/OpenS2S|OpenS2S]]", "[[论文笔记/TTS-1|TTS-1 (Inworld, 2025)]]", "[[论文笔记/DualSpeechLM|DualSpeechLM]]", "[[论文笔记/DialoSpeech|DialoSpeech]]", "[[论文笔记/Ming-UniAudio|Ming-UniAudio]]", "[[论文笔记/UniVoice|UniVoice]]"]
origin_paper: "[[论文笔记/Survey-Speech Language Models|Cui et al., Speech Language Models, 2024]]"
related_concepts: ["[[Speech Tokenizer]]", "[[Codec Language Model]]", "[[LLM-based TTS]]", "[[Semantic vs Acoustic Tokens]]", "[[Speech-Text Alignment]]", "[[Full-duplex Spoken Dialogue]]", "[[Audio Understanding]]", "[[Neural Vocoder]]", "[[Speech-LLM Integration Taxonomy]]", "[[Modality Adaptation for Speech LLM]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Speech Language Model (SpeechLM) 是一种端到端处理和生成语音的自回归基础模型。与传统 ASR+LLM+TTS 级联管线不同,SpeechLM 直接将语音编码为 token 或连续表征,用语言模型建模后再解码为语音,同时支持文本的跨模态输入输出。

**形式化定义** (Cui et al., 2024):
```
M^out = SpeechLM(M^in; θ)
```
其中 M^in 和 M^out 是多模态序列 (speech 和/或 text 的混合),θ 为模型参数。

**ASR+LLM+TTS 管线的三大问题**:
1. **信息丢失**: ASR 丢弃副语言信息 (pitch, timbre, emotion, tonality)
2. **高延迟**: 三个模块串行执行,复杂解码进一步增加延迟
3. **错误累积**: ASR 转写错误传播到 LLM,LLM 生成文本可能 TTS 无法合成

**SpeechLM 的解决方案**: 直接对语音 token 建模,保留副语言信息、消除级联延迟、避免跨模块错误传播。

## 三大组件

SpeechLM 由三个核心组件构成 (Survey Fig. 1b):

### 1. Speech Tokenizer
将连续语音波形编码为 token/表征,供语言模型处理。三类:
- **Semantic tokenizer** (HuBERT, w2v-BERT): 侧重语义内容
- **Acoustic tokenizer** (EnCodec, SoundStream): 侧重声学保真
- **Mixed tokenizer** (SpeechTokenizer, Mimi): 兼顾语义和声学

### 2. Language Model
对语音 token 进行自回归建模。核心适配:
- 将 text embedding matrix E_t 替换为 speech embedding matrix E_s
- 输出矩阵相应替换为 E'_s
- 或扩展词表同时支持 text 和 speech token

### 3. Vocoder
将语言模型输出的 token 合成为语音波形:
- **Direct synthesis**: 直接从 speech token 生成 (HiFi-GAN)
- **Input-enhanced synthesis**: 先转为 mel-spectrogram 等中间表征再合成 (CFM + HiFi-GAN)

## 分类体系

### 按 Features Modeled (IV-A)
| 类型 | 特点 | 代表系统 |
|------|------|----------|
| Semantic tokens (discrete) | 语义理解强,声学细节弱 | GSLM, TWIST, SpeechGPT |
| Paralinguistic tokens | 补充 F0/duration/style | pGSLM, SPIRIT-LM |
| Acoustic tokens (discrete) | 高保真但语义对齐差 | VioLA, Parrot |
| Mixed tokens (discrete) | 兼顾语义和声学 | Moshi, SpeechGPT-Gen |
| Continuous features | mel-spectrogram 或 latent | Spectron, Mini-Omni, SLAM-Omni |

### 按 Training Stages (IV-B)
1. **Cold initialization**: 从零训练 (GSLM, SUTLM, pGSLM)
2. **Continued pre-training**: 从 TextLM 继续训练 speech 能力 (TWIST, AudioPaLM, SPIRIT-LM, Moshi, Mini-Omni)
3. **Instruction-tuning**: 微调以遵循指令 (SpeechGPT, COSMIC, Llama-Omni)
4. **Post-alignment**: RLHF/DPO 对齐人类偏好 (Align-SLM, SpeechAlign)

### 按 Speech Generation Paradigm (IV-C)
1. **Traditional**: 接收完整输入,生成完整响应
2. **Real-time interaction**: 流式处理,低延迟响应
3. **Interactive Period Recognition (IPR)**: 识别是否应响应 (VITA, MiniCPM-o 2.6, FlexDuo)

## 代表模型演进

```
GSLM (2021, 首个 SpeechLM, HuBERT+Transformer, 冷启动)
  ↓
pGSLM (2022, 加入 F0/duration 副语言 tokens)
  ↓
AudioLM (2022, semantic→acoustic 两阶段, SoundStream)
  ↓
TWIST (2024, 证明 TextLM 初始化显著优于冷启动)
  ↓
SPIRIT-LM (2024, text-speech 交替 token 训练, pitch/style tokens)
  ↓
SpeechGPT (2023, 端到端 instruction-following SpeechLM)
  ↓
Moshi (2024, 全双工, Mimi tokenizer, RQ-Transformer, 8 codebook 并行)
  ↓
Mini-Omni (2024, text+7 acoustic streams 并行, 流式推理)
  ↓
VITA (2024, IPR, 多模态 vision+speech+text)
```

## 能力全景

SpeechLM 的下游应用覆盖三大类 (Survey Section V):

**语义相关**: 口语对话、语音翻译、ASR、关键词检测、TTS、意图识别、槽填充
**说话人相关**: 说话人识别/验证/分段、声音条件生成
**副语言相关**: 情感识别、语音分离、副语言增强生成

## 互补分类视角: Speech-LLM Integration Taxonomy

Yang et al. (2025) 提出了一种互补的分类视角: 从 **集成接口** 而非 **模型架构** 出发,将 speech-LLM 方法分为 text-based / latent-representation-based / audio-token-based 三类 [§1, Fig 1]。

- SpeechLM (本页) 主要对应 audio-token-based integration 路线
- Latent-representation-based 路线 (如 Qwen-Audio, SALMONN) 使用连续表征而非离散 tokens,需要 [[Modality Adaptation for Speech LLM]] 桥接语音编码器和 LLM
- Text-based 路线 (如 AudioGPT, [[LLM-enhanced ASR]]) 通过 ASR/TTS 文本中转,无需修改 LLM

两种分类互补: 本页 (Cui et al. 2024) 是 model-centric 视角,Yang et al. 2025 是 interface-centric 视角。详见 [[Speech-LLM Integration Taxonomy]]。

## 与 LLM-based TTS 的关系

LLM-based TTS (如 VALL-E 系列) 是 SpeechLM 在 TTS 任务上的特例:
- LLM-based TTS 专注于 text→speech 生成
- SpeechLM 是更广义的概念,支持 speech→speech, speech→text, text→speech, speech+text→speech+text 等多种模态组合
- SpeechLM 还具备理解能力 (ASR, 情感识别等),不仅限于生成

## 关键论文

- GSLM (Lakhotia et al., 2021): 首个 SpeechLM, 对比 CPC/wav2vec2/HuBERT
- AudioLM (Borsos et al., 2023): semantic → acoustic 两阶段层级建模
- TWIST (Hassid et al., 2024): TextLM 预训练加速 SpeechLM 收敛
- SPIRIT-LM (Nguyen et al., 2024): text-speech 交替 + paralinguistic tokens
- Moshi (Defossez et al., 2024): 全双工实时对话, Mimi tokenizer
- SpeechGPT (Zhang et al., 2023): 端到端 speech-text instruction-following
- AudioPaLM (Rubenstein et al., 2023): PaLM-2 为基础的大规模 SpeechLM

## 相关概念

- [[Speech Tokenizer]]: SpeechLM 的第一组件,将语音离散化
- [[Codec Language Model]]: SpeechLM 中直接建模 codec tokens 的子范式
- [[LLM-based TTS]]: SpeechLM 在 TTS 任务上的特例
- [[Semantic vs Acoustic Tokens]]: SpeechLM 中 token 类型选择的核心 trade-off
- [[Speech-Text Alignment]]: SpeechLM 中 speech 和 text 模态的对齐方法
- [[Full-duplex Spoken Dialogue]]: SpeechLM 的前沿交互范式
- [[Audio Understanding]]: SpeechLM 的理解能力体系

## 演进

Naive ASR+LLM+TTS 管线 (2020-) → GSLM (首个 SpeechLM, 2021) → AudioLM/pGSLM (层级建模+副语言, 2022) → TWIST/SPIRIT-LM (TextLM 初始化+跨模态训练, 2024) → SpeechGPT/Moshi/Mini-Omni (全双工 omni-model, 2024) → VITA/MiniCPM-o (IPR+多模态融合, 2024-2025)
