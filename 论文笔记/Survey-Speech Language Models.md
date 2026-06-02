---
type: paper-note
title: "Recent Advances in Speech Language Models: A Survey"
authors: [Wenqian Cui, Dianzhi Yu, Xiaoqi Jiao, Ziqiao Meng, Guangyan Zhang, Qichao Wang, Yiwen Guo, Irwin King]
year: 2024
venue: "arXiv:2410.03751"
tier: card
tags: [survey, speech-LM, multimodal, full-duplex, cold-start]
created: 2026-06-02
updated: 2026-06-02
---

## 概要

首篇全面综述 Speech Language Models (SpeechLMs) 的 survey。SpeechLM 区别于传统 ASR+LLM+TTS 管线方式,直接将语音编码为 token 后用语言模型端到端处理,解决了信息丢失、延迟累积和错误传播三大问题。Survey 系统地覆盖了 SpeechLM 的组件架构、训练策略、下游应用、评估方法和未来方向。

## 核心贡献

1. **SpeechLM 定义**: 端到端处理和生成语音的自回归基础模型,同时支持语音和文本的跨模态输入输出
2. **组件分类** (Section III): Speech Tokenizer (semantic/acoustic/mixed) + Language Model + Vocoder
3. **训练策略分类** (Section IV): Features Modeled (discrete/continuous) × Training Stages (pre-training/instruction-tuning/post-alignment) × Speech Generation Paradigm (traditional/real-time/IPR)
4. **综合分类体系** (Figure 4): 按组件和训练策略对 50+ SpeechLM 系统进行分类
5. **评估体系** (Section VI): 自动评估 (表征/语言/韵律/质量) + 人工评估 (MOS/SMOS/PMOS)

## 知识提取成果

本文用于概念库冷启动 #3,提取 Speech LM 领域的核心领域知识。

### 新建概念页 (6 个)

| 概念 | 核心内容 |
|------|----------|
| [[Speech Language Model]] | SpeechLM 整体定义、组件架构、分类体系、与 ASR+LLM+TTS 的区别 |
| [[Semantic vs Acoustic Tokens]] | 两类语音 token 的设计目标、trade-off、层级建模方案 |
| [[Codec Language Model]] | 直接建模 neural codec tokens 的 LM 范式 (VALL-E/Viola/AudioLM) |
| [[Full-duplex Spoken Dialogue]] | 实时双向语音交互、用户打断、同步响应 |
| [[Speech-Text Alignment]] | SpeechLM 中语音和文本模态的对齐方法 |
| [[Audio Understanding]] | SpeechLM 的语音/音频理解能力和任务体系 |

### 更新概念页 (2 个)

| 概念 | 更新内容 |
|------|----------|
| [[Speech Tokenizer]] | 补充 mixed objective tokenizer 详述、SpeechLM 视角下的三类体系 |
| [[LLM-based TTS]] | 补充 SpeechLM 上下文定位,CodecLM 关系澄清 |

## Survey 关键分类体系

### SpeechLM vs ASR+LLM+TTS 管线 (Figure 1)
```
ASR+LLM+TTS: Speech → ASR → Text → LLM → Text → TTS → Speech
  问题: 信息丢失(副语言信息)、高延迟(串行)、错误累积

SpeechLM:     Speech/Text → [Speech Tokenizer] → [Language Model] → [Vocoder] → Speech/Text
  优势: 保留副语言信息、低延迟(端到端)、无累积错误
```

### 组件选择 (Table II)
- Speech Tokenizer: Whisper / HuBERT / SpeechTokenizer / EnCodec / Mimi / w2v-BERT
- Language Model: LLaMA / Qwen2 / OPT / Transformer* / Mistral / GLM
- Vocoder: HiFi-GAN / CosyVoice Decoder / BigVGAN / SoundStream / VITS

### 训练策略分类 (Figure 4)
```
Features Modeled:
  ├─ Discrete: Semantic (GSLM, TWIST) / Paralinguistic (pGSLM, SPIRIT-LM)
  │           Acoustic (VioLA, Parrot) / Mixed (Moshi, SpeechGPT-Gen)
  └─ Continuous: Spectron, Mini-Omni, LauraGPT, SLAM-Omni

Training Stages:
  ├─ Cold Init Pre-Training: GSLM, SUTLM, pGSLM, LSLM, VioLA
  ├─ Continued Pre-Training: TWIST, AudioPaLM, SPIRIT-LM, Moshi, Mini-Omni, Spectron
  └─ Instruction-Tuning: SpeechGPT, SpeechGPT-Gen, COSMIC, Llama-Omni, Moshi

Speech Generation Paradigm:
  ├─ Traditional: TWIST, SPIRIT-LM, AudioPaLM, SpeechGPT
  ├─ Real-Time: dGSLM, NTPP, LSLM, VITA, Moshi, Mini-Omni 2, OmniFlatten
  └─ Interactive Period Recognition: VITA, MiniCPM-o 2.6, FlexDuo
```

## 代表模型谱系

| 模型 | 年份 | Tokenizer | LM | 特点 |
|------|------|-----------|-----|------|
| GSLM | 2021 | HuBERT+CPC+wav2vec2 | Transformer* | 首个 SpeechLM, 对比 3 种 tokenizer |
| AudioLM | 2022 | w2v-BERT + SoundStream | Transformer* | Semantic → Acoustic 两阶段 |
| TWIST | 2024 | HuBERT | OPT/LLaMA | 首次证明 TextLM 预训练加速 SpeechLM |
| SPIRIT-LM | 2024 | HuBERT+pitch+style | OPT | 交替 text-speech token 训练 |
| Moshi | 2024 | Mimi | Transformer* | 全双工, 8 codebook 同时生成, RQ-Transformer |
| SpeechGPT | 2023 | SpeechTokenizer | LLaMA | 端到端 speech-text instruction-following |
| Mini-Omni | 2024 | Whisper+ASR Adapter | Qwen2 | Text+7 acoustic streams 并行, streaming |
| VITA | 2024 | CNN+Transformer+MLP | Transformer* | IPR, 非查询音频过滤 |
| dGSLM | 2023 | HuBERT | Dialogue Transformer | 双说话人 channel, cross-attention |

## 关键发现

1. **Tokenizer 选择影响最大**: HuBERT semantic tokens 在语义理解最佳,但缺声学细节; EnCodec acoustic tokens 保真度高但语义对齐差; SpeechTokenizer/Mimi 混合路线是趋势
2. **TextLM 初始化显著优于冷启动**: TWIST 证明从 TextLM checkpoint 继续训练可加速收敛、提升理解能力
3. **Speech-text 对齐方式多样**: 单序列交替 (SPIRIT-LM) vs 多序列并行 (Mini-Omni) vs text-present vs text-independent
4. **全双工是前沿**: dGSLM → Moshi → VITA → MiniCPM-o 2.6 逐步实现用户打断和同步响应
5. **安全风险独特**: SpeechLM 的 toxicity 包含声学维度 (erotic speech 等), privacy 风险更高 (说话人身份推断)
