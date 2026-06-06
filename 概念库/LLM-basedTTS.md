---
type: concept
title: "LLM-based TTS"
aliases: [LLM TTS, 大语言模型语音合成, LLM-driven Speech Synthesis]
category: "model-family"
tags: [TTS, LLM, autoregressive, codec, zero-shot, in-context-learning, decoder-only]
key_papers: ["VALL-E (Wang et al., 2023)", "VALL-E X (Zhang et al., 2023d)", "VALL-E 2 (Chen et al., 2024a)", "VALL-E R (Han et al., 2024)", "ELLA-V (Song et al., 2024)", "RALL-E (Xin et al., 2024)", "[[论文笔记/MELLE|MELLE]]", "[[论文笔记/Spark-TTS|Spark-TTS]]", "HALL-E (Nishimura et al., 2024)", "[[论文笔记/SPEAR-TTS|SPEAR-TTS]]", "[[论文笔记/Make-A-Voice|Make-A-Voice]]", "Yang et al., When LLM Meet Speech, 2025", "[[论文笔记/Mega-TTS|Mega-TTS]]", "[[论文笔记/Fish-Speech|Fish-Speech]]", "[[论文笔记/GLM-TTS|GLM-TTS]]", "[[论文笔记/UniAudio|UniAudio]]", "[[论文笔记/FireRedTTS|FireRedTTS]]", "[[论文笔记/FireRedTTS2|FireRedTTS 2]]", "[[论文笔记/Mega-TTS2|Mega-TTS 2]]", "[[论文笔记/BASETTS|BASE TTS]]", "[[论文笔记/HierSpeech++|HierSpeech++]]", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/Step-Audio2.5|StepAudio 2.5]]", "[[论文笔记/TortoiseTTS|Tortoise TTS]]", "[[论文笔记/FELLE|FELLE]]", "[[论文笔记/LatentLM|LatentLM]]", "[[论文笔记/CLEAR|CLEAR]]", "[[论文笔记/VibeVoice|VibeVoice]]", "[[论文笔记/SpeechAlign|SpeechAlign]]", "[[论文笔记/RL-for-Audio-LLM|RL-for-Audio-LLM]]", "[[论文笔记/RIO|RIO]]", "[[论文笔记/FlexiVoice|FlexiVoice]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/TextrolSpeech|TextrolSpeech]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/VoXtream|VoXtream]]", "[[论文笔记/Multi-RewardGRPO|Multi-Reward GRPO]]", "[[论文笔记/VoxtralTTS|Voxtral TTS]]", "[[论文笔记/FishAudioS2|Fish Audio S2]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/MamTra|MamTra]]", "[[论文笔记/TADA|TADA]]", "[[论文笔记/SemaVoice|SemaVoice]]", "[[论文笔记/OmniVoice|OmniVoice]]", "[[论文笔记/LLaDA-TTS|LLaDA-TTS]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/TacoLM|TacoLM]]", "[[论文笔记/Vec-TokSpeech|Vec-Tok Speech]]", "[[论文笔记/RWKVTTS|RWKVTTS]]", "[[论文笔记/Muyan-TTS|Muyan-TTS]]", "[[论文笔记/DiffCSS|DiffCSS]]", "[[论文笔记/GOAT-TTS|GOAT-TTS]]", "[[论文笔记/EmoVoice|EmoVoice]]", "[[论文笔记/FPO|FPO]]", "[[论文笔记/LLMVoX|LLMVoX]]", "[[论文笔记/TTS-Transducer|TTS-Transducer]]", "[[论文笔记/BreezyVoice|BreezyVoice]]", "[[论文笔记/Koel-TTS|Koel-TTS]]", "[[论文笔记/IndexTTS|IndexTTS]]", "[[论文笔记/CSP-FT|CSP-FT (Wang et al., 2026)]]", "[[论文笔记/MAEStyle-RichTTS|MAE Style-Rich TTS]]", "[[论文笔记/UniTTS|UniTTS]]", "[[论文笔记/Prompt-Unseen-Emotion|PUE (Gao et al., 2025)]]", "[[论文笔记/JoyTTS|JoyTTS]]", "[[论文笔记/SpeechAccentLLM|SpeechAccentLLM]]", "[[论文笔记/RevivalwithVoice|Revival with Voice]]", "[[论文笔记/SMLLE|SMLLE]]", "[[论文笔记/UDDETTS|UDDETTS]]", "[[论文笔记/StreamMel|StreamMel]]", "[[论文笔记/MPO|MPO]]", "[[论文笔记/DeepDubbing|DeepDubbing]]", "[[论文笔记/Llasa+|Llasa+]]", "[[论文笔记/LatinX|LatinX]]", "[[论文笔记/MELA-TTS|MELA-TTS]]", "[[论文笔记/TTS-1|TTS-1 (Inworld, 2025)]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/CTDiffusion|CTDiffusion]]", "[[论文笔记/BridgeCode|BridgeCode]]", "[[论文笔记/TKTO|TKTO]]", "[[论文笔记/DialoSpeech|DialoSpeech]]", "[[论文笔记/HD-PPT|HD-PPT]]", "[[论文笔记/UniVoice|UniVoice]]", "[[论文笔记/E2E-VGuard|E2E-VGuard]]", "[[论文笔记/Ming-UniAudio|Ming-UniAudio]]", "[[论文笔记/BatonVoice|BatonVoice]]", "[[论文笔记/Align2Speak|Align2Speak]]", "[[论文笔记/Audiobook-CC|Audiobook-CC]]", "[[论文笔记/SAC|SAC]]", "[[论文笔记/GRPO-TTS|GRPO-TTS]]"]
origin_paper: "Xie et al., Controllable TTS in LLM Era, 2024"
related_concepts: ["[[SpeechTokenizer]]", "[[ResidualVectorQuantization]]", "[[Non-autoregressiveTTS]]", "[[SpeakerEmbedding]]", "[[ConditionalFlowMatching]]", "[[SpeechLanguageModel]]", "[[CodecLanguageModel]]", "[[SemanticvsAcousticTokens]]", "[[Speech-LLMIntegrationTaxonomy]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

LLM-based TTS 是受大语言模型 in-context learning 成功启发的语音合成范式。其核心思想是将 TTS 重构为条件语言建模任务: 将语音离散化为 token 序列,使用自回归 decoder-only transformer 生成语音 token,再解码为波形。

**典型架构** (如 Fig. 2 in survey):
```
Text + Instruction → [Text Encoder] → Text Tokens
Reference Speech → [Speech Encoder] → Audio Tokens (prompt)
[Decoder-only Transformer]: Text Tokens + Audio Prompt → Generated Audio Tokens
Audio Tokens → [Speech Decoder / Vocoder] → Waveform
```

**与传统 TTS 的本质区别**:
- 传统 TTS: 显式 duration/pitch/energy predictor → 确定性映射
- LLM-based TTS: 隐式建模所有 variation → in-context learning 从 prompt 推断

## 核心设计

### 1. 离散语音表示 (Discrete Speech Tokens)

LLM-based TTS 将连续语音信号离散化为 token,使其可被 LM 建模:
- **Neural codec tokens**: EnCodec (Defossez et al., 2023a), SoundStream, HiFi-Codec
- **Semantic tokens**: HuBERT/wav2vec2 离散化的自监督表示
- **Hierarchical tokens**: 粗粒度 semantic + 细粒度 acoustic (VALL-E 的 AR + NAR)

### 2. 两阶段生成 (Two-stage Pipeline)

VALL-E 开创的典型两阶段:
1. **AR 阶段**: 自回归生成第一层 coarse codec tokens (建模语义和韵律)
2. **NAR 阶段**: 非自回归生成剩余层 fine codec tokens (补充声学细节)

### 3. In-context Learning for Style

通过 prompt speech 实现零样本控制:
- 将参考语音的 codec tokens 作为 prefix
- 模型从 prefix 中隐式学习说话人、风格、情感
- 不需要显式 speaker embedding 或 style label

## VALL-E 系列演进

| 模型 | 年份 | 关键改进 |
|------|------|----------|
| VALL-E | 2023 | 开创 codec LM TTS, AR+NAR 两阶段 |
| VALL-E X | 2023 | 跨语言零样本 TTS |
| VALL-E 2 | 2024 | 重复感知采样 + 分组编码, 人类水平 |
| VALL-E R | 2024 | 单调对齐增强鲁棒性 |
| ELLA-V | 2024 | 对齐引导序列重排序 |
| RALL-E | 2024 | Chain-of-thought prompting |
| MELLE | 2024 | 连续值 mel 预测替代离散 token |
| HALL-E | 2024 | 分层 codec LM, 长语音合成 |

## 其他 LLM-based 方法

**Semantic token 路线**:
- SpearTTS (Kharitonov et al., 2023): semantic → acoustic 两阶段
- Make-a-Voice (Huang et al., 2023b): 离散表示统一语音合成

**Tokenizer 优化路线**:
- FireRedTTS (Guo et al., 2024a): 优化 tokenizer 架构改善重建质量
- CoFi-Speech (Guo et al., 2024b): 粗到细多尺度生成策略

**Hybrid 架构** (LLM + Flow/Diffusion):
- CosyVoice (Du et al., 2024): LLM 生成 semantic tokens + CFM 合成高保真语音
- NaturalSpeech 3 (Ju et al., 2024): factorized diffusion codec
- SimpleSpeech (Yang et al., 2024c): scalar latent transformer + flow-based

**Continuous-valued AR (Next-Token Diffusion) 路线**:
- LatentLM (Sun et al., 2024): 提出 next-token diffusion 统一框架,用 sigma-VAE 编码连续数据 + per-token diffusion head 自回归生成,在 TTS 上以 10x 更少解码步数超越 VALL-E 2
- CLEAR (Wu et al., 2025): 用 MLP rectified flow head + enhanced VAE (2048x 压缩) 实现单阶段零样本 TTS,RTF 0.18, 流式 96ms 首帧延迟
- VibeVoice (Peng et al., 2025): 基于 LatentLM + Qwen2.5,双 tokenizer (acoustic+semantic),实现 90 分钟多说话人对话,超越 Gemini 2.5 Pro TTS

**Dialogue/Multi-speaker 扩展**:
- FireRedTTS-2 (Xie et al., 2025): Text-speech interleaved format + dual-transformer (backbone + decoder) 实现长对话语音生成,支持 podcast 和交互式聊天;12.5Hz streaming tokenizer 缩短序列长度
- VibeVoice (Peng et al., 2025): next-token diffusion + 3200x causal tokenizer (7.5Hz) + Qwen2.5 7B,支持最多 4 说话人、最长 90 分钟对话,Preference/Realism/Richness MOS 均超越 Gemini 2.5 Pro
- [[论文笔记/MOSS-TTSD|MOSS-TTSD]] (Zhang et al., 2026): Qwen3-8B-base + MOSS-Audio-Tokenizer (16 层 RVQ, 2kbps/12.5Hz) + multi-head delay pattern,支持 5 说话人、60 分钟单次生成、零样本声音克隆;提出基于 forced alignment 的 TTSD-eval 评估框架;三阶段 curriculum learning 从单人到多人对话
- [[论文笔记/JoyVoice|JoyVoice]] (Yu et al., JD, 2025): E2E Transformer-DiT 联合训练(AR hidden states 直接 conditioning 全局因果 DiT) + MM-Tokenizer (12.5Hz) + 无分割多说话人序列建模(最多 8 说话人、5 分钟),SEED test-zh CER 0.97%,自建 MSMT-eval 多说话人基准 cpCER 1.88% (2spk-zh)

## 在可控性方面的特点

**优势**:
- 自然语言驱动控制: 通过 instruction prompt 直接指定风格
- 零样本声音克隆: 仅需几秒参考音频
- 上下文感知: 理解并生成语义一致的语音

**局限**:
- 离散 token 的量化损失 → 声学瑕疵
- 高计算成本: 长序列自回归推理慢
- 细粒度控制困难: 难以精确控制 pitch/energy/duration
- 稳定性问题: 可能出现 word skip/repeat

## 与 Speech Language Model / Codec Language Model 的关系

Cui et al. (2024) 的 SpeechLM survey 厘清了三者关系:
- **Speech Language Model (SpeechLM)** 是最广义的概念,指端到端处理和生成语音的自回归基础模型,支持 speech↔speech, speech↔text 等多种模态组合
- **Codec Language Model (CodecLM)** 是 SpeechLM 中直接建模 neural codec acoustic tokens 的子范式,可用于 ASR、TTS、ST 等多种任务
- **LLM-based TTS** 是 SpeechLM/CodecLM 在 TTS 任务上的特例,专注于 text→speech 生成

简言之: LLM-based TTS ⊂ CodecLM (codec 路线) 或 SpeechLM (semantic 路线),取决于使用的 token 类型。VALL-E 同时属于 CodecLM 和 LLM-based TTS; CosyVoice 属于 SpeechLM (semantic tokens) + LLM-based TTS。

详见 [[SpeechLanguageModel]] 和 [[CodecLanguageModel]]。

## Integration 视角下的 LLM-based TTS (Yang et al. 2025 补充)

Yang et al. (2025) [§5, Table 1] 从集成分类角度审视 LLM-based TTS:

- **Audio-token → Acoustic Tokens 路线**: VALL-E, LauraGPT, Neekhara et al. 直接用 neural codec tokens
- **Audio-token → Semantic Tokens 路线**: SpeechGPT, TWIST, Spirit-LM 先生成 semantic tokens 再转换
- **Audio-token → Semantic + Acoustic 路线**: AudioPaLM, Moshi 两阶段层级生成

Table 1 (LibriTTS TTS 结果) 显示 audio-token-based 方法在 TTS 上表现优于 text-based cascaded 方法,且 semantic + acoustic 联合路线获得最优 MOS。

**计算代价对比** [§6.3]: text-based 和 audio-token-based 的 LLM 推理成本相当 (受限于 LLM 大小),但 latent-representation-based 需同时运行 speech encoder 和 LLM,总成本更高。

详见 [[Speech-LLMIntegrationTaxonomy]] 的完整对比分析。

## 研究趋势

Survey 指出 TTS 模型架构演进路线:
```
Traditional CNN/RNN (Tacotron, DeepVoice) → 有限控制力
    ↓
Flow-based (Matcha-TTS, F5-TTS) → 快+少量控制
    ↓
LLM-based (VALL-E, InstructTTS) → 慢+自然语言控制
    ↓
Hybrid (CosyVoice) → 直觉控制 + 高保真
    ↓
Future: Instruction-Aware Frameworks → 精细指令控制
```

## 关键论文

- VALL-E (Wang et al., 2023a): 开创 neural codec language model TTS
- SpearTTS (Kharitonov et al., 2023): semantic + acoustic 两阶段 LM
- CosyVoice (Du et al., 2024): LLM + flow matching hybrid
- VoxInstruct (Zhou et al., 2024): instruction-to-speech 统一框架
- InstructSpeech (Huang et al., 2024a): multi-task LLM 语音编辑

## 相关概念

- [[SpeechTokenizer]]: LLM-based TTS 的前提, 将语音离散化
- [[ResidualVectorQuantization]]: codec token 的主要量化方法
- [[Non-autoregressiveTTS]]: LLM-based TTS 中 NAR 阶段 / 对比范式
- [[ConditionalFlowMatching]]: hybrid 架构中的声学生成器
- [[SpeakerEmbedding]]: 被 in-context prompt 部分取代

## 演进

Tacotron/FastSpeech (显式 variance predictor, 2017-2020) → VALL-E (codec LM, AR+NAR, 2023) → VALL-E 2 / RALL-E (鲁棒性增强, 2024) → Hybrid CosyVoice (LLM + Flow, 2024) → Instruction-aware 架构 (VoxInstruct, Step-Audio, 2024-2025) → Gen 4 "Native Agentic" 框架 (Any2Speech/BorderlessLongSpeech, 2026; 分层标注+CoT+Agent 接口,尚无定量验证)
