---
type: concept
title: "Speaker Embedding"
aliases: [说话人嵌入, Speaker Representation, d-vector, Speaker Encoder, 说话人编码]
category: "representation"
tags: [TTS, multi-speaker, voice-cloning, speaker-identity, adaptive-TTS]
key_papers: ["[[论文笔记/Survey-Voice Cloning|Azzuni & El Saddik 2025]]", "[[论文笔记/Mega-TTS|Mega-TTS]]", "[[论文笔记/FireRedTTS|FireRedTTS]]", "[[论文笔记/YourTTS|YourTTS]]", "[[论文笔记/NaturalSpeech 2|NaturalSpeech 2]]", "[[论文笔记/Mega-TTS 2|Mega-TTS 2]]", "[[论文笔记/BASE TTS|BASE TTS]]", "[[论文笔记/WavLM|WavLM]]", "[[论文笔记/Seed-VC|Seed-VC]]", "[[论文笔记/PersonaPlex|PersonaPlex (Roy et al., 2026)]]", "[[论文笔记/MambaVoiceCloning|MambaVoiceCloning (2026)]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/NaturalVoices|NaturalVoices]]", "[[论文笔记/TTSDS2|TTSDS2]]", "[[论文笔记/VoXtream|VoXtream]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/IDEA-TTS|IDEA-TTS]]", "[[论文笔记/LiveSpeech 2|LiveSpeech 2]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/TTSDS|TTSDS]]", "[[论文笔记/Controlling Emotion TTS NL Prompts|Bott et al. (Interspeech 2024)]]", "[[论文笔记/Meta Learning TTS 7000 Languages|Meta Learning TTS 7000 Languages]]", "[[论文笔记/UMETTS|UMETTS]]", "[[论文笔记/STTATTS|STTATTS]]", "[[论文笔记/Very Attentive Tacotron|Very Attentive Tacotron (Battenberg et al., 2025)]]", "[[论文笔记/XTTS|XTTS]]", "[[论文笔记/EmoSphere++|EmoSphere++]]", "[[论文笔记/DiffCSS|DiffCSS]]", "[[论文笔记/PROEMO|PROEMO]]", "[[论文笔记/SafeSpeech|SafeSpeech]]", "[[论文笔记/Low-Resource ForwardTacotron|Low-Resource ForwardTacotron (Kayyar et al., 2025)]]", "[[论文笔记/DMOSpeech|DMOSpeech]]", "[[论文笔记/FlexSpeech|FlexSpeech]]", "[[论文笔记/TTS-Transducer|TTS-Transducer]]", "[[论文笔记/DiVISe|DiVISe (Liu et al., 2025)]]", "[[论文笔记/BreezyVoice|BreezyVoice]]", "[[论文笔记/F5R-TTS|F5R-TTS]]", "[[论文笔记/IndexTTS|IndexTTS]]", "[[论文笔记/FaceSpeak|FaceSpeak]]", "[[论文笔记/CSP-FT|CSP-FT (Wang et al., 2026)]]", "[[论文笔记/MAE Style-Rich TTS|MAE Style-Rich TTS]]", "[[论文笔记/DiEmo-TTS|DiEmo-TTS]]", "[[论文笔记/GSA-TTS|GSA-TTS]]", "[[论文笔记/SpeechAccentLLM|SpeechAccentLLM]]", "[[论文笔记/MPE-TTS|MPE-TTS]]", "[[论文笔记/FMSD-TTS|FMSD-TTS]]", "[[论文笔记/DS-TTS|DS-TTS]]", "[[论文笔记/Revival with Voice|Revival with Voice]]", "[[论文笔记/DeepDubbing|DeepDubbing]]", "[[论文笔记/SpeechWeave|SpeechWeave]]", "[[论文笔记/Speaker Identity Unlearning|Speaker Identity Unlearning]]", "[[论文笔记/LatinX|LatinX]]", "[[论文笔记/MELA-TTS|MELA-TTS]]", "[[论文笔记/EmoSSLSphere|EmoSSLSphere]]", "[[论文笔记/MSR-Codec|MSR-Codec]]", "[[论文笔记/TMD-TTS|TMD-TTS]]", "[[论文笔记/HiStyle|HiStyle]]", "[[论文笔记/DialoSpeech|DialoSpeech]]", "[[论文笔记/ParsVoice|ParsVoice]]", "[[论文笔记/E2E-VGuard|E2E-VGuard]]", "[[论文笔记/UniVoice|UniVoice]]", "[[论文笔记/Audiobook-CC|Audiobook-CC]]"]
origin_paper: "Xu Tan et al., A Survey on Neural Speech Synthesis, 2021"
related_concepts: ["[[Speech Factorization]]", "[[Prosody Modeling]]", "[[Text-to-Speech Pipeline]]", "[[Speech Tokenizer]]", "[[Speaker Verification]]", "[[Voice Cloning Taxonomy]]", "[[Speaker Adaptation]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-02
---

## 定义

Speaker Embedding 是将说话人身份信息编码为固定维度向量的表示方法,用于在多说话人 TTS 系统中控制合成语音的音色特征。它回答 "谁来说 (who to say)" 的问题。

**获取方式**:
1. **Speaker lookup table**: 训练集中每个说话人一个可学习的 embedding (closed-set)
2. **Speaker encoder**: 从参考音频提取 speaker embedding (open-set / zero-shot)

## 两种范式

### Lookup Table (训练集内说话人)
- 每个说话人分配一个可学习的固定维度向量
- 训练时通过 speaker ID 索引
- 优势: 简单高效, 质量高
- 局限: 无法泛化到新说话人

### Speaker Encoder (零样本)
- 从参考音频提取说话人表示
- 训练: 通常在说话人验证任务上预训练
- 优势: 可泛化到未见说话人 (zero-shot)
- 局限: 表示可能不够精确, speaker similarity 低

**常见 speaker encoder 架构**:
- d-vector: DNN 在 speaker verification 上训练后取倒数第二层
- x-vector: TDNN-based, 统计池化层
- GE2E: Generalized end-to-end loss 训练的 encoder
- ECAPA-TDNN: 强 speaker verification 模型, 常用于 TTS

## 在 TTS 中的注入方式

Speaker embedding 注入 TTS 模型的常见方法:

| 方式 | 描述 | 代表工作 |
|------|------|----------|
| Concatenation | 拼接到 encoder/decoder 输入 | DeepVoice 2 |
| Addition | 加到 hidden states | Tacotron 2 multi-speaker |
| Conditional LayerNorm | 生成 LN 的 scale/bias | AdaSpeech |
| FiLM conditioning | $\gamma \cdot h + \beta$ | 各种现代模型 |
| Cross-attention | 作为 key/value | CosyVoice |
| Prefix/Prompt | 作为解码器前缀 | VALL-E, Seed-TTS |

## Adaptive TTS (语音适应)

Survey 定义的自适应 TTS 场景: 用少量目标说话人数据使源模型适应新声音。

### 适应策略

| 策略 | 数据需求 | 参数调整 | 代表工作 |
|------|----------|----------|----------|
| Few-data adaptation | 几分钟 ~ 几秒 | 全模型/部分模型 | Chen et al., Arik et al. |
| Few-parameter adaptation | 数十句 | 仅 speaker embedding / LN | AdaSpeech |
| Untranscribed data | 无转写语音 | 用 ASR 获取文本 | AdaSpeech 2 |
| Zero-shot adaptation | 仅参考音频 | 无微调 | DV3-Clone, SEA-TTS, SV-Tacotron |

### AdaSpeech 系列
- **AdaSpeech**: Conditional LayerNorm 从 speaker embedding 生成 scale/bias, 仅微调 LN 参数
- **AdaSpeech 2**: 利用 mel 重建 + latent alignment 适应无转写数据
- **AdaSpeech 3**: 从阅读风格适应到自发说话风格 (filled pauses, rhythm)

## Zero-shot Voice Cloning

不需要任何微调,仅通过参考音频实现声音克隆:

**传统方案** (Survey 时代):
- Speaker encoder (SV-Tacotron, SEA-TTS): 从参考音频提取 embedding
- 局限: 目标说话人与源说话人差异大时质量下降

**现代方案** (LLM-TTS 时代):
- VALL-E: 3秒 prompt → AR + NAR 生成, in-context learning
- CosyVoice: prompt 音频经 flow matching 提取 timbre
- Seed-TTS: self-distillation 增强 timbre disentanglement

## Voice Cloning Survey 视角 (Azzuni & El Saddik, 2025)

### SECS: Speaker Embedding 作为评估核心

Voice Cloning 综合 survey 揭示了 speaker embedding 在评估中的关键角色。Speaker Embedding Cosine Similarity (SECS) 是衡量 voice cloning 质量的核心指标:

$$\text{SECS} = \frac{E(\text{generated}) \cdot E(\text{reference})}{\|E(\text{generated})\| \cdot \|E(\text{reference})\|}$$

**重要发现**: SECS 结果高度依赖所选 speaker encoder。Survey Tables V/VI 显示同一 TTS 系统使用不同 encoder (x-vector vs GE2E vs ECAPA-TDNN) 报告的 SECS 值可差 0.1-0.3,限制了跨论文可比性。

### Speaker Encoder 在三类 Cloning 中的角色差异

| Cloning 类型 | Encoder 角色 | 训练/推理 | 冻结? |
|-------------|------------|---------|------|
| Speaker Adaptation | 提取初始 embedding 或验证质量 | 训练时为主 | 通常冻结 |
| Few-shot VC | 提取 representation + 验证 | 训练+推理 | 部分可训练 |
| Zero-shot VC | 核心组件: 推理时实时提取 identity | 推理时必须 | 通常冻结 (预训练) |

### 完整 Speaker Encoder 架构汇总

Survey 汇总的在 SECS 计算和 TTS 训练中常用的 speaker encoder:

| 架构 | 特点 | 典型用途 |
|------|------|---------|
| d-vector | DNN 倒数第二层输出 | 早期 multi-speaker TTS |
| x-vector | TDNN + 统计池化 | 标准 SV baseline |
| GE2E | 端到端 generalized loss | TTS 训练 (VStyclone, SC-GlowTTS) |
| ECAPA-TDNN | 通道注意力 + 传播聚合 | 当前最常用 SECS encoder |
| TitaNet-L | 1D 深度可分离卷积 | 高效推理 |
| WavLM | 大规模自监督预训练 | 跨领域迁移 |
| XLSR-53 | 跨语言自监督 | 多语言 TTS |
| CAM++ | 上下文感知掩码 | DINO-VITS 等 |
| H/ASP | 多层注意力统计池化 | YourTTS 多语言 |
| ResCNN | 残差 CNN | 双语 speaker embedding (Chen et al.) |

### Speaker Embedding 在跨语言 Voice Cloning 中的扩展

Survey Section IV.D 表明跨语言 voice cloning 对 speaker embedding 提出额外挑战:
- **Language-independent encoder**: Xin et al. 训练语言无关的 speaker encoder,生成的 embedding 用于单语多说话人 TTS
- **Bilingual embedding**: Chen et al. 用 ResCNN 建立英中双语 speaker embedding 网络
- **Speaker consistency loss**: Latent Filling 用 speaker embedding 一致性损失改善跨语言 (英→英, 韩→英) ZS-TTS

## 关键论文

- DeepVoice 2 (Arik et al., NIPS 2017): 首个 multi-speaker neural TTS (lookup table)
- DeepVoice 3 (Ping et al., ICLR 2018): 可扩展到数千说话人
- SV-Tacotron (Jia et al., NeurIPS 2018): speaker encoder 实现零样本 TTS
- SEA-TTS (Chen et al., ICLR 2019): sample efficient adaptive TTS
- AdaSpeech (Chen et al., ICLR 2021): conditional LN 高效适应
- VALL-E (Wang et al., 2023): in-context learning 重新定义零样本 TTS
- Cooper et al. (ICASSP 2020): 系统比较不同 speaker embedding 对 ZS-TTS 的影响 (LDE + angular softmax > x-vector)
- Desplanques et al. (ECAPA-TDNN, Interspeech 2020): 当前最常用 SECS encoder

## 相关概念

- [[Speech Factorization]]: 将 speaker 信息与 content/prosody 解耦
- [[Prosody Modeling]]: speaker embedding 编码音色, prosody 编码韵律
- [[Speech Tokenizer]]: 在 LLM-TTS 中,prompt token 部分取代 speaker embedding 的功能
- [[Conditional Flow Matching]]: 现代 TTS 中从 speaker prompt 恢复音色
- [[Speaker Verification]]: speaker encoder 在 SV 上预训练,提供 SECS 评估能力
- [[Voice Cloning Taxonomy]]: speaker embedding 在四类 cloning 方法中角色各异
- [[Speaker Adaptation]]: speaker embedding 是 adaptation 的核心调整对象

## 演进

Speaker ID one-hot (SPSS) → Learnable speaker embedding (DeepVoice 2, 2017) → Speaker encoder / d-vector (SV-Tacotron, 2018) → Conditional LayerNorm (AdaSpeech, 2021) → In-context prompt (VALL-E, 2023) → Self-distillation timbre disentanglement (Seed-TTS, 2024)
