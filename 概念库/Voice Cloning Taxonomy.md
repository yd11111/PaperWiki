---
type: concept
title: "Voice Cloning Taxonomy"
aliases: [语音克隆分类, Voice Cloning Classification, 声音克隆体系]
category: "taxonomy"
tags: [TTS, voice-cloning, speaker-adaptation, few-shot, zero-shot, multilingual, survey]
key_papers: ["[[论文笔记/Survey-Voice Cloning|Azzuni & El Saddik 2025]]", "[[论文笔记/Voxtral TTS|Voxtral TTS]]", "[[论文笔记/XTTS|XTTS]]", "[[论文笔记/SafeSpeech|SafeSpeech]]", "[[论文笔记/DS-TTS|DS-TTS]]", "[[论文笔记/Revival with Voice|Revival with Voice]]", "[[论文笔记/Speaker Identity Unlearning|Speaker Identity Unlearning]]", "[[论文笔记/LatinX|LatinX]]"]
origin_paper: "[[论文笔记/Survey-Voice Cloning|Azzuni & El Saddik 2025]]"
related_concepts: ["[[Speaker Adaptation]]", "[[Speaker Embedding]]", "[[Speech Factorization]]", "[[Zero-shot Speech Synthesis]]", "[[LLM-based TTS]]", "[[Speaker Verification]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Voice Cloning 是通过 TTS 系统复制特定说话人声音的技术。Azzuni & El Saddik (2025) 的综合性 survey 建立了标准化的分类框架,将 voice cloning 按数据需求和方法论差异分为四大类:

1. **Speaker Adaptation (SA)**: 微调 TTS 模型以复制目标说话人声音,需要有限的目标说话人数据
2. **Few-shot Voice Cloning (FS-TTS)**: SA 的特例,参考音频范围从几秒到最多 5 分钟
3. **Zero-shot Voice Cloning (ZS-TTS)**: 无需微调,使用 speaker encoder 从短音频推断说话人特征
4. **Multilingual Voice Cloning**: 跨语言维度的声音克隆,从双语到 50+ 语言

## 四类方法的对比

| 维度 | Speaker Adaptation | Few-shot VC | Zero-shot VC | Multilingual VC |
|------|-------------------|-------------|--------------|-----------------|
| 微调 | 是 (全/部分模型) | 是 (少量数据) | 否 | SA/FS/ZS 均可 |
| 参考数据量 | 数十句~几分钟 | 几秒~5分钟 | 仅推理时参考 | 取决于基础方法 |
| 推理时需要 | 无额外输入 | 无额外输入 | Speaker encoder + 参考音频 | + 语言标识/G2P |
| Speaker encoder | 可选 (训练用) | 可选 | 必须 | 通常需要 |
| 代表方法 | AdaSpeech, CLN | Attentron, Meta-TTS | VALL-E, NaturalSpeech 2 | YourTTS, VALL-E X |
| 主要优势 | 高保真 | 数据效率 | 即时部署 | 语言泛化 |
| 主要局限 | 需要微调时间 | 过拟合风险 | Speaker similarity 较低 | 发音/韵律一致性 |

## 各类方法的子课题 (Survey Fig. 3)

### Speaker Adaptation
- **Disentanglement**: Daft-Exprt (FiLM conditioning), TN-VQTTS (timbre-normalized VQ), adversarial speaker-prosody disentanglement
- **Speaker Representation & Verification**: d-vector, i-vector, speaker verification feedback constraint (Cai et al.), LDE-based verification
- **Handling Speech Variability**: GMVAE-Tacotron (style/accent/noise control), noisy speech transfer learning
- **Untranscribed Speech**: Semi-supervised ASR+TTS (Inoue et al.), VQ-VAE linguistic units (Zhang et al.), AdaSpeech 2 (mel reconstruction)
- **Parameter Efficiency**: 冻结特定模块 (character embedding, encoder), AdaVITS (iSTFT decoder), HyperTTS (hypernetwork)
- **Non-English**: VStyclone (Mandarin, GAN-based), prosodic features for Mandarin adaptation

### Few-shot Voice Cloning
- **Disentanglement**: Spoken content + voice factorization (Wang et al.), Attentron (fine-grained + coarse-grained encoder)
- **Speaker Representation**: Multi-speaker latent space (Deng et al.), GC-TTS (geometric constraints), GAN-based initial embedding (Lee et al.)
- **Optimization-based**: Meta-learning (MAML), Bayesian optimization (BOFFIN TTS), Meta-StyleSpeech (SALN + meta-learning)
- **Parameter Efficiency**: Voiceloop (shifting buffer), residual adapters (Morioka et al.), structured pruning (Huang et al.), DiT (adaptive LN)

### Zero-shot Voice Cloning
- **Disentanglement**: GenerSpeech (multi-level style adapter + MSLN), GZS-TV (disentangled representation), NaturalSpeech 3 (factorized diffusion), Mega-TTS (content/timbre/prosody/phase)
- **Emotion Transfer**: StyleTTS (AdaIN + style tokens), StyleTTS 2 (style diffusion + adversarial), DINO-VITS (self-supervised + emotion classifier)
- **Generalizable Speech Synthesis**: Glow-WaveGAN 2 (ZS-TTS + any-to-any VC), VoiceCraft (token-infilling NCLM), NaturalSpeech 2 (diffusion + RVQ)
- **Codec-based**: SPEAR-TTS (semantic → acoustic), VecTok Speech (speech codec + BPE), TacoLM (gated attention), MaskGCT (masked generative codec transformer)

### Multilingual Voice Cloning
- **2 languages**: Cross-lingual speaker adaptation (Xin et al.), bilingual speaker embedding (Chen et al.), VALL-E X
- **3-5 languages**: YourTTS (VITS + H/ASP encoder), Parrot-TTS (SSL + HuBERT), STEN-TTS (diffusion + STEN)
- **5+ languages**: Universal seq2seq (Yang et al.), LAML (articulatory features), CLAM-TTS (probabilistic RVQ), XTTS (VQ-VAE + GPT-2)

## 历史演进 (Survey Fig. 1 趋势)

Survey 的论文趋势图 (Fig. 1) 显示:
- 2015-2017: Speaker Adaptation 主导
- 2018-2019: Voice Cloning 和 Few-shot TTS 兴起
- 2020-2021: Zero-shot TTS 快速增长
- 2022-2024: Zero-shot TTS 论文量最高,codec-based 架构成为主流

**技术路线演进**:
```
Speaker LUT (DeepVoice 2, 2017)
  → Speaker Encoder (SV-Tacotron, 2018)
    → CLN-based adaptation (AdaSpeech, 2021)
      → Codec LM in-context (VALL-E, 2023)
        → Diffusion/Flow hybrid (NaturalSpeech 2/3, 2024)
          → Masked generative (MaskGCT, 2025)
```

## 在 TTS 中的应用

- **Entertainment**: 动画配音自动化、电影 dubbing
- **Personal Assistants**: 个性化虚拟助手声音、车载语音定制 (Cerence "My Car, My Voice")
- **Accessibility**: 失语症患者声音恢复 (dysphonia)、语音障碍辅助
- **Advertisement**: 品牌声音为不同受众定制
- **Misuse risks**: 身份冒充、金融诈骗 (2019 年 CEO 深伪语音案, $243K)、数据清洗

## 关键论文

- [[论文笔记/Survey-Voice Cloning|Azzuni & El Saddik (2025)]]: 建立四分类体系的综合性 survey
- DeepVoice 2/3 (2017/2018): 多说话人 TTS 先驱
- SV-Tacotron (Jia et al., NeurIPS 2018): Speaker encoder 开启 zero-shot TTS
- AdaSpeech (Chen et al., ICLR 2021): CLN 高效适应范式
- VALL-E (Wang et al., 2023): Codec LM 重新定义 zero-shot TTS

## 相关概念

- [[Speaker Adaptation]]: 四分类中的基础类别
- [[Speaker Embedding]]: 所有 cloning 方法的核心表示
- [[Speaker Verification]]: 评估 speaker similarity 的关键工具
- [[Speech Factorization]]: 贯穿所有 cloning 类别的核心子课题
- [[LLM-based TTS]]: Zero-shot VC 的主流现代范式
- [[Anti-spoofing and Deepfake Detection]]: Voice cloning 的伦理对偶

## 演进

Concatenative TTS (无克隆) → Speaker LUT multi-speaker (DeepVoice 2, 2017) → Speaker encoder zero-shot (2018) → CLN/meta-learning few-shot (2021) → Codec LM in-context zero-shot (VALL-E, 2023) → Factorized diffusion/flow (NaturalSpeech 3, 2024) → Masked generative (MaskGCT, 2025) → 50+ language multilingual (2024)
