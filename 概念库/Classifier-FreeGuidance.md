---
type: concept
title: "Classifier-Free Guidance"
aliases: [CFG, 无分类器引导, Classifier-free Diffusion Guidance]
category: "training-technique"
tags: [diffusion, guidance, conditional-generation, TTS, audio-generation]
key_papers: ["[[论文笔记/CosyVoice|CosyVoice]]", "[[论文笔记/CosyVoice2|CosyVoice 2]]", "[[论文笔记/NaturalSpeech3|NaturalSpeech 3]]", "[[论文笔记/Survey-AudioDiffusionModels|Survey-Audio Diffusion Models]]", "[[论文笔记/TortoiseTTS|Tortoise TTS]]", "[[论文笔记/FELLE|FELLE]]", "[[论文笔记/LatentLM|LatentLM]]", "[[论文笔记/CLEAR|CLEAR]]", "[[论文笔记/VibeVoice|VibeVoice]]", "[[论文笔记/SiTok|SiTok]]", "[[论文笔记/Chatterbox-Flash|Chatterbox-Flash]]", "[[论文笔记/VoxtralTTS|Voxtral TTS]]", "[[论文笔记/DiSTAR|DiSTAR]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/E2TTS|E2 TTS]]", "[[论文笔记/SemaVoice|SemaVoice]]", "[[论文笔记/TADA|TADA]]", "[[论文笔记/OmniVoice|OmniVoice]]", "[[论文笔记/SESD|SESD]]", "[[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]", "[[论文笔记/FlexSpeech|FlexSpeech]]", "[[论文笔记/Koel-TTS|Koel-TTS]]", "[[论文笔记/DMOSpeech|DMOSpeech]]", "[[论文笔记/TechSinger|TechSinger]]", "[[论文笔记/MAEStyle-RichTTS|MAE Style-Rich TTS]]", "[[论文笔记/CapSpeech|CapSpeech]]", "[[论文笔记/ZipVoice|ZipVoice]]", "[[论文笔记/ShallowFlowMatching|Shallow Flow Matching]]", "[[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning]]", "[[论文笔记/DeepDubbing|DeepDubbing]]", "[[论文笔记/DAIEN-TTS|DAIEN-TTS]]", "[[论文笔记/MELA-TTS|MELA-TTS]]", "[[论文笔记/ZipVoice-Dialog|ZipVoice-Dialog]]", "[[论文笔记/UniVoice|UniVoice]]", "[[论文笔记/Align2Speak|Align2Speak]]", "[[论文笔记/DashengAudioGen|Dasheng AudioGen]]"]
origin_paper: "Ho & Salimans, Classifier-Free Diffusion Guidance, 2022"
related_concepts: ["[[DiffusionModel]]", "[[Diffusion-basedTTS]]", "[[ConditionalFlowMatching]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Classifier-Free Guidance (CFG) 是一种不需要额外训练分类器即可实现条件引导的 diffusion 采样方法。其核心思想是在训练时同时学习条件模型和无条件模型,推理时通过线性外推增强条件信号的影响。

### 工作原理

**训练阶段**: 以一定概率 (如 10-20%) 随机丢弃条件信息 c,使模型同时学会:
- 条件去噪: epsilon_theta(x_t, t, c) — 给定条件 c 去噪
- 无条件去噪: epsilon_theta(x_t, t, null) — 不给条件去噪

**推理阶段**: 使用引导强度 w 进行外推:
```
epsilon_guided = (1 + w) * epsilon_theta(x_t, t, c) - w * epsilon_theta(x_t, t, null)
```

当 w = 0 时退化为标准条件生成; w > 0 时增强条件信号(如文本、说话人身份)的影响,w 越大生成越忠实于条件但多样性降低。

### 与 Classifier Guidance 的对比

| 维度 | Classifier Guidance | Classifier-Free Guidance |
|------|---------------------|--------------------------|
| 额外模型 | 需训练噪声 classifier | 不需要 |
| 训练方式 | 分离训练 diffusion + classifier | 单一模型,条件随机 dropout |
| 梯度来源 | classifier 对 class label 的梯度 | 条件/无条件输出的差值 |
| 灵活性 | 受限于 classifier 设计 | 任意条件类型 |
| 主流程度 | 早期方法,已被取代 | 当前主流 |

## 在 TTS 中的应用

### Guided-TTS 2 (Kim et al., 2022)

Zhang et al. (2023) 综述指出,Guided-TTS 2 [38] 是 CFG 在 TTS 中的典型应用 [§3.2.3]:
- 使用 speaker-conditional DDPM 替代 Guided-TTS 的无条件模型
- 通过 classifier-free guidance 将预训练的 diffusion model 适配到目标说话人
- 仅需短参考语音即可实现零样本多说话人 TTS

### EmoDiff (Guo et al., 2022)

EmoDiff [22] 使用类似 classifier guidance 的方式控制情感 [§3.2.5]:
- 先训练无条件声学 diffusion model
- 再训练情感分类器引导 diffusion trajectory
- 用软标签实现连续情感强度控制

### 在音频生成中的广泛应用

CFG 已成为 diffusion/flow-based 音频生成的标准技术:
- 文本到音频 (AudioLDM, Stable Audio)
- 文本到音乐 (MusicLM)
- 语音合成 (SoundStorm, VoiceBox)
- 在 [[ConditionalFlowMatching]] 中同样适用 (如 Matcha-TTS)

### 离散空间 CFG

OmniVoice (Zhu et al., 2026) 将 CFG 从连续空间扩展到离散 token 的 log-softmax 空间。推理时 batch 翻倍 (2*B),前 B 个为 conditional,后 B 个为 unconditional (仅含 masked target,无 text/style/prompt)。引导公式在 log-softmax 空间操作:

```
log_probs = log_softmax(c_log_probs + scale * (c_log_probs - u_log_probs))
```

注意外层还有一个 log_softmax -- 即双重 softmax normalize。guidance_scale 默认 2.0。这表明 CFG 原理不限于连续扩散/flow,同样适用于离散 masked generative modeling。

## 关键论文

- Ho & Salimans (2022): Classifier-Free Diffusion Guidance — 原始论文
- Dhariwal & Nichol (2021): Diffusion Models Beat GANs — classifier guidance 原始工作
- Guided-TTS 2 (Kim et al., 2022): CFG 用于零样本多说话人 TTS [§3.2.3]
- EmoDiff (Guo et al., 2022): 情感控制的 diffusion guidance [§3.2.5]

## 相关概念

- [[DiffusionModel]]: CFG 是 diffusion 条件生成的主流方法
- [[Diffusion-basedTTS]]: CFG 在 TTS diffusion 系统中广泛使用
- [[ConditionalFlowMatching]]: CFG 同样适用于 flow matching 框架
- [[SpeakerEmbedding]]: CFG 可用于说话人条件引导

## 演进

Conditional Diffusion (直接输入条件, 2020) --> Classifier Guidance (Dhariwal & Nichol, 2021, 需额外分类器) --> Classifier-Free Guidance (Ho & Salimans, 2022, 不需额外模型) --> 成为 diffusion/flow 条件生成标准 --> 在 TTS (Guided-TTS 2) / 音频 / 图像生成中广泛采用
