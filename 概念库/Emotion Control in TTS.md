---
type: concept
title: "Emotion Control in TTS"
aliases: [情感控制TTS, Emotional TTS, Emotion Synthesis, 情感语音合成, Affective Speech Synthesis]
category: "technique"
tags: [TTS, emotion, expressiveness, control, affective-computing, style]
key_papers: ["Li et al. (2021)", "MsEmoTTS (Lei et al., 2022)", "Emo-DPO (Gao et al., 2024)", "EmoSphere++ (Cho et al., 2024)", "Rong et al. (2025)", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/Step-Audio 2.5|StepAudio 2.5]]", "[[论文笔记/SC VALL-E|SC VALL-E]]", "[[论文笔记/NVSpeech|NVSpeech]]", "[[论文笔记/FlexiVoice|FlexiVoice]]", "[[论文笔记/EmotionThinker|EmotionThinker]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/TextrolSpeech|TextrolSpeech]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/NaturalVoices|NaturalVoices]]", "[[论文笔记/SpeechWorldModel|SpeechWorldModel]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/TTS-PRISM|TTS-PRISM]]", "[[论文笔记/Llama-VITS|Llama-VITS]]", "[[论文笔记/EmoSphere-TTS|EmoSphere-TTS]]", "[[论文笔记/Daisy-TTS|Daisy-TTS]]", "[[论文笔记/Controlling Emotion TTS NL Prompts|Bott et al. (Interspeech 2024)]]", "[[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]", "[[论文笔记/StoryTTS|StoryTTS]]", "[[论文笔记/UMETTS|UMETTS]]", "[[论文笔记/EmoSphere++|EmoSphere++]]", "[[论文笔记/AutoStyle-TTS|AutoStyle-TTS]]", "[[论文笔记/PROEMO|PROEMO]]", "[[论文笔记/EmoVoice|EmoVoice]]", "[[论文笔记/OpenOmni|OpenOmni]]", "[[论文笔记/CSP-FT|CSP-FT (Wang et al., 2026)]]", "[[论文笔记/DialogueAgents|DialogueAgents]]"]
origin_paper: "Xie et al., Controllable TTS in LLM Era, 2024"
related_concepts: ["[[Prosody Modeling]]", "[[Style Transfer in TTS]]", "[[Speech Factorization]]", "[[Differentiable Reward Optimization]]", "[[LLM-based TTS]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Emotion Control in TTS 旨在使合成语音表达特定的情感状态 (affective state),如快乐、悲伤、愤怒、恐惧、惊讶、厌恶等。这是可控 TTS 的核心任务之一,直接影响人机交互的自然度、叙事表现力和虚拟助手的情感智能。

**核心挑战** (Survey 指出):
- 情感与其他语音属性 (timbre, prosody) 深度纠缠
- 情感跨越多个粒度 (utterance-level, word-level, phoneme-level)
- 细粒度情感 (如讽刺) 难以标注和建模
- 长语音中保持情感一致性

## 情感建模方法

### 1. Emotion Embedding (情感嵌入)

直接学习情感的向量表示:
- **One-hot emotion label**: 离散情感类别 (angry, happy, sad, neutral)
- **Learned emotion embedding**: 从标注数据学习连续情感向量
- **Style alignment**: Li et al. (2021) 引入 emotion embeddings + style alignment 调制情感强度

### 2. 层级情感建模 (Hierarchical)

MsEmoTTS (Lei et al., 2022) 的多尺度方法:
- **Global level**: 整句情感基调
- **Utterance level**: 句子内情感变化
- **Local level**: 词/音素级情感细节
- 分层结构捕捉情感在不同粒度的表现

### 3. 对抗训练解耦 (Adversarial Disentanglement)

将情感与说话人身份分离:
- Cross-speaker emotion transfer (Li et al., 2022): 对抗训练消除 speaker 信息
- 结果: 可将一个说话人的情感迁移到另一个说话人声音中

### 4. DPO/RLHF 优化

Emo-DPO (Gao et al., 2024):
- 使用 Direct Preference Optimization 实现情感控制
- 将 LLM 对齐技术引入 TTS 情感调优
- 不需要显式情感标签,从偏好数据学习

### 5. Emotion-adaptive 表示

EmoSphere++ (Cho et al., 2024):
- Emotion-adaptive spherical vector 表示
- 零样本情感可控 TTS
- 球面空间建模情感分布

## 与其他控制维度的交互

Survey 特别指出情感控制的难点:

> "Emotion and other vocal traits are often intertwined and span multiple granularities, making fine-grained control especially difficult."

**纠缠问题**:
- 修改 pitch 可能同时影响情感和自然度
- 情感变化伴随 duration、energy、voice quality 的联动
- 同一情感在不同说话人中表现不同

**解耦方案**:
- 对抗训练: 分类器 + gradient reversal
- Information bottleneck: 独立分支编码 emotion vs speaker vs content
- Pre-trained model guidance: 预训练情感分类器引导特征分离 (An et al., 2022; Wang et al., 2023b)

## 数据集

| 数据集 | 语言 | 情感类别 | 特点 |
|--------|------|----------|------|
| IEMOCAP (Busso et al., 2008) | EN | 多类 | 交互式, 多模态 |
| RAVDESS (Livingstone & Russo, 2018) | EN | 8类 | 演员表演, 歌曲+语音 |
| RECOLA (Ringeval et al., 2013) | FR | 连续维度 | 协作互动 |
| Toloka (2024) | Multi | 多类 | 众包标注 |
| ESD | ZH/EN | 5类 | 平行语料 |

## 评估方法

- **Emotion accuracy**: 情感分类器对合成语音的识别准确率
- **MOS (expressiveness)**: 人类评估情感表现力
- **AB/ABX test**: 对比不同系统的情感表达质量
- **GPT-based evaluation** (Rong et al., 2025): 使用 LLM 评估情感一致性

## 在 TTS 中的应用

- 虚拟助手: 情感智能对话
- 有声书/影视: 角色情感表演
- 心理健康: 共情语音生成
- 游戏 NPC: 动态情感反应

## 关键论文

- Li et al. (2021): emotion embeddings + style alignment 调制强度
- MsEmoTTS (Lei et al., 2022): 多尺度情感迁移、预测与控制
- Li et al. (2022): 跨说话人情感解耦与迁移
- Emo-DPO (Gao et al., 2024): DPO 优化情感合成
- [[论文笔记/EmoSphere-TTS|EmoSphere-TTS]] (Cho et al., Interspeech 2024): AVD 伪标签 + 笛卡尔→球面坐标变换,解耦情感风格/强度控制
- EmoSphere++ (Cho et al., 2024): 球面向量零样本情感控制 (EmoSphere-TTS 扩展版)
- EmoVoice (Yang et al., 2025): LLM-based 自由文本情感提示

## 相关概念

- [[Prosody Modeling]]: 情感通过韵律变化实现
- [[Style Transfer in TTS]]: 情感是风格的子维度
- [[Speech Factorization]]: 情感与 speaker/content 解耦
- [[Differentiable Reward Optimization]]: Emo-DPO 的技术基础
- [[LLM-based TTS]]: 新范式下的情感控制方式

## 帧级 Arousal-Valence 条件控制 (EmoCtrl-TTS)

EmoCtrl-TTS (Wu et al., 2024) 在 flow-matching zero-shot TTS 上同时使用两组帧级条件: (1) arousal-valence 值 (来自 wav2vec 2.0-based extractor, chunk-wise 0.5s/0.25s) 控制时变情感; (2) laughter detector embedding (32 维) 控制 NV (笑声、哭泣等)。用 27k 小时伪标签真实情感数据训练,在 JVNV S2ST 上 Aro-Val SIM 0.643 (超越 ELaTE 0.548)。关键发现: laughter detector embedding 能泛化到哭泣等非笑声 NV; 两种 embedding 在某些数据上存在负面交互,需按数据源选择性启用。与 NVSpeech 的离散 PV 标签方法互补: EmoCtrl-TTS 用连续 embedding 实现帧级控制,NVSpeech 用离散标签实现 token-level 控制。详见 [[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]。

## 副语言发声方法 (NVSpeech)

NVSpeech (Liao et al., 2025) 从不同角度切入情感表达 — 不直接建模抽象情感状态,而是建模具体的副语言行为 (笑声、叹气、犹豫等),这些行为是情感的外在表现。通过在文本中显式插入 `[Laughter]`、`[Breathing]` 等标签实现 token-level 控制。与传统情感控制互补: 情感控制提供高层意图,PV 控制提供底层行为实现。详见 [[论文笔记/NVSpeech|NVSpeech]]。

## 演进

规则情感合成 (HMM, 2003) → Emotion embedding (2021) → 多尺度层级建模 (MsEmoTTS, 2022) → 跨说话人情感迁移 (2022) → 韵律嵌入分解 (Daisy-TTS, 2024) → DPO/RLHF 对齐 (Emo-DPO, 2024) → 零样本情感 (EmoSphere++, 2024) → LLM 自由文本情感 (EmoVoice, 2025) → 副语言行为控制 (NVSpeech, 2025)

## Plutchik 结构模型与韵律嵌入分解 (Daisy-TTS)

[[论文笔记/Daisy-TTS|Daisy-TTS]] (Chevi & Aji, 2024) 从 Plutchik 结构模型出发,提出韵律嵌入分解方法实现更宽广的情感模拟。核心思路: 用 emotion discriminator 训练 prosody encoder 学习情感可分离嵌入,再通过 PCA 分解实现四种情感操控 — 一级情感 (采样)、二级情感 (高斯混合)、强度 (缩放因子 alpha)、极性 (取反)。在 ESD 数据集上 MOS 和感知率均优于 Zhou et al. (2022b) baseline。该方法是 "情感表示 = 可分解韵律原型" 范式的首次探索。
