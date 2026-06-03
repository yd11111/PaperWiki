---
type: concept
title: "Emotion Control in TTS"
aliases: [情感控制TTS, Emotional TTS, Emotion Synthesis, 情感语音合成, Affective Speech Synthesis]
category: "technique"
tags: [TTS, emotion, expressiveness, control, affective-computing, style]
key_papers: ["Li et al. (2021)", "MsEmoTTS (Lei et al., 2022)", "Emo-DPO (Gao et al., 2024)", "EmoSphere++ (Cho et al., 2024)", "Rong et al. (2025)", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/Step-Audio 2.5|StepAudio 2.5]]", "[[论文笔记/SC VALL-E|SC VALL-E]]"]
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
- EmoSphere++ (Cho et al., 2024): 球面向量零样本情感控制
- EmoVoice (Yang et al., 2025): LLM-based 自由文本情感提示

## 相关概念

- [[Prosody Modeling]]: 情感通过韵律变化实现
- [[Style Transfer in TTS]]: 情感是风格的子维度
- [[Speech Factorization]]: 情感与 speaker/content 解耦
- [[Differentiable Reward Optimization]]: Emo-DPO 的技术基础
- [[LLM-based TTS]]: 新范式下的情感控制方式

## 演进

规则情感合成 (HMM, 2003) → Emotion embedding (2021) → 多尺度层级建模 (MsEmoTTS, 2022) → 跨说话人情感迁移 (2022) → DPO/RLHF 对齐 (Emo-DPO, 2024) → 零样本情感 (EmoSphere++, 2024) → LLM 自由文本情感 (EmoVoice, 2025)
