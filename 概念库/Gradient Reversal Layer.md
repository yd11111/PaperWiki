---
type: concept
title: "Gradient Reversal Layer"
aliases: [GRL, 梯度反转层, Domain-Adversarial Training]
category: "training-technique"
tags: [adversarial-training, disentanglement, domain-adaptation, TTS]
key_papers: ["[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/NaturalSpeech 3|NaturalSpeech 3]]", "[[论文笔记/EmoSphere++|EmoSphere++]]", "[[论文笔记/FaceSpeak|FaceSpeak]]"]
origin_paper: ""
related_concepts: ["[[Speech Tokenizer]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Gradient Reversal Layer (GRL) 是一种对抗训练技术,前向传播时为恒等函数(直接传递特征),反向传播时将梯度取反(乘以 -1)。通过在特征提取器与判别器之间插入 GRL,可以迫使特征提取器学到对判别器任务不变的表征,从而实现特征解耦。

原始提出于 domain adaptation 场景(Ganin et al., 2016): 让 feature extractor 提取 domain-invariant 的特征。

## 在 TTS 中的应用

在 IndexTTS2 中,GRL 用于 emotion-speaker disentanglement:
- Emotion perceiver conditioner 提取情感特征 e
- GRL 连接 e 到 speaker classifier
- 训练时: classifier 试图从 e 预测说话人身份,但 GRL 反转梯度使 emotion perceiver 被训练为 **无法** 编码说话人信息
- 结果: e 只包含情感/韵律属性,不含音色信息

联合损失函数: L_AR = -(1/(T+1)) * Σ log q(y_t) - α * log q(e),其中第二项为 GRL 驱动的对抗损失。

## 关键论文

- Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016 — 首次提出 GRL
- NaturalSpeech 3 (Ju et al., 2024): 在 factorized codec 中使用类似的解耦策略
- IndexTTS2 (Zhou et al., 2025): 在 TTS 中用 GRL 实现情感-音色正交化

## 相关概念

- Adversarial Training: GRL 的上位概念
- Contrastive Learning: 另一种特征解耦手段
- [[Speech Tokenizer]]: GRL 解耦的特征最终服务于 token 生成

## 演进

Domain Adaptation GRL (2016) → Style-Content Disentanglement → Emotion-Speaker Disentanglement in TTS (IndexTTS2, 2025)
