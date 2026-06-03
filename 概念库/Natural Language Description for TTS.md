---
type: concept
title: "Natural Language Description for TTS"
aliases: [文本描述语音合成, Description-based TTS, Text-prompted TTS, Prompt-based Style Control, 自然语言风格描述]
category: "technique"
tags: [TTS, controllability, natural-language, description, prompt, text-guided]
key_papers: ["PromptTTS (Guo et al., 2023)", "InstructTTS (Yang et al., 2024b)", "PromptStyle (Liu et al., 2023a)", "NansyTTS (Yamamoto et al., 2024)", "PromptTTS++ (Shimizu et al., 2024)", "PromptTTS 2 (Leng et al., 2023)", "FleSpeech (Li et al., 2025a)", "Parler-TTS (Lyth and King, 2024)", "[[论文笔记/TextrolSpeech|TextrolSpeech]]", "[[论文笔记/Controlling Emotion TTS NL Prompts|Bott et al. (Interspeech 2024)]]", "[[论文笔记/StoryTTS|StoryTTS]]", "[[论文笔记/UMETTS|UMETTS]]", "[[论文笔记/EmoVoice|EmoVoice]]"]
origin_paper: "Xie et al., Controllable TTS in LLM Era, 2024"
related_concepts: ["[[Style Transfer in TTS]]", "[[LLM-based TTS]]", "[[Instruction-Guided Speech Synthesis]]", "[[Speaker Embedding]]", "[[Prosody Modeling]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Natural Language Description for TTS 是一种通过自然语言文本描述来控制合成语音属性的技术范式。用户以自由文本形式指定目标语音的特征 (如 "一个年轻女性用温柔的声音缓慢说话"),系统据此生成符合描述的语音。

**与其他控制方式的区分**:
- **Style Tagging**: 离散标签 (happy, fast) → 有限组合
- **Reference Speech Prompt**: 需要参考音频 → 不总是可得
- **NL Description**: 自由文本描述 → 最直观、最灵活
- **Instruction-Guided**: 统一内容+风格为指令 → 更进一步

## 核心方法

### PromptTTS (Guo et al., 2023)
**开创性工作**: 第一个用文本描述控制 TTS 的系统
- 手动标注 5 个语音属性的文本描述: gender, pitch, speaking speed, volume, emotion
- 模型学习从文本描述中提取语音属性嵌入
- 局限: 描述模板化, 属性有限

### InstructTTS (Yang et al., 2024b)
**三阶段训练**: 从自然语言提取语义用于 TTS 控制
1. Stage 1: 语音-文本对预训练
2. Stage 2: 描述理解模块训练
3. Stage 3: 端到端联合优化
- 支持更自由的描述格式

### PromptTTS++ (Shimizu et al., 2024)
通过额外 speaker description prompts 增强提示丰富度
- 超越简单属性描述
- 支持复杂说话人特征描述

### PromptTTS 2 (Leng et al., 2023)
引入 variation network 建模 prompt 之外的残差变化
- 解决文本描述不完整导致的生成不稳定
- 对同一描述可生成多样化语音

### NansyTTS (Yamamoto et al., 2024)
跨语言描述控制:
- 描述控制器 (description controller) 与 TTS 可用不同语言训练
- 通过共享 timbre 和 style 表示实现跨语言泛化

### PromptStyle (Liu et al., 2023a)
可控风格迁移的文本描述方案

### FleSpeech (Li et al., 2025a)
灵活可控语音生成,支持多种提示格式

### Parler-TTS (Lyth and King, 2024)
合成标注 + 大规模训练的高保真描述 TTS

## 扩展方向

### 环境感知 (Environmental Context)
- VoiceLDM (Lee et al., 2024): 内容提示 → 环境音合成
- AST-LDM (Kim et al., 2024b): 扩展 AudioLDM 实现环境条件化
- MS2KU-VTTS (He et al., 2024): 混合环境图像到 prompt,沉浸式语音

### Speaker 描述增强
- PromptSpeaker (Zhang et al., 2023c): 说话人特征的文本描述
- ProEmo (Zhang et al., 2025a): 情感的自然语言描述

## 数据集需求

Survey 指出 description-based datasets 的特点:
- 配对: 语音样本 + 丰富自由文本描述
- 属性: 语调、韵律、说话风格、情感色调
- 代表: SpeechCraft (Jin et al., 2024), Parler-TTS annotations (Lyth and King, 2024)
- 区别于 tag-based datasets: 不限于预定义类别标签

## 技术挑战

1. **描述的模糊性**: 同一描述可对应多种合理语音
2. **描述的不完整性**: 文本无法完全捕捉所有语音特征 (PromptTTS 2 的 variation network 解决)
3. **评估困难**: 描述与语音的匹配度难以自动量化
4. **训练数据**: 高质量描述标注成本高,需要 LLM 辅助生成

## 在 TTS 中的应用

- 创意内容制作: 导演用文字描述角色声音
- 无障碍界面: 视障用户通过文本自定义 TTS 声音
- 虚拟形象: 用文本描述定义角色声音特征
- 批量生成: 无需收集参考音频即可指定多种风格

## 关键论文

- PromptTTS (Guo et al., ICASSP 2023): 开创文本描述控制 TTS
- PromptTTS 2 (Leng et al., ICLR 2023): variation network 增强多样性
- InstructTTS (Yang et al., 2024b): 三阶段从 NL 提取语义
- NansyTTS (Yamamoto et al., 2024): 跨语言描述控制
- Parler-TTS (Lyth and King, 2024): 大规模合成标注 + 高保真

## 相关概念

- [[Style Transfer in TTS]]: NL description 是风格控制的一种策略
- [[LLM-based TTS]]: LLM 使 NL description 理解更强
- [[Instruction-Guided Speech Synthesis]]: 更进一步统一内容与描述
- [[Speaker Embedding]]: NL description 可替代显式 speaker embedding
- [[Prosody Modeling]]: 描述中包含韵律指令

## 演进

Style tagging (GST, 离散标签, 2018) → Reference encoder (从音频提取, 2018-2022) → PromptTTS (文本描述5属性, 2023) → PromptTTS 2 (variation network, 2023) → InstructTTS (三阶段NL理解, 2024) → Parler-TTS (大规模合成标注, 2024) → FleSpeech (灵活多提示, 2025)
