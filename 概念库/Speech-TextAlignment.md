---
type: concept
title: "Speech-Text Alignment"
aliases: [语音文本对齐, Modality Alignment, Speech-Text Representation Alignment, 跨模态对齐, Text-Speech Alignment]
category: "technique"
tags: [speech-LM, alignment, multimodal, representation, training-strategy]
key_papers: ["SPIRIT-LM (Nguyen et al., 2024)", "Spectron (Nachmani et al., 2024)", "SpeechGPT (Zhang et al., 2023)", "Mini-Omni (Xie & Wu, 2024)", "Moshi (Defossez et al., 2024)", "Llama-Omni (Fang et al., 2024)", "Align-SLM (2024)", "SpeechAlign (2024)", "Yang et al., When LLM Meet Speech, 2025", "[[论文笔记/TADA|TADA]]", "[[论文笔记/STTATTS|STTATTS]]", "[[论文笔记/ZipVoice-Dialog|ZipVoice-Dialog]]", "[[论文笔记/MAVE|MAVE]]"]
origin_paper: "[[论文笔记/Survey-SpeechLanguageModels|Cui et al., Speech Language Models, 2024]]"
related_concepts: ["[[SpeechLanguageModel]]", "[[SemanticvsAcousticTokens]]", "[[SpeechTokenizer]]", "[[Full-duplexSpokenDialogue]]", "[[ModalityAdaptationforSpeechLLM]]", "[[Speech-LLMIntegrationTaxonomy]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Speech-Text Alignment 是 SpeechLM 中将语音和文本两种模态的表征进行对齐的技术,使模型能够在统一的表征空间中理解和生成两种模态。这是 SpeechLM 从 TextLM 继承语言知识、实现跨模态能力的关键。

Survey (Cui et al., 2024) 指出: "The primary goal of aligning text and speech representations is to leverage the strengths of text-based models to enhance speech-based models."

**核心挑战**: 训练 SpeechLM 显著难于训练 TextLM,因为:
- 文本是浓缩的知识形式,而语音需要模型独立学习口语规则
- 对齐有效但存在 trade-off: 文本对齐可增强语义但可能削弱副语言特征捕获

## 四种 Speech-Text Token 建模方式

Survey (Table IV) 归纳了四种在模型中组织 speech 和 text token 的方法:

### 1. Speech-only (仅语音)
```
[SPEECH] S12 S34 S33 ... S11 S59
```
- 仅提供语音序列,不含文本
- 用于纯语音生成/理解任务

### 2. Text-only (仅文本)
```
[TEXT] A quick brown fox jumps over a lazy dog.
```
- 仅提供文本序列
- 保留 TextLM 的原始文本能力

### 3. Concatenated speech-text (拼接)
```
[SPEECH] S12 S34 S33 ... S11 S59 [TEXT] A quick brown fox jumps over a lazy dog.
```
- 语音和文本序列简单拼接
- 模型学习 speech→text 或 text→speech 的映射

### 4. Alternating speech-text (交替)
```
[SPEECH] S12 S34 S33 [TEXT] brown fox jumps over [SPEECH] S11 S59
```
- 语音和文本 token 交替排列
- **SUTLM 实验表明此方式在跨模态评估中表现最佳**
- SPIRIT-LM 在 TextLM checkpoint 上用交替序列继续训练

## 对齐维度

### 单序列对齐 (Single-sequence)
将 text 和 speech token 放入同一个序列:
- **SPIRIT-LM**: 在 OPT checkpoint 上使用交替 text-speech tokens 继续训练
- **Spectron**: 将输入语音先转写为 text tokens,再预测 text token response,最终合成输出语音
- **SpeechGPT**: instruction-tuning 阶段将 speech-text 对拼接

**优势**: 强制模型在同一注意力窗口内学习两种模态的对应关系

### 多序列对齐 (Multi-sequence)
同时生成 text 和 speech 的多个序列:
- **Llama-Omni**: LLM 输出 hidden state 同时解码 text tokens 和 discrete speech tokens
- **Mini-Omni**: 同时生成 1 个 text token 序列 + 7 个 acoustic token 序列,按句子级别对齐
- **Moshi**: 生成 1 个 text 序列 + 1 个 semantic 序列 + 7 个 acoustic 序列,按 word level 对齐

**优势**: 并行生成多流,效率更高

## 推理模式

对齐后的模型有两种推理方式:

### Text-present Inference
推理时同时解码 text 和 speech,文本充当"锚":
- **优势**: 减少幻觉,增强推理能力 (Mini-Omni)
- **劣势**: 增加延迟

### Text-independent Inference
推理时仅解码 speech,不显式生成 text:
- **优势**: 效率高,延迟低
- **劣势**: 稳定性可能下降

Survey 指出两种模式孰优的问题仍是开放问题: "The question of whether to incorporate text modality to enhance SpeechLM performance remains an open question."

## Post-Alignment (后对齐)

训练后通过对齐技术优化模型行为:

### Align-SLM
- 发现 SpeechLM 对同一 prompt 生成不一致的语义内容
- 用 TextLM 在 ASR 转写后选择最优响应
- 然后用 DPO 对齐 SpeechLM 生成分布

### SpeechAlign
- 关注声学质量: "golden" speech tokens 与 LM 生成的 tokens 存在分布差异
- 导致 vocoder 从非分布内 token 合成时质量下降
- 用优化技术对齐 LM 输出的 token 分布与 golden 分布

## 训练策略 (Yang et al. 2025 补充)

Yang et al. (2025) [§4.3] 从集成视角补充了对齐训练中各模块的训练策略:

| 策略 | 适用场景 | 效果 |
|------|----------|------|
| 全模型微调 | Speech encoder + adapter + LLM 一起训练 | 最优性能,最高成本 |
| LoRA for LLM | 仅低秩适配 LLM 层 | Pham et al. (2024): 显著优于冻结 LLM |
| 仅训练 adapter | 冻结 encoder 和 LLM | 最轻量,性能受限 |
| 两阶段训练 | 先训练 encoder,再启动 LLM PEFT | Wu et al. (2023): 避免不稳定梯度干扰 |

**关键发现** (Pham et al., 2024): 对于 LLM 模块,LoRA > partial fine-tuning; 对于 encoder 模块,full fine-tuning > partial fine-tuning,但 partial 更具性价比。

详见 [[ModalityAdaptationforSpeechLLM]] 中的具体 adapter 架构。

## 关键发现

Survey 的核心发现:
1. **TextLM 初始化 + 交替训练效果最佳**: TWIST 和 SPIRIT-LM 证明从 TextLM 继续训练显著优于冷启动
2. **交替序列优于简单拼接**: SUTLM 比较了 4 种方式,交替排列在跨模态任务表现最好
3. **text-speech 对齐增强 speech feature 相似性**: SPIRIT-LM 的可视化显示对齐训练后 text 和 speech 特征的相似度显著提升
4. **对齐存在 trade-off**: 增强语义能力可能削弱副语言特征 (tone, emotion) 的捕获
5. **Post-alignment 仍 under-explored**: 安全对齐是未来重点方向

## 关键论文

- SPIRIT-LM (Nguyen et al., 2024): 交替 text-speech tokens, pitch/style 补充
- Spectron (Nachmani et al., 2024): 联合多目标监督对齐
- SUTLM (Chou et al., 2023): 对比 4 种 speech-text 建模方式
- Mini-Omni (Xie & Wu, 2024): 多序列并行对齐, text+7 audio streams
- Moshi (Defossez et al., 2024): word-level 多流对齐
- Align-SLM (2024): DPO 后对齐语义一致性
- SpeechAlign (2024): 声学质量后对齐

## 相关概念

- [[SpeechLanguageModel]]: 对齐是 SpeechLM 训练的核心环节
- [[SemanticvsAcousticTokens]]: 对齐方式取决于使用哪类 tokens
- [[SpeechTokenizer]]: 提供 speech tokens 供对齐
- [[Full-duplexSpokenDialogue]]: 全双工模型需要更复杂的对齐策略

## 演进

Speech-only modeling (GSLM, 2021, 无对齐) → Concatenated speech-text (SUTLM, 2023) → Alternating speech-text (SPIRIT-LM, 2024) → Multi-sequence 并行 (Mini-Omni/Moshi, 2024) → Post-alignment DPO (Align-SLM/SpeechAlign, 2024) → 开放问题: text-present vs text-independent 推理
