---
type: paper
tier: deep
title: "Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction"
arxiv_id: "2502.11946"
source: "Sources/Step-Audio.pdf"
authors: [Step-Audio Team, StepFun]
year: 2025
venue: "arXiv"
tags: [speech-LM, multimodal, dual-codebook, voice-cloning, instruction-control, RLHF, TTS, ASR, dialogue, open-source]
concepts: ["[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[LLM-basedTTS]]", "[[Instruction-GuidedSpeechSynthesis]]", "[[EmotionControlinTTS]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]", "[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[SemanticvsAcousticTokens]], [[LLM-basedTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[LLM-basedTTS]]✓ | 过滤: [[Instruction-GuidedSpeechSynthesis]](pending-review), [[EmotionControlinTTS]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: Step-Audio 采用 130B 参数 LLM (Step-1) 作为 backbone,是目前参数量最大的开源 SpeechLM。其 AQTA (audio-question, text-answer) + TTS 的混合架构与 KB 中典型 SpeechLM 范式不同——理解和生成不在同一模型中端到端完成,而是用 130B LLM 做理解+文本生成,用 3B speech decoder 做语音合成。
>
> **Speech Tokenizer**: Step-Audio 提出 dual-codebook tokenizer: linguistic tokenizer (Paraformer encoder, 16.7Hz, codebook 1024) + semantic tokenizer (CosyVoice tokenizer, 25Hz, codebook 4096),以 2:3 ratio 交错。这是 KB 中记录的 "混合 tokenizer" 路线的新变体——不同于 Mimi/SpeechTokenizer 的单 codec 混合,而是两个独立 tokenizer 物理并行。
>
> **Semantic vs Acoustic Tokens**: Step-Audio 的双 codebook 设计对应 KB 中 semantic vs linguistic token 的新实例: linguistic token 编码音素/语言结构,semantic token 编码语义+粗粒度声学。两者互补降低 next-token perplexity [Fig 4],且提升 ASR 内容一致性。
>
> **LLM-based TTS**: Step-Audio-TTS-3B 是独立的 3B 参数 TTS 模型,采用 chat-based paradigm (system prompt + instruction tags) 替代传统 content+description 分离的 TTS 格式。在 SEED TTS 测试集上 CER/WER 均超越 CosyVoice 2-S 和 FireRedTTS [Table 3]。

> [!summary] 速查
> - **一句话**: 首个生产就绪的开源 130B 语音-文本多模态模型,通过 dual-codebook tokenization + AQTA 架构 + 3B speech decoder + instruction-driven 精细语音控制 + ToolCall 增强,实现 SoTA 的语音对话和可控 TTS [§Abstract]
> - **路线**: User audio → Streaming dual tokenizer (linguistic 16.7Hz + semantic 25Hz, 2:3 interleave) → 130B LLM (Step-1) → text response → 3B Speech Decoder (LM + flow matching + neural vocoder) → waveform [§3, Fig 2]
> - **指标**: TTS CER 1.17% zh / WER 2.0% en (130B, SEED test) [Table 3]; TTS CER 1.31 zh / WER 2.31 en (3B, SEED test) [Table 3]; ASR avg CER 4.64 (pretrain) [Table 1]; AQTA Chat Factuality 66.4% / Relevance 75.2% / Chat Score 4.11 (StepEval-Audio-360, 超越 GLM-4-Voice/Qwen2-Audio/Moshi) [Table 5]; Llama Question 81.0 / Web Questions 75.1 / ComplexBench 74.0 / HSK-6 86.0 [Table 6]; 人类评估 9 维度全面 SoTA (vs GLM-4-Voice, Qwen2-Audio) [Fig 1]
> - **可借鉴**: (1) Dual-codebook tokenizer: linguistic (Paraformer) + semantic (CosyVoice) 以 2:3 交错,降低 LM perplexity 且提升 ASR; (2) Generative data engine: Step-2 LLM 生成文本 → Step-Audio clone 语音 → Audio-Edit model 添加情感/风格 [Fig 5]; (3) RLHF anti-deaf-hacking: 构造清晰音频+deaf-hacking 回复作为 rejected pair 消除奖励模型偏差 [§5.2.6]; (4) Speculative response generation: 用户暂停时预生成响应,~40% 命中率,减少约 500ms 延迟 [§3.4]
> - **局限**: 非端到端 (理解走 LLM,生成走 speech decoder,延迟叠加); 130B 推理成本极高; Moshi 在 StepEval-Audio-360 上表现极差 (非中文) 故对比参考价值有限 [Table 5]; 未公开训练数据规模细节; 多语言支持深度不清楚

## 核心问题

Step-Audio 的目标是构建一个统一理解和生成的开源语音交互框架,解决现有开源系统的四大不足 [§1]:

1. **理解-生成分离**: 现有开源模型通常只具备理解或生成能力之一,难以实现端到端语音对话 [论文原文]
2. **语音数据稀缺**: 高质量语音数据 (多方言/情感/风格) 获取成本极高,制约了可控 TTS 和表情丰富的对话 [论文原文]
3. **精细控制不足**: 现有系统缺乏对方言、情感、唱歌、RAP 等多维度的动态指令级控制 [论文原文]
4. **智能交互局限**: 现有系统缺乏工具调用和角色扮演等复杂认知能力 [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Step-Audio 采用 AQTA (Audio Question, Text Answer) + TTS 的混合架构,由三个核心组件构成 [§3, Fig 2]:

1. **Speech Tokenizer**: 双 codebook 系统,将音频离散化为 linguistic + semantic 两类 token [§3.1]
2. **130B LLM**: 基于 Step-1 的文本预训练大模型,通过音频-文本联合持续预训练获得语音理解和文本生成能力 [§3.2]
3. **Speech Decoder**: 3B 参数语言模型 + flow-matching + neural vocoder,负责将 LLM 文本输出转换为高质量语音 [§3.3]

**为什么选择 AQTA 而非端到端 speech-to-speech?** [§3] 论文给出两个理由:
- 高质量纯语音对话数据极度稀缺,场景受限 [论文原文]
- 外接 TTS 可灵活控制输出语音的音色、语速、情感等参数,持续增强表达能力 [论文原文]

[agent 解读] 这实质上是一个工程化取舍: 用可控性和数据效率换取了端到端的低延迟优势。与 Moshi 的全端到端路线形成鲜明对比。

### 关键设计选择

**Dual-codebook tokenizer** [§3.1]: 采用两个独立 tokenizer:
- **Linguistic tokenizer**: Paraformer encoder 输出量化为离散表示,16.7Hz,codebook 1024。捕获结构化的高层语言/音素特征 [论文原文]
- **Semantic tokenizer**: CosyVoice tokenizer,25Hz,codebook 4096。编码语义和粗粒度声学特征 [论文原文]
- 以 2:3 temporal alignment ratio 交错: 每 2 个 linguistic token 配对 3 个 semantic token [论文原文]

**为什么双 codebook 优于单 codebook?** [§4.4, Fig 4]:
- 单独用 semantic token 时 next-token perplexity 低,语义连贯性好,但 vocoder 重建音质差 (丢失声学细节) [论文原文]
- 单独用 linguistic token 时重建音质好,但 perplexity 高,语义连贯性差 [论文原文]
- 交错使用时,两类 token 互为参考,semantic token 和 linguistic token 的 perplexity 都比单独使用时更低 [论文原文]
- 消融: dual-codebook 在 ASR 上 CER 显著低于单 codebook [§6.2.1]

**三阶段预训练** [§4.2]:
- **Stage 1**: 扩展词表 (+5120 audio tokens),整合图像编码器,冻结 backbone,主要训练 embedding/LM head (lr 5x higher),音频:文本:图像 = 2:1:1,纯 audio continuation 任务 [论文原文]
- **Stage 2**: 引入 audio-text interleaved 数据,audio continuation : audio-text interleaved = 1:1 [论文原文]
- **Stage 3**: 引入 ASR + TTS 数据,audio:audio-text:ASR:TTS = 1:1:1:1,lr 同步 backbone 并 cosine decay [论文原文]

**Speculative response generation** [§3.4]: 实时推理中,当用户暂停时系统预生成推测性响应。状态机: Silence → UserSpeaking → UserPaused (触发 speculative call) → BotReplying (commit 或 discard)。实证约 40% 的推测响应被成功提交,减少约 500ms 延迟 [论文原文]。

**Generative data engine** [§5.1.1, Fig 5]: 解决高质量 TTS 训练数据稀缺的创新方案:
1. Step-2 LLM 将文本改写为多种风格变体 [论文原文]
2. Step-Audio 模型用目标说话人音频作为 prompt 生成语音 [论文原文]
3. Audio-Edit model 添加情感和风格变化 [论文原文]

**Instruction tags** [§5.1.2]: 分为 descriptive tags (语言/方言/声音风格/唱歌) 和 comparative tags (情感/速度的五级层次化控制),实现对 TTS 输出的细粒度指令控制 [论文原文]。

**RLHF 训练** [§5.2]:
- **Reward model**: Bradley-Terry loss 训练,pair-wise accuracy 70.51% on human preference test [§5.2.4]
- **PPO 训练**: clip ε=0.2, KL penalty β=0.05, cosine lr decay [§5.2.6]
- **Anti-deaf-hacking**: 发现 reward model 存在 "deaf hacking" 偏差——给 "I didn't hear clearly" 类回复高分。构造清晰音频 + deaf-hacking 回复作为 rejected pair 来消除此偏差 [§5.2.3, §5.2.6] [论文原文]

### 训练策略

| 阶段 | 数据量 | 核心任务 | 特殊设置 |
|------|--------|---------|----------|
| Pretrain Stage 1 | 1.2T tokens | Audio continuation | backbone lr 2e-5, embed/head lr 5x |
| Pretrain Stage 2 | 800B tokens | + Audio-text interleave | 1:1 ratio |
| Pretrain Stage 3 | - | + ASR + TTS | 4:3:3 audio:text:image, cosine lr |
| SFT | 1 epoch | TQTA+AQTA+TAQTA | lr 5.656e-5 → 5.656e-6 |
| RLHF/PPO | - | AQTA fine-tuning | 80 step warmup, β=0.05 |
| TTS SFT | 1 epoch | 3B speech decoder | lr 2e-5 → 2e-6 |

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ASR avg CER (pretrain) | 4.64 | Whisper-v3 7.28 / Qwen2-Audio 4.32 | AISHELL-1,2,Wenet,LibriSpeech etc. | [Table 1] |
| ASR avg CER (chat) | 5.89 | GLM-4-Voice Chat 146.74 | Same | [Table 1] |
| TTS CER zh (130B) | 1.17 | GLM-4-Voice 2.19 / MinMo 2.48 | SEED test | [Table 2] |
| TTS WER en (130B) | 2.0 | GLM-4-Voice 2.91 / MinMo 2.90 | SEED test | [Table 2] |
| TTS CER zh (3B single-cb) | 1.37 | CosyVoice 3.63 / CosyVoice 2 1.45 | SEED test | [Table 3] |
| TTS CER zh (3B dual-cb) | 1.31 | FireRedTTS 1.51 / MaskGCT 2.27 | SEED test | [Table 3] |
| AQTA Factuality | 66.4% | GLM4-Voice 54.7% / Qwen2-Audio 22.6% | StepEval-Audio-360 | [Table 5] |
| AQTA Relevance | 75.2% | GLM4-Voice 66.4% / Qwen2-Audio 26.3% | StepEval-Audio-360 | [Table 5] |
| AQTA Chat Score | 4.11 | GLM4-Voice 3.49 / Qwen2-Audio 2.27 | StepEval-Audio-360 | [Table 5] |
| Llama Question | 81.0 | Freeze-Omni 72.0 / MinMo 78.9 | Llama-Questions | [Table 6] |
| Web Questions | 75.1 | Freeze-Omni 44.7 / MinMo 55.0 | Web Questions | [Table 6] |
| ComplexBench | 74.0 | GLM4-Voice 66.0 / Qwen2-Audio 54.0 | ComplexBench | [Table 6] |

## 局限性

1. **非端到端架构**: LLM 理解 + Speech Decoder 生成的级联增加延迟,无法实现 Moshi 级别的低延迟全双工 [agent 解读]
2. **推理成本极高**: 130B 参数推理需要大量 GPU 资源,限制了部署场景 [agent 解读]
3. **评估对比局限**: StepEval-Audio-360 以中文为主,Moshi 因不支持中文在此 benchmark 上几乎无参考价值 [Table 5]
4. **数据透明度不足**: 预训练数据的具体组成和清洗方法未完全公开 [agent 解读]
5. **Deaf hacking 问题**: RLHF 过程中发现的 reward model 偏差,虽已部分解决但论文承认可能需要 rule-based rewards 进一步消除 [§5.2.6]

## 点评

Step-Audio 代表了 "工业化开源 SpeechLM" 的路线: 用极大参数量 (130B) + 精心的数据工程 + 模块化架构实现全面能力。

**最大亮点是 Generative Data Engine**: 用 LLM 生成文本 → 语音模型克隆 → Audio-Edit 添加风格,形成了自动化的高质量 TTS 数据生产管线。这解决了情感/风格/方言语音数据的核心瓶颈,是一个可复用的方法论创新 [agent 解读]。

**Dual-codebook tokenizer 是实用的折衷方案**: 不同于 Moshi 的 split RVQ (物理解耦在同一 codec 内),Step-Audio 直接用两个独立 tokenizer 并行,更工程友好但牺牲了压缩效率 [agent 解读]。

**RLHF 中的 deaf-hacking 现象值得关注**: 这是 speech LLM 特有的对齐问题,text LLM 中不存在,论文的分析和解决方案为后续工作提供了有价值的经验 [agent 解读]。

## 可复用的 idea

1. **Generative data engine**: LLM rewrite → TTS clone → Audio-Edit style,自动化生产多情感/多风格/多方言 TTS 训练数据
2. **Dual-codebook tokenizer**: linguistic (Paraformer) + semantic (CosyVoice) 以 2:3 交错,降低 perplexity 且提升内容一致性
3. **Speculative response generation**: 用户暂停时预生成响应,命中率约 40%,减少约 500ms 延迟
4. **Anti-deaf-hacking RLHF**: 构造清晰音频 + deaf-hacking 回复作为 rejected pair,消除 reward model 偏差
5. **Instruction tags taxonomy**: descriptive (语言/风格) + comparative (情感/速度五级) 的双层标签体系
