---
type: paper
tier: deep
title: "GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot"
arxiv_id: "2412.02612"
source: "Sources/GLM-4-Voice.pdf"
authors: [Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, Jie Tang]
year: 2024
venue: "arXiv (Zhipu AI / Tsinghua University)"
tags: [speech-LM, end-to-end, spoken-chatbot, speech-tokenizer, flow-matching, single-codebook, streaming, interleaved-data, pre-training, bilingual]
concepts: ["[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[ConditionalFlowMatching]]", "[[Single-codebookvsMulti-codebook]]", "[[StreamingSpokenDialogue]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[SemanticvsAcousticTokens]]"]
models: ["[[CosyVoice]]", "[[Whisper]]", "[[Moshi]]"]
tasks: []
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 3
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[ConditionalFlowMatching]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓ | 过滤: [[Single-codebookvsMulti-codebook]](pending-review), [[StreamingSpokenDialogue]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review) | 未命中但可能相关: 无

**[[SpeechLanguageModel]]**: GLM-4-Voice 属于 SpeechLM 演进中"从 TextLM 继续预训练" (continued pre-training) 路线,与 TWIST、SPIRIT-LM、Moshi 同属一脉。区别在于: TWIST 仅用无监督语音数据,SPIRIT-LM 依赖 speech-text 平行语料做交替训练(但受平行语料稀缺制约),Moshi 扩大无监督数据至 7M 小时但智能上限仍受限于纯语音数据。GLM-4-Voice 的核心突破在于通过合成交替数据 (synthetic interleaved data) 绕过平行语料瓶颈,从文本预训练语料合成 speech-text 交替数据,使预训练扩展到 1T tokens。

**[[SpeechTokenizer]]**: GLM-4-Voice 的 tokenizer 属于"监督式 semantic tokenizer"路线,与 CosyVoice 的 FSQ-SenseVoice 同源,但采用了不同的 ASR backbone (Whisper-large-v3 而非 SenseVoice),并通过 VQ bottleneck + pooling 实现 12.5Hz 超低帧率。这是当时已知最低帧率的实用 speech tokenizer 之一,后续 GLM-TTS (2025) 将帧率提升至 25Hz 以获得更好的语音质量。

**[[ConditionalFlowMatching]]**: GLM-4-Voice 的 speech decoder 直接复用 CosyVoice 的 CFM + HiFi-GAN 架构,将离散 speech token 恢复为自然语音。与 CosyVoice 不同的是,GLM-4-Voice 适配了 streaming 推理模式,通过 truncated audio fine-tuning 使 decoder 支持 chunk-wise 流式合成 (block size b=0.8s)。

**谱系定位**: GLM-4-Voice 在 SpeechLM 演进中处于 SPIRIT-LM (2024) → GLM-4-Voice (2024.12) → Moshi (2024) 的同期位置,是首个通过合成交替数据将 speech-text 预训练扩展到 1T tokens 的系统。在 audio-token-based integration 路线中,它与 Moshi 代表了"大规模预训练"策略,与 Llama-Omni/Mini-Omni 的"轻量微调"策略形成对比。

## 速查

> [!summary] 速查
> - **一句话**: 通过合成 speech-text 交替数据将预训练扩展到 1T tokens,结合 12.5Hz 单码本 VQ-Whisper tokenizer 和 streaming thoughts 推理模板,构建了首个兼具高智能和自然语音表达的端到端 spoken chatbot
> - **路线**: 用户语音 → VQ-Whisper tokenizer (12.5Hz, 175bps) → GLM-4-9B (扩展词表) → streaming thoughts (text+speech 交替输出) → CosyVoice CFM decoder + HiFi-GAN → 语音响应
> - **指标**: ChatGPT Score General QA 5.40 / Knowledge 5.20 (vs Moshi 2.42/3.60); UTMOS 4.45 (vs Moshi 3.90); ASR-WER 5.74% (vs Moshi 7.95%) [Table 6]
> - **可借鉴**: (1) 合成交替数据绕过 speech-text 平行语料瓶颈; (2) Streaming Thoughts 模板以固定 text:speech ratio 交替解码实现低延迟; (3) SFT 阶段分离 text/speech loss 解决不同学习速度问题
> - **局限**: 仅支持中英双语; 未探索全双工; tokenizer 12.5Hz 帧率下重建质量受限 (WER 8.43 on reconstruction); 无公开 SFT 数据详情; 推理依赖 streaming thoughts 比例的经验选择

## 核心问题

GLM-4-Voice 试图解决两个核心问题:

1. **SpeechLM 的智能瓶颈**: 语音数据相比文本数据极度稀缺。即使 Moshi 用了 7M 小时语音数据预训练,其知识和推理能力仍远不及 text LLM [§1]。如何把 text LLM 的知识高效迁移到 speech modality?

2. **语音自然度的缺失**: 轻量级方案 (如 Llama-Omni, Mini-Omni) 直接在 LLM 后接 TTS 模块并仅在 instruction data 上微调,虽然快捷但无法生成具有丰富副语言特征 (情感、语调、语速、方言) 的自然语音 [§1]。原因是缺少 dedicated speech pre-training。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

GLM-4-Voice 由三个组件构成 [§3, Fig 2]:

1. **Speech Tokenizer**: 基于 Whisper-large-v3 改造的单码本监督式 tokenizer,12.5Hz 帧率,175bps 比特率
2. **Language Model**: 从 GLM-4-9B-Base 继续预训练,扩展词表加入 speech tokens
3. **Speech Decoder**: 复用 CosyVoice 架构 (speech token encoder + CFM + HiFi-GAN vocoder)

**设计哲学**: 对自回归 Transformer 架构做最小修改,用单码本方案避免多码本带来的复杂架构调整 (如 Moshi 的 RQ-Transformer),同时保留模型的文本处理能力 [论文原文, §3]。输入和输出使用统一的 speech representation,支持对无监督语音数据做 next-token prediction 预训练 [论文原文, §3]。

### 关键设计选择

**1. VQ-Whisper Speech Tokenizer [§3.1, Fig 1]**

为什么选择 12.5Hz 单码本? 理想的 SpeechLM tokenizer 需满足三个条件: (1) 低采样率 + 单码本支持自回归生成, (2) 与文本对齐以迁移预训练 LLM 的知识, (3) 支持高质量语音合成 [论文原文, §3.1]。

具体实现:
- 在 Whisper-large-v3 encoder 中间插入 pooling layer + VQ layer
- Pooling 降低帧率: 原始 50Hz → 12.5Hz (4x downsample) [Table 1]
- VQ 用 EMA 更新码本向量 (codebook size = 16384, 即 14 bits/token),通过 reset 低使用率码本防止 codebook collapse [§3.1, 引用 Dhariwal et al.]
- 为支持 streaming 编码: 将 Whisper 的双向注意力替换为 block causal attention,将卷积层替换为 causal convolution [§3.1]
- 训练数据: 多个 ASR 数据集 (LibriSpeech, GigaSpeech, MLS-Eng, Wenet, CommonVoice, AISHELL-1) + 10K 小时中文私有数据 + 700K 小时无监督伪标签数据 [§3.1]

**评估结果** [Table 1]:
- 12.5Hz 变体在 ASR 准确率 (WER 2.10 LS-clean, 4.90 AISHELL-1) 和重建质量 (WER 8.43, MOSNet 3.39) 间达到最佳平衡
- 对比: SpeechTokenizer 50Hz/1.5K bps WER 9.97, Moshi (Mimi) 12.5Hz/1.1K bps WER 8.36
- 175bps 是所有方案中最低比特率,远低于 SpeechTokenizer 的 1.5K-4K bps

**为什么 175bps 如此之低仍可用?** [agent 解读] 因为这是监督式 semantic tokenizer,VQ bottleneck 主要编码语义信息 (由 ASR 任务保证),丢弃了大部分声学细节。声学恢复的任务交给下游 CFM decoder。这与 CosyVoice 的设计理念一致: tokenizer 只负责语义,CFM 负责声学。

**2. Speech Decoder 的 Streaming 适配 [§3.2]**

采用 CosyVoice 的 decoder 架构: speech token encoder + conditional flow matching + HiFi-GAN vocoder [§3.2]。

**Streaming 支持**: Fine-tuning 阶段引入 truncated audio samples (取前 n*b 秒的音频), 让 decoder 学会处理流式场景 [§3.2]。推理时, decoder 处理前 n*b 秒对应的 speech tokens, 用前 (n-1)*b 秒语音作为 prompt, 预测 (n-1)*b 到 n*b 秒的语音。最小延迟 = b 秒, GLM-4-Voice 设 b=0.8, 即至少 10 个 speech tokens 才能产出首段语音 [§3.2]。

**3. Streaming Thoughts 推理模板 [§3.3, Fig 2]**

GLM-4-Voice 将 speech-to-speech 解耦为两个子任务 [论文原文, §3.3]:
- Speech-to-Text: 从用户语音 Qs 生成文本回复 At
- Speech-and-Text-to-Speech: 从 Qs 和 At 生成语音回复 As

**问题**: 如果先完整生成 At 再生成 As, 首 token 延迟过高 [论文原文, §3.3]。

**解决方案 — Streaming Thoughts**: 模型交替输出 text tokens 和 speech tokens, 比例为 13 text : 26 speech [§3.3]。选择 1:2 是因为文本生成必须快于语音, 否则语音 tokens 会缺少必要的文本上下文 [论文原文, §3.3]。26 个 speech tokens 基于经验, 足以产生连贯的语音片段 [论文原文, §3.3]。

**总延迟公式** [§3.3]:
```
T_total = T_speech_tokenize + T_llm_prefill + T_llm_decode(23 tokens) + T_speech_decode(10 tokens)
```
首段语音仅需 23 个 LLM decoding 步 (13 text + 10 speech)。

### 训练策略

**Stage 1: Joint Speech-Text Pre-training [§4.1]**

从 GLM-4-9B-Base 初始化, 扩展词表加入 speech tokens, 在 1T tokens 上继续预训练 [§4.1]:

| 数据类型 | Speech Tokens | Text Tokens | Epochs |
|----------|--------------|-------------|--------|
| Interleaved Speech-Text | 455B | 279B | 0.90 |
| Unsupervised Speech | 31B | — | 2.10 |
| ASR + TTS | 11B | 3.5B | 2.07 |
| Text-only | — | ~30B | 0.03 |

[Table 2]

**Interleaved data 的合成方法**: 用 text-to-token model 将文本预训练语料转化为 speech tokens, 构造 speech-text 交替序列 [§4.1, Fig 2]。这是 SPIRIT-LM 方法的扩展 — SPIRIT-LM 受限于真实 speech-text 平行语料的稀缺, 而 GLM-4-Voice 通过合成绕过了这个瓶颈 [agent 解读]。

采样比例: 30% text data + 剩余为 interleaved + speech data [§4.1]。

训练配置: AdamW, β1=0.9, β2=0.95, 序列长度 8192, 学习率从 6e-5 线性衰减到 6e-6 [§4.1.1]。

**Stage 2: Supervised Fine-tuning [§4.2]**

SFT 数据:
1. **多轮对话数据**: 从文本数据筛选, 过滤代码/数学内容, 缩短长文本, 合成对应语音; 人工录制多样化语音输入 [§4.2.1]
2. **语音风格控制数据**: 包含速度、情感、方言等特定风格需求的高质量多轮对话 [§4.2.1]

**关键训练技巧 — 分离 loss**: 作者发现 text output 学习速度快于 speech output [论文原文, §4.2.2]。解决方案: 将每个样本拆分为两个训练信号 — 一个 mask speech loss 只学 text, 另一个 mask text loss 只学 speech [§4.2.2]。Speech output 训练 20 epochs, text output 仅 4 epochs [§4.2.2]。

防过拟合: weight decay 0.1, dropout 0.5 (hidden layers), gradient clip 1.0, 学习率 1e-5 → 1e-6 [§4.2.2]。

## 实验

### Base Model

| 指标 | 本文 (S→T) | 本文 (S→S) | Moshi (S→S) | SPIRIT-LM (S→S) | TWIST (S→S) | 数据集 | 出处 |
|------|-----------|-----------|-------------|-----------------|-------------|--------|------|
| Topic-StoryCloze | 93.6 | 82.9 | 83.0 | 82.9 | 66.6 | StoryCloze | [Table 3] |
| StoryCloze | 76.3 | 62.4 | 60.8 | 61.0 | 53.3 | StoryCloze | [Table 3] |
| Web Questions | 32.2 | 15.9 | 9.2 (S→S) / 26.6 (S→T) | — | 1.5 | Web Questions | [Table 4] |
| Llama Questions | 64.7 | 50.7 | 21.0 (S→S) / 62.3 (S→T) | — | 4.0 | Llama Questions | [Table 4] |
| TriviaQA | 39.1 | 26.5 | 7.3 (S→S) | — | — | TriviaQA | [Table 4] |

**关键发现**: S→T 性能始终优于 S→S, 尤其在 Spoken QA 任务上差距显著, 说明文本引导仍然是智能 speech chatbot 所必需的 [论文原文, §5.1]。但 GLM-4-Voice 显著缩小了 S→S 和 S→T 之间的差距, 特别是在 Llama Questions 上 (50.7 vs 64.7), 表明直接 speech-to-speech chatbot 的可行性 [论文原文, §5.1]。

### ASR / TTS (Base Model)

| 指标 | 本文 | Whisper-large-v3 | CosyVoice | 数据集 | 出处 |
|------|------|-----------------|-----------|--------|------|
| ASR WER (clean) | 2.82 | 2.50 | — | LibriSpeech | [Table 5] |
| ASR WER (other) | 7.66 | 4.53 | — | LibriSpeech | [Table 5] |
| ASR CER | 2.46 | 9.31 | — | AISHELL-1 | [Table 5] |
| TTS WER (LibriTTS) | 5.64 | — | 3.17 | LibriTTS | [Table 5] |
| TTS WER (Seed-en) | 2.91 | — | 3.39 | Seed-TTS-eval | [Table 5] |
| TTS WER (Seed-zh) | 2.10 | — | 3.10 | Seed-TTS-eval | [Table 5] |

### Chat Model

| 指标 | 本文 | Moshi | Llama-Omni | Mini-Omni | SpeechGPT | 出处 |
|------|------|-------|------------|-----------|-----------|------|
| General QA (GPT-4o) | **5.40** | 2.42 | 3.50 | 2.44 | 1.40 | [Table 6] |
| Knowledge (GPT-4o) | **5.20** | 3.60 | 3.90 | 1.10 | 2.20 | [Table 6] |
| UTMOS | **4.45** | 3.90 | 3.92 | 3.17 | 3.86 | [Table 6] |
| ASR-WER (%) | **5.74** | 7.95 | 9.18 | 25.28 | 66.57 | [Table 6] |

GLM-4-Voice 在所有 4 个指标上均大幅领先所有 baseline。UTMOS 4.45 表明语音自然度接近人类水平; ASR-WER 5.74% 说明 text-speech alignment 极高 [Table 6]。

## 局限性

1. **仅双语 (中英)**: 未探索多语言扩展, 可能受限于 GLM-4-9B 的预训练语言覆盖 [agent 解读]
2. **非全双工**: 不支持 full-duplex 交互 (用户和系统同时说话), 与 Moshi 的全双工能力相比是功能缺失 [agent 解读]
3. **Tokenizer 重建质量有限**: 12.5Hz/175bps 下重建 WER 8.43 [Table 1], 虽然 CFM decoder 部分补偿了声学损失, 但超低比特率仍可能在复杂声学场景下造成信息丢失 [agent 解读]
4. **SFT 数据不透明**: 多轮对话数据和风格控制数据的规模和构造细节未公开, 复现存在障碍 [agent 解读]
5. **Streaming Thoughts 比例经验选择**: 13 text : 26 speech 的比例基于经验而非系统消融 [§3.3], 不同场景下最优比例可能不同 [agent 解读]
6. **评估局限**: Chat model 评估仅限于英语 (为与英语-only baseline 公平比较, 限制了 GLM-4-Voice 的中文输出) [§5.2], 双语能力未被充分评估

## 点评

**核心创新在于数据工程而非架构**: GLM-4-Voice 的三个组件 (VQ-Whisper tokenizer, CosyVoice decoder, GLM-4-9B LLM) 在架构上并无重大创新, 关键贡献在于: (1) 合成 interleaved speech-text 数据突破平行语料瓶颈, 使 1T token 预训练成为可能; (2) Streaming Thoughts 推理模板在不修改模型架构的前提下解决延迟问题; (3) SFT 阶段的 text/speech loss 分离策略解决学习速度不对称问题。

**单码本路线的成功验证**: GLM-4-Voice 证明了单码本 (175bps) + CFM decoder 的组合对 SpeechLM 的可行性。相比 Moshi 的 8 codebook RQ-Transformer, 单码本方案保持了标准 AR LM 架构的简洁性, 同时通过 CFM decoder 补偿声学信息。

**与后续 GLM-TTS 的关系**: GLM-TTS (2025) 在 GLM-4-Voice 基础上将 tokenizer 帧率从 12.5Hz 提升至 25Hz, 词表从 16K 扩至 32K, 加入 Pitch Estimator, 改为非因果架构, 并引入 GRPO RL 对齐。这反映了 12.5Hz 在 TTS 质量上的不足。

**实验对比的局限**: 对比的 baseline (Moshi, Mini-Omni, Llama-Omni, SpeechGPT) 在模型规模和训练资源上差异显著, 且评估指标主要依赖 GPT-4o 打分, 人工评估缺失。

## 可复用的 idea

1. **合成 interleaved data 突破数据瓶颈**: 用 text-to-token model 将纯文本语料转为 speech-text 交替序列, 可应用于任何需要 speech-text 对齐训练的场景
2. **Streaming Thoughts 交替解码**: 固定比例交替输出 text 和 speech tokens, 在不改模型架构的前提下控制首 token 延迟, 可迁移到任何 speech-text 联合生成系统
3. **分离 loss 训练**: 对同一样本分别 mask text/speech output 独立训练, 解决多模态输出学习速度不对称问题
4. **VQ-Whisper tokenizer 方案**: 在成熟 ASR 模型中间插入 VQ bottleneck + pooling, 快速得到高质量 semantic tokenizer, 可迁移到其他 ASR backbone
5. **Truncated audio fine-tuning**: 对 decoder 引入 truncated audio 训练样本实现流式推理, 是一种简洁的 streaming 适配方法

---

检索命中: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[ConditionalFlowMatching]] | 过滤: [[Single-codebookvsMulti-codebook]](pending-review), [[StreamingSpokenDialogue]](pending-review), [[Speech-LLMIntegrationTaxonomy]](pending-review) | 未命中但可能相关: 无

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | WHY/HOW 解释充分,设计选择动机清晰 |
> | 可信赖 | pass | 所有关键数字与 PDF 交叉验证一致,出处标注覆盖率 ~95% |
> | 可区分 | pass | 来源标注覆盖率 ~90%,无推断写成断言 |
> | 可定位 | pass-with-fixes | KB 谱系定位清晰; models 字段已补充 Moshi |
> | 不污染 | pass | 挂接合理,无 overclaim |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> 详见 `_review/GLM-4-Voice-review.yml`
