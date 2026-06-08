---
type: paper
tier: deep
title: "Continual Pre-training for Codec-Based Speech LLMs: Balancing Understanding and Generation"
arxiv_id: "2502.16897"
source: "Sources/BalancingUnderstanding-Generation.pdf"
authors: [Jiatong Shi, Chunlei Zhang, Jinchuan Tian, Junrui Ni, Hao Zhang, Shinji Watanabe, Dong Yu]
year: 2025
venue: "arXiv"
tags: [speech-LM, codec, continual-pre-training, ASR, TTS, speech-translation, S2S-Trans, catastrophic-forgetting, modality-alignment]
concepts: ["[[CodecLanguageModel]]", "[[SpeechLanguageModel]]", "[[SemanticvsAcousticTokens]]", "[[ModalityAdaptationforSpeechLLM]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[SpeechTokenizer]]"]
models: ["[[SoundStream]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[SpeechLanguageModel]], [[SemanticvsAcousticTokens]], [[SpeechTokenizer]]; 3 个待确认: [[CodecLanguageModel]], [[ModalityAdaptationforSpeechLLM]], [[Speech-LLMIntegrationTaxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[CodecLanguageModel]], [[SemanticvsAcousticTokens]]✓, [[ModalityAdaptationforSpeechLLM]], [[SpeechTokenizer]]✓, [[Speech-LLMIntegrationTaxonomy]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文属于 codec-based SpeechLM 路线 ([[CodecLanguageModel]])。该路线直接在 neural audio codec (如 SoundStream, EnCodec) 产生的离散 acoustic tokens 上训练语言模型,其核心优势是高保真语音重建,但面临 semantic-acoustic gap 问题 — 纯 acoustic tokens 语义信息稀疏,导致理解任务 (ASR) 表现较差 ([[SemanticvsAcousticTokens]])。

**已有认知**: KB 已记录的 CodecLM 系统 (VioLA, AudioLM, VALL-E 等) 主要聚焦于生成能力。SpeechLM 的训练策略分类中,continued pre-training 是已知的第二阶段策略 (TWIST, AudioPaLM, SPIRIT-LM, Moshi, Mini-Omni 等均采用),但先前 CPT 研究多使用 SSL-based semantic tokens,codec-based CPT 的系统研究是空白。Modality Adaptation 概念页记录了三种主要适配方法 (Conv downsampling, CTC compression, Q-Former),但这些方法面向 latent-representation 路线;codec token 路线的模态对齐方法尚未被系统整理。

**创新判断**: 本文填补了 codec-based CPT 的空白 — 在 CodecLM 训练策略中明确拆分了 speech-only CPT vs joint speech-text CPT 的效果差异,并首次实现了仅用 codec tokens 的 single-pass S2S-Trans (无中间文本/语义 token)。这超越了 AudioPaLM 和 Seamless 仍依赖中间监督的做法。

## 速查

> [!summary] 速查
> - **一句话**: 提出 continual pre-training (CPT) 框架,通过在预训练文本 LLM 上继续训练 codec-discretized speech,使单一模型同时具备语音理解 (ASR) 和生成 (TTS) 能力,并首次实现 codec-only single-pass S2S-Trans
> - **路线**: Speech → SoundStream codec tokens → shared embedding (sum L' levels) → Qwen1.5-0.5B Transformer (CPT adapted) → parallel prediction heads → text tokens / codec tokens
> - **指标**: ASR WER 3.7/6.3 (LibriSpeech clean/other), TTS WER 3.7 + UTMOS 3.59 + SPK-SIM 0.65 (LibriTTS), S2S-Trans ASR-BLEU 33.4 + UTMOS 3.66 (GigaST EN-to-ZH) [Table I-IV]
> - **可借鉴**: (1) CPT 中 10% 纯文本数据即可有效缓解灾难性遗忘 (perplexity 38.59 vs 47.88); (2) 多层 codec tokens 直接求和为单一 embedding 而非展开为长序列,大幅简化了 RVQ 建模; (3) S2S-Trans 作为 held-out task 验证 CPT 的跨模态泛化能力
> - **局限**: 模型仅 0.5B 规模,ASR 性能落后于 7B 级 SSL-based 系统 (Qwen-Audio2 WER 1.3); codec tokens 在语义任务上仍有结构性劣势; 仅支持 EN-ZH 双语; 未开源完整训练代码

## 核心问题

本文解决的核心问题是: **codec-based speech LLM 如何在保留高保真语音生成能力的同时,不丢失语义理解能力?**

这个问题的根源是 codec tokens 的特性 — 它们编码了低级声学细节,擅长高保真重建,但与文本 token 的语义空间差距巨大 [§I]。当直接将 codec tokens 输入预训练文本 LLM 时,输入分布的剧烈变化会导致 catastrophic forgetting,破坏 LLM 已学到的语言能力 [§I]。

具体来说,论文追问两个子问题:
1. CPT 能否弥合 codec tokens 与 text tokens 之间的 modality gap? [§I]
2. 在 CPT 中,speech-only 和 joint speech-text 两种配置会产生怎样不同的理解-生成 trade-off? [§I]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由三个核心组件构成 [§III, Fig 1]:

1. **Speech Tokenizer**: 采用 SoundStream 架构的 neural codec [§IV],包含 encoder + quantizer (L codebooks) + decoder。语音信号 S 经编码器产生 hidden states Q,量化器将 Q 映射为多层离散码 C,通过码本 embedding M 重建 E,再由 decoder 重建信号。选取前 L' 层 codec streams 使用 [§III]。

2. **Shared Multimodal Embedding**: 多层 codec tokens C' 的 embedding 直接 **求和** (sum) 为单一 embedding E^speech [§III]。文本 tokens 直接通过 text embedding 层映射为 E^text。两者交错 (interleaved) 送入 Transformer [§III]。

   [agent 解读] 这种 sum-based 融合相比 flattening (展开为 L' 倍长序列) 大幅节省了序列长度,但代价是不同 codebook level 之间的信息被压缩到同一向量中,可能丧失层级区分能力。

3. **Transformer Backbone + Parallel Prediction Heads**: 基于 Qwen1.5-0.5B 架构 [§IV-B],保留原始 tokenizer 和 text embeddings 以保护语言能力。文本输出 head 部分初始化自 Qwen1.5,codec 相关词表随机初始化。每个 codec level 有独立的并行预测 head,避免了层级依赖建模 [§III]。

### 关键设计选择

**为什么用 CPT 而不是直接 SFT?** [论文原文] 当 codec tokens 直接输入预训练文本 LLM 时,输入分布与文本差异巨大,导致灾难性遗忘 [§I]。CPT 的作用是在 SFT 之前先将 speech modality 的知识注入 LLM Transformer base,使模型逐步适应 codec token 的分布,而非在 SFT 时突然面对完全陌生的模态 [§III]。实验证实: 没有 CPT 的模型无法在 S2S-Trans 上收敛 [§V-A, Table I]。

**为什么选 codec tokens 而非 SSL semantic tokens?** [论文原文] SSL tokens (HuBERT, w2v-BERT) 语义对齐好但声学保真度差,依赖 vocoder 重建语音,在多说话人合成中表现不佳 [§II]。Codec tokens 保留低级声学信息,能实现高保真生成和跨模态泛化 [§II]。

**Speech-only CPT vs Joint Speech-Text CPT**: [论文原文] 两种 CPT 配置的目标函数不同 [§III]:
- Speech-only CPT (Eq. 2): 仅在 codec tokens 上做 next-token prediction,注入语音模态知识但可能遗忘文本能力
- Joint CPT (Eq. 3): 在交错的 speech+text token 序列上做 next-token prediction,平衡两种模态

Joint CPT 的数据配比: 6 个语音任务各 15% + 10% 纯文本 (5% 通用 + 5% MT 语料) [§IV-B]。[论文原文] 即使少量文本数据也能起到 stabilizing regularizer 的作用,锚定模型的内部表征 [§IV-B]。

**为什么 S2S-Trans 不在 CPT 中训练?** [论文原文] 故意排除 S2S-Trans 以检验 CPT 是否能产生 emergent cross-lingual translation 能力 — 即模型是否能通过学习 ASR、TTS、S2T-Trans 等解耦任务,自发泛化到从未见过的 speech-to-speech mapping [§IV-A, §V-A]。

**Codec embedding 的 sum 策略**: [agent 解读] 对比 Moshi 使用 RQ-Transformer 按层级预测、VALL-E 使用 AR+NAR 两阶段、MusicGen 使用 codebook delay pattern,本文的 sum embedding + parallel heads 是最简单的方案。其合理性可能在于: (1) CPT 阶段模型已学会从 summed embedding 中解纠缠不同 level 的信息; (2) 0.5B 规模下复杂的层级建模收益有限。

### 训练策略

训练分两阶段 [§III-IV]:

1. **CPT 阶段**: 
   - 数据: ~140K 小时英语+中文语音,配对转录和翻译 [§IV-A]
   - 6 个 CPT 任务: speech continuation, language modeling, ASR, TTS, S2T-Trans, T2ST [§IV-A]
   - 序列格式: `(Condition)(Prompt)(Target)` + boundary tokens [§IV-A]
   - 基础设施: Megatron-LM, tensor parallelism=8, batch=640, seq_len=4096, 40K steps, AdamW (peak LR 1e-5), BFloat16, 943.5M params [§IV-B]
   - 词表扩展至 155,012 (含 padding 和 shape-adjustment tokens) [§IV-B]

2. **SFT 阶段**: 在 CPT 完成后,对 4 个下游任务 (ASR, TTS, S2T-Trans, S2S-Trans) 分别微调,目标为 P(R^tgt | R^inp) [§III, Eq. 4]

## 实验

| 指标 | 本文 (Joint CPT) | 本文 (Speech CPT) | No Init | Text LLM Init | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| ASR WER (clean) | **3.7** | 5.5 | 5.5 | 4.8 | LibriSpeech | [Table II] |
| ASR WER (other) | **6.3** | 8.9 | 9.5 | 8.5 | LibriSpeech | [Table II] |
| ASR CER | **7.2** | 13.0 | 15.5 | 13.1 | Aishell2 | [Table II] |
| TTS UTMOS | 3.59 | **3.65** | 3.01 | 2.78 | LibriTTS | [Table III] |
| TTS WER | **3.7** | **3.7** | 17.5 | 18.8 | LibriTTS | [Table III] |
| TTS SPK-SIM | 0.65 | **0.66** | 0.55 | 0.51 | LibriTTS | [Table III] |
| S2S ASR-BLEU | **33.4** | 28.0 | fail | fail | GigaST (EN-ZH) | [Table I] |
| S2S UTMOS | **3.66** | 3.41 | fail | fail | GigaST (EN-ZH) | [Table I] |
| S2T BLEU (EN-ZH) | **33.1** | 24.8 | 25.5 | 28.9 | CoVoST2 | [Table IV] |
| S2T BLEU (ZH-EN) | **16.1** | 5.4 | 5.8 | 9.9 | CoVoST2 | [Table IV] |
| Perplexity (text) | **38.59** | 47.88 | - | 36.51 | LAMBADA | [Table VI] |

**关键发现**:

1. **CPT 是 S2S-Trans 的必要条件**: 无 CPT 的模型 (No Init 和 Text LLM Init) 完全无法在 S2S-Trans 上收敛,而 CPT 模型达到 ASR-BLEU 33.4,接近级联系统 HW-TSC 的 33.6 [Table I]。

2. **Speech-only CPT 偏向生成,Joint CPT 偏向理解**: Speech CPT 在 TTS 上最优 (UTMOS 3.65 vs Joint 的 3.59),Joint CPT 在所有理解任务上最优 (ASR WER 3.7 vs Speech CPT 的 5.5) [Table II, III]。

3. **Joint CPT 有效缓解灾难性遗忘**: LAMBADA perplexity 仅从 36.51 升至 38.59 (+5.7%),而 Speech-only CPT 升至 47.88 (+31%) [Table VI]。10% 文本数据起到关键正则化作用 [§V-C]。

4. **与外部系统对比**: Joint CPT 的 ASR WER 3.7 (LibriSpeech clean) 优于 Mini-Omni (4.5) 和 Moshi (5.7),但落后于使用专用音频编码器的 Qwen-Audio2 (1.3) 和使用 SSL tokens 的 VoxtLM (2.7) [Table II, V]。TTS WER 3.7 优于 Moshi (4.7) 和 UniAudio (13.1) [Table III, V]。

## 局限性

1. **模型规模限制**: 仅 0.5B (Qwen1.5-0.5B),与 7B 级系统 (SALMONN, Qwen-Audio2) 的 ASR 差距较大。论文未探讨 CPT 策略在更大模型上的效果 [agent 解读]。

2. **Codec tokens 的结构性语义劣势**: 即使经过 Joint CPT,ASR 性能仍落后于使用 SSL-based semantic tokens 的系统,因为 codec tokens 的设计目标是声学重建而非语义对齐 [§V-B]。论文承认这是 codec-based 路线的固有限制 [§V-B]。

3. **仅支持双语**: CPT 数据仅包含英语和中文,且 S2S-Trans 仅测试 EN-to-ZH 方向 [§IV-A, IV-C]。多语言泛化能力未知。

4. **S2S-Trans 目标语音为单说话人**: GigaS2S 语料的目标端由单说话人 TTS 模型合成 [§IV-C],因此 S2S-Trans 评估未涵盖说话人保持或多说话人场景。

5. **翻译数据为机器翻译生成**: CPT 使用内部 MT 模型生成英汉翻译对 [§IV-A],翻译质量可能影响跨语言对齐效果,但论文未评估翻译质量本身 [agent 解读]。

6. **Codec tokenizer 未联合优化**: SoundStream codec 作为冻结组件使用,未与 LLM 联合训练。codec-LM co-design (如 Wu et al., NAACL 2025 提出的) 可能进一步提升效果 [agent 解读]。

## 点评

本文的核心价值在于对 CPT 策略的 **受控实验设计**。通过比较 No Init / Text LLM Init / Speech CPT / Joint CPT 四种配置在同一架构上的表现,清晰展示了 CPT 的两个维度效果: (1) 任何 CPT 都显著改善生成质量 (TTS WER 从 17-18 降到 3.7); (2) 文本数据混入量决定理解-生成 trade-off。这种系统性消融在 concurrent works (ESPnet-SpeechLM, Qwen2.5-Omni, Kimi-Audio) 中是缺失的 [§II]。

S2S-Trans 作为 held-out task 的实验设计尤其精巧 — 它证明了 CPT 不只是改善微调稳定性,而是能让模型涌现出对 unseen 任务的泛化能力。Non-CPT 模型完全无法收敛这一结果,强有力地支持了 CPT 是 codec-based speech LLM 的 **必要** 而非可选组件。

不过,0.5B 的模型规模使得绝对性能缺乏工程价值 — ASR WER 3.7 和 TTS UTMOS 3.59 在 2025 年并不突出。论文的贡献更多是方法论和设计原则层面的,而非可直接部署的系统。

## 可复用的 idea

1. **CPT 中混入少量文本数据缓解灾难性遗忘**: 仅 10% 文本数据 (5% 通用 + 5% MT) 即可将 perplexity 退化从 31% 降到 5.7%。这一比例可作为其他 multimodal CPT 的起始参考 [§IV-B, Table VI]。

2. **Held-out task 验证跨模态泛化**: 故意在 CPT 中排除某个复合任务 (如 S2S-Trans = ASR + MT + TTS),用它作为泛化能力的 litmus test。如果 CPT 后模型能完成这个 unseen 任务,说明 CPT 确实实现了深层的跨模态对齐,而不仅仅是表面的分布适配 [§IV-A, §V-A]。

3. **Multi-level codec tokens 直接求和为单一 embedding**: 相比 flattening 或 delay pattern 等复杂策略,直接 sum 多层 codec embeddings 极其简单,配合并行预测 heads 即可工作。适合资源受限场景或快速原型 [§III]。

4. **序列格式设计**: `(Condition)(Prompt)(Target)` 的统一格式支撑了 6 个不同任务的训练,只需改变三个槽位的内容。boundary token 标记每个 speech/text 段的起止 [§IV-A]。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节含因果解释 (WHY),关键设计选择均回答了"为什么选这个" |
> | 可信赖 | pass | 数字型 claim 有 Table 出处标注,指标名使用正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 来源标注清晰,4 处 agent 推断有标注 |
> | 可定位 | pass | KB 背景含谱系定位 + 已有认知对比 + 创新判断 |
> | 不污染 | pass | no-kb-update 模式,未进行反向更新 |
> 
> Issues: 2 (high: 0, medium: 0, low: 2)
> 详见 `_review/BalancingUnderstanding-Generation-review.yml`

---

检索命中: [[SpeechLanguageModel]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechTokenizer]]✓ | 参考(待确认): [[CodecLanguageModel]], [[ModalityAdaptationforSpeechLLM]], [[Speech-LLMIntegrationTaxonomy]] | 未命中但可能相关: 无
