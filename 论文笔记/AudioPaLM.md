---
type: paper
tier: deep
title: "AudioPaLM: A Large Language Model That Can Speak and Listen"
arxiv_id: "2306.12925"
source: "Sources/AudioPaLM.pdf"
authors: [Paul K. Rubenstein, Chulayuth Asawaroengchai, Duc Dung Nguyen, Ankur Bapna, Zalán Borsos, Félix de Chaumont Quitry, Peter Chen, Dalia El Badawy, Wei Han, Eugene Kharitonov, Hannah Muckenhirn, Dirk Padfield, James Qin, Danny Rozenberg, Tara Sainath, Johan Schalkwyk, Matt Sharifi, Michelle Tadmor Ramanovich, Marco Tagliasacchi, Alexandru Tudor, Mihajlo Velimirović, Damien Vincent, Jiahui Yu, Yongqiang Wang, Vicky Zayats, Neil Zeghidour, Yu Zhang, Zhishuai Zhang, Lukas Zilka, Christian Frank]
year: 2023
venue: "arXiv"
tags: [speech-LM, multimodal, speech-translation, ASR, S2ST, voice-transfer, zero-shot, decoder-only, PaLM, AudioLM]
concepts: ["[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[LLM-basedTTS]]", "[[AudioUnderstanding]]", "[[SemanticvsAcousticTokens]]", "[[Audio-LanguagePretraining]]"]
models: ["[[论文笔记/AudioLM|AudioLM]]", "[[论文笔记/SoundStorm|SoundStorm]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[LLM-basedTTS]], [[SemanticvsAcousticTokens]], [[Audio-LanguagePretraining]])
> AudioPaLM 在 [[SpeechLanguageModel]] 演进中处于 "continued pre-training" 阶段 (AudioLM → AudioPaLM → TWIST/SPIRIT-LM),是首批证明 text-only LLM 预训练权重可直接迁移到语音任务的大规模系统。[[SemanticvsAcousticTokens]] 将 AudioPaLM 归入 "semantic tokens + 后置 acoustic decoder" 路线,与 AudioLM 的串联策略一脉相承。[[SpeechTokenizer]] 记录了 AudioPaLM 所用的 w2v-BERT / USM 系列 tokenizer。[[LLM-basedTTS]] 的 Integration 视角将 AudioPaLM 定位为 "Semantic + Acoustic 联合路线",Table 1 显示此路线获最优 MOS。[[Audio-LanguagePretraining]] 提供了对比背景: AudioPaLM 属于 decoder-only 端到端路线而非 CLAP 式 two-tower 对比预训练路线。
> 检索命中: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[LLM-basedTTS]], [[SemanticvsAcousticTokens]] | 过滤: [[AudioUnderstanding]] (pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 text-only LLM (PaLM-2) 的词表扩展为 text+audio 联合词表,通过 fine-tuning 实现单一 decoder-only 模型在 ASR/AST/S2ST/TTS 多任务上的统一建模,SOTA speech translation + 零样本跨语言翻译 + 跨语言声音保持
> - **路线**: 音频 → w2v-BERT/USM semantic tokens (25Hz, 1024 vocab) → 扩展 PaLM-2 embedding matrix (text tokens + audio tokens) → decoder-only Transformer → text 或 audio tokens → SoundStorm/AudioLM 解码为波形
> - **指标**: CoVoST2 AST BLEU 37.8 (AudioPaLM-2, vs Whisper 29.1) [Table 2]; CVSS S2ST ASR-BLEU 32.5 (vs Translatotron 2 25.6) [Table 2]; FLEURS 零样本 AST BLEU 20.7 (无 AST 训练数据的语言, vs Whisper 19.6 有监督) [Table 3]; 主观 MOS 4.44 / SMOS 4.00 (vs Translatotron 2 3.96 / 3.51) [Table 4]
> - **可借鉴**: (1) 词表扩展法: 仅扩展 embedding matrix 即可将 text LLM 转为 multimodal LLM,架构零修改; (2) combined tasks (chain-of-thought 式多步解码) 显著提升复杂任务性能; (3) 文本 LLM 的翻译能力可直接迁移到语音翻译 (零样本跨语言)
> - **局限**: 未开源; 全参数 fine-tuning 必要(冻结权重不行); tokenizer 质量是瓶颈 (USM-v2 >> w2v-BERT); 加入 S2ST 任务会轻微降低 ASR/AST 性能; 8B 参数部署成本高

## 核心问题

AudioPaLM 试图回答: **如何在单一模型中统一语音理解和语音生成,同时继承 text LLM 的语言知识和 AudioLM 的声学生成能力?**

此前的方法存在根本限制 [§1-2]:
- **Encoder-decoder 方法** (Whisper, PaLI, Flamingo): 只能输出文本,无法生成语音 [§2.1]
- **AudioLM/SPEAR-TTS**: text 和 audio 词表不交互,只能单方向 (text→speech 或 speech→speech),无法统一 [§2.2]
- **SpeechLM (Hassid et al.)**: 用 text LLM 初始化但完全替换词表为 audio tokens,丢失了多模态能力 [§2.2]
- **级联系统 (ASR→MT→TTS)**: 丢失副语言信息 (说话人身份、韵律),累积错误,高延迟 [§2.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

AudioPaLM 的设计核心是极简的词表扩展 [§3, Fig 1]:

1. **Audio Tokenizer**: 将音频编码为离散 semantic tokens [§3.1]
   - w2v-BERT (多语言版) → k-means (K=1024) → 25 Hz semantic tokens
   - 或 USM encoder (2B 参数) → k-means → 25 Hz tokens
   - USM-v2: 增加 ASR 辅助 loss 训练的增强版 tokenizer
2. **Multimodal Decoder**: PaLM-2 8B decoder-only Transformer [§3.2]
   - 将 embedding matrix E 从 t x m 扩展为 (t+a) x m,新增 a 个 audio token embedding (初始化为 0)
   - 输入/输出共享 embedding (E' = E^T),与 PaLM 架构一致
   - 架构本身完全不变,只改 embedding 尺寸
3. **Audio Decoder**: SoundStorm 或 AudioLM stages 2+3,将 semantic tokens → SoundStream acoustic tokens → 波形 [§3.3]

### 关键设计选择

**为什么"扩展词表"而非"替换词表"?** [论文原文]
- SPEAR-TTS 和 SpeechLM 将 text 和 audio 词表分开 (encoder-decoder) 或完全替换 [§2.2],导致模型只能做单方向映射
- AudioPaLM 将 text 和 audio tokens 放入同一词表,使模型可以自由地从任何模态输入到任何模态输出 [§3.2]
- [agent 解读] 这本质上是将"多模态"问题简化为"更大词表的语言模型"问题,最大化复用 LLM 基础设施

**为什么必须全参数 fine-tuning?** [论文原文]
- 论文发现无法像 Flamingo 那样冻结大部分权重只训练 adapter [§3.2, §6]
- [agent 解读] audio tokens 是全新的 embedding (初始化为 0),模型需要重新学习 attention pattern 来处理这些新 token,仅训练 embedding 层不足以学会跨模态交互

**为什么 combined tasks (多步解码) 有效?** [论文原文]
- 对于 S2ST 这样的复杂任务,直接从源语音 audio tokens 映射到目标语音 audio tokens 太难 [§3.4]
- Combined task 让模型先输出源文本 (ASR),再输出目标文本 (MT),最后输出目标语音 (TTS),类似 chain-of-thought [§3.4]
- 关键: 这不是 pipeline — 模型在每一步都能 attend 到输入和所有先前输出,因此声学信息 (韵律、说话人身份) 可以跨步传递 [§3.4]
- 实验验证: combined tasks 使 AST BLEU 从 18.5 → 22.1 (USM-v1) 和 26.9 → 30.5 (USM-v2) [Table 8]

**为什么 text LLM 的翻译能力可以迁移到语音?** [论文原文]
- PaLM-2 在预训练中见过大量平行翻译文本 [§5.4.8]
- Fine-tuning 后,模型只需学会 audio → internal representation 的映射,翻译能力本身来自预训练 [§5.2]
- 证据: 在完全没有 AST 训练数据的 26 种语言上,AudioPaLM-2 的零样本 AST BLEU 为 20.7,甚至超过见过 40.6K 小时 AST 数据的 Whisper (19.6) [Table 3]
- PaLM → PaLM-2 升级导致零样本 AST 提升 107% (10.0 → 20.7) [Table 3],直接反映底层 text model 翻译能力的差异

**Task tag 设计** [论文原文]
- 用简单文本标签指定任务: `[ASR French]`, `[S2ST English French]`, `[ASR AST S2ST English French]` (combined) [§3.4]
- 标签用标准文本 tokenizer 编码,不引入特殊 token [§3.4]
- 实验发现可读性更强的标签 (如 "transcribe the following French audio") 与简短标签性能无差异 [§3.4]

### 训练策略

**数据混合** [§3.5, Table 1]:
- AST mixture: CoVoST2, VoxPopuli, CommonVoice 11, YouTube ASR (自动标注), WMT/TED MT
- S2ST mixture: 上述 + TTS 任务 + S2ST 任务 (VoxPopuli, CVSS, WMT/TED 合成, PaLM MT 合成)
- 大量使用合成数据: WMT/TED 文本经 TTS 合成为语音; PaLM-2 翻译 + 先前 AudioPaLM 合成目标语音 [Table 1]

**训练配置** [§3.6]:
- Adafactor optimizer, 恒定学习率 5e-5, dropout 0.1
- Input loss masking (仅对输出计算 loss)
- PaLM-2 实验: dropout 0.2, 学习率 linear ramp-up → 1e-4 → exponential decay → 1e-5 [§5.4.8]

**Voice conditioning** [§3.3]:
- 3 秒源语音 prompt (semantic tokens + SoundStream tokens) 作为 voice conditioning
- 使翻译后的语音保持原说话人的声音 [§3.3]
- 当原始音频 < 3 秒时,重复填充到 3 秒 [§3.3]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| AST BLEU | 37.8 (AudioPaLM-2 AST) | 29.1 (Whisper 1.5B) / 30.7 (USM-M) | CoVoST2 | [Table 2] |
| S2ST ASR-BLEU | 32.5 (AudioPaLM S2ST) | 25.6 (Translatotron 2) | CVSS | [Table 2] |
| ASR WER | 9.8 (AudioPaLM-2 AST) | 8.1 (MAESTRO) / 13.6 (Whisper) | VoxPopuli | [Table 2] |
| 零样本 AST BLEU (ASR-only 语言) | 20.7 (AudioPaLM-2) | 19.6 (Whisper, 有监督) | FLEURS | [Table 3] |
| 主观 MOS (音质) | 4.44 | 3.96 (Translatotron 2) / 3.88 (CVSS-T GT) | CVSS-T | [Table 4] |
| 主观 SMOS (声音相似度) | 4.00 | 3.51 (Translatotron 2) / 3.70 (CVSS-T GT) | CVSS-T | [Table 4] |
| 客观声音相似度 (cosine) | 0.40 | 0.18 (Translatotron 2) / 0.24 (CVSS-T GT) | CVSS-T | [Table 4] |

**核心 ablation 发现**:

| 因素 | 对比 | 影响 | 出处 |
| --- | --- | --- | --- |
| 预训练 vs 从头训练 | PaLM 8B finetuned vs 8B scratch | AST BLEU 18.4 vs 6.9, ASR WER 40.2 vs 63.3 | [Table 6] |
| Tokenizer 选择 | USM-v2 vs USM-v1 vs w2v-BERT | AST BLEU 26.9 vs 18.5 vs 15.2 | [Table 7] |
| Combined vs direct tasks | combined vs direct (USM-v2) | AST BLEU 30.5 vs 26.9 | [Table 8] |
| 多任务训练 (ASR+AST vs AST-only) | 加 ASR | AST BLEU 18.5 vs 16.0 | [Table 5] |
| 加 S2ST 任务 | AST+ASR+S2ST vs AST+ASR | 获得 S2ST 能力但 AST BLEU 降 2.7 | [Table 9] |
| 数据量 scaling | CoVoST2 only → full mixture | AST BLEU 30.5 → 35.4 | [Table 10] |
| SoundStorm vs AudioLM decoder | S2ST 解码 | ASR-BLEU 32.5 vs 31.2 | [Table 11] |
| PaLM-2 vs PaLM | 底层 LLM | AST BLEU 37.8 vs 35.4 (full data) | [Table 12] |
| 模型规模 (PaLM-2) | 128M / 1B / 8B | AST BLEU 18.3 / 31.6 / 37.8 | [Table 13] |

## 局限性

1. **Tokenizer 依赖** [§6]: 模型性能强烈依赖 audio tokenizer 质量,USM-v2 >> w2v-BERT (BLEU 差距 >10),tokenizer 改进空间仍大
2. **全参数 fine-tuning 必要** [§6]: 无法像 Flamingo 冻结大部分权重,意味着无法保证原有 text 能力不退化
3. **S2ST 与 AST 的 capacity 冲突** [Table 9]: 加入 speech 输出任务会挤占 text 输出任务的模型容量,导致 ASR/AST 轻微下降
4. **闭源**: 模型、数据、tokenizer 均未公开,难以复现
5. **评估覆盖不足** [§6]: 语音生成的 benchmark 不如文本成熟,评估体系待完善
6. [agent 解读] 25 Hz token rate 意味着约 6-8 个 audio tokens 对应 1 个 text token [§5.4.2],长音频处理时序列长度仍是挑战

## 点评

AudioPaLM 的核心贡献不在于某个技术模块的创新,而在于一个优雅的系统级 insight: **text LLM 和 audio LM 可以通过最小修改 (扩展 embedding matrix) 融合为统一的多模态 LM,且 text 预训练的知识 (尤其是翻译能力) 可以直接迁移到语音任务**。

这个 insight 的说服力来自两个实验:
1. **预训练 vs 从头训练** [Table 6]: BLEU 差距近 3 倍,说明 text LLM 的语言知识确实被音频任务利用了
2. **零样本翻译** [Table 3]: 在完全没有某些语言的 AST 数据时,模型仍能翻译,且 PaLM-2 相比 PaLM 的翻译能力提升直接体现在语音任务上 (+107%)

Combined tasks 的设计也很有洞察: 它不是简单的 pipeline (ASR→MT→TTS),而是让模型在每步都能回看原始音频,因此韵律和说话人信息可以"穿越"中间文本步骤影响最终语音输出。这解释了为什么 AudioPaLM 在 voice preservation 上显著优于级联系统。

从知识库视角看,AudioPaLM 验证了 [[SpeechLanguageModel]] 概念页中 "continued pre-training" 策略的核心假设: text LLM 的语言知识可以加速语音任务学习。后续的 TWIST (2024) 和 SPIRIT-LM (2024) 在更小规模上进一步验证了这一路线。

**时代局限**: AudioPaLM 的 tokenizer (w2v-BERT/USM) 仅生成 semantic tokens,需要外部 AudioLM/SoundStorm 恢复声学细节。2024-2025 年的混合 tokenizer (Mimi, SpeechTokenizer, LM-SPT) 和连续 VAE tokenizer (LatentLM, CLEAR) 路线有望消除这一瓶颈。

## 可复用的 idea

1. **Embedding matrix 扩展法**: 将新模态的离散 token 直接追加到 LLM 词表末尾,初始化为 0,全参数 fine-tune。简单但有效,适用于任何需要扩展 LLM 模态的场景。
2. **Combined tasks as chain-of-thought**: 将复杂 cross-modal 任务拆解为多步,在同一次自回归解码中串联 (非 pipeline),让模型在后续步骤中 attend 到先前步骤和原始输入。可推广到任何需要"中间推理步骤"的生成任务。
3. **底层 LLM 能力的零样本迁移**: 如果底层 text LLM 有某种能力 (如翻译),可以通过模态桥接将该能力迁移到新模态。换言之,**选择更好的底层 LLM 比改进上层架构可能更有效**。
4. **多任务训练的互惠效应**: ASR 任务帮助 AST (+2.5 BLEU),因为 ASR 帮模型学会将新的 audio tokens 与已有的 text 知识对齐 [§5.4.1]。在多模态训练中,看似辅助的任务可能起到"模态桥接"作用。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 核心 insight 清晰: 词表扩展 + text LLM 知识迁移; 关键设计选择回答了 WHY |
> | 可信赖 | pass | 数字 claim 均有 [Table N] / [§X.X] 标注; 指标方向正确 |
> | 可区分 | pass | [论文原文] / [agent 解读] 标注覆盖率 > 80% |
> | 可定位 | pass | KB 背景定位清晰,在 SpeechLM 演进线和 semantic-acoustic 路线中有明确位置 |
> | 不污染 | pass-with-fixes | AudioPaLM 已在多个概念页 key_papers 中被引用,无需新建实体页; 反向更新需追加描述 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> 详见 `_review/AudioPaLM-review.yml`
