---
type: paper
tier: deep
title: "GOAT-TTS: LLM-based Text-To-Speech Generation Optimized via A Dual-Branch Architecture"
arxiv_id: "2504.12339"
source: "Sources/GOAT-TTS.pdf"
authors: [Yaodong Song, Hongjie Chen, Jie Lian, Yuxin Zhang, Guangmin Xia, Zehan Li, Genliang Zhao, Jian Kang, Yongxiang Li, Jie Li]
year: 2025
venue: "arXiv (Technical Report)"
tags: [TTS, LLM-based, zero-shot, dual-branch, modality-alignment, streaming, multi-token-prediction, dialect, data-augmentation]
concepts: ["[[LLM-based TTS]]", "[[Conditional Flow Matching]]", "[[Speech Tokenizer]]", "[[Modality Adaptation for Speech LLM]]", "[[Speech Language Model]]"]
models: ["[[Whisper]]", "[[CosyVoice 2]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[LLM-based TTS]], [[Conditional Flow Matching]], [[Speech Tokenizer]], [[Speech Language Model]], [[CosyVoice 2]], [[Zero-shot Speech Synthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: GOAT-TTS 属于 [[LLM-based TTS]] 范式中"coarse-to-fine"路线的变体: AR LLM 生成离散 speech tokens → [[Conditional Flow Matching]] 渲染 mel spectrogram → vocoder 合成波形。这一流水线与 [[CosyVoice 2]]、Llasa、FireRedTTS 等同代系统一致。但 GOAT-TTS 的独特之处在于双分支架构设计: 一条分支用于 modality alignment (连续语音表征 → LLM 理解),另一条用于 speech token generation (LLM → 离散 token)。这种设计直接回应了 [[LLM-based TTS]] 中三个已知痛点: (1) 离散化 prompt 损失声学特征, (2) 依赖精确转录文本, (3) 微调导致 LLM 语言知识灾难性遗忘。
>
> **已有认知**: KB 中 [[Modality Adaptation for Speech LLM]] [待确认] 概述了语音→LLM 适配的三种方案 (Conv downsampling / CTC-based / Q-Former),GOAT-TTS 的 speech encoder + CNN projector 属于第一类 convolutional downsampling。[[Speech Tokenizer]] 页明确区分了 semantic/acoustic/self-supervised 三类 tokenizer,GOAT-TTS 虽未指明使用哪种 codec,但其 speech token 经 flow-matching 解码为 mel,符合 coarse-to-fine 的 semantic token → CFM 路线。[[CosyVoice 2]] 的 FSQ-SenseVoice tokenizer + Qwen2.5 初始化 + chunk-aware CFM 是当前最强 streaming baseline 之一,GOAT-TTS 在 SEED 测试集上与之直接对比。
>
> **创新判断**: 对比 KB 已有系统,GOAT-TTS 的核心新意在于: (a) 用连续 acoustic embedding 而非离散 token 编码 speech prompt,绕过信息瓶颈; (b) layer-wise freezing (冻结 bottom-N, 微调 top-K) 替代传统 LoRA/全参微调,保护 LLM 语言知识; (c) 将方言数据合成作为主要应用场景验证,而非仅做标准 TTS benchmark。
>
> 检索命中: [[LLM-based TTS]], [[Conditional Flow Matching]], [[Speech Tokenizer]], [[Speech Language Model]], [[CosyVoice 2]], [[Zero-shot Speech Synthesis]] | 过滤: [[Modality Adaptation for Speech LLM]](pending-review), [[Whisper]](pending-review), [[Speaker Adaptation]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 双分支 LLM 架构解决 speech prompt 离散化信息损失 + 微调遗忘问题,用连续 acoustic embedding 对齐模态 + layer-wise freezing 保护语言知识,同时支持原生流式推理
> - **路线**: Speech prompt → Whisper-Small encoder → CNN projector → 连续 acoustic embeddings + Text tokens → LLM (bottom-N frozen, top-K fine-tuned) → discrete speech tokens → Flow-matching model → Mel → Waveform
> - **指标**: SEED test-zh CER 1.53% (vs CosyVoice2 1.45%), test-en WER 2.24% (vs CosyVoice2 2.57%), test-hard WER 7.83% (vs CosyVoice2-S 8.08%); 方言 ASR WER: 中原 15.82% (vs CosyVoice2 20.01%), 西南 13.0% (vs CosyVoice2 19.52%) [Table 2, 3]
> - **可借鉴**: layer-wise freezing (冻结 bottom-N + 微调 top-K) 保护 LLM 预训练知识的策略; 用 speech-text continuation pairs 做模态对齐训练; 用 TTS 合成方言数据做 ASR 数据增强验证
> - **局限**: 未报告 MOS/DNSMOS 等主观质量评估; 未报告 speaker similarity; 方言合成质量与 GT 仍有显著差距 (WER ~2x GT); 未开源; 流式推理延迟未量化

## 核心问题

GOAT-TTS 针对 LLM-based TTS 的三个已知问题提出解决方案 [§1]:

1. **Speech prompt 离散化信息损失**: 现有方法将 prompt 语音编码为离散 token,不可逆地丢失了音色、韵律、情感等副语言特征 [论文原文]
2. **对精确 prompt 转录文本的依赖**: 推理时需要 speech prompt 的精确文字转录,限制工业部署灵活性 [论文原文]
3. **LLM 语言知识灾难性遗忘**: 在 speech token prediction 上微调时,LLM 原有的文本理解能力急剧退化 [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

GOAT-TTS 由三个核心组件构成 [§2, Fig 1]:

1. **共享预训练 LLM backbone**: 分为 bottom-N 层 (frozen) 和 top-K 层 (分支各自独立)
2. **Modality-alignment branch (模态对齐分支)**: 负责建立语音连续表征与文本语义的双向关联
3. **Speech-generation branch (语音生成分支)**: 负责从文本生成离散 speech token

**推理流程** [§2.2]:
```
Speech prompt → Whisper-Small encoder → CNN projector (3-layer CNN + linear) → continuous embeddings
                                                                                    ↓
Text input ────────────────────────────────────────────────────────────────→ LLM (Bottom-N frozen + Top-K fine-tuned)
                                                                                    ↓
                                                                        discrete speech tokens
                                                                                    ↓
                                                                     Flow-matching model → Mel → Waveform
```

关键设计: LLMS (alignment branch 的 top-K) 和 LLMT (generation branch 的 top-K) 初始时共享权重,但在训练中独立演化 [Fig 1]。这意味着 LLM 的 bottom-N 层是两个分支的公共基础,而 top-K 层分化为理解和生成两个专门化的功能模块 [agent 解读]。

### 关键设计选择

#### 1. 连续 acoustic embedding 替代离散 prompt tokenization

**为什么不用离散 token 编码 prompt?** 离散量化不可逆地丢弃了副语言信息 (音色、韵律、情感等),导致合成语音质量退化 [论文原文, §1]。

**怎么做**: 使用预训练 Whisper-Small 编码器提取语音 prompt 的连续 latent 表征,再通过 CNN projector (3 层 CNN + 线性变换层) 映射为 LLM 兼容的 embedding [§2]。这保留了完整的声学特征信息,同时消除了对 prompt 转录文本的依赖 [论文原文]。

**为什么选 Whisper-Small?** 论文未明确说明选择理由 [agent 解读: 可能因为 Whisper 编码器在多语言+多方言场景有良好泛化性,且 Small 版本参数量适中 (244M encoder),不会过度增加计算负担]。

#### 2. Layer-wise parameter freezing

**为什么不全参微调?** 全参微调 speech token prediction 会导致 LLM 原有文本理解能力灾难性遗忘 [论文原文, §1]。

**怎么做**: 将 LLM 分为两部分 [§2.2]:
- **Bottom-N 层 (frozen)**: 保留预训练语言知识 (语法、语义、上下文理解)
- **Top-K 层 (fine-tuned)**: 针对 speech token prediction 任务优化

**为什么这个分层有效?** [agent 解读: Transformer LLM 的底层通常编码通用的语言结构知识 (syntax, semantics),高层编码任务特定的映射。冻结底层保护了共享的语言基础,而只微调高层实现了跨模态映射的针对性优化。这与 LLM 的 probing 研究一致: 底层 feature 更通用且可迁移。]

#### 3. Multi-token prediction

**为什么引入?** 减少原始 LLM 输出与微调模型 speech token 预测之间的频率差异,同时加速推理 [论文原文, §1]。

**怎么做**: 将当前步的输出 token embedding 与下一步的输入特征拼接,在冻结的上层网络参数中融合时序上下文信息,生成时序一致的 latent 表征来引导后续 speech token 生成 [§2.2]。这一设计天然支持流式文本输入 + 流式语音输出 [论文原文]。

### 训练策略

采用两阶段训练 [§2, Fig 1 right]:

**Stage I: Modality-Alignment Training [§2.1]**

目标: 建立语音连续表征与文本语义的跨模态对齐。

- **训练数据**: Speech-text continuation pairs — LLM 消费语音转录后生成文本续写,将同样的语音 embedding 输入模型,让它预测相同的文本续写
- **数据构建策略** (两种互补路线):
  1. 利用大规模 ASR 语料的转录文本,用 LLM 生成续写文本 [§2.1]
  2. 用 TTS 将语义连贯的句子转语音,同时用 LLM 生成对应续写文本 [§2.1]
- **方言/情感标注**: 在输入文本前添加自然语言描述符 (方言规格、情感线索) 以学习模态依赖表征 [§2.1]
- **分步**: Step I 先用中英数据建立 projector 基础; Step II 在平衡的多语言/方言/情感数据上进一步优化 encoder 和 projector [§2.1]
- **LLM 全程冻结** [§2.1]
- **损失函数**: text loss (续写文本的 cross-entropy) [Fig 1]

**Stage II: Speech-Generation Training [§2.2]**

目标: 端到端语音合成能力。

- **训练数据**: ~150k 小时真实世界数据,组织为 <text query, speech query, text response, speech response> 四元组 [§2.2]
- **分步**:
  - Step I (冷启动): 中英用 <text query, text response, speech response>,方言用 <speech query, text response, speech response>,联合训练 [Table 1]
  - Step II: 全部用 <speech query, text response, speech response> 训练 prompt-driven 能力 [Table 1]
- **参数策略**: Speech encoder + projector + bottom-N 层冻结,仅 top-K 层微调 [§2.2]
- **损失函数**: speech token 的 cross-entropy loss [§2.2]
- **论文指出**: 两步训练策略比直接用 <speech query, text response, speech response> 训练更稳定,尤其在跨语言场景 [论文原文, §2.2]

## 实验

| 指标 | 本文 (GOAT-TTS) | CosyVoice 2 | CosyVoice 2-S | FireRedTTS | F5-TTS | Llasa-8B | MaskGCT | E2 TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CER (%) ↓ | 1.53 | 1.45 | 1.45 | 1.51 | 1.56 | 1.59 | 2.27 | 1.97 | SEED test-zh | [Table 2] |
| WER (%) ↓ | 2.24 | 2.57 | 2.38 | 3.82 | 1.83 | 2.97 | 2.62 | 2.19 | SEED test-en | [Table 2] |
| WER (%) ↓ | 7.83 | - | 8.08 | 17.45 | 11.09 | 11.75 | 10.27 | 8.67 | SEED test-hard | [Table 2] |

**方言 ASR 数据增强实验 (开源语料 KeSpeech)**:

| 指标 | GOAT-TTS 合成 | CosyVoice 2 合成 | GT (原始数据) | Baseline (无方言) | 方言 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER (%) ↓ | 15.82 | 20.01 | 7.54 | 38.56 | 中原方言 | [Table 3] |
| WER (%) ↓ | 13.0 | 19.52 | 9.58 | 38.25 | 西南方言 | [Table 3] |

**方言 ASR 数据增强实验 (内部语料, 5 种方言)**:

| 方言 | GOAT-TTS WER(%) | GT WER(%) | Baseline WER(%) | 出处 |
| --- | --- | --- | --- | --- |
| 粤语 | 26.48 | 23.83 | 89.54 | [Table 4] |
| 东北话 | 6.52 | 5.55 | 13.63 | [Table 4] |
| 河南话 | 24.19 | 21.16 | 61.91 | [Table 4] |
| 上海话 | 17.34 | 15.6 | 76.81 | [Table 4] |
| 四川话 | 14.36 | 11.5 | 41.53 | [Table 4] |

**实验分析**:

- 在 SEED 标准测试集上,GOAT-TTS 在 test-zh 排第三 (CER 1.53%),仅次于 CosyVoice2 (1.45%) 和 FireRedTTS (1.51%) [Table 2]
- 在 test-hard 上排第二 (WER 7.83%),仅次于 CosyVoice2 的非流式版本,且优于 CosyVoice2-S 的流式版本 (8.08%) [Table 2]
- 论文指出部分 test-zh 样本被错误合成为方言口音 (发音正确但 ASR 识别不佳),导致 CER 偏高 [§3.1]
- 方言合成数据增强效果显著: 对比 baseline (无方言数据) WER 降低 >59%,且全面优于 CosyVoice2 的合成数据 [Table 3]
- 内部 5 方言实验达到 GT 性能的 82.5%~88.9% (粤语/东北/河南/上海),验证了方言合成数据的实用性 [Table 4]

## 局限性

1. **缺乏主观评估**: 未报告 MOS、DNSMOS、CMOS 等人工/半自动评估指标,无法判断合成语音的实际听感质量 [agent 解读]
2. **缺乏 speaker similarity 评估**: 作为 zero-shot TTS 系统,未报告与 prompt 说话人的相似度指标 (SIM-O/SIM-R),这是 [[SEED-TTS-Eval]] 标准评估维度之一 [agent 解读]
3. **流式延迟未量化**: 宣称支持"real-time streaming",但未报告首包延迟 (first-packet latency) 或 RTF,无法与 CosyVoice 2 streaming 版本的延迟性能做有效对比 [agent 解读]
4. **方言合成与 GT 差距**: 方言 ASR WER 与 GT 仍有 ~2x 差距 (如中原 15.82% vs GT 7.54%),论文归因于训练语料方言覆盖不足 [§3.2.1]
5. **架构细节不足**: 未说明使用的具体 speech codec/tokenizer 类型、LLM backbone 具体型号及参数量、bottom-N 和 top-K 的具体层数 [agent 解读]
6. **未开源**: 代码和模型均未公开 [agent 解读]
7. **消融实验缺失**: 未验证各组件 (连续 vs 离散 prompt、layer freezing vs 全参微调、multi-token prediction) 的独立贡献 [agent 解读]

## 点评

GOAT-TTS 的核心思想有价值: 用连续 acoustic embedding 保留 prompt 信息并消除转录依赖、用 layer-wise freezing 保护 LLM 知识、双分支让理解与生成各有专属参数空间。这三个设计分别回应了 LLM-based TTS 的三个真实痛点。

但作为技术报告,论文的实验验证偏弱:
- SEED benchmark 只报告了 CER/WER,缺少 speaker similarity 和自然度评估,这在同期工作 (CosyVoice 2、Llasa、MaskGCT) 中都是标配
- 主要实验亮点是方言 ASR 数据增强,但这更多验证了 TTS 合成数据的下游价值,而非 TTS 本身的质量
- 架构细节关键参数 (LLM 型号、bottom-N/top-K 的 N/K 值、codec 类型) 缺失,可复现性低
- 没有消融实验,无法区分各创新点的独立贡献

**定位**: 在 [[LLM-based TTS]] 谱系中,GOAT-TTS 是来自中国电信 TeleAI 的工业探索,聚焦于方言场景的实用价值而非刷新 SOTA。其双分支 + layer freezing 的设计理念有参考意义,但实验证据尚不足以充分验证这些设计的有效性。

## 可复用的 idea

1. **Layer-wise freezing 策略**: 冻结 LLM bottom-N 层 + 微调 top-K 层,在跨模态微调中保护语言知识。比 LoRA 更粗粒度但更直接,适合对 LLM 知识保留有强要求的场景 [§2.2]
2. **Speech-text continuation pairs 做模态对齐**: 让 LLM 从语音 embedding 生成与文本同等的续写,建立跨模态语义对齐。这个训练范式来自 BLSP (Wang et al., 2023),GOAT-TTS 验证了它在方言+情感场景的扩展性 [§2.1]
3. **两步冷启动训练**: Stage II 中先用 text query 冷启动中英能力、speech query 冷启动方言,再统一切换到 speech query — 比直接混合训练更稳定 [§2.2, Table 1]
4. **TTS 合成方言数据做 ASR 数据增强**: 用 TTS 合成特定方言的大规模语音数据来训练 ASR,相比收集真实方言数据成本大幅降低,WER 可达 GT 的 80%+ [Table 3, 4]

> [!review] 审阅 — pass (2026-06-03)
> **结论**: pass — 5 原则满足, 3 low issues
> - [low] traceability-gap: vocoder 类型未标注 (论文未指明)
> - [low] template-compliance: frontmatter models 语义确认正确
> - [low] weak-reusability: 方言数据增强 idea 可补充采样策略细节
> 详见 `_review/GOAT-TTS-review.yml`
