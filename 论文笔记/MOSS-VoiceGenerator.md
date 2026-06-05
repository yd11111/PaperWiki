---
type: paper
tier: deep
title: "MOSS-VoiceGenerator: Create Realistic Voices with Natural Language Descriptions"
arxiv_id: "2603.28086"
source: "Sources/MOSS-VoiceGenerator.pdf"
authors: [Kexin Huang, Liwei Fan, Botian Jiang, Yaozhou Jiang, Qian Tu, Jie Zhu, Yuqian Zhang, Yiwei Zhao, Chenchen Yang, Zhaoye Fei, Shimin Li, Xiaogui Yang, Qinyuan Cheng, Xipeng Qiu]
year: 2026
venue: "arXiv"
tags: [TTS, instruction-following, voice-design, natural-language-description, autoregressive, codec-LM, RVQ, cinematic-data, open-source, expressive-speech]
concepts: ["[[Instruction-GuidedSpeechSynthesis]]", "[[NaturalLanguageDescriptionforTTS]]", "[[CodecLanguageModel]]", "[[ResidualVectorQuantization]]", "[[LLM-basedTTS]]"]
models: ["[[论文笔记/MOSS-TTS|MOSS-TTS]]", "[[论文笔记/MOSS-TTSD|MOSS-TTSD]]", "[[论文笔记/VoiceSculptor|VoiceSculptor]]", "[[论文笔记/InstructTTSEval|InstructTTSEval]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]"]
tasks: ["[[InstructedSpeechGeneration]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: MOSS-VoiceGenerator 属于 [[NaturalLanguageDescriptionforTTS]] 中"完全无参考音频的 voice design"方向,同时也是 [[Instruction-GuidedSpeechSynthesis]] [待确认] 的实践 -- 用户提供自然语言 timbre/style 描述 + 合成文本,模型直接生成新音色的语音。在已有知识库的演进线上,它位于 PromptTTS (2023, 5 属性模板) → Parler-TTS (2024, 合成标注) → VoxInstruct (2024, 统一指令) → VoiceSculptor (2026, CoT+RAG 细粒度) 之后,核心差异点是**数据来源**: 使用影视数据替代录音棚数据来获取更真实的声音多样性。

底层架构属于 [[CodecLanguageModel]] [待确认] 的 delay pattern 路线,直接复用同团队 [[论文笔记/MOSS-TTS|MOSS-TTS]] 的架构 + [[论文笔记/MOSS-TTSD|MOSS-TTSD]] 的 MOSS-Audio-Tokenizer (16 层 RVQ)。在 [[LLM-basedTTS]] 的分类中,这是 "decoder-only Transformer + discrete codec tokens + 自回归生成" 的纯离散路线。

**已有认知对比**:
- [[ResidualVectorQuantization]]: MOSS-VoiceGenerator 使用 MOSS-Audio-Tokenizer 的前 16 层 RVQ codebook,与 MOSS-TTS/MOSS-TTSD 共享 tokenizer。KB 已记录 RVQ 层级信息结构 (前层 coarse → 后层 fine),delay pattern 是 MusicGen 提出的多层 RVQ 并行建模策略。
- [[Instruction-GuidedSpeechSynthesis]] [待确认]: KB 区分了 NL Description (content+description 分离) 和 Instruction-Guided (统一指令) 两种范式。MOSS-VoiceGenerator 采用前者 -- voice description 与 synthesis text 拼接输入,属于 description-based 而非统一指令范式。与 VoxInstruct (需参考音频) 和 VoiceSculptor (CoT 推理 + RAG 检索) 相比,MOSS-VoiceGenerator 更强调数据侧的差异化 (影视数据 → 真实感)。
- [[InstructedSpeechGeneration]]: 任务页已记录 InstructTTSEval benchmark 的三层评估体系 (APS/DSD/RP)。MOSS-VoiceGenerator 在该 benchmark 上与商业模型 (Gemini-TTS-Pro) 和开源模型 (Qwen3-TTS-VD, MIMO-Audio) 进行了对比。

**创新判断**: 相对 KB 已有知识,MOSS-VoiceGenerator 的核心贡献不在架构 (直接复用 MOSS-TTS),而在 **数据工程**: (1) 影视数据采集 + 去噪 + 质量过滤的完整 pipeline; (2) Speech-CLAP embedding 模型实现 style-guided mining 从中性数据中挖掘表达性片段; (3) 主观评估全面优于开源 baseline。

> 检索命中: [[ResidualVectorQuantization]]✓, [[LLM-basedTTS]]✓, [[InstructedSpeechGeneration]]✓ | 参考(待确认): [[Instruction-GuidedSpeechSynthesis]], [[NaturalLanguageDescriptionforTTS]], [[CodecLanguageModel]] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 复用 MOSS-TTS 架构 (Qwen3 1.7B + MOSS-Audio-Tokenizer 16 层 RVQ),以影视数据为核心训练 instruction-driven voice design 模型,主观评估全面优于 Qwen3-TTS-VD / MiniMax / MIMO-Audio
> - **路线**: NL Timbre Instruction + Synthesis Text → Qwen3 Text Tokenizer → Decoder-only LM (1.7B, delay pattern) → 16-layer RVQ Audio Tokens → MOSS-Audio-Tokenizer Decoder → Waveform
> - **指标**: InstructTTSEval EN DSD 82.0% / RP 68.7% [Table 1]; 主观 pairwise preference 全维度 win > 50% vs MiMo-Audio (63.1%), MiniMax (61.9%), Qwen3-TTS-VD (61.9%) [Fig 4]
> - **可借鉴**: (1) 影视数据 + 去噪后过滤 (DNSMOS >= 3.0, 保留率 5%→45%) 获取多样真实语音; (2) Speech-text alignment embedding 做 style-guided mining,从中性数据中定向挖掘表达性片段; (3) English instruction rewriting 将有效训练样本翻倍解决英文韵律不稳定
> - **局限**: 仅支持中英文; 英文数据量不足 (7K vs 18K h) 导致部分英文风格韵律弱; 去噪可能引入伪影 (残余呼吸噪声/高频平滑); 输出偶有不稳定; 无客观音质指标 (MOS/PESQ) 报告

## 核心问题

1. **为什么现有 voice design 模型的声音不够真实?** 现有模型主要在录音棚数据上训练,产生的语音干净规整但缺乏真实人声的"生活感" -- 自然呼吸节奏、节奏不规则性和自发情感变化等微妙特征 [§1]。

2. **影视数据能否弥补这个差距?** 论文的核心假设: 让模型接触真实世界的声学多样性 (real-world acoustic variation) 会产生感知上更自然的语音。影视内容天然包含多样场景、角色和表达性说话风格 [§2.2]。

3. **如何从大量有噪声的影视源中获取高质量训练数据?** 通过两阶段 pipeline: Phase 1 直接采集影视音频并去噪过滤标注; Phase 2 通过 style-guided embedding mining 从内部 TTS 数据中扩充表达性片段 [§2.2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MOSS-VoiceGenerator 的模型架构直接复用 MOSS-TTS [§2.1],由三部分组成:
1. **Text Tokenizer**: Qwen3 原生 text tokenizer,将 voice description + synthesis text 拼接编码
2. **Decoder-only Language Model**: 基于 Qwen3,采用 delay pattern 自回归生成多层 RVQ tokens
3. **MOSS-Audio-Tokenizer**: 音频 tokenizer/detokenizer,使用前 16 层 RVQ codebook

推理流程 [Fig 1]: voice description 和合成文本拼接后输入 LM,LM 以 delay pattern 生成多层 RVQ tokens (每个 codebook 层相对第一层前移 j-1 帧),生成到 EOS 后由 audio detokenizer 解码为波形。

**设计选择的 WHY**: 论文指出离散自回归架构的四个优势 [§2.1] [论文原文]:
- 将语音生成统一为序列预测,可直接复用 LLM 训练/推理框架
- 自回归建模自然捕捉长距离依赖,保证全局韵律和音色一致性
- 离散框架无需 diffusion/flow-matching 的迭代推理,部署更简单高效
- 文本和语音共享序列空间,使指令-语音联合建模更无缝

### 关键设计选择

**1. 数据来源: 影视数据 vs 录音棚数据**

这是本文最核心的设计选择。论文认为录音棚数据产生的语音"干净清晰但缺乏真实人声的生活感" [§1] [论文原文]。影视数据的优势在于内在的场景、角色和说话风格多样性 [§2.2]。

[agent 解读] 这个选择的 trade-off 是: 多样性 vs 质量。影视数据噪声大 (原始仅 5% 通过 DNSMOS 3.0),需要额外的去噪步骤,而去噪本身可能引入伪影 (如残余呼吸噪声、高频细节损失) [§5]。论文选择"最大化样本保留和语音多样性",接受"轻微质量损失" [§2.2]。

**2. 1.7B vs 8B 模型选择**

论文对比了 1.7B 和 8B backbone [§2.3]:
- 1.7B 在 instruction-following 质量上与 8B 可比
- 1.7B 在生成多样性上更好
- 混入 10K h TTS-base 数据 (无指令,纯 text-speech pairs) 对 8B 无显著增益

[agent 解读] 论文怀疑是数据量不足以充分训练 8B 模型。最终选择 1.7B 仅用 instruction 数据作为发布版本。

**3. English Prosody Augmentation**

英文子集仅 ~7K h (vs 中文 ~18K h),早期检查点英文韵律不稳定 (频繁不自然停顿) [§2.3]。解决方案: 对每个英文音频样本生成两个语义等价但词汇不同的 instruction 变体,有效翻倍英文训练信号量,无需额外音频采集。

### 数据 Pipeline

**Phase 1: 影视数据采集与标注 [§2.2]**

1. **Speaker Diarization**: DiariZen 分割不同说话人
2. **去噪 + 质量过滤**: MossFormer2_SE_48K 去噪 → DNSMOS >= 3.0 过滤 (原始 5% → 去噪后 45-50%)
3. **单说话人过滤**: MOSS-TranscribeDiarize 确保仅保留单说话人片段
4. **ASR 转录**: Qwen3-Omni-30B-A3B-Instruct (数字/符号/标点识别准确)
5. **语言过滤**: Whisper-large-v3 过滤非中英文
6. **Captioning**: Gemini-2.5-Pro 生成详细 speech caption
7. **Instruction 生成**: Qwen3-32B 将 caption 转为 NL timbre instruction

结果: ~5,000 h 标注影视音频

**Phase 2: 数据增强 [§2.2]**

两个互补策略:
1. **Fine-tune Speech Caption Model**: 在 Phase 1 数据上微调 Qwen3-Omni-30B 作为 caption 模型
2. **Style-Guided Audio Mining**: 训练 speech-text alignment embedding 模型 (Speech-CLAP),将风格指令和语音映射到共享嵌入空间。用 GPT-5 生成多样化风格指令作为 query,从内部 TTS base 数据 (主要是中性朗读) 中检索 top-50 匹配,每次检索后从候选池中移除,防止重复。增加 ~10K h 表达性数据。

结合额外众包配音数据,最终数据集: ~25,000 h (中文 18,025 h + 英文 7,047 h) [§2.2]。

### 训练策略

从 Qwen3 checkpoint 初始化,端到端训练 [§2.3]:
- 输入: NL timbre instruction + target transcript (通过 chat template 拼接)
- 训练目标: 标准 next-token prediction loss (条件于文本输入)
- 全参数更新,不使用 LoRA 等参数高效方法

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| APS (EN) | 68.2% | Gemini-TTS-Pro 87.6%, MIMO-Audio 80.6%, Qwen3-TTS-VD 78.4% | InstructTTSEval-EN | [Table 1] |
| DSD (EN) | 82.0% | Gemini-TTS-Pro 86.0%, Qwen3-TTS-VD 78.8%, MIMO-Audio 77.6% | InstructTTSEval-EN | [Table 1] |
| RP (EN) | 68.7% | Qwen3-TTS-VD 72.0%, Gemini-TTS-Pro 67.2%, MIMO-Audio 59.5% | InstructTTSEval-EN | [Table 1] |
| APS (ZH) | 78.0% | Gemini-TTS-Pro 89.0%, Qwen3-TTS-VD 84.3% | InstructTTSEval-ZH | [Table 1] |
| DSD (ZH) | 80.0% | Gemini-TTS-Pro 90.1%, Qwen3-TTS-VD 82.9% | InstructTTSEval-ZH | [Table 1] |
| RP (ZH) | 74.0% | Gemini-TTS-Pro 75.5%, Qwen3-TTS-VD 77.4% | InstructTTSEval-ZH | [Table 1] |
| Overall Pref vs MIMO-Audio | 63.1% win | 26.9% lose | Internal 100 pairs | [Fig 4] |
| Overall Pref vs MiniMax | 61.9% win | 31.2% lose | Internal 100 pairs | [Fig 4] |
| Overall Pref vs Qwen3-TTS-VD | 61.9% win | 33.1% lose | Internal 100 pairs | [Fig 4] |
| Naturalness vs MIMO-Audio | 59.4% win | 33.1% lose | Internal 100 pairs | [Fig 4] |
| Naturalness vs MiniMax | 50.3% win | 36.5% lose | Internal 100 pairs | [Fig 4] |
| Instruction Following vs MIMO-Audio | 60.0% win | 30.0% lose | Internal 100 pairs | [Fig 4] |

**客观评估分析**: 在 InstructTTSEval 上,MOSS-VoiceGenerator 在 DSD 任务上表现突出 -- EN-DSD (82.0%) 超过 Gemini-TTS-Pro 以外的所有模型。EN-RP (68.7%) 超越 Gemini-TTS-Pro (67.2%) 但低于 Qwen3-TTS-VD (72.0%)。在 APS (显式指定 12 属性的精细控制) 上不如 Gemini-TTS-Pro、MIMO-Audio 和 Qwen3-TTS-VD,是所有列出模型中最低的 [Table 1]。

[agent 解读] APS 弱但 DSD/RP 强这一模式可能反映了影视数据训练的特点: 影视数据中的语音自然携带复杂混合属性,模型更擅长从自由描述/角色场景中推断声学特征,但在逐一精确控制 12 个离散属性时不如专门为此优化的模型。

**主观评估分析**: pairwise preference study (100 对, 3 annotators) 在三个维度上均全面优于 MIMO-Audio-7B-Instruct、MiniMax Voice Design 和 Qwen3-TTS-VD [Fig 4]。Naturalness 维度的优势尤其说明影视数据训练的效果 -- 模型擅长生成日常对话式声音,包含自然停顿、犹豫和节奏变化 [§3.2]。

## 局限性

1. **语言覆盖有限**: 仅中英文,低资源语言未覆盖 [§5]
2. **中英文数据不平衡**: 英文 7K h vs 中文 18K h,部分英文说话风格韵律偏弱 [§5]
3. **去噪伪影**: 去噪前过滤的 pipeline 可能引入残余呼吸噪声或高频平滑 [§5]
4. **输出稳定性不足**: 生成偶尔不稳定,计划在未来版本增强鲁棒性 [§5]
5. **缺少客观音质指标**: 论文未报告 MOS/PESQ/UTMOS 等客观音质指标 [agent 观察],难以判断绝对音质水平
6. **主观评估规模有限**: 仅 100 对样本的 pairwise study [agent 观察]

## 点评

**正面**:
- **数据工程的系统性值得学习**: 从影视原始数据到最终 25K h 训练集的完整 pipeline (去噪→过滤→diarization→ASR→captioning→instruction 生成) 是一套可复用的方法论
- **Style-guided mining 是亮点**: 用 speech-text embedding 从中性数据中定向挖掘表达性片段,解决了"大规模中性数据有但表达性数据少"这一普遍痛点
- **完全开源**: 模型、数据 pipeline、评估工具包全部开源,在商业模型主导的 voice design 领域难得
- **主观评估结果强**: 在 naturalness 维度全面胜出,实证验证了"真实数据 → 真实声音"的假设

**负面/存疑**:
- **架构贡献为零**: 直接复用 MOSS-TTS 的 Qwen3 + delay pattern + MOSS-Audio-Tokenizer,无任何架构创新
- **数据假设需更多验证**: "影视数据 → 更自然声音" 的因果链缺少消融 (如: 相同数据量下录音棚 vs 影视的对比)
- **去噪带来的 trade-off 未充分量化**: 去噪提升保留率 (5%→45%) 但引入伪影,论文承认但未量化其影响
- **客观评估未达顶尖**: APS 上不如 Gemini-TTS-Pro 和 Qwen3-TTS-VD,说明精细属性控制仍有差距
- **1.7B vs 8B 的选择逻辑不够有力**: 称"怀疑数据量不足",但未给出支撑实验

## 可复用的 idea

1. **影视数据采集 pipeline**: 完整的 diarization → denoising → quality filtering → ASR → captioning → instruction 生成流程,可直接用于构建表达性语音数据集
2. **Style-guided embedding mining**: 训练 speech-text alignment embedding 模型,用风格描述作为 query 从大规模中性数据中检索表达性片段。这个思路可泛化到任何"属性稀缺但数据充足"的场景
3. **Instruction rewriting 增强少资源语言**: 对同一音频生成多个语义等价指令变体,有效翻倍训练信号,无需额外音频采集。可用于解决多语言 TTS 中的数据不平衡问题
4. **DNSMOS 阈值 + 去噪组合策略**: 先去噪再以 DNSMOS >= 3.0 过滤,在数据保留率和质量之间取得平衡。5% → 45% 的保留率提升数字值得参考

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节因果解释充分,设计选择 WHY 清楚 |
> | 可信赖 | pass-with-fixes | baseline 数字有 3 处混淆 (已修正) |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率 ~85% |
> | 可定位 | pass | KB 背景谱系定位具体,创新判断有对比基准 |
> | 不污染 | pass | 无 KB 安全风险 |
> 
> Issues: 4 (high: 1, medium: 1, low: 2)
> 详见 `_review/MOSS-VoiceGenerator-review.yml`
