---
type: paper
tier: deep
title: "CoSTA: Cognitive-State-Conditioned TTS Data Augmentation Using ASR Transcripts for Alzheimer's Disease Detection"
arxiv_id: "2606.06170"
source: "Sources/Cognitive-State-ConditionedTTS.pdf"
authors: [Yin-Long Liu, Yuanchao Li, Yiming Wang, Yue Li, Rui Feng, Jiaxin Chen, Shaobo Liu, Liu He, Yuang Chen, Jiahong Yuan, Zhen-Hua Ling]
year: 2026
venue: "Interspeech 2026"
tags: [TTS, data-augmentation, Alzheimer-disease, AD-detection, speech-pathology, CosyVoice2, F5-TTS, ASR, WavLM, cognitive-state, flow-matching]
concepts: ["[[ConditionalFlowMatching]]", "[[Instruction-GuidedSpeechSynthesis]]", "[[LLM-basedTTS]]", "[[NaturalLanguageDescriptionforTTS]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/WavLM|WavLM]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[模型库/CosyVoice2|CosyVoice 2]]✓ | 参考: [[Instruction-GuidedSpeechSynthesis]][待确认], [[NaturalLanguageDescriptionforTTS]][待确认], [[模型库/WavLM|WavLM]][待确认] | 未命中但可能相关: 无

**谱系定位**: 本文将 TTS 技术从通常的"高保真语音合成"目标偏转到"病理语音数据增强"方向。核心思路是利用已有的高质量 TTS 模型 (CosyVoice2, F5-TTS) 作为数据生成器,通过条件控制使其合成具有特定认知状态特征 (AD vs HC) 的语音,解决临床数据稀缺问题。

**已有认知**:
- **CosyVoice2** [confirmed]: 阿里通义的 LLM + chunk-aware flow matching TTS,支持 instruction fine-tuning。本文直接利用其 instruction fine-tuning 能力来注入认知状态控制。CosyVoice2 在知识库中已有 30+ 篇下游引用,但作为临床数据增强工具是首次出现。
- **F5-TTS** [未建页]: 基于 Flow Matching + DiT 的非自回归 TTS,本文通过新增 Cognition Processing 模块实现认知状态条件化,修改方式与 CosyVoice2 的 instruction 路线不同。
- **Flow Matching (CFM)** [confirmed]: F5-TTS 的核心生成算法,将 Gaussian noise 变换为目标 mel spectrogram。本文中 F5-TTS 的 FM loss 保持不变,仅在 conditioning 输入中新增认知标签嵌入。
- **WavLM** [pending-review]: 本文用 WavLM Large 作为 AD 检测模型的特征提取器,利用其层级表示 (learnable softmax weighted fusion) + attentive temporal pooling。

**创新判断**: 本文的核心创新不在 TTS 模型本身,而在 (1) 将 TTS 的可控性从情感/风格扩展到认知状态这一临床维度,以及 (2) 系统性地验证了 ASR 转录文本相比人工转录作为 TTS 输入在数据增强中的意外优势。

## 速查

> [!summary] 速查
> - **一句话**: 用认知状态条件化的 CosyVoice2/F5-TTS 合成 AD/HC 语音做数据增强,配合 ASR 转录文本池,将 ADReSS 检测准确率提升 4.16% 至 85.83%
> - **路线**: ASR 模型集合 → 36 种转录文本 + MT → CS-Cond TTS (CosyVoice2 instruction/F5-TTS cognition label) → 合成语音 → 与原始数据混合 → WavLM-based AD 检测器
> - **指标**: 85.83% accuracy (ADReSS test, audio-only) vs 81.67% baseline (+4.16%); CS-Cond CosyVoice2 MCD 5.436 vs pretrained 6.854 (AD) [Table 1]; 最优增强倍率 2x [Fig 3]
> - **可借鉴**: (1) 用 instruction fine-tuning 实现病理语音条件化的最小改动方案; (2) ASR 错误作为"有益噪声"的数据增强思路; (3) 多 ASR 模型集合构建转录文本多样性
> - **局限**: 仅验证于 ADReSS 单一数据集 (108 训练 / 48 测试); 仅二分类 (AD vs HC); TTS 模型的病理特征保真度未深入分析; 过度增强 (>2.5x) 导致性能下降

## 核心问题

1. **数据稀缺**: 语音 AD 检测受制于隐私法规和患者可获得性,ADReSS 仅含 108 个训练样本 (~1.7h),模型极易过拟合
2. **传统增强的局限**: 信号级扰动 (加噪/变调/变速) 不引入新语义内容,也不建模病理特有的语音特征 (如非自然停顿、含混发音),甚至可能降低检测性能 (pitch shifting -2.5%) [Table 4]
3. **标准 TTS 的不适配**: 标准 TTS 优化目标是清晰度和自然度,会自动"修正"不流畅和韵律异常,从而抹去 AD 语音中的声学生物标记 [§1]
4. **文本源选择**: 用 TTS 做增强时,用人工转录 (MT) 还是 ASR 转录作为输入文本,此前未被系统研究

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CoSTA 是一个四组件框架 [Fig 1]:
1. **CS-Cond TTS 模型开发**: 将 CosyVoice2 和 F5-TTS 适配为可按认知状态 (AD/HC) 条件化的语音合成器
2. **多样化转录文本池构建**: 1 份 MT + 36 份 ASR 转录 (18 pretrained + 18 fine-tuned ASR 模型)
3. **语音增强策略**: 训练数据增强 + 测试时增强 (TTA)
4. **AD 检测模型**: WavLM-based 分类器

### 关键设计选择

**CS-Cond CosyVoice2 [§2.1.1]**: 利用 CosyVoice2 的 instruction fine-tuning 能力,设计自然语言指令 I^(c) 描述 AD/HC 语音特征。AD 指令包含"older patient with Alzheimer's disease...unnatural pauses, imprecise articulation"等描述,HC 指令包含"healthy adult...natural spoken style, clear stable voice, normal fluent pace"等描述。指令与目标 MT 通过 `<|endofprompt|>` 分隔符拼接后,经 text embed 模块编码。分别在 AD 和 HC 子集上微调,得到 CosyVoice2-AD 和 CosyVoice2-HC 两个专门化变体。

[agent 解读] 这种方法的优势在于**零架构修改** — 完全复用 CosyVoice2 已有的 instruction following 管线,仅设计适当的指令文本并分组微调即可。代价是 AD/HC 的声学差异被隐式编码在指令理解中,模型对病理特征的捕获取决于 instruction tuning 的泛化能力。

**CS-Cond F5-TTS [§2.1.2]**: 与 CosyVoice2 的 instruction 路线不同,F5-TTS 采用显式条件注入。新增 Cognition Processing 模块 (ConvNeXtv2 layers + RoPE encoding),将离散认知标签 l_c ∈ {Alzheimer, Health} 映射为 dense embedding e_c。Feature Aggregation 模块融合四路输入 (cognition embedding e_c, text embedding e_text, reference mel x_ref, noisy mel x_t),送入 DiT backbone 预测 flow。在混合 AD+HC 数据上统一训练一个模型 (而非 CosyVoice2 的分别训练两个)。

[agent 解读] CosyVoice2 用两个模型 (AD/HC 各一个),F5-TTS 用一个模型 + 标签条件化,这是 AR vs NAR 架构下不同的条件注入策略选择。F5-TTS 的显式标签嵌入在理论上更易控制,但需要额外的架构模块; CosyVoice2 的 instruction 方式更灵活但更不确定。

**ASR 转录文本池 [§2.2]**: 选取 4 大 SSL ASR 家族 (Wav2Vec2, HuBERT, WavLM, Whisper) 共 18 个预训练模型,用 DementiaBank 子集 (WLS, Lu, Kempler, ~3h) fine-tune 后得到 18 个微调版本,总计 36 个 ASR 模型。用全部 36 个模型转录 ADReSS 数据集,得到每个样本 36 份 ASR 转录 + 1 份 MT = 37 份文本源。WER 跨越 26.36%–68.55% 的广泛范围 [Fig 2],提供了多样化的转录质量和错误分布。

[论文原文] 作者的核心论点是: ASR 错误并非随机噪声,而是反映了 AD 患者的病理性声学特征 (如发音含混),这些错误在 TTS 合成中被保留,增加了训练数据的多样性并注入了诊断相关的病理线索 [§4.3]。

**增强策略 [§2.3]**:
- *Self-Reference (2x)*: 用说话人自己的语音和转录作为 TTS 参考,保留原始音色但引入转录文本变化
- *Intra-Class Cross-Synthesis (>2x)*: 从同类 (AD/AD 或 HC/HC) 中随机采样不同说话人的参考语音,组合语言内容与不同音色,进一步多样化

**测试时增强 (TTA) [§2.3.2]**: 在测试阶段,由于无法知道真实认知类别,不使用 CS-Cond TTS。改为微调一个不带认知指令条件的 zero-shot CosyVoice2,用 ASR 转录测试语音再合成一个变体,将原始和合成语音的分类概率取平均。

### 训练策略

- **CosyVoice2 微调**: Adam, lr=1e-5, 动态 batch (max 2000 frames/batch), 仅训练 1 epoch [§3.2]
- **F5-TTS 微调**: AdamW, lr=1e-5, batch=8, 训练 40 epochs [§3.2]
- **ASR 微调**: AdamW, lr=1e-5, batch=8, 训练 20 epochs [§3.2]
- **AD 检测**: AdamW, lr=5e-5, batch=8, 训练 30 epochs, 5 次独立运行取平均 [§3.2]
- **TTS 训练数据**: ADReSS 训练集按 5:1 划分为 TTS 训练集 (45 AD + 45 HC) 和 TTS 测试集 (9 AD + 9 HC),语音切分为 ~30s 片段 [§3.1]
- 所有实验在 NVIDIA A800 80GB 上进行

## 实验

| 指标 | 本文 (最佳) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| AD detection accuracy (audio-only) | 85.83% | 81.67% | ADReSS test | [Table 3] |
| CS-Cond CosyVoice2-AD MCD | 5.436 | 6.854 (pretrained) | TTS test set | [Table 1] |
| CS-Cond CosyVoice2-AD FAD | 2.192 | 8.542 (pretrained) | TTS test set | [Table 1] |
| CS-Cond F5-TTS-AD MCD | 5.734 | 6.187 (pretrained) | TTS test set | [Table 1] |
| CS-Cond CosyVoice2 surpass baseline ratio | 28/37 | 7/37 (pretrained) | ADReSS | [Table 2] |
| CS-Cond F5-TTS surpass baseline ratio | 24/37 | 16/37 (pretrained) | ADReSS | [Table 2] |
| ASR > MT ratio (CS-Cond CosyVoice2) | 20/36 | — | ADReSS | [Table 2] |
| TTA improvement (avg) | +0.98% | — | ADReSS | [Table 3] |
| vs prior best (AW-HuBERT) | 85.83% | 81.67% | ADReSS test | [Table 4] |
| vs noise addition DA | 85.83% | 82.50% | ADReSS test | [Table 4] |

**关键发现**:
1. **CS-Cond >> Pretrained**: CS-Cond 模型在 TTS 客观指标上全面优于 pretrained 版本,CosyVoice2-AD FAD 从 8.542 降至 2.192 (74% 下降) [Table 1],证实认知状态条件化使合成语音声学特征显著接近真实病理语音
2. **ASR > MT**: 在 CS-Cond CosyVoice2 中 20/36 ASR 配置超越 MT 驱动增强 [Table 2],最高配置 (fine-tuned w2v960 large lv) 达到 85.00% (+3.33%) vs MT 82.50% (+0.83%)
3. **增强倍率**: 最优在 1.5x-2.5x 范围,最佳单点为 2x,>2.5x 性能下降 [Fig 3]。[论文原文] 作者归因于过高比例的合成数据使模型过拟合于 TTS 系统的生成特征而非真实病理特征
4. **CosyVoice2 > F5-TTS**: 在 TTS 客观指标上 CS-Cond CosyVoice2 全面优于 CS-Cond F5-TTS [Table 1],在 AD 检测增强效果上 CosyVoice2 也更优 (28/37 vs 24/37)
5. **传统 DA 效果有限**: noise addition +0.83%, time stretching +0.41%, pitch shifting -2.50% [Table 4]

## 局限性

1. **数据集单一性**: 仅在 ADReSS (108 train / 48 test, Cookie Theft 任务) 上验证,泛化到其他 AD 数据集 (如 ADReSSo, Pitt Corpus) 或其他语言未知
2. **二分类简化**: 仅区分 AD vs HC,未覆盖 MCI (轻度认知障碍) 或 AD 严重程度分级
3. **病理特征分析缺失**: 论文未分析 CS-Cond TTS 实际学到了哪些病理声学特征 (停顿模式? 发音含混度? 韵律异常?),缺乏可解释性
4. **过度增强退化**: >2.5x 增强导致性能下降,说明合成语音与真实病理语音之间仍有 domain gap
5. **TTA 依赖 ASR**: 测试时增强需要预选一个"好的" ASR 模型,增加了超参数选择的复杂性
6. **CosyVoice2 instruction 设计**: AD/HC 指令文本的设计是手工的,不同指令措辞对结果的敏感性未探讨

## 点评

**方法论视角**: CoSTA 的核心洞察 — "ASR 错误是 AD 声学特征的文本级映射" — 虽然有实验支持 (ASR-driven augmentation 在多数情况优于 MT-driven),但因果关系尚不确定。ASR 错误也可能仅仅因为增加了文本多样性 (数据正则化效果) 而非真正编码了病理线索。需要控制实验区分这两种解释。

**TTS 应用视角**: 本文展示了现代 TTS 系统 (CosyVoice2, F5-TTS) 的可控性在临床场景的新用途。CosyVoice2 的 instruction fine-tuning 路线特别优雅 — 几乎零改动就实现了从"合成自然语音"到"合成病理语音"的转换。这种"最小修改"策略值得在其他条件化合成场景借鉴。

**临床落地视角**: 85.83% 的 audio-only 准确率在学术上有进步,但距离临床筛查的灵敏度/特异度要求仍有差距。更关键的限制是 ADReSS 数据集本身的规模 (48 个测试样本),统计功效有限。

**与 KB 已有知识的关联**: CosyVoice2 在知识库中已有 30+ 篇引用,但均集中在 TTS 质量/可控性/效率领域。本文首次将 CosyVoice2 用于非 TTS 的下游临床任务,展示了高质量 TTS 作为"通用数据生成器"的潜力。

> [!review] 审阅状态
> 结论: pass-with-fixes | high: 0 | medium: 2 | low: 2
> 审阅报告: [[_review/Cognitive-State-ConditionedTTS-review.yml]]
> 日期: 2026-06-08

## 可复用的 idea

1. **Instruction fine-tuning 做条件化合成**: 无需修改模型架构,仅通过设计描述性 instruction + 分组微调,即可让通用 TTS 合成特定属性的语音。可迁移到其他属性 (方言、年龄段、疾病类型)
2. **ASR 错误作为有益扰动**: 多个 ASR 模型的转录文本池作为 TTS 输入,提供了一种基于语言学的数据增强策略,可能适用于其他数据稀缺的语音分类任务 (如构音障碍检测、口吃检测)
3. **CS-Cond + cross-synthesis**: 认知标签条件化 + 类内跨说话人合成的组合策略,在保持类别一致性的同时最大化数据多样性
4. **Cognition Processing 模块**: F5-TTS 中新增的 ConvNeXtv2 + RoPE 标签嵌入模块,是一种轻量的离散条件注入方式,可迁移到其他 flow matching TTS 的条件控制场景
5. **TTA 概率平均**: 用 TTS 合成测试语音的变体做 test-time augmentation,在其他小数据语音分类任务中可能有用

---

检索命中: [[ConditionalFlowMatching]], [[LLM-basedTTS]], [[模型库/CosyVoice2|CosyVoice 2]] | 参考: [[Instruction-GuidedSpeechSynthesis]][待确认], [[NaturalLanguageDescriptionforTTS]][待确认], [[模型库/WavLM|WavLM]][待确认] | 未命中但可能相关: 无
