---
type: paper
tier: deep
title: "Affectron: Emotional Speech Synthesis with Affective and Contextually Aligned Nonverbal Vocalizations"
arxiv_id: "2603.14432"
source: "Sources/Affectron.pdf"
authors: [Deok-Hyeon Cho, Hyung-Seok Oh, Seung-Bin Kim, Seong-Whan Lee]
year: 2026
venue: "arXiv"
tags: [TTS, emotion, nonverbal-vocalization, data-augmentation, codec-LM, expressive-speech, VoiceCraft, EnCodec]
concepts: ["[[EmotionControlinTTS]]", "[[CodecLanguageModel]]", "[[ProsodyModeling]]", "[[Self-SupervisedSpeechRepresentation]]", "[[MaskedGenerativeModeling]]"]
models: ["[[模型库/EnCodec|EnCodec]]", "[[模型库/CosyVoice2|CosyVoice2]]", "[[模型库/CosyVoice3|CosyVoice3]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页: [[EnCodec]], [[ProsodyModeling]], [[EmotionControlinTTS]], [[CodecLanguageModel]], [[MaskedGenerativeModeling]], [[Self-SupervisedSpeechRepresentation]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[EnCodec]]✓, [[ProsodyModeling]]✓ | 过滤: [[EmotionControlinTTS]](pending-review), [[CodecLanguageModel]](pending-review), [[MaskedGenerativeModeling]](pending-review), [[Self-SupervisedSpeechRepresentation]](pending-review) | 未命中但可能相关: 无

**谱系定位**: Affectron 属于 Codec Language Model (NCLM) 范式下的 NV 生成方向。它以 VoiceCraft (330M NCLM) 为 backbone,使用 EnCodec 离散 token 作为语音表征。与 KB 中已有的 EmotionControlinTTS 方法谱系相比,Affectron 的切入点独特 — 不是建模抽象情感状态 (如 EmoSphere-TTS 的球面向量、UDDETTS 的 ADV 维度、EmoSteer-TTS 的激活 steering),而是聚焦于副语言发声 (NV) 这一情感的具体外在表现。

**已有认知**: KB 中 [[EmotionControlinTTS]] 页面已记录了两条与 NV 直接相关的方法路线: (1) EmoCtrl-TTS 用帧级 arousal-valence + laughter detector embedding 控制 NV; (2) NVSpeech 通过 18 类 word-level PV 分类体系和 token-level 显式标签实现 NV 控制。两者都是推理时控制 NV,而 Affectron 是训练时增强 NV 分布 — 路线互补。[[ProsodyModeling]] 页面也记录了 NVSpeech 将韵律建模扩展至副语言维度的贡献。

**创新判断**: Affectron 的核心创新在于"训练时情感感知数据增强"策略: 通过 Emotion2Vec embedding 驱动的 top-K NV 匹配 + 球面坐标角距离驱动的 top-K 位置路由,在 decoupled 语料上构造 NV 增强样本,避免了对大规模对齐标注数据的依赖。这与 CapSpeech 的 rule-guided randomized 策略形成对比 (AB test 中 Affectron 的增强一致优于 rule-guided)。

## 速查

> [!summary] 速查
> - **一句话**: 在 verbal-only 预训练 NCLM 上,通过情感驱动的 NV 数据增强 + 结构化 masking 实现多样化、情感一致的 NV 合成
> - **路线**: 文本 + 情感参考音频 → (训练时) Emotion2Vec NV 匹配 + 球面角距离位置路由 → NV 增强 token 序列 + NV 结构化 masking → VoiceCraft Transformer AR 生成 → EnCodec 解码 → 带 NV 的语音
> - **指标**: NV-Acc 37.75% (seen) / 36.90% (unseen) vs VoiceCraft 10.49/11.90 [Table 1]; NV type Acc@5 91.69%, JSD 0.0051 远优于 LLM baselines [Table 2]; AB test 一致偏好 affect-aware 增强 [Fig 3]
> - **可借鉴**: (1) 用 decoupled 语料 (语音和 NV 分开录制) + 情感 embedding 匹配来合成训练样本,绕过对齐标注; (2) 球面坐标角距离衡量局部情感稳定性选择 NV 插入位置; (3) 结构化 masking 让 verbal 上下文双向条件化 NV 生成
> - **局限**: 仅 EARS (100h) 小语料训练,无法与大规模系统直接对比; decoupled 设计无法建模 verbal-NV 重叠; 代码已开源

## 核心问题

现有 NV-TTS 面临两个根本困难:

1. **数据困境**: tag-controlled 方法 (ELaTE, EmoCtrl-TTS) 依赖 NV 对齐标注或 NV 检测器,检测器的偏差和错误传播导致 NV 位置不准 [§1]; spontaneous-style 方法 (Spontaneous-TTS) 依赖私有数据集,公开语料偏向基础 NV (呼吸、笑声),细粒度 NV (轻笑、窃笑) 建模困难 [§1]。
2. **建模困境**: 已有 NCLM 系统 (VALL-E, VoiceCraft) 主要面向 voice cloning,即使 prompt 中含 NV 线索也难以可靠生成细粒度 NV 韵律变化 [§2.2]。

Affectron 的核心思路: 不在推理时控制 NV,而在训练时构造情感对齐的 NV 增强样本,让 NCLM 在训练中学会 NV 的多样性和位置感知。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Affectron 是 VoiceCraft (Peng et al., 2024) 的 fine-tuning 框架,不修改模型架构,仅在训练流程中引入 NV 增强策略。VoiceCraft 是 330M 参数的 decoder-only Transformer NCLM,在 GigaSpeech (纯 verbal) 上预训练,使用 EnCodec 作为 speech tokenizer,通过 causal masking + delayed stacking 实现 token infilling [§3.2]。

训练时: NV 匹配 → NV 路由 → NV 增强样本构造 → NV 结构化 masking → AR 训练。
推理时: 直接从 NV-tagged 文本 + 情感参考音频生成,不需要匹配/路由步骤 [§4.5]。

### 关键设计选择

#### 1. Emotion-Driven Top-K NV Matching [§4.1]

**做什么**: 为每条 verbal utterance 选择情感一致的 NV 候选。

**怎么做**: 给定 verbal utterance u 和同一说话人的所有 NV 候选,用 Emotion2Vec (Ma et al., 2024) 提取情感 embedding,计算 u 与每个 NV 的 cosine similarity,取 top-K (K=10),通过 temperature-scaled softmax (tau=0.7) 归一化为概率分布,采样最多 2 个 NV [Eq. 1-2]。

**为什么这样设计**: [论文原文] 使用 top-K + temperature softmax 而非 argmax 是为了平衡情感一致性和 NV 多样性 — argmax 会导致过度集中于最相似的少数 NV 类型 [§4.1]。[论文原文] 选择 Emotion2Vec 而非通用 SSL 模型 (wav2vec 2.0, HuBERT, WavLM) 或 CLAP 是因为它在 NV 类型预测的 JSD (0.0051) 和 HD (0.0723) 上显著优于替代方案 [Appendix B, Table 4]。[agent 解读] 限制每条 utterance 最多 2 个 NV 采样是合理的工程约束 — 真实语音中每句话极少出现 3 个以上 NV。同说话人约束主要服务于声学过渡的平滑性而非情感匹配 [§G, Table 7]。

#### 2. Emotion-Aware Top-K Routing [§4.2]

**做什么**: 确定选中 NV 在 utterance 中的最佳插入位置。

**怎么做**: (1) 用 Montreal Forced Aligner 获取 word-level segment; (2) 用预训练情感属性预测器 (Wagner et al., 2023) 提取每个 verbal segment 和 NV 候选的 arousal-valence-dominance (AVD) 属性; (3) 将 AVD 从笛卡尔坐标变换到球面坐标 (r, theta, phi) [Eq. 11-12]; (4) 用角距离 (angular distance on unit sphere) 衡量 NV 与相邻 verbal segment 间的情感差异 [Eq. 3-4]; (5) 取角距离最小的 top-K 位置 (K=5),temperature-scaled softmax 采样插入点 [Eq. 5-6]。

**为什么球面角距离**: [论文原文] 论文分析了 EARS 和 NonverbalTTS 数据集中情感属性随时间变化的模式 — 短时间间隔内角距离更小,说明情感状态局部稳定 [Fig 1]。在情感变化最小的位置插入 NV 可以保持情感连贯性 [§3.1]。[论文原文] 比较了三种距离度量 (笛卡尔、径向、角距离),angular distance 在 NV 位置预测的 Acc@5 (44.06% vs 41.69/41.45) 和 JSD (0.0523 vs 0.0568/0.0544) 上均最优 [Appendix A, Table 3]。[agent 解读] 选择角距离而非笛卡尔距离的深层原因是: 角距离只关注情感方向变化,忽略强度差异 (径向分量),这对"位置选择"更合理 — 一个 happy 位置的情感强度可能变化但方向不变,仍适合插入对应类型的 NV。

#### 3. NV Structural Masking [§4.3]

**做什么**: 将 VoiceCraft 的 causal masking 机制针对性地应用于 NV token,使模型能利用双向 verbal 上下文生成 NV。

**怎么做**: (1) 按路由结果将 NV token 插入 verbal token 序列中; (2) 随机选择一个 NV span,在其周围构造 masked span (长度 l ~ Uniform(1, L),可包含相邻 verbal token); (3) 将 masked span 移至序列末尾,原位置插入 mask token; (4) 应用 delayed stacking (Copet et al., 2023) 实现多 codebook 并行 AR 建模 [§4.3]。

**为什么这样设计**: [论文原文] VoiceCraft 的 causal masking 本质是 bidirectional conditioning — mask span 移至序列末尾后,模型在生成 NV token 时同时看到前文和后文的 verbal context。作者假设这种双向条件化对 NV 合成特别有利,因为 NV 的情感表达需要与前后文语境一致 [§3.2]。[论文原文] 消融实验证实了这一点: 去掉 NV structural masking (退化为标准 causal masking) 后,NTN-MOS 和 NEC-MOS 均下降 [Fig 4, Table 1]。

### 训练策略

- Backbone: VoiceCraft 330M checkpoint (GigaSpeech pre-trained, verbal only) [§5.2]
- Fine-tuning: EARS dataset (~100h verbal + ~4h NV, 107 speakers, 15 NV types) [§5.1]
- 优化器: AdamW, lr=1e-5, batch size=100 (gradient accumulation), 50K steps [§5.2]
- 硬件: 4x NVIDIA RTX A6000, 5 天 [§5.2]
- Top-K 设置: EDNM K=10, EAR K=5, tau=0.7 (grid search 确定) [§5.2, Appendix H]
- Mask span: 数量 ~ truncated Poisson(1) in [1,3], 长度 ~ Uniform(1, 600) [§5.2]

## 实验

| 指标 | Affectron (full) | VoiceCraft | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| NV-Acc (seen) | 37.75% | 10.49% | EARS (seen) | [Table 1] |
| NV-Acc (unseen) | 36.90% | 11.90% | EARS (unseen) | [Table 1] |
| NV-Sim (seen) | 0.6118 | 0.5898 | EARS (seen) | [Table 1] |
| NV-Sim (unseen) | 0.5427 | 0.4766 | EARS (unseen) | [Table 1] |
| NV-EECS (seen) | 0.5748 | 0.6149 | EARS (seen) | [Table 1] |
| NV-EECS (unseen) | 0.5506 | 0.5479 | EARS (unseen) | [Table 1] |
| WER (seen) | 6.59 | 9.05 | EARS (seen) | [Table 1] |
| WER (unseen) | 8.31 | 10.50 | EARS (unseen) | [Table 1] |
| V-SECS (seen) | 0.8886 | 0.8927 | EARS (seen) | [Table 1] |
| V-SECS (unseen) | 0.8630 | 0.8690 | EARS (unseen) | [Table 1] |
| NV Type Acc@5 | 91.69% | - | NonverbalTTS | [Table 2] |
| NV Type JSD | 0.0051 | - | NonverbalTTS | [Table 2] |
| NV Location Acc@5 | 44.06% | - | NonverbalTTS | [Table 2] |
| NV Location JSD | 0.0523 | - | NonverbalTTS | [Table 2] |

**消融结论** [Table 1, Fig 4] (注: Table 1 的消融是累加式设计,从 DA-only 逐步加入 EDNM → EAR → NSM):
- DA-only (DA✓, EDNM✗, EAR✗, NSM✗): NV-Acc 58.78 (seen),随机 NV 拼接增加了 NV 类型覆盖但 NV-EECS 仅 0.5455 — 缺少情感匹配 [§6.2]
- +EDNM (DA✓, EDNM✓, EAR✗, NSM✗): NV-Acc 降至 35.83 但 NV-EECS 升至 0.5648 — 情感匹配牺牲 NV 多样性换取情感一致性 [§6.2]
- +EAR (DA✓, EDNM✓, EAR✓, NSM✗): NV-EECS 进一步升至 0.5707,位置路由改善 NV-verbal 情感整合 [§6.2]
- +NSM (Full, DA✓, EDNM✓, EAR✓, NSM✓): NV-Acc 回升至 37.75,NV-EECS 达 0.5748,NV-Sim 最高 0.6118 — 双向 verbal 上下文条件化 NV 生成是关键 [§6.2]
- VoiceCraft baseline (全✗): NV-Acc 仅 10.49 (seen),NV-EECS 却最高 0.6149 — [论文原文] 无增强时模型过拟合输入中的情感信息 [§6.2]

**与 NV-capable zero-shot TTS 对比** [Appendix F, Table 6]:
- CosyVoice2-0.5B / Fun-CosyVoice3-0.5B: verbal 指标强 (WER 1.97/1.65) 但 NV 相关指标弱 (NV-Acc 25.00/27.38)
- Dia-1.6B: NV 类别更广 (21 types) 但 NV-Acc 仅 13.10
- Affectron-330M: NV 指标全面领先 (NV-Acc 36.90, NV-Sim 0.5427),verbal 质量接近

**Filler 多样性** [§6.3, Fig 5]: VoiceCraft 只生成少数 filler 变体; Affectron 生成更多细粒度变体,分布更均衡。

## 局限性

1. **语料规模**: EARS 仅 ~100h verbal + ~4h NV,无法与大规模/私有语料训练的系统 (如 CosyVoice 系列) 直接对比 [§8]
2. **Decoupled 限制**: verbal 和 NV 分开录制,无法建模两者重叠的情况 (如边说边笑),而真实语音中 verbal-NV 重叠是常见现象 [§8]
3. **NV 检测器可靠性**: 评估中使用 FlexSED 标注 NV onset 位置,检测器本身的偏差可能影响位置评估 [agent 解读]
4. **Emotion2Vec 伪标签**: NV 匹配和路由都依赖 Emotion2Vec 伪标签,伪标签的噪声可能传播到增强样本中;不过 Appendix C 的分析表明 embedding 级对齐比 categorical 级更鲁棒 [Appendix C]
5. **推理时 NV 控制粒度**: 推理时需要手动在文本中插入 NV tag 并提供情感参考音频,NV 类型和位置由模型隐式决定,无法像 EmoCtrl-TTS 那样做帧级连续情感控制 [agent 解读]

## 点评

**优势**: Affectron 提出了一条巧妙的 NV 生成路线 — 不依赖大规模 NV 对齐标注,而是用情感 embedding 在 decoupled 语料上构造增强样本。球面坐标角距离用于位置路由有坚实的理论基础 (情感局部稳定性),并有消融实验支持。330M 小模型在 NV 指标上超越了 CosyVoice2/3 (0.5B) 和 Dia (1.6B),说明针对性的增强策略比单纯堆数据更有效。

**不足**: (1) decoupled 语料的根本限制使得系统无法处理 verbal-NV 重叠,这在真实对话中很常见; (2) NV-EECS 指标上 full model 反而不如 w/o DA 的变体 (0.5748 vs 0.6149 seen),论文解释为过拟合但不完全令人信服 — 情感一致性是 NV 系统的核心目标之一; (3) 与 NVSpeech 等 pipeline 方案的对比缺失,NVSpeech 能处理 18 类 PV 且有 ASR 自动标注。

**方法论启示**: 训练时增强 vs 推理时控制是两条互补路线。Affectron 证明了即使 NV 数据量极小 (~4h),只要匹配/路由策略设计得当,就能显著提升 NV 合成的多样性和情感一致性。

## 可复用的 idea

1. **Emotion-driven data augmentation**: 用情感 embedding 驱动的 top-K 匹配 + 采样,可推广到任何需要在训练时构造情感对齐样本的任务 (如 conversational TTS, audiobook TTS)
2. **球面角距离做 NV 位置路由**: AVD → 球面坐标 → 角距离的流程通用于任何需要衡量局部情感稳定性的场景 (如对话系统中选择 response 风格切换点)
3. **NV structural masking**: 将 causal masking 针对特定 token 类型 (NV) 应用,使双向上下文条件化非语言元素 — 可推广到其他副语言事件 (呼吸、停顿、犹豫)
4. **Decoupled corpus 利用**: 用分开录制的 verbal 和 NV 语料构造增强样本,避免昂贵的对齐标注 — 适用于低资源场景

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节因果解释充分,设计选择有明确理由和实验支撑 |
> | 可信赖 | pass | 数字标注覆盖率高,消融分析误读已修正 |
> | 可区分 | pass | 来源标注覆盖率高,事实/推断边界清晰 |
> | 可定位 | pass | KB 背景谱系定位清晰,创新判断有具体基准 |
> | 不污染 | pass | 无需新建概念页,反向更新为追加操作 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1) — 均已当场修正
> 详见 `_review/Affectron-review.yml`
