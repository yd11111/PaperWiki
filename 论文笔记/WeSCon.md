---
type: paper
tier: deep
title: "Word-Level Emotional Expression Control in Zero-Shot Text-to-Speech Synthesis"
arxiv_id: "2509.24629"
source: "Sources/WeSCon.pdf"
authors: [Tianrui Wang, Haoyu Wang, Meng Ge, Cheng Gong, Chunyu Qiang, Ziyang Ma, Zikang Huang, Guanrou Yang, Xiaobao Wang, Eng Siong Chng, Xie Chen, Longbiao Wang, Jianwu Dang]
year: 2025
venue: "NeurIPS 2025"
tags: [TTS, emotion-control, zero-shot, self-training, word-level-control, speaking-rate-control, attention-bias, CosyVoice2]
concepts: ["[[EmotionControlinTTS]]", "[[ConditionalFlowMatching]]", "[[ProsodyModeling]]", "[[SpeechTokenizer]]", "[[SpeakerEmbedding]]", "[[Speech-TextAlignment]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/Whisper|Whisper]]", "[[模型库/WavLM|WavLM]]", "[[模型库/wav2vec2.0|wav2vec 2.0]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 1 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: WeSCon 属于 [[EmotionControlinTTS]] 中的 word-level 细粒度控制方向,以 [[模型库/CosyVoice2|CosyVoice 2]] 为 backbone。在情感控制的演进线上,它位于 EmoCtrl-TTS (帧级 arousal-valence 连续 embedding, 2024) 和 ELaTE (flow-matching 笑声控制, 2024) 之后,但路线显著不同: EmoCtrl-TTS/ELaTE 依赖大规模含情感转换的训练数据 (EmoCtrl-TTS 用 27k 小时伪标签数据),WeSCon 提出不依赖此类数据的 self-training 方案。在 [[ProsodyModeling]] 中,WeSCon 与 word-level prosody 控制相关 (词级重音/节奏),但同时扩展到情感和语速的联合词级控制。CosyVoice 2 的 LM + [[ConditionalFlowMatching]] 架构为本文提供了语义-声学解耦基础: LM 编码语义/情感,flow matching 控制音色,这使得多轮推理中的 speaker consistency 可通过 flow matching 保证。[[SpeechTokenizer]] 的监督式设计 (FSQ-SenseVoice) 使 speech token 主要编码语义信息,为情感切换提供了操作空间。
>
> **已有认知**: 知识库已记录多种情感控制路线 — embedding-based (EmoSphere-TTS), DPO-based (Emo-DPO), activation steering (EmoSteer-TTS), ControlNet (TTS-CtrlNet), reward-guided (DiffRO-MTR)。其中 TTS-CtrlNet 与本文最相关: 两者都在 CosyVoice2/F5-TTS 上实现词级情感控制,但 TTS-CtrlNet 用 ControlNet 旁挂冻结模型 + 小数据训练,WeSCon 用 self-training 蒸馏 + 注意力偏置。EmoCtrl-TTS 通过 chunk-wise arousal-valence embedding 实现帧级控制,但需要大量含情感转换的训练数据。
>
> **创新判断**: 相比知识库已有方法,WeSCon 的核心新贡献是: (1) 首个不依赖含情感转换数据的 word-level 情感控制框架 (通过 self-training 绕过数据瓶颈); (2) 多轮推理 + transition smoothing 的 teacher 设计实现情感段衔接; (3) Dynamic Emotional Attention Bias (DEAB) 机制引导 student 模型在端到端推理中关注正确的情感 prompt。
>
> 检索命中: [[模型库/CosyVoice2|CosyVoice 2]]✓, [[ConditionalFlowMatching]]✓, [[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]✓, [[ProsodyModeling]]✓, [[SpeechTokenizer]]✓ | 过滤: [[EmotionControlinTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: WeSCon 通过两阶段 self-training 框架,首次在不依赖含情感转换数据的条件下实现了零样本 TTS 中的 word-level 情感和语速联合控制
> - **路线**: 文本 + 多个情感 prompt → [Stage 1: 多轮推理 + transition smoothing + dynamic speed control → teacher 生成伪标签] → [Stage 2: student TTS (CosyVoice2 + DEAB) 端到端推理] → flow matching + vocoder → 语音
> - **指标**: Emo2v. 0.882 / Aro. 0.468 (EN), DNSV 4.361 (EN, 越低越好, CosyVoice2 baseline 7.894), EMOS 3.70±0.17, NMOS 3.93±0.20 [Table 1, Table 2]; 零样本 TTS 性能 CER 1.47 / S-SIM 0.744 与 CosyVoice2 几乎一致 [Table 3]
> - **可借鉴**: (1) 多轮推理 + tail-to-head linkage 解决多情感段拼接的不连续性; (2) 用 self-training 绕过细粒度标注数据的稀缺; (3) Dynamic Emotional Attention Bias — 7 种预定义 bias 模板的加权组合,轻量且高效; (4) 语速控制通过简单的 prompt token 插值/下采样实现,50%-200% 范围内稳定有效
> - **局限**: (1) 仅支持离散情感类别,不支持混合/渐变情感; (2) 情感转换方案由 GPT-4o 预定义,缺乏上下文自适应; (3) ESD 数据集情感/说话人多样性有限导致 500h 后过拟合; (4) NeurIPS 2025 论文,代码计划开源但截至目前未见完整发布

## 核心问题

本文试图回答一个关键问题: **是否可以在不依赖含情感转换数据的条件下,实现有效的 word-level 情感和语速控制?** [§1]

这个问题重要是因为:
1. 当前 TTS 系统的情感控制局限于 utterance-level,而人类自然言语中情感和语速是逐词动态变化的 [§1]
2. 现有 word-level 控制方法 (ELaTE, EmoCtrl-TTS) 依赖大规模含 intra-sentence 情感转换的私有数据,极难获取 [§1, §2]
3. 仅从文本预测情感 (ED-TTS, EmoQ-TTS) 无法捕捉声学线索 (韵律、强度) [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

WeSCon 是一个两阶段 self-training 框架 [Fig 2]:

**Stage 1 (Teacher Model)**: 在冻结的 CosyVoice2 backbone 上,通过三个附加机制实现 word-level 控制:
- **多轮推理**: 按用户定义的 emotion plan 将句子分段,每段用不同的情感 prompt 合成,最后拼接 [§3.2.1]
- **Transition smoothing**: 在每轮推理时将上一轮的尾部 text/speech token 追加到当前 prompt,形成 tail-to-head linkage [§3.2.1]
- **Dynamic speed control**: 通过最近邻插值或下采样调整 prompt speech token 长度,控制每段的语速 [§3.2.2]

**Stage 2 (Student Model)**: CosyVoice2 原始模型 + 轻量 DEAB 模块,在 teacher 生成的伪标签上 self-training,实现端到端单次推理的 word-level 控制 [§3.3]

### 关键设计选择

#### 为什么选多轮推理而非直接建模多情感?

[论文原文] 作者利用了 CosyVoice2 已有的 utterance-level 零样本情感克隆能力 — 既然模型可以从 prompt 克隆一种情感,那么分段合成 + 使用不同情感 prompt 自然可以实现 word-level 控制,无需修改模型参数 [§3.2.1]。[agent 解读] 这是一种"不训练也能用"的 inference-time 策略,将 word-level 控制转化为已有 utterance-level 能力的组合。

#### 为什么需要 transition smoothing?

[论文原文] 直接拼接不同段会导致声学不连续 (acoustic discontinuities at segment boundaries) [§3.2.1]。smoothing 通过追加上一轮的尾部 token 作为 context,与 CosyVoice2 的 continuation-style generation 自然对齐 [§3.2.1]。[论文原文] 消融实验证实: 移除 smoothing 后 DNSV 从 4.980 升至 7.568,S-SIM 和 Emo2v. 也明显下降 [Table 4]。

**Content aligner 的作用**: 一个轻量的 non-causal Transformer + Conv 模块,训练于 ASR 数据,预测每个 speech token 对应的 text token [§3.2.1]。[agent 解读] 这使系统在每轮推理后能精确定位"上一段到哪个词结束了",从而实现准确的 tail-to-head 切分。

#### Dynamic speed control 的原理

[论文原文] CosyVoice2 的语速完全由 prompt speech tokens 的长度决定。通过对 prompt token 做最近邻插值 (延长 → 语速减慢) 或下采样 (缩短 → 语速加快) 即可控制 [§3.2.2]。有效范围为原始长度的 50%-200% [Appendix B, Fig 7]。

#### 为什么需要 self-training (Stage 2)?

[论文原文] Stage 1 的 teacher 虽然有效但推理复杂: non-causal content aligner + 多轮推理 + tail-to-head linkage 引入大量 inference overhead [§3.3]。Self-training 将这种能力蒸馏到 student 中,实现端到端单次推理 [§3.3]。

#### Dynamic Emotional Attention Bias (DEAB) 的设计

[论文原文] 在多情感 prompt 的端到端推理中,模型可能在生成 Emotion I 对应的语音时错误地关注 Emotion II 的 prompt,导致情感不一致 [§3.3.2]。

DEAB 的工作方式:
1. 一个轻量 causal Transformer (emotion aligner) 预测每个 speech token 的情感标签 [§3.3.2]
2. 将 text-speech 特征与预测的情感特征拼接,经 MLP 生成权重向量 ω ∈ R^{1×7} [§3.3.2]
3. 用 ω 线性组合 7 种预定义 attention bias 模板 B_temp ∈ R^{7×T×T},得到动态 bias [Eq. 1]
4. bias 与 softmax-normalized attention 逐元素相乘 (⊙),引导模型关注正确的情感 prompt 区域 [Eq. 2]

[agent 解读] 7 种预定义模板 (Appendix F) 覆盖了从标准 causal attention 到严格 emotion-aligned attention 的不同模式,model 学习的是"在不同时刻选择哪种 attention 模式",而非从头学习 bias 矩阵。这大幅降低了学习难度。

#### 数据格式设计

[论文原文] Student 的输入格式为 { S, E_I, C_prompt_I, E_II, C_prompt_II, ..., C_tgt, B, S_prompt_I, S_prompt_II, ..., S_tgt },其中 E_i 是情感标记 token [§3.3.2]。这与 CosyVoice2 原始格式 { S, C, B, S } 兼容 — 在单情感场景下退化为原始格式,保留预训练知识 [§3.3.2]。

[论文原文] 消融实验 [Table 4] 证实: 去掉 emotion flag → 全面指标下降; 用 naive 格式拼接 → CER 从 2.122 骤增至 4.141,说明与预训练格式对齐至关重要。

### 训练策略

**Stage 1**: Content aligner 在 200h 非情感中英文数据 (LibriSpeech-100-Clean + AISHELL-1) 上训练 400k 步,loss = token-level classification + boundary detection [Eq. 3]; TTS backbone 完全冻结 [§3.4]

**Stage 2**: 
- GPT-4o 生成情感转换文本 (Appendix D),ESD train-set 提供情感 prompt (无 intra-sentence transitions)
- Teacher 合成伪标签 → 基于 CER/WER + speaker similarity + emotion similarity 的 data filtering (top 50%) [Appendix E]
- Student 训练 600k 步,前 20k 步冻结 TTS 只训练 emotion aligner [§3.4]
- TTS learning rate 极低 (5e-7) 以保护零样本能力 [§3.4]
- Loss = speech token prediction [Eq. 4] + emotion prediction [Eq. 5]

**Speaker consistency 策略**: 多轮推理中优先从同一说话人的不同情感中选择 prompt; flow matching 端用目标说话人的参考样本提供 speaker identity [§3.2.3]。[agent 解读] 这利用了 CosyVoice2 的语义-声学解耦: LM 的 speech token 主要编码语义/情感,flow matching 负责音色重建,因此只要 LM 端不严重泄漏 speaker 信息,speaker consistency 就能通过 flow matching 保证。

## 实验

| 指标 | WeSCon (2nd) | CosyVoice2 | F5-TTS | Index-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Emo2v. ↑ | 0.882 | 0.866 | 0.869 | 0.858 | EN test | [Table 1] |
| Aro. ↑ | 0.468 | 0.446 | 0.447 | 0.434 | EN test | [Table 1] |
| DNSV ↓ | 4.361 | 7.894 | 8.972 | 8.967 | EN test | [Table 1] |
| S-SIM ↑ | 0.532 | 0.521 | 0.453 | 0.387 | EN test | [Table 1] |
| AutoPCP ↑ | 2.707 | 2.525 | 2.417 | 2.436 | EN test | [Table 1] |
| WER ↓ | 3.192 | 3.185 | 2.954 | 2.611 | EN test | [Table 1] |
| Emo2v. ↑ | 0.872 | 0.843 | 0.847 | 0.838 | ZH test | [Table 1] |
| DNSV ↓ | 4.210 | 7.612 | 9.134 | 8.521 | ZH test | [Table 1] |
| EMOS ↑ | 3.70±0.17 | 3.61±0.17 | 3.63±0.15 | 3.51±0.19 | 主观 | [Table 2] |
| SPMOS ↑ | 3.89±0.18 | 3.56±0.20 | 3.51±0.21 | 3.50±0.21 | 主观 | [Table 2] |
| NMOS ↑ | 3.93±0.20 | 3.29±0.23 | 2.84±0.26 | 2.97±0.25 | 主观 | [Table 2] |
| CER ↓ (ZS) | 1.47 | 1.45 | - | - | SEED test-zh | [Table 3] |
| S-SIM ↑ (ZS) | 0.744 | 0.748 | - | - | SEED test-zh | [Table 3] |

**核心发现**:
1. DNSV (transition smoothness) 改善最显著: CosyVoice2 7.894 → WeSCon 4.361 (EN), 降幅 45% [Table 1],说明 smoothing + 端到端推理大幅减少声学不连续
2. Emotion similarity 全面领先: 2nd-stage 略优于 1st-stage,归因于 data filtering 保留了高质量伪标签 [§4.2.1]
3. 零样本 TTS 能力几乎无损: CER 1.47 vs 1.45, S-SIM 0.744 vs 0.748 [Table 3],极小的退化证明 self-training + 低 LR 策略有效保护了预训练知识
4. 数据量甜点 ~500h [Fig 5]: 超过后指标下降,归因于 ESD 情感/说话人多样性有限导致过拟合

**消融关键结论** [Table 4]:
- w/o smoothing: DNSV 4.980 → 7.568 (+52%),证实 transition smoothing 不可或缺
- w/o attention bias: Emo2v. 0.872 → 0.837,S-SIM 0.599 → 0.575,证实 DEAB 是 student 关注正确 prompt 的关键
- w/o data format: CER 2.122 → 4.141 (↑95%),格式与预训练对齐极为重要
- w/o emotion flag: Emo2v. 0.872 → 0.831,情感标记 token 是定位情感切换位置的重要信号

## 局限性

1. **离散情感限制**: 仅支持预定义的离散情感类别 (happy, sad, angry, neutral, surprise),不支持连续 arousal-valence 控制或混合情感 (如"绝望 = 愤怒+悲伤") [§5]
2. **缺乏渐变建模**: 虽然实现了信号级平滑转换,但缺少语义级情感演化建模 — 人类言语中的情感变化常包含中间状态 [§5]
3. **预定义控制方案**: 情感转换方案由 GPT-4o 提前生成,推理时需用户指定 emotion plan,缺乏上下文自适应能力 [§5]
4. **数据多样性瓶颈**: ESD 仅 5 种情感 × 20 说话人,500h 后即过拟合 [Fig 5]。更大更多样的数据集可能释放更大潜力
5. **WER 轻微退化**: EN WER 3.192 vs CosyVoice2 3.185 几乎持平,但都比 Index-TTS (2.611) 和 F5-TTS (2.954) 高 [Table 1],这部分来自 backbone 本身的限制

## 点评

**优势**:
- 核心 insight 很巧妙: 将 word-level 控制问题转化为 utterance-level 能力的时序组合,再通过 self-training 蒸馏为端到端模型。这避开了最难的数据瓶颈
- 工程设计实用: tail-to-head linkage 利用了 continuation-style generation 的自然语义,DEAB 用 7 种预定义模板降低学习空间,数据格式兼容原始预训练格式
- 实验设计完善: 中英双语, 4 个强 baseline, 客观+主观评估, 全面消融, out-of-domain 泛化 (CASIA, Appendix H), 零样本保持验证

**不足**:
- 与 EmoCtrl-TTS 的对比缺失: 虽然在 related work 中讨论了 EmoCtrl-TTS,但实验中未将其作为 baseline (可能因为非公开模型)
- TTS-CtrlNet 同样用小数据在冻结模型上实现情感控制,且发表更早,但论文未讨论与之的关系
- 500h 过拟合限制暴露了方法对数据多样性的依赖 — 如果只有 ESD 这样的小数据集,self-training 的数据增强能力可能很快触顶
- 多轮推理的 Stage 1 teacher 推理成本高 (需 N 轮推理 + aligner),虽然 Stage 2 解决了这个问题但 teacher 本身仍是瓶颈

**与知识库已有方法的对比定位**:

| 方法 | 路线 | 数据需求 | 控制粒度 | 是否修改预训练参数 |
|------|------|----------|----------|-------------------|
| EmoCtrl-TTS | arousal-valence embedding | 27k h 伪标签 | 帧级 | 是 (全参微调) |
| TTS-CtrlNet | ControlNet 旁挂 | ~400h 公开数据 | 帧级 | 否 (冻结+旁挂) |
| EmoSteer-TTS | activation steering | ~7k 条 (零训练) | 全局 | 否 (零训练) |
| **WeSCon** | self-training + DEAB | ~500h 公开数据 | 词级 | 是 (低 LR 微调) |

## 可复用的 idea

1. **Self-training 绕过数据瓶颈**: 当目标任务需要的标注数据稀缺时,可以先设计一个 inference-time pipeline (teacher) 利用已有能力组合实现目标效果,再用 teacher 生成伪标签蒸馏到 student。这个范式对其他细粒度控制任务 (如 word-level 音高控制, word-level 停顿控制) 也适用

2. **Tail-to-head linkage for segment concatenation**: 在需要将多段语音拼接的场景中,将上一段尾部 token 作为下一段的 prefix context,利用模型的 continuation generation 能力实现自然过渡。适用于有声书、长文本分段合成等场景

3. **预定义 attention bias 模板**: 不让模型从头学习完整的 attention mask/bias,而是定义若干典型模式,让模型只学习"选哪个/怎么混合"。这大幅降低了学习难度,且产生的 bias 总是合理的 (因为模板已预设了合理结构)

4. **Prompt token 插值/下采样控制语速**: 极简单但有效的语速控制方法,50%-200% 范围稳定。可直接应用于其他基于 prompt 驱动的 TTS 系统

5. **数据格式与预训练兼容**: 新增控制能力时保持输入格式与原始预训练格式兼容 (单情感时退化为原始格式),最大限度保留预训练知识

> [!review] 审阅结论: pass-with-fixes (2026-06-04)
> - **结论**: pass-with-fixes, 1 个 low issue
> - **low**: frontmatter models 含 BigVGAN 但论文未直接使用 (已修正)
> - 详见 `_review/WeSCon-review.yml`
