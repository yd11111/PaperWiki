---
type: paper
tier: deep
title: "BatonVoice: An Operationalist Framework for Enhancing Controllable Speech Synthesis with Linguistic Intelligence from LLMs"
arxiv_id: "2509.26514"
source: "Sources/BatonVoice.pdf"
authors: [Yue Wang, Ruotian Ma, Xingyu Chen, Zhengliang Shi, Wanshun Chen, Huang Liu, Jiadi Yao, Qu Yang, Qingxuan Jiang, Fanghua Ye, Juntao Li, Min Zhang, Zhaopeng Tu, Xiaolong Li]
year: 2025
venue: "arXiv"
tags: [TTS, controllable, LLM, emotion, instruction-following, preference-optimization, cross-lingual, operationalism]
concepts: ["[[Instruction-Guided Speech Synthesis]]", "[[Emotion Control in TTS]]", "[[Prosody Modeling]]", "[[LLM-based TTS]]", "[[Conditional Flow Matching]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]"]
tasks: ["[[Instructed Speech Generation]]", "[[Zero-shot Speech Synthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 7
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: BatonVoice 属于 [[LLM-based TTS]] 中的 instruction-guided 可控合成分支。与 CosyVoice/VoxInstruct 等端到端指令跟随 TTS 不同,BatonVoice 采用"解耦"策略 -- 将指令理解与语音生成分离为两个独立模块。这在 [[Instruction-Guided Speech Synthesis]] [待确认] 的演进路线中属于一个新范式: 不是让 TTS 模型本身理解指令,而是让外部 LLM 先将指令转换为显式声学特征,再由专用 TTS 模型执行合成。

**已有认知**:
- [[Prosody Modeling]]: 韵律的物理维度 (pitch, energy, duration) 是 TTS 控制的基础。传统方法要么显式预测这些维度 (FastSpeech 2),要么隐式建模 (VALL-E in-context learning)。BatonVoice 回归显式路线,但用 LLM 替代 variance predictor。
- [[Emotion Control in TTS]] [待确认]: 现有情感控制方法包括 emotion embedding、DPO 优化、activation steering 等。BatonVoice 的解耦策略与这些方法正交 -- 不直接建模情感表示,而是将情感映射到可量化的声学特征。
- [[Conditional Flow Matching]]: CosyVoice 2 的 CFM decoder 将离散 speech token 转换为 mel spectrogram,BatonVoice 直接复用这一组件。
- [[Speech Tokenizer]]: CosyVoice 2 的 FSQ-SenseVoice tokenizer 将语音编码为离散 token,BatonVoice 使用相同的 tokenizer。

**创新判断**: 相对于 [[Instruction-Guided Speech Synthesis]] [待确认] 中的已有方法 (CosyVoice 需 556h, CosyVoice2 需 1500h 指令数据 [Table 1]),BatonVoice 的核心创新是"零指令数据"实现可控性 -- 通过 operationalism 思想将抽象指令转为可量化的 vocal features,绕过了昂贵的指令-语音标注。这与 [[Emotion Control in TTS]] [待确认] 中 DiffRO 的"从 reward model 蒸馏情感知识"思路有相似性 -- 都是避免人工标注的路线。

> 检索命中: [[LLM-based TTS]], [[Conditional Flow Matching]], [[Prosody Modeling]], [[Speech Tokenizer]], [[CosyVoice 2]] | 过滤: [[Instruction-Guided Speech Synthesis]](pending-review), [[Emotion Control in TTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 LLM 将自然语言指令解码为显式 vocal features (pitch/energy/spectral centroid),再由专用 TTS 模型 BatonTTS 从这些特征生成语音,实现零指令数据的可控合成
> - **路线**: Instruction + Text → LLM (conductor, Gemini 2.5 Pro) → JSON vocal features (Fv) → BatonTTS (Qwen3-1.7B + CosyVoice2 decoder) → Speech
> - **指标**: Emotion Acc. 57.6% (EN, 超 Minimax-2.5-HD 48.6% +9.0 abs) / WER 2.5 (Seed-TTS) / 中文零样本 Emotion Acc. 56.2% (超 CosyVoice 52.0%) [Table 1, Table 3]
> - **可借鉴**: (1) 将抽象控制需求"物化"为可量化特征作为中间表示,让 LLM 和 TTS 各做各擅长的事; (2) APO preference optimization 用预训练模型输出作 rejected sample,SFT 模型输出作 chosen sample,无需人工标注偏好数据; (3) vocal features 从 decoder 重建语音中提取而非原始语音,保证特征在合成空间可实现
> - **局限**: (1) 推理依赖外部强 LLM (Gemini 2.5 Pro) 生成 vocal features,增加延迟和成本; (2) 人类评估中被 Minimax-2.5-HD 超越 (win rate 30%),流畅度和自然度不足; (3) 特征维度有限 (仅 pitch/energy/spectral centroid),缺乏 duration/rhythm/breathing 等控制; (4) 仅英文训练,中文零样本 WER 2.1 与原生模型有差距

## 核心问题

本文要解决的核心问题是: 现有 LLM-based TTS 系统虽然使用 LLM 作为 backbone,但未充分利用 LLM 的语言理解能力 (linguistic intelligence),尤其是其指令跟随能力。具体表现为: (1) 训练可控 TTS 需要大量人工标注的指令-语音配对数据,成本高且标注一致性差 [§1]; (2) LLM 被当作纯粹的 sequence-to-sequence backbone,而非语言理解的智能体 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

BatonVoice 采用"conductor-orchestra"双模块设计 [§2.1]:

1. **Conductor (指挥)**: 外部 LLM (推理时用 Gemini 2.5 Pro),将用户指令 I + 文本 X 解释为细粒度 vocal features Fv,输出 JSON 格式的 word-segment 级特征列表
2. **Orchestra (乐队)**: BatonTTS,一个基于 LLM backbone (Qwen3-1.7B / Qwen2.5-0.5B) + 冻结的 CosyVoice2 speech decoder 的 TTS 模型,从 text + vocal features 条件生成语音

Vocal features 包含 5 个维度 [§2.1]:
- **Pitch mean** (平均基频, integer)
- **Pitch slope** (语调轮廓, integer)
- **Energy RMS** (平均信号幅度, float 3dp)
- **Energy slope** (能量变化趋势, integer)
- **Spectral centroid** (感知明亮度, integer)

特征在 word-segment 级别定义,每个 segment 由相邻词合并至超过 1 秒阈值 [§3.1],以确保韵律信息的稳定性。

[论文原文] 作者将此设计类比为"operationalism"哲学原则 -- 复杂概念通过可量化、可解释的操作来理解 [§1]。就像用传感器将不可感知的超声波转化为频率和振幅等可量化特征,BatonVoice 将抽象的"用讽刺的语气说"转化为具体的 pitch/energy 数值 [§1]。

[agent 解读] 这种设计的核心优势在于模块化: conductor 和 orchestra 可以独立升级。实验证实,更换更强的 conductor LLM (Qwen3-1.7B → Gemini 2.5 Pro) 可将 emotion accuracy 从 29.8% 提升到 57.6%,而无需重新训练 BatonTTS [Fig 3b]。

### 关键设计选择

**为什么用文本化的 vocal features 而非直接端到端?**

[论文原文] 作者认为端到端方法 "largely bypasses the LLM's inherent linguistic intelligence" [§1]。将指令理解和语音生成分离的好处: (1) LLM 只需做它擅长的事 -- 文本理解和推理; (2) 不需要人工标注指令-语音数据; (3) TTS 模型只需学习从 vocal features 到语音的映射,任务更简单 [§1]。

[agent 解读] 这实际上是将"模态对齐"的负担从 TTS 模型转移到了 LLM。对比 VoxInstruct/CosyVoice 需要 556-1500h 指令标注数据,BatonVoice 用 0h 指令数据达到了更好的情感控制效果,验证了这一设计的有效性。但代价是推理时需要额外调用一次强 LLM。

**为什么从 decoder 重建语音提取 vocal features 而非原始语音?**

[论文原文] 因为 speech decoder 无法完美重建原始语音的全部细节。从重建语音提取 features 可以确保这些 features 是 TTS 系统可以真正实现的 [§3.1]。

[agent 解读] 这是一个关键的工程洞察: 如果从原始语音提取 features,然后要求 BatonTTS 合成出与这些 features 匹配的语音,但 decoder 无法重建原始语音的某些特征,就会产生 train-test mismatch。从重建语音提取 features 消除了这个 gap。MCD 实验 [Table 5] 也验证了这一点: numerical features 方案的 MCD (1.54) 甚至低于 vocoder 重合成 baseline (2.46)。

**为什么选 APO-down 而非 DPO/RLHF?**

[论文原文] APO-down (Anchored Preference Optimization) 的 Term 1 将 chosen response 锚定在 SFT 模型附近 (防止偏离太远),Term 2 拉大 chosen-rejected 的 reward margin [§2.3]。Rejected 样本来自 pre-trained 模型 (无 features 条件),Chosen 样本来自 SFT 模型 (有 features 条件),这样 chosen 既有更好的可懂度,又隐式地加强了对 vocal features 的遵从 [§2.3]。

[agent 解读] 与 DPO 的区别在于额外的 anchor term,这对 TTS 场景更合适 -- TTS 中 SFT 模型已经有不错的基线质量,APO 防止 preference optimization 导致输出偏离正常语音分布。与 DiffRO (CosyVoice 3) 相比,APO 不需要 Gumbel-Softmax 可微采样,实现更简单。

### 训练策略

三阶段渐进训练 [§2.3]:

**Stage 1: Pre-Training** (基础 TTS 能力)
- 数据: VoxBox 103K hours,英文多说话人
- 目标: 标准 next-token prediction,text + speech tokens 拼接
- 基础设施: 80 GPUs, 3 epochs (~1 天), AdamW lr=1e-4, batch=640, DeepSpeed ZeRO-2 [§3.1]

**Stage 2: SFT** (vocal features 条件化)
- 数据: 377,619 utterances, >500 hours,来自 VCTK/VoxCeleb/EARS/Expresso/EmoVoice-DB/CapSpeech/Synthetic-Persona-Chat [Table 4]
- 特征提取: word-level alignment → 合并至 >1s segments → Parselmouth 提取 features → 转为 JSON 文本
- 训练序列: [text; Fv; speech_tokens],standard cross-entropy loss
- 关键: features 从 decoder 重建语音而非原始语音提取 [§3.1]
- 3 epochs

**Stage 3: Preference Optimization** (质量增强)
- 数据: 9,823 preference samples
- Rejected: pre-trained 模型生成的高 WER 或低 speaking rate 样本 (τ_wer=0.1, τ_sr=1.5 words/s)
- Chosen: SFT 模型在相同 vocal plan 下生成的高质量样本
- 方法: APO-down,SFT 模型作为 reference policy
- 1 epoch

Speech decoder 全程冻结,直接使用 CosyVoice2 预训练的 flow matching model + HiFi-GAN vocoder [§2.2]。

## 实验

| 指标 | BatonVoice-1.7B | BatonVoice-0.5B | CosyVoice | CosyVoice2 | Minimax-2.5-HD | Spark-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Emotion Acc. (EN) | 57.6% | 52.8% | 43.8% | 37.8% | 48.6% | 27.4% | Emotion (Saravia) | [Table 1] |
| WER (EN) | 2.5 | 2.9 | 3.4 | 2.1 | 1.5 | 1.9 | Seed-TTS | [Table 1] |
| Emotion Acc. (ZH) | 56.2% | - | 52.0% | - | 49.0% | 29.2% | Chinese Emotion | [Table 3] |
| WER (ZH) | 2.1 | - | 2.1 | 2.0 | 0.9 | 1.5 | Seed-TTS ZH | [Table 3] |
| Human win rate vs CosyVoice | 56% | - | - | - | - | - | Social IQa 50 pairs | [Table 2] |
| Human win rate vs Minimax-2.5-HD | 30% | - | - | - | - | - | Social IQa 50 pairs | [Table 2] |

**阶段消融** [Fig 3a]:
- Base (pre-train only): 23.2% (1.7B) / 25.8% (0.5B)
- +SFT: 52.2% (+29.0) / 51.6% (+25.8)
- +SFT+APO: 57.6% (+5.4) / 52.8% (+1.2)

**Conductor LLM 消融** [Fig 3b]:
- Qwen3-1.7B: 29.8%
- Qwen3-80B: 39.8%
- Qwen3-235B: 43.8%
- Qwen3-Max: 47.8%
- GPT-5: 49.6%
- o3 pro: 56.2%
- Gemini 2.5 Pro: 57.6%

正相关趋势明确: LLM 能力越强 → vocal features 预测越准 → 合成情感准确率越高。

**特征表示消融** [Table 5]:
- Vocoder resynthesis MCD: 2.46
- Caption-based features MCD: 2.62
- Numerical features MCD: 1.54
- w/o pitch: 1.63 (+0.09)
- w/o energy: 2.13 (+0.59)
- w/o spectral centroid: 1.57 (+0.03)

Numerical 表示显著优于 caption; 所有特征均有贡献,energy 影响最大。

## 局限性

1. **推理成本高**: 依赖外部强 LLM (Gemini 2.5 Pro) 做 conductor,增加延迟和 API 成本。用自己的 1.7B 模型作 conductor 时 emotion accuracy 从 57.6% 降到 29.8% [Fig 3b],说明目前解耦框架的效果严重依赖 conductor 的能力 [agent 解读]。

2. **自然度和流畅度不足**: 人类评估中被 Minimax-2.5-HD 超越 (win rate 仅 30%),作者承认在 fluency 和 naturalness 方面落后于商用系统 [§3.3, Table 2]。

3. **特征维度有限**: 仅建模 pitch/energy/spectral centroid 三个声学维度。缺少 duration/rhythm 控制 (论文通过 segment 中的 word 数量隐式编码 speaking rate [Appendix A]),缺少 breathing/非语言发声等维度 [§5]。

4. **评估局限**: Emotion accuracy 使用 Gemini 2.5 Pro 而非人类评估,可能存在偏差; 仅测试 5 种基本情感,未覆盖细粒度情感 (讽刺/犹豫等) [§3.1]。

5. **WER 竞争力一般**: WER 2.5 高于 CosyVoice2 (2.1)、Spark-TTS (1.9)、Minimax (1.5),说明情感增强可能以牺牲可懂度为代价 [Table 1]。

6. **训练数据规模偏小**: SFT 仅用 500h 英文数据,pretrain 103K h; 对比 CosyVoice2 pretrain 167K h + 1500h instruction data [Table 1],数据总量更少且全部为英文 [§3.1]。

## 点评

BatonVoice 提出了一个简洁且有效的思路: 与其让 TTS 模型学习理解指令,不如让 LLM 将指令"翻译"为 TTS 能直接执行的声学特征。这种 operationalism 哲学的落地方式颇有启发性 -- 它本质上是将不可控的隐式建模转化为可解释的显式建模。

**亮点**:
- 零指令数据实现了超越 VoxInstruct/CosyVoice 需要数百/数千小时标注数据才能达到的情感控制性能
- Conductor LLM 的可替换性证明了模块化设计的长期价值: 随着 LLM 进步,TTS 控制能力自动提升
- 从 decoder 重建语音提取 features 的工程洞察,解决了 train-test mismatch

**不足**:
- 推理时强依赖外部 LLM (Gemini 2.5 Pro),在实际部署中增加成本和延迟
- 人类评估中自然度落后于商用系统,说明"先规划再执行"的管线可能引入不自然的韵律断裂
- 特征空间太窄 (仅 5 维),难以覆盖复杂表现力需求

## 可复用的 idea

1. **"物化"控制需求为中间表示**: 将抽象指令转为可量化的数值特征,让两个模块各做各擅长的事。这一思路可泛化到其他模态控制任务 (视频/音乐的风格控制)。

2. **从重建信号提取训练特征**: 当 decoder 有信息损失时,从 decoder 重建输出而非原始信号提取条件特征,消除 train-test mismatch。适用于任何有 tokenize-detokenize bottleneck 的系统。

3. **Preference data 构造无需人工标注**: 用 pre-trained 模型 (无条件) 的输出作 rejected,SFT 模型 (有条件) 的输出作 chosen,WER/SR 阈值自动筛选。可复制到其他条件生成任务的 preference optimization。

4. **Conductor 可替换性**: 设计系统时将"理解"和"执行"分开,使系统能搭上基础模型进步的顺风车。

> [!review] 审阅 (2026-06-04, agent)
> **结论**: pass-with-fixes (0 high / 2 medium / 1 low)
> - [medium] traceability-gap: KB 背景中 VoxInstruct/CosyVoice 指令数据量误标 → 已修正为 CosyVoice 556h / CosyVoice2 1500h
> - [medium] overclaim: 局限性中数据规模对比表述 → 已修正为准确数据
> - [low] template-compliance: models 字段仅列 CosyVoice 2,未列其他 baseline → 可接受 (baseline 未全部有模型页)
> 详见 `_review/BatonVoice-review.yml`
