---
type: paper
tier: deep
title: "GSRM: Generative Speech Reward Model for Speech RLHF"
arxiv_id: "2602.13891"
source: "https://arxiv.org/abs/2602.13891"
authors: [Maohao Shen, Tejas Jayashankar, Osama Hanna, Naoyuki Kanda, Yancheng Wang, Katerina Zmolikova, Ruiming Xie, Niko Moritz, Anfeng Xu, Yashesh Gaur, Gregory Wornell, Qing He, Jilong Wu]
year: 2026
venue: "arXiv"
tags: [reward-model, generative-reward, RLHF, speech-naturalness, CoT-reasoning, acoustic-features, TTS-evaluation, online-RL]
concepts: ["[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[TTS Evaluation]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[LLM-based TTS]], [[Speech Tokenizer]], [[TTS Evaluation]])
> TTS Evaluation 中传统 MOS 预测器 (NISQA, UTMOSv2) 本质是标量回归器,缺乏可解释性且跨域泛化差;LLM-as-a-Judge 是新兴方向,但 frontier speech LLM (如 Gemini) 在 naturalness 判断上能力有限。LLM-based TTS 系统 (GPT-4o Voice Mode, Gemini Live) 的合成语音 naturalness 仍落后于人类。RLHF 在文本 LLM 中已成为关键对齐手段,但 speech RLHF 的 reward model 设计仍处于早期。
> 检索命中: [[LLM-based TTS]], [[Speech Tokenizer]], [[TTS Evaluation]] | 过滤: [[Differentiable Reward Optimization]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 Generative Speech Reward Model (GSRM),将语音自然度评估分解为 acoustic feature extraction + feature-grounded CoT reasoning 两阶段,实现可解释的多维度语音质量判断,并首次用于 online speech RLHF [论文原文]
> - **路线**: Raw audio → Phoneme-level forced alignment → Vowel-level acoustic features (pitch/intensity/duration) → Acoustic feature log → Text LLM (GPT-4o) CoT synthesis → SFT on Qwen2.5-Omni-7B → GSRM; Online RLHF: Speech LLM generator → GSRM verifier → reward → RL update
> - **指标**: Naturalness prediction PCC 0.465 (OOD), approaching human inter-rater 0.532 [Table 5]; Online RLHF: 82% A/B win rate in naturalness [Table 6]
> - **可借鉴**: (1) 将 speech evaluation 分解为 explicit feature extraction + reasoning,比端到端预测更可解释; (2) vowel-level acoustic features 作为 sufficient statistics; (3) GSRM 作为 universal verifier 实现 online speech RLHF
> - **局限**: (1) 依赖 forced alignment 工具 (phoneme-level),可能在 noisy/accented speech 上失败; (2) CoT 合成依赖 GPT-4o,成本高; (3) 仅在 conversational TTS 场景验证; (4) 31K 标注数据规模相对较小

## 核心问题

1. **传统 naturalness 评估器的局限**: 标量 MOS 预测器 (NISQA, UTMOSv2) 缺乏可解释性,跨域泛化差,且无法解释 *为什么* 某个语音不自然 [§1, §3] [论文原文]
2. **Frontier speech LLM 不擅长 speech evaluation**: Gemini-2.5-Pro 在 ConvTTS 测试集上 PCC 为 -0.050 (负相关!),说明即使最强 speech LLM 也难以可靠判断 naturalness [Table 2] [论文原文]
3. **Speech RLHF 缺乏可靠 verifier**: 文本 RL 的 verifier 可基于客观正确性,但语音质量是主观多维度的,需要新型 reward model [§5] [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

GSRM 将语音自然度判断分为两个显式阶段 [Fig 1, §4]:

**阶段 1: Acoustic Feature Extraction** [§4]
- Phoneme-level forced alignment 获取时间边界
- 仅保留 vowel segments (vowels 承载大部分韵律信息) [论文原文]
- 提取 6 维 vowel-level prosodic features [Table 3]:

| Feature | 含义 |
|---------|------|
| Pitch Level | 元音段平均 F0 |
| Pitch Var. | 段内频率变异性 |
| Pitch Slope | 频率变化趋势 (升/降) |
| Intensity Level | 平均 RMS 能量 |
| Intensity Var. | 段内强度波动 |
| Duration | 元音段时长 |

- 连续值离散化为 ordinal categories (very low / low / mid / high / very high) 增强鲁棒性 [论文原文]
- 输出: vowel-level acoustic feature log (结构化表格形式) [Fig 2]

**阶段 2: CoT Reasoning Synthesis** [§4]
- **Utterance-level evidence log**: GPT-4o 对每个 utterance 分析 4 个维度 [Table 4]:
  - Inferred Context (语义推断)
  - NSVs/Fillers (非语音发声/填充词)
  - Positive Attributes (优点)
  - Potential Issues (问题) [论文原文]
- **Global judgment CoT**: GPT-4o 基于所有 utterance evidence logs + oracle human ratings 合成全局判断,包含 per-sub-metric reasoning + 最终 7 维评分 [论文原文]
- 训练数据 = 拼接 (utterance evidence log + global judgment CoT) [论文原文]

**训练与部署** [§4]:
- Base model: Qwen2.5-Omni-7B (speech-in, text-out)
- SFT on 4,579 training examples (ConvTTS dataset), 10 epochs, full-parameter training [§6.1]
- 推理时: speech → GSRM → CoT reasoning + multi-dimensional ratings [论文原文]
- Test-time scaling: 多次采样取平均,K=4 即可获得较好收益 [Fig 4(b)] [论文原文]

### 关键设计选择

**1. 为什么用 vowel-level features 而非 frame-level?** [§4]
- Vowels 是声学稳定单元,携带韵律主要信息 (expressiveness, intonation, pacing) [论文原文]
- 减少信息冗余: 原始波形高维冗余,vowel features 作为 sufficient statistics [论文原文]
- Pitch features 贡献最大: 移除 pitch features 导致 PCC 下降 >0.12 [Fig 5] [论文原文]

**2. 多维度评估 rubric** [Table 1]
- 7 个子维度: Expressive Intensity, Expressive Correctness, Intonation, NSVS, Mispronunciation, Pacing, Human-likeness
- Intonation 和 Pacing 对整体 naturalness 贡献最大 [Fig 5] [论文原文]
- NSVS 和 Mispronunciation 贡献最小 [Fig 5] [论文原文]

**3. GSRM for Online RLHF** [§5, Fig 3]
- 扩展 GSRM 加入 semantic quality 评估 (language complexity, contextual awareness, spontaneity) [论文原文]
- Pipeline: Speech LLM generator → ASR transcription → GSRM (acoustic + semantic ratings) → reward aggregation → online RL update [论文原文]
- 训练集: 9.2K speech prompts [§6.4] [论文原文]

### 训练策略

- Human feedback 数据: ConvTTS (6,579 对话,62.8h 训练) + FDX-Conv (490 对话,OOD benchmark)
- 平均 6.8 标注者/样本,inter-rater PCC 0.533 (ConvTTS) / 0.532 (FDX-Conv)
- GSRM 训练: Qwen2.5-Omni-7B, SFT, 10 epochs, 4,579 examples
- Online RLHF: SFT base model → RL with GSRM verifier

## 实验

| 指标 | GSRM | AES-based Regressor | Gemini Speech Prompting | Human Oracle | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Pearson (↑) | **0.465** | 0.236 | -0.050 | 0.532 | FDX-Conv (OOD) | [Table 5] |
| Spearman (↑) | **0.427** | 0.224 | 0.140 | 0.489 | FDX-Conv (OOD) | [Table 5] |
| MSE (↓) | **0.230** | 0.265 | 0.531 | 0.336 | FDX-Conv (OOD) | [Table 5] |
| A/B Naturalness Win | **82%** | - | - | - | RLHF test | [Table 6] |
| A/B Intonation Win | **66%** | - | - | - | RLHF test | [Table 6] |
| A/B Pacing Win | **60%** | - | - | - | RLHF test | [Table 6] |

**关键发现**:
1. **GSRM 大幅超越所有 baseline**: PCC 0.465 vs 次优 AES 0.236,逼近 human inter-rater 0.532 [Table 5] [论文原文]
2. **Frontier speech LLM 无法评估 naturalness**: Gemini-2.5-Pro speech prompting PCC -0.050 (负相关!),text-based acoustic feature prompting (0.133) 反而更好 [Table 2] [论文原文]
3. **Evidence log 关键**: 移除 evidence log 导致 PCC 下降约 0.08 [Fig 5] [论文原文]
4. **Pitch 最重要**: 移除 pitch features 导致 PCC 下降 >0.12,intensity 影响较小 [Fig 5] [论文原文]
5. **Test-time scaling 有效**: K=4 samples 后 PCC 持续增长,K=16 达最优 [Fig 4(b)] [论文原文]
6. **Online RLHF 有效**: GSRM-trained model 在 naturalness A/B 测试中 82% 胜率 [Table 6] [论文原文]

## 局限性

1. **依赖 forced alignment**: Phoneme-level alignment 工具在 noisy/accented/non-standard speech 上可能失败,限制了鲁棒性 [agent 解读]
2. **CoT 合成成本**: 使用 GPT-4o 合成 4,579 个 CoT trajectories,成本高且不完全可控 [agent 解读]
3. **仅验证 conversational TTS**: ConvTTS 和 FDX-Conv 均为对话场景,未在 audiobook/singing/multilingual 场景验证 [agent 解读]
4. **Naturalness 偏重,忽略内容**: 主要评估韵律/自然度,未覆盖 WER/content accuracy 维度 [agent 解读]
5. **标注数据规模较小**: 31K ratings 在 deep learning 标准下不大,可能限制 scaling 潜力 [agent 解读]

## 点评

GSRM 是 **speech RLHF reward modeling 的重要突破**,核心洞见在于:

1. **Feature extraction + reasoning 解耦**: 不让 speech LLM 端到端预测分数 (它做不好,Gemini PCC=-0.05),而是先用 signal processing 提取 explicit acoustic cues,再让 text LLM reasoning [论文原文]
2. **Sufficient statistics 假设**: vowel-level features 作为 naturalness 判断的充分统计量,是信息论指导下的优雅设计 [论文原文]
3. **首次 online speech RLHF**: 将 GSRM 作为 universal verifier 接入 online RL loop,82% naturalness win rate 证明了端到端的有效性

**与 KB 已有知识的关系**:
- 与 [[TTS Evaluation]] 的核心议题直接相关: GSRM 回应了 "为什么传统 MOS predictor 不够" 的问题,提供了可解释的替代方案
- 与 [[Differentiable Reward Optimization]] [待确认] 互补: DiffRO 解决 "如何用 reward 优化 TTS",GSRM 解决 "如何获得好的 reward signal"
- 与 SpeechAlign 的关系: SpeechAlign 用 golden-vs-synthetic 构建 implicit reward,GSRM 用 explicit human feedback 构建 generative reward model

## 可复用的 idea

1. **Explicit feature extraction → reasoning**: 可推广到任何需要 speech quality judgment 的场景,不限于 naturalness
2. **Vowel-level acoustic features**: 简洁有效的语音韵律表征,适用于各种 speech analysis 任务
3. **GSRM as universal verifier**: online RL for speech 的通用方案,可替换不同 generator
4. **Structured annotation rubric**: 7 维子指标设计可作为 speech naturalness 评估的参考标准
