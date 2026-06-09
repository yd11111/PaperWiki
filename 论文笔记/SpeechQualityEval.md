---
type: paper
tier: deep
title: "Enabling Auditory Large Language Models for Automatic Speech Quality Evaluation"
arxiv_id: "2409.16644"
source: "Sources/SpeechQualityEval.pdf"
authors: [Siyin Wang, Wenyi Yu, Yudong Yang, Changli Tang, Yixuan Li, Jimin Zhuang, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Guangzhi Sun, Lu Lu, Yuxuan Wang, Chao Zhang]
year: 2025
venue: "arXiv"
tags: [speech-quality-assessment, auditory-LLM, MOS-prediction, speaker-similarity, multimodal-LLM, LoRA, A/B-testing, natural-language-description]
concepts: ["[[TTSEvaluation]]", "[[SpeakerEmbedding]]", "[[SpeakerVerification]]", "[[Self-SupervisedSpeechRepresentation]]", "[[AudioUnderstanding]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: ["[[模型库/WavLM|WavLM]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-09
updated: 2026-06-09
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeakerEmbedding]]✓, [[TTSEvaluation]], [[SpeakerVerification]], [[Self-SupervisedSpeechRepresentation]], [[AudioUnderstanding]], [[Speech-LLMIntegrationTaxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeakerEmbedding]]✓, [[TTSEvaluation]], [[SpeakerVerification]], [[Self-SupervisedSpeechRepresentation]], [[AudioUnderstanding]], [[Speech-LLMIntegrationTaxonomy]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文属于 TTS 评估方向中 "LLM-as-a-Judge" 的早期探索 (2024.09),将 auditory LLM (SALMONN / Qwen-Audio 系列) 应用于 MOS 预测、SIM 预测、A/B 测试和自然语言质量描述四个任务。在 [[TTSEvaluation]] 演进线中,本文处于 "Predicted MOS 自动化" 向 "LLM-as-Judge" 过渡的节点,早于后续的 GSRM (acoustic-feature-grounded CoT)、SpeechJudge (pairwise preference GRM)、TTS-PRISM (multi-dimensional diagnostic) 和 UniSRM (unified multi-task reward model) 等更成熟的方案。

**已有认知**: (1) [[SpeakerEmbedding]] 详细记录了 ECAPA-TDNN 等 speaker encoder 在 SECS/SIM 计算中的角色,本文用 ECAPA-TDNN 作为 SSL 多任务 baseline 的 SIM 预测下游模型; (2) [[SpeakerVerification]] 指出 SECS 结果高度依赖所选 speaker encoder,不同 encoder 可差 0.1-0.3; (3) [[Self-SupervisedSpeechRepresentation]] 覆盖了 WavLM 的 masked speech denoising 和 layer-wise 特性,WavLM-Large 在本文中作为多用途 SSL baseline; (4) [[Speech-LLMIntegrationTaxonomy]] 将 SALMONN 归类为 latent-representation-based integration (Q-Former adapter),Qwen-Audio 也属同类但用 pooling adapter; (5) [[AudioUnderstanding]] 记录了 auditory LLM 在多种理解任务上的能力和局限,本文将理解能力拓展到 speech quality 维度。

**创新判断**: 本文的核心价值在于 **首次系统性地将 auditory LLM 用于 MOS/SIM 预测 + A/B 测试 + NL 描述的多任务统一评估**,用 task-specific prompts + LoRA 微调实现单一模型覆盖多种评估任务。但从后续工作看,其方法论较为朴素 (直接回归分数 / 输出文本,无 CoT 推理、无 pairwise preference 建模),性能也未超越任务专用 SOTA。

## 速查

> [!summary] 速查
> - **一句话**: 用 LoRA 微调 auditory LLM (SALMONN/Qwen-Audio) + task-specific prompts,实现 MOS/SIM 预测、A/B 测试、NL 质量描述的多任务统一语音评估
> - **路线**: 音频 → audio encoder (BEATs+Whisper / Whisper-large-v2) → connection module (Q-Former / pooling) → frozen LLM (Vicuna / Qwen-7B) + LoRA → 文本输出 (分数 / 选择 / 描述)
> - **指标**: MOS 预测 BVCC sys-level LCC 0.884 (vs SOTA 0.939); SIM 预测 VoxSim LCC 0.816 (vs SOTA 0.835); A/B 测试 acc 0.698 (MOS diff>0.5 时 0.803) [Table I-IV]
> - **可借鉴**: dataset-specific prompt 策略提升跨数据集泛化; averaging multiple prompts 增强鲁棒性; 单一 LLM 模型统一多种评估任务的可行性验证
> - **局限**: A/B 测试整体准确率 <70% 不够实用; NL 描述缺乏专用训练数据导致输出单调; Qwen-Audio 系列在 MOS/SIM 上显著弱于 SALMONN; 无 CoT 推理能力,后续工作 (GSRM/SpeechJudge/UniSRM) 已大幅超越

## 核心问题

1. **auditory LLM 能否胜任语音质量评估?** 传统方案用任务专用小模型 (MOSNet/UTMOS/ECAPA-TDNN) 分别处理 MOS/SIM/A/B 测试,能否用一个 auditory LLM 统一覆盖?
2. **LoRA 微调足够吗?** 开源 auditory LLM 在未训练任务上无法准确遵循指令 [§I],微调能否弥补?
3. **dataset-specific prompt 能否解决跨数据集泛化?** MOS 评分标准因数据集而异 (NISQA/BVCC/SOMOS 各有特点),prompt 策略能否让模型区分?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

auditory LLM 由三个组件构成 [§III-A]:
1. **Audio encoder**: 提取音频特征
2. **Connection module**: 将 audio encoder 输出空间对齐到 LLM 文本输入空间
3. **LLM**: 基于对齐后的音频特征 + 文本 prompt 生成回答

三个模型架构差异 [§III-A]:

| 模型 | Audio Encoder | Connection Module | LLM Backbone | 可训练参数 |
|------|------|------|------|------|
| SALMONN | BEATs + Whisper (双编码器,冻结) | Window-level Q-Former (~0.33s/token, 88 tokens/30s) | Vicuna-v1.5-7B 或 v1.0-13B | Q-Former + LoRA |
| Qwen-Audio | Whisper-large-v2 (可训练) | Stride-2 pooling | Qwen-7B (冻结) | Audio encoder |
| Qwen2-Audio | 改进 Whisper + 改进 LLM | 同 Qwen-Audio | 改进 Qwen | + RLHF |

### 关键设计选择

**1. Task-specific Prompts [§III-B, Fig 1(b)]**

四个任务通过 prompt 区分,每个任务 3 个语义近似的 prompt 变体用于训练,测试时随机选 1 个:

- **MOS 预测**: "Could you please show me a score of the quality of this speech sample according to {Dataset} standards?" — 输出 1-5 分值 [论文原文]
- **SIM 预测**: "Please give a score of the speaker similarity of the voices in the audio." — 输出 1-6 分值(精度 0.1) [论文原文]
- **A/B 测试**: "Could you please indicate which speech sample has better quality?" — 输出 "The former" / "The latter" [论文原文]
- **NL 描述**: "Please evaluate the quality of the speech sample." — 输出自然语言评价 (noisiness, distortion, discontinuity, overall quality) [论文原文]

**2. Dataset-specific Prompts [§V-A, Table II]**

MOS 预测引入 "according to {NISQA/BVCC/SOMOS} standards" 短语,使模型根据数据集特点调整预测。消融实验显示 [Table II]:
- 对应数据集的 prompt 取得最高 system-level 性能 (BVCC standards → BVCC LCC 0.884)
- 平均 3 个 prompt 分数在 utterance-level 最优 (LCC 0.829) [论文原文]
- 使用非对应 prompt 会轻微损害性能 [论文原文]

[agent 解读] 这说明 MOS 标准确实存在数据集偏差 (NISQA 的评分分布与 BVCC 不同),prompt 有效地编码了这种偏差。但对新数据集,averaging 策略消除了创建新 prompt 的需求。

**3. LoRA 微调策略 [§IV-B]**

对 SALMONN: Q-Former + LLM 上的 LoRA (rank=8, scale=32); encoder 冻结。训练 10 epochs,验证集选最优。四个任务联合训练 [§III-B]。

**4. A/B 测试的输入处理 [§IV-A]**

两个语音样本用 2 秒静音分隔后拼接,总长超过 14 秒截断至 14 秒 (仅 SALMONN) [论文原文]。

**5. NL 描述的训练数据构建 [§IV-A]**

从 NISQA 数据集的 noisiness/distortion/discontinuity/overall quality 分数,按 rating descriptions 模板生成评价文本,再用 LLama3-8B-Instruct 改写增加多样性 [论文原文]。训练 10,899 / 验证 2,635 / 测试 712 样本。

### 训练策略

- **多任务联合训练**: MOS (NISQA+BVCC+SOMOS) + SIM (VoxSim) + A/B testing (SOMOS-derived) + NL descriptions (NISQA-derived),通过 prompt 区分任务 [§III-B]
- **Baseline 对比**: SSL 多用途 baseline (WavLM-Base+/Large) 联合训练 MOS + SIM,MOS 用 dataset-specific linear layers,SIM 用 ECAPA-TDNN [§IV-C, Fig 1(a)]

## 实验

| 指标 | 本文 (SALMONN vic1.5) | Single-task SOTA | Baseline (WavLM-Large) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS LCC (utt) | 0.861 | 0.894 (NISQA) | 0.850 | NISQA | [Table I] |
| MOS LCC (utt) | 0.826 | 0.899 (UTMOS) | 0.813 | BVCC | [Table I] |
| MOS LCC (sys) | 0.884 | 0.939 (UTMOS) | 0.884 | BVCC | [Table I] |
| MOS LCC (utt) | 0.644 | 0.687 (SSL-MOS) | 0.623 | SOMOS | [Table I] |
| MOS LCC (sys) | 0.894 | 0.911 (SSL-MOS) | 0.880 | SOMOS | [Table I] |
| SIM LCC | 0.796 | 0.835 (WavLM-ECAPA) | 0.658 | VoxSim | [Table III] |
| SIM LCC (SALMONN vic1.0) | **0.816** | 0.835 | 0.658 | VoxSim | [Table III] |
| A/B Acc | 0.670 (0.761) | - | - | SOMOS | [Table IV] |
| A/B Acc (Qwen2-Audio) | **0.698** (0.803) | - | - | SOMOS | [Table IV] |
| NL Description Corr (finetuned) | **0.64** | - | - | NISQA | [Table V] |
| NL Description Corr (Gemini 1.5 Pro, pretrained) | 0.43 | - | - | NISQA | [Table V] |

**关键发现**:

1. **SALMONN >> Qwen-Audio 系列**: 在 MOS 和 SIM 任务上,SALMONN 一致优于 Qwen-Audio/Qwen2-Audio。Qwen-Audio 微调后 SIM 全部预测为 1,完全失败 [§V-B]。[agent 解读] SALMONN 的双编码器 (BEATs + Whisper) + Q-Former 在音频质量感知上可能优于 Qwen-Audio 的单 Whisper 编码器,且 SALMONN 的 LoRA 对 LLM 的微调比 Qwen-Audio 仅调编码器更灵活。

2. **LLM backbone 质量影响大**: SALMONN (vic1.5, 7B) > SALMONN (vic1.0, 13B),作者推测 Vicuna 1.5 的 LLM 质量更高补偿了参数量劣势 [§V-A]。

3. **Utterance-level vs system-level 不一致**: vic1.5 在 SOMOS utterance-level 优于 vic1.0,但 system-level 相反,这在 SSL 模型中也有观察 [§V-A]。

4. **Dataset-specific prompt 有效但有限**: 对应 prompt 最优,averaging 最鲁棒,非对应 prompt 轻微损害 [Table II]。

5. **A/B 测试 <70%**: 整体准确率不足以实用,但 MOS 差距 >0.5 时达 80%,说明模型能区分明显质量差异但对细微差异无力 [§V-C]。

## 局限性

1. **A/B 测试准确率不够实用**: 整体 <70%,在 MOS 差距小的样本上近乎随机,难以替代人类 A/B 测试 [§V-C]
2. **NL 描述输出单调**: 缺乏专用多样化训练数据,微调后输出仍缺乏多样性和精细度 [§V-D]
3. **Qwen-Audio 系列表现差**: Qwen-Audio SIM 预测完全失败,MOS 也显著弱于 SALMONN,限制了方法的通用性 [Table I, III]
4. **无推理能力**: 直接输出分数/选择,无 CoT 推理过程,无法解释判断依据 (后续 GSRM、SpeechJudge 通过 acoustic-feature-grounded reasoning 大幅改进)
5. **评估方法局限**: NL 描述质量用 GPT-4o-mini 评分,其可靠性未验证 [§IV-D]
6. **MOS 预测未达 SOTA**: 与单任务 SOTA (NISQA/UTMOS/SSL-MOS) 仍有差距,尤其 system-level [Table I]
7. **SIM 预测的 MSE 较高**: LCC/SRCC 接近 SOTA 但 MSE 1.199-1.374 远高于 SOTA 的 0.943 [Table III]
8. **14 秒截断**: A/B 测试和 SIM 预测中 SALMONN 截断超长样本至 14 秒,可能丢失信息 [§IV-A]

## 点评

本文是 **auditory LLM 用于语音质量评估的早期系统性探索**,价值在于验证了"单一模型通过 prompt 切换评估任务"的可行性。SALMONN 在 MOS/SIM 上达到 competitive 但未超越 SOTA 的结果,证明 auditory LLM 有潜力但方法论尚不成熟。

**与后续工作的对比**:
- **GSRM** (2026): 引入 acoustic-feature-grounded CoT reasoning,将 naturalness 评估分解为特征提取+推理两阶段,OOD PCC 0.465 vs 本文无 OOD 泛化分析
- **SpeechJudge** (2025): 专攻 naturalness pairwise preference,77.2% accuracy (vs 本文 A/B 69.8%),且证明所有现有客观指标 (含 UTMOS) 在 naturalness 判断上接近随机
- **UniSRM** (2026): 同样追求多任务统一,但加入 reasoning-consistent rewards,在 A/B preference 达 65.06% (UniSRM-Bench 更难),cross-dataset PCC 0.498
- **TTS-PRISM** (2026): 多维度 diagnostic + schema-driven instruction tuning,远超简单分数预测

本文的 dataset-specific prompt 策略和 averaging 思路仍有参考价值,但整体方法论已被后续工作全面超越。作为时间线上的标记,本文清晰展示了 "直接用 LLM 回归分数" 路线的天花板,推动了领域向 reasoning-based 和 preference-based 评估转型。

## 审阅

> [!review] 审阅 pass (2026-06-09, auto)
> 0 high / 0 medium / 3 low issues — 详见 [[_review/SpeechQualityEval-review.yml]]
> - [low] traceability-gap: LoRA rank/scale 数字未标注出处
> - [low] traceability-gap: NL 描述训练样本数未标注出处
> - [low] template-compliance: frontmatter datasets 为空数组,实际用了 BVCC/SOMOS/NISQA/VoxSim

## 可复用的 idea

1. **Dataset-specific prompt for domain adaptation**: 在 prompt 中嵌入 "according to {Dataset} standards" 让模型学习不同评分标准的偏差,可推广到任何需要跨域适配的 LLM 评估任务
2. **Multi-prompt averaging for robustness**: 对新域外数据,averaging 多个 prompt 的预测比选择单一 prompt 更鲁棒,消除了为新数据集设计 prompt 的需求 [Table II]
3. **SSL multipurpose baseline design**: 用 task-specific layer weights + task-specific downstream heads (linear for MOS, ECAPA-TDNN for SIM) 构建公平的多任务 baseline [Fig 1(a)]
4. **Silence-separated concatenation for paired input**: 用 2 秒静音分隔两段语音作为 LLM 输入,简单有效地处理 pair comparison 任务 [§IV-A]
5. **LLM-based data augmentation for evaluation descriptions**: 用 LLM (LLama3) 改写模板化评价文本增加多样性,低成本获取 NL description 训练数据 [§IV-A]
