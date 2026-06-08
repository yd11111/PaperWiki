---
type: paper
tier: deep
title: "QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning and Descriptions"
arxiv_id: "2503.20290"
source: "Sources/QualiSpeech.pdf"
authors: [Siyin Wang, Wenyi Yu, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, Yu Tsao, Junichi Yamagishi, Yuxuan Wang, Chao Zhang]
year: 2025
venue: "ACL 2025"
tags: [speech-quality-assessment, natural-language-description, auditory-LLM, dataset, benchmark, MOS, low-level-perception, evaluation]
concepts: ["[[TTSEvaluation]]", "[[AudioUnderstanding]]", "[[Audio-LanguagePretraining]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个实体页: [[TTSEvaluation]], [[AudioUnderstanding]], [[Audio-LanguagePretraining]], [[Whisper]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: QualiSpeech 位于 TTS Evaluation 演进线中的 "LLM-as-Judge + 自然语言评估" 分支。TTSEvaluation 页已记录从 MOS → Predicted MOS → LLM-as-Judge → Distributional Evaluation (TTSDS/TTSDS2) → Multi-dimensional Diagnostic (TTS-PRISM) → Unified Reward Model (UniSRM) 的演进路径。QualiSpeech 的核心贡献在**数据层** — 它不是一个评估模型,而是第一个为自然语言语音质量评估提供细粒度人工标注的 dataset + benchmark。
>
> **已有认知**: TTSEvaluation 页的 LLM-as-Judge 节已提及 QualiSpeech 为代表工作之一,但未展开。GSRM/SpeechJudge/TTS-PRISM/UniSRM 等后续工作各自走 acoustic-feature-grounded / pairwise preference / multi-dimensional schema / multi-task 路线,而 QualiSpeech 走的是"NL 描述式评估数据"路线,为这些 evaluator 提供训练/评估资源。AudioUnderstanding 页记录了 auditory LLM 在 high-level 理解任务上的进展,但指出 low-level speech perception 任务被忽视 — QualiSpeech 正是填补此空白。SALMONN (Whisper+BEATs+Vicuna) 是 QualiSpeech 实验中用于 finetuning 的基础模型。
>
> **创新判断**: 与 BVCC (仅 MOS score,仅合成语音) 和 NISQA (仅真实语音,仅 4 维度) 相比,QualiSpeech 首次 (1) 统一覆盖合成+真实语音, (2) 提供 11 维低级感知标注, (3) 加入自然语言描述+推理。这是一个数据资源贡献而非模型创新,后续 GSRM/UniSRM 等模型工作的评估实验已部分使用 QualiSpeech 或类似数据。
>
> 检索命中: [[TTSEvaluation]]✓, [[AudioUnderstanding]]✓, [[Audio-LanguagePretraining]]✓, [[Whisper]]✓ | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个覆盖合成+真实语音的 11 维低级语音质量评估数据集,含自然语言描述和推理,配套 auditory LLM benchmark
> - **路线**: 多源语音 → 人工标注(7 项打分 + 4 项描述) → GPT 生成 NL 描述 → 人工修正 → 数据集;SALMONN-7B + LoRA finetune → NL 质量评估
> - **指标**: finetuned SALMONN noise PCC 0.70, distortion 时间定位 IoU 0.78 [Table 2]; 原始 auditory LLM 近零 PCC [Table 1]; GPT-4o-mini reasoning acc 0.46 vs Vicuna 0.28 [Table 4]
> - **可借鉴**: 三步标注流程(人工元信息 → GPT 生成描述 → 人工修正)可迁移到任何需要高质量 NL 标注的评估数据集; 对描述式评估的多维度拆分评估方法(precision/recall + GPT correlation + IoU)
> - **局限**: 每条样本仅 1 名标注员(vs MOS 多人); 英语单语; finetuned 模型推理能力受限于 Vicuna backbone; 数据集规模 ~10K 偏小

## 核心问题

QualiSpeech 要解决的核心问题是: **现有语音质量评估停留在标量分数 (MOS),无法解释"为什么给这个分数",缺乏细粒度的诊断信息** [§1]。具体痛点:

1. MOS 只给一个数字,不揭示分数背后的质量因素 (噪声类型? 失真时段? 停顿位置?)
2. 现有数据集要么只覆盖合成语音 (BVCC),要么只覆盖真实语音 (NISQA),没有统一的跨域评估数据
3. auditory LLM 虽有自然语言生成能力,但 low-level 语音感知能力未被系统评估和开发

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

QualiSpeech 的核心贡献是**数据集 + benchmark**,不是一个新模型。围绕三个组件:

1. **QualiSpeech Dataset**: 多源语音 + 11 维人工标注 + NL 描述 [§3.1]
2. **QualiSpeech Benchmark**: 7 维多选题形式的 auditory LLM 评估基准 [§3.2]
3. **Finetuned SALMONN-7B**: 在 QualiSpeech 上微调的语音质量评估模型 [§4.2]

### 数据集设计

**数据来源** [§3.1.1, Fig 2]:
- **合成语音 (49%)**: BVCC (4546 train) + 10 个近期 TTS 系统各 72 样本 (CosyVoice, F5-TTS, ChatTTS, XTTS v2, OpenVoice V1/V2, Parler-TTS Mini/Large, VoiceCraft-830M, E2 TTS)
- **真实语音 (24%)**: GigaSpeech 2000 样本 (按 UTMOS 预测质量分 4 组平衡采样) + NISQA LIVE 529 样本
- **模拟真实 (27%)**: NISQA SIM 2883 样本 (9 种基础失真类型的 1258 种组合,取 1/5)
- 合成语音 20% 混入 DNS Challenge 噪声 (SNR 0-15 dB),因为 TTS 模型通常训练在干净数据上,单独不会产生"既不自然又有噪声"的样本 [§3.1.1] [论文原文]
- 规模: train 10,558 / val 2,167 / test 1,852 [Table 5]

**标注体系** [§3.1.2, Fig 3]:

| 类型 | 维度 | 标注方式 |
|------|------|----------|
| 打分 (7 维) | 噪声/失真/语速/连续性/听力负担/自然度/整体质量 | 1-5 分 Likert 量表 |
| 描述 (4 维) | 噪声(类型+时间)/失真(类型+时间)/不自然停顿(时间)/声音感受(年龄+性别+语气) | 自由文本 |

**三步标注流程** [Fig 3]:
1. **Step 1 — 人工基础标注**: 听众对每条样本标注 7 项分数 + 4 项简短描述
2. **Step 2 — GPT 生成 NL 描述**: 将所有标注元信息输入 GPT-4o-mini,以 Chain-of-Thought 格式生成完整描述段落(先逐维度分析,再给出整体评价)
3. **Step 3 — 人工修正**: 标注员审核 GPT 生成文本,纠正幻觉/不准确,补充遗漏信息,改善推理逻辑

**为什么用三步而非直接人写**: 直接人写高质量 NL 描述成本极高,且标注员间一致性难保证; GPT 生成提供结构化起点,人工修正确保准确性 [agent 解读]。

### 关键设计选择

**为什么选 11 个维度**: 作者参考了 NISQA (noise, distortion, continuity), PESQ 相关研究 (Hu & Loizou, 2007), 以及语音学文献 (speaking speed: Ren et al., 2019; listening effort: Winn & Teece, 2021; naturalness: Sellam et al., 2023),目标覆盖基础退化 (noise, distortion) 和主观感知 (speed, continuity, naturalness, listening effort) 两个层面 [Appendix B] [论文原文]。

**为什么统一合成+真实语音**: 此前合成语音评估 (BVCC) 和真实语音评估 (NISQA) 被视为独立任务 — 合成语音通常无噪声但缺自然度,真实语音更受噪声影响。通过提供细粒度维度分数和 NL 推理,QualiSpeech 旨在建立能同时处理两类语音的通用评估模型 [§3.1] [论文原文]。

**Benchmark 为什么用多选题而非开放问答**: 当前 LLM 在未见过的开放式指令上响应不稳定,多选题提供可控的评估环境 [§3.2] [论文原文]。

**描述评估为什么用 4 种 metrics (precision/recall/GPT-correlation/IoU)**: 单一 metric 无法捕捉描述质量的不同维度 — precision/recall 衡量是否检测到噪声/失真的存在,GPT-correlation 衡量描述内容的语义相关性,IoU 衡量时间定位精度 [§3.3] [论文原文]。仅在模型成功检测到目标时才计算 correlation/IoU,避免 recall 低的模型被漏检样本拉低 [§3.3]。

### 训练策略

Finetuned 模型基于 SALMONN-7B [§4.2]:
- **Speech encoder**: Whisper (frozen) + BEATs (frozen 或 finetune)
- **Connector**: Q-Former
- **LLM backbone**: Vicuna-v1.5-7B + LoRA
- 默认仅训练 Q-Former connector + LoRA,encoder frozen [Fig 4]

**两阶段训练**:
1. **低级特征学习**: 多选题形式 — 选择分数 + 描述各维度 (basic/balanced/joint 策略) [§4.2.1]
2. **NL 描述生成**: 在联合训练 checkpoint 上继续 finetune,生成完整描述段落 [§4.2.2]

训练均为 10 epochs。

## 实验

### Benchmark 结果: 现有 auditory LLM 的质量感知能力

| 模型 | Noise PCC | Distortion PCC | Overall PCC | 出处 |
| --- | --- | --- | --- | --- |
| SALMONN-7B | 0.003 | 0.013 | 0.084 | [Table 1] |
| SALMONN-13B | 0.001 | 0.002 | 0.100 | [Table 1] |
| Qwen-Audio-Chat | 0.014 | -0.003 | 0.250 | [Table 1] |
| Qwen2-Audio-7B-Instruct | -0.048 | 0.056 | 0.112 | [Table 1] |
| WavLLM | -0.021 | -0.069 | 0.071 | [Table 1] |

所有模型 PCC 接近 0 或为负,表明现有 auditory LLM 无法有效评估语音质量。SALMONN-7B 严重数值偏好 — 将所有 listening effort 和 continuity 预测为 4,所有 noise 预测为 3 [§4.1]。

### Finetuned 模型: 低级特征学习

| 训练策略 | Noise | Distortion | Overall | Noise IoU | Distortion IoU | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| basic | 0.721 | 0.553 | 0.597 | 0.85 | 0.78 | [Table 2] |
| joint | 0.693 | 0.595 | 0.636 | 0.76 | 0.78 | [Table 2] |
| joint + balance | 0.696 | 0.614 | 0.660 | 0.75 | 0.79 | [Table 2] |

joint + balance 在多数维度最优。**关键发现**: 描述式评估中 distortion IoU 达 0.78,表明模型在识别失真后能准确定位时间段; 但 precision/recall 显示模型检测噪声/失真存在的能力仍有提升空间 [§4.2.1]。

### Finetuned 模型: NL 描述生成

| 描述格式 | Noise PCC | Overall PCC | Distortion IoU | 出处 |
| --- | --- | --- | --- | --- |
| revised concise | 0.656 | 0.630 | 0.78 | [Table 3] |
| concise with num | 0.703 | 0.622 | 0.77 | [Table 3] |
| concise | 0.642 | 0.582 | 0.73 | [Table 3] |
| detailed | 0.686 | 0.572 | 0.73 | [Table 3] |

包含数值分数的描述 (concise with num) 在 noise PCC 上最优; 人工修正版 (revised concise) 在 overall PCC 上最优 [Table 3]。**文本长度对性能影响不大**,但修正幻觉至关重要 [§4.2.2]。

### Reasoning 能力: 文本 LLM 推理实验

| 模型 | Overall Quality Reasoning Acc | 出处 |
| --- | --- | --- |
| Vicuna-v1.5-7B (SALMONN backbone) | 0.28 | [Table 4] |
| GPT-4o-mini | 0.46 | [Table 4] |

给定 ground-truth 低级特征,GPT-4o-mini 可超过所有 finetuned auditory LLM,说明 reasoning 瓶颈部分来自 LLM backbone 能力不足 [§4.2.2] [论文原文]。

### Encoder Finetuning 消融

| 设置 | Overall PCC | 出处 |
| --- | --- | --- |
| freeze | 0.636 | [Table 7] |
| Whisper finetune | 0.643 | [Table 7] |
| BEATs finetune | 0.654 | [Table 7] |
| Whisper + BEATs finetune | 0.670 | [Table 7] |

finetune encoder 可带来进一步提升,作者认为联合 finetune 使低级语音信息被转换为 LLM backbone 更易理解的格式 [Appendix J.1] [论文原文]。

### 泛化实验

仅在一种数据类型上训练的模型泛化到其他域表现差; 混合训练 (all) 在所有域上表现最优 [Table 8]。这验证了 QualiSpeech 统一合成+真实语音的设计必要性 [§4.2.1] [论文原文]。

### 多标注一致性

BVCC 测试子集上收集 3 份标注,互相关 PCC: noise 0.728, distortion 0.682, overall 0.603, speed 仅 0.316 [Table 10]。说明语音质量评估本身主观性强; 模型训练在单标注上对齐多标注均值 PCC 最高 (noise 0.831, overall 0.732) [Table 12],暗示多标注聚合可提供更稳定评估 [Appendix J.4]。

## 局限性

1. **单标注员**: 每条样本仅 1 名标注员,MOS 通常要求多人评估。互相关实验显示标注间一致性有限 (speed PCC 仅 0.316) [Appendix J.4]
2. **英语单语**: 仅覆盖英文语音,限制了跨语言泛化
3. **LLM backbone 瓶颈**: Vicuna-v1.5-7B 推理能力弱,NL reasoning 未能提升整体评分预测精度 [Table 4]
4. **数据集规模**: 总计 ~14.5K 样本 (10K train),在 deep learning 标准下偏小
5. **描述覆盖不完整**: 11 个维度无法穷尽所有低级语音特征 (如 reverberation, clipping 等)
6. **标注员背景**: 25 名标注员均为非母语英语者 (普通话母语,IELTS 7.0+),虽有文献支持非母语-母语听众相关性 [Appendix C],但仍可能引入系统偏差

## 点评

**贡献定位**: QualiSpeech 的主要价值是**资源贡献** — 它不提出新的模型架构,而是填补了自然语言语音质量评估领域的数据空白。在 GSRM/SpeechJudge/TTS-PRISM/UniSRM 等模型工作之前,QualiSpeech 为"auditory LLM 能否做细粒度质量评估"这一问题提供了第一个标准化答案。

**设计亮点**:
- 三步标注流程 (人工→GPT→人工修正) 是实用的 NL 标注方法论,平衡了成本和质量
- 评估 metrics 设计周全 — 4 种互补指标覆盖存在性检测、语义匹配和时间定位
- 统一合成+真实语音的设计理念正确,泛化实验也验证了这一点

**与后续工作的关系**: QualiSpeech 是 2025 初期工作,此后 GSRM (acoustic-feature-grounded CoT)、SpeechJudge (pairwise preference GRM)、TTS-PRISM (12 维 diagnostic) 和 UniSRM (多任务统一) 各走更深的模型路线。QualiSpeech 的角色更像是"问题定义者 + 初始数据" — 它证明了 auditory LLM 可以生成有意义的 NL 描述 (IoU 0.78),也暴露了 reasoning 瓶颈 (Vicuna 0.28 vs GPT 0.46),推动了后续工作选择更强 backbone (Qwen2.5-Omni-7B 等)。

**不足之处**:
- 模型实验深度有限 — 仅在 SALMONN-7B 一个架构上验证,未尝试 Qwen-Audio/Gemini 等更强 backbone 的 finetuning
- NL reasoning 的实验设计 (Table 4) 使用 ground-truth 特征输入 text LLM,但未探索如何让 auditory LLM 本身提升推理能力 (如 CoT prompting、RLHF 等)
- 与 NISQA 的 MOS 对齐 PCC 仅 0.55-0.66 [Table 6],整体质量分的跨数据集一致性存疑

## 可复用的 idea

1. **三步 NL 标注流程** (人工基础标注 → LLM 生成描述 → 人工修正): 可迁移到任何需要高质量 NL 评估标注的场景 (TTS style 评估、prosody 评估、audio editing 评估)
2. **描述式评估的多维 metrics 拆分**: precision/recall (存在性) + GPT-correlation (语义匹配) + IoU (时间定位) 组合,可用于任何涉及 temporal localization 的 NL 评估任务
3. **UTMOS 质量分层采样**: 对 GigaSpeech 等大规模数据,用 predicted MOS 分 4 组等比采样,确保质量分布均衡,可迁移到其他质量相关数据集构建
4. **合成数据混噪策略**: TTS 输出通常干净,手动混入噪声 (SNR 0-15 dB) 创造 "不自然+有噪声" 样本,填补训练数据的组合空白

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 数据集设计、标注流程、模型训练策略的因果逻辑清晰 |
> | 可信赖 | pass | 关键数字均有出处标注,指标使用正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注到位 |
> | 可定位 | pass | KB 背景提供了在 TTS Evaluation 演进线中的清晰定位 |
> | 不污染 | pass | 无新建概念页需求 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/QualiSpeech-review.yml`
