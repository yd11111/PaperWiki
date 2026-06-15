---
type: paper
tier: card
title: "TTS 语音评估方向工作汇总 (2025.6–2026.6)"
arxiv_id: ""
source: ""
authors: []
year: 2026
venue: "自编汇总"
tags: [survey, evaluation, benchmark, MOS, LLM-judge, TTS]
concepts: []
models: []
tasks: []
datasets: []
status: draft
created: 2026-06-15
updated: 2026-06-15
---

## 概述

本文汇总 2025 年 6 月至 2026 年 6 月期间 TTS 语音评估方向的主要工作,共 **23 篇**,按子方向分为四大类:

1. **TTS 综合 Benchmark** — 构建新的测试集/评估协议
2. **LLM/ALLM-based 评估器** — 用大模型做 judge/reward model
3. **MOS 预测方法** — 自动化 MOS 打分的模型与训练策略
4. **韵律/表现力评估** — 针对特定维度的评估

> 趋势判断: LLM-as-Judge 正在取代传统 MOS,从"给一个分"走向"多维诊断+可解释推理"; 同时 MOS 预测侧也在从回归转向偏好排序,与 RLHF 对齐。

---

## 一、TTS 综合 Benchmark (8 篇)

### [[EmergentTTS-Eval]] — TTS Emergent Abilities Benchmark
- **arXiv**: 2505.23009
- **核心**: 1645 test cases 覆盖 6 种挑战场景(复杂韵律/表现力/语言学),LALM-as-Judge 框架
- **关键结论**: GPT-4o-Audio (Ballad) 整体 win-rate 65%;Gemini 2.5 Pro 作为 judge 与人类 Spearman 90.5%
- **定位**: 目前覆盖面最广的 TTS 能力评估 benchmark

### [[MINT-Bench]] — 多语言 Instruction-Following TTS Benchmark
- **arXiv**: 2604.17958
- **核心**: 4 轴分层 taxonomy (难度/控制域/控制规格/细粒度模式),10 语言,3 层评估协议
- **关键结论**: Gemini 2.5-Flash EN 最强;Qwen3-TTS ZH 超越商用;LALM-human Spearman 67-77
- **定位**: instruction-following TTS 评估的多语言标准

### [[InstructTTSEval]] — 指令遵循 TTS 分层 Benchmark
- **arXiv**: 2506.16381
- **核心**: 12 个副语言特征 x 3 抽象层级 (APS/DSD/RP) x 2 语言 = 6K 测试用例
- **关键结论**: 商用 Gemini-Flash EN-Avg 88.7%;开源 VoxInstruct EN-Avg 仅 50.4%;人机一致率 79%
- **定位**: 与 MINT-Bench 互补,更侧重指令抽象层级的系统性

### [[NV-Bench]] — 非语言发声合成 Benchmark
- **arXiv**: 2603.15352
- **核心**: 14 类 NV (笑声/叹气等) 的 1651 条 paired GT 多语言数据,双维评估(指令对齐 + 声学保真度)
- **关键结论**: NV 合成远未解决,最佳系统 PCER 27.69%;IMOS vs PCER Spearman -0.65
- **定位**: 首个 NV-capable TTS 标准化 benchmark

### [[Swanbench-Speech]] — 长文本语音生成 Benchmark
- **arXiv**: 2605.28618
- **核心**: 首个系统性长篇幅语音生成评估基准,覆盖多样下游场景
- **定位**: 填补短句评估到长文本评估的空白

### [[CodecMOS-Accent]] — 跨口音 Codec/TTS MOS 评测集
- **arXiv**: 2603.14328
- **核心**: 24 系统 x 32 说话人 x 10 种英语口音,19600 条人工标注
- **关键结论**: Codec 和 TTS 在非标准口音上质量显著下降
- **定位**: MOS 评估中首次引入口音多样性维度

### [[CAST-Benchmark]] — 话语条件重音 TTS Benchmark (CAST)
- **arXiv**: 2604.10580
- **核心**: 对比性上下文对(同一句话 + 不同上下文 → 不同重音位置),测试 discourse-aware stress
- **关键结论**: 多数 TTS 系统缺乏 discourse-aware 重音能力
- **定位**: 韵律评估从"听起来自然"进化到"语义正确的重音"

### [[AudioMOSChallenge2025]] — AudioMOS Challenge 2025
- **arXiv**: 2509.01336
- **核心**: 首个涵盖语音/音乐/通用音频的自动质量预测挑战赛,三赛道,24 支队伍
- **关键结论**: Track 3 冠军 SRCC=0.955 vs baseline 0.749;SSL+集成为主流技术路线
- **定位**: MOS 预测方向的标准竞赛,推动了 UrgentMOS/DistilMOS 等后续工作

---

## 二、LLM/ALLM-based 评估器 (5 篇)

### [[AnyAudio-Judge]] — 动态 Rubric 音频指令评估器
- **arXiv**: 2606.03116
- **核心**: 将指令分解为 binary rubric items 逐项评估,7920 样本 benchmark + 105K 训练 + 专用 judge 模型
- **关键结论**: 超越 Gemini-2.5-Pro (ACC 85 vs 78);可直接作为 InstructTTS RL 的 reward model
- **定位**: rubric-based 评估范式的代表,兼做评估器和 reward model

### [[SpeechJudge]] — TTS 自然度偏好 Judge
- **arXiv**: 2511.07931
- **核心**: 99K pairwise 人类偏好数据 + 1K benchmark + GRPO 训练的 generative reward model
- **关键结论**: 77.2% accuracy (超 BTRM 72.7%);用于 TTS 后训练 N-CMOS +0.25
- **定位**: 首个 TTS naturalness 完整评估套件(数据+benchmark+模型)

### [[TTS-PRISM]] — 12 维分层诊断框架
- **arXiv**: 2604.22225
- **核心**: 中文 TTS 12 维评分 + 可解释推理,7B 模型单次推理,schema-driven instruction tuning
- **关键结论**: 12 维平均 LCC 0.717,超越 Gemini-2.5-Pro 和 Qwen3-Omni
- **定位**: "从单一 MOS 到多维诊断"的标杆,可解释性最强

### [[SpeechQualityLLM]] — LLM 多模态语音质量评估
- **arXiv**: 2512.08238
- **核心**: 将语音质量评估建模为 LLM 多模态理解任务,输入音频+文本,输出多维质量分数
- **定位**: LLM-as-Judge 在通信/VoIP 场景的应用,与 SpeechJudge 侧重 TTS 自然度互补

### [[CalibrationReasoningSQA]] — 校准-推理可解释评估
- **arXiv**: 2603.10175
- **核心**: 校准阶段对齐感知维度,推理阶段检测和分类音频伪影,超越单一 MOS
- **定位**: 与 TTS-PRISM 方向一致(可解释诊断)但技术路线不同(post-training vs schema-driven)

---

## 三、MOS 预测方法 (8 篇)

### [[GSRM]] — 生成式语音 Reward Model
- **arXiv**: 2602.13891
- **核心**: acoustic feature extraction + feature-grounded CoT reasoning 两阶段,可解释多维判断
- **关键结论**: 自然度预测 PCC 0.465 接近人类 inter-rater 0.532;Online RLHF 82% win rate
- **定位**: MOS → Reward Model 的桥梁,首次实现 online speech RLHF

### [[Vox-Evaluator]] — 多级评估器驱动纠错+DPO
- **arXiv**: 2510.20210
- **核心**: 统一三路输出(error localization + transcription + quality score),驱动 inference-time 纠错和 segment-level DPO
- **关键结论**: F5-TTS WER 1.73→1.42%;failure rate 12→6%
- **定位**: 评估器不只是打分,还能直接驱动 TTS 质量提升

### [[UrgentMOS]] — 多指标+偏好学习 MOS 预测
- **arXiv**: 2601.18438
- **核心**: 同时优化 MOS 回归和偏好排序,多质量维度,来自 URGENT Challenge 团队
- **定位**: MOS 预测方向 SOTA,Watanabe/Qian 团队出品

### [[DistilMOS]] — SSL 层级自蒸馏 MOS 预测
- **arXiv**: 2601.13700
- **核心**: 层级自蒸馏缓解 SSL fine-tune MOS 时的灾难性遗忘和过拟合
- **定位**: 解决 SSL→MOS 的核心工程问题

### [[SA-SSL-MOS]] — 频谱增强多采样率 MOS 预测
- **arXiv**: 2602.14785
- **核心**: 频谱增强让 SSL MOS 预测器支持 16-48kHz 多采样率
- **定位**: 实用性强,覆盖现实 TTS 输出的多种采样率

### [[DRASP]] — 双分辨率池化 MOS 预测
- **arXiv**: 2508.21407
- **核心**: 全局+帧级双分辨率注意力池化,更好编码质量信息
- **定位**: MOS 预测架构创新,来自中研院团队

### [[MOS-Reward]] — MOS→偏好 Reward Model
- **arXiv**: 2510.00743
- **核心**: 将 MOS 从绝对打分重定义为偏好排序,构建 MOS-Reward benchmark
- **关键结论**: 偏好对比绝对分数更稳定;Reward Model 可直接用于 TTS RLHF
- **定位**: 连接 MOS 评估与 RLHF,来自复旦 NLP 组

### [[MOS-Bias]] — MOS 性别偏差分析
- **arXiv**: 2603.10723
- **核心**: 首次系统分析 MOS 标注的性别偏差,男性听众系统性偏高
- **定位**: 挑战 MOS "金标准"假设,来自李宏毅+曹昱团队

### [[QAMRO]] — 排序优化的音频生成评估
- **arXiv**: 2508.08957
- **核心**: 自适应 margin ranking loss 替代回归损失,覆盖 TTS/TTM/TTA
- **定位**: 与 DRASP 同组(中研院),排序优化思路与 MOS-Reward 互补

---

## 四、韵律/表现力评估 (2 篇)

### [[ProsodyEval]] — 韵律多样性度量
- **arXiv**: 2509.19928
- **核心**: DS-WED (semantic token 加权编辑距离) + 首个人类标注韵律多样性数据集
- **关键结论**: DS-WED 与人类 PMOS 相关性 r=0.77,远超 F0 RMSE (0.30)
- **定位**: 首个可量化韵律多样性的自动指标

### [[IterateDifferentiate]] — 迭代放大区分度
- **arXiv**: 2603.24430
- **核心**: 递归用模型自身输出作为 reference,迭代合成放大差异,恢复客观指标对 SOTA 系统的区分力
- **关键结论**: UTMOSv2 system-level SRCC 从 0.118 提升至 0.464
- **定位**: 不训新模型,通过评估协议创新解决"SOTA 系统分不开"的问题

---

## 额外相关 (1 篇)

### [[VCBBench]] — 语音对话 LALM 评估
- **arXiv**: 2510.11098
- **核心**: 首个全真人录音的中文语音对话 LALM benchmark,三维评估(指令/知识/鲁棒性)
- **定位**: 虽非 TTS 评估,但 LALM 作为 judge 的能力直接影响 TTS 自动评估的天花板

---

## 趋势总结

| 趋势 | 代表工作 | 信号 |
|---|---|---|
| **LLM-as-Judge 替代 MOS** | TTS-PRISM, SpeechJudge, AnyAudio-Judge | 从单一分数到多维诊断+可解释推理 |
| **评估 → Reward Model** | GSRM, SpeechJudge, MOS-Reward, Vox-Evaluator | 评估器直接参与 TTS 训练 (RLHF/DPO) |
| **MOS 回归 → 偏好排序** | MOS-Reward, UrgentMOS, QAMRO | 偏好对更稳定,与 RLHF 范式对齐 |
| **Instruction-Following 评估** | MINT-Bench, InstructTTSEval, AnyAudio-Judge | InstructTTS 兴起催生专用 benchmark |
| **细粒度维度评估** | NV-Bench, CAST-Benchmark, CodecMOS-Accent, ProsodyEval | 从整体质量到口音/重音/NV/韵律逐项评估 |
| **MOS 本身的问题** | MOS-Bias, IterateDifferentiate, Survey-ResponsibleTTSEvaluation | MOS 有偏差/SOTA 分不开/评估伦理 |
