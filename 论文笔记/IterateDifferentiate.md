---
type: paper
tier: deep
title: "Iterate to Differentiate: Enhancing Discriminability and Reliability in Zero-Shot TTS Evaluation"
arxiv_id: "2603.24430"
source: "Sources/IterateDifferentiate.pdf"
authors: [Shengfan Shen, Di Wu, Xingchen Song, Dinghao Zhou, Liumeng Xue, Meng Meng, Jian Luan, Shuai Wang]
year: 2026
venue: "arXiv preprint"
tags: [TTS, evaluation, zero-shot, iterative-evaluation, metrics, score-saturation, discriminability, human-alignment, robustness]
concepts: ["[[TTSEvaluation]]"]
models: ["[[CosyVoice]]", "[[CosyVoice2]]", "[[CosyVoice3]]", "[[论文笔记/F5-TTS|F5-TTS]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/FireRedTTS2|FireRedTTS2]]", "[[论文笔记/GLM-TTS|GLM-TTS]]", "[[论文笔记/VoxCPM|VoxCPM1.5]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[CV3-Eval]]", "[[LibriTTS]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[TTSEvaluation]], [[Zero-shotSpeechSynthesis]], [[CosyVoice3]], [[SEED-TTS-Eval]], [[CV3-Eval]], [[ConditionalFlowMatching]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: I2D 位于 TTS 评估方法的演进线上,与 TTSDS/TTSDS2 (distributional evaluation)、SpeechJudge/GSRM (naturalness-specific GRM)、TTS-PRISM (multi-dimensional diagnostic)、MCLP (continuation likelihood) 等路线并列。这些方法各自从不同角度解决同一个核心问题: SOTA 模型时代传统指标失效。I2D 的独特之处在于它不引入新指标或新模型,而是改变评估协议本身(迭代合成 + 指标聚合)来复活已有指标的区分力。
>
> **已有认知**: TTSEvaluation 页面详细记录了当前评估体系的已知局限 — MOS ceiling effect、predicted MOS (DNSMOS/UTMOS) 跨域泛化差、WER/SIM 饱和后不反映感知增益。Yang et al. (2025) 的 Responsible Evaluation 框架提出三层评估标准,其中 Level 1 (Fidelity & Accuracy) 正是 I2D 要解决的层面。SEED-TTS-Eval 数据集页面也已标注"各系统分数趋于接近,区分度下降",与 I2D 的出发点一致。
>
> **创新判断**: 相比 KB 中已有的评估改进路线(TTSDS2 用 Wasserstein 距离做分布级评估、SpeechJudge 训练专用 naturalness GRM、TTS-PRISM 多维诊断),I2D 的思路更为正交: 它不需要训练任何新模型,而是通过"压力测试"式的迭代合成来暴露模型间的鲁棒性差异。这种方法论上的简洁性是其最大优势,但代价是计算成本的线性增长。
>
> 检索命中: [[TTSEvaluation]][待确认], [[Zero-shotSpeechSynthesis]]✓, [[CosyVoice3]][待确认], [[SEED-TTS-Eval]]✓, [[CV3-Eval]][待确认], [[ConditionalFlowMatching]]✓ | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过递归使用模型自身合成输出作为参考音频进行迭代合成,利用差异化退化放大模型间性能差距,从而恢复已有客观指标对 SOTA TTS 系统的区分力和人类对齐性
> - **路线**: (ref_wav, ref_text, target_text) → TTS model → synthesized_wav → 替换 ref_wav → 重复 N 次 → 聚合多轮指标(Mean/LWA/EWA/AUC) → 排名
> - **指标**: UTMOSv2 system-level SRCC 从 0.118 提升至 0.464 (Mean); DNSMOS 从 0.091 提升至 0.255; 10 轮后 SIM 标准差从 3.83 扩大至 12.15, CER 标准差从 0.64 扩大至 17.62 [Table 3, Fig 3]
> - **可借鉴**: 迭代合成作为"模型压力测试"的思路可迁移到任何生成模型评估场景 — 不需要新指标,只需让模型反复处理自身输出,观察退化速率; 5 轮迭代即可达到接近 10 轮的区分效果 [Fig 4]
> - **局限**: 计算成本线性增长(N 倍推理); 迭代评估偏好稳定性而非多样性/表现力; 仅评估开源模型,商业模型未覆盖; naturalness 与 speaker similarity 在低质量参考下存在冲突

## 核心问题

本文要解决什么? 当前零样本 TTS 系统在传统客观指标上发生严重的 **分数饱和 (score saturation)**: 11 个 SOTA 模型的 SIM 标准差仅 1.51%(去除 CosyVoice 后), CER 标准差仅 0.64%, UTMOSv2 标准差仅 0.12, DNSMOS 标准差仅 0.07 [§5.1, Fig 3]。当真实性能差距被压缩到评估模型自身噪声的量级以内时,指标波动不再反映真实排名,导致客观指标与人类判断的相关性极弱 — UTMOSv2 的 system-level SRCC 仅 0.118, DNSMOS 仅 0.091 [Fig 2]。

这不是指标本身有缺陷,而是评估协议的问题: 单轮合成产出的分数区间太窄,无法穿透评估噪声。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

I2D 不引入新的评估模型或指标,而是提出一种 **迭代合成协议**: 给定 TTS 模型 M 和评估样本 (ref_wav, ref_text, target_text),在每一轮迭代中将上一轮的合成结果作为下一轮的参考音频输入,目标文本保持不变。重复 N 轮后,对所有轮次的客观指标值进行聚合 [§3.1, Algorithm 1]。

核心机制: 迭代合成引入渐进的分布偏移 (distributional shift) — 合成语音逐步偏离模型训练数据分布。更强的模型退化更慢,更弱的模型退化更快,从而放大了原本被分数饱和掩盖的性能差异 [论文原文, §3]。

### 关键设计选择

**1. 为什么迭代合成能放大差异?**

作者提出两个互补机制 [论文原文, §3]:
- **参考音频质量退化**: 每轮合成都引入少量失真,累积后参考音频质量持续下降,迫使模型在越来越差的条件下工作
- **隐式错误放大**: 如果模型在第 j 轮产生可懂度错误(幻觉/遗漏),则第 j+1 轮的参考音频与目标文本不匹配,进一步加速退化 [§3.1]

cross-model 实验 [§5.4, Fig 5] 进一步证实: 当 F5-TTS 在第 6 轮获得 CosyVoice3-RL 的高质量参考音频后,性能立即跳升至接近 CosyVoice3-RL 的轨迹,但随后再次退化并偏离。反之,CosyVoice3-RL 获得 F5-TTS 的低质量参考后性能下降,但随迭代逐渐恢复。这说明退化主要由**参考质量渐进恶化**驱动,而非灾难性的分布外崩溃 [论文原文, §5.4]。

**2. 为什么不直接用低质量参考做单轮评估?**

[agent 解读] 单轮低质量参考测试只能暴露模型对噪声的瞬时鲁棒性,而迭代合成捕获的是模型在**持续压力下的稳定性曲线**。AUC/Mean 等聚合方法整合了退化轨迹的形状(是线性下降还是指数崩塌),这提供了比单点测试更丰富的信息。

**3. 指标聚合方法**

论文设计了四种聚合策略 [§3.3]:
- **Mean**: 所有轮次的算术平均 — 最简单,效果最好 [Table 3]
- **LWA (Linearly Weighted Average)**: 后期轮次权重线性增大
- **EWA (Exponentially Weighted Average)**: α=0.9 的指数衰减加权
- **AUC (Area Under Curve)**: 梯形法则计算指标曲线面积

对于 lower-is-better 指标(WER/CER),计算相关性时转换为 higher-is-better(如 1-CER) [§3.3]。

**4. 迭代次数选择**

5 轮即可达到接近 10 轮的 SRCC 水平 [Fig 4],作者推荐在计算资源有限时使用 5 轮 [论文原文, §5.2]。

### 训练策略

不适用 — 本文提出的是评估协议,不涉及模型训练。

## 实验

### 评估设置

- **模型**: 11 个开源零样本 TTS 模型,涵盖 AR (FireRedTTS2, Qwen3-TTS, VoxCPM1.5)、NAR (F5-TTS, MaskGCT)、Hybrid (CosyVoice 系列, GLM-TTS, IndexTTS2) [Table 2]
- **数据集**: Chinese (Seed-TTS-Eval test-zh, 2020 utterances), English (LibriTTS test-clean, 2915 utterances), Emotion (CV3-Eval, 300 samples) [§3.2]
- **客观指标**: WER/CER, SIM, DNSMOS, UTMOSv2, Emotion F1 [Table 1]
- **主观评估**: 100 样本 x iter1+iter10 + GT = 2290 样本, 5-6 评注员/样本, 11752 条标注; 三维度: Content Accuracy, Speaker Consistency, Overall Naturalness [§4.3]

### 核心结果

| 指标 | 单轮 SRCC (baseline) | Mean 聚合 SRCC | 提升 | 出处 |
| --- | --- | --- | --- | --- |
| UTMOSv2 | 0.118 | 0.464 | +0.346 | [Table 3] |
| DNSMOS | 0.091 | 0.255 | +0.164 | [Table 3] |
| SIM | 0.682 | 0.682 (Mean) / 0.727 (EWA) | +0.045 (EWA) | [Table 3] |
| 1-CER | 0.510 | 0.519 (LWA) | +0.009 | [Table 3] |

10 轮后 utterance-level SRCC 全部超过 0.6, system-level SRCC 全部超过 0.8 [Fig 2]。

### 模型排名关键发现

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER Mean (zh) | CosyVoice3-RL: 0.90 | CosyVoice: 11.18 | SEED-TTS-Eval | [Table 4] |
| WER Mean (en) | IndexTTS2: 2.22 | FireRedTTS2: 32.35 | LibriTTS | [Table 4] |
| SIM Mean (zh) | CosyVoice3-RL: 62.17 | CosyVoice: 23.70 | SEED-TTS-Eval | [Table 4] |
| UTMOSv2 Mean (zh) | Qwen3-TTS: 3.68 | CosyVoice: 2.76 | SEED-TTS-Eval | [Table 4] |
| Content Acc (iter1/10) | IndexTTS2: 4.88/4.59 | MaskGCT: 4.78/2.42 | 主观评估 | [Table 4] |
| Naturalness (iter1/10) | IndexTTS2: 4.30/2.56, Qwen3-TTS: 4.27/3.79 | MaskGCT: 3.81/1.31 | 主观评估 | [Table 4] |
| Emotion F1 weighted | IndexTTS2: 64.69 | CosyVoice: 39.57 | CV3-Eval Emotion | [Table 5] |

**模型鲁棒性分层** [§5.3]:
- **最强**: Qwen3-TTS — 10 轮后 Naturalness MOS 3.79(最高), UTMOSv2 4.04/3.68 (en/zh 均最优)
- **强**: CosyVoice3/3-RL, IndexTTS2, CosyVoice2 — 多维度一致性好
- **中等**: GLM-TTS, VoxCPM1.5 — 部分维度退化明显
- **弱**: FireRedTTS2 (生成极短/极长/无关内容), F5-TTS (语速加快+单调), MaskGCT (清晰度下降+不规则韵律), CosyVoice (电噪声+失真) [§5.3]

### 情感克隆的伪高分现象

CosyVoice 在 Sad 类别 F1 最高 (57.29),但这是因为迭代过程中其输出系统性地收敛为 sad-like 情感色调,导致 Sad recall 虚高而其他情感退化严重。这不反映真实的情感建模能力,而是一种退化模式 [论文原文, §5.3]。

## 局限性

1. **计算成本**: 迭代合成使推理成本线性增长 N 倍,10 轮评估意味着 10 倍推理时间 [§6]
2. **偏好稳定性而非多样性**: 迭代评估本质上奖励模型在压力下的稳定性,可能低估善于生成多样化表达但迭代稳定性较差的模型 [论文原文, §6]
3. **Naturalness-SIM 冲突**: 在零样本设置下,参考音频质量差时,自然度与说话人相似度存在张力 [论文原文, §6]
4. **仅覆盖开源模型**: 商业模型(如 Seed-TTS 闭源版、Azure TTS)因成本和访问限制未纳入 [论文原文, §6]
5. [agent 解读] **聚合策略选择**: 尽管 Mean 在 UTMOSv2/DNSMOS 上效果最好,但对 SIM 和 CER 提升有限(分别 +0/+0.009),说明迭代聚合对本身就有一定区分力的指标帮助不大
6. [agent 解读] **未与其他评估改进方法对比**: 论文未与 TTSDS2、SpeechJudge、TTS-PRISM 等近期评估方法进行系统性对比,无法判断 I2D 与这些方法是互补还是冗余

## 点评

**方法论的简洁性是最大亮点。** 不需要训练新模型、不需要收集新数据、不需要设计新指标 — 仅通过改变评估协议(让模型"自我喂食")就能显著提高 predicted MOS 的人类对齐性。UTMOSv2 SRCC 从 0.118 到 0.464 的提升幅度令人印象深刻。

**cross-model 实验是论文最有洞察力的部分。** 通过在第 6 轮交换参考音频,作者优雅地分离了"参考质量退化"和"分布外崩溃"两个因素,得出退化主要由前者驱动的结论。这意味着迭代评估实际上在测量模型的"自修复能力" — 好模型能从低质量参考中恢复,差模型会持续恶化。

**局限也很清晰:** 0.464 的 SRCC 虽然远好于 0.118,但绝对值仍然不高 — 作为人类对齐度指标来说,moderate correlation 级别。与 TTSDS2 (ρ≈0.67) 相比,I2D+UTMOSv2 的对齐度仍有差距。论文也未讨论 I2D 与 TTSDS2 等方法组合使用的可能性。

**对 TTS 评估领域的贡献**: I2D 为现有指标提供了一种低成本的增强手段,尤其适用于 predicted MOS 这类在 SOTA 模型区间严重饱和的指标。它作为 TTSDS2 和 SpeechJudge 等更重量级方法的互补工具有价值。

## 可复用的 idea

1. **"自我喂食"迭代测试作为鲁棒性探针**: 这个思路可迁移到任何生成模型(图像生成、语音转换、文本生成等) — 让模型反复处理自身输出,观察退化速率,作为模型质量的代理指标
2. **cross-model 参考交换实验**: 用于分离"输入质量影响"和"模型内在能力"两个因素,是一种通用的实验设计
3. **简单聚合 > 复杂加权**: Mean 比 LWA/EWA/AUC 效果更好(至少不差),提示在设计指标聚合时不需要过度工程化
4. **5 轮迭代足够**: 实际应用中不需要跑满 10 轮,5 轮即可达到 plateau,对控制评估成本有实际指导意义

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 两个退化机制 + cross-model 验证有完整因果链; 可借鉴 4 项具体 |
> | 可信赖 | pass | 核心数字交叉验证全部通过,出处覆盖 >90% |
> | 可区分 | pass | 4 处 [agent 解读] 明确标注,无断言式推测 |
> | 可定位 | pass | KB 背景给出 5 条评估路线谱系,创新判断有对比基准 |
> | 不污染 | pass | 评估方法论文,无新建概念需求,frontmatter 挂接合理 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/IterateDifferentiate-review.yml`
