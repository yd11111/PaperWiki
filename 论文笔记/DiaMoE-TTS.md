---
type: paper
tier: deep
title: "DiaMoE-TTS: A Unified IPA-Based Dialect TTS Framework with Mixture-of-Experts and Parameter-Efficient Zero-Shot Adaptation"
arxiv_id: "2509.22727"
source: "Sources/DiaMoE-TTS.pdf"
authors: [Ziqi Chen, Chaofan Ding, Gongyu Chen, Zihao Chen, Yihua Wang, Wei-Qiang Zhang]
year: 2025
venue: "arXiv"
tags: [TTS, zero-shot, dialect, multilingual, MoE, IPA, PEFT, LoRA, flow-matching, low-resource]
concepts: ["[[ConditionalFlowMatching]]", "[[Non-autoregressiveTTS]]", "[[PhonemeRepresentation]]", "[[SpeakerAdaptation]]", "[[MelSpectrogram]]"]
models: ["[[CosyVoice2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: DiaMoE-TTS 基于 F5-TTS 架构构建,F5-TTS 是一个使用 [[ConditionalFlowMatching]] (OT-CFM) + DiT backbone 的全 NAR TTS 系统。在知识库中,F5-TTS 已被大量系统作为基座引用(Cross-Lingual F5-TTS、F5R-TTS 等),但尚无独立模型页。本文的核心创新集中在**多方言前端统一**(IPA) + **方言感知路由**(MoE) + **参数高效适应**(LoRA/Conditioning Adapter),而非对 flow matching 生成范式的改进。

**已有认知**:
- [[ConditionalFlowMatching]] (confirmed): OT-CFM 在 TTS 中已成为主流生成范式,F5-TTS 使用 ConvNeXt V2 text encoder + DiT + flow matching。DiaMoE-TTS 保留了这一核心架构,创新集中在前端和适应层。
- [[Zero-shotSpeechSynthesis]] (confirmed): 当前 SOTA 由 CosyVoice 3、Qwen3-TTS 等系统把持,WER < 1.5%。DiaMoE-TTS 的 zero-shot 定义不同于主流——它关注**方言级别**的 zero-shot(用少量数据适应新方言),而非说话人级别。
- [[Cross-lingualVoiceCloning]] (confirmed): 跨语言克隆的核心挑战是音色-语言解耦。DiaMoE-TTS 的 IPA 统一前端与该任务高度相关,但本文聚焦在同一语系(汉语方言)内的变体。
- [[PhonemeRepresentation]] [待确认]: IPA 作为跨语言统一表示已被记录,DiaMoE-TTS 是首个系统性地将 IPA 应用于汉语方言 TTS 并配合 MoE 路由的工作。
- [[SpeakerAdaptation]] [待确认]: LoRA + Conditioning Adapter 属于 parameter-efficient adaptation,与 AdaSpeech 系列中的 CLN tuning 和 residual adapter 方法一脉相承,但 DiaMoE-TTS 特别之处在于适应的对象是**方言**而非说话人。
- [[Non-autoregressiveTTS]] [待确认]: F5-TTS 基座属于全 NAR 架构(flow matching + DiT),不涉及自回归建模。

**创新判断**: 本文的主要贡献不在生成模型层面,而在 TTS 系统的**前端统一**和**高效方言扩展**策略。与 Cross-Lingual F5-TTS 相比,后者通过 MMS forced alignment + speaking rate predictor 解决跨语言问题,而 DiaMoE-TTS 通过 IPA 统一音素空间 + MoE 方言路由解决方言问题,两者路线不同但均基于 F5-TTS。

> 检索命中: [[ConditionalFlowMatching]]✓, [[Zero-shotSpeechSynthesis]]✓, [[Cross-lingualVoiceCloning]]✓ | 参考: [[PhonemeRepresentation]](pending-review), [[SpeakerAdaptation]](pending-review), [[Non-autoregressiveTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 F5-TTS,通过 IPA 统一方言前端 + MoE 方言路由 + LoRA/Adapter PEFT,用 ~1.1k 小时数据实现多方言零样本 TTS
> - **路线**: 方言文本 → IPA G2P → Unified IPA Phonemes → ConvNeXt V2 Text Encoder → Dialect MoE → DiT (Flow Matching) → Mel Spectrogram → Vocoder
> - **指标**: 在 6 个方言上报告 WER/MOS/UTMOSv2,整体落后商业系统(Edge TTS/CosyVoice2/Qwen-TTS);消融证明 IPA+MoE 优于 pinyin w/o MoE (CD MOS 2.22 vs 1.23) [Table 2, Table 3]
> - **可借鉴**: (1) IPA 统一前端消除方言 G2P 歧义; (2) MoE 加方言分类辅助 loss 防止风格平均化; (3) 四阶段渐进训练范式(pretrain → transfer → MoE → PEFT 新方言); (4) 简单 pitch/time-scale 增强实现低资源适应
> - **局限**: WER 仍高(YUE 76.6%, NAN 92.4%);MOS 普遍 < 3.5;与商业系统差距明显;评估缺少 speaker similarity 指标;Peking Opera 等特殊领域 WER > 39%

## 核心问题

DiaMoE-TTS 要解决的问题: 如何用统一框架覆盖汉语多方言 TTS,并能以极少数据快速扩展到新方言?

现有挑战 [§1]:
1. **数据稀缺**: 大多数方言公开数据面向 ASR 而非 TTS,质量不足
2. **正字法不一致**: 不同方言的文字/拼音系统不统一(普通话用拼音,粤语用粤拼,闽南语用台罗等)
3. **前端建模困难**: 相同汉字在不同方言中发音差异巨大

论文声称这是首个基于开放数据的端到端零样本方言 TTS pipeline [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

基于 F5-TTS (OT-CFM + DiT) 的四阶段渐进训练框架 [§3.1, Fig 2]:

```
Stage 0: F5-TTS 预训练权重初始化 (Mandarin+English)
    ↓
Stage 1: 在统一 IPA 数据上联合训练 (普通话 + 多方言)
    ↓
Stage 2: 加入 Dialect MoE 模块, 继续联合训练
    ↓
Stage 3: 冻结主干, 仅训练 LoRA + Conditioning Adapter 适应新方言
```

**为什么分四阶段?** [论文原文] 渐进策略平衡了模型容量、数据效率和泛化能力——先从高资源语言获取强先验,再逐步引入方言知识,最后用 PEFT 避免灾难性遗忘 [§3.1]。[agent 解读] 这本质上是一种课程学习策略,避免低资源方言数据在早期训练中被高资源数据淹没。

### 关键设计选择

#### 1. IPA 统一前端

**做了什么**: 用 IPA (国际音标) 替代各方言的原始拼音/字符输入,构建统一的 phoneme space [§2.1, Fig 1]。

**为什么这样做?** [论文原文] 不同方言的拼音系统各异(普通话 pinyin、粤语 jyutping、闽南语 POJ/台罗),但 IPA 能一致地捕捉语音变体。例如"马"在普通话中为 [m, aᴹᴸ],粤语中为 [m, aᴸᴹ],通过 IPA 在同一符号空间中表示 [§2.1]。

[agent 解读] IPA 的真正价值在于:不同方言共享大量声母(如 /m/, /k/, /t͜ɕ/),IPA 让模型能跨方言复用这些共享的音素表示,为 transfer learning 提供天然基础。

**消融证据**: IPA vs Pinyin (both with MoE): CD 方言 WER 49.01% vs 93.29%, MOS 2.22 vs 1.23 [Table 3]。pinyin 输入几乎完全失效,因为模型无法处理不同方言拼音系统的冲突。

#### 2. Dialect-style MoE

**做了什么**: 在 text embedding 之后加入残差 MoE 模块 [§3.2]:

$$h' = \text{MoE}(h) + h$$

其中 $h = E_{\text{text}}(p_{\text{IPA}})$ 是文本编码器输出。

**为什么需要 MoE?** [论文原文] 多方言联合训练会导致风格同质化(style averaging),不同方言的韵律/声调特征被抹平。MoE 的多个 expert 各自专注于特定方言的 phonological characteristics,通过 learnable gating 动态路由 [§3.2]。

**辅助 loss**: 引入方言分类 loss 作为 gate 的监督信号 [§3.2]:

$$L_{\text{total}} = L_{\text{OT-CFM}} + \lambda \cdot L_{\text{dialect}}$$

$\lambda = 0.1$ (Stage 2), $\lambda = 0$ (其他阶段) [§3.2, Eq.4]。

[agent 解读] 方言分类 loss 是关键设计——如果没有它,MoE gate 只能通过重建 loss 间接学习路由,很可能退化为所有方言共享同一组 expert。显式分类监督强制 gate 感知方言身份。

**消融证据**: w/ MoE vs w/o MoE (both IPA): XA 方言 MOS 3.15 vs 2.33, WER 33.00% vs 41.09% [Table 3]。

#### 3. 低资源方言适应 (Stage 3)

**做了什么**: 冻结 Stage 2 训练好的主干(包括 MoE),仅训练 LoRA (rank=16, α=1) 作用于 attention Q/V 投影 + Conditioning Adapter [§3.3]。

**为什么冻结 MoE?** [论文原文] 保留 MoE 已学到的路由行为,避免在少量新方言数据上过拟合 [§3.3]。

**数据增强**: 简单的 pitch/time-scale 变换,系数 {0.85, 0.9, 0.95, 1.05, 1.1, 1.15},不改变方言风格 [§3.3]。

[agent 解读] 这种极简增强策略在 3 小时数据上是合理的——更复杂的增强(如 SpecAugment)可能破坏方言特有的韵律模式。但问题是 6 个变换因子可能不够,且论文未报告增强的消融实验。

### 训练策略

| 阶段 | 训练数据 | 可训参数 | 学习率 | 步数 |
| --- | --- | --- | --- | --- |
| Stage 0 | F5-TTS 预训练 (Mandarin+EN) | - | - | - |
| Stage 1-2 | 普通话 + 多方言 IPA (~1.1k h) | 全模型 (+MoE in S2) | 7.5e-5 (warmup 2k) | 200K [§4.2] |
| Stage 3 | 新方言 (~3h) | LoRA + Adapter | 1e-5 | 100K [§4.2] |

数据规模对比 [Table 1]: Edge TTS ~150k h, CosyVoice2 ~3000k h, Qwen-TTS ~700+400 h; DiaMoE-TTS 仅 ~1.1k h。

## 实验

| 指标 | 本文 (代表值) | Baseline Avg | 数据集/方言 | 出处 |
| --- | --- | --- | --- | --- |
| WER (CD) | 37.12% | 7.43% | Chengdu dialect | [Table 2] |
| WER (TJ) | 20.18% | 6.23% | Tianjin dialect | [Table 2] |
| WER (YUE) | 76.59% | 31.98% | Cantonese | [Table 2] |
| WER (NAN) | 92.41% | - | Southern Min | [Table 2] |
| MOS (CD) | 2.22 | 4.05 | Chengdu dialect | [Table 2] |
| MOS (TJ) | 1.66 | 2.64 | Tianjin dialect | [Table 2] |
| UTMOSv2 (CD) | 3.33 | 3.11 | Chengdu dialect | [Table 2] |
| UTMOSv2 (ZZ) | 2.86 | 3.01 | Zhengzhou dialect | [Table 2] |
| MOS w/o MoE (CD) | 2.46 | - | Chengdu ablation | [Table 3] |
| MOS w/o IPA (CD) | 1.23 | - | Chengdu ablation (pinyin) | [Table 3] |
| MOS Ours (CD) | 2.22 | - | Chengdu full model | [Table 3] |

**核心发现**:

1. **与商业系统差距显著**: WER 普遍高出 20-50 个百分点,MOS 低 1-2 分。论文承认这一差距并解释原因: 数据规模差异(1k vs 150k+小时)以及方言覆盖带来的建模复杂度增加 [§4.4.1]。

2. **消融验证了核心设计**: IPA + MoE 的组合显著优于任一单独使用。pinyin 输入近乎完全失效 (WER > 90%) [Table 3]。

3. **零样本方言适应可行**: 用 ~3h 数据 + PEFT 成功扩展到南京话和京剧(京白/韵白),但 WER 仍然很高(NJ 31.64%, Jingbai 39.95%, Yunbai 68.94%) [Table 2]。

4. **UTMOSv2 指标相对竞争**: 在部分方言上 UTMOSv2 与商业系统接近甚至略优(CD 3.33 vs avg 3.11),暗示生成语音的声学质量尚可,但可懂度(WER)是主要瓶颈 [Table 2]。

## 局限性

1. **绝对性能不足**: WER 在多数方言上 > 30%,部分超过 70%,远未达到实用标准。论文虽然强调"不追求超越商业系统",但差距过大,即使考虑数据量差异也说明方法本身仍有明显不足 [Table 2]。

2. **评估不完整**: 
   - 缺少 speaker similarity 指标,无法验证 zero-shot voice cloning 能力
   - 缺少推理速度/RTF 报告
   - 缺少 MoE 增强的消融(如增强 vs 无增强)
   - Baseline 覆盖不全(NAN/SJZ/NJ/Jingbai/Yunbai 无 baseline 对比)

3. **数据问题未充分讨论**: ASR 数据用于 TTS 的适配细节不够——噪声水平、分句策略、数据清洗流程均未详述,但这些对 TTS 质量影响巨大。

4. **MoE 设计缺乏分析**: expert 数量的选择依据、gate 的激活分布、不同 expert 是否真的学到了方言特异性特征——这些关键问题均无分析。

5. **京剧场景存疑**: 京剧的京白/韵白与普通方言差异极大(韵白近似文言发音),将其归为"方言适应"可能过度简化。

## 点评

DiaMoE-TTS 提出了一个方向正确但执行粗糙的方言 TTS 框架。IPA 统一前端 + MoE 方言路由 + PEFT 适应的三层设计在架构上是合理的:IPA 提供共享表示基础,MoE 保留方言差异,PEFT 降低扩展成本。消融实验也清楚地验证了 IPA 和 MoE 各自的贡献。

然而,实际性能与实用性之间存在巨大鸿沟。WER 普遍 > 30% 意味着合成语音有大量发音错误,即使在数据量差异的解释下,这也暗示 IPA G2P 的准确性、训练数据质量、或模型容量仍有根本性瓶颈。与 Cross-Lingual F5-TTS 对比——同样基于 F5-TTS,后者在跨语言场景下实现了 WER 2.5%(test-en),差距一个数量级。

值得注意的是本文的开源承诺(数据集构建方法 + 多方言数据 + 训练脚本),这对方言 TTS 社区有工程价值。但作为研究贡献,创新深度有限——IPA 前端、MoE routing、LoRA 适应都是已有技术的组合,缺少针对方言 TTS 独特挑战的深度思考(如声调系统差异建模、方言间语法影响的韵律差异等)。

## 可复用的 idea

1. **IPA 统一音素空间**: 对于任何需要处理多方言/多语言的 TTS 系统,IPA 前端可以直接替换 language-specific G2P,减少前端维护成本。但需要高质量的 IPA 词典——这才是真正的瓶颈。

2. **MoE + 方言分类辅助 loss**: 防止多方言/多风格联合训练的风格平均化问题。MoE 的 gate 用分类 loss 显式监督是关键,否则 expert 不会自动按方言分化。此策略可迁移到情感 TTS、多风格 TTS 等场景。

3. **四阶段渐进训练**: pretrain → transfer → MoE specialization → PEFT for new variety。这种课程学习范式适用于所有低资源语言/方言扩展场景。

4. **简单增强即可工作**: pitch/time-scale 变换保持方言风格不变,成本极低。但论文未报告增强的独立消融效果。

> [!review] 审阅
> 见 `_review/DiaMoE-TTS-review.yml`
