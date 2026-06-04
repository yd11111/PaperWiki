---
type: paper
tier: deep
title: "TMD-TTS: A Unified Tibetan Multi-Dialect Text-to-Speech Framework"
arxiv_id: "2509.18060"
source: "Sources/TMD-TTS.pdf"
authors: [Yutong Liu, Ziyue Zhang, Ban Ma-bao, Renzeng Duojie, Yuqing Cai, Yongbin Yu, Xiangxiang Wang, Fan Gao, Cheng Huang, Nyima Tashi]
year: 2026
venue: "ICASSP 2026 (submitted)"
tags: [TTS, multi-dialect, low-resource, Tibetan, dynamic-routing, flow-matching, dataset-generation]
concepts: ["[[Conditional Flow Matching]]", "[[Mel Spectrogram]]", "[[Duration Predictor]]", "[[Neural Vocoder]]", "[[Speaker Embedding]]", "[[Non-autoregressive TTS]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: TMD-TTS 构建于 Matcha-TTS 之上,而 Matcha-TTS 本身使用 [[Conditional Flow Matching]] (CFM/OT-CFM) 作为生成骨架。在 TTS 知识库中,CFM 是一类以 ODE 路径代替 SDE 的高效生成范式,已被 CosyVoice、F5-TTS、MaskGCT 等大量工作采用。TMD-TTS 的特殊之处在于它并非追求零样本语音克隆或大规模预训练,而是解决低资源多方言 TTS 的方言一致性问题。
>
> **已有认知**:
> - [[Speaker Embedding]]✓ 提供了从说话人身份到嵌入向量的标准范式(lookup table / speaker encoder);TMD-TTS 的 dialect embedding 在形式上等价于 speaker embedding 的特例,不同之处在于它编码的是方言身份而非个体身份。
> - [[Neural Vocoder]]✓ 记录了 HiFi-GAN → BigVGAN 的演进;TMD-TTS 使用 BigVGAN 作为外部 vocoder,属于典型的 semi-end-to-end 两阶段架构。
> - [[Duration Predictor]][待确认] 描述了 NAR TTS 中从音素序列预测帧时长的标准组件;TMD-TTS 复用 Matcha-TTS 的 duration predictor。
> - [[Mel Spectrogram]][待确认] 是 TMD-TTS 的中间表示。
> - [[VITS]][待确认] 是本文实验中的 baseline 之一(以 VITS2 形式出现)。
>
> **创新判断**: KB 中已有的 [[Speaker Embedding]] 和 [[Conditional Flow Matching]] 都关注高资源/通用场景,对低资源多方言场景的适配方案(dialect-specific dynamic routing)尚无记录。TMD-TTS 的 DSDR-Net(public+private FFN)在形式上类似 Mixture-of-Experts 的条件计算思路,但 KB 中尚无 MoE/条件计算的专门概念页。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Neural Vocoder]]✓, [[Speaker Embedding]]✓ | 过滤: [[Duration Predictor]](pending-review), [[Mel Spectrogram]](pending-review), [[VITS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 Matcha-TTS 基础上增加 dialect fusion + DSDR-Net(public/private FFN 动态路由),实现三方言统一藏语 TTS,并构建了 102h 合成方言数据集 TMDD
> - **路线**: 藏文字符 → Tokenizer → Text Encoder + Dialect Fusion(dialect embedding 线性投影加到 text hidden) → Duration Predictor → Flow Prediction Network (Transformer w/ DSDR-Net) → Mel Spectrogram → BigVGAN Vocoder → Waveform
> - **指标**: Ü-Tsang/Amdo/Kham 三方言: PESQ 3.03/3.13/3.05, STOI 94.52/94.92/93.17%, DECS 88.09/79.17/67.65%, nMOS 3.83/3.84/3.86 [Table 1]; 消融: 去掉 DSDR-Net DCA 从 80.25% 降至 60.12% [Table 2]
> - **可借鉴**: (1) "public FFN + private FFN" 的条件计算路由设计,可迁移到任何需要风格/方言/情感条件控制的 Transformer TTS; (2) 合成数据质量筛选 pipeline(DECS>0.8 + PESQ<3 的用 MetricGAN+ 增强 + 人工抽查)
> - **局限**: nMOS 最高 3.86(中等水平); RTF 0.031 略慢于 VITS2; 仅在藏语验证,未证明推广到其他低资源多方言场景; 未开源代码(截至论文发表)

## 核心问题

TMD-TTS 要解决的核心问题: **藏语三大方言(Ü-Tsang/Amdo/Kham)之间语音、词汇、语法差异大,缺乏大规模平行语音数据,现有多方言 TTS 方案(如 Xu et al. 2021)依赖多个 vocoder 且方言区分度不够** [§1]。

具体而言,先前工作 [17] 生成共享 mel-spectrogram 后用不同 WaveNet vocoder 分别转换各方言,这种 "后期方言化" 策略无法在声学建模阶段捕获细粒度的方言差异(节奏、语调) [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TMD-TTS 基于 Matcha-TTS [15] 构建,属于 semi-end-to-end 架构: 先生成 mel-spectrogram,再用预训练 BigVGAN vocoder 转为波形 [§2] [Fig 1(a)]。

完整 pipeline:
1. **Tokenization**: 藏文字符 → long-format tensor(216 字符词表) [§3.1]
2. **Text Encoder**: 将 token 序列编码为隐层特征 $h_{\text{text}}$
3. **Dialect Fusion Module**: 将 dialect ID $d_{id} \in \{0,1,2\}$ 映射为 embedding,归一化后线性投影,加到 text hidden features [§2.1, Eq.1-2]
4. **Duration Predictor**: 预测音素时长,指导上采样
5. **Flow Prediction Network**: 基于 CFM 的 Transformer decoder,其中 FFN 被 DSDR-Net 替换,融合方言信息生成 mel-spectrogram [§2.1]
6. **Vocoder**: 预训练 BigVGAN 将 mel 转波形

### 关键设计选择

#### 1. Dialect Fusion Module

**做了什么**: 将方言 ID 映射为可学习 embedding $h_{d_{id}}$,经 LayerNorm 归一化后,通过线性层投影,加到 text encoder 的隐层特征上 [§2.1, Eq.1-2]:

$$\hat{h}_{\text{text}} = h_{\text{text}} + \text{Linear}(h_{d_{id}})$$

**为什么这样设计**: [论文原文] 先前工作 [17] 在 mel-spectrogram 之后才引入方言信息("late-stage incorporation"),导致模型无法在声学建模阶段捕获方言差异 [§1]。dialect fusion 将方言信息前置到 text encoder 和 flow prediction network 两个阶段,使整个生成过程都是 "dialect-aware" 的 [§2]。

[agent 解读] 这种 "embedding + linear projection + residual addition" 的模式与 multi-speaker TTS 中的 speaker embedding 注入完全同构,区别在于编码的是 3 个方言类别而非说话人身份。128 维的 dialect embedding 对于 3 个类别来说维度偏高,可能是为了给 DSDR-Net 的路由提供更丰富的特征空间。

**消融证据**: 去掉 dialect fusion (用零向量替代) → DCA 从 80.25% 降至 74.15%, DECS 从 78.3% 降至 72.8% [Table 2]。

#### 2. DSDR-Net (Dialect-Specialized Dynamic Routing Network)

**做了什么**: 在 Transformer 的每个 block 中,用 "public FFN + private FFN" 替代标准 FFN [§2.1, Fig 1(b)]:

- **Multi-head self-attention** 先处理 $h_{\text{text}}$ 得到 $h_{\text{attn}}$ [Eq.3]
- **Public FFN**: 所有方言共享的前馈网络 → $\text{FFN}_{\text{public}}(h_{\text{attn}})$
- **Private FFN**: 3 个方言各有独立的 FFN $\{\text{FFN}_0, \text{FFN}_1, \text{FFN}_2\}$,根据 dialect ID 路由选择对应的一个 → $\text{FFN}_{\text{private},d}(h_{\text{attn}})$ [Eq.4-5]
- **合并**: 两路输出相加 → $\hat{h}_{\text{attn}} = \text{FFN}_{\text{public}}(h_{\text{attn}}) + \text{FFN}_{\text{private},d}(h_{\text{attn}})$ [Eq.6]

**为什么这样设计**: [论文原文] 标准 FFN 使用共享参数处理所有方言,无法学习方言特有的声学模式(节奏、语调差异);DSDR-Net 的 "conditional computation mechanism" 通过动态路由使模型学习不同方言的独立声学模式,提供 "finer-grained dialect modeling" [§2.1]。

[agent 解读] 这本质上是一种 hard routing 的 Mixture-of-Experts: 每次只激活 1 个 private expert(由 dialect ID 确定),加上一个始终激活的 shared expert。与 soft MoE (如 Switch Transformer 的 gating) 不同,DSDR-Net 的路由是确定性的(根据 dialect ID 直接索引),不需要学习 router。这种设计的优点是简单可靠(不存在 load balancing 问题),代价是方言数量增加时参数线性增长。对于 3 方言的场景,额外参数开销可接受。

**消融证据**: 去掉 DSDR-Net (用标准 FFN) → DCA 从 80.25% 降至 60.12%, DECS 从 78.3% 降至 58.6%;同时去掉两个模块 → DCA 33.42%, DECS 32.2% [Table 2]。DSDR-Net 的贡献(DCA 降 20 点)大于 dialect fusion(降 6 点),说明参数分离比信息注入更关键。

#### 3. 数据生成 Pipeline

**做了什么**: 用训练好的 TMD-TTS 批量合成方言语音,并通过三层质量筛选构建 TMDD 数据集 [§2.2, Fig 1(c)]:

1. **合成**: 从文本库顺序采样,每条文本合成 3 个方言的语音
2. **方言一致性筛选**: DECS > 0.8 才保留(用预训练方言分类器算 embedding cosine similarity)
3. **感知质量筛选**: PESQ < 3 或 DNSMOS < 2.7 的样本用 MetricGAN+ [18] 增强
4. **人工抽查**: 母语者最终审核

**为什么这样设计**: [论文原文] 藏语多方言平行语料极度匮乏(现有基线仅 ~800 样本),需要 TTS 合成来扩充数据 [§1]。质量筛选保证合成数据可用于下游任务(如 S2SDC) [§2.2]。

### 训练策略

- 训练 500k steps,Adam 优化器 [§3.1]
- BigVGAN vocoder 单独训练,AdamW + 指数衰减 [§3.1]
- 2x RTX 4090 GPU [§3.1]
- 192 维 DSDR-Net, 128 维 dialect embedding [§3.1]
- 训练数据: 179h 多方言藏语,包括 Ü-Tsang 44h, Kham 45h, Amdo 90h, 来自 1500+ 说话人,每方言 40k 训练样本 [§3.1]

## 实验

| 指标 | TMD-TTS | Matcha-TTS | VITS2 | SC-CNN | 数据集/方言 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| STOI(%) | 94.52 | 93.84 | 85.72 | 80.40 | Ü-Tsang | [Table 1] |
| PESQ | 3.03 | 2.43 | 2.00 | 1.62 | Ü-Tsang | [Table 1] |
| SI-SDR(dB) | 17.91 | 12.32 | 9.88 | 7.24 | Ü-Tsang | [Table 1] |
| DECS(%) | 88.09 | 65.20 | 41.91 | 65.04 | Ü-Tsang | [Table 1] |
| DCA(%) | 67.41 | 65.80 | 39.26 | 40.37 | Ü-Tsang | [Table 1] |
| nMOS | 3.83 | 3.73 | 3.18 | 2.83 | Ü-Tsang | [Table 1] |
| dMOC(%) | 76.64 | 73.33 | 69.15 | 65.14 | Ü-Tsang | [Table 1] |
| STOI(%) | 94.92 | 94.54 | 89.13 | 79.90 | Amdo | [Table 1] |
| PESQ | 3.13 | 2.34 | 1.98 | 1.65 | Amdo | [Table 1] |
| DECS(%) | 79.17 | 65.32 | 41.91 | 65.04 | Amdo | [Table 1] |
| DCA(%) | 87.78 | 75.42 | 39.26 | 59.63 | Amdo | [Table 1] |
| nMOS | 3.84 | 3.75 | 3.20 | 2.82 | Amdo | [Table 1] |
| STOI(%) | 93.17 | 91.47 | 82.25 | 76.09 | Kham | [Table 1] |
| PESQ | 3.05 | 2.32 | 1.87 | 1.47 | Kham | [Table 1] |
| DECS(%) | 67.65 | 63.48 | 46.01 | 19.16 | Kham | [Table 1] |
| nMOS | 3.86 | 3.73 | 3.18 | 2.67 | Kham | [Table 1] |
| RTF | 0.031-0.032 | 0.022-0.023 | 0.020-0.021 | 0.034-0.036 | 全部方言 | [Table 1] |

**消融实验** [Table 2]:

| 配置 | DCA(%) | DECS(%) |
| --- | --- | --- |
| TMD-TTS (full) | 80.25 | 78.3 |
| w/o DSDR-Net | 60.12 | 58.6 |
| w/o Dialect Fusion | 74.15 | 72.8 |
| w/o both | 33.42 | 32.2 |

**TMDD 数据集** [Table 3]: 从 122,700 条合成语音中筛选 32,714 条/方言(共 98,142 条, 102h),相比基线数据集(4,763 条, ~9.5h)扩大 20 倍样本量、11 倍时长。

**下游验证 (S2SDC)** [Table 4]: 用 TMDD 训练 DurFlex-EVC + BigVGAN 22K,MOS 达 3.63,优于基线数据集的 3.54。

## 局限性

1. **绝对合成质量中等**: nMOS 3.83-3.86 在 TTS 领域属于中等水平,距离高资源语言的 4.0+ 有差距,可能受限于藏语训练数据质量和规模 [agent 解读]
2. **推理速度**: RTF 0.031-0.032 慢于 VITS2 的 0.020-0.021,因为 DSDR-Net 引入了额外计算 [Table 1]
3. **方言数固定**: DSDR-Net 的 private FFN 数量 = 方言数,增加新方言需要增加参数并重新训练,不具备零样本方言泛化能力 [agent 解读]
4. **仅藏语验证**: 未在其他低资源多方言语言(如阿拉伯语方言、中国方言)上验证,泛化性未知 [agent 解读]
5. **未开源**: 代码和模型未公开(截至论文发表) [agent 解读]
6. **方言分类器评估的循环性**: DECS 和 DCA 依赖预训练方言分类器,但该分类器也是在有限藏语数据上训练的,评估可靠性受限 [agent 解读]
7. **下游 S2SDC 验证规模有限**: 仅用 DurFlex-EVC 一个模型验证合成数据效果,MOS 提升幅度不大(3.54→3.63) [Table 4]

## 点评

**优点**:
- 问题定义清晰: 低资源多方言 TTS + 合成数据生成,对藏语 NLP 社区有实际价值
- DSDR-Net 的 public/private FFN 设计简洁有效,消融实验充分证明了两个模块的互补贡献 [Table 2]
- 构建了完整的合成→筛选→验证 pipeline,而非仅提出模型

**不足**:
- 实验对比缺乏更强的 baseline: 未与 F5-TTS、CosyVoice 等更新的 TTS 系统对比(虽然这些模型可能不直接支持藏语)
- DSDR-Net 的 hard routing 设计虽然简单,但对方言边界模糊的情况(如双方言使用者)缺乏建模能力
- 论文未讨论不同方言间训练数据量不均(Amdo 90h vs Ü-Tsang 44h vs Kham 45h)对模型性能的影响

## 可复用的 idea

1. **Public + Private FFN 条件路由**: 对于任何需要按离散条件(方言/情感/风格)分化建模的 Transformer TTS,可以用 DSDR-Net 替换标准 FFN。优点: 实现简单、无 load balancing 问题、确定性路由避免训练不稳定。适用于条件数量较少(< 10)的场景。
2. **合成数据多级筛选 pipeline**: DECS > 阈值 → PESQ/DNSMOS 检查 → MetricGAN+ 增强 → 人工抽查,这套流程可迁移到任何用 TTS 合成数据增强下游任务的场景。
3. **Dialect embedding 注入方式**: 将条件 embedding 通过 Linear + residual addition 注入到 text encoder 和 acoustic decoder 两阶段,比仅在 decoder 侧注入更彻底。
