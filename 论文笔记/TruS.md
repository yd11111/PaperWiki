---
type: paper
tier: deep
title: "Erasing Your Voice Before It's Heard: Training-Free Speaker Unlearning for Zero-Shot Text-to-Speech"
arxiv_id: "2601.20481"
source: "Sources/TruS.pdf"
authors: [Myungjin Lee, Eunji Shin, Jiyoung Lee]
year: 2026
venue: "arXiv"
tags: [machine-unlearning, voice-privacy, zero-shot-TTS, speaker-identity, activation-steering, training-free, flow-matching, DiT, safety]
concepts: ["[[ConditionalFlowMatching]]", "[[SpeakerEmbedding]]", "[[SpeakerVerification]]", "[[Anti-spoofingandDeepfakeDetection]]", "[[EmotionControlinTTS]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 3 个待确认实体页: [[ConditionalFlowMatching]], [[SpeakerEmbedding]], [[Zero-shotSpeechSynthesis]], [[SpeakerVerification]], [[Anti-spoofingandDeepfakeDetection]], [[Emilia]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 [[Anti-spoofingandDeepfakeDetection]] 中"模型级 speaker 保护"方向,提出第五条防线 -- 推理时激活干预:
> | 路线 | 代表 | 操作端 | 时机 |
> |------|------|--------|------|
> | 被动检测 | ASVspoof 系列 | 输出端 | 事后 |
> | 主动扰动 | SafeSpeech | 数据端 | 事前 |
> | 水印溯源 | TraceableSpeech | 输出端 | 事后 |
> | Machine Unlearning (重训练) | [[论文笔记/SpeakerIdentityUnlearning|TGU/SGU (Kim et al., ICML 2025)]] | 模型端 | 事前 |
> | **推理时 Steering (免训练)** | **TruS (本文)** | **模型端** | **事前** |
>
> 本文是对 [[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning (Kim et al., ICML 2025)]] 的直接改进。Kim et al. 提出了 TGU/SGU 两种基于重训练的 unlearning 方案,但存在三个核心限制: (1) 高训练开销(TGU 430 GPU-hours, SGU 48 hours); (2) 每次新 opt-out 请求需重训; (3) 仅支持训练集中出现过的 seen speakers。TruS 同时解决了这三个问题。
>
> **已有认知**: [[ConditionalFlowMatching]] 页(confirmed)记录了 CFM 的 ODE 路径与向量场回归机制。TruS 基于 F5-TTS(一个 DiT-based CFM 模型),在 ODE 求解的每个 flow step 中对中间激活值施加 steering vector。[[SpeakerEmbedding]] 页(confirmed)记录了 speaker identity 在 TTS 模型中的编码方式。TruS 的核心发现是 DiT FFN 层的激活值包含强 speaker identity 信号,可以通过构造 ID-prototype 并减去投影来消除。[[Zero-shotSpeechSynthesis]] 页(confirmed)记录了零样本 TTS 通过 in-context learning 从 audio prompt 泛化到未见说话人的能力,这正是 unlearning 的核心挑战 -- 即使删除训练数据,模型仍可零样本克隆。
>
> **与 EmoSteer-TTS 的关系**: TruS 与 [[论文笔记/EmoSteer-TTS|EmoSteer-TTS (Xie et al., 2025)]] 同属"推理时激活操控"范式,但目标不同 -- EmoSteer-TTS 操控情感属性,TruS 操控说话人身份。TruS 论文直接引用 EmoSteer-TTS [19] 并指出其使用固定 top-k channel 选择缺乏动态适应性,而 TruS 的动态层选择机制是改进点。
>
> **创新判断**: 相对于 KB 中记录的 TGU/SGU,TruS 的突破在于: (1) 完全免训练(0 GPU-hours vs TGU 的 430 hours); (2) 首次实现对 unseen opt-out speakers 的保护; (3) 即时且可序列化的 unlearning 请求处理。代价是 seen opt-out 上的身份抑制略弱于 SGU(SIM-SO 0.477 vs 0.106),但 SGU 同时破坏了 retain 性能(SIM-R 0.290 vs TruS 的 0.678)。

## 速查

> [!summary] 速查
> - **一句话**: 首个 training-free 推理时 speaker unlearning 框架 -- 通过构造 ID-prototype 并在 DiT FFN 激活值上动态施加 steering vector,使 F5-TTS 在推理时即时阻止特定说话人声音的合成,同时保持 retain speakers 的质量不变
> - **路线**: N=30 个 retain speaker 语音 → F5-TTS DiT 各层 FFN 激活 → 平均得 ID-prototype P_Ret → opt-out speaker 激活 X_Opt 与 P_Ret 做差 → L2 归一化得 steering vector S → 逐层逐步 cosine similarity 动态选择干预点(τ = µ + kσ) → 投影减法消除身份分量 → ODE solver + vocoder → unlearned 语音
> - **指标**: SIM-SO 0.477 (F5-TTS 原始 0.657, 降 27%) / WER-SO 3.25 (vs TGU 4.03, SGU 3.70) / SIM-UO 0.488 (F5-TTS 0.668, 降 27%) / SIM-Emo 0.723 (vs F5-TTS 0.732, 仅降 1.2%) / 0 training hours (vs TGU 430h, SGU 48h) [Table 1-3]
> - **可借鉴**: (1) ID-prototype + 投影减法的身份消除策略可迁移到 voice conversion 的 speaker disentanglement; (2) 动态层+步选择机制(µ+kσ 阈值)比 EmoSteer-TTS 的固定 top-k 更灵活,可用于其他 activation steering 任务; (3) 仅需单条 opt-out 语音的 one-shot 设计极大降低了部署门槛
> - **局限**: 仅在 F5-TTS 上验证,未测试 AR/LLM-based TTS; 身份抑制不如重训练的 SGU 彻底(SIM-SO 0.477 vs 0.106); 未讨论恢复攻击/对抗攻击的鲁棒性; 未讨论相似声音的误伤; 代码已公布但未有大规模部署验证

## 核心问题

零样本 TTS 可从短参考语音克隆任意说话人声音,引发严重的隐私与伦理风险。现有防护方案存在根本局限 [§1]:

1. **水印/检测是事后的**: 只能追溯而无法阻止实时滥用 [论文原文]
2. **语音匿名化是替代性的**: 将身份 A 替换为身份 B,但目标是完全禁止特定身份的生成 [论文原文]
3. **现有 unlearning 需要重训练**: TGU [18] 每次新 opt-out 请求需 430 GPU-hours 重新训练,SGU [18] 需 48 hours [论文原文]
4. **仅限 seen speakers**: TGU/SGU 仅适用于训练集中出现过的说话人,而现实中 opt-out 请求最可能来自训练集外的用户 [论文原文]

核心目标: 在推理时即时阻止特定说话人声音的合成,无需重训练,且同时适用于 seen 和 unseen speakers。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TruS 是一个 plug-and-play 的推理时框架,不修改底层 TTS 模型权重。基于 F5-TTS [5](DiT-based flow matching model,预训练于 [[Emilia]] 数据集),但论文声称可推广到其他 DiT-based TTS 架构 [§2.1]。

核心机制分三步 [Fig. 2]:
1. **预计算 ID-prototype**(离线,一次性): 用 N=30 个 retain speakers 的语音,在每个 DiT block 的每个 flow step 提取 FFN 输出并平均
2. **计算 steering vector**(在线): 对每个 opt-out speaker,计算其激活与 ID-prototype 的差异方向
3. **动态干预**(在线): 根据 cosine similarity 自动选择干预点,施加投影减法

**关键观察**: DiT 模型的 FFN 层输出包含强 timbre 和 identity 信号 [论文原文][§2.2],因为 FFN 在非线性通道混合后集中了身份特异性特征 [引用 [27]]。[agent 解读] 选择 FFN 输出而非 attention 输出,符合 transformer interpretability 中"FFN 存储属性知识、attention 做序列路由"的共识。EmoSteer-TTS [19] 在情感维度上也验证了 FFN 输出承载了 paralinguistic 信息。

### 关键设计选择

#### 1. ID-Prototype 构建

从 N 个 retain speakers 的语音中提取 DiT 各层 FFN 激活值,逐层、逐 flow step 取平均 [Eq. 1]:

$$P_{\text{Ret}}^{(\ell,t)} = \frac{1}{N} \sum_{n=1}^{N} X_{\text{Ret}(n)}^{(\ell,t)}$$

[agent 解读] 使用内部激活而非外部 speaker embedding(如 ECAPA-TDNN)有两个优势: (1) 在 F5-TTS 自身的表示空间中操作,确保 steering 方向与模型生成机制对齐; (2) 捕获了逐层、逐步的动态身份表示,比单一全局 embedding 更精细。N=30 经验最优 [Table 5]: N=10 抑制不足(SIM-SO=0.535),N=50 在 seen data 上质量下降(WER-SO=3.71)。

#### 2. Identity-specific Steering Vector

给定 opt-out speaker 的激活 X_Opt,steering vector 定义为其与 ID-prototype 之差的 L2 归一化方向 [Eq. 2]:

$$S^{(\ell,t)} = \frac{X_{\text{Opt}}^{(\ell,t)} - P_{\text{Ret}}^{(\ell,t)}}{\|X_{\text{Opt}}^{(\ell,t)} - P_{\text{Ret}}^{(\ell,t)}\|_2}$$

[论文原文] S 代表 target speaker 在 latent space 中的 step-wise identity-related direction [§2.2]。仅需单条 opt-out 语音即可计算。

[agent 解读] L2 归一化确保 steering 强度由独立参数 α 控制,而非由激活幅度决定,使不同说话人的干预强度可比。

#### 3. 动态 Layer-Step 选择

核心洞察: "not all layers contribute equally to maintain speaker identity" [论文原文][§2.3]。Fig. 3 显示不同层在不同 flow step 的 cosine similarity 动态变化: 浅层(第 1 层)后期 step 相似度降低,深层(第 20 层)早期 step 相似度降低。

**两阶段过滤**:

**Stage 1 -- Layer-level**: 计算每层的步级平均 cosine similarity [Eq. 3]:
$$\bar{c}^{(\ell)} = \frac{1}{T} \sum_{t=1}^{T} c^{(\ell,t)}$$

基于全局统计设定动态阈值 [Eq. 5]:
$$\tau = \mu + k\sigma$$

选择 c̄^(ℓ) < τ 的层作为干预层。低相似度 = 偏离 ID-prototype 大 = 该层承载了较强的身份特异性信息 [论文原文]。

**Stage 2 -- Step-level**: 在每个选中的层 ℓ' 内,仅干预 c^(ℓ',t') < c̄^(ℓ') 的流步,进一步稀疏化干预点 [论文原文][§2.3]。

阈值 τ 随 opt-out speaker 变化(因为 cosine similarity 分布因人而异),实现了 per-speaker 自适应干预 [论文原文]。

Ablation [Table 4]: τ=µ+σ 是最优平衡点; all layers 干预导致 WER-SO 从 3.25 升至 3.71(过度干预破坏内容); µ 或 µ-σ 阈值导致身份抑制不足(SIM-SO 分别为 0.538, 0.567)。

#### 4. Unlearning via Projection Subtraction

对选定的干预点 (ℓ',t'),投影减法消除身份分量 [Eq. 6]:

$$\bar{X}_{\text{Opt}}^{(\ell',t')} = X_{\text{Opt}}^{(\ell',t')} - \alpha (X_{\text{Opt}}^{(\ell',t')} \cdot S^{(\ell',t')}) S^{(\ell',t')}$$

[论文原文] 这只移除与 identity direction 对齐的分量,保留语言和韵律内容,不引入不可控扰动 [§2.4]。α=1.2 经验值 [§3.1]。

[agent 解读] 投影减法的几何含义: 将激活值在 identity-specific 方向上的投影移除,保留正交分量(编码 linguistic content + prosody)。α=1.2 > 1 意味着轻微过补偿,可能是为了消除在非 steering 方向上的 identity 残余泄漏。相比 EmoSteer-TTS 的加法 steering(X + α * S),TruS 的减法设计更适合"消除"目标 -- 加法引入新方向,减法移除现有方向。

### 训练策略

**无训练**。TruS 完全在推理时工作。唯一预计算: 用 30 个 retain speakers 在 F5-TTS 上做一次前向传播生成 ID-prototype。

对比: TGU 430 A6000 GPU-hours, SGU 48 hours [Table 1]。TruS 的额外推理开销仅为计算 cosine similarity 和投影减法。

## 实验

| 指标 | TruS | F5-TTS (原始) | F5-TTS-FT | SGU | TGU | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER-R ↓ | 1.95† | 1.95 | 2.07 | 2.12 | 2.21 | LibriSpeech | [Table 1] |
| SIM-R ↑ | 0.678† | 0.678 | 0.654 | 0.290 | 0.549 | LibriSpeech | [Table 1] |
| Spk-ZRF-R | 0.908† | 0.908 | 0.911 | 0.935 | 0.921 | LibriSpeech | [Table 1] |
| WER-SO ↓ | 3.25 | 3.36 | 3.13 | 3.70 | 4.03 | Emilia (seen opt-out) | [Table 1] |
| SIM-SO ↓ | 0.477 | 0.657 | 0.656 | 0.106 | 0.510 | Emilia (seen opt-out) | [Table 1] |
| Spk-ZRF-SO ↑ | 0.929 | 0.925 | 0.924 | 0.959 | 0.933 | Emilia (seen opt-out) | [Table 1] |
| WER-UO ↓ | 3.26 | 2.03 | — | — | — | LibriSpeech (unseen) | [Table 2] |
| SIM-UO ↓ | 0.488 | 0.668 | — | — | — | LibriSpeech (unseen) | [Table 2] |
| Spk-ZRF-UO ↑ | 0.913 | 0.906 | — | — | — | LibriSpeech (unseen) | [Table 2] |
| SIM-Emo ↑ | 0.723 | 0.732 | — | — | — | CREMA-D | [Table 3] |
| Training hours | 0 | — | 52 | 48 | 430 | 2x A6000 | [Table 1] |

†: TruS 仅对 opt-out speakers 施加 steering,retain 生成与原始模型完全一致。

**关键发现**:

1. **Retain speakers 零损失** [Table 1]: TruS 的 retain 指标与原始 F5-TTS 完全一致,因为 steering 仅在检测到 opt-out 时激活。SGU 的 SIM-R 从 0.678 崩至 0.290(生成所有人的声音都变随机了),TGU 的 SIM-R 降至 0.549(重训练波及了 retain speakers)。

2. **身份抑制效果** [Table 1]: SIM-SO 从 0.657 降至 0.477(降 27%),同时 WER-SO 3.25 优于 TGU 4.03 和 SGU 3.70,说明 TruS 在抑制 identity 的同时保持了更好的语音内容可懂性。

3. **首次实现 unseen opt-out** [Table 2]: TGU/SGU 仅适用于训练集内 seen speakers。TruS 在 unseen opt-out 上 SIM-UO 从 0.668 降至 0.488(降 27%),WER-UO 轻微升至 3.26,证明方法可泛化到训练集外的说话人。

4. **情感保持** [Table 3]: SIM-Emo 仅从 0.732 降至 0.723(降 1.2%),同时 SIM-UO 从 0.217 降至 0.131(CREMA-D 上)。说明 steering 精准地消除了 speaker identity 方向,而保持了正交的 paralinguistic 属性。

5. **动态层选择的必要性** [Table 4]: τ=µ+σ 最佳; 全层干预虽然 SIM-SO 略低(0.462 vs 0.477),但 WER-SO 大幅恶化(3.71 vs 3.25); 更保守的阈值(µ 或 µ-σ)身份抑制不充分。

6. **Retain pool size** [Table 5]: N=30 最优,N=10 抑制不足(SIM-SO=0.535),N=50 反而在 seen data 上退化(WER-SO=3.71)。

## 局限性

1. **仅验证 DiT-based 架构**: 所有实验基于 F5-TTS 的 DiT blocks。是否适用于 AR-based TTS(VALL-E/CosyVoice 的 LLM 部分)或非 DiT diffusion 模型(UNet-based NaturalSpeech 2)完全未知。论文声称"generally applicable to DiT-based architectures" [§2.1] 但未在其他 DiT 模型上验证 [agent 解读]

2. **身份抑制不如重训练彻底**: SIM-SO=0.477 仍显著高于 SGU 的 0.106 [Table 1]。在安全关键场景中,0.477 的相似度可能仍被 speaker verification 系统识别为同一人(一般阈值 0.3-0.5) [agent 解读]

3. **完全缺失对抗/恢复攻击分析**: Kim et al. (ICML 2025) 至少测试了 fine-tuning 恢复攻击。TruS 未讨论: (1) 攻击者反向工程 steering vector 的可能性; (2) 多次生成取平均是否消除 steering 效果; (3) 对参考音频做微小扰动使 steering 失效 [agent 解读]

4. **相似声音误伤未探讨**: 如果两个人声音天然相似,对其中一人的 steering 是否影响另一人?Kim et al. 做了弱分析(Pearson r=0.14),TruS 完全未涉及 [agent 解读]

5. **单条参考语音的稳定性**: opt-out speaker 的 steering 仅基于单条参考。如果参考语音不能充分代表说话人的 identity 空间(如特殊语调/情感状态),steering 效果可能不稳定 [agent 解读]

6. **单语言验证**: 仅在 Emilia 英语子集上实验,跨语言场景未测试 [论文原文][§3.1]

## 点评

TruS 的核心贡献是范式转换: 将 speaker unlearning 从"修改模型权重"(TGU/SGU 的重训练范式)重构为"推理时信号处理"。这带来了三个实质性优势: (1) 零训练成本(0 vs 430 GPU-hours); (2) 首次支持 unseen speakers; (3) retain speakers 完全不受影响。这使 opt-out 机制变成可热插拔的推理插件,而非需要服务中断的模型重训练。

**方法设计的亮点**: (1) 从 NLP 的 activation steering (Rimsky et al. [22]; Turner et al. [23]) 迁移到 TTS 时,增加了 TTS 特有的 flow step 维度的动态性,比 NLP 中的静态层选择更精细; (2) 投影减法比 EmoSteer-TTS 的加法 steering 更适合"消除"目标; (3) ID-prototype 概念简洁优雅,通过平均消去 speaker-specific 信息获得 identity-neutral 锚点。

**值得质疑的点**: (1) SIM-SO=0.477 的安全性 -- 在 speaker verification 的常用阈值范围内,这个相似度可能仍然足以关联到原说话人,论文未讨论"什么水平才算安全"; (2) 完全缺失对抗鲁棒性分析是显著弱点 -- 由于 steering 完全在推理时施加且方法公开,攻击者理论上可以通过去除 steering 步骤恢复原始生成; (3) 与 EmoSteer-TTS 的技术同源性 -- 两者都在 DiT FFN 激活上做 steering,TruS 的新颖性主要在问题定义和动态选择机制,方法论创新较增量。

**与 KB 已有工作的关系**: TruS 进一步完善了 TTS 安全防线。TruS(推理时阻止) + TGU(模型端遗忘) + SafeSpeech(数据端扰动) + TraceableSpeech(事后溯源)可构成多层防护体系。但 TruS 和 TGU 在同一层级(模型端)互为替代而非互补 -- TruS 的免训练特性使其在大多数部署场景中更具实用价值。

## 可复用的 idea

1. **ID-prototype + 投影减法进行属性消除**: 对任意属性(speaker/emotion/accent),用 N 个不同样本的激活平均值作为 neutral 锚点,计算待消除样本与锚点的差异方向,投影减法移除该分量。关键是选择恰当的特征提取点(FFN vs attention)。可迁移到 voice conversion 的 speaker disentanglement、accent neutralization 等任务

2. **动态 µ+kσ 阈值进行层选择**: 比 top-k 或固定层集合更具适应性,k 作为唯一超参数控制保守/激进权衡。可用于任何需要选择性干预 transformer 中间层的场景(LLM detoxification、style transfer 等)

3. **推理时投影减法公式**: X_bar = X - alpha * (X . S) * S 是极简的属性控制公式,如果能找到其他属性的 steering direction(emotion/prosody/accent),同一框架可直接复用

4. **Training-free safety 插件模式**: 不修改模型权重,可在任意已部署模型上热加载,适合 production 环境的安全合规需求。这种"安全即插件"的设计思想值得 TTS 服务架构参考

## 审阅

<!-- 审阅由独立 subagent 完成 -->

---

检索命中: [[ConditionalFlowMatching]]✓, [[SpeakerEmbedding]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[SpeakerVerification]][待确认], [[Anti-spoofingandDeepfakeDetection]][待确认], [[Emilia]][待确认] | 未命中但可能相关: [[DiffusionModel]](DiT 基础), [[EmotionControlinTTS]](EmoSteer-TTS 关联)
