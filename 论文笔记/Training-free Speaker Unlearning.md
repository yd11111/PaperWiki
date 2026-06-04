---
type: paper
tier: deep
title: "Erasing Your Voice Before It's Heard: Training-free Speaker Unlearning for Zero-Shot Text-to-Speech"
arxiv_id: "2601.20481"
source: "Sources/Training-free Speaker Unlearning.pdf"
authors: [Myungjin Lee, Eunji Shin, Jiyoung Lee]
year: 2026
venue: "arXiv preprint"
tags: [machine-unlearning, voice-privacy, zero-shot-TTS, speaker-identity, activation-steering, training-free, inference-time, DiT, flow-matching]
concepts: ["[[Conditional Flow Matching]]", "[[Speaker Embedding]]", "[[Speaker Verification]]", "[[Anti-spoofing and Deepfake Detection]]", "[[Voice Cloning Taxonomy]]"]
models: []
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 3 个待确认实体页: [[Speaker Embedding]], [[Conditional Flow Matching]], [[Zero-shot Speech Synthesis]], [[Speaker Verification]], [[Anti-spoofing and Deepfake Detection]], [[Voice Cloning Taxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文是 [[Anti-spoofing and Deepfake Detection]] 中 **machine unlearning** 路线的第二篇工作,直接针对 [[论文笔记/Speaker Identity Unlearning|Kim et al. (ICML 2025)]] 提出的 retraining-based unlearning (TGU/SGU) 的核心瓶颈 -- 需要大量训练且无法处理 unseen opt-out speaker。KB 中已记录的 ZS-TTS 安全三层防线为: (1) 预防 (unlearning)、(2) 防护 (SafeSpeech: 数据端扰动)、(3) 溯源 (TraceableSpeech/Traceable TTS: 水印/指纹)。本文将 unlearning 从"训练时干预"推进到"推理时干预",是该路线的重大范式转变。
>
> **已有认知**: [[Speaker Embedding]] 页 (confirmed) 详细记录了 speaker embedding 的提取方式和注入方式,本文的核心操作对象正是 DiT 隐层中的 speaker identity 信息 -- 但不同于传统 speaker embedding 是显式的外部表示,本文操纵的是模型内部 FFN 激活中的 **隐式 identity 编码**。[[Conditional Flow Matching]] 页 (confirmed) 记录了 F5-TTS 使用的 flow matching 框架和 DiT backbone,本文正是在 F5-TTS 的 DiT blocks 上实施 activation steering。[[Zero-shot Speech Synthesis]] 页 (confirmed) 记录了当前 ZS-TTS 的主流方法和评估体系,本文的实验沿用了 SIM/WER/Emilia 等标准设置。[[Speaker Verification]] 页 [待确认] 记录了 ECAPA-TDNN 等 speaker encoder 的评估角色,本文正是使用 ECAPA-TDNN 的 SIM 指标衡量 unlearning 效果。
>
> **创新判断**: 相对于 KB 中记录的 Kim et al. (TGU/SGU),本文的创新在于: (1) **零训练成本**: TGU 需要 430 GPU 小时,本文 0 GPU 小时; (2) **泛化到 unseen speaker**: TGU 仅能 unlearn 训练集中的说话人,本文首次实现对未见说话人的 unlearning; (3) **即时响应**: 新的 opt-out 请求无需重训,只需一条参考语音即可在推理时阻断。与 SafeSpeech (数据端) 互补 -- SafeSpeech 保护用户音频不被 clone,TruS 则在推理时阻断已有模型对特定说话人的合成。
>
> 检索命中: [[Speaker Embedding]]✓, [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓, [[Speaker Verification]][待确认], [[Anti-spoofing and Deepfake Detection]][待确认], [[Voice Cloning Taxonomy]][待确认] | 过滤: 无 | 未命中但可能相关: [[Emotion Control in TTS]] (论文涉及情感保持评估)

## 速查

> [!summary] 速查
> - **一句话**: 首个 training-free 的 ZS-TTS speaker unlearning 框架,通过在推理时动态 steering DiT 隐层激活中的 identity 方向来抑制 opt-out speaker 的声音合成,同时保持语音内容和情感属性
> - **路线**: opt-out speaker 参考语音 + retain speaker 池(N=30) → 预计算 ID-prototype (各 DiT block 的 FFN 输出均值) → 推理时逐 step 计算 steering vector (opt-out 激活 - ID-prototype 的归一化差) → 动态选择 cos sim < tau 的 layer-step 组合 → 减去 identity 方向投影 → ODE solver + vocoder → unlearned 语音
> - **指标**: SIM-SO 0.477 vs TGU 0.510 (更低=更好的 unlearning) [Table 1]; WER-SO 3.25 vs TGU 4.03 [Table 1]; SIM-UO 0.488 (unseen, TGU 无法处理) [Table 2]; SIM-Emo 0.723 vs F5-TTS 0.732 (情感几乎不损) [Table 3]; 训练时间 0h vs TGU 430h [Table 1]
> - **可借鉴**: (1) "ID-prototype + dynamic threshold" 的两阶段 intervention selection 思路 -- 先用全局统计 (mu+k*sigma) 选 layer,再用 layer 内均值选 step,实现稀疏且自适应的干预; (2) 用 FFN 输出而非 attention 输出做 steering,因为 FFN 做了非线性通道混合后包含更强的 timbre/identity 信号 [§2.2]; (3) steering vector 的 L2 归一化确保方向纯净、强度由 alpha 单独控制
> - **局限**: 仅在 F5-TTS (DiT-based flow matching) 上验证,未测试 AR 架构; unseen opt-out 的 WER 上升明显 (UO: 2.03→3.26),seen opt-out WER 略有改善 (SO: 3.36→3.25); alpha=1.2 和 k=1 (对应 mu+sigma) 为经验值,缺乏理论指导; 未讨论声音相似 speaker 的误伤风险

## 核心问题

现有 ZS-TTS speaker unlearning 方法 (Kim et al., ICML 2025) 存在三个根本缺陷 [§1]:

1. **训练成本高**: TGU 需要 430 GPU 小时 (2x A6000) 进行重训,SGU 需要 48 GPU 小时 [Table 1]
2. **无法处理 unseen speaker**: TGU/SGU 只能 unlearn 训练集中见过的说话人,但真实世界中 opt-out 请求最可能来自训练集外的个人 [§1]
3. **无法即时响应**: 每当新的 opt-out 请求到来,需要从头重训,不适合规模化部署 [§1]

本文的目标: 将 unlearning 从 "data deletion + retraining" 范式转向 "inference-time control" 范式 -- 无需训练,仅需 opt-out speaker 的一条参考语音,即可在推理时阻断该说话人的声音合成。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TruS 建立在 F5-TTS [5] 之上,这是一个基于 [[Conditional Flow Matching]] 的 DiT-based ZS-TTS 模型,使用 Emilia 数据集预训练 [§2.1]。TruS 在推理时工作,不修改模型参数,通过操纵 DiT blocks 的 FFN 输出中的 identity-specific 激活方向来抑制 opt-out speaker 的身份信息 [论文原文][§2]。

**核心动机**: 现代 TTS 模型中,speaker identity 以结构化方向 (structured directions) 编码在隐层表示中。FFN 层的输出经过非线性通道混合后包含强烈的 timbre 和 identity 信号 [论文原文][§2.2, ref 27]。因此,通过在这些方向上做投影减法,可以选择性地移除 identity 信息而不破坏语言内容和韵律 [论文原文]。

**与 EmoSteer [19] 的区别**: EmoSteer 也在 TTS 中做 activation steering,但采用固定的 top-k 通道选择规则,对所有输入施加相同的干预。TruS 则为每个 opt-out speaker 动态计算 steering 方向和干预位置 [论文原文][§1]。

### 关键设计选择

#### 1. ID-prototype: 什么是"平均身份"?

TruS 首先用 N 个 retain speaker 的参考语音(每人一条)通过 F5-TTS 前向传播,提取每个 DiT block l 在每个 flow step t 的 FFN 输出,然后取平均得到 ID-prototype P_Ret^(l,t) [Eq. 1][§2.2]:

$$P_{\text{Ret}}^{(\ell,t)} = \frac{1}{N} \sum_{n=1}^{N} X_{\text{Ret}(n)}^{(\ell,t)}$$

**为什么用 FFN 输出**: [论文原文][§2.2] 引用 Lin et al. [27] 的发现 -- self-supervised speech transformer 的 FFN 层在非线性通道混合后编码了强烈的 speaker identity 信号。[agent 解读] 这与 LLM interpretability 中 FFN 编码 factual knowledge 的发现类似 -- FFN 的 key-value 结构天然适合存储属性级信息。

**retain speaker 数量 N=30**: 消融实验 [Table 5] 显示 N=10 时 SIM-SO=0.535 (抑制不够),N=50 时 seen data 性能反而下降 (SIM-SO=0.525, WER-SO=3.71),N=30 在所有指标上取得最佳平衡 [论文原文][§3.3]。

#### 2. Steering vector: 方向而非幅度

给定 opt-out speaker 的参考语音,提取其 FFN 输出 X_Opt^(l,t),steering vector 定义为 opt-out 激活与 ID-prototype 差值的 L2 归一化 [Eq. 2][§2.2]:

$$S^{(\ell,t)} = \frac{X_{\text{Opt}}^{(\ell,t)} - P_{\text{Ret}}^{(\ell,t)}}{\|X_{\text{Opt}}^{(\ell,t)} - P_{\text{Ret}}^{(\ell,t)}\|_2}$$

**为什么 L2 归一化**: [agent 解读] 归一化确保 steering vector 只编码 identity 偏离的 **方向**,具体干预强度完全由 alpha 参数控制。这比直接用差值更稳定 -- 不同 layer/step 的激活幅度差异很大,不归一化会导致某些位置过度干预而另一些不足。

#### 3. 动态层选择: 不是所有层都与 identity 相关

TruS 的关键创新之一是自动选择需要干预的 layer-step 组合。论文观察到 [Fig. 3][§2.3]: cos similarity 在不同 layer 和 step 上动态变化 -- 浅层在早期 step 相似度低(identity 编码活跃),深层在晚期 step 相似度增高(identity 已固化)。

**两阶段选择** [§2.3]:

阶段 1 -- Layer 级: 对每层 l 计算所有 step 的平均 cos similarity c_bar^(l) [Eq. 3],然后计算全局均值 mu 和标准差 sigma [Eq. 4],设动态阈值 tau = mu + k*sigma [Eq. 5]。c_bar^(l) < tau 的层被选为干预层。

阶段 2 -- Step 级: 在被选中的层 l' 内,只对满足 c^(l',t') < c_bar^(l') 的 step t' 施加干预 [§2.3]。

**为什么 k=1 (即 mu+sigma)**: 消融 [Table 4] 显示 mu-sigma 选择太少层 (SIM-SO=0.567, 抑制不足),mu 选中等数量 (SIM-SO=0.538),mu+sigma 最优 (SIM-SO=0.477, WER-SO=3.25),all 层几乎不比 mu+sigma 更好但 WER 大幅恶化 (WER-SO=3.71) [论文原文][§3.3]。

[agent 解读] 这个两阶段稀疏选择是 TruS 区别于 naive steering 的核心机制。如果盲目 steer 所有 layer-step,就像 "all" 策略那样,语音内容会被破坏; 而 mu+sigma 产生的稀疏干预集恰好覆盖 identity 最显著的位置。

#### 4. 推理时干预: 投影减法

在选中的层 l' 和 step t' 上,将激活中沿 steering vector 方向的分量减去 [Eq. 6][§2.4]:

$$\bar{X}_{\text{Opt}}^{(\ell',t')} = X_{\text{Opt}}^{(\ell',t')} - \alpha \left( X_{\text{Opt}}^{(\ell',t')} \cdot S^{(\ell',t')} \right) S^{(\ell',t')}$$

其中 alpha=1.2 控制干预强度 [§3.1]。

**为什么是投影减法而非加法/替换**: [论文原文][§2.4] 明确说这是为了"仅移除与 identity 方向对齐的分量,同时保留语言和韵律内容"。[agent 解读] 这在几何上等价于将激活投影到 steering vector 的正交补空间 (当 alpha=1 时),alpha=1.2 则稍微过补偿以确保更彻底的 identity 抑制。

### 训练策略

**无训练**。TruS 完全在推理时工作 [§3.1]:
- ID-prototype 可预计算并缓存 (仅需 30 个 retain speaker 各 1 条语音的前向传播)
- 每个新的 opt-out speaker 只需 1 条参考语音,实时计算 steering vector
- 可即时处理新的 opt-out 请求,无需重训

## 实验

| 指标 | TruS | TGU | SGU | F5-TTS (原) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER-R ↓ | 1.95 | 2.21 | 2.12 | 1.95 | LibriSpeech | [Table 1] |
| SIM-R ↑ | 0.678 | 0.549 | 0.290 | 0.678 | LibriSpeech | [Table 1] |
| WER-SO ↓ | 3.25 | 4.03 | 3.70 | 3.36 | Emilia (seen opt-out) | [Table 1] |
| SIM-SO ↓ | 0.477 | 0.510 | 0.106 | 0.657 | Emilia (seen opt-out) | [Table 1] |
| Spk-ZRF-SO ↑ | 0.929 | 0.933 | 0.959 | 0.925 | Emilia (seen opt-out) | [Table 1] |
| WER-UO ↓ | 3.26 | N/A | N/A | 2.03 | LibriSpeech (unseen opt-out) | [Table 2] |
| SIM-UO ↓ | 0.488 | N/A | N/A | 0.668 | LibriSpeech (unseen opt-out) | [Table 2] |
| Spk-ZRF-UO ↑ | 0.913 | N/A | N/A | 0.906 | LibriSpeech (unseen opt-out) | [Table 2] |
| SIM-Emo ↑ | 0.723 | N/A | N/A | 0.732 | CREMA-D (情感) | [Table 3] |
| 训练时间 | 0h | 430h | 48h | - | 2x A6000 | [Table 1] |

**关键发现**:

1. **Retain speaker 完全不受影响** [Table 1]: TruS 的 retain 指标 (WER-R=1.95, SIM-R=0.678) 与原 F5-TTS 完全一致,因为 TruS 只在 opt-out speaker 的推理时干预,retain speaker 走原路径 [论文原文][§3.2]
2. **优于 TGU 的 identity 抑制** [Table 1]: SIM-SO=0.477 vs TGU 0.510,且 WER-SO=3.25 vs TGU 4.03 -- 更强的 unlearning + 更好的内容保持 [论文原文][§3.2]
3. **SGU 的权衡**: SGU 达到最低 SIM-SO=0.106 和最高 Spk-ZRF=0.959,但 SIM-R=0.290 说明 retain speaker 的克隆质量严重崩溃 [论文原文][§3.2]
4. **首次实现 unseen opt-out speaker unlearning** [Table 2]: SIM-UO 从 0.668 降至 0.488,WER-UO 仅从 2.03 升至 3.26,证明 inference-time steering 可泛化到训练集外的个人 [论文原文][§3.2]
5. **情感保持** [Table 3]: SIM-Emo 仅从 0.732 降至 0.723 (降幅 1.2%),同时 SIM-UO 从 0.217 降至 0.131,说明 TruS 的 steering 精准地瞄准了 identity 而非 paralinguistic 属性 [论文原文][§3.2]

## 局限性

1. **仅在 F5-TTS 上验证**: 所有实验基于 DiT-based flow matching 架构。论文声称"generally applicable to other DiT-based TTS architectures" [§2.1],但未提供 AR 模型 (VALL-E, CosyVoice) 或非 DiT 架构的实验。对于 AR 模型中 speaker identity 的编码方式可能不同,steering 是否有效未知 [agent 解读]
2. **Unseen opt-out 的 WER 上升明显**: unseen opt-out 的 WER 从 2.03 升至 3.26 [Table 2],内容保真度损失不小; seen opt-out 的 WER 反而略有改善 (3.36→3.25 [Table 1]),说明 steering 对 unseen speaker 的干预精度不如 seen speaker [agent 解读]
3. **缺少恢复攻击分析**: Kim et al. 讨论了 fine-tuning 恢复攻击 [Table 12 of their paper],但 TruS 完全未讨论攻击者绕过推理时干预的可能性 -- 例如攻击者若能获取模型权重,可以直接跳过 TruS 的 steering 模块 [agent 解读]
4. **alpha=1.2 的鲁棒性**: 仅报告了 alpha=1.2 这一个值,未消融 alpha 对 identity suppression vs 内容保持的影响曲线 [agent 解读]
5. **未讨论声音相似 speaker 的误伤**: 与 forget speaker 声音高度相似的 retain speaker 是否会被部分抑制?Kim et al. 至少在 Appendix H 做了简要分析 (Pearson r=0.14),但本文完全未涉及 [agent 解读]
6. **单一参考语音的稳定性**: TruS 仅用 1 条 opt-out speaker 参考语音计算 steering vector [§2.2],未讨论不同参考语音的选择对 unlearning 效果的方差 [agent 解读]

## 点评

TruS 的核心价值在于将 speaker unlearning 从训练范式转向推理范式,这是一个本质性的效率跃迁 -- 430 GPU 小时 (TGU) → 0 小时。更重要的是,它首次解决了"unseen opt-out speaker"问题,这在实际部署中至关重要:现实世界中要求删除声音的用户最可能是训练集外的人。

**方法的优雅性**: 利用"FFN 编码 identity"的观察 + "cos similarity 的动态变化"来实现精准且稀疏的干预,避免了全局 steering 导致的内容破坏。投影减法在几何上直觉清晰,可解释性强。

**与 KB 已有工作的关系**: 本文与 [[论文笔记/Speaker Identity Unlearning|Kim et al. (ICML 2025)]] 形成清晰的技术对话 -- Kim et al. 定义了问题并提出首个 retraining-based 方案,TruS 将其推向 training-free 范式。两者共同扩展了 KB 中 [[Anti-spoofing and Deepfake Detection]] 页记录的 ZS-TTS 安全四层防线: 预防/模型级 (TGU unlearning) → 预防/推理级 (TruS steering) → 防护/数据级 (SafeSpeech) → 溯源 (Traceable TTS)。

**值得关注的弱点**: (1) 最大的安全隐患是 TruS 作为推理时插件,攻击者获取原始模型权重后可直接绕过 -- 而 TGU 修改了模型权重,绕过更难; (2) 缺少 ablation on alpha 和恢复攻击分析使安全性 claims 不够扎实; (3) 评估仅用 10 个 opt-out speaker,规模化场景 (数百 speaker 同时 opt-out) 的 ID-prototype 质量和计算效率未探讨。

总体而言,这是 speaker unlearning 方向一个重要的跟进工作,将 retraining-based 和 training-free 两条路线的 trade-off 呈现清楚: TGU 安全性更高但成本高且不能处理 unseen speaker; TruS 效率极高且覆盖 unseen speaker 但安全保障较弱。

## 可复用的 idea

1. **ID-prototype 作为"平均身份锚点"**: 用 N 个 retain speaker 的激活均值作为参照基准,任何偏离它的方向都是 identity-specific -- 这个思路可迁移到其他需要属性级 disentanglement 的场景 (如 emotion steering, accent steering)
2. **两阶段动态阈值 (global stats + local refinement)**: 先用全局 mu+sigma 选 layer,再用 layer-level mean 选 step。这种"先粗后细"的 intervention selection 比固定 top-k 更 robust,可用于 LLM interpretability 中的 activation patching
3. **投影减法做属性消除**: 将激活投影到目标属性方向的正交补空间,是一种通用的 training-free 属性控制方法,理论上可扩展到任何在隐层有结构化方向编码的属性

## 审阅

> [!review] 审阅结论: pass (2026-06-04)
> checklist v1.1 | 0 issues (0 high / 0 medium / 0 low)
> 方法四个设计选择均有 WHY 解释和消融支撑,因果来源标注完整 ([论文原文]/[agent 解读]),数字标注覆盖充分,KB 背景精准定位 (Kim et al. 后继 + 四层防线),局限性分析诚实 (安全性弱于 TGU/缺 alpha 消融/无恢复攻击)。详见 `_review/Training-free Speaker Unlearning-review.yml`。

检索命中: [[Speaker Embedding]]✓, [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓, [[Speaker Verification]][待确认], [[Anti-spoofing and Deepfake Detection]][待确认], [[Voice Cloning Taxonomy]][待确认] | 过滤: 无 | 未命中但可能相关: [[Emotion Control in TTS]]
