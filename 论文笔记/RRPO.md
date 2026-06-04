---
type: paper
tier: deep
title: "RRPO: Robust Reward Policy Optimization for LLM-Based Emotional TTS"
arxiv_id: "2512.04552"
source: "Sources/RRPO.pdf"
authors: [Cong Wang, Changfeng Gao, Yang Xiang, Zhihao Du, Keyu An, Han Zhao, Qian Chen, Xiangang Li, Yingming Gao, Ya Li]
year: 2025
venue: "arXiv preprint"
tags: [emotional-TTS, reinforcement-learning, reward-hacking, reward-model, robustness, LLM-TTS, post-training]
concepts: ["[[Differentiable Reward Optimization]]", "[[Emotion Control in TTS]]", "[[Gumbel-Softmax]]", "[[LLM-based TTS]]"]
models: ["[[CosyVoice 2]]"]
tasks: []
datasets: ["IEMOCAP", "ESD", "MER2023"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-based TTS]]✓, [[CosyVoice 2]]✓ | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Emotion Control in TTS]](pending-review), [[Gumbel-Softmax]](pending-review), [[Codec Language Model]](pending-review) | 未命中但可能相关: 无

**谱系定位**: 本文是 DiffRO (Gao et al., Interspeech 2025) 的直接后续工作,出自同一团队 (Tongyi Lab)。DiffRO 提出了在 token 空间通过 Gumbel-Softmax 可微采样实现 end-to-end reward 优化的框架,并通过 Multi-Task Reward (MTR) 实现了零样本情感控制。然而,DiffRO 的 Emotion Control 依赖 vanilla SER reward model,而 KB 中已有记录指出 RL-for-TTS 领域存在 reward hacking 问题 (Seed-TTS 最早发现, No Verifiable Reward for Prosody 提供了 GRPO 韵律坍缩的实验证据)。

**已有认知**: [[Differentiable Reward Optimization]][待确认] 页面已系统记录了 DiffRO 的核心机制 (Token2Text RM, Gumbel-Softmax, token-level KL) 及其与 GRPO/DPO/FPO/TKTO 等路线的对比。[[Emotion Control in TTS]][待确认] 页面覆盖了情感控制的多条路线 (embedding/层级/对抗/DPO/RL),其中 DiffRO-MTR 路线已被记录为"用 SER reward model 梯度间接引导 TTS LM 学习情感表达"。

**创新判断**: RRPO 的核心创新不在 policy optimization 算法本身 (仍沿用 DiffRO 框架),而在于**识别并解决 DiffRO 的 reward hacking 脆弱性** — 通过三层混合正则化 (Label Smoothing + Energy-Adaptive Mixup + Adversarial Training) 构建 robust RM,使 policy 无法通过生成 acoustic artifacts 获取虚假奖励。这填补了 DiffRO 演进线中"RM 质量保障"的空白。

## 速查

> [!summary] 速查
> - **一句话**: 在 DiffRO 框架中通过三层混合正则化 (LS + EAM + Adv) 构建鲁棒 Reward Model,解决 emotional TTS 中 policy 生成声学伪影骗取虚假奖励的 reward hacking 问题
> - **路线**: 文本+情感指令 → CosyVoice2 LLM (Gumbel-Softmax 可微采样) → speech tokens → Robust RM (SER head, hybrid regularization fine-tuned) → reward gradient → policy update
> - **指标**: E-MOS 3.78 (vs DiffRO 3.65, SFT 3.52), N-MOS 3.81 (vs DiffRO 3.61, SFT 3.72); SER WA on ESD 81.7% (vs baseline 64.4%); IEMOCAP 68.0% (cross-lingual) [Table 1, Table 2]
> - **可借鉴**: 三层 RM 正则化方案可迁移到任何基于可微 reward 的 TTS 后训练 — Label Smoothing 修正过度自信, Energy-Adaptive Mixup 平滑决策边界, Adversarial Training 增强扰动鲁棒性; 特别是 EAM 基于语音能量自适应计算混合系数的设计,比标准 Mixup 更适合语音信号
> - **局限**: 仅在单说话人中文数据 (10K 样本, 5 类情感) 上验证; 无零样本说话人泛化实验; 未与 GRPO/DPO 等非 DiffRO 路线对比; Adv 在部分数据集上与 EAM 存在 trade-off (Table 2 IEMOCAP/ESD 略降); 代码未开源

## 核心问题

本文要解决的核心问题: **DiffRO 的全可微优化虽然低方差高效率,但 RM 的任何缺陷都会被解析梯度放大** — vanilla SER RM 存在过度自信、脆弱决策边界、扰动敏感三个脆弱性,policy 可以通过生成非语义的声学伪影 (如不自然的嘴部咔嗒声、爆破音) 骗取高 reward 但牺牲感知质量。这是 DiffRO 框架从"有效"到"可靠"的关键瓶颈。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

RRPO 在 DiffRO 框架上做的改动集中在 Reward Model 端,policy optimization 端保持不变 [Fig 1]:

1. **Policy Model (可训练)**: CosyVoice2 的 Neural Codec Language Model,通过 Gumbel-Softmax 重参数化实现可微采样,接收 RM 梯度更新参数 [§2.1]
2. **Robust Reward Model (核心改进, RRPO 阶段冻结)**: 在预训练 SER Transformer encoder 基础上,先用混合正则化方案 fine-tune SER head,再在 policy optimization 阶段冻结,产生稳定的 reward signal [§2.2, Fig 1]
3. **优化流程**: reward gradient 通过 chain rule 直接回传到 policy — $\nabla_\theta J(\theta) = \nabla_\theta R_{\text{robust}}(\tau(\theta))$ [Eq. 6]

[论文原文] DiffRO 的解析梯度提供了精确的方向和幅度 (direction + magnitude),这是其相比 DPO/GRPO 仅估计梯度方向的核心优势,但同时也意味着 RM 的缺陷会被精确放大 [§2.1]。

### 关键设计选择

**为什么不直接修改 policy optimization 算法,而是修复 RM?**

[论文原文] 作者的核心假设是: reward hacking 的根源在于 RM 不够鲁棒,而非优化算法本身有问题。一个 robust RM 不容易被简单的声学伪影欺骗,从而迫使 policy 放弃捷径,转向学习与人类感知对齐的复杂情感特征 [§1, §2.1]。

[agent 解读] 这个设计选择与 NLP 中 RLHF 的经验一致 — reward model quality 是整个 RL pipeline 的瓶颈。但与 NLP 不同的是,语音 RM 面临的攻击面更广: 非语义声学伪影 (如 click, pop) 在文本空间不存在,但在语音空间可以轻易被生成并欺骗分类器。

**三层混合正则化的设计逻辑**:

1. **Label Smoothing (修正过度自信)** [§2.2.1]: 离散情感类别标签 (angry/happy/sad/surprised/fearful) 无法捕捉情感的连续和模糊本质。LS 将 hard one-hot 标签 $y$ 替换为 soft 分布 $y'_k = (1-\epsilon) \cdot y_k + \epsilon/K$ ($\epsilon=0.1$, $K=5$) [Eq. 2]。[论文原文] 这惩罚了过度自信的预测,提升了 RM 的鲁棒性 [§2.2.1]。

2. **Energy-Adaptive Mixup (修正脆弱决策边界)** [§2.2.2]: 标准 Mixup 对语音信号不适用,因为混合比例需要考虑不同片段的能量差异。EAM 根据两个混合语音片段的相对能量和持续时间自适应计算混合系数 $\lambda_i$ [Algorithm 1]:
   - 随机抽取两个样本的子片段 (长度 $l_{\text{mix}} \sim U[1, l_i/2]$)
   - 按目标 SNR 缩放第二个片段的能量: $E'_j = E_i \cdot 10^{r/10}$, $r \sim U[r_{\min}, r_{\max}]$
   - 通过 overlap-add 生成混合特征
   - 混合系数 $\lambda_i = (l_{\text{mix}}/l_i) \times (E'_j / (E_i + E'_j))$
   - 最终 loss 是对两个原始标签的自适应加权插值 [Eq. 3]

   [论文原文] EAM 鼓励 RM 学习数据点之间的平滑过渡,修正尖锐脆弱的决策边界,使 policy 更难利用此类漏洞 [§2.2.2]。

3. **Adversarial Training (修正扰动敏感性)** [§2.2.3]: 在高层 embedding $h'$ 上施加 worst-case 扰动 $\delta = \epsilon_{\text{adv}} \cdot \nabla_{h'} L_{\text{emo}} / \|\nabla_{h'} L_{\text{emo}}\|_2$ ($\epsilon_{\text{adv}}=0.5$) [Eq. 4]。[论文原文] 扰动施加在高层 embedding 而非低层声学特征上,目标是攻击模型对高层情感特征的理解,而非低层声学 [§2.2.3]。[agent 解读] 这个选择很关键 — policy 的 reward hacking 是通过生成声学伪影 (低层) 来影响高层判断,因此在高层做 adversarial training 直接对抗了这种攻击模式。

**最终训练目标**: $L_{\text{ser}} = L_{\text{emo}} + \alpha \cdot L_{\text{adv}}$ ($\alpha=0.5$) [Eq. 5],其中 $L_{\text{emo}}$ 包含了 LS + EAM,$L_{\text{adv}}$ 是对抗损失。

### 训练策略

两阶段训练 [§3.1.1]:
1. **RM Fine-tuning**: 在 10K 中文情感样本上用混合正则化微调预训练 SER 模型 (learning rate $1 \times 10^{-5}$)
2. **Policy Optimization**: 用 fine-tuned robust RM 的 reward signal 通过 DiffRO 优化 CosyVoice2 LM

[agent 解读] 值得注意的是,同一组 10K 数据同时用于 SFT baseline、RM fine-tuning 和 policy optimization [§3.1.1]。作者声称"过拟合风险通过鲁棒的混合正则化方案和预训练模型的强先验来缓解",但这仍是一个潜在隐患 — 三个阶段共享完全相同的数据分布可能导致过于乐观的评估。

## 实验

| 指标 | RRPO (本文) | DiffRO | SFT | CosyVoice2 (Baseline) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| E-MOS (↑) | **3.78 ± 0.08** | 3.65 ± 0.11 | 3.52 ± 0.06 | 3.27 ± 0.09 | 内部测试集 | [Table 1] |
| N-MOS (↑) | **3.81 ± 0.09** | 3.61 ± 0.13 | 3.72 ± 0.07 | 3.65 ± 0.06 | 内部测试集 | [Table 1] |
| SER WA - IEMOCAP (en) | 68.0 | 66.0 | — | — | IEMOCAP | [Table 2] |
| SER WA - MER2023 (zh) | **54.8** | 50.9 | — | — | MER2023 | [Table 2] |
| SER WA - ESD (zh) | 81.7 | 64.4 | — | — | ESD | [Table 2] |

**消融分析** (Table 2, 逐步添加正则化组件):
- LS alone: ESD 64.4→72.8 (+8.4pp), 其他数据集小幅提升
- +EAM: ESD 72.8→82.3 (+9.5pp), IEMOCAP 66.8→69.1 (+2.3pp) — 单组件改进最大
- +Adv (完整 RRPO): MER2023 52.7→54.8 (+2.1pp),但 IEMOCAP 69.1→68.0 (-1.1pp), ESD 82.3→81.7 (-0.6pp)

[论文原文] 作者解释 Adv 在部分数据集上的轻微退化与"泛化和对抗鲁棒性之间的已知 trade-off"一致,表明 Adv 主要在更具挑战性的数据分布上发挥作用 [§3.2.2]。

**Reward hacking 的关键证据** (Table 1): DiffRO 在 E-MOS 上具有竞争力 (3.65),但 N-MOS (3.61) 甚至低于 SFT baseline (3.72)。[论文原文] 这揭示了 policy 找到了一条"有害捷径" — 不学习复杂的真实情感,而是生成微妙的声学伪影来欺骗 RM [§3.2.1]。RRPO 的 N-MOS (3.81) 同时超越了 SFT 和 DiffRO,证明 robust RM 有效阻止了这种捷径。

**跨语言泛化** (Table 2): RM 仅在中文数据上 fine-tune,但在英文 IEMOCAP 上也有提升 (66.0→68.0),作者认为这表明模型学到了语言无关的情感基础表征 [§3.2.2]。

## 局限性

1. **实验规模有限**: 仅 10K 样本、单一男性说话人、5 类情感、中文单语。未在多说话人/多语言/更多情感类别上验证
2. **数据复用风险**: 同一 10K 数据集同时用于 SFT、RM fine-tuning 和 policy optimization [§3.1.1],评估可能过于乐观
3. **对比范围窄**: 未与 GRPO、DPO、FPO 等非 DiffRO 路线对比;未与 EmoSteer-TTS、RLAIF-SPA 等最新情感控制方法对比
4. **缺乏客观 TTS 指标**: 无 WER/CER/SIM 等标准 TTS 指标,仅有 MOS 和 SER accuracy
5. **消融中的 trade-off**: Adversarial Training 在 IEMOCAP 和 ESD 上略微退化,说明三个正则化组件并非总是正向叠加
6. **未讨论 reward hacking 的定量分析**: 仅通过 N-MOS 下降间接证明 reward hacking 存在,缺乏对声学伪影的直接分析 (如频谱分析、artifacts 检测)

## 点评

**优点**: RRPO 对一个真实且重要的问题给出了清晰的诊断和针对性的解决方案。DiffRO 的 reward hacking 现象 (E-MOS 提升但 N-MOS 反降) 是 RL-for-TTS 领域的共性问题,本文从 RM robustness 角度切入是合理的。三层正则化的设计逻辑清晰 (过度自信→LS, 脆弱边界→EAM, 扰动敏感→Adv),EAM 的能量自适应机制是对标准 Mixup 的有意义改进。

**不足**: 实验设计是最大短板。单说话人+单语言+10K 样本的规模使得结论的可泛化性存疑。更关键的是三阶段共用同一数据集,这在 RL 评估中是一个 significant confound。此外,与 DiffRO 同团队的工作对比缺少了与其他 RL 路线 (GRPO, DPO) 的横向比较。

**在领域中的位置**: RRPO 填补了 DiffRO 演进线中 "RM 鲁棒性保障" 的空白,但其贡献主要在 RM fine-tuning 技术层面,而非 policy optimization 范式创新。三层正则化方案本身并不新颖 (LS/Mixup/Adv 都是已有技术),创新在于将它们组合应用于 TTS reward model 的特定场景并提供了 reward hacking 的实证。

## 可复用的 idea

1. **Energy-Adaptive Mixup for speech reward models**: 基于语音能量自适应计算混合系数的数据增强,比标准 Mixup 更适合处理能量差异大的语音片段。可迁移到任何语音分类/回归任务的数据增强
2. **三层 RM 正则化组合**: LS (置信度) + EAM (决策边界) + Adv (扰动) 的三层防御思路可迁移到其他基于可微 reward 的后训练场景 (如 MOS prediction, speaker verification reward)
3. **Reward hacking 诊断方法**: 通过比较 E-MOS vs N-MOS 的不一致来检测 reward hacking — 如果某个维度的 reward 提升伴随着其他维度的感知质量下降,则存在 hacking
4. **高层 embedding 上做 adversarial training**: 在 RM 的 representation 层面而非输入层面做对抗训练,更直接地对抗 policy 通过低层声学操纵影响高层判断的攻击模式

## 审阅

> [!review] 审阅 (2026-06-04, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节 WHY/HOW 清晰,速查卡片可借鉴具体 |
> | 可信赖 | pass | 数字标注覆盖率高,指标名正确 |
> | 可区分 | pass | 论文原文/agent解读 标注清晰,覆盖率>90% |
> | 可定位 | pass | KB 背景谱系定位明确,创新判断有对比基准 |
> | 不污染 | pass | 无新建概念页,反向更新均为安全 append |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> - medium: "Policy Model (冻结训练)" 错误暗示 policy 冻结 — 已修正
> - low: 主观评估数据集标注为"内部测试集"(原文局限)
> 详见 `_review/RRPO-review.yml`
