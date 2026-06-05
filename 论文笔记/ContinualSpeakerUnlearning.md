---
type: paper
tier: deep
title: "Continual Speaker Identity Unlearning with Minimal Interference"
arxiv_id: "2605.25962"
source: "Sources/ContinualSpeakerUnlearning.pdf"
authors: [Jinju Kim, Yunsung Kang, Gyeong-Moon Park, Jong Hwan Ko]
year: 2026
venue: "arXiv preprint"
tags: [machine-unlearning, voice-privacy, zero-shot-TTS, speaker-identity, continual-learning, flow-matching, GDPR, safety, Fisher-information, gradient-projection]
concepts: ["[[ConditionalFlowMatching]]", "[[SpeakerEmbedding]]", "[[SpeakerVerification]]", "[[Anti-spoofingandDeepfakeDetection]]", "[[VoiceCloningTaxonomy]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 2 个待确认实体页: [[ConditionalFlowMatching]], [[SpeakerEmbedding]], [[Zero-shotSpeechSynthesis]], [[SpeakerVerification]], [[Anti-spoofingandDeepfakeDetection]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文是 [[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning]] (Kim et al., ICML 2025) 的直接后续,将 speaker identity unlearning 从"一次性同时遗忘"推进到"持续增量遗忘"。KB 中 [[Anti-spoofingandDeepfakeDetection]] 页已记录的三层安全防线(unlearning + perturbation + tracing)中,本文深化了 unlearning 层的部署可行性 -- 现实中遗忘请求不会一次性到达而是随时间持续积累。同时,KB 中已有 [[论文笔记/Training-freeSpeakerUnlearning|TruS]] (Lee et al., 2026) 提出推理时 unlearning (activation steering),与本文的训练时 continual unlearning 互补。
>
> **已有认知**: [[ConditionalFlowMatching]] 页(confirmed)记录了 VoiceBox 使用的 CFM 框架,本文继续以 VoiceBox 为 backbone。[[SpeakerEmbedding]] 页(confirmed)记录了 WavLM-TDCNN 等 speaker encoder,本文使用 WavLM-TDCNN 计算 SIM 评估遗忘效果。[[Zero-shotSpeechSynthesis]] 页(confirmed)记录了 ZS-TTS 的主流范式和评估体系。[[SpeakerVerification]] 页[待确认]记录了 SIM 计算标准做法,本文沿用同一评估体系并校准了 SIM 的 retention/forgetting 阈值(same-speaker lower bound 0.46, different-speaker upper bound 0.32)。前作 Speaker Identity Unlearning 笔记已详细记录了 TGU 的工作原理和局限,其中明确指出"无增量 unlearning 讨论"和"仅在 VoiceBox 上验证"两个局限,本文正是针对前者的解决方案。
>
> **创新判断**: 前作假设所有 forget speaker 一次性到达并同时遗忘,本文证明这一假设在现实部署中不成立 -- 逐个遗忘会导致"灾难性再学习"(先前遗忘的说话人身份复活)。这是一个与 continual learning 中的"灾难性遗忘"对偶的新问题。CORTIS 的两层机制(参数级 Fisher 定位 + 方向级正交投影)借鉴了 continual learning 的经典方法(OGD/GPM),但适配到了 unlearning 场景 -- 保护的不是"已学到的知识"而是"已遗忘的状态"。
>
> 检索命中: [[ConditionalFlowMatching]]✓, [[SpeakerEmbedding]]✓, [[Zero-shotSpeechSynthesis]]✓, [[SpeakerVerification]][待确认], [[Anti-spoofingandDeepfakeDetection]][待确认] | 过滤: 无 | 未命中但可能相关: [[VoiceCloningTaxonomy]][待确认]

## 速查

> [!summary] 速查
> - **一句话**: 提出 CORTIS 框架,首次解决 ZS-TTS 中持续增量 speaker identity unlearning 问题 -- 通过 Fisher-information 参数定位 + 正交子空间投影,使每次新遗忘不干扰先前已遗忘的说话人身份
> - **路线**: 遗忘请求 f_i 到达 → 计算 forget/remain/prior-forget Fisher → 对比显著性 top-k% 构建参数 mask M_i → 在 M_i 限定区域内执行 TGU unlearning → 收集梯度快照 → SVD 提取 rank-R 正交基 U_i → 后续请求的梯度投影到 U_{<i} 的正交补空间
> - **指标**: 5 次顺序遗忘后所有 forget speaker SIM < 0.18 (最差 0.198) [Fig 3]; S-R 保持 0.527 (原模型 0.649) [Table 7]; 3 次遗忘后 S-f1=0.172, S-f2=0.148, S-f3=0.124, S-R=0.557 [Table 1]; 每次请求仅需 3.5 小时 vs TGU 累积模式 87.5 小时 (3 次后) [Table 4]
> - **可借鉴**: "对偶灾难性"问题的形式化 -- continual learning 防止"灾难性遗忘"(已学知识丢失),continual unlearning 防止"灾难性再学习"(已遗忘知识复活),两者对称但需要不同机制; Fisher-information 对比显著性 mask 的设计(分子=forget Fisher, 分母=max(remain, prior forget) Fisher)可迁移到其他需要局部化更新的场景
> - **局限**: 仅在 VoiceBox (CFM-based NAR) 上验证,未测试 AR/LLM-based TTS; S-R 在 5 次遗忘后降至 0.527 (原模型 0.649,降幅 19%); 未讨论对抗性攻击(fine-tuning 恢复); 最长序列仅测试 5 个 speaker

## 核心问题

现实中 RTBF (Right to Be Forgotten) 请求是逐一到达的 -- 今天一个用户,下个月另一个用户。前作 [[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning]] (ICML 2025) 假设所有遗忘请求一次性到达,但本文证明这一假设导致两个根本性问题 [§3.1-3.2]:

1. **数据保留悖论 (Data Retention Paradox)**: 如果要做同时遗忘,服务商必须保留所有请求者的数据直到最终一次性处理 -- 但保留数据恰恰违反了 RTBF 要求被行使的权利 [论文原文][§3.2]

2. **灾难性再学习 (Catastrophic Re-learning)**: 逐个应用前作的 TGU 方法时,遗忘新 speaker f_2 会导致已遗忘的 f_1 的身份复活。具体机制: TGU 的 retain loss 保护了模型的整体质量,但没有提供任何监督信号保护先前遗忘 speaker 的参数不被覆盖;而 ZS-TTS 模型天然具有泛化能力,retain loss 提供的正则化足以使先前遗忘的 speaker 恢复 [论文原文][§3.2]。证据: 用 TGU 逐步遗忘 f1→f2→f3 后,S-f1 从请求 1 后的 0.164 反弹到请求 2 后的 0.612 [Table 1]

[agent 解读] 这是一个与 continual learning 中"灾难性遗忘"对偶但本质不同的问题: continual learning 中,学新任务会覆盖旧任务的知识(学 f2 导致忘 f1); continual unlearning 中,遗忘新 speaker 会恢复旧 speaker 的知识(忘 f2 导致记起 f1)。关键区别在于: continual learning 的标准对策(正则化)恰恰是 continual unlearning 的失败原因 -- 正则化保护模型能力 = 保护了模型克隆先前 speaker 的能力。

**形式化约束** [§3.1]:
- (C1) 顺序到达: 请求逐个到达,处理 f_i 时不知道未来的 f_{i+1}
- (C2) 遗忘后不保留数据: 处理完 f_i 后立即删除 D_{f_i},后续步骤不可访问

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CORTIS (Cumulative ORThogonal Identity Suppression) 在 VoiceBox (CFM-based ZS-TTS) 上实施,包含两个互补机制 [§4, Figure 2]:

1. **参数级定位** (Contrastive Parameter Localization): 通过 Fisher information 对比显著性,将每次 unlearning 更新限制在与当前 forget speaker 最相关的 top-k% 参数上
2. **方向级保护** (Orthogonal Projection): 将更新投影到先前遗忘子空间的正交补上,防止更新沿先前遗忘方向漂移

### 关键设计选择

#### 1. 对比参数定位 (Contrastive Parameter Localization) [§4.1]

**动机**: ZS-TTS 模型的表示高度纠缠,speaker identity 未被显式模块化分离。不加约束的优化器可能更新与 speaker identity 无关的参数,更危险的是,可能覆盖驱动先前成功遗忘的参数 [论文原文][§4.1]。

**具体实现**: 在遗忘序列 i 时:
1. 计算 forget set f_i 上的对角 Fisher Information Matrix F_{f_i}
2. 计算 remain set R_i 上的 Fisher F_{R_i}
3. 构建对比显著性图 [Eq. 2]:

$$\text{saliency}_i = \frac{F_{f_i} + \epsilon}{\max(F_{R_i}, F_{f_1}, \ldots, F_{f_{i-1}}) + \epsilon}$$

4. 取全局 top-k% (k=30) 作为可训练 mask M_i,其余参数冻结

**设计理由**: 分母取 element-wise max,意味着任何对 retain 质量或任何先前 forget speaker 重要的参数都会被压到显著性排名底部 -- 这是一个 soft guard,使每次遗忘操作集中在与当前 speaker 最相关且最不可能干扰其他功能的参数上 [论文原文][§4.1]。

[agent 解读] Fisher information 在这里的角色与 EWC (Elastic Weight Consolidation) 中类似,但方向相反: EWC 用 Fisher 找"不能动"的参数(保护旧任务),CORTIS 用 Fisher 找"应该动"的参数(定位遗忘目标)。分母中包含 prior forget Fishers 是关键创新 -- 确保当前遗忘操作不会修改先前遗忘操作所依赖的参数。

**实证验证**: Figure 6 报告了 5 个 forget speaker 之间 mask 的 pairwise Jaccard overlap,所有 off-diagonal 值 < 0.20,证实对比显著性成功将每个 speaker 的遗忘定位到大致不重叠的参数子集 [Appendix G]。

#### 2. 正交投影 (Orthogonal Projection on Cumulative Forget Subspace) [§4.2]

**动机**: 参数 mask 将更新限制在相关区域,但不约束区域内更新的方向。被 mask 选中的参数仍可能沿先前遗忘使用的方向漂移,导致 re-learning [论文原文][§4.2]。

**具体实现**:
1. 完成 f_i 的遗忘训练后,按固定间隔收集梯度快照 (i=1: 每 150 步/10K 步; i>1: 每 15 步/1K 步) [Appendix B]
2. 堆叠梯度快照,做 truncated SVD,取 top-R (R=40) 左奇异向量作为 per-speaker 正交基 U_i
3. 构建时先减去先前子空间,使 U_i 与 U_1,...,U_{i-1} 正交
4. 维护固定秩合并基: 用能量加权列堆叠 Phi_i = [U_1*Sigma_1 | ... | U_i*Sigma_i],取 rank-R_merge truncated SVD → U_{<i}
5. 每个优化步骤后,将 mask 内的权重更新 delta 投影到 U_{<i} 的正交补 [Eq. 3]:

$$\delta \leftarrow \delta - U_{<i} U_{<i}^\top \delta$$

**为什么用固定秩合并而非直接拼接**: 直接拼接 [U_1|...|U_i] 的累积基随 speaker 数线性增长,投影成本随请求序列长度增加。固定秩合并用 SVD 截断到 R_merge,使投影成本为常数,不受 i 影响 [论文原文][§4.2]。

[agent 解读] 这一设计直接借鉴了 continual learning 中的 OGD (Orthogonal Gradient Descent) 和 GPM (Gradient Projection Memory) 方法,但目标相反: OGD/GPM 保护"已学到的任务知识"不被新任务覆盖,CORTIS 保护"已执行的遗忘效果"不被新遗忘操作还原。对偶关系非常优雅。

#### 3. 底层遗忘机制

CORTIS 在 TGU (Teacher-Guided Unlearning) 之上运行。TGU 的基本原理沿用前作: teacher 仅以文本为条件生成随机音色语音,student 训练在遇到 forget speaker 时模仿 teacher 的随机输出。CORTIS 的两层机制(mask + projection)约束了 TGU 的更新范围和方向 [论文原文][Appendix B]。

### 训练策略

- 第一次遗忘 (i=1): 10K 步,学习率 5e-5,500 步 linear warmup + linear decay [Appendix B]
- 后续遗忘 (i>1): 仅 1K 步,学习率 5e-6 [Appendix B]
- k=30 (冻结 70% 参数) [Appendix B]
- R=40 (per-speaker 正交基维度) [Appendix B]
- 梯度快照: i=1 每 150 步采集, i>1 每 15 步采集 [Appendix B]
- Gram matrix trick 避免显式构建大规模 M_i 矩阵: G_i = M_i^T M_i 仅 C x C (C <= 200),再通过特征分解恢复左奇异向量 [Appendix A]

## 实验

| 指标 | CORTIS | TGU (seq.) | SGU (seq.) | UN | SelFT | Original | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-f1 (after Req 1) ↓ | 0.162 | 0.164 | 0.165 | 0.229 | 0.154 | 0.721 | LibriHeavy forget | [Table 1] |
| S-f1 (after Req 2) ↓ | 0.185 | **0.612** | 0.178 | 0.344 | 0.482 | 0.721 | LibriHeavy forget | [Table 1] |
| S-f1 (after Req 3) ↓ | 0.172 | **0.603** | 0.233 | **0.638** | **0.553** | 0.721 | LibriHeavy forget | [Table 1] |
| S-f2 (after Req 3) ↓ | 0.148 | **0.546** | 0.101 | **0.555** | **0.434** | 0.674 | LibriHeavy forget | [Table 1] |
| S-f3 (after Req 3) ↓ | 0.124 | 0.180 | 0.192 | 0.106 | 0.110 | 0.866 | LibriHeavy forget | [Table 1] |
| S-R (after Req 3) ↑ | 0.557 | 0.582 | 0.315 | 0.580 | 0.548 | 0.649 | LibriSpeech test-clean | [Table 1] |
| W-R (after Req 3) ↓ | 2.8 | 3.0 | 2.7 | 3.0 | 2.7 | 2.1 | LibriSpeech test-clean | [Table 1] |
| S-R (after Req 5) ↑ | 0.527 | - | - | - | - | 0.649 | LibriSpeech test-clean | [Table 7] |
| 训练时间/请求 | 3.5h | 29h | - | 29h | 22h | - | 2x A100 80GB | [Table 4] |
| Peak GPU Mem | 49.3 GB | 30.8 GB | - | 30.8 GB | 48.5 GB | - | 2x A100 80GB | [Table 4] |

**加粗**标记表示该 SIM 值 > 0.32 (forgetting failure threshold) [Appendix E]。

**关键发现**:

1. **TGU 遭遇灾难性再学习** [Table 1]: TGU 在请求 1 后 S-f1=0.164 (成功遗忘),但请求 2 后 S-f1 反弹到 0.612 (几乎恢复原始水平),请求 3 后仍为 0.603。TGU 仅能遗忘最近请求的 speaker,先前遗忘的 speaker 会被"记起"。

2. **SGU 遭遇灾难性遗忘** [Table 1]: SGU 的所有 forget speaker SIM 保持低位(<0.24),但 S-R 从 0.479 单调下降到 0.315 -- remain speaker 的克隆质量严重退化。[agent 解读] SGU 的破坏性更新虽然"殃及"了先前遗忘的 speaker(使其保持遗忘),但也殃及了 remain speaker。

3. **Continual unlearning 正则化方法不足** [Table 1]: UN 和 SelFT (来自图像域的 continual unlearning 方法) 成功保持了 S-R (0.580/0.548),但 S-f1 分别反弹到 0.638/0.553。限制参数漂移保护了 remain 性能,但不足以保护已遗忘身份 -- f_1 的身份编码方向在遗忘信号切换到 f_2 后就不再被保护 [论文原文][§5.4]。

4. **CORTIS 是唯一同时满足两个条件的方法** [Table 1]: (a) 所有 forget speaker SIM < 0.32 (S-f1=0.172, S-f2=0.148, S-f3=0.124); (b) S-R >= 0.46 (0.557)。没有任何 baseline 同时满足。

5. **长序列扩展性** [Fig 3, Table 7]: 5 次遗忘后,最差 forget SIM 为 0.198 (f1 at Req 4),所有 forget speaker SIM 始终 < 0.2; S-R 保持近平稳 (0.602→0.553→0.557→0.562→0.527),无急剧崩溃; 第 1 个遗忘的 speaker 在第 5 次请求后 SIM 仍为 0.178,未泄漏。

6. **Ablation: Mask alone 不够** [Table 2]: 去掉投影仅保留 mask,S-f1 和 S-f2 在请求 3 后反弹到 0.334/0.397,说明参数级定位能部分缓解 re-learning 但不充分 -- mask 内的参数仍可沿先前遗忘方向漂移。

7. **Ablation: Projection 对 mask 预算鲁棒** [Table 3]: k=20 vs k=30 的 forget SIM 相当,投影机制不依赖特定 mask 大小。但 k=20 过于限制导致 S-R 降至 0.523 (squeeze effect)。

8. **计算效率** [Table 4]: CORTIS 仅需 3K 步/请求 (vs TGU 10K 步),每请求 3.5 小时 (vs TGU 累积模式 87.5 小时在 3 次请求后),且成本为常数不随请求数增长。

## 局限性

1. **Backbone 范围有限**: 仅在 VoiceBox (CFM-based NAR) 上验证。作者明确指出 Fisher saliency 和梯度子空间投影在原理上是架构无关的,但缺少 AR codec-based (VALL-E) 和 diffusion-based (NaturalSpeech) 系统上的验证 [论文原文][§6]

2. **对抗鲁棒性未讨论**: 未研究针对发布模型的 fine-tuning 攻击、prompt engineering 或 activation-level 攻击来恢复已遗忘身份。前作 [Table 12] 已展示 15 min 数据 fine-tune 可部分恢复 speaker identity [论文原文][§6]

3. **Retain 质量的持续下降**: S-R 从 0.649 (原模型) 在 5 次遗忘后降至 0.527 (下降 19%) [Table 7]。虽然无急剧崩溃,但线性外推到数十次请求时 retain 质量可能不可接受 [agent 解读]

4. **序列长度仅测试 5 次**: 真实部署场景可能涉及数百甚至数千次遗忘请求。固定秩合并子空间能否在超长序列中保持遗忘耐久性存疑 [agent 解读]

5. **遗忘阈值的保守性**: SIM 的 forgetting failure threshold 设为 0.32 (基于不同说话人 SIM 上界),但真正的"安全遗忘"标准应取决于下游攻击能力而非统计分布 [agent 解读]

6. **无 MOS 主观评价**: 仅报告 WER 和 SIM 客观指标,缺少前作中的 SMOS/CMOS 主观评估 [agent 解读]

## 点评

本文的核心贡献在于识别并形式化了"灾难性再学习"这一新问题 -- 一个与 continual learning 中"灾难性遗忘"对偶但本质不同的现象。这一问题对 ZS-TTS 的安全部署具有直接实践意义: 如果遗忘请求必须积攒后一次性处理(前作假设),则服务商要么违反 RTBF(保留数据),要么面临不断增长的计算成本(累积遗忘)。CORTIS 提供了一个恒定成本的增量解决方案。

**方法设计的对称美学**: CORTIS 从 continual learning 借鉴了两个经典工具(Fisher-based regularization 和梯度子空间投影),但巧妙地翻转了它们的用途 -- 在 continual learning 中,Fisher 标识"重要参数(不能动)"和投影保护"已学方向"; 在 continual unlearning 中,Fisher 标识"遗忘相关参数(应该动)"且通过对比排除"已遗忘参数(不应再动)",投影保护"已遗忘方向"。这种对偶设计非常优雅。

**与 KB 已有工作的定位**: 本文进一步完善了 ZS-TTS 安全防线: (1) [[论文笔记/SpeakerIdentityUnlearning|TGU]] 解决了"能不能遗忘"的问题; (2) 本文解决了"能不能持续遗忘"的问题; (3) [[论文笔记/Training-freeSpeakerUnlearning|TruS]] 解决了"能不能零成本遗忘"的问题。三者分别对应训练时一次性/训练时增量/推理时无训练三种范式。未来可能的组合: 用 TruS 处理紧急请求(零延迟),用 CORTIS 定期批量巩固(高耐久性)。

**需要关注的弱点**: (1) 5 次请求后 S-R 已降 19%,长期部署的 retain 质量退化趋势未被充分分析; (2) 完全缺乏对抗性评估,鉴于前作已展示 fine-tune 恢复攻击的可行性; (3) VoiceBox 作为唯一 backbone 的局限性在 AR-dominant 的当前 ZS-TTS 生态中更加突出。

## 可复用的 idea

1. **灾难性再学习的形式化**: "在持续修改过程中,当前操作意外恢复先前已执行操作的效果"这一问题模式可迁移到: 模型编辑(持续编辑知识不回退)、概念擦除(持续擦除 diffusion model 概念不复活)、隐私保护(持续删除 LLM 知识不泄漏)

2. **对比 Fisher 显著性**: 用分子=目标 Fisher / 分母=max(约束 Fishers) 的设计,可用于任何需要"在不影响多个约束的前提下修改特定子集参数"的场景,如多任务学习中的选择性更新

3. **固定秩子空间合并**: 用能量加权 SVD 将任意多个正交基压缩到固定秩,使投影成本为常数 -- 这一 trick 可用于任何累积子空间约束的场景

---

> [!review] 审阅
> *待审阅 subagent dispatch*

检索命中: [[ConditionalFlowMatching]]✓, [[SpeakerEmbedding]]✓, [[Zero-shotSpeechSynthesis]]✓, [[SpeakerVerification]][待确认], [[Anti-spoofingandDeepfakeDetection]][待确认] | 过滤: 无 | 未命中但可能相关: [[VoiceCloningTaxonomy]][待确认]
