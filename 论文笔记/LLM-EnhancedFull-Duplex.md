---
type: paper
tier: deep
title: "LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems"
arxiv_id: "2502.14145"
source: "Sources/LLM-EnhancedFull-Duplex.pdf"
authors: [Hao Zhang, Weiwei Li, Rilin Chen, Vinay Kothapally, Meng Yu, Dong Yu]
year: 2025
venue: "Interspeech 2025"
tags: [full-duplex, dialogue-management, turn-taking, VAD, LLM, semantic-VAD, control-tokens, Tencent, cascaded-pipeline]
concepts: ["[[Full-duplexSpokenDialogue]]", "[[Turn-takinginSpokenDialogue]]", "[[StreamingSpokenDialogue]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeechLanguageModel]]✓, [[Full-duplexSpokenDialogue]], [[Turn-takinginSpokenDialogue]], [[StreamingSpokenDialogue]], [[SpokenDialogueEvaluation]], [[Speech-LLMIntegrationTaxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]], [[Full-duplexSpokenDialogue]], [[Turn-takinginSpokenDialogue]], [[StreamingSpokenDialogue]], [[SpokenDialogueEvaluation]], [[Speech-LLMIntegrationTaxonomy]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文属于全双工语音对话系统中的**级联/模块化**路线。与端到端方案 (Moshi 的 RQ-Transformer, LSLM 的 middle fusion + IRQ token, Freeze-Omni 的 chunk-level state prediction) 不同,本文将对话管理 (DM) 从核心对话引擎 (CDE) 中解耦,用一个轻量 LLM 专门处理轮次控制。

**已有认知**:
- [[Full-duplexSpokenDialogue]] 梳理了从 traditional → streaming → full-duplex 的渐进演进,以及 dGSLM/Moshi/LSLM/VITA/Freeze-Omni 等代表系统。本文的 6 模块级联架构 (AEC→Acoustic VAD→ASR→Semantic VAD→CDE→TTS) 最接近 CleanS2S 的结构化级联方案和 Wang et al. 的感知-动作-FSM 架构。
- [[Turn-takinginSpokenDialogue]] 记录了 turn-taking 从 VAD-only 到多模态检测再到端到端学习的演进。本文的 acoustic VAD 存在"无法区分有意打断和无意声音"的固有局限,正是现代系统转向语义方法的动机。Freeze-Omni 用 State 0/1/2 三状态,本文用 4 个 control tokens 做更细粒度的区分 (显式区分 real/fake INT)。
- [[Speech-LLMIntegrationTaxonomy]] [待确认] 将语音-LLM 集成分为 text-based / latent-representation / audio-token 三类。本文的 semantic VAD 属于 text-based integration — DM 处理的是 ASR 输出的文本,不直接处理语音信号。

**创新判断**: 本文的核心创新是将全双工轮次控制形式化为一个轻量 LLM 的 4-token 预测问题,并通过 DM/CDE 解耦实现独立优化。这与主流的端到端全双工趋势相反,选择了模块化路线以换取可控性和效率。

## 速查

> [!summary] 速查
> - **一句话**: 用 0.5B LLM (Hunyuan) 微调为"语义 VAD",预测 4 个控制 token 管理全双工对话轮次,将 DM 从 CDE 解耦以降低计算开销
> - **路线**: 用户语音 → AEC → Acoustic VAD → ASR → Semantic VAD (0.5B LLM, 预测 C-L/S-S/S-L/C-S) → CDE (仅在 S-S 时激活) → TTS
> - **指标**: 合成测试集 accuracy 97.85%, barge-in F1 0.999/1.000, 实录数据 accuracy 93.5%+ [Table 1, 3]
> - **可借鉴**: (1) DM/CDE 解耦思路可用于任何需要实时决策但核心推理昂贵的系统; (2) 合成全双工训练数据的 Algorithm 1 (topic+style+交互类型概率控制) 可复用; (3) 4-token 设计将复杂交互简化为分类问题
> - **局限**: 仅在合成数据上评估 barge-in; 实录实验仅覆盖 user state detection; 仅支持中文; 无延迟分析; 与 baseline 的对比使用不同测试集

## 核心问题

本文要解决的核心问题是: **如何在全双工语音对话系统中,以低计算开销实现精准的轮次管理?**

具体而言,全双工 SDS 需要处理三个难题 [§2.1]:
1. **Interfering speakers**: 背景说话人可能导致 ASR 和 DM 误判
2. **User pauses & hesitations**: 沉默不等于发言结束,可能导致过早响应或不必要延迟
3. **Unintentional interruptions**: backchannel ("I see")、对他人的说话等不应中断系统响应

现有方案要么用浅层特征 (acoustic VAD) 无法捕捉语义,要么将 DM 嵌入 CDE (大模型) 导致推理开销大且不可独立优化。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由 6 个模块串行组成 [§2.2, Fig 1(a)]:

1. **AEC** (Acoustic Echo Cancellation): 消除系统自身播放音频对麦克风的干扰
2. **Acoustic VAD**: speaker-aware 和 distance-aware,利用说话人嵌入和距离信息隔离目标说话人 [§2.2]
3. **ASR**: 将语音转为文本
4. **Semantic VAD** (本文核心): 0.5B LLM 作为 DM,基于 ASR 文本和历史响应预测控制 token
5. **CDE** (Core Dialogue Engine): 较大的 LLM,仅在 DM 预测 <|S-S|> 时被激活生成响应
6. **TTS**: 将响应文本合成为语音

前 4 个模块以短间隔运行确保实时性,CDE 仅按需激活以降低计算开销 [论文原文] [§2.2]。

### 关键设计选择

**为什么将 DM 与 CDE 解耦?**

论文明确指出 [§2.2, §1]: 将 DM 集成到 CDE 中 (如 NeuralFSM, VITA, Moshi) 虽然提高了自动化程度,但增加了推理开销。DM 需要以高频率运行 (每个短间隔都要决策),而 CDE 只需在确认用户完成提问后运行一次。解耦使得 (1) DM 用轻量模型处理高频决策,(2) CDE 保持高质量响应,(3) 两者可独立优化。[论文原文]

[agent 解读] 这一设计本质上是计算资源的分层分配: 高频低复杂度决策用小模型,低频高复杂度生成用大模型。类似于操作系统中内核态/用户态的分工。

**为什么用 4 个 control tokens 而非 2 个或 3 个?**

4 个 token 覆盖了 2 个维度 x 2 个任务 [§3.1, Fig 1(b)]:
- **User State Detection** (用户是否说完): <|C-L|> (未说完, 继续听) vs <|S-S|> (说完, 开始说)
- **User Intention Analysis** (barge-in 是否需要响应): <|S-L|> (real INT, 停止说话转监听) vs <|C-S|> (fake INT, 继续说话)

[agent 解读] 与 Freeze-Omni 的 State 0/1/2 对比: Freeze-Omni 合并了"继续听"和"假打断"为同一概念 (State 0 = 继续听),本文显式区分了 <|C-L|> (用户侧沉默时的等待) 和 <|C-S|> (用户侧有声音但是假打断),语义更精确。但 Freeze-Omni 多了 State 1 (backchannel feedback) 这个维度,本文不支持系统主动发出回传信号。

**为什么基于文本 (ASR 输出) 而非直接基于语音?**

论文将 Acoustic VAD 和 Semantic VAD 分工: acoustic VAD 用声学信号处理说话人干扰和基础活动检测,semantic VAD 用语义信息处理意图判断 [§2.2]。[论文原文]

[agent 解读] 这是 text-based integration 路线的选择。优势是 LLM 直接处理文本比处理语音更高效 (不需要 speech encoder/adapter),且 0.5B 模型在文本理解上已相当强。代价是丢失了副语言信息 (语调、犹豫的声学特征等),且受 ASR 错误影响 — 论文在 §4.3 也承认部分误判源于 ASR 错误。

### 训练策略

**数据生成** [§3.3, §3.4, Algorithm 1]:

由于不存在公开的带 control token 标注的全双工对话数据集,论文设计了一个两步数据生成方案:

1. **直接生成**: 用 20,000 个 prompt 通过 Yuanbao API 生成全双工对话。每个 prompt 指定: topic (200 主题池), speaking style (10 人设), QA 轮数 (2-12 轮), 交互类型概率 (P_real, P_fake, P_incomplete)。但 LLM 的 command-following 能力有限,仅 60% 数据符合预期格式。[论文原文]

2. **后处理增强**: 先生成标准 QA 对话 (LLM 更擅长),再通过受控后处理引入各种交互模式。[论文原文]

最终数据集: 11,990 对话, 80,338 轮。分布: Normal 47%, Fake INT 21%, Incomplete Q 19%, Real INT 13% [Fig 1(e)]。

[agent 解读] 数据的 60% 保留率表明直接让 LLM 生成带控制 token 的复杂格式有明显局限。两步策略 (先生成简单格式 + 后处理) 是实用的 workaround。

**模型训练** [§3.4]:
- 基座: Hunyuan 0.5B-dense-8k
- 词表: 扩展 4 个 control token
- 1500 steps, batch 128, lr 0.001 → 0.0001 (线性衰减)
- 为防止基座能力退化,训练集同时包含对应的无打断对话

**数据平衡** [§3.4]: 实验发现高比例 (>50%) 或全部为 real/fake INT 或 incomplete query 的极端场景导致过拟合和性能退化。[论文原文]

## 实验

### 合成测试集评估 [§4.1, §4.2]

| 指标 | <\|C-L\|> | <\|S-S\|> | <\|S-L\|> | <\|C-S\|> | 出处 |
| --- | --- | --- | --- | --- | --- |
| Recall | 0.926 | 0.989 | 0.999 | 1.000 | [Table 1] |
| Precision | 0.987 | 0.930 | 1.000 | 1.000 | [Table 1] |
| F1 | 0.956 | 0.959 | 0.999 | 1.000 | [Table 1] |

Overall accuracy: **97.85%** [Table 1]

**分析**: barge-in 检测 (<|S-L|>, <|C-S|>) 近乎完美 (F1 ≥ 0.999),因为同时说话时有丰富的上下文线索。User state detection (<|C-L|> vs <|S-S|>) 略低,因为仅依赖语义完整性判断,受说话风格和语言细微差异影响。[论文原文] [§4.2]

### 与相关工作对比 [§4.2, Table 2]

| 任务 | DuplexConv [18] F1 | RTTL-DG [23] F1 | 本文 F1 | 出处 |
| --- | --- | --- | --- | --- |
| User state (C-L) | 0.91 | 0.85 | 0.96 | [Table 2] |
| User state (S-S) | 0.89 | 0.52 | 0.96 | [Table 2] |
| Intention (S-L) | / | 0.62 | 0.96 | [Table 2] |
| Intention (C-S) | / | 0.95 | 1.00 | [Table 2] |

注意: 对比使用的是各自论文报告的数字,测试集不同,因此仅供参考。[论文原文] [§4.2]

### 实录数据评估 [§4.3, Table 3]

使用内部半双工 SDS 录音,仅评估 user state detection (C-L vs S-S):

| 方案 | Threshold | Accuracy | 出处 |
| --- | --- | --- | --- |
| Acoustic VAD only | 300ms | N/A (只能预测 S-S) | [Table 3] |
| Acoustic VAD only | 500ms | N/A | [Table 3] |
| + Semantic VAD | 300ms | 0.935 | [Table 3] |
| + Semantic VAD | 500ms | 0.962 | [Table 3] |
| + Semantic VAD | 800ms | 0.966 | [Table 3] |
| + Semantic VAD | 1800ms | 0.971 | [Table 3] |

Acoustic VAD 仅靠静音时长阈值只能判断 <|S-S|>,加入 semantic VAD 后 accuracy 均超 93.5%。误判部分来自 ASR 错误而非 VAD 本身。[论文原文] [§4.3]

## 局限性

1. **合成数据评估为主**: barge-in 检测 (S-L/C-S 的近乎完美 F1) 仅在合成测试集上验证,合成数据可能不充分反映真实场景的复杂性 (噪声、口音、方言)
2. **实录评估覆盖不完整**: 实录实验仅涉及 user state detection,未覆盖 barge-in 场景的真实表现
3. **无延迟报告**: 未提供 DM 决策延迟、端到端响应延迟等关键时间指标
4. **仅中文**: 数据集和实验均为中文,跨语言泛化能力未知
5. **ASR 依赖**: 语义 VAD 依赖 ASR 文本输入,ASR 错误直接影响 DM 准确率 (论文在 §4.3 承认)
6. **不支持系统侧主动行为**: 与 Freeze-Omni (State 1 = backchannel feedback) 或 Raon-SpeechChat (BC token) 不同,本文 DM 不能让系统主动发出回传信号
7. **对比公平性**: 与 DuplexConv/RTTL-DG 的对比使用不同测试集,无法确保公平
8. **无开源模型**: 仅开源了数据准备脚本,模型和完整数据集未公开

## 点评

**优点**:
- DM/CDE 解耦是实用的工程设计,符合工业部署需求: 高频决策用小模型降本,核心推理用大模型保质
- 4-token 设计将复杂的全双工交互行为清晰形式化,易于理解和扩展
- 数据生成的 Algorithm 1 设计细致 (topic/style 池 + 交互类型概率 + 极端场景发现),方法论可复用

**不足**:
- 论文的评估框架偏弱: 最强的结果 (barge-in F1 ≈ 1.0) 在合成数据上,最弱但最有说服力的实录实验仅覆盖子集任务
- 将全双工能力完全建立在 ASR 文本之上,丢失了大量声学线索 (语调、犹豫的声学模式、情感),这在 2025 年端到端方案 (Moshi, Freeze-Omni, LSLM) 已经展示了直接从语音信号建模的优势背景下显得保守
- 与端到端方案的关键 trade-off 未充分讨论: DM/CDE 解耦节省了多少计算? 增加了多少延迟? 这些数字的缺失使得核心论点 (效率 vs 质量平衡) 缺乏量化支撑

**定位**: 本文代表了一种"务实的模块化"路线 — 不追求端到端的优雅,而是用工程解耦换取可控性和部署灵活性。在工业场景 (需要稳定性、可独立迭代各模块) 中有实际价值,但学术贡献有限。

## 可复用的 idea

1. **DM/CDE 解耦模式**: 在任何需要"高频实时决策 + 低频重计算"的系统中可借鉴。例如 TTS 系统中用小模型做实时韵律决策,大模型做内容生成
2. **合成全双工数据的 Algorithm 1**: topic pool + style pool + 交互类型概率控制 + 极端场景测试的数据生成范式可直接复用于其他全双工/交互数据需求
3. **4-token 轮次控制形式化**: 将轮次管理简化为 {C-L, S-S, S-L, C-S} 四分类,清晰可扩展 (如加入 backchannel token)
4. **数据清洗策略**: LLM 生成复杂格式数据时 60% 保留率的经验 + 两步策略 (先生成简单格式再后处理) 对任何 LLM-generated 训练数据项目有参考价值

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法机制清晰,DM/CDE 解耦的 WHY 和 4-token 设计的 HOW 均有解释 |
> | 可信赖 | pass | 关键数字均标注出处,指标使用正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖充分 |
> | 可定位 | pass | KB 背景提供了与 Freeze-Omni/CleanS2S/LSLM 等的具体对比定位 |
> | 不污染 | pass | 无新建概念页需求,反向更新为追加 |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> 详见 `_review/LLM-EnhancedFull-Duplex-review.yml`
