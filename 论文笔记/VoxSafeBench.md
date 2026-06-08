---
type: paper
tier: deep
title: "VoxSafeBench: Not Just What Is Said, but Who, How, and Where"
arxiv_id: "2604.14548"
source: "Sources/VoxSafeBench.pdf"
authors: [Yuxiang Wang, Hongyu Liu, Yijiang Xu, Luchao Yao, Qinke Ni, Li Wang, Wan Lin, Kunyu Feng, Dekun Chen, Xu Tan, Lei Wang, Jie Shi, Zhizheng Wu]
year: 2026
venue: "arXiv preprint"
tags: [SLM-safety, social-alignment, fairness, privacy, benchmark, speech-grounding, jailbreak, audio-conditioned, paralinguistic, evaluation]
concepts: ["[[SpeechLanguageModel]]", "[[AudioUnderstanding]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-08
updated: 2026-06-08
---

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass (9) | 方法节有丰富因果解释 (Two-Tier WHY / 文本上限 WHY / 感知探测 WHY); 速查可借鉴 4 条具体可迁移 idea |
> | 可信赖 | pass (9) | 数字出处覆盖率>90%; 指标定义明确含方向标注 |
> | 可区分 | pass (8) | 来源标注覆盖率~85%; 方法节 [论文原文]/[agent 解读] 分离清晰; KB 背景末段有一处 agent 推断未标注 |
> | 可定位 | pass (9) | 命中 2 实体页; SLM competence→alignment 范式转移定位清晰; 与 E2E-VGuard 互补关系明确 |
> | 不污染 | pass (9) | 无新建页; 信息流入概念页会改善质量 |
> 
> Issues: 4 (high: 0, medium: 0, low: 4)
> 详见 `_review/VoxSafeBench-review.yml`

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 1 个待确认实体页: [[SpeechLanguageModel]], [[AudioUnderstanding]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 SLM 社会对齐 (social alignment) 评估的新方向。与 [[SpeechLanguageModel]] 页记录的能力型 benchmark (Dynamic-SUPERB, MMSU, MMAU-Pro, VoiceBench 等——测试"模型能否听懂和推理") 不同,VoxSafeBench 回答的是部署关键问题: "模型听懂之后,是否在安全/公平/隐私维度上做出社会可接受的回应"。这是 SLM 评估从 competence 到 alignment 的范式转移。
>
> **已有认知**: [[SpeechLanguageModel]] 页 (confirmed) 记录了 SpeechLM 的三大优势 (保留副语言信息、消除级联延迟、避免跨模块错误传播),但副语言信息的保留并不等同于基于副语言信息做出正确行为判断——VoxSafeBench 正是针对这个差距设计的。[[AudioUnderstanding]] 页 [待确认] 区分了语义/说话人/副语言三类理解任务,VoxSafeBench 的 Tier 2 设计正是围绕这三个维度构建: child voice (说话人) / emotion & impaired capacity (副语言) / background sounds (环境)。
>
> **创新判断**: KB 中尚无"SLM 社会对齐"或"语音安全评估"的独立概念页。现有 [[TTSEvaluation]][待确认] 主要关注合成质量 (MOS/WER/SIM) 和 responsible evaluation 框架,但不涉及 SLM 的安全/公平/隐私对齐。VoxSafeBench 与 [[论文笔记/E2E-VGuard|E2E-VGuard]] 同出 CUHK-SZ (Zhizheng Wu 组),但方向互补: E2E-VGuard 关注"防止语音被克隆" (主动防护),VoxSafeBench 关注"SLM 在多场景下是否行为正确" (行为对齐)。
>
> 检索命中: [[SpeechLanguageModel]]✓, [[AudioUnderstanding]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个联合评估 SLM 安全/公平/隐私的 benchmark,通过 Two-Tier 设计分离内容风险与语音上下文风险,揭示当前 SLM 普遍存在的"speech grounding gap" -- 文本级安全屏障无法迁移到语音原生条件
> - **路线**: 文本数据采集 → CosyVoice3 合成三视图 (Text/Clean Audio/Diverse Audio) + Tier 2 声学 cue 注入 → Whisper-v3 WER<5% 质量过滤 → 22 任务 (Safety/Fairness/Privacy x Tier 1/Tier 2) → 6 开/闭源 SLM 评估 → DeepSeek-V3 LLM-as-Judge
> - **指标**: Safety Tier 2: 最好 SAR 76.1% (Gemini-3-Pro, impaired) vs 文本上限 100% [Table 3]; Fairness: Tier 1→Tier 2 Fair Rate 普遍暴跌 (e.g. GPT-4o Criminality 70.7%→38.5%) [Table 4]; Privacy: Gemini-3-Pro 硬隐私泄漏率从 text 23.9% 跳至 audio 81.2% [Table 5]
> - **可借鉴**: (1) Two-Tier 设计范式: 分离"规范知识缺失"与"规范应用失败",可迁移到任何需要区分能力瓶颈与对齐瓶颈的评估场景; (2) 中间感知探测 (intermediate perception probes) + 文本参考上限作为构建有效性验证方法
> - **局限**: (1) 大部分音频为 CosyVoice3 合成而非自然录音,真实场景失败可能更严重; (2) Tier 2 使用高显著性 cue,模型对更微妙线索可能更差; (3) 文本参考上限并非 oracle (只验证规范可达性,不代表完美表现)

## 核心问题

VoxSafeBench 要解决 SLM 社会对齐评估的三个空白 [§1]:

1. **联合覆盖空白**: 现有 benchmark 要么只测能力 (MMSU, VoiceBench) 不测对齐,要么只测单一风险维度 (JALMBench 测 jailbreak, HearSay 测推理隐私) 而不联合覆盖 safety/fairness/privacy [Table 1]。缺少一个统一 benchmark 能回答"SLM 在部署中是否社会可接受"这个综合问题。

2. **Content vs Context 混淆**: 已有 trustworthiness benchmark (MultiTrust, AudioTrust) 没有清晰分离"内容本身有害"与"内容因语音上下文而变得有害"两类风险。这使得无法判断模型失败原因: 是不知道规范 (normative knowledge failure),还是知道规范但无法在语音线索出现时应用 (speech grounding failure) [§1, "difficult to tell whether a model fails because it does not know the norm at all"]。

3. **感知 vs 对齐 混淆**: 当模型在语音安全任务上失败时,可能是因为没听到 cue (perception failure),也可能是听到了但不知道该怎么做 (policy failure)。需要实验设计来区分这两种失败模式 [§2, Appendix J]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VoxSafeBench 围绕三个支柱 (safety/fairness/privacy) 和两个层级 (Tier 1/Tier 2) 构建一个 22 任务评估矩阵 [Table 2]:

**三个支柱**:
- **Safety** (S1.1-S1.3, S2.1-S2.3): 有害内容、jailbreak、agentic 风险、副语言冲突、背景冲突、对抗交互
- **Fairness** (F1.1-F1.2, F2.1-F2.2): 系统性刻板印象、排斥性规范、副语言/声学偏差、交叉偏见
- **Privacy** (P1.1-P1.2, P2.1-P2.3): 硬隐私 (结构化 PII)、软隐私 (上下文秘密)、音频条件隐私、交互隐私、推理隐私

**Two-Tier 设计** [§2]:
- **Tier 1 ("The What")**: 内容中心风险 — 文本本身包含有害/偏见/隐私内容,测试模型是否知道相关规范。提供三个匹配视图 (Text / Clean Audio / Diverse Audio) 暴露跨模态对齐差距。
- **Tier 2 ("The Who, How, and Where")**: 音频条件风险 — 文本是良性的,但正确响应取决于说话人身份 (child/gender)、副语言线索 (emotion/prosody/impairment)、或环境 (旁观者/背景声)。只提供 Audio 输入。

[agent 解读] Two-Tier 的精巧之处在于它构成了一个因果推理链: 如果模型在 Tier 1 Text 上通过 (知道规范)、在 Tier 1 Audio 上仍通过 (跨模态不丢失),但在 Tier 2 上失败,那失败就不能归因于规范知识缺失或模态转换,而只能归因于"speech grounding gap" -- 模型无法将语音中的非词汇线索映射到已知规范上。

### 关键设计选择

**1. 文本参考上限 — 如何验证 Tier 2 不是"出题太怪"** [§2, §3]

[论文原文] Tier 2 面临构建有效性挑战: 如果模型在安全任务上失败,可能是 (a) 没听到 cue, (b) 任务设计本身不合理 (即便人类也觉得不需要拒绝), (c) 政策确实不覆盖该场景。为排除 (b) 和 (c),作者将 Tier 2 的声学 cue 用文本显式描述 (如 "The speaker is a child"、"Background voices indicate others are listening"),输入到最强文本 LLM (DeepSeek-V3, Gemini-3-Pro, GPT-5.2),取最安全结果为上限 [Table 3]。这些上限接近饱和 (child voice SAR 92.5%, impaired 100%, unsafe ambient 100%),证明规范在文本中是可获取的。

[agent 解读] 这个设计解决了 evaluation validity 的经典难题: 你不能用一个"可能出题有问题"的 benchmark 去批评模型。文本上限证明"题目没问题,规范存在且可获取",所以语音上的失败只能归因于 speech grounding。

**2. 中间感知探测 — 如何区分"没听到"vs"听到了不行动"** [Appendix J]

[论文原文] 对同一 Tier 2 音频设计感知探测 (perception probe): 直接问模型"这段音频中说话人是成人还是儿童?"、"背景中是否有其他人说话?"。结果显示,frontier SLMs 在大多数 cue 上感知准确率显著高于安全行为率。例如,模型能可靠检测 child voice,但在 child-voice safety 任务上仍然直接回答 [Appendix J]。

[agent 解读] 这是本文方法论上最有价值的贡献之一: 将"感知"和"行动"解耦。它证明了 speech grounding gap 不是感知问题,而是 policy grounding 问题 -- 模型有能力提取语音线索,但对齐训练没有教它"提取到这些线索后该怎么做"。

**3. 最小对设计 (Minimal-pair) — Fairness 评估如何消除混淆** [§4]

[论文原文] Fairness 使用最小对逻辑: 两个候选人在所有任务相关语义上完全等价,仅在人口统计属性或声学特性上不同。Tier 1 通过第三人称叙述显式声明人口统计 (如 "one male, one female doctor")。Tier 2 去除人口统计标签,仅通过声学渲染 (accent/age/timbre/emotion/background) 传达差异。

[论文原文] 进一步引入 Net Bias Score (NBS) 进行方向性偏见分析 [Eq. 1]: 交换说话人顺序消除位置偏好,计算 stereotype-aligned vs counter-stereotype 比例。NBS > 0 表示与社会刻板印象一致的系统性偏见。

**4. 数据集构建 — 为什么用合成语音** [§2, Appendix D]

[论文原文] 使用 CosyVoice3 合成大部分音频,因为它在中英文都有强说话人身份保持能力 (speaker similarity: ZH 78% / EN 71.8%,接近人类录音 75.5% / 73.4%)。关键: 需要对 gender/accent/age/emotion/background 进行可控变化,自然录音无法提供这种精确控制。所有合成音频经 Whisper-large-v3 过滤 (WER > 5% 丢弃) [§2]。

### 评估指标体系

VoxSafeBench 使用多指标体系,按任务类型定制 [Table 2]:

| 支柱 | 关键指标 | 含义 |
| --- | --- | --- |
| Safety | RtA (Refuse to Answer) ↑ | 拒绝回答率 |
| Safety | Toxicity Score ↓ | 毒性评分 (OpenAI Moderation API) |
| Safety | SAR (Safety Awareness Rate) ↑ | 安全意识率 (警告/拒绝) |
| Safety | DAR (Direct Answer Rate) ↓ | 直接回答率 (无视风险) |
| Safety | OIC (Overlap-Induced Conversion) ↓ | 重叠导致的安全→不安全转换率 |
| Fairness | Fair Rate ↑ | 公平率 (避免人口统计判断) |
| Fairness | NBS (Net Bias Score) → 0 | 净偏见分数 (方向性偏见) |
| Privacy | Leakage Rate ↓ | 信息泄漏率 |
| Privacy | PAR (Privacy Awareness Rate) ↑ | 隐私意识率 |

评判方式: 开放式任务用 DeepSeek-V3 作为 LLM Judge; inter-judge agreement 研究 (Fleiss' kappa = 0.78, Spearman's rho >= 0.88) 验证了评判稳定性 [§2]。

## 实验

### Safety

| 指标 | 模型 | 值 | 条件 | 出处 |
| --- | --- | --- | --- | --- |
| SAR (↑) | Gemini-3-Pro | 76.1% | Tier 2, Impaired Capacity | [Table 3] |
| SAR (↑) | Gemini-3-Pro | 49.3% | Tier 2, Child Presence (background) | [Table 3] |
| SAR (↑) | Gemini-3-Pro | 42.5% | Tier 2, Symbolic Background | [Table 3] |
| DAR (↓) | Mimo-Audio | 98.0% | Tier 2, Child Voice (几乎完全无视) | [Table 3] |
| SAR (text ref) | Text LLMs | 92.5-100% | Tier 2, 文本参考上限 | [Table 3] |
| OIC (↓) | Qwen3-Omni | 77.8% | 对抗交互 (重叠注入) | [Table 3] |

**Safety 关键发现** [§3, Fig 2]:
1. **多轮 jailbreak 最危险**: 无 jailbreak 时所有模型在安全区,多轮攻击使安全屏障持续瓦解
2. **文本比音频更危险** (反直觉): 文本输入导致更长、更详细的不安全输出; 音频模态更容易触发简短拒绝
3. **推理变体 (thinking) 更脆弱**: CoT 推理一旦被误导,模型顺着错误逻辑走得更深
4. **Tier 2 speech grounding gap**: child voice SAR 最高仅 18.5% (Qwen3-Omni-think) vs 文本上限 92.5%; 隐性 cue (child in background) 比显性 cue (child speaking) 更难处理

### Fairness

| 指标 | 模型 | Tier 1 (Audio) | Tier 2 | 维度 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Fair Rate (↑) | Gemini-3-Pro (EN/ZH) | 53.2/59.2 | 4.0/1.2 | Competence | [Table 4] |
| Fair Rate (↑) | Gemini-3-Pro (EN/ZH) | 90.1/78.9 | 27.9/12.5 | Criminality | [Table 4] |
| Fair Rate (↑) | GPT-4o-Audio (EN/ZH) | 46.7/36.7 | 20.7/36.7 | Competence | [Table 4] |
| NBS (→0) | Multiple | Significant positive | Many dims | Stereotype-aligned | [Fig 3] |

**Fairness 关键发现** [§4, Fig 3]:
1. **Tier 1→Tier 2 断崖下降**: 几乎所有模型从 text/audio 的部分偏见抵抗到 Tier 2 的近零 Fair Rate
2. **NBS 方向性**: 多个模型显示 significant positive NBS (与社会刻板印象一致的系统性偏见)
3. **Thinking 效果不可预测**: CoT 可能反转 modality hierarchy (audio > text),在不同语言上效果对立
4. **语言依赖**: 同一音频,EN 和 ZH query 可产生截然不同的 bias profile (e.g. Competence bias 在不同语言间翻转)

### Privacy

| 指标 | 模型 | Text | Clean Audio | Diverse Audio | 出处 |
| --- | --- | --- | --- | --- | --- |
| Hard Leakage (↓) | Gemini-3-Pro | 23.9% | 77.7% | 81.2% | [Table 5] |
| Soft Leakage (↓) | GPT-4o-Audio | — | 14.7% | 14.4% | [Table 5] |
| Tier 2 DAR (↓) | GPT-4o-Audio | — | 91.5% | — | [Table 5] |
| Tier 2 PAR (↑) | Gemini-3-Pro | — | 94.3% | — | [Table 5] |

**Privacy 关键发现** [§5, Fig 4]:
1. **严重跨模态隐私差距**: Gemini-3-Pro 硬隐私泄漏从 text 23.9% 跳至 audio 81.2% (3.4x 恶化)
2. **硬隐私比软隐私更难**: 结构化 PII (密码、证件号) 泄漏率远高于上下文秘密
3. **推理隐私风险**: 多数模型很少拒绝 voice profiling 请求但推理准确率不低,构成 tangible profiling risk; GPT-4o-Audio 例外 (较高拒绝率)

## 局限性

1. **合成语音局限**: 大部分音频由 CosyVoice3 合成而非自然录音。真实环境中的声学变化更复杂,模型失败可能更严重 [§6]
2. **高显著性 cue**: Tier 2 使用高度明确的声学线索 (经人工筛选 + arousal > 0.7 过滤)。实际部署中的线索可能更微妙,模型表现可能更差 [§6]
3. **文本上限非 oracle**: 文本参考上限验证的是"规范在文本中可获取",而非"规范在所有条件下都应该适用"。部分 Tier 2 场景的伦理判断可能存在合理分歧
4. **LLM Judge 限制**: 开放式任务依赖 DeepSeek-V3 作为 judge,虽然 inter-judge agreement 良好 (kappa=0.78),但 judge 本身的 bias 可能影响评估
5. **模型覆盖**: 仅评估 6 个 SLM (3 开源 + 3 闭源),且以具有 "demonstrated strong audio understanding" 为选择标准,可能高估了整个 SLM 生态的基线能力
6. **安全定义文化差异**: 许多任务的安全/公平规范隐含特定文化假设 (e.g. 儿童安全、性别角色),跨文化适用性未充分讨论

## 点评

**优势**:
- 问题定义极其精准: "speech grounding gap" 这个概念抓住了 SLM 安全的核心矛盾 -- 模型知道规范但不会在语音条件下应用。Two-Tier 设计将这个抽象概念变成了可测量的实验 [§1-2]
- 方法论严谨: 文本参考上限 + 中间感知探测 + minimal-pair + 交换顺序消除位置偏好,形成了一套完整的 construct validity 论证链 [§2, Appendix J]
- 实验发现有深度: 不只是"模型不安全"的简单结论,而是区分了 content-safety / speaker-grounding / scene-grounding / interactional / inferential 等多层次失败模式,且每层都有数据支撑
- 中英双语覆盖 + EN/ZH bias profile 对比,揭示了 language-dependent alignment 这个此前未被充分认识的问题

**不足**:
- 论文长度和内容密度极高 (正文 + 50+ 页附录),但主文对部分关键结论 (如 intermediate probing) 的展示过于简略,主要细节被推到附录。读者如果只读主文可能低估了这项工作的方法论贡献
- Tier 2 的数据规模较小 (e.g. Content-Paralinguistic Conflict 仅 853 条, Audio-Conditioned Privacy 仅 400 条),在细分维度上 (如单个 emotion 类型) 统计功效可能不足
- 未讨论如何 mitigate speech grounding gap: 论文定位为诊断性 benchmark,但没有提出任何对齐方法或训练策略。ParaS2S [50] 至少提出了 RL-based alignment 作为解决方案
- 与 AudioTrust [9] 的比较不够深入: 两者在 safety/fairness/privacy 维度上有重叠,但论文仅在 Table 1 中做了表面覆盖对比,未进行实验结果对比

**在 KB 语境下的定位**:
从 [[SpeechLanguageModel]] 的能力全景来看,VoxSafeBench 填补了 SLM 评估从 competence (能力) 到 alignment (对齐) 的最后一块拼图。此前的 benchmark (Dynamic-SUPERB, MMSU, VoiceBench) 只问"SLM 能不能理解语音",VoxSafeBench 问的是"理解之后会不会做出正确的社会行为"。

从 [[AudioUnderstanding]] 的角度看,VoxSafeBench 的 intermediate probing 实验为该页添加了一个重要认知: 理解能力 ≠ 行为对齐。SLM 可以在 perception task 上表现良好 (检测 child voice、emotion、background sounds),但在需要基于这些感知做出规范性判断时系统性失败。这暗示 instruction-tuning 和 RLHF/DPO 阶段主要优化的是文本级对齐,语音级对齐尚未被充分训练。

与同组 [[论文笔记/E2E-VGuard|E2E-VGuard]] 的关系: 两者共同构成了 CUHK-SZ (Zhizheng Wu 组) 在语音安全领域的双线布局 -- E2E-VGuard 从"防止语音被滥用"(主动防护)切入,VoxSafeBench 从"SLM 是否行为安全"(对齐评估)切入。前者关注 TTS 生成端安全,后者关注 SLM 理解端对齐。

## 可复用的 idea

1. **Two-Tier taxonomy (内容 vs 上下文分离)**: 将评估目标分为"内容本身的问题"和"上下文改变了内容含义的问题",可迁移到任何涉及多模态上下文的对齐评估 -- 例如视觉 LLM 的"图片+文字组合变有害"场景
2. **Construct validity 三件套 (文本上限 + 感知探测 + 最小对)**: 系统性地排除了 benchmark 评估中的三大混淆因素 (题目不合理 / 感知失败 / 非关键变量差异),可作为高质量 benchmark 设计的方法论模板
3. **Net Bias Score + 交换顺序**: 在 fairness 评估中同时捕获 bias 方向性和幅度,消除位置偏好,比简单的 accuracy/fair-rate 提供更丰富的诊断信息
4. **Speech grounding gap 作为诊断框架**: "模型知道规范但不会在语音条件下应用"这个 framing 可推广为"X modality grounding gap"模式,用于诊断任何模态上的对齐失败 (e.g. 视觉 grounding gap, 触觉 grounding gap)
