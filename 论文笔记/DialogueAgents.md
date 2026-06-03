---
type: paper
tier: deep
title: "DialogueAgents: A Hybrid Agent-Based Speech Synthesis Framework for Multi-Party Dialogue"
arxiv_id: "2504.14482"
source: "Sources/DialogueAgents.pdf"
authors: [Xiang Li, Duyi Pan, Hongru Xiao, Jiale Han, Jing Tang, Jiabao Ma, Wei Wang, Bo Cheng]
year: 2025
venue: "arXiv"
tags: [TTS, dialogue-synthesis, multi-agent, multi-party, emotional-TTS, data-generation, zero-shot, evaluation]
concepts: ["[[Emotion Control in TTS]]", "[[Prosody Modeling]]", "[[Turn-taking in Spoken Dialogue]]", "[[TTS Evaluation]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[模型库/CosyVoice|CosyVoice]]✓, [[Prosody Modeling]]✓, [[Zero-shot Speech Synthesis]]✓, [[Emotion Control in TTS]][待确认], [[TTS Evaluation]][待确认], [[Turn-taking in Spoken Dialogue]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: DialogueAgents 处于 dialogue speech synthesis 与 multi-agent collaboration 的交叉点。它不提出新的 TTS 模型,而是将已有的零样本 TTS 系统 (CosyVoice) 嵌入多 agent 协作框架,通过迭代脚本优化来提升对话语音的情感和韵律质量。这与当前 TTS 领域"单模型端到端"的主流范式形成差异化路线。
>
> **已有认知**:
> - CosyVoice 是阿里巴巴提出的 LLM+OT-CFM 零样本 TTS 系统,支持 instruct 模式下的 style/paralinguistics 控制 [CosyVoice-instruct 变体],本文用其作为语音合成 agent
> - Emotion Control in TTS 领域已有丰富方法 (embedding/层级建模/DPO/副语言发声),本文用 agent 迭代而非模型结构改进来增强情感表达
> - Turn-taking 研究已从 VAD 发展到端到端建模 (Moshi/dGSLM),本文提出的 TMOS 指标关注对话轮次切换自然度,但未涉及端到端 turn-taking 建模
> - TTS Evaluation 领域正从 MOS 走向 distributional/LLM-as-Judge,本文仍使用传统 MOS + UTMOS + WER/CER
>
> **创新判断**: 本文的创新在于框架设计而非模型/算法。将 LLM (GPT-4o) 作为脚本生成器、CosyVoice 作为合成器、Qwen2-Audio 作为多模态语音评审组成闭环迭代,这种"生成-合成-评审-修改"的循环是本文独特贡献。但框架依赖闭源模型 (GPT-4o),复现受限。
>
> 检索命中: [[模型库/CosyVoice|CosyVoice]]✓, [[Prosody Modeling]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[TTS Evaluation]](pending-review), [[Turn-taking in Spoken Dialogue]](pending-review) | 未命中但可能相关: [[Spoken Dialogue Evaluation]]

## 速查

> [!summary] 速查
> - **一句话**: 用三个 agent (GPT-4o 写剧本 + CosyVoice 合成语音 + Qwen2-Audio 评审) 迭代生成多方多轮对话语音,并贡献双语数据集 MultiTalk
> - **路线**: Character Pool → Script Writer (GPT-4o) → Speech Synthesizer (CosyVoice) → Dialogue Critic (Qwen2-Audio) → Feedback → Script Writer (迭代 T 轮) → 最终对话语音 + 脚本
> - **指标**: 2 loop 最优 — MOS 3.75, EMOS 3.96, TMOS 3.78 [Table II]; 脚本 Emotiveness 3.96 [Table III]; WER(EN) 4.07%, CER(CN) 5.16% [Table II]; MultiTalk 4437 utterances / 32441s [Table V]
> - **可借鉴**: (1) 用多模态 LLM (Qwen2-Audio) 作为语音评审 agent 提供结构化反馈的思路; (2) EMOS/TMOS 两个对话级语音评估指标的定义; (3) Character Pool 设计 (角色描述+社交关系) 增加对话多样性
> - **局限**: 依赖 GPT-4o (闭源/成本高); CosyVoice 在长对话和多说话人交互中仍不稳定 (论文自述); 3 loop 过优化导致质量下降; MultiTalk 数据量偏小 (仅 ~9h); 对话级 MOS 仍未超过 4.0

## 核心问题

本文要解决的问题是: **现有对话语音合成数据集成本高、角色多样性不足、情感表达有限** [§I]。具体痛点包括:

1. 现有数据集 (DailyTalk, Fisher) 主要靠人工转录或录制,构建成本高 [§I]
2. 大多数数据集仅包含两个说话人的对话,缺乏多方参与 [Table I]
3. 情感表达和副语言特征 (停顿、强调、语气) 不充分 [§I]
4. KE-Omni 等自动化方案采用顺序协作,累积误差导致效果受限 [§I]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DialogueAgents 由三个 agent 组成闭环迭代系统 [§III, Fig 1]:

1. **Script Writer Agent** ($A_w$): GPT-4o,负责基于角色池 ($Pool_p$) 生成对话脚本 $S = A_w(p_{A_w}, Pool_p)$,脚本包含说话人标识和发言内容 [§III.B]
2. **Speech Synthesizer Agent** ($A_s$): CosyVoice,基于脚本和角色参考音频 ($Pool_a$) 合成语音 $D = A_s(S, Pool_a)$ [§III.B]
3. **Dialogue Critic Agent** ($A_c$): Qwen2-Audio,从 naturalness 和 clarity/emotiveness 两个维度评审合成语音,生成反馈 $F = A_c(D, p_{A_c})$ [§III.B]

**迭代流程**: 在第 $t$ 轮迭代 ($t \in [1, T]$) 中,Script Writer 接收上一轮反馈 $F_{t-1}$ 和脚本 $S_{t-1}$,生成修订脚本 $S_t = A_w(p_{A_w}, Pool_p, S_{t-1}, F_{t-1})$。修订内容包括调整对话内容、添加副语言标记 (如 `<strong>`, `[breath]`) 和情感标签 (如 `[Agreeable]`, `[Curious]`) [§III.C]。

### 关键设计选择

**为什么用三个独立 agent 而非端到端模型?**
[论文原文] 框架允许灵活替换每个 agent 的具体模型,可利用各领域最强模型的能力 [§III.D, 第 3 点]。[agent 解读] 这种设计也使得文本侧优化 (GPT-4o) 和语音侧合成 (CosyVoice) 各自发挥所长,避免单一模型在两端都做妥协。

**为什么引入 Dialogue Critic 而非让 Script Writer 自行迭代?**
[论文原文] Self-refine (Writer 自己迭代) 虽然也能改善脚本,但引入语音层面的 critic 反馈能进一步提升合成质量 [§IV.B]。Critic 基于实际合成语音而非文本进行评审,能捕捉文本无法表达的问题 (如情感不匹配、语音不自然) [§III.C]。[agent 解读] 这是 "生成-评审" 分离的经典思路,critic 充当质量守门人,打破了纯文本空间优化的天花板。

**为什么 2 loops 最优而非更多?**
[论文原文] 3 loops 时对话质量开始轻微下降,原因是过优化引入的噪声和干扰 [§IV.B]。[agent 解读] 迭代次数与质量的非单调关系类似于 RL 中的过拟合问题,多轮修改可能累积不一致的指令信号。

**Character Pool 设计**:
从 WenetSpeech4TTS 和 Common Voice 收集 30 个独特角色,每个角色有参考音频 $a_i$ 和人设描述 $p_i$ (年龄、性别、性格、语言习惯),并定义角色间社交关系 (亲属、朋友、同事) [§IV.A]。[论文原文] 角色扮演方法更有利于生成连贯、真实的对话脚本 [§III.A]。

### 训练策略

本文是框架论文,不涉及模型训练。三个 agent 使用现成模型:
- Script Writer: GPT-4o (直接调用 API)
- Speech Synthesizer: CosyVoice (开源零样本 TTS)
- Dialogue Critic: Qwen2-Audio (开源多模态模型)

评估中的 TTS 模型训练: Tacotron2 和 FastSpeech2 训练 900K steps (A40 GPU, batch 16),DailyTalk-Model 训练至收敛 [§V.C]。

## 实验

### 框架有效性 (Table II, Table III)

| 指标 | Writer+Synth | Writer(self)+Synth | +Critic(1L) | +Critic(2L) | +Critic(3L) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS | 3.63 | 3.63 | 3.71 | 3.75 | **3.78** | [Table II] |
| EMOS | 3.64 | 3.62 | 3.79 | **3.96** | 3.91 | [Table II] |
| TMOS | 3.59 | 3.60 | 3.71 | **3.78** | 3.76 | [Table II] |
| UTMOS(EN) | 4.186 | 4.224 | 4.302 | **4.316** | 4.162 | [Table II] |
| UTMOS(CN) | 2.962 | 3.043 | 3.138 | **3.318** | 3.064 | [Table II] |
| WER(EN) | 4.22 | 4.13 | 4.14 | **4.07** | 4.34 | [Table II] |
| CER(CN) | 5.19 | 5.28 | 5.23 | **5.16** | 5.36 | [Table II] |
| Naturalness | 4.32 | 4.45 | 4.49 | **4.59** | 4.54 | [Table III] |
| Emotiveness | 2.69 | 3.42 | 3.67 | **3.96** | 3.66 | [Table III] |

### 消融: Critic 贡献 (Fig 2)

人工 A/B 测试: 经 critic 修改后的脚本在 75.8% 的情况下被认为整体更好 [§IV.B, Fig 2]。

### 数据集评估 (Table VI)

| 模型 | LJSpeech | DailyTalk | VCTK | MultiTalk | 出处 |
| --- | --- | --- | --- | --- | --- |
| Ground Truth | 3.79 | 3.82 | 3.78 | 3.76 | [Table VI] |
| Tacotron2 | 3.65 | 3.71 | 3.69 | 3.67 | [Table VI] |
| FastSpeech2 | 3.72 | 3.74 | 3.70 | 3.69 | [Table VI] |
| DailyTalk-Model | - | - | 3.67 | 3.66 | [Table VI] |

MultiTalk 训练出的模型 MOS 与 DailyTalk 等现有数据集相当,且 MultiTalk 仅用不到 DailyTalk 四分之一的数据量 [§V.C]。

### MultiTalk 数据集统计 (Table V)

| 特征 | CN | EN | 出处 |
| --- | --- | --- | --- |
| 语句数 | 1,950 | 2,487 | [Table V] |
| Token 数 | 57,737 | 43,036 | [Table V] |
| 总时长 (s) | 14,085.73 | 18,355.47 | [Table V] |
| 平均角色数 | 3.54 | 3.06 | [Table V] |
| 角色总数 | 15 | 15 | [Table V] |

## 局限性

1. **闭源依赖**: Script Writer 使用 GPT-4o,成本高且不可控,限制了学术复现和工业部署 [agent 解读]
2. **合成语音质量上限**: CosyVoice 在多说话人长对话中仍不稳定 [§II.A],框架质量受限于底层 TTS 模型能力
3. **过优化问题**: 3 轮迭代即出现质量下降 [§IV.B],表明 critic 反馈信号本身存在噪声
4. **数据集规模偏小**: MultiTalk 总计约 32K 秒 (~9 小时),远小于大规模 TTS 数据集 (如 Emilia 100K+ h),作为训练数据用途有限 [Table V]
5. **评估指标简单**: EMOS 和 TMOS 虽有创新,但仍基于 1-5 分 MOS 框架,未解决 MOS 的 ceiling effect 和不可比性问题 [agent 解读]
6. **无端到端 turn-taking 建模**: 对话轮次切换完全依赖脚本文本中的角色标注,不涉及真正的语音级 turn-taking 建模 [agent 解读]
7. **MOS 绝对值偏低**: 各配置 MOS 在 3.63-3.78 范围,EMOS 最高 3.96,均未突破 4.0 [Table II]

## 点评

**优点**:
- 框架思路清晰: 将对话语音合成分解为"写-合-评"三个可独立优化的环节,每个环节选用当前最强模型,体现了模块化系统设计的优势
- Critic agent 引入语音层面反馈的思路有价值: 文本空间的优化无法捕捉语音合成的实际问题,跨模态反馈是合理的设计
- 副语言标记注入 (如 `<strong>`, `[breath]`, `[Curious]`) 的案例展示 [Table IV] 直观且有说服力

**不足**:
- 作为框架论文,技术深度有限: 三个 agent 都是直接调用已有模型,无模型架构或训练方法的创新
- Critic 的评审维度 (naturalness + clarity/emotiveness) 过于粗粒度,且评审结果如何映射到具体的脚本修改缺乏分析
- EMOS 和 TMOS 的定义依赖人工评分,没有给出自动化计算方案,限制了指标的实用性
- 仅与 30 个对话的 5 种变体做比较,缺乏与 SpeechAgents、KE-Omni 等同类工作的直接对比
- MultiTalk 的 Ground Truth MOS 仅 3.76 (EN) / 3.78 (CN) [Table VI],说明合成数据本身质量有限

**总体判断**: 本文是一篇工程导向的系统论文,提出了 agent 协作生成对话数据的可行框架。框架的核心价值在于将 LLM 的文本生成能力与 TTS 的语音合成能力通过 critic 反馈连接起来,但技术创新有限。MultiTalk 数据集的贡献受限于规模和质量。

## 可复用的 idea

1. **多模态 critic agent**: 用多模态 LLM (如 Qwen2-Audio) 评审合成语音,生成结构化文本反馈用于脚本优化 -- 这个思路可以泛化到任何"生成-评审"循环中,例如歌声合成的歌词-旋律匹配评审
2. **Character Pool + 社交关系**: 给说话人设定详细人设 (年龄/性格/语言习惯) 和彼此间社交关系,使 LLM 生成更符合角色的对话脚本 -- 可用于有声书、游戏配音等场景的数据生成
3. **副语言标记迭代注入**: 初始脚本不含副语言标记,通过 critic 反馈后由 Script Writer 有针对性地插入 `<strong>`, `[breath]`, 情感标签 -- 比一开始就加标记更可控,避免过度标注
4. **对话级语音评估维度**: EMOS (情感一致性) 和 TMOS (轮次切换自然度) 作为对话语音的补充评估维度,虽然目前仅有主观定义,但可以启发自动化对话语音质量指标的研究
