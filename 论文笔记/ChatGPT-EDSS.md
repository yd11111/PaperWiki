---
type: paper
tier: deep
title: "ChatGPT-EDSS: Empathetic Dialogue Speech Synthesis Trained from ChatGPT-derived Context Word Embeddings"
arxiv_id: "2305.13724"
source: "Sources/ChatGPT-EDSS.pdf"
authors: [Yuki Saito, Shinnosuke Takamichi, Eiji Iimori, Kentaro Tachibana, Hiroshi Saruwatari]
year: 2023
venue: "Interspeech 2023"
tags: [TTS, dialogue, empathy, emotion, ChatGPT, prompt-engineering, context-embedding, style-control, FastSpeech2, BERT]
concepts: ["[[EmotionControlinTTS]]", "[[GlobalStyleTokens]]", "[[NaturalLanguageDescriptionforTTS]]", "[[ProsodyModeling]]", "[[StyleTransferinTTS]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[EmotionControlinTTS]], [[GlobalStyleTokens]], [[NaturalLanguageDescriptionforTTS]], [[ProsodyModeling]], [[StyleTransferinTTS]], [[LLM-basedTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓ | 过滤: [[EmotionControlinTTS]][待确认], [[GlobalStyleTokens]][待确认], [[NaturalLanguageDescriptionforTTS]][待确认], [[StyleTransferinTTS]][待确认] | 未命中但可能相关: 无

**谱系定位**: 本文处于 Emotion Control in TTS 与 Natural Language Description for TTS 两条线的交叉点。在 Emotion Control 演进线中,2023 年 5 月的时间点位于 "多尺度层级建模 (MsEmoTTS, 2022)" 到 "DPO/RLHF 对齐 (Emo-DPO, 2024)" 之间,属于早期探索 LLM 辅助情感理解的工作。在 NL Description for TTS 演进线中,与 PromptTTS (Guo et al., ICASSP 2023) 同期,但关注点不同: PromptTTS 用文本描述直接控制 TTS 的 5 个属性; 本文用 ChatGPT 从对话历史中提取隐含的情感/意图/风格描述词,作为弱监督信号替代人工标注。

**已有认知**: 概念库已有大量关于情感控制和风格迁移的工作记录。GST (Wang et al., 2018) 提出无监督风格 token bank; TP-GSTs (Stanton et al., 2018) 从文本预测 style embedding。本文作者明确引用 TP-GSTs 作为相关工作,并提出了差异: TP-GSTs 不考虑对话历史。本文的 CCE (Guo et al., 2021) 是 conversational end-to-end TTS 的方法,用 BERT + GRU 从对话历史中提取隐式上下文嵌入。

**创新判断**: 本文的主要创新在于将 ChatGPT 引入 spoken dialogue 研究 --- 用 LLM 的阅读理解能力替代人工情感标注或数据驱动的黑箱上下文嵌入,生成可解释的对话上下文描述 (intention/emotion/style 三词)。这是 "LLM 作为外部语义分析器辅助 TTS" 范式的早期实例,与后来的 NL description TTS 和 LLM-based emotion control 路线相呼应。

## 速查

> [!summary] 速查
> - **一句话**: 用 ChatGPT 从对话历史中提取 intention/emotion/style 三个上下文词,其 BERT 嵌入替代人工情感标注或数据驱动 CCE 来条件化 EDSS 模型,效果持平
> - **路线**: 对话历史 → ChatGPT prompt → 三词 (intention, emotion, style) → BERT embedding → sum → Linear → FastSpeech 2 条件化 → mel → HiFi-GAN → waveform
> - **指标**: Naturalness MOS 3.52 (IES only) vs 3.43 (Emo only) vs 3.54 (CCE only); Similarity MOS 3.19-3.24 无显著差异 [Table 3]
> - **可借鉴**: 用 LLM 的阅读理解能力自动标注对话上下文,替代昂贵的人工情感标注; prompt 设计中包含 dialogue situation description 可改善结果相关性
> - **局限**: 仅在日语小数据集 (STUDIES) 上验证; ChatGPT 回答多样性大 (79% 的 intention 词出现 ≤ 5 次); reliability score 与 MOS 提升无相关性; 未验证在非共情对话场景的泛化

## 核心问题

1. **对话语音合成中的情感上下文从何而来?** 传统方法依赖人工标注的 utterance-level emotion labels 或黑箱的数据驱动 context embedding (CCE),前者成本高,后者不可解释。能否利用 LLM 的语言理解能力自动提取可解释的对话上下文? [§1]
2. **ChatGPT 提取的上下文信息是否可靠到足以替代人工标注?** 如果 ChatGPT 能从对话历史中准确理解说话意图、情感和风格,那么它的输出可以作为弱监督标签训练 EDSS 模型 [§1, §3.3]
3. **共情对话的 "共情" 如何体现在语音合成中?** EDSS 的核心是 listener 对 speaker 情感的共情回应,这要求模型理解对话历史中的情感流向,而不仅是当前句子的情感 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统分两步 [§3, Fig 2]:

**Step 1: ChatGPT 上下文词收集** [§3.1]
- 构造 text prompt 包含三部分: (1) 对话设定描述 (角色 + 场景); (2) 对话台词 (格式: "[turn ID] [speaker name] [content]"); (3) 要求 ChatGPT 为每行回答 intention/emotion/style 三个词
- Emotion 候选: neutral + Plutchik 八情感 (joy, anticipation, anger, disgust, sadness, surprise, fear, trust) [§3.1]
- Style 候选: cute, cool, quiet, polite, intellectual, honest, clear, gentle, gravelly, vibrant [§3.1]
- 长对话 (>5 turns) 切分为重叠 2 turns 的多段查询,避免 ChatGPT 中途挂起 [§3.1]

**Step 2: EDSS 模型训练** [§3.2]
- 三词的 BERT embedding 求和 → 线性投影到 256 维 → 作为 FastSpeech 2 的条件输入
- 重叠查询导致一句话可能有多组上下文词 → 取各词嵌入的平均值聚合 [§4.3]
- FastSpeech 2 先在 JSUT (~10h 日语单说话人) 上预训练 200K 步,再在 STUDIES 上微调 100K 步 [§4.1]
- HiFi-GAN vocoder 在相同训练数据上训练 350K 步 [§4.1]

### 关键设计选择

1. **为什么是三个词 (intention/emotion/style) 而非直接生成 embedding?** [论文原文] 作者将 ChatGPT 视为 "interactive context estimator",替代 CCE 的黑箱上下文嵌入。三词设计使上下文信息可解释、可检查 [§3.2]。[agent 解读] 这也规避了 ChatGPT 无法直接输出连续向量的限制 --- 通过 BERT 将离散词转为连续嵌入。

2. **为什么限制对话长度 ≤ 5 turns per query?** [论文原文] 当要求 ChatGPT 回答较长对话时,它倾向于在答案中途挂起 (hang) [§3.1]。[agent 解读] 2023 年 3 月的 ChatGPT 版本 context window 和稳定性有限,这一限制是工程约束。

3. **为什么预定义 emotion/style 类别而不完全自由生成?** [论文原文] 尽管预定义了类别,ChatGPT 实际生成的词高度多样 --- Neutral 类别下有 206 个不同 intention 词,130 个不同 emotion 词 [Table 2]。[agent 解读] 预定义类别是一种约束 prompt 的策略,但 ChatGPT 并不严格遵守,导致弱监督信号有较大噪声。

4. **与 TP-GSTs 的关系** [论文原文] ChatGPT-EDSS 可看作 TP-GSTs 的对话版本 --- 从文本提取 style-related 信息预测 prosody embedding。但 TP-GSTs 只看当前句子,不考虑对话历史 [§3.3]。

5. **弱监督视角** [论文原文] 作者将 ChatGPT-EDSS 视为 weakly supervised expressive TTS,用 context words 替代 ground-truth emotion labels 作为条件 [§3.3]。

### 训练策略

- 两阶段: JSUT 预训练 (200K iters) + STUDIES 微调 (100K iters) [§4.1]
- F0 用 WORLD vocoder 估计 [§4.1]
- Adam 优化器: 初始 lr=0.0625, beta1=0.9, beta2=0.98 [§4.1]
- HiFi-GAN: Adam lr=0.0003, beta1=0.8, beta2=0.99, 350K iters [§4.1]
- 语料: STUDIES corpus (日语共情对话,726 训练 / 72 验证 / 72 测试对话) [§4.1]

## 实验

| 指标 | 本文 (IES only) | Emo only | CCE only | Emo+IES | Emo+CCE | CCE+IES | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Naturalness MOS | 3.52±0.14 | 3.43±0.14 | 3.54±0.14 | 3.52±0.14 | 3.43±0.14 | 3.49±0.14 | [Table 3] |
| Similarity MOS | 3.19±0.15 | 3.20±0.15 | 3.24±0.14 | 3.21±0.14 | 3.24±0.14 | 3.20±0.14 | [Table 3] |

**上下文词分析** [§4.2, Table 1]:
- 平均 reliability score: Neutral 3.95, Happy 4.04, Angry 3.66, Sad 4.03 (均 > 3.6/5)
- Angry/Sad 最常见 intention 词为 "empathy" --- ChatGPT 理解了共情对话的本质
- Emotion 词基本对应 ground-truth 标签 (Happy→joy, Sad→sadness),但 Angry→trust, Neutral→anticipation 存在偏差
- Style 词集中在 "quiet/gentle/polite" --- ChatGPT 将 STUDIES 教师的风格概括为温和

**多样性分析** [§4.2, Table 2]:
- Neutral 类别下 intention 有 206 个 unique 词,其中 79% 出现 ≤ 5 次
- 即便预定义了 emotion/style 类别,实际生成仍高度多样

**Reliability vs MOS** [§4.3, Fig 4]:
- Reliability score 与 MOS improvement 之间无相关性
- 即便 reliability=5,MOS improvement 仍大幅波动

## 局限性

1. **数据规模极小**: 仅在 STUDIES corpus (~726 对话) 上验证,且为日语单一说话人 (女性教师),泛化性完全未知 [§4.1]
2. **ChatGPT 响应不可控**: 79% 的 intention 词出现 ≤ 5 次,噪声极大; 预定义类别约束效果有限 [Table 2, §4.2]
3. **人工参与仍不可免**: 需要 31 名 worker 手动复制 prompt → ChatGPT → 复制答案 → 评分,且需人工判断是否重发查询 [§4.1]; 虽然提到 API 可自动化,但本文实验未使用
4. **对话场景单一**: 仅测试了 school chit-chat (教师-学生) 场景,未验证其他对话领域 (如客服、医疗) [§5]
5. **Hallucination 风险未评估**: 作者在结论中提到需检查 ChatGPT 是否产生幻觉,但本文未做 [§5]
6. **效果仅"持平"**: IES 并未显著超越 Emo 或 CCE baseline,最大 MOS 差异 0.11 在 95% CI 范围内重叠 [Table 3]
7. **时效性限制**: 基于 2023 年 3 月版 ChatGPT (GPT-3.5 级别),5-turn 限制和挂起问题在现代 LLM 中可能不再存在

## 点评

**优点**:
- **开拓性**: 这是较早将 ChatGPT/LLM 引入 spoken dialogue TTS 研究的工作,在 2023 年 5 月具有新颖性
- **可解释性**: 三词上下文 (intention/emotion/style) 比 CCE 的黑箱 embedding 更透明,人类可以检查和修改
- **实验设计合理**: 6 种条件组合 (Emo/CCE/IES 及两两组合) 的消融实验覆盖全面
- **实用洞察**: 发现 ChatGPT 在共情对话中最常生成 "empathy" 作为 intention,证明其语义理解能力

**不足**:
- **贡献深度有限**: 本质上是一个 pipeline 拼接工作 (ChatGPT → BERT → FS2),缺乏对 "为什么 ChatGPT 词嵌入能 work" 的机制分析
- **效果无突破**: "持平" 的结果说明 ChatGPT 词嵌入包含的信息量与 emotion label 或 CCE 相当,但也未能证明 LLM 理解带来额外增益
- **规模限制**: 所有实验在极小数据集上进行,结论的可靠性受限
- **ChatGPT 版本依赖**: 结果高度依赖特定时间点的 ChatGPT 行为,可复现性存疑

**在演进中的位置**: 本文是 "LLM 辅助 TTS 风格/情感控制" 路线的早期探索节点。后续工作如 EmoVoice (2025) 直接用 LLM 生成自由文本情感提示,PUE (2025) 用百分比模板实现混合情感,RLAIF-SPA (2025) 用 LLM 标注韵律-情感标签作为 reward --- 这些工作在更大规模和更深层次上推进了 ChatGPT-EDSS 开启的方向。

## 可复用的 idea

1. **LLM 作为弱标注器**: 用 LLM 从对话历史自动生成 (intention, emotion, style) 三元组,替代人工标注。这个 idea 可迁移到任何需要对话上下文理解的 TTS/ASR 任务
2. **三维对话上下文分解**: 将对话上下文分解为 intention (交际目的) + emotion (情感状态) + style (说话风格) 三个独立维度,每个维度用一个词表示。这种结构化分解比整体 embedding 更可控
3. **Prompt 中包含 dialogue situation**: 作者发现在 prompt 中加入对话场景描述 (如 "listener prizes the speaker who got a good score") 可以改善 ChatGPT 回答的相关性,这是 prompt engineering 的实用技巧
4. **重叠查询 + 嵌入平均**: 对长对话的切片策略 (重叠 2 turns) 和多组上下文词的平均聚合,是处理 LLM 上下文窗口限制的通用方案

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法流程清晰,两步架构易理解 |
> | 可信赖 | pass | 数字均标注出处,结论与数据一致 |
> | 可区分 | pass | 与 TP-GSTs/CCE 的差异明确,KB 定位准确 |
> | 可定位 | pass | 在 Emotion Control 和 NL Description 两条线中定位清晰 |
> | 不污染 | pass | 未引入未经验证的推测性结论 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> - [medium] 论文局限性可更尖锐地指出 MOS 差异在统计上不显著 (CI 重叠),当前措辞 "持平" 较温和
> - [low] 可补充与同期 PromptTTS (Guo et al., 2023) 的直接对比讨论
> 详见 `_review/ChatGPT-EDSS-review.yml`

---

检索命中: [[ProsodyModeling]], [[LLM-basedTTS]] | 过滤: [[EmotionControlinTTS]](pending-review), [[GlobalStyleTokens]](pending-review), [[NaturalLanguageDescriptionforTTS]](pending-review), [[StyleTransferinTTS]](pending-review) | 未命中但可能相关: 无
