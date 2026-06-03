---
type: paper
tier: deep
title: "StoryTTS: A Highly Expressive Text-to-Speech Dataset with Rich Textual Expressiveness Annotations"
arxiv_id: "2404.14946"
source: "Sources/StoryTTS.pdf"
authors: [Sen Liu, Yiwei Guo, Xie Chen, Kai Yu]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, dataset, expressiveness, prosody, annotation, LLM, Mandarin, storytelling]
concepts: ["[[Prosody Modeling]]", "[[Emotion Control in TTS]]", "[[Style Transfer in TTS]]", "[[Global Style Tokens]]", "[[Natural Language Description for TTS]]", "[[TTS Evaluation]]"]
models: ["VQTTS (Du et al., 2022)"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Prosody Modeling]]✓, [[Emotion Control in TTS]][待确认], [[TTS Evaluation]][待确认], [[Style Transfer in TTS]][待确认], [[Global Style Tokens]][待确认], [[Natural Language Description for TTS]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: StoryTTS 位于 Expressive TTS 数据集建设的交叉点,同时触及韵律建模和情感控制两大方向。在韵律建模领域,已有方法主要从语音信号端入手: 通过 Reference Encoder / GST (Wang et al., 2018) 隐式学习风格,或通过 FastSpeech 2 的 variance adaptor 显式建模 pitch/duration/energy。StoryTTS 的不同之处在于 **从文本端出发**,系统性定义和标注文本中蕴含的、影响韵律的语言学特征。

**已有认知**:
- [[Prosody Modeling]] 指出韵律建模有两条路线: 显式 (pitch/duration/energy predictor) 和隐式 (reference encoder/VAE/flow),但对 **文本端的韵律线索** (如修辞手法、句式、角色扮演) 关注不足。StoryTTS 恰好补充了这一缺口。
- [[Emotion Control in TTS]] 记录了情感标签通常是离散类别 (happy/sad/angry),而 StoryTTS 的 emotional color 维度采用 **自由文本描述** (如 "worry and anxiety"),与 [[Natural Language Description for TTS]] 中 PromptTTS 等工作的思路相呼应。
- [[Style Transfer in TTS]] 梳理了风格控制从 Style Tagging → Reference Speech → NL Description → Instruction-Guided 的演进,StoryTTS 的标注框架可视为为 Style Tagging 提供更精细的标签体系。
- [[TTS Evaluation]] 中 MOS、MCD、log-F0 RMSE 是 StoryTTS 实验采用的核心指标。

**创新判断**: 与已有知识库对比,StoryTTS 的核心新意在于: (1) 首次系统性定义文本中影响语音表现力的五个维度并提供标注; (2) 用 LLM (GPT-4/Claude-2) 进行批量标注的方法论; (3) 从"评书" (Pingshu) 这一特殊语料源获取的高韵律变化数据。这三点在现有概念页中均未被覆盖。

## 速查

> [!summary] 速查
> - **一句话**: 首个同时包含语音和文本表现力标注的 TTS 数据集,从中文评书录音构建,附带 LLM 驱动的五维度文本表现力标注框架
> - **路线**: 评书录音 → VAD+Whisper 分割 → 人工校正 → LLM 五维度标注 (句式/修辞/场景/角色/情感) → Expressiveness Encoder 注入 VQTTS
> - **指标**: +ALL 融合全部标签: MCD 6.181 (baseline 6.904, ↓10.5%), log-F0 RMSE 0.402 (baseline 0.437, ↓8.0%), MOS 4.09±0.07 (baseline 3.88±0.07, ↑5.4%) [Table 4]
> - **可借鉴**: (1) 用 LLM few-shot annotation 替代人工标注降低成本,不同 LLM 负责不同维度 (Claude-2 做结构标注,GPT-4 做情感描述); (2) 将文本表现力分解为五个正交维度的分析框架; (3) 基于 BERT cross-attention 建模情感在句内的分布
> - **局限**: 单说话人 (无法验证多说话人泛化); 仅中文评书域 (域外迁移未验证); LLM 标注准确率未给出定量评估; Expressiveness Encoder 设计较简单 (embedding lookup + cross-attention)

## 核心问题

本文要解决的核心问题是: **现有 Expressive TTS 研究主要关注语音端的声学表现力 (acoustic expressiveness),而忽略了文本本身蕴含的、影响说话方式的表现力信息 (textual expressiveness)**。具体来说:

1. 现有方法依赖预训练语言模型的粗粒度语义表征 (如 BERT) 或基本句法结构 (如 dependency tree),未对"文本中哪些特征影响语音表现力"进行系统分析 [§1]
2. 缺少同时提供文本表现力标注和高表现力语音的数据集 [§1, Table 2]
3. 需要验证: 利用文本表现力标注是否确实能提升合成语音的表现力 [§4]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

StoryTTS 的贡献分两部分: 数据集构建 + 实验验证。

**数据集构建 pipeline**:
```
评书录音 (160章, ~64h)
  → VAD 分割 (移除长静音)
  → Whisper ASR 转写
  → 长段二次切割 (Aeneas 文音对齐)
  → 人工逐行校正识别错误
  → 标点增强
  → LLM 五维度标注
  → 最终: 33108 条, 60.9h, 16kHz, SNR 32dB [Table 1]
```

**实验验证架构**:
```
Phoneme → [t2v: Acoustic Model] → VQ Feature + Auxiliary (pitch/energy/PoV)
                    ↑
         Expressiveness Encoder
         (4 embedding tables + BERT cross-attention for emotional color)
                    ↓
VQ Feature + Auxiliary → [v2w: Vocoder] → Waveform
```

### 关键设计选择

#### 1. 数据源选择: 为什么选评书 (Pingshu)

[论文原文] 评书是中国传统口头艺术形式,表演者叙述故事、模仿声音、塑造角色。这种形式基于历史小说,使得评书不仅在语音韵律上丰富,在文本表现力上也多样,包括修辞手法、角色扮演等 [§2.1]。

[agent 解读] 选择评书而非常见的有声书/播客/情感语音,有两个隐含优势: (1) 评书中说话人会模仿不同角色,天然提供了多风格变化,即使只有一个说话人; (2) 评书文本本身富含修辞、对话、旁白等结构,使得文本表现力标注有充分素材。pitch 标准差 98.22 (远高于其他数据集的 39-58 范围) 证实了声学多样性 [Table 2]。

#### 2. 五维度文本表现力框架

[论文原文] 作者从语言学、修辞学和文学研究的视角,将与语音相关的文本表现力定义为五个维度 [§3.1]:

| 维度 | 类别数 | 具体类别 | 对语音的影响 |
|------|--------|----------|-------------|
| Sentence Pattern (句式) | 4 | 陈述/疑问/祈使/感叹 | 语调走向和情感基调 [§3.1] |
| Rhetorical Device (修辞手法) | 11 | 夸张/对偶/拟声/比喻/拟人/引用/反语/反问/设问/反复/转折 | 语音节奏和重音模式 [§3.1] |
| Scene (场景) | 3 | 角色扮演/旁白/内心独白 | 音色、语速和情感强度 [§3.1] |
| Imitated Character (模仿角色) | 19 类 | 基于年龄/性别/身份分为 19 种角色类型 | 直接决定音高和语速 [§3.1] |
| Emotional Color (情感色彩) | 开放 | 自由文本描述 (如 "worry and anxiety") | 整体情感状态 [§3.1] |

[agent 解读] 前四个维度是分类标签 (离散),第五个是开放描述 (连续)。这种混合设计的逻辑在于: 句式/修辞/场景/角色有明确的语言学分类依据,适合离散标注; 而情感色彩难以用有限类别覆盖 (如"忧虑而愤怒"不在标准情感类别中),因此采用自由描述。

#### 3. LLM 驱动的批量标注

[论文原文] 使用 Claude-2 标注句式、修辞手法、场景和模仿角色; 使用 GPT-4 标注情感色彩 (因 Claude-2 在情感描述上表现不佳) [§3.2]。

标注策略:
- Zero-shot 尝试准确率低,转为 few-shot [§3.2]
- Prompt 设计: 设定"语言学家"人设 → 告知输入文本是连续的 (需上下文) → 描述文本特征 (拟声/独白/角色扮演) → 要求按指定格式标注 [§3.2]
- Few-shot examples 包含多样化标注示例和理由说明 [§3.2]

[agent 解读] 将不同维度分配给不同 LLM 是务实的选择: Claude-2 擅长结构化分类 (句式/修辞等),GPT-4 擅长开放式情感描述。这种"LLM 互补使用"的思路可迁移到其他标注任务。但论文未报告标注准确率/一致性 (如 human-LLM agreement),这是一个显著遗漏。

#### 4. Expressiveness Encoder 设计

[论文原文] 为验证标注有效性,在 VQTTS [22] 基础上增加 Expressiveness Encoder [§4.1.2]:

- **句式/场景/修辞/角色**: 各一个 learnable embedding table,维度分别为 32/32/64/256。角色维度最大 (256),因 19 种角色类型需要更丰富的表示 [§4.1.2]
- **情感色彩**: BERT 提取词级 embedding + Sentence BERT 提取情感描述 embedding → cross-attention 建模情感在句内的分布 → 上采样到音素级 → 加到 encoder 输出 [§4.1.2]

[agent 解读] Cross-attention 的设计意图是: 情感在句子内部是非均匀分布的 (如感叹句结尾情感更强),通过 BERT 词级特征和情感描述之间的 attention,可以让模型学习情感在哪些位置更强烈。这比简单将情感嵌入加到全句上更精细。

### 训练策略

- Acoustic model: 300 epochs, batch size 8, single 2080 Ti [§4.2]
- Vocoder: 100 epochs on StoryTTS, shared across experiments [§4.2]
- G2P: 内部工具 [§4.2]
- Duration: Montreal Forced Aligner + Kaldi [§4.2]
- 数据划分: 5% test+validation, test set = 3 consecutive chapters [§4.2]

## 实验

| 指标 | 本文 (+ALL) | Baseline (VQTTS) | GT(Voc.) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MCD ↓ | 6.181 | 6.904 | 3.765 | StoryTTS test | [Table 4] |
| log-F0 RMSE ↓ | 0.402 | 0.437 | 0.322 | StoryTTS test | [Table 4] |
| MOS ↑ | 4.09±0.07 | 3.88±0.07 | 4.29±0.06 | StoryTTS test, 20 listeners | [Table 4] |

**各标签的单独贡献** (从 baseline 改善幅度排序) [Table 4]:

| 标签 | MCD ↓ | log-F0 RMSE ↓ | MOS ↑ |
| --- | --- | --- | --- |
| +Emotional Color | 6.271 | 0.412 | 3.92±0.08 |
| +Imitated Character | 6.508 | 0.411 | 3.96±0.07 |
| +Rhetoric Device | 6.633 | 0.421 | 3.93±0.07 |
| +Scene | 6.692 | 0.431 | 3.90±0.07 |
| +Sentence Pattern | 6.746 | 0.432 | 3.90±0.08 |
| Baseline | 6.904 | 0.437 | 3.88±0.07 |

**关键发现**:

1. **模仿角色 (Imitated Character) 在 MOS 上贡献最大** (3.96): [论文原文] 因为角色信息直接告诉模型当前在模仿谁的说话方式,模型可以高效学习不同角色的语音特征 [§4.3.2]

2. **情感色彩 (Emotional Color) 在客观指标上贡献最大** (MCD 6.271, 最低): [agent 解读] 这可能因为情感色彩通过 cross-attention 机制在音素级别提供了更精细的条件信息,而不仅是句子级 embedding

3. **句式和场景贡献最小**: [论文原文] 句式中陈述句占比过高,模型获取的信息有限; 场景虽然分布均匀,但类别太少 (仅 3 类),且角色扮演/内心独白中的角色差异使得场景标签提供的信息不够 [§4.3.2]

4. **全部融合 (+ALL) 效果最好**: [论文原文] 五个维度提供互补信息,融合后在主客观指标上均显著优于任何单标签 [§4.3.2]

## 局限性

1. **单说话人数据集**: 仅一位女性表演者,无法验证多说话人场景下文本表现力标注的泛化性 [agent 解读]
2. **域特异性**: 评书是特殊的表演形式,其文本表现力维度 (如模仿角色) 不一定适用于其他 expressive TTS 场景 (如新闻播报、对话) [agent 解读]
3. **LLM 标注质量未量化**: 未报告 LLM 标注与人工标注的 agreement、各维度标注准确率、标注一致性 (inter-annotator agreement) [agent 解读]
4. **Baseline 较弱**: VQTTS (2022) 不是当时最强的 TTS 模型,未与 VITS/FastSpeech 2 等广泛使用的 baseline 对比 [agent 解读]
5. **仅句子级实验**: 未探索段落/章节级的连续表现力建模,而评书本身是连续叙事 [agent 解读]
6. **标注框架的完备性**: 五维度是否穷尽了文本表现力? 如停顿标记 (pause marking)、语速指示 (speed cues) 等维度未纳入 [agent 解读]
7. **16kHz 采样率**: 在 2024 年较低,限制了高频声学细节的保真度 [agent 解读]

## 点评

StoryTTS 的主要价值在于 **提出了一个被忽视但重要的研究方向**: 文本中蕴含的、影响语音表现力的语言学特征。现有 Expressive TTS 工作主要从语音信号端 (reference encoder/GST/VAE) 或简单的文本语义 (BERT embedding) 入手,而 StoryTTS 系统性地分析了文本中的修辞、句式、场景等结构化特征对语音表现力的影响。

**数据集层面**: 61 小时的高韵律变化中文语音 (pitch std 98.22) + 完整的五维度文本标注,填补了 Expressive TTS 数据领域的空白。但 **单说话人 + 单域** 限制了其通用性。

**方法层面**: LLM 批量标注的方法论是实用的 — 用不同 LLM 擅长的能力处理不同维度,但 **缺少标注质量评估** 是硬伤。Expressiveness Encoder 的设计较为直接 (embedding lookup),与 2024 年的技术水平相比显得简单。

**实验层面**: 实验设计合理 (逐标签消融 + 全融合),验证了文本表现力标注确实有效。但 MOS 改善幅度有限 (3.88→4.09),且 MOS 的 confidence interval 重叠,统计显著性不够明确。

**与已有知识的关系**: 本文的文本表现力框架可视为对 [[Prosody Modeling]] 中 "Text Pre-training" 分支的具体化 — 不是用 BERT 隐式学习文本韵律信息,而是显式标注文本中的韵律相关特征。这与 [[Natural Language Description for TTS]] 的方向互补: 后者关注用自然语言描述目标语音属性,本文关注从文本本身提取影响语音的特征。

## 可复用的 idea

1. **LLM 互补标注策略**: 不同 LLM 各有擅长,将标注任务按维度分配给最擅长的 LLM,可降低成本并提升质量。这一策略可推广到任何多维度标注任务。

2. **五维度文本表现力框架**: 将文本表现力分解为句式/修辞/场景/角色/情感五个维度,可作为分析 expressive text 的通用框架,特别适用于有声书、影视剧本等场景。

3. **BERT cross-attention 建模句内情感分布**: 用 Sentence BERT 提取全局情感描述 + BERT 词级 embedding + cross-attention = 句内情感空间分布。这种设计可迁移到其他需要建模属性在序列内非均匀分布的场景。

4. **评书 (Pingshu) 作为 Expressive TTS 数据源**: 评书自然提供丰富的角色模仿、情感变化和修辞手法,是构建 expressive speech 数据集的高效数据源。

---

> [!review] 审阅状态: pass-with-fixes (2026-06-03)
> - **结论**: pass-with-fixes (2 issues resolved, 1 low pending)
> - **已修复**: models 字段误引 VITS→VQTTS; 数据集统计补 [Table 1] 标注
> - **待定 (low)**: datasets 字段为空 (待创建 StoryTTS 数据集页后回填)
> - 详见 `_review/StoryTTS-review.yml`

---

检索命中: [[Prosody Modeling]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[TTS Evaluation]](pending-review), [[Style Transfer in TTS]](pending-review), [[Global Style Tokens]](pending-review), [[Natural Language Description for TTS]](pending-review) | 未命中但可能相关: 无
