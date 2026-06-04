---
type: paper
tier: deep
title: "Computational Narrative Understanding for Expressive Text-to-Speech"
arxiv_id: "2509.04072"
source: "Sources/LibriQuote.pdf"
authors: [Gaspard Michel, Elena V. Epure, Christophe Cerisara]
year: 2025
venue: "Findings of ACL 2026"
tags: [TTS, dataset, expressive-speech, audiobook, zero-shot, flow-matching, evaluation, narrative, prosody]
concepts: ["[[Prosody Modeling]]", "[[Emotion Control in TTS]]", "[[Natural Language Description for TTS]]", "[[Conditional Flow Matching]]", "[[TTS Evaluation]]"]
models: ["[[论文笔记/Spark-TTS|Spark-TTS]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "F5-TTS"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于"表现力语音合成数据"方向,核心贡献是一个从有声书中提取的角色台词数据集 LibriQuote。与 [[Emilia]] (in-the-wild 多样性路线) 形成互补 -- Emilia 提供规模和多样性,LibriQuote 提供叙事语境标注的表现力语料。在方法实验上,本文验证了 [[Conditional Flow Matching]] (F5-TTS) 与自回归 LLM-TTS (SparkTTS) 在表现力微调上的差异化表现,这与 KB 中已有的"flow matching 更适合细粒度声学还原"认知一致。
>
> **已有认知**: [[Prosody Modeling]] 概念页已梳理从 GST 到 NVSpeech 的韵律建模演进,本文的"叙事语境伪标签"(speech verbs/adverbs) 可视为 [[Natural Language Description for TTS]] 的一种新形式 -- 不同于 PromptTTS 的显式属性描述,而是从文学叙事中自动提取的隐式说话风格指示。[[Emotion Control in TTS]] 概念页中 IndexTTS2 的 soft-instruction 情感解耦已有记录,本文进一步验证了 IndexTTS2 在表现力基准上的优势。[[TTS Evaluation]] 概念页中 LALM-as-a-Judge (Manku et al., 2025) 已被记录,本文是该方法在表现力评估场景的实际应用。
>
> **创新判断**: 相较于 KB 中已有的表现力 TTS 工作,本文的独特贡献在于 (1) 叙事感知分割 -- 按角色台词而非随机句子边界切分有声书; (2) 首次大规模自动提取 speech verbs/adverbs 伪标签; (3) 提出 ContextMOS 和 Win-Rate 两个基于 LALM 的表现力评估指标。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Prosody Modeling]]✓, [[Zero-shot Speech Synthesis]]✓ | 参考: [[Emotion Control in TTS]][待确认], [[TTS Evaluation]][待确认], [[Natural Language Description for TTS]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 从有声书角色台词中构建 5.3K 小时表现力语音数据集 LibriQuote,附带叙事语境伪标签,验证其对 flow-matching TTS 微调效果显著优于自回归 TTS
> - **路线**: LibriVox 有声书音频 → BookNLP 台词检测 → Zipformer ASR 转写 → Levenshtein 对齐 → 台词/叙述分割 → LLM (Phi-4) 提取 speech verbs/adverbs → 按表现力过滤子集 Qf (379h)
> - **指标**: F5-TTS FT(Qf) ContextMOS 3.33 vs 基线 2.95 (+0.38) [Table 2]; IndexTTS2-Context Win-Rate 54% 达到人类水平, CMOS +0.03 [Table 4]; SparkTTS 从头训练 Full(N∪Q) ContextMOS 3.30 vs 基线 2.94 [Table 2]
> - **可借鉴**: (1) 叙事感知分割思路 -- 按角色台词而非随机句边切分有声书,可获得更纯粹的表现力语料; (2) speech verbs/adverbs 伪标签 -- 低成本自动提取有声书中"how to say"信息; (3) LALM-as-Judge 评估表现力的 ContextMOS/Win-Rate 方案
> - **局限**: 数据集未做 WER-based 过滤可能含少量异常; 叙事语境条件仅在 SparkTTS 从头训练和 IndexTTS2 的情感预测中验证,未在 F5-TTS 等 flow-matching 模型上探索; LibriVox 业余朗读者本身表现力有限

## 核心问题

本文试图解决两个相互关联的问题:

1. **有声书数据被低估了表现力潜力**: 现有大规模有声书数据集 (LibriSpeech, LibriTTS, LibriHeavy) 要么丢弃角色台词,要么将台词与叙述混合在随机句子边界切分的片段中。这导致片段内 pitch 分布复杂 (含多种说话风格),增加 TTS 训练难度 [§2.2, Fig 1]。

2. **缺乏大规模表现力语音评估基准**: 现有情感语音数据集 (IEMOCAP, RAVDESS) 规模小、场景有限; Emilia 等 in-the-wild 数据集虽然多样,但缺少说话风格的显式标注 [§2.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

LibriQuote 不是一个模型,而是一个**数据集构建 pipeline + 评估框架**。核心思路是利用叙事结构 (narrative discourse) 将有声书音频分割为台词 (quotations) 和叙述 (narrations),并用 LLM 从文本上下文中提取说话风格伪标签。

Pipeline 分两大部分:
1. **数据构建**: 音频准备 → 文本准备 (台词检测) → ASR 转写 → 文本-音频对齐 → 台词分割 + 叙事语境提取
2. **评估框架**: 标准客观指标 (WER, SIM-O, E-Sim) + 基于 LALM 的语境指标 (ContextMOS, Win-Rate)

**数据集规模** [Table 1]:
- 训练集: 12,723 小时叙述 (N) + 5,359 小时台词 (Q),其中高表现力子集 Qf = 379 小时 (378K 条台词)
- 测试集: 7.4 小时台词 (5,598 条),15 位未见说话人 (8 男 7 女)
- 开发集: 5.1 小时台词 (2,921 条),与训练集说话人重叠但不与测试集重叠
- 说话人: 3,314 个,书籍: 2,991 本,均来自 LibriVox 的 Fiction 分类

### 关键设计选择

**为什么按台词而非句子边界切分?**
[论文原文] 作者在 LibriHeavy-small 的 120,000 个片段上分析发现,25% 的片段包含 1-12 个台词,且 pitch 标准差随台词数量显著增加 (Spearman ρ=0.218, p<0.001) [§2.2, Fig 1]。这意味着混合片段内存在多种 pitch 分布,TTS 模型可能倾向于学习更简单的叙述部分而忽略表现力台词。[agent 解读] 这本质上是一个数据清洁度问题 -- 按叙事结构切分可以得到说话风格更一致的语音片段,降低建模难度。

**为什么选择 speech verbs/adverbs 作为伪标签?**
[论文原文] 文学作品中,作者通过 speech verbs 和 adverbs (如 "he whispered softly") 显式描述角色的说话方式 [§3]。这些信息已被证明能增强阅读理解和角色认同 (Wolters et al., 2022; Van Krieken et al., 2017),对有声书朗读者也是设定语调的重要参考。形容词和名词虽然也可能包含说话风格信息,但在标注数据中出现频率较低,且 LLM 提取效果很差 [§3.3, Table 6]。

**为什么选择 Phi-4 作为标注 LLM?**
[论文原文] 在 400 个人工标注样本上比较了 Llama3.3-70b、Phi-4、Qwen3-32b、Qwen2.5-72b 四个模型。Phi-4 (带自信度过滤) 在 adverb 上获得最高精度 (0.95) 且推理速度最快 (6s/sample vs Qwen2.5-72b 的 62s),verb 精度也令人满意 (0.92) [§3.3, Table 6, Fig 2]。优先最大化精度是因为错误的伪标签比遗漏更有害。

**高表现力子集 Qf 的构建逻辑**:
[论文原文] 先纳入所有有非空 adverb 伪标签的台词 (adverb 通常提供精确的说话方式信息),再从剩余台词中纳入 verb 落入 89 个预定义表现力动词列表的台词 (如 "whispered", "screamed", "sobbed",排除 "said", "told" 等中性动词) [§3.3, Appendix B.2]。最终 Qf 包含 377,776 个台词 (占全部台词的 11%),共 379 小时 [Table 1]。

### 训练策略

本文在 SparkTTS 和 F5-TTS 两个基线上进行了多组实验:

**SparkTTS 实验** (自回归 LLM-based):
- **微调 FT(Qf)**: 用表现力子集 Qf 微调,学习率 1e-5, 3 epochs, cosine schedule + 5000 warmup steps [Appendix C.1]
- **微调 FT(Q)**: 用全部台词 Q 微调,设置类似但 warmup 10000 steps
- **从头训练 Scratch(Q)**: 仅用台词数据从头训练,学习率 3e-5, 10 epochs
- **从头训练 + Context**: 将文本条件替换为台词前后各一段的叙事语境 (用特殊标记替换引文内容),测试语境信息能否提升表现力 [Appendix C.2]
- **Full(N∪Q)**: 用全部台词 + 叙述从头训练
- **Full + FT(Qf)**: 先 Full 训练,再 Qf 微调

**F5-TTS 实验** (flow-matching based):
- **微调 FT(Qf)**: 用表现力子集微调,学习率 7.5e-5, 3 epochs, 2xH100 [Appendix D]

**关键发现 -- flow-matching vs 自回归的差异**:
[论文原文] SparkTTS 微调后表现力未显著提升 (ContextMOS 从 2.94 升至 2.97),但可懂度提升; 从头训练可提升表现力 (ContextMOS 3.30) 但牺牲可懂度 [§4.3]。F5-TTS 微调后**同时**提升了表现力和可懂度 (ContextMOS 从 2.95 升至 3.33, WER 从 6.9 降至 6.6) [§4.3, Table 2]。

[论文原文] 作者将这种差异归因于训练目标的不同: "the flow-matching loss of F5-TTS is better able to pick the important expressive information present in LibriQuote's quotations" [§4.3]。[agent 解读] 这可能因为 flow matching 直接在连续声学空间操作,能更好地捕捉韵律/情感等细粒度声学变化; 而自回归 LLM 操作在离散 token 序列上,表现力信息可能在量化时丢失。

## 实验

### LibriQuote-test 评估 (训练实验)

| 指标 | SparkTTS 基线 | FT(Qf) | FT(Q) | Scratch(Q)+Ctxt | Full(N∪Q) | F5-TTS 基线 | F5 FT(Qf) | GT | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER ↓ | 4.8 | 4.6 | 5.9 | 8.2 | 5.1 | 6.9 | 6.6 | 6.5 | [Table 2] |
| SIM-O ↑ | 0.46 | 0.47 | 0.46 | 0.40 | 0.41 | 0.53 | 0.54 | - | [Table 2] |
| E-Sim ↑ | 0.69 | 0.71 | 0.71 | 0.71 | 0.71 | 0.71 | 0.71 | 0.62 | [Table 2] |
| ContextMOS ↑ | 2.94 | 2.97 | 2.89 | 3.15 | 3.30 | 2.95 | 3.33 | 3.55 | [Table 2] |
| Win-Rate ↑ | 38% | 41% | 37% | 35% | 37% | 31% | 26% | - | [Table 2] |

### 基准数据集迁移性 (LibriSpeech-PC & SeedTTS test-en)

| 指标 | SparkTTS | FT(Qf) | FT(Q) | Scratch(Q) | F5-TTS | F5 FT(Qf) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER ↓ | 3.06 | 2.10 | 2.00 | 3.27 | 2.11 | 2.13 | LibriSpeech-PC | [Table 3] |
| SIM ↑ | 0.52 | 0.51 | 0.51 | 0.47 | 0.66 | 0.66 | LibriSpeech-PC | [Table 3] |
| WER ↓ | 2.64 | 2.07 | 1.90 | 5.14 | 1.69 | 1.64 | SeedTTS test-en | [Table 3] |
| SIM ↑ | 0.46 | 0.42 | 0.42 | 0.41 | 0.67 | 0.67 | SeedTTS test-en | [Table 3] |

### 零样本 TTS 基准测试 (Section 5)

| 指标 | GT | SparkTTS | F5-TTS | MaskGCT | IndexTTS2 | IndexTTS2-Context | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER ↓ | 6.5 | 4.8 | 6.9 | 7.6 | 5.2 | 5.4 | [Table 4] |
| SIM-O ↑ | - | 0.46 | 0.53 | 0.56 | 0.49 | 0.50 | [Table 4] |
| E-Sim ↑ | 0.62 | 0.69 | 0.71 | 0.72 | 0.63 | 0.64 | [Table 4] |
| ContextMOS ↑ | 3.55 | 2.94 | 2.95 | 2.94 | 3.33 | 3.45 | [Table 4] |
| Win-Rate ↑ | - | 38% | 31% | 28% | 46% | 54% | [Table 4] |
| CMOS | 0.0 | -0.97 | -0.98 | -0.98 | +0.25 | +0.03 | [Table 4] |
| MOS ↑ | 3.58 | 3.26 | 3.47 | 3.36 | 3.47 | 3.63 | [Table 4] |

**关键观察**:
1. IndexTTS2 在表现力上远超其他系统,CMOS 为正 (+0.25) 意味着被评为比人类朗读者更有表现力 [Table 4]。提供叙事语境 (IndexTTS2-Context) 进一步将 ContextMOS 从 3.33 提升至 3.45,Win-Rate 从 46% 提升至 54% 达到人类水平 [Table 4]。
2. SparkTTS、F5-TTS、MaskGCT 三者在表现力上接近 (ContextMOS ~2.94-2.95, CMOS ~ -0.98),均显著低于人类,说明这些系统虽然自然度高 (MOS 3.26-3.47),但未能捕捉到适当的韵律/情感 [Table 4]。
3. 情感相似度 E-Sim 与表现力评估趋势不一致: MaskGCT 的 E-Sim 最高 (0.72) 但 ContextMOS 最低 (2.94),而 IndexTTS2 的 E-Sim 最低 (0.63) 但 ContextMOS 最高 (3.33) [Table 4]。[论文原文] 作者认为这是因为 IndexTTS2 的情感预测可能产生错误,导致部分样本情感与语境不匹配 [§5.2]。
4. Ground truth WER 较高 (6.5),反映有声书业余朗读者的强情感/口音可能影响 ASR 识别 [§3.2]。

## 局限性

1. **数据质量**: 未对训练集做 WER-based 过滤,可能包含少量异常样本 [§7]。
2. **评估偏差**: LALM-as-a-Judge 存在固有偏见,自动化结果应主要用于模型间比较而非对单个模型下结论 [§7]。
3. **叙事语境利用不充分**: 仅在 SparkTTS 从头训练和 IndexTTS2 情感预测中探索了语境条件,未在 F5-TTS 等 flow-matching 模型上测试 [§7]。
4. **朗读者质量**: LibriVox 业余朗读者的表现力本身有限,某些台词未能充分传达应有的情感 [Appendix H, §6]。
5. **语言单一**: 仅覆盖英语,未扩展至多语言场景。[agent 解读]

## 点评

**优势**:
- **叙事感知分割**是一个巧妙且低成本的思路。现有有声书数据集 (LibriHeavy) 混合台词和叙述导致的多模态 pitch 分布问题是真实存在的 (Fig 1 的实验佐证有说服力),按叙事结构分割是一个自然且有效的解决方案。
- **实验设计全面**: 同时测试了微调和从头训练,覆盖了自回归和 flow-matching 两种架构,得出了 flow-matching 更适合表现力微调的有价值结论。
- **评估创新**: ContextMOS 和 Win-Rate 的 LALM-based 评估方案为表现力 TTS 评估提供了新工具,比传统 E-Sim 更能捕捉语境匹配度。
- **IndexTTS2 的发现**具有实践价值: 解耦情感预测与语音合成,并用叙事语境替代文本本身作为情感预测输入,大幅提升了表现力。

**不足**:
- F5-TTS 的 ContextMOS Win-Rate (26%) 反而低于基线 (31%),与 ContextMOS 上升 (2.95→3.33) 矛盾,论文未充分解释这一不一致。
- 未探索叙事语境条件在 flow-matching 模型上的效果,这是一个明显的遗漏 -- 如果 F5-TTS 微调在无语境下已表现优异,加入语境条件可能进一步提升。
- Speech verbs/adverbs 伪标签在训练中的直接使用仅限于 SparkTTS 的文本条件替换,没有系统地探索其作为额外条件信号的多种集成方式。

## 可复用的 idea

1. **叙事感知分割有声书数据**: 对任何使用有声书语料的 TTS 系统,按台词/叙述分割再分别建模是一个低成本高收益的数据预处理策略。尤其适用于中文有声书 (网文朗读),角色对话和旁白交替频繁。

2. **Speech verbs/adverbs 伪标签**: 可推广到任何有文本上下文的语音数据,不限于有声书。例如电影/电视剧字幕中的 stage directions、小说改编播客等。

3. **LALM-as-Judge 评估表现力**: ContextMOS 和 Win-Rate 的评估方案可直接复用于其他需要评估语音与语境匹配度的场景,如对话系统的情感表现力评估。

4. **解耦情感预测的语境条件**: IndexTTS2-Context 的做法 (用叙事语境而非文本本身预测情感) 可推广到所有支持解耦情感条件的 TTS 系统。

---

> [!review] 审阅 (2026-06-04, agent)
> **结论**: pass-with-fixes (0 high, 1 medium, 2 low)
> - (medium) frontmatter models 遗漏 F5-TTS → 已修复
> - (low) 局限性第5点未标注 [agent 解读] → 已修复
> - (low) 数据集核心统计分散 → 已在方法节补充规模概要
> 详见 `_review/LibriQuote-review.yml`

---

检索命中: [[Conditional Flow Matching]]✓, [[Prosody Modeling]]✓, [[Zero-shot Speech Synthesis]]✓ | 参考: [[Emotion Control in TTS]][待确认], [[TTS Evaluation]][待确认], [[Natural Language Description for TTS]][待确认] | 未命中但可能相关: 无
