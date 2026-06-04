---
type: paper
tier: deep
title: "ParsVoice: A Large-Scale Multi-Speaker Persian Speech Corpus for Text-to-Speech Synthesis"
arxiv_id: "2510.10774"
source: "Sources/ParsVoice.pdf"
authors: [Mohammad Javad Ranjbar Kalahroodi, Heshaam Faili, Azadeh Shakery]
year: 2026
venue: "arXiv"
tags: [TTS, dataset, low-resource, Persian, multi-speaker, speech-corpus, audiobook, data-pipeline, zero-shot]
concepts: ["[[Speaker Embedding]]", "[[TTS Evaluation]]", "[[Phoneme Representation]]"]
models: ["[[论文笔记/XTTS|XTTS (XTTSv2)]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个实体页: [[Speaker Embedding]], [[Zero-shot Speech Synthesis]], [[TTS Evaluation]], [[Phoneme Representation]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: ParsVoice 是一篇**数据集论文**,聚焦低资源语言(波斯语)的大规模 TTS 语料库构建。与本 KB 中已有的众多零样本 TTS 模型论文(CosyVoice, Seed-TTS, IndexTTS2 等)不同,它不提出新模型架构,而是解决**数据稀缺**这一前置瓶颈。KB 中已有 [[论文笔记/XTTS|XTTS]] 笔记,ParsVoice 正是用 XTTS 作为下游验证模型。
>
> **已有认知**: [[Speaker Embedding]] 页记录了 ECAPA-TDNN 作为当前最常用的说话人编码器,ParsVoice 的 speaker identification pipeline 正是基于它。[[TTS Evaluation]] 页指出 MOS 跨研究不可比(Yang et al. 2025),ParsVoice 论文本身也承认其 MOS 与 DeepMine 系统不可直接比较。[[Phoneme Representation]] 页记录了传统 TTS 依赖 G2P 前端,ParsVoice 的一个重要特点是使用 XTTS 实现**无音素**的波斯语 TTS,绕过了波斯语 G2P 的复杂性。
>
> **创新判断**: ParsVoice 的创新主要在数据工程层面(句子完整性验证、边界优化、波斯语特定质量评估),而非模型层面。相比 KB 中 [[数据集/Emilia|Emilia]](101K h 多语言)等大规模数据集,ParsVoice 规模较小(2.2K h)但针对极度低资源的波斯语场景。
>
> 检索命中: [[Speaker Embedding]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[TTS Evaluation]](pending-review), [[Phoneme Representation]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 构建最大公开波斯语 TTS 语料库(2,200h / 1,815 说话人),通过 ParsBERT 句子完整性验证 + 二分搜索边界优化 + 多维质量评估的自动化 pipeline 从有声书中提取 TTS-ready 数据
> - **路线**: 有声书音频 → WebRTC VAD 分割 → Google ASR 转写 → ParsBERT 句子完整性验证(不完整则迭代延伸边界) → 二分搜索+线性微调边界优化 → 音频/文本质量评分过滤 → ECAPA-TDNN 说话人聚类 → 标点恢复 → 最终 TTS 子集
> - **指标**: MOS 3.6/5, SMOS 4.0/5, 可懂度 MOS 4.0/5 (XTTS 微调, 22 名母语评分者); WER 10.40%(Google ASR) / 17.20%(独立 Whisper), Speaker Sim 80.0% (ECAPA-TDNN) [Table 3, Table 4]
> - **可借鉴**: (1) ParsBERT 句子完整性分类器 + 迭代边界延伸确保分割的句子级完整性(97.4% F1); (2) 二分搜索 + 线性微调的边界优化算法,利用 ASR 转写稳定性作为隐式对齐信号,平均去除 426.9ms 起始/181.6ms 结尾静音; (3) 多维质量评估框架(音频 0-100 + 文本 0-1),可迁移到其他低资源语言
> - **局限**: (1) 仅有声书风格(正式朗读),不含对话语音; (2) 仅验证 XTTS 一个架构,泛化性未知; (3) MOS 与 DeepMine 系统不可直接比较(不同评分者/条件); (4) 性别不平衡(67% 男性); (5) 说话人 ID 基于自动聚类而非人工标注

## 核心问题

本文要解决的核心问题是:**波斯语在开放语音资源中严重欠缺**。在 ParsVoice 之前,最大的开放波斯语 TTS 数据集仅 86 小时(ManaTTS, 单说话人) [§1]。DeepMine+ 虽有 480+ 小时,但商业受限 [§2.2]。这直接限制了波斯语多说话人 TTS、语音语言建模等下游任务的发展。

具体挑战:
1. **TTS 对数据质量要求极高**: 需要无噪音音频、完整标点句子、精确的句子级对齐、多说话人覆盖 [§1]
2. **波斯语文字系统特殊性**: 短元音不书写(如 "کتاب" 发音 ketab 而非 ktab)、Ezafe 连接元音不写但必须发音、一个字母可映射多个音素 [§1]
3. **有声书缺乏公开文本转录**: IranSeda 平台只提供音频,无对应文本;OCR 不可靠(有声书常有删节/改编);需 ASR-based 转写策略 [§3.1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ParsVoice 是一个 9 阶段的数据构建 pipeline(不是 TTS 模型),将原始有声书录音转化为 TTS-ready 的语音-文本语料库 [Fig 2]:

```
原始有声书 → (1) VAD 分割 → (2) ASR 转写 → (3) 句子完整性验证 + 边界延伸
    → (4) 二分搜索边界优化 → (5) 音频/文本质量评估
    → (6) 质量过滤 → (7) 说话人识别 → (8) 标点恢复 → (9) 最终 TTS 子集
```

数据逐步过滤: 5,824.7h(VAD 输出) → 5,127.3h(非空转写) → 4,096.2h(边界优化后) → 3,007.8h(质量过滤后) → 2,199.7h(最终 TTS 子集) [Table 2]

### 关键设计选择

#### 1. 为什么选 ASR-based 转写而非 OCR?

[论文原文] 有声书朗读经常与印刷版不一致(删节、改编、版本差异),OCR 对齐会引入系统性错误 [§3.1]。ASR-based 策略与版本无关(edition-agnostic),不依赖源文本,可自动化大规模应用。

#### 2. 句子完整性验证: 为什么需要 ParsBERT 分类器?

[论文原文] VAD 的静音边界不一定与句子边界对齐,一个片段可能在句子中间被截断 [§3.2]。[agent 解读] 这对 TTS 训练尤其有害,因为不完整的句子会破坏韵律模型学习。

**机制**: 对 ParsBERT 微调一个句子完整性二分类器(97.4% F1),训练数据来自 PersianPunc,正例是完整句子,负例通过从句尾删除 1-5 个词或截断最后一个词的 1-5 个字符生成,不完整样本以 3:2 比例过采样 [Appendix C]。当片段被判为不完整时,以 0.1 秒为步长迭代延伸边界,每次重新转写和判断,直到满足完整性条件或达到 5 秒上限 [§3.2]。86.3% 的片段直接通过,13.7% 需要延伸(平均 17.7 步) [Appendix E]。

#### 3. 边界优化: 为什么用二分搜索?

[论文原文] 即使转写准确,音频边界仍可能包含静音、背景噪声或声学伪影,这些会降低 TTS 模型性能 [§3.3, citing Zen et al. 2019; Jung et al. 2024]。

**机制**: 对每个片段的起始和结尾边界独立执行 [Fig 3]:
1. 初始裁剪 3 秒,重新转写
2. 如果转写变化(任何字符差异) → 裁剪过多,需回退
3. 二分搜索迭代缩小裁剪范围,直到转写完全一致
4. 线性微调(0.1 秒步长)精确定位边界

[agent 解读] 这个设计巧妙之处在于用 **ASR 转写的稳定性作为隐式对齐信号**: 如果裁掉的部分不影响转写,说明那部分是静音/噪声;一旦转写变化,说明开始切入语音内容了。81.2% 片段需要起始边界裁剪,50.4% 需要结尾裁剪,总共去除 507.0 小时(11.0%)的不需要内容 [Table 9]。

#### 4. 多维质量评估

**音频质量** (0-100 分) [Table 7]:
- SNR (>20dB: +35)、动态范围、时长(3-15s: +10)、背景音乐检测(inaSpeechSegmenter)、削波比、静音比
- ≥90 高质量 / 75-89 可接受 / <75 排除

**文本质量** (0-1 分) [Table 8]:
- 5 个维度加权: 字符质量(0.25)、长度质量(0.20)、重复得分(0.20)、语言复杂度(0.175)、音素覆盖(0.175)
- 文本过滤阈值: <0.5 排除

[agent 解读] 波斯语特有的字符质量维度(惩罚阿拉伯字母替代品)体现了语言特异性设计。

#### 5. 说话人识别: 为什么需要两阶段?

[论文原文] 约 40% 的有声书缺少叙述者名称,部分书有多个叙述者 [§3.5]。

**机制**: 基于 ECAPA-TDNN 嵌入的两阶段 pipeline:
1. **局部聚类**: 在每本书内,PCA 降维 + 多方法集成(Agglomerative/Spectral/GMM,取最高 silhouette score) → 每个片段的置信度 $c_i = 0.6\tilde{s}_i + 0.4d_i$,低于 0.4 排除 [Appendix F]
2. **全局合并**: 跨书本用加权质心嵌入的凝聚聚类(cosine similarity ≥ 0.85),每次合并要求至少 100 个共享片段防止误合并

Narrator purity 验证: 325 个有元数据的叙述者,中位纯度 1.00,均值 0.869,65.8% 的叙述者 ≥90% 时长分配到单一全局 speaker ID [§4]。

### 训练策略

**注意**: ParsVoice 本身是数据集,不涉及从零训练 TTS 模型。验证实验微调 XTTSv2:

- 仅调 GPT 组件,DVAE 等其他模块冻结 [§5.1]
- 新训练波斯语 BPE tokenizer,向 GPT 词表添加 2,500 个波斯语 token [§5.1]
- 单卡 A100,AdamW (lr=5e-6, wd=1e-2),170K steps,effective batch size 168 [§5.1]

## 实验

| 指标 | 本文 (XTTS+ParsVoice) | Baseline (Real Speech) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| MOS (自然度) | 3.59±0.09 | 4.72† | 90 合成样本,22 评分者 | [Table 3] |
| SMOS (说话人相似度) | 4.03±0.08 | 4.86† | 同上 | [Table 3] |
| Intel. MOS (可懂度) | 4.03±0.08 | — | 同上 | [Table 3] |
| Speaker Sim (ECAPA-TDNN) | 80.0% | — | 42 名未见说话人 | [Table 3] |
| WER (Google ASR) | 10.40% | 6.53% (FLEURS) | 90 样本 vs FLEURS | [Table 4] |
| WER (Whisper 独立) | 17.20% | 13.07% (FLEURS) | 同上 | [Table 4] |
| WER gap (Google) | +3.87pp | — | — | [Table 4] |
| WER gap (Whisper) | +4.13pp | — | — | [Table 4] |

跨系统参考(不同评分者/条件,不可直接比较 [§5.3]):
| 指标 | DeepMine Tacotron2† | DeepMine FastSpeech2† | 出处 |
| --- | --- | --- | --- |
| MOS | 3.94±0.04 | 4.12±0.06 | [Table 3, 原始论文数据] |
| SMOS | — | 3.98±0.11 | [Table 3, 原始论文数据] |

## 局限性

1. **风格单一**: 仅有声书(正式朗读),不含自发对话语音,训练的 TTS 在对话场景可能不自然 [§7]
2. **单一架构验证**: 仅微调 XTTS,受 GPU 限制未训练其他架构或大规模超参搜索 [§7]
3. **评估可比性差**: 与 DeepMine 系统的 MOS 比较使用不同评分者/条件 [§5.3]
4. **性别不平衡**: 有元数据的部分约 67% 男性 / 33% 女性,40% 有声书缺少元数据 [§7]
5. **自动 Speaker ID**: 说话人标签基于 ECAPA-TDNN 自动聚类,非人工标注 [§7]
6. **ASR 残留错误**: 虽经多阶段过滤,仍有少量转写错误 [§7]

## 点评

ParsVoice 是一篇扎实的数据集工程论文。它的核心价值不在于方法论创新,而在于为波斯语 TTS 社区提供了一个**量级跳跃**(86h → 2,200h)的公开资源。

**值得肯定的设计**:
- **二分搜索边界优化**用 ASR 转写稳定性作为对齐代理信号,避免了对 forced alignment 工具的依赖——对于缺乏成熟 alignment 工具的低资源语言是实用的。
- **ParsBERT 句子完整性分类器**解决了 VAD 切分与句子边界不对齐的问题,97.4% F1 说明这个方案可靠。
- pipeline 各阶段的设计动机清晰,论文对每个设计选择给出了理由,且在附录中提供了替代方案的对比(VAD 方法、ASR 系统)。

**需要注意的**:
- MOS 3.6/5 在当前 TTS 水平下不算高(对比英语 SOTA 系统通常 4.0+),但考虑到这是低资源语言且使用通用多语言模型的微调结果,是合理的基线。
- 论文谨慎指出了跨研究 MOS 不可比的问题,这一态度与 KB 中 [[TTS Evaluation]] 页记录的 Responsible Evaluation 框架一致。
- 使用 Google ASR 同时作为 pipeline 组件和评估器存在循环性问题,论文通过引入独立的 Whisper 评估器缓解了这一点(WER gap 相近: 3.87pp vs 4.13pp) [Table 4]。

**局限点评**: 仅有声书风格的数据在对话 TTS 场景下价值有限。波斯语 TTS 的下一步可能需要对话/自发语音数据集。性别不平衡也是公开数据集的常见问题。

## 可复用的 idea

1. **ASR 转写稳定性作为隐式对齐信号**: 二分搜索裁剪边界直到转写不变,适用于任何缺乏 forced alignment 工具的低资源语言
2. **句子完整性分类器 + 迭代边界延伸**: 解决 VAD 切分不完整句子的通用方案,仅需目标语言的 BERT 模型和标点数据集
3. **多维质量评估框架**: 音频 + 文本质量分离评估,各维度加权,阈值可调 —— 可直接迁移到其他语言的 TTS 数据构建
4. **两阶段说话人识别**: 局部(书内)→ 全局(跨书)聚类,带置信度过滤和合并验证 —— 适用于从有声书/播客构建多说话人数据集
5. **循环性评估的缓解**: 当 ASR 同时用于数据构建和评估时,引入独立 ASR 系统交叉验证

---

> [!review] 审阅 (2026-06-04, auto)
> **结论: pass-with-fixes** (0 high, 0 medium, 2 low)
> - (low) traceability-gap: 速查卡片"一句话"中 "25 times larger" 未标出处
> - (low) template-compliance: datasets 字段为空,评估用的 Persian Common Voice / FLEURS 可选列入
> 详见 `_review/ParsVoice-review.yml`

检索命中: [[Speaker Embedding]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[TTS Evaluation]](pending-review), [[Phoneme Representation]](pending-review) | 未命中但可能相关: 无
