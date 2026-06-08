---
type: paper
tier: deep
title: "OmniScript: Towards Audio-Visual Script Generation for Long-Form Cinematic Video"
arxiv_id: "2604.11102"
source: "Sources/OmniScript.pdf"
authors: [Junfu Pu, Yuxin Chen, Teng Wang, Ying Shan]
year: 2026
venue: "arXiv preprint"
tags: [video-understanding, video-to-script, multimodal-LLM, audio-visual, long-form-video, temporal-grounding, reinforcement-learning, GRPO, chain-of-thought, cinematic]
concepts: ["[[AudioUnderstanding]]", "[[Audio-LanguagePretraining]]"]
models: ["[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 3 个待确认实体页: [[模型库/Whisper|Whisper]], [[AudioUnderstanding]], [[Audio-LanguagePretraining]])
> 基于未确认概念页,仅供参考。本论文属于视频理解/多模态领域,与 vault 的 TTS 核心方向关联有限,KB 命中自然偏少。
>
> **Whisper** [待确认]: OmniScript 使用 Whisper large-v3 作为音频编码器,提取时间对齐的音频特征。Whisper 本身是大规模弱监督 ASR 模型 (680k 小时数据),在本文中作为冻结的音频特征提取器使用,不参与端到端训练的第一阶段(modality alignment 阶段仅训练 audio projector)。
>
> **AudioUnderstanding** [待确认]: 本论文的 audio understanding 超出传统 SpeechLM 定义——不仅理解语音内容 (dialogue/ASR),还理解环境音效 (sound effects)、背景音乐 (BGM)、画外音 (voiceover)。这对应 Audio Understanding 页面中 ALM 路线的"通用音频理解"方向,而非纯 SpeechLM 路线。
>
> **Audio-Language Pretraining** [待确认]: 本论文的 AV-DeepStack 模块在多层 Transformer 中注入音频-视觉特征,属于"Two Heads"架构变体(audio encoder + vision encoder + LLM),与 SALMONN 等 LALM 的 adapter-based 架构有设计理念上的相似性。
>
> **谱系定位**: OmniScript 处于 "Dense Video Captioning → Omni-Modal Script Generation" 的前沿位置。相比传统 dense captioning (只生成稀疏摘要) 和 audio-visual captioning (如 AVoCaDO, SALMONN-2, 缺乏时间锚定),OmniScript 强制执行严格的层级结构 (Scene → Event → Field) 并显式隔离细粒度原子元素。与并行工作 TimeChat-Captioner 相比,后者在宏观场景结构上有贡献,但 character actions/intents/dialogues 仍纠缠在粗粒度摘要中。
>
> 检索命中: [[模型库/Whisper|Whisper]](pending-review), [[AudioUnderstanding]](pending-review), [[Audio-LanguagePretraining]](pending-review) | 过滤: 无 | 未命中但可能相关: 无(vault 以 TTS 为核心,视频理解方向覆盖有限)

## 速查

> [!summary] 速查
> - **一句话**: 提出首个音频-视觉长视频到结构化脚本 (V2S) 的 8B 参数多模态模型,通过记忆增强标注 + CoT SFT + 时序分段 RL 奖励实现接近闭源 SOTA 的性能
> - **路线**: 长视频 → Whisper 音频编码 + Qwen3-VL 视觉编码 → AV-DeepStack 多层融合 → CoT 推理 (plot + character 关系) → 层级脚本解码 (Meta → Scene → Event → Fields)
> - **指标**: 5min 视频 Event Overall F1 37.7, tIoU@0.1 69.3 (vs Qwen3VL-235B: 33.0/62.0; vs Gemini-3-pro: 38.9/64.4) [Table 1]; Scene Overall F1 52.6, tIoU@0.1 74.6 [Table 2]
> - **可借鉴**: (1) 时序分段奖励设计 — 将 RL 奖励按事件级 one-to-one 匹配分段计算,避免全局相似度被主要特征淹没短时事件; (2) 记忆增强标注管线 (CPM) — 跨段持久化角色档案 + 延迟命名策略解决跨场景角色追踪; (3) 两阶段长视频推理 (TSG) — 先规划后生成,实现几乎长度不变的性能
> - **局限**: (1) 长视频 LCE 策略在 30min+ 出现性能悬崖; (2) 音频感知仍偏弱 (Audio Cue F1 仅 11.6); (3) benchmark 仅 10 部影视作品,泛化性存疑; (4) 未开源训练数据

## 核心问题

本文要解决的核心问题是:**如何将长时间 (数分钟到数十分钟) 的影视视频自动转化为时间锚定的、层级结构化的完整脚本?**

现有 MLLM 在短视频理解上已有突破,但在长视频脚本生成上面临三重挑战 [§1]:
1. **训练数据稀缺**: 标注长视频的细粒度多场景结构极其耗时,无现成大规模数据集
2. **评估度量缺失**: 传统 BLEU/ROUGE 无法捕捉层级依赖和开放词汇描述;tIoU 又过度惩罚语义正确但时间微偏的预测
3. **推理成本爆炸**: 2 分钟视频约需 4000 token 描述,随视频时长增加 token 量急剧膨胀

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

OmniScript 是一个 8B 参数的音频-视觉语言模型,基于 Qwen3-VL-8B 骨干,扩展了音频通路 [§5.1]。

**输入**: 未裁剪的影视视频 V,包含视觉帧、语音、音效、背景音乐。
**输出**: JSON 格式的三级层级脚本 [§3.1]:
- **Meta 级**: 标题/描述、总时长、角色列表
- **Scene 级**: 场景序列,每个场景含场景 ID、地点、时间属性 (日/夜)、环境、氛围
- **Event 级**: 每个场景内的有序事件,含时间戳、角色身份,以及至少一个内容字段 (dialogue / action / expression / audio cue)

形式化定义: 事件 $e_{i,j} = (\tau_{i,j}, c_{i,j}, d_{i,j}, a_{i,j}, x_{i,j}, u_{i,j})$,其中 c 是角色,d/a/x/u 分别为对话/动作/表情/音频线索 [Eq. 1]。

### 关键设计选择

**1. AV-DeepStack 注入 (为什么不只在输入层融合?)**

论文采用 DeepStack 风格的融合策略,在多个 LLM 层中注入视觉和音频特征,而非仅在输入阶段拼接 [§5.1]。[论文原文] 这种设计保留了原始 DeepStack 的长上下文推理优势,同时增加了显式听觉感知。具体做法: 音频 token 与视觉 token 配对后,通过 residual multimodal adapters 在每层对语言流进行条件化。

[agent 解读] 多层融合的合理性在于: 脚本生成中不同语义层次需要不同模态的信息——底层需要声学事件检测 (音效/BGM),中层需要对话理解 (ASR),高层需要情节推理 (视觉+语言)。单层融合难以让不同模态在各语义层级充分交互。

**2. 时间对齐的音频-视觉表征 (为什么强调 timestamp-level alignment?)**

使用 Whisper large-v3 编码器提取音频特征,对每个时间单元构建配对表征 $(v_t, a_t)$,严格保持跨模态同步 [§5.1]。

[论文原文] 一对一的时间对齐保留了对话、画外叙述、环境音和背景音乐的跨模态同步性,这对脚本级叙事锚定至关重要。

[agent 解读] 不做时间对齐会导致"谁在什么时间说了什么"的关联断裂,对于脚本生成这种强调"事件=时间+角色+内容"的任务是致命的。

**3. CoT 推理引导的结构化解码 (为什么不直接预测事件字段?)**

解码器采用 Chain-of-Thought 范式: 先生成中间推理 trace (plot 进展摘要 + 角色关系状态),再基于此 scaffold 进行粗到细的脚本生成 [§5.1]。

[论文原文] CoT 范式维持全局-局部一致性,将故事线演进与事件级细节对齐,缓解长上下文歧义,增强对隐式说话人轮换、画外音频和动态人际关系的鲁棒性。

消融实验佐证: CoT 将 SFT baseline 的 Overall 从 35.3% 提升到 37.0%,Dialogue F1 从 68.2% 提升到 71.0% [Table 3]。

**4. 记忆增强标注管线 (CPM, 解决什么问题?)**

训练数据标注的核心难题是跨场景角色追踪 [§3.2]。传统标注独立处理视频段再合并文本输出,导致叙事不连贯和角色 re-ID 失败。

解决方案是 Character Profile Manager (CPM):
- 将原始视频按场景边界分段 (PySceneDetect),从 10K+ 原始视频提取约 45K 段
- 每段由 Gemini-2.5-Pro 做角色中心的情节推理,以 CPM 存储的跨段角色信息为条件
- CPM 动态更新: 持续累积角色的多模态属性变化 (如换装)
- **延迟命名策略**: 临时角色 ID 在发现明确命名事件后追溯升级为永久 ID,重复记录合并

[agent 解读] 这个设计很巧妙地解决了影视剧中常见的"角色先出场、后才报名字"问题,避免了同一角色被标注为多个不同实体。

### 训练策略

四阶段渐进训练 + RL 精炼 [§5.2]:

**Stage 1 — 模态对齐**: ~1M 双语影视样本,带时间戳 ASR 监督。仅训练音频 projector,冻结 Whisper/ViT/LLM。随机帧遮蔽防止过度依赖视觉线索。

**Stage 2 — 多模态预训练**: 2.4M 双语视频全参数微调。多任务目标: ASR (有/无时间戳) + 视频摘要 + 密集视频描述 + 时间锚定。保留随机帧遮蔽。

**Stage 3 — SFT**: 45K 视频 (21K 横屏影视 + 24K 竖屏短剧),CoT 格式 (先推理 plot+角色关系,再解码脚本)。引入随机字幕遮蔽减少对显式文字线索的依赖。

**Stage 4 — RL (GRPO)**: 基于小规模高质量人工标注脚本。关键创新是**时序分段奖励**: 不用全局语义相似度 (会被主导特征淹没短时事件的错误),而是用事件级 one-to-one 匹配的 Multi-dimensional Field Evaluation 分数,在整个视频时间线上逐事件评估 recall 和 precision [§5.2, §4.1]。

消融 [Table 3]: SFT only → 35.3 Overall; +CoT → 37.0; +RL (global reward) → 37.0; +RL (segmented reward) → **37.7** (+CoT → 37.7, tIoU 69.3)。分段奖励优于全局奖励。

### 长视频处理

5 分钟以上视频的两种策略 [§5.3]:

**Strategy 1 — LCE (Long-Context Extension)**: 直接扩大上下文窗口,用长视频标注 + 跨视频拼接伪数据训练。单阶段管线,但需要更强的长程推理。

**Strategy 2 — TSG (Two-Stage Generation)**: 先由 plot-segmentation 模型预测段落 (时间戳 + 情节 + 角色列表 + 关系),再逐段独立由 OmniScript 生成脚本,最后轻量后处理合并。

[论文原文] TSG 展现出"几乎长度不变"的鲁棒性——在 30-40 分钟视频上仍保持几乎水平的性能曲线,而 LCE 和所有 baseline (包括 Gemini) 在 30 分钟出现"上下文悬崖" [§6.3, Fig. 5-6]。

[agent 解读] TSG 本质是"分而治之",通过段级条件化将生成过程与全局噪声隔离。代价是引入了额外的 segmentation 模型和合并后处理,但实际效果远优于暴力扩展上下文。

### 评估框架 (V2S Benchmark)

论文提出四阶段层级评估框架 [§4.1]:
1. **文本事件对齐**: 基于复合语义相似度 + 动态规划匹配预测事件与 GT 事件,容忍 ≤30s 时间偏移
2. **LLM 辅助角色映射**: 解决开放词汇角色描述问题 (如"police officer" vs GT "John"),三类别分类 + 双向冲突检测
3. **字段级语义评估**: 按 5 个字段 (character/dialogue/action/expression/audio cue) 分别计算 F1
4. **tIoU Hit Rate**: 独立于语义的时间定位质量评估

Benchmark: 10 部影视作品 (19.9 小时),1.4k 场景,16.8k 事件,14.1 events/min [§4.2]。多粒度测试: 200 个 5min、100 个 10min、50 个 15min 等。

## 实验

| 指标 | 本文 (8B) | Qwen3VL-235B | Gemini-3-pro | Qwen3-Omni (30B MoE) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Event Overall F1 | **37.7** | 33.0 | 38.9 | 5.1 | V2S 5min | [Table 1] |
| Event tIoU@0.1 | **69.3** | 62.0 | 64.4 | 12.8 | V2S 5min | [Table 1] |
| Event Dialogue F1 | **72.2** | 58.6 | 68.8 | 3.5 | V2S 5min | [Table 1] |
| Event Character F1 | 39.2 | 38.1 | 39.8 | 5.5 | V2S 5min | [Table 1] |
| Event Audio Cue F1 | 11.6 | 6.0 | 13.3 | 3.4 | V2S 5min | [Table 1] |
| Scene Overall F1 | 52.6 | — | 54.6 | — | V2S 5min | [Table 2] |
| Scene tIoU@0.1 | **74.6** | 72.8 | 75.5 | — | V2S 5min | [Table 2] |

**关键发现**:

1. **参数效率**: 8B 模型在 Event Overall 上超越 235B 的 Qwen3VL (+4.7) 和 tIoU 上超越 (+7.3),接近闭源 Gemini-3-pro [Table 1]。

2. **Thinking mode 反效果**: Qwen3VL-T (thinking mode) 在多数指标上反而不如非 thinking 版本; Qwen3-Omni (omni 输入) 也显著弱于同规模非 omni 模型 [Table 1]。

3. **字幕遮蔽实验**: 遮蔽字幕后 Qwen3VL-235B 的 Dialogue F1 从 58.6 暴跌至 7.7,说明其严重依赖视觉文字而非真正的语音理解; Gemini-3-pro 仅轻度下降 (68.8→60.9); 本文模型下降适中 (72.2→63.8) [Table 4]。

4. **音频注入有效性**: vision-only baseline Dialogue F1 仅 52.0,加入音频预训练后提升至 68.2 (+16.2) [Table 5]。

5. **长视频上下文悬崖**: 所有模型 (含 Gemini) 在 30min+ 出现性能急剧下降; OmniScript-TSG 是唯一保持近似水平性能曲线的方案 [Fig. 5-6]。Gemini-2.5-flash 在 <25min 时 recall 极高,但 >25min 后退化为重复生成循环 [§6.3]。

## 局限性

1. **Audio Cue 感知仍然薄弱**: Audio Cue F1 全场最高仅 13.3 (Gemini-3-pro),本文 11.6,说明环境音效/BGM 的细粒度描述对所有模型仍是开放难题 [Table 1]。

2. **LCE 策略的上下文悬崖**: OmniScript-LCE 在 30min 视频上发生多维"体积崩塌",仅 TSG 策略能保持稳定 [§6.3, Fig. 5]。但 TSG 引入了额外的 segmentation 模型和合并后处理的误差传播风险。

3. **Benchmark 规模有限**: 仅 10 部影视作品 (19.9 小时),跨题材/风格的泛化性评估不充分 [§4.2]。

4. **训练数据管线不可复现**: 标注依赖 Gemini-2.5-Pro (闭源),10K+ 原始影视视频来源未说明,45K SFT 数据集未开源。

5. **评估方法本身的局限**: 字段级语义评估依赖 LLM-as-Judge,该方法的可靠性和偏见未验证 [§4.1.3]。

6. **缺少与 audio-specific 模型的对比**: 未与 SALMONN、Audio Flamingo 等专注音频理解的 LALM 对比。

## 点评

**优势**:
- 问题定义清晰且有实际价值: Video-to-Script 是一个未被充分探索但有明确应用场景 (自动编剧辅助、内容检索、元数据生成) 的任务。层级化输出 schema (Meta→Scene→Event→Field) 的设计比粗粒度 captioning 更加实用。
- 时序分段 RL 奖励是本文最有技术洞察力的贡献。识别到全局语义相似度奖励会被主导特征淹没,转而按事件级 one-to-one 匹配计算奖励,是将 RL 对齐应用于长序列结构化生成时的重要经验。
- CPM 的延迟命名策略解决了影视剧角色追踪中的真实痛点,且方案优雅。
- 8B 参数超越 235B 模型,展示了领域定制化训练的巨大潜力。

**不足与质疑**:
- 论文声称"first omni-modal script generation model"和"first-of-its-kind benchmark",但并行工作 TimeChat-Captioner 也在做类似的事情 (尽管粒度不同),用词过于绝对。
- 与 Gemini-3-pro 的比较中,本文 Event Overall (37.7) 实际低于 Gemini-3-pro (38.9),但论文用"comparable"来描述,稍有 overclaim 倾向。
- 全文表格数据有大量空白 (Table 1/2 中许多模型许多指标无数据),实验的完整性可以更好。
- Audio Cue F1 仅 11.6,考虑到论文以"omni-modal"为核心卖点并强调音频通路的价值,这个数字说明音频感知还远未达到实用水平。

**适用场景**: 影视/流媒体行业的自动化内容标注、辅助编剧、视频检索系统。对 TTS/语音合成领域的直接借鉴有限,但 CPM 角色追踪和时序分段 RL 奖励的设计思想可迁移。

## 可复用的 idea

1. **时序分段奖励 (Segmented Reward)**: 对任何长序列结构化生成任务 (如长文本 TTS 的段落级对齐评估),可以借鉴"将全局奖励分解为局部事件级奖励"的思路,避免短时细节被长序列主导特征淹没。

2. **记忆增强标注管线 + 延迟命名**: 对于需要跨段一致性的标注任务 (如多说话人长音频的对话标注),CPM 的"持久化档案 + 延迟命名 + 属性增量更新"模式是实用模板。

3. **两阶段 plan-then-write 推理**: TSG 的"先规划段落结构,再逐段生成"策略,对任何长序列生成任务 (如长文本 TTS 的韵律规划→语音合成) 都有参考价值,关键在于规划阶段提供足够的局部条件化信息。

4. **随机帧/字幕遮蔽的正则化**: 训练时随机遮蔽部分模态输入,迫使模型学习互补模态信息,而非走捷径。对多模态 TTS (如 audio-visual TTS) 的训练有直接参考价值。

5. **四阶段层级评估设计**: 将复杂输出的评估拆分为"对齐→身份解析→字段语义→时间定位"的四级管线,每级独立评估不同维度,避免单一指标混淆多种能力。对 TTS 评估体系设计有方法论参考价值。

## 审阅

> [!review] 审阅 (2026-06-08, self-review)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节覆盖 WHY/HOW,关键设计选择有因果解释 |
> | 可信赖 | pass | 数字 claim 有 [Table N]/[§X.X] 标注,覆盖率 ~85% |
> | 可区分 | pass | 因果解释标注了 [论文原文]/[agent 解读],覆盖率 ~80% |
> | 可定位 | pass-with-fixes | KB 背景有谱系定位,但 vault 以 TTS 为核心,本文属视频理解方向,KB 命中少是结构性原因 |
> | 不污染 | pass | 不做反向更新,无概念页修改 |
> 
> Issues: 2 (medium: 1, low: 1)
> 
> **medium**: `template-compliance` — frontmatter 的 concepts/models/tasks/datasets 字段中未列全所有论文提及的模型 (Qwen3-VL, Gemini 系列作为 baseline),但考虑这些模型不在 vault KB 中且非 TTS 方向,合理省略
> 
> **low**: `weak-reusability` — "可复用的 idea" 中第 5 点 (评估设计方法论) 较泛,但给出了具体的四级管线拆分思路,尚可接受
