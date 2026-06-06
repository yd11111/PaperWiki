---
type: paper
tier: deep
title: "SALMONN: Towards Generic Hearing Abilities for Large Language Models"
arxiv_id: "2310.13289"
source: "Sources/SALMONN.pdf"
authors: [Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Chao Zhang]
year: 2024
venue: "ICLR 2024"
tags: [speech-LM, audio-understanding, multimodal, Q-Former, LoRA, instruction-tuning, dual-encoder, emergent-abilities, activation-tuning, LALM]
concepts: ["[[SpeechLanguageModel]]", "[[ModalityAdaptationforSpeechLLM]]", "[[AudioUnderstanding]]", "[[Audio-LanguagePretraining]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: SALMONN 属于 latent-representation-based Speech-LLM integration 路线 [Speech-LLMIntegrationTaxonomy],使用连续编码器特征而非离散 audio tokens 送入 LLM。在 KB 的 ALM 分类中被归为 "Two Heads" 架构 (Audio encoder + Text encoder + LM) [Audio-LanguagePretraining]。与 audio-token-based 路线 (SpeechGPT, TWIST, Spirit-LM) 的核心区别在于: SALMONN 不将语音离散化为 tokens,而是通过 Q-Former adapter 将连续表征映射到 LLM 输入空间 [ModalityAdaptationforSpeechLLM]。
>
> **已有认知**: KB 中已有 Modality Adaptation 页面记录了三种主要适配方法 (Conv downsampling, CTC compression, Q-Former),SALMONN 被列为 Q-Former 的代表系统。Q-Former 源自 BLIP-2,通过 learned queries + cross-attention 将变长输入映射为固定长度输出,在 Yu et al. (2024) 实验中性能排序为 Q-Former > CTC > Conv。SALMONN 使用的 Whisper encoder 已有模型页,记录了其 680k 小时弱监督训练和作为 SpeechLM 最流行 speech encoder 的地位。
>
> **创新判断**: SALMONN 的独特性在于 (1) dual encoder (Whisper + BEATs) 同时覆盖语音和非语音音频; (2) window-level Q-Former 的时序保持改进; (3) activation tuning 解决 task over-fitting 的方法论贡献。KB 中尚无专门讨论 dual encoder 设计或 activation tuning 的概念页。
>
> 检索命中: [[SpeechLanguageModel]]✓, [[ModalityAdaptationforSpeechLLM]][待确认], [[AudioUnderstanding]][待确认], [[Audio-LanguagePretraining]][待确认], [[Speech-LLMIntegrationTaxonomy]][待确认], [[Whisper]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个同时理解语音+音频事件+音乐的多模态 LLM,通过双编码器 (Whisper+BEATs) + window-level Q-Former + activation tuning 实现通用听觉能力和跨模态涌现能力
> - **路线**: Audio → Whisper encoder + BEATs encoder (冻结) → frame-level concat → Window-level Q-Former (可训练) → augmented audio tokens → Vicuna 13B + LoRA (LLM 冻结,LoRA 可训练) → text response
> - **指标**: ASR WER 2.1/4.9/10.0 (LibriSpeech clean/other, GigaSpeech); En2Zh BLEU4 33.1; AAC METEOR 24.0; Story diversity 82.57 (activation tuning 后 FR 从 0% → 100%); SAC accuracy 0.50 (FR 从 4% → 73%) [Table 3]
> - **可借鉴**: (1) LoRA scaling factor discount 发现跨模态涌现能力的诊断工具; (2) 12-sample activation tuning 作为极低成本的 task over-fitting 解法; (3) dual encoder 互补设计思路 (语音专精 + 通用音频语义)
> - **局限**: 仅支持理解不支持生成; 13B LLM backbone 推理开销大; activation tuning 仅用 12 个样本,泛化性存疑; OSR 受限于 Whisper 本身不支持重叠语音

## 核心问题

**想解决什么**: 当时 (2023 年 10 月) 的多模态 LLM 要么只处理语音 (如 SpeechGPT),要么只处理音频事件 (如 LTU),要么通过外部管线拼接 (如 AudioGPT)。没有一个端到端模型能同时"听懂"语音、音频事件和音乐,并在此基础上进行跨模态推理 [§1]。

**为什么难**: 三种声音的建模需求互相矛盾 — 语音需要精细的时序对齐 (ASR),音频事件需要高层语义 (captioning),音乐需要节拍/旋律理解。更关键的问题是: 即使模型架构支持,instruction tuning 会导致 task over-fitting,模型退化为只做 ASR/captioning 而丧失 LLM 的跨模态推理能力 [§3.2]。

**怎么切入**: (1) 用两个互补编码器分别捕获语音和非语音信息; (2) 用 window-level Q-Former 保持时序分辨率; (3) 用 activation tuning 恢复被 instruction tuning 抑制的涌现能力。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SALMONN 由四个组件构成 [Fig 1]:

1. **Whisper encoder (冻结)**: Whisper-Large-v2 的 encoder 部分,50Hz 输出帧率。训练于大量弱监督 ASR 数据,其特征适合建模语音内容,同时包含背景噪声信息 [§3.1, Gong et al. 2023a]。

2. **BEATs encoder (冻结)**: 自监督音频编码器,通过迭代式 "tokenize→mask→predict" 训练提取高层非语音音频语义。与 Whisper 恰好互补 — Whisper 强于语音,BEATs 强于非语音事件 [§3.1]。两个编码器输出帧率相同 (50Hz),逐帧沿特征维度拼接:
   ```
   Z = Concat(Encoder_whisper(X), Encoder_beats(X))   [Eq. 1]
   ```

3. **Window-level Q-Former (可训练)**: 对 BLIP-2 Q-Former 的改进。原始 Q-Former 将整个序列压缩为固定数量 N 个 token,适用于图像但不适用于变长音频。SALMONN 将编码器输出 Z 分割为 L 帧一组的窗口,每个窗口独立用 Q-Former 压缩为 N 个 token [Eq. 2]:
   ```
   H = [Q-Former(Q, Z_l)]  for l = 1, ..., ⌈T/L⌉
   ```
   设置 N=1 (每窗口一个 query), L=17 (~0.33s),30 秒音频产生 88 个 token [§4.1]。

4. **Vicuna 13B + LoRA (LLM 冻结, LoRA 可训练)**: LoRA 应用于 self-attention 的 query/value 矩阵,rank=8, scaling factor=4.0。LoRA 的作用不仅是参数高效微调,更是对齐 LLM 的 augmented input space 和 output space [§3.1]。

**总可训练参数**: ~33M (~0.24% of total) — 仅 Q-Former + LoRA [§4.1]。

### 关键设计选择

**为什么用双编码器而非单一编码器?** [论文原文] Whisper 虽然在背景噪声下也有一定能力 (Gong et al. 2023a),但其特征本质上是为语音优化的。BEATs 通过自监督迭代学习提取高层音频语义,两者特征互补。[agent 解读] 这是一个务实的工程选择: 与其训练一个同时覆盖语音+音频的新编码器 (成本极高),不如直接组合两个已有的 SOTA 编码器。代价是模型增大,但两个编码器都冻结,不增加可训练参数。

**为什么 window-level 而非 sequence-level Q-Former?** [论文原文] 原始 Q-Former 将整个序列映射为固定 N 个 token,这在 (1) 可变长度音频下效率低,(2) 丢失时序信息,对 ASR 这类需要单调对齐的任务不利。Window-level 的设计保证输出 token 数与输入长度成正比,且输出与输入保持单调对齐 [§3.1]。[agent 解读] 本质上 window-level Q-Former 是介于 Conv downsampling (纯机械) 和 sequence-level Q-Former (全局注意力) 之间的折中: 窗口内用 cross-attention 做智能压缩,窗口间保持时序结构。

**为什么 N=1 (单 query)?** [论文原文] 文中未详细解释。[agent 解读] N=1 意味着每 0.33s 窗口只产生一个 token,相当于将 50Hz 降到 ~3Hz,是非常激进的压缩。这可能是为了控制送入 LLM 的 token 数 (30s → 88 token),避免超过 LLM 的上下文窗口。但这也意味着同一窗口内的不同事件会被压缩到单个向量中。

### 训练策略

三阶段训练 [§3.2]:

**Stage 1 — Pre-training**: Q-Former + LoRA 在 ASR (LibriSpeech 960h + GigaSpeech 220h) + Audio captioning (WavCaps 2800h + AudioCaps + Clotho) 上训练。目的是建立基础的听觉-文本对齐,选择 ASR 和 AAC 因为它们包含语音和非语音的关键信息,且不需要复杂推理 [§3.2]。

**Stage 2 — Instruction Tuning**: 12 类任务共 ~2.3M 样本 (~4400h) [Table 1],包括 ASR, AST, AAC, PR, ER, MC, OSR, SV, GR, SQA, AQA, MQA。SQA/AQA/MQA 的问答对由 ChatGPT 基于文本标注生成 [§4.2]。

**Stage 3 — Activation Tuning** (本文核心贡献): 仅 12 个 story 样本,12 步训练 [§4.2]。

### Task Over-fitting 问题与分析

[论文原文] 仅经过 Stage 1+2 训练的 SALMONN 在 trained tasks 上表现良好,但几乎无法执行 untrained 的跨模态任务。模型甚至会违反 instruction,对任何输入都输出 ASR 转写 [§3.2]。

**贝叶斯分析** [§3.2, Eq. 3]: 作者将 task over-fitting 归因于 intrinsic conditional LM P_Λ(Y|X) 的偏差。由于训练数据中 ASR/AAC 等任务的输出简单且确定性强,P_Λ(Y|X) 被偏向短的、与输入强对齐的序列 (如转写)。即使给新的 instruction I',P_Λ(Y|X, I') 也很小,因为 P_Λ(Y|X) 已被 ASR 主导。

[agent 解读] 这个分析的本质是: instruction tuning 的任务分布高度不均衡 — ASR/AAC 数据量大且输出格式固定,其他任务数据量小且输出多样。LLM 学到的"默认行为"就是做 ASR,instruction 被忽略。这与 NLP 中 instruction tuning 的 task diversity 要求一致,但在跨模态场景下更严重,因为跨模态的训练数据本身就稀缺。

### Activation Tuning 的原理

**核心发现**: 降低 LoRA scaling factor (从 4.0 → ~2.0) 可以突然激活跨模态推理能力 [Fig 3]。[论文原文] 这证实了 intrinsic conditional LM 嵌入在 LoRA 中,因为只有 Q-Former 和 LoRA 在训练中更新。降低 LoRA 权重等价于削弱这个偏倚的条件 LM,让 backbone LLM 的原始推理能力重新显现 [§5.2]。

**问题**: 直接降低 LoRA scaling 虽能激活推理能力,但会严重损害 trained tasks (ASR WER 飙升) [Fig 3a]。

**解法**: 用降低 LoRA scaling 后模型自生成的 story responses 作为少样本训练数据,进行第三阶段 activation tuning。这是一种自监督方法: 模型在弱化 LoRA 时生成多样化响应,再用这些响应教会正常 LoRA 的模型 [§3.2]。

**效果 [Table 3]**: Activation tuning 后 ASR/PR 几乎不变 (WER 2.1→2.1, PER 4.2→4.2),而 SQQA FR 从 29%→98%, Story FR 从 0%→100%, SAC FR 从 4%→73%。

**为什么只需 12 个样本?** [论文原文] 未深入解释。[agent 解读] activation tuning 的目标不是教模型新知识,而是调整 P_Λ(Y|X) 的分布,使其不再极端偏向 ASR。12 个多样化的 story 样本足以打破这种偏倚,因为核心推理能力本就存在于 backbone LLM 中,只需被"激活"。

### 对比实验: 什么样的数据能激活?

[Table 4] 对比了不同 activation tuning 数据:
- **Story (完整方案)**: 效果最好,repeat rate 最低 (0.1%)
- **QA (Long)**: 可以激活但 repeat rate 高 (4.6%),因为 QA 答案多样性不如 Story
- **ASR (Long)**: 完全无法激活,因为 ASR 任务本身强化了偏倚分布
- **Story (Text-based)**: 无法激活,因为修改的是 P(Y|T_x, I) 而非 P(Y|X, I) — 文本和音频走不同路径
- **Story (LoRA only)**: 可以激活,说明 Q-Former 不是 activation 的必要更新模块

## 实验

| 指标 | 本文 (w/ AT) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER | 2.1 / 4.9 / 10.0 | Whisper: 2.2 / 5.1 / 9.2 | LibriSpeech clean/other, GigaSpeech | [Table 3a] |
| BLEU4 (En2Zh) | 33.1 | CoVoST2: 38.9 | CoVoST2 | [Table 3a] |
| METEOR (AAC) | 24.0 | WavCaps: 25.0 | AudioCaps | [Table 3a] |
| SPIDEr (AAC) | 40.3 | WavCaps: 48.5 | AudioCaps | [Table 3a] |
| PER | 4.2 | WavLM: 3.1 | LibriSpeech clean | [Table 3a] |
| Emotion Acc | 0.69 | SOTA: 0.81 | IEMOCAP Session 5 | [Table 3a] |
| SV Acc | 0.94 | - | VoxCeleb1 | [Table 3a] |
| SQQA Acc (FR) | 0.41 (0.98) | Whisper+Vicuna: 0.77 (1.00) | WikiQA | [Table 3b] |
| SF Acc (FR) | 0.41 (0.99) | Whisper+Vicuna: 0.46 (1.00) | SLURP | [Table 3b] |
| Story Diversity (FR) | 82.57 (1.00) | - | AudioCaps | [Table 3b] |
| SAC Acc (FR) | 0.50 (0.73) | - | In-house | [Table 3b] |

**关键观察**:
1. Level 1 (trained tasks): SALMONN 接近但普遍未超过专用模型/SOTA [Table 3a]。ASR 接近 Whisper,AAC 接近 WavCaps,这验证了端到端多任务模型的可行性但也显示 jack-of-all-trades 的代价。
2. Level 2 (untrained NLP tasks): Activation tuning 的提升是决定性的 — FR 从近乎 0 跃升到接近 100% [Table 3b]。但绝对准确率仍低于 Whisper+Vicuna cascade,说明端到端路线在语义理解上尚有差距 [Appendix B]。
3. Level 3 (novel tasks): Story 和 SAC 是全新任务,cascade 系统无法完成,SALMONN 首次展示了可行性。

## 局限性

1. **仅理解不生成**: SALMONN 只输出文本,不输出语音/音频。后续 SALMONN-E 等工作尝试补充生成能力 [§H]。
2. **Trained tasks 未达 SOTA**: 在 ASR/AAC/ER 等任务上均低于对应的专用模型,多任务学习的 trade-off 明显 [Table 3a]。
3. **SQQA 知识遗忘**: LoRA 的跨模态适配可能导致 LLM "忘记"部分文本知识,SQQA 准确率 (0.41) 远低于 Whisper+Vicuna (0.77) [Appendix B]。
4. **OSR 受限**: Whisper 本身不支持重叠语音,SALMONN 在此任务上 WER=23.0 vs 专用模型 7.6 [Table 3a]。
5. **Activation tuning 仅 12 样本**: 虽然论文视之为优势 ("few-shot"),但如此少的样本能否泛化到更多样的跨模态任务场景存疑。
6. **评估有限**: Level 3 任务使用 GPT-3.5 自动评估 [Appendix E],SAC 使用 in-house 数据,可复现性受限。

## 点评

**方法论贡献 > 性能贡献**: SALMONN 的最大价值不在于各任务的绝对性能 (均未达 SOTA),而在于 (1) 证明了单一端到端模型可以同时处理语音+音频+音乐三类声音,(2) 发现并分析了 task over-fitting 问题,(3) 提出了极低成本的 activation tuning 解法。

**Task over-fitting 分析的通用性**: 虽然论文聚焦于音频-LLM,但 task over-fitting 的贝叶斯分析 (P_Λ(Y|X) 偏倚) 适用于任何多任务 instruction tuning 场景。当主导任务的输出分布简单且确定性强时,模型的涌现能力会被抑制。这个洞察对后续的 omni-model 训练策略有普遍参考价值。

**Dual encoder 的取舍**: 拼接两个冻结编码器是实用主义的做法,但也意味着模型容量的一半 (BEATs) 在纯语音场景下是浪费的。后续工作 (如 Qwen-Audio) 转向单一但更强的编码器,可能是更优的方向。

**历史定位**: SALMONN 发表于 2023 年 10 月 (ICLR 2024),处于 Audio-LLM 爆发的早期。它与 LTU, Qwen-Audio 等同期工作共同定义了 latent-representation-based 音频理解 LLM 的基本范式: 冻结编码器 + 可训练 adapter + 冻结/轻量微调 LLM。这个范式至今仍是主流。

## 可复用的 idea

1. **LoRA scaling factor 作为诊断工具**: 降低 LoRA scaling 可以揭示模型内部是否存在被抑制的能力。如果降低 scaling 后涌现新能力,说明 LoRA 学到了偏倚分布而非新知识。这个技巧可用于调试任何使用 LoRA 的多任务模型。

2. **自生成数据做 activation tuning**: 用模型在"弱化适配器"状态下生成的响应训练"正常适配器"的模型。本质是 self-distillation 的变体,但操作极简 (仅需改 scaling factor → 生成 → 训练),可推广到其他 task over-fitting 场景。

3. **Window-level Q-Former**: 在需要保持时序信息的任务 (如 TTS prompt encoding) 中,window-level 处理比 sequence-level 更适合变长输入,在压缩效率和时序保真之间取得平衡。

4. **互补编码器拼接**: 当单一编码器难以覆盖所有需求时,拼接多个专精编码器 (冻结) 是快速原型验证的有效策略。关键是编码器的输出帧率必须对齐。

## 审阅

*(待独立 reviewer dispatch)*
