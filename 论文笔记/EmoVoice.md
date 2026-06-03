---
type: paper
tier: deep
title: "EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting"
arxiv_id: "2504.12867"
source: "Sources/EmoVoice.pdf"
authors: [Guanrou Yang, Chen Yang, Qian Chen, Ziyang Ma, Wenxi Chen, Wen Wang, Tianrui Wang, Yifan Yang, Zhikang Niu, Wenrui Liu, Fan Yu, Zhihao Du, Zhifu Gao, Shiliang Zhang, Xie Chen]
year: 2025
venue: "ACM MM 2025"
tags: [TTS, emotion, LLM-based, instruction, natural-language-description, controllability, dataset, evaluation]
concepts: ["[[Emotion Control in TTS]]", "[[LLM-based TTS]]", "[[Natural Language Description for TTS]]", "[[Instruction-Guided Speech Synthesis]]", "[[Semantic vs Acoustic Tokens]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]", "[[模型库/CosyVoice|CosyVoice]]"]
tasks: ["[[Instructed Speech Generation]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[LLM-based TTS]], [[模型库/CosyVoice 2|CosyVoice 2]], [[Semantic vs Acoustic Tokens]]; 3 个待确认实体页: [[Emotion Control in TTS]], [[Natural Language Description for TTS]], [[Instruction-Guided Speech Synthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: EmoVoice 位于两条演进线的交汇处:
> 1. **情感控制演进线**: Emotion embedding (2021) → 多尺度层级建模 (MsEmoTTS, 2022) → DPO 对齐 (Emo-DPO, 2024) → 零样本情感 (EmoSphere++, 2024) → **LLM 自由文本情感 (EmoVoice, 2025)**。EmoVoice 是该线的最新节点,首次将 freestyle NL emotion description 作为一等公民注入 LLM-based TTS。
> 2. **NL Description 演进线**: PromptTTS (5 属性模板, 2023) → InstructTTS (三阶段, 2024) → CosyVoice (统一指令, 2024) → CosyVoice 2 (streaming + instruction, 2024)。EmoVoice 与 CosyVoice 2 共享 Qwen2.5 backbone 和 CosyVoice semantic tokens,但专注于情感维度的 fine-grained NL control,是 NL Description 路线在情感子问题上的垂直深化。
>
> **已有认知**: KB 中 [[LLM-based TTS]] (confirmed) 已系统梳理了 LLM-based TTS 的核心设计(离散 token 表示、两阶段生成、in-context learning);[[模型库/CosyVoice 2|CosyVoice 2]] (confirmed) 记录了 FSQ-SenseVoice tokenizer 和双向流式方案;[[Semantic vs Acoustic Tokens]] (confirmed) 解释了 CosyVoice 监督式 semantic tokens 的工作原理。[[Emotion Control in TTS]] [待确认] 整理了从 emotion embedding 到 DPO 的方法谱系。
>
> **创新判断**: 相比 KB 中已有知识,EmoVoice 的关键新贡献是: (1) 将 LLM 的文本理解能力直接用于理解 freestyle emotion description,不需 PromptTTS 式的专用 style encoder;(2) phoneme boost parallel output (EmoVoice-PP) 是一种新的输出侧设计,灵感来自 CoT/CoM 但在 TTS 中首次以 parallel phoneme prediction 形式出现;(3) EmoVoice-DB 是首个带 NL emotion description 标注的 40 小时情感语音数据集。
>
> 检索命中: [[LLM-based TTS]]✓, [[模型库/CosyVoice 2|CosyVoice 2]]✓, [[Semantic vs Acoustic Tokens]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[Natural Language Description for TTS]](pending-review), [[Instruction-Guided Speech Synthesis]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 Qwen2.5 LLM 的情感 TTS,通过 freestyle 自然语言描述控制情感,phoneme parallel output 增强内容一致性,附带 40h 高质量情感数据集 EmoVoice-DB
> - **路线**: (Emotion Description + Text) → Qwen2.5-0.5B/1.5B LLM → CosyVoice 50Hz Semantic Tokens (group size=3) [+ Phoneme Tokens parallel] → CosyVoice Flow Matching + HiFi-GAN → Waveform
> - **指标**: EmoVoice(1.5B) 在 EmoVoice-DB 上 WER 2.62 / Emo_Sim 0.9118 / Recall 0.424 / MOS 3.507 (vs GPT-4o-mini-tts MOS 3.598) [Table 2, Table 3]; 中文 Secap 上 EmoVoice-PP WER 7.60 / Recall 0.434 [Table 4]
> - **可借鉴**: (1) Parallel phoneme output 作为辅助 supervision 降低 WER,尤其在 hard case 上效果显著 (18.07→11.68) [Table 6]; (2) 用 GPT-4o-audio 蒸馏构建情感数据集的完整 pipeline (text+description生成→语音合成→WER过滤); (3) Description augmentation 策略 (GPT-4o 改写 emotion description,WER 从 3.83 降到 2.73,相对降幅 29%) [Table 9]
> - **局限**: (1) 完全依赖合成数据训练,真实情感语音的泛化性未充分验证; (2) Emotion evaluation 指标 (emotion2vec similarity) 与人类感知对齐度低 (sentence-level Spearman ρ 仅 0.40) [Table 10]; (3) 不支持 zero-shot speaker cloning — 需要从同说话人选 prompt speech; (4) 代码/模型已开源但 in-house 中文数据未公开

## 核心问题

这篇论文解决的核心问题是: 如何让 TTS 模型通过自由格式的自然语言描述实现细粒度的情感控制? 此前的情感 TTS 要么只支持粗粒度类别标签 (happy/sad/angry),要么依赖于专用 style encoder 理解受限的文本描述; EmoVoice 的方案是直接利用 LLM 的文本理解能力来理解 freestyle emotion description,同时提出一个高质量的 NL emotion description 数据集填补数据空白。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

EmoVoice 以 Qwen2.5-0.5B (24 层 Transformer, 0.49B 参数) 为 backbone,用预训练 LLM 参数初始化 [§3.1]。输入是纯文本格式: `<SYSTEM>: Say this sentence with emotion of <Description>. \n <Text>.`,不需要参考情感音频 [§3.1]。

输出端生成 50Hz CosyVoice semantic tokens,然后通过 CosyVoice 的 flow matching module + HiFi-GAN vocoder 转换为音频波形 [§3.1]。LLM 词表扩展: 原始文本词表 V_t 加入音频 codebook V_a,形成 V_j = V_t ∪ V_a;原始 embedding 保持不变,音频 token embedding 随机初始化 [§3.1]。推理时从 output logits 中提取音频部分: x_a = logits[..., |V_t|:] [§3.1]。

**Semantic Group Modeling**: 采用 SLAM-Omni 提出的分组建模策略,每步预测 G=3 个 semantic tokens [§3.1]。使用一个 linear layer 将音频 logits L_a 投影为 group-sized logits L_g ∈ R^{|V_a|×G} [§3.1]。输入侧每步使用 group 内所有 semantic tokens 的 embedding 均值 [§3.1]。这带来 2.64x 训练加速且 WER 更低 [§5, Training Details] — [agent 解读] 分组建模缩短序列长度,降低了 LLM 的自回归建模难度。

### 关键设计选择

**为什么用 LLM 而非专用 prompt encoder?** [论文原文] 作者的动机是: 利用 LLM 在文本语义理解和情感分析方面的已有能力,通过 LLM 直接理解 emotion description 指令,无需像 PromptTTS 那样训练专用 prompt encoder [§1]。[agent 解读] 这也意味着 LLM 初始化至关重要 — 没有 LLM 初始化,WER 从 2.73 飙升到 6.16 [Table 8],说明 LLM 的语言知识对齐文本和语音 token 起到了决定性作用。

**EmoVoice-PP (Phoneme Parallel Boost)**: 这是核心变体设计。在输出侧并行预测 semantic tokens 和 phoneme tokens [§3.1]。具体做法: 将 phoneme 加入 Qwen2.5 词表 (扩展为 V_t'),分别从 logits 提取音频部分 x_a = logits[..., |V_t'|:] 和音频部分 x_p = logits[..., :|V_t'|] [§3.1]。训练时 phoneme 序列由 Phonemizer 工具提取作 teacher forcing [§3.1]。

**为什么 parallel 而非 serial?** [论文原文] 灵感来自 CoT (Chain-of-Thought) 和 CoM (Chain-of-Modality, SpeechGPT 提出): 让模型先"想"发音再生成音频 [§1, §3.1]。推理时 phoneme token rate (~11Hz) 低于 audio token rate (~17Hz),因此 phoneme 被更早预测出来,作为中间 supervision signal 引导后续 audio token 生成 [§3.1]。

[agent 解读] 作者的 ablation [Table 5] 对比了 6 种输出结构: (a) 纯音频, (b) serial phoneme→audio, (c) serial text→audio, (d) parallel text+audio, (e) parallel phoneme+audio (PP), (f) interleaved text+audio。在 fine-tuned 阶段差异不大,但在 pre-trained TTS 阶段 PP 的 WER 最低 (3.94/3.11 on EmoVoice-DB/Seed-TTS),且 hard-case 测试集上 PP 的 WER 优势极为明显 (11.68 vs 18.07 纯音频) [Table 6]。这说明 parallel phoneme output 的主要贡献是增强内容一致性 (content consistency) 而非情感表达。Phoneme 比 text 更有效的原因可能是: phoneme 与 audio token 的对齐更直接,减少了文字→发音的隐式映射负担。

### 训练策略

两阶段训练 [§3.2]:
1. **Pre-training**: 在标准 TTS 数据 (VoiceAssistant 3234h 英文 / Belle 6418h 中文, 均为 CosyVoice semantic tokens) 上预训练,格式为 `<SYSTEM>: Say this sentence. \n <Text>.` [§5]
2. **Fine-tuning**: 在情感数据 (LAION's Got Talent 200h + EmoVoice-DB ~40h) 上微调,格式加入 emotion description [§5]

训练细节: AdamW (β=0.9, 0.999), weight decay=0; 预训练 peak LR=1e-4 带 warmup 1000 步 + linear decay; fine-tuning peak LR=1e-5; 4× A800 80GB, batch size=6 [§5]。

推理: greedy search + repetition penalty 1.2; 用同说话人不同中性语音作为 flow matching 的 timbre prompt [§5]。

## 实验

| 指标 | EmoVoice(1.5B) | CosyVoice2 | PromptTTS | GPT-4o-mini-tts | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER↓ | 2.62 | 3.61 | 2.11 | 2.40 | EmoVoice-DB | [Table 2] |
| Emo_Sim↑ | 0.9118 | 0.8647 | 0.8709 | 0.9168 | EmoVoice-DB | [Table 2] |
| Recall↑ | 0.424 | 0.371 | 0.291 | 0.456 | EmoVoice-DB | [Table 2] |
| UTMOS↑ | 4.35 | 4.42 | 4.32 | 4.06 | EmoVoice-DB | [Table 2] |
| MOS (emotion)↑ | 3.507 | 2.138 | 1.415 | 3.598 | EmoVoice-DB | [Table 3] |
| WER↓ (中文) | 7.64 | 9.13 | — | 10.34 | Secap | [Table 4] |
| Emo_Sim↑ (中文) | 0.7933 | 0.7811 | — | 0.6184 | Secap | [Table 4] |
| WER↓ (PP, 中文) | 7.60 | 9.13 | — | 10.34 | Secap | [Table 4] |

**关键实验发现**:

1. **EmoVoice vs CosyVoice2**: 在 EmoVoice-DB 上 emotion MOS 3.507 vs 2.138,巨大差距 [Table 3]。[agent 解读] 但需注意 EmoVoice 在 EmoVoice-DB 上训练和测试,CosyVoice2 未在此数据上专门微调,属于 out-of-domain 对比,因此该差距部分来自训练数据分布优势。

2. **中文 Secap 结果**: EmoVoice/PP 在中文上也优于 CosyVoice2 和 GPT-4o 系列 [Table 4],但 GPT-4o 在中文上表现差是因为口音和韵律不自然 [§6.1.3]。

3. **Scaling LLM**: 0.5B→1.5B,预训练 WER 从 4.73 降到 3.83 (相对 -19%),MOS 从 3.163 升到 3.507 [Table 7, Table 3]。但计算成本大幅增加 [§6.2.3]。

4. **LLM 初始化效果**: 有 vs 无 LLM 初始化,EmoVoice WER 2.73 vs 6.16,EmoVoice-PP WER 3.06 vs 6.06,EmoVoice-PT WER 3.42 vs 7.87 [Table 8]。[论文原文] LLM 的语言理解和情感理解能力被有效迁移到 TTS 框架中 [§6.2.4]。

5. **Description Augmentation**: GPT-4o 改写 emotion description (每条生成 2 个改写版, 3x 数据量), WER 从 3.83 降到 2.73 (-28.7%) [Table 9]。[论文原文] 增强模型对情感语义的理解,降低对单一表达模式的过拟合 [§6.2.5]。

6. **Emotion 评估指标可靠性** [§7]: emotion2vec similarity 的 system-level Spearman ρ = 0.943 但 sentence-level 仅 0.405 [Table 10]。GPT-4o-audio 和 Gemini 作为评估器的 sentence-level ρ 更低 (0.257, 0.196)。结论: 现有指标在系统级对比有参考价值,但细粒度单句对比不可靠。

## 局限性

1. **数据依赖合成**: 英文模型完全依赖合成训练数据 (CosyVoice semantic tokens from VoiceAssistant + GPT-4o-audio 生成的 EmoVoice-DB),虽然论文声称 "demonstrate the feasibility of training a superior emotional TTS model with only synthetic data" [§1],但对真实世界复杂情感语音的泛化性缺乏验证。

2. **评估局限**: 论文自身发现 emotion similarity 在 sentence-level 与人类感知对齐度低 [Table 10],但主实验仍大量依赖此指标。MOS 评估仅 60 样本 × 30 评估者 [§6.1.2],规模有限。

3. **测试集偏差**: 主要在自建的 EmoVoice-DB 测试集上评估,训练和测试来自同分布 (GPT-4o 合成),对 CosyVoice2 等 baseline 不公平。中文 Secap 实验是跨域但规模有限 (600 样本) [§5]。

4. **不支持 zero-shot speaker cloning**: 推理时需要同说话人的中性语音作 prompt [§5],无法像 CosyVoice 那样零样本克隆任意说话人。

5. **情感范围有限**: EmoVoice-DB 覆盖 7 类基础情感 (angry, happy, sad, surprised, disgusted, fearful, neutral) [Table 1],虽然每类有 freestyle NL description 细粒度标注,但未覆盖更复杂的混合情感或讽刺等高阶情感。

6. **Group modeling 的信息损失**: group size=3 虽加速训练,但可能丢失 token 级精细信息 — 论文未分析 group size 对情感表达的影响。

## 点评

EmoVoice 的核心价值在于验证了一个直觉上合理但此前缺乏实证的假设: LLM 天生的文本理解能力可以被直接用于理解 freestyle emotion description,不需要额外的 prompt encoder。LLM 初始化实验 [Table 8] 是最有说服力的证据 — WER 的巨大差距表明 LLM 的语言知识不仅有用,而且是必要的。

Phoneme parallel output (EmoVoice-PP) 是一个轻量但有效的工程贡献。它在 hard case 上的效果 (WER 18.07→11.68) [Table 6] 比在标准测试集上更显著,说明 phoneme supervision 的主要作用是防止 content 退化而非提升情感表达。这个设计可迁移到其他 LLM-based TTS 系统中。

EmoVoice-DB 的构建 pipeline (GPT-4o 生成 text+description → GPT-4o-audio 合成 → WER 过滤) 是一个可复制的数据构建方案,但"用合成数据训练、在合成数据上测试"的实验范式存在系统性偏差风险。论文未在任何真实情感语音数据集上做 cross-domain 评估 (ESD、IEMOCAP 等),这是一个明显遗憾。

情感评估指标的讨论 [§7] 是论文的一个亮点 — 坦诚承认了 emotion2vec similarity 和 LLM 评估器在 fine-grained 层面的不足,为后续工作指出了方向。

## 可复用的 idea

1. **Parallel auxiliary token output**: 在 LLM-based TTS 的输出侧并行预测辅助 token (phoneme/text) 作为 intermediate supervision,以增强 content consistency。这个思路不限于情感 TTS,可推广到所有 LLM-based TTS 中。

2. **合成数据构建 pipeline**: GPT-4o 生成多样化 NL description + GPT-4o-audio 合成语音 + WER 过滤。Description augmentation (每条 description 用 GPT-4o 改写 2 版,3x 数据量) 对小规模数据的 robustness 提升显著。

3. **直接在 LLM 输入中拼接控制指令**: 不用专用 encoder/adapter,把 emotion description 直接写进 prompt format,让 LLM 自行理解。这是最简单的控制方式,在 LLM backbone 足够强时效果好。

4. **多输出结构的系统性对比**: 论文对 6 种 output structure (pure audio / serial phoneme / serial text / parallel phoneme / parallel text / interleaved) 做了完整对比 [Table 5, Fig 2],这个实验设计本身可作为后续 LLM-based TTS 工作的 reference。

> [!review] 审阅
> 审阅状态: 待审阅 (see `_review/EmoVoice-review.yml`)
