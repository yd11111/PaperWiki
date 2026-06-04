---
type: paper
tier: deep
title: "ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching"
arxiv_id: "2507.09318"
source: "Sources/ZipVoice-Dialog.pdf"
authors: [Han Zhu, Wei Kang, Liyong Guo, Zengwei Yao, Fangjun Kuang, Weiji Zhuang, Zhaoqing Li, Zhifeng Han, Dong Zhang, Xin Zhang, Xingchen Song, Lingxuan Ye, Long Lin, Daniel Povey]
year: 2025
venue: "arXiv"
tags: [zero-shot-TTS, flow-matching, NAR-TTS, dialogue-synthesis, turn-taking, curriculum-learning, stereo-audio, dataset, evaluation]
concepts: ["[[Conditional Flow Matching]]", "[[Non-autoregressive TTS]]", "[[Turn-taking in Spoken Dialogue]]", "[[Speech-Text Alignment]]", "[[Classifier-Free Guidance]]", "[[Spoken Dialogue Evaluation]]"]
models: ["[[论文笔记/ZipVoice|ZipVoice]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页: [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓, [[Non-autoregressive TTS]][待确认], [[Turn-taking in Spoken Dialogue]][待确认], [[Speech-Text Alignment]][待确认], [[Spoken Dialogue Evaluation]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: ZipVoice-Dialog 是 flow-matching-based NAR TTS 从单话者独白向多话者对话场景的首次成功扩展。它建立在 ZipVoice (Zhu et al., 2025) 的 Zipformer + CFM + speech infilling 架构之上,核心创新在于解决 flow matching 在多说话人对话场景下的两个新问题: (1) 多说话人导致 speech-text alignment collapse, (2) 缺乏说话人轮次控制。在知识库中,当前 turn-taking 的研究主要集中在交互式全双工系统 (Moshi、dGSLM、Freeze-Omni 等实时推理场景),ZipVoice-Dialog 解决的是一个不同子问题: 给定完整文本脚本,生成带正确说话人轮次分配的对话语音 (dialogue TTS/podcast generation)。这一分支的 AR 代表包括 MoonCast、Dia、Parakeet;ZipVoice-Dialog 是该分支的第一个 NAR 模型。
>
> **已有认知**: (1) [[Conditional Flow Matching]] 是当前 TTS 主流生成框架,ZipVoice 已证明 CFM + Zipformer + average upsampling 能在 123M 参数下达到与 F5-TTS (336M) 相当的独白质量; (2) [[Non-autoregressive TTS]] 的核心优势是并行生成 (推理速度快、无 exposure bias),但此前仅应用于单说话人场景; (3) [[Turn-taking in Spoken Dialogue]] 页面记录了端到端系统中隐式建模 turn-taking 的多种方法 (dGSLM dual-tower、Moshi multi-stream、Parrot dual-channel),但这些方案面向实时交互; (4) [[Speech-Text Alignment]] 在多说话人场景下比单说话人困难得多,因为模型需同时学习不同音色与对应文本的对齐; (5) [[Spoken Dialogue Evaluation]] 缺乏标准化的 dialogue TTS benchmark,ZipVoice-Dialog 提出的 cpWER/cpSIM 指标填补了这一空缺。
>
> **创新判断**: ZipVoice-Dialog 的核心创新在于两个简单但有效的方法: (a) monologue-to-dialogue curriculum learning 避免多说话人场景下的 alignment collapse; (b) learnable speaker-turn embeddings 替代复杂的说话人建模方案。此外,OpenDialog (6.8k 小时) 是该领域首个大规模开源对话数据集。与 concurrent work CoVoMix2 (Zhang et al., 2025) 的区别: ZipVoice-Dialog 不依赖预定义 timestamps,实现端到端 NAR 对话生成。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Non-autoregressive TTS]](pending-review), [[Turn-taking in Spoken Dialogue]](pending-review), [[Speech-Text Alignment]](pending-review), [[Spoken Dialogue Evaluation]](pending-review) | 未命中但可能相关: [[Full-duplex Spoken Dialogue]], [[Classifier-Free Guidance]]

> [!summary] 速查
> - **一句话**: 首个 NAR flow-matching 对话 TTS,通过 curriculum learning + speaker-turn embeddings 两个轻量设计使 123M 模型在对话生成上全面超越 1.6B Dia 和 2.7B MoonCast
> - **路线**: 文本 (interleaved [S1]/[S2] tokens) → Zipformer text encoder → average upsample + speaker-turn embedding → Zipformer vector field estimator (CFM) → Euler ODE solver → Vocos vocoder → waveform
> - **指标**: WER 3.25% / cpWER 3.27% / cpSIM 0.437 / UTMOS 3.07 (test-en), RTF 0.063 (15x faster than baselines) [Table 4]; CMOS +1.17 vs MoonCast, SMOS 3.86 vs 2.35 [Table 5]
> - **可借鉴**: (1) curriculum learning (monologue pretrain → dialogue finetune) 解决多说话人 alignment collapse,成本极低; (2) 两个 learnable embeddings 做 speaker disambiguation 比复杂 speaker encoding 更有效; (3) speaker exclusive loss 惩罚双通道同时活跃帧,适用于任何需要通道分离的生成任务
> - **局限**: 模型仅 123M,表达力受限 (MoonCast 在 expressiveness 上更好); 仅支持双人对话; stereo 模型性能弱于 mono (受限于双通道数据量); 主观评估仅中文

## 核心问题

对话语音生成相比独白 TTS 面临两个独有困难 [§1]:

1. **Speech-text alignment collapse**: 直接在对话数据上训练 flow matching 模型会导致输出不可理解。原因是两个说话人的不同音色使得模型难以学习正确的文本-语音对齐 [§4.1, Table 1: 无 curriculum learning 时 WER 高达 116.10%]。
2. **Speaker turn-taking accuracy**: 模型需要将正确的说话人音色分配给对应的文本段,但 vanilla flow matching 缺乏显式的说话人身份信号 [§4.2, Table 2: 用 "|" 分隔时 cpWER 37.82%]。

现有的 AR 对话 TTS 模型 (MoonCast, Dia, Parakeet) 虽然能处理这些问题,但面临高推理延迟和稳定性问题 (word skipping/repetition) [§1, §2.2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ZipVoice-Dialog 建立在 ZipVoice [§3] 之上,保持 123M 参数的 Zipformer 架构不变:

```
Text ([S1] Hi. [S2] Hey! [S1] Morning. [S2] Morning.)
    ↓
Text Encoder (Zipformer) → text features ŷ ∈ R^{F×N}
    ↓
Average Upsampling → text condition z ∈ R^{F×T}  (假设均匀 duration)
    + Speaker-Turn Embeddings (e_speaker(i))
    ↓
Vector Field Estimator (Zipformer) ← concat(unmasked speech, text condition, noisy features x_t)
    ↓
CFM loss: || v_t(x_t, z, ...) - (x_1 - x_0) ||² (仅 masked 区域)
    ↓
Inference: Euler ODE solver (16 steps) + time-dependent CFG → Vocos vocoder → waveform
```

训练采用 speech infilling 任务: 随机 mask 对话语音的一段,用 unmasked 部分作为 prompt [§3, §4.3]。

### 关键设计选择

**1. Monologue-to-Dialogue Curriculum Learning [§4.1]**

[论文原文] 直接在对话数据上训练 flow matching 导致 alignment collapse,生成的语音"听起来像人话但内容与文本不一致" [§8.1]。这是因为多说话人场景下 speech-text alignment 的学习难度远超单说话人场景 [§4.1]。

解决方案分两阶段:
- **Stage 1 (Monologue Pre-Training)**: 使用 ZipVoice 在 100K 小时独白数据上预训练的权重初始化,建立稳健的 speech-text alignment 基础 [§4.1]
- **Stage 2 (Dialogue Fine-Tuning)**: 在对话数据上微调,学习对话动态特征 (说话人轮次、音色切换等) [§4.1]

[agent 解读] 这个策略的核心 insight 是: alignment 是一个比对话特征更基础的能力,应该先在简单场景 (单说话人) 中建立,再迁移到复杂场景 (多说话人)。这与 NLP 中先预训练后微调的逻辑一致,但在 TTS 领域将"任务复杂度"作为 curriculum 的维度是新颖的。

效果 [Table 1]: curriculum learning 将 WER 从 116.10% 降至 5.47% (test-en),cpSIM 从 0.247 升至 0.444,同时 UTMOS 从 1.87 升至 3.28。

**2. Speaker-Turn Embeddings [§4.2]**

[论文原文] 为了让模型区分两个说话人并将正确的音色分配给对应文本段,作者在 text encoder 输出之后为每个 token 添加一个可学习的 speaker-turn embedding [§4.2, Eq. 2]:

```
ỹ_i = ŷ_i + e_{speaker(i)}
```

其中只有两个 embedding (对应 [S1] 和 [S2]),随机初始化后端到端训练。这些 embedding 不是传统的 speaker recognition embedding (i-vector/x-vector),而是仅表示"说话人 1"或"说话人 2"这一二元身份信号 [§4.2]。

[agent 解读] 这个设计极其简洁 -- 仅两个可学习向量就解决了说话人轮次分配问题。关键在于: 对话 TTS 不需要编码具体的说话人音色 (那由 prompt speech 提供),只需要告诉模型"这段文本属于哪个说话人"。将身份信号注入 text conditioning 而非 acoustic space 是合理的,因为 speaker turn 本质上是一个文本级别的标注。

效果对比 [Table 2] (test-en short):
- "|" 分隔 speaker turns: cpWER 37.82%
- "[S1]/[S2]" token 分隔: cpWER 31.34%
- "[S1]/[S2]" token + speaker-turn embedding: cpWER 5.82%

**3. Input Format [§4.3]**

(1) **Interleaved text input**: 多轮话语按开始时间排序,同一说话人的相邻话语合并为一轮,以 [S1]/[S2] token 前缀标识 [§4.3]。
(2) **Flexible speaker-turn prompts**: 训练时随机截取对话前缀 (可含多个 speaker turns) 作为 speech prompt,推理时支持灵活数量的 speaker-turn prompt [§4.3]。

[论文原文] ZipVoice-Dialog 通过 flow matching 目标隐式建模 token 和 turn 的 duration,不需要预定义 timestamps 或外部 duration predictor [§4.3]。这与 concurrent work CoVoMix2 (Zhang et al., 2025) 形成对比,后者依赖预定义 timestamps [§2.2]。

**4. Stereo Dialogue Extension (ZipVoice-Dialog-Stereo) [§A.1]**

三个策略将单通道模型扩展为双通道:
- **Weight initialization [§A.1.1]**: input/output projection 层的权重翻倍维度后复制单通道权重,最小化对预训练权重的扰动
- **Single-channel dialogue regularization [§A.1.2]**: 保留单通道和双通道两套 projection 层,训练时交替使用,防止在有限双通道数据上 catastrophic forgetting
- **Speaker exclusive loss [§A.1.3]**: 惩罚双通道同时有声帧,使用自适应能量阈值 τ (ground-truth 帧能量的中位数),鼓励通道分离 [Eq. 3-6]

### 训练策略

- **预训练**: ZipVoice 在 100K 小时独白数据上预训练 [§7.2]
- **对话微调**: 在 OpenDialog (6.8K 小时) + 内部数据集 (820 小时双通道) 上微调 60K updates,total batch size 4K seconds [§7.2]
- **Stereo 微调**: 在单通道模型基础上再微调 25K updates [§A.2]
- **推理**: Euler ODE solver, 16 sampling steps, time-dependent CFG [§3]

## OpenDialog 数据集 [§5]

**规模**: 6.8K 小时 (1759 小时中文 + 5074 小时英文),首个大规模开源对话语音数据集 [§5]。

**构建流程**:
1. **对话挖掘 [§5.1]**: 从 in-the-wild 音频中,通过 VAD → speaker diarization → ASR → LLM 分类器筛选对话
2. **说话人归属转写 [§5.2]**: 使用 WhisperD (fine-tuned Whisper) 获取准确的 speaker-attributed transcription;中文版本在内部数据上微调训练
3. **规则过滤 + DNSMOS 筛选 [§5.3]**: 过滤异常转写模式,DNSMOS < 2.8 的数据被移除

## 评估 Benchmark [§6]

**测试集**: test-zh (357 dialogues, 2.23h) + test-en (280 dialogues, 1.84h),均来自开源自然对话数据集,作为 out-of-domain 测试 [§6.1]。

**客观指标**:
- **WER**: 不考虑说话人的整体文字错误率 [§6.2]
- **cpWER (concatenated min-permutation WER)**: 将每个说话人的话语拼接后计算所有排列的最低 WER,反映 speaker turn-taking accuracy [§6.2]。cpWER - WER 的差值越大,说明越多话语被分配给了错误说话人
- **cpSIM (concatenated max-permutation speaker similarity)**: 分离说话人 → 提取 speaker embedding (WavLM-based ECAPA-TDNN) → 计算最优排列余弦相似度 [§6.2]
- **UTMOS**: 自动 MOS 预测,但对对话语音评分天然偏低 [§6.2]
- **RTF**: 实时因子 [§6.2]

## 实验

### 核心对比 [Table 4]

| 指标 | ZipVoice-Dialog (123M) | MoonCast (2.67B) | Dia (1.61B) | 出处 |
| --- | --- | --- | --- | --- |
| RTF ↓ | **0.063** | 0.953 | 1.663 | [Table 4] |
| WER ↓ (test-en) | **3.25** | 23.62 | 11.80 | [Table 4] |
| cpWER ↓ (test-en short) | **3.27** | 16.53 | 12.59 | [Table 4] |
| cpSIM ↑ (test-en) | **0.437** | 0.356 | 0.333 | [Table 4] |
| UTMOS ↑ (test-en) | **3.07** | 2.37 | 1.87 | [Table 4] |
| WER ↓ (test-zh) | **3.17** | 15.85 | -- | [Table 4] |
| cpSIM ↑ (test-zh) | **0.556** | 0.463 | -- | [Table 4] |

### 主观评估 [Table 5] (test-zh, 10 中文母语评估者)

| 指标 | ZipVoice-Dialog | MoonCast |
| --- | --- | --- |
| CMOS | 0.00 (reference) | -1.17 +/- 0.12 |
| SMOS ↑ | **3.86 +/- 0.11** | 2.35 +/- 0.14 |

### 数据集消融 [Table 3]

| 训练数据 | WER ↓ (test-en) | cpSIM ↑ (test-en) | UTMOS ↑ (test-en) |
| --- | --- | --- | --- |
| OpenDialog only (6.8k) | **3.34** | 0.428 | 3.04 |
| In-house only (0.8k) | 5.47 | **0.444** | **3.28** |
| All (7.6k) | 3.25 | 0.437 | 3.07 |

[论文原文] OpenDialog alone 足以训练高性能模型,性能不逊于 baselines。内部数据在 cpSIM 和 UTMOS 上略优,因为其人工标注质量更高 [§8.3]。

### Stereo 消融 [Table 6]

| 消融 | WER ↓ (test-en) | cpWER ↓ (test-en short) |
| --- | --- | --- |
| ZipVoice-Dialog-Stereo (full) | **4.67** | **4.93** |
| w/o speaker exclusive loss | 5.10 | 5.95 |
| w/o single-channel regularization | 5.56 | 6.09 |
| w/o single-channel initialization | 5.89 | 6.59 |

## 局限性

1. **模型规模限制表达力**: 123M 参数对推理速度有利,但天然限制了表达力上限。MoonCast 在 expressiveness 方面更好 (尽管其 AR 架构导致了稳定性问题) [Limitations, Table 5: MoonCast 评价者提到更好的 expressiveness]
2. **仅限双人对话**: 当前实现仅支持两个说话人,尽管方法本身可泛化到多人 [Limitations]
3. **Stereo 模型弱于 Mono**: 受双通道训练数据有限 (仅 820 小时) 影响 [§A.2, Table 6 vs Table 4]
4. **Speaker exclusive loss 限制重叠**: 该 loss 惩罚双通道同时活跃,因此不适用于需要重叠语音的场景 [§A.1.3]
5. **主观评估仅中文**: 受评估者可用性限制,CMOS/SMOS 仅在 test-zh 上进行 [Limitations]
6. **UTMOS 对对话语音的偏置**: UTMOS 对英文独白有固有偏好,对话场景的分数系统性偏低 [§6.2]

## 点评

**优势**:
- **极简有效的设计哲学**: 两个核心创新 (curriculum learning + speaker-turn embeddings) 都是极其简单的方法,但效果惊人。cpWER 从 37.82% 降到 5.82% 仅靠两个可学习向量,这体现了"正确的 inductive bias 比复杂模型更重要"的原则。
- **完整的贡献链**: 方法 + 数据集 (OpenDialog) + 评估 benchmark + 全部开源,这种"全链条"贡献对领域的推进价值远超单一方法创新。
- **实用性导向**: RTF 0.063 意味着真正可部署,123M 参数在边缘设备上也有运行可能,相比 Dia (1.61B) 和 MoonCast (2.67B) 有量级差距。

**可商榷之处**:
- **Expressiveness 的 trade-off**: 作者选择保持小模型以换取速度,但在对话场景中 expressiveness (情感变化、节奏控制) 可能比独白更重要。MoonCast 的主观评价中"更好的 expressiveness"值得关注。
- **cpWER 评估的局限**: cpWER 依赖 WhisperD 的 speaker-attributed transcription 准确性,且中文对话被排除在 cpWER 评估之外 [§6.2]。
- **Concurrent work 对比不充分**: 与 CoVoMix2 (Zhang et al., 2025) 的对比仅在方法描述层面,未提供实验数据。

## 可复用的 idea

1. **Curriculum learning by task complexity**: 在复杂任务中先用简单版本预训练建立基础能力,再微调到目标任务。不限于 TTS -- 任何从单一条件到多条件的生成任务都可借鉴。
2. **Minimal speaker disambiguation via learnable embeddings**: 仅用 2 个可学习向量就完成说话人轮次分配,避免了 speaker encoder / x-vector 等重型方案。关键 insight: 对话 TTS 需要的是"哪个说话人说这段"的指示信号,不是音色本身的编码。
3. **Speaker exclusive loss**: 自适应能量阈值 + 双通道活跃帧惩罚,是一种通用的通道分离策略,可迁移到 source separation、stereo speech generation 等场景。
4. **cpWER/cpSIM 评估方案**: 基于 concatenated min-permutation 的评估方法将 speaker attribution accuracy 和 content accuracy 统一到一个指标中,可作为对话 TTS 的标准评估 protocol。

> [!review] 审阅 (2026-06-04, auto)
> **结论**: pass-with-fixes
> - [fixed] Table 4 test-zh 行 MoonCast/Dia 数据交换 (medium, factual-error)
> - [noted] 速查"全面超越"措辞偏强 (low, overclaim)
> - [noted] datasets 列 Emilia 为间接使用 (low, template-compliance)
> 详见 `_review/ZipVoice-Dialog-review.yml`
