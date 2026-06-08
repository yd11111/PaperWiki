---
type: paper
tier: deep
title: "UniSep: Universal Target Audio Separation with Language Models at Scale"
arxiv_id: "2503.23762"
source: "Sources/UniSep.pdf"
authors: [Yuanyuan Wang, Hangting Chen, Dongchao Yang, Weiqin Li, Dan Luo, Guangzhi Li, Shan Yang, Zhiyong Wu, Helen Meng, Xixin Wu]
year: 2025
venue: "arXiv"
tags: [audio-separation, language-model, discrete-token, universal-model, pre-training, target-separation]
concepts: ["[[ResidualVectorQuantization]]", "[[CodecLanguageModel]]", "[[SpeechLanguageModel]]", "[[AudioTokenizerTaxonomy]]"]
models: ["[[SoundStream]]"]
tasks: []
datasets: ["[[AudioSet]]"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[ResidualVectorQuantization]], [[SpeechLanguageModel]], [[SoundStream]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: UniSep 将 Codec Language Model 范式从语音/音频生成迁移到了音频分离任务。已有 KB 中,CodecLM [待确认] 定义为"直接在 neural audio codec 的 RVQ 码本索引上训练语言模型进行语音建模和生成",代表系统如 VALL-E, AudioLM, UniAudio 均面向生成任务。UniSep 是该范式在分离任务上的首次应用。
>
> **已有认知**:
> - RVQ (confirmed): 多层级向量量化,前层编码 coarse 信息、后层编码 fine details。UniSep 使用 Nq=3 层,属于极低层数配置(通常 4-8 层),这意味着信息密度较低,但序列长度也较短。
> - SoundStream (confirmed): Google 提出的首个端到端 neural audio codec,使用 RVQ+全卷积 encoder-decoder,75Hz 帧率。UniSep 直接复用其预训练 codec。
> - SpeechLanguageModel (confirmed): 端到端处理和生成语音的自回归基础模型。UniSep 的多尺度 Transformer 来自 UniAudio(SpeechLM 的代表系统之一),但目标从生成转为分离。
>
> **创新判断**: 已有 CodecLM 系统全部面向生成(TTS/音乐生成/对话),将 LM 用于分离是跨任务迁移。Pre-training 策略(Audio Continuation + Audio Inpaint)则是减少监督数据依赖的新方法。
>
> 检索命中: [[ResidualVectorQuantization]]✓, [[SpeechLanguageModel]]✓, [[SoundStream]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[AudioTokenizerTaxonomy]](pending-review), [[AudioSet]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个基于 LM 的通用目标音频分离模型,将分离建模为离散 token 空间的 seq2seq 问题,统一处理语音/声音/音乐分离
> - **路线**: 混合音频 → SoundStream RVQ (Nq=3) tokenize → [mixture tokens + prompt tokens + target tokens] 拼接 → 多尺度 causal Transformer (12 global + 4 local layers, 535M) → AR 解码 target tokens → SoundStream decoder → 分离音频
> - **指标**: Speech DNSMOS 4.07 / MUSHRA 4.11 (Libri2Mix); Sound ViSQOL 4.10 / MUSHRA 3.73 (AudioSet); Music avg ViSQOL 4.23 / avg MUSHRA 3.72 (MUSDB18); LASS fine-tuning 用 0.12% audio-text pairs 达到 AudioSep 96.94% ViSQOL
> - **可借鉴**: (1) Audio Continuation + Audio Inpaint 两阶段 pre-training 利用无标注音频数据; (2) 统一模型优于单任务模型的实验证据; (3) 用 LM 做分离的范式迁移思路
> - **局限**: PESQ 等对齐敏感指标不适用于 AR 采样(作者承认); LASS fine-tuning 效果不如 AudioSep(数据量差 800x); 未与连续空间分离方法做公平对比; 535M 参数推理速度未报告; 代码基于 UniAudio 但分离部分未单独开源

## 核心问题

UniSep 要解决的核心问题是: **如何用一个统一模型实现跨域(语音/声音/音乐)、跨源数量的目标音频分离?**

此前的分离方法存在三个局限 [§I]:
1. **域特定**: 不同类型音频(语音/声音/音乐)需要训练不同模型
2. **有限数据**: 多数监督分离方法依赖有限训练集,无法覆盖日常声音的广泛多样性
3. **目标分离缺失**: GASS 等通用模型做盲源分离,但通用的目标音频分离(给定 prompt 提取特定源)尚未探索

UniSep 将问题转化为三个研究问题 [§I]:
- 如何在一个统一模型中建模不同类型音频的复杂信号(它们频率跨度等模式差异很大)?
- 统一目标分离是否优于域特定模型?
- 大规模数据能否提升目标分离性能?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UniSep 的架构分为三个模块 [§II, Fig 1]:

1. **Audio Tokenizer**: 使用预训练的 SoundStream codec 将所有音频(混合、prompt、目标)编码为离散 token 序列
2. **Language Model**: 多尺度 causal Transformer 对拼接的 token 序列做自回归建模
3. **Audio Decoder**: SoundStream decoder 将预测的 target tokens 重建为波形

**序列布局** [§II-B, Fig 1(b)]:
```
[a_start] mixture_tokens [a_end] [p_start] prompt_tokens [p_end] [a_start] target_tokens [a_end]
```
LM 以 causal 方式建模整个序列,推理时根据 mixture + prompt 自回归预测 target tokens。

### 关键设计选择

**1. 离散空间而非连续空间** [§I, §II-A]

[论文原文] 传统分离在频谱图或波形(连续空间)上操作,UniSep 改用离散 token 空间。通过 SoundStream 的 RVQ (Nq=3 层) 将每帧编码为 3 个 token,然后沿 Nq 维度 flatten 为一维序列 [§II-A]。

[agent 解读] 使用 Nq=3 是一个激进的低比特率选择(通常 codec LM 用 4-8 层)。这大幅缩短序列长度,但也限制了重建质量的上限。对分离任务来说,这可能是合理的 trade-off——分离更关注源的可辨识性而非高保真重建。

**2. Causal LM vs Prefix LM** [§II-B]

[论文原文] 作者考虑了两种训练方式:
- **Prefix LM**: mixture + prompt 作为前缀(双向 attention),只预测 target tokens [Eq. 2]
- **Causal LM**: 整个序列(包括 mixture + prompt)都做 next-token prediction [Eq. 3]

最终选择 causal LM,理由有二: (1) causal LM 增加训练难度,要求 LM 内部建模 condition tokens 之间的关系,从而提升泛化能力; (2) 初步实验中 causal LM 优于 prefix LM [§II-B]。

[agent 解读] 这个选择值得注意。直觉上 prefix LM 更自然(condition 部分不需要预测),但 causal LM 迫使模型理解 mixture 和 prompt 的内部结构,可能作为一种隐式正则化发挥作用。

**3. 多尺度 Transformer** [§III-B]

模型采用 UniAudio 的多尺度 Transformer 架构 [16]:
- 12 层 global transformer + 4 层 local transformer
- 8 attention heads,embedding 维度 1536
- 535M 参数
- 训练于 8 块 Tesla V100 GPU [§III-B]

[agent 解读] 多尺度 Transformer 来自 UniAudio,原本用于音频生成。其中 local transformer 处理 RVQ 层间的局部依赖,global transformer 处理帧间的长距离依赖。这种分层处理有助于 flatten 后的 RVQ token 序列建模。

**4. 评估指标选择** [§III-B]

[论文原文] 由于 UniSep 使用 AR 采样,输出与 ground truth 不存在严格的样本对齐。因此 PESQ 和 SI-SNRi 等依赖精确对齐的指标不适用 [30]。作者使用:
- 语音: PESQ(仍报告但承认不适用) + DNSMOS(非侵入式) + MUSHRA(主观)
- 声音/音乐: STFT distance + ViSQOL + MUSHRA
- 所有评估使用 codec 重建音频作为 reference(消除 codec 本身的影响) [§III-B]

### 训练策略

**Pre-training (无标注数据)** [§II-C, Fig 1(a)]

动机 [论文原文]: 分离模型的两个关键能力是"一致性"(输出的目标音频应无噪声且连贯)和"相关性"(输出应与 prompt 最相关)。为此设计两个预训练任务:

1. **Audio Continuation**: 给定音频前半部分的 tokens,预测后续 tokens。训练 LM 生成一致的音频序列 [§II-C]
2. **Audio Inpaint**: 随机 mask 20% 的 tokens,用 [MASK] 替换,让 LM 预测被 mask 的 tokens。训练 LM 理解音频段之间的相关性 [§II-C]

预训练数据: LibriLight 20k 小时 + AudioSet 5.8k 小时,共 25.8k 小时无标注音频,切分为 10 秒片段。预训练耗时约 2 周 [§III-A]。

**Separation fine-tuning (模拟数据)** [§III-A]

在预训练基础上,使用模拟的三元组数据 (mixture, prompt, target) 进行分离训练:
- **语音**: MLS 数据集,随机混合不同说话人片段(SNR -5 to 10 dB),从目标说话人中提取 3 秒作为 prompt,约 10k 小时 [§III-A]
- **声音**: AudioSet,使用 SED 系统检测并提取音频段,两种模拟方式(随机混合不同事件 / SED 检测后时间切分),约 10.1k 小时 [§III-A]
- **音乐**: MUSDB18,随机选取 stem 的 prompt 和 target,混合其他 stem,722 小时 [§III-A]
- **总计**: 约 36.5k 小时(含预训练数据)

## 实验

| 指标 | UniSep (w/ pretrain) | UniSep (w/o pretrain) | UniSep Single (best) | Best Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| DNSMOS ↑ | **4.07** | 4.01 | 3.96 | 3.93 (UniAudio) | Libri2Mix | [Table II] |
| MUSHRA ↑ | **4.11** | 3.89 | 3.71 | 3.72 (UniAudio) | Libri2Mix | [Table II] |
| PESQ ↑ | 2.23 | 2.14 | 2.09 | **2.89** (SpeakerBeam) | Libri2Mix | [Table II] |
| ViSQOL ↑ | **4.10** | 4.02 | 3.94 | 2.99 (CLAPSep) | AudioSet | [Table III] |
| STFT ↓ | **1.24** | 1.28 | 1.35 | 1.43 (USS-ResUNet30) | AudioSet | [Table III] |
| MUSHRA ↑ | **3.73** | 3.55 | 3.24 | 3.20 (CLAPSep) | AudioSet | [Table III] |
| avg ViSQOL ↑ | **4.23** | 4.20 | 4.11 | 3.60 (CLAPSep) | MUSDB18 | [Table IV] |
| avg MUSHRA ↑ | **3.72** | 3.52 | 3.24 | 2.97 (CLAPSep) | MUSDB18 | [Table IV] |
| ViSQOL ↑ (LASS) | 3.49 | - | - | **3.60** (AudioSep, 14k hrs) | AudioCaps | [Table V] |

**关键发现**:

1. **统一模型 > 单任务模型** [§IV]: 在所有三个域上,UniSep 统一模型均优于相同架构的单任务模型 (UniSep Single),证明跨域训练带来正迁移 [论文原文]

2. **预训练显著提升** [§IV-A, B, C]: 在语音/声音/音乐上,预训练一致性地提升了所有指标。例如语音 MUSHRA 从 3.89 → 4.11(+0.22),声音 MUSHRA 从 3.55 → 3.73(+0.18)

3. **PESQ 不适用于 AR 模型** [§IV-A]: UniSep 的 PESQ (2.23) 远低于 SpeakerBeam (2.89),但 DNSMOS 和 MUSHRA 均大幅领先。作者指出这是因为 AR 采样导致输出与参考不严格对齐,PESQ 依赖对齐故不适用 [论文原文]

4. **LASS 下游迁移** [§IV-D, Table V]: 用仅 16.7 小时 AudioCaps 数据 fine-tune,UniSep 达到 AudioSep 96.94% 的 ViSQOL 性能(3.49 vs 3.60),而 AudioSep 用了 14k 小时 audio-text pairs。相同数据量下(vs LASS 17.3 hrs),STFT 相对改善 36.09%

## 局限性

1. **AR 采样的固有限制**: 生成式分离无法保证与 ground truth 的样本对齐,导致传统分离指标(SI-SNRi, SDR, PESQ)不可直接使用。这使得与连续空间分离方法的公平对比变得困难 [§IV-A]

2. **计算效率未讨论**: 535M 参数 + AR 解码的推理速度未报告。相比轻量级分离模型(如 Conv-TasNet ~5M 参数),推理延迟可能高出数个量级 [agent 解读]

3. **Codec 质量上限**: 使用 Nq=3 的 SoundStream,重建质量本身受限。虽然评估时用 codec 重建作为 reference 消除了这一因素,但实际应用中 codec 瓶颈是真实存在的 [agent 解读]

4. **数据模拟的人工性**: 所有训练数据通过人工混合模拟,SNR 范围(-5 to 10 dB)和混合方式可能无法覆盖真实场景的复杂性(如混响、多源重叠等) [agent 解读]

5. **LASS 数据不足**: 与 AudioSep 的对比中,UniSep 仅用 16.7 小时 audio-text pairs,差距 800 倍。在公平数据条件下的性能差异未知 [Table V]

6. **音乐分离数据量小**: MUSDB18 仅 722 小时模拟数据,远少于语音(10k)和声音(10.1k),可能限制了音乐分离的上限 [Table I]

## 点评

**范式意义**: UniSep 的核心贡献不在于绝对性能,而在于证明了 CodecLM 范式可以从生成任务迁移到分离任务。这是一个有意义的范式探索——将分离视为"给定条件的序列生成"而非"信号处理中的滤波/掩码"。

**Pre-training 策略的简洁性**: Audio Continuation 和 Audio Inpaint 两个任务设计简单但有效,利用了大量无标注音频数据。这个策略可迁移到其他音频 LM 任务。

**统一模型的价值**: 实验证明统一模型优于单任务模型是该论文最有力的结论之一。跨域(语音/声音/音乐)训练带来的正迁移效应,对其他多任务音频系统也有参考价值。

**对比公平性存疑**: baseline 比较可能不够公平——SpeakerBeam 和 VoiceFilter 是较老的模型(2018-2019),而 CLAPSep 和 SoloAudio 的比较只是"demo-level"平均值 [§IV-B]。缺少与当前 SOTA 连续空间分离模型的系统对比。

**与 UniAudio 的关系**: UniSep 大量复用了同组的 UniAudio 架构和 codec。从 KB 中可知 UniAudio 是一个通用音频基础模型,UniSep 可视为其在分离子任务上的专门化探索。

## 可复用的 idea

1. **LM 做分离的范式**: 将分离建模为条件序列生成,可能适用于其他需要从混合中提取特定成分的任务(如说话人提取、去噪、去混响)

2. **Audio Continuation + Audio Inpaint pre-training**: 两个自监督预训练任务利用无标注音频增强 LM 对音频一致性和相关性的理解。可迁移到任何基于 audio tokens 的 LM 系统

3. **统一多域训练的正迁移**: 语音+声音+音乐联合训练优于各自单独训练,这一发现提示多域数据混合可能对其他音频任务也有益

4. **Codec 重建作为评估 reference**: 用 codec 重建音频而非原始音频作为参考,消除 codec 本身的质量损失,使评估聚焦于 LM 的分离能力。这是一个在 codec-based 系统中通用的评估技巧

5. **Causal LM 优于 Prefix LM 的发现**: 在条件生成任务中,让 LM 也建模条件序列的内部结构(而非仅用作前缀),可能作为隐式正则化提升泛化性

## 审阅

> [!review] 审阅: pass — 0 high / 0 medium / 4 low
> 自审通过。4 个 low issue 均为 traceability-gap 或 template-compliance,不影响可信性。
> 详见 `_review/UniSep-review.yml`。
