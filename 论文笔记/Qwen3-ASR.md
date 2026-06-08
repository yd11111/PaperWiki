---
type: paper
tier: deep
title: "Qwen3-ASR Technical Report"
arxiv_id: "2601.21337"
source: "Sources/Qwen3-ASR.pdf"
authors: [Qwen Team]
year: 2026
venue: "arXiv"
tags: [ASR, multilingual, LALM, forced-alignment, speech-recognition, language-identification, streaming, open-source, Qwen, singing-voice-recognition, NAR, reinforcement-learning]
concepts: ["[[SpeechLanguageModel]]", "[[AudioUnderstanding]]", "[[ModalityAdaptationforSpeechLLM]]", "[[LLM-enhancedASR]]"]
models: ["[[Whisper]]", "[[SenseVoice]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Qwen3-ASR 属于 Large Audio-Language Model (LALM) 范式,即将预训练音频编码器通过 projector 接入 LLM 进行端到端 ASR。这条路线不同于 [[LLM-enhancedASR]] 中的 text-based integration (rescoring/GER),后者 LLM 仅处理 ASR 产出的文本假设。Qwen3-ASR 的 AuT encoder + projector + Qwen3 LM 架构是 [[ModalityAdaptationforSpeechLLM]] 中 convolutional downsampling + linear projection 方案的典型实现。作为 [[SpeechLanguageModel]] 体系中 latent-representation-based 路线的一员,Qwen3-ASR 专注于 ASR 任务而非全模态交互。
>
> **已有认知**: [[Whisper]] [待确认] 是当前最广泛使用的大规模弱监督 ASR 系统 (680k 小时数据, encoder-decoder Transformer),其 encoder 后来成为众多 SpeechLM 的标准 speech feature extractor。[[SenseVoice]] [待确认] 是同团队 (阿里通义) 此前的 ASR+SER+AED+LID 统一理解模型,其 SenseVoice-Large encoder 是 CosyVoice 系列 S^3 tokenizer 的基础。Qwen3-ASR 则基于 Qwen3-Omni 基座,代表了该团队从 SenseVoice 到 LALM 范式的演进。
>
> **创新判断**: 相比 [[Whisper]] 的纯 encoder-decoder 弱监督路线,Qwen3-ASR 利用 LLM 的语言建模能力处理长文本、噪声鲁棒和多语言; 相比 [[LLM-enhancedASR]] 的 text-based GER 路线,Qwen3-ASR 是端到端无信息损失; Qwen3-ForcedAligner 则是首个基于 LALM 的 NAR forced aligner,区别于传统 MFA/NFA 的语言特定 pipeline。
>
> 检索命中: [[SpeechLanguageModel]]✓, [[AudioUnderstanding]](待确认), [[ModalityAdaptationforSpeechLLM]](待确认), [[LLM-enhancedASR]](待确认), [[Whisper]](待确认), [[SenseVoice]](待确认) | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 Qwen3-Omni 基座的 all-in-one ASR 家族 (0.6B/1.7B) + 首个 LALM-based NAR forced aligner,覆盖 52 语言/方言、歌声识别、流式推理
> - **路线**: 音频 → Fbank 100Hz → AuT encoder (8x下采样, 12.5Hz) → Projector → Qwen3 LM → ASR 文本; ForcedAligner: 音频 + 带 [time] 槽的转录 → AuT + Qwen3 LM → timestamp prediction layer (NAR)
> - **指标**: 1.7B LibriSpeech clean/other 1.63/3.38 WER, WenetSpeech 4.97/5.88 CER [Table 3]; 0.6B TTFT 92ms, throughput 2000s/s@128并发 [Table 2]; ForcedAligner AAS 32.4ms (vs MFA 141.3ms, NFA 101.2ms) on human-labeled [Table 9]
> - **可借鉴**: (1) 四阶段训练范式 (AuT预训练→Omni预训练→ASR SFT→GSPO RL); (2) ForcedAligner 的 slot-filling 重构 + NAR 解码 + 动态槽插入训练策略; (3) ASR SFT 阶段不使用自然语言指令以防止 instruction injection
> - **局限**: 仅限 ASR 任务 (不支持 TTS/对话等生成); 30 语言 Fleurs 全集上略逊 Whisper-large-v3; ForcedAligner 训练依赖 MFA 伪标签; 内部 benchmark 不可复现

## 核心问题

Qwen3-ASR 要解决的核心问题是: **当前开源 ASR 模型在公开 benchmark 上趋于饱和,但在真实世界场景 (多口音、多方言、噪声环境、歌声、长音频) 下仍有显著质量差距** [§1, §4.2.2]。同时,forced alignment 领域缺乏统一的多语言端到端解决方案 [§3.1]。

具体而言:
1. 公开 benchmark 的标注误差已接近模型误差,无法区分模型真实能力 [§4.1]
2. 商业 API (GPT-4o-Transcribe, Gemini-2.5-Pro) 在部分数据集上表现不一致 [§4.2.1]
3. 现有 forced aligner (MFA, NFA, WhisperX) 是语言特定的,且在长音频上精度急剧下降 [§4.6]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Qwen3-ASR 家族包含三个模型 [§2, §3]:

| 模型 | LLM backbone | AuT 参数量 | Hidden size | 推理模式 | 最大音频 |
|------|-------------|-----------|-------------|---------|---------|
| Qwen3-ASR-1.7B | Qwen3-1.7B | 300M | 1024 | Offline/Streaming | 1200s |
| Qwen3-ASR-0.6B | Qwen3-0.6B | 180M | 896 | Offline/Streaming | 1200s |
| Qwen3-ForcedAligner-0.6B | Qwen3-0.6B | (共享AuT) | 896 | NAR | 300s |

**AuT encoder** [§2.1, Fig 2(left)]: 基于 attention-encoder-decoder (AED) 架构的音频编码器。输入 Fbank 100Hz,经 3 层 Conv2D 进行 8x 下采样,32 层 self-attention,输出 12.5Hz token rate。采用 dynamic flash attention window (1-8s),使得单一模型同时支持 streaming (短 chunk) 和 offline (长序列) 推理 [论文原文]。

**Projector** [§2.1, Fig 2(right)]: 将 AuT encoder 的输出投影到 Qwen3 LM 的 embedding space。Qwen Tokenizer 对文本部分编码,与 AuT 输出拼接后送入 Qwen3 LM [论文原文]。

**输出格式** [§2.2]: Qwen3-ASR 不遵循自然语言指令,而使用固定输出模板。对可识别语音输出 `language {LID}<asr_text>{转录}`,对无语音输出 `language None<asr_text>`。这一设计是为了 **防止 instruction injection 和 instruction-following 失败** [论文原文]。

[agent 解读]: 不使用自然语言指令是一个有意的安全设计——传统 LLM 的 instruction-following 能力在 ASR 场景下反而是攻击面,恶意音频可能通过语音注入指令。固定模板消除了这一风险,代价是牺牲了灵活的多任务指令能力。

### 关键设计选择

**1. 基于 Qwen3-Omni 基座而非从头训练** [§2.1, §2.2]:

Qwen3-ASR 从 Qwen3-Omni (Xu et al., 2025b) 的预训练检查点出发,继承其"强大的音频理解能力"[§2.1]。AuT encoder 与 Qwen3-Omni 共享预训练,但在 Qwen3-ASR 中单独分离出来。

[agent 解读]: 这是一个"大模型蒸馏到小模型"的策略。Qwen3-Omni 是一个全模态大模型 (3T tokens 预训练),而 Qwen3-ASR 通过 SFT 将其专化为 ASR-only 模型,在保留语言知识的同时减少推理成本。与 Whisper 的从零弱监督训练路线形成对比。

**2. 动态注意力窗口 (1-8s)** [§2.1]:

AuT encoder 使用 dynamic flash attention window,窗口大小在 1s-8s 之间变化。短窗口用于 streaming 推理 (2s chunk + 5-token fallback),长窗口用于 offline 推理 [§4.5, Table 8]。

[agent 解读]: 这解决了 streaming/offline 统一的问题——传统方法需要分别训练两个模型。动态窗口使模型在训练时看到不同粒度的上下文,推理时根据需求切换即可。

**3. ASR SFT 不使用自然语言指令** [§2.2]:

SFT 阶段训练模型"成为仅 ASR 的模型,不遵循 prompt 中的自然语言指令"[§2.2]。同时学习利用 system prompt 中的 context tokens 作为 context biasing [§2.2]。

[论文原文]: "we train the model to be an ASR-only model that does not follow natural-language instructions in the prompt, in order to mitigate instruction injection and instruction-following failures."

**4. GSPO 强化学习** [§2.2]:

第四阶段使用 Group Sequence Policy Optimization (GSPO, Zheng et al. 2025) 进行 RL 微调。数据仅 50k utterances (35% 中英 + 35% 多语言 + 30% 功能性数据)。RL 对噪声鲁棒性、转录稳定性和复杂场景分析能力有"essential"作用 [§2.2]。

[agent 解读]: 50k utterances 的 RL 数据量非常小,但论文声称效果显著。这暗示 RL 的价值不在于大规模数据,而在于针对性地校准模型在困难场景下的行为——类似于 LLM alignment 中少量高质量 RLHF 数据的效果。

### Qwen3-ForcedAligner 架构与设计

**Slot-filling 重构** [§3.1, §3.2, Fig 3]: 将 forced alignment 重新定义为 slot-filling 任务。给定转录文本,在每个词/字符处插入 `[time]` 占位符表示起止时间戳,模型预测每个 slot 的 timestamp index [论文原文]。

**NAR 解码** [§3.4]: ForcedAligner 使用非自回归 (NAR) 解码——同时预测所有 timestamp slots 的索引。Timestamp index = 实际时间 / 80ms frame duration,最大 3750 类 (对应 300s) [§3.2]。

[论文原文]: 标准 LALM 的 next-token prediction 范式"不适合填充 timestamp slots" [§3.3]。ForcedAligner 采用 causal training (非移位序列),使模型在预测当前 slot 时能利用先前上下文,保持全局一致性。

**动态槽插入** [§3.3]: 训练时随机决定是否为每个词/字符插入起止 timestamp slots,增强泛化能力 [论文原文]。

**MFA 伪标签训练** [§3.3]: 使用 Montreal Forced Aligner (MFA) 生成伪标签。论文指出 MFA 伪标签"固有地包含噪声和系统性偏移",但 ForcedAligner"不是简单复制 MFA 输出,而是蒸馏和平滑这些伪标签,产生更稳定、偏移更小的预测"[§3.3]。

[agent 解读]: 这是一个有趣的"student surpasses teacher"现象——LLM 的全局上下文建模能力使其能发现并纠正 MFA 的局部对齐错误。causal training 确保时间戳的全局单调性,这是 frame-level MFA 难以保证的。

### 训练策略

四阶段训练 [§2.2]:

| 阶段 | 目标 | 数据 | 更新模块 |
|------|------|------|---------|
| 1. AuT 预训练 | 通用音频编码器 | ~40M 小时伪标签 ASR (中英) | AuT encoder |
| 2. Omni 预训练 | 多模态基座 | 3T tokens (音频+视觉+文本) | 全模型 |
| 3. ASR SFT | ASR 格式迁移 | 多语言 ASR + 非语音 + streaming + context biasing | 全模型 |
| 4. GSPO RL | 鲁棒性优化 | 50k utterances | 全模型 |

## 实验

### 公开 ASR 基准 [Table 3]

| 指标 | Qwen3-ASR-1.7B | Qwen3-ASR-0.6B | Whisper-large-v3 | GPT-4o-Transcribe | Gemini-2.5-Pro | 数据集 | 出处 |
|------|---------------|---------------|-----------------|-------------------|---------------|--------|------|
| WER | 1.63\|3.38 | 2.11\|4.55 | 1.51\|3.97 | 1.39\|3.75 | 2.89\|3.56 | LibriSpeech clean\|other | [Table 3] |
| WER | 8.45 | 8.88 | 9.76 | 25.50 | 9.37 | GigaSpeech | [Table 3] |
| CER | 4.97\|5.88 | 5.97\|6.88 | 9.86\|19.11 | 15.30\|32.27 | 14.43\|13.47 | WenetSpeech net\|meeting | [Table 3] |
| CER | 2.41 | 2.88 | 4.09 | 2.44 | 2.71 | Fleurs-zh | [Table 3] |

### 内部鲁棒性基准 [Table 4]

| 场景 | Qwen3-ASR-1.7B | Qwen3-ASR-0.6B | GPT-4o | Doubao-ASR | 出处 |
|------|---------------|---------------|--------|-----------|------|
| Dialog-Accented English | 16.07 | 16.62 | 28.56 | 20.41 | [Table 4] |
| Elders&Kids (zh) | 3.81 | 4.48 | 14.27 | 4.17 | [Table 4] |
| ExtremeNoise (zh) | 16.17 | 17.88 | 29.06 | 17.04 | [Table 4] |

### 多语言 ASR [Table 5]

| 指标 | Qwen3-ASR-1.7B | Whisper-large-v3 | Fun-ASR-MLT-Nano | 数据集 | 出处 |
|------|---------------|-----------------|-----------------|--------|------|
| WER avg | 8.55 | 8.62 | 28.70 | MLS (8 langs) | [Table 5] |
| WER avg | 9.18 | 17.25 | — | CommonVoice (13 langs) | [Table 5] |
| WER avg | 4.90 | 5.27 | 10.03 | Fleurs (12 langs) | [Table 5] |
| WER avg | 12.60 | **8.16** | 47.84 | Fleurs†† (30 langs) | [Table 5] |

### 语言识别 [Table 6]

| 模型 | MLS | CommonVoice | MLC-SLM | Fleurs | Avg | 出处 |
|------|-----|-------------|---------|--------|-----|------|
| Qwen3-ASR-1.7B | 99.9 | 98.7 | 94.1 | 98.7 | **97.9** | [Table 6] |
| Whisper-large-v3 | 99.9 | 92.7 | 89.2 | 94.6 | 94.1 | [Table 6] |

### 歌声与歌曲识别 [Table 7]

| 场景 | Qwen3-ASR-1.7B | GPT-4o | Whisper-large-v3 | 数据集 | 出处 |
|------|---------------|--------|-----------------|--------|------|
| Singing | 5.98 | 16.77 | 13.58 | M4Singer | [Table 7] |
| Songs with BGM (en) | 14.60 | 30.71 | N/A | EntireSongs-en | [Table 7] |
| Songs with BGM (zh) | 13.91 | 18.68 | N/A | EntireSongs-zh | [Table 7] |

### 推理效率 [Table 2]

| 模型 | 并发 | RTF | Throughput (s/s) | TTFT avg (ms) | 出处 |
|------|------|-----|-----------------|---------------|------|
| Qwen3-ASR-0.6B | 1 | 0.00923 | 108.34 | 92 | [Table 2] |
| Qwen3-ASR-0.6B | 128 | 0.11264 | 1136.36 | 3210 | [Table 2] |
| Qwen3-ASR-0.6B (online, 128) | — | 0.06400 | 2000.00 | — | [Table 2] |
| Qwen3-ASR-1.7B | 1 | 0.01482 | 67.48 | 102 | [Table 2] |

### Forced Alignment 精度 [Table 9]

| 评估集 | MFA | NFA | WhisperX | Qwen3-FA | 出处 |
|--------|-----|-----|----------|----------|------|
| MFA-Labeled Raw (avg, ms) | 161.1 | 129.8 | 133.2 | **42.9** | [Table 9] |
| MFA-Labeled Concat-300s (avg, ms) | 1742.4 | 246.7 | 2708.4 | **52.9** | [Table 9] |
| Human-Labeled (avg, ms) | 141.3 | 101.2 | — | **32.4** | [Table 9] |

### 流式推理 [Table 8]

| 模型 | 模式 | LibriSpeech | Fleurs-en | Fleurs-zh | Avg | 出处 |
|------|------|-------------|-----------|-----------|-----|------|
| Qwen3-ASR-1.7B | Offline | 1.63\|3.38 | 3.35 | 2.41 | 2.69 | [Table 8] |
| Qwen3-ASR-1.7B | Streaming | 1.95\|4.51 | 4.02 | 2.84 | 3.33 | [Table 8] |

## 局限性

1. **30 语言 Fleurs 全集上不及 Whisper-large-v3**: Fleurs†† (30 语言) WER 12.60 vs Whisper 8.16 [Table 5],表明在长尾语言上仍有提升空间。论文承认"model scaling improves robustness in more challenging multilingual regimes"但当前模型规模不足以覆盖所有语言 [§4.3.1]。

2. **内部 benchmark 不可复现**: 多个关键结论 (口音鲁棒性、方言评估、歌声) 基于不公开的内部测试集 [Table 4, Table 7 部分],外部无法验证。

3. **ForcedAligner 依赖 MFA 伪标签**: 训练数据的 timestamp 标注来自 MFA [§3.3],虽然论文声称模型"蒸馏和平滑"了伪标签,但 MFA 本身的系统性偏差可能仍存在于模型中。Human-labeled 测试集仅验证了 AAS 降低,未分析具体偏差模式。

4. **仅限 ASR 任务**: 作为 Qwen3-Omni 的 ASR 专化版本,无法执行 TTS、对话等生成任务。这是有意的设计选择,但限制了作为通用语音系统的适用范围。

5. **Streaming 模式有精度损失**: Streaming 推理相比 Offline 平均 WER 上升约 0.6-1.2 个绝对值 [Table 8],在实时性要求高的场景中需权衡。

6. **训练数据规模未完全透明**: AuT 预训练用了"约 40M 小时"伪标签数据 [§2.2],Omni 预训练用了 3T tokens [§2.2],但 ASR SFT 和 RL 阶段的数据量和来源描述不够详细。

## 点评

**优势明确**: Qwen3-ASR 在中英双语 ASR 上实现了开源 SOTA,尤其在噪声环境和方言场景下优势显著。四阶段训练范式 (预训练→Omni→SFT→RL) 提供了一个清晰的工程路线图,其中 GSPO RL 阶段仅用 50k 数据就带来明显鲁棒性提升,成本效益极高。

**ForcedAligner 是最大亮点**: 将 forced alignment 重构为 LLM 的 slot-filling 任务是一个优雅的设计。NAR 解码 + causal training 确保了效率和全局一致性,而动态槽插入提供了灵活的粒度选择。AAS 降幅 67-77% [Table 9] 相当惊人,尤其在长音频 (300s) 上的优势更明显 (其他方法 AAS 暴涨至 1000ms+,ForcedAligner 保持 50ms 级别)。

**定位准确**: 论文诚实地承认公开 benchmark 已触及标注误差上限,因此构建了内部评估体系。这是务实的做法,但也意味着核心差异化结论 (vs 商业 API) 依赖不可复现的内部数据。

**与 Whisper 的竞合关系**: Whisper-large-v3 在 LibriSpeech clean 和多语言 Fleurs 全集上仍有优势,但 Qwen3-ASR 在真实世界场景 (GigaSpeech、WenetSpeech、口音、噪声) 下全面领先。这反映了两种路线的差异: Whisper 的 680k 小时弱监督训练覆盖广但浅,Qwen3-ASR 的 LLM 基座 + SFT + RL 路线在困难场景下更鲁棒。

## 可复用的 idea

1. **Slot-filling 范式做 forced alignment**: 将时间戳预测转化为 LLM 的 slot-filling 任务,用 NAR 解码和 causal training 保证效率和全局一致性。这个范式可迁移到任何需要序列级对齐的任务 (字幕对齐、歌词对齐、口型同步)。

2. **ASR SFT 去除自然语言指令**: 将 ASR 模型的输出固定为模板格式,防止 instruction injection。这对任何安全敏感的语音前端系统都有参考价值。

3. **动态注意力窗口统一 streaming/offline**: 训练时使用 1-8s 的动态窗口,推理时切换窗口大小实现 streaming (2s) 或 offline (8s),避免了分别训练两个模型的开销。

4. **小规模 RL (GSPO) 做最后一公里优化**: 仅 50k utterances 的 RL 微调就能显著提升噪声鲁棒性和转录稳定性。对于已经有较好 SFT 基础的 ASR 模型,少量 RL 数据的边际收益很高。

5. **MFA 伪标签蒸馏**: ForcedAligner 在 MFA 伪标签上训练但精度超越 MFA,说明 LLM 的全局建模能力可以从有噪声的监督信号中提取更干净的模式——"student surpasses teacher"策略。
