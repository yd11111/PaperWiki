---
type: paper
tier: deep
title: "LLaMA-Omni: Seamless Speech Interaction with Large Language Models"
arxiv_id: "2409.06666"
source: "Sources/LLaMA-Omni.pdf"
authors: [Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, Yang Feng]
year: 2024
venue: "ICLR 2025"
tags: [speech-LM, streaming, real-time, non-autoregressive, CTC, speech-interaction, end-to-end, modular-SpeechLM]
concepts: ["[[SpeechLanguageModel]]", "[[ModalityAdaptationforSpeechLLM]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[StreamingSpokenDialogue]]", "[[Non-autoregressiveTTS]]", "[[SpeechTokenizer]]"]
models: ["[[Whisper]]", "[[HuBERT]]"]
tasks: []
datasets: ["InstructS2S-200K", "InstructS2S-Eval"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[ModalityAdaptationforSpeechLLM]], [[Speech-LLMIntegrationTaxonomy]], [[StreamingSpokenDialogue]], [[Non-autoregressiveTTS]])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: LLaMA-Omni 属于 **modular SpeechLM** 路线 -- 在已有文本 LLM (Llama-3.1-8B-Instruct) 外围挂载 Whisper 语音编码器和 NAR CTC 语音解码器,以极低训练成本获得 speech-in speech-out 交互能力。根据 [[SpeechLanguageModel]] 的分类体系,属于 "instruction-tuning" 训练阶段 + "real-time interaction" 生成范式。根据 [[Speech-LLMIntegrationTaxonomy]] 的分类,输入侧采用 latent-representation-based integration (Whisper encoder + convolutional downsampling adaptor),输出侧通过 NAR decoder 生成 HuBERT discrete units 再经 vocoder 合成。

**已有认知**:
- **Modality Adaptation** ([[ModalityAdaptationforSpeechLLM]]): LLaMA-Omni 的 speech adaptor 属于最基础的 **convolutional downsampling** 类别 (5x downsample + 2-layer MLP),是实现最简单、参数最少但固定降采样率的方案。KB 记载该方法排序为 Q-Former > CTC Compression > Conv Downsampling,本文选择最简单方案但仍取得 competitive 效果,说明 adaptor 复杂度并非 speech interaction 的瓶颈 [agent 解读]。
- **Streaming Spoken Dialogue** ([[StreamingSpokenDialogue]]): KB 已记载 LLaMA-Omni 的 NAR CTC decoder 方案,与 Mini-Omni (delayed parallel decoding)、Moshi (RQ-Transformer fully causal)、IntrinsicVoice (GroupFormer) 形成对比。LLaMA-Omni 的独特之处在于用 CTC 做变长对齐,无需预对齐 speech-text,是 streaming 方案中最轻量的之一。
- **Non-autoregressive TTS** ([[Non-autoregressiveTTS]]): LLaMA-Omni 的语音解码器本质上是一个 NAR 模型,用 CTC loss 训练而非传统 attention 或 duration predictor。NAR 的优势是 O(1) 推理复杂度,劣势是建模能力有限 -- 这一局限在后续 LLaMA-Omni 2 中通过换用 AR decoder 得到解决。
- **前身与后续**: 本文团队 (ICT/CAS Yang Feng 组) 后续提出 [[论文笔记/LLaMA-Omni2|LLaMA-Omni 2]],用 CosyVoice 2 的 AR 流式解码器替代 NAR CTC decoder,以牺牲一定延迟 (~600ms vs 236ms) 换取显著更自然的语音生成。

**创新判断**: 本文核心创新不在于单个组件,而在于 (1) 首次将 CTC-based NAR streaming decoder 用于 speech LLM 输出,实现 text+speech 同步生成; (2) 构造了面向语音交互场景的合成数据集 InstructS2S-200K; (3) 证明了 4 GPU 3 天即可训练一个 competitive 的 speech interaction 模型。

> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓ (confirmed) | 过滤: [[ModalityAdaptationforSpeechLLM]], [[Speech-LLMIntegrationTaxonomy]], [[StreamingSpokenDialogue]], [[Non-autoregressiveTTS]] (pending-review)

## 速查

> [!summary] 速查
> - **一句话**: 在 Llama-3.1-8B-Instruct 上挂载 Whisper 编码器和 CTC-based NAR 流式语音解码器,用 200K 合成数据训练,实现 236ms 延迟的 speech-in speech-out 交互
> - **路线**: Speech → Whisper-large-v3 encoder (frozen) → 5x downsample adaptor (concat + 2-layer MLP) → Llama-3.1-8B-Instruct → upsample (lambda=25) → 2-layer NAR Transformer decoder (CTC) → HuBERT discrete units → HiFi-GAN vocoder → Speech
> - **指标**: ChatGPT Score S2TIF 3.99, S2SIF 3.47; ASR-WER 10.82; UTMOS 3.93; Latency 236ms (streaming, Omega=10) [Table 2, Fig 3, Table 4]
> - **可借鉴**: (1) CTC 做 LLM hidden states → speech units 的变长对齐,免去预对齐开销; (2) upsample factor lambda 控制输出序列长度的设计简洁有效; (3) InstructS2S-200K 的三步数据构造流程 (指令改写→响应生成→语音合成) 可复制到其他语音交互数据集构建
> - **局限**: NAR decoder 建模能力有限导致语音自然度不如 AR 方案; 仅支持单轮英文交互; 训练数据仅~1K小时语音,规模受限; 无说话人/情感控制; ASR-WER (10.82) 高于级联系统 (3.78/6.77)

## 核心问题

GPT-4o 展示了低延迟语音交互的可能性,但开源社区缺乏对应方案 [§1]。现有路线有两类问题:
1. **级联系统 (ASR+LLM+TTS)** 延迟高,因为三个模块串行执行 [§1 para 2]
2. **已有 SpeechLM (如 SpeechGPT)** 虽然理论上可以端到端,但实践中为了质量需要先生成中间文本再生成语音 (chain-of-modality),延迟极高 (>4500ms) [§4.5, Table 5]

**本文要解决的问题**: 如何在保持高质量文本响应的同时,以极低延迟同步生成语音响应? [§1 para 3]

## 方法: 它怎么 work

> [!important] 区分来源
> [论文原文] = author's explanation; [agent 解读] = my inference.

### 整体架构

LLaMA-Omni 由四个组件构成 [§2, Fig 2]:
1. **Speech Encoder**: Whisper-large-v3 的 encoder 部分,参数全程冻结 [§2.1]
2. **Speech Adaptor**: 5x 下采样 (concat 相邻帧) + 2-layer MLP (ReLU 激活) [§2.2, Eq 1-2]
3. **LLM**: Llama-3.1-8B-Instruct,接收适配后的语音表征,自回归生成文本响应 [§2.3, Eq 3]
4. **Speech Decoder**: 2-layer NAR Transformer (same arch as LLaMA),用 CTC 从 LLM hidden states 预测 HuBERT discrete units [§2.4, Eq 4-5]

关键的数据流: 用户语音 → Whisper encoder → adaptor (长度 N/5) → LLM 生成 text token 序列 Y^T (长度 M) → 每个 text token 的 hidden state 上采样 lambda 倍 → speech decoder → CTC 对齐 → discrete units → HiFi-GAN vocoder → 语音波形。

### 关键设计选择

**选择 1: 为什么用 CTC 而不是 attention-based 或 transducer?** [论文原文] CTC 天然支持变长映射 (variable-length mapping),通过 blank token + collapsing function 将固定长度输入映射为变长输出 [§2.4]。[agent 解读] CTC 的 NAR 特性使得每个 text token 对应的 speech units 可以并行生成,不增加整体解码时间 -- 这是实现"同步生成"的关键。如果用 AR decoder,speech 生成速度会成为瓶颈。论文在 related work 中明确将 CTC-based methods 列为三种 streaming generation 方法之一 [§5 para 2]。

**选择 2: Upsample factor lambda = 25** [§4.1]。[agent 解读] 每个 text token 的 hidden state 被复制 25 次后送入 speech decoder。这意味着如果 LLM 生成 M 个 text token,speech decoder 的输入长度为 25M。由于目标 HuBERT units 的平均序列长度为 553.6 [Table 1],而平均文本响应长度为 39.5 tokens,所以 lambda=25 使得 25*39.5=987.5 >> 553.6,给 CTC 留出了足够的 blank 空间做对齐。

**选择 3: 为什么 speech decoder 只用 2 层?** [论文原文] Speech decoder 含 2 个 Transformer 层,hidden dim 4096, 32 heads, FFN dim 11008, 共 425M 参数 [§4.1]。[agent 解读] 选择 2 层是效率与表达力的平衡。由于 decoder 已经接收了 LLM 的 high-level hidden states 作为输入 (而非原始语音特征),其任务是将语义表征映射为 speech units,不需要很深的网络。更多层会增加计算量,与"低延迟"目标冲突。

**选择 4: HuBERT discrete units (K=1000) 而非 codec tokens** [§2.4]。[论文原文] 沿用 SpeechGPT 的做法,用 HuBERT 提取连续表征,k-means 量化为 1000 类离散 units,再 dedup 连续重复 [§2.4]。[agent 解读] 选择 HuBERT semantic units 而非 acoustic codec tokens (如 EnCodec) 有两个原因: (1) semantic units 更适合与文本对齐,因为它们编码语义而非声学细节; (2) 单 codebook 的 semantic units 可以用简单的 CTC 建模,而 multi-codebook codec tokens 需要更复杂的解码策略 (如 Moshi 的 depth transformer)。

**选择 5: Speech adaptor 的 5x downsampling** [§2.2, §4.1]。[论文原文] 每 k=5 个连续帧沿 feature dimension 拼接,然后通过 MLP 映射到 LLM embedding space [Eq 1-2]。[agent 解读] Whisper encoder 输出约 50 fps (对 16kHz 音频的 30ms hop),5x 下采样后变成 10 fps,一段 10 秒的语音输入变成约 100 个 token,接近文本 token 的长度量级,降低了 LLM 处理语音的计算负担。

### 训练策略

两阶段训练 [§2.5, Fig 2 right]:

**Stage 1: Speech understanding** -- 冻结 Whisper encoder,训练 adaptor + LLM (全参微调),使 LLM 能从语音输入直接生成文本响应。Loss = cross-entropy (Eq 3)。此阶段不涉及 speech decoder。

**Stage 2: Speech generation** -- 冻结 encoder + adaptor + LLM (全部),仅训练 speech decoder。Loss = CTC loss (Eq 5)。

[论文原文] 两阶段策略的设计逻辑: 第一阶段让 LLM 先学会理解语音和生成合适的文本响应; 第二阶段在冻结主体模型的情况下训练 speech decoder,确保语音输出与文本输出对齐 [§2.5]。

[agent 解读] 这种分阶段冻结策略有工程优势: Stage 2 仅训练 425M 参数的 decoder,显存需求远低于 Stage 1 (8B LLM)。同时避免了 speech decoder 的梯度影响 LLM 的文本生成能力。但代价是 speech decoder 只能"被动跟随" LLM 的 hidden states,无法通过反向传播影响 LLM 去生成更适合语音输出的 hidden states。

训练超参 [§4.1]:
- Stage 1 & 2: batch size 32, 3 epochs, cosine LR with 3% warmup
- Stage 1 peak LR: 2e-5; Stage 2 peak LR: 2e-4 (高 10x,因为 decoder 从零开始训练)
- 总训练时间: ~65 小时, 4x NVIDIA L40 GPUs

### InstructS2S-200K 数据集构造

三步流程 [§3]:
1. **Instruction Rewriting**: 用 Llama-3-70B-Instruct 改写文本指令,添加 filler words (hey, so, uh, um),将数字转为口语形式,使指令简短 [§3 Step 1]
2. **Response Generation**: 用 Llama-3-70B-Instruct 生成面向语音场景的简洁响应,禁止括号/有序列表等不可合成内容 [§3 Step 2]
3. **Speech Synthesis**: 指令用 CosyVoice-300M-SFT (随机男女声) 合成; 响应用 VITS (LJSpeech 训练的标准女声) 合成 [§3 Step 3]

数据来源: ~50K from Alpaca + ~150K from UltraChat (仅第一轮) [§3 para 4]。

统计: 指令平均 7.5s / 21.7 words; 响应平均 19.0s / 39.5 words; 平均 unit 序列长度 553.6 [Table 1]。

### 流式推理机制

[§2.6, Algorithm 1] LLM 自回归逐 token 生成文本响应。由于 speech decoder 使用 causal attention,每生成一个 text token,对应的 upsampled hidden states 就可以送入 decoder 生成 partial alignment。当累积的 discrete units 数量达到预定义的 chunk size Omega 时,立即送入 vocoder 合成一段语音播放给用户。

[论文原文] 由于 speech decoder 使用 NAR 建模,每个 text token 对应的 lambda 个 alignment 位置是并行生成的,因此同时生成 text+speech 的速度与只生成 text 几乎相同 [§2.6 para 2]。

## 实验

### 主要结果 (Offline)

| 指标 | LLaMA-Omni | SpeechGPT | SALMONN+Orca | Qwen2-Audio+Orca | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| ChatGPT Score (S2TIF) | **3.99** | 2.98 | 3.44 | 3.47 | InstructS2S-Eval | [Table 2] |
| ChatGPT Score (S2SIF) | **3.47** | 2.19 | 3.40 | 3.38 | InstructS2S-Eval | [Table 2] |
| ASR-WER | 10.82 | 45.00 | **3.78** | 6.77 | InstructS2S-Eval | [Table 2] |
| UTMOS | **3.93** | 3.90 | 3.83 | 3.61 | InstructS2S-Eval | [Table 2] |

### 主要结果 (Streaming)

| 指标 | LLaMA-Omni (Omega=10) | LLaMA-Omni (Omega=40) | LLaMA-Omni (Omega=100) | SpeechGPT (Omega=10) | 出处 |
| --- | --- | --- | --- | --- | --- |
| Latency (ms) | **236** | 347 | 563 | 4930 | [Table 4, 5] |
| ChatGPT Score | 3.54 | 3.52 | 3.49 | 2.16 | [Table 4, 5] |
| ASR-WER | 9.84 | 10.37 | 10.71 | 44.85 | [Table 4, 5] |
| UTMOS | 3.23 | 3.67 | 3.82 | 2.71 | [Table 4, 5] |
| WPS | 2.76 | 2.74 | 2.74 | 1.95 | [Table 4, 5] |

### 与级联系统的流式对比关键发现

[§4.5] 级联系统 (SALMONN/Qwen2-Audio + Orca) 在低延迟时有两个显著问题:
1. ChatGPT Score 和 ASR-WER 与 offline 差距大 -- 词级 streaming TTS 容易引入额外错误 [Fig 3a, 3b]
2. 低延迟时 WPS 急剧下降 (Theta=1 时 ~1.86 WPS vs offline ~3.3 WPS) -- 词间停顿频繁,自然度下降 [Fig 3d]

LLaMA-Omni 的优势: 语音率 (WPS) 在不同延迟条件下几乎不变 (~2.74-2.76) [Fig 3d],因为 streaming 是在 unit 级别做的而非词级别,整体韵律和节奏不受影响 [§4.5]。

### 人类评估

5 名评估者对 LLaMA-Omni (Omega=40, 347ms) vs 级联系统进行 side-by-side 比较 [§4.6]:
- vs SALMONN+Orca: helpfulness 48 win / 28 tie / 24 lose; naturalness 44 win / 23 tie / 33 lose [Fig 4]
- vs Qwen2-Audio+Orca: helpfulness 50 win / 20 tie / 30 lose; naturalness 42 win / 38 tie / 20 lose [Fig 4]

## 局限性

1. **NAR decoder 建模能力受限**: CTC-based NAR decoder 无法建模 unit 间的依赖关系,语音自然度不如 AR 方案。原文在结论中也承认需要"explore enhancing the expressiveness of generated speech responses" [§6]。后续 LLaMA-Omni 2 用 AR decoder 替代后进一步证实了这一局限 [agent 解读]
2. **ASR-WER 偏高**: 10.82 vs 级联系统的 3.78/6.77,说明 speech-text 对齐仍有较大改进空间 [Table 2]。[论文原文] 作者将其归因于 speech decoder 仅在 ~1K 小时数据上训练,远少于工业级 TTS 模型 [§4.4]
3. **单轮英文**: 不支持多轮对话,不支持多语言 [agent 解读, 基于数据集设计 §3]
4. **固定声音**: 响应语音用单一 LJSpeech 女声 (VITS),无说话人/情感/韵律控制 [§3 Step 3]
5. **评估局限**: 自建 InstructS2S-Eval (199 条指令) 作为唯一评估集,且 S2SIF 评估依赖 ASR 转写,无法反映副语言信息质量 [§4.2]

## 点评

**核心贡献的价值**: LLaMA-Omni 的最大价值不在于刷指标,而在于证明了一个 recipe: **在最新开源 LLM 上以极低成本 (4 GPU, 3 天, 200K 样本) 构建 competitive 的 speech interaction 系统是可行的**。这在 2024 年 9 月 (GPT-4o 刚发布不久) 的时间点上具有实际的开拓意义。

**CTC 做 speech decoder 的精妙之处**: 用 CTC 来处理 LLM hidden states → speech units 的变长映射,避免了两个棘手问题: (1) 不需要预对齐训练数据中的 text-speech 对应关系; (2) 不需要在推理时做额外的 duration prediction。CTC 的 blank + collapsing 机制天然解决了变长映射。这比 SpeechGPT 的 chain-of-modality (先全部生成文本再生成语音) 优雅得多,也比 Mini-Omni 的 delayed parallel decoding 概念上更简洁。

**数据构造的实用性**: InstructS2S-200K 的三步流程 (LLM 改写指令为口语风格 → LLM 生成简洁响应 → TTS 合成) 是一个可复制的 recipe。特别是"响应风格适配"这一步 -- 现有文本指令数据的响应通常冗长、包含格式化元素,不适合语音场景。Table 3 的 case study 直观展示了这一差异 [Appendix B]。

**作为 baseline 的角色**: LLaMA-Omni 已成为后续 speech interaction 工作的重要 baseline (被 Freeze-Omni, SLAM-Omni, LLaMA-Omni 2 等引用对比)。其开源实现 (GitHub + HuggingFace model) 降低了社区的入门门槛。

**与 Moshi 的对比**: 同期的 Moshi (Kyutai, 2024) 走的是完全不同的路线 -- 全量预训练、全因果架构、全双工、7B 规模。LLaMA-Omni 则是 modular 路线的极致精简: 复用已有 Whisper + Llama-3.1 + HuBERT + HiFi-GAN,仅训练 adaptor + LLM LoRA(实际是全参) + 425M decoder。两者代表了 native vs modular SpeechLM 的两种哲学。

## 可复用的 idea

1. **CTC 做 LLM-to-speech 的变长对齐**: 任何需要从 LLM hidden states 生成变长序列的场景 (不限于 speech units) 都可以考虑 CTC 方案,免去显式 duration 预测或 attention-based 对齐
2. **Upsample + CTC 组合**: 先用固定倍率 lambda 上采样 LLM hidden states,再用 CTC 的 blank 机制吸收多余长度。这比直接用 duration predictor 更灵活,因为 CTC 的对齐是隐式学习的
3. **InstructS2S 数据构造流程**: 用强 LLM 将文本指令数据改写为口语风格 + 生成简洁响应 + TTS 合成。适用于任何需要构建 speech interaction 训练数据的场景
4. **两阶段解耦训练**: Stage 1 训 understanding (adaptor + LLM), Stage 2 冻结主体训 generation (decoder)。适用条件: decoder 模块相对独立且不需要通过反向传播影响 LLM 内部表征。不适用于需要 LLM 根据 decoder 反馈调整生成的场景 (如 prosody-aware text generation)。LLaMA-Omni 2 正是将 Stage 2 的 decoder 换为 AR 模型来升级语音质量
5. **Streaming 的 chunk-based vocoder 输出**: 用 minimum chunk size Omega 控制延迟-质量 trade-off,从 236ms (Omega=10) 到 563ms (Omega=100),用户可根据应用场景灵活调节
