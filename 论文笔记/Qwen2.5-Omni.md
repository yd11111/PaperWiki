---
type: paper
tier: deep
title: "Qwen2.5-Omni Technical Report"
arxiv_id: "2503.20215"
source: "Alibaba Qwen Team"
authors: [Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, Junyang Lin]
year: 2025
venue: "arXiv"
tags: [omni-model, multimodal, streaming, speech-generation, thinker-talker, end-to-end, TMRoPE, DPO, flow-matching]
concepts: ["[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLM Integration Taxonomy]]", "[[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation]]", "[[概念库/ConditionalFlowMatching|Conditional Flow Matching]]", "[[概念库/SpeechTokenizer|Speech Tokenizer]]", "[[概念库/AudioUnderstanding|Audio Understanding]]", "[[概念库/Full-duplexSpokenDialogue|Full-duplex Spoken Dialogue]]", "[[概念库/StreamingSpokenDialogue|Streaming Spoken Dialogue]]", "[[概念库/LLM-basedTTS|LLM-based TTS]]"]
models: ["Qwen2.5-Omni-7B", "Qwen2.5-VL-7B", "Qwen2-Audio", "Whisper-large-v3", "MiniCPM-o", "MaskGCT", "CosyVoice2"]
tasks: ["multimodal-understanding", "speech-generation", "ASR", "TTS", "video-understanding", "audio-reasoning"]
datasets: ["LibriSpeech", "Fleurs", "CommonVoice", "seed-tts-eval", "OmniBench", "VoiceBench", "MMAU", "MMLU", "GSM8K"]
kb_context_sources: 7
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 7 个实体页: SpeechLanguageModel, Speech-LLMIntegrationTaxonomy, ModalityAdaptationforSpeechLLM, SpeechTokenizer, AudioUnderstanding, ConditionalFlowMatching, Full-duplexSpokenDialogue)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: 7 | 过滤: 0 | 未命中但可能相关: DiffusionModel (DiT 架构)

**定位**: Qwen2.5-Omni 在 KB 体系中跨越多个概念节点。从 [[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLM Integration Taxonomy]] 看,它是**混合集成路线**: 输入侧使用 latent-representation-based 方法 (Whisper/ViT encoder 产出连续表征送入 LLM),输出侧使用 audio-token-based 方法 (Talker 生成离散 speech tokens)。这种输入连续+输出离散的混合设计在该分类框架中较为独特,不同于纯 latent (Qwen-Audio, SALMONN) 或纯 token (SpeechGPT, Moshi) 路线。

**对比同类系统**: 在 [[概念库/SpeechLanguageModel|SpeechLM]] 演进中,Qwen2.5-Omni 与 Moshi (全双工, Mimi tokenizer, RQ-Transformer)、Mini-Omni (text+7 acoustic streams 并行)、VITA (IPR + 多模态) 同处 2024-2025 omni-model 阶段。但 Qwen2.5-Omni 的独特处在于 Thinker-Talker 分离设计 -- Thinker 专注文本生成,Talker 接收 Thinker 的高维表征生成语音,避免了 Moshi 用 Inner Monologue 统一建模 text+speech 可能带来的模态干扰。

**Streaming 技术关联**: [[概念库/StreamingSpokenDialogue|Streaming Spoken Dialogue]] 中的核心技术 (block-wise attention, causal convolution, delayed parallel decoding) 在 Qwen2.5-Omni 中有对应实现: block-wise audio/vision encoder + sliding-window DiT。但 Qwen2.5-Omni 并非全双工系统 (不支持打断),streaming 能力限于输出侧的低延迟生成。

**Speech codec + Flow Matching**: Qwen2.5-Omni 的 qwen-tts-tokenizer 生成离散 speech tokens,再由 [[概念库/ConditionalFlowMatching|Flow-Matching]] DiT 转换为 mel spectrogram,最后由 BigVGAN 合成波形。这是典型的 LLM-based TTS hybrid 路线 (token → CFM → vocoder),与 CosyVoice 系列的 semantic token + CFM 管线同构。

## 速查

> [!summary] 速查
> - **一句话**: 统一的端到端多模态模型,同时感知 text/image/audio/video 并**流式**生成 text + 自然语音,通过 Thinker-Talker 分离架构避免文本和语音生成的模态干扰
> - **路线**: `Audio/Image/Video → Block-wise Encoder + TMRoPE → Thinker (Transformer Decoder, text generation) → Talker (Dual-track AR, speech token generation) → Flow-Matching DiT → BigVGAN → Waveform`
> - **指标**: OmniBench 56.13% (SOTA, +13% over prev best) | VoiceBench 74.12 avg | seed-tts-eval WER 1.42/2.33/6.54 (zh/en/hard) | 单说话人 NMOS 4.51 (zh, 接近人类 4.51) | MMAU 65.60 (audio reasoning SOTA)
> - **可借鉴**: (1) Thinker-Talker 分离设计,用 hidden representation 传递语义而非串行 pipeline; (2) TMRoPE 将时间戳编码到位置嵌入中同步音视频; (3) DPO 优化语音生成稳定性 (基于 WER + 标点停顿错误率构建正负对)
> - **局限**: 未开源训练数据和训练代码 | 仅支持 streaming 输出,不支持全双工 (无打断能力) | 语音生成的 speaker similarity 弱于 Seed-TTS/CosyVoice 2 | 7B 参数,推理计算成本高

## 核心问题

Qwen2.5-Omni 要解决的核心问题是: **如何在单一模型中统一多模态理解和 text+speech 同步流式生成,同时避免模态间的训练/推理干扰?** [§1]

这个问题可分解为三个子问题:

1. **跨模态时间对齐** [§1, §2.2]: 视频包含音频和视觉两种时间流,如何让模型准确感知它们的时间同步关系?传统方法将音视频分别编码,丢失了时间对齐信息。

2. **多模态输出干扰** [§1, §2.3]: 同时训练 text 和 speech 两种输出时,两者会相互干扰 (text loss 和 speech loss 在梯度空间竞争),导致两个能力都下降。[论文原文]

3. **流式推理延迟** [§2.4]: 实时交互要求模型能边接收输入边生成输出,但多模态编码器通常需要完整输入 (non-causal attention),音频解码也需要完整 token 序列。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Qwen2.5-Omni 采用 **Thinker-Talker** 双模块架构 [§2.1, Fig 2]:

- **Thinker** (大脑): Transformer decoder-only LLM,配备 audio encoder (Whisper-large-v3 初始化) 和 vision encoder (Qwen2.5-VL ViT, 675M 参数)。处理所有模态输入,生成文本和高维语义表征。[§2.1]
- **Talker** (嘴巴): Dual-track autoregressive Transformer decoder (受 Mini-Omni 启发),接收 Thinker 的隐藏表征 + 采样后的 text token embedding,流式输出离散 speech tokens。[§2.1, §2.3]
- **Speech Decoder**: qwen-tts-tokenizer 定义的离散 speech tokens → Flow-Matching DiT (sliding-window block attention) → BigVGAN → waveform。[§2.4]

**关键连接**: Talker 不是独立模型,它**直接访问 Thinker 所有历史上下文信息** (共享 KV cache) [§2.1]。[论文原文] 这意味着 Thinker-Talker 是一个"内聚的单一模型",支持端到端训练和推理。

### 关键设计选择

#### 1. TMRoPE: Time-aligned Multimodal RoPE [§2.2, Fig 3]

将 M-RoPE (Qwen2.5-VL 的多模态旋转位置编码) 扩展为显式包含绝对时间信息:

- **三维分解**: 将 RoPE 拆为 temporal / height / width 三个分量 [§2.2]
- **文本和纯音频**: 三个分量使用相同 position ID (退化为 1D-RoPE),音频引入绝对时间编码 (1 temporal ID = 40ms) [§2.2]
- **视频+音频**: 按 2 秒为单位切分,每段内先放视觉表征再放音频表征 (time-interleaving),视频帧间的 temporal ID 按实际时间动态调整 [§2.2]

**为什么需要这样做** [agent 解读]: 视频帧率不固定 (动态采样),而音频帧率固定 (40ms/帧)。TMRoPE 通过绝对时间锚点让两种模态在注意力计算时能感知彼此的时间位置,避免了固定采样率假设导致的错位。

#### 2. Thinker-Talker 分离 [§2.1, §2.3]

**设计动机** [论文原文]: "inspired by the way humans utilize different organs to produce various signals, which are simultaneously coordinated through the same neural networks" [§1]。

- Thinker 独立训练 text generation loss,Talker 独立训练 speech token prediction loss [§2.3]
- Talker 需要两种信息来生成语音 [§2.3]: (1) Thinker 的高维表征 -- 隐式传达语气、态度,使流式生成更自然; (2) 采样后的离散 text token embedding -- 消除语音上相似但含义不同的歧义
- 语音生成**不需要 word-level 或 timestamp-level 对齐** [§2.3]。[论文原文] 这显著简化了训练数据需求和推理流程。

**为什么不在同一个 head 上同时输出 text+speech** [agent 解读]: Mini-Omni 的实践表明并行多 track 输出在同一 decoder 中训练时会出现 track 间干扰。Thinker-Talker 将问题分解: Thinker 的 text generation 完全不受 speech loss 影响,Talker 只负责从已经稳定的高维表征中提取语音信号。

#### 3. qwen-tts-tokenizer [§2.3]

论文对此描述非常简短: "efficiently represents key information of speech and can be decoded to speech streamingly through a causal audio decoder" [§2.3]。未公开具体架构细节。

[agent 解读] 从上下文推断,这是一个将语音编码为离散 token 的 codec,支持流式解码。与 CosyVoice 的 FSQ-SenseVoice 或 Moshi 的 Mimi 类似,但具体量化方法 (RVQ/FSQ/VQ) 和帧率未披露。

#### 4. Sliding-window DiT for Streaming [§2.4, Fig 4]

传统 flow-matching DiT 需要全序列 attention,不支持流式。Qwen2.5-Omni 提出:

- 将相邻 speech tokens 分组为 blocks [§2.4]
- DiT 的 attention mask 限制为 **4 blocks** (lookback 2 + 当前 1 + lookahead 1) [§2.4]
- Flow Matching 按 chunk 生成 mel spectrogram,每个 chunk 可访问上下文 blocks [§2.4]
- BigVGAN 也按固定感受野 chunk-by-chunk 生成波形 [§2.4]

### 训练策略

#### Pre-training: 三阶段渐进 [§3]

| 阶段 | LLM 参数 | 训练重点 | 数据量 | 序列长度 |
| --- | --- | --- | --- | --- |
| Stage 1 | 冻结 | Audio/Vision encoder + adapter | 大量 audio-text + image-text pairs | 8192 |
| Stage 2 | 解冻 | 全参数,多模态混合 | 800B image/video + 300B audio + 100B video+audio tokens | 8192 |
| Stage 3 | 解冻 | 长序列能力 | 扩展至长音频/视频数据 | 32768 |

**初始化** [§3]: LLM 从 Qwen2.5 初始化,vision encoder 同 Qwen2.5-VL,audio encoder 从 Whisper-large-v3 初始化。两个 encoder 先分别训练 adapter,再训练 encoder 本身,最后解冻 LLM 全参数联合训练。

[agent 解读] 这种 "先锁 LLM 训 encoder → 全解冻" 的策略与 [[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation]] 中 Wu et al. (2023) 提出的两阶段训练策略一致 -- 避免不稳定的 encoder 梯度干扰 LLM 的已有知识。

#### Post-training: Thinker + Talker 分离 [§4]

**Thinker** [§4.2]: 使用 ChatML 格式的 instruction-following 数据做 SFT,覆盖纯文本/图像/音频/混合模态对话。

**Talker** [§4.3]: 三阶段训练:

1. **ICL 训练**: 在多模态上下文 + 语音响应的对话数据上做 next-token prediction,学习从语义到语音的单调映射 + 韵律/情感/口音等副语言属性。引入**音色解耦技术** (timbre disentanglement) 防止模型将特定声音与低频文本模式绑定 [§4.3]。

2. **DPO 强化学习** [§4.3]: 构建 triplet (x, y_w, y_l) 数据集,基于 WER 和标点停顿错误率对生成语音排序,使用标准 DPO loss (Eq. 1) 优化。**目的**: 缓解预训练数据中标签噪声和发音错误导致的幻觉问题 [论文原文]。

3. **Speaker Fine-tuning** [§4.3]: 在 DPO 后的基础模型上做多说话人 instruction fine-tuning,使 Talker 可采用特定声音并提升自然度。

## 实验

### 多模态理解 (X→Text)

| 指标 | Qwen2.5-Omni-7B | 对比基准 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ASR WER | 1.8 / 3.4 | Qwen2-Audio: 1.6 / 3.6 | LibriSpeech test-clean/other | Table 2 |
| Audio Reasoning Avg | 65.60 | Gemini-1.5-Pro: 54.90, Qwen2-Audio: 49.20 | MMAU | Table 3 |
| VoiceBench Avg | 74.12 | MiniCPM-o: 71.69, Baichuan-Omni-1.5: 71.14 | VoiceBench | Table 3 |
| Speech MMLU | 65.6 | Qwen2-7B (text): 69.3, Qwen2-Audio: 33.2 | In-house voice-chat | Table 4 |
| Speech GSM8K | 85.4 | Qwen2-7B (text): 82.3, Qwen2-Audio: 18.4 | In-house voice-chat | Table 4 |
| MMMU_val | 59.2 | Qwen2.5-VL-7B: 60.0, GPT-4o-mini: 58.6 | MMMU | Table 5 |
| OmniBench Avg | 56.13% | Gemini-1.5-Pro: 42.91%, MiniCPM-o: 40.5% | OmniBench | Table 8 |
| Video-MME w/o sub | 64.3 | Qwen2.5-VL-7B: 65.1, GPT-4o-mini: 64.8 | Video-MME | Table 7 |

### 语音生成 (X→Speech)

| 指标 | Qwen2.5-Omni-7B (RL) | 对比基准 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Zero-shot WER (zh/en/hard) | 1.42 / 2.33 / 6.54 | CosyVoice 2: 1.45 / 2.57 / 6.83; MaskGCT: 2.27 / 2.62 / 10.27 | seed-tts-eval | Table 9 |
| Zero-shot SIM (zh/en/hard) | 0.754 / 0.641 / 0.752 | CosyVoice 2: 0.748 / 0.652 / 0.724; Seed-TTS_RL: 0.801 / 0.766 / 0.782 | seed-tts-eval | Table 9 |
| Single-speaker WER (zh/en/hard) | 1.29 / 1.86 / 6.59 (Speaker A) | Human: 1.25 / 2.14 / - | seed-tts-eval | Table 10 |
| Single-speaker NMOS (zh/en) | 4.51 / 4.62 (Speaker A) | Human: 4.51 / 4.46 / 4.51 | Self-curated | Table 10 |

**关键发现**:

1. **语音指令跟随接近文本指令** [§5.1.2, Table 4]: Speech MMLU 65.6 vs Text MMLU 69.3,差距仅 3.7 分;Speech GSM8K 85.4 甚至超过 Text 82.3。这说明 speech input 不再是理解能力的瓶颈,Qwen2.5-Omni 的端到端语音理解已接近纯文本能力。

2. **DPO 显著改善语音稳定性** [§5.2.1, Table 9]: ICL→RL 后 test-hard WER 从 7.97 降至 6.54 (18% 相对改善),说明 DPO 有效减少了注意力错位和发音错误。

3. **Speaker similarity 的差距** [Table 9]: SIM 0.754/0.641/0.752 vs Seed-TTS_RL 0.801/0.766/0.782,说明 Qwen2.5-Omni 作为通用 omni 模型在说话人克隆精度上仍不如专用 TTS 系统。[agent 解读] 这可能因为 qwen-tts-tokenizer 偏重语义而非声学细节。

4. **多模态理解 SOTA** [Table 8]: OmniBench 56.13% 大幅超越前最佳 42.91% (+13%),证明统一多模态训练确实带来了跨模态融合的显著提升。

## 局限性

1. **非全双工** [agent 解读]: 虽然支持 streaming 输出,但不支持用户打断 (interruption) 和同步双向通信。与 Moshi (全因果, 160ms) 和 Raon-SpeechChat (SIL/BOW/BC 状态控制) 相比,Qwen2.5-Omni 在实时交互能力上有明显差距。后续 Qwen3.5-Omni 已通过 ARIA 机制部分解决此问题。

2. **Speech codec 未披露细节** [§2.3]: qwen-tts-tokenizer 的架构、量化方法、帧率、codebook 大小等关键信息均未公开,无法评估其设计选择是否最优,也无法复现。

3. **Speaker similarity 偏低** [Table 9]: 零样本 SIM 最高 0.754 (zh),显著低于 Seed-TTS_RL 0.801。对于需要高保真声音克隆的应用场景,Qwen2.5-Omni 可能不够。

4. **Text→Text 能力下降** [Table 1]: 多模态联合训练导致纯文本能力在 Qwen2-7B 和 Qwen2.5-7B 之间,如 MMLU-Pro 47.0 vs Qwen2.5-7B 56.3,回退明显。这是多模态统一模型的常见 trade-off。

5. **无音频编辑/风格控制能力** [agent 解读]: 论文未讨论 Talker 是否支持细粒度语音风格控制 (如情感、语速指令),输出语音的风格主要取决于 ICL prompt 和 speaker fine-tuning,不如 Step-Audio 等系统灵活。

6. **延迟数据缺失** [agent 解读]: 论文未报告首包延迟 (first packet latency),尽管讨论了 streaming 设计,但无法与 Moshi (230ms)、Freeze-Omni 等系统做延迟对比。

## 点评

**Thinker-Talker 分离的工程智慧**: 这个设计看似简单,实际上优雅地解决了 omni 模型的核心矛盾 -- text generation 和 speech generation 在 loss landscape 上的竞争。Moshi 通过 Inner Monologue 让 LLM 同时生成 text+speech tokens,虽然统一但代价是两种能力都会打折扣。Qwen2.5-Omni 让 Thinker 全力做文本推理 (保持 LLM 原有能力),Talker 专注从已经"想好了"的语义中提取语音,各司其职。这个思路的本质是: **text generation 是高阶认知任务,speech generation 是低阶执行任务,两者不应在同一抽象层竞争参数**。

**TMRoPE 的实用价值**: 绝大多数多模态模型对音视频的时间对齐处理粗糙 (简单拼接或按固定帧率插值),TMRoPE 将绝对时间戳直接编码到位置嵌入中,使模型在 attention 层面就能感知跨模态的时间关系。虽然技术上不复杂 (只是 M-RoPE + 时间 ID),但这种"把物理时间写进数学结构"的思路值得借鉴。

**DPO 在 TTS 中的有效应用**: 论文用 WER + 标点停顿错误率作为 reward signal 构建偏好对,这比手工标注成本低且可大规模执行。test-hard WER 从 7.97 降到 6.54 (18% 相对改善) 证明了自动化偏好信号在 TTS robustness 上的有效性。这与 SpeechAlign 和 Multi-Reward GRPO 等工作的方向一致,说明 RLHF/DPO 在语音生成领域正在成为标准工具。

**隐含的代价**: 论文未讨论但值得注意: Talker 直接访问 Thinker 的所有 KV cache,这意味着推理时内存开销接近两倍 (Thinker 和 Talker 各自的 KV 存储)。对于 7B 级别模型,这在部署时可能是显著的约束。

## 可复用的 idea

1. **Thinker-Talker 分离用于任何 multi-output 系统** [§2.1, §2.3]: 当一个模型需要同时生成不同模态/粒度的输出时 (如 text + code, text + image),可将"思考"和"执行"分离到两个 module,用 hidden representation 连接而非串行 pipeline。关键是让 Talker 共享 Thinker 的 KV cache 而非重新编码,既保留上下文又避免 loss 竞争。

2. **TMRoPE 用于任何涉及时间对齐的多模态场景** [§2.2]: 当两种模态有不同采样率但需要时间同步时 (如字幕+视频、音乐+歌词、传感器数据+音频),可将绝对时间戳编码到 RoPE 的一个维度中,无需显式对齐模块。

3. **DPO with automatic speech metrics** [§4.3]: 用 ASR WER 和标点/停顿错误率自动构建 TTS 的偏好数据对 (y_w, y_l),替代人工标注。可推广到任何可自动评估的生成任务: 只要有可靠的自动评分,就能低成本构建 DPO 训练数据。

4. **Sliding-window block attention for streaming flow matching** [§2.4, Fig 4]: 将 DiT 的全局 attention 限制为 lookback 2 + current 1 + lookahead 1 blocks,实现流式 mel 生成。这个技术可直接移植到任何需要将 non-causal flow-matching 模型改造为 streaming 的场景。

5. **Block-wise encoder for chunked prefilling** [§2.4]: 将音频 encoder 从 full attention 改为 2 秒 block attention,使多模态编码器兼容 chunked prefill。这对部署多模态模型到在线推理框架 (如 vLLM) 很有实用价值。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 架构、训练流程、关键设计选择均有清晰描述和 section 引用 |
> | 可信赖 | pass | 实验数据均标注表格出处,原文/解读严格区分 |
> | 可区分 | pass | KB 背景中与 Moshi/Mini-Omni/VITA 做了定位对比 |
> | 可定位 | pass-with-fixes | 概念链接覆盖充分,但 qwen-tts-tokenizer 因信息不足无法与 KB 中的 tokenizer 分类对应 |
> | 不污染 | pass | 所有推断标注 [agent 解读] |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> - [medium] qwen-tts-tokenizer 的具体规格未知,笔记中已标注信息不足,但无法给出与 AudioTokenizerTaxonomy 的精确映射
> - [low] 论文未报告推理延迟,笔记局限性已提及但无法做系统级延迟对比
> 详见 `_review/Qwen2.5-Omni-review.yml`
