---
type: paper
tier: deep
title: "LLMVoX: Autoregressive Streaming Text-to-Speech Model for Any LLM"
arxiv_id: "2503.04724"
source: "Sources/LLMVoX.pdf"
authors: [Sambal Shikhar, Mohammed Irfan Kurpath, Sahal Shaji Mullappilly, Jean Lahoud, Fahad Khan, Rao Muhammad Anwer, Salman Khan, Hisham Cholakkal]
year: 2025
venue: "arXiv"
tags: [TTS, streaming, LLM-agnostic, autoregressive, speech-tokenizer, low-latency, plug-and-play, speech-LM]
concepts: ["[[Streaming Spoken Dialogue]]", "[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Codec Language Model]]", "[[Residual Vector Quantization]]", "[[Speech-LLM Integration Taxonomy]]"]
models: ["[[Whisper]]", "[[EnCodec]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[LLM-based TTS]], [[Speech Tokenizer]], [[Residual Vector Quantization]], [[Whisper]]; 2 个待确认页: [[Streaming Spoken Dialogue]] [待确认], [[Speech-LLM Integration Taxonomy]] [待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[Residual Vector Quantization]]✓, [[Streaming Spoken Dialogue]], [[Speech-LLM Integration Taxonomy]], [[Codec Language Model]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: LLMVoX 处于 [[Speech-LLM Integration Taxonomy]] 中 **Text-based Integration (Cascaded)** 路线 [待确认],即 ASR + LLM + TTS 管线串联。与端到端 Speech-LLM 路线 (如 Moshi、LLaMA-Omni、Freeze-Omni) 对立。根据已有知识,cascaded 方式的主要优势是保留 LLM 能力、模块可替换,主要劣势是延迟高。LLMVoX 的核心创新在于解决 cascaded 路线的延迟问题。

**已有认知**:
- [[Streaming Spoken Dialogue]] [待确认] 记录了多种 streaming 实现:Mini-Omni (delayed parallel decoding)、LLaMA-Omni (NAR CTC decoder)、Moshi (fully causal RQ-Transformer)、Freeze-Omni (chunk-wise streaming encoder)。LLMVoX 的 multi-queue streaming 是一种新的 streaming 机制,侧重在 **TTS 模块与 LLM 解耦** 的前提下实现流式。
- [[LLM-based TTS]] 记录了 codec LM TTS 范式(VALL-E 系列)和 hybrid 架构(CosyVoice)。LLMVoX 回到更轻量的 decoder-only transformer 直接预测 codec tokens,与 VALL-E 类似但规模远小 (30M vs 370M+)。
- [[Speech Tokenizer]] 和 [[Residual Vector Quantization]] 记录了各类语音离散化方案。LLMVoX 使用 WavTokenizer (单层 RVQ, 4096 entries) 作为 audio codec,属于 [[Single-codebook vs Multi-codebook]] 中的单码本路线。

**创新判断**: LLMVoX 的核心新意在于 (1) 极轻量 (30M) 的 LLM-agnostic streaming TTS,(2) multi-queue 双实例并行机制实现无限长度对话,(3) ByT5 G2P embedding 作为文本前端。这种"完全解耦 + 极轻量"的设计在已有知识库中没有对应物——Freeze-Omni 虽也 freeze LLM 但仍依赖 LLM hidden states,不是真正 LLM-agnostic。

> [!summary] 速查
> - **一句话**: 30M 参数的 LLM-agnostic autoregressive streaming TTS,通过 multi-queue 机制实现 475ms 端到端延迟,同时保留任意 LLM 的全部能力
> - **路线**: LLM text tokens → ByT5 G2P embedding + 前一帧 codec feature → concat + L2 norm + positional embedding → 4-layer decoder-only Transformer → speech token → WavTokenizer decoder → waveform (multi-queue 双实例并行)
> - **指标**: WER 3.70% / UTMOS 4.05 / 端到端延迟 475ms (Whisper Small + LLaMA-3.1-8B + LLMVoX) [Table 1]; Arabic CER 8.2% (streaming) [Table 3]; VSQA GPT score 6.41 / latency 1.05s [Table 4]
> - **可借鉴**: (1) multi-queue 双 TTS 实例 + 倍增 chunk size 的 streaming 方案,可用于任何需要长对话的 cascaded TTS 系统; (2) 直接复用 ByT5 G2P embedding layer 作为 phoneme-aware text frontend,避免显式 G2P 转换的延迟; (3) 仅用 LLM 输出文本 (非 hidden states) 作为 TTS 输入,实现真正的 plug-and-play
> - **局限**: 单说话人 / 无 voice cloning / ASR 未流式化 / 仅 2200h 训练数据 / WavTokenizer 单层 RVQ 可能限制音质上限 / 未评估韵律表现力

## 核心问题

1. **为什么已有 speech-enabled LLMs 需要微调 LLM?** 因为它们将 speech tokens 直接纳入 LLM 的词表或 hidden space,需要 LLM 学会处理新模态 [§1]。这会引发 catastrophic forgetting,损害 LLM 原有的推理和表达能力 [§1]。

2. **Cascaded pipeline (ASR+LLM+TTS) 虽然保留 LLM 能力,但为何延迟高?** 因为传统 TTS 模型需要等 LLM 生成完整文本 (或大段文本) 才能开始合成语音,且多数 TTS 使用非流式 decoder [§1]。LLM 文本是逐 token 增量产出的,而 TTS 不能利用这种增量性。

3. **LLMVoX 如何在保持 LLM 完整能力的同时实现低延迟?** 通过完全解耦: TTS 只接收 LLM 的文本输出 (非 hidden states),用独立的 30M 轻量 transformer 自回归生成 speech tokens,并通过 multi-queue streaming 与 LLM 并行运行 [§3, §4]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

LLMVoX 是一个纯 autoregressive 的 text-to-speech 系统,其架构由四个组件构成 [Fig 2]:

1. **WavTokenizer (frozen)**: 将 24kHz 语音波形编码为单层离散 token 序列,codebook size 4096,帧率 40-75 tokens/sec [§3.1]
2. **ByT5 G2P Embedding Layer (frozen)**: 将文本 word 分解为 byte-level sub-tokens,提取 phoneme-aware embedding (R^256),经 padding 对齐到 speech token 长度 [§3.2]
3. **Decoder-only Transformer (30M, trainable)**: 4 层, n_embd=768, n_head=8,接收 phoneme embedding + 前一步 codec feature 的拼接向量,自回归预测下一个 speech token [§3.4]
4. **WavTokenizer Decoder (frozen)**: 将生成的 speech token 序列解码回波形 [§3.1]

**数据流** [Fig 2]:
```
Text (from LLM) → ByT5 tokenizer → byte sub-tokens → ByT5 Embedding (R^256, frozen)
                                                        ↓
                                   [b_t ; f_{t-1}] → L2 norm → + positional embedding → z_t
                                                        ↓
                               4-layer Decoder-only Transformer (causal mask)
                                                        ↓
                                          p(S_t | S_{<t}, z, θ) → next speech token
                                                        ↓
                               Every n tokens → WavTokenizer Decoder → waveform chunk
```

### 关键设计选择

**1. 为什么使用 ByT5 G2P embedding 而非显式 G2P 转换?**

[论文原文] 两个理由: (1) ByT5 G2P 模型在 100+ 语言上微调,其 embedding 已编码丰富的跨语言音素信息; (2) 直接使用 embedding lookup 避免了显式 phoneme 预测的额外计算开销,降低延迟 [§3.2]。

[agent 解读] 这个设计隐含了一个重要权衡: 放弃了显式 phoneme 序列的可控性 (无法调整发音),换取了零额外延迟的 phoneme-aware 表征。对于 streaming 场景,这个 trade-off 是合理的。

**2. 为什么将 phoneme embedding 与前一帧 codec feature 拼接?**

[论文原文] 在每个时间步 t,输入向量 x_t = [b_t ; f_{t-1}],其中 b_t 是 phoneme embedding (R^256),f_{t-1} 是前一个 speech token 的 codec latent feature (R^512) [§3.3]。t=1 时用零张量初始化 f_0。

[agent 解读] 拼接 text + acoustic context 是 autoregressive TTS 的常见设计 (类似 VALL-E 的 text prefix + audio prefix)。但 LLMVoX 的独特之处在于: phoneme embedding 和 codec feature 是 **逐帧对齐** 的,而非前缀式。这意味着模型在每步都同时看到"该说什么音"和"之前说了什么声",实现隐式的 text-speech alignment。

**3. 为什么用单层 RVQ (WavTokenizer) 而非多层 (EnCodec)?**

[论文原文] WavTokenizer 使用单层 RVQ (4096 entries),产生紧凑表示,每秒 40-75 tokens [§3.1]。

[agent 解读] 单层 RVQ 极大简化了建模复杂度——不需要 VALL-E 式的 AR+NAR 两阶段,也不需要 Moshi 的 RQ-Transformer。这使 30M 参数的轻量模型成为可能。代价是重建质量可能不如多层 RVQ,但论文的 UTMOS 4.05 表明在对话场景下可接受。

**4. Multi-queue streaming 机制为什么有效?**

[论文原文] LLM 文本按句子边界分配到两个 FIFO 队列 Q1/Q2,两个 LLMVoX 副本分别从各自队列消费文本并生成 speech tokens [§4, Algorithm 1, Fig 3]。每 n 个 speech token 生成后立即解码为音频 chunk 并放入 producer queue,chunk size 每次翻倍 (n → 2n → 4n...) [§4]。

[论文原文] 倍增 chunk size 的理由: 利用之前音频 chunk 的播放时间作为 processing buffer,更大的 chunk 解码质量更好 [§4]。

[agent 解读] 这个双队列设计解决了 streaming TTS 的一个具体问题: 单个 TTS 实例处理长文本时,context window 可能不够。通过按句子拆分到两个实例,每个实例只需处理约一半的文本量。两个实例交替输出,实现"无限长度"对话。这是工程层面的巧妙设计,但也意味着需要 2x GPU 内存 (不过 30M 模型的 2x 几乎可忽略)。

### 训练策略

- **训练数据**: VoiceAssistant-400K (Mini-Omni 系列的数据集),包含 400K+ GPT-4o 生成的 QA pair 及对应合成语音,约 2200 小时英语单说话人数据 [§5]。Arabic: 450K 文本条目 + XTTS 合成语音,约 1500 小时 [§5]。
- **模型配置**: 4 layers, n_embd=768, n_head=8, micro-batch=4, gradient_accumulation=8, block_size=8192 tokens [§5]
- **优化器**: AdamW, lr=3e-4, weight_decay=0.1, 50K warmup + 1M steps cosine decay to 3e-6, grad clip 1.0 [§5]
- **硬件**: 4x A100, ~3 天, bfloat16 + flash-attention [§5]
- **损失函数**: 标准 cross-entropy on speech token sequence (causal mask) [§3.5]

## 实验

| 指标 | LLMVoX (本文) | Cascaded XTTS | Freeze-Omni | Moshi | GLM-4-Voice | LLaMA-Omni | MiniCPM-o 2.6 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT Score (General QA) ↑ | 6.14 | 6.70 | 3.48 | 2.71 | 5.24 | 3.44 | 5.46 | [Table 1] |
| GPT Score (Knowledge) ↑ | 7.62 | 7.70 | 4.98 | 3.91 | 5.67 | 3.84 | 6.21 | [Table 1] |
| GPT Score (Avg) ↑ | 6.88 | 7.20 | 4.23 | 3.31 | 5.30 | 3.64 | 5.84 | [Table 1] |
| UTMOS ↑ | 4.05 | 4.23 | 4.38 | 3.92 | 3.97 | 3.32 | 3.87 | [Table 1] |
| WER ↓ | 3.70% | 1.70% | 14.05% | 7.97% | 6.40% | 9.18% | 10.60% | [Table 1] |
| Latency ↓ | 475ms | 4200ms | 340ms | 320ms | 2500ms | 220ms | 1200ms | [Table 1] |

**关键发现**:

1. **LLM 能力保留最佳** [论文原文]: LLMVoX 与 XTTS 共享同一 base LLM (LLaMA 3.1 8B),GPT Score 非常接近 (6.88 vs 7.20),而 LLaMA-Omni 使用同一 LLM 却显著退化 (3.64),证明微调 LLM 确实损害其能力 [§6.4]。

2. **WER 最低**: 3.70% 显著优于所有端到端模型 (Freeze-Omni 14.05%, MiniCPM-o 10.60%),仅次于非流式 XTTS (1.70%),说明 text-speech alignment 质量很高 [§6.4]。

3. **延迟 vs 质量的 Pareto 最优** [论文原文]: 相比 XTTS 延迟降低 86% (4200→475ms),相比 Freeze-Omni WER 降低 74% (14.05→3.70%),实现了在延迟-质量空间中的更优 trade-off [§6.4, Fig 1]。

4. **Chunk size 影响** [Fig 6]: chunk size 从 20→640,UTMOS 从 3.75→4.41,WER 从 4.1%→3.6%,latency 保持 <1s (up to 160)。更大 chunk 提升韵律连贯性 [§6.4]。

5. **LLM 规模扩展** [Table 2]: 端到端延迟从 0.33s (Qwen2.5 0.5B) 到 1.91s (LLaMA 3.3 70B),LLMVoX 本身延迟贡献稳定 (~255ms) [Fig 5]。

6. **Arabic**: WER 23.4% / CER 8.2% (streaming),次于非流式 XTTS (WER 6.2% / CER 1.7%),但优于 ArTST、FastPitch、Tacotron 2、Seamless [Table 3]。

7. **VSQA 多模态** [Table 4]: 接入 Qwen2.5-VL-7B 后,GPT Score 6.41 > MiniCPM-o 2.6 的 6.32,延迟 1.05s vs 1.45s,展示了 plug-and-play 能力。

8. **人类评估** [Fig 4]: vs Freeze-Omni,Answer Relevance 赢 52% / 输 20%,Speech Quality 赢 62% / 输 20% [§6.4]。

## 局限性

1. **无 voice cloning**: 仅支持单说话人合成,无法生成特定说话人的声音。论文也承认这是主要局限 [§8]。
2. **ASR 未流式化**: Whisper 在管线中不是流式运行的,ASR 环节的延迟优化空间未被利用 [§8]。
3. **训练数据规模小且为合成数据**: 2200 小时来自 GPT-4o 合成的 QA pair + XTTS 合成语音,真实场景的泛化能力存疑。
4. **韵律表现力有限**: 单说话人 + 轻量模型 + 单层 RVQ 的组合可能限制了韵律多样性和情感表达,论文未评估 prosody 相关指标。
5. **WavTokenizer 单层 RVQ 的音质上限**: 虽然 UTMOS 4.05 可用,但距离 Freeze-Omni 的 4.38 和 XTTS 的 4.23 仍有差距。
6. **GPT Score 评估的局限**: 主要衡量 LLM 回答质量,通过 Whisper 转写间接评估;未直接衡量 TTS 的韵律、自然度、语速控制等维度。

## 点评

LLMVoX 在"LLM-agnostic streaming TTS"这个特定赛道做出了清晰的工程贡献。其核心 insight——完全依赖 LLM 文本输出 (而非 hidden states) 做 TTS——使得 30M 的轻量模型可以 plug-and-play 到任意 LLM/VLM,无需任何适配。这在实际部署中价值很大: 企业可以自由升级 LLM 而不影响 TTS 模块。

multi-queue streaming 机制是巧妙的工程设计,通过空间换时间 (两个 TTS 实例) 解决了 cascaded 系统的延迟瓶颈。chunk size 倍增策略也展现了对 streaming audio playback 的实践理解。

但从学术贡献角度看,LLMVoX 的模型架构本身缺乏新意——4 层 decoder-only transformer + cross-entropy loss 是最基础的设计。其优势主要来自 (1) WavTokenizer 的单码本简化和 (2) streaming 工程优化,而非建模创新。此外,单说话人、无 voice cloning 的限制使其实际应用场景偏窄。论文的比较对象 (LLaMA-Omni, Moshi, GLM-4-Voice) 多为 2024 年的系统,2025 年的 SOTA (如 Step-Audio 2.5, CosyVoice 3 等) 可能在延迟和质量上都已显著进步。

## 可复用的 idea

1. **ByT5 G2P embedding 复用**: 直接使用预训练 ByT5 G2P 模型的 embedding layer 作为 phoneme-aware text frontend,无需显式 G2P 转换,适用于任何需要低延迟 text 处理的场景。
2. **Multi-queue 双实例 streaming**: 将长文本按句子拆分到两个队列,两个 TTS 模型副本并行处理,交替输出音频 chunk。适用于任何 autoregressive TTS 的无限长度 streaming 需求。
3. **Chunk size 倍增策略**: 初始小 chunk (低首包延迟) → 逐步翻倍 (提升音质),利用播放 buffer 作为处理时间。简单有效的延迟-质量 trade-off。
4. **纯文本接口的 LLM-agnostic 设计**: 只依赖 LLM 的文本输出,不依赖 hidden states/logits,使 TTS 模块真正可插拔。可作为所有 cascaded speech 系统的设计原则。

---

检索命中: [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[Residual Vector Quantization]]✓, [[Streaming Spoken Dialogue]](pending-review), [[Speech-LLM Integration Taxonomy]](pending-review), [[Codec Language Model]](pending-review) | 过滤: 无 | 未命中但可能相关: 无
