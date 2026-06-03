---
type: paper
tier: deep
title: "Zero-Shot Text-to-Speech from Continuous Text Streams"
arxiv_id: "2410.00767"
source: "Sources/LiveSpeech2.pdf"
authors: [Trung Dang, David Aponte, Dung Tran, Tianyi Chen, Kazuhito Koishida]
year: 2024
venue: "Under Review (arXiv)"
tags: [TTS, zero-shot, streaming, Mamba, SSM, cross-attention, semantic-guidance, RVQ, autoregressive]
concepts: ["[[Residual Vector Quantization]]", "[[Semantic vs Acoustic Tokens]]", "[[Speech Tokenizer]]", "[[Speaker Embedding]]", "[[Codec Language Model]]", "[[Streaming Spoken Dialogue]]", "[[Speech-Text Alignment]]"]
models: ["[[模型库/EnCodec|EnCodec]]", "[[模型库/Whisper|Whisper]]", "[[模型库/wav2vec 2.0|wav2vec 2.0]]"]
tasks: ["[[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]"]
datasets: ["LibriLight", "LibriTTS"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Residual Vector Quantization]], [[Semantic vs Acoustic Tokens]], [[Speech Tokenizer]], [[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]], [[Speaker Embedding]], [[Speech Language Model]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: LiveSpeech 2 属于 Codec Language Model 路线 (自回归 LM 直接生成 audio codec tokens),但有两个显著偏离: (1) 用 Mamba (SSM) 取代 Transformer decoder 以实现 O(1) 解码复杂度,是该路线中最早在大规模实验中验证 SSM 可行性的工作之一; (2) 引入 grapheme tokens 作为 semantic 引导信号,与 AudioLM 的 semantic→acoustic 两阶段不同,LiveSpeech 2 将 grapheme 和 acoustic tokens 在同一步联合解码,实现推理时的文本引导对齐。
>
> **已有认知**: Zero-shot TTS 目前以 LLM+离散token、Diffusion、Coarse-to-fine hybrid 三条路线为主,SOTA 系统 (CosyVoice 3, Seed-TTS, IndexTTS2) 在 SEED-TTS-Eval 上的 WER 已低于 1.5%。Speaker embedding 注入方式已从早期 concatenation 演进到 cross-attention/prompt,ECAPA-TDNN 是最常用评估模型。Streaming TTS 在 spoken dialogue 领域关注较多 (Moshi, Mini-Omni 等),但 chunk-level text streaming 输入的 zero-shot TTS 场景目前较少专门研究。
>
> **创新判断**: 本文的核心新颖性在于将 Mamba 引入 zero-shot TTS 并解决 streaming text input 问题。RoPE cross-attention sliding window 和 inference-time semantic guidance 是针对 streaming 场景的专门设计,不同于 Moshi 等系统侧重 full-duplex 对话。
>
> 检索命中: [[Residual Vector Quantization]]✓, [[Semantic vs Acoustic Tokens]]✓, [[Speech Tokenizer]]✓, [[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]✓, [[Speaker Embedding]]✓, [[Speech Language Model]]✓ | 过滤: [[Codec Language Model]](pending-review), [[Streaming Spoken Dialogue]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 Mamba+RoPE cross-attention 的 streaming zero-shot TTS,支持无限长文本流输入,通过推理时 semantic guidance 保证内容对齐
> - **路线**: 文本流(BPE tokens) + 参考语音(Transformer encoder→64 embeddings) → Mamba decoder(12 层, cross-attention 融合) → EnCodec 16 codebook codes → EnCodec decoder → 语音
> - **指标**: CER 2.7/3.0 (3-10s/10s+), WER 3.1/4.1, SS 61.7/67.6, SMOS 3.4/3.4, NMOS 3.2/3.3 (LibriTTS test-clean) [Table 2]
> - **可借鉴**: (1) RoPE 在 cross-attention 中用于 text sliding window — 使固定上下文训练泛化到无限长推理; (2) grapheme token 联合解码 + transcript-guided sampling — 不需额外 ASR 即可推理时纠正对齐; (3) N-time sampling 利用 grapheme 输出做 CER 选择,无需外部 ASR
> - **局限**: 未开源; 仅英语验证; CER/WER 落后于 XTTS v2 等非流式模型; Mamba 内部状态不可控导致 hard guidance 效果差; 极短 chunk (单词级) 性能退化明显

## 核心问题

现有 zero-shot TTS 系统设计为处理完整句子,不支持文本流输入。当文本以短 chunk 持续到达时 (如 LLM 输出或实时翻译),存在三个基本挑战 [§1]:
1. **无限长生成**: 固定文本条件无法动态更新,无法生成超出训练长度的语音
2. **文本-语音同步**: 生成的语音必须跟上文本流的前沿,需要适应性时长控制
3. **块间平滑过渡**: 短 chunk 之间的语音必须无缝衔接,保持风格一致

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

LiveSpeech 2 沿用 LiveSpeech 的三组件架构 [§4]:

1. **Speech Encoder**: 6 层 8 头 Transformer encoder (77M 参数),将任意长参考语音压缩为固定长度 64 个 embeddings [§5.2]。[论文原文] 目的是大幅压缩参考语音表示以加速解码 [§4]。

2. **Text Tokenizer + Embedder**: 使用 Whisper BPE tokenizer (51,866 词表),将文本 chunk 编码为 word token embeddings [§4]。[论文原文] 选择 Whisper tokenizer 是因为覆盖面广且与上游 Whisper 模型兼容 [§4]。

3. **Mamba Decoder**: 12 层 Mamba blocks (671M 参数, hidden dim 1536),替代 Transformer decoder [§5.2]。通过 cross-attention (16 头) 整合 speech 和 text embeddings。

**Codebook 分组策略** [§5.2]: 前 6 层共享建模所有 codebooks,后 6 层将 17 个 codebook (1 grapheme + 16 acoustic) 分为 4 组 (4, 4, 4, 5),各组独立建模。低级 codebook 预测使用权重衰减 λ_cb = 0.1 [§5.2]。

### 关键设计选择

#### 设计选择 1: 为什么用 Mamba 而不是 Transformer

[论文原文] Transformer decoder 在每步解码时需要对全部历史 context 做 attention,复杂度 O(n);Mamba 将 context 压缩为固定大小的 state vector,每步解码 O(1) [§3.2]。[论文原文] 作者认为音频 token 通常长、冗余、偏向近期,一个压缩的 state 足以保证帧间平滑和语义连贯 [§3.2]。

[agent 解读] 这是一个有意思的 trade-off: Mamba 的 O(1) 解码速度对 streaming 场景至关重要 (每帧恒定时间输出),但代价是无法回溯任意历史位置的精确信息。后续 Areas for Improvement [§A.6] 中作者也指出,Mamba 的内部状态在 hard guidance 时 "只能 force input 但无法 force internal state",提示 SSM 的压缩状态在某些场景下不如 Transformer 灵活。

#### 设计选择 2: RoPE Cross-Attention 实现 text sliding window

**问题**: 训练时文本是固定的完整句子 (最多 75 word tokens),推理时文本以 chunk 持续到达,需要动态更新 context [§4.1]。

**方案**: 给每个 word token 分配基于到达时间的位置索引 — chunk S_i 中的 tokens 被赋予位置 τ_i, τ_i+1, ...,其中 τ_i 是 chunk i 到达的时间步 [§4.1, Fig 2]。Cross-attention 使用 RoPE 编码这些位置:

$$a_t = \text{Softmax}\left(\frac{q_t K^{(enr)T}}{\sqrt{d_k}} V^{(enr)}; \frac{\text{RoPE}(q_t, t) \text{RoPE}(K_t^{(txt)}, T_t)^T}{\sqrt{d_k}} V^{(txt)}\right)$$ [Eq. 1]

[论文原文] RoPE 的关键优势: 它编码相对位置,所以训练时看到的固定 context 可以泛化到推理时动态滑动窗口 [§4.1]。enrollment speech features 不使用位置编码 (position-agnostic) [§4.1]。

**推理时的窗口参数**: n_p (past chunks) 和 n_f (future chunks) 控制每步解码可见的 context 范围。当 n_f = 0 时系统立即开始生成;当 n_f > 0 时延迟到 chunk S_{i+n_f} 到达 [§4.1]。

[agent 解读] 这个设计巧妙地利用 RoPE 的相对位置特性解决训练-推理不一致问题,但存在一个假设: 训练时的 full context 可以通过位置关系泛化到推理时的 partial context。§A.5 中 streaming-aware training (动态 attention masking) 的实验表明,这个泛化确实存在 gap — 针对性微调后极端场景 (单词 chunk) 的 WER 从 40.7% 降至 4.4%。

#### 设计选择 3: Inference-Time Semantic Guidance

**训练**: 将 time-aligned graphemes (来自 CTC-based ASR 如 wav2vec 2.0) 作为额外 codebook,放在第一个 acoustic codebook 之前。CTC 输出中的 blank tokens 被替换为右侧最近非 blank token,并上采样到 75 fps 与解码步对齐 [§4.2]。

**推理**: 利用已生成的 grapheme 序列 G_{t-1} 和 transcript 来引导下一个 grapheme g_t 的采样 [Algorithm 1]:
1. CTC decode 去除重复和非字符 tokens
2. 找 transcript 的最佳匹配前缀 (最低 CER)
3. 确定 guiding tokens: 前缀末尾 token (staying) + 下一 token (moving forward)
4. 将 guiding tokens 概率放大 (1 + λ),与 top-k 候选合并重新归一化
5. 从合并集合中 top-k 采样

[论文原文] λ = 0 无引导; λ → ∞ 为 hard guidance (仅从 guiding tokens 选); 0 < λ < ∞ 为 soft guidance [§4.2]。作者选择 λ = 1 作为默认值 [§5.3]。

[论文原文] 与 AudioLM 的 semantic → acoustic 两阶段不同,本文进一步利用 transcript 在推理时灵活引导解码过程 [§4.2]。

[agent 解读] Semantic guidance 的核心价值在于它以极低额外开销 (grapheme 只是一个 codebook) 实现了推理时的内容纠错。传统方法需要生成后用 ASR 校验,本文将校验融入解码循环。但 §A.6 指出 Mamba 下 hard guidance 效果不佳 (WER 46.1% [Table 10]),原因是 SSM 的 state 无法被 "强制",只能 force input — 这是 SSM vs Transformer 在可控生成上的根本差异。

### 训练策略

- **Audio Codec**: EnCodec @ 12kbps, 16 codes/frame, 75 fps。因原始 EnCodec 也训练于通用音频/音乐,作者在 LibriLight 上重新训练了一个 speech-specific decoder (110M 参数) [§5.3]
- **数据**: LibriLight 60K 小时无标注语音;Whisper v3 + wav2vec 2.0 提取 transcript 和对齐;CER > 0.1 的样本被过滤;每个样本 ≤ 10s,配 ≤ 5s 同说话人 enrollment [§5.1]
- **优化**: 2M steps, batch size 32, 4 A100 GPUs, lr 5e-4, 200K warmup [§5.3]
- **Streaming 模拟**: 训练和评估时将 transcript 随机分为 2-4 word token 的 chunks,用 Whisper v3 alignment 推断到达时间 [§5.1]

## 实验

| 指标 | LiveSpeech 2 | XTTS v2 | YourTTS | LiveSpeech | Ground-truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CER (3-10s) | 2.7 | 1.9 | 3.8 | 3.3 | 1.6 | LibriTTS test-clean | [Table 2] |
| CER (>10s) | 3.0 | 2.2 | 3.3 | — | 1.4 | LibriTTS test-clean | [Table 2] |
| WER (3-10s) | 3.1 | 1.3 | 4.3 | 6.0 | 0.6 | LibriTTS test-clean | [Table 2] |
| WER (>10s) | 4.1 | 1.9 | 3.6 | — | 0.7 | LibriTTS test-clean | [Table 2] |
| SS (3-10s) | 61.7 | 60.3 | 48.6 | 59.3 | 75.3 | LibriTTS test-clean | [Table 2] |
| SS (>10s) | 67.6 | 64.8 | 55.2 | — | 83.9 | LibriTTS test-clean | [Table 2] |
| SMOS (3-10s) | 3.4 | 3.1 | 2.6 | 2.8 | 3.8 | LibriTTS test-clean | [Table 2] |
| NMOS (3-10s) | 3.2 | 3.0 | 2.5 | 2.5 | 3.7 | LibriTTS test-clean | [Table 2] |
| DNSMOS (3-10s) | 3.9 | 4.0 | 3.8 | 3.8 | 3.9 | LibriTTS test-clean | [Table 2] |
| DNSMOS (>10s) | 4.0 | 4.0 | 3.9 | — | 4.0 | LibriTTS test-clean | [Table 2] |

### 关键消融

**Semantic guidance 效果** [Table 3]:
- 无 semantic tokens: WER 7.3/13.4, SS 60.6/67.0
- 有 semantic tokens 但无 guidance: WER 6.7/5.6, SS 61.3/67.4
- 有 guidance (λ=1): WER 3.1/4.1, SS 61.7/67.6
- Semantic guidance 在长语音上对 WER 的改善尤为显著 (13.4 → 4.1, -69%) [Table 3]

**Chunk 长度影响** [Table 5]:
- 单词 chunk (l_min=l_max=1): WER 40.7/73.7,严重退化
- 2-4 words: WER 4.0/3.9,接近最优
- 需要 streaming-aware training 才能支持极短 chunk [§A.5]

**Look-ahead 影响** [Table 6]:
- n_p=1, n_f=1: WER 23.5/15.6,性能差
- n_p=2, n_f=2: WER 3.3/4.6,接近最优
- n_p=10, n_f=4: WER 3.0/3.3,最优,但延迟增加

**N-time sampling** [Table 4]:
- 5-time CER-based 选择: WER 2.0/3.1 (vs 1-time 3.1/4.1)
- Probability-based 选择: SS 61.9/68.6 (+1.0),可同时优化 CER 和 SS

## 局限性

1. **CER/WER 落后于非流式 SOTA**: 与 XTTS v2 (1.9/1.3) 相比,LiveSpeech 2 (2.7/3.1) 内容准确性有 gap,尤其考虑到 XTTS v2 使用了大量内部数据训练 [Table 2]
2. **仅英语验证**: 仅在 LibriLight/LibriTTS (英语) 上实验,多语言能力未知 [§4.2, §6]
3. **极短 chunk 退化**: 单词级 chunk WER 高达 40.7%,需要 streaming-aware training 缓解 [Table 5]
4. **SSM 限制 hard guidance**: Mamba 的内部状态不可直接控制,hard guidance (λ→∞) 导致 WER 46.1%,远差于 soft guidance [Table 10, §A.6]
5. **Grapheme 预测受 acoustic code 错误影响**: 共享层同时预测 grapheme 和 acoustic codes,acoustic 错误可能反向影响 grapheme 预测 [§A.6]
6. **未开源**: 代码和模型未公开,仅提供 audio samples

## 点评

LiveSpeech 2 抓住了一个被忽略但重要的场景: 上游系统 (LLM/翻译) 持续产出短文本 chunk 时的低延迟语音合成。多数 streaming TTS 研究关注 full-duplex 对话 (Moshi) 或句子级 streaming (BASE TTS),chunk-level text-stream-aware 的 zero-shot TTS 是一个有实际需求但少被研究的 niche。

三个设计贡献各有价值但深度不一:
- **Mamba decoder** 是最大的架构贡献,也是首次在大规模 zero-shot TTS 中验证 SSM 的竞争力。但论文对 Mamba vs Transformer 的对比不够充分 — 没有直接的同参数量 Transformer 对照实验,无法确定性能差异来自架构还是其他因素
- **RoPE cross-attention** 设计简洁高效,利用相对位置编码解决 train-test 不一致,是最 "干净" 的贡献
- **Semantic guidance** 作为推理时技巧很实用,但在 Mamba 上效果受限 (hard guidance 失效),也暴露了 SSM 在可控生成上的本质局限

实验设计的一个不足是 baseline 选择偏旧 — YourTTS/MetaVoice/SpeechX 不是当前 SOTA,缺少与 VALL-E 2, CosyVoice, Seed-TTS 等 2024 年系统的对比。不过考虑到论文提交时间 (2024.10) 和流式场景的特殊性,这可以理解。

## 可复用的 idea

1. **RoPE cross-attention sliding window**: 训练时用 full context,推理时通过 RoPE 相对位置自然泛化到 partial context — 适用于任何需要动态文本条件的序列生成任务
2. **Grapheme codebook + transcript-guided sampling**: 将轻量级 semantic token 嵌入解码循环,推理时用 transcript 引导采样 — 无需外部 ASR 即可实时纠正对齐,且 grapheme 输出可直接用于 N-time sampling 的 CER 选择
3. **Probability-based N-time selection**: 用 grapheme 序列的累积概率 (而非 CER) 选择最优输出,在保持 CER 的同时提升 SS (+1.0) [Table 4] — 可作为任何带 semantic token 的 TTS 系统的免费午餐
4. **Streaming-aware training via dynamic attention masking**: 训练时随机 mask 掉部分 text context 模拟 streaming 条件 [Algorithm 2-3] — 缩小 train-test gap 而无需改变推理架构

> [!review] 审阅 (auto, 2026-06-03)
> **结论**: pass-with-fixes
> - [medium] 实验表缺 DNSMOS (O-MOS) → 已补充
> - [low] 速查"首个"潜在 overclaim → 已修正
> 详见 `_review/LiveSpeech 2-review.yml`
