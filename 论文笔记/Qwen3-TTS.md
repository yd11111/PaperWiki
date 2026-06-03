---
type: paper
tier: deep
title: "Qwen3-TTS Technical Report"
arxiv_id: "2601.15621"
source: "Sources/Qwen3-TTS.pdf"
authors: [Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, Xinyu Zhang, Pei Zhang, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin]
year: 2026
venue: "arXiv"
tags: [TTS, LLM-based, streaming, multilingual, voice-cloning, instruction-following, speech-tokenizer, codec, zero-shot, cross-lingual, controllable]
concepts: ["[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Single-codebook vs Multi-codebook]]", "[[Residual Vector Quantization]]", "[[LLM-based TTS]]", "[[Conditional Flow Matching]]", "[[Speaker Embedding]]", "[[Instruction-Guided Speech Synthesis]]", "[[Token Rate and Bitrate Trade-offs]]"]
models: ["[[CosyVoice 3]]", "[[BigVGAN]]", "[[WavLM]]"]
tasks: ["[[Zero-shot Speech Synthesis]]", "[[Cross-lingual Voice Cloning]]", "[[Instructed Speech Generation]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[CV3-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Residual Vector Quantization]], [[LLM-based TTS]], [[Conditional Flow Matching]], [[Zero-shot Speech Synthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Qwen3-TTS 属于 LLM-based TTS 家族中 hybrid 路线(LLM + Flow/Codec)的最新成员。与 CosyVoice 系列(semantic token + CFM renderer)、Seed-TTS(AR + token diffusion)平行竞争。其独特之处是同时提供两种 tokenizer:25Hz 单码本(类似 CosyVoice 路线)和 12Hz 多码本(类似 Mimi/FireRedTTS 2 路线),覆盖了 [[Single-codebook vs Multi-codebook]] 两侧设计空间。
>
> **已有认知**:
> - Speech Tokenizer 领域正在从纯 semantic/acoustic 二分走向混合方案(SpeechTokenizer/Mimi),且出现了 continuous VAE tokenizer 新路线 [待确认]
> - 监督式 semantic token(CosyVoice S3 tokenizer)在 TTS 任务上的内容一致性(WER)优于自监督方案
> - RVQ 多码本设计提供层级信息结构但增加 LM 建模复杂度;当前趋势是减少码本数
> - CFM 在 TTS 中主要用作 second-stage renderer(token → mel),CosyVoice 3 使用 DiT-based CFM(300M 参数)
> - Zero-shot TTS 当前 SOTA: CosyVoice 3 在 SEED-TTS-Eval 上 CER 0.71%(zh), WER 1.45%(en)
> - Cross-lingual Voice Cloning 的 SOTA 由 CosyVoice 3 保持
>
> **创新判断**: Qwen3-TTS 的核心创新在于(1)双 tokenizer 设计覆盖不同延迟-质量 trade-off,(2)12Hz 极低帧率 + 16 层 RVQ 实现超低延迟流式,(3)dual-track LM + MTP 统一处理多码本生成,(4)5M 小时训练数据规模远超同类工作。
>
> 检索命中: [[Speech Tokenizer]]✓, [[Semantic vs Acoustic Tokens]]✓, [[Residual Vector Quantization]]✓, [[LLM-based TTS]]✓, [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Single-codebook vs Multi-codebook]](pending-review), [[Codec Language Model]](pending-review), [[Token Rate and Bitrate Trade-offs]](pending-review), [[Voice Cloning Taxonomy]](pending-review) | 未命中但可能相关: Differentiable Reward Optimization(DPO/GSPO 相关)

## 速查

> [!summary] 速查
> - **一句话**: Qwen3-TTS 通过双 tokenizer(25Hz 单码本 + 12Hz 16 层 RVQ)+ dual-track LM 架构,在 5M 小时数据上训练,实现 10 语言流式 TTS,在 zero-shot cloning/cross-lingual/controllability 上全面超越 CosyVoice 3 和 MiniMax-Speech
> - **路线**: Text(Qwen tokenizer）+ Reference Speech（speaker encoder / in-context）→ Qwen3 LM（dual-track: text + acoustic token 沿 channel 拼接）→ Speech Tokens → Code2Wav（25Hz: chunk-wise DiT + BigVGAN; 12Hz: causal ConvNet）→ Waveform
> - **指标**: WER 1.24%(test-en, SEED-TTS-Eval, 12Hz-1.7B, SOTA) [Table 5]; WER 0.77%(test-zh) [Table 5]; Speaker SIM 0.799(zh, 最高) [Table 6]; zh-to-ko WER 4.82(比 CosyVoice3 降 66%) [Table 7]; Voice Design APS 85.2(zh, 开源 SOTA) [Table 8]; 首包延迟 97ms(12Hz-0.6B) [Table 2]
> - **可借鉴**: (1) 双 tokenizer 策略: 同一 LM backbone 适配不同延迟需求,25Hz 适合高质量, 12Hz 适合超低延迟; (2) MTP 模块处理 RVQ 多码本序列,比 delay pattern 和 NAR 更适合流式; (3) 概率性激活 thinking pattern 提升 instruction following; (4) DPO + GSPO 后训练改善自然度和鲁棒性
> - **局限**: (1) 25Hz tokenizer 在长文本稳定性反而优于 12Hz,说明多码本路线在长上下文建模上仍有瓶颈 [Table 10]; (2) 论文未公开 DPO/GSPO 的具体 reward 设计和训练细节; (3) 仅报告 WER/SIM,未报告 MOS/PESQ 等感知指标; (4) 未与 F5-TTS/Seed-TTS 等非自回归方法在延迟-质量上做 fair comparison

## 核心问题

Qwen3-TTS 试图在一个统一框架内同时解决四个挑战:
1. **流式低延迟**: 首包延迟 <100ms,适合实时对话
2. **多语言覆盖**: 10 种语言的统一建模
3. **精细可控性**: 自然语言指令创建/编辑声音
4. **长文本稳定性**: 超过 10 分钟的连续合成不退化

核心设计选择是**双 tokenizer + dual-track LM**:通过两种不同的 speech tokenizer 覆盖不同的延迟-质量需求,同时共享 LM backbone。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Qwen3-TTS 由三个核心模块组成 [§3, Fig 3]:

1. **Speech Tokenizer**: 将语音编码为离散 token(两种可选)
2. **Qwen3 LM Backbone**: 以 dual-track 方式同时处理文本和语音 token
3. **Code2Wav Module**: 将 token 解码为波形(两种对应方案)

**Dual-track representation** [§3.1]: 文本 token 和声学 token 沿 **channel 轴**拼接(而非 sequence 轴)。这意味着收到一个文本 token 时,模型**立即预测对应的声学 token**,实现实时流式。[论文原文]

[agent 解读] 这与 CosyVoice/Seed-TTS 的 sequence-level 先生成所有 semantic token 再 second-stage 渲染不同。Qwen3-TTS 的 dual-track 设计更接近 Moshi 的 interleaved 建模,但简化为 text-speech 对,不需要处理 speech-speech 的双流。

### 两种 Tokenizer 的设计哲学

#### Qwen-TTS-Tokenizer-25Hz (单码本)

**架构** [§2.1]:
- 基于 Qwen2-Audio encoder,两阶段训练
- **Stage 1**: 在 Qwen2-Audio 上做 ASR continual pretraining,插入 resampling layer + VQ layer (codebook 32768),25 Hz
- **Stage 2**: 加入 conv-based mel decoder,通过 mel reconstruction loss 注入声学信息到 token

**流式解码** [§2.1]: 使用 DiT + Flow Matching 做 mel 重建,采用 sliding-window block attention: 每个 code block 只看 3-block lookback + 1-block lookahead。然后 BigVGAN 从 mel 合成波形,同样做 chunked processing。

[论文原文] 作者发现纯 semantic tokenizer 缺乏表现力,而纯 acoustic tokenizer 注入过多底层细节导致 LLM 建模困难和 long-horizon error accumulation。25Hz tokenizer 试图在语义和声学之间取得平衡。

[agent 解读] 这个设计思路与 CosyVoice 的 S3 tokenizer 类似(监督式 semantic token),但 Qwen3-TTS 的 VQ layer 直接嵌入 Qwen2-Audio(多模态大模型),而非 SenseVoice/MinMo 这样的 ASR encoder,理论上能从更丰富的 audio 理解中受益。但 Stage 2 的 mel reconstruction 会轻微降低 ASR 性能 [Table 3],这是刻意的 trade-off。

#### Qwen-TTS-Tokenizer-12Hz (多码本)

**架构** [§2.2]:
- 12.5 Hz,16 层 RVQ (codebook 2048)
- 受 Mimi 架构启发:第一层 VQ 编码语义(WavLM teacher distillation),后续 15 层 RVQ 编码声学残差
- GAN-based 训练 + multi-scale mel loss
- **全因果设计**: encoder/decoder 均为因果(无 lookahead),支持流式

**解码** [§2.2]: 因为有 16 层 codebook 提供充足信息,仅需**轻量因果 ConvNet** 即可重建波形,无需 speaker vector 提取或复杂 diffusion model。

[论文原文] 作者表示 12Hz tokenizer 的设计动机是 25Hz 单码本在超低延迟场景下不适用:单码本 + DiT 需要等待 lookahead tokens,增加延迟。

[agent 解读] 这实际上印证了 KB 中 [[Single-codebook vs Multi-codebook]] 的核心 trade-off: 单码本更适合 LM 建模但需要更重的 decoder,多码本信息自足但需要多流建模。Qwen3-TTS 的解法是两者都做,让用户按需选择。12Hz tokenizer 在 reconstruction benchmark 上全面 SOTA(PESQ_WB 3.21, SIM 0.95) [Table 4],说明 16 层 RVQ + 低帧率的组合确实在质量和效率上达到了很好的平衡。

### LM 架构差异

**25Hz variant** [§3.1]: backbone 将 text features 与前序 speech tokens 整合,通过 linear head 预测当前 speech token。然后送入 chunk-wise DiT 重建波形。

**12Hz variant** [§3.1]: 采用**层级预测**:
1. Backbone 接收**聚合后的 codebook features**,预测第 0 层 codebook (语义层)
2. **MTP (Multi-Token Prediction) 模块**生成剩余所有 residual codebooks (声学层)

[论文原文] 这种策略"captures intricate acoustic details, significantly enhancing vocal consistency and expressivity, while minimizing latency through single-frame instant generation" [§3.1]

[agent 解读] MTP 的优势相比 VALL-E 的 AR+NAR 两阶段: (1) 不需要额外的 NAR 模型, (2) 所有 codebook 可在单帧内一次性生成, (3) 适合流式场景。相比 MusicGen 的 delay pattern, MTP 更直接且延迟更低。这可能是本文在工程上最值得借鉴的设计之一。

### 关键设计选择

**1. 为什么要两种 tokenizer 而非统一一种?**

[论文原文] 25Hz 单码本在长文本稳定性上优于 12Hz(Table 10: 25Hz WER 1.517/1.225 vs 12Hz WER 2.356/2.812),而 12Hz 在延迟上碾压 25Hz(首包 101ms vs 150ms [Table 2])。两者各有优势场景。

[agent 解读] 这反映了一个深层 trade-off: semantic-rich 单码本 token 更有利于 LM 建模长距离依赖(因为每步 token 信息密度高,序列短),而多码本 token 虽然帧率低(12.5 vs 25 Hz)但每帧需要生成 16 个 token,总信息量反而更大,长上下文建模更难。

**2. 为什么选 Qwen2-Audio 而非 HuBERT/SenseVoice 做 25Hz tokenizer 的 backbone?**

[agent 解读] Qwen2-Audio 是多模态大模型,其 encoder 在大规模多语言 audio 数据上预训练,比 SenseVoice 等 ASR-specific encoder 有更丰富的表征。且 Qwen3-TTS 作为 Qwen 系列产品,使用自家 Qwen2-Audio 也有技术栈一致性的考虑。

**3. Speaker encoder 是独立模块还是集成在 LM 中?**

[论文原文] "We jointly train a learnable speaker encoder with the backbone" [§3.1] — speaker encoder 与 LM 联合训练,而非使用预训练的外部 speaker embedding(如 ECAPA-TDNN)。

[agent 解读] 联合训练的优势是 speaker representation 与 LM 的 feature space 对齐更好,但缺点是不能即插即用现有 speaker verification 模型。CosyVoice 3 也是类似路线(speaker condition integrated in CFM)。

### 训练策略

**预训练三阶段** [§3.2]:
1. **S1 (General)**: 5M+ 小时多语言数据,建立 text → speech 单调映射
2. **S2 (High-Quality)**: 用专用 pipeline 分层数据质量,在高质量数据上 continual pretraining。[论文原文] 作者指出这一阶段"缓解了初始阶段由噪声数据引起的幻觉"
3. **S3 (Long-Context)**: max token length 从 8192 扩至 32768,上采样长语音。改善长文本处理能力

**后训练三阶段** [§3.2]:
1. **DPO** (Direct Preference Optimization): 基于人类反馈构建多语言 preference pairs
2. **GSPO** (rule-based rewards): 用规则化 reward 全面增强能力和稳定性
3. **Speaker Fine-tuning**: 在 base model 上做轻量 speaker adaptation,使模型采用特定声音

[agent 解读] 5M 小时的数据规模是目前公开报告中最大的(CosyVoice 3 用 530K 小时,Seed-TTS 未公开)。DPO + GSPO 的双重后训练借鉴了 LLM 领域的 RLHF 经验,但具体 reward 设计未公开。GSPO 是较新的优化方法,论文未给出详细描述。

### 可控性机制

**Voice cloning** [§3.3]: 两种方式:
1. Speaker embedding: 从参考语音提取 embedding,实时克隆
2. In-context learning: 提供 text-speech pair 作为 prompt,更好保留韵律

**Voice design** [§3.3]: 继承 Qwen3 文本理解能力,支持通过自然语言描述创建新声音。[论文原文] 作者引入"概率性激活的 thinking pattern"以改善复杂描述的 instruction following。

[agent 解读] "概率性激活 thinking pattern" 可能类似于 chain-of-thought,即在生成语音前先让模型"思考"指令含义。这个设计借鉴了 LLM 领域的 reasoning 技术,应用到 TTS 的 instruction following 中是新颖的。

## 实验

### Tokenizer 评估

| 指标 | Qwen-TTS-Tok-12Hz | Mimi | FireRedTTS 2 Tok | XCodec2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| PESQ_WB | **3.21** | 2.88 | 2.73 | 2.43 | LibriSpeech test-clean | [Table 4] |
| PESQ_NB | **3.68** | 3.42 | 3.28 | 3.04 | LibriSpeech test-clean | [Table 4] |
| STOI | **0.96** | 0.94 | 0.94 | 0.92 | LibriSpeech test-clean | [Table 4] |
| UTMOS | **4.16** | 3.87 | 3.88 | 4.13 | LibriSpeech test-clean | [Table 4] |
| SIM | **0.95** | 0.87 | 0.87 | 0.82 | LibriSpeech test-clean | [Table 4] |

### Zero-shot TTS (Seed-TTS-Eval)

| 指标 | Qwen3-TTS-12Hz-1.7B | CosyVoice 3 | Seed-TTS | MiniMax | F5-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER test-zh | 0.77 | **0.71** | 1.12 | 0.83 | 1.56 | SEED-TTS-Eval | [Table 5] |
| WER test-en | **1.24** | 1.45 | 2.25 | 1.65 | 1.83 | SEED-TTS-Eval | [Table 5] |

### Multilingual Speaker Similarity (最佳结果, 12Hz-1.7B)

| 指标 | Qwen3-TTS | MiniMax | ElevenLabs | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| SIM-zh | **0.811** | 0.780 | 0.677 | TTS multilingual test | [Table 6] |
| SIM-en | **0.829** | 0.756 | 0.613 | TTS multilingual test | [Table 6] |
| SIM-ja | **0.798** | 0.776 | 0.738 | TTS multilingual test | [Table 6] |

### Cross-lingual (12Hz-1.7B)

| 指标 | Qwen3-TTS | CosyVoice 3 | CosyVoice 2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER zh-to-en | **2.77** | 2.98 | 6.47 | CV3-Eval | [Table 7] |
| WER en-to-zh | 4.77 | **5.09** | 13.5 | CV3-Eval | [Table 7] |
| WER zh-to-ko | **4.82** | 14.4 | 24.8 | CV3-Eval | [Table 7] |

### Controllability (12Hz-1.7B-VD, Voice Design)

| 指标 | Qwen3-TTS | VoiceSculptor | Hume | Parler-tts-large | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| APS-zh | 85.2 | 75.7 | 47.5 | - | InstructTTSEval | [Table 8] |
| DSD-en | **82.4** | 75.3 | 57.0 | 45.9 | InstructTTSEval | [Table 8] |

### Long Speech (CustomVoice)

| 指标 | Qwen3-TTS-25Hz-1.7B | Qwen3-TTS-12Hz-1.7B | VibeVoice | Higgs-Audio-v2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER long-zh | **1.517** | 2.356 | 22.619 | 5.505 | Internal | [Table 10] |
| WER long-en | **1.225** | 2.812 | 1.780 | 6.917 | Internal | [Table 10] |

### 流式效率

| 指标 | 12Hz-1.7B (conc=1) | 25Hz-1.7B (conc=1) | 12Hz-0.6B (conc=1) | 出处 |
| --- | --- | --- | --- | --- |
| First-Packet Latency | 101 ms | 150 ms | **97 ms** | [Table 2] |
| LM TPP | 21 ms | 56 ms | **19 ms** | [Table 2] |
| RTF | 0.313 | 0.253 | **0.288** | [Table 2] |

## 局限性

1. **25Hz vs 12Hz 的不统一**: 长文本场景 25Hz 明显优于 12Hz(WER 1.5 vs 2.4) [Table 10],说明多码本设计在 long-horizon 建模上仍有不足。用户需要根据场景手动选择 tokenizer,缺乏自适应机制。[agent 解读]

2. **感知质量指标缺失**: 全文未报告 MOS (Mean Opinion Score) 或 DNSMOS 等感知质量指标,仅用 WER 衡量内容一致性 + 余弦相似度衡量说话人保持。考虑到作者声称"human-like quality",缺少人工评测是明显不足。[agent 解读]

3. **训练细节不透明**: DPO 的 preference pair 构建方式、GSPO 的具体 reward 设计、"概率性激活 thinking pattern" 的实现细节均未公开。这使得 post-training 的核心 contribution 难以复现。[agent 解读]

4. **数据规模优势难以分离**: 5M 小时数据是已知最大规模,但论文未做数据规模消融实验。性能提升多少来自架构设计,多少来自数据规模,无法判断。[agent 解读]

5. **Voice Design 的 benchmark 局限**: InstructTTSEval 是较新 benchmark,Gemini/GPT-4o-mini-tts 的结果可能不是这些模型的最优配置。对比的公平性存疑。[agent 解读]

## 点评

Qwen3-TTS 是一个**工程驱动的全面系统**,而非单一方法创新。它的核心策略是"两条腿走路":

**双 tokenizer 的思路**值得关注。以往每个 TTS 系统只设计一种 tokenizer 并围绕其做优化(CosyVoice 的 FSQ、Seed-TTS 的 discrete tokens、VibeVoice 的 continuous VAE)。Qwen3-TTS 首次在一个统一框架内同时支持单码本和多码本路线,让不同场景(高质量/低延迟)各取所需。这种"不做选择"的思路在实际部署中可能比"做最优选择"更实用。

**12Hz tokenizer 的极致压缩**是技术亮点。在仅 12.5 Hz 帧率下通过 16 层 RVQ 实现 SOTA 重建质量(SIM 0.95,远超 Mimi 的 0.87) [Table 4],且因为因果 ConvNet decoder 无需 lookahead,首包延迟低至 97ms。这对实时对话场景有直接应用价值。

**训练数据规模(5M 小时)**是"大力出奇迹"的体现。但论文的实验设计不够透明:缺少数据规模消融、缺少 MOS 人工评测、post-training 细节不公开。作为 Technical Report 可以理解,但降低了学术贡献的可验证性。

**与 CosyVoice 3 的对比**是有意义的直接竞争:两者都是阿里系统,但架构路线不同。在 zero-shot WER 上两者各有胜负(zh: CosyVoice3 0.71 vs Qwen3 0.77; en: Qwen3 1.24 vs CosyVoice3 1.45) [Table 5]。跨语言场景 Qwen3-TTS 优势明显(zh-to-ko WER 4.82 vs 14.4) [Table 7]。

## 可复用的 idea

1. **双 tokenizer 策略**: 不纠结单/多码本的选择,在同一 LM backbone 上同时适配两种 tokenizer,按部署场景选择。可推广到其他 audio generation 系统。

2. **MTP for multi-codebook**: 用 Multi-Token Prediction 模块一次性生成所有 RVQ 层的 tokens,替代 VALL-E 式的 AR+NAR 两阶段或 MusicGen 式的 delay pattern。对任何使用 RVQ 多码本的 LM 系统都可借鉴。

3. **Qwen2-Audio 作为 tokenizer backbone**: 利用多模态大模型的丰富表征做 speech tokenization,而非从头训练。可推广到将其他 foundation model (如 Whisper large-v3) 改造为 tokenizer。

4. **概率性激活 thinking pattern**: 在 TTS 的 instruction following 中引入类似 chain-of-thought 的机制。这个 idea 可以应用到任何需要 instruction control 的生成模型中。

5. **三阶段预训练 + 三阶段后训练**: S1(通用) → S2(高质量) → S3(长上下文)的渐进式预训练,加上 DPO → GSPO → Speaker FT 的后训练流水线。这种"六步训练法"可以作为大规模 TTS 训练的参考范式。

> [!review] 审阅标记
> 审阅结论待生成。
