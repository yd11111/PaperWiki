---
type: paper
tier: deep
title: "TTS-1 Technical Report"
arxiv_id: "2507.21138"
source: "Sources/2507.21138.pdf"
authors: [Oleg Atamanenko, Anna Chalova, Joseph Coombes, Nikki Cope, Phillip Dang, Zhifeng Deng, Jimmy Du, Michael Ermolenko, Feifan Fan, Yufei Feng, Cheryl Fichter, Pavel Filimonov, Louis Fischer, Kylan Gibbs, Valeria Gusarova, Pavel Karpik, Andreas Assad Kottner, Ian Lee, Oliver Louie, Jasmine Mai, Mikhail Mamontov, Suri Mao, Nurullah Morshed, Igor Poletaev, Florin Radu, Dmytro Semernia, Evgenii Shingarev, Vikram Sivaraja, Peter Skirko, Rinat Takhautdinov, Robert Villahermosa, Jean Wang]
year: 2025
venue: "arXiv"
tags: [TTS, LLM-based, autoregressive, zero-shot, GRPO, RL-alignment, audio-codec, single-codebook, streaming, multilingual, emotion-control, audio-markup, 48kHz]
concepts: ["[[LLM-based TTS]]", "[[Speech Language Model]]", "[[Single-codebook vs Multi-codebook]]", "[[Differentiable Reward Optimization]]", "[[Emotion Control in TTS]]", "[[Codec Training Objectives]]", "[[Voice Cloning Taxonomy]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]", "[[模型库/Whisper|Whisper]]", "[[模型库/WavLM|WavLM]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: TTS-1 属于 [[LLM-based TTS]] 范式中的纯自回归 + 单码本路线。它使用 LLaMA 作为 [[Speech Language Model]] backbone,将语音离散化为单一码本 (65536 tokens) 的 token 序列后自回归生成。这与 CosyVoice 系列 (semantic tokens + CFM acoustic decoder) 和 VALL-E 系列 (多码本 AR+NAR) 都不同 -- TTS-1 选择了 X-codec2 架构的 **单码本 + 超分辨率 decoder** 路线,是 [[Single-codebook vs Multi-codebook]] 趋势中单码本大词表方案的代表。
>
> **已有认知**: (1) LLM-based TTS 已从 VALL-E 开创的 codec LM 范式演化出多条子路线 [confirmed]; (2) GRPO 用于 TTS RL alignment 已有 Multi-Reward GRPO、CosyVoice 3 DiffRO 等先例,TTS-1 的 GRPO 方案属于 audio-level RL 路线 [待确认]; (3) 情感/非语言控制方面,NVSpeech 和 EmoVoice 等已探索了文本标签和自由文本 prompt 两种方案,TTS-1 的 audio markup 属于文本标签路线 [待确认]; (4) [[Zero-shot Speech Synthesis]] 当前 SOTA 在 SEED-TTS-Eval 上 WER 已降至 1% 以下 (CosyVoice 3),TTS-1 的竞争力需看其在该 benchmark 上的表现 [confirmed]。
>
> **创新判断**: TTS-1 的主要创新不在单个组件,而在 **系统级工程整合** -- 将已有范式 (LLaMA backbone + X-codec2 + GRPO + LoRA 风格微调 + 流式推理) 组合为可部署的高质量 48kHz 多语言 TTS 系统,并开源训练代码。单独来看,每个组件都有先例;但作为完整系统 (尤其是 8.8B 参数规模 + 11 语言 + 48kHz + 开源),在工业级 TTS 中仍有参考价值。
>
> 检索命中: [[LLM-based TTS]]✓, [[Speech Language Model]]✓, [[Zero-shot Speech Synthesis]]✓, [[CosyVoice 2]]✓ | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Single-codebook vs Multi-codebook]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: Inworld AI 的 1.6B/8.8B 参数 LLaMA-based 自回归 TTS 系统,通过 pre-training → SFT → GRPO 三阶段训练在 human arena 中以 60% 胜率超越 11Labs、Cartesia、OpenAI TTS
> - **路线**: Text + Reference Audio → X-codec2 Encoder (16kHz, 50 tokens/s, 65536 codebook) → LLaMA SpeechLM (AR token generation) → X-codec2 Decoder (super-resolution to 48kHz) → Waveform
> - **指标**: TTS-1-Max WER 5.1% (EN avg, RL 后), arena 胜率 59-61% vs 11Labs/Cartesia/OpenAI [Table 8-9]; SIM 0.535 (decode with prompt) vs 0.495 (without) [Table 10]; DNSMOS 4.195 (48kHz codec) [Table 2]
> - **可借鉴**: (1) RMS loudness loss 解决流式音量不一致; (2) neutral-stylized speaker pairing + LoRA 实现风格控制同时保持说话人一致性; (3) 非发声区域拼接 + decoder context extension 解决流式拼接伪影
> - **局限**: 未报告 SEED-TTS-Eval 标准指标,难以直接与 CosyVoice 3 / Qwen3-TTS 等系统对比; arena 评估仅 400 票,统计显著性有限; 模型权重未开源

## 核心问题

TTS-1 试图解决的核心问题是: **如何将 LLM-based TTS 从研究原型推向可部署的工业级系统?** 具体包括:

1. 现有开源 TTS 模型缺乏 48kHz 高分辨率输出、稳定的多语言支持和可靠的实时流式能力 [§1]
2. SFT 后的 TTS 模型仍存在 hallucination (spurious audio tokens 导致 clicks/pops) 和质量不一致 [§3.5]
3. 风格/情感控制需要在保持说话人一致性的同时支持细粒度调节 [§3.6]
4. 流式推理中的音频拼接伪影和音量不一致问题 [§5.1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TTS-1 的架构由两个核心组件构成 [§2, Fig 1]:

1. **Audio Codec** (encoder + decoder): 基于 X-codec2 架构,将参考音频编码为 50 tokens/s 的离散序列 (单码本, K=65536),融合 acoustic 和 semantic 信息 [§2.1]。编码器处理 16kHz 输入;解码器通过 super-resolution 模块输出 48kHz 波形。

2. **SpeechLM**: LLaMA-3.2-1B (TTS-1) 或 LLaMA-3.1-8B (TTS-1-Max) 作为 backbone [§2.2]。词表从 128256 扩展到 193856 (新增 65536 audio tokens + 29 special tokens)。新 token embedding 用原始 embedding 矩阵的均值和协方差采样初始化 [§2.2]。

**推理流程**: Reference audio → encoder → audio tokens; audio tokens + reference text + target text → SpeechLM (AR generation) → generated audio tokens → decoder (with super-resolution) → 48kHz waveform [Fig 1]

### 关键设计选择

#### 1. 为什么选 X-codec2 单码本而非 RVQ 多码本?

论文给出三个理由 [§2.1]: (1) 1D causal 结构支持高效流式推理; (2) 开源实现方便从头训练; (3) 单码本极大压缩存储 (1 小时 48kHz 音频: 365MB 原始 → 0.19MB token)。[论文原文]

[agent 解读] 更深层的原因是: 单码本 = 单流 AR,与 text LLM 的标准 next-token prediction 完全一致,无需 VALL-E 那样的 AR+NAR 两阶段或 MusicGen 的 delay pattern。这使得可以直接复用 LLaMA 的训练基础设施和优化技巧 (FSDP, FlashAttention, torch.compile),大幅降低工程复杂度。

#### 2. 为什么需要 super-resolution 模块?

原始 X-codec2 仅支持 16kHz。TTS-1 通过在 decoder 中加入 interleaved strided 1D transposed convolution + ResNet blocks 实现超分辨率 [§2.1, Table 1]:
- 16kHz: hop=320, 无 stride
- 24kHz: hop=480, stride=1
- 48kHz: hop=160, stride=(3,2)

[论文原文] 作者特别指出,他们的 48kHz decoder 使用了比其他 iSTFT-based 模型显著更大的 hop length (160 vs CosyVoice 2 的 4),实验表明更大的 hop length 反而产生了更好的语音质量 [§2.1]。

#### 3. RMS loudness loss 解决了什么?

在 codec 训练中,decoder 输出经常无法保持原始音频的感知音量,尤其在短片段和高音段 [§3.2]。在流式场景下,这导致用户感知到突然的音量跳变。RMS loudness loss $L_{RMS}$ 通过惩罚原始/生成波形的 dB 级 RMS 差异来解决这个问题 [§3.2, Eq 3-4]:

$$L_{RMS} = E[(X_t - \hat{X}_t)^2], \quad X_t = 20\log_{10}\sqrt{\frac{1}{T}\sum x_t^2 + \epsilon}$$

[论文原文] $\lambda_{RMS} = 1.0$ 在 24kHz 和 48kHz 上都有效 [§3.2]。

#### 4. 三阶段训练为什么有效?

**Pre-training** [§3.3]: 约 1M 小时原始音频 (含 ~30K 小时非语音环境噪声) + 20B text tokens (RedPajama-v2 + LAION OIG)。音频分段为最长 40s,用 `<|speech_start|>` / `<|speech_end|>` 标记。加入 text tokens 的目的是防止 SFT 阶段文本理解能力退化 [§3.3]。[论文原文]

**SFT** [§3.4]: ~200K 小时高质量转写音频。过滤策略: 丢弃 DNSMOS 最低 20% + 去除 CPS 最快/最慢各 5% + 文本质量过滤 [§3.4]。关键发现: SFT 学习率必须初始化为 pre-training 的最终学习率 ($1.5 \times 10^{-5}$) [§3.4]。[论文原文]

**SFT 中混入 text instruction 数据会降低语音质量**: 尽管音频部分的 training loss 未受影响,但模型经常无法可靠生成语音 [§3.4]。[论文原文]

[agent 解读] 这与 CosyVoice 2 的经验相反 (CosyVoice 2 成功在 SFT 中加入 instruction-following 能力)。一个可能的解释是: TTS-1 使用单码本直接融合 semantic+acoustic 信息,text instruction 和 speech generation 的分布冲突在单码本空间中更严重;而 CosyVoice 的 semantic tokenizer + 独立 CFM decoder 架构天然分离了语义理解和声学生成。

**RL Alignment (GRPO)** [§3.5]: 使用 GRPO 对齐 SpeechLM 与人类偏好。composite reward:

$$R(p,c) = \alpha R_{wer}(c) + \beta R_{similarity}(p,c) + \gamma R_{dnsmos}(c) \quad [Eq\ 8]$$

- $R_{wer}$: Whisper-large-v3 转写 WER,经 $\exp(-2.5 \cdot WER)$ 映射到 $(0,1]$ [Eq 9]
- $R_{similarity}$: WavLM-large speaker verification 余弦相似度,归一化到 $[0,1]$ [Eq 10]
- $R_{dnsmos}$: DNSMOS 评分归一化到 $[0,1]$ [Eq 11]

关键设计 [§3.5]: (1) 禁用 reward scaling,使用 unscaled mean-centered advantages,避免 question-level difficulty bias [引用 Liu et al., 2025]; (2) reward pipeline 需将 token 解码为 48kHz 波形后再评估,确保 reward 反映最终音频质量; (3) 实验中 $\alpha = \beta = \gamma = 1.0$ [§3.5]。

**Pre-training 的关键性**: 消融实验 (100K 小时多语言数据) 显示,从 pre-trained SpeechLM 开始 SFT 比从 base LLaMA-3.2-1B-Instruct 开始可获得 ~15% WER 改进和 ~3% SIM 改进 [§3.4, Fig 5]。[论文原文]

#### 5. Audio Markup 系统的设计逻辑

8 种 speaking style + 7 种 non-verbal vocalizations,通过文本标签 (如 `[angry]`, `[breathe]`) 嵌入 prompt [Table 5]。

**关键难题**: 直接在 SFT 中 prepend style tag 无效 [§3.6]。[论文原文] 作者假设原因是 X-codec2 将 acoustic 和 semantic 信息纠缠在同一 latent space,模型无法从 prepended tag 中分离风格信息 [§3.6]。

**解决方案**: 构建 neutral-stylized 配对数据集 [§3.6]:
- 同一说话人的 neutral + stylized 语音配对
- 文本拼接 (style tag 作为分隔符),音频用 0.5-1.5s 静音间隔连接
- 每个 neutral 样本配对 1-5 个 stylized 样本
- 保留 ~30% 无配对 neutral 样本维持基础 TTS 能力
- 约 100K 样本,180 小时,340 说话人

**LoRA 优于全量 SFT**: 消融实验显示 LoRA 微调泛化性更好 (更低 eval loss) [§3.6, Table 6]。TTS-1 用 LoRA rank=16 / QKVO+MLP; TTS-1-Max 用 rank=32 / QKVO only [Table 6]。

[agent 解读] LoRA 在此场景下更优可能是因为: audio markup 训练数据量较小 (仅 180 小时),全量 SFT 容易在小数据上过拟合并损害已学到的一般 TTS 能力,而 LoRA 的低秩约束天然起到正则化作用。

### 训练策略

**计算开销** [§3.7]:
- SpeechLM pre-training + SFT: ~2 个月,4 节点 NVIDIA H100 (32 GPUs)
- 其他 (codec, RL, markup, 评估): ~3 个月,2 节点 A100 (16 GPUs)
- Throughput: TTS-1 ~46K tokens/s/GPU (PT) / ~18K tokens/s/GPU (SFT); TTS-1-Max ~8K tokens/s/GPU (PT) / ~4.8K tokens/s/GPU (SFT) [§3.3, §3.4]
- Pre-training wall time: TTS-1 ~2 days, TTS-1-Max ~10 days [§3.3]

**分布式策略**: TTS-1 用 DDP; TTS-1-Max pre-training 用 FSDP,SFT 改用 DeepSpeed Stage 2 (因 codec 推理占 ~4GB/GPU 额外内存) [§3.3, §3.4]。

### 流式推理设计

三个关键技术 [§5.1]:

1. **非发声区域拼接**: 只在音频片段的非发声区域 (|amplitude| < ε) 处切割拼接,避免波形不连续产生的 clicks/pops [Eq 13]。找不到非发声区域时,整段丢弃,token 留给下一 chunk 处理。

2. **音量稳定化**: 解码时扩展 context (将前一段末尾 ΔT 的 tokens 一并输入 decoder),解码后裁掉 context 部分。由于 decoder 远快于 token 生成 [Fig 9],这几乎不增加延迟。

3. **Prompt audio 解码**: 将 prompt audio tokens 和 generated tokens 一起输入 decoder,SIM 从 0.495 提升到 0.535 (English, non-streaming) [Table 10]。[论文原文] 作者认为这让 decoder 在声学空间中更好地对齐生成音频与参考音频。

**架构优化** (与 Modular 合作) [§5.2]: multi-step scheduler (GPU 上批量生成 token,无 CPU 中断)、batched decoder with custom kernels、Mojo 实现的 penalty sampling、MAX pipeline graph compiler。结果: 首 2 秒音频的 P90 延迟比 vanilla vLLM 快 ~70% [Fig 10]。

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| DNSMOS | 4.195 (48kHz decoder) | 4.178 (24kHz) / 4.110 (16kHz) | SEED-TTS-Eval EN subset | [Table 2] |
| WER (%) EN avg | 5.1 (TTS-1-Max, RL) | 6.9 (TTS-1-Max, SFT only) | Custom EN (short+mid+long) | [Table 8] |
| WER (%) EN avg | 6.3 (TTS-1, RL) | 7.9 (TTS-1, SFT only) | Custom EN (short+mid+long) | [Table 8] |
| SIM (non-streaming) | 0.535 (decode w/ prompt) | 0.495 (decode w/o prompt) | SEED-TTS-Eval EN subset | [Table 10] |
| Arena win rate | TTS-1-Max 59-61% | vs 11Labs/Cartesia/OpenAI | Internal arena, ~400 votes | [Table 9] |
| Arena win rate | TTS-1-Max 55.3% | vs TTS-1 | Internal arena | [Table 9] |
| P90 latency (first 2s) | ~30% of vLLM baseline | vLLM baseline | - | [Fig 10] |

**RL alignment 效果** [Table 8]: GRPO 使 TTS-1 的 WER 从 7.9% 降至 6.3% (相对 -20%),TTS-1-Max 从 6.9% 降至 5.1% (相对 -26%)。

**Arena 评估的局限**: 仅约 400 票,每对比较仅 28-52 票 [Table 9]; 使用 built-in English speakers (非零样本克隆); 评估者自行输入文本 (非标准化)。

**缺失的标准化对比**: 论文未在 SEED-TTS-Eval 的标准 test-zh/test-en 子集上报告 CER/WER/SIM,也未与 CosyVoice 3 (CER 0.71% zh, WER 1.45% en)、Qwen3-TTS (WER 1.24% en)、PilotTTS (WER 1.50% en) 等当前 SOTA 直接对比。

## 局限性

1. **缺乏标准 benchmark 对比**: 未报告 SEED-TTS-Eval 或 CV3-Eval 上的标准指标,自定义 benchmark 的 WER ~5% 远高于当前 SOTA (~1-2%),但无法确认是否为同一评估条件 [agent 解读]

2. **Reference audio 的 style bleed**: 缓存 prompt audio tokens 会导致参考音频的情感/风格特征 "泄漏" 到生成语音中,难以分离说话人身份与参考韵律 [§5.3]

3. **长序列质量退化**: 从短 prompt 生成长序列时质量下降 [§5.3]

4. **数据不均衡**: 训练数据的语言和风格分布不均导致跨语言质量差异,audio markup 的有效性因语言而异 [§5.3]

5. **解码参数 trade-off**: 低 temperature 提高 speaker similarity 但降低 expressiveness [§5.3]

6. **模型权重未开源**: 仅开源训练/建模代码,权重因安全考虑未公开 [§6]

## 点评

**优势**:
- **工程完整性**: 从 codec 训练到 RL alignment 到流式推理到 audio markup,覆盖了部署级 TTS 的全链路,这种端到端的系统级报告在学术论文中少见
- **实用的流式方案**: 非发声区域拼接 + 音量稳定化 + decoder context extension 三个技巧组合解决了流式拼接的核心工程痛点,思路简洁有效
- **LoRA + neutral-stylized pairing 的 style 控制方案**: 避免了额外 speaker embedding 模型 (如 MiniMax-Speech 和 CosyVoice 的做法),减少了系统复杂度
- **开源训练代码**: 虽然权重未公开,但 MIT 许可的训练/建模代码仍有参考价值

**不足**:
- **评估不充分**: 这是最大的问题。一篇声称 SOTA 的 tech report 未在任何公认 benchmark (SEED-TTS-Eval, LibriSpeech) 上报告标准指标。内部 arena 的 ~400 票在统计上不够 robust (每对仅 28-52 票),且使用 built-in speakers 而非零样本克隆,与多数学术评估设定不可比
- **技术创新有限**: 每个组件都有明确先例 (X-codec2 来自 LLaSA, GRPO 来自 DeepSeek, LoRA style fine-tuning 是常见方案, LLaMA backbone 是标准选择)。论文更像是一份工程集成报告而非方法创新论文
- **GRPO 实验不完整**: RL 实验仅在 1000 小时英语子集上进行 [§3.5],未展示多语言 RL 效果;reward 权重使用简单的 equal weighting ($\alpha=\beta=\gamma=1.0$) 而未做消融

## 可复用的 idea

1. **RMS loudness loss 用于 codec 训练** [§3.2]: 直接在 dB 域约束音量一致性,公式简单 ($\lambda_{RMS}=1.0$),对任何 iSTFT-based decoder 都适用,特别适合流式场景。已验证在 24kHz 和 48kHz 上均有效。

2. **Neutral-stylized speaker pairing + LoRA 实现风格可控** [§3.6]: 不需要额外的 speaker embedding 模型,通过数据构造让模型学会在同一说话人的 neutral 和 stylized 语音之间切换。LoRA 的低秩约束在小数据 (180h) 上提供了比全量 SFT 更好的泛化性。

3. **非发声区域拼接 (non-voicing concatenation)** [§5.1]: 流式 TTS 的通用方案 -- 检测波形的最后一个静音区域作为切割点,避免波形不连续。找不到就整段重新处理。简单但有效,可直接迁移到任何流式 TTS 系统。

4. **Decoder context extension 提升 speaker similarity** [§5.1, Table 10]: 将 prompt audio tokens 一并送入 decoder (而非只送 generated tokens),SIM 提升 ~8%。代价几乎为零 (decoder 远快于 AR token generation)。这个技巧对任何 codec-based TTS 都适用。

5. **Pre-training 阶段混入 text data 保留文本理解能力** [§3.3]: 在语音 pre-training 中混入 ~10% 高质量 text (RedPajama-v2 + LAION OIG instruction data),防止后续 SFT 阶段文本理解退化。注意: SFT 阶段混入 text instruction data 反而有害 [§3.4],仅 pre-training 阶段有效。

6. **SFT 学习率 = pre-training 最终学习率** [§3.4]: 一个简单但关键的 hyperparameter 选择,对最终语音质量有决定性影响。

> [!review] 审阅待补
> 审阅将在 Step 3.5 自动触发。
