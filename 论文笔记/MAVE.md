---
type: paper
tier: deep
title: "Speak, Edit, Repeat: High-Fidelity Voice Editing and Zero-Shot TTS with Cross-Attentive Mamba"
arxiv_id: "2510.04738"
source: "Sources/MAVE.pdf"
authors: [Baher Mohammad, Magauiya Zhussip, Stamatios Lefkimmiatis]
year: 2025
venue: "arXiv preprint"
tags: [speech-editing, zero-shot-TTS, Mamba, SSM, cross-attention, codec-LM, autoregressive, RVQ, efficiency]
concepts: ["[[CodecLanguageModel]]", "[[ResidualVectorQuantization]]", "[[Speech-TextAlignment]]", "[[Non-autoregressiveTTS]]", "[[TTSEvaluation]]", "[[PhonemeRepresentation]]"]
models: ["[[Whisper]]", "[[WavLM]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: MAVE 属于 [[CodecLanguageModel]] 范式,在 X-Codec (RVQ 8 codebooks, 50 Hz) 的离散 acoustic tokens 上做自回归生成。与 VALL-E 等典型 codec LM 不同,MAVE 用 Mamba (SSM) 替换 Transformer decoder 来建模 audio token 依赖,用 cross-attention 替换 token 拼接来实现文本条件注入。这是一条"高效 backbone + 显式跨模态注意力"的路线,区别于:
- **标准 AR codec LM** (VALL-E、VoiceCraft): Transformer decoder + text-audio 拼接,二次复杂度
- **NAR/Flow 方法** (FluentSpeech、VoiceBox、F5-TTS): 扩散/flow matching 并行生成,牺牲时序连贯性
- **Hybrid 方法** (CosyVoice): LLM 生成 semantic tokens + flow model 渲染 acoustic

**已有认知**: 
- [[ResidualVectorQuantization]] 的层级信息结构 (前层 coarse → 后层 fine) 是 codec LM 加权 loss 设计的基础 [confirmed]
- [[Zero-shotSpeechSynthesis]] 当前 SOTA (CosyVoice 3, Seed-TTS, IndexTTS2) 的 WER/SIM 数字远优于 MAVE 的实验结果,但这些系统规模大得多 (>1B, 100K+ h 数据) [confirmed]
- [[Speech-TextAlignment]] 归纳了 4 种 speech-text token 建模方式 (speech-only / text-only / concatenated / alternating),MAVE 的 cross-attention 方案不在这个分类中,因为它不把 text 和 audio 放入同一个序列 [待确认]
- [[CodecLanguageModel]] 的"序列长度"挑战 (200-400 tokens/s vs 12 phonemes/s) 正是 MAVE 用 Mamba 替换 Transformer 的核心动机 [待确认]
- [[Non-autoregressiveTTS]] 中 FluentSpeech 作为扩散方法代表,是 MAVE 的主要 NAR 对比基线 [待确认]

**创新判断**: MAVE 的核心创新是将 Mamba SSM 引入 codec LM 用于音频生成,并通过 cross-attention 解决 SSM 在跨模态条件注入上的固有缺陷。这在已有 KB 背景中没有先例 — 现有 codec LM 均基于 Transformer。但需要注意,论文仅与 VoiceCraft (2024) 和 FluentSpeech (2023) 对比,未与当前更强系统 (Seed-TTS, F5-TTS, MaskGCT) 进行比较。

> 检索命中: [[ResidualVectorQuantization]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[Speech-TextAlignment]](pending-review), [[Non-autoregressiveTTS]](pending-review), [[TTSEvaluation]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个将 Mamba (SSM) + cross-attention 成功应用于 text-conditioned speech editing 和 zero-shot TTS 的 codec LM,以 ~6x 内存优势超越 VoiceCraft
> - **路线**: Text → MFA phonemizer → 4-layer Transformer encoder → phoneme embeddings; Audio → X-Codec (8 RVQ codebooks, 50 Hz) → CM3 causal masking + token rearrangement → 12-layer Mamba decoder (每层后接 cross-attention to text) → delayed stacking → X-Codec decoder → output audio
> - **指标**: Speech editing MOS Nat 3.90 vs VoiceCraft 3.77 vs GT 4.00 [Table 1]; Zero-shot TTS SIM 0.57 vs VoiceCraft 0.55 vs GT 0.66 [Table 2]; 内存 6.2 GB vs VoiceCraft 37.9 GB [Table 4]; 均在 RealEdit / LibriTTS 上
> - **可借鉴**: (1) Mamba 处理 audio tokens + cross-attention 处理 text conditioning 的"分工"设计,利用 SSM 对高层特征的压缩记忆适配音频、cross-attention 保留文本精度; (2) CM3 causal masking + token rearrangement 实现 AR 框架下的双向上下文; (3) 不用 speaker embedding 而用 reference audio 前缀做 in-context speaker conditioning
> - **局限**: (1) 仅与 VoiceCraft/FluentSpeech 对比,未对比 Seed-TTS/F5-TTS/MaskGCT 等当前更强系统; (2) 仅训练 9K h GigaSpeech (英语 16kHz),规模受限; (3) 零样本 TTS 在长句 (>34 词) 时质量下降明显; (4) 代码未开源

## 核心问题

MAVE 要解决的核心问题是: **如何在保持高保真度的前提下,高效地进行 text-conditioned speech editing 和 zero-shot TTS?**

当前两类方法各有缺陷 [§1]:
1. **AR Transformer 方法** (VoiceCraft): 保真度好,但自注意力的二次复杂度限制上下文窗口,且处理 text-audio 长度不匹配效率低
2. **NAR 扩散/Flow 方法** (FluentSpeech, VoiceBox): 速度快,但时域连贯性和精细韵律控制差,尤其在嘈杂真实音频上

MAVE 的核心假设是: Mamba (SSM) 的线性复杂度 + 压缩状态机制天然适合音频 token 建模 (不需要逐 token 精确回忆,高层特征足够),而 cross-attention 可以在每个生成步骤显式访问文本信息,解决 SSM 对远距离文本的"模糊记忆"问题 [§3.2.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MAVE (830M 参数) 由三部分组成 [Fig 1]:

1. **Text Encoder**: 4-layer Transformer encoder,输入 MFA phonemized text,输出上下文化 phoneme embeddings z_text ∈ R^{M×d} [§3.2.2]
2. **Audio Tokenizer**: X-Codec [47],8 层 RVQ,50 Hz 帧率,每层 1024 entries,将音频编码为 K=8 个并行离散 token 流 [§3.1]
3. **Multimodal Decoder**: 12 层 Mamba block,每层后接一个 cross-attention module (query=Mamba output, key/value=z_text) [§3.2.2]

生成流程:
```
Text → MFA → Phonemes → Transformer Enc → z_text
Audio → X-Codec Enc → RVQ tokens → CM3 masking + rearrangement
→ [Mamba block → Cross-Attn(Q=audio, KV=z_text)] × 12
→ delayed stacking → X-Codec Dec → Output Audio
```

### 关键设计选择

**1. 为什么用 Mamba 替换 Transformer 来建模 audio tokens?**

Mamba (selective SSM) 以线性复杂度 O(L) 建模序列依赖,而 Transformer 自注意力为 O(L^2)。对于音频场景,RVQ 在 50 Hz 产生 200-400 tokens/s [§3],长音频的 token 序列极长。论文认为音频 token 建模不需要精确回忆每个历史 token,而需要保留高层特征 (speaker identity, prosody, acoustic continuity),Mamba 的压缩隐状态恰好适合 [§3.2.1] [论文原文]。

理论复杂度对比 [Appendix A.4]:
- Decoder-only Transformer: O(N_d · H · L_y · (L_x + L_y/2)) — L_y^2 项
- Mamba encoder-decoder: O(M_d · H · L_y · (L_x + 1)) — 仅 L_y 线性
- 当 L_y >> L_x 时差异显著 [论文原文]

**2. 为什么用 cross-attention 而非 token concatenation 来注入文本条件?**

这是论文最关键的设计选择。作者实验发现 (ablation Table 5): Mamba + token concatenation (text 和 audio 拼接后一起送入 Mamba) WER 高达 13.0,远差于 Mamba + cross-attention 的 7.8 [§4.4]。

论文解释 [§3.2.1]: Mamba 依赖 selective state-space 机制而非全局自注意力,对远距离 token 的记忆会"模糊化" (fuzzy memory)。当 text 和 audio tokens 拼接时,位于序列早期的 text tokens (音频序列极长) 的信息在到达后续 audio 生成步骤时已严重衰减。Cross-attention 通过独立的 encoder 输出在每个步骤直接访问完整文本表征,避免了这个问题 [论文原文]。

有趣的是,这种"记忆模糊"对 audio tokens 之间的建模反而不是问题,因为音频生成不需要精确保留每个低层 acoustic detail,保留 speaker identity 和 prosody 等高层模式就够了 [§3.2.1] [论文原文]。

**3. CM3 causal masking + token rearrangement**

为实现 speech editing (infilling),需要模型同时利用 masked span 前后的上下文。MAVE 采用 CM3 [1] 的因果掩码策略 [§3.1]:
- 训练时随机采样 m ~ Poisson(λ=1) 个 span (最多 N=3),长度 uniform[1,600] frames
- 将 masked span 移至序列末尾,插入 mask token M_j
- 例: [s1, s2, s3, s4, s5] 若 mask s2, s4 → [s1, M1, s3, M2, s5, M1, s2, M2, s4]
- 模型自回归地在最后一个 M1 后生成,重建 masked 内容 [论文原文]

这样做的巧妙之处在于: 模型在生成 s2 时能看到 s1 和 s3(s5 已在前面),实现了 AR 框架下的双向上下文访问 [agent 解读]。

**4. 不使用 speaker embedding 的 in-context speaker conditioning**

MAVE 不依赖显式 speaker embedding [§3.2.3]。对于 zero-shot TTS,将参考语音编码为 X-Codec tokens 后前置于输入序列,Mamba 通过长程递归从参考 tokens 中学习 speaker characteristics。对于 speech editing,周围未掩码音频已提供充分的说话人上下文,不需要额外参考 [论文原文]。

**5. Codebook 加权 loss**

X-Codec 的前 3 层 RVQ 编码 semantic/linguistic 内容,后 5 层编码 fine-grained acoustic details。加权 loss [§3.2.3, Table 7]:
- α = [0.25, 0.25, 0.25, 0.05, 0.05, 0.05, 0.05, 0.05]
- 遵循 SongGen [28] 的做法,优先保证感知重要特征的准确重建 [论文原文]

**6. Delayed stacking**

采用 codebook delay pattern [22, 9],确保第 k 层 code 在时间步 t 的生成以同时间步所有高层 [1,...,k-1] 的 code 为条件。这是 codec LM 中的标准做法 (MusicGen, SongGen 等) [§3.1] [agent 解读]。

### 训练策略

- **数据**: GigaSpeech 训练集,9K 小时,audiobooks + podcasts + YouTube,16kHz [§4]
- **优化器**: ScaledAdam + Eden Scheduler,lr=0.01 [Table 7]
- **Batch**: 400K frames (~133 min),50K steps with gradient accumulation [§4]
- **硬件**: 4× NVIDIA A100,约 4 天 [§4]
- **推理**: Nucleus sampling,p=0.8,temperature=1 [§4]
- **模型规模**: ~830M 参数 (12 decoder layers, dim 1808, 4 encoder layers) [Table 6]

## 实验

### Speech Editing (RealEdit benchmark)

| 指标 | MAVE | VoiceCraft | Ground Truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER (Whisper-L) ↓ | 7.5 | 8.4 | 6.8 | RealEdit (200 utts) | [Table 1] |
| WER (Whisper-M) ↓ | 5.9 | 6.9 | 5.2 | RealEdit | [Table 1] |
| MOS Naturalness ↑ | 3.90±0.08 | 3.77±0.08 | 4.00±0.08 | RealEdit (80 samples) | [Table 1] |
| MOS Intelligibility ↑ | 4.25±0.07 | 4.20±0.07 | 4.31±0.06 | RealEdit (80 samples) | [Table 1] |

Side-by-side MAVE vs GT (10 raters, 40 samples): 57.2% 认为两者相当, 24.8% 偏好原始, 18.0% 偏好 MAVE [§4.1]

三方对比 (MAVE vs VoiceCraft vs FluentSpeech, 14 samples, 20 raters) [Fig 2]:
- MAVE vs FluentSpeech: MAVE 获 62.9% naturalness preference, 59.3% intelligibility
- MAVE vs VoiceCraft: MAVE 获 33.6% naturalness preference (vs VoiceCraft 17.5%)

### Zero-shot TTS (LibriTTS)

| 指标 | MAVE | VoiceCraft | Ground Truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER (Whisper-L) ↓ | 7.4 | 9.3 | 6.2 | LibriTTS (372 utts) | [Table 2] |
| WER (Whisper-M) ↓ | 6.6 | 7.5 | 5.4 | LibriTTS | [Table 2] |
| SIM (WavLM-Large) ↑ | 0.57 | 0.55 | 0.66 | LibriTTS | [Table 2] |
| MCD ↓ | 4.73 | 4.75 | — | LibriTTS | [Table 2] |
| MOS Naturalness ↑ | 3.48±0.08 | 3.22±0.07 | 3.90±0.08 | LibriTTS (80 samples) | [Table 2] |
| MOS Intelligibility ↑ | 4.20±0.06 | 4.01±0.07 | 4.40±0.06 | LibriTTS (80 samples) | [Table 2] |

按句长分解 MOS (Table 3): 8-15 词时 MAVE Nat 3.71 vs GT 3.87 (差距小); >34 词时 MAVE Nat 3.46 vs GT 4.03 (差距大)。长句退化与 GigaSpeech 训练数据的中等句长分布一致 [§4.2] [论文原文]。

### Efficiency

| 指标 | MAVE | VoiceCraft (w/ KV cache) | VoiceCraft (w/o KV cache) | 出处 |
| --- | --- | --- | --- | --- |
| Avg/Max Memory (GB) | 6.2/6.5 | 37.9/40.2 | 9.2/9.7 | [Table 4] |
| Inference Time (RealEdit) | 5m17s | 4m33s | 22m31s | [Table 4] |
| Tokens/Sec | 53.5 | 74.9 | 15.1 | [Table 4] |

MAVE 内存 ~6x 低于带 KV cache 的 VoiceCraft。VoiceCraft 在 KV cache 模式下推理速度更快,但这是因为 RealEdit 的编辑段较短 (平均 ~85 tokens / 1.7s);对于长序列,MAVE 的线性复杂度理论上更优 [§4.3] [论文原文]。

### Ablation (Table 5, masked reconstruction, GigaSpeech val subset)

| Decoder | Conditioning | WER ↓ | MCD ↓ | F0 RMSE ↓ | PESQ ↑ | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Mamba (ours) | Cross-Attention | **7.8** | **4.29** | **0.280** | **2.08** | [Table 5] |
| Transformer | Cross-Attention | 10.8 | 4.58 | 0.293 | 1.99 | [Table 5] |
| Mamba | Concatenation | 13.0 | 4.48 | 0.284 | 2.02 | [Table 5] |

关键发现: Mamba+concat 最差 (验证 SSM 的跨模态 fuzzy memory 问题); Transformer+X-Attn 中等; Mamba+X-Attn 全面最优。两个组件缺一不可 [§4.4] [论文原文]。

## 局限性

1. **对比基线过时且有限**: 仅与 VoiceCraft (2024) 和 FluentSpeech (2023) 对比。未对比 Seed-TTS (CMOS -0.07 vs Human)、F5-TTS、MaskGCT、CosyVoice 3 等当前更强系统。以 SIM 为例,MAVE 0.57 vs 当前 SOTA IndexTTS2 0.865 (SEED-TTS-Eval),差距巨大,尽管评估协议不同 [agent 解读]
2. **训练规模受限**: 9K h GigaSpeech (英语 16kHz),远小于 Seed-TTS (数十万小时)、CosyVoice 3 (170K h) 等系统。语言覆盖仅英语 [§4]
3. **长句退化显著**: >34 词时 MOS Nat 3.46 vs GT 4.03,差距 0.57;短句 (8-15 词) 差距仅 0.16 [Table 3]
4. **推理速度非优势**: 在短段编辑任务上,VoiceCraft (w/ KV cache) 实际更快 (74.9 vs 53.5 tokens/s)。线性复杂度优势仅在长序列生成时体现,但未用长序列实验验证 [Table 4]
5. **未在标准 benchmark 评估**: 未报告 SEED-TTS-Eval 或 LibriSpeech test-clean 等标准评估集上的结果,降低了可比性 [agent 解读]
6. **代码未开源**: 论文未提供代码或模型权重,复现门槛高 [agent 解读]

## 点评

MAVE 的核心贡献是验证了 Mamba + cross-attention 这一组合在 speech editing / TTS 中的可行性,是 SSM 在 text-conditioned audio generation 中的首次成功应用。ablation study (Table 5) 清楚地展示了两个组件的互补性: Mamba 适合音频 token 的长程高层建模,cross-attention 弥补了 SSM 在跨模态条件注入上的固有缺陷。

然而,论文的实验设置存在明显不足:
1. **对比基线的时效性**: VoiceCraft (2024) 和 FluentSpeech (2023) 已不代表 SOTA。在论文提交时 (2025.10), CosyVoice 2/3、Seed-TTS、F5-TTS、MaskGCT 等系统已发布且性能远超 VoiceCraft [agent 解读]
2. **评估规模偏小**: MOS 评估仅 80 samples × 10 raters;side-by-side 仅 40 samples × 10 raters。SEED-TTS-Eval 等标准 benchmark 缺失 [agent 解读]
3. **效率论证不完整**: 论文强调 ~6x 内存优势,但实际推理速度在短段任务上反而更慢。理论上的长序列优势未用实验验证 [Table 4]

尽管如此,"将 SSM 引入 codec LM 并用 cross-attention 解决跨模态问题"这一架构思路值得关注。如果扩大训练规模并在标准 benchmark 上验证,Mamba backbone 可能成为 Transformer-based codec LM 的有竞争力的替代方案。

## 可复用的 idea

1. **SSM + cross-attention 的分工设计**: 用 SSM 处理长序列同模态依赖 (压缩记忆足够),用 cross-attention 处理跨模态条件注入 (需要精确访问)。这一分工原则可推广到其他"一长一短"的多模态生成场景 (如 text-conditioned music generation)
2. **CM3 causal masking for bidirectional AR**: 通过 token rearrangement 在纯 AR 框架内实现双向上下文访问。不引入新训练范式 (如 masked generative),仅靠序列操作实现 infilling
3. **Reference-based speaker conditioning without speaker embedding**: 将参考音频 tokens 直接前置于输入序列,利用 Mamba 的递归状态传递 speaker characteristics。避免了 speaker encoder 的额外参数和训练

> [!review] 审阅 (自动)
> 待审阅 — 见 `_review/MAVE-review.yml`
