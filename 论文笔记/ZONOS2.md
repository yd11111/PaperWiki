---
type: paper
tier: deep
title: "ZONOS2 Technical Report"
arxiv_id: "2606.24320"
source: "Sources/ZONOS2.pdf"
authors: [Gabriel Clark, Sofian Mejjoute, Mohamed Osman, George Close, Beren Millidge]
year: 2026
venue: "arXiv"
tags: [TTS, MoE, zero-shot, voice-cloning, autoregressive, codec-LM, multilingual, open-source, benchmark]
concepts: ["[[LLM-basedTTS]]", "[[ResidualVectorQuantization]]", "[[SpeakerEmbedding]]", "[[CodecLanguageModel]]", "[[TTSEvaluation]]", "[[PhonemeRepresentation]]"]
models: ["[[模型库/EnCodec|DAC]]", "Zonos-v0.1", "ZONOS2", "Qwen3-TTS", "Fish-S2-Pro", "VoxCPM2", "Cartesia-Sonic-3.5", "ElevenLabs-V3", "Gemini-3.1-Flash", "Inworld-TTS-2", "ZAYA1-8B"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis]]"]
datasets: ["[[数据集/SEED-TTS-Eval]]", "[[数据集/CV3-Eval]]", "ZTTS1-Eval", "FLEURS-R", "VoxBlink2"]
kb_context_sources: 6
status: draft
created: 2026-06-25
updated: 2026-06-25
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[ResidualVectorQuantization]]✓, [[SpeakerEmbedding]]✓, [[SEED-TTS-Eval]]✓, [[Zero-shotSpeechSynthesis]]✓, [[LLM-basedTTS]]✓, [[CodecLanguageModel]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ResidualVectorQuantization]], [[SpeakerEmbedding]], [[SEED-TTS-Eval]], [[Zero-shotSpeechSynthesis]], [[LLM-basedTTS]], [[CodecLanguageModel]] | 过滤: [[TTSEvaluation]](pending-review), [[PhonemeRepresentation]](pending-review), [[CV3-Eval]](pending-review) | 未命中但可能相关: MoE for TTS(无专页)

**谱系定位**: ZONOS2 属于 Codec Language Model / LLM-based TTS 路线,即 decoder-only transformer 直接在 RVQ-based audio codec tokens 上做自回归语言建模。与 VALL-E 开创的范式一脉相承,但首次在开源 TTS 中引入 Mixture-of-Experts (MoE) 架构。在 voice cloning 维度,采用经典的 speaker embedding 方案(ECAPA-TDNN encoder + LDA 降维),而非近期流行的 prompt-based in-context learning(如 CosyVoice 系列的 flow matching timbre extraction)。

**已有认知**: KB 中 [[SpeakerEmbedding]] 页记录了 speaker embedding 在 TTS 中的注入方式演进(从 lookup table 到 in-context prompt),ZONOS2 回归了单向量 prefix conditioning 的较早范式,但通过 LDA 降维和两阶段 annealing 解决了 embedding 过拟合问题,这是对该概念页"speaker encoder 局限"讨论的重要补充。[[SEED-TTS-Eval]] 页记录了当前 benchmark 的局限性(仅中英,评分模型过时),ZONOS2 提出的 ZTTS1-Eval 正是针对这些局限的系统性回应。

**创新判断**: MoE 在开源 TTS 中的首次应用是主要架构创新;byte-level tokenization 替代 G2P 在数据规模充足时的优势已有探索(如 Qwen3-TTS),但 ZONOS2 提供了更系统的 G2P 失败分析;ZTTS1-Eval 作为新 benchmark 填补了多语言+自发语音+韵律评估的空白。

## 速查

> [!summary] 速查
> - **一句话**: 首个 MoE 架构的开源 TTS 系统(8B total / 900M active),配合 6M 小时训练数据和新 benchmark ZTTS1-Eval,在 voice cloning 和多语言生成上达到 SOTA 级表现
> - **路线**: UTF-8 bytes + ECAPA-TDNN speaker emb (LDA→1024d) → decoder-only MoE transformer (28层, 900M active) → DAC 9-codebook delay pattern → waveform
> - **指标**: ZTTS1-Eval Clean en WER 2.76% / Spk.sim 78.6; ITW en WER 4.70% / Spk.sim 67.0; Quality Mode 显著改善非英语 WER (zh: 15.62→6.73) [Table IV, V]
> - **可借鉴**: (1) LDA 降维 speaker embedding 解决过拟合的两阶段训练方案; (2) 多 ASR 系统 ensemble 进行数据质量过滤; (3) ZTTS1-Eval 的 prosody diversity 评估维度(TTSDS2 + DS-WED)
> - **局限**: 非英语语言 WER 仍明显偏高(de 5.67%, ja 8.27%); MoE routing 在音频 token 上不稳定,需大量手动调参; Quality Mode 以牺牲 speaker similarity 换取 WER 改善,两者难以兼顾

## 核心问题

1. **如何在 TTS 模型中引入 MoE 而不牺牲推理效率?** — 传统 dense 模型扩大参数量会线性增加计算,MoE 允许 8B 参数但仅激活 900M
2. **Speaker embedding 的 bandwidth-overfitting 矛盾如何解决?** — 高带宽 embedding 携带 speaker identity 信息但也泄漏目标音频的 lexical/pause 信息,导致训练 shortcut
3. **现有 TTS benchmark (Seed-TTS-Eval) 的局限如何系统性弥补?** — 语言覆盖有限、评分模型过时、缺少韵律多样性度量

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ZONOS2 采用 decoder-only MoE transformer 骨干,直接在 DAC (Descript Audio Codec, 9 codebooks) 的 RVQ tokens 上做自回归语言建模 [§II]。

**输入序列组成**(按序拼接) [§II.C-F, Fig 1]:
1. Speaker embedding: ECAPA-TDNN 2048d → LDA 降至 1024d → 线性投影到 d_model=2048
2. Speaking-rate token: UTF-8 bytes/秒 → 离散 bucket → 单个 token
3. Quality/SNR/bandwidth/volume conditioning tokens (可选)
4. Text: UTF-8 byte 序列 (每个 byte ∈ [0,255] 对应一个 token)
5. Audio: DAC 9-codebook delay pattern tokens

**输出**: 延迟模式 (delay pattern) 下的 DAC tokens,每步预测 9 个 codebook 的 logits [§II.A]。

### 关键设计选择

**1. MoE 架构 [§II.F, Table II]**

28 层 transformer 中,前 3 层和最后 1 层为 dense,中间 24 层为 MoE (16 experts, top-1 routing),最后一个 MoE 层使用 top-2 routing。Router 设计基于 ZAYA1-8B [论文原文],采用 EDA (Exponential Depth Averaging) 跨层融合 router states + 三层 MLP with GeLU [§II.F]。总参数 8B,活跃参数 900M。

[agent 解读] 前 3 层和最后 1 层设为 dense 是为了 MoE routing 稳定性——论文报告音频 token 上的 routing 比文本数据显著更不稳定 [§IV.A],dense 层在序列开头和结尾提供稳定的 representation anchor。最后一个 MoE 层用 top-2 routing 也是稳定性考量。

**2. Byte-level Text Tokenization [§II.B, Table I]**

放弃 Zonos-v0.1 的 phoneme tokenization,改用原始 UTF-8 byte tokenization。论文给出了三个 G2P 失败案例 [Table I]:
- 中英 code-switch: 中文片段被英文 G2P 处理,输出乱码
- 过度匹配: alpharetrovirus → retroretrovirus (substring rule 误触发)
- 语言约束丢失: 西班牙语处理日文名 Satoshi → Satosi

[论文原文] 随着数据和模型规模增加,phonemization 的归纳偏置价值递减,byte-level 变体先追平后超越 phoneme-based 对标模型 [§II.B]。[agent 解读] 这与 Qwen3-TTS 的路线类似,均是在充分数据下放弃 G2P,但 ZONOS2 的数据规模 (6M h) 更大,为 byte tokenization 的收敛提供了更强保障。

**3. Speaker Embedding + LDA [§II.C]**

ECAPA-TDNN 从参考音频提取 2048d embedding e_x → LDA 投影到 1024d ê_x → 线性投影到 d_model。

LDA 的核心作用是 **decorrelation**: 保留区分不同说话人的方向,衰减同一说话人不同录音间变化的因素(duration, noise, lexical content, pause structure)[论文原文]。

[论文原文] 不做 LDA 时,speaker embedding 携带太多目标音频的 utterance-specific 信息(如文字内容和停顿位置),模型会走 shortcut——直接从 embedding 读出这些信息而非学习通用的文本到语音映射。LDA 通过最大化 between-speaker 方差 / within-speaker 方差来过滤这些 nuisance factors [§II.C]。

单 embedding 向量(而非完整参考序列)conditioning 的优势 [论文原文]: (1) 仅占序列 1 个位置,不随参考长度增长; (2) 2048d 高带宽足以捕获几乎所有 speaker characteristics [§II.C]。

**4. Delay Pattern [§II.A]**

对 DAC 的 N=9 codebooks 施加 shear 操作: codebook j 延迟 j 帧,使得 frame t 的 codebook j+1 token 在 codebook j token 之后立即生成。这将 within-frame 依赖转化为 autoregressive sequence dependency,每个 codebook 可以 condition on 前面的 codebooks [论文原文][§II.A]。

代价: streaming 解码需要 N-1=8 帧的 lookahead [§II.A]。

**5. Quality Conditioning [§II.E]**

两种机制:
- **Augmented embedding**: 训练时对 clone audio 施加噪声/压缩/混响增强,目标 DAC tokens 来自干净音频,使模型对低质量参考音频鲁棒 [§II.E]
- **Quality Mode token**: 最终 annealing 阶段,高质量训练子集配"Quality Mode"token;推理时启用可提升 intelligibility 和 UTMOS,但牺牲 speaker similarity [§IV.C]

### 训练策略

四阶段训练 [§IV]:

| 阶段 | Steps | Tokens | 关键特点 |
|------|-------|--------|---------|
| Pre-training | 77,500 | 2.9T | 基础 TTS,无 speaker embedding,无 quality conditioning [§IV.A] |
| Mid-training | 15,000 | 560B | 更严格的 transcript 一致性过滤 [§IV.B] |
| Annealing 1 | 10,000 | — | 引入 speaker embedding (target 的 random crop + loss masking),speaking-rate,quality tokens [§IV.C] |
| Annealing 2 | 10,000 | — | Speaker embedding 覆盖完整 target,移除 loss masking,引入 Quality Mode token [§IV.C] |

**数据**: 6.2M 小时,多语言(英语为主,含欧洲和亚洲语言)[Fig 3]。数据处理采用 VAD 分割 + 多 ASR 系统 ensemble 转写,通过 inter-ASR WER 阈值控制数据质量,pre-training 阈值低(数据多样性优先),annealing 阈值高(质量优先)[§III]。

**优化器**: Muon (base lr 5e-4, Muon lr 5e-3),weight decay 0.1,gradient clipping 0.5,cosine decay [§IV.A]。

**训练 loss**: NLL loss (masked) + MoE balancing loss (auxiliary-loss-free 方案,zero-mean balancing bias + 独立 AdamW 优化) [§II.G, Eq. 6-8]。Logit soft-capping with τ=15 [§II.G, Eq. 5]。

### ZTTS1-Eval Benchmark [§V]

| 维度 | Seed-TTS-Eval | CV3-Eval | ZTTS1-Eval |
|------|--------------|---------|-----------|
| 语言 | 2 | 9 | 17 (ITW) / 9 (Clean) |
| 音频类型 | read | read/expressive | read + spontaneous ITW |
| ASR scorer | Whisper-L / Paraformer | Whisper-L / Paraformer | **Qwen3-ASR** |
| Speaker scorer | WavLM | ERes2Net | **ReDimNet** |
| Quality scorer | — | DNSMOS | **MSR-UTMOS** |
| Prosody/diversity | — | — | **TTSDS2 + DS-WED** |

[Table III]

Clean set: 9 语言 x 500 utterances from FLEURS-R (~13h) [§V]
ITW set: 17 语言 x 1618 utterances from VoxBlink2 (~3h) [§V, Table VI]

## 实验

| 指标 | ZONOS2 8B | ZONOS2 QM | Qwen3-TTS | Fish-S2-Pro | VoxCPM2 | Cartesia-3.5 | 数据集 | 出处 |
|------|-----------|-----------|-----------|-------------|---------|-------------|--------|------|
| WER ↓ (en) | 2.76 | 3.99 | 1.94 | 3.60 | 4.23 | 2.56 | ZTTS1-Clean | [Table IV] |
| UTMOS ↑ (en) | 3.40 | 3.47 | 3.86 | 3.47 | 3.51 | 3.62 | ZTTS1-Clean | [Table IV] |
| Spk.sim ↑ (en) | 78.6 | 74.4 | 68.3 | 76.9 | 65.2 | 79.9 | ZTTS1-Clean | [Table IV] |
| WER ↓ (zh) | 15.62 | 6.73 | 2.91 | 4.33 | 5.01 | 4.53 | ZTTS1-Clean | [Table IV] |
| Spk.sim ↑ (zh) | 73.3 | 81.1 | 79.7 | 82.9 | 77.3 | 85.4 | ZTTS1-Clean | [Table IV] |
| WER ↓ (en ITW) | 4.70 | 2.21 | 1.05 | 2.09 | 1.69 | 1.40 | ZTTS1-ITW | [Table V] |
| Spk.sim ↑ (en ITW) | 67.0 | 56.9 | 61.5 | 65.0 | 68.1 | 70.2 | ZTTS1-ITW | [Table V] |
| Spk.sim (test-en) | 47.60 | — | — | — | — | — | Seed-TTS-Eval | [Table VII] |
| WER (test-en) | 2.05 | — | — | — | — | — | Seed-TTS-Eval | [Table VII] |

**TTSDS2 Prosody** [Fig 5]: ZONOS2 在 ITW English set 上 prosody 得分最高;在 Clean set 上与顶部模型竞争力相当。

**DS-WED** [Fig 6]: ZONOS2 在两个 eval set 上均展示了显著更高的 prosodic variation(生成多样性),远超所有对比系统。

**Allosaurus SR Distribution** [Fig 7]: ZONOS2 的 prosodic content distribution 最接近源音频分布。

**Quality Mode 效果**: 非英语 WER 显著改善(zh: 15.62→6.73, de: 5.67→3.74 [Table IV]),UTMOS 普遍提升,但 speaker similarity 下降(en: 78.6→74.4 [Table IV])。在 ITW set 上 Quality Mode 的正面效果更为一致,所有语言的 WER 和 UTMOS 均改善 [Table V]。

## 局限性

1. **非英语 WER 偏高**: zh baseline 模式 WER 15.62%,即使 Quality Mode 下也达 6.73%,远高于 Qwen3-TTS (2.91%);日语、韩语等语言 WER 也不理想 [Table IV]
2. **MoE routing 不稳定**: 论文坦承音频 token 上的 MoE balancing 比文本数据困难得多,normalized entropy 多次 collapse 到 0.6,需要大量手动干预学习率 [§IV.A, §VII.B]。[⚠️ 论文未详述] 具体为什么音频数据的 routing 更不稳定,仅给出了三个推测方向
3. **Quality Mode 的 speaker similarity 代价**: Quality Mode 提升 WER 和 UTMOS 但降低 speaker similarity(en: 78.6→74.4),本质上是 intelligibility 和 voice cloning fidelity 的 trade-off,两者难以同时最优 [§VI]
4. **GQA vs MHA trade-off**: 论文发现 MHA 比 GQA 更稳定且生成质量更高,但为了推理速度选择了 GQA [§VII.A]。[agent 解读] 这意味着当前架构可能未充分发挥注意力机制的能力
5. **Speaker embedding 过拟合问题**: 虽然 LDA + 两阶段 annealing 缓解了过拟合,但论文承认训练 horizon 仍然有限,causal leak 导致的推理不稳定(silent output 或 glossolalia)需要多种干预 [§VII.C]
6. **Benchmark 自评偏差**: ZTTS1-Eval 由论文作者提出,ZONOS2 在其上的优势(尤其是 prosody metrics)需要独立验证。论文也承认无法确认对比模型是否在 ZTTS1-Eval 数据上训练过 [§V]

## 点评

**优势**:
- **架构创新的实践价值**: MoE 在开源 TTS 中的首次应用,证明了 8B→900M active 的稀疏化路线在 TTS 中可行,为后续大规模 TTS 模型的效率优化开辟了新方向
- **工程细节的坦诚度**: 论文详细记录了 MoE routing 不稳定、speaker embedding 过拟合、GQA vs MHA 等实际困难和 trade-off,这种工程透明度在 TTS 论文中难能可贵
- **ZTTS1-Eval 的系统性**: 从 scoring stack 到语言覆盖到 prosody metrics 的全面升级,填补了现有 benchmark 的多个空白
- **开源承诺**: Apache 2.0 许可发布模型权重、推理代码和 benchmark

**不足**:
- **非英语性能差距明显**: 尽管训练数据 6.2M 小时,中文 WER 仍远落后于 Qwen3-TTS 和 Fish-S2-Pro,说明数据量不是万能的,数据质量和语言特异性处理仍然关键
- **消融实验不足**: 论文缺少系统性消融(如 MoE vs dense 在同等 FLOPs 下的对比、LDA 降维维度的影响、不同 codebook 数量的效果),许多设计选择停留在定性讨论
- **Speaker conditioning 方案相对保守**: 回归单向量 embedding + prefix conditioning,在 prompt-based in-context learning(CosyVoice, Seed-TTS)已成为主流的背景下,这个选择是否最优值得商榷

## 可复用的 idea

1. **LDA decorrelation for speaker embedding**: 用 LDA 投影过滤 speaker embedding 中的 nuisance factors (duration/noise/lexical content),延长训练 horizon。这个技巧可迁移到任何使用 speaker encoder conditioning 的 TTS 系统
2. **两阶段 speaker conditioning annealing**: Stage 1 用 random crop + loss masking 防止 shortcut learning; Stage 2 去除 masking 覆盖全序列。比一步到位引入 speaker embedding 更稳健
3. **Multi-ASR ensemble 数据过滤**: 用多个 ASR 系统的 inter-system WER 作为数据质量代理,不同训练阶段使用不同阈值。简单有效的数据清洗方案
4. **ZTTS1-Eval 的 prosody diversity metrics**: TTSDS2 + DS-WED 可作为通用的 TTS 韵律多样性评估工具,补充 WER/SIM 的不足
5. **Dense anchor layers in MoE**: 在 MoE backbone 的首尾保留 dense 层以稳定 routing,这个模式可能对其他模态的 MoE 应用也有参考价值
