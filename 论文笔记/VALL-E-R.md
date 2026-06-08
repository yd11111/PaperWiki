---
type: paper
tier: deep
title: "VALL-E R: Robust and Efficient Zero-Shot Text-to-Speech Synthesis via Monotonic Alignment"
arxiv_id: "2406.07855"
source: "Sources/VALL-E-R.pdf"
authors: [Bing Han, Long Zhou, Shujie Liu, Sanyuan Chen, Lingwei Meng, Yanming Qian, Yanqing Liu, Sheng Zhao, Jinyu Li, Furu Wei]
year: 2024
venue: "arXiv preprint"
tags: [TTS, zero-shot, codec-LM, monotonic-alignment, robustness, inference-efficiency, AR-NAR, phoneme-alignment]
concepts: ["[[ResidualVectorQuantization]]", "[[LLM-basedTTS]]", "[[Speech-TextAlignment]]", "[[PhonemeRepresentation]]", "[[Non-autoregressiveTTS]]"]
models: ["[[EnCodec]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ResidualVectorQuantization]]✓, [[LLM-basedTTS]]✓, [[EnCodec]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[CodecLanguageModel]][待确认], [[Speech-TextAlignment]][待确认] | 未命中但可能相关: 无

**谱系定位**: VALL-E R 是 VALL-E 家族的第三代改进(VALL-E → VALL-E X → VALL-E R),同期竞争者包括 ELLA-V(phoneme-acoustic interleaving)和 RALL-E(CoT prompting)。三者均针对 VALL-E 的鲁棒性问题提出不同解决路线。在 [[LLM-basedTTS]] 页的 VALL-E 系列演进表中,VALL-E R 被定位为"单调对齐增强鲁棒性"。

**已有认知**: [[ResidualVectorQuantization]] 页记录了 EnCodec 使用 8 层 RVQ,75Hz 采样率,codebook size 1024 的配置。[[EnCodec]] 页记录了其 encoder-decoder + RVQ 架构。[[Zero-shotSpeechSynthesis]] 页显示当前 SOTA(CosyVoice 3, IndexTTS2, Qwen3-TTS)已远超 VALL-E 系列水平。[[Speech-TextAlignment]][待确认] 页讨论了 speech-text token 的四种建模方式(speech-only, text-only, concatenated, alternating),VALL-E R 的 phoneme 同步预测本质上是一种 alternating alignment 的变体。

**创新判断**: VALL-E R 的两个贡献(codec-merging 和 monotonic alignment)分别解决 decoder-only codec LM 的效率和鲁棒性问题。codec-merging 是无需重训的推理加速方法(通过 average pooling + repeat 下采样 RVQ 第一层),monotonic alignment 则是将 encoder-decoder 时代的对齐先验引入 decoder-only 架构的首次尝试。相比 ELLA-V 的 phoneme-acoustic interleaving(引入大量额外 token),VALL-E R 的 monotonic alignment 不增加序列长度,是更高效的鲁棒性方案。

## 速查

> [!summary] 速查
> - **一句话**: 在 VALL-E 基础上引入 phoneme 单调对齐策略增强鲁棒性 + codec-merging 下采样加速推理,WER 逼近 ground truth 且推理速度提升 60%+
> - **路线**: Text → G2P → Phoneme + 3s Acoustic Prompt (Merged EnCodec codes) → AR model (同步预测 acoustic token + aligned phoneme, 受 monotonic alignment 约束) → NAR model (逐层生成 2-8th quantizer codes, 条件为 aligned phoneme) → EnCodec Decoder → Waveform
> - **指标**: Continuation WER 1.58% vs VALL-E 2.37% vs GT 1.41% [Table 1]; Cross-sentence WER 3.18% vs VALL-E 5.48% [Table 1]; Spk-Sim 0.876 vs VALL-E 0.875 [Table 1]; 推理速度 3.67s/10s vs VALL-E 10.27s/10s [Table 4]; QMOS 4.02 vs VALL-E 3.96 [Table 2]
> - **可借鉴**: (1) codec-merging: 通过 average pooling + repeat 在 RVQ 层间插入下采样模块,无需重训 codec 即可降低采样率,可迁移到任何 RVQ codec; (2) monotonic alignment: 在 decoder-only LM 中同步预测 phoneme 作为文本对齐锚点,用 Bernoulli 采样控制 phoneme pointer 推进,解决 attention degradation
> - **局限**: 仅在 LibriSpeech 960h 上训练和评估,规模较小; 未与 flow-based/diffusion-based 方法对比; codec-merging 仅下采样第一层(多层下采样质量显著下降); 未开源; monotonic alignment 依赖外部 forced aligner(MFA)提供训练对齐

## 核心问题

VALL-E 作为 codec LM TTS 的开创者,存在两个核心痛点 [§1]:

1. **鲁棒性不足** — decoder-only transformer 仅通过 self-attention 隐式捕获 phoneme 与 acoustic token 的单调关联,面对复杂/长序列时 attention 退化,导致漏字(word skipping)、重复(repetition)和错读(typos) [§1]
2. **推理效率低** — EnCodec 采样率 75Hz,10 秒语音需 750 步自回归,计算开销巨大 [§1]

同期工作 ELLA-V 和 RALL-E 虽然改善了鲁棒性,但前者引入大量额外 phoneme token(约 2x105 个/10s)导致推理更慢 [Table 4],后者引入 CoT 控制信息也增加推理步数 [Table 4]。VALL-E R 的目标是同时解决鲁棒性和效率两个问题。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VALL-E R 沿用 VALL-E 的 AR + NAR 两阶段架构 [§3],在此基础上引入两个改进:

1. **Codec-Merging Module**: 插入 EnCodec 的 encoder 和 RVQ 之间,对第一层离散码进行 2x 下采样(75Hz → 37.5Hz),减少 AR 步数 [§3.1]
2. **Monotonic Alignment Strategy**: AR 模型在训练时同步预测 acoustic token 和对应的 aligned phoneme;推理时用单调约束引导 phoneme pointer 逐步推进,确保生成覆盖所有 phoneme [§3.2]

### 关键设计选择

#### 1. Codec-Merging: 无需重训的下采样 [§3.1]

**为什么不重训 codec?** [论文原文] 论文指出这种方法"无需任何重训或微调"即可实现下采样 [§3.1],这是关键优势 — 重训 codec 成本高且可能影响其他层的重建质量。

**具体机制**: 在 EnCodec 的 encoder 和 VQ 之间插入 merge 模块 [Fig 2]:
- 对第 d 层残差 r_d(维度 F x T)做 average pooling 下采样为 F x (T/m_d)
- 再 repeat 上采样回原始长度 F x T
- 送入 VQ 层进行最近邻查找
- 效果: 连续 m_d 帧的 code 被强制相同,等效降低了第 d 层的分辨率 [§3.1]

**为什么只下采样第一层?** [论文原文] 实验表明只下采样第一层(2x)时,PESQ 和 STOI 几乎无损(PESQ-NB 3.5686 vs 原始 3.6188, STOI 0.9468 vs 0.9498);扩展到 4 层或 8 层时性能显著下降(8 层 2x: PESQ-NB 3.1820, STOI 0.9274) [Table 5]。[agent 解读] 这说明第一层编码的是 coarse 信息,时间冗余度高;后续层编码 fine details,分辨率更敏感。

#### 2. Monotonic Alignment: decoder-only 的对齐先验 [§3.2]

**为什么不能直接用传统 monotonic attention?** [论文原文] 传统 monotonic attention (He et al. 2019, Raffel et al. 2017) 基于 encoder-decoder 架构,依赖 cross-attention 实现;而 VALL-E 使用 decoder-only 架构,不存在 cross-attention [§2.2]。

**训练: 同步预测 phoneme + acoustic token** [§3.2.1]

AR 模型的训练目标从单纯预测 acoustic token 扩展为同时预测 acoustic token 和 aligned phoneme:

$$p(a^1_{1:T}, \hat{p}_{1:T} | p; \theta_{AR}) = \prod_{t=1}^{T} p(a_t, p_t | a_{1:t-1}, \hat{p}_{1:t-1}, p_{1:L}; \theta_{AR})$$

其中 $\hat{p}_{1:T}$ 是通过 forced aligner (MFA) 将 phoneme 序列 p 与 acoustic token 对齐后的重复 phoneme 序列 [§3.2.1]。

[agent 解读] 这意味着模型在训练时被迫学习"当前正在生成哪个 phoneme 的声学表征",从而在 self-attention 中建立显式的 phoneme-acoustic 对应关系,而非像 VALL-E 那样完全隐式学习。

**推理: Bernoulli 采样控制 phoneme pointer** [§3.2.2]

推理时,模型在每步同时输出 acoustic token 和 phoneme 预测。Phoneme pointer 通过 Bernoulli 采样决定是保持当前 phoneme 还是跳到下一个:

$$z_{i,j} \sim \text{Bernoulli}\left(\frac{1}{1 + \exp(e_{i,j})}\right)$$

其中 $e_{i,j}$ 是模型输出的当前 phoneme 概率 [§3.2.2]。

这保证了三个性质 [§3.2.2]:
- **Locality**: 每个 phoneme 对应一个或多个连续 acoustic tokens,不会错位
- **Monotonicity**: phoneme pointer 只能前进不能后退,防止重复
- **Completeness**: 每个 phoneme 至少对应一个 acoustic token,防止漏读

**NAR 阶段也利用 phoneme 信息** [§3.2.1]

NAR 模型以 aligned phoneme 序列 $\hat{p}_{1:T}$ 和已生成的前几层 acoustic tokens 为条件,逐层生成 2-8 层 tokens:

$$p(a^{2:8}_{1:T} | \hat{p}_{1:T}, p_{1:L}; \theta_{NAR}) = \prod_{n=2}^{8} p(a^n_{1:T} | \hat{p}_{1:T}, a^{1:n-1}_{1:T}, p_{1:L}; \theta_{NAR})$$

#### 3. Prosody Control: phoneme 作为 duration 控制接口 [§3.2.3]

[论文原文] 由于 VALL-E R 显式建模了 phoneme 序列,可以用预设的 phoneme 序列(来自另一段语音的 forced alignment 结果)替换自预测的 phoneme,从而独立控制韵律(duration)和音色(timbre) [§3.2.3]。这本质上实现了 voice conversion: 保持源语音的语言信息和韵律,替换为 prompt 的音色。

### 训练策略

- **数据**: LibriSpeech 960h [§4.1]
- **Codec**: EnCodec 24kHz, 8 层 RVQ, 75Hz [§4.1]; 对齐工具: MFA [§4.1]
- **Vocoder**: Vocos (与 EnCodec 对齐) [§4.1]
- **模型**: AR 和 NAR 均为 12 层 Transformer, 16 heads, dim 1024, FFN 4096, dropout 0.1 [§4.4]
- **训练**: 8x NVIDIA V100 16GB, 400K steps, AdamW, warmup 32K steps, peak lr 5e-4, linear decay + weight decay 0.01 [§4.4]
- **AR prompt**: 使用 first 3 seconds 作为 acoustic prompt [§4.4]

## 实验

| 指标 | VALL-E R | VALL-E | ELLA-V | VALL-T | Ground Truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER (Continuation) ↓ | 1.58% | 2.37% | 2.10% | 2.28% | 1.41% | LibriSpeech test-clean | [Table 1] |
| WER (Cross-sentence) ↓ | 3.18% | 5.48% | 7.15% | 4.16% | - | LibriSpeech test-clean | [Table 1] |
| Spk-Sim (Continuation) ↑ | 0.876 | 0.875 | 0.856 | 0.870 | 0.923 | LibriSpeech test-clean | [Table 1] |
| Spk-Sim (Cross-sentence) ↑ | 0.974 | 0.975 | - | 0.975 | - | LibriSpeech test-clean | [Table 1] |
| QMOS ↑ | 4.02 ± 0.20 | 3.96 ± 0.18 | - | - | 4.22 ± 0.11 | LibriSpeech test-clean | [Table 2] |
| SMOS ↑ | 3.89 ± 0.16 | 3.84 ± 0.21 | - | - | 4.18 ± 0.15 | LibriSpeech test-clean | [Table 2] |
| CMOS ↑ | +0.07 | 0.00 | - | - | +0.33 | LibriSpeech test-clean | [Table 2] |
| MCD-DTW-SL (Prosody) ↓ | 8.55 | 9.03 | 11.77 | - | - | LibriSpeech test | [Table 3] |
| MCD-DTW-SL (w/ preset phoneme) ↓ | 7.82 | - | - | - | - | LibriSpeech test | [Table 3] |
| Avg. Inference Time (10s speech) ↓ | 3.67s | 10.27s | 15.76s | 12.28s | - | - | [Table 4] |
| Avg. AR Steps ↓ | 375 | 750 | ~960 | ~855 | - | - | [Table 4] |

**关键实验发现**:

1. **鲁棒性**: VALL-E R 的 continuation WER (1.58%) 逼近 EnCodec 上界 (1.62%),几乎消除了 codec LM 引入的额外错误 [Table 1]。Cross-sentence 场景下 WER 3.18% 远优于 ELLA-V 的 7.15%,说明 ELLA-V 的局部推进策略在内容不连续时退化严重 [Table 1]。

2. **效率**: 2x 下采样将 AR 步数从 750 减半至 375,但实际速度提升超过 2x(3.67s vs 10.27s),因为 Transformer self-attention 的计算复杂度随序列长度二次增长 [§5.3]。

3. **消融 — Codec-Merging**: 移除 merged codec 后 WER 从 1.58% 微升至 1.65%,Spk-Sim 基本不变(0.876 vs 0.877),证明下采样不损害质量 [Table 6]。

4. **消融 — Monotonic Alignment**: 移除 MA 后模型退化为 VALL-E,WER 从 1.58% 升至 2.37%,证明 MA 是鲁棒性提升的主要来源 [Table 6]。

5. **top_p 鲁棒性**: VALL-E 在低 top_p 时因缺乏文本控制,模型倾向预测静音帧导致死循环和极高 WER;VALL-E R 即使在低 top_p 下仍保持正常生成,因为 phoneme pointer 提供了显式进度控制 [Fig 4]。

6. **Attention 可视化**: VALL-E 的 attention 随生成长度增加而逐渐分散,导致注意力混乱;VALL-E R 的 attention 均匀分布(不需要关注读到哪了),phoneme alignment path 呈现完整的单调对角线 [Fig 5]。

## 局限性

1. **数据规模有限**: 仅在 LibriSpeech 960h 上训练,远小于 VALL-E 原始论文的 60K h;未验证 scaling 行为 [agent 解读]

2. **评估范围单一**: 仅在 LibriSpeech test-clean 上评估,未测试多语言、噪声环境、罕见词等鲁棒性场景 [agent 解读]

3. **Codec-Merging 的天花板**: 仅第一层可安全下采样;3x 和 4x 虽"仅轻微下降"但论文未给出 TTS 端到端评估,只有 codec 重建指标 [Table 5]

4. **依赖外部对齐器**: 训练和推理均需 MFA 提供 phoneme-acoustic 对齐,增加了 pipeline 复杂度且对齐质量直接影响模型性能 [§4.1]

5. **未与非 AR 方法对比**: 同期的 VoiceBox (flow-based NAR) 和 CLaM-TTS (Mel-VAE) 推理速度已达 6.2s 和 4.15s [Table 4],与 VALL-E R 的 3.67s 差距不大,但这些方法不需要自回归,scalability 可能更好 [agent 解读]

6. **未开源**: 论文未提供代码或模型权重

## 点评

**核心价值**: VALL-E R 对 decoder-only codec LM 的两个核心痛点(鲁棒性和效率)给出了优雅的解决方案。Monotonic alignment 将传统 encoder-decoder TTS 的对齐先验无缝移植到 decoder-only 架构,思路清晰且实证有效。Codec-merging 是一个简洁的工程 trick,不需改动训练流程即可获得显著加速。

**方法论洞察**: VALL-E R 的 phoneme 同步预测本质上是一种显式的 speech-text alignment 机制。它将 decoder-only LM 中隐式的"attention 负担"(同时学习"读到哪了"和"怎么读")分解为显式的两个子任务: phoneme pointer 负责进度追踪,acoustic token 预测负责声学生成。这种"解耦"思路在后续 VALL-E 2 的 grouped codec 和 repetition-aware sampling 中也有体现。

**历史定位**: 从 2024 年回望,VALL-E R 处于 codec LM TTS 从"能用"到"好用"的过渡期。它解决了鲁棒性问题,但整个 codec LM 路线后来被 CosyVoice 等 hybrid 架构(LLM + flow matching)和 F5-TTS 等纯 NAR 方法超越。codec-merging 的思路在后续 SNAC 等多尺度 RVQ 设计中得到了更系统的发展。

## 可复用的 idea

1. **Codec-Merging 作为即插即用加速模块**: 对任何使用 RVQ codec 的 AR 系统,都可以尝试在第一层 RVQ 前插入 avg-pool + repeat 下采样,无需重训 codec 即可获得 2x+ 推理加速。适用条件: 第一层编码 coarse 信息且时间冗余度高。

2. **Decoder-only 中的显式对齐**: 在 decoder-only LM 中同步预测辅助序列(phoneme/duration/pitch)作为生成进度的显式锚点,用 monotonic constraint 限制辅助序列的推进。可推广到任何需要结构化输出的 decoder-only 生成任务。

3. **Bernoulli phoneme pointer**: 用 sigmoid + Bernoulli 采样实现软单调推进,比 hard attention 更灵活(允许一个 phoneme 对应多个 acoustic tokens),又比无约束 attention 更可控。可用于任何 monotonic sequence-to-sequence 任务。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节完整解释 WHY + HOW,含公式和推理流程 |
> | 可信赖 | pass | 所有数字标注出处,指标名正确,速查含具体数字 |
> | 可区分 | pass-with-fixes | 来源标注覆盖率约 80%,个别公式语义解读未标注 |
> | 可定位 | pass | KB 背景含谱系定位(VALL-E 系列 + 同期竞品) |
> | 不污染 | pass | 无反向更新 |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/VALL-E-R-review.yml`

---
检索命中: [[ResidualVectorQuantization]]✓, [[LLM-basedTTS]]✓, [[EnCodec]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[CodecLanguageModel]][待确认], [[Speech-TextAlignment]][待确认] | 未命中但可能相关: 无
