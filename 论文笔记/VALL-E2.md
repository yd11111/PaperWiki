---
type: paper
tier: deep
title: "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers"
arxiv_id: "2406.05370"
source: "Sources/VALL-E2.pdf"
authors: [Sanyuan Chen, Shujie Liu, Long Zhou, Yanqing Liu, Xu Tan, Jinyu Li, Sheng Zhao, Yao Qian, Furu Wei]
year: 2024
venue: "arXiv"
tags: [TTS, zero-shot, codec-LM, human-parity, sampling-strategy, grouped-modeling, AR-NAR]
concepts: ["[[LLM-basedTTS]]", "[[ResidualVectorQuantization]]", "[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[CodecLanguageModel]]", "[[Non-autoregressiveTTS]]"]
models: ["[[EnCodec]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["Libriheavy", "LibriSpeech", "VCTK"]
kb_context_sources: 4
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[LLM-basedTTS]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]], [[SpeechTokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]]✓, [[ResidualVectorQuantization]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechTokenizer]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[Non-autoregressiveTTS]](pending-review) | 未命中但可能相关: 无

**谱系定位**: VALL-E 2 是 [[论文笔记/VALL-E|VALL-E]] 的直接续作。LLM-based TTS 页的 VALL-E 系列演进表中已记录本文的两项改进(重复感知采样 + 分组编码)和"人类水平"结论。在 CodecLanguageModel 页中,VALL-E 2 的 grouped code modeling 被列为多层 RVQ 建模的三大方案之一(与 AR+NAR 两阶段、RQ-Transformer 并行方案并列)。

**已有认知**: RVQ 页记录了 EnCodec 的多层级信息结构(前层 coarse、后层 fine),这正是 VALL-E 2 grouped code modeling 的设计基础。SpeechTokenizer 页记录了 EnCodec 属于声学 tokenizer,而 SemanticvsAcousticTokens 页指出纯 acoustic tokens 在语义任务上接近随机(~50%)。VALL-E 系列始终走纯 acoustic token 路线,不使用 semantic tokens,这限制了其语义对齐能力但保证了声学保真度。

**与前作对比**: [[论文笔记/VALL-E|VALL-E]] 的精读笔记记录了两个核心痛点: (1) 鲁棒性不足 — AR 模型存在词漏/重复/错序; (2) 推理慢 — 受限于 EnCodec 75Hz 帧率的逐 token 生成。VALL-E 2 的两项改进分别精准对应这两个痛点。同时,VALL-E 的 speaker similarity 与 ground truth 仍有明显 gap (0.580 vs 0.754),这是 VALL-E 2 需要缩小的距离。

**创新判断**: VALL-E 2 不是架构级创新,而是在成熟范式上做工程层面的精准优化。Repetition Aware Sampling 解决了 codec LM 推理中 nucleus sampling 的 infinite loop 问题(VALL-E 系列的已知痛点); Grouped Code Modeling 缩短序列长度,同时缓解长上下文建模的困难。两项改进叠加使系统首次在 LibriSpeech/VCTK 上达到 human parity。

## 速查

> [!summary] 速查
> - **一句话**: 在 VALL-E 基础上引入 Repetition Aware Sampling(自适应切换 nucleus/random sampling 避免 infinite loop）和 Grouped Code Modeling（将 codec codes 按 group 建模缩短序列长度），首次在 LibriSpeech/VCTK 上达到 human parity 零样本 TTS
> - **路线**: Text → BPE → Text Tokens + Speech Prompt → EnCodec → Grouped Code Prompt → AR Transformer (RAS decoding, 生成 1st quantizer grouped codes) → NAR Transformer (greedy decoding, 生成 2-8th quantizer codes) → Vocos Decoder → Waveform
> - **指标**: WER 1.5% / SIM 0.782 / DNSMOS 3.987 (LibriSpeech, 3s prefix, single sampling, group=1) [Table 1]; SMOS 4.61 > GT 4.13 / CMOS +0.033 > 0 (LibriSpeech, 40 speakers) [Table 2]; WER 0.9% / SIM 0.487 (VCTK, 3s, single, group=2) [Table 4]
> - **可借鉴**: (1) Repetition Aware Sampling: 在 AR decoding 中监控 token 重复率,超阈值自动从 nucleus 切换到 random sampling,一行代码级改动解决 infinite loop; (2) Grouped Code Modeling: 将多帧 codec codes 拼为一组,每组作为一个 AR step,在 AR 层面缩短序列 G 倍且组内 NAR 生成,可直接迁移到任何 codec LM; (3) NAR 模型显式 split acoustic condition: 训练时随机切分 prompt 并使用全 8 层 codes,推理时自然利用完整 prompt 信息提升 speaker similarity
> - **局限**: 仅英文实验; human parity 结论仅基于 LibriSpeech/VCTK 两个 read speech 数据集; 未开源代码/权重; 仍依赖 off-the-shelf EnCodec (75Hz, 8层 RVQ),被 codec 帧率和码本设计约束

## 核心问题

VALL-E 开创了 codec LM TTS 范式,但存在两个结构性痛点 [§1]:

1. **稳定性** — 推理使用 random sampling 导致输出不稳定; nucleus sampling 用小 top-p 虽然更稳定,但会陷入 infinite loop(反复生成相同 silence codes); 五次采样排序可缓解但增加计算成本 [§1, 论文原文]
2. **效率** — AR 模型的帧率被 off-the-shelf codec (EnCodec, 75Hz) 锁定,每秒语音需自回归生成 75 个 token,推理速度慢且无法调整 [§1, 论文原文]

后续工作尝试解决这两个问题,但引入了新的代价 [§1, 论文原文]:
- 对齐增强方案 (ELLA-V, RALL-E): 依赖 forced-alignment model,不可避免引入对齐误差,且增加数据处理复杂度
- 全 NAR 方案 (SoundStorm, NaturalSpeech 2/3, Voicebox): 需要 frame-aligned text-speech 数据,且 pre-determined duration 限制了韵律搜索空间

VALL-E 2 的设计哲学是: 在保持 VALL-E 的简洁性(仅需 utterance-level speech-text pairs)的前提下,用最小侵入性改动解决稳定性和效率两个痛点 [§1, agent 解读]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VALL-E 2 沿用 VALL-E 的层级 AR+NAR 结构,但引入两处关键修改 [§3.2, Fig 2]:

1. **AR Transformer**: 生成第 1 层 codec codes 的 **grouped** sequence,使用 **Repetition Aware Sampling** 解码
2. **NAR Transformer**: 给定第 1 层 codes,非自回归逐层生成第 2-8 层 codes (与 VALL-E 相同)

两者共享同一 Transformer 架构: text embedding layer + code embedding layer + code prediction layer。AR 额外有 group embedding layer 和 group prediction layer; NAR 额外有 code ID embedding layer [§3.2]。

**与 VALL-E 的差异**: AR 和 NAR 模型的核心 Transformer 参数量和架构保持不变,改动集中在 (1) AR 的输入/输出层增加 group 处理; (2) AR 推理时的采样策略; (3) NAR 训练时显式 split acoustic condition [agent 解读]。

### 关键设计选择

#### 1. Grouped Code Modeling — 为什么能同时提速和提质 [§3.1, §3.3.1]

**机制**: 将 codec code sequence 按组大小 G 分组,每组 G 个连续 frames 的 codes 视为一个 AR step 的预测目标 [§3.1, Eq 1-2]:

```
原始序列: c_0, c_1, c_2, ..., c_{T-1}  (T frames, AR step = T)
分组后 (G=2): [c_0, c_1], [c_2, c_3], ..., [c_{T-2}, c_{T-1}]  (T/G groups, AR step = T/G)
```

组间: AR 自回归 — 每个 group 只能 attend to 前面的 groups [§3.3.1, Eq 11]
组内: NAR 非自回归 — 同一 group 内的 codes 并行预测(但采样时仍逐个以计算重复率) [§3.3.1]

**为什么 group embedding**: 不是简单 concat 多帧 embedding,而是用线性映射 W_g 将 G 个 code embedding 映射到一个 group embedding [§3.3.1, Eq 7]。[agent 解读] 这比简单 concat 更紧凑,且让模型学习帧间关系。

**为什么能提质而非只提速**: 论文给出的因果解释是"mitigating the long context modeling problem" [§3.1, 论文原文]。当 G=2 时序列长度减半, Transformer self-attention 能更好地建模全局依赖; 实验验证 G=2 时 WER 和 DNSMOS 均优于 G=1 (Table 1: WER 1.5→1.5 持平但 DNSMOS 3.947→3.966 提升) [§4.2.1]。[agent 解读] 但 G 过大 (G=8) 组内 NAR 预测负担过重,质量下降 (WER 2.5, SIM 0.766) [Table 1]。

**帧率 vs 组大小**: EnCodec 原始帧率 75Hz (13ms/frame),组大小 G 的等效帧率为 75/G Hz [§3.1]:
- G=1: 75Hz (等同 VALL-E)
- G=2: 37.5Hz — 最优 trade-off [Table 1-2]
- G=4: 18.75Hz — 推理 4x 加速,质量可接受
- G=8: 9.375Hz — 推理 8x 加速,质量明显下降

#### 2. Repetition Aware Sampling — 为什么能解决 infinite loop [§3.4.1, Algorithm 1]

**问题根因**: nucleus sampling (top-p) 的小 top-p 值产生高确定性输出,模型倾向重复前一帧的 code → 形成正反馈 loop → infinite repetition [§1, 论文原文]。random sampling 可打破 loop 但引入过多随机性 → 输出不稳定 [§1, 论文原文]。

**Repetition Aware Sampling (RAS) 机制** [Algorithm 1]:
1. 对每个 token ct',先用 **nucleus sampling** (top-p = v) 生成候选 code
2. 计算该 code 在前 K 个 tokens 的窗口内的 **重复率** r = (1/K) * Σ 1_{ct' = ct'-k}
3. 如果 r > tr (阈值), 丢弃 nucleus 结果, 改用 **random sampling** 从完整分布中重采

**超参数**: K=10 (窗口大小), tr=0.1 (阈值,即10个中有超过1个重复就切换), top-p v 从 0.0 到 0.8 搜索 [§4.1.3]

**为什么这个简单策略有效** [agent 解读]: 它利用了 codec codes 的物理特性 — 语音信号的相邻帧在 acoustic 层面本身高度相关,但真正的 infinite loop 会导致异常高的重复率。tr=0.1 的阈值足以区分 "正常帧间相关" 和 "病态 loop"。切换到 random sampling 打破 loop 后,后续帧重新回到正常分布,nucleus sampling 自动恢复。

**效果**: Figure 4 (LibriSpeech) 和 Figure 6 (VCTK) 显示: 无 RAS 时,小 top-p (0-0.3) 导致极高 WER (Figure 4a: GS×1 w/o RAS 在 top-p=0 时 WER ~6%); 有 RAS 后,即使 top-p=0 也能保持低 WER (GS×1 w/ RAS 在 top-p=0 时 WER ~1.6%),且 WER 对 top-p 不敏感 [§4.2.1]。这意味着 RAS 解锁了低 top-p 推理的稳定性,而低 top-p 本身产生更鲁棒的输出 [§4.2.1, 论文原文]。

**计算开销**: 几乎为零 — 仅多一次窗口内重复率计算,运行时远小于模型 forward pass [§3.4.1, 论文原文]。

#### 3. NAR 模型: 显式 split acoustic condition [§3.3.2]

VALL-E 2 在 NAR 训练中做了一个关键改进: 将所有 code sequences 显式分为 acoustic condition C_{<T'} (prompt 部分,使用全 8 层 codes) 和 target C_{>=T'} (生成部分,使用前 j-1 层 codes) [§3.3.2, Eq 15-17]。

**为什么重要**: Table 3 的消融显示,不做显式 split (用 ✦ 标记) 的 SIM 从 0.779 降至 0.731 (3s prefix, single sampling) [Table 3]。[论文原文] 的解释是: 显式 split 让 NAR 模型能从 prompt 的完整 8 层 codes 中提取更多 speaker 信息,而不 split 时 prompt 只被当作 target 的 prefix,仅使用前 j-1 层 [§4.2.3]。

#### 4. Vocos 替换 EnCodec Decoder [§4.1.1]

一个容易忽略的工程改动: VALL-E 2 用预训练 Vocos 模型替换了 EnCodec 自带的 decoder 来生成最终波形 [§4.1.1]。[agent 解读] Vocos 是基于 Fourier 的 neural vocoder,在相同 codec codes 输入下能生成更高质量的语音,这部分提升体现在 DNSMOS 指标中。

### 训练策略

**数据**: Libriheavy corpus — LibriVox 项目的 50K hours 英语有声书数据,约 7000 说话人,带转录标注 [§4.1.1]

**Text tokenization**: BPE (而非 VALL-E 的 phoneme) [§4.1.1]

**训练配置** [§4.1.1]:
- 16 × V100 32GB GPUs
- AdamW optimizer, warmup 32K steps → peak LR → linear decay
- AR model: 4 个变体对应 group size 1/2/4/8,共享同一 NAR model
- NAR model: acoustic condition 长度 T' 随机采样,上限为 min(utterance/2, random(3s, 30s))
- NAR training: 每步随机选 j ∈ [1,7] 计算 loss (Eq 17),而非遍历所有 j

**AR model 的 group size 1**: 即无分组,与 VALL-E baseline 使用相同 NAR model,仅 RAS 不同 [§4.1.1]

## 实验

| 指标 | 本文 (VALL-E 2, G=1) | 本文 (G=2) | Baseline (VALL-E) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER (%, single, 3s prefix) | 1.6 | 1.5 | 3.1 | LibriSpeech test-clean | [Table 1] |
| SIM (single, 3s prefix) | 0.782 | 0.777 | 0.773 | LibriSpeech test-clean | [Table 1] |
| DNSMOS (single, 3s prefix) | 3.987 | 4.000 | 3.985 | LibriSpeech test-clean | [Table 1] |
| WER (%, single, ref utt) | 0.7 | 0.6 | 0.8 | LibriSpeech test-clean | [Table 1] |
| SIM (single, ref utt) | 0.643 | 0.635 | 0.633 | LibriSpeech test-clean | [Table 1] |
| SMOS | 4.61±0.19 | 4.51±0.26 | 4.45±0.28 | LibriSpeech (40 speakers) | [Table 2] |
| CMOS vs GT | +0.033 | -0.167 | -0.268 | LibriSpeech (40 speakers) | [Table 2] |
| GT SMOS | 4.13±0.32 | - | - | LibriSpeech (40 speakers) | [Table 2] |
| WER (%, single, 3s) | 0.9 | 1.5 | 2.4 | VCTK (108 speakers) | [Table 4] |
| SIM (single, 3s) | 0.487 | 0.481 | 0.430 | VCTK | [Table 4] |
| SMOS (3s) | 4.42±0.15 | 4.47±0.13 | 4.32±0.16 | VCTK (60 speakers) | [Table 5] |
| CMOS vs GT (3s) | +0.207 | +0.163 | +0.028 | VCTK (60 speakers) | [Table 5] |
| SMOS (10s) | 3.95±0.10 | 4.26±0.42 | 3.50±0.49 | VCTK (60 speakers) | [Table 5] |

**关键发现**:

1. **Human parity 达成条件**: VALL-E 2 (G=1) 在 LibriSpeech 上三项指标均超过 ground truth — CMOS +0.033 > 0, WER 更低, SMOS 更高 [Table 2]。论文定义 human parity 为这三项指标都优于 GT [§1, 论文原文]。

2. **RAS 的决定性贡献**: VALL-E 2 (G=1, 即只有 RAS 改进) vs VALL-E 在 single sampling 下 WER 从 3.1% 降至 1.6% (LibriSpeech, 3s prefix) [Table 1]; Figure 4 显示 RAS 使 WER 对 top-p 几乎不敏感,这是 human parity 的关键 [§4.2.1]。

3. **Group size 2 是最优 trade-off**: G=2 在大多数指标上与 G=1 持平或更优(DNSMOS 3.966 vs 3.947,WER 1.5 vs 1.6),同时推理速度翻倍 [Table 1]。G=4 质量仍可接受,G=8 明显下降 [Table 1]。

4. **长 prompt 场景 grouped code 优势更明显**: VCTK 10s prompt 下,G=2 的 SMOS 4.26 大幅超过 VALL-E 的 3.50 [Table 5]。[论文原文] 解释: 长 prompt 使 Transformer 面临更长序列的建模困难,grouped code 通过缩短序列缓解了这一问题 [§4.3.1]。

5. **五次采样缩小 RAS 优势**: 五次采样+排序后 VALL-E 和 VALL-E 2 的 WER 差距缩小 (1.0 vs 1.0, Table 1),说明 RAS 主要解决 single sampling 的稳定性,多次采样的 selection 效果可以部分替代 [Table 1]。

6. **训练数据 scaling**: Figure 5 显示 10K hours 已基本饱和,50K hours 仅带来 marginal improvement [§4.2.3, 论文原文]。但低于 10K 时性能明显下降,尤其是 ref utterance prompt 设置 [Figure 5b]。

7. **Prompt 对 AR 模型的双重作用**: 消融 (Table 3) 显示去掉 AR prompt 后 SIM 骤降且 WER 显著上升。[论文原文] 解释: prompt 不仅提供 speaker identity,还约束了 one-to-many TTS 的搜索空间,使生成更稳定 [§4.2.3]。

## 局限性

1. **"Human parity" 的局限条件** — 结论仅在 LibriSpeech/VCTK 两个 read speech benchmark 上成立 [§1, 论文原文明确承认];实际对话、噪声、情感等场景未验证
2. **仅英文** — 训练和评估均限于英语有声书 [§4.1.1]
3. **未开源** — 代码和权重均未释放,可复现性受限
4. **仍依赖 off-the-shelf codec** — EnCodec 75Hz 帧率和 8 层 RVQ 设计不可调,grouped code 只能以整数倍缩减 [§3.1]
5. **Grouped Code 组内 NAR 的局限** — 组内 codes 在预测时无法 attend to 同组已预测的 codes (真正的 NAR),仅依赖组外上下文;当 G 增大时信息不足导致质量下降 [§3.3.1, agent 解读]
6. **Subjective 评估规模有限** — 主观评估仅 20 人,40/60 speakers (LibriSpeech/VCTK),置信区间较宽 [§4.1.2]
7. **隐私和安全** — 高质量零样本语音克隆的伦理风险 [§1, 论文原文]

## 点评

**与前作的关系**: VALL-E 2 是对 VALL-E 的"手术刀式"优化 — 没有重新设计架构(不像 NaturalSpeech 3 转向 diffusion/FACodec),而是在同一 AR+NAR 框架内精准解决两个已知痛点。这种最小侵入性改进的策略值得学习: 当系统的核心范式被验证有效时,优先优化 bottleneck 而非推翻重来。

**RAS 的可迁移性**: Repetition Aware Sampling 是一个优雅的工程解决方案 — 不修改模型结构,不增加训练成本,仅在推理时加一个几乎无代价的条件判断。这种解码层面的改进可以直接应用于任何使用 sampling-based decoding 的 codec LM (包括音乐/音效生成)。

**Grouped Code Modeling 的洞察**: 表面上是序列压缩的效率技巧,但论文用实验证明它通过缓解长上下文建模困难反而提升了质量 (G=2 优于 G=1)。这揭示了 Transformer 在极长 codec 序列上的 attention degradation 问题,也为后续工作 (如 CosyVoice 的 25Hz semantic tokens, FireRedTTS 2 的 12.5Hz streaming tokenizer) 选择低帧率 token 提供了支撑证据。

**"Human parity" 声明的审慎性**: 论文在 §1 明确限定结论范围(仅 LibriSpeech/VCTK),避免了过度泛化。但即便在这两个数据集上,human parity 的定义也值得讨论 — 合成语音的 WER 低于 ground truth 可能反映了 codec LM 的 regularization 效应(过于标准的发音),而非真正"超越人类" [agent 解读]。

**历史定位**: VALL-E 2 在 codec LM TTS 的演进中处于"成熟期"而非"开创期" — 它证明了 VALL-E 范式的 ceiling 很高(可达 human parity),但也暗示了这条路线的天花板: 后续的 CosyVoice/MaskGCT 等转向了 semantic tokens + flow matching 的 hybrid 路线,而非继续在纯 codec LM 上优化。

## 可复用的 idea

1. **Repetition Aware Sampling**: 对任何 AR 生成模型,监控 output token 的局部重复率并动态切换采样策略,是解决 repetitive generation 的零成本方案。可直接迁移到音乐 codec LM (MusicGen)、全双工语音 (Moshi) 等场景。关键超参数: 窗口 K=10, 阈值 tr=0.1。

2. **Grouped Code Modeling**: 对基于高帧率 codec 的 LM,将多帧 codes 分组为一个 AR step 是简单有效的序列压缩方案。组间 AR + 组内 NAR 的混合策略比纯 AR 或纯 NAR 更灵活。Group size 2 是推荐起点。

3. **显式 split acoustic condition in NAR**: NAR 训练时将 prompt 的全部 RVQ layers 暴露给模型 (而非仅前 j-1 层),推理时自然利用完整 prompt 信息。这一改动对 speaker similarity 提升显著 (SIM +0.048, Table 3) 且实现简单。

4. **用 Vocos 替换 EnCodec decoder**: codec codes → waveform 的最后一步,用更好的 vocoder (Vocos) 替换 codec 自带的 decoder,是低成本提升音质的通用策略。

5. **多次采样排序策略 (Eq 23)**: 用 SIM 和 WER 的字典序排序从多次采样中选最优 — SIM > 0.3 时按 WER 排序,否则按 SIM 排序。这一启发式适用于任何零样本 TTS 的 selection/reranking。

> [!review] 自动审阅 (2026-06-08)
> **结论:** pass
> **评分:** 理解 5 | 溯源 5 | 严谨 4 | 导航 5 | 安全 5
> **Claim 标注率:** 95% (38/40)
> **问题:** 0 high, 1 medium, 0 low
> - [medium/template-compliance] frontmatter > models: 仅列 EnCodec,缺少 Vocos 作为对比/组件模型
> 详见 `_review/VALL-E2-review.yml`
