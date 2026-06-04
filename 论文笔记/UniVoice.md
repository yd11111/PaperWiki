---
type: paper
tier: deep
title: "UniVoice: Unifying Autoregressive ASR and Flow-Matching based TTS with Large Language Models"
arxiv_id: "2510.04593"
source: "Sources/UniVoice.pdf"
authors: [Wenhao Guan, Zhikang Niu, Ziyue Jiang, Kaidi Wang, Peijie Chen, Qingyang Hong, Lin Li, Xie Chen]
year: 2025
venue: "arXiv (Work in Progress)"
tags: [TTS, ASR, unified-model, flow-matching, autoregressive, LLM, zero-shot, continuous-representation]
concepts: ["[[Conditional Flow Matching]]", "[[LLM-based TTS]]", "[[Speech Language Model]]", "[[Classifier-Free Guidance]]", "[[Mel Spectrogram]]", "[[Speaker Embedding]]"]
models: ["[[Whisper]]", "[[BigVGAN]]", "[[CosyVoice]]", "[[NaturalSpeech 2]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: [LibriHeavy, LibriSpeech]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Conditional Flow Matching]], [[LLM-based TTS]], [[Zero-shot Speech Synthesis]], [[Speaker Embedding]], [[CosyVoice]], [[Speech Language Model]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: UniVoice 位于 SpeechLM 中"统一模型"赛道,与 SpeechT5、LauraGPT、OpusLM 同类。但不同于 SpeechT5 的 encoder-decoder 架构、LauraGPT 的"continuous-in, discrete-out"设计、OpusLM 的全离散 multi-stream token 方案,UniVoice 选择**全程连续表征**(continuous-in, continuous-out),并在一个 transformer 中同时嵌入 AR (用于 ASR) 和 flow matching (用于 TTS)。
>
> **已有认知**:
> - [[Conditional Flow Matching]] 在 TTS 中已有成熟应用(CosyVoice 系列、F5-TTS、Matcha-TTS),通常作为独立的 second-stage renderer 将 speech tokens 转为 mel spectrogram。[[CLEAR]] 首次将 rectified flow 嵌入 AR LM 的每个 token 位置实现单阶段生成,但仅限 TTS。
> - [[LLM-based TTS]] 的主流架构是 LLM 生成离散 token + NAR/CFM 补充声学细节的两阶段管线(CosyVoice、Seed-TTS)。连续表征路线(MELLE、LatentLM、CLEAR)正在兴起,用 per-token diffusion/flow head 替代离散 codebook。
> - [[Zero-shot Speech Synthesis]] 主流方法是 speech prompt in-context learning 或 speech infilling (Voicebox, F5-TTS),UniVoice 的 text-prefix-conditioned infilling 属于后者。
> - [[Speaker Embedding]] 在 LLM-TTS 时代逐步被 in-context prompt 取代;XLSR-53 是跨语言 speaker encoder 的典型选择。
> - [[CosyVoice]] 的 LLM+OT-CFM 两阶段架构是 UniVoice 的直接对比基准。CosyVoice 用 x-vector 显式分离说话人建模,而 UniVoice-infilling 通过 masked context 隐式获取 speaker identity。
>
> **创新判断**: UniVoice 的核心创新点 --- 在单个 LLM 中通过 dual attention mask 同时训练 AR-ASR 和 FM-TTS --- 在已有知识库中没有先例。CLEAR 实现了 AR+flow 单阶段 TTS,但未涉及 ASR。VioLA/LauraGPT 做了统一但用离散 token。UniVoice 是首个在连续空间同时统一 ASR 和 TTS 的 LLM 框架。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[LLM-based TTS]]✓, [[Zero-shot Speech Synthesis]]✓, [[Speaker Embedding]]✓, [[CosyVoice]]✓, [[Speech Language Model]]✓ | 过滤: [[Classifier-Free Guidance]](待确认), [[Mel Spectrogram]](待确认), [[Codec Language Model]](待确认) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个在单个 LLM 中通过 dual attention mask 统一 AR-ASR 和 flow-matching-TTS 的连续表征框架,在两个任务上均达到或超过现有统一方法
> - **路线**: speech → Whisper encoder + adapter → LLM (causal mask for ASR / bidirectional mask for TTS) → text tokens (ASR) 或 mel spectrogram via flow matching (TTS) → BigvGAN vocoder
> - **指标**: ASR WER 3.0/6.3 (LibriSpeech clean/other); TTS SIM 0.56, WER 4.06, UTMOS 3.72 (LibriSpeech-PC) [Table 1]
> - **可借鉴**: dual attention mask 机制(causal vs bidirectional 在同一 transformer 中按任务切换),适用于任何需要同时建模 AR 和 non-AR 目标的多任务框架; text-prefix-conditioned infilling 替代显式 speaker embedding 的零样本方案
> - **局限**: 仅在 LibriHeavy 50K 小时数据+SmolLM2-360M 上验证,未扩展到大模型/大数据; TTS 质量(SIM 0.56, UTMOS 3.72)显著弱于专用系统(CosyVoice SIM 0.66, F5-TTS UTMOS 3.84); 仅限 ASR+TTS 两任务,未涉及对话/编辑等; 代码已开源但为 work in progress

## 核心问题

UniVoice 试图解决的核心问题是: **如何在一个统一的 LLM 框架内同时完成语音识别和语音合成,且避免离散 token 的信息损失?**

现有统一方法(VioLA、LauraGPT、OpusLM)依赖离散 speech token,量化不可避免地丢失声学细节,限制了 TTS 生成质量和 ASR 识别精度。而 AR 建模(适合 ASR 的序列预测)和 flow matching(适合 TTS 的高保真生成)在注意力机制上存在根本冲突: AR 需要 causal mask 防止信息泄漏,FM 需要 bidirectional attention 获取完整上下文 [§Introduction]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UniVoice 是一个 dual-branch hybrid 架构 [Fig 1],在一个预训练 LLM 中同时运行两条路径:

1. **ASR 路径**: speech → Whisper encoder → adapter → LLM (causal attention) → text tokens
2. **TTS 路径**: text tokens + noisy speech + masked speech → LLM (bidirectional attention) → flow matching velocity field → mel spectrogram → BigvGAN → waveform

两条路径共享同一个 LLM backbone (SmolLM2-360M),通过**切换 attention mask**实现任务区分 [§Model]。

### 关键设计选择

**1. Dual Attention Mask 机制**

这是本文的核心设计。为什么 ASR 用 causal mask 而 TTS 用 bidirectional mask? [论文原文] AR 建模的本质是序列预测,需要 causal mask 防止未来信息泄漏;而 flow matching 是非自回归的,在每个 time step 需要看到完整上下文才能准确估计速度场 [§Attention Mask Design]。消融实验验证: 如果 TTS 也用 AR mask,WER 从 4.66 涨到 9.85, SIM 从 0.56 降到 0.49, UTMOS 从 3.92 降到 2.23 [Table 4]。[agent 解读] 这说明 flow matching 对全局上下文的需求是刚性的,不能用因果限制来妥协。

**2. 连续表征 vs 离散 token**

[论文原文] 作者认为离散 speech token 的量化不可避免地丢失"感知上关键的声学细节",驱动他们探索连续表征替代方案 [§Introduction]。[agent 解读] 这一选择意味着 TTS 端不需要 codec tokenizer (如 EnCodec/SoundStream),而是直接在 mel spectrogram 空间操作。代价是失去了离散 token 带来的 sequence modeling 简便性。

**3. Text-prefix-conditioned Speech Infilling**

TTS 推理时不用显式 speaker embedding,而是用 text-prefix guided speech infilling [§Model, Fig 2b]: 将参考语音的 mel spectrogram 作为未被 mask 的上下文,目标语音位置被 mask 覆盖,模型通过 flow matching 预测被 mask 部分。这与 Voicebox、F5-TTS 的 speech infilling 范式一致,但在统一模型中实现。[论文原文] 作者还对比了 UniVoice-TTS-speaker 变体(用 XLSR-53 提取 global speaker embedding),infilling 方案在所有指标上全面优于 speaker embedding 方案: WER 4.66 vs 5.72, SIM 0.56 vs 0.29, UTMOS 3.92 vs 3.65 [Table 2]。[agent 解读] speaker embedding 只捕获 global timbre,而 infilling 的 masked context 保留了参考语音的完整细节(包括韵律、环境),因此效果更好。

**4. 为什么用 SmolLM2-360M 而非更大 LLM?**

[论文原文] 作者未显式解释,但在 limitation 中指出"当前系统基于相对较小的数据集和小语言模型训练,性能仍有提升空间" [§Limitation]。[agent 解读] 这可能是出于计算成本考虑的初步验证选择,360M 参数在统一模型中已经可以产生有意义的结果(OpusLM-0.4B 是直接对比)。

### 训练策略

**联合训练目标**:
$$L_{total} = \lambda L_{LM}(\theta) + L^{cf,m}_{audio}(\theta)$$

其中 $L_{LM}$ 是 ASR 的自回归 cross-entropy loss,$L^{cf,m}_{audio}$ 是 TTS 的 masked OT-CFM loss [Eq. 7]。

**关键超参数**:
- $\lambda = 0.005$ (ASR loss 权重极小) [§Experimental Setups]
- [论文原文] 作者解释: 在相同 transformer backbone 下,TTS 基于 flow matching 训练更困难,因此给予更高权重;ASR 任务相对简单,设置较小权重让模型优先学习更难的 TTS 任务 [Appendix B]
- 消融实验: $\lambda = 0.01$ 时 ASR WER-clean 从 3.01 涨到 4.21, TTS WER 从 4.06 涨到 4.66 [Table 3]

**CFG 训练**: 随机以 0.2 概率 drop text tokens, 以 0.3 概率 drop masked speech [§Experimental Setups]。推理时 CFG weight = 2, NFE = 32。

**Mask 策略**: 训练时随机 mask 70%-100% 的 mel frames [§Experimental Setups]。

## 实验

| 指标 | 本文 (UniVoice) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ASR WER-clean | 3.0% | SpeechT5 4.4%, LauraGPT 4.4%, OpusLM-0.4B 4.2%, Whisper-small 3.4% | LibriSpeech test-clean | [Table 1] |
| ASR WER-other | 6.3% | SpeechT5 10.4%, LauraGPT 7.7%, OpusLM-0.4B 8.7%, Whisper-small 7.6% | LibriSpeech test-other | [Table 1] |
| TTS SIM | 0.56 | SpeechT5 0.33, CosyVoice 0.66, F5-TTS 0.66, MaskGCT 0.66 | LibriSpeech-PC | [Table 1] |
| TTS WER | 4.06% | SpeechT5 5.91%, CosyVoice 3.59%, F5-TTS 2.54%, MaskGCT 2.49% | LibriSpeech-PC | [Table 1] |
| TTS UTMOS | 3.72 | SpeechT5 3.32, CosyVoice 4.17, F5-TTS 3.84, MaskGCT 3.85 | LibriSpeech-PC | [Table 1] |
| TTS CMOS | +0.00 | SpeechT5 -0.28, CosyVoice +0.06, GT +0.09 | LibriSpeech-PC | [Table 1] |
| TTS SMOS | 3.88 | SpeechT5 3.35, CosyVoice 3.96, GT 3.82 | LibriSpeech-PC | [Table 1] |

**统一模型对比**:
- UniVoice 在所有统一模型(SpeechT5、LauraGPT、OpusLM)中 ASR 和 TTS 均为最优 [Table 1]
- ASR: UniVoice (0.4B, 50K h) WER-clean 3.0% 优于 OpusLM-0.4B (4.2%, 213K h),接近 OpusLM-7B (2.3%, 213K h) [Table 1]
- TTS: UniVoice WER 4.06% 相比 SpeechT5 最优统一基线(5.91%)有 31% 相对改进 [Table 1]

**与专用系统的差距**:
- ASR: 接近但未超过 Whisper-large-v3 (1.9/3.5)、Zipformer (2.0/4.4) [Table 1]
- TTS: SIM 0.56 显著低于 CosyVoice/F5-TTS/MaskGCT 的 0.66; UTMOS 3.72 低于 CosyVoice 4.17 [Table 1]

**多任务互作用**:
- 联合训练让 TTS WER 从 4.66 (UniVoice-TTS) 降到 4.06 (UniVoice), 改善 13% [Table 1]; [论文原文] 作者认为联合 ASR-TTS 训练通过共享语言表征增强了语音可懂度 [§Main Results]
- 但联合训练也让 ASR WER-clean 从 2.5 (UniVoice-ASR) 涨到 3.0, TTS UTMOS 从 3.92 (UniVoice-TTS) 降到 3.72 [Table 1]; [论文原文] 说明统一架构中存在"多任务泛化与单任务优化之间的固有张力" [§Main Results]

## 局限性

1. **模型和数据规模小**: SmolLM2-360M + LibriHeavy 50K 小时,远小于 CosyVoice (170K h) / F5-TTS (100K h) / Whisper (680K h) [Table 1]。作者承认性能有提升空间 [§Limitation]
2. **TTS 质量与专用系统差距明显**: SIM 0.56 vs 专用系统 0.66,UTMOS 3.72 vs 4.17,说明参数共享对 TTS 声学质量有明显代价 [Table 1]
3. **任务范围有限**: 仅限 ASR+TTS,未涉及语音编辑、对话、翻译等任务。作者明确排除了语音对话应用 [§Model]
4. **未利用 LLM 对话能力**: LLM backbone 的原始对话/推理能力未被激活 [§Limitation]
5. **Infilling 方案的 duration 估计粗糙**: 依赖 F5-TTS 的 duration ratio 启发式,非学习型 duration predictor [Eq. 8]
6. **评估局限**: 仅在英文 LibriSpeech/LibriHeavy 上评估,无多语言、无噪声环境测试

## 点评

UniVoice 的核心贡献在于提出了一个清晰的架构方案来解决 AR-ASR 和 FM-TTS 在注意力机制上的根本冲突。dual attention mask 的设计直觉简洁,消融实验 [Table 4] 有力地验证了这一选择的必要性。联合训练让 TTS 鲁棒性(WER)受益于 ASR 的语言建模能力,这是多任务学习的理想表现。

但论文的实验说服力受限于规模: 0.4B 模型 + 50K 小时数据在 2025 年的 TTS 语境中偏小,无法判断方法的 scaling 能力。与专用系统(CosyVoice, F5-TTS)的差距相当大(SIM 差 0.10, UTMOS 差 0.45),部分原因可能是数据量差异(50K vs 100K-170K),部分是架构妥协的固有代价。论文坦诚地讨论了这些 trade-off,但缺少控制变量(如同等数据量下 UniVoice vs 单任务 F5-TTS)来分离数据因素和架构因素的影响。

从 unified SpeechLM 的角度看,UniVoice 在连续表征空间的统一比 VioLA/LauraGPT 的离散方案更优雅,也比 OpusLM 需要的 213K 小时数据更高效(0.4B 规模下 ASR WER-clean 3.0 vs 4.2)。但相比 CLEAR(单阶段 AR+flow TTS,RTF 0.18),UniVoice 未报告推理效率指标,无法判断统一架构是否引入了额外的推理开销。

## 可复用的 idea

1. **Dual attention mask**: 在同一个 transformer 中按任务切换 causal/bidirectional mask,是一种通用的多任务设计模式。适用于任何需要同时建模自回归序列预测和全局上下文生成的场景(如 ASR+TTS、text generation+image generation)
2. **极小 $\lambda$ 的联合训练**: ASR loss 权重设为 0.005,说明"容易任务加小权重"可以让模型优先学难任务,同时不损害简单任务。这是多任务学习中的实用经验
3. **Speech infilling 替代 speaker embedding**: masked context 的信息量远大于 global embedding,在不增加额外模块的前提下获得更好的 zero-shot cloning 效果 [Table 2]
4. **联合 ASR-TTS 训练增强 TTS 鲁棒性**: 共享语言建模能力让 TTS WER 改善 13%,暗示 ASR 数据可以作为 TTS 训练的免费正则化

> [!review] 审阅状态
> 待审阅 --- 审阅报告将在下方步骤生成
