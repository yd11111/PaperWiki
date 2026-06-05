---
type: paper
tier: deep
title: "CTC-TTS: LLM-based dual-streaming TTS with CTC alignment"
arxiv_id: "2602.19574"
source: "Sources/CTC-TTS.pdf"
authors: [Hanwen Liu, Saierdaer Yusuyin, Hao Huang, Zhijian Ou]
year: 2026
venue: "INTERSPEECH 2026 (submitted)"
tags: [TTS, streaming, CTC, alignment, LLM-based, dual-streaming, interleaving, zero-shot]
concepts: ["[[Speech-TextAlignment]]", "[[LLM-basedTTS]]", "[[CodecLanguageModel]]", "[[SpeechTokenizer]]", "[[PhonemeRepresentation]]", "[[Single-codebookvsMulti-codebook]]"]
models: ["[[论文笔记/LLMVoX|LLMVoX]]", "[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[数据集/SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: LLM-basedTTS✓, SpeechTokenizer✓, CosyVoice2✓, Speech-TextAlignment, CodecLanguageModel, PhonemeRepresentation)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]], [[SpeechTokenizer]], [[CosyVoice2]], [[Speech-TextAlignment]], [[CodecLanguageModel]], [[PhonemeRepresentation]] | 过滤: Speech-TextAlignment(pending-review), CodecLanguageModel(pending-review), PhonemeRepresentation(pending-review), Single-codebookvsMulti-codebook(pending-review) | 未命中但可能相关: 无

**谱系定位**: CTC-TTS 属于 LLM-based TTS 范式下的 **dual-streaming 子方向**。在 KB 中,LLM-based TTS 的主线演进从 VALL-E (离线 AR) 到 CosyVoice 2 (streaming, fixed-ratio interleaving),再到 ELLA-V (alignment-guided interleaving)。CTC-TTS 的位置在 ELLA-V 之后、LLMVoX 之上: 同样做 alignment-aware interleaving,但用 CTC 替换 MFA 获取对齐,用 bi-word 替换 ELLA-V 的 local advance 构造训练序列。

**已有认知**:
- Speech-Text Alignment 页 [待确认] 主要聚焦 SpeechLM 中的跨模态表征对齐(如 SPIRIT-LM 的交替训练),与本文的 forced-alignment 层面(phoneme-speech 时间对齐)是不同层次的"对齐"。本文的 alignment 更接近传统 TTS 中的 duration modeling。
- Single-codebook vs Multi-codebook 页记录了 WavTokenizer 是典型的单码本方案(K=4096, 75 tokens/s),CTC-TTS 正是基于此做单流 AR 建模。
- CosyVoice 2 采用 fixed-ratio interleaving (非对齐感知),是本文的间接对比目标; StreamMel 也属于固定比例交错方案。
- LLMVoX 采用 feature-dimension stacking + fixed-ratio interleaving,是本文单说话人实验的直接 baseline。

**创新判断**: 本文的核心创新在两个层面: (1) 用轻量级 CTC aligner 替换 MFA pipeline,降低对齐工具的复杂度; (2) 设计 bi-word 交错单元并提供两种实现(长度拼接 vs 特征堆叠),使同一对齐方案覆盖不同质量-延迟需求。与已有 KB 中记录的 ELLA-V (MFA + local/global advance) 相比,CTC-TTS 的 bi-word 策略本质上是一种 compact look-ahead 设计。

## 速查

> [!summary] 速查
> - **一句话**: 用 CTC aligner 替代 MFA 做 phoneme-speech 对齐,配合 bi-word 交错策略构建 LLM-based 双流式 TTS,同时提供质量优先(-L)和延迟优先(-F)两种变体
> - **路线**: Text → G2P (Phonetisaurus, IPA) → CTC aligner (Whistle, 25fps) → phoneme-speech alignment → bi-word interleaving (L: length concat / F: feature stack) → Decoder-only Transformer (AR) → WavTokenizer (single-codebook, 75 tokens/s) → chunk-aware streaming output
> - **指标**: 单说话人 CTC-TTS-L WER 1.50% / CER 0.79% / UTMOS 4.15 / FPL-A 210ms (vs LLMVoX WER 2.40% / FPL-A 167ms) [Table 1]; 多说话人 continuation CTC-TTS-L WER 4.82% / MOS 4.33 / SMOS 4.60 (vs MFA+ELLA-V WER 10.98% / MOS 3.94) [Table 2]; cross-speaker CTC-TTS-L WER 6.33% / MOS 4.23 [Table 3]
> - **可借鉴**: (1) bi-word block 设计 — 当前词 phonemes + separator + 下一词 phonemes + 当前词 speech tokens,提供 compact look-ahead 上下文; (2) CTC blank 分配策略 — 将 blank 归并到后续 phoneme,再按 1:3 帧率比映射到 speech tokens; (3) 两种 interleaving 变体覆盖不同部署需求
> - **局限**: 仅在英文单语场景验证(LibriSpeech / VoiceAssistant400K); 模型规模小(~160M); 未与 CosyVoice 2 等工业系统直接对比; 代码未开源(待 accept); G2P 依赖 WFST-based Phonetisaurus,未探索 neural G2P

## 核心问题

**已有方法的问题**: LLM-based TTS 做 dual-streaming 合成时面临两个关键设计决策 — (1) 如何获取 text-speech 对齐?(2) 如何组织训练序列(interleaving scheme)?现有方法要么用 fixed-ratio interleaving(CosyVoice 2, IST-LM, StreamMel)导致模型难以学习真实的时间依赖关系,要么依赖 MFA 这种 GMM-HMM pipeline 做 forced alignment(ELLA-V, SpeakStream)带来工具链复杂度。[§1]

**本文方案**: 用 CTC-based ASR 模型(Whistle)替换 MFA 获取 phoneme-speech 对齐,结合 bi-word 交错策略构造训练序列。CTC 对齐不追求帧级精确的 phoneme 边界,而是提供足够 LM 学习 phoneme-to-speech 映射的结构性对应关系。[§1, §3.1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CTC-TTS 由五个组件构成 [Fig 2]:
1. **G2P 模块** (Phonetisaurus, frozen): 英文文本 → IPA phonemes
2. **CTC ASR 模型** (Whistle, 115M, frozen): 语音 → phoneme-level 后验概率 → Viterbi 解码获取最优对齐路径
3. **Alignment-and-Interleaving 模块**: 基于 CTC 对齐构造 bi-word 交错训练序列
4. **Decoder-only Transformer** (trainable): 自回归预测 speech tokens + `<eob>` 符号
5. **Neural Audio Codec** (WavTokenizer, frozen): single-codebook, 75 tokens/s, 编解码

训练时仅 LM 和 phoneme embedding 可训练,其余模块冻结。[§3.3, Fig 2]

### 关键设计选择

#### 1. CTC 对齐替代 MFA

CTC ASR 模型在每帧输出 phoneme 后验概率,通过 Viterbi 算法找到最大概率对齐路径 π* [§3.1, Eq. 1]:

π* = argmax_π ∏_{t=1}^{T} P(π_t | x_t)

CTC 对齐的 blank 符号处理: 每个 blank 归并到其后续第一个 phoneme 标签。路径末尾的 blanks 归并到最后一个 phoneme。[§3.1]

**为什么 CTC 对齐足够?** [论文原文] CTC 对齐的 phoneme 标签通常滞后于实际发声起始,blank 可能对应静音或相邻 phoneme 区域 [§3.1, ref 17]。但这种"粗糙"对齐对 AR 模型学习 local phoneme groups → speech tokens 的映射已经足够,因为 LM 本质上是在对齐提供的结构性约束下学习这个映射,不需要帧级精确边界。[§3.1]

**帧率匹配**: CTC 模型 25fps, WavTokenizer 75fps,比率 1:3。因此每个 phoneme π̂_i 映射到 3 个 speech tokens [s_{3i-2}, s_{3i-1}, s_{3i}]。[§3.1]

**与 MFA 的关键区别**: MFA 是 GMM-HMM pipeline(需要预训练声学模型+发音词典+解码器),而 CTC alignment 只需要一个预训练 CTC ASR 模型 + Viterbi 解码,流程更轻量。[论文原文, §1]

#### 2. Bi-word 交错策略

对齐得到 word-level phoneme-speech 对应关系后,构造 bi-word block b_k [§3.2, Fig 1]:

```
b_k = [当前词 phonemes] + [separator] + [下一词 phonemes] + [当前词 speech tokens] + <eob>
```

其中 separator 包括空格、逗号、句号、问号、感叹号。最后一个词用 `<eos>` 占位下一词。完整训练序列 Y = b_1 ⊕ b_2 ⊕ ... [§3.2]

**为什么用 bi-word 而不是 uni-word?** [agent 解读] 包含下一个词的 phonemes 提供了 compact look-ahead 上下文。语音合成中,当前词的发音(尤其是 coarticulation 和韵律)受下一个词的影响。bi-word 设计以最小的 look-ahead 代价(仅一个词)捕获这种上下文依赖,平衡了质量和延迟。

#### 3. 两种 Interleaving 实现

**CTC-TTS-L (Length-wise concatenation)** [§3.2, Fig 1a]:
- Phoneme tokens 和 speech tokens 沿序列长度维度拼接
- 模型先读完 bi-word 的所有 phonemes,再生成 speech tokens 直到发出 `<eob>`
- 优势: phoneme 上下文完整,生成质量更高
- 代价: 需要读完前两个词的 phonemes 才能开始生成 → FPL 较高

**CTC-TTS-F (Feature-dimension stacking)** [§3.2, Fig 1b]:
- Phoneme embedding 和 speech embedding 沿特征维度堆叠(phoneme 256d + speech 768d)
- Phoneme 序列用 `<pad>` 补齐到与 speech 序列等长后逐帧堆叠
- 第一帧: 第一个 phoneme + 全零张量 → 即可开始生成
- 优势: 从第一个 phoneme 就能开始合成 → FPL 最低
- 代价: 每步只能看到局部 phoneme 信息,生成质量略低

[agent 解读] CTC-TTS-F 的设计借鉴了 LLMVoX 的 feature stacking 思路,但 LLMVoX 用 fixed-ratio interleaving,而 CTC-TTS-F 在此基础上加入了 CTC alignment + bi-word 结构。

### 训练策略

- 训练目标: 标准交叉熵损失,仅在 speech token 和 `<eob>` 位置计算,跳过 text token 位置 [§3.3, Eq. 2]
- 单说话人: 4 层 decoder, 768d, 12 heads, 1M steps, 4x RTX 3090 [§4.2]
- 多说话人: 12 层 decoder, 1024d, 16 heads, 320K steps [§4.2]
- 优化器: AdamW (β1=0.9, β2=0.95), cosine schedule, warmup to 3e-4 in 25K steps, weight decay 0.1 [§4.2]
- 推理: flash-attention + KV-Cache [§4.2]; 单说话人用 greedy search, 多说话人用 nucleus sampling [§5.1, §5.2]

零样本场景: 训练序列前缀包含预对齐的 prompt speech + text tokens (prompt tokens),推理时复用。[§3.3, Fig 2]

## 实验

### 单说话人流式实验 (vs LLMVoX)

| 指标 | LLMVoX | CTC-TTS-F | CTC-TTS-L | GT | 出处 |
| --- | --- | --- | --- | --- | --- |
| #Params | 31.5M | 33.6M | 34.7M | — | [Table 1] |
| WER% ↓ | 2.40 | 1.80 | **1.50** | — | [Table 1] |
| CER% ↓ | 1.36 | 1.04 | **0.79** | — | [Table 1] |
| FPL-A (ms) ↓ | 167 | **159** | 210 | — | [Table 1] |
| UTMOS ↑ | 4.15 | 4.15 | 4.15 | 4.27 | [Table 1] |

数据集: VoiceAssistant400K (1750h train / 50h val / 5h test) [§4.1]

### 多说话人零样本 — Continuation task

| 指标 | CTC-TTS-F | CTC-TTS-L | CTC+ELLA-V | MFA+ELLA-V | MFA+bi-word | GT | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER% ↓ | 5.20 | **4.82** | 12.01 | 10.98 | 5.14 | 1.92 | [Table 2] |
| CER% ↓ | 2.68 | **2.47** | 7.37 | 6.99 | 2.63 | 0.69 | [Table 2] |
| SPK ↑ | 0.930 | 0.929 | 0.928 | 0.928 | **0.930** | — | [Table 2] |
| UTMOS ↑ | 4.013 | **4.050** | 4.021 | 4.021 | 4.010 | 4.086 | [Table 2] |
| MOS ↑ | 4.31 | **4.33** | 4.00 | 3.94 | 4.25 | 4.28 | [Table 2] |
| SMOS ↑ | 4.58 | **4.60** | 4.39 | 4.44 | 4.50 | 4.60 | [Table 2] |

数据集: LibriSpeech 960h train, test-clean (4-10s, continuation) [§4.1]

### 多说话人零样本 — Cross-speaker task

| 指标 | CTC-TTS-F | CTC-TTS-L | CTC+ELLA-V | MFA+ELLA-V | MFA+bi-word | GT | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER% ↓ | 8.02 | **6.33** | 20.86 | 34.89 | 7.53 | — | [Table 3] |
| CER% ↓ | 4.20 | **3.21** | 11.73 | 19.58 | 3.99 | — | [Table 3] |
| SPK ↑ | **0.880** | 0.878 | 0.869 | 0.872 | 0.874 | — | [Table 3] |
| UTMOS ↑ | 3.903 | **3.971** | 3.848 | 3.873 | 3.840 | 3.527 | [Table 3] |
| MOS ↑ | 4.16 | **4.23** | 3.88 | 3.75 | 4.14 | 4.18 | [Table 3] |
| SMOS ↑ | 3.85 | **3.98** | 3.94 | 3.88 | 3.83 | 4.14 | [Table 3] |

数据集: Seed-TTS test-en (cross-speaker) [§4.1]

### 消融分析关键发现

1. **CTC alignment > MFA alignment 在 cross-speaker (OOD) 场景**: CTC+ELLA-V WER 20.86% vs MFA+ELLA-V 34.89% [Table 3],作者指出 CTC 对齐在 out-of-domain 场景泛化更好 [§5.2]
2. **bi-word > ELLA-V 的 local advance**: 在两种对齐方式下,bi-word 序列都大幅优于 ELLA-V 序列 (CTC-TTS-L WER 4.82% vs CTC+ELLA-V 12.01%) [Table 2],作者归因于 ELLA-V local advance 机制提供的上下文有限 [§5.2]
3. **MFA 在 in-domain continuation 上有优势**: MFA+ELLA-V continuation WER 10.98 优于 CTC+ELLA-V 12.01,但 cross-speaker 场景反转 [§5.2]
4. **-L vs -F 的 trade-off**: CTC-TTS-L 在所有质量指标上优于 CTC-TTS-F,但 FPL-A 更高 (210ms vs 159ms) [Table 1]

## 局限性

1. **语言覆盖**: 仅验证英文,未探索中文或多语言场景。G2P 使用 WFST-based Phonetisaurus,跨语言扩展性有限 [§6]
2. **模型规模**: ~160M 参数,远小于工业系统 (CosyVoice 2 ~500M+),无法判断 scaling 效果
3. **未与工业系统对比**: 直接 baseline 仅为 LLMVoX (31.5M) 和 ELLA-V 复现,缺少与 CosyVoice 2、Seed-TTS 等系统的对比
4. **CTC aligner 的局限**: 依赖预训练 CTC ASR 模型 (Whistle 115M, LibriSpeech-trained),在低资源语言或噪声环境下质量未知
5. **代码未开源**: 论文表示代码将在 accept 后发布 [§6]
6. **无主观 MOS 对 FPL 的综合评估**: 单说话人实验无主观评测; FPL-A 假设全文本可用,未报告真实流式场景延迟

## 点评

CTC-TTS 的核心贡献是 **将 CTC forced alignment 引入 LLM-based streaming TTS 的序列构造**,这是一个工程上有意义的改进: MFA 需要 GMM-HMM pipeline + Kaldi 工具链,而 CTC alignment 只需要一个现成的 CTC ASR 模型 + Viterbi 解码,降低了工具依赖。

bi-word interleaving 策略的设计直觉清晰: 一个词的 look-ahead 足以提供 coarticulation 和韵律上下文,同时保持 latency 可控。消融实验 (bi-word vs ELLA-V local advance) 充分验证了这一设计的优势。

**方法论层面的关切**:
- 模型规模偏小,在这个参数量级上的结论能否迁移到工业级系统 (500M+) 存疑
- CTC 对齐本质上是一种"软对齐" — 论文明确承认 phoneme labels 滞后实际发声 [§3.1],这种系统性偏差对合成质量的影响值得更深入分析
- cross-speaker 实验使用 Seed-TTS test-en,但论文的 UTMOS 竟然超过 Ground Truth (3.971 vs 3.527),这可能反映 UTMOS 对 LibriSpeech 训练数据的偏好而非真实质量优势

## 可复用的 idea

1. **CTC blank 归并策略**: 将 CTC 对齐路径中的 blank 归并到后续 phoneme,再按固定帧率比(1:3)映射到 speech tokens — 这个方案可用于任何需要 phoneme-speech 帧级对应的场景(如 duration predictor 的训练标签构造)
2. **bi-word interleaving 作为 compact look-ahead**: 仅一个词的 look-ahead 就能有效提升合成质量,这一发现对设计其他 streaming pipeline 的 buffer 策略有参考价值
3. **L vs F 两种变体的设计模式**: 同一对齐方案通过不同的序列组织方式(length concat vs feature stack)实现质量-延迟 trade-off,这种"一种对齐,两种实现"的思路可复用

---

检索命中: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[CosyVoice2]]✓, [[Speech-TextAlignment]](pending-review), [[CodecLanguageModel]](pending-review), [[PhonemeRepresentation]](pending-review) | 未命中但可能相关: 无
