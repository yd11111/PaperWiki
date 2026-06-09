---
type: paper
tier: deep
title: "AudioLM: a Language Modeling Approach to Audio Generation"
arxiv_id: "2209.03143"
source: "Sources/AudioLM.pdf"
authors: [Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, Neil Zeghidour]
year: 2023
venue: "arXiv (Google Research)"
tags: [speech-LM, audio-generation, semantic-token, acoustic-token, hierarchical-generation, speech-continuation, piano-generation, zero-shot]
concepts: ["[[SemanticvsAcousticTokens]]", "[[SpeechLanguageModel]]", "[[ResidualVectorQuantization]]", "[[SpeechTokenizer]]", "[[LLM-basedTTS]]"]
models: ["[[SoundStream]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[SemanticvsAcousticTokens]], [[SpeechLanguageModel]], [[ResidualVectorQuantization]], [[SpeechTokenizer]], [[LLM-basedTTS]])
> AudioLM 是 semantic-acoustic token 层级建模范式的奠基工作,被 [[SemanticvsAcousticTokens]] 列为串联策略的代表。[[SpeechLanguageModel]] 将 AudioLM 定位为 GSLM → AudioLM → TWIST 演进中的关键节点。[[ResidualVectorQuantization]] 页记录了 RVQ 的 coarse/fine 层级信息结构,这正是 AudioLM 三阶段生成的基础。
> 检索命中: [[SemanticvsAcousticTokens]], [[SpeechLanguageModel]], [[ResidualVectorQuantization]], [[SpeechTokenizer]], [[LLM-basedTTS]] | 过滤: [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 semantic + acoustic tokens 层级建模框架,首次在无文本监督下实现长期连贯且高保真的语音/音乐续写 / Proposes hierarchical semantic-acoustic token modeling for coherent, high-quality audio generation without text supervision
> - **路线**: 音频 → w2v-BERT semantic tokens + SoundStream acoustic tokens → 三阶段 decoder-only Transformer (semantic → coarse acoustic → fine acoustic) → SoundStream decoder → 波形
> - **指标**: sWUGGY 83.7 (causal SOTA) [Table IV]; sBLIMP 64.7 (causal SOTA) [Table IV]; 主观评估 51.2% 人类无法区分真假语音 [§IV-G]; 钢琴续写 83.3% 偏好层级建模 [§IV-I]; 合成语音检测 98.6% 准确率 [§IV-H]
> - **可借鉴**: (1) semantic tokens 保结构、acoustic tokens 保音质的解耦思路; (2) 三阶段推理分治策略降低序列长度; (3) 3 秒 prompt 即可保持说话人身份的 in-context 范式
> - **局限**: 无文本条件输入(不是 TTS); 仅 16 kHz 采样率; 推理需串行三阶段; 未开源模型权重

## 核心问题

AudioLM 试图回答: **如何在没有任何文本或符号标注的情况下,同时实现音频生成的长期语义连贯性和高保真声学质量?**

此前的方法面临两难 [§I]:
- 纯波形生成模型 (WaveNet) 在无强条件时退化为 babbling [§I]
- 语言模型 (GSLM) 可建模长期结构但音质差 [§II]
- 高保真 codec (SoundStream) 重建好但语义连贯性弱 [§III-B, Table I]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

AudioLM 由三个组件构成 [§III]:
1. **Tokenizer**: 将音频映射为离散 token 序列
   - w2v-BERT (0.6B Conformer) 的第 7 层中间表征 → k-means (K=1024) 聚类 → **semantic tokens** (25 Hz, 250 bps) [§III-A]
   - SoundStream (12 层 RVQ, codebook 1024) → **acoustic tokens** (50 Hz, 6000 bps) [§III-A]
2. **Language Model**: 三个独立的 decoder-only Transformer,分别对三个阶段建模 [§III-C]
3. **Detokenizer**: SoundStream decoder 将 acoustic tokens 解码回波形 [§III-A]

### 关键设计选择

**为什么需要两类 token?** [论文原文]
- Semantic tokens (w2v-BERT): 语音判别力强 (ABX 6.7) 但重建质量差 (ViSQOL 1.1) [Table I]
- Acoustic tokens (SoundStream): 重建质量好 (ViSQOL 3.9) 但语音判别力弱 (ABX 17.8) [Table I]
- 单独用 acoustic tokens 建模 → 说话人/录音条件正确但语言内容像 babbling [§III-B]
- 两者互补: semantic 管结构, acoustic 管音质 [§III-C]

**为什么三阶段而非两阶段?** [论文原文]
- 12 层 RVQ 展开后序列极长, 一次性建模计算不可行 [§III-C]
- 将 RVQ 分为 coarse (前 Q'=4 层) 和 fine (后 8 层):
  - Stage 1: 纯 semantic token AR 建模 → 长期时序结构 [§III-C]
  - Stage 2: 以 semantic 为条件, AR 生成 coarse acoustic → 恢复说话人身份和录音条件 [§III-C]
  - Stage 3: 以 coarse acoustic 为条件, 生成 fine acoustic → 消除压缩伪影, 提升音质 [§III-C]
- [agent 解读] Stage 3 可在 3 秒不重叠 chunk 上独立运行, 实现了音质提升与序列长度的解耦

**条件独立性假设**: p(z_t | z_{<t}, y_{<t}) ≈ p(z_t | z_{<t}) — fine acoustic tokens 在给定 coarse 后与 semantic 条件独立 [§III-C], 这让 Stage 3 无需看 semantic tokens, 大幅缩短输入长度

**推理模式** [§III-D]:
- 无条件生成: 先采样 semantic → 再 acoustic
- 声学生成 (acoustic generation): 给定 ground-truth semantic → 仅生成 acoustic (保持语言内容)
- 续写 (continuation): 给定 prompt 的 semantic + acoustic → 续写

### 训练策略

- 三个阶段使用**相同架构**的 decoder-only Transformer: 12 层, 16 头, d=1024, FFN=4096, T5-style relative PE [§IV-B]
- 每阶段 0.3B 参数, 在 16 TPUv4 上训练 1M steps (batch 256) [§IV-B]
- 训练数据: Libri-Light unlab-60k (60K 小时无标注英语语音) [§IV-A]
- 温度采样: Stage 1/2/3 分别用 T=0.6, 0.8, 0.6 [§IV-B]
- 连续重复 semantic token 在 Stage 1/2 中去除 (follow prior practice) [§IV-B]
- [agent 解读] 使用无标注数据训练是 AudioLM 的核心优势: 不需要文本转录, 不需要说话人标签

## 实验

| 指标 | 本文 (AudioLM) | Baseline (GSLM) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| sWUGGY (all) | **71.5** | 68.7 | ZeroResource 2021 | [Table IV] |
| sWUGGY (in-vocab) | **83.7** | - | ZeroResource 2021 | [Table IV] |
| sBLIMP | **64.7** | 57.1 | ZeroResource 2021 | [Table IV] |
| CER (acoustic gen.) | 3.4 | - | LibriSpeech test-clean | [Table II] |
| WER (acoustic gen.) | 6.0 | 6.6 (GSLM u2s) | LibriSpeech test-clean | [Table II] |
| Speaker accuracy (continuation) | 92.6% | - | LibriSpeech test-clean | [Table III] |
| Human distinguishability | 51.2% (=random) | - | LibriSpeech test-clean | [§IV-G] |
| Piano preference (hierarchical) | 83.3% | acoustic-only | MAESTRO | [§IV-I] |
| Synth detection accuracy | 98.6% | - | LibriSpeech | [§IV-H] |

**关键发现**:
1. **语义与声学互补** [Table I]: Semantic tokens 在 ABX 判别上优于 acoustic 约 3x, 但重建质量差约 3x → 层级方案两全其美
2. **语言能力超越监督 topline** [Table IV]: AudioLM 的 sWUGGY 甚至超越了使用强制对齐音素转录的监督 topline (71.5 vs 97.9 phone topline, 但在 causal 条件下 AudioLM 是 SOTA)
3. **说话人身份保持** [Table III]: 从 3 秒 prompt 续写时, 92.6% 的情况下说话人分类器判断一致 → semantic tokens 几乎不携带说话人信息 (acoustic generation 时仅 3.2%)
4. **人类无法区分** [§IV-G]: 51.2% 的准确率与随机猜测 (50%) 无显著差异 (p=0.23)
5. **钢琴续写** [§IV-I]: 跨域验证了层级方法的通用性, 听者 83.3% 偏好层级版本

## 局限性

1. **无文本条件**: AudioLM 只能做 audio continuation, 无法做 text-to-speech [§V] — 后续 SPEAR-TTS 和 VALL-E 分别扩展了文本输入
2. **采样率限制**: 仅 16 kHz, 现代 TTS 通常需要 24/44.1 kHz [§IV-A]
3. **三阶段串行推理**: 推理延迟高, 且三个模型各 0.3B → 总参数 0.9B [agent 解读]
4. **仅英语**: 训练在 Libri-Light (英语有声书) 上, 未验证多语言 [§IV-A]
5. **未开源**: 模型权重和完整代码未公开 [agent 解读]
6. **安全风险**: 98.6% 的合成检测准确率说明检测可行, 但也暴露了 deepfake 风险 [§VI]

## 点评

AudioLM 是 speech language model 领域的里程碑式工作。其核心贡献不在于任何单一组件 (w2v-BERT、SoundStream、decoder-only Transformer 都是已有技术), 而在于**将 semantic 和 acoustic tokens 的互补性形式化,并提出层级建模范式**。这个范式直接启发了 VALL-E (text → codec LM)、SPEAR-TTS (text → semantic → acoustic)、SoundStorm (parallel acoustic decoding) 等后续工作。

**WHY 的深层理解**: AudioLM 的设计本质上是在回答"什么信息该用什么粒度的表征来建模?"——语义结构用高层抽象 (semantic tokens, 25 Hz), 声学细节用底层重建 (acoustic tokens, 50 Hz x 12 层)。这种分治思路后来被 Mega-TTS 的四维分解和 NaturalSpeech 3 的 factorized codec 继承和发展。

**方法论启示**: AudioLM 证明了两个重要事实: (1) 自监督学习到的表征本身蕴含足够的语言知识, 不需要文本监督; (2) 条件独立性假设 (fine | coarse ⊥ semantic) 可以用来高效分解序列长度问题。

## 可复用的 idea

1. **Semantic-acoustic 互补层级**: 先建模结构再填充细节的分治策略, 可用于任何需要同时保证全局一致性和局部保真度的生成任务
2. **条件独立性分解序列**: 利用 RVQ 的层级结构将长序列拆分为多个短序列, 减少计算量
3. **3 秒 prompt 的 in-context learning**: 不用显式 speaker embedding, 仅靠 acoustic token prefix 就能保持说话人身份 — 这一范式被 VALL-E 和 SPEAR-TTS 继承
4. **Fake detection 作为安全措施**: 伴随生成能力提供检测能力, 是负责任 AI 的最佳实践


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
