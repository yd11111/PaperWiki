---
type: paper
tier: deep
title: "CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer based on Supervised Semantic Tokens"
arxiv_id: "2407.05407"
source: "Sources/CosyVoice.pdf"
authors: [Zhihao Du, Qian Chen, Shiliang Zhang, Kai Hu, Heng Lu, Yexin Yang, Hangrui Hu, Siqi Zheng, Yue Gu, Ziyang Ma, Zhijie Yan]
year: 2024
venue: "arXiv"
tags: [TTS, zero-shot, LLM-based, flow-matching, supervised-token, multilingual, coarse-to-fine, voice-cloning]
concepts: ["[[SpeechTokenizer]]", "[[ConditionalFlowMatching]]", "[[LLM-basedTTS]]", "[[Classifier-FreeGuidance]]", "[[SpeakerEmbedding]]", "[[SemanticvsAcousticTokens]]"]
models: ["[[CosyVoice2]]", "[[CosyVoice3]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]", "[[InstructedSpeechGeneration]]"]
datasets: ["LibriTTS", "AISHELL-3", "Common Voice", "LibriSpeech", "MLS"]
kb_context_sources: 3
status: draft
created: 2026-06-02
updated: 2026-06-02
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[SpeechTokenizer]], [[ConditionalFlowMatching]], [[Zero-shotSpeechSynthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[LLM-basedTTS]][待确认], [[Classifier-FreeGuidance]][待确认], [[SemanticvsAcousticTokens]][待确认] | 未命中但可能相关: 无

**谱系定位**: CosyVoice 是 LLM-based TTS 范式的早期关键工作之一,与 VALL-E (2023) 同属第一波将 LLM 引入 TTS 的系统。但与 VALL-E 使用无监督 EnCodec acoustic tokens 不同,CosyVoice 开创性地使用**监督式 semantic tokens**(从 ASR 模型提取),这是 Speech Tokenizer 演进线中"监督式 semantic tokenizer"分支的开山之作。后续 CosyVoice 2 (streaming) → CosyVoice 3 (多语言 scaling + DiffRO) 直接继承了这一设计。

**已有认知**:
- KB 已记录 Speech Tokenizer 的三分类(自监督/监督/声学),CosyVoice 正是"监督式"这一类的首创者
- CFM 页记录了 OT-CFM 在 TTS 中作为 fine stage 渲染器的角色,CosyVoice 是最早在 LLM-TTS 中采用 CFM 替代 DDPM 的系统之一
- Zero-shot Speech Synthesis 任务页已记录 CosyVoice 3 / Seed-TTS 等后续 SOTA,CosyVoice 是这条线的起点

**创新判断**: CosyVoice 的核心贡献在于将 ASR 监督信号注入 speech tokenizer,使 token 显式编码语义信息且与文本对齐。这一思路后来被整个 CosyVoice 系列继承,并影响了其他系统(如 Seed-TTS 也强调 tokenizer 是全系统瓶颈)。同时,LLM + CFM 的 coarse-to-fine 两阶段架构成为后续许多系统的标准范式。

## 速查

> [!summary] 速查
> - **一句话**: 首次将 ASR 监督训练的 semantic tokens 引入 TTS,配合 LLM + OT-CFM 两阶段架构实现可扩展的多语言零样本语音合成
> - **路线**: Text → BPE + TextEncoder → [S, x-vec, text_enc, T, speech_tokens, E] → LLM (AR) → speech tokens → OT-CFM (+ speaker emb + masked mel) → Mel → HiFi-GAN → Waveform
> - **指标**: LibriTTS WER 3.17% / SS 69.49 (大规模); 英文 WER 2.89% 接近人类 (2.66%); 中文 SS 81.58 超原始录音 (74.15) [Table 7-9]
> - **可借鉴**: (1) 在 ASR encoder 中间插入 VQ 层获取监督 semantic tokens,几乎不影响 ASR 性能; (2) x-vector 分离说话人建模,让 LLM 专注语义+韵律、CFM 专注音色+环境; (3) OT-CFM + cosine scheduler + CFG 的组合方案
> - **局限**: (1) 对 VQ 层插入位置和 codebook 大小未做充分消融; (2) 跨语言场景需省略 prompt 文本/token 来避免韵律泄露,但信息损失不可避免; (3) 指令微调数据规模有限(556h); (4) 已开源但论文未报 MOS

## 核心问题

当时 LLM-based TTS 的主流做法是使用无监督方式获取的 speech tokens(如 HuBERT 的自监督表征或 EnCodec/SoundStream 的声学 codec tokens)。这些 token **缺乏显式的语义信息,且与文本的对齐质量不高** [§1]。这导致两个问题:
1. 内容一致性差(WER 高) -- 如 VALL-E 的 WER 高达 18.70% [Table 7]
2. 说话人相似度受限 -- 因为 token 中语义和声学信息耦合,模型难以独立控制音色

CosyVoice 的核心问题是: **能否通过监督式训练获得更好的 speech tokens,使其天然携带语义信息且与文本对齐,从而大幅改善 zero-shot TTS 的质量?**

## 方法: 它怎么 work

### 整体架构

CosyVoice 采用 coarse-to-fine 两阶段架构,包含四个组件 [Fig 1(b)]:

1. **Text Encoder**: 将文本通过 BPE tokenizer + encoder 映射到与 speech token 对齐的语义空间 [§2.2, Eq.7]
2. **Supervised Semantic Speech (S³) Tokenizer**: 从 ASR 模型衍生,提取携带语义信息的离散 speech tokens [§2.1]
3. **LLM**: 以 text encodings + speaker embedding 为条件,自回归生成 speech token 序列 [§2.2]
4. **OT-CFM**: 以 speech tokens + speaker embedding + masked mel 为条件,将 tokens 转化为 Mel spectrogram [§2.3]

最后 HiFi-GAN 将 Mel 转为波形。

### 关键设计选择

#### 设计选择 1: 监督式 Semantic Tokens (S³ Tokenizer)

**WHY**: 无监督 tokens (HuBERT/EnCodec) 缺少显式语义信息。论文的核心假设是: 如果 token 本身就携带丰富的语义信息(因为从 ASR 模型提取),那么 LLM 生成这些 token 时就天然保证了内容一致性。

**HOW** [§2.1, Eq.1-5]:
- 取预训练 ASR 模型(小规模用 ESPNet Conformer,大规模用 SenseVoice-Large)
- 将 encoder 在第 6 层后拆成两半(Encoder1 + Encoder2)
- 在中间插入 Vector Quantizer (VQ): 单码本 4096 entries
- VQ 使用 EMA 更新码本 [Eq.3]: `c_μl := α·c_μl + (1-α)·h_l`
- Encoder2 之后接 ASR Decoder,整体仍以 ASR loss 训练
- 训练完成后,Encoder1 + VQ 作为 speech tokenizer

**关键发现**: VQ 插入后 ASR 性能几乎无损 -- test_clean WER 仅从 2.62% 升至 3.18% [Table 5],说明 VQ 层成功保留了语义信息。

**Evidence**: 多语言 S³ tokenizer 在 Common Voice 中文集上的 WER 甚至超过 Whisper-Large V3 (12.06% vs 12.55%),验证了 S³ tokens 与语义内容的强相关性 [Table 6]。

#### 设计选择 2: x-vector 分离说话人建模

**WHY**: 将语音建模分解为三个维度 [§1, contribution 3]:
- LLM 负责: 语义内容 + 韵律
- CFM 负责: 音色 + 环境信息

如果不引入显式 speaker embedding,LLM 需要同时学习说话人和语义,增加建模难度。

**HOW**: 使用预训练 voiceprint 模型 (CAM++) 提取 x-vector `v`,作为 LLM 输入序列的一部分 [Eq.6]:
```
[S, v, {ȳ_u}, T, {μ_l}, E]
```

#### 设计选择 3: OT-CFM 替代 DDPM

**WHY**: DDPM (如 TorToise TTS 使用的) 训练和推理都较慢。OT-CFM 学习确定性 ODE 路径而非随机 SDE,步数更少 [§2.3]。

**HOW** [Eq.9-13]:
- 构建从先验分布 p₀(X) = N(0, I) 到数据分布 q(X) 的概率密度路径
- OT flow: φ_t(X₀, X₁) = (1-(1-σ)t)X₀ + tX₁ [Eq.11]
- 神经网络匹配条件向量场: NN_θ(φ_t, t; v, {μ_l}, X̃₁) [Eq.12]
- 条件包括: speaker embedding v, speech tokens, masked mel X̃₁
- **Cosine scheduler** [Eq.13]: t := 1 - cos(½tπ),使生成初期(最难部分)分配更多步数
- **CFG**: 训练时以 p=0.2 随机丢弃条件; 推理时引导强度 β=0.7 [Eq.14]
- **Masked mel**: 从随机位置到末尾置零,让模型学会利用 prompt 的声学信息

### 零样本推理策略

**同语言** [Fig 2(a)]: 将 prompt speech 的文本和 tokens 与输入文本合并,prompt tokens 视为"已生成"的前缀,LLM 从此处续写。

**跨语言** [Fig 2(b)]: 省略 prompt 的文本和 tokens,只保留 speaker embedding 和 mel,避免源语言的韵律特征"泄漏"到目标语言。

### 训练策略

- LLM loss: 仅计算 speech tokens + E token 的交叉熵 [Eq.8],不对 text encodings 计算 loss
- CFM loss: OT-CFM 回归损失 [Eq.10]
- **Tiny** (LibriTTS): 4 x V100, 50 epochs, lr=1e-3
- **Normal** (大规模多语言): 64 x V100, 800K steps, lr=1e-4, warmup=10K

### CosyVoice-instruct

在 base 模型上做指令微调 [§2.4]:
- 支持 speaker identity / speaking style / fine-grained paralinguistics
- 移除了 LLM 中的 speaker embedding(由指令替代)
- 训练数据: 101h (speaker identity) + 407h (style) + 48h (paralinguistics) [Table 3]
- 支持 laughter / breath / emphasis / speaking-while-laughing 等细粒度控制

## 实验

| 指标 | 本文 (CosyVoice) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (%) | 3.17 (Exp-4, 大规模 BPE+S³) | VALL-E: 18.70 / UniAudio: 8.74 / SpearTTS: 6.14 | LibriTTS test-clean | [Table 7] |
| SS | 69.49 | VALL-E: 53.19 / UniAudio: 47.56 / SpearTTS: 51.71 | LibriTTS test-clean | [Table 7] |
| WER (%) 英文 | 2.89±0.18 (vs 原始 2.66) | ChatTTS: 8.32 | LibriTTS test-clean | [Table 8] |
| WER w/ 5x re-ranking | 1.51 | - | LibriTTS test-clean | [Table 8] |
| SS 英文 | 74.30±0.15 | 原始录音: 69.67 | LibriTTS test-clean | [Table 8] |
| CER (%) 中文 | 3.82±0.24 (vs 原始 2.52) | ChatTTS: 3.87 | AISHELL-3 | [Table 9] |
| CER w/ 5x re-ranking | 1.84 | - | AISHELL-3 | [Table 9] |
| SS 中文 | 81.58±0.16 | 原始录音: 74.15 | AISHELL-3 | [Table 9] |
| ASR 数据增强 WER | 2.04 (LS+MLS synth) | 原始 LS: 2.79 | LibriSpeech test-clean | [Table 11] |

**关键发现**:

1. **监督 token 显著优于无监督 token**: 对比 Exp-1 (Phone+HuBERT, WER 7.41%) vs Exp-3 (Phone+S³, WER 3.93%) vs VALL-E (Phone+EnCodec, WER 18.70%),在 SS 基本不变的情况下 WER 大幅降低 [Table 7]。这说明监督式 token 的语义保持能力远优于无监督方案。

2. **文本 tokenizer 也很关键**: Exp-2 (BPE+HuBERT, WER 5.05%) vs Exp-1 (Phone+HuBERT, WER 7.41%),仅替换文本 tokenizer 就显著改善 WER,但 SS 不变 (67.85) [Table 7]。

3. **数据规模带来显著提升**: Exp-4-LibriTTS (BPE+S³, WER 4.76%) vs Exp-4-Large-scale (WER 3.17%, SS 69.49 vs 65.94),大规模数据同时改善 WER 和 SS [Table 7]。

4. **SS 超过原始录音**: 英文 SS 74.30 vs 69.67,中文 SS 81.58 vs 74.15 [Table 8-9]。这看似反直觉,但原因是原始测试集中同一说话人的不同录音可能来自不同环境/设备,而 CosyVoice 生成时使用了统一的 speaker embedding + prompt mel,环境更一致。

5. **情感控制**: CosyVoice-instruct 在 sad (0.98 vs 0.45)、angry (0.83 vs 0.59)、disgusted (0.93 vs 0.46) 等情感上大幅优于 base [Table 10]。

6. **数据增强价值**: CosyVoice 生成的合成数据可替代真实数据训练 ASR (WER 3.00% vs 2.79%),结合后可大幅降低 WER 至 2.04% [Table 11]。

## 局限性

1. **消融不充分**: VQ 插入层位置(固定在第 6 层)和 codebook 大小(固定 4096)未做系统消融 [§4.1],论文明确标注为 "left for future work"。
2. **缺少 MOS 评估**: 全篇仅用客观指标(WER/CER + SS),未报告主观 MOS 评分 -- 这对于 TTS 系统来说是一个显著缺失。
3. **跨语言方案有信息损失**: 跨语言时需丢弃 prompt 文本和 tokens [§2.3.1, Fig 2(b)],这意味着失去了 prompt 语音的内容信息,可能影响生成质量。
4. **指令数据规模有限**: 仅 556 小时指令微调数据 [Table 3],paralinguistics 仅 48 小时,限制了细粒度控制的泛化能力。
5. **推理成本未讨论**: 论文未报告推理延迟或 RTF (real-time factor),对部署场景的适用性缺乏评估。
6. **SS 评估的局限**: SS 超过原始录音可能反映的是评估指标的局限(speaker embedding 的环境敏感性)而非真正的音色超越。

## 点评

CosyVoice 是一篇在正确方向上做出关键突破的工作。其核心洞察 -- **监督式 semantic tokens 比无监督 tokens 更适合 TTS** -- 看似简单但极其重要。在 VALL-E 之后,社区大量精力投入到改进 codec token 的建模方式(如 VALL-E 2 的重复感知采样、RALL-E 的 chain-of-thought),而 CosyVoice 选择从源头改进 token 质量,结果证明这是更高效的路径。

LLM + CFM 的 coarse-to-fine 架构设计也很精巧: 通过 x-vector 显式分离说话人信息,让 LLM 专注于序列级语义建模,CFM 专注于帧级声学渲染,各司其职。这一分工思路被后续 CosyVoice 2/3、F5-TTS、IndexTTS2 等广泛采用。

不足之处在于工程细节优于科学分析: 对 VQ 插入位置、codebook 大小等关键超参数缺乏消融,跨语言方案的信息损失也未量化分析。此外,作为 2024 年的 TTS 论文不报 MOS 是一个明显短板。

## 可复用的 idea

1. **在预训练 encoder 中间插入 VQ 获取监督 token**: 这是一种通用的、低成本的将任意预训练模型转化为 tokenizer 的方法。关键在于: (a) 选择语义信息已充分编码的中间层; (b) VQ 插入后继续端到端微调; (c) 下游任务 loss 天然保证了 token 的信息充分性。可推广到任何"需要离散化中间表征"的场景。

2. **x-vector 显式分离说话人信息**: 在 LLM 输入序列中加入 speaker embedding,让 token 只需编码语义+韵律,降低建模复杂度。这种 factorization 思路可用于任何需要多维度控制的生成系统。

3. **OT-CFM + cosine scheduler + CFG 组合**: cosine scheduler 将更多生成步数分配给初期(最难),CFG 以 p=0.2 dropout + β=0.7 引导,是一套经过验证的 flow matching 工程实践。

4. **ASR re-ranking**: 用 ASR 模型对多个随机种子的生成结果做 WER 排序取最优 [Table 8-9],是一种简单有效的离线质量提升策略 (WER 2.89% → 1.51%)。

5. **合成数据增强 ASR**: CosyVoice 生成的数据可直接替代真实数据,结合多样化文本(MLS)后效果甚至超越原始数据 [Table 11],说明高质量 TTS 可反哺上游任务。

> [!review] 自动审阅 v2 (2026-06-02)
> **结论:** pass-with-fixes
> **原则:** 复述 8 | 信赖 8 | 区分 7 | 定位 8 | 污染 8
> **Claim 标注率:** 89% (34/38)
> **问题:** 0 high, 2 medium, 2 low
> - [medium/bad-linking] frontmatter > models: 仍列出后继系统 CosyVoice 2/3 而非论文实际比较模型 (VALL-E 等)
> - [medium/template-compliance] 方法节因果解释无 [论文原文]/[agent 解读] 标签 (系统性缺失)
> **反向更新:** ✅
> **学习信号:** frontmatter_complete 检查应拆分为"字段存在"+"字段语义正确";models 字段应列本文模型+对比基准
