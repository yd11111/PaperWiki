---
type: paper
tier: deep
title: "Speech Token Prediction via Compressed-to-fine Language Modeling"
arxiv_id: "2505.24496"
source: "Sources/C2F-LM.pdf"
authors: [Wenrui Liu, Qian Chen, Wen Wang, Yafeng Chen, Jin Xu, Zhifang Guo, Guanrou Yang, Weiqin Li, Xiaoda Yang, Tao Jin, Minghui Fang, Jialong Zuo, Bai Jionghao, Zemin Liu]
year: 2025
venue: "ACM (under review)"
tags: [codec-language-model, speech-generation, token-compression, attention-mechanism, long-context, zero-shot-TTS, VALL-E]
concepts: ["[[CodecLanguageModel]]", "[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[TokenRateandBitrateTrade-offs]]", "[[Single-codebookvsMulti-codebook]]", "[[Speech-TextAlignment]]"]
models: ["[[模型库/EnCodec|EnCodec]]", "[[模型库/HuBERT|HuBERT]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: ["LibriSpeech", "LibriTTS", "MLS"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文属于 [[CodecLanguageModel]] 范式下的工作,针对 neural codec LM 中长序列 speech token 建模的核心难题提出通用解法。与已有 KB 中记载的路线相比:

- **与 codec 端压缩的区别**: [[TokenRateandBitrateTrade-offs]] 页记载了从 50Hz→4-5Hz 的 ultra-high-compression tokenizer (SyllableLM, Sylber),但这些方案在 codec 端高压缩会损失声学细节 [论文原文]。本文将压缩从 codec 层转移到 LM 层,codec 保持原始帧率以保证重建质量。

- **与多码本 vs 单码本选择的区别**: [[Single-codebookvsMulti-codebook]] 页记载了单码本 (WavTokenizer, BigCodec) 降低 token rate 的趋势。本文不改变码本数,而是在 LM 注意力层面做压缩,因此兼容单码本和多码本 codec。

- **与已有 alignment 方案的区别**: [[Speech-TextAlignment]] 页记载的对齐技术主要面向 SpeechLM 的语义对齐。本文关注的是 TTS 任务中 text token 与 speech token 之间的 monotonic alignment,通过缩短有效序列长度来改善 attention 稀疏性导致的对齐退化。

- **已有认知**: [[SemanticvsAcousticTokens]] 页确认 RVQ 第一层编码 semantic 信息、后续层编码 acoustic 信息的分层特性 (SpeechTokenizer 发现); 本文的实验进一步验证了这一点 — AR decoder 主要受益于 WER 改善 (语义), NAR decoder 主要受益于 SIM 改善 (声学)。

> 检索命中: [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[模型库/EnCodec|EnCodec]]✓, [[ResidualVectorQuantization]]✓ | 过滤: [[CodecLanguageModel]][待确认], [[TokenRateandBitrateTrade-offs]][待确认], [[Single-codebookvsMulti-codebook]][待确认], [[Speech-TextAlignment]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 neural codec LM 中,将长距离 speech token 压缩为 5Hz 紧凑表征 + 保留局部 token 滑窗,同时解决注意力稀疏和信息冗余问题,通用于各种 codec 和 LM 架构。
> - **路线**: Text + Speech prompt → [压缩 token W_0..W_k-1 (长距离)] + [原始 token (局部滑窗 N)] → AR decoder 预测第一层 → NAR decoder (仅滑窗) 预测后续层 → Codec decoder 合成波形
> - **指标**: VALL-E* (EnCodec, 960h): WER 5.23%→2.36% (-55%), SIM 56.64%→59.75% (+3.11pp) [Table 2]; WavTokenizer decoder-only: WER 5.94%→3.40% (-42.8%) [Table 2]; RTF 0.79→0.65 (-18%) [Table 6]
> - **可借鉴**: (1) 在 LM 层面做 token 压缩的思路可迁移到任何长序列生成任务; (2) 5Hz 压缩率 (对应 ~5 音节/秒) 作为语音压缩的 sweet spot; (3) 仅通过 attention mask + 插入压缩 token 实现,无需改架构
> - **局限**: 仅在 LibriSpeech 上评估 (英语朗读体); 代码未开源; 训练数据最大 44Kh 仍远小于工业级; vanilla inference 不加速 (需 KV cache eviction 才加速); NAR decoder 未用压缩 token

## 核心问题

Neural codec LM 将语音编码为长序列离散 token (7-8 秒语音 = 数百至上千 token),带来两个核心问题 [§1]:

1. **稀疏性破坏局部对齐**: speech token 序列远长于 text token 序列,导致 (a) 语义信息稀疏 — 同等语义内容分散在更多 token 中; (b) 注意力稀疏 — 更多 token 稀释注意力分布 [§1]。两者叠加使 LM 难以捕获 text-speech 之间的细粒度对齐。

2. **冗余引入语义噪声**: 语音信号天然存在声学冗余 (重复音色、静音段、同一音素的不一致表示等) [§1],这些冗余在离散 token 层面持续存在,对当前 token 的预测贡献不大甚至构成噪声。

已有方案的局限:
- 超高压缩 tokenizer (4-5Hz) 在 codec 端损失声学细节 [§1]
- 特定架构方案 (group codec, forced alignment) 依赖特定结构,泛化性差 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文不改变 neural audio codec 和 LM 的基础架构,而是在 LM 的输入和注意力层面做改造。以 VALL-E (AR decoder + NAR decoder) 为例 [§3.3]:

**输入 token 划分** (对于预测第 t 个 token) [§3.2]:
1. **Prompt tokens** [T, C_ref]: text prompt + speech prompt,提供全局语义和声学信息
2. **Long-range tokens** C_{0:t-N-1}: 远距离已生成 token,被压缩处理
3. **Local tokens** C_{t-N:t-1}: 近距离 token (滑窗内),保持原始粒度

**两个核心组件** [§3.3]:

**(1) Fine-grained Initial and Short-range Information**: 通过 prompt-local sliding window 保留 prompt + 局部 N 个 token 的原始粒度,确保 text-speech 对齐和副语言信息 (韵律、音色) 的完整性 [§3.3]。

**(2) Compressed Long-range Information**: 将远距离 token 分为 k 个不重叠的 span,每个 span 包含 G 个 token。在每个 span 末尾插入一个压缩 token W,通过 attention mask 限制 W 只能关注其所属 span 内的 token [§3.3]。LM 预测时仅关注 [prompt, W_0...W_{k-1}, local tokens],不直接看长距离原始 token [论文原文]。

压缩率 CR = frame_rate / G,目标设为 5Hz (即每秒 5 个压缩 token) [§3.3]。

### 关键设计选择

**为什么压缩在 LM 层而不是 codec 层?** 在 codec 端压缩到 4-5Hz 会丢失 prosody/timbre 等声学细节 [论文原文, §1]。本文保持 codec 的原始帧率 (50-80Hz) 保证高保真重建,仅在 LM 的注意力计算中引入压缩表示。这样 "retaining the original codec's high-fidelity reconstruction through raw tokens while processing compressed 5Hz semantics-like units" [§2]。[论文原文]

**为什么 5Hz 是 sweet spot?** 研究表明语音约含 5 个音节/秒 [§5.5, 引用 3, 11, 20],因此将 1 秒 speech token 压缩为 5 个紧凑表示是合理的语义粒度 [论文原文]。消融实验验证 CR=5Hz 最优: G 太小 (e.g., 5) 冗余未充分过滤,G 太大 (e.g., 15) 信息损失过多 [§5.5]。

**为什么 NAR decoder 只用滑窗不用压缩?** [agent 解读] 论文未显式解释,但从架构看: NAR decoder 主要处理声学信息 (timbre, prosody),其输入包含所有已有层的 codeword,信息更密集。双向滑窗已提供足够的上下文,且 NAR decoder 的序列长度本身已被每层独立处理所限制。

**W token 的实现**: W 是一个可学习的 embedding,不被 LM 预测 (损失中忽略 W 位置) [§3.4]。它通过 attention mask 约束只关注其对应 span 的 token,在前向传播中自然聚合该 span 的信息 [论文原文]。

**两种推理策略** [§3.4]:
- *Vanilla inference*: 每 G 步插入 W 的推理步 (无实际输出),不改变序列处理长度,不加速但改善质量。
- *Faster inference*: 驱逐长距离 token 的 KV cache,仅保留 [prompt, compressed tokens, local tokens] 的 KV cache。复杂度从 O(N_p + T) 降至 O(N_p + T/G + N_AR),其中 T 为已生成 speech token 总数 [§3.4]。

### 训练策略

- AR decoder: 使用完整 C2F (滑窗 + 压缩 token),交叉熵损失,W 位置损失忽略 [§3.4]
- NAR decoder: 仅使用 prompt-local 双向滑窗,不使用压缩 token [§3.4]
- 所有模型使用 960h LibriSpeech 训练;部分实验扩展到 20Kh (加 MLS,论文 §4 提及数据总量 44Kh 但 Table 2 实验结果对应 20Kh) [§4]
- Transformer: 1024 dim, 12 blocks, 16 heads, 加入 RoPE 和 SiLU (Llama-style) [§4]

## 实验

| 指标 | 本文 (VALL-E*^CF, 960h) | Baseline (VALL-E*, 960h) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER↓ | **2.36%** | 5.23% | LibriSpeech test | [Table 2] |
| SIM↑ | **59.75%** | 56.64% | LibriSpeech test | [Table 2] |
| UTMOS↑ | **4.18** | 4.16 | LibriSpeech test | [Table 2] |
| WER↓ (20Kh) | **2.21%** | 5.05% | LibriSpeech test | [Table 2] |
| SIM↑ (20Kh) | **57.72%** | 54.83% | LibriSpeech test | [Table 2] |
| MOS↑ | **3.48±0.28** | 3.29±0.27 | LibriSpeech test | [Table 3] |
| SMOS↑ | **4.08±0.20** | 3.79±0.19 | LibriSpeech test | [Table 3] |
| PMOS↑ | **3.75±0.31** | 3.12±0.17 | LibriSpeech test | [Table 3] |

**跨 codec 泛化** [Table 2]:

| Codec | LM | WER (baseline→C2F) | SIM (baseline→C2F) |
| --- | --- | --- | --- |
| WavTokenizer | decoder-only | 5.94%→**3.40%** | 47.59%→**47.99%** |
| BigCodec | decoder-only | 16.58%→**10.90%** | 46.04%→**46.51%** |
| HuBERT | decoder-only | 7.04%→**5.86%** | - | 
| EnCodec | VALL-E* | 5.23%→**2.36%** | 56.64%→**59.75%** |

**推理加速** [Table 6]: WavTokenizer (75Hz) 的 decoder-only LM,平均每步推理时间 0.0105s→0.0086s,RTF 0.7911→0.6482 (约 18% 加速)。

**关键消融发现** [Table 4, Table 5]:
- 仅用 prompt-local 滑窗 (无压缩) 已显著优于全序列 attention: VALL-E WER 5.23%→3.24% (N_AR=50), SIM 56.64%→58.65% (N_NAR=50) [Table 4]
- 最优参数: G=10 (EnCodec 50Hz→CR=5Hz), N_AR=50 (1s 语音), N_NAR=50 [Table 5]
- 增加滑窗至 N_AR=120 反而性能下降 (WER 4.66%),验证了长距离冗余引入噪声的假说 [Table 5]

## 局限性

1. **评估数据受限**: 仅在 LibriSpeech (英语朗读体) 上评估,未验证对话/情感/多语言场景的泛化性 [agent 解读]
2. **静态压缩率**: G 和 N 为固定超参,未探索根据内容动态调整压缩率 (论文自述为 future work) [§6]
3. **训练数据规模**: 最大 44Kh,远小于工业级系统 (如 VALL-E 原始 60Kh, Seed-TTS 100Kh+) [agent 解读]
4. **NAR 未用压缩**: NAR decoder 仅用滑窗,未探索压缩 token 对声学层建模的潜在收益 [agent 解读]
5. **代码未开源**: 仅提供 demo 页面,无法复现 [§Abstract]
6. **KV cache eviction 复杂度**: Faster inference 需要动态管理 KV cache,实现相比标准 AR 更复杂 [agent 解读]

## 点评

**强项**: (1) 问题分析扎实 — 稀疏性和冗余性的双重分析有说服力,用 attention 可视化直观验证 [Fig 3]; (2) 方法通用性好 — 仅靠 attention mask + 压缩 token 实现,不依赖特定 codec 或 LM 架构,在 4 种 codec + 2 种 LM 上一致有效; (3) 消融全面 — G, N, 滑窗、单独分析 AR/NAR 的贡献,覆盖了主要设计空间。

**弱项**: (1) 实验规模偏小 — 960h LibriSpeech 在 2025 年属于小数据实验,20Kh 虽有补充但仍不充分; (2) 与同期方法对比不足 — 未与 VALL-E 2 (group codec) 或 ELLA-V (alignment reordering) 做直接对比,尽管论文声称方法更通用; (3) 5Hz 压缩率的物理直觉 (音节率) 虽合理,但在非朗读体语音中可能不成立。

## 可复用的 idea

1. **LM 层压缩 token**: 在任何长序列 LM (speech/music/video) 中,可通过插入压缩 token + attention mask 实现信息瓶颈,在不改架构的前提下缩短有效序列长度。
2. **5Hz 语义粒度**: ~5 音节/秒作为语音信息的 coarse-grained 语义粒度,是 codec 压缩和 LM 建模的有用参考点。
3. **滑窗 + 压缩的组合**: 局部保持精细粒度、远距离做压缩的策略,可作为处理任何短距依赖信号 (语音、音乐等) 的通用模式。

> [!review] 审阅
> 见 `_review/C2F-LM-review.yml`。
> 结论: **pass-with-fixes** (0 high, 2 medium, 1 low)。
> 主要问题: (1) 训练数据规模标注 44Kh/20Kh 需澄清 (已修正); (2) 复杂度公式 T 未定义 (已修正)。
> 反向更新可放行。
