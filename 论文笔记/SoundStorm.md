---
type: paper
tier: deep
title: "SoundStorm: Efficient Parallel Audio Generation"
arxiv_id: "2305.09636"
source: "Sources/SoundStorm.pdf"
authors: [Zalan Borsos, Matt Sharifi, Damien Vincent, Eugene Kharitonov, Neil Zeghidour, Marco Tagliasacchi]
year: 2023
venue: "arXiv"
tags: [audio-generation, non-autoregressive, masked-generative, parallel-decoding, RVQ, acoustic-model, dialogue-synthesis, TTS]
concepts: ["[[MaskedGenerativeModeling]]", "[[ResidualVectorQuantization]]", "[[SemanticvsAcousticTokens]]", "[[SpeechTokenizer]]", "[[Non-autoregressiveTTS]]"]
models: ["[[模型库/SoundStorm|SoundStorm]]", "[[模型库/SoundStream|SoundStream]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]], [[SpeechTokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ResidualVectorQuantization]]✓, [[SemanticvsAcousticTokens]]✓, [[SpeechTokenizer]]✓ | 过滤: [[MaskedGenerativeModeling]][待确认] | 未命中但可能相关: 无

**谱系定位**: SoundStorm 是 AudioLM (Borsos et al., 2022) pipeline 中 acoustic generation stage 的高效替代方案。AudioLM 建立了 "semantic tokens → acoustic tokens" 的两阶段框架 (详见 [[SemanticvsAcousticTokens]] 和 [[CodecLanguageModel]]),但其 acoustic stage 使用自回归生成展开后的 SoundStream RVQ tokens,速度极慢 (O(T×Q) 步)。SoundStorm 将此阶段替换为 masked generative modeling,实现 O(Q) 步并行解码,比 AudioLM 快两个数量级。

**已有认知**:
- RVQ 页记录了 SoundStream 的层级结构: 前面层编码 coarse 信息,后面层编码 fine details -- SoundStorm 正是利用这一特性设计 coarse-to-fine 的逐层解码策略
- Masked Generative Modeling [待确认] 页记录 SoundStorm 是"首次将 mask-and-predict 用于音频"的工作 (源自 MaskGIT for images),后续被 MaskGCT 扩展到完整 TTS pipeline
- Semantic vs Acoustic Tokens 页记录了 AudioLM 的 semantic→acoustic 层级: SoundStorm 仅负责 acoustic 阶段,接收 semantic tokens 作为条件

**创新判断**: SoundStorm 的核心贡献是将 MaskGIT 的 iterative parallel decoding 推广到 RVQ 结构的 multi-level token 序列。关键 insight 是利用 RVQ 的层级条件独立性: fine level tokens 在给定 coarse level tokens 时可以并行采样。

> [!summary] 速查
> - **一句话**: 将 MaskGIT 的迭代并行解码推广到 RVQ 结构,按 RVQ 层 coarse-to-fine 逐级生成 SoundStream acoustic tokens,比 AudioLM 快 100 倍且质量更好
> - **路线**: Semantic tokens (conditioning, from AudioLM/SPEAR-TTS) → Conformer (bidirectional self-attention, sum frame embeddings) → Q separate heads → RVQ level-wise iterative parallel decoding → SoundStream tokens → SoundStream decoder → Waveform
> - **指标**: LibriSpeech test-clean (prompted) WER 2.99% / Audio quality 4.05 / Voice preservation 0.57 / Acoustic consistency 0.91; RTF 0.017 vs AudioLM ~1.7 [Table 1, Fig 3]
> - **可借鉴**: (1) Frame-level embedding sum 将 Q 层 RVQ tokens 压缩到与 frame 数等长的序列; (2) Coarse-to-fine RVQ-level decoding + level-wise confidence-based masking; (3) 最后一步用 greedy 替代 confidence-based 提升质量; (4) Acoustic consistency metric (contrastive model 测时间漂移)
> - **局限**: (1) 不做 text→semantic (需外部模型如 SPEAR-TTS); (2) Conformer 350M 参数,对长序列仍有二次复杂度; (3) 需 time-aligned conditioning signal; (4) 仅在 LibriSpeech + 对话数据上验证

## 核心问题

AudioLM 建立了 semantic→acoustic 的两阶段音频生成框架,但其 acoustic generation stage 使用自回归模型在展开的 SoundStream tokens 上逐 token 生成 [§1]。对于 Q=12 层 RVQ,T 帧 SoundStream:

- 自回归需要 T×Q 步,且 self-attention 的序列长度也是 T×Q → O(T²Q²) 复杂度
- 生成 30 秒音频需要约 100 秒 (RTF ~1.7) [Fig 3] [论文原文]

SoundStorm 的核心问题: **如何利用 RVQ 的层级结构实现高效并行解码,在保持或提升 AudioLM 质量的前提下将速度提升两个数量级?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SoundStorm 作为 AudioLM 的 acoustic generator 的 drop-in replacement [§1, Fig 1]:
- **输入**: time-aligned conditioning tokens (如 AudioLM 的 semantic tokens, 25 Hz)
- **输出**: SoundStream tokens (Q=12 层 RVQ, 50 Hz, codebook size 1024)
- **模型**: Conformer (350M params, 12 layers, 16 heads, dim 1024) + Q 个独立输出 head

### 关键设计选择

#### 设计选择 1: Frame-level Embedding Sum

**WHY**: [论文原文] 如果将所有 Q 层 RVQ tokens 展开为长序列,self-attention 的序列长度为 T×Q,计算成本随 bitrate (Q) 增长呈二次方。[§3.1] 解决方案是将同一帧的所有 token embeddings 求和,使序列长度等于帧数 T,与 Q 无关。

**HOW** [§3.1]:
- 将 conditioning tokens 与 SoundStream tokens 在帧级别交错排列
- 将同一帧对应的所有 token (conditioning + Q 层 RVQ) 的 embedding 求和
- 求和后的 embeddings 送入 Conformer 做双向 self-attention
- 输出侧用 Q 个独立 dense layer 作为 head,预测各层 target tokens
- [agent 解读] 这个设计利用了一个关键假设: 同一帧内各层 RVQ token 的信息可以通过加法组合。因为 RVQ 的数学定义就是残差的叠加 (z_hat = Σ q_i),所以 embedding sum 在语义上对应于重建向量的组合

**关键结果**: self-attention 的序列长度 = SoundStream 帧数 (50 Hz),独立于 RVQ 层数 Q → 可以处理任意长度音频 (分钟级) [论文原文]

#### 设计选择 2: RVQ Level-wise Coarse-to-fine Decoding

**WHY**: [论文原文] RVQ 的层级结构意味着 coarse levels 编码重要信息 (语义/韵律),fine levels 编码局部声学细节。后者在给定前者时具有条件独立性 -- 即 fine level tokens 可以并行采样 [§3.2]。

**HOW** [§3.2, §3.3]:
- 推理时,从所有 SoundStream tokens 被 mask 开始
- **逐层生成** (level-wise): 先生成第 1 层 (最 coarse),再生成第 2 层 (条件于第 1 层),依次到第 Q 层
- **层内并行**: 每层内使用 MaskGIT 的 confidence-based iterative decoding:
  1. 模型预测所有 masked 位置
  2. 按 confidence score 排序,保留 top-p_i 比例的预测
  3. 将低 confidence 位置 remask
  4. p_i 按 cosine schedule 递增
  5. 重复直到该层所有位置都被填充
- **Greedy at last iteration**: 每层最后一步使用 greedy decoding (取 argmax) 而非 confidence-based sampling [论文原文] 这改善了感知质量

**解码策略**: 使用 (16, 1, 1, ..., 1) iterations per RVQ level [§4.1]
- 第 1 层: 16 次迭代 (最重要)
- 第 2-12 层: 各 1 次 greedy decoding
- 总计 27 forward passes (预测 30 秒 × 50 Hz × 12 层 = 18000 tokens)
- [agent 解读] 第 1 层需要更多迭代是因为它承载最多信息量 (coarse semantic/prosody),后续层的 fine details 在条件于 coarse 后的分布更集中,greedy 即可

#### 设计选择 3: Training Masking Scheme

**WHY**: [论文原文] 训练时的 masking 方案需要模拟推理时的 coarse-to-fine level-wise decoding,同时支持 voice prompting [§3.2]。

**HOW** [§3.2]:
1. 随机采样 prompt 分界点 t ~ U{0, T-1}: 前 t 帧不 mask (prompt)
2. 随机采样当前 RVQ level q ~ U{1, Q}
3. 按 cosine schedule 随机 mask 第 q 层的 non-prompt tokens
4. 第 q 层以上 (q+1 到 Q) 的所有 non-prompt tokens 全部 mask
5. Loss 仅计算第 q 层被 mask 位置的交叉熵

[agent 解读] 这个训练方案优雅地统一了三种推理场景: (a) unprompted generation (t=0), (b) prompted generation (t>0), (c) 任意 RVQ level 的条件生成

### 训练策略

- 模型: Conformer, 350M 参数, 12 layers, 16 heads, dim 1024, FFN dim 4096 [§4.1]
- 位置编码: Rotary Positional Embeddings [§4.1]
- SoundStream: Q=12, 50 fps, codebook 1024, 6000 bps [§4.1]
- Conditioning: AudioLM semantic tokens (w2v-BERT, 25 Hz, k-means 1024), 重复到 50 Hz 匹配帧率 [§4.1]
- 训练数据: LibriLight 60K hours [§4.1]
- 训练: 10 epochs, 随机窗口 0-30 秒

## 实验

| 指标 | 本文 (SoundStorm) | Baseline (AudioLM) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (%) prompted short | 2.99 | 3.77 | LibriSpeech test-clean | [Table 1] |
| WER (%) prompted mid | 2.55 | 3.40 | LibriSpeech test-clean | [Table 1] |
| WER (%) prompted long | 3.36 | 3.75 | LibriSpeech test-clean | [Table 1] |
| CER (%) prompted short | 0.81 | 1.10 | LibriSpeech test-clean | [Table 1] |
| Audio quality (MOS-like) short | 4.01 | 3.91 | LibriSpeech test-clean | [Table 1] |
| Audio quality mid | 4.16 | 4.06 | LibriSpeech test-clean | [Table 1] |
| Audio quality long | 4.20 (original 3.99) | 4.08 | LibriSpeech test-clean | [Table 1] |
| Voice preservation (prompted) | 0.48 / 0.48 / 0.59 | 0.46 / 0.48 / 0.57 | LibriSpeech test-clean | [Table 1] |
| Acoustic consistency (prompted) | 0.96 / 0.96 / 0.91 | 0.48 / 0.59 / 0.86 | LibriSpeech test-clean | [Table 1] |
| Runtime (30s audio) | 0.5s | ~50s (Stage 2&3) | TPU-v4 | [Fig 3] |
| RTF | 0.017 | ~1.7 | TPU-v4 | [Fig 3] |
| Dialogue (30s) total | 2s | - | TPU-v4 | [§5] |

**关键发现**:

1. **100 倍加速**: SoundStorm 生成 30 秒音频仅需 0.5 秒 (RTF 0.017),加上 SoundStream decoding (0.1s) 和 semantic generation (1.4s),整个 pipeline 2 秒完成 30 秒对话 [Fig 3] [论文原文]

2. **WER/CER 全面优于 AudioLM**: 在 short/mid/long 三种长度的 prompted 和 unprompted 场景上,SoundStorm 的 WER 和 CER 均低于 AudioLM [Table 1]

3. **Acoustic consistency 显著改善**: 这是 SoundStorm 最突出的优势。AudioLM 在长序列上出现 acoustic drift (声学一致性从 short 0.48 降到 long 0.86),而 SoundStorm 保持 0.91-0.96 [Table 1, Fig 2] [论文原文] 因为 SoundStorm 的双向注意力可以全局感知 prompt 的声学特性,而 AudioLM 的单向注意力只能看到局部历史

4. **Audio quality 与 AudioLM 持平**: MOS-like score 两者相近 (4.01-4.20 vs 3.91-4.08) [Table 1]

5. **16 iterations 是质量-速度最优点**: [Fig 4] 第 1 层用 16 iterations vs 1 iteration (greedy) 可提升 0.1-0.2 的 audio quality score。超过 16 次无进一步提升 [论文原文]

6. **Fine levels 无需多次迭代**: 第 2-12 层用 greedy (1 iteration) 即可,增加迭代次数无统计显著改善 [§4.3] [论文原文] 这验证了 fine level tokens 的条件分布确实更集中

7. **对话合成能力**: 配合 SPEAR-TTS 的 text-to-semantic model (1.2B ByT5),可合成 30 秒多轮对话 (多说话人 + 填充词 + 语气词) [§5]

## 局限性

1. **仅做 acoustic generation**: SoundStorm 只是 pipeline 中的 acoustic generator,需要外部 text-to-semantic model (AudioLM / SPEAR-TTS) 提供条件 [§3]

2. **需要 time-aligned conditioning**: conditioning signal 必须与 SoundStream 帧对齐 [§3] [论文原文] "We leave the extension to other types of conditioning signals via cross-attention... for future work"

3. **Conformer 的二次复杂度**: 虽然序列长度降为 T (而非 T×Q),但 self-attention 仍为 O(T²)。对于分钟级音频 (T=50×60=3000),这仍然是一个开销

4. **未在 TTS 端到端评估**: 论文的 TTS 评估依赖 ground-truth semantic tokens [§4.2],而非端到端 text→waveform。实际使用中 text-to-semantic 的误差会传播

5. **仅在英文上验证**: 训练和评估仅在 LibriLight/LibriSpeech (英文) 上进行

6. **对话数据未公开**: §5 中的 100K 小时对话语料及 text-to-semantic model 未开源

## 点评

SoundStorm 是一篇算法层面的贡献大于系统层面的工作。其核心 insight -- **利用 RVQ 的层级条件独立性实现 coarse-to-fine 并行解码** -- 简洁而有力。将 MaskGIT 从 2D image tokens 推广到 RVQ 的 multi-level 1D audio tokens,这个推广并不平凡: 它需要重新设计 masking scheme 来同时处理 level-wise 和 frame-wise 的 mask。

Frame-level embedding sum 是一个精妙的工程选择: 它将 O(T×Q) 的序列长度压缩为 O(T),使模型复杂度与 RVQ 层数无关。这为使用更多 RVQ 层 (更高音质) 打开了大门,而不需要担心计算成本爆炸。

实验中最有说服力的是 acoustic consistency 的改善 [Fig 2]: AudioLM 的单向注意力导致长序列上 speaker identity 逐渐漂移,而 SoundStorm 的双向注意力天然解决了这个问题。这不仅是速度的提升,更是质量的提升。

SoundStorm 的影响体现在后续工作中: MaskGCT 将 masked generative modeling 扩展到完整 TTS pipeline (T2S + S2A),NaturalSpeech 3 的 factorized diffusion 也借鉴了 mask-and-predict 的离散 diffusion 公式。可以说 SoundStorm 建立了"非自回归生成 RVQ tokens"这一方向的技术范式。

## 可复用的 idea

1. **Frame-level embedding sum**: 将同一帧的多层 token embeddings 求和而非展开,使序列长度与层数无关。适用于任何基于 RVQ 的序列建模任务。

2. **RVQ level-wise coarse-to-fine decoding**: 按 RVQ 层逐级生成,每层内并行解码。第 1 层多迭代 (信息密集),后续层 greedy (条件分布集中)。这种非均匀的 iteration 分配策略 (16,1,1,...,1) 是经过实验验证的最优方案。

3. **统一 masking scheme**: 一个 masking 方案同时支持 (a) 任意 RVQ level 条件生成, (b) voice prompting, (c) unprompted generation。通过随机采样 (t, q, M) 三元组实现,训练高效。

4. **Acoustic consistency metric**: 训练 contrastive model 衡量生成音频与 prompt 的声学一致性随时间的变化 (cosine similarity of embeddings from non-overlapping crops)。比单纯的 speaker similarity 更全面。

5. **Confidence-based greedy hybrid**: 迭代解码过程中使用 confidence-based masking,但最后一步切换为 greedy decoding 提升质量。这个简单技巧在 SoundStorm 中贡献了可感知的质量提升。

---

检索命中: [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]], [[SpeechTokenizer]] | 过滤: [[MaskedGenerativeModeling]](pending-review) | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass-with-fixes
> 
> 结构检查: 速查卡片 ✗ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
