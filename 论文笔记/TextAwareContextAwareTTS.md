---
type: paper
tier: deep
title: "Text-aware and Context-aware Expressive Audiobook Speech Synthesis"
arxiv_id: "2406.05672"
source: "Sources/TextAwareContextAwareTTS.pdf"
authors: [Dake Guo, Xinfa Zhu, Liumeng Xue, Yongmao Zhang, Wenjie Tian, Lei Xie]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, audiobook, style-modeling, contrastive-learning, context-aware, prosody, expressiveness, VITS, language-model]
concepts: ["[[GlobalStyleTokens]]", "[[StyleTransferinTTS]]", "[[ProsodyModeling]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: ["[[VITS]]", "[[HuBERT]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[ProsodyModeling]], [[GlobalStyleTokens]], [[StyleTransferinTTS]], [[VITS]], [[HuBERT]], [[Self-SupervisedSpeechRepresentation]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]]✓, [[GlobalStyleTokens]][待确认], [[StyleTransferinTTS]][待确认], [[VITS]][待确认], [[HuBERT]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文处于 **audiobook 长文本 TTS 的风格建模** 赛道。在 KB 的演进线索中:
- [[GlobalStyleTokens]] 开创了无监督风格发现 (ICML 2018),但受限于固定数量 style tokens (通常 10-20 个),难以覆盖有声书中的丰富风格变化。TP-GST (Stanton et al., SLT 2018) 扩展了 GST 使其可从文本预测,但仍受限于 token 数量瓶颈。
- [[StyleTransferinTTS]] 的演进路线已从 style tagging → reference speech → NL descriptions → instruction-guided,本文属于 "reference speech → text-predicted" 的过渡路线,用跨模态对比学习实现 text-adaptive style。
- [[ProsodyModeling]] (confirmed) 记录了跨句韵律建模的多种方案 (context-aware text embeddings, acoustic contexts, hierarchical discourse),本文的 context encoder 属于文本上下文路线。
- [[VITS]] 是本文的 TTS backbone 之一 (通过 Bert-VITS2 变体),[[HuBERT]] 为 speech style encoder 提供特征提取。
- [[Self-SupervisedSpeechRepresentation]] 中的 contrastive learning 范式被本文迁移到跨模态 (speech → text) 风格空间构建。

**已有认知 vs 本文创新**: KB 中已有大量风格迁移工作,但多依赖参考音频或离散标签。本文的差异化在于: (1) 用 CLIP 式跨模态对比学习建立连续文本风格空间,不受 GST token 数量限制; (2) context encoder 是 plug-and-play 模块,验证了在 VITS-based 和 LM-based 两种范式上的通用性。

## 速查

> [!summary] 速查
> - **一句话**: 用 CLIP 式跨模态对比学习从文本预测风格 embedding + 跨句 context encoder,实现有声书 TTS 的多样风格和连贯韵律
> - **路线**: Text → T5 encoder + attention pooling → style embedding (与 HuBERT speech style 对齐) → VQ → Context Encoder (prev/cur/next sentences + BERT) → VITS/LM TTS → waveform
> - **指标**: TACA-VITS EMOS 3.93 vs VITS 3.61 (+0.32); TACA-LM EMOS 4.05 vs LM 3.80 (+0.25); text-speech style cosine similarity 0.93 (T5) vs 0.82 (BERT) [Table 1, §4.1]
> - **可借鉴**: (1) 用 speech style similarity 阈值 (alpha/beta) 自动构造跨模态对比学习正负样本,避免人工标注; (2) VQ 量化 style embedding 提升稳定性; (3) pre-training + context-aware fine-tuning 两阶段策略复用有/无上下文标注数据
> - **局限**: 仅中文有声书场景验证; 无消融分解 text-aware style vs context encoder 各自贡献; LM-based 模型 CER 高 (13.1%); 数据集未公开

## 核心问题

有声书 TTS 面临三重挑战 [§1]:
1. **风格多样性**: 专业朗读者在不同章节、场景 (叙述/对话/旁白) 中展现丰富风格,传统标签或参考音频难以覆盖
2. **跨句韵律连贯**: 单句 TTS 忽略上下文,导致长文本韵律不连贯
3. **推理可行性**: 推理时只有文本可用,不可能为每句手工指定风格标签或参考音频

现有方案的不足 [§1]:
- GST/TP-GST: token 数量固定 (通常 10-20),不足以覆盖有声书丰富风格
- CLAPSpeech: 只关注局部韵律 (word/phoneme level),不建模全局风格
- 上下文方法 (Xu et al. 2021, Lei et al. 2023): 只用文本或声学上下文的一种,未同时建模文本风格和跨句信息

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统分两大模块 [§2, Fig 1]:

**模块 1: Text-aware Style Modeling** — 建立从文本预测风格的连续空间
```
Speech → Chinese-HuBERT-Large (layer 6) → SRL Speech Encoder (frozen) → speech style embedding
                                                                              ↓ cross-modal supervision
Text → T5 (Randeng-T5-784M) → attention pooling → text style embedding
                                    ↓ contrastive loss + cosine similarity loss
                          Text-aware style space (dim=384)
```

**模块 2: Context Encoder** — 融合跨句信息和风格 [Fig 2]
```
[Prev sentence | Cur sentence | Next sentence] → Text Encoder → cross-sentence features
                                                                      ↓
Text style embedding → VQ (codebook 64*32) → Random Select ←→ Speech style embedding
                                                     ↓
                                            Context Encoder output → TTS model
```

两个模块组合后应用于两种 TTS 框架:
- **TACA-VITS**: 替换 Bert-VITS2 的 text encoder 为 context encoder [Fig 3]
- **TACA-LM**: 在 nanoGPT AR 语言模型中加入 context encoder,预测 HuBERT semantic tokens → Hubert-VITS 合成波形 [Fig 4]

### 关键设计选择

**1. 为什么用跨模态对比学习而不是直接从文本预测风格?**

[论文原文] 风格主要通过语音表达,文本与语音风格之间存在 gap。直接从文本预测风格缺少语音侧的监督信号 [§2.1]。CLIP 式对比学习可以将两个模态对齐到共享空间,比 TP-GST 的 token 权重预测更灵活,因为连续空间不受 token 数量限制 [§1]。

[agent 解读] 这实质上是把视觉-语言对齐 (CLIP) 的思路迁移到语音-文本风格对齐。关键优势在于: speech encoder 学到的风格空间是连续的 (不像 GST 的有限 token bank),文本 encoder 对齐到这个连续空间后,继承了丰富的风格表达能力。

**2. 为什么用 similarity 阈值构造正负样本而不是配对数据?**

[论文原文] 同一文本在不同上下文下可能有不同风格,简单的 speech-text 配对作为正样本不够准确。通过 speech style similarity 设定阈值 (alpha=0.60, beta=0.95),相似度 > beta 为正样本,< alpha 为负样本,中间为 unknown 样本 (不参与训练),实现半监督对比学习 [§2.1]。

[agent 解读] 这是 semi-supervised contrastive learning 的标准做法,alpha-beta 阈值引入了 margin,避免了模糊样本对训练的干扰。unknown 区域的设计是关键,它容忍了风格边界的模糊性。

**3. 为什么对 style embedding 做 VQ?**

[论文原文] VQ 使模型能有效学习和捕捉跨大量样本的风格表达共性,生成更稳定的表现性语音 [§2.2]。

[agent 解读] VQ 在此起到正则化作用 --- 连续 style embedding 直接注入可能引入噪声,离散化后迫使模型学到风格的 "典型模式",类似于 VQ-VAE 中 codebook 的聚类效应。codebook 大小 64*32 (product quantization) 在粒度和稳定性间取得平衡。

**4. Context encoder 为什么是 plug-and-play?**

[论文原文] context encoder 的输入是 phonemes + BERT embeddings + style embedding,输出替代 TTS 原有的 text encoder 输出,因此可以插入任何需要 text encoder 的 TTS 框架 [§2.2]。

[agent 解读] 这种设计使其既可用于 end-to-end 模型 (VITS) 也可用于 LM-based 模型,验证了通用性。但这也意味着 context encoder 的输出维度必须与原 text encoder 兼容,限制了其灵活性。

### 训练策略

**两阶段训练** [§2.2]:
1. **Pre-training (Base)**: 不使用上下文信息,在 100H-Multi-Style 数据上训练基础模型
2. **Context-aware fine-tuning (TACA)**: 加入上下文信息,在 20H-Audiobook-HQ (有句序标注) 上微调

[论文原文] 这样设计是为了充分利用有/无上下文标注的数据 --- 大部分数据没有句序信息,只有小部分有声书数据有 [§2.2]。微调时 style embedding 随机从 speech 或 text 获取,增强泛化性 [§2.2]。

**Text-aware style 训练** [§3.2]:
- Speech Encoder: 在 100H-Multi-Style 上训练 (batch=96)
- T5 text encoder: 在 6kH-Audiobook 上训练 (batch=64),冻结 speech encoder

**LM-based 模型训练** [§3.2]:
- Hubert-VITS: 300h 高质量音频训练 (batch=48)
- nanoGPT LM: 在 6kH-Audiobook 上预训练 (batch=24, 10x gradient accumulation)
- Semantic tokens: Chinese-Hubert-Base layer 9, codebook 1024, dim 128

## 实验

| 指标 | 本文 (TACA-VITS) | Baseline (VITS) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| MCD | 4.21 | 4.23 | 20H-Audiobook-HQ test | [Table 1] |
| CER | 5.99% | 5.80% | 20H-Audiobook-HQ test | [Table 1] |
| NMOS | 3.90+-0.098 | 3.84+-0.110 | 20H-Audiobook-HQ test | [Table 1] |
| EMOS | **3.93+-0.105** | 3.61+-0.143 | 20H-Audiobook-HQ test | [Table 1] |

| 指标 | 本文 (TACA-LM) | Baseline (LM) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| MCD | **8.15** | 8.92 | 20H-Audiobook-HQ test | [Table 1] |
| CER | **13.1%** | 13.9% | 20H-Audiobook-HQ test | [Table 1] |
| NMOS | **3.22+-0.099** | 2.91+-0.087 | 20H-Audiobook-HQ test | [Table 1] |
| EMOS | **4.05+-0.104** | 3.80+-0.112 | 20H-Audiobook-HQ test | [Table 1] |

| 指标 | T5 encoder | BERT encoder | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Text-speech style cosine similarity | **0.93** | 0.82 | 1000 sentences | [§4.1] |

**Style space 分析** [§4.1, Fig 5]:
- t-SNE 可视化 9 个章节 (450 句): 同章节句子聚类,但簇内有分散性 → 捕捉到章节级风格一致性 + 句子级风格细微差异
- 整体风格空间分布多样 → 覆盖了丰富的风格变化

**关键发现** [§4.2-4.3]:
- TACA-VITS: CER 略微上升 (5.80→5.99%) 但 EMOS 显著提升 (+0.32),说明 context encoder 主要改善表现力而非清晰度 [Table 1]
- TACA-LM: MCD 和 CER 均下降,论文认为 LM 的更强建模能力使其能从语义/风格条件中获益 [§4.2]
- LM 系统整体 CER 高 (13.1-13.9%),源于 AR 模型固有的漏读和重复问题 [§4.2]
- LM 在 EMOS 上优于 VITS (4.05 vs 3.93),论文归因于 LM 更强的语义理解和上下文风格表达能力 [§4.3]

## 局限性

1. **无消融实验**: 未分解 text-aware style modeling 和 context encoder 各自的贡献,无法判断哪个模块更关键 [agent 解读]
2. **数据集限制**: 仅在中文普通话有声书上验证,未测试跨语言或跨域泛化性 [agent 解读]
3. **LM 质量问题**: LM-based 模型 CER 13.1%,远高于 VITS 的 5.99%,实际部署可用性存疑 [Table 1]
4. **评估规模**: 仅 20 名听众参与 MOS 评估,测试集约 100 条,统计效力有限 [§4.3]
5. **MOS 绝对值偏低**: TACA-VITS NMOS 仅 3.90,与当前 SOTA 系统 (4.2+) 有差距,可能受限于 Bert-VITS2 底座和数据规模 [agent 解读]
6. **未公开代码和数据**: 虽有 demo 页但未见公开代码仓库,6kH-Audiobook 数据集未公开 [agent 解读]
7. **Context encoder 感受域未探索**: 只用了 prev/cur/next 三句,未探索更长上下文 (段落级) 的效果 [agent 解读]

## 点评

**优势**:
- 将 CLIP 的跨模态对齐思路引入 TTS 风格建模是自然且有效的迁移,alpha/beta 阈值构造正负样本的做法巧妙地处理了风格边界模糊性
- Context encoder 的 plug-and-play 设计验证了在两种不同 TTS 范式上的通用性,工程价值较高
- t-SNE 风格空间可视化 [Fig 5] 提供了直观的定性证据 --- 章节级聚类 + 句子级分散 --- 但需要更多定量分析

**不足**:
- 缺乏消融实验是主要遗憾: text-aware style 单独的贡献、context encoder 单独的贡献、VQ 的影响都未被分离验证
- 与同期 audiobook TTS 工作 (如 Lei et al. ICASSP 2023 的层级 Transformer) 缺乏直接对比
- VITS 系统 CER 略有上升但未讨论原因,可能是 context encoder 引入的额外信息干扰了发音准确性

**在 KB 中的定位**:
- 本文是 TP-GST → 连续风格空间 演进中的一个节点,用跨模态对比学习突破了 token 数量限制
- 在 audiobook TTS 的 context-aware 方法族中,属于纯文本上下文路线 (vs acoustic context vs hybrid)
- 与后续的 LLM-based TTS (CosyVoice, VALL-E) 相比,本文的 LM-based 方案 (nanoGPT) 规模较小,但验证了 context-aware 模块的可迁移性

## 可复用的 idea

1. **alpha/beta 阈值半监督对比学习**: 用已有表征的相似度自动构造正负样本,避免人工标注。可迁移到任何需要跨模态对齐但缺少配对标签的场景 (如 text → emotion、text → speaker style)
2. **VQ 正则化 style embedding**: 对连续 style embedding 做 VQ 再注入 TTS,兼顾表达多样性和生成稳定性。product quantization (64*32) 的设计值得参考
3. **Pre-training + context-aware fine-tuning 两阶段策略**: 当有上下文标注的数据远少于无标注数据时,先在大量无标注数据上预训练,再在小量有标注数据上微调,可迁移到其他需要上下文的任务

---

检索命中: [[ProsodyModeling]]✓, [[GlobalStyleTokens]][待确认], [[StyleTransferinTTS]][待确认], [[VITS]][待确认], [[HuBERT]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认] | 过滤: 无 | 未命中但可能相关: 无
