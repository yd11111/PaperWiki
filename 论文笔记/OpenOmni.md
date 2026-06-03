---
type: paper
tier: deep
title: "OpenOmni: Large Language Models Pivot Zero-shot Omnimodal Alignment across Language with Real-time Self-Aware Emotional Speech Synthesis"
arxiv_id: "2501.04561"
source: "Sources/OpenOmni.pdf"
authors: [Run Luo, Ting-En Lin, Haonan Zhang, Yuchuan Wu, Xiong Liu, Min Yang, Yongbin Li, Longze Chen, Jiaming Li, Lei Zhang, Yangyi Chen, Hamid Alinejad-Rokny, Fei Huang]
year: 2025
venue: "arXiv"
tags: [omnimodal, speech-LM, emotional-TTS, DPO, CTC, multimodal-alignment, real-time, zero-shot]
concepts: ["[[Speech Language Model]]", "[[Emotion Control in TTS]]", "[[Speech Tokenizer]]", "[[Speech-LLM Integration Taxonomy]]", "[[Modality Adaptation for Speech LLM]]", "[[Non-autoregressive TTS]]", "[[Streaming Spoken Dialogue]]", "[[Differentiable Reward Optimization]]", "[[Speech-Text Alignment]]"]
models: ["[[模型库/Whisper|Whisper]]", "[[模型库/CosyVoice|CosyVoice]]"]
tasks: [omnimodal-understanding, speech-to-text, text-to-speech, emotional-speech-synthesis, image-text-QA]
datasets: [OmniBench, MMBench, MMStar, HallusionBench, MathVista, MMMU, AI2D, RealWorldQA, LibriSpeech, AIShell-2, AV-Odyssey-Bench, EO2S-9K, O2S-300K]
kb_context_sources: 2
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[Speech Language Model]], [[Speech Tokenizer]]; 4 个待确认页参考)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: OpenOmni 属于 Omnimodal Language Model (OLLM) 范畴, 即 [[Speech Language Model]] 演进中的 VITA/MiniCPM-o 阶段 — 同时处理 text+image+speech 的多模态模型。在 [[Speech-LLM Integration Taxonomy]] 中, 其输入侧使用 latent-representation-based 集成 (Whisper encoder + 语音投影器 → LLM), 输出侧使用 audio-token-based 集成 (LLM hidden states → CTC speech decoder → discrete units → vocoder)。

**已有认知**: 
- Speech LLM 的三大组件 (tokenizer + LM + vocoder) 已由 Cui et al. (2024) 系统化; OpenOmni 遵循此范式但增加了 image encoder 成为 omnimodal [§1]
- [[Emotion Control in TTS]] 中 Emo-DPO (Gao et al., 2024) 已探索用 DPO 优化情感语音 [待确认]; OpenOmni 的 DEPO 将 DPO 从独立 TTS 扩展到 omnimodal 模型的 speech decoder 
- [[Streaming Spoken Dialogue]] 中 LLaMA-Omni 已使用 NAR CTC decoder 实现流式语音生成 [待确认]; OpenOmni 的 NAR 模式采用相似架构但增加了 MOE 稳定层和 AR 模式切换
- [[Modality Adaptation for Speech LLM]] 中 convolutional downsampling 是最基础的适配策略 [待确认]; OpenOmni 使用投影器 (projector) 对齐视觉/语音和 LLM embedding space

**创新判断**: 核心创新是 **language-as-pivot 的零样本跨模态对齐** — 利用 LLM 内部表征的泛化能力, 分别做 speech-text 和 image-text 对齐后, 隐式获得 speech-image 对齐, 避免依赖稀缺的三模态数据。这与 VITA 等依赖三模态数据的路线形成对比。第二个创新是将 DPO 应用于 CTC speech decoder 的情感注入。

检索命中: [[Speech Language Model]]✓, [[Speech Tokenizer]]✓ | 过滤: [[Emotion Control in TTS]](pending-review), [[Speech-LLM Integration Taxonomy]](pending-review), [[Modality Adaptation for Speech LLM]](pending-review), [[Streaming Spoken Dialogue]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用语言作为 pivot 分阶段做 speech-text 和 image-text 对齐, 隐式实现零样本三模态对齐; 加轻量 CTC speech decoder + DPO 实现实时情感语音
> - **路线**: Speech(Whisper) / Image(CLIP) → Projectors → Qwen2.5-7B LLM → Text output + Speech decoder(MOE+Qwen2.5-0.5B+CTC) → Vocoder → Speech
> - **指标**: OmniBench 37.40 (vs VITA 33.45, 7B vs 7x8B, 1.8M vs 5M data) [Table 2]; Emotion accuracy 67.9% (+7.6% over w/o DPO) [Table 5]; NAR: 30s speech <1s [§4.4]
> - **可借鉴**: (1) language-pivot 分阶段对齐策略可用于任何多模态扩展场景 — 只要两对模态各自与语言对齐, 第三对模态对齐可隐式获得; (2) MOE 层稳定 CTC 训练的经验 (无 MOE 时训练失败); (3) text-guided feature fusion 纠正 LLM 输出错误对 speech decoder 的干扰
> - **局限**: (1) 未探索更大规模三模态数据的潜力 [§5 Limitations]; (2) NAR CTC 训练难度仍高于 AR, MOE 是工程 patch 而非根本解决方案; (3) 情感控制依赖 CosyVoice 生成的合成偏好数据, 非真人标注; (4) 未开源训练代码和数据

## 核心问题

本文要解决的核心问题是: **如何在缺乏高质量三模态 (image-text-speech) 数据的条件下, 构建一个能同时理解图像、文本、语音并实时生成情感语音的开源 omnimodal 模型?**

这分解为两个子问题:
1. **数据稀缺**: 高质量 image-text-speech 三元组极少, 现有开源 OLLM (如 VITA) 依赖大量三模态数据且使用更大模型 (7x8B), 效率低
2. **实时情感语音**: 现有 OLLM 要么依赖外部 TTS (非端到端), 要么生成的语音缺乏情感一致性

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

OpenOmni 由四个核心模块组成 [§3.1, Fig 1]:
1. **LLM backbone**: Qwen2.5-7B-Instruct, 负责跨模态推理和文本生成
2. **Image encoder**: CLIP-ViT-L [§4.1], 提取视觉特征
3. **Speech encoder**: Whisper-large-v3 [§4.1], 提取语音特征
4. **Speech decoder**: 基于 Qwen2.5-0.5B-Instruct 的轻量解码器 [§4.1], 将 LLM 输出转为语音

各模块通过投影器 (projector) 连接: image-text projector 对齐视觉与 LLM, speech-text projector 对齐语音与 LLM [§3.1]。

### 关键设计选择

#### 1. Language-as-Pivot 分阶段对齐

**WHY**: 高质量三模态数据集 Do = {(xV, xS, y)} 极度稀缺, 但双模态数据 (image-text Di2t 和 speech-text Ds2t) 非常充足 (M >> K, N >> K) [论文原文, §3.1]。

**HOW**: 分两步完成, 文本作为共享 pivot:
1. **Stage 1 (Speech-Text Alignment)**: 训练 speech encoder + LLM 使语音和文本的 LLM 内部表征对齐: fφ(y) ≈ fφ(xS), 即同一语义的文本和语音在 LLM 内部空间中距离很近 [§3.2, Eq.1]
2. **Stage 2-3 (Image-Text Alignment)**: 在已有 speech-text 对齐的 LLM 上继续训练 image-text 对齐 [§3.3, Eq.2-3]

**零样本涌现**: 由于 LLM 内部表征对齐, 当给定图像 xV 和语音指令 xSq 时, 模型可以准确响应 — 即使从未在三模态数据上训练过。数学上: pθ(ya|EV(xV), DLLM(yq)) ≈ pθ(ya|EV(xV), DLLM(ES(xSq))), 因为 DLLM(yq) ≈ DLLM(ES(xSq)) [论文原文, §3.3]。[agent 解读] 这本质上是利用了 LLM 作为 "语义瓶颈" — 不同模态的信息在通过 LLM 时被投射到同一语义空间, 因此 LLM 无需显式见过三模态组合也能处理。

**为什么选 language 而非其他模态做 pivot**: [agent 解读] 因为 LLM 的文本 embedding 空间是预训练最充分的, 具有最强的语义组织结构; 其他模态通过与文本对齐, 间接获得了 LLM 的语义组织能力。

#### 2. MOE + CTC 的 NAR Speech Decoder

**WHY**: 为实现实时交互, 需要并行生成语音 [论文原文, §3.4]。AR 模式质量好但慢, NAR (CTC) 可实时但训练不稳定。

**HOW** [§3.4, Fig 3]:
1. LLM 输出 hidden states O = DLLM(xV, yq) = [h1,...,hN]
2. MOE 层 (4 experts) 将 O 映射为 C = [c1,...,cN]
3. Upsample 扩展序列长度
4. 小型 decoder-only Transformer (Qwen2.5-0.5B) 生成 output hidden states O = [o1,...,oM]
5. CTC loss 对齐 O 与 discrete speech unit 序列 [Eq.4]

**MOE 的关键作用**: 论文明确指出 "The speech decoder can hardly be trained successfully without this layer" [论文原文, §3.4]。[agent 解读] 这是因为中英双语 CTC 训练存在语言间冲突 — MOE 的不同 experts 可以隐式地处理不同语言/长度的样本, 类似于条件路由。消融实验 (Tab. 6) 证实 experts=1 时 WER 极高 (CER 113.6/129.7 on WeNetSpeech), experts=4 时显著改善 (8.5/8.4)。

**Text-Guided Feature Fusion**: LLM 在推理时可能输出错误文本, 导致 speech decoder 输入的条件特征语义错误。训练时将 LLM 输出特征与正确文本 embedding 融合后送入 speech decoder, 确保语义对齐正确 [论文原文, Appendix §9, Fig 4]。

#### 3. Direct Emotional Preference Optimization (DEPO)

**WHY**: 为让模型根据对话上下文自动生成情感一致的语音, 而不需要额外的情感控制模块或提示 [论文原文, §3.4]。

**HOW**: 基于 DPO [Eq.5-6] 和 Plutchik 情感模型, 构建偏好对:
- **正样本 yw**: 与对话历史情感一致的语音 (CosyVoice emotion-conditioned 生成)
- **负样本 yl**: 情感中性的语音 (CosyVoice 无条件生成)
- 使用 CTC 的 log 概率 (log π(y|x), Eq.7) 替代标准 DPO 中的序列概率
- Reference model πref 是 DPO 前的 speech decoder, 训练中固定 [§3.4]

**情感覆盖**: 9 种情感 (基于 Plutchik 的 8 种 + neutral), 中英双语各半, 共 9K 样本 [§4.1]。稀缺情感 (anger, sadness) 通过 GPT-4o-mini 增强 [§4.1]。

### 训练策略

5 阶段训练, 渐进式解锁能力 [Tab.1]:

| Stage | 任务 | Loss | LLM冻结? | 训练模块 | 成本 (GPU·H) |
|-------|------|------|----------|----------|-------------|
| I (1-1) | Speech-text alignment | Ls2t | Yes | Speech encoder + projector | 40 |
| II (2-1) | Image-text pretraining | Li2t | Yes | Image encoder + projector | 80 |
| III (2-2) | Image-text instruction tuning | LIi2t | **No** | All (含 LLM) | 500 |
| IV (3-1) | Speech generation (CTC) | Lctc | Yes | Speech decoder + MOE | 36 |
| V (3-2) | Emotional DPO | Ldpo | Yes | Speech decoder | 8 |

[论文原文, Tab.1]

**关键细节**:
- Stage I-II 冻结 LLM, 防止短文本 image-text pairs 损害 LLM 通用能力 [§3.3]
- Stage III 解冻 LLM 进行全参数 instruction tuning, 同时混入少量 TTS 转换的三模态数据增强 omnimodal 能力 [§3.3]
- Stage IV-V 再次冻结 LLM, 只训练 speech decoder — 这意味着 speech decoder 完全依赖 LLM 输出, 不会反向影响 LLM [agent 解读]
- 总计算成本 ≈ 664 GPU·H (8xA100), 相比 VITA 显著低廉 [§4.1]

### 数据构建

| 数据集 | 用途 | 规模 | 来源 |
|--------|------|------|------|
| OpenOmni-1-1 | Speech-text alignment | 1600h | WeNetSpeech + LibriSpeech + AIShell-4 + 80K MMEvol [§4.1] |
| OpenOmni-2-1 | Image-text pretraining | 595K pairs | LLaVA-Pretrain [§4.1] |
| OpenOmni-2-2 | Image-text IT + speech | 1.8M | MMEvol + O2S-300K [Appendix §6] |
| O2S-300K | Speech generation | 8000h, 300K pairs | MMEvol + UltraChat → CosyVoice 合成 [§4.1] |
| EO2S-9K | Emotion DPO | 9K pairs | MMEvol → Qwen2-72B 情感分类 → CosyVoice 条件/无条件生成 [§4.1] |

[§4.1, Appendix §6]

## 实验

### Omnimodal 理解 (OmniBench)

| 指标 | OpenOmni (7B) | VITA (7x8B) | AnyGPT (7B) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Overall | **37.40** | 33.45 | 7.01 | OmniBench | [Table 2] |
| Contextual & Environmental | **44.08** | 36.02 | 4.74 | OmniBench | [Table 2] |
| Identity & Relationship | **48.23** | 43.97 | 5.67 | OmniBench | [Table 2] |

OpenOmni 用 7B + 1.8M 数据超过 VITA 的 7x8B + 5M 数据, 验证了 language-pivot 对齐的数据效率 [§4.2]。

### 视觉-语言 (多 benchmark)

| 指标 | OpenOmni (7B) | VITA (8x7B) | MMEvol (7B) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MMBench-CN | **76.4** | — | 71.4 | MMBench | [Table 3] |
| HallusionBench | **44.2** | — | 42.9 | HallusionBench | [Table 3] |
| MMStar | 51.8 | — | 51.6 | MMStar | [Table 3] |
| MathVistaM | **52.7** | 44.9 | 52.4 | MathVista | [Table 3] |

[§4.3, Table 3]

### 语音理解与生成 (ASR/T2S)

| 指标 | OpenOmni (S2T) | OpenOmni (T2S) | VITA (S2T) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| AIShell-2 CER | 6.8 (S2T) | 6.9 (T2S) | 10.3 (S2T) | AIShell-2 | [Table 4] |
| LibriSpeech Test-clean WER | 3.1 (S2T) | 3.4 (T2S) | — | LibriSpeech | [Table 4] |
| LibriSpeech Test-other WER | 4.5 (S2T) | 5.6 (T2S) | — | LibriSpeech | [Table 4] |

OpenOmni 在 S2T 和 T2S 双任务上均优于其他 OLLM [§4.4, Table 4]。

### 情感语音生成

| 指标 | w/o DPO (ZH) | w/ DPO (ZH) | w/o DPO (EN) | w/ DPO (EN) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Overall Emotion Acc | 57.9% | **70.4%** | 62.6% | **65.4%** | EO2S-9K test | [Table 5] |
| Fearful | 54.8% | **78.4%** | 68.7% | **70.4%** | EO2S-9K test | [Table 5] |
| Sad | 60.2% | **90.7%** | 75.6% | **77.3%** | EO2S-9K test | [Table 5] |
| Surprised | 23.7% | **29.8%** | 7.5% | **13.9%** | EO2S-9K test | [Table 5] |

DEPO 平均提升 7.6% 情感准确率 (60.3% → 67.9%), 中文提升更显著 (+12.5%) [§4.5, Table 5]。

### 消融: MOE 和 Decoder 层数

| Layers | Experts | WeNetSpeech Test-Net CER | LibriSpeech Test-clean WER | 出处 |
| --- | --- | --- | --- | --- |
| 2 | 1 | 113.6 | 87.8 | [Table 6] |
| 2 | 2 | 16.7 | 10.7 | [Table 6] |
| 2 | 4 | 8.5 | 4.2 | [Table 6] |
| 4 | 4 | 7.3 | **3.8** | [Table 6] |
| 6 | 4 | **6.4** | 4.1 | [Table 6] |

- Expert 数量对训练稳定性至关重要: experts=1 完全失败, experts=4 显著改善 [§8, Table 6]
- 中文偏好更多层 (6 层最优), 英文偏好 4 层 [§8]

## 局限性

1. **三模态数据潜力未挖掘**: 论文承认未探索更大规模三模态数据是否能进一步提升性能 [§5 Limitations]
2. **CTC 训练脆弱性**: MOE 是工程 patch, CTC loss 在混合语言场景下的不稳定性是根本问题, 论文也承认 "finding an effective trade-off remains a significant challenge" [§5]
3. **合成偏好数据**: 情感 DPO 的正负样本均由 CosyVoice 合成, 质量受限于 CosyVoice 的情感控制能力, 非真人标注; 部分稀缺情感通过 GPT-4o-mini 增强文本, 引入分布偏差 [§4.1]
4. **情感覆盖不均**: Surprised 的改善幅度远小于其他情感 (EN: 7.5% → 13.9%), 说明 DPO 对某些情感效果有限 [Table 5]
5. **评估局限**: T2S 评估中用 Whisper 识别合成语音再算 WER, 中文部分甚至用 OpenOmni 自己做 ASR (self-recognition), 存在评估偏差风险 [§4.4]
6. **开源程度**: 论文标题含 "Open" 但代码和数据开源程度不明; arXiv v1 未提供 GitHub 链接

## 点评

**优势**:
- Language-pivot 思路简洁且有效, 数据效率显著: 7B 模型 + 1.8M 数据超越 7x8B + 5M 数据的 VITA, 验证了 "不需要三模态数据也能做 omnimodal" 的直觉
- 同时支持 AR 和 NAR 两种 speech decoder 模式, 为部署提供质量-速度的灵活权衡
- MOE 稳定 CTC 训练的发现虽然不算深刻, 但对后续工作有直接工程价值

**不足**:
- "Zero-shot omnimodal alignment" 的说法略有 overclaim — 实际在 Stage III 混入了少量 TTS 转换的三模态数据 [§3.3], 严格来说是 "near zero-shot"
- 情感评估只用 Emotion2Vec 分类器验证准确率, 没有人类主观评估 (MOS), 对合成语音的自然度和情感表现力无从判断
- 与 LLaMA-Omni 的 CTC decoder 方案高度相似 (同为 LLM hidden → NAR CTC → speech units), 但论文对此差异讨论不足

**总体**: 一篇执行扎实的系统性工作, language-pivot 对齐和 DEPO 都有明确的创新点, 实验覆盖全面。主要贡献在于证明了低资源 omnimodal 对齐的可行性路线, 以及将 DPO 扩展到 CTC speech decoder 的情感注入。

## 可复用的 idea

1. **Language-pivot 跨模态对齐**: 当需要对齐 N 个模态但缺乏 N 元组数据时, 选一个语义能力最强的模态 (如文本 LLM) 作为 pivot, 分别做 N-1 次双模态对齐, 利用 pivot 模态的内部表征泛化实现隐式全模态对齐。可推广到视频、触觉等新模态的接入。

2. **MOE 稳定 CTC 训练**: 在 CTC loss 训练中遇到多语言/多任务冲突时, 在 CTC decoder 前加 MOE routing 层, 让不同 experts 隐式分担不同条件, 可大幅改善训练稳定性。消融清晰 (experts 1→4: CER 113.6→8.5)。

3. **Text-guided feature fusion 纠正 LLM 输出错误**: 当 speech decoder 依赖 LLM hidden states 作为条件时, LLM 可能生成错误文本 → 错误条件特征。训练时用正确文本 embedding 融合/替代 LLM 输出, 是一种廉价的条件纠正方案。

4. **CTC + DPO 情感注入**: 将 DPO 应用于 CTC 的 token-level 概率 (Eq.7), 用合成的 emotion-conditioned vs neutral 语音构建偏好对, 是一种不需要真人标注就能实现情感控制的路线。

> [!review] 审阅: pass (2026-06-03)
> 4 low issues, 0 high/medium. 详见 `_review/OpenOmni-review.yml`
> - [low] frontmatter models 未列对比 baseline (影响不大, baseline 不在模型库)
> - [low] 总计算成本 664 GPU·H 为 agent 加总推算
> - [low] MOE expert routing 行为分析为 agent 推断 (已标注)
> - [low] text-guided feature fusion 描述可细化
