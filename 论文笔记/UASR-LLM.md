---
type: paper
tier: deep
title: "Adapting Speech Foundation Models for Unified Multimodal Speech Recognition with Large Language Models"
arxiv_id: "2510.22961"
source: "Sources/AdaptingSpeechFoundationModels.pdf"
authors: [Jing-Xuan Zhang, Genshun Wan, Jin Li, Jianqing Gao, Duo Zhao, Zhen-Hua Ling]
year: 2026
venue: "IEEE TNNLS (submitted)"
tags: [audio-visual-speech-recognition, speech-foundation-model, visual-speech-recognition, lipreading, LLM, knowledge-distillation, LoRA, multimodal-fusion, unified-model, parameter-efficient]
concepts: ["[[Self-SupervisedSpeechRepresentation]]", "[[ModalityAdaptationforSpeechLLM]]", "[[LLM-enhancedASR]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: ["[[模型库/WavLM|WavLM]]", "[[模型库/HuBERT|HuBERT]]", "[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个待确认实体页: [[模型库/WavLM|WavLM]], [[模型库/HuBERT|HuBERT]], [[概念库/LLM-enhancedASR|LLM-enhanced ASR]], [[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation for Speech LLM]], [[概念库/Self-SupervisedSpeechRepresentation|Self-Supervised Speech Representation]], [[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLM Integration Taxonomy]])
> [待确认] 所有命中页均为 pending-review,仅供参考。
> 检索命中: WavLM, HuBERT, LLM-enhanced ASR, Modality Adaptation for Speech LLM, Self-Supervised Speech Representation, Speech-LLM Integration Taxonomy | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文属于 latent-representation-based Speech-LLM Integration 路线 — 用连续语音表征而非离散 token 作为 LLM 输入。与 KB 中记录的 Modality Adaptation 三种方法 (Conv Downsampling / CTC Compression / Q-Former) 不同,本文用两层 FFN adaptor 不做帧压缩,直接送入 LLM。核心差异在于本文的目标不是通用音频理解,而是在 frozen SFM 中注入视觉信息,使单一模型同时支持 ASR + VSR + AVSR 三个任务 (即 Unified Speech Recognition, USR)。

**已有认知**: WavLM 通过 masked speech denoising + gated relative PE + 94k hr 多样化数据,成为 full-stack SSL 语音模型; HuBERT 通过离线 k-means + masked prediction 建立了自监督语音表征范式。KB 记录 WavLM 的层级信息分离 (bottom=speaker, top=content) 为本文用 WavLM 做 cross-modal attention 的 query 提供了理论基础。Modality Adaptation 概念页记录了 adapter 训练的两阶段策略 (Wu et al. 2023: 先稳定 encoder 再启动 PEFT),本文采用类似但更特化的两阶段: 先做 visual injection pretraining (知识蒸馏),再做 speech recognition finetuning (CE + CTC)。

**创新判断**: 相比 KB 中记录的 latent-representation 路线代表系统 (SALMONN, Qwen-Audio 等主要做 audio understanding),本文专注于将已有 SFM 改造为多模态 backbone,而非从头训练新编码器。与同领域 Llama-AVSR (分开的音频/视觉编码器 + temporal concatenation) 和 MMS-Llama (Q-Former adapter) 相比,本文的视觉注入模块 (VIM) 在 SFM 的每一层都融合视觉信息,实现更深度的跨模态交互。

> [!summary] 速查
> - **一句话**: 通过在 frozen SFM 每层插入视觉注入模块 + LLM decoder,实现单一模型统一 ASR/VSR/AVSR,在可比数据量下 LRS3 上超越各 baseline
> - **路线**: Lip video → DistillAV visual encoder → Visual Injection Modules (cross-attn) injected into each frozen SFM block → 2-layer FFN adaptor → LLM (Qwen 2.5-7B + LoRA) → text
> - **指标**: LRS3 clean (1759h/1759h): VSR 20.9% WER, ASR 0.84% WER, AVSR 0.69% WER [Table II]; noisy AVSR 3.2% avg WER (433h/433h, UASR-LLM-L) [Table II]
> - **可借鉴**: (1) tanh gating 初始化为 0 的注入模块设计,保证与原始 frozen model 的初始等价; (2) visual injection pretraining 用 clean audio 表征做 teacher 蒸馏,让 corrupted AV input 学会对齐到 clean 空间; (3) 辅助 CTC loss 用独立的 1k vocabulary 而非 LLM 原生 152k vocabulary,解决收敛困难
> - **局限**: 不做帧压缩 (50 fps 直接送 LLM),推理效率低,作者承认这是未来方向 [§V]; 视觉编码器依赖自有 DistillAV 预训练,非公开通用方案; 仅在英语 LRS3 上评测; VSR 仍落后于使用 100k hr 数据的 LP-Conformer [Table II]

## 核心问题

现有 speech foundation models (WavLM, Whisper) 在纯音频 ASR 上表现出色,但无法直接处理视觉模态 (唇读)。将 ASR、VSR、AVSR 统一到单一模型中面临两个挑战: (1) 如何在不破坏 SFM 预训练知识的前提下注入视觉信息; (2) 如何让单一模型在三个任务间共享参数而不互相干扰 [§I]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UASR-LLM 由三个核心组件构成 [§III-A, Fig 1]:

1. **Visual Encoder** (frozen): DistillAV 的视觉组件,提取唇部视频表征 $H_v \in \mathbb{R}^{T_v \times D_v}$ [§III-A1]
2. **SFM-AV** (SFM frozen + VIM trainable): 在 WavLM-large 的每一层前插入 Visual Injection Module (VIM),将视觉信息注入到音频表征中 [§III-A2]
3. **LLM Decoder** (LoRA finetuned): Qwen 2.5-7B 接收 SFM-AV 输出经 adaptor 投影的连续 token,根据 instruction prompt 生成转写文本 [§III-A3]

关键路径选择:
- **VSR/AVSR**: 经过 VIM 的 SFM-AV → adaptor → LLM
- **ASR**: 绕过 VIM,直接用原始 SFM → adaptor → LLM [Fig 3]

[agent 解读] ASR 时绕过 VIM 的设计保证了纯音频性能不受视觉注入的干扰,同时意味着 ASR 路径本质上就是标准的 "frozen SFM + adaptor + LoRA LLM" 方案。这也解释了为什么 ablation 中去掉 visual injection pretraining 反而 ASR 更好 — 共享参数在 VSR/AVSR 方向的优化会微弱干扰 ASR [Table IV]。

### 关键设计选择

**Visual Injection Module (VIM)** [§III-A2, Eq. 1-2]:
- 用 SFM 的 audio 表征 $H_i^a$ 做 query,视觉表征 $H_v$ 做 key/value,通过 cross-attention 融合
- 加入 relative position embedding $D_{av}$,编码音频帧与视频帧之间的真实时间间隔 (audio 50fps vs video 25fps) [§IV-A]
- 两个 tanh gating ($g_i, g_i'$) 初始化为 0 → 训练开始时 VIM 输出为 0,SFM 行为与原始 frozen model 完全一致 [§III-A2]

[论文原文] "It provides an initialization state identical to the original SFMs, which helps stabilize the training." [§III-A2]

[agent 解读] tanh-gating-zero-init 是 Whisper-Flamingo 提出的技术,本文沿用。这种设计的价值在于: 训练初期新注入模块不会干扰 SFM 的预训练表征,让优化从稳定基线出发逐步引入视觉信息。

**为什么不压缩帧** [§III-A3]:
- 本文不使用 frame reduction (如 Llama-AVSR 的帧压缩或 MMS-Llama 的 Q-Former)
- [论文原文] "This design choice reflects our primary focus on improving recognition performance rather than computational efficiency." [§III-A3]
- [agent 解读] 不压缩帧意味着 50fps 的音频表征直接送入 LLM,一秒音频 = 50 个 token。对于 LRS3 平均几秒的短句,这是可接受的;但对于更长音频场景,这会成为严重的效率瓶颈。

**LLM 的 CTC 辅助 loss** [§III-B2, Eq. 7]:
- 在 SFM-AV 输出接一个额外的 FF 投影层,训练 CTC loss
- [论文原文] "convergence proved challenging due to the large vocabulary size (e.g., 152k tokens for Qwen 2.5). Therefore, we adopted a separate, smaller 1k vocabulary specifically for the CTC loss, which significantly improved training stability." [§III-B2]
- [agent 解读] 用 1k subword CTC 而非 LLM 的 152k vocabulary 是一个实用的工程选择。CTC loss 的作用是直接优化 encoder 表征质量 (CE loss 的梯度需要穿过 LLM 回传),而 1k vocabulary 足以提供帧级对齐信号。

### 训练策略

**Stage 1: Visual Injection Pretraining** [§III-B1, Fig 2]:
- 目标: 让 SFM-AV 在接收视觉/音视频输入时产生接近 clean audio SFM 表征的输出
- Teacher: frozen SFM 处理 clean audio,取最后 k 层 instance-normalized 平均作为目标 $H^T$ [Eq. 3-4]
- Student: SFM-AV 处理 corrupted inputs (noise + span masking + modality dropout)
- 蒸馏 loss: L1 distance + negative cosine similarity [Eq. 6]
- Modality dropout: 概率 $p_v=0.5$ 只给 video (audio 全零), $1-p_v$ 给 audio+video [Eq. 5]
- 数据: LRS3 433h 或 LRS3+VoxCeleb2 1759h

[论文原文] 为什么用 corrupted inputs: 对输入施加噪声混合、span masking 和模态 dropout,是为了 "promote effective audio-visual fusion when predicting clean audio representations" [§III-B1]。

[agent 解读] 这个预训练设计的核心洞察是: 让模型在 audio 被破坏时学会从 visual 信息中补偿,在 visual 缺失时学会从 noisy audio 中提取信号。目标对齐到 clean audio 空间确保了 SFM-AV 的输出与原始 SFM 表征兼容。

**Stage 2: Speech Recognition Finetuning** [§III-B2, Fig 3]:
- 连接 SFM-AV → adaptor → LLM (LoRA)
- 三路 modality dropout: $p_v'$ (video-only), $p_a'$ (audio-only), $1-p_a'-p_v'$ (both)
- 实际设置: $p_v'=0.5, p_a'=0.25$ [§IV-A]
- Loss: $L = L_{CE} + 0.25 \cdot L_{CTC}$ [Eq. 7]
- 训练规模: 30k/60k/120k updates for 30h/433h/1759h finetuning data [§IV-A]

**Frozen 模块**: SFM 和 visual encoder 全程 frozen; LLM 通过 LoRA (rank 16, 12M params) 高效微调 [§IV-A]。

**参数量分布** [§IV-A]:
| 组件 | 参数量 | 状态 |
| --- | --- | --- |
| WavLM-large (SFM) | 317M | frozen |
| DistillAV visual encoder (Base/Large) | 103M/325M | frozen |
| Visual injection modules | ~58M | trainable |
| FFN adaptor | ~10M | trainable |
| LoRA weights | ~12M | trainable |
| **总可训练参数** | **~80M** | — |

## 实验

### 主要结果 (LRS3 benchmark)

| 指标 | 本文 (UASR-LLM-L) | Baseline (USR) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| VSR WER (30h finetune, 433h pretrain) | 30.6% | 36.0% | LRS3 test | [Table I] |
| ASR WER (30h finetune, 433h pretrain) | 2.2% | 3.2% | LRS3 test | [Table I] |
| AVSR WER (30h finetune, 433h pretrain) | 2.0% | 3.0% | LRS3 test | [Table I] |
| VSR WER (1759h finetune) | 20.9% | 21.5% (USR) | LRS3 test | [Table II] |
| ASR WER (1759h finetune) | 0.84% | 1.2% (USR) | LRS3 test | [Table II] |
| AVSR WER (1759h finetune) | 0.69% | 1.1% (USR) | LRS3 test | [Table II] |
| Noisy AVSR avg WER (433h finetune, 433h pretrain) | 3.2% | 9.6% (DistillAV-B) | LRS3 noisy test | [Table II] |

### 跨 SFM / LLM 泛化 (433h pretrain, 433h finetune)

| SFM | LLM | VSR | ASR | AVSR | 出处 |
| --- | --- | --- | --- | --- | --- |
| WavLM | Qwen 2.5-7B | 23.9% | 0.86% | 0.78% | [Table III] |
| Conformer-Speech | Qwen 2.5-7B | 24.5% | 0.76% | 0.89% | [Table III] |
| Whisper | Qwen 2.5-7B | 24.6% | 1.2% | 0.70% | [Table III] |
| WavLM | ChatGLM v3-6B | 24.3% | 1.1% | 0.95% | [Table III] |

### 关键 Ablation 发现

**Visual injection pretraining (VIPT)** [Table IV]:
- 去掉 VIPT (0h) → VSR 下降 (30.1% vs 27.3%), noisy AVSR 显著下降 (6.5% vs 3.5%), 但 ASR 反而最好 (2.2% vs 2.3%)
- [论文原文] "the audio processing capability of SFM itself remained unaffected by the visual injection pretraining stage. This phenomenon may be attributed to our parameter sharing strategy for USR in LLMs" [§IV-E1]
- [agent 解读] 这揭示了 USR 的固有 trade-off: 共享 LoRA 参数同时优化三个任务,VSR/AVSR 的增强会微弱影响 ASR。

**参数共享 vs 独立训练** [Table V]:
- 独立训练 (w/o shared): ASR 更好 (0.76% vs 0.86%), VSR 更差 (24.0% vs 23.9%)
- [agent 解读] 统一模型的 VSR 优势可能来自 cross-task 知识迁移 — 音频的语言模型知识迁移到唇读任务。

**LLM vs 传统 Transformer decoder** [Table V]:
- 无 LLM (6层 Transformer decoder + cross-attention): 所有指标大幅下降,尤其 ASR (2.6% vs 0.86%)
- [论文原文] "the strong contextual and language modeling capabilities of LLMs, learned from large text corpora, can be effectively transferred to enhance speech recognition performance" [§IV-E3]

## 局限性

1. **推理效率低**: 50fps 音频表征不压缩直接送 LLM,每秒音频 50 个 token;作者承认未来需要 dynamic pooling 或轻量 LLM [§V]
2. **数据规模限制**: 仅用 ~1.8k hr 音视频数据,VSR 仍落后于使用 100k hr 数据的 LP-Conformer (12.8% vs 20.9% WER) [Table II]
3. **视觉编码器依赖**: 依赖自有 DistillAV 预训练,不是通用开源方案;视觉编码器性能直接影响系统效果
4. **仅英语评测**: 全部实验在英语 LRS3 数据集上,多语言泛化未验证
5. **ASR-USR trade-off**: 统一模型的 ASR 性能略差于独立 ASR 模型 (共享参数干扰) [Table IV, V]
6. **无代码/模型开源信息**: 论文未提及代码开源

## 点评

本文的核心价值在于**证明了 frozen SFM 可以被高效地"改造"为多模态 backbone**: 仅通过 58M 的 VIM + 10M adaptor + 12M LoRA (~80M trainable params),在 frozen WavLM (317M) + frozen visual encoder (103-325M) + frozen LLM (7B) 上实现了统一的 ASR/VSR/AVSR。

**方法的设计逻辑清晰**: (1) VIM 的 zero-init tanh gating 确保训练从稳定基线出发; (2) visual injection pretraining 用知识蒸馏让 SFM-AV 学会在表征空间对齐视觉信息; (3) instruction-guided modality dropout 让单一模型覆盖三种输入模态。每一步的设计选择都有明确的 motivation。

**与 TTS 领域的连接点**: 虽然本文聚焦 ASR/VSR/AVSR 而非 TTS,但 (1) VIM 的 cross-attention + zero-init gating 思路可迁移到任何需要在 frozen backbone 中注入新模态信息的场景; (2) 两阶段训练 (先蒸馏对齐再端到端微调) 是 modality adaptation 的通用范式; (3) CTC 辅助 loss 用独立 vocabulary 的工程技巧值得记录。

**弱点**: 最大的遗憾是推理效率 — 不做帧压缩直接送 LLM 是一个刻意的简化,限制了实际部署价值。此外,1.8k hr 的数据量远小于工业级系统 (Google LP-Conformer 用 100k hr),在公平对比中本文的数据效率优势可能更值得强调。

## 可复用的 idea

1. **Zero-init tanh gating 注入模块**: 在 frozen backbone 每层插入新模块时,将 gating 初始化为 0,保证初始行为与原模型一致。可用于任何需要在预训练模型中注入新信号的场景 (如在 frozen speech LM 中注入情感/风格信息) [§III-A2]
2. **Clean-target 知识蒸馏 pretraining**: 用 frozen model 处理 clean input 做 teacher,让 augmented model 处理 corrupted input 做 student。目标是在表征空间对齐,而非在 label 空间对齐。适用于任何需要给已有模型"增加新能力"的预训练阶段 [§III-B1]
3. **CTC 辅助 loss 用独立小 vocabulary**: 当 LLM 的 vocabulary 太大导致 CTC loss 不收敛时,用独立的 1k subword vocabulary 训练 CTC。CTC 的作用是给 encoder 提供帧级对齐梯度,不需要与 LLM 共享词表 [§III-B2]
4. **Instruction-guided modality dropout**: 训练时随机 drop 某一模态输入并相应修改 instruction prompt,使单一模型学会处理不同模态组合。可扩展到 multi-condition TTS (有/无 reference audio, 有/无 emotion label 等) [§III-B2]

## 审阅

> [!review] 审阅 (2026-06-08, self-review)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节含完整因果解释,WHY/HOW 覆盖充分;速查可借鉴给出 3 个具体可迁移 trick |
> | 可信赖 | pass | 所有数字标注出处 (Table I-V, Fig 1-4, §I-V);WER 方向性正确;速查指标已修正为精确配置标注 |
> | 可区分 | pass | 7 处因果解释全部标注 [论文原文] 或 [agent 解读];无断言式推断 |
> | 可定位 | pass | KB 背景给出 latent-representation 路线定位,与 Llama-AVSR / MMS-Llama 明确对比;frontmatter 字段完整 |
> | 不污染 | pass | 未修改概念页 (按用户指令);concepts/models 引用已有页面,语义正确 |
> 
> Issues: 2 (high: 0, medium: 1, low: 1)
> 
> **Medium**: 速查指标 noisy AVSR 原引用 "2.4%" 未标明配置,已修正为 "3.2% avg WER (433h/433h, UASR-LLM-L) [Table II]" (traceability-gap)
> **Low**: 速查一句话原为 "全面超越 baseline",LP-Conformer (100k hr) 仍有更好 VSR,已修正为 "在可比数据量下超越各 baseline" (overclaim)
