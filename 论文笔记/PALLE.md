---
type: paper
tier: deep
title: "Pseudo-Autoregressive Neural Codec Language Models for Efficient Zero-Shot Text-to-Speech Synthesis"
arxiv_id: "2504.10352"
source: "Sources/PANLM.pdf"
authors: [Yifan Yang, Shujie Liu, Jinyu Li, Yuxuan Hu, Haibin Wu, Hui Wang, Jianwei Yu, Lingwei Meng, Haiyang Sun, Yanqing Liu, Yan Lu, Kai Yu, Xie Chen]
year: 2025
venue: "ACM MM 2025"
tags: [TTS, zero-shot, codec-LM, pseudo-autoregressive, masked-generative, parallel-decoding, non-autoregressive]
concepts: ["[[Codec Language Model]]", "[[Masked Generative Modeling]]", "[[Non-autoregressive TTS]]", "[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Conditional Flow Matching]]", "[[Speech-Text Alignment]]"]
models: ["[[模型库/CosyVoice 2]]", "[[模型库/EnCodec]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["LibriTTS", "LibriSpeech"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[Speech Tokenizer]]✓, [[Semantic vs Acoustic Tokens]]✓, [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓, [[模型库/CosyVoice 2]]✓, [[Masked Generative Modeling]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Conditional Flow Matching]], [[Zero-shot Speech Synthesis]], [[模型库/CosyVoice 2]], [[Codec Language Model]] | 过滤: [[Masked Generative Modeling]](pending-review), [[Non-autoregressive TTS]](pending-review), [[Speech-Text Alignment]](pending-review), [[Codec Language Model]](pending-review) | 未命中但可能相关: 无

**谱系定位**: PALLE 处于 zero-shot TTS 中 codec language model 范式的前沿,具体位于 AR 与 NAR 建模范式的交汇处。当前 codec LM TTS 形成三条路线: (1) AR 路线 (VALL-E, CosyVoice 系列) 序列生成有时序保障但推理慢; (2) NAR 路线 (MaskGCT, F5-TTS, E2 TTS) 并行生成快但缺乏时序建模导致鲁棒性问题; (3) 混合路线 (SoundStorm 的 semantic-to-acoustic 阶段用 masked generation)。PALLE 提出第四条路线 -- pseudo-autoregressive (PAR),在 masked generative transformer 中注入 span-level 时序约束,试图统一 AR 的时序建模和 NAR 的并行效率。

**已有认知**: 知识库中 [[Masked Generative Modeling]] 记录了 MaskGIT → SoundStorm → MaskGCT 的演进线,其核心是 confidence-based iterative parallel decoding; [[Codec Language Model]] 记录了从 VALL-E 到多任务 CodecLM 的发展; [[Speech Tokenizer]] 记录了从自监督到监督式 semantic token 的演进,PALLE 使用的 S3Tokenizer v2 来自 [[模型库/CosyVoice 2]]; [[Conditional Flow Matching]] 记录了 CFM 在 TTS 中作为 token-to-mel 渲染器的角色,PALLE 的 speech detokenizer 即来自 CosyVoice 2 的 CFM 模型。

**创新判断**: PALLE 的 PAR 范式是对 MaskGCT 式 NAR 建模的结构性改进 -- 不是简单加 causal mask (如 SyncSpeech),而是通过 span-level progressive commitment 在双向 transformer 中建立软时序约束,使推理步数与目标长度解耦 (O(1) vs O(T))。这是 masked generative modeling 在 TTS 中的新变体,知识库中尚无此类记录。

## 速查

> [!summary] 速查
> - **一句话**: 提出 Pseudo-Autoregressive (PAR) 建模范式,在双向 masked generative transformer 中注入 span-level 时序约束,实现固定步数的并行语音生成,兼具 AR 的时序建模和 NAR 的推理效率
> - **路线**: Text (BPE) + Speech prompt (S3Tokenizer v2, 25Hz) → PAR stage (固定 100 步,每步生成 1%T span 并 commit) → NAR refinement stage (7 步,confidence-guided re-masking) → CFM + HiFi-GAN (CosyVoice 2) → Waveform
> - **指标**: Cross-sentence WER-W 2.23% (vs MaskGCT 4.22%, E2 TTS 5.90%) / SIM-o 0.716 (vs MaskGCT 0.756) / RTF 0.06 (vs MaskGCT 0.65, F5-TTS 0.15); 仅用 LibriTTS 580h 训练 [Table 1]
> - **可借鉴**: (1) PAR 的 span-level progressive commitment -- 在 masked generation 中加入软时序约束的通用技巧; (2) 第二阶段 confidence-guided refinement 只需 7 步即可显著降低 WER; (3) 训练/推理 span 比例解耦 (训练 10%, 推理 1%) 可提升质量
> - **局限**: 仅在 LibriSpeech test-clean 评估 (英语, 干净条件); SIM-o 略低于 MaskGCT; 两阶段不能合并为单模型 (WER 恶化 20%); 未开源

## 核心问题

PALLE 要解决的核心问题是: **如何在 zero-shot TTS 中同时获得 AR 的时序建模能力和 NAR 的推理效率?**

具体来说:
1. AR 模型 (如 VALL-E) 推理步数 O(T) 随语音时长线性增长,且只能利用左侧上下文,存在 error accumulation [§1]
2. NAR 模型 (如 MaskGCT, F5-TTS) 推理效率高但缺乏时序建模,导致收敛慢、对齐不准、可懂度差 [§1]
3. 已有改进 (grouped AR, speculative decoding) 仍未脱离步数与时长线性绑定的本质 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

PALLE 是一个两阶段 codec language model [§4, Fig 2]:

**Stage 1 (PAR model)**: 一个双向 masked generative transformer,接收 padded text tokens 和 speech tokens (prompt + mask),以 pseudo-autoregressive 方式逐步生成目标 speech tokens。每步预测所有位置,但只 commit 最左侧的一个 span。

**Stage 2 (NAR refinement)**: 与 Stage 1 共享架构的另一个双向 masked generative transformer,对 Stage 1 的输出进行 confidence-guided 迭代精炼。选择最低 confidence 的 token 进行 re-mask 和 re-predict。

**Tokenizer**: 使用 CosyVoice 2 的 S3Tokenizer v2 (25Hz semantic tokens) 和 CosyVoice 2 的 CFM + HiFi-GAN 作为 speech detokenizer [§4.1]。

**Text-Speech 融合**: 采用 E2 TTS 风格的 feature-dimension fusion -- text 和 speech token 序列先 pad 到相同长度,分别 embedding 后在 feature 维度 concatenate,再过 linear projection [§4.1, Fig 3a]。[论文原文]

### 关键设计选择

**1. Pseudo-Autoregressive (PAR) 范式 [§3]**

PAR 的核心想法是: 在 NAR 的双向 masked generative transformer 中注入 span-level 的时序因果约束。[论文原文]

具体做法:
- **训练时**: 随机选择一个起始位置 s,mask 从 s 到序列末尾的所有 token,模型预测被 mask 的最左侧 k = floor(0.1T) 个 token [Eq 8-9, §4.2]。注意所有 self-attention 使用 full bidirectional attention,时序约束不来自 attention mask,而来自 loss 只计算在最左侧 span 上 [论文原文]。
- **推理时**: 从全 mask 状态开始,每步预测所有位置但只 retain 最左侧 k' = min(floor(r'T), N_left) 个 token,然后将这些 token commit 为确定值,下一步只 mask 剩余未生成的部分 [Eq 4-5, §3]。

**为什么 PAR 优于 AR**: PAR 每步 commit 的 span 覆盖目标序列的固定比例,因此推理步数是常数 O(1),不随语音时长增长。同时双向 attention 让模型在每步预测时能利用全局上下文 (已生成 + 尚未确定的位置),减少 error accumulation [§5.4]。[论文原文]

**为什么 PAR 优于 NAR**: MaskGCT 式 NAR 是 temporally unordered prediction -- 每步按 confidence 选择任意位置 unmask,没有时序约束。论文认为这导致 "poor alignment" 和 "robustness issues in high-entropy regions such as fast transitions or expressive prosody" [§2.3]。PAR 通过 span-level causal ordering 强制左→右的时序流,与语音的自然时序结构一致 [§3]。[论文原文]

**与 Grouped AR 的区别**: Grouped AR (如 VALL-E 2) 生成固定长度 span + 动态步数; PAR 生成动态长度 span + 固定步数。后者使推理复杂度与输出长度解耦 [§3]。[论文原文]

**2. 训练/推理 span 比例解耦 [§5.1, §5.5]**

训练时 span 比例 r = 0.1 (每步 10% 序列),推理时 r' = 0.01 (每步 1% 序列,共 100 步)。[agent 解读] 推理时用更小的 span 可以让每步 commit 更谨慎,虽然步数增加但每步计算是并行的,总推理时间仍远低于 AR。Fig 4 的消融实验证实了 100 步是效率-质量的最优平衡点。

**3. Confidence-guided NAR refinement (Stage 2) [§4.3]**

Stage 2 用 negative min-entropy 作为 confidence score,选择 quantile gamma = 0.05 (最低 5%) 的 token 进行 re-mask 和 re-predict,共 7 步 [§5.1]。为防止同一位置被反复修正,已更新 token 的 confidence 永久设为 1 [§4.3]。

[论文原文] 论文指出 Stage 2 虽然步数极少 (7 步),但效果显著: 从 Table 3 看,D2 vs C2 在 cross-sentence WER-W 上从 2.58% 降至 2.23% [Table 3]。

**4. Duration estimation [§4.3, Eq 11]**

采用 F5-TTS 的线性估计: T_gen = T_ref * (1 + L_gen / L_ref),基于 BPE token 长度比例 [§4.3]。消融实验表明估计时长与 GT 时长的性能差异很小,且估计时长在可懂度上甚至略优 [§5.3]。[论文原文]

### 训练策略

- **架构**: decoder-only Transformer, 12 层, 16 头, 1024 维, FFN 4096 维, + ConvNeXt V2 block (1024 维) + 卷积位置编码 (kernel 7); 总计 177M 参数 [§5.1]
- **两阶段训练**: Stage 1 训练 179k 步; Stage 2 在 Stage 1 基础上 finetune 87k 步 [§5.1]
- **硬件**: 8x V100 32GB, batch duration 600s/GPU [§5.1]
- **优化器**: ScaledAdam + Eden scheduler, peak LR 0.045 (Stage 1) / 0.005 (Stage 2) [§5.1]
- **Text tokenizer**: 2000-class BPE [§5.1]
- **推理**: Top-p sampling (p = 0.2~0.35) 略优于 greedy [§5.1]
- **联合训练失败**: 尝试将两阶段合并为 multitask 训练,但 Stage 2 loss 会恶化 Stage 1 性能,cross-sentence WER 增加 20%,论文归因于 halved parameter capacity [§5.5]。[论文原文]

## 实验

| 指标 | PALLE (est. dur) | PALLE (GT dur) | MaskGCT | F5-TTS | E2 TTS | CosyVoice 2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER-W↓ (cross-sent) | 2.23 | 2.35 | 4.22 | 2.77 | 5.90 | 3.00 | LibriSpeech test-clean | [Table 1] |
| WER-H↓ (cross-sent) | 2.83 | 2.87 | 4.47 | 2.92 | 3.80 | - | LibriSpeech test-clean | [Table 1] |
| SIM-o↑ (cross-sent) | 0.716 | 0.716 | 0.756 | 0.705 | 0.630 | 0.716 | LibriSpeech test-clean | [Table 1] |
| RTF↓ | 0.06 | 0.06 | 0.65 | 0.15 | 0.73 | 0.45 | A100 80GB | [Table 1] |
| SMOS↑ | 4.03 | 4.09 | 3.98 | 4.04 | 3.95 | - | LibriSpeech test-clean | [Table 2] |
| CMOS↑ (vs GT) | -0.15 | -0.09 | -0.25 | -0.20 | -0.41 | - | LibriSpeech test-clean | [Table 2] |
| WER-W↓ (continuation) | 2.31 | 2.31 | - | - | - | - | LibriSpeech test-clean | [Table 1] |
| SIM-o↑ (continuation) | 0.776 | 0.776 | - | - | - | - | LibriSpeech test-clean | [Table 1] |

**控制实验 (Table 3, 同条件对比)**:

| 系统 | 范式 | WER-W↓ (cross) | SIM-o↑ (cross) | RTF↓ |
| --- | --- | --- | --- | --- |
| VALL-E_stage-one | AR | 4.00 | 0.716 | 0.21 |
| MaskGCT_stage-one | NAR | 4.52 | 0.703 | 0.10 |
| PALLE_stage-one | PAR | 2.58 | 0.710 | 0.05 |
| PALLE_two-stage | PAR+NAR | 2.23 | 0.716 | 0.06 |

关键发现:
- PAR vs AR: WER-W 降低 36%, 推理快 4x, SIM 相当 [§5.4]
- PAR vs NAR: WER-W 降低 43%, 推理快 2x, SIM 略高 [§5.4]
- Stage 2 refinement: 仅 7 步即将 WER-W 从 2.58% 降至 2.23% [Table 3]
- Duration 消融: 1.1x 时长效果最佳 (略慢于 GT 语速更有利于可懂度) [Fig 5]

**注意**: PALLE 仅用 LibriTTS 580h 训练,而 MaskGCT/F5-TTS/E2 TTS 使用 Emilia 100K h (约 172 倍数据量) [Table 1]。这使 PALLE 的性能优势更为显著。

## 局限性

1. **评估范围有限**: 仅在 LibriSpeech test-clean (英语, 干净条件) 上评估,未验证多语言、噪声、表现力 [agent 解读]
2. **Speaker similarity 未达最优**: SIM-o 0.716 与 MaskGCT 的 0.756 有差距,可能因为 S3Tokenizer 丢弃了部分 speaker 信息而 MaskGCT 使用 acoustic codec [Table 1]
3. **两阶段无法合并**: 联合训练导致 cross-sentence WER 增加 20%,意味着必须维护两个独立模型 [§5.5]
4. **推理 span 比例需精细调优**: 训练 10% 但推理 1%,这种解耦虽有效但增加了超参搜索空间 [§5.1, Fig 4]
5. **未开源**: 仅提供 demo 页面,无代码/模型公开 [agent 解读]
6. **依赖外部组件**: S3Tokenizer v2 和 CFM detokenizer 均来自 CosyVoice 2,系统独立性受限 [§4.1]

## 点评

PALLE 提出的 PAR 范式是一个在概念层面有清晰创新的工作。它精准地抓住了 AR 和 NAR 的核心矛盾 -- 时序建模 vs 并行效率 -- 并给出了一个优雅的统一方案: 不是在 attention mask 上做文章 (如 SyncSpeech 的 causal mask),而是通过 loss 设计和 commit 策略在双向 transformer 中注入软时序约束。这是一种 "结构不变,行为约束" 的思路,比修改 attention 更灵活。

Table 3 的控制实验是本文最有说服力的部分: 同一 tokenizer/detokenizer/backbone 下,PAR 在 WER 和 RTF 上同时大幅优于 AR 和 NAR,排除了组件差异的干扰。

但也有值得注意的地方: (1) PALLE 使用的 S3Tokenizer v2 是 CosyVoice 2 在大规模数据上训练的监督式 semantic tokenizer,这是 SIM-o 不如使用 acoustic codec 的 MaskGCT 的可能原因,但也是 WER 大幅领先的原因之一; (2) 100 步推理虽然 RTF 极低 (0.06),但步数是固定的,对于极短语音可能存在不必要的计算; (3) 仅在 LibriSpeech 评估限制了结论的普适性。

## 可复用的 idea

1. **Span-level progressive commitment**: 在 masked generative model 中,不按 confidence 选位置 unmask,而是按时序 commit 固定比例的 span。这是一种通用的 "在 NAR 框架中注入时序结构" 的技巧,可迁移到其他序列生成任务 (如 image, video, music)
2. **训练/推理 span 比例解耦**: 训练用大 span (10%) 保证梯度信号充足,推理用小 span (1%) 提升质量。类似于 diffusion model 中训练/推理 timestep schedule 的解耦
3. **Confidence-guided refinement 作为后处理**: 仅 7 步即可显著降低 WER,可作为轻量后处理模块加到任何 masked generative TTS 系统上
4. **Linear BPE-level duration estimation**: 简单有效,甚至略优于 GT duration (可能因为避免了 prompt-target 语速不匹配),值得在缺乏 forced alignment 的场景采用
