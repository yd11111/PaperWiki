---
type: paper
tier: deep
title: "ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Training"
arxiv_id: "2605.30965"
source: "Sources/ImmersiveTTS.pdf"
authors: [Jun-Hak Yun, Seung-Bin Kim, Seong-Whan Lee]
year: 2026
venue: "arXiv"
tags: [environment-aware-TTS, diffusion-transformer, MM-DiT, flow-matching, representation-alignment, REPA, dual-stream, CLAP, WavLM, multi-modal]
concepts: ["[[ConditionalFlowMatching]]", "[[Classifier-FreeGuidance]]", "[[Diffusion-basedTTS]]", "[[Self-SupervisedSpeechRepresentation]]", "[[NaturalLanguageDescriptionforTTS]]", "[[DurationPredictor]]", "[[MelSpectrogram]]", "[[NeuralVocoder]]"]
models: ["[[模型库/WavLM|WavLM]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[ConditionalFlowMatching]]✓, [[Classifier-FreeGuidance]], [[Diffusion-basedTTS]], [[Self-SupervisedSpeechRepresentation]], [[NaturalLanguageDescriptionforTTS]], [[模型库/WavLM|WavLM]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]], [[Classifier-FreeGuidance]], [[Diffusion-basedTTS]], [[Self-SupervisedSpeechRepresentation]], [[NaturalLanguageDescriptionforTTS]], [[模型库/WavLM|WavLM]] | 过滤: [[Classifier-FreeGuidance]](pending-review), [[Diffusion-basedTTS]](pending-review), [[Self-SupervisedSpeechRepresentation]](pending-review), [[NaturalLanguageDescriptionforTTS]](pending-review), [[模型库/WavLM|WavLM]](pending-review) | 未命中但可能相关: 无

**谱系定位**: ImmersiveTTS 处于 environment-aware TTS 子领域,这是 [[NaturalLanguageDescriptionforTTS]] 页中标注的"环境感知"扩展方向之一。已知同方向工作包括 VoiceLDM (Lee et al., 2024, AudioLDM + U-Net + CLAP/SpeechT5 双 cross-attention) 和 UmbraTTS (flow matching + SER 控制)。本文的独特定位: 用 MM-DiT 双流架构替代 U-Net,引入 joint attention 显式建模跨模态交互,且在 flow matching 训练中加入 domain-specific REPA 对齐。

**已有认知**:
- [[ConditionalFlowMatching]](confirmed): flow matching 通过学习 ODE 向量场实现少步生成,在 TTS 中已有 VoiceFlow、F5-TTS、CosyVoice 系列等成熟实践。ImmersiveTTS 使用标准 rectified flow matching 目标。
- [[Classifier-FreeGuidance]][待确认]: CFG 通过训练时随机 mask 条件、推理时外推增强条件信号。ImmersiveTTS 扩展为 dual CFG,对 content 和 environment 两个条件独立引导,与 VoiceLDM 的做法一致。KB 页已记录 X-Voice 的 Decoupled CFG,ImmersiveTTS 的 dual CFG 是更早期的同类设计。
- [[Self-SupervisedSpeechRepresentation]][待确认] + [[模型库/WavLM|WavLM]][待确认]: WavLM 是 full-stack SSL 模型,bottom layers 编码 speaker info,top layers 编码 content。ImmersiveTTS 将 WavLM 作为 speech 域 REPA teacher,利用其语言学保真度; ATST-Frame 作为 audio 域 teacher。这是 REPA 策略从图像域 (DINOv2→DiT) 到音频域的 domain-specific 适配。

**创新判断**: 相比 KB 中已有的 environment-aware TTS 方法 (VoiceLDM 基于 U-Net, UmbraTTS 基于 flow matching 但无 REPA),ImmersiveTTS 的创新在于: (1) MM-DiT 双流架构替代 U-Net 实现跨模态 joint attention; (2) domain-specific 双 teacher REPA 同时改善语音保真度和环境音质; (3) 采样效率大幅提升 (25 NFEs vs 200)。

## 速查

> [!summary] 速查
> - **一句话**: 用 MM-DiT 双流架构 + flow matching + domain-specific REPA 实现环境感知 TTS,在 joint attention 中显式建模语音与环境音的交互
> - **路线**: content prompt → Text Encoder + MAS → frame-level prior; environment prompt → CLAP (global AdaLN) + Flan-T5 (token-level stream); dual-stream MM-DiT (12 double + 18 single blocks) → VAE decoder → HiFi-GAN
> - **指标**: AudioCaps WER 8.06% / FAD 5.80 / CLAP 0.308 / SN-MOS 4.20 (vs VoiceDiT: 11.68 / 9.07 / 0.263 / 3.47); 仅 25 NFEs vs 200 [Table 1]
> - **可借鉴**: (1) domain-specific dual-teacher REPA -- 不同域用不同 SSL teacher 对齐,比统一 teacher 效果更好; (2) MM-DiT 双流对 heterogeneous 模态融合的有效性; (3) WavLM 对齐 clean speech 而 ATST 对齐 mixed audio 的不对称 target 设计
> - **局限**: 训练仅用合成混合数据 (LibriTTS + WavCaps mixing),未在真实 in-the-wild 录音上验证; 缺乏 prosody/emotion 显式控制; 单任务 TTS/TTA 各自不及专用系统 (CosyVoice2 WER 5.23 vs 9.89, TangoFlux FAD 2.96 vs 7.81) [Table 7]

## 核心问题

1. **语音与环境音的联合生成为何困难?** 两类音频在时域动态和频谱结构上差异极大 -- 语音是时间局部的、语言学驱动的精细结构,环境音是时间扩展的、频谱弥散的背景信号。现有方法 (VoiceLDM, VoiceDiT) 未显式建模两者的跨模态交互,导致 speech-environment mismatch [§1]。
2. **如何让模型同时保持语言学精度和环境音保真度?** 仅依赖 flow matching 的生成损失不足以学到同时满足两个域约束的表征 [§3.4],需要额外的 feature-level alignment 引导。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ImmersiveTTS 在 AudioLDM2 VAE 的 latent space 上操作,使用 MM-DiT (Flux 架构变体) 作为去噪 backbone,flow matching 作为生成目标。系统接收两个文本输入: content prompt (转录文本) 和 environment prompt (环境描述) [§3.3, Fig 1]。

**处理流程**:
1. **Audio compression**: 目标音频经 mel spectrogram → AudioLDM2 VAE encoder → 8-channel latent (时频各下采样 4 倍) [§3.2]
2. **环境条件处理**: environment prompt 同时送入 CLAP (全局向量 → MLP → 与 timestep embedding 合并 → AdaLN) 和 Flan-T5-Large (token 序列 → linear → 作为 environment stream 输入) [§3.3]
3. **语音条件处理**: content prompt → text encoder → MAS 对齐 → frame-level prior µ → convolution → 与 noisy latent Z_t concat → speech stream 输入 [§3.3]
4. **双流交互**: 12 个 double-stream blocks 中,speech stream 和 environment stream 通过 joint attention 交换信息; 之后 speech stream 经 18 个 single-stream blocks 进一步精炼 [§3.3]
5. **REPA**: 从 speech stream 中间层提取 hidden features,通过 MLP 分别对齐到 WavLM 和 ATST-Frame 的表征 [§3.4]
6. **解码**: VAE decoder → mel → HiFi-GAN → waveform [§3.2]

### 关键设计选择

**为什么用 MM-DiT 双流而非 U-Net?** [论文原文] MM-DiT 的 dual-stream 设计允许两个模态在每一层通过 joint attention 进行双向交互,而 VoiceLDM 的 U-Net 只通过 cross-attention 单向注入条件。这使得语音生成过程可以"动态地关注并协调环境线索,而不丢失语言学结构" [§3.3]。[agent 解读] Joint attention 让 environment tokens 也能根据 speech tokens 调整自己的表示,形成真正的双向信息流,而不是 U-Net 中 text embeddings 被动注入的方式。

**为什么用双粒度环境条件 (CLAP + T5)?** [论文原文] CLAP 提供粗粒度全局声学语义 (modulate AdaLN),T5 提供细粒度 token-level 细节 (输入 environment stream),两者互补 [§3.3]。这遵循了 Auffusion (Xue et al., 2024) 的策略。

**为什么用 MAS 而非 duration predictor?** [论文原文] 采用 Glow-TTS (Kim et al., 2020) 的框架,text encoder 产生 hidden representation,MAS 估计 phone-level duration,展开为 frame-level prior µ [§3.3]。[agent 解读] MAS 提供了显式的时间对齐,使 content prompt 的语言学特征在时间上精确注入 speech stream,这对保持 intelligibility 至关重要。在环境感知场景中,背景音的存在使 alignment 更加重要。

**为什么用 domain-specific dual-teacher REPA?** [论文原文] 单独使用 WavLM 改善语音精度但环境音质无提升; 单独使用 ATST-Frame 改善环境音指标但降低语音清晰度; 统一 teacher (USAD) 虽然平衡但不如各域专用 teacher 的互补组合 [§5.3, Table 4]。[agent 解读] domain-specific pair 的成功源于"互补目标": WavLM 在干净语音上训练,其表征锚定语言学内容; ATST-Frame 在通用音频上训练,其表征编码环境声学事件。两者提供的梯度方向互补,共同约束模型在两个域都学好。

**WavLM 对齐 target 的不对称设计**: [论文原文] WavLM 的对齐 target 取自 LibriTTS 的 clean speech (混合前),确保其目标专注于语言学保真度; ATST-Frame 的 target 取自混合后的音频,捕捉完整声学场景 [Appendix A]。[agent 解读] 这是一个精巧的设计 -- 如果 WavLM 也对齐 mixed audio,其语言学信号会被环境音干扰; 反之 ATST 对齐 clean speech 则丢失环境信息。

### 训练策略

**四个损失函数联合训练** [§3.5]:
- L_Flow: flow matching velocity field 回归 (主生成目标)
- L_REPA: domain-specific dual-teacher 对齐 (WavLM + ATST-Frame, λ_k = 1)
- L_Prior + L_Dur: text encoder 和 duration predictor 的 MAS-based 监督
- 所有权重 λ = 1

**训练数据构造**: LibriTTS train-clean-360 (clean speech) + WavCaps (400k clips, 过滤掉含语音样本后剩 340k non-speech clips)。每个样本随机混合,SNR 在 2-10 dB 均匀采样。15% 概率跳过混合使用纯语音,确保模型保持干净语音生成能力 [§4.1]。

**Dual CFG**: 训练时以 0.1 概率独立 mask content 和 environment prompt。推理时用公式 (6) 双 CFG,默认 (ω_env, ω_cont) = (3, 3) [§3.5]。

**实现细节**: 400k steps, 2x A6000, AdamW lr=1e-4, batch size 8/GPU; 12 double-stream + 18 single-stream blocks, 6 heads, d=1024, ~450M 参数 [§4.1]。Timestep 采样: logit-normal (mean=0, var=1) [§3.5]。

## 实验

| 指标 | ImmersiveTTS | VoiceDiT | VoiceLDM | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| SN-MOS | 4.20 | 3.47 | 3.41 | AudioCaps test | [Table 1] |
| EC-MOS | 3.48 | 3.44 | 3.33 | AudioCaps test | [Table 1] |
| ON-MOS | 3.47 | 2.63 | 2.55 | AudioCaps test | [Table 1] |
| WER (%) | 8.06 | 11.68 | 16.45 | AudioCaps test | [Table 1] |
| FAD | 5.80 | 9.07 | 8.75 | AudioCaps test | [Table 1] |
| CLAP | 0.308 | 0.263 | 0.229 | AudioCaps test | [Table 1] |
| NFEs | 25 | 200 | 200 | - | [Table 1] |
| SN-MOS | 4.18 | 3.45 | 3.32 | Seed-TTS+AudioCaps aug | [Table 2] |
| ON-MOS | 3.23 | 3.12 | 2.91 | Seed-TTS+AudioCaps aug | [Table 2] |
| WER (%) | 4.48 | 7.08 | 11.20 | Seed-TTS+AudioCaps aug | [Table 2] |
| WER (%) single-task TTS | 9.89 | 11.08 | 14.01 | LibriTTS test | [Table 3] |
| UTMOS single-task TTS | 3.23 | 3.33 | 2.82 | LibriTTS test | [Table 3] |
| SECS | 0.8859 | 0.8942 | 0.7601 | LibriTTS test | [Table 3] |

**REPA ablation (AudioCaps test)** [Table 4]:
| 配置 | WER (%) | FAD | CLAP |
| --- | --- | --- | --- |
| Base (无 REPA) | 11.21 | 9.64 | 0.236 |
| WavLM only | 10.97 | 8.02 | 0.231 |
| ATST only | 13.77 | 8.78 | 0.271 |
| WavLM + ATST (ours) | 8.06 | 5.80 | 0.308 |

**Dual CFG sensitivity** [Fig 4, Fig 5]:
- 增大 ω_env > 3 显著恶化 WER (8.06 → 10.92+),FAD 变化小 [Fig 4]
- 增大 ω_cont 改善 WER (在 ω_cont=7 最低),但 FAD 从 4.62 单调升至 10.27,CLAP 从 0.325 降至 0.232 [Fig 5]
- 平衡点 (3, 3) 避免了两端的退化 [§D, Appendix]

**Mixing-based pipeline 对比** [Table 8, Table 9]:
- CosyVoice2 + TangoFlux 后混合在客观指标上最强 (AudioCaps WER 6.76, FAD 4.01, CLAP 0.452) [Table 8]
- 但 pipeline 需要分别生成+混合,无法建模语音-环境交互 [§F.2]

**采样步数分析** [Fig 2]: 9 步即超越 VoiceLDM/VoiceDiT (200 步) 的 WER 和 FAD,展现优异的 quality-efficiency trade-off。

## 局限性

1. **合成混合训练数据**: 仅用 LibriTTS + WavCaps 混合构造,未在 in-the-wild 真实录音上训练/验证,可能无法捕捉自然场景中的复杂声学交互 (如混响、多声源遮蔽) [§7]
2. **缺乏副语言控制**: 无 prosody/emotion/speaking style 的显式控制,限制了表达性 [§7]
3. **单任务不及专用系统**: 纯 TTS 性能弱于 CosyVoice2/3 (WER 9.89 vs 5.23/5.98),纯 TTA 弱于 TangoFlux (FAD 7.81 vs 2.96) [Table 7],统一模型的 trade-off 明显
4. **环境音质量有限**: EC-MOS 接近 VoiceDiT (3.48 vs 3.44 AudioCaps; 3.32 vs 3.38 augmented),环境音匹配方面优势不大 [Table 1, 2]
5. **训练数据规模较小**: LibriTTS 360h + WavCaps 340k clips,相比当前大规模 TTS 系统 (CosyVoice3 用 700k hrs) 数据量有限
6. **SNR 鲁棒性未充分探索**: 训练用 2-10 dB SNR,论文承认"不同 SNR 和场景难度下的鲁棒性尚未充分探索" [§7]

## 点评

ImmersiveTTS 提出了 environment-aware TTS 的一个有说服力的架构: MM-DiT 双流 + domain-specific REPA。核心贡献在于将 Flux/SD3 的 MM-DiT 架构从图像-文本领域迁移到语音-环境音领域,且 REPA 的 domain-specific dual-teacher 设计有理有据 -- Table 4 的 ablation 清晰展示了 WavLM (speech fidelity) 和 ATST-Frame (environmental fidelity) 的互补性。

**亮点**: (1) 采样效率出色,25 步即达到远优于 200 步 baseline 的质量,这归功于 flow matching + REPA 的组合加速收敛; (2) SN-MOS 4.20 和 ON-MOS 3.47 的绝对值表明联合模型的语音自然度可以逼近甚至超过独立 TTS baseline 的水平; (3) WavLM 对齐 clean speech vs ATST 对齐 mixed audio 的不对称 target 设计是精巧的 domain adaptation。

**不足**: (1) 与 pipeline 方案 (CosyVoice2 + TangoFlux) 的差距表明,在现有数据和模型规模下,end-to-end 统一建模尚未全面超越 cascade; (2) 环境音质量 (EC-MOS) 的提升有限,dual-teacher REPA 的环境音增益主要体现在 CLAP 而非感知质量; (3) 训练数据局限于合成混合,实用性有待验证。

## 可复用的 idea

1. **Domain-specific dual-teacher REPA**: 在多模态生成任务中,为不同模态选择不同的 SSL teacher 做 representation alignment,比统一 teacher 更有效。可迁移到: speech-music joint generation, audio-visual synthesis 等场景。
2. **不对称 alignment target**: 语音 teacher 对齐 clean 信号,环境 teacher 对齐 mixed 信号。这种根据 teacher 专长设计 target 的思路可迁移到任何多条件生成场景。
3. **MM-DiT 用于 heterogeneous audio modalities**: 将 Flux 的 image-text 双流适配为 speech-environment 双流,验证了 MM-DiT 在音频领域的跨模态融合能力。后续可探索 speech-music、speech-effects 等组合。
4. **Dual CFG 的 sensitivity 分析方法**: 固定一个 guidance scale 扫另一个,展现两个条件之间的 trade-off 曲面,对任何多条件生成系统都有参考价值。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节因果解释清晰,速查卡片实质 |
> | 可信赖 | pass | 数字全部标注出处,与 PDF 交叉验证正确 |
> | 可区分 | pass | 来源标注覆盖率~75%,整体架构段略低 |
> | 可定位 | pass | KB 背景谱系定位详细,创新判断有对比基准 |
> | 不污染 | pass | 反向更新为追加类型,无 factual-error 风险 |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/ImmersiveTTS-review.yml`
