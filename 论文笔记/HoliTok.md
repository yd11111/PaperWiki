---
type: paper
tier: deep
title: "HoliTok: A Continuous Holistic Tokenization with Robust Dual Capabilities of Speech Generation and Understanding"
arxiv_id: "2605.29948"
source: "Sources/HoliTok.pdf"
authors: [Bohan Li, Shi Lian, Hankun Wang, Yiwei Guo, Yu Xi, Zhihan Li, Da Zheng, Colin Zhang, Kai Yu]
year: 2026
venue: "arXiv"
tags: [continuous-tokenizer, VAE, progressive-training, unified-model, speech-generation, speech-understanding, AR-DiT, flow-matching, representation-distillation, TTS, ASR]
concepts: ["[[SpeechTokenizer]]", "[[VariationalAutoencoderforTTS]]", "[[ConditionalFlowMatching]]", "[[SemanticvsAcousticTokens]]", "[[Next-TokenDiffusion]]", "[[CodecTrainingObjectives]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: ["[[BigVGAN]]", "[[WavLM]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页: [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓, [[SemanticvsAcousticTokens]]✓, [[VariationalAutoencoderforTTS]][待确认], [[Next-TokenDiffusion]][待确认], [[CodecTrainingObjectives]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓, [[SemanticvsAcousticTokens]]✓ | 参考: [[VariationalAutoencoderforTTS]](pending-review), [[Next-TokenDiffusion]](pending-review), [[CodecTrainingObjectives]](pending-review) | 未命中但可能相关: 无

**谱系定位**: HoliTok 位于 **连续 VAE tokenizer + AR+DiT 下游架构** 路线的前沿节点。这条路线由 LatentLM (2024) 开创 per-token diffusion 范式, CLEAR/VibeVoice (2025) 推进流式与工业落地, Semantic-VAE (2025) 和 Ming-UniAudio (MingTok-Audio, 2025) 分别从语义正则化和统一生成-理解两个方向拓展。已有知识库记录了两个关键困境:

1. **重建-生成困境** ([[VariationalAutoencoderforTTS]][待确认]): Semantic-VAE 发现高维 VAE latent 重建好但下游 TTS 可懂度差, LongCat-AudioDiT 佐证了这一非单调关系。HoliTok 的渐进式训练策略试图通过先建立高保真重建再逐步引入语义,来绕过而非正面对抗这一困境。
2. **Semantic-Acoustic 矛盾** ([[SemanticvsAcousticTokens]]✓): 语义 token 利于理解但重建差, 声学 token 保真度高但语义不足。HoliTok 的核心主张是: 与其在下游模型中用复杂架构弥补 tokenizer 的不足(多 token 流/任务专用 encoder), 不如让 tokenizer 本身同时满足 decodable + learnable + informative 三个要求。

**已有认知**: [[Next-TokenDiffusion]][待确认] 记录了 AR+DiT 范式从 LatentLM 到 SemaVoice 的演进, HoliTok 的下游架构与 DiTAR/Ming-UniAudio 同属此范式但强调 tokenizer 端的创新而非架构端。[[ConditionalFlowMatching]]✓ 记录了 flow matching 在 TTS 中的广泛使用, HoliTok 的 DiT head 使用 flow matching 预测 latent patch。

**创新判断**: 相比 Semantic-VAE 仅做 WavLM 帧级对齐, HoliTok 的 Stage III 增加了 utterance-level x-vector 蒸馏 + 多任务 LM 监督 + 更强的 KL 正则化, 形成更完整的"downstream-aware enrichment"。相比 Ming-UniAudio 的 MingTok-Audio 保持底层 latent 不变而在上层加语义模块, HoliTok 通过渐进式训练让 **同一个 latent space** 同时兼顾声学保真和语义信息, 是更彻底的 holistic 方案。

## 速查

> [!summary] 速查
> - **一句话**: 渐进式三阶段训练的连续 VAE 语音 tokenizer,在同一个 25 Hz/128-dim latent 空间中同时实现高保真重建、可学习的 TTS 生成、和统一生成-理解建模
> - **路线**: 48kHz waveform → causal conv encoder (6 downsampling blocks, hop=1920) → temporal variational bottleneck (LSTM+normalizing flow) → 25Hz/128-dim latent → BigVGAN-style decoder → waveform; 下游: latent patches → PatchEncoder → Qwen2.5-0.5B LLM (AR) → DiT flow-matching head → latent → decoder
> - **指标**: 重建 PESQ 4.10/4.01 (NB/WB), WER 4.22%, SPKSIM 0.968 (LibriSpeech test-other) [Table 1]; TTS WER 1.33% (Seed-TTS-en) [Table 2]; 统一模型 HoliTok-Unite TTS avg WER 8.59%, ASR avg WER 8.02% [Table 3]; 可控 TTS 最佳 CLSP 和最低 WER [Fig 2]
> - **可借鉴**: (1) 渐进式 AE→VAE→downstream-aware 三阶段训练策略——先建立可靠解码流形,再冻结两端仅训练 bottleneck,最后联合优化+语义蒸馏; (2) implicit fidelity transfer 的理论分析框架——量化了 VAE 波形失真的上界与 Stage-I 失真和 AE-to-VAE latent shift 的关系; (3) DiT TTS-only 预训练初始化显著提升统一训练的 TTS 性能
> - **局限**: (1) 仅在语音域评估,未验证音乐/环境音的泛化性; (2) 下游仅用 AR+DiT 架构,未探索 pure DiT/NAR 架构; (3) 训练数据 ~500K h 含大量内部数据难以复现; (4) HoliTok-Unite 含 680M 语义 encoder,参数量显著增大

## 核心问题

1. **为什么需要一个"holistic"的连续 tokenizer?** 现有连续 tokenizer (Semantic-VAE, MingTok-Audio) 通常优化单一目标(重建或生成),在统一生成-理解模型中表现不一致——Semantic-VAE TTS 崩溃 (WER 102.32%), MingTok-Audio TTS 也弱于单独任务性能 [Table 3] [§1]。HoliTok 的核心命题是: 负担应由 tokenizer 承担,而非通过复杂下游架构弥补。
2. **如何在同一个 latent space 同时满足重建保真度、语义信息和可学习性?** 这三个目标存在张力: 强 KL 正则化有利于可学习性但损害重建, 语义蒸馏有利于理解但可能损害生成 [§1, §3]。HoliTok 用渐进式三阶段训练解耦这些目标。
3. **Continuous tokenizer 能否在统一 AR+DiT 架构中同时支持 TTS 和 ASR, 且不需要额外 trick?** 这是论文的核心实证问题。实验表明 HoliTok 是测试的所有表示中唯一能在统一架构中稳健运行的表示 [Abstract, §4.4]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

HoliTok 是一个基于低延迟 VAE 的语音 tokenizer, 将 48 kHz 波形编码为 25 Hz / 128-dim 连续 latent 序列 [§3.1]。

**Encoder**: 1D conv projection → 6 strided causal conv downsampling blocks (channel: 12→768, kernel sizes: 4,4,4,8,12,20, 下采样率: 2,2,2,4,6,10, 总 hop=1920), 每个 block 后接 6 层 dilated causal conv residual stack。最终投影到 128 维。近似因果, 仅在最后一层有 2-frame lookahead [§3.1]。

**Temporal Variational Bottleneck**: 4-layer LSTM + project-in/out linear + 1x1 conv 预测对角 Gaussian posterior 的 mean/log-scale。重参数化采样后, 额外使用 normalizing flow 计算 KL 时增强后验表达力 [§3.1]。[agent 解读] LSTM 的引入使 bottleneck 能建模帧间依赖, 这与 LatentLM 的 per-frame 独立后验不同,有助于产生更平滑、更容易被 AR 模型预测的 latent 序列。

**Decoder**: BigVGAN-style 生成器, 使用 AMPBlock + SnakeBeta 激活, 与 encoder 镜像结构的上采样模块, 同样有 2-frame lookahead [§3.1]。

**Supervision Network** (仅 Stage III 使用): 0.6B Transformer encoder + 预训练 Qwen2.5-0.5B decoder, 用于多任务 LM 监督 [§3.1]。

### 关键设计选择

**设计选择 1: 渐进式三阶段训练 (核心创新)**

为什么不一步到位训练一个同时满足重建+语义+可学习性的 VAE? [论文原文] 因为强 KL 约束会在 decoder 尚未学好高保真重建流形时就迫使表示丢弃声学细节 [§3.2]。

- **Stage I (500K steps)**: 确定性 autoencoder, 仅优化 reconstruction loss (multi-scale mel + adversarial + feature matching) [Eq. 1-2]。目的: 建立高保真重建流形 [§3.2]。
- **Stage II (50K steps)**: 冻结 encoder 和 decoder, 仅训练 temporal variational bottleneck, 用弱 KL (β_low=0.1) 将确定性 AE 空间转化为随机 VAE 空间 [Eq. 3, §3.2]。
  - **Implicit fidelity transfer** [Appendix A]: 冻结的预训练 decoder 和以重建为主导的目标函数约束 Stage-II 的变分采样留在高保真 AE 流形附近。形式化证明 VAE 波形失真上界为 2ε_AE + 2L²_ψ·δ_shift, 其中 ε_AE 是 Stage-I AE 失真, δ_shift 是 AE-to-VAE latent shift [Proposition 1]。[agent 解读] 这个分析提供了一个有用的直觉: 只要 bottleneck 产生的 latent 不偏离 AE latent 太远 (δ_shift 小), 且 decoder 足够平滑 (L_ψ 小), 就能保持 Stage-I 的重建质量。弱 KL 正是控制 δ_shift 的手段。
- **Stage III (200K steps)**: 解冻全部参数, 联合优化重建 + 强 KL (β_high=7) + 多粒度蒸馏 + 多任务监督 [Eq. 6, §3.3]。

**设计选择 2: 多粒度表示蒸馏 (Stage III)**

- **帧级**: 冻结 WavLM 第 23 层 (选择理由: 高层特征包含更丰富的语境语义信息 [agent 解读, 依据 SSL 层级信息分离的已知特性]), 用 cosine similarity loss 对齐 VAE latent 与 WavLM 隐表示, 时间插值适配不同帧率 [Eq. 4, §3.3]
- **话语级**: 聚合 latent 序列为 utterance-level 表示, 与 x-vector speaker embedding 对齐 [Eq. 4, §3.3]

[论文原文] 蒸馏让 latent space 保留超越波形重建所需的语义和副语言信息 [§3.3]。

**设计选择 3: 多任务 LM 监督 (Stage III)**

通过 supervision network (Transformer encoder + Qwen2.5-0.5B decoder) 将异构下游任务 (ASR, 情感识别, 音频 captioning, 声音事件检测) 统一为 task-conditioned language modeling [Eq. 5, §3.3]。

[论文原文] 蒸馏使 latent space 接近 frozen teacher 分布, 但不一定保留所有下游任务需要的信息。LM 监督确保 latent 保留对理解任务关键但对重建不必要的信息 [§3.3]。

[agent 解读] 这一设计与 Ming-UniAudio 的 MingTok-Audio 形成对比: MingTok-Audio 冻结底层 VAE latent, 在上层加一个独立的语义模块, 导致"inconsistent modeling space" [§1]。HoliTok 直接在 tokenizer 训练阶段就将理解信号注入 VAE latent space, 是更彻底的方案, 但代价是训练更复杂。

**设计选择 4: AR+DiT 下游架构**

与 DiTAR/Ming-UniAudio 类似, 使用 Qwen2.5-0.5B LLM + 18-layer DiT flow-matching head [§3.4]。
- **理解**: LLM 从 audio embedding 预测 text tokens (cross-entropy) [Eq. 9]
- **生成**: LLM 输出 causal hidden states, DiT 以 flow matching 方式预测 next latent patch [Eq. 10-11]
- **HoliTok-Base vs Unite**: Base 用 8-layer PatchEncoder 映射 VAE latent 到 LLM; Unite 用 Stage-III 的 causal semantic encoder 替代 PatchEncoder, 提供预建模的语义特征 [Table 4]

### 训练策略

**Tokenizer 训练**:
- 数据: ~500K h 语音 + 音乐 + 环境声 (含大量内部数据) [§4.1]
- Stage I: 500K steps, 9.6s 音频 crop
- Stage II: 50K steps, β_low=0.1, 冻结 encoder+decoder
- Stage III: 200K steps, β_high=7, per-GPU batch=1 (因 supervision network 需要) [Table 7, Appendix B]
- Optimizer: AdamW (lr=1e-4, betas=(0.8,0.99)), 指数衰减到 1e-6

**下游训练**:
- TTS: 95K h filtered Emilia, 200K steps; 可控 TTS 额外 50K steps (EmoVoice-DB, FCaps, PSCBase)
- 统一: TTS (Emilia) + ASR (AISHELL-1/2, GigaSpeech, MLS, CommonVoice, FLEURS, LibriSpeech), TTS:ASR ratio ~5:1 [§4.4]
- DiT 初始化: TTS-only 预训练 checkpoint → 显著提升统一训练的生成质量 [Table 8]

## 实验

| 指标 | 本文 (HoliTok) | Semantic-VAE | MingTok-Audio | Mel Spectrogram | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| PESQ (NB/WB) | 4.10/4.01 | 3.99/3.80 | 4.23/4.12 | 4.15/4.05 | LibriSpeech test-other | [Table 1] |
| STOI | 0.974 | 0.969 | 0.981 | 0.988 | LibriSpeech test-other | [Table 1] |
| WER (重建) | 4.22% | 4.15% | 4.27% | 3.96% | LibriSpeech test-other | [Table 1] |
| SPKSIM | **0.968** | 0.963 | 0.950 | 0.957 | LibriSpeech test-other | [Table 1] |
| EMOSIM | **0.995** | 0.993 | 0.992 | 0.988 | LibriSpeech test-other | [Table 1] |
| CR / TPS | **7.5x / 25** | 2.73x / 40 | 2.19x / 50 | 2.00x / 86 | — | [Table 1] |
| TTS WER (Seed-en) | **1.33%** | 1.42% | 1.84% | — | Seed-TTS-Eval en | [Table 2] |
| TTS Win-Rate (Emotion) | **25.5%** | 14.3% | 8.4% | — | Emergent-TTS | [Table 2] |
| TTS Win-Rate (Paralingistic) | **53.6%** | 44.2% | 39.8% | — | Emergent-TTS | [Table 2] |
| Unified TTS WER (Seed-en) | 7.20% (Unite) | 102.32% | 51.06% | — | Seed-TTS-Eval en | [Table 3] |
| Unified ASR WER (AISHELL-1) | 5.93% (Unite) | 15.81% | **5.01%** | — | AISHELL-1 test | [Table 3] |

**关键实验发现**:

1. **重建**: HoliTok 在 7.5x 最高压缩比下, 信号级指标略低于 MingTok-Audio/Mel, 但副语言保持最好 (SPKSIM, EMOSIM best), 且远超 vanilla VAE (同压缩率) [Table 1]。[agent 解读] 这说明渐进训练策略的价值不在提升峰值 PESQ, 而在用更紧凑的表示保留更多副语言信息。

2. **TTS (独立)**: HoliTok 在 zero-shot 和可控 TTS 上全面竞争力最强——Seed-TTS-en WER 最低, Emergent-TTS 情感和副语言 win-rate 均最高 [Table 2, Fig 2]。

3. **统一模型 (核心实验)**: 这是论文的核心发现。在同一 AR+DiT 架构下, Semantic-VAE 的 TTS 完全崩溃 (WER 102%), MingTok-Audio 的 TTS 也严重退化 (WER 51%) [Table 3]。[论文原文] 这说明独立重建质量或下游任务性能不等于统一模型中的可用性 [§4.4]。HoliTok-Base 已在 TTS 上大幅超越所有 baseline, HoliTok-Unite 进一步将 TTS avg WER 从 20.90%→8.59%, ASR avg WER 从 12.63%→8.02% [§4.4]。

4. **消融** [Table 8]:
   - DiT TTS-only 初始化对所有表示都显著改善生成, 但不能保证理解-生成平衡 [§C]
   - 去掉蒸馏: TTS WER 略降但 speaker similarity 降, ASR 无明显改善 [§C]
   - 去掉监督: TTS 完全崩溃 (WER 110%), 说明多任务监督对生成鲁棒性至关重要, 不仅仅帮理解 [§C]
   - 冻结 semantic encoder (Unite): ASR 退化, 说明下游训练中适配语义接口很重要 [§C]

## 局限性

1. **仅语音域**: 未在音乐、环境声等更广泛音频域验证, 尽管训练数据包含了这些 [§Limitations]
2. **仅 AR+DiT 下游**: 未探索 pure DiT 或 fully NAR 架构, 无法确认 HoliTok 的通用性 [§Limitations]
3. **数据不可复现**: 训练数据 ~500K h 含大量内部中英 TTS 语料 [§4.1]
4. **HoliTok-Unite 代价**: 需要额外 680M 语义 encoder (Stage-III causal), 总参数 861M vs Base 的 181M [Table 6]
5. **统一模型的绝对性能**: 即使是 HoliTok-Unite, 统一模型的 TTS WER (Seed-en 7.20%) 仍明显高于独立 TTS 的 1.33%, 说明统一训练仍有显著成本 [Table 2 vs Table 3]

## 点评

**定位精准, 但 claim 需谨慎解读**:

HoliTok 准确识别了当前连续 tokenizer 的核心痛点——在统一生成-理解模型中表示的"可用性"(usability) 而非单一任务性能。三阶段渐进训练是一个优雅的工程方案, implicit fidelity transfer 的形式化分析增加了方法的可解释性。

**最大贡献**: Table 3 的实验揭示了一个重要事实——Semantic-VAE 和 MingTok-Audio 这两个在独立任务上表现良好的表示, 在统一 AR+DiT 架构中几乎不可用 (TTS 崩溃)。这说明 tokenizer 的"holistic"设计不是锦上添花, 而是统一模型的必要条件。

**存疑点**:
1. 统一模型实验的公平性: 所有 baseline 都使用了相同的 AR+DiT 架构, 但 MingTok-Audio 原论文的协议不同 (作者使用了其"reported ablation protocol") [§4.1]。MingTok-Audio 在原系统中的表现可能更好, 论文的对比可能低估了 MingTok-Audio 在其原生架构下的能力。
2. HoliTok-Unite 的 causal semantic encoder (680M) 引入了显著的额外参数和预学习的语义特征, 这在某种程度上与 MingTok-Audio 的语义模块类似, 但论文将其定位为 HoliTok tokenizer 的一部分而非下游架构的一部分。这是否算"tokenizer 端解决问题"有些模糊。
3. 500K h 内部数据的不可复现性限制了验证。

## 可复用的 idea

1. **渐进式 AE→VAE 转换**: 先训练确定性 AE 建立解码流形, 再冻结两端仅训练 bottleneck 转为 VAE。可迁移到任何需要 VAE 但担心 KL-vs-fidelity trade-off 的场景。implicit fidelity transfer 分析提供了直觉和理论保证。
2. **多任务 LM 监督作为 tokenizer 训练的正则化**: 不仅仅是为了理解任务, 消融显示它对 **生成鲁棒性** 也至关重要 (去掉后 TTS 崩溃) [Table 8]。这暗示: 向 tokenizer 注入下游任务信号可以改善 latent space 的结构, 使其对 AR 生成更友好。
3. **DiT TTS-only 预初始化**: 在统一训练前用 TTS-only 数据预训练 DiT head, 对所有表示都有效 [Table 8]。简单但高效的 trick。
4. **Compression ratio 作为评估维度**: 论文用 CR = (f_s · ceil(log2(f_s))) / (f_z · d_z · 32) 统一量化了不同表示的信息压缩效率 [Eq. 13], 使得跨表示的公平对比更容易。

