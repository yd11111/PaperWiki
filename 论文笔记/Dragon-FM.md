---
type: paper
tier: deep
title: "Next Tokens Denoising for Speech Synthesis"
arxiv_id: "2507.22746"
source: "Sources/2507.22746.pdf"
authors: [Yanqing Liu, Ruiqing Xue, Chong Zhang, Yufei Liu, Gang Wang, Bohan Li, Yao Qian, Lei He, Shujie Liu, Sheng Zhao]
year: 2025
venue: "arXiv preprint"
tags: [TTS, zero-shot, flow-matching, autoregressive, chunk-AR, codec, FSQ, podcast, hybrid-AR-diffusion, low-frame-rate]
concepts: ["[[Conditional Flow Matching]]", "[[Finite Scalar Quantization]]", "[[Next-Token Diffusion]]", "[[Token Rate and Bitrate Trade-offs]]", "[[Single-codebook vs Multi-codebook]]", "[[Codec Language Model]]", "[[Mel Spectrogram]]", "[[Multi-scale STFT Discriminator]]"]
models: []
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: Dragon-FM 处于 AR-diffusion 混合路线上,与 KB 中 [[Next-Token Diffusion]] 页记录的 LatentLM/CLEAR 系列密切相关,但采用了不同的粒度选择:LatentLM/CLEAR 是 per-token diffusion head(每个 token 位置独立去噪),Dragon-FM 是 per-chunk flow matching(25 个 token 组成一个 chunk 同时去噪)。这一 chunk 粒度设计介于传统逐 token AR(如 VALL-E)与全序列 NAR(如 E2 TTS)之间,是"AR 步长"这一设计轴上的新探索点。

**已有认知**: [[Conditional Flow Matching]] 页(confirmed)记录了 CFM 在 TTS 中作为 fine-stage renderer 的标准用法(CosyVoice 系列),Dragon-FM 将 CFM 从 sequence-level 改为 chunk-level,与 CosyVoice 2 的 chunk-aware causal flow matching 有共鸣但实现不同——CosyVoice 2 用 attention mask 实现 chunk-aware,Dragon-FM 用 chunk-AR + chunk-内并行去噪。[[Finite Scalar Quantization]] 页 [待确认] 记录了 FSQ 的工作机制和 CosyVoice 系列的应用,Dragon-FM 的 codec 也采用 FSQ,但在 48kHz/12.5Hz 的极低帧率下,验证了 FSQ 在比 CosyVoice (25Hz) 更低帧率下的可行性。[[Token Rate and Bitrate Trade-offs]] 页 [待确认] 记录了帧率-质量的权衡和 Survey 发现(FSQ 在高采样率下可能退化),Dragon-FM 在 48kHz 高采样率 + 12.5Hz 极低帧率的组合下仍保持 SIM 0.916、WER 2.74,是对这一 trade-off 的新数据点。

**创新判断**: 对比 KB 中已有方法,Dragon-FM 的独特贡献在于:(1) 在"AR 步长"轴上提出了 chunk-level 这个中间方案,不同于 per-token(VALL-E/LatentLM)和 full-sequence(E2 TTS);(2) 提出 FSQ token embedding 作为连续向量直接被 flow matching 模型去噪预测,bridging discrete/continuous 的方式不同于 LatentLM 的 VAE latent;(3) 12.5Hz + 48kHz 的 codec 设计将帧率压缩到极限。

> 检索命中: [[Conditional Flow Matching]]✓, [[Finite Scalar Quantization]][待确认], [[Next-Token Diffusion]][待确认], [[Token Rate and Bitrate Trade-offs]][待确认], [[Single-codebook vs Multi-codebook]][待确认], [[Codec Language Model]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: Dragon-FM 将 AR 和 flow matching 统一到单一模型中,以 chunk-wise AR(2 秒/25 token 一组)跨 chunk 建模全局连贯性,chunk 内并行 flow matching 去噪,实现低延迟高质量 48kHz TTS
> - **路线**: 文本+说话人 prompt → 48kHz FSQ codec (12.5Hz) → chunk-AR Transformer (25 tokens/chunk) + chunk 内 flow matching → FSQ token embeddings → codec decoder → 48kHz waveform
> - **指标**: FAD 2.4 (12-step FM, 2s chunk); Codec CodecB: SIM 0.916, WER 2.74 (12.5Hz, 48kHz); TNFE 仅 2-4 (2s audio) vs VALL-E 150 vs E2 32 [Table 2-4]
> - **可借鉴**: (1) chunk-AR 粒度作为 per-token AR 与 full-sequence NAR 之间的折中方案,可调节 chunk 大小控制延迟-多样性平衡; (2) FSQ 离散 token embedding 直接作为连续 flow matching 的目标,无需 VAE——证明去噪模型有内在分类能力; (3) mean flow 优化将 chunk 内 FM 步数降至 2-4 步
> - **局限**: (1) 仅报告 FAD 和 codec 重建指标,无 MOS/SIM/WER 端到端合成指标; (2) 60K h podcast 内部数据集无法复现; (3) 未与 LatentLM/CLEAR/CosyVoice 3 等近期强 baseline 比较; (4) 未开源

## 核心问题

Dragon-FM 试图解决 AR 和 diffusion 模型各自的局限:
- **AR 模型**: 依赖 causal attention,无法利用未来上下文;逐 token 生成导致高延迟(VALL-E 生成 2s 音频需 150 TNFE) [§1]
- **Diffusion/Flow 模型**: 全序列并行去噪无法利用 KV cache;固定时长推理限制了韵律多样性(E2 TTS) [§1]
- **Codec 帧率困境**: 高帧率 codec 提升重建质量但增加 LM 序列长度;低帧率 mel vocoder 在 <20Hz 时质量急剧下降 [§4.3.3]

核心问题: 能否设计一个统一 AR 和 flow matching 的单一模型,同时获得 AR 的 KV cache 效率和 diffusion 的并行去噪能力,并在极低帧率下保持高保真度?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Dragon-FM 由两个核心组件构成 [§3, Fig 1]:

1. **48kHz Audio Codec**: 将 48kHz 波形压缩为 12.5Hz 的 FSQ 离散 token 序列
2. **Chunk-AR + Flow Matching 声学模型**: 以 chunk 为单位自回归生成 FSQ token,chunk 内用 flow matching 并行去噪

推理流程: 文本 + 说话人 prompt → Transformer (causal attention 跨 chunk, bidirectional attention 在 chunk 内) → 每个 chunk 通过 few-step flow matching 从噪声去噪为 FSQ token embeddings → codec decoder 重建 48kHz 波形 [§3.1, Fig 1]

### 关键设计选择

**设计选择 1: 12.5Hz FSQ Codec — 为什么不用 mel spectrogram?**

论文的实验表明 mel vocoder 在低帧率(<20Hz)下重建质量急剧下降(16Hz mel vocoder: SIM 0.904, WER 3.01),而 neural codec 在同等帧率下表现更优(12.5Hz CodecB: SIM 0.916, WER 2.74) [论文原文, §4.3.3, Table 4]。原因是 mel spectrogram 依赖 STFT 的短时平稳性假设,在低帧率下此假设不成立;neural codec 的端到端训练可以学到超越 STFT 窗口限制的表征 [论文原文, §4.3.3]。

Codec 架构是不对称的: encoder 使用双向 full-attention Transformer,decoder 使用 causal-attention Transformer,从而支持 streaming 合成 [论文原文, §3.1, Fig 2]。判别器使用 MPD + MS-STFT [论文原文, §3.1]。

FSQ tokenizer 将连续表征量化为离散 token,codebook 配置为 100x5 (CodecB,2902 bps) 或 100x9 (CodecA,3962 bps) [Table 4]。

**设计选择 2: Chunk-wise AR 而非逐 token AR — 为什么?**

[论文原文] 传统 AR 逐 token 生成(如 VALL-E)的序列长度和推理步数随音频时长线性增长。Dragon-FM 将 25 个 token 打包为一个 chunk(对应 2 秒音频),chunk 间做 AR,chunk 内做并行 flow matching [§3.1]。这样 2 秒音频只需 1 个 AR 步(vs VALL-E 150 步),32 秒音频只需 16 个 AR 步(vs VALL-E 2400 步) [Table 3]。

[agent 解读] 这一设计在 AR 步长轴上找到了一个中间点:逐 token AR(VALL-E)→ chunk-AR(Dragon-FM, 25 tokens/chunk)→ full-sequence(E2 TTS)。Chunk 大小的选择是延迟-多样性的 trade-off:更小 chunk(1s/12.5 tokens)带来更多 AR 循环从而可能增加采样多样性(FAD 2.1 vs 2.4 [Table 2]),更大 chunk 减少延迟但可能降低多样性。

**设计选择 3: FSQ Discrete Tokens as Continuous Vectors — 为什么不用 VAE latent?**

[论文原文] 一个关键 insight 是: continuous denoising models 具有内在的分类能力(intrinsic classification capabilities),无需架构修改即可预测离散 token [§3.1]。通过设计 effective token embeddings,可以利用这一潜力。Dragon-FM 将 FSQ 的离散 token embedding 视为连续向量,直接作为 flow matching 的预测目标 [§3.1]。

[agent 解读] 这与 LatentLM/CLEAR 路线形成对比:LatentLM 使用 sigma-VAE 将波形编码为连续 latent 再做 per-token diffusion;Dragon-FM 使用 FSQ 的离散 token embedding 作为连续目标。FSQ embedding 天然是结构化的(固定网格),比 VAE latent 更 "structured and compact" [论文原文, §3.1],可能简化了 flow matching 的学习任务。但代价是离散化引入的信息损失由 codec decoder 承担。

**设计选择 4: Mean Flow 加速**

[论文原文] 使用 mean flow (Geng et al., 2025) 大幅减少每个 chunk 的 flow matching 迭代步数,保持高生成质量且几乎无退化 [§3.1]。相比蒸馏技术(teacher-student consistency 等),mean flow 不需要额外训练或管理 teacher 模型 [§3.1]。

FAD 数据: 6 step FM → FAD 2.6; 12 step → 2.4; 24 step → 2.2 [Table 2]。说明即使 6 步 FM 也能保持可接受的质量。

### 训练策略

- 数据: 60K 小时公开英文 podcast 数据集,由内部 ASR 系统转录,过滤噪声段和异常 pitch/duration 片段 [§4.1]
- 不做音频去噪(避免去噪模型引入信号损失),依赖 speaker prompt 保证音色一致性 [§4.1]
- Speaker prompt 和 target speech 从同一说话人同一段落中随机选取 [§4.2]
- 训练 2 个 epoch,使用 EMA 稳定更新;Adam optimizer,lr 从 0.001 指数衰减到 0.0001,从 300K iteration 开始衰减 [§4.2]

## 实验

| 指标 | 本文 (Dragon-FM) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| FAD | 2.4 (A1: 2s chunk, 12.5Hz, 12 FM steps) | 2.1 (B1: 1s chunk, 12.5Hz, 12 FM steps) | 60K h podcast | [Table 2] |
| FAD | 2.2 (A2: 2s chunk, 12.5Hz, 24 FM steps) | 2.0 (C2: 2s chunk, 20Hz, 24 FM steps) | 60K h podcast | [Table 2] |
| TNFE (2s audio) | 2-4 | VALL-E: 150; E2: 32 | 理论分析 | [Table 3] |
| TNFE (32s audio) | 32-64 | VALL-E: 2400; E2: 32 | 理论分析 | [Table 3] |
| Codec SIM | 0.916 (CodecB, 12.5Hz) | Mel-VocoderA (25Hz): 0.937; Mel-VocoderC (16Hz): 0.904 | — | [Table 4] |
| Codec WER | 2.74 (CodecB, 12.5Hz) | Mel-VocoderA (25Hz): 2.39; Mel-VocoderC (16Hz): 3.01 | — | [Table 4] |
| Codec SIM | 0.905 (CodecC, 8.3Hz) | Recording: 1.000 | — | [Table 4] |

### 关键消融发现

**Chunk 大小**: 1s chunk (B1, FAD 2.1) 优于 2s chunk (A1, FAD 2.4),可能因为更多 AR 循环引入更多采样变异 [论文原文, §4.3.1]

**Token Rate**: 20Hz (C1, FAD 2.2) 优于 12.5Hz (A1, FAD 2.4),可能因为更高帧率 codec 重建质量更好,贡献了更好的 FAD [论文原文, §4.3.1]

**FM 步数**: 更多 FM 步数持续改善 FAD(A3 6-step: 2.6 → A1 12-step: 2.4 → A2 24-step: 2.2) [Table 2]

**Codec 帧率**: 12.5Hz FSQ codec 在 SIM 和 WER 上都优于 16Hz mel vocoder,但略逊于 25Hz mel vocoder [Table 4]。这验证了 neural codec 在极低帧率下的优势。

## 局限性

1. **评估不充分**: 仅报告 FAD(多样性)和 codec 重建指标(SIM/WER);缺少端到端 TTS 的标准指标(合成 MOS、合成 WER、speaker similarity for synthesis) [agent 解读]
2. **缺少主要 baseline 比较**: 未与 LatentLM、CLEAR、CosyVoice 3、Seed-TTS 等近期强 baseline 做端到端对比 [agent 解读]
3. **数据集不可复现**: 60K h podcast 数据通过内部 ASR 转录和过滤获得,外部无法复现 [§4.1]
4. **TNFE 理论分析简化**: TNFE 比较忽略了 text prompt tokenization 和 speaker prompt 大小对 KV cache 的影响,且将 VALL-E 的 NAR 模型简化了 [Table 3 footnote]
5. **未开源**: 模型和代码未开放 [agent 解读]
6. **Podcast 聚焦**: 实验仅在 podcast 场景评估,未验证短句/多说话人/情感等常见 TTS 场景 [agent 解读]

## 点评

Dragon-FM 提出了一个在概念上优雅的 AR-FM 统一框架,"next-token denoising" 的命名精确捕捉了 chunk-AR + chunk 内 flow matching 的本质。12.5Hz FSQ codec 的设计展示了 neural codec 在极低帧率下相对于 mel spectrogram 的明确优势。

然而,本文的评估是最大弱点。FAD 仅衡量生成分布与真实分布的距离,无法反映单样本的自然度和可懂度。缺少合成语音的 MOS、WER、SIM 等标准指标,也缺少与近期强系统的端到端对比。"FSQ discrete tokens 可以被 continuous denoising 模型直接预测"这一 insight 虽然有趣,但论文未提供消融实验来验证这一设计相对于使用 VAE latent 的优势。

从工程角度,"chunk-AR + chunk 内 FM"的架构在长音频生成(如 podcast)上有天然优势:KV cache 跨 chunk 保持,每个 chunk 内用少量 FM 步骤并行去噪,延迟可控(首字节延迟仅取决于第一个 chunk 的 FM 步数)。这一设计在流式 TTS 中也有潜力。

## 可复用的 idea

1. **Chunk-AR 粒度作为设计轴**: 在 per-token AR 和 full-sequence NAR 之间,chunk 大小是一个可调的超参数,控制延迟-多样性-效率的三角平衡。其他系统可以尝试不同 chunk 大小。
2. **FSQ embedding as continuous flow target**: 无需 VAE,直接用量化 token 的 embedding 作为 flow matching 目标,简化了 pipeline(不需要独立的 VAE encoder/decoder)。这对任何使用 FSQ 的系统都可迁移。
3. **Asymmetric codec (bidirectional encoder + causal decoder)**: 编码器用双向 attention 获取最优表征,解码器用 causal attention 支持 streaming。这一不对称设计可直接复用。
4. **Mean flow for few-step chunk denoising**: 在 chunk-level FM 中用 mean flow 替代 consistency distillation,避免 teacher 模型的额外成本。

---

> [!review] 审阅 (自动)
> 见 `_review/Dragon-FM-review.yml`

---

检索命中: [[Conditional Flow Matching]]✓, [[Finite Scalar Quantization]][待确认], [[Next-Token Diffusion]][待确认], [[Token Rate and Bitrate Trade-offs]][待确认], [[Single-codebook vs Multi-codebook]][待确认], [[Codec Language Model]][待确认] | 过滤: 无 | 未命中但可能相关: 无
