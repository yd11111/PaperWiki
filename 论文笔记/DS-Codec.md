---
type: paper
tier: deep
title: "DS-Codec: Dual-Stage Training with Mirror-to-NonMirror Architecture Switching for Speech Codec"
arxiv_id: "2505.24314"
source: "Sources/DS-Codec.pdf"
authors: [Peijie Chen, Wenhao Guan, Kaidi Wang, Weijie Wu, Hukai Huang, Qingyang Hong, Lin Li]
year: 2025
venue: "arXiv (Interspeech submission)"
tags: [audio-codec, single-codebook, VQ, product-quantization, dual-stage-training, speech-reconstruction]
concepts: ["[[Residual Vector Quantization]]", "[[Single-codebook vs Multi-codebook]]", "[[Codebook Collapse]]", "[[Codec Training Objectives]]", "[[Speech Tokenizer]]"]
models: ["[[模型库/EnCodec|EnCodec]]"]
tasks: []
datasets: ["LibriSpeech", "LJSpeech"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[Residual Vector Quantization]], [[Codebook Collapse]], [[Speech Tokenizer]], [[模型库/EnCodec|EnCodec]]; 2 个待确认: [[Single-codebook vs Multi-codebook]][待确认], [[Codec Training Objectives]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。

### 谱系定位

DS-Codec 位于 **single-codebook neural speech codec** 的演进线上。这条路线从 VQ-VAE (2017) 的单码本开始,经历了 RVQ 多码本时代 (SoundStream 2021, EnCodec 2022, DAC 2023),再到单码本回归 (BigCodec 2024, WavTokenizer 2024)。DS-Codec 是 2025 年对这一路线的进一步推进。

**已有认知**:
- **RVQ vs SVQ**: RVQ 重建质量更高但 token rate 高 (600 tokens/s @8层); SVQ token rate 低 (75-80 tokens/s) 适合 LLM 集成,但单码本覆盖高维空间困难。DS-Codec 选择 SVQ 路线。
- **镜像 vs 非镜像架构**: 传统 codec (EnCodec, DAC) 采用 encoder-decoder 镜像结构; 新一代 (FACodec, WavTokenizer) 转向非镜像结构,强调 decoder 更重要。DS-Codec 试图结合两者优势。
- **Codebook collapse**: 单码本 VQ 中码本利用率是关键问题。DAC 通过 factorized codes + L2-norm 从 62% 提升到 99%。DS-Codec 也采用了 DAC 的低维投影 + L2-norm 方案。
- **Product Quantization**: 将高维向量分组独立量化后组合,类似于 RVQ 变体中的 GVQ 思路,但沿 channel 维分组而非残差细化。PQ-VAE (2024) 已探索过此方案。
- **训练策略**: APCodec+ (2024) 提出了两阶段训练,但采用非镜像→decoder-only 的策略。DS-Codec 的创新在于镜像→非镜像的切换顺序。

### 创新判断

与现有工作的区别: (1) 训练策略的方向反转: 先镜像后非镜像,而非 APCodec+ 的先联合后 decoder; (2) 保留第一阶段 decoder 权重而非重新初始化; (3) 在 PQ 方案中用 4 个小码本 (16x16x16x16=65536) 替代单个大码本。

> 检索命中: [[Residual Vector Quantization]]✓, [[Codebook Collapse]]✓, [[Speech Tokenizer]]✓, [[模型库/EnCodec|EnCodec]]✓ | 过滤: [[Single-codebook vs Multi-codebook]](pending-review), [[Codec Training Objectives]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过先镜像训练(强化 codebook)再切换到非镜像架构(强化 decoder)的两阶段策略,在单码本条件下实现接近 ground truth 的语音重建质量
> - **路线**: 波形 → CNN+LSTM encoder (200x下采样) → VQ/PQ 单码本量化 → Transformer Block → CNN decoder (上采样) → 重建波形
> - **指标**: LibriSpeech test-clean: UTMOS 4.218 (GT 4.086), PESQ 2.862, STOI 0.941 (单码本 1.04 kbps); LJSpeech: UTMOS 4.451, PESQ 2.962 [Table 1, Table 2]
> - **可借鉴**: 镜像→非镜像的两阶段训练策略可迁移到其他 codec 设计; 第一阶段冻结时保留 decoder 权重+降低学习率的 warm-start 策略比 APCodec+ 的重新初始化更高效
> - **局限**: 仅在 LibriSpeech (1000h, 16kHz, 英语朗读) 上训练和评估,未验证多语言/多说话人/多采样率泛化; 无主观 MOS 评估; 未与 2024-2025 年最新 codec (SNAC, SiTok, TS3-Codec) 对比; 代码未开源

## 核心问题

传统 neural speech codec 面临一个架构选择困境: **镜像结构 (encoder-decoder 对称) 有利于 codebook 学习但 decoder 受限于 encoder 的镜像约束, 非镜像结构 decoder 更自由但 codebook 质量不如镜像结构**。如何兼得两者优势?

DS-Codec 的回答: 不选其一,而是**在训练的不同阶段分别利用两种架构的优势** — 先用镜像结构训练出高质量 codebook,再切换到非镜像结构释放 decoder 潜力。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DS-Codec 最终部署的架构为**非镜像结构**: CNN+LSTM Encoder → Quantization Module (VQ 或 PQ) → Transformer Block → CNN Decoder [Fig 1]。

- **Encoder**: 灵感来自 BigCodec [§2.1],由一系列残差 CNN blocks 组成,每个 block 使用 snake activation + 2 层单向 LSTM,通过卷积实现下采样。五个 block 累计下采样 200 倍 (16kHz → 80 Hz, 即 80 tokens/s) [§2.1]。
- **Decoder**: 镜像 Encoder 结构,使用转置卷积上采样 [§2.1]。
- **Transformer Block**: 受 LLaMA decoder layer 启发,集成残差 Attention + 残差 SwiGLU,配合 RMSNorm [§2.1]。[agent 解读] 该 block 被插入在量化模块和 decoder 之间,目的是利用 Transformer 的长程上下文建模能力,在离散 token 空间中补充序列级信息,弥补 CNN 感受野有限的问题。
- **Discriminator**: 与 BigCodec 相同,使用 HiFi-GAN 的 MPD + EnCodec 的 MS-STFT discriminator [§2.1]。

### 关键设计选择

#### 1. 镜像 vs 非镜像: 为什么先镜像后非镜像?

[论文原文] 镜像结构的 encoder 和 decoder 具有对称的信息容量,使得 codebook 学习更稳定——encoder 输出的 latent 和 decoder 期望的输入在同一表示空间,quantization module 的 input-output gap (MSE loss) 更小 [§3.5, Fig 2]。

[论文原文] 非镜像结构的 decoder 更自由,可以采用更强的架构 (如加入 Transformer Block),从而更好地利用 codebook 信息进行高保真重建,但 codebook 质量可能不如镜像训练 [§3.5]。

[agent 解读] 这种策略的深层逻辑是: codebook 质量是整个系统的瓶颈——如果量化后的信息损失大,再强的 decoder 也无法恢复。因此先确保 codebook 质量 (镜像阶段),再优化信息利用效率 (非镜像+Transformer 阶段)。

#### 2. 为什么不重新初始化 decoder?

[论文原文] 与 APCodec+ 不同,DS-Codec 在第二阶段保留了第一阶段 decoder 的参数权重并降低学习率 [§2.3]。[论文原文] 理由是 decoder 已在第一阶段展示出良好的重建能力,保留权重可以加速训练、减少不必要的调整 [§2.3]。

[agent 解读] 这实质上是 warm-start fine-tuning 策略: 第一阶段的 decoder 已学会了如何从镜像 latent 空间重建语音,这些知识在切换到非镜像架构后仍有价值,只需适配 Transformer Block 引入的新信息流。

#### 3. Product Quantization vs Vector Quantization

DS-Codec 探索了两种量化方案 [§2.2]:

- **DS-Codec-VQ**: 单码本 8192 entries,使用 DAC 的低维投影 (dim=8) + L2-norm 方案 [§2.2.1]。带宽 1.04 kbps。
- **DS-Codec-PQ**: 4 个子码本,各 16 entries (16^4 = 65536 等效码本大小),将 latent 沿 channel 维分成 4 段独立量化后拼接 [§2.2.2, Algorithm 1]。带宽 1.28 kbps。

[agent 解读] PQ 方案的优势在于: 用极小的子码本 (16 entries) 组合出大等效码本 (65536),避免了大码本的 codebook collapse 风险,同时保持低 token rate (仍然只产生一个组合 index)。但 PQ 的独立量化假设忽略了 channel 间的相关性。

### 训练策略

**Stage 1 — Mirror Training** [§2.3]:
- Encoder + Quantization Module + 镜像 Decoder 联合训练
- AdamW (beta1=0.8, beta2=0.9), lr 1e-4 → 1e-5, 1000 warmup steps
- Batch size 10, 1 秒随机裁剪
- 目标: 建立高质量 codebook

**Stage 2 — Decoder Training** [§2.3]:
- 冻结 Encoder + Quantizer 参数
- 加入 Transformer Block,切换为非镜像架构
- 重新初始化 discriminator (但不重新初始化 decoder)
- Batch size 24, lr 2e-5 → 1e-5
- 目标: 利用冻结的高质量 codebook,强化 decoder 重建能力
- 注意: DS-Codec-VQ 的第二阶段直接使用 BigCodec 的官方 checkpoint 作为起点 [§3.2],因此其改进包含 BigCodec 预训练 codebook 的贡献

[agent 解读] DS-Codec-VQ 的第二阶段直接使用 BigCodec 的官方 checkpoint 作为起点 [§3.2],这意味着 DS-Codec-VQ 实际上是 BigCodec + Transformer Block + 非镜像微调。这使得 DS-Codec-VQ 的性能提升中,BigCodec 预训练 codebook 的贡献难以分离。

## 实验

| 指标 | DS-Codec-VQ | DS-Codec-PQ | BigCodec | WavTokenizer | DAC (1cb) | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UTMOS ↑ | **4.218** | 4.214 | 4.108 | 3.784 | 1.246 | 4.086 | LibriSpeech | [Table 1] |
| PESQ ↑ | 2.862 | **2.882** | 2.681 | 2.114 | 1.056 | - | LibriSpeech | [Table 1] |
| STOI ↑ | **0.941** | 0.941 | 0.935 | 0.897 | 0.617 | - | LibriSpeech | [Table 1] |
| F1 Score ↑ | **0.944** | 0.943 | 0.942 | 0.911 | 0.552 | - | LibriSpeech | [Table 1] |
| UTMOS ↑ | **4.451** | 4.428 | 4.385 | 3.870 | - | 4.378 | LJSpeech | [Table 2] |
| PESQ ↑ | **2.962** | 2.886 | 2.822 | 1.948 | - | - | LJSpeech | [Table 2] |

**关键发现**:

1. **DS-Codec UTMOS 超过 GT**: DS-Codec-VQ 在 LibriSpeech 上 UTMOS 4.218 vs GT 4.086,在 LJSpeech 上 4.451 vs 4.378 [Table 1, 2]。[agent 解读] UTMOS 超过 GT 不意味着"更好",而是 UTMOS 作为 MOS 预测器可能对 codec 重建信号的某些频率特性给出偏高分数,这是已知的 metric gaming 现象。缺少人工 MOS 评估是一个局限。

2. **两阶段 vs APCodec+ 策略**: Table 3 对比了提出的镜像第一阶段 vs APCodec+ 的非镜像联合训练第一阶段。在同一模型架构下,镜像 Stage 1 在所有指标上优于 APCodec+ Stage 1 (UTMOS 4.123 vs 4.113, PESQ 2.768 vs 2.632) [Table 3]。第二阶段加入 Transformer Block 后进一步提升 (UTMOS 4.214) [Table 3]。

3. **镜像结构对 codebook 的影响**: Fig 2 显示镜像结构的 VQ input-output MSE loss 显著低于非镜像结构,尽管 VQ loss 本身略高 [§3.5, Fig 2]。[论文原文] 这表明镜像结构虽然 codebook 搜索过程的损失略大,但量化前后的信息保留更好,codebook 更具鲁棒性 [§3.5]。

4. **VQ vs PQ**: 两种量化方案性能接近,VQ 在 UTMOS 上略优 (4.218 vs 4.214),PQ 在 PESQ 上略优 (2.882 vs 2.862) [Table 1]。PQ 使用更大等效码本 (65536 vs 8192) 但带宽略高 (1.28 vs 1.04 kbps)。

5. **泛化性**: LJSpeech (非训练域) 上的结果验证了泛化能力,DS-Codec-VQ 在所有指标上优于 BigCodec 和 WavTokenizer (UTMOS 4.451 vs 4.385/3.870, PESQ 2.962 vs 2.822/1.948) [Table 2]。

## 局限性

1. **训练数据规模小**: 仅 LibriSpeech 1000h, 16kHz 英语朗读语音。对比 BigCodec 也仅用 LibriSpeech,但 WavTokenizer 使用了更多数据。未验证多语言、多采样率场景。
2. **缺少主观评估**: 仅有客观指标 (UTMOS/PESQ/STOI/F1),无人工 MOS 评分。UTMOS 超过 GT 的异常情况需要主观评估验证。
3. **Baseline 对比不够新**: 未对比 2024-2025 年最新 codec: SNAC (多尺度 RVQ)、SiTok (1-codebook Transformer)、TS3-Codec、Robust R-FSQ 等。
4. **下游任务未验证**: codec 的核心价值在于作为 TTS/语音生成的 tokenizer,但论文仅评估重建质量,未在 TTS 下游任务中验证 token 质量。
5. **DS-Codec-VQ 的贡献归因模糊**: 第二阶段直接使用 BigCodec checkpoint [§3.2],使得 DS-Codec-VQ 的改进中 BigCodec 预训练 codebook 与新训练策略的贡献难以分离。
6. **架构创新有限**: Encoder/Decoder 直接来自 BigCodec,Transformer Block 来自 LLaMA,量化方案 (VQ/PQ) 均为已有方法。核心贡献集中在训练策略。

## 点评

DS-Codec 提出了一个简洁且有实验支撑的训练策略: 利用镜像结构训练高质量 codebook,再切换到非镜像+Transformer 架构释放 decoder 潜力。Fig 2 的 MSE loss 对比有效展示了镜像结构在 codebook 鲁棒性上的优势,这是本文最有说服力的分析。

然而,论文的实验设计存在几个可以加强的地方: (1) DS-Codec-VQ 直接使用 BigCodec 官方 checkpoint 作为 decoder training 起点,这使得"两阶段训练"的贡献与"BigCodec 预训练"的贡献混杂; (2) 仅在 LibriSpeech 上训练和评估,泛化性结论依赖于 LJSpeech 上有限的验证; (3) UTMOS 超过 GT 的结果提示需要更审慎的评估,至少应包含 MOS 或 MUSHRA 主观实验。

在单码本 codec 的谱系中,DS-Codec 的两阶段训练策略是对 APCodec+ 策略的有效改进,证明了镜像→非镜像的切换顺序优于反向或联合训练。这一 insight (codebook 质量是瓶颈,应优先保证) 对其他 codec 设计有参考价值。

## 可复用的 idea

1. **先镜像后非镜像的两阶段训练**: 第一阶段用对称结构保证 quantization bottleneck 的质量,第二阶段冻结 bottleneck 并释放 decoder 自由度。这一策略适用于 bottleneck 训练不稳定或码本利用率不足的场景 (如 VQ-VAE 系列、image codec 等),对已充分收敛的系统收益可能有限。

2. **Warm-start decoder 而非重新初始化**: 保留第一阶段 decoder 权重 + 降低学习率,比重新初始化更高效。这与迁移学习中 fine-tuning 的最佳实践一致。

3. **用 quantization module 的 input-output MSE 作为 codebook 质量指标**: Fig 2 展示的 MSE loss 对比方法,可作为评估不同训练策略/架构对 codebook 鲁棒性影响的通用工具。

---

> [!review] 审阅状态
> 待审阅 — 见 `_review/DS-Codec-review.yml`

---

检索命中: [[Residual Vector Quantization]], [[Codebook Collapse]], [[Speech Tokenizer]], [[模型库/EnCodec|EnCodec]] | 过滤: [[Single-codebook vs Multi-codebook]](pending-review), [[Codec Training Objectives]](pending-review) | 未命中但可能相关: 无
