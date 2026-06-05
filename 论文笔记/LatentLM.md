---
type: paper
tier: deep
title: "Multimodal Latent Language Modeling with Next-Token Diffusion"
arxiv_id: "2412.08635"
source: "Sources/LatentLM.pdf"
authors: [Yutao Sun, Hangbo Bao, Wenhui Wang, Zhiliang Peng, Li Dong, Shaohan Huang, Jianyong Wang, Furu Wei]
year: 2024
venue: "arXiv"
tags: [multimodal, latent-language-model, next-token-diffusion, sigma-VAE, continuous-representation, TTS, image-generation, unified-model]
concepts: ["[[DiffusionModel]]", "[[Classifier-FreeGuidance]]", "[[SpeechTokenizer]]", "[[LLM-basedTTS]]"]
models: ["[[MELLE]]"]
tasks: [image-generation, multimodal-LLM, zero-shot-TTS]
datasets: [ImageNet, LibriSpeech, MS-COCO, LibriHeavy]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[ConditionalFlowMatching]], [[LLM-basedTTS]], [[SpeechTokenizer]], [[Classifier-FreeGuidance]], [[DiffusionModel]], [[SemanticvsAcousticTokens]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **Continuous vs Discrete Token 路线**: 当前 LLM-based TTS 主流使用离散 codec tokens (如 VALL-E, CosyVoice 系列),通过 RVQ 量化将连续语音压缩为离散序列 [[SpeechTokenizer]]。但离散化存在有损压缩的根本矛盾: 低 bitrate 导致重建质量损失,高 bitrate 导致序列过长 [[SemanticvsAcousticTokens]]。MELLE 率先在连续 mel-spectrogram 空间做 AR TTS,但使用简化高斯分布假设,无法建模复杂语音分布 [[MELLE]]。
>
> **Diffusion 在 TTS 中的角色**: [[DiffusionModel]] 可建模任意分布,已在 TTS 声学模型 (Grad-TTS)、vocoder (DiffWave)、条件生成 ([[ConditionalFlowMatching]]) 中广泛应用。Flow matching 作为 diffusion 的 ODE 近亲,以更少推理步数实现高质量生成,在 CosyVoice/F5-TTS 等系统中作为 "fine stage" 渲染器。[[Classifier-FreeGuidance]] [待确认] 是 diffusion 条件生成的标准引导方法。
>
> **本文定位**: LatentLM 提出用 VAE 编码连续数据为 latent vectors,用 per-token diffusion head 逐 token 自回归生成这些 latent vectors,实现离散 (文本) 和连续 (图像/音频/视频) 数据的统一建模。这是 "next-token diffusion" 范式的奠基工作。
>
> 检索命中: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[Classifier-FreeGuidance]](pending-review), [[DiffusionModel]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 LatentLM 框架,用 sigma-VAE 将连续数据编码为 latent vectors,用 next-token diffusion 在 causal Transformer 中统一建模离散和连续数据
> - **路线**: 连续数据 → sigma-VAE Encoder → latent vectors → Causal Transformer (离散用 softmax head / 连续用 diffusion head) → sigma-VAE Decoder → 原始数据
> - **指标**: 图像生成 FID 2.24 (ImageNet 256, 479M params) [Table 1]; TTS SIM 0.697, WER-C 1.2 (LibriSpeech, 15fps) [Table 4]; MLLM FID 14.54 + VQAv2 38.72 [Table 3]
> - **可借鉴**: sigma-VAE 防止 variance collapse 的设计; per-token lightweight diffusion head 复用 Transformer backbone 计算; 统一离散+连续的 causal 建模范式
> - **局限**: TTS 实验规模较小 (仅 LibriSpeech); 未与 CosyVoice/Seed-TTS 等工业级系统对比; CFG 对 diffusion head 的影响未深入分析

## 核心问题

多模态生成模型需要统一处理离散数据 (文本/代码) 和连续数据 (图像/音频/视频)。现有三种路线各有根本局限 [§1]:

1. **VQ-VAE 离散化路线** (RPG+21, WCW+23): 将所有连续数据量化为离散 tokens → 有损压缩瓶颈,低压缩比导致序列过长 [论文原文]
2. **Diffusion 统一路线** (BNX+23b, TYZ+23): 将离散数据也放入 diffusion 框架 → 损害离散数据的建模性能 [论文原文]
3. **权重共享路线** (ZYB+24): 连续数据用 sequence-level diffusion,离散数据用 next-token prediction,但它们有不同的目标和实现 (bidirectional vs causal attention),且双向 diffusion 限制变长序列 [论文原文]

LatentLM 的核心洞察: **在 causal Transformer 的每个位置上挂一个轻量 diffusion head**,对离散 token 用 softmax prediction、对连续 token 用 diffusion generation,两者共享同一 backbone,无冲突 [§1] [论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

LatentLM 由三个部分组成 [§2, Fig 2]:

1. **sigma-VAE**: 将连续数据 (图像/语音) 编码为 latent vectors,解码回原始数据 [§2.3]
2. **Causal Transformer**: 共享 backbone,基于 causal masking 做自回归生成 [§2]
3. **Diffusion Head**: 轻量神经网络,挂在每个 Transformer hidden state 上,用 diffusion 生成连续 latent vectors [§2.1]

具体流程 [§2, Eq. 1]:
- 输入序列 x = x_1 ... x_N 包含离散和连续 tokens
- 离散 token: lookup table 获取 embedding
- 连续 token: sigma-VAE encoder 压缩为 latent vector
- Causal Transformer 处理,输出 hidden states h_1 ... h_N
- 离散位置: softmax(h_i * W_v) → next-token prediction
- 连续位置: Diffusion(h_i) → 生成 latent vector → VAE decoder 恢复原始数据

### 关键设计选择

#### 1. Next-Token Diffusion (per-token diffusion head) [§2.1]

核心区别于 sequence-level diffusion: **每个 continuous token 位置有独立的 diffusion 过程**,条件是该位置的 Transformer hidden state h_i [论文原文]。

- 使用 DDPM 或 flow matching 作为 diffusion 实现 [§2.1]
- Forward process: 标准 Gaussian noise 添加,x_i^t = sqrt(alpha_t) * x_i + sqrt(1-alpha_t) * epsilon [Eq. 2]
- Reverse process: 轻量网络 epsilon_theta(x_i^t, t, h_i) 预测噪声 [Eq. 3]
- 训练: L_Diff = E[||epsilon - epsilon_theta(x_i^t, t, h_i)||^2] [Eq. 3]
- 推理: 从纯 Gaussian noise 出发,用 DPM-Solver 迭代去噪 [§2.1]

**Head Architecture**: 残差架构,包含 pre-RMSNorm + feedforward layers,使用 AdaLN-Zero 同时条件化 timestep t 和 Transformer 输出 h_i [§2.1] [论文原文]。

**为什么 per-token 而非 sequence-level?** [agent 解读] Sequence-level diffusion 需要双向注意力 (全序列同时去噪),与 causal Transformer 的单向注意力矛盾。Per-token diffusion 只需在每个位置独立去噪,完美兼容 causal 架构,且 Transformer backbone 只做一次 forward pass。

#### 2. sigma-VAE [§2.3, Fig 3]

解决标准 VAE 在自回归建模中的 **variance collapse** 问题 [论文原文]:

- 标准 VAE: z = mu + sigma * epsilon, sigma 是可学习参数 → 训练时 sigma 趋向 0,latent space 退化为确定性映射
- sigma-VAE: **固定 variance σ 为从 N(0, C_σ) 采样的标量**,不可学习 [Eq. 5]
  - mu = Encoder_phi(x)
  - z = mu + sigma * epsilon, epsilon ~ N(0,1), sigma ~ N(0, C_σ)
  - x_hat = Decoder_psi(z)
- 训练目标: minimize ||x_hat - x||^2 + beta * ||mu||^2 [Eq. 6]

**为什么固定 variance?** 自回归生成引入采样不确定性,latent space 的 variance 影响 diffusion head 的性能。较大 variance 使模型对 exposure bias 更鲁棒 [§2.3, Fig 6] [论文原文]。标准 VAE 低 variance tokenizer (为 image-level diffusion 调优的) 对 LatentLM 是次优的 [§3.1.3] [论文原文]。

#### 3. 训练与推理 [§2.2]

- 联合训练: L = L_LM + alpha * L_Diff [§2.2]
- L_LM: 标准交叉熵 (离散 tokens)
- L_Diff: diffusion 去噪损失 (连续 tokens)
- 实际训练时每个 forward pass 采样 4 个 diffusion timesteps [§2.2]
- 推理时 Transformer backbone 单次 forward,仅 diffusion head 需多步去噪 [§2.2] [论文原文]
- 使用 special tokens <BOD>/<EOD> 标记 diffusion head 的使用区间 [§2.2]

### 训练策略

**Image generation** [§3.1.1]: 24 layers, hidden 1024, heads 16, FFN 2730, diffusion head 6 layers。Batch 2048, 250K steps, AdamW, cosine lr 5e-4, CFG scale 1.65 [§3.1.1]。

**Multimodal LLM** [§3.2.1]: 1.3B Transformer, 24 layers, hidden 2048, seq len 4096, tiktoken-cl100k_base。Text:Image-text:Interleaved = 2:1:1。Batch 4M tokens, 50K steps (200B tokens) [§3.2.1]。

**TTS** [§3.3.1]: sigma-VAE 使用 ConvNeXt blocks, 1D causal convolution。压缩比 1600x/3200x/6400x。LatentLM 24 layers, 1024 hidden, 16 heads, FFN 4096。Diffusion head 3 layers feedforward [§3.3.1]。

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| FID↓ (Image, 256x256) | 2.24 (LatentLM-L, 479M) | 2.59 (GIVT-Causal-L+A, 1.67B) | ImageNet | [Table 1] |
| FID↓ (Image, 384x384) | **2.51** | 3.19 (256x256) | ImageNet | [Table 2] |
| Text Valid PPL↓ (MLLM) | **2.73** | 2.74 (Transfusion), 2.79 (VQ-MLLM) | - | [Table 3] |
| Text-to-Image FID↓ (MLLM) | **14.54** | 16.10 (Transfusion), 16.92 (VQ-MLLM) | MS-COCO | [Table 3] |
| VQAv2↑ (MLLM) | **38.72** | 35.36 (Transfusion), 30.19 (VQ-MLLM) | VQAv2 | [Table 3] |
| SIM↑ (TTS, 15fps, ref prompt) | **0.697** | 0.643 (VALL-E 2), 0.625 (MELLE) | LibriSpeech test-clean | [Table 4] |
| WER-C↓ (TTS, 15fps, ref prompt) | **1.2** | 1.5 (VALL-E 2), 1.5 (MELLE) | LibriSpeech test-clean | [Table 4] |
| WER-H↓ (TTS, 15fps, ref prompt) | **1.8** | 2.4 (VALL-E 2), 2.1 (MELLE) | LibriSpeech test-clean | [Table 4] |
| Throughput↑ (3.8B, batch 128) | 2.47x DiT | DiT baseline | H100 | [Fig 7a] |

**TTS 关键发现** [§3.3.4]:
- sigma-VAE 在 1600x 压缩比下实现 15 fps frame rate,比 VALL-E 2 (75 fps) 少 **10x 解码步数**,同时 SIM 和 WER 均更优 [Table 4] [论文原文]
- 进一步提升到 3200x (7.5fps) 和 6400x (3.75fps),质量平缓下降但推理速度大幅提升 [Table 4] [论文原文]
- sigma-VAE tokenizer 在 1600x 压缩比下重建质量优于 EnCodec (40x), DAC (10x-160x) 等主流 codec [Table 5] [论文原文]

**Image generation 关键发现** [§3.1]:
- LatentLM 在 causal generation 中以 479M params 超越 GIVT (1.67B) 和 LlamaGen-XXL (1.4B) [Table 1] [论文原文]
- Scaling curves 显示 LatentLM 在所有模型尺寸上 FID 均优于 DiT [Fig 4] [论文原文]
- sigma-VAE 偏好更大 variance 的 tokenizer,与 image-level diffusion 模型相反 [§3.1.3, Fig 6] [论文原文]

## 局限性

1. **TTS 实验范围有限**: 仅在 LibriSpeech test-clean 上评估,未涉及多语言、多说话人大规模场景 [agent 解读]
2. **TTS 对比基线偏旧**: 未与 CosyVoice 2/Seed-TTS 等 2024 工业级系统对比 [agent 解读]
3. **sigma-VAE C_σ 选择**: C_σ 是关键超参数,Fig 6 显示不同 CFG 下最优值不同,需仔细调参 [§3.1.3] [论文原文]
4. **推理效率**: 虽然 diffusion head 轻量,但每个连续 token 仍需多步去噪 (实验中 5-20 步) [§2.1]
5. **仅英语 TTS**: 训练数据为 LibriHeavy (英语有声书),未验证跨语言能力 [§3.3.2]

## 点评

LatentLM 的核心贡献在于提出了一个优雅的统一框架: **per-token diffusion head + sigma-VAE + causal Transformer**。这三个组件各自解决一个关键问题:

1. Per-token diffusion 解决了 causal 架构与连续数据生成的兼容性问题 — 不需要双向注意力
2. sigma-VAE 解决了 autoregressive 场景下的 variance collapse 问题 — 固定 variance 而非学习它
3. Causal Transformer 提供了离散和连续数据的统一建模平台 — softmax head 和 diffusion head 共存

**最大亮点**: 在 TTS 上以 10x 更少的解码步数超越 VALL-E 2,证明连续表示 + 高压缩比的路线优于离散 codec tokens [Table 4]。这为后续 CLEAR 和 VibeVoice 等工作奠定了基础。

**关注点**: sigma-VAE 与标准 VAE 的差异 (固定 vs 学习 variance) 看似微小但对 AR 建模至关重要 — 这是一个值得在其他连续 AR 系统中验证的设计。

## 可复用的 idea

1. **sigma-VAE 设计模式**: 当 VAE latent 被用于自回归 (非 diffusion) 下游时,固定 variance 防止 collapse 是通用技巧
2. **Per-token diffusion head**: 轻量 residual network + AdaLN-Zero,复用 Transformer backbone 计算,适用于任何需要在 LM 框架中生成连续数据的场景
3. **Compression ratio 与 latent dimension 的 trade-off** [Table 6]: 增大 latent dimension 可以补偿高压缩比带来的重建损失
4. **BOD/EOD special tokens**: 简洁地标记离散/连续模态切换点

---

检索命中: [[ConditionalFlowMatching]], [[LLM-basedTTS]], [[SpeechTokenizer]], [[SemanticvsAcousticTokens]] | 过滤: [[Classifier-FreeGuidance]](pending-review), [[DiffusionModel]](pending-review) | 未命中但可能相关: 无
