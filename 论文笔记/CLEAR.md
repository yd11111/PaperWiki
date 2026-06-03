---
type: paper
tier: deep
title: "CLEAR: Continuous Latent Autoregressive Modeling for High-quality and Low-latency Speech Synthesis"
arxiv_id: "2508.19098"
source: "https://arxiv.org/abs/2508.19098"
authors: [Chun Yat Wu, Jiajun Deng, Guinan Li, Qiuqiang Kong, Simon Lui]
year: 2025
venue: "arXiv"
tags: [TTS, zero-shot, continuous-representation, autoregressive, rectified-flow, VAE, streaming, low-latency]
concepts: ["[[Conditional Flow Matching]]", "[[Classifier-Free Guidance]]", "[[Speech Tokenizer]]", "[[LLM-based TTS]]", "[[Diffusion Model]]"]
models: ["[[MELLE]]"]
tasks: [zero-shot-TTS, streaming-TTS]
datasets: [LibriSpeech, LibriHeavy, LibriTTS]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[Conditional Flow Matching]], [[LLM-based TTS]], [[Speech Tokenizer]], [[Diffusion Model]], [[Classifier-Free Guidance]], [[Diffusion-based TTS]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **AR-based TTS 的离散 vs 连续之争**: 主流 LLM-based TTS (VALL-E, CosyVoice) 使用离散 codec tokens + 自回归生成 [[LLM-based TTS]]。离散化存在有损压缩问题,高比特率序列过长,低比特率质量损失 [[Speech Tokenizer]]。MELLE 率先在连续 mel-spectrogram 空间做 AR,但使用高斯分布假设限制了建模能力 [[MELLE]]。
>
> **Flow matching 在 TTS 中的角色**: [[Conditional Flow Matching]] 已广泛用于 coarse-to-fine TTS 的 "fine stage" (CosyVoice 用 OT-CFM, F5-TTS 用 flow matching),但多作为独立的第二阶段。Rectified flow 是 flow matching 的一种变体,学习直线 ODE 路径 [论文原文]。
>
> **本文定位**: CLEAR 将 rectified flow 作为轻量 MLP head 直接挂在 AR language model 上,用增强 VAE 编码连续 latent,实现**单阶段**端到端连续值 AR TTS。与 LatentLM 思路相似但独立发展,专注 TTS 场景并强调流式和低延迟。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓ | 过滤: [[Classifier-Free Guidance]](pending-review), [[Diffusion Model]](pending-review), [[Diffusion-based TTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 CLEAR,用增强 wav-VAE (shortcut connections + 2048x downsampling) 编码连续 latent + MLP-based rectified flow head 逐 token 建模连续分布,实现单阶段零样本 TTS,RTF 0.18 + 96ms 流式首帧延迟
> - **路线**: Text (phoneme) + Speech prompt → VAE encoder → continuous latents → AR Transformer (conditioning h_k) → Rectified Flow Head (per-token) → continuous latents → Causal VAE decoder → waveform
> - **指标**: WER 1.88%, UTMOS 4.22, SIM-o 0.59, RTF 0.18 (Base) / 0.29 (Large) on LibriSpeech Subset-B [Table 1]; streaming FFL 96ms [Table 4]; 仅 78 AR decoding steps per 10s
> - **可借鉴**: 1) shortcut-connected VAE 实现 2048x 压缩; 2) MLP rectified flow head 支持流式 (不需等全序列); 3) logit-normal timestep sampling 加速收敛; 4) 辅助 velocity direction loss
> - **局限**: 仅英语评估; SIM-o 相对偏低 (0.55-0.59); 不支持多 token 并行预测

## 核心问题

当前 AR-based TTS 使用离散 audio tokens 面临两个根本问题 [§1]:

1. **有损压缩**: neural audio codecs 在 tokenization 时引入量化损失,要达到接近连续表示的保真度需要极高比特率,导致序列极长 (如 1024 codebook size 下 1s 语音需要数百个 tokens) [§1] [论文原文]
2. **训练困难**: 离散 codec 训练对梯度近似技巧敏感 (如 VQ-GAN 需要 codebook re-initialization 等辅助技术) [§1] [论文原文]

两个额外的工程挑战:
- **概率分布建模**: 连续表示比离散 tokens 携带更丰富信息,需要能建模任意分布的方法 (MELLE 的高斯假设太简化) [§1] [论文原文]
- **训练/推理效率**: diffusion 技术可建模连续分布,但 DiT 类架构 (如 ARDiT) 需要等全序列生成后才能开始去噪,延迟大 [§1] [论文原文]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CLEAR 架构由三个组件组成 [§3, Fig 1 left]:

1. **wav-VAE Encoder (紫色)**: 将 16kHz 语音波形映射为紧凑连续 latent 序列
2. **Autoregressive Language Model (蓝色)**: 基于 Transformer decoder,接收文本 embedding + VAE latent,输出 conditioning vector h_k
3. **Rectified Flow Head (绿色)**: MLP 网络,每个 token 位置独立,以 h_k 为条件生成下一帧 latent

### 关键设计选择

#### 1. Enhanced wav-VAE with Shortcut Connections [§3.4, Fig 1 right, Appendix C.1]

- 架构: 对称 encoder-decoder,基于 oobleck blocks + snake activation [§3.4] [论文原文]
- **Shortcut connections**: 在 encoder downsample 和 decoder upsample 层之间加入 **非参数化** deep residual connections (space-to-channel 变换 + channel averaging) [§3.4] [论文原文]
- 为什么 shortcut? 在 2048x 极度压缩下,标准 VAE 难以保留信号保真度。Shortcut 让网络专注于学习残差变换,缓解优化困难 [§3.4] [论文原文]
- Downsampling ratio D = 2048: strides [2,4,4,8,8],从 16kHz 波形到 ~7.8 Hz latent [Appendix C.2]
- 训练: 多任务判别器 L_AE = L_ms + alpha * L_fm + beta * L_adv + gamma * L_kl, alpha=1.0, beta=5.0, gamma=0.1 [§3.4]
- VAE reconstruction WER 2.89%, SIM-o 0.63, UTMOS 4.08 → 接近 ground truth (2.47%, 0.69, 4.09) [Appendix C.1]

#### 2. MLP-based Rectified Flow Head [§3.2, Eq. 3-5]

- 使用 rectified flow (RF) 而非 DDPM: 定义 straight-line ODE 路径 y_k^t = (1-t)*y_k^0 + t*y_k^1 [§3.2] [论文原文]
- 去噪模块: MLP 网络由多个 residual blocks (ResBlock1D) 组成,每块包含 LayerNorm + Linear + SiLU + Linear + residual connection [§3.2]
- **条件化**: MLP 以 h_k (AR 模型 hidden state) 为条件去噪 [§3.2]
- 损失函数: rectified flow loss L_RF = E[||u_k - v(y_k^t | h_k, t)||^2] [Eq. 3] + auxiliary velocity direction loss L_D = 1 - E[cos(u_k, v(y_k^t | h_k, t))] [Eq. 4]
- **Logit-normal timestep sampling** [Eq. 5]: 对中间 timestep 加权更多 (因为中间段更难预测),参数 m=0, s=1 [§3.2] [论文原文]

**为什么 MLP 而非 DiT?** [论文原文] DiT 需等待全部 autoregressive hidden states 生成后才开始去噪 (因为 attention 需要全序列)。MLP 在每个 token 位置独立运行,收到 h_k 就立即开始去噪,支持流式输出 [§3.2]。

#### 3. Classifier-Free Guidance for Continuous Tokens [§3.3]

- 训练时以 20% 概率将所有 text embeddings 设为 0 [§3.3]
- 推理时: v_guided = w * v(y|h_k, t) + (1-w) * v(y|h_bar_k, t) [Eq. 6], 其中 h_bar_k = f(y_{<k}, 0) 为无文本条件 [§3.3]
- CFG scale w 默认 2.5, denoising steps 10 [§4.2]
- **CFG 策略选择**: 论文发现将 text embeddings 设 0 效果优于替换 h_k 为 null/learnable embedding [§3.3 footnote 2] [论文原文]

#### 4. 流式推理 [§5.3]

- 利用 causal VAE decoder + MLP flow head 的 per-token 独立性
- Chunk size Omega: 累积 Omega 个 tokens 后送入 causal VAE decoder 生成波形 [§5.3]
- 重叠区域: 相邻 chunks 重叠 Psi 帧,fade-in-fade-out 过渡 [§5.3]
- Omega=4 时 FFL (first-frame latency) = 96ms, WER 2.34%, UTMOS 4.27 [Table 4] [论文原文]

### 训练策略

**单阶段联合训练** [§3.5]:
- 输入序列 [S, x, T, y]: S=start token, x=text (phoneme), T=turn token, y=VAE latent [§3.5]
- 总损失 L = L_RF + L_D + L_s (stop prediction binary CE) [§3.5]
- Teacher forcing 训练
- Text: G2P → phonemes, vocabulary 203 [§4.2]
- EMA on model weights [§4.2]

**Model configurations** [§4.2]:
- CLEAR-Base: 24 Transformer blocks, embed 1024, FFN 4096, flow head 6 ResBlock1D layers x 1024 dim → 439M params
- CLEAR-Large: embed 1280, FFN 5120 → 686M params
- Training: Base 118 GPU-hours (6x RTX 4090), Large 112 GPU-hours (2x H20)

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER%↓ (Subset-A) | **1.83** (Large) | 2.32 (MegaTTS), 1.81 (NaturalSpeech3) | LibriSpeech test-clean | [Table 1] |
| UTMOS↑ (Subset-A) | **4.26** (Large) | 4.02 (MegaTTS), 4.30 (NaturalSpeech3) | LibriSpeech test-clean | [Table 1] |
| WER%↓ (Subset-B) | **1.88** (Large) | 2.21 (CLEAR-Base), 2.39 (DiTAR) | LibriSpeech test-clean | [Table 1] |
| UTMOS↑ (Subset-B) | 4.22 (Large) | **4.22** (CLEAR-Base), 4.22 (DiTAR) | LibriSpeech test-clean | [Table 1] |
| SIM-o↑ (Subset-B) | 0.59 (Large) | 0.59 (CLEAR-Base), **0.67** (DiTAR) | LibriSpeech test-clean | [Table 1] |
| RTF↓ | **0.18** (Base) | 0.29 (Large), 0.31 (F5-TTS), 0.55 (MELLE) | - | [Table 2] |
| Avg Decoding Steps↓ | **78** | 750 (VALL-E), 620 (MELLE), 400 (DiTAR) | - | [Table 2] |
| N-MOS↑ (Subjective) | **4.09** | 3.90 (CosyVoice-2), 3.75 (F5-TTS) | Subset-B | [Table 3] |
| Q-MOS↑ (Subjective) | **4.14** | 4.10 (CosyVoice-2), 3.45 (F5-TTS) | Subset-B | [Table 3] |
| Streaming WER%↓ (Omega=4) | 2.34 | 2.21 (non-streaming) | Subset-B | [Table 4] |
| Streaming FFL↓ (Omega=4) | **96ms** | - | Subset-B | [Table 4] |

**关键发现**:

1. **低延迟优势显著**: CLEAR-Base RTF 0.18,仅需 78 步 AR decoding (对比 VALL-E 750 步, MELLE 620 步),因为 VAE 的 2048x 压缩将 10 秒语音压缩到仅 ~78 个 latent tokens [Table 2] [论文原文]
2. **单阶段超越两阶段系统**: CLEAR-Base 在 Subset-B 上 WER/UTMOS 优于 CosyVoice (two-stage discrete AR + flow matching) 和 F5-TTS (NAR flow matching) [Table 1] [论文原文]
3. **主观质量领先**: N-MOS 4.09 超越 ground truth (4.01), Q-MOS 4.14 超越 ground truth (4.00) [Table 3] [论文原文]
4. **流式近乎无损**: chunk size 4 时 96ms 延迟,WER/UTMOS 与非流式几乎持平 [Table 4] [论文原文]
5. **SIM-o 是弱项**: 0.55-0.59 低于 NaturalSpeech3 (0.67) 和 DiTAR (0.67),作者承认 speaker similarity 需要改进 [Appendix A] [论文原文]

## 局限性

1. **Speaker similarity 不足**: SIM-o 0.55-0.59 低于多数 baseline (0.64-0.67) [Table 1, Appendix A] [论文原文]
2. **仅英语评估**: 训练数据为 LibriHeavy (英语有声书),未验证多语言能力 [Appendix A] [论文原文]
3. **无多 token 并行预测**: 逐 token 生成,未探索 multi-token prediction 加速 [Appendix A] [论文原文]
4. **VAE 与 LM 分离训练**: VAE 先训练,LM 在 frozen VAE latent 上训练,未探索联合优化 [agent 解读]
5. **Phoneme 输入**: 依赖 G2P 前端,非端到端 text 输入 [§4.2]

## 点评

CLEAR 与 LatentLM 几乎同时期提出了相似的核心思想 (连续 VAE latent + per-token diffusion/flow head + AR LM),但 CLEAR 专注 TTS 场景并带来了几个重要的工程贡献:

1. **Shortcut-connected VAE**: 2048x 极端压缩下仍保持高质量重建 (WER 2.89%, UTMOS 4.08),是工程上的关键突破 [Appendix C.1]
2. **MLP flow head 支持流式**: 与 DiT-based 方案 (DiTAR, ARDiT) 形成鲜明对比 — 牺牲一点表达力换取流式能力和推理速度
3. **96ms 首帧延迟**: 对实际产品部署极有价值

**待改进**: SIM-o 是明显短板。作者在 Appendix A 指出未来将 speaker embedding 集成到 LM 和 flow head 中,这可能是 VibeVoice 等后续工作已解决的方向。

## 可复用的 idea

1. **Shortcut-connected VAE**: 非参数化 space-to-channel 变换实现极端压缩下的高保真重建
2. **MLP rectified flow head**: 轻量、per-token 独立、支持流式,可替代 DiT 在延迟敏感场景
3. **Logit-normal timestep sampling**: 对中间 timestep 加权更多,加速 flow matching 收敛 [Eq. 5]
4. **Velocity direction loss**: cosine similarity 辅助损失加速 flow 方向学习 [Eq. 4]
5. **CFG on text prefix**: 将 text embedding 设 0 (而非替换 h_k) 效果更好 [§3.3]

---

检索命中: [[Conditional Flow Matching]], [[LLM-based TTS]], [[Speech Tokenizer]] | 过滤: [[Classifier-Free Guidance]](pending-review), [[Diffusion Model]](pending-review), [[Diffusion-based TTS]](pending-review) | 未命中但可能相关: 无
