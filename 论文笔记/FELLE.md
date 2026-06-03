---
tier: deep
title: "FELLE"
aliases: [FELLE TTS, Token-Wise Coarse-to-Fine Flow Matching TTS]
authors: ["Hui Wang", "Shujie Liu", "Lingwei Meng", "Jinyu Li", "Yifan Yang", "Shiwan Zhao", "Haiyang Sun", "Yanqing Liu", "Haoqin Sun", "Jiaming Zhou", "Yan Lu", "Yong Qin"]
year: 2025
arxiv_id: "2502.11128"
source: "https://doi.org/10.1145/3746027.3755494"
venue: "ACM Multimedia 2025"
tags: [TTS, zero-shot, autoregressive, flow-matching, continuous-token, mel-spectrogram, coarse-to-fine, LLM-TTS]
level: deep
status: draft
concepts: ["[[Conditional Flow Matching]]", "[[LLM-based TTS]]", "[[Classifier-Free Guidance]]", "[[Mel Spectrogram]]"]
models: ["[[模型库/MELLE|MELLE]]"]
tasks: [TTS, zero-shot-TTS]
created: 2026-06-03
updated: 2026-06-03
kb_sources: ["[[Conditional Flow Matching]]", "[[LLM-based TTS]]", "[[Speech Tokenizer]]"]
---
tier: deep

## KB 背景

本文涉及以下已有知识:

- **[[Conditional Flow Matching]]** (confirmed): FELLE 的核心创新在于将 token-wise flow matching 引入 AR mel-spectrogram 生成。与现有 CFM 应用(如 CosyVoice 用 CFM 做全局 mel 生成)不同,FELLE 在每个 AR step 内部使用 coarse-to-fine flow matching,是 CFM 在 token-level 粒度的新应用范式。
- **[[LLM-based TTS]]** (confirmed): FELLE 属于 LLM-based TTS 中的连续表示路线,直接继承 MELLE 的 continuous mel AR 框架,但用 flow matching 取代了 MELLE 的 regression loss + latent sampling 方案。
- **[[Speech Tokenizer]]** (confirmed): FELLE 不使用离散 speech tokenizer,而是直接在连续 mel-spectrogram frames 上做 AR,绕过了量化损失。
- **[[Classifier-Free Guidance]]** [待确认]: FELLE 在 coarse 和 fine flow matching 阶段均使用 CFG,训练时以 p_drop=0.1 随机 mask speech prompt,推理时 w=1.6 [§4.4, §5.2]。
- **[[Mel Spectrogram]]** [待确认]: FELLE 工作在 80-dim log-mel spectrogram 空间,16kHz 采样率,80 维 mel filter + STFT [§5.1]。
- **MELLE** [待确认]: FELLE 直接建立在 MELLE 框架之上。MELLE 首创了连续 mel AR,使用 regression loss + spectrogram flux loss + latent sampling module。FELLE 用 flow matching 替代了这三个组件,目标是更好地捕捉 mel 分布的多模态性和时间依赖。

> [!summary] 速查
> - **一句话**: 在 AR mel-spectrogram LM 中引入 token-wise coarse-to-fine flow matching,用前一帧初始化 flow 先验分布,显著提升 speaker similarity (SIM-o: 0.619 vs MELLE 0.591)
> - **路线**: Text(phoneme) + Speech Prompt(mel) → AR Transformer LM(hidden z^i) → C2F-FM(coarse→fine flow matching, 每帧独立) → Mel Spectrogram → HiFi-GAN Vocoder → Waveform
> - **指标**: Continuation MOS 3.836 / Cross-Sentence MOS 4.157 (超越 MELLE 4.036); WER-C 1.53% / SIM-o 0.619 (Cross-Sentence) [Table 1, 2]
> - **可借鉴**: (1) 动态先验: 用前一帧 mel 初始化 flow 的 p0,减少 NFE 和提升连贯性 (2) Coarse-to-fine 频域分解: 低频粗+高频细两阶段 flow matching (3) 在 AR 框架中 per-token 使用 flow matching 替代回归损失
> - **局限**: 仅在 LibriSpeech 960h 上评估,未在大规模数据上验证; 16kHz 采样率; 模型规模较小(12层 Transformer + 18M C2F-FM)

## 核心问题

### WHY: 为什么要做这个工作?

现有 AR continuous TTS 方法(如 MELLE)存在两个关键局限 [§1]:

1. **回归损失假设过简**: MAE/MSE 假设单峰分布,但 mel spectrogram 的条件分布本质上是多模态的 — 同一文本可对应多种有效的声学实现。简单回归导致模糊、过度平滑的预测 [§1] [论文原文]
2. **时间建模不足**: 现有方法主要依赖 AR 架构隐式捕捉时间依赖,缺乏显式的帧间时序建模机制。mel spectrogram 帧之间存在强时间和频率相关性,忽略这些会影响连贯性 [§1]

FELLE 的解决方案: 用 flow matching 替代回归损失 (解决问题 1 的分布假设),并用前一帧初始化 flow 先验 (解决问题 2 的时间建模) [§1]。

### WHAT: 核心贡献

1. **Token-wise flow matching for AR TTS** [§4]: 在每个 AR 预测步中使用 flow matching 生成 mel frame,替代 regression + latent sampling。保留多模态分布建模能力,无需预设分布假设 [§4]
2. **Dynamic prior distribution** [§4.2, Eq.2]: 使用前一帧 mel x^{i-1} 初始化当前帧的 flow 先验 p0 = N(x_0^d | x^{i-1}, sigma^2 I),而非标准 Gaussian。减少 flow 搬运距离,提升时间连贯性,加速收敛(3 NFE vs vanilla 的 7+ NFE) [§4.2, Table 3]
3. **Coarse-to-fine flow matching (C2F-FM)** [§4.2]: 将 mel frame 分解为低频 coarse 和高频 fine 两阶段,各用独立 flow matching 网络。Coarse 阶段下采样到偶数帧,fine 阶段恢复全分辨率残差。捕捉跨频带相关性 [§4.2, Fig.2]

## 方法详解

### 1. Problem Formulation [§3.2]

延续 MELLE 框架: 将零样本 TTS 建模为 mel spectrogram 序列的条件自回归分布 [§3.2, Eq.1]:

p(X|y) = prod_{i=0}^{L-1} p(x^i | x^{<i}, y, hat{x})

其中 X = [x^0, ..., x^{L-1}] 为完整 mel 序列,y 为文本,hat{x} 为 speech prompt。每帧 x^i 属于 R^D (D = mel 维度 = 80) [§3.2]。

### 2. Autoregressive Language Model [§4.1]

| 参数 | 值 |
|------|------|
| 架构 | 12 Transformer blocks (与 VALL-E/MELLE 一致) |
| 每 block | 16 注意力头, FFN dim 4096 |
| Embedding dim | 1024 |
| 激活函数 | ReLU |
| 文本表示 | Phoneme-based tokens |
| Mel 输入映射 | 三层全连接 (pre-net) |
| 输出 | Hidden state z^i 作为 C2F-FM 的条件 |

**关键**: LM 输出 z^i 不直接预测 mel frame,而是作为 flow matching 的条件输入。每步 z^i 同时服务两个角色: (a) C2F-FM 的条件引导; (b) stop prediction module 的输入 [§4.4]。

### 3. Dynamic Prior Distribution [§4.2, Eq.2]

**核心创新**: 用前一帧 x^{i-1} 初始化 flow 的初始分布 [§4.2]:

p_0(x_0^d | x^{i-1}) = N(x_0^d | x^{i-1}, sigma^2 I)

- i=0 时(无前一帧),使用标准 Gaussian N(0, sigma^2 I) [§4.2]
- sigma^2 = 0.10 为最优超参(continuation 和 cross-sentence 任务均最优) [Table 5]

**WHY 比 vanilla prior 好** [§4.2, Table 3]:
- 相邻 mel 帧高度相关 → 前一帧是当前帧的良好近似 → flow 只需学习 "增量" 而非 "全量"
- 实验: 动态先验在 3 NFE 达最优; vanilla prior 需 7+ NFE 才收敛 [Table 3]
- WER 和 SIM 均显著改善: WER-C 1.72→1.53, SIM-o 0.466→0.513 (continuation) [Table 3]

### 4. Coarse-to-Fine Flow Matching [§4.2, Fig.2]

**两阶段分解** [§4.2]:

**Coarse 阶段**: 
- 将 mel frame x^i 下采样: x^{i,c} = Downsample(x^i),保留偶数位置频带 [§5.2]
- Coarse flow matching 预测向量场 v_t^c(x^{i,c}, z^i; theta_FM^c) [§4.2]
- 条件: 仅 LM hidden state z^i [§5.2]

**Fine 阶段**:
- 计算残差: x^{i,f} = x^i - Upsample(x^{i,c}) [§4.2]
- Fine flow matching 预测 v_t^f(x^{i,f}, z^i, x^{i,c}; theta_FM^f) [§4.2]
- 条件: z^i + coarse 输出 x^{i,c} (训练用 GT,推理用预测值) [§5.2]

**架构细节** [§5.2]:
- Coarse 和 fine 共享相同 backbone: 3 个 residual blocks (layer norm + dual FC + SiLU) [§5.2]
- Timestep embedding: sinusoidal PE + 2 FC + SiLU [§5.2]
- **关键架构差异**: coarse stage 使用 single linear projection 投影 LM output; fine stage 使用 additional layers 融合 coarse-mel 辅助信息 [§5.2]
- 上采样: zero-insertion at odd positions [§5.2]
- Optimal scale: 3x1024 (18M params total, coarse + fine) [Table 4]

**训练目标** [§4.2, §4.3, Eq.3]:

L_C2F-FM = E[||u_t^c - v_t^c||^2] + E[||u_t^f - v_t^f||^2]

总损失 [§4.3]: L = L_C2F-FM + lambda * L_cond + alpha * L_stop

- L_cond = ||z_i - x_i||_1 + ||z_i - x_i||_2^2 (hybrid L1+L2 条件损失, beta=0.1) [§4.3]
- L_stop: Binary cross-entropy, sigma=0.01 [§4.3, §5.2]

### 5. Classifier-Free Guidance [§4.2, Eq.4, §5.2]

训练: p_drop = 0.1 随机 mask speech prompt → 学习 conditional 和 unconditional 两种模式 [§5.2]

推理: w = 1.6 (最优 CFG scale) [§5.2, Fig.4]

hat{v}_t(x^*; .) = w * v_t^*(x^*, c; theta) + (1-w) * v_t^*(x^*, epsilon; theta)

其中 * 属于 {c, f},c 表示 full conditions,epsilon 表示 masked conditioning [§4.2, Eq.4]。

**CFG scale 分析** [§6.3, Fig.4]: w=1.0 (无 CFG) 到 w=1.6 WER 持续下降; w>1.6 WER 回升; SIM 在 w=2.2 达峰后下降。WER 和 SIM 对 CFG 响应不同,需平衡 [§6.3]。

### 6. 推理流程 [§4.4, Fig.2b]

1. LM 逐步生成 z^i [§4.4]
2. Coarse FM: 从 N(x^{i-1,c}, sigma^2 I) 出发,3 NFE (Euler method) 生成 hat{x}^{i,c} [§4.4, §5.2]
3. Fine FM: 融合 hat{x}^{i,c} + z^i,3 NFE 生成 hat{x}^{i,f} [§4.4]
4. 合并: hat{x}^i = hat{x}^{i,c} (upsampled) + hat{x}^{i,f} [§4.4]
5. Stop prediction: z_i 经 linear → sigmoid,超阈值则停止 [§4.4]
6. HiFi-GAN vocoder: mel → waveform [§5.2]

## 实验结果

### Predicted MOS [Table 1]

| System | Continuation | Cross-Sentence |
|--------|-------------|----------------|
| Ground Truth | 4.043 +/- 0.32 | 4.043 +/- 0.32 |
| VALL-E | 1.828 +/- 0.24 | 1.965 +/- 0.27 |
| MELLE | **3.843** +/- 0.38 | 4.036 +/- 0.25 |
| FELLE | 3.836 +/- 0.39 | **4.157** +/- 0.19 |

[agent解读] Cross-sentence 场景中 FELLE MOS 4.157 超越 GT 4.043,表明其在跨语句语境下的韵律/音色一致性优于自然录音(可能因去除了自然语音中的环境变化)。MELLE 在 continuation 略优 (3.843 vs 3.836),但差异在置信区间内。

### Objective Metrics [Table 2]

| System | Continuation WER-C | WER-H | SIM-r | SIM-o | Cross-Sentence WER-C | WER-H | SIM-r | SIM-o |
|--------|-------------------|-------|-------|-------|---------------------|-------|-------|-------|
| MELLE | **1.53** | **2.22** | 0.517 | 0.480 | 2.21 | **2.80** | 0.633 | 0.591 |
| FELLE | **1.53** | 2.27 | **0.539** | **0.513** | **2.20** | 2.89 | **0.654** | **0.619** |
| VALL-E R | 1.58 | 2.32 | 0.397 | 0.363 | 3.18 | 3.97 | 0.395 | 0.365 |

**核心发现** [§6.1]:
- WER 与 MELLE 相当(已接近 GT 上界),证明 flow matching 不损害可懂度 [§6.1]
- **SIM 显著提升**: SIM-o (cross-sentence) 0.619 vs MELLE 0.591 (+4.7%)。这验证了 dynamic prior + C2F-FM 对说话人特征建模的优势 [§6.1] [论文原文]
- FELLE 在 WER 和 SIM 之间取得了最佳平衡 [§6.1]

### Ablation: Prior Distribution [Table 3]

| Prior | Mechanism | Cont. WER-C | SIM-o | Cross WER-C | SIM-o |
|-------|-----------|-------------|-------|-------------|-------|
| Dynamic (ours) | C2F (ours) | **1.53** | **0.513** | **2.20** | **0.619** |
| Vanilla | C2F | 1.72 | 0.466 | 2.72 | 0.580 |
| Dynamic | HFM | 1.78 | 0.451 | 2.82 | 0.579 |
| Dynamic | DFM | 1.88 | 0.451 | 3.66 | 0.575 |

**关键证据** [§6.2]: 
- Dynamic prior vs vanilla prior: 各指标全面提升。Vanilla prior 需 7+ NFE,dynamic 仅需 3 [§6.2] [论文原文]
- C2F vs HFM (holistic): C2F 在所有指标优于一次性整帧 flow matching [§6.2]
- C2F vs DFM (decoupled): DFM 独立生成低频高频,缺少跨频带协调,WER 最差 [§6.2]

### C2F-FM Network Scaling [Table 4]

| Scale | Params | Cont. WER-C | SIM-o | Cross WER-C | SIM-o |
|-------|--------|-------------|-------|-------------|-------|
| 3x512 | 6M | 1.61 | 0.458 | 2.45 | 0.553 |
| **3x1024** | **18M** | **1.53** | **0.513** | **2.20** | **0.619** |
| 6x1024 | 34M | 1.63 | 0.490 | 2.28 | 0.581 |
| 12x1024 | 77M | 1.55 | 0.475 | 2.30 | 0.579 |

[agent解读] 3x1024 是 sweet spot。更大模型(6x1024, 12x1024)反而退化,可能因过拟合。这暗示 C2F-FM 的 capacity 需求不大 — 大部分信息已由 LM hidden state z^i 编码,FM 只需学习残差分布。

### NFE Analysis [Fig.3]

- WER 在 NFE=3-4 达最优后随 NFE 增加开始回升 [§6.3]
- SIM 随 NFE 增加持续下降 [§6.3]
- [agent解读] 过多 NFE 导致 over-refinement: 细节增加但可能偏离说话人特征和发音清晰度。3 NFE (Euler) 是最佳权衡。

## 与已有方法的差异

| 维度 | FELLE | MELLE | VALL-E | CosyVoice |
|------|-------|-------|--------|-----------|
| Token 类型 | 连续 mel frame | 连续 mel frame | 离散 EnCodec RVQ | 离散 semantic + CFM |
| 生成机制 | AR LM + token-wise FM | AR LM + regression + LSM | AR + NAR | AR LM + global CFM |
| Flow matching 粒度 | per-token (每帧独立) | 无 | 无 | per-utterance (全局) |
| Prior distribution | Dynamic (前一帧) | N(y_t, I) non-standard | N/A | N(0, I) |
| Coarse-to-fine | 是 (freq-domain) | 否 | 否 (层级 RVQ) | 否 |
| 训练数据 | LibriSpeech 960h | LibriSpeech 960h | LibriLight 60Kh | 170Kh |

## 关键设计选择与证据

| 设计选择 | WHY | 证据 |
|---------|-----|------|
| Flow matching 替代 regression | 避免单峰分布假设,捕捉 mel 多模态 | SIM 提升 +4.7% [Table 2] |
| Dynamic prior (前一帧) | 减少 flow 搬运距离,加速收敛 | 3 NFE vs 7+ NFE [Table 3, §6.2] |
| Coarse-to-fine 分解 | 低频结构优先,高频细节补充 | C2F > HFM > DFM [Table 3] |
| CFG w=1.6 | 增强 speaker conditioning | WER 最优; SIM 近最优 [Fig.4] |
| 3x1024 C2F-FM | 避免过拟合,capacity 匹配 | 最佳综合指标 [Table 4] |

## 局限性分析

1. **数据规模**: 仅 LibriSpeech 960h,远小于 VALL-E/CosyVoice 的 60K-170Kh。大规模效果未验证 [§5.1]
2. **采样率**: 16kHz,低于当前主流 24kHz/48kHz,限制高频保真度 [§5.1]
3. **Vocoder**: 使用 HiFi-GAN,质量受 vocoder 上限约束 [§5.2]
4. **评估**: 仅 LibriSpeech test-clean,缺乏 in-the-wild 和跨语言评估
5. **推理效率**: per-token FM 增加了每步计算量 (3 NFE x 2 stages = 6 FM evaluations per frame),但整体推理时间未报告

---

检索命中: [[Conditional Flow Matching]], [[LLM-based TTS]], [[Speech Tokenizer]] | 过滤: [[Classifier-Free Guidance]](pending-review), [[Mel Spectrogram]](pending-review) | 未命中但可能相关: MELLE(模型页, pending-review)
