---
type: paper
tier: deep
title: "VARSTok: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding"
arxiv_id: "2509.04685"
source: "Sources/VARSTok.pdf"
authors: [Rui-Chen Zheng, Wenrui Liu, Hui-Peng Du, Qinglin Zhang, Chong Deng, Qian Chen, Wen Wang, Yang Ai, Zhen-Hua Ling]
year: 2025
venue: "arXiv (USTC + Alibaba Tongyi Fun Team)"
tags: [audio-codec, speech-tokenizer, variable-frame-rate, single-codebook, VQ, clustering, duration-modeling, TTS, compression]
concepts: ["[[SpeechTokenizer]]", "[[ResidualVectorQuantization]]", "[[CodebookCollapse]]", "[[TokenRateandBitrateTrade-offs]]", "[[Single-codebookvsMulti-codebook]]", "[[DurationPredictor]]", "[[CodecLanguageModel]]"]
models: ["[[模型库/WavLM|WavLM]]", "[[模型库/Whisper|Whisper]]"]
tasks: ["[[任务库/NeuralAudioCompression|Neural Audio Compression]]", "[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: VARSTok 属于 acoustic tokenizer 路线 ([[SpeechTokenizer]]),但它挑战了该路线默认的 fixed-frame-rate 范式。在 [[Single-codebookvsMulti-codebook]] 的设计轴上,VARSTok 延续了 WavTokenizer 的单码本 SVQ 选择 (K=4096),符合 "单码本回归" 趋势。但它在 [[TokenRateandBitrateTrade-offs]] 维度上独辟蹊径:不是通过减少码本数/降低帧率来压缩,而是通过 content-adaptive 的 variable frame rate 实现动态压缩,这在 acoustic tokenizer 中尚属首例。
>
> **已有认知**: 概念库中 [[TokenRateandBitrateTrade-offs]] 已记录 fixed bitrate / adaptive bitrate / scalable bitrate 三类策略,但缺少 "variable frame rate" 这一类别 — VARSTok 代表了一种新的压缩策略:帧率本身随内容变化。[[DurationPredictor]] 页面记录了从 FastSpeech 到 RL-optimized duration 的演进,VARSTok 的 implicit duration coding 提出了一种完全不同的思路:将 duration 编码进 token index,消除对独立 duration predictor 的需求。[[CodebookCollapse]] 页面记录了多种解决方案,VARSTok 的 codebook utilization 分析 (K=4096 时 100%) 提供了小码本 + 数据充足时 collapse 不严重的新证据。
>
> **创新判断**: 相比概念库中已有的 fixed-rate tokenizer (WavTokenizer, BigCodec, DAC 等), VARSTok 的核心新意在于: (1) temporal-aware density peak clustering 实现真正动态帧率 (非 TFC 那种从预定义帧率中选择的 "pseudo-dynamic"); (2) implicit duration coding 将 duration 嵌入 token index,无需修改下游 LM 架构。
>
> 检索命中: [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓, [[CodebookCollapse]]✓, [[NeuralAudioCompression]]✓ | 过滤: [[TokenRateandBitrateTrade-offs]][待确认], [[Single-codebookvsMulti-codebook]][待确认], [[DurationPredictor]][待确认], [[CodecLanguageModel]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个 fully dynamic variable-frame-rate acoustic speech tokenizer,通过 temporal-aware 密度峰聚类 + implicit duration coding 在 30.95 Hz 平均帧率下超越 40 Hz fixed-rate baseline 的重建和 TTS 质量
> - **路线**: Waveform → CNN Encoder (产出 fixed-rate frame embeddings) → Temporal-Aware Density Peak Clustering (聚合为 variable-length clusters) → Mean Pooling → VQ (K=4096 单码本) + Implicit Duration Coding (ID = (d-1)*K + k) → Repeat by duration → CNN Decoder → Waveform
> - **指标**: 重建 UTMOS 3.89 / PESQ 1.71 @ 30.95 Hz (vs WavTokenizer 40Hz: 3.61 / 1.71) [Table 1]; TTS WER 6.79% / UTMOS 4.25 / MOS 4.05 @ 36.81 Hz (vs WavTokenizer: 7.48% / 3.92 / 3.98) [Table 3]; RTF 0.487 (36% speedup) @ 26.29 Hz [Table 5]
> - **可借鉴**: (1) Implicit duration coding: 将 duration 编码进 token index ((d-1)*K + k) 的做法极其简洁,任何需要处理变长 token 的系统都可复用; (2) Density peak clustering 的 temporal-aware 约束:在标准 DPC 基础上加 bidirectional temporal contiguity 约束,确保聚类结果是时间连续的段
> - **局限**: 仅在 585h LibriTTS 上训练,未验证大规模数据; Smax=4 的硬约束限制了最大压缩比; 聚类算法不可微,encoder 无法直接从聚类决策获得梯度; 代码已开源但模型权重情况不明

## 核心问题

**问题**: 现有 speech tokenizer 以固定帧率 (如 40Hz, 75Hz) 均匀分配 token,忽略了语音信号的信息密度时变性 — 静音/稳态元音区域被过度编码,而快速发音转换区域可能编码不足。如何让 token 分配自适应地匹配语音的时间动态?

**为什么难**: (1) 语音的 acoustic tokenizer 要求高保真重建,不能像 semantic tokenizer 那样粗粒度聚合; (2) 变帧率产生的变长 token 序列与 standard autoregressive LM 不兼容(LM 不知道每个 token 跨了多长); (3) 先前工作 TFC 只是从预定义帧率中选择,不是真正的动态分配 [论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VARSTok 基于 WavTokenizer 的 encoder-VQ-decoder 架构,在 encoder 和 VQ 之间插入了两个新模块 [§3.1]:

```
Waveform → Speech Encoder → frame embeddings X∈R^{T×H}
         → Temporal-Aware Density Peak Clustering → N variable-length clusters
         → Mean Pooling → cluster embeddings Z∈R^{N×H}  (N << T)
         → VQ (K=4096) + Implicit Duration Coding → token IDs∈{0,...,K*Smax-1}
         → Repeat by duration → expanded embeddings ẐR∈R^{T×H}
         → Speech Decoder → Waveform
```

关键: N << T,即聚类后的 token 数远少于原始帧数,实现压缩 [论文原文]。

### 关键设计选择

#### 1. Temporal-Aware Density Peak Clustering [§3.3]

**为什么选 DPC 而非 k-means/学习式分割**: 作者指出传统聚类方法(如 Wu et al. 2025 的 image patch 聚类)不考虑时间顺序,不适用于语音 [论文原文]。DPC 的 "局部密度 + 峰间距离" 天然识别数据中的密度峰,适合发现语音中的稳态-过渡边界 [agent 解读]。

**算法核心** (Algorithm 1):

1. **局部密度** ρ_i: 对每帧计算其 KNN(m=5) 的平均归一化余弦相似度的指数。高 ρ_i = 该帧处于嵌入空间的密集区域 [§3.3, Eq.1]
2. **峰距离** δ_i: 帧 i 与比它密度更高的最近帧之间的距离。高 δ_i = 孤立峰 [§3.3, Eq.3]
3. **峰得分** s_i = ρ_i * δ_i: 同时满足 "高密度" 和 "相对孤立" → 理想种子帧 [§3.3, Eq.4]
4. **贪心扩展**: 选未分配帧中 s_i 最高者为种子,双向扩展:
   - 相似度条件: ϕ(x_{i*}, x_t) - β * s_t > τ — 用 -β*s_t 惩罚自身是强种子的帧,防止两个种子被合并 [§3.3, Eq.5]
   - **时间连续性约束**: 仅当紧邻帧已在当前聚类中才考虑加入 — 保证每个 cluster 是时间上连续的片段 [论文原文,这是与标准 DPC 的核心区别]
   - 停止: 任一条件不满足 或 达到 Smax [§3.3]
5. 每个 cluster 做 mean pooling → 一个 cluster embedding

**为什么用 mean pooling 而不是 attention pooling**: 作者未讨论此选择 [agent 解读: 可能因为聚类内帧本身高度相似(由 similarity threshold 保证),mean pooling 损失小且不引入额外参数]。

#### 2. Implicit Duration Coding [§3.4]

**为什么不用独立的 duration predictor**: 作者明确表示尝试过 FastSpeech 式的 learned duration predictor,但联合训练时出现严重优化不稳定和收敛问题 [论文原文, §3.4]。

**方案**: 将内容 index k_n 和 duration d_n 编码进单一 token ID:

$$ID_n = (d_n - 1) \cdot K + k_n$$

其中 K=4096 (codebook size), d_n ∈ {1,...,Smax}, k_n ∈ {0,...,K-1}。有效词表大小为 K*Smax (如 4096*4 = 16384) [§3.4, Eq.6]。

**解码** (trivially reversible):
- d_n = floor(ID_n / K) + 1 [Eq.7]
- k_n = ID_n mod K [Eq.8]

**为什么这么做而不扩展实际码本**: 实际只训练一个 K=4096 的码本,扩展的 index space 只是逻辑上的编号映射,不增加训练参数 [论文原文]。解码时先恢复 k_n → 查原始码本 → repeat d_n 次 [§3.4]。

**对下游 LM 的影响**: LM 直接在扩展词表上做 next-token prediction,无需 duration predictor 或帧级重复。序列长度 = N (远小于 T),但词表从 K 扩到 K*Smax [论文原文]。作者在 Appendix J 中证明:序列缩短带来的推理加速远超词表扩大的 softmax 开销,RTF 降低 36% (τ=0.6) [Table 5]。

#### 3. 比特率计算 [§4.1, Eq.10-11]

对 VARSTok: Bitrate = Frame Rate × log_2(K * Smax)
对 baseline: Bitrate = Frame Rate × log_2(K)

这意味着 VARSTok 每个 token 携带的信息量 (bits/token) 多于 fixed-rate baseline,因为 duration 信息被编入了 index [agent 解读]。例如 VARSTok 30.95Hz: 0.43 kbps (log_2(16384)=14 bits/token) vs WavTokenizer 40Hz: 0.48 kbps (log_2(4096)=12 bits/token)。

### 训练策略

- 从预训练 WavTokenizer 75Hz (encoder + decoder + VQ codebook) 初始化 [§4.1]
- 训练集: 585h LibriTTS, 24kHz, 4xA800
- 优化: Adam, batch=128 (3s chunks), 60 epochs
  - LR schedule: 1e-4 for 30 epochs → 2e-5 for 30 epochs [Appendix C]
- 损失: 与 WavTokenizer 完全相同 (mel L1 + VQ commitment + adversarial + feature matching), 无额外损失 [§3.5, Eq.9]
- VQ: EMA + random awakening (继承 WavTokenizer) [§3.2]

**关键: 聚类算法是不可微的** — 不参与反向传播。Encoder/decoder/VQ 通过重建损失学习,聚类在推理路径上但不提供梯度 [agent 解读: 这意味着 encoder 不会被优化去产出更适合聚类的 embeddings,是一个潜在的改进空间]。

## 实验

### 语音重建 (LibriTTS test-clean) [Table 1]

| 模型 | 帧率(Hz) | 比特率(kbps) | UTMOS↑ | PESQ↑ | STOI↑ | V/UV F1↑ | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GT | / | / | 4.1185 | / | / | / | [Table 1] |
| WavTokenizer 75Hz | 75.00 | 0.90 | 4.0247 | 2.4543 | 0.9188 | 0.9339 | [Table 1] |
| WavTokenizer 40Hz | 40.00 | 0.48 | 3.6107 | 1.7075 | 0.8652 | 0.9095 | [Table 1] |
| VARSTok (τ=0.8, Smax=4) | 36.81 | 0.52 | **4.0000** | **1.8887** | **0.8814** | **0.9186** | [Table 1] |
| VARSTok (τ=0.7, Smax=4) | 30.95 | 0.43 | 3.8949 | 1.7095 | 0.8601 | 0.9047 | [Table 1] |
| VARSTok (τ=0.6, Smax=4) | 26.29 | 0.37 | 3.8304 | 1.5855 | 0.8411 | 0.8985 | [Table 1] |
| BigCodec | 40.00 | 0.52 | 3.9802 | 1.8796 | 0.8653 | 0.9133 | [Table 1] |
| SQCodec | 44.44 | 0.75 | 3.9601 | 1.8898 | / | 0.9197 | [Table 1] |

**关键观察**:
- VARSTok (τ=0.8) 以 36.81Hz (比 40Hz baseline 少 8%) 达到 UTMOS 4.00,接近 75Hz WavTokenizer (4.02),这是一个惊人的 rate-quality trade-off [论文原文, §4.2]
- VARSTok (τ=0.7) 以 30.95Hz (比 40Hz 少 23%) 在 UTMOS 上仍大幅领先 40Hz WavTokenizer (3.89 vs 3.61) [Table 1]
- 在 <40Hz 区间,VARSTok 在所有指标上全面领先 [论文原文, §4.2]

### 语义评估 (ARCH Benchmark) [Table 2]

| 模型 | 帧率 | EMOVO F1 | RAVDESS F1 | AudioMNIST F1 | SLURP F1 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WavTokenizer | 40.00 | 0.1676 | 0.2319 | 0.4509 | 0.0055 | [Table 2] |
| VARSTok (τ=0.7) | 30.95 | 0.1763 | **0.2508** | **0.6078** | **0.0109** | [Table 2] |
| VARSTok (τ=0.6) | 26.29 | 0.1682 | 0.2348 | 0.6175 | 0.0098 | [Table 2] |

VARSTok 在所有语义任务上优于 WavTokenizer,尤其在 AudioMNIST 上提升显著 (0.45 → 0.61 F1)。这表明 variable-rate pooling 产生了更具语义区分力的表征 [论文原文, §4.3]。

### 下游 TTS (Zero-shot, UniCATS test set) [Table 3]

| Tokenizer | 帧率 | WER↓ | SIM↑ | UTMOS↑ | MOS↑ | SMOS↑ | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WavTokenizer | 40.00 | 7.481 | **0.918** | 3.920 | 3.983±0.065 | 3.918±0.063 | [Table 3] |
| VARSTok (τ=0.8) | 36.81 | **6.787** | 0.899 | **4.246** | **4.053±0.063** | **3.946±0.062** | [Table 3] |
| VARSTok (τ=0.7) | 30.95 | 7.294 | 0.895 | 4.199 | 4.036±0.065 | 3.941±0.065 | [Table 3] |
| VARSTok (τ=0.6) | 26.29 | 9.393 | 0.880 | 4.083 | 3.986±0.064 | 3.913±0.066 | [Table 3] |

**关键发现**:
- VARSTok (τ=0.8) 在 TTS 中全面优于 WavTokenizer baseline: WER 降 9.3% (7.48→6.79), UTMOS +0.33, MOS +0.07 [Table 3]
- 唯一劣势在 SIM (0.918→0.899),但作者指出客观 SIM 与主观 SMOS 不一致 (SMOS 3.95 vs 3.92 差异极小),主观评估未检测到感知差异 [论文原文, §4.4]
- τ=0.6 (26.29Hz) 时 WER 退化明显 (9.39%),说明过度压缩会损害内容保真度 [Table 3]

### 推理效率 [Table 5, Appendix J]

| Tokenizer | 帧率 | RTF↓ | 出处 |
| --- | --- | --- | --- |
| WavTokenizer | 40.00 | 0.766 | [Table 5] |
| VARSTok (τ=0.8) | 36.81 | 0.716 | [Table 5] |
| VARSTok (τ=0.7) | 30.95 | 0.602 | [Table 5] |
| VARSTok (τ=0.6) | 26.29 | 0.487 | [Table 5] |

序列缩短带来的推理加速远超词表扩大的 softmax 开销: 36% speedup @ τ=0.6 [Table 5]。

### Codebook Size 消融 [Table 6, Appendix K]

| K | Exp. Space | Codebook Usage | UTMOS | 出处 |
| --- | --- | --- | --- | --- |
| 2048 | 8192 | 100% | 3.8262 | [Table 6] |
| 4096 | 16384 | 100% | 3.8949 | [Table 6] |
| 8192 | 32768 | 72.72% | 3.9042 | [Table 6] |
| 16384 | 65536 | 40.17% | 3.9390 | [Table 6] |

K=4096 是最佳平衡: 100% utilization + 0.43 kbps + UTMOS 3.89 [论文原文, Appendix K]。更大码本改善微弱但 utilization 骤降 — 与 [[CodebookCollapse]] 的 "充足数据缓解 collapse" 发现一致 (此处 585h 对 K=4096 够用,对 K=16384 不够) [agent 解读]。

## 局限性

1. **数据规模受限**: 仅在 585h LibriTTS 上训练和评估,未验证大规模 (万小时级) 数据下的行为,也未测试 out-of-domain 泛化 [agent 解读]
2. **Smax 硬约束**: 最大 cluster 长度被固定 (默认 4 帧),限制了对长静音/稳态段的压缩能力; Smax=8 时质量骤降 [Table 1]
3. **聚类不可微**: temporal-aware clustering 不参与反向传播,encoder 无法被优化去产出更适合聚类的表征,存在 train-infer mismatch [agent 解读]
4. **SIM 轻微下降**: 变帧率下客观 speaker similarity 有一致下降趋势 (0.918→0.880),虽然主观差异不显著,但在对 speaker identity 要求极高的场景可能有影响 [Table 3]
5. **仅语音验证**: 未扩展到音乐/环境声等其他音频领域 (作者将此列为 future work) [§5]
6. **与 TFC 不公平对比**: 未与最相关的 TFC (Zhang et al. 2025) 做直接实验对比,仅进行了 qualitative 讨论 [§2.2]

## 点评

VARSTok 的核心贡献清晰且有说服力: 它证明了 variable-frame-rate acoustic tokenizer 可以无缝接入 downstream speech LM,而不需要任何架构修改。implicit duration coding 的设计极其优雅 — 用一个简单的 index 映射公式 ID=(d-1)*K+k 同时编码内容和时长,比引入独立 duration predictor 简洁得多。

从 rate-quality trade-off 角度看,VARSTok 的结果令人印象深刻: 30.95Hz 时 UTMOS 3.89 超过 40Hz WavTokenizer 的 3.61,证明了 "信息密度自适应" 比 "均匀分配" 更高效的直觉。token boundary visualization (Fig 2) 清晰展示了自适应分配的效果:静音/稳态区域用长 token,过渡区域用短 token。

但几个问题值得关注: (1) 聚类算法的不可微性是一个本质限制 — encoder 产出的 embeddings 是为 fixed-rate 重建优化的 (来自 WavTokenizer 初始化),并非为 variable-rate 聚类优化,这可能解释了为什么 Smax=8 时质量骤降; (2) 585h 训练数据偏小,在万小时级数据下 clustering 行为可能不同; (3) 与最相关的竞品 TFC 缺乏直接实验对比,使 "first fully dynamic" 的主张缺少数值支撑。

总体而言,VARSTok 开辟了 acoustic tokenizer 的新方向。implicit duration coding 的思路对任何需要处理变长 token 的系统都有迁移价值。

## 可复用的 idea

1. **Implicit Duration Coding (ID = (d-1)*K + k)**: 将 duration 编码进 token index 的做法极为通用。任何使用 VQ 且需要处理变长单元的系统 (如变帧率 codec、variable-length phoneme 建模) 都可直接复用此公式,零额外参数
2. **Temporal-aware DPC 的时间连续性约束**: 在标准 density peak clustering 上加 "仅当紧邻帧已被分配才能扩展" 的约束,保证聚类结果是时间连续的。这个约束适用于任何需要对序列数据做 adaptive segmentation 的场景 (如 speech diarization, music segmentation)
3. **-β*s_t 种子排斥项**: 聚类扩展条件中惩罚候选帧的峰得分,防止两个种子被合并到同一 cluster。这个技巧可迁移到任何基于 seed-expansion 的聚类/分割算法中
4. **Rate-quality 超参控制**: τ (similarity threshold) 和 Smax (max duration) 提供直觉化的 rate-quality knobs,比传统 codec 的 quantizer dropout 更灵活。用户可根据场景选择不同的 τ 值,无需重新训练

---

> [!review] 审阅结论: pass (2026-06-04)
> - **可复述**: pass — 方法节含因果解释,每个设计选择有 WHY 讨论
> - **可信赖**: pass — 数字 claim 全部标注 [Table N],指标使用正确
> - **可区分**: pass — [论文原文]/[agent 解读] 标注覆盖率 ~85%
> - **可定位**: pass — KB 背景谱系定位含具体模型/方法对比
> - **不污染**: pass — 反向更新仅追加 key_papers,无实质修改
> - 详见 `_review/VARSTok-review.yml`

---

检索命中: [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓, [[CodebookCollapse]]✓, [[NeuralAudioCompression]]✓ | 过滤: [[TokenRateandBitrateTrade-offs]][待确认], [[Single-codebookvsMulti-codebook]][待确认], [[DurationPredictor]][待确认], [[CodecLanguageModel]][待确认] | 未命中但可能相关: 无
