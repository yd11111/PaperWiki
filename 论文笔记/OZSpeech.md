---
type: paper
tier: deep
title: "OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching"
arxiv_id: "2505.12800"
source: "Sources/OZSpeech.pdf"
authors: [Hieu-Nghia Huynh-Nguyen, Ngoc Son Nguyen, Huynh Nguyen Dang, Thieu Vo, Truong-Son Hy, Van Nguyen]
year: 2025
venue: "arXiv preprint"
tags: [TTS, zero-shot, flow-matching, one-step-sampling, neural-codec, FACodec, non-autoregressive, efficiency]
concepts: ["[[Conditional Flow Matching]]", "[[Speech Factorization]]", "[[Duration Predictor]]", "[[Residual Vector Quantization]]", "[[Non-autoregressive TTS]]"]
models: ["[[模型库/NaturalSpeech 3|NaturalSpeech 3]]", "[[模型库/NaturalSpeech 2|NaturalSpeech 2]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[Conditional Flow Matching]], [[Zero-shot Speech Synthesis]], [[Speech Factorization]], [[Residual Vector Quantization]]; 2 个待确认实体页: [[模型库/NaturalSpeech 3|NaturalSpeech 3]] [待确认], [[Duration Predictor]] [待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: OZSpeech 属于 Zero-shot TTS 的 non-autoregressive/flow-based 阵营。其核心技术路线——使用 CFM 将中间分布映射到目标分布——与 CosyVoice 系列、F5-TTS、VoiceFlow 等系统同源,但 OZSpeech 的独特切入点在于: (1) 用 learned prior 替代 Gaussian noise 作为 flow matching 起点, 从而实现单步采样; (2) 直接在 FACodec 的离散 token 空间操作, 而非 mel spectrogram 空间。
>
> **已有认知 — Conditional Flow Matching**: CFM 通过学习 ODE 向量场将先验分布映射到目标分布, 是当前 TTS 领域 diffusion 的主流替代方案。典型系统 (F5-TTS 32 步, CosyVoice 2 10 步) 仍需多步 ODE 求解。VoiceFlow 通过 flow rectification 自蒸馏将步数降至 2 步,但仍非单步。OZSpeech 的单步 CFM 是对 CFM 采样效率的进一步探索。
>
> **已有认知 — Speech Factorization**: NaturalSpeech 3 提出 FACodec 将语音分解为 content/prosody/acoustic detail/timbre 四个子空间, OZSpeech 直接复用了 FACodec 作为其编解码器。FACodec 使用 Factorized VQ (FVQ) + GRL + timbre extractor 实现属性解耦。OZSpeech 在 FACodec 的 6 层量化器码本空间上直接建模, 这意味着模型继承了 FACodec 在属性解耦方面的优势和局限。
>
> **已有认知 — Duration Predictor**: OZSpeech 使用 FastSpeech 式 Duration Predictor 将音素对齐到 codec code 序列。已有知识表明 Duration Predictor 是 NAR TTS 的核心组件,其精度直接影响韵律自然度。近期工作 (DMOSpeech 2, FlexSpeech) 已通过 RL/DPO 优化 duration, 发现最优 duration 不等于真实 duration, OZSpeech 仍采用传统 MSE log-duration loss, 可能是其音质受限的原因之一。
>
> **创新判断**: OZSpeech 的核心创新在于将 OT-CFM 的起始分布从 Gaussian noise 替换为 learned prior, 并证明当 prior 足够接近 target 时可实现单步采样。这与 Consistency Models (蒸馏已训练的 diffusion model) 和 Shortcut Models (额外训练约束) 的加速路线不同, OZSpeech 不需要预训练 teacher 或额外约束, 仅一次联合训练即可。但其 UTMOS/SIM 指标不如基线, 说明单步采样在音质-速度 trade-off 上的代价仍然显著。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓, [[Speech Factorization]]✓, [[Residual Vector Quantization]]✓ | 过滤: [[模型库/NaturalSpeech 3|NaturalSpeech 3]](pending-review), [[Duration Predictor]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 learned prior 替代 Gaussian noise 作为 OT-CFM 起点, 实现 zero-shot TTS 的单步采样, WER 大幅领先基线但音质有所牺牲
> - **路线**: Text → Phoneme → Prior Codes Generator (6 seq, 层级依赖) → + Acoustic Prompt (FACodec codes, content masked) → OT-CFM Vector Field Estimator (1 step) → FACodec Decoder → Waveform
> - **指标**: WER 0.05 (vs F5-TTS 0.24/VALL-E 0.19, 3s prompt) [Table 1]; RTF 0.26 (vs F5-TTS 0.70, 3s) [Table 2]; NFE=1 (vs F5-TTS 32/NS2 200) [Table 2]; 模型 145M+102M (最小) [Table 2]; UTMOS 3.15 (vs VALL-E 3.68, 3s) [Table 1]
> - **可借鉴**: (1) learned prior + one-step CFM 联合训练策略——不需要 teacher model 或蒸馏阶段; (2) 量化器编码 (Quantizer Encoding) 让 Transformer 在单序列中同时建模 6 个量化器且不混淆; (3) 噪声容忍分析方法——系统评估不同 SNR 下 prompt 噪声对 TTS 的影响
> - **局限**: UTMOS/SIM-O 不如多步基线; Duration Predictor 使用传统取整方式导致时域失真; 仅在 LibriTTS 500h 上训练, 未验证 scaling; FACodec 冻结, codec 质量是瓶颈上限

## 核心问题

1. **多步 ODE 采样是 flow-based TTS 推理瓶颈**: F5-TTS 需要 32 步, NaturalSpeech 2 需要 200 步, 导致推理慢 [§1]
2. **Consistency Model / Shortcut Model 加速方案代价大**: CM 需完整 t∈[0,1] 训练轨迹 + 大量训练步数; Shortcut Model 需额外约束, 计算密集 [§1]
3. **传统 flow matching 在 mel/waveform 空间操作, 忽略语音的多属性结构**: 直接在 mel 空间做 flow matching 无法显式控制不同语音属性 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

OZSpeech 由三部分组成 [§3, Fig 1a]:
1. **Prior Codes Generator** $f_\psi$: 从音素序列生成 6 个量化器序列的先验码 (prior codes), 作为 flow matching 的起始分布
2. **Vector Field Estimator** $v_\theta$: 基于 Transformer 的向量场估计器, 以 prior codes + acoustic prompt 为输入, 单步预测 target codes
3. **FACodec** (冻结): NaturalSpeech 3 提出的属性分解 codec, 作为 waveform ↔ discrete codes 的编解码器

### 关键设计选择

#### 1. Learned Prior 替代 Gaussian Noise

**设计**: 传统 OT-CFM 从 $x_0 \sim \mathcal{N}(0, I)$ 出发, OZSpeech 从 learned prior $x_{pr} \sim p_{prior}$ 出发 [§3.1]

**WHY** [论文原文]: 当 prior 分布接近 target 分布时, flow 轨迹的距离和步数都大幅减少, 使得单步采样成为可能 [§3.3]。Prior 已经编码了有意义的语义信息(音素 → 内容码), 向量场只需学习"补全" prosody 和 acoustic detail 的差异 [agent 解读]。

**数学形式**: 将标准 CFM loss $\mathcal{L}_{CFM} = \mathbb{E}\|v_\theta(x_t, t) - \frac{x_1 - x_t}{1-t}\|^2$ 中的 $x_t$ 替换为 $x_{pr}$, $t$ 替换为隐式时间变量 $\tau$ [§3.3, Eq. 3]。$\tau$ 不再是均匀采样的, 而是由 prior 与 target 的距离隐式决定 [agent 解读]。

**与替代方案对比**: Consistency Model 需先训练完整 diffusion 再蒸馏 (两阶段); Shortcut Model 需同时学习 noise level 和 step size (额外约束); OZSpeech 一次联合训练搞定 [§1]。

#### 2. Prior Codes Generator 的层级结构

**设计**: 6 个量化器码序列按层级依赖顺序生成: $p(q_{1:6}|p;\psi) = p(q_1|p;f_{\psi_1})\prod_{j=2}^{6} p(q_j|q_{j-1};f_{\psi_j})$ [§3.2, Eq. 1]

**WHY** [论文原文]: 6 个码序列对应不同语音属性 (prosody/content/acoustic detail), 层级依赖让后续码序列可以利用前序码的信息 [§3.2]。

**架构细节**: 使用 FFT (Feed-Forward Transformer) blocks, encoder 2 层 + shared decoder 2 层 + 6 个专用层分别生成 6 个码序列 [Appendix C]。隐维度 256, 4 heads, FFN 1024 [Appendix C]。

**Duration Predictor**: 采用 FastSpeech 的 Duration Predictor, 将音素嵌入按预测时长复制后输入 decoder [§3.2]。训练使用 log-domain MSE loss [§3.2]。

#### 3. Folding + Quantizer Encoding

**设计**: 将 6 个量化器序列 (6×L×D) 折叠 (fold) 到单个序列 (L×D') 后送入 Transformer [§3.3]

**WHY** [论文原文]: 之前的工作 (VALL-E, NaturalSpeech 3) 对每个量化器独立建模, 计算成本高且串行处理慢。折叠后可以单次前向传播同时处理所有量化器 [§3.3]。

**Quantizer Encoding**: 为防止折叠后模型混淆不同量化器, 引入量化器编码 $Q(x) = x + \text{Dup}(\omega, L)$, 其中 $\omega \in \mathbb{R}^{6 \times 1 \times D}$ 是每个量化器的可学习标识符 [§3.3]。类比 Transformer 的 position encoding, 但作用对象是量化器维度 [agent 解读]。

#### 4. Acoustic Prompt 的使用方式

**设计**: 将 acoustic prompt 的 FACodec codes 与 prior codes 拼接, 但 content codes 被 mask 掉 [§3.3]

**WHY** [论文原文]: 只保留 prompt 的 prosody 和 acoustic detail, mask content 以防止不想要的内容迁移 [§3.3]。prompt 提供的是"怎么说" (音色/韵律), 而非"说什么" [agent 解读]。

#### 5. Anchor Loss 稳定训练

**设计**: 引入 Anchor Loss $\mathcal{L}_{anchor}$, 衡量估计的 target 分布与真实 target 之间的负对数似然 [§3.3, Eq. 4]

**WHY** [论文原文]: 防止 embedding collapse, 减少中间状态与 target 的距离, 稳定离散 token 数据上的 flow matching 训练 [§3.3]。来源于 Difformer (Gao et al., 2024) 的设计 [§3.3]。

### 训练策略

**联合训练**: Prior Codes Generator 和 Vector Field Estimator 端到端联合训练 [Appendix C]

**总损失**: $\mathcal{L}_{total} = \mathcal{L}_{prior} + \mathcal{L}_{dur} + \mathcal{L}_{CFM} + \mathcal{L}_{anchor}$ [§3.3]

**Prompt 构造技巧**: 训练时从 ground truth 中随机截取 1~3 秒片段作为 prompt (Arbitrary Segment strategy), 而非固定取开头 [Appendix B]。消融实验证明随机截取策略在所有指标上优于 First Segment [Table 3]。原因: 固定开头会导致模型过拟合于将 prompt 信息传到 target 开头位置 [§4.3]。

**训练配置**: 4x A100 80GB, batch size 16, AdamW lr=1e-4 [Appendix C]。仅使用 LibriTTS 500h 训练 [§4.1]。

## 实验

| 指标 | OZSpeech | F5-TTS | VoiceCraft | NS2 | VALL-E | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER (3s) ↓ | **0.05** | 0.24 | 0.18 | 0.09 | 0.19 | LibriSpeech test-clean | [Table 1] |
| WER (5s) ↓ | **0.05** | 0.32 | 0.19 | 0.09 | 0.19 | LibriSpeech test-clean | [Table 1] |
| UTMOS (3s) ↑ | 3.15 | **3.76** | 3.55 | 2.38 | **3.68** | LibriSpeech test-clean | [Table 1] |
| SIM-O (3s) ↑ | 0.40 | **0.53** | 0.51 | 0.31 | 0.40 | LibriSpeech test-clean | [Table 1] |
| SIM-R (3s) ↑ | 0.47 | - | 0.45 | 0.38 | **0.48** | LibriSpeech test-clean | [Table 1] |
| F0 Acc (3s) ↑ | **0.81** | 0.80 | 0.78 | 0.80 | 0.75 | LibriSpeech test-clean | [Table 1] |
| NFE ↓ | **1** | 32 | - | 200 | - | - | [Table 2] |
| RTF ↓ | **0.26** | 0.70 | 1.70 | 1.66 | 0.86 | A100 GPU | [Table 2] |
| #Params | **145M+102M** | 336M+13.5M | 830M+14M | 378M+14M | 594M+104M | - | [Table 2] |

**关键发现**:

1. **WER 大幅领先**: OZSpeech WER 0.05 vs 次优 NS2 0.09 (3s prompt), 相对降低 44% [Table 1]。作者认为 FACodec 的内容解耦 + learned prior 的内容码质量使得文本保真度极高 [agent 解读]
2. **推理效率显著**: NFE=1, RTF 0.26, 比次快的 F5-TTS (0.70) 快近 3 倍 [Table 2]
3. **模型最小**: 可训练参数仅 145M, 是 F5-TTS 的 43%, VALL-E 的 24% [Table 2]
4. **UTMOS 和 SIM-O 有差距**: UTMOS 3.15 vs VALL-E 3.68, SIM-O 0.40 vs F5-TTS 0.53 [Table 1]。作者归因于 FACodec 在声学表征和语义表征之间的平衡取舍 [§4.2]
5. **F5-TTS 在 500h 数据上完全失败**: 用 LibriTTS 500h 重训 F5-TTS, WER 超过 0.95, 说明传统 OT-CFM 方法强烈依赖大数据, 而 codec-based 方法受益于 codec 的预训练 [§4.2]

**噪声容忍分析** [§4.4, Table 4]:
- OZSpeech 的 WER 在 SNR 从 ∞ 降到 0dB 时几乎不变 (0.05→0.06), 而 VALL-E 从 0.19 飙升至 0.93 [Table 4]
- 微调后 (OZSpeech [♦]), 非 WER 指标在噪声条件下也显著改善, 如 SNR=0dB 时 UTMOS 从 1.72 提升到 2.58 [Table 4]
- 作者提出的噪声感知训练使 codec-based zero-shot TTS 系统具备隐式降噪能力 [§4.4]

**模型尺寸消融** [Appendix E, Table 5]:
- Small (100M) vs Base (145M): 参数减少 31%, 性能接近, 仅 F0 指标有波动 [Table 5]

## 局限性

1. **音质-速度 trade-off 仍然显著**: UTMOS 3.15 vs VALL-E 3.68, 单步采样在音质上付出了代价 [Table 1]。作者承认合成语音存在轻微失真 [Limitations]
2. **Duration Predictor 精度有限**: 使用传统 MFA + 取整方式获取 duration 标签, 连续 duration 取整为整数引入时域失真 [Limitations]。对比 DMOSpeech 2/FlexSpeech 的 RL/DPO 优化方案, OZSpeech 的 duration 建模较为原始 [agent 解读]
3. **FACodec 冻结 = 质量上限固定**: FACodec 本身的重建质量和解耦程度是系统的硬上限。SIM-R (0.47) 是 FACodec 重建后的相似度, 已接近其天花板 [agent 解读]
4. **仅 LibriTTS 500h 训练, 未验证 scaling**: F5-TTS 用 95K hours (Emilia) 训练, OZSpeech 仅 500h。论文未探索更大数据集下的性能变化 [Table 1]
5. **仅支持英语**: 未展示多语言或跨语言能力 [agent 解读]
6. **评估局限**: 仅使用 LibriSpeech test-clean, 未在 SEED-TTS-Eval 等更严格的 benchmark 上评测 [agent 解读]

## 点评

OZSpeech 提出了一个清晰且有效的加速思路: 与其在已训练的 diffusion model 上做蒸馏 (Consistency Model) 或加额外约束 (Shortcut Model), 不如直接学一个好的起始分布让单步即可到位。这个思路概念简洁, 且在 WER 和推理效率上取得了显著优势。

**优势**: (1) 单步采样 + 最小模型 + 最低 WER, 在实际部署场景下极具吸引力; (2) WER 对 prompt 长度和噪声鲁棒, 这在实际应用中很有价值; (3) 联合训练无需两阶段, 工程复杂度低。

**不足**: (1) UTMOS/SIM 的差距说明单步 flow matching 在感知质量上仍有代价, 对音质敏感的场景可能不够; (2) 与当前 SOTA (CosyVoice 3 WER 1.45% on SEED-TTS-Eval, Qwen3-TTS WER 1.24%) 不在同一评测体系下, 难以直接比较; (3) FACodec 冻结意味着音质上限被固定在 NaturalSpeech 3 时代的 codec 水平, 而 codec 技术在快速演进; (4) Duration Predictor 采用最朴素的方案, 是已知的瓶颈。

**与知识库中已有工作的关系**: OZSpeech 与 VoiceFlow (rectified flow, 2 步) 在"减少 CFM 采样步数"的目标上一致, 但方法路径不同。VoiceFlow 通过自蒸馏拉直 ODE 轨迹, OZSpeech 通过缩短起止距离。OZSpeech 的 Folding + Quantizer Encoding 方案与 NaturalSpeech 3 的逐量化器串行生成形成对比, 提供了一种更高效的多量化器并行建模思路。

## 可复用的 idea

1. **Learned prior + one-step flow matching**: 在任何需要加速 flow/diffusion 采样的场景中, 可以考虑训练一个 prior network 将起始分布拉近 target, 从而减少步数。不限于 TTS, 也适用于图像/视频生成。

2. **Quantizer Encoding**: 处理 multi-codebook discrete representation 时, 在折叠后加入可学习的量化器标识符, 让 Transformer 区分不同量化器。这是一个简单有效的 trick, 类比 segment embedding (BERT) 或 position encoding。

3. **噪声容忍评估协议**: 系统性评估不同 SNR (0/6/12dB) 下 TTS prompt 的鲁棒性, 并提出噪声感知微调方案。这种评估维度在其他 zero-shot TTS 工作中几乎没有被探索, 值得推广。

4. **Arbitrary Segment prompting**: 训练时从 ground truth 中随机位置截取 prompt (而非固定取开头), 可显著改善 zero-shot 泛化能力。消融实验验证了效果 [Table 3]。

> [!review] 审阅: pass (2026-06-03)
> 5 原则均满足, 3 low issues (datasets 空列表 / agent 推断已标注)。可反向更新。
> 详见 `_review/OZSpeech-review.yml`。
