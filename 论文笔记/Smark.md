---
type: paper
tier: deep
title: "Smark: A Watermark for Text-to-Speech Diffusion Models via Discrete Wavelet Transform"
arxiv_id: "2512.18791"
source: "Sources/Smark.pdf"
authors: [Yichuan Zhang, Chengxin Li, Yujie Gu]
year: 2025
venue: "arXiv preprint"
tags: [TTS, watermarking, diffusion, DWT, audio-security, model-agnostic, mel-spectrogram, copyright-protection]
concepts: ["[[Diffusion-basedTTS]]", "[[MelSpectrogram]]", "[[Anti-spoofingandDeepfakeDetection]]", "[[TTSEvaluation]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 0 个已确认实体页 + 4 个待确认实体页: [[Diffusion-basedTTS]][待确认], [[MelSpectrogram]][待确认], [[Anti-spoofingandDeepfakeDetection]][待确认], [[TTSEvaluation]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: 无 confirmed 页命中 | 参考: [[Diffusion-basedTTS]](pending-review), [[MelSpectrogram]](pending-review), [[Anti-spoofingandDeepfakeDetection]](pending-review), [[TTSEvaluation]](pending-review) | 未命中但可能相关: Audio Watermarking(概念库中无独立页), Discrete Wavelet Transform(概念库中无独立页)

**谱系定位**: Smark 处于 Diffusion-based TTS 与 Anti-spoofing/Deepfake Detection 的交叉地带。Diffusion-based TTS 页记录了 GradTTS (Popov et al., 2021) 是 SDE-based diffusion TTS 的代表,所有 diffusion TTS 共享 reverse diffusion 过程。Smark 正是利用这一共享机制,在 reverse diffusion 的中间 timestep 嵌入水印,实现跨模型通用性。Anti-spoofing 页提到 watermarking 作为 proactive traceability 手段,但现有工作 (如 [[论文笔记/TraceableSpeech|TraceableSpeech]]) 依赖 codec LM 架构,不适用于 diffusion TTS;GROOT 修改 initial mel 但忽略文本-语义对齐。Smark 填补了 diffusion TTS 通用水印方案的空白。

**已有认知**: Mel Spectrogram 页记录了 mel 是二维时频表示,diffusion TTS 以其为中间声学特征。TTS Evaluation 页的 Security & Traceability 节指出 imperceptible audio watermarking 是新兴方向。Diffusion-based TTS 页记录了 GradTTS、WaveGrad、PriorGrad 等关键模型。

**创新判断**: 与 TraceableSpeech (codec LM + 水印联合训练) 不同,Smark 的创新在于: (1) 操作在所有 diffusion TTS 共享的 reverse diffusion 过程中,而非特定模型架构; (2) 利用 DWT 分解 mel spectrogram,仅在低频 LL 子带嵌入水印,既保证不可感知性又利用 LL 子带的去噪鲁棒性; (3) 水印嵌入在单一 timestep 完成,计算开销极低。

> [!summary] 速查
> - **一句话**: 利用 DWT 在 diffusion TTS 共享的 reverse diffusion 过程中将水印嵌入 mel spectrogram 的低频 LL 子带,实现模型无关的通用语音水印
> - **路线**: Text → Text Encoder → Reverse Diffusion (X'_T → X'_t) → DWT 分解 X'_t 为 LL/LH/HL/HH → Watermark Embedder 修改 LL 子带 → IDWT 重建 → 继续 Reverse Diffusion → X'_0 (带水印语音) → Watermark Extractor (Conv2D + Pooling) → 提取水印 m'
> - **指标**: GradTTS+LJSpeech: PESQ 4.5467 / STOI 1.0 / MOS 3.1134 / ACC 1.0 [Table II]; 攻击鲁棒性: GradTTS 所有单/双重攻击下 ACC ≥ 0.97 [Table III]; 容量 50-5000bp 下 ACC ≥ 0.990 [Fig 4]
> - **可借鉴**: (1) DWT 低频子带嵌入策略 — 在扩散过程中间状态选择结构稳定区域做信息注入; (2) 强度预测器动态调整嵌入强度 — 根据局部信号特性自适应平衡不可感知性与鲁棒性; (3) 课程式多目标训练 — 先稳定主模型再逐步增大辅助目标权重
> - **局限**: 需要接入 reverse diffusion 过程 (需修改 TTS 推理代码); 未在商业 TTS 系统上验证; 实验仅覆盖英文; 未考虑 re-synthesis 攻击; 假设检验方法对极短语音 (<0.3s) 的有效性未讨论

## 核心问题

TTS diffusion 模型生成高质量语音,但带来 deepfake 和版权侵权风险 [§I]。音频水印是反制手段,但面临三重矛盾:

1. **后处理水印 vs 音质保留** — WavMark、AudioSeal 等后处理方法在生成后修改波形,二次修改破坏原始音频感知质量,且对常见音频处理攻击 (压缩、加噪) 鲁棒性有限 [§I, Table I]
2. **生成式水印 vs 模型通用性** — GROOT 修改 diffusion 初始 mel 条件实现 generative watermarking,但不考虑文本-语义对齐,导致语音质量下降且与特定模型绑定 [§I, Table I]; TraceableSpeech 依赖 VALL-E codec 架构,无法用于 diffusion TTS [§II.B]
3. **不可感知性 vs 鲁棒性 vs 通用性的三角困境** — 现有方法至多满足其中两项,无法同时达成三者 [§I]

Smark 的核心洞察: 所有 diffusion TTS 模型共享 reverse diffusion 过程,该过程的数学基础 (预测均值和方差的条件概率模型) 相同 [§III.A]。因此,在 reverse diffusion 中嵌入水印天然具有模型无关性。而 DWT 将 mel spectrogram 分解为低频 (LL) 和高频 (LH/HL/HH) 子带,LL 子带携带核心感知信息且对去噪过程鲁棒,是理想的嵌入位置 [§III.B] [论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构 [§IV.A, Fig 1]

Smark 在 TTS diffusion 模型的 reverse diffusion 过程中嵌入水印,包含三个组件:

1. **DWT/IDWT 分解与重建**: 在 reverse diffusion 的特定 timestep t,对中间 mel spectrogram X'_t 做 DWT 分解为 LL/LH/HL/HH 四个子带 [§IV.A]
2. **Watermark Embedder**: 将二进制水印 m 嵌入 LL 子带,输出修改后的 LL 子带 [§IV.B]
3. **Watermark Extractor**: 从最终生成的带水印语音中提取水印 m' [§IV.B]

流程 [Fig 1]:
- 推理: Text → Text Encoder → Reverse Diffusion 从 X'_T 开始 → 到达 timestep t 时,X'_t 经 DWT → Embedder 修改 LL → IDWT 重建 X̃'_t → 继续 reverse diffusion 到 X̃'_0 → 输出带水印语音
- 验证: 带水印语音 → Extractor → 提取 m' → 与原始 m 对比

### 关键设计选择

**1. DWT 低频 LL 子带嵌入 (核心创新)** [§III.B, §IV.A]

WHY 选择 DWT 而非直接在 mel 上嵌入: DWT 的多分辨率分析将信号分解为低频近似 (LL) 和高频细节 (LH/HL/HH)。LL 子带承载核心结构和感知信息 (基频轮廓、共振峰),对噪声和压缩具有天然抗干扰性;高频子带主要包含细节特征,在去噪过程中易被平滑掉 [§III.B] [论文原文]。

WHY 选择 LL 而非其他子带: 实验验证 [Table V] 显示 LL 子带在所有模型×数据集组合上均显著优于 LH/HL/HH — 例如 GradTTS+LJSpeech 上 LL 的 ACC=1.0 vs LH 的 0.5219、HL 的 0.4496、HH 的 0.8545 [Table V]。这是因为: (1) LL 子带分辨率降低,水印嵌入被限制在可控区域,避免影响听觉敏感的高频区 [论文原文]; (2) LL 子带在迭代去噪中更稳定,水印不会被后续 diffusion steps 冲刷掉 [论文原文]。

[agent 解读] LL 子带嵌入本质上是一种频率域信息隐藏: 人耳对低频结构变化不敏感 (在动态范围内),但信号处理攻击 (如重采样、压缩) 保留低频能量,因此 LL 嵌入天然同时满足不可感知性和鲁棒性。

**2. 自适应强度水印嵌入器** [§IV.B, Fig 2, Eq before §IV.B.2]

嵌入公式:
$$\tilde{X}'_t = X'_t + \alpha \cdot (2 \cdot Emb(\mathbf{m}) - 1), \quad \alpha \in \mathbb{R}$$

其中 Emb(m) 是水印的潜表示,α 是动态嵌入强度系数 [§IV.B]。

嵌入器包含三个子模块 [§IV.B, Fig 2]:
- **Encoding Layer**: 二进制水印 m → 全连接 + ReLU → latent vector [§IV.B]
- **Feature Enhancement Network**: 扩展后的水印通过膨胀卷积 (dilated convolution) 捕获多尺度时频上下文信息,输出单通道水印图 [§IV.B]
- **Intensity Predictor**: 根据语音信号的局部特性动态生成 α,在感知掩蔽区域嵌入更强,敏感区域嵌入更弱 [§IV.B]

WHY 使用自适应强度: 固定强度水印无法兼顾不同区域的感知特性 — 语音的有声段 (voiced) 可容忍更强嵌入,而静音段或瞬态段需要更弱嵌入。自适应策略最小化可听伪影,同时维持鲁棒的水印提取 [§IV.B] [论文原文]。

**3. 轻量水印提取器** [§IV.B, Fig 2]

提取器为简单 2D CNN: DWT(带水印 Mel) → Conv1D → Conv2D → 3 × ConvBlock (Conv2D + Conv2D) → Pooling → Linear → ReLU → Sigmoid → BatchNorm → 预测 m' [§IV.B, Fig 2]。

[agent 解读] 提取器输入是 DWT 后的 mel (而非原始 mel 或波形),说明提取也在 DWT 域进行,与嵌入域保持一致,避免了域间不匹配引起的精度损失。

**4. 嵌入 Timestep 选择** [§VI.C, Fig 6]

Smark 仅在 reverse diffusion 的单一 timestep t 嵌入水印,实验测试了 t=0 到 t=50 [§VI.C]。

关键发现 [Fig 6]:
- 音质指标: 从 t=0 开始逐步上升,在 t=40-45 附近达到峰值,t=50 略有下降 [Fig 6]
- ACC: 从 t=0 开始稳步上升,在 t=35-40 达到高水平 [Fig 6]
- 所有实验中统一选择 t=45 [§VI.C]

WHY 中间偏后的 timestep 最优: 太早 (t≈0) 时 mel 已接近最终输出,嵌入直接影响输出质量;太晚 (t≈50) 时噪声水平高,嵌入信号被噪声淹没。中间偏后的 timestep 兼顾了 mel 的结构稳定性 (已有清晰的声学结构) 和后续去噪步骤对嵌入微扰的正则化效应 [§VI.C] [论文原文]。

[agent 解读] 这一发现暗示了 diffusion reverse process 中存在一个"信息注入甜区" — 信号已足够结构化可以承载水印,但后续去噪仍能平滑嵌入引入的高频伪影。

### 训练策略 [§IV.C]

**三项联合优化**:

1. **Embedder Loss** L_emb [§IV.C.1]: LL 子带的 MSE 重建损失
$$\mathcal{L}_{emb}(\theta) = \frac{1}{H \times W} \sum_{h=1}^{H} \sum_{w=1}^{W} \left( X_t^{LL}[h,w] - \tilde{X}_t^{LL}[h,w;\theta] \right)^2$$

2. **Extractor Loss** L_ext [§IV.C.2]: 水印的二进制交叉熵 BCE(m, m'(ζ))

3. **Total Loss** [§IV.C.3]:
$$L_{total} = \lambda_{tts} \mathcal{L}_{tts} + \lambda_{emb} \mathcal{L}_{emb} + \lambda_{ext} \mathcal{L}_{ext}$$

超参: λ_tts=1, λ_emb=2, λ_ext=10 [§V.A.4]

**课程式训练策略** [§IV.C.3]: 初始阶段 λ_tts 权重高,保证 TTS 模型稳定收敛;随训练推进,λ_emb 和 λ_ext 权重逐步增大,精化水印嵌入和提取 [§IV.C.3]。α 也采用退火策略: 初始值大以建立鲁棒水印模式,逐步减小以最小化感知影响 [§IV.C.3] [论文原文]。

### 假设检验验证 [§V.F, Fig 3]

水印验证采用基于二项分布的假设检验 [§V.F]:
- H₀: ξ = 0.5 (随机猜测)
- H₁: ξ > 0.5 (成功嵌入)

实验显示 [Fig 3]:
- 1000 个无水印样本: bit-wise 精度 ξ̂ ≈ 0.5331 (接近随机)
- 1000 个带水印样本: ξ̂ ≈ 0.9983
- 验证阈值 τ ∈ [0.62, 0.97] 可同时控制 FPR ≤ 0.0017 和 FNR ≤ 0.01 [§V.F]

## 实验

### 实验设置 [§V.A]
- **数据集**: LJSpeech (22.05kHz, 单说话人), LibriTTS (24kHz, 多说话人), LibriSpeech (16kHz, 多说话人) [§V.A.1]
- **TTS 模型**: GradTTS, WaveGrad, PriorGrad — 三种架构不同的 diffusion TTS [§V.A.4]
- **Baseline**: AudioSeal (后处理), GROOT (generative), DeAR (后处理), WavMark (后处理), TimbreWM (后处理) [§V.A.2]
- **指标**: PESQ, STOI, MOS (MOSNet), ACC [§V.A.3]
- **水印容量**: 100bp (默认), 扩展测试 50-5000bp [§V.A.4]
- **硬件**: Intel i7-11700K, NVIDIA RTX 4090 [§V.A.4]
- **训练**: Adam, lr=1e-4, λ_tts=1, λ_emb=2, λ_ext=10 [§V.A.4]

### 无攻击保真度 [Table II]

| 数据集 | 方法 (Base Model) | PESQ↑ | STOI↑ | MOS↑ | ACC↑ | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| LJSpeech | AudioSeal (PriorGrad) | 1.7793 | 0.8931 | 1.5762 | 0.7983 | [Table II] |
| LJSpeech | **Smark (PriorGrad)** | **2.0856** | **0.9705** | **2.1541** | **0.9863** | [Table II] |
| LJSpeech | GROOT (WaveGrad) | 3.4325 | 0.9731 | 1.9635 | 1.0 | [Table II] |
| LJSpeech | DeAR (WaveGrad) | 3.1360 | 0.7508 | 1.8336 | 1.0 | [Table II] |
| LJSpeech | **Smark (WaveGrad)** | **4.5462** | **0.9957** | **2.0924** | **1.0** | [Table II] |
| LJSpeech | WavMark (GradTTS) | 3.4325 | 0.9692 | 2.9868 | 0.9998 | [Table II] |
| LJSpeech | TimbreWM (GradTTS) | 2.7032 | 0.7357 | 1.9916 | 0.9999 | [Table II] |
| LJSpeech | **Smark (GradTTS)** | **4.5467** | **1.0** | **3.1134** | **1.0** | [Table II] |

Smark 在所有模型×数据集组合上均取得最高或接近最高的 PESQ 和 STOI,同时保持近完美 ACC [Table II]。

### 攻击鲁棒性 [Table III]

LJSpeech 上 7 种单攻击 + 复合攻击:

| 方法 (模型) | CLP | Noise-W35 | SS-01 | AS-90 | EA-0315 | LP5000 | CLP+Noise | SS+AS | EA+LP | CLP+AS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Smark (PriorGrad) PESQ↑ | 2.0821 | 2.0537 | 2.1245 | 1.9763 | 2.0912 | 2.1148 | 2.0576 | 2.1302 | 1.9935 | 2.0789 |
| Smark (PriorGrad) ACC↑ | **0.9801** | **0.9775** | **0.9762** | **0.9830** | **0.9751** | **0.9805** | **0.9744** | **0.9792** | **0.9821** | — |
| Smark (WaveGrad) PESQ↑ | **3.5599** | **3.8957** | 3.6636 | **3.8886** | **3.7843** | **3.8791** | **3.6251** | **3.7706** | **3.8674** | **3.7224** |
| Smark (WaveGrad) ACC↑ | 0.9990 | 0.9945 | 0.9895 | — | 0.9918 | 0.9890 | 0.9958 | 0.9926 | 0.9894 | 0.9938 |
| Smark (GradTTS) PESQ↑ | **4.5475** | **4.5444** | **4.5319** | **4.5475** | **4.5475** | **4.5472** | **4.5474** | **4.5486** | **4.5466** | **4.5472** |
| Smark (GradTTS) ACC↑ | **1.0** | **0.9999** | **0.9998** | **1.0** | **1.0** | **1.0** | **1.0** | **0.9999** | **0.9997** | **0.9999** |

[Table III] 出处。Smark 在 GradTTS 上即使面对复合攻击,ACC 仍保持 ≥ 0.9997,PESQ 几乎不下降。

### 容量扩展 [§V.C.1, Fig 4]

50bp 到 5000bp,ACC 始终 ≥ 0.990 (GradTTS)。PESQ 和 MOS 在容量增加 100 倍后仍保持稳定,无明显下降趋势 [Fig 4]。

### 复合攻击+变容量 [§V.C.3, Fig 5]

在 6 种复合攻击 × 50-1000bp 容量组合下:
- GradTTS: 所有条件下 ACC 近乎完美,音质指标几乎不受影响 [Fig 5 第一行]
- WaveGrad/PriorGrad: 高容量 + 强攻击下 ACC 从约 1.0 降至约 0.95-0.97,伴随中度音质下降 [Fig 5 第二三行]

### Ablation: w/ vs w/o DWT [Table IV]

| 数据集 | 模型 | 方法 | PESQ↑ | STOI↑ | MOS↑ | ACC↑ |
| --- | --- | --- | --- | --- | --- | --- |
| LJSpeech | GradTTS | w/ DWT | **4.5467** | **1.0** | **3.1134** | **1.0** |
| LJSpeech | GradTTS | w/o DWT | 3.0458 | 0.7653 | 1.6821 | 0.5436 |
| LJSpeech | WaveGrad | w/ DWT | **4.5462** | **0.9957** | **2.0924** | **1.0** |
| LJSpeech | WaveGrad | w/o DWT | 3.2372 | 0.6806 | 1.5612 | 0.6735 |

[Table IV] 出处。移除 DWT 后所有指标大幅下降 — PESQ 降幅约 1.3-1.5,ACC 降至 0.5-0.67,接近随机水平,证明 DWT 是 Smark 成功的必要条件。

## 局限性

1. **需要侵入式修改推理流程** — 水印嵌入必须在 reverse diffusion 过程中执行,要求接入模型的推理管线。对于闭源或 API-only 的 TTS 系统无法使用 [agent 解读]
2. **仅验证三种 diffusion TTS 模型** — GradTTS、WaveGrad、PriorGrad 是相对早期的模型,未在近期主流模型 (如 Matcha-TTS、NaturalSpeech 系列) 上验证 [agent 解读]
3. **未考虑 re-synthesis 攻击** — 将带水印语音经另一 TTS 或 voice conversion 系统重新合成可能抹除水印,论文未讨论此类攻击 [agent 解读]
4. **假设检验依赖充足的信号长度** — 二项分布假设检验需要足够多的 bits (N=100) 才有统计效力,对极短语音的适用性未讨论 [agent 解读]
5. **仅英文数据集** — LJSpeech (单说话人)、LibriTTS/LibriSpeech (多说话人) 均为英文,未验证跨语言泛化 [§V.A.1]
6. **MOSNet 替代人工 MOS** — 音质评估使用 MOSNet 预测分数而非人工评分,MOSNet 在 TTS 水印场景下的准确性存疑 [agent 解读]
7. **与 flow matching TTS 的兼容性** — 论文明确聚焦 diffusion model,但当前 TTS 主流已转向 flow matching (Voicebox, F5-TTS, CosyVoice),Smark 能否直接适用于 flow matching 的 ODE 求解过程未讨论 [agent 解读]

## 点评

Smark 提出了一个极为清晰的设计直觉: diffusion TTS 共享 reverse diffusion 过程 → 在此过程中嵌入水印天然具有通用性;DWT 分解 mel spectrogram → LL 子带结构稳定 → 是理想的嵌入位置。这种层层递进的设计逻辑使得整个方法高度可解释。

实验设计是本文的强项。三个不同架构的 diffusion TTS 模型 (GradTTS、WaveGrad、PriorGrad) × 三个数据集 (LJSpeech、LibriTTS、LibriSpeech) 构成了 9 种组合,加上 5 种基线方法、7 种单攻击、6 种复合攻击、50-5000bp 容量扫描,实验覆盖面极广。特别是 GradTTS 上的结果近乎完美: 所有攻击下 ACC ≥ 0.9997,PESQ 几乎不下降 [Table III],说明 DWT LL 子带与 GradTTS 的去噪过程高度兼容。

Ablation 结果 (Table IV) 是本文最有说服力的证据: 移除 DWT 后 ACC 从 1.0 降至 0.54 (GradTTS),证明不是"随便往 mel 上加东西就行",而是 DWT 的频率分离提供了关键的结构保护。Table V 进一步确认只有 LL 子带有效,其他子带 ACC 均 < 0.86。

但本文也有明显的时代局限: 实验所用的三个 diffusion TTS 模型均来自 2020-2021 年,当前 TTS 领域已大规模转向 flow matching (Voicebox, F5-TTS, CosyVoice)。虽然论文声称"通用性",但未讨论 Smark 能否适配 flow matching 的确定性 ODE 过程。此外,使用 MOSNet 替代人工 MOS 评估在水印研究中可能引入偏差 — MOSNet 可能对水印引入的特定频率伪影不敏感。

从与 [[论文笔记/TraceableSpeech|TraceableSpeech]] 的对比看,两种方案各有优势: TraceableSpeech 在 codec LM (VALL-E) 架构中端到端联合训练水印,获得最高质量;Smark 作为轻量插件式方案,不修改模型训练过程,具有更好的部署灵活性。两者互补覆盖了 codec LM 和 diffusion 两大 TTS 范式。

## 可复用的 idea

1. **DWT 低频子带信息注入**: 在扩散/流匹配等迭代生成过程的中间状态,利用 DWT 分离频率,仅在结构稳定的低频区域注入辅助信息 (水印、条件信号、控制信号)。可推广到任何需要在生成过程中嵌入隐式信息的场景
2. **动态嵌入强度预测器**: 根据信号局部特性自适应调整信息嵌入强度,在感知不敏感区域做更强嵌入 — 可用于其他信息隐藏或条件注入任务
3. **课程式多目标训练**: 先稳定主任务 (TTS loss),再逐步增大辅助任务权重 (水印 loss),用退火策略调节嵌入强度 — 适用于多目标联合训练中主辅任务存在冲突的场景
4. **二项分布假设检验验证**: 将水印提取问题形式化为统计假设检验,提供严格的 FPR/FNR 控制 — 可作为任何 binary pattern 检测任务的可靠性验证框架

---

> [!review] 审阅结论: pending
> 待审阅

检索命中: 无 confirmed 页命中 | 参考: [[Diffusion-basedTTS]](pending-review), [[MelSpectrogram]](pending-review), [[Anti-spoofingandDeepfakeDetection]](pending-review), [[TTSEvaluation]](pending-review) | 未命中但可能相关: Audio Watermarking(概念库中无独立页), Discrete Wavelet Transform(概念库中无独立页)
