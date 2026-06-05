---
type: paper
tier: deep
title: "FNH-TTS: Mixture-of-Experts Duration Modeling for Robust Neural Speech Synthesis"
arxiv_id: "2508.12001"
source: "Sources/FNH-TTS.pdf"
authors: [Qingliang Meng, Luogeng Xiong, Wei Liang, Limei Yu, Huizhi Liang, Tian Li]
year: 2026
venue: "arXiv preprint"
tags: [TTS, duration-modeling, MoE, vocoder, non-autoregressive, VITS, discriminator, prosody]
concepts: ["[[DurationPredictor]]", "[[Non-autoregressiveTTS]]", "[[ProsodyModeling]]", "[[NeuralVocoder]]", "[[SpeakerEmbedding]]", "[[Multi-scaleSTFTDiscriminator]]", "[[VariationalAutoencoderforTTS]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[NeuralVocoder]]✓, [[ProsodyModeling]]✓, [[SpeakerEmbedding]]✓ + 3 个待确认页: [[DurationPredictor]][待确认], [[Non-autoregressiveTTS]][待确认], [[VITS]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[DurationPredictor]], [[Non-autoregressiveTTS]], [[VITS]], [[NeuralVocoder]], [[ProsodyModeling]], [[SpeakerEmbedding]] | 过滤: [[Multi-scaleSTFTDiscriminator]](confirmed, 辅助参考), [[VariationalAutoencoderforTTS]](pending-review) | 未命中但可能相关: 无

**谱系定位**: FNH-TTS 位于 VITS 系列的直接改进路线上。VITS (Kim et al., ICML 2021) 开创了 VAE+Flow+GAN 端到端 NAR TTS 范式,其后续 VITS2 通过对抗学习改善多说话人 duration prediction。FNH-TTS 沿这条路线继续,但方向不同于 VITS2 的对抗目标 -- 它从 Duration Predictor 的内部结构入手,引入 Mixture-of-Experts 机制。

**已有认知**:
- Duration Predictor 概念页记录了从确定性预测 (DDP) 到概率预测 (SDP) 再到 RL 优化 (DMOSpeech 2 GRPO, FlexSpeech DPO) 的演进路线。FNH-TTS 的 MoE-DP 属于 DDP 类别的结构创新,与概率化/RL 优化方向正交。
- Neural Vocoder 概念页记录了 HiFi-GAN → Vocos (iSTFT-based) 的演进。Vocos 已被定位为 "closing the gap between time-domain and Fourier-based vocoders"。FNH-TTS 将 Vocos 集成到 VITS 端到端框架中,是这一趋势的实际验证。
- Prosody Modeling 概念页将 duration 列为韵律的核心物理维度之一,并指出 "one-to-many mapping" 是韵律建模的根本挑战。FNH-TTS 的 MoE 多专家结构正是试图用结构化方法应对 duration 多样性。

**创新判断**: 已有 Duration Predictor 演进线主要关注 supervision strategy (概率化 SDP, RL 优化 GRPO/DPO) 和 alignment method (MAS, CTC, MFA),而 DP 内部结构的改进相对未被探索。MoE-DP 是首次将 Mixture-of-Experts 引入 Duration Predictor 的工作。此外,论文明确揭示了"更丰富的 duration 变化 → 加大 vocoder 合成难度"这一 duration-vocoder 耦合现象,这在已有概念页中未被记录。

> [!summary] 速查
> - **一句话**: 将 Mixture-of-Experts 引入 VITS 的 Duration Predictor,同时用 VOCOS vocoder + CoMBD/SBD 判别器应对更丰富 duration 变化带来的合成难度提升
> - **路线**: Phoneme + Speaker Emb → MoE-DP (1D Conv + Switch-Transformer, 8 experts) → Duration → VITS Flow/Prior → VOCOS (ConvNeXt + ISTFT) → Waveform; CoMBD + SBD 做对抗训练
> - **指标**: MOS 4.48 (LJSpeech) / 4.63 (VCTK) [Table 1]; Duration-category ACC 67.07% (最高) [Table 2]; CPU RTF 0.046 (最低) [Table 3]; 模型 47.73M 参数
> - **可借鉴**: (1) MoE 做 DP 的结构化路线,speaker-conditioned router 让不同专家专注不同 duration pattern; (2) "duration 改进必须配合 vocoder 升级"的联合优化思路; (3) Jensen-Shannon divergence 评估 phoneme-level duration 对齐质量的方法
> - **局限**: 仅在 VITS 框架内验证,未验证 MoE-DP 对 flow matching / diffusion TTS 的通用性; 评估未包含 speaker similarity 指标; 未开源

## 核心问题

FNH-TTS 要解决两个相互关联的问题:

1. **Duration Predictor 的结构性局限**: 现有 NAR TTS 中的 DP 通常是简单的 Conv1D 堆叠,产生过度平滑 (over-smoothed) 和说话人无关 (speaker-insensitive) 的 duration 预测 [§1]。这在多说话人合成中尤为突出 -- 不同说话人天然具有不同的语速模式,但传统 DP 倾向于将这些模式坍缩为高度重叠的分布 [论文原文]。

2. **Duration-Vocoder 耦合效应**: 当 DP 确实产生了更丰富的 duration 变化后,HiFi-GAN-based vocoder 反而出现光谱伪影 (spectral artifacts) 和不连续的时频结构 [§1, §4.1]。这意味着仅改善 duration modeling 不足以保证合成质量提升 -- vocoder 侧也必须同步增强 [论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

FNH-TTS 建立在 VITS 框架之上,保留 Text Encoder、Speaker Encoder、Posterior Encoder 和 Flow 模块不变 [§2]。主要修改两个组件:

1. **Duration Predictor**: 替换为 MoE Duration Predictor (MoE-DP)
2. **Vocoder + Discriminator**: 替换为 VOCOS-style vocoder + CoMBD + SBD

总损失函数: L = L_rec + L_kl + L_dur + L_adv + L_gen [Eq. 1]

其中 L_rec 和 L_kl 是 VITS 原始的重建和 KL 损失,保持不变。

### 关键设计选择

#### MoE Duration Predictor (MoE-DP)

**为什么用 MoE 而非简单加大 DP 容量**: MoE 的核心优势在于"稀疏路由下的专家专精" -- 不同专家可以学习不同的 duration 模式 (如不同说话人的语速模式、不同音素上下文的时长分布),而推理时只激活 top-k 专家,不显著增加计算开销 [论文原文, §2.1]。

**架构细节** [§2.1, §2.3]:
- 前端: 2 个 1D Conv blocks (kernel=3, hidden=192)
- 后端: 2 个 Switch-Transformer blocks (8 experts each, 4 heads, hidden=192)
- 输入: h_text + s (text encoder hidden + speaker embedding)
- **Speaker-conditioned routing**: router 的输入为 x + s,即包含说话人信息 [Eq. 2]。这使得路由决策本身就是说话人感知的 -- 同一音素在不同说话人下可能路由到不同专家 [论文原文]。

**路由公式**:
y = Σ_{i∈Ψ} p_i(x + s) · E_i(x) [Eq. 2]
其中 Ψ 是 top-k 选中的专家集合。

**负载均衡**: 使用 Switch Transformer 的 load balancing loss [Eq. 3-5],防止所有 token 路由到同一专家。L_dur = L_mas + L_aux [Eq. 6],其中 L_mas 来自 MAS 对齐结果,L_aux 是加权负载均衡损失。

**为什么选 Switch Transformer 而非其他 MoE 变体**: [agent 解读] Switch Transformer 采用 top-1 routing (每个 token 只激活 1 个专家),推理开销最小,适合 DP 这种轻量模块。相比 top-2 routing (如 GShard),更符合 NAR TTS 追求高效推理的目标。

#### VOCOS-style Vocoder

**为什么替换 HiFi-GAN**: 实验发现 MoE-DP 产生的更丰富 duration 变化显著增加了 HiFi-GAN 的合成难度,导致频谱伪影和时频不连续 [§2.2, §4.1]。[agent 解读] HiFi-GAN 的转置卷积上采样链可能在处理更多变的帧级输入时引入 aliasing,而 VOCOS 的 ISTFT 机制天然产生更连续的频谱。

**集成方式** [§2.2, Eq. 7]:
- 输入: posterior latent z + speaker embedding s
- 骨干: ConvNeXt blocks (8 blocks, intermediate 1536, hidden 512) 将 z+s 变换为 complex STFT 系数
- 输出: ISTFT 重建波形 ŵ = ISTFT(Backbone(z + s))
- hop=256, FFT=1024

#### Collaborative Multi-Band Discriminator (CoMBD) + Sub-Band Discriminator (SBD)

**CoMBD**: 在不同波形分辨率上操作,每个分辨率共享同一 MSD 架构。改善全局波形连贯性和局部结构 [§2.2]。

**SBD**: 用 PQMF 分析将波形分解为多个子带信号,每个子带用多尺度膨胀卷积提取特征。减少高频失真,改善低频稳定性 [§2.2]。

**为什么需要两个判别器**: [论文原文] 仅升级 vocoder 不足以消除 duration 多样性带来的频谱伪影。CoMBD 提供时域多尺度监督,SBD 提供频域子带监督,两者互补确保时频两个维度的合成鲁棒性。

### 训练策略

- 优化器: AdamW (β1=0.8, β2=0.99) [§2.3]
- 学习率: 2×10^-4,每 epoch 衰减 0.999 [§2.3]
- Batch size: 24 [§2.3]
- 硬件: 4× NVIDIA RTX 3090 [§2.3]
- MoE 配置: 2 个 Switch-Transformer blocks × 8 experts × top-k routing [§2.3]

## 实验

| 指标 | FNH-TTS | VITS Origin | F5-TTS | SparkTTS | StyleTTS2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MOS | 4.48 | 4.26 | 3.87 | 4.10 | 4.35 | LJSpeech | [Table 1] |
| MOS | 4.63 | 4.34 | 4.49 | 4.38 | - | VCTK | [Table 1] |
| WER | 2.59% | 3.41% | 8.70% | 7.36% | 5.78% | LJSpeech | [Table 1] |
| WER | 3.88% | 4.11% | 2.24% | 4.98% | - | VCTK | [Table 1] |
| Duration ACC | 67.07% | - | 11.17% | 66.77% | 59.28% | Libri460 | [Table 2] |
| Model Size | 47.73M | 39.53M | 337.09M | 506.63M | 145.53M | - | [Table 1] |

### 消融实验关键发现

1. **MoE-DP 单独使用反而降低 MOS** (LJ: 3.92, VCTK: 3.90 vs 原始 VITS 4.26/4.34) [Table 1]。原因: 更丰富的 duration 变化超出了 HiFi-GAN vocoder 的合成能力 [论文原文, §4.1]。

2. **MoE-DP + CoMBD/SBD** (无 VOCOS) 就能大幅恢复质量 (LJ: 4.20, VCTK: 4.43) [Table 1],说明判别器升级是恢复合成质量的主要因素。

3. **VOCOS 的贡献不在 MOS/WER,而在推理效率和频谱细节**: CPU RTF 从 0.352 (HiFi-GAN) 降至 0.046 (VOCOS),7.6x 加速 [Table 3]。频谱图分析显示 VOCOS 改善了高低频区域的精细纹理 [Fig 4]。

4. **WER 不是评估韵律建模的可靠指标**: F5-TTS 在 VCTK 上 WER 最低 (2.24%),但 MOS 低于 FNH-TTS [Table 1]。Duration ACC 低的系统 (F5-TTS: 11.17%) 可能 WER 反而好 -- 因为复杂韵律变化本身就增加 ASR 难度 [论文原文, §4.2]。

### Vocoder 重建质量 (Analysis-by-Synthesis)

| 指标 | FNH-TTS | VITS(HiFi-GAN) | VITS(HiFi-GANv2) | 出处 |
| --- | --- | --- | --- | --- |
| M-STFT↓ | 1.207 | 1.208 | 1.246 | [Table 3] |
| PESQ↑ | 2.425 | 2.441 | 2.231 | [Table 3] |
| MCD↓ | 1.654 | 1.677 | 1.814 | [Table 3] |
| RTF(CPU)↓ | 0.046 | 0.352 | 0.181 | [Table 3] |
| RTF(GPU)↓ | 0.00460 | 0.00748 | 0.00659 | [Table 3] |

### JS Divergence (Phoneme-level Duration 对齐)

VITS+MoE-DP 在 LJSpeech 上 JS=0.053,VCTK 上 JS=0.039,均低于 VITS+DDP (0.057/0.044) 和 VITS+SDP (0.087/0.066) [Table 4]。验证了 MoE-DP 在 phoneme 粒度上的 duration 建模改进。

## 局限性

1. **验证范围局限于 VITS**: MoE-DP 的效果仅在 VITS 框架内验证,未扩展到 flow matching (如 Matcha-TTS) 或 diffusion-based TTS,通用性未知。
2. **无 speaker similarity 评估**: 多说话人实验 (VCTK) 未报告 SECS 或 speaker similarity 指标,无法评估 speaker-conditioned routing 对音色保持的影响。
3. **MoE-DP 的专家专精缺乏可解释分析**: 论文未分析各 expert 实际学到了什么样的 duration 模式,无法确认 "不同专家专精不同 pattern" 的设计假设是否成立。
4. **未开源**: 截至论文发布 (2026-05),未提供代码或模型权重。
5. **评估数据集有限**: 未在 Seed-TTS-Eval 等更标准的零样本评估协议上测试。
6. **WER 批判但自己也依赖 WER**: 论文正确指出 WER 不适合评估韵律,但自身报告中仍大量使用 WER 作为对比指标。

## 点评

FNH-TTS 的核心价值不在于 MoE 或 VOCOS 这些单独组件 (它们都是已有技术),而在于揭示了一个被忽视的系统性问题: **duration modeling 和 vocoder 不能独立优化**。这个 "duration-vocoder 耦合效应" 的发现比具体方法更有洞察力 -- 它解释了为什么很多 duration 改进工作在最终合成质量上效果不明显。

从方法角度看,MoE-DP 是 Duration Predictor 结构创新的一个自然但有效的方向。Speaker-conditioned routing 的设计让 MoE 不仅仅是增加容量,而是提供了一种结构化的方式来处理多说话人 duration 分布的异质性。

不足之处在于实验的说服力: 仅在 VITS 框架内验证限制了结论的通用性,缺乏 speaker similarity 指标让多说话人场景的评估不完整。此外,对 MoE 专家行为的分析停留在宏观层面 (duration 分布可视化),未深入到 expert-level 的专精分析。

## 可复用的 idea

1. **MoE 做 Duration Predictor**: Switch-Transformer blocks 替换 DP 中的 FFN,speaker embedding 同时注入 router 和 expert input。这个设计可直接移植到任何需要 duration prediction 的 NAR TTS 系统中。
2. **Duration-Vocoder 联合优化思路**: 改善 duration modeling 后必须同步评估 vocoder 的承受能力,否则可能出现 "duration 更好但合成更差" 的反直觉结果。这对任何涉及 prosody 改进的 TTS 工作都是重要提醒。
3. **JS Divergence 作为 Duration 评估指标**: 对比预测与 GT 的 phoneme-level duration 序列的 Jensen-Shannon divergence,比 utterance-level accuracy 更精细。symmetric + bounded 的特性优于 KL divergence。
4. **Analysis-by-Synthesis 协议**: 将 GT 音频经 posterior encoder 再由 vocoder 重建,隔离评估 vocoder 重建能力,避免 duration prediction 和 alignment 的干扰。
5. **VOCOS 集成到 VITS**: 将 VOCOS 的 ConvNeXt + ISTFT 架构嵌入 VITS 的 z→waveform 路径,conditioned on z+s,实现 7.6x CPU 推理加速。

> [!review] 审阅: pass-with-fixes (2026-06-04)
> - 3 个 low 级别问题 (traceability-gap ×1, fact-inference-mixing ×1, template-compliance ×1)
> - 详见 `_review/FNH-TTS-review.yml`
> - 结论: 放行反向更新

---

检索命中: [[DurationPredictor]], [[Non-autoregressiveTTS]], [[VITS]], [[NeuralVocoder]], [[ProsodyModeling]], [[SpeakerEmbedding]] | 过滤: [[Multi-scaleSTFTDiscriminator]](confirmed), [[VariationalAutoencoderforTTS]](pending-review) | 未命中但可能相关: 无
