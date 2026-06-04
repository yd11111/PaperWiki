---
type: paper
tier: deep
title: "DiFlow-TTS: Compact and Low-Latency Zero-Shot Text-to-Speech with Factorized Discrete Flow Matching"
arxiv_id: "2509.09631"
source: "Sources/2509.09631.pdf"
authors: [Ngoc-Son Nguyen, Thanh V. T. Tran, Hieu-Nghia Huynh-Nguyen, Truong-Son Hy, Van Nguyen]
year: 2025
venue: "arXiv preprint"
tags: [TTS, zero-shot, discrete-flow-matching, FACodec, non-autoregressive, efficiency, compact, factorization, DiT]
concepts: ["[[Conditional Flow Matching]]", "[[Speech Factorization]]", "[[Duration Predictor]]", "[[Residual Vector Quantization]]", "[[Non-autoregressive TTS]]", "[[Masked Generative Modeling]]", "[[Prosody Modeling]]", "[[Speech Tokenizer]]"]
models: ["[[模型库/NaturalSpeech 3|NaturalSpeech 3]]", "[[模型库/NaturalSpeech 2|NaturalSpeech 2]]", "[[模型库/EnCodec|EnCodec]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[Conditional Flow Matching]], [[Speech Factorization]], [[Residual Vector Quantization]], [[Prosody Modeling]]; 2 个待确认实体页: [[Duration Predictor]] [待确认], [[Non-autoregressive TTS]] [待确认]; 1 个已确认任务页: [[Zero-shot Speech Synthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: DiFlow-TTS 在 TTS 生成范式谱系中占据一个独特位置——它是首个将 **离散 flow matching (DFM)** 直接应用于语音生成的系统。现有 flow matching TTS 系统(如 CosyVoice 系列、F5-TTS、Matcha-TTS)均在连续空间操作(mel spectrogram 或 continuous latent),而 DiFlow-TTS 直接在离散 codec token 的概率分布上定义 flow。这与 NaturalSpeech 3 的 factorized discrete diffusion 思路有渊源——两者都使用 FACodec 将语音分解为 content/prosody/acoustic 子空间,但 NS3 用的是 discrete diffusion,DiFlow-TTS 用的是 discrete flow matching。同时,DiFlow-TTS 与同组的 OZSpeech 共享 FACodec backbone 和 factorized 设计,但 OZSpeech 用 continuous OT-CFM + learned prior 实现单步采样,而 DiFlow-TTS 探索纯离散的 DFM 路线。
>
> **已有认知**: 概念库中 [[Conditional Flow Matching]] 已覆盖连续 flow matching 的丰富应用(CFM → rectified flow → shallow flow matching 演进线),但尚无 discrete flow matching 的条目。[[Speech Factorization]] 覆盖了 content/prosody/timbre 分解的多种方法(对抗训练、信息瓶颈、self-distillation),DiFlow-TTS 的 factorized 设计直接继承 NaturalSpeech 3 的 FACodec 路线。[[Duration Predictor]] [待确认] 已记录 FastSpeech 系 MFA-based 方案,DiFlow-TTS 的 PCM 模块本质是 duration predictor + content predictor 的组合。
>
> **创新判断**: 相较已有知识,DiFlow-TTS 的核心新意在于 (1) 将 DFM 从 NLP/图/蛋白质领域引入语音,(2) 在 DFM 框架中引入 factorized probability velocity field(分头预测 prosody 和 acoustic 的概率速度),这在离散 flow matching 文献中属首次。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Speech Factorization]]✓, [[Residual Vector Quantization]]✓, [[Prosody Modeling]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Duration Predictor]](pending-review), [[Non-autoregressive TTS]](pending-review) | 未命中但可能相关: Discrete Flow Matching (无独立页)

> [!summary] 速查
> - **一句话**: 首个将离散 flow matching 应用于零样本 TTS 的系统,通过 factorized probability velocity field 分头建模 prosody 和 acoustic token,以 122-164M 参数实现 UTMOS 3.98 + RTF 0.03-0.07,模型最小 11.7x、推理最快 34x 于 baseline
> - **路线**: Text → Phoneme Encoder → Duration Predictor → Length Regulator → Content Predictor → content embeddings; Speech Prompt → FACodec → (prosody tokens, acoustic tokens, speaker embedding); 两路 + masked tokens → Factorized Discrete Flow Denoiser (DiT + multi-head prediction) → prosody + acoustic tokens → FACodec Decoder → waveform
> - **指标**: UTMOS 3.98 / WER 0.05 / SIM-O 0.45 / F0 Acc 0.88 / F0 RMSE 7.97 (LibriSpeech test-clean, 3s prompt, 128 NFE) [Table 1]; MOS naturalness 4.18 / intelligibility 4.41 / similarity 4.42 [Table 2]; 164M params, RTF 0.07 (16 NFE) [Table 3]
> - **可借鉴**: factorized multi-head prediction 机制——用独立 prediction head 为不同属性子空间预测概率速度,可迁移到任何多属性离散生成任务; attribute-type embedding 区分不同属性流的简单有效设计
> - **局限**: SIM-O 仅 0.45(低于 F5-TTS 0.66、MaskGCT 0.67),speaker conditioning 过于简单(仅 global AdaLN);依赖预训练 FACodec(NaturalSpeech 3)不可端到端优化;仅在 470h LibriTTS 上训练,未验证大规模 scaling;代码/权重未开源

## 核心问题

现有 flow-based TTS 模型(Matcha-TTS、F5-TTS、CosyVoice 系列)都在**连续空间**上定义 flow——要么直接操作 mel spectrogram,要么先将离散 token 映射到连续表示再做 flow matching。这引入了一个中间层的信息损失,且与 speech codec 的离散本性存在 representation mismatch。能否直接在**离散 token 的概率分布空间**上做 flow matching,跳过连续空间中介?

此外,语音是多属性信号(content + prosody + acoustic + timbre),现有 DFM 方法处理的是同质离散序列(文本、蛋白质序列)。如何将 DFM 扩展到**结构化、多属性**的离散空间?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DiFlow-TTS 由三个模块组成 [§3, Fig 2]:

1. **Speech Tokenizer**: 使用预训练 FACodec (NaturalSpeech 3) 将语音分解为 prosody tokens (1 层 RVQ)、content tokens (2 层 RVQ)、acoustic tokens (3 层 RVQ) 和 speaker embedding [§3.2, Eq 2]
2. **Phoneme-Content Mapper (PCM)**: 从文本生成内容 token 和内容嵌入,包含 Phoneme Encoder + Duration Predictor + Length Regulator + Content Predictor [§3.3]
3. **Factorized Discrete Flow Denoiser (FDFD)**: 基于 DFM 的核心生成模块,以 content embedding + speaker embedding + reference prompt 的 prosody/acoustic tokens 为条件,生成目标语音的 prosody 和 acoustic tokens [§3.4]

最终将 PCM 预测的 content tokens + FDFD 生成的 prosody/acoustic tokens + speaker embedding 送入 FACodec Decoder 重建波形。

### 关键设计选择

**为什么选择 Discrete Flow Matching 而非 continuous FM 或 discrete diffusion?**

[论文原文] Diffusion-based 方法将训练和采样过程 inherently coupled——修改 noise schedule 或 rate matrix 需要重新训练 [§1]。Continuous flow matching 虽然灵活,但现有 speech FM 模型都定义在从离散输入导出的连续表示上,而非直接在离散分布上 [§1]。DFM 直接在离散概率分布上定义 flow,避免了连续中间表示的信息损失,且训练-采样解耦更灵活。

[agent 解读] 这个动机合理但需要注意:连续 FM 的中间表示损失在实践中可能很小(F5-TTS、CosyVoice 的连续 FM 已取得很好效果)。DFM 的真正优势可能更多在于 NFE 灵活性——同一模型可以用 4 步或 128 步推理,质量-速度 trade-off 非常平滑 [Table 5]。

**为什么用 Factorized Probability Velocity Field?**

[论文原文] 与传统 DFM 处理同质离散序列不同,语音的 prosody 和 acoustic 属性具有不同分布特征。通过为 prosody 和 acoustic 各设一个 prediction head (fφ, fω),模型可以对不同子空间的概率速度做独立预测,增强 prediction diversity 和 robustness [§3.4]。

[agent 解读] 这个设计与 NaturalSpeech 3 的 factorized diffusion model 有异曲同工之处,但关键区别在于 NS3 用**独立的 diffusion 模型**分别处理各属性,而 DiFlow-TTS 在**统一的 flow process** 中用 multi-head prediction 分头输出。这保留了属性间的交互(共享 DiT backbone 的隐表示),同时允许各属性有独立的生成分布。

**为什么保留 Duration Predictor 而非使用 duration-free 设计(如 F5-TTS)?**

[论文原文] PCM 模块 "inspired by conventional duration-based alignment mechanisms" [§3.3],将 phoneme 对齐到离散 codec token 空间(而非连续 mel 帧)。

[agent 解读] Duration predictor 给了 DiFlow-TTS 对生成长度的精确控制,这在资源受限场景(目标应用场景)中是优势。但这也带来了对外部强制对齐工具(MFA)的依赖。相比之下 F5-TTS/MaskGCT 完全不需要 phoneme alignment。

**Source Distribution 设计**: 从全 [MASK] token 序列出发,通过 κ_t 调度器线性插值到目标分布 [§3.1, Eq 1]。κ_t = t² (cubic polynomial family) [Appendix A]。

**Contextual Modeling (三步构造条件)** [§3.4]:
1. 输入构造: reference prompt 的 prosody/acoustic tokens 做 embedding,corrupted 序列 x_t 也做 embedding
2. 条件拼接: 沿时间维度拼接 reference 和 corrupted embeddings,content embedding 替代 corrupted content (reference content 设为 zero)
3. 统一嵌入: 加入 learnable attribute-type embedding (g_p, g_c, g_a),区分 prosody/content/acoustic 三种属性流 → 拼接 → flatten → 投影 → 送入 DiT

**Speaker Conditioning**: speaker embedding 通过 AdaLN 注入 DiT blocks——与 timestep embedding 求和后生成 scale/shift 参数调制每层特征 [§3.4]。

### 训练策略

三个 loss 联合训练 [§3.5, Eq 4]:
- L_dur: Duration Predictor 的 MSE loss (log scale)
- L_c: Content Predictor 的 cross-entropy loss
- L_FDFD: FDFD 的 cross-entropy loss (masked token recovery under varying masking ratios)

权重: λ_dur = 0.5, λ_c = 1.0, λ_FDFD = 1.0 [Appendix A]

训练配置: 4×A100 GPU, 315K steps, batch size 16, AdamW (lr=1e-4, weight decay 0.01), 200K warmup steps [Appendix A]

Reference prompt 选取: 训练时随机采样 ground-truth 序列 30% 长度的片段作为 reference [Appendix B.3]

FDFD 架构: DiT blocks, hidden size 768, 12 layers, 12 attention heads, RoPE [Appendix A]

## 实验

| 指标 | DiFlow-TTS (128 NFE) | F5-TTS [†] (100K h) | MaskGCT [†] (100K h) | OZSpeech [†] | VALL-E [⋄] | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UTMOS ↑ | **3.98** | 3.72 | 3.83 | 3.15 | 3.68 | LibriSpeech test-clean (3s) | [Table 1] |
| WER ↓ | **0.05** | 0.09 | 0.09 | **0.05** | 0.19 | LibriSpeech test-clean (3s) | [Table 1] |
| SIM-O ↑ | 0.45 | **0.66** | **0.67** | 0.40 | 0.40 | LibriSpeech test-clean (3s) | [Table 1] |
| F0 Accuracy ↑ | **0.88** | 0.83 | 0.77 | 0.81 | 0.75 | LibriSpeech test-clean (3s) | [Table 1] |
| F0 RMSE ↓ | **7.97** | 12.66 | 14.33 | 11.96 | 21.66 | LibriSpeech test-clean (3s) | [Table 1] |
| MOS Naturalness ↑ | **4.18** | 3.97 | 3.97 | 2.80 | 3.71 | MOS (30 listeners, 3s) | [Table 2] |
| MOS Similarity ↑ | **4.42** | 4.07 | 4.17 | 3.20 | 3.99 | MOS (30 listeners, 3s) | [Table 2] |

**模型大小与速度** [Table 3]:

| 模型 | #Params | NFE | RTF |
| --- | --- | --- | --- |
| DiFlow-TTS-Small | 122M | 4 / 16 | 0.03 / 0.05 |
| DiFlow-TTS | 164M | 4 / 16 | 0.03 / 0.07 |
| F5-TTS | 336M | - | 0.26 |
| MaskGCT | 1.43B | 50+45 | 0.46 |
| OZSpeech | 145M | 1 | 0.03 |

**消融实验** [Table 4]:
- 去掉 content embedding → UTMOS 从 3.978 暴降到 3.077,SIM-O 从 0.454 降到 0.333——**content embedding 是最关键的条件信号** [论文原文]
- 去掉 speaker embedding → SIM-O 从 0.454 降到 0.378,F0 RMSE 从 7.972 飙升到 20.868——说明 **prosody 受 speaker identity 强烈影响** [论文原文]
- 去掉 attribute embedding → 轻微降级,UTMOS 反而微升 (3.978→3.983)——attribute embedding 增强保真度但可能引入 minor redundancies [论文原文]
- single-head 替代 multi-head → 各指标 minor drop,multi-head 提升 prediction diversity [§4.3]

**Noisy Prompt 分析** [Fig 4]: DiFlow-TTS 在所有噪声水平下 UTMOS 最高,WER 几乎不随 SNR 降低而退化,展现出对噪声 prompt 的强鲁棒性。

**NFE vs Quality Trade-off** [Table 5]: 4 NFE 时 UTMOS 3.31 (RTF 0.03),16 NFE 时 3.86 (RTF 0.07),32 NFE 时 3.92,64 NFE 时 3.96,128 NFE 时 3.98——性能在 32 NFE 后趋于饱和,16 NFE 已是很好的 speed-quality 平衡点。

## 局限性

1. **Speaker similarity 不足**: SIM-O 0.45 显著低于 F5-TTS (0.66) 和 MaskGCT (0.67)。作者归因于过于简单的 speaker conditioning (global AdaLN) 和 FACodec 将 speaker identity 显式解耦出 token 的设计——模型需要额外机制注入 speaker 信息 [Limitations section]。这是当前设计的主要弱点。

2. **依赖 FACodec**: 系统绑定到 NaturalSpeech 3 的 FACodec,无法端到端优化 codec。作者提到未来计划支持 EnCodec 等直接在 VQ 中嵌入 speaker 信息的 codec [Limitations section]。

3. **训练数据规模有限**: 仅 470h LibriTTS,未验证在大规模数据 (10K+ hours) 上是否能进一步提升,特别是 speaker similarity。

4. **推理步数仍需权衡**: 虽然 4 NFE 已可运行 (RTF 0.03),但质量与 128 NFE 差距不小 (UTMOS 3.31 vs 3.98)。最佳 speed-quality 点 (16 NFE, UTMOS 3.86) 仍需 16 步迭代。

5. **评估局限**: 仅在 LibriSpeech test-clean 上评测,缺少 SEED-TTS-Eval 等更广泛 benchmark 的验证;未评估多语言和跨语言场景。

6. **MOS 与 SIM-O 矛盾**: MOS similarity (4.42) 远好于 SIM-O (0.45)——作者指出 SIM-O 这种 embedding 空间 proxy 可能惩罚了人耳听不到的 artifacts,而人类更关注 pitch/timbre/prosody 等感知线索 [§4.2]。这个解释合理但也意味着 objective speaker similarity 确实有问题。

## 点评

DiFlow-TTS 的核心价值不在于 SOTA 性能(speaker similarity 明显弱),而在于**方向探索**——验证了 discrete flow matching 在语音生成中的可行性。几个观察:

1. **DFM vs 连续 FM 的定位**: 论文将 DFM 定位为避免"连续中间表示信息损失"的方案,但 F5-TTS/CosyVoice 的连续 FM 在 speaker similarity 上远胜 DiFlow-TTS。真正的优势可能在于: (a) 与离散 codec 的天然兼容性,(b) 极其灵活的 NFE 选择 (4 到 128 步平滑过渡),(c) 紧凑的参数量 (122-164M vs MaskGCT 1.43B)。

2. **Factorized multi-head prediction 是有价值的设计**: 消融证明它有正向贡献但不大。更重要的是它提供了一个清晰的 conceptual framework——在统一 flow 中用 shared backbone + specialized heads 处理异构属性。

3. **与 OZSpeech 的对比**: 同组同 codec,OZSpeech 用连续 OT-CFM + learned prior 实现 1 步推理;DiFlow-TTS 用离散 DFM 需要 4-128 步。两者 RTF 相当 (均约 0.03),但 DiFlow-TTS 在 UTMOS/F0 指标上全面优于 OZSpeech——说明多步离散 flow 确实比单步连续 flow 能更好地捕获 prosodic nuance。

4. **数据效率令人印象深刻**: 仅用 470h 训练数据,UTMOS 和韵律指标超过使用 100K h 数据的 MaskGCT/F5-TTS。这可能归功于 FACodec 的预训练提供了强先验,也可能是 DFM 的训练效率优势。

5. **Speaker similarity 问题是系统性的**: FACodec 显式分离了 speaker identity,使得 speaker 信息只能通过 global AdaLN 注入——这是一个过于粗糙的通道。作者提出的 cross-attention timbre embedding 方案值得后续探索。

## 可复用的 idea

1. **Factorized multi-head prediction for DFM**: 在统一的 discrete flow 中为不同属性子空间设置独立 prediction head,共享 backbone 建模交互 + 专用 head 建模分布差异。可迁移到任何多属性离散生成任务(多轨音乐生成、多模态生成)。

2. **Attribute-type embedding**: 简单的 learnable embedding 区分不同属性流,帮助 Transformer 在 concatenated input 中辨别属性类型。成本极低但消融证实有效。

3. **PCM (Phoneme-Content Mapper) 设计**: 将 duration prediction 从连续帧对齐改为离散 token 对齐,hierarchical FFT blocks 逐层提取 n 个 content representations——可迁移到其他需要 text-to-discrete-token 对齐的场景。

4. **Cubic scheduler κ_t = t²**: 相比线性调度,cubic 在初期更慢(给模型更多时间学习全局结构)、后期更快(细节填充),是 DFM 调度的一个简单有效选择。

> [!review] 审阅状态
> 待审阅。见 `_review/DiFlow-TTS-review.yml`。
