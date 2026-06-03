---
type: paper
tier: deep
title: "ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching"
arxiv_id: "2506.13053"
source: "https://arxiv.org/abs/2506.13053"
authors: [Han Zhu, Wei Kang, Zengwei Yao, Liyong Guo, Fangjun Kuang, Zhaoqing Li, Weiji Zhuang, Long Lin, Daniel Povey]
year: 2025
venue: "arXiv"
tags: [zero-shot-TTS, flow-matching, NAR-TTS, model-compression, inference-acceleration, distillation]
concepts: ["[[Conditional Flow Matching]]", "[[Classifier-Free Guidance]]", "[[Non-autoregressive TTS]]", "[[Duration Predictor]]", "[[Speech-Text Alignment]]", "[[Neural Vocoder]]"]
models: ["[[论文笔记/E2 TTS|E2 TTS]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/F5R-TTS|F5R-TTS]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["Emilia", "LibriTTS"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页: [[Conditional Flow Matching]], [[Zero-shot Speech Synthesis]], [[Classifier-Free Guidance]], [[Non-autoregressive TTS]], [[Duration Predictor]], [[Speech-Text Alignment]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: ZipVoice 属于 flow-matching-based NAR zero-shot TTS 一脉,与 E2-TTS / F5-TTS 同源(均基于 speech infilling + conditional flow matching),但聚焦于效率问题。在知识库中,当前 flow matching TTS 的效率探索主要通过 consistency distillation (Seed-TTS)、ReFlow (VoiceFlow) 等方式实现,ZipVoice 的 flow distillation 方法与这些加速策略构成并行路线。
>
> **已有认知**: (1) [[Conditional Flow Matching]] 已成为 TTS 的主流生成框架,通过 ODE 路径实现比 diffusion 更少步数的生成; (2) [[Classifier-Free Guidance]] 在推理时需要额外的无条件推理 pass,是推理开销的重要来源; (3) 当前 zero-shot TTS 的对齐方式分为显式 (MFA/MAS + duration predictor) 和隐式 (filler token padding 如 E2-TTS)两派,ZipVoice 的 average upsampling 是第三种折中方案; (4) [[Duration Predictor]] 页面记录了 DMOSpeech 2 和 FlexSpeech 对 duration 优化的最新进展,ZipVoice 则完全绕过 phone-level duration prediction。
>
> **创新判断**: ZipVoice 的核心创新在于三方面——(a) Zipformer 从 ASR 迁移到 TTS flow matching backbone (此前未见类似尝试), (b) average upsampling 替代 filler token padding 和显式 duration prediction 的对齐策略, (c) flow distillation 将 CFG 的多次推理压缩为单次前向。这三者组合使得 123M 模型在 100K 小时数据上达到与 336M F5-TTS 相当的质量。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Classifier-Free Guidance]](待确认), [[Non-autoregressive TTS]](待确认), [[Duration Predictor]](待确认), [[Speech-Text Alignment]](待确认) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 Zipformer (ASR backbone) 替代 DiT/Transformer 作为 flow matching 的 vector field estimator,配合 average upsampling 对齐和 flow distillation 加速,实现 123M 参数下 SOTA 级 zero-shot TTS 质量,推理速度比 F5-TTS 快 ~24-33x
> - **路线**: Text → Zipformer Text Encoder → Average Upsample → Channel-wise Concat(text condition + speech condition + noisy speech) → Zipformer Vector Field Estimator → ODE Solver → Vocos Vocoder → Waveform
> - **指标**: LibriSpeech-PC: WER 1.64/SIM-o 0.668/UTMOS 3.98 (16NFE); Seed-TTS test-en: WER 1.70/SIM-o 0.697 (16NFE); ZipVoice-Distill(4NFE) RTF 0.0125 GPU / 1.22 CPU vs F5-TTS(32NFE) RTF 0.2958 GPU [Table I, Table II]
> - **可借鉴**: (1) Zipformer 的 U-Net downsampling + convolution + attention weight reuse 三要素可作为 flow matching backbone 的高效设计参考; (2) Average upsampling 作为极简对齐策略,无需 duration predictor 也无需 filler token,一行公式 d=floor(T/N); (3) Flow distillation 通过 teacher 两步推理构造 target vector field + CFG strength 作为 model input,同时消除 CFG 双倍开销和减少 NFE
> - **局限**: (1) SIM-o 不如 MaskGCT/E2-TTS 等大模型,speaker similarity 有损; (2) Average upsampling 的均匀 duration 假设与真实发音差距大,依赖 flow matching 模型隐式纠正; (3) 仅在英语和中文上评估; (4) 开源但尚无大规模社区验证

## 核心问题

ZipVoice 试图回答: **能否在不牺牲 SOTA 语音质量的前提下,将 flow-matching-based zero-shot TTS 的模型体积缩小到 1/3、推理速度提升到 30 倍?**

现有 SOTA zero-shot TTS (如 F5-TTS 336M、MaskGCT 1048M) 依赖大参数量维持建模能力,且 flow matching 的多步 ODE 采样 + CFG 的双倍推理开销导致推理慢。ZipVoice 要在三个维度同时突破: 更小的模型、更少的采样步数、消除 CFG 额外推理。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ZipVoice 建立在 conditional flow matching (CFM) 框架之上,采用 speech infilling 任务实现 zero-shot TTS [§II-B]。整体由两个子模型组成:

1. **Zipformer-based Text Encoder** (4 层, dim=192, FFN=512): 将 text tokens 编码为 text features [§II-C]
2. **Zipformer-based Vector Field Estimator** (5 stacks, [2,2,4,4,4] layers, dim=512, FFN=1536): 核心生成模型,从噪声中预测 vector field [§IV-B]

总参数量约 123M [§IV-B],对比 F5-TTS 336M 和 MaskGCT 1048M 显著更小。

**训练过程** [§II-B, Fig 1]: 对语音特征 x1 施加随机 mask(70%-100% 长度),拼接 text condition z、speech condition (1-m)⊙x1、noisy speech xt 作为输入,训练 vector field estimator 重建 masked 部分。

**推理过程** [§II-F, Fig 1]: 给定 prompt 语音及其转写 + 目标文本,text encoder 编码后 average upsample,prompt 语音作为 speech condition,从 Gaussian noise 出发用 Euler ODE solver 采样,最后通过 Vocos vocoder 合成波形。

### 关键设计选择

#### 1. 为什么选择 Zipformer 而非 DiT/Transformer 作为 backbone?

[论文原文] Zipformer 有三个特性使其特别适合 flow matching backbone [§II-C]:

- **U-Net-like 多分辨率处理**: 5 个 stack 以 [1x, 2x, 4x, 2x, 1x] 下采样率运行,跨分辨率 bypass 连接。U-Net 结构被广泛认为是 diffusion/flow 模型的有效归纳偏置 [§II-C]
- **卷积模块**: CNN 擅长捕捉相邻 hidden state 的强相关性,这在语音任务中特别重要,补充 Transformer 的全局建模 [§II-C]
- **注意力权重复用**: 在单层内的两个 self-attention 模块和一个 non-linear attention (NLA) 模块之间共享 Q/K 投影,减少参数的同时增强计算效率 [§II-C]

[agent 解读] 这三个特性的组合效果是: U-Net 提供多尺度特征交互(类似 U-DiT 的思路),CNN 在 TTS 的局部频谱结构上很关键(Conformer 已证明这一点),attention weight reuse 在固定参数预算下提升了有效容量。Zipformer 虽然是为 ASR 设计的,但其核心假设(语音信号的多尺度性、局部相关性)同样成立于 TTS 的声学建模。

**消融验证** [Table V]:
- 去掉 convolution: WER 从 1.69 → 9.79 (5.8x 劣化)
- 去掉 downsampling + bypass: WER 1.69 → 6.35
- 仅去掉 bypass (保留 downsample): WER → 98.89, UTMOS → 1.25 (几乎崩溃)
- 去掉 NLA: 三项指标一致小幅下降
- 去掉 share attention weight: 性能相近但效率降低

[agent 解读] bypass connection 的崩溃性影响值得注意 — 说明在 U-Net 结构中,跨分辨率信息传递比下采样本身更关键。这与 F5-TTS 中 flat U-Net style linked Transformer 的思路一致,后者也保留了 bypass 但去掉了真正的下采样。

#### 2. Average Upsampling: 为什么均匀 duration 假设能 work?

[论文原文] E2-TTS 用 filler token 将 text 填充到与 speech 等长,让模型隐式学习对齐,但这导致对齐不准和收敛慢 [§II-D]。F5-TTS 加了 ConvNeXt 来 refine padded text condition。ZipVoice 采用更简单的方案: 假设每个 token 的 duration 均匀,即 d = floor(T/N) [§II-D, Eq.5]。

具体步骤: 将每个 text embedding 重复 d 次,从 N 扩展到 d*N;如果 T > d*N,剩余位置用 filler embedding 填充 [§II-D]。

[论文原文] "While this uniform-duration assumption is theoretically simplistic and has a considerable gap between real durations, it provides a reasonable initial text condition for the flow-matching model. Empirically, this straightforward strategy significantly improves alignment accuracy." [§II-D]

**消融验证** [Table IV]:
- 去掉 average upsample (回到 E2-TTS 的 filler padding): WER 从 1.69 → 20.19 (12x 劣化)
- 改用 ConvNeXt refinement (F5-TTS 方案): WER 1.69 → 15.49 (仍显著更差)

[agent 解读] Average upsampling 的有效性背后的逻辑是: flow matching 模型只需要一个"大致合理"的初始对齐作为 prior,模型自己会在训练中学习修正这个初始对齐。均匀分配虽然不精确,但比完全随机的 filler padding 提供了更强的位置先验。这类似于 CosyVoice 系列中 flow matching 也不需要精确 duration prediction 的思路。

#### 3. Flow Distillation: 如何同时减少 NFE 和消除 CFG 开销?

[论文原文] 核心思想: 用预训练的 teacher 模型做 2 步 CFG 推理得到目标 vector field,让 student 用 1 步匹配这个 target [§II-E]。

**具体流程** [§II-E, Eq.6-9]:
1. 用 teacher θT 从 xt 做两步推理(每步带 CFG): xt → xtmid → xtdest
2. 计算 teacher vector field: vT = (xtdest - xt) / (tdest - t)
3. Student 学习 1 步回归到 vT,且 CFG strength ω 作为 student 的一个 input (通过 Fourier embedding + linear layer 注入)
4. 两个 step size 和 ω 都动态采样,而非固定值 [§II-E]

**关键创新**: 将 CFG strength 编码为模型输入 [§II-E]。[论文原文] "Since we want the student model to benefit from the CFG inference while avoiding the additional model evaluation of CFG, we modify the student model to be conditioned on the CFG strength." 这样推理时只需 1 次前向 + 传入 ω 参数,而非标准 CFG 的 2 次前向。

**二阶段自蒸馏** [§II-E, Eq.10]: 第一阶段用固定 teacher;第二阶段用 student 的 EMA 版本作为 teacher,迭代优化。

**消融验证** [Table VI]:
| 方法 | 4 NFE WER | 4 NFE UTMOS |
|------|-----------|-------------|
| 无蒸馏 | 2.11 | 3.84 |
| Consistency Distillation | 1.97 | 3.30 |
| ReFlow | 2.56 | 3.92 |
| 本文 Flow Distillation | **1.68** | **4.22** |

[agent 解读] Flow distillation 的关键优势在于: (1) consistency distillation 在 4 NFE 时反而比无蒸馏差 (UTMOS 3.30 vs 3.84),因为 ZipVoice 没有显式 duration,1-2 步无法建立合理对齐; (2) ReFlow 虽然 UTMOS 提升但 WER 劣化; (3) 本文方法在 4 NFE 下 WER 和 UTMOS 都优于无蒸馏版本,说明 teacher 的 2 步 CFG 推理确实提供了比单步更好的训练信号。

### 训练策略

- **数据**: Emilia 100K 小时(多语言); LibriTTS 585 小时(英语,用于开发和消融) [§IV-A]
- **Flow matching 阶段**: Emilia 1M updates / LibriTTS 60K updates,batch size 4K/2K seconds [§IV-C]
- **Flow distillation 阶段**: Emilia 62K updates / LibriTTS 12K updates [§IV-C]
- **Text condition dropout**: 20% probability for CFG [§IV-C]
- **Mask ratio**: 70%-100% [§IV-C]
- **Text token**: phoneme (Emilia) / character (LibriTTS) [§IV-C]
- **Vocoder**: Vocos (预训练于 LibriTTS) [§IV-D]

**推理策略** [§II-F]:
- 合成 duration 按 token 长度比例估计: T_synthesis = T_prompt * |y_synthesis| / |y_prompt| [§II-F, Eq.11]
- Time-dependent CFG: 早期 NFE 仅 drop text condition,后期 drop text + audio condition [§II-F]

## 实验

| 指标 | ZipVoice (16NFE) | ZipVoice-Distill (4NFE) | F5-TTS (32NFE) | MaskGCT (32NFE) | E2-TTS (32NFE) | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIM-o ↑ | 0.668 | 0.657 | 0.655 | 0.691 | 0.700 | 0.690 | LibriSpeech-PC | [Table I] |
| WER ↓ | 1.64 | 1.51 | 1.89 | 2.26 | 2.49 | 1.87 | LibriSpeech-PC | [Table I] |
| UTMOS ↑ | 3.98 | 4.05 | 3.89 | 3.91 | 3.47 | 4.10 | LibriSpeech-PC | [Table I] |
| SIM-o ↑ | 0.697 | 0.679 | 0.664 | 0.713 | 0.706 | 0.734 | Seed-TTS test-en | [Table I] |
| WER ↓ | 1.70 | 1.64 | 1.85 | 2.88 | 2.32 | 2.14 | Seed-TTS test-en | [Table I] |
| SIM-o ↑ | 0.751 | 0.748 | 0.750 | 0.773 | 0.713 | 0.755 | Seed-TTS test-zh | [Table I] |
| WER ↓ | 1.40 | 1.39 | 1.53 | 2.40 | 1.91 | 1.25 | Seed-TTS test-zh | [Table I] |
| CMOS ↑ | 0.17 | 0.05 | - | -0.08 | -0.03 | 0 | - | [Table I] |
| SMOS ↑ | 3.94 | 3.84 | - | 4.10 | 3.76 | 3.36 | - | [Table I] |
| RTF (GPU) ↓ | 0.0557 | 0.0125 | 0.2958 | - | - | - | - | [Table II] |
| RTF (CPU) ↓ | 9.553 | 1.220 | 37.284 | - | - | - | - | [Table II] |

**关键发现**:
1. ZipVoice 在 WER 和 UTMOS 上超过所有 NAR baseline (包括参数量大 2.7x 的 F5-TTS 和 8.5x 的 MaskGCT),但 SIM-o 偏低 [Table I]
2. ZipVoice-Distill (4NFE) 的 RTF 比 F5-TTS (32NFE) 快 23.7x (GPU) / 32.6x (CPU) [Table II]
3. Flow distillation 后 WER/UTMOS 提升, SIM-o 轻微下降 [Table I]
4. 小数据集 (LibriTTS 585h) 上 ZipVoice 仍优于 F5-TTS [Table III]
5. ZipVoice-Distill (4NFE) CPU RTF 1.22,接近实时,有移动端部署潜力 [Table II]

## 局限性

1. **Speaker similarity 偏弱**: SIM-o 在所有 benchmark 上均低于 MaskGCT 和 E2-TTS,SMOS 4.10 (MaskGCT) vs 3.94 (ZipVoice)。压缩模型在说话人音色保持上存在明显 trade-off [Table I]
2. **Average upsampling 的理论瑕疵**: 均匀 duration 假设与真实发音差距大,仅靠 flow matching 隐式纠正。对于节奏变化大的语言/说话风格,可能有更大误差。论文未报告 per-phoneme alignment accuracy [§II-D]
3. **语言覆盖有限**: 仅在英语和中文上评估,未验证多语言泛化能力 [§IV-A]
4. **依赖预训练 vocoder**: 使用 LibriTTS 上训练的 Vocos,可能在 out-of-domain 语音上质量受限 [§IV-D]
5. **Flow distillation 的 teacher 依赖**: 蒸馏质量受限于 teacher 模型的 2 步推理质量;对于需要更多 NFE 才能收敛的场景,teacher signal 可能不够好 [§II-E]

## 点评

**优势**:
- 清晰的工程思路: 将 Zipformer 从 ASR 迁移到 TTS flow matching,且通过充分消融验证了 U-Net、CNN、attention weight reuse 三个组件各自的贡献
- Average upsampling 的极简设计令人印象深刻: 一个 floor(T/N) 公式就显著超过了 ConvNeXt refinement (WER 1.69 vs 15.49)
- Flow distillation 同时解决了 NFE 过多和 CFG 双倍开销两个问题,设计优雅
- 作者来自 Xiaomi + Daniel Povey (Kaldi 创始人),Zipformer 也是 Povey 团队的工作,跨任务迁移有说服力

**不足**:
- Speaker similarity 的下降是实际部署中的关键问题,论文未深入分析原因(是参数量不够还是 average upsampling 的对齐误差导致)
- 缺少与 CosyVoice 系列 (AR+Flow hybrid) 的公平对比,这类系统在 SIM 上显著更强
- 未讨论 streaming/实时生成场景下的适用性

**整体评价**: ZipVoice 在"小模型 + 快推理 + 不牺牲质量"这个三角中给出了目前最好的 trade-off。123M 参数 + 4NFE + RTF 1.22 CPU 的组合使得 SOTA 级 zero-shot TTS 首次具备了边缘设备部署的可能性。但 speaker similarity 的 trade-off 需要在实际应用中评估是否可接受。

## 可复用的 idea

1. **ASR backbone → TTS flow estimator**: Zipformer 的 U-Net + CNN + attention reuse 组合可以作为通用的 flow matching backbone 设计模式,不限于 TTS
2. **Average upsampling 作为 "good enough" 对齐初始化**: 任何需要 text-speech 长度对齐的模型都可以用 floor(T/N) 作为无需训练的初始对齐,让生成模型自己学习修正。这比 MFA 简单得多,且消融显示效果优于 ConvNeXt refinement
3. **CFG strength 作为 model input 消除推理翻倍**: 将 guidance strength 编码为条件输入而非在推理时做两次前向传播,理论上可以推广到任何使用 CFG 的 diffusion/flow 模型
4. **两阶段蒸馏 (固定 teacher → EMA self-distillation)**: 先用固定好的 teacher 做初始蒸馏,再切换为 student EMA 迭代优化,避免了 teacher 与 student 差距过大的问题
5. **Time-dependent CFG 策略**: 早期 drop text only,后期 drop text+audio,在 speaker similarity 和 intelligibility 之间取得更好平衡

---

> [!review] 自动审阅
> 审阅报告: [[_review/ZipVoice-review.yml]]
