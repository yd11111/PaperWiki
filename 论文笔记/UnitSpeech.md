---
type: paper
tier: deep
title: "UnitSpeech: Speaker-adaptive Speech Synthesis with Untranscribed Data"
arxiv_id: "2306.16083"
source: "Sources/UnitSpeech.pdf"
authors: [Heeseung Kim, Sungwon Kim, Jiheum Yeom, Sungroh Yoon]
year: 2023
venue: "Interspeech 2023 (arXiv: 2306.16083)"
tags: [speaker-adaptation, diffusion-model, self-supervised-unit, voice-conversion, HuBERT, fine-tuning, untranscribed-data, classifier-free-guidance]
concepts: ["[[SpeakerAdaptation]]", "[[DiffusionModel]]", "[[Classifier-FreeGuidance]]", "[[Self-SupervisedSpeechRepresentation]]", "[[SpeechFactorization]]", "[[VoiceCloningTaxonomy]]"]
models: ["[[模型库/HuBERT]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeechFactorization]]✓, [[SpeakerAdaptation]], [[DiffusionModel]], [[Classifier-FreeGuidance]], [[Self-SupervisedSpeechRepresentation]], [[VoiceCloningTaxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechFactorization]], [[SpeakerAdaptation]], [[DiffusionModel]], [[Classifier-FreeGuidance]], [[Self-SupervisedSpeechRepresentation]], [[VoiceCloningTaxonomy]] | 过滤: 5 页 pending-review | 未命中但可能相关: 无

**谱系定位**: UnitSpeech 属于 Speaker Adaptation 中的 "Untranscribed Speech" 子课题。在 KB 中,这一子课题此前的代表方法为 (1) Inoue et al. 的半监督 ASR+TTS pipeline,(2) Zhang et al. 的 VQ-VAE 离散语言单元,(3) AdaSpeech 2 的 mel encoder 替代 text encoder。UnitSpeech 的核心差异在于:用 HuBERT self-supervised unit 替代 mel-spectrogram 作为内容表示,结合 diffusion-based decoder 进行微调,而非 AdaSpeech 2 的确定性 feed-forward decoder。

**已有认知**: [[SpeakerAdaptation]] 页记载了 AdaSpeech 系列的演进: CLN 高效适应(v1) → 无转写适应(v2) → 自发语音(v3) → Zero-shot(v4)。UnitSpeech 的思路与 AdaSpeech 2 最相近(都用替代 encoder 绕开转写需求),但选择了完全不同的替代信号和 decoder 架构。[[SpeechFactorization]] 页(confirmed)指出 content-speaker disentanglement 的信息瓶颈方法是主流路线之一,UnitSpeech 通过 HuBERT 的离散化天然实现内容-说话人解耦。[[DiffusionModel]] 页记载 Grad-TTS 是 diffusion TTS 的开创性工作之一,UnitSpeech 直接以 multi-speaker Grad-TTS 为 backbone。

**创新判断**: 相对于 AdaSpeech 2(mel encoder + FastSpeech 2),UnitSpeech 的创新在于:(1) 利用 HuBERT units 的天然内容-说话人解耦特性,比直接用 mel-spectrogram 作为替代输入更干净;(2) 搭配 diffusion decoder 的强生成能力,降低对数据量的需求;(3) 适应后的模型同时支持 TTS 和 VC 两个任务。

## 速查

> [!summary] 速查
> - **一句话**: 用 HuBERT 离散单元替代文本转写,实现基于 diffusion 的 speaker adaptation,单条无转写音频即可微调,同时支持 TTS 和 VC
> - **路线**: 语音→HuBERT→K-means 离散化→unit encoder→diffusion decoder (DDPM)→mel→HiFi-GAN→波形; 适应时冻结 unit encoder,仅微调 decoder
> - **指标**: TTS: MOS 4.13, CER 1.75%, SECS 0.935 (LibriTTS); VC: MOS 4.26, CER 3.55%, SECS 0.923 (LibriTTS) [Table 1, Table 2]
> - **可借鉴**: (1) 用 self-supervised discrete units 作为通用内容表示,绕开转写依赖; (2) 数据集 mel 均值作为 CFG 的 unconditional embedding,免去额外训练; (3) 冻结 encoder + 仅微调 decoder 的分离式适应策略
> - **局限**: CER 1.75% 高于 text-conditioned Grad-TTS 的 0.75% [Table 1]; 每个目标说话人需单独微调 500 步; 未开源预训练权重(仅开源代码)

## 核心问题

UnitSpeech 要解决的核心问题是:**如何用最少量的无转写参考语音,将预训练多说话人 TTS 模型适应到目标说话人,并同时获得 TTS 和 VC 能力?**

现有方案的不足 [§1]:
1. Speaker embedding 方法(YourTTS 等 zero-shot):部署简单但 speaker similarity 较低
2. 基于微调的方法:需要转写数据(AdaSpeech 等),限制了可用数据范围
3. AdaSpeech 2:用 mel encoder 绕开转写,但确定性 feed-forward decoder 限制了生成质量,且需要较多参考数据
4. Guided-TTS 2:利用 diffusion + classifier guidance 实现无转写适应,但需要额外训练 unconditional model,训练成本高

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UnitSpeech 由三个阶段组成 [Fig 1]:

**阶段 1: 预训练多说话人 TTS 模型** [§2.1]
- 基于 multi-speaker Grad-TTS: text encoder + duration predictor + DDPM decoder
- 与原版 Grad-TTS 的区别: 通道数翻倍以适应多说话人建模 [§3.1.2]; 使用标准正态分布作为先验(而非 Grad-TTS 的 text-aligned 先验)[§2.1] [论文原文]
- Speaker embedding 由独立 speaker encoder(在 VoxCeleb2 上用 GE2E loss 训练)提取 [§3.1.2]
- Text encoder output c_y 被设计为 speaker-independent:训练时不向 text encoder 提供 speaker embedding,使 encoder loss L_enc = MSE(c_y, X_0) 迫使 c_y 只编码内容 [§2.1] [论文原文]

**阶段 2: Unit Encoder 训练** [§2.2]
- Unit encoder 架构与 text encoder 完全相同,但输入从音素序列换为 HuBERT 离散单元 [§2.2] [论文原文]
- Unit 提取: 语音→HuBERT→K-means 聚类(K=200)→离散 unit 序列→上采样到 mel 长度→压缩为 squeezed unit u + unit duration d_u [§2.2]
- 训练时冻结 DDPM decoder,仅训练 unit encoder [§2.2]
- 训练目标与预训练相同: L = L_grad + L_enc,只是 c_y 替换为 c_u(unit encoder 的 aligned output)[§2.2]
- 训练效果: c_u 被映射到与 c_y 相同的空间,使 unit encoder 可以无缝替代 text encoder [§2.2] [论文原文]

**阶段 3: Speaker Adaptation + 推理** [§2.3]
- 适应:从目标说话人的参考语音提取 unit u' 和 duration d_u',冻结 unit encoder,仅微调 DDPM decoder [§2.3]
- TTS:用 text encoder output c_y 作为条件,fine-tuned decoder 生成个性化语音 [§2.3]
- VC:从源语音提取 HuBERT unit,经 unit encoder 得到 c_u,fine-tuned decoder 生成目标说话人的声音 [§2.3]

### 关键设计选择

**1. 为什么用 HuBERT units 而非 mel-spectrogram(AdaSpeech 2)?**

HuBERT 的离散化表示通过聚类数 K 天然控制了信息粒度:设置合适的 K 值可以约束 unit 主要包含语音内容,去除说话人信息 [§2.2] [论文原文]。相比之下,mel-spectrogram 包含说话人信息,作为 encoder 输入时存在内容-说话人纠缠 [agent 解读]。这与 [[SpeechFactorization]] 中信息瓶颈方法的原理一致:离散化本身就是一种信息瓶颈。

**2. 为什么冻结 unit encoder,只微调 decoder?**

冻结 unit encoder 是为了最小化适应过程中的发音退化 [§2.3] [论文原文]。如果同时微调 encoder,少量的适应数据可能让 encoder 过拟合到目标说话人的发音模式 [agent 解读]。分离式策略:encoder 保持内容编码能力不变,decoder 学习目标说话人的声学特征。

**3. Classifier-free guidance 的实现细节**

UnitSpeech 使用一种简化的 CFG [§2.3]:unconditional embedding e_Φ 直接设为训练集 mel-spectrogram 的均值 c_mel,而非像其他工作那样额外训练 [§2.3] [论文原文]。引导公式 [Eq. 4]:

```
ŝ(X_t, t|c_c, e_S) = s(X_t, t|c_c, e_S) + γ · α_t
α_t = s(X_t, t|c_c, e_S) - s(X_t, t|c_mel, e_S)
```

这里 c_c 是 text 或 unit encoder 的 aligned output。γ 的选择:TTS 用 γ=1.0,VC 用 γ=1.5 [§3.1.3]。这种设计利用了 encoder loss 将 encoder 输出空间拉近 mel-spectrogram 的特性,使 mel 均值成为合理的"无内容条件"基准 [agent 解读]。

**4. 标准正态先验 vs Grad-TTS 的 text-aligned 先验**

原版 Grad-TTS 用 aligned text encoder output 定义 diffusion 先验分布,UnitSpeech 改用标准正态分布 [§2.1]。论文未明确解释原因 [agent 解读]:可能是因为需要同时兼容 text 和 unit 两种输入,统一使用标准正态先验避免了先验分布与特定 encoder 绑定的问题。

### 训练策略

- 预训练 TTS: 4× RTX 8000 GPU,1.4M iterations,Adam lr=1e-4,batch size 64 [§3.1.2]
- Unit encoder: 200K iterations [§3.1.2]
- Speaker adaptation: 500 steps(默认),Adam lr=2e-5,单卡 <1 min [§3.1.2]
- Unit 提取: HuBERT + K-means (K=200),使用 textless-lib [§3.1.2]
- 采样: N=50 steps [§3.1.3]

## 实验

| 指标 | UnitSpeech | Guided-TTS 2 | Guided-TTS 2 (zs) | YourTTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS (TTS) | 4.13±0.10 | 4.16±0.10 | 4.10±0.11 | 3.57±0.13 | LibriTTS | [Table 1] |
| CER (TTS) | 1.75% | 0.84% | 0.80% | 2.38% | LibriTTS | [Table 1] |
| SMOS (TTS) | 3.90±0.13 | 3.90±0.13 | 3.71±0.14 | 3.34±0.15 | LibriTTS | [Table 1] |
| SECS (TTS) | 0.935 | 0.937 | 0.873 | 0.866 | LibriTTS | [Table 1] |

| 指标 | UnitSpeech | DiffVC | YourTTS | BNE-PPG-VC | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS (VC) | 4.26±0.09 | 3.97±0.09 | 3.88±0.10 | 3.86±0.10 | LibriTTS | [Table 2] |
| CER (VC) | 3.55% | 3.67% | 2.20% | 1.37% | LibriTTS | [Table 2] |
| SMOS (VC) | 3.83±0.13 | 3.69±0.13 | 3.56±0.12 | 3.50±0.14 | LibriTTS | [Table 2] |
| SECS (VC) | 0.923 | 0.909 | 0.763 | 0.851 | LibriTTS | [Table 2] |

**Ablation 要点** [Table 3]:
- **Unit 聚类数 K**: K=200 为最佳平衡; K 增大改善 VC 发音精度(K=500 CER 3.80% vs K=50 CER 12.64%)但对 TTS 影响不大 [Table 3]
- **微调步数**: 500 steps 为最佳权衡; speaker similarity 逐步上升并在 500 步收敛; >2000 步发音准确率下降 [Table 3]
- **参考语音长度**: 更长的参考语音同时改善发音和 speaker similarity; 5 秒即可获得可用效果 [Table 3]
- **CFG gradient scale γ**: γ=1.0(TTS)和 γ=1.5(VC)在发音改善和 speaker similarity 下降之间取得最佳平衡 [Table 3]

## 局限性

1. **发音准确率不及 text-conditioned 模型**: TTS CER 1.75% 显著高于 Grad-TTS 自身的 0.75%,VC CER 3.55% 高于 BNE-PPG-VC 的 1.37% [Table 1, Table 2]。这是用 unit 替代 text 的固有代价——unit 的内容表示精度低于文本 [agent 解读]
2. **每人需单独微调**: 虽然 500 步仅需不到 1 分钟,但无法像 zero-shot 方法那样即时部署。在大规模应用中,这意味着需要为每个新说话人存储独立的 decoder checkpoint [agent 解读]
3. **仅在 LibriTTS 上评估**: 未在 VCTK 等其他标准数据集上报告,跨域泛化能力未知 [agent 解读]
4. **vocoder 固定为 HiFi-GAN**: 使用预训练 universal HiFi-GAN 作为 vocoder [§3.1.3],未探索端到端或更先进的 vocoder 方案
5. **与 LLM-era 方法的差距**: 2023 年后 VALL-E 等 codec LM 方法在 zero-shot speaker similarity 上大幅超越微调方法,UnitSpeech 的微调范式在当前已不是主流路线 [agent 解读,基于 [[VoiceCloningTaxonomy]] 演进趋势]

## 点评

UnitSpeech 的核心贡献是将 HuBERT discrete units 引入 diffusion-based speaker adaptation,解决了 AdaSpeech 2 的两个关键缺陷:(1) mel-spectrogram 作为替代输入的内容-说话人纠缠问题,(2) 确定性 decoder 的生成能力瓶颈。这个组合思路——self-supervised discrete units 提供干净的内容表示 + diffusion decoder 提供强大的生成能力——是优雅的。

技术上最巧妙的设计是 unit encoder 与 text encoder 共享输出空间(通过相同的训练目标和 encoder loss 实现),使得适应后的 decoder 可以无缝切换 TTS 和 VC 两种模式。这种"一次适应,多任务通用"的设计在当时是有新意的。

然而从当前视角看,这篇工作的历史定位是 speaker adaptation 路线的成熟之作,而非开启新范式。Codec LM 方法(VALL-E, 2023)几乎同时出现,以 in-context learning 彻底绕开了微调需求,重新定义了 zero-shot TTS 的能力上限。UnitSpeech 所代表的"替代 encoder + diffusion 微调"范式,在 VALL-E 之后已不再是活跃的研究方向。

值得注意的是,UnitSpeech 中"用 dataset mel 均值作为 CFG unconditional embedding"的简化设计,避免了额外训练 unconditional model 的开销,这个具体 trick 在后续工作中被证明是实用的。

## 可复用的 idea

1. **Self-supervised units 作为通用内容接口**: HuBERT units 的聚类数 K 可控制内容粒度,K 越大保留越多细节(对 VC 有利),K 越小越"干净"(对 TTS 有利)。这种通过离散化实现信息瓶颈的思路适用于任何需要 content-speaker 解耦的场景
2. **冻结 encoder + 仅微调 decoder 的分离式适应**: 防止少量适应数据导致 encoder 过拟合,保持内容编码能力不变。这个策略可推广到 flow-matching/LM-based TTS 的微调场景
3. **数据集 mel 均值作为 CFG unconditional embedding**: 利用 encoder loss 将 encoder 输出空间拉近 mel-spectrogram 这一特性,无需额外训练即获得合理的 unconditional 基准。前提是系统存在将 encoder output 向 mel space 对齐的 loss
4. **Unit duration 作为显式对齐机制**: 从 HuBERT 输出中同时提取 unit sequence 和 duration,避免了在适应阶段依赖 MAS 进行文本-语音对齐

检索命中: [[SpeechFactorization]], [[SpeakerAdaptation]], [[DiffusionModel]], [[Classifier-FreeGuidance]], [[Self-SupervisedSpeechRepresentation]], [[VoiceCloningTaxonomy]] | 过滤: 5 页 pending-review | 未命中但可能相关: 无

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节4个设计选择均有因果解释,速查可借鉴具体 |
> | 可信赖 | pass | 数字标注覆盖率>85%;初稿VC表SMOS/SECS错位已修正 |
> | 可区分 | pass | [论文原文]/[agent 解读]标注覆盖率~85% |
> | 可定位 | pass | KB背景精准对标AdaSpeech 2和Guided-TTS 2 |
> | 不污染 | pass | 未新建概念页,frontmatter挂接合理 |
> 
> Issues: 3 (high: 1 fixed, medium: 0, low: 2)
> 详见 `_review/UnitSpeech-review.yml`
