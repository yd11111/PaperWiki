---
type: paper
tier: deep
title: "Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech"
arxiv_id: "2507.20140"
source: "Sources/2507.20140.pdf"
authors: [Taesoo Kim, Jinju Kim, Dongchan Kim, Jong Hwan Ko, Gyeong-Moon Park]
year: 2025
venue: "ICML 2025"
tags: [machine-unlearning, voice-privacy, zero-shot-TTS, speaker-identity, flow-matching, GDPR, safety]
concepts: ["[[Conditional Flow Matching]]", "[[Speaker Embedding]]", "[[Speaker Verification]]", "[[Classifier-Free Guidance]]", "[[Anti-spoofing and Deepfake Detection]]", "[[Voice Cloning Taxonomy]]"]
models: []
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页: [[Conditional Flow Matching]], [[Speaker Embedding]], [[Speaker Verification]], [[Classifier-Free Guidance]], [[Anti-spoofing and Deepfake Detection]], [[Voice Cloning Taxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文开辟了 ZS-TTS 中的 **speaker identity unlearning** 新方向,是 [[Anti-spoofing and Deepfake Detection]] 中从"被动检测/主动扰动"到"模型级遗忘"的第三条路线。KB 中已有的防护方案包括: (1) 被动检测(ASVspoof 系列)、(2) 主动扰动(SafeSpeech: 在音频中嵌入不可感知扰动)、(3) 水印溯源(TraceableSpeech)。本文提出的 machine unlearning 是一种截然不同的防护范式 -- 直接修改模型权重使其丧失复制特定说话人的能力,属于模型层面的干预。
>
> **已有认知**: [[Conditional Flow Matching]] 页(confirmed)记录了 VoiceBox 使用的 CFM 框架,本文正是在 VoiceBox 上实施 unlearning。[[Speaker Embedding]] 页(confirmed)记录了 speaker embedding 作为 ZS-TTS 中说话人身份载体的角色,理解这一表示对理解本文"为什么无法简单过滤训练数据"至关重要 -- ZS-TTS 通过 in-context learning 泛化到未见说话人,因此 exact unlearning(去除训练数据重训)无效。[[Speaker Verification]] 页[待确认]记录了 WavLM-TDCNN 等 speaker encoder 用于 SIM 计算的标准做法,本文使用同一体系评估 unlearning 效果并设计了新指标 spk-ZRF。[[Classifier-Free Guidance]] 页[待确认]记录了 CFG 的工作原理,本文利用 VoiceBox 的 CFG 推理公式中的无条件分量 v_t(w;theta) 来验证 unlearning 效果。
>
> **创新判断**: 相对于 KB 中记录的防护方法,本文的创新在于: (1) 首次在 ZS-TTS 中提出 machine unlearning 问题定义; (2) SafeSpeech 需要在数据端操作(用户主动嵌入扰动),而本文在模型端操作(服务商修改模型权重),两者互补; (3) 提出的 spk-ZRF 指标填补了 unlearning 评估中"随机性度量"的空白,KB 中 [[Speaker Verification]] 页现有的 SECS/SIM 指标只衡量相似度而无法衡量遗忘后的随机程度。
>
> 检索命中: [[Conditional Flow Matching]]✓, [[Speaker Embedding]]✓, [[Speaker Verification]][待确认], [[Classifier-Free Guidance]][待确认], [[Anti-spoofing and Deepfake Detection]][待确认], [[Voice Cloning Taxonomy]][待确认] | 过滤: 无 | 未命中但可能相关: [[Masked Generative Modeling]](VoiceBox 的 mask-infilling 范式,但非本文核心)

## 速查

> [!summary] 速查
> - **一句话**: 首次提出 ZS-TTS 中的 speaker identity unlearning 问题,通过 Teacher-Guided Unlearning (TGU) 使 VoiceBox 在遇到 forget speaker 的 audio prompt 时生成随机音色语音,同时保持对 remain speaker 的克隆质量
> - **路线**: pre-trained VoiceBox (θ) → 对 forget speaker 的 audio prompt: teacher θ 仅以 text y 为条件生成随机音色目标 x̄=θ(y) → student θ⁻ 训练拟合 x̄ 而非原说话人 → 对 remain speaker 保持原 CFM loss → 联合训练 (λ=0.2 forget, 0.8 remain)
> - **指标**: TGU SIM-F 0.169 (接近不同说话人的均值 0.09) [Table 1]; SIM-R 0.631 (仅比原模型 0.649 降 2.8%) [Table 1]; WER-R 2.5 vs 原模型 2.1 [Table 1]; spk-ZRF-F 0.871 (最高随机性) [Table 1]; SMOS-F 1.28 (几乎不像原说话人) [Table 4]; CMOS-R -0.02 (质量几乎无损) [Table 4]
> - **可借鉴**: 利用 pre-trained model 的无条件生成能力作为 unlearning 的"随机目标生成器" -- teacher θ 不接收 audio context 时自然产生随机说话人,省去构造 aligned random target 的难题;spk-ZRF 指标设计思路(用 JSD 度量生成分布与随机分布的距离)可迁移到其他 unlearning 场景的评估
> - **局限**: 仅在 VoiceBox (CFM-based NAR) 上验证,未测试 AR 架构 (VALL-E) 或 LLM-based TTS; 恢复实验 [Table 12] 显示 15 min 数据 fine-tune 可使 SIM-F 回升至 0.735(但 remain 性能崩溃); 未讨论声音相似但非 forget speaker 的潜在误伤(仅 Appendix H 简要分析)

## 核心问题

ZS-TTS 模型可以从 3 秒 audio prompt 复制任意说话人的声音,这引发了严重的隐私和伦理问题(GDPR "被遗忘权")。然而,现有的隐私保护方案存在根本局限 [§1]:

1. **Exact Unlearning(从训练集移除并重训)对 ZS-TTS 无效** -- ZS-TTS 通过 in-context learning 泛化到未见说话人,即使训练集中不含 forget speaker 的数据,模型仍能零样本复制其声音 [论文原文]
2. **简单匿名化不够** -- 高级攻击(model inversion、voice re-synthesis、targeted fine-tuning)可从匿名化嵌入中恢复身份信息 [论文原文]
3. **需要"随机性"而非"固定替代"** -- 如果 unlearned 模型对 forget speaker 总是输出固定的替代音色(如只改 pitch),攻击者可通过逆操作恢复原音 [论文原文]

因此,核心目标是: 使模型在遇到 forget speaker 的 audio prompt 时,输出 **随机且不可追溯** 的音色,同时对其他说话人保持原有克隆质量。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文在 VoiceBox (Le et al., 2024) 上实施 unlearning。VoiceBox 是一个基于 [[Conditional Flow Matching]] 的 NAR ZS-TTS 模型,通过 mask-infilling 范式从文本 y 和 audio context x_ctx 生成目标语音。

**核心思路**: 利用 pre-trained VoiceBox 自身的无条件生成能力(仅给 text 不给 audio prompt)作为"随机说话人生成器",为 unlearning 提供训练目标 [论文原文][§4.2]。

### 关键设计选择

#### 1. 为什么不用传统 MU 方法?

论文逐一排除了四种传统 machine unlearning 方法 [§5.2]:

| 方法 | 机制 | 失败原因 | 证据 |
|------|------|----------|------|
| Exact Unlearning | 移除 forget 数据重训 | ZS-TTS 的 in-context learning 可泛化到未见说话人 | SIM-F=0.687,几乎无变化 [Table 1] |
| Fine-Tuning | 仅用 remain 数据继续训练 | 同上,模型不需要见过 forget speaker 的数据就能克隆 | SIM-F=0.675,几乎无变化 [Table 1] |
| Negative Gradient | 对 forget 数据反转梯度 | 梯度爆炸导致模型整体性能崩溃 | WER-R=6.1, SIM-R=0.437 [Table 1] |
| KL Divergence | 最大化 teacher-student 在 forget 上的 KL | 模型学会对 forget 返回不可听的噪声而非不同声音 | WER-F=47.2(内容完全损坏) [Table 1] |

**为什么 NG 和 KL 会失败**: 语音中的 speaker style 与 linguistic content 在预训练过程中是纠缠的(entangled),这些方法无法选择性地只惩罚 speaker identity 而不破坏语音内容生成能力 [论文原文][§5.2]。[agent 解读] 这是 ZS-TTS unlearning 区别于图像/文本 unlearning 的核心难点 -- 在 diffusion image model 中"概念"和"画质"较容易解耦,但在语音中"谁在说"和"说了什么"深度纠缠。

#### 2. Sample-Guided Unlearning (SGU): 简单但有限

SGU 的思路 [§4.1]: 对 forget speaker 的语音 x_f,随机选一个 remain speaker 的语音 x_r,将两者拼接作为训练样本,mask x_r 部分让模型预测。这样模型在遇到 forget speaker 的 prompt 时,学会生成 remain speaker 的声音。

**问题**: VoiceBox 的 mask-infilling 依赖前后 audio context 做预测。拼接后只能 mask x_r 的全部(不能选择性 mask 中间),导致模型只有单侧 context,严重限制生成质量 [论文原文][§4.1]。如果在拼接语音中间 mask,两个说话人的 tempo/rhythm/style 不匹配会导致模型学到不自然的生成模式 [论文原文]。

结果: SGU 的 SIM-R 仅 0.523,比原模型下降 21%,remain speaker 的克隆质量严重受损 [Table 1]。

#### 3. Teacher-Guided Unlearning (TGU): 核心贡献

TGU 的关键洞察: VoiceBox 在仅给文本 y 而不给 audio context 时(即 x_ctx = ∅),会根据随机初始化的高斯噪声 x_0 生成不同的随机音色语音 [论文原文][§4.2]。

**具体流程** [§4.2, Figure 2(c)]:

1. 输入: forget speaker 的 (x_f, y)
2. Teacher θ (frozen pre-trained model) 仅用 y 生成 x̄ = θ(y),此时 x̄ 是一个说正确文本内容但音色随机的语音
3. Student θ⁻ 训练目标: 当给定 x_f 和 y 时,生成的语音应接近 x̄ 而非 x_f 的音色

**Forget loss** [Eq. 7]:
$$L_{\text{CFM-forget}}(\theta^-) = \mathbb{E}_{t,q(x_1),p_t(x_f|x_1)} \|m \odot u_t(x|\bar{x}) - v_t(w_f, y, x^f_{\text{ctx}}; \theta^-)\|^2$$

其中 x̄ = θ(y) 替代了原来的 ground truth x_1。

**Remain loss** [Eq. 8]: 对 remain speaker 使用标准 CFM loss,保持原有性能。

**总 loss** [Eq. 9]: L_total = λ·L_CFM-remain + (1-λ)·L_CFM-forget, λ=0.2。

**为什么 TGU 比 SGU 好**: [agent 解读] TGU 的关键优势在于: (1) 不需要拼接不同说话人的语音,避免了 tempo/rhythm 不匹配问题; (2) teacher 生成的 x̄ 与 forget speaker 的 y 天然 frame-wise 对齐(因为用同一文本生成),解决了 aligned random pair 的获取难题; (3) 每次训练 x_0 不同,x̄ 的音色也不同,自然引入了随机性。

### 训练策略

- 在每个 mini-batch 中,forget samples 以 20% 概率被选中 [Appendix B.1]
- TGU 训练 145K 步(实验 1)或 10K 步(实验 2,仅为预训练步数的 2-7%) [Appendix B.1]
- 批次大小: 75 秒音频 / batch
- 优化器: Adam, 峰值学习率 1e-4, 线性 warmup 5K 步后 decay
- 推理使用 CFG (α=0.7) + midpoint ODE solver (NFE=32)

### 新指标: spk-ZRF

**动机**: 传统 MU 评估指标(completeness、JS divergence)只比较 forget/remain 的性能差距,但模型对 forget set 展示一致模式不代表成功 unlearning,因为一致模式可被利用来逆向工程 [论文原文][§4.3]。

**设计** [§4.3, Eq. 10-12]:
1. 对每个 forget 样本 (x_s, y_i),生成: θ⁻(x_s, y_i) 和 θ(y_i)(后者为随机说话人参考分布)
2. 分别提取 speaker embedding(WavLM-TDCNN)
3. 对 embedding 做 softmax 转为概率分布
4. 计算两分布间的 Jensen-Shannon Divergence
5. spk-ZRF = 1 - mean(JSD),越接近 1 说明 unlearned model 的行为越接近随机

[agent 解读] spk-ZRF 本质上衡量的是"unlearned 模型对 forget speaker 的输出分布"与"完全随机说话人的输出分布"之间的匹配程度。这比单纯看 SIM 低更有信息量 -- SIM 低可能只是模型输出噪声(如 KL 方法),而 spk-ZRF 高才说明模型在生成正常语音的同时实现了 speaker identity 的随机化。

## 实验

| 指标 | TGU | SGU | NG | KL | Original | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER-R ↓ | 2.5 | 2.6 | 6.1 | 5.2 | 2.1 | LibriSpeech test-clean | [Table 1] |
| SIM-R ↑ | 0.631 | 0.523 | 0.437 | 0.408 | 0.649 | LibriSpeech test-clean | [Table 1] |
| WER-F ↓ | 2.4 | 2.5 | 5.0 | 47.2 | 2.1 | Forget eval set | [Table 1] |
| SIM-F ↓ | 0.169 | 0.194 | 0.402 | 0.179 | 0.708 | Forget eval set | [Table 1] |
| spk-ZRF-F ↑ | 0.871 | 0.866 | 0.842 | 0.810 | 0.846 | Forget eval set | [Table 1] |
| CMOS-R ↑ | -0.02±0.19 | -0.15±0.27 | - | - | 0.00 | LibriSpeech test-clean | [Table 4] |
| SMOS-R ↑ | 4.67±0.26 | 3.12±0.83 | - | - | 4.47±0.38 | LibriSpeech test-clean | [Table 4] |
| SMOS-F ↓ | 1.28±0.24 | 1.45±0.31 | - | - | 4.44±0.36 | Forget eval set | [Table 4] |

**关键发现**:

1. **TGU 在所有维度上达到最佳平衡** [Table 1]: SIM-F=0.169 (接近不同说话人的自然 SIM 均值 0.09 [Appendix C]),同时 SIM-R=0.631(仅降 2.8%),WER 几乎不变
2. **Exact Unlearning 和 Fine-Tuning 对 ZS-TTS 无效** [Table 1]: SIM-F 分别为 0.687 和 0.675,模型仍然能克隆 forget speaker
3. **NG 和 KL 导致模型整体崩溃** [Table 1]: NG 的 WER-R=6.1; KL 的 WER-F=47.2(输出变成不可听的噪声)
4. **可扩展性** [Table 2]: TGU 从 k=1 扩展到 k=10 个 forget speaker 后,SIM-R 几乎不变(0.624→0.631),而 SGU 的 SIM-R 从 0.586 降至 0.523
5. **OOD unlearning** [Table 3]: TGU 对训练集外的说话人也有效(SIM-F=0.186),证明 unlearning 不依赖 forget speaker 是否在训练集中
6. **恢复攻击** [Table 12]: 用 15 min forget speaker 数据 fine-tune 后,虽然 SIM-F 回升到 0.735,但 SIM-R 从 0.631 崩溃到 0.303,WER-R 从 2.5 升到 4.23 -- 本质上退化为单说话人 TTS 而非恢复原 ZS-TTS 能力

## 局限性

1. **仅在 VoiceBox 上验证**: 所有实验基于 CFM-based NAR 架构,未测试 AR 模型(VALL-E)、LLM-based TTS(CosyVoice)或 diffusion-based TTS(NaturalSpeech 2),这些架构的 unlearning 行为可能不同 [agent 解读]
2. **恢复攻击部分有效**: 15 min 数据 fine-tune 可使 SIM-F 回升到 0.735 [Table 12],虽然代价是 remain 性能崩溃,但攻击者可能只需要 forget speaker 的语音而不关心其他说话人 [agent 解读]
3. **相似声音的潜在误伤**: Appendix H 表明 TGU 对与 forget speaker 声音相似的 remain speaker 有弱正相关(Pearson r=0.14, p=0.0003),虽然统计上是弱相关,但在高相似度端可能存在风险 [论文原文]
4. **多样性轻微下降**: Diverse speech sampling FSD 从 170.2 升至 177.8 [Table 11],transient noise removal SIM 从 0.666 降至 0.641 [Table 10],说明 unlearning 对通用任务有微弱影响 [论文原文]
5. **forget set 规模限制**: 最多测试了 10 个 forget speaker,未探索大规模(如数百人)unlearning 的可行性 [agent 解读]
6. **无增量 unlearning 讨论**: 未探索连续接收 unlearning 请求的场景(每次新请求是否需要从头训练) [agent 解读]

## 点评

本文的核心贡献在于正式定义了 ZS-TTS 中的 speaker identity unlearning 问题并给出了首个可行方案。TGU 的设计展现了对 ZS-TTS 本质的深刻理解 -- 传统 MU 方法之所以失败,是因为 ZS-TTS 的核心能力(in-context generalization)恰恰使得"遗忘训练数据"这一传统思路不再适用。利用 pre-trained model 的无条件生成作为"随机目标"是一个优雅且实用的解决方案。

**与 KB 已有工作的互补关系**: 与 SafeSpeech(数据端防护)相比,TGU 在模型端操作,两者可叠加使用 -- SafeSpeech 保护用户的音频数据不被学到,TGU 则从模型中移除已有的 speaker 知识。与 TraceableSpeech(事后溯源)相比,TGU 是事前预防。三者构成了 ZS-TTS 安全的完整防线: 预防(unlearning) + 防护(perturbation) + 溯源(watermark)。

**值得关注的弱点**: (1) 恢复攻击的讨论不够充分 -- 虽然全模型性能会崩溃,但 attacker 只需要一个 speaker 的 fine-tune 并不困难; (2) 缺少对 AR 架构的实验,而 VALL-E/CosyVoice 等 AR 系统才是当前 ZS-TTS 的主流; (3) spk-ZRF 指标虽有创意,但其安全含义(多高才算"安全遗忘")缺乏理论分析。

发表于 ICML 2025,是该方向的开创性工作,为后续研究奠定了问题框架和 baseline。

## 可复用的 idea

1. **"用模型的无条件输出作为遗忘目标"**: 任何具有 conditional/unconditional 模式的生成模型都可借鉴 -- unconditional 输出天然提供了"不含待遗忘条件"的随机参考分布,省去构造 aligned negative target 的困难
2. **spk-ZRF 指标框架**: 用 JSD(model output distribution || random output distribution) 衡量遗忘程度的思路可迁移到其他模态(如图像中衡量概念遗忘的随机性)
3. **ZS-TTS 安全三层防线思路**: unlearning(模型端) + perturbation(数据端) + tracing(输出端) 构成完整安全栈

---

检索命中: [[Conditional Flow Matching]]✓, [[Speaker Embedding]]✓, [[Speaker Verification]][待确认], [[Classifier-Free Guidance]][待确认], [[Anti-spoofing and Deepfake Detection]][待确认], [[Voice Cloning Taxonomy]][待确认] | 过滤: 无 | 未命中但可能相关: [[Masked Generative Modeling]]
