---
type: paper
tier: deep
title: "Efficient Emotion and Speaker Adaptation in LLM-Based TTS via Characteristic-Specific Partial Fine-Tuning"
arxiv_id: "2501.14273"
source: "Sources/CSP-FT.pdf"
authors: [Tianrui Wang, Meng Ge, Cheng Gong, Chunyu Qiang, Haoyu Wang, Zikang Huang, Yu Jiang, Ye Ni, Yuheng Lu, Xiaobao Wang, Engsiong Chng, Xie Chen, Longbiao Wang, Jianwu Dang]
year: 2026
venue: "arXiv (v2, Mar 2026)"
tags: [TTS, domain-adaptation, fine-tuning, parameter-efficient, codec-language-model, emotion, speaker-adaptation, catastrophic-forgetting]
concepts: ["[[CodecLanguageModel]]", "[[SpeakerAdaptation]]", "[[EmotionControlinTTS]]", "[[LLM-basedTTS]]", "[[SpeakerEmbedding]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]", "[[模型库/VITS|VITS]]", "[[模型库/EnCodec|EnCodec]]", "[[模型库/HuBERT|HuBERT]]", "[[模型库/WavLM|WavLM]]", "[[模型库/Whisper|Whisper]]", "[[模型库/SenseVoice|SenseVoice]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[模型库/CosyVoice|CosyVoice]], [[LLM-basedTTS]], [[SpeakerEmbedding]]; 3 个待确认实体页: [[SpeakerAdaptation]] [待确认], [[CodecLanguageModel]] [待确认], [[EmotionControlinTTS]] [待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文处于 [[SpeakerAdaptation]] 的演进线末端 — 从传统 CLN/adapter 参数高效方法,到本文提出的基于层贡献分析的选择性微调。Speaker Adaptation 概念页记录的演进路线到 "In-context learning 逐渐取代微调 (VALL-E era, 2023-)" 为止,但本文指出 in-context learning 的 zero-shot 能力在 unseen 情感和说话人上仍不稳定,因此目标数据可用时仍需微调。本文的方法学定位是: 在 full fine-tuning (全参数,遗忘严重) 和 LoRA/PEFT (轻量但盲目) 之间找到一条中间路线 — 通过任务驱动的层贡献分析来选择性微调。
>
> **已有认知**: [[CodecLanguageModel]] 页描述了 codec LM 的核心架构 (autoregressive Transformer on discrete tokens),本文直接在此架构上操作。[[EmotionControlinTTS]] 页记录了情感控制从 embedding 到 DPO 的演进,但缺少从微调层选择角度的工作。[[SpeakerEmbedding]] 页记录了 x-vector/ECAPA-TDNN 在评估中的角色,本文使用 Resemblyzer 和 Emotion2vec 进行评估。[[模型库/CosyVoice|CosyVoice]] 页详细描述了其 S3 tokenizer + LLM + OT-CFM 架构,本文将其作为四个实验平台之一。
>
> **创新判断**: 相比 Speaker Adaptation 页记录的方法 (CLN tuning、module freezing、residual adapters、structured pruning),本文的创新在于用下游任务 (情感识别 + 说话人识别) 的 weighted-sum 分析来动态确定哪些层最相关,而非凭经验选择或均匀处理。这是一种 task-driven layer selection 策略,在概念库中尚无对应条目。
>
> 检索命中: [[SpeakerAdaptation]]✓, [[CodecLanguageModel]]✓, [[EmotionControlinTTS]]✓, [[模型库/CosyVoice|CosyVoice]]✓, [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓ | 过滤: 前三者为 pending-review | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过 weighted-sum 分析 Transformer 各层对情感/说话人控制的贡献,仅微调贡献最高和最低的两层,实现 ~8% 参数、~2x 加速下的高效域适应且显著缓解灾难性遗忘
> - **路线**: 预训练 codec LM → Stage 1: 冻结模型,用 weighted-sum + 下游分类器分析各层贡献 → 选 highest + lowest 两层 → Stage 2: 仅微调这两层做目标域 TTS 适应
> - **指标**: Fun-CosyVoice3.0 上 SS 94.8% / ERS 96.8% / WER 3.8% (vs Full FT: 94.5% / 97.0% / 12.1%), 11 个英语情感数据集 (244h); 训练速度 1.91-2.62x 加速 [Table 2, Fig 4]
> - **可借鉴**: 用下游任务的 layer-wise weighted-sum 分析来指导选择性微调,且分析结果跨数据集/跨语言可迁移 — 分析做一次,适应做多次
> - **局限**: 仅在 4 个 codec LM 上验证,未在 diffusion-based 或 NAR 架构上测试; 层选择基于任务权重平均,对情感/说话人权重分布不均的模型可能次优 (CosyVoice 案例 [Table 6])

## 核心问题

开源的 codec language TTS 模型 (如 CosyVoice、VALL-E X) 具有 zero-shot 情感和说话人克隆能力,但在 unseen 域上保真度和发音清晰度下降 [§1]。直接 full fine-tuning 存在两个问题: (1) 计算资源消耗大; (2) 在有限目标域数据上导致灾难性遗忘,表现为 WER 急剧上升 [§1, Table 2]。LoRA 等 PEFT 方法虽然减少参数,但盲目在所有层插入 adapter,忽略了不同层对情感/说话人控制的差异贡献,且引入额外计算开销 [§4.2.2]。

**核心问题**: 如何在有限目标域数据上高效适应 codec LM 的情感和说话人表达,同时最小化灾难性遗忘?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CSP-FT 分两个阶段 [§2.1, Fig 2]:

**Stage 1: Characteristic-Specific Analysis (特征分析)**
1. 将预训练 codec LM 冻结,作为 causal encoder 使用 [§2.2]
2. 在 N 层 Transformer 的每层输出上施加 layer normalization,然后用两组可学习权重 $W_e, W_s$ 通过 softmax weighted sum 聚合各层表示 [Eq. 2]
3. 聚合后的表示经 Conv (3 层 1D Conv + ReLU) + ASP (attentive statistics pooling) 转为 utterance-level 表示
4. 分别用 cross-entropy loss 训练情感识别和说话人识别两个分类任务
5. 训练完成后,$W_e$ 和 $W_s$ 反映各层对情感和说话人控制的贡献

**Stage 2: Characteristic-Specific Partial Fine-Tuning (选择性微调)**
1. 计算 $W_m = (W_e + W_s) / 2$ [Eq. 3]
2. 选择 $W_m$ 中权重最高和最低的两层
3. 仅微调这两层,冻结其余所有参数,在目标域 TTS 数据上做 next-token prediction 训练

### 关键设计选择

**为什么选最高 + 最低两层,而非仅选最高?**

[论文原文] 最高权重层已捕获最多情感/说话人信息,微调它是为了最大化利用其已有能力。最低权重层包含最少相关信息,微调它是为了增强其贡献潜力 (greatest potential for improvement)。两者互补: 最高层提供强适应能力但带来中等遗忘,最低层带来最小遗忘但仅中等适应能力,联合微调实现优秀适应 + 最小遗忘 [§2.3, §4.2.3]。

[agent 解读] 这个设计直觉可以类比正则化: 最高层负责"学新知识",最低层负责"稳住旧知识"。仅微调最高层虽然适应性强但 WER 恶化明显 (如 VALLE-X: WER 10.0% vs CSP-FT 7.0% [Table 2]);仅微调最低层适应性不足 (SS 和 ERS 较低)。两层组合本质上是在 adaptation-preservation tradeoff 上找到一个更优的 Pareto 点。

**为什么用下游分类任务而非直接分析激活?**

[论文原文] 各层对情感/说话人控制的贡献主要体现在其表示中包含的相关信息量。用分类任务训练 weighted sum 是一种间接但有效的度量方式 [§2.2]。

[agent 解读] 直接分析激活统计量 (如 CKA、probing) 也可以揭示层的功能分化,但 weighted-sum + 分类器的方式更直接地与目标任务对齐,且产出的权重可直接指导层选择。

**为什么情感和说话人权重取平均而非分别处理?**

[论文原文] 目标是同时学习情感和说话人控制,取平均权重进行统一选择 [§2.3]。

[agent 解读] 这是一个简化假设。Table 6 的 CosyVoice 案例表明,当两种权重分布不均时,平均可能不是最优选择 — 选第二大权重层反而略优于选最大权重层。但对其他三个模型,取平均是有效的。

**为什么 characteristic-specific weights 可以跨数据集迁移?**

[论文原文] 因为情感状态和说话人身份是 utterance-level 特征,理论上与语音内容无关 [§2.4, 引用 Du et al. 2022]。因此在一个带情感/说话人标注的开源数据集上做一次 Stage 1 分析,得到的层权重可直接用于其他目标域,无需重复分析。

### 训练策略

**Stage 1 (分析阶段)** [§3.2.2]:
- 4 个 TTS 模型完全冻结
- 仅训练层权重 $W$ 和下游分类模块
- 多任务学习 (情感 + 说话人)
- 244 小时英语情感数据,75 epochs,单张 3090 GPU
- Adam optimizer, lr warmup 0→5e-4 (前 8%) 后 decay

**Stage 2 (微调阶段)** [§3.2.3]:
- 10 epochs
- 峰值学习率设为官方预训练值的 ~5% (如 CosyVoice: 1e-4)
- 线性 warmup (前 8%) + 衰减
- 仅更新选定的两层参数

## 实验

### 主要结果: 英语情感域适应 [Table 2]

| 指标 | CSP-FT (Fun-CosyVoice3.0) | Full FT | LoRA (best) | Origin | 出处 |
| --- | --- | --- | --- | --- | --- |
| SS (%) ↑ | **94.8** | 94.5 | 93.3 | 91.2 | [Table 2] |
| ERS (%) ↑ | 96.8 | **97.0** | 94.3 | 92.0 | [Table 2] |
| WER (%) ↓ | **3.8** | 12.1 | 8.5 | 4.0 | [Table 2] |
| 参数量 (M) | 29.8/506.2 | 506.2/506.2 | 43.5/549.7 | - | [Table 2] |

| 指标 | CSP-FT (CosyVoice) | Full FT | LoRA (best) | Origin | 出处 |
| --- | --- | --- | --- | --- | --- |
| SS (%) ↑ | **92.6** | 92.5 | 92.0 | 88.9 | [Table 2] |
| ERS (%) ↑ | 95.5 | **95.6** | 93.6 | 89.8 | [Table 2] |
| WER (%) ↓ | **9.2** | 25.1 | 18.1 | 8.4 | [Table 2] |

| 指标 | CSP-FT (VALLE-X) | Full FT | Origin | 出处 |
| --- | --- | --- | --- | --- |
| SS (%) ↑ | **90.2** | 90.2 | 84.2 | [Table 2] |
| ERS (%) ↑ | **94.2** | 93.0 | 85.6 | [Table 2] |
| WER (%) ↓ | **7.0** | 9.5 | 8.4 | [Table 2] |

### 主观评估 [Table 3]

| 模型 | 方法 | SMOS ↑ | EMOS ↑ | NMOS ↑ | 出处 |
| --- | --- | --- | --- | --- | --- |
| Fun-CosyVoice3.0 | CSP-FT | **4.35** | 4.42 | **4.40** | [Table 3] |
| Fun-CosyVoice3.0 | Full FT | 4.28 | **4.45** | 3.45 | [Table 3] |
| Fun-CosyVoice3.0 | Origin | 3.95 | 3.88 | 4.35 | [Table 3] |
| CosyVoice | CSP-FT | **4.18** | 4.28 | **4.15** | [Table 3] |
| CosyVoice | Full FT | 4.15 | **4.30** | 2.85 | [Table 3] |

### 跨语言跨数据集迁移 [Table 4]

用英语数据集学到的 layer weights 直接用于中文 ESD 数据集微调:

| 模型 | 方法 | SS (%) ↑ | ERS (%) ↑ | CER (%) ↓ | 出处 |
| --- | --- | --- | --- | --- | --- |
| Fun-CosyVoice3.0 | CSP-FT | **85.7** | **94.1** | **1.2** | [Table 4] |
| Fun-CosyVoice3.0 | Full FT | 85.1 | 93.9 | 3.9 | [Table 4] |
| Fun-CosyVoice3.0 | Origin | 83.5 | 90.5 | 1.2 | [Table 4] |

### 训练速度 [Fig 4]

100 步训练时间对比 (相对 Full FT 的加速比):
- GPT-SoVITS: 1.91x
- VALLE-X: 2.62x
- CosyVoice: 2.12x
- Fun-CosyVoice3.0: 1.99x

### 灾难性遗忘动态 [Fig 3]

- Full FT 和 LoRA: 随 epoch 增加,SS/ERS 上升后平台,WER 持续上升
- CSP-FT: SS/ERS 上升同时 WER 显著更早稳定,VALLE-X 和 Fun-CosyVoice3.0 上 WER 甚至最终低于未微调模型

### 消融: 微调层数 [Table 5]

从 2 层 (CSP-FT) 到全模型 (Full FT),呈现 U 形性能曲线:
- 增加中间权重层反而降低性能 (Ours+1/6 到 Ours+4/6)
- 接近全模型时性能恢复但 WER 持续恶化
- Fun-CosyVoice3.0 最明显: 2层 WER 3.8% → 6层 10.8% → 全模型 12.1% [Table 5]

### 消融: 层选择敏感性 [Table 6]

固定一端,改变另一端选择:
- 改变最小权重层: 选第 2/3 小的层性能下降 (所有模型)
- 改变最大权重层: 影响更大,WER 恶化更显著 (GPT-SoVITS WER 8.5→11.3, Fun-CosyVoice3.0 WER 3.8→6.6) [Table 6]
- 例外: CosyVoice 选第 2 大权重层时 SS/ERS 略优但 WER 几乎不变 — 归因于情感/说话人权重分布不均 [§4.6.2]

### Codec LM 作为语音编码器的表现 [Table 1]

四个 codec LM 在情感识别和说话人识别上的表现与 SOTA 自监督模型相当:
- CosyVoice: Speaker Acc 94.97% (最高), Emotion Acc 70.48% [Table 1]
- Fun-CosyVoice3.0: Emotion Acc 72.44% (接近 WavLM_large 72.61%), Speaker Acc 86.87% (最低) [Table 1]
- 表明 tokenization 策略决定性能: ASR tokens 保留语义 (利于情感),acoustic tokens 保留声学 (利于说话人) [§4.1]

## 局限性

1. **架构限制**: 仅在 autoregressive Transformer 架构上验证,未测试 diffusion-based (如 NaturalSpeech 3)、NAR (如 SoundStorm) 或 hybrid 架构 [agent 解读]
2. **层选择的平均策略**: 情感和说话人权重取算术平均,对于两种权重分布差异大的模型 (如 CosyVoice) 可能不是最优 [§4.6.2, Table 6]
3. **情感类别局限**: 实验仅涵盖 8 种离散情感类别,未测试连续维度 (arousal-valence) 或更细粒度的情感控制 [agent 解读]
4. **评估局限**: SS 使用 Resemblyzer (非 ECAPA-TDNN),ERS 使用 Emotion2vec,不同评估工具可能给出不同结论 — [[SpeakerEmbedding]] 页指出 SECS 结果高度依赖所选 encoder [agent 解读]
5. **Stage 1 的数据需求**: 需要带有情感和说话人标注的语音数据来做 characteristic-specific analysis,虽然可跨数据集迁移但仍需标注数据 [§3.1]

## 点评

**优势**: CSP-FT 的核心洞察 — 不同 Transformer 层对不同特征的控制贡献不同 — 虽然直觉上不令人意外,但将其形式化为可操作的微调策略并系统验证,是扎实的工程贡献。特别是:
1. 实验设计完善: 4 个不同架构的 codec LM,覆盖 4 种 speech token 范式 (SSL tokens, acoustic tokens, semantic tokens + extra condition, ASR tokens),增强了结论的可信度
2. 跨数据集/跨语言迁移验证了方法的实用性 — 分析做一次,适应做多次

**不足**: 方法的适用范围可能比论文暗示的更窄:
1. 仅在 codec LM (autoregressive Transformer) 上验证,而 TTS 领域已有大量 diffusion/flow-based 方法
2. "最高 + 最低"的选择策略虽然在实验中表现最好,但缺乏更深入的理论解释 — 为什么最低层微调不会破坏它原有的功能?
3. 论文声称 codec LM 可作为 "highly effective speech encoders for perception tasks",但 Table 1 显示 GPT-SoVITS 和 VALLE-X 的情感识别准确率 (63-66%) 远低于 WavLM_large (72.6%),此声称有些过强

## 可复用的 idea

1. **Weighted-sum layer contribution analysis**: 用下游分类任务的 learnable weighted sum 分析 pretrained model 各层的特征贡献,可迁移到其他需要层选择的场景 (如 speech-LLM 中选择哪些层做 adapter)
2. **Highest + lowest 层选择策略**: 同时微调"最强层"(最大化利用) 和"最弱层"(最大提升空间) 的互补策略,可应用于其他特征适应任务
3. **跨数据集 layer weight 迁移**: utterance-level 特征 (说话人/情感) 的层贡献分布在不同数据集/语言间稳定,可节省重复分析成本
4. **U 形层数-性能曲线**: 微调层数既不是越多越好 (遗忘) 也不是越少越好 (能力不足),存在最优点 — 可作为其他 partial fine-tuning 工作的参考

> [!review] 审阅结论: pass-with-fixes (2026-06-03)
> - **可复述**: PASS — 方法节包含 4 个 WHY 设计选择,因果解释充分
> - **可信赖**: PASS-WITH-FIXES — 速查指标已补充数据集名; Table 1 数字已补标注
> - **可区分**: PASS — 所有因果解释标注了 [论文原文]/[agent 解读]
> - **可定位**: PASS — KB 背景谱系定位具体,引用 Speaker Adaptation 演进线
> - **不污染**: PASS — 无新概念页需创建; 反向更新均为追加
> - 详见 `_review/CSP-FT-review.yml`
