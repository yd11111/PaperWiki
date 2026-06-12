---
type: paper
tier: deep
title: "Interpreting and Steering a Text-to-Speech Language Model with Sparse Autoencoders"
arxiv_id: "2606.10029"
source: "Sources/TTS-SAE-Steering.pdf"
authors: [Nikita Koriagin, Georgii Aparin, Nikita Balagansky, Daniil Gavrilov]
year: 2026
venue: "arXiv"
tags: [TTS, interpretability, sparse-autoencoder, activation-steering, mechanistic-interpretability, LLM-TTS, training-free, controllability]
concepts: ["[[LLM-basedTTS]]", "[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[SpeechFactorization]]"]
models: ["[[模型库/CosyVoice3|CosyVoice 3]]", "[[论文笔记/SparseAutoencoderEmotion|SAE-Emotion (Du et al., 2026)]]", "[[论文笔记/EmoSteer-TTS|EmoSteer-TTS]]"]
tasks: []
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: reviewed
created: 2026-06-12
updated: 2026-06-12
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[LLM-basedTTS]], [[SpeechLanguageModel]], [[SpeechTokenizer]], [[SpeechFactorization]] + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]]✓, [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[SpeechFactorization]]✓ | 过滤: [[CosyVoice3]][待确认], [[CodecLanguageModel]][待确认] | 未命中但可能相关: 无

**谱系定位**: 本文属于 TTS 可解释性的新兴方向,将 NLP 中的 Sparse Autoencoder (SAE) 机制解释范式迁移到 TTS LM backbone 中。与已有工作的差异:

| 已有工作 | 操作位置 | 方法 | 聚焦属性 |
|---------|---------|------|---------|
| EmoSteer-TTS (Xie et al., 2025) | CFM DiT 层 | 稠密 difference-in-means steering vector | 情感 |
| SAE-Emotion (Du et al., 2026) | AR semantic backbone (IndexTTS2) | SAE 稀疏特征 | 情感 |
| **本文** | AR LM backbone (CosyVoice3 Qwen2.5) | SAE 稀疏特征 + modality-aware auto-interp | 笑声/性别/语速/口音 + 全层扫描分析 |

**已有认知**:
- [[LLM-basedTTS]] (confirmed): LLM-based TTS 将 TTS 重构为条件语言建模,CosyVoice3 正是典型 hybrid 架构 (LLM + CFM)。本文首次系统分析这类 TTS LM backbone 的内部表示结构。
- [[SpeechLanguageModel]] (confirmed): SpeechLM 处理 text+speech mixed sequence。本文发现 CosyVoice3 的残差流中 text/speech 模态特征随层深度发生显著迁移,揭示了 SpeechLM 的内部模态处理机制。
- [[SpeechTokenizer]] (confirmed): CosyVoice3 使用 25 Hz 监督式 semantic tokenizer (MinMo-based, FSQ)。SAE 分析的对象正是 LM 在这些 discrete speech tokens 上构建的表示。
- [[SpeechFactorization]] (confirmed): 本文的 SAE 特征自然实现了语音属性的解耦(笑声/性别/语速各由独立 SAE 特征编码),但机制不同于传统对抗训练或信息瓶颈 — 是通过稀疏字典学习从预训练模型中"发现"已有的解耦方向。

**创新判断**: 本文是首个对 generative TTS LM backbone 进行 SAE 分析的工作,其核心贡献不在于 SAE 技术本身(已有 NLP 先例),而在于 (1) 将 modality-aware auto-interp 适配到 text-speech 混合序列,(2) 揭示了 TTS LM 层内的模态特化规律,(3) 证明 SAE 特征不仅可解释而且因果可控。

## 速查

> [!summary] 速查
> - **一句话**: 首次在 TTS LM (CosyVoice3 Qwen2.5-0.5B) 上训练 BatchTopK SAE,发现可解释的 text/audio/mixed 特征,并通过 SAE latent space steering 实现因果可控的笑声注入、性别翻转和语速调节
> - **路线**: CosyVoice3 residual stream → BatchTopK SAE (d=16384, k=50) → modality-aware evidence extraction → Gemini auto-interp labeling → detection-style evaluation → SAE latent steering
> - **指标**: 笑声 P(laugh) 0.02→0.79 [Fig 4]; 性别 P(male) 0.063↔0.944 [§4.4]; 语速 voiced duration 2.75s↔10.57s [§4.4]; text-modal auto-interp AUROC 0.921, audio-modal 0.653 [Fig 2]
> - **可借鉴**: SAE 作为 training-free 可解释性+可控性工具,适用于任何 LM-based TTS 系统;modality-aware auto-interp pipeline 可直接复用;层级模态分析方法可帮助理解 TTS LM 内部机制
> - **局限**: 仅验证了 CosyVoice3-0.5B 一个模型;auto-interp 的 labeler 和 scorer 共用同一 Gemini 模型(循环评估风险);仅覆盖部分层的 detection-style 评估;音频特征 AUROC 明显低于文本特征

## 核心问题

LLM-based TTS 系统(如 CosyVoice3)的 LM backbone 在处理 text+speech 混合序列时,其残差流中到底编码了什么信息?不同层如何分配文本/语音模态的表示?这些表示能否被用于因果控制合成语音的属性?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文不提出新的 TTS 模型,而是在已有的 CosyVoice3 上附加分析和控制工具。CosyVoice3 的 LM backbone 是 Qwen2.5-0.5B (hidden size 896, 24 layers),接收 BPE 文本 token + 25 Hz 离散 speech token,自回归生成语音 token。[§3.1; 论文 §3.1 误记为 28 layers,但 §4.1 Figure 1 caption 明确为 "24-layer",且 layer sweep 覆盖 L0-L23 共 24 层]

分析流程 [Fig 5, Appendix A]:
```
CosyVoice3 residual stream (layer L)
  → BatchTopK SAE encoding (d=16384, k=50)
  → Top-20 activating positions per feature
  → Modality tagging (text/audio/mixed, threshold 0.8/0.2) [§3.3]
  → Evidence extraction (text window / 1-sec audio clip / both) [§3.2]
  → Gemini 3.0 Pro auto-interp labeling [§3.4]
  → Detection-style held-out evaluation [§3.5]
  → (Optional) SAE latent steering for causal control [§3.6]
```

### 关键设计选择

**1. Modality-aware feature tagging [§3.3]**

核心创新之一。对每个 SAE 特征,统计其 top-20 激活位置中落在 speech segment 的比例:
- speech fraction ≥ 0.8 → audio-modal
- speech fraction ≤ 0.2 → text-modal
- otherwise → mixed

[论文原文] 这种设计的动机是: CosyVoice3 训练在随机交错的 text-speech 序列上,绝对 token 位置在样本间没有稳定语义,因此必须根据 token 类型而非位置来划分特征模态。

**2. Modality-aware auto-interp [§3.4]**

根据特征模态提供不同类型的证据给 Gemini 3.0 Pro:
- Text-modal: 提供标注了激活 token 的文本上下文窗口
- Audio-modal: 提供以激活 speech token 为中心的 1 秒音频片段
- Mixed: 同时提供文本和音频证据

[agent 解读] 这是对 Paulo et al. (2024) text-only auto-interp 的关键适配。传统 auto-interp 仅处理文本 token,而 TTS LM 的混合序列需要多模态证据。将音频证据直接送给多模态 LLM (Gemini) 是自然的选择。

**3. SAE latent space steering [§3.6, Appendix C]**

在推理时对 speech-token 位置的残差向量进行干预:
1. 用冻结 SAE encoder 将残差向量 h 编码为稀疏 latent z = σ(W_enc h + b_enc)
2. 对选定特征施加扰动: z' = z + α · s ⊙ Z̄ (α=steering 强度, s=方向符号, Z̄=特征尺度)
3. 用 SAE decoder 解码回残差空间: ĥ' = W_dec z' + b_dec
4. 替换原始残差向量

[论文原文] 仅对 speech-token 位置干预,text prefix 和 speech prompt 保持不变。这确保了干预的精确性 — 只改变生成的语音属性,不影响语言内容。

**4. Probe-based feature selection [§4.3, Appendix G]**

使用下游声学探测器筛选控制特征:
- 对每个候选特征,生成一组 steered 样本
- 用外部语音指标(笑声概率/情感分类/口音分类)评分
- 选择指标变化最大的特征用于最终 steering 实验

[agent 解读] 这一步引入了外部验证信号,避免仅依赖 auto-interp 标签选择控制特征。这比 SparseAutoencoderEmotion 的方法(直接用 auto-interp 标签定位)更稳健。

## 关键公式

**SAE 编码-解码 [§3.1]**:

$$z = \text{BatchTopK}(W_{\text{enc}} h + b_{\text{enc}}, k=50)$$

$$\hat{h} = W_{\text{dec}} z + b_{\text{dec}}$$

其中 $h \in \mathbb{R}^{896}$ 是某一层的残差向量, $z \in \mathbb{R}^{16384}$ 是稀疏 latent, BatchTopK 保留 batch 内平均活跃特征数为 $k=50$。

**SAE 训练损失 [§3.1]**:

$$\mathcal{L} = \|h - \hat{h}\|_2^2 + \lambda_{\text{aux}} \mathcal{L}_{\text{dead}}$$

第一项为 reconstruction loss, $\mathcal{L}_{\text{dead}}$ 是 dead-feature auxiliary loss (Gao et al., 2024), 防止特征退化。

**SAE latent steering [§3.6, Appendix C]**:

$$z' = z + \alpha \cdot s \odot \bar{Z}$$

其中 $\alpha$ 是 steering 强度 (实验中取 $[-60, +60]$), $s \in \{-1, +1\}$ 是方向, $\bar{Z}$ 是目标特征在训练集上的平均激活幅度 (归一化尺度)。

**Modality tagging 规则 [§3.3]**:

$$\text{modality}(f) = \begin{cases} \text{audio} & \text{if } \frac{\#\text{speech positions in top-20}}{\text{20}} \geq 0.8 \\ \text{text} & \text{if } \frac{\#\text{speech positions in top-20}}{\text{20}} \leq 0.2 \\ \text{mixed} & \text{otherwise} \end{cases}$$

### 训练策略

**SAE 训练 [§3.1]**:
- 架构: BatchTopK SAE (Gao et al., 2024)
- 字典大小: d = 16,384
- 活跃特征数: k = 50 per token
- 训练数据: ~250M tokens from Emilia dataset
- 训练目标: 标准 reconstruction + sparsity + auxiliary dead-feature loss
- 层覆盖: 全 24 层扫描 L0-L23 (modality + reconstruction 分析), layer 20 作为详细 case study

**Concept probing [Appendix G]**:
- 线性 logistic regression probe (L-BFGS, MaxAbs scaling, 5-fold CV)
- 三个概念: laughter (VocalSound 500 clips), emotion (ESD 500 clips/class), accent (VCTK 0.92, 11 accents)
- 负样本: 500 Emilia-Yodas neutral-speech clips + 500 LJSpeech transcripts

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Laughter P(laugh) | 0.791 (α=+60) | 0.015 (baseline) | 40-voice × 10-text grid | [§4.4, Fig 4] |
| Gender P(male) female→male | 0.944 (α=-50) | 0.629 (baseline) | 40-voice × 10-text grid | [§4.4, Fig 4] |
| Gender P(male) male→female | 0.063 (α=+50) | 0.629 (baseline) | 40-voice × 10-text grid | [§4.4, Fig 4] |
| Speech rate (voiced duration) slower | 10.57s (α=-50) | 3.96s (baseline) | 40-voice × 10-text grid | [§4.4, Fig 4] |
| Speech rate (voiced duration) faster | 2.75s (α=+50) | 3.96s (baseline) | 40-voice × 10-text grid | [§4.4, Fig 4] |
| Auto-interp AUROC (text-modal) | 0.921 | — | Layer 20, 668 features | [§4.2, Fig 2] |
| Auto-interp AUROC (audio-modal) | 0.653 | — | Layer 20, 668 features | [§4.2, Fig 2] |
| Auto-interp AUROC (mixed) | 0.558 | — | Layer 20, 668 features | [§4.2, Fig 2] |
| Concept probe ROC-AUC (laughter, L8+) | 1.000 | — | VocalSound + Emilia | [Table 4] |
| Top-1 SAE feature ROC-AUC (laughter, L12) | 0.924 | — | VocalSound + Emilia | [Table 5] |

**Layer sweep 关键发现 [§4.1, Fig 1]**:

![Figure 1: Layer-wise modality composition — 三段式模态演进 (early mixed → audio commitment → final text reversal)](Sources/TTS-SAE-Steering/fig1-layer-sweep.png)


1. **三段式模态演进**: Early/middle layers (0-14) 以 mixed 和 audio 特征为主; Late layers (16-20) 被 audio-modal 特征主导 (layer 16: 76.1%, layer 20: 74.3%); Final layer 23 急剧翻转为 text-modal (83.1%)

2. **Audio-commitment zone**: Layers 16-20 是"声学承诺区",混合特征在此区间从 40.9% 坍缩到 4.1%,说明网络在此阶段将表示决定性地绑定到 speech-token 流上 [§4.1]

3. **Final layer reversal**: Layer 23 的 text-modal 特征占比突变到 83.1%。[论文原文] 作者认为这是因为最终残差流重新投射到 text-vocabulary-aligned 子空间用于输出 head [Appendix D]

4. **Reconstruction quality**: Text 位置的 EV 始终 ≥ audio 位置,最大 text-audio gap 在 audio-commitment layers (layer 20: 0.080) [Appendix D]

**Representative features [Table 3, Appendix F]**:

| 特征 | 模态 | Auto-interp 标签 | AUROC |
|------|------|-----------------|-------|
| 1376 | Text | 词 "British" 在说话人口音描述中 | 1.000 |
| 1305 | Text | 语音提示描述说话人音调为 "shrill" | 1.000 |
| 1330 | Text | 四位数年份 (2019, 1936) | 1.000 |
| 233 | Audio | 人类笑声 | 0.750 |
| 288 | Audio | 尖叫/喊叫/沉重呼吸 | 1.000 |
| 1225 | Audio | 清辅音 /k/ | 1.000 |
| 164 | Mixed | 口吃/假启动/犹豫标记 | 1.000 |
| 5543 | Mixed | 文本和语音中的音素序列 /ohl/ | 0.979 |

## 局限性

1. **单模型验证**: 仅在 CosyVoice3-0.5B 上验证,无法确认结论是否迁移到更大 TTS 模型或其他架构 (如 VALL-E, Fish-Speech) [§7]
2. **循环评估风险**: auto-interp 的 labeler 和 scorer 均使用同一 Gemini 模型,系统性幻觉会膨胀分数 [§7]
3. **音频特征可解释性较弱**: audio-modal AUROC (0.653) 显著低于 text-modal (0.921),说明当前 auto-interp pipeline 对音频证据的解释能力有限 [Fig 2]
4. **缺乏人类评估**: 所有 steering 效果仅用自动指标衡量,未进行主观听感评估
5. **无内容保持性定量评估**: steering 是否影响语音内容(WER/CER)未报告,仅声称"preserving spoken content" [§4.4]
6. **负样本策略局限**: 负样本来自其他特征而非 representation-neighbor,不测试表示空间邻近混淆 [§7]
7. **与 SAE-Emotion (Du et al., 2026) 缺乏直接对比** [agent 解读]: 两篇同期工作方法相似但未互引

## 点评

**优势**:
- 首次系统性地揭示了 TTS LM backbone 的内部模态处理机制,layer sweep 分析产出了有价值的 insight (三段式模态演进、audio-commitment zone、final layer reversal)
- Modality-aware auto-interp pipeline 设计巧妙,利用 text-speech boundary 将同一框架适配到多模态证据
- Steering 结果令人印象深刻 — 笑声概率从 0.02 到 0.79 的跨度说明 SAE 特征确实捕获了因果相关的表示方向

**不足**:
- 实验规模偏小: 仅一个 0.5B 模型,仅三个 steering 特征,仅 layer 20 详细分析
- 与同期 SAE-Emotion (Du et al., 2026) 高度互补但未交叉引用 — 后者在 IndexTTS2 上做 SAE,聚焦情感控制;本文在 CosyVoice3 上做 SAE,覆盖更广泛的属性
- 缺少 ablation: SAE dictionary size d 和 active features k 的选择依据未给出

**定位**: 这是一篇 "proof-of-concept + insight" 型工作,价值主要在于 (1) 证明 SAE 在 TTS LM 上 work,(2) 揭示模态处理的层级规律,(3) 提供一个可复用的 modality-aware auto-interp pipeline。但距离实用的可控 TTS 工具还有距离 — 需要更多模型验证、人类评估和内容保持性保障。

## 可复用的 idea

1. **Modality-aware auto-interp pipeline**: 对任何处理混合模态序列的 LM (如 audio LLM, multimodal LLM),都可以用 token-type boundary 将特征证据路由到不同模态,再用多模态 LLM 标注。这个 pipeline 可直接迁移到 Whisper、GLM-4-Voice 等系统。

2. **Layer-wise modality analysis 方法论**: 通过统计每层 SAE 特征的模态归属比例,可以绘制任何多模态 LM 的"模态处理地图"。这对理解 multimodal foundation model 的内部机制有通用价值。

3. **SAE latent steering vs. residual steering**: 通过 SAE 的编码-修改-解码循环进行干预,比直接在残差流上加向量更精确 — 干预被约束在 SAE 特征子空间内,减少了对其他表示维度的溢出。这个技术可用于任何 LM-based 生成系统的细粒度控制。

4. **Probe-based feature selection**: 先用 auto-interp 候选,再用外部声学探针筛选真正因果相关的特征,是一个比纯 auto-interp 更稳健的特征选择策略。

## 审阅

> [!review] 审阅 (2026-06-12, auto)
> **结论**: pass-with-fixes
> - 可复述 9 | 可信赖 8 | 可区分 9 | 可定位 9 | 不污染 9
> - ⚠️ [factual-error/medium] 论文 §3.1 说 "28 layers" 但 §4.1 说 "24-layer",笔记跟随 §3.1,实际应为 24 层 (L0-L23)
> - [traceability-gap/low] 局限性第7条 agent 观察未标注来源
> - [template-compliance/low] 速查卡片笑声指标简写值与实验表精确值微小差异
> 
> 详见 `_review/TTS-SAE-Steering-review.yml`
