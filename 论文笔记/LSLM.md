---
type: paper
tier: deep
title: "Language Model Can Listen While Speaking"
arxiv_id: "2408.02622"
source: "Sources/LSLM.pdf"
authors: [Ziyang Ma, Yakun Song, Chenpeng Du, Jian Cong, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xie Chen]
year: 2024
venue: "arXiv preprint"
tags: [full-duplex, speech-LM, turn-taking, interactive, streaming, decoder-only-TTS, SSL, fusion]
concepts: ["[[Full-duplexSpokenDialogue]]", "[[Turn-takinginSpokenDialogue]]", "[[SpeechLanguageModel]]", "[[StreamingSpokenDialogue]]", "[[Self-SupervisedSpeechRepresentation]]", "[[LLM-basedTTS]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓, [[Full-duplexSpokenDialogue]][待确认], [[Turn-takinginSpokenDialogue]][待确认], [[StreamingSpokenDialogue]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: LSLM 处于 SpeechLM 演进中"从半双工到全双工"的关键节点。在 Full-duplex Spoken Dialogue 的谱系中,LSLM 与 dGSLM (dual transformer + cross-attention, 2023)、Moshi (RQ-Transformer + 并行双流, 2024) 并列为早期全双工端到端 SLM,但采用了独特的"单说话者 AR 生成 + streaming SSL 监听"架构,而非 dGSLM 的双 transformer 或 Moshi 的统一多流。

**已有认知**:
- **全双工三阶段**: Traditional → Streaming → Full-duplex (Survey, Cui et al. 2024)。LSLM 是 Stage 3 的先驱之一。
- **Turn-taking 实现路线**: 从 VAD 打断检测 → 端到端 irq/n-irq 标记 (Mini-Omni 2) → chunk-level state prediction (Freeze-Omni) → SIL/BOW/BC 三状态建模 (Raon-SpeechChat)。LSLM 的 IRQ token 方案是此演进的早期代表。
- **SSL 表征**: vq-wav2vec 是 SSL 语音表征的第一代方案 (VQ + BERT 两阶段, 2020),后续被 wav2vec 2.0 (端到端 contrastive)、HuBERT (masked prediction + k-means) 等取代。LSLM 选用 vq-wav2vec 主要因其全卷积架构天然适合 streaming。
- **LLM-based TTS**: decoder-only token-based TTS 是 LLM-based TTS 的标准范式,LSLM 的 speaking channel 遵循此路线,使用单层离散 token (而非 VALL-E 式多层 RVQ)。

**创新判断**: LSLM 的核心创新在于将"边说边听"形式化为 FDM 问题 (Eq. 3),并系统探索了 early/middle/late 三种融合策略。这在当时 (2024.08) 是首个公开的端到端全双工 SLM 研究之一,早于 Moshi 的正式论文发表。

## 速查

> [!summary] 速查
> - **一句话**: 首个系统探索端到端"边说边听"的 speech language model,通过 middle fusion 将 streaming SSL 监听通道融入 AR TTS 生成,实现实时 turn-taking
> - **路线**: Text → Decoder-only Transformer (106M) → 离散 speech tokens → Vocoder; 同时 Streaming vq-wav2vec encoder → Projection → 融合到 Transformer 各层
> - **指标**: Command-FDM: WER 4.05%, F1 98.00% (clean); Voice-FDM: WER 5.33%, F1 95.50% (clean) [Table 2, 3]
> - **可借鉴**: (1) Middle fusion 策略 -- 在每个 Transformer block 注入监听信号,平衡生成质量与交互能力; (2) IRQ token 机制 -- 用特殊 token 实现端到端 turn-taking,无需外部 VAD; (3) 形式化 FDM loss (Eq. 3) 将全双工建模统一为条件语言建模
> - **局限**: 仅 106M 参数,训练于 LibriTTS 585h,未在真实对话数据上验证; 未实现 speech-in speech-out 对话 (输入是 text→TTS,不是真正的口语对话); 未开源模型权重

## 核心问题

1. **现有 SLM 无法实时交互**: 所有 turn-based SLM (SpeechGPT, LauraGPT 等) 只能"听完再说",无法在说话时处理外部输入 [§1]
2. **全双工建模的形式化缺失**: 在 LSLM 之前,没有工作系统地将"边说边听"形式化为语言建模问题 [§3]
3. **融合策略的未知**: 如何将监听通道的实时信号融入生成通道,early/middle/late 各有什么 trade-off [§4.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

LSLM 由三个模块组成 [§4, Fig 2]:

1. **Speaking channel (说话通道)**: Decoder-only Transformer (12 blocks, 768 dim, 106M params) 自回归生成离散 speech tokens。使用 vq-wav2vec 的 SSL encoder 将目标语音编码为连续 embedding,再通过量化操作 (Qnt) 得到单层离散 token [§4.1, Eq. 4-5]。训练目标是标准的 next-token prediction loss [Eq. 6]。推理时 Top-P 采样 (p=0.99, temp=1.0) 生成 token,GAN-based vocoder 恢复波形 [§4.1, Eq. 8]。

2. **Listening channel (监听通道)**: 同一个 vq-wav2vec SSL encoder 处理实时音频输入 (噪声、人声或两者混合),但**不做量化**,而是通过线性 projection 层将连续 embedding 映射到 AR 模型可处理的空间 [§4.2, Eq. 9-10]。[agent 解读] 不量化是关键设计 -- 量化会丢失细粒度的声学差异,而监听通道需要区分噪声与人声,连续表征保留了更丰富的信息。

3. **IRQ (Interruption) token**: 在 tokenizer 词表中添加特殊的 IRQ token [§4.3]。训练时,如果样本包含打断,模型在打断开始 mu=0.5 秒后输出 IRQ token 并停止生成 [§5.3]。[论文原文] "If a human interrupts, the model should stop speaking within a detection interval mu seconds after the interruption starts."

### 关键设计选择

**1. 三种融合策略** [§4.3, Fig 3]:

| 策略 | 融合位置 | 机制 |
|------|---------|------|
| Early Fusion | 输入层 | 监听 embedding 与说话 token embedding 在第一层之前相加 |
| Middle Fusion | 每个 Transformer block | 监听信号在每个 Transformer block 的输入处额外加入 |
| Late Fusion | 输出 logits | 监听信号在 softmax 之前与输出 logits 融合 |

[论文原文] Middle fusion 的优越性解释: "middle fusion achieving an optimal balance between speech generation and real-time interaction" [§Abstract]。

[agent 解读] 为什么 middle fusion 最优:
- **Early fusion 的问题**: 在输入层融合使模型难以区分说话和监听两个通道的信息,导致 next-token prediction 受损 (WER 33.56%, 严重退化) [Table 2]。这是因为两种截然不同的信号 (要生成的内容 vs 环境音) 在第一时间被混合,破坏了生成通道的表征。
- **Late fusion 的问题**: 仅在输出层融合限制了监听信息参与生成过程的深度。在噪声条件下,late fusion 的 precision 显著下降 (93.06%),说明模型无法充分利用监听信号进行噪声/人声区分 [Table 2]。
- **Middle fusion 的优势**: 每一层都注入监听信号,使模型在每个抽象层级都能感知外部输入,同时保持说话通道在输入层的独立编码。这是一种"渐进式信息注入"策略。

**2. 单层离散 token (而非多层 RVQ)**: [论文原文] "This design better meets the requirements for real-time interaction, as it eliminates the need to wait for the completion of AR token synthesis before performing NAR operations" [§4.1]。[agent 解读] 这是为 streaming 推理做的妥协 -- VALL-E 式的 AR+NAR 两阶段需要等 AR 完成才能开始 NAR,而 LSLM 的单层 token 可以边生成边送入 vocoder,满足实时性要求。

**3. Speaking/Listening 共享 SSL encoder**: [论文原文] 说话通道和监听通道使用同一个 vq-wav2vec encoder [§4.2, Eq. 9 引用 Eq. 4]。[agent 解读] 共享 encoder 保证了两个通道的表征空间一致,有助于模型学习两者之间的关系 (如区分"自己在说什么"和"外面传来了什么")。

**4. Timbre 与 content 解耦**: Vocoder 使用独立的 acoustic prompt A 提供音色 [§4.1, Eq. 8],AR 模型只需建模语义内容。[论文原文] "This decoupling of timbre from content allows the AR model to focus more on semantic information rather than paralinguistic information."

### 训练策略

训练数据组织 [§5.2-5.3]:
- **TTS 数据**: LibriTTS 585h 语音-文本对
- **噪声**: MUSAN Freesound (电话铃声、爆炸声、白噪声、交通噪声等)
- **打断数据**:
  - Command-based FDM: 22 位精品说话人合成 "Honey" 命令 (Speaker Dependent)
  - Voice-based FDM: Speech Commands Dataset (51k 条单词音频, Speaker Independent)

训练细节 [§5.3]:
- 20 epochs,每个样本 50% 概率添加噪声,50% 概率添加打断
- 打断样本: 在打断开始 0.5 秒后标注 IRQ token,之后的 speaking tokens 被截断
- AdamW,max LR 5e-4,无 weight decay,batch=4
- Warmup 5000 steps + cosine decay
- 选择 val loss 最低的 checkpoint

## 实验

| 指标 | 本文 (Middle Fusion, Clean) | 本文 (Middle Fusion, Noise) | Baseline (Vanilla TTS) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER (Command-FDM) | 4.05% | 4.51% | 4.28% | LibriTTS-testsetB | [Table 2] |
| Precision (Command-FDM) | 97.80% | 97.58% | N/A | 1000 utterances | [Table 2] |
| Recall (Command-FDM) | 98.19% | 97.18% | N/A | 1000 utterances | [Table 2] |
| F1 (Command-FDM) | 98.00% | 97.38% | N/A | 1000 utterances | [Table 2] |
| WER (Voice-FDM) | 5.33% | 8.50% | 4.28% | LibriTTS-testsetB | [Table 3] |
| Precision (Voice-FDM) | 95.21% | 87.69% | N/A | 1000 utterances | [Table 3] |
| Recall (Voice-FDM) | 95.78% | 82.77% | N/A | 1000 utterances | [Table 3] |
| F1 (Voice-FDM) | 95.50% | 85.15% | N/A | 1000 utterances | [Table 3] |

**关键发现**:

1. **Middle fusion 最优** [Table 2]: TTS 质量最高 (WER 4.05% < Vanilla TTS 4.28%,甚至有所改善),交互能力与 early fusion 持平 (F1 98.00% vs 97.20%),在噪声下远优于 late fusion (97.38% vs 94.89%)。

2. **Early fusion TTS 严重退化** [Table 2]: WER 从 4.28% 飙升至 33.56%,说明输入层融合严重干扰了生成能力。但交互能力反而不差 (F1 98.10%),因为模型虽然不能好好说话,但能识别打断。

3. **Voice-based FDM 更具挑战** [Table 3]: 相比 command-based,voice-based 的 WER 从 4.05% 升至 5.33% (clean),F1 从 98.00% 降至 95.50% (clean),在 noise 条件下更是降至 85.15%。这是因为多样化的打断词汇和未见说话人增加了识别难度。

4. **Ablation: 预训练+继续训练最优** [Table 4]: TTS backbone 和 SSL encoder 都加载预训练权重并继续训练 (✚ & ✚) 效果最好 (WER 4.05%, F1 98.00%)。固定 SSL encoder 参数 (✚ & ✓) 导致 F1 下降至 94.44%。[论文原文] "One potential reason is that the SSL encoder has not encountered diverse noise during pre-training, creating a bottleneck for extracting audio with mixed human voice and noise when using fixed pre-trained parameters" [§6.3]。

5. **IRQ token 概率可视化** [Fig 4]: 无打断时 IRQ 概率 < 1e-3;打断信号到达后概率快速上升,在 detection interval (0.5s) 内达到可采样水平。这证明模型学会了将监听通道的实时信号映射为 IRQ token 的生成概率。

## 局限性

1. **不是真正的口语对话系统**: LSLM 的输入是 text (待合成的文本),不是用户的语音。它解决的是"说话时能否检测打断"的问题,不是"如何在被打断后生成新的回复" [agent 解读]。

2. **规模受限**: 106M 参数,LibriTTS 585h 训练,远小于同期的 SpeechLM 系统 (如 Moshi 使用 7B+ 参数和大规模数据)。Voice-FDM 在噪声下 F1 仅 85.15%,说明泛化能力有限。

3. **Command-based FDM 过于简化**: 只用一个词 "Honey" 作为打断命令,且是 speaker-dependent 设置 (22 位固定说话人),不反映真实场景的复杂性 [§5.2]。

4. **评估不充分**: 无主观 MOS 评估,无与同期全双工系统的对比 (如 Moshi, dGSLM),无端到端对话质量评估 [agent 解读]。

5. **vq-wav2vec 的局限**: vq-wav2vec 是 2020 年的模型,后续被 wav2vec 2.0/HuBERT/WavLM 全面超越。选择它仅因其全卷积架构适合 streaming,但表征质量可能制约系统上限。

6. **未开源**: 仅提供 demo 页面,无模型权重或训练代码公开。

## 点评

LSLM 的核心价值在于**问题形式化**和**系统性消融**,而非系统性能本身。

**形式化贡献**: 将全双工建模从工程问题提升为条件语言建模问题 (Eq. 3: P(r_t | R_{1:t-1}, S_{1:t-1}, C)),为后续工作 (OmniFlatten, Freeze-Omni, Raon-SpeechChat 等) 提供了理论框架。

**消融贡献**: 三种融合策略的系统对比揭示了一个重要设计原则 -- 监听信号应在中间层渐进注入,而非在输入层 (破坏生成) 或输出层 (信息利用不充分)。这一发现对后续多通道/多模态 SLM 设计有直接指导意义。

**局限性是时代产物**: 2024 年 8 月,端到端全双工 SLM 仍处于非常早期的阶段。LSLM 没有做真正的 speech-in speech-out 对话,没有大规模数据,没有人类评估 -- 但它开了一个重要的头。后续的 Moshi、Mini-Omni 2、Freeze-Omni、OmniFlatten 等工作在不同维度推进了这个方向,而 LSLM 的 middle fusion 思想和 IRQ token 机制都被广泛借鉴。

**在 KB 谱系中的位置**: LSLM 是 Full-duplex SLM 演进中的关键节点,与 dGSLM (dual transformer) 和 Moshi (unified multi-stream) 代表了三条不同的全双工架构路线: (1) dGSLM: 双模型 + cross-attention; (2) LSLM: 单模型 + 外部 streaming encoder + fusion; (3) Moshi: 单模型统一建模双流。后续系统大多沿 (3) 的统一建模方向发展,但 LSLM 的 "外挂 streaming encoder" 思路在 VITA、FlexDuo 等系统中以不同形式延续。

## 可复用的 idea

1. **Middle fusion 策略**: 当需要将外部实时信号 (如用户输入、环境感知) 融入自回归生成过程时,在每个 Transformer block 注入而非输入层或输出层融合,是一种经验验证的有效模式。可应用于任何需要 "conditioning on real-time external input" 的 AR 模型。

2. **IRQ token 作为端到端 turn-taking 信号**: 在词表中添加特殊 token 来建模对话控制行为 (打断、回传、沉默),比外部 VAD 模块更优雅。后续的 irq/n-irq (Mini-Omni 2)、State 0/1/2 (Freeze-Omni)、SIL/BOW/BC (Raon-SpeechChat) 都是这一思路的扩展。

3. **共享 encoder + 不同处理路径**: 同一个 SSL encoder 在说话通道做量化 (离散 token),在监听通道不量化 (连续 embedding)。这种"同源异构"设计在需要同时处理不同粒度信息的场景中有参考价值。

4. **FDM 形式化 (Eq. 3)**: 将全双工建模表达为 P(r_t | R_{1:t-1}, S_{1:t-1}, C),清晰地指出"额外条件化于实时外部信号"是核心,为后续工作提供了可扩展的形式化基础。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass (9) | 三种融合策略因果解释充分,速查可借鉴含 3 个具体 trick |
> | 可信赖 | pass (9) | 数字标注覆盖率 >90%,指标使用正确 |
> | 可区分 | pass (9) | [论文原文]/[agent 解读] 标注一致,覆盖率 ~90% |
> | 可定位 | pass (9) | 谱系定位明确 (dGSLM/Moshi/LSLM 三路线对比) |
> | 不污染 | pass (9) | 所有概念页已存在,追加更新不引入新风险 |
> 
> Issues: 3 (high: 0, medium: 0, low: 3)
> 详见 `_review/LSLM-review.yml`

---
检索命中: [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓ | [[Full-duplexSpokenDialogue]][待确认], [[Turn-takinginSpokenDialogue]][待确认], [[StreamingSpokenDialogue]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认] | 未命中但可能相关: [[SpokenDialogueEvaluation]]
