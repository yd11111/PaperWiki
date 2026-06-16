---
type: paper
tier: deep
title: "BayLing-Duplex: Native Full-Duplex Speech Dialogue with a Single Autoregressive LLM"
arxiv_id: "2606.14528"
source: "Sources/BayLing-Duplex.pdf"
authors: [Qingkai Fang, Shoutao Guo, Yang Feng]
year: 2026
venue: "arXiv"
tags: [speech-LM, full-duplex, turn-taking, interleaved-sequence, DPO, autoregressive, dialogue, barge-in]
concepts: ["[[Full-duplexSpokenDialogue]]", "[[Turn-takinginSpokenDialogue]]", "[[SpeechLanguageModel]]", "[[StreamingSpokenDialogue]]"]
models: ["[[论文笔记/GLM-4-Voice|GLM-4-Voice]]", "[[论文笔记/Moshi|Moshi]]", "[[论文笔记/LLaMA-Omni|LLaMA-Omni]]", "[[论文笔记/LLaMA-Omni2|LLaMA-Omni 2]]"]
tasks: []
datasets: ["InstructS2S-Eval", "Llama Questions", "Web Questions", "Alpaca-Eval"]
kb_context_sources: 5
status: draft
created: 2026-06-16
updated: 2026-06-16
---

## KB 背景

> [!info] KB 背景 (基于 5 个实体页: [[Full-duplexSpokenDialogue]] [待确认], [[Turn-takinginSpokenDialogue]] [待确认], [[SpeechLanguageModel]]✓, [[StreamingSpokenDialogue]] [待确认], [[Speech-LLMIntegrationTaxonomy]] [待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Full-duplexSpokenDialogue]], [[Turn-takinginSpokenDialogue]], [[SpeechLanguageModel]], [[StreamingSpokenDialogue]], [[Speech-LLMIntegrationTaxonomy]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: BayLing-Duplex 属于 Full-duplex Spoken Dialogue 的第三阶段系统(同时听说),基于 native SpeechLM 路线(audio-token-based integration)。与 Moshi 同属"单模型全双工 SpeechLM",但架构路线截然不同:

- **Moshi** (Defossez et al., 2024): 并行双流 RVQ + Depth Transformer,需要百万小时级预训练,Inner Monologue 逐帧文本
- **BayLing-Duplex**: 单序列三通道交错(user speech / assistant text / assistant speech),block 粒度(非逐帧),基于现有 turn-based SpeechLM 微调

**已有认知**: KB 记录了 Full-duplex 系统的演进 — dGSLM (2023, 首个) → Moshi (2024, RQ-Transformer) → LSLM (边说边听, IRQ token) → OmniFlatten (block-by-block) → Raon-SpeechChat (SIL/BOW/BC 三状态, 2026)。BayLing-Duplex 与 Raon-SpeechChat 在时间上接近,均采用特殊状态 token + 单序列交错,但设计细节不同。

**创新判断**: BayLing-Duplex 的核心创新在于"将全双工行为完全还原为标准 next-token prediction":不引入新模块/辅助分类头/注意力 mask 技巧,仅添加 4 个对话状态 token 到标准词表。这使其理论上可迁移到任何自回归 LLM,且可复用现有训练和推理框架。此外,仅用 400K 样本(远少于 Moshi 的百万小时级数据)即达到有竞争力的性能,验证了"强 turn-based SpeechLM + 少量结构化微调 = 全双工"的技术路线。

## 速查

> [!summary] 速查
> - **一句话**: 在 GLM-4-Voice 上仅添加 4 个对话状态 token,通过 400K 样本 SFT + DPO 将 turn-based SpeechLM 转换为全双工系统,turn-taking 和打断决策完全归结为标准 next-token prediction
> - **路线**: 用户语音 → Speech Tokenizer (12.5Hz) → 三通道交错序列 (user speech / text / assistant speech, N:M:N=10:5:10 per block) → 9B LLM (GLM-4-Voice) → Speech Decoder (Flow Matching + HiFi-GAN)
> - **指标**: TT SR@3s 92.0% vs Moshi 71.9%, ISR@2s 100% vs 81.9%, S2S Score 3.39 vs 2.17 [Table 2, InstructS2S-Eval]; Llama Q. 46.0% vs 21.0%, Web Q. 18.1% vs 9.2% [Table 1]
> - **可借鉴**: (1) 对话状态 token 设计 — 仅 4 个 token 编码全双工所有状态,无需辅助头; (2) DPO 中 positive/negative 仅改 timing 不改 content,迫使梯度聚焦于状态 token; (3) SFT loss 的 token 权重调整(降 SILENCE 升 role token)避免稀有 token 被淹没
> - **局限**: 训练/评估全部合成语音(单说话者/近场/无噪);block 大小 N=10 锁死最低延迟 0.8s;无 backchannel 支持;未做 block size N 系统性扫描

## 核心问题

BayLing-Duplex 要解决的核心问题是: **如何以最小的架构改动,将现有强力 turn-based SpeechLM 转换为全双工系统?**

现有全双工方案面临两个困境:
1. Moshi 式方案需要百万小时级预训练 + 定制架构(并行 RVQ stream, Depth Transformer),成本极高 [§1]
2. 外部 VAD 方案从根本上受限 — VAD 无法访问对话语义,导致误切/延迟 [§1]

BayLing-Duplex 探索第三条路: **从强 turn-based checkpoint 出发,通过结构化的少量数据微调获得全双工能力**,且不引入任何新模块。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

BayLing-Duplex 基于 GLM-4-Voice(9B 参数),包含三个组件 [§2]:
1. **Speech Tokenizer**: 修改版 Whisper-large-v3 编码器 + VQ,将 16kHz 波形量化为离散 token,帧率 12.5Hz(每帧 80ms)
2. **LLM**: 9B decoder-only Transformer,从 GLM-4-9B 初始化,词表扩展了 speech token
3. **Speech Decoder**: Flow Matching 模型 + HiFi-GAN vocoder(均改编自 CosyVoice)

**核心创新在于序列布局,而非模型结构** [论文原文]: "the only addition is four special dialogue-state tokens that share the standard token vocabulary" [§1]。推理和训练均不需要新参数/新模块/新注意力 mask。

### 关键设计选择

#### 1. 多通道交错序列 (Multi-Channel Interleaved Sequence)

全双工对话被组织为一个单一的交错序列,分为固定大小的 block [§2.1]:

```
Block b: [N个 user speech tokens] [M个 text tokens] [N个 assistant speech tokens]
```

参数选择: N=10, M=5,每个 block 对应 Δt = 0.8s 的真实时间 [§2.1]。

**为什么选择 block 粒度而非逐 token 交错?** [论文原文]: "the coarser scheme keeps each utterance's text contiguous over several consecutive blocks, which we conjecture aligns better with the text distribution that the underlying LLM was pretrained on" [§5]。[agent 解读]: 逐 token 交错(如 OmniFlatten)会打碎文本的连续性,破坏 LLM 预训练时建立的文本分布;block 粒度保持了文本的局部连贯,降低了 LLM 适应成本。

**为什么 N=10, M=5?** [论文原文]: "N=10 matches the typical English minimum-perceptible-latency threshold while keeping Δt small enough for fluid turn-taking" [§2.1],对应 6.25 text tokens/s,接近 GLM-4-Voice turn-based 解码时的自然英语语速。[⚠️ 论文未详述]: 论文未做 N 的系统性扫描,仅选了一个值。

#### 2. 四个对话状态 Token

文本通道 Z 充当"内心独白"(inner monologue),通过 4 个特殊 token 编码对话状态 [§2.1]:

| Token | 含义 | 作用 |
|-------|------|------|
| [SILENCE] | 助手保持沉默 | 默认填充,听而不说 |
| [ASSISTANT] | 助手开始回复 | 标记 turn-taking onset |
| [PAD] | 文本已写完,语音仍在播放 | 维持语音生成 |
| [EPAD] | 文本和语音均完成 | 标记回复结束 |

**为什么用 token 而非分类头?** [论文原文]: "With this layout, all dialogue-state decisions reduce to next-token prediction over GLM-4-Voice's standard vocabulary, requiring no extra classification head, attention-mask trick, or state machine" [§2.1]。[agent 解读]: 这使得全双工能力完全内嵌于 LLM 的 next-token prediction 范式中,不引入任何外部控制逻辑,maximizing 现有训练/推理基础设施的复用。

#### 3. 因果偏移 (Causal Shift)

文本和助手语音通道相对用户通道偏移一个 block [§2.1]: 在 block b 中,模型已观察到 (b+1)Δt 时刻的用户语音,因此其输出的文本和语音 token 对应 [(b+1)Δt, (b+2)Δt) 的时间窗口。

[agent 解读]: 这个 one-block lookahead 确保了因果性 — 模型不会在时间上"偷看"未来的用户输入,同时使得助手可以在用户说完后立即开始回复(延迟恰好一个 block = 0.8s)。

#### 4. 沉默的连续编码

沉默段不用特殊 token 替换,而是让 speech tokenizer 正常编码沉默波形 [§2.1]: "Silence is tokenized by the same encoder rather than replaced by a special token, preserving acoustic continuity"。

[agent 解读]: 这避免了 silence token 与 speech token 之间的分布断裂,保持了声学连续性,可能有助于模型更自然地在沉默和说话之间过渡。

### 训练策略

#### Stage I: SFT (400K 样本)

在 400K 全双工对话样本上微调 GLM-4-Voice [§2.2]:
- 仅对文本通道和助手语音通道计算 cross-entropy loss(用户语音是 conditioning only)
- **关键细节 — token 权重**: [SILENCE] 在序列中占绝对多数(大部分时间助手在沉默),若 uniform weighting 会让 [ASSISTANT]/[EPAD] 等稀有 role token 的梯度被淹没 [§2.2]

消融实验证实 [Table 3]:
- Uniform weight (ω=1): TT SR@3s 仅 60.3%,模型几乎永远沉默
- ω_sil=0.1, ω_role=10: TT SR@3s 88.9%, S2S 3.23(最优 SFT 设置)
- 单独降低 ω_sil 或单独提高 ω_role 都不够,两者必须同时调整

训练超参: batch size 32, lr 1e-5, cosine schedule, 10% warmup, 1 epoch [§4.1]。Speech tokenizer 和 speech decoder 冻结,仅微调 LLM [§4.1]。

#### Stage II: DPO (200 步)

SFT 后模型已学会序列布局,但 timing 决策不够精准 [§2.2]。DPO 的 positive/negative pair 设计巧妙 [§3]:

| 场景 | Positive | Negative | 差异 |
|------|----------|----------|------|
| Turn-taking | 用户说完后 0.8s gap 开始回复 | 将 gap 改为 Uniform(2,5)s | 仅 timing |
| Interruption | 被打断后 δ_react~Uniform(0.8,2.0)s 停止 | 将 delay 改为 Uniform(3,5)s | 仅 timing |

**为什么只改 timing 不改 content?** [论文原文]: "positive and negative share the same user-channel audio and textual content wk, so the DPO objective is forced to focus its update on the dialogue-state tokens and not on textual content, which is essential for preserving response quality" [§3]。

DPO 训练目标含辅助 SFT 项防止灾难性遗忘 [§2.2]:
$$L = L_{DPO} + \lambda_{ftx} \cdot L_{SFT}(s^+)$$

最优超参: β=0.5, λ_ftx=0.5, 200 步, lr 3e-7 [§4.1]。

#### 数据构造

基于 LLaMA-Omni 2 (Fang et al., 2025b) 的 200K 多轮语音对话语料 [§3]:
- 文本来源: Alpaca + UltraChat,经 Llama3.3-70B-Instruct 重写
- 语音合成: CosyVoice zero-shot voice cloning,用户多样声音,助手统一声音
- 转为全双工: turn-taking 样本 200K + interruption 样本 200K = 400K 总量
- Turn-taking: 用户结束后 0.8s gap 插入助手回复; 用户间 gap ~Uniform(0.5, 3.0)s
- Interruption: 用户在助手回复中随机位置重新进入,助手 δ_react~Uniform(0.8, 2.0)s 后停止

### 推理

推理逐 block 进行 (Algorithm 1) [§2.3]:
1. 接收 N 个 user speech token
2. 自回归生成 M 个 text token(mask 限制在文本+状态 token)
3. 自回归生成 N 个 assistant speech token(mask 限制在 speech token)
4. 解码语音并在 (b+1)Δt 时刻播放

[论文原文]: 推理时必须对 text/speech slot 施加 vocab mask,"without masking, the LLM occasionally emits cross-channel tokens that corrupt the speech decoder's input" [§2.3]。训练时不需 mask,因为 cross-entropy loss 自然抑制了错误 token 类型。

[论文原文]: 用户 barge-in 时,当前 block 的 in-flight assistant speech token 会生成完毕才处理新用户音频,保持解码严格自回归 [§2.3]。

## 实验

| 指标 | BayLing-Duplex (+DPO) | BayLing-Duplex (SFT) | Moshi | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Llama Q. Acc. (%) | 46.0 | 44.3 | 21.0 | Llama Questions (N=300) | [Table 1] |
| Web Q. Acc. (%) | 18.1 | 18.0 | 9.2 | Web Questions (N=2032) | [Table 1] |
| TT SR@3s (%) | 92.0 | 88.9 | 71.9 | InstructS2S-Eval (N=199) | [Table 2] |
| S2S Score (1-5) | 3.39 | 3.23 | 2.17 | InstructS2S-Eval | [Table 2] |
| Overlap (s) ↓ | 1.10 | 1.51 | 2.07 | InstructS2S-Eval | [Table 2] |
| ISR@2s (%) | 100.0 | 91.4 | 81.9 | InstructS2S-Eval | [Table 2] |
| Q2 S2S Score (1-5) | 3.27 | 2.95 | 2.45 | InstructS2S-Eval | [Table 2] |

### 全双工 vs Turn-based 质量对比

| 模型 | Llama Q. Acc. | Web Q. Acc. | Alpaca S2S | 出处 |
|------|---------------|-------------|------------|------|
| Turn-based SFT | 45.3 | 15.9 | 3.16 | [Table 5] |
| BayLing-Duplex (SFT) | 44.3 | 18.0 | 3.23 | [Table 5] |

[论文原文]: "the duplex model is on par with or stronger than the turn-based one" — Llama Q. 降 1.0 点,Web Q. 升 2.1 点,Alpaca-Eval 升 0.07 [§4.3, Table 5]。全双工训练不牺牲生成质量。

### 消融: Token 权重 (Stage I)

| ω_role | ω_sil | TT SR@3s | S2S | ISR@2s | Q2 S2S | 出处 |
|--------|-------|----------|-----|--------|--------|------|
| 1 | 1 | 60.3 | 3.19 | 100.0* | 2.82 | [Table 3] |
| 1 | 0.1 | 82.4 | 3.13 | 89.6 | 2.78 | [Table 3] |
| 10 | 1 | 73.9 | 3.02 | 88.5 | 2.81 | [Table 3] |
| 10 | 0.1 | 88.9 | 3.23 | 91.4 | 2.95 | [Table 3] |

*注: uniform weighting 下 ISR@2s 100% 是退化结果 — 模型几乎从不说话,自然不需要停止 [§4.4]。

### 消融: DPO 超参 (Stage II)

β=0.5, λ_ftx=0.5 为最优: TT SR@3s 92.0%, S2S 3.39 [Table 4]。ISR@2s 在所有设置下均为 100%,说明 DPO 对 interruption 非常鲁棒 [§4.4]。β=0.1 使模型偏离 SFT policy 过远,S2S 降至 3.31 [Table 4]。

## 局限性

1. **合成数据限制**: 训练和评估全为合成语音 — 单说话者、近场、无噪声。真实部署须应对背景噪声、混响和竞争说话者 [Limitations]
2. **Block 大小固定**: N=10 锁死最低响应延迟 0.8s,减小 N 会降低延迟但压缩 per-block 文本预算,论文未做系统性扫描 [Limitations]
3. **交互模式不完整**: 仅覆盖 turn-taking 和 interruption,未涉及 backchannel、多方对话和情感感知的轮次切换 [Limitations]
4. **继承骨干局限**: 受限于 GLM-4-Voice 在罕见语言、代码切换和 OOD 声学条件下的能力 [Limitations]
5. **仅英语评估**: 虽然 GLM-4-Voice 支持多语言,但全双工能力仅在英语上验证
6. **baseline 有限**: 仅与 Moshi 对比,未与 OmniFlatten、Raon-SpeechChat、SALMONN-omni 等同期系统比较

## 点评

**优势**:
1. **极简主义设计哲学**: 在全双工 SpeechLM 领域中,BayLing-Duplex 可能是架构改动最小的方案 — 无新模块、无新 head、无新 attention mask,仅 4 个 token。这种"最少引入"原则值得学习。
2. **DPO pair 的精巧设计**: 正负样本仅改 timing 不改 content,迫使优化聚焦于对话状态 token,避免了 DPO 常见的质量退化问题。这个 trick 可推广到任何需要优化 timing 但保持 content 质量的场景。
3. **数据效率**: 400K 样本(远少于 Moshi 的百万小时级数据)即达有竞争力结果,验证了"强 pretrained checkpoint + 结构化微调"路线的可行性。
4. **消融充分**: token 权重消融清晰揭示了稀有 token 被淹没的问题及解决方案。

**不足**:
1. **评估生态单薄**: 仅与 Moshi 比较,缺少与 Raon-SpeechChat (同期, 同样用状态 token + 单序列交错)、FLM-Audio (最相关 concurrent work)、OmniFlatten 等系统的对比。
2. **真实场景验证缺失**: 全合成数据训练 + 全合成数据评估,无法判断方法在真实噪声/远场/多说话者条件下的鲁棒性。
3. **Block size trade-off 未探索**: N=10 是唯一尝试的值,0.8s 最低延迟在实际应用中偏高(人类感知阈值约 200-300ms)。

## 可复用的 idea

1. **"将对话状态编码为词表 token"的范式**: 适用于任何需要在自回归 LM 中注入控制信号的场景。4 个 token 覆盖了 listen/speak/speaking/done 四种状态,可推广到其他交互式生成任务。

2. **Timing-only DPO**: 在需要优化 timing/节奏但不想影响 content 质量时,构造 positive/negative 仅在 timing 维度有差异的 pair。这个方法论可迁移到 TTS 韵律优化、字幕时间轴对齐等任务。

3. **SFT loss 的 token 权重调整**: 当序列中某些 token 类型极度稀疏(如控制 token)时,uniform loss 权重会让这些 token 的学习信号被淹没。通过降低高频 token(silence)权重 + 提高稀有 token 权重的组合,解决类不平衡问题。这个方案比 focal loss 等更直接且可解释。

4. **"Turn-based → Full-duplex 微调"技术路线**: 对于资源有限的团队,先训练/获取强 turn-based SpeechLM,再通过少量结构化数据微调获得全双工能力,比从头训练全双工系统成本低几个数量级。

> [!review] 审阅: pass (2026-06-16)
> 审阅报告: [[_review/BayLing-Duplex-review.yml]]
> 所有关键数字与 PDF 原文交叉验证一致 (Tables 1-5)。方法节因果解释充分,来源标注覆盖率 ≥ 80%。3 个 low severity issues (frontmatter models/tasks 语义、§ 标注微调),均不影响可信性。
