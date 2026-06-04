---
type: paper
tier: deep
title: "DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling with Large Language Models"
arxiv_id: "2508.08961"
source: "Sources/2508.08961.pdf"
authors: [Yuanyuan Wang, Dongchao Yang, Yiwen Shao, Hangting Chen, Jiankun Zhao, Zhiyong Wu, Helen Meng, Xixin Wu]
year: 2025
venue: "AAAI 2026"
tags: [speech-LM, speech-tokenizer, dual-token, unified-model, understanding-generation, LoRA, semantic-token, acoustic-token]
concepts: ["[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Speech Language Model]]", "[[Self-Supervised Speech Representation]]", "[[Modality Adaptation for Speech LLM]]", "[[Audio Understanding]]", "[[Residual Vector Quantization]]", "[[Codec Language Model]]"]
models: ["[[HuBERT]]", "[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[Speech Tokenizer]]✓, [[Semantic vs Acoustic Tokens]]✓, [[Speech Language Model]]✓, [[Self-Supervised Speech Representation]], [[Modality Adaptation for Speech LLM]], [[Audio Understanding]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: DualSpeechLM 处于 SpeechLM 统一理解-生成的前沿。该领域的核心 trade-off 是 [[Semantic vs Acoustic Tokens]] 之间的信息取舍 — semantic tokens (HuBERT) 语义对齐好但缺声学细节, acoustic tokens (EnCodec/SoundStream) 保真度高但语义弱。现有方案包括: (1) 串联建模 (AudioLM: semantic→acoustic 两阶段), (2) Mixed tokenizer (SpeechTokenizer: RVQ 第一层蒸馏 HuBERT; Mimi/Moshi: 单 VQ 语义 + 额外 RVQ 声学), (3) 监督式 semantic tokens (CosyVoice: ASR encoder 内插 VQ)。DualSpeechLM 提出了第四种路线 — 不在 tokenizer 层面混合,而是在 LLM 建模层面分离: USToken 作为 LLM 输入 (understanding), acoustic token 作为 LLM 输出 (generation)。
>
> **已有认知**: [[Speech Tokenizer]] 页记录了 tokenizer 从 Mel→VQ-VAE→HuBERT→监督式→Mixed→连续 VAE 的演进。[[Modality Adaptation for Speech LLM]] 页讨论了语音编码器到 LLM 的适配方法 (Conv downsampling / CTC compression / Q-Former)。DualSpeechLM 的 USTokenizer 可视为一种新的模态适配: 用 understanding-driven loss 直接在 tokenizer 训练阶段对齐 LLM 输入空间,而非事后用 adapter 桥接。
>
> **创新判断**: 相对于已有知识库记录,DualSpeechLM 的核心新颖性在于 "input/output token 解耦" 设计 — 现有系统 (SpeechGPT, SpiritLM, Moshi) 都使用相同类型 token 作为 LLM 的输入和输出,DualSpeechLM 首次系统性地将两者分离。USTokenizer 的 understanding-driven loss (冻结 LLM 反向传播优化 VQ) 在概念上类似 CosyVoice 的监督式 tokenizer,但直接对齐 LLM 输入空间而非仅用 ASR 任务监督。
>
> 检索命中: [[Speech Tokenizer]]✓, [[Semantic vs Acoustic Tokens]]✓, [[Speech Language Model]]✓ | 过滤: [[Self-Supervised Speech Representation]](pending-review), [[Modality Adaptation for Speech LLM]](pending-review), [[Audio Understanding]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 dual-token 建模框架,用 understanding-driven semantic token 作 LLM 输入、acoustic token 作 LLM 输出,以 4.5K 小时数据和 LoRA 微调实现统一语音理解和生成的互增强
> - **路线**: Speech → Whisper encoder → 2x 下采样 Encoder → 单层 VQ (USToken, 25Hz/250bps) → [理解: USToken→Text LLM→文本输出] [生成: USToken→Text LLM→预测 USToken→AcousticGPT→acoustic token→WavTokenizer decoder→语音]
> - **指标**: ASR WER 4.22/9.71 (clean/other), TTS SIM 0.90/WER 9.25/DNSMOS 3.86 (clean); 仅用 4.5K hrs + LoRA, 优于 SpeechGPT (70K hrs) 和 SpiritLM (570K hrs) [Table 1, 2]
> - **可借鉴**: (1) input-output token 解耦 — 不同模态需求用不同 token 处理,避免单一 token 的 trade-off; (2) understanding-driven loss — 在 tokenizer 训练阶段就用冻结 LLM 反传对齐 VQ 空间; (3) CoC 随机条件策略 — 防止 AcousticGPT 过拟合单一条件信号
> - **局限**: 仅验证英语和少量翻译任务; 数据规模较小 (4.5K hrs) 未验证 scaling; AcousticGPT 仅 6 层,生成质量与 SOTA (Qwen2.5-Omni 等大规模系统) 仍有差距; 依赖外部 WavTokenizer 和 3D-Speaker

## 核心问题

构建同时擅长语音理解和生成的统一 SpeechLM 面临两个根本矛盾:

1. **模态鸿沟**: 语音 token 与文本 token 之间存在巨大分布差异,使文本 LLM 适配到语音 LLM 需要大规模配对数据 (SpeechGPT ~70K hrs, SpiritLM ~570K hrs) [§Introduction]
2. **信息需求冲突**: 理解任务需要高层语义 (与文本对齐),生成任务需要细粒度声学 (prosody, timbre),二者在同一 token 空间中互相拉扯 — 提升一方往往导致另一方退化 [Fig 1]

DualSpeechLM 的核心观点是: **不应强迫同一种 token 同时服务理解和生成,而应在建模层面解耦 input 和 output 的 token 类型** [§Introduction, 论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DualSpeechLM 由两个核心模块构成:

1. **USTokenizer** (Understanding-driven Speech Tokenizer): 生成高层语义 token (USToken),比特率仅 250 bps (25 Hz, codebook 1024)
2. **DualSpeechLM 框架**: 用 USToken 作为 Text LLM 的输入,通过 AcousticGPT 模块生成 acoustic token 作为输出

**理解流程**: Speech → USTokenizer → USTokens + Prompt → Text LLM (Phi3.5-3B) → Text output [§Methodologies]

**生成流程**: Speech → USTokenizer → USTokens (input) → Text LLM → 预测 USTokens (target) → AcousticGPT (conditioned on semantic hidden + speaker embedding) → Acoustic tokens → WavTokenizer decoder → Speech [§Methodologies, Fig 3]

### 关键设计选择

#### 1. USTokenizer: 为什么用 understanding-driven loss 而非传统 SSL 量化?

传统 semantic tokenizer 有两条路线: (a) SSL 表征 + k-means (HuBERT), (b) ASR encoder + VQ (CosyVoice)。两者都不显式考虑与 Text LLM 输入空间的对齐 [论文原文, §Related Work]。

USTokenizer 的做法是: 在 Whisper-medium encoder 之上加 2x 下采样 Encoder → 单层 VQ → Adapter → 冻结 Text LLM (Llama3.2-1B),用 LLM 的 understanding loss 反传优化整个 USTokenizer [§USTokenizer, Eq.2]:

```
L_USTokenizer = α·L_commit + β·L_under + γ·L_reconstruction
```

其中 L_under 就是在冻结 LLM 上做 ASR/SER/SQA 任务的交叉熵损失。这样 VQ codebook 被迫学到 LLM 能直接理解的语义表征 [论文原文]。

权重设置: α=1, β=5, γ=45。高 γ 保证重建质量 (声学信息保留), 适中 β 保证语义对齐 [Appendix D]。

**为什么这个设计 work** [agent 解读]: 关键在于 LLM 参数冻结 — 梯度只能通过 L_under 流向 VQ/Encoder,迫使 token 空间主动靠近 LLM 的输入分布,而非让 LLM 去适应一个未知的 token 空间。这降低了后续 DualSpeechLM 的模态对齐难度。

#### 2. Dual-Token 建模: 为什么分离 input/output token?

传统 SpeechLM (SpeechGPT, UniAudio 等) 使用同一种 token 作为 LLM 的 input 和 output。这导致:
- 用 acoustic token → 理解差 (语义弱) [Table 1, Baseline-Acoustic ASR WER 36.52]
- 用 semantic token → 生成差 (声学细节丢失) [Table 2, Baseline-Semantic TTS SIM 0.80/WER 21.72]
- 两者此消彼长 [Fig 1, 论文原文]

DualSpeechLM 的解决方案: **USToken 只负责输入 (理解), acoustic token 只负责输出 (生成)**, 通过 AcousticGPT 模块桥接两者 [§DualSpeechLM, 论文原文]。Text LLM 先从 USToken 预测 target USToken 序列,再由 AcousticGPT 将其转化为 acoustic token。

**为什么这个设计 work** [论文原文, §Results]: DualSpeechLM-Hubert (仅用 HuBERT 代替 USToken,仍保持 dual-token 架构) 也优于两个 baseline,说明 dual-token 架构本身 (而不仅是 USToken) 就能缓解 LLM 的压力 — AcousticGPT 承担了声学细节生成的任务,让 LLM 专注于语义建模。

#### 3. Semantic Supervision Loss: 为什么需要额外的语义损失?

在生成任务中,Text LLM 需要预测 target USToken 序列。Semantic supervision loss 就是这个预测的交叉熵损失 [Eq.4]:

```
L_semantic = -(1/L) Σ log p(U^tar_t | U^tar_<t, P, U^in; θ)
```

**为什么关键** [论文原文, §Ablation]: 消融实验显示,去掉 semantic loss 后 TTS WER 从 9.25 暴涨到 167.56 [Table 6]。原因是没有 semantic loss,LLM 无法产生准确的 USToken,AcousticGPT 收到的输入质量崩溃,导致生成完全失败。

#### 4. Chain-of-Condition (CoC): 为什么随机切换条件?

AcousticGPT 需要条件信号来生成 acoustic tokens。训练时从三个源中等概率随机采样: (a) prompt hidden states S_p, (b) predicted USTokens U^tar, (c) 两者拼接 [S_p; U^tar]。推理时始终使用拼接形式 [§Chain of Condition]。

**为什么这样设计** [论文原文]: CoC 作为正则化手段,防止 AcousticGPT 过拟合于单一条件信号。更重要的是,它缓解了 USToken 预测不准确带来的连锁错误 — 当 U^tar 质量差时,模型可以依赖更稳定的 S_p [§Chain of Condition]。

**消融证据**: 去掉 CoC 后 TTS/VC 性能均下降 [Table 6, 论文原文]。

### 训练策略

**USTokenizer 训练**:
- 基于 Whisper-medium encoder (冻结)
- 同时训练 ASR + SER + SQA 三个理解任务
- LLM: Llama3.2-1B (冻结)
- 4x A100-40GB, 500K steps [Appendix D, Table 10]

**DualSpeechLM 训练**:
- Text LLM: Phi3.5-3B, 用 LoRA (rank=16) 微调
- AcousticGPT: 6 层 causal transformer, dim=1024, 8 heads
- 同时训练 8 个任务 (4 理解 + 4 生成)
- 训练数据: ~4.5K hours
- 4x A100-40GB, 60K steps (USToken/HuBERT 输入) 或 160K steps (WavTokenizer 输入) [Appendix D, Table 11]
- 总损失: L_generation = λ·L_semantic + ξ·L_acoustic (λ=ξ=1) [Eq.6]
- Speech Decoder: WavTokenizer (frozen)
- Speaker Embedding: 3D-Speaker encoder (frozen) [Appendix B]

## 实验

### 理解任务 [Table 1]

| 指标 | DualSpeechLM (USToken) | DualSpeechLM (HuBERT) | Baseline-Acoustic | Baseline-Semantic | SpeechGPT | SpiritLM | Qwen2.5-Omni | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASR WER↓ (clean) | **4.22** | 5.56 | 36.52 | 5.70 | 42.73 | 6.0 | 1.8 | [Table 1] |
| ASR WER↓ (other) | **9.71** | 14.62 | 80.06 | 14.32 | 78.54 | 11.0 | 3.4 | [Table 1] |
| S2TT BLEU4↑ | **19.74** | 10.20 | 1.91 | 11.13 | 1.07 | - | 30.2 | [Table 1] |
| SER ACC↑ | **60.92** | 51.77 | 54.90 | 51.91 | - | - | 60.03 | [Table 1] |
| SQA BLEU4↑ | **44.38** | 42.59 | - | 42.01 | 3.58 | 7.62 | 42.59 | [Table 1] |

注: DualSpeechLM 仅用 4.5K hrs + LoRA (Phi3.5-3B); SpeechGPT 用 70K hrs + Full FT (LLaMA-7B); SpiritLM 用 570K hrs + Full FT (LLaMA-7B); Qwen2.5-Omni 用 300B tokens + Full FT (7B) [Table 1]。

### 生成任务 [Table 2, 3]

| 指标 | DualSpeechLM | Baseline-Acoustic | Baseline-Semantic | SpeechGPT | Qwen2.5-Omni | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| TTS SIM↑ (clean) | **0.90** | 0.88 | 0.80 | - | - | [Table 2] |
| TTS WER↓ (clean) | **9.25** | 22.11 | 21.72 | 22.15 | 3.73 | [Table 2] |
| TTS DNSMOS↑ (clean) | **3.86** | 3.76 | 3.29 | 3.97 | 4.10 | [Table 2] |
| VC SIM↑ (VCTK) | 0.80 | 0.80 | 0.81 | - | - | [Table 2] |
| T2ST BLEU4↑ (Es2En) | **26.77** | 8.52 | 18.05 | 14.62 | - | [Table 2] |
| TTS QMOS↑ | **3.89** | 3.67 | 2.26 | 3.53 | - | [Table 3] |

注: Qwen2.5-Omni TTS 数据用 Qwen-TTS API 评估,非直接可比 [Table 2, footnote]。

### 消融实验 [Table 6, 7]

| 配置 | ASR WER (clean)↓ | SQA BLEU4↑ | TTS WER (clean)↓ | TTS SIM↑ | 出处 |
| --- | --- | --- | --- | --- | --- |
| DualSpeechLM (full) | 4.22 | 44.38 | 9.25 | 0.90 | [Table 6, 7] |
| w/o understanding-driven loss | 4.81 | 37.67 | 8.59 | 0.90 | [Table 6, 7] |
| w/o reconstruction loss | 4.74 | 46.44 | 52.50 | 0.83 | [Table 6, 7] |
| w/o semantic loss | 4.31 | 43.86 | 167.56 | 0.80 | [Table 6, 7] |
| w/o CoC | - | - | 9.96 | 0.89 | [Table 6] |

### LLM backbone 泛化 [Table 4, 5]

USToken 在 Phi3.5-3B 和 Vicuna-7B 上均优于 HuBERT,证明 USToken 的优势不依赖特定 LLM backbone [Table 4, 5]。

### 与近期 tokenizer 对比 [Table 13]

| Tokenizer | ASR WER (clean/other)↓ | 出处 |
| --- | --- | --- |
| USTokenizer | **4.22/9.71** | [Table 13] |
| TAAE | 13.24/29.49 | [Table 13] |
| MimiCodec (L1) | 5.96/14.54 | [Table 13] |
| CosyVoice2 | 6.29/15.19 | [Table 13] |

## 局限性

1. **数据规模有限**: 仅验证 4.5K hrs,未展示 scaling law;与 Qwen2.5-Omni (300B tokens) 等大规模系统仍有显著差距 [Table 1, 2]
2. **语言覆盖窄**: 仅英语 + 少量翻译任务 (En→De, Es→En, Fr→En),未验证多语言泛化 [§Conclusions]
3. **AcousticGPT 容量有限**: 仅 6 层 transformer,生成质量 (DNSMOS 3.86 vs Qwen2.5-Omni 4.10) 仍有提升空间 [Table 2]
4. **依赖外部组件**: WavTokenizer (acoustic codec) 和 3D-Speaker (speaker embedding) 均为冻结外部模块,系统不是端到端优化 [agent 解读]
5. **互增强的不对称性**: 理解数据帮助生成的效果远大于生成数据帮助理解,生成侧的语义质量较弱,限制了对理解的反馈 [§Discussion, Appendix H]
6. **USTokenizer 训练成本**: 引入冻结 LLM 的 understanding-driven loss 导致训练内存增加 288%,虽不影响推理 [Table 15]

## 点评

DualSpeechLM 的核心贡献是将 "input token ≠ output token" 这一直觉形式化为系统设计。在 SpeechLM 领域,几乎所有前期工作都在 tokenizer 层面解决 semantic-acoustic trade-off (混合 tokenizer、多级 RVQ 分工等),DualSpeechLM 则在 LM 建模层面解决: 让 LLM 只看语义丰富的 USToken,让独立的 AcousticGPT 负责声学细节恢复。

方法论上最有说服力的证据是 Fig 1: baseline 在增加理解数据时生成性能几乎不动甚至退化,而 DualSpeechLM 的生成性能持续提升 — 这直接证明了 dual-token 解耦打破了单 token 建模的信息瓶颈。

消融实验设计清晰且关键发现突出: semantic loss 的消融 (TTS WER 9.25→167.56) 直接证明了 "LLM 必须先正确预测 semantic token,AcousticGPT 才能正常工作" 的因果链。

但也需要注意: 在绝对性能上,DualSpeechLM 与大规模系统 (Qwen2.5-Omni) 仍有显著差距。论文的价值更多在于方法论启发 — 以 4.5K hrs 数据展示出的效率优势,和 dual-token 设计打破理解-生成冲突的机制验证,而非 SOTA 数字本身。

## 可复用的 idea

1. **Input-output token 解耦设计**: 当 LLM 的输入和输出有不同的信息需求时 (如理解需语义、生成需声学),可以用不同类型的 token 分别服务,而非寻找一种万能 token。这个原则可推广到多模态 LLM 的其他场景 (如视觉理解用 semantic visual tokens,图像生成用 pixel-level tokens)。

2. **Understanding-driven tokenizer training**: 在 tokenizer 训练阶段就用下游 LLM 的 loss 反传对齐 VQ 空间。这比先训 tokenizer 再训 adapter 的两阶段方案更紧密,可能减少信息损失。可迁移到任何需要 tokenizer-LLM 对齐的场景。

3. **Chain-of-Condition (CoC) 随机条件正则化**: 训练时从多个条件源随机采样,推理时用全部拼接。这种 "training-time dropout on conditioning" 策略可用于任何有多条件输入的自回归生成模型,增强鲁棒性。

4. **Semantic supervision loss 作为中间监督**: 在 two-stage 生成 (semantic→acoustic) 中,显式监督第一阶段的输出质量,防止错误级联。消融证明这个 loss 是系统能 work 的必要条件 (去掉后 WER 暴涨 18x)。

> [!review] 审阅结论: pass (2026-06-04)
> - **结论**: pass — 5 原则均满足,无 high/medium issue
> - **问题**: 2 low (frontmatter tasks/datasets 空; 权重敏感性分析引用不够具体)
> - **详见**: [[_review/DualSpeechLM-review.yml]]
