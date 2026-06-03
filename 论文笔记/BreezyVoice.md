---
type: paper
tier: deep
title: "BreezyVoice: Adapting TTS for Taiwanese Mandarin with Enhanced Polyphone Disambiguation"
arxiv_id: "2501.17790"
source: "Sources/BreezyVoice.pdf"
authors: [Chan-Jan Hsu, Yi-Cheng Lin, Chia-Chun Lin, Wei-Chih Chen, Ho Lam Chung, Chen-An Li, Yi-Chang Chen, Chien-Yu Yu, Ming-Ji Lee, Chien-Cheng Chen, Ru-Heng Huang, Hung-yi Lee, Da-Shan Shiu]
year: 2025
venue: "arXiv"
tags: [TTS, zero-shot, voice-cloning, Taiwanese-Mandarin, polyphone-disambiguation, code-switching, LLM-based, flow-matching, domain-adaptation]
concepts: ["[[Conditional Flow Matching]]", "[[Speech Tokenizer]]", "[[LLM-based TTS]]", "[[Speaker Embedding]]", "[[Semantic vs Acoustic Tokens]]"]
models: ["[[模型库/CosyVoice|CosyVoice]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[模型库/CosyVoice|CosyVoice]], [[Conditional Flow Matching]], [[Speech Tokenizer]], [[LLM-based TTS]], [[Speaker Embedding]], [[Semantic vs Acoustic Tokens]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: BreezyVoice 是 CosyVoice (Du et al., 2024) 的 Taiwanese Mandarin domain adaptation。CosyVoice 是阿里巴巴提出的 LLM + OT-CFM coarse-to-fine TTS 架构,核心创新为监督式 S3 tokenizer (ASR encoder + VQ)。在知识库中,CosyVoice 已有完整的模型页和精读笔记,后续演进为 CosyVoice 2 (streaming) → CosyVoice 3 (MinMo tokenizer + DiffRO)。
>
> **已有认知**:
> - CosyVoice 的 S3 tokenizer 通过 ASR loss 监督训练,token 显式编码语义信息,在内容一致性上远优于 HuBERT semantic tokens 和 EnCodec acoustic tokens
> - OT-CFM 作为 fine stage renderer,从 speech token + speaker embedding 恢复声学细节 (音色、韵律微观结构)
> - LLM (decoder-only) 自回归生成 speech tokens,输入序列包含 x-vector + text encoding + conditioning speech tokens
> - Speaker embedding 在 CosyVoice 中使用 x-vector,通过 cross-attention 注入
> - CosyVoice 本身主要针对 Mandarin (普通话) + English + 其他语言,Traditional Chinese 字符被主要关联到 Cantonese,故 Taiwanese Mandarin 需要专门适配
>
> **创新判断基准**: 相对于 CosyVoice 原系统,BreezyVoice 的主要创新在于 (1) Taiwanese Mandarin 的域适配方案,(2) 多音字消歧的 phonetic augmentation,(3) 对 CosyVoice pipeline 中错误源的系统性分析,(4) Iconic Unit Augmented Speech Cloning 策略。后两项提供了对 LLM + CFM 管线内部机制的深入洞察。
>
> 检索命中: [[模型库/CosyVoice|CosyVoice]]✓, [[Conditional Flow Matching]]✓, [[Speech Tokenizer]]✓, [[LLM-based TTS]]✓, [[Speaker Embedding]]✓, [[Semantic vs Acoustic Tokens]]✓ | 过滤: [[Voice Cloning Taxonomy]](pending-review), [[Phoneme Representation]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 将 CosyVoice 适配到 Taiwanese Mandarin,通过 phonetic augmentation 解决多音字消歧问题,并提出 Iconic Unit Augmented Speech Cloning 将 PER 降低 61.2%,同时系统性揭示了 LLM 生成的 speech units 是 voice cloning 失败的主要原因
> - **路线**: Text (+g2pW phonetic symbols) → LLM → Speech Units → OT-CFM (+speaker embedding) → Mel → Vocoder → Waveform; 可选路径: iconic speaker LLM → Units → CFM (target speaker embedding) → Waveform
> - **指标**: PER 0.8% / SSL-MOS 4.46 (TCMD), 优于 4 个商用系统 [Table 1]; Iconic Unit Cloning: PER 3.4%→1.3% (61.2% 降低), speaker similarity 仅下降 2.51% [§6.1.3]; 语音克隆平均 speaker similarity 92.29% [§5.2]
> - **可借鉴**: (1) Iconic Unit Augmented Speech Cloning — 解耦内容生成和音色转换的两阶段策略,对所有 LLM+CFM 管线中的 long-tail speaker 问题有直接迁移价值; (2) 通过控制变量实验隔离错误源 (CFM conditions vs LLM conditions) 的分析方法论; (3) BERT-style phonetic symbol augmentation 策略 (50% 句子级 + 15% 字符级)
> - **局限**: (1) 仅训练了 LLM,未微调 S3 tokenizer/CFM,上限受限; (2) 评估数据集规模较小 (30 comparisons, 115 speakers); (3) 中文地名的 code-switching 表现不佳 [Table 2]; (4) 无消融实验量化各训练技巧的独立贡献

## 核心问题

BreezyVoice 要解决的核心问题是: **如何将一个主要针对普通话训练的大规模 TTS 系统 (CosyVoice) 适配到 Taiwanese Mandarin (台湾华语)?** 这包含三个子问题:

1. **语言域差距**: CosyVoice 的训练数据中 Traditional Chinese 字符主要关联 Cantonese 发音,而非 Taiwanese Mandarin,导致直接使用时发音不正确 [§2.2]
2. **多音字消歧**: Mandarin 中大量汉字存在多个读音 (如"行"可读"ㄒㄧㄥˊ"或"ㄏㄤˊ"),G2P 模型难以处理全部情况 [§2.3]
3. **Long-tail speaker 鲁棒性**: 资源受限训练导致部分说话人的 voice cloning 出现灾难性高错误率 (>10% PER),需要诊断和缓解 [§6]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

BreezyVoice 沿用 CosyVoice 的四组件架构 [§3.1]:

```
Text → [g2pW phonetic augmentation] → Augmented Text
                                           ↓
Reference Speech → [S3 Tokenizer] → Ucond  → [LLM] → Uoutput
                          ↓                      ↑
                   Speaker Embedding (v) ─────────┘
                                                   ↓
Reference Speech → Xcond ──→ [OT-CFM] ──→ Xoutput (Mel) → [Vocoder] → Audio
                               ↑
                          v, Ucond, Uoutput
```

四个组件 [§3.1]:
1. **S3 Tokenizer**: 监督式语义语音编码器,将参考语音编码为离散 speech units [§3.1.1]
2. **LLM**: 自回归生成目标语音的 speech units,接收 text tokens + speaker embedding + conditioning units 作为输入 [§3.1.2]
3. **OT-CFM**: 将 speech units 转化为 Mel spectrogram,条件包括 speaker embedding、conditioning units 和 masked mel [§3.1.3]
4. **g2pW**: 外部 Mandarin G2P 模型,用于多音字消歧并生成 phonetic symbols 辅助输入 [§3.1.4]

### 关键设计选择

#### 1. 仅训练 LLM,冻结其他组件

作者选择仅训练 LLM 模块,冻结 S3 tokenizer 和 OT-CFM [§3.2.2]。理由: LLM 被报告为主要捕获语义和韵律信息的组件,因此只需让 LLM 学习 Taiwanese Mandarin 的语义-韵律映射 [论文原文]。训练 3 个 epoch,学习率 1e-4,使用动态 batching (长度 8000) [§3.2.2]。

[agent 解读] 这一决策有两面性: 优势是训练成本低、不破坏预训练 tokenizer/CFM 的能力; 劣势是 S3 tokenizer 可能未充分适配 Taiwanese Mandarin 的音韵特征,而 CFM 无法学到新域的声学细节。

#### 2. Phonetic Symbol Augmentation (多音字消歧)

核心创新之一。采用 BERT-style 的数据增强策略将 Mandarin Phonetic Symbols (注音符号) 嵌入训练文本 [§3.2.1, Fig 1]:

- **句子级**: 50% 的句子进行 augmentation
- **字符级** (在 augmented 句子中):
  - 85% → 注音符号替换
  - 1% → 同音字替换 + 注音符号
  - 1% → 随机字替换 + 注音符号
  - 13% (隐含) → 保持原字符

[论文原文] 这个增强策略的设计目的是: (1) 最大化模型学习汉字发音的效率; (2) 发展模型优先关注辅助 phonetic symbols 的能力,使推理时可通过附加注音来纠正发音 [§3.2.1]。

[agent 解读] 1% 的同音字替换和 1% 的随机字替换模拟了文本中的"噪声",训练模型在汉字信息矛盾时依赖注音符号,这是一种 robustness training。

#### 3. Iconic Unit Augmented Speech Cloning

这是本文最具价值的分析性贡献 [§6.1.3]。对于 long-tail speaker 的灾难性失败 (PER >10%),作者通过控制变量实验定位错误源:

**Step 1: 排除 CFM 组件 [§6.1.1]**
- 实验: 替换 speaker embedding v → v_iconic (用 iconic speaker 的 embedding) → X^I_output ≈ X_output,说明 speaker embedding 不是主因
- 实验: 从 X_cond 重建 U_cond → X^recon_cond ≈ X_cond,说明 conditioning speech 不是主因
- 结论: **LLM 生成的 U_output 是主要错误源** [论文原文]

**Step 2: 分析 LLM 条件的影响 [§6.1.2]**
- 实验: 去掉所有 conditioning (Eq. 8: U^0_output = LLM(v, Y_output),仅保留 speaker embedding + text) → 61% 说话人恶化,39% 改善,整体 PER 上升
- 结论: conditioning 不可简单去除,in-distribution 的 LLM speaker embedding 对生成连贯语音很重要 [论文原文]

**Step 3: 提出两阶段方案 [§6.1.3]**
- 先用 iconic (高质量) speaker 的 LLM 生成 speech units (保证内容正确)
- 再用 CFM 将这些 units 转换为目标说话人的 Mel spectrogram (保证音色正确)
- 结果: PER 3.4% → 1.3% (降低 61.2%),speaker similarity 仅降低 2.51%
- 86/100 speakers 获得改善 [§6.1.3]

[agent 解读] 这个方案的本质洞察是: LLM 和 CFM 在 CosyVoice 架构中的信息分工并非完全正交 — LLM 虽然主要编码语义和韵律,但也受 speaker embedding 质量影响; CFM 虽然主要负责音色,但也能从 LLM 生成的 units 中恢复到不同说话人。Iconic Unit 策略巧妙地利用了 CFM 的 voice conversion 能力来弥补 LLM 在 long-tail speakers 上的脆弱性。

### 训练策略

- **数据**: 多样化的公开语音数据 + 配对转写; 无转写数据使用 Generative Fusion Decoding (LLM-辅助 ASR) 进行 pseudo-labeling [§3.2.1]
- **标点增强**: 使用 Breeze-7B LLM 为原始文本补充标点,使训练文本与下游使用对齐 [§3.2.1]
- **仅训练 LLM**: 3 epochs, lr=1e-4, 动态 batching (max length 8000) [§3.2.2]

## 实验

| 指标 | 本文 (BreezyVoice) | 最佳 Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| PER (%) | 0.80 | 0.43 (Service M) | TCMD | [Table 1] |
| SSL-MOS | 4.46 | 4.63 (Service Z) | TCMD | [Table 1] |
| Human Win Rate | 最高 (24/30 vs Z, 22/30 vs Y, 19/30 vs U) | — | TCMD | [Fig 2] |
| Voice Cloning Avg Speaker Similarity | 92.29% | — | 115 speakers | [§5.2] |
| Iconic Cloning PER (%) | 1.3 (vs 标准 3.4) | — | Spontaneous 100 spk | [§6.1.3] |
| Iconic Cloning ΔSpeaker Sim | -2.51% | — | Spontaneous 100 spk | [§6.1.3] |
| g2pW Polyphone Fix Rate | 7/8 corrected | — | 23 hard cases | [§6.2] |

**结果解读**:

1. **主观评估 (TCMD)**: BreezyVoice 在 30 次配对比较中一致性地优于 4 个商用系统 [Fig 2]。相对 Service Z 胜率 24/30,相对 Service U 胜率 19/30。

2. **客观评估 (TCMD)**: PER 0.80% 排名第二 (Service M 最低 0.43%),但 SSL-MOS 4.46 排名第二 (Service Z 最高 4.63%) [Table 1]。[agent 解读] 值得注意的是 Service M 的 PER 最低但 SSL-MOS 也最低 (3.46),说明发音准确性和整体音质可以脱钩。

3. **Code-switching (TCCSD)**: BreezyVoice 在 General/Entities/Abbreviations 三类上得分最高或并列最高 [Table 2],但在 Taiwan-related Toponyms (地名) 上仅得 3/10,这是唯一弱项。[论文原文] 作者承认地名类别在英文语料中较少见,作为 future work [§5.1.1]。

4. **Voice Cloning 深度分析**:
   - FormosaSpeech 的克隆效果普遍优于 Spontaneous Speech [Fig 3]
   - 超过半数 spontaneous samples PER < 3%,排除 ASR 不准确,人工检查几乎无错 [§5.2]
   - 灾难性失败 (>10% PER) 主要出现在 long-tail speakers [§6]

## 局限性

1. **训练范围有限**: 仅微调 LLM,S3 tokenizer 和 OT-CFM 全部冻结 [§3.2.2]。这意味着 tokenizer 可能无法最优地编码 Taiwanese Mandarin 特有的韵律特征,CFM 也无法适配新域的声学分布。

2. **评估规模小**: 主观评估仅 30 次比较/3 位标注者 [§5.1],voice cloning 评估 115 speakers [§5.2]。统计显著性不足以支撑强结论。

3. **缺乏消融实验**: 没有量化 phonetic augmentation、标点增强、pseudo-labeling 各训练技巧的独立贡献。无法判断哪个技巧最关键。

4. **中文地名 code-switching 弱**: Toponyms 类别仅 3/10 [Table 2],而其他系统最高也只有 7/10,说明这是个普遍难题但 BreezyVoice 没有特别优势。

5. **Iconic Unit 方案的局限**: 虽然 PER 大幅降低,但引入了额外延迟 (两次推理),且 speaker similarity 有轻微下降 (2.51%)。论文未讨论延迟开销。

6. **错误分析的深度有限**: 论文识别了 LLM 生成的 units 是主要错误源,但未进一步分析是什么导致 LLM 在某些 speakers 上失败 — 是训练数据分布问题?是 speaker embedding 的表征瓶颈?还是 LLM 容量不足?

## 点评

**优点**:
- 对 CosyVoice pipeline 中错误源的系统性诊断 (§6.1.1-6.1.2) 是本文最大价值。通过控制变量实验清晰地隔离了 LLM → U_output 是主要错误源,这个发现对所有采用 LLM + CFM 架构的 TTS 系统都有指导意义。
- Iconic Unit Augmented Speech Cloning 是一个简洁有效的工程方案,利用现有组件 (无需额外训练) 就能大幅提升 long-tail speaker 的鲁棒性。
- Phonetic augmentation 的 BERT-style 设计 (同音字/随机字噪声) 是有创意的,让模型学会在多音字场景下"听从"外部 phonetic guidance。

**不足**:
- 作为一篇适配工作,创新深度有限 — 架构完全沿用 CosyVoice,主要贡献在数据准备和分析层面。
- 评估不够严谨: 商用系统对比使用匿名字母标记且无法复现; 主观评估仅 3 位标注者。
- 论文未讨论训练数据规模和组成 (多少小时? 多少说话人?),难以判断域适配的数据效率。

**在 CosyVoice 谱系中的定位**: BreezyVoice 不是 CosyVoice 的架构改进,而是一个**域适配案例研究**,其主要价值在于 (1) 验证了 CosyVoice 架构在低资源域适配场景下的可行性; (2) 提供了对该架构内部工作机制的深入分析洞察。

## 可复用的 idea

1. **Iconic Unit Augmented Speech Cloning**: 在 LLM + CFM/Diffusion 管线中,当 long-tail speaker 导致 LLM 生成质量下降时,用 iconic speaker 的 LLM 输出替代,然后用 CFM 做 voice conversion。这个策略不需要额外训练,可直接应用于 CosyVoice 2/3、Seed-TTS 等系统。

2. **Pipeline 错误源隔离方法**: 通过逐组件替换 (替换 speaker embedding → 隔离 CFM; 重建 conditioning → 隔离 tokenizer; dropout conditions → 隔离 LLM) 来定位多阶段系统中的错误源。这是一种通用的 debug methodology。

3. **BERT-style phonetic augmentation**: 在训练文本中混入 phonetic symbols + 同音字噪声 + 随机字噪声的策略,让 TTS 模型学会在文字歧义时依赖外部 phonetic guidance。可迁移到任何需要 G2P 精确控制的 TTS 系统。

4. **仅训练 LLM 的轻量级域适配**: 对于 CosyVoice 类架构,冻结 tokenizer 和 CFM 仅微调 LLM 即可实现域适配,这表明 LLM 是架构中语言/域知识的主要载体。

> [!review] 审阅结论
> 待审阅,见 `_review/BreezyVoice-review.yml`
