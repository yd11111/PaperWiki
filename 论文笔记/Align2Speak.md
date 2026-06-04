---
type: paper
tier: deep
title: "Align2Speak: Improving TTS for Low Resource Languages via ASR-Guided Online Preference Optimization"
arxiv_id: "2509.21718"
source: "Sources/Align2Speak.pdf"
authors: [Shehzeen Hussain, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Subhankar Ghosh, Roy Fejgin, Ryan Langman, Mikyas Desta, Leili Tavabi, Jason Li]
year: 2025
venue: "Submitted to ICASSP 2026"
tags: [TTS, GRPO, preference-alignment, low-resource, multilingual, autoregressive, IPA, reinforcement-learning, online-RL, ASR-reward, speaker-similarity, PESQ]
concepts: ["[[LLM-based TTS]]", "[[Differentiable Reward Optimization]]", "[[Classifier-Free Guidance]]", "[[Speaker Verification]]", "[[Speaker Embedding]]", "[[TTS Evaluation]]", "[[Prosody Modeling]]"]
models: ["[[论文笔记/Koel-TTS|Koel-TTS]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页: [[LLM-based TTS]]✓, [[Speaker Embedding]]✓, [[Prosody Modeling]]✓, [[Differentiable Reward Optimization]][待确认], [[Speaker Verification]][待确认], [[Classifier-Free Guidance]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 LLM-based TTS 范式下的 **RL post-training** 分支。该分支的演进路线为: RLHF for NLP → RL for TTS on audio (Seed-TTS, 2024) → Preference optimization for codec LM (SpeechAlign, 2024) → Token-level DiffRO (CosyVoice 3, 2025) → Multi-Reward GRPO (Tencent, 2025)。本文是 GRPO-for-TTS 的又一实践,但聚焦于 **低资源语言适配** 这一独特应用场景,这是此前 GRPO/DiffRO 工作较少覆盖的方向。
>
> **已有认知**: 概念库中 [[Differentiable Reward Optimization]] 详细记录了 GRPO 与 DiffRO 在 TTS 中的对比: DiffRO 在 token 空间操作,GRPO 在 audio-level 操作。Tongyi 的对比实验 (RL-for-Audio-LLM) 发现 GRPO 超过 1500 步后可能退化。Multi-Reward GRPO 使用 5 维 reward,TTS-1 使用 3 维 reward (WER+SIM+DNSMOS)。本文同样使用 3 维 reward (CER+SSIM+PESQ),与 TTS-1 架构接近但规模更小。
>
> **创新判断**: 本文的独特贡献不在 GRPO 方法本身(已有多篇工作),而在于 (1) 将 GRPO 与 IPA-based multilingual TTS + few-shot fine-tuning 组合成完整的低资源语言适配 pipeline,(2) 证明 GRPO 在无配对数据时也能改善 TTS 质量(仅用 unpaired text + speaker prompts),(3) 在同一框架下对比 online GRPO vs offline DPO。基线模型 Koel-TTS 是同一团队 NVIDIA 的前作。
>
> 检索命中: [[LLM-based TTS]], [[Speaker Embedding]], [[Prosody Modeling]], [[Differentiable Reward Optimization]], [[Speaker Verification]], [[Classifier-Free Guidance]] | 过滤: [[TTS Evaluation]](pending-review, 与本文关联度较低) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 GRPO 在线强化学习 + ASR/SV/PESQ 多目标 reward 优化低资源语言 TTS,在仅 30 分钟配对数据下将 CER 降低 8 倍以上
> - **路线**: IPA 文本 → NAR Encoder → AR Decoder (Koel-TTS) → NanoCodec tokens → 波形; GRPO 阶段用 unpaired text+speaker prompt 生成多样本,CER+SSIM+PESQ 组合 reward 计算优势,更新策略网络
> - **指标**: 英语 GRPO vs DPO: CER 0.53 vs 0.55 (CFG), SSIM 0.783 vs 0.729 [Table 1]; 葡萄牙语 30min FT+GRPO: CER 3.94% vs baseline 33.00% (>8x 降低) [Fig 3, §3.3]
> - **可借鉴**: (1) IPA tokenization 实现语言无关的 TTS 基座,256 byte-level tokens 即可覆盖所有语音; (2) GRPO 不需要配对数据,只需 unpaired text + speaker audio 即可做 RL; (3) 分段线性 reward 归一化函数设计,将不同量纲指标映射到 [0,1]
> - **局限**: 仅在 Koel-TTS 380M 上验证; 无人工 MOS 评测; 低资源语言仅测试 3 种 (印地语/葡萄牙语/波兰语); 未讨论 GRPO 训练稳定性问题 (Tongyi 发现 >1500 步后 GRPO 退化)

## 核心问题

1. **低资源语言 TTS 的数据瓶颈**: 高质量配对 text-speech 数据昂贵且稀缺,但同一语言的 ASR 模型往往已有 (得益于大规模多语言预训练如 Whisper)。这种不对称性是否可以被利用? [§1]
2. **Offline vs Online preference optimization**: DPO 等离线方法缺乏动态反馈循环,GRPO 等在线方法是否能提供更强的对齐效果? [§1]
3. **无配对数据的 RL**: 当目标语言没有额外配对数据时,能否仅用 unpaired text + speaker prompt 通过 GRPO 改善 TTS? [§2.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

三阶段 pipeline [§2]:

```
Stage 1: 多语言基座训练 (6 语言, IPA, ~21k 小时)
    ↓
Stage 2: 低资源语言 fine-tuning (30min-5hr, 混合训练)
    ↓
Stage 3: GRPO 在线偏好优化 (unpaired text + speaker prompt, 多目标 reward)
```

基座模型采用 Koel-TTS [6],一个 380M 参数的 encoder-decoder 架构 [§2.1]:
- **NAR Encoder**: 非自回归 transformer,编码 IPA 文本
- **AR Decoder**: 自回归 transformer,以 cross-attention 条件于文本编码,生成 NanoCodec [15] 的低帧率 (21.5 FPS) 音频 codec tokens
- **训练目标**: next-frame prediction loss + parallel head [§2.1]

### 关键设计选择

**1. IPA tokenization (语言无关基座)**

论文选择 International Phonetic Alphabet 而非 grapheme 或 language-specific phoneme 作为文本表示 [§2.1]。IPA 提供标准化的语音表示,用 byte-level tokenizer 仅需 256 tokens 即可覆盖所有 IPA 符号。[论文原文] 这种 phonetic-based approach 使模型学习从 universal acoustic units 到 speech 的映射,对 zero-shot/few-shot 语言适配至关重要。

[agent 解读] IPA 的选择解决了 multilingual TTS 的一个根本问题: 不同语言的书写系统高度异构 (拉丁/天城文/西里尔等),但语音空间是共享的。使用 IPA 将问题从"学多种文字到语音的映射"简化为"学统一音素到语音的映射",大幅降低跨语言迁移的难度。

**2. 混合数据 fine-tuning (防止灾难性遗忘)**

fine-tuning 时将低资源语言数据与原始 21k 小时多语言数据混合,低资源数据上采样 5 倍以确保充分暴露 [§2.2]。[论文原文] 这防止了预训练知识的灾难性遗忘,同时让模型学习目标语言的 phonetic 和 prosodic 特征。

固定学习率 1e-5,最多 30k 步,选择低资源语言 validation loss 最低的 checkpoint [§3.1]。

**3. GRPO 多目标在线优化**

GRPO 的核心是 [§2.3]:
1. **采样**: 对每个 prompt (text + speaker audio),生成 K=12 个样本 (multinomial sampling, temperature 0.7)
2. **打分**: 对每个样本计算组合 reward R = 0.45 * R_cer + 0.45 * R_ssim + 0.10 * R_pesq [Eq. 1]
3. **计算优势**: group-relative advantage A_{i,k} = r_{i,k} - mean(r_i) [Eq. 2]
4. **更新策略**: 最大化优势加权的 log-likelihood L_GRPO(θ) [Eq. 3]

[论文原文] 关键设计: 省略了原始 GRPO 中的 KL penalty,"relying solely on the group-relative advantages stabilizes learning and speeds up training" [§2.3]。

**4. Reward 归一化设计**

三个 reward 的归一化方式不同 [§2.3]:
- **CER/SSIM**: 分段线性函数,由 (worst→0, baseline-avg→0.5, best→1) 三点定义 [Fig 2]
- **PESQ**: 简单线性 R_pesq = PESQ / 4.5

[agent 解读] 分段线性归一化的好处是: (1) 将不同量纲统一到 [0,1]; (2) 以 baseline 模型的平均水平为锚点 (0.5),使得 reward 信号始终反映相对于当前策略的改进而非绝对质量; (3) 计算极其轻量。这比 TTS-1 的 unscaled mean-centered advantages 更精细。

**5. Reward 模型选择 (评估时避免 reward hacking)**

[论文原文] CER 由 Whisper-Large-V3 计算 (训练和评估相同); SSIM 训练用 Titanet-Large,评估用 Titanet-Small,有意使用不同模型避免 overfitting 到特定 SV 模型 [§3.1]。

[agent 解读] 训练和评估使用不同 SV 模型是防止 reward hacking 的重要设计。如果评估用与训练相同的 reward model,CER/SSIM 的改进可能只反映对特定模型的过拟合而非真实质量提升。

**6. Prompt construction (解耦配对数据需求)**

[论文原文] GRPO prompt 中 speaker audio 的内容不需要匹配 text transcript [§2.3]。这 decouples 了对配对数据的需求,允许使用 non-parallel speech 和 text 数据。每个语言使用 15k text+speaker 对作为 prompt。

[agent 解读] 这是本文最关键的实用贡献之一。传统 TTS 训练需要配对数据,但 GRPO 只需要 text (可从任何文本语料获取) 和 speaker audio (可从 ASR 语料库或网络音频获取),二者无需对应。这极大降低了低资源语言的数据门槛。

### 训练策略

| 阶段 | 数据 | GPU | Batch | LR | Steps | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Stage 1: 基座训练 | 6 语言 ~21k hr | 32×H100 | 512 | 1e-4 (exp decay 0.998/1k) | ~300k | [§3.1] |
| Stage 2: Fine-tuning | 低资源 30min-5hr + 21k hr 混合 | - | 512 | 1e-5 (fixed) | ≤30k | [§3.1] |
| Stage 3: GRPO | 15k prompts/lang, K=12 samples | - | 64 prompts | 2e-7 (fixed) | ≤2k | [§3.1] |

CFG 在 GRPO 推理时以 50% 概率启用,scale=2.5,使模型在有无 CFG 时都能对齐 [§3.1]。

## 实验

### 英语 Online vs Offline 对比

| 指标 | Base | Base+DPO | Base+GRPO | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| CER (↓, no CFG) | 2.68 | 0.89 | **0.56** | Koel-TTS prompt set | [Table 1] |
| SSIM (↑, no CFG) | 0.637 | 0.667 | **0.759** | 同上 | [Table 1] |
| Squim-MOS (↑, no CFG) | 4.35 | **4.40** | 4.39 | 同上 | [Table 1] |
| CER (↓, CFG) | 0.57 | 0.55 | **0.53** | 同上 | [Table 1] |
| SSIM (↑, CFG) | 0.720 | 0.729 | **0.783** | 同上 | [Table 1] |

GRPO 在 CER 和 SSIM 上全面优于 DPO,MOS 基本持平 [Table 1]。无 CFG 时差距更大 (CER 0.56 vs 0.89),说明 GRPO 的对齐能力减少了对 CFG 的依赖 [agent 解读]。

### 低资源语言适配

| 语言 | 数据量 | 方法 | CER (↓) | 出处 |
| --- | --- | --- | --- | --- |
| 葡萄牙语 | 0 | Baseline | 33.00% | [Fig 3] |
| 葡萄牙语 | 0 | Baseline+GRPO | 降低显著 | [Fig 3] |
| 葡萄牙语 | 30min | FT only | ~16-20% | [Fig 3] |
| 葡萄牙语 | 30min | FT+GRPO | 3.94% | [Fig 3] |
| 印地语 | 30min | FT only | 54.05% | [Fig 3] |
| 印地语 | 30min | FT+GRPO | ~20% | [Fig 3] |
| 波兰语 | 30min | FT only | 28.30% | [Fig 3] |
| 波兰语 | 30min | FT+GRPO | ~12% | [Fig 3] |

关键发现 [§3.3]:
1. **GRPO 无配对数据也有效**: 直接对未见语言的 baseline 施加 GRPO (不做 FT),CER 和 SSIM 均显著改善 [Fig 3, dashed gray]
2. **GRPO + FT 互补**: FT 和 GRPO 的收益叠加,30min 葡萄牙语 CER 从 33% → 3.94% (>8x 降低) [§3.3]
3. **跨语言一致性**: 印地语/葡萄牙语/波兰语三种类型学差异大的语言均有效 [§3.3]
4. **SSIM/PESQ 也改善**: 不只是可懂度,speaker similarity 和音质均有提升 [Fig 3]

## 局限性

1. **仅在 Koel-TTS 380M 上验证**: 未在更大模型 (如 8.8B TTS-1) 或其他架构 (如 NAR flow-matching) 上测试,泛化性未知 [agent 解读]
2. **无人工 MOS 评测**: 所有质量评估均为自动指标 (CER/SSIM/PESQ/Squim-MOS),缺少主观听感验证 [agent 解读]
3. **低资源语言覆盖有限**: 仅 3 种语言 (印地语/葡萄牙语/波兰语),且均有一定程度的 ASR 支持 (Whisper)。对于 Whisper 也不支持的极低资源语言,本方法可能失效 [agent 解读]
4. **GRPO 稳定性未讨论**: Tongyi 的 RL-for-Audio-LLM 发现 GRPO 超过 1500 步后会退化,本文最多 2k 步但未报告训练曲线或稳定性分析 [agent 解读, 基于 [[Differentiable Reward Optimization]] 中的对比数据]
5. **Reward hacking 风险**: CER reward 使用 Whisper,但 Whisper 本身在低资源语言上可能不准确,导致 CER reward 噪声较大。论文未讨论这一风险 [agent 解读]
6. **省略 KL penalty 的影响**: 论文称省略 KL "stabilizes learning",但缺少消融实验支持。KL penalty 的缺失可能导致策略漂移过远,特别是长训练时 [agent 解读]

## 点评

**优势**: 本文的核心价值在于将 GRPO 应用于一个高度实用的场景 (低资源语言 TTS),并设计了一个完整的三阶段 pipeline。IPA tokenization + mixed fine-tuning + unpaired GRPO 的组合简洁有效,30 分钟数据就能达到较好的可懂度。online RL vs offline DPO 的对比也提供了有价值的经验。

**不足**: 从方法创新角度看,GRPO 本身不是新方法 (DeepSeek-Math 提出, Multi-Reward GRPO/TTS-1/Koel-TTS 等已在 TTS 中使用),multi-objective reward 的设计 (CER+SSIM+PESQ 加权和) 也较为直接。本文更像是一个 application paper,工程贡献大于方法贡献。评估维度也偏窄,缺少人工评测和更多低资源语言的验证。

**在领域中的位置**: 在 RL-for-TTS 的快速发展中,本文填补了"GRPO for low-resource TTS"这一空白。与 Multi-Reward GRPO (Tencent) 聚焦单码本中文 TTS、TTS-1 (Inworld) 聚焦大规模英语 TTS 不同,Align2Speak 展示了 GRPO 在 cross-lingual adaptation 场景下的潜力。

## 可复用的 idea

1. **IPA byte-level tokenization**: 256 tokens 覆盖所有语音,是构建 language-agnostic TTS 基座的低成本方案。可直接迁移到任何需要多语言 TTS 的场景。
2. **Unpaired GRPO**: GRPO 不需要配对数据,只需 text + speaker audio (不必对应)。对于数据稀缺场景,这极大降低了 RL post-training 的数据门槛。
3. **分段线性 reward 归一化**: 以 baseline 模型平均表现为锚点 (→0.5) 的分段线性映射,比简单 min-max 归一化更稳健。可直接用于任何多目标 reward 组合场景。
4. **训练/评估用不同 SV 模型**: 训练用 Titanet-Large,评估用 Titanet-Small,防止 reward hacking。这是 RL-for-TTS 的一个好实践。
5. **CFG 随机启用 (p=0.5)**: GRPO 时 50% 概率使用 CFG,使模型在 inference 时有无 CFG 都能工作,提高部署灵活性。

> [!review] 审阅: pass-with-fixes (2026-06-04)
> - **结论**: pass-with-fixes (0 high, 1 medium, 1 low)
> - **medium** [factual-error]: 速查卡片指标字段原将 baseline CER 33.00% 误标为 "FT-only"。已修正为 "baseline 33.00%"。
> - **low** [traceability-gap]: 低资源语言实验表格中部分数值为 Fig 3 近似读数 (~16-20%, ~20%, ~12%),论文仅提供图表未给精确数字,不可避免。
> - 可复述 ✓ | 可信赖 ✓ (修正后) | 可区分 ✓ | 可定位 ✓ | 不污染 ✓
