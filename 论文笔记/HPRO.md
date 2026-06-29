---
type: paper
tier: deep
title: "HPRO: Hierarchical Progressive Reward Optimization via Preference Extraction for Emotional Text-to-Speech"
arxiv_id: "2606.28249"
source: "Sources/HPRO.pdf"
authors: [Sihang Nie, Xiaofen Xing, Rui Xing, Haoming Li, Ruitong Xiao, Jingyuan Xing, Baiji Liu, Xiangmin Xu]
year: 2026
venue: "arXiv preprint"
tags: [emotional-TTS, reward-optimization, differentiable-optimization, speech-codec, preference-learning, hierarchical-supervision, content-style-disentanglement]
concepts: ["[[EmotionControlinTTS]]", "[[DifferentiableRewardOptimization]]", "[[Gumbel-Softmax]]", "[[SpeechFactorization]]", "[[ProsodyModeling]]"]
models: ["[[论文笔记/HPRO|HPRO]]", "[[论文笔记/CosyVoice2|CosyVoice2]]", "[[论文笔记/CosyVoice3|CosyVoice3]]", "[[论文笔记/HD-PPT|HD-PPT]]", "[[论文笔记/DiffRO|DiffRO]]", "[[论文笔记/RRPO|RRPO]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis]]"]
datasets: ["LSSED", "EmoVoice-DB", "LibriSpeech"]
kb_context_sources: 6
status: draft
created: 2026-06-29
updated: 2026-06-29
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[EmotionControlinTTS]]✓, [[DifferentiableRewardOptimization]]✓, [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓, [[SpeechFactorization]]✓, [[StyleTransferinTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: HPRO 位于 Emotional TTS + Differentiable Reward Optimization 的交叉领域。在 Emotion Control in TTS 的演进线上,HPRO 属于 DPO/RLHF 对齐路线的深化,直接继承 DiffRO (Tongyi Lab, 2025) 的可微 reward 框架,但解决了 DiffRO 在 emotional TTS 中的两个结构性问题(information conflict 和 scale gap)。在 DiffRO 演进线上,HPRO 是继 RRPO (加固 RM 鲁棒性) 之后的另一条改进路线 — RRPO 从 reward model 质量切入,HPRO 从 reward 空间结构和优化粒度切入。
>
> **已有认知**: (1) DiffRO 通过 Gumbel-Softmax 实现 token 空间可微优化,但在情感场景下易出现 reward hacking (语义退化); (2) RRPO 通过三层混合正则化加固 RM 鲁棒性来缓解 hacking,但未解决 content-emotion 共享表示空间的根本冲突; (3) HD-PPT (同组 ICASSP 2026) 提出了 content/prompt-preference token 的层级解码方案,是 HPRO 中 HD-Emo codec 的架构前驱; (4) SpeechFactorization 已有多种 content-style 解耦方法 (GRL/信息瓶颈/self-distillation),但在 reward optimization 中做结构化解耦尚无先例; (5) 多尺度情感建模 (MsEmoTTS/Multi-Step Hierarchical ED) 在 SFT 框架中已有探索,但在 reward optimization 中的层级化尚属首次。
>
> **创新判断**: HPRO 的核心新意在于将 speech factorization (content/style preference token 分离) 与 differentiable reward optimization (多尺度层级 reward) 结合,构建了一个"结构化偏好空间 + 渐进式层级优化"的完整框架。这不是简单的组合,而是针对 DiffRO 在 emotional TTS 中的具体失败模式 (information conflict + scale gap) 的系统性解法。
>
> 检索命中: [[EmotionControlinTTS]], [[DifferentiableRewardOptimization]], [[ProsodyModeling]], [[LLM-basedTTS]], [[SpeechFactorization]], [[StyleTransferinTTS]] | 过滤: [[EmotionControlinTTS]](pending-review), [[DifferentiableRewardOptimization]](pending-review), [[StyleTransferinTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 HD-Emo codec (content/style preference token 结构化分离) + 三阶段渐进层级 reward (frame→word→sentence),解决 DiffRO 在 emotional TTS 中的 information conflict 和 scale gap 两大结构性失配
> - **路线**: CosyVoice2 tokenizer → Gumbel-Softmax 可微采样 → HD-Emo codec (双 FSQ extractor → content/style preference tokens) → 层级 reward (Lcp+Lsp / LwVAD+LASR / LSER) → 渐进式三阶段训练 → CosyVoice2 FM+vocoder → waveform
> - **指标**: WER 4.02% (最低, vs CosyVoice2 5.45%), wVAD-CCC 0.339 (最高), EMO-SIM 0.672 (最高), MOS-N 4.171 (最高), MOS-E 3.650 (第二) [Table I, LSSED+EmoVoice-DB]
> - **可借鉴**: (1) FSQ 信息瓶颈做 content/style 结构化分离用于 reward 空间构建; (2) 渐进式从 dense 到 sparse 的层级 reward 调度策略; (3) stop-gradient 防止声学泄漏到 content branch
> - **局限**: 仅 246h 情感数据 (LSSED+EmoVoice-DB), 未在大规模数据上验证; DiffRO baseline 是自行模拟而非官方实现; 仅 categorical emotion (非连续 AV); 7 页短文,方法细节偏简

## 核心问题

HPRO 要解决的核心问题是: **如何在 LLM-based emotional TTS 的 post-training 中,通过偏好驱动优化提升情感表现力,同时避免语义退化 (reward hacking)?**

具体而言,论文识别了 DiffRO 范式直接应用于 emotional TTS 时的两个结构性失配:

1. **Information Conflict (信息冲突)**: 在单一语音 token 空间中,semantic content 和 emotional style 共享同一 latent representation。全局 reward 最大化成为冲突优化目标 — 增强情感强度不可避免地扰乱携带语言内容的声学结构 [§I]。RRPO 通过加固 RM 鲁棒性缓解了部分 hacking 现象,但未触及这个根本冲突 [论文原文]。

2. **Scale Gap (尺度缺口)**: 情感 reward (如 SER 分类) 是稀疏的句子级信号,而语音生成是稠密的帧级操作。这导致 credit assignment 困境 — 模型缺乏显式机制来定位句子内情感显著的片段 [§I]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

HPRO 由两个核心组件构成 [§II, Fig 3]:

**组件 1: HD-Emo Codec** — 作为 differentiable reward model,将 speech tokens 投射到结构化偏好空间,分离 content 和 style 的梯度路径。

**组件 2: Hierarchical Progressive Reward** — 在结构化偏好空间上,渐进式地从 frame-level → word-level → sentence-level 逐步引入层级 reward。

整体优化流程 [Fig 3]:
1. CosyVoice2 LLM 的 hidden states 投射到离散 speech token 空间
2. 通过 straight-through Gumbel-Softmax 采样获得可微 token 序列 S_hat
3. 冻结的 HD-Emo codec 将 S_hat 映射到 content preference tokens T_c 和 style preference tokens T_s
4. 在 T_c/T_s 的不同粒度上计算层级 reward
5. 梯度通过 relaxed token representation 反向传播到 LLM

### 关键设计选择

#### HD-Emo Codec: 结构化偏好空间构建

HD-Emo codec 的架构复用了同组前序工作 HD-PPT [18] 的设计理念,但重新适配为 reward model 接口 [论文原文, §II-A]。

**双流提取器** [Fig 2]:
- 输入: CosyVoice2 tokenizer 提取的 monotonic speech tokens
- 两个架构相同但参数不共享的 preference token extractor (8-layer Conformer)
- 各自通过 FSQ (Finite Scalar Quantization) 信息瓶颈压缩:
  - Content tokens T_c: FSQ codebook size = 1296 (大容量,保留语言信息)
  - Style tokens T_s: FSQ codebook size = 64 (小容量,仅保留情感风格)

**为什么用 FSQ 而非 VQ?** 论文未详细解释,但 FSQ [21] 的优势在于无需 codebook 学习和 EMA 更新,训练更稳定 [agent 解读]。codebook size 的 20:1 差异 (1296 vs 64) 体现了信息瓶颈的设计意图: content 需要更大容量编码丰富的语言信息,style 的信息瓶颈更紧以强制抽象情感特征 [agent 解读]。

**Content 分支的监督与隔离** [§II-A, Eq. 1]:
- 量化前的 latent representation 经 content adapter 送入 ASR decoder (Whisper-medium 初始化)
- ASR loss: 标准自回归负对数似然 L_ASR = -sum(log P(y_j | y_{<j}, T_c))
- **关键: stop-gradient 机制** — 切断从 reconstruction path 到 content extractor 的梯度流,确保 content extractor 仅由 ASR 监督更新 [论文原文]
- 这防止了 acoustic leakage: 如果允许 reconstruction 梯度回传,content extractor 会倾向于捕获韵律细节来辅助重建,破坏语义纯度 [论文原文]

**Style 分支的层级监督** [§II-A, Eq. 2-4]:
- 句子级: 预训练 emotion2vec 提供 soft emotion distribution p,CE loss 监督预测分布 p_hat
  - L_SER = -sum(p_i * log(p_hat_i))
- 词级: MFA 对齐获取词边界 → 预训练 wav2vec2-ft 模型预测 word-level VAD (Valence-Arousal-Dominance)
  - 使用 Concordance Correlation Coefficient (CCC) 度量一致性
  - L_word = sum_{k in {V,A,D}} (1 - CCC(v_k, v_hat_k))
  - 上下文窗口: 目标词 + 两侧各一个词,考虑过渡韵律 [论文原文]

**融合重建** [§II-A, Eq. 5]:
- Content representation X (来自 T_c) 和 style T_s 通过 Emo-FiLM [26] 机制动态调制:
  - X_tilde = X ⊙ gamma + beta (gamma, beta 由 T_s 投射)
- 调制后送入 speech token combiner (8-layer autoregressive transformer) 重建原始 speech tokens
- 重建用 CE loss 优化

#### Hierarchical Progressive Reward: 渐进式层级优化

**层级 Reward 定义** [§II-B1]:

1. **Frame-level reward** [Eq. 6]: 在 FSQ 的 latent 维度上,用 L1 regression 将生成的 pre-quantization representations (Z_hat_c, Z_hat_s) 与 ground-truth discrete preference tokens (T_c, T_s) 对齐:
   - L_cp = ||Z_hat_c - T_c||_1, L_sp = ||Z_hat_s - T_s||_1
   - 这提供了 dense frame-level 声学监督 [论文原文]

2. **Word-level reward**: 基于 MFA 词边界,施加 wVAD CCC loss (L_wVAD) 约束情感轨迹 + ASR CE loss (L_ASR) 保护语义一致性

3. **Sentence-level reward** [Eq. 2]: emotion2vec CE loss (L_SER) 对齐全局情感分布

4. **KL 正则化**: token-wise categorical KL divergence (L_KL) 约束 LLM 输出不偏离参考模型

总目标 [Eq. 7]: L_total = sum_i (lambda_i * L_i), i in {KL, cp, sp, wVAD, ASR, SER}

**渐进式三阶段策略** [§II-B2]:

为什么需要渐进式? 论文认为同时引入所有层级 reward 会导致早期 reward hacking — 模型在尚未建立稳定声学基础时就被全局情感目标驱动 [论文原文]。

| 阶段 | 新增 Loss | lambda 设定 | Gumbel tau | 设计目的 |
|------|-----------|-------------|------------|----------|
| Stage I: Frame warm-up | L_cp, L_sp, L_KL | lambda_KL=0.05, lambda_sp=2, lambda_cp=1 | tau=2 (smooth) | 建立结构化偏好空间的声学基础 |
| Stage II: Word refine | +L_wVAD, +L_ASR | lambda_KL=0.02, lambda_sp=2, lambda_cp=1, lambda_ASR=5, lambda_wVAD=1 | tau=1 | 精细化局部情感轨迹+语义保护 |
| Stage III: Sentence align | +L_SER | +lambda_SER=0.5 | tau=0.8 (sharp) | 统一全局情感风格 |

Gumbel 温度从 2→1→0.8 的退火策略: 早期高温提供平滑梯度帮助探索,随着监督结构化,逐步降低温度鼓励离散 token 选择的置信度 [论文原文]。

### 训练策略

**HD-Emo Codec 训练** [§III-A2]:
- Content branch 先在 LibriSpeech 960h 上用 ASR 监督预训练,再在 emotional datasets 上 continue training
- Content branch 冻结后,训练 style branch + reconstruction 模块
- 数据: LSSED (206h) + EmoVoice-DB (40h)
- 100 epochs, 8x RTX 4090, Adam lr=1e-4

**HPRO LLM 优化** [§III-A2]:
- Backbone: Qwen2.5-0.5B (CosyVoice2 架构)
- Adam optimizer, lr=1e-5
- 严格按三阶段渐进策略执行

## 实验

| 指标 | HPRO | CosyVoice2 | CosyVoice3 | IndexTTS2 | HD-PPT | TokenRecon (upper bound) | 出处 |
|------|------|------------|------------|-----------|--------|--------------------------|------|
| MOS-N↑ | **4.171** | 4.094 | 4.137 | 4.026 | 4.068 | - | [Table I] |
| MOS-E↑ | 3.650 | 3.530 | 3.538 | **3.692** | 3.547 | - | [Table I] |
| WER↓ | **4.02%** | 5.45% | 4.90% | 6.74% | 4.92% | 7.34% | [Table I] |
| wVAD-CCC↑ | **0.339** | 0.307 | 0.275 | 0.293 | 0.323 | 0.570 | [Table I] |
| EMO-SIM↑ | **0.672** | 0.613 | 0.611 | 0.526 | 0.646 | 0.775 | [Table I] |
| DNSMOS↑ | 3.73 | **3.76** | 3.72 | 3.53 | 3.75 | 3.58 | [Table I] |

**渐进策略消融** [Table II]:

| Model | WER↓ | wVAD-CCC↑ | EMO-SIM↑ | DNSMOS↑ |
|-------|------|-----------|----------|---------|
| CosyVoice2-SFT | 5.42% | 0.297 | 0.641 | 3.63 |
| +Frame | 4.85% | 0.332 | 0.650 | 3.71 |
| +Word | **3.99%** | **0.350** | 0.653 | 3.70 |
| +Sentence | 4.02% | 0.339 | **0.672** | **3.73** |

关键发现 [§III-C]:
- Frame-level 是不可或缺的声学基础: 直接为 dense frame-level generation 提供梯度桥梁,防止基本语音结构丢失 [论文原文]
- Word-level 达到最优 WER 和 wVAD-CCC: 边界感知监督作为中间结构锚 [论文原文]
- Sentence-level 显著提升 EMO-SIM (0.653→0.672),但 wVAD-CCC 和 WER 轻微回退: 这反映了全局情感梯度与局部声学约束在 LLM 统一 token 生成空间中的固有张力 [论文原文]

**Reward 组件消融** (非渐进式训练) [Table III]:

| Model | WER↓ | wVAD-CCC↑ | EMO-SIM↑ | DNSMOS↑ |
|-------|------|-----------|----------|---------|
| w/o content | 13.61% | 0.285 | 0.584 | 3.59 |
| w/o emotion | **3.80%** | 0.295 | 0.637 | **3.78** |
| w/o frame | 4.97% | 0.333 | 0.608 | 3.68 |
| w/o wvad | 4.10% | 0.310 | 0.659 | 3.75 |
| w/o frame&wvad (=DiffRO) | 4.35% | 0.315 | 0.662 | 3.73 |
| **HPRO** | 4.02% | **0.339** | **0.672** | 3.73 |

关键发现 [§III-D]:
- w/o content → WER 暴涨至 13.61%: 完美展示 information conflict — 无约束的情感优化必然破坏语言内容的声学结构 [论文原文]
- w/o emotion → WER 最低 (3.80%) 但情感缺失: 极端 trade-off,与 content-only 形成对照 [论文原文]
- **w/o frame&wvad (模拟 DiffRO)**: 虽维持 decent EMO-SIM (0.662),但 WER (4.35%) 和 wVAD-CCC (0.315) 均劣于 HPRO — 验证了单尺度全局 reward 的局限性 [论文原文]
- HPRO 同时实现最高情感 (EMO-SIM 0.672) 和最低 WER (4.02%): 证明结构化偏好空间 + 层级优化有效缓解了 information conflict [论文原文]

## 局限性

1. **数据规模有限**: 仅使用 246h 情感数据 (LSSED 206h + EmoVoice-DB 40h),远小于 DiffRO/RRPO 的万小时级数据。大规模场景下效果未知 [agent 解读]

2. **DiffRO baseline 非官方**: 由于 DiffRO 未开源,论文在自己框架中模拟其单尺度 reward 范式作为 baseline。模拟实现与原始 DiffRO 可能存在差异 [论文原文, §III-A1]

3. **仅 categorical emotion**: 使用 SER 分类 (emotion2vec) 和 wVAD 作为情感监督,但仅验证了 categorical emotion 场景。对连续 arousal-valence 控制、混合情感等复杂场景未涉及 [agent 解读]

4. **依赖外部工具**: MFA 强制对齐提供词边界,emotion2vec 和 wav2vec2-ft 提供情感标注 — 这些工具的质量直接影响 reward 信号可靠性。特别是 wVAD 预测模型本身可能存在偏差 [agent 解读]

5. **仅 7 页短文**: 方法描述相对简洁,缺少: (a) 各阶段训练步数/epoch 分配; (b) HD-Emo codec 自身的重建质量评估; (c) 更多 baseline 对比 (如 RRPO, EmoRL-TTS, GRPO-based 方法); (d) 不同情感类别的分项结果 [agent 解读]

6. **评估指标循环风险部分缓解但未完全消除**: 虽然论文指出 objective metrics 在最终波形上用外部模型计算 (非 HD-Emo codec 空间),但 wVAD-CCC 和 EMO-SIM 的定义与训练 reward 高度相关,存在隐性 metric alignment [agent 解读]

## 点评

HPRO 提供了一个结构清晰的方案来解决 DiffRO 在 emotional TTS 中的两个关键失败模式。其核心贡献不在于单个组件的新颖性 (content/style 分离已有大量先例,多尺度情感建模也不新),而在于将 speech factorization 与 reward optimization 的结合方式: 用结构化偏好空间为 reward 信号提供干净的梯度路径,用渐进策略为层级 reward 提供稳定的优化调度。

**值得肯定**: (1) 问题定义精准 — information conflict 和 scale gap 确实是 DiffRO emotional TTS 的两个痛点,此前 RRPO 仅解决了 RM 鲁棒性但未触及结构问题; (2) 消融实验设计好 — Table III 的 w/o content vs w/o emotion 完美展示了 information conflict,w/o frame&wvad 直接模拟 DiffRO baseline; (3) 渐进策略消融清晰展示了每层 reward 的增量贡献。

**需要注意**: (1) IndexTTS2 的 MOS-E 最高但 MOS-N 最低的分析 (论文认为是"过于强烈且刻板的情感表达") 是合理的质性解释,但缺少定量证据; (2) 与 RRPO 的对比缺失是遗憾 — RRPO 是 DiffRO emotional TTS 的直接竞争者,且同在 CosyVoice2 上; (3) HD-Emo codec 的 FSQ codebook size 选择 (1296 vs 64) 缺少消融。

**与 KB 已有工作的关系**: HPRO 在 EmotionControlinTTS 演进线上填补了"DiffRO 框架下的结构化情感优化"位置,与 RRPO (RM 鲁棒性) 和 DiffRO-MTR (零样本情感) 形成三条互补路线。从 SpeechFactorization 角度,HD-Emo codec 的 FSQ 双流设计可视为 information bottleneck 方法在 reward 空间的新应用。

## 可复用的 idea

1. **FSQ 信息瓶颈构建结构化 reward 空间**: 用不同 codebook size 的 FSQ 将语音表示分流为 content/style preference tokens,使 reward 梯度天然隔离。这一范式可扩展到其他多属性优化场景 (如 speaker-emotion 联合优化、prosody-content 联合优化)

2. **Stop-gradient 防声学泄漏**: 在多分支编码器中,切断 reconstruction 梯度到特定分支,确保该分支仅由目标监督更新。简单但关键

3. **渐进式 reward 调度 (dense→sparse)**: 先建立局部声学基础 (frame-level),再引入中间结构约束 (word-level),最后统一全局目标 (sentence-level)。这种从 dense 到 sparse 的调度策略可迁移到其他层级化 RL/reward 场景

4. **wVAD + 上下文窗口**: 用 wav2vec2-ft 提取 word-level Valence-Arousal-Dominance 轨迹作为细粒度情感 reward,配合 MFA 词边界和 +/-1 词的上下文窗口。提供了一种不依赖 SER 分类的连续情感信号

## 审阅

> [!review] 独立审阅 (2026-06-29)
> **结论**: pass-with-fixes (3 medium issues, 0 high)
> 详见 `_review/HPRO-review.yml`
>
> Issues:
> 1. [medium/traceability-gap] 速查路线缺出处标注 → 补 [§II, Fig 3]
> 2. [medium/template-compliance] venue 标注为 arXiv,需确认是否已被会议接收
> 3. [medium/traceability-gap] 融合重建架构细节出处应为 [§II-A, Eq. 5; §III-A2]
