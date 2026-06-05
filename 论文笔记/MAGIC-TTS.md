---
type: paper
tier: deep
title: "MAGIC-TTS: Fine-Grained Controllable Speech Synthesis with Explicit Local Duration and Pause Control"
arxiv_id: "2604.21164"
source: "Sources/MAGIC-TTS.pdf"
authors: [Jialong Mai, Xiaofen Xing, Xiangmin Xu]
year: 2026
venue: "arXiv"
tags: [TTS, duration-control, pause-control, flow-matching, zero-shot, controllable-TTS, token-level-control]
concepts: ["[[ConditionalFlowMatching]]", "[[DurationPredictor]]", "[[ProsodyModeling]]", "[[Non-autoregressiveTTS]]"]
models: ["[[论文笔记/MAGIC-TTS|MAGIC-TTS]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[ConditionalFlowMatching]], [[ProsodyModeling]], [[Zero-shotSpeechSynthesis]] + 3 个待确认页: [[DurationPredictor]], [[Non-autoregressiveTTS]], [[Emilia]])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: MAGIC-TTS 建立在 F5-TTS Base (CFM backbone) 之上,属于 flow matching NAR TTS 谱系。与 CosyVoice/Seed-TTS 等 LLM-based 系统不同,它沿用 F5-TTS 的纯 flow matching 路线,不涉及 AR LLM 阶段,因此更适合注入显式 token-level 控制。

**已有认知 — Duration 控制现状**: KB 中 [[DurationPredictor]] 页面记录了 duration 建模从 FastSpeech MFA-based prediction 到 MaskGCT total duration prediction 再到 DMOSpeech 2 RL-based optimization 的演进。现有方法要么提供 utterance-level 语速控制 (IndexTTS2 的 token 数控制),要么作为内部对齐变量 (VITS SDP, Glow-TTS MAS),**均未实现可靠的 token-level 局部时长+停顿双通道控制**。这恰好是 MAGIC-TTS 声称填补的空白。

**已有认知 — Prosody 可控性**: [[ProsodyModeling]] 页面记录了显式 vs 隐式韵律建模的谱系,以及 ProsodyEval 的发现: flow matching NAR 系统存在 mean-mode collapse 导致韵律单调。MAGIC-TTS 试图通过显式数值 conditioning 而非 variance predictor 来解决局部韵律可控性问题,是一个不同于传统 FastSpeech-style variance adaptor 的方案。

**创新判断**: MAGIC-TTS 的核心创新在于将 duration 和 pause **从内部对齐变量提升为外部可控条件**,并通过 zero-value correction + cross-validated supervision 使控制可靠。这与 KB 中已有的 duration predictor 工作在概念上不同 — 后者优化"预测"duration,MAGIC-TTS 优化"遵从"外部给定的 duration。

> 检索命中: [[ConditionalFlowMatching]]✓, [[ProsodyModeling]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[DurationPredictor]](pending-review), [[Non-autoregressiveTTS]](pending-review), [[Emilia]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 F5-TTS flow matching backbone 上注入 token-level duration+pause 数值条件,通过 zero-value correction 和 cross-validated 高置信度时长标注实现可靠的局部时长控制
> - **路线**: Text + Prompt Speech + (optional) Token-aligned Timing Track (d_i, p_i) → MLP timing encoders with zero-correction → residual injection into text embeddings → DiT-based CFM acoustic generator → Mel → Vocos vocoder → Waveform
> - **指标**: C-MAE 10.56ms / C-Corr 0.918 (controlled) vs 36.88ms / 0.588 (spontaneous) [Table 1]; Seed-TTS-Eval EN WER 3.434 / SIM 0.638, ZH CER 2.215 / SIM 0.738 (spontaneous, quality trade-off) [Table 7]
> - **可借鉴**: (1) zero-value correction (g(x)-g(0)) 使零值输入贡献零残差,防止高频零值 (如 pause=0) 产生 dense bias 淹没稀疏控制信号 (如 content duration); (2) cross-validation filtering (Stable-ts × MFA, B@150) 构建高置信度局部时长标注; (3) availability masking + duration dropout 使模型兼容有/无控制两种模式
> - **局限**: 未开源; spontaneous 模式下 Seed-TTS-Eval WER/CER 有显著退化 (EN WER 1.993→3.434); 评估主要在 B@150 子集上进行,规模和多样性有限; 中文 token-level 编辑评估受 MFA 词边界不稳定性限制

## 核心问题

现有 TTS 系统缺乏可靠的 **fine-grained 局部时长控制**: utterance-level 语速调整 (IndexTTS2, PromptTTS) 和 style-level 控制 ("slow speech") 无法精确指定特定 token 的时长或特定边界的停顿 [§1]。AR 系统 (VALL-E, CosyVoice) 中时长是隐式学习的,free-running rollout 使局部时长决策难以稳定 [§1]。传统 NAR 系统 (FastSpeech 2, VITS) 虽有显式 duration 变量,但这些变量是作为 **内部对齐/扩展中间量** 优化的,不是作为用户可控接口设计的 — 直接操纵这些中间量会导致音质退化 [§2.2, Appendix B]。

**MAGIC-TTS 要解决的问题**: 如何在高质量 flow-based zero-shot TTS 系统中实现 **token-level content duration 和 pause 的显式数值控制**,同时保持无控制时的自然合成质量?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MAGIC-TTS 在 F5-TTS Base (DiT-based CFM, hidden 1024, 22 transformer blocks, 16 heads) 的 text-side conditioning 路径上插入 timing 分支 [§4.1]。生成过程不变: 给定 text tokens y=(y_1,...,y_N) 和 acoustic prompt,通过 CFM 生成 mel spectrogram。新增的 token-aligned timing track r_i = (d_i, p_i) 是可选输入,d_i 为 token y_i 的 content duration (帧数),p_i 为该 token 关联的 pause (帧数) [§3.1]。

**关键设计**: timing 信息通过修改 text embedding 注入,不改变 flow matching 目标函数 [§3.2]。CFM loss 保持不变:

L_cfm = E[||M ⊙ (v_θ(x_t, t | c, h) - u)||²] [Eq. 5]

变化仅在 text condition h 的构造方式上 [§3.2, 论文原文]。

### 关键设计选择

**1. 残差注入 + 零值校正 (Zero-Value Correction)**

对每个 token i,text embedding e_i 被增强为 [§3.2, Eq. 6]:

ẽ_i = e_i + α_d · m_d · (g_d(log(1+s·d_i)) - g_d(0)) + α_p · m_p · (g_p(log(1+s·p_i)) - g_p(0))

其中:
- g_d, g_p: 轻量 MLP encoders
- log(1+s·x): 对数压缩,平滑短/长时长的动态范围 [§3.2, 论文原文]
- α_d, α_p: learnable gates,初始化为 0,让模型从 pretrained backbone 行为开始逐步学习 timing 分支的影响 [§3.2, 论文原文]
- m_d, m_p: availability masks

**为什么用 g(x)-g(0) 而不是直接 g(x)?** 在自然语音中,大多数 token 间 pause=0。如果 g_p(0) 非零,零 pause 也会在每个 token 位置产生 dense residual,pause 分支成为强全局 residual,削弱 content duration 的相对控制效果。减去 g(0) 使零输入贡献零残差,让 pause 和 content duration 在控制空间中真正独立 [§3.4, 论文原文]。这是 MAGIC-TTS 最关键的训练技巧。

**2. Cross-Validated 高置信度时长标注 (B@150)**

[agent 解读] Duration 控制比 pause 控制对标注质量更敏感: pause 只需判断边界是否有间隙,但 content duration 要求 token 内部的边界精确。如果标注把一个 token 的边界标错了,同一数值 d_i 在不同样本中对应不同声学区域,模型学到的映射就不可靠。

MAGIC-TTS 的解决方案 [§3.3]:
- **大规模标注 (Stage 1)**: 用 Stable-ts 对 ~30k 小时语料 (2.2M utterances) 生成 token-level 时长标签
- **高置信度子集 (Stage 2)**: 用 Stable-ts 和 MFA 交叉验证。在归一化文本轴上对齐两个 aligner 的输出,保留同时满足三个条件的样本: (1) 文本范围一致, (2) token grouping 顺序一致 (无交叉边界), (3) 每个 matched span 的 start/end/duration 差 ≤ 150ms [§3.3]
- 最终 B@150 子集: 202,086 utterances, 230.72 小时 (从 1300 万条中筛出 ~1.5%) [§4.1]
- SFT 阶段使用 MFA alignment 作为最终时长标签 [§3.3, 论文原文]

**3. Duration Dropout (Availability Masking)**

训练时随机以 probability 0.2 丢弃 timing track (设 availability masks 为零) [§4.1]。这让模型见到无控制条件,保持 spontaneous 合成能力 [§3.4, 论文原文]。

### 训练策略

**Stage 1: Duration-Conditioned Continued Pretraining (CPT)**
- 从 F5-TTS Base official checkpoint 起训 [§4.1]
- 数据: Emilia 子集,用 MNV-17 NV-aware ASR 重新解码,保留含至少一个 nonverbal vocalization 的样本 [§4.1]。[agent 解读] 保留含 NV 的样本使训练集中语义内容和 pause 交替出现,迫使模型更多依赖提供的 duration prior 而非仅从文本推断时长
- Stable-ts 时长标签, 2.2M utterances [§4.1]
- Dynamic batching 30k frames/GPU, 8×A800, lr 7.5e-5, warmup 20k, duration dropout 0.2, 2 epochs = 27k updates [§4.1]

**Stage 2: High-Confidence Local-Control SFT**
- B@150 cross-validated subset, MFA labels [§3.3]
- Same hyperparameters, warmup 1k, final checkpoint at SFT step 36k [§4.1]

## 实验

| 指标 | MAGIC-TTS (controlled) | MAGIC-TTS (spontaneous) | F5-TTS Base | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| C-MAE ↓ | 10.56 ms | 36.88 ms | 38.82 ms | B@150 100-sample test | [Table 1] |
| C-Corr ↑ | 0.918 | 0.588 | 0.594 | B@150 100-sample test | [Table 1] |
| P-MAE ↓ | 8.32 ms | 18.92 ms | 20.68 ms | B@150 100-sample test | [Table 1] |
| P-Corr ↑ | 0.793 | 0.283 | 0.225 | B@150 100-sample test | [Table 1] |
| F1@50 ↑ | 0.410 | 0.128 | 0.129 | B@150 100-sample test | [Table 1] |
| EN WER ↓ | — | 3.434 | 1.993 | Seed-TTS-Eval | [Table 7] |
| EN SIM ↑ | — | 0.638 | 0.667 | Seed-TTS-Eval | [Table 7] |
| ZH CER ↓ | — | 2.215 | 1.665 | Seed-TTS-Eval | [Table 7] |
| ZH SIM ↑ | — | 0.738 | 0.744 | Seed-TTS-Eval | [Table 7] |

**Scenario-based editing** [Table 2]: 在 uniform baseline track (content 170ms, punct 50ms) 上编辑局部 token,基线实现 171.07ms (bias 1.07ms),编辑后 content 达 207.40ms (target 225ms, bias 17.60ms),pause 达 236.67ms (target 260ms, bias 23.33ms)。

**Adjacent span editing** [Table 4]: 2-token/3-token 区域编辑,2.0x 因子下实现 1.777x/1.783x (strict aggregate); 英文最稳定 (1.806x/1.833x) [Table 5]。

**Ablation** [Table 3]:
- w/o zero correction: C-MAE 12.89 (+2.33), C-Corr 0.890 (-0.028)
- w/o cross-validated supervision: C-MAE 15.93 (+5.37), C-Corr 0.787 (-0.131)

## 局限性

1. **Spontaneous 质量退化**: 在 Seed-TTS-Eval 上,MAGIC SFT final 的 EN WER 从 F5-TTS Base 的 1.993 升至 3.434, ZH CER 从 1.665 升至 2.215 [Table 7]。作者承认这是"moderate trade-off" [Appendix B, 论文原文],但对追求无控制场景的纯质量用户来说,这是一个有实际影响的代价。

2. **评估受 MFA 限制**: 几乎所有定量评估依赖 MFA force alignment 重新提取的 duration 与 target 比较。MFA 本身有边界 jitter 和系统性偏差 (尤其中文 token 拆分不稳定) [§4.2, Appendix A],这使得报告的数字是保守下界但也引入了评估噪声,难以精确衡量控制能力的上限。

3. **规模有限**: 定量评估在 B@150 100-sample test set 上进行 [§4.2],scenario-based benchmark 仅 3 个 demo [§4.3],adjacent span editing 在 strict filter 后仅保留 14-58 rows [Tables 4-6]。这些规模难以反映真实应用中的鲁棒性。

4. **未开源**: 截至 2026-04 论文版本,无公开代码或模型权重,无法独立复现验证。

5. **仅 content+pause 两维控制**: 不涉及 pitch、energy 等其他韵律维度。[agent 解读] 但零值校正和 availability masking 的框架在概念上可扩展。

## 点评

MAGIC-TTS 解决了一个真实且被忽视的问题: **token-level 局部时长控制的可靠性**。现有系统要么只提供 utterance-level 控制 (IndexTTS2 的 token 数),要么虽有内部 duration 变量但直接操纵会崩 (Grad-TTS, YourTTS 的 Appendix B probing 证据 [Appendix B])。MAGIC-TTS 的核心洞察 — duration 作为外部 conditioning signal 而非内部 alignment intermediate — 在概念上干净,工程上通过 zero-value correction 和 cross-validated supervision 落地。

**最有价值的贡献**: zero-value correction trick。这不是一个复杂的技术,但抓住了一个容易被忽视的数值问题: 当 pause 大量为零时,naive encoder 会产生 dense bias 淹没 content duration 的稀疏控制信号。这个 trick 泛化性好,任何有"大量零值条件"的 conditional generation 场景都可借鉴。

**Quality trade-off 值得关注**: Seed-TTS-Eval 上 EN WER 从 1.993 升至 3.434 (~72% 增长) 不是 trivial 退化。Table 7 中 MAGIC CPT final 的 WER 2.521 + CER 2.322 + SIM 0.646/0.731 表明 SFT 阶段才引入了主要退化 [Table 7],而 CPT 阶段退化温和。这暗示 high-confidence SFT 在 sharpening 控制力的同时压缩了无控制泛化能力。

**与 KB 中 duration predictor 演进的定位**: MAGIC-TTS 与 DMOSpeech 2 (RL-optimized duration prediction)、FlexSpeech (DPO-based duration)、TED-TTS (inference-time duration steering) 方向完全不同。后三者都在优化 **预测** 更好的 duration;MAGIC-TTS 优化的是 **遵从** 外部给定的 duration。两个方向可以互补: 一个系统可以用 DMOSpeech-style predictor 提供 default duration,再让用户用 MAGIC-style interface 局部编辑。

## 可复用的 idea

1. **Zero-value correction (g(x)-g(0))**: 任何有大量零值条件输入的 conditional generation 场景 (e.g., sparse feature conditioning, optional control signals) 都可能遇到零值 dense bias 问题。减去 g(0) 是一个简单但有效的解耦方式。

2. **Cross-validation 构建高置信度标注**: 用两个独立 aligner (不同 error pattern) 的一致性过滤噪声标签。这在 any sequence labeling task 中都适用,特别是 forced alignment 质量对下游影响大的场景。

3. **Duration-as-external-condition 设计范式**: 将 duration 从 prediction target 变为 conditioning signal,分离 "系统应该生成什么时长" (predictor job) 和 "系统能否遵从给定时长" (acoustic generator job)。这种分离在需要精确控制的应用 (配音、辅助阅读、代码朗读) 中有实际价值。

4. **Availability masking + dropout**: 使模型在有/无控制条件时都能工作,避免依赖训练时总是存在的条件信号。对任何 optional conditioning 的模型设计都有参考意义。

## 审阅

(待审阅 agent 填写)
