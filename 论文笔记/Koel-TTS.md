---
type: paper
tier: deep
title: "Koel-TTS: Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidance"
arxiv_id: "2502.05236"
source: "Sources/Koel-TTS.pdf"
authors: [Shehzeen Hussain, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Subhankar Ghosh, Mikyas T. Desta, Roy Fejgin, Rafael Valle, Jason Li]
year: 2025
venue: "Preprint (NVIDIA)"
tags: [TTS, LLM-based-TTS, preference-alignment, DPO, RPO, classifier-free-guidance, zero-shot-TTS, autoregressive, codec-language-model, encoder-decoder, multilingual, speaker-similarity]
concepts: ["[[Classifier-FreeGuidance]]", "[[LLM-basedTTS]]", "[[CodecLanguageModel]]", "[[DifferentiableRewardOptimization]]", "[[SpeakerVerification]]", "[[FiniteScalarQuantization]]", "[[Speech-TextAlignment]]"]
models: ["[[XTTS]]", "[[论文笔记/E2TTS|E2 TTS]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 7
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 7 个实体页: [[Classifier-FreeGuidance]], [[LLM-basedTTS]], [[CodecLanguageModel]], [[DifferentiableRewardOptimization]], [[SpeakerVerification]], [[FiniteScalarQuantization]], [[Zero-shotSpeechSynthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Koel-TTS 属于 [[LLM-basedTTS]] 中的 encoder-decoder 自回归路线,区别于 decoder-only 的 VALL-E 系列和 CosyVoice 的 hybrid (LLM + CFM) 路线。其核心创新集中在 post-training 阶段:将 NLP 领域的偏好对齐 (DPO/RPO) 和 [[Classifier-FreeGuidance]] 引入自回归 codec token 预测模型。
>
> **已有认知**:
> - [[Classifier-FreeGuidance]] 在 TTS 中主要用于 diffusion/flow-based 非自回归模型 (如 CosyVoice, NaturalSpeech 3),在 LLM-based 自回归 token 预测模型中的应用尚少——仅有 Parakeet (Darefsky et al. 2024) 初步尝试,且只处理了 text-independent 条件 dropout
> - [[DifferentiableRewardOptimization]] 页面记录了 TTS 偏好对齐的演进: SpeechAlign (2024, DPO on codec LM) → Seed-TTS (2024, REINFORCE audio-level) → CosyVoice 3 (2025, DiffRO token-level) → GRPO (2025)。Koel-TTS 的 DPO/RPO 方案在时间线上与 SpeechAlign 并行,但方法论上有显著差异
> - [[SpeakerVerification]] 在 zero-shot TTS 中既作为评估指标 (SECS/SSIM) 又作为训练信号,Koel-TTS 同时利用了这两个角色
> - [[FiniteScalarQuantization]] 被用于 Koel-TTS 采用的 Low Frame-rate Speech Codec (LFSC, Casanova et al. 2025),该 codec 以 21.5 FPS 和 1.89 kbps 运行,8 个独立 codebook,codebook 独立性使并行预测成为可能
>
> **创新判断**: 相比已有工作,Koel-TTS 的贡献在于: (1) 将 CFG 从仅 text-independent dropout 扩展到同时 dropout text 和 context audio 两种条件; (2) 用 ASR + SV 自动化 reward 构建偏好对,避免了 SpeechAlign 使用 ground-truth 作 chosen 的分布不匹配问题; (3) 系统对比了三种 context conditioning 架构。
>
> 检索命中: [[Classifier-FreeGuidance]] [待确认], [[LLM-basedTTS]]✓, [[CodecLanguageModel]] [待确认], [[DifferentiableRewardOptimization]] [待确认], [[SpeakerVerification]] [待确认], [[FiniteScalarQuantization]] [待确认], [[Zero-shotSpeechSynthesis]]✓ | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 encoder-decoder 自回归 TTS 模型上,用 ASR/SV 驱动的偏好对齐 (DPO/RPO) + 同时 dropout text 和 audio 的 CFG,以 18-21K 小时数据在 zero-shot TTS 上达到 SOTA 可懂度和说话人相似度
> - **路线**: Text → NAR Encoder → Cross-attention → AR Decoder → N codebook tokens (并行) → LFSC Codec → Waveform; post-training: 采样多输出 → ASR CER + SV SSIM 排序 → Pareto 最优配对 → DPO/RPO 微调; 推理: conditional + unconditional logit 插值 (CFG scale gamma)
> - **指标**: CER 0.55% / WER 1.41% (Koel-TTS 380M, unseen, LibriTTS test-clean) vs GT CER 0.80%; SSIM 0.740 (1.1B multilingual); MOS 4.054/SMOS 3.826 均优于 VALL-E-X/XTTS-v2/StyleTTS-2/E2-TTS/F5-TTS [Table 3]
> - **可借鉴**: (1) Pareto 最优排序构建多目标偏好对——可推广到任何多指标优化场景; (2) CFG 对自回归 token 预测模型同样有效,无需额外微调即可大幅提升; (3) RPO 比 DPO 对超参更鲁棒且支持 non-high-contrast 对
> - **局限**: SSIM 指标上不及 flow-matching 模型 (F5-TTS 0.834, E2-TTS 0.848 vs Koel-TTS 0.740); 训练数据远少于竞品 (18-21K h vs 100K+ h); 代码已开源但开源时间晚于投稿; 对 challenging text (重复词) CER 仍有 4-5% 空间

## 核心问题

Koel-TTS 要解决的核心问题是: **自回归 LLM-based TTS 模型在推理时缺乏可控性——表现为幻觉 (hallucination)、不必要的发声 (undesired vocalizations)、以及同一输入的多次采样质量方差大** [§1]。传统 TTS 模型通过显式 duration/pitch 预测器实现确定性映射,而 LLM-based 模型隐式建模所有 variation,副作用是输出不稳定。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Koel-TTS 是一个 encoder-decoder Transformer,text 通过 NAR encoder 编码后经 cross-attention 注入 AR decoder,decoder 在每个时间步并行预测 N=8 个 codebook 的 token [§2.2]。使用 Low Frame-rate Speech Codec (LFSC, Casanova et al. 2025),帧率仅 21.5 FPS (bitrate 1.89 kbps),采用 [[FiniteScalarQuantization]] 保证 codebook 独立性,因此可以在单个时间步并行预测所有 codebook,不需要 delay pattern 或额外 NAR 阶段 [§2.1] [论文原文]。

三种 context audio conditioning 方案 [§2.2, Fig 1]:

| 方案 | 机制 | 优势 | 劣势 |
|------|------|------|------|
| **Decoder Context** | context audio tokens 直接拼接在 target tokens 前作为 AR decoder 输入 | 无需额外 encoder; 最好的 unseen speaker 泛化 [Table 1] | decoder 需处理更长序列 |
| **SV Conditioned** | 预训练 SV 模型 (TitaNet) 提取 speaker embedding → 投影+时间扩展+加到 text encoder 输出 → cross-attention | 迁移学习优势; 低数据场景有效 [论文原文] | speaker vector 是压缩表示,丢失 style/accent 细节 [§2.2] |
| **Multi Encoder** | 独立的 context NAR encoder 处理 context audio tokens → 交替层 cross-attention | 模态分离清晰; seen speaker SSIM 最高 | 对 unseen speaker 过拟合训练说话人 [§3.2] |

作者发现 Decoder Context 方案泛化最好 [agent 解读: 因为不引入额外编码瓶颈,decoder 的 self-attention 可以直接在 context 和 target 之间建立 token-level 的对应],因此最终的大模型 (1.1B multilingual) 采用此架构 [§3.5]。

### 关键设计选择

#### 1. 偏好对齐: DPO 和 RPO [§2.4]

**为什么不用 ground-truth 作 chosen?** [论文原文] SpeechAlign (Zhang et al. 2024) 使用 ground-truth audio 的 codec tokens 作为 chosen,但论文实验表明这导致 DPO 几乎可以瞬间区分 GT 和 generated (loss 几百步就降到接近零),在默认超参下模型退化 (CER > 90%)。即使精调超参 (beta=1.0, LR=1e-7) + early stopping,也不优于 baseline [Table 2, "GT as Chosen"]。原因是 GT 和模型生成的 token 分布本质不同——GT tokens 来自真实语音的 codec 编码,而模型生成的 tokens 来自自回归采样,两者分布 gap 太大 [§2.4] [论文原文]。

**解决方案: 从模型自身采样构建偏好对** [§2.4]:
1. 准备 58K text-audio 对 (含 800 条 LLM 生成的 challenging text: 重复词/数字/绕口令 + 50K 标准训练集文本)
2. 对每个输入采样 P=6 个输出 (top-k=80, temperature=0.7)
3. 用 Parakeet-TDT 1.1B ASR 模型计算 CER, 用 TitaNet-Large SV 模型计算 SSIM
4. **Pareto 最优排序**: 多目标 (CER↓, SSIM↑) 的非支配排序,同 rank 内按 CER 优先 (更重视可懂度) [Appendix A]
5. DPO: 取 rank 最高 vs 最低; RPO: top-2 vs bottom-2 全组合,丢弃任一维度 chosen 不优于 rejected 的对 [§2.4]

**RPO 的优势** [§2.4, §3.3]: RPO 引入了 reward gap 缩放因子,不把所有 chosen-rejected 对同等对待,而是按 reward 差距成比例缩放 loss。实验中 RPO 对 beta 超参和训练步数比 DPO 更鲁棒,且在非高对比度偏好对上也能有效工作 [论文原文]。

#### 2. Classifier-Free Guidance 用于自回归 token 预测 [§2.5]

**与已有 CFG 应用的区别** [论文原文]: 已有 CFG for LLM-based TTS 只 dropout text-independent 条件 (Darefsky et al. 2024, 即 Parakeet)。Koel-TTS 同时 dropout text 和 context audio (概率 10%),使模型学会无条件生成 [§2.5]。

推理时的 logit 插值:
```
l_cfg = gamma * l_c + (1 - gamma) * l_u
```
其中 gamma >= 1 控制引导强度。实验 sweep gamma 在 1~3 之间,选定 gamma=2.5 为最优 [§3.4, Fig 4]。

**CFG 的独特价值** [论文原文]: CFG 不需要任何额外微调即可获得显著提升,且可以与偏好对齐叠加使用——在偏好对齐后的模型上再用 CFG 可进一步提升所有指标 [Table 2]。代价是推理时 effective batch size 翻倍 (需同时计算 conditional 和 unconditional logits) [§2.5]。

#### 3. 单调对齐机制 [§2.3]

使用 cross-attention biasing (2D beta-binomial prior) + CTC loss 鼓励 text-audio 单调对齐 [§2.3]。Prior 在前 10K steps 应用,然后线性退火到 15K steps 关闭——因为推理时无法使用 prior,退火保证训练稳定性 [论文原文]。CTC loss 通过计算所有可能单调路径的似然来鼓励有效的单调采样 [§2.3]。

### 训练策略

- 基础训练: 16x A100, global batch 256, Adam LR 1e-4 + exponential decay (factor 0.998/1000 steps), ~200K steps (~40h) [Appendix E]
- Decoder: 12 layer, hidden 768, FFN 3072, causal conv kernel=3
- Encoder: 6 layer, non-causal, 同 decoder 规格
- 偏好微调: 最多 4000 mini-batch iterations, batch=64 pairs, LR 2e-7, beta=0.01, eta=1.0, 选 validation loss 最低的 checkpoint [§3.3]
- 大模型 (1.1B multilingual): hidden 1536, FFN 6144, decoder 16 layer, 32x A100, ~150K steps [Appendix E]

## 实验

| 指标 | Koel-TTS 380M (EN) | Koel-TTS 1.1B (Multi) | F5-TTS | E2-TTS | XTTS-v2 | VALL-E-X | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CER (%) ↓ | 0.55 | 0.63 | 1.23 | 1.29 | 0.99 | 6.65 | 0.80 | LibriTTS test-clean unseen | [Table 3] |
| WER (%) ↓ | 1.41 | 1.42 | 2.55 | 2.66 | 2.09 | 11.28 | 1.83 | LibriTTS test-clean unseen | [Table 3] |
| SSIM ↑ | 0.726 | 0.740 | 0.834 | 0.848 | 0.680 | 0.679 | 0.771 | LibriTTS test-clean unseen | [Table 3] |
| MOS ↑ | 4.054 | 4.058 | 3.930 | 3.889 | 3.715 | 3.532 | 3.937 | LibriTTS test-clean unseen | [Table 3] |
| SMOS ↑ | 3.826 | 3.848 | 3.785 | — | 3.786 | 3.229 | 3.709 | LibriTTS test-clean unseen | [Table 3] |

**偏好对齐消融 (Multi Encoder, unseen speakers)** [Table 2]:

| 配置 | CER (%) ↓ | SSIM ↑ | Squim-MOS ↑ |
| --- | --- | --- | --- |
| Baseline | 2.56 | 0.601 | 4.318 |
| +RPO | 0.79 | 0.641 | 4.389 |
| +DPO | 0.62 | 0.645 | 4.402 |
| +DPO (GT as Chosen) | 2.67 | 0.522 | 4.281 |
| +CFG (gamma=2.5) | 0.69 | 0.653 | 4.415 |
| +RPO+CFG | 0.51 | 0.674 | 4.392 |
| +DPO+CFG | 0.58 | 0.678 | 4.417 |

关键发现:
1. **DPO/RPO 均大幅改善可懂度和说话人相似度**,即使偏好数据中不含新说话人,unseen speaker 的 SSIM 也显著提升 [§3.3] [论文原文]
2. **GT-as-Chosen 明显失败**: unseen SSIM 从 0.601 下降到 0.522,证实了从模型自身分布采样构建偏好对的必要性 [Table 2]
3. **CFG 无需微调即可大幅提升**: baseline + CFG 的 CER/SSIM/MOS 已接近偏好对齐后的水平 [Table 2]
4. **偏好对齐 + CFG 叠加最优**: 所有指标上,DPO+CFG 或 RPO+CFG 达到最佳 [Table 2]
5. **自然度 "搭便车"**: 偏好对齐虽未显式优化 MOS/Squim-MOS,但自然度也随之提升——论文认为 CER 和 SSIM 是良好的人类偏好代理 [§3.3] [论文原文]

## 局限性

1. **SSIM 不及 flow-matching 系统**: F5-TTS (0.834) 和 E2-TTS (0.848) 在自动化 speaker similarity 上远超 Koel-TTS (0.740),虽然人工评估 (SMOS) 上 Koel-TTS 更优。论文将差异归因于 SSIM 偏重 timbre 而人类更关注 style+accent [§3.6] [论文原文]——但这也意味着 Koel-TTS 的 timbre 保持确实弱于 flow-matching 方案 [agent 解读]
2. **训练数据规模限制**: 18-21K 小时 vs F5-TTS/E2-TTS 的 100K+ 小时。虽然 Koel-TTS 以小数据量达 SOTA 可懂度,但 SSIM 差距可能部分源于数据量 [§3.6] [agent 解读]
3. **challenging text 上仍有空间**: 重复词等难句上 CER 仍有 4-5% (偏好对齐+CFG 后降至 ~4.7%) [Table 6],论文提到可能需要推理时 monotonic alignment 策略进一步改善 [Appendix F]
4. **CFG 推理开销翻倍**: 需同时计算 conditional 和 unconditional logits,effective batch size 翻倍 [§2.5]
5. **Decoder Context 架构的 context 长度限制**: 论文固定使用 5 秒 context slice,未讨论更长/更短 context 的影响,也未与 streaming 场景结合

## 点评

**优势**:
- 方法论清晰,消融完整: 三种架构 x DPO/RPO/CFG 的全组合实验提供了有信服力的证据
- "GT-as-Chosen 失败"的发现有重要参考价值——明确否定了 SpeechAlign 式的 ground-truth 偏好构建,对后续 TTS 偏好对齐研究有指导意义
- Pareto 最优排序处理多目标偏好是巧妙的工程选择,比简单加权更合理
- 以 18K 小时数据超越使用 100K+ 小时数据的模型 (在可懂度和人工 MOS 上),数据效率令人印象深刻

**不足**:
- 缺少 SEED-TTS-Eval benchmark 对比,而这是当前 zero-shot TTS 的主流评测集
- 未讨论与 CosyVoice 3 DiffRO 的关系——两者都在 2025 年发表,都做 TTS 后训练对齐,但路线不同 (sequence-level DPO/RPO vs token-level DiffRO)
- 评测 ASR 模型 (Parakeet-TDT) 与偏好训练 reward 模型相同,存在指标偏向性风险 [agent 解读]

## 可复用的 idea

1. **Pareto 最优排序构建多目标偏好对** [§2.4]: 当有多个不可通约的 reward 时 (如 CER 和 SSIM),用 Pareto front 而非简单加权来排序样本。可推广到任何多目标 RL/RLHF 场景。

2. **从模型自身采样而非用 GT 构建偏好数据**: 论文明确证实了 GT-as-chosen 的失败,并给出了原因——分布不匹配使 DPO 过拟合。这对所有 codec LM 的偏好对齐都有警示意义。

3. **CFG 对 AR token 预测模型同样有效**: CFG 不限于 diffusion/flow 模型,对自回归 token 预测同样可用。关键是同时 dropout text 和 audio 条件,使模型学会"什么都不给"时的 unconditional 行为。无需额外微调,纯推理时技巧。

4. **RPO > DPO 的稳健性**: 当偏好对的 quality gap 不均匀时 (现实中几乎总是如此),RPO 通过 reward gap 缩放提供更稳定的训练。实际操作中可先试 RPO 再考虑 DPO。

5. **CER + SSIM 作为人类偏好的代理**: 论文发现仅优化 CER 和 SSIM 就能"免费"获得自然度 (MOS) 提升,说明可懂度和说话人相似度是 TTS 质量的良好代理指标。

---

> [!review] 审阅状态 (2026-06-03, agent)
> **结论: pass-with-fixes** | 2 issues (0 high, 1 medium, 1 low)
> - (medium) concepts 字段中 [[DifferentiableRewardOptimization]] 关联间接——本文用 DPO/RPO 而非 DiffRO,概念页虽覆盖但命名可能误导
> - (low) datasets 字段为空,论文实际使用 LibriTTS/HiFiTTS/MLS/CML
> 详见 `_review/Koel-TTS-review.yml`
