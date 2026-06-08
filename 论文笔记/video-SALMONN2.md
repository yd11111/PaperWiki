---
type: paper
tier: deep
title: "video-SALMONN 2: Caption-Enhanced Audio-Visual Large Language Models"
arxiv_id: "2506.15220"
source: "Sources/video-SALMONN2.pdf"
authors: [Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zejun Ma, Chao Zhang]
year: 2025
venue: "Preprint (under review)"
tags: [audio-visual, speech-LM, multimodal, DPO, LoRA, video-captioning, video-QA, RLHF, av-LLM, data-annotation, preference-optimization]
concepts: ["[[ModalityAdaptationforSpeechLLM]]", "[[AudioUnderstanding]]", "[[Audio-LanguagePretraining]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[DifferentiableRewardOptimization]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: video-SALMONN 2 是 SALMONN 系列的第三代,从 [[论文笔记/SALMONN|SALMONN]] (音频 LLM, ICLR 2024) → [[论文笔记/video-SALMONN|video-SALMONN]] (av-LLM, ICML 2024) → video-SALMONN 2 (caption-enhanced av-LLM, 2025)。在 KB 分类中仍属于 latent-representation-based Speech-LLM integration 路线 [Speech-LLMIntegrationTaxonomy],但本代的核心创新从架构 (Q-Former 改进) 转向训练策略 (MrDPO + 数据自标注)。音频分支继续使用 Whisper-Large-v3 + window-level Q-Former [Whisper, ModalityAdaptationforSpeechLLM]。
>
> **已有认知**: KB 中 DifferentiableRewardOptimization 页面覆盖了 TTS 领域的 DPO 变种 (SpeechAlign, FPO, TKTO, GRPO 等),但 video-SALMONN 2 的 MrDPO 面向视频 captioning 而非 TTS,其核心创新 (multi-round reference refresh via LoRA proxy + gDPO loss) 与 TTS 领域的 iterative DPO (如 SpeechAlign 的 iterative self-improvement) 有方法论相似性但应用场景完全不同。AudioUnderstanding 页面将 SALMONN 列为 "Two Heads" LALM 的代表。
>
> **创新判断**: 与前代 video-SALMONN 的 MRC Q-Former 架构创新不同,video-SALMONN 2 将焦点从编码器设计转向后训练优化 (RL via DPO)。核心思路是: 高质量 captioning → 高质量 SFT 数据 → 通用 video QA 能力的正向循环。这一 "captioning 作为数据引擎" 的范式在 av-LLM 领域较新,此前 ShareGPT4Video 等使用 GPT-4 标注数据,而 video-SALMONN 2 通过 MrDPO 自我改进后的模型来标注。
>
> 检索命中: [[ModalityAdaptationforSpeechLLM]][待确认], [[AudioUnderstanding]][待确认], [[Audio-LanguagePretraining]][待确认], [[Speech-LLMIntegrationTaxonomy]][待确认], [[DifferentiableRewardOptimization]][待确认], [[Whisper]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过 MrDPO (multi-round DPO + LoRA proxy + gDPO loss) 持续优化视频 captioning 质量,再用优化后的 captioner 自标注高质量 SFT 数据,实现 captioning→QA 的能力迁移,在 7 个 video QA benchmark 上 SOTA
> - **路线**: Video frames → Visual Encoder → Visual Aligner + Audio → Whisper-Large-v3 → Audio Q-Former → interleaved synchronization → LLM backbone (LLaVA-OneVision / Qwen2.5-VL) + LoRA → text response; 训练: Audio Alignment → AV-SFT → MrDPO (6 rounds) → 数据自标注 → SFT 新模型
> - **指标**: Caption Total Error% 22.9 (vs GPT-4o 31.2, Qwen2.5-VL 39.2) [Table 1a]; Elo 1115 (vs GPT-4o 976) [Table 11]; Video-MME 73.4/79.7 (7B/72B) [Table 3]; MLVU 73.6 (7B, SOTA) [Table 3]
> - **可借鉴**: (1) MrDPO 的 LoRA proxy 机制: 每轮 merge 旧 LoRA 到 backbone + 初始化新 LoRA,避免 reference staleness; (2) 原子事件分解的 caption 评估方法 (missing rate + hallucination rate); (3) "captioner 作为数据引擎" 的自循环范式
> - **局限**: MrDPO 仅优化 captioning 不直接提升 QA; 依赖 GPT-3.5/4o 做 caption 评估; atomic event 分解的质量依赖外部 LLM; 训练资源仍然可观 (32x H800s for SFT)

## 核心问题

**想解决什么**: 视频 captioning 是 av-LLM 能力的基石 -- 高质量 caption 不仅直接服务描述任务,更通过 SFT 数据的质量改善间接提升 video QA。然而当时 (2025) 的开源 av-LLM 生成的 caption 要么缺失信息 (missing events),要么产生幻觉 (hallucinated events),且缺乏可靠的自动评估指标来驱动 RL 优化 [§1]。

**为什么难**: (1) 视频 captioning 的评估指标不可靠: BLEU/ROUGE-L 无法评估长详细 caption 的完整性和准确性 [§3.3.1]; (2) 标准 DPO 的 reference staleness: 单轮 DPO 中 reference model 固定,随训练推进 policy 与 reference 差距增大,导致优化停滞 [§3.3.2]; (3) 丢弃音频: 大多数视频模型忽略音频流,丢失语音和声音事件信息 [§1]。

**怎么切入**: (1) 用 atomic event 分解将 caption 评估转化为 LLM 可判断的子任务 (missing/incorrect/hallucinated events); (2) 提出 MrDPO,每轮通过 LoRA merge + re-init 刷新 reference policy; (3) 用 gDPO loss (DPO + ground-truth SFT) 稳定多轮训练; (4) 用 MrDPO 训练后的模型自标注高质量 caption 数据,SFT 新模型将 captioning 增益迁移到 QA。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

模型沿用 SALMONN 系列的双分支架构 [§3.1, Fig 1]:

- **视觉分支**: 视频帧 → Visual Encoder → Visual Aligner → visual tokens
- **音频分支**: 音频波形 → Whisper-Large-v3 Encoder → segment-level positional embedding → Audio Q-Former (window-level, 0.5s 窗口) → audio tokens
- **融合**: 视觉和音频 tokens 按时间同步交错排列 (interleaved synchronization),同一时间段的视觉帧 token 和对应音频 token 相邻放置
- **LLM backbone**: 融合后的 AV token 序列 + text prompt tokens → LLM → text response

与前代 video-SALMONN 的差异: 去掉了 BEATs 编码器 (仅保留 Whisper),去掉了 MRC Q-Former (回到标准 window-level Q-Former),visual backbone 从 InstructBLIP 换为 LLaVA-OneVision / Qwen2.5-VL 等更强的视觉 LLM [agent 解读]。

### 关键设计选择

#### 1. Atomic Event 分解评估 [§3.3.1]

**为什么需要新指标**: BLEU/ROUGE-L 对长详细 caption 无效,因为它们基于 n-gram 重叠,无法评估语义层面的完整性和准确性 [论文原文]。

**怎么做**: 
1. 用 GPT-4o 将 ground-truth caption 分解为 atomic events 列表 (每个 event 是一个独立的可验证视听事实)
2. 对模型生成的 caption,用 GPT-3.5 判断哪些 ground-truth events 缺失 (missing)、哪些描述错误 (incorrect)、哪些 caption 中的 events 不在 ground-truth 中 (hallucination)
3. Missing rate = #missing / #total_events; Hallucination rate = (#incorrect + #hallucinated) / #total_events; Total error rate = sum of both

**用于 DPO**: 对每个视频生成 caption pair,total error rate 更低的作为 preferred sample [§3.3.1]。

#### 2. MrDPO: Multi-round DPO with LoRA Proxy [§3.3.2, Fig 2]

**为什么不能直接多轮 DPO**: 标准 DPO 的 reference model 在训练开始时固定。多轮训练中,policy 持续更新但 reference 不变,两者差距越来越大,导致 KL 约束失效、优化停滞 [论文原文]。直接在每轮用上一轮的完整模型做 reference 又会导致 reference 过度偏向最近的更新,丢失原始 SFT model 的知识 [agent 解读]。

**MrDPO 的三步循环** (每轮 t):

1. **Merge**: 将上一轮训练好的 LoRA Δ_{t-1} merge 进 LLM backbone: W_t = W_{t-1} + α A_{t-1} B_{t-1} [Eqn. 2]
2. **Re-init**: 在 merged backbone 上新建一个全新的 LoRA proxy Δ̃_t (随机初始化)
3. **Train**: 只训练新 LoRA Δ̃_t,backbone (含所有历史 merged LoRA) 保持冻结

**Reference 刷新机制**: reference model = merged backbone Λ_t (不含新 LoRA),policy model = Λ_t + Δ̃_t。每轮开始时 reference 自动更新为包含所有历史 LoRA 的 backbone,避免 reference staleness [论文原文]。

**为什么用 LoRA proxy 而非直接 fine-tune**: (1) LoRA proxy 每轮重新初始化,避免在过时的参数空间中停滞 [Fig 3b]; (2) merge 操作渐进式强化 backbone,使 reference model 越来越强; (3) 始终只有一个 active LoRA,计算效率高 [论文原文]。

#### 3. gDPO Loss [Eqn. 3]

**问题**: 纯 DPO loss 多轮训练后容易陷入局部最优 [论文原文]。

**解法**: gDPO = 标准 DPO loss + λ × ground-truth caption 的 cross-entropy loss:

$$\mathcal{L}_{\text{gDPO}} = -\mathbb{E} \left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_{\text{win}}|x)}{\pi_{\text{ref}}(y_{\text{win}}|x)} - \beta \log \frac{\pi_\theta(y_{\text{lose}}|x)}{\pi_{\text{ref}}(y_{\text{lose}}|x)}\right)\right] + \lambda \mathbb{E} \log \pi_\theta(y_{\text{gt}}|x)$$

与 Iterative RPO 的区别: gDPO 使用 ground-truth caption 的 SFT loss 而非 chosen sample 的 SFT loss 来稳定训练 [论文原文]。λ = 0.1 效果最好 [Table 7, Appendix E]。

#### 4. Captioning→QA 迁移 [§3.4]

**发现**: MrDPO 显著提升 captioning 质量,但不直接提升 video QA,因为 QA 能力主要由 AV-SFT 阶段决定 [论文原文]。

**解法**: 用 MrDPO 训练后的模型重新标注 100k 视频的 caption → 用新 caption + 原有 QA 数据做 SFT → 得到 video-SALMONN 2+ 系列。这实现了 captioning 能力到 QA 能力的间接迁移: 更好的 caption → 更好的 SFT 数据 → 更强的视频理解 [论文原文]。

### 训练策略

三阶段 pipeline [§3.2, Fig 2]:

1. **Audio Modality Alignment**: 冻结 visual LLM + audio encoder,只训练 audio aligner。用 LibriSpeech-960h (ASR) + AudioCaps (captioning)。训练成本: 32x H800s, 3 hours [Table 4]。
2. **Audio-Visual SFT**: 冻结 encoders + LLM backbone,训练 audio aligner + LoRA。用 FineVideo + CinePile + LLaVA-Video-178k (~13k 视频)。训练成本: 32x H800s, 14 hours [Table 4]。
3. **MrDPO**: 6 轮 gDPO 训练,每轮仅训练 LoRA proxy。Round 1-5 用 8x H800s; Round 6 (加入 QA 数据) 用 32x H800s [Table 4]。

**模型家族**:
- video-SALMONN 2 (7B): 基于 LLaVA-OneVision-7B,1 fps,最高 110 帧
- video-SALMONN 2_{F-16} (7B): 基于 F-16,16 fps,最高 1760 帧
- video-SALMONN 2+ (3B/7B/72B): 基于 Qwen 2.5-VL 系列,10 fps,用自标注数据训练

## 实验

### 主要 Captioning 结果 [Table 1a]

| 指标 | video-SALMONN 2 (7B) | GPT-4o | Qwen2.5-VL (7B) | VideoLLaMA 3 (7B) | 出处 |
| --- | --- | --- | --- | --- | --- |
| Missing Rate ↓ | **10.0** | 17.0 | 21.9 | 44.9 | [Table 1a] |
| Hallucination Rate ↓ | **12.9** | 14.2 | 17.4 | 11.6 | [Table 1a] |
| Total Error ↓ | **22.9** | 31.2 | 39.2 | 56.5 | [Table 1a] |
| Elo Rating ↑ | **1115** | 976 | — | — | [Table 11] |

### MrDPO Ablation [Table 1b]

| 方法 | Total% ↓ | %Improve | 出处 |
| --- | --- | --- | --- |
| Visual only | 50.7 | — | [Table 1b] |
| + SFT | 41.8 | +17.6 | [Table 1b] |
| + DPO | 37.8 | +9.6 | [Table 1b] |
| + gDPO | 39.7 | -5.0 | [Table 1b] |
| + LoRA Proxy | 33.7 | +15.1 | [Table 1b] |
| + MrDPO (full) | **22.9** | +32.0 | [Table 1b] |

注: gDPO 单独加入后反而变差,但与 LoRA Proxy 结合后 MrDPO 的多轮训练显著获益 [Fig 3a]。gDPO 的 SFT 正则化从第 2 轮起才稳定发挥作用。

### Video QA 结果 [Table 3]

| 模型 | Video-MME | WorldSense | AVUT | Video-Holmes | DailyOmni | MLVU | LVBench | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| video-SALMONN 2 (7B)* | 67.4 | 48.6 | 65.6 | 40.7 | 66.3 | 68.3 | — | [Table 3] |
| video-SALMONN 2+ (7B)* | **73.4** | **50.9** | **69.5** | **46.9** | **71.8** | **73.6** | **49.7** | [Table 3] |
| Qwen2.5-VL (7B) | 65.1 | — | — | 27.8 | 40.7 | 70.2 | 45.3 | [Table 3] |
| GPT-4o | 71.9 | 42.6 | 56.6 | 42.0 | 56.5 | 64.6 | 30.8 | [Table 3] |
| video-SALMONN 2+ (72B)* | **79.7** | **56.5** | **72.2** | **57.8** | **79.4** | **80.4** | **55.5** | [Table 3] |
| Gemini-1.5 Pro* | 75.0 | 48.0 | 78.3 | 41.0 | — | — | 33.1 | [Table 3] |

\* 表示支持音频输入。video-SALMONN 2+ (7B) 在所有 benchmark 上超越同规模 visual-only 和 audio-visual 模型; 72B 版本在多数 benchmark 上超越 GPT-4o 和 Gemini-1.5 Pro。

### 数据标注迁移效果 [Table 2]

| 模型 | SFT 数据 | Video-MME Avg ↑ | 出处 |
| --- | --- | --- | --- |
| F-16 (7B) | 原始 | 65.0 | [Table 2] |
| F-16 (7B) | + 自标注 caption | **70.2** | [Table 2] |
| Qwen2.5-VL (72B) | 原始 | 73.3 | [Table 2] |
| Qwen2.5-VL (72B) | + 自标注 caption | **79.7** | [Table 2] |

### 模态贡献分析 [Table 9]

| 模型 | Miss% ↓ | Hall% ↓ | Total% ↓ | 出处 |
| --- | --- | --- | --- | --- |
| Visual Base | 23.3 | 27.4 | 50.7 | [Table 9] |
| + Visual only (V) | 21.4 | 14.1 | 35.4 | [Table 9] |
| + Audio only (A) | 62.6 | 22.0 | 84.0 | [Table 9] |
| + Audio-Visual (AV) | **10.0** | **12.9** | **22.9** | [Table 9] |

音频和视觉模态互补: 仅视觉会 miss 音频事件 (21.4 miss),仅音频视觉信息几乎完全缺失 (62.6 miss),AV 联合使两者显著下降。

## 局限性

1. **MrDPO 不直接提升 QA**: captioning 优化和 QA 能力之间存在间接迁移的 gap,需要额外的数据标注 + SFT 步骤 [§3.4]
2. **评估依赖外部 LLM**: atomic event 分解和评估均依赖 GPT-4o/3.5,引入评估噪声;论文用 Qwen3-4B 替代 GPT-3.5 后 Total% 相同 (33.7) [Table 5],但基线 LLM 的偏差仍然存在
3. **音频分支有限**: 音频编码器 (Whisper) 和 aligner 冻结,音频理解能力受限于 Whisper 和 SFT 数据质量;audio-only 模式几乎不可用 (Total% 84.0) [Table 9]
4. **训练成本**: 虽然 MrDPO 每轮只需 8x H800s / 2h,但完整 pipeline (alignment + SFT + 6 轮 MrDPO + 数据标注 + 新 SFT) 资源消耗不小
5. **caption benchmark 规模有限**: 仅 483 个视频,覆盖 14 类,可能不够多样化 [Table 8]
6. **未探索 reasoning**: 论文仅聚焦 captioning 和 QA,未涉及推理增强 (reasoning enhancement),而同组的 video-SALMONN-o1 (Sun et al., 2025) 专门探索此方向

## 点评

video-SALMONN 2 的核心价值不在于架构创新 (架构几乎直接借用 LLaVA-OneVision + SALMONN 音频分支),而在于提出了一个**完整的自我改进闭环**: 评估指标 (atomic events) → 优化方法 (MrDPO) → 数据引擎 (自标注) → 能力迁移 (captioning→QA)。这个 pipeline 思路比任何单一组件都更有启发性。

MrDPO 的 LoRA proxy 设计解决了 iterative DPO 中 reference staleness 和 training stagnation 两个实际工程问题,方法简洁且有效 (6 轮 MrDPO 将 Total% 从 41.8 降至 22.9)。gDPO 的 ground-truth SFT 正则化与 Iterative RPO 的 chosen SFT 正则化形成有趣对比 -- 使用 ground-truth 避免了 chosen sample 质量不稳定时对训练的干扰。

"captioner 作为数据引擎" 的思路尤其值得关注: 用 MrDPO 自标注的 caption 训练的新模型,在完全没有 MrDPO 训练的情况下就能显著超越原模型 (F-16: Video-MME 65.0→70.2)。这暗示数据质量可能比训练算法更重要 -- 好的 captioner 可以持续 bootstrap 整个生态。

## 可复用的 idea

1. **MrDPO 的 LoRA proxy merge-and-reinit 机制**: 适用于任何需要多轮 DPO 的场景 (包括 TTS RL)。与 TTS 中的 iterative self-improvement (如 SpeechAlign) 和 DiffRO 的 token-level 优化互补 -- MrDPO 提供了序列级 DPO 的稳定多轮训练范式。
2. **Atomic event 分解评估**: 将长文本评估转化为可验证子任务的思路可迁移到 TTS 评估 (如将音频描述分解为 atomic attributes: 内容准确性、说话人一致性、情感表达、韵律自然度等)。
3. **gDPO 的 ground-truth SFT 正则化**: 在 preference optimization 中加入 ground-truth 的交叉熵项,比 Iterative RPO 的 chosen-sample 正则化更稳定,可借鉴到 TTS DPO。
4. **Captioner→Data Engine→SFT 正循环**: "用优化后的模型标注更好的训练数据" 的范式在 TTS 中同样适用 -- 优化后的 TTS 系统可以生成更好的合成语音数据来训练下一代模型。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节完整覆盖 MrDPO 三步循环、gDPO loss、atomic event 评估,因果解释充分 |
> | 可信赖 | pass | 数字 claim 标注覆盖率高,指标名使用正确,无方向性错误 |
> | 可区分 | pass | 因果解释标注了 [论文原文]/[agent 解读],fact-inference 区分明确 |
> | 可定位 | pass | KB 谱系定位清晰 (SALMONN→video-SALMONN→v2),与 DiffRO 的跨域关联到位 |
> | 不污染 | pass | 本次不做反向更新,frontmatter 挂接合理,无 overclaim |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/video-SALMONN2-review.yml`

---

检索命中: [[ModalityAdaptationforSpeechLLM]], [[AudioUnderstanding]], [[Audio-LanguagePretraining]], [[Speech-LLMIntegrationTaxonomy]], [[DifferentiableRewardOptimization]], [[Whisper]] | 过滤: 全部 pending-review | 未命中但可能相关: 无
