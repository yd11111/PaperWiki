---
type: paper
tier: deep
title: "Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training"
arxiv_id: "2601.03065"
source: "Sources/Fine-GrainedContrastiveAudio-Text.pdf"
authors: [Yifan Yang, Bing Han, Hui Wang, Wei Wang, Ziyang Ma, Long Zhou, Zengrui Jin, Guanrou Yang, Tianrui Wang, Xu Tan, Xie Chen]
year: 2026
venue: "arXiv preprint"
tags: [contrastive-learning, speech-text-retrieval, speaking-style, paralinguistic, dual-encoder, fine-grained, pretraining, dataset, CLAP, evaluation]
concepts: ["[[Audio-LanguagePretraining]]", "[[NaturalLanguageDescriptionforTTS]]", "[[Self-SupervisedSpeechRepresentation]]", "[[StyleTransferinTTS]]", "[[TTSEvaluation]]"]
models: []
tasks: []
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个待确认实体页: [[Audio-LanguagePretraining]], [[NaturalLanguageDescriptionforTTS]], [[TTSEvaluation]], [[StyleTransferinTTS]], [[Self-SupervisedSpeechRepresentation]], [[Emilia]])
> 自动生成,不保证完整覆盖所有相关知识。基于未确认概念页,仅供参考。
> 检索命中: [[Audio-LanguagePretraining]][待确认], [[NaturalLanguageDescriptionforTTS]][待确认], [[TTSEvaluation]][待确认], [[StyleTransferinTTS]][待确认], [[Self-SupervisedSpeechRepresentation]][待确认], [[Emilia]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: CLSP 属于 CLAP 家族 (Audio-Language Pretraining),但有两个根本性差异: (1) 监督信号从 coarse/general audio captions 转为 fine-grained speaking style descriptions; (2) 聚焦 speech 而非 general audio (环境声/音乐)。现有 CLAP 系模型 (LAION-AI CLAP, GLAP, ParaCLAP) 用粗粒度或任务特定标签训练,在 speaker-centric 风格建模上表现接近随机猜测。

**已有认知**: KB 中 Audio-LanguagePretraining 页已覆盖 CLAP 对比学习范式 (InfoNCE loss)、Two-Tower 架构、下游任务 (检索/分类/captioning)。NaturalLanguageDescriptionforTTS 页记录了 PromptTTS → Parler-TTS → InstructAudio → CapTalk 的 NL 描述演进,但主要关注 TTS 生成端,对"评估端"(用 NL 描述评估语音风格一致性) 覆盖较少。TTSEvaluation 页已纳入 LLM-as-Judge、GSRM、SpeechJudge、TTSDS2 等路线,但尚未收录基于 speech-text contrastive similarity 的风格评估路线。

**创新判断**: 本文在 CLAP 家族中的创新点在于 (1) FCaps 数据集用端到端 pipeline 替代级联标注,避免了离散标签到自由文本重写的信息丢失; (2) 多正例对比训练 (multi-positive InfoNCE) + 动态任务调度器实现跨粒度泛化; (3) 作为 TTS 风格评估工具,Pearson r=0.893/0.903 显著超越所有 baseline。

## 速查

> [!summary] 速查
> - **一句话**: 提出 FCaps (47k h/19M captions) 和 CLSP 模型 (724M),首次用端到端 fine-grained style captions 训练 speech-text dual encoder,在检索/分类/风格相似度打分上大幅超越已有 CLAP 模型
> - **路线**: 语音 → SPEAR-XLarge encoder → mean pooling → MLP + L2 norm → 共享嵌入空间 ← L2 norm + MLP ← [CLS] ← RoBERTa-base ← caption text; 两阶段课程训练 (InfoNCE → multi-positive InfoNCE + 动态调度)
> - **指标**: 全局检索 S→T R@1 45.6 (vs ParaCLAP 2.1), 细粒度检索 S→T R@1 68.1 (vs GLAP 4.6); 零样本情感 WA 57.2 (IEMOCAP); 人类相关性 Pearson r=0.893/0.903/0.886 (Intrinsic/Situational/Fusion) [Table 3-5]
> - **可借鉴**: 端到端标注 pipeline (Qwen3-Omni captioner + agentic verification); 多正例对比训练处理一对多 speech-caption 关系; 动态任务调度器平衡粗/细粒度; CLSP 可直接作为 TTS style evaluation metric
> - **局限**: 仅英语; SPEAR 编码器限制了非英语泛化; 评估数据集有限 (ParaSpeechCaps test set 仅 241 clips); 零样本分类仍不够强 (IEMOCAP UA 56.1%)

## 核心问题

现有 speech-text 对比预训练模型 (CLAP 系) 用粗粒度 captions 或任务特定标签训练,无法捕捉说话风格的 fine-grained 变化 (语调轮廓、韵律节奏、情感变化等时间动态)。同时,现有 speech style-captioned 数据集要么规模小 (CapSpeech 33.6k h),要么用级联 pipeline (先标离散标签→LLM 重写),引入信息丢失和语义失配。

本文提出两个互补贡献: (1) FCaps 数据集通过端到端 annotation pipeline 直接从音频生成 fine-grained captions; (2) CLSP 模型通过 fine-grained + multi-granular 对比训练学习统一的 speech-text 表征。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CLSP 采用 CLAP 经典的 dual-encoder 架构 [§4.1, Fig 3]:

- **Speech encoder**: SPEAR-XLarge (599M 参数),一个 unified self-supervised speech/audio 表征模型 (Yang et al., 2025c),从最后一层提取 frame-level 表征,经 mean pooling → MLP projection → L2 normalization 得到 speech embedding $\mathbf{s} \in \mathbb{R}^d$
- **Text encoder**: RoBERTa-base (125M 参数),接受变长 caption (最长 512 tokens),取 final-layer [CLS] hidden state → MLP projection → L2 normalization 得到 text embedding $\mathbf{t} \in \mathbb{R}^d$
- **Total**: 724M 参数

[agent 解读] 选择 SPEAR 而非 HuBERT/WavLM 的原因可能是 SPEAR 同时覆盖 speech 和 general audio,在捕捉 paralinguistic cues (音色、情感、口音) 上更全面。选择 RoBERTa-base 而非更大的 LLM 可能是为了保持推理效率,因为 CLSP 的一个目标场景是大规模检索/评估。

### FCaps 数据集构建

**Caption 分类法** [§3.1]:
- **Global captions**: 话语级无时序的说话人属性总结 (年龄、性别、口音等 intrinsic traits + 语速、情感、表现力等 situational traits)
- **Fine-grained captions**: 包含时序结构和叙事逻辑的详细描述,追踪话语内的风格变化 (语调转折、韵律模式、强调方式、非语言发声等)

**端到端标注 pipeline** [§3.2, Fig 2]:

1. **Detailed Captioner**: Qwen3-Omni-30B-A3B-Captioner,直接从音频生成详细 caption。通过 user prompt 约束模型只描述说话人声音特征,抑制内容转写和环境描述 [论文原文,§3.2]
2. **Multi-positive captioning**: 对同一段语音用不同 random seed 生成多条 caption,产生语义一致但词汇/叙事角度不同的正例对 [论文原文,§3.2] — 这些 caption 都 grounded 在同一音频信号上,比纯文本 LLM 重写更可靠
3. **Agentic verification**: Qwen3-30B-A3B-Thinking-2507 作为 verification agent,按 checklist 逐条检查每条 caption — 过滤含背景噪声描述、环境音质评价、内容转写、缺失特征等 failure modes 的 caption [论文原文,§3.2]

[agent 解读] 端到端 pipeline 的核心优势是避免了级联方法的信息瓶颈: 级联方法先将连续的 paralinguistic 信息压缩为有限离散标签,再由 LLM 重写为自由文本 — 两步都会丢失信息。端到端方法让 multimodal captioner 直接从音频感知并描述,保留了更完整的 paralinguistic 细节。

**FCaps-Emilia** [§3.3]:
- 来源: Emilia 语音生成数据集 (He et al., 2024)
- captioner 跑 5 次 → verification agent 保留 1 条/utterance
- 规模: 18,131,371 条 fine-grained captions, 46,787 小时
- 不含 global captions

**FCaps-PSCBase** [§3.4]:
- 来源: PSC-Base (Diwan et al., 2025) — 包含 EARS, Expresso, VoxCeleb 的音频 + 人工标注的 style tags
- PSC-Base 原有的 captions 作为 global captions, 经规则化归一化去除 LLM 重写 artifacts
- captioner 跑 20 次 → verification → 保留 5-14 条/utterance 作为 fine-grained multi-positive views
- 规模: 140,602 条 global + 930,917 条 fine-grained, 267 小时

### 关键设计选择

**1. 两阶段课程训练** [§4.2]:

**Stage One** — 标准对称 InfoNCE [Eq 1]:
- 每条语音配一条 fine-grained caption
- 拉近配对 (speech, text),推远非配对
- 1.2M steps, 主要建立 fine-grained alignment

**Stage Two** — 多正例对称 InfoNCE [Eq 2-4]:
- 每条语音配两条 tokenized caption $\mathbf{y}_i, \hat{\mathbf{y}}_i$
- 构造 soft target distribution: $D_{i,j}$ 对 primary caption 分配权重 $\lambda$, secondary caption 分配 $1-\lambda$ ($\lambda=0.5$)
- 用 cross-entropy (CE) 替代 hard one-hot 目标
- Loss 是两个方向 (audio→text, text→audio) 的平均 [Eq 4]
- 4k fine-tuning steps

[论文原文] 设 $\lambda=0.5$ 基于 ablation (Appendix G.2); text→audio 方向每个 text embedding 对应单一 speech,所以用标准 one-hot target $D'$ [Eq 3]。

**2. 动态任务调度器** [§4.2]:
- Task 1: 配对 (global caption, fine-grained caption) — 鼓励跨粒度泛化
- Task 2: 配对 (fine-grained caption A, fine-grained caption B) — 增强 fine-grained 语义一致性判别
- 训练 step $t$ 时,Task 1 以概率 $p_t$ 采样,Task 2 以 $1-p_t$
- Dynamic scheduler: $p_t = \max(p_{\min}, p_0 - \frac{t}{T}(p_0 - p_{\min}))$ — 从 $p_0=0.95$ 线性降至 $p_{\min}=0.50$,过渡周期 $T=10000$ [Eq 5]

[agent 解读] 这个设计的直觉是: 先重点建立 global↔fine-grained 的跨粒度桥梁 (Task 1 为主),再逐步增加 fine-grained↔fine-grained 的判别能力 (Task 2 比重上升)。这避免了模型一开始就被 hard negative (两条语义接近但角度不同的 fine-grained captions) 困扰。

**3. 为什么需要两阶段** [§4.2, Appendix G.1]:

[论文原文] Table 8 的 ablation 显示: 仅 Stage 1 在 fine-grained retrieval 上强但 global retrieval 弱; 仅 Stage 2 两个都弱; 两阶段结合在所有任务上最优。

[agent 解读] Stage 1 用大规模数据 (FCaps-Emilia, 46k h) 建立 fine-grained alignment 基础; Stage 2 用 multi-positive supervision (FCaps-PSCBase 的多条 views) 做精调,引入 global↔fine-grained 跨粒度信号。这种 curriculum 也缓解了 multi-positive loss 对初始化敏感的问题。

### 训练策略

- 硬件: 8x NVIDIA A100 80GB GPU
- Batch: 每 GPU 800 秒语音
- Stage 1: 1.2M steps, ScaledAdam optimizer, Eden scheduler, peak LR 0.045 (speech encoder) / 0.001 (text encoder)
- Stage 2: 4k steps, 同上

## 实验

| 指标 | CLSP | 最强 Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Global S→T R@1 | 45.6% | ParaCLAP 2.1% | ParaSpeechCaps test | [Table 3] |
| Global T→S R@1 | 40.3% | LAION-CLAP 3.3% | ParaSpeechCaps test | [Table 3] |
| Global S→T mAP@10 | 58.7% | GLAP 3.4% | ParaSpeechCaps test | [Table 3] |
| Fine-grained S→T R@1 | 68.1% | GLAP 4.6% | ParaSpeechCaps test | [Table 3] |
| Fine-grained T→S R@1 | 67.2% | LAION-CLAP 1.2% | ParaSpeechCaps test | [Table 3] |
| Zero-shot Emotion WA/UA | 57.2/56.1 | ParaCLAP 46.1/46.5 | IEMOCAP | [Table 4] |
| Zero-shot Gender WA/UA | 100/100 | ParaCLAP 99.2/99.2 | RAVDESS | [Table 4] |
| Zero-shot Age WA/UA | 40.6/44.5 | Auden-Voice 38.5/— | CREMA-D | [Table 4] |
| Human Corr (Intrinsic) r/rho/tau | 0.893/0.858/0.668 | LAION-CLAP 0.679/0.664/0.467 | ParaSpeechCaps | [Table 5] |
| Human Corr (Situational) r/rho/tau | 0.903/0.878/0.694 | ParaCLAP 0.323/0.330/0.232 | ParaSpeechCaps | [Table 5] |
| Human Corr (Fusion) r/rho/tau | 0.886/0.858/0.670 | LAION-CLAP 0.588/0.597/0.425 | ParaSpeechCaps | [Table 5] |
| E2E vs Cascaded Correctness | 4.42 | 3.30 (cascaded) | FCaps-Emilia sample | [Table 2] |
| E2E vs Cascaded Coverage | 4.55 | 3.10 (cascaded) | FCaps-Emilia sample | [Table 2] |
| E2E vs Cascaded Naturalness | 4.92 | 4.15 (cascaded) | FCaps-Emilia sample | [Table 2] |

**检索任务的核心发现** [§5.2.2]: Baselines (LAION-AI CLAP, GLAP, ParaCLAP) 在 global 和 fine-grained 检索上接近随机猜测 (R@1 < 5%),而 CLSP 的 R@1 达 45.6%/68.1%。这个巨大差距说明: 用 coarse captions 训练的模型完全无法捕捉 speaking style 的 fine-grained 差异 [论文原文]。

**零样本分类** [§5.2.3]: 在 emotion/gender/age 三个维度上 CLSP 全面超越 baselines。Gender 达到 100% accuracy (RAVDESS),表明 fine-grained style captions 中固有的 speaker trait 信息被模型有效学习 [agent 解读]。

**人类相关性** [§5.2.4]: 20 名语音处理专家参与主观评测,CLSP 在所有维度上 Pearson/Spearman/Kendall 系数均大幅领先。尤其 Situational Traits (语速、情感、表现力等) 的 Pearson r=0.903,而最佳 baseline ParaCLAP 仅 0.323 [Table 5] — 说明 CLSP 对动态说话风格的建模能力是质的飞跃。

**Ablation 关键发现** [Appendix G]:
- Multi-stage: 两阶段缺一不可,单 Stage 2 表现最差 (Avg 36.2 vs 67.1) [Table 8]
- $\lambda=0.5$ 最优 (Avg 67.1 vs 65.2@0.3) [Table 9]
- Dynamic scheduler ($p_0$=0.95, $p_{\min}$=0.50, T=10000) 优于所有 static mixtures [Table 10]

## 局限性

1. **仅英语**: SPEAR 编码器和 FCaps 数据集均限于英语,多语言泛化未验证 [论文原文,Limitations]
2. **Paralinguistic 数据覆盖有限**: 罕见口音、极端情感表达、高表现力说话风格的数据稀疏 [论文原文]
3. **评测数据集规模小**: 检索评测仅用 ParaSpeechCaps test set 的 241 clips;主观评测仅 30 clips/category x 3 categories [§5.2.2, §5.2.4]
4. **零样本分类仍有差距**: IEMOCAP 4-class emotion UA 56.1%,距离 supervised SOTA 仍有较大差距 [Table 4]
5. **FCaps-Emilia 无 global captions**: 46k h 数据仅有 fine-grained 标注,可能限制模型在 global-level 任务上的表现 [§3.3]

## 点评

**优势**:
- 填补了 CLAP 家族在 fine-grained speaking style 建模上的空白。检索和人类相关性的巨大优势 (从近随机到 45-68% R@1; Pearson r 从 0.32 到 0.90) 不是增量改进,而是范式转变 — 说明用正确的监督信号训练比模型架构创新更关键
- 端到端标注 pipeline 在 correctness/coverage/naturalness 三维上全面超越级联方法 (Table 2, Fig 4),提供了 speech style captioning 的新标准做法
- Multi-positive contrastive training 的设计巧妙地处理了 speech-caption 的一对多关系,比硬负例挖掘更自然
- CLSP 作为 TTS evaluator 的潜力很大 — 比 LLM-as-Judge 成本低得多,比 MOS/WER/SIM 捕捉的信息维度更丰富

**不足**:
- 仅英语是核心限制,考虑到 CLSP 的一个重要应用场景是 TTS 评估,而现代 TTS 系统普遍支持多语言
- 评测基准的多样性和规模不足 — 241 clips 的检索测试集和 30 clips 的主观评测难以充分验证泛化能力
- 缺少与 LLM-as-Judge (如 SpeechLLM-as-Judges) 的直接对比,无法判断 CLSP 是否真的比 LLM-based evaluator 更优
- 论文未讨论 CLSP score 的 calibration — 即 similarity score 的绝对值是否有可解释的含义,还是只能用于 ranking

## 可复用的 idea

1. **端到端 speech captioning pipeline**: Qwen3-Omni captioner + user prompt 约束 + agentic verification 的三步组合,可迁移到其他 speech/audio annotation 任务 (如 emotion captioning, prosody annotation)
2. **Multi-positive contrastive training**: 对同一 anchor 生成多条语义等价但表述不同的正例,适用于任何 one-to-many 对应的对比学习场景 (如 TTS 多样性评估)
3. **Dynamic task scheduler**: 从 global→fine-grained 渐进的 curriculum 策略,可应用于其他多粒度表征学习
4. **CLSP 作为 TTS 风格评估 metric**: 计算 CLSP(generated_speech, target_style_description) 作为 instruction-following TTS 的自动化评估指标,替代昂贵的 LLM-as-Judge
5. **Caption taxonomy (global vs fine-grained)**: 两级 caption 体系可应用于 TTS 训练数据标注 — global 描述稳定属性,fine-grained 描述时序动态

## 审阅

> [!review] 审阅 (pending)
> **结论**: pending
> 
> 待独立审阅 agent 执行。
