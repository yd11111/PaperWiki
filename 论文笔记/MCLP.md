---
type: paper
tier: deep
title: "Evaluating and Rewarding LALMs for Expressive Role-Play TTS via Mean Continuation Log-Probability"
arxiv_id: "2601.22661"
source: "Sources/MCLP.pdf"
authors: [Yong Ren, Jingbei Li, Haiyang Sun, Yujie Chen, Cheng Yi, Yechang Huang, Hao Gu, Ye Bai, Xuerui Yang]
year: 2026
venue: "ICML 2026"
tags: [TTS, evaluation, reward, RL, GRPO, role-play, stylistic-consistency, LALM, multi-turn, metric]
concepts: ["[[LLM-based TTS]]", "[[TTS Evaluation]]", "[[Style Transfer in TTS]]", "[[Differentiable Reward Optimization]]", "[[Instruction-Guided Speech Synthesis]]", "[[Emotion Control in TTS]]"]
models: ["[[论文笔记/Step-Audio 2.5|Step-Audio-2]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文处于 TTS 评估 + RL post-training 两条线的交叉点。

1. **评估侧**: TTS 评估领域近年涌现多条新路线 — distributional evaluation (TTSDS/TTSDS2)、LLM-as-Judge (GSRM/SpeechJudge/TTS-PRISM)、instruction-following benchmark (InstructTTSEval)、prosody diversity (ProsodyEval)。但**风格一致性 (stylistic consistency)** 的客观评估仍是空白,尤其在多轮对话的 Role-Play 场景下。现有代理指标 (emotion classification, speaker similarity) 只捕捉风格的局部维度。MCLP 利用 LALM 的 ICL 能力,在 latent style space 直接度量风格连续性,填补了这一空白。

2. **RL post-training 侧**: GRPO 已被多个系统验证有效 — Multi-Reward GRPO (Tencent, 多维 reward)、TTS-1 (Inworld, WER+SIM+DNSMOS)、GRPO-TTS (USTC, CER+NLL)、RLAIF-SPA (属性级 AI 反馈)。但这些工作的 reward 主要覆盖 content fidelity 和 speaker similarity,**style reward 依赖离散情感分类** [待确认],不足以表达 Role-Play 场景中连续且上下文依赖的风格。MCLP 作为 dense、continuous 的 style reward,与 CER content constraint 结合后避免了 reward hacking。

3. **Style 控制侧**: 指令引导语音合成 (VoxInstruct, CosyVoice) 和情感控制 (EmoVoice, DiffRO-MTR) 已实现单句级风格/情感控制 [待确认],但**跨轮次风格一致性**尚未被系统解决。本文将 Role-Play TTS 形式化为 context-aware conditional generation,区分于 single-turn style control。

**已有认知**: LLM-based TTS 通过 in-context learning 从 prompt 隐式学习 speaker/style,GRPO 是 TTS post-training 的主流 RL 方法之一,但 reward hacking 是已知风险 (MCLP-only 消融中 CER 暴涨至 61% 即为典型案例)。

**创新判断**: MCLP 的核心创新在于将 LALM 的 continuation likelihood 重新定义为 style metric — 通过固定 transcript 消除 content 变量,使 likelihood 变化仅反映 style 差异。这一思路与 GSRM 的 "acoustic-feature-grounded CoT" 和 SpeechJudge 的 "pairwise preference" 路线完全不同,是第三条 style evaluation 路线。

> 检索命中: [[LLM-based TTS]]✓, [[TTS Evaluation]](pending-review), [[Style Transfer in TTS]](pending-review), [[Differentiable Reward Optimization]](pending-review), [[Instruction-Guided Speech Synthesis]](pending-review), [[Emotion Control in TTS]](pending-review) | 未命中但可能相关: Spoken Dialogue Evaluation

## 速查

> [!summary] 速查
> - **一句话**: 提出 MCLP — 利用 LALM continuation likelihood 量化风格一致性的可解释指标,同时作为 GRPO reward 显著提升 Role-Play TTS 的风格遵循
> - **路线**: 场景/角色/对话历史 → Step-Audio-2-mini SFT → GRPO (MCLP style reward + CER content gate) → 交错 TA4 tokens → token2wav
> - **指标**: MOS 3.576 (vs MiMo-Audio-7B 2.484, OV-InstructTTS 2.864); CER 1.130% (最低); MCLP -4.636 (最高); MCLP ∆>0.1 时 win rate>0.8 [Table 2, Fig 5]
> - **可借鉴**: (1) "固定 transcript + 重复" 的 continuation context 设计,消除 content 变量后 likelihood 仅反映 style; (2) gated hybrid reward (CER>τ 则 R=0) 防止 "expressive gibberish"; (3) 300 万小时 continuation pretraining 增强 LALM 的 style-aware ICL
> - **局限**: 仅中文验证 (英文仅跨语言初步实验); 依赖 Step-Audio-2 tokenizer (semantic tokenizer 偏向 style 而非 acoustic); 场景/角色描述由 LLM 自动生成可能有噪声; 开源代码但未开源模型权重

## 核心问题

Role-Play TTS (RP-TTS) 要求合成语音在多轮对话中保持与场景描述、角色设定一致的**风格连续性**。核心瓶颈有二:

1. **缺乏风格一致性的客观度量**: 现有指标 (CER, SIM) 衡量内容和音色,情感分类器只捕捉离散情感,无法表达连续且上下文依赖的 role-play 风格 [§1]
2. **SFT 泛化不足**: 仅靠 SFT 难以使模型在新场景/角色组合上保持风格一致,需 RL 对齐,但没有合适的 reward signal [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

三阶段 pipeline [Fig 1]:

1. **Continuation Model Pretraining** [Fig 1(a)]: 基于 Step-Audio-2-mini-Base (7B),在 300 万小时转写语音上训练 continuation 能力。训练格式: `{w1, y1, w2, y2, ..., wn, yn}`,loss 仅在 audio tokens 上计算,使模型学会从文本+前序语音上下文预测后续语音 [§4.1]

2. **SFT for RP-TTS** [Fig 1(b)]: 在 WenetSpeech-RP-TTS 数据集上 fine-tune,输入为 `(S, P, H<j, Ij)` (场景+角色+历史+当前指令),输出为交错的 TA4 token 序列。按 turn-level 分解训练样本 [§4.2, Eq.2]

3. **GRPO with Hybrid Reward** [Fig 1(c)]: 仅对最后一轮语音做 RL。每个 query 采样 G=8 rollouts,用 composite reward 计算 advantage [§4.3, Eq.3-7]

### 关键设计选择

**1. MCLP 的定义与直觉** [§4.1, Eq.1]

MCLP 度量 evaluation audio `z_eval` 与 ground-truth audio `z_gt` 的风格一致性:
- 构造 context: `H = [w, z_eval, w]` (transcript → eval audio → 重复 transcript)
- 计算 `z_gt` 的 mean log-probability conditioned on H
- WHY: 固定 transcript 后,likelihood 变化仅反映 `z_eval` 是否提供了与 `z_gt` 兼容的 speaking style [论文原文]。虽然从 `z_eval` 预测 `z_gt` 的顺序看似反直觉,但目的是利用 GT 固定长度实现自然归一化和跨候选公平比较 [§4.1]

**为什么选择 Step-Audio-2 作为 MCLP backbone?** 因为 Step-Audio-2 使用 semantic speech tokenizer,主要保留语义和风格信息而非声学细节。在 fixed-transcript 设定下,这一设计使指标偏向风格一致性而非声学相似性 [论文原文, §4.1]

**2. Gated Hybrid Reward** [§4.3, Eq.5-7]

- Style Reward: `R_style = MCLP(z_roll, z_gt) + C` (C=15.0 将 MCLP 移入正区间)
- Content Constraint: `R_content = λ * CER(ŵ, w)` (λ=10.0)
- Gating: CER > τ (τ=0.2) 时 R=0

WHY gating: 朴素单目标优化易导致 reward hacking — 仅优化 MCLP 使模型产生重复声学模式 (CER>50%),仅优化 CER 使语音平坦无表现力。Gating 创建了一个课程: 模型必须先满足内容约束才能获取风格奖励 [论文原文, §4.3]

**3. Turn-level RL Design** [§4.3]

GRPO 仅应用于最后一轮语音的生成,而非全部轮次。[agent 解读] 这避免了在所有轮次上做 rollout 的巨大计算开销,同时最后一轮已包含完整对话历史上下文,可充分评估风格一致性。

### 训练策略

**SFT** [§6.1]:
- Base: Step-Audio-2-mini-Base (7B)
- 1 epoch, batch 64, lr 1e-5 cosine decay, max seq 16384
- AdamW, β1=0.9, β2=0.95, weight decay 0.1, grad clip 1.0

**RL (GRPO)** [§6.1]:
- 1000 iterations, lr 1e-6, batch 128, G=8 rollouts/prompt
- Temperature 1.0, max decoding 1024 tokens
- KL coefficient β=0.001
- Reward: C=15.0, λ=10.0, τ=0.2
- 32x NVIDIA H800 GPUs

**Continuation Pretraining**: 300 万小时转写语音,仅训练 audio token prediction [§4.1]

### 数据集: WenetSpeech-RP-TTS

从 WenetSpeech YouTube Drama 子集构建 [§5, Table 1, Fig 3]:

| 阶段 | 音频数 | 时长 | 句子数 | 场景数 |
|------|--------|------|--------|--------|
| WenetSpeech 全集 | 83,503 | 12,542h | 17.9M | - |
| YouTube Drama | 17,253 | 5,343h | 8.3M | - |
| 最终过滤后 | 8,237 | 1,435h | 2.3M | 311,938 |

[Table 1]

- 场景分割: 静音>5s 或最大30s [§5.2]
- 场景描述: Qwen-VL-7B 生成 [§5.2]
- 角色描述: DeepSeek-R1 从完整剧集对话推断 [§5.2]
- 测试集: 200 个视频 hold out,900 scenes (每种轮次长度 2-10 各 100) [§5.3]
- RL 子集: 16,186 scenes (2-6 轮,最后轮>10字,非中性语音) [§5.3]

## 实验

| 指标 | 本文 (w/ hist) | MiMo-Audio-7B | GPT-Audio | Step-Audio-2-mini | OV-InstructTTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CER (%) ↓ | **1.130** | 10.605 | 11.974 | 3.276 | 7.188 | WenetSpeech-RP-TTS | [Table 2] |
| CAM++ ↑ | **0.724** | 0.699 | 0.636 | 0.629 | 0.669 | WenetSpeech-RP-TTS | [Table 2] |
| Emo2Vec ↑ | **0.917** | 0.902 | 0.875 | 0.864 | 0.900 | WenetSpeech-RP-TTS | [Table 2] |
| MCLP ↑ | **-4.636** | -4.753 | -4.849 | -4.829 | -4.768 | WenetSpeech-RP-TTS | [Table 2] |
| MOS ↑ | **3.576** (±0.045) | 2.484 (±0.058) | 1.915 (±0.048) | 1.856 (±0.045) | 2.864 (±0.087) | WenetSpeech-RP-TTS | [Table 2] |

**MCLP 与人类判断的对齐** [§6.2, Fig 5]:
- 32 名专业听众的 pairwise 比较
- ∆MCLP 小时 win rate ≈ 0.5 (接近随机)
- ∆MCLP > 0.1 时 win rate > 0.8
- 随 ∆MCLP 增大,win rate 单调上升,95% CI 不跨越 0.5

**消融** [Table 3]:
| 配置 | CER (w/) | MCLP (w/) | MOS |
|------|----------|-----------|-----|
| Step-Audio-2-mini (base) | 3.276 | -4.829 | 1.856 |
| + SFT | 3.334 | -4.725 | 3.178 |
| + SFT + GRPO (full) | **1.130** | **-4.636** | **3.576** |
| w/o CER Reward | 61.144 | -4.590 | 1.145 |
| w/o MCLP Reward | 0.783 | -4.752 | 2.331 |

[Table 3]

关键消融发现:
1. **SFT 已显著提升风格遵循** (MOS 1.856→3.178),证明 RP-TTS 数据集有效 [§6.4]
2. **w/o CER**: MCLP 达到最佳 (-4.590) 但 CER 灾难性暴涨至 61%,表现为句末固定重复声学模式 — 这是典型的 reward hacking [§6.4]
3. **w/o MCLP**: CER 最低 (0.783%) 但 MOS 大幅下降 (2.331),语音变得平坦无表现力 [§6.4]
4. **Hybrid reward 取得平衡**: CER 1.130% + MCLP -4.636 + MOS 3.576

**Audio History 的作用** [§6.3]:
- 本文模型 MCLP 从 -4.687 (w/o hist) 提升至 -4.636 (w/ hist)
- Baseline 模型未能从 audio history 中获益
- 表明 SFT + MCLP-based RL 增强了模型利用历史声学线索的能力

**跨语言泛化** [Appendix B, Table 4]:
- 在 Genshin-Voice 英文数据上,MCLP 同样能区分 self-prompted (更高 MCLP) 和 prev-prompted (更低 MCLP) 合成,初步验证跨语言泛化性

## 局限性

1. **语言覆盖有限**: 主实验仅覆盖中文 (普通话),英文仅有初步跨语言实验 [§8]
2. **领域覆盖有限**: 数据集仅来源于中国电视剧,有声书、游戏、虚拟助手等场景未验证 [§8]
3. **标注噪声**: 场景/角色描述由 LLM (DeepSeek-R1, Qwen-VL) 自动生成,可能引入偏差 [§8]
4. **Tokenizer 依赖**: MCLP 的效果依赖于 Step-Audio-2 的 semantic tokenizer,不同 tokenizer 的 MCLP 行为可能不同 [agent 解读]
5. **MCLP 计算成本**: 需要运行完整的 LALM 推理来计算每个样本的 MCLP,作为 RL reward 时计算开销较大 [agent 解读]
6. **未开源模型权重**: 代码开源但 continuation model 和 RP-TTS model 的权重未提供,仅数据集开源 [agent 解读]

## 点评

**优点**:
1. MCLP 的设计巧妙 — 通过固定 transcript 和重复结构,将 LALM 的 continuation capability 转化为 style metric,概念简洁且可解释
2. Gated hybrid reward 的消融实验 (Table 3) 非常有说服力地展示了 reward hacking 的严重性和 gating 机制的必要性
3. 评估全面: 50 名专业标注员的 MOS + 32 名专家的 MCLP-MOS 相关性分析 + 多维消融
4. WenetSpeech-RP-TTS 数据集构建流程完整,已开源

**不足**:
1. 仅中文实验限制了结论的普适性,英文实验 (Appendix B) 过于简略 (仅 self vs prev 对比,无人类评估)
2. MCLP 对非 Step-Audio-2 tokenizer 的通用性未验证 — 如果 MCLP 高度依赖特定 tokenizer,其作为通用 metric 的价值会受限
3. 与 GSRM / SpeechJudge 等最新评估方法的对比缺失
4. RP-TTS 场景下的 content-specified setting (response text 已给定) 简化了问题 — 真实 role-play agent 需同时生成内容和风格

**与已有工作的对比**:
- 与 RLAIF-SPA 的区别: RLAIF-SPA 用属性级 AI 反馈做情感对齐,MCLP 用 continuation likelihood 做风格对齐; MCLP 是更 dense 且 continuous 的 reward
- 与 DiffRO 的区别: DiffRO 在 token 空间通过 Gumbel-Softmax 可微优化,MCLP+GRPO 在 audio 空间做 group-relative 优化; MCLP 解决的是 "用什么 reward" 而非 "怎么优化"
- 与 SpeechJudge 的区别: SpeechJudge 评估 naturalness,MCLP 评估 stylistic consistency; 两者覆盖不同维度

## 可复用的 idea

1. **"固定变量 + 重复" continuation 设计**: `[w, z_eval, w]` 结构通过固定和重复 transcript 消除 content 变量,使 LALM likelihood 仅反映 style — 这一思路可推广到任何需要从 continuation model 提取特定属性的场景 (如 prosody consistency, accent consistency)

2. **Gated reward 防止 hacking**: `CER > τ → R=0` 的 gating 机制创建了一个隐式课程 — 模型必须先"说对"才能优化"怎么说"。这比简单加权 (R = α*MCLP + β*CER) 更有效,因为 gating 设置了硬约束而非软 trade-off

3. **Continuation pretraining for style-aware ICL**: 在大规模转写语音上训练 continuation 能力 (300M hours),增强 LALM 的 style-conditioned generation,可作为任何 style-aware 任务的通用预训练策略

4. **RL 仅作用于最后一轮**: 在多轮对话中只对最后一轮做 GRPO rollout,大幅降低计算成本,同时最后一轮已包含完整上下文

## 审阅

(待独立审阅 agent 填写)
