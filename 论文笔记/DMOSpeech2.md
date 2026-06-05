---
type: paper
tier: deep
title: "DMOSpeech 2: Reinforcement Learning for Duration Prediction in Metric-Optimized Speech Synthesis"
arxiv_id: "2507.14988"
source: "Sources/DMOSpeech2.pdf"
authors: [Yinghao Aaron Li, Xilin Jiang, Fei Tao, Cheng Niu, Kaifeng Xu, Juntong Song, Nima Mesgarani]
year: 2025
venue: "AAAI 2026"
tags: [TTS, diffusion, reinforcement-learning, duration-prediction, flow-matching, distillation, zero-shot]
concepts: ["[[ConditionalFlowMatching]]", "[[DurationPredictor]]", "[[Diffusion-basedTTS]]", "[[DifferentiableRewardOptimization]]", "[[SpeakerEmbedding]]", "[[DiffusionModel]]"]
models: ["[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[DurationPredictor]], [[ConditionalFlowMatching]], [[DifferentiableRewardOptimization]], [[Diffusion-basedTTS]], [[SpeakerEmbedding]], [[SEED-TTS-Eval]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: DMOSpeech 2 处于 diffusion-based TTS 演进线的前沿,具体位于 "flow matching distillation + RL post-training" 这一交叉点。在 [[Diffusion-basedTTS]] 的演进中,flow matching (F5-TTS, Voicebox) 已取代 DDPM 成为主流范式,而 DMOSpeech 系列在此基础上进一步引入 distribution matching distillation (DMD) 将推理步数从 32 压缩到 4。在 RL-for-TTS 维度,[[DifferentiableRewardOptimization]] 页面记录了两条路线: 一条是 CosyVoice 3 的 token-level DiffRO (在离散 token 空间优化),另一条是 audio-level GRPO (Multi-Reward GRPO, Tencent 2025)。DMOSpeech 2 开辟了第三条路线: 不对整个 TTS pipeline 做 RL,而是将 RL 范围缩小到 [[DurationPredictor]] 这一单一组件,用 GRPO 优化 duration policy。
>
> **已有认知**: [[DurationPredictor]] 页面追踪了从 FastSpeech length regulator 到 MaskGCT T2D model 的演进,涵盖了 MFA-based、MAS-based、E2E differentiable 等多种 duration 标签获取方式,但尚无 RL-based duration optimization 的记录。VITS 的 stochastic duration predictor 是概率 duration 建模的先驱,DMOSpeech 2 延续了这一思路但引入 GRPO reward 信号替代 variational lower bound。[[SEED-TTS-Eval]] 是当前零样本 TTS 的标准 benchmark,DMOSpeech 2 在此评估。
>
> **创新判断**: DMOSpeech 2 的核心新颖点在于: (1) 将 RL 靶向 duration predictor 而非整个 TTS pipeline,大幅降低计算成本 (4-step student 生成样本,非数百步); (2) teacher-guided sampling 解决 distillation 导致的 mode shrinkage; (3) RL-optimized duration 甚至超越 ground truth duration 的 WER。这些在现有 KB 中无先例。
>
> 检索命中: [[DurationPredictor]], [[ConditionalFlowMatching]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[DifferentiableRewardOptimization]][待确认], [[Diffusion-basedTTS]][待确认], [[DurationPredictor]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 RL (GRPO) 精确靶向 duration predictor 组件,以 SIM+WER 为 reward 优化 diffusion TTS 的时长预测,同时引入 teacher-guided sampling 恢复蒸馏模型的韵律多样性
> - **路线**: Text + Prompt → Duration Predictor (GRPO-optimized) → 预测总时长 L → Student Generator (4-step DMD-distilled flow matching) → Mel → Vocos Vocoder → Waveform
> - **指标**: WER 1.752 / SIM 0.698 (Seed-TTS-en), CER 1.527 / SIM 0.760 (Seed-TTS-zh), RTF 0.032 (5.2x faster than F5-TTS), 0.3B params [Table 1, Table 2]
> - **可借鉴**: (1) RL 不必作用于整个 pipeline,靶向瓶颈组件 (duration predictor) 成本低效果好; (2) teacher-guided sampling: 早期步骤用 teacher 建立韵律,后期步骤用 student 细化声学,两全其美; (3) reward diversity filter: 跳过 reward 区分度不足的 batch 避免无效训练
> - **局限**: (1) 仅优化 duration predictor,未将 RL 扩展到 teacher model 或其他组件; (2) GRPO 训练窗口极窄 (1.5K steps 后退化),超参数敏感; (3) teacher-guided sampling 需同时加载 teacher+student 参数 (0.6B); (4) 未开源训练 reward 模型 (ASR/SV),复现需自建

## 核心问题

Diffusion-based zero-shot TTS (如 F5-TTS) 将语音合成建模为 inpainting 任务,需要预先知道目标语音的总时长。这引入了一个结构性瓶颈: duration predictor 和 speech generator 之间缺乏可微连接,无法端到端优化。具体表现为 [论文原文]:

1. **Duration predictor 是优化盲区**: DMOSpeech 通过 DMD 蒸馏 + 直接 metric 优化解决了 speech generator 的优化问题,但 duration predictor 仍然独立训练 (self-supervised cross-entropy loss),其预测的时长对 SIM 和 WER 的影响无法反馈到优化过程 [§1]
2. **RL 直接应用于整个 TTS pipeline 代价过高**: 每个训练步需生成完整语音样本 (数百步采样),且 RL 的改进上界本质上是 best-of-N sampling,对输出多样性有限的小模型效果差 [§1]
3. **蒸馏导致 mode shrinkage**: DMD 蒸馏后 student model 的韵律多样性显著下降 (CVf0 从 0.666 降到 0.464),主要损失在韵律/节奏维度而非声学质量维度 [§3.3.1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DMOSpeech 2 分为两个训练阶段 [Fig 1]:

**阶段 1 (DMOSpeech, 继承)**: Distribution Matching Distillation
- Teacher: F5-TTS (300M, 32 steps flow matching) 在 Emilia 数据集 (95k hrs) 上训练
- Student: 通过 DMD 2 蒸馏为 4-step student generator,加上 multi-modal adversarial training + direct metric optimization (CTC loss for WER + SV loss for SIM)
- Duration Predictor: encoder-decoder transformer (类 DiTTo-TTS),预测总剩余时长 (非 phoneme-level duration),85K steps supervised training (cross-entropy loss on 300 duration classes, 100ms per bin) [§3.2.1]

**阶段 2 (新增)**: GRPO-based Duration Optimization
- 将 duration predictor 建模为 stochastic policy $\pi_\phi(L|x,p)$ [§3.2.2]
- 用 GRPO 算法优化,reward = CTC log-likelihood + $\lambda_{SIM}$ * cosine similarity [§3.2.2, Eq. 4]
- 仅需 1.5K 额外训练步 [§A.2.1]

### 关键设计选择

#### 1. 为什么 RL 靶向 duration predictor 而非整个 pipeline?

[论文原文] 传统 RL for TTS 需要在每个训练步生成完整语音样本 (数百步采样),计算开销巨大。DMOSpeech 2 利用已有的 4-step student model 生成样本计算 reward,将 RL 的计算成本压缩到传统方案的一小部分 [§1]。此外,duration prediction 是比整体语音生成更受约束的问题 (问题空间小),RL 更容易有效优化 [§1]。

[agent 解读] 这本质上是一种"分治策略": speech generator 已通过 DMD + direct metric optimization 解决了其优化问题,剩下的瓶颈是 duration predictor。RL 的上界是 best-of-N sampling,而 Table 3 显示 best-of-8 duration sampling 的 WER=1.723 远优于固定 duration (WER=3.750),说明 duration 空间确实有很大优化余地。

#### 2. Duration Predictor 的概率建模

Duration predictor 使用 softmax 输出 300 个 duration class 的概率分布 (每个 class 代表 100ms,最大 30 秒),而非回归预测单一值 [§3.2.1]。这使得:
- **采样多样性**: 可通过 Gumbel-softmax 温度参数 $\tau=0.7$ 控制探索程度 [§3.2.2, Eq. 9]
- **GRPO 可行**: 每个 training instance 采样 K=16 个 duration,生成 K 个语音样本,计算 reward 后做 group-level advantage normalization [论文原文, §3.2.2]

#### 3. GRPO 训练的稳定性措施

- **三个 policy**: current policy $\pi_\phi$, old policy $\pi_{old}$ (采样时的 policy), reference policy $\pi_{ref}$ (frozen supervised model); KL 散度 $\beta=0.04$ 防止偏离参考模型 [§3.2.2, Eq. 7]
- **Reward diversity filter**: 跳过 $\max(r) - \min(r) < 0.01$ 的 batch,避免所有 duration 产生类似质量的语音时做无效更新 [论文原文, §3.2.2]
- **训练步数严格控制**: 1.5K steps 后性能显著退化 (特别是 group=8 时 WER 从 ~2 飙升到 >30),表现为 reward hacking [§A.2.1, Fig 4]

#### 4. Teacher-Guided Sampling 解决 Mode Shrinkage

[论文原文] Diffusion 过程中不同噪声水平对应不同生成维度: 高噪声 (早期去噪步) 建立韵律结构 (音素时长、停顿、pitch 轮廓), 低噪声 (后期步骤) 细化声学细节 (音色、音质) [§3.3.1]。Student model 将这一层次过程压缩到 4 步,导致韵律多样性损失严重。

**解决方案** [Algorithm 2, §3.3.2]:
1. Teacher model (F5-TTS) 执行前 K 步 (约 6-14 步) 到切换点 $t_{switch}$,建立多样化的韵律结构
2. Student model 接管剩余 M 步 (2-3 步),从 $t_{switch}$ 到完成
3. Student 继承了 direct metric optimization 的优势 (更好的 SIM 和 WER),同时恢复了 teacher 的韵律多样性

**trade-off 控制**: $t_{switch}$ 可调 — 0.4-0.5 偏重多样性 (创意内容), 0.1-0.2 偏重效率 (实时系统) [§3.3.2]

### 训练策略

| 阶段 | 组件 | 数据 | 步数 | 备注 |
|------|------|------|------|------|
| 1 | F5-TTS Teacher | Emilia 95k hrs | 2M steps | batch 307,200 frames |
| 2 | Student (DMD 2) | Emilia | 200K steps | 半 batch size, LR 从 teacher 终点恢复 |
| 3 | Duration Predictor (supervised) | Emilia | 85K steps | cross-entropy, 300 classes |
| 4 | Duration Predictor (GRPO) | Emilia | 1.5K steps | group=16, $\lambda_{SIM}=3$, 8xH100 |

[§4.1]

## 实验

### 主实验 (Table 1)

| 指标 | DMOSpeech 2 (4步) | F5-TTS Teacher (32步) | w/o RL (4步) | Teacher-Guided (16步) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER↓ | **1.752** | 1.947 | 3.750 | 1.738 | Seed-TTS-en | [Table 1] |
| SIM↑ | **0.698** | 0.662 | 0.672 | 0.699 | Seed-TTS-en | [Table 1] |
| CER↓ | **1.527** | 1.695 | 2.000 | 1.468 | Seed-TTS-zh | [Table 1] |
| SIM↑ | **0.760** | 0.750 | 0.750 | 0.760 | Seed-TTS-zh | [Table 1] |
| RTF↓ | **0.032** | 0.167 | S/A | 0.094 | - | [Table 1] |
| CVf0↑ | 0.464 (student/RL 共用, RL 仅优化 duration 不影响 CVf0) | 0.666 | S/A | 0.593 | - | [Table 1] |

关键发现:
- RL-optimized duration predictor 将 WER 从 3.750 降到 1.752 (53.3% 相对改进),SIM 从 0.672 升到 0.698 [Table 1]
- DMOSpeech 2 (4步) 比 F5-TTS Teacher (32步) 在所有指标上更好,且 5.2x 更快 [Table 1]
- Teacher-guided sampling 恢复 89.1% 的 teacher 韵律多样性 (CVf0 0.593 vs teacher 0.666) [Table 1]
- CMOS 评估: DMOSpeech 2 vs w/o RL 在英文 CMOS-N -0.43** / CMOS-S -0.48** (p<0.01) [Table 1]
- DMOSpeech 2 vs Ground Truth: 英文自然度无统计显著差异,相似度 CMOS-S -0.13* (p<0.05) [Table 1]

### SOTA 对比 (Table 2)

| 指标 | DMOSpeech 2 (4步) | F5-TTS | CosyVoice 2 | Spark-TTS | MaskGCT | LLaSA-8B | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WER↓ | **1.752** | 1.947 | 3.358 | 2.308 | 2.622 | 3.994 | Seed-TTS-en | [Table 2] |
| SIM↑ | 0.698 | 0.662 | 0.641 | 0.572 | **0.713** | 0.594 | Seed-TTS-en | [Table 2] |
| CER↓ | **1.527** | 1.695 | 1.582 | 1.717 | 2.395 | 4.214 | Seed-TTS-zh | [Table 2] |
| RTF↓ | **0.032** | 0.167 | 0.527 | 1.784 | 2.397 | 1.374 | - | [Table 2] |
| #Params | 0.3B | 0.3B | 0.5B | 0.5B | 0.7B | 8B | - | [Table 2] |

关键发现:
- 0.3B 参数全面超越 8B LLaSA 在所有指标上 [Table 2]
- MaskGCT 的 SIM 最高 (0.713) 但 RTF 2.397, 比 DMOSpeech 2 慢 75x [Table 2]
- DMOSpeech 2 比 CosyVoice 2 快 16.5x, 比 Spark-TTS 快 55.8x [§4.3.2]

### Duration 消融 (Table 3)

| Duration 来源 | SIM↑ | WER↓ | 出处 |
| --- | --- | --- | --- |
| Best-of-8 Sampling | 0.724 | **1.723** | [Table 3] |
| DMOSpeech 2 (RL) | 0.698 | 1.752 | [Table 3] |
| Ground Truth Duration | 0.697 | 1.821 | [Table 3] |
| Speaking Rate Based | 0.682 | 2.028 | [Table 3] |
| Duration Predictor (no RL) | 0.672 | 3.750 | [Table 3] |

关键发现:
- RL-optimized duration 的 WER (1.752) 甚至优于 ground truth duration (1.821) [论文原文, §A.1]
- 这说明"最优 duration" 并非简单等于"真实 duration",RL 学到了对模型更友好的 duration 分布 [agent 解读]
- Best-of-8 是 RL 的理论上界,RL 已非常接近 (WER 1.752 vs 1.723) 但只需一次前向传递 [Table 3]

## 局限性

1. **RL 训练窗口极窄**: 1.5K steps 之后性能急剧退化 (group=8 时 WER 从 ~2 飙升到 >30),说明 GRPO 对 duration predictor 的优化极易 reward hacking [Fig 4, §A.2.1]

2. **未优化 teacher model**: teacher-guided sampling 虽恢复多样性,但 teacher model 本身未做 RL 优化,存在进一步提升空间。论文将此列为 future work [§5]

3. **Reward 模型未开源**: CTC-based ASR 和 SV model 均为自训练模型 (6-layer transformer on Emilia),未发布权重 [§B.9, B.10],复现需自建这些 reward 模型

4. **仅优化 SIM + WER 两个维度**: 未引入 MOS、韵律对齐、情感一致性等更多 reward signal,这限制了韵律自然度的优化空间 [agent 解读,对比 Multi-Reward GRPO 使用 5 个 reward]

5. **Teacher-Guided 需双模型参数**: 推理时需同时加载 teacher (0.3B) + student (0.3B) = 0.6B 参数,内存占用翻倍 [Table 2]

## 点评

**优势**:
- **RL 靶向思维精巧**: 不对整个 pipeline 做 RL (成本高、效果受限于 model diversity),而是识别出 duration predictor 这一瓶颈组件,用 4-step student 生成样本计算 reward,将 RL 的计算成本压缩到极致。这一"分而治之"的策略值得在其他多组件系统中借鉴
- **实验设计说服力强**: Table 3 的 duration 来源消融尤其关键 --- RL-optimized duration 超越 ground truth duration 的发现,有力证明了"最优 duration 不等于真实 duration"这一非直觉结论
- **teacher-guided sampling 理论洞察**: 清晰解释了 diffusion 过程中"早期步=韵律结构, 后期步=声学细化"的层次分工,并据此设计了分工策略,这一洞察可迁移到其他 diffusion 蒸馏场景

**不足**:
- GRPO 训练极度脆弱 (1.5K steps 窗口),缺乏对 reward hacking 的根本解决方案,仅靠 early stopping + KL 约束
- 论文未与 DiffRO (CosyVoice 3) 和 Multi-Reward GRPO 做直接对比,虽然这些方法针对不同组件,但 RL-for-TTS 的整体效率比较会有价值
- 评估仅用 Seed-TTS 测试集,未在 LibriSpeech、VCTK 等其他标准集上验证

## 可复用的 idea

1. **Component-level RL**: 识别 pipeline 中的优化盲区,将 RL 靶向单一瓶颈组件而非端到端,大幅降低 sample 生成成本。适用于任何多阶段生成系统 (如 codec LM + vocoder, 或 LLM + flow matching)

2. **Reward diversity filter**: 跳过 $\max(r) - \min(r) < 0.01$ 的 batch,只在能区分好坏的样本上训练。这是 GRPO 训练中的低成本高回报技巧,可直接应用于任何 RL post-training

3. **Teacher-guided sampling**: 在 diffusion 蒸馏场景中,用 teacher 处理建立结构的早期步骤,student 处理细化的后期步骤。$t_{switch}$ 作为效率-多样性的旋钮。可直接迁移到图像/音频/视频的 diffusion distillation

4. **Duration 作为概率分布而非标量**: 将 duration 建模为 softmax 分类 (300 bins) 而非回归,自然支持采样多样性和 GRPO 训练。类比于 LLM 中 token logits 的角色

> [!review] 审阅: pass-with-fixes (2026-06-03)
> - **结论**: pass-with-fixes (0 high, 1 medium, 2 low)
> - [medium] frontmatter models 字段: F5-TTS 错误链接到 NaturalSpeech 2 → 已修正为 MaskGCT + CosyVoice 2
> - [low] CVf0 标注不够清晰 → 已补充备注
> - [low] frontmatter models 自引 → 已移除
> - 详见 `_review/DMOSpeech 2-review.yml`
