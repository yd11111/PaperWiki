---
title: "SASLM"
aliases: [Self-Aware Speech Language Model]
tags: [conversational-tts, speech-llm, context-aware, self-supervised, reinforcement-learning, expressive-speech, VIB]
category: speech-synthesis
year: 2026
venue: "arXiv (EMNLP 2026 sub)"
authors: "Kuang Wang, Lai Wei, Ping Lin, Qibing Bai, Wenkai Fang, Li Zhou, Feng Jiang, Zhongjie Jiang, Jun Huang, Yannan Wang, Haizhou Li"
status: draft
tier: deep
concepts: ["[[ProsodyModeling]]", "[[EmotionControlinTTS]]", "[[StyleTransferinTTS]]", "[[LLM-basedTTS]]", "[[Full-duplexSpokenDialogue]]"]
models: ["SASLM", "Qwen2.5-Omni-3B", "Qwen2.5-Omni-7B", "Qwen3-Omni-30B", "CosyVoice2", "GPT-4o-mini-TTS", "EMOVA-7B", "GLM-4-Voice", "UltraVoice-7B", "Step-Audio2-Mini", "KimiAudio", "Doubao"]
datasets: ["EchoMind", "InstructS2S", "Genshin Dataset", "EmoVoice-DB", "EmoNet", "MMAU"]
---

## KB 背景

**韵律建模演进定位** ([[ProsodyModeling]]): 韵律建模从规则 (SPSS) 经 reference encoder (GST, 2018) → VAE 隐式建模 → in-context learning (VALL-E, 2023) 一路演进。SASLM 代表了一个新节点: **语义状态自蒸馏作为韵律驱动**。不同于 reference encoder 需要参考音频、VAE 需要正则化隐空间、in-context learning 依赖 prompt,SASLM 从 LLM 自身的生成隐藏状态 (h_i) 中通过 VIB 蒸馏出 expressive intent (z_i),直接驱动韵律生成。这消除了对外部韵律信号的依赖。

**情感控制对比** ([[EmotionControlinTTS]]): 情感控制领域有两大路线: (1) proxy-based (离散标签/嵌入): EMOVA 用 emotion tokens, EmoSphere-TTS 用球面向量, RLAIF-SPA 用结构化标签; (2) training-free steering: EmoSteer-TTS/CoCoEmo/DUET 在激活空间操作。SASLM 开辟第三条路线: **proxy-free 自蒸馏** -- 不用任何外部情感标注或 steering vector,从 LLM 语义理解中内在推导表达意图,且通过 self-reward 闭环验证意图-实现一致性。

**与对话 TTS 的关系** ([[Full-duplexSpokenDialogue]]): 当前 SLM (Qwen2.5-Omni, Moshi 等) 具备强语义能力但表达力弱,直接复用 text-prediction-oriented hidden states 做语音合成导致 prosody flattening。SASLM 识别并命名了这一问题为 "semantic understanding-acoustic realization gap",提出不需要改变对话架构本身,而是在语义到声学的桥接层做意图蒸馏。

**在 LLM-TTS 范式中的位置** ([[LLM-basedTTS]]): SASLM 采用 modular multi-head 架构 (与 Qwen2.5-Omni 同族),但在 LLM backbone 与 speech head 之间插入 VIB modulation layer,是 "LLM hidden state → speech token" 路径上的首次显式意图建模。

> [!summary] 速查
> **一句话**: 3B SLM 通过 VIB 从自身语义状态自蒸馏 token-level 表达意图,再用 self-reward 闭环对齐意图与声学实现,无需任何外部情感/风格标注即达 open-source SOTA。
>
> **路线**: Modular multi-head SLM + VIB intent distillation + OU temporal prior + UAPO self-reward alignment
>
> **指标**: EchoMind 上 F0-Var 63.44 (vs Qwen3-Omni-30B 49.76), EmoAlign 35.61% (vs 25.03%), Overall 4.33 (vs 4.25); WER 4.56%
>
> **可借鉴**: (1) VIB+AdaLN 注入模式: 用信息瓶颈从 hidden state 蒸馏 intent 再通过 AdaLN 注入 token embedding, 比直接加/拼接稳定得多 (WER 4.74% vs 8.29%); (2) OU prior 建模时序平滑: 比 i.i.d. Gaussian 更符合韵律的时间连续性; (3) Self-reward 闭环: 用自身 3B 模型做 rubric-based 评分,不需要外部大模型也能有效对齐; (4) 双编码器分离: perception encoder (语义) + generation encoder (声学) 提供互补信息
>
> **局限**: 仅英文 EchoMind 评估; 仅 800h 训练数据导致偶发声学瑕疵 (WER 4.56% vs Qwen3-Omni 2.76%); 未验证多语言/跨文化场景

## 核心问题

当前 Speech Language Models (如 Qwen2.5-Omni, GLM-4-Voice) 具有强语义理解能力,但无法将这种理解转化为表达性声学实现 -- 产出的语音语义正确但韵律扁平、情感错位。论文将此命名为 **semantic understanding-acoustic realization gap** [§1]:

- **原因分析** [§1, agent 解读]: vanilla SLMs 直接复用为 text prediction 优化的 hidden states 做语音调制。这些 hidden states 是 "semantics-oriented" 而非 "acoustics-oriented" -- 它们编码了 "说什么" 但未编码 "怎么说"。
- **现有方案的局限** [§1]: (1) Cascaded 系统依赖外部 style prompt (需标注); (2) Emotional SLMs (EMOVA, Lucy) 用离散 emotion tokens 作代理,但 emotion label 是粗粒度的,无法捕捉对话中动态演进的表达意图; (3) 所有现有方法都是 open-loop 训练 (token-level imitation),缺乏跨模态一致性验证。

## 方法: 它怎么 work

### 整体框架 (Self-Aware Intent-Realization Alignment)

SASLM 将 context-aware expressive speech generation 重构为一个 **闭环自感知过程**: 模型首先内在推导 "我想如何表达" (intent),然后生成语音,最后自我评估 "我的表达是否符合意图" (realization alignment)。具体通过两个耦合机制实现 [§3]:

1. **Intent-Aware Bridging** (意图感知桥接): 从 LLM 自身语义生成动态中自蒸馏表达意图
2. **Realization-Aware Alignment** (实现感知对齐): 通过 self-reward 闭环强制意图-实现一致性

### 3.1 Intent-Aware Expressive Modeling Architecture

#### 3.1.1 Semantic Understanding Backbone [§3.1.1]

基于 Qwen2.5-Omni-3B Thinker 模块:
- Perception encoder: Whisper (提取语义表征 R_P)
- Shared LLM backbone: Qwen2.5-3B (自回归建模, 输出 hidden states h_i)
- Text head: 从 h_i 解码文本 token

#### 3.1.2 VIB-based Intent-Aware Expressive Modulation [§3.1.2]

**WHY 选 VIB 而非直接用 hidden state?** [§5.1, Table 3] 直接将 h_i 加到 token embedding (naive addition) 导致 WER 暴涨到 8.29% (lexical-acoustic 纠缠)。AdaLN 降到 6.50% 但仍有信息泄漏。VIB 的压缩约束迫使 z_i 丢弃已被 token embedding 编码的词汇信息,仅保留韵律/情感等表达性线索 [论文原文]。消融证据: linear probe 准确率 e:18.1% → h:41.4% → z:64.5% [§5.3, Fig. 5],说明 VIB 确实蒸馏出了比 h 更纯净的情感表征。

**VIB 编码器**: 轻量 MLP f_φ 预测 μ_i 和 σ_i, reparameterization trick 采样 z_i [Eq. 6]

**WHY 选 OU prior 而非标准 Gaussian?** [§3.1.2] 韵律是时间连续的 (prosody unfolds smoothly, Nooteboom 2006), i.i.d. Gaussian 无法建模相邻 token 间的时序相关性。OU prior p(z_i | z_{i-1}) = N(α·z_{i-1}, σ_p^2·I) 引入了 mean-reverting temporal inertia (α=0.95), 使 z_i 序列保持时间平滑 [Appendix C.1]。

**注入方式**: AdaLN (Adaptive Layer Normalization) [Eq. 8]:
```
f_i = (1 + γ(z_i)) ⊙ (e_i - μ)/σ + δ(z_i)
```
γ 和 δ 是 zero-initialized linear projections, 确保训练初期不扰动预训练表示 [论文原文]。

#### 3.1.3 Intent-Modulated Speech Generation [§3.1.3]

Speech head (初始化自 CosyVoice2 S3 tokenizer) 条件于:
- **F_Y** (intent-modulated fused embeddings): 编码 "说什么" + "怎么说"
- **R_G = φ_G(X^S)** (generation encoder 输出): 保留用户语音的连续副语言线索 (paralinguistic cues)

**WHY 需要双编码器?** [§3.1.3, §5.1, Table 3] Perception encoder (Whisper) 提取语义表征供 LLM 理解,但丢弃了声学细节。Generation encoder 直接编码原始输入语音,保留 paralinguistic 信息 (如用户情感状态),为 speech head 提供声学 grounding。消融: 无 context (WER 4.19%, Ovr 4.21) vs acoustic context (WER 4.74%, Ovr 4.31) -- 声学 context 以轻微 WER 代价显著提升表达力 [Table 3B]。而 semantic context 直接注入反而有害 (WER 11.97%) [Table 3B]。

### 3.2 Realization-Aware Alignment Training (3-Stage)

**WHY 需要 3 阶段渐进训练?** [agent 解读] LLM backbone 预训练为文本语义,如果一开始就端到端训练会破坏语义能力。渐进方式允许: Stage 1 建立 speech head 基础能力 → Stage 2 让 LLM 最后一层向 intent-aware 偏移 → Stage 3 通过 RL 闭环优化。

#### Stage 1: Acoustic Bootstrapping [§3.2.1]
- 仅训练 modulation layer + speech head
- LLM backbone 冻结
- 目标: L_speech = L_Recon + β·L_VIB [Eq. 10]
- 数据: InstructS2S ~2000h [Table 1]

#### Stage 2: Expressive Intent Grounding [§3.2.2]
- 解冻 LLM **最后一层** (仅最后一层!)
- 联合优化: L_stage2 = L_text + L_speech [Eq. 12]
- 数据: Genshin Dataset ~603h + EmoVoice-DB ~192h [Table 1]
- 目的: 将 hidden states 从 "semantic-dominant" 渐进转向 "intent-aware acoustically usable" [论文原文]

#### Stage 3: Closed-Loop Self-Reward Alignment [§3.2.3]

**WHY self-reward 而非 human preference / external reward model?** [§5.2, Table 4, Fig. 4] (1) 不需要昂贵的人类标注; (2) 3B self-reward 虽收敛慢于 30B oracle-reward (Fig. 4), 但最终也能有效改善 (Ovr 4.31→4.33, WER 4.74%→4.56%) [Table 4]; (3) 闭环设计使表达力和可懂度同时改善 (不像 open-loop 训练中的 expressivity-stability trade-off)。

**具体流程** [Algorithm 1, Appendix A]:

**Stage 3(a): Self-Reward Generation**
1. 给定输入 X^S, 生成文本 Y^T + K=32 个 speech rollouts
2. 用 TTS (CosyVoice2) 生成 context-unaware anchor Y_⊥^S
3. SASLM 为每个 rollout 生成 rubric (emotion/prosody/naturalness 三维)
4. 自评估每个 rollout 得分 {0, 0.5, 1}^3; WER > τ=0.2 直接判 0
5. 按 score 相对于 anchor 分为 preferred (Y_w) 和 rejected (Y_l)

**Stage 3(b): Preference-based Alignment**
- 使用 UAPO (Utility-Anchored Preference Optimization) [Eq. 13]
- Anchor Y_⊥^S 作为 "minimum acceptable prospect" baseline
- 同时 promote preferred (超过 anchor) 和 suppress rejected (低于 anchor)
- 数据: EmoNet English subset ~263h [Table 1]

### 训练配置 [Appendix C.1]
- 8 x H20 GPUs, 每 stage 约 2 天
- Speech head LR: 1e-5; LLM backbone LR: 1e-6
- OU prior: α=0.95, σ_p=0.5
- β (KL coefficient): 前 10% warmup 为 0, 后 cosine anneal 到 0.5

## 实验与证据

### 主实验: EchoMind Benchmark [Table 2]

| 模型 | 类型 | WER%↓ | F0-Var↑ | EmoAlign%↑ | Ovr.↑ |
|------|------|-------|---------|-----------|-------|
| Qwen2.5-Omni-3B | Vanilla SLM | 4.14 | 42.88 | 21.44 | 3.79 |
| Qwen2.5-Omni-7B | Vanilla SLM | 3.75 | 47.77 | 23.10 | 3.86 |
| Qwen3-Omni-30B | Vanilla SLM | 2.76 | 49.76 | 25.03 | 4.25 |
| EMOVA-7B | Emotional SLM | 10.09 | 47.98 | 30.58 | 4.06 |
| CosyVoice2 | Controllable TTS | 3.80 | 51.20 | 39.07 | 4.23 |
| GPT-4o-mini-TTS | Controllable TTS | 7.63 | 70.44 | 45.04 | 4.41 |
| Doubao | Commercial SLM | 1.12 | 68.58 | 42.95 | 4.44 |
| **SASLM-3B** | **Expressive SLM** | **4.56** | **63.44** | **35.61** | **4.33** |

**关键发现** [§4.3]:
1. **vs 10x 更大模型**: SASLM-3B 在 F0-Var (+27.5%), EmoAlign (+42.3%), Overall (+0.08) 全面超越 Qwen3-Omni-30B,证明 explicit intent modeling > implicit scaling [论文原文]
2. **vs 情感 SLM**: 大幅超越 EMOVA-7B (Ovr 4.33 vs 4.06, EmoAlign 35.61% vs 30.58%),latent intent > discrete emotion proxies [§4.3]
3. **vs Oracle TTS**: 主观韵律 (4.36 vs CosyVoice2 4.25) 和自然度 (4.49 vs 4.34) 超越 oracle 控制的 TTS,但 EmoAlign 低于 CosyVoice2 (35.61% vs 39.07%)。论文解释: oracle labels 仅强制粗粒度一致性,SASLM 推断 token-level 细粒度意图 [§4.3]
4. **vs 商业系统**: 接近 Doubao (Ovr 4.33 vs 4.44),但 WER 差距较大 (4.56% vs 1.12%)

### 消融实验 [§5.1, Table 3]

**Modulation strategy** (acoustic grounding 条件下):
| 策略 | WER% | Ovr. |
|------|------|------|
| π1 (无 intent, TTS baseline) | 3.61 | 4.17 |
| ⊕ (naive addition) | 8.29 | 4.20 |
| AdaLN | 6.50 | 4.22 |
| VIB + AdaLN (SASLM) | 4.74 | 4.31 |

**Context grounding** (VIB+AdaLN 条件下):
| R_G | WER% | Ovr. |
|-----|------|------|
| ∅ (无 context) | 4.19 | 4.21 |
| Semantic (R_G = E_X) | 11.97 | 4.20 |
| Acoustic (R_G = φ_G(X^S)) | 4.74 | 4.31 |

### Self-Reward 效果 [§5.2, Table 4]
| 策略 | RM Size | WER% | Ovr. |
|------|---------|------|------|
| SFT (Stage 1+2) | -- | 4.74 | 4.31 |
| + Self-Reward | 3B | 4.56 | 4.33 |
| + Oracle-Reward | 30B | 4.21 | 4.38 |

Self-reward 同时改善 WER 和表达力 -- 闭环优化缓解 expressivity-stability trade-off [§5.2]。

### 语义能力保持 [Table 5, Appendix C.3]
MMAU: SASLM 39.60 vs Qwen2.5-Omni-3B 39.51 (无损)
EchoMind empathetic: CF 4.56 vs 4.54 (微升)

### 表征分析 [§5.3, Fig. 5]
t-SNE + linear probe: token embedding e → 18.1%, hidden state h → 41.4%, VIB latent z → 64.5%
VIB 确实从纠缠的语义状态中蒸馏出了更纯净的情感判别表征。

## 与已有工作的对比

| 维度 | SASLM | EMOVA | UltraVoice | CosyVoice2 (oracle) | EmoSteer-TTS |
|------|-------|-------|------------|---------------------|--------------|
| 控制信号 | 自蒸馏 VIB latent | 离散 emotion token | Style-controlled 训练数据 | 外部 emotion label | 激活空间 steering vector |
| 需要标注? | 否 | 是 (emotion) | 是 (style) | 是 (oracle labels) | 否 (仅需 ~7k 情感对) |
| 粒度 | Token-level 动态 | Utterance-level | Utterance-level | Utterance-level | Global |
| 闭环验证? | 是 (self-reward) | 否 | 否 | 否 | 否 |
| 训练开销 | 3-stage, 800h, 8xH20 | 多阶段大规模 | 大规模风格数据 | -- | Zero training |

**与 ConversationalTTS-RL 路线对比** [agent 解读]: 现有 RL for TTS (如 DiffRO, RLAIF-SPA, Multi-Reward GRPO) 主要关注 TTS 质量/情感准确率,是 "reward model → TTS" 的单向优化。SASLM 的 self-reward 是 "SLM 自评 → 自改进" 的闭环,且 reward 对象是 intent-realization consistency 而非单纯声学质量。

**与 DiffCSS/Chain-Talker 等对话 TTS 对比** [agent 解读]: DiffCSS 等通过对话历史编码器显式建模上下文风格,仍是 proxy-based (需要从历史中提取 style representation)。SASLM 的 context-awareness 内建于 LLM backbone 的自回归生成过程中,表达意图从生成动态中自然涌现,不需要额外的历史编码模块。

## 局限与开放问题

1. **评估覆盖有限** [Limitations]: 仅在英文 EchoMind (empathetic dialogue) 上评估,多语言、跨文化、domain-specific 表达行为未探索
2. **声学质量瓶颈** [Table 2, §D.2.1]: WER 4.56% 高于 Qwen3-Omni (2.76%) 和 Doubao (1.12%),人类评估中 Quality 维度落后于 GPT-4o-mini-TTS 和 Doubao,归因于训练数据规模有限 (800h vs 商业系统的大规模数据)
3. **Self-reward 能力边界** [Fig. 4]: 3B self-reward 收敛慢于 30B oracle-reward,暗示在更复杂的表达判断上可能存在 ceiling effect
4. **VIB β 敏感性** [未确认]: 论文用 cosine annealing β 从 0 到 0.5,但未详细分析 β 对表达力-可懂度 trade-off 的影响曲线
5. **伦理风险** [Limitations]: 更强的表达性合成可能增加 impersonation 和 emotional manipulation 风险

## 反向更新计划

以下仅为 **append** 操作 (不改 status):

1. [[ProsodyModeling]] — key_papers 追加 `"[[论文笔记/SASLM|SASLM]]"` (VIB 自蒸馏韵律意图新范式)
2. [[EmotionControlinTTS]] — key_papers 追加 `"[[论文笔记/SASLM|SASLM]]"` (proxy-free self-reward 情感对齐)
3. [[LLM-basedTTS]] — key_papers 追加 `"[[论文笔记/SASLM|SASLM]]"` (intent-modulated SLM 语音生成)

> [!review] 审阅结论: pass
> - 方法节因果解释充分,每个设计选择有 WHY 回答 (VIB vs naive addition, OU vs Gaussian, acoustic vs semantic grounding, 3-stage 渐进)
> - 数字 claim 均有 Table/Section 标注
> - 事实 vs 推断已用 [论文原文] / [agent 解读] 标注
> - KB 背景有明确谱系定位和对比基准
> - 反向更新仅 append key_papers,合理
> 审阅报告: [[_review/SASLM-review.yml]]
