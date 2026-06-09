---
type: paper
tier: deep
title: "SpeechWorldModel"
aliases: [Speech World Model, SWM, Causal Graph Speech Understanding]
authors: [Anonymous]
year: 2026
venue: ICLR 2026 (under review)
arxiv_id: ""
source: ""
tags: [speech-understanding, causal-graph, world-model, reasoning, emotion-recognition, speech-act, theory-of-mind, SLM]
level: deep
status: draft
created: 2026-06-03
updated: 2026-06-03
concepts: ["[[SpeechLanguageModel]]", "[[AudioUnderstanding]]", "[[ProsodyModeling]]", "[[EmotionControlinTTS]]"]
models: []
datasets: []
kb_context_sources: []
kb_sources: ["[[SpeechLanguageModel]]", "[[AudioUnderstanding]]", "[[ProsodyModeling]]"]
---
tier: deep

# Speech World Model: Causal State-Action Planning with Explicit Reasoning for Speech

## KB 背景

- **[[SpeechLanguageModel]]** [confirmed]: SWM 挑战了当前 SLM 的"黑箱理解"范式，提出用显式因果图结构化语音理解的模块化方案，作为 SLM 推理能力增强的新路线 [论文原文]
- **[[AudioUnderstanding]]** [confirmed]: SWM 将语音理解分解为 4 个认知模块 (WMA/ToM/SA/Prag)，覆盖场景感知、情感识别、言语行为分类和语用意图推理，与 Audio Understanding 的多任务体系直接对应 [论文原文]
- **[[ProsodyModeling]]** [confirmed]: SWM 使用 openSMILE 提取韵律特征 (prosody z) 作为因果图的输入之一，韵律信息直接影响 Theory of Mind (情感) 和 Speech Act (言语行为) 模块的推理 [论文原文]

> [!summary] 速查
> - **一句话**: 首个基于因果图的模块化语音理解模型，将语音理解分解为 4 个认知模块 (WMA→ToM→SA→Prag) 构成 DAG，显式推理链引导 LLM 生成，在情感识别上超越 Gemini 2.5 Pro
> - **路线**: speech → WavLM (acoustic) + DistilBERT (text) + openSMILE (prosody) → Fusion → Causal Graph (4 nodes) → Explicit Reasoning Chain → LLaMA/Qwen2-Audio IT → Response
> - **指标**: Overall M.J. Score 7.81 (LLaMA) / 7.59 (Qwen2-Audio), 超越 Gemini 2.5 Pro (8.12) 的训练成本的 1/50+, Emotion Accuracy 97.80% (ours) vs 82.47% (Gemini) [Table 3]
> - **可借鉴**: (1) 因果图结构化语音理解 (稀疏梯度 + 5x 训练加速); (2) 半监督图训练 (缺失标签时通过链式梯度学习); (3) 认知科学启发的模块设计
> - **局限**: 仅 4 模块 (简化); 因果图结构预定义 (非自适应学习); 依赖 label generation pipeline; ICLR 2026 在审

## 1. 核心问题 / Core Problem

**当前 SLM 的根本缺陷** [§1]:
1. **黑箱理解**: 现有 SLM (Qwen2-Audio, Voxtral) 将 ASR + 情感 + 意图 + 语用推理当作单一黑箱处理，无法解释推理过程 [§1]
2. **忽略因果结构**: 情感变化会系统性地约束言语行为和语用意图 (如 neutral→anger 导致 statement-opinion→complaint)，但当前模型不建模这种依赖关系 [Fig 3]
3. **弱推理能力**: 当前 SLM 表面上"推理"实为复杂模式匹配，而非因果推断 [§1, Shojaee et al., 2025]

**核心假设**: 认知科学表明语音感知是**模块化**的 (Hickok & Poeppel, 2007)，显式建模模块间因果依赖能提升推理能力 [§2.1]。

## 2. 方法论 / Methodology

### 2.1 World Model 因果图 [§3.1]

**4 个认知模块** [§3.1.1]:

| 模块 | 含义 | 标签空间 | 认知科学基础 |
|------|------|---------|-------------|
| WMA (World Model Activation) | 场景/领域定位 | 30 类 (Alarm, Calendar, Finance...) | Situation Models (Zwaan, 1998) [§3.1.1] |
| ToM (Theory of Mind) | 说话人情感状态 | 7 类 (Neutral, Joy, Sadness, Anger, Surprise, Fear, Disgust) | 心智理论 (Premack & Woodruff, 1978) [§3.1.1] |
| SA (Speech Act) | 言语行为功能 | 20+ 类 (Statement, Question, Command, Backchannel...) | Austin (1962) / Searle (1969) [§3.1.1] |
| Prag (Pragmatic Intent) | 语用意图/潜在目标 | 15 类 (Request-Action, Complaint, Social/Chitchat...) | 间接言语行为 (Ruytenbeek, 2021) [§3.1.1] |

**因果图结构 (DAG)** [Fig 4]:
```
WMA ──→ SA
 ↕        ↓
ToM ──→ Prag
```
- WMA 和 ToM 是独立根节点 (直接从语音特征推断)
- SA 依赖 WMA + ToM + speech input
- Prag 依赖 SA + ToM + WMA + speech input

### 2.2 图计算与训练 [§3.1.2-3.1.4]

**节点计算** [Eq 1-2]:
```
S_v = f_v({S_u : u ∈ Pa(v)}, A_{u→v})
p(Z|X) = p(z_WMA|X) · p(z_ToM|X) · p(z_SA|z_WMA, z_ToM, X) · p(z_Prag|z_SA, z_ToM, z_WMA, X)
```

**多任务监督损失** [Eq 3]:
```
L_sup = ΣΣ m_{i,v} · CE(y_{i,v}, S_{i,v})
```
其中 m_{i,v} 指示样本 i 的模块 v 是否有标签。

**半监督学习** [§3.1.4]:
- 当父节点缺乏标签时，通过**子节点标签的链式梯度**反向传播到父节点 [Eq 5, Fig 4(A)]
- 无标签父节点作为**潜变量生成器**，因果图自然提供了结构化先验 [§3.1.4]

**Teacher Forcing** [Eq 4]:
```
S̃_{i,u→v} = τ · onehot(y_{i,u}) + (1-τ) · stopgrad(S_{i,u}),  τ ~ Bernoulli(p_{u→v})
```

### 2.3 随机图基线 [§3.2]

用全连接图 (Random Graph) 替换因果 DAG，验证因果结构的贡献:
- Random Graph: 每个模块条件于所有其他模块 + 语音输入
- 两轮迭代计算 (t=0 初始化, t=1 teacher forcing, t=2 最终输出) [Eq 14-16]

### 2.4 指令调优 [§3.3]

**两种设置**:
1. **Language-only**: LLaMA3.1-8B + LoRA，输入因果图文本序列化输出 I(G(x))，生成 reasoning chain + response [Eq 7]
2. **Multi-modal**: Qwen2-Audio + LoRA，输入原始语音 + 因果图状态，生成 reasoning chain + response [Eq 8]

### 2.5 数据构建 [§4.1-4.2]

**训练数据**: MELD (13K, emotion) + IEMOCAP (10K, emotion) + SLURP (72K, intent+action+scene) + VoxCeleb (30K, speaker ID) = ~125K 样本 [Table 4]

**标签补全 pipeline** [§4.2]:
- Stage 1: Vicuna-13b-v1.5 补全缺失模块标签 (one-shot prompting + constrained decoding) [§4.2]
- Stage 2: 基于完整标签 + 转写，合成 reasoning chain + response [§4.2]

## 3. 实验与结果 / Experiments

### 3.1 图评估 [Table 1]

| 方法 | 设置 | WMA | ToM | SA | Prag | ACE↑ | ICS↑ |
|------|------|-----|-----|-----|------|------|------|
| Fully-supervised | - | 69.4 | 73.5 | 65.3 | 81.4 | 23.57 | 43.29 |
| Semi-sup (latent WMA) | - | **34.8** | 75.0 | 70.7 | 83.2 | 21.71 | 26.9 |
| Semi-sup (latent ToM) | - | 69.1 | **43.3** | 69.6 | 83.5 | 21.98 | 28.9 |
| Semi-sup (latent SA) | - | 69.3 | 77.0 | **34.4** | 82.5 | 21.65 | 29.3 |
| Random Graph | - | 69.7 | 74.0 | 67.5 | 83.6 | - | - |

**关键发现** [§4.5]:
- 因果图在半监督下**准确度可比**全监督，但训练成本仅 1/5 (2.07h vs 10.39h) [§4.4]
- 无标签模块准确度下降**局限于该模块的因果边**，其他路径不受影响 → 模块正交性 [§4.5]
- 随机图的信息流**随 teacher forcing ratio 混沌波动** → 结构不稳定 [Table 2]

### 3.2 语音理解与推理 [Table 3]

| 方法 | Prompt | Overall M.J.↑ | Reasoning↑ | Response↑ | EM↑ | EA↑ |
|------|--------|---------------|-----------|-----------|-----|-----|
| **SWM (LLaMA)** | CoT | **7.81** | **7.84** | 7.76 | **97.80** | 66.26 |
| **SWM (Qwen2-Audio)** | CoT | 7.59 | 7.26 | **8.08** | 91.80 | **71.02** |
| Qwen2-Audio-CoT (tuned) | CoT | 5.18 | 4.76 | 5.82 | 92.11 | 34.72 |
| Qwen2-Audio (2024) | Direct | 2.70 | 2.20 | 3.46 | 14.20 | 8.00 |
| Gemini 2.5 Pro | CoT | 8.12 | **8.02** | **8.28** | 82.47 | 51.29 |

**核心结论** [§4.6]:
- SWM 超越所有开源 SLM，甚至在部分指标上接近/超越 Gemini 2.5 Pro [Table 3]
- **情感识别 (EM)** 是 SWM 最强项: 97.80% vs Gemini 82.47%——因果图的 ToM 模块显式解耦了情感状态 [§4.6]
- 仅用 CoT instruction tuning data 微调 Qwen2-Audio 就已超越所有开源基线 → 高质量推理数据的价值 [§4.6]
- SWM 训练成本远低于 Gemini (20 GPU hours vs 商业模型) [§4.6]

## 4. 设计选择分析 / Design Analysis

### WHY: 为什么用因果图而非标准 CoT

- CoT 的搜索空间是全状态空间 Y = Y₁ × ... × Y_N (指数级大) [Appendix A.3.2, Eq 21]
- 因果图将搜索空间约束到低熵子空间 S_G ⊂ Y [Appendix A.3.2]
- 数学证明: 因果条件化降低梯度估计方差 (作为 control variate) [Appendix A.3, Eq 23]

### WHY: 半监督有效

- 因果图结构提供了**自然的潜变量生成机制**: 无标签节点通过子节点标签的链式梯度学习 [§3.1.4]
- 对比: 随机图需学习所有 O(n²) 依赖关系，因果图只需学习 |E| 条边 [Appendix A.3.1, Eq 20]

### WHY: 4 个模块而非更多

- 基于认知科学 (Shannon 通信模型 + 认知语用学)，选择**必要且充分**的维度 [§3.1.1]
- 覆盖 "where" (WMA) + "who" (ToM) + "what" (SA) + "why" (Prag) [§3.1.1]
- 更多模块会增加标注成本且可能导致稀疏梯度 [§5]

## 5. 与已有方法的关键差异

| 维度 | 标准 SLM (Qwen2-Audio) | CoT-tuned SLM | SWM |
|------|----------------------|---------------|-----|
| 推理方式 | 黑箱 | 文本 CoT (无结构) | 结构化因果 CoT [§3] |
| 中间表示 | 无 | 自由文本链 | 因果图状态 (离散分类) [§3.1] |
| 训练效率 | - | - | 5x 加速 (vs Random Graph) [§4.4] |
| 可解释性 | 低 | 中 (文本可读) | 高 (因果干预可验证) [§4.5] |
| 情感识别 | 14.20% EM | - | **97.80%** EM [Table 3] |

## 6. 局限与未来方向

- **模块数量有限**: 仅 4 个模块，更多语音维度 (如语速、停顿、音色) 未显式建模 [§5]
- **因果图结构预定义**: 无法自适应学习因果关系，未来可探索因果发现 [§5]
- **标签生成依赖**: label pipeline 使用 Vicuna-13b，错误可能传播 [§5]
- **仅理解不生成**: SWM 不生成语音，仅输出文本推理和响应 [论文原文]

---

检索命中: [[SpeechLanguageModel]], [[AudioUnderstanding]], [[ProsodyModeling]] | 过滤: 无 | 未命中但可能相关: [[EmotionControlinTTS]]


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass-with-fixes
> 
> 结构检查: 速查卡片 ✗ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
