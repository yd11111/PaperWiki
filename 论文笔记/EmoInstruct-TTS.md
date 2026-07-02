---
tags: [emotion-TTS, instruction-following, controllable-TTS, prosody, flow-matching, emotion-embedding, dual-path, intensity-control]
tier: deep
title: "EmoInstruct-TTS: Dual-Path Instruction-Guided Emotional Speech Synthesis"
arxiv_id: "2606.20650"
source: "Sources/EmoInstruct-TTS.pdf"
authors: [Minghui Wu, Ganjun Liu, Zikun Fang, Ting Meng, Hongchuan Wu, Bingao Xu, Yonglong Cai, Jiasheng Chen, Jun Du]
year: 2026
venue: "Interspeech 2026 (submitted)"
concepts: ["[[EmotionControlinTTS]]", "[[ProsodyModeling]]", "[[ConditionalFlowMatching]]", "[[NaturalLanguageDescriptionforTTS]]", "[[SpeakerEmbedding]]"]
models: []
tasks: []
datasets: []
status: draft
date_read: 2026-07-02
date_published: 2026-06
confidence: medium
importance: medium
read_time: ""
related_notes: ["[[EmoCtrl-TTS]]", "[[UMETTS]]", "[[InstructTTS]]", "[[EmoVoice]]", "[[WeSCon]]", "[[EmoSteer-TTS]]", "[[DiffRO]]", "[[RLAIF-SPA]]"]
kb_links: ["[[EmotionControlinTTS]]", "[[ProsodyModeling]]"]
---

> [!card] 速查卡片
> - **一句话总结**: 讯飞/USTC 提出双路径指令驱动情感 TTS 框架,用 Emotion2embed (语义+声学联合嵌入,覆盖 48 种情感状态) 和 ICE-Flow (指令→嵌入的条件 flow 模型) 实现细粒度情感类别+强度控制
> - **核心贡献**: (1) Emotion2embed: 语义-声学联合情感嵌入,覆盖 27 类情感 + 7 主类 × 3 强度级 = 48 种状态,带序数强度约束; (2) ICE-Flow: 从自然语言指令生成情感嵌入,配合分布正则化避免 mode collapse; (3) 双路径架构将语义规划与情感声学控制解耦
> - **方法关键词**: Emotion2embed, ICE-Flow, dual-path, Sentence-BERT + ECAPA-TDNN, ordinal intensity ranking, covariance regularization, CFM-TTS, BigVGAN
> - **基于什么**: CosyVoice2/CosyVoice3 pipeline (Qwen2.5-0.5B LLM + CFM decoder + BigVGAN)
> - **对比了谁**: CosyVoice2, CosyVoice3 (zero-shot 设定)
> - **数据集/规模**: ESD + CNCED; Dataset-Base 49,903 条 (Gemini-2.5 Pro 自动标注) + Dataset-Annotation 28,402 条 (人工标注), 7,600 评估集
> - **核心数字**: 21 情感-强度任务 ESMOS 4.25 (Female, vs CosyVoice3 3.98); 27 细粒度情感 ESMOS 3.92 (vs CosyVoice2 3.55); 48 类 ECS 0.870 (vs CosyVoice3 0.865); ICE-Flow 额外延迟 <5ms (<1% overhead)
> - **局限(作者自述)**: 依赖预定义的 48 类情感标签,未来方向是开放式自然语言描述驱动
> - **局限(我的判断)**: (1) 仅对比 CosyVoice 系列,未与 EmoCtrl-TTS/EmoSteer-TTS/WeSCon 等专门方法对比,竞争力存疑; (2) 48 类分类仍是离散的,不如 arousal-valence 连续空间灵活; (3) Emotion2embed 训练需要成对的 (语音, 文本描述) 标注数据,扩展成本高; (4) 5 页 Interspeech 短文,实验深度有限
> - **借鉴意义**: 语义-声学联合嵌入 + 序数强度约束的思路可推广; ICE-Flow 的分布正则化是抗 mode collapse 的好技巧; 双路径设计 (语义规划 + 情感控制分离) 是一种清晰的工程范式

## 📌 KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 [[ProsodyModeling]] + 1 个待确认实体页 [[EmotionControlinTTS]])

**谱系定位**: EmoInstruct-TTS 处于 instruction-driven emotional TTS 演进线上,是 InstructTTS (2023, NL description → style) → PromptTTS/Parler-TTS (2023-24) → EmoVoice (2025, LLM 自由文本情感 prompt) → EmoInstruct-TTS (2026, dual-path 结构化嵌入 + 指令) 的最新节点。

**与知识库已有方法的关系**:

- **vs EmoCtrl-TTS** [知识库已收录]: EmoCtrl-TTS 用帧级 arousal-valence + laughter embedding 做时变情感控制,依赖 27k 小时伪标签数据全参微调。EmoInstruct-TTS 用句级结构化嵌入 (Emotion2embed) + 自然语言指令双路径,数据规模小得多 (~50k 条)。EmoCtrl-TTS 是**连续帧级控制**,EmoInstruct-TTS 是**离散类别+强度级控制**。
- **vs EmoSteer-TTS** [知识库已收录]: EmoSteer-TTS 是 training-free 的激活 steering,不需要任何情感嵌入训练。EmoInstruct-TTS 需要训练 Emotion2embed + ICE-Flow,但提供了更显式的强度控制。
- **vs WeSCon** [知识库已收录]: WeSCon 实现词级情感控制 (通过 self-training + DEAB attention bias),EmoInstruct-TTS 是句级控制,粒度更粗。
- **vs DiffRO/RLAIF-SPA** [知识库已收录]: 这两者走 reward-guided 路线 (SER reward → RL 优化),不需要情感嵌入。EmoInstruct-TTS 走显式嵌入路线。

**创新判断**: 核心新贡献是 Emotion2embed 的**语义-声学联合编码 + 序数强度约束**,以及 ICE-Flow 的**分布正则化**。这两点在知识库已有方法中未被覆盖。但从知识库全局看,48 类离散情感标签系统在连续控制 (EmoCtrl-TTS/UDDETTS 的 AV 空间) 和 training-free 方法 (EmoSteer-TTS/DUET) 已经成熟的背景下,位置偏保守。

## 🔬 方法详解

### 整体架构: 双路径设计 [Fig 2]

EmoInstruct-TTS 将情感控制分为两条路径:

**Path 1: Instruction → Emotion2embed** (情感声学控制)
- 自然语言情感指令 → MiniLM 编码 → ICE-Flow → Emotion2embed 嵌入

**Path 2: Instruction + Text → LLM** (语义规划)
- 指令 + 内容文本 + Emotion2embed → Qwen2.5-0.5B (LoRA, rank 32, 9.87M params) → 语义 token 序列

**Speech Generation**: 语义 token + Emotion2embed + Speaker embedding → CFM-TTS → Mel → BigVGAN → 波形 [Eq. 11-12]

这种设计的核心思想是: 自然语言指令同时负责语义层面 (说什么、怎么说) 和情感层面的控制,但两条路径的表示方式不同 — LLM 处理语义,Emotion2embed 处理声学情感信号。

### Emotion2embed: 语义-声学联合情感表示 [§2.1.1]

**编码**: 给定情感语音 x 和配对文本描述 t:
- 语义分支: Sentence-BERT (bge-large-zh v1.5) 提取文本特征 f_text(t)
- 声学分支: ECAPA-TDNN 提取声学特征 f_acoustic(x)
- 拼接并线性投影: z_emo = W[f_text(t); f_acoustic(x)] + b, z_emo ∈ R^896 [Eq. 1]

**训练目标**: 多任务学习 [Eq. 2]:
- L_cls^emo: 情感分类 (27 类)
- L_cls^int: 强度分类 (low/medium/high)
- L_ord: 序数强度约束 — ℓ2 归一化后,对每种情感学习方向向量 u_e,通过 margin-based ranking loss 强制 low < medium < high 的投影顺序 [Eq. 3]

**覆盖范围**: 27 细粒度情感类别 + 7 主类 × 3 强度级 = 21 强度组合,合计 48 种情感状态。

**设计动机**: 纯语义嵌入 (如 InstructTTS 的文本 prompt) 缺乏声学锚定,纯声学嵌入 (如 EmoCtrl-TTS 的 wav2vec 2.0) 缺乏语义结构。Emotion2embed 通过联合编码同时获得语义可解释性和声学保真度。

### ICE-Flow: 从指令到嵌入的条件 Flow 模型 [§2.1.2]

**目的**: 推理时没有参考语音,需要从自然语言指令直接生成 Emotion2embed。

**编码**: MiniLM encoder 编码指令文本 → h_text

**训练**:
1. **样本级回归** [Eq. 6]: 用真实语音提取的 Emotion2embed 作为监督目标,MSE 损失
2. **分布级正则化** [Eq. 7]: 对齐生成嵌入与真实嵌入的协方差矩阵 (Frobenius 范数),防止 mode collapse — 即生成的嵌入不会退化为类别原型

**推理**: CFM sampler (25 步 Euler) + classifier-free guidance 控制指令遵循强度

**延迟**: MiniLM forward ~2ms + CFM 25 步 ~3ms,总计 <5ms,占端到端延迟 <1% [§3.5]

### 关键设计选择

1. **为什么不直接用 Emotion2vec?** 图 3(a,d) 的 t-SNE 对比显示,Emotion2vec 的情感类别严重重叠,强度结构混乱; Emotion2embed 形成更紧凑的聚类且强度方向一致 [Fig 3(b,e)]
2. **为什么需要分布正则化?** Table 1 显示,没有 L_dist 时 PCS=0.73 (仍有较强原型坍缩),加入后 PCS=0.67, VR=0.96 (接近真实分布), IOA 0.86→0.91

## 📊 实验与结果

### 实验设置

- **数据**: ESD + CNCED,合计 ~78k 条训练数据 + 7,600 评估
- **标注**: Dataset-Base 用 Gemini-2.5 Pro 自动生成 caption (弱监督), Dataset-Annotation 人工验证
- **Baseline**: CosyVoice2, CosyVoice3
- **评估**: MOS (自然度), ESMOS (情感相似度), SSMOS (说话人相似度), ECS (Emotion2embed 余弦相似度), WER
- **评估者**: 20 位语音专家, 每样本 3 人评分
- **设定**: 零样本, 男/女分开评估

### 主要结果

**21 种情感-强度任务** [Table 2]:

| 模型 | MOS (F) | ESMOS (F) | MOS (M) | ESMOS (M) |
|------|---------|-----------|---------|-----------|
| CosyVoice2 | 4.18 | 3.92 | 4.14 | 3.80 |
| CosyVoice3 | 4.15 | 3.98 | 4.08 | 3.92 |
| **EmoInstruct-TTS** | **4.28** | **4.25** | **4.25** | **4.10** |

ESMOS 提升显著: Female +0.27 vs CosyVoice3, Male +0.18 vs CosyVoice3 [Table 2]。

**27 种细粒度情感任务** [Table 3]:

| 模型 | MOS (F) | ESMOS (F) | MOS (M) | ESMOS (M) |
|------|---------|-----------|---------|-----------|
| CosyVoice2 | 3.98 | 3.55 | 3.90 | 3.48 |
| CosyVoice3 | 3.92 | 3.50 | 3.86 | 3.44 |
| **EmoInstruct-TTS** | **4.12** | **3.92** | **4.05** | **3.78** |

细粒度任务上优势更大: Female ESMOS +0.37 vs CosyVoice2, +0.42 vs CosyVoice3 [Table 3]。

**48 类客观评估** [Table 4]: ECS 0.870 (最高); WER 0.0259 (CosyVoice3 最低 0.0197)。CosyVoice3 的低 WER 归因于 speech-token-centric 建模,EmoInstruct-TTS 的 WER 仍然竞争性可接受。

### 消融实验

| 变体 | ESMOS (F/21) | WER (48) |
|------|-------------|----------|
| Dual-Path (完整) | 4.25 | 0.0259 |
| w/o Emo2emb (Text-Only) | 3.78 | 0.0329 |
| w/o Text Instruct (Emo2emb-Only) | 3.99 | **0.0486** |

关键发现 [Table 2, 4]:
- 去掉 Emotion2embed: ESMOS 大幅下降 (4.25→3.78),说明纯文本指令不足以传达细粒度情感 [Table 2]
- 去掉文本指令: ESMOS 接近 baseline (3.99 vs CosyVoice3 3.98),但 WER 暴涨 (0.0486),说明没有语义规划时语言质量失控 [Table 4]
- 这证明了双路径设计的必要性: 两条路径互补,缺一不可

### ICE-Flow 分布分析 [Table 1]

| 变体 | PCS↓ | VR≈1 | SWD↓ | IOA↑ |
|------|------|------|------|------|
| ICE-Mean (原型回归) | 0.89 | 0.65 | 0.58 | 0.78 |
| ICE-Sample (样本级) | 0.73 | 0.88 | 0.35 | 0.86 |
| ICE-Sample+Dist (完整) | **0.67** | **0.96** | **0.22** | **0.91** |

分布正则化将 VR 从 0.88 提升到 0.96 (接近理想 1.0),SWD 从 0.35 降到 0.22,证明协方差对齐有效抑制了 mode collapse [Table 1]。

## 🔗 与已有工作的对比

### vs EmoCtrl-TTS (Wu et al., 2024)

| 维度 | EmoCtrl-TTS | EmoInstruct-TTS |
|------|-------------|-----------------|
| 控制粒度 | 帧级 (0.5s/0.25s chunk) | 句级 |
| 情感表示 | 连续 arousal-valence + laughter embed | 离散 48 类嵌入 |
| 训练数据 | 27k 小时伪标签 | ~50k 条 (~几百小时) |
| 控制输入 | 参考语音 AV 值 | 自然语言指令 |
| 额外开销 | 需要参考语音 | ICE-Flow <5ms |

EmoCtrl-TTS 在时变控制和连续空间上更灵活,但需要参考语音; EmoInstruct-TTS 的指令驱动更符合用户交互场景。

### vs EmoSteer-TTS/DUET (Training-free 系列)

这些方法完全不需要训练,通过激活空间的 steering vector 控制情感。EmoInstruct-TTS 需要训练 Emotion2embed + ICE-Flow,但提供了显式的类别+强度解释性。两者路线互补: steering 系列适合快速部署,EmoInstruct-TTS 适合需要精细标签控制的场景。

### vs InstructTTS/Parler-TTS (指令驱动系列)

InstructTTS 和 Parler-TTS 用纯文本指令驱动风格,没有独立的情感嵌入。EmoInstruct-TTS 的核心改进是: 文本指令不足以传达细粒度声学情感 (消融实验 w/o Emo2emb ESMOS 3.78 证明了这一点 [Table 2]),需要额外的声学锚定嵌入。

### vs UMETTS (多模态 prompt)

UMETTS 用视觉+音频+文本多模态 prompt 对齐情感。EmoInstruct-TTS 只用文本指令但通过 ICE-Flow 桥接到声学空间。UMETTS 基于 VITS,EmoInstruct-TTS 基于 CosyVoice 系列,后端更现代。

### 未对比的缺失

论文未与 EmoCtrl-TTS、EmoSteer-TTS、WeSCon、DiffRO、RLAIF-SPA、UDDETTS 等当前主流 emotional TTS 方法对比。仅以 CosyVoice2/3 作为 baseline,这使得方法的竞争力判断存在不确定性。Interspeech 5 页限制可能是原因之一。

## 💡 启发与可迁移经验

1. **语义-声学联合嵌入 + 序数约束**: Emotion2embed 的设计思路 — 用文本描述提供语义结构,用声学特征提供保真度,用 margin-based ranking loss 强制强度排序 — 是一种通用的"受控嵌入空间"构造方法,可迁移到其他受控生成任务 (如说话风格、年龄控制)。

2. **分布正则化抗 mode collapse**: ICE-Flow 的协方差对齐策略 (L_dist) 简单且有效,可用于任何条件生成模型中 — 当模型倾向生成类别中心点时,用真实数据的协方差矩阵作为正则化目标,保持生成分布的多样性。

3. **双路径解耦的工程价值**: 将 "语义规划" 和 "风格/情感控制" 分到两条路径,允许独立升级。语义路径可换更强 LLM,情感路径可扩展类别或改用连续空间,互不干扰。这是一种模块化设计原则。

4. **Gemini-2.5 Pro 自动标注**: 用 MLLM 为语音数据自动生成 emotion caption 的 pipeline,49,903 条数据的弱监督预训练 + 28,402 条人工标注微调,是一种实用的 semi-supervised 数据扩展策略。

## ❓ 存疑与待验证

1. **竞争力存疑**: 仅与 CosyVoice2/3 对比,未与 EmoCtrl-TTS (ESMOS/Aro-Val SIM 有更细粒度的基准)、EmoSteer-TTS (training-free, F5-TTS 上 EI-MOS 4.00)、WeSCon (词级控制) 等当前 SOTA 方法对比。无法判断 Emotion2embed 相比 arousal-valence 连续空间或 activation steering 的真实优势。

2. **48 类离散标签的可扩展性**: 48 类情感标签是预定义的,覆盖范围受限。作者也承认这是局限,计划扩展到开放式描述。但在 UDDETTS 已实现 ADV 连续空间 (89.35% 覆盖率)、EmoSteer-TTS 实现多情感组合加法的背景下,离散标签系统的竞争力值得质疑。

3. **强度控制的真实效果**: Table 2 的 21 种情感-强度任务整体 ESMOS 高,但没有展示各强度级之间的区分度 (如 low vs high 的感知差异有多大)。IOA 0.91 是嵌入空间的指标,不等于合成语音中听感上的强度区分。

4. **ECS 指标的公正性**: 客观评估使用 Emotion2embed Cosine Similarity (ECS) 作为主要指标 — 但 ECS 正是用 EmoInstruct-TTS 自己训练的嵌入器计算的。这存在"自评"偏差。建议用 Emotion2vec 或独立 SER 分类器评估更公正。

5. **数据标注成本**: Emotion2embed 需要 (语音, 情感文本描述) 配对数据。28,402 条人工标注 + 49,903 条 Gemini-2.5 Pro 标注,成本不低。相比 EmoSteer-TTS 只需 ~7k 条情感语音 (不需要文本描述),数据效率偏低。

6. **CosyVoice3 baseline 的 WER 差距**: EmoInstruct-TTS WER 0.0259 vs CosyVoice3 0.0197 [Table 4],说明情感嵌入的引入对语义准确性有一定负面影响。消融中 Emo2emb-Only WER 暴涨至 0.0486 更是警示: 没有语义路径的约束,情感控制会严重损害内容准确性。

> [!review] 审阅 (auto, 2026-07-02)
> verdict: pass-with-fixes | avg: 4.4
> 详见 [[_review/EmoInstruct-TTS-review]]
