---
type: concept
title: "Emotion Control in TTS"
aliases: [情感控制TTS, Emotional TTS, Emotion Synthesis, 情感语音合成, Affective Speech Synthesis]
category: "technique"
tags: [TTS, emotion, expressiveness, control, affective-computing, style]
key_papers: ["Li et al. (2021)", "MsEmoTTS (Lei et al., 2022)", "Emo-DPO (Gao et al., 2024)", "EmoSphere++ (Cho et al., 2024)", "Rong et al. (2025)", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/Step-Audio2.5|StepAudio 2.5]]", "[[论文笔记/SCVALL-E|SC VALL-E]]", "[[论文笔记/NVSpeech|NVSpeech]]", "[[论文笔记/FlexiVoice|FlexiVoice]]", "[[论文笔记/EmotionThinker|EmotionThinker]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/TextrolSpeech|TextrolSpeech]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/NaturalVoices|NaturalVoices]]", "[[论文笔记/SpeechWorldModel|SpeechWorldModel]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/TTS-PRISM|TTS-PRISM]]", "[[论文笔记/Llama-VITS|Llama-VITS]]", "[[论文笔记/EmoSphere-TTS|EmoSphere-TTS]]", "[[论文笔记/Daisy-TTS|Daisy-TTS]]", "[[论文笔记/ControllingEmotionTTSNLPrompts|Bott et al. (Interspeech 2024)]]", "[[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]", "[[论文笔记/StoryTTS|StoryTTS]]", "[[论文笔记/UMETTS|UMETTS]]", "[[论文笔记/EmoSphere++|EmoSphere++]]", "[[论文笔记/AutoStyle-TTS|AutoStyle-TTS]]", "[[论文笔记/PROEMO|PROEMO]]", "[[论文笔记/EmoVoice|EmoVoice]]", "[[论文笔记/OpenOmni|OpenOmni]]", "[[论文笔记/CSP-FT|CSP-FT (Wang et al., 2026)]]", "[[论文笔记/DialogueAgents|DialogueAgents]]", "[[论文笔记/FaceSpeak|FaceSpeak]]", "[[论文笔记/Multi-StepHierarchicalED|Multi-Step Hierarchical ED (Inoue et al., 2025)]]", "[[论文笔记/MAEStyle-RichTTS|MAE Style-Rich TTS]]", "[[论文笔记/Prompt-Unseen-Emotion|PUE (Gao et al., 2025)]]", "[[论文笔记/DiEmo-TTS|DiEmo-TTS]]", "[[论文笔记/UDDETTS|UDDETTS]]", "[[论文笔记/MPE-TTS|MPE-TTS]]", "[[论文笔记/OpenS2S|OpenS2S]]", "[[论文笔记/TTS-CtrlNet|TTS-CtrlNet]]", "[[论文笔记/Spotlight-TTS|Spotlight-TTS]]", "[[论文笔记/CapSpeech|CapSpeech]]", "[[论文笔记/EME-TTS|EME-TTS]]", "[[论文笔记/NonverbalTTS|NonverbalTTS]]", "[[论文笔记/DeepDubbing|DeepDubbing]]", "[[论文笔记/LibriQuote|LibriQuote]]", "[[论文笔记/EmoSteer-TTS|EmoSteer-TTS]]", "[[论文笔记/EmoSSLSphere|EmoSSLSphere]]", "[[论文笔记/TTS-1|TTS-1 (Inworld, 2025)]]", "[[论文笔记/DiffRO|DiffRO]]", "[[论文笔记/ParaStyleTTS|ParaStyleTTS]]", "[[论文笔记/RLAIF-SPA|RLAIF-SPA]]", "[[论文笔记/HD-PPT|HD-PPT]]", "[[论文笔记/WeSCon|WeSCon]]", "[[论文笔记/UltraVoice|UltraVoice]]", "[[论文笔记/BatonVoice|BatonVoice]]", "[[论文笔记/Audiobook-CC|Audiobook-CC]]", "[[论文笔记/ECSS|ECSS]]", "[[论文笔记/MFCIG-CSS|MFCIG-CSS]]", "[[论文笔记/SASLM|SASLM]]"]
origin_paper: "Xie et al., Controllable TTS in LLM Era, 2024"
related_concepts: ["[[ProsodyModeling]]", "[[StyleTransferinTTS]]", "[[SpeechFactorization]]", "[[DifferentiableRewardOptimization]]", "[[LLM-basedTTS]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Emotion Control in TTS 旨在使合成语音表达特定的情感状态 (affective state),如快乐、悲伤、愤怒、恐惧、惊讶、厌恶等。这是可控 TTS 的核心任务之一,直接影响人机交互的自然度、叙事表现力和虚拟助手的情感智能。

**核心挑战** (Survey 指出):
- 情感与其他语音属性 (timbre, prosody) 深度纠缠
- 情感跨越多个粒度 (utterance-level, word-level, phoneme-level)
- 细粒度情感 (如讽刺) 难以标注和建模
- 长语音中保持情感一致性

## 情感建模方法

### 1. Emotion Embedding (情感嵌入)

直接学习情感的向量表示:
- **One-hot emotion label**: 离散情感类别 (angry, happy, sad, neutral)
- **Learned emotion embedding**: 从标注数据学习连续情感向量
- **Style alignment**: Li et al. (2021) 引入 emotion embeddings + style alignment 调制情感强度

### 2. 层级情感建模 (Hierarchical)

MsEmoTTS (Lei et al., 2022) 的多尺度方法:
- **Global level**: 整句情感基调
- **Utterance level**: 句子内情感变化
- **Local level**: 词/音素级情感细节
- 分层结构捕捉情感在不同粒度的表现

[[论文笔记/METTS|METTS]] (Zhu et al., 2023, 同组) 将多尺度情感建模扩展到跨语言场景: coarse-grained GST (L2 normalization 消除 magnitude 差异) 编码 language-agnostic 情感,fine-grained CVAE (以 text+coarse 为条件) 编码 language-specific 韵律细节。配合 formant-shift information perturbation 实现跨说话人跨语言情感迁移。中英双语 MOS 4.11 (intra) / 4.00 (cross-lingual) [Table II]。

### 3. 对抗训练解耦 (Adversarial Disentanglement)

将情感与说话人身份分离:
- Cross-speaker emotion transfer (Li et al., 2022): 对抗训练消除 speaker 信息
- 结果: 可将一个说话人的情感迁移到另一个说话人声音中
- [[论文笔记/ZET-Speech|ZET-Speech]] (Kang et al., 2023): 在零样本自适应 TTS 中使用 GRL+DAT 解耦 style vector 中的情感信息,结合 diffusion guidance (CG/CFG) 增强情感表达,是将零样本 TTS 与情感控制结合的早期工作

### 4. DPO/RLHF 优化

Emo-DPO (Gao et al., 2024):
- 使用 Direct Preference Optimization 实现情感控制
- 将 LLM 对齐技术引入 TTS 情感调优
- 不需要显式情感标签,从偏好数据学习

### 5. Emotion-adaptive 表示

EmoSphere++ (Cho et al., 2024):
- Emotion-adaptive spherical vector 表示
- 零样本情感可控 TTS
- 球面空间建模情感分布

## 与其他控制维度的交互

Survey 特别指出情感控制的难点:

> "Emotion and other vocal traits are often intertwined and span multiple granularities, making fine-grained control especially difficult."

**纠缠问题**:
- 修改 pitch 可能同时影响情感和自然度
- 情感变化伴随 duration、energy、voice quality 的联动
- 同一情感在不同说话人中表现不同

**解耦方案**:
- 对抗训练: 分类器 + gradient reversal
- Information bottleneck: 独立分支编码 emotion vs speaker vs content
- Pre-trained model guidance: 预训练情感分类器引导特征分离 (An et al., 2022; Wang et al., 2023b)

## 数据集

| 数据集 | 语言 | 情感类别 | 特点 |
|--------|------|----------|------|
| IEMOCAP (Busso et al., 2008) | EN | 多类 | 交互式, 多模态 |
| RAVDESS (Livingstone & Russo, 2018) | EN | 8类 | 演员表演, 歌曲+语音 |
| RECOLA (Ringeval et al., 2013) | FR | 连续维度 | 协作互动 |
| Toloka (2024) | Multi | 多类 | 众包标注 |
| ESD | ZH/EN | 5类 | 平行语料 |
| [[论文笔记/KazEmoTTS|KazEmoTTS]] (2024) | KZ | 6类 | 首个哈萨克语情感TTS, 74.85h, 3叙述者, CC-BY-4.0 |

## 评估方法

- **Emotion accuracy**: 情感分类器对合成语音的识别准确率
- **MOS (expressiveness)**: 人类评估情感表现力
- **AB/ABX test**: 对比不同系统的情感表达质量
- **GPT-based evaluation** (Rong et al., 2025): 使用 LLM 评估情感一致性

## 在 TTS 中的应用

- 虚拟助手: 情感智能对话
- 有声书/影视: 角色情感表演
- 心理健康: 共情语音生成
- 游戏 NPC: 动态情感反应

## 关键论文

- Li et al. (2021): emotion embeddings + style alignment 调制强度
- MsEmoTTS (Lei et al., 2022): 多尺度情感迁移、预测与控制
- Li et al. (2022): 跨说话人情感解耦与迁移
- Emo-DPO (Gao et al., 2024): DPO 优化情感合成
- [[论文笔记/EmoSphere-TTS|EmoSphere-TTS]] (Cho et al., Interspeech 2024): AVD 伪标签 + 笛卡尔→球面坐标变换,解耦情感风格/强度控制
- EmoSphere++ (Cho et al., 2024): 球面向量零样本情感控制 (EmoSphere-TTS 扩展版)
- EmoVoice (Yang et al., 2025): LLM-based 自由文本情感提示

## 相关概念

- [[ProsodyModeling]]: 情感通过韵律变化实现
- [[StyleTransferinTTS]]: 情感是风格的子维度
- [[SpeechFactorization]]: 情感与 speaker/content 解耦
- [[DifferentiableRewardOptimization]]: Emo-DPO 的技术基础
- [[LLM-basedTTS]]: 新范式下的情感控制方式

## 前端韵律分句的情感感知 (EmoPP)

[[论文笔记/EmotionAwareProsodic|EmoPP]] (Liu et al., 2023) 从 TTS 文本前端 (prosodic phrasing) 而非声学后端切入情感控制。通过 BERT (语言特征) + RoBERTa (文本情感预测) 联合训练 phrase break predictor,使停顿模式感知情感上下文。在 ESD 平行语料上实证验证了不同情感产生不同 phrase break 模式 (SMC 0.90-0.92),IEMOCAP 上 F1 78.43 (vs BERT+BiLSTM 77.48) [Table 2],接入 emotional TTS 后 EMOS 4.09 vs 3.84 [Table 3]。与后续声学层方法的区别: EmoPP 在文本处理阶段引入情感,产出的 phrase break sequence 作为 TTS 的输入之一,是上游切入点。

## 帧级 Arousal-Valence 条件控制 (EmoCtrl-TTS)

EmoCtrl-TTS (Wu et al., 2024) 在 flow-matching zero-shot TTS 上同时使用两组帧级条件: (1) arousal-valence 值 (来自 wav2vec 2.0-based extractor, chunk-wise 0.5s/0.25s) 控制时变情感; (2) laughter detector embedding (32 维) 控制 NV (笑声、哭泣等)。用 27k 小时伪标签真实情感数据训练,在 JVNV S2ST 上 Aro-Val SIM 0.643 (超越 ELaTE 0.548)。关键发现: laughter detector embedding 能泛化到哭泣等非笑声 NV; 两种 embedding 在某些数据上存在负面交互,需按数据源选择性启用。与 NVSpeech 的离散 PV 标签方法互补: EmoCtrl-TTS 用连续 embedding 实现帧级控制,NVSpeech 用离散标签实现 token-level 控制。详见 [[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]。

## 副语言发声方法 (NVSpeech)

NVSpeech (Liao et al., 2025) 从不同角度切入情感表达 — 不直接建模抽象情感状态,而是建模具体的副语言行为 (笑声、叹气、犹豫等),这些行为是情感的外在表现。通过在文本中显式插入 `[Laughter]`、`[Breathing]` 等标签实现 token-level 控制。与传统情感控制互补: 情感控制提供高层意图,PV 控制提供底层行为实现。详见 [[论文笔记/NVSpeech|NVSpeech]]。

[[论文笔记/NaturalEmotionalTTS|NaturalEmotionalTTS]] (Zhou et al., NAIST, 2026) 从**数据标注方案**角度切入 NV 控制,提出频率-时长编码标注方案: 离散发声 (如 laughter) 用音节重复控制频率,连续发声 (如 crying) 用字符重复控制时长。基于 Grad-TTS + arousal-valence emotion encoder 构建 NV emotional TTS。评估显示细粒度 NV 显著提升 eMOS (4.20 vs verbal-only 3.81) 和情感识别率 (82.0% vs 62.1%),尤其 fear (+36%) 和 happy (+17%) 受益最大,angry 改善有限 (缺乏专属 NV)。与 NVSpeech/EmoCtrl-TTS 的区别: 不依赖 LLM-based TTS 或大规模数据,纯数据标注驱动; 局限在于 backbone 过时 (Grad-TTS) 且规模极小 (739 条 NV)。

## 演进

规则情感合成 (HMM, 2003) → Emotion embedding (2021) → 多尺度层级建模 (MsEmoTTS, 2022) → 跨说话人情感迁移 (2022) → 韵律嵌入分解 (Daisy-TTS, 2024) → DPO/RLHF 对齐 (Emo-DPO, 2024) → 零样本情感 (EmoSphere++, 2024) → LLM 自由文本情感 (EmoVoice, 2025) → 副语言行为控制 (NVSpeech, 2025) → LLM prompt 混合情感 (PUE, 2025) → ADV 维度解耦控制 (UDDETTS, 2025) → Training-free 激活 steering (EmoSteer-TTS, 2025) → 结构化 AI 反馈 (RLAIF-SPA, 2025) → Self-training 词级情感控制 (WeSCon, NeurIPS 2025) → Training-free attention mask intra-utterance 多情感 (TED-TTS, 2026) → 双空间跨架构 plug-and-play (DUET, 2026) → SAE 稀疏特征 steering (SAE-Emotion, ICML 2026)

## DNN-based 层级 ED + Flow Matching (HierarchicalEmotionControl)

[[论文笔记/HierarchicalEmotionControl|Hierarchical Emotion Control]] (Inoue et al., 2024) 是同组 Multi-Step Hierarchical ED (2025) 的前序 journal 版本,将层级 ED 提取从 SVM 升级为 DNN 分类器,并在 MatchaTTS (OT-CFM) 上构建情感 TTS 框架。核心贡献: (1) Emotion Presence Recognizer (EPR) — 将多分类 SER 拆分为 4 个独立二分类器,各判断"是否存在某情感",输出非竞争的连续强度值; (2) OpenSMILE (word/phoneme) + WavLM (utterance) 混合声学特征,利用两者在不同粒度的互补优势; (3) GRL (gradient scale 0.5) 对抗训练解耦 speaker/gender 信息。EPR + Combination 配置在 ESD 英文子集上全面优于 SVM-based HED baseline: Emotion Score 0.369 vs 0.120, MIG 0.089 vs 1.843 [Table III, IX]。与后续 Multi-Step 版本的区别: 本文为单步并行提取三级 ED,Multi-Step 升级为 utterance→word→phoneme 顺序预测。

## 多步层级情感分布预测 (Multi-Step Hierarchical ED)

[[论文笔记/Multi-StepHierarchicalED|Multi-Step Hierarchical ED]] (Inoue et al., 2025) 提出多步预测框架,将情感量化为 utterance/word/phoneme 三级连续分布向量 (Hierarchical ED),并按 utterance→word→phoneme 顺序逐级预测,使高层情感上下文引导底层韵律。ED 通过 OpenSMILE + SVM 排序函数从音频中提取,支持训练时自动标注和推理时用户手动调整。在 FastSpeech 2 上验证了两种集成方式 (External 模型无关 / VA 内嵌),Multi-Step 在 WER (2.45% vs 4.61%) 和 MUSHRA 自然度上均优于 Single-Step baseline。BWS 测试中情感可控性全面优于 MsEmoTTS。关键发现: ED 数值差异相近但合成质量差异显著,说明多步预测学到的是层级依赖关系而非更准确的数值。

## Plutchik 结构模型与韵律嵌入分解 (Daisy-TTS)

[[论文笔记/Daisy-TTS|Daisy-TTS]] (Chevi & Aji, 2024) 从 Plutchik 结构模型出发,提出韵律嵌入分解方法实现更宽广的情感模拟。核心思路: 用 emotion discriminator 训练 prosody encoder 学习情感可分离嵌入,再通过 PCA 分解实现四种情感操控 — 一级情感 (采样)、二级情感 (高斯混合)、强度 (缩放因子 alpha)、极性 (取反)。在 ESD 数据集上 MOS 和感知率均优于 Zhou et al. (2022b) baseline。该方法是 "情感表示 = 可分解韵律原型" 范式的首次探索。

## LLM Prompt-based 混合情感 (PUE)

[[论文笔记/Prompt-Unseen-Emotion|PUE]] (Gao et al., 2025) 提出另一条混合情感路线: 利用 LLM 的 in-context learning 能力,通过 emotion-guided prompt (百分比模板: "α% happy, β% sad, ...") 实现零样本混合情感合成。训练时每个样本仅有一种情感 (对应参数=100%, 其余=0%),推理时调整百分比即可生成 outrage (surprise+angry)、disappointment (surprise+sad)、delight (surprise+happy) 等混合情感。基于 CosyVoice-300M-Instruct 架构,AB Preference 测试中 PUE 以 68-87% 偏好率超越 VITS-based mix baseline。与 Daisy-TTS 的 PCA 分解路线互补: PUE 通过文本 prompt 实现组合,Daisy-TTS 通过嵌入空间操作实现组合。

## LLM-TTS 中的 ADV 维度情感控制 (UDDETTS)

[[论文笔记/UDDETTS|UDDETTS]] (Liu et al., 2025) 是首个在 LLM-based TTS 中引入 Arousal-Dominance-Valence (ADV) 空间的框架,实现三维解耦的可解释情感控制。与 EmoSphere-TTS/EmoSphere++ 使用笛卡尔→球面坐标变换不同,UDDETTS 采用基于聚类的非线性分箱 (nonlinear binning) 将 ADV 连续值量化为 14x14x14 的离散 token,避免了球面变换导致的情感簇扭曲。通过半监督训练统一仅有 label 和同时有 label+ADV 标注的异构数据集,将 ADV 空间覆盖率从 60.83% 提升到 89.35%。三种推理模式: label-controlled / ADV-controlled / end-to-end (ADV predictor 从文本预测 pseudo-ADV)。在 ADV 控制实验中,SRC 达 0.85-0.92,表明感知情感与 ADV 值线性相关。详见 [[论文笔记/UDDETTS|UDDETTS]]。

## 自监督蒸馏解耦 (DiEmo-TTS)

[[论文笔记/DiEmo-TTS|DiEmo-TTS]] (Cho et al., Interspeech 2025) 提出基于 DINO 自监督蒸馏的跨说话人情感解耦方案,避免了 GRL 的 trade-off 和 VQ 的信息丢失问题。核心方法: (1) cluster-driven sampling: 用情感属性预测 + k-means 聚类构建跨说话人情感 cluster,基于 cluster 而非 utterance 构造 DINO 正样本对; (2) formant-based information perturbation: 利用共振峰与音色的相关性精准破坏说话人身份; (3) dual conditioning transformer 融合 emotion 和 speaker embedding。在 ESD 上 nMOS 4.23, eMOS 4.07, SECS 0.8505。代表了情感解耦从"对抗训练/信息瓶颈"向"自监督蒸馏"范式的迁移。

## ControlNet 范式的情感控制 (TTS-CtrlNet)

[[论文笔记/TTS-CtrlNet|TTS-CtrlNet]] (Jeong et al., 2025) 首次将图像领域 ControlNet (Zhang et al., 2023) 范式迁移至 flow-matching TTS,实现"插件式"时变情感控制。核心方法: 冻结预训练 F5-TTS (22 个 DiT blocks) 的全部参数,创建部分 block 的可训练副本作为 ControlNet,通过 zero-convolution 连接。arousal-valence 条件由 wav2vec 2.0-based SER 提取,经窗口滑动插值后输入 ControlNet。三项关键工程发现: (1) emotion-specific flow step [0, 0.1] -- 情感信息仅在 ODE 早期步骤决定,后期步骤不需 ControlNet; (2) selective block -- 通过逐 block skip 消融排除对 WER 关键的 block; (3) control scale lambda 提供推理时情感强度连续调节。仅用约 400 小时公开数据训练,Emo-SIM 0.751 / Aro-Val SIM 0.742 超越全模型微调的 EmoCtrl-TTS (0.697/0.643)。与 EmoCtrl-TTS 路线互补: EmoCtrl-TTS 用大数据全参数微调,TTS-CtrlNet 用小数据冻结+旁挂。详见 [[论文笔记/TTS-CtrlNet|TTS-CtrlNet]]。

## 重音-情感交互建模 (EME-TTS)

[[论文笔记/EME-TTS|EME-TTS]] (Li et al., 2025) 首次系统探索重音 (emphasis) 与情感在 TTS 中的交互关系。基于 EmoSpeech (FastSpeech 2 情感扩展) 架构,提出两个核心组件: (1) variance-based emphasis features — 用 pitch/duration 的局部-全局偏差 (重音区域平均值 − 全句平均值) 建模重音,由 EmphaClass SSL 分类器提供弱监督伪标签; (2) Emphasis Perception Enhancement (EPE) block — 替换原始 FFT block,通过 Conditional Cross Attention (情感嵌入重加权注意力) + Emphasis Adapter (对重音区域做注意力权重加性调制) 确保重音在不同情感条件下保持感知清晰。在 ESD 数据集上,主观情感准确率 Mean 0.67 (vs EmoSpeech 0.58),MOS 4.22 (vs 4.14),重音识别准确率 0.78 (vs w/o EPE 0.73)。与已有路线的区别: 不从情感表示本身出发 (如 ADV 维度/韵律嵌入分解),而从重音这个韵律子维度切入间接增强情感表达力。详见 [[论文笔记/EME-TTS|EME-TTS]]。

## Training-free 激活 Steering (EmoSteer-TTS)

[[论文笔记/EmoSteer-TTS|EmoSteer-TTS]] (Xie et al., 2025) 提出首个完全 training-free 的细粒度情感可控 TTS 方法,将 LLM 领域的 activation steering 技术迁移到 flow-matching TTS。核心发现: flow-matching TTS 的 DiT 层内部激活值隐式编码了情感信息,通过 difference-in-means 提取情感/中性语音对的激活差,筛选 top-k 情感相关 token 构造稀疏 steering vector,在推理时通过强度参数 α 注入激活实现连续控制。支持四种操作: 情感转换 (α>0)、插值 (连续 α)、擦除 (投影减法)、多情感组合 (加法)。在 F5-TTS/E2-TTS/CosyVoice2 三个模型上验证,F5-TTS+EmoSteer EI-MOS 4.00 超越 EmoSphere++ (3.50) 和 HED-TTS (2.59),EE-MOS 4.02。与所有已有路线的根本区别: 不需要任何训练或微调,仅需 ~7k 条情感语音构造 steering vectors。与 TTS-CtrlNet 的对比: TTS-CtrlNet 需训练 ControlNet 旁挂,支持帧级时变控制; EmoSteer-TTS 零训练,支持全局连续强度控制和多情感组合。详见 [[论文笔记/EmoSteer-TTS|EmoSteer-TTS]]。

## Reward-guided 零样本情感控制 (DiffRO-MTR)

[[论文笔记/DiffRO|DiffRO]] (Gao et al., Tongyi Lab, 2025) 提出了一条独特的情感控制路线: 不通过情感嵌入/标签/steering 直接建模情感,而是用 SER reward model 的梯度间接引导 TTS LM 学习情感表达。核心方法: Multi-Task Reward (MTR) 模型在 13000+ 小时伪标签数据上训练 SER 能力,通过 Gumbel-Softmax 可微 pipeline 将 SER reward (log P(emotion|tokens)) 反向传播到 LM。训练时使用指令模板 "Your emotion is {E}" 但无需情感标注数据。结果: HAPPY accuracy 1.00(zh)/0.92(en), SAD 0.76/0.96, ANGRY 0.84/0.92,全面超越 CosyVoice 2.0/F5-TTS/GPT-SoVITS [Table 3]。一个有趣发现: 系统自发学会合成笑声、抽泣、呼吸等音频事件来传达情感 [Fig 3]。与其他路线的区别: 不需要情感嵌入空间 (vs EmoSphere/UDDETTS)、不需要标注数据 (vs Emo-DPO)、不需要 steering vectors (vs EmoSteer-TTS),情感知识完全从 reward model 蒸馏。

## 结构化 RLAIF 属性级情感对齐 (RLAIF-SPA)

[[论文笔记/RLAIF-SPA|RLAIF-SPA]] (Yang et al., 2025) 提出将 emotional TTS post-training 框定为 multi-attribute alignment problem,用 GRPO 优化 4 维结构化 AI 反馈 (Structure/Emotion/Speed/Tone 标签匹配 + WER 惩罚)。与 Emo-DPO 使用整体偏好信号不同,RLAIF-SPA 将反馈分解为属性级维度,实现更精确的信用分配。核心方法: LLM (GPT-4o) 自动标注文本的 4 维韵律-情感标签作为训练目标,Qwen2-Audio 评估生成语音的标签匹配度作为 reward,Whisper 计算 WER 作为语义准确性约束。基于 MiniCPM-O 2.6 + Chat-TTS,在 LibriSpeech/MELD/ESD 上 SIM-O 和 SER Avg 均为最高。消融分析显示属性级标签 reward 提供了跨属性解缠效果。与 DiffRO 的区别: RLAIF-SPA 在 audio 空间用 GRPO 做 group-relative 优化,DiffRO 在 token 空间通过 Gumbel-Softmax 做可微优化。详见 [[论文笔记/RLAIF-SPA|RLAIF-SPA]]。

## 多语言球面情感 + SSL 离散 token (EmoSSLSphere)

[[论文笔记/EmoSSLSphere|EmoSSLSphere]] (Park & Nakamura, SSW 2025) 在 EmoSphere-TTS 基础上向多语言和 SSL 韵律建模方向扩展。核心增量: (1) HuBERT 第 9 层特征经语言分别 k-means (K=200) 离散化为韵律 token,提供与球面 AVD 互补的局部韵律控制信号; (2) DeBERTaV3 语义编码器通过 cross-attention 条件化情感/韵律模块,实现语义感知的情感生成。在英日双语 (ESD/JVNV) 上优于 EmoSphere-TTS (EN WER 19.58% vs 20.96%, JA CER 18.33% vs 19.26%; nMOS EN 4.13 vs 4.05, JA 3.94 vs 3.63 [Table 1, 2])。消融显示 k-means 离散化优于连续 HuBERT 特征,语言分别聚类的 token 实际捕获了通用韵律模式。局限: 仅单说话人小规模实验,无零样本能力,未与 EmoSphere++ 或 LLM-TTS 对比。详见 [[论文笔记/EmoSSLSphere|EmoSSLSphere]]。

## Robust Reward Model for Emotional TTS (RRPO)

[[论文笔记/RRPO|RRPO]] (Wang et al., Tongyi Lab, 2026) 从 reward model 鲁棒性角度切入情感控制: 在 DiffRO 框架中,vanilla SER RM 会被 policy 通过生成声学伪影 (嘴部咔嗒声、爆破音) 欺骗获取虚假奖励 (reward hacking),导致 E-MOS 提升但 N-MOS 反降。核心方法: 三层混合正则化 fine-tune RM — Label Smoothing (修正过度自信) + Energy-Adaptive Mixup (基于语音能量平滑决策边界) + Adversarial Training (在高层 embedding 上增强扰动鲁棒性)。CosyVoice2 上 E-MOS 3.78 / N-MOS 3.81 均为最优; SER WA ESD 64.4→81.7% [RRPO Table 1, Table 2]。与 DiffRO-MTR 路线 (零样本情感) 互补: DiffRO-MTR 关注"用 SER reward 引导情感学习",RRPO 关注"确保 SER reward 本身可靠"。与 RLAIF-SPA 的区别: RLAIF-SPA 在 audio 空间用多维结构化反馈,RRPO 在 token 空间通过加固 RM 从根源阻止 hacking。详见 [[论文笔记/RRPO|RRPO]]。

## Self-Training 词级情感控制 (WeSCon)

[[论文笔记/WeSCon|WeSCon]] (Wang et al., NeurIPS 2025) 提出首个不依赖含 intra-sentence 情感转换数据的 word-level 情感和语速联合控制框架。核心方法: 两阶段 self-training — (1) Teacher: 冻结 CosyVoice2 backbone,通过多轮推理 (每段用不同情感 prompt) + transition smoothing (tail-to-head linkage) + dynamic speed control (prompt token 插值/下采样) 实现 word-level 控制; (2) Student: CosyVoice2 + Dynamic Emotional Attention Bias (DEAB,7 种预定义 attention bias 模板的加权组合) 在 teacher 伪标签上 self-training,实现端到端单次推理。仅用 ~500h 公开 ESD 数据 (无情感转换标注),Emo2v. 0.882 / DNSV 4.361 (EN) 全面超越 CosyVoice2 (0.866 / 7.894) 和 F5-TTS/Index-TTS,EMOS 3.70±0.17, NMOS 3.93±0.20; 零样本 TTS 性能几乎无损 (CER 1.47 vs 1.45)。与 EmoCtrl-TTS 的核心区别: 不需要 27k h 含情感转换的伪标签数据; 与 TTS-CtrlNet 的区别: 控制粒度为词级 (非帧级),通过 self-training 蒸馏而非 ControlNet 旁挂。详见 [[论文笔记/WeSCon|WeSCon]]。

## Training-free Attention Mask 实现 Intra-utterance 多情感控制 (TED-TTS)

[[论文笔记/TED-TTS|TED-TTS]] (Liang et al., NUS, 2026) 提出首个在预训练 AR zero-shot TTS 上实现 intra-utterance segment-level 多情感控制的 training-free 框架。核心方法: (1) 2D causal attention mask — 在标准 causal attention 上叠加 segment-local condition visibility 约束,文本/semantic token 只能 attend 到所属 segment 的 emotion condition embedding,而文本/semantic 之间保留全局 causal attention 保证语义连贯; (2) Monotonic Stream Alignment (MSA) — Bayesian 在线对齐追踪算法,通过 monotonic prior + 动态 head selection + Gaussian smoothing 将 noisy multi-head attention 转化为稳定的 text-semantic alignment,驱动 mask 切换。在 IndexTTS2 上验证,SMOS 4.00-4.22 / NMOS 4.07-4.22 (EN) 全面超越独立合成拼接的对比方法 [Table 1, 2]。与 EmoSteer-TTS 的关键区别: EmoSteer-TTS 通过激活空间 steering vector 注入全局情感,TED-TTS 通过 attention mask 实现 segment-level 切换; 与 WeSCon 的区别: WeSCon 需 self-training (~500h ESD data),TED-TTS 完全零训练。局限: 仅 segment-wise 离散切换,不建模渐进情感过渡; EMOS (情感准确率) 低于 baseline (3.42 vs 4.07 EN),反映 intra-utterance 控制中情感准确度与过渡自然度的 trade-off。详见 [[论文笔记/TED-TTS|TED-TTS]]。

## SLM 激活 Steering 实现可组合混合情感 (CoCoEmo)

[[论文笔记/CoCoEmo|CoCoEmo]] (Wang et al., Univ. Melbourne, 2026) 将 activation steering 从 flow-matching DiT (EmoSteer-TTS) 转移到 SLM (Speech Language Model) 阶段,聚焦混合情感和文本-情感错配两个更复杂的场景。核心贡献: (1) cross-conditioning diagnostic 证明在 hybrid TTS (SLM + flow matching) 中,情感韵律主要由 SLM 编码,flow-matching 主要做声学渲染 [§2.1, Table 1]; (2) discriminability-driven 层/操作选择 — 对 SLM 每层每操作训练 linear probe,用分类准确率衡量线性可分性,CosyVoice2 中 layers 10-17 的 attn_output 最优 [Fig 3]; (3) 混合情感 steering — 单情感 steering vector 的加权组合,权重可来自 multi-rater 标注共识分布 [Eq. 7]; (4) multi-rater 混合情感评估协议 (Spearman rho, H-Rate)。在 CosyVoice2 和 IndexTTS2 上验证,Mixed-emotion E-SIM 0.795 / TEP 0.315 (alpha=5.0, CREMA-D), High-mismatch E-SIM 0.862 / TEP 0.504 (alpha=6.0, IEMOCAP),全面超越 instruction-based baseline [Table 2, 3]。与 EmoSteer-TTS 的区别: EmoSteer-TTS 在 flow-matching DiT 层操作且使用 top-k token 稀疏选择,CoCoEmo 在 SLM 层操作且仅在 last-token 位置 steering; 两者可互补(CoCoEmo 也可叠加在已有情感控制方法之上)。局限: 仅 utterance-level 控制,仅验证 5 类离散情感。详见 [[论文笔记/CoCoEmo|CoCoEmo]]。

## 参数空间 Task Vector 情感控制 (TaskVectorTTS)

[[论文笔记/TaskVectorTTS|TaskVectorTTS]] (Feng et al., SJTU, 2025) 提出在参数空间而非激活/嵌入空间操作的情感控制方法。核心方法: 对 F5-TTS 分别在情感数据上微调,计算 task vector (微调参数差 τ = θ_ft - θ_pre),通过缩放系数 β 连续控制情感强度 (ε = β·τ)。在跨风格 (方言+情感) 场景中,通过层级合并策略 (Hierarchical Merging) 将情感 LoRA E-Vector 分配到 DiT 后半层,避免与方言控制干扰。情感方言合成 MOS 2.83 (HE-Vector) vs CosyVoice2 1.87 [Table 3]。与 EmoSteer-TTS 的区别: EmoSteer 在激活空间操作且 training-free,TaskVectorTTS 在参数空间操作需微调; 与 TTS-CtrlNet 的区别: TTS-CtrlNet 用 ControlNet 旁挂,TaskVectorTTS 用 task vector 直接修改权重。局限: 仅在 F5-TTS 上有效,应用于 CosyVoice 时质量下降。详见 [[论文笔记/TaskVectorTTS|TaskVectorTTS]]。

## Multi-Agent 闭环 Composite-Instruction 控制 (AgentSteerTTS)

[[论文笔记/AgentSteerTTS|AgentSteerTTS]] (Kang et al., ICML 2026) 首次系统性地解决 composite-instruction 场景下的情感控制问题 — 即多属性组合指令 (如 "Happy but slightly Arrogant") 的可靠生成。论文识别出两个根本瓶颈: (1) 确定性映射在多模态分布下产生 mode averaging,导致 target attribute suppression 25-45%; (2) speaker-emotion entanglement 造成 composite 控制中 timbre-prosody trade-off。提出三模块闭环方案: Adversarial Disentanglement Module (双向 GRL + cross-covariance 正交约束) 解耦 speaker-emotion → Dual-Stream Anchoring Controller (检索 acoustic prototype + gated fusion) 锚定目标区域 → Fast-Slow Feedback Agent (latent gradient correction + MLLM perceptual critique) 推理时校准。在 composite benchmark 上 E-SIM 0.955 (vs IndexTTS2 0.864),CSR 0.78,S-SIM 0.841 [Table 2, 3]。与已有路线的区别: PUE/Daisy-TTS/EmoSteer-TTS/CoCoEmo 从嵌入空间/激活空间操作混合情感,AgentSteerTTS 通过检索增强 + 闭环校准在声学空间直接解决 composite alignment。局限: 依赖 100h 人工筛选 prototype library 和外部 MLLM evaluator。详见 [[论文笔记/AgentSteerTTS|AgentSteerTTS]]。

## 双空间 Plug-and-Play 跨架构情感控制 (DUET)

[[论文笔记/DUET|DUET]] (Zhang et al., Macquarie Univ., 2026) 在 EmoSteer-TTS 的 activation steering 基础上做了两项关键扩展: (1) 从 DiT-only 泛化到 5 种架构差异巨大的 backbone (DiT/Transformer/U-Net, 覆盖 diffusion 和 flow-matching 两大范式); (2) 在 hidden space steering 之外增加 mel-space guidance (通过可微 vocoder Vocos 反传 SER 梯度修正频谱细节),形成双空间联合控制。核心发现: 预训练 TTS 的 hidden states 中情感方向与说话人方向近正交 (|cos θ| = 0.029, F5-TTS),情感仅占 variance 的 8.5% 但 linearly decodable [Fig 1]。方法: linear probe 定位情感最可分层 → SVD 提取多方向判别子空间 (超越 EmoSteer-TTS 的 difference-in-means) → norm-adaptive steering (按 ||h|| 缩放) + cosine-scheduled mel guidance (trust-region 约束)。消融: hidden steering -24.3%, mel guidance -19.5% (互补) [Table 2]。在 ESD 上 DUET+GradTTS Avg 75.5% vs 最强 baseline Qwen3-TTS 46.8%; EMOS 3.93 (最高) [Table 1, 3]。与 EmoSteer-TTS 的区别: DUET 用 linear probe + SVD 多方向而非 difference-in-means, 增加了 mel-space guidance, 泛化到 diffusion backbone; 与 CoCoEmo 的区别: DUET 在 flow-matching/diffusion denoiser 内部操作,CoCoEmo 在 SLM 层操作,两者可互补。局限: 仅 3 类离散情感,angry 表现弱 (仅达 GT ceiling 49%),未验证连续 AV 控制。详见 [[论文笔记/DUET|DUET]]。

## LLM 对话情感推理 for CSS (JELLY)

[[论文笔记/JELLY|JELLY]] (Cha et al., ICASSP 2025) 从 CSS (Conversational Speech Synthesis) 角度切入情感控制: 不直接建模情感声学表征,而是用 LLM (Vicuna-7B) 推理对话上下文中目标话语应有的情感状态,再传递给 FastSpeech 2 合成。核心方法: Emotion-aware Q-former (EQ-former) — Whisper 32 层 + TLTR 层注意力 + Q-former 对齐情感到文本空间; Partial LoRA (PLoRA) — 为情感 embedding 和文本 embedding 分别设置独立 LoRA adapter,避免模态干扰; 三阶段训练 — (1) 情感-文本对齐 (80.6h 多数据集), (2) 文本预训练+语音微调 (DailyDialog 文本 + DailyTalk 20h), (3) 情感条件合成。在 DailyTalk 上 E-DMOS 3.987 (vs ECSS 3.914), Emotion F1 60.17 (vs ECSS 13.66) [Table I, II]。与 CapTalk/EmotionThinker 等后续 CoT 路线的区别: JELLY 用 LLM next-token prediction 隐式推理情感,不显式输出情感规划 token; 与 steering 系列 (EmoSteer/CoCoEmo) 的区别: JELLY 是"先推理后合成"的解耦策略,不在 TTS 内部操控情感表征。局限: FastSpeech 2 后端过时,DailyTalk 仅 20h/2 说话人。详见 [[论文笔记/JELLY|JELLY]]。

## CoT 显式规划对话 Turn-level 表达 (CapTalk)

[[论文笔记/CapTalk|CapTalk]] (Su et al., Hello Group, 2026) 提出了一条不同于 embedding/steering/reward 的情感-表达控制路线: 在对话 TTS 中使用 Chain-of-Thought 控制序列显式规划 turn-level 动态属性。CoT 包含 5 个属性 (emotion/tone/pitch/energy/speed),其中 emotion/tone 是高层情感-交际意图,pitch/energy/speed 是低层韵律实现 (建模为 speaker-internal relative prosody,归一化到各说话人基线)。CoT 在训练时通过 Qwen3-Omni 从语音中提取,推理时由模型从对话上下文自回归预测。400 样本评估中 CoT prediction accuracy 0.7675-0.9125,controllability success rate 0.7675-0.8675 [Table 3]; w/ CoT vs w/o CoT 人工偏好 65.5% vs 34.5% [Table 4]。与已有路线的根本区别: 不在 embedding/activation/参数空间操作情感表征,而是将表达意图外化为可读的文本 token 序列,作为生成的前置规划。详见 [[论文笔记/CapTalk|CapTalk]]。

## SAE 稀疏特征可解释情感 Steering (SAE-Emotion)

[[论文笔记/SparseAutoencoderEmotion|SAE-Emotion]] (Du et al., William & Mary, ICML 2026) 将 mechanistic interpretability 社区的 Sparse Autoencoder (SAE) 工具引入 TTS 情感控制,首次在 AR semantic backbone (IndexTTS2 layer-16 residual stream) 上分解情感信号为稀疏可解释 latent features。核心方法: 训练 k-sparse autoencoder (4096 维, Top-32 active) 将 residual stream 映射为 overcomplete 稀疏激活,通过 sentence-level selectivity score (paired emotion-neutral 激活频率差异) 选出 top-6 情感相关 features,等权组合后通过 SAE decoder 方向做 bidirectional steering (alpha>0 诱导, alpha<0 抑制)。关键发现: (1) 情感信号稀疏分布 — 绝大多数 features 的 selectivity 集中在 0 附近,仅极少数显著正偏; (2) 不同情感 top-6 features 无重叠; (3) 单 feature 对应可解释声学属性 (F0 +23.11 Hz, spectral centroid 单调变化); (4) 强 steering 下 WER 0.57% vs global steering 2.86%。Emo-SIM anger 0.912 / happiness 0.885 / sadness 0.880 (induction), EMOS 3.22 / NMOS 3.49 (人类评估最高) [Table 1, 2]。与 EmoSteer-TTS 的区别: EmoSteer-TTS 在 flow-matching DiT 中用 difference-in-means 稠密方向 + top-k token 选择,SAE-Emotion 在 AR semantic backbone 中用 SAE 分解为 feature-level 稀疏方向; 与 CoCoEmo 的区别: CoCoEmo 也在 SLM 层操作但用 linear probe + last-token steering,SAE-Emotion 在所有 token 位置操作; 与 DUET 的区别: DUET 用 SVD 多方向跨 5 种架构,SAE-Emotion 仅验证 1 种架构但可解释性更深。局限: 仅 IndexTTS2 完整验证,仅 3 类离散情感,SAE 训练需 56k activations。详见 [[论文笔记/SparseAutoencoderEmotion|SAE-Emotion]]。
