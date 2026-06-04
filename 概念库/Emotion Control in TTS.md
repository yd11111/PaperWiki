---
type: concept
title: "Emotion Control in TTS"
aliases: [情感控制TTS, Emotional TTS, Emotion Synthesis, 情感语音合成, Affective Speech Synthesis]
category: "technique"
tags: [TTS, emotion, expressiveness, control, affective-computing, style]
key_papers: ["Li et al. (2021)", "MsEmoTTS (Lei et al., 2022)", "Emo-DPO (Gao et al., 2024)", "EmoSphere++ (Cho et al., 2024)", "Rong et al. (2025)", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/Step-Audio 2.5|StepAudio 2.5]]", "[[论文笔记/SC VALL-E|SC VALL-E]]", "[[论文笔记/NVSpeech|NVSpeech]]", "[[论文笔记/FlexiVoice|FlexiVoice]]", "[[论文笔记/EmotionThinker|EmotionThinker]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/TextrolSpeech|TextrolSpeech]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/NaturalVoices|NaturalVoices]]", "[[论文笔记/SpeechWorldModel|SpeechWorldModel]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/TTS-PRISM|TTS-PRISM]]", "[[论文笔记/Llama-VITS|Llama-VITS]]", "[[论文笔记/EmoSphere-TTS|EmoSphere-TTS]]", "[[论文笔记/Daisy-TTS|Daisy-TTS]]", "[[论文笔记/Controlling Emotion TTS NL Prompts|Bott et al. (Interspeech 2024)]]", "[[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]", "[[论文笔记/StoryTTS|StoryTTS]]", "[[论文笔记/UMETTS|UMETTS]]", "[[论文笔记/EmoSphere++|EmoSphere++]]", "[[论文笔记/AutoStyle-TTS|AutoStyle-TTS]]", "[[论文笔记/PROEMO|PROEMO]]", "[[论文笔记/EmoVoice|EmoVoice]]", "[[论文笔记/OpenOmni|OpenOmni]]", "[[论文笔记/CSP-FT|CSP-FT (Wang et al., 2026)]]", "[[论文笔记/DialogueAgents|DialogueAgents]]", "[[论文笔记/FaceSpeak|FaceSpeak]]", "[[论文笔记/Multi-Step Hierarchical ED|Multi-Step Hierarchical ED (Inoue et al., 2025)]]", "[[论文笔记/MAE Style-Rich TTS|MAE Style-Rich TTS]]", "[[论文笔记/Prompt-Unseen-Emotion|PUE (Gao et al., 2025)]]", "[[论文笔记/DiEmo-TTS|DiEmo-TTS]]", "[[论文笔记/UDDETTS|UDDETTS]]", "[[论文笔记/MPE-TTS|MPE-TTS]]", "[[论文笔记/OpenS2S|OpenS2S]]", "[[论文笔记/TTS-CtrlNet|TTS-CtrlNet]]", "[[论文笔记/Spotlight-TTS|Spotlight-TTS]]", "[[论文笔记/CapSpeech|CapSpeech]]", "[[论文笔记/EME-TTS|EME-TTS]]", "[[论文笔记/NonverbalTTS|NonverbalTTS]]", "[[论文笔记/DeepDubbing|DeepDubbing]]", "[[论文笔记/LibriQuote|LibriQuote]]"]
origin_paper: "Xie et al., Controllable TTS in LLM Era, 2024"
related_concepts: ["[[Prosody Modeling]]", "[[Style Transfer in TTS]]", "[[Speech Factorization]]", "[[Differentiable Reward Optimization]]", "[[LLM-based TTS]]"]
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

### 3. 对抗训练解耦 (Adversarial Disentanglement)

将情感与说话人身份分离:
- Cross-speaker emotion transfer (Li et al., 2022): 对抗训练消除 speaker 信息
- 结果: 可将一个说话人的情感迁移到另一个说话人声音中

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

- [[Prosody Modeling]]: 情感通过韵律变化实现
- [[Style Transfer in TTS]]: 情感是风格的子维度
- [[Speech Factorization]]: 情感与 speaker/content 解耦
- [[Differentiable Reward Optimization]]: Emo-DPO 的技术基础
- [[LLM-based TTS]]: 新范式下的情感控制方式

## 帧级 Arousal-Valence 条件控制 (EmoCtrl-TTS)

EmoCtrl-TTS (Wu et al., 2024) 在 flow-matching zero-shot TTS 上同时使用两组帧级条件: (1) arousal-valence 值 (来自 wav2vec 2.0-based extractor, chunk-wise 0.5s/0.25s) 控制时变情感; (2) laughter detector embedding (32 维) 控制 NV (笑声、哭泣等)。用 27k 小时伪标签真实情感数据训练,在 JVNV S2ST 上 Aro-Val SIM 0.643 (超越 ELaTE 0.548)。关键发现: laughter detector embedding 能泛化到哭泣等非笑声 NV; 两种 embedding 在某些数据上存在负面交互,需按数据源选择性启用。与 NVSpeech 的离散 PV 标签方法互补: EmoCtrl-TTS 用连续 embedding 实现帧级控制,NVSpeech 用离散标签实现 token-level 控制。详见 [[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]。

## 副语言发声方法 (NVSpeech)

NVSpeech (Liao et al., 2025) 从不同角度切入情感表达 — 不直接建模抽象情感状态,而是建模具体的副语言行为 (笑声、叹气、犹豫等),这些行为是情感的外在表现。通过在文本中显式插入 `[Laughter]`、`[Breathing]` 等标签实现 token-level 控制。与传统情感控制互补: 情感控制提供高层意图,PV 控制提供底层行为实现。详见 [[论文笔记/NVSpeech|NVSpeech]]。

## 演进

规则情感合成 (HMM, 2003) → Emotion embedding (2021) → 多尺度层级建模 (MsEmoTTS, 2022) → 跨说话人情感迁移 (2022) → 韵律嵌入分解 (Daisy-TTS, 2024) → DPO/RLHF 对齐 (Emo-DPO, 2024) → 零样本情感 (EmoSphere++, 2024) → LLM 自由文本情感 (EmoVoice, 2025) → 副语言行为控制 (NVSpeech, 2025) → LLM prompt 混合情感 (PUE, 2025) → ADV 维度解耦控制 (UDDETTS, 2025)

## 多步层级情感分布预测 (Multi-Step Hierarchical ED)

[[论文笔记/Multi-Step Hierarchical ED|Multi-Step Hierarchical ED]] (Inoue et al., 2025) 提出多步预测框架,将情感量化为 utterance/word/phoneme 三级连续分布向量 (Hierarchical ED),并按 utterance→word→phoneme 顺序逐级预测,使高层情感上下文引导底层韵律。ED 通过 OpenSMILE + SVM 排序函数从音频中提取,支持训练时自动标注和推理时用户手动调整。在 FastSpeech 2 上验证了两种集成方式 (External 模型无关 / VA 内嵌),Multi-Step 在 WER (2.45% vs 4.61%) 和 MUSHRA 自然度上均优于 Single-Step baseline。BWS 测试中情感可控性全面优于 MsEmoTTS。关键发现: ED 数值差异相近但合成质量差异显著,说明多步预测学到的是层级依赖关系而非更准确的数值。

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
