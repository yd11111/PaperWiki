---
type: concept
title: "Style Transfer in TTS"
aliases: [语音风格迁移, Voice Style Transfer, Speaking Style Control, 风格控制, Zero-shot Style TTS]
category: "technique"
tags: [TTS, style, transfer, zero-shot, GST, reference-encoder, disentanglement]
key_papers: ["GST-Tacotron (Wang et al., 2018)", "MetaStyleSpeech (Min et al., 2021)", "GenerSpeech (Huang et al., 2022b)", "StyleTTS 2 (Li et al., 2023)", "StyleTTS-ZS (Li et al., 2024)", "MegaTTS 2 (Jiang et al., 2024)", "DEX-TTS (Park et al., 2024a)", "ControlSpeech (Ji et al., 2024c)", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/TextrolSpeech|TextrolSpeech]]", "[[论文笔记/IDEA-TTS|IDEA-TTS]]", "[[论文笔记/Controlling Emotion TTS NL Prompts|Bott et al. (Interspeech 2024)]]", "[[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]", "[[论文笔记/UMETTS|UMETTS]]", "[[论文笔记/StoryTTS|StoryTTS]]", "[[论文笔记/Daisy-TTS|Daisy-TTS]]", "[[论文笔记/EmoSphere++|EmoSphere++]]", "[[论文笔记/AutoStyle-TTS|AutoStyle-TTS]]", "[[论文笔记/RADKA-CSS|RADKA-CSS]]", "[[论文笔记/FlexSpeech|FlexSpeech]]", "[[论文笔记/FaceSpeak|FaceSpeak]]", "[[论文笔记/MAE Style-Rich TTS|MAE Style-Rich TTS]]", "[[论文笔记/DiEmo-TTS|DiEmo-TTS]]", "[[论文笔记/GSA-TTS|GSA-TTS]]", "[[论文笔记/TTS-CtrlNet|TTS-CtrlNet]]", "[[论文笔记/Spotlight-TTS|Spotlight-TTS]]", "[[论文笔记/CapSpeech|CapSpeech]]", "[[论文笔记/DS-TTS|DS-TTS]]", "[[论文笔记/ParaStyleTTS|ParaStyleTTS]]", "[[论文笔记/HiStyle|HiStyle]]", "[[论文笔记/UltraVoice|UltraVoice]]"]
origin_paper: "Xie et al., Controllable TTS in LLM Era, 2024"
related_concepts: ["[[Prosody Modeling]]", "[[Speaker Embedding]]", "[[Speech Factorization]]", "[[Global Style Tokens]]", "[[LLM-based TTS]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Style Transfer in TTS 旨在控制语音中的高层属性,包括语调 (tone)、正式程度 (formality)、话语模式 (discourse mode, 如新闻播报、日常对话、演讲),使 TTS 系统能适应不同的交流语境、受众和目标。

**与相关概念的区分**:
- **Prosody Control**: 底层声学特征 (pitch, duration, energy)
- **Emotion Control**: 情感状态 (happy, sad, angry)
- **Style Control**: 更高层的说话方式和表达风格

## 控制策略分类 (Survey 框架)

Survey 将风格控制策略分为四类:

### 1. Style Tagging (风格标记)
通过离散标签或连续值信号控制:
- **离散标签**: StyleTagging-TTS 用短语/词汇表示风格 (angry, happy)
- **连续信号**: DiffStyleTTS 层级化建模 pitch/energy/duration/style
- **隐空间修改**: Cauliflow 通过 flow-based model 调整 speech rate

### 2. Reference Speech Prompt (参考语音)
从参考音频中提取风格信息:
- MetaStyleSpeech: 自适应归一化 (adaptive normalization) 实现零样本风格
- GenerSpeech: 多级风格适配器 (multilevel style adapter) 泛化到域外
- SC VALL-E: style tokens + scale factors 控制情感、风格
- DEX-TTS: 分离时间不变 (time-invariant) 和时间变化 (time-variant) 风格
- MegaTTS 2: acoustic autoencoder 分离 prosody 和 timbre
- StyleTTS-ZS: 蒸馏时变风格扩散模型

### 3. Natural Language Descriptions (自然语言描述)
用文本描述目标风格 (见 [[Natural Language Description for TTS]])

### 4. Instruction-Guided (指令引导)
通过自由格式指令控制 (见 [[Instruction-Guided Speech Synthesis]])

## 关键技术

### Global Style Tokens (GST)
Wang et al. (2018) 提出的无监督风格表示:
- Style token bank: 一组可学习的 style embeddings
- Reference encoder: 从音频提取 query → attention over token bank
- 无需风格标签,自动发现风格聚类
- 支持风格插值和组合

### 风格解耦 (Style Disentanglement)
将风格与说话人/内容分离:
- **对抗训练**: gradient reversal 消除 speaker 信息 (An et al., 2022)
- **Information bottleneck**: 独立 encoder 分支编码不同属性
- **KL 正则化**: 约束隐空间结构

### 零样本风格迁移 (Zero-shot)
不需要目标风格的训练数据:
- MetaStyleSpeech: meta-learning 泛化到未见风格
- GenerSpeech: 多级适配器处理域外风格
- StyleTTS 2: 大规模语音语言模型 + 对抗训练
- ControlSpeech: 双向注意力 + 并行解码

## 在 TTS 中的应用

| 应用场景 | 风格维度 | 代表系统 |
|----------|----------|----------|
| 有声书朗读 | 叙述/对话/旁白 | MegaTTS 2 |
| 新闻播报 | 正式、权威 | VoxInstruct |
| 对话系统 | 自然、随意 | DailyTalk |
| 影视配音 | 角色扮演 | CosyVoice |
| 跨语言 | 保持源风格 | NansyTTS, XTTS |

## 研究趋势

Survey 总结的风格控制演进:
1. **Style tagging** (2018-2021): GST, 显式标签 → 有限表达多样性
2. **Reference speech prompt** (2021-2023): 零样本声音克隆, timbre 与 style 分离
3. **NL descriptions** (2023-2024): PromptTTS, 文本描述 → 用户友好
4. **Instruction-guided** (2024-): VoxInstruct, CosyVoice → 自由格式控制

## 在 SVS 中的风格迁移

Singing Style Transfer 是 Style Transfer in TTS 在歌声领域的延伸,但具有独特挑战 [Pan et al., 2026, §2]:

**Singing Voice Conversion (SVC)**: 将参考歌声的音色/风格迁移至新歌声。与 TTS voice cloning 不同,SVC 还需保持旋律和节奏的准确性。TCSinger (Zhang et al., 2024c) 用 RVQ 替换为 CVQ (Clustering VQ) 实现更稳定的风格压缩,并通过 LM 联合建模 prosody 和 style。

**Speech-to-Singing (STS)**: 将语音转换为歌声 (Li et al., 2023),需同时建模 musicality 和 style fidelity。

**Audio-based Transfer**: 参考音频分离为多个因素 (content, rhythm, timbre) 后零样本融合迁移 (Li et al., 2023; Dai et al., 2025)。

**Text-based Control for Singing**: PromptSinger (Wang et al., 2024a) 用自然语言 prompt 控制歌声音色、情感和响度; TechSinger (Guo et al., 2025b) 通过 [[Classifier-Free Guidance]] 控制歌唱技巧。

**MoE 风格控制**: TCSinger2 (Zhang et al., 2025b) 引入 MoE 路由策略选择专用生成专家,实现高质量风格控制。

**风格特定生成**: FreeStyler (Ning et al., 2025b) 用于说唱, SongSong (Hu et al., 2025) 用于古典艺术歌曲。

详见 [[Singing Voice Synthesis]]。

## 关键论文

- GST-Tacotron (Wang et al., ICML 2018): Global Style Tokens 开创无监督风格控制
- MetaStyleSpeech (Min et al., ICML 2021): meta-learning 零样本风格适应
- GenerSpeech (Huang et al., NeurIPS 2022): 多级风格适配器
- StyleTTS 2 (Li et al., NeurIPS 2023): 风格扩散 + 对抗训练达人类水平
- MegaTTS 2 (Jiang et al., ICLR 2024): prosody/timbre 解耦零样本

## 相关概念

- [[Prosody Modeling]]: 风格的底层声学实现
- [[Speaker Embedding]]: 与风格共同决定说话方式, 需解耦
- [[Speech Factorization]]: 风格迁移的前提
- [[Global Style Tokens]]: 风格迁移的奠基机制
- [[LLM-based TTS]]: 新范式下的风格控制方式

## 演进

固定风格合成 (SPSS) → GST 无监督风格发现 (2018) → VAE 风格隐空间 (2019) → Meta-learning 零样本 (2021) → 扩散+对抗达人类水平 (StyleTTS 2, 2023) → LLM in-context style (2023-) → 指令驱动自由风格 (VoxInstruct, 2024)
