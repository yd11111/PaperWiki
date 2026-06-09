# TTS 训练与评估

<!-- scope: 本页覆盖 TTS 训练数据工程、对齐与时长提取、域适配微调、情感控制、TTS 架构与方法、理解与多模态等。
     后训练优化(DPO/GRPO/RL/蒸馏)详见子页 [[TTS训练与评估-后训练优化]]。
     评估方法(MOS/自动指标/benchmark/安全)详见子页 [[TTS训练与评估-评估与基准]]。
     不包含: 生成模型架构本身(归零样本语音合成)、codec 设计(归语音编码与量化)、纯语音理解(归语音大模型与对话)。 -->

> [!note] 人工备注
> (此区域自动刷新时保留)

## 核心概念
- [[DifferentiableRewardOptimization]] — Token-level 可微 reward 优化, TTS post-training
- [[TTSEvaluation]] — TTS 评估方法论 (MOS, A/B, 自动指标)
- [[EmotionControlinTTS]] — TTS 情感控制
- [[AudioUnderstanding]] — 音频理解, 评估相关

## 代表模型

## 子页导航

### 后训练优化 (34 篇)
→ [[TTS训练与评估-后训练优化]]

代表论文:
- [[论文笔记/DiffRO|DiffRO]] — 2025, Token2Reward+Gumbel-Softmax+MTR, TTS 后训练优化原始论文
- [[论文笔记/F5R-TTS|F5R-TTS]] — 2025, GRPO 集成 NAR flow-matching TTS (作者 claim 首次)
- [[论文笔记/GRPO-TTS|GRPO-TTS]] — 2025, Whisper CER + NLL 复合 reward + GRPO 微调, 轻量级方案

### 评估与基准 (27 篇)
→ [[TTS训练与评估-评估与基准]]

代表论文:
- [[论文笔记/TTSDS|TTSDS]] — 2024, 分布级多因子评估 (作者 claim 首个), 5因子, ρ=0.60-0.83
- [[论文笔记/SpeechJudge|SpeechJudge]] — 2025, 99K pairwise + GRM, TTS naturalness 评估套件
- [[论文笔记/TTS-PRISM|TTS-PRISM]] — 2026, 12 维分层诊断框架, schema-driven instruction tuning

## 相关论文 (仅 deep/repro)

### Alignment & Front-end
- [[论文笔记/MultiTaskFrontEnd|Multi-Task Front-End]] — 2024, shared trunk + task-specific heads MTL 联合学习 TN/POS/HD 三个 TTS 前端任务
- [[论文笔记/BFA|BFA]] — 2025, Bournemouth Univ, CTC + universal phoneme encoder 强制对齐, 比 MFA 快 240x, onset+offset 双边界 + inter-phoneme gap 建模, TIMIT recall@60ms 87.9%
- [[论文笔记/CTC-TTS|CTC-TTS]] — 2026, CTC aligner 替代 MFA 做 phoneme-speech 对齐, bi-word 交错策略构建 LLM-based 双流式 TTS

### Domain Adaptation & Fine-tuning
- [[论文笔记/CSP-FT|CSP-FT]] — 2026, 基于 weighted-sum 层贡献分析的选择性微调, ~8% 参数 + 2x 加速, 缓解灾难性遗忘 (Fun-CosyVoice3.0 WER 3.8% vs Full FT 12.1%)
- [[论文笔记/Habibi|Habibi]] — 2026, 首个开源统一多方言阿拉伯语 TTS 框架, ASR→TTS 数据整理 + MSA→方言课程学习
- [[论文笔记/ZeSTA|ZeSTA]] — 2026, ZeSTA 框架, domain embedding + real-data oversampling 实现零样本 TTS 数据增强
- [[论文笔记/Cognitive-State-ConditionedTTS|CoSTA]] — 2026, 认知状态条件化 CosyVoice2/F5-TTS 合成 AD/HC 语音做数据增强, ADReSS 检测准确率 85.83%

### Emotion Control
- [[论文笔记/EmoSphere-TTS|EmoSphere-TTS]] — 2024, Interspeech, AVD 伪标签 + 球面坐标解耦风格/强度, nMOS 3.88, ECA 94.02%
- [[论文笔记/ControllingEmotionTTSNLPrompts|Controlling Emotion TTS NL Prompts]] — 2024, Interspeech, 情感文本作 NL prompt + SE block 融合 + curriculum learning, Cramer's V 0.80, MOS 3.37

### Data Engineering
- [[论文笔记/LowResourceSSL-TTS|Low-Resource SSL-Enhanced TTS]] — 2023, 100h HuBERT 构建 unit-based TTS 合成数据反哺 SSL 预训练, 语音数据需求降低 90%
- [[论文笔记/Emilia|Emilia]] — 2024, CUHK-SZ, 大规模多语言 in-the-wild 开源语音数据集 (作者 claim 首个)(101K+ h, 6 语种) + Emilia-Pipe 预处理 pipeline
- [[论文笔记/NaturalVoices|NaturalVoices]] — 2024, 大规模自发情感语音数据集(3846h, 2467 speakers)
- [[论文笔记/TITW|TITW]] — 2024, 标准化 noisy-TTS 训练数据集 (作者 claim 首批)(VoxCeleb1→TITW-Easy 173h/Hard 189h), DNSMOS 过滤 pipeline + KSKT/KSUT 评估协议
- [[论文笔记/StoryTTS|StoryTTS]] — 2024, 61h 中文评书表现力数据集, LLM 五维度文本表现力标注, MOS 4.09
- [[论文笔记/AutomatedTTSDataset|Automated TTS Dataset Generation]] — 2024, 首个开源端到端 TTS 数据集生成工具, phoneme-balanced 选句 + ASR-based QA, 覆盖 6 种语言
- [[论文笔记/SpeechWeave|SpeechWeave]] — 2025, ACL Industry, 合成 TTS 训练数据管线(keyphrase diversity + at-source normalization + 跨语言说话人标准化), 多样性↑10-48%, 规范化准确率 97%
- [[论文笔记/LibriQuote|LibriQuote]] — 2025 (ACL 2026 Findings), 叙事感知有声书表现力数据集(5.3K h 台词 + 12.7K h 叙述 + speech verb/adverb 伪标签), LALM-based 表现力评估 (ContextMOS/Win-Rate), F5-TTS 微调显著优于 SparkTTS 微调
- [[论文笔记/UltraVoice|UltraVoice]] — 2025, 大规模多维度细粒度风格控制语音对话数据集 (作者 claim 首个)(830h, 100K 样本), SFT 后 MOS +29-42%
- [[论文笔记/ParsVoice|ParsVoice]] — 2026, U Tehran, 公开波斯语 TTS 语料库 (作者 claim 最大) (2,200h, 1,815 speakers), ParsBERT 句子完整性验证 + 二分搜索边界优化 + 多维质量评估 pipeline, XTTS 微调 MOS 3.6/SMOS 4.0

### Understanding & Multimodal
- [[论文笔记/DiVA|DiVA]] — 2024, 跨模态 context distillation (冻结 text LLM 对 transcript 的响应分布作为自监督目标) 训练 Speech LLM
- [[论文笔记/Qwen2-Audio|Qwen2-Audio]] — 2024, 自然语言 prompt 替代层次标签简化预训练, SFT+DPO 三阶段训练实现 voice chat / audio analysis
- [[论文笔记/Audio-Thinker|Audio-Thinker]] — 2025, 首个系统性的音频-语言 RL 推理框架, 四层渐进式 reward 设计
- [[论文笔记/Qwen2.5-Omni|Qwen2.5-Omni]] — 2025, 统一端到端多模态模型, 同时感知 text/image/audio/video 并流式生成 text + 自然语音
- [[论文笔记/EmotionThinker|EmotionThinker]] — 2026, RL-based 可解释语音情感推理, GRPO-PTR, SER Avg 68.89%
- [[论文笔记/SpeechWorldModel|SpeechWorldModel]] — 2026, 因果图模块化语音理解(4模块DAG), EM 97.80% 超越 Gemini 2.5 Pro
- [[论文笔记/Ming-Flash-Omni|Ming-Flash-Omni]] — 2026, 基于 Ling-Flash-2.0 MoE (100B/6.1B active) 的全模态感知-生成模型
- [[论文笔记/ERNIE5.0|ERNIE 5.0]] — 2026, 首个公开的万亿参数统一自回归基础模型, 超稀疏 MoE + 模态无关路由 + 弹性训练
- [[论文笔记/Qwen3-ASR|Qwen3-ASR]] — 2026, 基于 Qwen3-Omni 基座的 all-in-one ASR 家族 (0.6B/1.7B) + 首个 LALM-based NAR forced aligner
- [[论文笔记/video-SALMONN-S|video-SALMONN S]] — 2026, 首个将 test-time training (TTT) 用作流式视频记忆机制的 av-LLM

### TTS Architecture & Methods
- [[论文笔记/LipToSpeech|Lip-to-Speech]] — 2023, lip-to-speech 分解为 lip-to-text + visual-conditioned TTS, 利用预训练 lip reading 模型
- [[论文笔记/UnitSpeech|UnitSpeech]] — 2023, HuBERT 离散单元替代文本转写, diffusion-based speaker adaptation, 单条无转写音频即可微调
- [[论文笔记/FrameWiseBreath|Frame-Wise Breath Detection]] — 2024, Conformer + 自训练帧级呼吸音检测, 无需人工标注, 提升 TTS 呼吸自然度
- [[论文笔记/TextToSpeechSynthesis|Text to Speech Synthesis]] — 2024, TTS 文献综述, 从传统到 neural TTS 的基本概念和代表模型
- [[论文笔记/DSM|DSM]] — 2025, Delayed Streams Modeling, 多模态序列预对齐到相同帧率并引入可控 delay
- [[论文笔记/Metis|Metis]] — 2025, SSL tokens 上无条件 masked generative pre-training (300K h), fine-tuning 适配 5 种语音生成任务
- [[论文笔记/SplitMeanFlow|SplitMeanFlow]] — 2025, Interval Splitting Consistency 恒等式训练平均速度场, MeanFlow 的微分恒等式是其极限特例
- [[论文笔记/DSFlow|DSFlow]] — 2026, dual supervision (endpoint + velocity alignment) + step-aware token 替代 adaLN-Zero, one-step flow matching
- [[论文笔记/RobustSpeechFlow|RobustSpeechFlow]] — 2026, TTS skip/repeat 失败模式转化为 contrastive flow matching 的 hard negatives
- [[论文笔记/SonoEdit|SonoEdit]] — 2026, NLP 知识编辑 (ROME/AlphaEdit) 迁移到 TTS, acoustic causal tracing 定位发音层 + null-space 约束
- [[论文笔记/UniAudio2.0|UniAudio 2.0]] — 2026, ReasoningCodec (reasoning tokens 5Hz + reconstruction tokens 12.5Hz) + functional 统一音频语言模型
- [[论文笔记/UniVocal|UniVocal]] — 2026, CosyVoice 2 基础上 refined cent token (1200-bin 音高离散化) + CoT interleaved generation 实现语音-歌声 code-switching

### Applications & Systems
- [[论文笔记/TableQuest|TableQuest]] — 2024, 真实金融报告语境下评估 LLM 表格理解能力的 benchmark
- [[论文笔记/DuIVRS-2|DuIVRS-2]] — 2026, FSM 约束 + CoT 选择式生成 + 双评估器迭代学习, <2B LLM 部署到百度地图电话 IVR 系统, 83.9% 任务成功率
- [[论文笔记/OmniScript|OmniScript]] — 2026, 首个音频-视觉长视频到结构化脚本 (V2S) 的 8B 多模态模型, 记忆增强标注 + CoT SFT + 时序分段 RL
- [[论文笔记/High-FidelityCodec-Inspired|CodecCap]] — 2026, 视频编解码 I-frame/P-frame 思想迁移到 dense video captioning

## 相关任务
- [[Zero-shotSpeechSynthesis]]

## 相关数据集
- [[SEED-TTS-Eval]] — ByteDance, 零样本 TTS 标准 benchmark
- [[Emilia]] — 101K+ hrs 多语言训练数据

## 演进脉络

```
Understanding:
  → EmotionThinker (2026, RL-based reasoning)
  → SpeechWorldModel (2026, causal graph)
  → NaturalVoices (2024, spontaneous emotion data)

Data Engineering:
  Emilia (2024, 101K h in-the-wild)
  → SpeechWeave (2025, 合成数据管线)
  → LibriQuote (2025, 叙事感知有声书)
  → UltraVoice (2025, 细粒度风格控制数据)
  → ParsVoice (2026, 波斯语 2,200h)

(Post-training 演进 → 详见 [[TTS训练与评估-后训练优化]])
(Evaluation 演进 → 详见 [[TTS训练与评估-评估与基准]])
```

## 历史参考

（暂无已废弃实体）
