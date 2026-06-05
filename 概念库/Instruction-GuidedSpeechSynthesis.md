---
type: concept
title: "Instruction-Guided Speech Synthesis"
aliases: [指令引导语音合成, Instruction-following TTS, Instruction-to-Speech, 指令TTS, Speech Instruction Following]
category: "technique"
tags: [TTS, instruction, controllability, LLM, multimodal, editing, generation]
key_papers: ["VoxInstruct (Zhou et al., 2024)", "CosyVoice (Du et al., 2024)", "AudioGPT (Huang et al., 2024b)", "SpeechGPT (Zhang et al., 2023b)", "FunAudioLLM (An et al., 2024)", "VoiceCraft (Peng et al., 2024b)", "InstructSpeech (Huang et al., 2024a)", "[[论文笔记/Step-Audio|Step-Audio]]", "[[论文笔记/Step-Audio2.5|StepAudio 2.5]]", "[[论文笔记/FlexiVoice|FlexiVoice]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/EmoVoice|EmoVoice]]", "[[论文笔记/InstructTTSEval|InstructTTSEval]]", "[[论文笔记/DeepDubbing|DeepDubbing]]", "[[论文笔记/HD-PPT|HD-PPT]]", "[[论文笔记/UltraVoice|UltraVoice]]", "[[论文笔记/BatonVoice|BatonVoice]]", "[[论文笔记/Audiobook-CC|Audiobook-CC]]"]
origin_paper: "Xie et al., Controllable TTS in LLM Era, 2024"
related_concepts: ["[[NaturalLanguageDescriptionforTTS]]", "[[LLM-basedTTS]]", "[[StyleTransferinTTS]]", "[[EmotionControlinTTS]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Instruction-Guided Speech Synthesis 将 TTS 重构为指令跟随任务: 用户提供一条自然语言指令,同时传达要说的内容 (content) 和如何说 (style/emotion/timbre),系统据此生成语音。这区别于传统的 content + description 分离范式。

**核心区别** (Survey Sec 3.2.4):
> "Description-based TTS methods separate inputs into content and description prompts, diverging from the unified instruction formats used in chatbots. Instruction-guided methods reframe TTS as a general instruction-to-speech task."

**示例**:
- NL Description: content="Hello" + description="young female, cheerful, fast"
- Instruction-Guided: "Say 'Hello' in a cheerful young female voice at a fast pace"

## 代表系统

### VoxInstruct (Zhou et al., 2024)
**定义性工作**: 将 TTS 重构为通用 instruction-to-speech 任务
- 单条自然语言 prompt 同时传达内容和风格描述
- 训练数据: 构造 <instruction, speech> 对
- 支持多属性同时控制

### CosyVoice (Du et al., 2024)
Hybrid LLM + Flow Matching 架构:
- LLM 驱动 semantic token 生成
- Flow matching 实现高保真合成
- 通过自然语言指令控制: speaker identity, emotion, pitch, speed, paralinguistic cues
- 使用 ASR 衍生的 supervised semantic tokens

### AudioGPT (Huang et al., 2024b)
多模态 LLM agent:
- 整合多个模块: 语音理解、合成、风格转换
- 统一指令接口
- 支持语音编辑和生成

### SpeechGPT (Zhang et al., 2023b)
端到端语音-文本 LLM:
- 将语音 token 整合到 LLM 词表
- 通过对话指令控制语音生成

### InstructSpeech (Huang et al., 2024a)
基于 multi-task LLM 的语音编辑:
- <instruction, input, output> triplets + task embeddings
- 分层适配器 (hierarchical adapters)
- 支持自由格式语音编辑: 插入、删除、替换
- 多步推理 (multi-step reasoning) 实现复杂编辑

### Step-Audio (Huang et al., 2025)
Instruction-driven TTS module:
- 动态控制方言 (dialects)、情感 (emotions)、唱歌 (singing)、说唱 (rapping)
- 多种说话风格

### VoiceCraft (Peng et al., 2024b)
指令引导的语音编辑:
- Decoder-only transformer + causal masking + delayed stacking
- 双向上下文感知的 instruction-guided editing
- 支持: insertion, deletion, substitution
- 保持高自然度

## 与 NL Description 的区别

| 维度 | NL Description | Instruction-Guided |
|------|----------------|-------------------|
| 输入格式 | content + description 分离 | 统一指令 |
| 灵活度 | 描述属性 | 任意复杂指令 |
| 交互范式 | 配置式 | 对话式 (chatbot-like) |
| 编辑能力 | 仅生成 | 生成 + 编辑 |
| 多步推理 | 不支持 | InstructSpeech 支持 |
| 代表系统 | PromptTTS, InstructTTS | VoxInstruct, CosyVoice |

## 技术架构

Survey 总结指令引导系统的共性:
1. **Instruction understanding**: LLM 理解指令中的控制意图
2. **Attribute extraction**: 从指令中提取风格/情感/说话人特征
3. **Conditional generation**: 以提取的属性为条件生成语音
4. **Multi-modal alignment**: 文本指令与语音特征的对齐

## 挑战与未来方向

Survey (Sec 5.2) 指出的关键挑战:
1. **精确控制**: 现有系统常产生偏离用户意图的语音
2. **多属性一致性**: 同时控制多个属性时可能冲突
3. **Grounding**: 将抽象指令落地为具体声学操作
4. **评估**: 传统指标难以衡量指令遵循度 (需 GPT-based evaluation)

## 在 TTS 中的应用

- 对话式语音助手: 用户直接描述想要的声音
- 内容创作: "用悲伤的低沉男声念这段旁白"
- 语音编辑: "把第二句的语速放慢，加一点停顿"
- 多语言: "用带法国口音的英语说这段话"

## 关键论文

- VoxInstruct (Zhou et al., 2024): instruction-to-speech 统一框架
- CosyVoice (Du et al., 2024): LLM + Flow matching 指令控制
- InstructSpeech (Huang et al., ICML 2024): multi-task LLM 语音编辑
- AudioGPT (Huang et al., AAAI 2024): 多模态 LLM agent
- Step-Audio (Huang et al., 2025): 方言/情感/唱歌的指令动态控制

## 相关概念

- [[NaturalLanguageDescriptionforTTS]]: 指令引导的前身, 内容与描述分离
- [[LLM-basedTTS]]: 指令引导的技术基础
- [[StyleTransferinTTS]]: 指令可实现风格迁移
- [[EmotionControlinTTS]]: 指令中可包含情感控制

## 统一 TTS+TTM 指令控制 (InstructAudio)

[[论文笔记/InstructAudio|InstructAudio]] (Qiang et al., Kuaishou/Tianjin Univ., 2025) 首次将 instruction-guided 范式从纯 TTS 扩展到统一 TTS+TTM (Text-to-Music) 框架。核心设计: 标准化 instruction-phoneme 输入格式,NL instruction 描述所有属性 (timbre/paralinguistic/musical),text/lyrics 统一转为 phoneme,用 MM-DiT (Joint DiT 14L + Single DiT 6L) 基于 CFM 训练同时生成语音和音乐。1.34B 参数,50K h 语音 + 20K h 音乐训练。Seed-TTS WER EN 1.52% / ZH 1.35% (best); 唯一同时支持 Gender/Age/Emotion/Style/Accent/Dialogue 纯文本控制; SongEval 全维度超越 ACE-Step/DiffRhythm+。局限: NMOS 3.46 低于 CosyVoice2 (3.65),纯文本控制的 one-to-many 模糊性导致音质下降。与 VoxInstruct 的区别: VoxInstruct 统一了 TTS 内的 content+style,InstructAudio 进一步统一了 TTS 与 TTM 两个任务。

## 统一 TTS+TTM+TTA 三模态指令控制 (UniSonate)

[[论文笔记/UniSonate|UniSonate]] (Qiang et al., Kuaishou/Tianjin Univ., 2026) 将 InstructAudio 的统一指令范式进一步扩展到 TTS+TTM+TTA (Sound Effects) 三模态。核心挑战: SFX 没有语言学内容,无法用 phoneme 序列作为 content input。解决方案: **Dynamic Token Injection** — 引入可学习 [SFX] token 作为伪音素,token 数量按 phoneme-to-duration 比率确定,使 phoneme 驱动架构无需修改即可处理非语言音频。配合三阶段课程学习 (speech→music→SFX) 缓解负迁移。1.34B 参数 (同 InstructAudio),新增 1.5M SFX clips。TTS WER EN 1.47% (best), SongEval Coh 3.18 (SOTA), TTA FAD 4.21 (competitive)。消融显示正向迁移: 联合训练 WER 2.24→1.47 [Table 6]。

## 演进

Style tagging (离散标签, 2018) → Reference prompt (参考音频, 2021) → NL description (文本描述, 2023) → **Instruction-guided** (统一指令, VoxInstruct, 2024) → Multi-step editing (InstructSpeech, 2024) → Omni-modal agent (Step-Audio, 2025) → 统一 TTS+TTM 指令控制 (InstructAudio, 2025) → **统一 TTS+TTM+TTA 三模态** (UniSonate, 2026) → **Open-Vocabulary InstructTTS** (叙事上下文指令 + reasoning chain, OV-InstructTTS, 2026)

## 评估 Benchmark

### MINT-Bench (Chen et al., 2026)

[[论文笔记/MINT-Bench|MINT-Bench]] 是首个结构化多语言 instruction-following TTS benchmark。基于分层多轴 taxonomy (10 原子属性 x Easy/Hard/Special 难度) 和三层评估协议 (内容一致性 → 指令遵循 → 条件感知质量),在 10 语言上系统评估指令遵循能力。关键发现: Easy 控制已相对可靠,但 compositional (Dynamic/Layered/Conflict) 和 extra-vocal (Disfluency/Dysphonia) 控制仍是瓶颈; 中文场景下 Qwen3-TTS 已超越所有商用系统 [Table 3]。与 InstructTTSEval 的区别: InstructTTSEval 用 True/False 二分判断,MINT-Bench 提供三级评分 + 条件感知质量奖励的更细粒度诊断。

## Open-Vocabulary 指令范式 (OV-InstructTTS)

[[论文笔记/OV-InstructTTS|OV-InstructTTS]] (Ren et al., CASIA/Tsinghua, 2026) 提出将 InstructTTS 从"声学属性的组合/复述"推向"源自叙事上下文的开放词汇指令"。核心方法: 构造 OV-Speech 数据集 (从有声书小说上下文提取结构化元素合成导演式指令,316K utterances) + reasoning-driven 框架 OV-InstructTTS-TEP (基于 Step-Audio-2-mini-Base LALM,先通过 `<think>` 推理链推断情感/声学/副语言属性,再生成 interleaved text+audio tokens)。消融显示 reasoning + enriched transcription 协同效应显著 (Gemini Score 71.57 vs 单独 reasoning 68.71 或单独 EP 66.98)。MOS 4.28 超越 GroundTruth (4.10), ICMOS 3.91 [Table 2]。与 VoxInstruct 的区别: VoxInstruct 统一了 content+style,OV-InstructTTS 进一步将指令空间从预定义属性开放化为叙事指令,并用 reasoning chain 桥接语义鸿沟。
