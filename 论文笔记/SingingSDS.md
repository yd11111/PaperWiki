---
type: paper
tier: deep
title: "SingingSDS: A Singing-Capable Spoken Dialogue System for Conversational Roleplay Applications"
arxiv_id: "2511.20972"
source: "Sources/SingingSDS.pdf"
authors: [Jionghao Han, Jiatong Shi, Masao Someki, Yuxun Tang, Lan Liu, Yiwen Zhao, Wenhao Feng, Shinji Watanabe]
year: 2025
venue: "EAIM2026 at AAAI"
tags: [spoken-dialogue-system, singing-voice-synthesis, SVS, roleplay, cascaded-pipeline, LLM, ASR, melody-control, interactive, VISinger, open-source]
concepts: ["[[SingingVoiceSynthesis]]", "[[SVSEvaluationMetrics]]", "[[MusicalScoreEncoder]]", "[[SpokenDialogueEvaluation]]"]
models: ["[[模型库/VITS|VITS]]", "[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[SingingVoiceSynthesis]](pending-review), [[SVSEvaluationMetrics]](pending-review), [[MusicalScoreEncoder]](pending-review), [[SpokenDialogueEvaluation]](pending-review) | 未命中但可能相关: 无

**谱系定位**: SingingSDS 处于 Spoken Dialogue Systems 与 [[SingingVoiceSynthesis]] 的交叉地带,将传统 SDS 的 speech-out 替换为 singing-out。在 SVS 谱系中,它不聚焦于合成质量或技巧控制的突破,而是关注将 SVS 嵌入交互式对话系统的**系统集成**问题。采用级联 ASR-LLM-SVS pipeline,SVS 后端为 VISinger 2 (基于 VITS 架构的端到端 SVS [Singing Voice Synthesis §端到端]),这是一种成熟但非前沿的 SVS 方案。

**已有认知**:
- [[LLM-basedTTS]] (confirmed): LLM 在 TTS 中主要用于 codec token 生成,但 SingingSDS 中 LLM 的角色不同 -- 它仅负责生成歌词文本响应,不参与语音/歌声 token 生成,更接近传统 NLG 用法。
- [[SpeakerEmbedding]] (confirmed): SingingSDS 的双语 VISinger 2 模型使用 192 维 learned speaker embedding 实现多歌手建模,中文模型则使用 Speaker ID 条件化 [Appendix C]。
- [[SingingVoiceSynthesis]] [待确认]: SingingSDS 使用的 VISinger 2 属于端到端 SVS 架构 (VITS→SVS),在已有分类中属于"高保真合成"基础目标,未涉及技巧控制或风格迁移。
- [[MusicalScoreEncoder]] [待确认]: SingingSDS 不使用传统的 Musical Score Encoder (无完整乐谱输入),而是通过 melody controller 提供 note-level 约束 (pitch + start/end time),比标准 SVS 的乐谱输入更松散。
- [[SVSEvaluationMetrics]] [待确认]: SingingSDS 使用 SingMOS (自动歌声质量预测) + PER (音素错误率,取代 SVS 中常用的 CER),并新增面向娱乐性的人工评估维度 (Novelty & Fun, Character Consistency, Lyric Quality),这些维度在现有 SVS 评估体系中不存在。
- [[SpokenDialogueEvaluation]] [待确认]: SingingSDS 的评估超越了传统 SDS 评估 (文本智能/语音质量/延迟),引入了娱乐价值和角色一致性维度,但评估规模较小 (20 prompts, 6 listeners)。

**创新判断**: SingingSDS 的创新不在单一技术组件,而在于**首次将 SVS 集成到交互式 SDS 中**的系统级贡献 -- melody-constrained lyric generation (LLM prompt 中嵌入音节约束) + melody alignment strategies (pitch-based vs lyric-aware) + 模块化架构 (350 种配置组合)。这是一个系统工程导向的工作,填补了 SDS 与 SVS 之间的空白。

## 速查

> [!summary] 速查
> - **一句话**: 首个将歌声合成 (SVS) 集成到交互式 spoken dialogue system 的开源系统,通过级联 ASR-LLM-SVS pipeline + melody-constrained lyric prompting 实现 speech-in, singing-out 角色扮演对话
> - **路线**: 用户语音 → ASR (Whisper/Paraformer) → 文本 → LLM (Llama3/Gemini + 角色 prompt + 音节约束) → 歌词文本 → G2P → SVS (VISinger 2 + melody controller + speaker embedding) → 歌声输出
> - **指标**: SingMOS 4.53-4.59, PER 0.12-0.61%, N&F 4.00-4.21, Char. Cons. 4.13-4.19, Lyric Qual. 3.35-3.86 (20 prompts, 4 配置组合, Whisper+Gemini+KiSing 最优) [Table 2]; SVS 延迟 ~0.16-0.19s [Table 2]
> - **可借鉴**: (1) melody phrase constraint prompting — 在 LLM prompt 中指定每行音节数以对齐旋律结构,无需模型微调; (2) pitch-based vs lyric-aware alignment 两种旋律对齐策略可推广到任何 melody-conditioned generation; (3) 模块化 registry-based 架构设计适用于多模态对话系统快速原型
> - **局限**: SVS 后端 (VISinger 2) 非最先进; 评估规模小 (20 prompts, 6 listeners); 仅支持中日双语; melody constraint 是 soft constraint (LLM 不保证精确音节数); 无与 end-to-end speech LLM 对话系统的延迟/质量对比; 仅支持 text → singing 的单向转换,无 singing → text 反向理解

## 核心问题

本文要解决的核心问题: **现有 Spoken Dialogue Systems 仅支持语音输出,而歌唱作为更具情感表达力、记忆性和娱乐性的交互模态,尚未被集成到交互式对话系统中** [§1]。同时,现有 SVS 系统本质上是非交互式的 — 基于预定义歌词和乐谱运行,缺乏对用户输入的动态响应机制 [§1]。将 SVS 嵌入 SDS 面临三个系统级挑战: (1) 如何让 LLM 生成的文本响应在语义合理的同时满足旋律结构约束 (音节数/短语结构); (2) 如何在保持可接受延迟的同时实现 singing-quality 输出; (3) 如何评估这种新型交互模态的质量 (传统指标无法衡量娱乐价值)。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SingingSDS 采用级联 ASR-LLM-SVS pipeline [§3, Fig 2]:

```
用户语音 s → [ASR(s, ℓ)] → 文本 st
                                ↓
                 [LLM(SystemPrompt(c, C), UserPrompt(st))] → 歌词响应 l
                                ↓
               [G2P] → 音素序列 lφ
                                ↓
    [MelodyControl(lφ, N)] + [Speaker v] → [SVS] → 歌声输出 Ŝ
```

**为什么选择级联而非端到端?** [论文原文] 作者最初考虑了 text-to-song 端到端方案,但测试发现 YuE 在 T4 GPU 上生成 5 秒音频需 40+ 秒延迟,不适合交互式使用 [Appendix A]。级联方案将 SVS 延迟控制在 0.16-0.19s [Table 2],但代价是需要预定义 melody 作为条件输入。

### 关键设计选择

#### 1. Melody Control 模块 [§3]

melody controller 提供 note-level 约束序列 N = (pi, τis, τie)_{i=1}^{n},其中 pi 为 MIDI pitch, τis/τie 为起止时间。支持两类 melody source:

- **Random melody** (baseline): 随机采样 pitch 和 duration,无短语结构,使用 forced alignment 将一个音节映射到一个 note。
- **Sampled melody**: 从歌曲数据集 (KiSing, Touhou MIDI, Yue 合成集) 检索。两种对齐策略:
  - **Pitch-based alignment**: one-to-one 映射,每个音节对应一个 note [§3, Fig 3 middle row]
  - **Lyric-aware alignment**: 保留原始歌曲中一个音节跨多个 note 的 melisma 结构 [§3, Fig 3 bottom row]

[agent 解读] lyric-aware alignment 保留原始旋律的 melisma 映射比 pitch-based 更贴近自然歌声的节奏感,但需要原始歌词-旋律对齐信息,这限制了可用的 melody 数据库范围。

#### 2. Phrase-Constrained LLM Prompting [§3, Appendix B.2]

当 melody 提供 phrase annotations 时,系统构建包含音节约束的 LLM prompt,例如:
```
请按照歌词格式回复,每句需遵循以下字数规则:
第1句: 5个字
第2句: 7个字
...
```

[论文原文] 这是一种 soft constraint — LLM 不保证严格遵循字数要求,但 prompt 引导输出趋向目标结构 [Appendix B.2]。中文因 character≈syllable 映射关系,字数 prompt 可近似控制音节数;日文需先转为 kana 再计数 [Appendix B.2]。

[agent 解读] 这种 prompt-based 音节控制方法轻量且无需模型修改,但其可靠性完全取决于 LLM 的指令遵循能力。论文未报告实际的音节匹配率统计,这是一个显著的评估缺失。

#### 3. SVS 后端: VISinger 2 [§4.1, Appendix C]

两个多歌手 VISinger 2 模型:
- 中文模型: ACE-Opencpop 数据集训练, Speaker ID 条件化, 40 歌手
- 中日双语模型: OpenCpop + KiSing + ACE-KiSing + M4Singer + Kiritan + Onikuru Kurumi + PJS + Namine Ritsu 混合训练, 192-dim learned speaker embedding + 3-way language ID

架构参数: hidden dim 192, text encoder 6 layers, posterior encoder 8 layers, 2 attention heads, 500 epochs, AdamW lr=2e-4 [Table 3, Table 4]。

#### 4. 角色系统 [§4.3, Appendix B.1]

两个原创虚构角色 (Limei, Yaoyin),通过 system prompt 定义背景、性格、说话风格、人物关系和特殊能力,遵循 OmniCharacter 的结构化 persona 格式 [§3]。角色 prompt 强制口语化表达、限制断句数 (≤4句)、禁止描写动作/表情。

### 模块化架构 [§4, Fig 4]

registry-based 组件设计: 5 ASR × 7 LLM × SVS × 5 melody = 350 种配置。Gradio web UI + YAML 配置模板 + CLI 接口。所有组件通过 central interface 连接。

### 训练策略

SingingSDS 本身无端到端训练 — ASR 和 LLM 使用预训练模型 (Whisper, Paraformer, Llama 3, Gemini 等),SVS 模型由作者使用 ESPnet GAN-SVS recipe 训练 [Appendix C.2]: 44.1kHz 采样率, MSE GAN adversarial loss, mel loss weight 45.0, pitch loss weight 10.0, duration loss weight 0.1, KL loss weight 1.0 [Table 4]。

## 实验

| 指标 | Whisper+Llama3+KiSing | Paraformer+Llama3+KiSing | Whisper+Gemini+KiSing | Whisper+Llama3+Touhou | 出处 |
| --- | --- | --- | --- | --- | --- |
| SingMOS ↑ | 4.53 | 4.47 | **4.59** | 4.52 | [Table 2] |
| PER (%) ↓ | 0.61 | **0.12** | 0.48 | 0.14 | [Table 2] |
| Large Jump Ratio | 0.11 | 0.13 | **0.09** | 0.28 | [Table 2] |
| N&F ↑ | 4.00 | 4.08 | **4.21** | 4.06 | [Table 2] |
| Char. Cons. ↑ | 4.17 | 4.13 | **4.19** | 4.13 | [Table 2] |
| Lyric Qual. ↑ | 3.35 | 3.41 | **3.86** | 3.70 | [Table 2] |
| ASR Lat. (s) ↓ | 0.80 | **0.44** | 0.55 | 0.82 | [Table 2] |
| LLM Lat. (s) ↓ | **1.87** | 1.79 | 5.79 | 2.22 | [Table 2] |
| SVS Lat. (s) ↓ | 0.19 | **0.16** | 0.18 | 0.19 | [Table 2] |

**KdConv 扩展评估** (450 utterances, Appendix F) [Table 5]:
- SVS-1 (random melody): SingMOS 4.53, PER 25%, Jump Ratio 35%
- SVS-2 (KiSing): SingMOS 4.27, PER 36%, Jump Ratio 4%
- SVS-3 (Touhou): SingMOS 4.43, PER 29%, Jump Ratio 12%

**关键发现**:
1. Whisper+Gemini 组合在所有感知质量指标上最优,但 LLM 延迟最高 (5.79s vs Llama3 ~1.8s) [Table 2]。Gemini 的歌词质量提升 (3.86 vs 3.35) 可能归因于其更强的指令遵循能力 [agent 解读]。
2. SVS 延迟始终极低 (~0.16-0.19s),系统瓶颈在 LLM 推理 [Table 2]。
3. Touhou 旋律的 Jump Ratio (0.28) 远高于 KiSing (0.09-0.13),反映其更丰富的音高动态 [Table 2]。
4. KdConv 实验中 random melody 的 SingMOS 最高 (4.53),说明对通用对话 utterance,随机旋律足以产生可接受的歌声输出 [Table 5, Appendix F.2]。
5. AudioBox Aesthetics 等自动指标与人类偏好不一致 — 在某些情况下偏好随机生成的不和谐音序列 [§2]。

## 局限性

1. **SVS 后端非前沿**: VISinger 2 是 2022 年模型,未与 DiffSinger、TCSinger 等更新 SVS 模型对比,无法判断 SVS 组件的质量天花板
2. **评估规模小**: 主观评估仅 20 prompts + 6 listeners,统计效力有限;KdConv 评估 (450 utterances) 未进行人工 MOS
3. **语言覆盖有限**: 仅中日双语,未验证对英语等其他语言的适用性
4. **Melody constraint 的可靠性未量化**: LLM 实际遵循音节约束的比例未报告,无法评估 prompt-based alignment 的真实有效性
5. **无与 end-to-end SDS 的对比**: 未与 Moshi、Kimi-Audio 等端到端对话系统比较延迟/用户体验
6. **单向模态转换**: 仅 speech→singing,不支持 singing→text 理解或 singing→singing 交互
7. **角色泛化能力未验证**: 仅测试 2 个虚构角色,新角色需要手工编写 prompt,无自动化角色构建

## 点评

SingingSDS 是一篇系统工程导向的 demo paper,其价值在于提出了一个新的交互范式 (speech-in, singing-out) 并提供了完整的开源实现。核心贡献是**问题定义**和**系统集成**,而非单一技术突破。

**值得肯定的**:
- 首次将 SVS 嵌入交互式 SDS,开辟了 singing-based dialogue 这一新的研究方向
- 模块化设计支持 350 种配置,便于 ablation 和研究
- 完整开源 (代码 + SVS 模型 + web demo),reproducibility 好
- Melody phrase constraint prompting 是一个巧妙的轻量方案,避免了模型修改
- 指出 AudioBox Aesthetics 等自动指标对歌声的不适用性,有参考价值

**需要警惕的**:
- 技术深度有限 — 所有组件 (ASR, LLM, SVS, melody) 都是现成的,系统的胶水代码价值有待商榷
- 评估设计不够严谨: 6 个 evaluator 的主观评估难以得出可靠结论; 缺少 inter-rater agreement 报告
- "首个" (first) 的声明基于 "to the best of our knowledge" [§2],由于缺乏系统的 prior work survey,难以完全确认
- 应用场景 (VR concerts, music games, theme parks) 的描述偏乐观,实际的 end-to-end 延迟 (~3-7s) 可能不满足实时交互需求

## 可复用的 idea

1. **Melody phrase constraint prompting**: 在 LLM prompt 中嵌入结构约束 (每行 N 个字/音节) 来间接控制生成输出的韵律结构,这种 prompt engineering 方法可迁移到 rap/poem/recitation 等有节律要求的语音生成场景
2. **Pitch-based vs lyric-aware alignment**: 两种旋律对齐策略的设计思路 — pitch-based (one-to-one, 简单) vs lyric-aware (one-to-many, 保留 melisma) — 可用于任何需要将文本与时间序列对齐的任务
3. **Registry-based modular architecture**: 通过 registry pattern 将 ASR/LLM/SVS 解耦为可插拔模块,便于 systematic benchmarking,适用于其他多模态对话系统
4. **Synthesized melody dataset pipeline**: 用 LLM 生成歌词 + Yue 合成音乐 + MFA 对齐 + RMVPE F0 提取 + ROSVOT note timing → 自动构建 music score corpus [Appendix D],可扩展到大规模 SVS 训练数据构建

## 审阅

> [!review] 审阅 (2026-06-04, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 因果解释充分,WHY cascaded/WHY phrase-constraint,速查可借鉴具体 |
> | 可信赖 | pass | 数字 claim 覆盖率 >90%,指标名正确,速查含具体数字+来源 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注 >80%,无推断写成断言 |
> | 可定位 | pass | KB 背景谱系定位具体,6 页检索命中,创新判断有对比基准 |
> | 不污染 | pass | 反向更新仅追加,无 overclaim 风险 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/SingingSDS-review.yml`
