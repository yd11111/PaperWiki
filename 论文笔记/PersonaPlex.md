---
type: paper
tier: deep
title: "PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models"
arxiv_id: "2602.06053"
source: "Sources/PersonaPlex.pdf"
authors: [Rajarshi Roy, Jonathan Raiman, Sang-gil Lee, Teodor-Dumitru Ene, Robert Kirby, Sungwon Kim, Jaehyeon Kim, Bryan Catanzaro]
year: 2026
venue: "ICASSP 2026 (under review)"
tags: [full-duplex, voice-cloning, role-conditioning, duplex-speech, speaker-similarity, conversational-AI, NVIDIA]
concepts: ["[[Full-duplexSpokenDialogue]]", "[[Turn-takinginSpokenDialogue]]", "[[VoiceCloningTaxonomy]]", "[[SpeakerEmbedding]]", "[[SpokenDialogueEvaluation]]", "[[StreamingSpokenDialogue]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[SpeakerEmbedding]], [[ProsodyModeling]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeakerEmbedding]]✓, [[ProsodyModeling]]✓ | 过滤: [[Full-duplexSpokenDialogue]](pending-review), [[Turn-takinginSpokenDialogue]](pending-review), [[VoiceCloningTaxonomy]](pending-review), [[SpokenDialogueEvaluation]](pending-review), [[StreamingSpokenDialogue]](pending-review) | 未命中但可能相关: 无
>
> **Speaker Embedding** [待确认]: PersonaPlex 通过 voice prompt (短音频样本) 实现 zero-shot voice cloning,属于 KB 中 "Speaker Encoder (零样本)" 范式的全双工对话场景延伸。KB 记录了 d-vector→x-vector→ECAPA-TDNN→in-context prompt (VALL-E) 的演进,PersonaPlex 的 voice prompt 继承了这一路线但应用于 Moshi 风格的全双工架构。使用 WavLM-TDNN speaker verification 模型评估 speaker similarity (SSIM)。[agent解读]
>
> **Prosody Modeling** [待确认]: 对话场景中的韵律建模与单句 TTS 有本质差异。PersonaPlex 需要在实时全双工交互中保持一致的角色韵律 (自然对话节奏、回传信号、打断后恢复等),这超越了传统 prosody modeling 的静态控制范畴。[agent解读]
>
> [待确认] **Full-duplex Spoken Dialogue**: KB 记录了从 dGSLM→Moshi→LSLM→VITA 的全双工演进。PersonaPlex 直接基于 Moshi 架构,但新增了此前全双工系统缺失的两个维度: voice conditioning 和 role conditioning。这是首次在全双工系统中实现零样本声音克隆。[agent解读]

> [!summary] 速查
> - **一句话**: 首个在全双工对话模型中同时实现 zero-shot voice cloning 和 text-based role conditioning 的系统,通过 Hybrid System Prompt (text+voice) 在 Moshi 架构上实现角色化对话 [§1]
> - **路线**: [text prompt (角色描述) + voice prompt (短音频样本)] → Hybrid System Prompt → Moshi (Depth+Temporal Transformer, Mimi codec) → 三通道 (user audio, agent text, agent audio) 自回归生成 [§3.1, Fig 1]
> - **指标**: DMOS 3.90 (Full-Duplex-Bench, SOTA) vs Moshi 2.83 / Gemini 3.72 [Table 1]; SSIM 0.57 (voice cloning, SOTA) vs Moshi 0.10 / Gemini 0.00 [Table 1]; Service-Duplex-Bench GPT-4o 均分 4.48 vs Gemini 4.73 / Moshi 1.75 [Table 4]; 发布模型 DMOS 2.95, SSIM 0.65 [Table 7, Appendix A]
> - **可借鉴**: (1) Hybrid System Prompt: text+voice 两段拼接实现双重 conditioning,顺序无关 [§3.1]; (2) 合成数据方案: Qwen-3-32B/GPT-OSS-120B 生成文本对话 → Dia/ChatterboxTTS 合成语音,覆盖客服+QA 场景 [§3.2]; (3) Service-Duplex-Bench: 50 场景 x 7 问题的角色遵循评估扩展 [§3.3]
> - **局限**: 仅英语 [论文原文]; 基于 Moshi 7B 架构限制了智能上限 [agent解读]; 未开源训练数据/代码,仅开源模型权重 (personaplex-7b-v1) [Appendix A]; text prompt 细节程度对角色遵循影响显著 [§3.2]

## 核心问题

PersonaPlex 要解决当前全双工对话系统的两大能力缺失 [§1]:

1. **声音固定**: 现有全双工模型 (Moshi, Gemini Live, GPT-realtime) 只能使用固定声音,无法根据应用场景切换角色声音 [论文原文]
2. **角色不可控**: 缺乏 text-based role conditioning,无法通过指令控制模型扮演特定角色 (如客服人员、教师等) [论文原文]
3. **Voice cloning + duplex 的双重挑战**: 将 voice conditioning 引入全双工流式系统面临延迟约束和 coupled speech-text dynamics 的耦合困难 [论文原文]

## 方法: PersonaPlex

### 架构 [§3.1, Fig 1]

PersonaPlex 基于 Moshi 架构 [2],接收三通道输入:

1. **User audio**: 用户实时语音输入
2. **Agent text**: 模型生成的文本 token (Inner Monologue)
3. **Agent audio**: 模型生成的语音 token

核心创新是 **Hybrid System Prompt** — 在对话开始前拼接两段 conditioning 信息 [§3.1]:

#### Text Prompt Segment [§3.1]
- 在 agent text 通道注入角色描述文本 token (如 "You are a customer service agent for...")
- Agent audio 通道静默 (padding)
- 通过强制 scenario-specific text tokens 实现 role conditioning [论文原文]

#### Voice Prompt Segment [§3.1]
- 在 agent audio 通道注入短音频样本 (speaker reference)
- Agent text 通道静默 (padding)
- 后续生成自动克隆该声音 [论文原文]
- User audio 通道替换为 440 Hz 正弦波以稳定 voice cloning [论文原文]

**顺序无关**: 实验表明 voice prompt 在前或 text prompt 在前不影响性能。实际部署中 voice prompt 放前面可启用 prefilling,在不需要 voice cloning 时减少延迟 [§3.1]

**训练**: 对 system prompt 部分 mask 掉 loss 反向传播; 遵循 Moshi 的 token imbalance 策略: 非语义 audio tokens loss 权重 0.02, padded text tokens 权重 0.3 [§3.1]

### 合成训练数据 [§3.2]

#### Dialog Transcript Generation [§3.2]
- 使用 **Qwen-3-32B** [20] 和 **GPT-OSS-120B** [21] 生成两人对话文本 [§3.2]
- **Service Scenarios**: 先采样服务域 (restaurant, bank...) → 选场景类型 (refund, enquiry...) → 生成高层描述 → 扩展为完整双人对话 [§3.2]
- **QA Scenarios**: 双轮问答对话,角色固定为 "wise and friendly teacher" [§3.2]

#### Dialog Speech Synthesis [§3.2]
- 使用 **26,296 个单说话人音频样本** (来自 VoxCeleb, Libriheavy, LibriTTS, CommonAccent, Fisher) [§3.2]
- 2,630 个样本保留用于 speaker similarity 测试 [§3.2]
- **Service 场景**: 使用 **Dia** [27] (多说话人 TTS) 同时生成双方语音,保留交互时序 [§3.2]
- **QA 场景**: 使用 **Chatterbox TTS** [28] (zero-shot TTS) 逐轮生成 [§3.2]
- 负持续时间静默插入模拟自然 turn-taking 的重叠和打断 [§3.2]

#### Role Context Generation [§3.2]
- 每个服务场景生成对应的 role context (包含代理名称、机构信息、产品详情等) [Table 3]
- Text prompt 精细度分三级: Minimal ("You enjoy having a good conversation") / Topic-specific / Highly detailed [Appendix A]

#### Voice Data [§3.2, Appendix A]
- 发布版本使用 TortoiseTTS [32] 合成声音 + Praat [33] 做 pitch 和 formant augmentation [Appendix A]
- 所有对话使用 ChatterboxTTS 统一生成 (speaker consistency 更好: SSIM 0.65 vs 0.57) [Appendix A]

### 数据规模 [§4]

- **1840 小时**客服对话 (105,410 dialogs) + **410 小时** QA 对话 (39,322 dialogs) [§4]
- Batch size 32, 最大序列长度 2048 tokens [§4]
- 24,576 training steps, 约 163.84 秒/step [§4]
- 8x A100 GPU, 6 小时训练 [§4]

### Service-Duplex-Bench [§3.3]

扩展 Full-Duplex-Bench [1] 的角色评估维度:

- 50 个服务角色场景,每场景 7 个问题 = 350 个评估问题 [§3.3]
- 7 类问题: Q0 (Proper Noun) → Q1 (Context details) → Q2 (Context details) → Q3 (Unfulfillable Request) → Q4 (Customer Rudeness) → Q5 (Unspecified) → Q6 (Unrelated) [Table 3]
- 测试 proper noun recall, context adherence, 不可满足请求处理, 客户粗鲁处理等能力 [§3.3]
- 训练场景与评估场景不重叠 [§3.2]

## 实验结果

### Dialog Naturalness & Voice Cloning [§4.1, Table 1]

| 模型 | DMOS (Full-Duplex) | DMOS (Service-Duplex) | SSIM |
|------|-------------------|----------------------|------|
| **PersonaPlex** | **3.90 +/- 0.15** | **3.59 +/- 0.12** | **0.57** |
| Gemini [14] | 3.72 +/- 0.14 | 3.22 +/- 0.14 | 0.00 |
| Qwen-2.5-Omni [7] | 3.70 +/- 0.13 | 2.37 +/- 0.20 | 0.07 |
| Freeze-Omni [19] | 3.51 +/- 0.18 | 2.38 +/- 0.21 | 0.05 |
| Moshi [2] | 2.83 +/- 0.13 | - | 0.10 |

- DMOS 评估: AMT 众包,8 个音频样本/评估者,1-5 分 [§4.1]
- SSIM: WavLM-TDNN speaker verification,cosine similarity [§4.1]
- PersonaPlex SSIM 0.57 显著领先 (其他模型基本为 0) [论文原文]
- Gemini 和 Qwen-2.5-Omni 不支持 voice cloning,SSIM 为 0 [论文原文]

### Full-Duplex-Bench [§4.2, Table 2]

PersonaPlex 在交互维度表现:
- **Smooth Turn Taking**: TOR 0.992, Latency 0.070 (最佳) [Table 2]
- **Backchannel**: TOR 0.025 (最低,最少误触), JSD 0.649 [Table 2]
- **User Interruption**: TOR 1.000, GPT-4o 4.210, Latency 0.400 [Table 2]
- **Pause**: Synthetic 0.584, Candor 0.662 [Table 2]

### Service-Duplex-Bench [§4.2, Table 4]

| 模型 | Q0 | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Mean |
|------|-----|-----|-----|-----|-----|-----|-----|------|
| Gemini | 4.6 | 4.7 | 4.8 | 4.9 | 4.5 | 4.7 | 4.9 | **4.73** |
| **PersonaPlex** | 4.6 | 4.6 | 4.4 | 4.5 | 4.5 | 4.3 | 4.5 | 4.48 |
| Freeze-Omni | 3.9 | 3.5 | 3.8 | 4.3 | 4.1 | 4.2 | 4.3 | 4.02 |
| Qwen-2.5-Omni | 1.3 | 1.6 | 2.6 | 3.4 | 3.3 | 3.6 | 3.5 | 2.76 |
| Moshi | 1.5 | 1.4 | 1.8 | 2.0 | 1.9 | 2.1 | 1.6 | 1.75 |

- PersonaPlex 在角色遵循上仅次于 Gemini (商业闭源),超越所有开源模型 [论文原文]
- Moshi 和 Qwen-2.5-Omni 在角色遵循方面几乎不可用 (Q0 仅 1.3-1.5) [论文原文]

### Dataset Scale Effect [§4.3, Table 5]

| Data % | SSIM | GPT-4o (FDB) | GPT-4o (SDB) |
|--------|------|-------------|-------------|
| 100% | **0.57** | 4.21 | **4.48** |
| 50% | 0.56 | **4.52** | 4.24 |
| 25% | 0.54 | 4.44 | 4.20 |
| 0% (Moshi) | 0.10 | 1.75 | - |

- Voice cloning 随数据量提升稳步改善 [§4.3]
- Full-Duplex-Bench 上少量数据即可达到强性能 [§4.3]
- Service-Duplex-Bench 上角色遵循随数据量持续改善 [§4.3]

### Released Checkpoint [Appendix A]

发布版 (personaplex-7b-v1) 相比论文实验版的改进:
- 额外训练 7,303 真实对话 (Fisher corpus, 1,217 小时) 改善 backchannel 和表情 [Appendix A]
- 使用 TortoiseTTS + Praat augmentation 合成声音替换真实 speaker 数据 [Appendix A]
- 统一使用 ChatterboxTTS 生成对话 (SSIM 从 0.57 提升至 0.65) [Appendix A]
- DMOS 2.95 (vs Moshi 2.44) [Table 7]

## 核心设计选择分析

### WHY: 为什么基于 Moshi 而非 cascaded 系统 [§1, §2]

1. **真正的全双工**: 半双工系统不能边听边说,不支持自然 turn-taking [论文原文]
2. **副语言保留**: 级联 ASR→LLM→TTS 丢失语气、情感等信息 [论文原文]
3. **已有可复现基线**: Moshi 是唯一完全开源的全双工模型 [agent解读]

### WHY: Hybrid System Prompt 而非独立 conditioning 模块 [§3.1]

1. **不改变底层架构**: 所有 conditioning 通过输入端注入,Moshi 参数初始化不变 [论文原文]
2. **灵活性**: 可独立使用 text-only (无 voice cloning) 或 voice-only (无 role conditioning) [论文原文]
3. **440 Hz 正弦波替换**: 在 voice prompt 段将 user audio 通道替换为正弦波,避免真实 user audio 干扰 voice cloning 稳定性 [论文原文]

### WHY: 合成数据而非真实对话 [§3.2]

1. 角色化客服对话的真实多轮数据极难获取 [agent解读]
2. 合成数据可精确控制角色描述的精细度 [§3.2]
3. 实验验证合成数据训练即可达到 SOTA [§4]

## 与已有工作的差异

| 维度 | Moshi | Gemini Live | PersonaPlex |
|------|-------|-------------|-------------|
| 全双工 | 是 | 是 | 是 |
| Voice cloning | 否 (SSIM 0.10) | 否 (SSIM 0.00) | 是 (SSIM 0.57) |
| Role conditioning | 否 | 是 (context prompt) | 是 (hybrid prompt) |
| 开源 | 是 | 否 | 模型开源 |
| 对话自然度 DMOS | 2.83 | 3.72 | **3.90** |

## 论文贡献与意义

1. **首个 voice+role conditioned 全双工模型**: 在不改变 Moshi 底层架构的前提下,通过 Hybrid System Prompt 实现双重 conditioning [§1]
2. **Service-Duplex-Bench**: 首个面向角色遵循的全双工评估 benchmark (50 场景 x 7 问题) [§3.3]
3. **合成数据方案的可行性**: 证明纯合成对话数据可训练出超越所有开源全双工模型的系统 [§4]
4. **首个开源达到商业系统自然度的全双工模型**: DMOS 3.90 超越 Gemini 3.72 [Table 1]

---

检索命中: [[SpeakerEmbedding]], [[ProsodyModeling]] | 过滤: [[Full-duplexSpokenDialogue]](pending-review), [[Turn-takinginSpokenDialogue]](pending-review), [[VoiceCloningTaxonomy]](pending-review), [[SpokenDialogueEvaluation]](pending-review), [[StreamingSpokenDialogue]](pending-review) | 未命中但可能相关: 无
