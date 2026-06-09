---
type: paper
tier: deep
title: "StepAudio 2.5 Technical Report"
arxiv_id: "2605.23463"
source: "Sources/Step-Audio2.5.pdf"
authors: [StepFun-Audio Team]
year: 2026
venue: "arXiv"
tags: [speech-LM, unified-foundation, ASR, TTS, realtime, RLHF, MTP, audio-language, task-specialization, MoE]
concepts: ["[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[LLM-basedTTS]]", "[[ModalityAdaptationforSpeechLLM]]", "[[Instruction-GuidedSpeechSynthesis]]"]
models: ["[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[SpeechLanguageModel]], [[SpeechTokenizer]], [[LLM-basedTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[LLM-basedTTS]]✓ | 过滤: [[ModalityAdaptationforSpeechLLM]](pending-review), [[Instruction-GuidedSpeechSynthesis]](pending-review), [[EmotionControlinTTS]](pending-review) | 未命中但可能相关: [[DifferentiableRewardOptimization]](pending-review)
>
> **Speech Language Model**: StepAudio 2.5 将 SpeechLM 推进到 "统一基座+三分支特化" 的新范式: 一个 MoE backbone 同时支持 ASR、TTS、Realtime 三个部署方向,每个方向通过不同的 post-training regime (SFT + RLHF + decoding constraints) 特化。这与 KB 中记录的 "单一模型多任务" 趋势一致,但首次系统论证了 "task specialization as directional inference" 的设计哲学。
>
> **Speech Tokenizer**: StepAudio 2.5 的 TTS 分支完全取消了 encoder-adapter 模块,音频 tokens 被视为 LLM 词表中的 "新语言",TTS 被重构为纯 next-token prediction。这与 KB 中 LLM-based TTS 的趋势一致,但走得更远——不再需要任何语音编码器参与 TTS 生成。
>
> **LLM-based TTS**: StepAudio 2.5 的 TTS 分支是 LLM-based TTS 的激进形态: 完全消除 encoder-adapter,纯 LLM decoder 生成 audio tokens,配合 RLHF + Generative Reward Model 实现人类偏好对齐。Arena 评估中以 69.1% overall win rate 超越 MiniMax-2.8-HD / ElevenLabs-v3 / Gemini-3.1-Flash-TTS [Fig 4]。

> [!summary] 速查
> - **一句话**: 统一音频-语言基座模型,通过共享 MoE backbone + task-specific post-training (RLHF-centric) + 专用解码策略,同时在 ASR/TTS/Realtime 三个方向达到 SOTA [§Abstract]
> - **路线**: 共享基座: Audio Encoder (frozen) → Adaptor → LLM Decoder (MoE, 2.2T pretrained) → 三分支: ASR (MTP-5 verifiable decoding) / TTS (纯 NTP, audio tokens as language, RLHF+GRM) / Realtime (progressive SFT + RLHF + generative rewards) [§2, Fig 1]
> - **指标**: ASR: avg CER 2.97 zh / avg WER 3.68 en / avg long-form 3.63 (全面 SOTA vs Qwen3-ASR, VibeVoice-ASR, FunASR-Nano) [Table 1]; ASR RTF 0.0053 (最快) [Table 2]; TTS: 69.1% overall arena win rate vs MiniMax/ElevenLabs/Gemini [Fig 4]; Realtime: 5 eval suites 全部 SOTA (+10 margin on human eval vs next-best) [Fig 5]
> - **可借鉴**: (1) MTP-5 for ASR: 5 个 lookahead branch 并行提议 tokens,autoregressive verification 确保正确性,RTF 低至 0.0053 [§4]; (2) Generative Reward Model (GRM): 生成式奖励模型替代标量 reward,捕获更细粒度的人类偏好 [§5.1]; (3) Progressive SFT for Realtime: conversational alignment → persona control → paralinguistic sensitivity 三阶段渐进 [§6.1.2]; (4) Dynamic rehearsal schedule: SFT 中交错通用推理任务防止灾难性遗忘 [§6.1.2]; (5) Million-scale persona matrix: 10K 种子 persona 通过 fission 算法扩展为百万级 persona-dialogue pairs [§6.2]
> - **局限**: 未公开模型参数量 (仅说 MoE backbone); TTS 评估采用 arena pairwise 而非标准 benchmark (MOS/WER),可比性有限 [§5.3]; Realtime 评估中部分基线通过 API 获取,非严格控制 [§6.3]; 数据规模极大 (2.2T tokens) 但数据组成透明度有限; 论文篇幅偏短 (19 页),部分细节不足

## 核心问题

StepAudio 2.5 的核心论点: 当文本和音频共享一个良好的多模态表征空间后,下游任务之间的差异不再是架构差异,而是 **操作模式的差异**——数据构造、优化目标和解码约束 [§1, §2.2]:

> "Once text and audio share a well-shaped representational space, the differences among downstream tasks migrate away from architecture toward operational regimes: data, objectives, and decoding constraints." [§1]

这挑战了为每个任务设计独立系统的传统思路,主张一个共享基座 + 针对性 post-training 即可 [论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

StepAudio 2.5 的核心是 **shared audio-language stack** [§2.1, Fig 1]:
- **Audio Encoder** (frozen): 将波形转换为紧凑的 acoustic embeddings [论文原文]
- **Adaptor**: 轻量级适配器将 embeddings 映射到 LLM 的 hidden space [论文原文]
- **LLM Decoder** (MoE, initialized from textual MoE LLM): 在统一序列空间中处理 text tokens 和 audio tokens [论文原文]

**设计哲学**: encoder 负责稳定的声学抽象,decoder 承载语义/上下文/指令遵循/生成等所有负担。这种非对称设计是"使模型家族连贯"的系统决策——让下游任务共享大部分模型 [§2.1] [论文原文]。

### 三个特化方向 (Task Specialization as Directional Inference)

**ASR 方向** [§4]: Audio embeddings condition decoder → generate transcript tokens
- 输出空间窄、离散、强锚定于语音信号 [论文原文]
- **MTP-5 (Multi-Token Prediction)**: 主分支预测 x_{t+1},5 个辅助分支分别预测 x_{t+2}...x_{t+6}。推理时用 autoregressive verification: 提议 6 个 token,一旦未来 token 与主分支预测不一致则回退 [§4, Fig 2] [论文原文]
- Branch weights: $w_h = \alpha^{h-1} / \sum \alpha^{j-1}$, H=5, α=0.9,指数衰减 [论文原文]
- MTP 训练分两步: (1) 冻结 backbone,仅训练 MTP blocks (lr 2e-4); (2) 解冻 adapter+decoder 联合校准 (lr 2e-5) [§4.1]

**TTS 方向** [§5]: Text + control instructions condition decoder → generate audio tokens
- **完全消除 encoder-adapter**: Audio tokens 被视为 LLM 词表中的 "新语言",TTS = pure next-token prediction [§5] [论文原文]
- 输出空间更丰富,核心挑战是忠实、自然、表达丰富 [论文原文]
- **两阶段 SFT**: (1) 全局指令零样本 TTS + 粗粒度控制; (2) 全局+内联指令精细控制 (utterance + span level) [§5.1]
- **Generative Reward Model (GRM)**: 训练生成式奖励模型 r_φ,评估候选响应 y 相对于参考 y* 的质量,产出 pairwise scalar preference score,经 reward shaping 变换后用于 PPO 优化 [§5.1, Eq. 1] [论文原文]
- **TTS 数据**: 采用 Emotional-Context-Speech annotation pipeline,结合 Montreal Forced Aligner + 量化声学特征 (F0, spectral centroid, RMS, MFCC, HNR) → LLM 生成 global + inline control descriptions [§5.2]

**Realtime 方向** [§6]: Audio understanding + response generation under strict latency
- 继承基座架构 (encoder → adaptor → decoder),产生显式 latent reasoning trace [论文原文]
- **核心挑战**: conversational coherence + persona consistency + paralinguistic sensitivity + reward sparsity [§6]
- **三阶段 Progressive SFT** [§6.1.2]:
  1. Conversational Alignment: 校准多轮对话,处理口语化表达 (不流畅/中断/打断)
  2. Persona and Stylistic Control: 百万级 persona matrix 条件训练,compositional generalization
  3. Paralinguistic Sensitivity: 真实口语交互数据,学习识别和响应副语言线索
- **Dynamic rehearsal schedule**: 持续交错通用推理任务,基于 validation metrics 动态调整,防止灾难性遗忘和风格漂移 [§6.1.2] [论文原文]
- **RLHF with Generated Rewards** [§6.1.3]: PPO + KL regularization,generative reward model 使用显式 interaction rubrics 评估,标准偏好比较管控自然度,rubric scores 管控一致性等指令敏感方面 [论文原文]
- **百万级 persona matrix** [§6.2]: 10K 人工撰写 + 验证的种子 persona → 算法 fission (重组 personality/verbal habits/emotional boundaries/interaction archetypes) → 百万级合成 persona → 配对百万级真实场景对话 [论文原文]

### 训练策略

**Foundation pretraining** [§3.2]:
- 从 textual MoE LLM 初始化
- Stage 1: 3B ASR tokens 对齐 speech-text,冻结 encoder+LLM,仅训练 adaptor [论文原文]
- Stage 2 (warmup): 128B tokens,引入 speech vocabulary,adaptor/embedding/output layer 使用更高 lr,MoE router 使用更低 lr [论文原文]
- Stage 2 (main): 800B text + 800B speech tokens,混合 ASR/TTS/S2TT/text-speech interleaved/speech-to-speech conversation,序列长度 16K [论文原文]
- Cooldown: 600B high-quality tokens,序列长度增至 32K,加入 Audio Caption + Instruct TTS [论文原文]

**Long-form ASR data** [§4.2, Fig 3]: 多系统 ROVER voting pipeline: VAD segmentation → 3 ASR systems 转写 → surface-form normalization → ROVER token-level voting → quality filtering (disagreement rate ê > 0.05 discard) → session re-composition → LLM refinement [论文原文]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ASR avg CER (Chinese) | 2.97 | Qwen3-ASR 3.17 / VibeVoice 10.19 | AISHELL-1/2, WenetSpeech, FLEURS zh | [Table 1] |
| ASR avg WER (English) | 3.68 | Qwen3-ASR 3.85 / VibeVoice 7.14 | LibriSpeech, Common Voice, VoxPopuli, FLEURS en | [Table 1] |
| ASR avg long-form | 3.63 | Qwen3-ASR 4.20 / VibeVoice 4.87 | LibriSpeech long, WenetSpeech long, Earnings22 | [Table 1] |
| ASR AISHELL-1 CER | 0.71 | Qwen3-ASR 1.49 / VibeVoice 5.19 | AISHELL-1 | [Table 1] |
| ASR RTF | 0.0053 | Qwen3-ASR 0.0094 / VibeVoice 0.1039 | 100 clips @ 30s | [Table 2] |
| TTS arena win rate (overall) | 69.1% | vs MiniMax-2.8-HD / ElevenLabs-v3 / Gemini-Flash-TTS | 774 prompts pairwise | [Fig 4] |
| TTS vs MiniMax-2.8-HD | 63.0% win | - | 774 prompts | [Fig 4] |
| TTS vs ElevenLabs-v3 | 80.0% win | - | 774 prompts | [Fig 4] |
| TTS vs Gemini-3.1-Flash-TTS | 59.4% win | - | 387 prompts | [Fig 4] |
| Realtime Human Eval | 80.41 | GPT-realtime 68.01 / Gemini-live 67.16 / Doubao 70.70 | Step-Dialogue-Human-Eval | [Fig 5] |
| Realtime Step-SPQA | 79.80 | GPT-realtime 63.20 / Doubao 33.80 | Step-SPQA | [Fig 5] |
| MTP-5 avg accepted length | 5.0/6 | MTP-3: 3.6/4, MTP-7: 6.1/8 | WenetSpeech meeting | [Table 3] |

## 局限性

1. **模型参数量未公开**: 仅说 "MoE backbone",无法评估计算需求和部署可行性 [agent 解读]
2. **TTS 评估方法论局限**: 仅用 arena pairwise win rate 评估,缺乏标准 benchmark (MOS, WER on SEED, speaker similarity) 的可比数据 [§5.3]
3. **Realtime 评估部分基线非本地**: Doubao-ASR-2603 通过 API 评估,非单卡本地推理 [§4.3]
4. **论文长度偏短 (19页)**: 对于覆盖 ASR+TTS+Realtime 三个完整系统的工作,许多实现细节不足 [agent 解读]
5. **数据规模极大但透明度有限**: 2.2T tokens 的详细构成和清洗方法未充分描述 [agent 解读]
6. **与 Step-Audio (v1) 的关系不完全清晰**: 架构从 AQTA+TTS 变为 unified foundation,但 tokenizer 设计 (dual-codebook?) 的继承/变化未明确说明 [agent 解读]

## 点评

StepAudio 2.5 提出了一个重要的系统观点: **task specialization = directional inference**。这不是架构创新而是范式认知创新——一旦接受"统一基座 + 方向性推理"的框架,很多设计决策就变得自然: ASR 方向需要 grounded decoding (MTP)、TTS 方向需要 preference alignment (RLHF)、Realtime 方向需要 persona conditioning + paralinguistic sensitivity [agent 解读]。

**MTP-5 for ASR 是最具技术亮点的贡献**: 利用语音信号的 grounding 属性 (acoustic determinism) 实现高效多 token 投机解码。RTF 0.0053 意味着 30 秒音频仅需 ~160ms 解码,这是生产部署的关键指标。论文提出的 insight——"grounding is not only a source of information; it is also a source of algorithmic structure"——值得更广泛的探索 [§4, Insight] [论文原文]。

**Generative Reward Model (GRM) 是 TTS RLHF 的重要进展**: 传统标量 reward 难以捕获 TTS 的多维质量 (自然度 + 表达力 + 指令遵循度 + persona 一致性)。生成式 reward model 输出更丰富的反馈信号,但论文对 GRM 的具体架构和训练细节描述过少 [agent 解读]。

**Realtime 的 million-scale persona matrix 展示了工业化 SFT 数据生产的前沿**: 从 10K 种子 persona 通过算法 fission 扩展为百万级,每个 persona 配对真实场景对话,这种规模的 persona 工程在学术界很难复现 [agent 解读]。

## 可复用的 idea

1. **MTP-5 verifiable decoding**: 5 个 lookahead branch + autoregressive verification,适用于任何 grounded generation 任务 (ASR, translation 等 acoustic-deterministic 场景)
2. **Generative Reward Model**: 替代标量 reward 的生成式评估,适用于多维度质量评估的 RLHF 任务
3. **Task specialization as directional inference**: 统一基座 + 方向性 post-training 的设计哲学
4. **Progressive SFT for dialogue**: conversational → persona → paralinguistic 的三阶段渐进,配合 dynamic rehearsal schedule 防遗忘
5. **Million-scale persona fission**: 少量人工种子 → 算法重组 → 百万级 persona-dialogue pairs
6. **ROVER voting for long-form ASR data**: 多 ASR 系统投票 + quality filtering + LLM refinement 生产高质量长音频标注


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排
