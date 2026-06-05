---
type: paper
tier: deep
title: "Qwen3.5-Omni Technical Report"
arxiv_id: "2604.15804"
source: "Sources/Qwen3.5-Omni.pdf"
authors: [Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Ting He, Shuai Bai, Keqin Chen, Peng Wang, Pei Zhang, Xinyu Zhang, Xinfa Zhu, Yuxuan Wang, Yuanjun Lv, Yuchong Sun, Yongqi Wang, Xiong Wang, Xian Shi, Zishan Guo, Ziyang Ma, Qwen Team]
year: 2026
venue: "arXiv"
tags: [omni-model, speech-LM, multimodal, TTS, ASR, streaming, MoE, voice-cloning, multilingual, RVQ, real-time, Thinker-Talker, ARIA, audio-visual]
concepts: ["[[SpeechLanguageModel]]", "[[ResidualVectorQuantization]]", "[[Speech-TextAlignment]]", "[[StreamingSpokenDialogue]]", "[[Full-duplexSpokenDialogue]]", "[[LLM-basedTTS]]", "[[Turn-takinginSpokenDialogue]]", "[[SpeakerAdaptation]]", "[[ModalityAdaptationforSpeechLLM]]"]
models: ["[[CosyVoice2]]", "[[CosyVoice3]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[CV3-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[ResidualVectorQuantization]]✓, [[LLM-basedTTS]]✓ | 过滤: [[StreamingSpokenDialogue]](pending-review), [[Full-duplexSpokenDialogue]](pending-review), [[Speech-TextAlignment]](pending-review) | 未命中但可能相关: 无

**谱系定位**: Qwen3.5-Omni 属于 Speech Language Model 中的 Omni-model 范式,即端到端处理和生成多模态 (text/image/audio/video) 的大规模自回归基础模型。与 SpeechLM survey 中梳理的演进路线一致,它从 Moshi (2024) 的全双工实时对话 → VITA/MiniCPM-o (IPR + 多模态, 2024-2025) 方向继续推进,但在规模上跃升到数百亿参数并加入了视觉模态。

**已有认知**:
- **RVQ**: Qwen3.5-Omni 的 Talker 使用 RVQ-based codec representation (继承自 Qwen3-Omni),已确认页面记录了 RVQ 的层级信息结构 (coarse→fine) 和可变比特率特性。本文的 MTP (Multi-Token Prediction) 模块正是利用 RVQ 的残差结构,在每个解码步一次性预测所有 residual codebooks。
- **Speech-Text Alignment**: 已有页面梳理了四种建模方式 (speech-only / text-only / concatenated / alternating)。Qwen3-Omni 的 dual-channel 对齐属于 multi-sequence parallel 范式;本文的 ARIA 是一种全新的单通道自适应对齐,替代了 MFA-derived 固定交错率。
- **Streaming**: 已有页面记录了 Moshi 的 RQ-Transformer (160ms 理论延迟)、Mini-Omni 的 delayed parallel decoding 等方案。Qwen3.5-Omni 的 Hybrid MoE + chunked prefilling + ARIA 单通道统一是新的延迟优化路线。
- **LLM-based TTS**: 已确认页面记录了 VALL-E 到 CosyVoice 的演进。Qwen3.5-Omni 的 Talker 是 LLM-based TTS 的变体,但以 Thinker 的 hidden state 为条件,而非独立的 text→speech pipeline。

**创新判断**: ARIA 是相对于已有 speech-text alignment 方法的独特贡献 — 现有方法要么用 MFA 预计算固定对齐 (CosyVoice 2),要么用固定交错率 (Qwen3-Omni dual-channel),ARIA 用自适应速率约束实现动态对齐,解决了 text/speech tokenizer 编码效率不匹配问题。此外,Hybrid-Attention MoE (含 Gated Delta Net) 同时用于 Thinker 和 Talker 是工程规模上的首次。

## 速查

> [!summary] 速查
> - **一句话**: 数百亿参数的全模态大模型,采用 Thinker-Talker + Hybrid MoE 架构,通过 ARIA 动态对齐和 RVQ+MTP 实现低延迟流式语音交互,在 215 个音频/音视频 benchmark 上达到 SOTA
> - **路线**: (Text/Audio/Image/Video) → AuT encoder + Vision encoder → Thinker (Hybrid MoE) → text tokens → Talker (Hybrid MoE + ARIA) → RVQ tokens → MTP (residual codebooks) → Code2Wav (causal ConvNet) → waveform
> - **指标**: SEED-TTS test-en WER 1.26 (SOTA) [Table 8]; MMAU 82.2 (超 Gemini-3.1 Pro 81.1) [Table 5]; Fleurs ASR avg WER 6.6% (vs Gemini-3.1 Pro 7.3%) [Table 13]; Plus 首包延迟 435ms (audio) / 651ms (video) [Table 1]; 29 语言 TTS 22/29 WER 最优 [Tables 9-10]
> - **可借鉴**: ARIA 的自适应速率约束 — 对任意前缀,累计 speech/text token 比不超过全局比率,简单但有效地解决了跨语言编码效率不匹配;Thinker→Talker 条件化生成 (用 hidden state 而非仅 text tokens) 保留了上下文韵律信息
> - **局限**: 仅 API 可用,模型权重未开源;Plus 首包延迟 435ms 在 8 并发下升至 955ms;论文未报告 MOS 等主观评估;消融实验不足,ARIA 与 dual-channel 的直接对比数据缺失

## 核心问题

Qwen3.5-Omni 要解决的核心问题是: **如何构建一个统一的端到端全模态大模型,使其在文本/视觉/音频/音视频理解-推理-生成-行动的全链路上都达到 SOTA,同时支持低延迟流式交互和多语言语音生成?**

具体子问题:
1. **规模化**: 如何将 Thinker-Talker 架构从 Qwen3-Omni (30B-A3B MoE) 扩展到更大规模?— Hybrid-Attention MoE
2. **对齐不稳定**: streaming 语音合成中 text/speech tokenizer 编码效率不匹配导致跳词、错音、数字渲染模糊 — ARIA
3. **长上下文**: 如何支持 10+ 小时音频和 400s 视频? — 256k context + 显式时间戳替代 TMRoPE 绝对时间编码
4. **多语言 TTS**: 如何在 29 种语言上实现高质量语音生成? — 大规模多语言预训练 + speaker fine-tuning

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Qwen3.5-Omni 延续 Qwen2.5-Omni 提出的 **Thinker-Talker** 二层架构 [§2.1]:

- **Thinker**: 负责多模态理解和文本生成。接收 text/audio/image/video 输入,通过 AuT (Audio Transformer) 和 SigLIP2 (Vision Encoder) 编码,产出 text tokens 和 high-level 表征。采用 Hybrid MoE Transformer 架构。
- **Talker**: 负责语音生成。以 Thinker 的高层表征 + 文本输出为条件,自回归生成 RVQ codec tokens,再通过 MTP 模块预测残差 codebook,最终由 Code2Wav (causal ConvNet) 流式合成波形。同样采用 Hybrid MoE Transformer。

**与 Qwen3-Omni 的五项关键改进** [§2.1]:
1. Thinker 和 Talker 均采用 Hybrid-Attention MoE (含 Gated Delta Net)
2. 支持 256k token 长上下文
3. Multi-codebook codec + 单帧立即合成
4. ARIA 替代 dual-channel 对齐
5. 多语言扩展 (113 语言 ASR + 36 语言 TTS)

### 关键设计选择

#### 1. AuT (Audio Transformer) 编码器

AuT 是从零训练的 attention-encoder-decoder 模型 [§2.2]:
- 消耗 40M 小时音频-文本对数据 (由 Qwen3-ASR 生成) [论文原文]
- 输入: Filter bank features → 4 个 Conv2D blocks 下采样 16x → self-attention layers
- 输出: 6.25Hz token rate (每 160ms 一个 audio token) [§2.2]
- 中英多语数据比例 3.5:3.5:3 (中:英:多语) [论文原文]
- 采用动态 attention window size 训练以平衡实时预填充缓存和离线理解性能 [论文原文]

**为什么从零训练而非用预训练模型?** [agent 解读] 论文未明确讨论,但 40M 小时的数据规模已远超 Whisper 等预训练模型的训练集,且需要与 Thinker 的 Hybrid MoE 架构深度集成,专门训练可以优化端到端适配。

#### 2. 显式时间戳替代 TMRoPE

**问题**: Qwen3-Omni 使用 TMRoPE 直接将 temporal position ID 绑定到绝对时间 [§2.3, §3]:
1. 长音视频输入导致 temporal position ID 过大且稀疏,削弱长距离时序建模 [论文原文]
2. 需要在不同帧率下大规模均匀采样训练数据,增加数据构建成本 [论文原文]

**解决方案**: 在每个 video/audio-video temporal patch 前插入格式化文本字符串表示的时间戳 (秒级),让模型"自然学习"时间编码 [§2.3]。对 audio 序列,在随机间隔处插入时间戳以改善跨模态对齐 [论文原文]。

**代价**: context length 略微增加,但换来更精确的时间感知,尤其在外推长上下文多模态输入时 [论文原文]。

#### 3. ARIA (Adaptive Rate Interleave Alignment)

这是本文最核心的语音生成创新 [§2.4]:

**问题**: Qwen3-Omni 采用 dual-channel (双通道) Talker 输入,即 text track 和 speech track 各走一路再交叉。但 text tokenizer 和 speech tokenizer 编码效率不一致 (对同一内容,text tokens 数和 speech tokens 数差异大),导致 streaming 合成中跳词、错音、数字渲染模糊 [§2.1, §2.4] [论文原文]。

**ARIA 的解决**: 将 dual-channel 统一为 **single-channel interleaved formulation** [§2.4]:
- 核心约束: 对于生成序列的任意前缀,**累计 speech-to-text token ratio 不得超过对应 item 级别的全局 ratio** [论文原文]
- [agent 解读] 这意味着 ARIA 并非固定 "每 N 个 text tokens 后接 M 个 speech tokens",而是根据当前 item (如一个词或短语) 的 text/speech 编码长度比,动态决定交错频率
- 支持跨语言灵活对齐 — 特别是对编码效率较低的语言 (如 CJK 语言一个字符可能对应更多 speech tokens) [论文原文]
- 自然支持 arbitrary text-token prefixes 后接 coherent speech-token continuation [论文原文]

**为什么不用 MFA (Montreal Forced Alignment)?** [agent 解读] MFA 需要预计算对齐,不兼容 streaming;且对低资源语言可能不可靠。ARIA 的约束是运行时动态执行的。

**效果**: ARIA 显著减少了 dual-channel 中的同步开销,使 token scheduling 更高效,更适配 streaming 增量生成 [§2.5] [论文原文]。

#### 4. Hybrid-Attention MoE + Gated Delta Net

Thinker 和 Talker 均基于 Qwen3.5 的 Hybrid MoE 架构 [§2.5]:
- **MoE 效率**: 稀疏激活,平衡容量与计算成本 [论文原文]
- **Gated Delta Net (GDN)**: 特别有效于加速长音视频序列建模,显著减少长上下文推理中的 KV-cache I/O 开销 [§2.5] [论文原文]
- [agent 解读] GDN 可能是一种线性注意力变体,用于替代部分 full attention 层,使 KV-cache 不随序列长度线性增长

#### 5. Talker 的 Multi-codebook 流式生成

Talker 直接在 RVQ tokens 上操作 [§2.4]:
- 自回归预测第一层 codebook 的主 token
- **MTP (Multi-Token Prediction) 模块** (Dense Transformer): 在每个解码步输出当前帧的所有残差 codebook tokens [Fig 2]
- Code2Wav (causal ConvNet): 逐帧将 multi-codebook tokens 转为波形,实现 frame-by-frame streaming 合成 [论文原文]

**Talker system prompt**: 与 Qwen3-Omni 不同,Qwen3.5-Omni 为 Talker 引入了专门的 system prompt 指定目标音色,支持零样本 voice cloning 和可控语音生成 [§2.4]。相比传统 speaker embedding,prompt 可编码更丰富的多模态线索 (文本描述 + codec 序列) [论文原文]。

### 训练策略

**预训练 3 阶段** [§3]:
1. **S1 Encoder Alignment**: 冻结 LLM (初始化自 Qwen3.5),分别训练 AuT + adapter 和 Vision encoder + adapter
2. **S2 General Stage**: 解冻全部参数,在 ~4T tokens 上训练 (text 0.92T, audio 1.99T, image 0.95T, video 0.14T, video-audio 0.29T),序列长度 32,768
3. **S3 Long Context**: 将最大 token 长度提升至 262,144,增加长音频和长视频比例

**Thinker 后训练 3 阶段** [§4.1]:
1. **Specialist Distillation**: 先独立训练 domain-specialized teacher models (text/vision/audio),再蒸馏到统一模型
2. **On-Policy Distillation (OPD)**: 解决 audio query 响应质量低于 text query 的问题 — 用 text-conditioned response 作为 distillation target 训练 audio-conditioned generation [论文原文]。[agent 解读] 这本质上是用"读到问题后的强回答"教模型"听到问题后也能给出同样强的回答"
3. **Interaction-Aligned RL**: 针对多轮交互中 code-switching、persona inconsistency、context 退化等问题设计 reward signal [论文原文]

**Talker 训练 4 阶段** [§4.2]:
1. **General Stage**: 20M+ 小时多语言语音数据 + 多模态上下文,包括 instruction-following 语音生成
2. **Long-Context Stage**: 数据质量分层 + 高质量子集 CPT + Qwen3-Omni-Captioner 去幻觉,扩展到 64k tokens
3. **RL Stage**: DPO + GSPO (Group Sequence Policy Optimization) 对齐人类偏好
4. **Speaker Fine-tuning**: 轻量化说话人微调,捕获目标说话人特征

## 实验

| 指标 | 本文 (Plus) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (zh\|en) | 0.99\|1.26 | CosyVoice 3: 0.71\|1.45; Qwen3-Omni: 1.07\|1.39 | SEED-TTS | [Table 8] |
| MMAU | 82.2 | Gemini-3.1 Pro: 81.1 | MMAU | [Table 5] |
| VoiceBench | 93.1 | Gemini-3.1 Pro: 88.9 | VoiceBench | [Table 5] |
| MMSU | 82.8 | Gemini-3.1 Pro: 81.3 | MMSU | [Table 5] |
| ASR avg WER | 6.6% | Gemini-3.1 Pro: 7.3%; GPT-4o-Transcribe: 10.4% | Fleurs top60 | [Table 13] |
| DailyOmni | 84.6 | Gemini-3.1 Pro: 82.7 | DailyOmni | [Table 7] |
| Qualcomm IVD | 68.5 | Gemini-3.1 Pro: 66.2 | Qualcomm IVD | [Table 7] |
| Cross-lingual avg | 最优 10/12 对 | CosyVoice 3/Qwen3-Omni 次之 | CV3-Eval | [Table 11] |
| First-packet latency (audio) | 435ms (1 conc.) | Flash: 235ms | 内部 vLLM | [Table 1] |
| Multilingual TTS WER best | 22/29 语言最优 | MiniMax / ElevenLabs 次之 | TTS multilingual | [Tables 9-10] |
| Custom-voice TTS WER best | 10/29 语言最优 | Gemini-2.5 Pro / GPT-Audio / MiniMax / ElevenLabs | TTS multilingual | [Table 12] |

**关键发现**:
1. **文本能力无退化**: Qwen3.5-Omni-Plus 在 text→text 任务上与 Qwen3.5-Plus-Instruct 持平 (MMLU-Pro 85.9 vs 79.9, IFEval 89.7 vs 89.7) [Table 4] [论文原文]。OPD + interaction-aligned RL 对 instruction-following 有正向贡献 [论文原文]
2. **视觉能力轻微提升**: 在视频理解任务 (VideoMME, MLVU, MVBench) 上超过 text-only Qwen3.5-Plus-Instruct [Table 6],论文归因于联合 video-audio 训练使动态视觉感知更强 [论文原文]
3. **语音生成 RLHF 效果**: RLHF 优化后 SEED-TTS test-en WER 从未报告的 base 值降至 1.26 [Table 8] [论文原文]
4. **跨语言 TTS 突破**: zh-to-ko 错误率从 CosyVoice 3 的 14.4 降至 4.03 (~72% relative reduction) [Table 11] [论文原文]

## 局限性

1. **缺少消融实验**: 论文未提供 ARIA vs dual-channel 的直接对比实验,ARIA 的独立贡献难以量化 [agent 解读]
2. **无 MOS 评估**: 所有 TTS 评估仅用 WER 和 speaker similarity,缺少人工听感 MOS 评分 [agent 解读]
3. **仅 API 可用**: 模型权重未开源,无法验证或复现 [论文原文]
4. **高并发延迟**: Plus 在 8 并发下首包延迟达 955ms (audio) / 1980ms (video) [Table 2],实际部署受限
5. **Custom-voice 局限**: 仅用单语数据微调却用于跨语言,10/29 语言最优说明部分语言仍有差距 [Table 12]
6. **AuT 训练数据闭源**: 40M 小时数据由 Qwen3-ASR 生成,属于合成数据的 self-distillation,质量上限受限于 Qwen3-ASR [agent 解读]

## 点评

Qwen3.5-Omni 是 Qwen omni 系列的工程集大成之作。从技术角度看,最值得关注的是两个贡献:

**ARIA 的简洁性**: 在 streaming speech synthesis 的 text-speech 对齐问题上,之前的方案要么需要 MFA 预计算 (CosyVoice 2),要么用固定交错率 (Qwen3-Omni dual-channel),要么用 delay pattern (MusicGen/Mini-Omni)。ARIA 的核心约束 — "对任意前缀,speech/text ratio 不超过全局 ratio" — 在形式上极其简洁,却能自适应处理不同语言的编码效率差异。遗憾的是论文缺乏消融来证明这个约束的具体贡献。

**OPD (On-Policy Distillation)**: 用 text-conditioned 的强响应来教 audio-conditioned 的弱响应,这是一个实用且优雅的跨模态质量提升策略。它暗示了一个更普遍的原则: 在多模态模型中,输入模态的变化不应改变输出质量,可以用强模态的输出作为弱模态的 distillation target。

**规模效应**: 4T tokens 预训练 + 20M+ 小时 Talker 预训练的数据规模,加上 specialist distillation + OPD + interaction-aligned RL + DPO + GSPO 的复杂后训练流水线,体现了工程投入的深度。但这也意味着 ablation 的缺失更加可惜 — 读者无法判断哪些组件是核心贡献,哪些是锦上添花。

## 可复用的 idea

1. **ARIA 自适应速率约束**: 对于任何需要 streaming interleave 两种不同速率 token 的场景 (不限于 speech-text),ARIA 的 "prefix ratio ≤ global ratio" 约束是一个通用且简洁的方案
2. **On-Policy Distillation (OPD)**: 在多模态模型中,用强模态的 response 作为 distillation target 教弱模态,减少跨模态质量差距。可迁移到 image-text、video-text 等场景
3. **显式时间戳替代 position ID**: 对于长序列时序建模,用文本字符串时间戳替代 sparse temporal position ID,降低外推难度和数据构建成本
4. **Talker system prompt 替代 speaker embedding**: 用自然语言 + codec 序列的 prompt 指定目标音色,比 d-vector/x-vector 更灵活,支持文本描述控制
5. **Specialist distillation → unified model**: 先独立训练各模态 teacher,再蒸馏到统一模型,避免多任务训练的相互干扰

---

检索命中: [[SpeechLanguageModel]]✓, [[ResidualVectorQuantization]]✓, [[LLM-basedTTS]]✓ | 过滤: [[StreamingSpokenDialogue]](pending-review), [[Full-duplexSpokenDialogue]](pending-review), [[Speech-TextAlignment]](pending-review) | 未命中但可能相关: 无
