---
type: paper
tier: deep
title: "MOSS-TTSD: Text to Spoken Dialogue Generation"
arxiv_id: "2603.19739"
source: "Sources/MOSS-TTSD.pdf"
authors: [Yuqian Zhang, Donghua Yu, Zhengyuan Lin, Botian Jiang, Mingshu Chen, Yaozhou Jiang, Yiwei Zhao, Yiyang Zhang, Yucheng Yuan, Hanfu Chen, Kexin Huang, Jun Zhan, Cheng Chang, Zhaoye Fei, Shimin Li, Xiaogui Yang, Qinyuan Cheng, Xipeng Qiu]
year: 2026
venue: "arXiv"
tags: [spoken-dialogue, multi-speaker, long-form, voice-cloning, codec-LM, autoregressive, RVQ, evaluation, multilingual]
concepts: ["[[LLM-basedTTS]]", "[[CodecLanguageModel]]", "[[ResidualVectorQuantization]]", "[[VoiceCloningTaxonomy]]", "[[Turn-takinginSpokenDialogue]]", "[[SpokenDialogueEvaluation]]", "[[AudioTokenizerTaxonomy]]"]
models: ["[[论文笔记/VibeVoice|VibeVoice]]", "[[论文笔记/FireRedTTS2|FireRedTTS 2]]", "[[论文笔记/ZipVoice-Dialog|ZipVoice-Dialog]]", "[[论文笔记/DialoSpeech|DialoSpeech]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: MOSS-TTSD 属于 [[LLM-basedTTS]] 中的 Codec Language Model 路线,与 [[论文笔记/VibeVoice|VibeVoice]] 和 [[论文笔记/FireRedTTS2|FireRedTTS 2]] 同属 "Dialogue/Multi-speaker 扩展" 方向。在已有知识库中,LLM-based TTS 的演进路线已覆盖从 VALL-E 到 CosyVoice 到 Instruction-aware 的脉络,但长对话多说话人方向的条目仍较少。

**已有认知对比**:
- **离散语音生成范式**: 与 [[CodecLanguageModel]][待确认] 中描述的 codec LM 范式一致 -- 直接在 neural codec 产生的 RVQ token 上做自回归生成。不同之处在于 MOSS-TTSD 仅建模前 16 层 RVQ(2kbps/12.5Hz),比常见的 4-8 层更多但通过低比特率保持长上下文可控性。
- **RVQ 多头延迟模式**: [[ResidualVectorQuantization]] 中记录了 RVQ 的多层建模策略(AR+NAR、delay pattern 等),MOSS-TTSD 采用 MusicGen 式 multi-head delay pattern 是已知方案的直接应用。
- **Voice Cloning**: [[VoiceCloningTaxonomy]][待确认] 将 zero-shot voice cloning 分类为 codec-based 路线(VALL-E 式 in-context learning),MOSS-TTSD 的 voice_clone + continuation 组合是对 reference-conditioned 和 continuation-based 两种范式的融合。
- **Turn-taking**: [[Turn-takinginSpokenDialogue]][待确认] 中记录了端到端和级联两种 turn-taking 实现方式。MOSS-TTSD 不做全双工实时对话,而是以 script-conditioned 方式(显式 speaker tags)处理轮次,属于 TTS 式 turn-taking 而非交互式 turn-taking。
- **评估**: [[SpokenDialogueEvaluation]][待确认] 总结了 11 维度评估框架,但缺少针对 script-to-dialogue 场景(非交互式对话)的专用 metric。MOSS-TTSD 提出的 TTSD-eval 用 forced alignment 替代 speaker diarization,填补了这一空白。

**创新判断**: 相对已有工作,MOSS-TTSD 的核心新意在于 (1) 将 LLM-based TTS 扩展到 60 分钟/5 说话人的长对话场景,(2) 提出不依赖 diarization 的评估方案 TTSD-eval。架构层面(Qwen3 + RVQ + delay pattern)是成熟技术的系统集成,数据工程(三阶段 curriculum + 合成数据增强)是主要工程贡献。

> 检索命中: [[LLM-basedTTS]]✓, [[ResidualVectorQuantization]]✓ | 参考(待确认): [[CodecLanguageModel]], [[VoiceCloningTaxonomy]], [[Turn-takinginSpokenDialogue]], [[SpokenDialogueEvaluation]] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 Qwen3-8B + MOSS-Audio-Tokenizer 的全离散 spoken dialogue 合成模型,支持 60 分钟单次生成、5 说话人、零样本声音克隆和多语言
> - **路线**: 对话脚本(含 [S1]/[S2] speaker tags + 可选参考音频) → Qwen3-8B-base 自回归生成 16 层 RVQ tokens(multi-head delay pattern） → MOSS-Audio-Tokenizer decoder → 波形
> - **指标**: ACC 0.9587/0.9626, SIM 0.7949/0.7326, WER 4.85%/9.88% (ZH/EN) [Table 1]; Elo ratings 全维度领先开源模型 [Fig 4]; vs ElevenLabs V3/Gemini/Doubao 主观偏好多数胜出 [Fig 5]
> - **可借鉴**: (1) voice_clone + continuation 双范式融合显著提升 speaker similarity; (2) 基于 forced alignment 的 TTSD-eval 避免 diarization 误差; (3) 三阶段 curriculum 从单人到多人的训练策略
> - **局限**: 非实时交互(script-conditioned, 非 full-duplex); 60 分钟上限依赖 65K token 上下文; 多说话人(3-5 人)训练依赖合成拼接数据,真实多人数据极少; TTSD-eval 依赖 forced alignment 质量; 未报告 MOS

## 核心问题

1. **为什么现有 TTS 模型在对话合成上不够好?** 现有 TTS 主要针对单句/单说话人设计,缺乏对话上下文建模,导致跨轮次的声学一致性差、说话人切换不自然、长时生成不稳定 [§1]。
2. **如何在全离散范式下实现长时多人对话?** 通过仅建模 16 层 RVQ(低比特率 2kbps/12.5Hz)配合 65K token 上下文的 Qwen3-8B,将最大训练上下文扩展到 3600 秒 [§4.1]。
3. **如何评估对话合成的说话人归属和相似度?** 现有 cpWER/cpSIM 依赖 speaker diarization 模型,误差随说话人数增加而显著恶化。TTSD-eval 通过 forced alignment 直接从脚本获取 speaker 标签,绕过 diarization [§5.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MOSS-TTSD 采用全离散语音生成范式 [§4.1]:
- **LLM backbone**: Qwen3-8B-base,用于自回归序列建模
- **Audio tokenizer**: MOSS-Audio-Tokenizer,12.5Hz 帧率,支持低至 2kbps 比特率
- **Token 生成策略**: multi-head delay pattern(源自 MusicGen [8]),自回归预测 RVQ codebook tokens
- **RVQ 层数**: 仅建模前 16 层(与 MOSS-TTS 全层建模不同)

输入格式: 对话脚本包含显式 speaker tags(`[S1]`/`[S2]` 等)+ 可选的 per-speaker 参考音频 [§4.1, Appendix A]。

[论文原文] 作者解释仅用 16 层 RVQ 的原因: "Thanks to the strong performance of MOSS-Audio-Tokenizer at a low bitrate of 2 kbps with a 12.5 Hz frame rate, the LLM can robustly model audio sequences in long-context, multi-speaker settings" [§4.1]。
[agent 解读] 减少 RVQ 层数直接缩短了每帧的 token 数量,使相同上下文窗口能覆盖更长时间的音频。16 层在保持可接受重建质量的同时,将序列长度控制在 65K token 内可覆盖约 3600 秒。

### 关键设计选择

**1. 为什么选 multi-head delay pattern 而不是 AR+NAR 两阶段?**
[agent 解读] 论文未直接讨论此选择的原因,但 multi-head delay pattern 允许在单个自回归 pass 中同时生成多层 RVQ token,避免了两阶段管线的额外复杂性。对于长对话场景,单 pass 生成更利于保持跨时间步的一致性。

**2. Voice clone + continuation 双范式融合**
MOSS-TTSD 结合两种声音克隆方式 [§4.3, Fig 2]:
- **Voice clone (reference-conditioned)**: 通过 chat template 中的参考音频 slot 显式指定说话人音色
- **Continuation**: 自回归 TTS 天然支持从前文音频续写

[论文原文] "We find that combining these two paradigms... substantially improves voice cloning performance" [§4.3]。Appendix C 的消融实验 [Table 4-5] 显示 voice_clone_and_continuation 在 SIM 上全面优于单独使用任一方式(EN SIM: 0.6828 vs 0.6075/0.6579; ZH SIM: 0.7590 vs 0.7160/0.7513 [Table 4])。

**3. 数据合成: 为什么要用拼接方式造多说话人数据?**
[论文原文] "Accurately annotating speaker identities over long contexts in multi-speaker scenarios (3–5 speakers) poses a significant challenge, both for traditional speaker diarization models and for end-to-end systems" [§3.2]。真实多人数据的标注质量不可靠,因此保留少量高质量真实数据并用单人段拼接来增强多说话人训练集。拼接时严格筛选 DNSMOS ≥ 3.4 且同采样率 [§3.2]。

### 训练策略

三阶段 curriculum learning [§4.2]:

| 阶段 | 数据 | 序列长度 | 目标 |
|------|------|----------|------|
| Stage 1 | 所有单人+双人(DNSMOS ≥ 2.8)+ voice cloning 数据 | 32K → 65K tokens | 适应长上下文 + 学习 speaker tag 控制 |
| Stage 2 | Stage 1 子集(DNSMOS ≥ 3.4, 采样率 ≥ 24kHz),降低单人比例 | 65K tokens | 高保真对话合成 |
| Stage 3 | Stage 2 + 多说话人真实数据 + 合成拼接数据 | 65K tokens | 1-5 说话人 + 改善轮次切换 |

[论文原文] 从 MOSS-TTS 的中间 checkpoint(基于 Qwen3-8B-base,在单人 TTS 数据上预训练,32K tokens)继续训练 [§4.2, Stage 1]。

**数据工程细节** [§3]:
- 数据管线与 MOSS-TTS 共享基础处理(数据收集、音频归一化、speaker diarization)[§3.1]
- ASR 使用 MOSS Transcribe Diarize 端到端生成带 speaker tag 的转写 [§3.1]
- 特定噪声域(电影、电竞)额外使用 MossFormer2 降噪 [§3.1]
- 训练数据保留 DNSMOS ≥ 2.8 [§3.1]
- 文本增强: 规则替换标点以改善对多样文本输入的鲁棒性 [§3.2, Appendix B]

### TTSD-eval 评估框架

[§5.1, Fig 3]:
1. 使用 MMS-FA (forced alignment) 获取输入脚本与生成音频之间的词级对齐
2. 按标点将音频切分为句子片段,直接从脚本的 speaker tags 分配说话人身份
3. 使用 wespeaker-SimAMResNet100 计算每个片段与所有候选说话人的相似度
4. **ACC (Speaker Attribution Accuracy)**: 预测的说话人标签与脚本中 ground truth 的匹配率
5. **SIM (Speaker Similarity)**: 每个片段与其 ground truth 说话人参考音频的相似度
6. **WER**: 使用 Whisper-large-v3 + Seed-TTS-eval 文本归一化 [§5.1]

[论文原文] 相对于 cpWER/cpSIM 的优势: "As the number of speakers increases from two to five or more, the error introduced by speaker diarization tends to grow substantially" [§5.1]。TTSD-eval 完全绕过 diarization,利用已知的脚本-音频对应关系。

## 实验

| 指标 | MOSS-TTSD | VibeVoice 7B | FireRedTTS-2 | Higgs Audio V2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| ZH ACC ↑ | 0.9587 | 0.9222 | 0.8798 | 0.9022 | TTSD-eval ZH | [Table 1] |
| ZH SIM ↑ | 0.7949 | 0.7590 | 0.7415 | 0.7383 | TTSD-eval ZH | [Table 1] |
| ZH WER ↓ | 4.85% | 5.70% | 8.18% | 7.68% | TTSD-eval ZH | [Table 1] |
| EN ACC ↑ | 0.9626 | 0.9554 | 0.9353 | 0.9025 | TTSD-eval EN | [Table 1] |
| EN SIM ↑ | 0.7326 | 0.7140 | 0.6961 | 0.6860 | TTSD-eval EN | [Table 1] |
| EN WER ↓ | 9.88% | 9.46% | 11.33% | 21.31% | TTSD-eval EN | [Table 1] |
| vs Eleven V3 (ZH) | ACC 0.9736 vs 0.9653 | - | - | - | TTSD-eval ZH | [Table 1] |
| vs Eleven V3 (ZH) | SIM 0.8165 vs 0.6970 | - | - | - | TTSD-eval ZH | [Table 1] |
| vs Gemini-2.5-pro (EN) | ACC 0.9655 vs 0.9537 | - | - | - | TTSD-eval EN | [Table 1] |
| vs Gemini-2.5-pro (EN) | SIM 0.7893 vs 0.6786 | - | - | - | TTSD-eval EN | [Table 1] |

**Elo Rating (主观评估, 开源模型)** [Fig 4]:
- ZH Overall: MOSS-TTSD ~1044, VibeVoice 7B ~998, FireRedTTS-2 ~982, VibeVoice 1.5B ~991, Higgs Audio V2 ~935
- EN Overall: MOSS-TTSD ~1027, VibeVoice 7B ~1010, VibeVoice 1.5B ~1005, Higgs Audio V2 ~990, FireRedTTS-2 ~941

**主观偏好 (vs 商业模型)** [Fig 5]:
- vs Doubao Podcast (ZH): Win 44.4%, Tie 20.2%, Lose 35.4%
- vs Eleven V3 (ZH): Win 35.8%, Tie 32.6%, Lose 31.6%
- vs Gemini-2.5-pro (EN): Win 45.5%, Tie 21.2%, Lose 33.3%
- vs Eleven V3 (EN): Win 33.3%, Tie 25.3%, Lose 41.4%

**Voice cloning 消融** [Table 4-5, Appendix C]:

| 配置 | EN SIM | ZH SIM | ZH ACC | ZH WER |
| --- | --- | --- | --- | --- |
| voice_clone only | 0.6075 | 0.7160 | 0.9387 | 6.07% |
| continuation only | 0.6579 | 0.7513 | 0.9254 | 5.46% |
| voice_clone + continuation | **0.6828** | **0.7590** | **0.9587** | 4.85% |

**测试集设计** [§5.1]: 中英各 50 个对话样本,20 对人工收集的说话人参考 + 30 对来自 seed-tts-eval,对话文本由 Gemini 2.5 Pro 生成,音频时长 30~720 秒,覆盖 podcast/配音/体育解说/相声等场景。

**实验设置说明**: 由于上下文长度和 GPU 内存限制,Higgs Audio V2 和 FireRedTTS-2 的 generation_chunk_buffer_size 设为 6(分块生成,每块仅条件化 prompt + 最近 6 块) [§5.1]。商业模型使用各自声音库(非统一参考音频),MOSS-TTSD 对应使用相同声音做零样本克隆 [§5.1]。

## 局限性

1. **非交互式**: MOSS-TTSD 是 script-to-speech 模型,不支持实时对话交互(full-duplex/streaming),应用场景限于预编排内容(podcast/audiobook/dubbing)
2. **评估偏差**: 与商业模型的对比使用了不同声音库(ElevenLabs/Gemini/Doubao 各自的私有声音 vs MOSS-TTSD 零样本克隆),评估条件不完全对等 [§5.1]
3. **多说话人数据瓶颈**: 3-5 说话人训练主要依赖合成拼接数据,真实多人对话数据极少 [§3.2];拼接数据可能无法完全模拟真实对话的韵律交互
4. **评估指标不完整**: 未报告 MOS(语音自然度/质量);TTSD-eval 的 ACC/SIM/WER 三个指标不能完整反映对话合成质量(如韵律自然度、情感一致性)
5. **EN WER 偏高**: 即使是最优模型(MOSS-TTSD)英文 WER 仍接近 10%,说明英文长对话可靠性仍有提升空间 [Table 1]
6. **上下文窗口限制**: 60 分钟上限受限于 65K token 上下文窗口 [§4.1],更长内容需要外部机制

## 点评

**优势**:
- 将 LLM-based TTS 扩展到长对话多说话人场景是实际需求驱动的合理方向,60 分钟/5 说话人的能力在开源模型中领先
- TTSD-eval 的设计理念(用 forced alignment 替代 diarization)解决了真实痛点,在评估方法论上有贡献
- voice_clone + continuation 的组合在消融实验中效果显著,是一个简单有效的工程 insight
- 迭代开发历史透明(v0→v1.0),有助于理解设计演进

**不足**:
- 架构层面创新有限:Qwen3-8B + RVQ + delay pattern 都是成熟组件的组合,无新的建模贡献
- 与 VibeVoice 7B(同样基于 LLM backbone + RVQ)的差异主要在数据工程层面,但论文未提供训练数据量和计算资源的对比
- 主观评估中 vs Eleven V3 (EN) 实际是 Lose (41.4% vs 33.3%) [Fig 5],但论文在 abstract 中声称"surpasses strong proprietary baselines"有 overclaim 嫌疑
- 缺少 MOS 评估是重要遗漏,ACC/SIM 高不等于语音听感好

**整体**: 这是一篇工程导向的系统论文,核心贡献在数据工程(curriculum + 合成增强)和评估工具(TTSD-eval),架构设计是已有技术的有效集成。在 spoken dialogue synthesis 这个新兴领域,开源 MOSS-TTSD 及 TTSD-eval 工具有实际价值。

## 可复用的 idea

1. **Voice clone + continuation 融合**: 将 reference-conditioned cloning(通过 chat template 中的参考 slot)与 autoregressive continuation(从前文续写)组合使用,在不增加模型复杂度的前提下显著提升 speaker similarity。可直接迁移到任何基于 LLM 的多说话人 TTS 系统。
2. **Forced-alignment 替代 diarization 做评估**: 当输入脚本已知说话人归属时,用 MMS-FA 做词级对齐直接获取 speaker 标签,避免 diarization 误差累积。TTSD-eval 的代码已开源,可直接用于其他 script-to-dialogue 系统的评估。
3. **三阶段 curriculum 策略**: 从单人→双人→多人的渐进训练,配合 DNSMOS 和采样率的逐步收紧,是扩展 TTS 到多说话人场景的实用训练方案。
4. **合成多说话人数据的拼接策略**: 对同一 speaker 的单人段做聚类,然后按规则交错拼接模拟多人对话。选择 DNSMOS ≥ 3.4 + 同采样率确保拼接自然度。解决了真实多人标注数据稀缺的问题。
5. **仅建模前 N 层 RVQ 以换取更长上下文**: 牺牲部分重建精度(只用 16 层而非全部层)来减少序列长度,使相同上下文窗口覆盖更长时间的音频。适用于需要长上下文但对音质要求可接受的场景。
