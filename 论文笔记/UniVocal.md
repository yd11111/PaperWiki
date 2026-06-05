---
type: paper
tier: deep
title: "UniVocal: Unified Speech-Singing Code-Switching Synthesis"
arxiv_id: "2606.01677"
source: "Sources/UniVocal.pdf"
authors: [Yufei Shi, Qian Chen, Wen Wang, Xiangang Li, Zhen-Hua Ling, Yang Ai]
year: 2026
venue: "arXiv"
tags: [TTS, SVS, code-switching, speech-singing, curriculum-learning, CoT, pitch-modeling, data-synthesis]
concepts: ["[[ConditionalFlowMatching]]", "[[SemanticvsAcousticTokens]]", "[[F0Modeling]]", "[[SingingVoiceSynthesis]]", "[[LLM-basedTTS]]", "[[ProsodyModeling]]", "[[Instruction-GuidedSpeechSynthesis]]"]
models: ["[[CosyVoice2]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: UniVocal 直接构建于 [[CosyVoice2]] (Du et al., 2024) 之上,属于 [[LLM-basedTTS]] 中 "Hybrid: LLM + Flow Matching" 路线的延伸。CosyVoice 2 的核心架构为: FSQ-SenseVoice semantic tokenizer → causal Transformer LM (0.5B) → chunk-aware flow matching → HiFi-GAN vocoder。UniVocal 在此基础上扩展到 speech + singing 的统一生成。

**已有认知 — Semantic tokens 的韵律缺陷**: [[SemanticvsAcousticTokens]] 页明确记录了 semantic tokens 的核心 trade-off: "captures linguistic content but discards acoustic details (pitch, timbre)"。pGSLM (Kharitonov et al., 2022) 和 SPIRIT-LM (Nguyen et al., 2024) 通过添加 paralinguistic tokens (F0, duration, style) 补充语义 token 的表现力。UniVocal 的 refined cent token 本质上是这一思路的新变体,但采用了高分辨率 (1200-bin) 的音高离散化方案。

**已有认知 — F0 建模**: [[F0Modeling]][待确认] 页梳理了 F0 在 TTS vs SVS 中的不同地位: TTS 中 F0 是韵律的柔性维度,SVS 中则是核心刚性约束。UniVocal 面临的挑战是需要同时处理两种模式的 F0 — 语音的自然韵律和歌声的精确旋律。

**已有认知 — SVS 现状**: [[SingingVoiceSynthesis]][待确认] 页记录了 SVS 领域的四大任务分类和架构范式。现有统一框架 (UniSyn, UniAudio) 仅能按 instruction 生成单一模式,无法在单条输出内自动切换 speech/singing。Bark 尝试混合生成但依赖显式 tag 且不稳定。UniVocal 定义的 SCS 任务是一个全新的子领域。

**已有认知 — CFM**: [[ConditionalFlowMatching]] 在 CosyVoice 系列中用于将 semantic tokens 渲染为高保真 Mel spectrogram。UniVocal 继承了这一架构,并在 flow matching 模块中增加了 cent token embedding 作为补充条件。

**创新判断**: UniVocal 的核心创新在于三点: (1) 定义 SCS 任务本身; (2) 用 1200-bin cent token + CoT interleaved generation 实现 "plan-then-generate" 的音高规划机制,这与 pGSLM 的 multi-stream 方案在机制上不同(interleaved AR vs 独立流); (3) 用 LLM 生成脚本 + stage-1 模型合成的数据管线解决 SCS 数据稀缺问题。

> 检索命中: [[CosyVoice2]]✓, [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[SingingVoiceSynthesis]](pending-review), [[F0Modeling]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 CosyVoice 2 基础上通过 refined cent token (1200-bin 音高离散化) + CoT interleaved generation + 两阶段 curriculum learning,实现从文本语义自动推断语音/歌声模式切换 (SCS)
> - **路线**: Text + Instruction → Causal Transformer LM (交替生成 cent token → semantic token) → Flow Matching (+ cent embedding 条件) → HiFi-GAN → Waveform
> - **指标**: SCSBench-Mixed F1(O) 0.871 / F1(S) 0.810 (超越级联 baseline); SeedTTS-EN UTMOS 4.21 (第一); Empathy E-MOS 2.26 接近 ElevenLabs 2.30; GTsinger WER 18.07 (第一) [Table 1-4]
> - **可借鉴**: (1) 用 modulo-1200 将 F0 投影到单八度 cent 空间,1 cent 量化误差 ~0.08% 频率偏差,兼顾语音韵律和歌声旋律; (2) CoT 式 interleaved token 生成让 LM 先 "plan" pitch 再生成 content; (3) LLM 生成语义脚本 + stage-1 模型合成音频的 SCS 数据构造 pipeline
> - **局限**: (1) 歌声训练数据依赖 Suno 合成歌曲,electronic tone 和歌词对齐问题限制了声学保真度上限; (2) 纯 implicit cue 场景切换成功率低 (Implicit-Only 部分失败); (3) 真实 SCS 场景泛化需要 explicit trigger (Real SCS F1 0.201 → Enhanced 0.730)

## 核心问题

UniVocal 要解决的核心问题是: 现有语音合成和歌声合成系统是高度特化的 — TTS 只能生成语音韵律、SVS 只能跟随乐谱、统一框架 (UniSyn, UniAudio) 只能根据 instruction 生成单一模式 — 没有系统能在单条输出内根据文本语义自动在语音和歌声之间切换 [§1]。这种 Speech-Singing Code-Switching (SCS) 在人类日常交流中是自然行为(对话中随口哼歌、叙事中插入旋律片段),但 AI 尚不具备 [§1]。

两个子问题: (1) 如何让模型仅从文本语义推断切换时机,不依赖显式 tag? (2) SCS 训练数据极度稀缺,如何构造有效的训练数据?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UniVocal 构建于 CosyVoice 2 框架之上 [§3],核心 backbone 是 24 层 causal Transformer (~0.5B 参数) 作为 text-to-vocal language model。输入为文本 (可选前缀 task instruction),输出为交替的 refined cent token 和 semantic token 序列 [§3.2, Fig 2]。下游使用修改后的 flow matching 模块 (增加 cent token embedding 条件) 生成 Mel spectrogram,再经预训练 HiFi-GAN vocoder 合成波形 [§3.2]。

三种任务通过 instruction 区分: TTS 无 instruction (沿用 CosyVoice 2 默认); SVS 使用风格描述 instruction; SCS 使用场景 instruction (如 "Generate a monologue.") [§3.2, Table 8]。关键在于: SCS instruction 仅定义全局场景,语音/歌声的细粒度切换由输入文本的语义内容自动驱动 [§3.2]。

### 关键设计选择

**1. Refined Cent Token — 为什么用 1200-bin 而不是 semitone/chromagram?**

[论文原文] 半音 (12-bin) 粒度满足音乐需求但对语音韵律过于粗糙; Vevo2 的 chromagram (12 semitone) 分辨率无法捕获自然语音的细粒度 F0 变化 [§2, §3.1]。cent scale 将每个半音细分为 100 cents (共 1200 cents = 1 octave),量化误差仅 1 cent (~0.08% 频率偏差),在感知上可忽略 [§3.1]。

具体编码: 将线性频率 $f_{Hz}$ 转为对数 cent 值 $f_{cent} = 1200 \times \log_2(f_{Hz}/440)$ (A4=440Hz 为参考),然后通过 $\mod 1200$ 投影到单八度范围,用 ceiling 操作离散化为整数 token。Unvoiced 区域赋值 -1 [§3.1, Eq. 1-2]。

[agent 解读] mod 1200 操作丢弃了绝对八度信息,这意味着 cent token 编码的是相对音高轮廓而非绝对频率。这是有意为之: (1) 对语音而言,韵律的关键是 pitch contour 的形状而非绝对值; (2) 对歌声而言,模型需要学习旋律模式,绝对音高可由 speaker embedding 和 flow matching 恢复。

消融实验验证了 1200-bin 是最优: 12-bin 在 empathetic speech (E-MOS 1.57) 上远弱于 1200-bin (1.85); 480-bin 也略弱 [Table 12, Appendix D.3]。

**2. Chain-of-Thought Interleaved Generation — 为什么先 pitch 后 content?**

[论文原文] 在每一帧 (25 Hz) 中,LM 先预测 cent token $c_t$ (音高信息),再以 $c_t$ 为条件预测 semantic token $s_t$ (语言内容) [§3.2, Eq. 3]。通过 logit masking 强制执行严格的交替顺序: cent 步骤 mask 掉 semantic vocabulary,反之亦然 [§3.2]。

[论文原文] 作者将此类比为 Chain-of-Thought: 模型先 "规划" prosodic/melodic 框架 (pitch contour),再在此框架下生成具体语言内容 [§3.2]。

[agent 解读] 这种 plan-then-generate 设计的直觉是: 韵律决策 (要不要唱、用什么音高模式) 应该先于内容决策。这与 pGSLM 的 multi-stream 预测 (同时但独立预测 F0/duration/semantic) 不同 — UniVocal 让 pitch 显式条件化 content,建立了从音高到内容的单向因果链。这可能是 empathetic speech 能力被 "解锁" 的原因: 模型被迫先为每一帧选择情感一致的音高轮廓,然后在此约束下生成 content。

关键配置: 论文提供两种推理模式: standard (不用 cent token,用于 SCS/TTS 等对齐要求高的任务) 和 expressive (含 CoT,用于 empathetic TTS/singing 等表现力任务) [§4.3.3]。

**3. Scalable Data Synthesis Pipeline — 为什么不直接收集真实 SCS 数据?**

[论文原文] 真实 SCS 数据极度稀缺,因此构造了 3 步合成 pipeline [§3.3, Appendix A.1]:
1. **语义文本生成**: 用 Gemini 2.5 Pro 生成 "boundary-blurring" 脚本 (独白/播客/有声书),包含两类切换线索 — implicit cue (语音与歌声的语义风格自然差异) 和 explicit cue (过渡短语如 "reminds me of a tune..."),约 50% 样本含 explicit cue [§3.3, Appendix A.1.2]
2. **统一声学合成**: 用 stage-1 对齐模型合成 speech 和 singing 片段,同一 speaker embedding 保证音色一致; speech 段额外条件化 emotion-specific reference audio 匹配文本情感 [§3.3]
3. **质量过滤**: WER ≥ 20% 的样本丢弃 (约 15%),10-20% WER 的保留但用 ASR 重转录文本 [Appendix A.1.2]

最终获得 11,769 样本 (262 小时),覆盖 3 场景 x 9 speaker timbres x 9 emotions [Table 7]。

[agent 解读] 这个 pipeline 的巧妙之处在于: 它不是简单拼接语音和歌声片段,而是通过 LLM 生成语义连贯的脚本,使切换自然嵌入叙事逻辑中。但缺点也很明显: 合成数据的分布与真实 SCS 有 gap (Real SCS F1 仅 0.201),模型在纯 implicit cue 场景下仍然困难 [Table 14, §5.3]。

**4. Two-Stage Curriculum Learning — 为什么不直接混合训练?**

[论文原文] 直接在 speech+singing+SCS 混合数据上训练 (w/o CL) 导致 SCS F1 仅 0.496 [Table 5],说明模型无法在一阶段内同时学会跨模式对齐和自动切换 [§5.2]。

- **Stage-1 (Latent Representation Alignment)**: 在 CosyVoice 2 基础上继续预训练,4:1 singing-to-speech 比例,使用 task-specific instruction 区分模式。目的: 在统一潜在空间中对齐 speech 和 singing 分布 [§3.4]。
- **Stage-2 (Autonomous Switching Learning)**: SFT 阶段,1:1:1 混合 (SCS:speech:singing) 防止灾难性遗忘,在 30,000 步内学习自动切换 [§3.4]。

训练资源: 4x NVIDIA A800,Stage-1 约 5 天,Stage-2 约 1 天 [Appendix B.2]。

### 训练策略

- **优化器**: AdamW, $\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1, gradient clipping 1.0 [Appendix B.1]
- **Stage-1 学习率**: 线性衰减 $2 \times 10^{-4} \to 0$, 70,000 步 (5,000 warmup) [Appendix B.1]
- **Stage-2 学习率**: 恒定 $1 \times 10^{-4}$, 30,000 步 [Appendix B.1]
- **数据**: Speech 960h LibriTTS (stage-1 全量, stage-2 200h); Singing 3,700h Suno (stage-1 全量, stage-2 200h + 10h GTSinger); SCS 262h 合成数据 (仅 stage-2) [§4.1]
- **Singing data cleaning**: 从 23,000h Suno → MelBand Roformer 源分离 → 60% 按 DNSMOS/SRMR 过滤 → dereverberation → VAD 分段 → FastWhisper 转写 + PPS 过滤 → 约 3,700h [Appendix A.2]

## 实验

| 指标 | 本文 (UniVocal) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| F1(O) | 0.871 | Gemini+Cosy2+LeVo: 0.607, Gemini+Bark: 0.465 | SCSBench-Mixed | [Table 1] |
| F1(S) | 0.810 | Gemini+Cosy2+LeVo: 0.566, Gemini+Bark: 0.199 | SCSBench-Mixed | [Table 1] |
| WER | 10.90 | Gemini+Cosy2+LeVo: 12.43, Gemini+Bark: 29.60 | SCSBench-Mixed | [Table 2] |
| SIM | 0.652 | Gemini+Cosy2+LeVo: 0.773 | SCSBench-Mixed | [Table 2] |
| UTMOS | 4.36 | Gemini+Cosy2+LeVo: 3.54, Gemini+Bark: 3.31 | SCSBench-Mixed | [Table 2] |
| WER (TTS) | 2.69 | F5-TTS: 2.15, CosyVoice 2: 2.96 | SeedTTS-EN | [Table 3a] |
| UTMOS (TTS) | 4.21 | CosyVoice 2: 4.18, F5-TTS: 3.68 | SeedTTS-EN | [Table 3a] |
| E-MOS | 2.26 | CosyVoice 2: 1.78, ElevenLabs: 2.30 | Empathy test | [Table 3b] |
| P-MOS | 2.22 | CosyVoice 2: 1.74, ElevenLabs: 2.47 | Empathy test | [Table 3b] |
| WER (SVS) | 18.07 | Vevo 1.5: 22.79, LeVo: 23.44 | GTsinger | [Table 4] |
| N-MOS (SVS) | 2.23 | Vevo 1.5: 2.17, LeVo: 2.41 | Fullsong | [Table 4] |
| M-MOS (SVS) | 2.18 | Vevo 1.5: 2.08, LeVo: 2.34 | Fullsong | [Table 4] |

**消融实验** [Table 5]:
- w/o CoT: SCS F1 0.810 (略升), E-MOS 2.03→1.78 的差距变小至 2.03, M-MOS 2.18→1.86 (显著下降) — CoT 对表现力任务至关重要
- w/o CL (无 curriculum learning): SCS F1 0.496 (大幅下降), WER 14.46 (最差) — 证明两阶段训练的必要性
- Empathy 能力来源: stage-1 的 emotionally diverse singing data 解锁表现力,CoT 进一步放大 [§5.2]

**Cent token 相关性验证** [Table 13]: 预测 cent token 与 GT cent token 的 SRCC 0.633-0.679, LCC 0.604-0.628,证明 cent token 确实在 "规划" 真实 pitch contour [Appendix D.4]。

**真实场景泛化** [Table 14]: Real SCS F1 仅 0.201; 手动插入 1 个 explicit cue 后 Enhanced Real SCS F1 跃升至 0.730 — 模型需要最小程度的语义锚点 [Appendix D.5]。

## 局限性

1. **歌声数据质量瓶颈**: 依赖 Suno 合成歌曲经源分离处理,存在 electronic tone 和歌词对齐问题,限制了歌声生成的声学保真度上限 [Limitations, §4.1]。SIM 指标略低于级联 baseline 可能与此相关 [§5.1]。

2. **Implicit cue 切换能力不足**: 模型在没有 explicit trigger phrase 的纯 implicit 场景下,无法可靠识别歌词式文本与叙事文本的语义差异 ("We were young and free..." 被误判为 prose) [§5.3, Table 6]。唯一的例外是 humming ("Mmm-hmm"),其独特的非词汇文本形式充当了 "strong implicit cue" [§5.3]。

3. **真实场景泛化 gap**: 合成训练数据与真实 SCS 存在分布差距,Real SCS F1 仅 0.201,必须添加 explicit trigger 才能泛化 (Enhanced 0.730) [Table 14, Appendix D.5]。

4. **评估精度有限**: Gemini 2.5 Pro 作为自动评估器,系统级排序与人类一致,但 sample-level 相关性仅中等 (Pearson r=0.343),因 F1 在短样本上常退化为二值 [Appendix C.2.4]。

5. **SIM 不足**: UniVocal 的全局 speaker similarity (SIM 0.652) 低于级联 baseline (0.773) [Table 2],但 intra-sample 一致性更优 [Fig 3] — 统一模型避免了跨模型音色漂移,但单体音色保真度仍有提升空间。

## 点评

UniVocal 的核心贡献不在于任何单一技术组件的创新深度,而在于 (1) 清晰定义了 SCS 这个新任务,(2) 提供了一个 end-to-end 的解决方案(数据 + 训练策略 + 模型 + benchmark),以及 (3) 证明了在 CosyVoice 2 这样的成熟 TTS backbone 上可以用相对低成本 (6 天 4xA800) 扩展出新的生成能力。

refined cent token + CoT interleaved generation 是本文最有启发性的设计。它不是简单地给 semantic token 补充 pitch 信息(如 pGSLM 的 multi-stream),而是建立了 pitch → content 的因果生成链 — 模型被迫先做音高规划,这个 "思考" 步骤不仅改善了歌声旋律,还 "附带" 解锁了 empathetic speech 能力 (E-MOS +0.48 vs CosyVoice 2 baseline)。这种 "结构约束带来意外收益" 的现象值得注意。

但论文也暴露了数据驱动方法的根本局限: SCS 能力高度依赖训练数据的分布覆盖。纯 implicit cue 的切换仍是未解决的挑战 — 模型学到的更多是 explicit trigger phrase 的模式匹配,而非真正的语义理解。Real SCS F1 从 0.201 到 Enhanced 0.730 的跳跃说明模型需要 "拐杖" 才能在分布外场景工作。

另一个值得关注的局限是 mod 1200 操作丢弃了绝对八度信息。对于语音韵律这是合理的,但对于歌声的旋律准确性(如八度跳跃),这种信息损失的影响论文未充分讨论。

## 可复用的 idea

1. **Cent token 作为通用 pitch 离散化方案**: 1200-bin cent scale (mod 1200) 提供了一种统一处理语音 F0 和歌声 pitch 的离散表示。0.08% 频率偏差的量化误差使其适用于任何需要高精度 pitch 建模的场景。可以考虑将此方案引入情感 TTS 或对话 TTS 中的 prosody planning。

2. **Interleaved CoT generation for prosody**: 在 LM 词汇表中同时包含 pitch token 和 content token,通过 logit masking 强制交替生成,实现 "先规划韵律,再填充内容" 的 plan-then-generate 范式。这种机制比 multi-stream 更简单 (无需多个预测头),且天然建立了 pitch → content 的因果关系。

3. **LLM-driven SCS data synthesis pipeline**: 用大模型生成语义连贯的 "boundary-blurring" 脚本 + 用 stage-1 模型合成音频 + WER 过滤的三步 pipeline,可推广到任何缺乏自然数据的跨模式任务 (如语音中插入音效、对话中插入 whisper 等)。

4. **Two-stage curriculum for cross-domain unification**: 先在混合数据上对齐表示空间 (stage-1: representation alignment),再用任务数据学习新能力 (stage-2: capability learning)。这种 "先共存后协作" 的训练策略可迁移到任何需要在已有模型上扩展新模态/任务的场景。
