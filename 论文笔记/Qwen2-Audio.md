---
type: paper
tier: deep
title: "Qwen2-Audio Technical Report"
arxiv_id: "2407.10759"
source: "Sources/Qwen2-Audio.pdf"
authors: [Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, Jingren Zhou]
year: 2024
venue: "arXiv"
tags: [speech-LM, audio-understanding, multimodal, LALM, instruction-tuning, DPO, natural-language-prompt, voice-chat, audio-analysis, Whisper]
concepts: ["[[概念库/SpeechLanguageModel|Speech Language Model]]", "[[概念库/AudioUnderstanding|Audio Understanding]]", "[[概念库/ModalityAdaptationforSpeechLLM|Modality Adaptation]]", "[[概念库/Audio-LanguagePretraining|Audio-Language Pretraining]]", "[[概念库/Speech-LLMIntegrationTaxonomy|Speech-LLM Integration Taxonomy]]"]
models: ["Qwen2-Audio", "Qwen-Audio", "Whisper-large-v3", "Qwen-7B", "SALMONN", "Gemini-1.5-pro"]
tasks: ["ASR", "S2TT", "SER", "VSC", "audio-captioning", "instruction-following"]
datasets: ["LibriSpeech", "Aishell2", "Fleurs", "CommonVoice15", "CoVoST2", "Meld", "VocalSound", "AIR-Bench"]
kb_context_sources: 4
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Qwen2-Audio 属于 latent-representation-based Speech-LLM integration 路线 [Speech-LLMIntegrationTaxonomy],使用 Whisper-large-v3 编码器产出的连续表征直接送入 LLM。在 KB 的 ALM 架构分类中属于 "Two Heads" 变体 (单一 Audio encoder + LM),但不同于 SALMONN 的双编码器设计 -- Qwen2-Audio 选择用单一但更强的编码器 (Whisper-large-v3 替代 Qwen-Audio 的 Whisper-large-v2) 覆盖语音和非语音理解。
>
> **同期对比**: KB 已有 SALMONN (ICLR 2024, dual Whisper+BEATs encoder, window-level Q-Former, activation tuning) 的精读笔记。两者同属 2023-2024 年 Audio-LLM 早期工作,但设计哲学不同: SALMONN 强调编码器互补性和涌现能力激活,Qwen2-Audio 强调训练流程简化和交互模式统一。Qwen2-Audio 的后代 Qwen2.5-Omni (2025) 进一步演化为 Thinker-Talker 架构,增加了语音生成能力。
>
> **适配方法**: Qwen2-Audio 使用 pooling layer (stride=2) 作为 modality adaptation [ModalityAdaptationforSpeechLLM],属于最简单的 convolutional downsampling 类型。这与 SALMONN 的 Q-Former (性能更优但更复杂) 形成对比。Qwen2-Audio 选择简单适配器可能因为其编码器本身已足够强 (Whisper-large-v3 含 1.55B 参数),不需要复杂的表征转换。
>
> 检索命中: [[SpeechLanguageModel]]confirmed, [[AudioUnderstanding]]pending, [[ModalityAdaptationforSpeechLLM]]pending, [[Audio-LanguagePretraining]]pending | 过滤: 无 | 未命中但可能相关: [[Speech-LLMIntegrationTaxonomy]]

## 速查

> [!summary] 速查
> - **一句话**: Qwen-Audio 的升级版,用自然语言 prompt 替代层次标签简化预训练,并通过 SFT+DPO 三阶段训练实现 voice chat / audio analysis 双模式无缝交互,在 AIR-Bench 上超越 Gemini-1.5-pro
> - **路线**: Audio (16kHz) → 128-ch mel-spectrogram → Whisper-large-v3 encoder (冻结→微调) → Pooling (stride=2, ~40ms/frame) → Qwen-7B LLM → text response
> - **指标**: LibriSpeech WER 1.6/3.6 (clean/other) [Table 2] | AIR-Bench chat 7.18/6.99/6.79/6.77 (speech/sound/music/mixed, GPT-4 eval) [Table 2] | CoVoST2 avg BLEU 35.6 (7 directions) [Table 2] | VocalSound ACC 93.92% [Table 2]
> - **可借鉴**: (1) 用自然语言 prompt 替代 hierarchical tags 做多任务预训练,降低预训练-微调 gap; (2) voice chat + audio analysis 双模式联合训练,无需 system prompt 切换; (3) DPO 优化 factuality 和 behavior adherence
> - **局限**: 仅输出文本不输出语音; 技术报告内容较短 (7 页正文),架构和训练细节披露不足; 未公布预训练数据具体来源和规模; DPO 数据集未公开细节

## 核心问题

**想解决什么**: Qwen-Audio (2023) 虽然首次实现了多任务通用音频理解,但存在两个问题: (1) 预训练使用复杂的层次标签 (hierarchical tags) 标注不同任务和数据,导致预训练与下游使用之间存在格式 gap; (2) 模型只能被动分析音频,缺乏主动语音交互能力 [§1]。

**为什么难**: 同一段音频可能同时包含环境声、多人对话和语音指令,模型需要**不依赖 system prompt** 就能自动判断应该做声音分析还是执行语音指令。这意味着模型必须理解音频的内在结构,而不是依赖外部的任务标识 [§2]。

**怎么切入**: 三个层面的简化与增强 -- (1) 预训练阶段用自然语言 prompt 替代 hierarchical tags; (2) SFT 阶段联合训练 voice chat 和 audio analysis 两种交互模式; (3) DPO 阶段对齐人类偏好 [§2, Fig 2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Qwen2-Audio 由两个主要组件构成 [§2, Fig 2]:

1. **Audio Encoder**: 基于 Whisper-large-v3 (从 Qwen-Audio 的 Whisper-large-v2 升级)。输入 16kHz 音频,转为 128-channel mel-spectrogram (25ms window, 10ms hop),通过编码器输出帧级表征。额外加入 stride=2 的 pooling layer,使每个输出帧对应约 40ms 原始音频 [§2]。

2. **Large Language Model**: Qwen-7B,接收 audio token + text token 混合序列,自回归生成文本。总参数量 8.2B [§2]。

训练目标是标准的 next-token prediction [Eq. 1]:
```
P_θ(x_t | x_{<t}, Encoder_φ(a))
```
其中 θ 和 φ 分别是 LLM 和 audio encoder 的参数,两者均可训练 [§2]。

### 关键设计选择

**为什么用自然语言 prompt 替代 hierarchical tags?** [论文原文] Qwen-Audio 使用复杂的 hierarchical tags 标注任务类型 (如 `<|ASR|>`, `<|AAC|>` 等),这在预训练阶段有效但造成了预训练与 SFT/实际使用之间的格式断裂。自然语言 prompt (如 "Detect the language and recognize the speech:") 不仅消除了这个 gap,还发现能改善模型的泛化能力和指令遵循能力 [§2]。[agent 解读] 这本质上是将任务标识从 special tokens 转为 in-context instruction,使预训练阶段的输入格式与 SFT 阶段一致,减少了分布偏移。

**为什么只用单一编码器而不是 SALMONN 式双编码器?** [agent 解读] 论文未直接讨论此选择,但 Qwen2-Audio 从 Whisper-large-v2 升级到 Whisper-large-v3,后者在更多语言和更大规模数据上训练,本身已覆盖更广泛的音频类型。单编码器的优势是架构简洁、无需对齐两个编码器的帧率和表征空间。在 VocalSound (非语音任务) 上 93.92% 的准确率 [Table 2] 也说明 Whisper-large-v3 的编码能力足以覆盖语音以外的声音。

**为什么 voice chat 和 audio analysis 可以不用 system prompt 切换?** [论文原文] 两种模式在 SFT 阶段联合训练,模型学会了根据音频内容和指令形式自动判断交互模式。例如,同一段音频中包含键盘打字声和用户提问 "What is this sound?",模型能自动识别语音部分是指令、键盘声是分析对象 [§2]。[agent 解读] 这种能力的前提是 SFT 数据中包含了足够多的混合场景样本,使模型学会区分"音频中的指令"和"待分析的音频内容"。

### 训练策略

三阶段训练 [§2, Fig 2]:

**Stage 1 -- Multi-task Pre-training**: 使用自然语言 prompt 标注不同任务,在大规模音频-文本配对数据上训练。覆盖 ASR、AAC (Audio Captioning) 等任务。相比 Qwen-Audio 进一步扩大了数据量 (具体规模未披露,仅展示了 Fig 3 的数据分布饼图,按小时统计) [§2]。

**Stage 2 -- Supervised Fine-tuning (SFT)**: 使用高质量 instruction-following 数据微调。[论文原文] 前期研究强调 SFT 数据的质量和复杂度对模型性能有关键影响,因此实施了严格的质量控制 [§2]。两种交互模式 (audio analysis + voice chat) 联合训练:
- **Audio Analysis**: 用户提供音频 + 文本/语音指令,模型分析音频内容。适用于离线场景 [§2]。
- **Voice Chat**: 用户通过语音自由对话,模型作为语音助手。适用于在线交互场景 [§2]。

**Stage 3 -- Direct Preference Optimization (DPO)**: 使用人工标注的 (input, good_response, bad_response) 三元组,通过 DPO 损失 [Eq. 2] 优化模型对齐人类偏好,重点改善 factuality 和 behavior adherence [§2]。

### 与 Qwen-Audio 的关键差异

| 维度 | Qwen-Audio | Qwen2-Audio |
|------|-----------|-------------|
| 编码器 | Whisper-large-v2 | Whisper-large-v3 |
| 预训练策略 | Hierarchical tags | Natural language prompts |
| 训练阶段 | Pre-training + SFT | Pre-training + SFT + DPO |
| 交互模式 | 仅 audio analysis | Audio analysis + Voice chat |
| 模式切换 | N/A | 无需 system prompt,自动识别 |

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER | 1.6 / 3.6 | Qwen-Audio: 2.0/4.2, Whisper-large-v3: - | LibriSpeech test-clean/other | [Table 2] |
| WER | 3.0/3.0/2.9 | Qwen-Audio: 3.0/3.0/2.9, Paraformer-large: 2.9/3.3/3.1/3.3 | Aishell2 Mic/iOS/Android | [Table 2] |
| WER (zero-shot) | 7.5 | Whisper-large-v3: 7.7 | Fleurs zh | [Table 2] |
| WER | 8.6/6.9/5.9/9.6 | Whisper-large-v3: 9.3/12.8/10.9/10.8 | Common Voice 15 en/zh/yue/fr | [Table 2] |
| BLEU (avg 7 dirs) | ~35.6 | Qwen-Audio: ~33.0, SpeechLLaMA: ~23.2 | CoVoST2 | [Table 2] |
| SER ACC | 0.553 | Qwen-Audio: 0.557, WavLM: 0.542 | Meld | [Table 2] |
| VSC ACC | 0.9392 | Qwen-Audio: 0.9289, Pengi: 0.6035 | VocalSound | [Table 2] |
| AIR-Bench chat (GPT-4) | 7.18/6.99/6.79/6.77 | Gemini-1.5-pro: 6.97/5.49/5.06/5.27, Qwen-Audio: 6.47/6.95/5.52/6.08 | Speech/Sound/Music/Mixed | [Table 2] |

**关键观察**:

1. **ASR 性能稳步提升**: LibriSpeech test-clean WER 从 Qwen-Audio 的 2.0 降至 1.6,test-other 从 4.2 降至 3.6 [Table 2]。在 Common Voice 15 上全面超越 Whisper-large-v3 (需注意 Qwen2-Audio 在 Common Voice 上非 zero-shot,而 Whisper 是 zero-shot) [Table 2]。

2. **翻译能力显著提升**: CoVoST2 平均 BLEU 从 Qwen-Audio 约 33.0 提升至约 35.6,多个翻译方向均有改善 [Table 2]。

3. **指令遵循能力是最大亮点**: AIR-Bench 四个维度全面超越 Gemini-1.5-pro 和 Qwen-Audio。Speech 维度 7.18 vs Gemini 6.97 (+3.0%),Music 维度 6.79 vs Gemini 5.06 (+34.2%),Mixed 维度 6.77 vs Gemini 5.27 (+28.5%) [Table 2]。[agent 解读] Music 和 Mixed 维度的巨大提升说明自然语言 prompt 预训练 + DPO 显著增强了模型处理复杂混合音频的能力。

4. **SER 小幅下降**: Meld 情感识别从 0.557 (Qwen-Audio) 略降至 0.553 [Table 2]。[agent 解读] 这可能是多任务扩展的 trade-off,但降幅极小,不影响整体判断。

5. **Gemini-1.5-pro 的 SAFETY 问题**: 报告指出 Gemini-1.5 因 SAFETY 限制拒绝返回约 1/5 的 AIR-Bench 测试样本,导致其实际样本数减少 [§3.2]。这意味着 Gemini 的得分基于更少的样本,对比可能不完全公平。

## 局限性

1. **仅理解不生成**: 与 SALMONN 一样,Qwen2-Audio 只输出文本,不输出语音。后续 Qwen2.5-Omni 才通过 Thinker-Talker 架构增加了语音生成能力。

2. **技术细节披露不足**: 作为 "Technical Report" 仅 7 页正文,未提供: (a) 预训练数据的具体来源列表和总规模,仅有 Fig 3 饼图; (b) SFT 数据的详细组成; (c) DPO 数据的构建方法和规模; (d) audio encoder 和 LLM 各阶段的冻结/微调策略。

3. **Pooling layer 设计未充分论证**: 使用 stride=2 pooling 作为 modality adaptation 是最简单的方案,论文未讨论为什么不用更高效的 Q-Former 或 CTC compression (已有文献表明 Q-Former > Conv > simple pooling [Yu et al., 2024])。

4. **评估局限**: (a) Common Voice 15 非 zero-shot,而 Whisper 是 zero-shot,对比不完全公平 [Table 2 注释]; (b) AIR-Bench 使用 GPT-4 自动评估,可能引入评估偏差; (c) 缺少 audio captioning (AAC) 的独立评估。

5. **模式切换的鲁棒性未验证**: 论文声称不需要 system prompt 切换模式,但仅通过案例展示,缺少系统性的消融实验证明模型何时会误判模式。

## 点评

**工程进步大于方法论创新**: Qwen2-Audio 的核心贡献是工程层面的 -- 更强的编码器、更大的数据、更规范的训练流程 (自然语言 prompt + DPO)。相比 SALMONN 的 activation tuning 和 dual encoder 等具有方法论创新性的设计,Qwen2-Audio 更像是一次扎实的系统升级。

**自然语言 prompt 替代 hierarchical tags 的洞察值得重视**: 虽然看似简单,但这个设计选择消除了预训练-微调的格式断裂,使模型在预训练阶段就开始学习指令遵循。这与 NLP 领域 "pre-training with instructions" 的趋势一致,是将 LLM 最佳实践迁移到 audio-language 模型的成功案例。

**单编码器 vs 双编码器的路线选择**: 与 SALMONN 选择 Whisper+BEATs 双编码器不同,Qwen2-Audio 选择升级单一编码器 (v2→v3)。结果表明,足够强的单编码器可以同时覆盖语音和非语音任务 (VocalSound 93.92%),验证了简洁架构的可行性。这也解释了为什么后续主流方向 (Gemini, GPT-4o) 倾向于单编码器设计。

**Qwen 音频系列演进**: Qwen-Audio (2023, hierarchical tags) → Qwen2-Audio (2024, natural language prompt + DPO) → Qwen2.5-Omni (2025, Thinker-Talker + speech generation)。每代解决前代的一个核心缺陷: 第一代建立基础能力,第二代提升指令遵循和交互灵活性,第三代增加语音输出闭环。

## 可复用的 idea

1. **自然语言 prompt 替代 special token 做多任务预训练**: 在任何多任务预训练场景中,用自然语言描述任务 (如 "Detect the language and recognize the speech:") 替代 special token (如 `<|ASR|>`),可以减少预训练与下游使用之间的格式偏移,且让模型在预训练阶段就开始学习指令遵循。适用于任何 encoder-LLM 架构的多任务系统。

2. **双模式联合训练 (voice chat + audio analysis)**: 当构建语音交互系统时,将"被动分析"和"主动对话"两种场景的数据混合训练,使模型自动学习模式判断,而非依赖 system prompt 或外部路由。关键前提是训练数据中需要包含足够多的混合/歧义场景。

3. **DPO 用于音频模型的 factuality 优化**: DPO 不仅适用于文本 LLM,在 audio-language 模型中同样有效。可构造 (audio_input, 高质量response, 低质量response) 三元组来改善模型的事实准确性和指令遵循度。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节解释了三个关键设计选择的 WHY,可借鉴项具体可迁移 |
> | 可信赖 | pass | 数字标注覆盖率>90%,所有指标均标注出处 [Table 2],指标方向无误 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率>80%,推断均明确标记 |
> | 可定位 | pass | KB 背景与 SALMONN 对比清晰,谱系定位 (Qwen-Audio→Qwen2-Audio→Qwen2.5-Omni) 准确 |
> | 不污染 | pass | 无反向更新,frontmatter 概念/模型挂接合理 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/Qwen2-Audio-review.yml`
