---
type: paper
tier: deep
title: "Raon-Speech Technical Report"
arxiv_id: "2605.23912"
source: "Sources/Raon-Speech.pdf"
authors: [KRAFTON AI]
year: 2026
venue: "arXiv"
tags: [speech-LM, full-duplex, bilingual, SpeechLM, turn-taking, knowledge-distillation, preference-optimization, English-Korean]
concepts: ["[[SpeechLanguageModel]]", "[[Full-duplexSpokenDialogue]]", "[[ModalityAdaptationforSpeechLLM]]", "[[Turn-takinginSpokenDialogue]]", "[[StreamingSpokenDialogue]]", "[[ResidualVectorQuantization]]", "[[SemanticvsAcousticTokens]]", "[[SpeakerEmbedding]]", "[[SpokenDialogueEvaluation]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: ["[[Moshi]]", "[[EnCodec]]", "[[Whisper]]"]
tasks: []
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: reviewed
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeechLanguageModel]], [[Full-duplexSpokenDialogue]], [[ModalityAdaptationforSpeechLLM]], [[Turn-takinginSpokenDialogue]], [[StreamingSpokenDialogue]], [[Speech-LLMIntegrationTaxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Raon-Speech 属于 latent-representation + audio-token 混合集成路线的 SpeechLM: 输入侧用连续表征 (speech encoder + MLP adaptor 映射到 LLM embedding space,对应 [[ModalityAdaptationforSpeechLLM]] 中的 convolutional downsampling 类方法),输出侧用 RVQ codec tokens (Mimi, 1 semantic + 15 acoustic,对应 [[SemanticvsAcousticTokens]] 的 mixed tokenizer 范式)。这一输入连续-输出离散的设计与 Qwen2-Audio、SALMONN 等 latent-representation 系统和 Moshi、AudioLM 等 audio-token 系统均不同,是一种折衷方案。
>
> **全双工定位**: Raon-SpeechChat 在 [[Full-duplexSpokenDialogue]] 的演进中承接 Moshi (2024) 和 PersonaPlex (2026) 的路线,但做了三点区别: (1) 用单一自回归序列交错三种模态 (user speech / assistant text / assistant speech),而非 Moshi 的并行双流; (2) 引入 SIL/BOW/BC 显式状态 token 分离"何时说"与"说什么",而 Moshi 用单一 PAD token; (3) 引入 text lookahead 让文本先于语音生成,减少语义漂移。
>
> **已有认知**: KB 中 [[SpeechLanguageModel]] 页面已记录 SpeechLM 从 GSLM → Moshi 的演进; [[Turn-takinginSpokenDialogue]] 页面覆盖了 Freeze-Omni 的 chunk-level state prediction (State 0/1/2) 和 Mini-Omni2 的 irq/n-irq 标记,但尚无 SIL/BOW/BC 三状态方案; [[StreamingSpokenDialogue]] 页面记录了 causal encoder 和 RQ-Transformer 等流式架构,Raon-SpeechChat 的 Voxtral causal encoder 替换方案是新增案例。
>
> **创新判断**: 与 KB 已有知识对比,Raon-Speech 的主要新增贡献在 (1) on-policy self-distillation 的 KD 训练方案 (text-conditioned teacher 促进 speech-text 对齐 + backbone KD 防遗忘); (2) SimPO 偏好优化用于 SpeechLM post-training; (3) 42 benchmark 的大规模英韩双语评估 (新建 KVoiceBench/KOpenAudioBench/KMMAU 三个韩语基准); (4) SIL/BOW/BC 状态建模分离交互行为与语言内容。
>
> 检索命中: [[SpeechLanguageModel]]✓ | 过滤: [[Full-duplexSpokenDialogue]][待确认], [[ModalityAdaptationforSpeechLLM]][待确认], [[Turn-takinginSpokenDialogue]][待确认], [[StreamingSpokenDialogue]][待确认], [[Speech-LLMIntegrationTaxonomy]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 9B 英韩双语 SpeechLM + 全双工扩展,通过三阶段训练 (alignment → KD pre-training → SimPO post-training) 在 42 个 benchmark 上建立最强 speech-centric profile,全双工版通过 SIL/BOW/BC 状态建模实现自然实时对话
> - **路线**: speech → AuT/Voxtral encoder → MLP adaptor → Qwen3-VL-8B backbone → Speech Generation Expert (semantic token) → RCP (15 acoustic tokens) → Mimi codec decoder → waveform; 全双工: 单序列交错 user speech + assistant text + assistant speech
> - **指标**: VoiceBench avg 76.79 (best), MMAU 78.68 (best), MMLU-Pro 64.05 (best); 韩语全面领先; FDB v1.0 interruption TOR 0.980 (best), backchannel TOR 0.091 (best) [Table 4, 5, 6]
> - **可借鉴**: (1) on-policy self-distillation: text-conditioned model 做 teacher 促进 speech-text 对齐,同时用冻结 backbone 做 text KD 防遗忘; (2) SIL/BOW/BC 三 token 状态建模,将"何时说"与"说什么"解耦; (3) RMSNorm (init scale 0.02) 稳定 adaptor-LLM 对齐
> - **局限**: (1) UTMOS 感知自然度不是最优 (Qwen2.5-Omni 的 UTMOS 更高) [Table 4]; (2) FDB v2.0 多轮长对话表现弱于 PersonaPlex 和 MiniCPM-o 4.5 [Table 6]; (3) 仅支持英韩双语; (4) 1.38M 小时数据中依赖大量 in-house 数据,复现难度高

## 核心问题

Raon-Speech 要回答两个层次的问题:

1. **如何在 <10B 参数规模下构建高质量双语 SpeechLM?** 现有轻量模型在英语之外的语言 (尤其韩语) 上表现不佳,且在获得语音能力的同时往往遗忘文本能力 [§1]。
2. **如何从半双工 SpeechLM 扩展到全双工实时对话?** 现有全双工模型在时间感知和交互自然度上仍有不足,尤其是需要精细实时通信的场景 (如游戏) [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

**Raon-Speech** (9.1B) 在 Qwen3-VL-8B-Instruct backbone 上扩展语音理解和生成模块 [§2.1, Fig 2]:

**输入侧 (语音理解)**:
- Speech encoder: AuT 预训练模型,12.5 Hz token rate [§2.1]
- Input adaptor: 2-layer MLP (GELU), 2048→4096,后接 RMSNorm (init weight=0.02) [§2.1]
- 论文原文解释: RMSNorm 的小初始化 scale 使语音 embedding 的 norm 在初始化时匹配 LLM embedding 的 norm,稳定对齐训练 [§2.1]

**输出侧 (语音生成)**:
- Output adaptor: 与 input adaptor 同架构,将上一步 codec embedding 映射到 LLM 输入空间 [§2.1]
- Speech Generation Expert: 4-layer decoder-only Transformer (hidden 2048, FFN 6144), 从 LLM hidden states 预测 semantic token [§2.1, Table 8]
- RCP (Residual Code Predictor): 5-layer, hidden 1024, 从 Qwen3-Omni-30B-A3B 初始化,预测 15 个 acoustic token (残差式逐层预测) [§2.1]
- Speech codec: Mimi (RVQ-based, 12.5 Hz, 32 codebook 取前 16), causal sliding window attention (10s window) [§2.1, Table 8]
- Speaker encoder: ECAPA-TDNN (冻结, 从 speechbrain 初始化), 192→4096 线性投影 [§2.1]

[agent 解读] 这种"LLM 只负责 semantic token, 小模型负责 acoustic tokens"的分工设计平衡了语义质量和生成效率 -- LLM 的 hidden states 包含足够语义信息,但让 8B 模型直接预测 16 层 codec tokens 计算代价过高,因此用专门的小模型 (SGE 205M + RCP 146M) 完成从语义到声学的映射。

**Raon-SpeechChat** (9.8B) 在 Raon-Speech 基础上做三项修改 [§2.2, Fig 3]:

1. **Causal encoder 替换**: AuT (非因果, full-context attention) → Voxtral-Mini-4B-Realtime (因果, sliding window 15s), 支持流式输入 [§2.2]
2. **Token 序列交错**: 单一自回归序列中交错 user speech / assistant text / assistant speech, 用 word-level alignment 对齐文本和语音 [§2.2]
3. **状态建模**: 三种特殊 token [§2.2]:
   - **SIL** (silence): 显式编码沉默监听行为,预测 SIL = 保持沉默
   - **BOW** (beginning of word): 在每个 assistant text token 前发出,标记"即将说一个新词",分离 when-to-speak 和 what-to-say
   - **BC** (backchannel): 专用于回传信号 (区别于 BOW),支持推理时独立控制回传频率
   - [论文原文] 对比 Moshi 的设计: Moshi 用单一 PAD token 同时表示沉默和非文本位置,Raon-SpeechChat 用 SIL 专门编码沉默 [§2.2]

4. **Text lookahead**: 助手开始说话后,text 先于 speech 生成,减少语义漂移 [§2.2]。[论文原文] 在全双工场景下尤其重要,因为语音必须在严格延迟约束下增量生成 [§2.2]

### 关键设计选择

**为什么选 Qwen3-VL-8B 做 backbone?** 论文原文: "strong multilingual text capabilities" [§2.1]。[agent 解读] 选择 VL 变体而非纯 text 版本可能是因为 VL 预训练已经建立了跨模态 (vision-text) 的对齐能力,有助于进一步扩展到 speech 模态。

**为什么用 Mimi 而非 EnCodec?** [agent 解读] Mimi 是 streaming-native 设计 (causal encoder/decoder + sliding window attention),与全双工实时生成的需求天然匹配。Mimi 也是 mixed tokenizer (first codebook = semantic, rest = acoustic),在 [[SemanticvsAcousticTokens]] 中代表了兼顾语义和声学的路线。

**为什么 RCP 从 Qwen3-Omni-30B 初始化?** 论文原文: "to accelerate convergence" [§2.1]。[agent 解读] RCP 需要学习从 semantic token 到 15 层 acoustic tokens 的映射,这个任务与 Qwen3-Omni 中同类模块的功能一致,迁移参数可以显著减少训练成本。

**为什么用 on-policy self-distillation?** [§3.1] 两个 teacher 服务不同目的:
- Audio 输入: teacher = 同一模型以 text transcript 为条件 → 显式鼓励 speech-text 表征对齐 [§3.1, 引用 Hu et al., 2026b; Wang et al., 2025]
- Text 输入: teacher = 预训练前的 backbone LLM → 缓解灾难性遗忘 [§3.1]
- [论文原文] KL loss 基于 student 自身生成的 trajectory (on-policy) 计算,论文发现这种方式在获取新语音能力和减少遗忘上均优于 off-policy KD [§3.1, 引用 Agarwal et al., 2024]

**为什么用 SimPO 而非 DPO?** 论文原文: SimPO "eliminates the need for a separate reference policy",提升计算和内存效率 [§3.1]。

**为什么 SIL/BOW/BC 要分开?** [论文原文] (1) SIL 与 PAD 分离: 显式编码沉默监听,让模型区分"主动沉默"和"正在说话中的填充"; (2) BOW 与 text token 分离: 解耦交互行为 (何时说) 和语言内容 (说什么),让模型通过不同 token 类型分别学习; (3) BC 与 BOW 分离: 让模型区分短回传反馈和正式轮次开始,推理时可独立控制回传频率甚至完全禁用 [§2.2]

### 训练策略

**Raon-Speech: 4 阶段** [§3.1, Table 2]

| 阶段 | 可训练模块 | 冻结模块 | 数据/任务 | 目标 | 步数 |
|------|-----------|---------|-----------|------|------|
| Understanding alignment | Input adaptor | Backbone + encoder + 其他 | STT, SpeechQA, SpokenQA | CE | 13.5k |
| Generation alignment | Output adaptor + SGE + RCP | Backbone + codec | TTS | CE | 65.5k |
| E2E pre-training w/ KD | 除 encoder, codec, speaker encoder 外全部 | Encoder + codec + speaker encoder | 全部 5 任务 | CE + KL (on-policy KD) | 60k |
| Post-training | 同上 | 同上 | Curated SFT (全部任务) | CE + SimPO | 800 |

关键细节:
- Understanding alignment 使用 on-the-fly 音频增强 (noise, reverb, channel distortion, bandwidth degradation) [§3.1]
- 输入音频分割为 8s chunks [§3.1]
- TextQA 的 ground-truth 由 backbone LLM 自身生成,确保与原始输出分布一致 [§3.1]
- Speaker embedding dropout 0.2 [§3.1]

**Raon-SpeechChat: 4 阶段 (从 Raon-Speech 继续训练)** [§3.2, Table 2]

| 阶段 | 核心改动 | 数据 | 步数 |
|------|---------|------|------|
| Causal alignment | 替换 encoder, 两步: adaptor-only → 全模型 | 同 Raon-Speech | 10k + 10k |
| Full-duplex pre-training | 全双工交错数据, text lookahead, SIL loss weight 0.5, PAD loss weight 0.75 | 全双工 + 10% Raon-Speech 数据 | 25k |
| Full-duplex fine-tuning I | 高质量对话数据 | High-quality conversation | 5k |
| Full-duplex fine-tuning II | BOW→BC 替换, BC CE loss ×50 | Synthetic + scenario-specific | 5k |

[论文原文] Fine-tuning II 中将 BC embedding 从 BOW embedding 初始化,同时将 BC 的 CE loss 权重提高 50 倍以缓解标签不平衡 [§3.2]。

### 数据

总量: 1.38M 小时英韩语音+文本 [§4]

**SpeechLM 数据** [§4.1]:
- Audio-text paired: 公开 + in-house
- Audio-only: 公开语料 + 网络音频, 用 Whisper 生成伪标签转写
- Text-only: 阅读理解/常识推理/指令跟随, 用 Qwen3-TTS 合成语音或直接作为 TextQA

数据预处理 [§4.1]: 标准化 (神经标点恢复) → 过滤 (STT 错误率/强制对齐/音频质量评分) → 重标注 (Whisper 重转写 + LLM 精炼 QA) → 再平衡 (跨域/跨任务/跨格式)

**全双工数据** [§4.2]:
- Real conversation: 13.21K 小时 (公开 + in-house)
- Synthetic conversation: 106.33K 小时, 四阶段管线 [§4.2, Appendix D]:
  1. Dialogue generation: Qwen3 LLM 生成对话
  2. Timeline construction + Qwen3-TTS 合成
  3. Timing refinement: 音频回传预测模型 + 规则
  4. Barge-in text truncation: 随机截断 + 强制对齐

## 实验

### Raon-Speech: 英语 benchmark [Table 4]

| 指标 | Raon-Speech | 最强 baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ASR WER | 1.44 | 1.38 (Kimi-Audio) | LibriSpeech-clean | [Table 4] |
| ASR WER | 2.89 | 2.70 (Kimi-Audio) | LibriSpeech-other | [Table 4] |
| Speech Gen WER | 2.01 | 2.30 (Qwen2.5-Omni) | LibriSpeech-clean | [Table 4] |
| Speech Gen WER | 1.93 | 3.42 (HyperCLOVA X) | Seed | [Table 4] |
| Speech Gen UTMOS | 3.26 | 3.83 (Step-Audio 2) | LibriSpeech-clean | [Table 4] |
| VoiceBench avg | 76.79 | 76.06 (Fun-Audio-Chat) | VoiceBench | [Table 4] |
| OpenAudioBench | 70.21 | 74.82 (MiniCPM-o 4.5) | OpenAudioBench | [Table 4, 9] |
| MMAU (Speech) | 78.68 | 77.18 (Qwen2.5-Omni) | MMAU test-mini | [Table 4] |
| MMAU-Pro | 64.65 | 62.74 (Qwen2.5-Omni) | MMAU-Pro | [Table 4] |
| MMLU-Pro | 64.05 | 61.12 (Qwen2.5-Omni) | MMLU-Pro | [Table 4] |
| MMLU-Redux | 78.87 | 74.70 (Qwen2.5-Omni) | MMLU-Redux | [Table 4] |

### Raon-Speech: 韩语 benchmark [Table 5]

| 指标 | Raon-Speech | 最强 baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ASR CER | 6.56 | 10.22 (Step-Audio 2) | KSponSpeech-clean | [Table 5] |
| KVoiceBench | 66.62 | 50.12 (Fun-Audio-Chat) | KVoiceBench | [Table 5] |
| KMMAU | 71.83 | 64.76 (Step-Audio 2) | KMMAU | [Table 5] |
| KMMLU-Pro | 46.85 | 43.23 (Fun-Audio-Chat) | KMMLU-Pro | [Table 5] |

[agent 解读] 韩语优势幅度远大于英语: KSponSpeech-clean CER 6.56 vs 次优 10.22 (35%+ 领先), KVoiceBench 66.62 vs 50.12 (33% 领先)。这反映了双语训练 + 大规模韩语数据的收益。英语上 ASR 和 speech gen 的 UTMOS 不是最优, 说明语音自然度还有提升空间。

### Raon-SpeechChat: Full-Duplex-Bench [Table 6]

| 指标 | Raon-SpeechChat | 最强 baseline | Benchmark | 出处 |
| --- | --- | --- | --- | --- |
| Pause TOR (↓) | 0.212 | 0.182 (MiniCPM-o) | FDB v1.0 Synthetic | [Table 6] |
| Backchannel TOR (↓) | 0.091 | 0.236 (PersonaPlex) | FDB v1.0 | [Table 6] |
| Smooth TOR (↑) | 0.832 | 0.891 (MiniCPM-o) | FDB v1.0 | [Table 6] |
| Interruption TOR (↑) | 0.980 | 0.910 (Freeze-Omni) | FDB v1.0 | [Table 6] |
| Interruption Judge (↑) | 2.790 | 3.408 (MiniCPM-o) | FDB v1.0 | [Table 6] |
| FDB v2.0 TT Fluency | 3.552 | 3.984 (MiniCPM-o) | FDB v2.0 | [Table 6] |
| FDB v2.0 IF | 3.042 | 3.534 (MiniCPM-o) | FDB v2.0 | [Table 6] |
| FDB v2.0 Task Metric | 2.944 | 3.241 (MiniCPM-o) | FDB v2.0 | [Table 6] |

[agent 解读] Raon-SpeechChat 在 FDB v1.0 上表现突出: backchannel TOR 0.091 远优于次优 0.236 (PersonaPlex), interruption TOR 0.980 几乎完美。但在 FDB v2.0 多轮会话上明显落后于 MiniCPM-o 4.5 和 PersonaPlex,说明长期对话一致性和多轮指令跟随是短板。User interruption Judge 分数 (2.790) 也低于 MiniCPM-o (3.408),说明虽然能检测到打断,但打断后的响应质量还有差距。

## 局限性

1. **语音自然度不是最优**: Speech gen UTMOS 在多个 benchmark 上低于 Qwen2.5-Omni 和 Step-Audio 2 mini,说明生成语音的感知自然度有差距 [Table 4, 5]
2. **长对话能力弱**: FDB v2.0 三项指标均显著落后于 MiniCPM-o 4.5,多轮指令跟随和任务完成率不足 [Table 6]
3. **打断响应质量**: interruption TOR 接近完美但 Judge 分数 (2.790) 不是最高,说明能检测打断但响应内容质量有待提升 [Table 6]
4. **仅双语**: 只支持英韩,未扩展到更多语言 [§7]
5. **数据依赖**: 大量 in-house 数据和专有 pipeline,可复现性受限 [§4]
6. **FDB v1.5 User Backchannel 弱**: Resume 率仅 0.398,低于 MiniCPM-o (0.520) 和 Freeze-Omni (0.480),说明在用户回传时维持说话的能力不足 [Table 6]

## 点评

Raon-Speech 是一个工程完成度非常高的系统级报告。几个值得注意的点:

1. **训练策略的精细设计**: 四阶段训练中每个阶段的冻结/解冻策略都有清晰的目的 -- alignment 阶段不动 backbone 避免破坏预训练知识, E2E 阶段用双教师 KD 同时促进 speech-text 对齐和防遗忘, post-training 用 SimPO 进一步消除退化行为。这套训练 recipe 的设计理念比单纯的实验结果更有借鉴价值。

2. **全双工设计的解耦思路**: SIL/BOW/BC 三 token 方案把交互行为 (when-to-speak) 和语言内容 (what-to-say) 显式解耦,比 Moshi 的单一 PAD token 方案更具可控性。推理时可以独立调节 backchannel 频率甚至禁用,这种可控性对实际部署很有价值。

3. **韩语基准的贡献**: 新建 KVoiceBench、KOpenAudioBench、KMMAU 三个韩语基准,填补了非英语 SpeechLM 评估的空白。但这些基准主要通过翻译+TTS 合成构建,与真正的韩语母语者录制的基准可能存在分布差异。

4. **诚实的结果呈现**: 论文明确指出 UTMOS 不是最优,FDB v2.0 落后于部分 baseline,没有选择性报告。这种透明度增加了结果的可信度。

5. **待观察的问题**: RCP 从 30B 模型初始化这一操作虽然加速了收敛,但意味着 Raon-Speech 的语音生成质量在一定程度上依赖于 Qwen3-Omni 的预训练结果, 这个跨系统的知识迁移链条是否稳健有待验证。

## 可复用的 idea

1. **On-policy self-distillation 双教师方案**: audio 输入用 text-conditioned 同模型做 teacher (促进 cross-modal 对齐), text 输入用冻结原始 backbone 做 teacher (防遗忘)。可迁移到任何 multimodal LLM 的模态扩展训练中。

2. **RMSNorm 小初始化 (0.02) 稳定 adaptor-LLM 对齐**: 一个简单但有效的 trick,确保新模态 embedding 不会在初始化时破坏 LLM 的 embedding 空间。

3. **SIL/BOW/BC 状态建模**: 将全双工对话的交互行为分解为可控的离散状态,比端到端隐式学习更具可控性和可解释性。特别是 BOW 把"决定说话"和"说什么"分开,BC 把"回传"和"正式发言"分开,推理时都可独立控制。

4. **Text lookahead**: 在全双工生成中让 text 先于 speech 一帧生成,减少语义漂移。简单但对全双工场景下的语音质量可能很关键。

5. **BC loss ×50 权重放大**: 处理全双工数据中 backchannel 标签极度不平衡的简单策略。

## 审阅

> [!review] 审阅: pass, 0 high (2026-06-06)
> 详见 `_review/Raon-Speech-review.yml`
>
> **结论**: pass — 无 high issue,1 medium issue 已修正
>
> **已修正**:
> - OpenAudioBench 数值误标 81.33 → 实际 70.21 (非 best,论文原文 §5.1 说 "remaining competitive") [factual-error, medium → fixed]
>
> **低风险遗留**:
> - frontmatter tasks 字段为空 (论文定义的任务可映射到 spoken QA / ASR / TTS / full-duplex dialogue,但 vault 中无对应任务页) [template-compliance, low]
> - frontmatter models 仅列 3 个模型,未列 Qwen3-VL-8B (backbone) 和 Voxtral (因 vault 无页面) [template-compliance, low]
