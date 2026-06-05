---
type: paper
tier: deep
title: "Covo-Audio Technical Report"
arxiv_id: "2602.09823"
source: "Sources/Covo-Audio.pdf"
authors: [Tencent]
year: 2026
venue: "arXiv"
tags: [speech-LM, end-to-end, full-duplex, LALM, speech-dialogue, audio-understanding, intelligence-speaker-decoupling, empathy, multimodal]
concepts: ["[[SpeechLanguageModel]]", "[[Full-duplexSpokenDialogue]]", "[[SpeechTokenizer]]", "[[ModalityAdaptationforSpeechLLM]]", "[[ConditionalFlowMatching]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[Turn-takinginSpokenDialogue]]", "[[AudioUnderstanding]]"]
models: ["[[Whisper]]", "[[WavLM]]", "[[BigVGAN]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: SpeechLanguageModel, Full-duplexSpokenDialogue, SpeechTokenizer, ModalityAdaptationforSpeechLLM, ConditionalFlowMatching, Speech-LLMIntegrationTaxonomy)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[Full-duplexSpokenDialogue]]✓, [[SpeechTokenizer]]✓, [[ModalityAdaptationforSpeechLLM]]✓, [[ConditionalFlowMatching]]✓, [[Speech-LLMIntegrationTaxonomy]]✓ | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: Covo-Audio 属于端到端 Speech Language Model (SpeechLM) 的最新进展。按 KB 中的 Speech-LLM Integration Taxonomy (Yang et al. 2025),它采用了 **混合集成方式**: 输入端使用 latent-representation-based (Whisper encoder + conv adapter 映射连续表征到 LLM 空间),输出端使用 audio-token-based (WavLM-based VQ tokenizer 生成离散 speech tokens)。这种 continuous-in, discrete-out 的设计在 KB 现有分类中较少见,大多数系统要么全连续 (Qwen2.5-Omni) 要么全离散 (GLM-4-Voice)。

**全双工上下文**: KB 中 Full-duplex Spoken Dialogue 页面记录了从 dGSLM → Moshi → OmniFlatten 的演进。Covo-Audio-Chat-FD 采用 hybrid dual-stream (连续输入+离散输出),不同于 Moshi 的全离散双流。KB 中记录 Moshi turn-taking 96.8%, Freeze-Omni 99.1%,Covo-Audio 在此基础上进一步提升到 99.7% 并大幅提升 pause handling (97.6% vs ~51%)。

**语音 tokenizer**: KB 中 SpeechTokenizer 页面指出 Mimi (Moshi) 也使用 WavLM 作为 semantic 信息来源。Covo-Audio 的 tokenizer 同样基于 WavLM-large + 单层 VQ (codebook 16384),但训练目标不同: 加入了 ASR loss + TTS reconstruction loss + pitch loss 三目标联合训练,追求 acoustic-semantic alignment。

**Modality Adapter**: KB 中记录了三种主要 adapter 方式 (conv downsampling > CTC > Q-Former)。Covo-Audio 采用 3 级 conv downsampling (各含 2 linear + 1 conv, stride=2),将 Whisper 50 Hz 输出降至 6.25 Hz,属于最基础但工程可靠的方案。

**Flow Matching decoder**: KB 中 CFM 页面记录了 FM 在 TTS 中广泛使用。Covo-Audio 的 speech decoder 采用两阶段: FM decoder 将离散 token → 连续 acoustic latent → BigVGAN vocoder 重建 24K 波形。

## 速查

> [!summary] 速查
> - **一句话**: 腾讯 7B 端到端 LALM,通过 hierarchical tri-modal interleaving 预训练 + intelligence-speaker decoupling 后训练,在语音对话/全双工/理解多任务上全面 SOTA 或 competitive
> - **路线**: 连续音频 → Whisper encoder → 3 级 conv adapter (6.25Hz) → Qwen2.5-7B → 交错 text+discrete speech tokens → FM decoder + BigVGAN → 24K 波形
> - **指标**: URO-Bench 中文 AlpacaEval 90.02, VStyle 共情 anxiety 5.00; Full-duplex turn-taking 99.7%, pause handling 97.6%; MMSU 66.64 (SOTA); AIR-Bench avg 80.86 (SOTA); 预训练 2T tokens
> - **可借鉴**: Intelligence-speaker decoupling 技术 --- 用多说话人训练解耦说话人与智力,再用 TTS 数据伪对话(masked text loss)转移高质量声音,成本远低于为每个声音收集对话数据
> - **局限**: 未开源全部版本(仅 Chat 版); 全双工版 pause handling 仍偶有 early-response; GaokaoEval 因长静音段导致 FD 版劣化明显; 音频理解推理能力仍不如 Qwen2.5-Omni 的 CoT

## 核心问题

本文要解决端到端 LALM 的三个核心挑战 [§1]:
1. **智力与自然度的矛盾**: Thinker-Talker 架构 (Qwen-Omni 系列) 通过中间文本推理保留智力,但牺牲了端到端语音指令遵循和对话可控性
2. **Intelligence-speaker 耦合**: 端到端模型中对话智力与说话人声音深度绑定,为每个新声音都需要收集大量高质量对话数据,成本极高
3. **全双工**: 需要同时听说+打断+轮次切换,但多数系统需要 word-level text-speech alignment,实现复杂

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Covo-Audio 由四个组件构成 [§2.1, Fig 2]:
1. **Audio Encoder**: Whisper-large-v3 (预训练冻结),输出 50Hz 帧率
2. **Audio Adapter**: 3 级下采样模块,每级含 2 个 linear + 1 个 conv (stride=2),50Hz → 6.25Hz [§2.1]
3. **LLM Backbone**: Qwen2.5-7B-Base,扩展词表加入 discrete audio tokens [§2.1]
4. **Speech Decoder**: FM-based decoder → BigVGAN vocoder → 24K 波形 [§2.1]

**Speech Tokenizer** 基于 WavLM-large + 单层 VQ (codebook 16384, 25Hz),通过多任务损失预训练: ASR loss (语义锚定) + TTS reconstruction loss (声学保真) + pitch loss (韵律保持) [§2.1]。

[agent 解读] 这里选择单层 VQ 而非 RVQ 值得注意。单 VQ token 序列对 LLM 的建模负担最小 (25Hz * 1 codebook = 25 tokens/s),但信息量有限; 声学细节完全交给下游 FM decoder 恢复。这与 Moshi 的 Mimi (1 semantic + 7 acoustic codebook) 形成对比 --- Covo-Audio 选择了更激进的压缩,把重建压力推给 decoder。

### 关键设计选择

#### 1. Hierarchical Tri-modal Speech-Text Interleaving [§2.2]

预训练中三种模态共存于统一序列:
- **连续声学表征** (a_c): Whisper encoder 输出
- **离散语音 token** (a_d): WavLM-VQ tokenizer 输出
- **文本 token** (t): 标准 text token

**两种交织模式**:
- Sequential: a_c → t → a_d (渐进链)
- Parallel: a_c → t|a_d (连续特征对齐文本-离散联合单元) [§2.2]

**层级策略**: phrase-level interleaving (细粒度声学-文字对齐) + sentence-level interleaving (保持长程语义完整性) [§2.2]。

[论文原文] "previous interleaving methods, such as those utilized in GLM-4-Voice, primarily operate at the word or character level. While effective for local modality alignment, such granular interleaving often sacrifices the semantic integrity of long-form utterances" [§2.2]。

[agent 解读] 这是对 GLM-4-Voice 的字符级交织的直接批评。Covo-Audio 的多尺度策略在 KB 现有 Speech-TextAlignment 讨论中属于新方向: 既不完全像 SPIRIT-LM 的 token-level 交替,也不像 AudioPaLM 的 sequence-level 拼接,而是两者兼有。

#### 2. Intelligence-Speaker Decoupling [§2.4, Fig 3]

核心思路是**将对话智力从说话人身份中解耦**,分三步:
1. **多说话人对话训练**: 用随机生成的千种说话人训练对话,使智力不依赖特定声音 [§2.4]
2. **TTS 数据转换**: 将高质量 TTS 录音构造为伪对话格式 (pseudo-context) [§2.4]
3. **Masked text loss**: 在 TTS 伪对话训练中排除 text response 部分的 loss 计算,保留对话推理能力 [§2.4]

[论文原文] "experiments show that this approach successfully transfers the naturalness of TTS speakers while maintaining intellectual levels comparable to those of dialogue speaker" [§2.4]

[agent 解读] masked text loss 是关键设计: 如果对 text response 也计算 loss,模型会被 TTS 数据中简单构造的文本答案"拉低"智力; 排除 text loss 后,模型只从 TTS 数据学习语音表达,不影响已有的推理链路。这比常见的"用 TTS 合成对话数据"路线更优雅 --- 后者受限于 TTS 系统本身的表达上限。

#### 3. Native Full-Duplex [§2.5, Fig 4]

从 Half-duplex (Covo-Audio-Chat) 演进到 Full-duplex (Covo-Audio-Chat-FD):

**架构改动**:
- Audio encoder 改为 chunk streaming 模式,支持实时输入 [§2.5]
- 用户流与模型流以 1:4 比例 chunk-interleaved (因输入 6.25Hz vs 输出 25Hz,每 chunk = 0.16s) [§2.5]
- 引入三种特殊 token: THINK (listening-only), SHIFT (进入说话), BREAK (停止说话) [§2.5]

**关键差异**: 与 Moshi 和 OmniFlatten 不同,Covo-Audio-Chat-FD 采用 **hybrid dual-stream** (连续输入 + 离散输出),不需要 word-level text-speech alignment [§2.5]。

**训练策略**: 将全双工直接放入预训练阶段 (Table 1 Stage 2: 5B tokens),而非多阶段渐进微调。半双工和全双工数据联合训练 [§2.5]。

[论文原文] "We found this simple-yet-effective approach can yield more competitive results" (相比 OmniFlatten 的多阶段渐进训练) [§2.5]

### 训练策略

**预训练** (2T tokens total) [§2.2, Table 1]:
- Stage 1 (Modality Bridging): 仅训练 adapter, ASR 任务, 200K h 多语言 ASR 数据, 30B tokens, 50K steps, LR=1e-4 [§2.2]
- Stage 2 (Modality Fusion): adapter + LLM 联合训练, 多任务 (ASR 80B + TTS 160B + Audio-only 240B + Speech Continuation 160B + Interleave 360B + Text-only 180B + Full-Duplex 5B + ...), 总计 ~2T tokens, 500K steps, LR=3e-5, seq_len=8192 [§2.2]

**后训练** (Spoken Dialogue) [§2.3, Table 2]:
- 任务配比: General Intelligence 0.4 + Spoken Dialogue 0.3 + Speech Understanding 0.1 + Speech Generation 0.1 + Audio Understanding 0.1 [Table 2]
- 10M text instruction data for T2T → 部分转为 TTS 语音 → T2A/A2T/A2A [§2.3]
- KL distillation: 用更强 text LLM 提供 top-20 logits 软目标,防止 T2T 智力退化 [§2.3]
- 口语化: 用 text LLM 重写 assistant responses 为口语风格 [§2.3]
- 共情: 7 情感类别 (joy/anger/sadness/fear/disgust/depression/surprise) 的情感对话数据 [§2.3]
- 训练参数: seq_len=8192, LR=1e-5, 50K steps [§2.3]

**音频理解** [§2.7]:
- Stage 1: AudioSkill dataset (8M pairs), 32K steps, backbone LR=3e-6, adapter LR=1e-5 [§2.7]
- Stage 2 (CoT): AudioMCQ + AF-Think (1M instances), 2 epochs [§2.7]
- Stage 3 (GRPO): AVQA benchmark, 复合 reward = R_accuracy + R_format + R_consistency + R_thinking [§2.7, Eq 1-2]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| A2A-tSC | 83.3 | 88.3 (SIMS) | StoryCloze | [Table 3] |
| A2T-tSC | 95.7 | 95.5 (Step-Audio-2-mini-Base) | tSC | [Table 3] |
| T2T-tSC | 99.4 | 98.6 (Qwen2.5-7B-Base) | tSC | [Table 3] |
| sBLIMP | 61.6 | 62.3 (AlignSLM, 含 RL) | sBLIMP | [Table 3] |
| sWUGGY | 74.90 | 75.36 (AlignSLM, 含 RL) | sWUGGY | [Table 3] |
| ASR Aishell-1 WER | 1.96 | 2.46 (GLM-4-Voice-Base) | Aishell-1 | [Table 4] |
| Seed-TTS Test-en WER | 2.44 | 2.57 (Whisper-large-v3) | Seed-TTS Eval | [Table 4] |
| URO-Bench zh AlpacaEval | 90.02 | 85.90 (MiMo-Audio) | URO-Bench | [Table 5a] |
| URO-Bench en Gsm8kEval | 85.68 | 80.00 (GPT-4o Audio) | URO-Bench | [Table 5b] |
| VCB-Bench TIF | 93.07 | 90.45 (Qwen3-Omni) | VCB-Bench | [Table 6] |
| VCB-Bench Robustness SV | 88.94 | 88.60 (Fun-Audio-Chat) | VCB-Bench | [Table 6] |
| VStyle Anxiety zh | 5.00 | 4.77 (Qwen2.5-Omni) | VStyle | [Table 7] |
| Full-Duplex Turn-taking | 99.7% | 99.1% (Freeze-Omni) | Custom | [Table 9] |
| Full-Duplex Pause Handling | 97.6% | 53.2/51.2% (Moshi/Freeze-Omni) | Custom | [Table 9] |
| Full-Duplex Backchanneling | 93.89% | N/A (Moshi/Freeze-Omni 无数据) | Custom | [Table 9] |
| Full-Duplex Interruption | 96.81% | N/A | Custom | [Table 9] |
| MMAU avg (7B) | 75.30 | 74.70 (MiMo-Audio) | MMAU-v05.15.25 | [Table 11] |
| MMSU avg | 66.64 | 61.40 (Audio Flamingo 3) | MMSU | [Table 11] |
| AIR-Bench avg | 80.86 | 71.98 (Qwen3-Omni) | AIR-Bench | [Table 10] |
| CoVoST2 en-zh | 49.84 | 49.12 (Step-Audio 2 mini) | CoVoST2 | [Table 10] |
| ASR avg WER | 4.71 | 2.76 (Step-Audio 2) | Multi-dataset | [Table 10] |

**Intelligence-Speaker Decoupling 验证** [Table 5]: Covo-Audio-Chat-TTS (用 TTS 声音替代对话声音) 在 URO-Bench 中英文对话性能与 Covo-Audio-Chat 相当,证明解耦成功。

**Full-Duplex 与 Half-Duplex 对比** [Table 5 vs Table 8]: Covo-Audio-Chat-FD 在 URO-Bench 上仅有轻微下降 (如中文 AlpacaEval 84.90 vs 90.02),但在全双工指标上大幅领先其他全双工模型 (Moshi, Freeze-Omni)。

## 局限性

1. **仅开源 Chat 版**: Covo-Audio-Chat-FD 和基础预训练版未开源,限制了社区复现和验证 [§1, GitHub]
2. **Early-response 问题**: FD 版在含长静音段的测试集 (如 GaokaoEval) 上显著劣化,因为模型倾向于在 pause 处过早响应 [Table 8 footnote]
3. **ASR 并非 SOTA**: 平均 WER 4.71 显著弱于 Step-Audio 2 (2.76) 和 Seed-ASR (3.07),Fleurs 中/英 WER 分别达 6.64/5.08 [Table 10]
4. **通用知识偏弱**: VCB-Bench GK 49.95 (vs Qwen3-Omni 66.86)、DC 64.95 (vs MiMo-Audio 82.78) [Table 6]
5. **音频推理可提升**: MMAU 和 MMSU 的推理分项未达 Qwen2.5-Omni 水平,论文自认 CoT 增强阶段仍有优化空间 [§3.5]
6. **主观共情仍有差距**: 尽管 VStyle 客观分高,论文承认主观语音共情测试仍弱于 Doubao 等产品级系统 [§3.2]

## 点评

**优势**:
- **系统工程完整度极高**: 在单一 7B 模型中覆盖 ASR/TTS/对话/共情/全双工/音频理解六大能力,且在多数 benchmark 上 competitive,这是罕见的工程成就
- **Intelligence-speaker decoupling 是真正的创新**: 这解决了端到端 LALM 产品化的核心成本问题,且 masked text loss 方案简洁有效
- **全双工训练策略务实**: 直接放入预训练而非多阶段渐进,结果反而更好; hybrid dual-stream 避免了 word-level alignment 的复杂性
- **Tri-modal interleaving 设计合理**: 多尺度 (phrase + sentence) 解决了 GLM-4-Voice 字符级交织的长程语义碎片化问题

**不足**:
- **Tech report 风格导致方法描述偏浅**: 很多关键决策 (如为什么选 WavLM 而非 HuBERT 做 tokenizer, FM decoder 的具体架构) 缺乏详细 ablation
- **评估缺乏消融实验**: 没有逐步去掉 tri-modal interleaving / KL distillation / TTS decoupling 的消融,无法判断各模块的边际贡献
- **VCB/URO 评估依赖 LLM-as-a-Judge**: 论文自己也指出 LLM judge 可能偏重语义而忽略语音表达质量 [§3.2]

## 可复用的 idea

1. **Intelligence-Speaker Decoupling** [§2.4]: 在端到端语音模型中,先用多说话人训练解耦智力和声音,再用 TTS 数据(masked text loss)转移目标声音。适用于任何需要灵活声音定制的对话系统
2. **Hierarchical Tri-modal Interleaving** [§2.2]: 在预训练中同时使用连续/离散/文本三模态,phrase-level + sentence-level 多尺度交织。可用于任何需要深度 speech-text 对齐的 SpeechLM 预训练
3. **Native Full-Duplex 预训练** [§2.5]: 将全双工数据直接放入预训练(而非仅后训练),single-step 训练优于多阶段渐进。适用于构建全双工 SpeechLM
4. **GRPO + 复合 reward 用于音频推理** [§2.7]: R_accuracy + R_format + R_consistency + R_thinking 四分量 reward,可用于提升音频/语音 LLM 的 CoT 推理质量
5. **KL distillation 防智力退化** [§2.3]: 在跨模态训练中用强 text LLM 的 top-K logits 做软目标,防止 T2T 能力下降。通用于任何将 text LLM 扩展到新模态的场景

## 审阅

> [!review] 审阅 (2026-06-05, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 三大设计选择有 WHY 解释,速查可借鉴具体可迁移 |
> | 可信赖 | pass | 18 指标全标注出处,抽查数字与原文一致 |
> | 可区分 | pass | 3+3 处来源标注覆盖主要因果解释 |
> | 可定位 | pass | KB 背景 6 页,谱系定位具体,与 Moshi/GLM-4-Voice 对比明确 |
> | 不污染 | pass | 仅 append 更新,无新建页,无 factual error |
> 
> Issues: 2 (high: 0, medium: 0, low: 2)
> 详见 `_review/Covo-Audio-review.yml`

---

检索命中: [[SpeechLanguageModel]], [[Full-duplexSpokenDialogue]], [[SpeechTokenizer]], [[ModalityAdaptationforSpeechLLM]], [[ConditionalFlowMatching]], [[Speech-LLMIntegrationTaxonomy]] | 过滤: 无 | 未命中但可能相关: 无
