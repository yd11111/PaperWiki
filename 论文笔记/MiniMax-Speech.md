---
type: paper
tier: deep
title: "MiniMax-Speech"
aliases: [MiniMax Speech, Speech-02-HD]
authors: ["MiniMax"]
year: 2025
arxiv_id: "2505.07916"
source: "Sources/MiniMax-Speech.pdf"
venue: "arXiv"
tags: [TTS, zero-shot, voice-cloning, autoregressive, flow-matching, VAE, speaker-encoder, multilingual, emotion-control, LoRA]
concepts: ["[[LLM-basedTTS]]", "[[ConditionalFlowMatching]]", "[[SpeakerEmbedding]]", "[[VoiceCloningTaxonomy]]", "[[VariationalAutoencoderforTTS]]", "[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]"]
models: ["Seed-TTS", "CosyVoice2", "ElevenLabs-Multilingual-v2"]
tasks: [TTS, zero-shot-TTS, voice-cloning, cross-lingual-TTS, emotion-control]
datasets: [Seed-TTS-eval, Mozilla-Common-Voice, Artificial-Arena]
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
kb_sources: ["[[LLM-basedTTS]]", "[[ConditionalFlowMatching]]", "[[SpeakerEmbedding]]", "[[VoiceCloningTaxonomy]]", "[[VariationalAutoencoderforTTS]]"]
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认/相关实体页: LLM-basedTTS, ConditionalFlowMatching, SpeakerEmbedding, VoiceCloningTaxonomy, VariationalAutoencoderforTTS)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: LLM-basedTTS, ConditionalFlowMatching, SpeakerEmbedding, VoiceCloningTaxonomy, VariationalAutoencoderforTTS | 过滤: 无 | 未命中但可能相关: 无

- **[[LLM-basedTTS]]** (confirmed): MiniMax-Speech 属于 AR Transformer + Flow Matching 的 hybrid 路线,与 CosyVoice 系列同属 LLM+Flow 架构。但与 VALL-E/CosyVoice 的 one-shot prompting 不同,MiniMax-Speech 通过 learnable speaker encoder 实现"真正的 zero-shot"(无需参考文本),这是架构层面的根本差异。LLM-basedTTS 页面记录了 in-context learning 范式中 speaker conditioning 通常需要 text-audio pair 作为 prompt,MiniMax-Speech 正是要解耦这一依赖。
- **[[ConditionalFlowMatching]]** (confirmed): MiniMax-Speech 的 flow matching 模块不直接对 mel spectrogram 建模,而是对 Flow-VAE encoder 产出的连续 latent 建模 [§2.2.1]。这与 CosyVoice 系列对 mel spectrogram 做 CFM 形成对比。CFM 概念页记录了 VAE latent 作为 flow matching target 可以提升重建上限 (消除 mel bottleneck),MiniMax-Speech 的 Flow-VAE 正是这一路线的具体实例。
- **[[SpeakerEmbedding]]** (confirmed): MiniMax-Speech 的核心创新是 learnable speaker encoder (与 AR Transformer 联合训练),对比 SpeakerEmbedding 页面记录的两种范式 — lookup table (closed-set) 和 pre-trained speaker encoder (open-set),MiniMax-Speech 提出第三条路: jointly-trained speaker encoder,兼具 open-set 泛化性和 task-specific 优化。这继承了 Tortoise TTS (Betker 2023) 的思路。
- **[[VoiceCloningTaxonomy]]** (pending-review): MiniMax-Speech 对"zero-shot"做了更严格的定义:仅用无转写的参考音频提取 speaker identity,不提供任何 text-audio pair 作为 prompt。按此定义,VALL-E/CosyVoice 2/Seed-TTS 等"zero-shot"系统实质上属于"one-shot"(需要 paired prompt)。这一术语澄清对 cloning taxonomy 有参考价值。
- **[[VariationalAutoencoderforTTS]]** (pending-review): MiniMax-Speech 提出 Flow-VAE,在 VAE encoder-decoder 之间引入 normalizing flow 将 posterior 从标准正态放宽到更灵活的分布。这与 VITS 的 VAE+normalizing flow 思路相似 (增强 prior 表达力),但 Flow-VAE 的目标是增强 encoder 的信息表达力 (posterior → flow → standard normal),而非增强 prior [§2.2.2]。VAE 概念页记录的"重建-生成困境"在此处得到一种新解法: 用 flow model 提升 VAE latent 的信息密度,而非调整维度或引入语义正则化。

> [!summary] 速查
> - **一句话**: AR Transformer + learnable speaker encoder 实现无需参考转写的零样本语音克隆,Flow-VAE 增强 flow matching 的 latent 表达力,TTS Arena 排名第一
> - **路线**: Text(BPE) + Reference Audio → Speaker Encoder(固定维度条件向量) → AR Transformer(离散 audio tokens, 25 tok/s) → Flow Matching(Transformer, 条件: AR output + speaker embedding + optional prompt) → Flow-VAE Decoder(latent → waveform)
> - **指标**: Seed-TTS-eval test-zh WER 0.83 / SIM 0.783 (zero-shot), WER 0.99 / SIM 0.799 (one-shot); test-en WER 1.65 / SIM 0.692 (zero-shot); Artificial Arena ELO 第一 [Table 1, Fig 4]
> - **可借鉴**: (1) Learnable speaker encoder 联合训练,比冻结 SV encoder 在 WER 上优势显著 (1.252 vs 1.400) [Table 4]; (2) Flow-VAE 用 normalizing flow 放宽 VAE posterior 约束,提升 latent 信息密度 [§2.2.2]; (3) Speaker embedding 作为可微调向量实现 PVC (professional voice cloning),仅优化一个向量即可适配新说话人 [§4.3]; (4) LoRA-based emotion control 用 neutral/random 参考音频做对比学习,解耦情感与内容 [§4.1]
> - **局限**: 不公开训练数据规模和模型参数量; 消融实验仅在中文子集上做; 无流式推理讨论; Artificial Arena 排名时效性强; 多语言评估中粤语 WER 仍高 (34.1%)

## 核心问题

### WHY: 为什么要做这个工作?

现有 AR TTS 模型在 voice cloning 时通常需要参考音频 **及其转写文本** 作为 prompt (one-shot),这带来三个问题 [§1]:

1. **语义/语言失配**: prompt 的文本内容与目标文本不同,导致 prosody bias 和跨语言时的发音错误 [论文原文]
2. **解码空间受限**: one-shot prompt 的韵律模式会约束生成多样性,模型倾向模仿 prompt 的韵律而非自由生成 [论文原文]
3. **需要转写**: 参考音频必须有准确文本转写,增加了部署复杂度并引入转写错误风险 [论文原文]

另一方面,使用预训练 Speaker Verification (SV) 模型作为固定 speaker encoder 也有局限: SV 任务的训练数据和优化目标与 TTS 不一致,encoder 无法被 TTS 任务优化 [§1, 论文原文]。

### WHAT: 核心贡献

1. **Learnable speaker encoder**: 与 AR Transformer 联合训练,实现不依赖参考转写的 zero-shot voice cloning [§2.1]
2. **Flow-VAE**: 在 VAE encoder-decoder 之间引入 normalizing flow,将 VAE posterior 约束从标准正态放宽到更灵活的分布,提升 latent 信息密度 [§2.2.2]
3. **32 语言支持**: 在 Seed-TTS-eval 上 SOTA WER/SIM,Artificial Arena TTS 排名第一 [§3.2, §3.3]
4. **可扩展框架**: 基于 speaker encoder 的解耦表示,无需改基座模型即可扩展到 emotion control (LoRA)、T2V、PVC [§4]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MiniMax-Speech 由三个模块组成 [§2, Fig 1]:

```
Text(BPE) ──────────────────────────────────────┐
                                                 ↓
Reference Audio → Speaker Encoder → condition  → AR Transformer → discrete audio tokens
                       ↓                                              ↓ (upsample)
                   speaker emb ──────────────→ Flow Matching ←── AR output (c')
                                                     ↓
                                              Flow-VAE Decoder → waveform
```

**Audio tokenizer**: Encoder-VQ-Decoder 架构,对 mel spectrogram 做量化,25 tokens/s,使用 CTC 监督确保 token 保留语义信息 [§2]。[agent 解读] 这与 Tortoise TTS 的 VQVAE mel-token 路线一脉相承,但加了 CTC 损失使 token 保留更多语义信息。

**Text tokenizer**: BPE [§2]。

### 关键设计选择

#### 1. Learnable Speaker Encoder vs Pre-trained SV Encoder

**设计**: Speaker encoder 与 AR Transformer 联合训练,从参考音频 (不同于目标音频) 提取固定维度条件向量 [§2.1]。

**WHY**: [论文原文] 预训练 SV encoder 的优化目标是区分说话人 (verification),而 TTS 需要的 speaker representation 还须包含有助于语音合成的信息。联合训练让 encoder 能针对 TTS 任务优化,提供更丰富的 speaker-specific 信息 [§2.1]。此外,learnable encoder 可在全语言训练数据上训练,比预训练 SV encoder 有更广的语言覆盖 [§2.1]。

**关键约束**: 训练时参考音频 **必须与目标音频不同** (来自同一说话人但不同句子)。[论文原文] 如果使用相同音频,会导致 semantic leakage — encoder 可以直接传递内容信息给 decoder,绕过 text conditioning,退化生成质量 [§3.6]。

**消融验证** [Table 4]:
- Learnable encoder zero-shot: WER 1.252, SIM 0.730
- Pre-trained SpkEmbed zero-shot: WER 1.400 (+11.8%), SIM 0.746
- OnlyPrompt one-shot: WER 1.207, SIM 0.726

[agent 解读] SpkEmbed 的 SIM 反而略高于 learnable encoder (0.746 vs 0.730),说明 SV encoder 确实善于提取 speaker identity;但 WER 显著更差 (1.400 vs 1.252),表明 SV 优化目标引入了对 TTS 不利的表示偏差 (可能过度保留了与语义无关的声学细节)。Learnable encoder 在 one-shot 模式下 SIM 提升到 0.746 且 WER 仅 1.243,是最佳平衡点。

#### 2. Zero-shot vs One-shot 的重新定义

MiniMax-Speech 对 LLM 术语做了更严格的迁移 [§2.1, Fig 2]:

| 模式 | 条件 | 代表系统 | MiniMax 定义 |
|------|------|----------|-------------|
| Zero-shot | 仅 speaker encoder 条件向量 | MiniMax-Speech (Fig 2b) | 真正的 zero-shot |
| One-shot | Speaker encoder + text-audio pair 作为 prompt | MiniMax-Speech (Fig 2c) | One-shot |
| "One-shot (only prompt)" | 仅 text-audio pair 作为 prompt | VALL-E, CosyVoice 2, Seed-TTS (Fig 2a) | 其他论文称"zero-shot"的方法 |

**核心论点**: [论文原文] zero-shot 模式的 WER (0.83) 优于 one-shot (0.99),因为不受 prompt 韵律偏差约束,模型有更大的解码自由度生成自然韵律 [§3.2]。但 one-shot SIM (0.799) 高于 zero-shot (0.783),因为额外的 paired prompt 提供了更精细的声学线索 [§3.2]。

#### 3. Flow-VAE: 增强 VAE Latent 的信息密度

**动机**: [论文原文] 传统 TTS 中 flow matching 对 mel spectrogram 建模,而 mel 作为信息瓶颈限制了最终语音质量上限。VAE 端到端训练的 latent 比 mel 有更强的表示能力,但标准 VAE 强制 posterior 服从标准正态分布,限制了 latent 的信息表达 [§2.2.1]。

**设计** [§2.2.2, Fig 3a]: 在 VAE encoder 输出和 KL 约束之间插入 normalizing flow $f_\theta$:

1. VAE encoder 输出 $\tilde{z} \sim N(\mu_\phi(x), \sigma_\phi(x))$
2. Flow model $f_\theta$ 对 $\tilde{z}$ 做可逆变换
3. KL 散度计算在 flow 输出与标准正态之间: $L_{kl} = D_{KL}(q_\phi(\tilde{z}|x) || p(\tilde{z}))$ [Eq. 1-4]

**WHY**: [论文原文] 这样 VAE encoder 只需输出一个任意正态分布 (而非标准正态),flow model 负责将其映射到标准正态。更灵活的 posterior 分布让 encoder 能更准确地捕获数据的复杂结构 [§2.2.2]。

[agent 解读] 这与 VITS 的 posterior flow 思路相同 (Kim et al., ICML 2021),但方向不同: VITS 用 flow 增强 prior (text → z 的路径),而 MiniMax-Speech 的 Flow-VAE 用 flow 增强 posterior (audio → z 的路径)。两者的共同本质是用 normalizing flow 弥合 Gaussian 假设与真实分布的 gap。

**消融验证** [Table 5, 6]:
- Resynthesis: Flow-VAE 在所有指标上优于 VAE (NB-PESQ 4.34 vs 4.27, WB-PESQ 4.30 vs 4.20, MS-STFT-LOSS 0.62 vs 0.67)
- TTS: Flow-VAE latent 上训练的 flow matching 在 WER 和 SIM 上均优于 VAE latent

#### 4. Flow Matching 的条件信息

Flow matching 模块使用 Transformer 架构,条件包括 [§2.2.1, Fig 3b]:
- AR Transformer 输出 $c$ (经 Conv1d + Upsample 得到 $c'$)
- Speaker embedding $v$ (从 speaker encoder 提取)
- (可选) Prompt 连续语音特征 $x_p$ (训练时以一定概率使用当前句开头的信息)

[agent 解读] 这种 speaker embedding + prompt 双路条件注入与 CosyVoice 2 的设计一致 (论文明确引用了 CosyVoice 2),说明在 flow matching 阶段同时注入全局音色和局部声学线索已成为工程共识。

### 训练策略

论文对训练细节披露极少:
- **训练数据**: 32 语言多语言语音数据集,经双 ASR 验证 + VAD/ASR 时间戳精化标点 + 多说话人验证模型保持音色一致 [§3.1]
- **数据规模**: 未公开 [agent 解读]
- **模型参数量**: 未公开 [agent 解读]
- **训练硬件/时长**: 未公开 [agent 解读]

## 实验

### Voice Cloning (Seed-TTS-eval)

| 模型 | Cloning Mode | test-zh WER ↓ | test-zh SIM ↑ | test-en WER ↓ | test-en SIM ↑ | 出处 |
|------|-------------|--------------|--------------|--------------|--------------|------|
| Ground Truth | - | 1.25 | 0.750 | 2.14 | 0.730 | [Table 1] |
| Seed-TTS | one-shot | 1.12 | 0.796 | 2.25 | 0.762 | [Table 1] |
| CosyVoice 2 | one-shot | 1.45 | 0.748 | 2.57 | 0.652 | [Table 1] |
| MiniMax-Speech | zero-shot | **0.83** | 0.783 | **1.65** | 0.692 | [Table 1] |
| MiniMax-Speech | one-shot | 0.99 | **0.799** | 1.90 | 0.738 | [Table 1] |

### Multilingual (24 languages, vs ElevenLabs Multilingual v2)

MiniMax-Speech 在中文/粤语/泰语/越南语/日语等声调语言上显著优于 ElevenLabs (如中文 WER 2.252 vs 16.026, 越南语 0.880 vs 73.415) [Table 2]。在 SIM 上全部 24 语言超越 ElevenLabs。

**粤语例外**: MiniMax WER 34.1%,仍远优于 ElevenLabs 的 51.5%,但绝对值仍高 [Table 2]。[agent 解读] 粤语的 ASR 评估本身准确率有限,且粤语书面/口语差异大,高 WER 可能部分来自评估方法而非合成质量。

### Cross-lingual

Zero-shot 跨语言 WER 全面低于 one-shot [Table 3],如法语 4.497 vs 5.489,芬兰语 4.527 vs 8.112。[论文原文] Zero-shot 不受中文 prompt 的语言干扰,speaker encoder 提取的音色信息是语言无关的 [§3.5]。

### Subjective (Artificial Arena)

在 Artificial Arena (人类偏好排行榜) 上,以 Speech-02-HD 名义参赛,**全部使用 zero-shot 生成**,排名第一,超越 OpenAI、ElevenLabs、Google、Microsoft、Amazon [§3.3, Fig 4]。

### Speaker Condition Ablation [Table 4]

| 方法 | Cloning Mode | WER ↓ | SIM ↑ |
|------|-------------|-------|-------|
| Speaker Encoder | zero-shot | 1.252 | 0.730 |
| Speaker Encoder | one-shot | 1.243 | 0.746 |
| SpkEmbed (pre-trained SV) | zero-shot | 1.400 | 0.746 |
| SpkEmbed (pre-trained SV) | one-shot | 1.704 | 0.744 |
| OnlyPrompt | one-shot | 1.207 | 0.726 |

[agent 解读] 值得注意的是 SpkEmbed + one-shot 的 WER 1.704 远高于 SpkEmbed zero-shot 的 1.400,这表明 pre-trained SV embedding 与 prompt exemplar 之间存在某种冲突,可能因为两者编码了不一致的 speaker 信息。Learnable encoder 在 one-shot 时则没有这种退化 (1.243 vs 1.252),说明联合训练使 encoder 能与 prompt conditioning 协调工作。

### Flow-VAE Ablation [Table 5, 6]

Resynthesis: Flow-VAE 在 SELF-SIM (+0.006)、NB-PESQ (+0.07)、WB-PESQ (+0.10)、MS-STFT-LOSS (-0.05) 上均优于 VAE [Table 5]。

TTS: Flow-VAE latent 在 SIM 上一致优于 VAE latent (test-zh zero-shot 0.751 vs 0.747, one-shot 0.782 vs 0.776) [Table 6]。WER 差异极小。[论文原文] 主观听感上 Flow-VAE 在稳定性方面有显著优势 [§3.7]。

## Extensions

### Emotion Control via LoRA [§4.1]

每种情感类别训练独立 LoRA 模块,推理时动态加载。训练数据格式: <reference audio, text, target emotive audio>。

**关键发现**: reference audio 的情感类型影响效果 [论文原文]:
- 情感一致参考: 输出情感过度依赖参考,LoRA 失控
- 中性参考: 最高表现力 (情感对比最大)
- 随机情感参考: 最鲁棒的 speaker similarity (迫使模型解耦 identity 与 expression)

同时收集同文本不同情感的配对数据,训练模型学习文本内容与情感表达的解耦 [§4.1]。

### Text to Voice (T2V) [§4.2]

从 AR Transformer 和 flow matching 模型中提取 timbre 表示,用 PCA 压缩到 128 维。训练紧凑的 timbre generation model,从文本描述 + 结构化属性 (语速/性别/语言/音高/音量) 预测 timbre 向量。使用随机 mask 增强鲁棒性。受 Spark-TTS 启发 [§4.2]。

### Professional Voice Cloning (PVC) [§4.3]

将 speaker encoder 输出的条件向量视为可学习参数,冻结 AR Transformer,仅微调这个向量。推理时用微调后的向量替代 speaker encoder 实时输出。

[agent 解读] 这本质上是把 speaker embedding 当作 prompt tuning (Li & Liang, 2021) 中的 soft prompt,是 PEFT 思想在 TTS 领域的巧妙应用。每个说话人仅需存储一个向量,可扩展到数千说话人而不增加模型开销。

## 局限性

1. **训练细节缺失**: 数据规模、模型参数量、训练成本均未公开,难以复现 [agent 解读]
2. **消融局限**: Speaker condition 和 Flow-VAE 消融仅在中文子集上做,未验证多语言泛化 [§3.6, §3.7]
3. **评估时效性**: Artificial Arena 排名随时间变化,2025 年 5 月的排名不代表当前状态 [agent 解读]
4. **无流式讨论**: 论文未讨论推理延迟、流式能力、RTF 等部署关键指标 [agent 解读]
5. **粤语质量**: WER 34.1% 显著高于其他语言,该语言的支持仍有提升空间 [Table 2]
6. **缺少开源**: 模型和数据均未开源,仅有 demo 页面 [agent 解读]

## 点评

**MiniMax-Speech 的核心贡献不在于架构创新的深度,而在于对 speaker conditioning 范式的系统性思考和工程整合的完成度。**

论文对 zero-shot vs one-shot 的术语重新定义 [§2.1] 具有理论清洁性: 按照 LLM 的严格定义,需要 text-audio pair 作为 in-context example 的方法确实更符合 one-shot 的语义。这一澄清对后续研究有参考价值。

Learnable speaker encoder 的思想来源于 Tortoise TTS (Betker, 2023) 和 XTTS (Casanova et al., 2024),但 MiniMax-Speech 通过大规模 32 语言训练和系统消融 [Table 4] 首次在工业级系统上验证了其有效性。消融中 SpkEmbed + one-shot 的 WER 异常退化 (1.704) 是一个有价值的 negative finding,值得后续研究关注。

Flow-VAE 的贡献相对增量: VAE + normalizing flow 的组合在 VITS (2021) 中已有先例,Flow-VAE 将其用于不同位置 (增强 posterior 而非 prior),在 TTS 指标上的提升幅度有限 (WER/SIM 改善 < 1%) [Table 6]。论文声称的主观稳定性优势 [§3.7] 缺乏定量支撑。

Extensions 部分 (emotion LoRA, T2V, PVC) 展示了 speaker encoder 解耦表示的工程价值,尤其是 PVC 的 embedding-only fine-tuning 极其轻量。但这些扩展均无定量评估 (无 emotion accuracy 数字、无 T2V 的 MOS、PVC 也无对比实验),仅有定性描述。

**与已知系统对比定位**: MiniMax-Speech 在 Seed-TTS-eval 上 WER 显著优于 Seed-TTS 和 CosyVoice 2,SIM 与 Seed-TTS 持平。考虑到它使用的是更严格的 zero-shot 设置 (无 transcribed prompt),这个结果更加突出。但缺失的训练数据规模信息使得公平性难以判断。

## 可复用的 idea

1. **Learnable speaker encoder 联合训练** [§2.1]: 将 speaker encoder 从 frozen pre-trained 改为 jointly trained,使其对 TTS 任务优化。关键约束: 训练时参考音频必须与目标不同,防止 semantic leakage [§3.6]
2. **PVC: Embedding-only fine-tuning** [§4.3]: 冻结全部模型参数,仅微调 speaker embedding 向量即可适配新说话人。比 LoRA/SFT 更轻量,可扩展到千人级
3. **Emotion LoRA 的参考音频策略** [§4.1]: 训练 emotion LoRA 时用中性/随机情感参考音频,而非情感一致参考,防止情感依赖参考而非 LoRA 控制
4. **Flow-VAE** [§2.2.2]: 在 VAE 中插入 normalizing flow 放宽 posterior 约束,适用于任何需要提升 VAE latent 信息密度的场景

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节包含因果解释和设计动机,关键设计选择回答了 WHY |
> | 可信赖 | pass | 数字 claim 均有 Table/Section 标注,覆盖率 >80% |
> | 可区分 | pass | [论文原文] 和 [agent 解读] 标注清晰,覆盖率 >80% |
> | 可定位 | pass | KB 背景提供了与 CosyVoice/VALL-E/Tortoise/VITS 的定位对比 |
> | 不污染 | pass | 未创建新概念页,反向更新为追加 key_papers |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> 详见 `_review/MiniMax-Speech-review.yml`
