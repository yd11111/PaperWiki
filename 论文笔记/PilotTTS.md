---
type: paper
tier: deep
title: "PilotTTS: A Disciplined Modular Recipe for Competitive Speech Synthesis"
arxiv_id: "2605.27258"
source: "Sources/PilotTTS.pdf"
authors: [Bowen Li, Shaotong Guo, Zhen Wang, Yang Xiang, Mingli Jin, Yihang Lin, Jiahui Zhao, Weibo Xiong, Dongrui Zhang, Keming Chen, Yunze Gao, Zeyang Lin, Yuze Zhou, Yue Liu]
year: 2026
venue: "Amap Voice Technical Report"
tags: [TTS, zero-shot, autoregressive, LLM-based, Q-Former, data-engineering, emotion, paralinguistic, dialect, coarse-to-fine]
concepts: ["[[ConditionalFlowMatching]]", "[[SpeakerEmbedding]]", "[[SpeechTokenizer]]", "[[FiniteScalarQuantization]]", "[[LLM-basedTTS]]", "[[EmotionControlinTTS]]", "[[Single-codebookvsMulti-codebook]]"]
models: ["[[模型库/CosyVoice3|CosyVoice 3]]", "[[模型库/w2v-BERT|w2v-BERT]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[ConditionalFlowMatching]], [[SpeakerEmbedding]], [[SpeechTokenizer]], [[LLM-basedTTS]], [[Zero-shotSpeechSynthesis]], [[SEED-TTS-Eval]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: PilotTTS 属于 LLM-based TTS 的 "Hybrid 架构" 路线 (LLM 生成 semantic tokens + CFM 合成语音),与 CosyVoice 系列、Seed-TTS 同属 coarse-to-fine 范式。具体而言,它直接复用 CosyVoice 3 的 FSQ 单码本 speech tokenizer 和 HiFi-GAN vocoder,使用 Qwen3-0.6B 作为 AR backbone,但在 speaker conditioning 上提出了独特的 Q-Former 双路径设计,这与 CosyVoice 系列的 prompt 续写方式和 Seed-TTS 的 self-distillation 解耦方式形成对比。
>
> **已有认知**: 
> - [[ConditionalFlowMatching]] 是 coarse-to-fine TTS 中 "fine stage" 的标配,CosyVoice 3 已将 CFM 扩至 300M DiT backbone;PilotTTS 沿用相同配置 [§3.4]
> - [[SpeakerEmbedding]] 在零样本 TTS 中有两种范式:audio token continuation (高保真但长 prompt 成本高) vs speaker encoder (鲁棒但丢细节);PilotTTS 提出 Q-Former + CAMPPlus 双路径融合两者优点
> - [[SpeechTokenizer]] 在 TTS 中的核心 trade-off 是语义编码 vs 声学保留;PilotTTS 复用 CosyVoice 3 的 FSQ 单码本 tokenizer (25 Hz, codebook 6561),免去了多码本建模的复杂性
> - [[SEED-TTS-Eval]] 是当前零样本 TTS 的标准 benchmark;此前最高 SIM 为 Seed-TTS (test-zh 0.796, test-en 0.762),PilotTTS 刷新至 0.862/0.815
>
> **创新判断 (基于 KB)**: PilotTTS 的核心创新不在模型架构,而在 (1) Q-Former 双路径 conditioning 设计实现 speaker-style 解耦,(2) 200K 小时数据下通过可复现数据管线达到 SOTA 级性能,降低了零样本 TTS 的数据门槛。
>
> 检索命中: [[ConditionalFlowMatching]]✓, [[SpeakerEmbedding]]✓, [[SpeechTokenizer]]✓, [[LLM-basedTTS]]✓, [[Zero-shotSpeechSynthesis]]✓, [[SEED-TTS-Eval]]✓ | 过滤: [[FiniteScalarQuantization]](pending-review), [[EmotionControlinTTS]](pending-review), [[CosyVoice3]](pending-review), [[w2v-BERT]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review) | 未命中但可能相关: Q-Former(无独立概念页)

## 速查

> [!summary] 速查
> - **一句话**: 以可复现数据管线 + Q-Former 双路径 speaker-style 解耦 conditioning 为核心,用 200K 小时数据和 Qwen3-0.6B 构建零样本 TTS 系统,在 SEED-TTS-Eval 上取得最高 speaker similarity (0.862/0.815) 和最低英文 WER (1.50%)
> - **路线**: Text → Qwen3 tokenizer → Qwen3-0.6B AR (conditioned on CAMPPlus spk emb + Q-Former style tokens) → semantic tokens → DiT-CFM decoder → mel spectrogram → HiFi-GAN → waveform
> - **指标**: SIM 0.862 (test-zh) / 0.815 (test-en), CER 0.87% (test-zh), WER 1.50% (test-en), 情感控制主类平均成功率 88.1%, 副语言整体 85.1% [Table 1-4]
> - **可借鉴**: (1) cross-sample paired training 策略实现 speaker-style 解耦 -- 不同内容的同说话人音频做参考,迫使 conditioner 只编码说话人属性; (2) 三阶段数据管线完全用开源工具构建,从 raw audio 到 200K h 高质量训练数据; (3) 方言合成的 mixed-prompt sampling 策略,用预训练模型合成普通话做伪平行数据
> - **局限**: (1) 无显式 style modeling module -- Q-Former 隐式建模风格,细粒度表达力可能受限; (2) 单码本 FSQ 信息容量上限低于多码本 RVQ,难以扩展到歌唱/背景音乐; (3) mel + vocoder 的间接重建引入额外失真; (4) 未开源训练数据,仅开源管线工具和模型权重

## 核心问题

PilotTTS 试图回答: **资源受限团队能否用更少数据、更简单架构构建竞争力达 SOTA 级的零样本 TTS?** 当前 TTS 系统普遍依赖百万小时级专有数据、多码本分层架构和多阶段训练,导致复现门槛极高。PilotTTS 的 thesis 是: 数据质量 > 数据规模,架构整合 > 架构创新 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

PilotTTS 采用四阶段 coarse-to-fine 架构 [§3.1, Fig 3]:

1. **Speech Tokenizer**: 直接复用 CosyVoice 3 的 FSQ 单码本 tokenizer (基于 MinMo,25 Hz,codebook 6561),将语音转换为离散 semantic tokens [§3.2]
2. **AR Text-to-Semantic Module**: Qwen3-0.6B 自回归生成 semantic token 序列,条件包括 text tokens、CAMPPlus speaker embedding 和 Q-Former style condition tokens [§3.3]
3. **CFM Decoder**: 300M DiT backbone,将 semantic tokens + speaker embedding + reference mel 转换为 mel spectrogram,10 步去噪 [§3.4]
4. **HiFi-GAN Vocoder**: mel spectrogram → waveform [§3.4]

### 关键设计选择

#### 1. Q-Former 双路径 Conditioning [§3.3.1]

**问题**: 如何从参考音频提取说话人信息?现有两种范式各有缺陷:
- Audio token continuation (如 VALL-E): 保留细粒度声学线索,但对噪声/短 prompt 退化严重,且长 prompt 推理成本高 [论文原文]
- Speaker embedding (如传统 d-vector): 鲁棒但丢失音色细节和动态风格 [论文原文]

**解决方案**: 双路径设计 [§3.3.1]:
- **路径 1 -- Q-Former Semantic Content Adapter**: 32 个可学习 query vectors 通过 cross-attention 从冻结的 w2v-BERT 2.0 encoder 输出中提取 style condition tokens。Q-Former 包含 Conformer block + linear projection [Fig 3 右侧]
- **路径 2 -- CAMPPlus Speaker Encoder (冻结)**: 提取全局静态 speaker identity embedding

**为什么这样设计**: 作者认为,CAMPPlus 已经编码了 speaker identity,所以 Q-Former 可以专注于提取 speaker identity 之外的动态 speaking style (语速、韵律轮廓等) [论文原文]。这种功能分工避免了两个模块编码冗余信息 [agent 解读: 类似于 information bottleneck 的设计哲学 -- 用独立信息源约束各模块的编码范围]。

消融实验验证了设计的必要性 [§4.6, Table 6]:
- 去掉 Q-Former condition tokens (w/o both) → CER 从 1.13% 升至 1.41% (test-zh), test-hc 退化 35% → Q-Former 对发音准确性 "不可或缺"
- 去掉 CAMPPlus (w/o spk) → SIM 全面下降 (test-hc: 0.8470 → 0.8355) → speaker embedding 提供互补的音色保真度
- 完整双路径在 content accuracy 和 speaker fidelity 之间取得最佳平衡

#### 2. Cross-sample Paired Training [§3.3.2]

**策略**: 对每个训练样本,用同一说话人的**不同内容**语音作为参考来提取 speaker embedding s 和 style condition c。

**为什么**: 迫使 conditioner 只编码与内容无关的说话人属性 (timbre, speaking style),而非 copy 参考音频的具体内容 [论文原文]。这是 speaker-style 解耦的基础,也为后续情感控制和方言合成提供了前提 -- 解耦后可以独立操控 style 维度 [agent 解读]。

#### 3. AR 生成的输入序列格式 [§3.3.2, Eq 4-5]

输入序列组成: `x = [s, c, e_BT, |lang|, |emo|, e_Text, e_ET, e_BA, e_Audio, e_EA]`

其中:
- `s`: CAMPPlus speaker embedding
- `c`: Q-Former 的 32 个 style condition tokens
- `|lang|`, `|emo|`: 语言和情感 control tags
- `e_Text`, `e_Audio`: text/audio token embeddings
- `e_BT/ET`, `e_BA/EA`: text/audio boundary markers

模型自回归预测 audio token 序列: $p(e_{Audio} | x_{<Audio}) = \prod_{i=1}^{N_s} p(e_{Audio,i} | x_{<Audio}, e_{Audio,<i})$ [Eq 5]

### 训练策略

#### Pre-training [§4.1]

- **数据**: 约 200K 小时中英文语音,来源公开渠道,经三阶段数据管线处理
- **AR Module**: 基于 Qwen3-0.6B (0.6B 参数)
- **CFM Decoder**: DiT backbone, ~300M 参数

#### Post-training [§3.3.3-3.3.5]

通过 SFT (supervised fine-tuning) 在特定数据上实现三种控制能力:

| 能力 | 数据量 | 控制方式 | 类别数 |
| --- | --- | --- | --- |
| 情感控制 [§3.3.3] | ~2,200h | `\|emo\|` tag | 7 主类 + 4 扩展 |
| 副语言 [§3.3.4] | ~200h | 文本中嵌入标签 / 隐式推断 | 5 (LAUGH/BREATH/CRY/COUGH/LAUGH_SPAN) |
| 方言合成 [§3.3.5] | ~16,000h | `\|lang\|` tag | 14 种中文方言 |

**方言合成的巧妙设计** [§3.3.5]: 
- 发现预训练后模型即使方言 prompt 也能稳定生成普通话 → 利用此特性,为每个方言说话人用预训练模型合成 3 条普通话,构建 "方言-普通话" 伪平行数据
- 微调时 mixed-prompt sampling: 目标始终为方言语音,参考以 50% 概率来自普通话或方言 → 迫使模型从风格多样的 prompt 中提取说话人身份,而非复制 prompt 风格 [论文原文]

### 数据处理管线 [§2, Fig 2]

三阶段管线,全部使用公开工具:

**Stage 1 -- Quality Assessment & Enhancement** [§2.1]:
- 采样率统一 → SAD + SCD (speaker change detection) 分段 [pyannote]
- DNSMOS 预测感知质量 + SenseVoiceSmall 语音/非语音分类 + SNR 估计
- 低质量片段经 resemble-enhance 去噪增强

**Stage 2 -- Label Annotation** [§2.2]:
- 多 ASR 系统 (Paraformer + FireRedASR + Whisper + 内部模型) 交叉验证转写
- OSD (overlapping speech detection) [pyannote/segmentation-3.0] + forced alignment
- Qwen3-Force-Alignment 韵律标注 + 3D-Speaker-Toolkit 说话人标记
- 频谱 rolloff 分析检测低带宽录音

**Stage 3 -- Quality Filtering** [§2.3]:
- 截断检测 + 合成语音检测 (防止爬取数据中混入 TTS 生成的音频)
- 多维联合筛选 (声学质量 + 语音有效性 + 转写可靠性 + 重叠 + 说话人一致性 + 截断风险 + 合成可能性 + 频谱质量)
- 被排除样本保留元数据,不丢弃,支持未来不同质量要求的数据集构建

## 实验

| 指标 | PilotTTS | CosyVoice 3-0.5B | Seed-TTS | Qwen3-TTS-0.6B | MiniMax-Speech | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CER (%) test-zh | **0.87** | 1.16 | 1.12 | 1.18 | 0.83 | SEED-TTS-Eval | [Table 1] |
| SIM test-zh | **0.862** | 0.780 | 0.796 | -- | -- | SEED-TTS-Eval | [Table 1] |
| WER (%) test-en | **1.50** | 2.02 | 2.25 | 1.64 | 1.65 | SEED-TTS-Eval | [Table 1] |
| SIM test-en | **0.815** | 0.718 | 0.762 | -- | -- | SEED-TTS-Eval | [Table 1] |
| 情感成功率 (主类) | **88.1%** | 83.8% | -- | -- | -- | 主观评测 | [Table 2] |
| 情感成功率 (全类) | 85.7% | 82.5% | -- | -- | -- | 主观评测 | [Table 2]^1^ |
| 情感SIM w/ control | **0.7329** | 0.6940 | -- | -- | -- | 主观评测 | [Table 3] |
| 副语言整体成功率 | **85.1%** | 80.4% | -- | -- | -- | 主观评测 | [Table 4] |

^1^ 注: 论文 Table 2 计算得出 PilotTTS Avg.(All)=85.7%, CosyVoice 3=82.5%,但正文 §4.3 写 "CosyVoice 3 leads marginally (81.4% vs. 80.2%)",数值与 Table 不一致,原始论文可能存在 typo。本笔记采用 Table 值。

> 注: SIM 评估使用 speaker embedding cosine similarity [§4.1],但论文未指定使用哪个 speaker encoder 模型。不同 encoder 的 SIM 结果可差 0.1-0.3,影响跨论文可比性。

**关键发现**:

1. **数据效率**: PilotTTS 仅用 200K h 数据,SIM 超越使用百万级数据的 Seed-TTS (test-zh: +0.066, test-en: +0.053),验证了数据质量 > 数据规模的论点 [§4.2]

2. **Speaker similarity 优势显著**: 在两个测试集上 SIM 均大幅领先第二名,作者归因于双路径 conditioning 同时捕获静态音色和动态风格 [§4.2]

3. **情感控制下 speaker similarity 保持最好**: 有/无情感控制的 SIM drop 最小 (0.8101 → 0.7329),说明解耦设计在调制情感时能较好保留说话人音色 [Table 3]

4. **副语言独有能力**: PilotTTS 是唯一支持 LAUGH_SPAN (94.6%) 和 CRY (61.9%) 的系统 [Table 4]

5. **消融验证 conditioning 设计** [Table 6]:
   - Q-Former condition tokens 是发音准确性的关键 (去掉后 test-hc CER 退化 35%)
   - CAMPPlus speaker embedding 一致性地改善 speaker similarity
   - 双路径在 accuracy 和 fidelity 间取得最佳平衡

## 局限性

1. **无显式 style 建模**: Q-Former 隐式捕获风格,缺乏强表达力的专用 style module,细粒度表达细节可能受限 [§5, 作者自述]
2. **单码本信息容量上限**: FSQ 单码本 (6561 entries) 架构简单但信息容量低于多码本 RVQ,难以扩展到歌唱和背景音乐等高复杂度音频 [§5, 作者自述]
3. **Mel + vocoder 间接重建**: 不是端到端波形生成,中间 mel spectrogram 表示引入额外失真 [§5, 作者自述]
4. **评估局限**: 零样本评估仅用 SEED-TTS-Eval,未提供 MOS 主观自然度评分;情感/副语言/方言评估采用定制测试集,与其他系统不完全可比 [agent 解读]
5. **训练数据未开源**: 虽然管线工具都是公开的,但具体的 200K h 数据源和筛选结果未开放 [agent 解读]
6. **Qwen3-TTS 和 MiniMax-Speech 未报 SIM**: 无法与这两个内容准确性最强的系统做完整比较 [Table 1 脚注]

## 点评

**优势**:
- **工程导向的高效路线**: PilotTTS 的价值不在于提出全新架构,而在于证明 "用公开组件 + 严谨数据工程" 可以达到 SOTA 水平。这对资源受限团队具有强参考价值。
- **Q-Former 双路径 conditioning 有新意**: 从视觉-语言模型 (BLIP-2) 借鉴 Q-Former 到 TTS speaker conditioning,通过 cross-sample paired training 实现自然的 speaker-style 解耦,设计简洁优雅。
- **数据管线可复现性**: 三阶段管线每个模块都标注了具体工具来源 (DNSMOS, SenseVoiceSmall, pyannote, resemble-enhance 等),是少见的高透明度数据工程报告。
- **多维控制能力整合**: 在同一框架内支持情感/副语言/方言,通过 post-training SFT 实现,不需要额外模型。

**不足**:
- **对 CosyVoice 3 组件的依赖**: 直接复用 CosyVoice 3 的 tokenizer + HiFi-GAN,系统的技术独立性有限;如果 CosyVoice 3 tokenizer 有局限,PilotTTS 同样受限。
- **SIM 评估方法未详细说明**: 使用哪个 speaker encoder 计算 cosine similarity?不同 encoder 的 SIM 结果可差 0.1-0.3 (参见 [[SpeakerEmbedding]] 关于 SECS 的讨论),需注意跨论文可比性。
- **200K h 数据的来源模糊**: "collected from publicly available sources" 未具体列出数据集名称,实际复现仍有一定障碍。

## 可复用的 idea

1. **Cross-sample paired training 实现 speaker-style 解耦**: 训练时参考音频来自同一说话人的不同内容,迫使 conditioner 只编码说话人属性。设计简单但有效,可应用于任何需要 speaker-content 解耦的 TTS/VC 系统。

2. **Q-Former 从 w2v-BERT 提取固定长度 style tokens**: 32 个可学习 query 通过 cross-attention 从变长语音表征中提取固定长度条件 tokens。这种 "信息瓶颈式" 压缩可迁移到其他需要从变长音频提取固定表示的场景。

3. **方言合成的伪平行数据构建**: 利用预训练模型为方言说话人生成普通话语音,构建 "方言-普通话" 对,缓解方言数据稀缺;配合 mixed-prompt sampling 进一步增强泛化。

4. **三阶段数据管线设计**: Quality Assessment → Label Annotation → Quality Filtering 的分层设计,被排除样本保留元数据不丢弃,支持灵活复用。特别是合成语音检测步骤对大规模爬取数据的清洗很有价值。

5. **CAMPPlus + Q-Former 双路径 conditioning**: 将 speaker identity (静态) 和 speaking style (动态) 分别用不同模块和不同粒度编码,再联合输入 LLM。这种功能分工 + 互补的设计思路可推广到其他多条件生成任务。

---

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes
> - [medium] 论文 Table 2 与正文 §4.3 的 overall emotion 成功率数值不一致 → 已在实验表格添加脚注标注
> - [low] SIM 评估未指定 speaker encoder 型号 → 已在实验表格添加注释
> 详见 `_review/PilotTTS-review.yml`
