---
type: paper
tier: deep
title: "Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation"
arxiv_id: "2407.05361"
source: "Sources/Emilia.pdf"
authors: [Haorui He, Zengqiang Shang, Chaoren Wang, Xuyuan Li, Yicheng Gu, Hua Hua, Liwei Liu, Chen Yang, Jiaqi Li, Peiyang Shi, Yuancheng Wang, Kai Chen, Pengyuan Zhang, Zhizheng Wu]
year: 2024
venue: "IEEE SLT 2024"
tags: [dataset, large-scale, multilingual, in-the-wild, speech-generation, TTS, preprocessing-pipeline, open-source]
concepts: ["[[SpeechTokenizer]]", "[[Self-SupervisedSpeechRepresentation]]"]
models: ["[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: ["[[数据集/Emilia|Emilia]]"]
kb_context_sources: 1
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页: [[SpeechTokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **[[SpeechTokenizer]]** (confirmed): Speech Tokenizer 将连续波形转为离散 token。本文不直接涉及 tokenizer 设计,但 Emilia 数据集是训练高质量 speech tokenizer 和 TTS 模型的关键数据来源。Survey 发现 domain-specific 训练数据对 tokenizer/TTS 质量至关重要 — Emilia 的 in-the-wild 多样性正是解决现有 audiobook 数据风格单一问题的关键。
>
> **[[Self-SupervisedSpeechRepresentation]]** [待确认]: Emilia 数据集的 diversity 分析使用了 WavLM 提取 acoustic features 和 Sentence-BERT 提取 semantic features,展示了 SSL 表征在数据集分析中的应用。
>
> 检索命中: [[SpeechTokenizer]] | 过滤: [[Self-SupervisedSpeechRepresentation]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个大规模、多语言、多样化的开源 in-the-wild 语音数据集 (101K+ 小时, 6 语种) + 首个开源预处理 pipeline (Emilia-Pipe),专为训练自然、自发的语音生成模型设计
> - **路线**: In-the-wild 音频 → Standardization (24kHz, mono, -20dBFS) → Source Separation (UVR-MDX-Net) → Speaker Diarization (pyannote) → VAD 细粒度切分 (Silero-VAD, 3-30s) → ASR (WhisperX) → Quality Filtering (语种/DNSMOS/duration) → 标注数据
> - **指标**: 处理后 DNSMOS P.835 OVRL 3.26 (vs MLS 3.33); Emilia-Test AR WER 6.6% + SIM-O 0.618 + FSD 12.73 (vs MLS-trained: WER 8.4% + SIM-O 0.577 + FSD 24.73); 多语种 TTS 6 语种均有效 [Table 4, 5]
> - **可借鉴**: (1) 完整的 in-the-wild 音频预处理 pipeline 开源可复用; (2) DNSMOS P.835 OVRL ≥ 3.0 作为自动质量筛选阈值; (3) 双 ASR 信心 + duration outlier 过滤策略; (4) 处理速度 2.5h/min — 大规模数据处理方案
> - **局限**: 无人工转写(全 ASR 标注); 无细粒度标签(情感/风格/韵律); 24kHz 采样率 (非 44.1kHz); 部分语种规模偏小 (德/法/日/韩 各 ~1-2K h)

## 核心问题

当前语音生成模型主要在 audiobook 数据 (如 LibriTTS, MLS) 上训练,这些数据以正式朗读风格为主,缺乏真实人类口语中的自发性、多样性和自然度 [§1]。核心问题是: **如何构建一个大规模、多语言、多样化的 in-the-wild 语音数据集,使语音生成模型能够生成更自然、更自发的语音?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构: Emilia-Pipe 预处理 Pipeline [§2, Fig 1]

Emilia-Pipe 是一个 6 步开源预处理流水线,将 in-the-wild 原始音频转换为高质量的带标注训练数据 [§2]:

```
原始音频 → [1. Standardization] → [2. Source Separation] → [3. Speaker Diarization]
→ [4. Fine-grained Segmentation by VAD] → [5. ASR] → [6. Filtering] → 标注数据
```

#### Step 1: Standardization [§2.1]
- 统一格式: WAV, mono, 24 kHz, 16-bit
- 音量归一化: target -20 dBFS, 限制 ±3 dB
- 波形归一化: 除以最大振幅 [§2.1]

#### Step 2: Source Separation [§2.2]
- 使用 Ultimate Vocal Remover (UVR-MDX-Net Inst 5) 分离人声与背景音乐/噪声 [§2.2]
- [论文原文] 背景噪声和音乐会负面影响语音生成性能 [§2.2]

#### Step 3: Speaker Diarization [§2.3]
- 使用 pyannote/speaker-diarization-3.1 实现说话人分割 [§2.3]
- 三个组件: speaker segmentation + speaker embedding + clustering
- 确保每段音频仅包含单个说话人,与现有数据集兼容 [§2.3]

#### Step 4: Fine-grained Segmentation by VAD [§2.4]
- 使用 Silero-VAD 进一步切分超过 30 秒的段落 [§2.4]
- 拼接同一说话人的 VAD 段落,确保每段 3-30 秒 [§2.4]

#### Step 5: ASR [§2.5]
- 使用 Whisper-Medium via WhisperX + faster-whisper + CTranslate2 推理引擎 [§2.5]
- 速度: 比官方 Whisper 快 4x [§2.5]
- 复用 Step 4 的 VAD 结果,避免重复计算 [§2.5]

#### Step 6: Filtering [§2.6]
三层过滤 [§2.6]:
1. **语种过滤**: 丢弃非目标语种或 model language confidence < 80% 的数据
2. **质量过滤**: DNSMOS P.835 OVRL ≥ 3.0 [§2.6]
3. **Duration outlier 过滤**: 平均 phone duration 超出 1.5×IQR 的段落丢弃 [§2.6]

**处理效率** [§2.7]: 600 小时原始数据 → 176.22 小时 (29.43%),处理速度约 2.5 小时/分钟 (8×RTX 4090, ~4 小时) [§2.7, Table 2]

### 关键设计选择

#### 1. In-the-wild vs Audiobook [§1]

[论文原文] Audiobook 数据 (LibriTTS, MLS) 以正式朗读风格为特征,真实口语中的停顿、呼吸、语速变化、情感表达等自然现象很少出现 [§1]。Emilia 从视频平台和播客采集,涵盖访谈、辩论、体育解说、有声书等,捕捉了广泛的真实说话风格 [§3.1]。

#### 2. 为何选择 DNSMOS P.835 OVRL 作为质量阈值? [§2.6, §2.7]

[论文原文] DNSMOS P.835 OVRL 是非侵入式语音质量评估指标,与人类评分高度相关 [§2.6]。阈值 3.0 在保留率和质量之间取得平衡: 原始数据平均 2.50,过滤后提升至 3.26 [Table 2]。

#### 3. 开源 pipeline 设计 [§1]

[论文原文] 现有 pipeline (AutoPrepWild, WenetSpeech4TTS) 依赖私有模型,处理速度未公开,且仅支持单语种。Emilia-Pipe 是首个全开源、多语种、高效的 pipeline [§1]。

### Emilia 数据集 [§3]

- **规模**: 101,654 小时 [§3.1]
- **语种**: 英语 (46.8K h), 中文 (49.9K h), 德语 (1.6K h), 法语 (1.8K h), 日语 (1.7K h), 韩语 (0.2K h) [§3.1, Fig 2]
- **来源**: 多样化视频平台和播客 — 访谈、辩论、体育解说、有声书 [§3.1]
- **采样率**: 24 kHz [§2.1]

#### 质量分析 [§3.2.1]
DNSMOS P.835 OVRL 3.26 ± 0.14,在 9 个数据集中排第三 (仅次于 MLS 3.33 和 Libri-Light 3.25) [Table 3]。[论文原文] 尽管来自 in-the-wild,经 Emilia-Pipe 处理后质量可比拟 studio recording 和 audiobook 数据集 [§3.2.1]。

#### 多样性分析 [§3.2.2]
- **Acoustic diversity**: WavLM features → PCA → 散点图显示 Emilia 比 MLS 分布更广 [Fig 3(a)]
- **Semantic diversity**: Sentence-BERT text features → PCA → Emilia 覆盖更广语义空间 [Fig 3(b)]
- [论文原文] Emilia 在 acoustic 和 semantic 特征空间上都展示了显著更高的多样性 [§3.2.2]

## 实验

| 指标 | AR (Emilia) | AR (MLS) | NAR (Emilia) | NAR (MLS) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER ↓ | 6.6% | 7.7% | 7.4% | 8.2% | Emilia-Test | [Table 4] |
| SIM-O ↑ | 0.618 | 0.587 | 0.601 | 0.528 | Emilia-Test | [Table 4] |
| FSD ↓ | 12.73 | 20.76 | 14.07 | 15.94 | Emilia-Test | [Table 4] |
| CMOS ↑ | 0.19 | -0.36 | 0.28 | 0.36 | LibriSpeech-Test | [Table 4] |
| SMOS ↑ | 3.73 | 3.13 | 3.76 | 3.62 | Emilia-Test | [Table 4] |

**Baselines** [§4.1.1]:
- **AR (AR+SoundStorm)**: Autoregressive semantic token prediction → SoundStorm acoustic token generation [§4.1.1]
- **NAR (VoiceBox)**: Flow-matching based non-autoregressive speech synthesis [§4.1.1]

**关键发现**:
1. **Emilia vs MLS on Emilia-Test**: Emilia 训练的模型在自发语音测试集上全面超越 MLS 训练的模型,尤其是 FSD 显著改善 (12.73 vs 20.76),说明 in-the-wild 数据显著提升自然语音生成能力 [Table 4, §4.2]
2. **Emilia vs MLS on LibriSpeech-Test**: 在正式朗读风格测试集上,两者在 WER 和 SMOS 上接近,Emilia 不牺牲正式语音质量 [Table 4, §4.2]
3. **AR 模型从 diverse data 获益更大**: AR 模型在 Emilia-Test 上的 FSD 提升 (20.76→12.73) 比 NAR (15.94→14.07) 更显著 [Table 4],表明 AR TTS 更依赖训练数据的多样性 [论文原文]
4. **多语种 TTS** [Table 5]: 6 语种均展示合理性能,中文 WER 最低 (4.1%),英语 WER 次之 (5.8-6.2%),验证了数据集的多语种有效性

## 局限性

- **无人工转写**: 全部依赖 Whisper ASR 自动标注,转写准确率受限于 ASR 性能,尤其对方言、口语化表达、歌唱等场景 [agent 解读]
- **无细粒度标签**: 不含情感、风格、韵律、副语言等标签,限制了可控 TTS 训练的直接适用性 [agent 解读]
- **小语种规模有限**: 德/法/日/韩 各仅 1-2K 小时,与英/中 (各 ~47-50K h) 差距大 [Fig 2]
- **版权和伦理**: 从公开视频/播客采集,可能包含版权内容,论文未详细讨论数据许可 [agent 解读]
- **采样率**: 24 kHz 低于音乐领域常用的 44.1/48 kHz [§2.1]

## 点评

1. **基础设施级贡献**: Emilia 填补了语音生成领域的关键数据空白。现有 TTS 模型在 audiobook 数据上训练导致合成语音"太正式"，Emilia 提供了大规模 in-the-wild 数据支撑自然语音生成。101K 小时的规模远超此前所有开源数据集 [Table 1] [agent 解读]。

2. **Pipeline 的工程价值**: Emilia-Pipe 将原始音频到训练数据的完整流程标准化和开源,处理速度 2.5h/min 意味着每天可处理 ~3600 小时原始数据 (8×4090),使社区可以快速构建自己的大规模数据集 [agent 解读]。

3. **In-the-wild 数据的有效性验证**: 实验充分证明 in-the-wild 数据不仅不损害正式语音质量,还显著提升自然语音生成。AR 模型在 Emilia-Test 上 FSD 从 20.76 降至 12.73,这一改善幅度是决定性的 [agent 解读]。

4. **数据集论文的典范**: 从 pipeline 设计、质量分析、多样性分析到下游任务验证,论文结构完整,可复现性高 [agent 解读]。

5. **与 NVSpeech 的互补**: NVSpeech 使用了 Emilia 的子集作为其自动标注数据的来源之一,体现了 Emilia 作为基础数据设施的生态位 [agent 解读]。

## 可复用的 idea

1. **Emilia-Pipe 6 步预处理 pipeline**: 可直接复用于任何语种的 in-the-wild 音频到 TTS 训练数据的转换
2. **DNSMOS P.835 OVRL ≥ 3.0 作为自动质量阈值**: 非侵入式语音质量指标在大规模数据过滤中的标准化应用
3. **WavLM + Sentence-BERT 多样性分析方法**: acoustic + semantic 双维度 PCA 可视化评估数据集多样性
4. **Duration outlier 过滤**: 平均 phone duration 的 1.5×IQR 过滤规则可复用于任何 ASR 标注的语音数据
5. **Source separation + Speaker diarization + VAD 三级分割**: 从混合音频到单说话人短片段的标准流程
