---
type: paper
tier: deep
title: "Amphion: An Open-Source Audio, Music, and Speech Generation Toolkit"
arxiv_id: "2312.09911"
source: "Sources/Amphion.pdf"
authors: [Xueyao Zhang, Liumeng Xue, Yicheng Gu, Yuancheng Wang, Jiaqi Li, Xi Chen, Haorui He, Chaoren Wang, Songting Liu, Zihao Fang, Haopeng Chen, Tze Ying Tang, Jun Han, Kai Chen, Haizhou Li, Junan Zhang, Lexiao Zou, Mingxuan Wang, Zhizheng Wu]
year: 2024
venue: "arXiv (ICASSP 2024 companion)"
tags: [toolkit, open-source, TTS, text-to-audio, singing-voice-conversion, vocoder, audio-codec, unified-framework, reproducible-research]
concepts: ["[[NeuralVocoder]]", "[[DiffusionModel]]", "[[Text-to-SpeechPipeline]]", "[[TTSEvaluation]]", "[[SingingVoiceSynthesis]]", "[[VariationalAutoencoderforTTS]]"]
models: ["[[VITS]]", "[[NaturalSpeech2]]", "[[BigVGAN]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[NeuralVocoder]], [[VITS]], [[NaturalSpeech2]], [[Zero-shotSpeechSynthesis]], [[BigVGAN]], [[TTSEvaluation]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Amphion 是一个统一 toolkit 而非单一模型。它整合了知识库中多个已有实体 -- 包括 vocoder 家族 (HiFi-GAN/BigVGAN/DiffWave/WaveNet, 见 [[NeuralVocoder]])、端到端 TTS 模型 (VITS, 见 [[VITS]])、零样本 TTS (VALL-E/NaturalSpeech 2, 见 [[Zero-shotSpeechSynthesis]]、[[NaturalSpeech2]])、GAN vocoder (见 [[BigVGAN]]) 以及评估指标体系 (见 [[TTSEvaluation]])。在开源 toolkit 生态中,Amphion 的定位是覆盖 audio/music/speech 三域的统一框架,区别于 ESPnet (侧重 ASR+TTS)、SpeechBrain (侧重 speech processing)、AudioCraft (侧重 audio/music generation) 等单领域工具。
>
> **已有认知**: (1) Neural vocoder 已从 AR (WaveNet) 演进到 GAN (HiFi-GAN) 和 diffusion (DiffWave),HiFi-GAN 是 2020-2023 事实标准 [NeuralVocoder]; (2) VITS 首次证明端到端单模型 TTS 可超越两阶段系统 [VITS]; (3) NaturalSpeech 2 用 latent diffusion + speech prompting 实现零样本 TTS [NaturalSpeech2]; (4) Zero-shot TTS 已有 LLM+token、diffusion、coarse-to-fine 三条主流路线 [Zero-shotSpeechSynthesis]; (5) TTS 评估面临 MOS ceiling、指标不可比等系统性挑战 [TTSEvaluation]。
>
> **创新判断**: Amphion 的贡献不在于提出新模型架构,而在于**基础设施层面的统一与标准化** -- 将分散在不同仓库的模型纳入同一框架,统一数据处理、训练流程和评估方式,降低复现门槛。这填补了"跨任务统一 + 公平比较"的工具层空白。
>
> 检索命中: [[NeuralVocoder]]✓, [[VITS]][待确认], [[NaturalSpeech2]][待确认], [[Zero-shotSpeechSynthesis]]✓, [[BigVGAN]][待确认], [[TTSEvaluation]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 统一 TTS/TTA/SVC/Vocoder/Codec 五类任务的开源 toolkit,提供标准化训练流程和评估指标,目标是降低音频生成研究的复现门槛
> - **路线**: Text/Audio Input → 统一数据处理层 → 任务框架层 (TTS/TTA/SVC) → 模型架构层 → Vocoder/Codec → Audio Output + Evaluation Suite
> - **指标**: Multi-speaker TTS MOS 3.61 (VITS, 与 Coqui/ESPnet 持平) [Table 4]; Zero-shot VALL-E SIM-O 0.51 / WER 0.034 (匹配官方) [Table 5]; TTA FD 20.47 / IS 8.78 (优于官方 AudioLDM) [Table 6]; SVC MOS 3.52 / SMOS 3.69 (优于 SoftVC) [Table 7]
> - **可借鉴**: 四层系统架构设计 (共享底层 → 任务层 → 模型层 → Recipe/Demo) 可作为构建多任务音频研究平台的参考范式; 统一评估 pipeline 的组织方式
> - **局限**: v0.1 各任务仅集成 1-4 个代表模型,覆盖面有限; 未包含 flow matching 等 2024 主流方法; 论文未报告大规模数据训练实验; 仅为框架介绍论文,技术深度有限

## 核心问题

Amphion 试图解决音频生成研究领域的三个实践问题 [§1]:
1. **复现不一致**: 不同开源仓库对同一算法的实现差异导致性能不可比
2. **流程不完整**: 多数仓库只关注模型架构,忽略数据预处理、特征提取、训练细节和系统评估
3. **入门门槛高**: 分散的仓库和缺乏系统指导对新手不友好

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Amphion 的系统架构采用**四层自底向上设计** [§2.1, Fig 2]:

1. **共享构建块层**: Dataset, Feature Extractor, Sampler, DataLoader, Optimizer, Scheduler, Trainer, 通用 Module -- 所有任务共用 [论文原文]
2. **任务框架层**: 每个生成任务统一 TaskLoader (数据/特征用法), TaskFramework (任务框架), TaskTrainer (训练流程) [论文原文]
3. **模型架构层**: 每个具体模型指定 ModelArchitecture + ModelTrainer [论文原文]
4. **用户层**: Recipe (模型配方) + Pre-trained Models + Interactive Demos + Educative Visualizations [论文原文]

[agent 解读] 这种分层设计的核心思路是: 将跨任务的通用逻辑下沉到底层共享,使得新增一个任务或模型时只需实现差异化部分,不必重建数据处理和训练循环。这是标准的 framework 设计模式,但在音频生成领域此前缺乏跨 TTS/TTA/SVC 统一的实现。

### 任务分类体系

从输入角度将音频生成统一为三类 [§2]:

| 类别 | 输入 | 代表任务 | 约束强度 |
|------|------|----------|----------|
| Text → Waveform | 离散文本 token | TTS, SVS | 严格约束内容 |
| Descriptive Text → Waveform | 描述性文本 | TTA, TTM | 宽泛引导风格/内容 |
| Waveform → Waveform | 连续波形 | VC, SVC, EC, AC, ST | 变换保留部分特征 |

[agent 解读] 这种三分法的价值在于统一了"输入类型 → 约束强度"的对应关系: Text→Waveform 的文本严格约束语音内容, Descriptive Text 仅引导, Waveform→Waveform 保留源信号大部分结构。这为框架设计中数据处理和条件注入方式的统一提供了概念基础。

### 关键设计选择

**为什么采用两阶段生成架构?** 多数音频生成模型先生成 mel spectrogram 等中间特征,再用 vocoder/codec 生成波形 [§2.2]。[论文原文] Amphion 因此将 vocoder 和 codec 作为独立模块集成,支持 GAN-based (HiFi-GAN, BigVGAN, MelGAN), autoregressive (WaveNet, WaveRNN), flow-based (WaveGlow), diffusion-based (DiffWave) 等多种 vocoder,以及 FACodec 等 neural codec [Table 1]。

**为什么选择这些代表模型?** v0.1 从三类任务中各选一个代表 (TTS, TTA, SVC) 进行集成 [§2.2]。[论文原文] 这是为了确保框架能适应不同任务类型,为后续扩展打基础。

**评估指标的统一**: Amphion 内置了 F0 相关指标 (FPC, V/UV F1), 频谱失真指标 (PESQ, STOI, FAD, MCD, SI-SNR, SI-SDR), 可懂度指标 (WER, CER), 说话人相似度 (cosine similarity, Resemblyzer, WavLM) [Table 1, 右列]。[agent 解读] 统一评估套件是 toolkit 论文的核心差异化 -- 它使不同模型在同一评估条件下可公平比较。

### 训练策略

各任务的训练配置 [§3]:

| 模型 | 参数量 | 训练数据 | 数据规模 |
|------|--------|----------|----------|
| VITS (TTS) | 30M | HiFi-TTS | -- |
| VALL-E (ZS-TTS) | 250M | MLS | 45K h |
| NaturalSpeech 2 (ZS-TTS) | 201M | Libri-light | -- |
| AudioLDM (TTA) | 710M | AudioCaps | -- |
| DiffWaveNetSVC (SVC) | 31M | Opencpop+SVCC+VCTK+OpenSinger+M4Singer | 170.3 h |
| FACodec (Codec) | 140M | Libri-light | -- |
| HiFi-GAN (Vocoder) | 24M | LibriTTS | ~600 h |
| BigVGAN (Vocoder) | 112M | LibriTTS | ~600 h |

[Table 2, §3.1-3.4]

## 实验

### Multi-Speaker TTS

| 指标 | Amphion v0.1 (VITS) | Coqui TTS (VITS) | SpeechBrain (FS2) | TorToiSe | ESPnet (VITS) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| CER ↓ | 0.06 | 0.06 | 0.06 | 0.05 | 0.07 | [Table 4] |
| WER ↓ | 0.10 | 0.12 | 0.11 | 0.09 | 0.11 | [Table 4] |
| FAD ↓ | 0.84 | 0.54 | 1.71 | 1.90 | 1.28 | [Table 4] |
| MOS ↑ | 3.61 | 3.69 | 3.54 | 3.61 | 3.57 | [Table 4] |

### Zero-Shot TTS (VALL-E Continuation)

| 指标 | Amphion v0.1 | Proprietary (VALL-E) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SIM-O ↑ | 0.51 | 0.51 | LibriSpeech test-clean | [Table 5] |
| WER ↓ | 0.034 | 0.038 | LibriSpeech test-clean | [Table 5] |

注: 训练数据和测试 duration 不完全一致 -- Amphion 用 MLS 45K h (10-20s) 训练,测试 10-20s; 官方用 Libri-Light 60K h,测试 4-10s [§3.1.2]。

### Text to Audio

| 指标 | Amphion v0.1 (AudioLDM) | Official AudioLDM | Diffsound | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| FD ↓ | 20.47 | 27.12 | 47.68 | AudioCaps test | [Table 6] |
| IS ↑ | 8.78 | 7.51 | 4.01 | AudioCaps test | [Table 6] |
| KL ↓ | 1.44 | 1.86 | 2.52 | AudioCaps test | [Table 6] |

### Singing Voice Conversion

| 指标 | Amphion v0.1 (DiffWaveNetSVC) | SoftVC (VITS) | Ground Truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS ↑ | 3.52 | 2.98 | 4.67 | SVCC 2023 | [Table 7] |
| SMOS ↑ | 3.69 | 3.43 | 3.96 | SVCC 2023 | [Table 7] |

### Vocoder

| 指标 | Amphion v0.1 (HiFi-GAN) | Official HiFi-GAN | ESPnet HiFi-GAN | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| PESQ ↑ | 3.55 | 3.43 | 3.55 | LibriTTS | [Table 8] |
| M-STFT ↓ | 1.09 | 1.98 | 1.12 | LibriTTS | [Table 8] |
| F0RMSE ↓ | 188 | 177 | 188 | LibriTTS | [Table 8] |
| FPC ↑ | 0.88 | 0.88 | 0.86 | LibriTTS | [Table 8] |

[论文原文] Amphion HiFi-GAN 得益于 Time-Frequency Representation-based Discriminators (Gu et al., ICASSP 2024) 的辅助,在频谱重建上优于官方版本 [§3.4]。

## 局限性

1. **技术深度有限**: 作为 toolkit 介绍论文,未深入分析任何一个模型的设计决策或改进,主要是集成和对比 [agent 解读]
2. **模型覆盖面窄**: v0.1 仅集成少数代表模型,缺少 2024 主流方法 (flow matching TTS, masked generative 等) [agent 解读]
3. **实验条件不完全可比**: VALL-E 复现的训练数据 (MLS 45K h) 和测试 duration (10-20s) 与官方不同,SIM-O 匹配但因果关系不确定 [Table 5, §3.1.2]
4. **评估局限**: 主观评估仅 10 名有经验听众,每条件 10 个样本,统计效力有限 [§3]
5. **无大规模训练验证**: 未展示在大规模数据 (>100K h) 上训练的效果,与 2024-2025 大模型 TTS 趋势脱节 [agent 解读]
6. **框架演进缺失**: 论文基于 v0.1,未讨论后续版本的架构变化和新增能力 [agent 解读]

## 点评

Amphion 是 CUHK-SZ 吴志正团队发布的音频生成统一 toolkit。作为系统/工具论文而非方法论文,其核心贡献在基础设施层:

**价值**: (1) 将 TTS/TTA/SVC/Vocoder/Codec 纳入同一框架,解决了跨仓库复现不一致的痛点; (2) 内置统一评估 pipeline,使公平比较成为可能; (3) 实验证明框架复现质量与原始仓库/论文持平或更优,验证了框架的可靠性。

**局限**: 论文更像一份"产品说明书"而非研究贡献 -- 没有提出新方法、新架构或新发现。v0.1 的模型集成较为基础 (VITS/VALL-E/AudioLDM/DiffWaveNetSVC),没有覆盖当时已出现的 flow matching、masked generative 等新范式。

**生态角色**: Amphion 团队后续发布了 Emilia 大规模数据集、MaskGCT 等重要工作,Amphion 实际上是这些后续工作的基础平台。从生态视角看,Amphion 的价值不在于这篇论文本身,而在于它作为持续迭代平台承载了一系列高影响力工作。

## 可复用的 idea

1. **四层系统架构**: 共享底层 (数据/训练) → 任务框架层 → 模型架构层 → 用户层 (Recipe/Demo) 的分层设计,适用于构建任何多任务研究平台 [§2.1]
2. **三类任务统一视角**: 按输入约束强度 (Text/Descriptive Text/Waveform) 统一不同音频生成任务的分类框架 [§2]
3. **TF-Rep Discriminators**: 集成 Time-Frequency Representation Discriminators (MS-SB-CQTD, MSSTFTD) 提升 HiFi-GAN 频谱重建质量的实践 [§3.4, Table 8]
4. **统一评估套件**: 将 F0 指标、频谱失真、可懂度、说话人相似度等异构指标整合在同一框架中的组织方式 [Table 1]
