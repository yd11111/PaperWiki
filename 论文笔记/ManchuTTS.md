---
type: paper
tier: deep
title: "ManchuTTS: Towards High-Quality Manchu Speech Synthesis via Flow Matching and Hierarchical Text Representation"
arxiv_id: "2512.22491"
source: "Sources/ManchuTTS.pdf"
authors: [Suhua Wang, Zifan Wang, Xiaoxin Sun, D.J. Wang, Zhanbo Liu, Xin Li]
year: 2025
venue: "arXiv"
tags: [TTS, flow-matching, low-resource, endangered-language, non-autoregressive, hierarchical-representation, contrastive-learning, agglutinative]
concepts: ["[[ConditionalFlowMatching]]", "[[Non-autoregressiveTTS]]", "[[DurationPredictor]]", "[[MelSpectrogram]]", "[[PhonemeRepresentation]]", "[[ProsodyModeling]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]]✓, [[ProsodyModeling]]✓ | 过滤: [[Non-autoregressiveTTS]](pending-review), [[DurationPredictor]](pending-review), [[MelSpectrogram]](pending-review), [[PhonemeRepresentation]](pending-review) | 未命中但可能相关: 无

**谱系定位**: ManchuTTS 属于 Conditional Flow Matching (CFM) 在 TTS 中的应用,但定位与主流 CFM-TTS 系统显著不同。已有 CFM-TTS 工作(CosyVoice 系列、F5-TTS、Matcha-TTS、VoiceFlow 等)均面向高资源语言,且通常作为 coarse-to-fine pipeline 的 fine stage renderer。ManchuTTS 则将 CFM 用于极低资源的濒危语言场景(5.2h 训练数据),且不采用 speech token→mel 的两阶段架构,而是 phoneme→mel 的单阶段 NAR 生成。

**已有认知**:
- CFM 通过学习 ODE 向量场将高斯噪声映射到 mel spectrogram,推理步数少于 diffusion model [ConditionalFlowMatching]
- NAR TTS 通过 Duration Predictor + Length Regulator 实现并行生成,消除了 AR 的逐帧依赖 [Non-autoregressiveTTS, DurationPredictor]
- Prosody Modeling 在 TTS 中涵盖 duration/pitch/energy 三维度,可通过显式 predictor 或隐式生成模型建模 [ProsodyModeling]
- 音素表示是 TTS 前端的标准输入,IPA 可统一跨语言发音表示 [PhonemeRepresentation]

**创新判断**: ManchuTTS 的核心创新不在 CFM 本身,而在将 hierarchical linguistic conditioning (phoneme-syllable-prosody 三级文本表征) 注入 CFM 的向量场条件中。这一做法在 KB 中没有先例 —— 已有 CFM-TTS 均使用单层文本条件。此外,hierarchical contrastive alignment loss 用于强化多粒度文本-声学对应,也是 CFM-TTS 文献中未见的手段。

## 速查

> [!summary] 速查
> - **一句话**: 针对满语黏着特性设计三级文本表征(音素-音节-韵律)+层级对比损失,在仅 5.2h 数据上实现 MOS 4.52 的 CFM-based NAR TTS
> - **路线**: 满语文本→IPA phoneme+syllable+prosody 三级特征→Conv1D phoneme encoder→8 层 DiT + 三层 cross-modal attention→CFM ODE→80-dim mel spectrogram→vocoder
> - **指标**: MOS 4.52 (vs GT 4.68, F5-TTS 4.12, VITS 3.95); WER 12.4%; AWPA 92.7%; Prosodic Naturalness 89.4%; RTF 0.12; 86ms latency [Table III, IV, VI]
> - **可借鉴**: 三级文本表征+层级对比损失的思路可迁移到其他黏着语/低资源语言 TTS; cross-modal attention 的 self→cross→self 三层结构是轻量级多模态对齐方案
> - **局限**: 仅支持实验室录音,方言变体精度下降; 情感表达有限; 6.24h 数据集规模极小且尚未完全开源; 与主流系统(CosyVoice/VALL-E)的架构差距大,zero-shot 能力弱

## 核心问题

1. **满语 TTS 面临什么独特挑战?** 满语是 UNESCO 认定的"极度濒危"语言,全球流利使用者不足 100 人。作为黏着语,其词根+后缀的复杂形态变化(元音和谐、协同发音)对常规 TTS 模型构成语音学上的挑战。同时,数据极度稀缺(本文构建的首个数据集仅 6.24h),传统方法(Tacotron 2/FastSpeech 2)在此规模下效果退化严重 [§I]。

2. **为什么现有低资源 TTS 方法不够用?** LRSpeech 等方法假设有最低数据门槛,对真正的濒危语言不适用。F5-TTS 等 flow-based 模型缺乏对黏着语形态复杂性的设计。显式对齐方法在低资源条件下错误率达 25%-35%,duration predictor 产生僵硬节奏,NAR 架构难以捕捉精细韵律变化 [§I]。

3. **本文提出了什么解决方案?** ManchuTTS 框架,核心是将三级语言学条件 c = {c_phon, c_syll, c_pros} 注入 conditional flow matching 的向量场,使 ODE 路径受层级化文本特征引导。辅以三层 cross-modal attention 实现渐进式对齐,hierarchical contrastive loss 强化结构化声学-语言对应 [§II]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ManchuTTS 的 pipeline [Fig 3]:
```
满语文本 → IPA 音素化 (vocab=1024)
    ↓
三级特征提取:
  c_phon: phoneme-level IPA 特征 (元音和谐、协同发音)
  c_syll: syllable/word-level [root+suffix] 结构特征 (词缀分解、音节转换)
  c_pros: prosody-level 全局语调节奏特征 (句型相关 pitch 轮廓、能量分布)
    ↓
Conv1D Phoneme Encoder (2 layers, 256 channels)
    ↓
8-layer DiT backbone (512 dim, 4 heads) + 三层 cross-modal attention
    ↓
CFM ODE: x_t = (1-t)·x_0 + t·x_1, 学习 v_t(x_t|c;θ) ≈ x_1 - x_0
    ↓
80-dim Mel Spectrogram
    ↓
Vocoder → Waveform
```

**参数规模**: 86.4M (Conv1D encoder 2 层 + DiT 8 层 + LSTM duration predictor 3 层 128 hidden) [Table II]

### 关键设计选择

**设计 1: 三级文本表征 (Hierarchical Text Representation)** [§II-B]

[论文原文] 满语作为黏着语,单一音素级表征不足以捕捉其形态学复杂性。设计三级:
- **Phoneme level**: IPA 序列,捕捉细粒度语音特征(元音和谐、协同发音模式)
- **Syllable/word level**: 将复杂词缀分解为 [root+suffix] 结构,建模音节间转换和词根-后缀韵律耦合
- **Prosody level**: 预测全局语调和节奏模式,根据句子类型(如疑问句升调)调整 pitch 轮廓和能量分布

[agent 解读] 这三级对应了语言学中从 segmental 到 suprasegmental 的自然层级。与 FastSpeech 2 的 variance adaptor (显式 pitch/energy/duration predictor) 相比,本文将语言学先验显式编码为条件信号,让模型学习语言学层级与声学的对应关系,而不是单独预测韵律参数。

**设计 2: 三层 Cross-Modal Attention** [§II-A]

[论文原文] 遵循 "self-attention → interaction → self-attention" 范式实现渐进式跨模态对齐:

Layer 1 — Intra-modal Self-Attention [Eq. 4-5]:
- 文本特征 F_t 和声学特征 F_a 各自做 self-attention,增强模态内部表征

Layer 2 — Cross-modal Cross-Attention [Eq. 6-7]:
- 双向 cross-attention: F_t attend to F_a, F_a attend to F_t
- 实现互信息交换和特征空间融合

Layer 3 — Intra-modal Self-Attention [Eq. 8-9]:
- 再次 self-attention 巩固跨模态交互后的内部一致性

[agent 解读] 这种 self→cross→self 的三层结构类似 PerceiverIO 的设计思路,但更轻量。与 F5-TTS 直接在 DiT 中用 cross-attention 不同,ManchuTTS 把跨模态交互结构化为独立的三层模块,可能在低资源条件下有更好的收敛性。

**设计 3: Hierarchical Contrastive Alignment Loss (HCA)** [§II-C, Eq. 10]

[论文原文] 在 CFM 损失之外,引入层级对比对齐损失:

L_HCA = Σ_k λ_k · E[-log(exp(s(x_1, c^(k))) / (exp(s(x_1, c^(k))) + Σ exp(s(x_1, c_neg^(k)))))]

对每个层级 k (phoneme/syllable/prosody),构造正样本对 (x_1, c^(k)) 和负样本对 (x_1, c_neg^(k)),最大化正样本互信息。

[agent 解读] 这本质上是 InfoNCE loss 的层级化版本。在低资源场景下,对比学习可以提供额外的自监督信号,帮助模型学习文本-声学的结构化对应。与 CosyVoice 等使用 Classifier-Free Guidance 不同,HCA loss 在训练时直接强化对齐质量,而非推理时引导。

**设计 4: CFM 条件化** [§II, Eq. 1-3]

[论文原文] 线性插值路径 x_t = (1-t)·x_0 + t·x_1,向量场 u_t(x_t|c) = x_1 - x_0。训练目标:

L_CFM(θ) = E_{t,x_0,x_1,c} ||v_t(x_t|c;θ) - (x_1 - x_0)||^2

其中 c = {c_phon, c_syll, c_pros}。推理时从高斯噪声出发,求解 ODE 生成语音。

[agent 解读] 这是标准 OT-CFM 公式,创新在于条件 c 是三级层级化的。与 VoiceFlow/Matcha-TTS 等使用单层 phoneme hidden 做条件不同,ManchuTTS 将语言学层级结构显式注入 flow 的 vector field conditioning,使 ODE 路径由多粒度语言学特征联合塑造。

### 训练策略

- **数据**: 自采集满语语音数据集,6.24h,44.1kHz/16-bit PCM,SNR≥25dB,覆盖 90% 核心音素+6 种语调类型 [§III-A]
- **划分**: 训练 5.2h (1892 句), 验证 0.52h (236 句), 测试 0.52h (236 句) [Table I]
- **训练配置**: 8×RTX 4090, 300K steps, batch=256, AdamW (β1=0.9, β2=0.98), cosine decay (warmup 5K steps, peak 3e-4 → 1e-5), gradient clipping 1.2, dropout 0.1-0.15, 约 48h [Table II]
- **输入**: 80-dim mel spectrogram, 50ms 帧长, 12.5ms 帧移
- **Speaker embedding**: 64-dim, 训练集 gender-balanced

## 实验

| 指标 | ManchuTTS | F5-TTS | VITS | Glow-TTS | FastSpeech 2 | Tacotron 2 | GT | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MOS ↑ | 4.52±0.11 | 4.12±0.11 | 3.95±0.12 | 3.89±0.13 | 3.45±0.15 | 3.21±0.18 | 4.68±0.09 | [Table III] |
| MCD ↓ | 5.83±0.23 | 5.99±0.23 | 6.18±0.24 | 6.34±0.25 | 7.15±0.28 | 7.92±0.31 | 4.40±0.20 | [Table III] |
| F0-RMSE ↓ | 18.7±2.1 | 20.3±2.5 | 21.7±2.7 | 22.4±2.8 | 29.8±3.7 | 34.2±4.3 | 14.2±2.0 | [Table III] |
| WER ↓ | 12.4±1.8 | 15.6±2.1 | 17.8±2.3 | 18.9±2.4 | 24.1±2.9 | 28.7±3.2 | 11.6±0.20 | [Table III] |
| SIM ↑ | 84.7±3.2 | 79.4±3.5 | 76.9±3.7 | 75.6±3.8 | 69.8±4.1 | 63.1±4.5 | 90.5±1.5 | [Table III] |
| PESQ ↑ | 3.21±0.09 | 3.08±0.08 | 3.02±0.09 | 2.97±0.10 | 2.68±0.11 | 2.45±0.12 | 3.89±0.10 | [Table III] |

**Ablation — 三级引导效果** [Table IV, Fig 5]:

| 配置 | 层级 | MOS | AWPA | Prosodic Naturalness |
| --- | --- | --- | --- | --- |
| A | Phoneme only | 3.94±0.14 | 70.8% | 70.4% |
| B | + Syllable | 4.21±0.12 | 84.7% | 80.5% |
| C | + Prosody (full) | 4.52±0.11 | 92.7% | 89.4% |

从 A→C: AWPA 提升 31% (70.8→92.7%), 韵律自然度提升 27% (70.4→89.4%)。音节层的加入使词根-后缀转换平滑(能量间隙缩短 12ms, MCD 额外下降 0.8dB),韵律层使疑问句 pitch 轮廓与母语者相关性达 0.91 (vs 无韵律层 0.63) [§III-B]。

**数据规模敏感性** [Fig 6]:
- 1h: MOS 3.22, WER 27.3%
- 2h: MOS ~3.8, DNSMOS 改善趋缓
- 5.2h: MOS 4.52, WER 12.4%
- 约 5h 为实用饱和点 [§III-B-3]

**跨语言 zero-shot 迁移** [Table V]:
- 满语→鄂温克语: MOS 3.78±0.15, WER 19.8% (vs 直接训练 0.8h 鄂温克语: MOS 4.01, WER 15.3%)
- 保持了关键语音特征(元音共振峰 r=0.87) [§III-B-4, Fig 8]

**推理效率** [Table VI]:
- RTF 0.12, 首 token 延迟 86ms, VRAM 4.1GB (RTX 4090)
- 支持 8 路并发流
- INT8 量化后在 Jetson Orin Nano (8GB) 上 3x 实时合成

**主观测试** [Table VII]:
- Naturalness MOS: 4.52±0.11 (vs GT 4.68)
- Clarity: 4.61±0.10 (vs GT 4.53,合成语音清晰度甚至略高)
- 63% 老年听者认为词尾发音清晰,但指出长句情感表达有限、偶有重音漂移

## 局限性

1. **数据局限**: 仅 6.24h 实验室录音,不覆盖方言变体,方言场景精度下降 [§IV]
2. **情感表达有限**: 长句中情感表现不足,偶有重音漂移 [§III-D]
3. **与主流 TTS 架构差距大**: 86.4M 参数的单阶段系统,不支持 in-context learning 或 zero-shot voice cloning [agent 解读]
4. **数据集开放性**: 仅承诺公开"对齐的语音-文本子集",非完全开源 [§III-A]
5. **评估局限**: 母语评估者极少(20 人),且 MOS 评估的统计效力受限; AWPA 和 Prosodic Naturalness 由 2 名语言学家二元分类评估,主观性较强 [Table IV note]
6. **zero-shot 迁移**: 满语→鄂温克语 zero-shot MOS 3.78,与直接训练仍有差距 (4.01),WER 差距更大 (19.8% vs 15.3%) [Table V]

## 点评

**定位价值**: ManchuTTS 的核心价值不在 TTS 技术前沿突破,而在将 CFM+层级化语言学条件的框架应用于极端低资源的濒危语言场景。这是一个被主流 TTS 研究忽视但有重要文化和语言保护意义的方向。

**方法论可迁移性**: 三级文本表征 + 层级对比损失的思路有普适性。任何黏着语(如土耳其语、蒙古语、芬兰语)的低资源 TTS 都可借鉴。self→cross→self 的三层跨模态 attention 也是一种轻量级多模态对齐方案。

**实验可信度疑虑**:
- 所有 baseline 使用相同 5.2h 数据训练,但未说明 baseline 是否做了 hyperparameter search。F5-TTS 本身设计用于大规模数据,5.2h 下可能严重受限,这可能放大了 ManchuTTS 的优势 [agent 解读]
- MOS 4.52 在 5.2h 低资源条件下极高,接近 GT 4.68,需谨慎看待
- CBVC (Cloning-based Voice Conversion) baseline MOS 4.28 与 ManchuTTS 差距较小(0.24),但论文未深入分析两者的差异

**技术局限**: 86.4M 参数的系统在 2025 年的 TTS 生态中较小,不支持 zero-shot speaker adaptation 或 instruction-guided 合成。与 CosyVoice 系列(LLM+CFM)相比,本文仍是传统 NAR TTS 的改进,只是添加了 CFM 和层级化条件。

**文献引用质量**: 部分参考文献与论文主题关联度低(如 [2] ImagePose 是视觉生成工作,[3] 是 talking face 生成,[5] Wave-u-mamba 是语音超分辨率),可能存在不相关引用的问题 [agent 解读]。

## 可复用的 idea

1. **三级语言学条件化 CFM**: 将 phoneme/syllable/prosody 作为分层条件注入 flow matching 的向量场,而非单一隐层表征。可用于其他需要多粒度文本控制的 TTS 场景。

2. **层级对比对齐损失 (HCA)**: 在 InfoNCE 框架下对每个语言学层级构造正负样本,最大化文本-声学互信息。可作为低资源 TTS 的辅助训练信号,不限于 CFM 架构。

3. **self→cross→self 三层跨模态 attention**: 渐进式对齐方案,结构简单且层级清晰。可迁移到任何需要文本-声学隐式对齐的 NAR TTS 系统。

4. **数据规模饱和点分析**: 系统性地测试了 1h/2h/5.2h 的效果递增,表明约 5h 为实验室录音条件下的实用饱和点。这为其他低资源语言 TTS 数据收集提供了量级参考。
