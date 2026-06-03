---
type: paper
tier: deep
title: "VoxCPM: Tokenizer-Free TTS for Context-Aware Speech Generation and True-to-Life Voice Cloning"
arxiv_id: "2509.24650"
source: "Sources/VoxCPM.pdf"
authors: [Yixuan Zhou, Guoyang Zeng, Xin Liu, Xiang Li, Renjie Yu, Ziyang Wang, Runchuan Ye, Weiyue Sun, Jiancheng Gui, Kehan Li, Zhiyong Wu, Zhiyuan Liu]
year: 2025
venue: "arXiv preprint"
tags: [TTS, zero-shot, tokenizer-free, hierarchical-modeling, flow-matching, end-to-end, semi-discrete, LLM-based]
concepts: ["[[Finite Scalar Quantization]]", "[[Conditional Flow Matching]]", "[[Residual Vector Quantization]]", "[[Semantic vs Acoustic Tokens]]", "[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Classifier-Free Guidance]]", "[[Variational Autoencoder for TTS]]", "[[Next-Token Diffusion]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]", "[[模型库/CosyVoice 3|CosyVoice 3]]", "[[模型库/MELLE|MELLE]]", "[[模型库/HierSpeech++|HierSpeech++]]", "[[模型库/NaturalSpeech 2|NaturalSpeech 2]]"]
tasks: ["[[任务库/Zero-shot Speech Synthesis|Zero-shot Speech Synthesis]]"]
datasets: ["[[数据集/Emilia|Emilia]]", "[[数据集/SEED-TTS-Eval|SEED-TTS-Eval]]", "[[数据集/CV3-Eval|CV3-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页 + 1 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: VoxCPM 处于 [[LLM-based TTS]] 谱系中 "Hybrid 架构 (LLM + Diffusion)" 路线的最新发展,但与 CosyVoice 系列的 "discrete semantic tokens + 独立 CFM renderer" 两阶段管线不同,VoxCPM 追求的是**端到端统一框架**,无需外部 [[Speech Tokenizer]]。这使它更接近 DiTAR / CLEAR / VibeVoice 等 continuous-valued AR 路线,但独创地引入了 [[Finite Scalar Quantization]] [待确认] 作为内部正则化瓶颈(不是预测目标),解决了纯连续模型的 error accumulation 问题。

**已有认知**:
- [[Semantic vs Acoustic Tokens]] 指出: 离散 token 保稳定但牺牲表达力,连续信号保保真度但 entangle semantic/acoustic。VoxCPM 的 semi-discrete residual 方案是对这一 trade-off 的新解法。
- [[Conditional Flow Matching]] 在 CosyVoice 系列中作为独立的 second-stage renderer; VoxCPM 将 flow matching 作为唯一训练目标统一全模型,更接近 CLEAR 的单阶段路线。
- [[Residual Vector Quantization]] 的层级信息结构(coarse→fine)启发了 VoxCPM 的 TSLM+FSQ(semantic skeleton) + RALM(acoustic residual) 层级设计,但 VoxCPM 用 FSQ 替代 RVQ 第一层,用连续残差替代 RVQ 后续层。
- [[Zero-shot Speech Synthesis]] 当前 SOTA: CosyVoice 3-1.5B (CER 0.71%, WER 1.45%)。VoxCPM 仅 0.5B 参数,目标是在开源系统中达到最优。

**创新判断**: 相比 CosyVoice / IndexTTS2 等依赖外部 speech tokenizer 的系统,VoxCPM 的核心创新在于: (1) FSQ 不作为 tokenizer 的量化层,而是作为模型内部的可微正则化瓶颈; (2) TSLM + RALM 残差分工实现隐式 disentanglement; (3) 全模型端到端训练,消除 tokenizer→LM→diffusion 多阶段割裂。

> 检索命中: [[Conditional Flow Matching]]✓, [[Residual Vector Quantization]]✓, [[Semantic vs Acoustic Tokens]]✓, [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[Zero-shot Speech Synthesis]]✓ | 过滤: [[Finite Scalar Quantization]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过可微 FSQ 瓶颈实现 TSLM(语义骨架) + RALM(声学残差) 的隐式分工,端到端训练无需外部 speech tokenizer 的 0.5B zero-shot TTS
> - **路线**: Text → TSLM (MiniCPM-4 init) → FSQ 瓶颈(semi-discrete skeleton) + RALM(连续残差) → LocDiT (local diffusion) → Causal VAE decoder → 16kHz waveform
> - **指标**: SEED-TTS-Eval EN-WER 1.85% / ZH-CER 0.93% (开源 SOTA); CV3-Hard-EN WER 7.89% (超闭源 CosyVoice 3); RTF 0.17 on RTX 4090 [Table 3, 4]
> - **可借鉴**: FSQ 作为内部正则化瓶颈(不是预测目标)诱导 semantic-acoustic 分工; 残差连接实现功能分离但不割裂架构; WSD 两阶段 lr schedule 显著提升 zero-shot similarity
> - **局限**: 仅支持中英文; 无显式韵律/情感控制; 仅 16kHz (非高保真 24kHz/44.1kHz); 内部数据集 1.8M 小时不可复现

## 核心问题

VoxCPM 要解决 TTS 中两大技术路线的根本矛盾:

1. **离散 token 路线的量化天花板**: EnCodec/SoundStream 等 RVQ-based tokenizer 不可逆地丢弃声学细节,且随维度增加 codebook 指数膨胀。现有 SOTA (CosyVoice、IndexTTS2) 采用 LM + 独立 diffusion 两阶段管线缓解,但造成 **semantic-acoustic 割裂** — LM 在离散空间不感知声学,diffusion 做局部渲染缺乏全局语义上下文 [§1]。

2. **纯连续路线的 error accumulation**: MELLE、DiTAR 等直接在连续空间自回归生成,但 semantic planning 和 acoustic rendering **任务纠缠** — 模型同时做全局规划和局部渲染,关注重心被底层声学纹理拖走,长序列下语义连贯性崩溃 [§1]。

VoxCPM 的核心假设: **需要显式的架构分离来隔开 semantic planning 和 acoustic rendering,但这种分离必须在端到端可微框架内实现,而非多阶段割裂** [§3.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VoxCPM 由五个模块组成,自回归地逐 patch 生成连续 VAE latent [§3.2, Fig 1]:

```
Text T → [TSLM (24L, MiniCPM-4 init)] → FSQ bottleneck → semi-discrete skeleton h^FSQ
                                                          + RALM (6L) → residual h^residual
                                                          ↓
                                              h^final = h^FSQ + h^residual
                                                          ↓
                                                   [LocDiT (4L)] → z_i (VAE latent patch)
                                                          ↓
                                                  [Causal Audio VAE] → 16kHz waveform

Historical context: Z_{<i} → [LocEnc (4L)] → E_{<i} (fed to TSLM and RALM)
```

Patch size = 2, 即 TSLM/RALM 工作在 12.5 Hz token rate, VAE latent 在 25 Hz frame rate [Table 1]。

### 关键设计选择

**1. FSQ 作为正则化瓶颈,而非 tokenizer**

[论文原文] "Unlike multi-stage TTS systems that treat quantization as a means to obtain discrete prediction targets, our approach uses quantization solely as a regularization mechanism to constrain the hidden state space" [§3.1]。

为什么选择 FSQ 而非 VQ/RVQ: [论文原文] 随维度增加,VQ/FSQ 直接做 codebook → codebook size 指数增长,LM 难以准确预测 [§3.1]。VoxCPM 的 FSQ 用 256 维 x 9 levels,理论隐式 codebook 巨大 (9^256),但这不是问题因为 FSQ 不是预测目标 — 它只是中间瓶颈。[agent 解读] 这是对 FSQ 的创新性使用: CosyVoice 系列中 FSQ 产生 discrete tokens 作为 LM 的预测目标; VoxCPM 中 FSQ 只做中间正则化,下游 LocDiT 的预测目标仍是连续 VAE latent。

**2. TSLM + RALM 残差分工**

[论文原文] FSQ 强制 TSLM 输出通过量化瓶颈 → 只保留 stable semantic-prosodic skeleton → TSLM 被引导去建模 "可以活过量化" 的信息(内容、韵律大结构)[§3.3.2]。RALM 接收量化残差,专职恢复 FSQ 丢弃的声学细节(speaker identity, spectral fine structure, micro-prosody)[§3.3.3]。

[agent 解读] 这种分工是 FSQ 瓶颈自然诱导的,而非人工设计 — 训练时梯度通过 STE 传过 FSQ,TSLM 会自然学习编码"量化友好"的语义信息,声学细节则"溢出"到 RALM 的残差通道。这比 CosyVoice 的外部 tokenizer 分离更优雅:不需要独立训练 tokenizer,分工边界由端到端训练自动确定。

**3. 预训练 LLM 初始化 (MiniCPM-4-0.5B)**

[论文原文] TSLM 使用 MiniCPM-4-0.5B 作为初始化,而非随机初始化,使其具备 "richer contextual understanding and more natural prosody prediction directly from raw text" [§3.3.1]。

[agent 解读] 与 CosyVoice 2/3 使用 Qwen/Llama 初始化类似,但 VoxCPM 直接用 BPE text tokens(字符级中文切分)而非 phoneme,消除了对外部 phonemizer 的依赖,也意味着模型需要自行学习 grapheme-to-phoneme 映射。

**4. LocDiT 的 outpainting 设计**

[论文原文] LocDiT 接收 h^final 作为条件,同时以前一 patch z_{i-1} 作为额外上下文,"framing the task as outpainting rather than independent patch generation" — 经验验证显著提升输出质量 [§3.3.4]。这是对 DiTAR 的 local diffusion 设计的继承和扩展。

**5. CFG 用于 LM guidance**

训练时以 0.1 概率 mask 掉 LocDiT 的 LM 条件,推理时使用 CFG scale=2.0 增强条件引导 [§3.3.4, Table 9]。[agent 解读] CFG 在这里的作用是加强 TSLM+RALM 对 LocDiT 的语义引导,防止 diffusion decoder 退化为不依赖条件的纯分布拟合。

### 训练策略

**单目标端到端训练**: 全模型使用 flow matching loss L_FM + stop prediction loss L_Stop 联合训练 [§3.4]。FSQ 的梯度通过 STE 传播。[论文原文] 这允许 "each component to learn its specialized role in a coordinated manner, guided by the unified objective" [§3.4]。

**两阶段 WSD learning rate schedule** [§4.1, Table 2]:
- Stable phase: lr=1e-4, batch=4096 tokens, 400K iter
- Decay phase: lr 从 1e-4 退火到 5e-6, batch 翻倍到 8192, 100K iter

[论文原文] Decay phase "是实现最优性能的关键,特别是 zero-shot speaker similarity" [§4.5]。[agent 解读] batch size 翻倍 + lr 退火的组合很可能在稳定训练后期帮助模型学到更精细的 speaker-specific 声学映射。

**Audio VAE**: DAC-like 架构,causal CNN,stride sequence [2,5,8,8],640x 下采样 → 25 Hz latent。训练目标: Mel-spectrogram loss + GAN loss + KL divergence (weight=5e-5)。独立于主模型训练 [§3.5]。

## 实验

### 主要结果

| 指标 | VoxCPM (0.5B) | CosyVoice 3 (0.5B) | DiTAR (0.6B) | IndexTTS 2 (0.5B) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| EN-WER ↓ | **1.85%** | 2.02% | **1.69%** | 2.23% | SEED-TTS-Eval | [Table 3] |
| EN-SIM ↑ | **72.9%** | 71.8% | 73.5% | 70.6% | SEED-TTS-Eval | [Table 3] |
| ZH-CER ↓ | **0.93%** | 1.16% | 1.02% | 1.03% | SEED-TTS-Eval | [Table 3] |
| ZH-SIM ↑ | 77.2% | **78.0%** | 75.3% | 76.5% | SEED-TTS-Eval | [Table 3] |
| Hard-CER ↓ | 8.87% | - | **5.83%** | 7.97% | SEED-TTS-Eval Hard | [Table 3] |
| CV3-ZH-CER ↓ | **3.40%** | 3.89%* | - | 3.58% | CV3-Eval | [Table 4] |
| CV3-EN-WER ↓ | **4.04%** | 5.24%* | - | 4.45% | CV3-Eval | [Table 4] |
| CV3-Hard-EN WER ↓ | **7.89%** | 9.04%* | - | - | CV3-Hard-EN | [Table 4] |
| N-MOS (ZH) | 4.10 | - | - | 4.25 | 主观评估 | [Table 5] |
| S-MOS (EN) | **4.18** | - | - | 4.16 | 主观评估 | [Table 5] |

*CosyVoice 3 数据来自闭源版本

### 消融: FSQ 瓶颈维度 [Table 6]

| 配置 | EN-WER ↓ | ZH-CER ↓ | ZH-Hard CER ↓ |
| --- | --- | --- | --- |
| w/o FSQ (d1024, 纯连续) | 3.67% | 2.30% | **24.92%** |
| FSQ d4s9 (过紧) | 5.18% | 4.05% | 19.55% |
| FSQ d128s9 | 3.43% | **1.67%** | 16.76% |
| **FSQ d256s9 (选定)** | **2.98%** | 1.77% | **18.19%** |
| FSQ d1024s9 (过松) | 3.07% | 2.38% | 20.38% |

[论文原文] 去掉 FSQ 后 Hard case CER 从 18.19% 暴涨到 24.92%,"validates our core hypothesis: entangling semantic planning and acoustic rendering in a continuous space leads to instability" [§4.3]。

[agent 解读] d256 不一定在所有子指标上最优(d128 的 ZH-CER 更低),但在综合平衡(EN-WER + ZH-CER + Hard stability)上是最佳 sweet spot。Hard case 的急剧退化清晰展示了纯连续模型在长/难句上的 error accumulation。

### 消融: RALM 的作用 [Table 7]

| 配置 | EN-WER ↓ | ZH-Hard CER ↓ |
| --- | --- | --- |
| Default (TSLM + FSQ + RALM) | **2.98%** | **18.19%** |
| w/o RALM: TSLM (24L) → LocDiT | 4.34% | 25.00% |
| w/o RALM: TSLM (30L) → LocDiT | 5.35% | 30.40% |
| w/o E_{<i} in RALM | 4.91% | 27.17% |

[论文原文] 纯连续方案 "TSLM → LocDiT" 类似 DiTAR,性能显著退化; 扩大 TSLM 到 30 层反而更差,确认 "the challenge is fundamental to the learning objective rather than parameter allocation" [§4.4]。

[agent 解读] 这组消融最有说服力: 即使把 RALM 的 6 层参数补偿给 TSLM(24→30 层),性能反而更差 — 说明不是参数量问题,而是**架构分离**本身带来的归纳偏置才是关键。

### WSD 训练阶段效果 [Table 8]

| Phase | EN-WER | ZH-SIM | ZH-Hard CER |
| --- | --- | --- | --- |
| Stable | 2.05% | 75.1% | 13.22% |
| Decay | **1.85%** | **77.2%** | **8.87%** |

Decay 阶段使 ZH-Hard CER 从 13.22% 降到 8.87%,SIM 提升 2.1 个百分点 [Table 8]。

### CFG 效果 [Table 9]

CFG=1.0 (无 CFG): EN-WER 16.32%, ZH-CER 14.47% → 完全崩溃
CFG=2.0 (最优): EN-WER 1.85%, ZH-CER 0.93%
CFG=5.0 (过强): EN-WER 12.78% → 过度条件导致发散 [Table 9]

### 推理效率

RTF 0.17 on single RTX 4090 [§1]。

## 局限性

1. **语言覆盖有限**: 仅优化中英文,对其他语言泛化能力不确定 [§5 Limitations]
2. **缺乏显式可控性**: 无细粒度韵律/情感控制机制,依赖模型从文本隐式推断 [§5 Limitations]
3. **采样率限制**: AudioVAE 仅支持 16kHz,无法满足高保真应用 (24kHz/44.1kHz) 需求 [§5 Limitations]
4. **数据不可复现**: 1.8M 小时内部数据集无法公开,Emilia (95K h) 上的结果有明显差距 (EN-WER 2.34% vs 1.85%),说明模型对大规模数据依赖较强
5. **Hard case 仍有差距**: ZH-Hard CER 8.87% 仍远高于 DiTAR 的 5.83% [Table 3],BPE-based 路线在发音稳定性上可能不如 phoneme-based

## 点评

**优点**:

1. **理论清晰**: "FSQ 不是 tokenizer 而是正则化瓶颈" 这一 insight 概念上优雅。CosyVoice 系列中 tokenizer 和 LM 是独立训练的两个模块,存在信息流断裂; VoxCPM 将量化内化为模型的一部分,通过端到端训练让分工边界自动确定。
2. **消融充分**: FSQ 维度扫描 + RALM 消融 + WSD 效果 + CFG 效果 + 参数补偿实验,逐个验证核心假设,特别是 "30 层 TSLM 不如 24 层 TSLM + 6 层 RALM" 是极有说服力的对比。
3. **工程务实**: 0.5B 参数在 RTX 4090 上 RTF 0.17,开源 Apache 2.0,有 demo/模型/代码,实用性强。

**不足**:

1. **主要 baseline 缺失**: 没有与 DiTAR 在 CV3-Eval 上的对比; 没有与 CosyVoice 3 开源版的公平对比(Table 4 中 CosyVoice 3 标注闭源)。
2. **t-SNE 分析的解释力有限**: Fig 2-3 展示了 TSLM-FSQ 和 RALM 输出的聚类模式,但 t-SNE 本身不能证明 disentanglement 的程度和质量。
3. **16kHz 限制使 DNSMOS 偏低**: Table 4 中 VoxCPM 的 DNSMOS (3.59-3.74) 低于 IndexTTS2 (3.75-3.95),可能部分因为 16kHz 采样率本身就限制了感知质量。

## 可复用的 idea

1. **FSQ 作为内部正则化瓶颈**: 不把量化作为 tokenizer 的预测目标,而是作为模型内部的可微瓶颈层,诱导 semantic/acoustic 自然分工。这个 trick 可迁移到任何需要隐式 disentanglement 的 seq2seq 任务。
2. **残差分工 = 功能分离不割裂架构**: h_final = FSQ(TSLM) + RALM 的残差连接,让两个模块各司其职但仍在同一梯度流中,比多阶段管线更紧凑。
3. **参数补偿消融 (24L+6L vs 30L+0L)**: 证明架构分离的归纳偏置比等量参数更有价值,是验证 "组件分工假设" 的标准实验范式。
4. **WSD schedule + batch doubling for similarity**: 两阶段学习率策略在 stable phase 后用 decay + batch size x2 显著提升 zero-shot speaker similarity,值得在其他 TTS 模型训练中尝试。
