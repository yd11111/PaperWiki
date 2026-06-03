---
type: paper
tier: deep
title: "SongGen: A Single Stage Auto-regressive Transformer for Text-to-Song Generation"
arxiv_id: "2502.13128"
source: "https://arxiv.org/abs/2502.13128"
authors: [Zihan Liu, Shuangrui Ding, Zhixiong Zhang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, Jiaqi Wang]
year: 2025
venue: "arXiv"
tags: [text-to-song, song-generation, autoregressive, codec-LM, voice-cloning, music-generation, open-source]
concepts: ["[[Residual Vector Quantization]]", "[[Codec Language Model]]", "[[Singing Voice Synthesis]]", "[[Speech Tokenizer]]", "[[Classifier-Free Guidance]]"]
models: ["[[模型库/EnCodec|EnCodec]]"]
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[Residual Vector Quantization]], [[Speech Tokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **[[Residual Vector Quantization]]** (confirmed): RVQ 是现代 neural audio codec 的核心量化方法,通过递归量化残差逐步逼近输入。SongGen 使用的 X-Codec 基于 RVQ,8 层 codebook,每层 1024 entries。RVQ 的层级信息结构(前层 coarse / 后层 fine）天然适合 codebook-delay pattern 建模。
>
> **[[Speech Tokenizer]]** (confirmed): Speech Tokenizer 将连续音频波形转为离散 token。SongGen 使用声学 tokenizer (X-Codec) 而非 semantic tokenizer,直接操作 RVQ acoustic tokens。Survey 发现没有万能 tokenizer,domain-specific 训练对生成质量至关重要。
>
> **[[Singing Voice Synthesis]]** [待确认]: SVS 从歌词+乐谱生成歌声。SongGen 属于 "Text-to-Song Generation" 子任务 — 从文本直接生成完整歌曲(歌声+伴奏),是 SVS + 音乐生成的融合。SVS 概念页已收录 SongGen 为代表工作。
>
> **[[Codec Language Model]]** [待确认]: CodecLM 直接在 neural audio codec tokens 上训练语言模型。SongGen 是 CodecLM 在 song generation 领域的应用,使用 codebook-delay pattern 处理多层 RVQ tokens。
>
> 检索命中: [[Residual Vector Quantization]], [[Speech Tokenizer]] | 过滤: [[Singing Voice Synthesis]](pending-review), [[Codec Language Model]](pending-review), [[Musical Score Encoder]](pending-review), [[Classifier-Free Guidance]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个全开源的单阶段自回归 Transformer，直接从歌词+文本描述+可选参考声音生成完整歌曲（歌声+伴奏），支持 mixed mode 和 dual-track mode 两种输出模式
> - **路线**: (歌词→VoiceBPE→Lyrics Encoder) + (描述→FLAN-T5→Text Embedding) + (参考声→MERT→Voice Embedding) → 拼接 cross-attention 条件 → Transformer Decoder (codebook-delay pattern) → X-Codec 解码 → 歌曲波形
> - **指标**: Mixed Pro: FAD 1.71 / KL 0.69 / CLAP 0.35 / OVL 3.96 / REL 3.86; Interleaving A-V: FAD 1.87 / OVL 3.95 / VQ 4.15 (MusicCaps test) [Table 1]
> - **可借鉴**: (1) Auxiliary vocal token prediction target 缓解 mixed audio 中人声被伴奏淹没的 learning bias; (2) Interleaving A-V 双轨 token pattern 利用 Transformer 层级分工(低层跨轨/高层轨内); (3) VoiceBPE 音素级 tokenizer 适配歌唱场景; (4) Curriculum learning 调整 codebook loss 权重
> - **局限**: 最长 30 秒/16kHz; 仅 2K 小时训练数据; Mixed mode 人声清晰度仍不如 dual-track; 无法生成完整歌曲结构; X-Codec 16kHz 限制音质

## 核心问题

传统 text-to-song 方法采用多阶段 pipeline（text-to-vocal → vocal-to-accompaniment，或 text-to-MIDI → text-to-vocal → V2A），训练和推理流程复杂,缺乏对歌声和伴奏的统一控制 [§1]。SongGen 的核心问题是: **能否用单个自回归 Transformer 直接从文本输入生成完整歌曲?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SongGen 由一个自回归 Transformer Decoder + off-the-shelf neural audio codec (X-Codec) 组成 [§3.1, Fig 2]:

1. **编码阶段**: 三种用户输入分别编码 →
   - 歌词 → VoiceBPE tokenizer → Lyrics Encoder (6-layer Transformer, hidden 1024) → $E_{\text{lyrics}} \in \mathbb{R}^{T_l \times F_l}$ [§3.3]
   - 文本描述 → 冻结 FLAN-T5 encoder → $E_{\text{text}} \in \mathbb{R}^{T_t \times F_t}$ [§3.3]
   - (可选) 3 秒参考声 → 冻结 MERT encoder + 1D Conv 聚合 → $E_{\text{voice}} \in \mathbb{R}^{T_v \times F_v}$ [§3.3]
2. **条件拼接**: 三种嵌入各经 projection → 沿时间维度拼接 $E_{\text{cond}} = \hat{E}_{\text{voice}} \oplus \hat{E}_{\text{text}} \oplus \hat{E}_{\text{lyrics}}$ [Eq.4]
3. **生成阶段**: Transformer Decoder (24 layers, 1024 hidden) 通过 cross-attention 接收条件,自回归预测 audio tokens [§3.1]
4. **解码阶段**: X-Codec decoder 将 tokens 合成波形 [§3.1]

### 关键设计选择

#### 1. 音频 Tokenization: X-Codec + Codebook-Delay Pattern [§3.2.1]

使用 X-Codec (Ye et al., 2024),一种基于 RVQ 的 codec,特点是同时考虑 acoustic 和 semantic 信息 [论文原文]。参数: $N_q=8$ codebooks, $K=1024$ entries/codebook, $f_s=16$ kHz, $f_r=50$ Hz [§3.2.1]。

Codebook-delay pattern (来自 MusicGen): 相邻 codebook 之间错开一步,使多层 RVQ 可在单个 Transformer 中自回归建模 [§3.2.1]。

**为什么选 X-Codec 而非 EnCodec/DAC?** [agent 解读] Ablation [Table 5, Appendix D] 表明 X-Codec 在 FAD/KL/CLAP/PER/SECS 上全面超越 EnCodec 和 DAC,且训练 loss 更稳定、收敛更快。[论文原文] X-Codec 整合了 semantic 信息,对 song generation 任务高度有效 [§D]。

#### 2. Mixed Mode: Auxiliary Vocal Token Prediction [§3.2.2]

Mixed mode 直接生成混合音频 tokens,但存在 learning bias: 伴奏能量高、频谱稳定,模型倾向优先学习伴奏而忽视人声 [论文原文]。

**Mixed Pro 方案**: 引入辅助 vocal token 预测目标 — 额外一组 linear heads 预测从纯人声轨编码的 X-Codec tokens,与 mixed tokens 帧对齐 [§3.2.2, Fig 3(a)]:

$$\mathcal{L}_{\text{mixed-pro}} = \mathcal{L}_{\text{mixed}} + \lambda \mathcal{L}_{\text{vocal}}$$

其中 $\lambda=0.1$ 控制 vocal loss 权重 [§B]。辅助 heads 仅训练时使用,不影响推理 [论文原文]。

**效果**: Mixed Pro 在 PER 和 VQ 指标上显著优于 basic Mixed [Table 1],表明辅助目标有效缓解了 learning bias [论文原文]。

#### 3. Dual-Track Mode: Interleaving Pattern [§3.2.3]

Dual-track mode 分离 vocal 和 accompaniment 两轨,探索了两类 token 组合模式 [§3.2.3, Fig 3]:

**Parallel pattern** (3 variants): vocal 和 accompaniment tokens 沿 codebook 维度拼接,每步同时输出 $N_q$ vocal + $N_q$ accompaniment tokens。Standard / Parallel A-V / Parallel V-A [Fig 3(b)]。

**Interleaving pattern** (2 variants): vocal 和 accompaniment tokens 沿时间维度交错。Interleaving A-V (accompaniment 先) / Interleaving V-A (vocal 先) [Fig 3(c)]。

**关键发现**: Interleaving A-V 表现最佳 [Table 1]。[论文原文] 这是因为 interleaving pattern 在 Transformer 低层学习跨轨交互 (inter-track)，高层专注轨内特征 (intra-track)，注意力可视化 [Fig 5(d-f)] 证实了这种层级分工: 低层呈均匀注意力,高层呈棋盘格注意力 [§3.2.3, §4.2]。而 parallel pattern 无法在到达 heads 前解耦 vocal 和 accompaniment 信息 [论文原文]。

**伴奏先于人声的优势**: A-V 优于 V-A [Table 1],可能因为伴奏提供了更稳定的频谱框架,先生成伴奏可为后续人声提供上下文 [agent 解读]。

#### 4. Lyrics Conditioning: VoiceBPE + Cross-Attention [§3.3]

采用 VoiceBPE (Casanova et al., 2024)，一种 6681-token 的 voice Byte-Pair Encoding tokenizer，生成类音素 token [§3.3]。

**为什么不用 T5/word-level tokenizer?** [论文原文] Word-level tokenizer (如 T5) 导致 token 与音素稀疏对应，训练样本不足。VoiceBPE 更好地泛化到未见词，且适配歌唱中的音素时长和音高变化 [§3.3]。Ablation [Table 4] 验证: VoiceBPE + Lyrics Encoder + cross-attention 组合在 FAD/PER/SECS 上全面最优。

**Cross-attention vs Prepend**: [论文原文] Cross-attention 允许 decoder 专注音频生成,比 prepend (在 token 前拼接转写) 更稳定有效 [§4.3]。

#### 5. Voice Conditioning: MERT Encoder [§3.3]

使用冻结 MERT (Li et al., 2024b) encoder 提取声音特征嵌入，支持 3 秒参考片段的零样本 voice cloning [§3.3]。

**为什么选 MERT?** [论文原文] MERT 在声乐技巧检测和歌手识别任务上达到 SOTA,可提供 robust 的声音音色和歌唱技巧特征 [§3.3]。

### 训练策略

**三步 Mixed Mode 训练** [§3.5]:
1. **Step 1: Modality Alignment** — 全量数据训练全模型，对齐条件输入与音频输出
2. **Step 2: Voice-Free Support** — 50% 概率 drop 参考声音输入,冻结相关模块,仅微调 decoder
3. **Step 3: High-Quality Fine-tuning** — 用严格质量筛选数据子集 (edit_dist ≤ 5%, CLAP_src ≥ 25%, energy > 1000) 微调全模型,约 100K 高质量对 [§3.5]

**Dual-Track Mode 训练** [§3.5]: 从 Mixed Mode Step 1 预训练模型初始化,先适配 decoder (Step 1.5),再解冻全模型,后续步骤同 Mixed Mode [§3.5]。

**Curriculum Learning for Codebook Loss Weights** [§3.5]: 初始前 3 个 codebook 权重 0.25,其余 0.05;随训练逐步平衡。[论文原文] 让模型先学最重要的 coarse components,再逐步精化 fine details [§3.5]。

### 数据处理 Pipeline [§3.4]

从 MSD (8000h) + FMA + MTG-Jamendo 收集音频,经过 [§3.4]:
1. **Source Separation**: Demucs 分离 vocal/accompaniment
2. **VAD**: 检测有声段,切片~15s
3. **Lyric Recognition**: Whisper-large-v2 + Whisper-larger-v3 双模型转写,edit distance > 20% 的丢弃
4. **Captioning**: LP-MusicCaps-MSD 生成描述,CLAP 评分过滤
5. **最终数据**: 540K 英语有声片段,约 2K 小时 [§3.4]

## 实验

| 指标 | SongGen Mixed Pro | SongGen Inter. A-V | Stable Audio Open | MusicGen | Parler-tts* | Suno | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FAD ↓ | 1.71 | **1.87** | 4.87 | 5.17 | 4.13 | - | MusicCaps | [Table 1] |
| KL ↓ | **0.69** | **0.69** | 1.15 | 0.89 | 1.00 | - | MusicCaps | [Table 1] |
| CLAP ↑ | **0.35** | **0.35** | 0.28 | 0.09 | 0.19 | - | MusicCaps | [Table 1] |
| PER ↓ | **40.58** | 39.46 | - | - | 58.61 | - | MusicCaps | [Table 1] |
| OVL MOS ↑ | 3.96 | 3.95 | 3.01 | 3.15 | 2.58 | 4.28 | MusicCaps | [Table 1] |
| VQ MOS ↑ | 4.07 | **4.15** | 1.29 | - | 2.28 | 4.22 | MusicCaps | [Table 1] |
| HAM MOS ↑ | **4.01** | 3.82 | - | - | 2.35 | **4.33** | MusicCaps | [Table 1] |

**关键发现**:
- SongGen 在所有客观指标上大幅超越开源 baseline (Stable Audio Open, MusicGen, Parler-tts) [Table 1, §4.2]
- 与 Suno (商业产品) 相比,SongGen 在 REL (文本相关性) 和 VQ (声音质量) 上有优势,但 OVL 和 HAM 略低 [Table 1]
- Mixed Pro 的 HAM (和声) 优于 Interleaving A-V,但后者的 VQ (声音质量) 更好 [Table 1, §4.2] — 两种模式各有优势
- Ablation: HQFT 和 curriculum learning 均显著提升性能 [Table 3]; VoiceBPE + lyrics encoder + cross-attention 最优 [Table 4]; X-Codec >> EnCodec >> DAC [Table 5]

## 局限性

- **30 秒上限**: 训练数据切片约 15s,最长生成 30s,无法生成完整歌曲结构 (verse-chorus) [§A]
- **16 kHz 采样率**: X-Codec 限制,音质不及 24kHz/44.1kHz 商业产品 [§A]
- **仅 2K 小时训练数据**: 公开数据严重稀缺,远少于 Suno 等商业系统的数据量 [§3.4]
- **Mixed mode 人声清晰度**: 即使有 auxiliary vocal target,mixed audio 中人声仍不如 dual-track 清晰 [Table 1, PER 对比]
- **英语限制**: 训练数据仅含英语有声歌曲 [§3.4]

## 点评

SongGen 是 text-to-song generation 领域的重要里程碑:

1. **简洁有效的单阶段设计**: 相比 Melodist (V2A)、MelodyLM (text-to-MIDI + V2A) 等多阶段方法,SongGen 用单个 Transformer 统一处理,训练/推理更简洁。这种范式回归了"让模型自己学"的大模型哲学 [agent 解读]。

2. **Token pattern 的深入探索**: 对 mixed/dual-track 两大模式各 3+ 种 token pattern 的系统实验是本文最大贡献。Interleaving A-V 的层级分工发现 (低层 inter-track / 高层 intra-track) 具有普适意义,可迁移到其他多轨生成任务 [agent 解读]。

3. **Auxiliary vocal target 的巧妙设计**: 用额外预测头强化人声学习,训练时加推理时不加,零额外推理成本。思路简洁有效 [agent 解读]。

4. **全开源承诺**: 模型权重+训练代码+标注数据+预处理 pipeline 全部开源,填补了该领域开源空白 [§Abstract]。

5. **与 SVS 的关系**: SongGen 不使用乐谱输入 (无 [[Musical Score Encoder]])，而是从自由文本描述生成,属于 text-to-song 而非传统 SVS。这意味着它的音乐控制力弱于乐谱驱动的 SVS,但使用门槛大幅降低 [agent 解读]。

## 可复用的 idea

1. **Auxiliary prediction target 缓解 learning bias**: 当模型在混合信号中倾向忽略某个子信号时,引入该子信号的辅助预测头。可推广到任何"主/次信号混合"场景 (如带背景噪声的语音、多乐器混音)
2. **Interleaving token pattern + 层级注意力分工**: 多轨/多通道信号的 Transformer 建模可采用 interleaving 排列,利用低层学跨通道关系、高层学通道内特征
3. **VoiceBPE 音素级 tokenizer**: 歌唱场景下,音素级比词级 tokenizer 更适合,因为需要建模音素时长和音高变化
4. **Curriculum learning for codebook weights**: RVQ 多层 loss 权重从不均匀逐步平衡,先学 coarse 再精化
5. **数据质量筛选 pipeline**: 双 ASR 模型 + edit distance 过滤 + CLAP 评分 + 能量过滤,可复用于任何音频-文本对数据清洗
