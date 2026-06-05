---
type: paper
tier: deep
title: "Raon-OpenTTS: Open Models and Data for Robust Text-to-Speech"
arxiv_id: "2605.20830"
source: "Sources/Raon-OpenTTS.pdf"
authors: [Semin Kim, Seungjun Chung, Taehong Moon, Sangheon Lee, Minyoung Ahn, Keon Lee, Nam Soo Kim, Jaewoong Cho, Ludwig Schmidt, Kangwook Lee, Dongmin Park]
year: 2026
venue: "arXiv"
tags: [TTS, zero-shot, open-data, data-curation, DiT, flow-matching, robustness-evaluation]
concepts: ["[[ConditionalFlowMatching]]", "[[Diffusion-basedTTS]]", "[[TTSEvaluation]]", "[[MelSpectrogram]]", "[[NeuralVocoder]]"]
models: ["[[论文笔记/F5-TTS|F5-TTS]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[论文笔记/CosyVoice2|CosyVoice 2]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/Llasa|Llasa]]", "[[论文笔记/IndexTTS2|IndexTTS2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]", "[[CV3-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: Raon-OpenTTS 属于 continuous/flow-matching TTS 路线。其 DiT 架构直接复用 F5-TTS (Chen et al., 2025),后者是 DiT + flow matching 的代表性 NAR TTS 系统。论文核心贡献不在架构创新,而在 **开放数据规模化** — 用 510K-615K 小时全开放数据训练,挑战 CosyVoice 3 (~1M h 闭源) 和 Qwen3-TTS (~5M h 闭源) 的性能。

**已有认知**:
- [[ConditionalFlowMatching]]✓: CFM/flow matching 在 TTS 中已成主流声学生成方法,F5-TTS 采用 DiT+FM 实现高效合成。Raon-OpenTTS 沿用此路线,未改架构。
- [[Zero-shotSpeechSynthesis]]✓: 零样本 TTS 当前 SOTA 由 CosyVoice 3/Qwen3-TTS/PilotTTS 把持,均依赖大规模闭源数据。开放数据模型 (F5-TTS, MaskGCT) 与闭源模型存在明显差距。
- [[Diffusion-basedTTS]] [待确认]: FM 已取代传统 DDPM 成为主流,DiT 架构从 DiTTo-TTS 到 F5-TTS 到 CosyVoice 3 一路扩展。
- [[TTSEvaluation]] [待确认]: 当前评估存在标准化问题,WER/SIM 作为主要客观指标各有局限。评估鲁棒性是新兴关注点。
- [[Emilia]] [待确认]: 100K h 开放 TTS 数据集,是 F5-TTS/MaskGCT 的主要训练集。Raon-OpenTTS-Pool 包含 Emilia 作为组成数据源之一 (47K h 英语子集)。
- [[SEED-TTS-Eval]]✓: 标准零样本 TTS 评估基准,含 test-en/test-zh/test-hard 三子集。

**创新判断**: 本文的核心贡献是证明了"开放数据可以缩小与闭源数据的差距"。与 F5-TTS (100K h Emilia) 相比,Raon-OpenTTS 在相同架构下通过 5x 数据规模 + 多源聚合 + 质量过滤实现了显著性能提升。这回应了 TTS 领域长期未被系统验证的关键问题。

> 检索命中: [[ConditionalFlowMatching]]✓, [[Zero-shotSpeechSynthesis]]✓, [[SEED-TTS-Eval]]✓, [[Emilia]] [待确认], [[Diffusion-basedTTS]] [待确认], [[TTSEvaluation]] [待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个系统性验证大规模开放数据可训练出与闭源数据 TTS 模型性能可比的零样本 TTS,发布 615K h 开放数据集 + 过滤管线 + DiT 模型 (0.3B/1B)
> - **路线**: 11 个开放语音语料库 → Raon-OpenTTS-Pool (615K h) → 三维过滤 (DNSMOS/WER/SR, 15th percentile) → Raon-OpenTTS-Core (510K h) → DiT (F5-TTS 架构) + flow matching → mel spectrogram → HiFi-GAN 16kHz
> - **指标**: Seed-TTS-Eval WER 1.78% / SIM 0.749 (1B); CV3-Hard WER 6.15% / SIM 0.775 (1B, 双第一); Raon-Eval Overall WER 2.81% / SIM 0.695 (1B, 最佳) [Table 1, 5, 6]
> - **可借鉴**: (1) 多源开放数据的聚合策略和 combined rank-based 三维过滤 (15th percentile); (2) YouTube-Commons 预处理管线 (UVR-MDX → PyAnnote → Silero VAD → Whisper-large-v3); (3) Raon-OpenTTS-Eval 四声学域 (Clean/Noisy/Wild/Expressive) 6K prompt 评估方法论
> - **局限**: 仅英语; 无架构创新 (直接用 F5-TTS); HiFi-GAN 16kHz vocoder 限制音频带宽; 训练算力需求 9K GPU-hours (B200); CMOS 主观评估仅 second-best (MaskGCT/Qwen3-TTS/VoxCPM 在部分条件下更优)

## 核心问题

本文要回答的核心问题是: **大规模开放数据能否缩小与闭源数据 TTS 模型的性能差距?** 具体拆解为:

1. 在固定架构 (DiT/F5-TTS) 下,数据规模从 100K h (Emilia) 扩展到 510K h 带来多大提升?
2. 多源数据聚合 vs 单一数据源,在相同数据量下谁更优?
3. 什么样的过滤策略能在保留数据多样性的同时去除低质量样本?
4. 开放数据模型在多种声学条件下 (Clean/Noisy/Wild/Expressive) 的鲁棒性如何?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Raon-OpenTTS 的模型架构**完全复用 F5-TTS**,是 DiT (Diffusion Transformer) + flow matching 的非自回归 TTS 系统 [§3.3]。作者明确表示"intentionally adopt the original architecture without alteration in order to isolate the effect of data scale and coverage"[§3.3]。

- **输入**: 字符级文本 (vocabulary 5,512) + 参考语音 mel spectrogram
- **生成**: 80-channel log mel-spectrogram @ 16kHz, hop size 256
- **声码器**: HiFi-GAN (pretrained on LibriTTS, 16kHz)
- **推理**: 32 NFE steps (ODE sampling)
- **模型规模**:
  - 0.3B: 22 layers, 16 heads, dim 1024, FFN 2048 (与 F5-TTS 原始配置相同) [Table 11]
  - 1B: 28 layers, 22 heads, dim 1408, FFN 5632 [Table 11]

[agent 解读] 这是一个有意的实验设计选择 — 通过控制架构变量,使所有性能差异可归因于数据因素。这使得论文的数据贡献更有说服力。

### 关键设计选择

#### 1. 数据收集: 多源聚合策略

Raon-OpenTTS-Pool 由 11 个公开英语语音数据集聚合而成,总计 615K h / 240M segments [§3.1, Table 2]:

- **TTS 导向数据**: LibriTTS-R (552h), HiFiTTS2 (37K h) — 高质量有声书录音
- **ASR 导向数据**: GigaSpeech (10K h), People's Speech (28K h), LibriHeavy (42K h) — 扩展说话人/域/声学多样性
- **In-the-wild 数据**: Raon-YouTube-Commons (335K h, 占 54.5%) — 最大单一来源
- **多语言数据的英语子集**: Emilia (47K h), Emilia-YODAS (92K h), VoxPopuli (17K h)

筛选标准: 仅收录 >500h 的数据集,语音段限制 <30s [§3.1]。排除 Common Voice (避免与 Seed-TTS-Eval 数据泄露) [§3.1]。

[论文原文] 作者认为"including many small datasets increases dataset fragmentation and management complexity",因此设置 500h 下限 [§3.1]。

[agent 解读] 这个 500h 阈值比较保守,可能遗漏一些高质量小规模数据集 (如 DAPS, VCTK 等),但考虑到 Pool 已达 615K h,边际收益确实有限。

#### 2. YouTube-Commons 预处理管线

YouTube-Commons 是 long-form 录音 (15 min ~ 数小时),需要完整的预处理管线 [§3.1]:

(a) Audio standardization: 16kHz mono + loudness normalization
(b) Source separation: UVR-MDX (去背景音乐/非人声)
(c) Speaker diarization: PyAnnote 3.1 (确保每段单一说话人)
(d) VAD segmentation: Silero VAD (切分为 3-30s 片段)
(e) ASR transcription: Whisper-large-v3

[agent 解读] 这个管线与 Emilia-Pipe 高度相似 (6 步 vs 5 步,工具选择也接近)。主要区别在于: (1) Emilia-Pipe 用 WhisperX + Whisper-Medium,Raon 用 Whisper-large-v3; (2) Emilia-Pipe 采样率 24kHz,Raon 统一到 16kHz。管线本身没有方法论创新,但在 YouTube-Commons 上的规模化应用 (335K h) 是新的。

#### 3. 三维质量过滤

从 Raon-OpenTTS-Pool 过滤出 Raon-OpenTTS-Core (510K h / 194M segments, 保留率 84.7%) [§3.2]:

三个质量维度:
- **WER-based**: Whisper-small 转写 vs 原始标注,过滤严重 mismatch (阈值 0.35)
- **DNSMOS-based**: 感知音质,过滤严重噪声/失真 (阈值 2.24)
- **SR (Speech Ratio) -based**: Silero VAD 估计语音占比,过滤非语音 (阈值 0.79)

**Combined filtering** (最优策略): 三维各自计算 absolute rank → 取平均 rank → 移除 bottom 15th percentile [§3.2]。

[论文原文] 作者解释 combined filtering 优于单维度过滤的原因是"prevents any single noisy signal from dominating the filtering decision and yields more stable performance"[§3.2]。

[论文原文] 15th percentile 阈值被选择是因为"lies near the low-quality tail of each empirical distribution rather than the main body"[§3.2]。

过滤消融结果 (Table 3): Combined 15% 在 9 种策略中 average rank 最优 (3.40); 过度过滤 (50%) 反而降低性能 (average rank 4.90),说明数据多样性的保留很重要。

Per-dataset 保留率差异显著 [Table 4]: LibriTTS-R 97.7% → People's Speech (Dirty) 48.2%。YouTube-Commons 保留 82.9%,说明预处理管线有效。

### 训练策略

- **训练硬件**: NVIDIA B200 GPUs, distributed training [§3.3]
- **0.3B**: 225K steps, batch 35K frames/GPU, LR 7.5e-5, 50K warmup + linear decay; ~1K GPU-hours
- **1B**: 550K steps, batch 14K frames/GPU, LR 1e-4, 50K warmup + linear decay; ~9K GPU-hours
- **Gradient clipping**: max norm 1.0
- **音频存储**: 64kbps Opus 格式 (存储效率优化) [§3.1]

[agent 解读] 1B 模型的 9K GPU-hours 在当前大规模 TTS 训练中算中等。CosyVoice 3 的 1.5B 模型在 ~1M h 数据上训练,算力成本估计高一个量级。Raon 通过选择 16kHz 采样率和 80-channel mel 在效率上做了折中。

## 实验

### Seed-TTS-Eval [Table 1]

| 指标 | 本文 (1B) | 本文 (0.3B) | F5-TTS | CosyVoice 3 (0.5B) | Qwen3-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER (%) ↓ | **1.78** | 1.95 | 2.04 | 2.50 | **1.46** | Seed-TTS-Eval-EN | [Table 1] |
| SIM ↑ | **0.749** | 0.687 | 0.671 | 0.698 | 0.715 | Seed-TTS-Eval-EN | [Table 1] |

Raon-1B 在 open-weight 模型中 SIM 第一、WER 第二 (仅次于 Qwen3-TTS)。0.3B 已超越同架构同规模的 F5-TTS (WER 1.95 vs 2.04, SIM 0.687 vs 0.671),说明数据规模的收益独立于模型规模。

### CV3-Hard-EN [Table 5]

| 指标 | 本文 (1B) | VoxCPM | Qwen3-TTS | CosyVoice 3 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER (%) ↓ | **6.15** | 6.44 | 7.89 | 10.77 | CV3-Hard-EN | [Table 5] |
| SIM ↑ | **0.775** | 0.670 | 0.666 | 0.740 | CV3-Hard-EN | [Table 5] |
| DNSMOS ↑ | 3.85 | 3.78 | 3.87 | **3.98** | CV3-Hard-EN | [Table 5] |

CV3-Hard 上 Raon-1B WER 和 SIM 双第一,展示了在复杂语言文本上的优势。F5-TTS 因无法处理长文本输入而在 CV3-Hard-EN 上大部分样本生成失败 [Table 5 footnote]。

### Raon-OpenTTS-Eval [Table 6]

| 指标 | 本文 (1B) | CosyVoice 3 | VoxCPM | F5-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Overall WER (%) ↓ | **2.81** | 4.43 | 9.48 | 25.08 | Raon-Eval | [Table 6] |
| Overall SIM ↑ | **0.695** | 0.647 | 0.642 | 0.542 | Raon-Eval | [Table 6] |
| Wild WER (%) ↓ | **5.61** | 8.31 | 43.83 | 136.03 | Raon-Eval Wild | [Table 6] |
| Expressive WER (%) ↓ | **2.77** | 5.49 | 2.66 | 3.46 | Raon-Eval Expressive | [Table 6] |

Wild 条件下的优势尤为突出 — F5-TTS WER 136.03% (近乎崩溃) vs Raon 5.61%,说明多源 in-the-wild 训练数据对鲁棒性至关重要。多个 AR baseline (CosyVoice 2 Wild WER 49.73%, Qwen3-TTS 79.14%) 在 Wild 条件下均显著退化,凸显了非自回归 + 多源训练的鲁棒性优势 [§5.3]。

### 数据消融

**多源 vs 单源 @ 47K h** [Table 9]: Pool-Matched-47K Overall WER 3.09% > Emilia 3.33%; SIM 0.605 > 0.558。在 Wild 和 Expressive 条件下提升更显著。

**YouTube-Commons 的贡献** [Table 10]: 加入 YouTube-Commons (615K vs 280K h) 后 Wild WER 从 7.62% 降至 6.15%,Clean WER 从 2.17% 降至 1.72%。但 Noisy WER 从 4.21% 升至 6.79%,说明 in-the-wild 数据并非所有声学域都受益。

### 主观评估 [Table 7, 8]

CMOS (相对 Raon-1B): MaskGCT -0.01 (最接近), CosyVoice 3 -0.13, F5-TTS -0.68。
SMOS: Raon-1B 3.70 (最高), Qwen3-TTS 3.59, MaskGCT 3.58。

## 局限性

1. **仅英语**: 未验证多语言场景,而 CosyVoice 3/Qwen3-TTS 支持多语言 [§7]
2. **无架构创新**: 模型完全复用 F5-TTS,未探索更高效架构 (如 Mamba-Transformer hybrid、一致性蒸馏等)
3. **16kHz 限制**: HiFi-GAN vocoder 仅支持 16kHz,音频带宽受限;当前 SOTA 多使用 24kHz 或更高
4. **YouTube-Commons 的噪声问题**: Noisy 条件下加入 YC 数据反而恶化 WER (4.21% → 6.79%) [Table 10],说明 in-the-wild 数据的域分布控制仍有改进空间
5. **过滤策略偏保守**: 仅移除 bottom 15%,未探索数据恢复/修正技术 [§7]
6. **无流式推理**: 论文未讨论流式/低延迟部署
7. **CMOS 仅 second-best**: 虽然客观指标领先,但人类偏好评分仅与 MaskGCT (-0.01) 和 VoxCPM (-0.05) 打平

## 点评

**核心价值**: 这是 TTS 领域的 "DataComp" 时刻 — 就像 DataComp/FineWeb 对 LLM 的意义一样,Raon-OpenTTS 系统性地证明了开放数据通过规模化和精选可以接近闭源数据的性能。这对可复现研究意义重大。

**方法论贡献**: (1) Combined rank-based filtering 作为多维数据过滤策略简单有效; (2) Raon-OpenTTS-Eval 的四域评估框架 (Clean/Noisy/Wild/Expressive, 6K prompts) 比 Seed-TTS-Eval 的单域评估更全面,有望成为社区标准。

**数据 vs 架构的启示**: 0.3B 模型在完全相同架构下已大幅超越 F5-TTS (同 0.3B),证明在当前 TTS 领域,数据规模和质量的边际收益仍大于架构优化。这与 LLM 领域的 scaling law 发现一致。

**不足**: 论文没有探索 data mixing ratio 优化 (不同数据源的最优配比),仅做了"加不加 YouTube-Commons"的二元对比。此外,Noisy 条件下加入 in-the-wild 数据反而恶化性能这个反直觉发现值得更深入分析。

## 可复用的 idea

1. **Combined rank-based 多维过滤**: 对每个样本在每个质量维度上计算 rank → 平均 rank → 统一阈值过滤。比单维度过滤更鲁棒,可应用于任何多维数据过滤场景
2. **YouTube-Commons 预处理管线**: UVR-MDX → PyAnnote → Silero VAD → Whisper-large-v3 的五步流程,可直接复用于大规模 in-the-wild 语音数据构建
3. **四声学域评估框架**: Clean/Noisy/Wild/Expressive 分域评估,每域多数据集,通过 stratified sampling 确保覆盖。可作为任何零样本 TTS 系统的标准化鲁棒性测试
4. **控制变量的实验设计**: 固定架构仅变数据,是验证数据贡献的教科书级实验设计

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法 WHY/HOW 清晰,速查卡片具体 |
> | 可信赖 | pass | 数字标注覆盖率高;初始 2 处表格数字错误已修正 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注清晰 |
> | 可定位 | pass | KB 背景谱系定位精准 |
> | 不污染 | pass | 仅追加更新,无新建概念页 |
> 
> Issues: 4 (high: 2 fixed, medium: 0, low: 2 open)
> 详见 `_review/Raon-OpenTTS-review.yml`
