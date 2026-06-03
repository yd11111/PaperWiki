---
title: "TTSDS2"
aliases: [TTSDS2, Text-to-Speech Distribution Score 2, TTS Distribution Score 2]
authors: [Anonymous]
year: 2026
venue: ICLR 2026 (under review)
arxiv_id: ""
source: ""
tags: [TTS-evaluation, benchmark, distributional-metrics, Wasserstein-distance, multilingual, objective-metrics, MOS-correlation]
level: deep
status: draft
created: 2026-06-03
updated: 2026-06-03
concepts: ["[[TTS Evaluation]]", "[[Self-Supervised Speech Representation]]", "[[Prosody Modeling]]", "[[Speaker Embedding]]"]
models: ["[[模型库/HuBERT|HuBERT]]", "[[模型库/WavLM|WavLM]]", "[[模型库/Whisper|Whisper]]"]
datasets: []
kb_sources: ["[[TTS Evaluation]]", "[[Self-Supervised Speech Representation]]", "[[Prosody Modeling]]", "[[Speaker Embedding]]"]
---

# TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems

## KB 背景

- **[[TTS Evaluation]]** [pending-review]: TTSDS2 是当前最全面的 TTS 客观评估指标对比研究，直接回应了 TTS 评估领域的核心痛点——MOS 不可比、客观指标跨域泛化差、缺乏多语言 benchmark [待确认]
- **[[Self-Supervised Speech Representation]]** [pending-review]: TTSDS2 的 GENERIC 和 INTELLIGIBILITY 因子直接使用 HuBERT/WavLM/wav2vec 2.0 的 SSL 表征作为特征 [待确认]
- **[[Prosody Modeling]]** [confirmed]: TTSDS2 的 PROSODY 因子衡量 pitch (WORLD F0) + speaking rate (HuBERT/Allosaurus token rate) + prosody embeddings [论文原文]
- **[[Speaker Embedding]]** [confirmed]: TTSDS2 的 SPEAKER 因子使用 d-Vector + WeSpeaker 衡量说话人身份保真度 [论文原文]

> [!summary] 速查
> - **一句话**: 首个在 20 个 TTS 系统 x 4 域 x 14 语言上全面验证的分布式客观评估指标，平均 Spearman ρ≈0.67，是唯一在所有条件下 ρ>0.5 的指标
> - **路线**: TTS 合成音频 → 5 因子 (Generic/Speaker/Prosody/Intelligibility) x 多特征 → Wasserstein-2 距离 vs 真实分布 → 归一化分数 (0-100)
> - **指标**: 平均 Spearman ρ=0.67 across MOS/CMOS/SMOS x 4 domains; 11,282 人工 MOS ratings; 14 语言 benchmark; 20 TTS 系统排名 [Table 2, 3]
> - **可借鉴**: (1) 分布级评估代替样本级评估的范式; (2) 因子化多维评估 (而非单一分数); (3) 可复现的自动化 benchmark pipeline
> - **局限**: 需 50-100 样本才能计算; 不适合评估单个样本; 多语言 MOS 人工验证缺失; ICLR 2026 在审

## 1. 核心问题 / Core Problem

**TTS 评估的三大困境** [§1.1-1.2]:
1. **MOS 不可比**: 不同论文的 MOS 评分因评估协议、听众、条件差异而无法直接比较 [§1.2]
2. **客观指标跨域脆弱**: 现有指标 (UTMOS, NISQA, PESQ 等) 在 audiobook (Clean) 域表现尚可，但在 noisy/wild/children 域急剧退化 [§3.4, Table 3]
3. **缺乏多语言 benchmark**: 无公开 TTS 评估覆盖多语言 [§1.1]

**当前 TTS 水平**: 20 个系统中 3 个达到 MOS 人类水平 (±0.05), 4 个系统合成语音被评为**优于真实录音** [§1.2]。

## 2. 方法论 / Methodology

### 2.1 TTSDS2 指标设计 [§2]

**核心思想**: TTS 合成本质是 one-to-many 问题 (同一文本可有多种合法语音)，因此评估应在**分布层面**比较合成语音与真实语音的相似度，而非逐样本比较 [§2]。

**5 个因子** [Table 1]:

| 因子 | TTSDS2 特征 | 衡量维度 |
|------|-----------|----------|
| GENERIC | WavLM (actv.) + HuBERT (base) + wav2vec 2.0 (base) | 总体分布相似性 [§2] |
| SPEAKER | d-Vector + WeSpeaker | 说话人身份保真度 [§2] |
| PROSODY | WORLD F0 + HuBERT speaking-rate + Allosaurus speaking-rate + Prosody embeddings | 韵律 (pitch, rhythm, duration) [§2] |
| INTELLIGIBILITY | wav2vec 2.0 ASR actv. + Whisper (small) ASR actv. | 可懂度 [§2] |

**vs TTSDS v1**: 移除 ENVIRONMENT 因子 (不稳定)；PROSODY 改用 speaking-rate 替代 HuBERT token length (后者对真实语音分数过低)；GENERIC 新增 WavLM；INTELLIGIBILITY 改用 ASR 模型激活值替代 WER (WER 跨域不稳定) [§2]。

### 2.2 Wasserstein-2 距离 [§2, Eq 1]

分布比较使用 2-Wasserstein 距离 (W₂)，又称 Earth Mover's Distance:
- 对称、可区分非重叠分布 (优于 KL 散度)
- 高维向量用 Frechet distance 近似 [§2]

**归一化** [Eq 1]:
```
TTSDS2(D, D̃, D^NOISE) = 100 × W₂^NOISE / (W₂^REAL + W₂^NOISE)
```
- 0-100 分，>50 表示更像真实语音而非噪声 [§2]
- D^NOISE 包含 uniform noise, Gaussian noise, all-ones, all-zeros [§2]

### 2.3 数据收集与人工评估 [§3.1-3.2]

**4 个评估域** [§3.1]:
| 域 | 来源 | 目的 |
|------|------|------|
| CLEAN | LibriTTS test split | 基线 (audiobook) [§3.1] |
| NOISY | LibriVox 2025 (无 SNR 过滤) | 噪声鲁棒性 [§3.1] |
| WILD | YouTube 2025 (inspired by Emilia) | 多样说话风格 [§3.1] |
| KIDS | My Science Tutor Corpus | 儿童语音泛化 [§3.1] |

**人工评估规模**: 200 标注员 (Prolific), 50 人/域, MOS + CMOS + SMOS, 共 11,282 条评分 [§3.2]。

## 3. 实验与结果 / Experiments

### 3.1 TTS 系统排名 [Table 2]

Top 5 (by TTSDS2 score):
| 系统 | MOS | CMOS | SMOS | TTSDS2 |
|------|-----|------|------|--------|
| Ground Truth | 3.70 | 0.00 | 4.37 | 93.21 |
| E2-TTS | **3.41** | -0.23 | **4.37** | **91.73** |
| Vevo | 3.36 | **0.08** | 4.01 | 90.20 |
| F5-TTS | 3.33 | -0.34 | 4.10 | 91.16 |
| MaskGCT | 3.28 | -0.17 | 4.39 | **91.76** |

Bottom 3:
| 系统 | MOS | TTSDS2 |
|------|-----|--------|
| NaturalSpeech2 | 2.05 | 81.71 |
| SpeechT5 | 1.98 | 84.84 |
| ParlerTTS | 2.39 | 84.88 |

### 3.2 跨域 Spearman 相关性 [Table 3]

| 指标 | Clean | Noisy | Wild | Kids | **Avg** |
|------|-------|-------|------|------|---------|
| **TTSDS2** | **0.75/0.69/0.73** | **0.59/0.54/0.71** | **0.75/0.71/0.75** | **0.61/0.50/0.70** | **~0.67** |
| TTSDS v1 | 0.60/0.62/0.52 | 0.49/0.61/0.66 | 0.57/0.67/0.70 | 0.52/0.60/0.60 | ~0.58 |
| X-Vector | 0.46/0.42/0.56 | 0.40/0.29/0.77 | 0.82/0.62/0.70 | 0.57/0.55/0.45 | ~0.55 |
| SQUIM | 0.68/0.37/0.46 | 0.48/0.60/0.62 | 0.79/0.75/0.57 | 0.55/0.72/0.77 | ~0.57 |
| UTMOSv2 | 0.39/0.25/0.09 | 0.36/0.19/0.16 | 0.12/0.03/0.05 | 0.03/0.03/-0.05 | ~0.12 |

**核心发现** [§3.4]:
- TTSDS2 是唯一在所有 4 个域 x 3 个主观指标上 ρ>0.5 的客观指标 [§3.4]
- Speaker Similarity (X-Vector, RawNet3) 在 Wild 域强但 Clean 域弱 [§3.4]
- MOS prediction 网络 (UTMOSv2, NISQA) 在 KIDS 域完全失效 (接近 0 或负相关) [§3.4]
- Signal-based (PESQ, MCD) 整体表现最差 (多数负相关) [Table 3]

### 3.3 多语言 benchmark [§4, Fig 3]

覆盖 14 语言，2+ 系统/语言:
- Top: Chinese (n=6), English (n=20), Japanese (n=4)
- 语言间 TTSDS2 分数变化可由 typological distance 解释 (Uriel+ 距离与 TTSDS2 相关 ρ=-0.39/-0.51) [§4.2]

### 3.4 可复现 benchmark pipeline [§4.1, Algorithm 1]

自动化流程:
1. YouTube 关键词搜索 (10 keywords x 14 languages) → 视频抓取
2. Whisper diarization → 单说话人提取
3. FastText LID → 语言过滤
4. XNLI → 内容过滤 (争议话题)
5. Pyannote → crosstalk 过滤; Demucs → 音乐过滤
6. 50 speaker pairs/language → Reference + Synthesis 分割
7. TTS 合成 → TTSDS2 计算

**设计意图**: 每季度可重新运行，避免数据污染 (模型训练数据泄漏到测试集) [§4.1]。

## 4. 设计选择分析 / Design Analysis

### WHY: 分布级评估而非样本级

- TTS 是 one-to-many: 同一文本可有多种合法语音，逐样本比较无法捕捉这种多样性 [§2]
- 分布级评估只需 50-100 样本即可稳定 (得益于多特征集成) [§2]
- 消除了参考语音的内容匹配需求 (non-matching reference) [§2]

### WHY: 因子化设计

- 单一分数无法解释**为什么**一个系统好/差 [§1.4]
- 5 因子分别衡量 generic similarity / speaker identity / prosody / intelligibility [§2]
- 用户可关注特定维度 (如只看 prosody 或 speaker) [§2]

### WHY: 移除 ENVIRONMENT 因子

- TTSDS v1 的 ENVIRONMENT (VoiceFixer+PESQ, WADA-SNR) 在跨域测试中不稳定 [§2]
- 移除后总体相关性反而提高 [§2]

## 5. 与已有方法的关键差异

| 维度 | TTSDS v1 | TTSDS2 |
|------|----------|--------|
| Avg ρ | 0.58 | **0.67** (+10%) [§1.3] |
| 因子数 | 5 | 4 (移除 Environment) [Table 1] |
| 韵律特征 | HuBERT token length | Speaking-rate + Allosaurus [Table 1] |
| Generic | HuBERT + wav2vec 2.0 | + WavLM [Table 1] |
| 可懂度 | WER | ASR model activations [Table 1] |
| 多语言 | 无 | 14 语言 benchmark [§4] |
| 人工标注 | 无公开 | 11,282 ratings [§3.2] |
| 可复现 | 手动 | 自动化 pipeline (Algorithm 1) [§4.1] |

## 6. 局限与未来方向

- **不适合单样本评估**: 需 50-100 样本才能计算可靠分数 [§2]
- **多语言 MOS 未验证**: 14 语言 benchmark 缺乏对应人工 MOS 验证 [§4.2]
- **模型依赖**: 特征提取器 (HuBERT, WavLM, Whisper) 的偏见可能传递到评估 [论文原文]
- **ICLR 2026 在审**: 匿名论文，代码/数据链接匿名化 [论文原文]

---

检索命中: [[TTS Evaluation]](pending-review), [[Self-Supervised Speech Representation]](pending-review), [[Prosody Modeling]], [[Speaker Embedding]] | 过滤: [[TTS Evaluation]](pending-review), [[Self-Supervised Speech Representation]](pending-review) | 未命中但可能相关: 无
