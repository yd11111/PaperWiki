---
type: paper
tier: deep
title: "TTSDS - Text-to-Speech Distribution Score"
aliases: [TTSDS, Text-to-Speech Distribution Score, TTS Distribution Score]
arxiv_id: "2407.12707"
source: "Sources/TTSDS.pdf"
authors: [Christoph Minixhofer, Ondřej Klejch, Peter Bell]
year: 2024
venue: "INTERSPEECH 2024"
tags: [TTS-evaluation, benchmark, distributional-metrics, Wasserstein-distance, objective-metrics, MOS-correlation, factor-analysis]
concepts: ["[[TTSEvaluation]]", "[[Self-SupervisedSpeechRepresentation]]", "[[ProsodyModeling]]", "[[SpeakerEmbedding]]", "[[SpeakerVerification]]"]
models: ["[[模型库/HuBERT|HuBERT]]", "[[模型库/wav2vec2.0|wav2vec 2.0]]", "[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个实体页: [[TTSEvaluation]], [[Self-SupervisedSpeechRepresentation]], [[SpeakerEmbedding]], [[ProsodyModeling]], [[SpeakerVerification]])
> 自动生成,不保证完整覆盖所有相关知识。

- **[[TTSEvaluation]]** [pending-review]: TTSDS 是 distributional TTS evaluation 路线的开创性工作,提出"分布距离替代逐样本评分"的范式。后续 TTSDS2 (ICLR 2026) 基于此改进,移除了 ENVIRONMENT 因子、改用 ASR 激活值替代 WER。该概念页已记录 TTS 评估从 MOS → Predicted MOS → LLM-as-Judge → Distributional 的演进线。TTSDS 是这条演进线中 distributional 范式的起点 [待确认]
- **[[Self-SupervisedSpeechRepresentation]]** [pending-review]: TTSDS 的 GENERAL 因子直接使用 HuBERT base 和 wav2vec 2.0 base 的中间层表征作为分布特征。PROSODY 因子也使用了 HuBERT token length 作为时长代理。SSL 模型在此充当特征提取器而非下游任务模型 [待确认]
- **[[ProsodyModeling]]** [confirmed]: TTSDS 的 PROSODY 因子通过三类特征衡量韵律: WORLD F0 (pitch)、Masked Prosody Model (SSL prosody representations)、HuBERT token length (segmental duration 代理)。这与概念页中"韵律的物理维度: Duration/Pitch/Energy"对应前两维
- **[[SpeakerEmbedding]]** [confirmed]: TTSDS 的 SPEAKER 因子使用 d-vector 和 WeSpeaker 两种 speaker encoder。概念页记录了 d-vector 是 DNN 倒数第二层输出的早期 speaker embedding 方法; WeSpeaker 是更新的 speaker verification 系统
- **[[SpeakerVerification]]** [pending-review]: TTSDS 的 SPEAKER 因子本质上利用了 speaker verification 系统的表征来衡量合成语音与真实语音的说话人分布差异,而非传统的逐样本 SECS 比较 [待确认]

**谱系定位**: TTSDS 处于 TTS 评估从"逐样本 MOS/predicted MOS"到"分布级评估"的范式转折点。它从计算机视觉的 FID (Fréchet Inception Distance) 和音频领域的 Fréchet Audio Distance 汲取灵感,但创新在于(1)多因子分解而非单一距离,(2)在 TTS 低样本量下仍能保持鲁棒相关性。TTSDS2 继承并扩展了这一范式。

> [!summary] 速查
> - **一句话**: 首个将 TTS 评估从逐样本 MOS 推向分布级多因子评估的 benchmark,通过 Wasserstein 距离衡量合成语音在 5 个因子上与真实语音的分布差异,在 35 个跨时代 TTS 系统上实现 0.60-0.83 的 Spearman 相关
> - **路线**: TTS 合成音频 → 5 因子 (Environment/Speaker/Prosody/Intelligibility/General) x 多特征 → 每个特征计算与真实语音和噪声的 Wasserstein-2 距离 → 归一化 0-100 分 → 因子均值 = TTSDS
> - **指标**: Spearman ρ=0.60 (Blizzard'08 MOS) / 0.79 (BTTF MOS) / 0.83 (TTS Arena Elo); 每个数据集仅需 80-100 样本; 35 个 TTS 系统 (2008-2024) [Fig 2]
> - **可借鉴**: (1) 分布级评估而非样本级评估,天然处理 one-to-many; (2) 因子化分解使评估可解释("prosody 好但 environment 差"); (3) 噪声锚点归一化使分数具有绝对含义 (>50=更像真实语音)
> - **局限**: 需 80-100 样本/系统才能计算; 不适合单样本评估; 个别因子 (General, Environment) 独立相关性不稳定; OpenVoice v2 得分与 TTS Arena 排名不一致

## 核心问题

TTS 评估面临三个结构性困境 [§1]:

1. **MOS 不可跨研究比较**: 不同研究的 MOS 因评估协议、听众、条件差异而无法横向对比。Le Maguer et al. (2022) 已证明这一点 [ref 5]
2. **单一客观指标不泛化**: 现有 MOS prediction 网络 (UTMOS, WVMOS) 在训练域表现良好,但跨域/跨时代表现不稳定 (UTMOS: 0.05-0.85 波动) [§4]
3. **缺乏因子化分析**: 现有评估给出单一分数,无法回答"为什么 A 比 B 好" — 是韵律更好还是可懂度更高? [§1]

**论文核心主张**: TTS 评估应从"单一 MOS 分数"转向"多因子分布距离",因为 TTS 是 one-to-many 问题 — 同一文本可有多种合法语音,逐样本比较无法捕捉这种多样性 [§2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

TTSDS 将 TTS 评估定义为"合成语音分布与真实语音分布的距离",分解为 5 个因子,每个因子用 2-3 个特征衡量 [§2, Table 1]:

| 因子 | 特征 | 衡量维度 |
|------|------|----------|
| **General** | HuBERT (base) 中间层 + wav2vec 2.0 (base) 中间层 | 总体 SSL 表征分布相似性 [§2.1] |
| **Environment** | VoiceFixer+PESQ + WADA SNR | 噪声/伪影水平 [§2.1] |
| **Prosody** | WORLD F0 + Masked Prosody Model + HuBERT token length | 韵律 (pitch + SSL prosody + duration) [§2.1] |
| **Intelligibility** | wav2vec 2.0 ASR WER + Whisper (small) ASR WER | 可懂度 [§2.1] |
| **Speaker** | d-vector + WeSpeaker | 说话人身份保真度 [§2.1] |

### 关键设计选择

**WHY 分布距离而非样本距离**: 作者认为 TTS 是 one-to-many mapping — 同一文本可生成多种合法语音。因此不应比较单个合成样本与参考样本的距离,而应比较整体分布 [论文原文, §2]。[agent 解读] 这也消除了参考文本对齐的需求,合成语音和参考语音甚至不需要同一文本。

**WHY 多因子而非单一距离**: 作者指出类似 FID 的单一分布距离在语音领域未被广泛采用,原因之一是语音质量由多个独立维度决定 (韵律/可懂度/说话人/环境),单一距离无法捕捉这种多维性 [论文原文, §1]。因子化设计使用户可根据应用场景选择关注特定维度 (如游戏角色配音关注 prosody,语言学习应用关注 intelligibility) [论文原文, §1]。

**WHY Wasserstein-2 距离**: 选择 2-Wasserstein 距离 (W2) 而非 KL 散度或 JS 散度,因为 W2 是对称的,且能区分不重叠的分布 (KL 散度在此情况下为无穷大) [论文原文, §2.3]。对于高维特征 (如 SSL 表征),使用 Frechet distance 近似 (假设高斯分布) [论文原文, §2.3, Eq 2]。

**WHY 噪声锚点归一化**: 不直接报告 W2 距离,而是与噪声数据集的距离做归一化 [Eq in §2.4]:

```
S = 100 × W_noise / (W_real + W_noise)
```

S>50 意味着合成语音更接近真实语音而非噪声。[agent 解读] 这个设计提供了绝对可解释性 — 分数不依赖于被比较的其他系统,而是有固定的语义锚点。

### 距离计算

**一维特征** (如 F0, WER, SNR): 对排序后的样本直接计算 W2 [Eq 1, §2.3]:

$$W_2(\hat{P}_1, \hat{P}_2) = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - y_i)^2}$$

**多维特征** (如 SSL 表征, speaker embedding): 假设高斯分布,使用 Frechet distance 计算 [Eq 2, §2.3]:

$$W_2(\hat{P}_1, \hat{P}_2) = \sqrt{\|\mu_1 - \mu_2\|^2 + D_B(\Sigma_1, \Sigma_2)}$$

其中 $D_B$ 是 unnormalized Bures metric [§2.3]。

### 评分流程

1. 对每个 system × feature 组合,从 80-100 个样本中提取特征
2. 计算与最近真实语音数据集的 W2 距离 ($W_{real}$) 和最近噪声数据集的 W2 距离 ($W_{noise}$) [§2.4]
3. 归一化为 0-100 分 [§2.4]
4. 同一因子内的多个特征取平均 → 因子分数
5. 所有因子分数取平均 → TTSDS 总分 [§3]

### 训练策略

TTSDS 不涉及训练 — 它是一个**推理时 (test-time) 评估工具**,所有特征提取器 (HuBERT, wav2vec 2.0, Whisper, d-vector, WeSpeaker 等) 均为预训练冻结模型 [论文原文, §2.1]。

## 实验

### 验证数据集

TTSDS 使用三个跨时代的数据集验证,覆盖 2008-2024 年的 35 个 TTS 系统 [§3]:

| 数据集 | 时期 | 系统数 | 主观指标 | 系统类型 |
|--------|------|--------|----------|----------|
| Blizzard 2008 | 2008 | 22 | MOS | Unit selection + HMM [§3] |
| BTTF (Back to the Future) | 2013+2021 | 混合 | MOS | Unit selection + Neural (FastPitch/Tacotron) [§3] |
| TTS Arena | 2023-2024 | 9 | Elo Rating | LLM-based TTS (StyleTTS 2, XTTSv2 等) [§3] |

参考数据集: LibriTTS, LibriTTS-R, LJSpeech, VCTK, Blizzard 训练集 (各随机抽 100 句) [§3]。
噪声数据集: ESC 背景噪声 + 均匀噪声 + 正态噪声 + 全零 + 全一 [§3]。

### Spearman 相关性

| 指标 | Blizzard'08 | BTTF | TTS Arena | 出处 |
|------|-------------|------|-----------|------|
| **TTSDS** | **0.60** | **0.79** | **0.83** | [Fig 2] |
| UTMOS | ~0.85 | ~0.10 | ~0.05 | [Fig 2] |
| WVMOS | ~0.05 | ~0.80 | ~0.10 | [Fig 2] |
| General | ~0.30 | ~0.50 | ~0.40 | [Fig 2] |
| Prosody | ~0.45 | ~0.50 | ~0.60 | [Fig 2] |
| Speaker | ~0.55 | ~0.10 | ~0.55 | [Fig 2] |

### TTS Arena 系统排名 [Table 2]

| 系统 | UTMOS | WVMOS | Gen | Env | Int | Pro | Spk | TTSDS | Elo |
|------|-------|-------|-----|-----|-----|-----|-----|-------|-----|
| StyleTTS 2 | 4.36 | 4.48 | 93.7 | 84.7 | 91.6 | 89.8 | 71.5 | 86.3 | 1237 |
| XTTSv2 | 3.89 | 4.36 | 94.3 | 79.3 | 91.4 | 90.5 | 72.6 | 85.6 | 1232 |
| OpenVoice | 4.10 | 4.57 | 91.7 | 88.0 | 91.6 | 91.8 | 68.8 | 86.4 | 1158 |
| Parler TTS | 3.97 | 4.16 | 94.7 | 80.8 | 87.5 | 83.0 | 74.1 | 84.0 | 1140 |
| OpenVoice v2 | 4.29 | 4.75 | 90.7 | 91.2 | 91.6 | 88.6 | 68.7 | 86.2 | 1120 |

### 因子相关性随时间的变化 [Fig 3]

关键发现 [论文原文, §4.1]:

1. **Environment 因子在 BTTF 中最重要**: 可能因为 2013 系统有明显伪影而 2021 系统没有 [§4.1]
2. **Prosody 因子随时间增强**: 从 Blizzard'08 到 TTS Arena,prosody 与主观评估的相关性持续上升,表明评估者在其他因素改善后更关注韵律 [§4.1]
3. **Speaker 因子在 BTTF 中失效**: unit selection 系统拼接真实语音片段,自然产生逼真的 speaker embedding,导致 speaker 分数无法区分好坏 [§4.1]
4. **个别因子独立相关性有限**: 但组合后 (TTSDS) 显著优于单一因子,说明多因子组合是关键 [§4.1]

### Wilcoxon 显著性检验 [Fig 4]

系统间的特征级显著差异检验显示: 最差系统可与最好系统显著区分,但**高性能系统之间无显著差异** [论文原文, §4.1]。作者指出这与历史上的主观评估面临同样困难 [§4.1]。

## 局限性

1. **需要 80-100 样本**: 无法评估单个合成样本或极少样本场景 [§2.4]
2. **个别因子不稳定**: General 和 Environment 因子在不同数据集上的相关性波动大 (General: 0.30-0.50; Environment: 高在 BTTF 但低在其他) [Fig 2]
3. **高性能系统难区分**: Wilcoxon 检验显示 top 系统间无显著差异,这是分布方法在有限样本下的固有限制 [Fig 4]
4. **OpenVoice v2 异常**: TTSDS 给出高分 (86.2) 但 TTS Arena Elo 排名低 (1120),作者推测是配置差异 [Table 2]
5. **BTTF 中 Speaker/Intelligibility 失效**: unit selection 系统拼接真实语音使 speaker 分数虚高; intelligibility 出现负相关 [§4.1]
6. **未测试多语言场景**: 仅在英语数据上验证 [§3]
7. **特征提取器偏见**: 所有特征提取器均在特定数据上训练,可能存在域偏见 [agent 解读]

## 点评

**贡献的根本价值**: TTSDS 的核心贡献不在于具体的特征选择或距离计算(这些都可以替换),而在于提出了一个范式转换 — 从"给合成样本打分"到"比较合成分布与真实分布"。这个范式具有三个根本优势: (1) 天然处理 one-to-many,(2) 不需要参考文本对齐,(3) 提供可解释的因子分解。TTSDS2 的改进(移除 Environment、更换特征)恰恰证明了这个范式框架的持久价值,而非具体实现的重要性。

**与 TTSDS2 的关系**: TTSDS 是概念验证 (proof of concept),TTSDS2 是工程化升级。TTSDS 用 5 因子在 35 个系统上验证了分布评估的可行性; TTSDS2 基于经验移除了不稳定的 Environment 因子,改进了 Prosody (speaking-rate 替代 HuBERT token length) 和 Intelligibility (ASR 激活值替代 WER),并扩展到多语言、多域。

**方法论洞察**: 每个因子用多个特征 (2-3 个) 衡量的设计是关键 — 作者的结果表明个别因子独立相关性有限,但组合后显著改善。这暗示 TTS 评估中的"因子"概念可能本身就不是完全独立的维度,而是互相交织的,多特征+多因子的冗余设计恰好抵消了单一特征的不稳定性。

**局限的结构性原因**: TTSDS 在 BTTF 数据集上的 Speaker/Intelligibility 失效揭示了一个深层问题 — 分布距离方法对"系统类型跨度大"的场景有风险。当评估集包含根本不同的技术路线 (unit selection vs neural),某些因子的语义可能发生变化 (unit selection 的 speaker embedding 来自真实语音片段,含义完全不同于 neural TTS 的 speaker embedding)。

## 可复用的 idea

1. **噪声锚点归一化**: 不直接报告距离,而是与噪声的距离做归一化,提供绝对可解释的分数 (>50=好)。这个 trick 可用于任何分布距离型评估 [§2.4]
2. **因子化分解 + 等权平均**: 将复杂评估分解为独立因子,每个因子用冗余特征衡量,最终等权平均。简单但有效,避免了加权带来的过拟合 [§3]
3. **跨时代验证**: 用不同时期的系统 (2008/2013+2021/2023-2024) 验证评估指标的稳定性,比仅在当代系统上验证更具说服力 [§3, §4]
4. **分布距离评估范式**: 将 TTS 评估从"给样本打分"转为"比较分布",天然解决 one-to-many 和参考对齐问题,可迁移到其他生成任务 (图像/视频/音乐生成) [§2]

---

> [!review] 审阅 — pass-with-fixes
> 审阅报告: [[_review/TTSDS-review.yml]]
> 结论: pass-with-fixes | 3 low issues (Fig 2 近似值 / venue 推断 / datasets 空)
> 5 个原则均满足,无 high/medium issue,可放行反向更新

---

检索命中: [[TTSEvaluation]](pending-review), [[Self-SupervisedSpeechRepresentation]](pending-review), [[ProsodyModeling]]✓, [[SpeakerEmbedding]]✓, [[SpeakerVerification]](pending-review) | 过滤: [[TTSEvaluation]](pending-review), [[Self-SupervisedSpeechRepresentation]](pending-review), [[SpeakerVerification]](pending-review) | 未命中但可能相关: 无
