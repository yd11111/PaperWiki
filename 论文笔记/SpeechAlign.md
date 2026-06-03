---
type: paper
tier: deep
title: "SpeechAlign: Aligning Speech Generation to Human Preferences"
arxiv_id: "2404.05600"
source: "https://arxiv.org/abs/2404.05600"
authors: [Dong Zhang, Zhaowei Li, Shimin Li, Xin Zhang, Pengyu Wang, Yaqian Zhou, Xipeng Qiu]
year: 2024
venue: "arXiv"
tags: [RLHF, DPO, preference-optimization, codec-LM, zero-shot-TTS, self-improvement, distribution-gap]
concepts: ["[[Codec Language Model]]", "[[Semantic vs Acoustic Tokens]]", "[[Residual Vector Quantization]]", "[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[TTS Evaluation]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页: [[LLM-based TTS]], [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Residual Vector Quantization]], [[TTS Evaluation]])
> LLM-based TTS 将 TTS 重构为 codec language modeling,核心是 AR 模型生成 coarse tokens + NAR 模型补充细节 (VALL-E 开创)。训练时 NAR 模型使用 golden AR tokens,推理时却接收 synthetic AR tokens,产生 distribution gap。此前所有 codec LM 均通过 SFT 训练,未有工作将人类偏好学习引入 speech generation。TTS Evaluation 指出 WER 和 SIM 是标准客观指标但各有局限。
> 检索命中: [[LLM-based TTS]], [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Residual Vector Quantization]], [[TTS Evaluation]] | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Codec Language Model]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首次将人类偏好学习 (DPO/RLHF-PPO/CoH/BoN) 应用于 speech codec language model,通过 golden vs synthetic AR tokens 构建偏好数据集实现迭代自改进 [论文原文]
> - **路线**: Text → SpeechGPT AR model → AR tokens → SoundStorm NAR model → Speech; 偏好优化在 AR model 层面操作
> - **指标**: LibriSpeech: WER 6.0 / SIM 0.90 (DPO-Iter3); VCTK: WER 7.9 / SIM 0.83; 相比 baseline SFT (WER 7.2 / SIM 0.87) 显著改进 [Table 3]
> - **可借鉴**: (1) 利用 golden vs synthetic token 的天然对比构建偏好数据,无需人工标注; (2) 迭代自改进 (iterative DPO) 可持续提升性能; (3) DPO 在 codec LM 上有效
> - **局限**: (1) 仅优化 AR model,未涉及 NAR model 的偏好优化; (2) 仅在 LibriSpeech/VCTK 英语数据上验证; (3) SpeechGPT 基线较老,未在更强的 codec LM 上测试; (4) 偏好数据集仅考虑整体质量,未区分 pitch/timbre/rhythm 等细粒度维度

## 核心问题

1. **Distribution gap**: Codec LM 的 AR+NAR 两阶段范式中,NAR 模型训练时使用 golden AR tokens,推理时使用 synthetic AR tokens,两者分布不一致导致性能下降 [§2.2-2.3]
2. **如何将偏好学习引入 codec LM**: speech codec tokens 是数值序列,人类无法直接对 token 做偏好判断;如何构建有效的偏好数据集? [§3.1]
3. **持续改进**: 单次偏好优化效果有限,如何实现迭代自改进? [§3.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SpeechAlign 基于 AR+NAR 两阶段 codec LM (SpeechGPT AR + SoundStorm NAR),使用 SpeechTokenizer (8层 RVQ) 作为 speech tokenizer [§2.1]:
1. AR model 从文本生成第一层 codec tokens (AR tokens)
2. NAR model 从 AR tokens 生成后续层 codec tokens (NAR tokens)
3. Codec decoder 从所有层 tokens 重建波形

偏好优化在 AR model 层面操作:
- **Preferred data**: Golden AR tokens (从真实语音编码)
- **Dis-preferred data**: Synthetic AR tokens (由 AR model 生成)
- 人类验证确认 71% 情况下偏好 golden tokens [Table 2] [论文原文]

### 关键设计选择

**1. Distribution Gap 分析** [§2.2-2.3]
- t-SNE 可视化显示 golden 与 synthetic AR tokens 形成两个明显不同的聚类 [Fig 2(a)] [论文原文]
- NAR model 使用 golden tokens 时 WER 5.9/SIM 0.93,使用 synthetic tokens 时 WER 7.2/SIM 0.87 [Table 1],证实 distribution gap 确实降低性能 [论文原文]
- 对齐后 (DPO 优化后) 两类 tokens 聚合为一个聚类 [Fig 2(b)] [论文原文]

**2. 偏好数据构建** [§3.1]
- 从 LibriSpeech 随机采样 50K speech-text pairs
- 用 SpeechTokenizer 编码真实语音得到 golden AR tokens y_g
- 用 AR model 从文本生成 synthetic AR tokens y_s
- 构成偏好数据集 D_pf = {(x, y_g, y_s)}, 无需人工标注 [论文原文]
- 人类验证 (100 samples): 71% 偏好 golden, 21% 平手, 8% 偏好 synthetic [Table 2] [论文原文]

**3. 四种偏好优化方法** [§3.2]

| 方法 | 核心思路 | 是否需要 reward model |
|------|---------|---------------------|
| Chain-of-Hindsight (CoH) | 在 prompt 中标注 high/low quality 引导生成 | 否 |
| DPO | 直接在 preferred/dis-preferred 对上优化,隐式 reward | 否 |
| RLHF-PPO | 训练 reward model 后用 PPO 优化 | 是 |
| Best-of-N (BoN) | 采样 N 个候选,用 reward model 选最佳 | 是 (推理时) |

**4. 迭代自改进** [§3.3, Algorithm 1]
- 第 t 轮优化后得到更新的 AR model p^ar_{θ_t}
- 用 p^ar_{θ_t} 重新生成 synthetic tokens,构建新偏好数据集
- 迭代执行偏好优化,逐步将 weak model 转化为 strong model [论文原文]
- 合成数据集逐轮累积: Iter 0 = 50K, Iter 1-3 = 各加 50K → 100K [§4.1] [论文原文]

### 训练策略

- 基于 SpeechGPT AR model (预训练) 在 LibriSpeech 上 continue finetuning 3500 steps (batch 256, lr 1e-5, 8xA100) [§4.1]
- CoH: batch 32, 12000 steps
- DPO: batch 128, 2000 steps, lr 5e-7
- Reward model: batch 32, 1000 steps, lr 1e-5
- PPO: batch 16, 1000 steps, lr 1e-5

## 实验

| 指标 | SpeechAlign-DPO-Iter3 | SpeechAlign-RLHF-PPO | Baseline (SFT) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER (↓) | **6.0** | 7.1 | 7.2 | LibriSpeech | [Table 3] |
| SIM (↑) | **0.90** | 0.89 | 0.87 | LibriSpeech | [Table 3] |
| WER (↓) | **7.9** | 8.5 | 8.8 | VCTK | [Table 3] |
| SIM (↑) | **0.83** | 0.80 | 0.79 | VCTK | [Table 3] |
| Human Win Rate | **50%** | 53.33% | - | LibriSpeech | [Fig 1] |

**关键发现**:
1. **DPO 迭代有效**: Iter1 → Iter2 → Iter3 逐轮改进,WER 从 6.7 → 6.2 → 6.0,SIM 从 0.88 → 0.89 → 0.90 [Table 3] [论文原文]
2. **偏好学习 vs Continue SFT**: Continue SFT (使用相同 golden tokens) 反而使 WER 恶化至 8.0,证明改进来自偏好学习而非简单数据增强 [Table 3, §4.3] [论文原文]
3. **小模型也有效**: 130M 参数的 AR model 上 DPO-Iter1 将 WER 从 11.4 降至 9.3 [Fig 4(b)] [论文原文]
4. **泛化到 unseen speakers**: VCTK (不在训练集中的说话人) 上同样有效 [Table 3] [论文原文]
5. **偏好数据量阈值**: 50K 足以获得显著改进,250K 收益递减 [Fig 4(a), §5.1] [论文原文]

## 局限性

1. 仅优化 AR model,NAR model 的 distribution gap 也存在但未解决 [§8] [论文原文]
2. 偏好数据集仅捕获整体偏好,未建模 sound quality/rhythm/timbre 等细粒度维度 [§8] [论文原文]
3. 基线系统 (SpeechGPT + SoundStorm) 在 2024 已较旧,未在更强系统上验证
4. 仅英语验证,未测试多语言场景
5. 偏好数据集完全自构建 (golden vs synthetic),可能不完全反映真实人类偏好 [agent 解读]

## 点评

SpeechAlign 是 **首篇将偏好学习系统性引入 codec language model** 的工作,填补了 LLM-based TTS 在 alignment 方向的空白。其最大洞见在于: 利用 golden vs synthetic AR tokens 的天然对比,无需人工标注即可构建偏好数据集。迭代 DPO 的持续改进也很有说服力。

**与后续工作的关系**:
- SpeechAlign 在 token 层面做 DPO,后续 CosyVoice 3 的 [[Differentiable Reward Optimization]] [待确认] 进一步在 token 层面实现可微 reward 优化,可视为更高效的方案
- RIO (同期工作) 提出 reverse inference 作为自动偏好选择,不需要 pairwise 数据

**方法论价值**: 证明了 (1) codec LM 可以像 text LM 一样做 alignment, (2) 迭代自改进在语音领域有效, (3) DPO 优于 PPO 在此场景下

## 可复用的 idea

1. **Golden vs Synthetic token 对比**: 可推广到任何 two-stage TTS 系统,利用 teacher-forcing 输入与模型生成输入的差异做偏好数据
2. **Iterative self-improvement for speech**: 不断用更新后模型生成新偏好数据,循环优化
3. **偏好优化在 discrete token 空间**: 比在波形空间做 RLHF 更高效
