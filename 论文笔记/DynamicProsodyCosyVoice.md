---
type: paper
tier: deep
title: "Dynamic Prosody Prediction in LLM-based TTS for Improving Speaker Similarity"
arxiv_id: "2606.15267"
source: "Sources/DynamicProsodyCosyVoice.pdf"
authors: [Zhenwei Mou, Liping Chen, Yajun Hu, Zhen-Hua Ling, Xin Fang, Jianqing Gao]
year: 2026
venue: "arXiv preprint"
tags: [TTS, prosody, LLM-based, speaker-similarity, CosyVoice, chain-of-thought, zero-shot, style-transfer]
concepts: ["[[ProsodyModeling]]", "[[LLM-basedTTS]]", "[[SpeechFactorization]]", "[[SpeakerEmbedding]]"]
models: ["[[模型库/CosyVoice]]", "[[论文笔记/RALL-E|RALL-E]]", "[[论文笔记/Vevo2|Vevo 1.5]]", "[[论文笔记/F5-TTS|F5-TTS]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[数据集/Emilia]]", "ESD", "AISHELL-3", "WenetSpeech"]
kb_context_sources: 6
status: draft
created: 2026-06-17
updated: 2026-06-17
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[ProsodyModeling]], [[LLM-basedTTS]], [[模型库/CosyVoice]], [[SpeakerEmbedding]], [[ConditionalFlowMatching]], [[SpeechFactorization]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]]✓, [[LLM-basedTTS]]✓, [[模型库/CosyVoice]]✓, [[SpeakerEmbedding]]✓, [[ConditionalFlowMatching]]✓, [[SpeechFactorization]]✓ | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: 本文处于 LLM-based TTS 的韵律建模前沿。在 KB 中,CosyVoice 系列属于 Hybrid 架构(LLM 生成 semantic tokens + OT-CFM 合成语音),其核心设计是通过 x-vector 显式分离说话人建模,但**缺乏对韵律/说话风格的显式建模** --- 韵律信息被隐式编码在 speech tokens 中。这正是本文要解决的问题。

**已有认知**: ProsodyModeling 页记录了韵律建模从显式(FastSpeech 2 variance adaptor)到隐式(in-context learning)的演进。RALL-E 和 Vevo1.5 采用 CoT prompting 在 LLM-TTS 中显式预测韵律 token,但它们是**静态预计算**(先预测全部韵律再生成语音)。SpeechFactorization 页指出"fine-grained prosody disentanglement 是开放前沿"。

**创新判断**: 本文的"动态韵律预测"(根据已生成的语音 token 逐音节预测韵律)是对 CoT 静态韵律预测的直接改进,填补了 CosyVoice 架构在显式韵律建模上的空白。这在 KB 中属于 ProsodyModeling 演进线上的新节点:从静态 CoT 预测推进到**条件于已生成语音的动态预测**。

## 速查

> [!summary] 速查
> - **一句话**: 在 CosyVoice LLM 阶段引入逐音节动态韵律 token 预测(条件于已生成语音),通过"先预测韵律再生成语音 token"的交替策略解耦韵律和音色,提升 speaker similarity
> - **路线**: Text(syllable-level) + Speaker Emb + Reference Prosody/Speech Tokens → LLM 交替生成 [PQ→Prosody Token→Speech Tokens→EOSL] → Flow Matching → Waveform
> - **指标**: ESD preference 51.5% vs CosyVoice(50k) 28.8%; SIM 0.884 vs 0.875; Emotion ACC 86.56% vs 84.32%; 50k 训练 vs 开源 CosyVoice 170k 仍优(ESD preference 44.8% vs 32.7%)
> - **可借鉴**: 动态 prosody token 交替生成策略可迁移到其他 LLM-TTS 架构;k-means prosody quantization 简单有效;用小数据训练但通过显式韵律建模弥补数据量差距的思路
> - **局限**: 仅验证中文(音节=字符);仅验证 CosyVoice 框架;AISHELL-3 中性韵律场景优势不明显;未与 Seed-TTS/MaskGCT 等更强 baseline 对比;推理速度未报告

## 核心问题

当前 LLM-based TTS(如 CosyVoice)将说话风格隐式编码在 speech tokens 中,缺乏对韵律/说话风格的显式建模。已有的 CoT prompting 方法(RALL-E、Vevo1.5)虽然显式预测韵律,但采用**静态预计算**策略 --- 在生成任何语音 token 之前就预测整段话的全部韵律,无法利用已生成语音中的风格信息。这导致韵律学习不充分,限制了合成语音的 speaker similarity。

**核心假设**: 韵律预测应该是一个**动态过程** --- 当前音节的韵律应依赖于已生成的前序语音(而非仅依赖文本和参考语音),因为语音流中的风格信息是逐步展开的。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

基于 CosyVoice 框架 [§3.1],在 LLM 阶段引入显式韵律 token 预测。整体流程:

```
Speaker Embedding (CAM++) + Text Embedding (syllable-level)
  ↓
LLM (decoder-only Transformer, 14 layers)
  逐音节交替生成:
  [PQ] → Prosody Token(yi) → Speech Tokens(Si) → [EOSL]
  ↓
Flow Matching → Waveform
```

关键区别于 CosyVoice 原版 [Fig 1a vs Fig 2]:
- **CosyVoice 原版**: LLM 直接从 text + speaker embedding 生成 speech tokens,无显式韵律建模
- **CoT prompting (RALL-E/Vevo1.5)** [Fig 1b]: 先一次性预测所有 prosody tokens,再生成 speech tokens(两阶段串行)
- **本文 (Dynamic)** [Fig 2]: 在每个音节位置,先预测该音节的 prosody token,再生成该音节的 speech tokens,交替进行

### 关键设计选择

**1. Prosody Token 定义 [§3.2]**

对每个音节提取 4 维韵律特征向量 $g_i = [d_i, e_i, h_i, r_i]$:
- $d_i$: duration(音节时长)
- $e_i$: mean energy(平均能量)
- $h_i$: mean pitch(平均基频)
- $r_i$: pitch range(基频范围 = max - min)

通过 **k-means 聚类**(K=512)将连续韵律向量量化为离散 prosody token [Eq 3]。聚类在 WenetSpeech 数据集上训练。

[agent 解读] 选择 k-means 而非 VQ-VAE 等端到端方法可能是为了简单性和可解释性 --- 4 维特征空间上 512 个聚类中心足够捕捉常见韵律模式,且训练成本极低。但这意味着 prosody token 的表达能力受限于 512 个离散类别和 4 个手工特征维度。

**2. 动态预测机制 [§3.1, Eq 1-2]**

韵律预测(Eq 1):
$$y_i = p(C^p | v, X, q_{1:i-1}, S_{1:i-1})$$

语音预测(Eq 2):
$$z_{i,t} = p(C^s | v, X, q_{1:i-1}, S_{1:i-1}, q_i, s_{i,1:t-1})$$

关键: 第 $i$ 个音节的韵律 $y_i$ 条件于**前序所有音节的韵律和语音 tokens** ($q_{1:i-1}, S_{1:i-1}$),而非仅条件于文本和参考语音。[论文原文] 论文称这使得韵律预测能利用"style specific to target speech" [§1]。

**3. Prosody Query Embedding (PQ) [§3.1]**

使用特殊的 PQ embedding 作为 LLM 输入来触发韵律 token 预测。在每个音节的 speech tokens 之前插入 PQ,LLM 在 PQ 位置输出韵律 token 的概率分布。

[agent 解读] PQ 的作用类似于 CoT 中的"思考提示符" --- 它告诉 LLM "在这里先预测韵律",从而在统一的自回归框架内实现韵律和语音的交替生成,无需修改 Transformer 架构。

**4. EOSL Token [§3.1]**

引入 end-of-syllable (EOSL) token 标记每个音节的 speech tokens 结束(最后一个音节用 EOS)。这使得 LLM 可以自回归地确定每个音节的帧数(duration),而非依赖外部 duration predictor。

[agent 解读] EOSL 的引入使 duration 隐式地由 LLM 和 prosody token 共同决定 --- prosody token 编码了目标 duration 信息(k-means 特征包含 duration),而 EOSL 在解码时实际控制帧数。这是 duration 预测的双重保障。

### 训练策略

**Loss Function [§3.3, Eq 4]**:

$$L = -\alpha \frac{1}{I}\sum_{i=1}^{I} \hat{y}_i \log y_i - (1-\alpha) \frac{1}{\sum(T_i+1)} \sum_{i=1}^{I}\sum_{t=1}^{T_i+1} \hat{z}_{i,t} \log z_{i,t}$$

- 韵律 CE loss + 语音 CE loss 的加权和
- $\alpha = 0.5$(等权)
- 韵律 loss 按音节数归一化,语音 loss 按总帧数归一化 [论文原文]

**训练细节 [§4.2]**:
- LLM: 14 layers, 16 heads, 1024 dim, 4096 FFN dim (decoder-only Transformer)
- 训练数据: ~50k hours (WenetSpeech 10k + Emilia 中文 50k,去除 MFA 失败和语言误分类后约 50k)
- 音节边界: MFA (Montreal Forced Aligner) [§4.1]
- 训练步数: 800k steps, 8x MLU 580 GPUs
- 学习率: $10^{-4}$, warmup 10k steps
- 推理: top-p=0.8, speech top-k=25, prosody top-k=15

## 实验

### 数据集 [§4.1]

| 数据集 | 规模 | 特点 | 用途 |
| --- | --- | --- | --- |
| WenetSpeech | 10k hrs | 多领域中文 | 训练 + k-means |
| Emilia (中文) | 50k hrs | 大规模多样 | 训练 |
| ESD | 350 utt, 10 speakers, 5 emotions | 情感韵律丰富 | 评估 |
| Internal (iFlytek) | 230 utt | 多样说话风格 | 评估 |
| AISHELL-3 | test set, 214 speakers | 韵律中性 | 评估 |

### 主观评估

**MOS (自然度)** [Table 1]:

| 指标 | 本文 | CosyVoice(50k) | CoT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS | 4.07 +/- 0.06 | 4.01 +/- 0.07 | 4.00 +/- 0.09 | ESD | [Table 1] |
| MOS | 3.99 +/- 0.07 | 3.97 +/- 0.07 | 3.98 +/- 0.08 | Internal | [Table 1] |
| MOS | 4.06 +/- 0.07 | 4.06 +/- 0.06 | 4.03 +/- 0.10 | AISHELL-3 | [Table 1] |

[agent 解读] MOS 差异较小且置信区间重叠,说明动态韵律预测**不损害自然度**,但自然度提升本身不显著。

**Preference Test (Speaker Similarity)** [Table 2, p<0.01]:

| 指标 | Prefer 本文(%) | No Pref(%) | Prefer Baseline(%) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Preference | 51.5 | 19.7 | 28.8 | CosyVoice(50k) | ESD | [Table 2] |
| Preference | 50.9 | 21.4 | 28.8 | CoT | ESD | [Table 2] |
| Preference | 48.2 | 18.6 | 33.2 | CosyVoice(50k) | Internal | [Table 2] |
| Preference | 47.7 | 21.4 | 30.9 | CoT | Internal | [Table 2] |
| Preference | 33.6 | 45.9 | 20.4 | CosyVoice(50k) | AISHELL-3 | [Table 2] |
| Preference | 33.6 | 40.5 | 25.9 | CoT | AISHELL-3 | [Table 2] |

关键发现: 在情感/风格丰富的 ESD 和 Internal 数据集上优势明显(~20pp 领先);在韵律中性的 AISHELL-3 上优势较小(~8-13pp 领先,且 No Preference 占比很高)。[论文原文] 这符合预期 --- 韵律显式建模在风格多样场景收益更大 [§4.3]。

### 客观评估 [Table 3]

| 指标 | 本文 | CosyVoice(50k) | CoT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| CER(%) | 5.66 | 6.38 | 6.14 | ESD | [Table 3] |
| SIM | 0.884 | 0.875 | 0.876 | ESD | [Table 3] |
| Emotion ACC(%) | 86.56 | 84.32 | 84.52 | ESD | [Table 3] |
| Pitch Corr(%) | 80.32 | 79.52 | 79.31 | ESD | [Table 3] |
| Pitch RMSE | 80.81 | 83.61 | 82.82 | ESD | [Table 3] |
| Energy Corr(%) | 94.91 | 94.08 | 94.03 | ESD | [Table 3] |
| Energy RMSE | 5.93 | 6.42 | 6.39 | ESD | [Table 3] |
| CER(%) | 10.44 | 13.69 | 13.60 | Internal | [Table 3] |
| SIM | 0.821 | 0.802 | 0.799 | Internal | [Table 3] |
| CER(%) | 10.19 | 11.59 | 11.61 | AISHELL-3 | [Table 3] |
| Pitch Corr(%) | 92.66 | 90.59 | 90.51 | AISHELL-3 | [Table 3] |
| Pitch RMSE | 5.91 | 6.66 | 6.52 | AISHELL-3 | [Table 3] |

**全面优于 CosyVoice(50k) 和 CoT**:
- CER 在所有数据集上显著降低(ESD: 5.66 vs 6.38; Internal: 10.44 vs 13.69),说明韵律显式建模同时改善了语音内容清晰度
- SIM(emotion2vec+ cosine similarity)和 Emotion ACC 在 ESD 上均最优
- Pitch/Energy 的 Corr 更高、RMSE 更低,直接证明韵律建模能力提升

### 与开源模型对比 [Table 4, §4.5]

50k 训练的本文方法 vs 100k-170k 训练的开源模型:

| 指标 | Prefer 本文(%) | No Pref(%) | Prefer Baseline(%) | Baseline (训练量) | 数据集 | p-value | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Preference | 52.8 | 19.4 | 27.8 | Vevo1.5 (100k) | ESD | <0.01 | [Table 4] |
| Preference | 50.9 | 26.4 | 22.7 | F5-TTS (100k) | ESD | <0.01 | [Table 4] |
| Preference | 44.8 | 22.4 | 32.7 | CosyVoice (170k) | ESD | <0.01 | [Table 4] |
| Preference | 47.0 | 16.6 | 36.4 | Vevo1.5 (100k) | Internal | <0.01 | [Table 4] |
| Preference | 60.4 | 15.5 | 24.1 | F5-TTS (100k) | Internal | <0.01 | [Table 4] |
| Preference | 38.2 | 26.4 | 35.4 | CosyVoice (170k) | Internal | 0.10 | [Table 4] |
| Preference | 35.6 | 41.2 | 23.2 | Vevo1.5 (100k) | AISHELL-3 | <0.01 | [Table 4] |
| Preference | 28.6 | 50.0 | 21.4 | F5-TTS (100k) | AISHELL-3 | <0.01 | [Table 4] |
| Preference | 27.7 | 43.2 | 29.1 | CosyVoice (170k) | AISHELL-3 | 0.65 | [Table 4] |

关键发现:
- 在情感丰富的 ESD 上,50k 训练的本文方法甚至优于 170k 训练的 CosyVoice 开源版 (44.8% vs 32.7%, p<0.01)
- 在 AISHELL-3 中性场景中,与 CosyVoice 开源版差异不显著 (p=0.65),与 Internal 上和 CosyVoice 的对比也不显著 (p=0.10)
- [论文原文] 论文称这表明动态韵律预测可以**弥补小规模训练数据的韵律学习不足** [§4.5]

## 局限性

1. **仅验证中文**: 中文中字符=音节,音节边界天然清晰;推广到英语等需要额外的音节分割处理 [⚠️ 论文未详述]
2. **仅基于 CosyVoice 框架**: 未在其他 LLM-TTS(如 VALL-E、MaskGCT)上验证通用性
3. **韵律特征手工设计**: 4 维特征(duration, mean energy, mean pitch, pitch range)可能丢失细粒度韵律信息(如 pitch contour shape、微停顿模式);k-means 量化进一步压缩信息
4. **AISHELL-3 场景优势不明显**: 在韵律中性场景中,动态预测的收益有限,说明方法对风格多样性有依赖
5. **缺少推理效率分析**: 动态交替预测增加了序列长度(每个音节额外预测一个 prosody token),但论文未报告 RTF 或延迟影响
6. **未与最强 baseline 对比**: 缺少 Seed-TTS、MaskGCT、CosyVoice 2/3 等更新更强系统的对比
7. **MFA 依赖**: 训练和评估依赖 MFA 的音节边界对齐;MFA 失败的数据被丢弃,可能引入选择偏差

## 点评

**优势**:
- 动态韵律预测的 idea 直觉清晰且实现简洁 --- 在统一自回归框架内通过 PQ embedding 和 EOSL token 实现韵律-语音交替生成,无需修改 Transformer 架构
- 实验设计合理: 三个数据集覆盖情感丰富/多样风格/中性韵律三种场景,客观+主观评估全面
- 与开源模型的对比(50k vs 170k)有力地论证了"显式韵律建模弥补数据量差距"的假设
- CER 同步改善是意外收获 --- 韵律显式建模可能帮助 LLM 更好地对齐文本和语音

**不足**:
- 创新增量较小 --- 本质是将 CoT 的"先全部预测韵律,再全部生成语音"改为"逐音节交替",idea 自然但技术深度有限
- 韵律 token 的设计(4 维特征 + k-means)偏传统,与 RALL-E 的 duration/pitch token 设计类似,未探索更现代的表示方式(如 VQ-VAE、continuous prosody latent)
- 消融不充分: 未消融 $\alpha$ 权重的影响、k-means 聚类数量、韵律特征维度的选择

**与 KB 中相关工作的定位**: 在 ProsodyModeling 的演进线上,本文位于"CoT 静态预测 → 动态预测"的过渡节点。方向正确(条件于已生成语音做动态预测),但相比同期的 ProsodyEval(DS-WED 韵律多样性度量)和 NoVerifiableRewardForProsody(RL 优化韵律)等工作,本文的方法论深度稍浅。

## 可复用的 idea

1. **PQ (Prosody Query) Embedding 触发机制**: 用特殊 embedding 在自回归序列中触发不同类型 token 的预测,可推广到其他多类型 token 交替生成场景(如情感 token + 语音 token)
2. **动态条件韵律预测**: "先预测韵律,用韵律 condition 后续语音生成,再用生成的语音 condition 下一个韵律预测"的循环依赖策略
3. **小数据 + 显式韵律建模弥补数据量**: 在资源受限场景下,通过显式建模 variation information 提升模型利用数据的效率
4. **EOSL Token 的音节级分段**: 在自回归 TTS 中引入音节级分段标记,使 duration 由模型自适应决定而非外部 predictor

---

*检索命中: [[ProsodyModeling]], [[LLM-basedTTS]], [[模型库/CosyVoice]], [[SpeakerEmbedding]], [[ConditionalFlowMatching]], [[SpeechFactorization]] | 过滤: 无 | 未命中但可能相关: 无*
