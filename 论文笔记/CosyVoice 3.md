---
type: paper
tier: deep
title: "CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training"
arxiv_id: "2505.17589"
source: "Sources/CosyVoice3.pdf"
authors: [Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, Jieping Ye]
year: 2025
venue: "arXiv preprint"
tags: [TTS, zero-shot, multilingual, scaling, post-training, speech-tokenizer, reinforcement-learning]
concepts: ["[[Finite Scalar Quantization]]", "[[Conditional Flow Matching]]", "[[Differentiable Reward Optimization]]", "[[Speech Tokenizer]]", "[[Gumbel-Softmax]]"]
models: ["[[CosyVoice 3]]", "[[CosyVoice 2]]", "[[MinMo]]"]
tasks: ["[[Zero-shot Speech Synthesis]]", "[[Cross-lingual Voice Cloning]]", "[[Instructed Speech Generation]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[CV3-Eval]]"]
kb_context_sources: 0
status: reviewed
created: 2026-06-01
updated: 2026-06-01
---

## KB 背景

> [!info] KB 背景 (KB 检索未启用 — P1 阶段)
> P3 阶段启用 KB 检索后,此节将自动填充知识库背景。

## 核心问题

CosyVoice 2 虽然在中英文广播场景下表现良好,但在以下方面存在明显局限:
1. **语言覆盖不足** — 仅支持中英文,无法应对 in-the-wild 多语言场景
2. **领域/风格多样性差** — 数据域单一,文本格式有限
3. **数据与模型规模未充分探索** — 未验证 TTS 领域的 scaling law
4. **缺少有效的 post-training 策略** — 无法在预训练后进一步提升内容一致性

CosyVoice 3 试图通过四个维度解决:新 speech tokenizer、DiffRO post-training、数据规模从 1 万小时扩至 100 万小时、模型参数从 0.5B 扩至 1.5B [§1]。

## 方法: 它怎么 work

### 整体架构

CosyVoice 3 沿用 CosyVoice 2 的 coarse-to-fine 两阶段架构 [§2]:
1. **LM 阶段**: 自回归 LLM 将文本 token 映射为离散 speech token(语义层)
2. **CFM 阶段**: Conditional Flow Matching 模型将 speech token 转换为 Mel spectrogram(声学层),再由 vocoder 合成波形

关键改进集中在 speech tokenizer 设计、post-training 策略、以及 scaling。

### 关键设计选择

#### 1. 基于 MinMo 的监督式多任务 Speech Tokenizer [§2.1]

**为什么有效**: CosyVoice 2 使用 SenseVoice-Large ASR 编码器 + FSQ;CosyVoice 3 改用 MinMo(一个在 140 万小时语音上预训练的多模态 LLM)作为 backbone。MinMo 本身已在对话、多语种 ASR、情感识别等任务上达到 SOTA,因此其 intermediate representations 天然携带丰富的副语言信息(情感、语种、说话人特征)。

具体做法:
- 在 MinMo 的 Voice Encoder_1(12 层 Transformer + RoPE)中间插入 FSQ 模块 [§2.1]
- FSQ 将中间表征投影到 D 维低秩空间,每维量化到 [-K, K],再投影回原始维度 [Eq.1]
- Speech token 通过 (2K+1) 进制索引计算得到 [Eq.2]
- 量化后的表征继续通过 Voice Encoder_2 和 MinMo LLM 进行多任务监督训练(ASR 365K h、LID 85K h、SER 48K h、AED 21K h、SA 11K h)[Table 3]

**为什么多任务监督比自监督更好**: 自监督 tokenizer(如 HuBERT、W2v-BERT 2.0)学到的 token 混杂语义和声学信息;监督式 tokenizer 只保留语义,声学干扰被过滤,使得下游 CFM 可以更好地从 speaker prompt 中提取音色特征,而非从 token 中读取(已被 Table 12 实验验证:在 3000 小时数据上,监督 tokenizer 在 speaker similarity 和 content consistency 上全面优于 W2v-BERT 2.0 和 SoundStream)[Table 12]。

Token rate: 25 Hz(每秒 25 个 speech token)[§2.1]。

#### 2. Differentiable Reward Optimization (DiffRO) [§2.2]

**核心问题**: 传统 RL 用于 TTS 时,需要 CFM/vocoder 把 speech token 转成音频后才能计算 reward,计算量大且生成的音频高度相似导致正负样本难以区分。

**DiffRO 的解决方案**: 直接在 token 层面优化,绕过音频生成。
1. 训练一个 Token2Text model(类似反向 ASR):输入 speech token 序列,输出文本的后验概率
2. Reward = log P_ASR(正确文本 | speech tokens) [Eq.4] — 衡量 token 序列是否清晰可被识别
3. 使用 Gumbel-Softmax 对 LLM 输出的 token logits 采样,使梯度可通过 [Eq.3]
4. 加入 KL 散度约束防止偏离参考模型 [Eq.5-6] — 但关键差异: KL 计算在 **token-level logits** 上(而非 sequence-level posterior),粒度更细

**Multi-task Reward (MTR)**: 除 ASR reward 外,还可加入 SER、AED、MOS 等下游任务的 reward [Eq.7],通过指令控制语音属性。

**为什么有效**: DiffRO 将 RL 的信号直接反馈到 LLM 的 token 选择上,无需穿过 CFM/vocoder 的计算图。相当于让 LLM "知道"每个 token 选择对最终可懂度的影响。实验显示 DiffRO 带来 20%~50% 的相对 WER 改进 [§5.4],在低资源语言和跨语言场景中尤为显著(韩语相对改进 68.7%)[§5.4]。

#### 3. Pronunciation Inpainting [§2.3]

解决多音字问题: 将中文单音字替换为拼音、英文单音词替换为 CMU 音素,混合输入使模型可控发音。RepMono + MixPhn 方案在中英文分别达到 100% 和 100% 纠正率 [Table 13]。

#### 4. Self-training for Text Normalization [§2.4]

用 LLM (Qwen-Max) 做正向/逆向 TN,构造 raw text-audio 配对,使系统可以直接合成含数字/特殊符号的原始文本。

#### 5. Instructed Speech Generation [§2.5]

instruction-following 数据从 1500 小时扩至 5000 小时,风格类型从有限扩至 100+ 种(情感、语速、音色、方言、角色扮演等)[Table 1]。支持自然语言指令和细粒度标记(如 `[laughter]`、`[breath]`、`<strong>XXX</strong>`)。

### 训练策略

训练流程分四阶段 [Fig.2b]:
1. **Large-scale Pretraining**: 用全部 100 万小时数据,从 text-based LLM 初始化 → 零样本 LM + CFM
2. **DiffRO Post-training**: 在筛选数据上用 DiffRO 优化 → 提升内容一致性
3. **Continual Pretraining**: 在情感/指令/多语言数据上持续预训练,Text2Token LM 不变 → 转移能力到 SFT 模型
4. **Speaker Fine-tune**: 在多说话人数据上微调,随机 mask speaker/style prompt 防止灾难性遗忘 [§2.6.2]

模型规模:
- LM: 0.5B → 1.5B 参数 [§4.2]
- CFM: 采用 DiT 架构,从 100M 扩至 300M 参数,去掉了 CosyVoice 2 的 text encoder 和 length regularization module [§4.2]

## 实验

| 指标 | 本文 (CosyVoice 3-1.5B_RL) | Baseline (CosyVoice 2) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER (%) test-zh | 0.71 | 1.45 | SEED-TTS-Eval | [Table 4] |
| WER (%) test-en | 1.45 | 2.57 | SEED-TTS-Eval | [Table 4] |
| CER (%) test-hard | 5.66 | 6.83 | SEED-TTS-Eval | [Table 4] |
| SS (WavLM) test-zh | 0.775 (0.836) | 0.748 (0.806) | SEED-TTS-Eval | [Table 4] |
| MOS (中文) | 4.48 | 4.25 | 主观评估 | [Fig.4] |
| MOS (英文) | 4.43 | 4.34 | 主观评估 | [Fig.4] |
| MOS (平均) | 4.45 | 4.36 | 主观评估 | [Fig.4] |
| Cross-lingual WER zh→en | 5.09 | 13.5 | CV3-Eval | [Table 7] |
| Style SIM (Expresso) | 68.25 | 60.98 | Expresso | [Table 14] |

**内容一致性**: CosyVoice 3 相对 CosyVoice 2 在 test-zh 上提升 44%,test-en 上提升 51% [§5.1]。DiffRO 在所有条件下带来 12%~35% 的额外相对提升 [§5.1]。

**多语言覆盖**: CosyVoice 3 是唯一支持全部 9 种语言的系统(zh, en, ja, ko, de, es, fr, it, ru),其他开源模型仅覆盖部分 [Table 5]。

**跨语言克隆**: CosyVoice 3-1.5B + DiffRO 在 4 个方向的 WER 均优于所有 baseline;CosyVoice 2 因日文字符转换问题在 ja 方向表现很差(48.1%),CosyVoice 3 显著改善(3.05%) [Table 7]。

**情感克隆**: DiffRO-EMO 变体在 text-related 和 text-unrelated 子集上均达到最高情感准确率(happy 0.98/0.98, sad 0.68/0.50, angry 0.84/0.68) [Table 9]。

**Speech Tokenizer 对比**: 在 170K 小时数据上,CosyVoice 3.0 tokenizer 相比 CosyVoice 2.0 tokenizer 在 test-en CER 上从 2.57 降至 2.46,SS 从 0.736 升至 0.747 [Table 12]。

## 局限性

1. **音色不可控** — 无法通过文本指令编辑音色(timbre),对角色扮演类应用有限 [§7]
2. **歌唱生成能力弱** — 当前 tokenizer 和 LM 未纳入歌唱数据 [§7]
3. **大模型未充分发挥** — 1.5B 模型在 test-hard 上反而略逊于 0.5B(5.66 vs 5.09),因可用于 post-training 的高质量数据不足 [§5.1]
4. **Speaker similarity 的 "hacking" 问题** — DiffRO 优化 WER 时可能略降 speaker similarity,需引入 speaker 相似度 reward 来平衡 [§5.4]
5. **hard samples 上 DiffRO 效果有限** — 稀有词、绕口令、重复词等对 reward model 仍构成挑战 [§5.4]

## 点评

**优势**:
- DiffRO 是一个优雅的工程创新: 绕开 CFM/vocoder 在 token 层直接优化,大幅降低 RL 在 TTS 中的计算成本,且适用于所有基于离散 token 的 TTS 系统
- 监督式多任务 tokenizer 的设计思路清晰: 让 FSQ 插入一个已经"见过"丰富副语言信息的 backbone,比从头训练语义更丰富
- 数据工程扎实: multilingual pipeline 的 6 步处理(VAD → 降噪 → ASR → 标点 → 音量 → 过滤)是工业级实践的很好参考
- CV3-Eval benchmark 填补了多语言 + in-the-wild TTS 评估的空白

**不足**:
- 1.5B 模型表现不稳定(部分指标逊于 0.5B),说明 scaling 尚未找到最优配方
- 缺少与 VALL-E 2、Voicebox 等强 baseline 的直接对比(可能因开源模型不可得)
- DiffRO 的 multi-task reward 部分(MTR)实验覆盖不够充分,仅展示了 EMO 一个变体

## 可复用的 idea

1. **Token-level RL**: 在离散 token 空间用 Gumbel-Softmax + token-level KL 做可微 reward 优化,适用于任何 LM → discrete token → downstream renderer 的 pipeline
2. **监督式 tokenizer 设计模式**: 在大型预训练 speech model 的中间层插入量化模块,通过多任务监督压入目标信息(语义),同时排除不需要的信息(声学细节)
3. **Pronunciation Inpainting**: 用 mixed text+phoneme 输入解决多音字,比纯 G2P 或纯 BPE 更灵活且可控
4. **跨语言能力迁移**: 通过 continual pretraining + 辅助多语种数据,将单语说话人变为多语说话人(polyglot training)
5. **Multilingual data pipeline**: 6 步处理流程(特别是 cross-validation ASR + MFA 标点对齐)可作为标准参考
