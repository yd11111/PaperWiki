---
type: paper
tier: deep
title: "Dict-TTS: Learning to Pronounce with Prior Dictionary Knowledge for Text-to-Speech"
arxiv_id: "2206.02147"
source: "Sources/Dict-TTS.pdf"
authors: [Ziyue Jiang, Zhe Su, Zhou Zhao, Qian Yang, Yi Ren, Jinglin Liu, Zhenhui Ye]
year: 2022
venue: "NeurIPS 2022"
tags: [TTS, polyphone-disambiguation, G2P, dictionary-knowledge, end-to-end, unsupervised, attention, Gumbel-Softmax, prosody]
concepts: ["[[PhonemeRepresentation]]", "[[Gumbel-Softmax]]", "[[ProsodyModeling]]", "[[Text-to-SpeechPipeline]]", "[[MelSpectrogram]]", "[[VariationalAutoencoderforTTS]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页: [[ProsodyModeling]]✓, [[PhonemeRepresentation]], [[Gumbel-Softmax]], [[Text-to-SpeechPipeline]], [[MelSpectrogram]], [[VariationalAutoencoderforTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]], [[PhonemeRepresentation]], [[Gumbel-Softmax]], [[Text-to-SpeechPipeline]], [[MelSpectrogram]], [[VariationalAutoencoderforTTS]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: Dict-TTS 处在 TTS 前端(G2P)与声学建模的交叉地带。[[PhonemeRepresentation]] 概念页 [待确认] 描述了 TTS 前端从规则 G2P → 神经 G2P → Character 直接输入 → BPE tokens 的演进脉络。Dict-TTS 是 Character 直接输入路线的一个变体:不依赖外部 G2P 工具,而是通过字典查询实现端到端的发音消歧。这条路线在概念页中记载较少,Dict-TTS 提供了一个重要的中间方案 — 既不完全依赖外部 G2P 模块(如 pypinyin/G2PM),也不完全让模型自己学(如 Tacotron 的 character 输入),而是引入字典作为结构化先验知识。

**已有认知**: [[Gumbel-Softmax]] [待确认] 记载了该技巧在 VQ 离散化(wav2vec 2.0)和 token-level RL(CosyVoice 3 DiffRO)中的应用。Dict-TTS 提供了一个更早期(2022)的 TTS 应用场景:用 Gumbel-Softmax 实现字典中多个发音候选的可微选择,这在时间线上早于 CosyVoice 3 的应用。[[ProsodyModeling]]✓ 已确认概念页指出"Text Pre-training"是韵律建模的一种方法(通过 BERT/GPT 预训练获取隐含韵律的文本表示)。Dict-TTS 提供了另一种途径:通过字典语义匹配获取语义信息,作为韵律建模的辅助输入。

**创新判断**: Dict-TTS 的核心创新在于将"查字典"这一人类行为形式化为可微的注意力机制(S2PA),实现了无标注的端到端多音字消歧。这在 KB 中没有直接对应的概念。

## 速查

> [!summary] 速查
> - **一句话**: 将在线字典作为结构化先验知识,通过语义-发音注意力(S2PA)实现无需音素标注的端到端多音字消歧
> - **路线**: Character Sequence → Semantic Encoder → S2PA (查字典匹配语义→聚合发音权重→Gumbel-Softmax 采样) → Linguistic Encoder (融合语义+发音 embedding) → VAE Generator → Mel → HiFi-GAN → Waveform
> - **指标**: Biaobei PER-O 2.12% (接近 pypinyin 2.78%), 预训练后 PER-O 1.54%; MOS-P 4.03 (高于 phoneme-based 3.89); 三语言验证 [Table 1, 2, 3]
> - **可借鉴**: (1) 将外部结构化知识(字典)通过注意力机制注入端到端模型的范式; (2) Gumbel-Softmax 在离散发音选择中的应用; (3) 利用 ASR 数据无监督预训练 G2P 模块
> - **局限**: 字典不含句法信息(影响韵律); 日语效果不如开源 G2P (kanji 读音需经验规则); 仅在 mel-spectrogram TTS 上验证,未扩展到 LLM-TTS

## 核心问题

Dict-TTS 要解决的核心问题是:**如何在不依赖音素标注的前提下,让 character-based TTS 系统准确处理多音字(polyphone)?**

现有方案的三个痛点 [§1]:
1. **规则方法**覆盖面有限,难以扩展到 out-of-domain 场景
2. **神经 G2P 模型**需要大量标注数据,构建成本高
3. **端到端 character-based TTS**(如 Tacotron)隐式学习发音,但会将字符表示拉向声学空间(同音不同义的字符被混淆),导致语义理解和多音字消歧能力退化

论文的出发点很直觉:人在不确定发音时会查字典。Dict-TTS 将这一行为形式化 — 用注意力机制"查"在线字典,匹配语义模式获取正确发音 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Dict-TTS 基于 PortaSpeech 构建,包含四个主要组件 [§3.1, Fig 2]:

1. **Semantic Encoder**: 4 层 Feed-Forward Transformer + relative position encoding,从输入字符序列提取语义表示 z [§A.2]
2. **S2PA Module (Semantics-to-Pronunciation Attention)**: 核心创新模块,匹配输入语义与字典条目,输出发音 embedding 和语义 embedding [§3.3]
3. **Linguistic Encoder**: 4 层 Feed-Forward Transformer,融合 S2PA 输出的语义和发音信息 [§A.2]
4. **Variational Generator + Multi-Length Discriminator**: 沿用 PortaSpeech 的 VAE-based generator,用 HiFiSinger 的 Multi-Length Discriminator 替换 flow-based post-net [§3.1]

**总参数量**: 17.125M (其中 S2PA 仅 0.404M) [Table 5]

### 关键设计选择

#### 1. 字符空间 vs 声学空间的分析 [§3.2]

[论文原文] 论文通过对比 phoneme-based 和 character-based TTS 的表示空间,给出了 S2PA 设计的理论动机:
- **Phoneme-based TTS**: 音素序列中同音不同义词(如 "Whether" / "Weather")被映射到相同表示,语义歧义影响韵律建模
- **Character-based TTS**: mel 重建 loss 会将字符表示拉向声学空间,同音字(如 "火"/"伙")的表示按发音聚类而非按语义聚类,这阻碍了多音字消歧

[论文原文] 结论:字符表示应保持在语义空间,由 S2PA 显式处理语义→发音的映射,避免端到端训练将表示拉向声学空间 [§3.2]。

[agent 解读] 这个分析指出了 character-based TTS 的一个根本矛盾:模型需要语义理解来消歧多音字,但 mel loss 的梯度天然将表示推向声学空间。S2PA 通过引入字典这个外部锚点,为语义空间提供了独立的监督信号。

#### 2. S2PA: 语义-发音注意力 [§3.3]

**字典构建**: 从在线字典网站爬取,每个字符 c_i 有 m 个可能发音 p_{i,j},每个发音对应一个字典条目 e_{i,j}(包含释义、用法、翻译) [§3.3]。用预训练的 XLM-R 离线提取每个条目的语义表示 k,存储到磁盘 [§A.1]。

**语义匹配**: 对输入字符 t_i,用 Semantic Encoder 输出的语义向量 z_i 作为 query,与字典条目的语义表示 k 做 scaled dot-product attention [Eq. 1]:

```
[a_{i,1,1}, ..., a_{i,m,u}] = [k_{i,1,1}, ..., k_{i,m,u}] · z_i^T / √d
```

**语义 embedding 检索**: s'_i = softmax(a) · k,即加权聚合匹配到的字典语义信息 [§3.3]。

**发音消歧**: 将每个发音对应的所有条目注意力权重求和,得到发音级别的权重 w_{i,j},然后用 Gumbel-Softmax 采样最可能的发音 [Eq. 2, 3]:

```
w̃_{i,j} = exp((log(w_{i,j}) + g_{i,j}) / τ) / Σ exp((log(w_{i,l}) + g_{i,l}) / τ)
p'_i = Σ w̃_{i,j} · p_{i,j}
```

[agent 解读] S2PA 的设计巧妙在于两层聚合:(1) 条目级 softmax 提取语义信息(连续的,信息丰富);(2) 发音级 Gumbel-Softmax 选择发音(近似离散的,明确的)。这比 NLR [16] 将所有发音条目拼接后联合 attend 更明确 [Appendix E]。

#### 3. 为什么用 Gumbel-Softmax 而非 softmax [§4.4]

[论文原文] 消融实验显示,直接用 softmax 加权(而非 Gumbel-Softmax 采样)会导致 PER-S 从 1.08% 升至 1.19%、SER-S 从 6.50% 升至 7.75% [Table 4]。原因是当两个发音的权重接近(如 0.6 和 0.4)时,加权混合会产生模糊的发音 embedding,而 Gumbel-Softmax 通过采样强制选择一个 [§4.4]。

[agent 解读] 这与 [[Gumbel-Softmax]] 概念页描述的"承诺程度"直接对应:多音字在特定上下文中只有一个正确发音,需要 hard selection 而非 soft blending。

#### 4. 与 NLR 的关键差异 [Appendix E]

Dict-TTS 与同期工作 NLR [16] 都利用字典知识,但有五点本质区别:
- Dict-TTS 对每个发音条目分别做注意力并显式消歧;NLR 拼接所有条目联合 attend
- Dict-TTS 分析并维持了语义空间与声学空间的分离
- Dict-TTS 支持 ASR 数据预训练
- NLR 更关注低资源场景
- Dict-TTS 基于 PortaSpeech (NAR),NLR 基于 AR Transformer TTS

### 训练策略

**端到端训练**: S2PA 模块通过 TTS decoder 的 mel 重建 loss 反向传播梯度进行训练,不需要任何音素标注 [§3.4]。温度参数 τ 按 Gumbel-Softmax 原论文 [23] 的策略退火 [§4.1]。

**ASR 数据预训练**: 由于 S2PA 不依赖音素标注,可在大规模 ASR 数据集(WenetSpeech, 10000+ 小时 [61])上预训练,提升语义理解和泛化能力 [§3.4]。预训练时引入 resemblyzer 提取的 speaker embedding 以适应多说话人场景 [§A.3]。预训练 600k 步至收敛。

**兼容人工规则**: 对于无法从字典学到的发音规则(如普通话变调规则"一"在四声前读二声),可直接将发音权重 w_{i,j} 强制设为 ground truth 值 [§G, Appendix G]。

## 实验

| 指标 | 本文 (Dict-TTS) | Baseline (Phoneme/pypinyin) | Baseline (Character) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| PER-O | 2.12% | 2.78% | — | Biaobei | [Table 1] |
| PER-S | 1.08% | 1.14% | 3.73% | Biaobei | [Table 1] |
| SER-S | 6.50% | 7.00% | 30.50% | Biaobei | [Table 1] |
| PER-O (预训练) | 1.54% | 2.78% | — | Biaobei | [Table 2] |
| PER-S (预训练) | 0.79% | 1.14% | 3.73% | Biaobei | [Table 2] |
| MOS-P | 4.03±0.05 | 3.89±0.08 (pypinyin) | 3.82±0.08 | Biaobei | [Table 3] |
| MOS-Q | 3.91±0.04 | 3.95±0.06 (pypinyin) | 3.88±0.07 | Biaobei | [Table 3] |
| DTW (pitch) | 52.4 | 52.6 (pypinyin) | 53.1 | Biaobei | [Table 3] |
| Duration Error | 34.4 ms | 35.3 ms (pypinyin) | 36.2 ms | Biaobei | [Table 6] |
| Avg Pitch Error | 1232.3 | 1308.8 (pypinyin) | 1424.6 | Biaobei | [Table 6] |

**跨语言验证** [Table 1]:
- **日语 (JSUT)**: PER-O 3.73% (字符 1.55%), PER-S 2.57% (phoneme/pyopenjtalk 0.92%) — Dict-TTS 不如开源 G2P,因为日语 kanji 的音读/训读选择需要经验规则而非纯语义
- **粤语 (Common Voice HK)**: PER-S 1.23% (phoneme/pycantonese 1.45%) — Dict-TTS 超越开源 G2P

**消融实验** [Table 4]:
- 去掉字典语义 embedding: CMOS-P 下降 0.280,确认语义信息对韵律的正面影响
- 去掉 Gumbel-Softmax (改用 weighted sum): PER-S 从 1.08% 升至 1.19%,SER-S 从 6.50% 升至 7.75%

## 局限性

1. **缺乏句法信息**: 字典知识不包含输入文本的树状句法结构,而句法信息对韵律建模(特别是短语边界和重音模式)很重要 [§5]
2. **日语效果有限**: kanji 的音读/训读选择很大程度上是经验性的(不完全由语义决定),Dict-TTS 在 JSUT 上不如开源 G2P [Table 1]
3. **字典质量依赖**: 论文直接爬取在线字典未做优化,"a well-designed dictionary" 可以进一步提升效果 [§5]
4. **仅验证 mel-spectrogram TTS**: 基于 PortaSpeech (2021) 架构,未在更现代的 LLM-TTS 或 codec-based 架构上验证
5. **评测语言有限**: 仅中文、日语、粤语三种语言,未覆盖英语等拼音文字(虽然论文声称可扩展 [Appendix H])

## 点评

Dict-TTS 的核心贡献是将"查字典"这一朴素直觉形式化为可微的端到端框架。论文的理论分析(§3.2 关于语义空间 vs 声学空间)为 S2PA 的设计提供了合理的动机,而非纯粹的工程堆叠。

**亮点**:
- 无监督多音字消歧达到接近(甚至超越)有监督 G2P 系统的水平,这在实际部署中有巨大价值(不需要为每种新语言标注音素数据)
- 语义信息的副产品效应 — 字典语义不仅帮助消歧,还改善了韵律(MOS-P 4.03 高于 phoneme-based 3.89),这是论文的一个意外发现
- S2PA 模块仅 0.404M 参数,计算开销极小

**不足**:
- 日语实验暴露了方法的根本限制:当发音选择不完全由语义决定时(如 kanji 的音读/训读),纯语义匹配路线不够用
- 论文发表于 2022 年,彼时 LLM-TTS 范式尚未兴起。在 token-based TTS(VALL-E, CosyVoice 等)中,G2P 问题的形态已经改变(BPE tokenizer + 大规模数据可能隐式学到发音规则)

**历史定位**: Dict-TTS 代表了 character-based TTS 路线在 mel-spectrogram 时代的一次有意义的探索。它提供了一种"轻量级外部知识注入"的范式,但随着 LLM-TTS 时代的到来(模型本身具备强大的语义理解能力),其核心价值 — 将外部语义知识注入 TTS 前端 — 的必要性有所降低。不过,"用结构化知识显式辅助离散选择"的思路(S2PA + Gumbel-Softmax)在其他场景下仍有参考价值。

## 可复用的 idea

1. **外部知识注入的注意力范式**: 将结构化知识库(字典条目)用预训练模型(XLM-R)离线编码,推理时通过注意力检索。这个模式可迁移到任何需要查询外部知识的端到端系统。

2. **两层聚合策略**: 条目级 softmax(连续语义检索) + 发音级 Gumbel-Softmax(离散选择)。这种"先软后硬"的两层聚合适用于任何需要从多个候选中做出离散决策的场景。

3. **ASR 数据反哺 G2P**: 由于 S2PA 不依赖音素标注,可直接在 ASR 数据上预训练。这启示:如果 G2P 模块能以无监督方式训练,就能利用大量弱监督语音数据提升前端性能。

4. **语义空间 vs 声学空间的分析框架**: §3.2 的分析方法(可视化字符表示在两个空间中的分布)是诊断 character-based TTS 问题的有用工具。

---

检索命中: [[ProsodyModeling]]✓, [[PhonemeRepresentation]][待确认], [[Gumbel-Softmax]][待确认], [[Text-to-SpeechPipeline]][待确认], [[MelSpectrogram]][待确认], [[VariationalAutoencoderforTTS]][待确认] | 过滤: 无 | 未命中但可能相关: 无
