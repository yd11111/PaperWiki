---
type: paper
tier: deep
title: "Vec-Tok Speech: Speech Vectorization and Tokenization for Neural Speech Generation"
arxiv_id: "2310.07246"
source: "Sources/Vec-Tok-Speech.pdf"
authors: [Xinfa Zhu, Yuanjun Lv, Yi Lei, Tao Li, Wendi He, Hongbin Zhou, Heng Lu, Lei Xie]
year: 2023
venue: "arXiv preprint"
tags: [speech-codec, LLM-TTS, zero-shot-TTS, voice-conversion, S2ST, semantic-token, speech-vector, BPE, style-transfer, disentanglement]
concepts: ["[[Semantic vs Acoustic Tokens]]", "[[Speech Factorization]]", "[[Speech Language Model]]", "[[Residual Vector Quantization]]", "[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Neural Vocoder]]"]
models: ["[[模型库/WavLM|WavLM]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Semantic vs Acoustic Tokens]], [[Speech Factorization]], [[Speech Language Model]], [[Residual Vector Quantization]], [[LLM-based TTS]], [[Speech Tokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Vec-Tok Speech 是 LLM-based TTS 谱系中的一个早期探索(2023.10),时间上紧随 VALL-E (2023.01) 和 SPEAR-TTS (2023.02)。与同期工作的核心区别在于 codec 设计: VALL-E 使用 EnCodec 的 RVQ 多层离散 tokens,SPEAR-TTS 使用 w2v-BERT semantic + SoundStream acoustic 两阶段;Vec-Tok Speech 则提出"连续向量(vectors) + 离散语义 token(K-means)"的双轨表征,试图在两种表征粒度间取得更好的 trade-off。
>
> **已有认知**:
> - [[Semantic vs Acoustic Tokens]]: 语义 token 与声学 token 的 trade-off 是 SpeechLM 设计的核心问题。语义 token 语义连贯但声学细节弱,声学 token 高保真但序列长且语义对齐差。Vec-Tok Speech 的方案是: 不对声学信息做离散量化(保留连续向量),只对语义信息做离散化。
> - [[Speech Factorization]]: 语音因子分解(content/speaker/style 解耦)是可控 TTS 的前提。Vec-Tok Speech 通过 utterance-level mean normalization 分离 speaker 信息,将 speaker identity 放入连续向量、linguistic+style 放入 semantic tokens。
> - [[Residual Vector Quantization]]: RVQ 是主流 codec (SoundStream, EnCodec) 的量化核心,但压缩所有信息到离散 token 导致信息冗余和序列过长。Vec-Tok Speech 明确回避 RVQ,改用 K-means 提取低比特率语义 token。
> - [[LLM-based TTS]]: VALL-E 开创的范式,核心是将 TTS 视为条件语言建模。Vec-Tok Speech 沿用这一范式但简化了 LM 负担——LM 只需建模单层语义 token 序列(而非 VALL-E 的多层 RVQ token)。
> - [[Speech Tokenizer]]: K-means 聚类是最早的 semantic tokenizer 方案(HuBERT 即用此方法),Vec-Tok Speech 沿用并加入 BPE 压缩。
>
> **创新判断**: 相对于 KB 已有认知,Vec-Tok Speech 的核心新颖点在于: (1) 显式保留连续向量而非全部量化,用连续向量承载声学保真度; (2) 通过 inverse K-means 模型(Conformer)将离散 token + speaker prompt 向量还原为连续向量,实现 speaker identity 的 prompt-based 注入; (3) BPE 压缩 semantic token 序列(从 50 Hz 降至约 16 Hz)。
>
> 检索命中: [[Semantic vs Acoustic Tokens]]✓, [[Speech Factorization]]✓, [[Speech Language Model]]✓, [[Residual Vector Quantization]]✓, [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓ | 过滤: [[Self-Supervised Speech Representation]](pending-review), [[Token Rate and Bitrate Trade-offs]](pending-review), [[Style Transfer in TTS]](pending-review), [[Single-codebook vs Multi-codebook]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 提出 Vec-Tok Codec,将语音分为连续向量(WavLM 6 层特征,承载全部声学细节)和离散语义 token(K-means 聚类,承载语言和副语言信息),LM 只建模语义 token,解码时用 inverse K-means + vocoder 恢复高保真语音
> - **路线**: 语音 → WavLM → 连续向量(vectorization); 连续向量 → mean-norm → K-means → 语义 token(tokenization); 文本/源语音 → AR LM → BPE 语义 token → inverse K-means(+ speaker 向量 prompt) → 连续向量 → HiFi-GAN → 波形
> - **指标**: ZS-TTS Naturalness MOS 3.92 vs VALL-E X 3.72 / Bark 3.68 [Table 3]; Speaker SIM MOS 3.87 vs 3.67/3.73; SCS 0.909 vs 0.852/0.866; ZS-VC SCS 0.927 vs LM-VC 0.892 [Table 1]; 50k 小时训练, 600M 参数
> - **可借鉴**: (1) BPE 压缩 speech token 序列(50→16 Hz)降低 LM exposure bias + 增加 context 覆盖; (2) utterance-level mean normalization 作为轻量 speaker 解耦手段; (3) "保留连续向量 + 只量化语义"的 codec 设计思路,避免 RVQ 的信息压缩瓶颈
> - **局限**: 仅发表在 arXiv preprint,未正式发表; 推理需 CLVP rescoring(256 候选序列,计算成本高); 未报告实时因子 RTF; 代码已开源但社区关注度有限; 与后续 CosyVoice/NaturalSpeech 3 等方案相比缺乏 diffusion/flow 基声学建模

## 核心问题

当前 LLM-based 语音生成面临一个根本矛盾: RVQ-based codec(如 EnCodec)将所有语音信息压缩到离散 token 中,虽然低比特率但重建质量受损且序列长度大,给 LM 建模带来挑战;而直接使用连续向量虽保真度高,但 LM 无法自回归建模连续表征 [§1]。Vec-Tok Speech 的核心问题是: **如何同时获得高保真重建(需要连续表征的丰富信息)和高效语言建模(需要低比特率离散 token)?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Vec-Tok Speech 由三个核心组件组成 [§3.1, Fig 1]:

1. **Codec Encoder**: 从语音波形中提取两类表征:
   - **Speech vectors** (连续): WavLM 第 6 层输出,包含细粒度声学信息
   - **Semantic tokens** (离散): 对 speech vectors 做 mean normalization → K-means 聚类
2. **Language Model**: 自回归建模 BPE 编码后的 semantic token 序列
3. **Codec Decoder**: 由 inverse K-means 模型 + HiFi-GAN vocoder 组成,将 semantic tokens + speaker prompt 向量重建为波形

形式化 [§3.1, Eq. 1-3]:
```
{vec, tok} = Encoder(wave)
tok_hat = LM(tok, text)
wave_hat = Decoder(vec, tok_hat)
```

### 关键设计选择

**选择 1: 为什么用 WavLM 第 6 层作为 speech vector?**

[论文原文] WavLM 是在大规模数据上预训练的 SSL 模型,第 6 层被证明包含丰富的声学信息,足以高保真重建语音 [§3.2.1, 引 Baas et al. 2023]。[agent 解读] 较低层(如第 6 层)保留更多底层声学细节(声道特征、能量包络),而较高层则更侧重语义抽象。选择第 6 层是在"足够保真"和"不过度抽象"之间的平衡。

**选择 2: 为什么用 utterance-level mean normalization 做 speaker 解耦?**

[论文原文] 在时间轴上平均的语音向量与说话人身份高度相关(附录 A.2 的 t-SNE 可视化证实了这一点)[§3.2.2]。因此减去 utterance-level mean 可以去除 speaker-related 信息,保留 linguistic + para-linguistic 信息。[agent 解读] 这是一种极其轻量的 speaker 解耦方法,无需对抗训练或额外编码器,依赖 WavLM 特征空间的固有结构: speaker 信息集中在 global mean 中,而 content/style 信息集中在 frame-level variation 中。

**选择 3: 为什么用 K-means(300 clusters)而非 RVQ?**

[论文原文] RVQ-based codec 试图在离散 token 中编码所有语音属性(相位、音色、韵律、内容),导致信息冗余并增加下游 LM 的预测难度。多层 RVQ token 序列极长,也给生成模型带来挑战 [§2.1]。[agent 解读] 用 K-means 替代 RVQ 的核心思路是: 既然高保真重建已交由连续向量保证,离散 token 只需编码语义信息就够了。K-means 聚类自然产生扁平的单层 codebook(300 entries),序列是一维的,LM 只需一个阶段即可建模,远比 VALL-E 的 AR+NAR 两阶段简单。

**选择 4: 为什么引入 BPE 压缩?**

[论文原文] WavLM 产生 50 tokens/sec 的 semantic token 序列,导致 exposure bias 和短 context 覆盖问题。BPE 将频繁共现的 token 对合并(vocab size 8192),压缩到约 16 tokens/sec,降低了 exposure bias 并增加了 LM 的有效 context 长度 [§3.3]。实验证实去除 BPE 后 naturalness MOS 从 3.92 降至 3.80,CER 从 3.7% 升至 4.6% [Table 3]。

**选择 5: 为什么设计 inverse K-means 模型?**

[论文原文] 由于 semantic tokens 经过 mean normalization 去除了 speaker 信息,需要一个模型将 token 还原为包含目标说话人信息的 speech vectors。Inverse K-means 模型使用 K-means 聚类中心作为 look-up table 的 token embedding,将 speaker prompt 的 speech vectors 拼接在 token embedding 前面作为 Conformer 的输入。Self-attention 的全局感受野使得 token embedding 可以从 prompt 向量中捕获音色等声学信息 [§3.2.2, Eq. 8-9]。

### 训练策略

系统各组件分开训练:

1. **Vocoder (HiFi-GAN V1)**: 以 WavLM 第 6 层输出为输入,用 MPD+MSD 对抗训练 + feature matching loss + mel 重建 loss [§3.2.1, Eq. 6]。4x 3090 GPU, batch size 4/GPU, 2M steps。
2. **Inverse K-means**: Decoder-only Conformer (6 blocks, 8 heads, dim 1024, FFN 4096)。训练使用 MSE 重建 loss + SSIM loss [§3.2.2, Eq. 9],额外引入 mask prediction 增强语言理解(随机 mask 10% token 或替换 10%)。8x 3090 GPU, batch size 12/GPU, 500k steps。
3. **TTS LM**: LLaMA 架构 (12 layers, 12 heads, hidden 1536, FFN 6144), 以 phoneme + BPE prompt tokens 为输入,交叉熵 loss [§3.3, Eq. 10]。8x A800 GPU, batch size 64/GPU, 10 epochs, cosine LR schedule。
4. **CLVP**: BERT 架构 (6 layers, 8 heads, hidden 512),用于 TTS 推理时对 256 候选序列 rescoring。
5. **S2ST LM**: 同 TTS LM 架构,以源语言 semantic tokens 为输入 [§3.3, Eq. 11]。

## 实验

| 指标 | 本文 (Vec-Tok Speech) | Baseline | 数据集/任务 | 出处 |
| --- | --- | --- | --- | --- |
| Naturalness MOS (↑) | 3.86 | LM-VC: 3.78 | Intra-lingual ZS-VC | [Table 1] |
| Speaker Sim MOS (↑) | 3.93 | LM-VC: 3.74 | Intra-lingual ZS-VC | [Table 1] |
| SCS (↑) | 0.927 | LM-VC: 0.892 | Intra-lingual ZS-VC | [Table 1] |
| CER% (↓) | 3.0 | LM-VC: 3.2 | Intra-lingual ZS-VC | [Table 1] |
| Naturalness MOS (↑) | 3.81 | LM-VC: 3.60 | Cross-lingual ZS-VC | [Table 2] |
| SCS (↑) | 0.919 | LM-VC: 0.886 | Cross-lingual ZS-VC | [Table 2] |
| Naturalness MOS (↑) | 3.92 | VALL-E X: 3.72 / Bark: 3.68 | ZS-TTS | [Table 3] |
| Speaker Sim MOS (↑) | 3.87 | VALL-E X: 3.67 / Bark: 3.73 | ZS-TTS | [Table 3] |
| SCS (↑) | 0.909 | VALL-E X: 0.852 / Bark: 0.866 | ZS-TTS | [Table 3] |
| CER% (↓) | 3.7 | VALL-E X: 5.3 / Bark: 5.8 | ZS-TTS | [Table 3] |
| WER% (↓) | 3.4 | VALL-E X: 3.9 / Bark: 3.6 | ZS-TTS | [Table 3] |
| Naturalness MOS (↑, w/o BPE) | 3.80 | w/ BPE: 3.92 | ZS-TTS ablation | [Table 3] |
| CER% (↓, w/o BPE) | 4.6 | w/ BPE: 3.7 | ZS-TTS ablation | [Table 3] |
| Style Sim MOS (↑) | 3.87 | - | ZS Style Transfer TTS | [Table 3] |
| BLEU (↑) | 21.56 | GigaST: 22.30 | En→Zh S2ST | [Table 4] |
| Naturalness MOS (↑) | 3.69 | - | S2ST | [Table 4] |
| SCS (↑) | 0.904 | - | S2ST | [Table 4] |

**关键消融**: 去除 inverse K-means (用 frame-level phoneme 替代 semantic tokens) 导致 speaker similarity MOS 从 3.93 降至 3.67 [Table 1],证实 inverse K-means + prompt 机制对 speaker identity 注入的关键作用。

## 局限性

1. **推理效率**: CLVP rescoring 需要生成 256 候选序列并逐一打分,推理成本很高。论文未报告 RTF 或推理延迟 [agent 解读]。
2. **Speaker 解耦的粗糙性**: utterance-level mean normalization 是一种非常粗糙的 speaker 解耦方法。从 KB 中 [[Speech Factorization]] 页面可知,更先进的方法(如对抗训练、self-distillation、信息瓶颈)能实现更精细的解耦 [agent 解读]。
3. **缺乏 diffusion/flow 声学建模**: 解码路径 (inverse K-means → vocoder) 完全依赖确定性映射,没有引入 diffusion/flow matching 等概率声学模型。后续工作如 CosyVoice(LLM + CFM)和 NaturalSpeech 3(factorized diffusion)在这方面更为先进 [agent 解读]。
4. **评估有限**: 仅与 LM-VC、VALL-E X、Bark 做对比,未与 NaturalSpeech 2 等同期强 baseline 对比。主观评估仅 30 名参与者 [§4]。
5. **仅 arXiv preprint**: 未在主流会议/期刊正式发表,表明可能存在审稿过程中暴露的问题 [agent 解读]。
6. **连续向量的传输成本**: 高保真重建依赖传输 WavLM 第 6 层的全部连续向量,信息量远大于 RVQ token,在流式/低带宽场景下不适用 [agent 解读]。

## 点评

Vec-Tok Speech 的核心 insight——"不要把所有信息都塞进离散 token,保留连续向量承载声学保真度,只把语义信息量化给 LM"——在 2023 年 10 月发表时是一个有前瞻性的设计直觉。它预见了后来 CosyVoice 系列"semantic token + flow matching"的双轨思路,只是 Vec-Tok Speech 用确定性的 inverse K-means + vocoder 替代了 CosyVoice 的 CFM。

从 KB 已有知识看,这篇工作在 [[Semantic vs Acoustic Tokens]] 的 trade-off 谱系中占据一个独特位置: 它既不是 AudioLM 式的"两种离散 token 串联",也不是 SpeechTokenizer 式的"混合 tokenizer",而是"连续 + 离散"的异构组合。这种设计完全避开了 RVQ 的 codebook collapse、多层建模等难题,代价是连续向量需要额外传输/存储。

BPE 压缩 speech token 的技巧(50→16 Hz)在 2023 年相当有见地,直接预示了后来 LatentLM/CLEAR 等工作追求更低 token rate 的趋势。实验证据(Table 3 ablation)也清晰支撑了其有效性。

不足之处在于系统偏"组装式": WavLM 特征提取、K-means 聚类、BPE 编码、Conformer 逆映射、CLVP rescoring、HiFi-GAN vocoder——组件众多但缺乏端到端优化,这可能限制了系统的上限。

## 可复用的 idea

1. **BPE 压缩 speech token 序列**: 将 subword tokenization 思想迁移到 speech token,用频繁 token pair merging 降低序列长度和 LM exposure bias。这个技巧成本极低,可直接插入任何基于离散 speech token 的 pipeline。
2. **Utterance-level mean normalization 做 speaker 解耦**: 利用 SSL 特征空间中 global mean ≈ speaker identity 的经验规律,零参数实现粗粒度 speaker-content 分离。适合快速原型验证。
3. **Inverse K-means 的 prompt-based speaker injection**: 用目标说话人的连续向量作为 prefix prompt,通过 Conformer self-attention 注入 speaker identity 到内容 token embedding。这种 prompt-based 注入范式比显式 speaker embedding 更灵活。
4. **"连续+离散"双轨表征策略**: 在需要同时满足高保真重建和高效生成的场景中,不必将所有信息量化为离散 token,可保留连续向量承载难以量化的声学细节。
