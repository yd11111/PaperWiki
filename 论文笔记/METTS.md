---
type: paper
tier: deep
title: "METTS: Multilingual Emotional Text-to-Speech by Cross-speaker and Cross-lingual Emotion Transfer"
arxiv_id: "2307.15951"
source: "Sources/METTS.pdf"
authors: [Xinfa Zhu, Yi Lei, Tao Li, Yongmao Zhang, Hongbin Zhou, Heng Lu, Lei Xie]
year: 2023
venue: "arXiv (TASLP format)"
tags: [TTS, emotional-TTS, cross-lingual, multilingual, emotion-transfer, disentanglement, GST, CVAE, VQ, information-perturbation, non-autoregressive]
concepts: ["[[EmotionControlinTTS]]", "[[GlobalStyleTokens]]", "[[SpeechFactorization]]", "[[VariationalAutoencoderforTTS]]", "[[StyleTransferinTTS]]"]
models: []
tasks: ["[[Cross-lingualVoiceCloning]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: METTS 处于 Emotion Control in TTS 与 Cross-lingual Voice Cloning 的交叉地带。KB 中 EmotionControlinTTS 页面记录了从 emotion embedding (2021) → 多尺度层级建模 (MsEmoTTS, 2022) → 跨说话人情感迁移 (2022) 的演进链,METTS 正是这条链上"将情感迁移进一步扩展到跨语言场景"的关键节点。GlobalStyleTokens 页面记载 GST 是无监督风格表示的奠基机制,METTS 在此基础上增加 L2 normalization + 半监督约束使 GST 编码 language-agnostic 情感。SpeechFactorization (confirmed) 页面总结了对抗训练、信息瓶颈、self-distillation 等解耦范式,METTS 的 formant-shift information perturbation 属于"辅助技术"类目中的信号扰动方案。VariationalAutoencoderforTTS 页面覆盖了 CVAE 在 TTS 中的应用,METTS 的 fine-grained CVAE 模块是 VAE 用于"风格解耦与控制"的一个实例。Cross-lingualVoiceCloning (confirmed) 页面记录了跨语言克隆的核心挑战(音色-语言解耦、韵律模式差异),METTS 是该任务中"同时处理情感迁移"的早期尝试。
>
> **已有认知**: 情感控制的主流路线已发展到 LLM prompt (PUE)、activation steering (EmoSteer-TTS/DUET)、reward-guided (DiffRO) 等更现代方案,METTS 的 GST+CVAE 属于 pre-LLM 时代的代表性方法。跨语言 TTS 已有 CosyVoice 3、Qwen3-TTS 等大规模系统,METTS 的 DelightfulTTS backbone + 小规模数据 (~41h) 代表了早期小模型方案。
>
> **创新判断**: METTS 的核心创新在于将"多尺度情感建模"与"跨语言解耦"结合 -- coarse-grained language-agnostic 表示解决跨语言情感迁移,fine-grained language-specific 表示保留语言特有韵律,这种双层设计在 KB 中尚未被其他论文完整覆盖。VQ-based emotion matcher 实现 reference-free 推理也是一个有价值的工程设计。
>
> 检索命中: [[SpeechFactorization]]✓, [[Cross-lingualVoiceCloning]]✓ | 过滤: [[EmotionControlinTTS]](pending-review), [[GlobalStyleTokens]](pending-review), [[StyleTransferinTTS]](pending-review), [[VariationalAutoencoderforTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出多尺度情感建模(coarse-grained GST + fine-grained CVAE)+ formant-shift 信息扰动 + VQ emotion matcher,实现单语说话人的跨语言跨说话人情感语音合成
> - **路线**: Text → DelightfulTTS Conformer Encoder → Variance Adaptor (+Speaker Embedding) → Mel Decoder; 情感条件来自 (a) Reference mel → Formant Perturb → GST (coarse) + CVAE (fine) 或 (b) Text+EmotionID → VQ Emotion Matcher → coarse embedding + Fine-grained Predictor
> - **指标**: MOS 4.11 (intra-lingual, CN speakers) / 4.00 (cross-lingual); Speaker SIM 3.94/3.94; Emotion SIM 4.12/3.44; Cosine SIM 0.813 (intra) / 0.753 (cross); 全面优于 CET 和 M3 [Table II-V]
> - **可借鉴**: (1) L2 normalization 消除跨语言 magnitude 差异,使 GST 编码 language-agnostic 情感; (2) Formant shift 作为 pre-processing 步骤去除 speaker timbre,比 GRL 更稳定; (3) VQ 将回归问题转化为分类问题,简化 text→emotion embedding 的映射
> - **局限**: 仅双语(中英),仅女性说话人,数据规模极小(~41h),backbone 过时(DelightfulTTS/MelGAN),英语训练数据无情感标注导致跨语言英语情感表达偏弱,未开源

## 核心问题

METTS 要解决的核心问题是: **如何在数据高效的条件下(每个说话人只说一种语言,部分说话人只有中性语音)实现跨说话人、跨语言的情感语音合成?**

这个问题包含三个相互纠缠的子挑战 [§I]:
1. **Foreign accent problem**: 跨语言合成时,源语言的韵律模式(特别是情感韵律)会泄漏到目标语言,产生外国口音
2. **Speech entanglement problem**: Speaker timbre、emotion、language 三个因素在语音信号中深度纠缠,跨说话人情感迁移时会导致音色泄漏
3. **Emotional diversity problem**: Reference-based 情感合成在跨语言场景下如何选择合适的参考信号(L1 语言的参考如何匹配 L2 语言的文本)

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

METTS 以 DelightfulTTS [19] 为 backbone,包含 text encoder、variance adaptor 和 mel-spectrogram decoder,均使用 improved Conformer blocks [Fig 1] [§III.A]。在此骨架上叠加三个核心模块:

1. **Multi-scale emotion modeling**: GST layer (coarse-grained, language-agnostic) + CVAE module (fine-grained, language-specific)
2. **Information perturbation**: Formant shift pre-processing 去除 reference 中的 speaker timbre
3. **VQ-based emotion matcher**: 将 coarse-grained 表示量化为 codebook,通过 MLP+CC 匹配 text 和 emotion ID

两种推理模式:
- **METTS-REF**: 输入参考 mel → 提取 coarse + fine 情感 embedding → 合成
- **METTS-ID**: 输入 text + emotion ID → emotion matcher 从 codebook 选取 coarse embedding → fine-grained predictor 预测 fine embedding → 合成(reference-free)

### 关键设计选择

#### 1. 多尺度情感建模: 为什么分 coarse 和 fine?

[论文原文] 作者认为多语言语音的情感表达可以分解为: (a) 跨语言共享的相似韵律模式(如愤怒=高 pitch,悲伤=低 pitch)[21][22]; (b) 因语言发音方式不同而产生的细粒度韵律差异 [23][24] [§III.B]。

**Coarse-grained (GST layer)**:
- Reference encoder (CNN+GRU) → Multi-head attention over style token bank → coarse embedding
- 关键设计: **L2 normalization** 应用于 coarse embedding,消除跨语言和跨说话人的 magnitude 差异,使情感控制仅基于方向(角度)信息 [59] [§III.B]
- 半监督 emotion classifier 约束: 仅对有情感标注的样本训练分类器,无标注样本不参与分类器优化但通过 embedding 训练声学模型 [§III.B]

[agent 解读] L2 normalization 是一个巧妙设计 -- 不同语言/说话人的情感表达在 magnitude 上可能差异很大(如中文情感表达比英文更"外显"),归一化到单位球面后,只保留方向信息,天然具备 language-agnostic 特性。

**Fine-grained (CVAE module)**:
- Conformer + GRU 从 mel 提取 frame-level emotion embedding → Phone average 得到 phoneme-level embedding → 推导分布的 mean/variance [§III.B]
- 条件: text encoder output + coarse-grained emotion embedding
- Fine-grained predictor: 从 text 预测 fine-grained 分布,使用 normalizing flow (参照 VITS [7]) 增强预测分布的表达力 [§III.B]

[agent 解读] CVAE 以 text 和 coarse emotion 为条件,学到的 fine-grained 表示自然与输入语言绑定(因为 text 是 language-specific 的),从而实现"语言特有的情感韵律细节"。这种条件化设计是解决 foreign accent problem 的关键 -- 跨语言时,coarse embedding 传递通用情感意图,CVAE 根据目标语言 text 生成适配该语言的韵律细节。

#### 2. Information perturbation: 为什么用 formant shift?

[论文原文] 语音共振峰主要由声道的大小、形状和位置决定,这些高度特定于每个说话人,代表其声音身份 [60]。通过随机偏移参考语音的 formant frequency,可以去除 speaker-specific 的音色信息 [§III.C]。

具体实现: 对原始波形应用动态 formant shift 函数 $f_s$,得到 $\widetilde{Wave} = f_s(Wave)$,提取扰动后的 mel 作为情感表示提取的输入 [§III.C]。训练时每步随机扰动,使 GST 和 CVAE 学到 speaker-independent 的情感表示。

[agent 解读] 相比 GRL 对抗训练(M3 使用但不稳定 [72]),formant shift 是一种更直接、更稳定的解耦策略 -- 在信号层面物理地破坏 speaker identity 信息,而非在表示层面通过优化目标间接实现。这与 KB 中 SpeechFactorization 记录的 Seed-VC 的 "external timbre shifter" 方案思路一致,但 METTS 用 formant shift 而非 VC 模型,更轻量。

#### 3. VQ-based emotion matcher: 为什么不直接回归?

[论文原文] 直接建模双语文本表示到情感表示的回归关系非常困难,因此用 VQ 将 coarse emotion 量化为 N x M 个聚类中心(N 个聚类/情感类别,M 个情感类别),将回归问题转化为分类问题 [§III.D]。

Matcher 结构 [Fig 2]:
1. 用 k-means 对所有训练样本的 coarse embedding 按情感类别聚类,得到 N*M 个 reference embeddings (codebook)
2. MLP 将 text encoder output + emotion ID 映射为 text-emotion vector
3. 计算 text-emotion vector 与 codebook 的 correlation coefficient (CC) 矩阵(类似 scaled dot-product attention) [Eq. 1]
4. 选取 CC 最高的 embedding 作为 coarse emotion representation
5. Matcher classifier 监督 CC 矩阵,确保选中的 embedding 正确对应输入 text [§III.D]

[agent 解读] 这个设计的核心 insight 是: text→emotion 的映射空间太大且连续,直接回归容易陷入 mode averaging;通过 VQ 先将情感空间离散化,再从有限的候选中选择,大幅降低了问题复杂度。N=64 时在准确率和多样性之间取得最佳平衡 [Table VIII, IX]。

### 训练策略

**两阶段训练** [§III.E]:

1. **Pre-training (METTS-REF)**: 使用 reference mel 提供情感条件
   - $L_{pretrain} = 0.05 \cdot L_{kl} + L_{prosody} + 0.1 \cdot L_{emo} + L_{ssim} + L_{iter}$ [Eq. 2]
   - $L_{prosody}$: pitch/energy/duration 的 L1 loss
   - $L_{emo}$: 半监督 cross-entropy(仅有标注样本)
   - $L_{kl}$: 从 text 预测 phoneme-level emotion 分布的 KL divergence
   - $L_{ssim}$: mel structural similarity [62]
   - $L_{iter}$: 每个 Conformer block 的 mel L1 loss

2. **Fine-tuning (METTS-ID)**: 使用 ground-truth clustering center 替代 reference embedding,联合训练 emotion matcher
   - $L_{finetune} = L_{match} + L_{disc} + L_{base'}$ [Eq. 3]
   - $L_{match}$: matcher 的 cross-entropy loss
   - $L_{disc}$: 冻结 GST layer 作为判别器,判断生成 mel 的情感类别
   - 冻结 GST layer 和 emotion classifier 稳定联合训练 [§III.E]

## 实验

| 指标 | METTS-REF | METTS-ID | CET | M3 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS (CN text, CN spk) | 4.11+-0.12 | 4.07+-0.13 | 3.65+-0.14 | 2.69+-0.19 | 内部中英双语 | [Table II] |
| Speaker SIM (CN text, CN spk) | 3.94+-0.16 | 3.88+-0.16 | 3.69+-0.12 | 3.35+-0.15 | 同上 | [Table II] |
| Emotion SIM (CN text, CN spk) | 4.12+-0.14 | 3.95+-0.13 | 4.00+-0.11 | 3.21+-0.18 | 同上 | [Table II] |
| MOS (EN text, CN spk) | 4.00+-0.11 | 4.06+-0.18 | 3.01+-0.19 | 2.49+-0.23 | 同上 | [Table II] |
| MOS (CN text, EN spk) | 3.91+-0.14 | 4.02+-0.15 | 3.08+-0.16 | 2.72+-0.15 | 同上 | [Table III] |
| MOS (EN text, EN spk) | 3.95+-0.14 | 4.05+-0.18 | 2.89+-0.12 | 2.41+-0.19 | 同上 | [Table III] |
| Cosine SIM (CN text, CN spk) | 0.813 | 0.805 | 0.726 | 0.754 | 同上 | [Table IV] |
| Cosine SIM (EN text, CN spk) | 0.753 | 0.711 | 0.638 | 0.673 | 同上 | [Table IV] |
| CER (CN text, CN spk) | 0.48 | 0.48 | 0.35 | 11.02 | 同上 | [Table IV] |
| WER (EN text, CN spk) | 5.60 | 5.46 | 12.65 | 55.32 | 同上 | [Table IV] |

**Ablation 关键发现** [Table VI, VII]:
- w/o GST: emotion SIM 暴跌 (4.12→3.19, CN text/CN spk),证明 coarse-grained language-agnostic 表示是跨语言情感迁移的关键
- w/o CVAE: cross-lingual naturalness 显著下降 (4.00→3.43, EN text/CN spk),证明 fine-grained language-specific 表示对消除 foreign accent 至关重要
- w/o Perturb: emotion SIM 略有上升 (4.12→4.19) 但 speaker SIM 暴跌 (3.94→3.55, cross-lingual),说明 formant shift 在解耦 speaker timbre 方面不可或缺,emotion SIM 上升是因为 speaker timbre 与 emotion 纠缠后"假性"提升了情感表达
- VQ clusters N=64 在所有指标上最优,N=32 准确率高但多样性不足,N=96 多样性高但准确率下降 [Table VIII, IX]

**T-SNE 可视化** [Fig 3]:
- 按 emotion 着色: 中文情感清晰聚类,部分英文情感 embedding 与中文混合 → 证明 coarse-grained 表示的 language-agnostic 特性
- 按 speaker 着色: 情感 embedding 无明显 speaker 聚类 → 证明 speaker-independence

## 局限性

1. **语言覆盖极窄**: 仅中英双语,且英语训练数据无情感标注,导致英语情感表达偏弱 [§VII]。作者承认需要"emotional English corpus"来改善 [§VII]
2. **说话人多样性不足**: 仅 4 个女性说话人(2 中 2 英),未验证男性/混合场景
3. **数据规模极小**: ~41 小时,与现代大规模 TTS 系统(数万小时)不可同日而语
4. **Backbone 过时**: DelightfulTTS + MelGAN vocoder,在 2023 年已不是 SOTA(对比 VITS、NaturalSpeech 系列)
5. **Evaluation 局限**: MOS/SMOS 由 22 名有基础英语技能的志愿者评价,样本量偏小;缺少与 ground truth 的直接对比
6. **未验证零样本泛化**: 所有评估均在训练集说话人上进行,未验证对未见说话人的泛化能力
7. **情感类型受限**: 仅 6 类离散情感,未探索连续情感空间(arousal-valence)或混合情感

## 点评

METTS 是一篇在 pre-LLM 时代系统性解决"跨语言+跨说话人+情感"三重迁移问题的工作。核心贡献不在于单个模块的创新(GST 和 CVAE 都是已有技术),而在于将它们有机组合并赋予新的语义: coarse=language-agnostic, fine=language-specific。这种"分层解耦 + 分别赋予语义"的设计思路具有方法论价值。

**与 KB 中已有方法的对比**:
- vs MsEmoTTS (2022, 同组工作): MsEmoTTS 是单语言多尺度情感,METTS 扩展到跨语言
- vs DiEmo-TTS (2025): DiEmo-TTS 用 DINO 自监督蒸馏替代 GRL 做情感-音色解耦,也用了 formant perturbation,但未处理跨语言问题
- vs 现代大模型方案 (EmoSteer-TTS, DiffRO, WeSCon): METTS 的 GST+CVAE 路线已显过时,现代方法在 flow-matching/LLM backbone 上通过 activation steering/reward guidance 实现更强的情感控制,且不需要显式的情感标注

**历史定位**: METTS 代表了"显式解耦 + 多尺度建模"范式在跨语言情感 TTS 上的顶点。后续发展方向已转向: (1) 大规模预训练 + in-context learning 取代显式 emotion embedding; (2) activation/parameter space manipulation 取代 reference-based 控制; (3) 连续 AVD 空间取代离散情感标签。

## 可复用的 idea

1. **L2 normalization 使 style embedding language-agnostic**: 当需要从多语言数据中提取共享的风格/情感表示时,对 embedding 做 L2 归一化是一个简单有效的 trick,消除 magnitude 差异,保留方向信息
2. **Coarse-fine 双层情感分解**: "共享 + 语言特有" 的分层思路可扩展到其他需要跨域迁移的场景(如跨说话人韵律迁移时分离 speaker-agnostic 和 speaker-specific 韵律)
3. **Formant shift 作为轻量 speaker disentanglement**: 比 GRL 对抗训练更稳定,比 Seed-TTS 的 VC-based perturbation 更轻量,适用于训练数据有限的场景
4. **VQ 将回归转分类**: 当 embedding 空间的直接回归困难时,先 k-means 聚类再分类选择是一个通用降复杂度策略

## 审阅

(待独立审阅 agent 填写)
