---
type: paper
tier: deep
title: "The Codec Language Model-based Zero-Shot Spontaneous Style TTS System for CoVoC Challenge 2024"
arxiv_id: "2412.01100"
source: "Sources/CodecLMSpontaneousStyle.pdf"
authors: [Shuoyi Zhou, Yixuan Zhou, Weiqin Li, Jun Chen, Runchuan Ye, Weihao Wu, Zijian Lin, Shun Lei, Zhiyong Wu]
year: 2024
venue: "ISCSLP 2024 CoVoC Challenge"
tags: [TTS, codec-LM, zero-shot, voice-cloning, spontaneous-speech, CFG, delay-pattern, LLaMA, Mandarin]
concepts: ["[[CodecLanguageModel]]", "[[Classifier-FreeGuidance]]", "[[LLM-basedTTS]]", "[[SemanticvsAcousticTokens]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓, [[Zero-shotSpeechSynthesis]]✓, [[CodecLanguageModel]], [[Classifier-FreeGuidance]], [[VoiceCloningTaxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]], [[SemanticvsAcousticTokens]], [[Zero-shotSpeechSynthesis]], [[CodecLanguageModel]], [[Classifier-FreeGuidance]], [[VoiceCloningTaxonomy]] | 过滤: 无 | 未命中但可能相关: [[ProsodyModeling]], [[StyleTransferinTTS]]

**谱系定位**: 本文属于 Codec Language Model 范式中的 LLM-based TTS 路线,具体是 VALL-E 系列开创的 AR codec LM for zero-shot TTS 的延伸。与主流 VALL-E 系列的 AR+NAR 两阶段不同,本文采用 delay pattern(源自 MusicGen)在单一 AR 框架内同时预测多层 RVQ tokens。在 token 层级上采用 semantic→acoustic 两阶段生成,与 AudioLM/SPEAR-TTS 一脉相承。

**已有认知**: KB 中 [[Classifier-FreeGuidance]] 已记录 CFG 在 TTS 中的广泛应用(Guided-TTS 2、CosyVoice、NaturalSpeech 3 等),但主要集中在 diffusion/flow 连续空间。本文将 CFG 应用于 AR codec LM 的离散 logits 空间,与 [[Classifier-FreeGuidance]] 中记录的"离散空间 CFG"(OmniVoice)属于同一技术路线,但本文更早(2024 vs 2026)且用于增强文本→语义和语义→声学两个阶段的条件引导。

**创新判断**: 本文的核心贡献在于将 delay pattern + CFG 的组合应用于自发风格语音克隆。KB 中尚未有专门针对 spontaneous speech style 建模的概念页,这是一个相对空白的子领域。数据驱动的自发风格建模(不使用显式标签)是与 SponLMTTS 的关键区分点。

## 速查

> [!summary] 速查
> - **一句话**: LLaMA-based codec LM + delay pattern + CFG 实现零样本自发风格语音克隆,在 CoVoC 2024 受限赛道获自然度 MOS 第一(3.80)
> - **路线**: 文本→MT5 encoder(+LoRA)→text encoding; 语音→HuBERT+K-means→semantic tokens + DAC→acoustic tokens; LLaMA AR 生成 semantic tokens→delay pattern AR 生成 acoustic tokens→DAC decoder→波形
> - **指标**: 自然度 MOS 3.80(第1）、质量 MOS 3.84（第2）、相似度 MOS 3.49（第2）、自发风格 MOS 3.33（第3）、CER 10.29%（第2）、SECS 0.797（第4）[Table 2, 3]
> - **可借鉴**: delay pattern 在自发语音韵律建模中的适配性；CFG 同时应用于 semantic 和 acoustic 两阶段的双重引导策略；渐进式预训练→微调的数据策略
> - **局限**: 纯数据驱动无显式自发行为标签,自发风格 MOS 仅 3.33；voice cloning 性能中等(SECS 0.797 仅排第4)；仅在 CoVoC 受限赛道评估,无标准 benchmark 对比

## 核心问题

本文要解决的核心问题是: **如何在零样本语音克隆场景下同时实现高自然度和自发风格建模?** 现有零样本克隆系统虽能复制音色,但在日常对话场景中的自发风格(犹豫、停顿、语气词等副语言现象)表现不足。传统自发风格建模方法(如 SponLMTTS)又不支持多说话人/零样本场景。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统是一个两阶段的 LLaMA-based codec language model [§3.2, Fig 1]:

**第一阶段 (Text → Semantic Tokens)**: 以 MT5 text encoder 的输出 E_txt 为条件,自回归预测 HuBERT semantic token 序列 S。Semantic tokens 通过 HuBERT 9 层 hidden states + K-means(codebook size=500, 50Hz)获得,并去除连续重复 token [§3.1]。生成结束标志为 \<S_eos\> token。

**第二阶段 (Semantic → Acoustic Tokens)**: 以 E_txt、semantic tokens S 和 speech prompt 的 acoustic tokens A_p 为条件,通过 delay pattern 自回归生成 DAC acoustic tokens。DAC 使用 12 层 codebook、16kHz 采样率、总下采样率 320 [§3.1]。

**骨干网络**: 基于 LLaMA transformer,12 层,hidden size 1024,FFN 4096,使用 flash attention。最大序列长度 2560(text 512 + ST+AT 2048)[§4.1]。

### 关键设计选择

**1. Delay Pattern for Multi-codebook Prediction [§3.2]**

[论文原文] 受 MusicGen 和 ParlerTTS 启发,采用 delay pattern 排列多层 RVQ acoustic tokens: 每一层 RVQ token 相对前一层偏移一个时间步。这使得在时间步 t 预测第 k 层 AT 时,可以 condition on 同一时间步第 k-1 层的预测结果。每个时间步同时预测 K 个 token,使用 K 个独立线性头从最终 transformer block 的 hidden state 投射为 K 组 logits。

[论文原文] 作者认为 delay pattern 的 AR 结构"更适合建模自发语音中丰富的韵律变化" [§1]。

[agent 解读] 这一判断的合理性在于: 自发语音的韵律高度不可预测(突然停顿、拖长音节等),AR 建模能逐步决策每个时间步的声学表现,而 NAR 方式(如 VALL-E 第二阶段)需要一次性并行预测所有细粒度信息,可能难以捕捉自发韵律的局部变化。delay pattern 进一步允许同层内的声学一致性(同时间步的不同 RVQ 层可互相 condition),比 VALL-E 的 AR 第一层 + NAR 其余层更连贯。

**2. 加权 Loss [§3.2, Eq. 3]**

第一层 RVQ 承载更多信息,因此引入 alpha 权重调整各层 loss: alpha_k = {5, 2, 1, 0.5, 0.5, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1} for 12 层 DAC [§4.1]。总 loss = semantic token loss + 加权 acoustic token loss。

**3. MT5 Text Encoder + LoRA [§3.1]**

使用预训练 MT5-base 的 text encoder(12 层 transformer, hidden 768),插入 LoRA adapters(alpha=16, r=16)到 query 和 value 层。训练时 MT5 主体冻结,仅 LoRA 可训练 [§4.1]。

[agent 解读] 选择 MT5 而非传统 phoneme 序列的原因是: MT5 预训练了丰富的语义理解能力,对中文文本的 sub-word tokenization 能更好地捕获语义信息,从而改善韵律表现,尤其是对自发语音中的语气词和口语表达。这与 LLM-based TTS 范式中用 LLM text encoder 替代 phoneme pipeline 的趋势一致。

### 训练策略

**三阶段渐进训练 [§4.2]:**

1. **预训练阶段 1** (450K iter): 仅使用 WenetSpeech4TTS(12800h),batch size 64,学习率 1e-4 起步 + 前 10K iter warmup → 建立基础语音合成和声音克隆能力
2. **预训练阶段 2** (250K iter): 在 WenetSpeech4TTS + MAGICDATA-RAMC(180h) + HQ-Conversations(100h) 全部数据集上继续训练 → 引入对话和自发风格数据
3. **微调阶段** (70K iter): 仅使用高质量子集: WenetSpeech4TTS Premium(~1000h) + HQ-Conversations(100h) → 提升合成稳定性和自发风格能力

总训练: 770K iterations,8 x A100 GPUs。

**数据预处理 [§2]:**
- WenetSpeech4TTS: 仅 Premium 子集质量可靠,Standard/Basic/Rest 存在音质差、音量低、说话人重叠、背景噪声等问题
- MAGICDATA-RAMC: 按转写分割句子,去除重叠语音、噪声、音乐、笑声段
- HQ-Conversations: 高质量自发风格对话,200 说话人,100 小时

**CFG 训练 [§3.3, §4.2]:**
- 以 0.1 概率 mask 整个 text encoding sequence 或 semantic token sequence,支持推理时的无条件生成

### Classifier-Free Guidance 策略 [§3.3]

CFG 同时应用于两个生成阶段:

**Semantic token 阶段 [Eq. 4]**: 引导强度 gamma,在 text encoding 条件与无条件之间插值:
```
log P_hat(S_t) = gamma * logP(S_t | E_txt, S_<t) + (1-gamma) * logP(S_t | empty, S_<t)
```

**Acoustic token 阶段 [Eq. 5-6]**: 采用双重引导:
- 第一步 (alpha): text encoding 条件引导,增强文本对声学 token 的影响
- 第二步 (beta): semantic token 条件引导,增强语义 token 对声学 token 的影响

推理时参数: alpha=1.3, beta=1.5, gamma=1.5 [§4.2]。

[agent 解读] 双重 CFG 的设计是本文的一个亮点。在 acoustic token 阶段分别对 text 和 semantic token 做引导,意味着可以独立调节两种条件信号的强度。这在自发语音场景下可能特别有用: 适度的 text 引导(alpha=1.3)确保内容准确但不过度约束韵律自由度,较强的 semantic 引导(beta=1.5)确保声学细节与语义表征一致。

### 推理细节 [§4.2]

- 长句在标点处分割,保证每段至少 30 字符
- 分段合成后用 100ms 静音间隔拼接
- 这种策略规避了模型在训练中未见过的超长句子

## 实验

| 指标 | 本文 | 排名 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| 自然度 MOS | 3.80 (std 0.11) | 1st | CoVoC 2024 | [Table 2] |
| 质量 MOS | 3.84 (std 0.16) | 2nd | CoVoC 2024 | [Table 2] |
| 相似度 MOS | 3.49 (std 0.12) | 2nd | CoVoC 2024 | [Table 2] |
| 自发风格 MOS | 3.33 (std 0.12) | 3rd | CoVoC 2024 | [Table 2] |
| 平均 MOS | 3.61 | 3rd | CoVoC 2024 | [Table 2] |
| CER | 10.29% | 2nd | CoVoC 2024 | [Table 3] |
| SECS | 0.797 | 4th | CoVoC 2024 | [Table 3] |

**Case Study [§4.4, Fig 2]:**
- 模型能生成带自发现象的语音: "嗯" 的拖长表示思考停顿,以及犹豫时的音节延伸
- 通过 mel spectrogram 可视化验证模型对不同字符发音时长的区分能力

## 局限性

1. **自发风格建模深度不足**: 纯数据驱动方法(无显式自发行为标签),自发风格 MOS 仅 3.33,排第3。论文也承认引入显式标签(如 SponLMTTS 的 19 类标签)"有潜力进一步提升性能" [§4.3.2]
2. **声音克隆性能中等**: SECS 0.797 排第4,说明在音色复制方面仍有较大提升空间。[agent 解读] 可能原因: 微调时混入了 WenetSpeech4TTS Premium 以稳定质量,但牺牲了部分自发性和说话人相似度 [§4.3.2]
3. **评估局限**: 仅在 CoVoC 受限赛道评估,无法与 VALL-E、CosyVoice 等主流系统在标准 benchmark(如 SEED-TTS-Eval)上直接对比
4. **模型规模较小**: 12 层 transformer,hidden 1024,相比当前主流系统(CosyVoice 1.5B、Qwen3-TTS 1.7B 等)规模偏小
5. **文本内容鲁棒性**: CER 10.29% 在挑战赛排第2,但绝对值仍较高,说明 intelligibility 仍有改进空间

## 点评

本文是一篇典型的 **challenge system paper**,聚焦于工程实践和系统组合,而非方法创新。核心贡献在于验证了三个技术选择在自发风格零样本克隆场景下的有效性:

1. **Delay pattern vs AR+NAR**: 相比 VALL-E 的 AR 第一层 + NAR 其余层,delay pattern 在单一 AR 框架内处理多层 codebook,对自发韵律建模可能更合适。但论文缺少消融实验直接验证这一假设。

2. **双阶段 CFG**: 将 CFG 从单一条件扩展到两个阶段三个条件(text→semantic, text→acoustic, semantic→acoustic)的引导,是一个有启发性的设计。但同样缺少各 CFG 组件的消融。

3. **数据策略**: 三阶段渐进训练 + 高质量数据微调的实践经验有参考价值,尤其是对 WenetSpeech4TTS 各子集质量的评估。

**不足之处**: 作为 challenge paper,缺少消融实验是最大遗憾。delay pattern vs 其他多 codebook 策略、CFG 各阶段的独立贡献、不同训练阶段的增益,这些关键问题都没有定量回答。自发风格的建模深度也有限 -- 完全依赖数据驱动,没有设计专门的自发行为建模机制。

## 可复用的 idea

1. **双重 CFG 策略**: 在多条件生成中,对不同条件信号使用不同 CFG 引导强度,允许独立调节各条件的影响力。这一策略可推广到任何多条件 AR 生成系统
2. **Delay pattern 在 TTS 中的应用**: 相比 MusicGen 中的音乐生成,delay pattern 在语音(尤其是自发风格语音)中的适配性得到了初步验证
3. **MT5 + LoRA 作为文本编码器**: 利用多语言预训练 text encoder 替代 phoneme pipeline,保留语义信息的同时通过 LoRA 高效适配,这在中文 TTS 场景下是一个值得考虑的选择
4. **渐进式数据策略**: 大规模预训练 → 全数据继续训练 → 高质量微调的三阶段策略,在数据质量参差不齐时有实践价值

## 审阅

> [!review] 审阅 pass (2026-06-08, 0 high / 0 medium / 3 low)
> - **可复述** ✓ 三个关键设计选择均有因果解释; 速查卡片 4 个具体可迁移 trick
> - **可信赖** ✓ 数字 claim 标注覆盖率 ~90%; 指标名无混淆; 无方向性错误
> - **可区分** ✓ 因果解释来源标注覆盖率 ~85%; 推断均有限定词
> - **可定位** ✓ KB 背景 6 个实体页定位; 创新判断有对比基准 (SponLMTTS)
> - **不污染** ✓ 未新建概念页, 未执行反向更新
> - Low issues: datasets 字段空 (无实体页可链接); 2 处 traceability gap (骨干网络段/局限性第4点)
> - 详见 `_review/CodecLMSpontaneousStyle-review.yml`

---
检索命中: [[LLM-basedTTS]], [[SemanticvsAcousticTokens]], [[Zero-shotSpeechSynthesis]], [[CodecLanguageModel]], [[Classifier-FreeGuidance]], [[VoiceCloningTaxonomy]] | 过滤: [[CodecLanguageModel]](pending-review), [[Classifier-FreeGuidance]](pending-review), [[VoiceCloningTaxonomy]](pending-review) | 未命中但可能相关: [[ProsodyModeling]]
