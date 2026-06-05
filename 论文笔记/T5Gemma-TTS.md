---
type: paper
tier: deep
title: "T5Gemma-TTS Technical Report"
arxiv_id: "2604.01760"
source: "Sources/T5Gemma-TTS.pdf"
authors: [Chihiro Arata, Kiyoshi Kurihara]
year: 2026
venue: "arXiv"
tags: [TTS, encoder-decoder, zero-shot, voice-cloning, multilingual, codec-LM, duration-control, PM-RoPE, XCodec2, subword-tokenization]
concepts: ["[[CodecLanguageModel]]", "[[LLM-basedTTS]]", "[[Single-codebookvsMulti-codebook]]", "[[DurationPredictor]]", "[[VoiceCloningTaxonomy]]", "[[PhonemeRepresentation]]"]
models: ["[[CosyVoice2]]", "[[XTTS]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]"]
datasets: ["[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[LLM-basedTTS]], [[CodecLanguageModel]], [[Single-codebookvsMulti-codebook]], [[DurationPredictor]], [[VoiceCloningTaxonomy]], [[PhonemeRepresentation]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-basedTTS]]✓, [[CodecLanguageModel]][待确认], [[Single-codebookvsMulti-codebook]][待确认], [[DurationPredictor]][待确认], [[VoiceCloningTaxonomy]][待确认], [[PhonemeRepresentation]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: T5Gemma-TTS 属于 [[CodecLanguageModel]] 范式中的一个非典型变体。绝大多数 codec LM TTS 系统(VALL-E、Llasa、Seed-TTS 等)采用 decoder-only 架构,将文本 token 作为 prefix 拼接到 audio token 序列前。T5Gemma-TTS 回到 encoder-decoder 架构,通过 cross-attention 注入文本表征,这是一条相对小众但在传统 TTS(Tacotron/FastSpeech)中非常成熟的路线。这一选择直接回应了 decoder-only 架构的"文本 dilution"问题——当音频序列远长于文本序列时,文本 conditioning 逐渐被稀释。

**XCodec2 与单码本趋势**: 本文使用的 XCodec2 是 [[Single-codebookvsMulti-codebook]] 页面中记录的单码本方案代表(50Hz, 65536 vocab)。与 Llasa 的区别在于骨干架构:Llasa 用 LLaMA decoder-only,T5Gemma-TTS 用 T5Gemma encoder-decoder。

**Duration control 定位**: [[DurationPredictor]] 页面记录了从 FastSpeech 到 VoiceStar PM-RoPE 的 duration control 演进。PM-RoPE 不是传统意义上的 duration predictor(不预测每个 phoneme 的帧数),而是在 cross-attention 中注入归一化进度信号,让 decoder "知道"自己生成到了目标语音的哪个位置。T5Gemma-TTS 直接复用 VoiceStar 的 PM-RoPE 设计,贡献在于验证其在预训练多语言骨干上的泛化性。

**Phoneme vs subword 选择**: [[PhonemeRepresentation]] 页面指出 LLM-TTS 时代正从 phoneme 输入向 BPE/subword 迁移。T5Gemma-TTS 直接使用 T5Gemma 的 SentencePiece 256K subword vocabulary,避免了多语言 phonemizer 的工程成本,但牺牲了 phoneme 天然的单调对齐特性(PM-RoPE 在 phoneme 序列上效果最佳,在 subword 上是否同样有效尚未消融)。

**创新判断**: 本文的创新不在于提出新技术(PM-RoPE 来自 VoiceStar,XCodec2 来自 Llasa,T5Gemma 是现有预训练模型),而在于将三者组合并在大规模多语言数据上验证——证明 encoder-decoder + PM-RoPE 在非英语语言上同样有效,且预训练骨干的多语言知识能实现对未见语言(韩语)的 zero-shot 泛化。

## 速查

> [!summary] 速查
> - **一句话**: 用 T5Gemma 预训练 encoder-decoder 骨干 + XCodec2 单码本 + PM-RoPE 进度感知 cross-attention,实现持久文本 conditioning 和可控时长的多语言 zero-shot TTS
> - **路线**: 文本→T5Gemma encoder(26层, bidirectional)→cross-attention(PM-RoPE)→T5Gemma decoder(26层, AR)→XCodec2 tokens→waveform
> - **指标**: 日语 CER 0.126 / SIM 0.677(vs XTTS v2 0.622, CI不重叠) [Table 2]; 韩语(未见语言) SIM 0.747 / CER 0.082 [Table 2]; PM-RoPE 关闭→CER 0.129→0.982 [Table 3]
> - **可借鉴**: PM-RoPE 作为进度信号注入 cross-attention 的方法可迁移到任何 encoder-decoder 序列生成任务;subword 输入替代 phoneme 消除多语言前端工程
> - **局限**: RTF 0.8-2.0 远慢于 NAR 方法;UTMOS 低于 flow-matching/diffusion 后端;PM-RoPE 对 subword 输入(vs phoneme)的影响未做消融;英语评估有 LibriHeavy/LibriSpeech 数据泄漏

## 核心问题

1. **Decoder-only codec LM 的文本 conditioning 怎么就弱了?** 在 decoder-only 架构中,文本 token 序列和音频 token 序列拼接后做统一 causal self-attention。当音频序列长度 S >> 文本长度 T 时,文本 token 在每个 attention window 中占比越来越小,conditioning 信号被稀释。这在长文本/有声书合成中尤其严重 [§1]。
2. **AR 生成中 duration 为什么难控制?** 自回归模型没有显式的位置锚点来感知"当前生成进度",不知道自己已经说到哪了,导致时长匹配不稳定 [§1]。
3. **多语言零样本 TTS 中,phoneme 前端是否必要?** 每种语言需要专门的 phonemizer,维护成本高,而预训练 LLM 的 subword vocabulary 天然覆盖多语言 [§1, §2.2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

T5Gemma-TTS 是一个 encoder-decoder autoregressive codec language model [§3.1]:

```
Text → T5Gemma Encoder (26层, bidirectional) → H_enc ∈ R^{T×d}
           ↓ (cross-attention, 每层都有)
Reference audio → XCodec2 → audio tokens → T5Gemma Decoder (26层, causal AR) → next audio token
           ↑
    PM-RoPE 进度信号注入
```

**参数规模**: 2B encoder + 2B decoder = 4B 总参数,基于 T5Gemma-2b-2b checkpoint 初始化 [§3.1]。

**为什么选 encoder-decoder 而非 decoder-only?** [论文原文] Encoder 产生固定大小的表征矩阵 H_enc ∈ R^{T×d},通过独立的 cross-attention 路径注入每一层 decoder,这条路径与 causal audio self-attention 解耦。因此无论音频序列多长,文本 context 始终完整保留 [§1]。[agent 解读] 这是 attention bottleneck 角度的论证——cross-attention 的 key-value 来自固定的 encoder 输出,不会被 audio token 稀释,而 decoder-only 的 self-attention 中 text token 必须与所有 audio token 竞争注意力资源。

### 关键设计选择

**1. XCodec2 作为 audio tokenizer** [§3.1]

选择理由(三点,均为论文原文 [§3.1]):
- 单码本设计:避免 RVQ 多码本的 interleaved prediction,生成任务退化为标准 next-token prediction
- 50Hz 帧率:低于 DAC 的 86Hz,减少序列长度和计算量
- 65536 码本大小:远大于 EnCodec 单码本的 1024,提供更精细的声学量化而无需残差码本

[agent 解读] 本文使用的是日语 fine-tuned 版 XCodec2(decoder 在日语语音上微调,encoder 与原版相同)。这可能轻微不利于非日语的重建质量,但评估指标(CER/WER/SIM)作用于生成波形,因此影响跨所有语言一致 [§3.1]。

**2. PM-RoPE (Progress-Monitoring Rotary Position Embedding)** [§3.2]

直接复用 VoiceStar [8] 的设计,无修改。核心机制:

对 decoder 位置 j (共 S 步) 和 encoder 位置 i (共 T 个 text token),定义进度 position ID:
- p_dec_j = j/(S-1) × s (decoder 进度)
- p_enc_i = i/(T-1) × s (encoder 位置)
- s = 2000 为固定缩放常数

两套独立 RoPE 模块分别作用于 cross-attention 的 query(decoder 侧)和 key(encoder 侧),使 attention score 编码 decoder 当前进度与 encoder 位置的对齐关系 [§3.2, Eq. 1-3]。

[论文原文] PM-RoPE 使模型能 "attend to text tokens proportional to its generation stage",即根据生成阶段调整对不同文本位置的注意力 [§3.2]。

[agent 解读] PM-RoPE 本质上给 cross-attention 加了一个软单调对齐偏置——decoder 在生成早期更关注文本开头,后期更关注文本结尾。这类似于 location-sensitive attention 的思路,但通过 RoPE 的相对位置编码实现,更优雅且无需额外参数。

**3. 直接 subword 输入(无 phoneme 转换)** [§2.2]

与 VoiceStar 使用 phoneme 不同,T5Gemma-TTS 直接使用 T5Gemma 的 SentencePiece tokenizer(256K subword vocabulary),牺牲 phoneme 的单调对齐特性,换取两个实际好处 [§2.2]:
- 避免每种语言的 phonemizer 工程成本
- 保留预训练 embedding 权重中的多语言语义信息

[论文原文] phoneme vs subword 选择对 PM-RoPE duration control 效果的影响 **未做消融**,是 open question [§2.2]。

**4. Duration estimation** [§3.3]

推理时用简单的 phoneme-count 比例估计目标时长:
D̂ = (D_ref / N_ref) × N_tgt

其中 D_ref 为参考音频时长,N_ref/N_tgt 为参考/目标文本的 phoneme 数。不同语言用不同 phoneme 计数工具(espeak-ng for EN, pyopenjtalk for JA, Unicode 字符数 for ZH)。未见语言 fallback 到英语估计器 [§3.3]。

[论文原文] 这是 "convenience heuristic",非高精度 predictor;估计误差可能导致 duration 相关质量下降 [§3.3]。

### 训练策略

- **数据**: ~170K 小时多语言语音(EN ~100K h from LibriHeavy, ZH ~50K h from Emilia, JA ~20K h from Emilia+其他) [§3.4]
- **损失**: 标准 next-token cross-entropy over audio tokens [§3.4, Eq. 5]
- **优化器**: AdamW, peak lr=1e-4, weight decay=0.01, gradient clipping to unit norm [§3.4]
- **Schedule**: 2% linear warmup (~2900 steps) + linear decay to 0, 共 ~143K steps [§3.4]
- **Batch**: 动态 batching,每 GPU ≤ 30K tokens,8 GPU 有效 ~240K tokens/update [§3.4]
- **硬件**: 8× AMD MI300X, 约 2 周 [§3.4]
- **精度**: bfloat16 mixed precision,float32 master weights [§3.4]

## 实验

| 指标 | 本文 (T5Gemma-TTS) | XTTS v2 | F5-TTS | CosyVoice 2 | Kokoro | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| JA CER↓ | **0.126±0.018** | 0.177±0.031 | 1.138±0.110 | 0.213±0.032 | 0.139±0.016 | JSUT | [Table 2] |
| JA SIM↑ | **0.677±0.016** | 0.622±0.017 | 0.642±0.020 | 0.497±0.019 | 0.185±0.011 | JSUT | [Table 2] |
| ZH CER↓ | 0.131±0.039 | 0.126±0.028 | 0.155±0.032 | 0.076±0.032 | **0.071±0.030** | AISHELL-1 | [Table 2] |
| ZH SIM↑ | 0.722±0.017 | 0.623±0.014 | **0.864±0.015** | 0.612±0.018 | 0.279±0.021 | AISHELL-1 | [Table 2] |
| EN WER↓ | 0.128±0.060‡ | **0.052±0.026** | 1.069±0.058 | 0.193±0.033 | 0.077±0.064 | LibriSpeech‡ | [Table 2] |
| KO CER↓ (unseen) | 0.082±0.026 | **0.046±0.012** | 0.934±0.022 | 0.090±0.032 | 1.514±0.101 | FLEURS | [Table 2] |
| KO SIM↑ (unseen) | **0.747±0.029** | 0.741±0.010 | 0.589±0.069 | 0.607±0.023 | 0.071±0.012 | FLEURS | [Table 2] |
| DA (PM-RoPE on) | **0.79±0.11** | — | — | — | — | JSUT 50 utt | [Table 3] |
| DA (PM-RoPE off) | 0.46±0.13 | — | — | — | — | JSUT 50 utt | [Table 3] |
| CER (PM-RoPE off) | 0.982±0.037 | — | — | — | — | JSUT 50 utt | [Table 3] |

‡ 英语结果为上界估计,LibriHeavy 是 LibriSpeech 超集,存在训练/测试重叠 [§4.1]。
† Kokoro 非 zero-shot voice cloning 系统(使用固定 preset voices,不做 speaker adaptation),SIM 值不反映 voice cloning 能力,仅作 intelligibility/naturalness 参考 [§4.1]。

**关键发现**:

1. **日语 SIM 统计显著**: T5Gemma-TTS (0.677) vs XTTS v2 (0.622),95% CI 不重叠,是本文最强的定量结论 [§5.1]
2. **韩语跨语言泛化**: 韩语完全不在训练集中,但 SIM 0.747(数值最高)、CER 0.082(具竞争力)[§5.2]。作者推测原因是韩语与日语在音韵特征上接近(agglutinative morphology, verb-final syntax),加上 SentencePiece 词表中有 2388 个含韩文 Hangul 的 token [§5.2]
3. **PM-RoPE 不可或缺**: 关闭 PM-RoPE 导致几乎完全的合成失败(CER 0.129→0.982, SIM 0.666→0.109)[§5.3]。但需注意这不是控制实验(模型是带 PM-RoPE 训练的,关闭它测试的是 inference-time 影响,不是训练贡献)[§4.2]
4. **Naturalness 短板**: UTMOS 系统性低于 Kokoro 和 CosyVoice 2,作者归因于 XCodec2 的量化天花板和缺乏 diffusion/flow-matching 精细化阶段 [§5.5]

## 局限性

1. **推理速度**: RTF 0.8-2.0,远慢于 F5-TTS (~0.15) 和 Kokoro (~0.05),作者定位为 offline/batch 场景(有声书等)[§6]
2. **Naturalness gap**: 无 diffusion/flow-matching 后处理,UTMOS 不及 CosyVoice 2 和 Kokoro [§5.5]
3. **欧洲语言泛化弱**: 法语 WER 0.475、德语 WER 0.453,远差于 XTTS v2(0.08/0.06),显示泛化局限于音韵类似语言 [Table 2]
4. **PM-RoPE 消融不充分**: flag-switch 实验而非 from-scratch 控制实验;subword vs phoneme 对 PM-RoPE 效果的影响未探索 [§4.2, §2.2]
5. **英语评估有数据泄漏**: LibriHeavy ⊃ LibriSpeech test-clean [§4.1]
6. **UTMOS 的跨语言有效性存疑**: UTMOS22 仅在英语上训练和验证,非英语 UTMOS 分数仅作近似参考 [§4.1]
7. **Duration estimation 粗糙**: 基于 phoneme 计数的简单比例估计,对未见语言用英语 fallback,估计误差未定量分析 [§3.3]

## 点评

**优势**:
- 清晰地指出 decoder-only 的文本 dilution 问题并给出结构性解决方案(encoder-decoder + cross-attention),比 VALL-E 系列的 prefix 方案更优雅
- PM-RoPE 的多语言验证有实际价值——证明这种 duration control 机制不是 English-specific 的
- 统计处理审慎:提供 bootstrap 95% CI,明确标注重叠/不重叠,主动声明英语数据泄漏,这在 TTS 论文中难得

**值得质疑**:
- 核心技术均为组合已有工作(PM-RoPE from VoiceStar, XCodec2 from Llasa, T5Gemma backbone),系统级创新而非方法级创新
- PM-RoPE "ablation" 用同一 checkpoint 开关 flag,无法区分 PM-RoPE 的 training-time 贡献和 inference-time 必要性,结论力度有限
- 日语数据仅 ~20K h 但结果最好,中文 ~50K h 但 CER 不如 XTTS v2(0.131 vs 0.126),可能存在数据质量/数据泄漏等未讨论的混淆因素
- F5-TTS 在日语上 CER > 1.0(完全失败)但在中文上 SIM 最高——这个极端差异的 baseline 行为未被讨论,影响对比公平性

## 可复用的 idea

1. **Encoder-decoder 解决长序列 text conditioning 退化**: 任何 decoder-only seq2seq 系统中,如果 conditioning 序列远短于生成序列,cross-attention 注入是更鲁棒的方案。可迁移到音频编辑、长文本配音等场景
2. **PM-RoPE 作为通用进度信号**: 在 cross-attention 中注入归一化 progress 的思路可用于任何需要控制输出长度的生成任务(如 code generation 控制长度、music generation 控制段落)
3. **Subword 替代 phoneme 降低多语言门槛**: 直接复用 LLM tokenizer 的 subword vocabulary 可省去 G2P 工程,对快速扩展语言覆盖有实际价值
4. **统计报告范式**: bootstrap CI + 明确标注数据泄漏风险 + 区分 statistically significant vs numerically highest,值得其他 TTS 评估论文学习

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节 WHY/HOW 清晰,设计选择有对比论证 |
> | 可信赖 | pass-with-fixes | 数字验证正确,出处覆盖>90%;表格 bold 和 Kokoro 说明已修正 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注一致,无断言式推断 |
> | 可定位 | pass | KB 背景 5 维度谱系定位,创新判断平衡 |
> | 不污染 | pass | 反向更新均为 append,无 factual error 风险 |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> 详见 `_review/T5Gemma-TTS-review.yml`
