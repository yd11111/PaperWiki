---
type: paper
tier: deep
title: "Vevo2: A Unified and Controllable Framework for Speech and Singing Voice Generation"
arxiv_id: "2508.16332"
source: "Sources/Vevo2.pdf"
authors: [Xueyao Zhang, Junan Zhang, Yuancheng Wang, Chaoren Wang, Yuanzhe Chen, Dongya Jia, Zhuo Chen, Zhizheng Wu]
year: 2025
venue: "arXiv"
tags: [TTS, SVS, voice-conversion, singing-voice, speech-editing, unified-generation, chromagram, post-training, GRPO, zero-shot, controllability]
concepts: ["[[SpeechTokenizer]]", "[[ConditionalFlowMatching]]", "[[LLM-basedTTS]]", "[[SemanticvsAcousticTokens]]", "[[SpeechFactorization]]", "[[SingingVoiceSynthesis]]"]
models: ["[[MaskGCT]]", "[[TechSinger]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: SpeechTokenizer, ConditionalFlowMatching, LLM-basedTTS, SemanticvsAcousticTokens, SpeechFactorization, SingingVoiceSynthesis)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: SpeechTokenizer (VQ-VAE/semantic/acoustic tokenizer 谱系), ConditionalFlowMatching (OT-CFM 用于 mel 生成), LLM-basedTTS (AR+FM two-stage), SemanticvsAcousticTokens (content-style 属于 mixed token 路线), SpeechFactorization (timbre disentanglement 方法), SingingVoiceSynthesis (SVS 任务与级联/E2E 架构)
> 过滤: MaskedGenerativeModeling (Vevo2 不使用 masked generative,用 AR 建模)
> 未命中但可能相关: DifferentiableRewardOptimization (GRPO 相关), ProsodyModeling (chromagram-based prosody)
>
> **定位**: Vevo2 在 KB 中的位置:
> - **SpeechTokenizer**: Vevo2 提出两个定制 VQ-VAE tokenizer (prosody 6.25Hz + content-style 12.5Hz),重建目标不是 SSL hidden states (MaskGCT 路线) 或 ASR encoder (CosyVoice 路线),而是 chromagram + Whisper features 的混合目标。这是第三条重建目标路线。
> - **SemanticvsAcousticTokens**: content-style tokenizer 同时编码 linguistic content (via Whisper) 和 melody/style (via chromagram),属于 mixed token 路线但与 SpeechTokenizer/Mimi 不同 — 不是在 RVQ 层级上分离 semantic/acoustic,而是用混合重建目标让单码本同时编码两类信息。
> - **ConditionalFlowMatching**: Vevo2 的 FM stage 遵循 Vevo (ICLR 2025) 的设计,使用 DiT backbone (16 layers, 363M params),加入 REPA 策略对齐 w2v-BERT 2.0 features 加速训练。
> - **SpeechFactorization**: Vevo2 实现 timbre disentanglement 的方式是架构层面分离 — AR stage 用 content-style tokens (已排除 timbre) + style reference prompt,FM stage 用 timbre reference prompt 控制音色。不使用对抗训练或 self-distillation。
> - **SingingVoiceSynthesis**: Vevo2 打破了 SVS 必须依赖乐谱标注 (MIDI/music notation) 的惯例,用 chromagram prosody tokenizer 从任意音频 (包括器乐) 提取旋律信息,实现 notation-free SVS。

## 速查

> [!summary] 速查
> - **一句话**: 通过基于 chromagram 的统一 prosody tokenizer 和显式/隐式韵律学习策略,将 speech 和 singing voice 生成统一到单一 AR+FM 两阶段框架中,并用 GRPO 多目标后训练同时优化可懂度和旋律跟随能力
> - **路线**: Text + Prosody source (optional) + Style ref → [AR Transformer 509M, Qwen2.5-0.5B init] → content-style tokens (12.5Hz) + Timbre ref → [FM Transformer 363M] → Mel spectrogram → [Vocos 255M] → Waveform
> - **指标**: Zero-shot TTS: WER 11.48% / SIM 0.689 (expressive speech, post-trained); SVS: WER 9.83% / FPC 0.710 (Chinese, post-trained); SVC: WER 11.64% / SIM 0.601 (English); Speech editing: WER 16.83% / SIM 0.799; 训练数据 101K h speech (Emilia) + 7K h singing (SingNet)
> - **可借鉴**: (1) Chromagram VQ-VAE 作为 notation-free prosody 表示 — 八度无关+噪声鲁棒+跨音频类型通用; (2) EPL/IPL 随机切换策略统一 speech-singing 训练; (3) GRPO 双目标后训练避免单目标过优化导致的能力退化; (4) Fixed frame rate tokenizer 的 2:1 长度关系实现 AR 模型的 duration control
> - **局限**: 12.5Hz content-style token rate 导致 FM-only SVC 可懂度弱于 50Hz 方案; 后训练依赖 M4Singer 的 MIDI 标注渲染器乐; singing voice 数据仅 7K h (vs speech 101K h) 比例悬殊; 未在 streaming/实时场景评估

## 核心问题

统一 speech 和 singing voice 生成面临两个根本挑战 [§I]:

1. **数据不兼容**: 现有 SVS 数据集依赖专家标注的乐谱 (MIDI + 歌词-音符精确对齐) [§I],这类数据稀缺且组织形式与大规模语音数据完全不一致,限制了统一训练的可扩展性。
2. **控制维度不统一**: Speech 和 singing 对控制信号的需求差异巨大 — speech 通常只需 text 输入,prosody 由模型隐式推断;而 singing 需要对 melody 的显式控制 [§III-B],这使得在单一框架中实现 text/prosody/style/timbre 的灵活组合非常困难。

[agent 解读] 此前的统一方案 (UniSyn, UniAudio) 通过 task identifier 切换模式,本质上仍是多任务学习而非真正统一 — 两个域的生成特性 (speech 隐式韵律 vs singing 显式旋律) 没有被桥接。Vevo2 试图通过统一的表示和训练策略从根本上解决这个问题。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Vevo2 采用两阶段管线 [Fig 1]:

**Stage 1 — AR Content-Style Modeling**: AR Transformer (509M, Qwen2.5-0.5B 初始化 [§IV-B]) 以 text tokens + 可选 prosody tokens + style reference 的 content-style tokens 为输入,自回归预测目标 content-style tokens。通过 EPL/IPL 策略统一 speech 和 singing 的生成模式。

**Stage 2 — FM Acoustic Modeling**: FM Transformer (363M, 16 layers [§IV-B]) 将 AR 生成的 content-style tokens 转为 Mel spectrogram,以 timbre reference 的 Mel 和 content-style tokens 作为条件。

**Vocoder**: Vocos-based vocoder (255M [§IV-B]),在 speech + singing 数据上 fine-tune,将 Mel spectrogram 转为波形。

### 关键设计选择

#### 1. 统一 Prosody Tokenizer [§III-A]

**设计**: VQ-VAE 以 chromagram features 为重建目标,6.25 Hz (56.25 bps), codebook size 512, 38M params [§IV-B]。

**为什么选择 chromagram 而非 F0**: 论文给出三个理由 [§III-A]:
1. **Octave-free**: F0 在 speech 和 singing 之间分布差异巨大 (singing 音域更宽 [§III-A]),chromagram 去除了八度信息,使得跨域的韵律模式更容易统一建模。
2. **Notation-free**: chromagram 可从原始音频自动提取,无需乐谱标注,适合大规模数据处理。
3. **跨音频类型鲁棒**: chromagram 作为频谱能量的统计分布,对噪声和 pitch 检测误差不敏感,且对人声和器乐声都有良好表示能力 — 这直接支撑了 humming-to-singing 和 instrument-to-singing 等应用。

**VQ 量化的必要性**: [论文原文] 连续 chromagram 虽然可以直接用作条件信号,但 VQ 离散化 (1) 更适合序列建模管线,(2) 充当信息瓶颈压缩旋律/韵律的关键线索,同时抑制残留的声学特征 (如 timbre) [Appendix A]。可视化实验 [Fig 6, 7] 显示: Mel spectrogram 按 instrument type 聚类 (timbre 依赖强); 连续 chromagram 已显著缓解但仍有残留; VQ prosody tokens 在 timbre invariance 和 melody clusterability 上均最优。

#### 2. 统一 Content-Style Tokenizer [§III-A]

**设计**: 与 prosody tokenizer 类似的 VQ-VAE 架构,但操作在 12.5 Hz (175 bps), codebook size 16384, 44M params [§IV-B]。

**双重建目标**: 同时重建 chromagram features 和 Whisper-medium encoder features (均为 50Hz, 4x downsample) [§III-A]。

[论文原文] Whisper features 编码 linguistic content 和 style,但对 singing voice 的 melody 建模能力有限 [§III-A, citing [53]],因此额外加入 chromagram 重建目标补偿。这使得 content-style tokens 同时编码 linguistic content、melody 和 style,同时实现 robust timbre disentanglement [Table III]。

[agent 解读] 这个双重建目标设计在 KB 的 mixed token 路线中是独特的 — SpeechTokenizer 和 Mimi 在 RVQ 层级上分离 semantic/acoustic,而 Vevo2 在重建目标层面混合 semantic (Whisper) 和 prosodic (chromagram) 信息。代价是单码本需要更大的 vocabulary (16384 vs MaskGCT 的 8192) 来容纳两类信息。

#### 3. Speech-Singing Joint Training: EPL/IPL [§III-B]

**核心问题**: Speech 生成 (TTS) 通常只需 text 输入,prosody 由 in-context learning 隐式推断;而 singing 需要 melody 的显式控制 [§III-B]。如何让单一模型同时支持两种生成模式?

**方案**: 设计两种训练策略,对每个训练样本 (无论 speech 还是 singing) 以 50%/50% 概率随机选择:

- **IPL (Implicit Prosody Learning)** [Eq 3]: 输入序列 `[I_ipl, T, <start-cs>, Q_cs, <end-cs>]`,仅从 text 推断 prosody,等价于标准 zero-shot TTS 训练。
- **EPL (Explicit Prosody Learning)** [Eq 4]: 输入序列 `[I_epl, T, <start-p>, Q_p, <end-p>, <start-cs>, Q_cs, <end-cs>]`,显式提供 prosody tokens 作为条件。

两种策略都只对 content-style token 段做 next-token prediction [§III-B]。

[论文原文] 关键设计: 不限制 IPL 用于 speech、EPL 用于 singing,而是对所有样本 (speech + singing) 等概率随机切换。这假设统一的训练模式比域特异的模式更好地桥接 speech 和 singing [§III-B]。

[agent 解读] 这个设计隐含一个重要假设: singing 样本在 IPL 模式下学习 "从 text 推断 melody" 的能力 (类似 TTS 的韵律推断),而 speech 样本在 EPL 模式下学习 "根据 prosody tokens 精确控制韵律"。两个域通过共享训练目标相互增强。实验 [Table I] 验证了这个假设: 联合训练在两个域都优于单域训练。

#### 4. Multi-Objective Post-Training (GRPO) [§III-C]

**动机**: 预训练后模型的 text-following 和 prosody-following 稳定性仍不够理想,尤其在面对分布外数据 (如 MIDI 渲染的器乐声) 时 [§III-C]。

**双 reward 设计**:
- **Intelligibility reward** [§III-C, Eq 5]: 用 INTP 数据集 (250K preference speech pairs [§IV-A]) 训练 Bradley-Terry reward model,从预训练模型初始化。输入 [T, Q_cs] 或 [T, Q_p, Q_cs] 序列输出可懂度 reward。
- **Prosody similarity reward** [§III-C]: 给定 singing sample u 的 MIDI-rendered instrumental music 提取 prosody tokens → 模型以 text + prosody tokens 为条件生成 content-style tokens → 用 content-style tokenizer decoder 重建 chromagram → 与 ground-truth chromagram 计算 cosine similarity。

**GRPO 优化** [Eq 6]: 对每个 prompt 生成 K 个 completions,分别计算 r_int 和 r_pro,各自做 group normalization 后相加作为 advantage。

**后训练数据组成** [§IV-B]: (1) 20K speech samples from INTP, (2) 20K singing samples from M4Singer, (3) M4Singer MIDI 渲染的 16 种乐器 (钢琴/弦乐/木管/铜管/民族乐器)。每个 iteration 随机选择一种乐器渲染。

[agent 解读] 后训练的真正价值不只是改善 reward 指标,而是解决训练-推理分布不匹配: 预训练时 prosody tokens 只来自人声,推理时可能来自器乐。GRPO 通过将器乐数据引入训练循环,有效弥合了这个 gap。消融实验 [Fig 5] 证明: 仅用 intelligibility reward 时 melody accuracy 从 65% 降到 50% (单目标过优化);双目标联合训练使 text accuracy 从 75% 升到 90%,melody accuracy 也上升。

### 训练策略

**Tokenizer 训练** [§IV-B]: 两个 tokenizer 在 4x A100 上训练 300K steps (~48h), lr=1e-4。

**AR 模型** [§IV-B]: 从 Qwen2.5-0.5B 初始化,扩展 vocabulary 加入 prosody 和 content-style tokens,总参数 509M。使用 CosyVoice 2 的 BPE tokenizer 策略 (mask one-to-many tokens 防止过长发音)。8x A100, 500K steps (~168h), lr=5e-4。

**FM 模型** [§IV-B]: 采用 Vevo 的 FM transformer 架构,16 layers, 16 heads, dim 1024, 363M params。引入 REPA 策略 (第 5 层 hidden → 对齐 w2v-BERT 2.0 features) 加速收敛。8x A100, 700K steps (~168h), lr=7.5e-5。

**Vocoder** [§IV-B]: 从 Vevo 的 speech vocoder fine-tune,Vocos 架构 255M params。4x A100, 572K steps (~72h)。

**Post-training** [§IV-B]: lr=5e-6, batch size 3, rollout K=8, KL coefficient 0.1。8x A100, 9K steps (~48h)。

## 实验

| 指标 | Vevo2 | Vevo2-base | MaskGCT | CosyVoice 2 | F5-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER (TTS, expressive speech) | **11.48** | 14.32 | 13.42 | 11.20 | 11.77 | Genshin-Voice | Table I |
| SIM (TTS, expressive speech) | 0.689 | 0.681 | **0.736** | 0.706 | 0.695 | Genshin-Voice | Table I |
| WER (TTS, singing voice) | **7.66** | 15.78 | 11.71 | 16.18 | 16.12 | GTSinger/SingStyle111 | Table I |
| N-CMOS (TTS, singing voice) | **0.00** | -0.28 | -1.74 | -1.68 | -1.89 | GTSinger/SingStyle111 | Table I |
| WER (SVS, Chinese) | 9.83 | 23.60 | - | - | - | GTSinger | Table II |
| FPC (SVS, Chinese) | 0.710 | 0.674 | - | - | - | GTSinger | Table II |
| WER (speech editing) | **16.83** | 23.54 | - | - | 21.35 | Genshin-Voice | Table II |
| SIM (speech editing) | **0.799** | 0.795 | - | - | 0.733 | Genshin-Voice | Table II |
| WER (SVC, English, full) | **11.64** | 29.82 | - | - | - | GTSinger/SingStyle111 | Table III |
| SIM (SVC, English, full) | **0.601** | 0.587 | - | - | - | GTSinger/SingStyle111 | Table III |
| DC (duration control) | 97.1-98.4% | - | - | - | - | SE/SLE/SVC | Table V |

### 关键发现

**统一建模的互利性** [§V-A, Table I]: Speech+singing 联合训练相比 speech-only: expressive speech WER 15.52→14.32; singing WER 19.39→15.78, N-CMOS -1.02→-0.28。singing voice 的提升更为显著,说明 singing 从 speech 数据量的补充中获益最大。

**后训练的有效性** [§V-A, Table I]: Post-training 在所有指标上进一步提升 — WER 14.32→11.48 (speech), 15.78→7.66 (singing)。singing 域的 WER 提升尤其惊人 (降低 51.4%),这归因于 MIDI-rendered 器乐数据弥合了 prosody token 的分布 gap。

**12.5Hz 低帧率的 trade-off** [§V-C, Table III]: Vevo2-FM (仅 FM stage, 12.5Hz content-style tokens) 在 SVC 的 WER 上 (29.82) 弱于 Vevo-FM (50Hz, 24.05) 和 CosyVoice2-FM (25Hz, 23.59)。但加入 AR stage 后 WER 大幅改善 (11.64),说明 text input + intelligibility alignment 有效弥补了低帧率的信息损失。

**Duration control** [§V-D, Table V]: 利用 prosody tokenizer (6.25Hz) 和 content-style tokenizer (12.5Hz) 的固定 2:1 帧率关系,AR 模型学会了这个长度模式。通过对 prosody source 的 chromagram 做线性缩放即可控制生成时长。DC 达 97%+,证明即使 AR 模型也能实现有效 duration control。

**Pitch region control** [§V-E, Table VI]: 对 source waveform 做 F0 shift 后提取 chromagram / content-style tokens,可实现 pitch region 控制。SIM 提升显著 (SVC English: 0.538→0.587 for FM-only, 0.556→0.601 for full),但 WER 略有上升。

## 局限性

1. **低帧率 content-style tokenizer 的信息瓶颈** [Table III]: 12.5Hz 的帧率在纯 FM 模式 (VC/SVC) 下可懂度明显弱于 25-50Hz 方案,必须依赖 AR stage 的 text 输入补偿。这限制了 Vevo2 在 style-preserved VC (不需要 text 的纯 FM 任务) 中的竞争力。

2. **后训练数据的局限**: GRPO 的 prosody reward 依赖 M4Singer 数据集的 MIDI 标注来渲染器乐,这本身就是一种标注依赖。如果要扩展到更多旋律风格,仍需 MIDI 标注。[agent 解读]

3. **数据比例失衡**: 101K h speech vs 7K h singing (~14:1) 的数据比例意味着 singing 域的学习仍然受限。论文未探讨更大规模 singing 数据的影响。[agent 解读]

4. **Zero-shot TTS 指标尚有差距**: 在 regular speech (SeedTTS test sets, Table VII) 上,WER 3.639 (test-en) vs CosyVoice 2 的 2.890,SIM 0.693 vs MaskGCT 的 0.711。Vevo2 的优势主要体现在 expressive speech 和 singing voice 域,而非 regular speech。

5. **Melody-MOS trade-off** [Table III]: Full Vevo2 的 Melody-MOS (2.24-2.38) 低于 FM-only 模式 (2.90-2.91) 和 SeedVC (2.89-2.93),说明 AR stage 引入的 text-following 在一定程度上"覆盖"了 melody-following,两者存在竞争关系。

## 点评

**优势明确且可验证**: Vevo2 的核心贡献 — chromagram-based prosody tokenizer — 解决了一个真实痛点: SVS 对乐谱标注的依赖。这个设计的验证非常扎实: 可视化 [Fig 6-7] 直观证明 timbre invariance 和 melody clusterability; 消融 [Table I] 证明互利性; 后训练消融 [Fig 5] 证明双目标优于单目标。

**架构延续性强**: Vevo2 建立在 Vevo (ICLR 2025) 基础上,AR stage 用 Qwen2.5-0.5B 初始化 (LLM backbone 路线),FM stage 沿用 Vevo 的 DiT。这种增量创新模式使得贡献点清晰: tokenizer 设计 + EPL/IPL 训练策略 + GRPO 后训练。

**实验覆盖面广**: 覆盖 10+ 种任务 (TTS, SVS, VC, SVC, editing, style conversion, humming/instrument-to-singing, duration/pitch control),在每个任务上都有可比基线和定量评估。这在 speech generation 领域较为少见,通常系统只在 2-3 个任务上评估。

**不足**: (1) 与 KB 中已有系统的对比不够直接 — MaskGCT 在 SIM 指标上仍领先,CosyVoice 2 在 regular speech WER 上更好,Vevo2 的优势主要在 singing 域和 expressiveness,但这些域的评估较 niche。(2) 实际应用价值存疑: 真实场景中"统一 speech 和 singing"的需求有多大?大多数产品要么需要高质量 TTS,要么需要高质量 SVS,统一框架在两者上都不是最优的风险较大。(3) 后训练依赖 MIDI 渲染器乐的设计限制了泛化。

## 可复用的 idea

1. **Chromagram VQ-VAE 作为 notation-free melody 表示**: 对任何需要 melody/prosody 条件控制的系统 (SVS, speech editing, voice conversion) 都有价值。VQ 量化提供的信息瓶颈效果 [Appendix A, Fig 6-7] 尤其值得关注 — 从连续 chromagram 到 VQ tokens,timbre invariance 显著提升。可以直接复用到基于 F0 的 prosody 控制方案中作为替代。

2. **EPL/IPL 随机切换训练**: 适用于任何需要同时支持"有条件"和"无条件"控制的生成模型。核心思想是不按数据域划分训练策略,而是对所有样本随机应用两种模式。这比 classifier-free guidance 的 unconditional dropout 更精细 — CFG 只是随机丢弃条件,EPL/IPL 是两种不同的输入格式。

3. **Fixed frame rate 的 duration control**: 利用两个 tokenizer 的固定帧率比 (2:1) 让 AR 模型隐式学习长度关系,推理时通过 chromagram 线性缩放控制时长。这个 trick 对其他使用固定帧率 tokenizer 的 AR-based TTS 系统也适用,且几乎零成本。

4. **GRPO 双目标后训练的单目标过优化警示**: Fig 5 清楚展示仅优化 intelligibility 导致 melody accuracy 下降 (65%→50%)。对所有使用 RLHF/GRPO 后训练的语音生成系统,这是一个重要的经验: 必须同时优化多个可能竞争的目标,否则会出现 reward hacking。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 架构、tokenizer 设计、EPL/IPL 策略、GRPO 后训练均有清晰描述,关键方程有引用 |
> | 可信赖 | pass | [论文原文] vs [agent 解读] 标注明确; 指标引用带 Table 号; KB 定位准确 |
> | 可区分 | pass | 与 MaskGCT (VQ-VAE on w2v-BERT), CosyVoice (supervised semantic tokens), Vevo (predecessor) 的区别清晰; 低帧率 trade-off 有定量分析 |
> | 可定位 | pass-with-fixes | KB 背景充分; 但 chromagram tokenizer 的新颖性与 ProsodyModeling 概念页的关系未充分展开 |
> | 不污染 | pass | 无臆造数据; 局限性坦诚 |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> - ⚠️ [completeness] 速查中指标选取偏 expressive speech 和 singing 域,regular speech (SeedTTS test-en WER 3.639 vs CosyVoice 2 的 2.890) 未提及
> - ⚠️ [sourcing] §V-E pitch region control 的因果解释 ("we hypothesize that...") 未标注为论文原文
> - ℹ️ [cosmetic] FM stage 的 REPA 策略值得展开 (align 5th layer hidden to w2v-BERT 2.0 features)
> 详见 `_review/Vevo2-review.yml`
