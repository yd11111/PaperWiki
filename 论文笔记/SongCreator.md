---
type: paper
tier: deep
title: "SongCreator: Lyrics-based Universal Song Generation"
arxiv_id: "2409.06029"
source: "Sources/SongCreator.pdf"
authors: [Shun Lei, Yixuan Zhou, Boshi Tang, Max W. Y. Lam, Feng Liu, Hangyu Liu, Jingcheng Wu, Shiyin Kang, Zhiyong Wu, Helen Meng]
year: 2024
venue: "NeurIPS 2024"
tags: [song-generation, lyrics-to-song, singing-voice, dual-sequence-LM, attention-mask, multi-task, editing, accompaniment, latent-diffusion]
concepts: ["[[SingingVoiceSynthesis]]", "[[CodecLanguageModel]]", "[[SemanticvsAcousticTokens]]", "[[DiffusionModel]]", "[[Self-SupervisedSpeechRepresentation]]", "[[Classifier-FreeGuidance]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 3 个实体页: SemanticvsAcousticTokens (confirmed), SingingVoiceSynthesis [待确认], CodecLanguageModel [待确认]; 参考: DiffusionModel [待确认], Self-SupervisedSpeechRepresentation [待确认], Classifier-FreeGuidance [待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: SemanticvsAcousticTokens (BEST-RQ semantic tokens + LDM 解码), SingingVoiceSynthesis (lyrics-to-song 属 Text-to-Song 分类), CodecLanguageModel (DSLM 是 LM-based 音频建模), DiffusionModel (LDM renderer), Self-SupervisedSpeechRepresentation (BEST-RQ 属 SSL), Classifier-FreeGuidance (训练中使用 CFG-inspired dropout)
> 过滤: 无
> 未命中但可能相关: MaskedGenerativeModeling (non-causal mask training 有关联)
>
> **定位**: SongCreator 在 KB 中的位置:
> - **SingingVoiceSynthesis**: SVS 概念页将 SongCreator 归入 "Text-to-Song Generation" 子任务,与 Jukebox、SongGen、YuE 并列。SongCreator 是 NeurIPS 2024 发表的较早期工作,在 lyrics-to-song (含歌声+伴奏) 任务上是 Jukebox 之后第二个发表的学术系统。
> - **SemanticvsAcousticTokens**: SongCreator 使用 BEST-RQ 提取 semantic tokens 作为 LM 和 LDM 之间的桥梁 [§3.1],属于 "semantic → renderer" 两阶段范式 (类 AudioLM)。KB 中 BEST-RQ 尚无独立页面,但其在 Survey benchmark 中以 ViSQOL 3.05 领先 HuBERT/MERT/MusicFM [Table 13]。
> - **CodecLanguageModel**: DSLM 是一种专为歌曲设计的 LM 变体 — 双解码器 + 双向交叉注意力。与标准 codec LM (单解码器 AR) 不同,DSLM 显式分离 vocals/accompaniment 两条序列,通过 BCA 建模互信息。
> - **DiffusionModel**: LDM (VAE + U-Net diffusion) 负责将 semantic tokens 渲染为 44.1kHz 音频。这是 MeLoDy (Lam et al., 2024) 的复现,属于 latent diffusion 路线。
> - **与 [[论文笔记/Vevo2|Vevo2]] 的关系**: 两者都尝试统一歌声与伴奏的生成,但思路完全不同。SongCreator 用 dual-sequence LM 显式建模两条序列的交互; Vevo2 用 chromagram VQ-VAE 统一 prosody 表示 + 单 AR LM。SongCreator 可控制歌声和伴奏的独立声学条件 (双 prompt),这是 Vevo2 不具备的能力。

## 速查

> [!summary] 速查
> - **一句话**: 提出双序列语言模型 (DSLM) 将歌声与伴奏作为两条交互序列建模,配合注意力掩码策略实现单系统完成 8 种歌曲生成/编辑任务
> - **路线**: Lyrics → [Lyrics Encoder] → [DSLM: Vocal Decoder + Accompaniment Decoder (BCA 交互) → Song Decoder (非自回归融合)] → Semantic tokens → [LDM (VAE + U-Net Diffusion)] → 44.1kHz Song
> - **指标**: Lyrics-to-song FAD 2.14 / Musicality MOS 4.25 (vs GT 4.3); Lyrics-to-vocals Musicality MOS 3.98 (vs GT 3.89); AB test 60% preferred over Jukebox [Table 3, 4, 15]
> - **可借鉴**: (1) BCA 模块让双序列 LM 的两条 token 流实时交互信息; (2) 注意力掩码策略 (BR/A2V/V2A/None x Causal/Non-causal) 用统一架构适配多种任务; (3) 20% BCA=None dropout 类似 CFG 避免过度依赖序列间交互
> - **局限**: 最长仅 30s 生成 (不支持完整歌曲结构); 无法通过文本描述控制风格/流派; BEST-RQ 对伴奏中歌声信息编码不充分导致清晰度受限; 未开源

## 核心问题

歌曲 (song) 由歌声 (vocals) 和伴奏 (accompaniment) 两个互相关联但又各自独立的组成部分构成 [§1]。现有工作要么只处理其中一个方面 (SVS 只生成歌声、text-to-music 只生成伴奏、accompaniment generation 只生成伴奏),要么将歌声+伴奏作为整体建模 (Jukebox),忽略了两者之间的相互影响 [§1]。

Jukebox 的两个核心局限 [§1]:
1. **不解耦**: 将 vocals+accompaniment 视为一个 entity,忽略两者的 mutual influence,导致歌声不自然、旋律性差,且无法独立控制歌声和伴奏的声学条件。
2. **任务单一**: 仅支持 lyrics-to-song 单一任务,无法泛化到 song editing、accompaniment-to-vocal 等场景。

SongCreator 的核心思路: 将歌声和伴奏视为 **独立但相关的两条序列**,通过专门设计的双序列语言模型 (DSLM) 和注意力掩码策略,在单一系统中实现多种歌曲生成任务。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SongCreator 是一个两阶段系统 [Fig 1]:

**Stage I — Language Modeling (DSLM)**: 接收 lyrics + 可选的 vocal/accompaniment prompt,输出三条 semantic token 序列 (vocals Sv, accompaniment Sa, song Ss)。DSLM 由三个子组件组成: Lyrics Encoder + Vocal Decoder + Accompaniment Decoder (通过 BCA 交互) + Song Decoder (非自回归融合)。

**Stage II — Latent Diffusion**: LDM (VAE + U-Net diffusion) 将 semantic tokens 渲染为 44.1kHz 音频。基于 Stable Audio 实现,VAE 对 DAC 编码器做 diagonal Gaussian 改造,diffusion model 在 32-dim latent space 上以 semantic tokens 为条件生成 [Appendix A.3]。

**Token 提取**: BEST-RQ (0.6B params, 100K hours data) [§3.1, Appendix A.2] 提取 50Hz semantic tokens,再经 VQ 模块量化 (codebook 16384, dim 32) [Appendix A.2]。训练数据涵盖完整歌曲、独立人声和器乐,确保对不同音乐元素的全面理解 [Appendix A.2]。

### 关键设计选择

#### 1. 双序列语言模型 (DSLM) [§3.2]

**动机**: [论文原文] 直接在 vocals+accompaniment 的拼接序列上做 token-by-token 建模 (如 SingSong 方案) 面临 Transformer 的二次复杂度问题 [§3.2]。

**架构** [Fig 2]:

- **Lyrics Encoder**: 4-layer Transformer encoder (1024 hidden, 16 heads), max 256 tokens [Table 12]。提取歌词的发音和语义信息,作为 cross-attention 的 key/value。

- **Vocal Decoder & Accompaniment Decoder**: 各 8 层 DSLM block (1024 hidden, max 1500 tokens)。每个 DSLM block 包含: Self-Attention (SA) → Cross-Attention (CA, 对 lyrics) → Bidirectional Cross-Attention (BCA, 对另一条序列) → Feed-Forward。两个 decoder 分别自回归预测 vocals 和 accompaniment 的 semantic token 序列。

- **Song Decoder**: 4-layer feed-forward Transformer (1024 hidden)。接收 vocal decoder 和 accompaniment decoder 的 embedding 拼接 (Ev, Ea ∈ R^{T×de} → Es ∈ R^{T×2de}),**非自回归** 生成 song semantic tokens [Eq 5, §3.2]。

[论文原文] 非自回归的 song decoder 设计是因为: 输入已包含完整的 vocals 和 accompaniment 信息,不需要额外的自回归建模; 其作用是学习如何自然融合两条序列,同时通过 song loss 帮助模型减少 source separation 工具引入的 artifact [§3.4]。

**Prompt 控制** [§3.2]: Vocal prompt Sv_hat 控制说话人、旋律、节奏; accompaniment prompt Sa_hat 控制乐器、音乐旋律、节奏。prompt 的 semantic tokens 作为 prefix 传入对应 decoder,模型通过 in-context learning 预测后续 token [§3.2]。

#### 2. 双向交叉注意力 (BCA) [§3.2]

**定义** [Eq 1-3]: 在 vocal decoder 中,BCA 让当前 vocal token 的表示 attend to accompaniment decoder 的输出:

```
Q_v = H_v W_Q^v, K_v = H_a W_K^v, V_v = H_a W_V^v
A_v = softmax((Q_v K_v^T) / sqrt(d_k) + M)
```

其中 H_v, H_a 分别为 vocal/accompaniment decoder 的前一层输出,M 为 mask matrix 控制哪些 token pair 可以互相 attend。

[agent 解读] BCA 的核心价值在于: 在 vocal decoder 生成歌声 token 时,可以 attend to 已生成的 accompaniment token,从而让歌声"配合"伴奏 (反之亦然)。这是 SongCreator 区别于 Jukebox (整体建模) 和 SingSong (独立建模) 的关键创新。消融实验 [Fig 3] 证明了这一点: 移除 BCA 后 lyrics-to-song 的 preference 从 85% 降到 14%。

#### 3. 注意力掩码策略 [§3.3, Table 2]

SongCreator 为 SA 和 BCA 分别设计了多种 mask 策略,不同任务使用不同组合:

**SA 掩码** (控制序列内部 token 的可见性):
- **Causal**: 标准自回归,每个 token 只看到左边 (生成任务)
- **Non-causal**: 所有 token 互相可见 (用于已确定的输入序列,提供完整上下文)

**BCA 掩码** (控制两条序列之间的交互方式):
- **BR (Bidirectional)**: 双向可见,但 token t 只能 attend 到另一序列的 ≤t 位置。用于同时生成 vocals 和 accompaniment (lyrics-to-song)。
- **A2V (Accompaniment-to-Vocals)**: vocal 序列可见完整 accompaniment,反向不可见。用于给定伴奏生成歌声 (accompaniment-to-song)。
- **V2A (Vocals-to-Accompaniment)**: accompaniment 序列可见完整 vocals,反向不可见。用于给定歌声生成伴奏 (vocals-to-song)。
- **None**: 两条序列互不可见。用于独立生成器乐 (music continuation)。

**8 种任务的具体配置** [Table 2]:

| 任务 | SA (Vocal, Accompaniment) | BCA | 说明 |
|------|---------------------------|-----|------|
| Lyrics-to-song | Causal, Causal | BR | 同时生成,双向交互 |
| Lyrics-to-vocals | Causal, Causal | BR | 同时生成,仅取 vocals 输出 |
| Accompaniment-to-song | Causal, Non-causal | A2V | 伴奏已知 (non-causal), vocal 生成 |
| Vocals-to-song | Non-causal, Causal | V2A | 歌声已知, accompaniment 生成 |
| Music continuation | None, Causal | None | 仅 accompaniment decoder |
| Song editing | Causal, Causal | BR | 同 lyrics-to-song |
| Vocals editing | Causal, None | None | 仅 vocal decoder |
| Vocals editing in song | Causal, Non-causal | A2V | 同 accompaniment-to-song |

[agent 解读] 这个设计的精妙之处在于: 通过改变 mask matrix M (而非模型结构或参数),同一个 DSLM 可以适配 8 种不同任务。这比传统多任务学习 (每个任务一个 task head) 更优雅,因为不需要额外参数,且所有任务共享底层表示。

### 训练策略

**多任务训练** [§3.4]: 同时在三种训练任务上优化,每种任务对应不同的 SA/BCA mask 配置:

1. **Song generation from lyrics**: SA=Causal/Causal, BCA=BR (80%) 或 None (20%)。
   - [论文原文] 20% 的 None 策略灵感来自 classifier-free guidance 相关工作 [54, 62],确保不干扰 BCA 的训练 [§3.4]。
   - [agent 解读] 这实际上等价于 CFG 的 unconditional dropout: 以 20% 概率让模型学会不依赖另一条序列的信息独立生成,这在推理时提供了 music continuation (None) 和独立歌声生成 (Vocal Only) 的能力。

2. **Song generation from pre-determined accompaniment/vocals**: 已知序列用 non-causal SA,待生成序列用 causal SA,BCA 用 A2V 或 V2A。已知序列随机 mask 20% tokens 鼓励模型学习上下文关系 [§3.4]。

3. **Song editing**: 结合前两种任务,随机选取目标序列末尾一段 span 替换 audio prompt,用 <EDIT> token 区分编辑任务和生成任务 [§3.4]。

**损失函数**: 对 vocals、accompaniment、song 三条序列分别计算 cross-entropy loss,取总和优化 DSLM [§3.4]。Non-causal 策略下对所有 token (不仅是 masked token) 计算 loss [§3.4]。额外 20% 概率 mask lyrics 鼓励无条件生成 [§3.4]。

**训练配置**: 8500 hours song data (~270K songs), 经 ASR+VAD 切分为 1.7M clips (每段 ≤ 30s), Demucs 分离 vocals/accompaniment [§4.1]。DSLM ~0.6B params, 8x A800, 500K steps, batch size 8/GPU [§4.1]。

## 实验

| 指标 | SongCreator | Best Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Lyrics-to-song FAD ↓ | **2.14** | 2.31 (MusicGen) | Internal | [Table 3] |
| Lyrics-to-song Musicality MOS ↑ | **4.25** | 3.55 (GPT V&S) | Internal | [Table 3] |
| Lyrics-to-song Quality MOS ↑ | **4.08** | 3.64 (GPT V&S) | Internal | [Table 3] |
| Lyrics-to-vocals Musicality MOS ↑ | **3.98** | 3.64 (GPT) | Internal | [Table 4] |
| Prompt lyrics-to-song Musicality ↑ | **4.01** | 3.46 (MusicGen) | Internal | [Table 5] |
| Prompt lyrics-to-song Similarity ↑ | **3.82** | 3.27 (MusicGen) | Internal | [Table 5] |
| Prompt lyrics-to-vocals SECS ↑ | **0.68** | 0.66 (VALL-E) | Internal | [Table 6] |
| Vocals-to-song FAD ↓ | 1.88 | **1.46** (Single) | Internal | [Table 7] |
| Accompaniment-to-song FAD ↓ | **1.24** | 1.64 (GPT) | Internal | [Table 8] |
| Song editing Musicality MOS ↑ | **4.01** | 3.84 (GPT) | Manual 30 | [Table 10] |
| AB test vs Jukebox ↑ | 60% | 38.5% | Jukebox demos | [Table 15] |
| AB test vs SingSong ↓ | 30% | **54.1%** | SingSong demos | [Table 16] |
| RTF (inference speed) | 2.793 | 1.525 (GPT) | V100 | [Table 20] |

### 关键发现

**Lyrics-to-song 大幅领先** [§4.2, Table 3]: SongCreator 与 Ground Truth 的差距仅为 Musicality 0.05、Quality 0.01,远超所有 baseline。这验证了 DSLM 的 dual-sequence 建模方案在 lyrics-to-song 任务上的有效性。

**BCA 的学习互利性** [§4.2, Table 4]: 有趣的是,SongCreator (同时建模 vocals 和 accompaniment) 在 lyrics-to-vocals 任务上也优于 SongCreator (Vocal Only),Musicality 3.98 vs 3.68。[论文原文] 学习 vocals-accompaniment 关系对单独生成 vocals 也有益 [§4.2]。

**BCA 消融** [§4.3, Fig 3]: 移除 BCA 后,lyrics-to-song 的 AB preference 从 85% 暴跌至 14%,lyrics-to-vocals 从 72% 降至 22%。这是 DSLM 最核心的设计验证。

**SA mask 消融** [§4.3, Fig 4]: 移除 non-causal mask 训练后,所有三个需要 non-causal 的任务 (lyrics-to-song, vocals-to-song, accompaniment-to-song) 都显著退化,尤其 vocals-to-song (68% → 26%)。

**BCA mask 消融** [§4.3, Table 18-19]: Lyrics-to-song 中 BR 策略以 76-85% 的 preference 大幅优于 A2V/V2A/None; accompaniment-to-song 中 A2V 以 59% 优于 BR (27%)。这验证了为每个任务选择特定 BCA mask 的必要性。

**与 SingSong 的差距** [Table 16]: 在 vocals-to-song 的 AB test 中,SingSong 以 54.1% vs 30% 领先。[论文原文] 可能原因是 SingSong 使用了大规模高质量数据集 (46K hours vs SongCreator 的 8.5K hours) [§4.2]。

**推理速度** [Table 20, Appendix F]: SongCreator RTF 2.793,快于 MusicLM (14.545) 和 GPT Vocals & Song (3.059),但慢于 GPT (1.525) 和 MusicGen (2.104)。DSLM 的双序列同步建模带来额外开销,但比级联方案 (先 vocals 再 song) 更高效 [Appendix F]。

## 局限性

1. **最大生成长度 30s** [§5]: 受限于训练数据裁剪长度,SongCreator 无法生成具有完整结构 (verse-chorus-bridge) 的歌曲。这在实际应用中是致命限制。

2. **无文本描述控制** [§5]: 缺乏 text description → style/genre 的控制通路,只能通过 audio prompt 控制声学条件。相比 Suno/Udio 的 text prompt 方式,可控性维度受限。

3. **BEST-RQ 对歌声编码的局限** [§5]: [论文原文] 伴奏的干扰使得 BEST-RQ 难以完全编码歌声信息,导致合成歌声清晰度受限。未来需要为歌曲设计更好的 semantic 表示。

4. **数据规模瓶颈**: 仅 8.5K hours 的训练数据 (对比 Suno 等工业系统的数据量级)。在 vocals-to-song 的 AB test 中输给使用 46K hours 数据的 SingSong [Table 16],暗示数据量是重要瓶颈。

5. **Source separation 引入的噪声**: 训练数据中 vocals 和 accompaniment 由 Demucs 分离获得 [§4.1],separation 的不完美 (泄漏、artifact) 会影响 DSLM 的学习。论文通过 song decoder 的联合 loss 缓解但未彻底解决。

## 点评

**核心贡献清晰**: DSLM 的 dual-sequence 建模思路是解决 song generation 中 vocals-accompaniment 交互的自然方案。与 Jukebox 的整体建模和 SingSong 的独立建模相比,DSLM 在显式建模两者关系的同时保持各自的独立性。消融实验 (Fig 3, 4, Table 18-19) 充分验证了每个设计选择的必要性。

**注意力掩码的统一设计精巧**: 通过 SA (causal/non-causal) x BCA (BR/A2V/V2A/None) 的组合,用同一个模型覆盖 8 种任务。这种方法避免了多任务学习中常见的 task-specific head 设计,参数效率更高。与 UniLM [32] 和 GLM [33] 在 NLP 中的类似思路一脉相承,但 SongCreator 将其扩展到双序列的交叉注意力维度。

**评估不够严格**: (1) 所有实验使用内部数据集,无公开可复现的 benchmark。(2) FAD 和 MOS 评测的样本量和评估细节 (如 MOS 评估者数量 25 人,AB test 20 人 [Appendix G.2]) 相对有限。(3) 缺少与 Suno/Udio 等工业系统的对比 (尽管后者未公开方法,但可用 demo 做 AB test)。

**与后续工作 (Vevo2, SongGen) 的对比**: SongCreator (NeurIPS 2024) 是早期学术探索。后来的 SongGen (2025) 用单阶段 AR + X-Codec 实现了更简洁的 song generation; Vevo2 (2025) 通过 chromagram tokenizer 实现了 notation-free 的统一 speech-singing 框架。SongCreator 的 DSLM 虽然设计精巧,但"分离再融合"的思路增加了系统复杂度,且依赖 source separation 工具的质量。

## 可复用的 idea

1. **双向交叉注意力 (BCA) 建模两条相关序列的交互**: 适用于任何需要同时生成两条相关但独立的序列的场景 (如对话系统中的两个说话人、视频中的画面和音频)。BCA 的 mask matrix 设计 (BR/A2V/V2A/None) 提供了灵活的交互模式控制。

2. **注意力掩码策略实现任务统一**: 用 mask matrix 切换而非模型结构切换来适配多任务,参数效率极高。可直接应用于任何多任务 Transformer 系统。训练时混合多种 mask 策略进一步提升每个任务的性能。

3. **20% BCA=None dropout**: 训练 lyrics-to-song 时以 20% 概率使用 BCA=None,类似 CFG 的 unconditional dropout,让模型学会独立生成能力。这个简单 trick 同时赋予了模型 music continuation 和 solo vocal generation 的能力,值得在类似双流架构中借鉴。

4. **Song Decoder 非自回归融合 + song loss 缓解 source separation artifact**: 当训练数据来自不完美的 source separation 时,通过联合 loss (同时预测分离信号和混合信号) 可以缓解分离噪声。这个思路可泛化到任何依赖 signal decomposition 训练数据的系统。

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> Issues: 3 (high: 0, medium: 0, low: 3)
> 详见 `_review/SongCreator-review.yml`

---

检索命中: [[SemanticvsAcousticTokens]], [[SingingVoiceSynthesis]], [[CodecLanguageModel]], [[DiffusionModel]], [[Self-SupervisedSpeechRepresentation]], [[Classifier-FreeGuidance]] | 过滤: 无 | 未命中但可能相关: MaskedGenerativeModeling
