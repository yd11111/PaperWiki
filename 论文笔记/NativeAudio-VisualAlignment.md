---
type: paper
tier: deep
title: "Native Audio-Visual Alignment for Generation"
arxiv_id: "2605.30073"
source: "Sources/NativeAudio-VisualAlignment.pdf"
authors: [Longbin Ji, Guan Wang, Xuan Wei, Zhenyu Zhang, Shuohuan Wang, Chenye Yang, Xiangrui Liu, Yu Sun, Jingzhou He]
year: 2026
venue: "arXiv preprint"
tags: [audio-video-generation, MMDiT, diffusion, multimodal, timbre-control, audio-visual-alignment, joint-generation]
concepts: ["[[Classifier-FreeGuidance]]", "[[DiffusionModel]]", "[[SpeakerEmbedding]]"]
models: ["[[CosyVoice]]", "[[CosyVoice2]]"]
tasks: []
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: NAVA 属于 diffusion-based 多模态生成系统,使用 MMDiT (Multi-Modal Diffusion Transformer) 架构进行联合音频-视频去噪。其 Condition-Factorized CFG 机制直接扩展了 [[Classifier-FreeGuidance]] 的标准方法,将单一 guidance 分解为 text/alignment/timbre 三个独立方向 -- 这与 KB 中已记录的 VoXtream2 多条件 CFG 和 X-Voice 解耦 CFG 形成有趣对应。Timbre-in-Context Conditioning 本质上是 [[SpeakerEmbedding]] 在音频-视频联合生成场景中的新注入范式: 不用全局 embedding 或辅助分支,而是将 timbre token 内嵌到 prompt 的对应语音片段中,通过 cross-attention 路径注入。评估使用 [[SEED-TTS-Eval]] 的 EN 子集,与 [[CosyVoice]] 和 [[CosyVoice2]] 等纯语音模型对比。
>
> **已有认知 vs 本文创新**: KB 中的 CFG 概念页已记录了从标准 CFG 到多条件解耦 CFG 的演进(VoXtream2 三条件、X-Voice 非对称预热),但 NAVA 的 factorized CFG 是首次在音频-视频联合生成中应用,且增加了 alignment guidance 这个跨模态同步方向。Timbre-in-Context 与 KB 中的 speaker embedding 注入方式(concat/add/cross-attn/prefix)形成新的范式: 将 timbre 作为 prompt 内的 span-level 条件,而非全局条件。
>
> 检索命中: [[SpeakerEmbedding]], [[SEED-TTS-Eval]], [[CosyVoice2]] | 过滤: [[Classifier-FreeGuidance]](pending-review), [[DiffusionModel]](pending-review), [[Diffusion-basedTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: NAVA 提出 "先对齐再融合" 的 MMDiT 架构,将音频-视频同步与语义条件解耦,在联合音视频生成中以 6.3B 参数实现最优同步和视频质量
> - **路线**: Text/Timbre --> Context Tokens; Video/Audio --> separate VAE latents --> Hierarchical Alignment Layers (modality-aware AV self-attn + context cross-attn, 10 blocks) --> Unified Fusion Layers (shared projection + joint denoising, 20 blocks) --> Audio + Video output [§3.1, Fig 2]
> - **指标**: Verse-Bench Sync-C 7.791 (best) / Sync-D 7.566 (best) / Video Quality 0.659 (best) / WER 0.099 (best); Seed-TTS Speaker Similarity 66.7 (best among AV models, competitive with audio-only CosyVoice2 65.2) [Table 1, Table 2]
> - **可借鉴**: (1) Condition-Factorized CFG -- 将不同条件轴(text/alignment/timbre)分别 dropout+guidance,可迁移到任何多条件扩散系统; (2) Timbre-in-Context -- 将参考音色作为 span-level context token 而非全局 embedding,天然支持多说话人; (3) Rate-aware RoPE rescaling 处理异构 token rate
> - **局限**: 长尾音频事件(罕见动物叫声、音乐、歌唱)生成仍弱; 代码未开源; 训练成本极高(~107K H100 GPU-hours); 论文为 Baidu 内部工作,数据 pipeline 不可复现

## 核心问题

**联合音频-视频生成中,如何让音频和视频在生成过程中"原生"同步,而不是事后对齐?**

现有开源方法面临两难 [Fig 1]:
1. **双塔设计** (Ovi, LTX, MoVA): 音频和视频在各自空间独立生成,跨模态对应仅通过后期交互模块建立 -- "后对齐" 削弱了细粒度的音视频协同演化 [§1]
2. **全统一三模态设计** (daVinci-MagiHuman): 将文本、音频、视频 token 放入同一 attention 空间 -- 高层语义控制与低层音视频同步在同一表示空间内耦合,可能阻碍专门的同步结构形成 [§1]

NAVA 的核心 insight 是**解耦**: 音视频同步(synchronization)和语义条件(conditioning)应该在不同的空间中完成。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

NAVA 采用 **Context-Conditioned Native Audio-Visual Alignment** 范式 [§2.1]:

1. **音频-视频先在专用同步空间中交互** (native AV alignment):
   `[h'_a, h'_v] = SelfAttn([h_a, h_v])` -- joint self-attention 让音视频 token 直接建立事件级对应,不插入 context token [Eq.4]

2. **语义/音色条件通过 cross-attention 外部注入**:
   `[h̃_a, h̃_v] = CrossAttn([h'_a, h'_v], c)` -- context 作为外部调制信号,不参与同步空间 [Eq.5]

[论文原文] 这一设计的动机是: 同步是低层的时序对应(lip-speech, impact sound),条件是高层的语义引导(说什么、什么风格),两者不应在同一表示空间内竞争 [§2.1]。

### 关键设计选择

#### 1. Align-then-Fuse MMDiT [§2.2, Fig 2]

30 个 MMDiT blocks 分两段:

**前 10 层 -- Hierarchical Alignment Layers (HAL)**:
- **Modality-Decoupled Alignment Projection**: 音频 (spectrogram latents) 和视频 (video latents) 有不同的时空结构和 token rate,直接共享投影会过早压缩模态特性 --> 先用模态专用投影映射,再放入共享交互空间 [§2.2]
- **Audio-Video Joint Self-Attention & FFNs**: 在共享空间内反复交叉交互。关键技术: **Rate-aware RoPE rescaling** `θ_rope = TR_v / TR_a` 将异构 token rate 统一到可比较的时间坐标系 [Eq.6]
- **Context-Guided Cross-Attention & FFNs**: 文本和音色条件通过独立的 cross-attention 注入,避免进入 AV 同步的 self-attention 空间

[agent 解读] HAL 的设计逻辑类似于 NLP 中的 "先让不同模态学会互看,再统一表示" 的渐进策略。RoPE rescaling 是必要的,因为视频 token rate (4x16x16 压缩) 和音频 token rate (multi-channel spectrogram) 差异巨大。

**后 20 层 -- Unified Fusion Layers (UFL)**:
- **Modality-Shared Unified Projection**: 音视频共享投影参数,此时 HAL 已经缩小了表示差异,共享投影更稳定高效 [§2.2]
- 仍保留 context cross-attention,语义引导不中断

**消融验证** [Table 3]:
- UFL-only (5B): Sync-C 7.643, IB 33.22 -- 无早期对齐,同步较弱
- HAL-only (7.7B): Sync-C 7.030, Video Quality 66.62 -- 无后期融合,视频质量和 IB 退化
- HAL+UFL (6.3B): Sync-C 7.684, IB 34.34, Video Quality 67.67 -- 最佳平衡

#### 2. Timbre-in-Context Conditioning [§2.3]

多说话人场景需要指定"谁说哪句话"。NAVA 的做法:

1. 对每个语音片段 S_i 的参考音频 R_i,用 timbre encoder 提取 timbre token: `s_i = E_tim(R_i)` [Eq.7]
2. 将 timbre token 嵌入对应语音片段的 prompt 结构: `S_i -> [<S>, s_i, Text(S_i), <E>]` [Eq.8]
3. 最终 context 序列: `c = Augment(P; {(S_i, s_i)})` [Eq.9]
4. 通过已有的 context cross-attention 路径注入,**无需额外 speaker-control 分支**

[论文原文] "Because timbre information is represented in the context pathway, the mechanism requires no auxiliary speaker-control branch or backbone modification." [§2.3]

[agent 解读] 这是一个优雅的设计: 将 timbre 从全局条件变为 span-level 条件,天然支持组合式多说话人控制。与传统 speaker embedding 的全局注入 (concat/add/cross-attn) 形成鲜明对比,更类似 in-context learning 的思路。

#### 3. Condition-Factorized CFG [§2.4.3]

三个独立 guidance 方向 [Eq.10-11]:
- `Delta_text` = 有文本 vs 无文本 --> 控制 prompt 遵循度
- `Delta_align` = 有 AV 交互 vs 无 AV 交互 --> 控制音视频同步
- `Delta_timbre` = 有 timbre vs 无 timbre --> 控制音色一致性

最终: `v̂ = v + s_text * Delta_text + s_align * Delta_align + s_timbre * Delta_timbre`

**训练支持** [§2.4.2]:
- **Random Cross-modality Attention Masking** (20%): 随机 mask AV 间的 cross-modal attention,提供 alignment guidance 的训练对比
- **Random Timbre-in-Context Conditioning** (20%): 随机 drop timbre tokens,提供 timbre guidance 的训练对比

**消融** [Table 4]:
- Alignment CFG: Sync-C 6.170 --> 7.791, Sync-D 8.755 --> 7.566, IB 0.355 --> 0.402 (巨幅提升)
- Timbre CFG: ASV 65.5 --> 66.7 (speaker similarity 提升),WER 3.78 --> 4.20 (轻微代价)

### 训练策略

**Progressive Multi-Task Training** [§2.4.1], 三阶段:
1. **Audio initialization**: audio-only : AV = 3:1,稳定音频 pathway 同时保持从 Wan2.2-5B 继承的视频能力 [§2.4.1]
2. **Joint training**: audio-only : AV = 1:2,在高质量音频 + 完整 AV 数据上提升同步 [§2.4.1]
3. **Instruction fine-tuning**: 高质量 AV 数据,多说话人对话、复杂运动、镜头控制 [§2.4.1]

**基础设施** [§3.1, Appendix §6.4]:
- 初始化自 Wan2.2-5B (视频生成 backbone) [§3.1]
- Video VAE: Wan2.2-VAE (4x16x16 压缩), Audio VAE: LTX2.3-VAE (multi-channel spectrogram) [§3.1]
- 128 H100 GPUs, batch size 512, 70K steps, lr 5e-5 (AdamW) [§3.1]
- 总计 ~107,520 H100 GPU-hours (Stages 1-2: 80,640h + Stage 3: 26,880h) [Appendix §6.4]

**数据**: ~15M clips (from 20M audio + 100M video clips after filtering), 160K high-quality for SFT [Appendix §6.2]

## 实验

| 指标 | NAVA (6.3B) | Ovi 1.1 (10B) | MoVA (18B) | daVinci (15B) | LTX 2.3 (19B) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Sync-C ↑ | **7.791** | 7.484 | 7.289 | 7.149 | 7.248 | [Table 1] |
| Sync-D ↓ | **7.566** | 7.979 | 7.808 | 7.816 | 7.690 | [Table 1] |
| IB Score ↑ | 0.313 | 0.199 | 0.269 | 0.269 | **0.337** | [Table 1] |
| Video Quality ↑ | **0.659** | 0.636 | 0.603 | 0.600 | 0.576 | [Table 1] |
| WER ↓ | **0.099** | 0.102 | 0.126 | 0.151 | 0.106 | [Table 1] |
| PQ ↑ | 6.861 | 5.843 | **7.233** | 5.956 | 6.946 | [Table 1] |
| FD ↓ | 0.833 | 0.942 | 0.922 | 0.931 | **0.829** | [Table 1] |

**Seed-TTS 语音质量** [Table 2]:

| 模型类别 | 模型 | WER ↓ | Speaker Sim ↑ |
| --- | --- | --- | --- |
| Audio-only | CosyVoice | 4.29 | 60.9 |
| Audio-only | CosyVoice2 | 2.57 | 65.2 |
| Audio-only | Qwen2.5-Omni | 2.72 | 63.2 |
| Audio-Video | DreamID-Omni | 31.76 | 35.7 |
| Audio-Video | **NAVA** | **4.20** | **66.7** |

[agent 解读] NAVA 在 AV 联合生成模型中大幅领先 DreamID-Omni (WER 31.76 vs 4.20),且 speaker similarity 66.7 甚至超过纯语音模型 CosyVoice2 (65.2)。这说明 Timbre-in-Context Conditioning 在保持 AV 同步的同时,不牺牲语音质量。

**User Study** [Fig 4]:
- T2AV: NAVA win rate 67.5%/60.0%/80.0% vs Ovi/LTX/daVinci (overall quality)
- TI2AV: NAVA win rate 43.9%/37.5%/26.2%/48.8% vs Ovi/MoVA/LTX/daVinci -- LTX 在 TI2AV 场景更有竞争力

## 局限性

1. **长尾音频事件**: 罕见动物声音、音乐、歌唱、复杂混合场景音的生成仍然有限 [§5]
2. **训练成本极高**: ~107K H100 GPU-hours (约 160 块 H100 跑 4 周),不可复现
3. **数据不可复现**: 大规模私有数据 pipeline (Koala-36M 仅占 20%),绝大部分是电影/TV/TED 素材
4. **IB Score 非最优**: 虽然 sync 和 video quality 最优,但语义一致性 (IB 0.313) 不及 LTX 2.3 (0.337)
5. **Audio quality 非最优**: PQ 6.861 弱于 MoVA 7.233; FD 0.833 略弱于 LTX 0.829
6. **未开源**: 代码和模型权重未公开,仅有项目页面

## 点评

**NAVA 的核心价值在于提出了一个清晰的设计原则: synchronization 和 conditioning 应该解耦。** 这个 insight 看似简单,但对比三种范式(双塔/全统一/NAVA)的公式化描述 [Eq.1-5],可以看到 NAVA 的 formulation 确实在概念层面比前两者更合理: 双塔的 posterior alignment 是"各做各的再对齐",全统一的 tri-modal attention 是"一锅炖",而 NAVA 是"先让要同步的先同步,再加外部条件"。

Align-then-Fuse 的渐进设计值得关注: 先用 modality-specific projection 保护异构表示的独特性,再在后期用 shared projection 压缩共享表示。这与 speech-LLM 领域中 modality adapter 的设计思想一致 -- 不要过早融合。消融 [Table 3] 也支持这一结论。

Timbre-in-Context Conditioning 是本文最有创意的设计: 将全局 speaker embedding 变成 span-level context token,天然支持多说话人。这个想法可迁移到对话式 TTS 中,让每个话轮自带音色条件。

Condition-Factorized CFG 的消融结果很有说服力: alignment CFG 单独就带来 Sync-C +1.6 的提升 [Table 4]。这表明训练时的 structured dropout 策略(random cross-modality masking)确实让模型学会了"有 AV 交互 vs 无 AV 交互"的区别。

**主要疑点**: (1) 与 LTX 2.3 对比时,LTX 的 IB Score (0.337) 和 FD (0.829) 优于 NAVA -- 说明 NAVA 的同步增强可能以牺牲部分语义一致性和音频分布匹配为代价; (2) user study 中 TI2AV 场景 NAVA 对 LTX 的 overall quality 仅 26.2% win rate,说明在有 image condition 时 NAVA 的优势减弱; (3) 训练资源门槛极高,实际可复现性接近零。

## 可复用的 idea

1. **Context-Conditioned Native Alignment 范式**: 在任何需要多模态同步的生成系统中,先让需要同步的模态在专用空间中交互,再通过 cross-attention 注入外部条件。可迁移到 audio-motion、speech-gesture 等场景。

2. **Timbre-in-Context Conditioning**: 将参考音色编码为 context token 嵌入到对应 speech span 中,通过 cross-attention 注入。可直接迁移到对话式 TTS 的多说话人控制 -- 不需要全局 speaker embedding,每个话轮自带音色。

3. **Condition-Factorized CFG + Structured Dropout**: 对多条件扩散系统,训练时对不同条件独立 dropout,推理时分别引导。这个 pattern 可迁移到任何有 >2 个条件的扩散/flow-matching 系统。

4. **Rate-aware RoPE rescaling**: 处理异构 token rate 的 joint attention 场景,`theta_rope = TR_target / TR_source` 将不同 token rate 的序列统一到可比较的时间坐标系。

5. **Align-then-Fuse 渐进架构**: 前段 modality-specific projection 保护各模态独特表示,后段 shared projection 鼓励压缩融合。可作为多模态 Transformer 的通用设计模式。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节 WHY 解释充分,速查卡片 5 个可复用 idea 具体可迁移 |
> | 可信赖 | pass | 主要数字均有 [Table N] 标注,指标名称正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率 >90% |
> | 可定位 | pass | KB 背景谱系定位具体(CFG 演进线 + speaker embedding 范式对比) |
> | 不污染 | pass | 无反向更新,挂接合理 |
> 
> Issues: 3 (high: 0, medium: 0, low: 3)
> 详见 `_review/NativeAudio-VisualAlignment-review.yml`
