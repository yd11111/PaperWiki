---
type: paper
tier: deep
title: "MIDI-VALLE: Improving Expressive Piano Performance Synthesis Through Neural Codec Language Modelling"
arxiv_id: "2507.08530"
source: "Sources/MIDI-VALLE.pdf"
authors: [Jingjing Tang, Xin Wang, Zhe Zhang, Junichi Yamagishi, Geraint Wiggins, György Fazekas]
year: 2025
venue: "ISMIR 2025"
tags: [music-synthesis, codec-LM, MIDI-to-audio, piano, RVQ, zero-shot, AR-NAR]
concepts: ["[[ResidualVectorQuantization]]", "[[CodecLanguageModel]]"]
models: ["[[EnCodec]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 2 个待确认实体页: [[ResidualVectorQuantization]]✓, [[EnCodec]]✓, [[SemanticvsAcousticTokens]]✓, [[CodecLanguageModel]][待确认], [[MusicalScoreEncoder]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ResidualVectorQuantization]]✓, [[EnCodec]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[MusicalScoreEncoder]](pending-review) | 未命中但可能相关: 无

**谱系定位**: MIDI-VALLE 属于 CodecLM 范式在音乐合成领域的迁移应用。CodecLM 页记录了 VALL-E 作为首个大规模 codec LM TTS 系统,以及 AR+NAR 两阶段生成策略。MIDI-VALLE 继承了这一范式,但将条件从 phoneme text 替换为 MIDI token sequence,将 speaker prompt 替换为 piano performance prompt。

**已有认知**: RVQ 页详细记录了多层级量化方法,其中第一层编码 coarse 信息、后续层编码 fine details 的层级信息结构正是 MIDI-VALLE AR(预测第 1 层)+ NAR(预测 2-4 层)设计的理论基础。EnCodec 页记录了 EMA codebook 训练和 RVQ 架构。SemanticvsAcousticTokens 页指出纯 acoustic tokens 语义信息稀疏但声学保真度高 -- MIDI-VALLE 选择 acoustic tokens 路线正是因为音乐合成对声学保真度的要求远高于语义理解。VALL-E 笔记已详细分析了 AR+NAR 的两阶段 codec 生成架构。

**创新判断**: MIDI-VALLE 的核心贡献不在模型架构创新(基本沿用 VALL-E),而在于验证了 codec language modeling 范式从语音到音乐的跨域迁移可行性,以及 Octuple MIDI tokenisation 相比 piano-roll 在 EPR-EPS 管线兼容性上的优势。这是一个应用驱动而非方法驱动的工作。

## 速查

> [!summary] 速查
> - **一句话**: 将 VALL-E 的 codec language modeling 范式从语音迁移到钢琴演奏合成,用 Octuple MIDI tokenisation 替代 piano-roll 表征,在 ATEPP/Maestro 上 FAD 降低 75%+
> - **路线**: Performance MIDI → Octuple Tokenisation (pitch/vel/dur/IOI/pos/bar) + Audio Prompt → Piano-Encodec → AR Decoder (1st codebook) → NAR Decoder (2-4th codebooks, conditioned on 3s audio prompt) → Piano-Decoder → Waveform
> - **指标**: FAD 3.329 vs M2A 11.014 (ATEPP-GT), FAD 11.281 vs 34.479 (Maestro-GT) [Table 4]; 听力测试 202:58 偏好票 [§6.2]; Piano-Encodec Spec. 0.123 vs Encodec 0.304 [Table 3]
> - **可借鉴**: TTS codec LM 范式向音乐合成的迁移方法论; Octuple tokenisation 解决 EPR-EPS MIDI 表征不一致问题; domain-specific codec fine-tuning (Piano-Encodec) 大幅提升重建质量
> - **局限**: jazz 泛化差 (Pijama FAD 102 仍很高); MIDI-audio prompt 对齐敏感(timing 偏移导致开头异常); 长段拼接有声学不连续; 未与 MusicGen/AudioLM 等音乐生成系统对比; 仅 4 层 RVQ 较浅

## 核心问题

传统 Music Performance Synthesis (MPS) 的两阶段管线面临两个结构性问题 [§1]:

1. **MIDI 表征不一致**: EPR 模型输出 tokenised MIDI 或连续特征,EPS 模型期望 piano-roll 输入。两者在时间信息(note timing, pedal treatment)上存在表征鸿沟,导致直接对接不可行,需额外 fine-tuning [§2.1]。

2. **泛化能力差**: 现有 EPS 模型(如 M2A)主要在 Maestro 数据集上训练,该数据集录音环境单一(钢琴比赛),导致模型无法适应多样化的声学环境和音色变化 [§2.1]。Tang et al. [3] 尝试在更多样的 ATEPP 上 fine-tune M2A,但仍出现混乱的混响和背景噪声。

MIDI-VALLE 的核心思路: 既然 TTS 领域的 VALL-E 通过 codec language modeling 解决了 zero-shot speaker adaptation 问题,能否将同一范式迁移到 MIDI-to-audio 任务,同时用统一的离散 token 表征解决 MIDI 不一致问题?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MIDI-VALLE 沿用 VALL-E 的 AR+NAR 两阶段架构 [§3.2, Fig 1]:

1. **输入**: MIDI token 序列 x = {x_0, ..., x_L} + 3 秒 audio prompt
2. **Audio Encoding**: Piano-Encodec 将音频编码为 C_{T x 4} 的 codec matrix (4 层 RVQ)
3. **AR Decoder**: 自回归预测第 1 层 codec tokens c_{t,1},不使用 acoustic prompt,仅基于 MIDI tokens
4. **NAR Decoder**: 非自回归生成第 2-4 层 codec tokens c_{t,2:4},条件化于 3 秒 audio prompt C_tilde
5. **Decoding**: Piano-Encodec decoder 重建音频 y_hat = decodec(C_hat)

训练目标: max p(C | x, C_tilde) [§3.2]

### 关键设计选择

**1. MIDI Tokenisation: Octuple 方法替代 Piano-roll [§3.1.2]**

[论文原文] Piano-roll 将音符编码在固定时间网格上,仅表示 onset 和 duration,缺乏捕捉细微时间变化(影响 articulation)的分辨率和灵活性。Octuple 方法为每个音乐特征使用独立词汇表,实现 note-wise encoding,产生 K x N 数组(特征数 x 音符数)。

扩展 Octuple 方法: 额外 tokenise inter-onset interval (IOI),捕捉连续音符之间的起始时间差异 [§3.1.2]。

词汇表大小 [Table 1]: Pitch 92, Velocity 68, Duration 1156, IOI 772, Position 388, Bar 20 (vs 语音 phoneme 仅 512)。

[agent 解读] 这个设计选择有两层意义: (1) 统一 EPR 和 EPS 的 MIDI 表征,因为 Octuple 在两个阶段中都可使用; (2) 多维度独立编码比 piano-roll 的 2D grid 更精确地保留了演奏中的 expressive timing 信息。

**2. Audio Tokenisation: Piano-Encodec [§3.1.1]**

基于 MusicGen 的方法,fine-tune EnCodec 产生 4 层 RVQ codebook (codebook size 2048, 32kHz mono, 50Hz frame rate) [§4.2]。

[论文原文] Fine-tuning 大幅提升了钢琴音频的重建质量: spectrogram distortion 从 0.304 降至 0.123, chroma distortion 从 0.478 降至 0.140 [Table 3]。

**3. NAR Prompt 策略的修改 [§3.2]**

[论文原文] VALL-E 的 NAR decoder 使用 neighbouring-context 策略(邻近上下文作为条件),而 MIDI-VALLE 改为使用固定 3 秒 audio prompt。原因是音乐中声学特征在段落间变化远比语音快,neighbouring-context 无法保持一致性。

[agent 解读] 这是一个合理的 domain adaptation: 语音中 speaker identity 在 utterance 内基本稳定,但钢琴演奏的声学特征(力度、共鸣、踏板)在乐句间可能剧烈变化。固定 prompt 确保了 acoustic environment 和 timbre 的一致性。

**4. AR 不使用 Acoustic Prompt [§3.2]**

[论文原文] AR decoder 仅基于 MIDI tokens 自回归预测第 1 层 codec tokens,不接收 acoustic prompt 输入。

[agent 解读] 这与 VALL-E 的做法一致: 第 1 层 RVQ 编码 coarse acoustic structure,主要由内容(MIDI)决定;细节(timbre, environment)由 NAR 的 prompt conditioning 注入。

### 训练策略

- 基于非官方 VALL-E 实现 [25]
- 优化器: ScaledAdam,base lr 0.05,Eden scheduler [§4.2]
- AR 和 NAR 联合训练,梯度同步更新 [§4.2]
- 架构: 12 attention layers, 16 heads, hidden dim 1024 (AR 和 NAR 共享架构但参数独立) [§3.2]
- 训练数据: ATEPP 8825 recordings, ~700h, 分割为 15-20s clips [§4.1]
- 收敛: ~300k steps, 2.5 days on 2x A100 GPUs [§4.2]
- Piano-Encodec fine-tuning: 40 epochs, 1 day on 1x A100 [§4.2]

## 实验

| 指标 | MIDI-VALLE | M2A (baseline) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| FAD (vs GT) | 3.329 | 11.014 | ATEPP | [Table 4] |
| FAD (vs RC) | 2.659 | 11.463 | ATEPP | [Table 4] |
| FAD (vs GT) | 11.281 | 34.479 | Maestro | [Table 4] |
| FAD (vs RC) | 9.168 | 33.753 | Maestro | [Table 4] |
| FAD (vs GT) | 102.022 | 274.153 | Pijama (jazz) | [Table 4] |
| Spec. (vs RC) | 0.199 | 0.214 | ATEPP | [Table 4] |
| Spec. (vs RC) | 0.206 | 0.224 | Maestro | [Table 4] |
| Chroma (vs GT) | 0.436 | 0.421 | ATEPP | [Table 4] |
| Chroma (vs GT) | 0.428 | 0.387 | Maestro | [Table 4] |
| Listening test (total votes) | 202 | 58 | All datasets | [§6.2, Fig 3] |
| Piano-Enc. Spec. | 0.123 | 0.304 (orig. Encodec) | ATEPP | [Table 3] |

**关键发现**:
- FAD 在 ATEPP 和 Maestro 上降低 >75%,表明 codec language modeling 在感知音质上远优于 M2A 的 spectrogram+vocoder 管线 [§6.1]
- Spectrogram distortion (vs reconstruction) MIDI-VALLE 更低,表明更好的 timbre/acoustic 重建 [§6.1]
- Chroma distortion 在 ATEPP 上相近,在 Maestro 上 M2A 略优 -- 因为 M2A 原生在 Maestro 上训练 [§6.1]
- Pijama (jazz) 上 MIDI-VALLE 仍大幅优于 M2A 的 FAD,但绝对值仍很高 (102),表明古典→爵士泛化有限 [§6.1]
- MIDI-VALLE 仅用 transcribed MIDI 训练,但能泛化到 recorded MIDI (Maestro),而 M2A 反向泛化困难 [§6.1]
- 听力测试中 MIDI-VALLE 在 system compatibility evaluation 中跨所有 EPR 系统均被偏好,验证了 Octuple tokenisation 的管线兼容性优势 [§6.2, Fig 3]

## 局限性

1. **Jazz 泛化失败**: 古典音乐训练的模型对爵士的复杂和弦、切分节奏、即兴装饰音表现差(Pijama FAD 仍达 102) [§6.1]。[agent 解读] 这是训练数据分布问题,非架构固有缺陷,但暴露了 codec LM 在音乐多样性上的 scaling 需求。

2. **MIDI-Audio Prompt 对齐敏感**: 3 秒 prompt 截断如果发生在音符中间,会导致生成开头出现异常音符或遗漏 [§6.2]。[agent 解读] 这是 MIDI-audio 时间对齐的固有难题,与 TTS 中 phoneme-audio alignment 类似但更困难(音乐的 polyphony 和 pedal 使对齐更复杂)。

3. **长段拼接不连续**: 将多个 15-20s 合成段拼接时,声学特征存在不连续 [§6.2]。[agent 解读] 这是 segment-level synthesis 的通病,VALL-E 在语音上也有类似问题。

4. **踏板信息缺失**: 由于 ATEPP 踏板转写精度有限,训练时排除了踏板信息 [§4.1]。[agent 解读] 对古典钢琴而言,踏板对音色和延音有决定性影响,这可能是重建质量的一个隐性瓶颈。

5. **评估基线单一**: 仅与 M2A 一个系统对比,未涉及 DDSP-Piano、MusicGen 或其他音乐生成系统。

6. **RVQ 层数浅**: 仅使用 4 层 RVQ,而 TTS 领域的 VALL-E 用 8 层。论文未讨论更多层是否能提升音乐的细节重建。

## 点评

**值得肯定的**:
- 将 TTS 领域的 codec LM 范式成功迁移到音乐合成,这本身是有价值的跨域验证。VALL-E → MIDI-VALLE 的类比关系清晰(phoneme→MIDI tokens, speaker prompt→performance prompt)。
- Octuple tokenisation 解决了 MPS 管线中 EPR-EPS MIDI 表征不一致的实际工程问题,这个 contribution 比架构创新更实用。
- Piano-Encodec 的 domain-specific fine-tuning 效果显著(spectrogram distortion 降低 60%),验证了 neural codec 在特定音频域上 specialization 的重要性。

**值得商榷的**:
- 架构创新有限 -- 本质上是 VALL-E 的直接迁移,仅修改了输入 tokenisation 和 NAR prompt 策略。这更像是一个应用论文而非方法论文。
- 评估不够全面: 仅一个 baseline (M2A),未涉及其他音乐生成系统;未报告生成延迟、模型参数量等实用指标。
- 对 TTS 知识库的启示有限: 本文更多是从 TTS 向音乐迁移知识,而非为 TTS 提供新洞察。

## 可复用的 idea

1. **Domain-specific codec fine-tuning**: Piano-Encodec 的策略(用目标域数据 fine-tune 通用 codec)可迁移到其他垂直音频场景(如歌声、环境声)。效果证明: fine-tuning 比从头训练 codec 更高效,1 天 1 GPU 即可完成 [§4.2, Table 3]。

2. **Octuple-style 多特征独立词汇表**: 当条件信号有多个维度(如 SVS 中的 pitch+duration+lyrics)时,为每个维度使用独立词汇表 + embedding pooling,比合并到单一序列更能保留细粒度信息 [§3.1.2]。

3. **Zero-shot acoustic adaptation 从 TTS 到 music**: 3 秒 prompt 控制 acoustic environment 的策略可用于其他需要环境自适应的音频合成任务 [§3.2, §6.2]。

4. **AR 不接收 prompt / NAR 接收 prompt 的分工**: 让 AR 专注内容预测、NAR 注入风格/环境信息的分离策略,在音乐比语音更有效(因为音乐的声学变化更剧烈) [§3.2]。

## 审阅

> [!review] 审阅 pass-with-fixes (2026-06-08)
> **结论**: pass-with-fixes | 0 high, 2 medium, 2 low
> - (medium) **factual-error**: agent 解读错误地将 AR 不用 prompt 类比为"与 VALL-E 一致",实际 VALL-E AR 接收 acoustic prompt,这是差异点非一致点
> - (medium) **traceability-gap**: 推理时 AR 可选择性接收 audio prompt (§3.2 inference),训练-推理行为差异未捕获
> - (low) frontmatter datasets 为空,应列 ATEPP/Maestro/Pijama
> - (low) Pijama 听力测试 M2A 被偏好的结论未提及
> 详见 `_review/MIDI-VALLE-review.yml`

---

检索命中: [[ResidualVectorQuantization]]✓, [[EnCodec]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[CodecLanguageModel]](pending-review), [[MusicalScoreEncoder]](pending-review) | 未命中但可能相关: 无
