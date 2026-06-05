---
type: paper
tier: deep
title: "UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions"
arxiv_id: "2604.22209"
source: "Sources/UniSonate.pdf"
authors: [Chunyu Qiang, Xiaopeng Wang, Kang Yin, Yuzhe Liang, Yuxin Guo, Teng Ma, Ziyu Zhang, Tianrui Wang, Cheng Gong, Yushen Chen, Ruibo Fu, Chen Zhang, Longbiao Wang, Jianwu Dang]
year: 2026
venue: "arXiv preprint"
tags: [TTS, TTM, TTA, unified-audio, instruction-control, MM-DiT, flow-matching, natural-language-description, music-generation, sound-effects, curriculum-learning, positive-transfer, multi-modal]
concepts: ["[[Instruction-GuidedSpeechSynthesis]]", "[[NaturalLanguageDescriptionforTTS]]", "[[ConditionalFlowMatching]]", "[[Diffusion-basedTTS]]", "[[VariationalAutoencoderforTTS]]", "[[PhonemeRepresentation]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/CosyVoice|CosyVoice]]", "MaskGCT", "F5-TTS", "E2-TTS", "ZipVoice", "ACE-Step", "DiffRhythm+", "AudioLDM", "GenAU", "InstructAudio"]
tasks: ["[[任务库/InstructedSpeechGeneration]]"]
datasets: ["[[数据集/SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]]✓, [[Instruction-GuidedSpeechSynthesis]][待确认], [[NaturalLanguageDescriptionforTTS]][待确认], [[Diffusion-basedTTS]][待确认], [[VariationalAutoencoderforTTS]][待确认], [[PhonemeRepresentation]][待确认] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: UniSonate 是 [[论文笔记/InstructAudio|InstructAudio]] (Qiang et al., 2025) 的直接后继,来自同一团队(Tianjin Univ. + Kuaishou/Kling Team)。InstructAudio 首次将 instruction-guided NL description 控制从 TTS 扩展到统一 TTS+TTM 框架,但不涵盖 sound effects (TTA)。UniSonate 将统一范围进一步拓展至 TTS+TTM+TTA 三模态,是 instruction-guided audio generation 路线迄今覆盖面最广的系统。

**已有认知对比**:
- Conditional Flow Matching (confirmed): UniSonate 沿用与 InstructAudio 相同的 CFM 训练目标和 MM-DiT 架构(Joint DiT 14L + Single DiT 6L),核心架构未变,创新集中在输入表示和训练策略上。
- Instruction-Guided Speech Synthesis [待确认]: UniSonate 延续了 VoxInstruct→CosyVoice→InstructAudio 的统一指令路线,但面临一个新挑战——SFX 没有语言学内容,无法直接用 phoneme 序列作为 content input。动态 token 注入是对这一路线的非平凡扩展。
- Natural Language Description for TTS [待确认]: 相比 InstructAudio 仅控制 timbre/paralinguistic/musical 属性,UniSonate 的 NL description 还需覆盖 SFX 的声学事件描述(如 "underwater bubbles"),描述空间进一步扩大。
- Variational Autoencoder for TTS [待确认]: UniSonate 使用 Mel-VAE (基于 SecoustiCodec 系列) 将 44.1kHz 波形压缩到 43Hz 连续 latent (1024x 降采样),与 InstructAudio 相同的 codec 组件。
- Phoneme Representation [待确认]: UniSonate 对 TTS/TTM 使用 G2P + Zipformer phoneme encoder,但为 SFX 引入可学习 [SFX] token 作为伪音素,通过语音语料的 phoneme-to-duration 比率确定 token 数量,实现跨模态的统一 content 表示。

**创新判断**: 相比 InstructAudio,UniSonate 的核心增量是 (1) 动态 token 注入机制将非结构化 SFX 符号化为伪语言学序列,和 (2) 三阶段课程学习缓解跨模态优化冲突。架构本身(MM-DiT + CFM)与 InstructAudio 基本一致。

## 速查

> [!summary] 速查
> - **一句话**: 首个用 flow matching + MM-DiT 统一 TTS/TTM/TTA 三模态生成的框架,通过动态 token 注入将非结构化 SFX 融入 phoneme 驱动架构,课程学习实现正向迁移
> - **路线**: NL instruction → Qwen2.5-7B (frozen) + content (phoneme for speech/music, [SFX] tokens for effects) → Zipformer → Joint DiT 14L + Single DiT 6L → CFM ODE solver → Mel-VAE decoder → speech/music/SFX
> - **指标**: TTS WER EN 1.47% / ZH 1.25% (best); SongEval Coh 3.18 (SOTA); TTA FAD 4.21 (competitive); 联合训练 vs 单任务: WER 2.24→1.47, Coh 3.11→3.18 [Table 3, 4, 5, 6, 7]
> - **可借鉴**: (1) 动态 token 注入: 用 phoneme-to-duration 比率 lambda 将 SFX 符号化为伪音素序列,复用 phoneme 驱动架构; (2) 三阶段课程学习: speech→music→SFX 渐进扩展避免负迁移; (3) 正向迁移的经验证据: 联合训练显著提升各单任务性能
> - **局限**: TTA 与专用 SOTA (GenAU-L FAD 2.07) 差距明显 (FAD 4.21); 仅支持 2-20s 短片段; 纯文本控制的 one-to-many 模糊性; 1.3B 参数推理成本高; 未开源

## 核心问题

这篇论文要解决的核心问题是: **如何在一个统一的概率框架内同时生成语音、音乐和音效,并使用一致的纯文本指令接口,而不依赖参考音频?**

这个问题的核心难点在于结构化与非结构化语义表示之间的内禀不协调 [§1]:
1. 语音和音乐具有内在的时间结构——phoneme/note 与声学实现之间需要精确的时间对齐
2. 音效 (SFX) 是整体性的、非结构化的,缺乏刚性时间边界
3. 简单拼接数据集训练会导致负迁移——SFX 的高方差会干扰语音的精细发音 [§1]

前身 InstructAudio 成功统一了 TTS+TTM,但无法处理非结构化 SFX——"the integration of unstructured environmental sounds remains an unresolved optimization conflict" [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UniSonate 复用 InstructAudio 的 MM-DiT + CFM 框架,包含两个并行处理流 [§3.1, Fig 2]:

```
文本模态流 (Conditioning):
  NL instruction → Qwen2.5-7B (frozen) → E_I ∈ R^{B×L_I×D}
  content → Zipformer phoneme encoder → E_C ∈ R^{B×L_C×D}
  C_text = Concat(E_I, E_C) ∈ R^{B×(L_I+L_C)×D}

音频模态流 (Generation):
  target waveform (44.1kHz) → Mel-VAE encoder (1024x 降采样, 43Hz) → x_0
  训练: x_t = t·x_1 + (1-t)·x_0 (flow matching 线性插值)
  推理: Gaussian noise → ODE solver (Euler) → x_0 → Mel-VAE decoder → waveform

Joint DiT (14 layers): 两流通过 joint attention 交互
  → QKV from text + audio concatenated → full attention → split back
Single DiT (6 layers): 仅 audio self-attention,细化声学细节
```

**架构参数**: 1.34B 参数,flow matching FFN dim=1024,RoPE 位置编码,32x A800 80GB 训练,batch size=16/GPU,Adam lr=1e-4 [§4.2]。

### 关键设计选择

#### 1. Instruction-Content Alignment 范式

[论文原文] 将条件信号解耦为两路: Instruction (高层属性控制) 和 Content (时间结构控制) [§3.2]。

| 模态 | Instruction | Content |
|------|-------------|---------|
| TTS | 说话人属性 (gender, age, emotion, style, accent) | 转录文本 → phoneme 序列 |
| TTM | 音乐属性 (genre, instrument, tempo, mood) | 歌词 → phoneme 序列 |
| TTA | 声学事件描述 ("underwater bubbles...") | 可学习 [SFX] token 序列 |

**多说话人对话支持**: 为每个说话人提供独立 NL description,在 content 序列中用 speaker-id token ([S0], [S1]) 前缀各自话语 [§3.2]。

#### 2. Dynamic Token Injection (核心创新)

**问题**: SFX 没有语言学内容,无法用 phoneme 序列表示。如果不解决这个问题,就需要修改 phoneme 驱动架构或放弃 SFX 支持。

**解决方案**: 引入可学习的 [SFX] special token 作为伪音素 [§3.2]。关键不在于引入 token 本身,而在于如何确定 token 数量:

1. 从语音语料计算全局缩放因子 lambda = (1/N) * sum(len(P_i) / duration(A_i)),即平均 phoneme-to-duration 比率 [Eq. 2]
2. 对于目标时长为 T_target 的 SFX,content 序列为: C_sfx = [SFX] × floor(lambda · T_target) [Eq. 3]

[论文原文] 使用重复 token 而非单个 global duration embedding 是为了创建"temporal anchors",让 MM-DiT 的 cross-attention 能"walk through"序列,模拟 phoneme 的单调对齐 [§3.2]。

[agent 解读] 这个设计的巧妙之处在于: 它把 SFX 的时间展开问题转化为序列建模问题,使 transformer 用处理语言的同一套离散符号推理机制来处理非语言音频。lambda 的引入确保了 SFX token 密度与 phoneme 密度在数量级上一致,避免了 attention 分配失衡。但一个明显的限制是,所有 [SFX] token 共享同一可学习嵌入,这意味着模型完全依赖 instruction 中的 NL description 来区分不同类型的音效,content 序列本身不携带任何语义信息。

#### 3. Multi-Stage Curriculum Learning

[论文原文] 直接联合训练会导致优化冲突和负迁移——SFX 的高方差会使模型难以收敛于语音的精细发音细节 [§3.3]。

训练按复杂度递增分三阶段 [Algorithm 1]:
- **Stage 1 (Speech Anchoring)**: 1 epoch,仅用 speech 数据 D_S,建立精细发音能力
- **Stage 2 (Semantic Expansion)**: 2 epochs,加入 music 数据 D_S ∪ D_M,扩展到半结构化内容
- **Stage 3 (Universal Generalization)**: 直到收敛,全数据 D_S ∪ D_M ∪ D_E

[agent 解读] 课程学习的关键假设是: 语音是最结构化的模态,先锚定精确的时间对齐能力,再逐步引入更低结构化的模态,使模型在保持结构化能力的基础上适应非结构化数据。这与 pretrain→finetune 思路不同——三个阶段使用同一目标函数,只改变数据组成。

### 训练策略

训练目标为标准 CFM velocity field regression [Eq. 1]:

L_CFM = E_{t,x_0,x_1,C_text} ||v_θ(t, C_text, x_t) - (x_1 - x_0)||²

其中 t ∈ [0,1], x_t = t·x_1 + (1-t)·x_0。推理时用 Euler ODE solver 生成。

**数据**: 50K h speech + 20K h music (同 InstructAudio) + 新增 1.5M SFX clips,44.1kHz,2-20s,中英 1:1,含 0.5% 对话数据 [§4.1]。

## 实验

### TTS 性能 (Word Error Rate)

| 指标 | 本文 | F5-TTS | ZipVoice | CosyVoice2 | InstructAudio | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER-EN (%) ↓ | **1.47** | 1.89 | 1.70 | 2.57 | 1.52 | Seed-TTS | [Table 3] |
| WER-ZH (%) ↓ | **1.25** | 1.53 | 1.40 | 1.45 | 1.35 | Seed-TTS | [Table 3] |

### TTS 指令控制

| 指标 | 本文 | CosyVoice2 | InstructAudio | 出处 |
| --- | --- | --- | --- | --- |
| Gender Acc (%) | 100 | N/A | 100 | [Table 2] |
| Emotion Acc (%) | 80 | 58.33 | 83.33 | [Table 2] |
| Dialogue Acc (%) | 93.33 | N/A | 90 | [Table 2] |
| LSD ↓ | **1.79** | 2.57 | 1.88 | [Table 2] |
| MCD ↓ | **5.46** | 7.11 | 5.71 | [Table 2] |
| QMOS | 3.83 | **3.90** | 3.73 | [Table 2] |
| NMOS | 3.50 | **3.65** | 3.46 | [Table 2] |

### TTM 性能

| 指标 | 本文 | ACE-Step | DiffRhythm+ | InstructAudio | 出处 |
| --- | --- | --- | --- | --- | --- |
| SongEval Coh ↑ | **3.18** | 2.89 | 2.68 | 3.08 | [Table 4] |
| SongEval Mus ↑ | **3.07** | 2.87 | 2.61 | 2.98 | [Table 4] |
| MMOS ↑ | **3.01** | 2.88 | 2.79 | 2.91 | [Table 4] |

### TTA 性能

| 指标 | 本文 | AudioLDM-L | GenAU-L | EzAudio-XL | 出处 |
| --- | --- | --- | --- | --- | --- |
| FAD ↓ | 4.21 | 4.32 | **2.07** | 3.64 | [Table 5] |
| FD ↓ | 30.21 | 29.50 | **14.58** | 14.98 | [Table 5] |
| IS ↑ | 8.22 | 8.17 | 10.43 | **11.38** | [Table 5] |
| CLAP ↑ | 0.156 | 0.208 | 0.300 | **0.314** | [Table 5] |

### 消融实验 (正向迁移验证)

| 配置 | WER-EN ↓ | WER-ZH ↓ | LSD ↓ | MCD ↓ | 出处 |
| --- | --- | --- | --- | --- | --- |
| TTS-Only | 2.24 | 1.40 | 2.63 | 8.70 | [Table 6] |
| Joint (full) | **1.47** | **1.25** | **1.79** | **5.46** | [Table 6] |

| 配置 | Coh ↑ | Mus ↑ | Mem ↑ | Cla ↑ | Nat ↑ | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| TTM-Only | 3.11 | 3.00 | 3.04 | 2.92 | 2.84 | [Table 7] |
| Joint (full) | **3.18** | **3.07** | **3.10** | **2.99** | **2.90** | [Table 7] |

## 局限性

1. **TTA 与专用 SOTA 差距明显**: FAD 4.21 vs GenAU-L 2.07,CLAP 0.156 vs EzAudio 0.314 [Table 5]。[论文原文] 承认统一表示可能难以捕捉非结构化声学环境的极端多样性 [§6]。

2. **仅支持短片段**: 2-20s clips,无法生成长音频(如完整歌曲或有声书) [§6]。attention 机制的内存限制和缺乏层次化长期规划结构是瓶颈。

3. **纯文本控制的 one-to-many 模糊性**: "a sad song" 可对应大量不同声学实现,导致生成结果可能不符合用户未明说的偏好 [§6]。NMOS 3.50 vs CosyVoice2 (参考音频引导) 3.65 [Table 2] 体现了这一固有代价。

4. **推理成本高**: 1.3B 参数 diffusion 模型需多步去噪,限制了实时低延迟场景 [§6]。

5. **[SFX] token 的信息瓶颈**: [agent 解读] 所有 [SFX] token 共享同一可学习嵌入,时间结构的全部信息来自 token 数量而非 token 内容。对于内部结构复杂的 SFX(如先远后近的脚步声),这种表示可能不足。

## 点评

**与 InstructAudio 的增量**: UniSonate 的核心增量是将 InstructAudio 的 TTS+TTM 统一框架扩展到包含 SFX 的三模态。架构(MM-DiT + CFM, 1.34B 参数)和训练规模(speech 50K h + music 20K h)完全相同,新增 1.5M SFX clips。这使得 UniSonate 更像 InstructAudio 的"v2"而非独立新框架。

**正向迁移是最有价值的发现**: 消融实验 [Table 6, 7] 清楚地表明联合训练不仅没有负迁移,反而显著提升了各单任务性能。这对统一音频生成方向具有指导意义——多样化的声学数据帮助共享编码器学到更鲁棒的特征。但论文未能解释为什么课程学习能实现正向迁移,而直接联合训练会导致负迁移(缺少不使用课程学习的消融)。

**动态 token 注入的巧妙与局限**: 用 phoneme-to-duration 比率将 SFX 时长映射为 token 序列长度是一个优雅的工程方案,使得无需修改架构即可支持新模态。但代价是 SFX 的 content 表示极为贫乏——所有语义信息压缩到 instruction 中的 NL description,content 序列仅编码时长信息。TTA 指标落后专用模型(FAD 差 2x)可能部分源于此。

**评估的局限**: TTS 用 Seed-TTS 测试集(标准),但 TTM 用自建 500-sample 测试集,TTA 用 AudioCaps。SFX 评估缺少主观 MOS。SongEval 是较新的 benchmark,DiffRhythm+ 在短时长上可能不利(论文承认做了截断 [§A.1])。

## 可复用的 idea

1. **动态 token 注入模式**: 当需要将缺乏内在结构的模态(如 noise, SFX, ambient)融入序列建模架构时,可用领域相关的密度比率将目标时长映射为伪 token 序列长度。这个 pattern 可推广到任何 phoneme-driven TTS 系统需要支持非语言音频的场景。

2. **课程学习的结构化→非结构化渐进策略**: 多模态联合训练时,按模态的"结构化程度"从高到低渐进引入,可以先锚定精确对齐能力再泛化。这个策略在其他多模态系统(如 text+image+video)中也可能适用。

3. **正向迁移作为统一建模的验证信号**: 如果联合训练后各单任务都提升,说明统一架构有效;如果某个任务下降,说明架构/训练策略需要改进。消融实验设计值得参考。
