---
type: paper
tier: deep
title: "Toward Natural Emotional TTS with Non-Verbal Expression"
arxiv_id: "2605.25504"
source: "Sources/NaturalEmotionalTTS.pdf"
authors: [Wangzixi Zhou, Bagus Tris Atmaja, Sakriani Sakti]
year: 2026
venue: "IEEE (preprint)"
tags: [TTS, emotion, non-verbal-vocalization, fine-grained-control, diffusion, expressiveness]
concepts: ["[[EmotionControlinTTS]]", "[[Diffusion-basedTTS]]", "[[ProsodyModeling]]", "[[MelSpectrogram]]"]
models: ["[[CosyVoice2]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: 本文属于 Emotional TTS 中的**副语言发声 (Non-Verbal Vocalization) 控制**路线。与 KB 中已有的两条 NV 路线形成对比:

1. **NVSpeech (Liao et al., 2025)**: 在 LLM-based TTS 中通过词表扩展微调实现 18 类 word-level PV 显式控制,基于 CosyVoice2 backbone。控制粒度为离散 token-level 标签 (`[Laughter]`, `[Breathing]` 等)。
2. **NonverbalTTS (Borisov et al., 2025)**: 用 17h 开源 NVTTS 数据微调 CosyVoice-300M,采用粗粒度 `<laugh>` 标签。
3. **EmoCtrl-TTS (Wu et al., 2024)**: 用连续 laughter detector embedding (32 维) 实现帧级 NV 控制,发现可泛化到哭泣等非笑声 NV。

本文走的是**数据标注驱动 + 传统 diffusion TTS backbone** 的路线 — 不依赖 LLM-based TTS 或大规模预训练,而是通过精细标注方案 (频率/时长编码) 让小型 Grad-TTS 学习 NV 生成。这在方法论层面与上述三条路线截然不同。

**已有认知**: 
- Grad-TTS 是基于 SDE 的 diffusion TTS [§Diffusion-basedTTS],使用 WaveGrad U-Net 作为 mel decoder,支持高质量合成 (MOS 4.44, LJSpeech)
- HiFi-GAN 是 GAN-based vocoder 的事实标准,14M 参数,mel → waveform [§NeuralVocoder]
- CosyVoice2 已支持 NV (coughs, sighs, breathing, laughing),但其指令数据集为私有 [§CosyVoice2]
- Emotional TTS 的主流方向已转向 LLM-based / activation steering / reward-guided 路线 [§EmotionControlinTTS]

**创新判断**: 本文的创新不在模型架构 (Grad-TTS + emotion encoder 是成熟方案),而在**数据标注方案** — 提出细粒度 NV 转录系统 (频率控制 via 音节重复 + 时长控制 via 字符重复),这是一个被忽视但实用的贡献。但 backbone 选择 (Grad-TTS) 已显过时,限制了方法的可迁移性。

> 检索命中: [[NeuralVocoder]]✓, [[ProsodyModeling]]✓, [[CosyVoice2]]✓ | 过滤: [[EmotionControlinTTS]][待确认], [[Diffusion-basedTTS]][待确认], [[MelSpectrogram]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出细粒度 NV 标注方案 (频率+时长编码),基于 Grad-TTS+emotion encoder 构建 NV emotional TTS,显著提升表现力 (eMOS 4.20) 和情感识别率 (78.8%)
> - **路线**: EARS 语料 (女声 NV) → 细粒度标注 (style/频率/时长 tag) → NV Processor (style parser + discrete unit parser + duration parser) → NV token → Emotional Grad-TTS (+ arousal-valence 条件) → mel spectrogram → HiFi-GAN → waveform
> - **指标**: eMOS 4.20 (vs verbal-only 3.81), 识别准确率 78.8% (vs verbal-only 65.5%), sad 98.3%, fear 82.7%, happy 82.5% [Fig 2, Table 3]
> - **可借鉴**: 细粒度 NV 标注方案 (音节重复=频率, 字符重复=时长) 可迁移到任何 TTS 系统的 NV 数据构建
> - **局限**: 仅 739 条女声 NV, backbone 为过时的 Grad-TTS (非 LLM-based), 未开源代码/模型, angry 识别率仅 64.3%, naturalness 有轻微下降

## 核心问题

1. **现有 NV 标注为何不够?** 现有数据集 (NVTTS, AMI) 使用粗粒度标签如 `<laugh>`,无法控制 NV 的频率 (笑几次) 和时长 (哭多长),导致 TTS 模型只能"有/无"切换,不能精细调控 [§1]
2. **如何实现频率和时长的精细控制?** 通过标注方案设计: 离散发声 (ha, whep) 用音节重复控制频率,连续发声 (wuu, ah) 用末字符重复控制时长 [§3.2, Table 2]
3. **细粒度标注是否真的比粗粒度更有效?** 是,在 eMOS 和情感识别准确率上全面优于 coarse-grained NV (NVTTS 数据),尤其在 happy 情感上差距巨大 [§5.3, Fig 2]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由两部分组成 [§3-4]:

**Part 1: 数据标注** — 从 EARS 语料库提取 60 位女性说话人的 NV 音频,经 silence-based 分割 (pydub, -40 dBFS, min 200ms) 得到 739 条 2-6 秒片段,再通过 Whisper 初步转录 + 人工修正得到细粒度标注 [§3.1-3.2]

**Part 2: NV Emotional TTS** — 在 Grad-TTS 基础上增加 emotion encoder (arousal-valence 连续标签),并在 text processing pipeline 前端加入 NV Processor [§4]

### 关键设计选择

**1. 细粒度 NV 标注方案 [§3.2, Table 1-2]**

6 种 NV 类别:
| 类别 | 转录类型 | 控制维度 | 数量 |
|------|----------|----------|------|
| Laughter-open | "Ha" (离散) | 频率 (ha ha ha = 3 次) | 266 |
| Laughter-closed | "Ha" (离散) | 频率 | 220 |
| Cheering | "Wo ho" / "Yo" (连续) | 时长 (字符重复) | 262 |
| Yelling | "Hey" (连续) | 时长 | 328 |
| Crying | "Whep" / "Wuu" / "Sneeze" (混合) | 频率+时长 | 230 |
| Screaming | "Ah" (连续) | 时长 | 154 |

核心设计: 
- **离散发声** (如 "ha"): 重复音节控制频率 → `<(Laughter-open) ha ha ha>` = 笑三声 [§3.2]
- **连续发声** (如 "wuu"): 重复末字符控制时长 → `wuuuuu` 中每个额外 "u" 约增加 0.2 秒 [§3.2]

[agent 解读] 这种编码方式巧妙地将 NV 的物理属性 (频率/时长) 映射到文本 token 序列长度上,使标准文本处理 pipeline 无需结构性改动即可建模这些属性。

**2. NV Processor [§4, Fig 1]**

在 text cleaning pipeline 最前端,包含三个子模块:
- **Style Parser**: 识别 NV 类别 (crying, laughter-open 等)
- **Discrete Unit Parser**: 计算离散发声的出现次数
- **Duration Parser**: 根据连续发声字符长度计算时长

输入: `<(crying) wuuuuu whep> why you do this to me`
→ 先解析出 NV 段,提取 style=crying, discrete_count(whep)=1, duration(wuuuuu)=长
→ 生成 NV token → 与文本 token 拼接送入 Grad-TTS [§4]

**3. Emotion 条件 [§4]**

使用 Russell circumplex model 的 arousal-valence 二维连续标签 [§4]。对缺少连续标注的 EXPRESSO 和 ESD 数据,用预训练 SER 模型 (Wagner et al., 2023) 预测 arousal-valence 值 [§5.1]。

[agent 解读] 选择 arousal-valence 而非离散情感标签是合理的,因为 NV 的情感属性天然更适合连续表示 (如 screaming 可以是 fear 也可以是 excitement,取决于 arousal 和 valence 的组合)。

### 训练策略

- **数据构成**: 9 小时混合数据集 (EXPRESSO + SEMAINE + ESD),22.05 kHz,英语女性说话人 [§5.1]
- **声学特征**: 80 维 mel spectrogram [§5.1]
- **Vocoder**: HiFi-GAN [§5.1]
- **训练**: 400k iterations, 单 GPU (NVIDIA RTX A6000, 48GB) [§5.1]
- **对比设置**: 三组 — (1) Only verbal, (2) Verbal + coarse-grained NV (NVTTS 语料), (3) Verbal + fine-grained NV (本文方案) [§5.2.1]

## 实验

| 指标 | Fine-grained NV | Coarse NV | Only Verbal | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| nMOS | ~3.4 (推算) | ~3.4 (推算) | 3.54 | 主观测试 (N=15) | [Fig 2] |
| eMOS | **4.20** | ~3.9 (推算) | 3.81 | 主观测试 (N=15) | [Fig 2] |
| Emotion Acc. (avg) | **78.8%** | - | 65.5% | 四情感识别 | [Fig 2] |
| Happy Acc. | **82.5%** | 低 | 中 | 识别任务 | [Table 3, Fig 3] |
| Sad Acc. | **98.3%** | 高 | 高 | 识别任务 | [Table 3, Fig 3] |
| Fear Acc. | **82.7%** | N/A | ~47% | 识别任务 | [Table 3, Fig 3] |
| Angry Acc. | 64.3% | N/A | 中 | 识别任务 | [Table 3, Fig 3] |

**Emotion-specific 分析 [§5.3.1]**:
- **Happy**: Coarse NV 表现极差,因其 laughter 样本"安静且低沉",被误判为 sad [§5.3.1]
- **Sad**: 三种方法均表现优秀 (>90%),说明 verbal prosody 本身就足以传达悲伤 [§5.3.1]
- **Angry**: 改善有限,因 anger 缺乏独特 NV — yelling 是通用高 arousal 表达,不专属于 anger [§5.3.1]
- **Fear**: 改善最大 (+36%),screaming 是 fear 的高度专属 NV [§5.3.1]

**偏好测试 [§5.3.2, Fig 4]**:
- Happy: cheering ("Wo ho", "Yo") 明显优于 laughter ("ha ha"),62% 参与者将 laughter-closed 排最低 [Fig 4a]
- Sad: 多元素组合 `<(crying) wuuuuuuu whep>` 被 56% 参与者排第一,单一元素 (`whep` 或 `wuuuuuuu`) 排名低 [Fig 4b]

## 局限性

1. **数据规模极小**: 仅 739 条 NV + 9 小时混合数据,远小于 CosyVoice2 的训练规模,限制了泛化能力 [§3.2, §5.1]
2. **仅女性说话人**: 标注和实验均限于女性,性别泛化未验证 [§3.1]
3. **Backbone 过时**: Grad-TTS (2021) 作为 backbone,缺少 LLM-based TTS 的语言理解和 zero-shot 能力,难以与 CosyVoice2/F5-TTS 等现代系统竞争 [agent 解读]
4. **Angry 控制薄弱**: 缺乏 anger-specific NV (yelling 太通用),识别率仅 64.3% [§5.3.1]
5. **Naturalness 下降**: 加入 NV 后 nMOS 从 3.54 降至约 3.4,反映 NV 合成质量尚有提升空间 [Fig 2]
6. **评估规模小**: 仅 15 名参与者,60 个样本,统计效力有限 [§5.2.1]
7. **未开源**: 代码、模型、标注数据均未开源 (仅有 demo 页面) [§1]

## 点评

**优势**:
- 细粒度 NV 标注方案是实用的贡献 — 用音节/字符重复编码频率/时长的方法简单优雅,可迁移到任何 TTS 系统的数据准备阶段
- 实验设计清晰: 三组对比 (verbal-only / coarse / fine-grained) + emotion-specific 分析 + 偏好测试,逻辑完整
- Emotion-specific 分析揭示了 NV 与情感的非均匀映射 (fear/happy 受益大, angry 受益小),这一观察有指导价值

**不足**:
- 方法论贡献偏轻 — NV Processor 本质是规则化的文本预处理,无学习成分; emotion encoder 是标准 embedding 注入
- Backbone 选择 (Grad-TTS) 使整个系统在 2026 的 TTS landscape 中显得过时,无法证明方法在现代 LLM-TTS 上的有效性
- 数据规模和评估规模都太小,难以得出强结论
- 与 NVSpeech (18 类 PV, CosyVoice2 backbone, word-level 控制) 和 EmoCtrl-TTS (连续 embedding, 帧级控制) 相比,本文的技术深度和系统完整性不足

**在领域中的位置**: 本文更像是一篇数据工程/标注方案的 contribution,而非模型或方法论的 breakthrough。其核心价值在于**证明了细粒度 NV 标注对情感识别的显著提升效果**,以及**揭示了不同情感对 NV 的依赖程度差异**。这些实证发现对后续工作 (如在 LLM-based TTS 中设计更精细的 NV 控制) 有参考价值。

## 可复用的 idea

1. **频率-时长编码标注方案**: 用音节重复编码离散 NV 频率 + 字符重复编码连续 NV 时长,可应用于任何 NV 数据集的标注流程
2. **NV-情感非均匀映射**: Fear 和 happy 最依赖 NV (准确率提升 36% 和 17%),angry 最不依赖 (缺乏专属 NV) — 在设计情感 TTS 时可优先为高依赖情感配置 NV 资源
3. **多元素组合 > 单一元素**: Sad 情感中 `wuuuuuuu whep` (连续+离散组合) 优于任一单独元素,说明 NV 组合的丰富度对表现力至关重要
4. **Cheering > Laughter for happy**: 在 TTS 场景中 cheering 比 laughter 更能传达快乐,颠覆直觉 — 可能因 laughter 在 TTS 合成中更难保真
