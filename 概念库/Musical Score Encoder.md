---
type: concept
title: "Musical Score Encoder"
aliases: [乐谱编码器, Score Encoder, Score Conditioning, Music Score Input, 乐谱条件化]
category: "architecture-component"
tags: [SVS, singing, score, MIDI, alignment, content-representation]
key_papers: ["[[论文笔记/Survey-Synthetic Singers|Synthetic Singers (Pan et al., 2026)]]", "[[论文笔记/TechSinger|TechSinger]]"]
origin_paper: "Pan et al., Synthetic Singers: A Review of Deep-Learning-based SVS Approaches, 2026"
related_concepts: ["[[Singing Voice Synthesis]]", "[[Duration Predictor]]", "[[Phoneme Representation]]", "[[F0 Modeling]]", "[[Speech-Text Alignment]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Musical Score Encoder 是 SVS 系统中将乐谱信息 (MIDI pitch, note boundary, duration, meter) 与歌词 (lyrics) 融合编码为模型输入的模块。它回答两个问题: "唱什么" (what to sing) 和 "何时唱" (when to sing),是 SVS 区别于 TTS 的核心输入侧组件 [Pan et al., 2026, §4.1]。

**与 TTS Text Encoder 的本质差异**:

| 维度 | TTS Text Encoder | Musical Score Encoder |
|------|------------------|----------------------|
| 输入 | 文本/音素序列 | 音素 + MIDI pitch + note duration + note boundary |
| 对齐 | 音素→帧 (单调) | 音符→音素→帧 (含 melisma) |
| 时长来源 | 数据驱动预测 | 乐谱显式指定 + 局部细化 |
| 音高信息 | 无 (由韵律模块处理) | 乐谱直接提供目标 MIDI pitch |

## 输入表示

### 标准管线
```
歌词文本 → [G2P] → 音素序列
                         ↓
乐谱 MIDI → pitch (音高), note boundary (音符边界), duration (时值), meter (拍号)
                         ↓
        [融合] → 音素 + 乐谱联合表示 → Content Encoder
```

G2P 生成音素序列后,与乐谱中的 MIDI pitch、音符时值、音符边界等信息融合 (Lu et al., 2020; Liu et al., 2022a)。乐谱信息通常以嵌入向量形式与音素嵌入拼接或相加。

### Melisma 问题
Melisma (一音多音节/花腔) 是 SVS 独有的对齐难题: 一个音节可能跨越多个不同音高的音符,或一个音符内包含多个音素,形成一对多映射。TTS 中不存在此问题。

## 三种对齐方式 [§4.1]

### 1. 外部强制对齐 (External Forced Alignment)
使用 MFA (Montreal Forced Aligner) 或 Praat 等工具预计算音素/音节时长标签。
- 优势: 简单可靠,适合级联系统
- 局限: 依赖外部工具质量,无法端到端优化

### 2. 可学习单调对齐 (Learnable Monotonic Alignment)
随机时长预测 (stochastic duration prediction) 建模节奏不确定性 (Zhang et al., 2022b)。
- 保持单调性约束 (不回退)
- 支持端到端训练
- 代表: VISinger 系列

### 3. 可学习上采样/长度调节器 (Learnable Up-sampling / Length Regulator)
将 token 级状态扩展至帧级 (He et al., 2023; Zhang et al., 2024b)。
- 与 FastSpeech length regulator 类似但接受乐谱时长约束
- 支持端到端优化

### 对齐质量改进 [§4.1]
近期工作持续改进对齐质量:
- **RL-based 优化**: 引入感知目标改善对齐 (Li et al., 2025)
- **Masked-token 表示**: 稳定单调对齐 (Zhang et al., 2025b)
- 对齐质量直接影响 SVS 的鲁棒性和自然度 (Jiang et al., 2025)

## 标注工具生态 [§5.2, Fig 3]

SVS 训练数据至少需要: 歌词文本 + 音素时长 + 音符信息。三类标注任务:

| 标注任务 | 工具 | 方法 |
|----------|------|------|
| Audio-Text Alignment (音频文本对齐) | Praat, MFA, SOFA (CTC-loss) | HMM/CTC 强制对齐 |
| Note Transcription (音符转写) | Parselmouth, ROSVOT (Conformer), MusicYOLO | 音高/时长提取 |
| Style Detection (风格检测) | STARS (Guo et al., 2025c) | 端到端多任务标注 |

## 在 SVS 中的应用

### 级联系统
Content Encoder 输出条件化 Acoustic Model (音高+时长约束):
- XiaoiceSing: FastSpeech encoder + MIDI pitch embedding
- DiffSinger: 音素+乐谱信息条件化浅扩散过程
- TechSinger: 多尺度条件化 + 技巧局部编码

### 端到端系统
Content Encoder 直接条件化波形生成器:
- VISinger: 音素+乐谱→VITS prior encoder
- SiFiSinger: 音素+乐谱→源滤波器 F0 控制信号

## 关键论文

- XiaoiceSing (Lu et al., 2020): 首个高质量集成 SVS 系统,定义标准 G2P+score 管线
- DiffSinger (Liu et al., 2022a): 乐谱条件化浅扩散,成为级联系统标杆
- VISinger (Zhang et al., 2022b): 端到端对齐中引入可学习单调对齐
- SinTechSVS (Zhao et al., 2024): 注意力局部乐谱模块编码歌唱技巧

## 相关概念

- [[Singing Voice Synthesis]]: Musical Score Encoder 服务的核心任务
- [[Duration Predictor]]: TTS 中的对应组件,SVS 中由乐谱部分替代
- [[Phoneme Representation]]: 共享的音素表示层
- [[F0 Modeling]]: 乐谱提供的 MIDI pitch 是 F0 建模的目标

## 演进

手工乐谱规则 (VOCALOID, 2004) → G2P+MIDI 嵌入融合 (XiaoiceSing, 2020) → 浅扩散条件化 (DiffSinger, 2022) → 可学习单调对齐 (VISinger, 2022) → 注意力局部乐谱技巧编码 (SinTechSVS, 2024) → RL 优化对齐 + masked-token 稳定化 (2025)
