---
type: concept
title: "Singing Voice Synthesis"
aliases: [SVS, 歌声合成, 歌唱合成, Singing Synthesis, 歌声生成]
category: "task"
tags: [SVS, singing, music, vocal-synthesis, score-conditioned, pitch-control]
key_papers: ["[[论文笔记/Survey-Synthetic Singers|Synthetic Singers (Pan et al., 2026)]]", "[[论文笔记/Make-A-Voice|Make-A-Voice]]", "[[论文笔记/SongGen|SongGen]]"]
origin_paper: "Pan et al., Synthetic Singers: A Review of Deep-Learning-based SVS Approaches, 2026"
related_concepts: ["[[Text-to-Speech Pipeline]]", "[[Musical Score Encoder]]", "[[F0 Modeling]]", "[[SVS Evaluation Metrics]]", "[[Prosody Modeling]]", "[[Neural Vocoder]]", "[[Style Transfer in TTS]]", "[[Diffusion-based TTS]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Singing Voice Synthesis (SVS) 从文本歌词和符号化乐谱 (musical score) 生成高保真歌声。与 TTS 的根本区别在于: SVS 接受更丰富的输入 (歌词 + 乐谱 + 可选控制信号),并施加更严格的约束 — 合成歌声必须在清晰发声的同时严格遵循指定旋律 [Pan et al., 2026, Abstract]。

**SVS vs TTS 核心差异**:

| 维度 | TTS | SVS |
|------|-----|-----|
| 输入 | 文本/音素 | 歌词 + 乐谱 (MIDI pitch, note boundary, duration, meter) |
| 音高约束 | 语调 (相对, 柔性) | 旋律 (绝对音高, 严格) |
| 时长约束 | 自然节奏 | 节拍/BPM 对齐 |
| 表现力维度 | 情感/风格/韵律 | 歌唱技巧 (vibrato, falsetto, breath) + 情感 |
| 对齐难度 | 音素→帧 (单调) | 音符→帧 (melisma: 一音多音节) |
| 数据稀缺性 | 大规模语音数据可用 | 高质量歌声数据极为有限 [§B.4] |

## 四大任务分类 [§2, Fig 1]

### 1. Hi-Fidelity Synthesis (高保真合成)
基础目标: 从乐谱+歌词生成自然、清晰的歌声。
- 代表: XiaoiceSing, DiffSinger, HiFiSinger, VISinger, RMSSinger

### 2. Controllable Synthesis (可控合成)
在高保真基础上提供对音色、歌唱技巧、情感的细粒度控制。
- 代表: MuSE-SVS (多歌手+情感), SinTechSVS (歌唱技巧), PromptSinger (自然语言控制), TechSinger (CFG 技巧控制)

### 3. Singing Style Transfer (歌唱风格迁移)
将参考歌声的风格 (音色、唱法) 迁移至新歌声,包括 Singing Voice Conversion (SVC) 和 Speech-to-Singing (STS)。
- 代表: TCSinger (RVQ→CVQ 风格压缩+LM), ExpressiveSinger (级联 diffusion 控制), TCSinger2 (对比学习+MoE)

### 4. Text-to-Song Generation (文本到歌曲生成)
从文本直接生成完整歌曲 (歌声+伴奏),是 SVS + 音乐生成的融合任务。
- 代表: JukeBox, SongGen (单阶段 AR Transformer), YuE (LLaMA2+渐进式条件化), DiffRhythm/MuDiT (非自回归), Levo (混合 token+DPO)

## 架构范式 [§3, Fig 2]

### Cascaded (级联)
```
乐谱 + 歌词 → [Content Encoder] → [Acoustic Model] → Mel/F0/UV → [Vocoder] → 波形
```
- Acoustic model 学习 "乐谱→频谱特征" 映射
- 手工中间表示 (Mel, F0, UV) 提供辅助监督,降低学习难度
- 在低资源场景下更稳定 [§B.2]
- 代表: HiFiSinger, DiffSinger, ByteSing, WeSinger, RMSSinger, TechSinger

### End-to-End (端到端)
```
乐谱 + 歌词 → [Content Encoder] → [Generator] → Acoustic Tokens → [Decoder] → 波形
```
- 直接生成波形,避免级联误差累积
- 需要更多数据,对对齐和时序建模更敏感 [§B.2]
- 代表: VISinger (VITS→SVS), VISinger2 (DSP增强), SiFiSinger (F0源滤波器), UniSyn (多条件VAE), CSSinger (半流式VAE)

### 级联 vs 端到端讨论 [§B.2]
与 TTS 不同,SVS 社区仍大量使用级联系统。原因: 级联管线依赖 Mel spectrogram 等手工目标提供辅助监督,在低资源设置下确保更高的性能下限。端到端系统天花板更高但面临数据需求大、训练不稳定、对齐敏感等挑战。

## 核心技术 [§4]

### 内容表示 (Content Representation) [§4.1]
- G2P 音素序列 + 乐谱信息 (MIDI pitch, note boundary, duration) 融合
- 对齐方式: 外部强制对齐 (MFA) / 可学习单调对齐 / 可学习上采样
- 详见 [[Musical Score Encoder]]

### 声学表示 (Acoustic Representation) [§4.1]
- **手工特征**: Mel, F0, UV (级联系统主流)
- **离散 token**: Neural codec token (Encodec/RVQ) — HiddenSinger, TokSing
- **连续 latent**: VAE/RQ-VAE 编码器 — VITS 类系统 (VISinger)
- 详见 [[Semantic vs Acoustic Tokens]]

### 语义表示 (Semantic Representation) [§4.1]
- 早期: 手工频谱+统计特征 (SVM/HMM)
- 深度学习: wav2vec 2.0, HuBERT, WavLM
- 新方向: LLM 接口 (SeCap 用 Q-Former 将语音转为 LLaMA style-aware token)

### 控制技术 [§4.2]
- **音频驱动风格迁移**: 参考音频分离 (content, rhythm, timbre) → 零样本迁移
- **文本驱动风格控制**: PromptSinger (自然语言), SinTechSVS (注意力局部乐谱), 自适应归一化
- **CFG**: 条件生成器 + 推理时调节条件强度 (TechSinger)
- **MoE**: 路由策略选择专用生成专家 (TCSinger2)

## 训练策略 [§A]

### 数据增强 [§A.1]
公开歌声数据集规模有限且标注不一致,数据增强至关重要:
- **乐谱感知 pitch/tempo 变换**: 半音移调 + BPM 比例时间拉伸,同步更新 MIDI/F0/duration
- **频谱扰动**: frequency/time masking, spectrogram-domain mixup
- **F0 扰动**: 添加小 vibrato, 微调 UV 边界 → 跨域泛化
- **语音数据注入**: 利用语音-歌声共享属性扩展风格/韵律覆盖

### 预训练与微调 [§A.2]
- SSL backbone (wav2vec 2.0, HuBERT) 冻结 + SVS head 训练
- PEFT (LoRA 等) 做歌手风格适配
- Domain adapter 桥接语音与歌声

### 多阶段训练 [§A.2]
级联系统: 先训 duration/F0 predictor, 再联合微调声学模型+vocoder, 对齐条件分布

## MLLM 对 SVS 的贡献 [§C]

1. **数据标注**: Whisper/FireRedASR 转写歌词 + GPT-4o/Gemini 分析旋律上下文
2. **内容理解与生成**: 辅助作词、旋律建议、风格感知歌曲结构化
3. **表现力预测**: 文本 prompt 描述歌唱风格/技巧 → 引导生成
4. **歌声与歌曲生成**: YuE, Seed-Music, Suno 等端到端大模型
5. **评估**: MLLM-as-a-judge 评估歌声表现力
6. **歌声理解**: 反向任务 — 从歌声提取语义信息

## 关键论文

- XiaoiceSing (Lu et al., 2020): 高质量集成 SVS 系统,FastSpeech→SVS
- DiffSinger (Liu et al., 2022a): 浅扩散 DDPM 用于 mel 生成,改善频谱细节
- VISinger (Zhang et al., 2022b): 首个将 VITS 引入 SVS 的端到端系统
- RMSSinger (He et al., 2023): 扩散 pitch predictor 提升 F0 自然度
- SiFiSinger (Cui et al., 2024): 源滤波器模型 F0 控制的端到端 SVS
- TCSinger (Zhang et al., 2024c): RVQ 风格迁移 + 多层级风格控制
- PromptSinger (Wang et al., 2024a): 自然语言 prompt 控制歌声
- TechSinger (Guo et al., 2025b): CFG 歌唱技巧控制 + flow matching

## 相关概念

- [[Musical Score Encoder]]: SVS 特有的乐谱输入编码
- [[F0 Modeling]]: 歌声的音高精确控制 (区别于 TTS 韵律)
- [[SVS Evaluation Metrics]]: 歌声专用评估指标
- [[Text-to-Speech Pipeline]]: SVS 早期系统继承的架构
- [[Prosody Modeling]]: 共享概念但 SVS 增加 vibrato/musical rhythm
- [[Neural Vocoder]]: SVS 级联系统使用的波形合成器
- [[Style Transfer in TTS]]: 歌唱风格迁移是其延伸

## 演进

VOCALOID (拼接合成, 2004) → HMM-based SVS (Saino et al., 2006) → XiaoiceSing (FastSpeech→SVS, 2020) → DiffSinger (扩散声学模型, 2022) → VISinger (VITS→SVS 端到端, 2022) → TCSinger (零样本+风格控制, 2024) → TechSinger (歌唱技巧+flow matching, 2025) → Text-to-Song (YuE/Suno, 2025)
