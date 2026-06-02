---
type: paper-note
title: "Survey: Synthetic Singers — A Review of Deep-Learning-based SVS Approaches"
authors: [Changhao Pan, Dongyu Yao, Yu Zhang, Wenxiang Guo, Jingyu Lu, Zhiyuan Zhu, Zhou Zhao]
affiliation: Zhejiang University
year: 2026
venue: "arXiv:2601.13910"
arxiv_id: "2601.13910"
tier: card
tags: [survey, SVS, singing-voice-synthesis, singing, music, pitch-control, score-conditioning, vocal-synthesis]
source: "/Users/xiangshu/PaperWiki/Sources/Survey-SVS-2026.pdf"
created: 2026-06-02
updated: 2026-06-02
---

## 概要

Pan et al. 首篇系统综述深度学习歌声合成 (SVS) 方法,覆盖任务分类、架构范式、核心建模技术、控制方法、数据集、标注工具和评估体系。Survey 将 SVS 任务分为四类 (高保真合成/可控合成/歌唱风格迁移/文本到歌曲生成),将架构分为级联和端到端两大范式,并深入分析内容表示、声学表示、语义表示三类 SVS 表示和音频/文本驱动两类控制技术。附录覆盖训练策略、推理加速和 MLLM 对 SVS 的贡献。

> [!summary] 速查
> - **一句话**: 首篇全面 SVS 综述,系统梳理任务→架构→建模→控制→评估全链路
> - **路线**: 乐谱+歌词 → Content Encoder → Acoustic Model/Generator → Vocoder/Decoder → 歌声波形
> - **指标**: FFE, F0 RMSE, MCD, MOS, SingMOS, MOS-S, MOS-C
> - **可借鉴**: SVS 的乐谱条件化和 F0 精确控制技术可迁移至 expressive TTS
> - **局限**: 主要聚焦 DL 方法,不涉及传统拼接/统计参数合成; 端到端 vs 级联的实证对比有限

## 核心贡献

1. **任务四分类** [§2]: Hi-Fidelity / Controllable / Style Transfer / Text-to-Song
2. **架构二分法** [§3]: 级联 (acoustic model + vocoder) vs 端到端 (直接生成波形)
3. **三类表示** [§4.1]: Content (乐谱+歌词) / Acoustic (Mel/token/latent) / Semantic (SSL/LLM)
4. **两类控制** [§4.2]: Audio-based style transfer / Text-based style control
5. **评估体系** [§5.3]: 五维度 (Accuracy / Expressiveness / Quality / Similarity / Controllability)
6. **MLLM 视角** [§C]: 六方面贡献 (标注/理解/表现力/生成/评估/歌声理解)

## 知识提取记录

### 新建概念页 (4)

| 概念页 | 来源章节 | 核心内容 |
|--------|----------|----------|
| [[Singing Voice Synthesis]] | 全文 | SVS vs TTS 差异, 四大任务, 两大架构, 核心技术 |
| [[Musical Score Encoder]] | §4.1 | 乐谱编码, G2P+MIDI 融合, melisma, 三种对齐方式 |
| [[F0 Modeling]] | §2, §4.1, §5.3 | SVS 音高控制, vibrato, diffusion pitch predictor, 源滤波器 |
| [[SVS Evaluation Metrics]] | §5.3 | FFE, F0 RMSE, MCD, SingMOS, MOS-S, MOS-C, AXY |

### 更新概念页 (5)

| 概念页 | 更新内容 |
|--------|----------|
| [[Prosody Modeling]] | 追加 SVS 韵律建模: vibrato, musical rhythm, 歌唱技巧 |
| [[Style Transfer in TTS]] | 追加歌唱风格迁移: SVC, STS, TCSinger, PromptSinger |
| [[Duration Predictor]] | 追加 SVS 时长预测: 乐谱约束, melisma, 三种对齐方式 |
| [[Neural Vocoder]] | 追加 SVS vocoder: task-specific vocoder, WORLD 遗产 |
| [[Diffusion-based TTS]] | 追加 SVS 应用: DiffSinger, RMSSinger diffusion pitch, TechSinger flow matching |

## Survey 关键发现

### SVS vs TTS 核心差异
- SVS 接受更丰富输入 (乐谱 + 歌词) 但施加更严格约束 (音高精确, 节奏对齐)
- SVS 社区仍大量使用级联系统 (与 TTS 端到端主流不同) [§B.2]
- 歌声数据极度稀缺 (<100K 小时 vs 语音 >100K 小时) [§B.4]

### 架构趋势 [§B.2]
- 级联系统在低资源设置下更稳定 (手工中间表示提供辅助监督)
- 端到端系统天花板更高但训练难度大
- Neural codec / continuous latent 作为中间表示正在弥合两者差距

### 开放问题
- Quality vs Controllability trade-off [§B.3]: 过度控制可能损害音频质量
- AR vs NAR [§B.2]: 歌声的序列性特征使 AR 回归关注,但 NAR 推理更快
- Continuous vs Discrete representation [§B.3]: 离散对齐 LLM 但可能丢失细节,连续保真但对齐敏感
