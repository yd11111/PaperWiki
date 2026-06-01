# 综述冷启动计划

> 目标:通过 11 篇 TTS 综述批量丰富概念库,完成知识库的领域认知冷启动。

## 执行原则

- 不为综述建新模板/skill
- 每篇综述产出:概念页的批量创建/更新(主要价值)+ 一个轻量笔记(记录从中提取了什么)
- 概念页更新可以是实质修改(重写定义/补充分类体系)→ status: pending-review
- 一次性工作,不是持续流程

## 执行步骤

### Phase 1: 获取 PDF

对每篇综述:
1. 检查本地是否已有
2. 没有的通过 Playwright 从 arXiv 下载到 Sources/

### Phase 2: T0 骨架综述 (3 篇,建立全局框架)

按顺序处理:

**#1 Xu Tan 2021 — A Survey on Neural Speech Synthesis (2106.15561)**
- 预期产出:TTS 整体分类体系(前端/声学模型/声码器)、演进时间线、核心概念页批量创建
- 更新: 概念页 10-20 个(Mel Spectrogram, Attention-based TTS, Duration Predictor, WaveNet, Tacotron, FastSpeech, Neural Vocoder 等)

**#2 Xie 2024 — Controllable TTS in LLM Era (2412.06602)**
- 预期产出:可控 TTS 分类(风格/情感/韵律/说话人)、LLM-era 新范式、与经典方法的对比
- 更新: 概念页(Prosody Control, Style Transfer, Emotion TTS, LLM-based TTS, Prompt-based Control 等)

**#3 Cui 2024 — Speech Language Models (2410.03751)**
- 预期产出:Speech LM 分类体系(tokenizer 类型/模型架构/训练范式)、与文本 LLM 的类比
- 更新: 概念页(Speech LM, Audio LM, Multimodal LLM, Speech-Text Alignment 等)

### Phase 3: T1 专题综述 (6 篇,按子领域深入)

**#4 Zhang 2023 — Audio Diffusion Models**
- 更新: Diffusion Model, Score Matching, DDPM, Classifier-Free Guidance 在音频中的应用

**#5 Yang 2025 — When LLM Meet Speech**
- 更新: Speech-LLM 接口设计、multi-turn dialogue、instruction following

**#6 Azzuni 2025 — Voice Cloning**
- 更新: Speaker Embedding, Speaker Encoder, Few-shot VC, Zero-shot Voice Cloning 技术路线

**#7 Mousavi 2025 — Discrete Audio Tokens (Codec)**
- 更新: RVQ/FSQ/VQ 对比、codec 在下游生成中的应用、token rate trade-off

**#8 Ji 2024 — WavChat Spoken Dialogue**
- 更新: Full-duplex, Turn-taking, Barge-in, Spoken Dialogue 系统架构

**#9 Pan 2026 — Synthetic Singers (SVS)**
- 更新: SVS vs TTS 差异、歌唱合成特有概念(pitch/vibrato/breath)

### Phase 4: T2 补充 (2 篇)

**#11 Yang 2025 — Responsible TTS Evaluation**
- 更新: 评估方法论、MOS 的局限性、新评估指标

**#14 2025 — Audio-Language Models**
- 更新: Audio understanding、跨模态对齐

## 每篇综述的处理流程

```
① 读 PDF(或 Playwright 获取关键章节)
② 提取:分类体系、概念定义、演进时间线、关键 milestone
③ 对照现有概念页:
   - 已有但浅 → 实质修改(重写定义,补充分类/演进)
   - 不存在 → 新建(status: pending-review)
④ 写一条轻量笔记(论文笔记/xxx.md, tier: card):记录"从这篇综述提取了哪些知识"
⑤ Git commit: [ingest/survey] 综述名 — 更新 N 概念页, 新建 M 概念页
⑥ log.md 追加记录
```

## 预期总产出

- 概念页: 从当前 13 个 → 预计 50-80 个(覆盖 TTS 全领域核心概念)
- 已有概念页: 大部分会被实质重写(从"1 篇论文的浅理解"升级为"综述级系统定义")
- MOC: 可能需要新增 3-5 个子主题 MOC

## 完成标准

- T0 完成后:vault 拥有 TTS 领域的骨架级分类体系,概念页覆盖主要技术路线
- T1 完成后:每个子领域有专题级深度,概念页有完整演进线和方法对比
- 所有新建/重写的概念页 status: pending-review,等你批量确认
