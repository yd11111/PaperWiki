---
type: paper-note
title: "When Large Language Models Meet Speech: A Survey on Integration Approaches"
authors: [Zhengdong Yang, Shuichiro Shimizu, Yahan Yu, Chenhui Chu]
year: 2025
venue: "arXiv:2502.19548v2"
arxiv_id: "2502.19548"
tier: card
tags: [survey, speech-LLM, integration, text-based, latent-representation, audio-token, ASR, TTS, S2TT, modality-adaptation]
concepts: ["[[Speech-LLM Integration Taxonomy]]", "[[Modality Adaptation for Speech LLM]]", "[[LLM-enhanced ASR]]"]
created: 2026-06-02
updated: 2026-06-02
---

## 概要

首篇专门聚焦 **语音与 LLM 的集成方法** 的 survey。与 Cui et al. (2024) 从统一 SpeechLM 模型视角不同,本文从 **集成接口** 角度出发,将现有方法分为三大类: text-based, latent-representation-based, audio-token-based integration。覆盖了各方法在 ASR、S2TT、S2ST、TTS 等任务上的对比分析,并总结了 6 大开放挑战。

## 核心贡献

1. **三分类 taxonomy** [Fig 1, Fig 2]: Text-based (cascaded / rescoring / GER) | Latent-representation-based (Conv downsampling / CTC compression / Q-Former) | Audio-token-based (semantic / acoustic / both)
2. **Modality Adaptation 详述** [§4.2, Fig 4]: 首次系统对比三种 adapter 架构 (Conv / CTC / Q-Former) 及训练策略 (full FT / LoRA / adapter-only / two-stage)
3. **跨方法定量对比** [Table 1]: 在 ASR (LibriSpeech, Fleurs, AISHELL-2, VoxPopuli), S2TT (CoVoST2), S2ST (CVSS), TTS (LibriTTS) 上对比不同集成方法
4. **6 大开放挑战** [§7]: text-based 的副语言信息丢失, latent 的表征对齐, audio-token 的语义-声学 gap, fair comparison 缺乏, multilingualism, real-time processing

## 与 Cui et al. 2024 (SpeechLM Survey) 的区别

| 维度 | Cui et al. 2024 | Yang et al. 2025 (本文) |
|------|----------------|----------------------|
| 视角 | Model-centric (统一 SpeechLM) | Interface-centric (集成方法) |
| 分类轴 | Features modeled + Training stages + Generation paradigm | Integration approach (text / latent / audio-token) |
| 覆盖范围 | SpeechLM 内部架构 + 下游任务 + 评估 | 集成接口 + Adapter 架构 + 训练策略 + 跨方法对比 |
| 独有内容 | 全双工/IPR/Post-alignment | LLM Rescoring/GER, Modality Adaptation, 定量对比表 |

## 知识提取成果

### 新建概念页 (3 个)

| 概念 | 核心内容 |
|------|----------|
| [[Speech-LLM Integration Taxonomy]] | 三分类框架 (text / latent / audio-token), 优劣对比, 适用场景 |
| [[Modality Adaptation for Speech LLM]] | Conv downsampling, CTC compression, Q-Former; LoRA/PEFT 训练策略 |
| [[LLM-enhanced ASR]] | LLM Rescoring (N-best 重排序) + LLM GER (生成式纠错), HyPoradise/H2T |

### 更新概念页 (3 个)

| 概念 | 更新内容 |
|------|----------|
| [[Speech Language Model]] | 补充 integration taxonomy 互补视角, 增加 Yang et al. 引用 |
| [[Speech-Text Alignment]] | 补充训练策略 (LoRA/PEFT/两阶段训练) |
| [[LLM-based TTS]] | 补充 integration 视角下的 TTS 路线对比 + 计算代价分析 |

## Survey 关键分类 [Fig 2]

```
Speech-LLM Integration
├── Text-based [§3]
│   ├── Cascaded Integration → AudioGPT, HuggingGPT
│   ├── LLM Rescoring → Chen 2023c, Udagawa 2022
│   └── LLM GER → HyPoradise, Whispering Llama, MMGER
├── Latent-representation-based [§4]
│   ├── Conv Downsampling → BLSP, SALM, SpeechVerse, SLAM-ASR
│   ├── CTC Compression → Hono 2023, Speech-Llama
│   ├── Q-Former → SALMONN, COSMIC, XLLM, DesSTA
│   └── Other → SLM, Qwen-Audio, Qwen2-Audio, LLaST, BESTOW
└── Audio-token-based [§5]
    ├── Semantic Tokens → VoxtLM, SpeechGPT, TWIST, Spirit-LM
    ├── Acoustic Tokens → VALL-E, LauraGPT, Neekhara et al.
    └── Semantic + Acoustic → AudioPaLM, Moshi, Hibiki, Emova
```

## 关键发现

- 集成深度排序: Latent-representation > Audio-token > Text-based [§6.1]
- Q-Former > CTC Compression > Conv Downsampling (adapter 效果排序) [§4.2]
- LoRA for LLM 显著优于冻结 LLM [§4.3, Pham et al., 2024]
- PEFT 在 GER 中可能优于 full fine-tuning (防止过拟合) [§3.3, Chen et al., 2023a]
- Fair comparison 是最大瓶颈: backbone LLM/数据量/训练策略变量太多 [§7.4]
