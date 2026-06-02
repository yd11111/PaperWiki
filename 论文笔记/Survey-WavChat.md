---
type: paper-note
title: "WavChat: A Survey of Spoken Dialogue Models"
authors: [Shengpeng Ji, Yifu Chen, Minghui Fang, Jialong Zuo, Jingyu Lu, Hanting Wang, Ziyue Jiang, Long Zhou, Shujie Liu, Xize Cheng, Xiaoda Yang, Zehan Wang, Qian Yang, Jian Li, Yidi Jiang, Jingzhen He, Yunfei Chu, Jin Xu, Zhou Zhao]
year: 2024
venue: "arXiv:2411.13577"
arxiv_id: "2411.13577"
tier: card
tags: [survey, spoken-dialogue, speech-LM, full-duplex, streaming, turn-taking, evaluation, cold-start]
created: 2026-06-02
updated: 2026-06-02
---

## 概要

WavChat 是首篇系统综述 spoken dialogue models 的 survey,涵盖级联 (ASR+LLM+TTS) 和端到端两大范式。核心贡献在于系统梳理了 spoken dialogue 的四大核心技术: (1) 语音表征设计, (2) 训练范式与对齐策略, (3) 流式/双工/交互架构, (4) 训练数据与评估方法。论文提出 11 维度评估框架,汇总 8 个 benchmark,并在每节末尾附有 discussion 指出开放问题。

## 核心贡献

1. **系统分类**: 将 spoken dialogue models 按时间线整理并分为级联和端到端两大类 (Figure 1 & 2)
2. **四大技术模块**: 语音表征 (Section 3)、训练范式 (Section 4)、流式/双工/交互 (Section 5)、数据与评估 (Section 6)
3. **五种架构范式**: Text-output only / Chain-of-Modality / Interleaving text-speech / Parallel generation / Speech-to-speech generation (Figure 5)
4. **四阶段训练流程**: Text LLM pre-training → Modality adaptation → Dialogue fine-tuning → Preference optimization (Figure 6)
5. **11 维度评估框架**: 覆盖 text intelligence, speech quality, streaming latency, speech intelligence, audio/music U&G, multilingual, context learning, interaction, multimodal, security (Table 3)
6. **交互机制分类**: Turn-taking cues / turn-end detection vs prediction / overlap (interruption vs backchannel) (Section 5.2.2)

## Survey 结构

```
Section 2: Overall — 九大功能场景 + 级联 vs E2E 系统概览
Section 3: Representation — 输入侧 (semantic/acoustic) + 输出侧表征 + 4 个 trade-off 讨论
Section 4: Training Paradigm — 5 种对齐架构 + 4 阶段训练 + generation strategies
Section 5: Streaming/Duplex/Interaction — 3 项 streaming 核心技术 + duplex 分级 + 交互机制
Section 6: Data & Evaluation — 4 阶段训练数据 + 11 维度评估 + 8 个 benchmark
Section 7: Conclusion — 开放问题汇总
```

## 知识提取成果

本文用于概念库冷启动,提取 spoken dialogue 领域的交互/流式/评估知识。

### 新建概念页 (3 个)

| 概念 | 核心内容 |
|------|----------|
| [[Turn-taking in Spoken Dialogue]] | 轮次切换三类交互 (interruption/backchannel/normal), 五项交互能力, 级联 vs E2E 实现 |
| [[Streaming Spoken Dialogue]] | 三项核心技术 (causal conv/attention/queue), E2E streaming 系统, 延迟优化汇总 |
| [[Spoken Dialogue Evaluation]] | 11 维度两级评估框架, 8 个 benchmark 对照, 开放问题 |

### 更新概念页 (1 个)

| 概念 | 更新内容 |
|------|----------|
| [[Full-duplex Spoken Dialogue]] | 追加 SyncLLM/Parrot/Freeze-Omni/CleanS2S 系统, 交叉引用新建页 |

## 关键分类体系

### 级联 vs E2E (Figure 2)
```
级联: Speech → [ASR] → Text → [LLM] → Text → [TTS] → Speech
  变体: Speech → [Encoder] → [LLM] → Text → [TTS] → Speech  (直接编码输入)
  
E2E:  Speech → [Tokenizer] → Tokens → [LLM] → Tokens → [Detokenizer] → Speech
  含 text 对齐: 部分系统同时输出 text (semantic tokens + text + acoustic tokens)
```

### 五种对齐架构 (Figure 5)
| 架构 | 特点 | 代表系统 |
|------|------|----------|
| Text-output only | 语音输入→文本输出, 外接 TTS | Qwen-Audio, SALMONN, VITA |
| Chain-of-Modality | 先生成文本再生成语音 | SpeechGPT, EMOVA |
| Interleaving text-speech | 交替 token 训练 | Spirit-LM, USDM |
| Parallel generation | 同时输出 text + speech | PSLM, LLaMA-Omni, Moshi, Mini-Omni |
| Speech-to-speech | 不依赖文本中间表示 | SyncLLM, IntrinsicVoice, Align-SLM |

### Duplex 三级 (Figure 8)
```
Simplex: 单向通信, 固定方向
Half-Duplex: 双向但不同时, walkie-talkie 式
Full-Duplex: 同时双向, 支持 overlap/interrupt/backchannel
```

## 代表系统一览

| 系统 | 类型 | 架构特点 | Duplex | Streaming |
|------|------|----------|--------|-----------|
| AudioGPT | 级联 | ASR+ChatGPT+TTS | Half | No |
| Qwen2-Audio | 级联 | Whisper+Qwen+TTS | Half | No |
| SALMONN | 级联 | Whisper+BEATs+LLM | Half | No |
| dGSLM | E2E | Dual-tower DLM + cross-attention | Full | No |
| SpeechGPT | E2E | Chain-of-Modality, SpeechTokenizer | Half | No |
| Moshi | E2E | RQ-Transformer, Mimi, Inner Monologue | Full | Yes |
| Mini-Omni | E2E | SNAC, delayed parallel decoding | Half→Full | Yes |
| LLaMA-Omni | E2E | Whisper+LLM+NAR CTC decoder | Half | Yes |
| VITA | E2E | Dual-model, state tokens, IPR | Full | Partial |
| IntrinsicVoice | E2E | HuBERT+GroupFormer | Half | Partial |
| SyncLLM | E2E | Time-sync chunks, interleaved tokens | Full | Yes |
| OmniFlatten | E2E | Progressive training, block-by-block | Full | Yes |
| Freeze-Omni | E2E | Chunk-level state (0/1/2), 3-stage | Full | Yes |

## 关键发现

1. **Streaming 三项核心技术**: causal convolution + causal attention + queue management 是 E2E streaming 的架构基础
2. **Turn-taking 三类事件**: interruptions, backchannels, normal turn exchanges — VAD 不足以区分,需更复杂的检测方法
3. **评估严重不足**: 11 维度中 Interaction Capability, Security, Multimodal 等维度无标准 benchmark
4. **Text guidance trade-off**: text-guided 提高质量但增加延迟,parallel 降低延迟但可能牺牲质量
5. **Progressive training 趋势**: OmniFlatten, SyncLLM 等采用渐进式训练从半双工过渡到全双工
6. **RL/DPO 在对话中几乎空白**: 偏好优化在 TTS 已有探索,但在 spoken dialogue 几乎未被应用
