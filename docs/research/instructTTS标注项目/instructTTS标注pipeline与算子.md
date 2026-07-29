---
title: "instruct TTS 标注 pipeline 与算子 v0.1"
created: 2026-07-28
tags: [research, tts, instruct, annotation, pipeline, operators]
---

# instruct TTS 标注 pipeline 与算子 v0.1

> 配套 [[instructTTS成品数据格式规范]] / [[instructTTS标注方案设计]]。回答:成品 JSON 的**每个字段由哪个算子、在哪个阶段、吃什么输入填进去**,并给出完整数据 pipeline。
> **可视化流程图见同目录 `instructTTS_pipeline.html`(浏览器打开)。**

## 0. 总原则

- 挂在现有 **VAD / ASR(词级时间戳)/ 说话人分离** 之后,输入 = 几十秒片段 + ASR 文本;
- **两阶段**:第一阶段产结构化标签(`labels`),第二阶段产自然语言(`nl`);
- 阶段内按依赖排序,能并行的并行;**中性门**在情感首步就挡掉 ~90% 中性数据,省 Omni 算力。

## 1. 字段 → 算子映射(每个位置怎么填)

| 成品字段 | 输入 | 算子 | 阶段 |
|---|---|---|---|
| `text` | 音频 | (上游 ASR,已有) | 上游 |
| `word_prosody[]` | 音频 + 文本 | **WordVoice-5A**(MFA+Qwen3FA 对齐 + F0/energy/duration/boundary/tone)| S1 |
| `prosody.{pitch_level,pitch_range,energy,rate,pitch_tone}` | word_prosody | **分位档聚合器**(pilot 定阈值,pitch 性别内归一)| S1 |
| `labels.gender` / `labels.age` | 音频段 | **开源分类器 checkpoint**(gender/age)| S1 |
| `environment` / `quality.*` | 音频段 | **DNSMOS + SNR + Silero VAD** | S1 |
| `emotion.l1_base` / `is_expressive` | 音频段 | **SER**(emotion2vec+ / SenseVoice)+ 中性门 | S1 |
| `register`(text_pred)| ASR 文本 | **text-LLM 7 类分类**(兼质量门,剔断句破碎/其他无效)| S1 |
| `register`(audio_pred 校验)| 音频段 | **audio-OmniLLM 复核** | S1 |
| `emotion.l2_fine` / `intensity` | 音频 + 文本 + L1 | **audio-OmniLLM**(受控清单选类)| S1 |
| `text_challenges` | 文本 | **规则/正则引擎** | S1 |
| `cross_validation.*` | 上述双标结果 | **交叉验证 reconciler**(一致/不一致处置)| S1 末 |
| `nl.l3_emotion_desc` | 文本 + L1 + L2 | **text-LLM**(以标签为锚)| S2 |
| `nl.global_desc` / `instructions[]` | 全部结构化标签 + L3 | **text-LLM**(四阶段 prompt programming)| S2 |
| `context.*` | 片段元信息 | (上游 分离/切分,已有) | 上游 |
| `labels_phase2.*` | — | (第二期算子,现留 null)| 二期 |

## 2. 算子清单

| # | 算子 | 类型 | 输入 → 输出 | 状态 |
|---|---|---|---|---|
| 0 | VAD / ASR / 说话人分离 | 上游 | 原始音频 → 片段 + 词级文本 | **已有** |
| 1 | **WordVoice-5A** | 内部 DSP | 音频+文本 → 词级 Duration/Boundary/Energy/Pitch/Tone | **已有** |
| 2 | 分位档聚合器 | 规则 | word_prosody → 句级 5/3 档 | 待建(轻)|
| 3 | gender/age 分类器 | 开源 ckpt | 音频 → gender/age | 选型待定(许可缺口)|
| 4 | SER | 开源 ckpt | 音频 → 情感主类 + 中性门 | 选型待定(emotion2vec+/SenseVoice)|
| 5 | 质量算子 | 开源 | 音频 → DNSMOS/SNR/环境 | 现成 |
| 6 | register text-LLM | 文本 LLM | 文本 → 7 类 + 质量门 | prompt 已定稿 |
| 7 | register audio 校验 | audio-Omni | 音频 → register 复核 | 待接 |
| 8 | emotion L2 | audio-Omni | 音频+文本+L1 → 细类+强度 | 待接(Part C 定配方)|
| 9 | text_challenges | 规则 | 文本 → 挑战标签 | 待建(轻)|
| 10 | 交叉验证 reconciler | 逻辑 | 双标 → 一致性 + 处置 | 待建 |
| 11 | L3 描述 | 文本 LLM | 文本+L1+L2 → 自由描述 | 待建(prompt)|
| 12 | global_desc/指令扩写 | 文本 LLM | 全标签+L3 → NL 指令 | 待建(四阶段 prompt)|

**用到的模型/工具类别**(仅新增):内部 WordVoice-5A、开源 gender/age ckpt、SER(emotion2vec+/SenseVoice)、audio-OmniLLM(Qwen3-Omni)、text-LLM(Qwen3-32B/GPT)、DNSMOS/SNR/VAD、规则引擎。

## 3. Pipeline 流程(执行顺序)

```
上游(已有):原始音频 → VAD → ASR(词级) → 说话人分离 → 几十秒片段 + 文本
                                    │
        ┌───────────────────────────┼───────── 第一阶段 · 结构化标注 ─────────────────┐
        │                                                                              │
   [并行组]                                                                            │
   ① WordVoice-5A → 分位聚合 → prosody + word_prosody                                  │
   ② gender/age 分类器 → gender/age                                                    │
   ③ 质量算子(DNSMOS/SNR/VAD) → environment/quality  ──(pass=false 剔除)              │
   ④ SER → emotion.l1_base + 中性门 ──(中性→归档,不进 L2)                            │
   ⑤ register: text-LLM 主判 ──(断句破碎/其他无效→整条剔除)→ audio-Omni 校验          │
   ⑥ text_challenges 规则                                                             │
        │                                                                              │
   [依赖 ④]                                                                            │
   ⑦ emotion L2(audio-Omni:音频+文本+L1)→ l2_fine + intensity                       │
        │                                                                              │
   [汇总]                                                                              │
   ⑧ 交叉验证 reconciler(emotion: SER vs Omni roll-up;register: text vs audio)       │
        │                                                                              │
        └──────────────────────────────────────────────────────────────────────────┘
                                    │  结构化标签就绪
        ┌───────────────────────────┼───────── 第二阶段 · 自然语言扩写 ───────────────┐
   ⑨ L3(text-LLM:文本+L1+L2)→ l3_emotion_desc                                       │
   ⑩ global_desc + instructions(text-LLM 四阶段 prompt)                              │
        └──────────────────────────────────────────────────────────────────────────┘
                                    │
                              成品 JSON(见规范)
```

**依赖要点**:⑦ 依赖 ④(L1);⑧ 依赖 ④⑤⑦;第二阶段依赖第一阶段全部。①②③⑤⑥ 可并行。质量门(③)与质量文本门(⑤)尽早剔除坏数据,省下游算力。

## 4. 交叉验证节点(详见 [[instructTTS工具验证与交叉验证方案]])

- **emotion**:SER 主类 vs Omni 细类 roll-up → 一致入库 / 不一致(人工/加权/丢弃);
- **register**:text-LLM vs audio-Omni → 不一致以语音为准或人工;
- 结果落入 `annotation_meta.cross_validation`。

## 5. 待办(算子侧)

1. 定 SER + gender/age checkpoint(见验证方案 Part A);
2. 分位聚合器 + text_challenges 规则(轻量,先建);
3. Omni L2 / L3 / global_desc 的 prompt(Part C 定 Omni 输入配方后写);
4. 交叉验证 reconciler 逻辑(阈值来自验证方案 Part B)。
