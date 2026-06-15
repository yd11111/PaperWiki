# Session 5: 国际大厂 — Voxtral + Meta + Google + OpenAI

> 调研日期: 2026-06-04

---

## 团队1: Mistral AI / Voxtral

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | Mistral AI (法国巴黎, 2023年成立) |
| **GitHub** | github.com/mistralai (25 repos, 无专门音频repo) |
| **HuggingFace** | mistralai |
| **融资** | 2025年多轮融资, 估值超60亿美元 (Microsoft/NVIDIA投资) |

### 语音技术发布时间线

| 时间 | 事件 | 说明 |
|------|------|------|
| 2025.07 | **Voxtral Mini/Small** | 多模态音频理解, Apache 2.0, 40分钟音频, 106位作者 |
| 2026.02 | **Voxtral Realtime** | 原生流式ASR, Apache 2.0, 480ms延迟, 13语言, 169位作者 |
| 2026.03 | **Voxtral TTS** | 表现力多语言TTS, CC BY-NC, AR语义+FM声学混合架构, 自研Voxtral Codec (VQ-FSQ混合量化), 3秒克隆, **人评胜率68.4% vs ElevenLabs**, 189位作者 |

### 技术栈

- **Codec**: Voxtral Codec — VQ-FSQ混合量化方案
- **TTS架构**: AR生成语义token + Flow-matching生成声学token (主流两阶段)
- **ASR**: Delayed Streams Modeling + Causal Audio Encoder (原生流式)
- **底座**: Mistral LLM backbone

### 开源策略

**激进开源**: 理解模型Apache 2.0, TTS模型CC BY-NC, 全部提供HuggingFace权重

### 判断

语音是Mistral从LLM向多模态扩展的核心路径。不到一年完成理解-ASR-TTS三件套。团队从106到189人快速扩张。**开源TTS质量已可挑战ElevenLabs。**

---

## 团队2: Meta / FAIR

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | Meta Platforms (FAIR实验室) |
| **GitHub** | facebookresearch (6个speech/audio repo, **4个已archived**) |
| **产品** | AudioCraft生态 + Seamless翻译 + Audiobox生成 |

### 语音技术时间线

| 时间 | 事件 | 说明 |
|------|------|------|
| 2023.06 | **Voicebox** | Flow-matching TTS, 50K小时, 零样本, NeurIPS 2023 |
| 2023.06 | **AudioCraft** 开源 | MusicGen+AudioGen+EnCodec, MIT, **23.3K stars** |
| 2023.12 | **Seamless系列** | 100+语言S2S/S2T/T2S/T2T |
| 2023.12 | **Audiobox** | 统一flow-matching音频生成 |
| 2024.02 | **Spirit LM** | 文本-语音交替多模态, 7B |
| 2025.02 | **Audiobox Aesthetics** | 统一音频质量评估 |
| 2025.07 | **收购PlayHT** | PlayHT平台2025.12关闭 |

### 判断

**2023年达到巅峰后进入整合期。** 一年四大系统(Voicebox/AudioCraft/SeamlessM4T/Audiobox), 但2024-2026明显放缓。重心转向Llama, 语音能力内化到产品。收购PlayHT获取商业TTS比自研更快。**6个speech repo中4个已archived。**

---

## 团队3: Google DeepMind

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | Google DeepMind + Google Labs |
| **GitHub** | google-deepmind (**无公开speech/TTS repo**) |
| **产品** | Gemini全栈 + Lyria音乐 + NotebookLM |

### 语音技术时间线

| 时间 | 事件 | 说明 |
|------|------|------|
| 2023.05 | **SoundStorm** | 非自回归并行音频生成, 30秒音频0.5秒生成 |
| 2023.06 | **AudioPaLM** | PaLM-2+AudioLM融合 |
| 2024.09 | **NotebookLM Audio** | 文档转播客双人对话 (用SoundStorm-based技术) |
| 2025.Q3 | **Gemini 2.5** | 原生音频理解, 32 tokens/秒, 9.5小时音频 |
| 2026.04 | **Gemini 3.1 Flash TTS** | 专用TTS, 30预设声音, 72+语言, **Audio Tags细粒度控制** |
| 2026.04 | **Gemini 3.1 Flash Live** | 实时语音对话, 低延迟 |
| 2026 | **Lyria 3** | 3分钟全曲音乐生成, 真实人声 |

### 技术栈

- **TTS**: Gemini Flash TTS — 支持Audio Tags系统(自然语言内联控制, 类SSML但更灵活)
- **实时对话**: Gemini Live — 低延迟全双工
- **音乐**: Lyria 3
- **安全**: SynthID — 所有AI生成音频含不可感知水印

### 定价

| 产品 | 价格 |
|------|------|
| Gemini 3.1 Flash TTS | 文本$1/M tokens, 音频$20/M tokens |
| Gemini 3.1 Flash Live | 音频输入$3/M tokens, 输出$12/M tokens |
| Lyria 3 Pro | $0.08/首 |

### 判断

**投入最大, 产品线最完整。** 语音深度嵌入Gemini原生多模态体系。Audio Tags系统是创新方向。**但高度闭源, GitHub无speech repo, 学术社区无法使用。** 战略投资Hume AI补充情感语音。

---

## 团队4: OpenAI

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | OpenAI |
| **GitHub** | openai (Whisper, **101K stars**) |
| **产品** | Whisper(ASR) + GPT-4o Voice Mode + TTS API + Realtime API |

### 语音技术时间线

| 时间 | 事件 | 说明 |
|------|------|------|
| 2022.09 | **Whisper** | 通用ASR, MIT开源, 多语言, 101K stars |
| 2024.05 | **GPT-4o** | 原生多模态, 端到端语音对话 |
| 2024.10 | **Realtime API** | 开发者实时语音对话API |
| 2025.Q3 | **GPT-4o-Transcribe** | 专用转录模型, 2.2% WER |

### 技术栈

完全不透明。GPT-4o Voice Mode架构从未公开, 无论文, 无技术博客。Whisper是唯一公开的技术贡献。

### 判断

**定义了端到端语音AI的产品范式, 但技术透明度最低。** GPT-4o Voice Mode催生了整个行业的追赶(CosyVoice/GLM-4-Voice/Qwen-Omni等)。Whisper(101K stars)仍是最大开源贡献。

---

## 横向对比矩阵

| 维度 | Mistral/Voxtral | Meta/FAIR | Google/Gemini | OpenAI |
|------|:---:|:---:|:---:|:---:|
| **ASR** | Voxtral Realtime (13语言) | SeamlessM4T (100+语言) | Gemini原生 | Whisper (99语言) |
| **TTS** | Voxtral TTS (FM) | Voicebox (未完全开源) | Gemini Flash TTS (72+语言) | tts-1/tts-1-hd |
| **语音克隆** | 3秒零样本 | Voicebox零样本 | 未明确 | 未提供 |
| **实时对话** | 未发布 | 未发布 | Gemini Live | Realtime API |
| **音乐生成** | 无 | MusicGen/AudioGen | Lyria 3 | 无 |
| **开源程度** | ★★★★★ (激进) | ★★★ (选择性) | ★ (完全闭源) | ★★ (Whisper后闭源) |
| **技术透明度** | 高 | 中 | 低 | 最低 |
| **战略优先级** | Tier 2 (核心扩展) | Tier 3 (降级整合) | Tier 1 (核心战略) | Tier 1 (核心差异化) |
| **2024-2026活跃度** | ★★★★★ (最快) | ★★ (放缓) | ★★★★ (持续) | ★★★ (产品侧) |

### 关键人才流动

| 人物 | 从→到 |
|------|------|
| Neil Zeghidour | Google → Kyutai/Moshi |
| Alexander H. Liu | Meta/FAIR → Mistral/Voxtral |
| Xu Tan | Microsoft → Moonshot AI |

### 范式转移

> **OpenAI用GPT-4o定义了"端到端多模态语音交互"范式, 直接淘汰传统ASR+LLM+TTS级联方案。** Google用Gemini Live和NotebookLM证明产品化可行。Mistral用Voxtral TTS证明开源可挑战商业领导者。Meta在消化已有技术储备。

### 核心结论

**语音AI正在从独立子领域变成多模态LLM的原生能力。** 竞争焦点: 全双工实时对话 + 情感表现力控制 + Audio Tags式自然语言控制。
