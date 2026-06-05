# Session 3: 国内新锐+游戏/社交 — 京东 + 天工 + Soul + 米哈游

> 调研日期: 2026-06-04

---

## 团队1: 京东 (JD Explore Academy / JD Health International)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | 京东集团 (JD.com Inc.) |
| **研究主体** | JD Explore Academy Speech Lab + JD Health International (jdh-algo) |
| **GitHub** | jdh-algo (10 repos), jea-speech (1 repo, demo only) |
| **HuggingFace** | jdh-algo (JoyTTS-v1, JoyHallo-v1等) |
| **核心人物** | Fan Yu (JoyVoice一作), Jun Zhao / Guoxin Wang (JoyHallo/JoyTTS), Sheng Shi / Xuyang Cao (JoyVASA) |
| **产品线** | 数字人 (JoyHallo/JoyVASA系列) + 对话TTS (JoyVoice/JoyTTS) |

**重要澄清**: HAM-TTS核心作者可能已离开京东体系 (后出现在F5-TTS作者列表, 上交+上海AI Lab)

### 论文时间线 (2024-2026)

| 时间 | 论文 | arXiv | 方向 |
|------|------|-------|------|
| 2026.04 | Hallo-Live (合作, Fudan+) | 2604.23632 | 实时流式数字人 |
| 2025.12 | **JoyVoice** | 2512.19090 | 长上下文多人对话TTS |
| 2025.07 | **JoyTTS** | 2507.02380 | LLM-based语音聊天机器人 |
| 2024.11 | **JoyVASA** | 2411.09209 | 扩散式音驱肖像动画 |
| 2024.09 | **JoyHallo** | 2409.13268 | 中文数字人 (Hallo优化版) |
| 2024.03 | **HAM-TTS** | 2403.05989 | 层次声学建模zero-shot TTS |

### 技术栈

| 维度 | 技术 |
|------|------|
| **Tokenizer** | MM-Tokenizer (12.5Hz, multitask semantic + Mel loss) |
| **生成模型** | AR Transformer + DiT (JoyVoice); MiniCPM-o + CosyVoice2 (JoyTTS) |
| **数据规模** | HAM-TTS 650k hours (合成扩增); JoyTTS 2000 hours对话数据 |
| **特色** | JoyVoice支持8人/5分钟单次生成 (长上下文多人对话) |

### 架构演进

```
HAM-TTS (2024.03, 0.8B, discrete token)
    → JoyVoice (2025.12, 0.5B, E2E Transformer-DiT + MM-Tokenizer)
    → JoyTTS (2025.07, MiniCPM-o + CosyVoice2集成)

JoyHallo (2024.09, SD1.5 UNet)
    → JoyVASA (2024.11, Diffusion Transformer, 含动物)
```

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| JoyVASA | 869 | 扩散肖像动画, MIT |
| JoyHallo | 519 | 中文数字人, MIT, 含训练代码 |
| JoyTTS | 41 | LLM聊天机器人, 含训练代码 |
| JoyVoice | 5 | 仅demo页面, **未开源模型** |

### 判断

- **优势**: 数字人全栈能力; 长上下文多人对话是差异化; 开源态度好 (JoyHallo/JoyTTS含训练代码)
- **短板**: TTS基础模型未开源; 核心研究团队规模偏小; HAM-TTS核心人物可能已流失
- **定位**: 电商客服数字人 + 对话TTS

---

## 团队2: 昆仑万维 / 天工 (Kunlun Inc. / Skywork AI)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | 昆仑万维 (Kunlun Inc.) |
| **研究品牌** | Skywork AI |
| **GitHub** | SkyworkAI (20+ repos) |
| **核心人物** | Yahui Zhou (研究负责人), **Heyang Xue** (MoE-TTS一作, VISinger/Learn2Sing系列), Xuchen Song, Max W. Y. Lam (MusiCoT一作) |
| **产品线** | Mureka (AI音乐平台), SkyReels (视频生成), 天工AI助手 |

### 论文时间线 (2025-2026)

| 时间 | 论文 | arXiv | 方向 |
|------|------|-------|------|
| 2026.02 | **SkyReels-V4** | 2602.21818 | 多模态视频-音频联合生成 |
| 2026.01 | **SkyReels-V3** | 2601.17323 | 音频引导视频生成 |
| 2025.08 | **MoE-TTS** | 2508.11326 | Description-based TTS + MoE |
| 2025.06 | **SkyReels-Audio** | 2506.00830 | 音频驱动数字人 |
| 2025.03 | **MusiCoT** | 2503.19611 | Chain-of-Thought音乐生成 |
| 2025.02 | **SkyReels-A1** | 2502.10841 | 表情肖像动画 |

### 技术栈

| 维度 | 技术 |
|------|------|
| **TTS** | MoE-TTS: Description-based Voice Design, frozen LLM + speech MoE experts |
| **音乐** | MusiCoT: CLAP-based chain-of-thought, AR音乐生成 |
| **视频-音频** | SkyReels-V4: dual-stream MMDiT联合生成 |
| **产品** | Mureka API (歌词/歌曲/BGM一键生成) + MCP server |

### 独特技术赌注

1. **Description-based Voice Design**: 自由文本描述控制TTS, 对标ElevenLabs Voice Design
2. **视频-音频联合生成**: SkyReels-V4的dual-stream MMDiT
3. **音乐AI产品化**: Mureka是少数有完整API+MCP集成的音乐AI平台

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| SkyReels-V2 | 6,996 | 无限长视频生成 |
| SkyReels-V3 | 474 | 多模态视频 |
| Mureka-mcp | 98 | 音乐生成MCP server |
| MoE-TTS | — | **未开源**, 仅demo |

### 判断

- **优势**: 音乐AI产品化最成熟 (Mureka); Description-based TTS技术领先; SkyReels视频生态强大; Heyang Xue有深厚SVS背景
- **短板**: TTS模型未开源; TTS仅1篇纯论文; 语音是视频生态的附属方向
- **定位**: AI音乐创作 + 视频配音

---

## 团队3: Soul App (Soulgate)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | Soulgate Inc. (Soul App) |
| **研究团队** | Soul AI Lab |
| **GitHub** | Soul-AILab (11 repos) |
| **核心人物** | **Xinsheng Wang** (通讯作者, 全线核心, NWPU出身: VISinger/Opencpop), **Shunshun Yin**, **Ming Tao**, **Hanke Xie** |
| **学术合作** | **NWPU** (Lei Xie), **SJTU X-LANCE** (Kai Yu, Xie Chen) |
| **产品线** | Soul App语音功能 (语音聊天/K歌/播客/数字人) |

### 论文时间线 (2025-2026, 20+篇)

| 时间 | 论文 | arXiv | 方向 | 会议 |
|------|------|-------|------|------|
| 2026.06 | **SoulX-Transcriber** | 2606.02400 | 多人语音转录 | |
| 2026.04 | **MINT-Bench** | 2604.17958 | 多语言TTS benchmark | |
| 2026.03 | **Joint Speaker Diarization** | 2603.25377 | 说话人分割+识别 | |
| 2026.03 | **OmniCodec** | 2603.20638 | 通用音频编解码器 | |
| 2026.03 | **SoulX-Duplug** | 2603.14877 | 全双工对话 | Interspeech 2026 |
| 2026.03 | **SoulX-LiveAct** | 2603.11746 | 小时级实时人体动画 | |
| 2026.02 | **SoulX-FlashHead** | 2602.07449 | 实时流式数字人 | |
| 2026.02 | **SoulX-Singer** | 2602.07803 | 零样本歌声合成 (42K hrs) | |
| 2025.12 | **SoulX-FlashTalk** | 2512.23379 | 实时音驱Avatar | |
| 2025.10 | **SoulX-Podcast** | 2510.23541 | 多方言长播客生成 | |
| 2025.10 | **DialoSpeech** | 2510.08373 | 双人对话语音生成 | |
| 2025.10 | **SAC** | 2510.16841 | 语义-声学双流Codec | **ACL 2026 Main** |
| 2025.09 | **SenSE** | 2509.24708 | 语义感知语音增强 | |
| 2025.09 | **UniSS** | 2509.21144 | 表达性语音翻译 | |
| 2025.08 | **RAP** | 2508.05115 | 实时音驱肖像动画 | |
| 2025.08 | **OSUM-EChat** | 2508.09600 | 共情语音聊天机器人 | |
| 2025.08 | **Llasa+** | 2508.06262 | Llasa加速+流式 | |
| 2025.03 | **Spark-TTS** | 2503.01710 | BiCodec + Qwen2.5 TTS | **ACL 2025** |
| 2025.03 | **Teller** | 2503.18429 | 实时流式数字人 (AR) | **CVPR 2025** |
| 2025.02 | **Llasa** | 2502.04128 | Llama-based TTS (1B/3B/8B) | |
| 2025.01 | **FleSpeech** | 2501.04644 | 多模态prompt TTS | |

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Codec** | **SAC** (语义-声学双流, ACL 2026); **BiCodec** (Spark-TTS, 单流解耦); **OmniCodec** (通用低帧率) |
| **生成模型** | Llama-based AR (Llasa 1B/3B/8B); Qwen2.5-based AR+CoT (Spark-TTS); LLM+FM (DialoSpeech) |
| **歌声** | SoulX-Singer (42K小时, 中英粤, MIDI+melody条件) |
| **全双工** | SoulX-Duplug (plug-and-play语义VAD); Llasa+ (多token预测+验证) |
| **数字人** | Teller (CVPR 2025) → RAP → FlashTalk (14B) → FlashHead (1.3B) → LiveAct (小时级) |
| **语音理解** | SoulX-Transcriber (多人转录); Speaker Diarization |
| **评估** | MINT-Bench (多语言指令TTS); SoulX-Singer-Eval |

### 架构演进

```
Codec路线:
BiCodec (Spark-TTS, 2025.03) → SAC (2025.10, ACL 2026) → OmniCodec (2026.03)

TTS生成路线:
FleSpeech (2025.01) → Llasa (2025.02, scaling law) → Spark-TTS (2025.03, ACL 2025)
    → Llasa+ (2025.08, 加速) → DialoSpeech (2025.10) → SoulX-Podcast (2025.10)

歌声: SoulX-Singer (2026.02, 42K hrs, 工业级开源)

全双工: OSUM-EChat (2025.08) → SoulX-Duplug (2026.03)

数字人: Teller (CVPR 2025) → RAP → FlashTalk (14B) → FlashHead (1.3B) → LiveAct (小时级)
```

### 独特技术赌注

1. **全栈自研**: codec→TTS→SVS→数字人→全双工, 技术栈完整度最高
2. **Scaling Law验证**: Llasa是第一个系统验证TTS train-time + inference-time compute scaling的工作
3. **工业级歌声合成**: 42,000小时训练, 中英粤三语, 配套开源benchmark
4. **学术合作模式**: 深度绑定NWPU (Lei Xie) + SJTU X-LANCE (Kai Yu, Xie Chen)

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| SoulX-Podcast | 3,412 | 多方言播客, Apache 2.0 |
| SoulX-FlashTalk | 1,320 | 14B实时数字人 |
| SoulX-LiveAct | 1,120 | 小时级人体动画 |
| SoulX-FlashHead | 827 | 1.3B流式数字人 |
| SoulX-Singer | 707 | 零样本歌声合成, Apache 2.0 |
| SoulX-Duplug | 241 | 全双工模块 |
| SoulX-Transcriber | 156 | 多人转录 |
| SAC | 104 | ACL 2026 Codec |
| **总计** | **~7,900** | |

### 判断

**四家中最大的黑马。** 以社交App公司身份, 建立了最完整的语音技术栈和最密集的研究产出。核心在于:
1. 绑定NWPU+SJTU X-LANCE两大顶级实验室
2. 核心人物Xinsheng Wang从学术到工业的完整积累 (VISinger/Opencpop → Llasa/Spark-TTS)
3. CVPR 2025 + ACL 2025/2026顶会收录

**短板**: 公司品牌认知度低; 社交App主业与语音研究的战略契合度待验证

---

## 团队4: 米哈游 (miHoYo / HoYoverse)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | 米哈游 (miHoYo) / HoYoverse / Cognosphere |
| **GitHub** | 无公开研究org |
| **HuggingFace** | 无公开模型 |
| **核心人物** | 无法确认 (零公开论文) |

### 调研结果

**米哈游在arXiv上的公开研究产出为零。** 搜索"miHoYo"/"HoYoverse"/"Cognosphere"/"米哈游"均无结果。

**重要澄清**:
- **SpeechRole** (2508.02013) 来自**复旦大学NLP组**, 不是米哈游
- **EmoMix** (2306.00648) 来自平安科技, 不是米哈游
- 未找到任何米哈游直接参与的语音/TTS学术论文

### 技术现状推测

- 原神/崩坏/绝区零等游戏有**中/英/日/韩四语配音**, 全部真人配音
- 未公开任何AI配音替代计划
- 拥有大量高质量多语言配音数据 (几千小时级), 如果投入AI配音研发, 数据优势显著

### 判断

**完全不透明。** 零公开研究、零开源、零学术合作。可能反映: (a)完全依赖人工配音, (b)内部研发但高度保密, (c)使用第三方技术。游戏行业对AI配音的接受度仍有争议 (声优工会问题)。

---

## 横向对比矩阵

| 维度 | JD (京东) | Kunlun (天工) | Soul App | miHoYo (米哈游) |
|------|:---:|:---:|:---:|:---:|
| **研究产出** | 中 (6篇) | 低 (6篇, TTS仅1) | **极高 (20+篇)** | 零 |
| **技术栈完整度** | 中 (TTS+数字人) | 低 (TTS+音乐+视频) | **极高 (全栈自研)** | 未知 |
| **核心Codec** | MM-Tokenizer | 依赖LLM MoE | **SAC/BiCodec/OmniCodec** | 无 |
| **TTS模型** | JoyVoice (0.5B) | MoE-TTS | **Llasa (1B/3B/8B) + Spark-TTS** | 无 |
| **歌声合成** | 无 | Mureka (产品) | **SoulX-Singer (42K hrs)** | 无 |
| **数字人** | JoyHallo/JoyVASA | SkyReels-Audio | **Teller+FlashTalk+FlashHead+LiveAct** | 无 |
| **全双工** | 无 | 无 | **SoulX-Duplug** | 无 |
| **产品化** | 京东数字人客服 | **Mureka (API+MCP)** | Soul App内置 | 游戏配音 (人工) |
| **开源Stars** | ~1,430 | ~7,570 (含视频) | **~7,900 (纯语音+数字人)** | 0 |
| **顶会** | 无确认 | 无确认 | **CVPR+ACL×2** | 无 |
| **学术合作** | 浙大 | 无确认 | **NWPU+X-LANCE** | 无 |
| **差异化场景** | 电商客服 | AI音乐 | 社交语音+虚拟形象 | 游戏配音 |
| **技术赌注** | 长上下文多人 | Voice Design | 全栈+Scaling | 保密 |

### 关键发现

1. **Soul App是最大黑马**: 社交App公司建立了最完整的语音技术栈, 20+篇论文, CVPR/ACL顶会, 7,900+ stars开源。核心是绑定NWPU+X-LANCE + Xinsheng Wang的学术积累
2. **JD语音团队分散**: JD Explore Academy (JoyVoice) 和 JD Health (JoyHallo/JoyTTS) 是两个团队, 25人 vs 3人
3. **Kunlun重视频轻语音**: 语音是SkyReels视频生态的附属, MoE-TTS是唯一纯TTS论文, 但Mureka音乐产品化最成熟
4. **米哈游完全不透明**: 零公开研究, SpeechRole和EmoMix都不是米哈游的工作
