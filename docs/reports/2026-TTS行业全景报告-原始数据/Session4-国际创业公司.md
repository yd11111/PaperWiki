# Session 4: 国际语音AI创业公司 — ElevenLabs + Cartesia + Sesame + Fish Audio

> 调研日期: 2026-06-04

---

## 团队1: ElevenLabs

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | ElevenLabs Inc. (2022年成立, 纽约/伦敦, 全球14城) |
| **创始人** | Piotr Dabkowski (CEO, 前Google), Mati Staniszewski (CEO, 前Palantir) |
| **GitHub** | github.com/elevenlabs (25 repos, SDK为主) |
| **HuggingFace** | 无公开模型 (全闭源) |
| **员工** | 200+ |

### 融资时间线

| 轮次 | 时间 | 金额 | 估值 | 领投 |
|------|------|------|------|------|
| Seed | 2023初 | 未公开 | - | - |
| Series A | 2023.06 | $19M | ~$100M | Nat Friedman, Daniel Gross |
| Series B | 2024.01 | $80M | $1.1B | a16z |
| Series C | 2025.01 | $180M | $3B+ | a16z |
| Series D | 2026.02 | $500M | $11B | Sequoia Capital |
| **累计** | | **$781M** | | a16z, Sequoia, ICONIQ, Lightspeed, BOND |

**收入**: 2025年底 ARR $330M, 2026.05 突破 $500M ARR

### 产品/技术时间线

| 时间 | 事件 | 说明 |
|------|------|------|
| 2025.06 | Eleven v3 (alpha) | 高表现力TTS |
| 2025.08 | Eleven Music | AI音乐生成 |
| 2025.09 | ElevenAgents | 对话AI平台 (2M+ agent, 33M+ 对话) |
| 2025.11 | Scribe v2 Realtime | 实时STT |
| 2026.01 | Scribe v2 | 批量STT, 56类实体检测 |
| 2026.02 | Eleven v3 GA | 正式版 |
| 2026.03 | 波兰语ASR评测 | 第三方确认Scribe v2为SOTA级 (arXiv:2603.02246) |
| 2026.04 | 企业本地部署 | On-Premise方案 |
| 2026.05 | Music v2, Dubbing v2 | 产品线扩展 |

### 技术栈

| 维度 | 技术 |
|------|------|
| **TTS架构** | 完全闭源; v3支持Audio Tags情感控制; 多模型策略 |
| **模型系列** | v3 (高表现力/高延迟), Multilingual v2 (29语言), Flash v2.5 (~75ms), Turbo v2.5 (~250ms) |
| **STT** | Scribe v2: 56类实体检测, 100关键词提示, 多语言 |
| **语言覆盖** | TTS 70+语言 |
| **Voice Cloning** | Instant (1分钟) + Professional (30分钟) |
| **部署** | Cloud API / VPC (AWS/GCP) / On-Premise / On-Device |

### 产品矩阵

| 产品 | 定位 | 定价 |
|------|------|------|
| TTS API | 核心TTS, 4模型 | Free 10K chars → Business $990/mo |
| ElevenAgents | 企业对话AI | 电话/Web/App多通道 |
| Scribe (STT) | 批量/实时转录 | 含API套餐 |
| Music v2 | AI音乐 | 独立产品线 |
| Dubbing v2 | AI配音/翻译, 90+语言 | 按时长计费 |
| Voice Library | 10,000+预制声音 | 含API |

### 开源情况

几乎完全闭源。开源仅限SDK和UI组件:
- Python SDK: 3K stars
- UI组件库: 2.3K stars
- MCP Server: 1.4K stars

### 判断

**语音AI领域商业化最成功的公司。** $500M ARR + $11B估值。从单一TTS进化为"Audio OS"全栈平台 (TTS+STT+Music+Dubbing+Video+Agents)。**完全闭源, 无学术贡献, 但商业执行力无人能及。** 最大风险: 闭源策略vs开源追赶。

---

## 团队2: Cartesia

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | Cartesia Inc. (2023年成立, 旧金山) |
| **创始人** | Karan Goel (CEO), **Albert Gu** (Mamba/S4发明者, CMU教授), Christopher Re (Stanford教授, 顾问) |
| **GitHub** | github.com/cartesia-ai (Edge 407 stars) |
| **HuggingFace** | cartesia-ai (Rene 1.3B) |
| **员工** | ~26人 (2024) → 50+人 (2026估) |

### 融资时间线

| 轮次 | 时间 | 金额 | 领投 |
|------|------|------|------|
| Seed | 2023-2024 | ~$5M | 未公开 |
| Seed extension | 2024.12 | $22M | Index Ventures |
| Series A | 2025.03 | $64M | Kleiner Perkins |
| **累计** | | **~$91M** | Index, KP, A* Capital, Conviction, General Catalyst |

### 技术/产品时间线

| 时间 | 事件 | 说明 |
|------|------|------|
| 2021-2022 | S4 | Albert Gu, 结构化状态空间 |
| 2023.12 | **Mamba** | Albert Gu + Tri Dao, 选择性SSM |
| 2024.05 | **Mamba-2** | SSD框架, 核心层2-8x加速 |
| 2024.09 | Sonic TTS + Edge | 首发TTS产品 |
| 2025.03 | **Sonic 2.0** | 90ms延迟 (Turbo 40ms), 1.5x偏好率领先 |
| 2026 | **Sonic 3.5** | 42语言, 当前旗舰 |
| 2026 | **Ink-2** | STT产品 |
| 2026 | **Line** | Voice Agent平台 |

### 技术栈

| 维度 | 技术 |
|------|------|
| **核心架构** | **State Space Model (SSM)** — Mamba/Mamba-2变体, 非Transformer |
| **TTS** | Sonic系列 (1 → 2 → 3 → 3.5) |
| **STT** | Ink-2 |
| **语言** | 42语言 |
| **延迟** | Sonic 2 Turbo **40ms** (全行业最低之一) |
| **SSM优势** | 验证损失低20%, WER低2x, NISQA+1分, 吞吐量4x, 推理速度2x (vs Transformer) |
| **部署** | Cloud API / On-Premise / **On-Device** (Apple M系列, Metal kernels) |

### 独特技术赌注

1. **SSM替代Transformer**: 核心赌注 — 线性复杂度 vs 二次复杂度
2. **On-Device**: SSM固定内存占用天然适合边缘部署
3. **Voice Agent全栈 (Line)**: TTS+STT+LLM一体化
4. **学术创始人优势**: 创始团队=Mamba发明人, SSM领域最强话语权

### 开源情况

部分开源:
- Edge推理库: Apache 2.0, 407 stars
- Rene 1.3B语言模型: HuggingFace公开
- Sonic/Ink模型权重: 闭源

### 判断

**SSM架构在语音AI的最佳代言人。** 40ms延迟是结构性优势。团队虽小(~26人)但学术根基极强。**如果SSM在大规模下被证实优于Transformer, Cartesia将获巨大回报; 反之则面临Transformer优化追平的风险。** $64M Series A (KP领投) 说明顶级VC认可此赌注。

---

## 团队3: Sesame AI

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | Sesame AI Inc. (~2023-2024成立, SF/Bellevue/NYC) |
| **创始人** | **Brendan Iribe** (Oculus VR创始人), **Nate Mitchell** (Oculus VR联合创始人) |
| **核心人物** | Ankit Kumar (ML负责人), Johan Schalkwyk (前Google Speech负责人), Dan Lyth, Sefik Emre Eskimez |
| **GitHub** | github.com/SesameAILabs (CSM **14.7K stars**) |
| **HuggingFace** | sesame (csm-1b) |
| **投资方** | a16z, Sequoia, Spark Capital, Matrix Partners (~$100M+) |

### 技术/产品时间线

| 时间 | 事件 | 说明 |
|------|------|------|
| 2025.02 | **CSM博客** | "Crossing the Uncanny Valley of Conversational Voice" |
| 2025.02 | **CSM-1B开源** | Apache 2.0, GitHub+HuggingFace |
| 2025.06 | CoVoMix2 (NeurIPS) | CSM作为baseline被超越 |
| 2026.05 | **iOS App** | 39国, 4个AI Agent (Maya/Miles/Simone/Charlie) |
| 2027 (计划) | 智能眼镜 | AI语音硬件 |

### 技术栈

| 维度 | 技术 |
|------|------|
| **核心架构** | **CSM (Conversational Speech Model)** — 多模态自回归 |
| **Backbone** | Llama架构Transformer (基于Llama-3.2) |
| **Audio Decoder** | 小型Transformer, 逐codebook预测RVQ编码 |
| **Tokenizer** | Mimi (split-RVQ, 12.5Hz, 1语义+N-1声学codebook) |
| **关键设计** | 单阶段端到端 (无语义token瓶颈), Decoder仅1/16帧训练 |
| **模型规模** | Tiny 1B+100M, Small 3B+250M, Medium 8B+300M |
| **训练数据** | ~100万小时公开英语音频 |
| **特色** | **Voice Presence**: 情感智能+对话动态+上下文感知+一致性人格 |

### 独特技术赌注

1. **对话韵律 (Voice Presence)**: 专注AI对话"感觉真实", 而非通用TTS
2. **单阶段端到端**: 避免语义token瓶颈, 直接RVQ建模
3. **硬件赌注**: 2027年智能眼镜, 其他三家都没有
4. **消费级产品路线**: 不做API/B2B, 直接B2C AI伴侣

### 开源情况

选择性开源:
- **CSM-1B**: Apache 2.0, 14.7K stars
- fine-tuned版本 (iOS App用): 闭源

### 判断

**Oculus创始人团队做AI语音, 赌的是"AI对话体验"新品类。** CSM对话韵律技术领先, 但商业模式高度不确定: 无B2B收入, 无API变现, 纯消费级产品。**2027年智能眼镜如果成功可能开辟新市场; 如果失败, 缺乏收入缓冲。** CSM已被CoVoMix2等后续工作超越, 需要迭代。

---

## 团队4: Fish Audio

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | Fish Audio / 鱼声科技 (~2023成立, 中国) |
| **GitHub** | github.com/fishaudio (fish-speech **31K stars**) |
| **HuggingFace** | fishaudio (S2-Pro等) |
| **核心人物** | Shijia Liao (廖世嘉), Yuxuan Wang, Tianyu Li, Yifan Cheng |
| **团队规模** | 14+ 研究人员 |

### 论文/产品时间线

| 时间 | 论文/事件 | 说明 |
|------|----------|------|
| 2024.11 | **Fish-Speech v1.4** (arXiv:2411.01156) | Dual-AR架构 + GFSQ + FF-GAN |
| 2025.05 | Fish-Speech v1.5.1 | GitHub release |
| 2026.03 | **Fish Audio S2** (arXiv:2603.08823) | 多阶段训练 + 指令跟随 + GRPO对齐 |

### 技术栈

| 维度 | 技术 |
|------|------|
| **核心架构** | **Dual Autoregressive (Dual-AR)** — 快慢双自回归 |
| **Slow AR** | 4B参数 decoder-only Transformer, 时间轴主语义codebook |
| **Fast AR** | 400M参数, 填充剩余9个残差codebook |
| **Codec** | RVQ (10 codebooks, ~21Hz) |
| **Tokenizer** | 直接用LLM做语言特征提取, **无G2P** |
| **量化器** | GFSQ (Grouped Finite Scalar Vector Quantization), ~100%码本利用率 |
| **后训练** | **GRPO** (语义准确性+指令遵循+声学偏好+音色相似度) |
| **推理** | SGLang (continuous batching, paged KV cache), RTF 0.195 |
| **语言** | 80+语言 |
| **训练数据** | 1000万+小时 |

### Benchmark (S2-Pro)

| 指标 | 结果 |
|------|------|
| Seed-TTS Eval WER (中文) | **0.54%** (SOTA) |
| Seed-TTS Eval WER (英文) | **0.99%** (SOTA) |
| Audio Turing Test | 0.515 (超Seed-TTS和MiniMax-Speech) |
| EmergentTTS-Eval胜率 | 81.88% |

### 独特技术赌注

1. **Dual-AR架构**: 快慢双AR解决RVQ质量-效率权衡, Fish独创
2. **LLM-native语言理解**: 直接LLM替代G2P, 天然多语言
3. **GRPO对齐**: RLHF式后训练引入TTS, 多维reward优化
4. **全开源生态**: 模型+推理引擎+SDK全开源

### 开源情况

**四家中开源最激进:**
- Fish-Speech S2-Pro (4B): 权重+fine-tuning+SGLang (Fish Audio Research License)
- Fish-Speech v1.x: 完整开源
- Bert-VITS2: 8.8K stars
- Fish Diffusion: 745 stars
- SDK: Python/TypeScript/Go/n8n

### 判断

**开源TTS的GitHub Star王者。** 31K stars + Benchmark SOTA (WER 0.54%) + 80+语言。Dual-AR和GRPO是TTS领域前沿创新。**但融资不透明, 品牌认知远低于ElevenLabs, 开源变现效率待验证。** Fish Audio Research License限制可能阻碍企业采用。

---

## 附录: 其他关注公司

### Hume AI (已被Google收购)

| 项目 | 详情 |
|------|------|
| **重大变动** | **2026年1月, Google收购Hume AI团队** (TechCrunch报道) |
| **创始人** | Alan Cowen (情感计算研究者, GoEmotions作者) |
| **技术** | EVI 3 (speech-to-speech LLM), Octave 2 (TTS), TADA (开源LLM-TTS) |
| **特色** | 情感理解 (48+情绪, 600+语音描述符, 50+语言) |
| **EVI 3性能** | 延迟~1.2s, 30种情绪/风格调制超GPT-4o/Gemini/Sesame |
| **现状** | 团队已被Google吸收, 品牌走向不确定 |

### Deepgram

| 项目 | 详情 |
|------|------|
| **定位** | ASR + TTS + Voice Agent全栈平台 |
| **产品** | Nova (STT), Speak (TTS), Flux (统一Voice Agent API) |
| **客户** | Twilio, Cloudflare, Sierra, IBM, Vapi |
| **竞争位置** | 偏ASR侧, TTS非核心优势 |

### Rime

| 项目 | 详情 |
|------|------|
| **定位** | 超低延迟B2B TTS, 电话/IVR场景 |
| **合规** | SOC 2 + HIPAA |
| **客户** | Fortune 500, ConverseNow, Domino's (销售+15%) |

---

## 横向对比矩阵

| 维度 | ElevenLabs | Cartesia | Sesame AI | Fish Audio |
|------|:---:|:---:|:---:|:---:|
| **架构** | 闭源(未知) | **SSM (Mamba)** | Llama Transformer | **Dual-AR** |
| **模型规模** | 未公开 | 未公开 | 1B/3B/8B | 4B+400M |
| **训练数据** | 未公开 | 未公开 | ~100万小时 | ~1000万小时 |
| **语言数** | 70+ | 42 | 英语为主 | 80+ |
| **最低延迟** | Flash 75ms | **Turbo 40ms** | ~1s | TTFA <100ms |
| **融资总额** | $781M | ~$91M | ~$100M+ | 未公开 |
| **估值** | $11B | 未公开 | 未公开 | 未公开 |
| **ARR** | $500M+ | 未公开 | 无收入 | 未公开 |
| **开源** | ★ (全闭源) | ★★ (部分) | ★★★ (CSM-1B) | ★★★★★ (全开源) |
| **GitHub Stars** | SDK 3K | Edge 407 | CSM 14.7K | **fish-speech 31K** |
| **论文发表** | 0 | 通过学术创始人 | 0 arXiv | 2篇arXiv |
| **B2B vs B2C** | 两者兼顾 | 纯B2B | 纯B2C | B2B API+社区 |
| **STT** | Scribe v2 | Ink-2 | 无 | 无 |
| **Music** | Music v2 | 无 | 无 | 无 |
| **Voice Agent** | ElevenAgents | Line | iOS App | 无 |
| **硬件** | 无 | On-Device | 2027眼镜(计划) | 无 |
| **独特赌注** | Audio OS平台化 | SSM替代Transformer | 对话韵律+硬件 | Dual-AR+GRPO+开源 |

### 关键发现

1. **ElevenLabs商业遥遥领先**: $500M ARR, $11B估值, 但技术完全不透明
2. **Cartesia SSM赌注独一无二**: 40ms延迟是结构性优势, 但需证明SSM大规模可行
3. **Sesame路线最另类**: B2C+硬件, Oculus基因, 不做API — 高风险高回报
4. **Fish Audio开源最激进且质量最高**: 31K stars + WER SOTA, 中国公司做全球开源TTS
5. **Hume AI被Google收购(2026.01)**: 情感语音赛道被巨头吸收
6. **四条路径**: 平台SaaS(ElevenLabs) vs 架构创新(Cartesia) vs 体验产品(Sesame) vs 开源社区(Fish)
