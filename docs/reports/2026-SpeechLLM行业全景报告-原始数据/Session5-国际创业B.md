# Session 5: 国际创业 B — Fixie AI / Hume AI / ElevenLabs / Deepgram

> **报告范围**: Speech LLM / Omni / 全双工对话
> **调研日期**: 2026-06-08
> **数据来源**: GitHub API, HuggingFace, arXiv, Google Search, 官方网站, 公开技术报告, 融资新闻
> **5 层搜索覆盖**: Layer 1 (GitHub/HuggingFace org) ✓ | Layer 2 (核心人搜索) ✓ | Layer 3 (arXiv affiliation) 部分 | Layer 4 (产品/竞赛) ✓ | Layer 5 (引用网络) 部分
> **特别说明**: 本 Session 覆盖的 4 家公司偏产品化导向,学术论文产出较少。信息主要来自产品文档、技术博客、融资新闻和开源项目。

---

## 1. Fixie AI (Ultravox) — 开源 Speech LLM

### 1.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | Fixie.ai Inc. (品牌名已转为 Ultravox.ai) |
| 团队 | Ultravox 团队; 分布式,总部 Seattle |
| GitHub | [github.com/fixie-ai](https://github.com/fixie-ai) (68 repos, 756 followers) |
| HuggingFace | [huggingface.co/fixie-ai](https://huggingface.co/fixie-ai) |
| 核心人物 | Zach Koch (Co-Founder, CEO, ex-Shopify), Matt Welsh (Co-Founder, ex-OctoML/Apple/Google/Harvard), Justin Uberti (CTO), Zhongqiang Huang, William Held (DiVA 论文第一作者, Georgia Tech) |
| 产品线 | Ultravox 开源 Speech LLM 模型 (v0.4 → v0.7), Ultravox Realtime API (托管平台), thefastest.ai (AI 速度基准测试) |
| 官网 | [ultravox.ai](https://ultravox.ai) |
| 定位 | 开源 Speech LLM 基础设施; 不做端到端语音生成,专注语音理解 + LLM 推理,输出文本再外接 TTS |
| 融资 | $17M 种子轮 (2023.03, Redpoint Ventures 等); 后续融资未公开 |

### 1.2 论文/模型时间线 (2024-2026)

| 时间 | 论文/模型 | arXiv ID / 来源 | 核心贡献 | 与 Speech LLM 关系 |
|------|---------|----------------|---------|-------------------|
| 2024.10 | DiVA (Ultravox 原型) | 2410.02678 (ACL 2025) | 无指令训练数据的端到端语音助手蒸馏; 用 text-only LLM 的回复做自监督 | **学术基础**: Ultravox 架构的理论来源, 36 次引用 |
| 2025.02 | Ultravox v0.5 | HuggingFace release | 基于 Llama 3.3 70B + whisper-large-v3-turbo; 多语言支持; 多种 LLM 骨干可选 | **多骨干策略**: 同时支持 Llama 3.1 8B/70B, Qwen 2.5 72B 等 |
| 2025.09 | Ultravox v0.5 (HF Collection) | HuggingFace | whisper-large-v3-turbo (fine-tuned) + frozen LLM | 模型集合发布 |
| 2025 (具体月份不详) | Ultravox v0.6 | GitHub release | 默认模型升级; 权重推送至 HuggingFace | 中间迭代 |
| 2025.12 | Ultravox v0.7 | Blog 2025.12.04 | 基于 GLM 4.6 (355B, 160 experts/layer, MoE); Big Bench Audio 91.8% (97% with thinking) | **旗舰版本**: 切换至 GLM 4.6 骨干, AIEWF benchmark SOTA |
| 2026.02 | When Do Speech LLMs Behave Like ASR→LLM Pipelines? | 2602.17598 | 评估 Speech LLM 行为是否等价于 ASR+LLM 级联 | 学术贡献: 对 Speech LLM 范式的批判性分析 |

### 1.3 技术栈全景

| 维度 | Ultravox 技术栈 |
|------|----------------|
| **语音编码器** | whisper-large-v3-turbo (fine-tuned); 通过 multimodal projector 将 Whisper encoder 输出映射到 LLM 嵌入空间 |
| **LLM 骨干** | 可插拔: v0.5 使用 Llama 3.3 70B; v0.7 使用 GLM 4.6 (355B, MoE, 160 experts); 也支持 Llama 3.1 8B/70B, Qwen 2.5 72B |
| **语音解码器** | 无自研 TTS — Ultravox 是理解模型,输出文本,需外接第三方 TTS (ElevenLabs, Deepgram Aura 等) 或 Bring Your Own TTS |
| **对话策略** | 级联式: Speech → Ultravox (理解+推理) → Text → 外接 TTS → Speech; 平台层提供 VAD, barge-in, turn-taking |
| **训练方法** | DiVA 蒸馏: 用 text-only LLM 对 transcript 的回复作为 soft label 训练 speech LLM,无需标注的 instruction data |
| **推理延迟** | 依赖托管平台; Ultravox Realtime API 定位"低延迟"; AIEWF benchmark 在延迟+智能综合排名中领先 |
| **多语言** | 26 语种 (Arabic, Chinese, French, German, Hindi, Japanese, Spanish 等) |
| **情感/副语言** | 依赖外接 TTS 的能力; Ultravox 本身侧重语义理解而非声学生成 |

### 1.4 架构演进

```
DiVA 论文 (蒸馏范式, 2024.10)
    ↓ 理论基础
Ultravox v0.4 (初版开源模型, 2024)
    ↓ 扩展骨干 + 多语言
Ultravox v0.5 (Llama 3.3 70B + whisper-large-v3-turbo, 2025.02)
    ↓ 迭代优化
Ultravox v0.6 (中间版本, 2025)
    ↓ 切换至 MoE 骨干
Ultravox v0.7 (GLM 4.6 355B MoE, Big Bench Audio SOTA, 2025.12)
    ↓ 平台化
Ultravox Realtime API (托管平台, tools/RAG/telephony/voice cloning)
```

### 1.5 独特技术赌注

1. **"理解即产品" 路线**: Ultravox 不做语音生成,专注语音理解+LLM推理,将 TTS 解耦为可替换组件。这与 Moshi/GLM-4-Voice 等端到端路线截然不同 — 赌注在于:语音理解是核心价值,语音生成是可替换的commodity。
2. **骨干可插拔 (Backbone-agnostic)**: 同一架构可适配不同 LLM (Llama/Qwen/GLM), 跟随 LLM 生态的最新进展,无需从头训练。v0.7 从 Llama 切换到 GLM 4.6 就是这一策略的体现。
3. **DiVA 蒸馏 (无指令数据训练)**: 不需要标注的 speech-instruction 数据,仅用 text LLM 对 transcript 的回复做蒸馏。显著降低数据收集成本,但可能牺牲 speech-specific 能力。
4. **平台化 + 开源双轨**: 模型权重开源 (Apache 2.0), 同时提供托管 Realtime API 盈利。开发者可以自托管或使用 API。

### 1.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| ultravox | 4,400 | 核心 Speech LLM 模型 + 训练代码 |
| ai-benchmarks | 88 | AI API 基准测试套件 |
| ultradox | 43 | 文档站 |
| thefastest.ai | 42 | AI 模型速度排行网站 |
| hisanta.ai | 36 | 语音对话 Demo (Talk to Santa) |
| ultravox-client-sdk-js | 27 | JavaScript 客户端 SDK |
| ultravox-client-sdk-python | 16 | Python 客户端 SDK |

**HuggingFace 模型**:
- fixie-ai/ultravox-v0_7-glm-4_6 (latest)
- Ultravox v0.5 Collection (多骨干: Llama 3.1 8B/70B, Qwen 2.5 72B 等)
- 模型权重全部开放

### 1.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **开源 Speech LLM 中关注度最高的西方项目**: 4.4K stars, 被学术论文广泛引用; (2) **骨干可插拔策略灵活**: 可快速跟进最新 LLM, 不被单一模型绑定; (3) **DiVA 蒸馏方法创新**: 无需 speech-instruction 数据, 降低训练成本 (ACL 2025, 36 citations); (4) **平台化产品完善**: RAG, tools, telephony, voice cloning, multi-stage calls 等企业级功能; (5) 26 语种多语言支持 |
| **短板** | (1) **无自研 TTS**: 语音生成完全依赖第三方, 无法控制端到端体验; 与 Moshi/GLM-4-Voice 等端到端方案相比, 在延迟和一致性上有劣势; (2) **无全双工能力**: 不支持真正的 full-duplex, 依赖 VAD 等工程组件实现 barge-in; (3) 模型架构本身创新有限 (multimodal projector + LLM 是标准范式); (4) 融资规模小 ($17M seed), 与 ElevenLabs ($781M) 差距巨大; (5) 团队学术产出少, 主要靠 DiVA 一篇论文 |
| **下一步推测** | (1) Ultravox v0.8+ 可能引入更大骨干或自研 TTS; (2) 可能与 GLM/Llama 4 等最新 LLM 深度集成; (3) 平台化 Realtime API 可能向 voice agent 方向扩展; (4) 可能通过合作或收购补齐 TTS 能力; (5) 更多语言和 domain-specific 微调 |

---

## 2. Hume AI — 情感语音 AI

### 2.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | Hume AI |
| 团队 | Hume AI Research Lab; 约 50+ 人 |
| GitHub | [github.com/HumeAI](https://github.com/HumeAI) (28 repos, 441 followers) |
| HuggingFace | HumeAI (tada-1b, tada-3b-ml 等) |
| 核心人物 | Alan Cowen (Founder, CEO, Chief Scientist; UC Berkeley Psychology PhD, 前 Google AI 情感计算团队负责人), Janet Ho (COO, ex-Zynga/Rakuten), John Beadle (创始投资人, CFO), Trung Dang, Sharath Rao (TADA 论文作者) |
| 产品线 | EVI (Empathic Voice Interface, 语音对话), Octave (TTS), Expression Measurement (多模态情感识别), TADA (开源 TTS 模型) |
| 官网 | [hume.ai](https://hume.ai) |
| 定位 | **情感智能语音 AI**; 核心差异化在于将情感理解融入语音交互的全链路, 而非仅做 TTS 或 ASR |
| 融资 | $12.7M Series A (2023.01, USV); $50M Series B (2024.03, EQT Ventures, $219M 估值); 总融资 ~$72.8M |
| 备注 | CB Insights 报告 2026.01 有"Reverse Acqui-Hire"事件, 具体情况待确认 |

### 2.2 论文/产品时间线 (2024-2026)

| 时间 | 论文/产品 | arXiv ID / 来源 | 核心贡献 | 与 Speech LLM 关系 |
|------|---------|----------------|---------|-------------------|
| 2024.03 | HUME-VOCALBURST | 2403.14048 | 大规模情感非语言发声数据集 | 情感理解数据基础 |
| 2024.03 | EVI 1 (产品) | 官网发布 | 首个情感智能语音对话接口 API | **核心产品**: 情感感知的语音对话 |
| 2024.12 | Octave 1 (产品) | 官网发布 | 首个 LLM-based TTS; 根据文本语义自动推断语气、情感 | TTS 组件: 理解文本含义并控制表达 |
| 2025.05 | EVI 3 (产品) | 官网发布 | 100K+ 自定义声音; ~300ms 响应; 自然打断处理 | **旗舰版本**: 情感对话系统升级 |
| 2025.10 | Octave 2 (产品) | 官网发布 | 延迟 ~100ms (40% 加速); SambaNova 专用推理硬件 | TTS 显著加速 |
| 2026.01 | EV4-mini (产品) | 官网发布 | 11 语种多语言支持 (英日韩西法等) | 多语言扩展 |
| 2026.02 | TADA | 2602.23068 (arXiv) | Text-Acoustic Dual Alignment; 1:1 文本-声学对齐; 基于 Llama 3.2; 开源 | **技术突破**: 消除内容幻觉, 5x 加速 |
| 2026.03 | TADA 开源 | GitHub/HuggingFace | TADA-1B, TADA-3B-ML; 9 语种; 流式生成 | 首个开源模型 |

### 2.3 技术栈全景

| 维度 | Hume AI 技术栈 |
|------|---------------|
| **语音编码器** | Expression Measurement API: 48 维面部表情 + 48 维语音韵律 + 53 维文本情感; TADA: tada-codec (自研音频编码器) |
| **LLM 骨干** | TADA: 基于 Llama 3.2 (1B / 3B); EVI/Octave: 内部模型 (未公开架构) |
| **语音解码器** | TADA: Flow-matching decoder (10 步); Octave: 内部 TTS 系统 (未公开); 关键创新: 1:1 text-acoustic 对齐消除帧率约束 |
| **对话策略** | EVI: WebSocket 流式音频 → 情感分析 + 语义理解 → 情感感知回复生成; 支持自然打断 |
| **训练方法** | RLHE (Reinforcement Learning from Human Expression): 用人类表达偏好优化模型输出; 基于百万量级人类反应数据集 |
| **推理延迟** | EVI 3: ~300ms 响应 (实际含传输 ~1.2s); Octave 2: ~100ms TTFB; TADA: ~0.12x RTF on H100 |
| **多语言** | EVI: 11 语种; TADA: 9 语种 (Arabic, Chinese, German, Spanish, French, Italian, Japanese, Polish, Portuguese) |
| **情感/副语言** | **核心差异化**: 48 维情感识别 (语音韵律); RLHE 训练; Octave 根据文本语义自动推断 whisper/shout/情感调制; 无需显式标签 |

### 2.4 架构演进

```
Semantic Space Theory (Alan Cowen 学术研究, 30+ 情感维度)
    ↓ 理论基础
Expression Measurement API (48D facial + 48D vocal + 53D textual)
    ↓ 情感分析组件
EVI 1 (Empathic Voice Interface, 情感语音对话, 2024.03)
    ↓ 迭代
EVI 3 (100K voices, ~300ms, 打断处理, 2025.05) → EV4-mini (11 语种, 2026.01)
    
并行 TTS 线:
Octave 1 (LLM-based 情感 TTS, 2024.12) → Octave 2 (40% 加速, ~100ms, 2025.10)
    ↓ 学术化 + 开源化
TADA (Text-Acoustic Dual Alignment, 1:1 对齐, 2026.02-03)
    ├── TADA-1B (Llama 3.2 1B)
    └── TADA-3B-ML (Llama 3.2 3B, 多语言)
    
关键训练创新:
RLHE (Reinforcement Learning from Human Expression) ←── 替代标准 RLHF, 用人类表达偏好
```

### 2.5 独特技术赌注

1. **情感作为核心 (Emotion-First)**: Hume 赌注于"情感理解是 Voice AI 的核心差异化",而非单纯追求低延迟或高自然度。他们的学术基础 (Semantic Space Theory, 40+ 论文, 3000+ 引用) 提供了科学护城河。
2. **RLHE (Reinforcement Learning from Human Expression)**: 用人类情感表达偏好替代标准 RLHF 的标量 reward,让模型学习"什么样的回复让人感觉好"而非"什么回复正确"。这是独有的训练方法。
3. **TADA 1:1 对齐**: 每个文本 token 动态对应一个语音段,消除固定帧率约束。关键优势: (a) 消除内容幻觉 (conventional TTS 常见问题); (b) 上下文长度 = 纯文本,大幅降低计算成本; (c) 5x 生成加速。
4. **伦理框架 (The Hume Initiative)**: 6 条原则 (beneficence, emotional primacy, scientific legitimacy, inclusivity, transparency, consent), 由独立伦理委员会监督。这既是差异化,也是对 EU AI Act 等监管风险的前瞻性应对。

### 2.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| tada | 989 | 开源 Speech Language Model (TADA-1B/3B) |
| hume-api-examples | 247 | API 示例项目 |
| hume-python-sdk | 174 | Python SDK |
| hume-typescript-sdk | 79 | TypeScript SDK |
| hume-react-sdk | 45 | React SDK |
| competitions | 32 | ML 竞赛 |
| hume-swift-sdk | 18 | Swift SDK |
| wsds | 7 | 大规模多模态数据集工具 |

**注意**:
- EVI (Empathic Voice Interface) **未开源** — 仅通过 API 提供
- Octave **未开源** — 仅通过 API 提供
- Expression Measurement **未开源** — 仅通过 API 提供
- TADA 是唯一开源模型

### 2.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **独特的情感维度定位**: 在 Voice AI 竞争中,情感理解是唯一不可复制的差异化 — 基于 10+ 年学术研究; (2) **TADA 技术创新**: 1:1 text-acoustic 对齐是 TTS 领域的新范式,消除帧率约束和内容幻觉; (3) **商业化进展**: 100K+ 开发者使用 API, 与 Toyota/Volkswagen/Northwell Health 等大客户合作; (4) **开源策略**: TADA 开源建立社区信任 |
| **短板** | (1) **融资规模有限**: $72.8M vs ElevenLabs $781M, 资源差距巨大; (2) **核心产品 EVI/Octave 闭源**: 开源仅 TADA, 核心竞争力无法被社区验证; (3) **情感 AI 的监管风险**: EU AI Act 对情感识别有严格限制, 可能影响欧洲市场; (4) 2026.01 的 "Reverse Acqui-Hire" 事件暗示可能存在团队/方向变动; (5) 多语言覆盖有限 (11 语种 vs ElevenLabs 32 语种) |
| **下一步推测** | (1) TADA 可能整合进 EVI 实现端到端情感语音对话; (2) 进一步扩展多语言支持; (3) 可能寻求更大融资或战略合作; (4) Expression Measurement + EVI + TADA 三位一体的情感 AI 平台; (5) 在医疗/心理健康/客服等垂直领域深耕 |

---

## 3. ElevenLabs — 语音合成巨头

### 3.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | ElevenLabs (总部 London/New York, 于 2022 年创立) |
| 团队 | ~223 人; 14 个城市办公 (London, NYC, SF, Warsaw, Dublin, Tokyo, Seoul, Singapore 等) |
| GitHub | [github.com/elevenlabs](https://github.com/elevenlabs) (25 repos, 2.7K followers) |
| HuggingFace | [huggingface.co/elevenlabs](https://huggingface.co/elevenlabs) (有限公开模型) |
| 核心人物 | Mati Staniszewski (Co-Founder, CEO; ex-Palantir, Imperial College London 数学), Piotr Dabkowski (Co-Founder, CTO; ex-Google, Oxford + Cambridge AI/ML, NeurIPS publication) |
| 产品线 | TTS API (Multilingual v2 / Turbo v2.5 / Eleven v3), Voice Cloning, Conversational AI (ElevenAgents), Dubbing Studio, GenFM, Sound Effects, ElevenReader, Voiceover Studio |
| 官网 | [elevenlabs.io](https://elevenlabs.io) |
| 定位 | **Voice AI 领域估值最高的独立公司**; 从 TTS 起家,扩展到全音频栈 (TTS + STT + Dubbing + Conversational AI + SFX + Music) |
| 融资 | $19M Series A (2023.01); $80M Series B (2024.01, $1.1B, a16z); $250M Series C (2025.01, $3.3B, ICONIQ); $500M Series D (2026.02, $11B, Sequoia); 总融资 $781M; Employee Tender Offer at $6.6B (2025.09) |

### 3.2 产品/模型时间线 (2022-2026)

| 时间 | 产品/事件 | 来源 | 核心贡献 | 与 Speech LLM 关系 |
|------|---------|------|---------|-------------------|
| 2022.01 | 公司创立 | — | Mati Staniszewski + Piotr Dabkowski 联合创立 | 创始: 解决 TTS 自然度问题 |
| 2022.08 | Beta 平台上线 | — | 首版 TTS API | TTS 产品化起点 |
| 2023.01 | 用户突破 100 万 | — | 5 个月内快速增长 | 市场验证 |
| 2024.01 | Series B ($1.1B 估值) | — | 成为独角兽 | 资本验证 |
| 2024 (具体月份不详) | Multilingual v2 | 产品文档 | 29 语种高质量 TTS; 稳定、逼真 | **标准模型**: 内容创作导向 |
| 2024 (具体月份不详) | Turbo v2.5 | 产品文档 | 32 语种低延迟 TTS; 300% 速度提升; 为 Conversational AI 优化 | **实时模型**: 对话场景优化 |
| 2024.07 | Iconic Voices | 产品发布 | 历史人物 AI 语音表示 | 品牌化创新 |
| 2024.10 | 收购 Omnivore | 企业新闻 | 自动化语音管线,强化媒体本地化 | 战略收购 |
| 2024.11 | GenFM | 产品更新 | PDF/电子书/文章 → AI 播客 (双主持人) | 新形态内容生成 |
| 2025.01 | Series C ($3.3B) | — | $250M 融资 | 估值三倍增长 |
| 2025 ARR | $330M+ ARR | 融资新闻 | 年度经常性收入超 3.3 亿美元 | 商业化成功 |
| 2026.02 | Series D ($11B) | 融资新闻 | $500M 融资, Sequoia 领投; Eleven v3 Conversational model 发布 | **最新**: 对话模型升级, 准备 IPO |
| 2026.02 | Eleven v3 Conversational | 产品发布 | 改进表达性和对话 turn-taking | 对话 AI 能力增强 |
| 2026 (进行中) | ElevenAgents | 企业平台 | 企业级语音/对话 Agent 部署平台; Deutsche Telekom, Square, Revolut 等客户 | 企业 Voice Agent 平台 |

### 3.3 技术栈全景

| 维度 | ElevenLabs 技术栈 |
|------|-----------------|
| **语音编码器** | 自研 (未公开架构); 支持 Voice Cloning (极少训练数据复制声音); Speech-to-Speech 模式 |
| **LLM 骨干** | 自研 (未公开); Eleven v3 Conversational model 为最新; ~7 人专职音频 AI 研究团队 |
| **语音解码器** | 自研 TTS 引擎 (未公开架构); Multilingual v2 (高质量), Turbo v2.5 (低延迟), Eleven v3 (对话优化) |
| **对话策略** | ElevenAgents: 全托管对话 Agent 平台; Conversational AI API 支持实时语音交互; Turbo v2.5 专为对话优化 |
| **训练数据规模** | 未公开; 但 41% 的 Fortune 500 使用 ElevenLabs 暗示大规模数据飞轮 |
| **推理延迟** | Turbo v2.5: 300% 速度提升 (vs 标准模型); 具体 TTFB 未公开; Conversational AI 定位"低延迟" |
| **多语言** | Multilingual v2: 29 语种; Turbo v2.5: 32 语种; 对话模型: 70+ 语种 (ElevenCreative 平台) |
| **情感/副语言** | 模型自动推断情感、停顿、笑声、呼吸; 创始动机就是解决 TTS 的"机器感"; 无显式情感识别 API |

### 3.4 架构演进

```
创始 (TTS 自然度研究, 2022)
    ↓
Beta TTS API (2022.08) → 100 万用户 (2023.01)
    ↓ 多语言扩展
Multilingual v2 (29 语种, 高质量 TTS)
    ↓ 低延迟优化
Turbo v2.5 (32 语种, 300% 加速, Conversational AI 优化)
    ↓ 对话模型
Eleven v3 Conversational (改进 turn-taking + 表达性, 2026.02)
    
并行产品扩展:
Voice Cloning → Dubbing Studio → GenFM → Sound Effects → Music
    ↓
ElevenCreative (内容创作平台) + ElevenAgents (企业 Agent 平台) + ElevenAPI (开发者基础设施)
    ↓
全音频栈: TTS + STT + Dubbing + Conversational AI + SFX + Music → "Audio General Intelligence"
```

### 3.5 独特技术赌注

1. **"全音频栈" 路线 (Full Audio Stack)**: ElevenLabs 赌注于覆盖所有音频 AI 任务 — TTS, STT, Voice Cloning, Dubbing, SFX, Music, Conversational AI。目标是成为"Audio General Intelligence"的平台,而非在单一技术上做到最好。
2. **商业化驱动创新**: 与学术驱动的竞争对手不同,ElevenLabs 的创新是产品化优先。$330M ARR 和 41% Fortune 500 使用率提供了无与伦比的数据飞轮。
3. **Voice Marketplace 网络效应**: 用户可以创建、分享、变现语音配置文件。这创造了类 UGC 的网络效应 — 更多声音 → 更多用户 → 更多数据 → 更好模型。
4. **无学术论文策略**: ElevenLabs 几乎没有公开学术论文。所有技术 know-how 作为商业机密保留。这在 AI 领域非常罕见,反映了创始人的商业优先理念。

### 3.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| elevenlabs-python | 2,996 | 官方 Python SDK |
| ui | 2,257 | ElevenLabs UI 组件库 (基于 shadcn/ui) |
| elevenlabs-mcp | 1,395 | MCP Server |
| examples | 608 | 示例项目 |
| elevenlabs-js | 428 | JavaScript SDK |
| skills | 324 | Agent 技能集合 |
| elevenlabs-swift-sdk | 112 | Swift SDK |
| packages | 106 | TypeScript Agents SDK |

**注意**:
- **无开源 TTS/STT/Conversational 模型** — 所有模型完全闭源
- GitHub 开源仅限 SDK 和工具,不包含模型权重或训练代码
- 这是所有 4 家公司中**最产品化、最闭源**的

### 3.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **估值和融资最高**: $11B 估值, $781M 总融资, 是 Voice AI 领域的绝对领导者; (2) **收入最高**: $330M+ ARR, 商业化成功远超其他所有 Speech LLM 公司; (3) **客户覆盖最广**: 41% Fortune 500, Meta, Epic Games, Salesforce, Duolingo, NVIDIA 等; (4) **多语言最强**: 32-70+ 语种; (5) **产品矩阵最完善**: TTS + STT + Dubbing + SFX + Music + Conversational AI, 从 API 到 Studio 到 Agent 平台 |
| **短板** | (1) **无公开学术论文**: 技术深度无法验证, 不参与学术社区; (2) **完全闭源**: 无任何模型权重公开, 与开源趋势背道而驰; (3) **技术架构未知**: 外部无法评估其与端到端 Speech LLM 方案 (Moshi, GLM-4-Voice) 的技术差距; (4) **Voice AI 竞争加剧**: OpenAI (GPT-4o Voice), Google (Gemini Voice), 字节 (Seed-TTS) 等巨头入场; (5) **收入依赖 API**: 类 SaaS 模式的 churn 风险; (6) Deepfake 伦理/监管风险 |
| **下一步推测** | (1) **IPO 准备** — 创始人已明确提及 "build toward IPO"; (2) 推出端到端语音对话模型 (对标 GPT-4o Voice Mode); (3) 更多企业 Agent 平台功能; (4) 可能的战略收购 (补齐 ASR/Speech LLM 能力); (5) "Audio General Intelligence" 研究方向推进 |

---

## 4. Deepgram — 语音理解基础设施

### 4.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | Deepgram (总部 San Francisco, 2015 年创立) |
| 团队 | Remote-first; 分布在 20+ 美国州 + 5+ 国家 |
| GitHub | [github.com/deepgram](https://github.com/deepgram) (114 repos, 637 followers) |
| HuggingFace | deepgram (有限公开模型) |
| 核心人物 | Scott Stephenson (Co-Founder, CEO; University of Michigan, 暗物质探测器波形分析出身), Noah Shutty (联合创始人) |
| 产品线 | Nova (ASR 系列: Nova-2, Nova-3), Flux (对话 ASR), Aura (TTS 系列: Aura, Aura-2), Voice Agent API, Audio Intelligence API |
| 官网 | [deepgram.com](https://deepgram.com) |
| 定位 | **Voice AI 基础设施平台**; 从端到端深度学习 ASR 起家,扩展到 TTS 和 Voice Agent; 企业 API 导向 |
| 融资 | 种子/早期轮 (2015-2022); $47M Series B extension (2023.03); $130M Series C (2026.01, $1.3B, 独角兽); 总融资 ~$215M+ |
| 收购 | OfOne (2026, 语音 AI 用于餐饮 drive-thru) |

### 4.2 产品/模型时间线 (2024-2026)

| 时间 | 产品/模型 | 来源 | 核心贡献 | 与 Speech LLM 关系 |
|------|---------|------|---------|-------------------|
| 2024 (持续) | Nova-2 | 产品文档 | 端到端深度学习 ASR; 30+ 语种; 多个垂直领域特化模型 (医疗/金融/电话/会议等) | 语音理解基座: 10 个垂直特化模型 |
| 2024 (具体不详) | Aura 1 | 产品发布 | 首版 TTS API | 语音生成能力起步 |
| 2025.04 | Aura-2 | 产品发布 (Blog) | 企业级 TTS; sub-200ms TTFB (优化至 90ms); 高级打断处理; end-of-thought 检测 | **TTS 升级**: 低延迟企业级 |
| 2025.06 | Voice Agent API | 产品发布 | 统一 STT+LLM+TTS 管线; Barge-in + Turn-taking; 自带 LLM 编排 | **核心产品**: 一站式 Voice Agent |
| 2025 (持续) | Nova-3 | 产品更新 | 50+ 语种 ASR; WER 降低 54.2% (streaming) / 47.4% (batch) vs 竞品; 自助词汇定制 (业界首创); 多语种 code-switching | **ASR SOTA**: 自助定制是独特能力 |
| 2025 | Nova-3 扩展 | 年度回顾 | 新增 German, Dutch, Swedish, Danish | 多语言持续扩展 |
| 2025 | Aura-2 获奖 | 年度回顾 | 2025 Contact Center Technology Award | 行业认可 |
| 2025 (具体不详) | Flux Multilingual | 产品发布 | 首个多语种对话语音识别模型; 模型集成 end-of-turn 检测; 10 语种 | **对话 ASR**: 专为 Voice Agent 设计 |
| 2026.01 | Series C ($1.3B) | 融资新闻 | $130M 融资, 独角兽 | 资本验证 |
| 2026.04 | 自托管更新 | 产品更新 | Nova-3 Gujarati, Aura-2 速度控制, 多语种数字格式化 | 持续迭代 |

### 4.3 技术栈全景

| 维度 | Deepgram 技术栈 |
|------|----------------|
| **语音编码器** | Nova-3: 自研端到端深度学习 ASR (非传统 pipeline); 50+ 语种; 多语种 code-switching |
| **LLM 骨干** | Voice Agent API: 内置 LLM 编排; 支持 BYO LLM (接入第三方 LLM) |
| **语音解码器** | Aura-2: 自研 TTS; sub-200ms TTFB (优化至 90ms); 支持 BYO TTS (接入第三方 TTS) |
| **对话策略** | Voice Agent API: 统一 STT+LLM+TTS pipeline; 模型驱动的 barge-in / turn-taking; 运行时编排 (mid-session 控制, prompt 更新, model switching); 支持 cloud/VPC/on-prem |
| **训练方法** | 端到端深度学习 (创始源自暗物质探测器波形分析); 非传统 ASR pipeline; 具体训练细节未公开 |
| **推理延迟** | Aura-2: sub-200ms TTFB (优化 90ms); Flux: ultra-low latency; Voice Agent: VAQI 71.5 (vs OpenAI 67.2, ElevenLabs 55.3) |
| **多语言** | Nova-3: 50+ 语种; Flux: 10 语种; Nova-2: 30+ 语种 |
| **情感/副语言** | 无显式情感能力; 聚焦于语音理解准确性和延迟 |

### 4.4 架构演进

```
暗物质探测器波形分析 (University of Michigan, 2015 前)
    ↓ 端到端深度学习应用于语音
早期 Deepgram ASR (自研端到端模型, 2015-2022)
    ↓
Nova-1 → Nova-2 (30+ 语种, 10 个垂直特化模型)
    ↓ 准确率大幅提升
Nova-3 (50+ 语种, WER 降低 54.2%, 自助词汇定制, 2025)
    
TTS 线:
Aura 1 (首版 TTS, 2024) → Aura-2 (企业级, sub-200ms, 2025.04)
    
对话线:
Flux (对话 ASR, 模型集成 turn detection) + Voice Agent API (统一 STT+LLM+TTS, 2025.06)
    ↓
"Powered by Deepgram" 品牌 + OfOne 收购 (餐饮 drive-thru)
```

### 4.5 独特技术赌注

1. **端到端深度学习 ASR (从零开始)**: Deepgram 不使用传统 ASR pipeline (acoustic model + language model + decoder), 而是完全端到端的深度学习。这种路线在 2015 年极为激进, 现在被证明是正确的。
2. **自助词汇定制 (Self-serve Vocabulary Customization)**: "首个提供自助定制的 Voice AI 模型" — 用户可以不经重训练直接定制术语。这对企业客户 (医疗/金融/法律) 极为重要。
3. **VAQI (Voice Agent Quality Index)**: 自定义评估指标 (延迟 + 打断率 + 响应覆盖率), Deepgram 在此指标上超越 OpenAI 和 ElevenLabs。这反映了 Deepgram 对端到端语音对话质量的关注, 而非单一组件性能。
4. **BYO (Bring Your Own) 策略**: 用户可替换 Voice Agent API 中的任何组件 (LLM, TTS), 同时保留 Deepgram 的编排和流式管线。这种模块化策略降低了客户锁定风险。
5. **垂直特化**: 10 个垂直领域特化 ASR 模型 (医疗/金融/电话/会议/drive-thru/汽车/ATC 等), 这在竞争对手中独一无二。

### 4.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| deepgram-python-sdk | 439 | 官方 Python SDK |
| deepgram-js-sdk | 262 | 官方 JavaScript SDK |
| deepgram-go-sdk | 85 | 官方 Go SDK |
| deepgram-rust-sdk | 66 | 社区 Rust SDK |
| deepgram-dotnet-sdk | 53 | 官方 .NET SDK |
| recipes | 25 | 使用示例 |
| deepgram-api-specs | 7 | API 规范 |

**注意**:
- **无开源 ASR/TTS 模型** — Nova-3, Aura-2 等核心模型完全闭源
- GitHub 开源仅限 SDK 和工具
- 无公开学术论文

### 4.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **ASR 技术最强**: Nova-3 在 50+ 语种上显著领先竞品 (WER 降低 54.2%); (2) **企业级部署能力**: 支持 cloud/VPC/on-prem, 满足合规需求; (3) **Voice Agent API 先发优势**: 2025.06 发布, VAQI 超越 OpenAI 和 ElevenLabs; (4) **垂直特化**: 10 个行业特化模型是独特壁垒; (5) **延迟领先**: Aura-2 sub-200ms TTFB (90ms 优化), Flux ultra-low latency; (6) 200K+ 开发者, 处理 50,000+ 年的音频 |
| **短板** | (1) **TTS 能力相对弱**: Aura-2 起步晚 (2025.04), 在 TTS 自然度上可能不及 ElevenLabs; (2) **无学术论文**: 技术深度无法被外部评估; (3) **完全闭源**: 与开源趋势不符; (4) **融资规模有限**: $215M vs ElevenLabs $781M; (5) **无自研 LLM**: Voice Agent API 的 LLM 编排依赖第三方; (6) **品牌知名度不及 ElevenLabs** |
| **下一步推测** | (1) Nova-4 可能引入更多语种和更强的上下文理解; (2) Aura-3 可能大幅提升 TTS 自然度 (缩小与 ElevenLabs 的差距); (3) Voice Agent API 可能整合更多 LLM 能力 (可能自研); (4) OfOne 收购后在餐饮/零售垂直深耕; (5) 可能向端到端 Speech LLM 方向发展 (从 ASR 基础设施转向完整对话系统) |

---

## 5. 横向对比矩阵

### 5.1 技术路线对比

| 维度 | Fixie AI (Ultravox) | Hume AI | ElevenLabs | Deepgram |
|------|-------------------|---------|------------|---------|
| **核心定位** | 开源 Speech LLM (理解) | 情感语音 AI | 全音频栈商业平台 | 语音理解基础设施 |
| **架构范式** | Audio encoder + Multimodal Projector + LLM (text output) | Emotion-aware Speech-to-Speech; TADA 1:1 Alignment | 闭源 TTS/STT/Conversational (产品优先) | 端到端深度学习 ASR + TTS + Voice Agent |
| **LLM 骨干** | 可插拔: GLM 4.6 355B (v0.7) / Llama 3.3 70B | TADA: Llama 3.2 1B/3B; EVI: 内部模型 | 内部自研 (未公开) | BYO LLM (编排层) |
| **自研 TTS** | 无 (外接第三方) | Octave + TADA (开源) | 核心能力 (Multilingual v2 / Turbo v2.5 / v3) | Aura-2 (sub-200ms) |
| **自研 ASR** | 无 (Whisper encoder) | 无 (Expression Measurement 不是 ASR) | 有 (具体不详) | 核心能力 (Nova-3, 50+ 语种 SOTA) |
| **全双工** | 不支持 (VAD-based barge-in) | EVI 支持自然打断 (非严格全双工) | 公开信息有限 | Voice Agent API 支持 barge-in |
| **情感能力** | 无 | **核心差异化** (48 维, RLHE) | 隐式 (模型推断) | 无 |
| **多语言** | 26 语种 | EVI: 11 语种; TADA: 9 语种 | 32-70+ 语种 | Nova-3: 50+ 语种 |
| **开源程度** | ★★★★★ (模型权重 + 训练代码) | ★★☆☆☆ (仅 TADA 开源) | ★☆☆☆☆ (仅 SDK) | ★☆☆☆☆ (仅 SDK) |
| **学术论文** | 1 篇核心 (DiVA, ACL 2025) | 1 篇核心 (TADA, arXiv 2026) + 多篇情感 | **0 篇** | **0 篇** |

### 5.2 商业规模对比

| 指标 | Fixie AI | Hume AI | ElevenLabs | Deepgram |
|------|---------|---------|------------|---------|
| **估值** | 未公开 (小型) | $219M (2024.03) | **$11B** (2026.02) | **$1.3B** (2026.01) |
| **总融资** | $17M | $72.8M | **$781M** | ~$215M |
| **ARR** | 未公开 | 未公开 (~$100M 预测) | **$330M+** | 未公开 |
| **开发者/用户** | 未公开 | 100K+ | 1M+ 注册用户 | 200K+ 开发者 |
| **团队规模** | 小型 (<50?) | ~50+ | **~223** | 中型 |
| **创立年份** | 2022 | 2021 | 2022 | **2015** (最早) |

### 5.3 GitHub Stars 对比

| 团队 | 最高 Stars 项目 | Stars | 总 Stars (相关) |
|------|---------------|-------|----------------|
| Fixie AI | ultravox | **4,400** | ~4,600 |
| Hume AI | tada | 989 | ~1,600 |
| ElevenLabs | elevenlabs-python | 2,996 | ~8,400 (SDK/工具) |
| Deepgram | deepgram-python-sdk | 439 | ~900 (SDK/工具) |

**注意**: ElevenLabs 和 Deepgram 的 stars 主要来自 SDK/工具,非模型本身。Fixie AI 的 ultravox 是唯一开源的 Speech LLM 模型。

### 5.4 技术栈覆盖对比

| 能力 | Fixie AI | Hume AI | ElevenLabs | Deepgram |
|------|---------|---------|------------|---------|
| 语音理解 (ASR/SLU) | ✓ (Whisper + LLM) | 部分 (情感分析) | ✓ (STT API) | **✓✓** (Nova-3 SOTA) |
| 语音生成 (TTS) | ✗ (外接) | ✓ (Octave/TADA) | **✓✓** (最强 TTS) | ✓ (Aura-2) |
| 语音对话 | ✓ (平台层) | ✓ (EVI) | ✓ (Conversational AI) | ✓ (Voice Agent API) |
| 情感识别 | ✗ | **✓✓** (核心) | ✗ (隐式) | ✗ |
| 语音克隆 | ✓ (平台) | ✓ (Octave) | **✓✓** (核心) | ✗ |
| 配音/翻译 | ✗ | ✗ | **✓✓** (Dubbing Studio) | ✗ |
| 音效/音乐 | ✗ | ✗ | ✓ (SFX + Music) | ✗ |

---

## 6. 关键发现

### 发现 1: "全栈"vs"专精"的路线分化

四家公司呈现清晰的路线分化:

- **ElevenLabs** 走"全音频栈"路线 — 从 TTS 扩展到 STT, Dubbing, SFX, Music, Conversational AI,目标是成为"Audio General Intelligence"平台。$781M 融资和 $330M ARR 支撑这一战略。
- **Deepgram** 走"语音理解基础设施"路线 — 以 ASR 为核心向 TTS 和 Voice Agent 扩展,强调企业部署和垂直特化。
- **Fixie AI** 走"开源 Speech LLM"路线 — 专注语音理解 + LLM 推理,不做生成,是最纯粹的 Speech LLM 公司。
- **Hume AI** 走"情感 AI"路线 — 唯一将情感理解作为核心差异化的公司,横跨理解和生成。

这种分化反映了 Voice AI 市场的碎片化 — 目前没有任何一家公司能在所有维度 (理解/生成/情感/多语言/低延迟) 上同时领先。

### 发现 2: 学术论文极度稀缺

与 Session 1-4 的中国/法国公司形成鲜明对比,本 Session 的 4 家公司**几乎没有学术论文产出**:
- Fixie AI: 1 篇 (DiVA, ACL 2025)
- Hume AI: 1 篇 (TADA, arXiv 2026) + 情感计算论文 (非 Speech LLM 核心)
- ElevenLabs: **0 篇**
- Deepgram: **0 篇**

这意味着我们**无法从学术角度评估** ElevenLabs 和 Deepgram 的核心技术深度。它们的技术壁垒是通过产品化和数据飞轮而非公开研究建立的。这也反映了美国创业公司更倾向于将技术作为商业机密,而非学术贡献。

### 发现 3: 估值鸿沟反映市场定价逻辑

| 公司 | 估值 | 核心驱动 |
|------|------|---------|
| ElevenLabs | $11B | **收入驱动** ($330M ARR, ~33x 收入倍数) |
| Deepgram | $1.3B | **基础设施定位** (200K 开发者, 50K 年音频) |
| Hume AI | $219M | **技术/IP 驱动** (情感 AI 差异化, 学术基础) |
| Fixie AI | 未公开 (小) | **社区驱动** (4.4K stars, 开源) |

市场明确表示: 在 Voice AI 领域,**收入规模 >> 技术创新 >> 开源社区**。ElevenLabs 的估值是 Hume 的 50x,尽管 Hume 的学术基础和技术独特性可能更强。

### 发现 4: "对话"成为必争之地

所有 4 家公司都在 2025-2026 年加速进入 Conversational AI / Voice Agent:
- ElevenLabs: Eleven v3 Conversational + ElevenAgents (企业)
- Deepgram: Voice Agent API (2025.06, VAQI SOTA)
- Hume AI: EVI 3 → EV4-mini
- Fixie AI: Ultravox Realtime API + tools/RAG

这表明**单一组件 (TTS 或 ASR) 不再是可持续的商业模式** — 市场正在向"完整对话系统"集中。Deepgram 从 ASR 向 Voice Agent 的转型、ElevenLabs 从 TTS 向 Conversational AI 的扩展,都印证了这一趋势。

### 发现 5: 开源 vs 闭源的生态割裂

| | 开源阵营 | 闭源阵营 |
|--|---------|---------|
| **代表** | Fixie AI (Ultravox), Hume AI (TADA) | ElevenLabs, Deepgram |
| **优势** | 社区验证, 学术引用, 开发者信任 | 数据飞轮, 商业秘密, 收入规模 |
| **风险** | 难以盈利, 技术被复制 | 缺乏学术验证, 难以吸引研究者 |

在 Speech LLM 领域,闭源公司 (ElevenLabs, Deepgram) 在收入和估值上远超开源公司 (Fixie AI, Hume AI)。但开源项目 (Ultravox 4.4K stars, TADA 989 stars) 在技术影响力和社区建设上有独特价值。这种"闭源赚钱,开源影响"的格局可能持续。

---

## 附: 数据来源标注

| 信息类型 | 来源 | 可信度 |
|---------|------|--------|
| GitHub stars/repos | GitHub API (2026-06-08 实时查询 via Playwright) | 高 |
| 融资/估值数据 | 官方公告, Crunchbase, TechCrunch, Reuters, CNBC | 高 |
| 产品信息/技术栈 | 官方网站, 产品文档, 官方博客 | 高 |
| arXiv 论文 | arXiv (2026-06-08 查询) | 高 |
| Hume AI 详细信息 | Contrary Research (contrary.com), 官方网站 | 高 |
| ElevenLabs 详细信息 | Contrary Research, 融资新闻, 官方博客 | 高 |
| Deepgram Voice Agent API | 官方文档 (deepgram.com) | 高 |
| 团队人物 | Crunchbase, LinkedIn, GitHub org | 中 |
| 收入数据 (ARR) | 融资新闻报道引用 (非经审计) | 中 |
| 技术架构推断 | 基于公开 API 文档和产品描述 (非论文) | 中-低 |
| ElevenLabs/Deepgram 核心模型架构 | **公开信息极度有限** — 无论文, 无技术报告 | 低 |
