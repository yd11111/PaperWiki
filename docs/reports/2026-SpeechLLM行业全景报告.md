# 2026 Speech LLM / Omni / 全双工对话 行业全景报告

> 调研时间: 2026-06-08 | 覆盖: ~36 团队/实验室 | 方法: 10 Session 原始数据交叉汇总 | 原始数据: 10 个 Session, ~5083 行

---

## 一、技术路线图

### 1.1 四大架构范式

2024 年 GPT-4o 定义端到端语音交互之后,行业形成四条架构路线:

```
① 原生端到端统一:  文本/语音/图像 → [单一 Transformer] → 文本/语音/图像
                   代表: GPT-4o, Gemini 3.1, Qwen3.5-Omni, GLM-4-Voice, ERNIE 5.0

② 模块化端到端:    语音 → [Whisper frozen] → [LLM] → [Flow Matching/CosyVoice decoder]
                   代表: Fun-Audio-Chat, Kimi-Audio, LLaMA-Omni, Higgs Audio, Ultravox

③ Thinker-Talker:  语音 → [Thinker: 文本空间推理] → [Talker: 语音生成]
                   代表: Qwen3.5-Omni (ARIA 框架)

④ Pipeline 级联:   语音 → [ASR] → [LLM] → [TTS] → 语音
                   状态: 部署主流,研究退场
```

**判断**: ②模块化端到端是当前最务实的路线 — 端到端训练保留副语言,模块化推理便于组件升级。①统一模型只有大厂有资源做。④Pipeline 在研究中已死,部署中仍占 60%+。[S1, S3, S7, S8, S10]

### 1.2 语音 Tokenizer — 当前最大技术路线之争

| 路线 | 代表 | 团队 | 优势 | 劣势 | Session |
|------|------|------|------|------|---------|
| **离散 semantic tokens** | S3Tokenizer, HuBERT VQ, USTokenizer | 阿里, Meta, 清华 | LLM 自回归兼容,生态成熟 | 信息瓶颈,高帧率增加序列长度 | S1, S7, S9 |
| **超低码率离散** | 175bps ASR-VQ, FlexiCodec | 智谱, 港中深 | 序列压缩→LLM 效率 | 副语言细节可能丢失 | S1, S8 |
| **连续表征** | Ming-UniAudio, MELLE | 蚂蚁, CUHK | 无量化损失 | 与 LLM 离散词表不兼容 | S3, S9 |
| **Tokenizer-free** | VoxCPM | 清华/OpenBMB | 无显式 tokenizer,直接 mel→LLM | 复现难,主流生态不兼容 | S3, S9 |
| **双 codebook** | Dual-codebook | 阶跃 | 语义+声学并行 | 独创,生态窄 | S1 |

**判断**: 短期离散 semantic tokens 仍占主导 (Qwen3.5-Omni 采用 S3Tokenizer 是最强信号)。连续/Tokenizer-free 路线有理论优势但缺大规模验证。超低帧率 (12.5Hz→6.25Hz) 是离散路线的内部演进方向。[S1, S3, S8, S9, S10]

### 1.3 全双工对话 — 六条路线并行,零收敛

| 路线 | 代表系统 | 团队 | 核心机制 | Session |
|------|---------|------|---------|---------|
| **多流并行** | Moshi | Kyutai | Inner Monologue + 用户/系统双流并行解码 | S4 |
| **DRSR 双分辨率** | Fun-Audio-Chat | 阿里 FunAudioLLM | 5Hz LLM 规划 + 25Hz SRH 执行 | S1 |
| **Codec-free** | SALMONN-omni | 字节/清华 | 去掉离散 codec,直接建模波形 | S1 |
| **Intelligence-Speaker 分离** | Covo-Audio | 腾讯 | 智能体与说话人解耦,99.7% 轮替准确率 | S2 |
| **中间融合** | LSLM | X-LANCE/SJTU | TTS 生成流中途融入用户语音 | S8 |
| **Channel Routing** | 清华全双工论文 | 清华/CUHK | Channel Fusion vs Cross-Attention 路由 | S9 |

**判断**: 全双工是当前分歧最大的方向。每个团队路线完全不同,且缺乏统一 benchmark (每家自定义指标,无法横向比较)。预计 2027 年出现标准 benchmark 后才开始收敛。[S1, S2, S4, S8, S9, S10]

#### 全双工技术成熟度

| 维度 | 当前水平 | 差距 | Session |
|------|---------|------|---------|
| 打断处理 | 基本可用 (Covo-Audio 99.7%) | 语义级打断 (理解打断意图) 未解决 | S2 |
| 轮替管理 | 规则+模型混合 | 自然过渡 (无 "OK" 提示) 仍困难 | S10 |
| 延迟 | 200-500ms 首包 | 人类对话 ~200ms,仍有差距 | S1 |
| 并行理解+生成 | Moshi 实现 | 理解质量在并行时下降 | S4 |
| 情感连续性 | 几乎未解决 | 全双工中情感状态追踪是开放问题 | S9, S10 |
| 标准化评估 | **不存在** | 每个团队自定义,无法横向比较 | S10 |

### 1.4 技术栈收敛点

经过 36 团队交叉比对,以下技术选择已形成明确共识:

| 收敛点 | 共识内容 | 证据 | Session |
|--------|---------|------|---------|
| **LLM 骨干** | Qwen (国内) + LLaMA (国际) 二选一 | Qwen: 6/8 国内团队; LLaMA: LLaMA-Omni/SALMONN/Ultravox/Higgs | S1-S9 |
| **语音编码器** | Whisper-Large-v3 (frozen) + Adapter | Fun-Audio-Chat/LLaMA-Omni/Kimi-Audio/Ultravox/Higgs 均使用 | S1, S3, S5, S6, S8 |
| **语音生成** | Flow Matching (OT-CFM) | 取代 DDPM/DDIM,几乎所有 2025-2026 新系统默认使用 | S1, S3, S4, S10 |
| **训练 pipeline** | 模态对齐 → 多任务 SFT → DPO/RLHF | Fun-Audio-Chat (Multi-Task DPO), StepAudio (三模式 RLHF), 智谱 (GRPO) | S1, S2, S10 |
| **Post-training** | 从 1 家到 5+ 家,但方法论仍分散 | 字节 SpeechJudge, 阿里 DiffRO, 阶跃三模式, 智谱 GRPO | S1, S10 |

### 1.5 MoE vs Dense — 未定之争

| 路线 | 代表 | 总参数 / 激活参数 | Session |
|------|------|:---:|---------|
| MoE | Fun-Audio-Chat, Qwen3.5-Omni, StepAudio 2.5 | 30B-A3B 级 | S1 |
| Dense | LLaMA-Omni (8B), SALMONN-omni (13B), Kimi-Audio (7B) | 7-13B | S1, S3, S8 |

**判断**: MoE 在语音任务的优势尚未被充分验证。阿里/阶跃押注 MoE,但学术界和多数创业公司仍用 Dense (MoE 训练基础设施门槛高)。[S1, S10]

### 1.6 Speech LLM vs Cascaded — 竞争现状

| 维度 | Speech LLM (端到端) | Cascaded (ASR→LLM→TTS) | Session |
|------|-------------------|----------------------|---------|
| 理解质量 | 接近 (Qwen3.5-Omni 逼近 Whisper+GPT-4) | 仍领先 | S1, S10 |
| 生成质量 | 快速追赶 (VoxCPM UTMOS 4.37) | 领先 (专用 TTS 质量更高) | S3, S10 |
| 延迟 | **优势** (单次推理, ~200ms 首包) | 劣势 (3 次推理叠加) | S1, S10 |
| 副语言保留 | **优势** (端到端保留语气/情感) | 劣势 (ASR 丢失副语言) | S10 |
| 全双工 | **可实现** (Moshi, Fun-Audio-Chat) | 极难 (pipeline 延迟太高) | S4, S10 |
| 可解释性 | 劣势 (黑盒) | 优势 (中间文本可审计) | S10 |
| 部署现状 | 少数 (GPT-4o, Gemini) | **主流** (60%+ 产品) | S10 |

**转折点预测**: 当 Speech LLM 的 WER + TTS 质量同时达到 Cascaded 的 95% 时,延迟优势将使其成为默认。预计 **2026 年底 - 2027 年初**。[S10]

**"伪端到端"现象**: 多个系统看似端到端,实为模块化 — Fun-Audio-Chat = Whisper(frozen) + LLM + CosyVoice(frozen); Kimi-Audio = Whisper + Qwen2-7B + FM decoder; Qwen3.5-Omni 内部仍是 Thinker + Talker 分离。"端到端训练 + 模块化推理"正在成为最优折中。[S1, S3, S10]

---

## 二、团队能力矩阵

### 2.1 国内工业界 (14 个团队)

| 团队 | 核心系统 | LLM 骨干 | 全双工 | 独特赌注 | 开源 Stars | Session |
|------|---------|---------|:---:|---|:---:|---------|
| **阿里 FunAudioLLM** | Fun-Audio-Chat, CosyVoice 3 | Qwen3-30B-A3B | DRSR 全双工 | 监督 token + DiffRO + DRSR | 21.5K (CosyVoice) | S1 |
| **阿里 Qwen** | Qwen3.5-Omni | MoE (自研) | Thinker-Talker | ARIA 对齐 + 215 benchmarks | — | S1 |
| **字节跳动** | SALMONN-omni, VibeVoice | LLaMA 系 | Codec-free | Codec-free 全双工 + Seed-TTS 生态 | — | S1 |
| **智谱** | GLM-4-Voice | GLM-4 | Streaming Thoughts | 175bps 超低码率 + 合成交错数据 | 3.2K | S1 |
| **阶跃星辰** | StepAudio 2.5 | MoE (自研) | 实时 | 三模式 RLHF + MTP-5 + RTF 0.0053 | 1.5K | S1 |
| **百度** | ERNIE 5.0, Eureka-Audio | 自研 | 未知 | 万亿参数统一 Omni | — | S2 |
| **讯飞** | Spark 系列 | 未公开 | 未知 | 市场份额大,公开研究近零 | 0 | S2 |
| **腾讯** | Covo-Audio 7B | LLaMA 系 | Intelligence-Speaker 分离 | 99.7% 轮替准确率 | — | S2 |
| **小米** | MiMo-Audio 7B, OmniVoice | Qwen | ZipVoice NAR | Daniel Povey 加盟效应 | — | S2 |
| **月之暗面** | Kimi-Audio 7B | Qwen2-7B | 未公开 | 13M 小时训练数据 + hybrid 输入 | 4.6K | S3 |
| **MiniMax** | MiniMax-Speech | 未公开 | 无 | TTS Arena #1,零 Speech LLM 论文 | 0 | S3 |
| **面壁/OpenBMB** | VoxCPM | MiniCPM | 无 | Tokenizer-free, ICLR 2026 Spotlight | 27.5K | S3 |
| **蚂蚁** | Ming-UniAudio | 16B MoE | Ming-Omni | 连续 VAE tokenizer + free-form 编辑 | — | S3 |
| **Boson AI** | Higgs Audio | LLaMA 系 | 无 | Evaluation-first + EmergentTTS-Eval | 8.1K | S6 |

### 2.2 国际工业界 (14 个团队)

| 团队 | 定位 | 核心能力 | 融资/估值 | 开源 Stars | 独特赌注 | Session |
|------|------|---------|----------|:---:|---|---------|
| **OpenAI** | 闭源标杆 | GPT-4o 原生多模态, Realtime API | 大厂 | 102K (Whisper) | 定义端到端范式 | S7 |
| **Google** | 闭源标杆 | Gemini 原生, USM 300+ 语种, 9.5hr 上下文 | 大厂 | — | Audio Tags + SynthID + 超长上下文 | S7 |
| **Meta** | 开源先锋 | EnCodec + Spirit-LM + Seamless 101 语种 | 大厂 | 39K+ (累计) | 最激进开源,但 4/6 repo archived | S7 |
| **Microsoft** | 范式创造者 | VALL-E (Codec LM 鼻祖), NaturalSpeech 3, Phi-4 | 大厂 | — | 核心团队已解体,Azure 140+ 语种 | S7 |
| **Apple** | 严重落后 | 无公开 Speech LLM 研究 | 大厂 | 0 | 端侧优先,潜在收购需求 | S7 |
| **ElevenLabs** | 商业霸主 | 全音频栈 (TTS+配音+Agent) | **$11B** / $781M | 0 | 零论文 + $330M ARR | S5 |
| **Kyutai** | 全双工先驱 | Moshi (Inner Monologue + Mimi codec) | 非营利 €300M | 10.4K | 发明全双工,DSM 框架 | S4 |
| **Deepgram** | ASR API | Nova-3 50+ 语种, Voice Agent API | $1.3B | — | 理解侧深耕 | S5 |
| **Hume AI** | 情感 AI | TADA 1:1 对齐, RLHE 情感强化学习 | $72.8M (Series B) | — | 情感优先 | S5 |
| **Sesame** | 对话韵律 | CSM (Voice Presence), 计划硬件 | ~$100M+ | 14.7K | 韵律差异化 + 硬件雄心 | S4 |
| **Cartesia** | SSM 路线 | Mamba 架构, Sonic TTS 40ms 延迟 | 已融多轮 | — | SSM 替代 Transformer | S4 |
| **Fixie AI** | 开源理解 | Ultravox (backbone-agnostic, 纯理解) | $17M Seed | 4.4K | 理解侧专精 | S5 |
| **Resemble AI** | TTS + 安全 | Chatterbox TTS + 深伪检测双轨 | — | 25K | 生成+检测对冲 | S6 |
| **AssemblyAI** | ASR API | 600M Conformer, Universal-2 | $115M+ | — | 最大独立 ASR 模型 | S6 |

### 2.3 学术实验室 (8 个)

| 实验室 | 核心人物 | 标杆成果 | 核心方向 | 顶会 | 影响力 | Session |
|--------|---------|---------|---------|------|--------|---------|
| **ICT/CAS** | 葛万锋 | LLaMA-Omni (ICLR 2025), TARS (ACL 2026) | 数据高效 Speech LLM (仅 200K 样本) | ICLR + ACL | 急升 | S8 |
| **X-LANCE/SJTU** | 吴梦玥 | LSLM (首个全双工探索), SLAM-LLM, Audio Interaction Model | 全双工 + 语音交互系统 | ACL 2026 | 稳升 | S8 |
| **CUHK-SZ 港中深** | 武执政 | MaskGCT (ICLR, 210 cit.), Amphion 9.8K stars, Emilia 20万hr | 非 AR 生成 + 低帧率 tokenizer + 基础设施 | ICLR, NeurIPS | 急升 | S8 |
| **清华 THUHCSI** | 吴志勇 | VoxCPM 27.5K stars (ICLR Spotlight), DualSpeechLM, UniAudio | Tokenizer-free + 统一理解/生成 | ICLR, AAAI | 急升 | S9 |
| **CUHK Helen Meng** | Helen Meng, 吴习印 | EmotionThinker (**ICLR 2026 Oral**), MELLE, ReasoningCodec | 情感建模 + 连续 AR + 统一模型 | ICLR Oral | 急升 | S9 |
| **NTU** | 李宏毅 | SUPERB, Dynamic-SUPERB | 评测标准制定 | ICLR | 持续高位 | S10 |
| **东京大学** | — | J-CHAT 76K hrs, 语音 tokenization 理论 | 日语数据集 + tokenizer 理论 | — | 稳定 | S9 |
| **NII (日本)** | — | 深伪检测竞赛 | 语音安全 | — | 稳定 | S9 |

**注**: LLaMA-Omni 出自 ICT/中科院,**非** CMU。SpeechGPT 出自复旦,**非**清华。

### 2.4 综合领先度排名

| 维度 | Top 3 | Session |
|------|-------|---------|
| **全栈能力 (理解+生成+对话+全双工)** | 阿里 FunAudioLLM > 字节 > OpenAI | S1, S7 |
| **全双工技术深度** | Kyutai (Moshi) > 阿里 (DRSR) > 腾讯 (Covo-Audio) | S1, S2, S4 |
| **开源影响力** | Whisper 102K > VoxCPM 27.5K > Chatterbox 25K | S3, S6, S7 |
| **商业化收入** | ElevenLabs ($330M ARR) >> OpenAI >> Google | S5, S7 |
| **学术产出 (Speech LLM)** | 清华 > CUHK Helen Meng > ICT/CAS | S8, S9 |
| **训练数据规模** | VoxCPM2 200万hr > CosyVoice 3 100万hr > Kimi-Audio 13M hr | S1, S3, S9 |
| **评测标准主导** | NTU (SUPERB) > Boson AI (EmergentTTS-Eval) > 字节 (Seed-TTS-Eval) | S6, S10 |

---

## 三、技术趋势判断

### 3.1 上升趋势

| # | 方向 | 2024 参与者 | 2026 参与者 | 关键信号 | Session |
|---|------|:---:|:---:|---|---------|
| 1 | **全双工对话** | 2 (Moshi+LSLM) | 7+ | 6 条技术路线并行,但无标准 benchmark | S1, S2, S4, S8, S9 |
| 2 | **情感/副语言建模** | 1 (Hume TADA) | 5+ | EmotionThinker ICLR 2026 **Oral**; Hume $72.8M | S5, S9 |
| 3 | **Post-training 对齐** | 1 (字节 RL) | 5+ | DiffRO/SpeechJudge/GRPO 独立创新涌现 | S1, S10 |
| 4 | **推理效率** | 0 | 5+ | DRSR 5Hz (50% GPU 减), Speculative Decoding, ZipVoice NAR | S1, S2, S9 |
| 5 | **Voice Agent 框架** | 1 (OpenAI Realtime API) | 5+ | LiveKit 19K + Pipecat 12.7K stars, 占 YC 22% | S6, S7 |
| 6 | **语音安全** | 少量 | 多方 | NII 深伪检测 + 清华 VGuard + Google SynthID | S7, S9 |

### 3.2 稳定/饱和方向

| 方向 | 说明 | Session |
|------|------|---------|
| 零样本语音克隆 | 3-10 秒参考已成基线,不再差异化 | S1, S4 |
| 多语言 ASR (30+) | Whisper/SenseVoice/USM 已覆盖,增量空间有限 | S1, S7 |
| 基础 Speech LLM 理解 | VoiceBench 上开源系统逼近 GPT-4o,差距快速收窄 | S10 |

### 3.3 下降趋势

| # | 方向 | 证据 | Session |
|---|------|------|---------|
| 1 | **纯 Pipeline (ASR→LLM→TTS)** | 所有 2025-2026 新论文都做端到端或模块化端到端 | S10 |
| 2 | **纯 ASR/纯 TTS 独立研究** | 被 Speech LLM 理解/生成两侧吸收,顶会独立论文减少 | S10 |
| 3 | **GAN 声码器** | HiFi-GAN 仍用但不再是研究方向,被 Flow Matching 取代 | S10 |
| 4 | **微软语音研究** | Xu Tan→Moonshot, 核心团队解体, VALL-E/NaturalSpeech 停更 | S7 |
| 5 | **Meta FAIR 语音** | 6 个 speech repo 中 4 个 archived, 收购 PlayHT 替代自研 | S7 |
| 6 | **闭源模型的研究影响力** | GPT-4o 强但无法复现; 开源 (Kimi-Audio, Fun-Audio-Chat) 在 benchmark 上逼近 | S10 |

### 3.4 三大范式转移

**范式 1: 独立语音系统 → LLM 原生语音能力**

```
旧: 独立 ASR + 独立 TTS (Whisper + VITS/CosyVoice)
     ↓ GPT-4o 催化
新: 语音是 LLM 原生输入/输出模态 (Qwen3.5-Omni, Gemini, GPT-4o)
状态: 过渡期 — 模块化端到端是当前最优折中
```

**范式 2: 离散 token 垄断 → 表征路线多元化**

```
旧: 离散 token 是唯一选项 (EnCodec, HuBERT VQ)
     ↓ VoxCPM + Ming-UniAudio 挑战
新: 离散 / 连续 / Tokenizer-free / 超低帧率 四条路线竞争
状态: 活跃分化期 — 离散仍主导但不再是唯一
```

**范式 3: 单轮交互 → 全双工持续对话**

```
旧: 用户说完 → 系统回复 (半双工)
     ↓ Moshi/LSLM 概念验证
新: 持续双向语音流 (理解+生成同时进行)
状态: 早期 — 6+ 条路线,无标准,无收敛
```

---

## 四、空白地带

| # | 空白 | 现状 | 潜在价值 | 难度 | Session |
|---|------|------|----------|------|---------|
| 1 | **全双工标准化评估** | 6+ 团队自定义指标,无法横向比较 | 极高 — 标准定义者将主导方向收敛 | 中 | S10 |
| 2 | **长对话一致性 (10+ 轮)** | 所有 demo 都是 1-5 轮,无人报告质量衰减 | 高 — 实际部署核心问题 | 高 | S10 |
| 3 | **语音 Chain-of-Thought** | 推理在文本空间完成 (Thinker-Talker),非语音空间 | 高 — 文本 CoT 是 2025 最大热点,语音几乎空白 | 极高 | S10 |
| 4 | **语音安全 (jailbreak/deepfake)** | 仅清华 VGuard + NII 深伪检测 | 高 — 语音克隆门槛已极低 (Chatterbox 25K stars) | 中 | S9, S10 |
| 5 | **非英中 Speech LLM** | 日语仅 J-CHAT 76K hrs,其他语种几乎空白 | 高 — USM 300+ 语种 ASR 未延伸到生成侧 | 中 | S7, S9 |
| 6 | **多方对话 (3+ 人)** | 所有全双工研究都是 1:1 | 中 — 会议/播客需求大 | 高 | S10 |
| 7 | **个性化/记忆 Speech LLM** | 全部无状态,不记住用户偏好 | 中 — 产品需求强,无学术研究 | 中 | S10 |
| 8 | **端到端延迟标准化** | 各家报告口径不同 (首包 vs 端到端 vs RTF) | 高 — 无法公平比较系统 | 低 | S10 |

---

## 五、时间线 (2023 → 2024 → 2025 → 2026)

### 2023: 概念验证年

```
05月 ── SpeechGPT (复旦): 首个端到端 Speech LLM,证明 LLM 可直接处理语音 token
06月 ── AudioPaLM (Google): 首个大规模多模态音频-文本 LLM
10月 ── UniAudio (清华/CUHK): 统一 11 种音频任务
```

### 2024: 范式定义年

```
03月 ── GPT-4o (OpenAI): 原生多模态语音,定义行业标准
03月 ── Gemini 1.5 Pro (Google): 原生音频理解
06月 ── LSLM (X-LANCE/SJTU): 首个全双工学术探索
07月 ── FunAudioLLM (阿里): CosyVoice + SenseVoice 双基座开源,奠定国内生态
07月 ── SALMONN v2 (字节/清华): 双编码器理解架构
09月 ── Moshi (Kyutai): 首个完整全双工系统开源,€300M 非营利,里程碑
10月 ── GLM-4-Voice (智谱): 合成交错数据训练的端到端 Omni
11月 ── Qwen2-Audio (阿里): 音频理解基座
12月 ── LLaMA-Omni (ICT/CAS): 仅 200K 样本的数据高效 Speech LLM → ICLR 2025
```

### 2025: 工程爆发年

```
01月 ── Kimi-Audio (月暗): 13M 小时训练,开源 4.6K stars
02月 ── Spirit-LM (Meta): 交错语音-文本 LM → ICLR 2025
04月 ── StepAudio 2.5 (阶跃): 统一 MoE + 三模式 RLHF
05月 ── CosyVoice 3 (阿里): 100万小时 + DiffRO → Qwen3.5-Omni Talker 基础
06月 ── Qwen3.5-Omni (阿里 Qwen): ARIA + Thinker-Talker,215 benchmarks
06月 ── ERNIE 5.0 (百度): 万亿参数统一 Omni
08月 ── DualSpeechLM (清华/CUHK): USTokenizer → AAAI 2026
09月 ── VoxCPM (清华/OpenBMB): Tokenizer-free,27.5K stars → ICLR 2026 Spotlight
09月 ── Chatterbox (Resemble AI): 开源对话 TTS,25K stars 爆发
10月 ── EmotionThinker (CUHK Helen Meng): ICLR 2026 Oral,情感 Speech LLM
11月 ── SALMONN-omni (字节): Codec-free 全双工
12月 ── Fun-Audio-Chat (阿里): DRSR + 全双工旗舰
12月 ── Covo-Audio (腾讯): Intelligence-Speaker 分离,99.7% 轮替
```

### 2026: 分化年

```
01月 ── ElevenLabs 估值 $11B: 语音 AI 商业化里程碑,零论文
02月 ── UniAudio 2.0 (清华/CUHK): ReasoningCodec + 160B tokens
05月 ── 清华/CUHK 全双工论文: Channel routing 策略系统研究
```

---

## 六、投资与市场

### 6.1 融资概览

| 公司 | 估值 | 累计融资 | 最新轮次 | 核心定位 | Session |
|------|:---:|:---:|---|---|---------|
| **ElevenLabs** | **$11B** | **$781M** | Series C (2025) | TTS API + 全音频栈 | S5 |
| **Deepgram** | $1.3B | ~$215M | Series B+ | ASR API (Nova-3) | S5 |
| **Hume AI** | — | ~$219M | Series B $72.8M | 情感 AI | S5 |
| **AssemblyAI** | — | $115M+ | Series C | 600M Conformer ASR | S6 |
| **Kyutai** | N/A | €300M (捐赠) | 非营利 | 全双工 Speech LLM | S4 |
| **Fixie AI** | — | $17M | Seed | 开源理解 (Ultravox) | S5 |
| **Play.ht** | — | $24M+ | Series A | 对话 TTS | S6 |

### 6.2 投资核心洞察

**洞察 1: 分发能力 >> 技术创新 (当前阶段)**

ElevenLabs ($11B, 零论文, $330M ARR) vs Kyutai (€300M 非营利, 发明全双工) — 当前估值逻辑是"离收入越近越贵",而非技术领先度。[S5, S10]

**洞察 2: Voice Agent 基础设施成为独立赛道**

LiveKit (19K stars) + Pipecat (12.7K stars) 增速超过 Speech LLM 模型本身。市场正从"谁有最好的模型"转向"谁能最快部署语音交互"。占 YC 最新一期 22%。[S6]

**洞察 3: 国内外投资逻辑差异**

| 维度 | 国际 | 国内 | Session |
|------|------|------|---------|
| 融资主体 | 独立创业公司 | 大厂内部团队 | S1-S6 |
| 估值驱动 | ARR / API 调用量 | 大模型整体估值 | S5 |
| 风险 | GPT-4o/Gemini 降维打击 | 同质化价格战 | S10 |

### 6.3 市场规模预测

| 细分市场 | 2024 | 2026 预测 | 2028 预测 | Session |
|---------|:---:|:---:|:---:|---------|
| 语音 AI API (TTS+ASR) | $3-5B | $8-12B | $15-25B | S10 |
| Conversational AI / Voice Agent | $10-15B | $20-30B | $40-60B | S10 |
| 语音克隆/配音 | $0.5-1B | $2-4B | $5-10B | S10 |

---

## 七、竞赛与评估演进

### 7.1 Benchmark 演进阶段

| 阶段 | 时间 | 代表 benchmark | 评估重点 | Session |
|------|------|---------------|---------|---------|
| 单任务独立评测 | 2021-2023 | SUPERB, ML-SUPERB | ASR/SER/SI 经典任务 | S10 |
| Speech LLM 专用评测 | 2024 | VoiceBench, Dynamic-SUPERB, AudioBench | 指令遵循、泛化、推理 | S10 |
| 边界能力 + 交互质量 | 2025-2026 | EmergentTTS-Eval, VAQI, VoxRole | edge case、端到端对话质量 | S6, S10 |
| **缺失**: 全双工评测 | — | 不存在 | 打断/轮替/延迟 tradeoff 无标准 | S10 |

### 7.2 会议趋势

| 趋势 | 说明 | Session |
|------|------|---------|
| ICLR/NeurIPS 成为 Speech LLM 主阵地 | VoxCPM Spotlight, EmotionThinker **Oral**; 语音从信号处理社区"升级"到 ML 主流 | S10 |
| ACL/NLP 社区多模态转向 | LLaMA-Omni 2 (ACL 2025), TARS (ACL 2026); NLP 从纯文本→ speech-text 联合 | S10 |
| ICASSP/Interspeech 身份转型 | 核心创新向 ICLR/NeurIPS 迁移,两个传统语音会议面临定位挑战 | S10 |
| 评估驱动系统设计 | Boson AI "evaluation-first" — 先做 EmergentTTS-Eval 再做 Higgs Audio | S6 |
| 中文评估严重滞后 | 大部分 benchmark 英文为主,中文 Speech LLM 评估依赖各公司内部 | S10 |

### 7.3 评估缺失维度

| 缺失维度 | 说明 | 重要性 | Session |
|---------|------|--------|---------|
| 全双工对话质量 | 无标准评估打断/轮替/并行 | 极高 | S10 |
| 延迟 vs 质量 tradeoff | 无统一延迟测量标准 | 高 | S10 |
| 长对话一致性 | 10+ 轮质量衰减无人研究 | 高 | S10 |
| 副语言保真度 | 笑声/停顿/语气精度 | 中高 | S10 |
| 语音安全对抗 | jailbreak + 深伪检测 | 高 | S9, S10 |

### 7.4 参赛系统趋势

1. 端到端逐步取代 pipeline: VoiceBench 2024 SOTA 多为 cascaded, 2025 起 Qwen-Audio/Kimi-Audio 领先 [S10]
2. 开源追赶闭源: LLaMA-Omni/Kimi-Audio/Fun-Audio-Chat 在 VoiceBench 上逼近 GPT-4o [S10]
3. **全双工缺乏标准评估是当前最大短板** — 每个团队自定义指标,无法横向比较 [S10]

---

## 八、关键人才流动

| 人物 | 从 → 到 | 影响 | Session |
|------|---------|------|---------|
| **Xu Tan** | Microsoft Research → Moonshot AI (多模态 VP) | 微软 TTS 核心 (VALL-E/NaturalSpeech) 流失 | S7 |
| **鄢志杰** | 阿里通义负责人 → 腾讯 (2025.02) | 通义语音领导力变动 | S2 |
| **林俊旸** | 阿里 Qwen Tech Lead → 离职 (2026.03) | Qwen 语音方向不确定性 | S1 |
| **Daniel Povey** | Kaldi 创始人 → 小米 | 学术→工业标志事件; 小米产出显著提升 | S2 |
| **Neil Zeghidour** | Google → Kyutai/Moshi | Google 语音核心人才外流 | S4 |
| **Alexander H. Liu** | Meta/FAIR → Mistral/Voxtral | Meta 语音人才外流 | S7 |
| **Zhuo Chen, Jian Wu** | Microsoft Research → ByteDance | WavLM 团队→字节 | S7 |

**趋势**: (1) 学术→工业 (2) 纯语音→多模态大模型 (3) 微软/Meta 人才向创业公司/中国大厂扩散 (4) 中国大厂间人才流动加速 (鄢志杰→腾讯, Xu Tan→月暗)

---

## 九、开源策略分类

| 策略 | 团队 | 特征 | Session |
|------|------|------|---------|
| **完全闭源** | OpenAI GPT-4o, Google Gemini, ElevenLabs, 讯飞, MiniMax, 百度 | 纯 API/产品,不公开技术细节 | S2, S3, S5, S7 |
| **选择性开源** | 阿里 (CosyVoice 开源/Fun-Audio-Chat 部分), 字节 (SALMONN 开源/Seed-TTS 闭源), 智谱, 腾讯 | 基础能力开源,核心竞争力闭源 | S1, S2 |
| **激进开源** | Meta (Spirit-LM/Seamless), Kyutai (Moshi), ICT (LLaMA-Omni), 清华 (VoxCPM 27.5K), 港中深 (Amphion 9.8K) | 全栈开源,Apache/MIT,学术为主 | S3, S4, S7, S8, S9 |
| **开源获客→API 变现** | Resemble AI (Chatterbox 25K + API), Fixie (Ultravox + API), Boson AI (Higgs Audio + API) | 开源模型引流,API 收费 | S5, S6 |
| **基础设施开源** | LiveKit (19K), Pipecat (12.7K) | 不做模型,做部署框架 | S6 |

**判断**: 开源在 Speech LLM 领域的影响力已超过闭源 — VoxCPM 27.5K, CosyVoice 21.5K, Moshi 10.4K 构成的开源生态正在加速追赶 GPT-4o/Gemini。但商业上 ElevenLabs ($11B, 完全闭源) 仍是赢家。[S3, S5, S10]

---

## 十、个人规划参考

### 10.1 高价值研究方向 (学术)

| 方向 | 理由 | 竞争度 | 入手难度 | Session |
|------|------|:---:|:---:|---------|
| **全双工对话评测** | 6+ 团队做全双工但无标准 benchmark,定义评测=主导方向 | 低 | 中 | S10 |
| **语音情感/副语言** | ICLR 2026 Oral (EmotionThinker) + Hume $72.8M 双重验证 | 中 | 中 | S5, S9 |
| **推理效率** | DRSR 5Hz 已证明 50% 加速,但方法论尚未系统化 | 中 | 中 | S1, S9 |
| **语音安全** | 生成能力泛化 (Chatterbox 25K) vs 安全研究严重滞后 | 低 | 中 | S6, S9 |
| **Tokenizer 设计** | 10+ 路线分化极快,基础设施层影响上层所有系统 | 高 | 高 | S1, S8, S9 |

### 10.2 空白机会 (差异化)

| 方向 | 理由 | Session |
|------|------|---------|
| **全双工标准 benchmark** | 当前完全空白,先做标准的人主导话语权 | S10 |
| **语音 Chain-of-Thought** | 文本 CoT 是 2025 最大热点,语音空间推理几乎无人做 | S10 |
| **多方对话 Speech LLM** | 会议/播客场景需求大,当前全部 1:1 | S10 |
| **个性化 Voice Agent** | 产品需求强烈但零学术研究 | S10 |
| **非英中 Speech LLM** | 日语/阿拉伯语等几乎空白 | S7, S9 |

### 10.3 技术赌注评估

| 赌注 | 赔率 | 风险 | 代表 | Session |
|------|------|------|------|---------|
| Tokenizer-free 路线 | 高 | 高 (生态不兼容) | VoxCPM (清华) | S3, S9 |
| 连续 VAE tokenizer | 高 | 中 (缺大规模验证) | Ming-UniAudio (蚂蚁) | S3 |
| SSM 替代 Transformer | 高 | 高 (需证明大规模可行) | Cartesia (Mamba) | S4 |
| Intelligence-Speaker 解耦 | 中 | 中 (仅腾讯一家) | Covo-Audio | S2 |
| Thinker-Talker 分离 | 中 | 低 (阿里已验证) | Qwen3.5-Omni | S1 |
| 端到端 Speech LLM 取代 Pipeline | 低 (几乎确定) | 低 | 全行业 | S10 |

### 10.4 关键时间窗口

```
2026 H2 ── Speech LLM 理解+生成同时达到 Cascaded 95%,转折点
2027 H1 ── 全双工 benchmark 出现,路线开始收敛
2027 H2 ── 端到端 Speech LLM 成为新产品默认架构
2028    ── 全双工成为 Voice Agent 默认能力
```

---

## 附录

### A1. 数据来源

| Session | 覆盖 | 行数 |
|---------|------|:---:|
| Session 1: 国内大厂 A | 阿里双团队 + 字节 + 智谱 + 阶跃 | 538 |
| Session 2: 国内大厂 B | 百度 + 讯飞 + 腾讯 + 小米 | 395 |
| Session 3: 国内新锐 | 月暗 + MiniMax + 面壁 + 蚂蚁 | 411 |
| Session 4: 国际创业 A | Kyutai + Sesame + Thinking Machines + Cartesia | 457 |
| Session 5: 国际创业 B | Fixie + Hume + ElevenLabs + Deepgram | 529 |
| Session 6: 国际创业 C | AssemblyAI + Play.ht + Resemble + LiveKit/Pipecat + Boson | 594 |
| Session 7: 国际大厂 | OpenAI + Google + Meta + Microsoft + Apple | 597 |
| Session 8: 学术 A | ICT/CAS + X-LANCE/SJTU + CUHK-SZ | 434 |
| Session 9: 学术 B | 清华 THUHCSI + CUHK Helen Meng + 日本学术界 | 462 |
| Session 10: 会议/竞赛/趋势 | 跨 Session 综合分析 | 666 |
| **合计** | **~36 团队/实验室** | **~5083** |

### A2. 训练数据规模 Top 10

| 排名 | 系统 | 数据量 | 团队 | Session |
|------|------|--------|------|---------|
| 1 | VoxCPM2 | 200 万小时 | 清华/OpenBMB | S9 |
| 2 | CosyVoice 3 | 100 万小时 | 阿里 FunAudioLLM | S1 |
| 3 | Emilia | 20 万小时 | 港中深 | S8 |
| 4 | UniAudio 2.0 | 160B tokens (~16.5万小时) | 清华/CUHK | S9 |
| 5 | Kimi-Audio | 13M 小时 | 月暗 | S3 |
| 6 | J-CHAT | 7.6 万小时 | 东京大学 | S9 |
| 7 | Fun-Audio-Chat | 百万小时级 | 阿里 FunAudioLLM | S1 |
| 8 | ERNIE 5.0 | 未公开 (预期百万级) | 百度 | S2 |
| 9 | StepAudio 2.5 | 未公开 | 阶跃 | S1 |
| 10 | SALMONN-omni | 未公开 | 字节 | S1 |

### A3. GitHub Stars Top 15 (语音 AI 项目, 2026.06)

| 排名 | 项目 | Stars | 团队 | 类型 |
|------|------|:---:|------|------|
| 1 | Whisper | 102K | OpenAI | ASR |
| 2 | VoxCPM | 27.5K | 清华/OpenBMB | TTS (tokenizer-free) |
| 3 | Chatterbox | 25K | Resemble AI | 对话 TTS |
| 4 | AudioCraft | 23K | Meta | 音频生成套件 |
| 5 | CosyVoice | 21.5K | 阿里 | 多语言 TTS |
| 6 | LiveKit | 19K | LiveKit | Voice Agent 基础设施 |
| 7 | CSM | 14.7K | Sesame | Speech LLM |
| 8 | Pipecat | 12.7K | Daily.co | Voice Agent 框架 |
| 9 | Seamless | 12K | Meta | 语音翻译 |
| 10 | Moshi | ~10.4K | Kyutai | 全双工 Speech LLM |
| 11 | Amphion | 9.8K | 港中深 | 工具包 |
| 12 | SenseVoice | 8.5K | 阿里 | ASR |
| 13 | Higgs Audio | 8.1K | Boson AI | Speech LLM |
| 14 | Kimi-Audio | 4.6K | 月暗 | Speech LLM |
| 15 | Ultravox | 4.4K | Fixie | Speech LLM |

### A4. 五大判断 (Summary)

1. **端到端 Speech LLM 将在 2027 年成为新产品默认架构** — 延迟优势 + 副语言保留 + 全双工能力使其不可逆转地取代 Cascaded pipeline。转折点在 2026 年底。

2. **全双工对话是下一个关键战场,但远未收敛** — 6+ 条完全不同的技术路线并行,缺乏统一评估标准。预期 2027 年开始收敛。

3. **语音情感/副语言是端到端系统的核心差异化** — EmotionThinker ICLR Oral + Hume $72.8M = ML 社区和资本市场双重认可。这是端到端相对 Cascaded 的最大结构性优势。

4. **开源生态已成为不可忽视的力量** — VoxCPM 27.5K, CosyVoice 21.5K, Moshi 10.4K, Kimi-Audio 4.6K 等开源项目质量逼近闭源,且迭代更快。

5. **商业价值分配极度不均: 分发 >> 技术** — ElevenLabs ($11B, 零论文) vs Kyutai (€300M, 发明全双工) 说明当前阶段分发能力决定估值。但长期看端到端技术壁垒将重新分配价值。

---

> **报告完成时间**: 2026-06-08
> **依赖数据**: Session 1-10 原始数据 (~5083 行) + 公开信息
> **免责**: 信息截至 2026-06-08,闭源团队信息基于公开资料推断
