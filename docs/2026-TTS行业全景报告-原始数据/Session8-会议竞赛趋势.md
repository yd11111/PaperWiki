# Session 8: 会议/竞赛/投资/趋势汇总

> 调研日期: 2026-06-05
> 数据来源: Playwright 搜索 + Sessions 1/3/4/5/6/7 原始数据交叉分析

---

## Part 1: 会议热点统计

### 1.1 ICASSP 2025 (Hyderabad, India, 2025.04)

| 指标 | 数据 |
|------|------|
| **投稿量** | 6,947 篇 |
| **录用量** | 3,145 篇 |
| **录用率** | 45.27% |
| **同比变化** | 投稿↑20% (2024: 5,796), 录用↑12% (2024: 2,812) |

**Speech/TTS 相关热点 Topic:**

| 热点方向 | 代表工作 | 趋势 |
|----------|----------|------|
| LLM-based TTS / Speech LM | TTS-Transducer, SALMA Workshop | 新兴热点, SALMA Workshop 专门设立 |
| 语音编码/Codec | 多种 codec 设计论文 | 从 EnCodec 延伸, 方向分化 |
| 零样本语音合成 | 多种 zero-shot TTS | 继续火热 |
| 语音安全/深伪检测 | CodecFake 系列 | 新兴关切 |
| 全双工对话 | 早期探索论文 | 开始出现 |

**大厂参与:**
- Google: 20+ 篇 accepted papers + 多个 workshop/lecture
- Apple: 多篇 accepted papers (语音识别、音频处理)
- NTT: 在 Interspeech 2025 有 18 篇 (ICASSP 未单独公布)

### 1.2 ICASSP 2026 (预计 2026.06, 已公布 accepted papers)

| 指标 | 说明 |
|------|------|
| **状态** | accepted papers 列表已公布 (cmsworkshops.com) |
| **关键变化** | Speech/Audio 方向进一步向 LLM 融合; Codec 相关论文大幅增长 |

**来自本报告各 Session 确认的 ICASSP 2026 收录论文:**

| 论文 | 来源团队 | 方向 |
|------|----------|------|
| HumDial Challenge | NTU + NPU 联合 | 全双工对话竞赛 |
| SP-MCQA (TTS evaluation) | CUHK-SZ 武执政 | TTS 评估 |
| AnyAccomp (singing accompaniment) | CUHK-SZ 武执政 | 歌声伴奏 |

### 1.3 Interspeech 2025 (Leeuwarden, Netherlands, 2025.08)

| 项目 | 详情 |
|------|------|
| **地点** | 荷兰吕伐登 (Leeuwarden) |
| **时间** | 2025年8月 |
| **附属活动** | 第13届 ISCA Speech Synthesis Workshop (SSW13, 2025.08.24-26) |
| **主题** | "Speech Technology for Less-Resourced Languages" |

**Special Sessions (确认):**

| Special Session | 方向 | 说明 |
|----------------|------|------|
| Responsible Speech and Audio Generative AI | TTS/VC/歌唱/音乐安全 | 关注生成式语音 AI 的负责任使用 |
| Challenges in Speech Data Collection, Curation, and Annotation | 数据 | 反映数据质量成为瓶颈 |
| Recent Advances and Future Directions in Voice Conversion | 语音转换 | Survey Talk by Tomoki Toda |
| Speech-related Biosignals | 生物信号 | 口腔运动/神经信号与语音 |

**Challenges:**

| Challenge | 任务 |
|-----------|------|
| Audio-Visual Speaker Diarization | 视听说话人分割 |
| Audio-Visual Speech Recognition | 视听语音识别 |
| Audio-Visual Diarization and Recognition (联合) | 视听联合任务 |

**TTS 相关论文趋势:**
- "Fairness in Dysarthric Speech Synthesis" — 使用 F5-TTS 的障碍语音合成公平性
- Voice Conversion 专题 Survey Talk (Tomoki Toda)
- 低资源语言 TTS 成为重点议题 (与 Blizzard Challenge 2025 呼应)

**来自本报告各 Session 确认的 Interspeech 2025/2026 收录论文:**

| 论文 | 来源 | 会议 |
|------|------|------|
| Speech Speculative Decoding | 清华 THUHCSI | Interspeech 2025 |
| LIST (Language-Independent Speech Token) | USTC 凌震华 | Interspeech 2025 |
| SoulX-Duplug | Soul App + NPU | Interspeech 2026 |

### 1.4 NeurIPS 2025 (Vancouver, 2025.12)

**确认的 Speech/Audio 相关论文:**

| 论文 | 来源 | 方向 |
|------|------|------|
| **SALMONN-omni** | ByteDance | Codec-free 全双工 Speech LLM |
| **MMAR** | ByteDance | 多模态深度推理 Benchmark |
| **E2E-VGuard** | 清华 THUHCSI | LLM-TTS 防御 (安全) |
| **Metis** | CUHK-SZ 武执政 | Foundation Speech Generation |
| **TadiCodec** | CUHK-SZ 武执政 | Text-aware Diffusion Tokenizer |
| **CoVoMix2** | (多校) | 多说话人对话合成 |

**热点关注:**
- Speech LM / 全双工对话系统首次大规模出现在 NeurIPS (此前以 ICASSP/Interspeech 为主场)
- 语音安全 (deepfake defense) 开始被 ML 顶会接受
- Codec/Tokenizer 设计进入 ML 顶会 (TadiCodec)

### 1.5 ICLR 2025 / ICLR 2026

**ICLR 2025 (Singapore, 2025.04):**

| 论文 | 来源 | 引用 | 方向 |
|------|------|------|------|
| **MaskGCT** | CUHK-SZ 武执政 | 210 | 非自回归 TTS |
| **WavTokenizer** | ZJU Zhou Zhao | 189 | 音频 Codec |
| **FlowDec** | (学术) | - | Flow-based 全带宽 Codec |

**ICLR 2026 (确认收录):**

| 论文 | 来源 | 方向 |
|------|------|------|
| **ELLSA** | ByteDance | 视觉+语音+动作多模态全双工 |
| **ParaS2S** | ByteDance | 副语言感知 S2S + RL |
| **STITCH** | NTU + Microsoft | Chunked 推理 SLM |
| **SHANKS** | NTU | 同时听+想 SLM |

**趋势:** ICLR 2025 speech 相关论文 100+ 篇 (GitHub 统计), 语音/音频方向在 ML 顶会的占比持续上升。

### 1.6 ACL / EMNLP 2025-2026 (Speech-Language 交叉)

**ACL 2025:**

| 论文 | 来源 | 方向 |
|------|------|------|
| **Spark-TTS** | NPU + Soul App | BiCodec + Qwen2.5 TTS (144 cit.) |
| **MELLE** | CUHK + Microsoft | 无 VQ 的 AR 语音合成 |
| **OmniCharacter** | CUHK | 角色扮演语音交互 |
| **WavRAG** | ZJU | 音频检索对话 |

**ACL 2026 (已确认):**

| 论文 | 来源 | 方向 |
|------|------|------|
| **SAC** | NPU + Soul App | 语义-声学双流 Codec (Main Conference) |
| **LLM-Codec** | NTU 李宏毅 | 音频编码 (Findings) |

**AAAI 2025/2026:**

| 论文 | 来源 | 会议 | 方向 |
|------|------|------|------|
| StableVC | NPU | AAAI 2025 | 语音转换 |
| Codec Does Matter | (Xu Tan合作) | AAAI 2025 | Codec 分析 |
| **VARSTok** | 阿里通义 | AAAI 2026 Oral | 变长语义 token |
| **KALL-E** | NPU | AAAI 2026 | Next-distribution TTS |
| WenetSpeech-Yue | NPU | AAAI 2026 | 粤语数据集 |
| **DualSpeechLM** | 清华 THUHCSI | AAAI 2026 | 统一 SLM |

**ICML 2025:**

| 论文 | 来源 | 方向 |
|------|------|------|
| **DiTAR** | ByteDance | AR + 连续扩散 TTS |
| **Llasa** (共同作者Xu Tan) | 多校 | LLM-based TTS Scaling |

**CVPR 2025:**

| 论文 | 来源 | 方向 |
|------|------|------|
| **Teller** | Soul App + NPU | 实时流式数字人 |

### 1.7 会议热点趋势总结

| 方向 | 2024 | 2025 | 2026 | 趋势 |
|------|------|------|------|------|
| LLM-based TTS | 萌芽 | **爆发** | 主流 | 急剧上升 |
| 语音 Codec/Tokenizer | 热门 | **顶会密集收录** | 持续 | 持续热门 |
| 全双工/Speech LM | 少量 | 增长 | **大量顶会收录** | 急剧上升 |
| 零样本 TTS | 热门 | 趋稳 | 略降 | 趋于饱和 |
| Post-training (RL/DPO) | 萌芽 | 增长 | **多团队采用** | 快速上升 |
| 语音安全/伪造检测 | 少量 | 明显增长 | 专门 Session | 上升 |
| 传统韵律/声码器 | 下降 | 继续下降 | 边缘化 | 下降 |
| 歌声/音乐生成 | 少量 | 增长 | 多方投入 | 上升 |

---

## Part 2: 竞赛与 Challenge

### 2.1 Blizzard Challenge 2024 / 2025

**Blizzard Challenge 2024:**
- 传统 Blizzard Challenge 格式, 英语为主
- 评估方法: 标准化 MOS 评测协议 (自 2005 年以来基本不变)

**Blizzard Challenge 2025:**

| 项目 | 详情 |
|------|------|
| **任务** | 为 Bildts (荷兰语变体) 合成语音 |
| **场景** | 低资源语言合成 (数据稀缺) |
| **依托** | 第13届 ISCA Speech Synthesis Workshop (SSW13), 赫尔辛基大学承办 |
| **评估** | 多维度评估 (MOS + 更细粒度的指标) |
| **关键论文** | "A Multi-dimensional Evaluation of the 2025 Blizzard Challenge" (Shirali-Shahreza, 被引2) |

**参赛系统技术趋势:**

| 趋势 | 说明 |
|------|------|
| **神经 TTS 主导** | 参赛系统几乎全部采用神经网络方法, 传统方法消失 |
| **低资源挑战** | 任务设计从高资源英语转向极低资源语言, 测试真实场景泛化能力 |
| **F5-TTS 等开源方案被广泛采用** | 多个参赛系统基于开源 TTS 框架 (如 F5-TTS) 进行适配 |
| **kNN-TTS 等新范式** | Idiap 提交的 kNN-TTS 尝试检索增强方法 |
| **无需海量数据的合成** | 证明"竞争力的低资源语音合成不需要海量数据集" |

**Blizzard Challenge 演进 (2020-2025):**

| 年份 | 任务语言 | 关键变化 |
|------|----------|----------|
| 2020-2023 | 英语/中文/印地语 | 标准高资源 TTS 评测 |
| 2024 | 英语 | 传统格式 |
| 2025 | **Bildts (荷兰语变体)** | 首次聚焦极低资源语言; 与 SSW13 co-located |

### 2.2 VoiceMOS Challenge 2024 / AudioMOS Challenge 2025

**VoiceMOS Challenge 2024 (@ SLT 2024):**

| 项目 | 详情 |
|------|------|
| **Track 1** | MOS prediction for "zoomed-in" systems (细粒度系统差异预测) |
| **Track 2** | MOS prediction for singing voice (歌声质量预测) |
| **Track 3** | Semi-supervised MOS prediction (半监督设置) |
| **论文** | "The VoiceMOS Challenge 2024: Beyond Speech Quality Prediction" (被引 52) |
| **关键发现** | OOD 泛化仍然是主要问题; 监督和零样本之间存在差距; 大多数团队无法在所有轨道上保持一致性能 |

**AudioMOS Challenge 2025 (品牌升级):**

| 项目 | 详情 |
|------|------|
| **名称变化** | 从 "VoiceMOS" 更名为 "AudioMOS", 反映范围扩展到通用音频 |
| **结果公布** | 2025.06.16 已向参赛者发布结果 |
| **演进方向** | 从纯语音 MOS 预测 → 歌声 + 通用音频质量评估 |

**VoiceMOS 系列演进:**

| 年份 | 名称 | 重点变化 |
|------|------|----------|
| 2022 | VoiceMOS Challenge | 首届, 语音自然度 MOS 预测 |
| 2023 | VoiceMOS Challenge 2023 | OOD 泛化关注 |
| 2024 | VoiceMOS Challenge 2024 | +歌声 +半监督 +zoomed-in |
| 2025 | **AudioMOS Challenge 2025** | 品牌升级, 范围扩展到通用音频 |

### 2.3 Singing Voice Conversion Challenge 2025

| 项目 | 详情 |
|------|------|
| **组织方** | Voice Conversion Challenge (VCC) 系列 |
| **新趋势** | "shifted the focus to not only singer identity conversion, but also..." 扩展到更广泛的歌声控制 |
| **分析论文** | "An Extensive Analysis of the Singing Voice Conversion Challenge 2025" (arXiv, 2025.06) |
| **状态** | 首次专门的歌唱合成转换竞赛, 反映歌声合成赛道升温 |

### 2.4 HumDial Challenge (ICASSP 2026)

| 项目 | 详情 |
|------|------|
| **组织方** | NTU (李宏毅) + NPU (谢磊) 联合组织 |
| **方向** | 全双工人机对话 (Human Dialogue) |
| **依托** | ICASSP 2026 |
| **意义** | 首个专门的全双工对话 Challenge, 由两大语音实验室联合发起 |

### 2.5 其他相关 Challenge

| Challenge | 年份 | 方向 | 说明 |
|-----------|------|------|------|
| Codec Challenge (IEEE MMSP) | 2024 | 音频编解码 | 低比特率+高质量 |
| SpeechJudge / TTS 评估 | 2025-2026 | TTS 自动评估 | 多个团队自建 reward model |
| SEED-TTS-Eval | 2024+ | TTS 评估 | 字节发起, 已成事实标准 (数十篇论文采用) |
| Full-Duplex-Bench v1-v3 | 2025-2026 | 全双工评估 | NTU 李宏毅组持续迭代 |

### 2.6 竞赛趋势总结

| 趋势 | 说明 |
|------|------|
| **从高资源到低资源** | Blizzard 2025 转向 Bildts; Interspeech 2025 主题为低资源语言 |
| **从语音到通用音频** | VoiceMOS → AudioMOS; 歌声/音乐评估纳入 |
| **全双工 Challenge 诞生** | HumDial Challenge (ICASSP 2026), 由 NTU+NPU 联合发起 |
| **事实标准形成** | SEED-TTS-Eval (字节) 成为 TTS 评估事实标准; Full-Duplex-Bench 系列 |
| **歌声合成赛道独立** | SVC Challenge 2025 专门化; SVS 论文数量增长 |
| **RL/Reward Model 驱动评估** | SpeechJudge (字节), UniSRM (清华), TTS-PRISM (清华+小米) |

---

## Part 3: 投资与市场

### 3.1 语音 AI 公司融资概览

| 公司 | 总融资 | 最新估值 | 最新轮次 | 时间 | 领投 |
|------|--------|----------|----------|------|------|
| **ElevenLabs** | **$781M** | **$11B** | Series D ($500M) | 2026.02 | Sequoia Capital |
| **Cartesia** | ~$91M | 未公开 | Series A ($64M) | 2025.03 | Kleiner Perkins |
| **Sesame AI** | ~$100M+ | 未公开 | (未公开轮次) | - | a16z, Sequoia, Spark |
| **Hume AI** | (已被收购) | - | - | 2026.01 | 被 Google 收购 |
| **Deepgram** | $86M+ | ~$1B | Series B ($47M) | 2024 | Madrona |
| **Rime** | 未公开 | 未公开 | - | - | - |

**ElevenLabs 融资时间线:**

| 轮次 | 时间 | 金额 | 估值 |
|------|------|------|------|
| Seed | 2023初 | 未公开 | - |
| Series A | 2023.06 | $19M | ~$100M |
| Series B | 2024.01 | $80M | $1.1B |
| Series C | 2025.01 | $180M | $3B+ |
| Series D | 2026.02 | $500M | **$11B** |

**ElevenLabs 收入:**
- 2025年底 ARR $330M
- 2026.05 突破 **$500M ARR**
- 从 $0 到 $500M ARR 用了不到 3 年

### 3.2 大厂收购动态

| 事件 | 时间 | 详情 |
|------|------|------|
| **Meta 收购 PlayHT** | 2025.07 | PlayHT 平台 2025.12 关闭; Meta 获取商业 TTS 技术, 比自研更快 |
| **Google 收购 Hume AI 团队** | 2026.01 | TechCrunch 报道; Hume AI 团队被 Google 吸收, 品牌走向不确定 |
| **Mistral AI 持续扩张** | 2025-2026 | 多轮融资, 估值超 60 亿美元 (Microsoft/NVIDIA 投资); 语音团队从 106→189 人 |

### 3.3 市场规模预测

| 市场 | 2025 规模 | 预测规模 | 年份 | CAGR | 来源 |
|------|----------|----------|------|------|------|
| Voice AI Platform | $5.2B | $48.6B | 2034 | 28.6% | Market Intelo (2026.06) |
| AI Voice Generator | $4.16B | $20.71B | 2031 | 30.7% | Entrepreneur Loop (2026.02) |
| AI Voice Lab | - | $50.16B | 2035 | - | Precedence Research (2026.02) |
| Conversational AI | $11.6B (2024) | - | - | - | Contrary Research |

**关键市场信号:**
- Voice AI 融资从 2023 到 2024 **暴增 8 倍**, 达 $21 亿 (Landbase, 2026.02)
- Voice agent 方向在 2024 H2 爆发: 占 Y Combinator 最新一期 **22%** (a16z, 2025.01)
- Voice AI 是 2024-2026 增长最快的 AI 子赛道之一

### 3.4 中国市场补充

| 公司 | 融资/估值 | 语音 AI 投入 |
|------|----------|-------------|
| 阿里 (通义+Qwen) | 大厂内部投入 | 双团队 (FunAudioLLM 35+ 篇 + Qwen 5 篇), 5M+ 小时数据 |
| 字节跳动 | 大厂内部投入 | 40+ 篇论文, 已部署豆包/剪映 |
| 阶跃星辰 | ~$15B 估值 (整体) | 100+ 人语音团队, 8 篇论文, Apache-2.0 全开源 |
| 智谱 AI | ~$3B 估值 (整体) | 语音方向 15-20 人, 2026 年减速 |
| Fish Audio | 未公开 | 31K stars 开源, WER SOTA, 1000 万小时数据 |
| Soul App | 未公开 (社交主业) | 20+ 篇论文, 7900+ stars, CVPR+ACL 顶会 |

---

## Part 4: 跨 Session 趋势综合

> 基于 Session 1 (阿里+字节+阶跃+智谱) / Session 3 (京东+天工+Soul+米哈游) / Session 4 (ElevenLabs+Cartesia+Sesame+Fish Audio) / Session 5 (Mistral+Meta+Google+OpenAI) / Session 6 (NTU+USTC+NPU+清华) / Session 7 (ZJU+CUHK+CMU+其他) 的交叉分析。

### 4.1 技术路线收敛点 (共识)

以下技术已被大多数玩家采用, 形成行业共识:

| 收敛点 | 采用者 | 说明 |
|--------|--------|------|
| **LLM backbone 做 TTS** | 阿里/字节/阶跃/智谱/Qwen/Fish/Voxtral/NPU/CUHK-SZ/清华 | 几乎所有团队都在用 LLM 架构做 TTS, AR 解码为主 |
| **语义+声学两阶段** | CosyVoice/Voxtral/GLM-TTS/Spark-TTS/MaskGCT/DiTAR | AR 生成语义 token + Flow Matching/DiT/掩码生成 声学 rendering |
| **Flow Matching 做声学建模** | CosyVoice(FM)/GLM-TTS(FM)/Voxtral(FM)/MaskGCT(Masked) | 替代传统扩散模型, 更高效 |
| **Post-training (RL/DPO/GRPO)** | 字节(RL→DPO)/阿里(DiffRO)/阶跃(三模式RLHF)/智谱(GRPO)/Fish(GRPO) | TTS 后训练从探索变为标配 |
| **SEED-TTS-Eval 作为评估标准** | 字节/阿里/阶跃/Fish/Voxtral/多数学术组 | 字节主导的评估基准已成事实标准 |
| **零样本语音克隆** | 所有 TTS 系统 | 3-10 秒参考音频克隆已成基线能力 |
| **多语言支持 (30+)** | ElevenLabs(70+)/Fish(80+)/Voxtral(42)/Gemini(72+) | 多语言从差异化变为基线 |

### 4.2 技术路线分歧点 (赌注)

以下技术各家路线不同, 是关键赌注:

| 分歧点 | 阵营A | 阵营B | 阵营C |
|--------|-------|-------|-------|
| **底层架构** | Transformer (绝大多数) | **SSM/Mamba** (Cartesia) | — |
| **Tokenizer 路线** | 监督语义 token (阿里 CosyVoice) | 自监督 (HuBERT/Mimi) | VQ-FSQ 混合 (Voxtral); Dual-codebook (阶跃); 175bps 超低码率 (智谱) |
| **生成范式** | AR 自回归 (大多数) | 非 AR 掩码生成 (MaskGCT/Metis) | AR+DiT 混合 (DiTAR/DiSTAR); 直接波形 DiT (WavTTS) |
| **Codec 策略** | 单层 Codec (MagiCodec) | 多层 RVQ (EnCodec 系) | 语义-声学双流 (SAC); 无 Codec (MELLE/WavTTS) |
| **TTS 定位** | 独立 TTS 系统 (CosyVoice/Spark/Fish) | LLM-native (Qwen3-TTS/Gemini Flash TTS) | 统一理解+生成 (UniVocal/DualSpeechLM) |
| **推理加速** | 蒸馏/剪枝 (传统) | **SSM 线性复杂度** (Cartesia, 40ms) | SplitMeanFlow 20x 加速 (字节) |
| **音频推理** | 无 (大多数) | Audio-CoT (字节) | **MGRD 模态锚定推理** (阶跃 R1 系列) |
| **开源策略** | 全闭源 (ElevenLabs/Google/OpenAI) | 全开源 (Fish/阶跃/Voxtral) | 选择性开源 (CosyVoice/CSM) |
| **商业模式** | B2B API SaaS (ElevenLabs/Cartesia) | B2C 消费级 (Sesame/Soul) | 大厂内嵌 (Google/OpenAI/字节) |

### 4.3 上升趋势 (2024 → 2025 → 2026 越来越多人做的方向)

| 方向 | 2024 参与者 | 2026 参与者 | 关键信号 |
|------|-----------|-----------|----------|
| **1. Post-training (RL/DPO/GRPO)** | 字节 (首个 RL) | 字节+阿里+阶跃+智谱+Fish | 从 1 家到 5+ 家, DiffRO/SpeechJudge 等独立创新 |
| **2. 全双工/实时对话** | 字节 LSLM, OpenAI GPT-4o | 字节 ELLSA, 阿里 Fun-Audio-Chat, Qwen-Omni, 阶跃 Dual-Brain, Soul SoulX-Duplug, NTU FDB v3, HumDial Challenge | 从 2 家到 7+ 家, 首个竞赛 (HumDial) |
| **3. 歌声/音乐合成** | 少量独立探索 | Soul 42K hrs SVS, NPU DiffRhythm, ElevenLabs Music v2, Kunlun Mureka, 字节 Vevo2, Google Lyria 3 | 从边缘到多方投入, 独立产品线出现 |
| **4. 音频推理 (Audio Reasoning)** | 无 | 阶跃 R1/R1.5/R1.1, 字节 Audio-CoT, NTU STITCH/SHANKS | 全新赛道, 2025 年从 0 到多方竞争 |
| **5. Voice Agent / 实时 API** | OpenAI Realtime API | ElevenLabs Agents (2M+ agent), Cartesia Line, Deepgram Flux, 阶跃 Realtime API | Voice Agent 占 YC 22% |
| **6. 语音安全/Deepfake 防御** | 少量 | 清华 E2E-VGuard, NTU CodecFake+, Interspeech Special Session, SynthID | 从学术到产品 (Google SynthID) |
| **7. TTS 可控性 (Instruction/Audio Tags)** | 少量 | Google Audio Tags, 阶跃 EditX 30+ 风格, ElevenLabs v3 Audio Tags, Kunlun Description-based | 自然语言控制 TTS 替代 SSML |
| **8. Codec 设计创新** | EnCodec 主导 | MagiCodec/SAC/OmniCodec/WavTokenizer/BiCodec/DualCodec/TadiCodec/LLM-Codec/VARSTok | 从 1-2 种到 10+ 种, 分化极快 |
| **9. 低资源/方言 TTS** | 少量 | NPU WenetSpeech 方言系列, Blizzard 2025 Bildts, 清华 DiaMoE-TTS, Interspeech 低资源主题 | 竞赛+数据集+模型三方驱动 |

### 4.4 下降趋势 (越来越少人做的方向)

| 方向 | 说明 | 证据 |
|------|------|------|
| **1. 传统 Mel-spectrogram 声码器** | HiFi-GAN/WaveNet 等传统声码器不再是研究热点 | USTC 仍坚持但影响力下降; 新系统直接端到端 |
| **2. 级联式 ASR+LLM+TTS** | 被端到端多模态替代 | OpenAI GPT-4o 定义范式; 所有大厂跟进端到端 |
| **3. 独立 G2P 模块** | 被 LLM 内嵌语言理解替代 | Fish Audio 直接用 LLM 替代 G2P; 多数新系统不依赖 G2P |
| **4. 小模型 TTS (< 100M)** | 被 1B+ 大模型替代 | CosyVoice 3 1.5B, Llasa 1B/3B/8B, Fish 4B, Seed 未公开(极大) |
| **5. 单语种 TTS 研究** | 多语言成为基线 | 所有新系统默认支持 30+ 语言 |
| **6. 纯文本驱动 TTS (无语音 prompt)** | 零样本克隆成为标配 | 2024 年后几乎没有纯文本 TTS 新论文 |
| **7. Meta FAIR 语音研究** | 6 个 speech repo 中 4 个 archived | 2024-2026 无旗舰新工作; 收购 PlayHT 代替自研 |
| **8. 微软 TTS 研究** | 核心团队解体 | Xu Tan→Moonshot; NaturalSpeech 系列停更; VALL-E 无后续 |
| **9. CMU TTS 研究** | Alan Black 基本退出前沿 | 北美顶校 TTS 合成方向边缘化 |

### 4.5 空白地带 (应该有人做但目前做得少的)

| 空白 | 现状 | 潜在价值 |
|------|------|----------|
| **1. TTS 可解释性** | 几乎无人做 | LLM-based TTS 是黑盒, 缺乏对"为什么这样读"的解释 |
| **2. 长文本一致性 (> 10 分钟)** | 仅 Soul SoulX-Podcast (5分钟+) | 有声书/播客需要小时级一致性, 目前无系统验证 |
| **3. 多说话人对话 TTS** | 仅 JoyVoice (8人), DialoSpeech (2人) | 多说话人场景 (会议/播客/影视) 需求大, 解决方案极少 |
| **4. 端侧部署 (1B 以下)** | 仅 Cartesia Edge, 阶跃有小模型计划 | 手机/IoT/车载需求大, 但学术关注极低 |
| **5. 跨语种声音克隆** | 零碎探索 | 用中文参考音频生成英语, 保持说话人特征, 系统性研究很少 |
| **6. 对话韵律建模** | 仅 Sesame CSM (Voice Presence) | 对话中的韵律动态 (打断/犹豫/共情) 缺乏系统方法论 |
| **7. TTS 水印标准化** | Google SynthID, 清华 VoiceMark/Latent-Mark, 浙大 | 缺乏行业统一标准; AI Act 等法规即将要求 |
| **8. 障碍语音 TTS** | Interspeech 2025 有论文但极少 | 构音障碍等特殊人群的 TTS 需求未被满足 |
| **9. 实时语音编辑** | 仅阶跃 EditX (3B) | 实时修改语音的情感/风格/语速, 编辑类 API 极少 |
| **10. 开源 TTS 评估基准** | SEED-TTS-Eval 由字节主导 | 需要更独立的第三方评估标准 |

### 4.6 时间线关键节点 (2024-2026)

```
2024.05  OpenAI GPT-4o — 定义端到端语音交互范式, 引发行业追赶
    |
2024.06  Seed-TTS — 字节旗舰, 定义 eval 标准
    |
2024.07  CosyVoice + FunAudioLLM — 阿里开源全栈语音工具链
    |
2024.09  Cartesia Sonic + Edge — SSM TTS 首发产品
    |
2024.10  OpenAI Realtime API — 开发者实时语音 API, 催生 Voice Agent 赛道
    |
2024.11  GLM-4-Voice — 智谱端到端语音聊天 (引用 243)
    |
2024.12  CosyVoice 2 — 流式 TTS 里程碑
    |
2025.01  ElevenLabs Series C ($180M, $3B) — 语音 AI 首个独角兽大额融资
    |
2025.02  Step-Audio v1 (130B) — 最大语音模型; DiTAR (ICML) — AR+DiT 新范式
         CSM-1B (Sesame) — 对话韵律模型开源 (14.7K stars)
         Llasa (1B/3B/8B) — TTS Scaling Law 验证
    |
2025.03  Spark-TTS (ACL 2025) — 开源 LLM-TTS 标杆 (11K stars)
         Cartesia Series A ($64M, KP 领投) — SSM 赌注获顶级 VC 认可
    |
2025.05  CosyVoice 3 (1.5B, DiffRO) — 阿里旗舰, post-training 创新
         Qwen3-TTS (5M+ 小时, 97ms TTFB) — Qwen 独立 TTS
    |
2025.06  MagiCodec (字节) — 新一代 Codec 基建
    |
2025.07  Meta 收购 PlayHT — 大厂通过收购获取 TTS; SplitMeanFlow 20x 加速部署豆包
    |
2025.08  Interspeech 2025 — 低资源语言 + Responsible AI 成为主题
    |
2025.09  ElevenAgents — 2M+ agent, Voice Agent 平台化
    |
2025.10  MaskGCT (ICLR 2025, 210 cit.) — 非 AR TTS 新范式
         SAC (ACL 2026) — 语义-声学双流 Codec
    |
2025.11  Step-Audio-R1 — 首个音频推理模型, 新赛道
         SpeechJudge (字节) — TTS Reward Model 基建
    |
2025.12  Voxtral TTS — 开源 TTS 质量挑战 ElevenLabs (胜率 68.4%)
         GLM-TTS — 智谱最后一篇语音论文 (此后无新作)
    |
2026.01  Google 收购 Hume AI — 情感语音被巨头吸收
         HumDial Challenge (ICASSP 2026) — 首个全双工竞赛
    |
2026.02  ElevenLabs Series D ($500M, $11B) — 语音 AI 赛道最大融资
         SoulX-Singer (42K hrs) — 工业级歌声合成开源
    |
2026.03  Voxtral TTS 正式发布 — 法国开源挑战者
    |
2026.04  Gemini 3.1 Flash TTS + Live — Google 原生 TTS/实时对话
         StepAudio 2.5 — 三模式统一 RLHF
    |
2026.05  Qwen3-TTS 正式发布; ElevenLabs $500M ARR
    |
2026.06  WavTTS (字节) — 直接波形 DiT, 可能的下一代架构
         UniVocal (阿里) — 统一理解+生成
```

### 4.7 综合判断矩阵

| 维度 | 领先者 | 追赶者 | 观望/退出 |
|------|--------|--------|-----------|
| **论文产出密度** | 字节 (40+), NTU (60+), NPU (40+) | 阿里 (35+), Soul (20+), 清华 (35+) | 智谱 (3), 米哈游 (0) |
| **商业化收入** | ElevenLabs ($500M ARR) | Google/OpenAI (内嵌), 字节 (豆包), Fish (API) | Sesame (无收入), 学术 |
| **开源影响力** | Fish (31K), CSM (14.7K), Spark-TTS (11K), CosyVoice (9K) | MegaTTS3 (6.1K), AudioCraft (23K但archived) | ElevenLabs/Google/OpenAI (闭源) |
| **评估标准主导** | 字节 (SEED-TTS-Eval), NTU (SUPERB/FDB) | CUHK-SZ (Amphion), NPU (MINT-Bench) | 其他 |
| **Post-training** | 字节 (RL→DPO→SpeechJudge), 阿里 (DiffRO) | 阶跃 (三模式RLHF), 智谱 (GRPO), Fish (GRPO) | 大多数学术组 |
| **全双工对话** | 字节 (LSLM→ELLSA), NTU (FDB v1-v3) | 阿里 (Fun-Audio-Chat), 阶跃 (Dual-Brain), Soul (SoulX-Duplug) | 大多数 |
| **音频推理** | 阶跃 (R1/R1.5/R1.1, 独占赛道) | 字节 (Audio-CoT), NTU (STITCH/SHANKS) | 大多数 |
| **多模态融合** | Google (Gemini 原生), OpenAI (GPT-4o) | 字节 (Seedance), 阿里 (Qwen-Omni), 阶跃 (StepAudio) | 大多数学术 |
| **低延迟** | Cartesia (40ms, 结构性优势) | Qwen3-TTS (97ms), ElevenLabs Flash (75ms) | 大多数 >200ms |

### 4.8 三大范式转移

**范式转移 1: 从独立 TTS 到 LLM-native 语音能力**

```
旧范式: 独立 TTS 系统 (Tacotron → VITS → CosyVoice)
         ↓ GPT-4o 催化
新范式: 语音是 LLM 的原生输出模态 (Qwen-Omni, Gemini Flash TTS, UniVocal)
状态: 过渡期 — 独立系统和 LLM-native 共存
```

**范式转移 2: 从手工特征工程到 Post-training 对齐**

```
旧范式: 精心设计 tokenizer + 架构 + 数据配比
         ↓ RL/DPO/GRPO 引入
新范式: 用 reward model 自动优化 (SpeechJudge, DiffRO, GRPO 四维)
状态: 早期 — Post-training 已被验证有效, 但 reward model 设计仍分散
```

**范式转移 3: 从 Transformer 独占到架构多元化**

```
旧范式: Transformer-only (GPT-style / Llama-style)
         ↓ Mamba/SSM 挑战
新范式: SSM (Cartesia) + DiT (字节) + Masked Generative (MaskGCT) + Dual-AR (Fish) 多路线并行
状态: 活跃竞争期 — 尚未形成新共识
```

### 4.9 区域格局

| 区域 | 学术 | 工业 | 趋势 |
|------|------|------|------|
| **中国大陆** | NPU/ZJU/USTC/清华 (系统+Codec+歌声) | 字节/阿里/阶跃/智谱/Fish/Soul | 论文产出和开源影响力全球最大 |
| **美国** | CMU (退出合成方向) | OpenAI/Google/Meta/ElevenLabs/Cartesia/Sesame | 商业化领先, 学术衰退 |
| **欧洲** | Idiap (kNN-TTS) | Mistral/Voxtral | 开源新势力 (Voxtral TTS 胜率 68.4% vs ElevenLabs) |
| **东亚 (中国外)** | NTU (评测), KAIST (视觉-语音) | - | 评测方向有全球领导力 |
| **港澳台** | CUHK-SZ (MaskGCT/Amphion), CUHK (DualSpeechLM) | - | 微软退出后最大学术受益者 |

---

## 调研质量自检

- [x] ICASSP 2025/2026 数据已通过搜索确认
- [x] Interspeech 2025 Special Session 和 Challenge 已确认
- [x] NeurIPS 2025 / ICLR 2025/2026 speech papers 已交叉验证
- [x] ACL/EMNLP/AAAI/ICML/CVPR 收录论文已从 Sessions 1-7 汇总
- [x] Blizzard Challenge 2025 任务和趋势已确认
- [x] VoiceMOS → AudioMOS 品牌演进已确认
- [x] 融资数据来自多个来源交叉验证
- [x] 市场规模数据标注了来源和日期
- [x] 跨 Session 趋势分析基于 6 个 Session 的完整数据
- [x] 所有网络访问使用 Playwright
