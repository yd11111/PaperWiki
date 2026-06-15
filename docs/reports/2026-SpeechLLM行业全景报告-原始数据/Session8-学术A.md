# Session 8: 学术 A — ICT/CAS (LLaMA-Omni) / X-LANCE (SLAM-LLM) / CUHK-SZ (Amphion+MaskGCT)

> **报告范围**: Speech LLM / Omni / 全双工对话
> **调研日期**: 2026-06-08
> **数据来源**: arXiv API, GitHub API, vault 已有论文笔记, 公开技术报告
> **5 层搜索覆盖**: Layer 1 (GitHub org) ✓ | Layer 2 (核心人搜索) ✓ | Layer 3 (arXiv affiliation) ✓ | Layer 4 (产品/竞赛) 部分 | Layer 5 (引用网络) ✓ (通过 vault 笔记)

> [!warning] 归属勘误
> 原始任务将 LLaMA-Omni 系列归属于 CMU,实际该系列来自 **中科院计算所 (ICT/CAS)** Yang Feng 团队 (ictnlp)。Ichigo 来自独立研究者 (Homebrew Research),非 CMU。原始任务中归属 X-LANCE 的 WavSLM (Concordia/Mila)、DualSpeechLM (CUHK/Tsinghua)、OpenS2S (CASIA) 以及归属 CUHK-SZ 的 OpenOmni (SIAT/Alibaba) 均为误归属,本报告已修正。相关论文仍在 vault 中可查阅,但不在本 session 的实验室覆盖范围内。

---

## 1. ICT/CAS — Yang Feng NLP 团队 (ictnlp)

### 1.1 基本信息

| 项目 | 详情 |
|------|------|
| 机构 | 中国科学院计算技术研究所 (Institute of Computing Technology, CAS) |
| 实验室 | 自然语言处理研究组 (ICTNLP) |
| 导师 | Yang Feng (冯洋) |
| GitHub | [github.com/ictnlp](https://github.com/ictnlp) (90 repos) |
| 核心学生 | Qingkai Fang (方清凯, LLaMA-Omni 系列一作), Shoutao Guo, Shaolei Zhang, Zhengrui Ma, Yan Zhou |
| 研究方向 | 同声传译 (simultaneous translation), 多模态 LLM, 语音交互, RAG, LLM 微调 |
| 定位 | 从同声传译扩展到端到端语音交互的 NLP 实验室; LLaMA-Omni 系列是 open-source modular SpeechLM 的重要标杆 |

### 1.2 论文时间线 (2024-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2024.09 | LLaMA-Omni | 2409.06666 | NAR CTC speech decoder + Whisper encoder → Llama-3.1-8B; 226ms 延迟 | **首个低延迟开源端到端 SpeechLM**: ICLR 2025 |
| 2024 | StreamSpeech | — | 同声传译: ASR+翻译+TTS all-in-one | 同传技术积累,为语音交互提供基础 |
| 2025.05 | LLaMA-Omni 2 | 2505.02625 | AR streaming speech decoder (CosyVoice 2 架构), Gate Fusion, 0.5B-32B 系列 | **SpeechLM 升级**: ACL 2025, AR 替代 NAR 提升自然度 |
| 2025 | FreezeEmpath | — | 冻结 LLM 训练共情语音聊天机器人 | 共情对话扩展 |
| 2025 | CSLM | — | 高效跨语言 Speech LM 训练 | 多语言扩展 |

### 1.3 技术栈全景

| 维度 | ICT/CAS 技术栈 |
|------|---------------|
| **语音编码器** | Whisper-large-v3 encoder (frozen) + 5x downsample adapter (FFN) |
| **LLM 骨干** | V1: Llama-3.1-8B-Instruct; V2: Qwen2.5 系列 (0.5B/1.5B/3B/7B/14B/32B) |
| **语音解码器** | V1: NAR CTC decoder (低延迟但自然度有限); V2: AR TTS LM (Qwen2.5-0.5B init) + chunk-aware causal flow matching + HiFi-GAN (复用 CosyVoice 2 架构) |
| **对话策略** | V1: 同步 text + speech 生成 (parallel output); V2: Read-Write 流式交替 (R=3, W=10) |
| **训练数据规模** | 200K 合成多轮对话样本 (InstructS2S-200K); 远少于同期其他系统 |
| **推理延迟** | V1: 226ms (NAR 优势); V2: ~583ms (7B), ~663ms (14B) |
| **多语言** | V2-32B 支持中英双语; 其余版本仅英语 |
| **情感/副语言** | 不支持; 训练数据仅包含常规对话 |

### 1.4 架构演进

```
StreamSpeech (同声传译 All-in-One)
    ↓ 同传→对话技术迁移
LLaMA-Omni V1 (NAR CTC decoder, Llama-3.1-8B, 2024.09)
    ├── Whisper encoder + adapter → LLM → 同步输出 text + speech
    ├── NAR CTC: 延迟极低 (226ms) 但自然度有限
    └── InstructS2S-200K 合成数据
    ↓ NAR → AR, CosyVoice 2 解码器
LLaMA-Omni 2 (AR streaming decoder, Qwen2.5, 2025.05)
    ├── Gate Fusion (hidden states + text embedding, element-wise sigmoid)
    ├── Read-Write 流式: R=3 text tokens → W=10 speech tokens
    ├── 0.5B-32B 多规模系列
    └── S2T-S2S gap 显著缩小 (Web Questions: 3.2 vs GLM-4-Voice 16.3)
    ↓ 方向扩展
FreezeEmpath (冻结 LLM 共情对话) + CSLM (跨语言 SLM)
```

### 1.5 独特技术赌注

1. **Modular SpeechLM + 极少数据**: 200K 合成样本即可达到竞争力性能 (vs GLM-4-Voice 百万小时),核心信念是"好的预训练 LLM + 好的流式解码器 = 少数据高质量"。
2. **Gate Fusion**: LLM hidden states (上下文信息) + text embedding (精确文本) 通过 element-wise sigmoid gate 自适应融合,是从 LLM 向下游生成模块传递信息的简洁方案。消融显示去掉 gate 后 WER 从 3.26→4.89。
3. **从 NAR→AR 的务实演进**: V1 用 NAR CTC 换取低延迟 (226ms),V2 识别到 NAR 自然度瓶颈后切换到 AR (CosyVoice 2),接受 ~400ms 延迟增加换取显著质量提升。
4. **同声传译→语音交互**: 团队从 StreamSpeech (同声传译) 自然过渡到 SpeechLM,复用了增量处理、流式输出等核心技术。

### 1.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| LLaMA-Omni | 3,100 | ICLR 2025, 端到端语音交互, Llama-3.1-8B |
| StreamSpeech | 1,300 | All-in-one 同声传译 |
| LLaVA-Mini | 576 | 多模态 LLM (图像+视频) |
| BayLing | 315 | 百聆: 中英文 LLM |
| LLaMA-Omni 2 | 273 | ACL 2025, AR 流式解码, Qwen2.5 多规模 |
| FlexRAG | 237 | RAG 框架 |
| Auto-RAG | 235 | 自动 RAG |

### 1.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) 数据效率极高: 200K 样本超越 GLM-4-Voice (百万小时),为资源有限的团队提供了可复现路线; (2) 开源完善: 代码+模型全开源,LLaMA-Omni 3.1K stars 是学术 SpeechLM 中最高; (3) 务实演进: NAR→AR 切换清晰,消融实验系统性强,提供了大量工程参考 (Gate Fusion, R:W 比例, TTS LM 预训练策略); (4) 多规模系列: 0.5B-32B 覆盖不同部署场景; (5) ICLR 2025 + ACL 2025 双顶会 |
| **短板** | (1) 不支持全双工: 纯 turn-based 半双工交互,与 Moshi/LSLM 方向差距明显; (2) 不支持情感/副语言控制: 训练数据仅常规对话; (3) 仅用合成数据: 200K 样本全为 LLM+TTS 合成,未在真人对话数据上验证; (4) ~600ms 延迟仍高于 LLaMA-Omni V1 (226ms) 和 Moshi (230ms); (5) 团队规模较小,论文产出量有限 |
| **影响力** | LLaMA-Omni 是开源 modular SpeechLM 的重要基准线,被 OpenS2S 等后续工作直接参考; InstructS2S-200K 数据构建范式被广泛复用 |
| **未来方向推测** | (1) 全双工能力引入; (2) 多语言扩展 (CSLM 方向); (3) 情感/共情对话 (FreezeEmpath 方向); (4) 与更大 LLM (Qwen3, Llama 4) 结合; (5) 可能从 modular 向 native SpeechLM 演进 |

---

## 2. X-LANCE / 上海交通大学 — Xie Chen & Kai Yu 实验室

### 2.1 基本信息

| 项目 | 详情 |
|------|------|
| 机构 | 上海交通大学 (Shanghai Jiao Tong University, SJTU) |
| 实验室 | X-LANCE (Cross Media Language Intelligence Lab, 跨媒体语言智能实验室) |
| 导师 | Xie Chen (陈谢), Kai Yu (俞凯) |
| GitHub | [github.com/X-LANCE](https://github.com/X-LANCE) (47 repos) |
| 核心学生 | Ziyang Ma (马子阳, LSLM/Audio Interaction Model 一作), Wenxi Chen (陈文希, SLAM-Omni/SAC), Guanrou Yang (杨冠柔, WavCube), Yakun Song, Zhikang Niu, Ruiqi Yan |
| 研究方向 | 语音语言模型 (SLM), 全双工对话, 语音 codec, TTS, 对话系统, 多模态理解 |
| 定位 | 国内 Speech LLM 基础研究产出最密集的学术实验室; LSLM 是全双工 SLM 的开创性工作; SLAM-LLM 是重要的 Speech LLM 研究 toolkit |

### 2.2 论文时间线 (2023-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2023.09 | VoiceFlow | 2309.05027 | Rectified Flow Matching TTS (ICASSP 2024) | TTS 基础: Flow Matching 路线 |
| 2024.06 | TacoLM | 2406.15752 | GaTed Attention Codec LM, 零样本 TTS | Codec LM 路线探索 |
| 2024.08 | **LSLM** | 2408.02622 | **首个端到端"边说边听" SLM**: middle fusion + IRQ token | **全双工里程碑**: 形式化 FDM 问题,三种融合策略系统对比 |
| 2024.09 | CoT-ST | 2409.19510 | 链式思维同声传译 | 理解+翻译能力 |
| 2024.12 | **SLAM-Omni** | 2412.15649 | 音色可控端到端语音交互; 单阶段训练; 仅 15h 数据+4 GPU | **SpeechLM 交互**: 首个单阶段训练的对话系统 |
| 2025.03 | Spark-TTS | 2503.01710 | 单流解耦 speech tokens 的 LLM-based TTS | LLM-based TTS 路线 |
| 2025.05 | EmoVoice | 2504.12867 | LLM-based 情感 TTS with freestyle 文本提示 | 情感语音生成 |
| 2025.09 | Semantic-VAE | 2509.22167 | 语义对齐的 VAE 潜在表示用于 TTS | 语音表示学习 |
| 2025.10 | SAC | 2510.16841 | 语义-声学双流量化 neural speech codec | 语音 tokenizer 基础设施 |
| 2025.10 | **UniVoice** | 2510.04593 | 统一 AR ASR + Flow Matching TTS 的 LLM | **统一 SpeechLM**: 理解+生成统一 |
| 2025.10 | **UltraVoice** | 2510.22588 | 细粒度风格控制的语音对话模型 | **对话型 SpeechLM**: 情感+风格可控对话 |
| 2025.10 | DiSTAR | 2510.12210 | Diffusion over token AR 的语音生成 | 下一代语音生成架构 |
| 2025 | ProsodyEval | 2509.19928 | 韵律多样性评估指标和基准 | 评估基础设施 |
| 2025 | 语音评估 Position Paper | 2510.06927 | 负责任的 TTS 评估立场 | 评估方法论 |
| 2026.01 | ReStyle-TTS | 2601.03632 | 相对和连续风格控制的零样本 TTS | 风格控制 TTS |
| 2026.01 | CSP-FT | 2501.14273 | 特征特定部分微调的情感和说话人适配 | 高效微调 |
| 2026.05 | **WavCube** | 2605.06407 | 统一语音表示: SSL 压缩+声学注入,理解+生成 | **统一表示**: 为 SpeechLM 提供新 tokenizer 路线 |
| 2026.05 | X-Voice | 2605.05611 | 30 语言零样本跨语言语音克隆 | 多语言 TTS |
| 2026.06 | **Audio Interaction Model** | 2606.05121 | Always-on perceive-decide-respond loop; SoundFlow 框架; StreamAudio-2M | **下一代 LALM**: 实时交互+proactive 助手 |
| 2026.06 | **HoliTok** | 2605.29948 | 连续整体 tokenization,理解+生成双能力 | **统一 tokenizer**: 连续表示路线 |
| 2026.06 | WavTTS | 2606.03455 | 直接原始波形建模的零样本 TTS | 新生成范式 |

*注: SLAM-LLM toolkit 论文 accepted by IEEE JSTSP (2026), 作为统一框架收录上述多个工作。*

### 2.3 技术栈全景

| 维度 | X-LANCE 技术栈 |
|------|---------------|
| **语音编码器** | LSLM: vq-wav2vec (streaming 卷积); SLAM-Omni: Whisper; SAC: 双流量化; WavCube: SSL bottleneck + 声学注入; HoliTok: 连续 VAE |
| **LLM 骨干** | LSLM: 106M decoder-only Transformer; SLAM-Omni: 基于 SLAM-LLM 框架; UniVoice: Qwen2.5 系列; Audio Interaction: 自研 LALM |
| **语音解码器** | LSLM: GAN vocoder; SLAM-Omni: grouped semantic tokens + vocoder; UniVoice: Flow Matching decoder; DiSTAR: Diffusion + token AR |
| **对话策略** | LSLM: middle fusion + IRQ token (全双工); SLAM-Omni: 单阶段训练 + 多轮对话; Audio Interaction: perceive-decide-respond loop |
| **训练数据规模** | LSLM: LibriTTS 585h; SLAM-Omni: 最低 15h; UltraVoice: 自建大规模数据集; Audio Interaction: StreamAudio-2M (260 万样本) |
| **推理延迟** | LSLM: 实时 (streaming + IRQ); SLAM-Omni: 低延迟 (grouped tokens 加速) |
| **多语言** | X-Voice: 30 语言; SLAM-Omni: 中英; 其余多为英语或中英 |
| **情感/副语言** | EmoVoice: freestyle 文本情感控制; UltraVoice: 细粒度风格控制; WeSCon: 词级情感控制 |

### 2.4 架构演进

```
VoiceFlow (Flow Matching TTS, ICASSP 2024)
    ↓ TTS 基础
TacoLM (Codec LM TTS, 2024.06)
    ↓ 从 TTS → SLM
LSLM (边说边听, 2024.08)  ────────────────────────────────── 全双工开创
    ├── Middle Fusion (每层注入监听信号)
    ├── IRQ token (端到端 turn-taking)
    └── 形式化 FDM: P(r_t | R, S, C)
    ↓ 交互系统化
SLAM-Omni (音色可控语音交互, 2024.12) ← SLAM-LLM toolkit 支撑
    ├── 单阶段训练 (无需 ASR/TTS 预训练)
    ├── 15h 数据+4 GPU 即可训练
    └── Grouped semantic tokens 加速推理
    ↓ 理解+生成统一
UniVoice (统一 ASR+TTS, 2025.10) ← Codec: SAC (语义-声学双流)
UltraVoice (风格控制语音对话, 2025.10)
    ↓ 表示学习革新
WavCube (统一语音表示, 2026.05)
HoliTok (连续整体 tokenization, 2026.06)
    ↓ 下一代实时交互
Audio Interaction Model (always-on LALM, 2026.06)
    ├── SoundFlow 框架: perceive-decide-respond loop
    ├── StreamAudio-2M: 260 万流式数据, 7 能力 28 子任务
    └── Proactive-Sound-Bench 新基准

并行基础设施:
SLAM-LLM toolkit (IEEE JSTSP 2026) ← 统一框架, 1K+ stars
Spark-TTS (2025.03) ← 单流解耦 tokens
DiSTAR (2025.10) ← Diffusion + Token AR
评估: ProsodyEval + Position Paper on TTS Evaluation
```

### 2.5 独特技术赌注

1. **Middle Fusion 全双工范式 (LSLM)**: 首次系统证明在 AR 生成的每个 Transformer block 注入实时监听信号是最优融合策略。Early fusion 破坏生成 (WER 33.56%), Late fusion 噪声下脆弱, Middle fusion 两者兼顾。后续 OmniFlatten, FlexDuo 等系统都借鉴了这一思路。
2. **单阶段训练 (SLAM-Omni)**: 颠覆多阶段训练范式,证明仅 15h 数据+4 GPU 即可训练出有竞争力的语音对话系统。核心在于 grouped semantic tokens 减少序列长度 + 历史文本压缩减少上下文开销。
3. **Always-on Perceive-Decide-Respond (Audio Interaction Model)**: 从 offline LALM → streaming single-task → unified online LALM 的跨越,提出"始终在线"的交互范式,支持实时 ASR、流式指令跟随、proactive 助手。
4. **统一表示路线 (WavCube + HoliTok)**: 不走多码本/混合 tokenizer 的复杂路线,而是探索单一连续/压缩表示同时服务理解和生成。WavCube 用两阶段训练 (语义压缩→声学注入) 在 8x 压缩下接近 WavLM SUPERB 性能。
5. **SLAM-LLM 开放平台**: 作为研究 toolkit 支持 ASR/TTS/SLM/audio captioning 等多种任务,降低了 Speech LLM 研究门槛。

### 2.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| AniTalker | 1,600 | ACM MM 2024, 语音驱动说话人脸生成 |
| SLAM-LLM | 1,036 | Speech/Language/Audio/Music LLM toolkit (IEEE JSTSP) |
| VoiceFlow-TTS | 374 | ICASSP 2024, Flow Matching TTS |
| Spark-TTS | (外部合作) | LLM-based 单流 TTS |

**注意**: LSLM 仅提供 demo 页面,未开源模型权重或训练代码。SLAM-Omni 在 SLAM-LLM 框架内提供完整复现。Audio Interaction Model (2026.06) 为最新工作,开源状态待确认。

### 2.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **全双工方向产出最密集**: 从 LSLM (形式化开创) → SLAM-Omni (实用系统) → Audio Interaction Model (下一代),持续推进全双工/实时交互前沿; (2) **技术栈覆盖最广**: codec (SAC), tokenizer (WavCube, HoliTok), LLM-TTS (UniVoice, Spark-TTS), 对话 (SLAM-Omni, UltraVoice), 评估 (ProsodyEval),全链路自研; (3) **SLAM-LLM 生态**: 1K stars 的统一框架降低研究门槛; (4) **论文产出量惊人**: 2024-2026 发表 20+ 篇相关论文; (5) **方法论贡献**: LSLM 的 middle fusion 和 IRQ token 被广泛借鉴 |
| **短板** | (1) LSLM 未开源是遗憾; (2) 实验规模普遍较小 (LSLM 106M/585h, SLAM-Omni 15h); (3) 缺少 large-scale 预训练的 flagship model (vs Qwen-Omni 4T tokens); (4) 产品化/工业落地不明确; (5) 团队论文分散在多个方向 (TTS/codec/SLM/evaluation),焦点不够集中; (6) GitHub stars 相对不高 (SLAM-LLM 1K) |
| **影响力** | LSLM 是全双工 SLM 的三大开创性工作之一 (与 dGSLM, Moshi 并列); SLAM-LLM 是国内使用最广泛的 Speech LLM 研究框架; middle fusion 策略被 OmniFlatten/FlexDuo/VITA 等后续工作借鉴 |
| **未来方向推测** | (1) Audio Interaction Model 可能发展为完整的 always-on 对话系统; (2) WavCube/HoliTok 的统一表示可能成为下一代 SpeechLM 的 tokenizer 基础; (3) 与工业界合作 (Spark-TTS 已有合作迹象) 推动产品化; (4) StreamAudio-2M 数据集可能成为流式 LALM 的标准训练资源 |

---

## 3. CUHK-SZ — Zhizheng Wu 实验室 (Amphion/MaskGCT 团队)

### 3.1 基本信息

| 项目 | 详情 |
|------|------|
| 机构 | 香港中文大学 (深圳) (The Chinese University of Hong Kong, Shenzhen) |
| 实验室 | Speech, Audio and Music Intelligence (SAMI) Lab |
| 导师 | Zhizheng Wu (吴志正) |
| GitHub | [github.com/open-mmlab/Amphion](https://github.com/open-mmlab/Amphion) (OpenMMLab 托管) |
| 核心学生 | Yuancheng Wang (王远成, MaskGCT/Metis/Amphion 核心贡献者), Xueyao Zhang, Haoyue Zhan, Dekun Chen (FlexiVoice), Chaoren Wang, Liumeng Xue, Haorui He (Emilia), Jiaqi Li |
| 研究方向 | 语音合成, 语音 tokenizer, 语音生成基础模型, Speech LLM 分析与对齐, 语音评估, 数据集构建 |
| 定位 | 国内语音生成基础设施最完善的学术实验室; Amphion (9.8K stars) 是最大的开源语音生成 toolkit; 近期从纯 TTS 向 Speech LLM 分析和对齐方向扩展 |

### 3.2 论文时间线 (2023-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2023.12 | Amphion | 2312.09911 | 开源 Audio/Music/Speech 生成 toolkit (ICASSP 2024) | **基础设施**: 集成 TTS/VC/SVC/codec/vocoder |
| 2024.03 | NaturalSpeech 3 | 2403.03100 | 分解式 codec + diffusion 零样本 TTS (与 MSRA 合作) | 语音生成 SOTA |
| 2024.07 | Emilia | 2407.05361 | 200K+ 小时多语言语音数据集 (SLT 2024) | **数据基础设施**: 被 MaskGCT/Metis 等广泛使用 |
| 2024.09 | **MaskGCT** | 2409.00750 | Masked generative codec transformer, 全 NAR 零样本 TTS | **语音生成里程碑**: 无需 alignment/duration prediction 的 NAR TTS |
| 2025.02 | **Metis** | 2502.03128 | 基础语音生成模型: 300K h 掩码生成预训练 → 5 任务微调 | **基础模型**: 首个大规模 speech generation foundation model |
| 2025.05 | Vevo2 | 2508.16332 | 统一可控语音+歌声生成 (IEEE TASLP) | 统一生成框架 |
| 2025.08 | NVSpeech | 2508.04195 | 副语言发声建模: 174K 句 573h 词级标注 | 副语言数据+模型 |
| 2025.08 | TaDiCodec | 2508.16790 | 文本感知 diffusion speech tokenizer, 6.25Hz 单码本 | **SLM 基础**: 为 Speech LM 设计的 tokenizer |
| 2025.10 | FlexiCodec | 2510.00981 | 动态帧率 codec 3-12.5Hz (ICLR 2026) | **codec 创新**: 自适应帧率 |
| 2025.10 | TASLA | 2510.14934 | 多层动态注意力的文本对齐 speech tokens, ~2.62Hz | 超低帧率 tokenizer |
| 2025.11 | SpeechJudge | 2511.07931 | 99K 对语音评估数据 + 生成式 reward model | **评估基础设施** |
| 2025.12 | Aliasing-Free Synthesis | 2512.20211 | 抗混叠 vocoder/codec (IEEE TASLP) | 信号处理基础 |
| 2026.01 | **FlexiVoice** | 2601.04656 | 自然语言指令风格控制 TTS: DPO + 多目标 GRPO | 可控 TTS |
| 2026.01 | **TARS** | 2601.05543 | RL 框架对齐 text/speech 条件轨迹 (ACL 2026) | **核心 Speech LLM**: 缩小模态推理差距 |
| 2026.01 | VoxPrivacy | 2601.19956 | SLM 交互隐私评估基准 | SLM 安全评估 |
| 2026.02 | SiTok | 2602.06602 | 简单图像 tokenizer (与 Apple 合作) | tokenizer 方法论 |
| 2026.03 | **Anatomy of Modality Gap** | 2603.01502 | 跨层 CKA 分析 speech vs text 表示 | **Speech LLM 分析**: 揭示模态差距机制 |
| 2026.03 | NV-Bench | 2603.15352 | 1651 句多语言非语言发声基准 (Interspeech 2026) | 评估基础设施 |
| 2026.04 | MimicLM | 2604.11552 | 反转数据构建: 合成输入+真实目标的零样本语音模仿 | 语音克隆新范式 |
| 2026.04 | VoxSafeBench | 2604.14548 | SLM 社会对齐评估: safety/fairness/privacy | SLM 安全评估 |
| 2026.06 | **Entity Binding Failures** | 2606.04474 | Speech LLM 推理中的实体绑定失败 + EA-CoT 干预 | **Speech LLM 分析**: 诊断推理缺陷 |

### 3.3 技术栈全景

| 维度 | CUHK-SZ 技术栈 |
|------|---------------|
| **语音编码器** | MaskGCT: VQ-VAE on w2v-BERT 2.0 (semantic codec); TaDiCodec: 文本感知 diffusion; FlexiCodec: 动态帧率; TASLA: 多层注意力 2.62Hz |
| **LLM 骨干** | TARS: 7B Speech LLM (与 Microsoft 合作); 多数 TTS 工作不涉及 LLM backbone |
| **语音解码器** | MaskGCT: Masked generative transformer (T2S + S2A 两阶段, 全 NAR); Metis: 掩码生成预训练+任务微调; FlexiVoice: DPO+GRPO 对齐 |
| **对话策略** | 实验室目前不直接做全双工对话系统,但通过 TaDiCodec/TARS/Entity Binding 等工作为 Speech LLM 提供基础研究 |
| **训练数据规模** | Emilia: 200K+ 小时 (6 语言); Metis: 300K 小时预训练; MaskGCT: 100K 小时 |
| **推理延迟** | MaskGCT: NAR (50 步迭代, 非实时); TASLA: 2.62Hz 极低帧率 → 超短序列 |
| **多语言** | Emilia: 6 语言 (en/zh/de/fr/ja/ko); MaskGCT: 中英; FlexiVoice: 中英 |
| **情感/副语言** | NVSpeech: 14 类非语言发声; FlexiVoice: 自然语言指令情感控制; SpeechJudge: 自然度评估 |

### 3.4 架构演进

```
NaturalSpeech 3 (分解式 codec + diffusion, 2024.03, 与 MSRA)
    ↓ 语音生成基础
MaskGCT (全 NAR masked generative TTS, 2024.09) ← Emilia 200K h 数据支撑
    ├── VQ-VAE semantic codec (w2v-BERT 2.0)
    ├── T2S: Masked Generative Transformer (695M, 50 步)
    ├── S2A: 逐层 Masked Generation (353M)
    └── 无需 alignment/duration prediction
    ↓ 从任务特定 → 基础模型
Metis (Speech Generation Foundation Model, 2025.02)
    ├── 300K h 掩码生成预训练
    ├── 5 任务微调 (TTS/VC/TSE/SE/Lip2Speech)
    └── <20M trainable params 或 300x 更少数据
    ↓ Tokenizer 演进
TaDiCodec (文本感知 Diffusion Tokenizer, 2025.08)  ─── 为 SLM 设计
FlexiCodec (动态帧率 3-12.5Hz, ICLR 2026, 2025.10)
TASLA (文本对齐 2.62Hz, 2025.10)
    ↓ Speech LLM 方向扩展
TARS (RL 对齐 speech-text 轨迹, ACL 2026, 2026.01) ← 与 Microsoft 合作
Anatomy of Modality Gap (CKA 分析, 2026.03) ← 揭示模态差距本质
Entity Binding Failures (推理诊断, 2026.06)  ← EA-CoT 干预

并行基础设施:
Amphion toolkit (9.8K stars, ICASSP 2024) ← 统一开源平台
Emilia dataset (200K+ h, SLT 2024)  ← 数据基础设施
SpeechJudge (99K pairs, 2025.11)  ← 评估基础设施
NV-Bench (非语言发声基准, 2026.03)
VoxSafeBench + VoxPrivacy (安全评估, 2026) ← SLM 安全方向
```

### 3.5 独特技术赌注

1. **Masked Generative 路线 (MaskGCT → Metis)**: 完全放弃自回归,用 masked generation 做语音。MaskGCT 首次证明全 NAR 可在 TTS 上达到 SOTA (SIM-O 0.728, WER 2.47% on Seed-TTS test-en)。Metis 将此路线扩展为通用 speech generation foundation model。
2. **极低帧率 tokenizer (TaDiCodec/FlexiCodec/TASLA)**: 从 50Hz/25Hz 推低到 6.25Hz/3Hz/2.62Hz,大幅减少 LM 需要建模的序列长度,是提升 Speech LLM 效率的关键基础设施。
3. **Speech LLM 分析与对齐 (TARS + Anatomy + Entity Binding)**: 不直接构建 SpeechLM 系统,而是深入分析现有系统的缺陷 (模态推理差距、实体绑定失败) 并提出修复方案 (RL 对齐, CoT 干预)。这种"基础研究"定位独特。
4. **完整基础设施生态**: Amphion (toolkit) + Emilia (data) + SpeechJudge (eval) + VoxSafeBench (safety) 构成完整的 Speech LLM 研究基础设施栈。
5. **Emilia 大规模数据集**: 200K+ 小时、6 语言的开源语音数据集,被 MaskGCT、Metis、OpenS2S 等多个项目使用,是语音生成领域最大的开源训练数据之一。

### 3.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| Amphion | 9,800 | 开源 Audio/Music/Speech 生成 toolkit (含 MaskGCT, Vevo, codecs, vocoders, eval) |
| Emilia | (含在 Amphion) | 200K+ 小时 6 语言语音数据集 |
| MaskGCT | (含在 Amphion) | 全 NAR masked generative zero-shot TTS |
| Metis | (含在 Amphion) | 基础语音生成模型 |
| FlexiCodec | (含在 Amphion) | 动态帧率 codec (ICLR 2026) |
| SpeechJudge | (含在 Amphion) | 语音自然度评估 |

**注意**: Amphion 9.8K stars 是学术语音生成项目中最高的 (超过 ESPnet 的 8.6K)。几乎所有 CUHK-SZ 的语音工作都整合在 Amphion 框架下,形成统一的开源生态。

### 3.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **开源基础设施最完善**: Amphion 9.8K stars + Emilia 200K h + SpeechJudge,构成完整的语音研究平台; (2) **Tokenizer/codec 创新密度最高**: TaDiCodec/FlexiCodec/TASLA 在帧率压缩方面持续突破; (3) **Speech LLM 分析独特**: TARS/Anatomy/Entity Binding 提供了他人不做的机制分析; (4) **学术影响力大**: MaskGCT 被 IndexTTS2 采用; Emilia 被多个项目使用; (5) **与工业界合作广泛**: MSRA (NaturalSpeech 3), Microsoft (TARS), Apple (SiTok) |
| **短板** | (1) **不直接做 Speech LLM 对话系统**: 无自研端到端 SpeechLM/全双工系统; (2) MaskGCT 偏 TTS 而非对话; (3) Metis 虽是基础模型但仅验证 TTS/VC/SE 等传统任务; (4) Speech LLM 分析工作 (TARS 等) 依赖他人的 SpeechLM 系统; (5) Amphion 虽大但 Speech LLM 相关模块尚不完善 |
| **影响力** | Amphion 是语音生成领域最大的开源 toolkit; Emilia 是最大的开源语音数据集之一; MaskGCT 的 masked generative 路线影响了后续 NAR TTS 发展; 极低帧率 tokenizer 研究为 Speech LLM 效率优化提供了基础 |
| **未来方向推测** | (1) 可能将 Metis 扩展为完整的 Speech LLM (理解+生成+对话); (2) 极低帧率 tokenizer + TARS 对齐可能催生自研 SpeechLM; (3) Amphion 可能集成 Speech LLM 训练 recipe; (4) VoxSafeBench/VoxPrivacy 方向可能发展为 SLM safety 的标准评估; (5) 与 Microsoft 的合作可能加速 Speech LLM 方向投入 |

---

## 4. 横向对比矩阵

### 4.1 技术路线对比

| 维度 | ICT/CAS (LLaMA-Omni) | X-LANCE (SLAM-LLM) | CUHK-SZ (Amphion) |
|------|----------------------|---------------------|-------------------|
| **核心定位** | Modular SpeechLM 系统 | 全双工/实时交互研究 + 全栈技术 | 语音生成基础设施 + SLM 分析 |
| **旗舰系统** | LLaMA-Omni 2 (ACL 2025) | Audio Interaction Model (2026) | Metis (基础模型) + TARS (对齐) |
| **架构范式** | Whisper + LLM + AR streaming decoder | Middle fusion 全双工; 单阶段训练 | Masked generative; RL 对齐 |
| **全双工** | 不支持 | **支持** (LSLM 开创, Audio Interaction 推进) | 不做 (提供基础研究) |
| **LLM 规模** | 0.5B-32B (Qwen2.5) | 106M-数 B (多种 backbone) | 不直接训练 SpeechLM |
| **训练数据** | 200K 合成样本 (极少) | 15h-260 万样本 (跨度大) | 100K-300K h (大规模) |
| **Tokenizer 创新** | 复用 CosyVoice 2 | SAC, WavCube, HoliTok | TaDiCodec, FlexiCodec, TASLA |
| **开源 Stars** | 3,373 (LLaMA-Omni + V2) | 1,036 (SLAM-LLM) | 9,800 (Amphion) |
| **顶会论文** | ICLR 2025, ACL 2025 | IEEE JSTSP 2026 | ICLR 2026, ACL 2026, AAAI 2026 |
| **论文产出 (2024-2026)** | ~5 篇 (聚焦) | ~25 篇 (广覆盖) | ~20 篇 (基础设施+分析) |

### 4.2 能力维度对比

| 能力 | ICT/CAS | X-LANCE | CUHK-SZ |
|------|---------|---------|---------|
| 端到端语音对话 | ✓ (LLaMA-Omni 2) | ✓ (SLAM-Omni) | ✗ (不直接做) |
| 全双工对话 | ✗ | ✓ (LSLM, AIM) | ✗ |
| 情感/风格控制 | ✗ | ✓ (UltraVoice, EmoVoice) | ✓ (FlexiVoice, NVSpeech) |
| 多语言 | 中英 (32B) | 30 语言 (X-Voice) | 6 语言 (Emilia) |
| Speech LLM 分析 | ✗ | ✗ | ✓ (TARS, Anatomy, Entity Binding) |
| 语音 tokenizer 创新 | ✗ (复用) | ✓ (SAC, WavCube, HoliTok) | ✓ (TaDiCodec, FlexiCodec, TASLA) |
| 开源 toolkit | ✗ | ✓ (SLAM-LLM) | ✓ (Amphion) |
| 大规模数据集 | ✗ | ✓ (StreamAudio-2M) | ✓ (Emilia 200K h) |
| 评估基准 | ✗ | ✓ (ProsodyEval) | ✓ (SpeechJudge, NV-Bench, VoxSafeBench) |

### 4.3 GitHub Stars 对比

| 团队 | 最高 Stars 项目 | Stars | 说明 |
|------|---------------|-------|------|
| CUHK-SZ | Amphion | 9,800 | 语音生成 toolkit (含 MaskGCT, Metis, Emilia 等) |
| ICT/CAS | LLaMA-Omni | 3,100 | 端到端 SpeechLM (ICLR 2025) |
| X-LANCE | AniTalker | 1,600 | 说话人脸生成 (非 SpeechLM) |
| X-LANCE | SLAM-LLM | 1,036 | Speech LLM toolkit (IEEE JSTSP) |

---

## 5. 关键发现

### 发现 1: 三个实验室形成互补的分工

三个实验室在 Speech LLM 生态中占据了不同但互补的位置:
- **ICT/CAS** 做"应用层" — 提供可直接使用的端到端 SpeechLM 系统 (LLaMA-Omni 系列)
- **X-LANCE** 做"研究层" — 探索全双工/实时交互的前沿问题 (LSLM, Audio Interaction Model)
- **CUHK-SZ** 做"基础设施层" — 提供 toolkit (Amphion), 数据 (Emilia), 评估 (SpeechJudge), 和机制分析 (TARS)

这种自然分工意味着它们之间不是竞争关系,而是生态互补关系。例如,LLaMA-Omni 复用 CosyVoice 2 的 tokenizer,CUHK-SZ 的低帧率 tokenizer 未来可能被 X-LANCE 或 ICT/CAS 的系统采用。

### 发现 2: X-LANCE 是学术全双工 SLM 的核心推动者

在三个实验室中,X-LANCE 是唯一持续推进全双工/实时交互研究的。从 LSLM (2024, 形式化全双工) → SLAM-Omni (2024, 实用系统) → Audio Interaction Model (2026, always-on LALM),形成了清晰的技术演进线。LSLM 的 middle fusion 策略和 IRQ token 机制影响了后续多个系统的设计。

### 发现 3: CUHK-SZ 正在从 TTS 向 Speech LLM 转型

CUHK-SZ 传统强项是语音合成 (MaskGCT, NaturalSpeech 3),但 2026 年明显加速向 Speech LLM 方向扩展:
- **TARS** (ACL 2026): 用 RL 缩小 speech-text 模态推理差距
- **Anatomy of Modality Gap**: 用 CKA 分析揭示 SLM 内部模态表示差异
- **Entity Binding Failures**: 诊断 Speech LLM 推理缺陷
- **VoxSafeBench/VoxPrivacy**: 建立 SLM 安全评估标准

这种从"造组件"到"分析+修复系统"的转型,结合其强大的基础设施 (Amphion + Emilia),使 CUHK-SZ 在 Speech LLM 基础研究方面具有独特优势。

### 发现 4: 数据效率是学术实验室的核心竞争策略

与大厂动辄百万小时数据不同,三个学术实验室都在数据效率上做文章:
- **ICT/CAS**: 200K 合成样本超越 GLM-4-Voice (百万小时)
- **X-LANCE**: SLAM-Omni 仅 15h 数据+4 GPU 即可训练
- **CUHK-SZ**: Metis 用 <20M 可训练参数或 300x 更少数据达到 SOTA

这反映了学术实验室的资源约束倒逼出的方法论创新 — 不是更多数据,而是更好的架构、预训练策略和数据构建方法。

### 发现 5: 学术实验室的归属容易混淆

在调研过程中发现,多篇论文的实验室归属被广泛误传:
- LLaMA-Omni 常被误归为 CMU (实际 ICT/CAS)
- DualSpeechLM (CUHK/Tsinghua) 常与 X-LANCE 混淆
- OpenOmni (SIAT/Alibaba) 常与 CUHK-SZ 混淆

这提示在学术 Speech LLM 领域,作者流动性大、跨机构合作频繁,需要通过论文 affiliation 而非 arXiv 搜索结果来确认归属。

---

## 附: 数据来源标注

| 信息类型 | 来源 | 可信度 |
|---------|------|--------|
| 论文技术细节 | vault 已有精读笔记 (LSLM, LLaMA-Omni2, MaskGCT, Amphion) | 高 (经审阅) |
| GitHub stars/repos | GitHub API (2026-06-08 查询) | 高 |
| arXiv 论文列表 | arXiv 搜索 + vault 交叉验证 | 高 |
| 团队人物/归属 | 论文作者列表 + GitHub org + WebFetch 验证 | 中-高 |
| 实验室定位描述 | 基于论文产出模式的推断 | 中 |
| CUHK-SZ Speech LLM 转型判断 | 基于 2026 年论文趋势的推断 | 中 |
| 未来方向推测 | 基于技术演进和团队特点的推断 | 低-中 |
| CMU 实际 Speech LLM 贡献 | 公开信息有限,未深入调研 | 低 |
