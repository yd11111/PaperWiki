# Session 3: 月之暗面 + MiniMax + 面壁智能 + 蚂蚁集团

> 调研日期: 2026-06-08
> 方法: 5层搜索法 (组织搜索 + 核心人搜索 + affiliation搜索 + 产品/竞赛反推 + 引用网络)
> 方向: Speech LLM / Omni / 全双工对话

---

## 团队1: 月之暗面 / Moonshot AI (Kimi-Audio)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 月之暗面 (Moonshot AI) |
| **团队名** | KimiTeam (Kimi-Audio 团队) |
| **GitHub org** | [MoonshotAI](https://github.com/MoonshotAI) (38 个公开仓库) |
| **HuggingFace** | [moonshotai](https://huggingface.co/moonshotai) (Kimi-Audio-7B, Kimi-Audio-7B-Instruct) |
| **核心人物** | **Xu Tan** — 从 Microsoft Research 加入, Kimi-Audio 通讯作者; **Jianwei Yu** — 语音方向核心; **Dongchao Yang** — 语音生成/编辑; **Songxiang Liu** — 语音处理; **Yichong Leng** — 语音系统; **Zeqian Ju** — 音频 LLM |
| **产品线** | Kimi 大模型 (K2/K2.5/K2.6 文本/视觉), Kimi-Audio (语音基座), Kimi Code (编程助手), Kimi API Platform |
| **开源态度** | **积极开源** — Kimi-Audio 权重 + 推理代码 + 预训练权重 + 微调代码 + 评估工具箱全部开源; Kimi-K2 (10.8k stars), Kimi-VL 等也开源 |

### 论文时间线 (2024-2026)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2025.04 | **Kimi-Audio Technical Report** (2504.18425) — KimiTeam | preprint | **音频基座模型旗舰** |
| 2025.04 | **Kimi-Audio-Evalkit** — MoonshotAI | 开源工具 | 音频评估标准化 |
| 2025.05 | **Kimi-Audio-7B Finetune** — MoonshotAI | 开源代码 | 微调示例 |
| 2025.02 | **MoBA: Mixture of Block Attention** — MoonshotAI | preprint, 2.1k stars | 长上下文 LLM (间接支撑) |
| 2025.02 | **Moonlight: Muon is Scalable for LLM Training** (2502.16982) — MoonshotAI | preprint | 优化器 (间接支撑) |
| 2025.06 | **Kimi-K2** — MoonshotAI | 开源, 10.8k stars | 1T MoE LLM 基座 |
| 2025.06 | **Kimi-K2.5** — MoonshotAI | 开源, 2k stars | 多模态 Agent |
| 2026.06 | **Attention Residuals** — MoonshotAI | preprint, 3.3k stars | 注意力架构创新 |

> **说明**: 月之暗面在 Speech LLM 方向的核心产出集中在 Kimi-Audio 一篇技术报告 (但内容极为扎实)。"MoonVoice" 名称在公开信息中未出现,可能为内部产品名或尚未发布。Kimi-Audio 的定位是"音频基座模型",覆盖理解+生成+对话,而非单独的 TTS/ASR 论文。Kimi 系列文本/视觉模型迭代极快 (K2→K2.5→K2.6, 2025-2026), 语音方向投入时间较晚但一出手即 SOTA 水平。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **语音编码器** | 混合输入: (1) **Whisper encoder** 提取连续声学特征,下采样到 12.5Hz; (2) **VQ tokenizer** 提取离散语义 tokens (12.5Hz)。两路 concat 后输入 LLM |
| **LLM 骨干** | **Qwen 2.5-7B** 初始化, transformer-based, 共享层处理多模态输入 |
| **语音解码器** | 并行 text head + audio head 自回归生成 → **flow matching detokenizer** (chunk-wise streaming, look-ahead) → **BigVGAN** vocoder → 24kHz 波形 |
| **对话策略** | 端到端 speech conversation; 支持多轮对话 (audio-text interleaving); 并行 text/audio token 生成 |
| **训练数据规模** | **1300 万小时** 音频数据 (speech + music + sound) + 文本数据预训练 |
| **推理延迟** | chunk-wise streaming detokenizer 支持低延迟; 具体 RTF 未公开 |
| **多语言** | 中英为主; Whisper 编码器天然支持多语言 |
| **情感/副语言** | Speech Conversation 评估: Speed Control 4.30 (超 GPT-4o 4.21), Emotion Control 4.27 (超 GPT-4o 4.05); 共情 3.39 (低于 GPT-4o 3.87) |

### 架构演进

```
月之暗面成立 (2023) — 专注长上下文 LLM
    ↓ 文本模型基础
Moonshot V1 (2024) — 128k 长上下文文本模型
    ↓ 多模态扩展
Kimi-VL (2025) — MoE 视觉语言模型
    ↓ 音频能力
Kimi-Audio-7B (2025.04) — 音频基座模型:
    ├── 混合输入: continuous Whisper + discrete semantic tokens
    ├── Qwen 2.5-7B 骨干 + 并行 text/audio heads
    ├── Flow Matching streaming detokenizer + BigVGAN
    └── 13M 小时预训练, ASR/AQA/AAC/SER/SEC/ASC + 端到端对话
    ↓ 模型迭代
Kimi-K2 (2025.06) — 1T MoE 文本基座 (32B active)
Kimi-K2.5 / K2.6 (2025-2026) — 多模态 Agent
    ↓ (推测) 下一步
Kimi-Audio v2 或 Kimi-Omni — 可能基于 K2/K2.5 架构升级音频能力
```

### 独特技术赌注

1. **混合音频输入 (Continuous + Discrete)**: 同时使用 Whisper 连续声学特征和 VQ 离散语义 tokens 作为输入,在 Speech LLM 中较为独特。连续特征保留声学细节,离散 tokens 提供语义结构,相比纯离散 (Moshi/Step-Audio) 或纯连续 (LatentLM) 方案更灵活
2. **13M 小时超大规模预训练**: 训练数据规模在开源 Audio LLM 中最大 (MiMo-Audio 1亿小时声称更大但可能有统计口径差异),覆盖 speech/music/sound 三类音频
3. **统一评估工具箱 (Kimi-Audio-Evalkit)**: 业界首个开源的音频 LLM 标准化评估工具,支持复现 baseline 结果,推动公平对比
4. **并行 Text/Audio Head 生成**: LLM 自回归同时输出文本 token 和音频 semantic token,实现真正的 speech-to-speech 对话
5. **从 Qwen 初始化**: 利用强大的预训练文本 LLM 作为骨干,快速获得语言理解能力,再通过大规模音频预训练适配音频

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| MoonshotAI/Kimi-Audio | 4.6k | 音频基座模型, 推理代码 + Instruct 权重 + 预训练权重 (Apache-2.0 + MIT) |
| MoonshotAI/Kimi-Audio-Evalkit | 171 | 音频评估工具箱 |
| moonshotai/Kimi-Audio-7B | HF 模型 | 预训练权重 |
| moonshotai/Kimi-Audio-7B-Instruct | HF 模型 | 指令微调权重 |
| moonshotai/Kimi-Audio-GenTest | HF 数据集 | 生成评测数据集 |
| Kimi-K2 | 10.8k | 1T MoE LLM (非语音, 但为未来语音升级提供基座) |

### 判断

- **优势**: Kimi-Audio 一出手即达到多项 SOTA (ASR AISHELL-1 WER 0.60, MMAU sound 73.27, VoiceBench Avg 76.93); 13M 小时超大规模预训练; 开源彻底 (权重 + 代码 + 微调 + 评估工具箱); 评估工具箱填补行业空白; Xu Tan (前微软) 带来的语音研究深度; Kimi 主模型迭代极快,未来可能快速升级音频能力
- **短板**: 目前仅一篇 Kimi-Audio 技术报告,论文产出密度低; 无独立的 TTS 研究 (依赖端到端生成); 无全双工对话论文; Speech Conversation 中共情 (Empathy 3.39) 和风格控制 (Style 4.09) 弱于 GPT-4o (3.87/4.54); "MoonVoice" 未见公开信息,可能尚未发布; 语音方向起步较晚 (2025.04), 积累时间短
- **下一步推测**: 可能基于 Kimi-K2 (1T MoE) 骨干升级 Kimi-Audio 到更大规模; 可能增加全双工对话能力; 可能发布独立的 TTS/voice cloning 产品; Kimi-Audio-Evalkit 可能成为行业标准评估工具
- **关键观察**: 月之暗面走的是"一步到位"策略 — 不像阿里 FunAudioLLM 分 SenseVoice/CosyVoice/FunASR 多条线独立发展,而是用一个 Kimi-Audio 一次性覆盖 ASR/AQA/AAC/SER/对话等全部任务。这种策略的优势是架构统一,劣势是每个子任务的深度可能不如专门模型

---

## 团队2: MiniMax (MiniMax-Speech)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | MiniMax (稀宇科技) |
| **团队名** | MiniMax Speech Team |
| **GitHub org** | [MiniMax-AI](https://github.com/MiniMax-AI) (MiniMax-01, MiniMax-M3 等, 无独立语音开源) |
| **HuggingFace** | [MiniMaxAI](https://huggingface.co/MiniMaxAI) (MiniMax-Speech 系列模型, MiniMax-Speech-Tech-Report) |
| **核心人物** | **Bowen Zhang** — MiniMax-Speech 一作; **Haozhe Zhang** — 通讯作者; **Congchao Guo** — 核心; **Junjie Yan (颜水成的学生?)** — MiniMax 联合创始人/CTO; **Geng Yang** — 核心 |
| **产品线** | MiniMax Speech-01/2.8 (TTS API), MiniMax Music 2.6 (音乐生成), Hailuo (视频生成), Talkie (AI 角色对话), MiniMax M3 (文本 LLM), MiniMax Code, 海螺 AI |
| **开源态度** | **半开源** — MiniMax-Speech 技术报告 + 部分模型权重在 HuggingFace; MiniMax-01 (text LLM) 开源; 但 Speech 模型主要通过商业 API 提供 |

### 论文时间线 (2024-2026)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2024.08 | **Speech-01** — MiniMax 产品发布 | 产品公告 | 商业 TTS 模型 |
| 2024.12 | **MiniMax-01** — MiniMax-AI | tech report | 456B MoE 文本模型 (Lightning Attention) |
| 2025.05 | **MiniMax-Speech** (2505.07916) — MiniMax | preprint | **TTS 旗舰论文** |
| 2026.xx | **MiniMax Speech 2.8** — MiniMax | 产品发布 | 最新 TTS 模型版本 |
| 2026.xx | **MiniMax M3** — MiniMax-AI | 开源 | 新一代文本 LLM |
| 2026.xx | **MiniMax Music 2.6** — MiniMax | 产品发布 | 音乐生成 |

> **说明**: MiniMax 在语音方向以产品驱动为主,学术论文产出少 (仅 1 篇 MiniMax-Speech)。Speech-01 (2024.08) 作为商业产品发布时无技术论文,直到 2025.05 才发表 MiniMax-Speech 技术报告。MiniMax 的语音方向主要服务于 Talkie (AI 角色对话) 和 API 商业化,而非独立的 Speech LLM / Omni 研究。**MiniMax 在 Speech LLM / Omni 方向无公开研究**,MiniMax-Speech 严格来说是 TTS 模型而非 Speech LLM。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **语音编码器** | **Learnable Speaker Encoder** — 从参考音频提取 timbre 特征,不需要参考音频的文本转写; Flow-VAE 增强音频质量 |
| **LLM 骨干** | 自回归 Transformer (具体参数量未公开); MiniMax-01 (456B MoE) 为文本基座但与 Speech 模型关系未明确 |
| **语音解码器** | **Flow-VAE** — 提升合成音频的整体质量; 自回归生成 |
| **对话策略** | TTS 定位 (文本到语音),非端到端对话模型; 通过 Talkie 产品实现 cascaded 对话 (ASR → LLM → TTS) |
| **训练数据规模** | "millions of hours of high-quality audio data" (Speech-01 描述); 具体数值未公开 |
| **推理延迟** | Speech-01: 延迟降低 30% (相对基线); 超长文本 (1000 万字符) 单次合成; 5 秒快速 voice cloning |
| **多语言** | MiniMax-Speech: **32 语言**; Speech-01: 11 语言 |
| **情感/副语言** | Speech-01: 情感智能 (从文本预测情感线索, 喜怒哀乐等); 上下文情感理解; 数千种声音特征组合 |

### 架构演进

```
MiniMax 成立 (2021) — AI 产品公司
    ↓ 文本 LLM
MiniMax-01 (2024.12) — 456B MoE (Lightning Attention)
    ↓ 语音产品
Speech-01 (2024.08) — 首代商业 TTS: RL + diffusion, 11 语言
    ↓ 技术突破
MiniMax-Speech (2025.05) — AR Transformer TTS:
    ├── Learnable Speaker Encoder (无需参考文本)
    ├── Flow-VAE 增强音频质量
    ├── 32 语言零样本 TTS
    └── TTS Arena #1, SOTA voice cloning
    ↓ 版本迭代
MiniMax Speech 2.8 (2026) — 最新版 TTS
    ↓ 并行: 音乐/视频
MiniMax Music 2.6 — 音乐生成
Hailuo 2.3 — 视频生成
    ↓ 产品矩阵
Talkie — AI 角色对话 (Speech 作为语音引擎)
海螺 AI — 综合 AI 助手
```

### 独特技术赌注

1. **Learnable Speaker Encoder (免转写)**: MiniMax-Speech 的核心创新 — 传统 voice cloning 需要参考音频的文本转写, MiniMax-Speech 的 speaker encoder 直接从音频提取 timbre,消除了 ASR 步骤,支持更快的零样本 cloning
2. **Flow-VAE**: 结合 Flow Matching 和 VAE 的混合生成架构,提升合成音频的保真度和自然度
3. **产品驱动的 TTS 路线**: 不追求端到端 Speech LLM / Omni 模型,而是专注做最好的 TTS 产品。TTS Arena 排名第一验证了这条路线的商业价值
4. **超长文本合成**: Speech-01 支持 1000 万字符单次合成,远超多数模型 (10 万字符上限), 适用于有声书等长内容场景
5. **32 语言覆盖**: MiniMax-Speech 支持 32 种语言,在 TTS 模型中属上游水平

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| MiniMaxAI/MiniMax-Speech | HF 模型集 | 模型权重在 HuggingFace (collection 形式); 无独立 GitHub 仓库 |
| MiniMax-Speech-Tech-Report | HF Space | 技术报告交互页面 |
| MiniMax-AI/MiniMax-01 | GitHub | 456B MoE 文本模型 (非语音) |
| MiniMax-AI/MiniMax-M3 | GitHub | 新一代文本模型 (非语音) |
| Speech-01 / Speech 2.8 | 不开源 | 商业 API, 核心模型闭源 |

### 判断

- **优势**: TTS Arena 排名第一,语音合成质量在商业产品中领先; 32 语言覆盖广; Learnable Speaker Encoder 的 zero-shot voice cloning 效果好; Talkie (AI 角色对话) 提供海量真实用户反馈; MiniMax 作为独角兽融资充裕 (据报估值超 25 亿美元); 产品矩阵完整 (语音+音乐+视频+文本)
- **短板**: **Speech LLM / Omni 方向公开研究为零** — 无端到端语音理解模型、无语音对话模型、无全双工研究; 学术论文仅 1 篇 (MiniMax-Speech), 论文密度极低; 开源程度低 (核心模型闭源, 仅 HuggingFace 部分权重); GitHub 社区影响力弱; 技术透明度不足, 无法深入评估其语音能力的技术深度
- **下一步推测**: 可能发布 MiniMax-Omni 或 MiniMax-Voice (端到端语音对话模型); Talkie 产品可能推动全双工对话需求; 可能将 Speech 能力集成到 M3 多模态模型中; MiniMax-Speech 2.8 可能发布更新技术报告
- **关键观察**: MiniMax 是本 session 中最偏"产品公司"的团队。与其他三家 (Kimi-Audio/VoxCPM/Ming-UniAudio 都有独立的学术技术报告) 不同,MiniMax 在语音方向的核心竞争力在于**产品化能力**而非研究深度。TTS Arena 排名第一证明其产品质量,但在 Speech LLM / Omni 这个方向上,MiniMax 的公开研究投入明显不足

---

## 团队3: 面壁智能 / ModelBest + THUHCSI (VoxCPM)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 面壁智能 (ModelBest Inc.) + 清华大学人机语音交互实验室 (THUHCSI) |
| **团队名** | VoxCPM Team (OpenBMB 社区) |
| **GitHub org** | [OpenBMB](https://github.com/OpenBMB) (VoxCPM 等); [modelbest](https://github.com/modelbest) |
| **HuggingFace** | [openbmb](https://huggingface.co/openbmb) (VoxCPM2, VoxCPM1.5, VoxCPM-0.5B) |
| **核心人物** | **Yixuan Zhou (周逸轩)** — VoxCPM 一作; **Guoyang Zeng** — VoxCPM 核心; **Zhiyong Wu (吴志勇)** — THUHCSI 清华教授, senior; **Zhiyuan Liu (刘知远)** — 清华 NLP 教授, OpenBMB 负责人; **Xin Liu** — 核心; **Xiang Li** — 核心 |
| **产品线** | VoxCPM / VoxCPM2 (TTS), MiniCPM 系列 (文本 LLM), ChatCPM, BMTools |
| **开源态度** | **高度开源** — VoxCPM 全系列 Apache-2.0 开源; OpenBMB 社区是国内最活跃的开源 LLM 社区之一; 27.5k stars |

### 论文时间线 (2024-2026)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2025.09 | **VoxCPM: Tokenizer-Free TTS** (2509.24650) — ModelBest + THUHCSI | **ICLR 2026** | **TTS 旗舰论文** |
| 2025.09 | **VoxCPM-0.5B** — OpenBMB | 开源发布 (HF #1 Trending) | 首版开源模型 |
| 2025.12 | **VoxCPM1.5** — OpenBMB | 开源发布 (GitHub #1 Trending) | SFT/LoRA + 44.1kHz |
| 2026.04 | **VoxCPM2** — OpenBMB | 开源发布 | 2B, 30 语言, Voice Design, 48kHz |
| 2024-2025 | **MiniCPM-4** — OpenBMB | 多篇论文 | VoxCPM 的 LLM 骨干基座 |

> **说明**: VoxCPM 是面壁智能在语音方向的核心项目,技术上是 TTS 模型而非 Speech LLM (不包含语音理解能力)。但其 **tokenizer-free** 架构在 TTS 领域是独特的技术路线,与主流的 discrete token 路线形成鲜明对比。VoxCPM 的成功主要体现在开源影响力 (27.5k stars, 双#1 Trending) 和工程完善度 (pip install + CLI + vLLM 部署)。ICLR 2026 接收进一步验证了其学术价值。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **语音编码器** | **无外部 tokenizer** — 全端到端; Causal Audio VAE (DAC-like, stride [2,5,8,8], 640x 下采样 → 25 Hz latent; VoxCPM2: AudioVAE V2 支持 16kHz 输入 → 48kHz 输出) |
| **LLM 骨干** | **MiniCPM-4** (VoxCPM2: 2B 参数) 初始化 TSLM; VoxCPM-0.5B: 0.5B |
| **语音解码器** | **LocDiT** (4 层 local diffusion transformer, outpainting 设计) → Causal Audio VAE decoder → 波形 |
| **对话策略** | 纯 TTS 模型,无语音理解/对话能力; 可通过外部 ASR+LLM 构建 cascaded 对话 |
| **训练数据规模** | VoxCPM-0.5B: 约 180 万小时内部数据 + 9.5 万小时 Emilia; VoxCPM2: **超 200 万小时** 多语言语音数据 |
| **推理延迟** | VoxCPM2: RTF ~0.30 (RTX 4090); Nano-vLLM 加速后 **RTF ~0.13**; VRAM ~8 GB |
| **多语言** | VoxCPM2: **30 语言** (含中文 9 种方言); VoxCPM-0.5B/1.5: 中英双语 |
| **情感/副语言** | VoxCPM2: **Voice Design** (自然语言描述生成新声音); **Controllable Voice Cloning** (保持音色 + 风格引导); 上下文感知韵律 |

### 架构演进

```
OpenBMB / 面壁智能 (清华系, 2022) — MiniCPM 小模型路线
    ↓ 文本 LLM 基础
MiniCPM 系列 (2024-2025) — 小参数高性能 LLM
    ↓ 语音扩展
VoxCPM-0.5B (2025.09) — tokenizer-free TTS:
    ├── FSQ 内部正则化瓶颈 (非预测目标)
    ├── TSLM (24L, MiniCPM-4 init) → FSQ → RALM (6L) → LocDiT
    ├── 端到端 flow matching 训练
    └── ICLR 2026 接收, HF #1 Trending
    ↓ 工程完善
VoxCPM1.5 (2025.12) — SFT/LoRA 微调 + 44.1kHz 输出 (GitHub #1 Trending)
    ↓ 规模化
VoxCPM2 (2026.04) — 2B 参数:
    ├── 30 语言 + 中文 9 方言
    ├── Voice Design (文本描述→声音)
    ├── Controllable Voice Cloning (音色保持+风格控制)
    ├── 48kHz AudioVAE V2 (16kHz 输入 → 48kHz 输出, 内置超分)
    ├── vLLM-Omni 官方支持 (OpenAI 兼容 API)
    └── 丰富生态: VoxCPM.cpp / ONNX / ANE / Rust / ComfyUI
```

### 独特技术赌注

1. **Tokenizer-free 路线 (核心差异)**: VoxCPM 的最大赌注 — 不使用外部 speech tokenizer (EnCodec/SpeechTokenizer 等),而是用 FSQ 作为内部可微正则化瓶颈诱导 semantic-acoustic 自然分工。这与 CosyVoice/IndexTTS2/MiniMax-Speech 等依赖离散 token 的主流路线完全不同
2. **FSQ 内部正则化瓶颈**: FSQ 不是量化层的预测目标,而是模型内部的可微瓶颈。TSLM 输出经 FSQ 强制量化 → 只保留 "可以活过量化" 的语义信息 → RALM 通过残差恢复声学细节。消融实验证明这种架构分离的归纳偏置比等量参数更有价值
3. **TSLM + RALM 残差分工**: h_final = FSQ(TSLM) + RALM, 两个模块各司其职但在同一梯度流中端到端训练,比多阶段管线更紧凑
4. **Voice Design (文本描述→声音)**: VoxCPM2 独创功能 — 用自然语言描述 (性别、年龄、语调、情感、语速) 生成全新声音,不需要参考音频。这在 TTS 产品中极为少见
5. **极致开源 + 工程完善**: Apache-2.0, pip install, CLI, WebUI, vLLM-Omni 官方支持, 社区生态 (VoxCPM.cpp/ONNX/ANE/Rust/ComfyUI) 极其丰富

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| OpenBMB/VoxCPM | **27.5k** | 全系列 TTS 模型, Apache-2.0, pip install |
| openbmb/VoxCPM2 | HF 模型 | 2B, 30 语言, 48kHz (HuggingFace + ModelScope) |
| openbmb/VoxCPM1.5 | HF 模型 | 0.6B, 中英, 44.1kHz |
| openbmb/VoxCPM-0.5B | HF 模型 | 0.5B, 中英, 16kHz (ICLR 2026) |
| a710128/nanovllm-voxcpm | 社区 | Nano-vLLM 高性能推理引擎 |
| vllm-project/vllm-omni | 官方支持 | vLLM 官方 omni-modal 扩展, VoxCPM2 原生支持 |
| VoxCPM.cpp / ONNX / ANE | 社区 | CPU/CUDA/Vulkan/ONNX/Apple 多平台推理 |

### 判断

- **优势**: **27.5k stars 是本 session 乃至整个行业调研中开源影响力最大的语音项目** (超过 CosyVoice 21.5k); ICLR 2026 接收验证学术价值; tokenizer-free 路线在技术上独树一帜; VoxCPM2 的 30 语言 + Voice Design + 48kHz 功能矩阵完整; 工程完善度极高 (pip install + CLI + vLLM + 多平台推理); 清华 THUHCSI + 面壁智能的联合保证了研究和工程的双重质量
- **短板**: **纯 TTS 模型,无语音理解/对话/Omni 能力** — 在 Speech LLM 方向上定位狭窄; 仅 1 篇学术论文 (VoxCPM ICLR 2026), VoxCPM2 技术报告尚未发布 ("Coming soon"); 无全双工对话研究; Hard case CER (8.87%) 仍高于 DiTAR (5.83%); BPE 路线在发音稳定性上可能不如 phoneme 路线
- **下一步推测**: VoxCPM2 技术报告即将发布; 可能扩展到语音理解 (VoxCPM + ASR → Speech LLM); 可能推出 VoxCPM-Omni; MiniCPM 系列 + VoxCPM 可能整合为统一多模态模型; 可能进一步扩大语言覆盖
- **关键观察**: VoxCPM 是 "小团队做大影响" 的典范。面壁智能 + 清华 THUHCSI 的团队规模远小于阿里/字节/腾讯,但凭借独特的 tokenizer-free 路线、极致的开源策略和工程完善度,在社区影响力上已超越多数大厂语音项目。**27.5k stars 证明了开源 TTS 的市场需求巨大**

---

## 团队4: 蚂蚁集团 / Inclusion AI (Ming-UniAudio)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 蚂蚁集团 (Ant Group) — Inclusion AI 品牌 |
| **团队名** | Inclusion AI (蚂蚁集团 AI 研究品牌, 原 AntAudio / Ant Research) |
| **GitHub org** | [inclusionAI](https://github.com/inclusionAI) (45 个仓库, Ming 系列); [AntAudio](https://github.com/AntAudio) |
| **HuggingFace** | inclusionAI 相关 (Ming 系列模型) |
| **核心人物** | **Jingdong Chen (陈景东)** — 蚂蚁 AI 首席科学家, 多篇 senior; **Jun Zhou** — 核心研究员; **Qingpei Guo** — Ming-Flash-Omni 通讯作者; **Chunxiang Jin** — Ming-UniAudio 通讯作者; **Canxiang Yan** — Ming-UniAudio 一作; **Ming Yang** — 核心 |
| **产品线** | 支付宝智能助手, 蚂蚁金融大模型, Ming 系列 (Omni/Flash-Omni/UniAudio), Ling 系列 (文本 LLM), Ring 系列 (推理模型), AWorld (Agent) |
| **开源态度** | **积极开源** — Ming 系列通过 inclusionAI GitHub 开源; Ling/Ring 系列开源; Ming-UniAudio 开源 (448 stars) |

### 论文时间线 (2024-2026)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2025.06 | **Ming-Omni: A Unified Multimodal Model for Perception and Generation** (2506.09344) — Inclusion AI | preprint | **统一多模态 Omni** |
| 2025.10 | **Ming-UniAudio: Speech LLM for Joint Understanding, Generation and Editing** (2511.05516) — Inclusion AI | preprint | **语音 LLM 旗舰** |
| 2025.10 | **Ming-Flash-Omni: Sparse, Unified Architecture** (2510.24821) — Inclusion AI | preprint (v3: 2026.03) | 稀疏 MoE Omni |
| 2026.xx | **Ling-2.5-1T / Ring-2.5-1T** — Inclusion AI | 开源发布 | 文本/推理基座 |
| 2025-2026 | **UI-Venus / DR-Venus / AWorld / cuLA / LLaDA2.X** — Inclusion AI | 多篇论文 | Agent/视觉/推理 (非语音) |

> **说明**: 蚂蚁集团通过 Inclusion AI 品牌在 2025 年密集发布了 Ming 系列多模态模型。Ming-UniAudio 是其语音 LLM 的核心工作,在连续统一表示方面有独特贡献。Ming-Omni 和 Ming-Flash-Omni 则是统一多模态 Omni 模型 (含语音感知和生成)。值得注意的是,Ming-UniAudio 的论文署名为 "Ant Group" 旗下机构,确认了蚂蚁集团的关联。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **语音编码器** | **MingTok-Audio** (1.35B 参数 VAE tokenizer): causal transformer encoder → VAE latent Gaussian (Zlatent, 低维) → Semantic Module (Whisper large-v3 init) → Zuni (高维, LLM 输入); 三阶段训练 (声学重建 → 语义蒸馏 → LLM 联合) |
| **LLM 骨干** | **16.8B MoE** (2.8B active parameters); Ming-Flash-Omni: Ling-Flash-2.0 (100B total, 6.1B active) |
| **语音解码器** | **Per-token flow matching head** — LLM hidden states → flow matching → Zlatent → MingTok-Audio decoder → 波形 |
| **对话策略** | 理解+生成+编辑统一框架; 自由形式语音编辑 (无需 timestamp/MFA); 语义编辑: CoT + [MASK]; Ming-Omni: 统一图文音视频感知生成 |
| **训练数据规模** | ~**39 万小时** (16kHz, 中英 1:1) [Ming-UniAudio]; 预训练 400K:800K 小时 (理解:生成) |
| **推理延迟** | 未公开具体数据 |
| **多语言** | 中英为主; MingTok-Audio 方言 ASR: 粤语 WER 5.51 (vs Kimi-Audio 41.49, **大幅领先**) |
| **情感/副语言** | 声学编辑支持降噪/变速/变调; 语义编辑支持方言转换; 自由形式指令驱动 |

### 架构演进

```
蚂蚁集团 AI 研究 (2020s) — 金融 AI / 风控 / NLP
    ↓ Inclusion AI 品牌建立
Ling 系列 (2024-2025) — 文本 LLM 基座 (Ling-2.5-1T)
    ↓ 多模态探索
Ming-Omni (2025.06) — 统一多模态感知+生成:
    ├── 图/文/音/视频多模态
    ├── Ling MoE 骨干 + modality-specific routers
    └── 语音感知 + 语音生成能力
    ↓ 语音专项
Ming-UniAudio (2025.10) — 语音 LLM 旗舰:
    ├── MingTok-Audio: VAE-based 连续统一 tokenizer (1.35B)
    ├── 16.8B MoE LLM (2.8B active)
    ├── 理解 + 生成 + 编辑三合一
    ├── 首创无 timestamp 自由形式语音编辑
    └── 中文 Seed-TTS WER 0.95% (SOTA)
    ↓ 效率升级
Ming-Flash-Omni (2025.10, v3: 2026.03) — 稀疏 MoE Omni:
    ├── 100B total / 6.1B active (ultra-sparse)
    ├── 计算效率大幅提升
    └── 视觉理解比肩 Gemini 2.5 Pro
    ↓ 开源
inclusionAI/Ming-UniAudio (448 stars) — 代码+模型
inclusionAI/Ming (656 stars) — Ming 系列整合
```

### 独特技术赌注

1. **连续统一表示 (Zuni/Zlatent 双层)**: Ming-UniAudio 的核心创新 — 用单一 VAE-based 连续 tokenizer 同时服务理解和生成,不依赖离散 token。Zuni (高维, LLM 输入) 和 Zlatent (低维, flow matching 用) 的双层设计解决了理解需高维语义、生成需低维高效的矛盾
2. **LLM 语义蒸馏入 Tokenizer**: MingTok-Audio 第三阶段训练中,用冻结 LLM 的 ASR CE loss 反向传播优化 tokenizer,使连续表示不仅保留声学还对齐语义。这在 continuous tokenizer 路线中是首创
3. **自由形式语音编辑**: 首个不需要 timestamp、不需要 MFA 对齐、纯自然语言指令驱动的语音编辑系统。涵盖语义编辑 (删/插/换) + 声学编辑 (降噪/变速/变调/方言转换)
4. **Semantic Module Freezing 策略**: 联合训练时冻结 tokenizer 中的语义模块,防止理解-生成优化方向冲突导致的 representation drift (AVG WER 4.35 vs 不冻结 6.86), 简单有效
5. **Ming 全家桶**: 从 Ming-Omni (统一多模态) → Ming-UniAudio (语音专项) → Ming-Flash-Omni (效率优化), 形成完整的多模态 Omni 技术栈

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| inclusionAI/Ming-UniAudio | 448 | 语音 LLM, 代码+模型权重开源 |
| inclusionAI/Ming | 656 | Ming 系列整合仓库 |
| inclusionAI/Ling-V2.5 | 20 | 文本 LLM 基座 |
| inclusionAI/Ring-V2.5 | 44 | 推理模型 |
| inclusionAI/LLaDA2.0-Uni | 758 | 扩散语言模型 |
| inclusionAI/AWorld | 1.2k | Agent 平台 |
| Ming-Omni / Ming-Flash-Omni | 论文 | 技术报告已发, 模型通过 Ming 仓库发布 |

### 判断

- **优势**: Ming-UniAudio 的连续统一表示是技术上最有深度的创新之一 — 解答了 "连续 VAE 能否同时服务理解和生成" 这一开放问题; 中文 TTS WER 0.95% (SOTA); 方言 ASR 大幅领先 (粤语 5.51 vs Kimi-Audio 41.49); 自由形式语音编辑是独创; Ming 全家桶覆盖 Omni + 语音专项 + 效率优化; 蚂蚁集团的金融场景提供独特的产品需求 (客服/风控/智能助手)
- **短板**: 开源影响力有限 (Ming-UniAudio 448 stars, 远低于 VoxCPM 27.5k 和 Kimi-Audio 4.6k); SIM 指标偏低 (未报告最终模型 SIM, 消融中参考值偏低); 编辑 baseline 缺失 (自由形式编辑无外部对比); 50Hz 帧率偏高 (长音频场景 token 序列长); 训练数据 39 万小时远小于 Kimi-Audio 1300 万小时; 无全双工对话研究; 品牌知名度 ("Inclusion AI") 在国际社区不如 MoonshotAI/OpenBMB
- **下一步推测**: Ming-Omni 可能升级加入端到端语音对话; Ming-UniAudio 可能发布更大规模版本; Ming-Flash-Omni 的稀疏 MoE 可能进一步优化; 可能在支付宝/蚂蚁金融场景落地; MingTok-Audio 可能独立开源作为通用 speech tokenizer
- **关键观察**: 蚂蚁集团在 Speech LLM 方向的技术深度被低估了。Ming-UniAudio 的连续统一表示 + 自由形式编辑在技术创新性上不逊于任何竞品,但缺乏像 VoxCPM (27.5k stars) 那样的开源影响力和社区运营能力。Inclusion AI 品牌的知名度也远不如 MoonshotAI 或 OpenBMB,导致其研究成果的 visibility 偏低

---

## 横向对比矩阵

| 维度 | 月之暗面 (Kimi-Audio) | MiniMax (MiniMax-Speech) | 面壁智能 (VoxCPM) | 蚂蚁集团 (Ming-UniAudio) |
|------|---------------------|------------------------|------------------|------------------------|
| **Speech LLM 论文数** | 1 (Kimi-Audio) | 1 (MiniMax-Speech, TTS) | 1 (VoxCPM, TTS, ICLR 2026) | 3 (UniAudio + Omni + Flash-Omni) |
| **端到端 Speech LLM** | Kimi-Audio-7B (理解+生成+对话) | 无 (纯 TTS) | 无 (纯 TTS) | Ming-UniAudio (理解+生成+编辑) |
| **Omni 模型** | 无独立 Omni (Kimi-Audio 含语音交互) | 无 | 无 | Ming-Omni + Ming-Flash-Omni |
| **全双工能力** | 无公开论文 | 无 | 无 | 无 |
| **TTS 质量** | 端到端生成 (非专门 TTS) | TTS Arena #1 | Seed-TTS EN-WER 1.85% (开源 SOTA) | 中文 Seed-TTS WER 0.95% (SOTA) |
| **语音理解** | AISHELL-1 WER 0.60 (SOTA) | 无 | 无 | ContextASR 8/12 项 SOTA |
| **语音编辑** | 无 | 无 | 无 | **首创** 自由形式编辑 |
| **Speech Codec/Tokenizer** | VQ 语义 token + Whisper 连续特征 | Learnable Speaker Encoder + Flow-VAE | **Tokenizer-free** (FSQ 内部瓶颈) | MingTok-Audio (1.35B VAE tokenizer) |
| **LLM 骨干** | Qwen 2.5-7B | 自研 Transformer (未公开) | MiniCPM-4 (0.5B-2B) | 16.8B MoE (2.8B active) |
| **训练数据规模** | **1300 万小时** | "数百万小时" (未公开) | **200 万小时+** | 39 万小时 |
| **多语言** | 中英 | **32 语言** | **30 语言** | 中英 + 方言 |
| **开源程度** | 高 (权重+代码+评估+微调) | 低 (商业 API 为主) | **极高** (Apache-2.0, 全生态) | 中 (代码+模型开源) |
| **GitHub Stars** | 4.6k (Kimi-Audio) | 无独立语音仓库 | **27.5k** (VoxCPM) | 448 (Ming-UniAudio) |
| **核心人物** | Xu Tan (前微软) | Bowen Zhang | 周逸轩 + 吴志勇 (清华) | 陈景东 (蚂蚁首席科学家) |
| **技术路线** | 混合输入 Audio LLM | 商业 TTS 产品 | Tokenizer-free TTS | 连续统一表示 Speech LLM |
| **产品落地** | Kimi (ChatBot 语音版) | Talkie (AI 角色对话) | 开源社区 (vLLM-Omni) | 支付宝/蚂蚁金融 |
| **技术透明度** | 高 (tech report 详细) | 中 (论文有但产品细节少) | **高** (论文详细 + 消融充分) | **高** (论文极详细) |

## 关键发现

1. **Speech LLM vs TTS 的路线分野明显**: 本 session 4 家中,只有月之暗面 (Kimi-Audio) 和蚂蚁 (Ming-UniAudio) 在做端到端 Speech LLM (语音理解+生成),面壁 (VoxCPM) 和 MiniMax (MiniMax-Speech) 严格来说是 TTS 模型。但 TTS 模型的开源影响力 (VoxCPM 27.5k) 远超 Speech LLM (Kimi-Audio 4.6k),说明市场对高质量开源 TTS 的需求更直接。

2. **Tokenizer-free 路线的崛起**: VoxCPM 的 tokenizer-free 架构和 Ming-UniAudio 的连续统一表示都挑战了主流的 discrete token 路线。两者的技术思路不同 (VoxCPM 用 FSQ 作为内部正则化, Ming-UniAudio 用 VAE + 语义蒸馏),但都指向同一个方向: **离散 token 可能不是语音表示的最优解**。这与 Session 1-2 中的 CosyVoice/Seed-TTS 等 discrete token 路线形成对比。

3. **全双工对话是 4 家共同的空白**: 本 session 4 家团队**均无全双工对话论文**。与 Session 2 中腾讯 Covo-Audio-FD (99.7% turn-taking) 形成鲜明对比。这可能反映了: (a) 全双工是更大厂的战场; (b) 新锐团队优先解决基础能力 (理解/生成) 再做交互; (c) 全双工的工程复杂度要求更大团队。

4. **开源策略与影响力高度正相关**: VoxCPM (Apache-2.0, pip install, vLLM, 27.5k stars) > Kimi-Audio (权重+代码+评估, 4.6k stars) > Ming-UniAudio (代码+模型, 448 stars) > MiniMax-Speech (商业 API, 无独立仓库)。面壁智能和月之暗面的开源策略远比 MiniMax 激进,社区影响力也相应更大。

5. **"清华系" vs "大厂系" 的不同打法**: 面壁智能 (清华 THUHCSI + 面壁) 走的是"学术创新 + 极致开源"路线 (ICLR 2026 + 27.5k stars); 月之暗面 (Xu Tan 等微软系) 走"技术旗舰 + 评估标准化"路线; 蚂蚁 (Inclusion AI) 走"全家桶 Omni"路线; MiniMax 走"产品商业化"路线。在 Speech LLM 方向上,学术创新和开源影响力 (面壁 > 月之暗面) 与产品化能力 (MiniMax > 蚂蚁) 存在明显 trade-off。
