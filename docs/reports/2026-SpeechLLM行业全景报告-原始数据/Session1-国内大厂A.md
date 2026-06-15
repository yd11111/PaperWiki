# Session 1: 国内大厂 A — 阿里通义 / 字节豆包 / 智谱 / 阶跃星辰

> **报告范围**: Speech LLM / Omni / 全双工对话
> **调研日期**: 2026-06-08
> **数据来源**: arXiv API, GitHub API, vault 已有论文笔记, 公开技术报告
> **5 层搜索覆盖**: Layer 1 (GitHub/HuggingFace org) ✓ | Layer 2 (核心人搜索) 部分 | Layer 3 (arXiv affiliation) ✓ | Layer 4 (产品/竞赛) 部分 | Layer 5 (引用网络) ✓ (通过 vault 笔记)

---

## 1. 阿里通义 — FunAudioLLM 团队

### 1.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | 阿里巴巴集团 |
| 团队 | 通义语音团队 (Tongyi Speech Team, Tongyi Lab) |
| GitHub | [github.com/FunAudioLLM](https://github.com/FunAudioLLM) (13 repos) |
| HuggingFace | [huggingface.co/FunAudioLLM](https://huggingface.co/FunAudioLLM) |
| 核心人物 | Shiliang Zhang (张士良), Zhifu Gao, Zhihao Du, Qian Chen, Yafeng Chen; GitHub org 唯一公开成员: @LauraGPT |
| 产品线 | 通义听悟 (语音理解), 通义千问 (语音交互), CosyVoice (开源 TTS), SenseVoice (开源 ASR), Fun-Audio-Chat (语音对话) |
| 定位 | 理解+生成双基座 → 端到端语音对话; 开源生态最完善的国内 Speech LLM 团队 |

### 1.2 论文时间线 (2024-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2024.07 | FunAudioLLM | 2407.04051 | SenseVoice + CosyVoice 双基座框架 | **基础设施**: 理解+生成组件,支撑语音交互 pipeline |
| 2024.07 | CosyVoice | (含在 FunAudioLLM) | S3 tokenizer + OT-CFM TTS | 语音生成基座,后续所有 Speech LLM 的 decoder |
| 2024.07 | SenseVoice | (含在 FunAudioLLM) | NAR multi-task ASR+SER+AED+LID | 语音理解基座,RTF 0.007 |
| 2024.12 | CosyVoice 2 | 2412.10117 | Streaming TTS, finite scalar quantization | 支撑低延迟语音对话的关键 |
| 2025.01 | MinMo | 2501.06282 | 多模态 LLM 语音交互 | **核心 Speech LLM**: 端到端语音理解+文本生成 |
| 2025.05 | CosyVoice 3 | 2505.17589 | 100 万小时 scaling + DiffRO RL | 9 语种 TTS,成为 Qwen3.5-Omni Talker 的基础 |
| 2025.06 | DrVoice | 2506.09349 | Parallel speech-text 模型, DRSR 架构 | **核心 Speech LLM**: Fun-Audio-Chat 的前身 |
| 2025.12 | Fun-Audio-Chat | 2512.20156 | DRSR + Core-Cocktail + Multi-Task DPO + 全双工 | **旗舰 Speech LLM**: 8B/30B-A3B,全双工 |
| 2025 | Fun-ASR (v3) | - | 31 语言 ASR,千万小时训练 | 语音理解能力升级 |
| 2025 | ThinkSound | NeurIPS 2025 | CoT 引导的统一音频生成 | 音频生成扩展 |

### 1.3 技术栈全景

| 维度 | FunAudioLLM 技术栈 |
|------|-------------------|
| **语音编码器** | Whisper-Large-v3 (frozen) + Adapter; S3Tokenizer (CosyVoice 3, FSQ, 25Hz, frozen) |
| **LLM 骨干** | Qwen3-30B-A3B (MoE) 或 Qwen3-VL-8B (dense); Fun-Audio-Chat 中 LLM 处理 5Hz grouped tokens |
| **语音解码器** | CosyVoice 3 Speech Detokenizer (Flow Matching + HiFi-GAN), frozen |
| **对话策略** | Parallel Joint Speech-Text (同时输出 text + speech); 全双工变体支持 parallel input stream |
| **训练数据规模** | CosyVoice 3: 100 万小时; Fun-Audio-Chat: 百万小时级多样语音 |
| **推理延迟** | LLM 帧率 5Hz (vs 常见 12.5-25Hz); ~50% GPU hours 减少; UTMOS 4.37 |
| **多语言** | CosyVoice 3: 9 语种; SenseVoice: 50+ 语种; Fun-ASR: 31 语种 |
| **情感/副语言** | Multi-Task DPO 的 voice empathy 维度; S3 tokenizer 天然编码副语言 |

### 1.4 架构演进

```
SenseVoice (NAR ASR+SER, 2024.07)
    ↓ 理解基座
CosyVoice (S3 tokenizer + CFM, 2024.07) → CosyVoice 2 (Streaming, FSQ, 2024.12) → CosyVoice 3 (Scaling+DiffRO, 2025.05)
    ↓ 生成基座                                                                                  ↓
FunAudioLLM Pipeline (ASR→LLM→TTS, 2024.07)                                              S3Tokenizer 供给
    ↓                                                                                           ↓
MinMo (Multimodal LLM + voice interaction, 2025.01) ─────────────────────────────→ Fun-Audio-Chat (Parallel LALM, 2025.12)
    ↓                                                                                  ↓ 核心创新: DRSR (5Hz LLM + 25Hz SRH)
DrVoice (DRSR 架构原型, 2025.06)  ─────────────────────────────────────────────→  Fun-Audio-Chat-Duplex (全双工, 2025.12)
```

### 1.5 独特技术赌注

1. **DRSR (Dual-Resolution Speech Representations)**: 将 LLM 帧率降至 5Hz,用 SRH (Speech Refined Head) 在 25Hz 恢复精度。这是"规划与执行分离"的思路 — LLM 做粗粒度语义规划,SRH 做细粒度声学细化。
2. **Core-Cocktail Training (模型合并防遗忘)**: 高 LR 微调后与原始 LLM 做 α=0.5 加权合并,再低 LR 精调。比 EWC/replay buffer 更简单。
3. **Post-training-only 路线**: 不做大规模 audio-text 预训练,仅靠后训练 pipeline (pre-alignment → SFT → DPO) 达到竞争力。显著降低训练成本。
4. **S3 Supervised Semantic Tokenizer**: 在 ASR encoder 中间层插入 VQ,用 ASR loss 监督,tokens 天然携带语义+副语言信息。

### 1.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| CosyVoice | 21,501 | 多语言 TTS,支持推理+训练+部署 |
| SenseVoice | 8,477 | 多语言 ASR+SER+AED,50+ 语言 |
| ThinkSound | 1,365 | NeurIPS 2025, CoT 音频生成 |
| FunMusic | 1,354 | 音乐/歌曲/音频生成工具 |
| Fun-ASR | 1,230 | 31 语言端到端 ASR |
| Fun-Audio-Chat | 961 | 语音对话 LALM |
| FunCineForge | 430 | 影视音频 |
| FunAudioLLM-APP | 384 | 应用 demo |
| CV3-Eval | 186 | TTS 评估 benchmark |
| FunResearch | 34 | 研究论文代码 |

### 1.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) 开源生态最完善: CosyVoice 21K stars 是国内语音开源项目之最; (2) 技术栈完整: 从 tokenizer 到 ASR/TTS 到 LALM 到全双工,全链路自研; (3) DRSR 效率优势: LLM 5Hz 处理使计算成本降低约 50%; (4) 产品化经验: 通义听悟等产品的工程反馈 |
| **短板** | (1) Fun-Audio-Chat 只做 post-training,缺乏大规模 speech-text 预训练,理论上限可能受限; (2) 多轮对话记忆丢失、语音指令跟随不稳定 (自述); (3) 与 Qwen 团队的 Omni 系列存在内部竞争/协同关系不清晰; (4) 缺少标准 MOS 人工评估 |
| **下一步推测** | (1) Fun-Audio-Chat 可能与 Qwen-Omni 深度融合; (2) 引入大规模预训练 (目前是短板); (3) CosyVoice 4 / S3Tokenizer v2 持续迭代; (4) 更多语言覆盖和产品化落地 |

---

## 2. 阿里 Qwen 团队 — Qwen-Audio / Qwen-Omni

### 2.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | 阿里巴巴集团 |
| 团队 | Qwen 团队 (通义千问); 与通义语音团队有技术协同但独立运作 |
| GitHub | [github.com/QwenLM](https://github.com/QwenLM) |
| HuggingFace | [huggingface.co/Qwen](https://huggingface.co/Qwen) |
| 核心人物 | Jin Xu, Zhifang Guo, Yunfei Chu, Junyang Lin; Qwen 大团队 |
| 产品线 | 通义千问 (语音交互 API), Qwen-Omni 系列 (全模态大模型) |
| 定位 | 全模态统一大模型路线,从 text LLM 扩展到 audio/vision/video |

### 2.2 论文时间线 (2023-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2023.11 | Qwen-Audio | 2311.07919 | 首个通用音频理解大模型 | **起点**: Audio understanding LLM |
| 2024.07 | Qwen2-Audio | 2407.10759 | 升级版音频理解 | 多语言音频理解能力增强 |
| 2025.03 | Qwen2.5-Omni | 2503.20215 | Thinker-Talker 架构,text+speech 同时生成 | **重要里程碑**: 首个 Qwen 端到端语音对话 |
| 2025.09 | Qwen3-Omni | 2509.17765 | 30B-A3B MoE, dual-channel Talker | 多模态对话能力增强 |
| 2026.01 | Qwen3-ASR | 2601.21337 | 独立 ASR 模型 | 语音理解组件 |
| 2026.01 | Qwen3-TTS | 2601.15621 | 独立 TTS 模型 | 语音生成组件 |
| 2026.04 | Qwen3.5-Omni | 2604.15804 | ARIA 对齐 + Hybrid MoE + 29 语言 TTS | **旗舰**: 全模态 SOTA, 215 个 benchmark |

### 2.3 技术栈全景

| 维度 | Qwen-Omni 技术栈 |
|------|-----------------|
| **语音编码器** | AuT (Audio Transformer, 从零训练, 40M 小时数据, 6.25Hz) |
| **LLM 骨干** | Thinker: Hybrid-Attention MoE (含 Gated Delta Net), 从 Qwen3.5 初始化; 256K context |
| **语音解码器** | Talker: Hybrid MoE → RVQ tokens → MTP (Multi-Token Prediction) → Code2Wav (causal ConvNet); 帧级流式合成 |
| **对话策略** | Thinker-Talker 二层架构; ARIA (Adaptive Rate Interleave Alignment) 替代 dual-channel |
| **训练数据规模** | 预训练 ~4T tokens (text+audio+image+video); Talker 20M+ 小时 |
| **推理延迟** | Plus: 首包延迟 435ms (audio, 1 并发), 8 并发升至 955ms |
| **多语言** | ASR: 113 语言; TTS: 36 语言 (29 语言有评估数据, 22/29 最优) |
| **情感/副语言** | Talker system prompt 支持 voice cloning; RLHF (DPO + GSPO) 对齐 |

### 2.4 架构演进

```
Qwen-Audio (Audio Understanding, 2023.11) → Qwen2-Audio (Upgraded, 2024.07)
    ↓                                                ↓
    └────── 理解分支 ──────────────────────────────────┘
                                                      ↓
                                            Qwen2.5-Omni (Thinker-Talker 首版, 2025.03)
                                                      ↓ TMRoPE + dual-channel
                                            Qwen3-Omni (30B-A3B MoE, 2025.09)
                                                      ↓ 问题: dual-channel 对齐不稳定
                                            Qwen3.5-Omni (ARIA + Hybrid MoE + 显式时间戳, 2026.04)
                                                      
并行独立组件:
    Qwen3-ASR (2026.01) ←── AuT encoder 的独立部署版
    Qwen3-TTS (2026.01) ←── Talker 的独立部署版
```

### 2.5 独特技术赌注

1. **ARIA (Adaptive Rate Interleave Alignment)**: 约束任意前缀的 speech/text ratio 不超过全局 ratio,动态处理跨语言编码效率差异。形式简洁但有效。
2. **Thinker-Talker 分层架构**: Thinker 负责理解+推理+文本生成, Talker 以 Thinker hidden state 为条件生成语音。避免信息丢失 (vs 仅传 text tokens)。
3. **On-Policy Distillation (OPD)**: 用 text-conditioned 的强回答教 audio-conditioned 的弱回答,减少跨模态质量差距。
4. **全模态统一**: text/image/audio/video 统一处理,是国内最接近 GPT-4o 全模态能力的模型。

### 2.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| Qwen2.5-Omni | 4,017 | 首版 Thinker-Talker |
| Qwen3-Omni | 3,818 | MoE 版本 |
| Qwen2-Audio | 2,074 | 音频理解 LLM |
| Qwen-Audio | 1,899 | 初代音频理解 |

**注意**: Qwen3.5-Omni **仅 API 可用,模型权重未开源**。

### 2.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) 全模态能力最强: text/image/audio/video 统一,MMAU 82.2 超 Gemini-3.1 Pro; (2) 多语言覆盖最广: 113 语言 ASR + 36 语言 TTS; (3) 训练资源最充裕: 4T tokens 预训练 + 20M+ 小时 Talker 训练; (4) ARIA 是目前最优雅的 speech-text streaming 对齐方案 |
| **短板** | (1) 最新旗舰 Qwen3.5-Omni 未开源; (2) 首包延迟 435ms 在高并发下翻倍; (3) 缺少消融实验,各组件独立贡献不清晰; (4) 无 MOS 评估; (5) 与 FunAudioLLM 团队的分工和技术复用关系不透明 |
| **下一步推测** | (1) Qwen4-Omni 可能开源权重; (2) 延迟优化 (可能引入 speculative decoding); (3) 全双工能力增强 (目前 Qwen-Omni 系列不支持全双工); (4) 与 FunAudioLLM 更深度融合或明确分工 |

### 2.8 阿里双团队关系说明

阿里在 Speech LLM 领域存在两个独立但有协同的团队:

- **FunAudioLLM (通义语音)**: 专注语音技术,从 ASR/TTS 基座向上构建语音对话系统 (Fun-Audio-Chat)。技术路线: post-training + parallel LALM。
- **Qwen 团队**: 从 LLM 角度出发扩展全模态能力 (Qwen-Omni 系列)。技术路线: 大规模预训练 + Thinker-Talker。

**交叉点**: CosyVoice 3 的 S3Tokenizer 被 Fun-Audio-Chat 使用; CosyVoice 2/3 也被 Qwen-Omni 的 Talker 训练引用; MinMo (2025.01) 似乎是两团队的协作产出; Qwen3-ASR 生成的 40M 小时数据被 Qwen3.5-Omni 的 AuT 训练使用。

---

## 3. 字节豆包 — Seed-ASR / SALMONN / 豆包语音

### 3.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | 字节跳动 (ByteDance) |
| 团队 | 字节跳动 AI Lab / Seed 团队; 与清华大学合作密切 |
| GitHub | [github.com/bytedance](https://github.com/bytedance) (无独立 Speech LLM org); SALMONN: bytedance/SALMONN |
| HuggingFace | ByteDance (公开模型有限) |
| 核心人物 | Changli Tang, Wenyi Yu, Guangzhi Sun (SALMONN); Ye Bai, Jitong Chen (Seed-ASR); Philip Anastassiou, Jitong Chen (Seed-TTS); Zhiliang Peng, Jianwei Yu (VibeVoice) |
| 产品线 | 豆包 (Doubao) 语音交互, 火山引擎语音 API, Seed-TTS/ASR 服务 |
| 定位 | 产品驱动型: 豆包产品的语音能力是核心; 学术线以 SALMONN 系列为主 |

### 3.2 论文时间线 (2023-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2023.10 | SALMONN | 2310.13289 | 通用音频理解 LLM (Whisper+BEATs→Vicuna) | **起点**: 音频理解 LLM |
| 2024.06 | Seed-TTS | 2406.02430 | 高质量多功能 TTS 家族 (含 Seed-TTS eval) | 语音生成基座 |
| 2024.06 | video-SALMONN | 2406.15704 | 音视频理解 LLM | 多模态扩展 |
| 2024.07 | Seed-ASR | 2407.04675 | LLM-based ASR,上下文感知 | 语音理解基座 |
| 2024.09 | Speech Quality Eval | 2409.16644 | 语音质量自动评估 | 评估工具 |
| 2024.11 | SALMONN-omni v1 | 2411.18138 | Codec-free LLM 全双工对话 | **核心 Speech LLM**: 首次全双工 |
| 2025.02 | video-SALMONN-o1 | 2502.11775 | 推理增强音视频 LLM | 推理能力扩展 |
| 2025.05 | SALMONN-omni v2 | 2505.17060 | 独立式全双工 Speech LLM (无 codec 注入) | **旗舰 Speech LLM**: 无辅助组件全双工 |
| 2025.06 | video-SALMONN 2 | 2506.15220 | Caption 增强音视频 LLM | 多模态增强 |
| 2025.08 | VibeVoice | 2508.19205 | Next-token diffusion 长语音合成 | 新一代 TTS |
| 2025.10 | video-SALMONN S | 2510.11129 | 流式音视频 LLM | 流式能力 |
| 2025.11 | SALMONN-Guard | 2511.10222 | 音频内容安全 | 安全防护 |
| 2026.01 | VibeVoice-ASR | 2601.18184 | 长音频语音理解框架 | ASR 升级 |

### 3.3 技术栈全景

| 维度 | 字节豆包技术栈 |
|------|--------------|
| **语音编码器** | SALMONN: Whisper + BEATs (dual encoder); Seed-ASR: LLM-based encoder; VibeVoice-ASR: 自研 |
| **LLM 骨干** | SALMONN-omni: 基于 Vicuna/LLaMA 系列; 豆包产品: 内部大模型 |
| **语音解码器** | SALMONN-omni: Codec-free (不依赖 speech codec); Seed-TTS: 自回归 + diffusion; VibeVoice: next-token diffusion |
| **对话策略** | SALMONN-omni: 无辅助组件 (无 VAD/interrupter/状态预测) 的独立全双工; 豆包产品: 工程化 pipeline |
| **训练数据规模** | Seed-TTS: 未公开但规模极大; SALMONN: 基于公开数据 + 内部数据 |
| **推理延迟** | 豆包产品: 实测约 200-400ms 首包; SALMONN-omni: 未公开具体延迟 |
| **多语言** | Seed-TTS: 未公开; Seed-ASR: 多语言; VibeVoice: 多说话人长语音 |
| **情感/副语言** | Seed-TTS: 高度自然; SALMONN: 情感识别; 豆包产品: 支持多种音色和情感 |

### 3.4 架构演进

```
SALMONN (Audio Understanding LLM, 2023.10)
    ↓ Whisper+BEATs 双编码器
video-SALMONN (Audio-Visual, 2024.06) → video-SALMONN-o1 (Reasoning, 2025.02) → video-SALMONN 2 (2025.06) → video-SALMONN S (Streaming, 2025.10)
    ↓ 理解分支
SALMONN-omni v1 (Codec-free 全双工, 2024.11)
    ↓ 去除辅助组件
SALMONN-omni v2 (独立式全双工, 2025.05) ←── 不依赖 VAD/interrupter/状态预测器

并行生成线:
Seed-TTS (2024.06) ─── 产品化 → 豆包语音
Seed-ASR (2024.07) ─── 产品化 → 豆包语音 / 火山引擎

新一代:
VibeVoice (Next-token Diffusion TTS, 2025.08) → VibeVoice-ASR (长音频理解, 2026.01)
```

### 3.5 独特技术赌注

1. **Codec-free 全双工**: SALMONN-omni 不使用 speech codec 注入,不依赖 VAD/interrupter 等辅助组件,是一种更端到端的全双工方案。
2. **Next-token Diffusion (VibeVoice)**: 用 diffusion 替代离散 token 预测,在连续隐空间中自回归生成,理论上信息保留更完整。
3. **Dual-encoder 理解 (SALMONN)**: Whisper + BEATs 双编码器分别捕获语音和非语音音频信息。
4. **Seed-TTS Eval**: 成为行业标准 TTS 评估 benchmark (被 GLM-4-Voice, Step-Audio, CosyVoice 等广泛使用)。

### 3.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| bytedance/SALMONN | 1,443 | SALMONN 系列全家桶 |
| bytedance/video-SALMONN-2 | 195 | 音视频 LLM |
| bytedance/Make-An-Audio-2 | 197 | 文本驱动音频生成 |
| bytedance/neurst | 307 | 神经语音翻译 |

**注意**:
- Seed-TTS **未开源** (仅提供 API 服务和 Seed-TTS-Eval benchmark)
- Seed-ASR **未开源** (仅通过火山引擎 API 提供)
- SALMONN-omni **未开源** (仅论文)
- VibeVoice **未开源** (仅论文)
- 豆包语音底层模型 **未开源**

### 3.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) 产品化最成熟: 豆包是国内用户量最大的 AI 助手之一,语音交互是核心卖点; (2) Seed-TTS 质量极高: 在自然度方面可能是国内最强; (3) SALMONN 学术影响力大: 1443 stars,被广泛引用; (4) Seed-TTS-Eval 成为行业标准 benchmark; (5) SALMONN-omni 的 codec-free 全双工是独特技术路线 |
| **短板** | (1) **开源程度最低**: 核心模型 (Seed-TTS, Seed-ASR, SALMONN-omni, VibeVoice) 均未开源; (2) 学术线 (SALMONN) 和产品线 (Seed/豆包) 的关系不透明; (3) 无统一的开源 Speech LLM 框架; (4) SALMONN-omni 的具体延迟、部署方案未公开 |
| **下一步推测** | (1) VibeVoice 可能成为下一代 TTS 基座 (替代 Seed-TTS); (2) SALMONN-omni 可能整合 VibeVoice 实现更自然的全双工; (3) 豆包产品可能推出全双工语音对话; (4) Seed-ASR 2.0 可能进一步升级 |

---

## 4. 智谱 — GLM-4-Voice / GLM-TTS

### 4.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | 智谱 AI (Zhipu AI) / 清华大学 THUDM |
| 团队 | 智谱 AI 语音团队; 源自清华 KEG 实验室 |
| GitHub | [github.com/zai-org](https://github.com/zai-org) (原 THUDM, 已迁移) |
| HuggingFace | THUDM / zai-org |
| 核心人物 | Aohan Zeng (曾奥涵), Zhengxiao Du (杜正霄), Mingdao Liu, Jie Tang (唐杰), Yuxiao Dong (董予巍) |
| 产品线 | 智谱清言 (GLM 语音交互), GLM-4-Voice 开源模型, GLM-TTS 开源模型 |
| 定位 | 学术转产品型: 清华 THUDM 基础研究 → 智谱产品化; 端到端 spoken chatbot 路线 |

### 4.2 论文时间线 (2024-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2024.12 | GLM-4-Voice | 2412.02612 | 合成交替数据 + Streaming Thoughts + VQ-Whisper tokenizer | **核心 Speech LLM**: 首个通过合成 interleaved data 实现 1T token 预训练的 spoken chatbot |
| 2025.12 | GLM-TTS | 2512.14291 | GRPO RL 对齐 + 25Hz tokenizer + Pitch Estimator | TTS 独立组件升级 |

### 4.3 技术栈全景

| 维度 | 智谱技术栈 |
|------|----------|
| **语音编码器** | VQ-Whisper: Whisper-large-v3 中间插入 pooling + VQ, 12.5Hz 单码本 (16384 entries, 175bps); block causal attention 支持 streaming |
| **LLM 骨干** | GLM-4-9B-Base (继续预训练); 1T tokens (455B interleaved speech-text + 31B 无监督 speech + 11B ASR/TTS) |
| **语音解码器** | CosyVoice 架构 (speech token encoder + CFM + HiFi-GAN); streaming 适配 (truncated audio fine-tuning, b=0.8s) |
| **对话策略** | Streaming Thoughts: 13 text tokens + 26 speech tokens 交替输出; 首段语音仅需 23 个 LLM decoding 步 |
| **训练数据规模** | 预训练 1T tokens; 合成 interleaved data 455B speech + 279B text; 无监督 31B; ASR/TTS 14.5B |
| **推理延迟** | 首段语音: T_speech_tokenize + T_prefill + T_decode(23 tokens) + T_speech_decode(10 tokens) |
| **多语言** | 仅中英双语 |
| **情感/副语言** | SFT 阶段语音风格控制数据; 语速/情感/方言可控 |

### 4.4 架构演进

```
GLM-4-9B-Base (Text LLM)
    ↓ 扩展词表 + 1T tokens 继续预训练
GLM-4-Voice (端到端 spoken chatbot, 2024.12)
    ├── VQ-Whisper tokenizer (12.5Hz, 175bps)
    ├── Streaming Thoughts (text:speech = 13:26 交替解码)
    └── CosyVoice decoder (streaming 适配)
    ↓ tokenizer 升级 + RL 对齐
GLM-TTS (25Hz, 32K 词表, Pitch Estimator + GRPO, 2025.12) ←── TTS 组件独立升级
```

### 4.5 独特技术赌注

1. **合成交替数据 (Synthetic Interleaved Data)**: 用 text-to-token model 将文本预训练语料转为 speech-text 交替序列,绕过 speech-text 平行语料瓶颈。使 1T token 预训练成为可能。
2. **Streaming Thoughts**: 固定比例 (13 text : 26 speech) 交替输出 text 和 speech tokens,不修改模型架构即控制首 token 延迟。
3. **超低比特率 semantic tokenizer**: 175bps 是已知最低比特率的实用 speech tokenizer (vs SpeechTokenizer 1.5K bps, Moshi 1.1K bps)。
4. **分离 loss 训练**: SFT 阶段对 text 和 speech output 分别 mask loss 独立训练,解决学习速度不对称。

### 4.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| zai-org/GLM-4-Voice | 3,184 | 端到端中英语音对话模型 |
| zai-org/GLM-TTS | 1,017 | 可控情感表达零样本 TTS |

### 4.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) 数据工程创新最突出: 合成 interleaved data 是解决 speech-text 平行语料瓶颈的关键突破; (2) 单码本路线成功验证: 175bps + CFM decoder 证明极简 tokenizer 的可行性; (3) Streaming Thoughts 推理模板简洁实用; (4) 模型完全开源 (GLM-4-Voice + GLM-TTS) |
| **短板** | (1) **仅支持中英双语**: 多语言能力严重不足; (2) **不支持全双工**: 与 Moshi/SALMONN-omni 相比是功能缺失; (3) Tokenizer 重建质量有限 (WER 8.43); (4) SFT 数据不透明; (5) 团队规模和资源相比阿里/字节有差距; (6) 论文发布后迭代速度较慢 (2024.12 → 2025.12 仅出 GLM-TTS) |
| **下一步推测** | (1) GLM-5-Voice 可能支持更多语言和全双工; (2) tokenizer 帧率可能从 12.5Hz 提升至 25Hz+ (GLM-TTS 已做); (3) 视觉模态集成 (对标 Qwen-Omni); (4) 产品端智谱清言的语音能力增强 |

---

## 5. 阶跃星辰 — Step-Audio

### 5.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | 阶跃星辰 (StepFun) |
| 团队 | StepFun Audio Team |
| GitHub | [github.com/stepfun-ai](https://github.com/stepfun-ai) |
| HuggingFace | [huggingface.co/stepfun-ai](https://huggingface.co/stepfun-ai) |
| 核心人物 | Fei Tian (田飞), Xiangyu Tony Zhang, Yuxin Zhang, Boyong Wu, Chao Yan; 团队规模较大 |
| 产品线 | 跃问 (语音交互助手), Step-Audio 系列开源模型, Claw (语音技能平台) |
| 定位 | 最激进的 Speech LLM 创业公司; 从 130B 参数开始,快速迭代至统一基座 |

### 5.2 论文时间线 (2025-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2025.02 | Step-Audio | 2502.11946 | 130B AQTA + 3B speech decoder + dual-codebook tokenizer | **V1**: 首个 130B 开源 SpeechLM |
| 2025.06 | Step-Audio-AQAA | 2506.08967 | 端到端表达式 Audio Language Model | **端到端升级**: 不经 text 中间表示 |
| 2025.07 | Step-Audio 2 | 2507.16632 | 端到端多模态 LLM, 工业级 | **V2**: 端到端架构 |
| 2025.11 | Step-Audio-EditX | 2511.03601 | 3B 音频编辑模型, RL 训练 | 音频编辑能力 |
| 2025.11 | Step-Audio-R1 | 2511.15848 | 音频推理模型 | 音频推理扩展 |
| 2026.04 | Step-Audio-R1.5 | 2604.25719 | 推理增强 | 推理能力升级 |
| 2026.05 | StepAudio 2.5 | 2605.23463 | 统一 MoE 基座 + ASR/TTS/Realtime 三方向特化 | **旗舰**: 统一基座,AISHELL-1 CER 0.71 |

### 5.3 技术栈全景

| 维度 | Step-Audio 技术栈 (V1 → 2.5 演进) |
|------|--------------------------------|
| **语音编码器** | V1: Dual-codebook (Paraformer linguistic 16.7Hz + CosyVoice semantic 25Hz, 2:3 交错); V2.5: Frozen Audio Encoder + Adaptor |
| **LLM 骨干** | V1: Step-1 130B; V2.5: MoE backbone (从 textual MoE LLM 初始化, 参数量未公开); 2.2T tokens 预训练 |
| **语音解码器** | V1: 3B Speech Decoder (LM + flow matching + vocoder); V2.5: 三方向特化 (ASR/TTS/Realtime) |
| **对话策略** | V1: AQTA (Audio Question, Text Answer) + 外接 TTS; V2.5: 统一基座 + directional inference |
| **训练数据规模** | V1: 1.2T + 800B tokens 预训练; V2.5: 2.2T tokens + 800B+800B speech+text main stage |
| **推理延迟** | V1: Speculative response ~500ms 减少; V2.5: ASR RTF 0.0053 (最快); Realtime 评估 SOTA |
| **多语言** | 中英为主 + 部分方言 |
| **情感/副语言** | V1: Instruction tags (descriptive + comparative 五级控制); V2.5: 百万级 persona matrix; Generative Reward Model |

### 5.4 架构演进

```
Step-Audio V1 (130B AQTA + 3B TTS, 2025.02)
    ├── Dual-codebook tokenizer (Paraformer + CosyVoice)
    ├── Generative Data Engine (LLM rewrite → clone → Audio-Edit)
    ├── Speculative Response Generation (~40% 命中)
    └── RLHF + Anti-deaf-hacking
    ↓ 端到端化
Step-Audio-AQAA (端到端表达式, 2025.06)
    ↓
Step-Audio 2 (端到端多模态, 2025.07)
    ↓ 统一基座
StepAudio 2.5 (统一 MoE 基座, 2026.05)
    ├── ASR 方向: MTP-5 verifiable decoding (RTF 0.0053)
    ├── TTS 方向: Pure NTP (无 encoder-adapter) + Generative Reward Model + RLHF
    └── Realtime 方向: Progressive SFT (3 阶段) + 百万级 persona matrix + RLHF
    
并行:
Step-Audio-EditX (3B 音频编辑, RL, 2025.11)
Step-Audio-R1 (音频推理, 2025.11) → Step-Audio-R1.5 (2026.04)
StepAudio-Skills (Claw 技能, ongoing)
```

### 5.5 独特技术赌注

1. **Task Specialization as Directional Inference**: "一旦 text 和 audio 共享良好表征空间,下游任务差异不是架构差异而是操作模式差异"。统一基座 + 方向性 post-training。
2. **MTP-5 for ASR**: 5 个 lookahead branch + autoregressive verification,利用语音信号的 grounding 属性实现高效多 token 投机解码。RTF 0.0053。
3. **Generative Reward Model (GRM)**: 替代标量 reward 的生成式评估,捕获 TTS 的多维质量。
4. **百万级 Persona Matrix**: 10K 种子 persona → 算法 fission → 百万级 persona-dialogue pairs,工业化 SFT 数据生产。
5. **Anti-deaf-hacking RLHF**: 发现并解决 speech LLM 特有的 reward hacking 问题。

### 5.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| Step-Audio 2 | 1,460 | 端到端多模态 Speech LLM |
| Step-Audio-EditX | 926 | 3B 音频编辑模型 (RL) |
| Step-Audio-R1 | 672 | 音频推理模型 |
| Step-Audio-R1.5 | (含在 R1 中) | 推理增强版 |
| StepAudio-Skills | 26 | Claw 语音技能 |
| Step-Audio (V1) | 26 | 初代 130B |

**注意**: StepAudio 2.5 论文已发但开源状态待确认。

### 5.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) 迭代速度最快: 16 个月内从 V1 (130B AQTA) 进化到 2.5 (统一 MoE 基座),发布 7+ 篇论文; (2) 技术创新密度高: dual-codebook, generative data engine, anti-deaf-hacking, MTP-5, GRM, persona matrix; (3) Realtime 人类评估全面 SOTA (超 GPT-realtime, Gemini-live, Doubao); (4) ASR RTF 0.0053 是工业级最快; (5) 产品+研究紧密结合 |
| **短板** | (1) 模型参数量未公开 (MoE backbone); (2) TTS 评估仅用 arena pairwise,缺乏标准 benchmark; (3) 部分评估 baseline 非本地推理 (通过 API); (4) 数据透明度有限; (5) 商业化压力 (创业公司资源有限); (6) 多语言覆盖不足 |
| **下一步推测** | (1) StepAudio 3.0 可能引入视觉模态 (对标 Qwen-Omni); (2) 全双工实时对话进一步优化; (3) 更多语言和方言支持; (4) Step-Audio-R 系列可能发展为独立的音频推理产品线; (5) 开源社区建设 (目前 stars 较低) |

---

## 6. 横向对比矩阵

### 6.1 技术路线对比

| 维度 | 阿里 FunAudioLLM | 阿里 Qwen | 字节豆包 | 智谱 | 阶跃 |
|------|-----------------|-----------|---------|------|------|
| **架构范式** | Parallel LALM (5Hz+25Hz) | Thinker-Talker (Hybrid MoE) | Codec-free LLM (SALMONN-omni) | Text LLM + 合成交替预训练 | 统一基座 + 方向特化 |
| **LLM 规模** | 8B dense / 30B-A3B MoE | 数百亿 MoE (未公开具体参数) | 未公开 | 9B | MoE (未公开) |
| **预训练规模** | Post-training only | ~4T tokens | 未公开 | 1T tokens | 2.2T tokens |
| **Tokenizer** | S3Tokenizer (FSQ, 25Hz) | AuT (6.25Hz) + RVQ codec | Codec-free | VQ-Whisper (12.5Hz, 175bps) | Dual-codebook → MoE 统一 |
| **全双工** | 支持 (Fun-Audio-Chat-Duplex) | 不支持 | 支持 (SALMONN-omni) | 不支持 | 支持 (Realtime 方向) |
| **多语言 TTS** | 9 语种 | 36 语种 (29 有评估) | 未公开 | 2 语种 (中英) | 中英+方言 |
| **开源程度** | ★★★★★ (最高) | ★★★☆☆ (最新未开源) | ★☆☆☆☆ (最低) | ★★★★☆ | ★★★☆☆ |

### 6.2 性能对比 (可比数据)

| 指标 | FunAudioLLM | Qwen | 字节 | 智谱 | 阶跃 |
|------|-------------|------|------|------|------|
| **SEED-TTS WER (en)** | CosyVoice 3: 1.45 | Qwen3.5-Omni: **1.26** | Seed-TTS: N/A | GLM-4-Voice: 2.91 | Step-Audio 130B: 2.0 |
| **SEED-TTS CER (zh)** | CosyVoice 3: **0.71** | Qwen3.5-Omni: 0.99 | Seed-TTS: N/A | GLM-4-Voice: 2.10 | Step-Audio 130B: 1.17 |
| **MMAU** | Fun-Audio-Chat 30B: **77.9** | Qwen3.5-Omni: **82.2** | N/A | N/A | N/A |
| **VoiceBench** | Fun-Audio-Chat 30B: 85.63 | Qwen3.5-Omni: **93.1** | N/A | N/A | N/A |
| **Llama Questions** | N/A | N/A | N/A | GLM-4-Voice: 50.7 | Step-Audio: **81.0** |
| **AISHELL-1 CER** | SenseVoice-L: 2.09 | Qwen3-ASR: 1.49 | Seed-ASR: N/A | GLM-4-Voice: 2.46 | StepAudio 2.5: **0.71** |
| **全双工 Turn-taking** | Fun-Audio-Chat: **100%** | N/A | SALMONN-omni: N/A | N/A | N/A |
| **Realtime 人类评估** | N/A | N/A | N/A | N/A | StepAudio 2.5: **80.41** |

*注: 各团队评估 benchmark 和条件不完全一致,横向对比仅供参考。*

### 6.3 GitHub Stars 对比 (Speech LLM 相关)

| 团队 | 最高 Stars 项目 | Stars | 总 Speech LLM 相关 Stars |
|------|---------------|-------|------------------------|
| FunAudioLLM | CosyVoice | 21,501 | ~35,000+ |
| Qwen | Qwen2.5-Omni | 4,017 | ~11,800 |
| 阶跃 | Step-Audio 2 | 1,460 | ~3,100 |
| 字节 | SALMONN | 1,443 | ~1,800 |
| 智谱 | GLM-4-Voice | 3,184 | ~4,200 |

---

## 7. 关键发现

### 发现 1: 阿里双团队的分化与融合

阿里在 Speech LLM 领域形成了独特的"双团队"格局:
- **FunAudioLLM** 走 post-training + parallel LALM 路线,优势在开源生态和计算效率 (5Hz LLM)
- **Qwen** 走大规模预训练 + Thinker-Talker 全模态路线,优势在多语言和全模态能力

两者共享 S3Tokenizer (CosyVoice 3) 和部分数据基础设施,但技术路线明显不同。这种"内部竞争+协同"的模式在国内大厂中独一无二,也带来资源分配和产品定位的挑战。

### 发现 2: 全双工是核心分水岭

在四个团队中,明确支持全双工的有: FunAudioLLM (Fun-Audio-Chat-Duplex), 字节 (SALMONN-omni), 阶跃 (Realtime 方向)。**智谱和 Qwen 均不支持全双工**,这是它们目前的功能缺口。全双工是从"语音助手"到"自然对话"的关键能力,不支持全双工意味着用户体验的天花板。

### 发现 3: 开源程度差异巨大

FunAudioLLM 是国内 Speech LLM 开源的绝对领导者 (CosyVoice 21K stars),字节是开源最保守的 (核心模型全部闭源)。这种差异反映了不同的商业策略: 阿里通过开源建立生态获取开发者,字节通过产品化 (豆包) 直接获取用户。

### 发现 4: 阶跃的迭代速度惊人

作为创业公司,阶跃星辰在 16 个月内发布了 7+ 篇 Speech LLM 论文,从 130B AQTA 架构进化到统一 MoE 基座,技术创新密度在四个团队中最高。MTP-5 (RTF 0.0053) 和百万级 persona matrix 等创新具有独特价值。但资源限制使其在多语言覆盖和全模态能力上不及阿里。

### 发现 5: 技术路线趋向收敛

尽管起点不同,四个团队的技术路线正在趋向收敛:
- **统一基座**: 从分离的 ASR+LLM+TTS 走向统一的端到端模型
- **RLHF 对齐**: 所有团队都在引入 RL/DPO/GRPO 进行语音质量和交互对齐
- **MoE 架构**: Qwen3.5-Omni 和 StepAudio 2.5 都采用 MoE
- **Tokenizer 监督化**: 从无监督/自监督 tokenizer 走向 ASR 监督的 semantic tokenizer

---

## 附: 数据来源标注

| 信息类型 | 来源 | 可信度 |
|---------|------|--------|
| 论文技术细节 | vault 已有精读笔记 (FunAudioLLM, GLM-4-Voice, Step-Audio, Step-Audio2.5, Fun-Audio-Chat, Qwen3.5-Omni, CosyVoice3) | 高 (经审阅) |
| GitHub stars/repos | GitHub API (2026-06-08 实时查询) | 高 |
| arXiv 论文列表 | arXiv API (2026-06-08 查询) | 高 |
| 产品信息 | 公开资料 + 论文引用 | 中 |
| 团队人物 | 论文作者列表 + GitHub org | 中 |
| HuggingFace 模型 | API 查询 (部分失败) | 中 |
| 性能数据横向对比 | 各团队论文自报数据,评估条件可能不一致 | 中 (需谨慎解读) |
| 字节内部产品技术栈 | 推测 (基于公开论文和产品信息) | 低 |
