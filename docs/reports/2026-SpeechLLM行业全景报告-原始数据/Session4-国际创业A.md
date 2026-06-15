# Session 4: 国际创业 A — Kyutai / Sesame / Thinking Machines / Cartesia

> **报告范围**: Speech LLM / Omni / 全双工对话
> **调研日期**: 2026-06-08
> **数据来源**: GitHub API, HuggingFace, arXiv, vault 已有论文笔记 (Moshi), 公开技术报告, 官方网站
> **5 层搜索覆盖**: Layer 1 (GitHub/HuggingFace org) ✓ | Layer 2 (核心人搜索) ✓ | Layer 3 (arXiv affiliation) 部分 | Layer 4 (产品/竞赛) ✓ | Layer 5 (引用网络) ✓ (通过 vault 笔记)

---

## 1. Kyutai — Moshi / DSM / Unmute

### 1.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | Kyutai — 法国非营利开放科学 AI 实验室 |
| 团队 | Kyutai Research Lab; 由法国电信企业家 Xavier Niel (Iliad) 创立并资助,约 €300M |
| GitHub | [github.com/kyutai-labs](https://github.com/kyutai-labs) (29 repos, 1.7K followers) |
| HuggingFace | [huggingface.co/kyutai](https://huggingface.co/kyutai) |
| 核心人物 | Alexandre Défossez (首席科学家, 原 Meta FAIR), Laurent Mazaré (工程负责人, 原 HuggingFace), Neil Zeghidour (原 Google DeepMind), Patrick Pérez (实验室主任, 原 Valeo VP AI), Edouard Grave, Manu Orsini, Hervé Jégou |
| 产品线 | Moshi (全双工对话), Mimi (neural audio codec), DSM (Delayed Streams Modeling, STT/TTS 框架), Unmute (语音 AI 系统), Pocket-TTS (轻量级 TTS), Hibiki (流式语音翻译), Moshi-RAG (RAG 增强对话), Invincible Voice (语音修复) |
| 官网 | [kyutai.org](https://kyutai.org) |
| 定位 | 全双工语音对话先驱; 开放科学理念,所有核心模型全部开源; 从 codec → dialogue → STT/TTS → 翻译全链路覆盖 |

### 1.2 论文时间线 (2024-2026)

| 时间 | 论文/项目 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|---------|----------|---------|-------------------|
| 2024.10 | Moshi | 2410.00037 | 首个全双工实时语音对话大模型; Helium 7B + Mimi codec + RQ-Transformer + Inner Monologue | **里程碑**: 全双工对话的开创者,160ms 理论延迟 |
| 2024.10 | Mimi (含在 Moshi) | — | Split RVQ (1 VQ semantic + 7 RVQ acoustic), 12.5Hz, 1.1kbps, adversarial-only training | **核心组件**: 混合语义-声学 codec,被多个后续工作引用 |
| 2025.09 | DSM (Delayed Streams Modeling) | 2509.08753 | 统一的流式 sequence-to-sequence 框架,形式化 Moshi 的 delay 机制 | **技术框架**: STT/TTS 基础设施 |
| 2025.09 | Kyutai STT | (含在 DSM) | 1B (en/fr) + 2.6B (en) 流式 ASR 模型; Semantic VAD; H100 可处理 400 并发流 | 语音理解基座 |
| 2025.09 | Kyutai TTS 1.6B | (含在 DSM) | 流式 TTS, 支持 voice cloning; PyTorch/Rust/MLX 多平台 | 语音生成基座 |
| 2025 | Unmute | — | 语音 AI 系统: STT → LLM → TTS pipeline; 兼容 OpenAI Realtime API 协议 | **产品化**: 可替换任意 text LLM |
| 2025 | Pocket-TTS | — | CPU 可运行的超轻量级 TTS | 端侧部署 |
| 2025 | Hibiki | — | 流式语音翻译 (同声传译), 基于 DSM 框架 | 跨语言对话 |
| 2026 | Moshi-RAG | — | 全双工 Speech LM + 异步知识检索, 提升事实性 | **核心 Speech LLM 升级**: 不牺牲实时性的同时增强知识 |
| 2026 | Invincible Voice | — | 语音修复: 为失去声音的人恢复语音 | 社会应用 |

### 1.3 技术栈全景

| 维度 | Kyutai 技术栈 |
|------|-------------|
| **语音编码器** | Mimi: Causal Conv + Split RVQ (1 VQ semantic + 7 RVQ acoustic), 12.5Hz, 1.1kbps; WavLM 蒸馏注入语义层; adversarial-only training (去除重建 loss, MUSHRA 81.0) |
| **LLM 骨干** | Helium 7B (自研 Transformer, 2.1T text tokens 预训练); Moshi v1 在此基础上做 audio pretraining + multi-stream post-training |
| **语音解码器** | Mimi decoder (causal Conv + adversarial training); Kyutai TTS 1.6B (DSM 框架, 流式生成); Pocket-TTS (轻量版) |
| **对话策略** | Multi-stream: 用户和 Moshi 的音频流拼接为 17 个子序列 (2Q+1, Q=8 codebook); 无显式 speaker turn — 模型可同时说话和倾听; Inner Monologue: 每步先预测 text token 再预测 audio tokens |
| **训练数据规模** | Helium: 2.1T text tokens; Audio pretrain: 7M 小时无监督音频; Multi-stream: PyAnnote 模拟 + Fisher 2000h 真实对话 + 20K+ 小时合成指令数据 |
| **推理延迟** | Moshi: 160ms 理论延迟 (12.5Hz frame rate); Unmute TTS: ~450ms (多 GPU 部署) vs ~750ms (单 GPU); DSM STT: H100 支持 400 并发流实时处理 |
| **多语言** | Moshi v1: 仅英语; Kyutai STT 1B: 英语+法语; Hibiki: 流式翻译支持多语言 |
| **情感/副语言** | Mimi codec 的 split RVQ 天然编码副语言信息; TTS 支持 voice cloning (voice donation 项目收集真人声音) |

### 1.4 架构演进

```
Helium 7B (Text LLM, 2.1T tokens)
    ↓ 扩展到 audio
Mimi Codec (Split RVQ, adversarial-only, 12.5Hz, 1.1kbps, 2024.10)
    ↓ 编解码基座
Moshi (7B, Multi-stream 全双工, Inner Monologue, 2024.10)
    ├── RQ-Transformer (Temporal 7B + Depth 6L)
    ├── Multi-stream (17 子序列, 无显式 turn)
    └── Inner Monologue (text as prefix scaffold)
    ↓ 框架化
DSM (Delayed Streams Modeling, 2025.09) ←── 统一形式化 Moshi/Hibiki 的 delay 机制
    ├── Kyutai STT (1B en/fr + 2.6B en, Semantic VAD)
    ├── Kyutai TTS 1.6B (流式, voice cloning)
    └── Hibiki (流式语音翻译)
    ↓ 产品化
Unmute (STT→任意LLM→TTS pipeline, OpenAI Realtime API 兼容, 2025)
    ↓ 端侧
Pocket-TTS (CPU 可运行, 2025)
    ↓ 知识增强
Moshi-RAG (异步 RAG + 全双工, 2026)

并行社会项目:
Invincible Voice (语音修复, 2026)
```

### 1.5 独特技术赌注

1. **全双工作为第一原理**: Kyutai 从一开始就以全双工为设计目标,而非先做 turn-based 再添加全双工。Multi-stream 架构完全移除 speaker turn 概念,模型天然可在任何时刻同时说话和倾听。
2. **Inner Monologue**: 每个时间步先预测 text token 再预测 audio tokens,带来三重价值: (1) 语言质量大幅提升 (spoken QA 精度约 3 倍); (2) 统一 ASR/TTS/dialogue 到同一架构; (3) 近乎零额外推理成本。这是"text as scaffold for speech"的开创性思路。
3. **Split RVQ + Adversarial-only Codec**: 物理解耦语义 (1 VQ) 与声学 (7 RVQ),去除重建 loss 只保留对抗训练,MUSHRA 从 58.8 提升到 81.0。揭示了客观指标与人类感知的严重脱节。
4. **DSM 形式化**: 将 Moshi 的 delay 机制抽象为通用框架,统一 STT/TTS/翻译等流式任务。
5. **完全开放科学**: 作为非营利机构,所有核心模型 (Moshi, Mimi, DSM, Unmute, Pocket-TTS, Hibiki) 全部开源,代码+权重+论文一并发布。

### 1.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| moshi | 10,350 | 全双工语音对话模型 + Mimi codec (Apache-2.0) |
| pocket-tts | 4,571 | CPU 可运行的轻量级 TTS (MIT) |
| delayed-streams-modeling | 2,900 | DSM 框架 STT/TTS 模型 (MIT/Apache) |
| hibiki | 1,500 | 流式语音翻译 (Rust 实现) |
| unmute | 1,322 | 语音 AI 系统 (STT→LLM→TTS, MIT) |
| moshi-finetune | 457 | Moshi 微调工具 |
| moshi-rag | 100 | RAG 增强全双工对话 (Apache-2.0) |
| invincible-voice | 92 | 语音修复项目 (MIT) |
| tts_longeval | 30 | TTS 长文本评估 benchmark |

**总计 Speech 相关 Stars: ~21,300+**

### 1.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **全双工先驱**: Moshi 是全双工语音对话的里程碑,后续 SALMONN-omni、Fun-Audio-Chat-Duplex 等均引用借鉴; (2) **开源程度最高**: 非营利定位 + 全部核心模型开源,社区影响力大 (moshi 10K stars); (3) **技术栈完整**: 从 codec (Mimi) → 对话 (Moshi) → STT/TTS (DSM) → 翻译 (Hibiki) → 产品 (Unmute) → 端侧 (Pocket-TTS) → RAG 增强全链路; (4) **核心团队极强**: Défossez (EnCodec 作者), Zeghidour (SoundStream 联合作者), Mazaré (HuggingFace candle), Pérez, Grave (FAIR); (5) **DSM 形式化框架** 使 STT/TTS 可独立部署,Unmute 支持接入任意 LLM |
| **短板** | (1) **仅英语+法语**: 多语言覆盖严重不足,Moshi v1 仅支持英语; (2) **7B 规模偏小**: 与 130B (Step-Audio) 或 MoE 架构 (Qwen-Omni) 相比,知识容量有限; (3) **知识遗忘**: 音频训练导致 MMLU 从 54.3 降至 49.8; (4) **非营利模式的可持续性**: 依赖捐赠资金,长期竞争力不确定; (5) **安全评分中等** (ALERT 83.05 vs GPT-4 99.98); (6) 缺乏大规模产品化验证 |
| **下一步推测** | (1) Moshi v2 可能增加多语言和更大 LLM backbone; (2) Moshi-RAG 可能是知识增强的主要方向; (3) DSM 框架可能扩展到更多流式任务; (4) Unmute 可能发展为类 OpenAI Realtime API 的开源替代; (5) 端侧部署 (Pocket-TTS + MLX) 可能成为差异化方向 |

---

## 2. Sesame — CSM (Conversational Speech Model)

### 2.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | Sesame (SesameAILabs) — 美国旧金山 AI 创业公司 |
| 团队 | SesameAILabs; 创始团队背景待公开确认 |
| GitHub | [github.com/SesameAILabs](https://github.com/SesameAILabs) (13 repos, 无公开成员) |
| HuggingFace | [huggingface.co/sesame](https://huggingface.co/sesame) |
| 核心人物 | 团队成员不公开 (GitHub org 无公开成员); CSM 论文作者待确认 |
| 产品线 | CSM (Conversational Speech Model, 开源); 语音对话产品 (demo) |
| 定位 | 对话语音生成专精; CSM 以极高社区热度快速崛起,14.7K stars |

### 2.2 论文时间线 (2025)

| 时间 | 论文/项目 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|---------|----------|---------|-------------------|
| 2025.03 | CSM (Conversational Speech Model) | — | 对话语音生成模型; 基于 Llama backbone + 多 codebook 语音生成; 支持多轮对话上下文 | **核心产品**: 对话语音生成 |
| 2025 | Moshi fork | — | Fork 了 kyutai-labs/moshi,可能用于内部研发 | 技术借鉴/研究 |
| 2025 | faster-whisper-plus | — | Fork + 增强版 faster-whisper ASR | ASR 工具链 |
| 2025 | silentcipher | — | Fork 自 Sony 的音频水印工具 | 安全/版权 |

**注意**: Sesame 目前没有公开发表的 arXiv 论文。CSM 的技术细节主要通过 GitHub README 和 HuggingFace 模型卡发布。

### 2.3 技术栈全景

| 维度 | Sesame 技术栈 |
|------|-------------|
| **语音编码器** | 基于 Mimi codec (借鉴 Kyutai); 可能使用自研或修改版 tokenizer |
| **LLM 骨干** | 基于 Llama 架构; CSM 使用 Transformer backbone 进行语音 token 预测 |
| **语音解码器** | 多 codebook 自回归生成; 从 semantic tokens 到 acoustic tokens 逐层生成 |
| **对话策略** | 多轮对话上下文建模; 将对话历史 (多个说话人的 audio tokens) 作为条件输入 |
| **训练数据规模** | 未公开 |
| **推理延迟** | 未公开具体数据 |
| **多语言** | 英语为主 (公开信息有限) |
| **情感/副语言** | CSM 设计目标之一是生成自然、有表情的对话语音 |

### 2.4 架构演进

```
Moshi (kyutai-labs/moshi fork, 研究参考)
    ↓ 技术借鉴
CSM v1 (Conversational Speech Generation Model, 2025.03)
    ├── Llama backbone + 多 codebook 语音生成
    ├── 对话上下文建模 (多说话人 audio tokens)
    └── 14.7K GitHub stars
    ↓ (后续迭代不明)
sglang fork (高性能推理框架适配)
torchtune/torchtitan forks (训练基础设施)
```

### 2.5 独特技术赌注

1. **对话上下文建模**: CSM 的核心差异化在于将完整的多轮对话上下文 (包括多个说话人的 audio tokens) 作为条件输入,使生成的语音在韵律、节奏上与对话上下文自然衔接。
2. **开源社区策略**: 以单一开源模型 (CSM, 14.7K stars) 快速积累社区关注度,是 Speech LLM 领域增长最快的开源项目之一。
3. **Moshi 技术路线继承**: Fork 了 Moshi 代码库,可能在 Kyutai 的全双工技术基础上进行迭代。
4. **推理优化**: Fork 了 sglang (高性能 LLM 推理框架),暗示在推理延迟优化方面有投入。

### 2.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| csm | 14,659 | 对话语音生成模型 (Apache-2.0) |
| whisperX (fork) | 69 | ASR 工具 (fork from m-bain) |
| faster-whisper-plus (fork) | 50 | 增强版 faster-whisper (fork) |
| wavtools (fork) | 40 | 浏览器音频录制工具 |
| moshi (fork) | 24 | 全双工对话模型 (fork from kyutai-labs) |
| sglang (fork) | 22 | 高性能推理框架 (fork) |
| silentcipher (fork) | 21 | 音频水印 (fork from sony) |

**注意**: 除 CSM 外,其余全部为 fork 项目。原创开源仅 CSM 一个项目。

### 2.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **社区热度极高**: CSM 14.7K stars 超过 Moshi (10.4K),是最受关注的对话语音生成模型之一; (2) 对话上下文建模是有价值的技术方向; (3) 技术选型务实: 借鉴 Moshi/Mimi 的成熟技术 |
| **短板** | (1) **信息透明度极低**: 无公开论文、无公开团队成员、无公开技术报告; (2) **原创开源仅 CSM 一个项目**,其余全是 fork; (3) **技术细节缺失**: 无法评估训练数据规模、推理延迟、多语言能力等关键指标; (4) **未见全双工对话能力**: CSM 定位为"对话语音生成"而非完整的对话系统; (5) 商业模式和融资情况不透明 |
| **下一步推测** | (1) 可能发布技术论文补充 CSM 的理论基础; (2) 可能在 CSM 基础上构建完整的对话系统 (借鉴已 fork 的 Moshi); (3) 可能推出商业化 API 或产品; (4) sglang fork 暗示正在优化推理部署 |

---

## 3. Thinking Machines Lab — Tinker

### 3.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | Thinking Machines Lab — 美国 AI 研究与产品公司 |
| 团队 | 由前 ChatGPT、Character.AI、Mistral、Meta AI (PyTorch, Fairseq, Segment Anything, OpenAI Gym) 核心成员创立 |
| GitHub | [github.com/thinking-machines-lab](https://github.com/thinking-machines-lab) (7 repos, 1.6K followers, 4 public members) |
| HuggingFace | — (主要通过 Tinker API 提供训练服务) |
| 核心人物 | @nealwu (Neal Wu), @YujiaBao (Yujia Bao), @ekzhang (Eric Zhang), @Mars-tin; 具体身份需进一步确认 |
| 产品线 | Tinker (LoRA 训练 API + CLI) |
| 官网 | [thinkingmachines.ai](https://thinkingmachines.ai) |
| X | [@thinkymachines](https://x.com/thinkymachines) |
| 定位 | LLM 训练基础设施公司; 提供 LoRA 微调 API,支持 Qwen/Llama/DeepSeek/GPT-OSS 等开源模型 |

### 3.2 与 Speech LLM 的关系

**重要说明**: Thinking Machines Lab 目前 **不是一家 Speech LLM 公司**。其核心产品 Tinker 是一个通用 LLM 训练 API,目前仅支持文本模型的 LoRA 微调。

然而,将其纳入本 Session 的理由如下:

1. **"Advanced multimodal capabilities"**: 官网明确将高级多模态能力列为核心愿景之一,表示"we see multimodality as critical to enabling more natural and efficient communication"
2. **"Human-AI collaboration"**: 强调"multimodal systems that work with people collaboratively",这与语音交互高度相关
3. **强大的技术背景**: 团队来自 ChatGPT、Character.AI (语音对话)、Meta FAIR (Fairseq/语音研究),具备向语音方向扩展的能力
4. **Tinker 基础设施潜力**: 如果 Tinker 扩展到支持 speech LLM 的训练 (如 Qwen-Omni, CSM 等),将填补当前市场空白

### 3.3 论文时间线

| 时间 | 项目 | 核心贡献 | 与 Speech LLM 关系 |
|------|------|---------|-------------------|
| 2025 | Tinker | LoRA 训练 API,4 个核心函数 (forward_backward, optim_step, sample, save_state) | **间接相关**: 可能用于 Speech LLM 的后训练 |
| 2025 | batch_invariant_ops | 批次不变操作 (1K stars) | 通用训练优化 |
| 2025 | Manifolds | 模块化流形 (122 stars) | 研究探索 |
| 2025-2026 | Interaction Models | "A Scalable Approach to Human-AI Collaboration" (博客文章) | **概念相关**: 人机交互范式 |

### 3.4 技术栈全景

| 维度 | Thinking Machines 技术栈 |
|------|------------------------|
| **核心产品** | Tinker: LoRA 训练 API (forward_backward → optim_step → sample → save_state) |
| **支持模型** | Qwen 3.6/3.5/3 系列, Llama 3.1-3.3, DeepSeek V3.1, GPT-OSS 120B/20B, Kimi K2.5/K2.6, NVIDIA Nemotron 3 |
| **训练方式** | LoRA 微调 (不支持 full fine-tuning); 研究表明 LoRA 与 full fine-tuning 学习性能匹配 |
| **部署** | 云端 GPU 集群; 用户无需管理基础设施 |
| **语音/多模态** | **目前不支持**; 未列出任何 speech/audio 模型 |

### 3.5 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| tinker-cookbook | 3,433 | Post-training 教程和示例 (Apache-2.0) |
| batch_invariant_ops | 1,024 | 批次不变操作 (MIT) |
| tinker | 499 | 训练 API + CLI (Apache-2.0) |
| tinker-project-ideas | 176 | 社区项目创意 |
| manifolds | 122 | 模块化流形研究 (MIT) |
| tinker-feedback | 39 | 反馈追踪 |

### 3.6 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **团队背景极强**: ChatGPT/Character.AI/Mistral/Meta AI 核心成员; (2) Tinker 的简洁 API 设计如果扩展到 speech LLM,将是有价值的基础设施; (3) 社区参与度高 (tinker-cookbook 3.4K stars); (4) 明确的多模态愿景 |
| **短板** | (1) **目前与 Speech LLM 无直接关联**: 无语音模型、无语音论文、无语音产品; (2) Tinker 仅支持文本 LLM 的 LoRA 微调; (3) 公开研究成果有限 |
| **下一步推测** | (1) Tinker 可能扩展到支持多模态模型训练 (语音+视觉); (2) "Interaction Models" 研究可能发展为语音交互系统; (3) Character.AI 背景的成员可能推动语音对话能力 |

**诚实评估**: 在 Speech LLM / Omni / 全双工对话领域,Thinking Machines Lab 目前的贡献为零。将其纳入本 Session 是基于其团队背景和多模态愿景的前瞻性判断,而非已有成果。

---

## 4. Cartesia — Sonic / Ink / Line

### 4.1 基本信息

| 项目 | 详情 |
|------|------|
| 公司 | Cartesia AI — 美国 AI 创业公司,专注于 SSM (State Space Model) 架构 |
| 团队 | 核心团队来自 Stanford,SSM/Mamba 架构的发明者 |
| GitHub | [github.com/cartesia-ai](https://github.com/cartesia-ai) (30 repos, 216 followers) |
| HuggingFace | [huggingface.co/cartesia-ai](https://huggingface.co/cartesia-ai) |
| 核心人物 | Albert Gu (CTO/联合创始人, Mamba/S4/HiPPO 作者, Stanford), Karan Goel (CEO/联合创始人, Stanford), Tri Dao (Mamba 合著者, Flash Attention 作者); 其他研究员: Simran Arora, Sabri Eyuboglu, Christopher Ré (Stanford 教授/顾问) |
| 产品线 | Sonic (TTS API), Ink (STT API), Line (语音 Agent 平台); Edge (端侧推理) |
| 官网 | [cartesia.ai](https://cartesia.ai) |
| X | [@cartesia_ai](https://x.com/cartesia_ai) |
| 定位 | SSM 架构在语音领域的产品化先锋; "Architecting AI that learns and interacts like humans"; 超低延迟 + 端侧部署为核心卖点 |

### 4.2 论文时间线 (2020-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2020 | HiPPO | 2008.07669 | 循环记忆的最优多项式投影 | **理论基础**: SSM 的数学基础 |
| 2022 | S4 (Structured State Spaces) | 2111.00396 | 首个实用的 SSM 序列模型 | **架构起源**: 长序列建模基础 |
| 2022 | SaShiMi (It's Raw!) | 2202.09729 | SSM 用于原始音频生成 | **直接相关**: 证明 SSM 在音频域的可行性 |
| 2022 | S4ND | 2210.06583 | 多维 SSM (图像/视频) | SSM 多模态扩展 |
| 2023.12 | Mamba | 2312.00752 | 选择性状态空间模型; 线性时间序列建模 | **核心架构**: Cartesia 产品的基础架构 |
| 2024 | Mamba-2 / SSD | 2405.21060 | "Transformers are SSMs"; 结构化状态空间对偶性 | 架构统一理论 |
| 2024 | Based | — | 简单线性注意力语言模型 | 效率优化 |
| 2024 | Zoology | 2312.04927 | 高效语言模型的召回率测量与改进 | 模型评估 |
| 2025 | Llamba | 2502.14458 | 蒸馏循环模型的规模化 | 模型压缩/蒸馏 |
| 2025-2026 | Sonic 3.5 | — (未公开论文) | 最新版 TTS 模型; SSM 架构; 超低延迟 | **旗舰 TTS**: 商业化 |
| 2025-2026 | Ink 2 | — (未公开论文) | 最新版 STT 模型; SSM 架构; 流式转录 | **旗舰 STT**: 商业化 |

### 4.3 技术栈全景

| 维度 | Cartesia 技术栈 |
|------|---------------|
| **语音编码器** | Ink (STT): 基于 SSM/Mamba 架构的流式转录模型; "最快最准确的流式转录模型" (官方声称) |
| **LLM 骨干** | 基于 Mamba/SSM 架构 (非 Transformer); 线性时间复杂度; 支持长上下文推理 |
| **语音解码器** | Sonic (TTS): 基于 SSM/Mamba 架构的语音生成模型; 支持 voice cloning, 多语言, 情感控制 |
| **对话策略** | Line: 语音 Agent 平台,编排 Ink (STT) + 任意 LLM + Sonic (TTS) 构建语音 Agent; 类似 Unmute 的 pipeline 架构 |
| **训练数据规模** | 未公开 |
| **推理延迟** | **超低延迟** 是核心卖点 (具体数值通过 API 提供,未公开基准测试); SSM 的 O(n) 复杂度 vs Transformer 的 O(n²) 理论上给出延迟优势 |
| **多语言** | 支持多语言 (官网有专门的 Languages 页面) |
| **情感/副语言** | 支持 AI Voice Generator, Voice Cloning, Voice Changer, Voice Conversion 等丰富的语音控制能力 |

### 4.4 架构演进

```
HiPPO (循环记忆理论, 2020)
    ↓ 理论到模型
S4 (Structured State Spaces, 2022)
    ├── SaShiMi (原始音频生成, 2022) ←── 证明 SSM 在音频域可行
    └── S4ND (多维信号, 2022)
    ↓ 选择性机制
Mamba (Selective SSM, 线性时间, 2023.12)
    ↓ 理论统一
Mamba-2 / SSD (Transformers are SSMs, 2024)
    ↓ 产品化
Sonic (SSM-based TTS, API 服务)
    ↓ 迭代
Sonic 2 → Sonic 3 → Sonic 3.5 (2025-2026, 最新)
    
并行:
Ink (SSM-based STT) → Ink 2 (2025-2026, 最新)
    
平台:
Line (Voice Agent Platform, 编排 STT+LLM+TTS)
Edge (端侧推理框架, 408 stars)

模型压缩:
Llamba (蒸馏循环模型, 2025)
```

### 4.5 独特技术赌注

1. **SSM 替代 Transformer 路线**: Cartesia 是唯一一家将 SSM (State Space Model) 作为核心架构用于语音的公司。SSM 的理论优势: O(n) 时间复杂度 (vs Transformer O(n²)), 固定大小状态 (适合流式推理), 原生长上下文支持。这是一个大胆的架构赌注 — 如果 SSM 在语音任务上胜过 Transformer,Cartesia 将拥有独特壁垒。
2. **端侧部署 (Edge/On-device)**: SSM 的固定状态大小天然适合端侧设备 (手机、机器人),Cartesia 已提供 on-device 部署方案。这在隐私敏感场景 (医疗、金融) 有独特价值。
3. **Voice Agent 平台 (Line)**: 不仅提供模型 API,还提供完整的语音 Agent 编排平台,覆盖从 STT → LLM → TTS → 部署的全栈。客户包括 ServiceNow、Retell AI、Zomato 等。
4. **学术→产品的直接转化**: 从 S4 论文 (2022) 到 Sonic 商业产品的转化仅用约 2 年,是学术研究产品化的范例。
5. **SaShiMi (SSM audio generation)**: 早在 2022 年就证明了 SSM 在原始音频波形生成上的可行性,为后续 Sonic TTS 奠定了技术基础。

### 4.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| edge | 408 | 端侧推理框架 |
| cartesia-js | 131 | JavaScript SDK |
| cartesia-python | 123 | Python SDK |
| line | 100 | 语音 Agent 平台 SDK (Apache-2.0) |
| cartesia-livekit-voice-agent | 21 | LiveKit 集成 demo |
| dev-showcase | 20 | 开发者展示 |
| cartesia-mcp | 13 | MCP server |

**注意**: Cartesia **不开源核心模型** (Sonic, Ink)。开源的主要是 SDK/工具/demo。这与其商业化 API 模式一致。Mamba/S4 等基础研究论文代码在个人或 Stanford 仓库开源,不在 Cartesia org 下。

### 4.7 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) **独特的架构赌注**: SSM/Mamba 路线在语音领域独此一家; 如果成功,拥有不可复制的技术壁垒; (2) **核心团队世界级**: Albert Gu (SSM 发明者), Tri Dao (Flash Attention), Christopher Ré (Stanford) — 同时拥有理论创新和工程能力; (3) **商业化成熟度高**: 已有 ServiceNow、Zomato 等企业客户; (4) **端侧部署差异化**: SSM 天然适合边缘设备; (5) **产品全栈**: 从模型 (Sonic/Ink) 到平台 (Line) 到部署 (Edge/Cloud/On-prem) |
| **短板** | (1) **核心模型不开源**: Sonic/Ink 仅通过 API 提供,无法验证其技术细节; (2) **不支持端到端对话**: Cartesia 提供的是 pipeline 组件 (STT+TTS),不是端到端 Speech LLM; (3) **缺少学术评估**: 无公开 benchmark 对比数据 (如 SEED-TTS eval, LibriSpeech WER 等); (4) **SSM 路线的不确定性**: Mamba 在 NLP 任务上已被证明不如 Transformer,在语音任务上的优势尚未被独立验证; (5) **缺少全双工能力**: 作为 pipeline 系统,全双工需要额外的工程 |
| **下一步推测** | (1) 可能发布 Sonic/Ink 的技术论文和 benchmark; (2) Line 平台可能增加端到端对话能力; (3) 端侧部署 (Edge) 可能成为核心竞争力; (4) 可能推出 Mamba-based 端到端 Speech LLM; (5) 如果 SSM 在语音任务上被证明优于 Transformer,Cartesia 将成为关键赢家 |

---

## 5. 横向对比矩阵

### 5.1 技术路线对比

| 维度 | Kyutai | Sesame | Thinking Machines | Cartesia |
|------|--------|--------|-------------------|----------|
| **架构范式** | Multi-stream Transformer (RQ-Transformer + Inner Monologue) | Transformer (Llama backbone + multi-codebook) | N/A (通用 LLM 训练 API) | SSM/Mamba (非 Transformer) |
| **端到端 vs Pipeline** | 端到端 (Moshi) + Pipeline (Unmute) | 端到端 (CSM, 生成侧) | N/A | Pipeline (Ink + LLM + Sonic) |
| **LLM 规模** | 7B (Helium) | 未公开 | 支持训练 1B-550B 文本 LLM | 未公开 |
| **Tokenizer/Codec** | Mimi (Split RVQ, 12.5Hz, 1.1kbps, adversarial-only) | 基于 Mimi (推测) | N/A | 未公开 (SSM 内部表示) |
| **全双工** | **支持** (Moshi, 开创者) | 不明确 (CSM 是生成模型) | N/A | 不支持 (pipeline) |
| **多语言** | 英语+法语 (STT 1B) | 英语为主 | 支持多语种文本 LLM 训练 | 多语言 (具体未公开) |
| **开源程度** | ★★★★★ (全部开源) | ★★★☆☆ (CSM 开源,其余 fork) | ★★★★☆ (Tinker API 部分开源) | ★☆☆☆☆ (仅 SDK 开源) |
| **商业模式** | 非营利/开放科学 | 待确认 | SaaS (训练 API) | SaaS (语音 API) + 私有部署 |

### 5.2 关键指标对比

| 指标 | Kyutai | Sesame | Thinking Machines | Cartesia |
|------|--------|--------|-------------------|----------|
| **最高 Stars 项目** | moshi (10,350) | csm (14,659) | tinker-cookbook (3,433) | edge (408) |
| **总 Speech 相关 Stars** | ~21,300+ | ~14,900 | 0 | ~800 |
| **arXiv 论文数 (Speech)** | 2+ (Moshi, DSM) | 0 | 0 | 1 (SaShiMi, 间接) |
| **产品/API** | Unmute (开源部署) | CSM (开源模型) | Tinker (训练 API) | Sonic/Ink/Line (商业 API) |
| **企业客户** | 公开信息有限 | 公开信息有限 | UC Berkeley, Princeton, Stanford, Redwood | ServiceNow, Retell AI, Zomato, Sanas 等 |
| **推理延迟** | Moshi: 160ms 理论; Unmute TTS: ~450ms | 未公开 | N/A | "超低延迟" (未公开具体数据) |
| **全双工** | ✅ (首创) | ❌/不确定 | N/A | ❌ |
| **端侧部署** | ✅ (Pocket-TTS, MLX) | ❌ | ❌ | ✅ (Edge 框架) |

### 5.3 GitHub Stars 对比

| 团队 | 最高 Stars 项目 | Stars | 总 Speech LLM 相关 Stars |
|------|---------------|-------|------------------------|
| Sesame | csm | 14,659 | ~14,900 |
| Kyutai | moshi | 10,350 | ~21,300+ |
| Thinking Machines | tinker-cookbook | 3,433 | 0 (非 Speech LLM) |
| Cartesia | edge | 408 | ~800 |

---

## 6. 关键发现

### 发现 1: Kyutai 是全双工对话的事实标准制定者

Kyutai 的 Moshi 不仅是全双工对话的先驱,更通过一系列技术创新 (Inner Monologue, Split RVQ, Multi-stream) 定义了这个领域的技术范式。后续的 SALMONN-omni、Fun-Audio-Chat-Duplex、Step-Audio Realtime 等系统都受 Moshi 影响。同时,Kyutai 通过 DSM 框架将 Moshi 的技术形式化,使 STT/TTS 可独立部署和组合,展现了从"创新到基础设施"的完整路径。作为非营利实验室,其全面开源的策略使其成为 Speech LLM 开源社区的核心贡献者。

### 发现 2: Sesame 的 CSM 代表了"社区先行"的崛起路径

Sesame 以单一开源模型 (CSM, 14.7K stars) 在短时间内获得了超过 Moshi 的社区关注度。但与 Kyutai 的深厚技术积累相比,Sesame 的技术透明度极低 — 无论文、无公开团队、无详细技术报告。这种"社区先行、细节后补"的策略在短期内有效,但长期竞争力取决于后续技术实力的验证。值得注意的是,Sesame fork 了 Moshi 代码库,暗示其技术路线在 Kyutai 的基础上迭代。

### 发现 3: SSM vs Transformer 是语音领域的关键架构分歧

Cartesia 是唯一一家全面押注 SSM (State Space Model) 架构用于语音的公司。其创始人 Albert Gu 是 S4/Mamba 的发明者,Tri Dao 是 Flash Attention 作者。SSM 的理论优势 (线性复杂度、固定状态、长上下文) 在语音的流式处理和端侧部署中可能比 NLP 更有价值。但目前缺少独立的 benchmark 验证。如果 SSM 被证明在语音任务上优于 Transformer,Cartesia 将成为独特的技术赢家;反之,则面临重大路线风险。这是本 Session 中最值得关注的技术赌注。

### 发现 4: "对话"能力有明确的层次分化

四家公司在"对话"能力上存在明确的层次差异:
- **全双工对话** (最高): Kyutai Moshi — 可同时说话和倾听,无显式 turn
- **对话语音生成** (中): Sesame CSM — 可根据对话上下文生成自然语音,但不是完整对话系统
- **Pipeline 语音交互** (基础): Cartesia Line / Kyutai Unmute — STT→LLM→TTS 级联,响应延迟较高
- **不涉及语音** (无): Thinking Machines Tinker — 纯文本 LLM 训练

### 发现 5: 开源策略的两极分化

这 4 家公司展现了截然不同的开源策略:
- **Kyutai**: 全面开源 (非营利使命),所有核心模型+代码+权重
- **Sesame**: 选择性开源 (CSM 开源,团队和技术细节不公开)
- **Thinking Machines**: 基础设施部分开源 (Tinker cookbook/API),核心服务 SaaS
- **Cartesia**: 仅开源 SDK/工具,核心模型完全闭源 (API-only)

这反映了从"开放科学"到"商业 API"的光谱分布。Kyutai 的非营利模式使其成为社区的信任锚点,而 Cartesia 的闭源策略虽然限制了社区验证,但支撑了其商业化路径。

---

## 附: 数据来源标注

| 信息类型 | 来源 | 可信度 |
|---------|------|--------|
| Kyutai 论文技术细节 | vault 已有精读笔记 (Moshi) + GitHub README | 高 (经审阅) |
| Kyutai GitHub 数据 | GitHub API (2026-06-08 实时查询) | 高 |
| Sesame GitHub/产品 | GitHub API (2026-06-08 实时查询) | 高 |
| Sesame 技术细节 | GitHub README + 公开 demo | 中 (缺乏论文验证) |
| Thinking Machines 信息 | 官网 + GitHub (2026-06-08 查询) | 高 |
| Cartesia 产品/研究 | 官网 + GitHub + arXiv 论文 | 中-高 |
| Cartesia Sonic/Ink 技术细节 | 官方声称,未公开 benchmark | 低 (未独立验证) |
| 团队人物 | GitHub 公开成员 + 论文作者列表 + 官网 | 中 |
| 融资/商业信息 | 公开报道 + 推测 | 低 |
| Sesame/Thinking Machines 的 Speech LLM 能力评估 | 基于现有公开信息的推测 | 低 (信息有限) |
