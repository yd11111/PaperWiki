# Session 9: 学术 B — 清华大学 / 香港中文大学 / 日本学术界 (NII / 东京大学 / 名古屋大学)

> **报告范围**: Speech LLM / Omni / 全双工对话
> **调研日期**: 2026-06-08
> **数据来源**: arXiv 论文, GitHub API, vault 已有论文笔记, 公开技术报告, 实验室主页
> **5 层搜索覆盖**: Layer 1 (GitHub org) ✓ | Layer 2 (核心人搜索) ✓ | Layer 3 (arXiv affiliation) ✓ | Layer 4 (产品/竞赛) 部分 | Layer 5 (引用网络) ✓ (通过 vault 笔记)

> [!warning] 归属勘误与说明
> 1. **SpeechGPT 非清华**: 原始任务假设 SpeechGPT 出自清华,实际由 **复旦大学** 邱锡鹏团队 (Dong Zhang, Xipeng Qiu 等) 开发,arXiv:2305.11000,Apache-2.0,1.4K stars。本报告不将其计入清华产出。
> 2. **智谱/GLM-4-Voice**: 源于清华 THUDM/KEG 实验室 (唐杰),但已在 **Session 1 (智谱)** 中作为工业界团队完整覆盖。本 session 仅简要引用,不重复。
> 3. **SALMONN 系列**: 由清华 EE (Chao Zhang) 与字节跳动合作完成,已在 **Session 1 (字节豆包)** 中作为工业界团队完整覆盖。本 session 聚焦清华侧的贡献视角,不重复字节侧内容。
> 4. **CUHK vs CUHK-SZ**: 香港中文大学 (CUHK 本部) 的 Helen Meng 团队与香港中文大学 (深圳) 的 Amphion/MaskGCT 团队不同,后者已在 **Session 8** 中覆盖。
> 5. **Dong Yu 归属**: Dong Yu 曾任腾讯 AI Lab 首席科学家,近期可能转向学术界。其团队的 Covo-Audio 等工作与 CUHK 有合作关系但主要归属不确定,本报告单独列出作为 CUHK 关联组。
> 6. **日本学术界**: 原任务聚焦 Kyoto/NII,实际调研发现日本 Speech LLM 的核心产出分布在 NII (东京)、东京大学、名古屋大学/NAIST 等多个机构,已相应扩展。

---

## 1. 清华大学 — THUHCSI / OpenBMB (Zhiyong Wu 团队)

### 1.1 基本信息

| 项目 | 详情 |
|------|------|
| 机构 | 清华大学 (Tsinghua University) |
| 实验室 | THUHCSI (Tsinghua Human-Computer Speech Interaction Lab) / OpenBMB |
| 导师 | **Zhiyong Wu (吴志勇)** — 语音交互方向 lead; **Zhiyuan Liu (刘知远)** — NLP/OpenBMB 方向, VoxCPM 合作 |
| GitHub | [github.com/thuhcsi](https://github.com/thuhcsi) (94 repos); [github.com/OpenBMB](https://github.com/OpenBMB) (VoxCPM) |
| 核心学生/合作者 | Yixuan Zhou (VoxCPM 一作), Yuanyuan Wang (DualSpeechLM/UniSRM 一作), Dongchao Yang (UniAudio 系列, 与 CUHK 共同), Hui Lu (full-duplex 一作), Guoyang Zeng (VoxCPM2) |
| 研究方向 | TTS (codec LM, spontaneous style), 语音编解码, 语音安全 (VGuard), 语音数据集 (SpeechCraft), 语音交互, speech LLM (VoxCPM 扩展, DualSpeechLM) |
| 定位 | 从传统 TTS 工程能力出发,通过 VoxCPM 进入 LLM-based TTS 主赛道,并向统一 speech understanding+generation (DualSpeechLM) 和全双工对话扩展 |

### 1.2 论文时间线 (2023-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2023.10 | UniAudio | 2310.00704 | 165K 小时, 1B 参数, 11 种音频生成任务统一 LLM | **基础**: 通用音频生成 LLM (与 CUHK 合作) |
| 2023.12 | SECap | 2312.10381 | HuBERT + LLaMA 语音情感描述 | 语音理解 + LLM (AAAI 2024) |
| 2024.06 | UniAudio 1.5 | 2406.10056 | LLM-Codec: 将音频映射到 LLM text 词表 | 跨模态 in-context learning |
| 2024.08 | VoxInstruct | 2408.15676 | 多语言 codec LM, 指令到语音 | LLM-based 可控 TTS (ACM MM 2024) |
| 2024.08 | SpeechCraft | 2408.13608 | 大规模表达性语音数据集 + LLaMA 注释 | LLM-driven 数据工程 (ACM MM 2024) |
| 2024.09 | Comparing Discrete/Continuous LLM-ASR | 2409.00800 | 离散 vs 连续语音表征的 LLM-based ASR 对比 | 技术分析 (Interspeech 2024) |
| 2024.09 | SoCodec | 2409.00933 | 语义有序多流语音编解码器 | LLM-TTS 编码改进 (SLT 2024) |
| 2024.09 | CoFi-Speech | 2409.11630 | 多尺度 coarse-to-fine codec LM | TTS 质量提升 |
| 2024.12 | Codec LM Spontaneous Style TTS | 2412.01100 | LLaMA-based codec LM + delay pattern | 自发风格语音合成 |
| 2025.02 | DiffCSS | 2502.19924 | 扩散式对话语音合成, LM-based backbone | 对话 TTS (ICASSP 2025) |
| 2025.05 | Speech Speculative Decoding | 2505.15380 | 语音 AR 加速 | 推理效率 (Interspeech 2025) |
| 2025.08 | DualSpeechLM | 2508.08961 | 双 token 统一 speech understanding + generation | **核心 Speech LLM**: USTokenizer + 双路建模 (AAAI 2026, 与 CUHK 合作) |
| 2025.09 | VoxCPM | 2509.24650 | Tokenizer-free TTS, 1.8M 小时, 0.5B | **旗舰 TTS**: 开源 SOTA, 30 语言 (VoxCPM2 为 2B) |
| 2025.09 | VoxRole | 2509.03940 | 语音角色扮演 benchmark | 对话评估 |
| 2025.11 | E2E-VGuard | 2511.07099 | 端到端语音安全防护 | 安全 (NeurIPS 2025) |
| 2026.02 | UniAudio 2.0 | 2602.04683 | ReasoningCodec + 100B text + 60B audio tokens | **升级版音频基座** (与 CUHK 合作) |
| 2026.05 | Full-Duplex Spoken Dialogue | 2605.10199 | LLM 全双工路由策略: channel fusion vs cross-attention | **前沿**: 全双工对话 (与 CUHK 合作) |

### 1.3 技术栈全景

| 维度 | THUHCSI 技术栈 |
|------|---------------|
| **语音编码器** | VoxCPM: Tokenizer-free (直接 mel→TSLM); DualSpeechLM: USTokenizer (understanding-driven VQ, 冻结 LLM 反向传播训练); UniAudio: HuBERT semantic + RVQ acoustic |
| **LLM 骨干** | VoxCPM2: MiniCPM-4 (2B); DualSpeechLM: Qwen2-7B / LLaMA-3-8B; UniAudio: 1B 自研; Full-duplex: text LLM 扩展 |
| **语音解码器** | VoxCPM: Diffusion AR (TSLM + RALM 端到端); DualSpeechLM: acoustic token decoder (EnCodec + HiFi-GAN); UniAudio: multi-scale transformer + RVQ decoder |
| **对话策略** | Full-duplex paper: channel fusion (语义强但易被打断破坏) vs cross-attention routing (鲁棒但语义弱); DualSpeechLM: Chain-of-Condition (CoC) 策略 |
| **训练数据规模** | VoxCPM2: 200 万小时 (30 语言); UniAudio 2.0: 100B text + 60B audio tokens; UniAudio: 165K 小时 |
| **推理延迟** | VoxCPM2: 流式合成支持; Speech Speculative Decoding: AR 加速方案 |
| **多语言** | VoxCPM2: 30 语言; UniAudio: 多语言; DualSpeechLM: 英语为主 |
| **情感/副语言** | VoxInstruct: 指令控制情感/风格; SpeechCraft: 大规模情感标注; SECap: 情感描述 |

### 1.4 架构演进

```
UniAudio (通用音频 LM, 165K hrs, 1B, 2023.10)
    ↓ 编码器改进
SoCodec (语义有序多流, 2024.09) + CoFi-Speech (多尺度, 2024.09)
    ↓
UniAudio 1.5 (LLM-Codec, 跨模态 in-context, 2024.06)
    ↓ 统一理解+生成
DualSpeechLM (USTokenizer 输入 + acoustic token 输出, AAAI 2026, 2025.08)
    ↓ 基座升级
UniAudio 2.0 (ReasoningCodec, 100B+60B tokens, 2026.02)

并行 TTS 线:
VoxInstruct (指令控制 codec LM, 2024.08) → Spontaneous Style TTS (2024.12)
    ↓ 规模化
VoxCPM (0.5B, 1.8M hrs, tokenizer-free, 2025.09) → VoxCPM2 (2B, 2M hrs, 30 语言, MiniCPM-4 backbone)
    ↓ 产品化
VoxCPM2: 27K GitHub stars, Apache-2.0, 商用免费

全双工对话线:
Full-Duplex Spoken Dialogue (channel fusion vs cross-attention routing, 2026.05)
```

### 1.5 独特技术赌注

1. **Tokenizer-free TTS (VoxCPM)**: 跳过离散 speech tokenizer 步骤,直接从文本到连续语音表征,用 TSLM (Text-Semantic LM) + RALM (Residual Acoustic Model) 的层级建模替代传统 tokenize→AR→detokenize 管线。理论上避免了 tokenizer 引入的信息损失。
2. **USTokenizer (DualSpeechLM)**: "理解驱动"的 speech tokenizer — 不用 ASR loss 或自监督目标训练,而是冻结 LLM 用 LLM 的梯度反向传播优化 VQ encoder,使 speech tokens 天然对齐 LLM 输入空间。这是一种"以终为始"的 tokenizer 设计。
3. **Input/Output Token 解耦**: DualSpeechLM 首次系统性地将 LLM 的输入 token (semantic, 用于理解) 和输出 token (acoustic, 用于生成) 分离,打破了同类系统 (SpeechGPT, SPIRIT-LM, Moshi) 使用同一类 token 的惯例。
4. **ReasoningCodec (UniAudio 2.0)**: 将音频编码分为 "reasoning tokens" (高层语义,对齐文本) 和 "reconstruction tokens" (声学细节,重建波形),在 tokenizer 层面实现理解-生成的分工。

### 1.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| OpenBMB/VoxCPM | **27,554** | Tokenizer-free TTS, 30 语言, 2B 参数, Apache-2.0 商用免费 |
| yangdongchao/UniAudio | 604 | 通用音频生成 LLM (与 CUHK 合作) |
| yangdongchao/UniAudio2 | 212 | UniAudio 2.0 (ReasoningCodec) |
| thuhcsi/Crystal | 230 | 多语言 TTS 合成引擎 (C++) |
| thuhcsi/SpeechCraft | 192 | 大规模表达性语音数据集 |
| thuhcsi/SECap | 180 | 语音情感描述 (AAAI 2024) |
| thuhcsi/VoxInstruct | 100 | 指令到语音 codec LM |

### 1.7 清华其他相关实验室 (简要引用)

**THUDM / KEG Lab (唐杰)** → 智谱 AI:
- GLM-4-Voice (2024.12): 合成交替数据 + Streaming Thoughts,3.2K stars → **详见 Session 1 (智谱)**
- GLM-TTS (2025.12): GRPO RL 对齐,1.0K stars → **详见 Session 1 (智谱)**

**清华 EE (Chao Zhang)** → 与字节合作:
- SALMONN 系列 (2023.10-2026): dual encoder + Q-Former → codec-free 全双工 → ELLSA 四模态 → **详见 Session 1 (字节豆包)**
- ELLSA (ICLR 2026): SA-MoE 架构,首个 listen+look+speak+act 全双工模型
- 贡献视角: Chao Zhang 团队提供了 audio understanding 的学术基础,ByteDance 提供了工程化和产品化能力

**THUNLP (刘知远 NLP 组)**: 主要通过 OpenBMB/VoxCPM 与 THUHCSI 合作,Zhiyuan Liu 是 VoxCPM 合作导师

### 1.8 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) VoxCPM2 是国内 TTS 领域 stars 数第二高的开源项目 (仅次于 CosyVoice 21K),27K stars + Apache-2.0 商用免费,生态影响力巨大; (2) 技术路线多元且前沿: tokenizer-free (VoxCPM), input/output 解耦 (DualSpeechLM), reasoning codec (UniAudio 2.0); (3) 与 CUHK Helen Meng 团队的深度合作形成跨校优势; (4) 完整的数据-工具链: SpeechCraft 数据集 + VoxInstruct 可控生成 + SECap 情感理解 |
| **短板** | (1) VoxCPM 系列偏向 TTS,在 Speech LLM 对话方向的系统级产出不如 GLM-4-Voice 或 SALMONN; (2) 全双工对话仅有 2026.05 的路由策略研究论文,尚无完整系统; (3) DualSpeechLM 仅在 LibriSpeech/SmolLM2 规模验证,未扩展到大模型/大数据; (4) 实验室内部多条技术路线 (tokenizer-free vs discrete token vs continuous) 之间的收敛方向不明确 |
| **影响力** | VoxCPM2 通过 27K stars 和商用免费策略已成为国内 TTS 开源生态的核心之一; DualSpeechLM 的 input/output 解耦思路被后续工作引用; UniAudio 系列在学术界影响显著 |
| **未来方向推测** | (1) VoxCPM 可能向 Speech LLM 方向扩展 (VoxCPM-Chat?); (2) DualSpeechLM 可能扩展到更大模型规模; (3) full-duplex 路由策略可能集成到完整系统; (4) 与 OpenBMB/MiniCPM 生态的更深整合 |

---

## 2. 香港中文大学 (CUHK) — Helen Meng 团队

### 2.1 基本信息

| 项目 | 详情 |
|------|------|
| 机构 | 香港中文大学 (The Chinese University of Hong Kong) |
| 实验室 | Department of Systems Engineering & Engineering Management; Microsoft-CUHK Joint Lab; Tsinghua-CUHK Joint Research Centre |
| 导师 | **Helen Meng (蒙美玲)** — IEEE Fellow, Patrick Huen Wing Ming Professor; MIT 博士 (SB/SM/PhD); 1998 年起在 CUHK |
| 核心教员 | **Xixin Wu (吴西鑫)** — 语音生成方向核心, DualSpeechLM/UniAudio/CoFi-Speech 的共同通讯; **Dongchao Yang (杨东超)** — UniAudio 系列一作 |
| GitHub | [yangdongchao](https://github.com/yangdongchao) (UniAudio 系列); 无独立实验室 org |
| 核心学生/合作者 | Yuanyuan Wang (DualSpeechLM, UniSRM), Dingdong Wang (EmotionThinker), Lingwei Meng (MELLE), Haohan Guo (CoFi-Speech), Hui Lu (full-duplex) |
| 研究方向 | 语音生成 (LLM-based TTS, codec LM), 音频理解 (UniAudio), 语音情感 (EmotionThinker), 全双工对话, 语音安全, 语音分离 |
| 定位 | 华语语音学术界的旗舰实验室; 从多语言语音处理传统出发,近年全面转向 LLM-based 语音生成与理解,产出密度极高 |

### 2.2 论文时间线 (2023-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2023.01 | InstructTTS | 2301.13662 | 自然语言风格提示的表达性 TTS | 早期 LLM-TTS 交叉 |
| 2023.09 | Multi-Scale Acoustic Prompts | 2309.11977 | 改进 VALL-E 类 codec LM 零样本 TTS | LLM-TTS 改进 (ICASSP 2024) |
| 2023.10 | UniAudio | 2310.00704 | 165K 小时, 1B, 11 种音频生成任务 | **通用音频生成基座** (与 Tsinghua 合作) |
| 2024.06 | UniAudio 1.5 | 2406.10056 | LLM-Codec: 冻结 LLM 的 few-shot 音频学习 | 跨模态 in-context learning |
| 2024.06 | CoLM-DSR | 2406.08336 | Neural codec LM 用于构音障碍语音重建 | 医疗应用 (Interspeech 2024) |
| 2024.07 | MELLE | 2407.08551 | 无 VQ 的连续值 AR TTS, mel 直接预测 | **连续 AR TTS** (ACL 2025, 与 Microsoft 合作) |
| 2024.07 | Spontaneous Style TTS | 2407.13509 | 基于 LM 的自发风格 TTS | 对话 TTS (Interspeech 2024) |
| 2024.08 | SimpleSpeech 2 | 2408.13893 | Flow-based scalar latent transformer | TTS 架构探索 |
| 2024.09 | SoCodec | 2409.00933 | 语义有序多流编解码器 | LLM-TTS 编码器 (SLT 2024) |
| 2024.09 | CoFi-Speech | 2409.11630 | Coarse-to-fine 多尺度 codec LM | TTS 质量提升 |
| 2024.09 | SongCreator | 2409.06029 | 双序列 LM 歌曲生成 | 音乐生成 |
| 2024.09 | MT-LLM | 2409.08596 | 多说话人 ASR + LLM | 语音理解 (ICASSP 2025) |
| 2025.03 | UniSep | 2503.23762 | LLM-based 音频分离 | 音频理解扩展 (ICME 2025) |
| 2025.08 | DualSpeechLM | 2508.08961 | USTokenizer + 双 token 统一模型 | **核心 Speech LLM** (AAAI 2026, 与 Tsinghua 合作) |
| 2025.09 | Bridging LM-TTS gap | 2509.17021 | 训练-推理差异缓解 | TTS 改进 |
| 2026.01 | EmotionThinker | 2601.15668 | GRPO-PTR + CoT 语音情感推理 | **语音理解 SOTA** (ICLR 2026 Oral) |
| 2026.02 | UniAudio 2.0 | 2602.04683 | ReasoningCodec, 100B text + 60B audio tokens | **升级版音频基座** (与 Tsinghua 合作) |
| 2026.05 | Full-Duplex Spoken Dialogue | 2605.10199 | Channel fusion vs cross-attention routing | **前沿**: 全双工路由 (与 Tsinghua 合作) |

### 2.3 技术栈全景

| 维度 | CUHK Helen Meng 团队技术栈 |
|------|--------------------------|
| **语音编码器** | UniAudio: HuBERT semantic + 自研 RVQ acoustic; DualSpeechLM: USTokenizer (understanding-driven VQ); MELLE: 无 tokenizer (连续 mel); CoFi-Speech: 多尺度 codec; UniAudio 2.0: ReasoningCodec |
| **LLM 骨干** | UniAudio: 1B 自研; DualSpeechLM: Qwen2-7B / LLaMA-3-8B; MELLE: 自研 AR 模型; EmotionThinker: Speech LLM (具体未公开) |
| **语音解码器** | UniAudio: multi-scale transformer + vocoder; MELLE: 连续 mel → vocoder (单阶段); CoFi-Speech: stack-of-scale generation |
| **对话策略** | Full-duplex (2026): channel fusion vs cross-attention 路由策略; DualSpeechLM: Chain-of-Condition |
| **训练数据规模** | UniAudio: 165K 小时; UniAudio 2.0: 100B text + 60B audio tokens; MELLE: 未公开 |
| **推理延迟** | MELLE: 单阶段连续 AR (vs 两阶段 VQ+decode); 具体延迟数据有限 |
| **多语言** | UniAudio: 多语言支持; MELLE: 英语为主 |
| **情感/副语言** | EmotionThinker (ICLR 2026 Oral): GRPO-PTR RL + prosody-enhanced SER; SECap: 情感描述; InstructTTS: 自然语言风格控制 |

### 2.4 架构演进

```
InstructTTS (NL 风格提示 TTS, 2023.01)
    ↓ LLM-based TTS 基础
Multi-Scale Acoustic Prompts (VALL-E 改进, 2023.09)
    ↓
UniAudio (通用音频 LM, 165K hrs, 11 tasks, 2023.10)
    ↓ 编码器改进
SoCodec (语义有序多流, 2024.09) + CoFi-Speech (多尺度, 2024.09)
    ↓
UniAudio 1.5 (LLM-Codec, 跨模态 in-context, 2024.06)
    ↓ 解耦理解与生成
MELLE (连续 mel AR, 无 VQ, ACL 2025, 2024.07) ──── 连续表征路线
DualSpeechLM (USToken in + acoustic out, AAAI 2026, 2025.08) ──── 离散双 token 路线
    ↓ 统一升级
UniAudio 2.0 (ReasoningCodec, 100B+60B tokens, 2026.02)
    
情感/推理线:
SECap (HuBERT+LLaMA 情感描述, 2023.12)
    ↓
EmotionThinker (GRPO-PTR + CoT, ICLR 2026 Oral, 2026.01)

全双工线:
Full-Duplex Spoken Dialogue (routing 策略研究, 2026.05)
```

### 2.5 独特技术赌注

1. **连续 AR TTS (MELLE)**: 完全去除 VQ,直接在连续 mel 空间自回归预测,用 regression loss + spectrogram flux loss + variational inference 替代 cross-entropy。避免 VQ 引入的信息损失和 codebook collapse 问题。
2. **ReasoningCodec (UniAudio 2.0)**: 在 tokenizer 层面分离 "reasoning" (高层语义规划) 和 "reconstruction" (低层声学) 两类 token,使一个统一模型同时具备理解 (用 reasoning tokens) 和生成 (用 reconstruction tokens) 能力。
3. **EmotionThinker 的 GRPO-PTR**: 将语音情感识别重构为推理问题,用 Chain-of-Thought 生成可解释的情感判断。Progressive Trust-aware Reasoning Reward 动态调整推理奖励的权重,防止 reward hacking。ICLR 2026 Oral 级别。
4. **CoFi 多尺度编码**: 在 codec 层面引入不同时间分辨率的 token 序列,粗粒度控制全局结构,细粒度控制声学细节。解决标准 codec LM 的 recency bias 问题。

### 2.6 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| yangdongchao/UniAudio | 604 | 通用音频生成 LLM |
| inclusionAI/Ming-UniAudio | 448 | Ming-UniAudio: Speech LLM (理解+生成+编辑) |
| yangdongchao/UniAudio2 | 212 | UniAudio 2.0 (ReasoningCodec) |
| dingdongwang/EmotionThinker | (新) | ICLR 2026 Oral |

**注意**: DualSpeechLM、CoFi-Speech、MELLE 等论文的代码开源程度有限。UniAudio 系列代码已开源。

### 2.7 CUHK 关联 — Dong Yu 团队

Dong Yu 曾任腾讯 AI Lab 首席科学家,近期可能转向学术界 (具体归属公开信息有限)。其团队在 Speech LLM 方向有大量产出,与 CUHK 存在合作关系但非直接 CUHK 编制。

**核心论文 (2024-2026)**:

| 时间 | 论文 | arXiv ID | 核心贡献 |
|------|------|----------|---------|
| 2024.06 | GPST | 2406.00976 | 层级 transformer 语音 LM (ACL 2024) |
| 2024.09 | Preference Alignment for LM-TTS | 2409.12403 | 偏好对齐改进 TTS |
| 2025.02 | Balancing Understanding & Generation | 2502.16897 | 持续预训练统一 codec Speech LLM (ASRU 2025) |
| 2025.08 | Audio-Thinker | 2508.08039 | RL 引导音频推理 |
| 2025.10 | U-Codec | 2510.16718 | 5Hz 超低帧率 neural codec |
| 2025.10 | VCB Bench | 2510.11098 | 中文语音聊天评估 benchmark |
| 2025.12 | AzeroS | 2601.06086 | 自生成指令免调优 speech LLM |
| 2026.02 | Covo-Audio | 2602.09823 | 7B 端到端 LALM, 全双工 |

**Covo-Audio 技术特点**:
- 7B 参数端到端 LALM,直接处理连续音频输入输出
- 三个变体: Covo-Audio (基座), Covo-Audio-Chat (对话), Covo-Audio-Chat-FD (全双工)
- "Intelligence-Speaker Decoupling": 分离对话智能与语音渲染,降低部署成本
- 26 位作者,团队规模大

### 2.8 判断

| 维度 | 评估 |
|------|------|
| **优势** | (1) 学术产出密度极高: 2023-2026 年间在 Speech LLM 相关方向发表 20+ 篇论文,覆盖生成/理解/情感/全双工/安全全链路; (2) 技术路线最多元: 同时探索离散 token (UniAudio, CoFi-Speech), 连续表征 (MELLE), 解耦设计 (DualSpeechLM, ReasoningCodec); (3) EmotionThinker 获得 ICLR 2026 Oral,证明顶会竞争力; (4) 与 Tsinghua THUHCSI 的深度合作形成互补 (CUHK 偏方法论, Tsinghua 偏工程化) |
| **短板** | (1) 开源影响力不足: 最高 stars 项目 UniAudio 仅 604 stars,与工业界 (CosyVoice 21K, VoxCPM 27K) 差距大; (2) 无完整的 speech-to-speech 对话系统: 全双工仅有路由策略研究,无端到端系统; (3) 研究碎片化: 多条技术路线并行但缺乏统一框架整合; (4) 评估数据有限: 很多工作仅在 LibriSpeech 等英语数据上验证 |
| **影响力** | 学术引用高; UniAudio 是通用音频生成 LLM 的早期标杆; MELLE 的连续 AR 思路被 LatentLM, CLEAR 等后续工作发展; DualSpeechLM 的 input/output 解耦是理论贡献 |
| **未来方向推测** | (1) UniAudio 3.0 可能整合 DualSpeechLM 和 ReasoningCodec; (2) full-duplex 路由策略可能发展为完整系统; (3) EmotionThinker 的 RL-for-audio-reasoning 可能扩展到更多任务; (4) 与 Dong Yu 团队的关系可能深化 |

---

## 3. 日本学术界 — NII / 东京大学 / 名古屋大学

> [!info] 说明
> 日本学术界在 Speech LLM / 全双工对话方向的直接产出有限,但在基础设施 (deepfake 检测、语音编解码、对话语料) 方面有独特贡献。以下按实验室分别介绍,最后给出整体判断。

### 3.1 NII — Yamagishi Lab (Junichi Yamagishi)

#### 基本信息

| 项目 | 详情 |
|------|------|
| 机构 | National Institute of Informatics (NII), Tokyo |
| 实验室 | Yamagishi Lab (Digital Content and Media Sciences Research Division) |
| 导师 | **Junichi Yamagishi** — Professor, Ph.D., 主要研究语音合成、说话人验证、媒体取证 |
| GitHub | [github.com/nii-yamagishilab](https://github.com/nii-yamagishilab) (65 repos) |
| 核心成员 | 博士后和博士生 (具体名单未公开) |
| 研究方向 | 语音合成、说话人验证、**Deepfake 检测** (ASVspoof 系列的核心组织者)、语音隐私 |
| 定位 | 全球 deepfake/spoofing 检测领域的学术领导者; TTS 方面有传统积累但非 Speech LLM 主力 |

#### 与 Speech LLM 相关论文 (2024-2026)

| 时间 | 论文 | arXiv ID | 与 Speech LLM 关系 |
|------|------|----------|-------------------|
| 2023.12 | ZMM-TTS | 2312.14398 | 零样本多语言 TTS, 自监督离散表征 |
| 2025.03 | QualiSpeech | 2503.20290 | 用 auditory LLM 做语音质量评估 benchmark (与 Tsinghua/ByteDance 合作, SALMONN 系列衍生) |
| 2025.07 | MIDI-VALLE | 2507.08530 | VALL-E 框架迁移到钢琴演奏合成 (codec LM for music) |
| 2026.02 | Deepfake Word Detection | 2602.22658 | 微调 Whisper 做 next-token prediction 检测合成语音 |
| 2026.03 | RL for Deepfake Detection | 2603.02914 | 借鉴 LLM 的 GRPO 微调语音 deepfake 检测器 |

#### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| project-NN-Pytorch-scripts | 362 | 神经网络语音处理脚本 |
| multi-speaker-tacotron | 266 | 多说话人 Tacotron (ICASSP 2020) |
| ZMM-TTS | 184 | 零样本多语言 TTS |
| mos-finetune-ssl | 111 | SSL-based MOS 评估 |
| self-attention-tacotron | 114 | 增强型 Tacotron TTS |
| AntiDeepfake | 52 | SSL-based deepfake 检测 |
| MIDI-VALLE | 9 | VALL-E for piano |

#### 判断

Yamagishi Lab 的核心竞争力在 deepfake 检测和语音隐私 (ASVspoof 5, VoicePrivacy Challenge),而非 Speech LLM。在 TTS 方面有传统积累 (Tacotron, HMM-TTS 时代的贡献),但向 LLM-based 语音生成的转型较慢。QualiSpeech 是与 SALMONN 团队 (Tsinghua/ByteDance) 的合作成果,代表了从评估侧介入 Speech LLM 生态的策略。

### 3.2 东京大学 — Takamichi / Saruwatari 实验室

#### 基本信息

| 项目 | 详情 |
|------|------|
| 机构 | 东京大学 (The University of Tokyo) |
| 实验室 | Saruwatari Lab (猿渡研究室), Graduate School of Information Science and Technology |
| 核心教员 | **Shinnosuke Takamichi (高道慎之介)** — 语音合成/分析方向 lead; **Hiroshi Saruwatari (猿渡洋)** — 实验室主任; **Yuki Saito** — TTS/VC |
| GitHub | 无统一 org; 个人仓库分散 |
| 研究方向 | TTS, 语音编解码分析, 日语语音资源, 对话语音合成, 音声分析 |
| 定位 | 日本 Speech LLM 相关研究的最活跃学术组; 独特贡献在 speech tokenization 理论分析和日语对话语料 |

#### 与 Speech LLM 相关论文 (2023-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2023.01 | Learning to Speak from Text | 2301.12596 | 零样本多语言 TTS, 利用多语言 LM 的跨语言迁移 | LLM 知识迁移到 TTS |
| 2023.05 | ChatGPT-EDSS | 2305.13724 | ChatGPT 上下文理解驱动的共情对话语音合成 | LLM-driven TTS |
| 2023.06 | How GSLM Encodes Noisy Speech | 2306.00697 | GSLM 对含噪语音的编码行为分析 | Speech LM 理论分析 |
| 2023.09 | Do learned speech symbols follow Zipf's law? | 2309.09690 | 离散语音符号的统计语言学分析 | Speech tokenization 理论 |
| 2024.04 | **RALL-E** | 2404.03204 | Chain-of-Thought 改进 VALL-E 鲁棒性 (WER 5.6→2.5) | **Codec LM 改进** (与 Microsoft/U Tokyo 合作) |
| 2024.07 | **J-CHAT** | 2407.15828 | 76,000 小时日语对话语料, 自动化构建 | **关键基础设施**: 日语 spoken dialogue LM 训练资源 |
| 2024.08 | SaSLaW | 2408.06858 | 音视频自我中心对话语料 | 环境感知 TTS |
| 2024.09 | BigCodec | 2409.05377 | 159M 参数语音 codec, 1.04 kbps | Codec 基础设施 |
| 2025.05 | Speech Tokenization for SLM | 2505.17446 | 分段和词表大小对 Speech LM 的影响分析 | **Speech LM 理论** |
| 2025.09 | Analysing Neural Audio Codecs | 2509.01390 | Neural codec 的统计和语言学属性分析 | Codec 理论分析 |

#### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| (J-CHAT 语料) | (HuggingFace) | 76,000 小时日语对话, 开源 |
| RALL-E | (公开代码) | CoT 改进 VALL-E |
| BigCodec | (公开代码) | 159M codec |

#### 判断

Takamichi/Saruwatari 实验室的独特价值在于 **理论分析** 和 **日语资源**: 他们系统性地研究了 speech tokenization 的语言学属性 (Zipf's law, entropy, 分段粒度对 SLM 的影响),这些分析对 Speech LLM 的 tokenizer 设计有基础指导意义。J-CHAT (76K 小时日语对话) 是日语 spoken dialogue LM 训练的关键资源。RALL-E 的 CoT prompting 改进 codec LM 鲁棒性的思路具有通用价值。但该实验室尚未构建完整的 Speech LLM 系统。

### 3.3 名古屋大学 / NAIST — Tomoki Toda

#### 基本信息

| 项目 | 详情 |
|------|------|
| 机构 | 名古屋大学 (Nagoya University) / 原 NAIST |
| 导师 | **Tomoki Toda (戸田智基)** — 语音变换领域的全球领导者, IEEE Fellow |
| 研究方向 | 语音变换 (VC), ASR, 病理语音, 音质评估, 异常声音检测 |
| 定位 | 传统语音信号处理强校; 在 Speech LLM 方向的直接投入有限 |

#### 与 Speech LLM 相关论文 (2024-2026)

| 时间 | 论文 | arXiv ID | 核心贡献 | 与 Speech LLM 关系 |
|------|------|----------|---------|-------------------|
| 2025.05 | CMT-LLM | 2506.12059 | 上下文多说话人 ASR + LLM | LLM-based ASR (Interspeech 2025) |
| 2025.10 | SpeechLMScore | 2510.00639 | 用 Speech LM 做无参考语音严重度评估 | SLM 应用 |
| 2025.06 | PMF-CEC | 2506.11064 | 音素增强多模态 ASR 纠错 (对比 LLM 方法) | LLM-adjacent (TASLP 2025) |
| 2026.06 | Zero-shot Cross-lingual SER | 2606.06200 | 情感判别表征学习 | Interspeech 2026 |

#### 判断

Toda 实验室的核心强项在语音变换 (voice conversion) 和语音质量评估,在这些领域具有全球领先地位。但在 Speech LLM / 端到端对话方向,其投入和产出明显不足。近期论文 (CMT-LLM, SpeechLMScore) 显示了开始将 LLM 融入传统任务 (多说话人 ASR, 语音评估) 的趋势,但尚未产出独立的 Speech LLM 系统。

### 3.4 日本学术界整体判断

**核心发现**: 日本学术界在 Speech LLM / 全双工对话方向 **缺乏系统级产出**。三个代表性实验室均未构建完整的端到端 speech-to-speech 对话系统或 speech language model。相比中国 (清华、CUHK) 和美国 (CMU/UT Austin) 的学术界在 Speech LLM 方向的密集投入,日本学术界的转型明显滞后。

**独特贡献方向**:
- **Deepfake 检测 + 语音安全** (Yamagishi/NII): ASVspoof 系列是全球标准,VoicePrivacy Challenge 引领隐私保护研究
- **Speech tokenization 理论分析** (Takamichi/U Tokyo): Zipf's law, codec 语言学属性,分段粒度影响 — 这些理论工作为 Speech LLM 的 tokenizer 设计提供基础
- **日语对话资源** (Takamichi/U Tokyo): J-CHAT 76K 小时是日语 SLM 训练的唯一大规模开源资源
- **Codec LM 改进** (Takamichi + Microsoft): RALL-E 的 CoT prompting 思路

**滞后原因推测**:
1. 日本语音研究传统强在信号处理和统计建模,向深度学习/LLM 的范式转换慢于中美
2. 计算资源限制: 日本学术界缺乏大规模 GPU 集群来训练 B 级参数的 Speech LLM
3. 研究人口规模: 相比中国学术界的巨量博士生产出,日本实验室规模较小
4. 产业-学术循环弱: 日本缺乏类似中国 (清华→智谱, Tsinghua EE→ByteDance SALMONN) 的产学转化路径

---

## 4. 横向对比矩阵

### 4.1 机构对比

| 维度 | 清华 THUHCSI/OpenBMB | CUHK Helen Meng | NII Yamagishi | U Tokyo Takamichi | Nagoya Toda |
|------|---------------------|-----------------|---------------|-------------------|-------------|
| **核心方向** | TTS (VoxCPM) + 统一模型 (DualSpeechLM) | 通用音频 LM (UniAudio) + 情感 (EmotionThinker) | Deepfake 检测 + 语音隐私 | Speech tokenization 理论 + 日语资源 | VC + 语音质量评估 |
| **Speech LLM 系统** | 无完整系统 (有全双工研究) | 无完整系统 (有路由策略研究) | 无 | 无 | 无 |
| **技术路线** | Tokenizer-free (VoxCPM) + 离散解耦 (DualSpeechLM) | 多路线并行 (离散/连续/解耦/reasoning) | 传统 TTS + deepfake | 分析+语料+改进 (RALL-E) | 传统 VC + LLM-adjacent |
| **最高影响力论文** | VoxCPM2 (27K stars) | EmotionThinker (ICLR 2026 Oral) | ASVspoof 系列 | RALL-E / J-CHAT | Voice conversion 系列 |
| **开源最高 Stars** | VoxCPM: 27,554 | UniAudio: 604 | project-NN-scripts: 362 | J-CHAT (HF) | N/A |
| **产出密度 (2024-2026)** | ~15 篇 SpeechLM 相关 | ~20 篇 SpeechLM 相关 | ~3 篇 SpeechLM 相关 | ~6 篇 SpeechLM 相关 | ~3 篇 SpeechLM 相关 |
| **全双工能力** | 路由策略研究 (2026) | 路由策略研究 (2026, 合作) | 无 | 无 | 无 |
| **多语言** | VoxCPM2: 30 语言 | 英语为主 | ZMM-TTS: 多语言 | 日语为主 | 日语/英语 |
| **与工业界关系** | OpenBMB/ModelBest 产品化 | 与 Microsoft 合作 (MELLE) | 企业合作 | 与 Microsoft 合作 (RALL-E) | 有限 |

### 4.2 技术路线对比

| 技术选择 | 清华 THUHCSI | CUHK Helen Meng | 日本学术界 |
|---------|------------|-----------------|-----------|
| **Tokenizer 哲学** | Tokenizer-free (VoxCPM) 或 Understanding-driven (DualSpeechLM) | ReasoningCodec (理解+重建分离) 或 无 VQ (MELLE) | 理论分析 (Zipf/Heaps/entropy 视角) |
| **LLM 融合方式** | Backbone 扩展 (MiniCPM-4 → VoxCPM2) | 多种: 自研1B / Qwen2-7B / LLaMA-3-8B | 间接: RALL-E (CoT), GSLM 分析 |
| **音频表征** | 连续 (VoxCPM) + 离散 (DualSpeechLM USToken) | 离散 (UniAudio RVQ) + 连续 (MELLE mel) + 混合 (ReasoningCodec) | 离散 (BigCodec, RALL-E) |
| **训练规模** | VoxCPM2: 2M hrs; UniAudio 2.0: 100B+60B tokens | UniAudio: 165K hrs; DualSpeechLM: 小规模 | 小规模 (学术级) |
| **产品化路径** | VoxCPM → ModelBest API | 纯学术 | 纯学术 |

---

## 5. 关键发现

### 发现 1: 清华三路并进,VoxCPM 异军突起

清华在 Speech LLM 方向形成了三个独立但有交叉的团队:
- **THUDM/KEG → 智谱** (GLM-4-Voice): 数据工程创新 (合成交替数据),已完全产业化
- **Tsinghua EE → ByteDance** (SALMONN): 从音频理解到全双工到 embodied AI (ELLSA),学术-工业深度绑定
- **THUHCSI/OpenBMB** (VoxCPM): 开源 TTS 超级项目 (27K stars),正从 TTS 向 Speech LLM 扩展

VoxCPM2 的 27K GitHub stars + Apache-2.0 + 30 语言商用免费,使其在开源 TTS 生态中仅次于 CosyVoice。但 THUHCSI 在 Speech LLM 对话系统方向的系统级产出仍弱于其在 TTS 方向的积累。

### 发现 2: CUHK 是方法论高地,但工程化不足

Helen Meng 团队在 2023-2026 年间产出了 20+ 篇 Speech LLM 相关论文,覆盖离散 token、连续表征、编码器设计、情感推理等多条前沿路线,包括 ICLR 2026 Oral (EmotionThinker)、ACL 2025 (MELLE)、AAAI 2026 (DualSpeechLM) 等顶会论文。但这些工作更多是方法论探索,缺乏集成为完整 Speech LLM 系统的工程化步骤。相比之下,与 CUHK 存在合作关系的 Dong Yu 团队 (Covo-Audio) 则在系统构建方面更完整。

### 发现 3: 日本学术界在 Speech LLM 方向存在明显代际滞后

三个代表性日本语音实验室 (NII, U Tokyo, Nagoya) 均未产出独立的 Speech LLM 系统或端到端对话模型。与中国 (清华 3 个团队 + CUHK) 和美国 (CMU, MIT, UT Austin) 学术界的密集 Speech LLM 研究相比,日本学术界在这一方向的投入明显不足。不过,日本研究者在 **基础设施层** (deepfake 检测标准 ASVspoof、speech tokenization 理论分析、日语对话语料 J-CHAT 76K hrs、codec 语言学属性分析) 的贡献不可替代,这些工作虽不直接构建 Speech LLM,但为整个领域提供了评估工具和理论基础。

### 发现 4: 学术界 vs 工业界的全双工鸿沟

本 session 覆盖的所有学术组 (包括清华 THUHCSI 和 CUHK) 均未产出完整的全双工语音对话系统。清华 THUHCSI + CUHK 的全双工研究 (2026.05) 停留在路由策略分析层面,未构建端到端系统。相比之下,工业界 (Moshi, SALMONN-omni, Fun-Audio-Chat, StepAudio 2.5 Realtime) 已有多个全双工系统落地。学术界在全双工方向主要提供理论框架和分析 (如 LSLM 的 FDM 形式化, 本文的 routing 策略对比),而非系统工程。

唯一例外是 ELLSA (Tsinghua EE + ByteDance),但其更偏向 embodied AI 场景而非纯语音对话。

### 发现 5: Tsinghua-CUHK 轴线是华语 Speech LLM 学术界的核心

清华 THUHCSI 与 CUHK Helen Meng 团队的合作覆盖了 UniAudio 系列 (2023-2026)、DualSpeechLM (AAAI 2026)、UniSRM、Full-duplex dialogue (2026) 等多个核心项目。Dongchao Yang、Yuanyuan Wang 等核心学生同时出现在两校的论文中。加上 Tsinghua-CUHK Joint Research Centre (2006 年成立) 的制度支持,这条合作轴线是华语 Speech LLM 学术研究的核心产出通道。

---

## 附: 数据来源标注

| 信息类型 | 来源 | 可信度 |
|---------|------|--------|
| 论文技术细节 | vault 已有精读笔记 (SALMONN, GLM-4-Voice, GLM-TTS, DualSpeechLM, UniAudio, ELLSA, LSLM, Dynamic-SUPERB) | 高 (经审阅) |
| GitHub stars/repos | GitHub API (2026-06-08 实时查询) | 高 |
| arXiv 论文列表 | arXiv 搜索 + WebFetch (2026-06-08 查询) | 高 |
| 实验室信息 | 官方网站 (NII, CUHK) + GitHub org 页面 | 中-高 |
| 作者归属 | 论文 PDF + GitHub repo + 公开资料 | 中 (部分论文未公开具体归属) |
| SpeechGPT 归属修正 | arXiv:2305.11000 作者信息 + GitHub repo (0nutation/SpeechGPT) | 高 (Fudan/Xipeng Qiu 确认) |
| Dong Yu 归属 | 公开信息有限,无法确认当前 CUHK 编制 | 低-中 |
| 日本学术界评估 | arXiv 作者搜索 + 实验室网站 + GitHub | 中 (可能遗漏非英文发表的工作) |
| VoxCPM 27K stars | GitHub API | 高 |
