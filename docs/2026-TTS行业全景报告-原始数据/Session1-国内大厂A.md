# Session 1: 阿里双团队 + 字节 + 阶跃 + 智谱

> 调研日期: 2026-06-04
> 方法: 5层搜索法 (组织搜索 + 核心人搜索 + affiliation搜索 + 产品/竞赛反推 + 引用网络)

---

## 团队1a: 通义语音 / FunAudioLLM (阿里巴巴)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 阿里巴巴 - 通义实验室 - 语音团队 |
| **团队名** | FunAudioLLM |
| **GitHub org** | [FunAudioLLM](https://github.com/FunAudioLLM), [modelscope](https://github.com/modelscope) |
| **HuggingFace org** | [FunAudioLLM](https://huggingface.co/FunAudioLLM) |
| **核心人物** | **叶杰平 (Jieping Ye)** — VP, 鄢志杰离职后直接管理; **张士良 (Shiliang Zhang)** — 技术Lead; **杜智昊 (Zhihao Du)** — CosyVoice系列一作 |
| **人员变动** | 前负责人 **鄢志杰 (Zhijie Yan)** 2025.02 离职加入腾讯 |
| **外部合作者** | **武执政 (Zhi-Zheng Wu)** — 港中文(深圳)副教授, 非阿里员工 |
| **产品线** | 通义千问语音版, FunASR (开源ASR), CosyVoice API |

### 论文时间线 (2024-2026, 35篇, 8篇顶会)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2026.06 | **UniVocal** — 统一语音理解+生成 | preprint | 统一模型 |
| 2026.06 | **MELA-TTS** — 多情感长音频TTS | preprint | 情感TTS |
| 2026.05 | **VARSTok** — 变长语义token (AAAI 2026 Oral) | AAAI 2026 | Tokenizer |
| 2025.12 | **Fun-Audio-Chat** — 双分辨率全双工对话 | preprint | 全双工 |
| 2025.12 | **FunMusic** — 音乐生成 | preprint | 音乐 |
| 2025.05 | **CosyVoice 3** — 1.5B/1M小时/可微分reward | preprint | TTS旗舰 |
| 2025.05 | **ThinkSound** — CoT引导音频生成 | preprint | 音频生成 |
| 2024.12 | **CosyVoice 2** — chunk-aware FM流式 | preprint | 流式TTS |
| 2024.07 | **CosyVoice** — 监督语义token+FM | preprint | TTS |
| 2024.07 | **SenseVoice** — 非自回归多语言理解 | preprint | ASR |
| 2024.07 | **FunAudioLLM** — 语音理解+生成双模型报告 | preprint | 系统 |
| ... | (另有25+篇覆盖codec/评估/VC/数据等) | | |

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Tokenizer/Codec** | 监督多任务语义tokenizer (CosyVoice系列); VARSTok变长token (AAAI 2026 Oral) |
| **生成模型** | LLM (AR语义token) + Flow Matching (声学rendering); CosyVoice 3: 1.5B参数 |
| **Post-training** | DiffRO (可微分reward优化, CosyVoice 3核心创新); SFT多阶段 |
| **流式/效率** | Chunk-aware Flow Matching (CosyVoice 2); 原生因果流式 |
| **评估** | 自建多维度评测 |
| **其他** | 全双工(Fun-Audio-Chat, 5Hz语义+25Hz声学双分辨率); 音乐(FunMusic); 配音; 唱歌 |

### 架构演进

```
CosyVoice (2024.07) — 监督语义token + Flow Matching
    ↓ +流式
CosyVoice 2 (2024.12) — chunk-aware FM
    ↓ +scaling +post-training
CosyVoice 3 (2025.05) — 1.5B/1M小时/DiffRO
    ↓ +全双工
Fun-Audio-Chat (2025.12) — 双分辨率全双工
    ↓ +统一
UniVocal (2026.06) — 统一理解+生成

并行分支:
VARSTok (2026) — 变长token (AAAI 2026 Oral, 独立tokenizer创新)
MELA-TTS (2026) — 多情感长音频
```

### 独特技术赌注

1. **监督语义token**: 用ASR/TTS多任务监督训练tokenizer, 而非自监督
2. **DiffRO (可微分Reward优化)**: TTS post-training的新范式, 直接优化reward而非用RL近似
3. **VARSTok 变长token**: 自适应压缩率, AAAI 2026 Oral, 学术新意极高
4. **双分辨率全双工**: 5Hz语义+25Hz声学, 解耦理解和生成的时间分辨率

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| CosyVoice | ~9k+ | 全栈开源 (tokenizer+LLM+FM+vocoder) |
| FunASR | ~8k+ | 开源ASR工具箱 |
| SenseVoice | ~4k+ | 开源语音理解 |

### 判断

- **优势**: 开源生态最完善; 监督token验证最充分; DiffRO在post-training方向领先; 产出密度极高(CosyVoice 3后仍密集发新工作)
- **短板**: instruct可控能力弱(CosyVoice偏zero-shot, 不是instruction-driven); 鄢志杰离职可能影响团队凝聚力
- **下一步推测**: UniVocal方向(统一模型); 更大规模scaling; DiffRO推广到更多任务
- **与Qwen语音差异**: 通义偏语音专用工具链(tokenizer+codec+TTS+ASR), Qwen偏LLM多模态语音能力

---

## 团队1b: Qwen语音 (阿里巴巴)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 阿里巴巴 - 通义实验室 - 千问团队 |
| **团队名** | Qwen Team (语音子方向) |
| **GitHub org** | [QwenLM](https://github.com/QwenLM) |
| **HuggingFace org** | [Qwen](https://huggingface.co/Qwen) |
| **核心人物** | **周靖人 (Jingren Zhou)** — CTO; **Hangrui Hu** — Qwen3-TTS一作 (原FunAudioLLM, 参与CosyVoice 1, 后转入Qwen团队) |
| **人员变动** | 前Tech Lead **林俊旸 (Junyang Lin)** 2026.03 离职 |
| **产品线** | 通义千问 (Qwen系列大模型的语音能力) |

### 论文时间线 (2024-2026)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2026.05 | **Qwen3-TTS** — 双tokenizer/5M+小时/97ms TTFB | tech report | TTS |
| 2026.04 | **Qwen3-Omni** — MoE多模态 | tech report | 多模态 |
| 2025.11 | **Qwen2.5-Omni** — Thinker-Talker架构 | tech report | 多模态 |
| 2024.07 | **Qwen2-Audio** — 多任务音频理解 | preprint | 音频理解 |
| 2024.01 | **Qwen-Audio** — 统一音频理解 | preprint | 音频理解 |

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Tokenizer** | Qwen3-TTS: 双tokenizer (语义+声学), 与CosyVoice独立自研 |
| **生成模型** | Qwen3-TTS: LLM-native TTS, ~2B级别; Qwen3-Omni: MoE统一模型 |
| **Post-training** | SFT; RL细节未公开 |
| **流式/效率** | Qwen3-TTS: 97ms TTFB, 原生流式 |
| **数据规模** | 5M+小时训练数据 (行业最大之一) |

### 架构演进

```
Qwen-Audio (2024.01) — 统一音频理解 (纯理解)
    ↓
Qwen2-Audio (2024.07) — 多任务音频理解 (仍纯理解)
    ↓ +生成能力
Qwen2.5-Omni (2025.11) — Thinker-Talker架构 (首次加入语音生成)
    ↓ +MoE
Qwen3-Omni (2026.04) — MoE多模态统一
    ↓ +独立TTS
Qwen3-TTS (2026.05) — 独立高质量TTS系统
```

### 独特技术赌注

1. **LLM-native TTS**: 不是独立TTS系统, 而是LLM的语音输出能力
2. **Thinker-Talker**: 分离思考(文本)和表达(语音), 语音不拖累推理
3. **超大数据规模**: 5M+小时, 可能是已公开的最大TTS训练集之一
4. **97ms TTFB**: 极低首包延迟

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| Qwen2.5-Omni | ~5k+ | 模型权重+代码 |
| Qwen3-TTS | 新发布 | HuggingFace开源 |

### 判断

- **优势**: 数据规模巨大(5M+小时); LLM backbone强(Qwen系列); 97ms TTFB极低; 背靠Qwen生态用户量大
- **短板**: 语音方向起步晚(2025底才加生成); 团队核心人较少; 林俊旸离职影响
- **下一步推测**: Qwen4系列进一步整合语音; TTS能力持续提升

### 通义 vs Qwen 对比

| 维度 | 通义语音 (FunAudioLLM) | Qwen语音 |
|------|----------------------|----------|
| **定位** | 语音专用工具链 | LLM多模态语音能力 |
| **核心产品** | CosyVoice/FunASR/SenseVoice | Qwen-Audio/Qwen3-TTS |
| **Tokenizer** | 监督语义token (自研) | 双tokenizer (独立自研) |
| **数据规模** | 1M小时 | 5M+小时 |
| **论文数** | 35+ | 5 |
| **开源影响** | CosyVoice ~9k stars | Qwen2.5-Omni ~5k stars |
| **技术路线** | 模块化(tokenizer+LM+FM) | LLM-native(内嵌到Qwen) |
| **人员重叠** | Hangrui Hu 从通义转Qwen (唯一已知) | 零 |
| **Post-training** | DiffRO (领先) | 未公开 |
| **流式延迟** | ~200ms (CosyVoice 2) | 97ms (Qwen3-TTS) |

**关键差异**: 通义做"语音领域的基础设施", Qwen做"大模型的语音能力"。两者技术栈完全独立, 但Hangrui Hu的转移说明存在人才流动。

---

## 团队2: 字节跳动 / Seed-TTS (ByteDance)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | 字节跳动 ByteDance |
| **团队名** | ByteDance Speech / Seed Team |
| **GitHub** | [BytedanceSpeech](https://github.com/BytedanceSpeech) (seed-tts-eval等); [bytedance](https://github.com/bytedance) (SALMONN, MegaTTS3等) |
| **HuggingFace** | 无独立org; 模型散布或不公开 |
| **核心人物** | **Yuxuan Wang (王宇轩)** — 总负责人; **Yuping Wang (王雨平)** — 高级管理; **Zhuo Chen** — TTS研究lead (前Microsoft Research, WavLM共同作者); **Dongya Jia** — DiTAR/DiSTAR一作; **Jian Wu** — 核心研究员 (前Microsoft, WavLM共同作者) |
| **产品线** | 豆包 (Doubao) 语音能力, 剪映 (CapCut) 配音, Seedance 音视频生成 |
| **开源态度** | 核心模型不开源("AI safety"); 评估工具/SALMONN/MegaTTS3开源 |

### 论文时间线 (2024-2026, 40+篇)

#### 2024 (16篇)

| 时间 | 论文 | 方向 | 说明 |
|------|------|------|------|
| 2024.06 | **Seed-TTS** | TTS | 旗舰, 定义eval标准 |
| 2024.08 | **LSLM** | 全双工 | 首个listening-while-speaking LLM |
| 2024.09 | **Seed-ASR** | ASR | LLM-based ASR旗舰 |
| 2024.09 | **Seed-Music** | 音乐 | 统一音乐生成框架 |
| 2024.11 | **Seed-VC** | VC | DiT零样本语音转换 |
| 2024.11 | **SALMONN-omni v1** | 全双工 | codec-free全双工LLM |
| ... | (另有10篇: VoiceShop/T-CLAP/SD-Eval/video-SALMONN等) | | |

#### 2025 (22篇)

| 时间 | 论文 | 方向 | 说明 |
|------|------|------|------|
| 2025.02 | **DiTAR** | TTS | AR+连续扩散, **ICML 2025** |
| 2025.02 | **MegaTTS 3** | TTS | 稀疏对齐Latent DiT (与浙大合作) |
| 2025.06 | **MagiCodec** | Codec | 高斯注入单层codec, Zipf分布, LM友好 |
| 2025.07 | **SplitMeanFlow** | 推理加速 | **20x加速, 已部署豆包** |
| 2025.07 | **Seed LiveInterpret 2.0** | 翻译 | 端到端同传+声音克隆 |
| 2025.08 | **Vevo2** | 唱歌+TTS | 统一语音+唱歌 (与港中深) |
| 2025.09 | **ARDM-DPO** | Post-training | DiTAR的DPO后训练 |
| 2025.10 | **DiSTAR** | TTS | 离散RVQ+AR+掩码扩散 |
| 2025.10 | **ELLSA** | 多模态全双工 | 视觉+语音+动作, **ICLR 2026** |
| 2025.11 | **SpeechJudge** | 评估/Reward | 语音自然度reward model |
| 2025.11 | **ParaS2S** | Post-training | 副语言感知S2S+RL, **ICLR 2026** |
| 2025.12 | **JoyVoice** | 对话TTS | 长上下文多说话人 |
| ... | (另有10篇: Audio-CoT/QualiSpeech/MMAR等) | | |

#### 2026 (4+篇)

| 时间 | 论文 | 方向 | 说明 |
|------|------|------|------|
| 2026.06 | **WavTTS** | TTS | **最新!** 直接波形DiT, 跳过codec |
| 2026.04 | **Speech VAE Distillation** | 表征 | 统一重建/理解/生成 |
| 2026.04 | **Seedance 2.0** | 视频+音频 | 170+作者, 已商业化 |
| 2026.02 | **ALIVE** | 视频+音频 | 音视频联合生成 |

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Tokenizer/Codec** | MagiCodec (单层, 高斯注入, Zipf分布, 开源); Seed-TTS原始tokenizer (类Betker) |
| **生成模型** | 演进线: AR tokens(Seed-TTS) → AR+DiT(DiTAR, ICML) → AR+掩码扩散(DiSTAR) → 直接波形DiT(WavTTS) |
| **Post-training** | RL(Seed-TTS, TTS领域首个) → DPO(ARDM-DPO) → Reward Model(SpeechJudge, 基于Qwen2.5-Omni-7B) |
| **推理加速** | SplitMeanFlow: 20x加速, **已部署豆包生产** |
| **流式/全双工** | LSLM → SALMONN-omni → ELLSA(视觉+语音+动作, ICLR 2026) |
| **评估** | SEED-TTS-Eval (已成行业事实标准); SpeechJudge; SD-Eval; MMAR; QualiSpeech |
| **其他** | ASR(Seed-ASR); 音乐(Seed-Music); VC(Seed-VC); 翻译(LiveInterpret); 编辑(VoiceShop/AudioMorphix); 视频(Seedance) |

### 架构演进

```
Seed-TTS (2024.06): AR离散tokens + token diffusion
    ↓
DiTAR (2025.02, ICML): AR连续tokens + next-token diffusion
    ↓                               ↓
SplitMeanFlow (2025.07)      DiSTAR (2025.10): 离散RVQ + AR + masked diffusion
(20x加速, 部署豆包)                  ↓
    ↓                               ↓
ARDM-DPO (2025.09)          WavTTS (2026.06): 直接波形空间DiT
(DiTAR DPO)                  ← 最新前沿, 跳过所有codec

并行: MagiCodec (2025.06) — 新一代codec基建
并行: SpeechJudge (2025.11) — reward model基建
```

### 独特技术赌注

1. **DiT全面押注**: 几乎所有后续工作都基于Diffusion Transformer
2. **Post-training先行者**: TTS领域首个系统应用RL, 持续迭代到DPO+reward model
3. **全双工全栈**: LSLM→SALMONN-omni→ELLSA, 最完整的全双工研究路线
4. **评估基建**: SEED-TTS-Eval已成事实标准 (被数十篇论文采用)
5. **"Seed"品牌矩阵**: Seed-TTS/ASR/Music/VC/LiveInterpret/Seedance

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| SEED-TTS-Eval | 1.6k | 评估工具+测试集 |
| SALMONN系列 | 1.4k | 代码+权重 |
| MegaTTS 3 | 6.1k | 代码+部分权重 (WaveVAE encoder不公开) |
| MagiCodec | 开源 | 代码+预训练模型 |
| Seed-TTS本体 | 不开源 | "AI safety" |

### 重要发现

- **Seed-TTS 2.0已存在**: Step-Audio-EditX论文提及"Doubao-Seed-TTS-2.0"作为baseline, 已部署豆包
- **核心团队来自Microsoft Research**: Zhuo Chen和Jian Wu (WavLM共同作者) 约2023-2024加入字节
- **WavTTS (2026.06)**: 最新方向, 直接波形DiT, 可能是下一代架构

### 判断

- **优势**: 论文产出量最大(40+篇); 技术深度(DiTAR ICML/ELLSA ICLR); 产品闭环(SplitMeanFlow部署豆包); 评估主导权(Seed-TTS-Eval)
- **短板**: 核心模型不开源(社区影响力不如CosyVoice); 产品信息不透明; 唱歌/音乐偏弱
- **下一步推测**: WavTTS可能升级为Seed-TTS 3.0; MagiCodec+DiSTAR组合产品化; SpeechJudge驱动自动RLHF闭环

---

## 团队3: 阶跃星辰 / StepAudio (StepFun)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | 阶跃星辰 (StepFun) |
| **GitHub org** | [stepfun-ai](https://github.com/stepfun-ai) (31 repos) |
| **HuggingFace org** | stepfun-ai |
| **核心人物** | **张祥雨 (Xiangyu Zhang)** — StepAudio 2.5通讯作者, 旷视联合创始人; **蒋大信 (Daxin Jiang)** — 几乎所有论文senior; **余刚 (Gang Yu)** — R1系列senior; **田飞 (Fei Tian)** — R1技术负责人 |
| **团队规模** | Step-Audio v1作者145人, StepAudio 2.5作者101人; 核心音频团队30-50人 |
| **产品线** | StepFun开放平台, Claw AI助手, Step-Realtime API (WebSocket实时语音对话) |

### 论文时间线 (2025-2026, 8篇)

| 时间 | 论文 | arXiv | 方向 |
|------|------|-------|------|
| 2025.02 | **Step-Audio** | 2502.11946 | 130B语音-文本多模态, dual-codebook |
| 2025.06 | **Step-Audio-AQAA** | 2506.08967 | 端到端Audio-Query-Audio-Answer |
| 2025.07 | **Step-Audio 2** | 2507.16632 | 轻量化(7B级), Qwen2初始化 |
| 2025.10 | **Mind-Paced Speaking** | 2510.09592 | Dual-Brain实时CoT推理 |
| 2025.11 | **Step-Audio-EditX** | 2511.03601 | 首个开源LLM-based音频编辑(3B) |
| 2025.11 | **Step-Audio-R1** | 2511.15848 | 首个音频推理模型, MGRD |
| 2026.04 | **Step-Audio-R1.5** | 2604.25719 | RLHF替代RLVR |
| 2026.05 | **StepAudio 2.5** | 2605.23463 | 三模式统一RLHF (ASR/TTS/Realtime) |

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Tokenizer** | Dual-codebook: Linguistic(Paraformer, 16.7Hz) + Semantic(CosyVoice tokenizer, 25Hz), 2:3交错 |
| **生成模型** | v1: 130B LLM→文本→3B Speech Decoder(LM+FM+Vocoder); v2: 7B级(Qwen2初始化) |
| **Post-training** | SFT→DPO→PPO/GRPO→RLHF; 特色: anti-deaf-hacking, generative reward modeling, MGRD |
| **推理** | R1: MGRD (Modality-Grounded Reasoning Distillation); R1.1: Dual-Brain实时推理 |
| **编辑** | EditX: 3B, large-margin synthetic data, PPO, 30+种说话风格 |
| **流式** | Speculative response generation; WebSocket实时API |

### 架构演进

```
Step-Audio v1 (2025.02): 130B LLM + 3B TTS
    ↓
Step-Audio-AQAA (2025.06): 端到端Audio→Audio
    ↓
Step-Audio 2 (2025.07): 轻量化7B, Qwen2初始化
    ↓
Step-Audio-R1 (2025.11): 32B音频推理
    ↓
Step-Audio-R1.1 (2026.01): Dual-Brain实时推理
    ↓
Step-Audio-R1.5 (2026.04): RLHF优化
    ↓
StepAudio 2.5 (2026.05): 三模式统一RLHF
```

### 独特技术赌注

1. **Dual-codebook tokenizer**: linguistic+semantic物理并行, 2:3交错
2. **MGRD**: 解决audio LM的"文本代理推理"问题, 让推理链锚定声学特征
3. **Dual-Brain实时推理**: Formulation Brain(推理)+Articulation Brain(生成)分离
4. **三模式RLHF**: ASR用verifiable decoding, TTS用preference RLHF, Realtime用generative reward modeling
5. **Large-margin synthetic data**: EditX纯靠合成数据做情感/风格编辑

### 开源情况

| 项目 | Stars | 许可 |
|------|-------|------|
| Step-Audio 2 mini | 1.5k | Apache-2.0 |
| Step-Audio-EditX | 925 | Apache-2.0 |
| Step-Audio-R1 | 671 | Apache-2.0 |

**全部Apache-2.0**, 开源策略激进。

### 判断

- **优势**: 投入最重(百人团队); 覆盖面最广(理解+生成+编辑+推理+实时); R1系列音频推理方向开创性; 开源策略激进
- **短板**: 130B推理成本高; 轻量模型独创性有限(Qwen2初始化); 自建benchmark公正性待验证
- **下一步推测**: StepAudio 3.0整合R1.5+2.5; 可能推出端侧小模型(1B级)

---

## 团队4: 智谱 / GLM-4-Voice (Zhipu AI)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司** | 智谱AI (Zhipu AI) |
| **GitHub** | [THUDM](https://github.com/THUDM) (GLM-4-Voice); [zai-org](https://github.com/zai-org) (GLM-TTS) |
| **HuggingFace** | THUDM; zai-org/ZhipuAI |
| **核心人物** | **唐杰 (Jie Tang)** — 清华教授, 智谱联合创始人; **曾奥涵 (Aohan Zeng)** — GLM-4-Voice/GLM-5一作; **李乃含 (Naihan Li)** — GLM-TTS核心, 前Microsoft TTS |
| **团队规模** | GLM-4-Voice: 8人; GLM-TTS: 13人; 语音方向总计~15-20人 |
| **产品线** | 智谱清言, Z.ai开放平台 (CogTTS品牌) |

### 论文时间线 (2024-2026, 3篇)

| 时间 | 论文 | arXiv | 方向 |
|------|------|-------|------|
| 2024.11 | **Scaling Speech-Text Pre-training** | 2411.17607 | 175bps tokenizer + 合成交错数据 |
| 2024.12 | **GLM-4-Voice** | 2412.02612 | 端到端语音聊天, 9B, 引用243 |
| 2025.12 | **GLM-TTS** | 2512.14291 | 生产级TTS, 1.5B, GRPO四维reward |

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Tokenizer** | Whisper encoder + VQ, 175bps, 12.5Hz, 单codebook (业界最低码率之一) |
| **生成模型** | GLM-4-Voice: GLM-4-9B + CosyVoice FM Decoder; GLM-TTS: Llama架构AR(1.5B) + FM DiT + Vocos2D |
| **Post-training** | GLM-TTS: GRPO (CER+SIM+Emotion+Laughter四维reward) |
| **流式** | GLM-4-Voice: 10个token即可开始生成 |
| **Voice定制** | GLM-TTS: LoRA-based (15%参数, 1小时数据) |

### 架构演进

```
175bps Tokenizer (2024.11) — 超低码率+合成交错数据方法论
    ↓
GLM-4-Voice (2024.12) — 端到端spoken chatbot, 9B
    ↓
GLM-TTS (2025.12) — 独立生产级TTS, 1.5B, GRPO RL
    ↓
[2026年无新论文]
```

### 独特技术赌注

1. **175bps超低码率**: 12.5Hz单codebook, 远低于主流(EnCodec ~6000bps)
2. **合成交错数据**: 文本→TTS合成语音→交错排列, scale to 1T tokens
3. **GRPO四维reward**: 同时优化CER/SIM/Emotion/Laughter
4. **Vocos2D**: 1D→2D卷积改善频率子带建模

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| GLM-4-Voice | 3.2k | 全套开源 (tokenizer+9B+decoder) |
| GLM-TTS | 1k | 代码+权重; RL版权重"coming soon" |

### 判断

- **优势**: 175bps tokenizer独特性极强; GLM-4-Voice引用243(最高); 精简团队效率高; 学术底蕴深(清华THUDM)
- **短板**: **2026年无新论文, 语音方向明显减速**; GLM-5/4.5/4.5V均无语音能力; Bug较多(66 issues); 两个GitHub org(THUDM vs zai-org)看似半独立
- **下一步推测**: 语音在智谱战略中可能非核心; 若继续可能做更低码率tokenizer或voice agent
- **风险**: 如果2026下半年仍无新工作, 将被竞争对手拉开差距

---

## 横向对比矩阵

| 维度 | 通义FunAudioLLM | Qwen语音 | 字节Seed-TTS | 阶跃StepAudio | 智谱GLM |
|------|----------------|----------|-------------|--------------|---------|
| **论文数(24-26)** | 35+ | 5 | 40+ | 8 | 3 |
| **最新论文** | 2026.06 | 2026.05 | 2026.06 | 2026.05 | 2025.12 |
| **团队规模** | 中 | 小 | 大 | 极大(100+) | 极小(15-20) |
| **投入力度** | ★★★★★ | ★★★★ | ★★★★★ | ★★★★★ | ★★ |
| **模型规模** | 1.5B | ~2B | 未公开(极大) | 130B→3B | 9B→1.5B |
| **数据规模** | 1M小时 | 5M+小时 | 未公开 | 未公开 | 100K小时 |
| **Tokenizer路线** | 监督语义token | 双tokenizer | MagiCodec(Zipf) | Dual-codebook | 175bps ASR-VQ |
| **生成架构** | LLM+FM | LLM-native | DiT系列 | 130B LLM+FM | LLM+FM |
| **Post-training** | DiffRO | 未公开 | RL→DPO→Reward | PPO/DPO/GRPO/RLHF | GRPO四维 |
| **流式TTFB** | ~200ms | 97ms | 未公开 | ~500ms | 支持 |
| **全双工** | Fun-Audio-Chat | Qwen2.5-Omni | LSLM→ELLSA(ICLR) | Dual-Brain | GLM-4-Voice |
| **音频推理** | 无 | 无 | Audio-CoT | R1/R1.5(领先) | 无 |
| **音频编辑** | 无 | 无 | VoiceShop | EditX(开源) | 无 |
| **开源Stars** | CosyVoice ~9k | Qwen2.5-Omni ~5k | MegaTTS3 6.1k | Step-Audio2 1.5k | GLM-4-Voice 3.2k |
| **引用数** | CosyVoice 高 | Qwen-Audio 高 | Seed-TTS 高 | Step-Audio 107 | GLM-4-Voice 243 |
| **产品化** | CosyVoice API | 千问语音 | 豆包/剪映 | Realtime API | 智谱清言 |
| **独特赌注** | 监督token+DiffRO | LLM-native+大数据 | DiT+评估标准 | RLHF全覆盖+推理 | 超低码率 |

### 关键发现

1. **字节和通义是产出最密集的两支团队**, 2026年仍在高频发论文
2. **阶跃在RLHF和音频推理方向独树一帜**, R1系列是独有赛道
3. **智谱语音方向明显减速**, 2026年无新论文, GLM-5/4.5均无语音
4. **Qwen语音虽起步晚但数据规模最大(5M+小时)**, 流式延迟最低(97ms)
5. **通义和Qwen技术栈完全独立**, 仅Hangrui Hu一人有跨团队经历
6. **Post-training是最激烈的竞争轴**: DiffRO(通义) vs RL+DPO+SpeechJudge(字节) vs 三模式RLHF(阶跃) vs GRPO(智谱)
7. **Seed-TTS 2.0已部署但未发论文**, WavTTS(直接波形DiT)可能是下一代

---

## 调研质量自检

- [x] GitHub/HuggingFace org 都实际搜索了
- [x] 核心人的名字都在 arXiv 上搜过了
- [x] 搜索关键词覆盖了 speech/audio/voice/TTS/codec/tokenizer/vocoder/ASR
- [x] 找到了"非TTS标题"的相关工作 (如LSLM/ELLSA/SpeechJudge/Audio-CoT)
- [x] 论文时间线尽可能完整
- [x] 确认了开源情况和影响力指标
- [ ] 竞赛参与情况: 未在本session深入 (待Session 8补充)
- [ ] 招聘JD反推: 未在本session深入
