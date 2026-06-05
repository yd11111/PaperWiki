# Session 6: 学术界 A — 李宏毅 + 中科大 + 西工大 + 清华

> 调研日期: 2026-06-04

---

## 实验室1: 李宏毅组 (NTU, 台湾)

### 基本信息

| 项目 | 详情 |
|------|------|
| **大学** | National Taiwan University (NTU), EE & CS&IE |
| **实验室** | Speech Processing and Machine Learning Lab |
| **核心人物** | **Hung-yi Lee (李宏毅)** — 教授, h-index 52, ~295 papers |
| **关键学生** | Cheng-Han Chiang (STITCH/SHANKS), Kai-Wei Chang (TiCo, 与MIT), Guan-Ting Lin (Full-Duplex-Bench系列) |
| **GitHub** | dynamic-superb org (200 stars), speech-trident (1.2K stars) |

### 论文时间线 (2024-2026, speech/audio方向, 60+篇)

**2026 (30+篇, 精选)**

| 时间 | 论文 | 方向 |
|------|------|------|
| 2026.04 | LLM-Codec (ACL 2026 Findings) | 音频编码 |
| 2026.04 | Full-Duplex-Bench-v3 | Full-Duplex benchmark |
| 2026.04 | ASPIRin: RL for Full-Duplex SLMs | Full-Duplex SLM |
| 2026.04 | VIBE: Voice-Induced Bias Evaluation | LALM公平性 |
| 2026.04 | NVBench: Non-Verbal Vocalizations | TTS benchmark |
| 2026.03 | TiCo: Time-Controllable SLM (与MIT) | SLM |
| 2026.03 | TASTE-Streaming | 语音token化 |
| 2026.03 | MOS-Bias: Gender Bias in MOS | 评测公平性 |
| 2026.03 | Latent-Mark: Audio Watermark | 音频水印 |
| 2026.01 | ICASSP 2026 HumDial Challenge (与NPU) | Full-Duplex竞赛 |

**2025 (30+篇, 精选)**

| 时间 | 论文 | 方向 | 会议 |
|------|------|------|------|
| 2025.10 | SHANKS: Simultaneous Hearing+Thinking | SLM推理 | **ICLR 2026** |
| 2025.07 | STITCH: Chunked Reasoning (与Microsoft) | SLM推理 | **ICLR 2026** |
| 2025.07 | Full-Duplex-Bench v1.5 | benchmark | |
| 2025.06 | Discrete Audio Tokens Survey | 综述 | |
| 2025.04 | **SLM Survey** (127 citations) | 综述 | |
| 2025.04 | TASTE: Text-Aligned Speech Tokenization | token化 | |
| 2025.03 | Full-Duplex-Bench v1 | benchmark | |
| 2025.02 | Gender Bias in Instruction TTS | 公平性 | |

### 研究方向全景

**核心方向 (做什么)**:
1. **SLM/LALM评测与benchmark** — SUPERB→Dynamic-SUPERB→Full-Duplex-Bench系列, 定义行业标准
2. **Full-Duplex对话系统** — FDB v1-v3 + STITCH + SHANKS + ASPIRin + TiCo, benchmark+方法完整布局
3. **LALM分析** — 大量工作分析内部机制、幻觉、偏见、校准
4. **语音token化** — TASTE→TASTE-Streaming, LLM-Codec
5. **语音AI公平性** — 2025-2026新方向, MOS-Bias/VIBE/Gender Bias, 形成体系
6. **Deepfake检测** — CodecFake/CodecFake+, SingFake, 音频水印

**不做什么**: 不做端到端TTS系统 (不与工业界竞争模型规模); 不做大规模数据集; 不做唱歌合成

### 与工业界关系

- **Microsoft**: STITCH + SHANKS (ICLR 2026), Cheng-Han Chiang联合发表
- **MIT CSAIL**: TiCo, Kai-Wei Chang与James Glass合作
- **NPU**: HumDial Challenge联合组织 (ICASSP 2026)
- 保持学术独立性, 不深度绑定企业

### 判断

**亚洲最重要的语音AI评测与分析实验室。** 在SLM/LALM评测和Full-Duplex方向具有全球领导地位。不与工业界竞争模型规模, 而是定义评测标准和分析框架, 成为所有SLM/LALM开发者的必引参考。**SLM Survey 127 citations, STITCH/SHANKS双入ICLR 2026。** 2025-2026年新开辟语音AI公平性方向。

---

## 实验室2: 中科大语音实验室 (USTC)

### 基本信息

| 项目 | 详情 |
|------|------|
| **大学** | 中国科学技术大学 (USTC), 电子工程与信息科学系 |
| **实验室** | 语音及语言信息处理实验室 |
| **核心教授** | **凌震华 (Zhen-Hua Ling)** — TTS韵律/声码器/增强; **李日荣 (Li-Rong Dai)** — 高级, 近年发文少 |
| **关键学生** | **Yang Ai (艾洋)** — 声码器/codec/相位预测; Rui-Chen Zheng — 语音重建; Nai-Qian Wu — 有声书TTS |
| **与讯飞关系** | USTC是讯飞学术摇篮, 但当前凌震华组论文中直接与讯飞联合署名不多, 已形成一定独立性 |

### 论文时间线 (2024-2026, 精选)

**2026**

| 时间 | 论文 | 方向 |
|------|------|------|
| 2026.06 | UniVocal: Speech-Singing Code-Switching | 歌唱合成 |
| 2026.05 | CFMDCTCodec: MDCT域低比特率Codec | 语音编码 |
| 2026.05 | Ultra-Low-Bitrate Mel-Spectrogram Coding (250bps) | 语音编码 |
| 2026.04 | LatentFlowSR: Audio Super-Resolution | 音频超分 |
| 2026.02 | ParaGSE: Parallel Generative Speech Enhancement | 语音增强 |
| 2026.01 | FunCineForge: Zero-Shot Movie Dubbing | 电影配音 |

**2025**

| 时间 | 论文 | 方向 |
|------|------|------|
| 2025.11 | IDMap: Voice Anonymization | 声纹匿名 |
| 2025.10 | Perception Inconsistency in Speaker Anonymization | 声纹匿名 |
| 2025.09 | TTS Stability (与阿里CosyVoice2合作, 后撤回) | TTS稳定性 |
| 2025.09 | Noise Robustness for Neural Speech Codecs | 语音编码 |
| 2025.05 | UDDETTS: Unified Emotions for TTS | 情感TTS |
| 2025 | PhonemeVec: Phoneme-Level Prosody (ACM TALIP) | TTS韵律 |
| 2025 | LIST: Language-Independent Speech Token (Interspeech) | 多语言TTS |

**2024**

| 时间 | 论文 | 方向 |
|------|------|------|
| 2024.12 | DiffStyleTTS (COLING 2025) | TTS韵律 |
| 2024 | High-quality Speech Bandwidth Extension (TASLP, 38 cit.) | 带宽扩展 |
| 2024 | MDCTCodec (SLT, 18 cit.) | 语音编码 |

### 研究方向

1. **声码器/相位预测** — Yang Ai主导, MDCT域codec系列
2. **极低比特率语音编码** — MDCTCodec→CFMDCTCodec, 250bps-0.65kbps
3. **TTS韵律建模** — DiffStyleTTS, PhonemeVec, 有声书韵律 (传统强项)
4. **声纹匿名化/隐私** — 2025年显著扩展
5. **语音增强/带宽扩展** — 传统强项

### 与工业界关系

- **讯飞**: 历史深厚但当前论文联合署名少
- **阿里**: 2025年与CosyVoice2团队合作TTS稳定性 (后撤回)
- **快手**: 凌震华参与Kwai Keye-VL技术报告

### 判断

**传统语音信号处理强校, LLM-era转型保守。** 在声码器/相位/韵律/带宽扩展方面有深厚积累, 但在LLM-based TTS/SLM/Full-Duplex等热门方向参与度不高。**声纹匿名化和极低比特率编码是差异化方向。** 影响力趋势: 稳中有降。

---

## 实验室3: 西工大 ASLP (NPU)

### 基本信息

| 项目 | 详情 |
|------|------|
| **大学** | 西北工业大学 (NPU), 计算机学院 |
| **实验室** | Audio, Speech and Language Processing (ASLP) Lab |
| **核心人物** | **谢磊 (Lei Xie)** — 教授, h-index 63, 18000+ citations |
| **关键学生** | **Xinsheng Wang** (Spark-TTS/SoulX核心), Xinfa Zhu (KALL-E/Llasa), Jixun Yao (DiffRhythm), Wenjie Tian (dLLM-ASR), Xuelong Geng (OSUM), Hongfei Xue (WenetSpeech) |
| **GitHub** | SparkAudio org (Spark-TTS **11K stars**) |
| **学术服务** | Senior Area Editor for IEEE/ACM TASLP and IEEE SPL |

### 论文时间线 (2024-2026, 40+篇, 精选)

**2026**

| 时间 | 论文 | 方向 | 会议 |
|------|------|------|------|
| 2026.06 | SoulX-Transcriber | 多人转录 | |
| 2026.03 | SoulX-Duplug | Full-Duplex | Interspeech 2026 |
| 2026.03 | OmniCodec | 通用音频Codec | |
| 2026.02 | SoulX-Singer (42K hrs) | 歌声合成 | |
| 2026.02 | EmoOmni | 情感多模态 | |
| 2026.01 | dLLM-ASR | Diffusion LLM ASR | |
| 2026.01 | WenetSpeech-Wu | 吴语数据集 | |
| 2026 | KALL-E | next-distribution TTS | **AAAI 2026** |
| 2026 | WenetSpeech-Yue | 粤语数据集 | **AAAI 2026** |
| 2026.01 | HumDial Challenge (与NTU) | Full-Duplex竞赛 | ICASSP 2026 |

**2025**

| 时间 | 论文 | 方向 | 会议 |
|------|------|------|------|
| 2025.10 | SoulX-Podcast | 多方言播客 | |
| 2025.10 | SAC (语义-声学双流Codec) | 语音编码 | **ACL 2026 Main** |
| 2025.10 | DialoSpeech | 双人对话TTS | |
| 2025.10 | DiffRhythm 2 | 歌曲生成 | |
| 2025.07 | DiffRhythm+ | 可控歌曲 | |
| 2025.03 | **Spark-TTS** (144 cit., 11K stars) | TTS系统 | **ACL 2025** |
| 2025.03 | DiffRhythm | 端到端歌曲生成 | |
| 2025.02 | **Llasa** (1B/3B/8B) | TTS Scaling | |
| 2025.02 | Audio-FLAN | 大规模数据集 | |
| 2025.01 | OSUM (30 cit.) | 开源语音理解 | |
| 2025.01 | FleSpeech | 多模态prompt TTS | |
| 2025 | StableVC (39 cit.) | 语音转换 | **AAAI 2025** |

### 研究方向全景

**核心方向**:
1. **LLM-based TTS系统** — Spark-TTS (11K stars, 144 cit.), Llasa, KALL-E, FleSpeech
2. **歌曲/歌唱生成** — DiffRhythm→DiffRhythm+→DiffRhythm 2, SoulX-Singer
3. **方言数据集** — WenetSpeech-Yue/Chuan/Wu系列
4. **对话式TTS** — SoulX-Podcast, DialoSpeech
5. **Full-Duplex** — SoulX-Duplug, HumDial Challenge
6. **语音理解** — OSUM→OSUM-Pangu
7. **语音编码** — SAC (ACL 2026), BiCodec

### 与工业界关系

- **SoulX系列**: 与Soul App深度合作 (SoulX-Singer/Podcast/Duplug等)
- **阿里/Qwen**: LLM-ForcedAligner在QwenLM org下
- **WeNet生态**: 工业级ASR工具包, 被广泛采用
- **学生去向**: 阿里、腾讯、京东等大厂

### 开源贡献

| 项目 | Stars | 说明 |
|------|-------|------|
| **Spark-TTS** | **11K** | 最受欢迎开源LLM-TTS |
| VoxBox | — | 10万小时标注数据 |
| WeNet | — | 工业级ASR |
| DiffRhythm | — | 歌曲生成 |
| OSUM | — | 开源语音理解 |
| WenetSpeech系列 | — | 方言数据集 |

### 判断

**2024-2026年亚洲学术界影响力上升最快的语音实验室。** Spark-TTS (144 citations, 11K stars) 是2025年最成功的开源TTS。DiffRhythm开创歌曲生成方向。WenetSpeech方言系列填补数据空白。**系统工程能力极强, 能快速构建完整系统并开源。** 与Soul App深度合作带来从codec到数字人的全栈成果。

---

## 实验室4: 清华语音组 (多个独立组)

清华至少有**三个独立语音/音频研究组**:

### 4A. THUHCSI — 吴志勇组 (清华深圳)

| 项目 | 详情 |
|------|------|
| **实验室** | Human-Computer Speech Interaction Lab |
| **核心人物** | **吴志勇 (Zhiyong Wu)** — 实验室主任 |
| **合作** | 与CUHK Helen Meng/Xixin Wu建有"清华-中大媒体科技联合研究中心" |
| **关键学生** | Dongchao Yang (UniAudio系列), Rui Niu, Zijian Lin |

**论文 (2025-2026, 精选)**:

| 时间 | 论文 | 方向 | 会议 |
|------|------|------|------|
| 2026.05 | LoSATok: Low-dim Semantic-Acoustic Tokenizer | 音频编码 | |
| 2026.05 | UniSRM: Unified Speech Reward Model | 评测 | |
| 2026.05 | Full-Duplex Routing | Full-Duplex | |
| 2026.04 | SPG-Codec: Ultra-Low-Bitrate | 编码 | |
| 2026.04 | TTS-PRISM (与小米) | TTS评测 | |
| 2026.02 | UniAudio 2.0 (与CUHK) | 音频LM | |
| 2025.11 | E2E-VGuard: LLM-TTS防御 | TTS安全 | **NeurIPS 2025** |
| 2025.09 | VoxCPM (与THUNLP) | TTS系统 | |
| 2025.08 | **DualSpeechLM** | 统一SLM | **AAAI 2026** |
| 2025.08 | VoxInstruct: Instruction TTS | 可控TTS | |
| 2025.06 | LeVo: Song Generation | 歌曲 | |
| 2025.05 | VoiceMark: Watermarking | 水印 | |
| 2025.05 | Speech Speculative Decoding | TTS加速 | Interspeech 2025 |
| 2025.02 | DiffCSS: Conversational Speech | 对话TTS | |
| 2025.01 | DrawSpeech: Prosodic Sketches TTS | 可控TTS | |

**方向**: 可控TTS, 语音编码, 歌曲生成, 视觉配音, **语音安全** (E2E-VGuard NeurIPS), SLM (DualSpeechLM AAAI)

**与工业界**: 与小米合作紧密 (TTS-PRISM), 与CUHK联合研究中心

### 4B. CSLT — 张卫强组 (清华电子系)

| 项目 | 详情 |
|------|------|
| **实验室** | Center for Speech and Language Technologies |
| **核心人物** | **张卫强 (Wei-Qiang Zhang)** |

**论文**:

| 时间 | 论文 | 方向 |
|------|------|------|
| 2026.05 | Dolphin-CN-Dialect | 方言ASR |
| 2025.12 | YingMusic-SVC (Flow-GRPO) | 歌唱转换 |
| 2025.09 | **DiaMoE-TTS** (MoE方言TTS) | 方言TTS |
| 2025.03 | **Dolphin** (40种东方语言ASR) | 多语言ASR |
| 2024.06 | **GigaSpeech 2** | 数据集 |

**方向**: 多语言/方言ASR (Dolphin), 方言TTS (DiaMoE-TTS), 异常声音检测

### 4C. THUNLP — 刘知远组 (清华计算机系)

| 项目 | 详情 |
|------|------|
| **核心人物** | **刘知远 (Zhiyuan Liu)** — NLP方向, VoxCPM senior author |
| **与语音交叉** | VoxCPM: tokenizer-free TTS, 利用CPM系列LLM |

### 判断

**清华布局分散但总量可观。** THUHCSI (吴志勇) 覆盖面最广 — TTS/VC/编码/安全/评测, E2E-VGuard (NeurIPS) + DualSpeechLM (AAAI) 是高质量产出。CSLT (张卫强) 在方言ASR/TTS有独特价值 (Dolphin, DiaMoE-TTS)。THUNLP (刘知远) 偶尔跨界 (VoxCPM)。**整体影响力不如NPU (缺少Spark-TTS级标杆) 或NTU (缺少SUPERB级标准)。** 与CUHK联合研究和小米合作是优势。

---

## 四实验室横向对比

| 维度 | NTU 李宏毅 | USTC 凌震华 | NPU 谢磊 | 清华 (多组) |
|------|:---:|:---:|:---:|:---:|
| **论文产出 (24-26)** | **60+篇** | 25+篇 | **40+篇** | 35+篇 |
| **核心定位** | 评测+分析 | 信号处理 | **系统构建+开源** | 覆盖面广 |
| **SLM/LALM评测** | **★★★★★** | — | — | ★ |
| **Full-Duplex** | **★★★★★** | — | ★★★ | ★★ |
| **LLM-based TTS** | — | — | **★★★★★** | ★★★ |
| **传统TTS/声码器** | — | **★★★★★** | — | ★★ |
| **语音编码** | ★ | **★★★★** | ★★★ | ★★★ |
| **歌曲/歌唱** | — | — | **★★★★★** | ★★ |
| **方言数据集** | — | — | **★★★★★** | ★★ (CSLT) |
| **语音安全** | ★★★ (伪造) | ★★★ (匿名) | — | **★★★★** (防御+水印) |
| **公平性/偏见** | **★★★★★** | — | — | — |
| **顶会收录** | ICLR×2 | COLING | **ACL×2, AAAI×2** | NeurIPS, AAAI |
| **开源Stars** | ~1.4K | 低 | **11K+ (Spark-TTS)** | 中 |
| **工业合作** | Microsoft, MIT | 讯飞(弱化), 快手 | **Soul App, 阿里** | 小米, CUHK |
| **影响力趋势** | 持续上升 | 稳中有降 | **急剧上升** | 稳步上升 |

### 关键发现

1. **NPU (谢磊) 是2024-2026年上升最快的学术组**: Spark-TTS (11K stars, 144 citations) + DiffRhythm + WenetSpeech方言系列, 系统工程能力极强
2. **NTU (李宏毅) 垄断了评测方向**: SUPERB→Dynamic-SUPERB→Full-Duplex-Bench, 所有SLM/LALM开发者的必引参考
3. **USTC在LLM-era转型保守**: 传统信号处理强项仍在, 但热门方向参与度低
4. **清华分散但无标杆**: 三个组覆盖面广, 但缺少一个方向做到全球第一
5. **合作网络**: NTU↔NPU (HumDial), NPU↔Soul App (SoulX全线), 清华↔CUHK (UniAudio), 清华↔小米 (TTS-PRISM)

### 研究方向分化图

```
NTU (评测+公平性) ←→ NPU (系统+开源+歌曲)
        ↕ HumDial                    ↕ SoulX合作
        ↓                            ↓
    Full-Duplex方向              TTS系统方向

USTC (信号处理/编码)   清华 (安全/可控/多模态)
   独立, 少交叉            ↔ CUHK + 小米
```
