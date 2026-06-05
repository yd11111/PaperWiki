# Session 7: 学术界 B — 浙大 + 港中文 + CMU + 其他

> 调研日期: 2026-06-04

---

## 实验室1: 浙江大学 — Zhou Zhao 组

### 基本信息

| 项目 | 详情 |
|------|------|
| **大学/实验室** | 浙江大学计算机学院, Zhou Zhao 教授课题组 |
| **核心人物** | **Zhou Zhao** (教授/PI); **Shengpeng Ji**, **Ziyue Jiang**, **Zhenhui Ye**, **Jinglin Liu** (核心学生) |
| **关键离职** | **Yi Ren** (2022硕士毕业→TikTok→现HeyGen新加坡, 领导基础视频模型); **Rongjie Huang** (毕业→港中文Helen Meng组博后) |

### 论文时间线 (2024-2026, 18+篇)

| 时间 | 论文 | 会议 | 引用 | 方向 |
|------|------|------|------|------|
| 2026 | AST (training-free speech editing) | arXiv | - | 语音编辑 |
| 2026 | Long-form speech benchmarking | arXiv | - | 评估 |
| 2025 | **WavTokenizer** (acoustic discrete codec) | **ICLR 2025** | **189** | Codec |
| 2025 | **MegaTTS 3** (latent diffusion zero-shot TTS) | arXiv | 32 | TTS (与字节合作) |
| 2025 | TCSinger 2 (multilingual SVS) | ACL Findings | 15 | 歌声 |
| 2025 | TechSinger (technique controllable SVS) | AAAI 2025 | 22 | 歌声 |
| 2025 | Speech watermarking | AAAI 2025 | 14 | 安全 |
| 2025 | WavRAG (audio retrieval for dialogue) | ACL 2025 | - | 对话 |
| 2025 | FluentEditor2 (speech editing) | TASLP | 2 | 编辑 |
| 2025 | C2F-LM (compressed-to-fine) | ACM | 8 | TTS |
| 2024 | Mega-TTS 2 | ICLR 2024 | 77 | TTS |
| 2024 | TCSinger | EMNLP 2024 | 31 | 歌声 |
| 2024 | Frieren (video-to-audio) | NeurIPS 2024 | - | 音频生成 |
| 2024 | WavChat survey | arXiv | 106 | 综述 |
| 2024 | TextrolSpeech | ICASSP 2024 | - | 可控TTS |

### 研究方向

1. **零样本TTS**: Mega-TTS系列 (1→2→3), 从韵律prompt→稀疏对齐latent diffusion
2. **音频Codec**: **WavTokenizer** (189 citations, 最具影响力的独立贡献)
3. **歌声合成**: TCSinger系列, TechSinger
4. **语音对话/多模态**: WavChat综述, WavRAG
5. **语音编辑**: FluentEditor系列, AST

### 与工业界关系

- **与字节深度合作**: MegaTTS系列 (3是ZJU+ByteDance联合), MegaTTS 2已部署TikTok
- **人才流出**: Yi Ren→TikTok→HeyGen; Rongjie Huang→CUHK
- **WavTokenizer是独立贡献**: 由留校成员完成, 是ZJU最具独立影响力的工作

### 判断

- **优势**: 全球TTS学术界产出最大的组之一; WavTokenizer(189 citations)证明独立研究能力; 歌声合成(TCSinger)独特
- **短板**: 核心人才外流(Yi Ren, Rongjie Huang); MegaTTS依赖字节合作
- **定位**: 学术产出极高但面临代际交替挑战

---

## 实验室2: 港中文系统 (三个不同组)

### 2A. 港中文(本部) — Helen Meng / Xixin Wu 组

| 项目 | 详情 |
|------|------|
| **核心人** | **Helen Meng** (教授); **Xixin Wu** (助理教授) |
| **新成员** | Rongjie Huang (从ZJU来的博后) |

**论文**:
| 时间 | 论文 | 说明 |
|------|------|------|
| 2025 | **DualSpeechLM** (AAAI 2026) | 双token统一理解/生成 |
| 2025 | **MELLE** (ACL 2025) | 无VQ的AR语音合成 (与Microsoft合作) |
| 2025 | OmniCharacter (ACL 2025) | 角色扮演语音交互 |

**定位**: Speech LM统一理解/生成; 学术枢纽连接Microsoft+ZJU+蚂蚁

### 2B. 港中文(深圳) — 武执政 (Zhizheng Wu) / Mel Lab

| 项目 | 详情 |
|------|------|
| **核心人** | **武执政 (Zhizheng Wu)** — 助理教授, 9409 citations; **Yuancheng Wang** — 核心博士生 |
| **注意** | 非阿里员工, 是独立研究者, 曾参与NaturalSpeech系列 |

**论文**:
| 时间 | 论文 | 会议 | 引用 | 方向 |
|------|------|------|------|------|
| 2026 | **Metis** (foundation speech generation) | NeurIPS 2025 | 18 | TTS |
| 2026 | SP-MCQA (TTS evaluation) | ICASSP 2026 | 1 | 评估 |
| 2026 | AnyAccomp (singing accompaniment) | ICASSP 2026 | 2 | 歌声 |
| 2026 | NV-Bench (nonverbal vocalization) | arXiv | - | 评估 |
| 2025 | **MaskGCT** (masked generative codec transformer) | **ICLR 2025** | **210** | TTS |
| 2025 | **Vevo** (voice imitation via disentanglement) | arXiv | 63 | VC |
| 2025 | DualCodec (low-frame-rate codec) | arXiv | 21 | Codec |
| 2025 | TadiCodec (text-aware diffusion tokenizer) | NeurIPS 2025 | 7 | Codec |
| 2025 | Amphion v0.2 | arXiv | 10 | 工具 |
| 2024 | Amphion v0.1 | IEEE SLT | 73 | 工具 |
| 2024 | NaturalSpeech 3 | ICML 2024 | - | TTS (与Microsoft合作) |

**独特贡献**:
1. **MaskGCT** (210 citations) — 2024-2025被引最多的TTS论文之一, masked generative非自回归路线
2. **Amphion** — 唯一的全栈学术开源音频生成工具包, 提供VALL-E/NaturalSpeech复现
3. **Metis** — 从MaskGCT演进的foundation speech generation model

**判断**: **2024-2025崛起最快的学术组。** 从NaturalSpeech 3合作者到独立研究者, 武执政已建立完全独立的研究议程。**微软TTS退出后, 学术端最大的受益者。**

### 2C. CUHK MMLab

MMLab在TTS/speech方向无显著工作, 强项在vision/talking face。

---

## 实验室3: CMU (卡内基梅隆)

### 基本信息

| 项目 | 详情 |
|------|------|
| **传统TTS** | **Alan W. Black** (Festival创始人) — 已基本退出TTS合成方向 |
| **语音主力** | **Shinji Watanabe** (LTI) — 极度活跃, 但方向是ASR→Speech LM, 不是TTS |
| **新兴力量** | Siddharth Dixit, Soham Deshmukh (Audio LM/reasoning) |

### 论文 (TTS相关)

| 时间 | 论文 | 说明 |
|------|------|------|
| 2025 | Spoken LM survey | 127 citations, Watanabe领导 |
| 2025 | Mellow (NeurIPS) | 小型音频LM for reasoning |
| 2024 | UniAudio (ICML) | 多校合作, CMU非主导 |
| 2024 | Codec-SUPERB (SLT) | Codec评估基准 |

### 判断

**CMU在TTS合成方向已经边缘化。** Alan Black不再活跃在前沿; Watanabe方向偏理解不是合成; CMU的TTS贡献主要通过合作论文。**北美顶校在TTS方向确实在衰退。**

---

## 实验室4: 其他重要组/个人

### Xu Tan (谭旭) — Microsoft → Moonshot AI

| 项目 | 详情 |
|------|------|
| **现职** | **Moonshot AI (月之暗面/Kimi) — Research VP of Multimodality** |
| **前职** | Microsoft Research Asia Principal Research Manager, NaturalSpeech/VALL-E系列核心领导者 |
| **最新论文** | Llasa (ICML 2025, 共同作者); Codec Does Matter (AAAI 2025); 仍以合作者身份发speech论文 |

**判断**: 去了Moonshot做多模态, 不再全职TTS。**微软TTS人才流出的最典型案例 — 利用TTS经验进入通用多模态。**

### 微软MSR语音组现状

| 人物 | 去向 |
|------|------|
| **Xu Tan** | Moonshot AI |
| **Long Zhou, Shujie Liu, Furu Wei** | 仍在Microsoft, 方向偏通用 |
| **Chengyi Wang** (VALL-E一作) | 2025-2026无新一作论文 |

**最后产出**: NaturalSpeech 3 (ICML 2024), MELLE (2024, 与CUHK合作), E2 TTS (Interspeech 2024)

**确认: 微软TTS研究组确实在收缩/解体。** 核心人才离开后, 不再有flagship项目。

### Haizhou Li (李海州) — 港中文(深圳)/NUS

| 项目 | 详情 |
|------|------|
| **方向** | 情感/韵律/口音的表达性TTS, conversational speech synthesis |
| **产出** | 稳定但偏传统 (accent/emotion/conversation) |
| **判断** | 学术资深但不在codec LM/零样本TTS主战场 |

### KAIST — Joon Son Chung / Ji-Hoon Kim 组

| 项目 | 详情 |
|------|------|
| **特色** | 专注**视觉-语音多模态** (face→speech, lip→speech, video dubbing) |
| **论文** | VoiceCraft-Dub (ICCV 2025), Face-StyleSpeech (ICASSP 2025), AlignDiT (ACM MM 2025) |
| **判断** | 在talking face/dubbing交叉领域非常强, 不在纯TTS主赛道 |

### SNU (首尔国大)

- EmoSphere++ (情感TTS), RapFlow-TTS (flow matching)
- 中等活跃度, 方向偏情感/韵律控制

---

## 全球学术版图总结

### 微软退出后的权力真空 — 谁在填补?

1. **武执政 Mel Lab (CUHK-SZ)** — MaskGCT(210 citations)+Amphion, 最直接继承者
2. **ZJU Zhou Zhao组** — WavTokenizer+Mega-TTS系列, 但后者依赖ByteDance合作
3. **工业界** (非学术) — ByteDance/Alibaba/HeyGen等

### 北美 vs 亚洲

| 维度 | 北美 | 亚洲 |
|------|------|------|
| 活跃TTS组数量 | 1-2 (Watanabe偏理解) | 6+ (ZJU/CUHK-SZ/SJTU/CUHK/清华/KAIST/SNU) |
| 旗舰论文 | 低(合作参与为主) | 高(MaskGCT/WavTokenizer/MegaTTS3/DualSpeechLM全来自亚洲) |
| 开源工具 | Codec-SUPERB(评估) | Amphion(全栈)/WavTokenizer |

**结论: TTS研究重心已完全转移到亚洲。** 北美(CMU)在TTS合成基本退出。

### 人才流动方向

| 人物 | 从→到 | 趋势 |
|------|------|------|
| Xu Tan | Microsoft→Moonshot AI | TTS专家→多模态VP |
| Yi Ren | ZJU学术→TikTok→HeyGen | TTS→视频生成 |
| Rongjie Huang | ZJU→CUHK | 留学术但偏多模态 |
| 微软TTS组 | Microsoft→分散 | 核心外流, 组解体 |

**趋势**: 人才从(1)学术→工业, (2)纯TTS→多模态/大模型, (3)北美→亚洲

### 研究范式分布

| 范式 | 代表 | 状态 |
|------|------|------|
| Masked Generative(非AR) | 武执政(MaskGCT/Metis) | 上升中 |
| AR Codec LM | ZJU(Mega-TTS)/SJTU(TacoLM) | 主流但趋同 |
| LLM-based TTS | Llasa(多校)/DualSpeechLM(CUHK) | 新兴热点 |
| 连续值AR(无VQ) | MELLE(Microsoft+CUHK) | 有前景但Microsoft停更 |
| 表达性/情感TTS | Haizhou Li/SNU | 稳定但非主流 |
| 视觉-语音多模态 | KAIST(Chung组) | 独特赛道 |
| Codec设计 | ZJU(WavTokenizer)/CUHK-SZ(DualCodec) | 基础设施层 |
