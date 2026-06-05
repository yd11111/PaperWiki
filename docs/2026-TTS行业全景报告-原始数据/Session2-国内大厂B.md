# Session 2: MiniMax + 小红书/FireRedTTS + 科大讯飞 + 蚂蚁/Ming

> 调研日期: 2026-06-05
> 方法: 5层搜索法 (组织搜索 + 核心人搜索 + affiliation搜索 + 产品/竞赛反推 + 引用网络)

---

## 团队1: MiniMax / 海螺AI

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | MiniMax (稀宇科技) |
| **团队名** | MiniMax Speech |
| **GitHub org** | [MiniMaxAI](https://github.com/MiniMaxAI) (无公开 repos) |
| **HuggingFace org** | [MiniMaxAI](https://huggingface.co/MiniMaxAI) (tech report space, 多语言测试集) |
| **核心人物** | **Bowen Zhang** — MiniMax-Speech 一作; **Haozhe Zhang** — 通讯作者; **Junjie Yan (闫俊杰)** — CEO/联合创始人, 挂名作者; 共 20 位作者 |
| **产品线** | 海螺AI (Hailuo AI), MiniMax Audio API (Speech 2.5 Turbo / Speech 2.6 HD), 海螺视频 (Hailuo Video) |
| **开源态度** | 核心模型不开源; 仅发布 tech report + 多语言测试集; API 商业化 |

### 论文时间线 (2024-2026, 1篇核心)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2025.05 | **MiniMax-Speech** — AR Transformer + 可学习说话人编码器 + Flow-VAE | preprint (2505.07916) | TTS旗舰 |

> **说明**: MiniMax 在 TTS 领域仅公开 1 篇论文, 但该论文引用量极高 (40+ citations in ~1年), 且模型商业表现强劲 (TTS Arena #1)。其他产品迭代 (Speech 2.5/2.6) 未发表论文, 属于纯商业迭代。MiniMax-01 (LLM, 456B MoE) 是独立的语言模型工作。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Tokenizer/Codec** | 可学习说话人编码器 (learnable speaker encoder), 从参考音频提取 timbre features, 不需要转写; Flow-VAE 增强声学质量 |
| **生成模型** | 自回归 Transformer; 支持 32 种语言 |
| **Post-training** | 公开信息有限; LoRA 实现情感控制 (扩展能力, 不改基础模型) |
| **流式/效率** | Speech 2.5 Turbo: 低延迟商业版; Speech 2.6 HD: 高质量版 |
| **评估** | TTS Arena #1 (2025); 发布 TTS-Multilingual-Test-Set (24语言) |
| **其他** | 扩展功能: T2V (text-to-voice, 从文本描述合成音色); PVC (professional voice cloning, 微调音色); LoRA 情感控制 |

### 架构演进

```
MiniMax-Speech (2025.05) — AR Transformer + learnable speaker encoder + Flow-VAE
    ↓ 商业迭代
Speech 2.5 Turbo (商业) — 低延迟优化版
    ↓
Speech 2.6 HD (商业) — 高质量版
    ↓ 扩展
LoRA 情感控制 / T2V 文本生成音色 / PVC 专业克隆

(公开论文仅 1 篇, 商业迭代不发表论文)
```

### 独特技术赌注

1. **可学习说话人编码器**: 不像 CosyVoice 用 ASR 监督或 ECAPA-TDNN 固定提取, 而是端到端学习 timbre features, 无需参考音频转写
2. **Flow-VAE**: 在 AR 基础上用 Flow-VAE 增强音质, 而非常见的 flow matching decoder
3. **模型不开源 + API 商业化**: 与字节类似, 核心模型不公开, 靠商业 API 变现
4. **"基础模型不动"的扩展策略**: LoRA/T2V/PVC 均不修改基础模型, 保持模块化

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| MiniMax-Speech 模型 | 不开源 | 仅 API 商业化 |
| TTS-Multilingual-Test-Set | HF dataset | 24 语言测试集 (开源) |
| MiniMax-Speech Tech Report | HF space | 论文在线阅读 |

### 判断

- **优势**: TTS Arena #1 (行业最佳商业 TTS 之一); 32 语言支持广; 模块化扩展策略 (LoRA/T2V/PVC) 灵活; 海螺生态 (视频+音频) 完整
- **短板**: 仅 1 篇论文, 技术深度不透明; 完全不开源, 社区影响力弱; 团队研究产出密度低 (对比字节 40+ 篇、阿里 35+ 篇); 没有 ASR/VC/全双工等横向扩展
- **下一步推测**: 可能继续纯商业路线, 不发表更多论文; 全双工对话能力 (跟进海螺AI产品需求); 视频+语音联合生成 (配合 Hailuo Video)
- **关键观察**: MiniMax 是"一篇论文打天下"的典型案例 -- 靠产品和商业化而非论文量取胜

---

## 团队2: 小红书 / FireRedTeam

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 小红书 (Xiaohongshu / RedNote) - Super Intelligence 部门 - 基础算法实验室 |
| **团队名** | FireRedTeam |
| **GitHub org** | [FireRedTeam](https://github.com/FireRedTeam) (20 repos, 568 followers) |
| **HuggingFace** | [FireRedTeam](https://huggingface.co/FireRedTeam) |
| **核心人物** | **Hao-Han Guo (郭浩翰)** — FireRedTTS/SoCodec/FireRedTTS-1S 一作; **Kun Xie (谢坤)** — FireRedTTS-2 一作; **Fei-Yu Shen (沈飞宇)** — 核心作者; **Yao Hu (胡耀)** — senior/管理层; **Fenglong Xie (谢凤龙)** — 多篇 senior |
| **产品线** | 小红书 APP 内语音功能 (配音/聊天/播客) |
| **外部合作** | Tan Lee (港中文语音组) — PodAgent/SoCodec 合作者; Helen Meng (港中文) — SoCodec |

### 论文时间线 (2024-2026, 6篇+)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2024.09 | **FireRedTTS** — Foundation TTS框架 (400M LM + CFM + 248k小时) | arXiv (2409.03283), 96 citations | TTS旗舰 |
| 2024.09 | **SoCodec** — 语义有序多流语音 codec | SLT 2024, 20 citations | Codec |
| 2025.03 | **FireRedTTS-1S** — 升级版流式 TTS | arXiv, 9 citations | 流式TTS |
| 2025.03 | **PodAgent** — 综合播客生成框架 | ACL 2025 Findings, 7 citations | 播客生成 |
| 2025.09 | **FireRedTTS-2** — 长对话多说话人 TTS (dual-transformer) | arXiv (2509.02020), 31 citations | 对话TTS |
| 2025.09 | **FireRedChat** — 可插拔全双工语音交互 | arXiv (2509.06502), 16 citations | 全双工 |

> 另有 FireRedASR (开源工业级 ASR)、FireRedASR2S、FireRedVAD 等非 TTS 但相关的开源工作。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Tokenizer/Codec** | SAST (Semantic-Aware Speech Tokenizer, HuBERT + ECAPA-TDNN + Clip&Shuffle) → SoCodec (语义有序多流) → FireRedTTS-2: 12.5Hz streaming tokenizer (Whisper + RVQ) |
| **生成模型** | FireRedTTS: 400M decoder-only Transformer → FireRedTTS-2: Dual-Transformer (backbone预测第1层RVQ + decoder预测剩余层), 基于 Qwen2.5 |
| **Post-training** | Instruction tuning (13种副语言行为); 三阶段课程学习 (mono → dialogue → SFT) |
| **流式/效率** | FireRedTTS-1S: Streamable decoder (multi-stream LM + Mel Codec); FireRedTTS-2: text-speech interleaved format 流式逐句生成 |
| **评估** | SEED-TTS-Eval; 自建 podcast 评测 |
| **其他** | ASR (FireRedASR 系列, 开源 SOTA 中文 ASR); VAD (FireRedVAD); 全双工 (FireRedChat); 播客 (PodAgent) |

### 架构演进

```
FireRedTTS (2024.09) — SAST + 400M AR LM + Flow-Matching decoder + BigVGAN
    ↓ +codec创新                    ↓ +流式
SoCodec (2024.09)             FireRedTTS-1S (2025.03)
语义有序多流codec               升级版流式TTS
    ↓ 融合                          ↓
FireRedTTS-2 (2025.09) — 12.5Hz streaming tokenizer + dual-transformer + interleaved format
    ↓ +全双工
FireRedChat (2025.09) — 可插拔全双工系统 (cascaded + semi-cascaded)

并行:
PodAgent (2025.03) — 多智能体播客生成框架 (ACL Findings)
FireRedASR → FireRedASR2S — 开源工业级 ASR
```

### 独特技术赌注

1. **12.5Hz 超低帧率 tokenizer**: 帧率仅为主流方案 (25-50Hz) 的一半, 大幅缩短序列长度, 解锁长对话建模
2. **Dual-Transformer**: backbone + decoder 分工, 比 delay-pattern 更好利用上下文
3. **Text-speech interleaved format**: 多说话人对话建模为单序列 [S1]text+audio[S2]text+audio..., 天然支持流式
4. **Clip&Shuffle 防信息泄漏**: tokenizer 训练时打乱声学编码器输入, 防止 content 信息泄漏到 timbre 表示
5. **全栈语音工具链**: TTS + ASR + VAD + 全双工 + 播客, 覆盖面仅次于字节和阿里

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| FireRedTTS | 911 | 代码+权重 (MPL-2.0) |
| FireRedTTS2 | 1.4k | 代码+权重 (Apache-2.0) |
| FireRedASR | 1.9k | 工业级 ASR, 中文 SOTA |
| FireRedChat | 535 | 全双工交互系统 |
| FireRedASR2S | 535 | ASR + VAD + LID + Punc 全家桶 |
| FireRedVAD | 412 | 100+ 语言 VAD |
| FireRed-OpenStoryline | 2.9k | AI 视频编辑 agent (非语音) |

### 判断

- **优势**: 开源全栈 (TTS/ASR/VAD/全双工/播客) 生态完善; 论文质量高 (FireRedTTS-2 31次引用); 12.5Hz tokenizer 在长对话场景有独特优势; 团队产出稳定 (每半年一个版本); ASR 开源 SOTA
- **短板**: 品牌认知度不如阿里/字节; TTS 未成为行业事实标准; 英语能力较弱 (EN overall error 12%); 社区影响力中等 (对比 CosyVoice 9k, Spark-TTS 11k)
- **下一步推测**: FireRedTTS-3 可能进一步统一 ASR+TTS; 全双工 + LLM 集成 (FireRedChat 升级); podcast/长音频场景继续深耕
- **关键观察**: 小红书在语音方向是"闷声做事型" -- 论文不多但质量高, 开源不搞噱头但覆盖面广

---

## 团队3: 科大讯飞 (iFLYTEK)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 科大讯飞 (iFLYTEK) - 中国最大语音 AI 公司 |
| **团队名** | iFLYTEK Research (上海/合肥) + 中科大 (USTC) 合作 |
| **GitHub org** | [iflytek](https://github.com/iflytek) (63 repos, 543 followers) |
| **HuggingFace** | 无独立 org; 讯飞星火模型通过商业 API 提供 |
| **核心人物** | 公开论文中可见: **Jun Du (杜俊)** — USTC 教授, 与讯飞深度合作; **Yu Hu (胡煜)** — iflytek.com 邮箱, 多篇论文; **Jiahong Yuan (袁甲鸿)** — USTC, 与讯飞合作 |
| **产品线** | 讯飞开放平台 (iFLYTEK Open Platform) — 中国最大语音 API; 讯飞星火 (Spark) 大模型; 讯飞输入法; 讯飞翻译机; 讯飞智能会议系统 |
| **开源态度** | 核心 TTS 技术完全不开源; GitHub 以企业工具 (agent/RPA) 为主; 无 TTS 相关开源 |

> **重要澄清**: Spark-TTS 与科大讯飞无关! Spark-TTS (BiCodec) 的作者来自 HKUST、Mobvoi (出门问问)、西北工大等, 是独立的开源社区项目 (SparkAudio Open Source Community)。讯飞星火 (iFLYTEK Spark) 是商业产品, Spark-TTS 是学术项目, 两者仅同名无关联。

### 论文时间线 (2024-2026)

> **公开信息有限**: 科大讯飞在 TTS 领域几乎没有公开 arXiv 论文。以下为可确认的与讯飞有关的语音研究:

| 时间 | 论文 | 方向 | 说明 |
|------|------|------|------|
| 2025.08 | **EGGCodec** — EGG 信号的 neural codec | Codec (非TTS) | USTC + iFLYTEK 合作 (yuhu@iflytek.com) |
| 2026 | **DARS** — 构音障碍韵律合成 | 医疗TTS | iFlytek Co., Ltd + Huawei |
| 2026 | **端到端构音障碍语音重建** | 医疗语音 | iFlytek Co., Ltd |
| 2026 | **Human or Machine? Turing Test** | 评估 | iFLYTEK-Spark 作为被测系统之一 |

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Tokenizer/Codec** | 公开信息有限; 商业系统内部技术未公开 |
| **生成模型** | iFLYTEK-Spark TTS: 商业 API, 技术细节未公开 |
| **Post-training** | 不详 |
| **流式/效率** | 讯飞开放平台提供流式 TTS API |
| **评估** | 在 "Human or Machine?" 论文中, iFLYTEK-Spark 的人类误判率仅 6.43%, 表明商业 TTS 质量高 |
| **其他** | ASR (讯飞是中国 ASR 市场份额最大的公司); 翻译; 输入法; 会议系统 |

### 架构演进

```
(公开信息极为有限, 以下为推测性描述)

讯飞开放平台 TTS (传统参数合成/统计方法, ~2012起)
    ↓
讯飞星火大模型语音能力 (2023+) — 基于 Spark LLM 的语音生成
    ↓
iFLYTEK-Spark TTS API (2024+) — 商业最新版

并行:
EGGCodec (2025) — USTC-iFLYTEK 合作, EGG 信号 codec (非主流 TTS)
DARS (2026) — 医疗/构音障碍 TTS (特殊场景)
```

### 独特技术赌注

1. **商业 API 壁垒**: 讯飞的核心优势不在论文而在产品 -- 中国最大的语音开放平台, B 端客户基数最大
2. **全场景覆盖**: 输入法/翻译/会议/教育/医疗/车载, 语音能力嵌入到几乎所有 to-B 场景
3. **USTC 合作网络**: 与中科大有深度绑定 (人员/项目), 基础研究通过高校合作完成
4. **不发论文路线**: 与 MiniMax 类似, 讯飞选择不公开技术细节, 靠商业规模竞争

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| astron-agent | 8.5k | 企业 agent 工作流平台 (非语音) |
| astron-rpa | 5.2k | RPA 自动化 (非语音) |
| skillhub | 3.3k | agent skill 注册中心 (非语音) |
| iFly-Skills | 179 | 语音/OCR/翻译等 skill 集合 (封装 API, 非模型开源) |

> **注意**: iFLYTEK 的 GitHub 63 个 repos 中, 无一与 TTS 模型/训练相关。全部为企业工具、SDK 封装、教程等。

### 判断

- **优势**: 中国语音市场份额最大; B 端客户基数庞大 (政府/教育/医疗/金融); 20+ 年语音技术积累; 产品矩阵完善 (翻译机/输入法/会议)
- **短板**: TTS 研究产出极度匮乏 (几乎零 arXiv 论文); 完全不开源; 在 LLM 时代的技术竞争力不透明; 社区影响力趋近于零; 在 TTS Arena 等公开排行榜上表现一般
- **下一步推测**: 可能通过讯飞星火大模型整合语音能力 (类似 Qwen3-TTS 路线); 继续以 to-B 商业为主, 不太可能转向开源
- **关键观察**: 讯飞是"商业巨头, 研究侏儒"的典型 -- 在 LLM 时代 TTS 研究论文几乎缺席, 靠存量商业优势维持市场地位。这与其在 ASR 时代的技术领导力形成鲜明对比。Turing Test 论文中 iFLYTEK-Spark 人类误判率仅 6.43% (最低之一), 说明其商业 TTS 产品在质量上仍有一定水准, 但远未达到 MiniMax/Seed-TTS 的先进水平。

---

## 团队4: 蚂蚁集团 / Ming (Inclusion AI)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 蚂蚁集团 (Ant Group) - Inclusion AI 实验室 |
| **团队名** | Inclusion AI |
| **GitHub org** | 无独立语音 GitHub org; Ming 系列模型通过 HuggingFace 发布 |
| **HuggingFace org** | [inclusionAI](https://huggingface.co/inclusionAI) (2.1k+ followers) |
| **核心人物** | **Canxiang Yan (严灿祥)** — Ming-UniAudio 一作; **Jingdong Chen** — senior; **Jun Zhou** — senior; **Ming Yang** — senior; 共 25 位作者 |
| **产品线** | 蚂蚁集团内部 AI 能力 (客服/风控/支付宝语音); Ming 系列大模型 |
| **开源态度** | 模型权重开源 (HuggingFace); 论文发表活跃; Ming-UniAudio-16B-A3B + MingTok-Audio 均开源 |

### 论文时间线 (2024-2026, 6篇+)

| 时间 | 论文 | 方向 | 说明 |
|------|------|------|------|
| 2024 | **Codec does matter** — 探索 codec 语义缺陷 | Codec分析 | 前期研究 |
| 2025.05 | **Ming-Lite-Uni** — 轻量统一多模态模型 | 多模态 | 统一架构探索 |
| 2025.06 | **Ming-Omni** — 统一多模态感知+生成 (7B) | 多模态旗舰 | 首个 Ming 统一模型 |
| 2025.10 | **Ming-UniAudio** — 语音理解+生成+编辑统一 (16B MoE) | 语音LLM旗舰 | 连续 VAE tokenizer + free-form editing |
| 2025.10 | **Ming-Flash-Omni** — 稀疏统一多模态架构 | 多模态效率 | MoE 版本, 更高效 |
| 2025.10 | **Ming-UniVision** — 统一图像理解+生成 | 视觉 | 视觉 tokenizer |
| 2026.01 | **Robust Speech Emotion Recognition** | 语音情感 | 鲁棒 SER |
| 2026.04 | **AT-ADD Challenge** — 全类型音频深度伪造检测 | 安全/检测 | 组织竞赛 |

> 另有 BR-ASR (ASR), Ditto (talking head), 多篇 deepfake detection 等相关工作。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **Tokenizer/Codec** | **MingTok-Audio**: 因果 Transformer VAE, 3 阶段训练 (重建→语义蒸馏→联合), 统一连续表示; 高维 Zuni (LLM输入) + 低维 Zlatent (flow matching 用) |
| **生成模型** | Ming-UniAudio: 16.8B MoE LLM (2.8B active), per-token flow matching head; Ming-Omni: 7B 统一模型 |
| **Post-training** | 公开信息有限; 三阶段课程学习 (理解pretraining → 生成pretraining → SFT) |
| **流式/效率** | Ming-Flash-Omni: 稀疏 MoE 架构提升推理效率 |
| **评估** | ContextASR benchmark (8/12 项 SOTA); Seed-TTS-Eval (中文 WER 0.95%); Audio-Edit-Benchmark (自建) |
| **其他** | Free-form speech editing (首个无 timestamp 的自然语言指令编辑); Deepfake detection (AT-ADD 竞赛组织方); Talking head (Ditto) |

### 架构演进

```
"Codec does matter" (2024) — 发现 codec 语义缺陷 → 为连续 tokenizer 奠基
    ↓
Ming-Lite-Uni (2025.05) — 轻量统一多模态探索
    ↓
Ming-Omni (2025.06) — 7B 统一感知+生成
    ↓ +语音专精                    ↓ +效率
Ming-UniAudio (2025.10)      Ming-Flash-Omni (2025.10)
16B MoE, MingTok-Audio      稀疏 MoE, 离散→连续 acoustic tokens
free-form editing
    ↓ 开源
MingTok-Audio (HF, 1B)
Ming-UniAudio-16B-A3B (HF, 18B)

并行:
Ming-UniVision (2025.10) — 视觉统一 tokenizer
AT-ADD Challenge (2026) — 深度伪造检测竞赛
```

### 独特技术赌注

1. **连续 VAE tokenizer 统一路线**: MingTok-Audio 是首个有效融合语义和声学的连续 tokenizer, 使同一表示兼容理解/生成/编辑三种任务
2. **LLM 语义蒸馏**: tokenizer 训练时用冻结 LLM 的 ASR 损失反向传播, 将语义信息注入连续表示 (相比传统 HuBERT 等自监督方法)
3. **Free-form speech editing**: 首个不需要 timestamp/MFA 对齐的自然语言指令驱动语音编辑, 涵盖语义编辑 (删/插/换) + 声学编辑 (降噪/变速/变调/方言转换)
4. **Ming 统一模型家族**: 从 Omni (视觉+语音+文本) 到 UniAudio (语音专精) 到 Flash (高效), 形成矩阵
5. **Semantic module freezing**: 联合训练时冻结 tokenizer 语义模块防 representation drift, 显著提升性能 (AVG WER 4.35 vs 6.86)

### 开源情况

| 项目 | Stars/Likes | 说明 |
|------|-------------|------|
| MingTok-Audio | 29 likes (HF) | 连续语音 tokenizer, 1B 参数, 权重开源 |
| Ming-UniAudio-16B-A3B | 79 likes (HF) | 语音 LLM, 18B 参数 (MoE 2.8B active), 权重开源 |
| Ming-Flash-Omni | 14 citations | 稀疏多模态模型 |
| Ming-Omni-2.0 | 100B | 最新版本 (细节待公开) |

### 判断

- **优势**: 连续 tokenizer 路线在学术上最前沿; 理解+生成+编辑统一是独特差异化; 开源诚意高 (模型权重全放); Ming 家族矩阵布局完整 (视觉+语音+多模态); ContextASR 多项 SOTA; Seed-TTS-WER 中文 0.95% 是最佳之一
- **短板**: TTS 纯生成质量 (SIM 偏低: 中文 0.75, 英文 0.68, 低于 CosyVoice 3 的 0.78); 语音编辑 deletion WER 偏高 (22-27%); 起步较晚 (2025.06 才发第一篇); 社区影响力有限 (对比 CosyVoice/Spark-TTS); 无独立 TTS 产品线
- **下一步推测**: Ming-UniAudio 2.0 可能提升 SIM/TTS 质量; 与 Ming-Omni-2.0 (100B) 整合; free-form editing 可能成为差异化杀手锏; deepfake detection 作为安全侧产品
- **关键观察**: 蚂蚁是 Session 2 四个团队中**唯一真正走"统一模型"路线**的 -- 不是做独立 TTS 系统, 而是让语音成为 LLM 的原生能力。连续 VAE tokenizer 的赌注如果成功, 可能颠覆当前 discrete token 主导的格局。

---

## Session 2 跨团队对比

### 产出密度对比

| 团队 | 公开论文数 | 开源项目数 | 最高 Stars | 技术路线 |
|------|-----------|-----------|-----------|---------|
| **MiniMax** | 1 | 0 (测试集除外) | 无 | AR Transformer + learnable speaker encoder |
| **小红书/FireRedTeam** | 6+ | 7+ | 2.9k (OpenStoryline) / 1.9k (ASR) | LLM + semantic token + dual-transformer |
| **科大讯飞** | ~0 (TTS) | 0 (TTS) | 8.5k (agent, 非语音) | 商业 API, 技术不公开 |
| **蚂蚁/Ming** | 6+ | 2 (HF) | 79 likes (HF) | 连续 VAE tokenizer + MoE + per-token flow |

### 技术路线矩阵

| 维度 | MiniMax | 小红书 | 科大讯飞 | 蚂蚁/Ming |
|------|---------|--------|---------|----------|
| **Tokenizer** | learnable speaker encoder | SAST → 12.5Hz streaming RVQ | 不公开 | MingTok-Audio (连续 VAE) |
| **生成架构** | AR Transformer | Dual-Transformer | 不公开 | MoE LLM + per-token flow |
| **流式能力** | 商业流式 API | FireRedTTS-1S/2 | 商业流式 API | Ming-Flash-Omni |
| **全双工** | 无 | FireRedChat | 不公开 | Ming-Omni |
| **编辑能力** | T2V/PVC/LoRA | 无 | 无 | Free-form editing (首创) |
| **ASR** | 无 | FireRedASR (开源 SOTA) | 商业 SOTA (市场份额最大) | BR-ASR |

### 开源策略分类

| 策略 | 团队 | 特征 |
|------|------|------|
| **完全封闭** | MiniMax, 科大讯飞 | 模型不开源, 纯 API 商业化; 论文极少或不发 |
| **全栈开源** | 小红书/FireRedTeam | TTS+ASR+VAD+全双工 均开源; 代码+权重 |
| **模型开源** | 蚂蚁/Ming | HuggingFace 上开源权重; 无 GitHub 代码仓 |

### Session 2 团队定位总结

- **MiniMax**: "一篇论文 + 商业产品" — 靠 TTS Arena #1 的产品质量取胜, 不靠论文量
- **小红书/FireRedTeam**: "全栈开源工匠" — 覆盖 TTS/ASR/VAD/全双工/播客, 质量高但品牌弱
- **科大讯飞**: "语音老牌, 研究缺席" — 商业市场份额大, 但 LLM 时代 TTS 研究几乎空白
- **蚂蚁/Ming**: "统一模型赌注" — 连续 tokenizer + 理解/生成/编辑统一, 学术最前沿但 TTS 纯质量待提升
