# Session 2: 百度 + 讯飞 + 腾讯 + 小米

> 调研日期: 2026-06-08
> 方法: 5层搜索法 (组织搜索 + 核心人搜索 + affiliation搜索 + 产品/竞赛反推 + 引用网络)
> 方向: Speech LLM / Omni / 全双工对话

---

## 团队1: 百度 / ERNIE Team

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 百度 (Baidu) |
| **团队名** | ERNIE Team (文心团队) |
| **GitHub org** | [PaddlePaddle](https://github.com/PaddlePaddle) (PaddleSpeech 等); [baidu-research](https://github.com/baidu-research) |
| **HuggingFace** | [baidu](https://huggingface.co/baidu) (ERNIE-4.5-VL 等模型) |
| **核心人物** | **Haifeng Wang (王海峰)** — 百度 CTO, ERNIE 5.0 一作; **Hua Wu (吴华)** — NLP 首席科学家, 多篇 senior; **Yu Sun (孙宇)** — ERNIE 系列核心; **Jingzhou He (何景舟)** — 语音方向 lead; **Dan Zhang** — Eureka-Audio 一作; **Shuwei He (何树伟)** — Omni 模型核心开发, Eureka-Audio/ERNIE 5.0 核心贡献 |
| **产品线** | 文心一言 (ERNIE Bot), 文心大模型 API (ERNIE 4.5/5.0), 百度地图语音助手 (DuIVA/DuIVRS), PaddleSpeech 工具箱 |
| **开源态度** | ERNIE 4.5 部分开源 (HuggingFace); ERNIE 5.0 仅 tech report; Eureka-Audio 未开源权重; PaddleSpeech (早期工具箱) 开源 |

### 论文时间线 (2024-2026)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2022.05 | **PaddleSpeech** — All-in-One Speech Toolkit | NAACL 2022 Demo, 45 citations | 语音工具箱 |
| 2025.xx | **ERNIE 4.5 Technical Report** — 多模态大模型 (含语音) | preprint | 多模态基座 |
| 2026.01 | **MoE Adapter for Large Audio Language Models** (2601.02967) — ERNIE Team | preprint, Baidu | 音频理解 (MoE adapter) |
| 2026.01 | **Bridging the Audio-Text Reasoning Gap** (2601.16547) — ERNIE Team + 清华深研 | preprint | 音频推理 |
| 2026.02 | **ERNIE 5.0 Technical Report** (2602.04705) — 万亿参数统一多模态模型 | preprint, 4 citations | Omni 旗舰 |
| 2026.02 | **Eureka-Audio** (2602.13954) — 1.7B 轻量音频语言模型 | preprint, 1 citation | 音频理解 (小模型) |
| 2026.05 | **DuIVRS-2** (2605.17900) — LLM-based 交互语音响应系统 | preprint | 语音交互产品 |
| 2026.05 | **Native Audio-Visual Alignment for Generation** (2605.30073) — ERNIE Team | preprint | 音视频对齐 |
| 2026.05 | **High-Fidelity Codec-Inspired Residual Modeling** (2605.26967) — ERNIE Team | preprint | 视频理解 (codec思路) |

> **说明**: 百度在 Speech LLM 方向的核心产出集中在 ERNIE 5.0 (统一多模态 Omni) 和 Eureka-Audio (轻量音频理解) 两条线。与阿里 (FunAudioLLM 全栈) 和字节 (Seed 系列) 相比, 百度在独立的 TTS/ASR/全双工论文上产出较少, 主要整合进 ERNIE 大模型体系中。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **语音编码器** | Whisper-based audio encoder (Eureka-Audio); 自研 audio tokenizer (ERNIE 5.0, 将音频转为 discrete tokens 与文本统一训练) |
| **LLM 骨干** | ERNIE 5.0: 万亿参数 ultra-sparse MoE, 所有模态从头训练 next-group-of-tokens prediction; Eureka-Audio: 1.7B lightweight backbone |
| **语音解码器** | ERNIE 5.0: 统一自回归生成 (text/image/video/audio 共用解码); 具体 vocoder 未公开 |
| **对话策略** | DuIVRS-2: LLM-based 交互式语音响应, 含 ASR 纠错; 全双工能力: 公开信息有限 |
| **训练数据规模** | ERNIE 5.0: 大规模预训练 (具体音频数据量未公开); Eureka-Audio: 未公开 |
| **推理延迟** | 未公开具体数据; ERNIE 5.0 强调 elastic training 支持不同延迟-性能 trade-off |
| **多语言** | ERNIE 5.0 原生多语言; Eureka-Audio 支持英中 |
| **情感/副语言** | ERNIE 5.0 支持语音情感理解 (benchmark 评估); Eureka-Audio 支持 paralinguistic 分析 |

### 架构演进

```
PaddleSpeech (2022) — 传统语音工具箱 (ASR/TTS/分类)
    ↓ LLM 时代转型
ERNIE 4.0/4.5 (2024-2025) — 文本为主 + 语音插件式接入
    ↓ 原生多模态
ERNIE 5.0 (2026.02) — 万亿参数 Omni: text+image+video+audio 从头统一训练
    ↓ 并行: 轻量路线
Eureka-Audio (2026.02) — 1.7B 轻量 ALM: Whisper encoder + MoE Adapter + 小 LLM
    ↓ 研究支撑
MoE Adapter (2026.01) — 音频适配器创新 (解决梯度冲突)
Bridging Audio-Text Gap (2026.01) — 音频推理增强
    ↓ 产品落地
DuIVRS-2 (2026.05) — 百度地图语音交互系统 (cascaded LLM pipeline)
```

### 独特技术赌注

1. **统一自回归 Omni**: ERNIE 5.0 是国内首个公开的万亿参数统一自回归 Omni 模型, 所有模态 (text/image/video/audio) 从头用 next-group-of-tokens prediction 联合训练, 不是后接的模态适配器
2. **Elastic Training**: 单次预训练同时学习不同深度/expert/sparsity 的子模型家族, 支持灵活的性能-延迟 trade-off, 这在业界较为独特
3. **MoE Adapter for Audio**: 提出稀疏化的 MoE 适配器解决音频多属性 (语音/音乐/环境声) 之间的梯度冲突问题, 而非简单的 dense adapter
4. **轻量路线并行**: 在 ERNIE 5.0 旗舰之外, 同时发展 1.7B 的 Eureka-Audio, 目标是小模型匹配 7B-30B 性能
5. **产品导向的 Voice Interaction**: DuIVRS/DuIVA 系列专注于真实场景 (百度地图) 的语音交互, 而非纯 benchmark 导向

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| PaddleSpeech | 10.8k | 早期语音工具箱 (ASR/TTS/分类), 非 LLM 路线 |
| ERNIE 4.5 VL 系列 | HF 模型 | 部分视觉-语言模型开源 |
| ERNIE 5.0 | 不开源 | 仅 tech report, 核心模型不公开 |
| Eureka-Audio | 不开源 | 仅论文, 模型权重未发布 |

### 判断

- **优势**: 万亿参数 Omni 统一架构 (ERNIE 5.0) 在国内规模最大; Elastic Training 是独特的部署创新; ERNIE Team 论文密度在 2026 年加速 (音频方向 5+ 篇); 百度地图等落地场景提供真实反馈
- **短板**: Speech LLM 方向论文起步晚 (2026 年才集中发力), 比阿里/字节晚 1-2 年; 独立的 TTS/ASR/全双工论文缺失 (全部整合进 ERNIE 体系); 开源力度弱, 社区影响力有限; 缺少独立的 speech codec 或 speech tokenizer 研究
- **下一步推测**: ERNIE 5.0 可能推出语音对话版本 (voice chat); Eureka-Audio 可能扩展到 7B 并开源; 可能发布独立的 Speech LLM 技术报告 (从 ERNIE 体系中拆分)
- **关键观察**: 百度走的是"大一统 Omni"路线, 语音不是独立方向而是 ERNIE 多模态能力的一部分。这与阿里 (FunAudioLLM 独立团队) 和腾讯 (Covo-Audio 独立项目) 的策略不同

---

## 团队2: 科大讯飞 / iFLYTEK Research

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 科大讯飞 (iFLYTEK Co., Ltd.) + 中国科学技术大学 (USTC) |
| **团队名** | iFLYTEK Research / iFLYTEK AI Research + USTC 语音及语言信息处理国家工程研究中心 |
| **GitHub org** | [iflytek](https://github.com/iflytek) (主要是 Agent/RPA 工具, 无 Speech LLM 开源) |
| **HuggingFace** | 无专门的 org (公开信息有限) |
| **核心人物** | **Zhi-Ling Zheng (郑之灵)** — USTC 教授, 语音合成; **Dan Su** — iFLYTEK Research; **Yongzhe He** — Spark 语音模型; 大量 USTC-iFLYTEK 联合培养博士生 |
| **产品线** | 讯飞星火 (Spark) 大模型, 讯飞输入法, 讯飞听见 (转写), 讯飞 AI 翻译机, 讯飞开放平台 (ASR/TTS API), Spark Speech-to-Speech Translation |
| **开源态度** | 产品/模型几乎不开源; GitHub 以 Agent 工具为主; 学术论文主要走 USTC 联合发表渠道 |

### 论文时间线 (2024-2026)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2024.06 | **Generative Pre-trained SLM with Efficient Audio Tokenizer** (2406.00976) — USTC + iFLYTEK | preprint | 语音语言模型 |
| 2024.12 | **Diffusion-based Hierarchical Prosody Modeling for TTS** — USTC + iFLYTEK | preprint | TTS 韵律 |
| 2025.02 | **Audio-Visual Representation Learning via Knowledge Distillation** — iFLYTEK-Speech model | preprint | 音视频表征 |
| 2025.10 | **Adapting Speech Foundation Models with LLM** (2510.22961) — iFLYTEK Research | preprint | 语音基础模型 + LLM |
| 2026.01 | **Streaming Speech Recognition with Decoder-Only LLM** (2601.22779) — iFLYTEK | preprint, 2 citations | 流式 ASR |
| 2026.01 | **SLM-SS: Speech Language Model for Generative Speech Separation** — iFLYTEK AI Research | preprint | 语音分离 |
| 2026.04 | **Beyond Monologue: Interactive Talking-Listening Avatar** (2604.10367) — USTC + iFLYTEK | preprint | 全双工数字人 |
| 2026.06 | **Cognitive-State-Conditioned TTS Data Augmentation** (2606.06170) — USTC | preprint | 认知状态 TTS |

> **说明**: 讯飞在 Speech LLM 方向的公开论文显著少于百度/腾讯/小米。其核心技术 (星火语音交互) 主要以产品形式释放, 不发表独立论文。USTC 联合发表的论文更偏基础研究 (TTS 韵律、语音分离、ASR), 而非端到端 Speech LLM 或 Omni 模型。**这是 4 家中 Speech LLM 公开研究最少的团队。**

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **语音编码器** | iFLYTEK-Speech (自研语音基础模型, 用于 AVSR/ASR); WavLM 等开源模型作为 baseline |
| **LLM 骨干** | 讯飞星火 (Spark) 大模型系列 (4.0 Turbo / Spark X1 深度推理); 具体架构未公开论文 |
| **语音解码器** | 讯飞 TTS API (支持多语言/多音色/情感); 具体 LLM-based TTS 技术未公开 |
| **对话策略** | 星火语音交互: cascaded 方案 (ASR → Spark LLM → TTS); TrueVoice 交互技术 (2024.08 升级); 公开信息有限 |
| **训练数据规模** | 海量语音数据 (讯飞 20+ 年语音技术积累, 数十亿小时级别), 但具体用于 Speech LLM 的数据量未公开 |
| **推理延迟** | Spark 语音交互: 产品级实时响应; 具体技术指标未公开 |
| **多语言** | Spark Multilingual (2024.10 发布, 9 语言); SPARK Speech-to-Speech Translation (同传) |
| **情感/副语言** | TrueVoice 交互技术 (更自然的语音交互); 产品端支持情感语音 |

### 架构演进

```
讯飞传统语音技术 (1999-2023) — ASR/TTS/语音评测, 20+ 年积累
    ↓ LLM 时代
星火 Spark 1.0-3.0 (2023-2024) — 文本 LLM + 语音 API cascaded 接入
    ↓ 产品升级
Spark 4.0 Turbo (2024.10) — 多语言 + TrueVoice 交互 + 9 语言支持
    ↓ 深度推理
Spark X1 (2025.01) — 深度推理版, 行业首个 (类 o1)
    ↓ 语音同传
SPARK Speech-to-Speech Translation (2025) — 实时语音翻译大模型
    ↓ 学术探索
Beyond Monologue (2026.04) — USTC + iFLYTEK 全双工数字人 (学术探索)

(公开论文极少, 核心技术以产品发布为主, 不走学术发表路线)
```

### 独特技术赌注

1. **产品优先、论文极少**: 讯飞是 4 家中唯一将语音交互核心技术完全封装在产品中的公司, 几乎不发表独立的 Speech LLM 论文
2. **20+ 年语音数据积累**: 讯飞在中国语音市场占有率领先, 拥有海量的语音交互数据, 这是其最大的竞争壁垒
3. **TrueVoice 交互技术**: 2024 年 8 月升级的语音交互技术, 目标是更自然的人机对话, 但具体架构未公开
4. **语音同传**: SPARK Speech-to-Speech Translation 是国内最早的实时语音翻译大模型之一
5. **USTC 联合培养**: 与 USTC 的深度绑定确保了基础研究人才供给, 但研究成果多以学术论文发表, 不直接对应产品

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| iflytek/astron-agent | ~1k | Agent/RPA 工具 (非语音方向) |
| iFly-Skills | 少量 | 讯飞技能集合 (OCR/翻译/语音 API 封装) |
| Spark 模型 | 不开源 | 核心模型完全闭源, 仅 API 商业化 |
| iFLYTEK-Speech | 不开源 | 自研语音基础模型, 仅在论文中提及 |

### 判断

- **优势**: 中国语音市场领先 (输入法/转写/翻译机); 20+ 年语音数据积累; 产品矩阵完整 (to B + to C); Spark 同传翻译能力; 与 USTC 深度绑定的人才供给
- **短板**: **Speech LLM 方向公开研究严重不足** — 无端到端 Speech LLM 论文、无 Omni 模型论文、无独立的 speech codec 研究; 开源生态接近空白; 在 Turing Test benchmark 中 iFLYTEK-Spark 表现一般 (被 Qwen2.5-Omni 等超越); 技术透明度极低, 无法评估其端到端能力
- **下一步推测**: 可能在 2026 下半年发布 Spark Omni 或 Spark Voice 技术报告; 可能加大 USTC 联合研究在 Speech LLM 方向的投入; 产品端可能率先实现端到端语音交互但不公开技术细节
- **关键观察**: 讯飞在 Speech LLM 研究竞争中明显落后于其产品市场地位。作为中国语音技术龙头, 其在端到端 Speech LLM / Omni 模型方向的公开研究几乎为零, 形成了"产品强、研究弱 (至少在公开层面)"的反差。这可能意味着: (a) 核心技术保密不发表; (b) 仍在 cascaded 路线上, 尚未转向端到端; 或 (c) 正在追赶但未发布

---

## 团队3: 腾讯 / Tencent AI Lab

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 腾讯 (Tencent) — 多个研究团队 |
| **团队名** | Tencent AI Lab (Covo-Audio 核心团队, Dong Yu 领导) + ARC Lab + Tencent Hunyuan |
| **GitHub org** | [Tencent](https://github.com/Tencent) (Covo-Audio 开源); [TencentARC](https://github.com/TencentARC) |
| **HuggingFace** | [tencent](https://huggingface.co/tencent) (Covo-Audio-Chat 模型) |
| **核心人物** | **Dong Yu (俞栋)** — Tencent AI Lab 副主任, 语音领域权威, Covo-Audio senior; **Meng Yu (余萌)** — Covo-Audio 核心; **Wenfu Wang** — Covo-Audio 一作; **Hao Zhang** — 全双工对话管理一作; **Rilin Chen** — 多篇核心作者 |
| **产品线** | 腾讯混元 (Hunyuan) 大模型, 微信/QQ 语音功能, 腾讯云语音 API, 腾讯会议转写 |
| **开源态度** | Covo-Audio-Chat 开源 (CC BY 4.0); Hunyuan 系列部分开源; 研究论文积极发表 |

### 论文时间线 (2024-2026)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2025.02 | **LLM-Enhanced Dialogue Management for Full-Duplex SDS** (2502.14145) — Tencent AI Lab | preprint, 12 citations | 全双工对话管理 |
| 2025.11 | **Harmony: Harmonizing Audio and Video Generation** — Tencent Hunyuan | preprint | 音视频联合生成 |
| 2025.11 | **UniAVGen: Unified Audio and Video Generation** — Tencent Hunyuan | preprint | 统一音视频生成 |
| 2026.01 | **Towards Fine-Grained and Multi-Granular Contrastive Audio-Text** (2601.03065) — Tencent | preprint | 音频-文本对比学习 |
| 2026.02 | **Covo-Audio Technical Report** (2602.09823) — Tencent | preprint, 3 citations | **Speech LLM 旗舰** |
| 2026.04 | **OmniScript: Audio-Visual Script Generation** (2604.11102) — ARC Lab, Tencent | preprint | 音视频脚本生成 |
| 2026.04 | **Unified Audio Schema for Perception-Aware AudioLLMs** (2604.12506) — Tencent 关联 | preprint | 音频 LLM schema |
| 2026.05 | **HunyuanVideo-Avatar** — Tencent Hunyuan | preprint | 数字人 (语音驱动) |
| 2026.06 | **UAT: Unified Audio-Text Diffusion** (2606.04939) — Tencent | preprint | 统一音频生成 |
| 2026.06 | **AnyAudio-Judge: Dynamic Rubric-Based Benchmark** (2606.03116) — Tencent Hunyuan | preprint | 音频评估 |

> **说明**: 腾讯在 Speech LLM 方向的核心突破是 Covo-Audio (2026.02), 这是国内大厂中较早的独立端到端 LALM。Tencent AI Lab (Dong Yu) 专注对话/语音, Hunyuan 团队偏音视频生成, ARC Lab 偏多模态理解。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **语音编码器** | Whisper encoder → 3 级 conv adapter (6.25 Hz 极低帧率); WavLM-large based VQ tokenizer (codebook 16384, 单层) |
| **LLM 骨干** | Qwen2.5-7B (Covo-Audio); Hunyuan-7B (其他项目) |
| **语音解码器** | Flow Matching decoder → BigVGAN vocoder (24K); 离散 speech tokens (WavLM-based) |
| **对话策略** | Intelligence-Speaker Decoupling (解耦智力与音色); Hybrid Dual-Stream 全双工 (连续输入+离散输出); LLM-Enhanced Dialogue Management (VAP + Turn-taking) |
| **训练数据规模** | 预训练 2T tokens (Covo-Audio); 三阶段: hierarchical tri-modal interleaving 预训练 + 后训练 + 全双工训练 |
| **推理延迟** | 全双工 turn-taking 99.7%, pause handling 97.6%; 具体首包延迟未公开 |
| **多语言** | 中英为主 (Covo-Audio); Hunyuan 多语言 |
| **情感/副语言** | VStyle 共情评估 anxiety 5.00; 情感理解 + 共情回复; empathetic response generation |

### 架构演进

```
腾讯传统语音 (微信/QQ/会议) — cascaded ASR+NLU+TTS
    ↓ AI Lab 全双工研究
LLM-Enhanced Dialogue Management (2025.02) — 全双工对话管理框架 (VAP + LLM)
    ↓ 端到端突破
Covo-Audio (2026.02) — 7B 端到端 LALM: continuous-in + discrete-out
    ├── Covo-Audio-Base: 预训练 (speech-text modeling, 理解)
    ├── Covo-Audio-Chat: 对话版 (instruction following, 共情)
    └── Covo-Audio-Chat-FD: 全双工版 (hybrid dual-stream)
    ↓ 并行: Hunyuan 多模态
Harmony / UniAVGen / HunyuanVideo-Avatar — 音视频联合生成
AnyAudio-Judge — 音频评估 benchmark
UAT — 统一音频-文本扩散模型
```

### 独特技术赌注

1. **Intelligence-Speaker Decoupling**: Covo-Audio 的核心创新 — 用多说话人训练解耦对话智力与说话人声音, 再用 TTS 数据伪对话 (masked text loss) 转移高质量声音, 成本远低于为每个声音收集对话数据。这是与 Qwen-Omni Thinker-Talker 完全不同的思路
2. **Continuous-in, Discrete-out 混合架构**: 输入端连续 (Whisper encoder, 不量化), 输出端离散 (WavLM VQ tokens), 在 LALM 中较少见, 兼顾理解精度和生成质量
3. **6.25 Hz 极低帧率**: 3 级 conv downsampling 将 50 Hz Whisper 输出压缩到 6.25 Hz, 大幅减少 LLM 计算量
4. **Tri-modal Interleaving 预训练**: Text, speech token, continuous speech 三模态交错预训练, 而非简单拼接
5. **Full-duplex 领先指标**: Turn-taking 99.7% (超 Freeze-Omni 99.1%), Pause handling 97.6% (超 Moshi ~51%), 是公开数据中最强

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| Tencent/Covo-Audio | 162 | 端到端 LALM, 开源推理代码 + Chat 权重 (CC BY 4.0) |
| tencent/Covo-Audio-Chat | HF 模型 | 对话版模型权重 (HuggingFace) |
| Covo-Audio-Chat-FD | 未开源 | 全双工版未公开 |
| Hunyuan 语音相关 | 无 | 语音相关模型未独立开源 |

### 判断

- **优势**: Covo-Audio 是国内大厂中少有的独立端到端 LALM, 技术深度透明; Intelligence-Speaker Decoupling 是实用创新; 全双工指标领先; Dong Yu 团队在语音领域有深厚积累; 开源 (CC BY 4.0) 态度积极
- **短板**: Covo-Audio 开源影响力有限 (162 stars, 远低于 CosyVoice 21.5k); 缺少独立的 TTS/ASR 开源项目; 全双工版未开源; Hunyuan 语音能力与 Covo-Audio 的关系不明确 (是否合并?); 论文产出密度低于阿里 FunAudioLLM
- **下一步推测**: Covo-Audio 可能合并进 Hunyuan 体系; 可能开源全双工版; 可能发布独立的 speech codec; UAT 统一音频扩散模型可能扩展到 speech
- **关键观察**: 腾讯走的是"AI Lab 做研究突破 + Hunyuan 做产品集成"的双轨路线。Covo-Audio 的技术深度 (intelligence-speaker decoupling, tri-modal interleaving) 在国内大厂中属上乘, 但开源社区影响力和论文密度需要提升

---

## 团队4: 小米 / MiSpeech (Xiaomi LLM-Core Team)

### 基本信息

| 项目 | 详情 |
|------|------|
| **公司/机构** | 小米 (Xiaomi) — LLM-Core Team + MiSpeech |
| **团队名** | Xiaomi LLM-Core Team (MiMo-Audio); Xiaomi MiSpeech (TTS/ASR, 与 Daniel Povey 合作) |
| **GitHub org** | [XiaomiMiMo](https://github.com/XiaomiMiMo) (MiMo 全系列开源) |
| **HuggingFace** | [XiaomiMiMo](https://huggingface.co/XiaomiMiMo) (MiMo-Audio 等模型) |
| **核心人物** | **Dong Zhang** — MiMo-Audio 一作; **Daniel Povey** — Xiaomi 语音首席科学家 (Kaldi 创始人), ZipVoice/OmniVoice lead; **Han Zhu (朱含)** — ZipVoice/OmniVoice 一作; **Gang Wang** — MiMo-Audio 核心; **Jinlong Xue** — MiMo-Audio 核心 |
| **产品线** | 小爱同学 (语音助手), MiMo 大模型系列, MiMo-V2-TTS (API), 小米手机/IoT 语音交互 |
| **开源态度** | **高度开源** — MiMo (推理模型), MiMo-Audio (音频 LLM), MiMo-V2.5-ASR, MiMo-Audio-Tokenizer, MiMo-VL 等全系列开源 |

### 论文时间线 (2024-2026)

| 时间 | 论文 | 会议/期刊 | 方向 |
|------|------|----------|------|
| 2025.02 | **Llasa: Scaling Train-Time and Inference-Time Compute for LLM-based TTS** (2502.04128) — 小米 MiSpeech + 西工大 | ICML 2025 | LLM-based TTS |
| 2025.06 | **ZipVoice: Fast and High-Quality Zero-Shot TTS** (2506.13053) — Xiaomi + Daniel Povey | TASLP, 30 citations | 快速 TTS |
| 2025.07 | **ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation** (2507.09318) — Xiaomi + Daniel Povey | preprint, 9 citations | **NAR 口语对话** |
| 2025.10 | **MiMo-Audio: Audio Language Models are Few-Shot Learners** (2512.23808) — Xiaomi LLM-Core | preprint, 23 citations | **音频 LLM 旗舰** |
| 2025.12 | **MiMo-VL-Miloco Technical Report** (2512.17436) — Xiaomi | preprint, 9 citations | 多模态 VL |
| 2026.03 | **Xiaomi MiMo-V2-TTS** — Xiaomi MiMo Team | 产品发布 | 产品级 TTS |
| 2026.04 | **OmniVoice: Omnilingual Zero-Shot TTS with Diffusion LM** (2604.00688) — Xiaomi + Daniel Povey | preprint, 6 citations | 全语言 NAR TTS |
| 2026.05 | **Llasa+: Free Lunch for Accelerated Streaming Llama-Based TTS** (2508.06262) — 小米 MiSpeech + 西工大 | preprint | 流式加速 TTS |
| 2026.05 | **MiMo-V2.5-ASR** — Xiaomi MiMo Team | 开源发布 | 多语言 ASR |
| 2026.06 | **Interspeech 2026 Audio Encoder Capability Challenge** (2603.22728) — Xiaomi Inc. 参与 | Interspeech 2026 | 音频编码器评测 |

> **说明**: 小米在 Speech LLM 方向有两条明确的技术线: (1) MiMo-Audio (LLM-Core Team, 音频 LLM 大模型); (2) ZipVoice/OmniVoice (MiSpeech + Daniel Povey, NAR TTS + 对话)。两条线互相支撑 (MiMo-Audio 使用 MiMo-TTS 合成训练数据)。

### 技术栈全景

| 维度 | 技术 |
|------|------|
| **语音编码器** | MiMo-Audio-Tokenizer (自研统一 tokenizer, 兼具 semantic 提取和高保真重建); Whisper (ASR 部分) |
| **LLM 骨干** | MiMo-7B-Base (自研推理模型, scaling next-token prediction 预训练); Qwen3-0.6B (OmniVoice LLM 初始化) |
| **语音解码器** | MiMo-Audio: 多层 codebook → vocoder; ZipVoice: Flow Matching 4 步生成; OmniVoice: 离散扩散 masked generation |
| **对话策略** | ZipVoice-Dialog: NAR spoken dialogue generation (第一个 NAR 口语对话模型); MiMo-Audio: few-shot in-context learning |
| **训练数据规模** | MiMo-Audio: 1 亿小时+ 音频预训练 (百万小时级); MiMo-V2-TTS: 大规模 TTS 训练 |
| **推理延迟** | ZipVoice: RTF 0.014 (极快); OmniVoice: RTF 0.032; MiMo-Audio: 7B 标准推理速度 |
| **多语言** | OmniVoice: 全语言 (omnilingual) 零样本 TTS; MiMo-V2.5-ASR: 多语言/方言; ZipVoice: 英中 |
| **情感/副语言** | MiMo-Audio: 语音情感/风格识别 + few-shot 风格迁移; 支持 voice conversion, style transfer, speech editing |

### 架构演进

```
Daniel Povey 加入小米 (2022) — Kaldi 创始人
    ↓ TTS 基础
Llasa (2025.02, ICML) — 单层 VQ codec + LLaMA TTS, scaling law 验证
    ↓ 加速
ZipVoice (2025.06, TASLP) — Flow Matching 4步快速 TTS (RTF 0.014)
    ↓ 对话扩展
ZipVoice-Dialog (2025.07) — 首个 NAR 口语对话生成模型
    ↓ 音频 LLM
MiMo-Audio (2025.10) — 7B 音频语言模型, 1亿小时预训练, few-shot 能力涌现
    ↓ 全语言 TTS
OmniVoice (2026.04) — 全语言 NAR TTS, 离散扩散 + LLM 初始化
    ↓ 加速 + 流式
Llasa+ (2026.05) — MTP 加速 + 流式 codec (XCodec2-S)
    ↓ ASR
MiMo-V2.5-ASR (2026.05) — 多语言多方言 ASR
    ↓ 产品
MiMo-V2-TTS (2026.03, 产品) — 基于 MiMo-Audio 多层 codebook 的产品级 TTS
小爱同学 — 集成 MiMo 系列能力
```

### 独特技术赌注

1. **Few-shot Audio LLM**: MiMo-Audio 通过 1 亿小时预训练实现 few-shot 涌现 — 无需 task-specific fine-tuning, 仅靠 in-context examples 就能完成新的音频任务, 类比 GPT-3 的文本 few-shot
2. **NAR 口语对话 (ZipVoice-Dialog)**: 全行业首个非自回归口语对话生成模型, 4 步 flow matching 生成对话级语音, 延迟极低
3. **Daniel Povey 效应**: Kaldi 创始人带来的语音工程底蕴 + 学术声誉, 使小米在语音领域的学术影响力远超其公司体量
4. **MiMo-Audio-Tokenizer**: 统一的音频 tokenizer, 同时支持 semantic 信息提取和高保真重建, 是 MiMo-Audio 和 MiMo-V2-TTS 的共享基础设施
5. **全栈自研 + 全面开源**: 从 tokenizer → LLM backbone → Audio LLM → TTS → ASR 全部自研并开源, 在国内大厂中开源程度仅次于阿里 FunAudioLLM

### 开源情况

| 项目 | Stars | 说明 |
|------|-------|------|
| XiaomiMiMo/MiMo-Audio | 1.0k | 7B 音频语言模型, 开源 (Apache-2.0) |
| XiaomiMiMo/MiMo | 2.2k | 推理大模型 (MiMo-Audio 的 backbone) |
| XiaomiMiMo/MiMo-V2.5-ASR | 262 | 多语言 ASR |
| XiaomiMiMo/MiMo-Audio-Tokenizer | 开源 | 统一音频 tokenizer |
| XiaomiMiMo/MiMo-Audio-Training | 开源 | 训练代码 |
| XiaomiMiMo/MiMo-Audio-Eval | 开源 | 评估代码 |
| XiaomiMiMo/MiMo-VL | 643 | 多模态 VL (非语音) |
| Llasa (ICML 2025) | 外部 | 与西工大联合, 独立开源 |
| ZipVoice / OmniVoice | 未独立开源 | 论文发表, 代码未公开 (可能后续开源) |

### 判断

- **优势**: 论文质量高且方向全面 (Audio LLM + TTS + ASR + Spoken Dialogue); Daniel Povey 带来的学术深度; MiMo-Audio few-shot 能力在开源模型中 SOTA; 开源生态完整 (tokenizer + model + training + eval); ZipVoice-Dialog 是行业首创 (NAR spoken dialogue); 小爱同学提供真实产品反馈
- **短板**: 缺少全双工对话能力的论文 (ZipVoice-Dialog 是 NAR 生成, 非实时全双工); MiMo-Audio 开源影响力 (1k stars) 仍低于头部项目; 端到端 voice interaction 产品 (小爱同学) 的技术细节未公开; Daniel Povey 的研究偏 TTS/codec, 全双工方向需要补充
- **下一步推测**: MiMo-Audio 可能扩展到全双工对话; ZipVoice/OmniVoice 可能独立开源; MiMo-V2-TTS 可能发表技术报告; 可能推出 MiMo-Omni (统一多模态); 小爱同学可能集成端到端语音交互
- **关键观察**: 小米在 Speech LLM 方向的表现远超其公司体量, 主要得益于 Daniel Povey 的加盟和 LLM-Core Team 的投入。两条技术线 (MiMo-Audio + MiSpeech/ZipVoice) 的互补性强: 前者做理解+推理, 后者做高效生成。在 4 家中, 小米的技术深度和开源程度仅次于腾讯 (Covo-Audio), 但论文产出密度和多样性更高

---

## 横向对比矩阵

| 维度 | 百度 (ERNIE) | 讯飞 (Spark) | 腾讯 (Covo-Audio) | 小米 (MiMo-Audio) |
|------|-------------|-------------|-------------------|-------------------|
| **Speech LLM 论文数** | 5+ (2026) | 2-3 (基础研究) | 5+ (2025-2026) | 8+ (2025-2026) |
| **端到端 Speech LLM** | ERNIE 5.0 (Omni 子能力) | 无公开论文 | Covo-Audio (独立项目) | MiMo-Audio (独立项目) |
| **全双工能力** | 未公开 | 产品端有, 论文无 | Covo-Audio-FD (99.7% turn-taking) | ZipVoice-Dialog (NAR, 非实时全双工) |
| **独立 TTS 研究** | 无 | 无公开论文 | 无独立 TTS | Llasa, ZipVoice, OmniVoice (3+ 篇) |
| **独立 ASR 研究** | 无 (PaddleSpeech 早期) | 流式 ASR (1 篇) | 无独立 ASR | MiMo-V2.5-ASR |
| **Speech Codec** | 无独立研究 | 无 | WavLM-based VQ tokenizer | MiMo-Audio-Tokenizer |
| **LLM 骨干** | 自研 ERNIE 5.0 (万亿 MoE) | 自研 Spark | Qwen2.5-7B | 自研 MiMo-7B |
| **训练数据规模** | 未公开 | 未公开 | 2T tokens | 1 亿小时+ |
| **开源程度** | 低 (PaddleSpeech 早期) | 极低 | 中 (Chat 版开源) | 高 (全系列开源) |
| **GitHub Stars** | PaddleSpeech 10.8k | ~1k (Agent 工具) | 162 (Covo-Audio) | 1k (MiMo-Audio) + 2.2k (MiMo) |
| **核心人物** | 王海峰/吴华 (NLP 背景) | 未公开 (产品导向) | 俞栋 (语音权威) | Daniel Povey (Kaldi 创始人) |
| **架构路线** | 统一 Omni (all-in-one) | Cascaded (推测) | 端到端 LALM | Audio LLM + NAR TTS |
| **产品落地** | 文心一言/百度地图 | 星火/讯飞输入法/翻译机 | 微信/QQ (推测) | 小爱同学 |
| **技术透明度** | 中 (tech report 有) | 低 (产品封装) | 高 (论文详细) | 高 (论文+开源代码) |

## 关键发现

1. **Speech LLM 投入与传统语音市场地位不成正比**: 讯飞作为中国语音技术龙头, 在 Speech LLM 研究方向的公开产出几乎为零, 与其产品市场地位形成强烈反差。小米凭借 Daniel Povey 加盟, 在语音研究深度上已超越讯飞。

2. **两种路线分化明显**: 百度走"统一 Omni"路线 (ERNIE 5.0 将语音作为子能力), 腾讯和小米走"独立 Speech LLM"路线 (Covo-Audio / MiMo-Audio 作为专门项目)。后者在技术深度和透明度上更优, 但前者在多模态协同上可能有长期优势。

3. **全双工是关键分水岭**: 只有腾讯 (Covo-Audio-FD) 有明确的全双工论文和指标; 小米的 ZipVoice-Dialog 是 NAR 口语对话但非全双工; 百度和讯飞在全双工方向几乎空白 (至少公开层面)。

4. **开源策略差异巨大**: 小米最开源 (tokenizer + model + training + eval 全部公开), 腾讯中等 (Chat 模型开源), 百度和讯飞几乎不开源核心 Speech LLM 能力。开源程度直接影响社区影响力和迭代速度。

5. **Daniel Povey 效应显著**: 小米在语音方向的论文质量和产出密度, 在国内大厂中仅次于阿里 FunAudioLLM, 远超百度和讯飞。这主要归功于 Daniel Povey (Kaldi 创始人) 的加盟, 证明了顶级语音研究者对公司技术路线的影响力。Povey 团队的 ZipVoice (TASLP, 30 citations) 和 OmniVoice 在 NAR TTS 方向已建立学术声誉。
