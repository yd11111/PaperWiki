---
type: paper
tier: deep
title: "VoiceSculptor: Your Voice, Designed By You"
arxiv_id: "2601.10629"
source: "Sources/VoiceSculptor.pdf"
authors: [Jingbin Hu, Huakang Chen, Linhan Ma, Dake Guo, Qirui Zhan, Wenhao Li, Haoyu Zhang, Kangxiang Xia, Ziyu Zhang, Wenjie Tian, Chengyou Wang, Jinrui Liang, Shuhan Guo, Zihang Yang, Bengu Wu, Binbin Zhang, Pengcheng Zhu, Pengyuan Xie, Chuan Xie, Qiang Zhang, Jie Liu, Lei Xie]
year: 2026
venue: "arXiv"
tags: [TTS, instruction-following, voice-design, voice-cloning, controllable, CoT, RAG, open-source, fine-grained-control]
concepts: ["[[Instruction-Guided Speech Synthesis]]", "[[Natural Language Description for TTS]]", "[[LLM-based TTS]]", "[[Speaker Embedding]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]"]
tasks: ["[[Instructed Speech Generation]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页: CosyVoice 2, Instructed Speech Generation, LLM-based TTS, Speaker Embedding, Instruction-Guided Speech Synthesis [待确认], Natural Language Description for TTS [待确认])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: VoiceSculptor 处于 **instruction-guided speech synthesis** 的演进线上,介于 NL description 范式 (PromptTTS, Parler-TTS) 和统一指令范式 (VoxInstruct, InstructAudio) 之间。与 VoxInstruct 将 content + style 压缩到单一 token 序列不同,VoiceSculptor 通过 CoT 显式分解属性推理,实现更精细的解耦控制。

**已有认知**:
- **Instructed Speech Generation** 当前开放问题指出: "音色 (timbre) 尚不可通过文本指令控制,需要额外研究"。VoiceSculptor 正是试图解决这个问题 — 通过 voice design 模块生成 prompt waveform,再用 voice clone 模块实现 timbre transfer。
- **CosyVoice 2** 作为 VoiceSculptor 的 voice clone 后端,提供流式零样本 TTS 能力。VoiceSculptor 将其定位为"可插拔的下游合成器",voice design 与 voice clone 解耦。
- **LLM-based TTS** 领域,LLaSA (基于 LLaMA + XCodec2) 是 VoiceSculptor voice design 模块的基座模型,代表"将 TTS 重构为 sequence-to-sequence generation"的路线。
- **NL Description for TTS [待确认]** 的核心技术挑战包括描述的模糊性和不完整性。VoiceSculptor 通过 CoT 和 RAG 两个机制应对: CoT 消解属性间的纠缠,RAG 补偿分布外指令的理解不足。

**创新判断**: VoiceSculptor 的独特定位是 **voice design + voice clone 的解耦架构** — 先用 LLM 从 NL 指令"设计"一个声音 (生成 prompt waveform),再将这个 prompt 喂给零样本 TTS 模型克隆。这避免了在单一模型中同时承担指令理解 + 高保真合成的复杂度,但引入了中间 prompt 的信息瓶颈。

> 检索命中: [[模型库/CosyVoice 2|CosyVoice 2]]✓, [[Instructed Speech Generation]]✓, [[LLM-based TTS]]✓, [[Speaker Embedding]]✓ | 过滤: [[Instruction-Guided Speech Synthesis]](pending-review), [[Natural Language Description for TTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 开源 instruction-following voice design 系统,通过 CoT 属性推理 + RAG 指令增强 + voice design/clone 解耦架构,实现自然语言驱动的细粒度语音属性控制,开源 SOTA on InstructTTSEval-Zh
> - **路线**: NL instruction → (RAG 检索相似指令) → LLaSA-3B (CoT 属性推理 → speech attribute tokens → XCodec2 audio tokens) → prompt waveform → CosyVoice2-0.5B (voice clone: Flow Matching + HiFi-GAN) → target speech
> - **指标**: InstructTTSEval-Zh: APS 75.7%, DSD 64.7%, RP 61.5%, AVG 67.6% (开源 SOTA); IMOS 3.67; vs MiMo-Audio 7B AVG 64.5%, vs VoxInstruct AVG 47.5%; 低于 Gemini 2.5 Pro AVG 84.8% [Table 1]
> - **可借鉴**: (1) CoT 属性分解: 将 NL 指令通过 chain-of-thought 显式映射到结构化属性 token,可迁移到任何需要精细控制的条件生成任务; (2) voice design/clone 解耦: 属性控制和高保真合成分到两个专门模块,各自优化; (3) 属性 token dropout (p=0.2) 防止对显式控制信号的过拟合
> - **局限**: 主要评估中文; 与 Gemini 差距仍大 (AVG 67.6 vs 84.8); 重度依赖 RAG (去掉 RAG AVG 下降 8.2%); 老人/儿童声音效果差; 重复合成稳定性不足

## 核心问题

现有开源 TTS 系统在零样本语音克隆上已很强 (CosyVoice2, F5-TTS, LLaSA 等),但**细粒度指令控制能力**仍显著落后于商用系统 (Gemini, GPT-4o)。具体表现为 [§1]:
1. 控制信号是低带宽的、纠缠的连续向量,缺乏属性间的显式解耦
2. 主要依赖参考音频条件化,无法从纯文本指令直接生成目标声音
3. 开源与商用系统之间存在显著的 instruction-following 能力鸿沟

VoiceSculptor 要回答: **如何用开源方案实现接近商用系统的、自然语言驱动的细粒度语音属性控制?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VoiceSculptor 由两个解耦模块组成 [§2.1, Fig 1]:

1. **Voice Design (VD)**: 基于 LLaSA-3B,从 NL 指令生成 prompt waveform
   - LLaSA = LLaMA + XCodec2 (neural audio codec)
   - 将 TTS 重构为 sequence-to-sequence: NL instruction → discrete audio tokens → XCodec2 decode → waveform
2. **Voice Clone (VC)**: CosyVoice2-0.5B-LLM
   - 输入: VD 生成的 prompt waveform + target text
   - 输出: 保持 prompt 风格的目标文本语音
   - Flow Matching + HiFi-GAN

[agent 解读] 这种 design/clone 解耦避免了在单一模型中同时优化"理解 NL 指令" + "高保真合成"的困难。VD 模块专注于属性理解和粗粒度声音生成,VC 模块专注于音质和 timbre fidelity。代价是 prompt waveform 作为中间表示的信息瓶颈 — VC 模块只能"听到"VD 产生的声音,无法直接访问指令语义。

### 关键设计选择

#### 1. CoT-based Fine-grained Attribute Modeling [§2.3]

[论文原文] 直接以显式属性 token 条件化语音合成会导致脆弱的控制和对结构化输入的过度依赖,限制对多样 NL 指令的泛化能力 [§2.3]。

**解决方案**: 将属性信息组织为 CoT 推理步骤,作为 NL 指令和声学实现之间的中间语义表示 [§2.3]:
- 模型在 unified autoregressive 框架中联合建模: instruction text → CoT attribute tokens → discrete speech tokens
- 训练目标: joint cross-entropy loss 同时覆盖 text tokens 和 speech tokens [§2.3]

[agent 解读] CoT 的核心价值不在于"推理",而在于**显式解耦**: 将"pitch=high, speed=fast, emotion=happy"等属性作为中间 token 明确产出,避免它们在 latent space 中纠缠。这使得每个属性都可以被独立控制和诊断。

**属性 token dropout** (p=0.2) [§2.3]: 训练时随机移除部分 attribute tokens,迫使模型从 NL 指令本身推断属性,而非完全依赖显式 token。

[论文原文] 这个 dropout 策略不仅没有降低性能,反而提升了模型的鲁棒性和泛化能力 [§3.3]。

#### 2. Retrieval-Augmented Instruction Generalization [§2.4]

[论文原文] 为提升模型处理分布外 NL 指令的泛化能力和鲁棒性 [§2.4]:

- 构建 500K in-domain 指令的向量数据库,使用 Qwen3-Embedding-0.6B 编码,存储于 Milvus
- 推理时: 输入指令 → embedding → cosine 相似度检索 → 注入最相关的 in-domain 指令作为上下文
- 所有报告结果均启用 RAG [§3.1]

[agent 解读] RAG 的巨大贡献 (AVG +8.2%, RP +13%) 暴露了一个隐忧: 模型本身的指令泛化能力可能相当有限 — 论文也承认"模型在仅依赖内部表示时文本理解和泛化能力相对受限" [§3.5]。这使得 RAG 从"增强"变为"必需"组件。

#### 3. Joint Text-Speech CE Loss [§2.3]

同时在 text tokens 和 speech tokens 上计算 cross-entropy loss [§2.3]:
- text-token CE: 强制模型捕捉语义意图和属性关系
- speech-token CE: 引导准确的声学生成
- 多层监督比单纯 speech token 训练提供更丰富的学习信号

### 训练策略

**数据 pipeline** [§2.2, Fig 2]:
- 来源: in-the-wild + in-house 数据
- 预处理: 去噪、VAD、多说话人检测、MOS 过滤
- 标注三层: 
  - 声学层 (Gemini 2.5 Pro → 多维标注 → DeepSeek 翻译/描述 → regex 去幻觉)
  - 情感层 (Emo2Vec + Qwen3-72B + SenseVoice + Qwen3-Omni 交叉验证)
  - 韵律层 (DataSpeech 提取 + VoxProfile 年龄/性别 + 人工校准)

**训练配置** [§3.2, Table 2-3]:
| 阶段 | 数据 | 规模 | Epochs |
| --- | --- | --- | --- |
| CPT (Continual Pre-training) | CPTData4 (in-the-wild + VoxBox + in-house) | 9000h | 2 |
| SFT (Supervised Fine-Tuning) | SFTData3 (in-the-wild + in-house) | 4000h | 3 |

- 模型: LLaSA-3B (最佳), 也测试了 1B
- 硬件: 8x A100 (3B), 8x L40 (1B)

## 实验

### 主结果 (InstructTTSEval-Zh)

| 模型 | APS↑ (%) | DSD↑ (%) | RP↑ (%) | AVG↑ (%) | 出处 |
| --- | --- | --- | --- | --- | --- |
| Gemini 2.5-Pro* | 89.0 | 90.1 | 75.5 | 84.8 | [Table 1] |
| Gemini 2.5-Flash* | 88.2 | 90.9 | 77.3 | 85.4 | [Table 1] |
| GPT-4o-Mini-TTS* | 54.9 | 52.3 | 46.0 | 51.1 | [Table 1] |
| MiMo-Audio-7B | 70.1 | 66.1 | 57.1 | 64.5 | [Table 1] |
| **VoiceSculptor-VD** | **75.7** | 64.7 | **61.5** | **67.6** | [Table 1] |
| VoiceSculptor-VD & VC | 77.2 | 65.1 | 59.6 | 67.3 | [Table 1] |
| VoxInstruct | 47.5 | 52.3 | 42.6 | 47.5 | [Table 1] |
| ElevenLabs* | 42.8 | 50.9 | 59.1 | 50.9 | [Table 1] |

*标 * 为商用闭源系统。VoiceSculptor 在开源系统中 AVG 最高。

### Scaling Study (Table 3)

| Model Size | Training Data | IMOS↑ | APS↑ | AVG↑ | 出处 |
| --- | --- | --- | --- | --- | --- |
| 1B | SFTData1 (1000h) | 3.09 | 51.3 | 45.1 | [Table 3] |
| 3B | SFTData1 (1000h) | 3.24 | 59.2 | 50.6 | [Table 3] |
| 1B | SFTData2 (3700h) | 3.35 | 61.5 | 54.4 | [Table 3] |
| 3B | SFTData2 (3700h) | 3.58 | 72.4 | 61.8 | [Table 3] |
| 3B | CPT 9000h + SFT 4000h | **3.67** | **75.7** | **67.6** | [Table 3] |

模型从 1B→3B: 各指标一致提升。CPT 提供更好的初始化 [§3.2]。

### Ablation Studies

| 组件 | 有 | 无 | 差值 (AVG) | 出处 |
| --- | --- | --- | --- | --- |
| CoT attribute tokens | 67.6% | 63.5% | +4.1% | [Table 4] |
| Text CE Loss | 67.6% | 61.8% | +5.8% | [Table 5] |
| RAG | 67.6% | 59.4% | +8.2% | [Table 6] |

三个组件均有显著贡献,其中 **RAG 贡献最大** (+8.2%),尤其在 RP (+13.0%) 上。

## 局限性

1. **仅中文主评估**: 训练数据以中文为主,未评估英文或多语言能力 [§4]
2. **与 Gemini 差距显著**: AVG 67.6% vs 84.8% (Gemini 2.5 Pro),仍有 17.2% 的差距 [Table 1]
3. **重度依赖 RAG**: 去掉 RAG 后 AVG 从 67.6% 降至 59.4%,暴露模型自身指令理解能力不足 [Table 6]
4. **合成稳定性**: 相同指令重复合成时无法稳定保持属性控制 [§4]
5. **老人/儿童声音**: 自然度和音色一致性不足,源于训练数据覆盖不足 [§4]
6. **长静默/延迟**: 合成和交互中偶现长静默 [§4]
7. **中间 prompt 信息瓶颈**: VD 生成的 prompt waveform 是 VC 的唯一条件,指令语义无法直接传递给 VC 模块 [agent 解读]
8. **评估局限**: InstructTTSEval-Zh 使用 Gemini-as-Judge,可能存在 self-preference bias [agent 解读,参考 InstructTTSEval 已知问题]

## 点评

**Voice Design / Voice Clone 解耦是本文最有价值的架构洞察。** 将"理解用户想要什么声音"和"高质量合成那个声音"分到两个专门模块,各自可独立优化和替换。VD 产出的 prompt waveform 作为通用接口,理论上可以喂给任何零样本 TTS 系统 (CosyVoice2, F5-TTS 等)。Table 1 中 VoiceSculptor-VD&VC 的结果证明风格从 VD 到 VC 的传递确实可行 (AVG 67.3% vs VD-only 67.6%,几乎无损) [Table 1]。

**CoT 属性推理的实际效果值得关注。** 从 Table 4 看,CoT 带来 AVG +4.1% (63.5→67.6),说明显式的属性分解 reasoning 确实提升了控制精度。但这个提升远小于 RAG (+8.2%),暗示当前瓶颈更多在于模型的 NL 理解能力而非属性建模方式。

**开源生态贡献显著。** 在 VoxInstruct (AVG 47.5%) 和 MiMo-Audio 7B (AVG 64.5%) 之上建立了新的开源 SOTA。完整开源代码和模型,为 instruction-following TTS 研究提供了可复现的 baseline。

**但 RAG 依赖暴露了根本性弱点。** 论文坦承模型"文本理解和泛化能力相对受限" [§3.5],RAG 从增强手段实质上变成了必需组件。这意味着在 RAG 数据库覆盖不到的指令空间,系统可能会显著退化。未来需要增强模型自身的指令理解能力而非外部拐杖。

**数据 pipeline 的工程价值高。** 多模型交叉验证 (Emo2Vec + Qwen3-72B + SenseVoice + Qwen3-Omni for emotion) + 人工校准的 annotation 流程 [§2.2] 是构建大规模 instruction-speech paired 数据的完整参考。

## 可复用的 idea

1. **CoT-based attribute decomposition for conditional generation**: 在条件生成任务中,用 chain-of-thought 将高层条件 (NL 指令) 显式分解为结构化中间属性,再从属性生成目标。可迁移到 image generation from description、music generation from text 等场景
2. **Attribute token dropout as regularizer**: 训练时以 p=0.2 随机 drop 显式控制 token,迫使模型从 NL 指令本身推断意图,防止对显式信号的过拟合。通用于任何使用辅助 conditioning token 的系统
3. **Design/Clone decoupled architecture**: 将"理解指令设计声音"和"高保真合成"分到两个模块,用中间 prompt waveform 作为通用接口。VC 模块可热插拔替换,VD 产出的 prompt 也可被缓存复用
4. **Multi-model cross-validation for emotion annotation**: 用 4 个互补模型独立标注情感后交叉验证,显著提升标注可靠性,适用于任何主观属性的大规模标注场景
