---
type: paper
tier: deep
title: "InstructAudio: Unified Speech and Music Generation with Natural Language Instruction"
arxiv_id: "2511.18487"
source: "Sources/InstructAudio.pdf"
authors: [Chunyu Qiang, Kang Yin, Xiaopeng Wang, Yuzhe Liang, Jiahui Zhao, Ruibo Fu, Tianrui Wang, Cheng Gong, Chen Zhang, Longbiao Wang, Jianwu Dang]
year: 2025
venue: "ICASSP 2026"
tags: [TTS, TTM, unified-audio, instruction-control, MM-DiT, flow-matching, natural-language-description, music-generation, dialogue-TTS, multi-attribute-control]
concepts: ["[[Instruction-Guided Speech Synthesis]]", "[[Natural Language Description for TTS]]", "[[Conditional Flow Matching]]", "[[Diffusion-based TTS]]", "[[Singing Voice Synthesis]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]", "[[模型库/CosyVoice|CosyVoice]]"]
tasks: ["[[任务库/Instructed Speech Generation]]"]
datasets: ["[[数据集/SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Instruction-Guided Speech Synthesis]][待确认], [[Natural Language Description for TTS]][待确认], [[Conditional Flow Matching]]✓, [[Singing Voice Synthesis]][待确认], [[Diffusion-based TTS]][待确认], [[模型库/CosyVoice 2|CosyVoice 2]]✓ | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: InstructAudio 处于 Instruction-Guided Speech Synthesis 和 Text-to-Music (TTM) 的交叉地带。在 TTS 侧,它延续了 PromptTTS→InstructTTS→VoxInstruct→CosyVoice 的 NL description 控制路线,但进一步实现了纯文本控制(无需参考音频)的 timbre 属性(gender, age)——这是 CosyVoice 2 仍依赖参考音频的短板。在 TTM 侧,它将 NL description 控制范式从 TTS 拓展到音乐生成,与 ACE-Step/DiffRhythm+ 竞争。核心创新在于用统一的 instruction-phoneme 输入格式和 MM-DiT 架构同时覆盖两个任务。

**已有认知对比**:
- CosyVoice 2 (confirmed): LLM + chunk-aware CFM 的 hybrid 架构,支持 emotion/style/accent 文本控制,但 timbre 控制仍需参考音频。InstructAudio 号称用纯文本替代了参考音频进行 timbre 控制,代价是 MOS 略低(因 one-to-many mapping 模糊性)。
- Instruction-Guided Speech Synthesis (pending-review): VoxInstruct 开创了统一指令格式,CosyVoice 实现了 LLM+Flow 的 hybrid 指令控制。InstructAudio 的独特贡献是将这一范式扩展到 TTM。
- Conditional Flow Matching (confirmed): InstructAudio 使用 CFM 训练 MM-DiT,与 CosyVoice 系列/F5-TTS 等共享这一生成范式,但架构上采用 Stable Diffusion 3 风格的 joint/single DiT 分层,而非传统 U-Net。
- Singing Voice Synthesis (pending-review): SVS 通常需要乐谱输入,InstructAudio 的 TTM 路线不需要显式乐谱(歌词+NL description 即可),与传统 SVS 的输入范式不同。

## 速查

> [!summary] 速查
> - **一句话**: 首个用统一 NL instruction + phoneme 输入格式同时控制 TTS 和 TTM 的框架,基于 MM-DiT 架构实现 timbre/paralinguistic/musical 多属性纯文本控制
> - **路线**: NL instruction + text/lyrics → Qwen2.5-7B (frozen) instruct encoder + Zipformer phoneme encoder → Joint DiT (14L) + Single DiT (6L) → CFM ODE solver → Mel-VAE decoder → speech/music
> - **指标**: Seed-TTS WER EN 1.52% / ZH 1.35% (best); Gender Acc 100%, Emotion Acc 83.33%, Dialogue Acc 90%; SongEval Coherence 3.08 / Musicality 2.98 (best) [Table 1, 2, 3]
> - **可借鉴**: (1) 统一 instruction-phoneme 格式消除 TTS/TTM 输入异构性; (2) Frozen 大 LLM (Qwen2.5-7B) 做 instruct encoder 的高效利用; (3) Joint DiT + Single DiT 分层设计平衡跨模态对齐和单模态细节
> - **局限**: NMOS 3.46 低于 CosyVoice2 (3.65),纯文本控制的 one-to-many 模糊性导致音质下降; 音乐限制在 5-20s 短片段; 无开源代码

## 核心问题

这篇论文要解决的核心问题是: **TTS 和 TTM 作为两个共享声学建模基础的任务,长期被独立开发,输入控制条件高度异构(TTS 依赖参考音频做 timbre 控制,TTM 依赖专业标注做音乐属性控制),如何用统一的自然语言指令同时控制两者?**

具体痛点 [§1]:
1. 现有 TTS 模型(如 CosyVoice 2)可以文本控制 emotion/style,但 timbre(gender, age)仍需参考音频
2. 现有 TTM 模型(如 DiffRhythm+)缺少歌手 timbre 控制; ACE-Step 虽全面但不支持语音
3. 已有统一尝试(Vevo2, UniAudio, AudioBox)要么依赖参考音频,要么需要不同任务的不同输入格式

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

InstructAudio 采用 MM-DiT (Multimodal Diffusion Transformer) 架构,灵感来自 MM-Audio 和 Stable Diffusion 3 [§2.1, Fig 2]:

```
文本模态: NL instruction → Qwen2.5-7B (frozen) → instruct embedding ∈ R^{B×L1×D}
         text/lyrics → G2P → Zipformer phoneme encoder → phoneme embedding ∈ R^{B×L2×D}
         → temporal concatenation → C_text ∈ R^{B×(L1+L2)×D}

音频模态: target mel → Mel-VAE encoder → VAE latent
         → linear interpolation with Gaussian noise → x_t (flow matching path)

Joint DiT (14 layers): C_text 和 x_t 通过 joint attention 交互
  → QKV from both modalities concatenated → scaled dot-product attention → split back

Single DiT (6 layers): 仅处理 audio latent (self-attention only)

训练: CFM objective: ||v_θ(t, C_text, x_t) - u(t, x_t)||²
推理: ODE solver 从 Gaussian noise 生成 VAE latent → Mel decoder → waveform
```

**关键设计**: Joint DiT 层处理跨模态对齐,Single DiT 层专注音频细节增强。[论文原文] 认为这种分层设计能增强语音和歌声质量 [§2.4]。[agent 解读] 这与 Stable Diffusion 3 的 MM-DiT 设计一致——文本-图像 joint attention 后加 image-only refinement,InstructAudio 将其迁移到文本-音频场景。

### 关键设计选择

**1. 统一 Instruction-Phoneme 输入格式 [§2.2]**

[论文原文] 为什么选择统一格式而非任务特定格式: TTS 和 TTM 的输入异构性(TTS 需要 timbre 参考音频 + 文本,TTM 需要专业音乐标注 + 歌词)是联合建模的主要障碍。通过将所有控制条件统一为 NL instruction + phoneme 两个通道,消除了这一障碍。

- TTS: instruction 描述 speaker 的 gender/age/emotion/style/accent; text 转为 phoneme
- TTM: instruction 描述 singer timbre/genre/instrument/rhythm/atmosphere; lyrics 转为 phoneme
- 对话: instruction 分别描述两个说话人; text 用 [S0]/[S1] 标签区分

**2. Frozen Qwen2.5-7B 做 Instruct Encoder [§3.1]**

[agent 解读] 选择 frozen 大 LLM 而非可训练小模型做 instruct encoder,可能出于两个考虑: (1) 7B LLM 对自然语言指令的理解能力远超小模型,特别是对复杂多属性描述; (2) frozen 避免了大 LLM 微调的计算开销,同时利用其语义表示。这与 AudioBox 等使用 T5 或 CLAP 做条件编码的路线不同。

**3. 不需要 text-upsampling alignment [§2.1]**

[论文原文] 与 F5-TTS、ZipVoice 等 NAR 架构不同,InstructAudio 不需要文本到音频的显式上采样对齐。[agent 解读] 这可能得益于 MM-DiT 的 joint attention 机制——文本和音频在注意力层中自由交互,模型隐式学习对齐,类似于 E2 TTS 的做法。

**4. Latent Audio Codec (SecoustiCodec VAE) [§2.3]**

Mel-VAE 将 44.1kHz 音频编码为 43Hz 的连续 latent (1024x 压缩),使 MM-DiT 在压缩空间中训练,兼顾效率和重建质量。[论文原文] 这源于他们先前的 SecoustiCodec 框架。

### 训练策略

- 训练数据: 50K 小时语音 + 20K 小时音乐,互联网来源 [§3.1]
- 数据分布: 中英比 1:1,男女比 1:1,90%+ 中性情感,0.5% 对话数据 [§3.1]
- 音频长度: 2-20s,统一 44.1kHz [§3.1]
- 硬件: 32 × A800 80GB,batch size 16/GPU [§3.1]
- 优化器: Adam, lr 1e-4 [§3.1]
- 参数量: 1.34B (14 Joint DiT + 6 Single DiT, flow matching FFN dim 1024, RoPE) [§3.1]
- 冻结模块: instruct encoder (Qwen2.5-7B), mel encoder, mel decoder [§2.1]

## 实验

| 指标 | 本文 | CosyVoice2 | ZipVoice | F5-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER EN (%) ↓ | **1.52** | 2.57 | 1.70 | 1.89 | Seed-TTS | [Table 1] |
| WER ZH (%) ↓ | **1.35** | 1.45 | 1.40 | 1.53 | Seed-TTS | [Table 1] |
| Gender Acc (%) ↑ | **100.0** | -- | -- | -- | Internal 500 | [Table 2] |
| Age Acc (%) ↑ | 86.67 | -- | -- | -- | Internal 500 | [Table 2] |
| Emotion Acc (%) ↑ | **83.33** | 58.33 | -- | -- | Internal 500 | [Table 2] |
| Style Acc (%) ↑ | **86.67** | 65.00 | -- | -- | Internal 500 | [Table 2] |
| Dialogue Acc (%) ↑ | **90.00** | -- | -- | -- | Internal 500 | [Table 2] |
| Speaker SIM ↑ | 0.76 | 0.68 | -- | -- | Internal 500 | [Table 2] |
| Emotion SIM ↑ | 0.71 | 0.53 | -- | -- | Internal 500 | [Table 2] |
| LSD ↓ | **1.88** | 2.57 | -- | -- | Internal 500 | [Table 2] |
| MCD ↓ | **5.71** | 7.11 | -- | -- | Internal 500 | [Table 2] |
| QMOS ↑ | 3.73 | **3.90** | -- | -- | Internal 100 | [Table 2] |
| NMOS ↑ | 3.46 | **3.65** | -- | -- | Internal 100 | [Table 2] |

**TTM 对比**:

| 指标 | 本文 | ACE-Step | DiffRhythm+ | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Genre Acc (%) ↑ | 92.78 | **94.44** | 51.33 | Internal 500 | [Table 3] |
| Gender Acc (%) ↑ | **98.89** | 96.11 | 22.22 | Internal 500 | [Table 3] |
| SongEval Coherence ↑ | **3.08** | 2.89 | 2.68 | Internal 500 | [Table 3] |
| SongEval Musicality ↑ | **2.98** | 2.87 | 2.61 | Internal 500 | [Table 3] |
| QMOS ↑ | 2.82 | **3.30** | 3.04 | Internal 100 | [Table 3] |
| MMOS ↑ | **2.91** | 2.88 | 2.79 | Internal 100 | [Table 3] |

**关键发现**:

1. **WER 最优**: InstructAudio 在 Seed-TTS benchmark 上 WER 最低 (EN 1.52%, ZH 1.35%),尽管使用纯文本控制 + 随机说话人,数据量 (50K+20K) 也少于多数 baseline (100K+) [Table 1]
2. **控制能力全面覆盖**: 唯一同时支持 Gender/Age/Emotion/Style/Accent/Dialogue 文本控制的模型 [Table 1, 2]
3. **MOS 劣势源于模态缺失**: QMOS 3.73 vs CosyVoice2 3.90, NMOS 3.46 vs 3.65。[论文原文] 认为这是纯文本控制引入的 one-to-many mapping 模糊性导致的,CosyVoice2 有参考音频辅助,不公平对比 [§3.3]
4. **SongEval 全面最优**: 5 个 SongEval 子维度均超越 ACE-Step (3B) 和 DiffRhythm+ (1B),但 QMOS 低于 ACE-Step [Table 3]
5. **评估公平性注意**: 音乐评估使用 5-20s 短片段,对 ACE-Step/DiffRhythm+ 等长音乐模型不利 [§3.4]

## 局限性

1. **MOS 劣势**: 纯文本控制的 one-to-many mapping 导致平均音质和自然度低于使用参考音频的系统。这是该范式的固有代价 [§3.4]
2. **音乐长度受限**: 联合建模要求音乐片段与语音对齐 (5-20s),无法生成长音乐 [§3.4]
3. **评估公平性**: TTM 对比中对长音乐模型 (DiffRhythm+) 不利; CosyVoice2 对比中 CosyVoice2 使用了额外参考音频 [§3.3, 3.4]
4. **无开源**: 仅提供 demo 页面,无代码/模型开源
5. **对话数据极少**: 训练数据中仅 0.5% 为对话数据,90% 达到的对话控制准确率可能在复杂场景下下降 [agent 解读]
6. **Instruct encoder 开销**: Qwen2.5-7B 虽然 frozen,但推理时仍需运行 7B 模型做编码,实际部署成本高 [agent 解读]
7. **消融缺失**: 未提供 Joint DiT vs Single DiT 层数/比例的消融,也无 instruct encoder 选择 (7B vs 小模型) 的对比 [agent 解读]

## 点评

**优势**:
- 抓住了一个真实痛点: TTS 和 TTM 的输入异构性确实阻碍了统一建模。统一 instruction-phoneme 格式是一个简洁有效的解决方案
- 实验设计较全面: 13 个评估维度覆盖 WER/控制准确率/相似度/失真/主观评估
- 数据效率: 以较小数据量 (70K 总计) 超越 100K+ 训练的多数 TTS baseline

**不足**:
- 对比不够公平: CosyVoice2 使用参考音频的 MOS 对比对 InstructAudio 不利; DiffRhythm+ 被截断到短片段也有偏差
- 缺少消融: MM-DiT 的 Joint/Single 层分配、instruct encoder 选择、数据配比等关键设计决策均无消融
- "统一"的程度有限: 语音和音乐共享 DiT 权重,但实际训练数据是分开标注的 (语音用 TTS 属性,音乐用 TTM 属性),跨模态迁移效果未验证
- 音质差距: 纯文本控制的 MOS 劣势是根本性的,论文承认但未提出解决方案

**定位**: 这是 Instruction-Guided Speech Synthesis 路线向 TTM 的首次扩展,验证了统一框架的可行性。但在 TTS 质量上仍不如 CosyVoice2 (有参考音频时),在 TTM 音质上不如 ACE-Step。价值更多在于概念验证 (proof-of-concept) 和全面可控性,而非单任务 SOTA。

## 可复用的 idea

1. **统一 instruction-phoneme 输入格式**: 将异构控制条件(参考音频/专业标注)统一为 NL instruction,是解决多任务音频生成输入不一致的通用策略。可迁移到 sound effect 生成、voice conversion 等
2. **Frozen 大 LLM 做条件编码器**: 直接复用 Qwen2.5-7B 的语义理解能力,避免微调开销。可推广到需要复杂条件理解的其他生成任务
3. **Joint DiT + Single DiT 分层**: 跨模态对齐和单模态细化的分离,可迁移到任何多模态生成场景 (如 text-to-video、text-to-3D)
4. **对话标签 [S0]/[S1]**: 简单的特殊 token 方案实现多说话人对话合成控制,设计开销极低

## 审阅

*(待独立审阅 agent 补充)*
