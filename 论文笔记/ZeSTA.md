---
type: paper
tier: deep
title: "ZeSTA: Zero-Shot TTS Augmentation with Domain-Conditioned Training for Data-Efficient Personalized Speech"
arxiv_id: "2603.04219"
source: "Sources/ZeSTA.pdf"
authors: [Youngwon Choi, Jinwoo Oh, Hwayeon Kim, Hyeonyu Kim]
year: 2026
venue: "arXiv (Maum AI / Humelo)"
tags: [TTS, data-augmentation, speaker-adaptation, fine-tuning, zero-shot, low-resource, domain-conditioning, personalization]
concepts: ["[[SpeakerAdaptation]]", "[[SpeakerEmbedding]]", "[[VoiceCloningTaxonomy]]", "[[VariationalAutoencoderforTTS]]"]
models: ["[[VITS]]", "[[CosyVoice2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-05
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[Zero-shotSpeechSynthesis]]✓, [[SpeakerAdaptation]][待确认], [[VITS]][待确认], [[CosyVoice2]]✓, [[SpeakerEmbedding]]✓, [[VoiceCloningTaxonomy]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 
> **谱系定位**: ZeSTA 处于 voice cloning 分类体系中 Speaker Adaptation 和 Zero-shot TTS 两大范式的交叉点。VoiceCloningTaxonomy 将这两者视为独立类别(SA 需要微调,ZS-TTS 不需要),而 ZeSTA 提出用 ZS-TTS 的输出作为 SA 微调的数据源,构建了跨范式的桥梁。
> 
> **已有认知**: SpeakerAdaptation 页面涵盖了丰富的参数效率方法(AdaSpeech CLN、adapter、structured pruning 等)和无转写数据方法(AdaSpeech 2),但**未覆盖合成数据增强作为 adaptation 策略**的路线。Related Works 中的 VC-based augmentation (Huybrechts 2021) 和 TTS-by-TTS (Hwang 2021) 依赖训练目标说话人的 VC/TTS 模型,不是 zero-shot 路线。ZeSTA 填补的正是"用现成 ZS-TTS 生成合成数据辅助微调"这一空白。
> 
> **创新判断**: Domain conditioning 的思路本质上是将 domain 信息作为额外条件变量(类似 SpeakerEmbedding 页面中描述的 speaker embedding 注入方式),但目标不是控制"谁说",而是控制"来自哪个域"。ZeSTA 对 VITS 的修改极为轻量(仅在 speaker embedding matrix 中增加 64-dim domain embedding),在 VAE+Flow+GAN 架构上验证了这一策略的有效性。CosyVoice 2 作为两个 ZS-TTS 源之一,其 SECS 在 LibriTTS 上 0.794,代表了当前 ZS-TTS 的 speaker similarity 水平。
> 
> 检索命中: [[Zero-shotSpeechSynthesis]]✓, [[CosyVoice2]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[SpeakerAdaptation]](pending-review), [[VITS]](pending-review), [[VoiceCloningTaxonomy]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 ZeSTA 框架,通过 domain embedding + real-data oversampling,解决用 ZS-TTS 合成数据增强轻量 TTS 微调时的 speaker similarity 下降问题
> - **路线**: 目标说话人少量录音 → ZS-TTS(Fish-Speech/CosyVoice 2)生成合成语音 → domain embedding 区分 real/synth + real 数据 3x oversampling → VITS 微调 → 推理时设 d=real
> - **指标**: DC+OS 将 SECS 从 0.765(naive mixing, FS)恢复到 0.815(接近 Real 100% 的 0.832),同时保持 CER/WER 改善;ABX 偏好 70.8%(FS, LibriTTS) [Table 3, Table 4]
> - **可借鉴**: domain embedding 思路极简且不改基础架构,可移植到任何使用合成数据微调的 TTS 系统;real-data oversampling 作为 complementary trick 成本为零
> - **局限**: 仅在 VITS 上验证,未扩展到 LLM-based TTS;8+6 个说话人规模较小;未与 LoRA/adapter 等参数效率方法对比;合成数据量固定为 90%,未探索最优比例

## 核心问题

**问题**: 个性化 TTS 微调在目标说话人数据极度有限时表现不佳。用 ZS-TTS 生成的合成语音做数据增强虽然能提升可懂度(CER/WER 改善),但会导致 speaker similarity 下降 [§1]。

**根因分析** [agent 解读]: 合成语音虽然语言内容稳定(来自强 ZS-TTS 系统),但音色与目标说话人存在系统性偏差(Table 2: Fish-Speech SECS 仅 0.763, CosyVoice 2 为 0.794)。当大量(90%)这类偏差数据与少量(10%)真实数据混合微调时,模型的声学生成模块被合成域的音色特征所偏置,产生 speaker identity drift。

**目标**: 在保留合成数据带来的可懂度增益的同时,恢复被 naive mixing 损害的 speaker similarity。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ZeSTA 框架分为三个组件 [§3, Fig 1]:

1. **ZS-TTS 数据增强** [§3.1]: 用公开的 ZS-TTS 模型(Fish-Speech、CosyVoice 2)以目标说话人的录音为 reference prompt,合成额外的语音数据。选择 Real 10% 中最长的 utterance 作为 prompt 以最大化 speaker-style 覆盖 [§4.1]。

2. **Domain-Conditioned Training (DC)** [§3.2]: 引入一个 binary domain label d ∈ {real, synthetic},将 TTS 优化目标从 p(y|x) 变为 p(y|x,d)。关键洞察 [论文原文]: 文本编码器 f_text 提取的语言表示 h_ling 是 speaker-agnostic 的,所以合成数据对文本编码器的训练(即语言增强)不受影响;而 domain label 只在声学生成模块 g(h_ling, d) 中起作用,调制 domain-specific 的声学特征 [§3.2]。

3. **Real-Data Oversampling (OS)** [§3.3]: 将真实数据重复 3 倍,进一步强调真实目标说话人的样本。不改变模型架构或推理流程。

**推理时**: 只需将 domain label 设为 d=real,模型即自动生成更接近真实说话人特征的语音 [Fig 1(b)]。

### 关键设计选择

**1. Domain embedding 实现** [§4.1]: 复用 VITS 原有的 multi-speaker embedding matrix,将 hidden size 从 256 缩减到 64。这是一个极轻量的修改 --- 仅增加一个 64-dim 的可学习向量来区分 real/synth 域。

**为什么用 64-dim?** Table 5 消融显示:16-dim 语言增强不足(CER 退化),256-dim 过度编码域差异导致 SECS 下降,64-dim 是最优平衡 [§4.3, Table 5]。[agent 解读]: 这暗示 domain embedding 容量过大时,模型可能将过多信息编码到 domain 维度(包括本应留给说话人的变异),从而损害 speaker similarity。

**2. 条件概率分解的合理性** [§3.2]: 论文将 domain conditioning 类比为 multi-speaker TTS 中 speaker embedding 的注入方式 [论文原文]。文本编码器保持 speaker/domain-agnostic,speaker/domain 相关信息在声学生成阶段注入。[agent 解读]: 这意味着 domain embedding 和 speaker embedding 在功能上是解耦的 --- domain embedding 编码"合成 vs 真实"的系统性偏差,speaker embedding 编码"谁",两者可独立变化。

**3. OS 与 DC 的互补关系** [§4.2]: OS alone 效果有限且不稳定(Table 3: 部分配置 SECS 几乎无改善);DC alone 有效恢复 SECS 但略损 CER/WER;DC+OS 结合后 SECS 和 CER/WER 均获最佳平衡 [论文原文]。[agent 解读]: DC 先消除合成域偏置,使得 real 样本的信号不被合成域淹没,然后 OS 进一步放大 real 样本的权重,两步的顺序依赖关系解释了为什么 OS alone 不 work。

### 训练策略

- **预训练**: VCTK 多说话人语料,400 epochs, 4x A100, AdamW (lr=2e-4) [§4.1]
- **微调**: lr=1e-5, batch=32, 600 epochs, 单卡 A100 [§4.1]
- **低资源模拟**: 每个说话人仅保留 10% 训练语句作为 Real,剩余 90% 用 ZS-TTS 合成 [§4.1]
- **合成数据过滤**: 扩展合成数据(+Extra Synth, 800 VCTK 文本)时,用 Whisper medium 过滤 WER>5% 的样本以去除 hallucination [§4.2]

## 实验

| 指标 | 本文 (DC+OS, FS) | Naive mixing (FS) | Real 10% | Real 100% | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| SECS ↑ | 0.815 | 0.765 | 0.818 | 0.832 | LibriTTS | [Table 3] |
| CER ↓ | 4.765 | 4.738 | 5.932 | 6.539 | LibriTTS | [Table 3] |
| WER ↓ | 10.563 | 10.348 | 12.520 | 13.645 | LibriTTS | [Table 3] |
| SECS ↑ | 0.799 | 0.764 | 0.813 | 0.840 | YoBind | [Table 3] |
| CER ↓ | 4.174 | 4.081 | 5.063 | 4.743 | YoBind | [Table 3] |
| SECS ↑ (DC+OS, CV2) | 0.815 | 0.789 | 0.818 | 0.832 | LibriTTS | [Table 3] |
| SECS ↑ (DC+OS, CV2) | 0.804 | 0.774 | 0.813 | 0.840 | YoBind | [Table 3] |
| MOS (DC+OS, FS) | 3.92±0.27 | 3.86±0.37 | 3.58±0.42 | 3.67±0.34 | LibriTTS | [Table 4] |
| ABX preference | 70.8% | (baseline) | -- | -- | LibriTTS | [Table 4] |

**关键发现**:

1. **Naive mixing 悖论** [§4.2]: 合成数据增强显著改善 CER/WER(Real 10% 的 5.932→4.738),但 SECS 从 0.818 降至 0.765。Real 100% 的 CER 反而更差(6.539 vs 4.738),论文认为这是因为 TTS 合成语音变异性低于自然语音 [论文原文]。

2. **DC 恢复 speaker similarity** [§4.2]: DC 将 FS 的 SECS 从 0.765 恢复到 0.807(+0.042),CER 略增(4.738→4.962),说明 domain conditioning 有效隔离了合成域的声学偏置。

3. **跨 ZS-TTS 源的一致性** [§4.2]: Fish-Speech 和 CosyVoice 2 两个架构完全不同的 ZS-TTS 源呈现一致趋势,表明方法不依赖特定生成器。

4. **Speaker-matched vs Speaker-mismatched** [§4.4, Table 6]: Speaker-mismatched 合成数据(同性别不同说话人)几乎无法提升 SECS(0.792 vs baseline 0.818),而 speaker-matched 达到 0.807。t-SNE 可视化 [Fig 2] 显示 mismatched 数据与 real 数据的分离更严重。[agent 解读]: 这证明合成数据的价值不仅来自"语音稳定性降低声学变异",更来自"speaker-consistent 的音色信息",否则 mismatched 数据的语言增强效果应与 matched 相当。

5. **主观评估** [§4.2, Table 4]: ABX 测试中,所有 DC+OS 配置均显著优于 naive mixing(p<0.05),FS 70.8%, CV2 61.8%(LibriTTS);MOS 与 Real 100% 和 naive mixing 相当,说明不牺牲自然度。

## 局限性

1. **架构覆盖面窄** [agent 解读]: 仅在 VITS(2021 年架构)上验证,未测试 LLM-based TTS(如 VALL-E、CosyVoice)或 diffusion-based TTS。论文在 §5 承认未来需"extending ZeSTA to diverse TTS architectures"。对于现代 LLM-based TTS,domain embedding 的注入方式可能需要重新设计(例如是加到 prompt token 还是 decoder hidden states)。

2. **说话人规模有限**: 仅 8(LibriTTS)+6(YoBind)= 14 个说话人,且均为阅读式语音。对于情感、口音、嘈杂环境等场景的泛化性未知。

3. **数据比例固定**: Real 10% + Synth 90% 是唯一测试的比例,未探索不同比例(如 30/70、50/50)下的最优配置。

4. **未与参数效率方法对比**: 未与 LoRA、adapter、CLN 等参数效率微调方法对比或结合。[agent 解读]: ZeSTA 的 domain embedding 与这些方法正交,组合使用可能进一步提升效果。

5. **OS 因子固定为 3**: 未消融不同 oversampling 倍率的影响。

## 点评

**优势**:
- 方法极度简单(仅加一个 64-dim embedding + 数据复制),无需修改基础架构,工程成本几乎为零
- 实验设计系统:两个 ZS-TTS 源 × 两个数据集 × 消融 DC/OS/embedding size/speaker consistency,逻辑完整
- 揭示了一个有实际意义的现象:naive 混合合成数据会降低 speaker similarity,且 OS alone 不能解决
- t-SNE 可视化和 speaker-matched/mismatched 消融提供了对 domain embedding 机制的直觉理解

**不足**:
- [agent 解读] VITS 在 2026 年已是遗产架构,实际工业部署更可能使用 LLM-based 或 flow-based 模型。方法的实际影响取决于能否迁移到现代架构
- 缺少 domain embedding 学到了什么的分析(例如 real/synth embedding 的余弦距离、对 flow module 不同层的影响)
- ABX 测试的 listener 仅 18 人,统计功效有限
- 未讨论当 ZS-TTS 质量极高(SECS > 0.85)时 domain gap 是否会缩小到 DC 不再必要的程度

## 可复用的 idea

1. **Domain embedding 范式**: 在任何混合数据(真实+合成)微调场景中,用一个轻量 binary embedding 区分数据来源,推理时设为"真实"域。可推广到: voice conversion 数据增强、ASR 数据增强(TTS 合成训练数据)、多说话人模型中区分 studio vs wild 录音。

2. **Real-data oversampling 作为 DC 的 complement**: 在 DC 消除域偏置后,简单重复真实数据即可进一步提升目标域性能。这是一个零成本的通用 trick。

3. **ZS-TTS 输出过滤**: 用 Whisper 对合成数据做 WER 过滤(阈值 5%)去除 hallucination,适用于任何使用 TTS 合成数据的 pipeline。

4. **Longest-prompt 选择策略**: 选择最长 utterance 作为 ZS-TTS 的 reference prompt 以最大化 speaker-style 覆盖 [§4.1]。

## 审阅

(待独立审阅 agent 填充)
