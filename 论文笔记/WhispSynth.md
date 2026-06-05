---
type: paper
tier: deep
title: "WhispSynth: Scaling Multilingual Whisper Corpus through Real Data Curation and A Novel Pitch-free Generative Framework"
arxiv_id: "2603.14853"
source: "Sources/WhispSynth.pdf"
authors: [Tianyi Tan, Jiaxin Ye, Yuanming Zhang, Xiaohuai Le, Xianjun Xia, Chuanzeng Huang, Jing Lu]
year: 2026
venue: "arXiv"
tags: [whisper-speech, data-engine, DDSP, pitch-removal, TTS-data, corpus-construction, multilingual, voice-conversion]
concepts: ["[[ConditionalFlowMatching]]", "[[F0Modeling]]", "[[NeuralVocoder]]", "[[SpeakerEmbedding]]"]
models: ["[[CosyVoice3]]", "[[BigVGAN]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[ConditionalFlowMatching]], [[NeuralVocoder]], [[SpeakerEmbedding]] + 3 个待确认: [[CosyVoice3]], [[F0Modeling]], [[BigVGAN]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]]✓, [[NeuralVocoder]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[CosyVoice3]](pending-review), [[F0Modeling]](pending-review), [[BigVGAN]](pending-review) | 未命中但可能相关: 无

**谱系定位**: 本文不是一个新 TTS 架构,而是一个面向耳语语音合成的**数据引擎**(data engine)。它构建在 [[CosyVoice3]] 之上,利用 CosyVoice3 的 LLM+CFM 管线生成初始耳语音频,然后通过 DDSP-based pitch-free 后处理移除残留 F0,产出高保真耳语语料。最终产物 CosyWhisper 本质是 CosyVoice3 的 CFM 模块在 WhispSynth 上的 fine-tune。

**已有认知**: KB 中 [[F0Modeling]] 页面 [待确认] 详述了 F0 在 TTS/SVS 中的建模方法,但主要聚焦于 F0 的**预测与控制**,未涉及 F0 的**移除**场景。本文提出的 pitch-free 方法是 F0 建模的反向操作 — 利用 DDSP 源滤波器模型分离谐波分量后丢弃,保留噪声分量以模拟真实耳语的非周期性激励。[[NeuralVocoder]] 页面中 DDSP 仅作为歌声合成的子类提及,本文将 DDSP 重新定位为信号处理工具(分离谐波/噪声)而非生成器。[[BigVGAN]] 的 MRD+SPD 判别器被复用于 DDSP vocoder 的对抗训练。

**创新判断**: 与 KB 中已有的 normal-to-whisper 方法(如 SeedVC 零样本 voice conversion、传统 LPC/WORLD vocoding)相比,本文的核心创新在于将问题分解为"先用强 TTS 合成、再用 DDSP 去除残留 pitch"的两阶段 pipeline,而非端到端训练一个耳语生成器。这种分解思路的优势在于可以复用现有大规模 TTS 模型的语言理解能力,仅需解决 pitch 域的后处理问题。

## 速查

> [!summary] 速查
> - **一句话**: 提出 DDSP-based pitch-free 后处理 + CosyVoice3 的数据引擎,将嘈杂的真实耳语数据升级为 118h 高保真合成耳语语料(WhispSynth),fine-tune 出的 CosyWhisper W-MOS 4.53 超过真实录音 4.33
> - **路线**: 真实耳语 WhispReal (118h) → CosyVoice3 合成初始耳语 → DDSP pitch detection → 谐波/噪声分离 → 丢弃谐波+OLA 重建 → WhispSynth (118h, 24kHz) → CFM fine-tune → CosyWhisper
> - **指标**: W-MOS 4.53 vs GT 4.33 vs CosyVoice3 3.40 [Table 4]; CER 12.76%/WER 29.22% [Table 4]; VTR 0.88 [Table 4]; WhispSynth CER/WER 比 WhispReal 降 46% [Table 5]
> - **可借鉴**: (1) 用 DDSP 源滤波器做信号后处理(分离谐波→丢弃)而非生成,思路可迁移到任何需要移除周期性成分的场景; (2) 三阶段 DDSP 训练策略(正常语音对抗训练→耳语继续训练→半监督双焦点); (3) 只 fine-tune CFM 不动 LLM 和 vocoder 的选择,表明耳语与正常语音的差异主要在声学建模层
> - **局限**: (1) 依赖 CosyVoice3 闭源预训练权重; (2) 未评估麦克风差异的影响 [Limitations]; (3) 开源模型内嵌音频水印可能影响声学特性 [Limitations]; (4) 多语言验证仅日/韩两种且数据来自 YouTube ASMR [Appendix D]; (5) WER 29.22% 仍然较高

## 核心问题

1. **数据瓶颈**: 公开耳语语音数据集规模小、录音质量参差、说话人多样性不足,直接制约 text-to-whisper 研究 [§1]
2. **TTS 模型的耳语生成缺陷**: 现有开源 TTS 模型在零样本耳语生成上严重失败,原因是语料和 vocoder 都针对有声语音优化,speaker encoder 将耳语误判为嘶哑或噪声 [§1]
3. **合成耳语的残留 pitch**: 即使 CosyVoice3 在百万小时数据上训练,其合成耳语仍会引入残留 F0 伪影,因为神经模块主要在有声语音上训练 [§3.2]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

系统由三个层次组成:

1. **WhispReal** (数据整理层): 整合 6 个公开耳语语料库 + 新录制的 WhispNJU (45h, 77 说话人, 普通话),形成 118h/479 说话人的双语(英/中)真实耳语数据集 [§2]
2. **WhispSynth** (数据引擎层): CosyVoice3 生成初始耳语 → DDSP pitch-free 后处理 → 产出约 118h 的 24kHz 高保真合成耳语 [§3]
3. **CosyWhisper** (模型层): 在 WhispSynth 上 fine-tune CosyVoice3 的 CFM 模块 [§4]

### 关键设计选择

**为什么用 DDSP 做 pitch 移除而非端到端训练耳语生成器?**

[论文原文] TTS 模型训练在有声语音上,天然具备耳语生成的"潜在能力"(dormant reservoir of whisper-like priors),因为正常语音中的清辅音段(devoiced segments)与耳语共享几乎相同的气动-发音约束 [§1]。因此问题不是从零训练耳语生成,而是利用这种潜在能力并移除不需要的 F0 残留。

[agent 解读] 这是一个 pragmatic 的工程决策:与其训练一个需要大量耳语数据的端到端模型,不如复用 CosyVoice3 的语言理解能力(LLM 已在百万小时上训练),只解决输出域的后处理问题。这将数据需求从"大量耳语"降低到"足以训练一个 DDSP vocoder"。

**DDSP 源滤波器分解** [§3.3]:

将音频信号分解为谐波分量 H 和噪声分量 N: `S[i] = H[i] + N[i]` [Eq.1]。谐波部分用锯齿波信号通过 LTV-FIR 滤波器整形 [Eq.2-4],噪声部分用均匀分布噪声通过另一个 LTV-FIR 滤波器 [Eq.5]。所有参数通过 Conformer 网络从 mel spectrogram 帧级预测 [Eq.6]。

**Pitch-free 后处理流程** [§3.1, Algorithm 1]:

```
CosyVoice3 合成初始耳语 → DDSP pitch detection 检测残留 F0 段 →
对每个含 F0 的段: DDSP 分解为 (H, N) → 丢弃 H, 保留 N →
加窗 + Overlap-Add 重建 → 最终 pitch-free 耳语
```

[论文原文] 保留噪声分量是因为它携带了耳语的频谱包络和非周期性激励信息 [§3.1, Appendix A]。

**为什么只 fine-tune CFM 而不动 LLM 和 HiFi-GAN?**

[论文原文] 耳语与正常语音的区别主要在声学建模(acoustic modeling)而非语义内容,因此选择只微调负责 token→acoustic feature 转换的 CFM 模块 [§4]。

[agent 解读] 这个选择很合理:LLM 处理的是语义 token 序列,与发音模式无关;HiFi-GAN 从 mel spectrogram 合成波形,耳语的 mel 与正常语音结构差异不大(主要是能量分布和谐波缺失)。CFM 是连接语义和声学的桥梁,正是需要适配的地方。

### 训练策略

**DDSP vocoder 三阶段训练** [§3.4]:

1. **对抗训练(正常语音)**: 将 DDSP 生成器集成到 BigVGAN 框架,使用 MRD + SPD 判别器 + LS-GAN loss。[论文原文] 对抗训练对改善谐波结构建模和抑制伪影至关重要 [§3.4 (i)]
2. **继续训练(耳语语音)**: 在 WhispReal 上继续训练,不使用对抗目标以加速训练。[论文原文] 两阶段流程先学鲁棒的 pitch-conditioned 合成,再特化耳语生成,避免直接在有限耳语数据上训练的不稳定性 [§3.4 (ii)]
3. **半监督双焦点训练**: WhispReal 中部分样本含有可感知的 pitch 轮廓(说话人失误或声带疲劳所致)。训练方案始终将输出视为谐波+随机分量之和,但根据输入类型重新加权梯度。[论文原文] 耳语和正常语音共享 loss,但梯度重新加权谐波/噪声项,使两类数据相互增强 [§3.4 (iii)]

**CosyWhisper CFM fine-tune** [§4, Appendix B]:

官方 CosyVoice3 脚本仅支持 LLM 训练,作者扩展为 CFM 训练。关键改动: 添加 token projection layer、用直接 embedding lookup 替换 encoder、修改 conditioning 机制、增加 token-to-mel alignment (repeat_interleave) [Listing 2 vs Listing 1]。

### WhispNJU 数据集 [§2.2]

77 名母语为普通话的说话人(37 男 40 女,22-26 岁),每人录制 250 句(约 1 小时),每句正常语音和耳语各一遍。基于 THCHS-30 数据集的文本,分为 A/B/C/D 四组(各 250 句)。总计约 85 小时(含正常+耳语对),其中耳语约 45 小时。

## 实验

| 指标 | 本文 (CosyWhisper) | Baseline (CosyVoice3) | Ground Truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| W-MOS ↑ | **4.53±0.20** | 3.40±0.51 | 4.33±0.33 | WhispReal test | [Table 4] |
| DNSMOS ↑ | 3.08 | 3.00 | 2.80 | WhispReal test | [Table 4] |
| UTMOS ↑ | 1.48 | 1.47 | 1.44 | WhispReal test | [Table 4] |
| MCD ↓ | 59.31 | 56.03 | 0.00 | WhispReal test | [Table 4] |
| CER/WER ↓ | 12.76%/29.22% | 12.51%/9.81% | 39.30%/37.58% | WhispReal test | [Table 4] |
| SpkSim ↑ | 0.80 | 0.83 | 1.00 | WhispReal test | [Table 4] |
| VTR ↓ | 0.88 | 0.86 | 0.88 | WhispReal test | [Table 4] |

**Normal-to-Whisper Conversion 对比** [Table 4]:

| 方法 | W-MOS ↑ | VTR ↓ | CER/WER ↓ |
| --- | --- | --- | --- |
| Whisper-Effect (Roh et al., 2025) | — | 0.12 | 59.46%/80.35% |
| toWhisper (LPC-based) | 1.02±0.29 | 0.92 | 12.07%/9.27% |
| Normal2Whisper (WORLD) | 2.18±0.47 | 0.94 | 11.92%/28.28% |
| SeedVC | 1.30±0.29 | 0.71 | 58.65%/19.97% |
| Pitch-free Model (ours) | — | 0.98 | 21.31%/45.18% |

**Ablation: WhispReal vs WhispSynth 作为训练数据** [Table 5]:

| 训练数据 | DNSMOS ↑ | UTMOS ↑ | CER/WER ↓ | SpkSim ↑ | VTR ↓ |
| --- | --- | --- | --- | --- | --- |
| WhispReal | 2.93 | 1.33 | 28.3%/46.5% | 0.75 | 0.77 |
| WhispSynth | **3.08** | **1.48** | **12.8%/29.2%** | **0.80** | **0.70** |

[论文原文] WhispSynth 训练的模型在所有关键指标上一致优于 WhispReal: CER/WER 降 46%, VTR 降 9%, 自然度提升 8% [§5.5]。这证实合成数据不仅弥补了真实耳语数据的噪声和不一致性,还提供了更好的训练信号。

**多语言扩展** [Appendix D, Table 6]:

| 语言 | 方法 | W-MOS ↑ | VTR ↓ | CER ↓ |
| --- | --- | --- | --- | --- |
| 韩语 | CosyVoice3 | 3.35±1.70 | 0.83 | 27.21% |
| 韩语 | CosyWhisper | **3.96±1.04** | **0.75** | **23.32%** |
| 日语 | CosyVoice3 | 2.88±1.36 | 0.96 | 39.42% |
| 日语 | CosyWhisper | **4.03±0.58** | **0.82** | **36.22%** |

## 局限性

1. **WER 问题**: CosyWhisper 的 WER (29.22%) 显著高于 CosyVoice3 (9.81%),虽然 W-MOS 更高但清晰度有所牺牲 [Table 4]。[agent 解读] 这暗示 CFM fine-tune 在追求耳语真实感时可能损害了发音清晰度,存在 intelligibility-whisperness trade-off
2. **对 CosyVoice3 的依赖**: 整个 pipeline 依赖 CosyVoice3 的预训练能力,且官方脚本需要 hack 才能支持 CFM fine-tune [§4, Appendix B]。可复现性受限于 CosyVoice3 的开源程度
3. **多语言验证有限**: 日语/韩语验证数据来自 YouTube ASMR 视频手动整理,每种语言仅 4 个说话人(2男2女),统计效力不足 [Appendix D]
4. **评估指标局限**: 作者自己指出现有客观指标与耳语特性对齐不佳(poorly aligned),人工评估仍不可替代 [§5.2]
5. **麦克风差异未评估**: 不同录音设备对模型性能的影响未系统测试 [Limitations]
6. **音频水印**: 开源模型嵌入实时音频水印可能影响声学特性 [Limitations]

## 点评

本文的定位很有趣 — 它本质上是一个**数据工程**工作,而非模型创新。核心 insight 是"TTS 模型已经有耳语合成的潜在能力,问题只在于 pitch 残留",这个观察驱动了整个 pipeline 的设计。

**优势**: (1) 问题分解巧妙,将端到端耳语生成拆解为"复用强 TTS + DSP 后处理",降低了数据和训练成本; (2) DDSP 三阶段训练策略设计合理,特别是半监督双焦点训练处理了真实耳语数据中的 pitch 混入问题; (3) WhispReal 的整理工作本身有价值,统一了 6 个异质数据源的格式和划分; (4) 开源承诺明确(MIT 许可,代码/模型/数据均计划开放)

**不足**: (1) W-MOS 超过 GT (4.53 vs 4.33) 的结果需谨慎解读 — 可能意味着合成耳语比真实录音"更像耳语"(过度风格化),而非质量更高; (2) CosyWhisper 的 WER 大幅上升(29.22% vs CosyVoice3 的 9.81%)是一个严重的 trade-off,论文未充分讨论; (3) Pitch-free model 单独作为 conversion 工具时 CER/WER 很高(21.31%/45.18%),说明 DDSP 分离过程确实损失了部分语音信息

## 可复用的 idea

1. **DDSP 作为信号处理工具**: 将 DDSP 的源滤波器分解用于移除不需要的信号成分(而非合成),这个思路可迁移到去除背景音乐中的人声(反向应用)、去除特定频段污染等场景
2. **"强模型 + 轻后处理"范式**: 与其训练一个领域特化的端到端模型,不如复用通用大模型的能力并用轻量后处理适配目标域。类似 LoRA 的逻辑但在信号层面操作
3. **只 fine-tune CFM 的选择**: 在 coarse-to-fine TTS 中,声学特性的适配可以只动 fine stage (CFM),不需要改动 coarse stage (LLM) 和 vocoder。这为其他语音风格适配(如情感语音、低语、喊叫)提供了高效路径
4. **合成数据优于真实数据的场景**: 当真实数据质量差(噪声大、不一致)时,用强模型生成的合成数据反而能提供更好的训练信号。WhispSynth vs WhispReal 的对比 [Table 5] 是一个清晰的案例

---

检索命中: [[ConditionalFlowMatching]]✓, [[NeuralVocoder]]✓, [[SpeakerEmbedding]]✓ | 过滤: [[CosyVoice3]](pending-review), [[F0Modeling]](pending-review), [[BigVGAN]](pending-review) | 未命中但可能相关: 无
