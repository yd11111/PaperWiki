---
type: paper
tier: deep
title: "Frame-Wise Breath Detection with Self-Training: An Exploration of Enhancing Breath Naturalness in TTS"
arxiv_id: "2402.00288"
source: "Sources/FrameWiseBreath.pdf"
authors: [Dong Yang, Tomoki Koriyama, Yuki Saito]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, breath-detection, self-training, Conformer, sound-event-detection, naturalness, semi-supervised, pseudo-labeling]
concepts: ["[[ProsodyModeling]]", "[[MelSpectrogram]]", "[[TTSEvaluation]]", "[[DurationPredictor]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[ProsodyModeling]]✓, [[MelSpectrogram]], [[VITS]], [[TTSEvaluation]], [[DurationPredictor]], [[VariationalAutoencoderforTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ProsodyModeling]], [[MelSpectrogram]], [[VITS]], [[TTSEvaluation]], [[DurationPredictor]], [[VariationalAutoencoderforTTS]] | 过滤: 5 页 pending-review | 未命中但可能相关: 无

**谱系定位**: 本文属于 TTS 韵律建模中的**副语言发声 (Paralinguistic Vocalizations)** 子领域。[[ProsodyModeling]] 页已记录 NVSpeech 将韵律扩展至副语言发声维度(笑声、叹气、呼吸等非语言声音),但 NVSpeech 侧重"在 TTS 中插入和合成 PV",本文则聚焦更上游的问题 --- 如何自动检测呼吸音的精确位置。两者互补: 本文提供检测工具,NVSpeech 式方法消费检测结果。

**已有认知**:
- 呼吸音是 PINT (Pause-Internal Phonetic Particles) 的一种,与 silence、tongue clicks、filled pauses 并列 [ProsodyModeling §副语言发声]
- VITS 作为端到端 TTS 的代表模型,已被广泛用作 TTS 实验的 backbone [VITS 页]
- Mel Spectrogram 是呼吸检测的核心输入特征 [MelSpectrogram 页]
- MFA (Montreal Forced Aligner) 是获取 pause 位置的标准工具 [DurationPredictor 页]
- MOS 测试是 TTS 自然度评估的主要手段,但存在 ceiling effect 等局限 [TTSEvaluation 页]

**创新判断**: KB 中尚无专门的呼吸检测概念页。本文的核心创新在于(1)提出无需人工标注的呼吸检测训练方法(规则初始化 + 自训练),(2)设计帧级检测模型(10ms 分辨率 vs 基线 50ms)。这填补了从韵律建模到副语言发声合成之间的"检测"环节。

## 速查

> [!summary] 速查
> - **一句话**: 提出基于 Conformer + 自训练的帧级呼吸音检测模型,无需人工标注即可准确检测呼吸位置,提升 TTS 合成语音中呼吸音的自然度
> - **路线**: 语音 → MFA pause 检测 → 规则特征提取(Duration/ZCR/VMS/NA-VMS）→ 高精度初始标注 → Conformer+BiLSTM 检测模型 → 自训练伪标签迭代 → breath marks 插入文本 → VITS TTS 训练
> - **指标**: 呼吸检测 IoU 0.836 (vs 基线 0.710) [Table 3a]; MOS2 (呼吸自然度) 3.55 (vs VITS 3.34) [Table 4]
> - **可借鉴**: (1) 规则高精度标注 + 自训练伪标签的半监督策略,可迁移到其他副语言发声检测; (2) Conformer + 下采样/上采样实现帧级检测的架构设计
> - **局限**: (1) 推理时呼吸位置由 GT 语音检测给出,实际应用需额外的语言模型预测 breath marks; (2) 未区分吸气/呼气; (3) 仅在 LibriTTS-R (有声书) 上验证; (4) MOS1 整体自然度提升不显著 (3.37 vs 3.35)

## 核心问题

1. **为什么 TTS 需要关注呼吸音?** 呼吸音是感知自然度的重要因素 [§1],但常规 TTS 训练数据中呼吸音未被显式标注,导致合成语音的呼吸表现不自然,甚至被 deepfake 检测利用 [§1, ref 10]。

2. **现有呼吸检测方法有什么问题?** 两类方法各有短板: 规则方法不需要人工标注但精度不足; 机器学习方法精度更高但依赖大量人工标注,且现有工作仅在小规模语料上验证 [§1]。

3. **本文要解决什么?** 提出一个不需要人工标注训练数据的呼吸检测方法,在大规模多说话人语料上训练帧级检测模型,并验证其对 TTS 呼吸自然度的提升效果 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

本文方法包含三个阶段:

**阶段 1: 规则特征提取与自动标注** [§2.1]
- 使用 MFA 对 LibriTTS-R 语料做强制对齐,检测所有 pause 位置
- 基于四种声学特征对 pause 进行分类:
  1. **Duration**: 排除 <300ms 的短暂停顿(主要是短暂 silence 和噪声)[§2.1]
  2. **ZCR (Zero-Crossing Rate)**: 音频信号过零率,区分 silence 与其他声音 [§2.1, Eq.1]
  3. **VMS (Variance of Mel-Spectrogram)**: Mel 频谱在频率维度的方差,辅助区分 silence [§2.1] --- 这是本文新提出的特征
  4. **NA-VMS (Normalized Average of VMS)**: VMS 的归一化均值,量化"VMS-Min 面积占 Max-Min 面积的比例",有效区分呼吸音与 tongue clicks(tongue clicks 表现为 VMS 短暂尖峰但 NA-VMS 低) [§2.1, Eq.2, Fig 1] --- 这也是本文新提出的特征
- 使用阈值组合 [Table 2] 自动标注呼吸/非呼吸数据,呼吸标注精度 0.982,非呼吸标注精度 1.000 [§2.1]

**阶段 2: Conformer 检测模型** [§2.2]
- 输入: Mel-spectrogram + ZCR + VMS (三通道)
- Downsampling: 两层 2D-CNN (kernel 3x3, stride 2x2),将 10ms 帧率压缩到 40ms [Fig 2]
- Encoder: 8 个 Conformer blocks (4 attention heads, conv kernel 31, hidden 256) [§2.2]
- Upsampling: 两层 transposed 1D-CNN (kernel 3, stride 2),恢复到 10ms 帧率 [Fig 2]
- Decoder: BiLSTM → Linear → Sigmoid,输出每帧呼吸概率 [§2.2]
- 损失: Binary cross-entropy,忽略未标注帧 (label=-100) [§2.2, Eq.3]

**阶段 3: 自训练** [§2.3, Algorithm 1]
- 初始训练: 用阶段 1 标注的呼吸/非呼吸数据训练初始检测器 D_theta_0
- 迭代伪标签: 每轮用当前检测器的预测生成新标注 --- 预测概率 > alpha_k 标为呼吸,< beta_k 标为非呼吸 [Algorithm 1]
- 动态阈值: alpha_k, beta_k 在验证集上调整,初始精度目标 0.98,每轮降低 0.02 [§3.1]
- 终止条件: 验证集性能下降时停止,返回上一轮最优检测器 [Algorithm 1]

### 关键设计选择

1. **为什么用 Conformer 而不是纯 CNN?** [论文原文] Conformer 在 ASR 中表现突出 [§2.2]。[agent 解读] 呼吸检测需要同时捕捉局部声学模式(CNN 擅长)和长程上下文依赖(self-attention 擅长),Conformer 的 conv+attention 混合架构天然适合。

2. **为什么需要上采样模块?** [论文原文] 下采样提高计算效率但降低检测分辨率(基线模型仅 50ms)。上采样恢复时间维度,实现 10ms 帧级检测 [§2.2]。[agent 解读] 10ms 分辨率对精确定位呼吸边界至关重要,因为呼吸音的起止时间直接影响 breath mark 的准确性。

3. **为什么自训练比单纯增加训练数据有效?** [论文原文] 消融实验显示,不用伪标签直接继续训练(iteration 1)性能反而下降 (IoU 0.740 vs 0.809) [Table 3b]。[agent 解读] 自训练通过动态阈值逐步扩展标注集,在标注质量和数量之间取得平衡;而直接在未标注数据上训练会引入过多噪声。

4. **VMS 和 NA-VMS 的互补性**: [论文原文] VMS 帮助区分 silence 和有声 pause(silence 的 ZCR 和 VMS 都低),NA-VMS 帮助区分呼吸和 tongue clicks(tongue clicks 有短暂 VMS/ZCR 尖峰但 NA-VMS 低)[§2.1]。消融证实去掉 VMS 或 ZCR 均显著降低 IoU [Table 3b]。

### 训练策略

- **数据**: LibriTTS-R "train-clean-100" + "train-other-500" 子集 [§3.1]
- **音频处理**: 16kHz 采样率,128 Mel bands,log-scale [§3.1]
- **优化器**: AdamW,线性学习率调度(10% warmup),峰值学习率 2e-5 [§3.1]
- **每轮训练**: 10 epochs,batch size 64,单张 A100 GPU [§3.1]
- **自训练迭代**: 最优性能出现在第 3 轮,第 4 轮开始退化 [Table 3a]

## 实验

### 呼吸检测实验 [§3.1]

| 指标 | Proposed (Iter.3) | Baseline (Iter.3) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| IoU | 0.836 | 0.710 | LibriTTS-R test-clean | [Table 3a] |
| Precision | 0.924 | 0.836 | LibriTTS-R test-clean | [Table 3a] |
| Recall | 0.897 | 0.824 | LibriTTS-R test-clean | [Table 3a] |

**消融实验** [Table 3b]:

| 消融条件 | IoU | 出处 |
| --- | --- | --- |
| Proposed (Iter.0, full) | 0.777 | [Table 3b] |
| w/o ZCR | 0.631 (-0.146) | [Table 3b] |
| w/o VMS | 0.677 (-0.100) | [Table 3b] |
| w/o non-breath | 0.702 (-0.075) | [Table 3b] |
| Proposed (Iter.1) | 0.809 | [Table 3b] |
| Iter.1 w/o pseudo-label | 0.740 (-0.069) | [Table 3b] |

### TTS 实验 [§3.2]

| 指标 | Ground truth | VITS | VITS w/ baseline | VITS w/ proposed | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS1 (overall) | 4.03+-0.12 | 3.35+-0.15 | 3.27+-0.15 | 3.37+-0.14 | [Table 4] |
| MOS2 (breath) | 3.92+-0.13 | 3.34+-0.17 | 3.50+-0.14 | 3.55+-0.15 | [Table 4] |

**关键发现**:
- MOS1: VITS w/ baseline 反而降低了整体自然度 (3.27 < 3.35),说明不准确的呼吸检测会损害 TTS 训练 [§3.2]
- MOS2: 两种 breath-informed 方法都提升了呼吸自然度,但 proposed 略优 (3.55 vs 3.50) [§3.2]
- VITS w/ proposed 能为训练数据中缺乏呼吸音的说话人合成呼吸音 (speaker "3630" 和 "1811") [§4]
- Baseline 模型因错误检测导致 TTS 误将 silence 学为呼吸音,仅能为 "1811" 合成呼吸 [§4]

## 局限性

1. **推理时的 breath mark 来源未解决**: 论文在 TTS 测试时使用 GT 语音的检测结果插入 breath marks,实际部署需要语言模型从文本预测呼吸位置,论文承认这一点但未提供解决方案 [§4]

2. **未区分吸气与呼气**: 论文指出呼吸包含吸气和呼气,但 LibriTTS-R 中呼气很少,因此未做区分 [§2.1]。这限制了模型在呼气频繁的场景(如对话、运动)中的适用性

3. **整体自然度提升有限**: MOS1 结果中 VITS w/ proposed (3.37) 与 VITS (3.35) 的差异在置信区间内不显著,呼吸检测主要改善呼吸相关自然度而非整体质量 [Table 4]

4. **仅在有声书数据上验证**: LibriTTS-R 来自有声书朗读,呼吸模式相对规律;自发对话、情感语音等场景的呼吸模式可能不同

5. **规则标注的阈值依赖观察**: 规则方法的特征阈值基于训练数据观察和验证集微调确定 [§2.1],泛化到其他语料/语言可能需要重新调整

## 点评

**优势**:
- 方法设计完整且自洽: 从特征分析 → 规则标注 → 模型训练 → 自训练 → TTS 验证,形成了完整的 pipeline
- 消融实验充分: 每个组件(VMS/ZCR/non-breath/pseudo-label)都有独立验证,说服力强
- 自训练策略的动态阈值设计巧妙: 逐轮降低精度要求以扩大伪标签覆盖,兼顾质量和数量
- 提出的 VMS/NA-VMS 特征有物理直觉且消融验证有效

**不足**:
- TTS 实验的 baseline 较弱: 仅与 Szekely et al. 2019 的 CNN-LSTM 对比,未与更现代的声音事件检测方法比较
- MOS1 提升不显著是核心弱点: 如果整体自然度几乎不变,breath detection 的实用价值受限
- 推理时 breath mark 预测的缺失使得 end-to-end 应用不完整
- 评估只用 MOS,未使用 PESQ/STOI 等客观指标补充

**与 KB 已有知识的定位**: 本文是 ProsodyModeling 页中"副语言发声建模"方向的上游工作。NVSpeech 从 TTS 合成端解决 PV 的可控插入,本文从检测端解决呼吸音的自动标注问题。两者结合可实现: 自动检测 → 标注 → TTS 训练 → 可控合成的完整链路。

## 可复用的 idea

1. **规则高精度标注 + 自训练**: 先用严格规则获得少量高精度标注(precision ~0.98),再通过自训练逐步扩展,适用于任何标注成本高的音频事件检测任务(如 filled pauses、tongue clicks、laughter)

2. **下采样-Conformer-上采样架构**: 通过下采样降低 Conformer 的计算量,再上采样恢复时间分辨率,可迁移到其他需要帧级精细检测的任务

3. **NA-VMS 特征**: 归一化 VMS 均值作为区分不同类型声音事件的特征,可能对其他非语言声音(如 clicks、lip smacks)的检测也有效

4. **TTS 中 breath mark 的训练策略**: 在训练文本中插入检测到的 breath marks 让 TTS 模型学习呼吸模式,比隐式学习更可控,且能为无呼吸说话人"转移"呼吸能力
