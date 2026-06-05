---
type: paper
tier: deep
title: "STTATTS: Unified Speech-To-Text And Text-To-Speech Model"
arxiv_id: "2410.18607"
source: "Sources/STTATTS.pdf"
authors: ["Hawau Olamide Toyin", "Hao Li", "Hanan Aldarmaki"]
year: 2024
venue: "arXiv"
tags: [multi-task-learning, ASR, TTS, encoder-decoder, parameter-efficient, speech-text-unified, low-resource, Arabic-TTS, voice-conversion]
concepts: ["[[Self-SupervisedSpeechRepresentation]]", "[[SpeakerEmbedding]]", "[[NeuralVocoder]]", "[[MelSpectrogram]]", "[[Speech-TextAlignment]]", "[[Attention-basedTTS]]", "[[Text-to-SpeechPipeline]]", "[[TTSEvaluation]]"]
models: ["[[模型库/Whisper|Whisper]]", "[[模型库/wav2vec2.0|wav2vec 2.0]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: STTATTS 属于统一 speech-text encoder-decoder 模型家族,与近年 Speech Language Model 的发展方向相关但路线不同。SpeechLM 领域主流路线是离散 speech token + LLM 自回归建模 ([[SpeechLanguageModel]]),而 STTATTS 延续 SpeechT5 的连续表征 + mel spectrogram 路线。从 [[Speech-TextAlignment]] 视角看,STTATTS 采用 concatenated/shared encoder-decoder 方式,而非当前主流的 alternating tokens 或 multi-sequence 对齐。
>
> **已有认知**:
> - [[SpeakerEmbedding]]: STTATTS 使用 x-vector 做多说话人 TTS,这是经典的 speaker encoder 方案,在 KB 中属于 "Speaker Encoder (零样本)" 类别中的 TDNN-based 统计池化方法。
> - [[NeuralVocoder]]: STTATTS 使用 HiFi-GAN 做波形合成,这是 2020-2023 最广泛使用的 vocoder,在 KB 中有详细记录 (MRF 生成器 + MPD+MSD 判别器)。
> - [[MelSpectrogram]] [待确认]: 80-dim log mel-filterbank 作为 TTS 输出目标,是中期 neural TTS 的标准中间表示。
> - [[Self-SupervisedSpeechRepresentation]] [待确认]: SpeechT5 的预训练属于 speech-text 联合自监督学习,与 wav2vec 2.0/HuBERT 的纯语音 SSL 不同,但共享 encoder 架构。
>
> **创新判断**: STTATTS 的核心创新不在单项模块,而在于 multi-task 联合训练策略——通过一个 task fusion module 在同一 encoder-decoder 中同时优化 ASR 和 TTS,参数量仅为单任务模型之和的 ~50%。这与 SpeechT5 (分别微调) 和 VoxtLM (离散 token + decoder-only) 形成差异。
>
> 检索命中: [[SpeakerEmbedding]]✓, [[NeuralVocoder]]✓, [[SpeechLanguageModel]]✓ | 过滤: [[MelSpectrogram]](pending-review), [[Self-SupervisedSpeechRepresentation]](pending-review), [[Speech-TextAlignment]](pending-review) | 未命中但可能相关: SpeechT5, ArTST (无实体页)

## 速查

> [!summary] 速查
> - **一句话**: 通过 task fusion module + multi-task loss 在单个 SpeechT5 encoder-decoder 中同时学习 ASR 和 TTS,参数量仅为分离训练的 ~50%,性能可比单任务模型
> - **路线**: 语音/文本 → modal-specific pre-net → shared transformer encoder → task fusion module (128-dim task vector + FC) → shared decoder → modal-specific post-net → 文本/mel spectrogram → HiFi-GAN
> - **指标**: English ens WER 4.84 / TTS CER 3.18 (155M params) vs VoxtLM_base WER 6.5 / TTS CER 3.5 (350M params); English enl WER 2.99 / TTS CER 2.10 [Table 3, 5]
> - **可借鉴**: (1) 极简 task fusion module (128-dim vector + FC) 在共享 encoder 输出上条件化任务,几乎无额外参数开销; (2) warm fine-tuning 策略解决 ASR/TTS 数据不平衡问题; (3) 联合训练改善 ASR 性能 (ASR+TTS 数据 > 单 ASR 数据)
> - **局限**: 合成语音 MOS 3.0-3.4 (enl 环境),自然度仍有差距; 仅比较 VoxtLM 一个 joint-task baseline; 未尝试更大模型规模; 阿拉伯语数据太少 (32h) 导致 TTS 质量有限

## 核心问题

ASR 和 TTS 传统上使用独立模型训练,各自拥有不同的学习目标、训练数据和参数,导致两个大型网络的冗余。问题是:**能否在一个模型中同时学习 ASR 和 TTS,在保持各自性能的同时大幅减少参数量?**

相关工作中:
- SpeechT5 虽然做了联合预训练,但下游 ASR/TTS 仍需分别微调,最终是两套独立参数 [论文原文, §2.1]
- VoxtLM 通过离散 token + decoder-only 实现联合建模,但参数量大 (350M-1.3B) [论文原文, §5.2]
- 其他方法 (SpeechGPT, VioLA) 的代码和模型不公开,无法直接比较 [论文原文, §7]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

STTATTS 基于 SpeechT5 架构 [§3.1],由三个层次组成:

1. **Modal-specific Pre/Post-nets**: 处理不同模态的输入输出转换
   - Speech encoder pre-net: 6 层 1D 卷积 + GELU (复用 wav2vec 2.0 架构), 512→768 上采样 [§3.1]
   - Speech decoder pre-net: 2 层 Linear+ReLU,将 mel 上采样到 768 维; 拼接 x-vector 说话人嵌入后下采样回 decoder hidden size [§3.1]
   - Speech decoder post-net: Linear 预测 log mel-filterbank + 5 层 1D 卷积生成残差精修 + Linear 预测 stop token [§3.1]
   - Text encoder/decoder pre-nets: 共享 embedding,token→768 维 [§3.1]
   - Text decoder post-net: 映射 hidden state 到 token 概率分布 (softmax) [§3.1]

2. **Unified Encoder-Decoder**: 标准 Transformer [Vaswani et al., 2017]
   - Encoder: 12 层, d_model=768, d_ff=3072 [§3.1]
   - Decoder: 6 层, d_model=768, d_ff=3072 [§3.1]
   - 使用 SpeechT5 (English) / ArTST (Arabic) 预训练权重初始化 [§4.3]

3. **Task Fusion Module**: STTATTS 的核心创新 [§3.2]
   - 每个任务用 128 维可学习向量表示
   - 拼接到 encoder 输出后,通过 FC 层投影回 encoder embedding size
   - 输出作为 decoder 的输入

**关键设计**: 与 SpeechT5 的分离微调不同 (Figure 2 Right: 每个任务一份完整模型),STTATTS 所有任务共享同一 encoder-decoder backbone,仅通过 task fusion module 区分任务 (Figure 1),参数增量几乎可忽略 [论文原文, §3]。

### 关键设计选择

**为什么用 task fusion module 而不是其他方案?**

论文 §6.2 对比了多种替代方案:

1. **Y-decoder (modal-specific decoders)**: 共享 encoder + 独立 text/speech decoder
   - 参数量 211M vs STTATTS 155M [Table 7]
   - Arabic: ASR WER 10.37 vs 10.22, TTS CER 8.31 vs 6.22 → STTATTS 在 TTS 上显著更好 [Table 7]
   - English: ASR WER 5.67 vs 4.84, TTS CER 4.36 vs 3.18 → STTATTS 在两项上都更好 [Table 7]
   - [agent 解读] Y-decoder 虽然给每个任务独立解码能力,但无法实现 ASR-TTS 间的知识共享,尤其在解码器层面

2. **Multi-stage training (交替冻结)**: 先训练 ASR 再训练 TTS (或反之)
   - TTS-first: ASR WER 109.94 — 灾难性遗忘 [Table 7]
   - ASR-first: TTS CER 67.87 — 同样灾难性 [Table 7]
   - [论文原文] "ASR requires more computations to process the input in the encoder, while TTS works with discrete text input" [§6.2] — 这解释了为什么 ASR-first 对 encoder 更有利

3. **Task-specific adapters (Houlsby et al., 2019)**: 冻结 backbone + 64-dim adapter
   - ASR WER 150%, TTS loss ~0.8 (vs STTATTS ~0.4) → 失败 [§6.2]
   - [agent 解读] 64-dim adapter 容量太小,无法承载 ASR/TTS 的全部任务特异性知识; task fusion module 的设计更合理,因为它在 encoder-decoder 之间提供条件化,而非在内部层添加适配

**为什么 task fusion 放在 encoder 之后而不是之前?**

论文 §6.6 实验了将 task fusion 放在 encoder 之前:
- ASR CER 4% (尚可) 但 TTS CER 80% (失败) [§6.6]
- [agent 解读] 放在 encoder 前会干扰特征提取阶段,而放在 encoder 后可以在已提取的统一表征基础上进行任务条件化,这是一个关键的架构洞察

### 训练策略

**Multi-task loss**: L = L_asr + L_tts [Eq. in §3.3]
- L_asr = L_ce (decoder cross-entropy) + L_ctc (CTC loss from ESPNet) [§3.3]
- L_tts = L_1 (mel reconstruction) + L_bce (stop token) + L_attn (guided attention, 加速 TTS 收敛) [§3.3]
- 每步为每个任务计算 loss,按该步中每个任务的样本数归一化 [§3.3]
- 梯度累积 k 步后更新 [§3.3]

**Warm fine-tuning** [§4.3, §6.4]:
- **Arabic**: 先在 MGB2 数据集 (较大) 上微调 TTS,再联合训练 ASR+TTS
  - 原因: TTS 数据太少 (16h),需要先让模型学到基本的语音合成能力 [论文原文]
- **English (enl)**: 前 200K 步用 LS-500 (大量 ASR 数据),后续切换到 LS-100+360 (较少)
  - 原因: ASR 数据远大于 TTS 数据 (960h vs ~500h),直接联合训练会导致 TTS 性能下降 [论文原文]
- 效果: Arabic TTS CER 从 9.94 降到 6.22; English enl TTS CER 从 3.28 降到 2.10 [Table 8]

**数据不平衡发现** [§6.3]:
- TTS 对数据量更敏感: 100h TTS 数据产生的语音 MOS 仅 1.5 [§6.3]
- ASR 反而受益于更多 TTS 数据: ens 中增加 TTS 数据后 ASR WER 从 5.61 降到 4.84 [§6.3]
- [agent 解读] 这暗示联合训练中 TTS 任务的梯度信号对 encoder 表征有正向影响,相当于一种隐式的数据增强

**预训练权重的关键性** [§6.7]:
- 无预训练: TTS CER 10x 退化; ASR CER 仅 +1% [§6.7, Fig 4]
- [论文原文] "Pre-training is particularly crucial for TTS tasks" [§6.7]
- [agent 解读] TTS 从 text→mel 的映射空间更大,需要预训练提供良好初始化; ASR 从 speech→text 的映射相对更受约束

## 实验

| 指标 | STTATTS (ens, 155M) | STTATTS (enl, 155M) | SpeechT5 ASR (151M) | SpeechT5 TTS (145M) | VoxtLM_base (350M) | VoxtLM_large (1.3B) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ASR WER ↓ | 4.84 | 2.99 | 4.4 | - | 6.5 | 4.6 | LibriSpeech test-clean | [Table 3, 4, 5] |
| ASR CER ↓ | 1.63 | 0.90 | - | - | - | - | LibriSpeech test-clean | [Table 3] |
| TTS CER ↓ | 3.18 | 2.10 | - | 6.3 | 3.5 | 3.9 | LibriTTS test | [Table 3, 4, 5] |
| TTS Naturalness MOS ↑ | 3.36 | 3.00 | - | - | - | - | - | [Table 3] |
| TTS Intelligibility MOS ↑ | 4.00 | 4.38 | - | - | - | - | - | [Table 3] |
| TTS WV-MOS ↑ | 4.40 | 4.26 | - | - | - | - | - | [Table 3] |

**Arabic 结果** [Table 3, 4]:

| 指标 | STTATTS (155M) | ArTST ASR (151M) | ArTST TTS (145M) | Whisper-small (244M) | Whisper-large (1550M) |
| --- | --- | --- | --- | --- | --- |
| ASR WER ↓ | 10.22 | 7.59 | - | 32.2 | 23.4 |
| TTS CER ↓ | 6.22 | - | 9.61 | - | - |
| TTS WV-MOS ↑ | 3.69 | - | - | - | - |

**Voice Conversion** (enm + VC, 联合三任务) [Table 6]:
- VC CER 1.58, Speaker Similarity 0.99 — 无额外参数开销
- TTS WV-MOS 4.28 (对比 enm 无 VC 的 4.24, 基本持平) [Table 3, 6]
- ASR WER 3.59 (对比 enm 无 VC 的 3.47, 轻微退化) [Table 3, 6]

**关键发现**:
1. STTATTS 155M params 达到与 SpeechT5 单任务模型 (151M+145M = 296M) 相当的性能,参数减少 ~50% [Table 4]
2. 对比 VoxtLM_base (350M),STTATTS 在两项任务上都更优,参数仅为其 44% [Table 5]
3. 更多训练数据持续改善性能 (ens→enm→enl) [Table 3]
4. Arabic 低资源场景下 (仅 32h 数据),STTATTS ASR 显著优于 Whisper-small 和 Whisper-large [Table 4]

## 局限性

1. **TTS 自然度有限**: MOS 最高 3.36 (ens),距离人类语音仍有差距;作者自己也承认 "subjective MOS evaluation was rather difficult to conduct as most outputs were intelligible but somewhat noisy and unnatural" [§7]
2. **baseline 比较不充分**: 仅与 VoxtLM 做 joint-task 比较;其他模型 (VioLA, SpeechGPT) 代码不公开 [§7]
3. **未探索模型规模扩展**: 受限于需要从头预训练,未尝试更大模型 [§7]
4. **Arabic TTS 质量受限**: 仅 16h TTS 数据 + 缺少变音符号导致元音发音不准 [§7]
5. **mel spectrogram 路线的时代局限**: [agent 解读] 在 2024 年的 SpeechLM 浪潮中,基于 mel spectrogram + vocoder 的 pipeline 在自然度上已落后于基于离散 token 的端到端方法 (VALL-E, CosyVoice 等)
6. **无零样本评估**: 多说话人 TTS 使用已知说话人 x-vector,未评估零样本 voice cloning 能力

## 点评

**优势**:
- 问题定义清晰: 不做"万能大模型",而是聚焦"如何用最少参数同时做好 ASR+TTS"
- 实验设计扎实: 多语言 (English/Arabic)、多数据规模 (ens/enm/enl)、多架构变体 (Y-decoder/multi-stage/adapter/task-fusion-position) 的消融实验提供了丰富的工程洞察
- 开源: 代码和 checkpoints 公开

**不足**:
- task fusion module 的设计过于简单 (仅 128-dim vector + FC),虽然参数效率高,但可能限制了模型在更复杂场景下的表达能力
- TTS 评估缺少与当代 LLM-based TTS (VALL-E, CosyVoice 等) 的比较,定位不够清晰
- 论文声称是"truly cross-modal"但实际仍是 task-conditioned 的,输入输出模态在推理时是确定的

## 可复用的 idea

1. **Task fusion module 设计模式**: 用可学习 task vector + FC 在 encoder-decoder 间注入任务条件,几乎零参数开销。这个模式可推广到任何需要在共享 backbone 上多任务的场景,不限于 speech。
2. **Warm fine-tuning 平衡数据不平衡**: 先用大量数据训练一个任务的基础能力,再联合训练。这是一个通用的 multi-task 训练策略,适用于任何数据不平衡的 multi-task 场景。
3. **Task fusion position insight**: 放在 encoder 后 (feature space) 而非 encoder 前 (input space) 效果远好于后者,提供了多任务条件化应放在哪里的经验。
4. **联合训练的正则化效应**: ASR 受益于 TTS 数据的增加 (WER 从 5.61→4.84),暗示多任务训练作为隐式正则化的价值。

> [!review] 审阅 (auto, 2026-06-03, v1.1)
> **结论**: pass-with-fixes | 可复述 9 · 可信赖 8 · 可区分 9 · 可定位 9 · 不污染 9
> **问题**: 1 medium (VC 实验跨设置比较,已修正) + 2 low (速查出处/frontmatter 留空)
> 详见 `_review/STTATTS-review.yml`
