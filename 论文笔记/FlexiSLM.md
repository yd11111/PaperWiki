---
type: paper
tier: deep
title: "FlexiSLM: A Dynamic and Controllable Frame Rate Spoken Language Model"
arxiv_id: "2606.31247"
source: "Sources/FlexiSLM.pdf"
authors: [Jiaqi Li, Chaoren Wang, Xiaohai Tian, Mingjie Chen, Xinyu Liang, Xu Li, Yufan Lin, Junwen Qiu, Jun Zhang, Lu Lu, Haizhou Li, Zhizheng Wu]
year: 2026
venue: "arXiv preprint"
tags: [speech-LM, dynamic-frame-rate, frame-rate-control, thinker-talker, speech-to-speech, inference-efficiency, FlexiCodec, multi-task-SLM]
concepts: ["[[SpeechLanguageModel]]", "[[SpeechTokenizer]]", "[[SemanticvsAcousticTokens]]", "[[ConditionalFlowMatching]]", "[[FiniteScalarQuantization]]", "[[TokenRateandBitrateTrade-offs]]"]
models: ["[[论文笔记/FlexiCodec|FlexiCodec]]", "[[模型库/HuBERT|HuBERT]]", "[[模型库/SenseVoice|SenseVoice]]", "[[模型库/EnCodec|EnCodec]]"]
tasks: []
datasets: ["[[数据集/LibriSpeech|LibriSpeech]]", "[[数据集/Emilia|Emilia]]"]
kb_context_sources: 5
status: draft
created: 2026-07-01
updated: 2026-07-01
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 2 个待确认实体页: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓, [[FiniteScalarQuantization]][待确认], [[TokenRateandBitrateTrade-offs]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[FiniteScalarQuantization]](pending-review), [[TokenRateandBitrateTrade-offs]](pending-review) | 未命中但可能相关: 无

**[[SpeechLanguageModel]]**: FlexiSLM 属于 SLM 的 thinker-talker 范式 (与 Qwen2.5-Omni 同类),即 decoder-only LLM backbone (Thinker) 处理文本推理,专用 Transformer (Talker) 预测语音 token。现有 SLM 均使用固定帧率表征 (Kimi-Audio 12.5Hz, Qwen2.5-Omni 25Hz),FlexiSLM 是首个将动态帧率引入 SLM 框架的工作。

**[[SpeechTokenizer]]**: FlexiSLM 使用 FlexiCodec 的 FSQ semantic tokens 作为 Talker 的预测目标。FlexiCodec 通过 SenseVoice ASR 特征提取 12.5Hz 语义表征,经 frame merging + FSQ 量化产生动态帧率 discrete tokens。这与 CosyVoice 系列的监督式 semantic token 路线一脉相承,但新增了帧级压缩维度。

**[[ConditionalFlowMatching]]**: FlexiSLM 的音频解码器 (Audio Decoder) 使用 VoiceBox 风格的 NAR flow matching 模型 (363M 参数) 从 FSQ semantic tokens 生成 mel spectrogram,再经 Vocos vocoder 合成 24kHz 波形。该 flow matching 模型在 12.5Hz 固定帧率序列上工作 (dynamic tokens 通过 repeat 插值恢复到 12.5Hz)。

**[[TokenRateandBitrateTrade-offs]]** [待确认]: FlexiSLM 的核心主张是通过动态帧率实现 token rate-quality trade-off 的帕累托改善。概念页记录的公式 $\text{Token Rate} = \text{Frame Rate} \times N_q$ 在 FlexiSLM 中简化为 Frame Rate (单码本 FSQ),帧率从 12.5Hz 降至 4.0Hz 直接等比降低 token 数量。

**[[FiniteScalarQuantization]]** [待确认]: FlexiCodec 使用 FSQ 量化 ASR 语义特征,每帧产生一个离散 token + 一个帧长度属性。FlexiSLM 的 Talker 通过两个并行 LM head 分别预测 FSQ code 和 frame length。

## 速查

> [!summary] 速查
> - **一句话**: 首个支持动态可控帧率的 Spoken Language Model,通过 frame merging + 直接帧率条件化实现 4.0-12.5Hz 连续可调,在 12.5Hz 下超越所有 7B SLM baseline
> - **路线**: Speech → Qwen2.5-Omni Audio Encoder (25Hz) → Frame Merging (≤12.5Hz) → Qwen2.5-7B Thinker (LoRA/full) → Talker Transformer (630M, parallel FSQ+len heads) → FlexiCodec FSQ tokens → Flow Matching decoder → Vocos → 24kHz waveform
> - **指标**: 12.5Hz Overall s2s 67.2 (vs Qwen2.5-Omni 63.3, Kimi-Audio 57.2); 6.25Hz RTF 0.59 (vs 12.5Hz RTF 1.17, 半速推理); TTS WER 2.14% @12.5Hz
> - **可借鉴**: (1) 直接帧率条件化 (sinusoidal encoding of target FR) 替代 threshold 控制,误差 <0.1Hz; (2) Talker-to-Thinker 反向连接让 Thinker 感知已说内容; (3) 三阶段训练 (Talker pretrain → multi-task LoRA → full finetune+T2T link)
> - **局限**: 非流式推理; 无 RLHF/DPO 后训练; 训练数据不含 reasoning/multi-turn; 4.0Hz 质量下降明显 (s2s 56.5); 代码和数据尚未开源

## 核心问题

FlexiSLM 要解决两个关键问题:

1. **语音信息密度不均匀与固定帧率的矛盾**: 现有 SLM 以固定帧率 (12.5-50Hz) 表征语音,对静音段和信息稀疏段浪费大量计算。语音的时变信息密度天然适合动态帧率编码,但此前动态帧率 codec (FlexiCodec) 仅在 0.3B TTS 管线中验证,未扩展到端到端 SLM [§1]。

2. **推理时无法做 quality-speed trade-off**: 固定帧率 SLM 部署后帧率不可调,无法在不同设备/网络/预算下灵活选择质量与速度的平衡点。FlexiSLM 要让单一模型覆盖 4.0-12.5Hz 全范围,无需重训 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

FlexiSLM 采用 thinker-talker 架构 (如 [Fig 2] 所示),包含五个核心组件:

1. **Audio Encoder**: 预训练 Qwen2.5-Omni audio encoder,提取 25Hz 连续语音特征 [§3.1]
2. **Frame Merging Module**: 出现两次 — (a) 输入侧将 25Hz 特征压缩为 ≤12.5Hz 动态序列; (b) FlexiCodec tokenizer 内部将 12.5Hz ASR 特征压缩后量化 [§3.1, §3.2]
3. **LLM Backbone (Thinker)**: Qwen2.5-7B-Instruct 初始化,处理文本推理 [§3.1]
4. **Talker Transformer**: 630M 参数,从 Thinker hidden states 解码 FlexiCodec 的动态帧率 FSQ tokens + frame lengths [§3.1]
5. **Audio Decoder**: 冻结的 NAR flow-matching Transformer (363M) + Vocos vocoder,从 FSQ tokens 合成 24kHz 语音 [§3.1]

### 关键设计选择

#### 1. Frame Merging 机制 [§3.2]

Frame Merging 是 FlexiSLM 的核心创新之一。给定固定帧率特征序列 $x_1, x_2, \ldots, x_T$,计算相邻帧的余弦相似度:

$$s_t = \frac{x_t \cdot x_{t+1}}{\|x_t\| \|x_{t+1}\|}, \quad t = 1, \ldots, T-1$$

若 $s_t$ 超过合并阈值 $\tau$,则将 $x_t$ 和 $x_{t+1}$ 分组并取均值。贪心地从左到右扫描,连续高相似度帧合并为单个平均表征 $\bar{x}_k$,附带帧长属性 $l_k$ 记录原始帧数 [§3.2]。

合并后的特征与原始特征交错排列 (interleave),送入轻量 Transformer (local attention, 20M 参数) 进行重新对齐,最终取对应平均特征位置的表征作为输出 [§3.2]。

**[agent 解读]**: 交错+local attention 的设计目的是让合并后的表征"看到"原始帧上下文,补偿均值操作的信息损失。这比直接用均值表征更鲁棒。

#### 2. 直接帧率控制 (Direct Frame Rate Control) [§3.3]

这是 FlexiSLM 相对于 FlexiCodec 的关键改进。

**Baseline: 合并阈值控制 (Threshold Control)**

FlexiCodec-TTS 通过调节阈值 $\tau$ 间接控制帧率:高 $\tau$ → 少合并 → 高帧率;低 $\tau$ → 多合并 → 低帧率。但 [论文原文] 指出三个局限 [§3.3]: (1) 同一 $\tau$ 在不同话语上产生的帧率方差大 (如 $\tau$=0.90 对应 3.91-10.74Hz, $\sigma$≈0.70); (2) 一对多映射增加建模歧义; (3) 对用户不直观。

**FlexiSLM: 直接帧率条件化**

FlexiSLM 直接将目标帧率 $r$ 作为条件信号输入 Talker 和 Frame Merging Module。帧率 $r$ 通过正弦位置编码:

$$\text{PE}(r) = [\sin(r\omega_1), \cos(r\omega_1), \ldots, \sin(r\omega_d), \cos(r\omega_d)]$$

其中 $\omega_i = 10000^{-2i/d}$ 为频率基 [§3.3]。该编码作为 Talker 输入的一部分,序列中每个位置接收相同的帧率条件。

训练时随机采样合并阈值,计算每条话语的实际平均帧率,将该经验帧率作为条件信号。推理时用户直接指定目标帧率 [§3.3]。

**[agent 解读]**: 这一设计将"用户意图"(目标帧率) 与"执行机制"(frame merging) 解耦。训练时 model 学到的是"帧率 r → 对应质量的语音",而非"阈值 τ → 不确定帧率 → 语音",降低了建模歧义。

#### 3. Talker 输入-输出结构 [§3.1, Fig 3]

Talker 的输入在每个位置拼接四个信号:
- Thinker LLM 最后一层 hidden state
- 目标帧率的正弦编码
- 前一步已发出的 FSQ token embedding
- 前一步已发出的 frame length token embedding

输出两个并行流: FlexiCodec FSQ codes + frame lengths,通过两个独立 LM head 并行预测 [§3.1]。

**Token delay**: FSQ token 流相对文本流延迟 5 个 token (防止语音先于对应文本),frame length 再延迟 1 位 (先知道 speech token 再预测其时长) [§3.1, Fig 3]。

#### 4. Talker-to-Thinker 连接 [§3.1]

标准 thinker-talker 架构是单向的 (Thinker → Talker)。FlexiSLM 增加了可选的反向连接: 将 Talker 已发出的 speech token embedding 反馈到 Thinker LLM 的下一步输入 [§3.1]。

**[论文原文]**: 这让 Thinker 显式感知"已经说了什么",提供声学上下文连贯性 [§3.1]。

**[agent 解读]**: 反向连接在 Stage 3 (full fine-tuning) 才启用,Stage 1-2 禁用。消融实验 (Table 10) 显示在 Stage 2 (LoRA) 提前启用会导致严重退化 (ASR WER 2.92 → 7.75, s2t AVG 68.7 → 60.5),论文归因于 LoRA 容量不足以吸收反馈信号造成的"不稳定反馈循环" [§E]。

### 训练策略

三阶段训练 (24 A100 80G GPUs) [§3.4]:

**Stage 1: Talker Pre-training** — 冻结 LLM,仅训练 Talker。纯 TTS 数据 (Emilia-EN + MLS, ~100K hours)。Talker-to-Thinker 连接禁用 [§3.4]。

**Stage 2: Multi-task LoRA Fine-tuning** — 激活 Frame Merging Module + LoRA Thinker + Talker。混合任务 (Table 2): FlexiSLM-Data (9.9K hours s2s dialogue, 3x oversampling) + TTS (Emilia+MLS) + ASR (LibriSpeech+MLS) + audio understanding (LLaSO-instruct)。LoRA rank=32, α=64 [§3.4, §C]。

**Stage 3: Full Fine-Tuning** — LoRA 合并回 LLM,全参数训练。启用 Talker-to-Thinker 连接 [§3.4]。

**FlexiSLM-Data 构建** [Appendix A]: 从 Qwen3-Omni-30B 蒸馏 s2s 对话数据。文本 prompt (10 个公开数据集, Table 7) → Qwen3-Omni 生成文本回复 → Qwen3-TTS + Fish-Audio 合成语音 → 质量过滤 (格式/正确性/ASR WER<20%) → 最终 1.4M 样本, 9.9K hours [Appendix A]。

## 关键公式

**Frame merging 判断条件**:
$$s_t = \frac{x_t \cdot x_{t+1}}{\|x_t\| \|x_{t+1}\|} \quad \text{if } s_t > \tau \Rightarrow \text{merge}$$

**平均帧率定义**:
$$\text{Average Frame Rate} = \frac{\text{number of frames after merging}}{\text{Total Audio duration in seconds}}$$

**直接帧率条件编码** (sinusoidal positional encoding):
$$\text{PE}(r) = [\sin(r\omega_1), \cos(r\omega_1), \ldots, \sin(r\omega_d), \cos(r\omega_d)], \quad \omega_i = 10000^{-2i/d}$$

**训练损失** [§3.5]:
$$\mathcal{L} = \lambda_{\text{text}} \mathcal{L}_{\text{text}} + \lambda_{\text{speech}} \mathcal{L}_{\text{speech}} + \lambda_{\text{speech\_len}} \mathcal{L}_{\text{speech\_len}}$$

其中 $\lambda_{\text{text}}=2$, $\lambda_{\text{speech}}=\lambda_{\text{speech\_len}}=1$。$\mathcal{L}_{\text{text}}$, $\mathcal{L}_{\text{speech}}$, $\mathcal{L}_{\text{speech\_len}}$ 分别为文本 token、FlexiCodec FSQ code 和帧长度的 cross-entropy loss。非语音序列时 $\mathcal{L}_{\text{speech}}=\mathcal{L}_{\text{speech\_len}}=0$ [§3.5]。

**帧率随机采样** (训练时): 输入帧率目标 $r \sim U(4, 12.5)$ Hz, FlexiCodec 合并阈值 $\tau \sim U(0.85, 1.0)$ [§3.5]。

## 实验

### 主实验: SLM 综合评测 (Table 3, Kimi-Audio-Evalkit)

| 指标 | FlexiSLM 12.5/12.5 | FlexiSLM 6.25/6.25 | Qwen2.5-Omni 25/50 | Kimi-Audio 12.5/12.5 | Mimo-Audio 6.25/6.25 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Overall s2t | 72.4 | 70.2 | 66.7 | 69.7 | 70.6 | [Table 3] |
| Overall s2s | 67.2 | 64.3 | 63.3 | 57.2 | 59.0 | [Table 3] |
| ASR clean WER | 1.98 | 2.55 | 2.38 | 1.80 | - | [Table 3] |
| ASR other WER | 5.79 | 6.37 | 4.21 | 2.45 | - | [Table 3] |

FlexiSLM 12.5Hz 在 s2s Overall 上超越所有 7B baseline 3.9-10.0 分,在 s2t 上超越 Qwen2.5-Omni 5.7 分 [Table 3]。即使降至 6.25Hz,s2s 仍高于所有 7B baseline [§4.2]。

### 帧率可控性 (Table 4)

| 控制方式 | 目标 | 实际均值 (Llama Q) | σ | 出处 |
| --- | --- | --- | --- | --- |
| 阈值 τ=0.90 | ~8Hz | 8.34 | 0.70 | [Table 4] |
| 直接 FR=6.25Hz | 6.25 | 6.25 | 0.05 | [Table 4] |
| 直接 FR=4.0Hz | 4.0 | 3.99 | 0.05 | [Table 4] |

直接帧率控制误差 <0.1Hz,方差降低一个数量级 (σ: 0.70 → 0.05) [Table 4, §4.3.1]。

### 推理效率 (Table 5)

| 配置 (In/Out Hz) | RTF | TFLOPs | 出处 |
| --- | --- | --- | --- |
| FlexiSLM 12.5/12.5 | 1.17 | 4.57 | [Table 5] |
| FlexiSLM 12.5/6.25 | 0.59 | 3.41 | [Table 5] |
| FlexiSLM 6.25/6.25 | 0.57 | 2.73 | [Table 5] |
| Qwen2.5-Omni 25/50 | 1.57 | 5.26 | [Table 5] |

输出帧率是推理加速的主驱动: 12.5→6.25Hz 输出, RTF 近半 (1.17→0.59) [§4.3.2]。FlexiSLM 12.5Hz 比 Qwen2.5-Omni 快 1.3x; 输出降至 6.25Hz 后快 2.7x [§4.3.2]。

### 消融实验 (Table 6, 8 GPU 训练, Stage 2)

| 消融变体 | s2t/s2s AVG | ASR clean/other | TTS WER | 出处 |
| --- | --- | --- | --- | --- |
| Baseline (动态 in+out, 直接 FR 控制) | 68.7/63.0 | 2.92/7.20 | 3.11 | [Table 6] |
| w/o 动态输出 (uniform merging) | 67.7/61.0 | 3.14/7.67 | 4.95 (+59%) | [Table 6] |
| w/o 动态输入 (uniform merging) | 67.5/62.9 | 2.97/7.97 | 3.12 | [Table 6] |
| w/ 阈值控制替代直接 FR 控制 | 68.2/61.7 | 2.96/7.24 | 3.53 | [Table 6] |

- 去除动态输出帧率: TTS WER 恶化 59% (3.11→4.95), s2s AVG 下降 2.0 分 [§4.4]
- 去除动态输入帧率: s2t AVG 下降 1.2 分,ASR other WER 恶化 (7.20→7.97) [§4.4]
- 阈值控制替代直接 FR 控制: s2s 下降 1.3 分,TTS WER 上升 (3.11→3.53) [§4.4]

### 语音生成质量 (Table 8, E2TTS-Evalkit)

| 模型 | TTS WER | Dialog WER | 出处 |
| --- | --- | --- | --- |
| FlexiSLM 12.5Hz | 2.14 | 4.52 | [Table 8] |
| FlexiSLM 6.25Hz | 2.87 | 5.83 | [Table 8] |
| Qwen2.5-Omni | 3.18 | 6.33 | [Table 8] |
| Qwen3-Omni | 3.34 | 4.32 | [Table 8] |

FlexiSLM 12.5Hz TTS WER (2.14%) 优于 Qwen2.5-Omni (3.18%) 和 Qwen3-Omni (3.34%) [Table 8, §D]。

### 音频理解 (Table 11, LLaSO-Eval)

FlexiSLM 12.5Hz 平均准确率 65.8%,超过 Gemini 2.5-Pro (48.3%)、LLaSO-3B (58.3%)。值得注意的是,音频理解在激进压缩 (4.0Hz) 下仍保持 64.1%,与 12.5Hz 差异极小,因为这些是序列级分类任务,全局声学统计足够 [§F, Table 11]。

## 局限性

1. **非流式**: 当前不支持 streaming 推理,音频解码器需适配因果操作才能支持实时对话 [Limitations]
2. **无后训练优化**: 未使用 RLHF/DPO,响应质量和对齐性有提升空间 [Limitations]
3. **训练数据覆盖不足**: 不含 reasoning 任务、多轮对话、选择题,泛化受限 [Limitations]
4. **低帧率质量仍有提升空间**: 4.0Hz s2s 56.5,较 6.25Hz 的 64.3 下降 7.8 分,ASR WER 从 2.55/6.37 恶化到 4.47/9.53 [Table 3]
5. **ASR 性能不及最优**: LibriSpeech clean 1.98 / other 5.79,落后于 Kimi-Audio (1.80/2.45),可能与 audio encoder 选择和训练数据有关 [Table 3]
6. **推理加速受限于 Thinker 序列长度**: FlexiSLM 的 Thinker 处理完整 speech-length 序列 (为 Talker-to-Thinker 双向信息流服务),削弱了帧率降低带来的部分速度增益 [§4.3.2]

## 点评

**优点**:
- **系统级创新**: 首次将动态帧率从 codec 层面扩展到端到端 SLM,验证了在理解和生成双侧同时使用动态表征的可行性
- **实用性强的帧率控制**: sinusoidal encoding + 直接帧率条件化是一个简洁优雅的设计,误差 <0.1Hz,让单一模型成为连续可调的"帧率旋钮"
- **充分的实验验证**: 消融设计到位 — 分别验证了输入/输出动态帧率和直接/阈值控制的贡献,Appendix E 进一步消融了 encoder/backbone/merging Transformer 等组件
- **良好的工程性**: 三阶段训练策略合理 (先学说话 → 学多任务 → 打通双向连接),FlexiSLM-Data 构建管线具有复现价值

**不足**:
- **Thinker 序列长度问题**: 为保持 Talker-to-Thinker 双向连接,Thinker 需要处理 speech-length 序列,这抵消了部分帧率降低的计算收益。与 Qwen2.5-Omni (Thinker 仅处理到文本生成结束) 的架构差异使得 RTF 对比不完全公平
- **FlexiCodec 的局限被继承**: 作为 tokenizer, FlexiCodec 的 FSQ 单码本方案在声学细节上依赖 flow matching 解码器补偿,FlexiSLM 论文未评估 speaker similarity 或 prosody 保真度
- **缺乏 MOS 等主观评估**: 所有评估基于 GPT-5.5 judge 或 WER,无人类 MOS 评估,难以全面评判语音质量
- **与 patching 策略的对比缺失**: Fun-Audio-Chat (5Hz via 5x patching) 和 Mimo-Audio (6.25Hz via 4x patching) 使用 grouping/patching 降低 LLM 输入帧率,是互补的降帧策略。论文未消融 frame merging + patching 的组合效果

## 可复用的 idea

1. **直接帧率条件化 (sinusoidal PE of target rate)**: 适用于任何需要连续控制生成粒度的场景 (不限于帧率,可推广到 resolution/quality 等)
2. **Frame Merging + interleave + local attention re-alignment**: 简洁的自适应压缩方案,可用于任何固定帧率 encoder 的后处理
3. **三阶段 thinker-talker 训练**: Stage 1 冻结 LLM 只练 Talker → Stage 2 LoRA 多任务 → Stage 3 full finetune + 反向连接。特别是"反向连接需要 full-parameter capacity"的发现 (Table 10) 有迁移价值
4. **蒸馏 s2s 对话数据管线**: 用 30B SLM 生成文本回复 + TTS 合成语音 + 质量过滤,低成本构建大规模 s2s 训练数据

---

> [!review] 审阅: pass-with-fixes (2026-07-01)
> 报告: `_review/FlexiSLM-review.yml`
> 3 low issues: (1) frontmatter tasks 为空; (2) 点评中 Qwen2.5-Omni Thinker 行为描述缺 [agent 解读] 标注; (3) 表头命名可更精确。无 high/medium issue,不阻塞反向更新。

> 检索命中: [[SpeechLanguageModel]]✓, [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[FiniteScalarQuantization]](pending-review), [[TokenRateandBitrateTrade-offs]](pending-review) | 未命中但可能相关: 无
