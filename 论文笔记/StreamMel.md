---
type: paper
tier: deep
title: "StreamMel: Real-Time Zero-shot Text-to-Speech via Interleaved Continuous Autoregressive Modeling"
arxiv_id: "2506.12570"
source: "Sources/StreamMel.pdf"
authors: [Hui Wang, Yifan Yang, Shujie Liu, Jinyu Li, Lingwei Meng, Yanqing Liu, Jiaming Zhou, Haoqin Sun, Yan Lu, Yong Qin]
year: 2025
venue: "arXiv"
tags: [TTS, zero-shot, streaming, autoregressive, continuous-token, mel-spectrogram, interleaving, low-latency, single-stage]
concepts: ["[[MelSpectrogram]]", "[[LLM-basedTTS]]", "[[VariationalAutoencoderforTTS]]", "[[CodecLanguageModel]]"]
models: ["[[模型库/MELLE|MELLE]]", "[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: StreamMel 处于 **连续值自回归 TTS** 路线的延伸位置。该路线的里程碑是 [[论文笔记/MELLE|MELLE]] [待确认],首次证明零样本 TTS 可以绕过向量量化,直接在连续 mel-spectrogram 上做自回归建模,通过 spectrogram flux loss + latent sampling module 解决连续 AR 的训练目标和采样两大挑战。StreamMel 在 MELLE 的基础上增加了**流式能力**,是该路线的流式化延伸。

**已有认知**:
- **[[MelSpectrogram]]** [待确认]: 80 维 log-mel 是 neural TTS 最广泛使用的中间声学特征,与离散 speech tokens 相比保留了连续信息但丢失相位,需要 vocoder 恢复波形。StreamMel 直接在此空间建模。
- **[[模型库/CosyVoice2|CosyVoice 2]]** ✓: 流式零样本 TTS 代表,采用 FSQ-SenseVoice tokenizer + LLM + chunk-aware flow matching 的**两阶段**流式方案。StreamMel 通过单阶段设计消除了两阶段间的延迟瓶颈。
- **[[LLM-basedTTS]]** ✓: 以 VALL-E 为代表的自回归 codec LM 范式。StreamMel 属于该范式中"连续值 AR"子路线 (与 MELLE/FELLE/LatentLM/CLEAR 同属),但进一步解决了流式推理问题。
- **[[CodecLanguageModel]]** [待确认]: 在离散 neural codec tokens 上做语言建模的范式。StreamMel 明确挑战了该范式"必须离散化"的假设,证明连续 mel + 交错序列可同时实现流式和高质量。
- **[[VariationalAutoencoderforTTS]]** [待确认]: StreamMel 的 latent decoder 直接继承 MELLE 的 latent sampling module (VAE reparameterization trick),为连续空间自回归提供概率采样机制。
- **[[Zero-shotSpeechSynthesis]]** ✓: 当前 SOTA 在 SEED-TTS-Eval 上 (CosyVoice 3, Qwen3-TTS),StreamMel 在 LibriSpeech 上的流式成绩已接近离线 SOTA。

**创新判断**: StreamMel 的核心创新在于**交错 (interleaving) 策略**,将文本 token 和连续 mel frame 按固定 n:m 比率交替排列,使自回归模型无需等待完整输入即可逐帧生成。这是连续值 AR 路线首次实现真正的流式合成。对比 IST-LM (同样采用交错但基于离散 token),StreamMel 证明连续表示在交错框架下仍然可行且更优。

> 检索命中: [[模型库/MELLE|MELLE]] [待确认], [[模型库/CosyVoice2|CosyVoice 2]]✓, [[MelSpectrogram]] [待确认], [[CodecLanguageModel]] [待确认], [[Zero-shotSpeechSynthesis]]✓, [[LLM-basedTTS]]✓ | 过滤: [[VariationalAutoencoderforTTS]] [待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个单阶段流式零样本 TTS,通过文本-mel 交错序列在连续 mel-spectrogram 空间做自回归建模,同时实现低延迟和高质量
> - **路线**: 文本 phonemes + speech prompt mel → 按 n:m 比率交错排列 → Transformer decoder 自回归预测 mel frames → latent sampling → mel spectrogram → vocoder → waveform
> - **指标**: Cross-sentence WER-H 2.76% / SIM-O 0.622 / MOS 4.14 / SMOS 4.27 (LibriSpeech test-clean); FPL-A 0.01s / FPL-L 0.04s (NVIDIA A100) [Table II-IV]
> - **可借鉴**: 文本-声学交错序列策略可直接复用于任何连续值 AR 模型的流式化改造; n:m ratio 作为延迟-质量 trade-off 的简单旋钮
> - **局限**: 仅在 LibriSpeech (英文朗读) 上验证; 需外部 vocoder; 未与大规模数据训练的系统 (Emilia/Libriheavy) 做公平对比

## 核心问题

现有零样本 TTS 面临两个相互矛盾的需求:

1. **离线系统** (MELLE, FELLE, VALL-E R) 质量高,但需要完整输入文本后才开始合成,无法用于实时场景 [§I]
2. **流式系统** (CosyVoice 2, IST-LM, SMLLE) 支持流式,但存在两个结构性缺陷:
   - **多阶段管线**: LM 预测语义表示 → 独立声学模型生成语音特征,后者需要缓冲累积的 token 才能产生稳定输出,引入延迟瓶颈 [§I]
   - **离散表示依赖**: 量化过程不可避免地降低表示保真度,增加系统复杂度 [§I]

StreamMel 的核心问题: **能否设计一个单阶段、基于连续表示的流式零样本 TTS 系统,同时消除多阶段延迟和量化损失?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

StreamMel 由三个组件构成 [§II-B, Fig 1]:

1. **Interleaved Input**: 文本 phoneme tokens 和 mel-spectrogram frames 按固定 n:m 比率交错排列成统一序列 z
2. **Mel-spectrogram Language Model**: Transformer decoder (12 层, 16 heads, FFN 4096) 在交错序列上做因果自回归
3. **Latent Decoder**: 从 decoder 输出预测 mel frame (latent sampling) + 停止概率 (stop prediction)

### 关键设计选择

#### 1. 文本-声学交错策略 (Interleaving)

给定 phoneme 序列 x = [x₀,...,x_{L-1}] 和 mel 序列 y = [y₀,...,y_{T-1}],按固定 n:m 比率交错:

z = [x₀,...,x_{n-1}, y₀,...,y_{m-1}, x_n,...,x_{2n-1}, y_m,...,y_{2m-1}, ...] [Eq. 1]

phoneme 用完后,剩余 mel frames 连续排列。每个 mel frame y_t 在序列 z 中的位置 t' 由 Eq. 2 计算,条件概率为 p(y_t | z_{<t'}) [Eq. 3]。

**为什么选交错而非 prefill?** 交错允许新文本 token 被增量追加,生成可以在不等待完整输入的情况下开始 [论文原文]。这使得模型可以精细控制语言上下文与声学上下文的平衡 [论文原文]。

**为什么 n:m = 1:4 最优?** 消融实验 [Table V] 显示:极端比率 3:1 和 1:5 显著降低精度和质量。1:4 的稀疏文本插入提供了足够的语义引导而不会破坏声学预测的稳定性 [论文原文]。[agent 解读] 这本质上反映了 phoneme 和 mel frame 的信息密度差异 --- 一个 phoneme 大约对应 4 个 mel frames (考虑到 ~10ms hop size 和 ~40-50ms 平均音素时长),因此 1:4 接近自然对齐比率。

**为什么文本 token 不预测?** 文本 token 从输入流直接复制,模型只预测 mel frames [§II-B.1]。[agent 解读] 这避免了文本预测误差的累积,并且在流式场景中文本是已知输入,不需要预测。

#### 2. 连续 mel 而非离散 token

**为什么选连续表示?** 离散表示 (neural codec tokens) 需要量化过程,不可避免地降低表示保真度并增加系统复杂度 [§I]。基于连续表示的 TTS 方法已展现出更优的性能 [§I, refs 16-17] [论文原文]。计算机视觉领域也观察到类似局限 [§I, ref 18] [论文原文]。

#### 3. Latent Sampling (继承自 MELLE)

在每个解码步 t',如果当前位置对应 mel frame (非 fill token),decoder 输出 e_{t'} 用于预测高斯分布参数 μ_{t'} 和 log σ²_{t'},通过 reparameterization trick 采样 z_{t'} = μ_{t'} + σ_{t'} ⊙ ε (ε ~ N(0,I)),再经 MLP 投影到 mel 空间得到 ŷ_t [§II-B.3]。

**为什么需要 latent sampling?** [agent 解读] 连续空间的自回归模型如果直接做回归,会陷入 mode averaging (过度平滑)。VAE 的 reparameterization 引入随机性,使模型能生成多样化的输出,同时 KL 正则化防止方差爆炸。这个设计直接来自 MELLE [ref 11]。

#### 4. Fill Token

交错序列中,文本 token 位置不产生 mel 输出,这些位置标记为 fill token,在 loss 计算中被忽略 [§II-B.1, Fig 1]。[agent 解读] 这个设计允许模型在统一序列上做因果推理,同时避免在文本位置产生无意义的声学预测。

### 训练策略

**损失函数** [§II-C, Eq. 4]:

L = αL_reg + λL_KL + βL_flux + γL_stop

- **L_reg** (α=2): L1 + L2 mel 重建损失,确保频谱精确重建
- **L_KL** (λ=0.05): KL 散度损失,正则化 latent 分布向标准高斯 prior
- **L_flux** (β=1): 频谱 flux 损失 (来自 MELLE [ref 11]),鼓励帧间动态变化,惩罚静态重复帧
- **L_stop** (γ=0.5): 二元交叉熵,指导停止预测

训练数据: LibriSpeech ~960 小时英文朗读语音 [§III-A]。
固定交错比率 n:m = 1:4 用于训练和评估 [§III-A]。

## 实验

### 评估设置

两种场景 [§III-B]:
- **Continuation**: 给定 transcript + 前 3 秒语音,续说
- **Cross-sentence**: 给定参考语音 + 新文本,用参考音色合成新文本

### 主要结果

**Continuation (Table I)**:

| 指标 | StreamMel (streaming) | IST-LM (streaming) | MELLE-L (offline) | FELLE (offline) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER-C | 1.65 | - | 1.53 | 1.53 | LibriSpeech test-clean | [Table I] |
| WER-H | 2.41 | 3.60 | 2.22 | 2.27 | LibriSpeech test-clean | [Table I] |
| SIM-R | 0.534 | - | 0.517 | 0.539 | LibriSpeech test-clean | [Table I] |
| SIM-O | 0.504 | - | 0.480 | 0.513 | LibriSpeech test-clean | [Table I] |

**Cross-sentence objective (Table III)**:

| 指标 | StreamMel (streaming) | SMLLE (streaming) | MELLE-L (offline) | MELLE (offline, Libriheavy) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER-C | 2.10 | 5.14 | 2.21 | 1.47 | LibriSpeech test-clean | [Table III] |
| WER-H | 2.76 | 6.37 | 2.80 | 2.10 | LibriSpeech test-clean | [Table III] |
| SIM-R | 0.656 | 0.516 | 0.633 | 0.664 | LibriSpeech test-clean | [Table III] |
| SIM-O | 0.622 | 0.489 | 0.591 | 0.625 | LibriSpeech test-clean | [Table III] |

**Cross-sentence subjective (Table II)**:

| 指标 | StreamMel | IST-LM | SMLLE | GT | 出处 |
| --- | --- | --- | --- | --- | --- |
| MOS | 4.14±0.16 | 4.12±0.16 | 3.10±0.19 | 4.32±0.15 | [Table II] |
| SMOS | 4.27±0.15 | 4.22±0.14 | 3.36±0.17 | 4.41±0.14 | [Table II] |

**延迟 (Table IV)**:

| 指标 | StreamMel | SyncSpeech | CosyVoice* | F5-TTS* | 出处 |
| --- | --- | --- | --- | --- | --- |
| FPL-A (s) | **0.01** | 0.06 | 0.22 | 1.27 | [Table IV] |
| FPL-L (s) | **0.04** | 0.11 | 0.35 | 1.98 | [Table IV] |

### 消融实验

**交错比率 n:m (Table V)**:
- 1:4 最优: WER-W 2.77, SIM-O 0.622
- 3:1 和 1:5 均显著退化 (WER-W 分别 4.26 和 8.98)
- 非流式 (不交错) 表现最佳 (WER-W 2.37) 但不支持流式

**Reduction factor r (Table VI, n:m=1:1)**:
- r 增大 → RTF 显著降低 (r=1: 0.700 → r=4: 0.179) 但 SIM-O 持续下降 (0.605 → 0.454)
- WER 在 r 增大时反而下降,表明多帧预测可缓解误差累积 [论文原文]

**Sample times (Fig 2)**:
- 采样次数增加 → WER 持续下降 + similarity 持续上升
- 增益在 4-6 次后逐渐饱和

## 局限性

1. **数据规模有限**: 仅在 LibriSpeech (~960h, 英文朗读) 上训练和评估,未验证多语言、情感表达、对话场景 [agent 解读]
2. **非端到端**: 仍需外部 vocoder (HiFi-GAN) 从 mel 合成波形,未实现真正的端到端流式 waveform 生成 [agent 解读]
3. **对比不够公平**: Table IV 中 CosyVoice/F5-TTS 等模型在大规模数据 (LibriTTS/Emilia) 上训练,StreamMel 在 LibriSpeech 上训练,FPL 对比受模型规模和训练数据影响 [agent 解读]
4. **固定交错比率**: n:m 全局固定 (1:4),不同说话速率或语言可能需要自适应比率 [agent 解读]
5. **RTF 未完整报告**: Table IV 仅报告了 FPL,未报告完整的 RTF 和端到端推理时延; Table VI 中 RTF 仅在 n:m=1:1 条件下报告 [agent 解读]
6. **未与同期连续值流式方法对比**: 如 VoXtream (continuous AR streaming) 等同属连续值路线的流式系统未纳入对比 [agent 解读]

## 点评

**优势**:
- **概念简洁且有效**: 单阶段、连续表示、交错序列 --- 三个设计选择的组合非常干净,每一个都有明确的动机和消融支撑
- **延迟优势显著**: FPL-A 0.01s 是目前已报告的最低首包延迟之一,交错设计使得只需 n 次 TTS 前向传播即可输出第一个音频包
- **质量接近离线**: 流式条件下 cross-sentence WER-H 2.76 接近离线 MELLE-L 的 2.80,SIM-O 0.622 甚至超过 MELLE-L 的 0.591

**不足**:
- **泛化性存疑**: 所有结果限于 LibriSpeech 英文朗读场景。对比 CosyVoice 系列和 IST-LM 等在多语言/多场景下验证的系统,StreamMel 的实用性证据不足
- **缺少大规模验证**: 当前主流零样本 TTS 训练数据已进入万小时级别 (Emilia 101kh),StreamMel 在 960h 上的表现是否能随数据 scale 而提升,仍是问号
- **消融设计有遗憾**: Table VI 的 reduction factor 消融在 n:m=1:1 而非最优的 1:4 上进行,降低了实践参考价值

## 可复用的 idea

1. **文本-声学交错序列设计**: 将 phoneme 和 acoustic frame 按固定比率交错,可直接应用于任何需要流式化的连续值自回归模型 (如 LatentLM、CLEAR 的流式改造)
2. **n:m ratio 作为延迟-质量旋钮**: 提供了一个简单直觉的超参数来控制流式 TTS 的延迟-质量 trade-off,无需修改模型架构
3. **Fill token 机制**: 在统一序列中标记"无预测目标"的位置,使异构模态 (文本/声学) 可以共享一个因果自回归框架
4. **Sample times 作为推理质量旋钮**: 多次 latent sampling 再选最优,以额外推理成本换取质量提升,适用于对质量要求高但不急于实时的场景

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes
> - [x] 可复述: 方法节包含因果解释, 设计选择有 WHY
> - [x] 可信赖: claim 标注覆盖率 ~90%, 指标名正确
> - [x] 可区分: 因果解释来源标注完整, 无推断写成断言
> - [x] 可定位: KB 背景谱系清晰, frontmatter 齐全
> - [x] 不污染: 反向更新均为追加, 无 factual error 风险
> **Issues**: 1 medium (Table I 数据已修正) + 1 low (datasets 为空)
> 详见 `_review/StreamMel-review.yml`
