---
type: paper
tier: deep
title: "SMLLE: Zero-Shot Streaming Text to Speech Synthesis with Transducer and Auto-Regressive Modeling"
arxiv_id: "2505.19669"
source: "Sources/SMLLE.pdf"
authors: [Haiyang Sun, Shujie Hu, Shujie Liu, Lingwei Meng, Hui Wang, Bing Han, Yifan Yang, Yanqing Liu, Sheng Zhao, Yan Lu, Yanmin Qian]
year: 2025
venue: "arXiv"
tags: [TTS, streaming, zero-shot, transducer, autoregressive, mel-spectrogram, semantic-token, frame-by-frame, low-latency]
concepts: ["[[SemanticvsAcousticTokens]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[MelSpectrogram]]", "[[DurationPredictor]]", "[[Non-autoregressiveTTS]]", "[[SpeakerEmbedding]]"]
models: ["[[模型库/MELLE|MELLE]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: ["LibriSpeech", "LibriTTS"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: SMLLE 处于 streaming TTS 与 LLM-based zero-shot TTS 两条路线的交汇点。当前 LLM-based TTS (VALL-E 系列, CosyVoice 系列) 在零样本质量上已接近人类水平,但它们是句子级系统——必须拿到整句文本才开始生成,延迟高。而 streaming TTS 方案 (如 LiveSpeech 2 的 chunk-level 生成) 要么依赖 lookahead 机制引入额外延迟,要么在逐帧模式下质量严重退化 (如 LiveSpeech2-limit WER 40.7%)。SMLLE 的定位是首个在逐帧 (frame-by-frame) 模式下实现零样本质量的 streaming TTS。
>
> **已有认知**:
> - [[SemanticvsAcousticTokens]] (confirmed): SMLLE 使用 SpeechTokenizer 的第一层 codec 作为 semantic tokens,属于 mixed tokenizer 路线中的语义层。Semantic tokens 与文本对齐好但缺声学细节;SMLLE 的 Transducer 将文本转为 semantic tokens,再由 AR 模型从 semantic tokens 恢复 mel spectrogram,形成两阶段解耦。
> - [[LLM-basedTTS]] (confirmed): SMLLE 的 AR 阶段直接继承 MELLE 的连续 mel-spectrogram 自回归框架,包括 latent sampling module 和 spectrogram flux loss。SMLLE 与 VALL-E 系列的竞争关系不在于取代 LLM-based TTS 的质量,而在于为其增加 streaming 能力。
> - [[SpeechTokenizer]] (confirmed): SMLLE 使用 SpeechTokenizer (Zhang et al., ICLR 2024) 的第一层 RVQ codes 作为 semantic tokens。SpeechTokenizer 是 mixed tokenizer——第一层蒸馏 HuBERT 语义,后续层编码声学残差。SMLLE 仅使用第一层,因此操作的是纯语义表示。
> - [[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]] (confirmed): SMLLE 在 LibriSpeech test-clean (unseen speakers) 上评估零样本能力,SIM 0.516,与 VALL-E (0.580) 可比。当前 SOTA (CosyVoice 3, IndexTTS2) 已远超此水平。
> - [[MelSpectrogram]] [待确认]: SMLLE 的 AR 模型输出 80-dim log-magnitude mel spectrogram,通过 BigVGAN-V2 vocoder 合成波形。与离散 codec token 路线不同,SMLLE 延续了 MELLE 的"连续 mel 预测"范式。
> - [[DurationPredictor]] [待确认]: SMLLE 完全避免了显式 duration predictor。Transducer 的对齐路径隐式提供 duration 信息 (通过 blank 符号在 lattice 中的分布),text 沿垂直路径复制得到 duration-aligned text X'。这类似于 TTS-Transducer 的做法,但 SMLLE 将此对齐信息直接传入 AR 阶段而非 NAR 阶段。
>
> **创新判断**: Transducer + AR mel prediction 的组合是新颖的流式 TTS 架构。与 TTS-Transducer (Bataev et al., 2025) 对比: TTS-Transducer 用 Transducer 直接预测 codec tokens (非流式),SMLLE 用 Transducer 预测 semantic tokens 再接 AR mel 生成 (流式)。与 Transduce-and-Speak / VALL-T 对比: 它们的最终语音生成仍是句子级,SMLLE 的 AR 阶段也是逐帧流式。Delete <Bos> Mechanism 是本文独创的工程技巧。
>
> 检索命中: [[SemanticvsAcousticTokens]]✓, [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]✓ | 过滤: [[MelSpectrogram]](pending-review), [[DurationPredictor]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 首个逐帧流式零样本 TTS——Transducer 实时生成 semantic tokens 并提供 duration 对齐,AR 模型逐帧将 semantic tokens + 对齐文本转为 mel spectrogram,Delete <Bos> Mechanism 以最小延迟获取未来文本信息。
> - **路线**: Text (phoneme) → Conformer Encoder → Transducer (RNNT) → Semantic Tokens + Duration-Aligned Text → Delete <Bos> Mechanism → AR Transformer (MELLE-based) + Latent Sampling → Mel Spectrogram → BigVGAN-V2 → Waveform
> - **指标**: WER-C 5.14% / WER-H 6.37% / SIM 0.516 (streaming, LibriSpeech test-clean); SMLLE-R5: WER-C 1.03% / WER-H 1.67% / SIM 0.578 (5x repeated Transducer sampling); MOS 3.18 (vs MELLE 3.39, GT 3.61) [Table 1, 3]
> - **可借鉴**: (1) Transducer 对齐路径直接提供 duration-aligned text,免去 forced alignment 或 duration predictor; (2) Delete <Bos> Mechanism 让 AR 模型以几乎零延迟获取未来文本; (3) 两阶段解耦使 Transducer 和 AR 各自可独立优化
> - **局限**: 仅在 LibriSpeech 960h 训练,未在大规模数据验证; SIM (0.516) 与当前 SOTA (>0.8) 差距大; MOS (3.18) 低于句子级 MELLE (3.39); 重复采样 (R5) 破坏了流式特性; 未提供 RTF/真实延迟数据; 未开源

## 核心问题

1. **如何实现零样本逐帧流式 TTS,在不依赖 lookahead 的情况下保持合理质量?** — 用 Transducer 流式生成 semantic tokens + 提供 duration alignment,AR 模型逐帧生成 mel [§1, §2]
2. **如何让逐帧 AR 生成在仅看到当前及过去文本的约束下保持稳定?** — Delete <Bos> Mechanism: 删除 duration-aligned text 中的 <bos> token,让模型提前接触未来文本,以最小延迟换取稳定性 [§2.2.1]
3. **Transducer 的 semantic token 预测质量对最终语音影响有多大?** — 重复采样实验表明 Transducer 是系统瓶颈: 5 次采样后 WER-C 从 5.14% 降到 1.03% [Table 1, Fig 4]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SMLLE 是一个两阶段流式 TTS 框架 [§2, Fig 1]:

**阶段 1: Transducer Stage (文本 → semantic tokens)**
- 将 phoneme 序列 X = {<bos>, x_1, ..., x_T, <eos>} 作为输入,semantic token 序列 Y = {b, y_1, ..., y_S} 作为目标 [§2.1]
- Semantic tokens 来自 SpeechTokenizer 第一层 codec [§2.1, ref 20]
- Transducer 由三部分组成 [§2.1]:
  - **Encoder**: Conformer (文本编码器),编码输入文本序列
  - **Predictor**: 2-layer LSTM (语音编码器),自回归建模已预测的 semantic tokens
  - **Joint-Net**: 两个线性层,计算 emission probability p(i,j) 和 wait probability (blank) ∅(i,j)
- 训练: RNNT loss,对所有合法单调对齐路径做边际化 [§2.1, Eq 1-2]
- 推理: 流式逐 token 生成,同时通过垂直路径复制获得 duration-aligned text X' [§2.1]

**阶段 2: Autoregressive Stage (semantic tokens + text → mel spectrogram)**
- 输入: semantic tokens Y, duration-aligned text X' (经 DBM 处理后为 X^D), 前一帧 mel spectrogram m_{t-1} [§2.2.2, Eq 4]
- AR 模型在第 t 步接收 y_t, x^D_t, m_{t-1},预测 m_t [§2.2.2]
- **Latent Sampling Module** (继承自 MELLE): hidden state e_t → linear layer 预测 μ_t 和 log σ_t^2 → reparameterization sampling z_t → MLP + residual → mel frame m_t [§2.2.2, Eq 5-6, Fig 3]
- **无 Post-Net**: 流式场景下无法访问未来帧,不使用 MELLE 的 Post-Net 细化 [§2.2.2]
- 训练 loss: L_AR = L_reg + λ·L_KL + β·L_flux (regression + KL divergence + spectrogram flux),无 stop prediction loss (由 Transducer 决定 duration) [§2.2.2, Eq 7]

**Vocoder**: BigVGAN-V2,从 mel spectrogram 合成 16kHz 波形,hop size 320 [§3.2]

### 关键设计选择

**为什么用 Transducer 而不是直接用 LLM 做流式?**

[论文原文] 现有 LLM-based TTS 需要处理完整句子才开始生成,延迟高 [§1]。Transducer 天然具有流式能力: 每接收一个新文本 token,就可以决定生成 0 或多个 semantic tokens (通过 blank 机制),实现实时文本到语义的转换 [§2.1]。

[agent 解读] 这是 ASR 领域 Transducer 的经典优势——在 ASR 中 Transducer 可以边听边转写,SMLLE 将其反转为边读文本边生成 semantic tokens。相比之下,直接让 AR 模型做 text→mel 的流式生成面临一个困难: 文本和 mel 的长度比极不对称 (一个 phoneme 对应几十帧 mel),AR 模型难以在没有对齐信息的情况下稳定生成。Transducer 的对齐路径解决了这个问题。

**为什么不直接用 Transducer 输出 mel,而要加 AR 阶段?**

[agent 解读] 论文未显式讨论这个问题,但从架构设计可以推断: Transducer 的输出是离散 token (semantic tokens 是 VQ codebook 中的 index),而 mel spectrogram 是连续值。直接用 Transducer 预测连续 mel frames 在数学上可行 (类似 CTC 的连续扩展),但会失去 Transducer 的离散建模优势。两阶段解耦让每个模块各自最优化: Transducer 专注对齐 + 语义,AR 模型专注声学细节。这也是 Transduce-and-Speak [17] 和 VALL-T [18] 的共同策略。

**为什么用 SpeechTokenizer 的第一层而不是 HuBERT 或 EnCodec?**

[论文原文] Semantic tokens 是 "text-related representations that are decoupled from speech" [§2.1],选择 SpeechTokenizer 第一层是因为它经过 HuBERT 蒸馏,编码语义信息。

[agent 解读] SpeechTokenizer 第一层 RVQ 蒸馏了 HuBERT 语义,是一种 mixed tokenizer 的语义层。相比纯 HuBERT k-means tokens,SpeechTokenizer 的 VQ codebook 更适合作为序列建模的目标 (固定词表大小)。相比 EnCodec,去掉了声学残差层可以降低 Transducer 的建模负担,让对齐更稳定。

**Delete <Bos> Mechanism (DBM) 的设计动机**

[论文原文] 在 Transducer 的对齐中,<bos> 对应语音开头的静默段,没有明确的语音学含义。通过删除 duration-aligned text 中的 <bos> tokens,AR 模型可以更早地接触到后续文本,从而在几乎不增加延迟的前提下提高生成稳定性 [§2.2.1]。删除的 <bos> 数量由末尾等量的 <eos> 补齐以保持序列长度一致 [§2.2.1, Eq 3]。

[agent 解读] DBM 本质上是一种"零成本 lookahead": 语音开头的静默段对应的文本 token 是无意义的 <bos>,将其替换为后续有意义的文本 token,相当于让 AR 模型在静默段时就提前看到了真正的文本内容。这比 chunk-level lookahead (如 LiveSpeech 2 需等 4-8 个未来文本) 的延迟小得多。

### 训练策略

- **Transducer Stage**: 
  - Text encoder: Conformer (基于 WeNet 配置) [§3.2]
  - Speech encoder: 2-layer LSTM [§3.2]
  - Optimizer: Adam, lr=0.001, warmup 25K steps [§3.2]
  - 训练: 12 epochs 达最低 RNNT loss [§3.2]
  - Text: eSpeak phoneme 提取 [§3.1]
  
- **AR Stage**:
  - 12 blocks, hidden dim 1024, 16 heads, FFN dim 4096, dropout 0.1 [§3.2]
  - Mel input: 3 层 linear,dropout 0.5 [§3.2]
  - Optimizer: Adam, lr=0.0005, warmup 30K steps [§3.2]
  - β=0.5, λ=5e-2 (λ 在 10K steps 后才生效) [§3.2]

- **数据**: LibriSpeech 960h,80-dim log-mel spectrogram,16kHz [§3.1]
- **Vocoder**: BigVGAN-V2,upsample [5,4,2,2,2,2],hop size 320,400K steps [§3.2]
- **两阶段分别训练**: Transducer 和 AR 模型各自独立训练 [§3.2]

## 实验

| 指标 | 本文 (SMLLE) | 本文 (SMLLE-R5) | VALL-E | MELLE | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER-C (sentence-level comparison) | 5.14% | 1.03% | 5.9% | 1.47% | 1.61% | LibriSpeech test-clean | [Table 1] |
| WER-H | 6.37% | 1.67% | — | 2.10% | 2.15% | LibriSpeech test-clean | [Table 1] |
| SIM | 0.516 | 0.578 | 0.580 | 0.681 | 0.734 | LibriSpeech test-clean | [Table 1] |
| MOS (40 speakers) | 3.18 | — | 3.02 | 3.39 | 3.61 | LibriSpeech test-clean | [Table 3] |
| CMOS (similarity) | 3.54 | — | 3.81 | 3.46 | 4.13 | LibriSpeech test-clean | [Table 3] |

### 与流式 TTS 系统对比 [Table 2]

| 系统 | WER | SIM | FTL | 数据集 |
| --- | --- | --- | --- | --- |
| LiveSpeech2-chunk | 3.1% | 0.617 | (4~8)d_text + d_model | LibriTTS test-clean |
| LiveSpeech2-limit | 40.7% | 0.582 | d_text + d_model | LibriTTS test-clean |
| SMLLE | 6.66% | 0.528 | d_text + d_model | LibriTTS test-clean |
| SMLLE-R5 | 1.22% | 0.603 | — | LibriTTS test-clean |

**关键发现**: LiveSpeech2-limit 在帧级模式下 WER 退化到 40.7%,而 SMLLE 在同等约束 (d_text + d_model 延迟) 下仅 6.66%,差距悬殊 [Table 2]。

### 重复采样消融 [Fig 4]

| 采样次数 | WER-C | WER-H | SIM (×10) |
| --- | --- | --- | --- |
| 1 | 5.16 | 6.37 | 5.16 |
| 2 | 3.44 | 5.47 | 5.61 |
| 3 | 2.35 | — | 5.71 |
| 4 | 1.91 | — | 5.71 |
| 5 | 1.67 | 1.67 | 5.78 |

随采样次数增加,WER 和 SIM 单调改善,且前两次采样提升最显著 [§4.2]。

### DBM 消融 [Table 4]

| Prompt DBM | WER-C | WER-H | SIM |
| --- | --- | --- | --- |
| 是 | 10.26% | 13.28% | 0.487 |
| 否 | 5.14% | 6.37% | 0.516 |

对 prompt 应用 DBM 严重降低性能,因为训练时模型未见过两段文本同时使用 DBM 的模式 [§4.3]。

## 局限性

1. **数据规模与 speaker similarity**: 仅在 LibriSpeech 960h 上训练,SIM 0.516 与当前 SOTA (IndexTTS2 0.865, CosyVoice 3 0.796) 差距显著。未验证 scaling 行为 [agent 解读]。
2. **MOS 低于句子级系统**: MOS 3.18 低于同样基于 MELLE 的句子级版本 (3.39) 和 GT (3.61),streaming 模式的质量折损明显 [Table 3]。
3. **重复采样破坏流式性**: SMLLE-R5 需要对 Transducer 采样 5 次并选最优,这本质上已不是 streaming——需要等一整句生成 5 遍后再选 [agent 解读]。论文未讨论此矛盾。
4. **缺乏延迟量化**: 虽然定义了 FTL = d_text + d_model,但未给出具体毫秒级延迟数字,也未报告 RTF [agent 解读]。
5. **两阶段分别训练**: Transducer 和 AR 模型分别训练,未端到端联合优化。AR 模型训练时用 GT semantic tokens 和 GT 对齐,推理时用 Transducer 预测的 tokens 和对齐,存在 train-test mismatch [agent 解读]。这可能是重复采样提升巨大的原因之一——更好的 Transducer 输出减少了 mismatch。
6. **Baseline 局限**: 流式对比仅有 LiveSpeech 2 一个 baseline [Table 2],未与 Speech-T、VoXtream、chunk-level AR 等其他流式方案对比。
7. **未开源**: 仅提供 demo 页面,代码和权重未公开。

## 点评

**优势**:
1. 问题定位精准: 逐帧流式零样本 TTS 是实际需求痛点,现有方案要么不流式 (MELLE/VALL-E),要么流式但质量崩坏 (LiveSpeech2-limit 40.7% WER)。SMLLE 在这个空白点取得了实用水平 (6.37% WER-H)。
2. DBM 设计巧妙: 利用语音起始静默段的无意义 <bos> token 位置"偷"到未来文本信息,是零额外延迟的 lookahead,思路优雅。
3. 两阶段解耦架构清晰: Transducer 管对齐和语义,AR 管声学,各自可独立优化。重复采样实验清晰证明了 Transducer 是瓶颈,为后续改进指明了方向。

**不足**:
1. 实验规模偏小: 仅 LibriSpeech 960h,且 SIM 0.516 在当前零样本 TTS 竞争中偏弱。
2. 重复采样 (R5) 的定位模糊: 它提供了最强结果 (WER-C 1.03%),但已不是真正的 streaming。论文应更清晰地讨论这一 trade-off。
3. 延迟分析缺失: 对于 streaming TTS 论文,不报告毫秒级 RTF 和首帧延迟是重大缺失。
4. 与 TTS-Transducer (Bataev et al., 2025) 的实验对比缺失: 同为 Transducer-based TTS,仅在 Related Work 中提及,未做直接数值对比。

**定位**: SMLLE 是首个在逐帧模式下实现可用零样本质量的 streaming TTS。其 Transducer + MELLE-AR 的两阶段架构为 streaming TTS 提供了一条可行路线。核心贡献在于证明了这条路线的可行性和 Transducer 作为对齐模块的价值,但距离实际部署 (需要更高 SIM、更低延迟、更大数据规模) 还有差距。

## 可复用的 idea

1. **Transducer 提供 duration alignment**: Transducer 的对齐路径天然给出文本到语音帧的单调映射,无需额外 forced alignment 或 duration predictor。任何需要 text-speech alignment 的场景都可以考虑 Transducer 替代 MFA/CTC。
2. **Delete <Bos> Mechanism**: 利用序列中"语义空位" (如静默段对应的无意义 token) 的位置提前注入未来信息。这个思路可以泛化到任何序列生成任务中存在可预测无意义位置的场景。
3. **两阶段 semantic-to-acoustic 流式 pipeline**: Transducer (离散语义,可流式) → AR (连续声学,可逐帧) 的架构模式。这种解耦使得两阶段可以用不同的模型架构和训练策略分别优化,也可以独立替换某一阶段。
4. **重复采样作为质量上界探测**: 对 Transducer 多次采样取最优,虽不适合真实流式场景,但可用于离线分析 Transducer 质量的潜力上限,指导后续优化方向。

---

> [!review] 审阅状态 (2026-06-03, agent-auto)
> **结论: pass-with-fixes**
> - 无 high/medium issue
> - 2 low: frontmatter concepts 中 Speaker Embedding 和 Non-autoregressive TTS 的挂接偏弱 (不影响反向更新)
> 详见 `_review/SMLLE-review.yml`

检索命中: [[SemanticvsAcousticTokens]], [[LLM-basedTTS]], [[SpeechTokenizer]], [[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]] | 过滤: [[MelSpectrogram]](pending-review), [[DurationPredictor]](pending-review) | 未命中但可能相关: 无
