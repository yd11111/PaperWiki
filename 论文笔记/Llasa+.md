---
type: paper
tier: deep
title: "Llasa+: Free Lunch for Accelerated and Streaming Llama-Based Speech Synthesis"
arxiv_id: "2508.06262"
source: "Sources/Llasa+.pdf"
authors: [Wenjie Tian, Xinfa Zhu, Hanke Xie, Zhen Ye, Wei Xue, Lei Xie]
year: 2025
venue: "arXiv preprint"
tags: [TTS, LLM-based-TTS, inference-acceleration, multi-token-prediction, speculative-decoding, streaming, speech-codec, single-codebook, autoregressive, open-source]
concepts: ["[[LLM-basedTTS]]", "[[CodecLanguageModel]]", "[[Single-codebookvsMulti-codebook]]", "[[StreamingSpokenDialogue]]", "[[SpeechTokenizer]]"]
models: ["[[论文笔记/Llasa|Llasa]]", "[[模型库/CosyVoice2|CosyVoice 2]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 3
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[CosyVoice2]]✓)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: Llasa+ 是 Llasa 的推理加速与流式化扩展。Llasa 在 LLM-based TTS 中代表"彻底对齐标准 LLM"路线 -- 单层 VQ codec (X-codec2) + 单 LLaMA Transformer, 已验证 train-time 和 inference-time scaling law。现有 LLM-based TTS 的推理加速探索有限: VALL-E 2 的 Grouped Code Modeling 需要重训 backbone, CosyVoice 2 用 chunk-aware causal flow matching 优化解码端但未解决 AR token 预测本身的瓶颈。Llasa+ 直接切入 AR token 预测环节, 是 TTS 领域系统探索 multi-token prediction + verification 加速的首个工作。
>
> **已有认知**: LLM-based TTS 的推理瓶颈在自回归 token 逐个生成; 单码本路线 (Llasa 的 X-codec2, 50Hz, 65536 entries, FSQ) 序列长度已经是最短的 (vs RVQ 多层展开), 但 AR 本身仍然慢。NLP 领域 DeepSeek-V3 的 MTP 训练策略和 speculative decoding 的 draft-then-verify 范式是成熟方案, 但在 TTS 中的适用性未经验证 -- 语音 token 的分布特征 (高码本维度、韵律连续性) 与文本 token 不同。Streaming TTS 需要 codec decoder 也支持因果推理, X-Codec2 原始使用非因果 Transformer decoder。
>
> **创新判断**: Llasa+ 的核心贡献不在提出 MTP 本身 (来自 DeepSeek-V3), 而在三方面: (a) 证明 frozen backbone + plug-and-play MTP modules 在 TTS 中可行, 无需重训大模型; (b) 提出适配 TTS 的 verification 算法 (特别是 EOS 验证), 使 MTP 不牺牲质量甚至略有提升; (c) 将 X-Codec2 改造为因果流式版本 (XCodec2-S), 仅解冻 decoder 即保留 ~95% 原始质量。
>
> 检索命中: [[LLM-basedTTS]], [[SpeechTokenizer]], [[CosyVoice2]] | 过滤: [[CodecLanguageModel]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review), [[StreamingSpokenDialogue]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 在冻结的 Llasa backbone 上加挂可插拔 MTP 模块 + verification 算法, 实现 1.48x 推理加速且不降质量; 附带 XCodec2-S 支持流式波形重建
> - **路线**: Text tokens → [Frozen Llasa-1B backbone] → hidden states → [MTP1 (LLaMA block)] → token prediction 1 → [MTP2 (LLaMA block)] → token prediction 2 → [Verification against backbone logits] → accepted tokens → [XCodec2-S causal decoder] → streaming waveform
> - **指标**: Seed-TTS-eval-en WER 2.499% (vs Llasa baseline 3.220%), SIM 0.575 (vs 0.572), speed-up 1.48x [Table I]; XCodec2-S WER 3.239% / SPK-SIM 0.795 / UTMOS 4.029 (vs XCodec2 2.470/0.821/4.127) [Table II]
> - **可借鉴**: Frozen backbone + plug-and-play MTP 加速策略可推广到任何 LLM-based 生成模型; verification 中对 EOS token 的专门处理是稳定性关键; MTP 在 TTS 中不仅加速还可能改善质量 (更长距离 historical context)
> - **局限**: 仅在 LibriTTS (~960h) 上训练 MTP, 未验证大数据场景; 仅基于 Llasa-1B, 未测 3B/8B; 第二个 MTP 模块加速增益有限 (~15.66% vs 第一个 ~32.21%); XCodec2-S 质量仍低于非因果 XCodec2

## 核心问题

LLM-based TTS 系统 (如 Llasa) 将 TTS 完全对齐标准 LLM 范式, 获得了 scaling 的好处, 但也继承了 AR 推理的根本瓶颈: **每步只能生成一个 token, 推理速度随序列长度线性增长**。同时, 实时交互场景需要 streaming 合成能力, 而 Llasa 的 X-Codec2 使用非因果 decoder, 不支持流式波形重建。Llasa+ 试图回答: **能否在不重训大模型、不损失生成质量的前提下, 通过轻量级附加模块同时解决 AR 推理加速和流式合成两个问题?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Llasa+ 由三个组件构成 [§II, Fig 1]:

1. **Frozen Llasa-1B backbone**: 16 层 LLaMA decoder, hidden size 2048, 32 attention heads, FFN 8192, 参数完全冻结 [§III-B]
2. **两个 MTP 模块**: 每个由一个 Linear Projector + 一个 LLaMA block 构成, 共享 frozen LM head, 级联在 backbone 之后 [§II-A]
3. **XCodec2-S**: X-Codec2 的因果版本, 仅 decoder 部分解冻训练, 支持流式波形重建 [§II-C]

**设计理念**: backbone 完全冻结 → MTP 模块作为"插件"训练 → 训练成本极低 (仅 LibriTTS, 4xA800, 20 epochs) → 可推广到任何 LLM-based 模型 [§I]。[论文原文]

### 关键设计选择

#### Multi-Token Prediction (MTP) 模块

每个 MTP 模块结构简单: 一个 Linear Projection + 一个 LLaMA decoder layer (与 backbone 相同配置) [§II-A]。

**级联工作方式**: backbone 产生 hidden states $h^0_{0:t}$, 依次经过 MTP1 和 MTP2 [§II-A]:
- MTP1: $h^1_{0:t} = \text{MTP}_1(h^0_{0:t})$ → 通过 shared LM head 预测 $S_{1:t+1}$ (下一个 token)
- MTP2: $h^2_{0:t} = \text{MTP}_2(h^1_{0:t})$ → 通过 shared LM head 预测 $S_{2:t+2}$ (下下个 token)

**训练 loss**: 仅计算 MTP 模块的 cross-entropy loss, backbone 不参与训练 [Eq. 2]:
$$L_{MTP} = \sum_{k=1}^{N-1} L_{CE}(S_{0:T-k-1}, G_{k+1:})$$

**为什么共享 LM head**: 减少参数量和训练难度, 因为 LM head 本身已经学会了 token 到 logits 的映射 [§II-A]。[论文原文]

**为什么不用 DeepSeek 原始 MTP 设计**: DeepSeek-V3 的 MTP 将 ground-truth tokens 也作为 MTP 输入。但在 TTS 中, 训练时用 ground-truth tokens, 推理时用 predicted tokens, 这种 train-inference mismatch 会加剧级联错误。实验证实 Llasa-DPSK-MTP1/2 性能均劣于对应的 Llasa-MTP1/Llasa+ [Table I, §IV-B]。[论文原文]

**为什么用 attention layer 而非 MLP**: Llasa-MLP (用 MLP 替代 attention) 在 WER 上从 2.745 升至 4.375, SIM 下降约 0.05, 加速比也从 39.83% 降至 37.76%。Attention 机制对序列 token 预测至关重要 [§IV-B]。[论文原文]

#### Verification 算法

MTP 预测的 token 不总是准确的。不加 verification 时, WER 从 3.070 暴涨到 14.372, SIM 从 0.570 跌到 0.463 [Table III]。[论文原文]

**核心思路**: 利用 frozen backbone 的"可信"输出来验证 MTP 的"不可信"预测 [§II-B, Algorithm 1]:

1. 在时间步 t, backbone 生成可信 token $S_t$, MTP1 预测 $S'_{t+1}$, MTP2 预测 $S'_{t+2}$。三个 token 全部暂时加入序列。
2. 在时间步 t+1, backbone 对包含三个 token 的序列计算 logits:
   - 用 backbone 的 $\text{logits}_{t+1}$ 验证 $S'_{t+1}$: 检查 $S'_{t+1}$ 是否在 $\text{logits}_{t+1}$ 的 top-k 中
   - 如果 $S'_{t+1}$ 不在 top-k → 两个 MTP token 都被拒绝, 从 $\text{logits}_{t+1}$ 重新采样
   - 如果 $S'_{t+1}$ 在 top-k → 接受 $S'_{t+1}$, 继续验证 $S'_{t+2}$
3. 同理验证 $S'_{t+2}$

**EOS token 的特殊处理**: MTP 产生的 EOS 必须经过专门验证, 否则模型容易异常终止。实验显示不做 EOS 验证时 Llasa+ WER 从 3.070 升至 3.422 [Table III, §V-2]。[论文原文]

**为什么这个 verification 算法本质上类似 speculative decoding**: [agent 解读] 这和 NLP 中的 speculative decoding (用小模型起草、大模型验证) 在机制上非常相似。区别在于: (1) "小模型"是挂在大模型后面的 MTP 模块而非独立模型; (2) verification 用的是 top-k 匹配而非 token-level probability ratio; (3) 验证发生在已经将 token 加入序列之后 (需要 replace), 而不是先验证再加入。

**top-k 的影响**: top-k 越大, 接受率越高 → 加速越大, 但质量可能下降。top-k=100 时 WER 最低 (所有变体), top-k=500 时加速最优且质量不降 [Fig 2]。[论文原文]

#### XCodec2-S (Streaming X-Codec2)

将 X-Codec2 的非因果 Transformer decoder 改为因果版本 [§II-C]:
- Encoder + VQ 模块: frozen (来自预训练 X-Codec2)
- Decoder: 因果 Transformer, 仅预测基于历史 context 的 STFT magnitude + phase → iSTFT 重建
- Conv1d 层作为可训练 adapter
- 部分 conv1d 层保留少量 local lookahead (固定窗口), 在不影响流式能力的前提下增强质量 [§II-C]

**适配器设计**: 线性层将 2048 维输入映射到 1024 维输出 [§III-B]。[论文原文]

**为什么不增加更多可训练参数**: 实验发现加 trainable linear (WER 3.239→3.446) 或改 conv kernel size (7→5, WER→3.971) 反而降低质量, 可能因为 LibriTTS 训练数据量相对预训练数据太少, 影响泛化 [Table II, §IV-C]。[论文原文]

### 训练策略

**MTP 模块**: 4xA800, batch size 256, 20 epochs, max lr 1e-4, 4000 warmup steps, cosine scheduler, AdamW (0.9, 0.999) [§III-B]。

**XCodec2-S**: 8x4090, batch size 96, 280k steps, max lr 1e-4, 3000 warmup steps, cosine scheduler, AdamW (0.8, 0.9) [§III-B]。

**训练数据**: 仅 LibriTTS (train-clean-100 + train-clean-360 + train-other-500, ~960 hours) [§III-A]。[论文原文]

## 实验

### MTP 变体对比 (Seed-TTS-eval-en) [Table I]

| 模型 | Top-k | WER(%)↓ | SIM↑ | Speed-Up(%)↑ | 出处 |
| --- | --- | --- | --- | --- | --- |
| Llasa (baseline) | - | 3.220 | 0.572 | 0 | [Table I] |
| Llasa-Valle | 500 | 23.565 | 0.436 | 200.00 | [Table I] |
| Llasa-DPSK-MTP1 | 500 | 4.485 | 0.560 | 42.62 | [Table I] |
| Llasa-DPSK-MTP2 | 500 | 4.728 | 0.562 | 49.26 | [Table I] |
| Llasa-MLP | 500 | 4.375 | 0.566 | 37.76 | [Table I] |
| Llasa-MTP1 (top-k=100) | 100 | 2.642 | 0.583 | 27.81 | [Table I] |
| Llasa-MTP1 (top-k=500) | 500 | 2.745 | 0.571 | 39.83 | [Table I] |
| Llasa-MTP1 (top-k=1000) | 1000 | 3.080 | 0.562 | 46.02 | [Table I] |
| Llasa+ (top-k=100) | 100 | 2.499 | 0.575 | 41.68 | [Table I] |
| Llasa+ (top-k=500) | 500 | 3.070 | 0.570 | 47.87 | [Table I] |

**关键发现**:
1. **MTP + verification 不仅加速, 还提升质量**: Llasa-MTP1 (top-k=100) WER 2.642% < baseline 3.220%, SIM 0.583 > 0.572。作者认为 MTP 模块使模型能利用更长距离的历史信息 [§IV-A]
2. **第二个 MTP 模块增益有限**: MTP2 额外带来约 15.66% 的加速, 远低于 MTP1 的 32.21%。三个 MTP 模块 (MTP3) 仅增加 <10% 加速, 不值得额外成本 [§IV-B]
3. **DeepSeek 原始 MTP 设计不适用于 TTS**: Llasa-DPSK 变体全面劣于对应版本, 验证了 train-inference mismatch 的影响 [§IV-B]
4. **VALL-E 2 的 Grouped Code Modeling 完全失败**: Llasa-Valle WER 23.565%, 因为需要重训 backbone [Table I]

### XCodec2-S 评估 (LibriSpeech test-clean) [Table II]

| 模型 | WER(%)↓ | STOI↑ | PESQ-WB↑ | PESQ-NB↑ | SPK-SIM↑ | UTMOS↑ | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ground Truth | 1.960 | 1.000 | 4.640 | 4.550 | 1.000 | 4.090 | [Table II] |
| XCodec2 (non-causal) | 2.470 | 0.919 | 2.433 | 3.036 | 0.821 | 4.127 | [Table II] |
| XCodec2-S | 3.239 | 0.913 | 2.340 | 2.932 | 0.795 | 4.029 | [Table II] |
| XCodec2-S + linear | 3.446 | 0.911 | 2.321 | 2.931 | 0.793 | 4.023 | [Table II] |
| XCodec2-S + conv1d (k=5) | 3.971 | 0.912 | 2.315 | 2.921 | 0.793 | 4.028 | [Table II] |

**关键发现**: XCodec2-S 保留了 XCodec2 约 95% 的性能 (UTMOS 4.029/4.127, STOI 0.913/0.919), 仅解冻 decoder 的因果化改造代价极小 [§IV-C]。

### Verification 消融 [Table III]

| 模型 | WER(%)↓ | SIM↑ | Speed-Up(%)↑ | 出处 |
| --- | --- | --- | --- | --- |
| Llasa+ | 3.070 | 0.570 | 47.87 | [Table III] |
| Llasa+ w/o Verification | 14.372 | 0.463 | 200.00 | [Table III] |
| Llasa+ w/o EOS top-k | 3.422 | 0.570 | 48.14 | [Table III] |
| Llasa-MTP1 | 2.745 | 0.571 | 39.83 | [Table III] |
| Llasa-MTP1 w/o Verification | 11.471 | 0.505 | 100.00 | [Table III] |
| Llasa-MTP1 w/o EOS top-k | 3.319 | 0.570 | 40.33 | [Table III] |

**关键发现**: 去掉 verification 后 WER 暴涨 3-5x, SIM 大幅下降, 证明 verification 是 MTP 在 TTS 中可用的必要条件 [§V-1]。EOS 专门验证也是稳定性的关键 [§V-2]。

## 局限性

1. **仅在 LibriTTS (~960h) 上训练**: MTP 模块和 XCodec2-S 都只用 LibriTTS 训练, 未验证在大规模数据 (如 Llasa 原文的 250k hours) 下是否仍有效, 以及泛化到中文/多语言场景的能力 [§III-A]
2. **仅基于 Llasa-1B**: 未测试 Llasa 3B/8B 的加速效果。8B 模型推理更慢, 加速需求更大, 但 MTP 模块与 backbone 的规模匹配可能不同
3. **第二 MTP 模块收益递减**: MTP2 仅增加约 15.66% 加速 (vs MTP1 的 32.21%), 边际收益下降明显, 扩展到更多 MTP 模块不划算 [§IV-B]
4. **XCodec2-S 质量损失**: SPK-SIM 从 0.821 降到 0.795, WER 从 2.470 升到 3.239, 对于 speaker similarity 敏感的场景可能不够 [Table II]
5. **Verification 与 top-k 的耦合**: top-k 的选择影响加速比和质量的平衡, 需要针对不同场景调优 [Fig 2]
6. **16kHz 采样率限制**: 继承自 Llasa/X-Codec2, 限制了合成语音的频率上限

## 点评

**优势**:
1. **"免费午餐"的设计哲学很实用**: backbone 完全冻结, MTP 模块轻量 (单个 LLaMA layer), 训练数据仅需 LibriTTS, 4xA800 即可训练。这种"不动大模型, 只加插件"的策略具有很强的实用价值和推广性
2. **意外发现: MTP 改善质量**: WER 从 3.220→2.499 (best config), 这不仅是加速, 还是质量提升。作者的解释 (MTP 利用更长历史信息) 合理 -- 类似于 speculative decoding 中多候选采样的 diversity benefit [§IV-A]
3. **Verification 算法设计精巧**: 与 NLP 的 speculative decoding 不同, 用 top-k 匹配而非概率比验证, 适应了 TTS 中 token 分布更分散的特点; EOS 专门处理体现了工程细节
4. **XCodec2-S 改造极简**: 仅解冻 decoder + 因果化, 保留 95% 质量, 是低成本 streaming 方案的良好示范

**不足**:
1. **实验规模偏小**: 仅 Llasa-1B + LibriTTS, 在 Llasa 原文的 scaling 语境下 (1B→8B, 80k→250k hours) 这只是最小规模的验证。MTP + verification 在大模型/大数据上是否同样有效存疑
2. **缺乏端到端 TTS 评估**: Table I 的 WER/SIM 评估似乎使用 XCodec2 (非 streaming) 进行波形重建 [agent 解读], 未报告 MTP + XCodec2-S 联合使用时的端到端 streaming TTS 质量
3. **缺乏感知评估**: 无 MOS, 无 MUSHRA, 仅有 WER 和 SIM 两个客观指标, 对合成自然度的判断不完整
4. **与 Llasa test-time scaling 的关系未讨论**: Llasa 原文通过 best-of-N + PRM/ORM 实现 inference-time compute scaling (256x 候选), Llasa+ 的 MTP+verification 也是一种 inference-time 策略, 两者是否可以互补? 论文未讨论 [agent 解读]
5. **RTF / 实际延迟未报告**: 仅报告 speed-up ratio (%), 未给出 RTF (Real-Time Factor) 或绝对延迟数字, 难以判断实际部署价值

## 可复用的 idea

1. **Frozen backbone + plug-and-play MTP 加速**: 任何基于 LLM 的生成模型 (不限 TTS) 都可以用这种方式加速 -- 冻结主模型, 挂 1-2 个轻量 MTP 模块, 用 verification 保证质量。训练成本极低 (仅需 MTP 模块的少量数据训练)
2. **Top-k verification 替代 probability-based verification**: 在 token 分布较分散 (码本大, 如 65536) 的场景, top-k 匹配比概率比匹配可能更稳健; EOS 的专门验证策略可直接复用
3. **因果化现有非因果 codec decoder**: 仅解冻 decoder + 因果 attention + conv1d adapter, 不动 encoder/quantizer, 是将任何现有 codec 快速适配为 streaming 版本的最小改造路径
4. **MTP 对历史 context 的利用可能改善 TTS 质量**: 不仅是加速手段, MTP 使模型在训练时"看到"更远的未来 token, 类似于 multi-step TD learning, 可能改善韵律连贯性

> [!review] 审阅 (2026-06-04, agent)
> **结论**: pass-with-fixes (3 issues: 0 high, 1 medium, 2 low)
> - [medium] 缺乏端到端 streaming TTS 评估 (MTP + XCodec2-S 联合), 但这是原论文的局限而非笔记的问题; 笔记已在点评中指出
> - [low] 速查卡片"指标"字段中 speed-up ratio 的表述 "1.48x" 与论文一致, 但实际 Table I (top-k=500) 为 47.87% 即约 1.48x, 两处一致无误
> - [low] 可复用 idea #4 (MTP 改善质量的机制类比 multi-step TD) 为 agent 推测, 已隐含标注
> 详见 `_review/Llasa+-review.yml`

---

检索命中: [[LLM-basedTTS]], [[SpeechTokenizer]], [[CosyVoice2]] | 过滤: [[CodecLanguageModel]](pending-review), [[Single-codebookvsMulti-codebook]](pending-review), [[StreamingSpokenDialogue]](pending-review) | 未命中但可能相关: 无
