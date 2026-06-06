---
type: paper
tier: deep
title: "JoyVoice: Long-Context Conditioning for Anthropomorphic Multi-Speaker Conversational Synthesis"
arxiv_id: "2512.19090"
source: "Sources/JoyVoice.pdf"
authors: [Fan Yu, Tao Wang, You Wu, Lin Zhu, Wei Deng, Weisheng Han, Wenchao Wang, Lin Hu, Xiangyu Liang, Xiaodong He, Yankun Huang, Yu Gu, Yuan Liu, Yuxuan Wang, Zhangyu Xiao, Ziteng Wang]
year: 2025
venue: "arXiv preprint"
tags: [TTS, zero-shot, multi-speaker, long-form, conversational, E2E, flow-matching, LLM-based, streaming, RL, DPO]
concepts: ["[[ConditionalFlowMatching]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[FiniteScalarQuantization]]", "[[SemanticvsAcousticTokens]]", "[[Diffusion-basedTTS]]", "[[TokenRateandBitrateTrade-offs]]", "[[DifferentiableRewardOptimization]]", "[[SpeakerEmbedding]]"]
models: ["[[模型库/CosyVoice3|CosyVoice 3]]", "[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/Whisper|Whisper]]"]
tasks: ["[[任务库/Zero-shotSpeechSynthesis|Zero-shot Speech Synthesis]]"]
datasets: ["[[数据集/SEED-TTS-Eval|SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 6 个实体页: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[SpeechTokenizer]]✓, [[SemanticvsAcousticTokens]]✓, [[Zero-shotSpeechSynthesis]]✓, [[CosyVoice3]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]], [[LLM-basedTTS]], [[SpeechTokenizer]], [[SemanticvsAcousticTokens]], [[Zero-shotSpeechSynthesis]], [[CosyVoice3]] | 过滤: [[FiniteScalarQuantization]](pending-review), [[Diffusion-basedTTS]](pending-review), [[TokenRateandBitrateTrade-offs]](pending-review), [[DifferentiableRewardOptimization]](pending-review) | 未命中但可能相关: 无

**谱系定位**: JoyVoice 属于 LLM-based TTS 的 "one-stage E2E" 分支。在 LLM-basedTTS 概念页记录的演进线中,传统 two-stage cascaded 系统(CosyVoice 系列、Seed-TTS 等)先用 AR LLM 生成离散 semantic tokens,再用独立的 flow matching/diffusion model 渲染声学细节,两阶段独立训练。DiTAR (2025) 率先提出 AR hidden states 直接 conditioning DiT 的 E2E 联合训练思路。VibeVoice (Microsoft, 2025) 采用 continuous VAE tokenizer + next-token diffusion 实现 E2E,支持最多 4 说话人、90 分钟对话。JoyVoice 与 DiTAR 在架构层面最接近(AR hidden → DiT conditioning + 联合训练),但关键差异在于 JoyVoice **同时保留了离散 token**(用于 AR loss)和连续 hidden states(用于 DiT conditioning),而 DiTAR 仅使用连续 hidden states。

**已有认知**: CosyVoice 3 的 MM-Tokenizer 概念(多任务监督 FSQ tokenizer)与 JoyVoice 的 MM-Tokenizer 思路一致,但 JoyVoice 基于 Whisper-large-v3 而非 MinMo,且支持 12.5Hz 更低帧率。CosyVoice 2 已提出 chunk-aware causal flow matching 实现流式,JoyVoice 的 dynamic chunk FM 与之类似但采用随机 chunk 训练而非四种固定 mask。在 multi-speaker 长对话方面,VibeVoice 和 MOSS-TTSD 是直接竞品,但它们分别限于 4 说话人和 5 说话人,JoyVoice 支持 8 说话人。

**创新判断**: (1) E2E 联合训练不是全新概念(DiTAR 在先),但 JoyVoice 的"离散 token + 连续 hidden 双信号"设计是独特的 — 用离散 token 提供稳定性,用连续 hidden 提供信息丰富的 DiT conditioning; (2) 多说话人无分割序列建模(最多 8 人)是较新的探索; (3) APO (Acoustic Preference Optimization) 是 DPO 在 TTS token 级的具体实例化。

## 速查

> [!summary] 速查
> - **一句话**: 提出 E2E Transformer-DiT 联合训练架构,通过 AR hidden states 直接 conditioning 全局因果 DiT,支持最多 8 说话人、5 分钟长对话的零样本语音合成
> - **路线**: Text + Speaker Tags/Embeddings → Qwen2.5-0.5B AR Transformer → 离散 token (AM loss) + 连续 hidden states → Global Causal DiT (FM loss) → Mel → Vocoder → Waveform
> - **指标**: SEED-TTS-Eval test-zh CER 0.97% / test-en WER 1.69% / test-hard CER 5.55% [Table 3]; MSMT-eval 2spk-zh cpCER 1.88% [Table 5]; 12.5Hz E2E 与 25Hz 性能相当 [Table 2]
> - **可借鉴**: (1) 离散 token + 连续 hidden 双信号设计兼顾稳定性与信息量; (2) 无分割的多说话人序列建模(speaker tag + 隐式 attention 对齐); (3) 随机 chunk 训练统一流式/非流式; (4) 基于 CER=0 vs CER>0 构建 APO preference pairs
> - **局限**: 4 说话人以上质量明显下降(训练数据不足); RL 训练仍在进行中; 不支持音乐/音效; 未开源

## 核心问题

JoyVoice 试图解决两个关键问题:

1. **Two-stage cascaded 架构的信息瓶颈**: 传统 coarse-to-fine TTS(如 CosyVoice 系列)中,AR LLM 生成离散 speech tokens,flow matching model 从这些 tokens 恢复声学细节。但离散量化阻断了梯度传播,两阶段独立训练,AR 不知道下游 FM 需要什么信息,FM 也无法反馈修正 AR 的错误 [§1, §2.1]。

2. **多说话人长对话合成的缺失**: 现有大规模 TTS 主要面向单人短语音或双人对话,缺乏支持 3+ 说话人、长上下文(>1 分钟)的统一框架。VibeVoice 是少数支持多说话人的大规模模型,但其局限于特定架构 [§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

JoyVoice 采用 E2E-Transformer-DiT 架构 [Fig 2],由三大模块组成:

1. **Causal Autoregressive Transformer**: 基于 Qwen2.5-0.5B 初始化,输入 system prompt(speaker tags + speaker embeddings) + 文本(带 speaker tags) + speech tokens,以 next-token prediction 方式生成离散 speech tokens [§2.1, Eq.2]。

2. **Dynamic Chunk Diffusion Transformer (DiT)**: 使用 F5-TTS 的 DiT 架构,~300M 参数。以 AR Transformer 的 hidden states h_AM 作为 conditioning 输入,通过 flow matching 预测 mel-spectrogram [§2.1, Eq.3]。

3. **MM-Tokenizer + Vocoder**: MM-Tokenizer 将音频转为离散 token 表示(供 AR 训练目标使用),vocoder 将 mel-spectrogram 重建为波形 [§2.3, Fig 3]。

**联合训练目标**: L = L_AM + λ · L_FM [Eq.1],其中 L_AM 是标准 next-token cross-entropy loss [Eq.2],L_FM 是 flow matching velocity field regression loss [Eq.3]。关键梯度流: ∂L/∂θ_AM = ∂L_AM/∂θ_AM + λ · (∂L_FM/∂h_AM) · (∂h_AM/∂θ_AM) [Eq.4],即 FM loss 的梯度通过 h_AM 回传到 AM 参数。

### 关键设计选择

**为什么选择 E2E 而非 cascaded?** 论文给出四个论据 [§2.1]:
1. **Instruction-aware acoustic refinement**: cascaded 系统中 FM 仅接收离散语音 tokens(缺少指令/风格/说话人信息),E2E 中 FM 接收完整的 hidden states,包含全部上下文信息 [论文原文]。
2. **Multi-speaker 建模能力**: 离散 semantic tokens 包含有限的说话人信息,cascaded 系统需要预知说话人边界。E2E 的连续 hidden states 信息更丰富,模型可自动学习说话人区分,无需显式声学边界 [论文原文]。
3. **更好的 intelligibility**: 联合训练的 WER 显著低于 cascaded,表明 FM 的梯度帮助 AM 生成更利于准确重建的 hidden representations [论文原文]。
4. **对 tokenizer 压缩的鲁棒性**: 从 25Hz 压缩到 12.5Hz 时,cascaded baseline 明显退化(test-zh CER 1.16→1.57,test-en WER 1.74→2.19),而 E2E 几乎不受影响(CER 0.97→1.01,WER 1.70→1.63) [Table 2]。[论文原文] 解释为联合优化让 AM 在 hidden states 中编码了互补的连续信息,弥补了离散 token 的信息损失。

**为什么保留离散 token(而非纯连续)?** [agent 解读] 论文提到"strategically reincorporate discrete speech tokens, akin to those in previous two-stage models like CosyVoice, to bolster system stability" [§1]。这是与 DiTAR/VibeVoice 等纯连续路线的关键区别 — 离散 token 提供了一个稳定的训练锚点(cross-entropy loss 是成熟目标),而连续 hidden states 提供丰富信息。两者互补。

**为什么用全局因果 DiT 而非局部 DiT?** [论文原文] 指出 one-stage 模型(如 DiTAR 的局部 DiT)的有限感受野会损害音色相似度。JoyVoice 的全局因果 DiT 可以 attend 到整个历史序列,增强长上下文 speaker similarity [§1]。

**多说话人无分割序列建模** [§2.2]: 与逐 utterance 处理(如 MoonCast)不同,JoyVoice 构造统一输入序列 I = P; T; S [Eq.5-6]:
- P = {spk_0, e_0, ..., spk_{N-1}, e_{N-1}}: 所有说话人 tag + embedding 作为 prefix
- T = {spk_{i0}, t_0, ..., spk_{iM-1}, t_{M-1}}: 所有 turn 的文本(带 speaker tag)
- S = {s_1, ..., s_T}: 整段对话的连续 speech token 序列(不分割)

[论文原文] 认为这种设计保留了跨 turn 的上下文依赖、turn-taking 动态和长程韵律模式,Transformer attention 可以自动建立未分割 speech tokens 与对应文本/说话人的映射关系。

### MM-Tokenizer

基于 Whisper-large-v3 构建 [§2.3, Fig 3]:
- 在 encoder 第 12 层 Transformer block 的 linear layer 中插入 FSQ (Finite Scalar Quantization) 量化模块
- 多任务监督训练: ASR, SER (语音情感识别), AED (音频事件检测), AEC (音频事件描述), SV (说话人验证), AD (年龄检测), GC (性别分类)
- 外部 Audio Decoder 模块从 encoder 输出重建 mel-spectrogram,确保离散表示同时保留语义和声学信息
- 训练目标: L_total = L_semantic + β · L_recon [§2.3]
- 通过额外 CNN 层实现 4x/8x 下采样,分别对应 25Hz 和 12.5Hz token rate

[agent 解读] 与 CosyVoice 3 的 tokenizer 思路高度一致(多任务监督 + FSQ),但 JoyVoice 基于 Whisper-large-v3 而 CosyVoice 3 基于 MinMo,且 JoyVoice 增加了 audio decoder 进行声学重建(双目标优化)。CosyVoice 3 未报告 12.5Hz 版本,JoyVoice 则证明 E2E 框架下 12.5Hz 与 25Hz 性能相当。

### Dynamic Chunk Flow Matching

采用因果 chunk-wise FM 实现流式 [§2.4]:
- 训练时每步随机选择 chunk size,消除固定 chunk 限制
- 推理时可灵活选择任意 chunk size,适应不同延迟需求
- 每个 chunk 的 self-attention 只能 attend 到历史 chunks(causal mask)

[agent 解读] 与 CosyVoice 2 的四种固定 mask(non-causal/full-causal/chunk-M/chunk-2M)统一训练方案不同,JoyVoice 的随机 chunk 更简洁,省去了多种 mask 的设计成本。

### 训练策略

**Curriculum Learning** [§2.6]: 两阶段:
1. 第一阶段: 大规模单人短语音(<1 分钟)预训练,获得高质量 base model
2. 第二阶段: 混合全数据集(长单人 + 多说话人长音频 + 原短单人)继续微调,音频最长 5 分钟,说话人最多 8 人

**TTS-Frontend-Free** [§2.5]: 通过大规模数据增强(覆盖 TN/ITN、多音字、罕见字)替代传统文本前端预处理模块,模型基于 Qwen 文本模型初始化。所有实验和 demo 均不使用预处理模块。

**APO (Acoustic Preference Optimization)** [§2.7]:
- 对给定文本生成 N 个候选 speech token 序列
- 按 CER 评估: CER=0 → chosen set,CER>0 → rejected set [Eq.12-13]
- 构造所有 (chosen, rejected) pair,用 DPO loss 优化 [Eq.11,14]
- [论文原文] 强调这是 token-level 偏好优化,适用于 TTS 中 token 质量对整体语音自然度和可懂度有显著影响的场景

**多说话人 SFT (mSFT)** [§4.6]: 使用 speaker prompt tag (如 "You are Speaker 1") 条件化,12 小时混合数据(每人 2-3 小时)。LoRA-SFT 适用于数据有限(<1 小时)场景。还尝试了双说话人播客数据微调。

## 实验

| 指标 | 本文 (JoyVoice E2E) | Baseline (JoyVoice-Cascade) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER test-zh | 0.97% | 1.13% | SEED-TTS-Eval | [Table 3] |
| WER test-en | 1.69% | 1.75% | SEED-TTS-Eval | [Table 3] |
| CER test-hard | 5.55% | 6.07% | SEED-TTS-Eval | [Table 3] |
| SS (WavLM) test-zh | 0.786 | 0.780 | SEED-TTS-Eval | [Table 3] |
| SS (ERes2Net) test-zh | 0.827 | 0.825 | SEED-TTS-Eval | [Table 3] |
| CER 12.5Hz E2E test-zh | 0.95% | 1.57% (Cascade) | SEED-TTS-Eval | [Table 3] |
| CER+RL test-zh | 0.73% | — | SEED-TTS-Eval | [Table 3] |

**与外部 baseline 对比** [Table 3]:
- 在无 RL 条件下,JoyVoice E2E 的 test-zh CER 0.97% 优于 CosyVoice 2 (1.45%)、F5-TTS (1.56%)、MaskGCT (2.27%),略低于 CosyVoice 3-0.5B_RL (0.75%) 和 MiniMax-Speech (0.99%)
- JoyVoice-CascadeRL 的 test-zh CER 0.73%,在 single-speaker 模型中接近最优
- 多说话人模型(MoonCast/MOSS-TTSD/VibeVoice)的 SEED 指标均大幅低于 JoyVoice

**MM-Tokenizer 评估** [Table 1]:
- MM-Tok 25Hz 在 ASR 任务上 WER/CER 轻微下降(AISHELL-1: 2.53 vs Whisper 5.14),12.5Hz 进一步轻微退化
- SER (ESD accuracy): 0.886 (25Hz), 0.850 (12.5Hz)
- 保持了 Whisper baseline 的竞争性语义保留能力

**Tokenizer + Architecture 消融** [Table 2]:
- 在 25Hz 下,E2E MM-Tok 在 3 个 test set 中 2 个优于 Cascade MM-Tok 和 S3-Tokenizer Cascade
- 在 12.5Hz 下,E2E 几乎不受压缩影响,Cascade 显著退化

**Streaming 评估** [Table 4]:
- JoyVoice-Streaming (chunk=48/24) 与 JoyVoice full-context 保持相当的 CER/WER
- 加入 LLM speaker embedding 后 CER 略有增加但 SS 大幅提升(test-hard SS: 0.773→0.799 for full, 0.809→0.820 for chunk=48)
- JoyVoice-Streaming chunk=24 with speaker embedding 在 SS 上超越 CosyVoice 3-0.5B [Table 4]

**多说话人 MSMT 评估** [Table 5]:
- JoyVoice 在自建 JoyVoice-MSMT-eval 上全面领先:
  - test-zh-msmt 2spk: CER 1.44% / cpCER 1.88% vs VibeVoice-7B 1.80% / 7.57%
  - test-en-msmt 2spk: WER 3.36% / cpWER 3.61% vs VibeVoice-7B 4.19% / 6.58%
- 4spk 质量退化明显(cpCER 13.34%),论文归因于训练数据中 4+ 说话人对话的不足 [Fig 4]

## 局限性

1. **4+ 说话人退化**: 4 说话人场景的 cpWER 显著高于 2-3 说话人,论文坦承训练数据中高说话人数对话覆盖不足 [§6]。
2. **RL 训练未完成**: 大规模 RL 仍在进行中,论文提到计划用 RL 提升长期多说话人稳定性和拟人化情感表达 [§6]。
3. **仅语音生成**: 不支持音乐、音效等通用音频生成 [§6]。
4. **未开源**: 论文未提供模型权重或代码。
5. **评估局限**: MSMT benchmark 为自建,外部可比性有限;多说话人评估依赖 pyannote 开源模型,其性能影响 cpWER 指标 [§4.5]。
6. **与 CosyVoice 3 对比不完整**: CosyVoice 3-0.5B_RL 在 test-zh CER (0.75%) 和 test-hard CER (5.09%) 上优于 JoyVoice E2E (0.97% / 5.55%),但 JoyVoice 的 RL 版本 (JoyVoice-CascadeRL test-zh 0.73%) 表明 RL 可进一步提升 E2E 性能 [agent 解读]。

## 点评

**优势**:
- E2E 联合训练思路清晰,Table 2 的 12.5Hz 消融实验是最有说服力的证据 — 它直接证明了 E2E 中 AR hidden states 编码了超出离散 token 的互补信息,使系统对 tokenizer 压缩具有鲁棒性。
- 多说话人无分割序列建模(P;T;S)是一个简洁有效的设计,避免了 utterance-level 处理带来的强制对齐和人工分割问题。
- JoyVoice-MSMT-eval benchmark 的 cpWER 指标引入是有价值的,cpWER 同时评估语音识别和说话人区分,比单独的 CER 更适合多说话人场景。

**疑问**:
- 论文未报告 E2E 版本的 RL 结果(仅有 JoyVoice-CascadeRL),无法判断 APO 对 E2E 架构的提升幅度。
- 训练数据规模未明确报告(仅描述了 pipeline 流程和数据格式),无法判断与 CosyVoice 3 (1M h) 或 VibeVoice 的数据规模对比。
- 论文称 E2E 优于 cascaded,但 Table 3 中 JoyVoice-CascadeRL 的 test-zh CER (0.73%) 优于 JoyVoice E2E (0.97%),说明 cascaded + RL 的组合在特定指标上仍有竞争力。

## 可复用的 idea

1. **离散 token + 连续 hidden 双信号 E2E 训练**: 用离散 token 的 CE loss 提供稳定训练信号,用连续 hidden states 向下游 DiT 传递丰富信息。这种"两条腿走路"的设计可迁移到其他 coarse-to-fine 架构。

2. **无分割多说话人序列格式 I = P; T; S**: 将所有说话人信息前置(tag + embedding),文本带 tag 排列,speech tokens 连续不分割。让 attention 自动学习对齐,省去 forced alignment 和 speaker boundary detection。

3. **随机 chunk 训练统一流式/非流式**: 比 CosyVoice 2 的四种固定 mask 方案更简洁,训练时随机 chunk size,推理时灵活选择。

4. **CER-based APO preference pair 构建**: 简单直接 — CER=0 为 chosen,CER>0 为 rejected。无需复杂的 reward model,适用于任何可用 ASR 评估的 TTS 系统。

5. **E2E 对 tokenizer 压缩的鲁棒性**: 论文证明 E2E 联合训练可在不牺牲性能的前提下将 token rate 从 25Hz 降到 12.5Hz,对长文本/长对话场景的序列长度减半意义重大。

> [!review] 自动审阅 (2026-06-06)
> **结论:** pass
> **原则:** 复述 9 | 信赖 8 | 区分 9 | 定位 9 | 污染 9
> **Claim 标注率:** 91% (32/35)
> **问题:** 0 high, 1 medium, 2 low
> - ⚠️ [traceability-gap] 实验表第6行: baseline 1.57% 来自 Table 2 但标注为 [Table 3]
> - 💡 [traceability-gap] 实验表第7行: "CER+RL" 行标题未标明是 CascadeRL
> - 💡 [template-compliance] 未提及论文的四语种支持(日文、韩文)
> **反向更新:** ✅ 安全
