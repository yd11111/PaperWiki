---
type: paper
tier: deep
title: "CosyVoice 2: Scalable Streaming Speech Synthesis with Large Language Models"
arxiv_id: "2412.10117"
source: "Sources/CosyVoice2.pdf"
authors: [Zhihao Du, Yuxuan Wang, Qian Chen, Xian Shi, Xiang Lv, Tianyu Zhao, Zhifu Gao, Yexin Yang, Changfeng Gao, Hui Wang, Fan Yu, Huadai Liu, Zhengyan Sheng, Yue Gu, Chong Deng, Wen Wang, Shiliang Zhang, Zhijie Yan, Jingren Zhou]
year: 2024
venue: "arXiv"
tags: [TTS, zero-shot, streaming, LLM-based, coarse-to-fine, flow-matching, FSQ, instructed-TTS, reinforcement-learning, multilingual]
concepts: ["[[SpeechTokenizer]]", "[[FiniteScalarQuantization]]", "[[ConditionalFlowMatching]]", "[[Classifier-FreeGuidance]]", "[[SemanticvsAcousticTokens]]", "[[LLM-basedTTS]]", "[[CodecLanguageModel]]"]
models: ["[[模型库/CosyVoice2|CosyVoice 2]]", "[[模型库/CosyVoice|CosyVoice]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[InstructedSpeechGeneration]]"]
datasets: ["LibriSpeech"]
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeechTokenizer]], [[ConditionalFlowMatching]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechTokenizer]]✓, [[ConditionalFlowMatching]]✓, [[ResidualVectorQuantization]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[FiniteScalarQuantization]][待确认], [[Classifier-FreeGuidance]][待确认], [[CodecLanguageModel]][待确认] | 未命中但可能相关: 无

**谱系定位**: CosyVoice 2 是 CosyVoice 系列的第二代,延续 CosyVoice 1 开创的"监督式 semantic tokens + LLM + CFM"架构。KB 中已记录 CosyVoice 1 的 S3 tokenizer (VQ 插入 ASR encoder) 和 OT-CFM 渲染方案。CosyVoice 2 的主要增量在于: (1) 用 FSQ 替换 VQ 改善 codebook 利用率; (2) 用预训练文本 LLM (Qwen2.5) 初始化 text-speech LM; (3) 统一 streaming/non-streaming 生成; (4) 升级 instructed generation 能力。

**已有认知**:
- FSQ 概念页 [待确认] 详细记录了 FSQ 的 round-based 量化机制和 100% codebook 利用率优势,CosyVoice 2 是 FSQ 在 TTS tokenizer 中的首次应用
- CFM 概念页记录了 OT-CFM 在 CosyVoice 系列中的演化,CosyVoice 2 进一步引入 chunk-aware causal mask 实现流式 flow matching
- Semantic vs Acoustic Tokens 概念页记录了 CosyVoice 开创的"监督式 semantic tokens"路线,CosyVoice 2 继承并优化

> [!summary] 速查
> - **一句话**: CosyVoice 的流式升级版,用 FSQ 替代 VQ + 预训练 LLM 初始化 + chunk-aware causal flow matching 实现近无损流式零样本 TTS
> - **路线**: Text (BPE) → Text-Speech LM (Qwen2.5 init, AR) → supervised speech tokens (FSQ-SenseVoice, 25Hz) → Chunk-aware CFM (causal Conv-Transformer UNet) + speaker emb → Mel → vocoder → Waveform
> - **指标**: LibriSpeech test-clean WER 2.45% / NMOS 3.90 / SS 0.751 (streaming); SEED test-zh CER 1.45% / SS 0.806 (offline) [Table 5, 6]
> - **可借鉴**: (1) FSQ 替代 VQ 实现 100% codebook 利用率 + 更低 ASR error; (2) 文本 LLM 初始化 text-speech LM 降低 CER 18%; (3) 四种 attention mask (non-causal/full-causal/chunk-M/chunk-2M) 统一训练实现隐式自蒸馏; (4) DPO + differentiable ASR reward 用于 SFT 阶段 RL 微调
> - **局限**: (1) 仅支持中英日韩四种语言; (2) 无法通过文本指令控制音色 (timbre); (3) 不支持歌声合成; (4) 日语因字符集重叠导致性能较差

## 核心问题

CosyVoice 1 虽然在零样本 TTS 上取得突破,但存在两个关键问题 [§1]:

1. **不支持流式合成**: 大部分零样本 TTS 模型在非流式(offline)模式下工作,需要完整合成整个句子后才返回波形,导致高延迟,不适合 LLM-based voice chat 等交互场景 [论文原文]
2. **VQ codebook 利用率低**: CosyVoice 1 的 VQ tokenizer (4096 entries) 仅利用 23% 的 codebook [Table 4],大量码字浪费 [论文原文]

CosyVoice 2 试图回答: **能否在保持或超越 CosyVoice 1 质量的前提下,实现低延迟流式合成,并统一 streaming/non-streaming 在同一模型中?**

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CosyVoice 2 沿用 CosyVoice 1 的 coarse-to-fine 两阶段架构 [Fig 1],但每个组件都有重大改进:

1. **Text Tokenizer**: BPE tokenizer,直接处理原始文本,无需 G2P [§2.1]
2. **Supervised Semantic Speech Tokenizer**: FSQ 替代 VQ,插入 SenseVoice-Large ASR encoder [§2.2]
3. **Unified Text-Speech LM**: Qwen2.5-0.5B 初始化,统一 streaming/non-streaming [§2.3]
4. **Chunk-aware Causal Flow Matching**: 支持多种 attention mask 的 causal CFM [§2.4]
5. **Vocoder**: 预训练 vocoder 将 Mel → waveform

### 关键设计选择

#### 设计选择 1: FSQ 替代 VQ (Supervised Semantic Speech Tokenizer)

**WHY**: CosyVoice 1 的 VQ (codebook size 4096) 仅利用 23% 的码字 [Table 4]。[论文原文] FSQ 通过固定网格量化天然避免 codebook collapse,可实现 100% 利用率。

**HOW** [§2.2, Eq.1-2]:
- 将 FSQ 模块插入 SenseVoice-Large ASR 编码器的 Encoder_1 (6 层 Transformer) 之后
- 中间表征 H 先投影到 D 维低秩空间,每个维度取值范围 [-K, K],做 bounded round
- 量化后投影回原始维度: H_bar = ROUND(Proj_down(H)), H_hat = Proj_up(H_bar)
- 训练时用 STE (Straight-Through Estimator) 近似梯度
- speech token μ_i 通过 (2K+1)-ary 编码计算 [Eq.2]: μ_i = Σ h_bar_{i,j} * (2K+1)^j
- 结果: codebook size 6561,利用率 100% [Table 4]

**Evidence**: FSQ vs VQ 在 SenseVoice-Large encoder 内的对比 [Table 4]:
- Codebook 利用率: FSQ 6561 (100%) vs VQ 963 (23%)
- ASR error rate (CommonVoice EN): FSQ 10.67% vs VQ 18.26%
- ASR error rate (Fluers CN): FSQ 4.43% vs VQ 5.03%
- [agent 解读] FSQ 的 100% 利用率意味着每个 speech token 都承载有效信息区分度,这直接导致了更低的 ASR error,因为更多的 codebook entries 意味着更细粒度的语义表征

#### 设计选择 2: 预训练 LLM 初始化 text-speech LM

**WHY**: [论文原文] 预训练文本 LLM 已具备丰富的语言知识和上下文理解能力,可以直接复用,减少 text-speech LM 从头学习语言结构的负担 [§2.3]。

**HOW** [§2.3]:
- 使用 Qwen2.5-0.5B 作为 backbone
- 移除 CosyVoice 1 的 text encoder 和 speaker embedding (x-vector)
- 发现 speaker embedding 不仅包含说话人身份,还泄漏了语言和副语言信息,损害韵律自然度和跨语言能力 [论文原文]
- 将 speech tokens 简单附加到文本 token 之后训练 next-token prediction

**Evidence**: 模块消融 [Table 7] 逐步验证每个改动的贡献:
- CosyVoice → + LLM init: CER(test-zh) 3.63→2.96 (-18.5%), WER(test-hard) 11.75→9.94 (-15.4%)
- + Drop Spk Emb: CER 2.96→2.56, WER 9.94→9.66 [论文原文] 移除 speaker embedding 后内容错误显著减少,说明 speaker embedding 确实引入了信息泄漏
- + FSQ: CER 2.56→1.45 (-43.4%), WER 9.66→6.83 (-29.3%)
- + Pitch Loss: CER 1.45→1.19, WER 6.83→6.29 (在 tokenizer 训练中加入 pitch loss 约束进一步提升)

#### 设计选择 3: 统一 Streaming/Non-streaming LM

**WHY**: [论文原文] 实际部署需要同时支持两种模式 -- non-streaming 用于离线批量合成,streaming 用于 voice chat 等低延迟场景。统一模型可降低部署成本 [§2.3]。

**HOW** [§2.3, Fig 2]:
- **Non-streaming 模式**: 序列构造为 [S, text_tokens, T, speech_tokens, E],标准自回归
- **Streaming 模式**: 按 N:M 比例交替排列 text 和 speech tokens。如果下一个 token 是文本 token,模型预测 "filling token" (提示拼接 N 个 text tokens);文本用完后添加 T (turn of speech) token,剩余 speech tokens 顺序生成
- 两种模式共用同一 LM,同时训练

**ICL / SFT 推理差异** [§2.3]:
- ICL Non-streaming: [S, prompt_text, text, T, prompt_speech] → 生成至 E
- ICL Streaming: [S, mixed_text_speech, T, remaining_speech] → 每 M 个 token 输出一次
- SFT Non-streaming: [S, text, T] → 直接生成
- SFT Streaming: [S, first_N_text] → 交替生成

#### 设计选择 4: Chunk-aware Causal Flow Matching

**WHY**: [论文原文] 现有 flow matching 模型在 offline 模式下工作,所有 speech tokens 必须全部生成后才能采样 Mel spectrogram,不适合流式场景 [§2.4]。

**HOW** [§2.4, Fig 3]:
- 使用 causal Conv-Transformer UNet 作为 flow matching 的 backbone
- 将多步 flow estimation 视为堆叠的更深网络 (重复 UNet 10 次),使展开后的网络因果化
- 设计四种 attention mask 用于不同延迟需求:
  1. **Non-causal Mask**: 全局注意力,最佳质量,用于 offline
  2. **Full-causal Mask**: 仅关注过去帧,最低延迟
  3. **Chunk-M Mask**: 关注过去帧 + M 帧未来,适合首个 chunk
  4. **Chunk-2M Mask**: 关注过去帧 + 2M 帧未来,适合后续 chunk
- 训练时从四种 mask 中随机采样,实现**隐式自蒸馏** [论文原文]: 拥有更多上下文的 mask 自然充当 teacher,指导上下文更少的 mask

**OT flow 公式** [Eq.3-8]:
- 向量场: ω_t(φ_t^OT(X_0, X_1) | X_1) = X_1 - X_0
- 流路径: φ_t^OT(X_0, X_1) = (1-t)X_0 + tX_1
- UNet 学习: ν_t(φ_t^OT(X_0, X_1)|θ) = UNet_θ(φ_t^OT, t; v, {μ}_{1:L}, X̃_1)
- CFG [Eq.10]: ν̃_t = (1+β) · ν_t(cond) - β · ν_t(uncond), β=0.7, NFE=10
- Cosine scheduler [Eq.9]: t := 1 - cos(½tπ)

#### 设计选择 5: Reinforcement Learning for SFT

**WHY**: [论文原文] SFT 微调后部分说话人的内容一致性 (WER) 可能变差,需要进一步优化 [§2.8]。

**HOW** [§2.8, Eq.13-15]:
- 使用 DPO (Direct Preference Optimization) 进行偏好优化
  - reward function 基于 WER (ASR error) 和 SS (speaker similarity)
  - preferred sample x^w 和 rejected sample x^l 从 TTS 系统反复采样生成
- 使用 differentiable ASR reward (L_ASR):
  - 将 LM 预测的 token μ_i 转回低秩表征 H_bar,通过 ASR backend 重新预测输入文本
  - Gumbel-Softmax 采样使过程可微
  - L_ASR = -log P(Y|H_hat; θ_ASR) [Eq.15]
  - [论文原文] differentiable ASR reward 不需要反复合成音频获取偏好对,效率更高

### 训练策略

- Speech tokenizer: 200K 小时数据 (110K 中文 + 100K 英文) [Table 2]
- CosyVoice 2 主模型: 约 167K 小时 (130K 中文 + 30K 英文 + 4.6K 日文 + 2.2K 韩文) [Table 3]
- Instructed generation: 1500 小时指令数据,支持情感/语速/方言/角色扮演/精细控制 [Table 1]
- mSFT (multi-speaker fine-tuning): 多说话人同时微调,最少 400 条录音即可 [§2.7]

## 实验

| 指标 | 本文 (CosyVoice 2) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER (%) | 2.47 (offline) / 2.45 (streaming) | Human: 2.66, CosyVoice: 2.89 | LibriSpeech test-clean | [Table 5] |
| NMOS | 3.96 (offline) / 3.90 (streaming) | Human: 3.84, CosyVoice: 3.93 | LibriSpeech test-clean | [Table 5] |
| SS | 0.745 (offline) / 0.751 (streaming) | Human: 0.697, CosyVoice: 0.743 | LibriSpeech test-clean | [Table 5] |
| CER (%) test-zh | 1.45 (offline) / 1.45 (streaming) | Seed-TTS: 1.12, MaskGCT: 2.27 | SEED eval | [Table 6] |
| SS test-zh | 0.806 / 0.812 | Seed-TTS: 0.796, MaskGCT: 0.774 | SEED eval | [Table 6] |
| WER (%) test-en | 2.57 / 2.38 | E2 TTS: 2.19, F5-TTS: 1.83 | SEED eval | [Table 6] |
| WER (%) test-hard | 6.83 / 8.08 | Seed-TTS: 7.59, MaskGCT: 10.27 | SEED eval | [Table 6] |
| MOS-I (instructed) | 4.11 | CosyVoice-Instruct: 3.14 | in-house Chinese | [Table 10] |
| CER (%) test-ja | 18.79 | - | CommonVoice JA | [Table 9] |
| CER (%) test-ko | 7.98 | - | CommonVoice KO | [Table 9] |

**关键发现**:

1. **Streaming 几乎无损**: offline vs streaming 在 typical cases (test-zh, test-en) 上几乎无差异 [Table 6, 8]。仅在 challenging cases (test-hard) 上 streaming 略有退化 (6.83→8.08%) [论文原文] 这归因于 streaming 模式下上下文信息的丢失 [§4.3]

2. **人类水平质量**: LibriSpeech 上 CosyVoice 2 的 WER (2.47%) 低于人类录音 (2.66%),NMOS (3.96) 高于人类 (3.84),SS (0.745) 高于人类 (0.697) [Table 5] [论文原文]

3. **FSQ 是最大贡献因子**: 消融中 +FSQ 步骤带来 CER 从 2.56→1.45 的最大跌幅 [Table 7]

4. **RL 微调有效但有 trade-off**: L_ASR + L_DPO 组合在目标说话人上降低 WER (7.15→6.64%) 且提升 SS (0.795→0.796),但在 SEED hard 集上 DPO 反而变差 (7.90→8.31%) [Table 11] [论文原文] 原因是 hard samples 含大量重复,可能在 DPO 中被当作 rejected samples

5. **Streaming FM 的 speaker similarity 略高**: [Table 8] streaming FM 的 SS 略高于 offline FM,可能因为 streaming 首个 chunk 的 prompt-to-generation ratio 更高 [agent 解读]

## 局限性

1. **语言覆盖有限**: 仅支持中英日韩四种语言,且日语因与中文字符集重叠导致性能显著降低 (CER 18.79%) [Table 9, §6]
2. **无法通过指令控制音色**: 指令系统支持情感/语速/方言/角色,但不支持 timbre 控制 (如"用低沉的男声说") [§6]
3. **不支持歌声合成** [§6]
4. **RL 微调的泛化性**: DPO 在 out-of-domain (SEED hard) 上可能变差 [Table 11]
5. **首包延迟分析仅为理论推导**: 未报告实际测量的端到端延迟数据 [§2.5]
6. **消融未覆盖 FSQ 超参数**: D (低秩维度) 和 K (每维取值范围) 的选择未做消融

## 点评

CosyVoice 2 是一篇工程驱动的系统论文,其价值在于将多个正确的设计选择整合到一个统一框架中。最核心的贡献是 FSQ tokenizer -- 从 VQ 到 FSQ 的替换带来了消融中最大的性能提升 [Table 7],且几乎不增加系统复杂度。这与 FSQ 概念页中记录的"FSQ 天然避免 codebook collapse"完全吻合。

统一 streaming/non-streaming 的方案设计精巧: 四种 attention mask 的随机训练实现了隐式自蒸馏,这是一个值得借鉴的通用技巧 -- 在训练时混合不同"难度"的任务,让简单任务的梯度帮助难任务。

相对薄弱的部分是科学分析: 移除 speaker embedding 的动机 ("包含语言和副语言信息,损害韵律") 缺乏定量验证; RL 微调的效果混合 (目标域有效,OOD 可能变差) 但论文未深入分析根因。

## 可复用的 idea

1. **FSQ 替代 VQ 用于 speech tokenizer**: 一行代码级改动 (round 替代 codebook lookup),即可获得 100% codebook 利用率和更低 ASR error。适用于任何使用 VQ 的 speech tokenizer。

2. **文本 LLM 初始化 text-speech LM**: 直接复用 Qwen/LLaMA 等预训练 LLM 作为 text-speech LM 的初始化,省去从头训练语言理解的成本,CER 可降低 ~18%。

3. **四种 mask 统一训练实现隐式自蒸馏**: 将 non-causal/full-causal/chunk-M/chunk-2M 四种 attention mask 混合训练,自然形成 teacher-student 关系。可推广到任何需要同时支持多种推理模式的模型。

4. **Differentiable ASR reward**: 将 speech tokenizer 的 ASR backend 冻结后直接用 log P(Y|H_hat; θ_ASR) 作为可微 reward,配合 Gumbel-Softmax 实现端到端 RL 微调,避免了传统 DPO 需要反复合成音频的高成本。

5. **Multi-speaker fine-tuning (mSFT)**: 同时微调多个说话人 (而非逐个) 以保留预训练模型的韵律和发音覆盖,用 speaker-prompt tag 区分不同说话人。最少 400 条录音即可获得良好效果 [§4.6]。

---

检索命中: [[SpeechTokenizer]], [[ConditionalFlowMatching]], [[ResidualVectorQuantization]], [[SemanticvsAcousticTokens]] | 过滤: [[FiniteScalarQuantization]](pending-review), [[Classifier-FreeGuidance]](pending-review), [[CodecLanguageModel]](pending-review) | 未命中但可能相关: 无


> [!review] 审阅 (2026-06-09, batch-auto)
> **结论**: pass
> 
> 结构检查: 速查卡片 ✓ | 方法 ✓ | 实验 ✓ | KB背景 ✓
> 详细审阅待后续安排

## 代码级分析

> [!info] 代码来源
> - 仓库: https://github.com/FunAudioLLM/CosyVoice
> - commit: 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
> - 分析日期: 2026-06-10
> - 备注: CosyVoice2 对应 `Qwen2LM` 类 + `CausalMaskedDiffWithXvec` 类,配置文件 `cosyvoice2.yaml`

### 架构验证

论文 Fig 1 的架构与代码对应关系:

| 论文组件 | 代码实现 | 文件位置 |
|---------|---------|---------|
| Text-Speech LM (Qwen2.5 init) | `Qwen2LM` 继承 `TransformerLM`,内部 `llm = Qwen2Encoder(Qwen2ForCausalLM)` | `cosyvoice/llm/llm.py:257-296` |
| FSQ Speech Tokenizer | 外部 ONNX `speech_tokenizer_v2.onnx` (或 batch ONNX) | `cosyvoice/cli/cosyvoice.py:156` |
| Chunk-aware Causal CFM | `CausalMaskedDiffWithXvec` + `CausalConditionalCFM` | `cosyvoice/flow/flow.py:149`, `flow_matching.py:196` |
| Vocoder | 同 CosyVoice v1 HiFi-GAN | `cosyvoice/hifigan/` |

**关键差异 vs 论文**:
1. 论文说移除了 text encoder 和 speaker embedding,代码验证: `Qwen2LM` 确实没有 `text_encoder`/`text_encoder_affine_layer`/`spk_embed_affine_layer`,text token 直接通过 `self.llm.model.model.embed_tokens(text_token)` 获取 Qwen2 的 word embedding (`llm.py:366-367`)
2. 论文说使用 Qwen2.5-0.5B 初始化: 代码中 `Qwen2Encoder.__init__` 从 `pretrain_path` 加载 `Qwen2ForCausalLM`,配置 `cosyvoice2.yaml` 中指定 `CosyVoice-BlankEN` 目录 (`cosyvoice.py:150`)

### 论文未写的实现细节

1. **Bistream (流式) 序列构造** (`cosyvoice/llm/llm.py:302-349`): 训练时以 50% 概率选择 bistream 模式,将 text 和 speech 按 `mix_ratio=[5,15]` 交替排列。具体实现: 每 5 个 text token 后接 15 个 speech token,末尾组不满时追加 task_id + 剩余 speech + EOS。`fill_token = speech_token_size + 2` 用于标记需要填充 text token 的位置。

2. **LLM 输出 vocab 扩展** (`cosyvoice/llm/llm.py:275-277`): CosyVoice2 的 `llm_decoder` 输出 `speech_token_size + 3` 维,额外 3 个 token 为: EOS(`speech_token_size`), unused(`speech_token_size+1`), fill_token(`speech_token_size+2`)。stop_token_ids 包含这三个。

3. **DPO 训练的实现** (`cosyvoice/llm/llm.py:407-456`): `forward_dpo` 方法将 chosen 和 rejected speech tokens 在 batch 维度拼接,一次前向计算两组 logits,然后计算 per-sample log probability 用于 DPO loss。这是标准的 offline DPO 实现。

4. **vLLM 加速** (`cosyvoice/llm/llm.py:506-534`): 推理时支持 vLLM 后端,将 lm_input 的 embedding 传入 vLLM engine (通过 `enable_prompt_embeds` 参数),实现高效批量推理。

5. **Bistream 推理** (`cosyvoice/llm/llm.py:552-661`): 流式推理 `inference_bistream` 接受 text 作为 Generator (逐步产生 text chunk),每收到足够 text token (5 个) 就拼接到 LLM 输入并解码 15 个 speech token,直到收到 fill_token 则暂停等待下一组 text。

6. **Causal CFM 的统一训练** (`cosyvoice/flow/flow.py:201`): 训练时以 50% 概率选择 streaming 模式 (`streaming = True if random.random() < 0.5 else False`),这对应论文中"四种 mask 随机采样"的简化实现(代码中只有 streaming/non-streaming 二选一)。

7. **Causal CFM 固定随机噪声** (`cosyvoice/flow/flow_matching.py:199-200`): `CausalConditionalCFM` 在初始化时预生成一个固定的随机噪声张量 `rand_noise = torch.randn([1, 80, 50*300])`,推理时从中截取而非实时生成。这保证了流式生成时不同 chunk 使用相同的噪声,避免拼接处的不连续。

8. **Pre-lookahead 机制** (`cosyvoice/flow/flow.py:159,261`): `pre_lookahead_len=3` 表示流式推理时额外向前看 3 个 token (约 120ms),这是论文中 chunk-M mask 的实际实现参数。

9. **流式推理的 token hop** (`cosyvoice/cli/model.py:258`): `token_hop_len=25`(对应 1 秒的 speech token),且使用 `stream_scale_factor=2` 逐步增大 hop 长度到 `token_max_hop_len=100`,前几个 chunk 更短以降低首包延迟。

### 训练 pipeline 拆解

```
原始音频
  → speech_tokenizer_v2.onnx (FSQ) → speech_token (codebook=6561)
  → mel_spectrogram → speech_feat (80-dim)
  → campplus.onnx → embedding (192-dim, 仅 CFM 使用)
  → Qwen2.5 BPE tokenizer → text_token

训练:
  text_token → Qwen2.embed_tokens → text_token_emb
  speech_token → speech_embedding → speech_token_emb
  
  50% 概率 unistream: [sos, text_emb, task_id, speech_emb]
  50% 概率 bistream: [sos, text_5, speech_15, text_5, speech_15, ..., task_id, remaining_speech]
  
  LLM: Qwen2ForCausalLM forward → hidden_states[-1]
  → llm_decoder (Linear → 6564) → LabelSmoothingLoss
  
  CFM: 50% streaming / 50% non-streaming
  speech_token → input_embedding → causal_encoder → encoder_proj → h
  条件: h + spk_embedding + masked_mel
  → CausalConditionalCFM.compute_loss → MSE on velocity
```

### 推理 pipeline 拆解

```
输入文本 + prompt 音频
  → prompt 音频 → speech_tokenizer_v2 → prompt_speech_token
  → prompt 音频 → mel_extractor → prompt_feat
  → prompt 音频 → campplus → flow_embedding (仅 CFM 使用)
  
  Stage 1 (LLM, 异步线程):
    Non-streaming: [sos, text_emb(prompt+input), task_id, prompt_speech_emb] → AR decode
    Streaming: bistream 交替生成,每收到 5 text → 输出 15 speech
    top-k=25, min_ratio=2, max_ratio=20
    
  Stage 2 (CFM, 主线程):
    等待 token_hop_len (25) + pre_lookahead_len (3) 个 token
    token → causal_encoder → repeat_interleave(token_mel_ratio=2) → h
    prompt_mel 作为条件
    固定噪声 + 10 步 Euler ODE + cosine scheduler + CFG(0.7)
    → mel (streaming: 逐 chunk 生成)
    
  Vocoder: mel → HiFi-GAN → waveform
  后处理: fade_in_out 消除 chunk 拼接噪声
```

### 关键超参数表

| 参数 | 论文值 | 代码实际值 | 备注 |
|------|--------|-----------|------|
| FSQ codebook size | 6561 | 6561 (vocab_size) | `speech_token_size` 在 YAML 配置 |
| LLM backbone | Qwen2.5-0.5B | Qwen2ForCausalLM | 从 `CosyVoice-BlankEN` 加载 |
| LLM output vocab | 未明确 | 6564 (6561+3) | speech + EOS + unused + fill |
| Mix ratio (bistream) | N:M | 5:15 | `mix_ratio=[5,15]` |
| CFM steps | 10 | 10 | `n_timesteps=10` |
| Token-mel ratio | 未明确 | 2 (1 token = 2 mel frames) | `token_mel_ratio` |
| Pre-lookahead | 未明确 | 3 tokens | `pre_lookahead_len=3` |
| Stream token hop | 未明确 | 25 (初始) → 100 (最大) | `token_hop_len`, `token_max_hop_len` |
| Stream scale factor | 未明确 | 2x | 每次 hop 翻倍 |
| Silent token 过滤 | 未提及 | 最多连续 5 个 | `max_silent_token_num=5` |

### 复现 checklist (基于代码)

- [ ] 环境依赖: 同 CosyVoice + vllm (可选), Qwen2.5 预训练权重
- [ ] 数据准备: speech_tokenizer_v2 (FSQ ONNX), mel, speaker embedding, Qwen BPE tokens
- [ ] 预训练模型依赖: Qwen2.5-0.5B (CosyVoice-BlankEN), speech_tokenizer_v2.onnx, campplus.onnx, HiFi-GAN
- [ ] 训练命令: `cosyvoice/bin/train.py` + cosyvoice2.yaml
- [ ] 推理命令: `CosyVoice2(model_dir).inference_zero_shot(text, prompt_text, prompt_wav, stream=True)`
- [ ] 已知坑: (1) FSQ tokenizer 训练代码缺失; (2) DPO 训练需要额外构造 reject samples; (3) vLLM 需要自定义 embedding 输入支持

### 代码质量与可复现性评估

- **工程质量**: 4/5 - 继承 CosyVoice 的良好结构,bistream 逻辑较复杂但有详细日志
- **文档完善度**: 3/5 - 缺少 bistream 训练和 DPO 微调的详细文档
- **社区活跃度**: 5/5 - 与 CosyVoice 共享仓库,持续维护
- **复现难度**: 3/5 - 推理易复现,流式推理需要理解 bistream 协议;训练需要 FSQ tokenizer
