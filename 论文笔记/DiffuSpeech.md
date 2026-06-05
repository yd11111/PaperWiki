---
type: paper
tier: deep
title: "DiffuSpeech: Silent Thought, Spoken Answer via Unified Speech-Text Diffusion"
arxiv_id: "2601.22889"
source: "Sources/DiffuSpeech.pdf"
authors: [Yuxuan Lou, Ziming Wu, Yaochen Wang, Yong Liu, Yingxuan Ren, Fuming Lai, Shaobing Lian, Jie Tang, Yang You]
year: 2026
venue: "Preprint (arXiv)"
tags: [speech-LM, diffusion, masked-diffusion, speech-to-speech, chain-of-thought, reasoning, multimodal, non-autoregressive]
concepts: ["[[SpeechLanguageModel]]", "[[MaskedGenerativeModeling]]", "[[Diffusion-basedTTS]]", "[[SemanticvsAcousticTokens]]", "[[NeuralVocoder]]", "[[LLM-basedTTS]]"]
models: ["[[论文笔记/Moshi|Moshi]]", "[[论文笔记/LLaMA-Omni2|LLaMA-Omni 2]]", "[[论文笔记/LLaDA-TTS|LLaDA-TTS]]", "[[论文笔记/CoT-ST|CoT-ST]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[SpeechLanguageModel]], [[SemanticvsAcousticTokens]], [[NeuralVocoder]], [[LLM-basedTTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]]✓, [[SemanticvsAcousticTokens]]✓, [[NeuralVocoder]]✓, [[LLM-basedTTS]]✓ | 过滤: [[Diffusion-basedTTS]](pending-review), [[MaskedGenerativeModeling]](pending-review) | 未命中但可能相关: 无

**谱系定位**: DiffuSpeech 是首个将 masked diffusion language model (MDLM) 扩展到 speech-text 双模态的统一生成系统,同时支持语音理解和生成。在 [[SpeechLanguageModel]] 的分类体系中,它属于 "continued pre-training + instruction-tuning" 训练路线,但独特之处在于用 diffusion (而非自回归) 作为生成范式 -- 此前所有 SpeechLM (GSLM, SpiritLM, Moshi, SpeechGPT, Mini-Omni) 均为 autoregressive。在 [[SemanticvsAcousticTokens]] 的光谱中,DiffuSpeech 使用 HuBERT semantic tokens (25Hz, 500 codes),属于纯 semantic token 路线,声学细节由 frozen HiFi-GAN [[NeuralVocoder]] 恢复。与 [[LLM-basedTTS]] 中主流 AR 路线 (VALL-E 系列, CosyVoice) 的最大区别在于: DiffuSpeech 通过双向 attention + 迭代去噪实现 speech 和 text 的联合生成,而非单向左到右。

**创新判断对比**: DiFFA (Zhou et al., 2025) 是此前最接近的工作,将 diffusion 用于 speech-to-text 理解,但不支持语音生成。DiffuSpeech 补全了 speech generation 这一环。在 reasoning 方面,TARS (Wang et al., 2026) 探索了 SpeechLM 的 CoT 推理,但只输出文本; DiffuSpeech 的 "Silent Thought, Spoken Answer" 范式首次实现了 text reasoning + speech reply 的联合生成。

## 速查

> [!summary] 速查
> - **一句话**: 首个 diffusion-based speech-text LM,通过 masked diffusion 联合生成文本推理链和语音回复,实现 "Silent Thought, Spoken Answer"
> - **路线**: Speech (HuBERT 25Hz → 500 discrete tokens) + Text (LLaDA 128K vocab) → Unified MDLM backbone (masked diffusion, selective masking) → Text reasoning + Speech tokens → HiFi-GAN vocoder → Waveform
> - **指标**: S→S QA: LlamaQ 68.5% (+4.3 vs Moshi), WebQ 49.7% (+9.0 vs Moshi) [Table 2]; TTS WER: 6.2% LS-Clean (best among generative models) [Table 3]; MMLU 66.2% (+0.3 vs LLaDA base) [Table 4]
> - **可借鉴**: (1) Modality-specific selective masking: 仅 mask target 不 mask condition,让不同模态共享一个 diffusion backbone; (2) "Silent Thought" 范式: text CoT traces 不发声但指导 speech 生成质量; (3) Confidence-based unmasking schedule: 先填高确信 token 后填难 token,类似 MaskGIT 但用于 speech-text 混合序列
> - **局限**: HuBERT 25Hz/500 codes 语音质量上限受限 (无 speaker similarity 评估); ThinkingTalk 数据集仅 26K 样本且用合成语音; 未评估延迟/RTF; 仅支持英语

## 核心问题

DiffuSpeech 试图解决两个问题:

1. **Speech LLM 缺乏显式推理**: 现有 SpeechLM 直接从输入语音生成输出语音,没有 chain-of-thought 推理过程。语音生成是不可逆的 -- 一旦产生错误音频无法回退修正 [§1]。文本 LLM 已证明 CoT 推理显著提升答案质量,但 SpeechLM 尚未具备这一能力。

2. **自回归范式对 speech 的固有局限**: 所有现有 SpeechLM 都是自回归的,只能左到右顺序生成,无法利用双向上下文 [§1]。这对语音尤其不利,因为韵律和内容天然纠缠 -- 一个词的发音可能取决于后续内容 [论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DiffuSpeech 的架构基于 LLaDA-8B-Instruct (一个已有的 masked diffusion text LLM),扩展为 speech-text 双模态 [Fig 2]:

1. **Speech Encoder**: Frozen HuBERT-base (pretrained on 960h LibriSpeech) + Linear Quantizer (500 codes, 25Hz) [§3.1, §A.2]。沿用 SpiritLM 的 tokenization pipeline。
2. **Unified Vocabulary**: V = V_text (128K from LLaDA) ∪ V_special (9 tokens) ∪ V_speech (500 tokens),总计 126,973 tokens [Table 7]。Speech token ID 偏移以避免碰撞。
3. **Backbone**: LLaDA-8B 的 masked diffusion Transformer,对统一词表上的混合序列执行 masked diffusion。
4. **Vocoder**: Frozen HiFi-GAN,从 HuBERT tokens 重建波形 [§A.2]。

序列格式 (以核心 S2S 任务为例) [§3.1, Eq. 2]:
```
x = [τ_s2s, <|sos|>, s_user, <|eos|>, <|sot|>, t_think, <|eot|>, <|sos|>, s_reply, <|eos|>]
```
其中 `t_think` (文本推理) 和 `s_reply` (语音回复) 都是 target,`s_user` 是 condition。

### 关键设计选择

**为什么用 masked diffusion 而非 AR?** [论文原文] 作者认为 AR 模型的单向生成无法捕捉 speech 的全局依赖 (韵律与内容纠缠),而 masked diffusion 的双向 attention 允许 thinking traces 和 speech tokens 在生成过程中相互 inform [§1, §3.2]。实验验证: diffusion 在 ASR/TTS 上最终 WER 低于 AR (7.1% vs 11.4% ASR, 10.5% vs 14.7% TTS),尽管 AR 前期收敛更快 [Fig 5]。[agent 解读] 这一优势可能来自 masked diffusion 的多步 refinement -- 模型可以在后续步骤中修正早期错误,而 AR 的错误会累积。

**Selective Masking** [§3.3, Eq. 7]: 只对 target (thinking + reply) 应用 masking,condition (user input) 保持 unmasked。[论文原文] 这确保模型学习正确的条件生成 p(y|c, τ)。[agent 解读] 这是 MDLM 用于条件生成的标准做法,与 MaskGCT 中只 mask 目标序列的策略一致,关键是将其扩展到了 speech-text 混合序列。

**Modality-specific masking schedules**: 沿用 MMaDA (Yang et al., 2025b) 的做法,对不同模态使用不同的 masking schedule [§3.2]。使用 cosine schedule: γ(t) = cos(π/2 · (1-t)) [Eq. 6],从完全 mask (t=1) 平滑过渡到完全 unmask (t=0)。

**Confidence-based unmasking** [§3.4, Algorithm 1]: 推理时每步预测所有 masked positions 的 token,选择 confidence 最高的 top-k_i 个 unmask,k_i = ⌈n·(T-i+1)/T⌉ 按线性 schedule 递增。[论文原文] 这让模型先确定容易的 token,再 refine 困难的 [§3.4]。[agent 解读] 这本质上是 MaskGIT-style 的迭代解码,但首次应用于 speech+text 混合序列。

**为什么 HuBERT 而非 codec?** [agent 解读] 论文未明确解释,但从设计看: (1) HuBERT 25Hz + 500 codes 产生极短序列,降低 diffusion 的计算负担; (2) semantic tokens 与文本对齐更好,有利于 text-speech 联合 diffusion; (3) 声学细节由 frozen HiFi-GAN 恢复,解耦了语义建模和声学重建。这与 SpiritLM 的 tokenization 选择一致。

### 训练策略

**两阶段训练** [§3.3, Fig 2b]:

| 阶段 | 目标 | 任务比例 | 数据 | 配置 |
|------|------|----------|------|------|
| Stage 1: Speech-Text Alignment | 学习 speech-text 对应关系 | ASR 40%, TTS 40%, LM 20% | LibriHeavy (29K hrs), VoxPopuli, CommonVoice, RefinedWeb | 32×H20, 22 天, lr=1e-5 [Table 9] |
| Stage 2: Instruction Following | 学习 "Silent Thought, Spoken Answer" | S2S 30%, S2T 30%, T2T 20%, ASR 10%, TTS 10% | ThinkingTalk (26K), CommonVoice, VoxPopuli | 8 GPU, lr=5e-6 [Table 10] |

[论文原文] Stage 1 中保留 20% text LM 任务是为了防止灾难性遗忘 [§3.3]。

**ThinkingTalk 数据集构建** [§4]:
- 来源: Smoltalk (88.1%, 实用知识), SoundMind (5.6%, 逻辑推理) [Table 1]
- 三阶段 pipeline: (1) Qwen3-32B 改写为口语风格 + 添加 thinking traces; (2) LLM judge 7 维度过滤 (保留率 81.7%); (3) MegaTTS3 合成用户语音 (4 speakers) + Qwen3-Omni 合成助手语音 (1 speaker) [Fig 3]
- 规模: 26,387 samples, 319 hours, thinking 均 77.5 words, reply 均 102 words [Table 1]

**推理配置**: 64 diffusion steps, 线性 unmasking schedule, temperature 1.0 [Table 11]。

## 实验

| 指标 | 本文 (DiffuSpeech) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| S→S Accuracy | 68.5% | Moshi 64.2%, MinMo 63.8% | LlamaQ | [Table 2] |
| S→S Accuracy | 33.5% | Moshi 30.5%, MinMo 25.5% | TriviaQA | [Table 2] |
| S→S Accuracy | 49.7% | Moshi 40.7%, MinMo 39.9% | WebQ | [Table 2] |
| S→T Avg | 57.6% | MinMo 54.0% (+3.6) | 4 benchmarks | [Table 2] |
| S→S Avg | 50.6% | Moshi 45.1% (+5.5) | 3 benchmarks | [Table 2] |
| TTS WER | 6.2% | MinMo 6.7%, Moshi 7.0% | LS-Clean | [Table 3] |
| TTS WER | 10.3% | MinMo 10.9%, Moshi 10.6% | VoxPopuli | [Table 3] |
| ASR WER | 3.0% | MinMo 1.8%, Moshi 5.5% | LS-Clean | [Table 3] |
| MMLU | 66.2% | LLaDA 65.9% (+0.3) | MMLU | [Table 4] |
| TriviaQA (text) | 60.3% | LLaDA 55.6% (+4.7) | TriviaQA | [Table 4] |
| GSM8K | 72.8% | LLaDA 70.3% (+2.5) | GSM8K | [Table 4] |
| MMSU (speech reasoning) | 39.0% | MinMo 43.2%, Moshi 24.0% | MMSU | [Table 15] |

**关键发现**:

1. **S→S 退化更小**: 其他模型从 S→T 到 S→S 大幅退化 (MinMo 平均降 14 points),DiffuSpeech 退化更小 [Table 2]。[论文原文] 这归因于 diffusion 的联合生成 -- thinking traces 和 speech tokens 在去噪过程中相互 inform [§5.2]。

2. **Speech 训练不损害 text 能力**: DiffuSpeech 在所有 3 个 text benchmarks 上轻微超越 base model LLaDA [Table 4]。[论文原文] 多任务 speech-text 训练甚至可能 benefit text understanding [§5.2]。

3. **Thinking traces 对 diffusion 增益更大**: 添加 thinking traces 后,DiffuSpeech 平均提升 +13.4 points vs SpiritLM +10.5 points [Table 5]。[论文原文] 这可能因为 diffusion 的联合生成能更好地利用 reasoning traces [§5.3]。

4. **Diffusion 后期超越 AR**: 训练前期 AR 收敛更快,但 ~10-15K steps 后 diffusion 反超,最终 WER 大幅领先 [Fig 5]。

5. **Sample efficiency**: 减少 denoising steps 对性能影响温和 -- 512 steps (2x 加速) 时 TTS WER 反而改善 (6.20% vs 8.45%),SpeechQA 几乎不变 (72.06% vs 72.13%) [Table 6]。

## 局限性

1. **语音质量受限**: HuBERT 25Hz + 500 codes + HiFi-GAN 的语音质量上限远低于现代 codec-based 系统 (如 CosyVoice 的 FSQ 或 EnCodec 的 RVQ)。论文未报告 MOS、speaker similarity 或 PESQ 等感知指标,仅用 WER 衡量 TTS 质量,这可能掩盖了合成语音的自然度问题。
2. **合成数据**: ThinkingTalk 完全由合成语音构成 (MegaTTS3 + Qwen3-Omni),仅 4 个参考说话人,多样性有限。模型是否能泛化到真实人声场景未验证。
3. **未报告延迟**: 64 步 diffusion 的推理速度未与 AR baseline 对比。虽然 Table 6 展示了步数减少的影响,但未给出实际 RTF 或 wall-clock time。
4. **仅英语**: 所有实验均为英语,跨语言能力未验证。
5. **Baseline 公平性**: 部分 baseline (如 SpiritLM) 模型规模和训练数据量显著小于 DiffuSpeech (基于 8B LLaDA + 29K hrs speech data)。
6. **无 speaker control**: 系统不支持零样本声音克隆或说话人控制,这对实际部署是关键缺失。

## 点评

**优势**:
- **范式开创性**: 首次将 masked diffusion LLM 扩展到 speech generation,打破了 SpeechLM = AR 的假设。实验有力证明 diffusion 可以且在某些指标上优于 AR。
- **"Silent Thought" 范式有实验支撑**: thinking traces 对两种架构 (AR, diffusion) 都有增益,且 diffusion 获益更多 [Table 5],这不是 cherry-pick 而是系统性验证。
- **Text 能力保持良好**: 多模态训练后 MMLU/TriviaQA/GSM8K 均不降反升,说明 selective masking + task proportioning 策略有效。
- **Ablation 充分**: AR vs Diffusion 训练动态、thinking traces 效果、denoising steps 效率,三个 ablation 都有明确结论。

**不足**:
- **语音质量评估不完整**: 只报 WER 不报 MOS/PESQ/speaker-sim,无法判断合成语音是否真正可用。HuBERT+HiFi-GAN 的音质上限是已知瓶颈。
- **"首个" claim 需精确限定**: 论文称 "first diffusion-based speech-text LLM supporting both understanding and generation",但 LLaDA-TTS 已将 LLaDA 用于 TTS。DiffuSpeech 的 "首个" 更准确地说是 "首个同时支持 understanding + generation 的 diffusion SpeechLM"。
- **Scale 问题**: ThinkingTalk 仅 26K 样本,相比 SpeechGPT 的 instruction data 或 Moshi 的 7M hours 极小。结果能否 scale up 有待验证。
- **推理效率未充分分析**: 虽然 diffusion 可以并行 unmask 多个 token,但每步都需完整前向传播。与 AR 的 KV cache 相比,实际 wall-clock time 可能并不占优。

## 可复用的 idea

1. **Selective masking for conditional multimodal generation**: 在 MDLM 框架中,只 mask target modality,保持 condition modality unmasked。这是一个简洁的条件生成适配方式,可迁移到任何需要条件生成的 diffusion LM 场景 (如 image captioning、code generation)。

2. **"Silent Thought" as training paradigm**: 在 speech output 之前插入不发声的 text reasoning,训练时 jointly generate,推理时只输出 speech。这是一种通用的 "internal monologue" 训练策略,可以应用于任何 speech generation 任务来提升质量,不依赖于 diffusion 架构。

3. **Synthetic dataset with LLM-judge quality filtering**: 用 Qwen3-32B 改写 + 7 维度 LLM judge 过滤 (correctness 和 oral suitability 加权更高) 的 pipeline,retention 81.7%,是一个可复用的高质量口语 QA 数据集构建方案。

4. **Diffusion training dynamics insight**: AR 前期收敛快、diffusion 后期超越的 crossover 现象 [Fig 5] 提示: 对 diffusion speech models 要有耐心,不要因前几千步的落后就放弃。这对资源分配和训练策略决策有实际参考价值。

5. **Step reduction without quality loss**: 512 steps vs 1024 steps 几乎无损甚至改善 [Table 6],说明 diffusion speech models 可以用远少于理论需要的步数,实际部署时可大幅加速。

## 审阅

> [!review] 审阅 (pending)
> 待 dispatch reviewer subagent

检索命中: [[SpeechLanguageModel]], [[SemanticvsAcousticTokens]], [[NeuralVocoder]], [[LLM-basedTTS]] | 过滤: [[Diffusion-basedTTS]](pending-review), [[MaskedGenerativeModeling]](pending-review) | 未命中但可能相关: 无
