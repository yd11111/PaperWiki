---
type: paper
tier: deep
title: "FlexiVoice: Enabling Flexible Style Control in Zero-Shot TTS with Natural Language Instructions"
arxiv_id: ""
source: "Sources/FLEXIVOICE.pdf"
authors: [Anonymous]
year: 2026
venue: "Under review at ICLR 2026"
tags: [TTS, instruction-following, style-control, zero-shot, DPO, GRPO, disentanglement, progressive-post-training, emotion-control, LLM-TTS]
concepts: ["[[LLM-based TTS]]", "[[Conditional Flow Matching]]", "[[Speech Tokenizer]]", "[[Instruction-Guided Speech Synthesis]]", "[[Natural Language Description for TTS]]", "[[Style Transfer in TTS]]", "[[Emotion Control in TTS]]", "[[Differentiable Reward Optimization]]"]
models: []
tasks: []
datasets: ["[[Emilia]]"]
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[LLM-based TTS]], [[Conditional Flow Matching]], [[Speech Tokenizer]])
> 检索命中: [[LLM-based TTS]]✓, [[Conditional Flow Matching]]✓, [[Speech Tokenizer]]✓ | 过滤: [[Instruction-Guided Speech Synthesis]](pending-review), [[Natural Language Description for TTS]](pending-review), [[Style Transfer in TTS]](pending-review), [[Emotion Control in TTS]](pending-review), [[Differentiable Reward Optimization]](pending-review) | 未命中但可能相关: 无

**[[LLM-based TTS]]**: FlexiVoice 属于 LLM-based TTS 家族, 以 LLM 为核心生成 discrete speech tokens, 然后通过 flow matching 转为 mel + vocoder 合成波形。与 CosyVoice 2 架构相似, 但重点在于 instruction-following 能力的 post-training。LLM-based TTS 的典型问题——难以精确控制细粒度属性——正是 FlexiVoice PPT 框架要解决的。

**[[Conditional Flow Matching]]**: FlexiVoice 使用 flow matching 将 speech tokens 转换为 mel-spectrogram, 再通过 vocoder 合成波形 [§3]。这与 CosyVoice 系列的 token→CFM→vocoder pipeline 一致。

**[[Speech Tokenizer]]**: FlexiVoice 使用 frozen speech tokenizer 将语音转为 discrete tokens 作为 LLM 的训练目标 [§3.1]。Tokenizer 的选择直接决定了下游 style 信息的编码能力。

## 速查

> [!summary] 速查
> - **一句话**: 首个通过 Progressive Post-Training (DPO→Decoupling GRPO→Instruction GRPO) 系统性解决 zero-shot TTS 中 Style-Timbre-Content conflict 的 instruction-following TTS 系统 [论文原文]
> - **路线**: Text + Instruction + Reference Speech → LLM (pre-trained on Emilia + FlexiVoice-Instruct) → Speech Tokens → Flow Matching → Mel → Vocoder → Waveform; Post-training: S1 (DPO on emotion) → S2 (Decoupling GRPO with r_ser + r_sv rewards) → S3 (Instruction GRPO with ALM reward) [§3, Fig.1]
> - **指标**: TO-Easy ACC-I: 97.4% EN / 99.8% ZH (vs CosyVoice2 baseline N/A); TO-Hard ACC-I: 89.4% (vs VoxInstruct 17.8%); TR-Easy ACC-I: 89.4% EN / 81.8% ZH with SV 91.0% / 98.8%; InstructTTSEval Avg: 79.3 EN / 70.8 ZH (vs Gemini-pro 80.3 / 84.8, MiMo-Audio 72.6 / 70.5); CMOS up to +0.9 vs FlexiVoice-Base [Table 2, 3, 4]
> - **可借鉴**: (1) Progressive Post-Training curriculum (easy emotions → hard disentanglement → complex instructions), 证明顺序至关重要 (逆序 Avg 54.7 vs 正序 88.7) [Table 5]; (2) Multi-objective GRPO 用 SER + SV rewards 实现 style-timbre 解耦 [§3.2.2]; (3) FlexiVoice-Instruct 4316hrs 指令-语音数据集构建方法 (LLM metadata annotation) [§4]
> - **局限**: ACC-T 非零 (6.6% EN TO-Hard) 说明文本情感泄漏未完全消除; WER/CER 略高于 base model (style 代价) [Table 3]; 未开源; 依赖 SER/SV 外部 reward models

## 核心问题

在 zero-shot TTS 中, 实现灵活风格控制面临一个核心矛盾——**Style-Timbre-Content Conflict** [§1]:

1. **Timbre leakage**: 标准监督训练下, 模型过度依赖参考语音的声学 prior, 忽略指令中的 style 信息 [§1] [论文原文]
2. **Content leakage**: 模型从文本推断 prosody (如 "I'm so sad" 自带悲伤语调), 忽略指令 (如 "用快乐的声音") [§1] [论文原文]
3. **Instruction-reference conflict**: 当指令风格和参考语音风格矛盾时, 模型无法正确解耦 [§1] [论文原文]

现有 instruction-based TTS (VoxInstruct, PromptTTS, Parler-TTS) 要么不支持参考语音 (仅 text-only), 要么无法在指令与参考风格矛盾时正确解耦 [§2] [论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

FlexiVoice = LLM core + frozen speech tokenizer + flow matching vocoder [§3]:

```
Pre-training:
  Text + Instruction ("Speak the following text") → LLM input template
  GT Speech → Speech Tokenizer (frozen) → Target tokens
  LLM: predict target tokens from (text, instruction, [optional reference tokens])
  Training: only LLM core, other modules frozen [§3.1]

Post-training (Progressive Post-Training / PPT):
  S1: Multi-modality DPO (emotion instructions) [§3.2.1]
  S2: Decoupling GRPO (conflicting reference + instruction) [§3.2.2]
  S3: Instruction GRPO (complex open-ended instructions) [§3.2.3]
```

### 关键设计选择

**1. Pre-training: FlexiVoice-Base [§3.1]**

基于 Emilia (He et al., 2024) + FlexiVoice-Instruct (自建 4316hrs) 预训练。LLM core 在 text + instruction + optional reference speech 条件下预测 speech tokens。仅训练 LLM 参数, tokenizer 和 flow matching 模块冻结 [§3.1]。默认指令 "Speak the following text" 用于无显式指令的数据 [§3.1]。

**2. S1: Multi-modality DPO [§3.2.1]**

解决核心问题: 让模型同时处理 instruction 和 reference speech, 产生指令一致的情感语音。

方法: 使用 SER 数据集 (ESD, Zhou et al., 2021) 构造 preference pairs:
- Winner (y_w): 同说话人的目标情感语音
- Loser (y_l): 同说话人的不同情感语音
- Input (x): instruction template "Use {label} emotion to read it" + text + reference [§3.2.1]

DPO loss 直接对齐, 不需要显式 reward model [§3.2.1]:
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[\log \sigma \left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**3. S2: Decoupling GRPO [§3.2.2]**

解决核心问题: 当参考语音或文本的情感与指令矛盾时, 模型应遵从指令而非被干扰。

方法: 构造冲突场景 (如 Happy 指令 + Sad 参考), 使用 multi-objective GRPO:
- **r_ser**: Emotion2vec-Large SER 概率分数 → style 约束 (罚 style leakage from ref/text) [§3.2.2]
- **r_sv**: CAM++ speaker verification binary → timbre 约束 (保证 speaker identity) [§3.2.2]

Multi-objective advantage [§3.2.2]:
$$A^i_{\text{emo}} = \frac{r^i_{\text{ser}} - \text{mean}(r^i_{\text{ser}})}{\text{std}(r^i_{\text{ser}})} + \frac{r^i_{\text{sv}} - \text{mean}(r^i_{\text{sv}})}{\text{std}(r^i_{\text{sv}})}$$

为什么 S2 要在 S1 之后: DPO (S1) 建立了基本的 instruction-emotion 对齐; S2 在此基础上通过冲突场景强化解耦。如果直接做 S2, 模型缺乏基本对齐能力, GRPO reward 信号太稀疏 [Table 5 消融] [论文原文]。

**4. S3: Instruction GRPO [§3.2.3]**

解决核心问题: 将 S1-S2 的情感解耦能力泛化到复杂、开放式指令 (gender, pitch, texture, speed, accent, tone, personality 等 12 种属性)。

方法:
- 使用 Kimi-Audio-7B-Instruct (Ding et al., 2025) 作为 ALM reward model, 判断生成语音是否匹配指令 → r_llm ∈ {1, 0} [§3.2.3]
- 仅使用 instruction + text (丢弃 reference), 避免 reference 干扰 [§3.2.3]
- 混入 S2 数据防止灾难性遗忘 [§3.2.3]
- 最终 advantage 是 S2 和 S3 的混合: A^i = A^i_emo (for S2 inputs) 或 A^i_ins (for S3 inputs) [§3.2.3]

### FlexiVoice-Instruct Dataset [§4]

4316 小时高质量指令-语音数据集:
- **Emilia** (4000+ hrs): 利用 metadata (source title, tags) + transcription, 用 Deepseek-V3 生成自然语言风格描述 [§4.2]
- **Game voice acting** (300+ hrs): 利用角色名 + 台词, 用 Deepseek-V3 推断角色个性和说话风格 [§4.2]
- 关键: LLM annotator 先评估 metadata 的信息价值, 过滤噪声数据 [§4.2]

### 训练策略

- Pre-training: LLM core only, other modules frozen [§3.1]
- S1 DPO: β hyperparameter, ESD 数据, 简单情感模板 [§3.2.1]
- S2 GRPO: K completions per input, Emotion2vec-Large + CAM++ rewards, conflicting scenarios [§3.2.2]
- S3 GRPO: Kimi-Audio-7B-Instruct reward, instruction+text only (no reference), mix S2 data [§3.2.3]

## 实验

### Multi-modality Control & Disentanglement [Table 2]

| 配置 | FlexiVoice | FlexiVoice-Base | CosyVoice2 | VoxInstruct | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| TO-Easy ACC-I↑ (EN) | **97.4** | 72.4 | - | 70.6 | MEAD/CSEMOTIONS | [Table 2] |
| TO-Hard ACC-I↑ (EN) | **89.4** | 39.4 | - | 17.8 | MEAD/CSEMOTIONS | [Table 2] |
| TO-Hard ACC-T↓ (EN) | **6.6** | 30.6 | - | 41.2 | MEAD/CSEMOTIONS | [Table 2] |
| TR-Easy ACC-I↑ (EN) | **89.4** | 58.8 | 65.6 | 58.5 | MEAD/CSEMOTIONS | [Table 2] |
| TR-Hard ACC-I↑ (EN) | **78.2** | 48.8 | 61.0 | 49.7 | MEAD/CSEMOTIONS | [Table 2] |
| TR-Hard ACC-R↓ (EN) | **10.6** | 32.2 | 14.4 | 23.9 | MEAD/CSEMOTIONS | [Table 2] |

### InstructTTSEval [Table 4]

| 模型 | APS (EN) | DSD (EN) | RP (EN) | Avg (EN) | APS (ZH) | DSD (ZH) | RP (ZH) | Avg (ZH) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **FlexiVoice** | **81.2** | **85.2** | **71.4** | **79.3** | **81.2** | **85.2** | **71.4** | **79.3** |
| Gemini-pro | 87.6 | 86.0 | 67.2 | 80.3 | 89.0 | 90.1 | 75.5 | 84.8 |
| MiMo-Audio-7B | 80.6 | 77.6 | 59.5 | 72.6 | **75.7** | **74.3** | 61.5 | 70.5 |
| FlexiVoice-Base | 63.6 | 75.0 | 60.6 | 66.4 | 56.7 | 59.1 | 59.5 | 58.4 |

### PPT 消融 [Table 5]

| Training Strategy | Decoupling Avg (EN) | InstructTTSEval Avg (EN) |
| --- | --- | --- |
| FlexiVoice-Base | 54.9 | 66.4 |
| + S3 (wrong order) | 54.7 | 72.3 |
| + S3→S1→S2 | 84.4 | 74.8 |
| + S1→S2+S3 (Joint) | 74.6 | 75.5 |
| **+ S1→S2→S3 (PPT, Ours)** | **88.7** | **79.3** |

## 局限性

1. **Style 代价**: FlexiVoice 的 WER/CER 略高于 FlexiVoice-Base (5.99 vs 5.01 TO-Easy EN WER), 因为 expressive prosodic variation 对 ASR 不友好 [Table 3, §5.2] [论文原文]
2. **ACC-T 非零**: TO-Hard 中 ACC-T 6.6% (EN) 说明文本情感泄漏未完全消除 [Table 2] [agent 解读]
3. **SV score trade-off**: FlexiVoice-Base 在某些 TR 场景下 SV 略高于 FlexiVoice, 因为 base model 倾向克隆参考语音的 prosody (高 SV 但低 ACC-I) [§5.2] [论文原文]
4. **依赖外部 reward models**: S2 依赖 Emotion2vec-Large + CAM++, S3 依赖 Kimi-Audio-7B-Instruct, 这些模型的能力上限决定了 FlexiVoice 的能力天花板 [agent 解读]
5. **中文性能弱于英文**: InstructTTSEval ZH Avg 70.8 vs EN 79.3, 可能因为 Emilia 中英文数据分布不均 [Table 4] [agent 解读]

## 点评

**为什么这篇值得关注**: FlexiVoice 是第一个系统性解决 instruction-following TTS 中 Style-Timbre-Content conflict 的工作。其 Progressive Post-Training framework 提供了一个可复制的三阶段 curriculum, 且消融实验 (Table 5) 充分验证了每个阶段的必要性和顺序的重要性。

**关键洞见**:
1. **顺序至关重要**: S1 (DPO) 必须先于 S2 (GRPO), 因为 DPO 建立的基本 emotion-instruction 对齐是 GRPO reward 信号有效的前提。逆序 (S3 first) 的 Decoupling Avg 仅 54.7 vs 正序 88.7 [Table 5] [论文原文]
2. **Progressive > Joint**: S2+S3 联合训练 (74.6 / 75.5) 劣于 progressive S1→S2→S3 (88.7 / 79.3), 因为 S2 的解耦目标 (用固定 classifier 严格约束) 和 S3 的泛化目标 (用 ALM 开放评估) 梯度冲突 [§5.4] [论文原文]
3. **Style-Timbre 是 trade-off, 不是免费午餐**: FlexiVoice 的 SV score 在某些 TR 场景下低于 base model, 因为改变情感必然修改 prosodic features (pitch, energy), 而这些 features 也是 speaker embedding 的组成部分 [§5.2] [论文原文]

**与 VoxInstruct 的对比**: VoxInstruct 是统一 instruction-to-speech 框架, 但不支持 reference speech; FlexiVoice 同时支持 instruction + reference, 且通过 PPT 实现两者的解耦 [agent 解读]。

**与 CosyVoice2 的对比**: CosyVoice2 支持 reference speech 但缺乏 instruction-following; FlexiVoice 在 CosyVoice-like 架构上通过 PPT 添加了 instruction capability [Table 2] [agent 解读]。

## 可复用的 idea

1. **Progressive Post-Training (PPT) curriculum**: DPO (简单对齐) → Multi-objective GRPO (hard disentanglement) → ALM-reward GRPO (complex generalization) — 可迁移到任何需要多属性解耦的生成任务
2. **Multi-objective GRPO for disentanglement**: r_ser (style) + r_sv (timbre) 双 reward 显式解耦 — 可扩展到更多维度 (如 r_prosody, r_content)
3. **LLM metadata annotation for instruction dataset**: 利用 Emilia 的 source metadata 让 LLM 生成自然指令 — 低成本构建大规模指令数据集

---

检索命中: [[LLM-based TTS]], [[Conditional Flow Matching]], [[Speech Tokenizer]] | 过滤: [[Instruction-Guided Speech Synthesis]](pending-review), [[Natural Language Description for TTS]](pending-review), [[Style Transfer in TTS]](pending-review), [[Emotion Control in TTS]](pending-review), [[Differentiable Reward Optimization]](pending-review) | 未命中但可能相关: 无
