---
type: paper
tier: deep
title: "VibeVoice Technical Report"
arxiv_id: "2508.19205"
source: "https://arxiv.org/abs/2508.19205"
authors: [Zhiliang Peng, Jianwei Yu, Wenhui Wang, Yaoyao Chang, Yutao Sun, Li Dong, Yi Zhu, Weijiang Xu, Hangbo Bao, Zehua Wang, Shaohan Huang, Yan Xia, Furu Wei]
year: 2025
venue: "arXiv"
tags: [TTS, long-form, multi-speaker, conversational, next-token-diffusion, LLM-TTS, streaming, sigma-VAE, podcast]
concepts: ["[[Classifier-Free Guidance]]", "[[Speech Tokenizer]]", "[[LLM-based TTS]]", "[[Semantic vs Acoustic Tokens]]", "[[Diffusion Model]]"]
models: ["[[MELLE]]"]
tasks: [long-form-TTS, multi-speaker-TTS, conversational-speech-generation, zero-shot-TTS]
datasets: [SEED-TTS-Eval, LibriTTS, CommonVoice]
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个实体页: [[LLM-based TTS]], [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[Classifier-Free Guidance]], [[Diffusion Model]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **长对话语音生成的挑战**: 当前 LLM-based TTS 在短句合成上表现优异 [[LLM-based TTS]],但长对话 (podcast、audiobook) 场景面临三重挑战: (1) 序列长度导致 attention 计算爆炸; (2) 多说话人 turn-taking 建模; (3) 生成稳定性。FireRedTTS 2 通过 12.5Hz streaming tokenizer 缩短序列,CosyVoice 2 用 chunk-aware causal flow matching 实现流式,但多数方案仍受限于生成长度和稳定性。
>
> **Continuous representation 路线**: [[Semantic vs Acoustic Tokens]] 的核心 trade-off 在于保真度 vs 序列长度。LatentLM (Sun et al., 2024) 提出 next-token diffusion + sigma-VAE 框架,以极高压缩比 (1600-6400x) 将连续数据编码为短序列,在 TTS 上以 10x 更少解码步数超越 VALL-E 2。
>
> **本文定位**: VibeVoice 将 LatentLM 框架应用于**长对话多说话人**场景,引入 3200x 因果语音 tokenizer (7.5 Hz)、双 tokenizer (acoustic + semantic) 混合表示,结合 Qwen2.5 LLM backbone,实现最长 90 分钟、最多 4 人的对话语音合成。
>
> 检索命中: [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[Semantic vs Acoustic Tokens]]✓ | 过滤: [[Classifier-Free Guidance]](pending-review), [[Diffusion Model]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 LatentLM 的 next-token diffusion 框架,结合 3200x causal acoustic tokenizer + semantic tokenizer + Qwen2.5 LLM,实现最长 90 分钟多说话人对话语音合成,在 preference/realism/richness 三项主观评测中全面超越 Gemini 2.5 Pro TTS 和 ElevenLabs v3
> - **路线**: Voice prompts + Text scripts (multi-speaker) → [Acoustic tokenizer (continuous VAE) + Semantic tokenizer] → Hybrid features → Qwen2.5 LLM (1.5B/7B) → Token-level Diffusion Head (CFG, DPM-Solver++) → Acoustic VAE Decoder → Waveform (up to 90min)
> - **指标**: Podcast MOS: Preference 3.75/3.71, Realism 3.59/3.59, Richness 3.44/3.75 (1.5B/7B) vs Gemini 2.5 Pro 3.55/3.78/3.65 [Table 1]; Short-utterance WER 1.16 (zh), 3.04 (en) at 7.5fps [Table 2]; Tokenizer PESQ 3.068, UTMOS 4.181 (test-clean) [Table 3]
> - **可借鉴**: 1) 双 tokenizer (acoustic + semantic) 的混合表示实现长对话稳定性; 2) 3200x 因果 tokenizer 仅 7.5 Hz,speech-to-text token ratio 约 2:1; 3) curriculum learning from 4K to 64K context; 4) ASR proxy task 训练 semantic tokenizer
> - **局限**: 仅中英双语; 不处理重叠语音; 不支持非语音音频 (背景音乐/音效)

## 核心问题

长对话多说话人语音合成是 TTS 领域的 "最后一公里" 问题 [§1]:

1. **生成长度**: 现有系统多限于单句或短段落,podcast/audiobook 需要 10-90 分钟连续生成 [§1] [论文原文]
2. **多说话人**: 自然 turn-taking、不同说话人音色一致性、角色分配 [§1] [论文原文]
3. **真实感**: 不仅需要高音质,还需要自然的犹豫、语气变化、互动节奏 — 即 "vibe" [§1] [论文原文]
4. **序列效率**: 传统 codec tokens (50-75 Hz) 在 90 分钟对话场景下产生 ~400K tokens,远超任何 LLM 的 context window [agent 解读]

VibeVoice 的解决方案: 7.5 Hz 因果 tokenizer (3200x 压缩) 使 90 分钟对话仅需 ~40K tokens,配合 64K context window 的 LLM 即可处理 [§1] [论文原文]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

VibeVoice 由四个组件组成 [§2, Fig 2]:

1. **Acoustic Tokenizer**: sigma-VAE encoder/decoder,将语音波形编码为连续 latent features [§2.1]
2. **Semantic Tokenizer**: 确定性 encoder (无 VAE),提取内容特征,ASR proxy task 训练 [§2.1]
3. **LLM (Qwen2.5)**: 处理混合输入 (voice features + text scripts),预测 hidden states [§2.2]
4. **Diffusion Head**: 4-layer lightweight network,以 LLM hidden state 为条件生成 acoustic VAE features [§2.2]

### 关键设计选择

#### 1. 双 Tokenizer 设计: Acoustic + Semantic [§2.1]

**Acoustic Tokenizer** [§2.1]:
- 基于 sigma-VAE (from LatentLM),使用 hierarchical Transformer encoder-decoder (7 stages) [§2.1]
- 1D depth-wise causal convolutions (非 self-attention),支持流式 [§2.1]
- 6 个下采样层,累计 **3200x** 压缩,从 24kHz 输入到 7.5 tokens/s [§2.1] [论文原文]
- 每个 encoder/decoder ~340M params [§2.1]
- 训练目标: 跟随 DAC (KSL+23),包括 discriminator 和 loss [§2.1]

**Semantic Tokenizer** [§2.1]:
- 架构与 acoustic tokenizer encoder 相同,但**无 VAE 组件** [§2.1] [论文原文]
- 目标: 确定性内容提取 (content-centric) [§2.1]
- 训练: ASR proxy task — 输出被数层 Transformer decoder 解码以预测 text transcripts [§2.1] [论文原文]
- 预训练后丢弃 ASR decoder,仅保留 encoder 用于特征提取 [§2.1]

**为什么双 tokenizer?** [agent 解读] Acoustic tokenizer 保留声学细节但语义信息分散,semantic tokenizer 显式对齐文本内容。混合两者的特征为 LLM 提供更丰富的上下文,有助于长对话的内容一致性和生成稳定性。这延续了 [[Semantic vs Acoustic Tokens]] 的混合路线思想。

#### 2. Input Representation [§2.2]

模型输入 X 拼接两部分 [§2.2]:
- **Voice features**: X = [Speaker_1 : z_1, ..., Speaker_N : z_N] — 每个说话人的 acoustic latent 作为 voice prompt
- **Text scripts**: + [Speaker_1 : T_1, Speaker_2 : T_2, ..., Speaker_N : T_N] — 带角色标识的文本

生成的语音段被 acoustic + semantic tokenizer 编码为 hybrid representation,用于 auto-regressive 建模 [§2.2] [论文原文]。

#### 3. Token-Level Diffusion Head [§2.2]

直接复用 LatentLM 的 next-token diffusion 框架 [§2.2]:
- 条件: LLM hidden state h_i [§2.2]
- 训练: 预测 noise,优化反向去噪过程 [§2.2]
- 推理: Gaussian noise → iterative refinement → target acoustic VAE feature z_a,i [§2.2]
- **Classifier-Free Guidance**: CFG scale 1.3,interpolate conditional (guided by h_i) and unconditional predictions [§2.2] [论文原文]
- **DPM-Solver++**: 加速采样,iterative denoising step = 10 [§2.2] [论文原文]

#### 4. 训练配置 [§2.2]

- LLM backbone: Qwen2.5 1.5B 和 7B 两个版本 [§2.2]
- Diffusion head: 4 layers (from LatentLM) [§2.2]
- **Frozen tokenizers**: 预训练 acoustic/semantic tokenizer 在 VibeVoice 训练中冻结,仅训练 LLM + diffusion head [§2.2] [论文原文]
- **Curriculum learning**: 输入序列长度从 4,096 tokens 逐步增加到 65,536 tokens [§2.2] [论文原文]
- 为什么 curriculum? [agent 解读] 直接在超长序列上训练不稳定且低效。逐步增长让模型先学会短对话生成,再扩展到长对话。

### 训练策略

- 数据: 未具体公开训练数据细节,仅提及使用语音和文本对齐的对话数据
- Speech-to-text token ratio: 约 2:1,即约两个 speech tokens 对应一个 BPE text token [§1] [论文原文]
- Scaling: 从 1.5B 到 7B,更大模型在 richer timbre, more natural intonation, enhanced transfer 上有显著提升 [§1] [论文原文]

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Preference MOS↑ (Podcast) | 3.75 (7B) | 3.55 (Gemini 2.5 Pro), 3.43 (SesameAI-CSM) | 8 conversations, 1hr | [Table 1] |
| Realism MOS↑ (Podcast) | 3.59 (7B) | 3.78 (Gemini 2.5 Pro), 3.03 (SesameAI-CSM) | 8 conversations | [Table 1] |
| Richness MOS↑ (Podcast) | 3.75 (7B) | 3.65 (Gemini 2.5 Pro), 3.03 (SesameAI-CSM) | 8 conversations | [Table 1] |
| Average MOS↑ (Podcast) | **3.76** (7B) | 3.66 (Gemini 2.5 Pro), 3.40 (ElevenLabs) | 8 conversations | [Table 1] |
| WER↓ Whisper (Podcast) | **1.11** (1.5B), 1.29 (7B) | 1.73 (Gemini 2.5 Pro), 2.81 (Mooncast) | 8 conversations | [Table 1] |
| SIM↑ (Podcast) | 0.548 (1.5B), **0.692** (7B) | 0.623 (ElevenLabs) | 8 conversations | [Table 1] |
| CER↓ (Short, test-zh) | **1.16** | 1.45 (CosyVoice 2), 1.12 (Seed-TTS) | SEED test sets | [Table 2] |
| WER↓ (Short, test-en) | 3.04 | **2.57** (CosyVoice 2), 2.62 (MaskGCT) | SEED test sets | [Table 2] |
| SIM↑ (Short, test-zh) | 0.744 | 0.748 (CosyVoice 2), **0.796** (Seed-TTS) | SEED test sets | [Table 2] |
| Tokenizer PESQ↑ (test-clean) | **3.068** | 2.738 (DAC 4Q), 2.72 (Encodec 8Q) | LibriTTS | [Table 3] |
| Tokenizer UTMOS↑ (test-clean) | **4.181** | 3.433 (DAC 4Q), 3.04 (Encodec 8Q) | LibriTTS | [Table 3] |

**关键发现**:

1. **长对话全面领先**: VibeVoice-7B 在 podcast 场景的 Average MOS 3.76,超越 Gemini 2.5 Pro (3.66) 和 ElevenLabs v3 (3.40) [Table 1] [论文原文]。可合成 5000+ 秒音频 (Fig 1),支持最多 4 个说话人 [§1]。

2. **7B vs 1.5B scaling**: 7B 在所有主观指标上显著优于 1.5B,特别是 Richness (3.75 vs 3.44) 和 SIM (0.692 vs 0.548) [Table 1] [论文原文]。更大模型在 timbre richness 和 cross-lingual transfer 上有明显优势 [§1]。

3. **短句也有竞争力**: 尽管主要训练目标是长对话,VibeVoice-1.5B 在 SEED test sets 上 CER 1.16 (zh), WER 3.04 (en),与 CosyVoice 2, Seed-TTS 等专注短句的系统可比 [Table 2] [论文原文]。

4. **超高压缩 tokenizer**: 7.5 Hz tokenizer (单 quantizer) 在 PESQ 3.068 和 UTMOS 4.181 上**领先所有 baseline tokenizers** (包括 multi-quantizer 方案如 EnCodec 8Q, DAC 4Q),证明 continuous VAE 在极高压缩比下的优势 [Table 3] [论文原文]。

5. **极低 WER**: Podcast 场景 WER (Whisper) 仅 1.11 (1.5B),优于所有对比系统 [Table 1]。

## 局限性

1. **仅中英双语**: 其他语言的输入可能导致异常输出 [§4] [论文原文]
2. **不处理非语音音频**: 不生成背景音乐、音效等 [§4] [论文原文]
3. **不建模重叠语音**: 对话中说话人之间无重叠 [§4] [论文原文]
4. **Deepfake 风险**: 高质量合成可能被滥用于冒充和虚假信息 [§4] [论文原文]
5. **训练数据未公开**: 无法评估数据规模和多样性的影响 [agent 解读]
6. **短句 SIM 略低**: test-zh SIM 0.744 低于 Seed-TTS 0.796 [Table 2] [论文原文]

## 点评

VibeVoice 是 LatentLM 框架在工业级 TTS 场景的成功落地,证明了 next-token diffusion + sigma-VAE 范式的实用性:

1. **规模化验证**: 90 分钟 / 4 说话人 / 64K context,这是目前已公开的最长对话语音生成能力
2. **架构简洁**: 相比 CosyVoice 系列的 text encoder + LLM + flow matching 三模块,VibeVoice 去掉了独立的 text encoder 和 length regularization,voice+text 直接拼接输入 LLM [§1] [论文原文]
3. **Tokenizer 是关键**: 3200x 因果 tokenizer 的 7.5 Hz frame rate 使 90 分钟仅需 ~40K tokens,这是长对话可行性的根基

**与 CLEAR 的对比** [agent 解读]:
- 共同点: 都使用 continuous VAE latent + per-token diffusion/flow head + AR LM
- VibeVoice 使用 DDPM (LatentLM 框架), CLEAR 使用 rectified flow
- VibeVoice 有 semantic tokenizer 辅助, CLEAR 仅用 acoustic VAE
- VibeVoice 基于 Qwen2.5 预训练 LLM, CLEAR 从头训练 Transformer
- VibeVoice 重点是长对话/多说话人, CLEAR 重点是低延迟/流式

**关注点**: 训练数据未公开是评估 VibeVoice 的主要障碍。podcast 评估集仅 8 段对话 (~1 小时),样本量偏小。

## 可复用的 idea

1. **双 tokenizer 混合**: acoustic (VAE) + semantic (ASR-trained) 的混合特征为 LLM 提供语义+声学双重信息,提升长对话稳定性
2. **Curriculum learning for context length**: 从 4K → 64K 逐步增长 context window,适用于任何长序列生成任务
3. **Speaker-interleaved input format**: [Speaker_k : z_k] 的格式简洁地编码多说话人信息
4. **7.5 Hz tokenizer**: speech-to-text token ratio 约 2:1,使语音和文本在序列中长度匹配,避免严重不平衡

---

检索命中: [[LLM-based TTS]], [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]] | 过滤: [[Classifier-Free Guidance]](pending-review), [[Diffusion Model]](pending-review) | 未命中但可能相关: 无
