---
tier: deep
title: "Tortoise TTS"
aliases: [TorToise, TorToise-v2, Better Speech Synthesis Through Scaling]
authors: ["James Betker"]
year: 2023
arxiv_id: "2305.07243"
source: "Sources/TortoiseTTS.pdf"
venue: "arXiv preprint"
tags: [TTS, zero-shot, autoregressive, diffusion, DDPM, voice-cloning, scaling, VQVAE, re-ranking]
level: deep
status: draft
concepts: ["[[DiffusionModel]]", "[[Classifier-FreeGuidance]]", "[[LLM-basedTTS]]", "[[MelSpectrogram]]", "[[NeuralVocoder]]", "[[SpeechTokenizer]]", "[[VoiceCloningTaxonomy]]"]
models: []
tasks: [TTS, zero-shot-TTS, voice-cloning]
created: 2026-06-03
updated: 2026-06-03
kb_sources: ["[[NeuralVocoder]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]"]
---
tier: deep

## KB 背景

本文涉及以下已有知识:

- **[[NeuralVocoder]]** (confirmed): TTS pipeline 最后一级,Tortoise 使用 UnivNet vocoder 将 mel spectrogram 合成波形。本文是 GAN vocoder 实际部署的早期范例。
- **[[LLM-basedTTS]]** (confirmed): Tortoise 是 LLM-based TTS 的早期先驱之一,将 GPT-2 风格的自回归 Transformer 用于语音生成,早于 VALL-E。其 "AR decoder + DDPM" 两阶段范式影响了后续整个 LLM-TTS 路线。
- **[[SpeechTokenizer]]** (confirmed): Tortoise 使用 VQVAE 将 mel spectrogram 离散化为 8192 码本的 token 序列,供 AR decoder 建模。这是 codec-based TTS 之前的 "mel-token" 路线。
- **[[Classifier-FreeGuidance]]** [待确认]: Tortoise 在 DDPM 推理中使用 CFG (guidance constant = 2),同时对 speaker conditioning 和 AR activation 做 guidance [§4]。
- **[[VoiceCloningTaxonomy]]** [待确认]: Tortoise 属于 zero-shot voice cloning 类别,通过 speech conditioning input 实现声音克隆,无需微调 [§2.2.1]。

> [!summary] 速查
> - **一句话**: 将图像生成领域的 AR Transformer + DDPM 范式迁移到 TTS,通过大规模数据训练和 CLVP re-ranking 实现高质量多说话人语音合成
> - **路线**: Text + Reference Audio → VQVAE tokens → AR Transformer(GPT-2) → CLVP re-ranking → DDPM(mel spectrogram) → UnivNet Vocoder → Waveform
> - **指标**: 非标准化评估(使用自建 CLVP-based 评估套件 + wav2vec intelligibility); 定性优于同期开源 TTS
> - **可借鉴**: (1) AR+DDPM 两阶段解耦设计 (2) CLVP re-ranking 策略 (3) "TorToise Trick" — 在 AR latent space 微调 DDPM (4) 大规模 web-scraped 数据集构建流程
> - **局限**: 推理慢(64步 DDIM + AR); 无标准化 benchmark; 22kHz 采样率; 缺乏与 VALL-E 等的直接对比; 独立研究者资源有限

## 核心问题

### WHY: 为什么要做这个工作?

传统 TTS 研究受三个约束限制 [§1.1]: (1) 追求高效模型导致只能在小数据集上训练; (2) 大规模转录语音数据集不可用; (3) encoder-decoder 架构难以 scaling。作者认为图像生成领域的 AR Transformer + DDPM 方法论可以直接迁移到语音合成,核心假设是 **规模化训练通用架构比设计领域专用架构更有效** [§1.2, §7]。

### WHAT: 核心贡献

1. **AR + DDPM 两阶段架构** [§2]: 将 TTS 分解为 (a) AR Transformer 生成离散 speech tokens (解决 text-speech 对齐), (b) DDPM 将 tokens 解码为连续 mel spectrogram (建模高质量声学细节) [§2.1]
2. **CLVP re-ranking** [§2.3]: 类比 DALL-E 的 CLIP re-ranking,训练 Contrastive Language-Voice Pretrained (CLVP) 模型,从 AR 的多个候选中选出最佳输出 [§2.3]
3. **"TorToise Trick"** [§2.2.2]: 先在 VQVAE 离散码上训练 DDPM,收敛后切换到 AR 模型的 activation latents 上微调 DDPM。AR latent space 比离散 token 更语义丰富,显著提升 DDPM 效率和质量 [§2.2.2]
4. **大规模数据集构建** [§5, Appendix A]: 从 LibriTTS + HiFiTTS (896h) 出发,另外从互联网爬取 49,000 小时有声书和播客数据,通过分类器管道过滤噪声/音乐/多人语音,用微调的 wav2vec2-large 转录 [Appendix A]

## 方法详解

### 1. VQVAE Tokenizer [§B.1, Table 1]

| 参数 | 值 |
|------|------|
| 架构 | 1D Conv ResNet encoder + decoder |
| 码本大小 | 8192 tokens |
| 编码维度 | 256 (codebook dim) |
| 压缩率 | 4x (在 mel spectrogram 上) |
| 训练损失 | MSE reconstruction + commitment loss |
| 批大小 | 8192 |
| 训练量 | 360M samples |

VQVAE 在 mel spectrogram 上工作,将其压缩 4x 后量化为 8192 个离散码 [§B.1]。与后续 SoundStream/EnCodec 的 RVQ 方案不同,Tortoise 使用单层 VQ 且不操作原始波形。

### 2. Autoregressive Prior [§B.2, Table 2]

AR decoder 采用标准 GPT-2 架构 [§B.2]:

| 参数 | 值 |
|------|------|
| 架构 | Transformer stack with causal masking |
| 层数 | 30 |
| 模型维度 | 1024 |
| 注意力头数 | 16 |
| 文本 tokenization | Custom BPE, 256 tokens wide |
| 批大小 | 1024 |
| 训练量 | 119M samples |

**输入格式** [§B.2]:
```
<SC><BT><T><T><T>...<T><ET><BM><M><M><M>...<EM>
```
其中 SC = speech conditioning encoding, T = text tokens, M = MEL tokens。

**关键设计**: Speech conditioning encoding 由独立 encoder 从参考音频的 mel spectrogram 提取,产生单一向量放在注意力上下文最前面 [§2.2.1, §B.2]。每个训练样本产生 2 个 conditioning encodings 取平均。最大序列长度 = 402 text tokens + 604 MEL tokens。

**WHY speech conditioning**: 为 AR 和 DDPM 提供目标说话人的音色、语调等先验,缩小条件输出空间 [§2.2.1]。

### 3. CLVP Re-ranking [§2.3, §B.3, Table 3]

**核心思想**: TTS 数据集天然包含 text-speech pairs,可训练对比模型判断 text-speech 匹配度 [§2.3]。

| 参数 | 值 |
|------|------|
| 架构 | Dual transformer stacks (text + MEL) |
| 深度 | 20 |
| 模型维度 | 768 |
| 注意力头数 | 12 |
| 损失函数 | Contrastive loss |
| 训练量 | 80M samples |

**推理流程** [§4]: AR 生成大量候选 → CLVP 为每个候选评分 → 取 top-k → 送入 DDPM。这使得不需要运行昂贵的 DDPM 就能筛选高质量候选 [§2.3]。

### 4. Diffusion Decoder [§B.4, Table 4]

| 参数 | 值 |
|------|------|
| 架构 | Alternating full attention + conv resblocks |
| 深度 | 10 |
| 模型维度 | 1024 |
| 注意力头数 | 16 |
| 损失函数 | MSE (weight 1) + VLB (weight n) |
| 训练量 | 65M samples |

**"TorToise Trick" 详解** [§2.2.2, §B.4]:
- 初始训练: DDPM 学习将 VQVAE 离散码转为 mel spectrogram
- 微调阶段: 切换为从 AR 模型的 activation latents (连续表示) 生成 mel
- **WHY**: AR latent space 比离散 token 语义更丰富,DDPM 在此空间更高效。这是 "对模型输出质量贡献最大的单一设计决策" [§B.4] [论文原文]
- **HOW**: 将 MEL token 替换为 AR 对应位置的 activation vector,rescale 到 [-1, 1] 后作为 DDPM 输入 [§B.4]

**Classifier-Free Guidance** [§B.4]: 对 speech conditioning signal 和 AR activations 两个条件信号做 CFG。训练时 15% 概率 dropout 这两个条件。推理时 guidance constant = 2 [§B.4]。

### 5. 推理流程 [§4]

1. 将 text + conditioning audio 送入 AR decoder,解码大量候选 (nucleus sampling P=0.8, repetition penalty=2, temperature=0.8) [§4]
2. CLVP 对每个候选评分,选 top-k [§4]
3. 每个 top-k 候选送入 DDPM 解码为 mel spectrogram [§4]
4. UnivNet vocoder 将 mel 合成波形 [§4]
5. DDIM 采样配置: 64 steps, linear schedule, CFG constant=2 [§4]

## 数据集 [§5, Appendix A]

**基础数据**: LibriTTS + HiFiTTS = 896 小时转录语音 [§5]

**扩展数据集**: 从互联网爬取有声书和播客共 49,000 小时 [§5, Appendix A]:
- 按 500ms 静音切分,保留 5-20 秒片段 [Appendix A]
- 分类器管道过滤: 背景噪音、音乐、低质量(如电话录音)、多人语音、混响 [Appendix A]
- 使用微调的 wav2vec2-large 做转录,特别微调了标点符号预测 [Appendix A]
- 最终在 LibriTTS + HiFiTTS 上微调 AR decoder [§B.2]

## 实验与评估 [§6]

作者指出同期 SOTA TTS 系统多为闭源,难以公平对比 [§6]。自建评估方案:
- **CLVP-based 评估**: 类似 FID 的 text-speech 距离度量 [§6] [agent解读: 类似 Frechet Inception Distance 思路,但用 CLVP 特征空间]
- **wav2vec intelligibility**: 使用开源 wav2vec 模型评估可懂度 [§6]
- **定性对比**: 与同期 TTS 系统的样本对比 [§6]

## 关键设计选择与证据

| 设计选择 | WHY | 证据 |
|---------|-----|------|
| AR + DDPM 两阶段 | AR 擅长跨域对齐(text→speech),DDPM 擅长高质量连续生成 | 图像领域 DALL-E + Improved DDPM 的成功 [§1.2, §2.1] |
| VQVAE tokenization | 将连续 mel 离散化供 AR 建模 | 8192 codebook 足够编码语音信息 [§B.1] |
| CLVP re-ranking | 避免 AR 的随机采样产生低质量样本 | DALL-E 中 CLIP re-ranking 效果显著 [§1.2.3, §2.3] |
| TorToise Trick | AR latent 比 discrete token 更丰富 | "单一最大质量提升" [§B.4] [论文原文] |
| CFG on two signals | 同时增强 speaker 和 content 条件 | 图像 diffusion 中 CFG 效果显著 [§B.4] |
| 大规模数据 | 类比 GPT/DALL-E 的 scaling 经验 | 训练曲线表明远未过拟合 [§7, Appendix C] |

## 与已有方法的差异

| 维度 | Tortoise (2023) | VALL-E (2023) | 传统 TTS (Tacotron 2) |
|------|-----------------|---------------|----------------------|
| Token 类型 | VQVAE mel tokens (8192) | EnCodec RVQ tokens (多层) | 无(直接预测 mel) |
| AR 架构 | GPT-2 (30 层) | GPT-2 (12 层) | Attention-based seq2seq |
| 声学解码 | DDPM (连续 mel) | NAR Transformer (残差层) | 无(直接输出 mel) |
| Re-ranking | CLVP (对比模型) | 无 | 无 |
| 训练数据 | ~50K 小时 | ~60K 小时 | ~24 小时 |
| 开源 | 完全开源 | 未开源 | 开源(实现) |
| 采样率 | 22/24 kHz | 16 kHz | 22 kHz |

[agent解读] Tortoise 和 VALL-E 几乎同期提出(2023),但设计哲学不同: VALL-E 使用 codec tokens + AR+NAR 两阶段直接生成多层离散码; Tortoise 使用 mel tokens + AR+DDPM 组合,引入 re-ranking 和 TorToise Trick。两者共同验证了 "将 TTS 视为语言建模任务 + scaling" 的可行性,奠定了 LLM-based TTS 的基础。

## 局限性分析

1. **推理速度**: 64 步 DDIM + AR 采样 + 多候选 CLVP re-ranking,推理极慢 [§4]
2. **评估标准化**: 无标准 benchmark 对比,自建评估可能有偏 [§6]
3. **采样率限制**: 22kHz (stack 本身) / 24kHz (DDPM output + UnivNet),不支持 48kHz 全频段 [§B.4]
4. **独立研究者资源限制**: 8x RTX-3090 训练约 1 年,多次实验迭代受限 [§3]
5. **架构建议但未实现** [Appendix C]: 作者列出 7 项改进(如 relative PE, 更好 FFN, 更高采样率等),表明方案远未最优

## 历史意义 [agent解读]

Tortoise TTS 的历史意义在于:
1. **验证了 "通用架构 + scaling" 假说**: 几乎不使用音频专用设计,仅靠 Transformer + DDPM + 大数据即超越同期 TTS [§7]
2. **AR + Diffusion 范式的早期范例**: 后续 VALL-E 2 / NaturalSpeech 2 / Seed-TTS 等都采用类似的 "粗粒度 AR + 细粒度 diffusion/flow" 设计
3. **开源贡献**: 完全开源模型权重和代码 (github.com/neonbjb/tortoise-tts),推动了社区发展
4. **CLVP 思路**: re-ranking 策略后来在 VALL-E 2 (重复感知采样)、ELLA-V 等中以不同形式出现

---

检索命中: [[NeuralVocoder]], [[LLM-basedTTS]], [[SpeechTokenizer]] | 过滤: [[Classifier-FreeGuidance]](pending-review), [[VoiceCloningTaxonomy]](pending-review), [[MelSpectrogram]](pending-review), [[DiffusionModel]](pending-review) | 未命中但可能相关: 无
