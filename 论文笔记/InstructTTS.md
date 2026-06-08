---
type: paper
tier: deep
title: "InstructTTS: Modelling Expressive TTS in Discrete Latent Space with Natural Language Style Prompt"
arxiv_id: "2301.13662"
source: "Sources/InstructTTS.pdf"
authors: [Dongchao Yang, Songxiang Liu, Rongjie Huang, Chao Weng, Helen Meng]
year: 2023
venue: "arXiv (v2: Jun 2023)"
tags: [expressive-TTS, natural-language-prompt, discrete-diffusion, VQ-VAE, cross-modal, metric-learning, mutual-information, style-control, Mandarin-TTS, classifier-free-guidance]
concepts: ["[[NaturalLanguageDescriptionforTTS]]", "[[Diffusion-basedTTS]]", "[[Classifier-FreeGuidance]]", "[[StyleTransferinTTS]]", "[[GlobalStyleTokens]]", "[[ResidualVectorQuantization]]"]
models: []
tasks: ["[[InstructedSpeechGeneration]]"]
datasets: []
kb_context_sources: 1
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页: [[ResidualVectorQuantization]]; 参考 5 个未确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ResidualVectorQuantization]]✓ | 过滤: [[NaturalLanguageDescriptionforTTS]](pending-review), [[Diffusion-basedTTS]](pending-review), [[Classifier-FreeGuidance]](pending-review), [[StyleTransferinTTS]](pending-review), [[GlobalStyleTokens]](pending-review) | 未命中但可能相关: 无
>
> **谱系定位**: InstructTTS 处于 NL Description for TTS 演进线的早期阶段。KB 记录的演进为: Style tagging (GST, 2018) → Reference encoder (2018-2022) → PromptTTS (2023) → InstructTTS (2023) → Parler-TTS (2024) → ...。InstructTTS 与 PromptTTS 同期,但技术路线不同: PromptTTS 使用结构化 5 属性描述 (gender/pitch/speed/volume/emotion),InstructTTS 允许自由格式自然语言描述,更贴近真实使用场景。两者都属于 content/description 分离范式 (与后续 VoxInstruct 的统一指令范式不同)。
>
> **已有认知**: Classifier-Free Guidance (CFG) 在 TTS 领域已被广泛应用于连续 diffusion 和 flow matching 系统 (如 Guided-TTS 2, CosyVoice)。InstructTTS 将 CFG 引入离散 diffusion 空间,KB 中 CFG 页面已记录了"离散空间 CFG"的后续发展 (如 OmniVoice 在 log-softmax 空间操作)。RVQ 方面, KB 记录了 RVQ/GVQ/GRVQ 三种量化变体,InstructTTS 正是较早探索这三种变体在 TTS 生成中影响的工作之一。
>
> **创新判断**: InstructTTS 的核心创新在于两个层面: (1) 将离散 diffusion model 首次应用于 TTS 声学建模 (此前离散 diffusion 主要用于图像和音频事件生成); (2) 提出三阶段训练策略从自然语言中提取 style embedding,并用 cross-modal metric learning 对齐文本-语音风格空间。与同期 PromptTTS 相比,InstructTTS 不约束 style prompt 的形式,允许更自由的自然语言表达。

> [!summary] 速查
> - **一句话**: 首次将离散 diffusion model 用于 TTS 声学建模,结合三阶段训练的 cross-modal style prompt embedding 和互信息最小化,实现自由格式自然语言风格控制 [§I]
> - **路线**: Content Prompt → Phoneme Encoder → Variance Adaptor; Style Prompt → RoBERTa (3-stage trained) → Style Encoder; Speaker ID → Speaker Embedding; 三路融合 → SALN Adaptor → Discrete Diffusion Decoder (Mel-VQ-Diffusion 或 Wave-VQ-Diffusion) → VQ Tokens → Vocoder/Codec Decoder → Waveform [Fig 1]
> - **指标**: Mel-VQ-Diffusion MOS 4.35 / RMOS 4.22; Wave-VQ-Diffusion (GRVQ) MOS 3.95 / RMOS 4.32; Baseline MOS 4.04 / RMOS 3.85 [Table II]; AXY preference 0.72 (Mel) / 0.84 (Wave) vs baseline [Table III]
> - **可借鉴**: (1) 离散 diffusion + VQ 中间表征: 将 mel 预测转化为离散 token 分类问题,避免连续空间的时频相关性建模困难; (2) Cross-modal metric learning 三阶段训练获取鲁棒 style embedding; (3) MI 最小化解耦 style-speaker 和 style-content,避免 audio encoder 泄漏非 style 信息; (4) 改进的 mask-and-uniform 策略,按 RVQ 层信息量递减原则差异化 masking
> - **局限**: (1) 仅在 44h 内部中文数据集 NLSpeech 上验证,无公开数据集实验; (2) 推理速度受限 (100 diffusion steps); (3) Wave-VQ-Diffusion 语音质量 MOS 3.95 低于 Mel 方案 4.35; (4) 未与同期 PromptTTS 直接对比; (5) 代码和数据均未公开

## 核心问题

现有表达性 TTS 控制风格的方式存在两个根本局限 [§I] [论文原文]:

1. **离散标签方式受限于预定义类别**: 使用 categorical style labels (如 happy/sad/angry) 只能生成训练集中已有的有限风格,无法覆盖真实场景的多样需求 [§I] [论文原文]。

2. **参考音频方式不可解释且选取困难**: reference speech 方式虽然可泛化到域外风格,但从参考音频提取的 style representation 不易理解,且用户难以精确找到匹配需求的参考音频 [§I] [论文原文]。

InstructTTS 提出用自然语言作为风格描述 (如 "Sigh tone in full of sad mood with some helpless feeling"),这带来两个研究挑战 [§I] [论文原文]:
- 如何训练一个能从自然语言 prompt 中捕获语义并控制合成风格的语言模型
- 如何设计声学模型来有效建模表达性 TTS 的 one-to-many 映射问题

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

InstructTTS 由五个模块组成 [§IV, Fig 1]:

1. **Content Encoder**: 遵循 FastSpeech2 架构,4 层 FFT block (hidden 256, heads 2, kernel 9, filter 1024) + Variance Adaptor (预测 duration/pitch/energy) [§IV-A]
2. **Style Encoder**: 三部分 -- 预训练 prompt encoder (RoBERTa) + adaptor layer + audio encoder [§IV-C, Fig 1(b)]
3. **Speaker Embedding**: 将 speaker ID 映射为 embedding [§IV]
4. **SALN Adaptor**: Style-Adaptive Layer Normalization,将 style 信息注入 content representation [§IV, Fig 1(a)]
5. **Discrete Diffusion Decoder**: 生成 VQ acoustic tokens (Mel-VQ-Diffusion 或 Wave-VQ-Diffusion) [§IV-D/E]

**关键信号流** [Fig 1, Algorithm 1]:
- 训练时: audio encoder 从 GT mel 提取 style embedding z_e,同时 prompt encoder 提取 style embedding z_p,两者对齐 (L2 loss);条件 y = content features + z_e + speaker embedding
- 推理时: 直接用 prompt encoder 输出 z_p 替代 z_e,y = content features + z_p + speaker embedding [Algorithm 2]

### 关键设计选择

**设计 1: 三阶段 Style Prompt Embedding 训练**

**为什么**: 需要一个 prompt embedding 模型满足两个条件: (1) 能捕获风格 prompt 的语义信息; (2) embedding 空间平滑均匀,可泛化到训练未见的风格描述 [§IV-B] [论文原文]。

**怎么做** [§IV-B]:
- Stage 1: 在中文数据上从头训练 RoBERTa 基础语言模型 (因为开源预训练模型主要是英文的) [论文原文]
- Stage 2: 用少量中文 NLI 数据以 SimCSE 方式 (InfoNCE loss) 微调,获得更好的语义表示 [论文原文]
- Stage 3: Cross-modal representation learning -- 构建 audio-text retrieval 任务,用 style prompt 和对应 audio 的正负样本对训练,使 text embedding 和 audio embedding 对齐到共享语义空间 [§IV-B, Fig 2] [论文原文]

**证据**: Table V 显示 cross-modal learning 后 STS-B 的 Spearman 相关系数从 80.4% 提升到 80.94%,表明 cross-modal 学习不仅对齐了模态,还改善了文本语义表示本身 [Table V] [论文原文]。Table VI 显示 InfoNCE loss 的 R@1 检索性能 (15.62%) 优于 contrastive ranking loss (11.62%) [Table VI] [论文原文]。

**设计 2: 离散 Diffusion 建模 (Mel-VQ-Diffusion)**

**为什么选离散而不是连续**: mel spectrogram 在时间和频率轴上高度相关,直接预测连续 mel 困难且存在 GT-predicted 的 gap (vocoder 在 GT mel 上训练,predicted mel 上推理) [§IV-D] [论文原文]。通过 VQ-VAE 将 mel 映射到离散 latent space,减少时频相关性,且将预测问题转化为离散 token 分类 [§IV-D] [论文原文]。

**怎么做** [§IV-D, Fig 3]:
1. 预训练 Mel-VQ-VAE: encoder 将 mel 下采样 (时间 2x, 频率 20x) 后量化到 codebook (K=512, dim=256),decoder 重建 mel;加 adversarial loss 提升重建质量 [§V-B1]
2. Mel-VQ-Diffusion decoder: 基于 mask-and-uniform transition matrix 的离散 diffusion,12 层 8 头 Transformer (dim 256),T=100 steps [§IV-D2, §V-B3]
3. 使用 HiFi-GAN vocoder 从重建 mel 生成波形 [§IV-D]

**Transition matrix 设计** [Eq 2-4]: 每个 token 有概率 gamma_t 被 mask,概率 K*beta_t 被 uniform resampling,概率 alpha_t 保持不变。这实现了从 clean tokens 到完全 masked/random 的渐进腐蚀 [论文原文]。

**设计 3: Wave-VQ-Diffusion + U-Transformer**

**为什么还需要 waveform 方案**: Mel-VQ-Diffusion 虽然语音质量好,但频率维度 20x 下采样可能损失 pitch 信息,导致韵律保真度不如直接建模波形 [§VII-A1] [论文原文]。

**核心挑战**: 使用 neural audio codec (如 EnCodec) 的 RVQ 产生的 token 序列很长 (10s 音频 + 8 codebooks + 240x 下采样 = 8000 tokens),标准 Transformer 无法处理 [§IV-E] [论文原文]。

**U-Transformer 解决方案** [§IV-E, Fig 5]: 沿 codebook number 维度先下采样 (CNN),在 latent space 做 Transformer denoising,再上采样恢复 codebook 维度,最后用不同的 output layer 分别预测各 codebook 的 tokens [论文原文]。

**改进的 Mask-and-Uniform 策略** [§IV-E, Eq 9]:

**为什么改进**: RVQ 各层信息量递减 -- 第一层编码 text/style/speaker 主要信息,后续层编码 fine-grained acoustic details。标准策略假设所有 token 等同重要,违反了 "easy-first-generation" 原则 [§IV-E] [论文原文]。

**怎么做**: 根据 token 在 RVQ 层中的位置 i,动态调整 alpha/gamma/beta。前向过程先 mask 最后一层 (信息最少,最难恢复),最后 mask 第一层 (信息最多,最易恢复),使逆向过程遵循 easy-first 生成行为 [§IV-E, Eq 9] [论文原文]。

**证据**: Table VIII 显示改进策略 (I-MAR) 在所有指标上优于标准策略 (MAR),STOI 从 0.582 提升到 0.615 [Table VIII] [论文原文]。

**设计 4: 互信息最小化 (MI Minimization)**

**为什么需要**: audio encoder 从 GT mel 提取 style embedding 时,可能泄漏 speaker identity 和 content 信息 (因为 mel 中包含这些信息)。如果 style embedding 混入了 speaker/content,推理时用 text prompt 替代 audio encoder 时会产生不匹配 [§IV-C] [论文原文]。

**怎么做**: 使用 CLUB (Contrastive Log-ratio Upper Bound) 方法 [52] 同时最小化: (1) style-speaker MI: I(z_e; z_sid),确保 style 不包含 speaker 信息; (2) style-content MI: I(z_e; c),确保 style 不包含 content 信息 [§IV-C, Eq 10] [论文原文]。

**证据**: Table VII 消融显示 MIM + CFG 组合效果最好。Mel-VQ-Diffusion: 无 MIM 无 CFG 时 FFE=0.42, 加 MIM 降至 0.37, 加 CFG 降至 0.32, 两者都加降至 0.30; Wave 版本改进更大 (FFE 0.39→0.23) [Table VII] [论文原文]。

**设计 5: Classifier-Free Guidance 在离散 Diffusion 中**

**为什么引入**: 在 diffusion 后期步骤,当 x_t 已包含较多信息时,模型可能忽略条件信号 y [§IV-D2] [论文原文]。

**怎么做**: 利用 Bayes 定理推导 (Eq 6-7),训练时以 10% 概率用可学习 null vector n 替代条件 y;推理时外推条件/无条件输出差值 (Eq 8): p(x_{t-1}|x_t,n) + (lambda+1)*(p(x_{t-1}|x_t,y) - p(x_{t-1}|x_t,n)) [§IV-D2] [论文原文]。

**与连续 CFG 的区别**: 这里操作在离散 token 的 categorical distribution 上,而非连续空间的 noise prediction 上 [agent 解读]。

### 训练策略

**整体训练目标** [Eq 10]:
L = L_diff + L_var + lambda_1 * I(z_e; c) + lambda_2 * I(z_e; z_sid) + lambda_3 * D_Euc(z_p, z_e) - beta_1 * F_1 - beta_2 * F_2

其中 L_diff 为 diffusion loss (VLB), L_var 为 duration/pitch/energy 预测 loss,I(.) 为互信息 (最小化),D_Euc 为 prompt-audio embedding 距离 (最小化),F_1/F_2 为 MI 估计模型的似然 (最大化) [§IV-F1] [论文原文]。

**训练流程**: 先预训练 Mel-VQ-VAE/Neural Audio Codec (669h 混合数据: 内部中文 300h + VCTK + AISHELL3 + LibriTTS) → 固定后端到端训练 InstructTTS (NLSpeech 44h) [§V-A/B]。

**推理**: T=100 步,每步 delta_t=1。直接用 style prompt embedding 替代 audio encoder 输出 [Algorithm 2, §IV-F2]。

## 实验

### 主要结果 (Table II)

| 指标 | GT | GT(voc) | Baseline | InstructTTS (Mel) | InstructTTS (Wave-GRVQ) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MCD ↓ | 5.02 | 5.75 | 5.59 | **5.59** | 5.68 | NLSpeech | [Table II] |
| SSIM ↑ | 0.695 | 0.422 | **0.487** | 0.487 | 0.384 | NLSpeech | [Table II] |
| STOI ↑ | 0.893 | 0.663 | 0.732 | **0.732** | 0.615 | NLSpeech | [Table II] |
| GPE ↓ | 0.006 | 0.433 | 0.392 | 0.384 | **0.359** | NLSpeech | [Table II] |
| VDE ↓ | 0.076 | 0.286 | 0.246 | 0.193 | **0.151** | NLSpeech | [Table II] |
| FFE ↓ | 0.08 | 0.33 | 0.30 | 0.27 | **0.23** | NLSpeech | [Table II] |
| MOS ↑ | 4.62 | 4.41 | 4.04 | **4.35** | 3.95 | NLSpeech | [Table II] |
| RMOS ↑ | 4.65 | 4.61 | 3.85 | 4.22 | **4.32** | NLSpeech | [Table II] |

关键发现 [§VII-A] [论文原文]:
- Mel-VQ-Diffusion 语音质量更好 (MOS 4.35 vs 3.95),Wave-VQ-Diffusion 韵律保真度更好 (FFE 0.23 vs 0.27, VDE 0.151 vs 0.193)
- Mel 方案频率 20x 下采样可能损失 pitch 信息; Wave 方案在时域直接建模,保留韵律但牺牲部分 acoustic detail [论文原文]
- InstructTTS (Wave) 的 RMOS (4.32) 高于 Mel 方案 (4.22),说明韵律保真度对风格相关性的感知更重要 [agent 解读]

### AXY 偏好测试 (Table III)

| 对比 | 7-point score (正=偏好 Y) | 出处 |
| --- | --- | --- |
| Baseline vs InstructTTS (Mel) | 0.72 | [Table III] |
| Baseline vs InstructTTS (Wave) | 0.84 | [Table III] |

### 情感分类测试 (Table IV)

| 模型 | Sad | Happy | Angry | Overall | 出处 |
| --- | --- | --- | --- | --- | --- |
| GT | 100% | 88.8% | 94.7% | 95.2% | [Table IV] |
| Baseline | 64.28% | 66.6% | 68.15% | 66.7% | [Table IV] |
| InstructTTS (Mel) | 71.42% | 66.6% | 68.4% | 69.1% | [Table IV] |
| InstructTTS (Wave) | 71.42% | 55.5% | **84.21%** | **71.42%** | [Table IV] |

Wave 方案在 Angry 情感上准确率远高于其他方案 (84.21% vs 68.15%),但在 Happy 上最低 (55.5%)。总体上 InstructTTS 优于 baseline [Table IV] [论文原文]。

### Audio Codec 重建对比 (Table IX)

| 模型 | Nq | PESQ | STOI | 出处 |
| --- | --- | --- | --- | --- |
| RVQ (ours, 4 CB) | 4 | 3.24 | 0.91 | [Table IX] |
| GVQ (ours) | 4 | 3.11 | 0.91 | [Table IX] |
| GRVQ (ours) | 4 | **3.63** | **0.95** | [Table IX] |
| Encodec | 12 | 3.21 | 0.95 | [Table IX] |

GRVQ 以 4 codebooks 超越 Encodec 12 codebooks 的重建质量 (PESQ 3.63 vs 3.21),且 STOI 持平 (0.95 vs 0.95) [Table IX] [论文原文]。这解释了为什么 Wave-VQ-Diffusion 默认选择 GRVQ [论文原文]。

### 消融: CFG + MIM (Table VII)

| 配置 | MCD ↓ | SSIM ↑ | FFE ↓ | 出处 |
| --- | --- | --- | --- | --- |
| InstructTTS (Mel) - 无 MIM 无 CFG | 5.75 | 0.421 | 0.42 | [Table VII] |
| InstructTTS (Mel) + MIM | 5.65 | 0.442 | 0.37 | [Table VII] |
| InstructTTS (Mel) + CFG | 5.66 | 0.451 | 0.32 | [Table VII] |
| InstructTTS (Mel) + MIM + CFG | **5.59** | **0.487** | **0.30** | [Table VII] |
| InstructTTS (Wave) - 无 MIM 无 CFG | 5.85 | 0.350 | 0.39 | [Table VII] |
| InstructTTS (Wave) + MIM + CFG | **5.68** | **0.384** | **0.23** | [Table VII] |

MIM 和 CFG 各自带来改善,两者组合效果最好。Wave 方案的改善幅度更大 (FFE: 0.39→0.23) [Table VII] [论文原文]。

## 局限性

1. **数据规模和公开性**: 仅在 44 小时内部中文数据集 NLSpeech 上验证,无公开数据集实验。无法与其他方法在同一 benchmark 上公平对比 [§III, §V] [agent 解读]。

2. **推理速度**: 使用 100 diffusion steps,论文自认这限制了推理速度 [§VIII] [论文原文]。

3. **Wave 方案质量不足**: Wave-VQ-Diffusion 的 MOS 仅 3.95 (vs Mel 方案 4.35),论文承认仍有提升空间 [§VII-A2] [论文原文]。同期 VALL-E 也面临类似的 codec-based 语音质量问题 [§VII-A1] [论文原文]。

4. **缺少直接对比**: 未与同期的 PromptTTS (Guo et al., 2023) 直接对比,两者针对同一问题但使用不同 style prompt 格式 (自由 NL vs 结构化属性) [agent 解读]。Baseline 是改编的 StyleSpeech 模型,并非最强 baseline [§V-C]。

5. **Style Prompt 语言限制**: NLSpeech 为中文数据集,style prompts 也是中文,RoBERTa 从头在中文上训练。系统未验证跨语言泛化能力 [agent 解读]。

6. **Prompt 多义性未解决**: 论文未讨论同一 style prompt 可能对应多种合理语音的 one-to-many 问题 (后续 PromptTTS 2 通过 variation network 部分解决此问题) [agent 解读]。

## 点评

**InstructTTS 的核心贡献在于两个"首次"的交叉**:

第一个"首次"是将自由格式自然语言 (而非结构化属性标签) 用于 TTS 风格控制。与同期 PromptTTS 要求 prompt 包含明确的属性词 (如 "low-pitch, high-speaking speed") 不同,InstructTTS 允许诸如"充满悲伤情绪的叹息语气,带着些许无奈" 这样的自然描述 [§II-C, Table I]。这更符合真实用户的表达方式,但也使机器学习问题更困难,因为模型需要从长且多义的自然语言中提取 compact style representation。

第二个"首次"是将离散 diffusion model 用于 TTS 声学建模。此前离散 diffusion 主要在图像 (VQ-Diffusion) 和音频事件 (DiffSound, 同一作者) 领域。InstructTTS 展示了离散 diffusion 在人声合成中的可行性,且提出了针对 RVQ 多 codebook 结构的改进 masking 策略,这是对离散 diffusion 在结构化离散空间中应用的有价值探索。

**互信息最小化是有洞察力的设计**: audio encoder 提取 style embedding 时不可避免地接触到 speaker 和 content 信息,如果不加约束,训练时 style embedding 会成为包含所有信息的 "catch-all" 表征,导致推理时用 text prompt 替代后产生不一致。MI 最小化强迫 audio encoder 只保留 style 相关信息,从而确保训练-推理的 modality gap 最小化。Table VII 的消融清晰验证了这一设计的有效性。

**历史定位**: InstructTTS 论文首版 arXiv 2023 年 1 月,与 VALL-E 同月发表,代表了 2023 年初 TTS 领域的两个并行方向: VALL-E 走 codec LM (AR 生成 discrete tokens),InstructTTS 走 discrete diffusion (NAR 生成 discrete tokens)。后续发展证明 LM 路线 (VALL-E → CosyVoice → Seed-TTS) 成为主流,discrete diffusion TTS 未被大规模跟进,但 InstructTTS 的 NL style control 思路通过 VoxInstruct 等系统得到延续和发展。从 NL Description 演进线看,InstructTTS 是从"结构化标签"到"统一指令"的重要中间站。

## 可复用的 idea

1. **Cross-modal metric learning 对齐 text-style 空间**: 三阶段训练 (预训练 LM → NLI 微调 → audio-text contrastive learning) 是一种通用的将文本描述与感知属性对齐的方法。在任何"用文字描述感知目标"的生成任务中都可参考 (如音乐风格描述、图像风格描述) [§IV-B, Tables V-VI]。

2. **MI 最小化做 modality-specific 解耦**: 当训练和推理使用不同模态的条件输入时 (训练用 audio,推理用 text),MI 最小化可约束共享表征只编码目标信息。CLUB 作为 MI upper bound 估计器的选择值得关注 [§IV-C, Table VII]。

3. **离散 diffusion 中的 hierarchy-aware masking**: 对于有结构化层级的离散表征 (如 RVQ 的 coarse-to-fine layers),按信息量差异化 masking schedule,让生成过程遵循 easy-first 原则。可迁移到任何基于 RVQ tokens 的离散生成模型 [§IV-E, Eq 9, Table VIII]。

4. **将连续预测转为离散分类降低建模难度**: 用预训练 VQ-VAE 将高相关性的连续表征 (mel spectrogram) 转为低相关性的离散 tokens,将回归问题转为分类问题。尤其适用于高维、高相关性的目标空间 [§IV-D]。

## 审阅

> [!review] 审阅 (2026-06-08, self)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 五模块架构 + 三阶段训练的 WHY/HOW 均有因果解释;Mel-VQ-Diffusion 和 Wave-VQ-Diffusion 两条路线清晰对比 |
> | 可信赖 | pass | 数字型 claim 标注 [Table N]/[§X.X] 覆盖率 >90%;指标名 (MCD/SSIM/STOI/GPE/VDE/FFE/MOS/RMOS/PESQ) 使用正确 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖方法节所有因果解释 |
> | 可定位 | pass | KB 背景有具体谱系定位 (对标 PromptTTS、VALL-E、VoxInstruct);创新判断基于技术路线对比 |
> | 不污染 | pass | 未创建新概念页;概念引用均指向已有实体页 |
> 
> Issues: 2 (high: 0, medium: 2, low: 0)
> 
> **medium-1 (traceability-gap)**: 点评节中关于 "discrete diffusion TTS 未被大规模跟进" 的判断缺少具体文献支撑 — 虽然从 KB 演进线可推断,但应标注 [agent 解读] 以区分来源。
> **medium-2 (template-compliance)**: 论文 venue 为 arXiv preprint (v2 Jun 2023),后续是否被会议录用未确认;frontmatter 标注为 arXiv 是准确的但可补充说明该论文的实际发表状态。
