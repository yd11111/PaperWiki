---
type: paper
tier: deep
title: "UNISON: Unified Sound Generation and Editing via Deep LLM Fusion"
arxiv_id: "2605.31530"
source: "Sources/UNISON.pdf"
authors: [Zhaoqing Li, Haoning Xu, Jingran Su, Yaofang Liu, Zhefan Rao, Huimeng Wang, Jiajun Deng, Tianzi Wang, Zengrui Jin, Rui Liu, Haoxuan Che, Xunying Liu]
year: 2026
venue: "arXiv"
tags: [unified-audio, text-to-audio, TTS, zero-shot, audio-editing, flow-matching, MM-DiT, deep-fusion, multi-task, latent-diffusion]
concepts: ["[[ConditionalFlowMatching]]", "[[Classifier-FreeGuidance]]", "[[DiffusionModel]]", "[[Instruction-GuidedSpeechSynthesis]]", "[[LLM-basedTTS]]", "[[VariationalAutoencoderforTTS]]"]
models: ["[[CosyVoice2]]", "[[Whisper]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[InstructedSpeechGeneration]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[Emilia]]", "[[AudioSet]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[Zero-shotSpeechSynthesis]]✓, [[SEED-TTS-Eval]]✓ | 过滤: [[Classifier-FreeGuidance]](pending-review), [[Instruction-GuidedSpeechSynthesis]](pending-review) | 未命中但可能相关: 无

**谱系定位**: UNISON 处于 "统一音频生成" 这条发展线上,与 AudioBox (flow-matching infilling)、UniAudio (next-token prediction)、Audio-Omni (hybrid MLLM cross-attention)、UniSonate (phoneme-driven MM-DiT) 等系统并行竞争。在 TTS 线上,它与 [[ConditionalFlowMatching]] 的 CFM+DiT 路线(CosyVoice 系列、F5-TTS、MaskGCT)共享 flow matching 基础,但区别在于:

1. **文本条件方式不同**: 现有 CFM-TTS 系统通常使用单层 LLM 嵌入(最后一层或倒数第二层)广播到所有 DiT block。UNISON 提出 layer-wise deep fusion,将 LLM 各层隐状态逐层注入对应 DiT block。这一思路源自图像生成领域(Tang et al., 2025; HiDream),在音频领域属首次应用。
2. **统一程度不同**: CosyVoice、F5-TTS 等为 TTS 专用系统,AudioBox、UniSonate 虽走统一路线但仍依赖 task-specific 辅助模块(phoneme encoder、mel encoder)。UNISON 用一个 mask channel + VAE channel concatenation 统一所有任务,无额外编码器。
3. **KB 已有知识**: [[Classifier-FreeGuidance]][待确认] 记录了 CFG 在 flow-matching TTS 中的广泛使用(dropout 10-20%, scale 0.7-4.5);UNISON 采用 10% dropout + scale 4.5。[[Instruction-GuidedSpeechSynthesis]][待确认] 记录了从 VoxInstruct 到 UniSonate 的统一指令范式演进;UNISON 延续此路线但用纯文本 LLM 指令替代 phoneme pipeline。

**创新判断**: 相对于 KB 已有认知,UNISON 的核心新贡献是 (1) layer-wise deep LLM fusion 在音频生成中的首次应用,以及 (2) 用最少的 task-specific 设计(仅一个 mask channel)实现 generation+editing 的真正统一。参数效率(621-732M vs Audio-Omni 3.05B)也值得关注。

## 速查

> [!summary] 速查
> - **一句话**: 用 layer-wise deep LLM fusion + channel-mask 统一架构,在 621-732M 参数的单模型中同时实现 T2A、TTS、零样本克隆、音频编辑和时序组合,性能匹敌或超越各任务的专用模型
> - **路线**: Text instruction → frozen Qwen2.5-Omni-7B (逐层 hidden states) → learned projectors → DeepFusion MM-DiT (flow matching) ← [z_t || z_s || mask] via frozen MMAudio VAE → ODE solver → VAE decode → waveform
> - **指标**: T2A FAD 1.558 / CLAP 0.503 (AudioCaps, D24); TTS WER 1.27% EN / CER 0.92% ZH (Seed-TTS, D24); 音频编辑 overall FD 12.38 / CLAP 0.364 (vs MMEDIT 20.60/0.257) [Table 1, 2, 5]
> - **可借鉴**: (1) Deep fusion: 将 frozen LLM 不同层隐状态通过 uniform sampling 映射到 DiT 各 block,低层给词汇/音素信息、高层给语义信息,改善复合指令跟随; (2) Channel-mask 统一: 仅用一个标量 mask channel (0/1/2) 区分 generation/editing/cloning,source audio 通过同一 VAE 编码后 channel concatenation,避免 task-specific 模块; (3) 零初始化 zs/m 连接权重,使训练初期行为类似纯去噪,逐步学习利用 source/mask 通道
> - **局限**: (1) MMAudio VAE 对语音高频 formant 重建有损,是 TTS 音质上限的瓶颈; (2) 编辑训练数据为合成混合(简单 RMS mixing),与真实场景的混响/遮蔽/Lombard 效应有差距; (3) 不支持音乐生成; (4) 仅支持中英文; (5) 未开源

## 核心问题

现有统一音频系统存在两个根本限制: (1) 各任务仍依赖异构辅助模块(mel encoder、phoneme front-end、duration predictor 等),碎片化了潜空间,限制了跨任务知识迁移; (2) 文本条件通常取 LLM 单层表示广播到所有 DiT 层,丢失了 LLM 内部的层级语义结构(浅层偏词汇/句法,深层偏抽象语义),削弱了对复合音频指令的跟随能力。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UNISON 是一个 latent diffusion (flow matching) 框架,所有任务共享同一 VAE、同一 DiT backbone、同一前向传播路径 [§1]。

**输入构造**: 对每个训练样本,构建文本指令 + 可选 source waveform。用 frozen MMAudio VAE 编码得到 target latent z 和 source latent z_s。采样 flow time t,构造 noisy target z_t = (1-sigma_t)*z + sigma_t*epsilon。通道拼接 X = [z_t || z_s || m],经 Conv-MLP embedder 映射为 audio tokens h_0 [§3.1]。

**文本条件**: frozen Qwen2.5-Omni-7B Thinker 运行一次,返回所有 L=28 层隐状态。通过 uniform sampling 选取 D 个层(D = DiT block 数),每个 block k 接收对应 LLM 层 i_k 的隐状态经 learned linear projector 投影后的文本 token [§3.4, Eq.2]。

**DiT backbone**: DeepFusion MM-DiT,采用 double-stream 架构。每个 block 中音频 token 和文本 token 通过 joint attention 交互,但只有音频 token 通过 MLP 更新;文本 token 每层重新从 LLM 获取(ephemeral),不在 block 间传递。最终线性头输出 velocity v_theta,仅预测 target channels [§3.4]。

**输出**: 100 步 Euler ODE solver 从噪声积分到 clean latent,VAE decode 得波形 [§3.5]。

### 关键设计选择

**1. Layer-wise deep LLM fusion (vs 单层 embedding)**

[论文原文] Probing studies (Tenney et al., 2019; Clark et al., 2019) 表明 transformer LM 的表示是层级化的: 浅层编码词汇/句法,深层编码抽象语义。将单层 final hidden state 广播到所有 DiT block 会丢弃这种层级结构,限制对复合指令(同时指定说话人属性、声学事件、时序结构)的跟随能力 [§1]。

[论文原文] 具体实现: 从 L=28 层 Qwen 中 uniform sampling 选 D 层,i_k = 1 + k * (L-1)/(D-1),第 k 个 DiT block 收到第 i_k 层的投影。这确保浅 DiT block 看到浅 Qwen 层(词汇/音素结构),深 DiT block 看到深层语义 [§3.4]。

[agent 解读] 这个设计的核心直觉是 "depth-matched conditioning": 生成过程中,DiT 浅层负责粗粒度布局(什么声音在哪里),深层负责细粒度细节(精确音色/语调);LLM 浅层的词汇信息恰好匹配粗粒度需求,深层的语义信息匹配细粒度需求。这在图像领域已被 Tang et al. (2025) 验证,UNISON 是音频领域的首次应用。

消融验证: D24-L (penultimate layer only) CLAP 0.175 vs D24-O (deep fusion) CLAP 0.180,FD 22.71 vs 20.46 [Table 8]。

**2. Ephemeral text tokens (vs persistent text stream)**

[论文原文] 每个 DiT block 的文本 token 不传递到下一个 block,而是每层从 Qwen 重新获取。因为 Qwen 的 hidden states 已经编码了丰富语义,DiT 不需要重新学习语言结构;跳过 text MLP 也节省计算 [§3.4]。

消融验证: D24-OL (deep fusion + persistent last-layer stream) 虽然在 T2A 上 FD/CLAP 最好(20.18/0.187),但 TTS WER 最高(5.52%),因为双重文本条件引入冗余噪声 [Table 8]。D24-O (仅 deep fusion, ephemeral text) 在 T2A 和 TTS 之间取得最佳平衡 [论文原文]。

**3. Channel-mask 多任务统一**

[论文原文] 所有任务共享同一网络,仅通过 (z, z_s, m) 和文本指令区分:
- m=0: 生成任务(T2A, TTS, T2AS, timed composition),z_s=0
- m=1: 编辑任务,z_s 为 source audio 的 VAE latent
- m=2: 零样本 TTS,z_s 编码参考前缀,tag 区分参考区和合成区 [§3.3]

[agent 解读] 这个设计的关键优势是: source/reference audio 通过与 target 完全相同的 frozen VAE 编码,确保两者在同一潜空间中操作。这避免了 Audio-Omni 的 separate mel encoder 和 MMEDIT 的 separate Qwen2-Audio encoder 导致的潜空间碎片化。

**4. 零初始化渐进学习**

[论文原文] Conv-MLP embedder 中连接 z_s 和 m 到 token space 的权重零初始化,而 z_t 使用标准初始化。这使训练初期行为等同于纯去噪,模型逐步学习利用 source 和 task channels [§3.4]。

[agent 解读] 这是一种常见的 residual learning 技巧(类似 ControlNet 的 zero-conv),解决了多任务训练初期梯度冲突的问题。

**5. Double-stream vs single-stream**

消融结果: S32-O (single-stream, 32 blocks) 在所有指标上都差于 D24-O (double-stream, 24 blocks) — FD 23.19 vs 20.46, CLAP 0.169 vs 0.180, WER 4.84% vs 4.33% [Table 8]。[论文原文] 共享 normalization 和 QKV 投影阻止了模态特异性表示的形成;double-stream 通过分离特征空间、仅在 joint attention 中交互来避免此问题 [§4.4]。

### 训练策略

**Online multi-task data synthesis**: 不构建静态数据集,而是在 GPU 端 on-the-fly 从原始音频/语音片段构造所有任务变体。包括 RMS normalization、SNR 控制混合、边界 fade-in/out、随机时间偏移 [§3.6, Table 10]。

**Two-stage curriculum**: Stage 1 (前 150K 步) 仅训练生成任务,建立稳定的生成先验。Stage 2 引入所有编辑任务,约 70% 生成 / 30% 编辑 [§3.7]。

[agent 解读] 分阶段课程学习是解决 generation vs editing 梯度冲突的关键。如果一开始就同时训练 "add event" 和 "remove event",模型会收到矛盾的梯度信号。先学好 "如何生成",再学 "如何在已有基础上修改",是更稳定的路径。

**Task-homogeneous batching**: 每个 mini-batch 只包含单一任务类型,防止 batch 内梯度冲突 [§3.7]。

**训练配置**: AdamW (beta1=0.9, beta2=0.95), lr=1e-4 cosine decay + 2000 step warmup, batch 56/GPU x 8 H800, BF16, EMA 0.999, CFG dropout 0.1 [§4.1] [Table 13]。

**训练数据**: ~36M clips (~57K hours): 2.3M audio clips (WavCaps + AudioSet + VGGSound, ~6.4K h) + 33.7M speech clips (LibriTTS + WenetSpeech + [[Emilia]] EN/ZH, ~50.7K h) [Table 11]。编辑评估: SNR in [-3, 3] dB (audio editing), 0 dB (T2AS), 10 dB (speech-in-scene) [§4.2]。

## 实验

| 指标 | 本文 (D24) | 本文 (D20) | Baseline (最强) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| T2A FAD | 1.558 | 1.756 | Audio-Omni 2.535 / MMAudio-L 5.893 | AudioCaps | [Table 1] |
| T2A CLAP | 0.503 | 0.467 | GenAU-L 0.561 (20x data) | AudioCaps | [Table 1] |
| T2A FD | 16.28 | **15.82** | MMAudio-L 16.53 | AudioCaps | [Table 1] |
| T2A IS | 10.90 | **12.04** | MMAudio-L 11.98 | AudioCaps | [Table 1] |
| Pure TTS WER-EN | **1.27%** | 1.42% | Audio-Omni 1.35% | Seed-TTS | [Table 2] |
| Pure TTS CER-ZH | **0.92%** | 1.11% | UniSonate 1.25% | Seed-TTS | [Table 2] |
| ZS TTS WER-EN | **1.50%** | 1.80% | ZipVoice 1.70% | Seed-TTS | [Table 2] |
| ZS TTS CER-ZH | **0.89%** | 1.71% | CosyVoice 2 1.45% | Seed-TTS | [Table 2] |
| Gender TTS Accuracy | 100% | 100% | - | 300 balanced | [Table 3] |
| Audio Edit FD (overall) | **12.38** | 13.44 | MMEDIT 20.60 | 1200 constructed | [Table 5] |
| Audio Edit CLAP (overall) | **0.364** | 0.289 | MMEDIT 0.257 | 1200 constructed | [Table 5] |
| Speech-Scene Delete Removal | **99.16%** | 95.72% | - | 200 constructed | [Table 6] |
| Timed Composition per-seg CLAP | 0.308 | 0.311 | - | 150 constructed | [Table 7] |

**D24 vs D20 trade-off**: D24 (16kHz, 732M, 24 blocks) 在 FAD/CLAP/WER 等语义相关指标上更优; D20 (44.1kHz, 621M, 20 blocks) 因更高带宽 VAE 在 FD/IS/LSD 等频谱保真指标上更优 [§4.3.1]。

**LLM scale**: D24-O-3B 相比 D24-O-7B,FD 20.46→21.53, CLAP 0.180→0.174, WER 4.33%→5.61% [Table 8],确认更大 LLM 直接提升语义跟随和语音清晰度。

**关键观察**:
- UNISON 在 TTS 上**不使用 phoneme encoder**,纯文本 LLM 指令即可达到甚至超越使用 G2P 的系统(UniSonate, MaskGCT) [§4.3.2]
- 多任务训练未降低 TTS 质量: pure TTS WER 1.27% 优于所有单任务 baseline [Table 2]
- 性别控制完全通过文本指令实现(无 speaker embedding/gender label),准确率 100% [Table 3]
- UNISON 是评估范围内唯一同时覆盖 T2A + TTS + zero-shot cloning + audio editing + speech-in-scene editing + timed composition 的系统 [Table 9]

## 局限性

1. **VAE 瓶颈**: 依赖 MMAudio VAE,该 VAE 为环境音设计,对语音高频 formant、细微韵律变化、耳语等的重建保真度有限,是零样本 TTS 音色细节的上限 [Limitations §1]
2. **合成编辑数据**: 编辑训练数据通过 RMS mixing 构造,缺乏真实场景的混响/遮蔽/Lombard 效应;标注来自 AudioSet/WavCaps,未经人工验证,存在 label noise [Limitations §2]
3. **模型与数据规模**: 621-732M DiT + ~36M clips (~57K h) 在近期 scaling 趋势中属中等;架构支持 scaling 但未验证 [Limitations §3]
4. **语言与领域**: 仅支持中英文语音;不覆盖音乐生成(缺乏开放许可的高质量音乐数据) [Limitations §4]
5. **评估局限**: 编辑和混合生成任务无公开 benchmark,使用自构造测试集,难以直接与后续工作对比
6. **无 speaker similarity 指标**: TTS 评估仅报告 WER/CER,缺少说话人相似度(SIM)指标,难以评估零样本克隆的音色保真度

## 点评

UNISON 的核心价值在于用极简的 task-specific 设计(一个标量 mask channel)实现了真正的生成-编辑统一,同时在参数效率上(621-732M vs Audio-Omni 3.05B)具有明显优势。Layer-wise deep fusion 是一个概念上简洁、实验上有效的想法,将图像生成领域的 insight 成功迁移到了音频,消融实验(Table 8)清晰地验证了其对语义跟随的提升。

方法论上的 elegance 体现在: 所有任务共享同一 VAE/DiT/forward pass,仅凭 mask channel 和 channel concatenation 区分任务。这种设计使得跨任务知识迁移(如生成能力帮助编辑)自然发生。消融显示 double-stream 架构和 7B LLM scale 都是关键因素。

但也有值得注意的问题:
- **TTS 评估不完整**: 缺少 speaker similarity 指标,仅有 WER/CER 难以全面评估零样本克隆质量。KB 中 [[SEED-TTS-Eval]] 的标准评估包含 SIM,UNISON 的缺失是显著遗漏。
- **编辑 benchmark 可复现性**: 所有编辑/混合任务使用自构造测试集,与 MMEDIT、SDEdit 等对比可能存在 evaluation bias。
- **VAE 选择的矛盾**: 强调"统一"却使用为环境音设计的 MMAudio VAE,实际上为语音质量设了上限。这暗示真正的统一系统可能需要一个同时优化语音和通用音频的 VAE。
- **与 UniSonate 的对比不完全公平**: UniSonate (1.34B) 覆盖了音乐生成,UNISON 不覆盖;且 UniSonate 的 CLAP 未报告,无法直接在 T2A 上对比。

## 可复用的 idea

1. **Layer-wise deep LLM fusion**: 将 frozen LLM 各层隐状态通过 learned projectors 注入到 DiT 对应 block。实现简单(uniform sampling + linear projection),对任何使用 LLM 条件的 DiT 系统都适用。关键: 文本 token 每层"刷新"(ephemeral),不在 block 间传递,节省计算并避免冗余。
2. **Channel-mask 多任务统一**: 用一个标量 mask channel (0/1/2) + source latent channel concatenation 统一 generation/editing/cloning。source/target 共享同一 VAE 确保潜空间一致。可推广到其他 conditional generation 任务(如 video-to-audio)。
3. **零初始化渐进学习**: 新增条件通道(source/mask)的连接权重零初始化,使训练初期等同于无条件去噪,逐步学习利用新通道。避免多任务训练初期的梯度冲突。
4. **Task-homogeneous batching + two-stage curriculum**: 每个 batch 只含一种任务类型 + 先训练生成再引入编辑,解决 generation vs editing 的梯度冲突。
5. **Online GPU-side data synthesis**: 不预构建静态数据集,而是在 GPU 端 on-the-fly 从原始音频构造所有任务变体(mixing, fading, temporal offset)。减少存储开销,增加数据多样性。

---

检索命中: [[ConditionalFlowMatching]]✓, [[LLM-basedTTS]]✓, [[Zero-shotSpeechSynthesis]]✓, [[SEED-TTS-Eval]]✓ | 过滤: [[Classifier-FreeGuidance]](pending-review), [[Instruction-GuidedSpeechSynthesis]](pending-review) | 未命中但可能相关: 无

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节因果解释充分,消融验证清晰 |
> | 可信赖 | pass | 核心数字交叉验证通过,2 处 traceability gap (medium/low) |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率 >80% |
> | 可定位 | pass | KB 背景谱系定位详细,创新判断有对比基准 |
> | 不污染 | pass | 无新概念页,反向更新均为追加操作 |
> 
> Issues: 4 (high: 0, medium: 1, low: 3)
> 详见 `_review/UNISON-review.yml`
