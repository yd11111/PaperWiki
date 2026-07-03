---
type: paper
tier: deep
title: "LLM can Read Spectrogram: Encoder-free Speech-Language Modeling"
arxiv_id: "2606.10231"
source: "Sources/Mel-LLM.pdf"
authors: [Ruchao Fan, Yiming Wang, Yuxuan Hu, Bo Ren, Yufei Xia, Xiaofei Wang, Yao Qian, Shujie Liu, Jinyu Li]
year: 2026
venue: "arXiv"
tags: [speech-LLM, encoder-free, mel-spectrogram, ASR, speech-understanding, TTS, LoRA, multimodal, native-multimodal]
concepts: ["[[MelSpectrogram]]", "[[ModalityAdaptationforSpeechLLM]]", "[[SpeechLanguageModel]]", "[[LLM-basedTTS]]", "[[SemanticvsAcousticTokens]]", "[[AudioUnderstanding]]", "[[VariationalAutoencoderforTTS]]", "[[LLM-enhancedASR]]"]
models: ["[[Mel-LLM]]", "[[Whisper]]", "[[MELLE]]", "Phi-4-MM", "Gemma-4-12B-it", "WavLLM", "SALMONN", "LTU-AS", "VibeVoice"]
tasks: [ASR, speech-understanding, TTS, emotion-recognition, speaker-verification, audio-classification]
datasets: ["[[LibriSpeech]]", "GigaSpeech", "MLS-English", "SPGISpeech", "CommonVoice 15", "VoxPopuli", "TED-LIUM", "AMI", "Earnings-22", "FLEURS", "Libriheavy", "LibriSpeech-PC", "ESC-50", "IEMOCAP", "VoxCeleb", "GTZAN", "MMAU-mini", "MMLU-speech"]
kb_context_sources: 6
status: draft
created: 2026-07-03
updated: 2026-07-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 3 个待确认实体页: [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓, [[MelSpectrogram]][待确认], [[ModalityAdaptationforSpeechLLM]][待确认], [[MELLE]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 当前 Speech-LLM 主流范式是 encoder-projector-LLM 三件套 ([[SpeechLanguageModel]]), 语音编码器 (Whisper/Conformer) 是必选件, 通过 [[ModalityAdaptationforSpeechLLM]] (Conv downsampling / CTC compression / Q-Former) 桥接到 LLM。本文直接挑战这一范式,去掉 encoder 主体,仅保留轻量卷积下采样 + 线性投影。
>
> **已有认知 - Mel Spectrogram**: [[MelSpectrogram]] 在 TTS 中是声学模型与 vocoder 之间的经典中间表示, 但在 Speech-LLM 理解任务中, mel 通常经过编码器处理后才进入 LLM。直接将 mel patch 喂给 LLM 是一个未被充分验证的方向。近期 LongCat-AudioDiT 和 WavTTS 指出 mel 作为中间表示存在相位丢失和 compounding error 问题, 但本文关注的是 mel 作为 LLM 输入而非输出。
>
> **已有认知 - Token 类型 trade-off**: [[SemanticvsAcousticTokens]] 的核心 trade-off (语义连贯 vs 声学保真) 在本文中以新形式出现 -- encoder-free 直接暴露 mel 保留了更多声学细节 (paralinguistic cues), 但削弱了语义锚定 (semantic anchoring), 这是 token 二分法在连续表征维度的映射。
>
> **创新判断**: 相对于 KB 中已知的 modality adaptation 方法 (Conv/CTC/Q-Former), encoder-free 是最极端的简化 -- 直接跳过编码器, 将原始频谱特征交给 LLM 自行学习。Vision 领域 Fuyu/EVE/Tuna-2 已验证此路线, 但语音领域的系统性 encoder-based vs encoder-free 对比此前缺失。MELLE 提供了连续 mel AR 的 TTS 框架, 本文的 TTS 部分直接继承。
>
> 检索命中: [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[MelSpectrogram]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review), [[MELLE]](pending-review) | 未命中但可能相关: 无

> [!summary] 速查
> - **一句话**: 去掉 Speech-LLM 中的专用语音编码器, 将 mel spectrogram patches 通过线性投影直接喂给 LLM, 证明 LLM 自身参数足以学习语音-文本对齐
> - **路线**: 80-dim log-mel → MVN → (optional Conv downsample r=8) → Linear Proj → Phi-4-MM (32L, LoRA r=320) → text (ASR/QA) 或 mel frames (TTS via VAE decoder)
> - **指标**: ASR OpenASR avg WER 7.12% (encoder-free) vs 5.61% (encoder+pretrained LoRA) vs 6.97% (random encoder) [Table II]; 10x 数据缩小差距至 +3.8% rel [Table III]; IEMOCAP emotion 40.69→66.16, gender 92.57→97.73 [Table VI]
> - **可借鉴**: (1) layer-freezing 分析发现 LLM 上层 (L24+) 对语音适配贡献小, 冻结后几乎无损 -- 可用于高效训练; (2) Phi-4-MM 多模态预训练初始化对低资源 encoder-free 至关重要 (7.12% vs 7.44% random init); (3) 直接暴露 mel 给 LLM 反而有助于 paralinguistic 任务
> - **局限**: (1) MMLU-speech 显著下降 (53.12→41.30), 语义锚定是瓶颈; (2) TTS 仅为 proof-of-concept, WER 11.03 远落后于 latent diffusion baseline 4.2; (3) 未开源; (4) 仅英语实验

## 核心问题

Speech-LLM 主流范式依赖大型预训练语音编码器 (Whisper 600M+) 将音频转换为语义表征后再送入 LLM。这引入三个问题 [§I]:
1. **计算开销**: 编码器本身参数量大, 增加推理成本
2. **表征不匹配**: 编码器的表征不一定适合 LLM 的内部处理, late fusion 导致 representational mismatch
3. **信息瓶颈**: LLM 只能通过编码器的压缩表征访问语音, 编码器决定了保留哪些信息 (semantic or acoustic)

**本文的核心问题**: LLM 能否直接读取 mel spectrogram, 不经过专用语音编码器, 仅靠自身参数学习语音理解?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Mel-LLM 基于 Phi-4-MM (14B, 32 层, d=3072), 通过 LoRA (r=320, alpha=640) 适配 [§IV-A, Table I]。架构分为理解和生成两条路径, 共享 LLM backbone [Fig 1]:

**理解路径** (Fig 1a):
```
80-dim log-mel → MVN normalization → (Conv ↓r, optional) → Linear Proj → LLM → text response
```

**生成路径** (Fig 1b):
```
text input → LLM → VAE decoder (linear → μ/σ → reparameterize → residual MLP → postnet) → mel frames
```

关键参数 [Table I]:
- Removed: Transformer/Conformer speech encoder blocks
- Retained: NeMoConv downsampling (8x, 1024 channels, 12.6M params)
- Audio projection: 2-layer MLP, 1024→3072→3072, 12.6M params
- Total non-LLM audio stack: 25.2M params (vs 编码器通常 600M+)
- 主 token rate: 12.5Hz
- 训练加速: 1.57x vs encoder-based baseline

### 关键设计选择

**1. 为什么 encoder-free 可行?**

[论文原文] Vision 领域 Fuyu/EVE/Tuna-2 已证明 LLM 可直接处理原始 pixel patches 无需 vision encoder [§I]。语音的 mel spectrogram 与图像类似, 也是时频二维表征, LLM 有足够容量在内部学习 modality-specific processing [§I]。

[agent 解读] 80-dim mel spectrogram 的信息密度远低于原始波形, 且 mel 映射本身就是一种先验压缩 (模拟人耳非线性感知)。相比视觉 patch embedding 需要处理数百维像素, mel patch 的低维特性可能使 encoder-free 更容易收敛。

**2. 为什么保留轻量卷积下采样?**

[论文原文] 卷积层仅用于 temporal downsampling purpose, 不做特征提取 [§I]。在 r=1 (无下采样) 时直接禁用卷积层 [§III-B]。

[agent 解读] 这是效率与性能的平衡 -- 100Hz mel 直接输入 LLM 序列过长 (10s 语音 = 1000 tokens), 8x 下采样到 12.5Hz (125 tokens/10s) 使训练可行。消融实验 (Table IV) 确认了这一 trade-off。

**3. 为什么 Phi-4-MM 初始化关键?**

[论文原文] Phi-4-MM 的预训练 LoRA 已包含 speech-text alignment 知识 (在 12.5Hz token rate 下训练), 随机初始化模型必须从零学习这些对齐 [§V-A]。

[agent 解读] 这暗示 encoder-free 方法对初始化非常敏感 -- 没有编码器提供的 inductive bias, LLM 需要从多模态预训练中获得替代的 prior。这一依赖可能限制了方法的通用性 (需要合适的多模态 checkpoint)。

**4. Acoustic-Semantic Trade-off**

[论文原文] 去掉编码器后, LLM 被迫直接建模 Phi-4-MM 的语义编码器可能压缩掉的信息 (speaker traits, prosody, timbre, emotion, music texture) [§V-C]。但同时也移除了 semantic anchor, 增加了 audio-to-reasoning bottleneck [§V-C]。

[论文原文] MMLU-speech 的下降不太可能完全由文本推理能力灾难性丢失造成, 因为 text-only MMLU 仅从 61.0 降到 58.68。真正的瓶颈在 semantic anchoring 和 audio-to-reasoning alignment [§V-C]。

### 训练策略

**ASR 训练** [§IV-B]:
- 数据: ~64k hours 公开英语数据 (LibriSpeech 960h + GigaSpeech 10kh + MLS-English 44kh + SPGISpeech 5kh + CommonVoice 15 + VoxPopuli + TED-LIUM + AMI + Earnings-22 + FLEURS)
- 硬件: 16x H100, DeepSpeed ZeRO Stage-1
- 优化: AdamW, lr=1e-4, linear warmup-decay (9000 warmup steps), gradient clip 1.0, batch size 512
- LLM 层冻结, 仅 LoRA 可训练 (~503M params)
- 训练 3 个 epoch, 更多 sweep 无额外收益

**扩展 Speech Understanding 训练** [§IV-B]:
- 在 ASR 数据基础上增加 speech QA + audio/speech understanding data
- 总计 55.3M weighted examples (Phi-4-MM post-training data 子集)
- 包含: multi-turn SQA, WavLLM-style QA, LTU-AS instruction data, VoxCeleb, MOSEI, AudioSet/FSD50K/AudioCaps/Clotho

**TTS 训练** [§IV-B]:
- 数据: Libriheavy 50k hours English
- Dropout 0.5, KL weight 0.05, stop loss weight 1.0, flux loss 0.5
- 5 epochs

**Production-scale 训练** [§IV-B]:
- ~10x 匿名内部数据, 单次迭代

## 实验

### ASR: OpenASR Leaderboard

| 系统 | Encoder Init | LoRA Init | Avg WER(%) | 出处 |
| --- | --- | --- | --- | --- |
| Whisper-Large-V3 | N/A | N/A | 7.44 | [Table II] |
| Gemma-4-12B-it | None | N/A | 26.76 | [Table II] |
| Phi-4-MM (zero-shot) | N/A | N/A | 6.14 | [Table II] |
| Phi-4-MM + FT (encoder) | Pretrained | Pretrained | 5.61 | [Table II] |
| Phi-4-MM + Random Enc FT | Random | Pretrained | 6.97 | [Table II] |
| **Mel-LLM (Phi-4-MM init)** | **None** | **Pretrained** | **7.12** | [Table II] |
| Mel-LLM (Random init) | None | Random | 7.44 | [Table II] |

Mel-LLM (encoder-free, Phi-4-MM init) 仅比 random encoder baseline 高 0.15% WER (7.12 vs 6.97), 比 pretrained encoder 高 1.51% (7.12 vs 5.61) [Table II]。

### ASR: Production-Scale 数据缩放

| Test Set | Enc-Init | Enc-Free (limited) | Delta rel. | Enc-Free (10x) | Delta rel. | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Call Center | 15.92 | 18.28 | +14.8% | 16.74 | +5.2% | [Table III] |
| Conversation | 15.83 | 17.10 | +8.0% | 16.25 | +2.7% | [Table III] |
| Dictation | 5.80 | 6.40 | +10.3% | 5.99 | +3.3% | [Table III] |
| **Average** | **12.52** | **13.93** | **+11.3%** | **12.99** | **+3.8%** | [Table III] |

数据量从有限公开数据扩展到 10x 内部数据后, encoder-free 与 encoder-init 的差距从 +11.3% 缩小到 +3.8% relative [Table III]。

### Token Rate 消融

| Token Rate | Avg WER(%) | Training Speedup | 出处 |
| --- | --- | --- | --- |
| 100Hz | 6.58 | 0.33x | [Table IV] |
| 50Hz | 6.71 | 0.65x | [Table IV] |
| 25Hz | 7.21 | 1.09x | [Table IV] |
| 12.5Hz | 7.12 | 1.57x | [Table IV] |
| 6.25Hz | 8.02 | 1.88x | [Table IV] |

[论文原文] 12.5Hz 略优于 25Hz (7.12 vs 7.21), 因为 Phi-4-MM 的预训练 LoRA 在 12.5Hz token rate 下训练, 12.5Hz 直接受益于此初始化, 而 25Hz 需要适配不匹配的帧率 [§V-B]。

### Layer-wise Freezing 消融

| 系统 | Avg WER(%) | 出处 |
| --- | --- | --- |
| Mel-LLM (all LoRA) | 7.12 | [Table V] |
| + freeze L28-31 | 7.40 (+0.28) | [Table V] |
| + freeze L24-31 | 7.43 (+0.31) | [Table V] |
| + freeze L20-31 | 7.77 (+0.65) | [Table V] |
| + freeze L16-31 | 7.94 (+0.82) | [Table V] |

[论文原文] 冻结 L24-31 仅降低 0.31%, 但冻结延伸到 L20 及以下后性能快速下降。这表明上层 (24+) 已编码高层语言语义 (reasoning, discourse, text generation), 可不经语音适配直接迁移; 下层 (0-23) 对将原始频谱模式转化为语言有意义表征更为关键 [§V-B]。

### Speech/Audio Understanding

| 任务 | Phi-4-MM | Mel-LLM | Mel-LLM+ASR init | 出处 |
| --- | --- | --- | --- | --- |
| ESC-50 (ACC↑) | 35.95 | 67.85 | 67.00 | [Table VI] |
| IEMOCAP (ACC↑) | 40.69 | **66.16** | **72.36** | [Table VI] |
| Gender (ACC↑) | 92.57 | **97.73** | **98.21** | [Table VI] |
| Age (MAE↓) | 8.06 | 8.39 | **7.56** | [Table VI] |
| GTZAN (ACC↑) | 40.54 | 46.85 | **53.55** | [Table VI] |
| WavLLM-spk (ACC↑) | 76.64 | **87.98** | **92.10** | [Table VI] |
| MMAU-mini (ACC↑) | 50.70 | **54.10** | **56.30** | [Table VI] |
| MMLU-speech (ACC↑) | **53.12** | 41.30 | 36.47 | [Table VI] |
| MMLU-Text (ACC↑) | 61 | 58.68 | 56.73 | [Table VI] |

**模式**: Mel-LLM 在依赖声学/副语言线索的任务上显著优于 Phi-4-MM (emotion +25.47, gender +5.16, speaker +11.34), 但在知识密集型 spoken QA (MMLU-speech) 上下降 11.82 [Table VI]。

### TTS (Preliminary)

| 系统 | WER(%)↓ | UTMOS↑ | 出处 |
| --- | --- | --- | --- |
| Phi-4-MM + latent diffusion | 4.2 | 3.41 | [Table VII] |
| Mel-LLM (Random init) | - (no audible output) | - | [Table VII] |
| Mel-LLM (Phi-4-MM, no norm) | 11.03 | 3.10 | [Table VII] |
| Mel-LLM (Phi-4-MM, MVN) | 14.75 | 3.25 | [Table VII] |

TTS 仅为 proof-of-concept, 连续 mel 生成可行但远不及 latent diffusion baseline (WER 11.03 vs 4.2, UTMOS 3.10 vs 3.41) [Table VII]。随机初始化可收敛但产生不可听的输出 [Table VII]。

## 局限性

1. **Semantic anchoring gap**: MMLU-speech 从 53.12 降至 41.30, 表明 encoder-free 虽保留声学细节但削弱了语义锚定能力 [Table VI]。作者指出这不是文本推理能力的灾难性丢失 (text MMLU 仅微降), 而是 audio-to-reasoning alignment 问题 [§V-C]。
2. **对多模态预训练的强依赖**: 随机初始化 vs Phi-4-MM 初始化差 0.32% WER [Table II], 且 TTS 中随机初始化完全失败 [Table VII]。这限制了方法对 LLM 选择的通用性。
3. **TTS 距实用差距大**: 连续 mel AR 的 TTS WER 11.03 vs latent diffusion 4.2, 高保真连续生成仍是开放难题 [Table VII]。
4. **仅英语实验**: 所有 ASR 和理解实验基于英语, 多语言泛化未验证。
5. **与 Gemma 4 12B 的对比不充分**: 文中提到 Gemma 4 12B 也去除了编码器, 但因无公开学术对比数据, 无法 apple-to-apple 比较 [§III-A, footnote 1]。
6. **内部数据不可复现**: production-scale 实验使用匿名内部数据, 关键的"数据缩放关闭差距"结论无法独立验证 [Table III]。

## 点评

**亮点**:
- **系统性消融设计出色**: token rate、layer-wise freezing、encoder-based vs encoder-free 的多维度消融提供了丰富的 insight, 尤其 layer-freezing 分析 (L24+ 冻结几乎无损) 对理解 LLM 内部分工有价值
- **acoustic-semantic trade-off 的发现有启发性**: 去掉编码器 → paralinguistic 任务大幅提升 → 编码器是信息瓶颈的论证链完整
- **实验规模大**: 64k hours 公开数据 + production-scale 验证, 不是 toy experiment

**不足**:
- **"encoder-free" 名称有误导性**: 仍保留 NeMoConv 下采样 (12.6M params) + MLP 投影 (12.6M params), 总计 25.2M。虽然远小于 Whisper 600M+, 但并非完全 "encoder-free", 更准确的说法是 "lightweight-encoder" 或 "encoder-block-free"
- **TTS 结论单薄**: TTS 部分仅一个表格, 无 speaker similarity 指标, 无消融, 作为 "proof-of-concept" 说服力有限
- **对比系统选择受限**: 主要对比自家 Phi-4-MM 系列, 缺少与 OWSM / Qwen-Audio / Whisper-AT 等其他 Speech-LLM 的对比
- **MMLU-speech 下降分析不够深入**: 虽指出 "semantic anchoring" 是瓶颈, 但未提出具体解决方案的实验, 仅在 conclusion 中列出未来方向

**定位**: 这是一篇实验驱动的 architectural exploration 论文, 核心贡献是提供了 encoder-based vs encoder-free 的首个系统性对比。结论温和但实用: encoder-free 可行且在 paralinguistic 任务上有优势, 但 semantic reasoning 是代价, 数据是关键。

## 可复用的 idea

1. **Layer-freezing 用于训练效率**: 冻结 LLM 上层 LoRA (L24+) 在 ASR 任务上几乎无损, 可用于资源受限场景的训练加速
2. **直接 mel 输入增强 paralinguistic 理解**: 绕过编码器的信息压缩可提升情感/说话人/性别等声学任务, 对需要保留低层声学特征的应用有参考价值
3. **多模态预训练作为 encoder-free 的 bootstrap**: 当缺少大规模语音数据时, 从多模态 checkpoint (如 Phi-4-MM) 初始化可以弥补编码器缺失的 inductive bias
4. **Token rate 与预训练帧率匹配**: 12.5Hz 优于 25Hz 的反直觉结果说明, encoder-free 系统应匹配 LLM 预训练时的 token rate 以最大化迁移效果

---

检索命中: [[SpeechLanguageModel]]✓, [[LLM-basedTTS]]✓, [[SemanticvsAcousticTokens]]✓ | 过滤: [[MelSpectrogram]](pending-review), [[ModalityAdaptationforSpeechLLM]](pending-review), [[MELLE]](pending-review) | 未命中但可能相关: 无

> [!review] agent-v3 2026-07-03 — **pass-with-fixes**
> 5 原则全部通过。方法节设计选择解释充分且来源标注清晰,实验数据与 PDF 交叉验证一致。已修复: frontmatter datasets 追加 LibriSpeech-PC。详见 `_review/Mel-LLM-review.yml`。
