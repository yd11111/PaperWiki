---
type: paper
tier: deep
title: "Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens"
arxiv_id: "2503.01710"
source: "Sources/Spark-TTS.pdf"
authors: [Xinsheng Wang, Mingqi Jiang, Ziyang Ma, Ziyu Zhang, Songxiang Liu, Linqin Li, Zheng Liang, Qixi Zheng, Rui Wang, Xiaoqin Feng, Weizhen Bian, Zhen Ye, Sitong Cheng, Ruibin Yuan, Zhixian Zhao, Xinfa Zhu, Jiahao Pan, Liumeng Xue, Pengcheng Zhu, Yunlin Chen, Zhifei Li, Xie Chen, Lei Xie, Yike Guo, Wei Xue]
year: 2025
venue: "arXiv"
tags: [TTS, LLM-based-TTS, speech-codec, single-codebook, zero-shot, controllable-TTS, chain-of-thought, speech-factorization, open-source]
concepts: ["[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Semantic vs Acoustic Tokens]]", "[[Finite Scalar Quantization]]", "[[Single-codebook vs Multi-codebook]]", "[[Speech Factorization]]", "[[Codec Language Model]]", "[[Speaker Embedding]]"]
models: ["[[CosyVoice]]", "[[CosyVoice 2]]", "[[EnCodec]]"]
tasks: ["[[Zero-shot Speech Synthesis]]", "[[Instructed Speech Generation]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[Emilia]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[Speech Tokenizer]]✓, [[Semantic vs Acoustic Tokens]]✓, [[LLM-based TTS]]✓, [[Speech Factorization]]✓, [[Zero-shot Speech Synthesis]]✓, [[Single-codebook vs Multi-codebook]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: LLM-based TTS 领域有两大主流路线: (1) 多阶段方案 -- AR LM 生成 semantic tokens + flow matching/diffusion 渲染声学细节 (CosyVoice 系列, Seed-TTS, FireRedTTS); (2) 多码本 RVQ 方案 -- AR+NAR 两阶段预测多层 codec tokens (VALL-E 系列)。两者均面临架构复杂度高的问题。近期出现了第三条路线: 单码本 codec + 单 Transformer (Llasa),但 Llasa 仅支持 zero-shot cloning,不支持属性控制。Spark-TTS 属于这第三条路线的扩展 -- 单流 codec (BiCodec) + 单 LM (Qwen2.5),但增加了 chain-of-thought 属性控制能力。
>
> **已有认知**: Speech Tokenizer 概念库确认了三类 tokenizer (自监督/监督/声学); semantic vs acoustic tokens 的核心 trade-off 是语义连贯性 vs 声学保真度; FSQ 在 CosyVoice 系列中验证了比 VQ 更稳定(100% codebook 利用率); 单码本方案序列短利于 LM 建模,但重建质量通常低于 RVQ。Speech Factorization 确认了 speaker-content 解耦的多种技术路线。
>
> **创新判断**: Spark-TTS 的创新不在单一技术上,而在整合方式 -- BiCodec 将 semantic tokens (VQ, 50 TPS) 和 global tokens (FSQ, 固定 32 个) 解耦为两种互补 token 类型,使得单个 LM 即可同时处理内容和属性控制,无需 flow matching 等第二阶段。对比 Llasa (FSQ 单码本 65K) 和 CosyVoice (semantic + CFM 两阶段),Spark-TTS 找到了一个中间路线。
>
> 检索命中: [[Speech Tokenizer]], [[Semantic vs Acoustic Tokens]], [[LLM-based TTS]], [[Speech Factorization]], [[Zero-shot Speech Synthesis]], [[Single-codebook vs Multi-codebook]] | 过滤: [[Finite Scalar Quantization]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: BiCodec 将语音解耦为 semantic tokens (语义,50 TPS VQ) + global tokens (音色,固定 32 个 FSQ),配合 Qwen2.5-0.5B 实现单 LM 的 zero-shot TTS 和 CoT 属性控制。
> - **路线**: Text + (Reference Audio 或 Attribute Labels) → Qwen2.5-0.5B (fine-tuned) → [Fine-grained Attributes →] Global Tokens → Semantic Tokens → BiCodec Decoder → Waveform
> - **指标**: Seed-TTS-eval: CER 1.20% / SIM 0.672 (zh), WER 1.98% / SIM 0.584 (en); BiCodec 重建: STOI 0.92, UTMOS 4.18, SIM 0.80 @ 650 bps; TTS 音质: UTMOS 4.35 (LibriSpeech test-clean) 超越 CosyVoice2 (4.23) 和 GT (4.08) [Table 4, 5]
> - **可借鉴**: (1) Global tokens 用 learnable queries + cross-attention + FSQ 从 Mel 提取固定长度说话人表征,避免了传统 speaker embedding 的信息瓶颈和 RVQ 的多码本复杂度; (2) Chain-of-thought 属性预测 -- coarse label 先预测 fine-grained value 再预测 global/semantic tokens,将可控性嵌入 LM 的自然生成流程
> - **局限**: Speaker similarity 显著弱于多阶段方案 (SIM 0.672 vs Seed-TTS 0.796 / CosyVoice2 0.748 [Table 4]); semantic 和 global tokens 之间缺乏显式解耦约束; 仅在英中双语评估

## 核心问题

Spark-TTS 要解决的核心问题是: **如何在保持与文本 LLM 架构完全统一的前提下,实现 zero-shot voice cloning 和细粒度属性控制?**

现有方案的痛点:
1. **多阶段复杂度**: CosyVoice 等需要 AR LM + flow matching 两阶段,偏离文本 LLM 范式 [§1]
2. **多码本复杂度**: VALL-E 等需要 RVQ 多层预测策略 (AR+NAR 或 delay pattern),增加架构复杂度 [§1]
3. **声音创建缺失**: 现有系统大多只能 reference-based cloning,无法按属性规格生成全新声音 [§1]
4. **评估标准缺失**: 大量系统使用私有数据集,难以公平比较 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Spark-TTS 由两个核心组件构成 [§3, §4, Fig 2-3]:

1. **BiCodec (语音 tokenizer)**: 将语音分解为两类互补 token
   - **Semantic tokens**: 50 TPS,单层 VQ (codebook 8192),编码语言内容 [§3.1]
   - **Global tokens**: 固定 32 个,FSQ (codebook 4096),编码说话人属性和全局声学特征 [§3.1]
   - **Decoder**: 从两类 token 重建波形 [§3.1]

2. **Speech Language Model**: Qwen2.5-0.5B fine-tuned,decoder-only transformer [§4.1]
   - Zero-shot 模式: Text + Global tokens (from reference) → Semantic tokens [§4.1, Fig 3]
   - Voice creation 模式: Text + Attribute labels → Fine-grained values → Global tokens → Semantic tokens (CoT) [§4.1, Fig 3]

### 关键设计选择

**为什么用 wav2vec 2.0 特征而非原始音频作为 semantic encoder 输入?**
[论文原文] 作者选择 wav2vec 2.0 (XLSR-53) 的第 11/14/16 层特征平均值作为 semantic tokenizer 的输入,因为这些层与语义和音素信息有强相关性 (Pasad et al., 2023 验证了不同层特征与 words/phonemes 的对齐程度) [§3.2]。[agent 解读] 这个选择使 BiCodec 的 semantic tokens 从源头就富含语义信息,避免了从原始波形学习语义的困难,但也引入了对 wav2vec 2.0 预训练质量的依赖。

**为什么 global tokens 用 learnable queries + cross-attention 而非直接 pooling?**
[论文原文] 全局信息通过 ECAPA-TDNN 编码 Mel 频谱后,用一组 learnable queries 通过 cross-attention 提取固定长度序列 [§3.2, Eq.1]。[agent 解读] 直接 pooling 会将所有全局信息压缩为单一向量,表达力有限; cross-attention + learnable queries 允许模型学习从不同角度 "查询" 全局特征,生成 32 个各有侧重的 token。这比 TiCodec 的 Group VQ (gvq) 方法效果好得多 -- BiCodec gvq-32 SIM 仅 0.74,而 FSQ-32 达 0.80 [Table 2]。

**为什么 global tokens 用 FSQ 而非 VQ?**
[论文原文] 对于需要表示时间无关全局信息的 token 集合,FSQ 比 VQ 更能避免训练 collapse 风险 [§3.2]。[agent 解读] 全局信息的分布相比帧级语义信息更加稀疏和多样 (speaker 空间高维),VQ 的 codebook 更容易在这种场景下退化; FSQ 天然 100% 利用率的特性 (概念库已确认) 在此尤为关键。

**Chain-of-thought 属性控制的工作机制**
[论文原文] 在 voice creation 模式下,系统先接收 coarse-grained 属性标签 (gender / pitch level / speed level),然后按顺序预测: fine-grained 属性值 (具体 pitch 数值和 speed 数值) → global tokens → semantic tokens [§4.1, Fig 3]。训练时两种模式 (zero-shot 和 control) 混合训练 [§4.3, Eq.2-3]。[agent 解读] CoT 设计的价值在于: 从 coarse label 到 fine value 的映射让模型学会了属性标签到具体声学参数的内隐知识,而 fine value 先于 global tokens 生成又约束了后续 token 的属性一致性。这种层级依赖关系自然适合自回归建模。

### 训练策略

**BiCodec 训练** [§3.3]:
- 端到端 GAN 训练: 多尺度 Mel 重建 loss (L1) + Multi-period discriminator + Multi-band multi-scale STFT discriminator + VQ codebook loss + commitment loss [§3.3]
- wav2vec 2.0 重建 loss: 量化后的 token 通过 ConvNeXt 预测器恢复原始 wav2vec 2.0 特征,确保语义保持 [§3.3]
- 训练稳定性技巧: 初始阶段 global embedding 直接从 gf pooling 获取而非走 FSQ,使用 teacher-student L1 loss 引导 FSQ codebook;稳定后切换为完整 FSQ pipeline [§3.3]
- 训练数据: LibriSpeech 960h + Emilia-CN/EN 1000h 各,共 ~3000h,16 kHz [§6.1]
- 收敛: ~800K steps,batch 约 614.4 秒语音 [§6.1]

**LM 训练** [§6.1]:
- Backbone: Qwen2.5-0.5B-Instruct
- 数据: VoxBox 全量训练集 (~102.5K hours)
- Optimizer: AdamW (beta1=0.9, beta2=0.96)
- Epochs: 3,batch size 768

## 实验

| 指标 | 本文 (Spark-TTS) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER (zh) | 1.20% | Seed-TTS 1.12%, CosyVoice2 1.45% | Seed-TTS-eval test-zh | [Table 4] |
| SIM (zh) | 0.672 | Seed-TTS 0.796, CosyVoice2 0.748 | Seed-TTS-eval test-zh | [Table 4] |
| WER (en) | 1.98% | F5-TTS 1.83%, CosyVoice2 2.57% | Seed-TTS-eval test-en | [Table 4] |
| SIM (en) | 0.584 | Seed-TTS 0.762, F5-TTS 0.647 | Seed-TTS-eval test-en | [Table 4] |
| UTMOS (TTS) | 4.35 | CosyVoice2 4.23, GT 4.08 | LibriSpeech test-clean | [Table 5] |
| BiCodec STOI | 0.92 | X-Codec2 0.92, StableCodec 0.91 | LibriSpeech test-clean | [Table 1] |
| BiCodec PESQ-NB | 3.13 | X-Codec2 3.04, StableCodec 2.91 | LibriSpeech test-clean | [Table 1] |
| BiCodec UTMOS | 4.18 | StableCodec 4.23, X-Codec2 4.13 | LibriSpeech test-clean | [Table 1] |
| BiCodec SIM | 0.80 | X-Codec2 0.82, BigCodec 0.84 | LibriSpeech test-clean | [Table 1] |
| Gender Acc | 99.77% | Parler-TTS 98.12%, VoxInstruct 82.99% | PromptTTS test | [Table 3] |
| Llasa-1B CER (zh) | 1.20 vs 1.89 | Llasa-1B 1.89, Llasa-8B 1.59 | Seed-TTS-eval | [Table 4] |
| Llasa-1B SIM (zh) | 0.672 vs 0.669 | Llasa-1B 0.669, Llasa-8B 0.684 | Seed-TTS-eval | [Table 4] |

**关键实验发现**:

1. **智能性 (CER/WER) 极强**: Spark-TTS 中文 CER 1.20% 仅次于闭源 Seed-TTS (1.12%),英文 WER 1.98% 仅次于 F5-TTS (1.83%) [Table 4]。[论文原文] 这归因于 BiCodec 的 semantic feature-based 设计和 VoxBox 数据集的高质量文本标注 [§6.4]。

2. **音质超越 GT**: UTMOS 4.35 不仅超过 CosyVoice2 (4.23),甚至超过 Ground Truth (4.08) [Table 5]。[agent 解读] UTMOS 超 GT 在 codec-based TTS 中并非罕见 (训练数据丰富的模型倾向于生成 "更干净" 的语音),但这确实说明生成质量很高。

3. **Speaker similarity 是短板**: SIM 0.672 (zh) / 0.584 (en) 显著低于多阶段方案 (Seed-TTS 0.796/0.762) 和同期 CosyVoice2 (0.748/0.652) [Table 4]。[论文原文] 作者归因于 AR 模型在推理时引入更大的说话人变异性,以及 global 和 semantic tokens 之间缺乏显式解耦约束 [Limitation]。

4. **参数效率极高**: 0.5B 参数 + 100K h 数据 vs Llasa-8B + 250K h,CER/WER 更低 (1.20 vs 1.59 zh, 1.98 vs 2.97 en),SIM 相当 (0.672 vs 0.684 zh) [Table 4]。[agent 解读] BiCodec 的双 token 设计 (semantic + global) 比 Llasa 的 FSQ 单码本 (65K) 提供了更高效的信息编码。

5. **Global token 设计验证**: FSQ + learnable queries (32 tokens) 的 SIM 0.80,显著优于 GVQ (0.74) 和更短序列 (8 tokens: 0.74, 16 tokens: 0.77) [Table 2]。[agent 解读] 这证明了全局信息需要足够的表达容量,且 FSQ 比 GVQ 在此场景下优势明显。

6. **属性控制精准**: 性别准确率 99.77%,远超 VoxInstruct (82.99%) 和 Parler-TTS (98.12%) [Table 3]。[Fig 4-5] 显示 pitch 和 speed 的 coarse-grained 和 fine-grained 控制均有较高精度。

## 局限性

1. **Speaker similarity 弱于多阶段方案**: 单阶段 AR 生成的随机性导致 SIM 低于 Seed-TTS、CosyVoice2 等多阶段或 NAR 方法 [Limitation]
2. **缺乏显式 disentanglement 约束**: Global tokens 和 semantic tokens 之间没有对抗训练或信息瓶颈等显式解耦机制,timbre 信息可能在 semantic tokens 中泄漏 [Limitation]
3. **评估局限**: 仅中英双语,未涵盖 zero-shot 跨语言场景和低资源语言 [agent 解读]
4. **BiCodec 训练数据规模小**: 仅用 ~3000h 训练 codec,远少于 X-Codec2 的 150K h [Table 9],在域外数据上可能泛化受限

## 点评

Spark-TTS 在 LLM-based TTS 的设计空间中找到了一个有价值的折中点。与知识库中已有的方案对比:

**vs CosyVoice 系列 (两阶段)**: Spark-TTS 去掉了 flow matching 第二阶段,BiCodec decoder 直接从 token 重建波形,架构更简洁。代价是 SIM 显著下降 (0.672 vs CosyVoice2 0.748),说明 CFM 在恢复说话人细节上仍不可替代。但 UTMOS 4.35 > CosyVoice2 4.23 表明单阶段在音质上可以超越两阶段。

**vs Llasa (单码本 + 大 LM)**: Spark-TTS 用更小的模型 (0.5B vs 8B) 达到更好的内容一致性,核心优势来自 BiCodec 的双 token 设计。semantic tokens 保证语义,global tokens 保证音色,比 Llasa 的 FSQ 单码本 (所有信息压入同一序列) 更高效。但 Spark-TTS 还增加了属性控制维度,这是 Llasa 没有的。

**BiCodec 的设计值得关注**: 它本质上是 semantic tokens + speaker embedding 的结合,但通过 learnable queries + FSQ 将 speaker embedding 扩展为一个 32 token 的序列,比传统单向量 embedding 更有表达力。这比 TiCodec (GVQ) 效果好很多。

**CoT 属性控制是差异化亮点**: 在文本 LLM 范式内实现粗细粒度控制 (pitch/speed/gender),无需额外模块,是当前 open-source TTS 中少有的能力。

**主要限制**: SIM 短板是结构性的 -- 没有 CFM 或 diffusion 来精细渲染声学细节,AR 生成的随机性不可避免地损害说话人一致性。作者在 Limitation 中也指出了这一点,并提出了 formant/pitch perturbation 的未来方向。

## 可复用的 idea

1. **Learnable queries + cross-attention 提取固定长度 global tokens**: 这种从 ECAPA-TDNN speaker embedding 提取可变角度全局信息的方法,比直接 pooling 更灵活,且与 LM 的 token-based 输入兼容。可迁移到任何需要将连续全局特征注入 LM 的场景。

2. **CoT 属性预测 pipeline**: coarse label → fine-grained value → global tokens → semantic tokens 的层级 CoT 生成,让属性控制自然融入自回归解码流程,无需额外的条件注入机制。这个模式可扩展到情感、方言等更多属性。

3. **FSQ + learnable queries 用于全局量化**: FSQ 天然避免 codebook collapse,配合 learnable queries 可以编码多维全局信息。对比 GVQ 的显著优势 (SIM 0.80 vs 0.74 [Table 2]) 表明这是一个值得采纳的量化策略。

4. **BiCodec 训练稳定性技巧**: 初始阶段用 gf pooling 直接获取 global embedding (跳过 FSQ),配合 teacher-student L1 loss 引导 FSQ codebook 学习,稳定后再切换为完整 FSQ pipeline [§3.3]。这种渐进式训练策略可迁移到其他需要同时训练量化层和 decoder 的场景。

5. **混合 zero-shot + control 训练**: 每个音频样本同时构造 zero-shot (reference audio prompt) 和 control (attribute label prompt) 两个训练样本 [§4.3],让同一个 LM 同时学会两种推理模式。

> [!review] 审阅 (2026-06-03, agent)
> **结论**: pass (3 low issues, 0 high/medium)
> - [low] frontmatter models 未包含 Llasa (最直接对比方案)
> - [low] BiCodec 训练各 loss 权重原文未给出,已标注
> - [low] Llasa 对比行格式略不一致
> 详见 `_review/Spark-TTS-review.yml`
