---
type: paper
tier: deep
title: "video-SALMONN S: Memory-Enhanced Streaming Audio-Visual LLM"
arxiv_id: "2510.11129"
source: "Sources/video-SALMONN-S.pdf"
authors: [Guangzhi Sun, Yixuan Li, Xiaodong Wu, Yudong Yang, Wei Li, Zejun Ma, Chao Zhang]
year: 2026
venue: "ICML 2026"
tags: [audio-visual, video-understanding, streaming, test-time-training, long-term-memory, av-LLM, TTT, KV-cache-compression, episodic-memory, benchmark]
concepts: ["[[AudioUnderstanding]]", "[[ModalityAdaptationforSpeechLLM]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 0 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: video-SALMONN S 是 [[论文笔记/video-SALMONN|video-SALMONN]] (ICML 2024) 的 streaming 后继,从短视频 (~25s) 离线理解扩展到 3+ 小时流式理解。在 KB 的集成分类中仍属于 latent-representation-based Speech-LLM integration 路线 [Speech-LLMIntegrationTaxonomy],但骨架从 Vicuna 13B 升级为 Qwen3-VL 8B,音频编码从双头 (Whisper+BEATs) 简化为 Whisper-Large-v3 + window-level Q-Former [ModalityAdaptationforSpeechLLM]。SALMONN 系列 (清华/字节) 演进路线: SALMONN (纯音频, ICLR 2024) → video-SALMONN (短视频, ICML 2024) → SALMONN-omni (全双工对话, 2025) → video-SALMONN 2 (captioning-enhanced, 2025) → **video-SALMONN S** (streaming, ICML 2026)。
>
> **已有认知**: KB 中 AudioUnderstanding 页面记录了 SpeechLM 与 ALM 两条理解路线,SALMONN 系列属于 ALM 中的 "Two Heads" 架构 [AudioUnderstanding]。ModalityAdaptation 页面记录了 Q-Former 路线的优势与局限 [ModalityAdaptationforSpeechLLM]。video-SALMONN S 不再使用 MRC Q-Former (video-SALMONN 的核心创新),而是将 Q-Former 仅用于音频对齐;视觉编码依赖 Qwen3-VL 的原生 ViT + deepstack。video-SALMONN S 的核心创新 (TTT-based memory, KV-cache reading) 不在 KB 现有概念中。
>
> **创新判断**: video-SALMONN S 的核心贡献在于 (1) 首次将 test-time training (TTT) 引入流式视频 LLM 作为长期记忆机制; (2) TTTMEM layer 增加 long-span prediction objective 增强远程依赖; (3) modality-aware memory reading 的 KV-cache 压缩; (4) ELViM benchmark 评估情景记忆能力。这些均超出 KB 现有的流式对话/Q-Former 范畴,属于视频理解领域的记忆机制创新。
>
> 检索命中: [[AudioUnderstanding]][待确认], [[ModalityAdaptationforSpeechLLM]][待确认], [[Speech-LLMIntegrationTaxonomy]][待确认], [[Whisper]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个将 test-time training (TTT) 作为流式视频记忆机制的 av-LLM,通过 TTTMEM layer (含 long-span prediction) + 两阶段训练 + modality-aware memory reader,以固定 16k token 记忆预算处理 3+ 小时视频,在长视频 benchmark 上比 streaming/non-streaming baseline 高 3-7%,在自建 ELViM benchmark 上高 14.2%
> - **路线**: Video (1 FPS, 360p) → Qwen3-VL ViT (冻结) → TTTMEM layer (fast-weight MLP, 更新 per chunk) → similarity discarding (保持 N 个 memory token) → [可选] modality-aware memory reading (prompt-dependent KV-cache 压缩) → Qwen3-VL 8B LLM + LoRA → text response; Audio → Whisper-Large-v3 encoder → window-level Q-Former → 注入 LLM
> - **指标**: Video-MME long 71.3% (+1.9% vs streaming best); LVBench 55.6% (+4.0%); VideoEvalPro 58.9% (+7.1%); ELViM 46.7% (+14.2% vs non-streaming, +8.2% vs PEMF); StreamingBench 67.1% (competitive) [Table 1]; 推理 ~10s/样本, TTTMEM 仅增加 0.1s [Table 3]
> - **可借鉴**: (1) TTT 作为隐式参数记忆: fast-weight 更新将短期表征持续压缩为模型参数,固定预算下不丢信息; (2) long-span prediction: 在自重建 loss 基础上加"预测 T 步前 chunk"的正则项,增强远程信息保持; (3) two-stage training: stage 1 联合训练 TTT 参数,stage 2 冻结 TTT 仅训练 LLM + 更长序列; (4) modality-aware reading: 按 prompt 与 KV 的 attention score 选取相关 token,类似 human 从长期记忆中提取工作记忆
> - **局限**: 仅在 Qwen3-VL 8B 上验证; ELViM 基准数据创建依赖 Gemini-2.5-Pro + Qwen3-VL 辅助; audio token 仅约视觉的 1/75 且绕过 TTTMEM,音频长期记忆未建模; 代码开源但模型权重未见公开; similarity discarding 仍有不可逆信息损失

## 核心问题

**想解决什么**: 流式视频理解中的长期记忆问题。现有 streaming video LLM 要么通过 token merging/discarding 维持固定预算但累积信息损失 (MovieChat, PEMF),要么依赖外部检索但需要复杂设计 (Flash-VStream, Dispider)。当视频超过 1 小时,这些方法的性能显著低于 offline 模型 [§1]。

**为什么难**: streaming 约束要求固定内存预算 (LLM 上下文有限),但视频信息量随时间线性增长。token merging 本质上是有损压缩,信息在极长视频上不可避免地丢失; 检索方法需要知道"查什么",但在 streaming 场景下 query 时机未知 [§1]。

**怎么切入**: 将 test-time training (TTT) 引入作为一种参数化记忆。TTT 的 fast-weight MLP 在测试时逐 chunk 更新,将观察到的视频信息持续编码进模型参数,绕过了 token-level 有损压缩的根本限制。即使 token 被丢弃,fast-weight 中仍保留信息的压缩摘要 [§3.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

video-SALMONN S 由四个主要组件构成 [Fig 1]:

1. **Qwen3-VL ViT + deepstack (冻结)**: 视觉编码器,以 1 FPS 和 360p 分辨率处理视频帧,输出视频编码 Xt [§5.1]。

2. **TTTMEM layer (可训练)**: 核心创新。一个 MLP 模型,其 fast weight Wt 在测试时逐 chunk 更新。输入每 chunk (12-16 帧) 的视觉编码,通过梯度下降更新 fast weight 以最小化重建 loss + long-span prediction loss,然后用更新后的 MLP 输出新的 token 表征 Zt [§3.1, Fig 2]。音频 token 因数量极少 (约视觉的 1/75) 绕过 TTTMEM,直接进入后续阶段 [§3.1]。

3. **Similarity discarding**: TTTMEM 输出的新 token Zt 与前一步的 memory token Z_tilde_{t-1} 拼接后,丢弃与下一 token cosine similarity 最高的 K 个 token,保持固定 N 个 memory token [§3.1]。[论文原文] 选择 discarding 而非 merging 是为了更好地利用 TTTMEM 的长期表征能力,避免极长视频中 merging 导致的 over-smoothing [§3.1]。

4. **Qwen3-VL 8B LLM + LoRA (LoRA rank=128)**: 接收固定数量的 memory token 作为 KV-cache,生成文本回答 [§5.1]。

5. **音频编码**: Whisper-Large-v3 encoder + window-level Q-Former (窗口 0.5s),继承自 video-SALMONN 2 [§5.1]。

**与 video-SALMONN 的关键架构差异**: video-SALMONN 使用自定义 MRC Q-Former 融合三模态,输入限 ~25s; video-SALMONN S 完全基于 Qwen3-VL 的原生视觉编码,用 TTTMEM 替代 MRC Q-Former 的功能,支持 3+ 小时输入。

### 关键设计选择

**为什么用 TTT 而非现有记忆机制?** [论文原文] 现有方法通过 token merging 或 external memory 管理长期信息,但前者在极长序列上累积信息损失,后者需要复杂检索设计。TTT 提供了第三条路: 将信息编码到模型参数 (fast weight) 中,实现隐式的参数化记忆,无需显式存储或检索 [§1]。

[agent 解读] TTT 在这里的核心洞察是: token 是离散的、有容量限制的存储,而神经网络参数是连续的、可压缩的存储。一个 512x512 的 fast-weight 矩阵有 262K 个参数,理论上的信息容量远大于 16k 个 token 向量。且 TTT 的梯度更新是增量式的,新信息被 incremental 地编入权重,天然适配 streaming。

**TTTMEM 的 long-span prediction objective** [§3.1, Eq. 2-4]:

标准 TTT 仅有自重建 loss: f(theta_K * Xt; Wt-1) ≈ theta_V * Xt [Eq. 2]。TTTMEM 额外增加 long-span prediction loss: f(theta_K' * Xt; Wt-1) ≈ theta_V' * X_{t-T} [Eq. 3],要求 fast weight 在处理当前 chunk 时仍能预测 T 步之前的 chunk 信息。

[论文原文] 随着序列变长,旧信息不可避免地被新更新覆盖。Long-span prediction 作为时间一致性正则化器,鼓励 fast weight 保持对远距离输入的预测能力,从而改善长期信息保持 [§3.1]。

[论文原文] 实验显示 T=2 (约 2048 tokens 间隔) 是最优 trade-off: T 过大导致预测信号噪声太大,效果反而下降; T 在 1 个 minibatch 内 (T=1) 影响有限 [Table 10, Appendix G]。

**TTTMEM 的内部结构**: 2 层全连接 + GeLU 激活; 输入编码被分为 8 个 head (每个 512 维),每个 head 由一个独立 fast weight 处理,共 16 个 512x512 fast weight 矩阵 [§5.1]。使用 1 步 SGD 更新,可训练学习率 η [§5.1]。

**为什么 discarding 优于 merging?** [论文原文] 在极长视频中 merging 操作反复平均会导致 token 表征趋于相似 (over-smoothing),丧失区分性。Discarding 虽然直接丢失信息,但 fast-weight 中已保留了被丢弃 token 的摘要,TTTMEM 的 long-span 目标鼓励保留与远距离输入相关的信息 [§3.1]。

### 训练策略

**两阶段训练** [§3.2, Fig 3]:

| 阶段 | TTTMEM 参数 θ | LoRA | 输入帧数 | 目的 | 训练量 |
| --- | --- | --- | --- | --- | --- |
| Stage 1 (Cold-start) | 更新 | 更新 | ≤1024 (4 FPS) | 建立 fast-weight 更新机制 | 3 epochs, 48h on 32xH800 [§5.2] |
| Stage 2 (Scale-up) | 冻结 | 更新 | ≤2048 | 扩展到更长序列 + 更多 memory | 1 epoch, 16h on 32xH800 [§5.2] |

[论文原文] Stage 1 冻结 θ 参数会导致 TTT 功能紊乱; Stage 2 继续更新 θ 参数在长序列上显存溢出 (因为 θ 的梯度需要通过整个 fast-weight 更新链传播)。两阶段策略在 GPU 显存和序列长度之间取得了平衡 [§3.2]。

**训练数据** [§5.2]:
- 音频对齐: LibriSpeech 960h + CommonVoice + WavCaps + AudioCaps
- 视频理解: FineVideo + CinePile + ~100k LLaVA-Video-178k 采样
- 继承 video-SALMONN 2 的音频对齐器

### Modality-Aware Memory Reading

[§3.3, Eq. 6-10] 在 Transformer 的每一层,将 memory token 分为固定大小 chunk,逐 chunk 与 prompt token 做 attention,按 attention score 选取最相关的 KV pair:

1. 将 N 个 memory token 分为 ceil(N/m) 个 chunk
2. 每个 chunk 拼接 prompt token 后做 self-attention [Eq. 6]
3. 计算 prompt 到每个位置的平均 attention score [Eq. 7]
4. 对多模态 token 位置,选 top-m 个最高分的 KV pair [Eq. 8-10]
5. 非多模态 token (文本/时间戳) 的 KV pair 保留不筛选 [Eq. 9-10]

[论文原文] 这模拟了人脑从长期记忆中提取任务相关的工作记忆的过程 [§3.3]。

[agent 解读] 与 AdaReTaKe 等 training-free KV-cache 压缩不同,这里的 memory reading 是在一个已经过 TTTMEM 增强的 memory 上操作。Table 2 显示 reading 在 ELViM 上带来 +2.8% 的提升 (43.9% → 46.7%),说明即使 TTTMEM 已做了很好的信息压缩,prompt-guided 检索仍能进一步提取相关信息。

## 实验

### 主要结果 [Table 1]

所有系统基于相同 Qwen3-VL backbone 和训练数据,仅长期记忆机制不同。16k memory token budget。

| 指标 | 本文 (w/ reading) | 本文 (w/o reading) | Similarity Merging | PEMF | Qwen3-VL (non-streaming) | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Video-MME long | 71.3% | 71.3% | 69.4% | 68.6% | 60.1% | [Table 1] |
| LVBench | 55.6% | 55.4% | 51.6% | 49.5% | 47.4% | [Table 1] |
| VideoEvalPro | 58.9% | 55.8% | 51.8% | 50.4% | 54.9% | [Table 1] |
| ELViM | 46.7% | 43.9% | 38.5% | 38.2% | 28.1% | [Table 1] |
| StreamingBench (R/O/C) | 78.9/57.5/41.5 | 78.4/57.5/40.9 | 78.1/59.5/39.2 | 79.1/57.1/40.8 | 76.9/42.5/37.5 | [Table 1] |

### 消融实验 [Table 2]

| 配置 | Video-MME Long | LVBench | VideoEvalPro | ELViM | 出处 |
| --- | --- | --- | --- | --- | --- |
| Similarity Merging | 69.4% | 51.6% | 51.8% | 35.4% | [Table 2] |
| K-means Clustering | 67.8% | 49.8% | 50.7% | 33.8% | [Table 2] |
| Similarity Discarding | 69.2% | 51.1% | 51.5% | 36.8% | [Table 2] |
| + Mamba-2 | 70.3% | 53.5% | 54.8% | 39.6% | [Table 2] |
| + TTT-video (standard) | 70.0% | 54.3% | 54.9% | 42.3% | [Table 2] |
| + TTTMEM w/o Stage 2 | 71.3% | 55.1% | 55.5% | 43.6% | [Table 2] |
| + Stage 2 (v-SALMONN S) | 71.3% | 55.4% | 55.8% | 43.9% | [Table 2] |
| + AdaReTaKe reading | 70.8% | 55.1% | 56.5% | 43.6% | [Table 2] |
| + Proposed reading | 71.3% | 55.6% | 58.9% | 46.7% | [Table 2] |

**关键观察**:

1. **TTTMEM vs TTT-video**: long-span prediction 贡献约 1-2% 提升。在 LVBench 上 54.3% → 55.1% (+0.8%), 在 ELViM 上 42.3% → 43.6% (+1.3%) [Table 2]。Fig 8 进一步显示 long-span prediction 在 evidence 和 query 时间距离越远时优势越大。

2. **TTT vs Mamba-2**: TTT-video 在长视频理解上一致优于 Mamba-2 (另一种 RNN-type 记忆),尽管两者都是序列压缩机制 [Table 2]。

3. **Stage 2 training**: 在 VideoEvalPro (+0.3%) 和 ELViM (+0.3%) 上有modest改善,Video-MME long 上无变化 [Table 2]。

4. **Proposed reading vs AdaReTaKe**: 提出的 modality-aware reading 在 ELViM 上比 AdaReTaKe 高 3.1% (46.7% vs 43.6%),在 VideoEvalPro 上高 2.4% [Table 2]。

5. **Memory token 效率** [Fig 6]: TTTMEM 仅需 merging 的 ≤25% memory token 即可达到相同准确率。4k token TTTMEM ≈ 16k token merging。

6. **帧数扩展性** [Fig 7]: merging 在帧数 >2048 后性能下降,TTTMEM 维持稳定上升至 4096 帧后达到平台。

### ELViM Benchmark

1849 个问题,1021 个 target video,15 个类别。平均合并视频长 2920s (~49 min),最长 5518s (~92 min) [Table 4]。目标视频与问题间距至少 30 min。

ELViM 验证: Qwen3-VL 和 Gemini-2.5-Pro 在仅看到问题时间点前的部分视频时准确率接近随机猜测 (22-23%),但在给完整视频时接近 100% (98.7-99.3%),确认问题确实需要视频记忆 [Table 6]。

### 运行时开销 [Table 3]

| 模型 | GPU Mem (Avg/Peak) | 推理时间 | 出处 |
| --- | --- | --- | --- |
| Similarity Merging | 21.9/24.1 GB | 10.4s | [Table 3] |
| v-SALMONN S (w/o reading) | 22.2/24.3 GB | 10.5s | [Table 3] |
| v-SALMONN S (w/ reading) | 22.5/24.4 GB | 10.9s | [Table 3] |

TTTMEM 仅增加 0.1s 和 0.3 GB 开销; reading 增加 0.4s [Table 3]。TTTMEM 的 TFLOP 相对 Qwen3-VL 仅增加 0.000235 [Table 7]。

### 更大记忆预算 [Table 9, Appendix F]

128k memory token (单 H800 极限) 下 ELViM 达 54.9% (vs 16k 的 46.7%),LVBench 达 59.8% (vs 55.6%) [Table 9]。

## 局限性

1. **仅验证于 Qwen3-VL 8B**: 未在其他 backbone (Llama, Gemma) 上测试。deepstack embedding 的处理方式 (直接 discard) 是 Qwen3-VL 专属设计 [Appendix E]。

2. **音频长期记忆缺失**: 音频 token 绕过 TTTMEM (因数量少),意味着音频的长期记忆完全依赖 LLM 的 context window。对于 3 小时视频中的音频事件回忆,模型可能受限 [§3.1]。

3. **ELViM 数据创建依赖强模型**: 问题生成和过滤使用 Gemini-2.5-Pro 和 Qwen3-VL,可能引入这些模型的偏见; human collation 仅验证最终正确性 [§4.1]。

4. **similarity discarding 仍不可逆**: 虽然 fast weight 保留了被丢弃 token 的摘要,但这种保留是隐式的、有损的。论文未量化 fast weight 实际保留了多少被丢弃 token 的信息。

5. **训练成本高**: 总计 64 小时 on 32xH800 GPU,不含音频对齐器预训练 [§5.2]。

6. **仅文本输出**: 与 video-SALMONN 系列一贯局限相同,不支持语音或视频生成。

## 点评

**TTT 作为流式记忆的深层洞察**: video-SALMONN S 的核心贡献不是"又一个长视频理解模型",而是发现了 TTT 的 fast-weight 机制天然适配 streaming 场景的属性。token-level 记忆 (merging/discarding) 本质是显式存储,容量上限由 token 数决定; TTT fast-weight 是隐式参数化存储,信息密度更高。Fig 6 的实验 (TTTMEM 仅需 25% memory token) 直接验证了这一优势。

**long-span prediction 的精妙设计**: 标准 TTT 的自重建 loss 会让 fast weight 不断适应最近的输入而遗忘旧信息 (catastrophic forgetting 的变体)。Long-span prediction 本质上是一种 replay-free 的 continual learning 正则化: 不需要真正回放旧数据,只需要在更新时额外约束"还能预测 T 步前的输入",成本极低但效果显著 (Fig 8 的时间距离分析非常 convincing)。

**modality-aware reading 的启发**: 这个设计将 human cognitive science 中的"从长期记忆提取工作记忆"概念具体化为 KV-cache 压缩策略。与 AdaReTaKe 的 training-free 方法相比,优势在于: (1) 区分多模态和文本 token 的重要性 (文本/时间戳 KV 始终保留); (2) 经过 TTTMEM 增强的 memory 已经过信息压缩,reading 是在更高质量的 KV 上操作。

**与 video-SALMONN 的对比**: 两篇论文面对的根本问题不同。video-SALMONN 的挑战是"如何在多模态间平衡注意力" (→ MRC Q-Former + diversity loss + mixed training); video-SALMONN S 的挑战是"如何在有限预算下保持长期记忆" (→ TTTMEM + long-span + memory reading)。后者的问题更难,因为信息损失随时间累积; 前者的信息在短视频中始终可用,只是分配问题。

**ELViM benchmark 的价值**: 现有视频 benchmark 多测短期感知或片段推理,ELViM 首次测试"从过去经验中学习并迁移"的能力,这对 future AI agent 至关重要。非 streaming baseline 在 ELViM 上仅 28.1% (接近随机 25%),说明现有 offline 模型的帧采样策略根本无法触及 30 分钟前的信息。

**局限的诚实性**: 论文对 audio 绕过 TTTMEM 的处理比较坦率但未充分讨论。在 3 小时视频中,语音对话可能包含关键信息 (如人名、地点),而这些信息的长期记忆完全未建模。这是一个有意义的未来方向。

## 可复用的 idea

1. **TTT 作为流式记忆**: 在任何需要固定预算处理无限长序列的场景中,用 fast-weight MLP 将历史信息编码进参数,替代显式 token-level 存储。适用于: 流式语音对话的上下文维护、实时音频事件监测、长时间会议记录理解等 [§3.1]。

2. **Long-span prediction 正则化**: 在 TTT 或任何 continual learning 场景中,通过要求模型在更新时仍能预测 T 步前的输入,防止 catastrophic forgetting。成本仅为额外一个 loss 项,无需存储旧数据。T 的选择需要在信号质量和距离之间 trade-off [§3.1, Table 10]。

3. **Two-stage training for parameter-efficient long-context**: Stage 1 用中等长度序列联合训练所有参数 (建立机制); Stage 2 冻结机制参数,仅训练 LLM adapter + 扩展序列长度。这种策略可推广到任何受 GPU 显存限制的长序列训练场景 [§3.2]。

4. **Modality-aware KV-cache 压缩**: 按 prompt 与各位置的 attention score 筛选 KV pair,保留文本/时间戳 token 不筛选。在多模态 LLM 的推理效率优化中,这比 uniform 压缩更有效 [§3.3]。

5. **Similarity discarding > merging for long sequences**: 当有隐式记忆 (如 TTT fast weight) 补偿信息损失时,discarding 优于 merging,因为避免了 over-smoothing。但无隐式记忆时不一定成立 [§3.1]。

6. **ELViM 式 benchmark 设计**: 通过"先展示信息 → 插入无关内容 → 考查回忆"的范式测试长期记忆。过滤掉模型可通过先验知识/短期推理回答的问题,确保测试的是真正的记忆能力 [§4.1]。

## 审阅

> [!review] 审阅: pass (0 high, 0 medium, 0 low)
> 审阅日期: 2026-06-08 | checklist v1.1 | self-review (同 session)
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | TTTMEM/long-span/two-stage/discarding 的 WHY 充分 |
> | 可信赖 | pass | ~95% 数字 claim 有 Table/Fig 出处 |
> | 可区分 | pass | ~85% 因果解释标注 [论文原文]/[agent 解读] |
> | 可定位 | pass | SALMONN 系列 5 篇演进路线 + 4 KB 实体对比 |
> | 不污染 | pass | 用户指令禁止反向更新,未执行 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/video-SALMONN-S-review.yml`
