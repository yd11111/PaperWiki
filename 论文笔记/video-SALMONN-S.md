---
type: paper
tier: deep
title: "video-SALMONN S: Memory-Enhanced Streaming Audio-Visual LLM"
arxiv_id: "2510.11129"
source: "Sources/video-SALMONN-S.pdf"
authors: [Guangzhi Sun, Yixuan Li, Xiaodong Wu, Yudong Yang, Wei Li, Zejun Ma, Chao Zhang]
year: 2026
venue: "ICML 2026 (submission)"
tags: [streaming, video-understanding, test-time-training, TTT, long-term-memory, audio-visual, av-LLM, KV-cache, memory-reading, benchmark, LoRA, Qwen3-VL]
concepts: ["[[AudioUnderstanding]]", "[[ModalityAdaptationforSpeechLLM]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 0 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。基于未确认概念页,仅供参考。
>
> **谱系定位**: video-SALMONN S 是 SALMONN 系列的第四代流式版本,从 [[论文笔记/SALMONN|SALMONN]] (音频 LLM, ICLR 2024) → [[论文笔记/video-SALMONN|video-SALMONN]] (av-LLM, ICML 2024) → [[论文笔记/video-SALMONN2|video-SALMONN 2]] (caption-enhanced, 2025) → video-SALMONN S (streaming, ICML 2026)。核心方向从架构创新 (MRC Q-Former) → 训练优化 (MrDPO) → 长时记忆 (TTTMEM) 的演进。本文切换 backbone 至 Qwen3-VL 8B,音频编码仍用 Whisper-Large-v3 + Q-Former aligner [Whisper, ModalityAdaptationforSpeechLLM],但核心创新从编码器/适配器转向 LLM 前的 recurrent memory 层。
>
> **已有认知**: KB 中 AudioUnderstanding 页面将 SALMONN 列为 "Two Heads" LALM 代表 [AudioUnderstanding]。ModalityAdaptationforSpeechLLM 页面记录了 Q-Former adapter 路线 (BLIP-2 → 语音),video-SALMONN S 继续沿用 Q-Former 做音频对齐但不再作为核心贡献。本文的核心创新 -- test-time training (TTT) 作为流式记忆机制 -- 属于视频理解而非语音/TTS 领域,KB 中暂无相关概念页。
>
> **创新判断**: 与 SALMONN 系列前三代相比,video-SALMONN S 的焦点从"理解什么"转向"记住多久"。TTT (Sun et al., 2024; Dalal et al., 2025) 是一种 RNN-like 机制,通过在测试时更新 MLP 的 fast weights 将序列历史编码到模型参数中。video-SALMONN S 的独特贡献是 TTTMEM -- 在标准 TTT 基础上加入 long-span prediction loss 以增强远程依赖建模。这一机制与 KB 中已讨论的 streaming 技术 (因果注意力、chunk-wise processing) 有概念联系,但属于完全不同的参数化记忆路线。
>
> 检索命中: [[AudioUnderstanding]][待确认], [[ModalityAdaptationforSpeechLLM]][待确认], [[Whisper]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个将 test-time training (TTT) 用作流式视频记忆机制的 av-LLM,通过 TTTMEM 层 (含 long-span prediction loss) + 两阶段训练 + 模态感知记忆读取,在 16k token 预算下处理 3+ 小时视频,长视频 benchmark 上超越流式/非流式基线 3-7%,ELViM 上超越 14.2%
> - **路线**: Video (1 FPS) → Qwen3-VL ViT encoder → Video encodings Xt + Audio → Whisper-Large-v3 → Q-Former aligner → TTTMEM 层 (fast weight 更新 + long-span prediction) → Similarity discarding (固定 N token) → Qwen3-VL 8B LLM + LoRA (rank 128) → [可选] Modality-aware memory reading → text response
> - **指标**: Video-MME long 71.3% (vs Qwen3-VL 60.1%, +11.2%) [Table 1]; LVBench 55.6% (vs 47.4%, +8.2%) [Table 1]; ELViM 46.7% (vs non-streaming 28.1%, +18.6%; vs PEMF 38.2%, +8.5%) [Table 1]; 仅用 25% memory tokens 达到 merging 同等精度 [Fig 6]; TTTMEM 仅增加 0.000235 TFLOPs 开销 [Table 7]
> - **可借鉴**: (1) TTT 作为参数化长期记忆: 将序列历史编码到 MLP fast weights 中,比 token merging/discarding 更高效地保留远程信息; (2) Long-span prediction loss: 用 t-T 步的重建目标作为时间一致性正则化,鼓励 fast weights 保留远程依赖; (3) 两阶段训练: Stage 1 冷启动 TTT 参数 + LLM LoRA,Stage 2 冻结 TTT 投影参数只保留 fast weight 更新规则 + 扩展上下文,解耦参数学习与容量扩展; (4) Similarity discarding 优于 merging: 避免 over-smoothing,让 TTTMEM 的 fast-weight 状态保留有用摘要
> - **局限**: 仅在 8B backbone 上验证; ELViM 依赖 Gemini-2.5-Pro 生成问题; 音频 tokens bypass TTTMEM (无视觉-音频联合记忆); 仅支持理解不支持生成; Stage 1 训练需 32xH800 48h

## 核心问题

**想解决什么**: 未来 AI agent 需要处理任意长度的连续视频流并从中学习。现有流式视频 LLM (streaming video LLM) 采用 token merging 或 KV-cache 压缩维持固定内存预算,但在长视频 (3+ 小时) 上因累积信息丢失而性能退化,甚至不如离线模型 [§1]。核心需求是: 在固定内存预算下,如何有效地将短期多模态表征转化为可保持数小时的长期记忆?

**为什么难**: (1) Token merging/discarding 的信息损失是不可逆的 -- 被合并或丢弃的 token 包含的信息永久丢失,且随视频长度累积 [§1]; (2) KV-cache 压缩 (如 ReTaKe, AdaReTaKe) 仅在推理时选择重要 KV-pair,不产生新的信息表征 [§2.1]; (3) 外部记忆方法 (如 Flash-VStream) 需要复杂的检索机制,限制了可扩展性 [§2.2]; (4) 训练 RNN-type 长期记忆层 (如 Mamba) 时,GPU 显存消耗随序列长度线性增长,直接训练长序列代价过高 [§3.2]。

**怎么切入**: 用 test-time training (TTT) 作为隐式参数化记忆 -- 不同于 token-level 的显式记忆 (merging/discarding/retrieval),TTT 通过在测试时对 MLP fast weights 做梯度更新,将序列历史编码到模型参数中 [§1]。这种参数化记忆的容量不受 token 数量限制,且信息保留是渐进衰减而非突然丢失。为增强长程保持,引入 long-span prediction loss 作为时间一致性正则化 [§3.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

video-SALMONN S 由四个阶段构成 [Fig 1]:

1. **视觉编码**: 视频以固定帧率 (默认 1 FPS) 输入 Qwen3-VL 的视觉编码器,输出 video encodings Xt [§3]。

2. **音频编码**: Whisper-Large-v3 编码音频,window-level Q-Former (窗口 0.5s) 对齐到文本空间。音频 tokens 数量仅为视觉 tokens 的 1/75,因此 bypass TTTMEM 直接进入 token discarding 阶段 [§3.1]。

3. **TTTMEM 层**: 核心创新。对视觉 encodings 进行 recurrent 处理,通过 fast weight 更新将历史信息编码到 MLP 参数中,输出 incoming memory tokens Zt [§3.1, Fig 2]。

4. **记忆管理 + LLM**: incoming tokens Zt 与前一步的 memory tokens Z̃t-1 合并后,通过 cosine similarity discarding 保持固定 N 个 memory tokens。这些 tokens 送入 Qwen3-VL 8B LLM (LoRA rank=128)。可选地,在 LLM 内部启用 modality-aware memory reading 进一步压缩 KV-cache [§3.3]。

**文本 tokens 的处理**: Qwen3-VL 的 deepstack embeddings 中,文本和时间戳 tokens 与多模态 tokens 交错。TTTMEM 仅处理多模态 tokens,处理后将文本 tokens 加回并保持原始相对位置 [§3, based on Qwen3-VL design]。

### 关键设计选择

**为什么用 TTT 而非 Mamba 或其他 RNN?** [论文原文] Table 2 的 ablation 直接对比了三种 RNN-type 记忆机制: Mamba-2、TTT-video (标准 TTT)、TTTMEM。结果显示 TTT-video 在长视频理解上一致优于 Mamba-2 (LVBench 54.3% vs 53.5%),TTTMEM 进一步提升 1-2% [Table 2]。[agent 解读] TTT 的优势可能源于其自监督重建目标 -- Mamba 通过 selective state space 更新记忆,而 TTT 通过显式的重建 loss 驱动 fast weight 更新,后者对信息保留的优化更直接。

**为什么选择 discarding 而非 merging?** [论文原文] Table 2 显示 similarity discarding 与 similarity merging 性能相当 (LVBench 51.1% vs 51.6%),但 discarding 被选择因为它能"更好地利用 TTTMEM 层的长程上下文表示能力,避免 extremely long videos 中的 over-smoothing" [§3]。[agent 解读] merging 将两个 token 平均融合为一个,多次 merging 后 token 趋于均值 (over-smoothing); discarding 保持 token 的原始表征,让 TTTMEM 的 fast-weight 状态承担信息保留责任。这是显式记忆 (token) 与隐式记忆 (参数) 的分工。

**为什么 deepstack embeddings 不需要 TTTMEM?** [论文原文] Appendix E (Table 8) 显示,对 Qwen3-VL 的 4 层 deepstack embeddings 全部应用 TTTMEM 不仅不提升性能 (LVBench 54.7% vs 55.4%),还将额外显存和运行时开销乘以至少 4 倍 [Table 8]。因此仅在 Transformer 输入处应用 TTTMEM,deepstack tokens 直接按 TTTMEM 输出的索引 discarding [§5.1]。

### TTTMEM 层的机制

TTTMEM 的核心是一个带 fast weights W 的 MLP,工作方式类似 RNN [Fig 2]:

**输入分块**: 整个输入序列被分为固定大小的 chunks X1, X2, ..., Xt,每个 chunk 包含约 12-16 帧。chunk 按时间顺序依次送入 TTTMEM [§3.1]。

**两个训练目标** [§3.1, Eq. 2-4]:
1. **重建 loss** (reconstruction): l_recon(t) = ||f(θK * Xt; Wt-1) - θV * Xt||^2。用当前输入的 K 投影重建 V 投影,确保 fast weights 能编码当前信息。
2. **Long-span prediction loss**: l_long-span(t) = ||f(θ'K * Xt; Wt-1) - θ'V * Xt-T||^2。用当前输入预测 T 步之前的输入,迫使 fast weights 保留远程信息。

**Fast weight 更新** [Eq. 4-5]:
```
Wt = Wt-1 - η∇l(Xt, Xt-T; Wt-1)   (SGD with one step)
Zt = f(θQ * Xt; Wt)                 (output tokens using updated weights)
```

[论文原文] long-span prediction loss 的作用是"时间一致性正则化" -- 当序列变长时,旧信息不可避免地被新更新覆盖。long-span prediction 鼓励模型保持对 t-T 时刻输入的预测性表征,从而改善长序列上的信息保留 [§3.1]。

**架构参数** [§5.1]:
- 2 层全连接层 + GeLU 激活
- 8 heads x 512 维 = 16 个 512x512 fast weight 矩阵
- T=2 (默认 long-span prediction 距离)
- SGD 单步更新
- 总计 0.000235 TFLOPs 开销 (vs 基模型 3.15 TFLOPs) [Table 7]

**Long-span prediction 的效果** [Fig 8]: 在 LVBench 上按 query 与 evidence 的时间距离分组,long-span prediction 在所有距离上一致优于无 long-span 的 TTT,且优势随距离增大而增大 -- 在 >5000s 距离段 (>83 min) 提升最显著。

**Prediction span T 的选择** [Table 10]: T 从 1 到 32 的实验显示,T=2 (对应 2048 tokens 间隔) 是最佳 trade-off。更长的 span 产生更嘈杂的预测目标,效果递减 [Appendix G]。

### 训练策略

**两阶段训练** [§3.2, Fig 3]:

| 阶段 | 更新参数 | 帧率 | 最大帧数 | 训练量 | 资源 |
| --- | --- | --- | --- | --- | --- |
| Stage 1 (冷启动) | TTTMEM θ + LoRA | 4 FPS | 1024 | 3 epochs, lr=2e-5 | 48h, 32xH800 |
| Stage 2 (扩容) | LoRA only (TTTMEM θ 冻结, fast weight 更新保留) | 1 FPS | 2048 | 1 epoch | 16h, 32xH800 |

[论文原文] 两阶段设计解决 GPU 显存瓶颈: Stage 1 用较短序列学习 TTTMEM 的 θ 投影矩阵 (静态参数); Stage 2 冻结 θ 但保留 fast weight 更新规则,用更长序列和更多 memory tokens 优化 LLM 从记忆中提取信息的能力 [§3.2]。

[agent 解读] 这种解耦策略的精巧之处: Stage 1 学习 "怎么编码" (θ 投影),Stage 2 学习 "怎么使用" (LLM 的 LoRA)。一旦编码规则 (θ) 确立,它自然泛化到更长序列 -- fast weight 更新规则是不变的,只是应用次数增加。

**预训练数据** [§5.2]:
- 音频对齐器: LibriSpeech 960h, CommonVoice, WavCaps, AudioCaps
- AV 微调: FineVideo, CinePile, ~100k from LLaVA-Video-178k
- 基于 video-SALMONN 2 (Tang et al., 2025a) 的训练流程

### 模态感知记忆读取

**动机**: 记忆 tokens Z̃ 已包含完整视频历史,但回答特定问题时无需全部记忆,全部使用既浪费计算又影响性能 [§3.3]。

**机制** [§3.3, Eq. 6-10]: 在每个 Transformer 层内执行迭代 KV-cache 压缩:
1. 将 N 个 memory tokens 分为大小 m 的 chunks,共 ⌈N/m⌉ 个
2. 对每个 chunk: 将 chunk tokens 与 prompt tokens 拼接,执行标准 self-attention
3. 计算 prompt 对每个位置的平均注意力分数 (跨所有 head) 作为重要性 [Eq. 7]
4. 仅保留重要性最高的 m 个多模态 token KV-pairs; 非多模态 token (文本、时间戳) 始终保留 [Eq. 8-10]
5. 经过 ⌈N/m⌉ 轮迭代后,剩余 KV-pairs 包含高度浓缩的任务相关信息

[论文原文] 这模拟了人脑从长期记忆中提取任务相关信息到工作记忆的过程 [§3.3]。

**效果**: Table 2 显示 modality-aware reading 在所有长视频 benchmark 上一致优于 AdaReTaKe (training-free KV-cache 压缩方法)。特别是 ELViM 46.7% vs AdaReTaKe 43.6% (+3.1%) [Table 2]。启用 reading 后可将 memory tokens 翻倍 (32k) 而保持同等 GPU 显存 (~24 GB) [§5.1]。

## 实验

### 基线设定

所有系统基于相同 Qwen3-VL 8B backbone 和相同训练数据,仅记忆机制不同 [Table 1]:
- 非流式: Qwen3-VL 8B (均匀采样), video-SALMONN 2+
- 流式: Similarity Merging (MovieChat), PEMF (StreamForest)
- 读取: AdaReTaKe

### 主要结果 (16k memory tokens)

| 指标 | v-SALMONN S (w/ reading) | v-SALMONN S (w/o reading) | Qwen3-VL 8B | Similarity Merging | PEMF | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Video-MME long | 71.3 | 71.3 | 60.1 | 69.4 | 68.6 | [Table 1] |
| LVBench | 55.6 | 55.4 | 47.4 | 51.6 | 49.5 | [Table 1] |
| VideoEvalPro | 58.9 | 55.8 | 54.9 | 51.8 | 50.4 | [Table 1] |
| ELViM | 46.7 | 43.9 | 28.1 | 38.5 | 38.2 | [Table 1] |
| StreamingBench (avg) | 67.1 | 66.8 | 61.8 | 66.8 | 67.1 | [Table 1] |

**关键观察**:

1. **长视频全面领先**: vs 非流式 Qwen3-VL +11.2% Video-MME long, +8.2% LVBench, +18.6% ELViM; vs 流式基线 PEMF +2.7% VME long, +6.1% LVBench, +8.5% ELViM [Table 1]。

2. **ELViM 优势最显著**: 非流式基线仅 28.1% (接近随机猜测),因为稀疏帧采样可能完全错过目标视频 [Table 1]。PEMF/merging ~38% 说明虽然实时感知好但长期记忆差 -- 即使正确感知当前进度也无法回忆 30 分钟前的目标视频 [§6.1]。

3. **StreamingBench 上竞争力**: StreamingBench 不含需要长期记忆的长视频,video-SALMONN S 仍保持竞争性 (67.1 vs PEMF 67.1),说明 TTTMEM 不损害短期感知 [Table 1]。

### 消融结果 (16k, w/o reading)

| 配置 | VME Long | LVBench | VideoEvalPro | ELViM | 出处 |
| --- | --- | --- | --- | --- | --- |
| Similarity Merging | 69.4 | 51.6 | 51.8 | 35.4 | [Table 2] |
| K-means Clustering | 67.8 | 49.8 | 50.7 | 33.8 | [Table 2] |
| Similarity Discarding | 69.2 | 51.1 | 51.5 | 36.8 | [Table 2] |
| + Mamba-2 | 70.3 | 53.5 | 54.8 | 39.6 | [Table 2] |
| + TTT-video | 70.0 | 54.3 | 54.9 | 42.3 | [Table 2] |
| + TTTMEM w/o Stage 2 | 71.3 | 55.1 | 55.5 | 43.6 | [Table 2] |
| + Stage 2 (v-SALMONN S) | 71.3 | 55.4 | 55.8 | 43.9 | [Table 2] |
| v-SALMONN S + AdaReTaKe | 70.8 | 55.1 | 56.5 | 43.6 | [Table 2] |
| v-SALMONN S + Reading | 71.3 | 55.6 | 58.9 | 46.7 | [Table 2] |

**消融拆解**:
1. **Discarding ≈ Merging > Clustering**: token 下采样方式中 clustering 最差 [Table 2]。
2. **TTTMEM > TTT-video > Mamba-2**: long-span prediction 贡献 +1-2% over TTT-video [Table 2]。
3. **Stage 2 贡献**: +0.2-0.4% (适度但一致) [Table 2]。
4. **Memory reading >> AdaReTaKe**: 尤其在 ELViM (+3.1%) 和 VideoEvalPro (+2.4%) [Table 2]。

### 记忆效率

**Memory token 用量** [Fig 6]: TTTMEM 在 4k tokens 下达到 merging 在 16k tokens 下的精度,即仅需 ≤25% 的 memory tokens。在 2k tokens 下 TTTMEM 仍保持合理精度,而 merging 显著退化。

**帧数扩展** [Fig 7]: 16k memory budget 下,merging 在帧数从 2048 增长到 4096+ 时性能显著下降; TTTMEM 保持稳定上升趋势直到 4096 帧后进入平台期。平台期可通过增加 memory tokens (32k) 推迟到 10k+ 帧 [§6.2]。

**128k memory** [Table 9]: ELViM 54.9% (w/ reading, vs 16k 的 46.7%),LVBench 59.8% (vs 55.6%)。30 分钟以内的视频 (1800 帧 at 1FPS) 在 128k budget 下无需任何 token 压缩 [§6.1]。

### 运行时开销

| 模型 | GPU 显存 (avg/peak) | 推理时间 | 出处 |
| --- | --- | --- | --- |
| Similarity Merging | 21.9/24.1 GB | 10.4s | [Table 3] |
| PEMF | 21.9/24.1 GB | 10.4s | [Table 3] |
| v-SALMONN S (w/o reading) | 22.2/24.3 GB | 10.5s | [Table 3] |
| v-SALMONN S (w/ reading) | 22.5/24.4 GB | 10.9s | [Table 3] |

TTTMEM 仅增加 0.1s 推理开销; reading 增加 0.4s (iterative KV-cache prefill); 总体显存开销基本相同 [Table 3]。

## 局限性

1. **仅 8B backbone**: 所有实验基于 Qwen3-VL 8B,未验证 TTTMEM 在更大模型 (如 72B) 上的效果。video-SALMONN 2 已在 72B 上验证 caption 能力,video-SALMONN S 可能在更大模型上有更大收益 (或可能因 LoRA rank 相对于模型规模的比例变化而不同)。

2. **音频不经过 TTTMEM**: 音频 tokens 直接 bypass TTTMEM (因为数量仅为视觉 tokens 的 1/75) [§3.1]。这意味着音频信息的长期记忆完全依赖 token discarding,没有参数化记忆保护。对于语音密集的长视频 (如教学讲座),音频信息可能在长时间后丢失。

3. **ELViM 的评估局限**: ELViM 问题由 Gemini-2.5-Pro 生成,虽然有人工校对,但问题的多样性和难度可能受限于 LLM 的生成偏好。此外,目标视频与查询间隔固定为 ≥30 分钟,未测试更极端的时间距离 (如 6-12 小时)。

4. **仅支持理解不支持生成**: 与 SALMONN 系列一贯特点一致,video-SALMONN S 只输出文本,不支持语音或视频生成。

5. **训练成本**: Stage 1 需要 32xH800 GPU 训练 48 小时,Stage 2 需 16 小时。虽然比 LLM 预训练低很多,但对学术实验室仍是显著门槛。

6. **Qwen3-VL 依赖**: 模型深度集成 Qwen3-VL 的 deepstack embedding 设计,TTTMEM 的 deepstack 处理策略 (直接 discard) 是针对 Qwen3-VL 的特定设计 [Appendix E],迁移到其他 LLM backbone 可能需要重新设计。

## 点评

**从 token 记忆到参数记忆的范式转换**: video-SALMONN S 的核心洞察是将流式视频理解的记忆问题从"保留哪些 tokens"转变为"参数编码了什么信息"。传统的 merging/discarding 是在 token 层面做 lossy compression,一旦信息被丢弃就不可恢复。TTTMEM 则通过 fast weight 更新将信息编码到 MLP 参数中,即使对应的 token 被 discarded,其信息仍以参数形式保留。Fig 6 的 25% memory tokens 实验最直观地说明了这一点 -- 参数记忆有效地补偿了 token 记忆的缺失。

**Long-span prediction 的简洁与有效**: 与复杂的注意力机制或外部检索相比,long-span prediction loss 极其简洁 -- 仅是在标准 TTT 的重建 loss 上额外添加一个 t-T 步的重建目标。Fig 8 的时间距离分析漂亮地验证了其效果: 正好在需要长程记忆的场景 (>3000s distance) 上提升最大。T=2 对应约 2048 tokens 间隔,恰好跨越一个 TTT minibatch 边界,这可能是其有效性的关键 -- 它强迫 fast weights 在 minibatch 更新时保留上一个 minibatch 的信息。

**SALMONN 系列的演化轨迹**: SALMONN → video-SALMONN → video-SALMONN 2 → video-SALMONN S 展示了一条清晰的研究路线: 理解范围 (音频→视频) → 理解深度 (QA→caption) → 理解时长 (秒级→小时级)。有趣的是,每一代的核心创新都在不同的系统层次: Q-Former 架构 → DPO 训练 → TTT 记忆。这说明团队在系统地攻克 av-LLM 的不同瓶颈,而非在单一方向上迭代。

**与 speech/TTS 领域的联系**: 虽然本文聚焦视频理解,但 TTT-as-memory 的思路对流式语音处理也有启示。流式 TTS/ASR 同样面临长程依赖问题 (如长篇演讲中的一致性、语调延续),现有方法多依赖 causal attention + positional encoding,参数化记忆可能是一个互补方向。特别是 TTTMEM 的计算开销极低 (0.000235 TFLOPs, 0.1s latency),使其在实时系统中可行。

**ELViM 的价值与局限**: ELViM 填补了流式视频 LLM 评估的重要空白 -- 现有 benchmark (Video-MME, StreamingBench) 主要测试短期理解,ELViM 首次要求模型从 30+ 分钟前的视频中回忆程序性知识。然而,ELViM 的构建依赖 Gemini-2.5-Pro 生成问题 + Qwen3-VL 8B 过滤,这种 "用模型评模型" 的方法可能在问题分布上有偏。Table 6 的验证 (完整视频下 Qwen3-VL 98.7% 正确) 虽然缓解了担忧,但理想的 benchmark 应有更多人工标注。

## 可复用的 idea

1. **TTT 作为流式系统的长期记忆**: 任何需要固定内存处理无限长序列的场景都可考虑 TTTMEM。关键设计: (a) 将序列分为固定大小 chunks; (b) 每个 chunk 通过 SGD 单步更新 MLP fast weights; (c) 用重建 loss + long-span prediction loss 联合优化。适用条件: 序列长度远超上下文窗口; 需要保留远程信息; 可接受 O(chunk_size) 的额外计算。可推广到流式 ASR (长篇演讲的说话人一致性)、流式对话 (多轮对话的上下文保持) 等 [§3.1]。

2. **两阶段训练的解耦策略**: 当训练一个包含 "学习编码规则" 和 "学习使用编码" 两个目标的系统时,先用短序列学编码规则 (Stage 1),再冻结规则、用长序列学使用方式 (Stage 2)。这种解耦避免了长序列训练的 GPU 显存瓶颈。可推广到任何 encoder-decoder 系统的长序列适配 [§3.2]。

3. **Long-span prediction 作为时间一致性正则化**: 当 fast weight 更新存在 "旧信息被覆盖" 的风险时,额外添加一个预测 t-T 步输入的 loss 迫使模型保留远程信息。T 的选择应约等于一个 minibatch 的间隔。这个技巧不限于 TTT,可推广到任何 recurrent 系统的长程保持 [§3.1, Table 10]。

4. **Similarity discarding 优于 merging 的条件**: 当系统有参数化记忆 (如 TTTMEM) 时,token discarding 优于 merging -- merging 的 over-smoothing 反而干扰参数记忆的信息保留。但当无参数化记忆时,merging ≈ discarding [Table 2]。这暗示了一个一般原则: 显式记忆 (token) 和隐式记忆 (参数) 应协同设计,而非独立优化 [§3, §6.2]。

5. **Modality-aware prompt-dependent KV-cache 压缩**: 在多模态 LLM 中,不同模态的 KV-pairs 重要性不同。压缩时保留所有非多模态 (文本/时间戳) KV-pairs,仅压缩多模态 KV-pairs。这比一视同仁地压缩所有 KV-pairs 更安全,因为文本 tokens 通常是关键的语义锚点 [§3.3, Eq. 8-10]。

## 审阅

> [!review] 审阅: pass (0 high, 1 medium, 1 low)
> 审阅日期: 2026-06-08 | checklist v1.1
> 详见 `_review/video-SALMONN-S-review.yml`
>
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节含因果解释; 关键设计选择有 why; 速查可借鉴字段具体 |
> | 可信赖 | pass | 数字 claim 均有 [Table/Fig/§] 标注; 指标名正确 |
> | 可区分 | pass | 因果解释标注了 [论文原文] vs [agent 解读]; 无断言式推断 |
> | 可定位 | pass | KB 背景有谱系定位 + 系列对比; 但 KB 重叠有限 (视频理解 vs TTS KB) |
> | 不污染 | pass | 无反向更新 (per instructions); concepts/models 挂接合理 |
>
> Issues: 2 (high: 0, medium: 1, low: 1)

---

检索命中: [[AudioUnderstanding]][待确认], [[ModalityAdaptationforSpeechLLM]][待确认], [[Whisper]][待确认] | 过滤: 无 | 未命中但可能相关: 无
