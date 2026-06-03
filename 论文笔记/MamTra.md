---
type: paper
tier: deep
title: "MamTra: A Hybrid Mamba-Transformer Backbone for Speech Synthesis"
arxiv_id: "2603.12342"
source: "Sources/MamTra.pdf"
authors: [Tan Dat Nguyen, Sangmin Bae, Joon Son Chung, Ji-Hoon Kim]
year: 2026
venue: "Interspeech 2026 (submitted)"
tags: [TTS, hybrid-architecture, Mamba, SSM, Transformer, knowledge-distillation, efficiency, LLM-based-TTS]
concepts: ["[[LLM-based TTS]]", "[[Non-autoregressive TTS]]", "[[Speech Language Model]]", "[[Attention-based TTS]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个实体页: [[LLM-based TTS]]✓, [[模型库/CosyVoice 2|CosyVoice 2]]✓, [[Codec Language Model]], [[Speech Language Model]]✓, [[SEED-TTS-Eval]]✓)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: MamTra 处于 LLM-based TTS 的效率优化分支。当前 LLM-based TTS 主流路线(VALL-E, CosyVoice 系列, Llasa)均依赖自回归 Transformer backbone,带来 O(L^2) 复杂度和大 KV cache 问题。已有两类优化方向: (1) 推理优化(GQA, KV cache compression, speculative decoding),本质不改变 attention 的二次复杂度; (2) 架构替代,用 Mamba 等线性模型替换 attention,但纯 Mamba 在 TTS 中尚未被系统研究。MamTra 属于第 (2) 类,且是首个系统研究 Mamba-Transformer 混合架构在 TTS 中的工作。
>
> **已有认知**: KB 中 [[LLM-based TTS]] 已记录 Hybrid 趋势(LLM + Flow/Diffusion),但尚未涉及 Mamba 替代 attention 层的效率优化方向。CosyVoice 2 作为 baseline 在 KB 中有详细记录(WER 2.57% test-en, CER 1.45% test-zh on SEED-TTS-Eval),且已知其使用 Qwen2.5 backbone。Zonos-v0.1 是唯一已有的混合 TTS 系统但细节未公开。
>
> **创新判断**: MamTra 的核心新颖性在于: (1) 首次系统地探索 Mamba 在 TTS backbone 中的位置和比例; (2) 提出从预训练 Transformer 到 Mamba 的结构化权重转移(Q→C, K→B, V→x 映射); (3) 仅用 2% 训练数据通过蒸馏恢复性能。这填补了 KB 中"LLM-based TTS 效率优化"维度的空白。
>
> 检索命中: [[LLM-based TTS]](confirmed), [[模型库/CosyVoice 2|CosyVoice 2]](confirmed), [[Speech Language Model]](confirmed), [[SEED-TTS-Eval]](confirmed) | 过滤: [[Codec Language Model]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将预训练 Transformer TTS backbone 中的部分 attention 层替换为 Mamba 层,通过结构化权重转移 + 多级蒸馏,仅用 2% 训练数据实现 34% 推理显存降低且几乎不损失语音质量
> - **路线**: 预训练 CosyVoice 2 Transformer → 选择替换策略(BlockBeg/Importance) → Attention Q,K,V 权重映射到 Mamba C,B,x → 多级蒸馏(CE + logits KL + embedding MSE) → 混合 Mamba-Transformer backbone
> - **指标**: MamTra 1:1 在 SEED-TTS-eval test-en 上 WER 2.28%(teacher CosyVoice 2: 2.03%),NMOS 3.66 vs 3.68,UTMOS 4.16 vs 4.15,SSIM 0.72 vs 0.66; VRAM 降低 34% [Table 3, Fig 3]
> - **可借鉴**: Attention→SSM 的结构化权重映射(Q→C, K→B, V→x),可用于任何需要将 Transformer 层转为线性复杂度的场景; 分层替换策略分析(哪些层更适合 Mamba)可指导混合架构设计
> - **局限**: 基于 CosyVoice 2 单一 backbone,泛化到其他 TTS 系统未验证; 1:11 比例下质量明显下降; 仅在英文 LibriTTS(0.5kh)上训练/评估,中文和多语言未涉及; 代码未在论文发表时开源

## 核心问题

1. **LLM-based TTS 的效率瓶颈能否通过混合架构解决?** Autoregressive Transformer 的 O(L^2) 复杂度在长文本合成(podcast/audiobook/对话)中成为部署障碍,纯 Mamba 替换又牺牲全局建模能力 -- 混合架构能否同时保持质量和效率?

2. **如何避免混合架构从零训练的高昂成本?** 已有尝试(Zonos-v0.1)需要从头预训练,既昂贵又缺乏可复现性。能否从现有预训练 Transformer 出发,以极低成本转化为混合模型?

3. **Mamba 层应该放在哪里、放多少?** 混合架构的设计空间巨大(位置 x 比例 x 选择策略),TTS 场景下的最优配置是什么?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MamTra 以 CosyVoice 2 (基于 Qwen2.5, 0.5B, 24 层) 为 teacher Transformer,将其中部分 Transformer 层替换为 Mamba-2 层,形成 interleaved hybrid 架构 [§2, Fig 1]。

核心 pipeline:
```
CosyVoice 2 (24层 Transformer, 预训练)
    ↓ 选择替换策略
部分层 Transformer → Mamba-2 (权重从 Attention 映射初始化)
    ↓ 多级蒸馏 (CE + Logits KL + Embedding MSE)
MamTra hybrid backbone (混合 Transformer + Mamba 层)
```

### 关键设计选择

#### 1. Attention 到 Mamba 的结构化权重映射 [§2.3, Fig 2]

论文建立了 self-attention 和 Mamba SSM 之间的数学等价关系。通过去除 softmax 将 attention 线性化:

- 标准 attention: $y_t = \text{softmax}(QK^\top/\sqrt{d_k}) \cdot V$ [Eq. 1]
- 线性化 attention: $h_t = m_{t-1,t} h_{t-1} + K_t V_t$, $y_t = Q_t^\top h_t / \sqrt{d_k}$ [Eq. 4]
- Mamba SSM: $h_t = A_t h_{t-1} + B_t x_t$, $y_t = C_t h_t$ [Eq. 2]

对比 Eq. 2 和 Eq. 4 得到直接映射 [论文原文]:
- **Q (query) → C (output projection)**: query 在 attention 中提取信息的角色等价于 SSM 中的输出投影
- **K (key) → B (input-dependent parameter)**: key 在 KV product 中的角色等价于 SSM 的输入依赖参数
- **V (value) → x (projected input)**: value 等价于 SSM 接收的投影输入

这使得预训练 Transformer 的 Q/K/V 投影权重可直接复用来初始化 Mamba 层,避免从零训练 [§2.3]。

**为什么这个映射 work**: [agent 解读] 线性化 attention 去掉 softmax 后,attention 退化为一种线性 RNN,其状态更新形式与 Mamba 的选择性 SSM 结构一致。关键在于两者都通过隐状态 h_t 的递归更新来捕获序列历史,只是 Mamba 额外引入了选择性(输入依赖的 A/B/C),使得这种初始化为后续蒸馏提供了一个比随机初始化更好的起点。

#### 2. 混合策略的系统性探索 [§2.2, Table 1, Fig 4]

论文探索了 4 类替换策略:
- **Interleaved**: BlockBeg(每 block 开头放 Transformer)/ BlockEnd
- **Contiguous**: Front / Middle / Back / Sandwich(Transformer 集中在头/中/尾/两端)
- **Data-driven**: Importance-based(根据 cosine similarity 或 WER 标准移除不重要层)

在 Transformer:Mamba 比例从 1:1 到 1:11 的范围内评估。

**核心发现** [论文原文]: BlockBeg 策略在各比例下一致地获得最低 CE loss 和 WER [Fig 4]。但随着 Mamba 比例增加,WER-based importance 选择的优势越来越明显(类似结构化剪枝的观察) [§4.2]。

最终选择: 1:1 和 1:3 用 BlockBeg; 1:5 和 1:11 用 WER importance strategy [§4.2]。

**为什么 BlockBeg 在低比例下最优**: [agent 解读] BlockBeg 保证每个 block 的第一层是 Transformer,确保周期性地刷新全局上下文。这比将 Transformer 集中在某一区域(Front/Back)更好,因为全局信息需要在网络各处都能被获取,而非仅在入口或出口处。

#### 3. 多级蒸馏 [§2.4, Eq. 5, Table 4]

仅靠权重映射不够,因为去除 softmax 从根本上改变了 attention 动态 [论文原文]。蒸馏目标:

$$\mathcal{L} = \mathcal{L}_{CE} + \mathcal{L}_{logits} + \mathcal{L}_{emb}$$

- **L_CE (cross-entropy)**: ground-truth 输出监督,提供语言学准确性 [§2.4]
- **L_logits (skew KL divergence)**: 对齐 teacher/student 的 logits 分布,恢复生成行为 [§2.4]
- **L_emb (embedding MSE)**: 约束 token embedding,对齐语义-声学空间 [§2.4]

Ablation [Table 4] 表明: 去除 L_CE 导致最严重的 WER 退化(3.48→6.70%),L_logits 次之(→6.13%),L_emb 影响相对较小(→5.63%)。三个损失主要影响语言准确性而非感知质量(UTMOS/SSIM 几乎不变) [§4.4]。

### 训练策略

- 基于 CosyVoice 2 backbone (Qwen2.5, 0.5B, 24层) [§3]
- 训练数据: LibriTTS (0.5kh),仅为 teacher 训练数据量 (170kh) 的约 **0.3%** [Table 3]
- 优化器: Adam, lr=1e-5, dynamic batch size 40k tokens [§3]
- 探索实验: 15 epochs; 最终对比实验: 50 epochs [§3]
- 权重初始化: 复用预训练 Transformer 权重(含映射到 Mamba 的权重),显著快于 Xavier/Kaiming 随机初始化 [Fig 6]

## 实验

| 指标 | MamTra 1:1 | CosyVoice 2 (Teacher) | Zonos-v0.1 | Llasa-1B | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER (%) ↓ | 2.28 | 2.03 | 3.42 | 3.54 | SEED-TTS-eval test-en | [Table 3] |
| NMOS ↑ | 3.66±0.16 | 3.68±0.16 | 3.18±0.18 | 3.64±0.19 | SEED-TTS-eval test-en | [Table 3] |
| UTMOS ↑ | 4.16 | 4.15 | 3.63 | 4.13 | SEED-TTS-eval test-en | [Table 3] |
| SSIM ↑ | 0.72 | 0.66 | 0.67 | 0.46 | SEED-TTS-eval test-en | [Table 3] |
| TFLOPs ↓ | 1.64 | 1.78 | 7.32 | 4.53 | per-token@2048 ctx | [Table 3] |
| VRAM reduction | 34% vs CosyVoice 2, 17% vs Zonos | -- | -- | -- | SEED-TTS-eval avg | [Fig 3] |
| WER (%) LibriTTS ↓ | 2.26 | 2.04 | 3.37 | 3.07 | LibriTTS test-clean | [Table 3] |

**效率-质量 trade-off 曲线** [Table 3]:

| 配置 | Ratio | Strategy | TFLOPs | WER (test-en) | NMOS |
| --- | --- | --- | --- | --- | --- |
| MamTra 1:1 | 12T:12M | BlockBeg | 1.64 | 2.28 | 3.66 |
| MamTra 1:3 | 6T:18M | BlockBeg | 1.57 | 3.26 | 3.65 |
| MamTra 1:5 | 4T:20M | WER | 1.54 | 2.53 | 3.65 |
| MamTra 1:11 | 2T:22M | WER | 1.52 | 3.99 | 3.27 |

**长度压力测试** [§4.3]: 在 LibriTTS-based 长文本测试(1-62 words)中,MamTra 1:5 (WER strategy) 展现出比更保守的 1:3 (BlockBeg) 更好的鲁棒性(WER 2.28% vs 2.99%),说明在长上下文条件下层的选择比保留更多 Transformer 层更重要 [§4.3]。

**Cache 增长分析** [Fig 5, Table 2]: Mamba state 几乎恒定,KV cache 增长仅与剩余 Transformer 层数相关。MamTra 实现 sub-quadratic 计算和 sub-linear cache 增长,理论上在 2048 context length 下每 token 减少 1.4e10 FLOPs [§4.1]。

**收敛速度** [Fig 6]: 复用预训练权重初始化的 MamTra 从训练一开始就显著低于 Xavier/Kaiming 初始化的 loss,且最终 loss 更低,验证了 "initialize-and-train" 策略的有效性 [§4.4]。

## 局限性

1. **单一 backbone 依赖**: 所有实验仅基于 CosyVoice 2 (Qwen2.5 0.5B),未验证在其他 TTS backbone(如 Llama-based Llasa, 更大模型)上是否同样有效 [agent 解读]
2. **仅英文评估**: 训练数据为 LibriTTS (英文),未涉及中文或多语言场景,而 CosyVoice 2 本身是中英双语系统 [agent 解读]
3. **激进比例下质量下降**: 1:11 ratio 的 WER (3.99%) 已劣于 Zonos-v0.1 baseline (3.42%),说明替换比例存在明确上限 [Table 3]
4. **蒸馏依赖 teacher 质量**: 方法假设有高质量 pre-trained teacher,不适用于需要从零构建 TTS 系统的场景 [agent 解读]
5. **SSIM 不升不降**: 所有 MamTra 配置的 SSIM (0.72) 略高于 teacher (0.66),但论文未解释这一现象 [Table 3] [agent 解读: 可能与蒸馏过程中 embedding alignment 有关]
6. **论文较短(Interspeech 格式)**: 限于 4 页正文,许多设计细节(如 Mamba-2 的具体超参、各层替换的精确 attention head 数量)未充分展开 [agent 解读]

## 点评

**优势**:
- **实用导向**: 解决了 LLM-based TTS 部署中的实际痛点(VRAM, 长上下文 latency),34% 内存降低对边缘设备部署有直接价值
- **成本极低**: 仅需 0.5kh LibriTTS(约为 teacher 的 0.3%)+ 15-50 epochs 训练,极大降低了混合架构的准入门槛
- **系统性分析**: 首次在 TTS 领域系统比较了 7 种 Mamba-Transformer 混合策略 x 4 种比例,提供了清晰的设计指南
- **数学基础扎实**: Attention→SSM 的权重映射有清晰的线性代数推导,不是 ad-hoc 的工程 trick

**不足**:
- **评估范围窄**: 仅英文、仅 0.5B 模型,作为"首次系统研究"的 claim 需要更广泛的验证
- **与 Zonos 对比不公平**: Zonos 是 1.6B/46层/200kh 的模型,MamTra 是 0.5B/24层/0.5kh 微调,模型规模和数据量差异巨大,Table 3 的直接对比需审慎解读
- **缺少端到端延迟**: 论文仅报告 VRAM 和 FLOPs,未报告实际推理速度(tokens/sec)或首 token 延迟

**在知识库中的定位**: 这篇论文填补了 [[LLM-based TTS]] 在**架构效率优化**维度的空白。之前 KB 中的效率优化主要是推理层面(speculative decoding, KV cache compression),MamTra 首次提供了**架构层面**的替代方案。其 Attention→Mamba 权重映射技术也为其他语音任务(ASR, 对话)中的模型压缩提供了可复用思路。

## 可复用的 idea

1. **Attention→SSM 结构化权重映射** (Q→C, K→B, V→x): 可应用于任何需要将预训练 Transformer 转为线性复杂度架构的场景。特别是对于已有大型预训练 TTS 模型的团队,可以极低成本获得效率提升。

2. **分层替换策略分析方法论**: BlockBeg / Contiguous / Importance 的系统比较框架可用于其他混合架构设计(如 Transformer + Linear Attention, Transformer + RWKV)。

3. **多级蒸馏组合 (CE + logits KL + embedding MSE)**: 特别是 skew KL divergence 用于 logits 对齐,在其他模型压缩/架构转换任务中可复用。

4. **"先映射初始化,后少量数据蒸馏"范式**: 将大模型迁移到新架构的通用策略,成本比从零预训练低 1-2 个数量级。

5. **Length-stress 测试的发现**: 在长上下文下,"选对层"比"保留更多 Transformer 层"更重要,这个 insight 对边缘部署中的模型裁剪有指导意义。

---

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass
> **摘要**: 因果解释充分(3 处 WHY 均有来源标注),出处标注覆盖 ≥90%,KB 定位精准。已修正 1 处方向性措辞。1 个 low issue (frontmatter 未列所有 baseline 模型)。反向更新风险低(仅追加 key_papers)。
> 详见 `_review/MamTra-review.yml`
