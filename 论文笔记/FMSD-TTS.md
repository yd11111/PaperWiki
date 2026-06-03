---
type: paper
tier: deep
title: "FMSD-TTS: Few-shot Multi-Speaker Multi-Dialect Text-to-Speech Synthesis for Ü-Tsang, Amdo and Kham Speech Dataset Generation"
arxiv_id: "2505.14351"
source: "https://arxiv.org/abs/2505.14351"
authors: [Yutong Liu, Ziyue Zhang, Ban Ma-bao, Yuqing Cai, Yongbin Yu, Renzeng Duojie, Xiangxiang Wang, Fan Gao, Cheng Huang, Nyima Tashi]
year: 2025
venue: "arXiv"
tags: [TTS, multi-speaker, multi-dialect, few-shot, low-resource, Tibetan, speaker-identity, non-autoregressive, flow-matching, dynamic-routing]
concepts: ["[[Speaker Embedding]]", "[[Conditional Flow Matching]]", "[[Duration Predictor]]", "[[Non-autoregressive TTS]]", "[[Speaker Adaptation]]"]
models: ["[[VITS]]", "[[BigVGAN]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页: [[Speaker Embedding]]✓, [[Conditional Flow Matching]]✓, [[Speaker Adaptation]], [[Voice Cloning Taxonomy]], [[Duration Predictor]], [[Non-autoregressive TTS]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Speaker Embedding]]✓, [[Conditional Flow Matching]]✓ | 过滤: [[Speaker Adaptation]](pending-review), [[Voice Cloning Taxonomy]](pending-review), [[Duration Predictor]](pending-review), [[Non-autoregressive TTS]](pending-review) | 未命中但可能相关: 无

**谱系定位**: FMSD-TTS 属于 semi-end-to-end NAR TTS 系统,基于 Matcha-TTS (CFM-based) 架构,使用显式 duration predictor + flow matching 生成 mel spectrogram + BigVGAN vocoder。在 KB 的 Voice Cloning Taxonomy 四分类中,FMSD-TTS 属于 **Few-shot Voice Cloning** 类别 — 使用 ECAPA-TDNN 提取 speaker embedding + 少量参考音频进行说话人克隆,但其核心创新不在 voice cloning 本身,而在 **方言控制** 维度。

**已有认知**:
- Speaker Embedding 页记录了 ECAPA-TDNN 是当前最常用的 speaker encoder 架构 [confirmed],FMSD-TTS 直接沿用;注入方式采用 Addition (加到 hidden states),与 KB 记录的常见方式一致
- Conditional Flow Matching 页记录了 Matcha-TTS 作为 CFM 在 TTS 中的代表工作 [confirmed],FMSD-TTS 以其为 backbone
- Duration Predictor 页 [待确认] 记录了 FastSpeech 2 式的显式 duration 预测范式,FMSD-TTS 继承该范式
- Non-autoregressive TTS 页 [待确认] 记录了 NAR TTS 用 Matcha-TTS/F5-TTS 等 flow-based 方法实现并行生成,FMSD-TTS 属此类

**创新判断**: 相比 KB 中已有的 few-shot/zero-shot TTS 工作(CosyVoice、NaturalSpeech 2 等聚焦多说话人),FMSD-TTS 的创新在于引入 **方言维度的显式建模** — 通过 dialect embedding + DSDR-Net 动态路由实现方言特征的精细控制,这在 KB 现有概念中尚无对应条目。DSDR-Net 的设计思路类似 Mixture-of-Experts 的硬路由变体,但针对固定数量的方言(3 个)做确定性路由而非软门控。

## 速查

> [!summary] 速查
> - **一句话**: 基于 Matcha-TTS 的藏语少样本多说话人多方言 TTS,通过 Speaker-Dialect Fusion + DSDR-Net 动态路由实现方言精细控制,同时保持说话人身份一致性
> - **路线**: 藏文字符 → Tokenization → Text Encoder (含 speaker-dialect fusion) → Duration Predictor → Upsampling → Flow Prediction Network (含 speaker-dialect fusion) → Mel Spectrogram → BigVGAN → Waveform
> - **指标**: DCA 80.25% vs Matcha-TTS 65.80% / VITS2 44.44%; DECS 0.80 vs 0.65 / 0.37; SECS 0.56 vs 0.37 / 0.41; nMOS 3.83 vs 3.73 / 3.18 [Table 1]
> - **可借鉴**: (1) DSDR-Net 的 public FFN + private FFN 硬路由设计 — 可迁移到任何需要按离散属性(方言/风格/情感)做精细控制的 TTS/SVS 场景; (2) Reference Loss (cosine similarity 最小化) 解耦 speaker 与 dialect embedding 的简洁方案
> - **局限**: 仅评估 NAR 架构,未探索 AR 模型; speaker encoder 未在藏语上 fine-tune; 方言转换评估仅采用一种 VC 模型; 训练数据非公开(仅合成数据公开)

## 核心问题

FMSD-TTS 要解决的核心问题是:**在低资源语言(藏语)中,如何用有限的参考音频同时控制说话人身份和方言特征来合成高质量的多方言语音?**

具体挑战包括 [§1]:
1. 藏语三大方言(卫藏、安多、康巴)之间存在显著的语音差异(节奏、声调),但缺乏大规模并行语料
2. 现有方法(Xu et al., 2021)需要为每种方言训练独立的 vocoder,且方言信息注入太晚,难以捕捉核心方言特征
3. 在多方言合成中,说话人身份和方言特征容易纠缠 — 改变方言时说话人音色也会改变

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

FMSD-TTS 基于 Matcha-TTS [§3.1] 构建,是一个 semi-end-to-end 的 NAR TTS 系统:

```
输入: 藏文字符 + 参考音频 + 方言 ID
  ↓
Tokenization → 216 字符词表的长格式 tensor
  ↓
Reference Encoder (ECAPA-TDNN, VoxBlink2 预训练) → speaker embedding (d=192)
  ↓
Dialect Embedding Layer → dialect embedding (d=128)
  ↓
Speaker-Dialect Fusion → fused embedding (d=128)
  ↓
Text Encoder (含 DSDR-Net) → conditioned hidden representation
  ↓
Duration Predictor → phoneme durations → Upsampling
  ↓
Flow Prediction Network (含 speaker-dialect fusion) → Mel Spectrogram
  ↓
BigVGAN (16kHz) → Waveform
```

与 Matcha-TTS 的关键区别: 在 Text Encoder 和 Flow Prediction Network 中都注入了 speaker-dialect fusion 信息,且 Transformer FFN 被替换为 DSDR-Net [§3.1] [论文原文]。

### 关键设计选择

#### 1. Speaker-Dialect Fusion Module [§3.2]

**WHY 选择分离再融合而非联合建模?** 作者认为 speaker identity 和 dialect characteristics 是两种不同性质的信息 — speaker embedding 需要从真实音频中提取(因为说话人数量开放),而 dialect embedding 可以用可学习的 lookup table 表示(因为方言数量固定为 3 种) [论文原文]。

**HOW**:
1. **Speaker embedding 提取**: 参考音频随机裁剪到 3 秒 → ECAPA-TDNN → L2 归一化 [Eq.1-2]
   - 随机裁剪是为了提升鲁棒性,避免模型依赖特定片段 [论文原文]
2. **Dialect embedding 提取**: dialect ID (0/1/2 对应 wz/ad/kb) → Embedding Layer → L2 归一化 [Eq.3-4]
3. **融合**: $\hat{h}_{text} = h_{text} + \text{Linear}(h_{spk} \| h_{did})$ [Eq.5]
   - 将 speaker (d=192) 和 dialect (d=128) concatenate 后通过线性层映射到 d=128,再加到 text hidden states

**[agent 解读]**: 选择 addition 而非 FiLM/cross-attention 注入方式,可能是因为模型规模较小(41.5M 参数),简单的 addition 已足够;且 L2 归一化确保两种 embedding 在同一尺度上,避免一方主导。

#### 2. DSDR-Net (Dialect-Specialized Dynamic Routing Network) [§3.3]

**WHY 不用标准 FFN?** 标准 Transformer 的 FFN 用共享参数处理所有方言,无法建模不同方言之间细微的声学差异(节奏、声调);而完全独立的模型又缺乏跨方言的知识共享 [论文原文]。

**HOW**: 在 Transformer 的每个 FFN 位置替换为 public FFN + private FFN 的组合:
```
h_attn = MultiHeadSelfAttention(h_text)
h_out = FFN_public(h_attn) + FFN_private[d](h_attn)  [Eq.6-9]
```

其中:
- `FFN_public`: 所有方言共享,学习跨方言通用的语言特征
- `FFN_private[d]`: 每种方言一个独立 FFN (共 3 个),通过 dialect ID 硬路由选择

**[agent 解读]**: 这本质上是一种 **硬路由 MoE** (Mixture of Experts) 的简化版:
- 与标准 MoE 的区别: 路由不通过 gating network 学习,而是由 dialect ID 直接确定,因此是确定性硬路由
- 优势: 无 load balancing 问题,无 gating 训练不稳定性;每个方言都保证有专属参数
- 局限: 方言数量必须预先固定,无法扩展到新方言

#### 3. Reference Loss [§3.4]

**WHY 需要解耦?** 当 speaker embedding 和 dialect embedding 高度相关时,模型可能将方言信息泄露到 speaker embedding 中,导致改变方言时说话人音色也变化 [论文原文]。

**HOW**: 最小化 speaker embedding 和 dialect embedding 的余弦相似度:
$$loss_{ref} = \frac{h_{spk} \cdot h_{did}}{||h_{spk}|| \cdot ||h_{did}||}$$
[Eq.10]

**[agent 解读]**: 由于两个 embedding 已经 L2 归一化,cosine similarity 等价于 dot product,直接作为 loss 项即可。这是一种正交性约束的简洁实现,迫使 speaker 和 dialect 表征空间正交。

### 训练策略

- **数据**: 210+ 小时藏语语音 (44h 卫藏 + 45h 康巴 + 90h 安多), 1500+ 说话人 [§4.1]
- **训练集**: 每方言 40,000 条,共 120,000 条 [§4.1]
- **Mel 参数**: 80-bin, SR=16kHz, window=1024, hop=256 [§4.1]
- **优化器**: Adam, lr=1e-4, weight decay=1e-2 [§4.1]
- **训练步数**: 500K steps, batch_size=16/GPU [§4.1]
- **Speaker encoder**: ECAPA-TDNN, VoxBlink2 预训练, 冻结 [§4.2, agent 解读: 根据 Fig.2 特征可视化推断 encoder 未随 TTS 联合训练]
- **Vocoder**: BigVGAN, 100K steps 单独训练 [§4.1]
- **硬件**: 2× NVIDIA RTX 4090 [§4.1]

## 实验

| 指标 | FMSD-TTS | Matcha-TTS+multi-voc | VITS2+multi-voc | SC-CNN+multi-voc | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| DCA(%) ↑ | **80.25** | 65.80 | 44.44 | 40.74 | Tibetan multi-dialect | [Table 1] |
| DECS ↑ | **0.80** | 0.65 | 0.37 | 0.31 | Tibetan multi-dialect | [Table 1] |
| SECS ↑ | **0.56** | 0.37 | 0.41 | 0.30 | Tibetan multi-dialect | [Table 1] |
| nMOS ↑ | **3.83** | 3.73 | 3.18 | 2.82 | Tibetan multi-dialect | [Table 1] |
| sMOS ↑ | **3.57** | 3.03 | 3.52 | 2.97 | Tibetan multi-dialect | [Table 1] |
| dMOC ↑ | **76.67** | 73.33 | 69.15 | 65.14 | Tibetan multi-dialect | [Table 1] |
| RTF ↓ | 0.032 | 0.023 | **0.021** | 0.036 | — | [Table 1] |
| Params | **41.5M** | 54.6M | 72.2M | 64.1M | — | [Table 1] |

**消融实验** [Table 2]:

| 配置 | DCA(%) | DECS |
| --- | --- | --- |
| FMSD-TTS (full) | **80.25** | **0.80** |
| w/o DSDR-Net | 60.12 | 0.58 |
| w/o Dialect ID | 74.15 | 0.72 |
| w/o Dialect ID + DSDR-Net | 33.42 | 0.32 |

消融结论 [§5.1]:
- DSDR-Net 贡献最大: 移除后 DCA 降 20.13 个百分点,DECS 降 0.22 [论文原文]
- Dialect ID 也有显著贡献: 移除后 DCA 降 6.10 个百分点 [论文原文]
- 两者协同效应明显: 同时移除后 DCA 仅 33.42%,远低于单独移除任一组件 [论文原文]

**下游应用验证** [§5.5, Table 3]:
- 用 FMSD-TTS 合成 2,700 对并行多方言数据集
- 在 S2SDC (speech-to-speech dialect conversion) 任务上测试
- DurFlex-EVC + BigVGAN 22K: MOS 3.63 vs 16K: MOS 3.23

## 局限性

1. **架构覆盖不足**: 仅在 NAR 模型 (Matcha-TTS) 上验证,未探索 AR 架构 (Tacotron, VITS) 的表现 [§7]
2. **Speaker encoder 域不匹配**: ECAPA-TDNN 在 VoxBlink2 (非藏语) 上预训练,未在藏语上微调,可能导致 speaker consistency 次优 [§7]
3. **方言转换评估不充分**: S2SDC 任务仅用 DurFlex-EVC 一种模型,缺少与其他 VC 模型的对比 [§7]
4. **评估指标依赖预训练模型**: DCA 和 DECS 依赖自训练的 SDR/SDE 模型,这些模型本身的准确性会影响评估可靠性 [agent 解读]
5. **方言数量固定**: DSDR-Net 硬路由设计要求方言数量预先确定,新增方言需重新训练模型 [agent 解读]
6. **MOS 绝对值不高**: nMOS 3.83 / sMOS 3.57 在绝对标准上仍有提升空间(高质量英语 TTS 通常 >4.0),但考虑到低资源场景可理解 [agent 解读]

## 点评

**优势**:
- 问题定位精准: 低资源多方言 TTS 确实是一个被严重忽视的领域,藏语三方言的并行数据稀缺问题真实存在
- DSDR-Net 设计简洁有效: public + private FFN 的硬路由方案兼顾了跨方言共享和方言特异性建模,消融实验充分验证了其有效性
- 系统贡献完整: 不仅提出模型,还发布了合成数据集和评估工具包,对社区有实际价值

**不足**:
- baseline 偏弱: SC-CNN 和 VITS2 + multi-vocoder 的设计不太合理 — 强 baseline 应该是在统一模型中注入方言信息(如 VITS2 + dialect embedding),而非简单多 vocoder 方案
- 与现代 LLM-based TTS (VALL-E, CosyVoice) 缺乏对比: 这些系统的 in-context learning 能力可能天然支持方言迁移
- Reference loss 的有效性未通过消融单独验证: Table 2 只消融了 DSDR-Net 和 Dialect ID,未展示 reference loss 的贡献
- 评估的 SECS 绝对值偏低 (0.56): 即使是最好的结果,说话人相似度仍不理想

## 可复用的 idea

1. **Public + Private FFN 硬路由**: 当控制维度是有限离散集(方言/情感类别/风格)时,用确定性路由分配专属 FFN 比 soft MoE 更稳定。可直接迁移到多情感 TTS、多风格 SVS
2. **Cosine similarity 正交约束 (Reference Loss)**: 迫使两种 embedding 空间正交以解耦,比 adversarial training 更简单、更稳定。可用于任何需要解耦 speaker/style/emotion 的场景
3. **随机裁剪参考音频**: 训练时对参考音频做 random crop 到固定长度 (3s) 提升 speaker encoder 鲁棒性。简单但有效的数据增强策略
4. **DCA + DECS 评估框架**: 方言一致性的客观评估方案 — 训练方言分类器 (SDR) 和方言 embedding 模型 (SDE),分别衡量离散分类准确率和连续嵌入相似度。可迁移到口音/方言/语种评估

---

检索命中: [[Speaker Embedding]]✓, [[Conditional Flow Matching]]✓ | 过滤: [[Speaker Adaptation]](pending-review), [[Voice Cloning Taxonomy]](pending-review), [[Duration Predictor]](pending-review), [[Non-autoregressive TTS]](pending-review) | 未命中但可能相关: 无

> [!review] 审阅 (2026-06-03, auto)
> **结论**: pass-with-fixes (0 high / 1 medium / 2 low)
> - (medium) frontmatter.tasks 已修正: 移除 [[Zero-shot Speech Synthesis]],FMSD-TTS 是 few-shot 非 zero-shot
> - (low) 训练策略中 speaker encoder 冻结推断已补充标注
> - (low) 随机裁剪参考音频是常见做法,保留但已知悉
> 详见 `_review/FMSD-TTS-review.yml`
