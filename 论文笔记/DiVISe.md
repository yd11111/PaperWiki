---
type: paper
tier: deep
title: "DiVISe: Direct Visual-Input Speech Synthesis Preserving Speaker Characteristics And Intelligibility"
arxiv_id: "2503.05223"
source: "Sources/DiVISe.pdf"
authors: [Yifan Liu, Yu Fang, Zhouhan Lin]
year: 2025
venue: "arXiv preprint"
tags: [video-to-speech, V2S, audio-visual, speaker-preservation, mel-spectrogram, vocoder, HiFi-GAN, AV-HuBERT, conformer, end-to-end]
concepts: ["[[Mel Spectrogram]]", "[[Neural Vocoder]]", "[[Speaker Embedding]]", "[[Self-Supervised Speech Representation]]", "[[Speaker Verification]]"]
models: ["[[模型库/HuBERT|HuBERT]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 4 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: DiVISe 是 Video-to-Speech (V2S) 合成领域的工作,与本 vault 的 TTS 主线有交叉但方向不同。V2S 从无声视频生成语音,而 TTS 从文本生成语音。两者共享的核心模块包括: (1) [[Neural Vocoder]] (confirmed) — DiVISe 使用 HiFi-GAN 将预测的 mel spectrogram 转为波形,这是 TTS 领域 2020-2023 最广泛使用的 vocoder; (2) [[Mel Spectrogram]] [待确认] — DiVISe 以 128 维 mel spectrogram 为中间表示,与 TTS 系统的 80 维 mel 有差异但原理相同; (3) [[Speaker Embedding]] (confirmed) — DiVISe 的核心贡献之一是无需 speaker embedding 即可保留说话人特性,这与 TTS 领域中 speaker embedding 从 lookup table 到 in-context prompt 的演进趋势相关。
>
> **已有认知**: KB 中 [[Self-Supervised Speech Representation]] [待确认] 记录了 HuBERT 的 masked prediction + offline k-means 范式,DiVISe 使用的 AV-HuBERT 是其音视频扩展版本。[[模型库/HuBERT|HuBERT]] [待确认] 页面记录了 BASE 95M / LARGE 317M / X-LARGE 964M 三个规模,DiVISe 使用 LARGE (325M) 作为视觉骨干。[[Speaker Verification]] [待确认] 页面详细记录了 SECS 和 EER 两个指标的定义和局限性,DiVISe 在评估中同时使用了这两个指标。
>
> **创新判断**: 相较于 KB 中记录的 TTS 系统 (均以文本为输入),DiVISe 的创新在于: (1) 证明 V2S 任务中 mel-based vocoder 显著优于 unit-based vocoder 在说话人特性保留上; (2) 端到端、无 speaker embedding 的 V2S 架构。这些发现虽不直接适用于 text-to-speech,但为 vocoder 选择 (mel vs unit) 提供了跨任务的实证证据。
>
> 检索命中: [[Neural Vocoder]]✓, [[Speaker Embedding]]✓ | 过滤: [[Mel Spectrogram]](pending-review), [[Self-Supervised Speech Representation]](pending-review), [[模型库/HuBERT|HuBERT]](pending-review), [[Speaker Verification]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 端到端 V2S 模型,用 AV-HuBERT + Conformer 直接从无声视频预测 mel spectrogram,不需要任何 speaker embedding 就能保留说话人特性
> - **路线**: 无声视频帧 (25 FPS) → AV-HuBERT Encoder (325M) → Linear Upsampling (4x, 25→100 Hz) → Conformer (4 blocks) → 128-dim Mel Spectrogram → HiFi-GAN Vocoder → 16kHz 波形
> - **指标**: LRS3 SECS 0.624 / EER 28.66% (visual-only SOTA); WER 35.68%; PESQ 1.17; 人类主观 speaker matching 3.90/5.0 [Table 3-6]
> - **可借鉴**: mel-based vocoder 显著优于 unit-based vocoder 在说话人特性保留上的实证 (SECS 0.894 vs 0.555, Table 1),vocoder 输入表示的选择是 speaker similarity 的关键瓶颈
> - **局限**: 需要预处理提取 96x96 嘴部区域 (不适合实时场景); 仅在英语 (LRS2/LRS3) 上测试; 绝对音质指标 (PESQ 1.17-1.23) 仍然较低; 与需要额外信息的方法相比,内容准确度仍有差距

## 核心问题

DiVISe 要解决的核心问题是: **在 Video-to-Speech 合成中,如何在不使用任何额外音频输入 (speaker embedding) 的情况下,同时保留说话人特性和语音可懂度?**

先前的 V2S 方法面临两难:
1. 早期方法 (Lip2Wav, SVTS, Multi-Task) 需要额外的 speaker embedding 作为音频先验,限制了实际应用 [§1]
2. ReVISE 虽然通过 AV-HuBERT 预训练消除了 speaker embedding 依赖,但其 Unit-HiFiGAN vocoder 在将离散 SSL units 转回波形时丢失了说话人特性 [§3]
3. DiffV2S 使用 diffusion model 生成 mel spectrogram,但训练阶段仍需 speaker embedding [§1]

DiVISe 的核心洞察是: **speaker 特性丢失的根因在 vocoder 的输入表示 (acoustic units vs mel spectrogram),而非 V2S 前端模型** [§3, Table 1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DiVISe 由两部分组成 [§4, Fig 2]:
1. **V2S Frontend**: 从无声视频帧生成 mel spectrogram
2. **Neural Vocoder**: 将 mel spectrogram 转为音频波形

V2S Frontend 的三个模块:

| 模块 | 功能 | 参数规模 | 输入/输出 |
|------|------|---------|----------|
| AV-HuBERT Encoder | 视觉特征提取 | 325M (LARGE) | 视频帧 → 嵌入 $e_v \in \mathbb{R}^{T_v \times C}$ |
| Linear Upsampling | 时间对齐 | ~11M | $e_v$ → $e_m \in \mathbb{R}^{kT_v \times C/k}$ (k=4, 25→100 Hz) |
| Conformer Module | 时序建模 + mel 预测 | ~10.29M | $e_m$ → $\tilde{S} \in \mathbb{R}^{kT_v \times F}$ (F=128) |

### 关键设计选择

**1. Mel Spectrogram 而非 Acoustic Units 作为预测目标**

这是 DiVISe 最核心的设计选择。论文通过定量和定性分析论证了为什么选 mel 而非 unit [论文原文]:

- Unit-HiFiGAN 在频域上与原始语音差异很大,而 HiFi-GAN 从 mel 重建的频域紧密贴合原始语音 [Fig 1, §3]
- 定量对比: HiFi-GAN SECS 0.894 vs Unit-HiFiGAN 0.555; EER 22.96% vs 40.52%; PESQ 2.91 vs 1.17 [Table 1]
- **原因**: acoustic units 是离散化后的语义表征,在聚类过程中丢弃了说话人相关的声学细节 (音色、基频微变化等) [agent 解读]; 而 mel spectrogram 保留了完整的频域信息,vocoder 可以从中恢复说话人特性 [论文原文]

**2. AV-HuBERT 预训练的视觉骨干**

使用音视频联合预训练的 AV-HuBERT (而非纯视觉模型) 作为 encoder [论文原文]:
- AV-HuBERT 在预训练阶段同时看到视频和音频,学到了从视觉线索推断语音内容和说话人特性的能力 [§4.1]
- 消融实验表明: 无预训练时 DiVISe 的 SECS 从 0.6242 降至 0.4842, WER 从 35.68% 升至 93.98% [Table 7]
- **为什么预训练对 DiVISe 比对 ReVISE 更有效?** DiVISe 的 mel 预测目标能够利用预训练中编码的说话人信息,而 ReVISE 的 unit 预测目标会丢弃这些信息 [论文原文, §5.4.2]

**3. Conformer 模块的引入**

在 upsampling 后加入 4-block Conformer (256-dim, 4 heads) [§4.1]:
- 目的: 捕获长程时序依赖,平滑 upsampling 后的特征 [论文原文]
- 效果: 主要提升可懂度 (ESTOI 0.324→0.331, WER 38.41→35.68),对说话人特性保留影响很小 (SECS 0.6130→0.6242) [Table 8]
- 参数代价仅 10.29M (~3% 总参数) [Table 15]

**4. Linear Reshaping Upsampling**

使用线性重塑替代 ReVISE 的转置卷积进行上采样 [§4.1]:
- 将 $e_v \in \mathbb{R}^{T_v \times C}$ reshape 为 $e_m \in \mathbb{R}^{kT_v \times C/k}$ (k=4) [agent 解读: 即将每帧的 C 维特征拆成 k 个 C/k 维帧]
- 优势: 计算更高效,去掉了转置卷积的额外计算开销 [§6]

**5. Vocoder 两阶段训练**

HiFi-GAN vocoder 的训练分两步 [§4.2.2]:
1. **预训练**: 在 LJSpeech 上用 GT mel spectrogram 训练 400k updates
2. **微调**: 在 V2S frontend 生成的 mel spectrogram 上微调 (LRS3: 30k, LRS2: 50k updates),弥合 GT mel 与预测 mel 之间的分布差异 [Table 14]
- 微调效果: SECS 0.5505→0.6242 (+0.0737), EER 32.32→28.66 [Table 9]; 对说话人匹配影响显著,对可懂度影响小 [论文原文]

### 训练策略

- **V2S Frontend 训练**: L1 loss 最小化预测 mel 与 GT mel 的差异: $L_1 = \|S - \tilde{S}\|_1$ [Eq. 1]
- **训练资源**: 8 GPU, <45k updates (~26 小时); 低资源设置用 4 GPU, 11.25k updates [Table 13]
- **Tri-stage LR**: (10%, 20%, 70%) 线性升 → 恒定 → 线性降至 5%, peak LR 6e-5 [Table 12]
- **冻结策略**: 开始训练时冻结 AV-HuBERT 参数 5000 updates (full resource) / 1250 updates (low resource) [Table 13]
- **数据限制**: 每个视频截取前 4 秒用于训练; 推理时加载全长视频 [§A.1]

## 实验

### 主要结果: 说话人特性保留

| 指标 | DiVISe | ReVISE | DiffV2S | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| SECS↑ | **0.624** | 0.538 | 0.625 | LRS3 | [Table 4] |
| SECS↑ | **0.609** | 0.521 | 0.581 | LRS2 | [Table 4] |
| EER↓ | **28.66** | 41.17 | — | LRS3 | [Table 3] |
| Speaker Matching MOS | **3.90**±0.14 | 1.80±0.18 | — | LRS3 | [Table 6] |

### 主要结果: 语音可懂度

| 指标 | DiVISe (433h) | ReVISE (433h) | DiffV2S (433h) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER↓ | **35.68** | 36.03 | 39.2 | LRS3 | [Table 2] |
| PESQ↑ | 1.17 | 1.15 | — | LRS3 | [Table 2] |
| ESTOI↑ | **0.331** | 0.290 | 0.284 | LRS3 | [Table 2] |
| STOI↑ | **0.527** | 0.491 | — | LRS3 | [Table 2] |
| WER↓ | **36.24** | 36.95 | 52.7 | LRS2 | [Table 2] |
| PESQ↑ | **1.22** | 1.12 | — | LRS2 | [Table 2] |

### 音视频同步

| 指标 | DiVISe (Vocoder) | ReVISE | DiffV2S | 出处 |
| --- | --- | --- | --- | --- |
| LSE-C↑ | **7.85** | 7.11 | 7.28 | [Table 5] |
| LSE-D↓ | **6.52** | 7.20 | 7.27 | [Table 5] |

### 关键消融结论

| 消融项 | 影响 | 出处 |
|--------|------|------|
| 去掉 AV-HuBERT 预训练 | SECS 0.6242→0.4842; WER 35.68→93.98 (灾难性下降) | [Table 7] |
| 去掉 Conformer | SECS 0.6242→0.6130 (微降); WER 35.68→38.41 (可懂度下降) | [Table 8] |
| 去掉 Vocoder 微调 | SECS 0.6242→0.5505 (显著下降); WER 变化小 | [Table 9] |
| BASE (103M) 替代 LARGE (325M) | 所有指标下降,但 DiVISe 比 ReVISE 受益更大 | [Fig 3] |
| 30h 替代 433h | 性能下降,但 DiVISe 在 30h 已超过 ReVISE 在 433h 的 speaker matching | [Table 2] |

## 局限性

1. **预处理瓶颈**: 需要 dlib 检测 68 个面部关键点 + 仿射变换提取 96x96 嘴部区域,不适合实时应用 [§8]
2. **仅英语评估**: 只在 LRS2 和 LRS3 (英语) 上测试,多语言泛化未知 [§8]
3. **绝对音质仍低**: PESQ 1.17-1.23 距人类语音的 4.0+ 有很大差距; WER 35-37% 表明内容恢复仍有较大误差 [Table 2] [agent 解读]
4. **MOS 悖论**: Unit-HiFiGAN 的 MOS (4.37) 反而略高于 HiFi-GAN (4.24) [Table 16],因为 unit vocoder 生成的中性语音在不考虑说话人身份时更讨人喜欢 [论文原文, §C]; 这暗示 mel-based vocoder 在保留说话人特性时可能引入了一些声学噪声 [agent 解读]
5. **Vocoder 依赖固定**: HiFi-GAN 在 LJSpeech (单说话人) 上预训练,通过微调适应 V2S 输出,但更现代的 vocoder (BigVGAN 等) 可能进一步提升质量 [agent 解读]
6. **模型规模大**: 总参数 350M (backbone 325M + conformer 10M + vocoder 14M),限制了边缘部署 [Table 15]

## 点评

**优点**:
1. **问题分析扎实**: 先用清晰的对比实验 (Table 1, Fig 1) 定位了问题根因 (vocoder 表示选择),再提出方案,因果链条完整
2. **方法简洁有效**: 核心改动仅是将预测目标从 units 换为 mel + 加一个轻量 Conformer,训练流程简单
3. **评估全面**: 同时覆盖说话人相似度 (SECS/EER)、可懂度 (PESQ/STOI/WER)、音视频同步 (LSE-C/D) 和人类主观评价
4. **消融充分**: 逐一验证了预训练、Conformer、vocoder 微调、数据量、模型规模的贡献

**不足**:
1. **数据集有限**: LRS2/LRS3 是相对干净的 TED/BBC 视频,不含真实场景的噪声、遮挡、多说话人等挑战
2. **缺少声学质量分析**: PESQ 1.17 的绝对值很低,但论文未深入分析生成语音的具体质量问题 (如频谱模糊、背景噪声等)
3. **V2S 领域特有意义**: 对 TTS 领域的直接启发有限 (TTS 不从视频输入); 但 mel vs unit 的 vocoder 对比具有跨任务参考价值

## 可复用的 idea

1. **Vocoder 输入表示决定说话人保真度**: mel spectrogram 保留说话人特性远优于离散 SSL units (SECS 差距 ~0.34, Table 1)。这对任何需要保留说话人特性的语音合成系统 (包括 TTS、voice conversion) 都有参考价值 — 如果下游需要 speaker similarity,应避免在 vocoder 输入端使用离散化过重的表示
2. **Vocoder 微调弥合分布差异**: 即使 vocoder 和前端分开训练,在前端生成的 mel 上微调 vocoder (30-50k steps) 可显著提升说话人匹配 (+0.07 SECS, Table 9),成本低但收益大
3. **端到端简单架构的扩展性**: DiVISe 的 "预训练骨干 + upsampling + 轻量解码器" 的模式可以迁移到其他跨模态合成任务,且随数据量和参数量的增长收益比复杂架构更明显 [Fig 3]

---

> [!review] 审阅状态: pass-with-fixes (2026-06-03)
> 3 low issues (template-compliance, traceability-gap, weak-reusability)。无 high/medium issue。
> 详见 `_review/DiVISe-review.yml`

---

检索命中: [[Neural Vocoder]]✓, [[Speaker Embedding]]✓ | 过滤: [[Mel Spectrogram]](pending-review), [[Self-Supervised Speech Representation]](pending-review), [[模型库/HuBERT|HuBERT]](pending-review), [[Speaker Verification]](pending-review) | 未命中但可能相关: 无
