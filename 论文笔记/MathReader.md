---
type: paper
tier: deep
title: "MathReader: Text-to-Speech for Mathematical Documents"
arxiv_id: "2501.07088"
source: "Sources/MathReader.pdf"
authors: [Sieun Hyeon, Kyudan Jung, Nam-Joon Kim, Hyun Gon Ryu, Jaeyoung Do]
year: 2025
venue: "ICASSP 2025"
tags: [TTS, pipeline, OCR, accessibility, LaTeX, mathematical-TTS, document-reader]
concepts: ["[[Text-to-Speech Pipeline]]", "[[TTS Evaluation]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 3
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 3 个待确认实体页: [[VITS]], [[Text-to-Speech Pipeline]], [[TTS Evaluation]])
> 自动生成,不保证完整覆盖所有相关知识。全部来源均为 pending-review,仅供参考 [待确认]。
>
> **谱系定位**: MathReader 是一个应用层 TTS pipeline,不涉及 TTS 模型本身的创新。它使用 VITS 作为后端语音合成模型。在 [[Text-to-Speech Pipeline]] 的五阶段演进中,MathReader 并不属于任何一个阶段的推进,而是在"前端文本分析"这一环节做了数学公式场景的适配——通过 OCR + T5 翻译将 LaTeX 公式转为可朗读的英文,再送入标准 TTS。在 [[TTS Evaluation]] 中,WER 被用作可懂度指标,且该页面明确将"数学符号/公式"列为 TTS 评估中不足的维度之一,MathReader 正好针对这个缺口。
>
> **已有认知**: VITS (Kim et al., ICML 2021) 是端到端并行 TTS 模型,MOS 4.43,67x 实时速率,可直接从音素序列生成波形。MathReader 直接使用 VITS 而未做修改。
>
> **创新判断**: MathReader 的创新不在 TTS 模型层面,而在前端预处理——用 fine-tuned T5-small 将 LaTeX 翻译为 spoken English,这是一个工程集成创新而非模型创新。
>
> 检索命中: [[VITS]](pending-review), [[Text-to-Speech Pipeline]](pending-review), [[TTS Evaluation]](pending-review) | 过滤: 无 confirmed 页面 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 用 OCR + fine-tuned T5-small (LaTeX→spoken English 翻译) + VITS 组成 pipeline,解决现有 TTS 文档朗读器无法正确读出数学公式的问题
> - **路线**: PDF → Nougat-small OCR → mmd 文件 → 正则提取 LaTeX 公式 → T5-small 翻译为 spoken English → 替换原文中公式 → VITS TTS → 语音
> - **指标**: WER 0.281 (vs Edge 0.510, Acrobat 0.617); CER 0.148 (vs Edge 0.341, Acrobat 0.454) [Table II]; 单页平均 23.62 秒 [Table IV]
> - **可借鉴**: 将"公式→口语化英文"建模为翻译任务并用小模型 (T5-small) 解决,低成本高效率的工程思路;pipeline 分段设计使每个模块可独立替换
> - **局限**: 仅处理英文数学文档;评估数据集为手工标注的小规模测试集(具体规模未报告);无主观评估(MOS);依赖 Nougat OCR 质量;T5-small 对复杂嵌套公式的翻译准确性未深入分析

## 核心问题

现有 TTS 文档朗读器(Microsoft Edge、Adobe Acrobat 等)在读取学术文档时,对数学公式的处理存在系统性缺陷 [§I]:
1. LaTeX 编译后的公式渲染为特殊视觉形式,传统 TTS 系统将其作为普通文本识别,导致公式被错误朗读(如将求和符号读成字母"p") [Table III]
2. 低分辨率或老旧文档中的公式无法被正确识别时,TTS 系统直接跳过 [Fig 2]

核心挑战: 如何在不修改 TTS 模型本身的前提下,让 TTS 系统正确朗读文档中的数学公式?

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MathReader 是一个 5 阶段级联 pipeline [Fig 1, §III]:

```
PDF → [OCR: Nougat-small] → mmd 文件 → [提取 LaTeX] → LaTeX 公式列表
    → [T5-small 翻译] → spoken English → [替换原文] → 纯文本 → [VITS TTS] → 语音
```

这是一个典型的"分治"策略: 将"读数学文档"这个难题拆解为"识别→翻译→合成"三个已有解的子问题,然后用管道串联 [agent 解读]。

### 关键设计选择

**1. 为什么用 OCR 而非直接解析 PDF 文本?**

论文指出,直接提取 PDF 文本(如 Edge/Acrobat 的做法)无法理解公式的数学含义——PDF 中的公式是渲染后的视觉元素,其文本层只包含字符坐标而非语义结构 [论文原文, §I]。老旧文档甚至只是扫描图像,没有文本层 [论文原文, §I]。因此需要 OCR 将视觉内容转为结构化的 LaTeX 代码。

**2. 为什么用 T5-small 而非 GPT-4 或规则映射?**

- 规则映射(如 Sanmiguel & Martini 2015)无法处理大量异常情况 [论文原文, §II, ref 15]
- GPT-4 (Kortemeyer 2023) 虽然效果好但成本高且推理速度慢,不适合实时服务 [论文原文, §II, ref 19]
- T5-small 作为预训练翻译模型,经 fine-tune 后在 LaTeX→spoken English 这个"翻译任务"上已足够准确,且推理快(单页 4.86 秒)[论文原文, §III.C, Table IV]

**3. 为什么选 Nougat-small 做 OCR?**

Nougat 基于 Swin Transformer,能将 PDF 直接转为 mmd (markup) 格式,输出中自然包含 LaTeX 公式标记 `\[...\]` 和 `\(...\)`,方便后续公式提取 [论文原文, §III.A]。[agent 解读] 这意味着 OCR 和公式定位一步完成,无需额外的公式检测模块。

**4. LaTeX 公式提取**

利用 Nougat 输出的 mmd 文件中的特殊标记 `\[...\]` (display math) 和 `\(...\)` (inline math) 来定位所有公式片段 [§III.B]。这是一个简单的正则匹配步骤。

### 训练策略

T5-small 在公开的 (LaTeX, spoken English) 配对数据集 [ref 27, MathBridge] 上 fine-tune [§IV.B]:
- 硬件: NVIDIA H100
- 训练 20 epochs,选验证 loss 最低的 checkpoint
- 学习率: 1e-4,线性调度
- Batch size: 48
- 输入/输出最大长度: 325 tokens

## 实验

| 指标 | MathReader | Microsoft Edge | Adobe Acrobat | MathReader w/o T5 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER (↓) | **0.281** | 0.510 | 0.617 | 0.663 | [Table II] |
| CER (↓) | **0.148** | 0.341 | 0.454 | 0.491 | [Table II] |

**评估方法**: 用各系统将测试文档转为语音 wav → 用 Naver Clova Note STT 转写 → 与人工标注的 ground truth 计算 WER/CER [§IV.C]。

**消融结果**: 去掉 T5-small 后(直接将 OCR 输出送 TTS),WER 从 0.281 升至 0.663,说明 LaTeX→spoken English 翻译是关键环节 [Table II]。

**时间性能** [Table IV]:

| 阶段 | 平均耗时 (秒/页) |
| --- | --- |
| OCR (Nougat-small) | 12.54 |
| 提取 LaTeX | 0.01 |
| 翻译+替换 (T5-small) | 4.86 |
| TTS (VITS) | 6.21 |
| **总计** | **23.62** |

测试环境为 Intel Xeon Platinum 8480+ + NVIDIA H100。最短输出语音 134 秒,说明生成速度远快于播放速度 [§IV.E]。

**定性对比** [Table III]: 对于公式 $\sum_{n=1}^{5} \frac{1}{n} - \frac{1}{n+1}$,MathReader 输出 "Sum from n equals 1 to 5 of 1 over n minus 1 over n plus 1",而 Edge 输出 "p 5, n equals 1, 1 n minus 1 n plus 1",Acrobat 输出 "5, p, 11 n n+1, n=1"——后两者完全丧失了数学语义。

## 局限性

1. **评估规模和多样性不足**: 测试数据集为手工标注,规模未明确报告;仅评估英文数学文档 [agent 解读]
2. **无主观评估**: 仅用 WER/CER 衡量,缺少 MOS 等人类感知评估;考虑到数学朗读的特殊性(如"cos x"是否优于"cosine of x"),WER 未必完全反映用户体验 [agent 解读]
3. **OCR 瓶颈**: 整个 pipeline 的正确性严重依赖 Nougat OCR 质量;OCR 错误会传递到后续所有阶段 [agent 解读]
4. **T5-small 的翻译能力上限未探索**: 未分析复杂嵌套公式、矩阵、分段函数等的翻译准确率 [agent 解读]
5. **STT 评估的循环依赖**: 用 STT 转写来评估 TTS 输出,STT 自身的错误(尤其是数学术语的同音词,如 y/why, T/Tee)引入额外噪声 [论文原文, §IV.C]
6. **仅英文**: 未讨论其他语言的数学公式朗读 [agent 解读]

## 点评

MathReader 解决了一个实际且重要的问题——数学文档的 TTS 朗读,对视障用户尤其有价值。然而作为一篇 ICASSP 短文,其技术贡献主要在于工程集成而非模型创新:

**优点**:
- 问题定位准确: 识别出现有 TTS 文档朗读器在数学公式上的系统性失败,并给出了清晰的失败原因分析
- 方案设计合理: 将"公式朗读"建模为翻译任务,用小模型实现,兼顾准确性和效率
- 实用性强: 开源代码,单页 23.62 秒可用于实时服务
- 对比充分: 与 Edge/Acrobat 的对比以及消融实验都很有说服力

**不足**:
- 评估不够全面: 缺少主观评估、大规模测试、错误类型分析
- 技术深度有限: 每个组件(Nougat/T5/VITS)均为现成模型,pipeline 设计相对直接
- 未讨论更复杂的场景: 如图表、化学公式、混合语言文档
- 可扩展性分析缺失: 对不同领域(物理/化学/计算机科学)的公式适用性未验证

## 可复用的 idea

1. **"公式→口语化"建模为翻译任务**: 这个 framing 非常优雅——将领域特定的格式化文本(LaTeX/代码/化学式)转为自然语言,是一个通用的"可读性适配"范式,可推广到代码朗读、化学公式朗读等场景
2. **用小模型解决垂直翻译任务**: T5-small (60M 参数)在 LaTeX→spoken English 这个特定任务上足够好,避免了 GPT-4 的成本;这提示在定义良好的窄域翻译任务中,小模型 fine-tune 往往优于大模型 zero-shot
3. **OCR 标记自带公式定位**: 选择 Nougat 是因为它输出的 markup 自然包含公式边界标记,省去了额外的公式检测步骤——选工具时考虑输出格式与下游需求的匹配

> [!review] 自动审阅
> 审阅报告: [[_review/MathReader-review.yml]]
