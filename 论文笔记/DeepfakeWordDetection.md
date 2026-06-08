---
type: paper
tier: deep
title: "Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper"
arxiv_id: "2602.22658"
source: "Sources/DeepfakeWordDetection.pdf"
authors: [Hoan My Tran, Xin Wang, Wanying Ge, Xuechen Liu, Junichi Yamagishi]
year: 2026
venue: "arXiv"
tags: [deepfake-detection, ASR, Whisper, anti-spoofing, partial-spoof, word-level-detection, vocoder, fine-tuning]
concepts: ["[[Anti-spoofingandDeepfakeDetection]]", "[[NeuralVocoder]]", "[[LLM-enhancedASR]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 5 个实体页: Anti-spoofing and Deepfake Detection, Whisper, Neural Vocoder, VITS, CosyVoice)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 deepfake detection 领域的新分支 -- partial spoof / word-level localization。当前 KB 中 [[Anti-spoofingandDeepfakeDetection]] 主要覆盖"主动防护"路线 (watermarking/unlearning/poisoning),对"被动检测"尤其是 word-level 定位方法覆盖较少。本文提出的方法是将检测能力嵌入已有 ASR 模型 (Whisper),而非训练独立检测器,属于**模型复用**而非**模型新建**的思路。
>
> **已有认知**: Whisper 是 OpenAI 的大规模弱监督 ASR 模型,encoder-decoder Transformer 结构,680k 小时多语言数据训练 [Whisper 页]。其 multitask token format (special tokens 控制任务类型) 为本文的 token 插入方案提供了天然基础。Neural Vocoder 页覆盖了 HiFi-GAN、WaveGlow 等本文用作训练数据生成的声码器。VITS 和 CosyVoice 是本文测试集中使用的 TTS 系统。
>
> **创新判断**: 相比已有的 partial spoof detection (专门训练 ResNet + RNN/attention 等独立模型),本文的核心新意在于**零架构改动**地复用 Whisper -- 仅通过在训练文本中插入 `<TOF>`/`<EOF>` 标记 token,将检测任务融入 next-token prediction,无需修改模型结构、损失函数或训练算法。这是一种极低成本的多任务增强策略。
>
> 检索命中: [[Anti-spoofingandDeepfakeDetection]][待确认], [[Whisper]][待确认], [[NeuralVocoder]], [[VITS]][待确认], [[CosyVoice]] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过在转录文本中插入标记 token (`<TOF>`/`<EOF>`),将 Whisper 微调为同时执行语音转录和合成词检测的双任务模型,无需任何架构或算法改动
> - **路线**: 输入语音波形 -> Whisper encoder (mel spectrogram) -> decoder next-token prediction -> 输出带 `<TOF>`/`<EOF>` 标记的转录文本 -> 标记间的词即为检测到的合成词
> - **指标**: In-domain (Ft.Voc->E.Voc): FAR 7.22%, FRR 0.52%, WER 0.87% [Table 2]; In-domain (Ft.TTS->E.TTS): FAR 1.38%, FRR 1.79%, WER 2.20% [Table 2]; 均与 ResNet 专用检测器持平
> - **可借鉴**: "Token 插入"范式 -- 任何 seq2seq 模型都可以通过在输出序列中插入特殊 token 来低成本增加新任务,无需改架构/改 loss/加分类头; 用 vocoder copy-synthesis 生成训练数据替代昂贵的 TTS 合成数据
> - **局限**: Out-of-domain 性能严重退化 (FAR 高达 78.6%); vocoded 训练数据无法泛化到 TTS 合成词; 跨语言场景下 FRR 接近 90%; 未开源代码

## 核心问题

本文要解决的核心问题是: **如何低成本地检测语音中被替换的合成词 (synthetic words)?**

传统方案需要从头设计和训练专用检测模型 (如 ResNet),涉及完整的数据、模型设计、训练、部署流程。本文探索一种更经济的替代方案: 利用已有的 ASR 模型 (Whisper),通过最小改动使其在转录的同时完成合成词检测。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Whisper 本身是 encoder-decoder Transformer,通过 next-token prediction 将语音转为文本序列。本文的方法**不改变 Whisper 的任何模型结构或训练算法** [§2.1],仅修改训练数据的标注格式:

对于包含合成词的语音,在对应文本 token 序列中,将合成词用 `<TOF>` (Token-Of-Fake start) 和 `<EOF>` (End-Of-Fake) 包围。例如:
- 原始转录: `"I present to you the human genome"`
- 若 "I"、"present"、"the" 是合成词,标注为: `"<TOF> I <EOF> <TOF> present <EOF> to you <TOF> the <EOF> human genome"` [§2.1, Fig 1]

推理时,Whisper 照常做 next-token prediction,输出序列中 `<TOF>` 和 `<EOF>` 之间的词即被标记为合成词。

### 关键设计选择

**1. 为什么不加分类头?**

一种朴素方案是在 Whisper 上加一个 REAL/FAKE 分类头,同时输出 token 序列 y_{1:M} 和标签序列 c_{1:M}。但这需要修改模型架构、设计双任务 loss 的权重超参数 [论文原文, §2.1]。Token 插入方案将检测任务完全融入已有的 next-token prediction,避免了所有这些额外设计。[agent 解读] 这本质上是将分类问题转化为生成问题 -- 模型不需要"判断"每个词是真是假,而是在转录时"生成"标记 token,检测能力隐式编码在 decoder 的生成分布中。

**2. 为什么复用已有 token 而不新建?**

如果为 `<TOF>` 和 `<EOF>` 添加两个新 token,需要扩展 Whisper 的词表并新增 embedding 向量 [论文原文, §2.1]。本文选择复用 Whisper 词表中已有但极少使用的 token: `'!!!!!!'` 作为 `<TOF>`,`'~~~'` 作为 `<EOF>`。[agent 解读] 这是又一个"最小改动"的体现 -- 连 embedding 层都不需要修改,完全在数据层面完成任务定义。

**3. 为什么用 vocoded 数据训练?**

用多种 TTS 系统合成训练数据需要额外的文本/音频输入和多个生成模型。本文受 Wang & Yamagishi (2023) 启发,提出用 vocoder copy-synthesis 生成训练数据 [§2.2]: 给定一段真实语音,用 WhisperX 对齐获得词边界,随机选 1-5 个词,用 vocoder 重新合成这些词的波形段并替换原始段。[论文原文] vocoded 波形被认为保留了说话人身份,同时引入了类似 TTS 合成器产生的 artifacts [§2.2, 引用 [15]]。

[agent 解读] 这种方案的隐含假设是: 所有合成方法引入的 artifacts 在某种程度上是共享的 (比如频谱不连续、相位不自然等)。后续实验表明这个假设在 in-domain 成立但 out-of-domain 不成立。

### 训练策略

- 使用预训练的 Whisper Large v3 [§3.2]
- 全模型微调 (尝试过 LoRA 但无改善) [§3.2]
- 学习率 1e-5,batch size 8 (单 H100 GPU) [§3.2]
- 训练 5 个 epoch,选验证集最优 checkpoint [§3.2]
- 训练数据约 60k 条语音,覆盖 5 种语言 (Ft.Voc) 或仅英语 (Ft.TTS) [Table 1]

## 实验

### 数据设置

| 数据集 | 规模 | 语言 | 域 | 合成器 | 用途 |
| --- | --- | --- | --- | --- | --- |
| Ft.Voc | 60,596 | en,es,fr,it,de | Audiobook | HiFi-GAN, NSF, NSF+GAN, WaveGlow, WORLD, GL | 训练 |
| Ft.TTS | 60,596 | en | Audiobook | JETS, YourTTS, XTTS, SoVITS, CosyVoice, ElevenLab | 训练 |
| E.Voc | 3,000 | en,es,fr,it,de | Audiobook | 同 Ft.Voc | 测试 (in-domain) |
| E.TTS | 3,000 | en | Audiobook | 同 Ft.TTS | 测试 (in-domain) |
| E.AV1M | 3,000 | en | YouTube | YourTTS, VITS | 测试 (out-of-domain) |
| E.PE | 3,000 | en | Studio | VoiceCraft, SSR-speech | 测试 (out-of-domain) |

[Table 1]

### 评估指标

- **WER**: 去除 `<TOF>`/`<EOF>` 后计算 [§3.3]
- **FAR (false acceptance rate)**: 合成词被误判为真实的比例 [§3.3]
- **FRR (false rejection rate)**: 真实词被误判为合成的比例 [§3.3]
- 即使词被识别错误 (ASR error),只要 REAL/FAKE 标签正确就不算检测错误 [§3.3]

### In-domain 结果

| 配置 | WER (%) | FAR (%) Whisper | FAR (%) ResNet | FRR (%) Whisper | FRR (%) ResNet | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| Ft.Voc -> E.Voc | 0.87 | 7.22 | 7.15 | 0.52 | 3.81 | [Table 2] |
| Ft.TTS -> E.TTS | 2.20 | 1.38 | 0.15 | 1.79 | 3.13 | [Table 2] |

**关键发现**: 在 domain-matched 条件下,微调 Whisper 的合成词检测性能与专用 ResNet 检测器持平,同时 WER 显著低于预训练 Whisper (0.87% vs 23.89% on E.Voc) [§3.4.1]。

### Cross-method 结果 (同 domain 不同合成器)

| 配置 | FAR (%) Whisper | FRR (%) Whisper | 出处 |
| --- | --- | --- | --- |
| Ft.TTS -> E.Voc | 18.86 | 78.59 | [Table 2] |
| Ft.Voc -> E.TTS | 76.16 | 9.01 | [Table 2] |

**关键发现**: Ft.Voc 训练的 Whisper 在 E.TTS 上 FAR 高达 76.16%,表明**用 vocoded 数据训练的模型无法检测 TTS 合成词** [§3.4.2]。反之,Ft.TTS 训练的模型在 E.Voc 上 FRR 接近 80%,且跨语言分析表明这主要源于**未见语言** (非英语的 FRR 接近 90%) [Table 3]。

### Out-of-domain 结果

| 配置 | WER (%) | FAR (%) Whisper | FRR (%) Whisper | 出处 |
| --- | --- | --- | --- | --- |
| Ft.Voc -> E.AV1M | 23.17 | 39.98 | 7.70 | [Table 4] |
| Ft.Voc -> E.PE | 5.01 | 78.60 | 9.61 | [Table 4] |
| Ft.TTS -> E.AV1M | 20.47 | 16.04 | 59.97 | [Table 4] |
| Ft.TTS -> E.PE | 5.77 | 8.74 | 87.89 | [Table 4] |

**关键发现**: Out-of-domain 检测严重不稳定 -- FAR 和 FRR 往往此消彼长 [§3.5]。即使合成器在训练集中出现过 (如 YourTTS 在 Ft.TTS 中),换数据域后仍然退化 (E.AV1M 来自 YouTube vs 训练数据来自 Audiobook),说明**数据域差异比合成器差异更重要** [§3.5]。

### 词长度分析

Figure 2 的 duration 分析揭示: FRR 随词长增加而升高 [§3.4.2]。[论文原文] 一个假说是微调后的 Whisper 仅在未发现任何 artifact 时才判定为 REAL,词越长越可能包含类似 artifact 的模式,导致更多误报。

## 局限性

1. **Out-of-domain 泛化差**: 这是本文最大的局限。跨域 (audiobook -> YouTube/studio) 和跨合成器的检测性能严重不稳定,当前方法**不适用于开放场景** [§3.5, §4]
2. **Vocoded 训练数据的假设不成立**: 论文假设 vocoder artifacts 可以近似 TTS artifacts,但实验否定了这一点 -- Ft.Voc 训练的模型在 E.TTS 上 FAR 76.16% [Table 2]
3. **跨语言局限**: Ft.TTS 仅含英语数据,在非英语 vocoded 测试集上 FRR 接近 90% [Table 3]
4. **WER 在 OOD 场景下退化**: 微调后的 Whisper 在 E.AV1M 上 WER (23.17%) 甚至高于预训练 Whisper (14.72%) [Table 4],说明微调可能损害了 Whisper 在 OOD 数据上的原有转录能力
5. **评估仅限词级别**: 只评估了词替换场景,未考虑句子级插入/删除等更复杂的编辑操作 [agent 解读]
6. **未开源**: 论文未提及代码或模型发布计划

## 点评

**方法论亮点**: "Token 插入"范式极其优雅 -- 零架构改动、零新 loss、零超参数调整,将一个全新的检测任务完全融入已有 seq2seq 模型的 next-token prediction 框架。这种思路的通用性值得关注: 任何需要在序列中标注特定区域的任务 (如 named entity recognition、code vulnerability detection) 都可以用类似方式嵌入 seq2seq 模型。

**实验设计优秀**: 论文设计了四种数据匹配/不匹配的组合 (matched domain + matched method / matched domain + mismatched method / mismatched domain),并辅以 duration 分析和跨语言分解,对泛化失败的原因做了较深入的诊断。

**泛化性是根本瓶颈**: In-domain 结果虽好但实用价值有限 -- 现实中的 deepfake 语音几乎总是来自未知合成器和未知域。论文坦诚承认了这一点,但缺乏有效的改进方向。[agent 解读] 一个可能的方向是用更多样化的 vocoder/TTS 组合 + 域增强训练,但这本质上回到了 domain generalization 的老问题。

**与 KB 中已有工作的对比**: KB 中 [[Anti-spoofingandDeepfakeDetection]] 主要覆盖"主动防护"路线 (watermarking, unlearning, poisoning)。本文属于"被动检测"路线,且聚焦于更精细的 word-level localization 而非 utterance-level binary detection。这两条路线是互补的: 主动防护在源头控制,被动检测在终端守卫。

## 可复用的 idea

1. **Token 插入多任务增强**: 在 seq2seq 模型的输出序列中插入特殊标记 token,将新任务零成本嵌入已有模型的 next-token prediction。适用于任何需要在序列中标注区域的下游任务
2. **Vocoder copy-synthesis 数据增强**: 用 vocoder 对真实语音做 copy-synthesis,低成本生成大量"部分合成"训练数据,避免依赖多个 TTS 系统。注意: 本文实验表明 vocoded 数据仅能检测同类 artifacts,跨合成器泛化有限 (Ft.Voc->E.TTS FAR 76.16% [Table 2])
3. **复用已有低频 token**: 当需要新 special token 但不想扩展词表时,复用模型词表中已有但极少使用的 token (如 `'!!!!!!'`),避免修改 embedding 层

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节 WHY 解释充分, 每个设计选择有因果链 |
> | 可信赖 | pass | 数字标注覆盖率 >90%, 指标使用正确 |
> | 可区分 | pass | 因果解释来源标注覆盖率约 85% |
> | 可定位 | pass | KB 背景谱系定位清晰, 创新判断有对比基准 |
> | 不污染 | pass | 跳过反向更新, 无 KB 污染风险 |
> 
> Issues: 3 (high: 0, medium: 2, low: 1)
> 详见 `_review/DeepfakeWordDetection-review.yml`
