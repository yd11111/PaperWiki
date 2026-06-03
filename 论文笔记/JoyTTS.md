---
type: paper
tier: deep
title: "JoyTTS: LLM-based Spoken Chatbot With Voice Cloning"
arxiv_id: "2507.02380"
source: "Sources/JoyTTS.pdf"
authors: [Fangru Zhou, Jun Zhao, Guoxin Wang]
year: 2025
venue: "arXiv preprint"
tags: [TTS, spoken-chatbot, voice-cloning, LLM-based, end-to-end, open-source]
concepts: ["[[LLM-based TTS]]", "[[Speaker Embedding]]", "[[Speech Tokenizer]]", "[[Speech Language Model]]", "[[Mel Spectrogram]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]", "[[模型库/CosyVoice|CosyVoice]]"]
tasks: []
datasets: ["[[数据集/SEED-TTS-Eval|SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 6 个已确认实体页: [[LLM-based TTS]], [[Speaker Embedding]], [[Speech Tokenizer]], [[Speech Language Model]], [[模型库/CosyVoice 2|CosyVoice 2]], [[模型库/CosyVoice|CosyVoice]])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: JoyTTS 属于 **modular SpeechLM** 路线 — 在已有文本 LLM (MiniCPM-o/Qwen-7B) 外围挂载 TTS 模块 (CosyVoice2),通过 hidden state 桥接实现端到端语音对话。这条路线与 [[论文笔记/LLaMA-Omni 2|LLaMA-Omni 2]] 思路相似(也复用 CosyVoice 2 作为 TTS 后端),但 JoyTTS 的集成方式更简单直接: 仅通过 MLP 映射 LLM hidden states 到 TTS embedding 空间,不涉及额外的 modality adapter 或 speech encoder。

**已有认知**:
- [[模型库/CosyVoice 2|CosyVoice 2]] 作为独立 TTS 系统在 SEED-TTS-Eval test-zh 上 CER 1.45%、SS 0.748,是当前零样本 TTS 的强 baseline。JoyTTS 以此为 TTS 后端,但集成后性能 (SS 0.73, WER 5.09) 相比独立 CosyVoice2 有明显回退。
- [[LLM-based TTS]] 谱系中,JoyTTS 属于 "Hybrid 架构 (LLM + 独立 TTS)" 子类,但与 CosyVoice 自身的 LLM+CFM 两阶段设计不同,JoyTTS 是在 LLM-Chat 和 LLM-TTS 之间做模块级拼接。
- [[Speech Language Model]] 分类体系中,MiniCPM-o 属于 "instruction-tuning" 阶段 + "IPR (Interactive Period Recognition)" 生成范式的 omni-model,JoyTTS 本质上是改善其 TTS 后端的工程优化。

**创新判断**: JoyTTS 的核心贡献是工程集成 + 开源,而非方法创新。hidden state MLP 桥接是已有技术(LLaMA-Omni 2 等已有类似实践)。论文价值主要在于提供了可复现的训练代码和完整流程。

检索命中: [[LLM-based TTS]]✓, [[Speaker Embedding]]✓, [[Speech Tokenizer]]✓, [[Speech Language Model]]✓, [[模型库/CosyVoice 2|CosyVoice 2]]✓, [[模型库/CosyVoice|CosyVoice]]✓ | 过滤: [[Voice Cloning Taxonomy]](pending-review), [[Speech-LLM Integration Taxonomy]](pending-review), [[Streaming Spoken Dialogue]](pending-review), [[Mel Spectrogram]](pending-review) | 未命中但可能相关: MiniCPM-o(无独立模型页)

> [!summary] 速查
> - **一句话**: 将 MiniCPM-o 的 LLM-Chat 与 CosyVoice2 的 TTS 模块通过 hidden state MLP 桥接,构建带 voice cloning 的端到端语音聊天机器人
> - **路线**: Text → Tokenizer → LLM-Chat (Qwen-7B) → MLP(3584→768) + Text Embedding → LLM-TTS (CosyVoice2) → Generator → Mel/Audio
> - **指标**: SEED-TTS-zh SS 0.73 / WER 5.09 (vs CosyVoice2 独立: SS 0.748 / WER 1.45) [Table 1]; 延迟 1.8s on 4090D [§4]
> - **可借鉴**: LLM hidden states 作为语义桥接送入 TTS 模块,让 TTS 获得对话上下文理解能力; 两阶段训练 (独立→联合) 的稳定性策略
> - **局限**: 集成后 WER 大幅恶化 (5.09 vs 1.45); 无消融实验; 仅 5 页无深度分析; 训练数据全为合成语音; 仅中文评估

## 核心问题

JoyTTS 要解决的问题: 现有端到端语音聊天机器人 (如 Qwen2.5-Omni, LLaMA-Omni 2) 虽然提供了对话能力,但**不支持 voice cloning 且缺乏开源训练代码** [§1.1]。MiniCPM-o 虽然开源,但其原有 TTS 模块 (GPT-SoVITS) 的 voice cloning 能力有限 [§1.1],需要替换为更强的 TTS 后端。

**核心假设**: 用 CosyVoice2 替换 MiniCPM-o 的 GPT-SoVITS 模块,并通过 LLM hidden states 桥接两个模块,可以在保持对话能力的同时获得更好的 voice cloning 质量。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

JoyTTS 由四个模块串联组成 [§1.2, Fig 1]:

1. **Tokenizer Module**: 将文本和音频转换为 token 序列,输入 LLM-Chat
2. **LLM-Chat Module**: MiniCPM-o 的 Qwen-7B LLM,处理输入 token,产出文本输出和 hidden layer features [§1.2]
3. **LLM-TTS Module**: 接收 LLM-Chat 的 hidden states + text embeddings,生成 speech tokens [§1.2]
4. **Generator Module**: 将 speech tokens 转为 mel spectrogram 和音频波形 [§1.2]

**TTS Embedding 生成** [§1.2, Fig 2, Eq. 1]:

LLM-Chat 输出的 hidden states $h_i$ (3584 维) 通过 MLP 映射到 768 维,再与 text embeddings $\text{Emb}(y_i)$ 相加,形成 TTS embed:

$$\text{TTS}_{embed} = \text{Emb}(y_i) + \text{MLP}(h_i)$$

其中 $y_i$ 是 LLM-Chat 输出的文本 token。

**推理时的 voice cloning 流程** [§1.2]: prompt text 和 prompt wave 均先通过 LLM-Chat 处理得到 TTS embed,作为先验知识输入 LLM-TTS。LLM-Chat 的输入文本格式为 "Please repeat the following text. text_prompt"。

### 关键设计选择

**Q: 为什么用 hidden states 桥接而不是简单级联?**
论文认为 LLM-Chat 的 hidden layer features "提供了对文本上下文语义的更深理解",帮助 LLM-TTS "更好地把握对话的上下文和语义" [论文原文, §1.2]。[agent 解读] 这实质上是让 TTS 模块获得 LLM 对对话内容的理解,理论上可以产生更符合语境的韵律和情感表达,但论文未提供消融证据证明这一假设。

**Q: 为什么选择 CosyVoice2 替换 GPT-SoVITS?**
论文指出 MiniCPM-o 原有的 GPT-SoVITS TTS 模块 voice cloning 能力有限,CosyVoice2 "提供先进的语音合成能力,显著提升了 voice cloning 的自然度和准确性" [论文原文, §1.1]。从 Table 1 可见,gpt-sovits 的 SS 仅 0.55,而 CosyVoice2 独立达 0.748 [Table 1]。

### 训练策略

**两阶段训练** [§3]:

**第一阶段 — 独立训练**:
- LLM-Chat 模块优先完全训练,确保其 hidden states 和 text labels 精确对齐 [§3]
- LLM-TTS 模块独立训练文本到语音的转换 [§3]
- [论文原文] "通过分别训练这些模块,允许每个模块专注于优化各自的功能而不互相干扰" [§3]

**第二阶段 — 联合训练**:
- 将 LLM-Chat 和 LLM-TTS 整合做联合训练 [§3]
- 使用统一损失函数: $\text{Loss} = L_{\text{LLM-Chat}} + L_{\text{LLM-TTS}}$ [Eq. 2]
- [论文原文] "联合训练允许微调文本生成和语音合成之间的交互,确保对话不仅在上下文上恰当,而且以自然和吸引人的方式传达" [§3]

### 数据构建

**训练数据** [§2]:
- 400K 多轮文本对话样本,约 2000 小时
- 来源: RedGPT + GeneratedChat0.4M (两个开源数据集)
- 文本对话 → CosyVoice2 合成音频 (全部为合成语音,非真人录音)
- 使用 WenetSpeech4TTS 的 text-audio pairs 作为 prompt 增强 voice cloning 能力

**数据增强** [§2]:
1. 将对话文本切分为不同长度的短片段,帮助模型适应不同文本长度 [§2]
2. 引入特殊标点符号,改善模型解读和生成细腻语音模式的能力 [§2]

## 实验

| 指标 | JoyTTS | CosyVoice2 (独立) | GPT-SoVITS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| SS ↑ | 0.73 | **0.748** | 0.55 | SEED-TTS-zh | [Table 1] |
| WER ↓ | 5.09 | **1.45** | 5.13 | SEED-TTS-zh | [Table 1] |

**延迟**: 1.8 秒 (NVIDIA 4090D, 无工程优化) [§4]

**关键观察** [agent 解读]:
- JoyTTS 的 SS (0.73) 比独立 CosyVoice2 (0.748) 低约 2.4%,voice cloning 能力轻微下降
- JoyTTS 的 WER (5.09) 比独立 CosyVoice2 (1.45) 高出 3.5 倍,内容准确性显著恶化
- JoyTTS 相比 GPT-SoVITS,SS 提升显著 (+0.18),WER 基本持平 (5.09 vs 5.13)
- 论文对 WER 大幅退化的原因未做任何分析或讨论

## 局限性

1. **内容准确性严重退化**: WER 从 CosyVoice2 的 1.45 恶化到 5.09 (3.5x),论文未分析原因 [Table 1]
2. **无消融实验**: 未验证 hidden state 桥接的贡献、MLP 维度选择、联合训练 vs 独立训练的影响
3. **训练数据全为合成语音**: 400K 对话全由 CosyVoice2 合成,缺乏真人录音,可能引入合成瑕疵累积
4. **评估不充分**: 仅在 SEED-TTS-zh 一个中文子集上评估,无英文评估,无 MOS 主观评估,无对话质量评估
5. **无与同类系统对比**: 未与 Qwen2.5-Omni、LLaMA-Omni 2 等端到端对话系统直接对比
6. **论文篇幅过短** (5 页): 缺乏深度技术分析,方法描述和实验都较为粗略

## 点评

JoyTTS 本质上是一个**工程集成项目而非方法创新论文**。其核心贡献在于:
1. 将 MiniCPM-o (LLM) 和 CosyVoice2 (TTS) 拼接成带 voice cloning 的语音聊天机器人
2. 提供了完整的开源训练代码和推理脚本

然而,作为技术论文存在明显不足:
- **核心假设未验证**: hidden state 桥接是否真的比简单级联更好? 无对比实验
- **性能反而退化**: 集成后 WER 暴增 (5.09 vs 1.45),说明 LLM-Chat 和 LLM-TTS 的联合训练可能引入了对齐问题,但论文完全回避了这个讨论
- **与已有 pipeline 的关系不清**: LLaMA-Omni 2 同样复用 CosyVoice 2 作为 TTS 后端,且有更完善的 modality adapter 设计和更全面的评估,JoyTTS 的技术贡献相对增量

从知识库定位看,JoyTTS 为 "modular SpeechLM + voice cloning" 方向提供了一个简单但可复现的开源 baseline。其价值更多体现在工程参考而非学术推进。

## 可复用的 idea

1. **LLM hidden states → MLP → TTS embedding**: 最简单的 LLM-TTS 模块桥接方式,实现成本极低 — 一个 MLP 层 (3584→768) 加 element-wise addition 即可。虽然论文未证明其优于简单级联,但这种低成本集成策略适合快速原型验证
2. **两阶段训练 (独立→联合)**: 先独立训练确保各模块对齐,再联合优化交互,是多模块系统的稳健训练策略
3. **合成对话数据构建流程**: 文本对话数据集 + TTS 合成 + 随机 speaker prompt,可低成本构建大规模多说话人对话训练数据
