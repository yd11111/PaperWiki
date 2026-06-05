---
type: paper
tier: deep
title: "UltraVoice: Scaling Fine-Grained Style-Controlled Speech Conversations for Spoken Dialogue Models"
arxiv_id: "2510.22588"
source: "Sources/UltraVoice.pdf"
authors: [Wenming Tu, Guanrou Yang, Ruiqi Yan, Wenxi Chen, Ziyang Ma, Yipeng Kang, Kai Yu, Xie Chen, Zilong Zheng]
year: 2025
venue: "arXiv preprint"
tags: [spoken-dialogue, style-control, dataset, emotion, controllable-TTS, instruction-following, SFT, expressive-speech]
concepts: ["[[EmotionControlinTTS]]", "[[Instruction-GuidedSpeechSynthesis]]", "[[StyleTransferinTTS]]", "[[SpeechLanguageModel]]", "[[SpokenDialogueEvaluation]]", "[[NaturalLanguageDescriptionforTTS]]"]
models: ["[[CosyVoice]]", "[[Whisper]]"]
tasks: ["[[InstructedSpeechGeneration]]"]
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: UltraVoice 定位在 **spoken dialogue 数据集** 与 **controllable TTS 数据集** 的交叉点。KB 中 [[InstructedSpeechGeneration]] (confirmed) 记录了通过自然语言指令控制语音属性的任务定义,当前 SOTA 以 CosyVoice 3 为代表,但侧重 TTS 场景而非对话场景。[[SpeechLanguageModel]] (confirmed) 定义了端到端 spoken dialogue 模型的技术范畴 (SLAM-Omni、VocalNet 等均属此类),其核心问题之一是"模型能说但不知道怎么说"。[[CosyVoice]] (confirmed) 在本文中作为语音合成工具用于数据构建。
>
> **已有认知 vs 本文增量**: [[EmotionControlinTTS]] [待确认] 和 [[Instruction-GuidedSpeechSynthesis]] [待确认] 详细记录了情感控制和指令引导合成的技术演进,但均聚焦于 TTS 系统本身的可控性。[[StyleTransferinTTS]] [待确认] 梳理了风格控制从 GST 到 instruction-guided 的演进线。UltraVoice 的增量在于: **将多维度风格控制从 TTS 领域迁移到 spoken dialogue 领域**,填补了对话场景缺乏风格控制训练数据的空白。
>
> **创新判断**: KB 中 Instructed Speech Generation 任务页已记录 InstructTTSEval benchmark,但其仅评估 TTS 系统; UltraVoice 的新贡献是构建了一个 dialogue-native 的风格控制数据集,使 spoken dialogue 模型 (非 TTS 模型) 首次获得多维度风格控制能力。
>
> 检索命中: [[InstructedSpeechGeneration]]✓, [[CosyVoice]]✓, [[SpeechLanguageModel]]✓ | 过滤: [[EmotionControlinTTS]](pending-review), [[Instruction-GuidedSpeechSynthesis]](pending-review), [[StyleTransferinTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 构建首个大规模 (830h, 100K 样本) 多维度细粒度风格控制语音对话数据集,SFT 后端到端 spoken dialogue 模型在风格控制 (MOS +29-42%, IFR +14-40pp) 和通用对话能力 (URO-Bench +8-11%) 上同时显著提升
> - **路线**: UltraChat 文本 → GPT-4o 风格指令注入 → GPT-4o-audio-preview/Edge-TTS/CosyVoice 语音合成 → Whisper ASR 质控过滤 → SFT spoken dialogue 模型
> - **指标**: SLAM-Omni-0.5B IFR 28.30→68.39% (+40.09pp); VocalNet-8B MOS 2.85→3.68 (+29.12%); URO-Bench VocalNet-7B SFT Avg Basic 81.56 超过 Qwen2.5-Omni-7B 的 70.69 [Table 4, Table 5, Table 10]
> - **可借鉴**: 用 GPT-4o 作为"风格指令扩写器"批量生成多样化 style prompt 的数据增强策略; 用 Whisper ASR CER<20% 作为合成语音质控门槛
> - **局限**: 数据全合成 (GPT-4o 合成文本 + TTS 合成语音),无真人对话数据; 仅单轮对话; 多语言控制在 LLaMA backbone 上失效 (Language IFR +0.00~0.33pp); 固定单一 response 音色

## 核心问题

当前端到端 spoken dialogue 模型 (如 SLAM-Omni, VocalNet, Mini-Omni) 能够进行语音对话,但输出语音风格单一 (neutral/monotonous),**缺乏"怎么说"的能力** [§1]。这一问题的根源在训练数据: 现有 spoken dialogue 数据集 (InstructS2S, VoiceAssistant) 是将文本对话简单 TTS 转写而成,缺乏真实的副语言信息; 而 controllable TTS 数据集 (SpeechCraft, EmoVoice-DB) 虽有风格标注但缺乏对话结构,将其强行塞入对话格式会退化为非交互 TTS 任务 [§1]。

核心研究问题: **如何构建一个足够大规模、足够多样、且含指令的对话数据集,使 spoken dialogue 模型学会多维度细粒度的语音风格控制?** [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

UltraVoice 本身是一个**数据集**而非模型,核心是一条 bottom-up 的四步数据生成 pipeline [§3]:

```
Step 1: Text Corpus Curation
  ↓ (UltraChat → 200K clean QA pairs)
Step 2: Style Injection & Response Generation
  ↓ (GPT-4o 生成 6 维度风格指令 + 风格化文本响应)
Step 3: Stylized Speech Synthesis
  ↓ (GPT-4o-audio-preview / Edge TTS + CosyVoice VC 合成语音)
Step 4: Quality Control & Filtering
  ↓ (Whisper ASR, CER<20%, duration<30s)
最终产出: 100,770 对话样本, 832.92 小时
```

### 关键设计选择

**1. 为什么选 UltraChat 作为文本来源?** [论文原文] 因为它已被 LLaMA-Omni、Mini-Omni、SLAM-Omni 等广泛采用,内容简洁且不依赖外部引用。选取 "Question About the World" 和 "Creation and Generation" 两类,过滤含 URL/引用/长引文的对话 [§3 Step 1]。[agent 解读] 这使 UltraVoice 的文本分布与现有 spoken dialogue 模型的训练数据天然对齐,减少了域漂移。

**2. 为什么用 GPT-4o 做风格指令注入?** [论文原文] 需要生成多样且自然的风格 prompt,GPT-4o 能利用语义相似表达 (如 "respond in a joyful tone" vs "reply with a cheerful voice") 提供足够的指令多样性 [§3 Step 2]。同时,GPT-4o 也负责将数字转为口语形式、将代码类问题改写为自然语言 [§3 Step 2]。

**3. 六个风格控制维度的选择逻辑**: [论文原文] 覆盖 emotion (7 类), speed (3 级), volume (3 级), accent (6 种英语变体), language (中/日/韩), composite (速度+音量+情感组合) [§3, Table 3]。[agent 解读] 前三者 (emotion/speed/volume) 对应韵律学三要素的感知层映射; accent 和 language 覆盖跨方言/语言场景; composite 测试多维度联合控制能力。

**4. 为什么 response 用固定音色?** [论文原文] 确保所有风格化输出的一致性 [§3 Step 3]。instruction 端随机采样 Seed-TTS-Eval 语料中的多样说话人音色,模拟真实用户的多样性 [§3 Step 3]。

**5. 不同维度的 TTS 模型选择** [Table 2]:
- Emotion/Speed/Volume/Language/Composite: GPT-4o-audio-preview (最高表现力)
- Accent: Edge TTS (支持多口音但无自定义音色) + CosyVoice-300M VC (统一音色)
- General QA: CosyVoice-300M (剥除模板化短语后重合成)

[agent 解读] 这是一个实用主义选择: GPT-4o-audio-preview 在大多数维度上表现力最强,但不支持口音控制,因此对口音维度采用 Edge TTS + VC 的两步方案。CosyVoice 在本文中的角色是工具而非评估对象。

**6. 质控门槛的设定** [论文原文] 使用 Whisper-large-v3 做 ASR,保留 CER<20% 且 duration<30s 的样本 [§3 Step 4]。[agent 解读] CER 20% 是一个相对宽松的阈值,这可能是因为情感化/变速/变音量的语音本身 ASR 难度更高,过严门槛会过度筛除有效样本。

### 训练策略

**模型选择**: 在 4 个 spoken dialogue 模型上做 SFT:
- SLAM-Omni-0.5B (Qwen2 backbone, Whisper-small-v3, CosyVoice1 decoder) [Table 14]
- VocalNet-1B (LLaMA3.2, Whisper-large-v3, CosyVoice2) [Table 14]
- VocalNet-7B (Qwen2.5, Whisper-large-v3, CosyVoice2) [Table 14]
- VocalNet-8B (LLaMA3.1, Whisper-large-v3, CosyVoice2) [Table 14]

**SFT 配置**: 
- SLAM-Omni: lr=1e-5, 5 epochs, 4xA100-80G, FP16 [Table 11]
- VocalNet: lr=5e-5, 3 epochs, 4xA100-80G, BF16 [Table 12]
- 上下文长度统一 4096 tokens

**TTS 验证**: 在 EmoVoice-0.5B 的预训练 checkpoint 上 SFT,产出 UltraVoice-0.5B-SFT [Table 13]

## 实验

### 风格控制能力 (内部测试集)

| 指标 | 本文 (SFT) | Baseline (Base) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| IFR Avg (SLAM-Omni-0.5B) | 68.39% | 28.30% (+40.09pp) | UltraVoice test (2300 samples) | [Table 10] |
| IFR Avg (VocalNet-8B) | 59.35% | 44.74% (+14.61pp) | UltraVoice test | [Table 10] |
| MOS Avg (SLAM-Omni-0.5B) | 3.06 | 2.15 (+42.33%) | UltraVoice test | [Table 4] |
| MOS Avg (VocalNet-7B) | 3.59 | 2.73 (+31.50%) | UltraVoice test | [Table 4] |
| MOS Avg (VocalNet-8B) | 3.68 | 2.85 (+29.12%) | UltraVoice test | [Table 4] |
| GPT-4o Ground Truth MOS | 4.60 | — | UltraVoice test | [Table 4] |

### 通用对话能力 (URO-Bench)

| 指标 | 本文 (SFT) | Baseline (Base) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| URO Basic Avg (VocalNet-7B) | 81.56 | 74.66 (+9.24%) | URO-Bench EN | [Table 5] |
| URO Pro Avg (VocalNet-7B) | 52.70 | 47.34 (+11.32%) | URO-Bench EN | [Table 5] |
| URO Basic Avg (VocalNet-8B) | 71.59 | 64.88 (+10.34%) | URO-Bench EN | [Table 5] |
| vs Qwen2.5-Omni-7B Basic | 81.56 | 70.69 | URO-Bench EN | [Table 5] |
| vs GLM4-Voice-9B Basic | 81.56 | 70.61 | URO-Bench EN | [Table 5] |

### TTS 验证

| 指标 | 本文 (UltraVoice-0.5B-SFT) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| WER | 3.97% | EmoVoice-0.5B: 19.82% | UltraVoice test | [Table 6] |
| Emo Sim | 0.95 | EmoVoice-0.5B: 0.94 | UltraVoice test | [Table 6] |
| WER (out-of-domain) | 5.41% | EmoVoice-0.5B: 2.73% | EmoVoice-DB | [Table 6] |
| Emo Sim (out-of-domain) | 0.89 | EmoVoice-0.5B: 0.91 | EmoVoice-DB | [Table 6] |

### 数据集质量指标

| 指标 | 值 | 出处 |
| --- | --- | --- |
| 总样本数 | 100,770 | [Table 3] |
| 总时长 | 832.92 hours | [Table 3] |
| 平均 CER | 5.93% | [Table 3] |
| 平均 UTMOS | 4.00 | [Table 3] |
| 平均对话时长 | 29.35 seconds | [§3.2] |

## 局限性

1. **全合成数据,无真人对话**: 文本由 GPT-4o 生成,语音由 TTS 合成。虽然通过质控保证了基本质量 (UTMOS 4.00),但合成语音缺乏真实对话中的犹豫、重叠、打断等自然现象 [agent 解读]。

2. **仅单轮对话**: 当前版本只有单轮 QA 对。作者明确承认这导致小模型在 Pro Reasoning 上下降 (SLAM-Omni-0.5B 从 24.72→20.07 [Table 5]),并提出未来需加入多轮对话 [§4.3]。

3. **多语言控制在 LLaMA backbone 上几乎无效**: VocalNet-1B 和 8B (LLaMA backbone) 的 Language IFR 几乎不变 (+0.00pp 和 +0.33pp),而 Qwen backbone 模型有显著提升。作者归因于 LLaMA 多语言预训练不足 [§4.2],但也意味着数据集的多语言子集对这类模型无用。

4. **评估依赖 Gemini-2.5-Flash**: MOS 和 IFR 均由 ALM (Gemini) 自动评估,无人类评估。虽然引用了 ALM-human 一致性研究 [§4.1],但 KB 中 InstructTTSEval 已指出 Gemini-as-Judge 存在 self-preference bias。

5. **GPT-4o 作为数据合成上限**: 多数维度的合成依赖 GPT-4o-audio-preview,其表现力上限直接决定了数据集的天花板。Ground Truth MOS 为 4.60 (非完美 5.0) [Table 4]。

6. **固定音色限制多样性**: 所有 response 使用单一固定音色 [§3 Step 3],无法训练模型做 timbre-adaptive 的风格控制。

## 点评

**核心贡献的定位**: UltraVoice 填补了一个真实存在的空白 — spoken dialogue 模型有了架构但缺乏风格控制训练数据。文章选择了一条实用主义路线: 不改模型,只造数据,然后验证"好数据 + SFT = 好效果"。实验设计覆盖了内部评估 (风格控制) + 外部评估 (URO-Bench) + 迁移验证 (TTS),论证链完整。

**最有价值的发现**: SFT on UltraVoice 不仅提升风格控制能力,还**同时提升**了通用对话能力 (URO-Bench +8-11%)。这违反了"风格控制 vs 对话质量 trade-off"的直觉。[agent 解读] 可能的解释是: (1) UltraVoice 的文本质量 (GPT-4o 生成) 本身高于原始训练数据; (2) 风格多样性作为一种数据增强,提升了模型的泛化能力。

**与 KB 已有知识的关系**: KB 中 [[Instruction-GuidedSpeechSynthesis]] 记录的演进线是从 style tagging→reference prompt→NL description→instruction-guided,UltraVoice 将这条线从 TTS 延伸到 spoken dialogue。但值得注意的是,UltraVoice 的"指令"仍然是 content+style 分离的格式 (用户说一句话并附带风格要求),而非真正的对话式指令 (如"从现在开始用悲伤的语气跟我聊天")。

**潜在问题**: 全合成 pipeline 的可复现性高但也意味着上限受限于 GPT-4o。如果 GPT-4o 的情感表达本身有偏差 (如某些情感类别表现力弱),这些偏差会传递到训练数据和下游模型。

## 可复用的 idea

1. **GPT-4o 作为风格指令多样化器**: 用 LLM 将单一情感标签 (如 "happy") 扩展为多种自然语言表达 (如 "respond in a joyful tone" / "reply with a cheerful voice"),成本低且效果好。可迁移到任何需要指令多样性的数据增强场景。

2. **Edge TTS + CosyVoice VC 的口音合成方案**: Edge TTS 支持多口音但无自定义音色,通过后接 VC 统一音色。这种"先控制再统一"的两步方案可用于其他需要解耦控制维度的场景。

3. **Whisper CER 门槛做合成质控**: 用 ASR 的 CER 作为合成语音质量的代理指标,CER<20% 作为阈值。简单实用,适用于任何大规模语音数据构建。

4. **固定音色 response + 随机音色 instruction**: 在对话数据中,instruction 端模拟用户多样性 (多音色),response 端保持一致性 (单音色)。这种不对称设计有助于模型学会区分"理解多样输入"和"生成一致输出"。

5. **通用对话能力作为风格控制 SFT 的"bonus"**: 实验表明高质量风格控制数据的 SFT 不会损害通用能力,甚至可能提升。这为"数据质量 > 数据纯度"的假设提供了支持。

> [!review] 审阅结果: pass-with-fixes (2026-06-04)
> **结论**: pass-with-fixes (2 issues: 1 medium fixed, 1 low deferred)
> - [x] ~~factual-error (medium): 速查局限误写"未开源模型权重",实际论文声明开源 dataset + checkpoints~~ → 已修正
> - [ ] template-compliance (low): models 字段仅列 KB 已有模型页 (CosyVoice/Whisper),SLAM-Omni/VocalNet 待建页后补充
> 详见 `_review/UltraVoice-review.yml`
