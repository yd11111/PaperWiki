---
type: paper
tier: deep
title: "STITCH: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models"
arxiv_id: ""
source: "ICLR 2026"
authors: [Cheng-Han Chiang, Xiaofei Wang, Linjie Li, Chung-Ching Lin, Kevin Lin, Shujie Liu, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang]
year: 2026
venue: "ICLR 2026"
tags: [speech-LM, reasoning, chain-of-thought, interleaved-generation, latency, spoken-dialogue, math-QA]
concepts: ["[[Speech Language Model]]", "[[Speech Tokenizer]]", "[[Streaming Spoken Dialogue]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 2
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[Speech Language Model]], [[Speech Tokenizer]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[Speech Language Model]]✓, [[Speech Tokenizer]]✓ | 过滤: [[Streaming Spoken Dialogue]](pending-review), [[Speech-LLM Integration Taxonomy]](pending-review), [[Codec Language Model]](pending-review), [[LLM-based TTS]](pending-review) | 未命中但可能相关: 无
>
> **Speech Language Model**: SpeechLM 的演进路线从 GSLM→AudioLM→TWIST→Moshi 形成了从单模态到全双工的连续发展。STITCH 在此演进中定位为 interleaved SLM 的推理增强,填补了 SpeechLM "只能重复不能思考" 的空白。KB 记录了 text-speech interleaved 范式 (GLM-4-Voice) 和 thinker-talker 范式 (Qwen2.5-Omni),STITCH 在前者基础上首次引入 unspoken reasoning。[agent解读]
>
> **Speech Tokenizer**: STITCH 基于 GLM-4-Voice 的 interleaved text-speech token 架构,text token 与 speech token 交替生成 (N_text=13, N_speech=26)。KB 记录了 speech tokenizer 在 SpeechLM 中的角色分布,GLM-4-Voice 使用其专有 speech encoder/decoder。[agent解读]

> [!summary] 速查
> - **一句话**: 首次为 interleaved SLM 引入 unspoken chain-of-thought reasoning,通过交替生成 reasoning chunks 与 text-speech chunks 实现"边想边说",在不增加延迟的前提下将数学推理准确率提升 15%+ [§1]
> - **路线**: speech input → SLM (GLM-4-Voice-9B) → [SOPR] reasoning tokens [EOPR] + text tokens + speech tokens → speech decoder → audio output; reasoning/text/speech 三种 token 交替生成,reasoning 在 audio 播放间隙生成 [§3.2, Fig 1]
> - **指标**: STITCH-R 数学 QA 平均 78.70% (vs no-reasoning 62.98%, TBS 79.12%),延迟仅 N_reason+N_text+N_speech 而非 N_full [Table 1a]; STITCH-S 78.04% 且延迟 = N_text+N_speech (与无推理基线完全相同) [Table 1a]; 非推理任务平均 55.97-57.03%,不降低 [Table 1b]; UTMOSv2 > 3.10, GPT-4o fluency > 4.66 [Table 2]
> - **可借鉴**: (1) 利用 audio 播放的空闲 GPU 时间生成 reasoning tokens,零额外延迟 [§3.2, Fig 1]; (2) 推理 chunk 长度 N_token 可在推理时调节 (60-100),无需重训 [§7]; (3) 可接入外部更强推理模型 (reasoning augmentation model) 替换自身推理 [§8]
> - **局限**: 仅验证数学 QA 场景 [§4.2]; 基于 GLM-4-Voice 单一 SLM,未与其他 SLM 对比 [§4.3]; LoRA 微调失败,需全量微调 [Appendix F]; 仅数学数据微调会导致过拟合 [Appendix F]

## 核心问题

STITCH 要解决 SLM 中 reasoning 与 latency 的根本矛盾 [§1]:

1. **SLM 缺乏内在思考能力**: 当前 interleaved SLM (如 GLM-4-Voice) 的 text token 只是语音内容的文字转录,不包含独立推理过程。模型直接将输入映射到输出,没有 unspoken 的内部思考步骤 [论文原文]
2. **朴素 CoT 引入不可控延迟**: 简单在语音输出前插入完整 text CoT (TBS 方法) 虽能提升质量 (79.12%),但推理长度不可控 (150-360 tokens),造成数秒延迟 [Table 2, §3.1]
3. **人类是边想边说的**: 人类进行复杂推理时,内部思考与言语输出并行进行,而非先完整思考再开口 [§1]

## 方法: STITCH

### 背景: Interleaved SLM 架构 [§2, §3]

STITCH 建立在 interleaved text-speech SLM 范式之上。GLM-4-Voice 为代表:

- 单一 backbone model 交替生成 N_text=13 个 text token 和 N_speech=26 个 speech token [§2]
- Text token 是 speech token 内容的文字预览,确保 speech 生成时语义已确定 [§2]
- Speech token 经 speech decoder 合成为 t_chunk 秒音频 (~2秒) [§3.2]
- 另一范式 thinker-talker (Qwen2.5-Omni) 使用独立 talker model 将文本转为语音,但微调更难 [§3]

### TBS: Think Before Speaking [§3.1]

作为朴素基线,在 interleaved text-speech 序列前插入完整 reasoning span:

```
input x → [SOR] reasoning z [EOR] → t₁ s₁ t₂ s₂ ... [论文原文]
```

- [SOR]/[EOR] 标记推理起止 [§3.1]
- 训练数据 D_TBS: 对每个 (x, y) 对,用 GPT-4o 生成 reasoning z [§4.1]
- **问题**: 推理必须完整生成后才能开始语音输出,延迟 = N_full + N_text + N_speech,N_full 不可控 [§3.1]

### STITCH-R: Reasoning First [§3.2]

核心创新: 将完整 CoT 切分为固定长度的 partial reasoning chunks (N_reason=100 tokens),与 text-speech chunks 交替生成 [§3.2, Fig 2c]:

```
z₁ ∘ t₁ ∘ s₁ ∘ z₂ ∘ t₂ ∘ s₂ ∘ ... [论文原文]
```

**同步思考-说话的物理基础** [§3.2, Fig 1]:
- A100-80G GPU 上 vLLM 生成速度约 80 tokens/sec [§3.2]
- N_speech=26 个 speech token 的音频时长 t_chunk ≈ 2秒 [§3.2]
- 在 t_chunk 时间内可生成 80 x 2 = 160 tokens [§3.2]
- 其中 N_text + N_speech = 13 + 26 = 39 tokens 用于文本和语音 [§3.2]
- **剩余 160 - 39 = 121 tokens 的生成能力用于 reasoning** [§3.2]
- 因此设定 N_reason = 100 (留有余量) [§3.2]

**特殊 token**: [SOPR] (start of partial reasoning) 和 [EOPR] (end of partial reasoning) 包裹每个 reasoning chunk; 最后一个 chunk 额外有 [EOR] 表示推理结束 [§3.2]

**训练数据构造**: 将完整 CoT z 按 N_reason 切分为 {z₁, z₂, ...},与 interleaved text-speech 序列交织; 若 reasoning chunk 数多于 text-speech chunk 数 (推理比回答慢),则丢弃该样本 [§3.2]

**延迟**: 首包延迟 = N_reason + N_text + N_speech = 100 + 13 + 26 = 139 tokens [Table 1a],远小于 TBS 的 N_full + N_text + N_speech [论文原文]

### STITCH-S: Speaking First [§3.3]

进一步消除首包 reasoning 延迟 — 先说再想 [§3.3, Fig 2d]:

```
t₁ ∘ s₁ ∘ z₁ ∘ t₂ ∘ s₂ ∘ z₂ ∘ ... [论文原文]
```

- 首包延迟 = N_text + N_speech,与无推理基线完全相同 [§3.3]
- 第一个 text-speech chunk 不依赖推理,通常是对问题的复述 [§6]
- 训练数据构造: 在每个 reasoning chunk 前插入 text-speech chunk [§3.3]
- 同样丢弃 reasoning chunk 数多于 text chunk 数的样本 [§3.3]

## 实验

### 训练设置 [§4.1]

- 基于 GLM-4-Voice-9B 全量微调 (冻结 speech encoder 和 decoder) [§4.1]
- 训练数据 ~400K instances: VoiceAssistant400K (177K, 对话) + Tulu-3 数学 (220K) + Natural Question + TriviaQA (70K, 知识问答) [Table 3]
- 数学数据: 用 GPT-4o-mini-TTS 合成问题音频,GPT-4o 生成 CoT,GPT-4o 重写答案为口语风格 [§4.1]
- 使用 LlamaFactory 微调,32x A100-80G GPUs,17 小时 [Appendix C]

### 主要结果 [§5, Table 1]

**数学推理 (Table 1a)**:

| 模型 | 延迟 | AddSub | MultiArith | SingleEq | SVAMP | GSM8K | 平均 |
|------|------|--------|------------|----------|-------|-------|------|
| GLM-4-Voice | N_t+N_s | 59.42 | 62.00 | 71.00 | 44.00 | 29.00 | 53.08 |
| No reasoning | N_t+N_s | 66.06 | 70.69 | 77.98 | 64.43 | 35.73 | 62.98 |
| TBS | N_full+N_t+N_s | 79.82 | 85.63 | 89.91 | 75.29 | 64.94 | 79.12 |
| **STITCH-R** | N_r+N_t+N_s | 78.90 | 88.51 | **93.58** | **73.83** | 58.70 | **78.70** |
| **STITCH-S** | N_t+N_s | 81.65 | 87.93 | 91.74 | 72.15 | 56.72 | 78.04 |

**关键发现** [§5]:
1. TBS 比无推理基线平均高 26.04%,证明 reasoning 对 SLM 确实有效 [论文原文]
2. STITCH-R 仅比 TBS 低 0.42%,但延迟从不可控 N_full 缩减至固定 N_reason=100 [论文原文]
3. STITCH-S 比 TBS 低 1.08%,但延迟与无推理基线完全相同 [论文原文]
4. 所有推理模型在 GSM8K (最难) 上相比无推理基线提升最大 (TBS 近乎翻倍) [论文原文]
5. McNemar's test 证实 STITCH-R 和 STITCH-S 显著优于无推理基线 (p < 0.05) [§5]

**非推理任务 (Table 1b)**: STITCH-R 55.97%, STITCH-S 57.03%,与基线 (55.19-55.98%) 持平,推理训练不损害非推理能力 [论文原文]

### 语音质量 [§5, Table 2]

- UTMOSv2: STITCH-R 3.10, STITCH-S 3.17,与 GLM-4-Voice 3.10 相当 [Table 2]
- GPT-4o fluency score: STITCH-R 4.74, STITCH-S 4.66,与 TBS 4.78 相当 [Table 2]
- 推理 token 的插入不影响语音感知质量和流畅度 [论文原文]

### 推理 chunk 长度调节 [§7, Fig 3]

训练时固定 N_token=100,推理时可调:
- 通过在生成 N'_token 个 reasoning tokens 后强制插入 [EOPR] 实现 [§7]
- N'_token >= 80 时,准确率恢复至 N_token=100 的 90% [§7]
- N'_token >= 70 时仍优于 no-reasoning 基线 [§7]
- 允许部署者根据硬件能力灵活调整 [agent解读]

### Reasoning Augmentation [§8]

用外部更强模型替换 STITCH-R 自身的推理:
- ASR 转写用户语音 → 外部模型生成 CoT → 切分为 N'_token chunks → 注入 STITCH-R [§8]
- 更强模型 (GLM-4-9B-Chat, Llama-3.1-8B) 改善准确率,更弱模型 (Llama-3.2-1B) 反而降低 [Fig 3c]
- 证明 STITCH-R 真正利用 reasoning 内容影响语音输出,而非只利用格式 [agent解读]

### 失败尝试 [Appendix F]

1. **LoRA 微调**: GSM8K 准确率仅 ~35% (与无推理相当),推理内容格式正确但数学逻辑错误; 原因: 教 GLM-4-Voice 数学推理需要大量参数更新 [Appendix F]
2. **仅数学数据微调**: 模型对所有输入都用数学推理回答,严重过拟合; 需混合多类型数据 [Appendix F]
3. **自定义 attention mask**: 让 reasoning tokens 无法 attend text/speech tokens,理论上更纯净; 实际 TBS 性能从 79.12% 降至 72.47%,反而有害 [Appendix F]

### 人类评估 [§5, Table 6]

- STITCH-S 响应感 > STITCH-R > TBS (7 分制比较评分) [Table 6]
- STITCH-S vs TBS: 均分 1.687 [Table 6]
- STITCH-R vs TBS: 均分 1.164 [Table 6]
- STITCH-S vs no-reasoning: 均分 0.290 (差异不显著),因为延迟相同 [Table 6]

## 核心设计选择分析

### WHY: 为什么选择 interleaved SLM 而非 thinker-talker [§3]

1. Interleaved SLM 只需一个 backbone model,架构更简单 [论文原文]
2. Thinker-talker 需要精心调优 text 和 speech 的对齐,微调更困难 [论文原文]
3. Qwen-2.5-Omni 的 speech tokenizer 未公开,无法微调 talker model [论文原文]

### WHY: 为什么 STITCH-S 不显著低于 STITCH-R [§6]

- 第一个 text chunk 通常只是对问题的复述 (rephrase),不包含实质答案 [§6]
- 因此第一个 text chunk 不依赖推理即可正确生成 [论文原文]
- 定量验证: 仅 5% 的 STITCH-R 样本中 text chunk 包含未在先前 reasoning 中出现的新计算 [§6]

### WHY: reasoning 总是在 text-speech 之前结束 [§5]

- 训练数据构造时丢弃了推理 chunk 数多于 text chunk 数的样本 [§3.2, §3.3]
- 模型学会在语音结束前完成推理 [论文原文]
- 统计: GSM8K 上 STITCH-R 平均 3.22 个 reasoning chunks vs 5.72 个 text chunks [Table 2]

## 与已有工作的差异

| 维度 | GLM-4-Voice | TBS | STITCH-R | STITCH-S |
|------|-------------|-----|----------|----------|
| 推理能力 | 无 | 完整 CoT | 分块 CoT | 分块 CoT |
| 首包延迟 | N_t+N_s | N_full+N_t+N_s | N_r+N_t+N_s | N_t+N_s |
| 推理长度可控 | N/A | 不可控 | 可调 (N'_token) | 可调 |
| 推理可替换 | N/A | 否 | 是 (augmentation) | 否 |

## 论文贡献与意义

1. **首次为 SLM 引入 unspoken reasoning**: 此前 text-only LLM 有 CoT,audio LLM 有初步探索 (AudioReasoner),但 interleaved SLM 从未有过 unspoken thinking [论文原文]
2. **零额外延迟的推理增强**: STITCH-S 的设计精妙地利用了 audio 播放的物理时间,在用户无感知的情况下植入推理过程 [agent解读]
3. **推理-语音系统的可行性验证**: 证明推理被 text-speech 打断后仍可保持连贯性 (仅 0.42% 性能损失) [论文原文]

---

检索命中: [[Speech Language Model]], [[Speech Tokenizer]] | 过滤: [[Streaming Spoken Dialogue]](pending-review), [[Speech-LLM Integration Taxonomy]](pending-review) | 未命中但可能相关: 无
