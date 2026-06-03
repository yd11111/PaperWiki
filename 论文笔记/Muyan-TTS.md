---
type: paper
tier: deep
title: "Muyan-TTS: A Trainable Text-to-Speech Model Optimized for Podcast Scenarios with a $50K Budget"
arxiv_id: "2504.19146"
source: "Sources/Muyan-TTS.pdf"
authors: [Xin Li, Kaikai Jia, Hao Sun, Jun Dai, Ziyang Jiang]
year: 2025
venue: "arXiv preprint"
tags: [TTS, LLM-based, zero-shot, podcast, open-source, VITS, SoVITS, inference-acceleration]
concepts: ["[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Speaker Embedding]]", "[[Speaker Adaptation]]", "[[Variational Autoencoder for TTS]]", "[[Self-Supervised Speech Representation]]"]
models: ["[[VITS]]", "[[CosyVoice 2]]", "[[HuBERT]]", "[[Whisper]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: Muyan-TTS 属于 [[LLM-based TTS]] 中的 hybrid 路线 (LLM + VITS-based decoder),但与 CosyVoice 系列 (LLM + Flow Matching) 不同,它用 VITS/SoVITS 作为解码器而非 flow matching。在已有 KB 中,这种 LLM + VITS 组合的架构是新的变体 -- GPT-SoVITS 也有此路线,但 Muyan-TTS 用了更大的预训练 LLM (Llama-3.2-3B) 替代 GPT-SoVITS 的原始 AR 模型。

**已有认知**:
- [[LLM-based TTS]] 页记载了 LLM + Flow/Diffusion hybrid 架构 (CosyVoice 系列),但 LLM + VITS 的具体组合尚未覆盖
- [[Speech Tokenizer]] 页记录了 HuBERT semantic token (25Hz) 的提取方式,Muyan-TTS 的 token 化方案正是用 HuBERT embedding + GPT-SoVITS quantizer,属于 semantic token 路线
- [[Zero-shot Speech Synthesis]] 页当前 SOTA 为 CosyVoice 3 (WER 1.45% / CER 0.71%),Muyan-TTS 的 WER 3.44% (LibriSpeech) 有较大差距
- [[CosyVoice 2]] 是论文的主要对比对象之一,其 LibriSpeech WER 2.91% 优于 Muyan-TTS 的 3.44%
- [[VITS]] [待确认] 是 Muyan-TTS decoder 的架构基础,VITS 的 G2P 特性被用于抑制 LLM 幻觉

**创新判断**: 本文的核心价值不在于架构创新(LLM + SoVITS 组合已有先例),而在于: (1) 完整开源训练流程 + 数据处理管线, (2) 面向 podcast 场景的优化, (3) $50K 预算约束下的实践报告。在架构层面属于 GPT-SoVITS 的 LLM 升级版。

> 检索命中: [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[CosyVoice 2]]✓, [[Zero-shot Speech Synthesis]]✓, [[Speaker Embedding]]✓ | 过滤: [[VITS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 Llama-3.2-3B + SoVITS decoder 的开源 LLM-based TTS,面向 podcast 场景,以 $50K 预算完成 100K+ 小时数据训练,提供完整可复现的训练流程
> - **路线**: Text → Llama-3.2-3B (生成 HuBERT-quantized audio tokens) → SoVITS decoder (phoneme + speaker embedding → waveform)
> - **指标**: LibriSpeech WER 3.44% / MOS 4.58 / SIM 0.37; SEED WER 4.09% / MOS 4.32 / SIM 0.41; 推理速度 r=0.33 (所有对比模型中最快) [Table 3, Table 5]
> - **可借鉴**: (1) 数据处理管线: MSS→DeReverb→DeEcho→Denoise 四步清洗 + NISQA MOS>3.8/4.5 筛选; (2) 用 VITS decoder 抑制 LLM 幻觉的思路; (3) SFT 只需几十分钟数据、15 分钟训练即可适应新说话人
> - **局限**: SIM 得分显著低于主流模型 (0.37 vs CosyVoice2 的 0.70); 不支持流式推理 (G2P 依赖完整 phoneme); 多语言能力弱 (训练数据偏英语); 不支持 instruction-following

## 核心问题

本文要解决三个问题:
1. **开源完整性缺失**: 现有 LLM-based TTS (如 CosyVoice 2, Step-Audio) 虽开源模型权重,但不提供训练代码和高效推理框架,限制了可复现性和定制化 [§1]
2. **Podcast 场景无专用模型**: 尚无公开的 TTS 模型专门为 podcast 长音频场景优化 [§1]
3. **GPT-SoVITS 的 LLM 瓶颈**: GPT-SoVITS 提供了训练代码,但其 AR 模型缺乏预训练 LLM 的语义理解能力,影响合成质量 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

Muyan-TTS 是一个两阶段 cascade 系统 [§3.1, Fig 1]:

**Stage 1 - LLM (Llama-3.2-3B)**:
- 输入: text tokens (来自 Llama tokenizer) + reference audio tokens (来自 HuBERT + GPT-SoVITS quantizer)
- 输出: target audio tokens (1024 个离散 token 的序列)
- 建模方式: 将 text 和 audio 构成 parallel corpus,用 next-token prediction 训练

**Stage 2 - SoVITS Decoder**:
- 输入: LLM 生成的 audio tokens + phoneme 序列 + speaker embedding
- 输出: 语音波形
- 架构: 基于 GPT-SoVITS 的 VITS-based 模型

### 关键设计选择

**为什么用 VITS-based decoder 而不是 flow matching?**
[论文原文] VITS 的 G2P (grapheme-to-phoneme) 特性提供了更结构化的语音建模,能有效抑制 LLM 引入的幻觉 (word skip/repeat),提高发音准确性 [§3.1]。[agent 解读] 这是一个鲁棒性优先于音质的设计取舍 -- flow matching (如 CosyVoice 2 使用的) 通常能生成更高保真的语音,但 VITS 的 phoneme-level 建模在对齐稳定性上有天然优势,代价是失去了流式能力 (需要完整 phoneme 序列)。

**为什么用 Llama-3.2-3B 替代 GPT-SoVITS 原始 AR?**
[论文原文] 预训练 LLM 的语义理解能力能增强 TTS 对文本内容的理解,生成更自然、上下文连贯的语音 [§1]。[agent 解读] 3B 参数量是在性能和成本之间的权衡: 比 CosyVoice 2 的 0.5B Qwen 更大以获取更强的语言能力,又不像 Step-Audio (3B 但数据量 1M+ 小时) 那样在数据量上追求极致。

**Speech tokenization 方案**:
- 用 HuBERT 从音频中提取 embedding (25Hz token rate) [§3.1]
- 用 GPT-SoVITS 预训练的 quantizer 将连续 embedding 离散化为 1024 个 audio token [§3.1]
- 将这 1024 个 token 注入 LLM 词表 (`<|audio token 0|>` 到 `<|audio token 1023|>`) [§3.3]
- [agent 解读] 这种方案直接复用了 GPT-SoVITS 的 quantizer-decoder pipeline,避免了从零训练 tokenizer 的巨大成本,是 $50K 预算约束下的务实选择

### 训练策略

**数据处理** [§3.2, Fig 2]:
```
原始 podcast 音频 (150K+ hours)
  → MSS 分离人声
  → DeReverb 去混响
  → DeEcho 去回声
  → Denoise 降噪
  → 切分为句子 (>5s)
  → NeMo speaker diarization (单人过滤)
  → NISQA MOS > 3.8 筛选
  → Whisper-large-v3 转写 (英文) / FunASR (中文)
  → 最终 100K+ hours 平行语料
```
数据处理消耗 60K A10 GPU hours (~$30K),占总预算 60% [Table 1]。

**LLM 预训练** [§3.3]:
- 在 parallel corpus 上 continue pre-training Llama-3.2-3B
- 无监督格式: `Hey, great to have you in Chatpods. <|audio token 520|>...<|audio token end|>`
- 不使用特殊的 turn-of-speech token (模型自然区分两种模态) [§3.3]
- 80x A100 (80GB, NVLink), 15 epochs, lr=1e-4, 约 10 天完成 [§3.3]
- 成本: 19.2K A100 GPU hours (~$19.2K) [Table 1]

**SFT (Post-training)** [§3.4]:
- 用目标说话人的几十分钟~几小时录音进行有监督微调
- 采用 Alpaca instruction-following 格式 + Llama3 template [§3.4]
- 10 epochs, lr=1e-5, 8x A100 (40G, PCIe), 1 小时数据仅需 ~15 分钟训练 [§3.4]

**SoVITS Decoder 训练** [§3.5]:
- 在 GPT-SoVITS 预训练基础上,用高质量子集 (MOS>4.5, 10K hours) 继续训练 [§3.5]
- 8 epochs, 8x A100 (80GB, NVLink), 约 1 周 [§3.5]
- 成本: 1.34K A100 GPU hours (~$1.34K) [Table 1]

## 实验

### 零样本 TTS (Table 3)

| 指标 | Muyan-TTS | CosyVoice2 | Step-Audio | Spark-TTS | FireRedTTS | GPT-SoVITS v3 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WER(%) LibriSpeech ↓ | 3.44 | **2.91** | 5.22 | 27.36 | 9.58 | 6.02 | [Table 3] |
| MOS LibriSpeech ↑ | 4.58 | 4.81 | **4.90** | 3.66 | 5.00 | 4.28 | [Table 3] |
| SIM LibriSpeech ↑ | 0.37 | **0.70** | -- | 0.45 | 0.48 | 0.31 | [Table 3] |
| WER(%) SEED ↓ | 4.09 | 2.98 | **2.73** | 3.04 | 9.58 | 4.74 | [Table 3] |
| MOS SEED ↑ | 4.32 | 4.22 | **4.90** | 4.04 | 4.07 | 3.86 | [Table 3] |
| SIM SEED ↑ | 0.41 | **0.66** | -- | 0.57 | 0.46 | 0.51 | [Table 3] |

注: Step-Audio 不支持显式音色控制,使用默认声音 "Tingting" 合成,SIM 不可比 [§4.2.1]。FireRedTTS 的 LibriSpeech MOS 5.00 疑似异常值 [agent 解读]。

### SFT 对比 (Table 4)

| 指标 | Muyan-TTS (base) | Muyan-TTS-SFT | 出处 |
| --- | --- | --- | --- |
| WER(%) ↓ | **3.44** | 4.48 | [Table 4] |
| MOS ↑ | 4.58 | **4.97** | [Table 4] |
| SIM ↑ | 0.37 | **0.46** | [Table 4] |

SFT 后 MOS 和 SIM 显著提升,但 WER 轻微退化 (3.44% → 4.48%)。作者将此归因于 SFT 模型对文本格式 (如末尾句号) 的敏感性 [§4.2.2]。

### 推理速度 (Table 5)

| 模型 | r (推理时间/合成时长) ↓ | 出处 |
| --- | --- | --- |
| **Muyan-TTS** | **0.33** | [Table 5] |
| GPT-SoVITS v3 | 0.48 | [Table 5] |
| FireRedTTS | 0.61 | [Table 5] |
| Step-Audio | 0.90 | [Table 5] |
| Spark-TTS | 1.31 | [Table 5] |
| CosyVoice 2 | 2.19 | [Table 5] |

Muyan-TTS 推理速度最快,仅需 0.33 秒生成 1 秒语音。[agent 解读] 这主要得益于: (1) VITS decoder 本身是并行合成,比 flow matching 的多步迭代快; (2) vLLM 加速 LLM 推理; (3) sentence splitting + 并行推理 (虽然评测中禁用了)。

### SoVITS 训练消融 (Table 6)

| 训练配置 | WER(%) ↓ | MOS ↑ | SIM ↑ | 出处 |
| --- | --- | --- | --- | --- |
| MOS>3.8, epoch=4 | 6.91 | 4.92 | 0.40 | [Table 6] |
| MOS>3.8, epoch=8 | 6.11 | 4.93 | 0.35 | [Table 6] |
| MOS>4.5, epoch=4 | 5.83 | 4.92 | 0.43 | [Table 6] |
| MOS>4.5, epoch=8 | **4.48** | **4.97** | **0.46** | [Table 6] |

关键发现: 高质量数据 (MOS>4.5) 对性能的影响 > 训练轮数。低质量数据上多训反而可能伤害 SIM (0.40 → 0.35) [§4.2.4]。

## 局限性

1. **不支持流式推理**: VITS decoder 的 G2P 模块需要完整 phoneme 序列,无法 chunk-by-chunk 生成 [§6]
2. **多语言能力薄弱**: 训练数据以英文 podcast 为主,中文只是少量补充 [§6]
3. **不支持 instruction-following**: 训练数据中无 instruction-level 标注 [§6]
4. **SIM 得分低**: base model 的 SIM 在所有模型中垫底 (LibriSpeech 0.37),voice cloning 能力弱,作者承认预训练未针对此优化 [§4.2.1]
5. **SFT 后 WER 退化**: 文本格式敏感性导致泛化性下降 [§4.2.2]

## 点评

**优势**:
- **开源完整度极高**: 不仅开源模型,还开源数据处理管线 + 完整训练代码 + 推理框架,是目前 LLM-based TTS 中可复现性最强的工作之一
- **成本透明**: $50K 的预算明细 (数据处理 $30K + LLM $19.2K + decoder $1.34K) 对工业界有实际参考价值
- **推理速度领先**: r=0.33 显著优于所有对比模型,适合实时应用
- **数据处理管线设计合理**: MSS→DeReverb→DeEcho→Denoise 四步清洗 + 质量筛选的流程可直接复用

**不足**:
- **架构创新有限**: 本质上是将 GPT-SoVITS 的 AR 模型替换为 Llama-3.2-3B,SoVITS decoder 直接沿用,设计空间探索不足
- **SIM 问题严重**: 零样本场景 SIM 仅 0.37-0.41,远低于 CosyVoice 2 (0.66-0.70) 和 Spark-TTS (0.45-0.57),voice cloning 实用性存疑
- **评测不够全面**: (1) MOS 使用 NISQA 自动评估而非人工打分; (2) 未在 SEED-TTS-Eval 标准分割上测试 (自定义了 prompt 方式); (3) FireRedTTS 的 LibriSpeech MOS 5.00 异常高,质疑评测一致性
- **论文写作粗糙**: 6 页篇幅,缺少对关键设计选择的消融 (如 LLM 规模影响、token rate 影响),训练细节不够充分

## 可复用的 idea

1. **数据处理管线**: MSS + DeReverb + DeEcho + Denoise 四步清洗流程,加上 NISQA MOS 阈值筛选 (3.8 vs 4.5 的差异在 SoVITS 实验中有量化验证),可直接应用于其他 TTS 数据准备 [§3.2]
2. **VITS 抑制 LLM 幻觉**: 用 G2P-based decoder 缓解 LLM autoregressive 生成的稳定性问题,是一种简单有效的 robustness 方案 [§3.1]
3. **SFT 快速适应**: Llama3 template + Alpaca 格式的 instruction-following SFT,几十分钟数据 + 15 分钟训练即可适应新说话人,成本极低 [§3.4]
4. **预算分配参考**: 数据处理 60% + LLM 训练 38% + Decoder 训练 2% 的成本结构,提示数据质量投入的重要性 [Table 1]

> [!review] 审阅结论: pass (2026-06-03)
> 3 个 low issue,无 high/medium。因果解释有来源标注,数字 claim 覆盖良好。
> 详见 `_review/Muyan-TTS-review.yml`
