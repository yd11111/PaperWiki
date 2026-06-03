---
type: paper
tier: deep
title: "OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models"
arxiv_id: "2604.00688"
source: "Sources/OmniVoice.pdf"
authors: [Han Zhu, Lingxuan Ye, Wei Kang, Zengwei Yao, Liyong Guo, Fangjun Kuang, Zhifeng Han, Weiji Zhuang, Long Lin, Daniel Povey]
year: 2026
venue: "Preprint (under review)"
tags: [TTS, zero-shot, non-autoregressive, discrete-diffusion, masked-generation, multilingual, multi-codebook, LLM-initialization]
concepts: ["[[Non-autoregressive TTS]]", "[[Masked Generative Modeling]]", "[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Residual Vector Quantization]]", "[[Classifier-Free Guidance]]", "[[Single-codebook vs Multi-codebook]]", "[[Diffusion-based TTS]]"]
models: ["[[模型库/SoundStorm|SoundStorm]]", "[[模型库/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: OmniVoice 定位于 [[Non-autoregressive TTS]] 的离散 token 路线 [待确认],具体继承 [[Masked Generative Modeling]] [待确认] 中 SoundStorm/MaskGCT 开创的 mask-and-predict 范式,但有两个根本性突破:

1. **架构简化**: MaskGCT 需要 text-to-semantic + semantic-to-acoustic 两阶段级联,OmniVoice 直接 text-to-multi-codebook-acoustic 单阶段,绕过了 semantic token 瓶颈。这与 KB 中 [[Speech Tokenizer]] 记录的主流"先 semantic 后 acoustic"分层路线相悖 -- OmniVoice 论证了单阶段离散 NAR 也可以达到 SOTA。

2. **LLM 初始化进入 NAR**: KB 中 [[LLM-based TTS]] 记录的 LLM 初始化仅在 AR TTS (CosyVoice 系列等) 中成功,OmniVoice 首次在 NAR 架构中成功复用 AR LLM 权重 (Qwen3-0.6B),且 bidirectional attention 与 causal 预训练不冲突。

**已有认知**: [[Residual Vector Quantization]] 的多层码本结构 (coarse → fine) 天然支持层级生成,SoundStorm/MaskGCT 都采用 per-layer masking schedule 来对齐 RVQ 层级结构。OmniVoice 的 full-codebook random masking 打破了这一惯例。[[Classifier-Free Guidance]] 在连续空间扩散模型中广泛使用,OmniVoice 将其扩展到离散 token 的 log-softmax 空间。

**创新判断**: 对比 KB 中 [[Zero-shot Speech Synthesis]] 记录的当前 SOTA (CosyVoice 3 WER test-zh 0.71%, IndexTTS2 SS 0.865),OmniVoice 在 Seed-TTS test-zh 上达到 WER 0.84% / SIM-o 0.777,与 SOTA 竞争但未全面超越;其独特价值在于 600+ 语言覆盖 + 全开源训练数据。

> 检索命中: [[LLM-based TTS]]✓, [[Zero-shot Speech Synthesis]]✓, [[Speech Tokenizer]]✓, [[Residual Vector Quantization]]✓ | 过滤: [[Non-autoregressive TTS]](pending-review), [[Masked Generative Modeling]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个单阶段离散 NAR TTS 支持 600+ 语言,通过 full-codebook random masking + LLM 初始化实现 SOTA 智能度和说话人相似度
> - **路线**: Text tokens + Multi-codebook acoustic tokens (prompt+masked target) → Bidirectional Transformer (Qwen3-0.6B init) → C 个 codebook-specific prediction heads → 32-step iterative unmasking → Higgs-audio decoder → Waveform
> - **指标**: LibriSpeech-PC WER 1.30% / SIM-o 0.729; Seed-TTS zh WER 0.84% / SIM-o 0.777; FLEURS-102 avg CER 4.00% (82/102 语言 CER≤5%); CMOS +0.44 vs ground truth; RTF 0.032 (16 steps, batch=1) [Table 1, 4, 2, 8]
> - **可借鉴**: (1) Full-codebook random masking: 打破 per-layer masking 惯例,训练效率 C 倍提升且质量更好; (2) 用 AR LLM 权重初始化 bidirectional NAR 模型,免去从头训练; (3) 581k 小时全开源数据 + 语言级重采样公式 (Eq. 2) 的低资源语言扩展策略
> - **局限**: 仅用开源数据训练,部分语言质量受限; 数字/数学模式处理弱; 离散 NAR 尚无类似 flow distillation 的推理加速方法; 32 步推理仍慢于 flow-based 单步方案

## 核心问题

本文要解决两个交叉的难题:

1. **两阶段级联的瓶颈**: 当前 SOTA 离散 NAR TTS (如 MaskGCT) 需要 text→semantic→acoustic 两阶段,存在 error propagation 和 information bottleneck (低比特率 semantic token 丢失声学细节) [§1]。单阶段替代方案 (如 Gállego et al. 2025) 一直在智能度上落后于两阶段系统 [§1]。

2. **语言覆盖的极端不均衡**: 现有多语言 zero-shot TTS 最多覆盖数十种语言 [§2.2],而全球有数百种低资源语言缺乏 TTS 支持。MMS 虽覆盖 1000+ 语言但不支持 zero-shot voice cloning [§2.2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

OmniVoice 采用单阶段离散 NAR 架构,核心是 discrete masked diffusion objective + bidirectional Transformer [§2.1]:

```
输入:
  Text tokens Y = [instruct tokens | transcript tokens]
  Acoustic token matrix X ∈ R^{T×C} (T 时间步, C=8 codebooks)
     → X_prompt (prefix, unmasked) + X_target (部分 masked)

处理:
  Text → text embedding layer
  Acoustic → C 个 codebook-specific embedding layers → 同时间步求和 → unified embedding
  [Y; X_prompt; X_target] → Bidirectional Transformer (Qwen3-0.6B backbone)

输出:
  C 个 codebook-specific prediction heads → 每个预测对应 codebook 的 token 概率分布
  Loss = -Σ_{(t,c)∈M} log P(x_{t,c} | X, Y; θ)  [Eq. 1]
```

[论文原文] 这种端到端架构"bypassing the complexity and limitations of cascaded pipelines"[§2.1],因为不存在 semantic token 中间表示,所以不会有 information bottleneck 或 error propagation。

[agent 解读] 与 MaskGCT 的根本区别在于: MaskGCT 先用 VQ-VAE 将语音编码为 semantic token 再用 masked generation 生成 acoustic token (两次量化),OmniVoice 直接从 text 到 8-codebook acoustic token (一次量化),减少了信息丢失环节。

### 关键设计选择

#### 1. Full-Codebook Random Masking [§2.1.1]

[论文原文] Per-layer masking (SoundStorm/MaskGCT 采用) 每次只在单个 codebook 层随机 mask 并计算 loss,导致"optimizes only a sparse subset of the token matrix at each iteration, leading to suboptimal training efficiency"[§2.1.1]。

OmniVoice 的方案: 对整个 T×C token 矩阵,**每个 entry 独立** 以 Bernoulli(p_t) 采样 mask,p_t ~ U(0,1) [§2.1.1]:

| 策略 | 每次迭代参与 loss 的 token 比例 | 效果 |
| --- | --- | --- |
| Per-layer (SoundStorm/MaskGCT) | 平均 ~1/C (仅选中的一层) | 收敛慢 |
| Full-codebook random | 平均 ~50% (全矩阵的一半) | C 倍更多 token 参与训练 |

[论文原文] "on average, 50% of the tokens are used for loss computation, C times more than per-layer masking strategy, significantly accelerating convergence and boosting generative quality" [§2.1.1]。

[agent 解读] 为什么 per-layer masking 不是必须的? Per-layer 设计是为了对齐逐层推理 (先生成低层再生成高层),但 OmniVoice 的推理也打破了这个约束 -- 它用 confidence-based 选择 + layer penalty 来鼓励 (但不强制) 低层先 unmask。这说明 RVQ 层间的严格顺序约束可以被放松。

消融验证 [Table 5]:
- SoundStorm-style mask: WER 3.00, SIM-o 0.661
- MaskGCT-style mask: WER 2.04, SIM-o 0.660
- Full-codebook random mask: WER 1.57, SIM-o 0.697
- Full-codebook + single-codebook loss: WER 2.85 (对照,证明 dense loss 而非仅 masking 策略的贡献)

#### 2. LLM 初始化 [§2.1.2]

[论文原文] 单阶段离散 NAR TTS 在智能度上一直不如两阶段或 AR 系统。OmniVoice 用 Qwen3-0.6B 预训练权重初始化 bidirectional Transformer backbone,"OmniVoice is the first NAR TTS model that successfully leverages LLM initialization"[§2.1.2]。

[论文原文] 为什么 causal LLM 的权重能用于 bidirectional 架构? 作者的回答是"we empirically find that their pre-trained knowledge translates well to our bidirectional architecture"[§2.1.2]。

[agent 解读] 这可能因为 Transformer 的 self-attention 权重和 FFN 权重编码的语言知识 (词汇关系、句法结构) 在本质上是双向的 -- causal mask 只是推理时的约束,不影响权重本身的知识表达。CosyVoice 系列在 AR TTS 中也利用了 LLM 初始化,但它们是 causal→causal,而 OmniVoice 是 causal→bidirectional,这是更大胆的跨范式迁移。

消融验证 [Table 6]:
| 初始化方式 | LR | LibriSpeech WER | Seed-en WER | Seed-zh WER |
| --- | --- | --- | --- | --- |
| LLM init | 1e-4 | 1.57 | 1.72 | 0.89 |
| Random init (最优 LR) | 5e-4 | 2.56 | 2.07 | 1.01 |

[Table 6] 即使穷搜学习率,随机初始化的 WER 仍显著高于 LLM 初始化,确认语言先验对智能度的关键贡献。

#### 3. 多语言数据策略 [§2.2]

- **数据来源**: 50 个开源数据集, 581k 小时, 600+ 语言 [§2.2, Fig. 3]
- **语言重采样**: 低资源语言按 r_i = max(1, round((D_max/D_i)^{1-β})) 上采样,β=0.8 [Eq. 2]
- **文本处理**: 直接用 LLM 的 subword tokenizer,不做 G2P 转换 [§2.2]

[论文原文] 这个策略的关键选择是"eliminating cumbersome grapheme-to-phoneme conversion and language-specific text normalization"[§2.2],这对 600+ 语言至关重要,因为大多数低资源语言没有可用的 G2P 工具。

#### 4. 多维可控性 [§2.3]

- **Prompt denoising**: 训练时对 prompt 注入噪声,搭配 `<|denoise|>` 指令 token,使模型从噪声 prompt 生成干净语音 [§2.3.1]
- **Speaker-attribute voice design**: 通过属性 (性别/年龄/口音等) 指令控制音色,无需音频 prompt [§2.3.2]
- **Phonetic override**: 混合 pinyin/phoneme 替换,解决多音字和专业术语问题 [§2.3.3]

### 训练策略

- Backbone: Qwen3-0.6B bidirectional Transformer; 总参数量 0.8B (含 audio tokenizer + prediction heads) [§3.2, Table 1]
- Tokenizer: Higgs-audio (8 codebooks) [§3.2]
- Optimizer: AdamW, peak LR 1e-4, cosine schedule, 3% warmup [§3.3]
- Precision: BF16, sequence packing 8192 tokens/GPU [§3.3]
- Hardware: 8x H800 GPUs [§3.3]
- 多语言版: 2M updates, 9.66 天; Emilia 版: 300k updates, 1.33 天 [§3.3]

### 推理

32 步 iterative unmasking [§3.4]:
1. 使用 time-shifted schedule: r_n = τ·(n/N) / (1 + (τ-1)·(n/N)), τ=0.1 [Eq. 3]
2. Position selection: 对 log-softmax confidence 施加 temperature T=5 后采样 (引入随机性)
3. Token assignment: 选定位置后取 argmax (确定性)
4. Layer penalty: 鼓励先 unmask 低层 token
5. Classifier-free guidance: guidance scale = 2, 在 log-softmax 空间应用 [§3.4]
6. Duration estimation: 基于 script-dependent character weight 的比例缩放 [Eq. 4]

## 实验

### 中英文评估

| 指标 | OmniVoice | OmniVoice-Emilia | MaskGCT | F5-TTS | CosyVoice3 | Qwen3-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIM-o ↑ | **0.729** | 0.697 | 0.691 | 0.655 | 0.694 | 0.704 | LibriSpeech-PC | [Table 1] |
| WER ↓ | **1.30** | 1.57 | 2.26 | 1.89 | 1.59 | 1.60 | LibriSpeech-PC | [Table 1] |
| UTMOS ↑ | 4.28 | 4.23 | 3.91 | 3.89 | 4.28 | **4.41** | LibriSpeech-PC | [Table 1] |
| SIM-o ↑ | **0.741** | 0.717 | 0.713 | 0.664 | 0.696 | 0.708 | Seed-TTS en | [Table 1] |
| WER ↓ | 1.60 | 1.72 | 2.88 | 1.85 | 2.17 | **1.54** | Seed-TTS en | [Table 1] |
| SIM-o ↑ | 0.777 | 0.765 | 0.773 | 0.750 | **0.778** | 0.766 | Seed-TTS zh | [Table 1] |
| WER ↓ | **0.84** | 0.89 | 2.40 | 1.53 | 1.14 | 1.15 | Seed-TTS zh | [Table 1] |
| CMOS ↑ | **+0.44** | +0.42 | -0.38 | - | - | +0.40 | 主观 | [Table 2] |
| SMOS ↑ | **3.80** | 3.58 | 3.20 | - | - | 3.65 | 主观 | [Table 2] |

OmniVoice-Emilia 在相同 Emilia 训练数据下全面超越 NAR baselines (F5-TTS, ZipVoice, MaskGCT);完整多语言版在说话人相似度和智能度上与 CosyVoice 3 / Qwen3-TTS 等 AR 系统竞争 [§4.1]。

### 多语言评估

| 指标 | OmniVoice | MiniMax-Speech | ElevenLabs v2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Avg SIM-o ↑ | **0.830** | 0.766 | 0.655 | MiniMax-24 | [Table 3] |
| Avg WER ↓ | **2.850** | 3.774 | 10.950 | MiniMax-24 | [Table 3] |

| 指标 | OmniVoice | Ground truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Avg CER ↓ | **4.00%** | 5.11% | FLEURS-102 | [Table 4] |
| Languages CER≤5% | **82** | 75 | FLEURS-102 | [Table 4] |
| Languages CER≤10% | **95** | 92 | FLEURS-102 | [Table 4] |

[论文原文] OmniVoice 在 FLEURS-102 上的 avg CER (4.00%) 甚至低于 ground truth (5.11%),作者谨慎指出"we do not claim OmniVoice can generate speech of better quality than the ground truth for all languages. However, OmniVoice's performance has exceeded the measurement capability of existing ASR models"[§4.2]。

[agent 解读] 这个结果的合理解释是: OmniVoice 生成的语音比 FLEURS 原始录音更"干净"(经过 speech restoration 预处理的训练数据),ASR 模型对干净语音识别更准,导致 CER 反而更低。这不等同于合成质量超越真人。

### 消融实验

1. **Masking 策略** [Table 5]: Full-codebook random > MaskGCT-style > SoundStorm-style (WER: 1.57 vs 2.04 vs 3.00)
2. **LLM 初始化** [Table 6]: LLM init WER 1.57 vs random init 最优 WER 2.52 (差距显著)
3. **Prompt denoising** [Table 7]: 开启后 UTMOS 4.23→4.32, SIM-o 0.697→0.668 (更干净但更标准化)

### 推理速度

| Steps | BS=1 RTF | BS=8 RTF | 出处 |
| --- | --- | --- | --- |
| 16 | 0.0319 | 0.0224 | [Table 8] |
| 32 | 0.0598 | 0.0414 | [Table 8] |

[Table 8] 16 步推理 RTF 0.032,优于 ZipVoice 同设定下的 0.056,且 16 步质量仍可接受 (Appendix B)。

## 局限性

1. **仅开源数据**: 数据标注质量和声学质量不均匀,有提升空间 [§E]
2. **数字/数学处理**: 未做 text normalization,复杂数字序列可能出错 [§E]
3. **推理加速**: 连续 NAR (如 flow matching) 可用 flow distillation 减少步数,离散 NAR 目前无类似技术 [§E]
4. **指令跟随能力**: 受限于 instruction-tuning 数据的多样性和质量 [§E]
5. **Prompt denoising 的 trade-off**: 开启后 SIM-o 降低 (0.697→0.668),即去噪能力以牺牲部分说话人相似度为代价 [Table 7]

## 点评

**核心贡献**: OmniVoice 在两个方向上推进了 NAR TTS 的边界:

1. **架构简化的成功验证**: 此前 MaskGCT 的成功建立在"两阶段是必要的"假设上,OmniVoice 证明单阶段离散 NAR 可以达到甚至超越两阶段系统 -- 只要训练效率 (full-codebook masking) 和初始化 (LLM) 到位。这简化了系统设计,减少了维护成本。

2. **LLM→NAR 的知识迁移**: 首次验证 causal LLM 权重可以初始化 bidirectional NAR 架构并带来显著智能度提升,这打开了一个新的研究方向 -- NAR TTS 不必从零学语言知识。

**对 600+ 语言覆盖的态度应审慎**: Fig. 4 和 Table 10 显示,低资源语言 (<10 小时) 虽然有不少 CER<5%,但也有多个语言 CER>10% (如 Urdu 28.73%, Cantonese 21.92%, Lao 25.51%)。"600+ 语言"的宣传值与实际质量保证之间存在差距。

**与 ZipVoice 的关系**: 同一团队 (Xiaomi + Daniel Povey) 的前序工作 ZipVoice 是 flow-matching NAR,OmniVoice 转向 discrete diffusion,表明该团队在探索 NAR TTS 的不同建模范式。

## 可复用的 idea

1. **Full-codebook random masking**: 对任何使用 multi-codebook token 的 masked generation 任务 (音乐、音效、多模态),打破 per-layer masking 惯例可能都能获得类似的训练加速。核心 insight 是 dense loss > sparse loss,即使推理时仍需近似层级顺序。

2. **AR LLM → NAR 初始化**: 对 bidirectional Transformer 架构,直接用 causal LLM 预训练权重初始化是一个低成本的提升手段。前提是 NAR backbone 与 LLM 架构相同 (OmniVoice 的 backbone 就是 Qwen3 架构);对架构不同的 NAR 模型需要额外适配。

3. **语言级重采样公式**: Eq. 2 的 r_i = max(1, round((D_max/D_i)^{1-β})) 是一个简洁实用的低资源语言上采样方案,β 可调 (0=均匀, 1=自然分布, 0.8=温和平衡)。

4. **Script-dependent duration estimation**: 用 per-character 权重根据文字系统 (CJK vs Latin) 调整目标时长,比统一字符计数更合理,且不需要额外 duration predictor。

> [!review] 审阅结论: pass-with-fixes (0 high, 1 medium, 2 low)
> - **medium**: frontmatter models 字段补充了 MaskGCT/Qwen3-TTS 对比链接 (已修正)
> - **low**: 补充了总参数量 0.8B 说明 (已修正)
> - **low**: "AR LLM → NAR 初始化"可复用 idea 补充了架构一致前提 (已修正)
> 详见 `_review/OmniVoice-review.yml`
